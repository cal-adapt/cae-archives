#!/usr/bin/env python
"""
Download HDP weather-station observations to local disk, hourly, for the same
domain and window as the gridded products.

    python fetch_hdp.py sfcWind --dry-run          # selection funnel, no fetch
    python fetch_hdp.py sfcWind                    # ~2 h, resumable
    python fetch_hdp.py sfcWind --min-coverage 0.5

Parallel, four terminals plus a merge:

    python fetch_hdp.py sfcWind --shard 0 4        # each writes its own store
    python fetch_hdp.py sfcWind --shard 1 4
    python fetch_hdp.py sfcWind --shard 2 4
    python fetch_hdp.py sfcWind --shard 3 4
    python fetch_hdp.py sfcWind --merge            # concat -> hdp_hourly.zarr

Output:

    /shared/data/hdp/hdp_hourly.zarr      stations x hourly time
    /shared/data/hdp/stations.csv         full metadata for every selected station
    /shared/data/hdp/selection.json       the filters used, and what they yielded

SLOW BY CONSTRUCTION, BUT PARALLELISABLE. Each HDP asset is ONE station's zarr
store, so retrieval cost scales linearly with station count -- roughly two hours
for 800 stations. It is latency-bound rather than bandwidth-bound, so it
coexists happily with the gridded downloads and two levers speed it up:

  * --batch-size N asks for N stations per query instead of one. Round trips
    fall by a factor of N and this is the cheapest win. The cost is that a batch
    is all-or-nothing: one bad station loses the batch, and stations with
    disjoint records can collapse a shared batch to time=0. Modest values are
    safe; 1 is the conservative default.

  * --shard K N splits the station list K-of-N ways, round robin so the load
    balances despite very uneven network sizes. Each shard writes its OWN store,
    because appending along station_id is not concurrency-safe -- two processes
    appending to one zarr would corrupt it. --merge concatenates them afterwards,
    which is cheap: the whole archive is under a gigabyte.

RESUMABLE. Stations are appended one at a time, and a restart reads the
station_ids already present and fetches only the remainder. A failure costs one
station, not the run.

STORED RAW. No QC is applied. `qc_wind` masks values outside [0, 75] m/s and
drops stations with >1% bad values, but that threshold is a downstream choice and
baking it into a two-hour artefact would freeze it. The outlier report is printed
at the end so the decision is informed.

SELECTION METADATA IS SAVED SEPARATELY. `coverage` -- record-span overlap with
the window -- is what the threshold is applied to, but it does not survive into
the zarr. stations.csv carries it, so a store built at 10% can be subset to 25%
later with a join instead of a refetch.

HEIGHT IS STANDARDISED, SITING IS NOT. HDP QA/QCs every station to a 10 m
anemometer whatever the parent network uses natively (CIMIS 2 m, RAWS 6.1 m),
which matches ERA5 u10/v10 and the AE 10 m variables. Where the mast stands is
not standardised and matters a great deal -- airports sit on flat open ground,
fire-weather masts on ridges. Keep the `network` coordinate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .hdp_stations import (
    load_station_metadata, metadata_summary, select_from_metadata,
    domain_polygon, build_zarr, per_network_coverage, outlier_report,
    catalog_df, EXCLUDE_NETWORKS, ANEMOMETER_HEIGHT_M,
)

DEFAULT_ROOT = "/shared/data"
DEFAULT_META = "s3://wecc-historical-wx/4_merge_wx_v2/all_network_stationlist_merge.csv"

# Match fetch_local.py.
TIME_SLICE = ("1980-01-01", "2014-12-31")

# 10% rather than 25%: a looser store is a strict superset and can be subset
# later from stations.csv, where a tighter one means another two-hour fetch.
MIN_COVERAGE = 0.10
FREQ = "h"
BATCH = 1        # one station per fetch; see the module docstring

# One station per zarr chunk while building. Appends must land on chunk
# boundaries, and a batch does NOT reliably do that: build_zarr's default
# station_chunk=20 with batches of 8 misaligns at the first offset that is not
# a multiple of 20, and a batch that returns fewer stations than requested --
# routine here, since stations with no data in the window are skipped -- would
# misalign any fixed chunk. A chunk of 1 is aligned by construction. The cost is
# many small chunks, which --merge rechunks away to (50 stations, whole time).
BUILD_STATION_CHUNK = 1


def filter_to_catalog(selected: dict) -> dict:
    """Drop station_ids the HDP catalog does not have.

    THE VALIDATOR IS ALL-OR-NOTHING. climakitae rejects an entire query when any
    station_id in it is absent from the catalog -- "Initial validation checks
    failed", and the batch comes back empty. At batch_size=48, two stale IDs
    therefore cost 48 stations, and a network whose every batch contains one
    produces nothing at all. That is exactly how ASOSAWOS came back with zero.

    The station-list CSV carries ~15,900 rows and is not the same population as
    the catalog, so stale IDs are expected rather than exceptional. Filtering
    here is metadata-only and costs one catalog read.
    """
    df = catalog_df()
    known = set(df.station_id.astype(str))
    out, dropped = {}, 0
    for net, ids in selected.items():
        keep = [i for i in ids if str(i) in known]
        gone = len(ids) - len(keep)
        dropped += gone
        if gone:
            print(f"    {net:<12} {gone:>4} of {len(ids):>4} not in catalog")
        if keep:
            out[net] = keep
    print(f"  dropped {dropped} station(s) absent from the HDP catalog")
    return out


def _done_for(ds, variable: str) -> set:
    """Stations that already have data for THIS variable.

    Resume was keyed on station_id alone, which is correct for one variable and
    wrong for a shared store: a second variable's fetch then skips every station
    the first one wrote and collects only the complement, giving two populations
    with ZERO overlap. That is silent, and it makes any joint analysis --
    humidity and wind at the same site, which is what a fire-weather index needs
    -- impossible.
    """
    if variable not in ds:
        return set()
    got = (ds[variable].notnull().sum("time") > 0).compute().values
    return {str(s) for s, g in zip(ds.station_id.values, got) if g}


def shard_of(selected: dict, k: int, n: int) -> dict:
    """Take shard k of n, round robin within each network.

    Round robin rather than contiguous slices because network sizes are wildly
    uneven -- RAWS alone can be half the archive -- so contiguous blocks would
    leave one worker with the tail and the rest idle. Interleaving gives every
    shard a similar mix of networks and therefore a similar runtime.
    """
    out = {net: ids[k::n] for net, ids in selected.items()}
    return {net: ids for net, ids in out.items() if ids}


def repair_store(path: Path) -> tuple[int, int]:
    """Truncate every station_id-dimensioned array to the shortest one.

    An append that raises partway leaves the store ragged: to_zarr extends its
    arrays one at a time, so a failure after `sfcWind` but before `network`
    gives 994 stations of data and 898 of network, and xr.open_zarr then refuses
    the store outright with "conflicting sizes for dimension 'station_id'".

    Truncating to the shortest is the safe direction. The dropped stations are
    exactly the ones whose write did not complete, and resume refetches them
    because it compares the station_ids actually present.
    """
    import zarr

    g = zarr.open_group(str(path), mode="a")
    lens = {}
    for name_, arr in g.arrays():
        dims = list(arr.attrs.get("_ARRAY_DIMENSIONS", []))
        if "station_id" in dims:
            lens[name_] = (arr.shape[dims.index("station_id")], dims)
    if not lens:
        raise RuntimeError(f"{path.name}: no station_id arrays found")

    sizes = {n: v[0] for n, v in lens.items()}
    n_min, n_max = min(sizes.values()), max(sizes.values())
    if n_min == n_max:
        print(f"  {path.name}: consistent at {n_min} stations, nothing to do")
        return n_max, n_min

    print(f"  {path.name}: ragged -- " +
          ", ".join(f"{n}={v}" for n, v in sorted(sizes.items())))
    for name_, (n, dims) in lens.items():
        if n == n_min:
            continue
        arr = g[name_]
        shape = list(arr.shape)
        shape[dims.index("station_id")] = n_min
        try:
            arr.resize(tuple(shape))
        except TypeError:            # zarr 2 takes varargs
            arr.resize(*shape)
    zarr.consolidate_metadata(str(path))
    print(f"  {path.name}: truncated {n_max} -> {n_min} stations")
    return n_max, n_min


def repair_all(out_dir: Path, name: str) -> None:
    stem = name.replace(".zarr", "")
    paths = (sorted(out_dir.glob(f"{stem}.shard*of*.zarr"))
             + sorted(out_dir.glob(f"{stem}.staging*.zarr")))
    if (out_dir / name).exists():
        paths.append(out_dir / name)
    if not paths:
        print(f"  no stores found under {out_dir}")
        return
    for p in paths:
        try:
            repair_store(p)
            ds = xr.open_zarr(p, consolidated=True)
            print(f"    reopens cleanly: {ds.sizes['station_id']} stations x "
                  f"{ds.sizes['time']:,} steps")
        except Exception as e:
            print(f"    {p.name}: {type(e).__name__}: {e}")


def backfill_coords(ds: xr.Dataset, out_dir: Path) -> xr.Dataset:
    """Fill missing lat/lon from stations.csv.

    HDP stores carry lat/lon as data variables and some stations arrive without
    them. A station with no position is silently useless -- sample_grid_at_points
    cannot place it, so it drops out of every gridded comparison with nothing
    raised. The selection metadata has the coordinates, so this is recoverable
    rather than a reason to discard the station.
    """
    csv = out_dir / "stations.csv"
    if not csv.exists():
        return ds
    meta = pd.read_csv(csv)
    meta = meta.set_index(meta.station_id.astype(str))

    filled = {}
    for c in ("lat", "lon", "elevation"):
        if c not in ds.coords or c not in meta.columns:
            continue
        vals = np.asarray(ds[c].values, dtype="float64")
        miss = ~np.isfinite(vals)
        if not miss.any():
            continue
        ids = [str(i) for i in ds.station_id.values]
        got = 0
        for i in np.where(miss)[0]:
            if ids[i] in meta.index:
                v = meta.at[ids[i], c]
                if pd.notnull(v):
                    vals[i] = float(v)
                    got += 1
        print(f"  backfilled {got} of {int(miss.sum())} missing {c} "
              "from stations.csv")
        filled[c] = (ds[c].dims, vals)
    return ds.assign_coords(filled) if filled else ds


def merge_shards(out_dir: Path, name: str, var: str,
                 keep: bool = False, force: bool = False) -> Path:
    """Concatenate shard stores into one archive along station_id.

    Valid because build_zarr reindexes every station onto the same canonical
    time axis derived from the window, so the shards differ only in which
    stations they hold. The axes are compared rather than assumed.
    """
    stem = name.replace(".zarr", "")
    shard_paths = sorted(out_dir.glob(f"{stem}.shard*of*.zarr"))
    staging = sorted(out_dir.glob(f"{stem}.staging*.zarr"))
    shards = shard_paths + staging
    if not shards:
        raise FileNotFoundError(
            f"nothing to merge: no {stem}.shard*of*.zarr or "
            f"{stem}.staging*.zarr under {out_dir}")

    # Every shard must be present. Merging early takes whatever has finished and
    # then DELETES it, so a premature merge silently discards the stations the
    # other shards were still fetching -- and the loss is invisible afterwards,
    # because the result looks like a complete store with fewer stations.
    expected = {int(p.name.split("of")[1].split(".")[0]) for p in shard_paths}
    n_expected = max(expected) if expected else 0
    have_k = sorted(int(p.name.split(".shard")[1].split("of")[0])
                    for p in shard_paths)
    missing_k = [k for k in range(n_expected) if k not in have_k]
    already_merged = (out_dir / name).exists()
    if missing_k and not force:
        if already_merged:
            # The absent shards were probably consumed by an earlier merge, so
            # this is the normal way a premature merge gets completed.
            print(f"  shards {missing_k} absent; assuming they are already in "
                  f"{name}. Check the network counts afterwards.")
        else:
            raise RuntimeError(
                f"found shards {have_k} of {n_expected}, missing {missing_k}.\n"
                "The missing ones are still running or failed. Merging now "
                "would keep only these and delete them. Wait, or pass --force "
                "--keep-shards if you really want a partial merge.")

    # An already-merged store is folded back in, so a merge that ran early can
    # be completed later rather than having lost its stations for good.
    #
    # THE MERGE IS PER VARIABLE, NOT PER STATION. A station is "already present"
    # only if the existing store holds data for THIS variable at it. Testing
    # station_id alone discards an entire humidity fetch when the store already
    # holds wind for the same sites -- the merge reports "dropping 2235
    # already-seen" and keeps the empty column, which reads as success.
    existing = out_dir / name
    prev = None
    prev_have: set = set()
    if existing.exists():
        prev = xr.open_zarr(existing, consolidated=True)
        if var in prev:
            got = (prev[var].notnull().sum("time") > 0).compute().values
            prev_have = {str(s) for s, g in zip(prev.station_id.values, got)
                         if g}
        print(f"  {name}: {prev.sizes['station_id']} stations already merged, "
              f"{len(prev_have)} with '{var}'")

    ref_time = prev.time.values if prev is not None else None
    parts, seen = [], set(prev_have)
    for p in shards:
        ds = xr.open_zarr(p, consolidated=True)
        if ref_time is None:
            ref_time = ds.time.values
        elif not np.array_equal(ds.time.values, ref_time):
            raise RuntimeError(
                f"{p.name} has a different time axis ({ds.sizes['time']:,} vs "
                f"{len(ref_time):,} steps). Shards were built for different "
                "windows; rebuild them with the same --time-slice.")
        # Two kinds of duplicate, and only the first was handled before.
        #
        #   WITHIN the shard -- build_zarr appends batch by batch, so a run that
        #     died partway and was restarted refetches stations it had already
        #     written. The store then holds the same id twice, and align() fails
        #     with "the (pandas) index has duplicate values".
        #   ACROSS shard and existing store -- a station that already carries
        #     this variable.
        #
        # Keep the FIRST occurrence within the shard: later appends of the same
        # station are re-fetches of identical data, so either is equivalent.
        ids = [str(x) for x in ds.station_id.values]
        first, keep_idx = set(), []
        for i, sid in enumerate(ids):
            if sid in first or sid in seen:
                continue
            first.add(sid)
            keep_idx.append(i)
        n_internal = len(ids) - len(set(ids))
        n_seen = len(set(ids) & seen)
        if n_internal:
            print(f"  {p.name}: {n_internal} duplicate id(s) WITHIN the shard "
                  "(interrupted run refetched them); keeping the first of each")
        if n_seen:
            print(f"  {p.name}: dropping {n_seen} station(s) that already "
                  f"have '{var}'")
        if len(keep_idx) != len(ids):
            ds = ds.isel(station_id=keep_idx)
        seen |= {str(s) for s in ds.station_id.values}
        print(f"  {p.name}: {ds.sizes['station_id']} stations")
        parts.append(ds)

    if not parts or sum(p.sizes["station_id"] for p in parts) == 0:
        raise RuntimeError(
            f"every shard station already has '{var}' in {name}; nothing to "
            "merge. Shards left in place.")

    new = (parts[0] if len(parts) == 1
           else xr.concat(parts, dim="station_id", coords="minimal",
                          compat="override"))

    # align() cannot proceed with a repeated label on either side, and the
    # failure message names pandas rather than the store, so check here where
    # the cause is obvious.
    for label, obj in (("shards", new), ("existing store", prev)):
        if obj is None:
            continue
        sid = [str(x) for x in obj.station_id.values]
        if len(sid) != len(set(sid)):
            import collections
            worst = collections.Counter(sid).most_common(3)
            raise RuntimeError(
                f"{label} contain duplicate station ids, e.g. {worst}. "
                "Run `--repair` first, or move the store aside and refetch.")

    if prev is None:
        merged = new
    else:
        # Combine rather than concatenate. The two sides OVERLAP in station_id
        # -- the same site can carry wind in the existing store and humidity in
        # the new one -- so concatenating would duplicate the axis and leave
        # each station with one NaN row per variable. An outer align on
        # station_id unions the sites, and merge fills each variable where it
        # exists.
        prev_a, new_a = xr.align(prev, new, join="outer", exclude=["time"])
        merged = prev_a.combine_first(new_a)
        n_new = len(set(str(s) for s in new.station_id.values) - prev_have)
        print(f"  combined: {merged.sizes['station_id']} stations, "
              f"{n_new} newly carrying '{var}'")

    # Clear encoding BEFORE chunking. Each shard carries its own
    # encoding['chunks'] from build_zarr, and concat keeps it; zarr then tries
    # to honour that stale shape against the new dask blocks and refuses with
    # "would overlap multiple Dask chunks".
    for v in merged.variables:
        merged[v].encoding = {}

    # Whole time series per chunk, a few dozen stations wide. Every consumer
    # reduces over time per station -- resample to monthly, count valid hours --
    # so a time-contiguous chunk is exactly one read.
    merged = merged.chunk({"time": -1, "station_id": 50})
    merged = backfill_coords(merged, out_dir)

    # String coords must be EAGER numpy of dtype object, matching what build_zarr
    # writes. Two things go wrong otherwise: variable-length strings as
    # fixed-width numpy take their width from the longest value present, so
    # shards would carry mismatched dtypes; and a dask-backed object array makes
    # xarray raise "infer_type must be called on a dtype=object array", since it
    # cannot inspect the values to pick a storable dtype without computing them.
    # Chunking first and materialising these afterwards satisfies both.
    for c in ("station_id", "network"):
        if c in merged.coords:
            # Keep the ORIGINAL dims. Assigning a bare array to `network` would
            # make it a new dimension of its own rather than a coordinate along
            # station_id, which quietly changes the store's shape.
            merged = merged.assign_coords(
                {c: (merged[c].dims,
                     np.array([str(v) for v in merged[c].values], dtype=object))})
    for c in ("lat", "lon", "elevation"):
        if c in merged.coords:
            merged = merged.assign_coords({c: merged[c].compute()})

    path = existing
    tmp = out_dir / (name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    print(f"\n  writing {merged.sizes['station_id']} stations x "
          f"{merged.sizes['time']:,} steps")
    merged.to_zarr(tmp, mode="w", consolidated=True, zarr_format=2)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    print(f"  wrote {path}")

    if not keep:
        for p in shards:
            shutil.rmtree(p)
        print(f"  removed {len(shards)} shard store(s)")
    return path


def domain_from_local(root: Path, variable: str | None = None):
    """WRF d03 perimeter, taken from a local product store.

    Point-in-polygon against the TRUE perimeter, not a lat/lon bounding box: the
    domain is rotated, so its box encloses large parts of Nevada and Arizona the
    model never covers -- on an early sample the box over-counted by about a
    third. Reading it from a store fetch_local.py already wrote means no catalog
    access and guarantees the same footprint the gridded comparison uses.
    """
    dirs = ([root / variable] if variable
            else sorted(d for d in root.iterdir()
                        if d.is_dir() and d.name not in
                        ("grids", "hdp", "era5", "conus404")))
    for d in dirs:
        for p in sorted(d.glob("wrf-*.zarr")):     # curvilinear: a real perimeter
            ds = xr.open_zarr(p, consolidated=True)
            poly = domain_polygon(ds)
            print(f"  perimeter from {d.name}/{p.name}: {poly.shape[0]} points")
            return poly
    raise FileNotFoundError(
        f"no wrf-*.zarr under {root}. Run fetch_local.py first, or pass "
        "--bbox to use a plain rectangle instead.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variable", nargs="?", default="sfcWind",
                    help="HDP variable, CF-style (default: sfcWind)")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--name", default="hdp_hourly.zarr")
    ap.add_argument("--meta-csv", default=DEFAULT_META)
    ap.add_argument("--time-slice", nargs=2, default=list(TIME_SLICE),
                    metavar=("START", "END"))
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE,
                    help=f"record-span overlap with the window "
                         f"(default {MIN_COVERAGE})")
    ap.add_argument("--n-per-network", type=int, default=None,
                    help="cap stations per network (default: keep all)")
    ap.add_argument("--networks", nargs="*", default=None,
                    help="restrict to these networks")
    ap.add_argument("--bbox", nargs=4, default=None, type=float,
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                    help="rectangle instead of the d03 perimeter")
    ap.add_argument("--variable-dir", default=None,
                    help="which fetch_local.py folder supplies the perimeter")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=BATCH,
                    help="stations per query. >1 cuts round trips but a bad "
                         "station loses the whole batch (default 1)")
    ap.add_argument("--shard", nargs=2, type=int, default=None,
                    metavar=("K", "N"),
                    help="fetch shard K of N into its own store; merge later")
    ap.add_argument("--merge", action="store_true",
                    help="concatenate shard stores into one archive and exit")
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--refresh-metadata", action="store_true",
                    help="rewrite stations.csv from the full selection and "
                         "exit; fetches nothing")
    ap.add_argument("--no-catalog-check", action="store_true",
                    help="skip filtering the selection against the HDP "
                         "catalog (not recommended: one stale id fails its "
                         "whole batch)")
    ap.add_argument("--force", action="store_true",
                    help="merge even when shards are missing (partial result)")
    ap.add_argument("--repair", action="store_true",
                    help="truncate ragged stores left by a failed append, "
                         "then exit. Resume refetches the dropped stations.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="rebuild from scratch instead of fetching the "
                         "stations not already present")
    a = ap.parse_args()

    root = Path(a.root)
    out_dir = root / "hdp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tslice = (a.time_slice[0], a.time_slice[1])

    if a.repair:
        print(f"repairing stores in {out_dir}")
        repair_all(out_dir, a.name)
        return 0

    if a.merge:
        print(f"merging shards in {out_dir}")
        path = merge_shards(out_dir, a.name, a.variable, a.keep_shards,
                            force=a.force)
        ds = xr.open_zarr(path, consolidated=True)
        print(f"\nstore: {ds.sizes['station_id']} stations x "
              f"{ds.sizes['time']:,} hourly steps")
        per_network_coverage(ds, var=a.variable)
        return 0

    merged_path = out_dir / a.name
    if a.shard:
        k, nshard = a.shard
        if not 0 <= k < nshard:
            ap.error(f"--shard K N needs 0 <= K < N, got {k} {nshard}")
        path = out_dir / a.name.replace(".zarr", f".shard{k}of{nshard}.zarr")
    else:
        # NEVER fetch into the merged product. Its coords are written as one
        # chunk spanning every station, so no append can align to them --
        # "Specified Zarr chunks encoding['chunks']=(N,) for variable named
        # 'lat' would overlap multiple Dask chunks". Staging keeps the merged
        # store immutable and lets --merge fold new stations in.
        k = nshard = None
        path = out_dir / a.name.replace(".zarr", ".staging.zarr")

    print(f"variable   : {a.variable} at {ANEMOMETER_HEIGHT_M} m (HDP standard)")
    print(f"time slice : {tslice[0]} -> {tslice[1]}")
    print(f"destination: {path}")
    print(f"excluded   : {list(EXCLUDE_NETWORKS)}")
    print(f"mode       : {'DRY RUN' if a.dry_run else 'write'}\n")

    # --- 1. Domain ---------------------------------------------------------
    print("domain:")
    region = np.asarray(a.bbox, float) if a.bbox else domain_from_local(
        root, a.variable_dir)

    # --- 2. Metadata and the selection funnel ------------------------------
    print(f"\nreading {a.meta_csv}")
    # The variable is passed through: the CSV counts observations per variable
    # in separate columns, and selecting on the wrong one returns stations that
    # measure something else entirely.
    meta = load_station_metadata(a.meta_csv, target=tslice, region=region,
                                 exclude=EXCLUDE_NETWORKS, var=a.variable)
    metadata_summary(meta, min_coverage=a.min_coverage)

    ok = meta[meta.has_var & meta.in_region]
    print(f"\ncoverage ladder ({len(ok)} with '{a.variable}', inside domain):")
    for thr in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90):
        mark = "  <-- selected" if abs(thr - a.min_coverage) < 1e-9 else ""
        print(f"  span >= {thr:>4.0%}: {int((ok.coverage >= thr).sum()):>5}{mark}")

    selected = select_from_metadata(meta, min_coverage=a.min_coverage,
                                    n_per_network=a.n_per_network, seed=a.seed)

    # Filter BEFORE any narrowing, so stations.csv records exactly the
    # population that could be fetched.
    if not a.no_catalog_check:
        print("\nchecking the selection against the HDP catalog:")
        selected = filter_to_catalog(selected)

    # stations.csv describes the WHOLE selection, always -- every network, every
    # shard. It is the denominator for every delivery figure and the source for
    # coverage-based subsetting, so it must not depend on which subset this
    # invocation happens to fetch. Deriving it after --networks or --shard would
    # silently rewrite it as that subset, which is how a 20-network archive came
    # to be measured against a 2-network selection.
    full_ids = {str(i) for ids in selected.values() for i in ids}
    rows_full = meta[meta.station_id.astype(str).isin(full_ids)]
    if not a.dry_run or a.refresh_metadata:
        rows_full.to_csv(out_dir / "stations.csv", index=False)
        print(f"\nmetadata for {len(rows_full)} stations (all networks) -> "
              f"{out_dir / 'stations.csv'}")
    if a.refresh_metadata:
        return 0

    # Only now narrow to what this invocation fetches.
    if a.networks:
        selected = {n: v for n, v in selected.items() if n in a.networks}
        print(f"restricted to {a.networks}")
    if nshard is not None:
        selected = shard_of(selected, k, nshard)
        print(f"\nshard {k} of {nshard}:")
    for net, ids in sorted(selected.items(), key=lambda kv: -len(kv[1])):
        print(f"  {net:<12} {len(ids):>4}")

    # Save the full metadata rows for the selection. `coverage` does not survive
    # into the zarr, and without it a looser store cannot be subset later.
    sel_ids = {str(i) for ids in selected.values() for i in ids}
    rows = rows_full

    # --- 3. Resume ---------------------------------------------------------
    already: set[str] = set()
    # Stations already in the MERGED store count as done, or an unsharded top-up
    # would refetch the whole archive into staging.
    if merged_path.exists() and merged_path != path and not a.no_resume:
        try:
            mg = xr.open_zarr(merged_path, consolidated=True)
            already |= _done_for(mg, a.variable)
            print(f"{merged_path.name}: {len(already)} stations already merged")
        except Exception as e:
            print(f"could not read {merged_path.name} ({type(e).__name__})")

    if path.exists() and not a.no_resume:
        try:
            ex = xr.open_zarr(path, consolidated=True)
            already |= _done_for(ex, a.variable)
            print(f"resuming: {ex.sizes['station_id']} stations in "
                  f"{path.name} ({ex.sizes['time']:,} steps), "
                  f"{len(already)} known in total")
        except ValueError as e:
            if "conflicting sizes" in str(e):
                # A failed append left arrays at different lengths. Repairing in
                # place is far cheaper than refetching the shard.
                print(f"{path.name} is ragged from an interrupted append; "
                      "repairing in place")
                repair_store(path)
                ex = xr.open_zarr(path, consolidated=True)
                already |= {str(v) for v in ex.station_id.values}
                print(f"resuming: {len(already)} stations after repair")
            else:
                raise
        except Exception as e:
            print(f"existing store unreadable ({type(e).__name__}: {e}) "
                  "-- rebuilding")

    todo = {n: [i for i in ids if str(i) not in already]
            for n, ids in selected.items()}
    todo = {n: ids for n, ids in todo.items() if ids}
    n_todo = sum(len(v) for v in todo.values())
    print(f"\n{n_todo} stations to fetch across {len(todo)} networks "
          f"({len(sel_ids)} selected, {len(already)} stored)")

    if not a.dry_run:
        (out_dir / "selection.json").write_text(json.dumps(
            {"variable": a.variable, "time_start": tslice[0],
             "time_end": tslice[1], "min_coverage": a.min_coverage,
             "n_per_network": a.n_per_network, "seed": a.seed,
             "excluded_networks": list(EXCLUDE_NETWORKS),
             "region": "d03 perimeter" if a.bbox is None else list(a.bbox),
             "selected": {n: len(v) for n, v in selected.items()},
             "total_selected": len(full_ids),
             "batch_size": a.batch_size,
             "shard": None if nshard is None else [k, nshard]}, indent=1))

    if a.dry_run:
        print(f"\nwould write {path}")
        return 0
    if n_todo == 0:
        print("nothing to fetch; store is complete")
        return 0

    # --- 4. Fetch and append -----------------------------------------------
    # Append only if THIS store already exists. Deriving it from `already`
    # would be wrong now that `already` also counts stations in the merged
    # store: a fresh staging store would be opened in append mode and fail with
    # "append_dim='station_id' does not match any existing dataset dimensions".
    append = path.exists()
    started = time.time()
    for net, ids in sorted(todo.items(), key=lambda kv: -len(kv[1])):
        print("\n" + "=" * 60 + f"\n{net} ({len(ids)} stations)\n" + "=" * 60)
        try:
            build_zarr(net, ids, str(path), time_slice=tslice, var=a.variable,
                       freq=FREQ, batch_size=a.batch_size,
                       station_chunk=BUILD_STATION_CHUNK, append=append)
            append = True
        except RuntimeError as e:
            # Every station in this network was empty or failed. Not fatal --
            # the others still carry the analysis.
            print(f"  {net} produced nothing: {e}")

    if not append:
        raise RuntimeError("no network produced any data")

    # --- 5. Verify ---------------------------------------------------------
    ds = xr.open_zarr(path, consolidated=True)
    print(f"\nstore: {ds.sizes['station_id']} stations x "
          f"{ds.sizes['time']:,} hourly steps "
          f"({(time.time() - started) / 3600:.1f} h)")
    per_network_coverage(ds, var=a.variable)

    got = {str(i) for i in ds.station_id.values}
    lost = sel_ids - got
    if lost:
        # sfcwind_nobs counts a station's WHOLE record, not the window, so a
        # station spanning 1997-2022 can pass the filter and return nothing.
        print(f"\n{len(lost)} selected stations returned no data in window:")
        print(rows[rows.station_id.astype(str).isin(lost)]
              .groupby("network").size().sort_values(ascending=False).to_string())

    print()
    # Only wind has default plausibility bounds; other variables must supply
    # their own, and [0, 75] m/s would pass every impossible humidity value.
    if a.variable == "sfcWind":
        outlier_report(ds, var=a.variable, top=10)
    else:
        print(f"(no default outlier bounds for '{a.variable}' -- see "
              "rh_analysis.qc for humidity)")
    on_disk = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"\non disk {on_disk / 1024 ** 3:.2f} GB")
    if nshard is not None:
        print(f"\nshard {k}/{nshard} done. When every shard has finished:\n"
              f"  python fetch_hdp.py {a.variable} --merge")
    else:
        print(f"\nstaged in {path.name}. Fold into {a.name} with:\n"
              f"  python fetch_hdp.py {a.variable} --merge")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)