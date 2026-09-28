#!/usr/bin/env python
"""
Split the shared HDP store into one archive per variable.

    python split_hdp_stores.py --dry-run
    python split_hdp_stores.py

    /shared/data/hdp/hdp_hourly.zarr          -> hdp_wind_hourly.zarr  (sfcWind)
    /shared/data/hdp/hdp_hourly.staging.zarr  -> hdp_hurs_hourly.zarr  (hurs)

WHY SEPARATE. A shared store makes `station_id` the natural key for every
bookkeeping decision, and that is wrong once a station can carry one variable
and not another. Three separate bugs came out of it: resume skipped stations
that had wind but not humidity; the merge dropped an entire humidity fetch as
"already seen"; and a partial merge left duplicate station_id labels that only
surfaced later inside xr.align. One store per variable makes each fetch
independent and the key unambiguous.

The cost is that a joint analysis -- humidity and wind at the same site and
hour, which is what a fire-weather index needs -- has to align two stores at
read time. That is one xr.align call, and unlike the alternative it fails
loudly when the populations differ.

NOTHING IS REFETCHED. The staging store already holds the completed humidity
fetch; it is deduplicated and renamed rather than rebuilt.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def load_ids(ds) -> list[str]:
    return [str(s) for s in ds.station_id.values]


def dedupe(ds: xr.Dataset, label: str) -> xr.Dataset:
    """Drop repeated station_id labels, keeping the copy with the most data.

    Keeping the copy with the most non-null values rather than the first: after
    a partial merge the duplicates are typically one full record and one empty,
    and taking the first would sometimes take the empty one.
    """
    ids = load_ids(ds)
    if len(set(ids)) == len(ids):
        print(f"  {label}: {len(ids)} stations, no duplicate ids")
        return ds

    pos: dict[str, list[int]] = {}
    for i, sid in enumerate(ids):
        pos.setdefault(sid, []).append(i)
    n_dupe = sum(len(v) - 1 for v in pos.values())
    print(f"  {label}: {n_dupe} duplicate row(s) across "
          f"{sum(1 for v in pos.values() if len(v) > 1)} station(s)")

    score = sum(ds[v].notnull().sum("time") for v in ds.data_vars).compute()
    keep = sorted(max(idxs, key=lambda i: float(score.values[i]))
                  for idxs in pos.values())
    print(f"  {label}: {len(ids)} -> {len(keep)} stations")
    return ds.isel(station_id=keep)


def write(ds: xr.Dataset, path: Path, var: str, dry_run: bool) -> None:
    """Write one variable's archive, keeping only stations that carry it."""
    if var not in ds:
        raise KeyError(f"'{var}' not in this store (have {sorted(ds.data_vars)})")

    has = (ds[var].notnull().sum("time") > 0).compute().values
    n_drop = int((~has).sum())
    if n_drop:
        print(f"  dropping {n_drop} station(s) with no '{var}' data")
    out = ds[[var]].isel(station_id=np.where(has)[0])

    print(f"  {path.name}: {out.sizes['station_id']} stations x "
          f"{out.sizes['time']:,} steps")
    if dry_run:
        print(f"  would write {path}")
        return

    # load() before touching the coords: an object-dtype coord that is still a
    # dask array makes zarr warn and materialise it anyway during the write.
    out = out.load()
    for v in out.variables:
        out[v].encoding = {}
    # Object dtype for the string coords, matching what build_zarr writes: a
    # fixed-width numpy string takes its width from the longest value present,
    # so a later append of a longer id would be refused.
    for c in ("station_id", "network"):
        if c in out.coords:
            out = out.assign_coords(
                {c: (out[c].dims,
                     np.array([str(x) for x in out[c].values], dtype=object))})
    # Whole time series per chunk: every consumer reduces over time per station.
    # Chunk the DATA only. Chunking the whole dataset would make the object
    # string coords dask arrays again, which is what the warning above is about.
    out = out.chunk({"time": -1, "station_id": 50})
    for c in ("station_id", "network", "lat", "lon", "elevation"):
        if c in out.coords:
            out[c].load()

    tmp = path.parent / (path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    out.to_zarr(tmp, mode="w", consolidated=True, zarr_format=2)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    print(f"  wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/shared/data/hdp")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-originals", action="store_true",
                    help="do not remove the shared store and staging store "
                         "afterwards")
    a = ap.parse_args()

    root = Path(a.root)
    shared = root / "hdp_hourly.zarr"
    staging = root / "hdp_hourly.staging.zarr"

    print(f"root: {root}")
    print(f"mode: {'DRY RUN' if a.dry_run else 'write'}\n")

    # --- wind, from the shared store ---------------------------------------
    if shared.exists():
        print("=== wind ===")
        ds = xr.open_zarr(shared, consolidated=True)
        print(f"  source: {shared.name}, {ds.sizes['station_id']} stations, "
              f"{sorted(ds.data_vars)}")
        ds = dedupe(ds, shared.name)
        write(ds, root / "hdp_wind_hourly.zarr", "sfcWind", a.dry_run)
    else:
        print(f"{shared} not found -- skipping wind")

    # --- humidity, from the staging store ----------------------------------
    # The completed fetch lives here. It is deduplicated and renamed, not
    # rebuilt: the alternative is 45 minutes of round trips for data already
    # on disk.
    print("\n=== humidity ===")
    if staging.exists():
        ds = xr.open_zarr(staging, consolidated=True)
        print(f"  source: {staging.name}, {ds.sizes['station_id']} stations, "
              f"{sorted(ds.data_vars)}")
        ds = dedupe(ds, staging.name)
        write(ds, root / "hdp_hurs_hourly.zarr", "hurs", a.dry_run)
    elif (root / "hdp_hurs_hourly.zarr").exists():
        print("  hdp_hurs_hourly.zarr already exists and no staging store")
    else:
        print(f"  {staging.name} not found. Fetch humidity with:\n"
              "    python src/fetch_hdp.py hurs --name hdp_hurs_hourly.zarr "
              "--batch-size 48")

    # --- verify -------------------------------------------------------------
    if a.dry_run:
        return 0
    print("\n=== result ===")
    import pandas as pd
    for f, var in (("hdp_wind_hourly.zarr", "sfcWind"),
                   ("hdp_hurs_hourly.zarr", "hurs")):
        p = root / f
        if not p.exists():
            continue
        d = xr.open_zarr(p, consolidated=True)
        ids = load_ids(d)
        print(f"\n{f}: {d.sizes['station_id']} stations, "
              f"unique ids: {len(set(ids)) == len(ids)}")
        print(pd.Series([str(n) for n in d.network.values])
              .value_counts().head(8).to_string())

    # Overlap: the stations a joint fire-weather analysis can use.
    w, h = root / "hdp_wind_hourly.zarr", root / "hdp_hurs_hourly.zarr"
    if w.exists() and h.exists():
        wi = set(load_ids(xr.open_zarr(w, consolidated=True)))
        hi = set(load_ids(xr.open_zarr(h, consolidated=True)))
        print(f"\nwind {len(wi)} | humidity {len(hi)} | both {len(wi & hi)}")
        print("The overlap is what a joint analysis can use: fire danger is a "
              "simultaneous\ncondition, so it needs both variables at the same "
              "site and hour.")

    if not a.keep_originals:
        for p in (shared, staging):
            if p.exists():
                shutil.rmtree(p)
                print(f"\nremoved {p.name}")
        print("Pass --keep-originals to retain them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
