#!/usr/bin/env python
"""
Download Analytics Engine gridded data to local disk: one zarr per product and
cadence, with every model/member kept along the ensemble dimension.

    python fetch_local.py wspeed --dry-run           # sizes, no download
    python fetch_local.py wspeed                     # monthly, all products
    python fetch_local.py wspeed --cadence mon 1hr

Layout:

    /shared/data/{variable}/
        loca2_mon.zarr          all 46 members along `sim`
        wrf-gcm_mon.zarr        the 5 bias-adjusted models along `sim`
        wrf-era5_mon.zarr       single run
        _manifest.json


VARIABLE NAMES DIFFER BY PRODUCT. LOCA2 publishes `wspeed`; WRF publishes
`wspd10mean` / `wspd10max`. Pass a canonical name from VARIABLES below, or give
explicit ids with --loca2-var / --wrf-var.

SIZE. Monthly is small. Hourly is not: d03 at 3 km over 35 years is order 100 GB
per member uncompressed. Always --dry-run an hourly job first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from climakitae import ClimateData
import pandas as pd
import xarray as xr
from climakitae import load
import dask

DEFAULT_ROOT = "/shared/data"
ZARR_FORMAT = 2
GRID_LABEL = "d03"

# CMIP6 historical runs end in 2014; SSPs pick up from 2015.
TIME_SLICE = ("1980-01-01", "2014-12-31")

# Canonical name -> per-activity variable_id. Anything unlisted still works via
# --loca2-var / --wrf-var.
VARIABLES = {
    "wspeed":    {"LOCA2": "wspeed",  "WRF": "wspd10mean"},
    "wspd10max": {"LOCA2": None,      "WRF": "wspd10max"},
    "tas":       {"LOCA2": "tas",     "WRF": "t2"},
    "tasmax":    {"LOCA2": "tasmax",  "WRF": "t2max"},
    "tasmin":    {"LOCA2": "tasmin",  "WRF": "t2min"},
    "pr":        {"LOCA2": "pr",      "WRF": "prec"},

    # --- relative humidity --------------------------------------------------
    # LOCA2 publishes NO MEAN RH -- only the daily extremes. That is not an
    # oversight to work around: `hursmin` is the afternoon minimum, which is the
    # fire-weather variable, so LOCA2 supports the fire use case directly even
    # though it cannot enter a comparison of mean humidity at all.
    #
    # The midrange (hursmax + hursmin) / 2 is NOT the daily mean -- RH's diurnal
    # cycle is asymmetric, so the midrange sits low by an amount that grows with
    # the diurnal range, biasing dry regions most. See rh_analysis.loca2_daily.
    "hurs":      {"LOCA2": None,       "WRF": "rh"},
    "hursmax":   {"LOCA2": "hursmax",  "WRF": None},
    "hursmin":   {"LOCA2": "hursmin",  "WRF": None},
}

# One entry per product: the query that returns every member at once.
#   filter -> value for climakitae's filter_unadjusted_models processor,
#             None to leave the processor off entirely.
PRODUCTS = {
    "loca2":    dict(activity="LOCA2", experiment="historical",
                     institution="UCSD"),
    "wrf-gcm":  dict(activity="WRF",   experiment="historical",
                     institution="UCLA"),
    "wrf-era5": dict(activity="WRF",   experiment="reanalysis",
                     institution="UCLA"),
}

MEMBER_DIMS = ("sim", "simulation", "member_id", "member")

# Target size of one written chunk. Kept well under the 2 GiB zarr v2 codec
# ceiling, and small enough that a failed write costs little.
TARGET_CHUNK_MB = 128


def member_dim(da) -> str | None:
    return next((d for d in MEMBER_DIMS if d in da.dims), None)


def rechunk(da: xr.DataArray, target_mb: int = TARGET_CHUNK_MB) -> xr.DataArray:
    """Uniform, time-contiguous chunks sized by bytes. Required, not an optimisation.

    Two hard zarr constraints make this mandatory rather than nice to have:

      * CHUNKS MUST BE UNIFORM except the last. climakitae concatenates sources
        of differing length, so the time axis arrives as e.g.
        (915, 1695, 1695, ...) and zarr refuses it outright.
      * NO BUFFER OVER 2 GiB. A d03 field is 495 x 559 float32 = 1.06 MB per
        timestep, so a plausible-looking 3660-step chunk is 4 GB and the codec
        raises "Codec does not support buffers of > 2147483647 bytes".

    Sizing from a byte budget rather than a step count makes the rule hold for
    any grid, dtype or cadence. Time is the contiguous dimension because every
    downstream metric reduces over it, and the member dim is chunked at 1
    because metrics_over_members walks members one at a time -- a chunk spanning
    several members would make each single-member read pull its neighbours too.
    """
    md = member_dim(da)
    chunks = {d: -1 for d in da.dims}
    if md:
        chunks[md] = 1
    if "time" in da.dims:
        per_step = da.dtype.itemsize
        for d in da.dims:
            if d not in ("time", md):
                per_step *= int(da.sizes[d])
        n = max(1, int(target_mb * 1024 ** 2) // max(per_step, 1))
        chunks["time"] = min(int(da.sizes["time"]), n)
        print(f"    chunk: {chunks['time']} steps x 1 member "
              f"= {chunks['time'] * per_step / 1024 ** 2:.0f} MB")
    return da.chunk(chunks)


def fetch(spec: dict, variable_id: str, table_id: str,
          time_slice: tuple[str, str],
          verbosity: int
         ) -> xr.Dataset:
    """One ClimateData query for a whole product, returned lazily."""
    

    processes = {"time_slice": time_slice}
    q = (ClimateData(verbosity=verbosity)
         .catalog("cadcat")
         .activity_id(spec["activity"])
         .experiment_id(spec["experiment"])
         .table_id(table_id)
         .grid_label(GRID_LABEL)
         .variable(variable_id)
         .processes(processes))
    if spec["institution"]:
        q = q.institution_id(spec["institution"])
    q.show_query()
    ds = q.get()
    # print(ds)
    # raise
    if ds is None:
        raise RuntimeError(
            f"query returned nothing: {spec['activity']}/{variable_id}/{table_id}")
    return ds.to_dataset(name=variable_id) if isinstance(ds, xr.DataArray) else ds


def write(da: xr.DataArray, path: Path, variable: str, table_id: str,
          attrs: dict, dry_run: bool,
          target_mb: int = TARGET_CHUNK_MB) -> dict:
    """Write one product to zarr, atomically."""
    da = rechunk(da, target_mb)
    size = da.nbytes / 1024 ** 3
    md = member_dim(da)
    n_mem = int(da.sizes[md]) if md else 1
    info = {"path": str(path), "gb_uncompressed": round(size, 2),
            "members": n_mem, "dims": {k: int(v) for k, v in da.sizes.items()}}

    print(f"    {n_mem} member(s), {dict(da.sizes)}, ~{size:.2f} GB")
    if md:
        names = [str(m) for m in da[md].values]
        print(f"    {md}: {names if len(names) <= 8 else names[:8] + ['...']}")
    if dry_run:
        print(f"    would write {path}")
        return info

    # Strip inherited encoding: netCDF compressors and fill values conflict with
    # the zarr writer, and numpy 2 StringDType is not writable at all.
    ds = da.to_dataset(name=variable)
    for v in ds.variables:
        ds[v].encoding = {}
        if ds[v].dtype.kind in ("T", "U"):
            ds[v] = ds[v].astype(object)
    ds.attrs.update(attrs)

    tmp = path.parent / (path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    t = time.time()
    from dask.diagnostics import ProgressBar
    try:
        with ProgressBar(minimum=2.0):
            ds.to_zarr(tmp, mode="w", consolidated=True,
                       zarr_format=ZARR_FORMAT)
    except Exception:
        # A partial store is dead weight on a shared filesystem, and would make
        # the next run's free-space estimate wrong.
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.rename(path)              # only now does it exist under its real name
    dt = time.time() - t
    print(f"    wrote {path.name} in {dt:.0f}s "
          f"({size / max(dt, 1) * 1024:.0f} MB/s)")
    info["seconds"] = round(dt, 1)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variable", nargs="?",
                    help=f"canonical name, one of: {', '.join(VARIABLES)}")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--cadence", nargs="*", default=["mon"],
                    choices=["mon", "day", "1hr"])
    ap.add_argument("--products", nargs="*", default=list(PRODUCTS),
                    choices=list(PRODUCTS))
    ap.add_argument("--time-slice", nargs=2, default=list(TIME_SLICE),
                    metavar=("START", "END"))
    ap.add_argument("--loca2-var", default=None)
    ap.add_argument("--wrf-var", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verbosity", type=int, default=-1)
    ap.add_argument("--chunk-mb", type=int, default=TARGET_CHUNK_MB,
                    help=f"target written chunk size (default {TARGET_CHUNK_MB})")
    a = ap.parse_args()

    if not a.variable:
        ap.error("a variable is required")

    known = VARIABLES.get(a.variable, {})
    vid_for = {"LOCA2": a.loca2_var or known.get("LOCA2"),
               "WRF": a.wrf_var or known.get("WRF")}
    if not any(vid_for.values()):
        ap.error(f"'{a.variable}' is not in VARIABLES; pass --loca2-var "
                 "and/or --wrf-var explicitly")

    dask.config.set(scheduler="threads", num_workers=a.workers)

    tslice = (a.time_slice[0], a.time_slice[1])
    out = Path(a.root) / a.variable
    out.mkdir(parents=True, exist_ok=True)

    print(f"variable   : {a.variable}  "
          f"(LOCA2={vid_for['LOCA2']}, WRF={vid_for['WRF']})")
    print(f"time slice : {tslice[0]} -> {tslice[1]}")
    print(f"destination: {out}")
    print(f"cadences   : {a.cadence}   products: {a.products}")
    print(f"mode       : {'DRY RUN' if a.dry_run else 'write'}"
          f"{' (overwrite)' if a.overwrite else ''}")
    print(f"free space : {shutil.disk_usage(out).free / 1024 ** 3:.0f} GB "
          f"at {a.root}\n")

    manifest, failures = [], []
    for table_id in a.cadence:
        for product in a.products:
            spec = PRODUCTS[product]
            vid = vid_for[spec["activity"]]
            path = out / f"{product}_{table_id}.zarr"

            if vid is None:
                print(f"[{product}/{table_id}] no variable_id -- skipping\n")
                continue
            if path.exists() and a.overwrite and not a.dry_run:
                shutil.rmtree(path)
            if path.exists() and not a.dry_run:
                print(f"[{product}/{table_id}] exists, skipping "
                      f"(--overwrite to rebuild)\n")
                continue

            print(f"[{product}/{table_id}] querying")
            try:
                ds = fetch(spec, vid, table_id, tslice, a.verbosity)
                ds = dask.optimize(ds)[0]
                da = ds[vid] if vid in ds else ds[list(ds.data_vars)[0]]
                manifest.append(write(
                    da, path, a.variable, table_id,
                    {"product": product, "variable_id": vid,
                     "table_id": table_id, "grid_label": GRID_LABEL,
                     "time_start": tslice[0], "time_end": tslice[1]},
                    a.dry_run, target_mb=a.chunk_mb))
                # if not a.dry_run:
                    
            except Exception as e:
                failures.append(f"{product}/{table_id}: {type(e).__name__}: {e}")
                print(f"[{product}/{table_id}] FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
            print()

    total = sum(m.get("gb_uncompressed", 0) for m in manifest)
    print("=" * 62)
    print(f"{len(manifest)} stores, ~{total:.1f} GB uncompressed")
    for f in failures:
        print(f"  FAILED {f}")

    if not a.dry_run and manifest:
        mpath = out / "_manifest.json"
        old = json.loads(mpath.read_text()) if mpath.exists() else []
        mpath.write_text(json.dumps(old + manifest, indent=1))
        print(f"manifest: {mpath}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())