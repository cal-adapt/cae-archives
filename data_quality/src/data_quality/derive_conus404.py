#!/usr/bin/env python
"""
Derive CONUS404 wind-speed aggregations from the local components.

    python derive_conus404.py --dry-run
    python derive_conus404.py

Output:

    /shared/data/conus404/conus404_speed_mon.zarr   wspeed (+ wspeed_vector)
    /shared/data/conus404/conus404_speed_day.zarr   wspeed, wspeed_max
    /shared/data/conus404/conus404_diurnal.zarr     month x hour climatology

WHAT THE SOURCE CADENCE DECIDES. CONUS404 publishes no wind-speed variable, only
U10/V10, so what these components mean depends on which product was fetched:

  * HOURLY source -- the components are instantaneous. sqrt(u^2+v^2) per step,
    aggregated afterwards, is the true SCALAR mean. Everything below is valid.

  * DAILY or MONTHLY source -- the components are already time-averaged, so
    their magnitude is the VECTOR mean, roughly 20% BELOW the scalar mean. That
    is not a rounding difference: it is the size of the model discrepancies the
    project is measuring, and it lands squarely on the number the "ERA5 was the
    wrong yardstick" argument rests on (CONUS404 at 1.546 against ERA5, LOCA2 at
    0.903 on CONUS404's own grid). A 20% low bias moves both, in the direction
    that weakens the conclusion.

So a non-hourly source is written as `wspeed_vector`, never as `wspeed`, and the
attributes say why. It remains fine for SPATIAL PATTERN work -- pattern_corr and
std_ratio are invariant under a uniform scaling -- and unusable for ratio,
var_ratio or extremes.

MULTIPLE STORES ARE CONCATENATED. Appending along time is serial, so a long
fetch is best split across terminals into per-year-range stores. Any store
matching the glob is opened and joined on time, deduplicated and sorted.

GRID-RELATIVE COMPONENTS. U10/V10 are relative to the rotated projection rather
than true north. Irrelevant for magnitude, which is invariant under the
rotation; wrong for direction, which needs COSALPHA/SINALPHA from the static
store.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

import dask
import numpy as np
import xarray as xr

DEFAULT_ROOT = Path("/shared/data")
ZARR_FORMAT = 2
TARGET_CHUNK_MB = 128


def open_source(root: Path, pattern: str) -> tuple[xr.Dataset, list[str]]:
    """Open every matching store and concatenate along time."""
    paths = sorted((root / "conus404").glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"nothing matching {pattern} under {root / 'conus404'} "
            "-- run fetch_conus404.py first")
    parts = [xr.open_zarr(p, consolidated=True) for p in paths]
    for p, d in zip(paths, parts):
        print(f"  {p.name}: {d.sizes['time']:,} steps "
              f"{str(d.time.values[0])[:10]} -> {str(d.time.values[-1])[:10]}")
    ds = parts[0] if len(parts) == 1 else xr.concat(
        parts, dim="time", coords="minimal", compat="override")
    if len(parts) > 1:
        # Year-range stores can overlap at a boundary if one was rebuilt.
        _, keep = np.unique(ds.time.values, return_index=True)
        ds = ds.isel(time=np.sort(keep))
    return ds, [p.name for p in paths]


def cadence(ds: xr.Dataset) -> str:
    h = float(np.median(np.diff(ds.time.values[:400])
                        .astype("timedelta64[h]").astype(int)))
    return {1: "1hr", 24: "day"}.get(int(h),
                                     "mon" if 27 * 24 <= h <= 32 * 24 else "other")


def rechunk(ds: xr.Dataset, target_mb: int = TARGET_CHUNK_MB) -> xr.Dataset:
    """Uniform, time-contiguous chunks sized by bytes.

    Chunks must be uniform except the last and no buffer may exceed 2 GiB; a
    byte budget satisfies both on any grid, where a fixed step count silently
    violates the second one as the grid gets finer.
    """
    if "time" not in ds.dims:
        return ds.chunk({d: -1 for d in ds.dims})
    per_step = 0
    for v in ds.data_vars:
        if "time" not in ds[v].dims:
            continue
        n = ds[v].dtype.itemsize
        for d in ds[v].dims:
            if d != "time":
                n *= int(ds.sizes[d])
        per_step += n
    n_steps = min(int(ds.sizes["time"]),
                  max(1, int(target_mb * 1024 ** 2) // max(per_step, 1)))
    return ds.chunk({"time": n_steps} | {d: -1 for d in ds.dims if d != "time"})


def write(ds: xr.Dataset, path: Path, attrs: dict, overwrite: bool) -> None:
    ds = rechunk(ds).copy()
    for v in ds.variables:
        ds[v].encoding = {}
    ds.attrs.update(attrs)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists():
        print(f"  {path.name} exists (--overwrite to rebuild)")
        return
    tmp = path.parent / (path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        ds.to_zarr(tmp, mode="w", consolidated=True, zarr_format=ZARR_FORMAT)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.rename(path)
    gb = sum(ds[v].nbytes for v in ds.data_vars) / 1024 ** 3
    print(f"  wrote {path.name}  {dict(ds.sizes)}  {gb:.2f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--pattern", default="conus404_hourly*.zarr",
                    help="glob for source stores (default: %(default)s)")
    ap.add_argument("--allow-vector", action="store_true",
                    help="proceed on a non-hourly source. Output is the VECTOR "
                         "mean, ~20%% low; valid for spatial pattern only.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    dask.config.set(scheduler="threads", num_workers=a.workers)
    root = Path(a.root)
    out = root / "conus404"

    print(f"source stores matching {a.pattern}:")
    ds, names = open_source(root, a.pattern)
    cad = cadence(ds)
    gb = sum(ds[v].nbytes for v in ds.data_vars) / 1024 ** 3
    print(f"\n  combined: {dict(ds.sizes)}  {cad}  ~{gb:.1f} GB to read")
    print(f"  {str(ds.time.values[0])[:13]} -> {str(ds.time.values[-1])[:13]}")

    scalar = cad == "1hr"
    if not scalar:
        msg = (f"source is {cad}, so U10/V10 are already time-averaged. Their "
               "magnitude is the VECTOR mean, ~20% below the scalar mean.")
        if not a.allow_vector:
            print(f"\nREFUSING: {msg}\n"
                  "Fetch the hourly product for level comparisons, or pass "
                  "--allow-vector if you only need spatial pattern.")
            return 1
        print(f"\nWARNING: {msg}\n"
              "Output is named wspeed_vector. Valid for pattern_corr and "
              "std_ratio; not for ratio, var_ratio or extremes.")

    name = "wspeed" if scalar else "wspeed_vector"
    w = np.sqrt(ds.U10 ** 2 + ds.V10 ** 2)
    w.name = name
    w.attrs.update(units="m s-1",
                   long_name=("10 m scalar wind speed, formed hourly" if scalar
                              else f"10 m VECTOR-mean wind speed from {cad} "
                                   "components -- biased ~20% low"))
    if "lat" in ds.coords:
        w = w.assign_coords(lat=ds.lat, lon=ds.lon)

    products: list[tuple[str, xr.Dataset]] = []
    mon = xr.Dataset({name: w.resample(time="MS").mean()})
    products.append(("conus404_speed_mon.zarr", mon))

    if scalar:
        # Daily mean for the cadence check, daily max for extremes. The max is
        # over HOURLY SAMPLES, not model timesteps -- WRF's wspd10max is higher
        # by construction, so that comparison is a bound rather than equality.
        products.append(("conus404_speed_day.zarr", xr.Dataset({
            "wspeed": w.resample(time="1D").mean(),
            "wspeed_max": w.resample(time="1D").max()})))
        # Month x hour climatology. The gridded comparison has never had one;
        # monthly means cannot see a sea breeze or a nocturnal downslope jet.
        products.append(("conus404_diurnal.zarr",
                         w.groupby("time.month").map(
                             lambda g: g.groupby("time.hour").mean("time")
                         ).to_dataset(name="wspeed")))

    if a.dry_run:
        for fname, d in products:
            mb = sum(d[v].nbytes for v in d.data_vars) / 1024 ** 2
            print(f"  would write {fname}: {dict(d.sizes)}  {mb:.0f} MB")
        return 0

    print(f"\ncomputing (one pass over {gb:.1f} GB)...")
    t = time.time()
    from dask.diagnostics import ProgressBar
    with ProgressBar(minimum=5.0):
        computed = dask.compute(*[d for _, d in products])
    print(f"  done in {(time.time() - t) / 60:.1f} min\n")

    attrs = {"source": f"CONUS404 {cad} components, local: {', '.join(names)}",
             "aggregation": ("scalar speed formed hourly, then aggregated"
                             if scalar else
                             "VECTOR mean of already-averaged components; "
                             "~20% below the scalar mean. Spatial pattern only."),
             "components": "grid-relative; magnitude is rotation-invariant, "
                           "direction is not (see conus404_static)"}
    for (fname, _), d in zip(products, computed):
        write(d, out / fname, attrs, a.overwrite)

    dm = float(computed[0][name].mean())
    print(f"\ndomain mean {name}: {dm:.3f} m/s")
    if not scalar:
        print(f"  scalar mean would be roughly {dm / 0.8:.3f} m/s "
              "if the ~20% vector penalty holds here as it does for ERA5")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
