#!/usr/bin/env python
"""
Download ERA5 from the Earthmover/Arraylake store to local disk, cropped to a
bounding box and a time window, kept at native HOURLY frequency.

    python fetch_era5.py --list                      # variables in the store
    python fetch_era5.py u10 v10 --dry-run           # crop, size, no read
    python fetch_era5.py u10 v10                     # ~11 GB for CA, 1980-2014
    python fetch_era5.py t2m d2m sp

Output:

    /shared/data/era5/era5_hourly.zarr        one store, variables added over time
    /shared/data/era5/bbox.json               the crop actually used

HOURLY IS THE POINT. Every consumer aggregates differently and they cannot share
a pre-reduced file:

  * monthly and daily scalar means -- speed must be formed HOURLY and averaged
    afterwards; averaging the components first gives the vector mean, about 20%
    lower in this domain
  * daily maxima, for the extremes comparison
  * the vector-vs-scalar averaging test, which needs the components themselves
    rather than any speed

Storing the raw components once and reducing locally makes each of those a disk
read instead of a multi-decade cloud query, and removes the cached-aggregation
layer (and its stale-window failure mode) entirely.

THE BOUNDING BOX comes from /shared/data/grids/bbox.json when it exists, so this
matches the crop the analysis uses. Failing that, pass --bbox explicitly.

APPENDING VARIABLES. Re-running with new variables adds them to the same store,
provided the box and window match what is already there; a mismatch is refused
rather than silently written, since two variables on different grids cannot be
combined afterwards.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import dask
import numpy as np
import xarray as xr

DEFAULT_ROOT = "/shared/data"
ZARR_FORMAT = 2
TARGET_CHUNK_MB = 128

# Match fetch_local.py: CMIP6 historical ends 2014.
TIME_SLICE = ("1980-01-01", "2014-12-31")

# California / WRF d03 fallback, used only when no bbox.json exists.
FALLBACK_BBOX = (31.0, 43.8, -125.5, -113.0)
BBOX_PAD = 0.25

ERA5_REPO = "earthmover-public/era5-surface-aws"
ERA5_BRANCH = "main"
# The `temporal` group stores long time series in 12x12 tiles. The `spatial`
# group stores one global map per hour and is roughly 40x slower for a regional
# multi-decade query -- the single most important choice in this script.
ERA5_GROUP = "temporal"


def open_era5(repo=ERA5_REPO, branch=ERA5_BRANCH, group=ERA5_GROUP,
              login=False) -> xr.Dataset:
    """Open the store lazily.

    chunks={} keeps the on-disk chunking as dask arrays. chunks=None, as the
    Arraylake docs suggest, skips dask entirely -- fine for a peek, wrong for a
    multi-decade read.
    """
    from arraylake import Client

    client = Client()
    if login:
        client.login()
    session = client.get_repo(repo).readonly_session(branch)
    return xr.open_dataset(session.store, engine="zarr", consolidated=False,
                           zarr_format=3, chunks={}, group=group)


def resolve_bbox(root: Path, explicit) -> tuple[tuple[float, ...], str]:
    """(lat_min, lat_max, lon_min, lon_max) and where it came from."""
    if explicit:
        return tuple(float(v) for v in explicit), "--bbox"
    bpath = root / "grids" / "bbox.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
        return ((b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"]),
                str(bpath))
    return FALLBACK_BBOX, "built-in fallback (no grids/bbox.json)"


def crop(ds: xr.Dataset, bbox, pad: float = BBOX_PAD) -> xr.Dataset:
    """Bounding-box crop, handling the store's own conventions.

    Three things vary and all three bite: longitude may run 0-360 rather than
    -180-180, latitude may descend, and slicing a descending axis needs its
    bounds reversed. Reversing with a step slice rather than sortby matters --
    sortby is a fancy index and builds one task per chunk, which is ruinous here.
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_min, lat_max = lat_min - pad, lat_max + pad
    lon_min, lon_max = lon_min - pad, lon_max + pad

    is_360 = float(ds.longitude.max()) > 180
    if is_360:
        lo, hi = lon_min % 360, lon_max % 360
        if lo > hi:
            raise ValueError("bounding box wraps the prime meridian")
    else:
        lo, hi = lon_min, lon_max

    descending = float(ds.latitude[0]) > float(ds.latitude[-1])
    lat_slice = slice(lat_max, lat_min) if descending else slice(lat_min, lat_max)
    out = ds.sel(latitude=lat_slice, longitude=slice(lo, hi))
    if descending:
        out = out.isel(latitude=slice(None, None, -1))
    if is_360:
        out = out.assign_coords(
            longitude=(((out.longitude + 180) % 360) - 180))
    if not np.all(np.diff(out.longitude.values) > 0):
        out = out.sortby("longitude")

    print(f"  cropped to {out.sizes['latitude']} x {out.sizes['longitude']} cells "
          f"({float(out.latitude.min()):.2f}..{float(out.latitude.max()):.2f} N, "
          f"{float(out.longitude.min()):.2f}..{float(out.longitude.max()):.2f} E)")
    return out


def rechunk(ds: xr.Dataset, target_mb: int = TARGET_CHUNK_MB) -> xr.Dataset:
    """Uniform, time-contiguous chunks sized by bytes.

    Two zarr constraints make this mandatory rather than an optimisation: chunks
    must be uniform except the last, and no single buffer may exceed 2 GiB. A
    byte budget satisfies both on any grid or dtype, where a fixed step count
    silently violates the second one on a finer grid.

    Time is the contiguous dimension because every consumer reduces over it.
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
    print(f"  chunk: {n_steps} steps = {n_steps * per_step / 1024 ** 2:.0f} MB "
          f"({int(np.ceil(ds.sizes['time'] / n_steps))} chunks)")
    return ds.chunk({"time": n_steps} | {d: -1 for d in ds.dims if d != "time"})


def check_compatible(path: Path, ds: xr.Dataset, bbox, tslice) -> list[str]:
    """Existing store must share the grid and window. Returns vars already there.

    Two variables cropped to different boxes cannot be combined afterwards, and
    the mismatch would surface much later as a silent alignment intersection
    rather than an error. Refusing here is the cheap place to catch it.
    """
    old = xr.open_zarr(path, consolidated=True)
    for axis in ("latitude", "longitude"):
        if old.sizes[axis] != ds.sizes[axis] or not np.allclose(
                old[axis].values, ds[axis].values):
            raise RuntimeError(
                f"{path.name} has a different {axis} axis "
                f"({old.sizes[axis]} vs {ds.sizes[axis]} cells). It was built "
                f"for a different bounding box; move it aside or use --overwrite.")
    if old.sizes["time"] != ds.sizes["time"]:
        raise RuntimeError(
            f"{path.name} covers {old.sizes['time']:,} steps, this crop has "
            f"{ds.sizes['time']:,}. Different time window; move it aside "
            "or use --overwrite.")
    return list(old.data_vars)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variables", nargs="*",
                    help="ERA5 variable names, e.g. u10 v10 t2m")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--name", default="era5_hourly.zarr")
    ap.add_argument("--time-slice", nargs=2, default=list(TIME_SLICE),
                    metavar=("START", "END"))
    ap.add_argument("--bbox", nargs=4, default=None, type=float,
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                    help="override grids/bbox.json")
    ap.add_argument("--pad", type=float, default=BBOX_PAD)
    ap.add_argument("--group", default=ERA5_GROUP,
                    choices=["temporal", "spatial"])
    ap.add_argument("--login", action="store_true",
                    help="run the Arraylake interactive login first")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=TARGET_CHUNK_MB)
    ap.add_argument("--list", action="store_true",
                    help="print the store's variables and exit")
    a = ap.parse_args()

    dask.config.set(scheduler="threads", num_workers=a.workers)
    root = Path(a.root)
    out_dir = root / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / a.name
    tslice = (a.time_slice[0], a.time_slice[1])

    print(f"opening {ERA5_REPO} [{a.group}]")
    src = open_era5(group=a.group, login=a.login)

    if a.list:
        print(f"\n{len(src.data_vars)} variables:")
        for v in sorted(src.data_vars):
            print(f"  {v:<12} {src[v].attrs.get('long_name', '')}")
        print(f"\ndims: {dict(src.sizes)}")
        print(f"time: {str(src.time.values[0])[:10]} -> "
              f"{str(src.time.values[-1])[:10]}")
        return 0

    if not a.variables:
        ap.error("name at least one variable (or use --list)")
    missing = [v for v in a.variables if v not in src.data_vars]
    if missing:
        ap.error(f"not in the store: {missing}. Use --list to see what is.")

    bbox, origin = resolve_bbox(root, a.bbox)
    print(f"bbox from {origin}: {bbox[0]:.2f}..{bbox[1]:.2f} N, "
          f"{bbox[2]:.2f}..{bbox[3]:.2f} E  (pad {a.pad})")

    ds = crop(src[list(a.variables)], bbox, a.pad).sel(time=slice(*tslice))
    if ds.sizes["time"] == 0:
        raise RuntimeError(f"no ERA5 steps in {tslice[0]}..{tslice[1]}")
    print(f"  {ds.sizes['time']:,} hourly steps "
          f"{str(ds.time.values[0])[:13]} -> {str(ds.time.values[-1])[:13]}")

    size = sum(ds[v].nbytes for v in ds.data_vars) / 1024 ** 3
    free = shutil.disk_usage(out_dir).free / 1024 ** 3
    print(f"  {list(ds.data_vars)}  ~{size:.2f} GB uncompressed "
          f"({free:.0f} GB free)")

    if path.exists() and a.overwrite and not a.dry_run:
        shutil.rmtree(path)
    mode, append = "w", []
    if path.exists():
        append = check_compatible(path, ds, bbox, tslice)
        todo = [v for v in a.variables if v not in append]
        if not todo:
            print(f"\n{path.name} already holds {a.variables} -- nothing to do "
                  "(--overwrite to rebuild)")
            return 0
        print(f"\nappending {todo} to existing {append}")
        ds, mode = ds[todo], "a"

    ds = rechunk(ds, a.chunk_mb)
    ds.attrs.update(source=f"{ERA5_REPO} [{a.group}]",
                    time_start=tslice[0], time_end=tslice[1],
                    lat_min=bbox[0], lat_max=bbox[1],
                    lon_min=bbox[2], lon_max=bbox[3], pad_deg=a.pad,
                    note="native hourly; consumers aggregate as they need")

    if a.dry_run:
        print(f"\nwould write {path} (mode={mode})")
        return 0

    for v in ds.variables:
        ds[v].encoding = {}

    t = time.time()
    from dask.diagnostics import ProgressBar
    if mode == "w":
        # Write to .tmp and rename, so an interrupted run never leaves a
        # partial store that looks complete to the next one.
        tmp = path.parent / (path.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            with ProgressBar(minimum=2.0):
                ds.to_zarr(tmp, mode="w", consolidated=True,
                           zarr_format=ZARR_FORMAT)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        tmp.rename(path)
    else:
        with ProgressBar(minimum=2.0):
            ds.to_zarr(path, mode="a", consolidated=True,
                       zarr_format=ZARR_FORMAT)

    dt = time.time() - t
    print(f"\nwrote {path} in {dt:.0f}s ({size / max(dt, 1) * 1024:.0f} MB/s)")

    (out_dir / "bbox.json").write_text(json.dumps(
        {"lat_min": bbox[0], "lat_max": bbox[1],
         "lon_min": bbox[2], "lon_max": bbox[3], "pad_deg": a.pad,
         "origin": origin, "time_start": tslice[0], "time_end": tslice[1]},
        indent=1))

    final = xr.open_zarr(path, consolidated=True)
    print(f"store now holds {list(final.data_vars)}, {dict(final.sizes)}")
    on_disk = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"on disk {on_disk / 1024 ** 3:.2f} GB "
          f"({size / max(on_disk / 1024 ** 3, 1e-9):.1f}x compression)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
