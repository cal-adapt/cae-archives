#!/usr/bin/env python
"""
Download CONUS404 from the USGS OSN pod to local disk, cropped to a bounding box
and a time window, kept at native HOURLY frequency.

    python fetch_conus404.py --list                   # variables in the store
    python fetch_conus404.py U10 V10 --dry-run        # crop, size, no read
    python fetch_conus404.py U10 V10                  # year by year, resumable

Output:

    /shared/data/conus404/conus404_hourly.zarr    hourly, appended by year
    /shared/data/conus404/conus404_static.zarr    HGT, LANDMASK, VAR_SSO, lat, lon
    /shared/data/conus404/bbox.json               the crop actually used

THIS IS THE BIG ONE. Cropped to California, 35 years of hourly U10/V10 is order
180 GB uncompressed -- an order of magnitude more than the equivalent ERA5,
because the CONUS404 store is chunked as SPATIAL TILES rather than time runs. A
regional query therefore reads few tiles but every time chunk, and there is no
way to make it cheap. --dry-run reports the number before anything is read.

WRITTEN IN BLOCKS, AND RESUMABLE. A single 180 GB write that dies at 80% loses
everything, so the read is appended along `time` in blocks of roughly a year.

The blocks are INDEX-based, not calendar years. Appending to zarr requires each
write to begin on a chunk boundary, and a calendar year does not: 1980 is a leap
year at 8784 hourly steps, 1981 is 8760, and no chunk size divides both. Year
boundaries therefore leave a short final chunk and the next append fails with
"would overlap multiple Dask chunks". Every block is instead an exact multiple of
the chunk length, so appends always land cleanly; only the final block is short.

The crop indices are pinned in the store attributes and re-verified on every
append, because two blocks cropped to different y/x windows could not be
concatenated afterwards.

GRID-RELATIVE COMPONENTS. U10/V10 are relative to the rotated projection, not to
true north. That is irrelevant for wind SPEED, whose magnitude is invariant under
the rotation, and wrong for wind DIRECTION, which needs COSALPHA/SINALPHA to
rotate. Fetch those too if direction is ever needed; they are 2D statics.

CURVILINEAR CROP. lat/lon are 2D fields over projected x/y, so the box is applied
by finding the index window that contains it -- not by coordinate slicing, which
would silently do nothing here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from dask.diagnostics import ProgressBar
import dask
import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_ROOT = "/shared/data"
ZARR_FORMAT = 2
TARGET_CHUNK_MB = 128

# CONUS404 hourly begins 1979-10-01. Match fetch_local.py at the other end.
TIME_SLICE = ("1980-01-01", "2014-12-31")

FALLBACK_BBOX = (31.0, 43.8, -125.5, -113.0)
BBOX_PAD = 0.25

ENDPOINT = "https://usgs.osn.mghpcc.org/"
URLS = {
    "hourly": "s3://hytest/conus404/conus404_hourly.zarr",
    "daily": "s3://hytest/conus404/conus404_daily.zarr",
    "monthly": "s3://hytest/conus404/conus404_monthly.zarr",
}
# 2D fields worth keeping: terrain height, land mask, sub-grid orographic
# variance. They cost nothing and the terrain analysis needs them.
STATIC = ("HGT", "LANDMASK", "VAR_SSO", "COSALPHA", "SINALPHA")


def open_conus404(product="hourly") -> xr.Dataset:
    """Open the store lazily. Anonymous, no egress fees."""
    import fsspec

    fs = fsspec.filesystem("s3", anon=True,
                           client_kwargs={"endpoint_url": ENDPOINT})
    return xr.open_dataset(fs.get_mapper(URLS[product]), engine="zarr",
                           consolidated=True, chunks={})


def resolve_bbox(root: Path, explicit) -> tuple[tuple[float, ...], str]:
    if explicit:
        return tuple(float(v) for v in explicit), "--bbox"
    for cand in (root / "grids" / "bbox.json", root / "era5" / "bbox.json"):
        if cand.exists():
            b = json.loads(cand.read_text())
            return ((b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"]),
                    str(cand))
    return FALLBACK_BBOX, "built-in fallback (no bbox.json found)"


def crop_indices(ds: xr.Dataset, bbox, pad: float = BBOX_PAD) -> tuple[int, ...]:
    """(y0, y1, x0, x1) index window containing the box.

    lat/lon are 2D over projected x/y, so `.sel(lat=...)` does nothing useful.
    The window is the bounding index range of every cell inside the box, which
    over-covers slightly at the corners -- correct, since a tight mask would not
    be rectangular and could not be stored as an array.
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    inside = ((ds.lat >= lat_min - pad) & (ds.lat <= lat_max + pad)
              & (ds.lon >= lon_min - pad) & (ds.lon <= lon_max + pad)).compute()
    if not bool(inside.any()):
        raise RuntimeError(f"no CONUS404 cells inside {bbox}")
    ys = np.where(inside.any("x"))[0]
    xs = np.where(inside.any("y"))[0]
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def time_chunk(ds: xr.Dataset, var: str, y: int, x: int,
               target_mb: int) -> int:
    """Steps per written chunk: a byte budget, rounded to the source chunking.

    Two zarr constraints make chunking mandatory rather than optional -- chunks
    must be uniform except the last, and no buffer may exceed 2 GiB -- so the
    size comes from a byte budget that holds on any grid or dtype.

    Rounding to a MULTIPLE of the source time chunk matters here specifically:
    an unaligned write chunk straddles source chunks and turns the write into a
    dask shuffle, which on a 180 GB read is the difference between working and
    exhausting memory.
    """
    per_step = ds[var].dtype.itemsize * y * x
    want = max(1, int(target_mb * 1024 ** 2) // max(per_step, 1))
    src = ds[var].chunksizes.get("time", (want,))[0]
    n = max(src, (want // src) * src) if src else want
    print(f"  chunk: {n} steps = {n * per_step / 1024 ** 2:.0f} MB "
          f"(source time chunk {src})")
    return int(n)


def write_static(ds: xr.Dataset, sl: dict, path: Path, dry_run: bool,
                 overwrite: bool) -> None:
    """2D fields plus the lat/lon of the cropped window."""
    have = [v for v in STATIC if v in ds]
    sub = ds[have].isel(**sl) if have else None
    if sub is None:
        print("  no static fields present")
        return
    sub = sub.reset_coords(drop=True).assign_coords(
        lat=ds.lat.isel(**sl), lon=ds.lon.isel(**sl))
    mb = sum(sub[v].nbytes for v in sub.data_vars) / 1024 ** 2
    print(f"  static: {have}, ~{mb:.1f} MB")
    if dry_run:
        print(f"  would write {path}")
        return
    # if path.exists() and overwrite:
    #     shutil.rmtree(path)
    if path.exists():
        print(f"  static exists, skipping {path.name}")
        return
    for v in sub.variables:
        sub[v].encoding = {}
    sub.load().to_zarr(path, mode="w", consolidated=True,
                       zarr_format=ZARR_FORMAT)
    print(f"  wrote {path.name}")


def existing_state(path: Path) -> tuple[int, pd.Timestamp, dict, list[str]]:
    """(steps written, last timestamp, stored crop attrs, variables)."""
    ds = xr.open_zarr(path, consolidated=True)
    attrs = {k: ds.attrs.get(k) for k in ("y0", "y1", "x0", "x1")}
    return (int(ds.sizes["time"]), pd.Timestamp(ds.time.values[-1]),
            attrs, list(ds.data_vars))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variables", nargs="*", help="e.g. U10 V10 T2")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--name", default="conus404_hourly.zarr")
    ap.add_argument("--product", default="hourly",
                    choices=["hourly", "daily", "monthly"])
    ap.add_argument("--time-slice", nargs=2, default=list(TIME_SLICE),
                    metavar=("START", "END"))
    ap.add_argument("--bbox", nargs=4, default=None, type=float,
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    ap.add_argument("--pad", type=float, default=BBOX_PAD)
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete and rebuild rather than resume")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=TARGET_CHUNK_MB)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    dask.config.set(scheduler="threads", num_workers=a.workers)
    out_dir = Path(a.root) / "conus404"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / a.name
    t0, t1 = pd.Timestamp(a.time_slice[0]), pd.Timestamp(a.time_slice[1])

    print(f"opening CONUS404 {a.product} from {ENDPOINT}")
    src = open_conus404(a.product)

    if a.list:
        print(f"\n{len(src.data_vars)} variables:")
        for v in sorted(src.data_vars):
            print(f"  {v:<14} {src[v].dims}  "
                  f"{src[v].attrs.get('long_name', '')[:50]}")
        print(f"\ndims: {dict(src.sizes)}")
        print(f"time: {str(src.time.values[0])[:13]} -> "
              f"{str(src.time.values[-1])[:13]}")
        return 0

    if not a.variables:
        ap.error("name at least one variable (or use --list)")
    missing = [v for v in a.variables if v not in src.data_vars]
    if missing:
        ap.error(f"not in the store: {missing}. Use --list to see what is.")

    bbox, origin = resolve_bbox(Path(a.root), a.bbox)
    print(f"bbox from {origin}: {bbox[0]:.2f}..{bbox[1]:.2f} N, "
          f"{bbox[2]:.2f}..{bbox[3]:.2f} E  (pad {a.pad})")
    y0, y1, x0, x1 = crop_indices(src, bbox, a.pad)
    sl = {"y": slice(y0, y1), "x": slice(x0, x1)}
    ny, nx = y1 - y0, x1 - x0
    print(f"  window y[{y0}:{y1}] x[{x0}:{x1}] -> {ny} x {nx} cells")

    if not a.no_static:
        write_static(src, sl, out_dir / "conus404_static.zarr",
                     a.dry_run, a.overwrite)

    sub = src[list(a.variables)].isel(**sl).sel(time=slice(t0, t1))
    n = sub.sizes["time"]
    if n == 0:
        raise RuntimeError(f"no CONUS404 steps in {t0.date()}..{t1.date()}")
    size = sum(sub[v].nbytes for v in sub.data_vars) / 1024 ** 3
    free = shutil.disk_usage(out_dir).free / 1024 ** 3
    print(f"\n  {n:,} steps {str(sub.time.values[0])[:13]} -> "
          f"{str(sub.time.values[-1])[:13]}")
    print(f"  {list(sub.data_vars)}  ~{size:.1f} GB uncompressed "
          f"({free:.0f} GB free)")
    if size > free:
        print("  WARNING: estimate exceeds free space even before compression")

    if path.exists() and a.overwrite and not a.dry_run:
        shutil.rmtree(path)

    nsteps = time_chunk(sub, a.variables[0], ny, nx, a.chunk_mb)
    # Block length: about a year, rounded UP to a whole number of chunks so
    # every append starts on a chunk boundary.
    block = int(np.ceil(8760 / nsteps)) * nsteps
    print(f"  block: {block} steps ({block / 8760:.2f} yr, "
          f"{block // nsteps} chunks)")

    start = 0
    if path.exists():
        written, last, stored, have = existing_state(path)
        if [stored[k] for k in ("y0", "y1", "x0", "x1")] != [y0, y1, x0, x1]:
            raise RuntimeError(
                f"{path.name} was built for window {stored}, this run computes "
                f"y[{y0}:{y1}] x[{x0}:{x1}]. Different bounding box -- move it "
                "aside or use --overwrite.")
        if sorted(have) != sorted(a.variables):
            raise RuntimeError(
                f"{path.name} holds {have}, this run asks for {a.variables}. "
                "Appending along time cannot add variables; use --overwrite.")
        if written % nsteps:
            # A store whose length is not a whole number of chunks has a short
            # final chunk, and nothing can be appended to it cleanly. Stores
            # written by an earlier calendar-year version look like this.
            raise RuntimeError(
                f"{path.name} holds {written:,} steps, not a multiple of the "
                f"{nsteps}-step chunk. Its last chunk is short, so no append "
                "can align. Rebuild with --overwrite.")
        if not np.array_equal(
                xr.open_zarr(path, consolidated=True).time.values,
                sub.time.values[:written]):
            raise RuntimeError(
                f"{path.name} timestamps do not match the first {written:,} "
                "steps of this crop. Different window -- use --overwrite.")
        start = written
        print(f"\n  resuming: {written:,} steps present, ends {last}")
        if start >= n:
            print("  already complete")
            return 0

    blocks = [(i, min(i + block, n)) for i in range(start, n, block)]
    print(f"  {len(blocks)} block(s) to write, steps {start:,}..{n:,}")

    if a.dry_run:
        print(f"\nwould write {path} in {len(blocks)} appends")
        return 0

    attrs = {"source": f"CONUS404 {a.product} via {ENDPOINT}",
             "y0": y0, "y1": y1, "x0": x0, "x1": x1,
             "lat_min": bbox[0], "lat_max": bbox[1],
             "lon_min": bbox[2], "lon_max": bbox[3], "pad_deg": a.pad,
             "note": "U10/V10 are grid-relative; magnitude is invariant under "
                     "the rotation, direction is not"}

    from dask.diagnostics import ProgressBar
    started = time.time()
    for k, (lo, hi) in enumerate(blocks):
        piece = sub.isel(time=slice(lo, hi)).chunk(
            {"time": nsteps, "y": -1, "x": -1})
        for v in piece.variables:
            piece[v].encoding = {}
        gb = sum(piece[v].nbytes for v in piece.data_vars) / 1024 ** 3
        first = (lo == 0)
        stamp = str(piece.time.values[0])[:10]

        print(f"  [{k + 1}/{len(blocks)}] steps {lo:,}..{hi:,} from {stamp}, "
              f"{gb:.1f} GB ({'create' if first else 'append'})")
        t = time.time()
        with ProgressBar(minimum=5.0):
            if first:
                piece.assign_attrs(attrs).to_zarr(
                    path, mode="w", consolidated=True, zarr_format=ZARR_FORMAT)
            else:
                piece.to_zarr(path, mode="a", append_dim="time",
                              consolidated=True, zarr_format=ZARR_FORMAT)
        dt = time.time() - t
        done = hi - start
        rate = done / max(time.time() - started, 1)
        eta = (n - hi) / rate / 3600 if rate else 0
        print(f"       {dt / 60:.1f} min ({gb / max(dt, 1) * 1024:.0f} MB/s), "
              f"ETA {eta:.1f} h")

    (out_dir / "bbox.json").write_text(json.dumps(
        {"lat_min": bbox[0], "lat_max": bbox[1], "lon_min": bbox[2],
         "lon_max": bbox[3], "pad_deg": a.pad, "origin": origin,
         "y0": y0, "y1": y1, "x0": x0, "x1": x1,
         "time_start": str(t0.date()), "time_end": str(t1.date())}, indent=1))

    final = xr.open_zarr(path, consolidated=True)
    on_disk = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"\nwrote {path} in {(time.time() - started) / 3600:.1f} h")
    print(f"  {dict(final.sizes)}, {list(final.data_vars)}")
    print(f"  {str(final.time.values[0])[:13]} -> {str(final.time.values[-1])[:13]}")
    print(f"  on disk {on_disk / 1024 ** 3:.1f} GB "
          f"({size / max(on_disk / 1024 ** 3, 1e-9):.1f}x compression)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)