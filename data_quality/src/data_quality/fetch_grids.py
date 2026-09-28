#!/usr/bin/env python
"""
Prefetch the reference datasets into /shared/data/grids, so no analysis stage
touches Arraylake or the USGS pod again.

    python fetch_grids.py --dry-run                  # sizes and the crop, no reads
    python fetch_grids.py --only era5                # the cheap, important one
    python fetch_grids.py --only conus404
    python fetch_grids.py                            # everything

Outputs:

    /shared/data/grids/
        era5_hourly.zarr        u10, v10, hourly, cropped to the product bbox
        wrf_elevation.nc        d03 terrain, for sub-grid relief
        conus404_mon.zarr       10 m scalar speed, formed hourly then averaged
        conus404_day.zarr       daily mean and daily max of the same
        conus404_static.zarr    HGT, LANDMASK, VAR_SSO, lat, lon
        bbox.json               the crop actually used, and what defined it

WHY ERA5 IS STORED RAW AND CONUS404 IS NOT.

ERA5 cropped to California is small -- roughly 51 x 49 cells at 0.25 deg, about
6 GB for 35 years of hourly u10/v10. Keeping the hourly components means every
aggregation the analysis needs can be recomputed locally at disk speed: monthly
and daily scalar means, daily maxima for the extremes stage, and the
vector-vs-scalar averaging test, which needs the components rather than a speed.

CONUS404 hourly over the same box is order 180 GB, because its store is chunked
as spatial tiles rather than time runs. So it is reduced ON THE WAY IN: one pass
over the hourly data produces the monthly mean, daily mean and daily max in a
single dask graph, and only those are written. `--conus404-raw` overrides this
if the components are ever genuinely needed.

THE GRID FOLDER IS VARIABLE-AGNOSTIC. d03 and the LOCA2 grid are fixed, so one
ERA5 crop and one terrain file serve every variable. Nothing here is written per
variable, and nothing needs refetching when a second variable is downloaded.

THE CROP COMES FROM THE DATA, ONCE. The bounding box is the union of the
footprints of every product store under --root, padded, using the same
crop_era5_to_source the analysis uses -- so the ERA5 grid here is identical to
the one the metrics are computed on, rather than merely similar. It is then
FROZEN in bbox.json and reused. That matters: if a later variable were published
on a marginally different footprint, recomputing the union would silently crop
ERA5 differently and metrics computed before and after would no longer be
comparable. Use --rebuild-bbox to recompute deliberately, which invalidates
every grid file already written.
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

from .ae_era5_comparison import (
    load_era5, load_wrf_elevation, load_conus404,
    crop_era5_to_source, get_latlon, BBOX_PAD,
)

DEFAULT_ROOT = "/shared/data"
ZARR_FORMAT = 2
TARGET_CHUNK_MB = 128

# Match fetch_local.py. CONUS404 begins 1979-10; ERA5 covers far more.
TIME_SLICE = ("1980-01-01", "2014-12-31")

WRF_ELEVATION_URL = "s3://cadcat/wrf/derived-vars/elevation_wrf.nc"

TARGETS = ("era5", "elevation", "conus404")


# ----------------------------------------------------------------------------
# Footprint
# ----------------------------------------------------------------------------


def product_bbox(root: Path, variable: str | None = None
                 ) -> tuple[tuple[float, ...], list[str]]:
    """Union bounding box of every local product store under `root`.

    Products cover different areas -- WRF fills its rotated d03 domain including
    ocean, LOCA2 is land-masked -- so the union is what ERA5 must span for any
    of them to be comparable against it.

    Every variable folder is scanned unless one is named. The grids are the same
    for all variables, so more stores only makes the union better constrained;
    it does not make the answer variable-specific.
    """
    if variable:
        dirs = [root / variable]
    else:
        dirs = sorted(d for d in root.iterdir()
                      if d.is_dir() and d.name != "grids")
    stores = sorted(p for d in dirs for p in d.glob("*_*.zarr"))
    if not stores:
        raise FileNotFoundError(
            f"no product stores under {root}. Run fetch_local.py first.")

    lat_min = lon_min = np.inf
    lat_max = lon_max = -np.inf
    used = []
    for p in stores:
        ds = xr.open_zarr(p, consolidated=True)
        try:
            lat, lon = get_latlon(ds)
        except Exception:
            print(f"  {p.name}: no lat/lon found, skipping")
            continue
        lat_min = min(lat_min, float(lat.min()))
        lat_max = max(lat_max, float(lat.max()))
        lon_min = min(lon_min, float(lon.min()))
        lon_max = max(lon_max, float(lon.max()))
        used.append(f"{p.parent.name}/{p.name}")
        print(f"  {p.parent.name}/{p.name:<24} "
              f"{float(lat.min()):.2f}..{float(lat.max()):.2f} N, "
              f"{float(lon.min()):.2f}..{float(lon.max()):.2f} E")

    if not used:
        raise RuntimeError("no store yielded lat/lon coordinates")
    print(f"\n  union: {lat_min:.2f}..{lat_max:.2f} N, "
          f"{lon_min:.2f}..{lon_max:.2f} E  (pad {BBOX_PAD} deg applied downstream)")
    return (lat_min, lat_max, lon_min, lon_max), used


def bbox_as_dataset(bbox) -> xr.Dataset:
    """A two-point Dataset carrying the bbox corners.

    crop_era5_to_source only reads min/max of the source lat/lon, so handing it
    the corners gives exactly the crop the analysis would compute -- including
    the pad, the 0-360 handling and the descending-latitude fix -- without
    duplicating any of that logic here.
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    return xr.Dataset(coords={"lat": ("lat", [lat_min, lat_max]),
                              "lon": ("lon", [lon_min, lon_max])})


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------


def rechunk(ds: xr.Dataset, target_mb: int = TARGET_CHUNK_MB) -> xr.Dataset:
    """Time-contiguous chunks sized by bytes, not by a step count.

    Every downstream metric reduces over time, so time is the dimension to keep
    whole per chunk. The budget is in bytes because a plausible-looking step
    count silently becomes gigabytes on a finer grid, and zarr v2 codecs refuse
    any buffer over 2 GiB.
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
    n_steps = max(1, int(target_mb * 1024 ** 2) // max(per_step, 1))
    n_steps = min(int(ds.sizes["time"]), n_steps)
    print(f"    chunk: {n_steps} steps = "
          f"{n_steps * per_step / 1024 ** 2:.0f} MB")
    return ds.chunk({"time": n_steps}
                    | {d: -1 for d in ds.dims if d != "time"})


def write(ds: xr.Dataset, path: Path, attrs: dict, dry_run: bool,
          target_mb: int = TARGET_CHUNK_MB, overwrite: bool = False) -> dict:
    """Write one dataset to zarr, atomically."""
    ds = rechunk(ds, target_mb)
    size = sum(ds[v].nbytes for v in ds.data_vars) / 1024 ** 3
    info = {"path": str(path), "gb_uncompressed": round(size, 2),
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "vars": list(ds.data_vars)}
    print(f"    {dict(ds.sizes)}, {list(ds.data_vars)}, ~{size:.2f} GB")

    if dry_run:
        print(f"    would write {path}")
        return info
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists():
        print(f"    exists, skipping {path.name} (--overwrite to rebuild)")
        info["skipped"] = True
        return info

    # Strip inherited encoding: netCDF compressors and fill values conflict with
    # the zarr writer, and numpy 2 StringDType is not writable at all.
    ds = ds.copy()
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
        # the next run's disk estimate wrong.
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.rename(path)
    dt = time.time() - t
    print(f"    wrote {path.name} in {dt:.0f}s "
          f"({size / max(dt, 1) * 1024:.0f} MB/s)")
    info["seconds"] = round(dt, 1)
    return info


# ----------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------


def do_era5(out: Path, bbox, tslice, dry_run, overwrite, login) -> list[dict]:
    """Hourly u10/v10 cropped to the product footprint.

    Read from the `temporal` group, NOT `spatial`. The spatial layout stores one
    global map per hour and is roughly 40x slower for a regional multi-decade
    query.
    """
    ds = load_era5(login=login)[["u10", "v10"]]
    crop = crop_era5_to_source(ds, bbox_as_dataset(bbox))
    crop = crop.sel(time=slice(*tslice))
    print(f"    {crop.sizes['time']:,} hourly steps "
          f"{str(crop.time.values[0])[:10]} -> {str(crop.time.values[-1])[:10]}")
    return [write(crop, out / "era5_hourly.zarr",
                  {"source": "ERA5 surface via Arraylake, temporal group",
                   "time_start": tslice[0], "time_end": tslice[1],
                   "note": "components kept raw so any aggregation can be "
                           "recomputed locally, including daily maxima and "
                           "the vector-vs-scalar averaging test"},
                  dry_run, overwrite=overwrite)]


def do_elevation(out: Path, dry_run, overwrite) -> list[dict]:
    """WRF d03 terrain. A single small netCDF; copied rather than converted."""
    path = out / "wrf_elevation.nc"
    if path.exists() and not overwrite:
        print(f"    exists, skipping {path.name}")
        return []
    elev = load_wrf_elevation()
    size = sum(elev[v].nbytes for v in elev.data_vars) / 1024 ** 2
    print(f"    {dict(elev.sizes)}, {list(elev.data_vars)}, ~{size:.1f} MB")
    if dry_run:
        print(f"    would write {path}")
        return []
    for v in elev.variables:
        elev[v].encoding = {}
    elev.load().to_netcdf(path)
    print(f"    wrote {path.name}")
    return [{"path": str(path), "gb_uncompressed": round(size / 1024, 3)}]


def do_conus404(out: Path, bbox, tslice, dry_run, overwrite,
                raw: bool) -> list[dict]:
    """CONUS404, reduced on the way in unless --conus404-raw.

    Speed is formed HOURLY and averaged afterwards. CONUS404 ships no wind-speed
    variable, and its monthly product holds mean *components*, whose magnitude
    carries a ~20% vector-averaging bias. Components are grid-relative, which is
    irrelevant for speed since magnitude is invariant under that rotation.
    """
    ds = load_conus404("hourly")
    lat_min, lat_max, lon_min, lon_max = bbox

    inside = ((ds.lat >= lat_min - BBOX_PAD) & (ds.lat <= lat_max + BBOX_PAD)
              & (ds.lon >= lon_min - BBOX_PAD)
              & (ds.lon <= lon_max + BBOX_PAD)).compute()
    ys = np.where(inside.any("x"))[0]
    xs = np.where(inside.any("y"))[0]
    ds = ds.isel(y=slice(ys.min(), ys.max() + 1),
                 x=slice(xs.min(), xs.max() + 1))
    print(f"    subset to y={ds.sizes['y']}, x={ds.sizes['x']}")

    written = []

    # Static fields first: 2D, cost nothing, and needed for terrain work.
    static = ds[[v for v in ("HGT", "LANDMASK", "VAR_SSO") if v in ds]]
    if len(static.data_vars):
        written += [write(static.reset_coords(drop=True).assign_coords(
                              lat=ds.lat, lon=ds.lon),
                          out / "conus404_static.zarr",
                          {"source": "CONUS404 hourly store, static fields"},
                          dry_run, overwrite=overwrite)]

    sub = ds[["U10", "V10"]].sel(time=slice(*tslice))
    n = sub.sizes["time"]
    gb = n * sub.sizes["y"] * sub.sizes["x"] * 4 * 2 / 1024 ** 3
    print(f"    {n:,} hourly steps, ~{gb:.0f} GB to read")

    if raw:
        return written + [write(
            sub, out / "conus404_hourly.zarr",
            {"source": "CONUS404 hourly U10/V10, grid-relative components",
             "time_start": tslice[0], "time_end": tslice[1]},
            dry_run, overwrite=overwrite)]

    w = np.sqrt(sub.U10 ** 2 + sub.V10 ** 2)
    w.name = "wspeed"
    w.attrs.update(units="m s-1",
                   long_name="10 m scalar wind speed, formed hourly")

    mon = w.resample(time="MS").mean().to_dataset(name="wspeed")
    day = xr.Dataset({"wspeed": w.resample(time="1D").mean(),
                      "wspeed_max": w.resample(time="1D").max()})
    for d in (mon, day):
        d["lat"], d["lon"] = ds.lat, ds.lon

    print(f"    reducing in one pass (~{gb:.0f} GB read, "
          f"{(mon.wspeed.nbytes + day.wspeed.nbytes * 2) / 1024 ** 3:.1f} GB written)")
    if dry_run:
        for name, d in (("conus404_mon.zarr", mon), ("conus404_day.zarr", day)):
            written.append(write(d, out / name, {}, True))
        return written

    # ONE dask.compute for both cadences: the hourly read is the entire cost,
    # and computing them separately would pay it twice.
    from dask.diagnostics import ProgressBar
    with ProgressBar(minimum=2.0):
        mon_c, day_c = dask.compute(mon, day)

    attrs = {"source": "CONUS404 hourly, speed formed hourly then aggregated",
             "time_start": tslice[0], "time_end": tslice[1]}
    written.append(write(mon_c, out / "conus404_mon.zarr", attrs, False,
                         overwrite=overwrite))
    written.append(write(day_c, out / "conus404_day.zarr", attrs, False,
                         overwrite=overwrite))
    return written


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--variable", default=None,
                    help="restrict the footprint scan to one variable folder "
                         "(default: every variable under --root)")
    ap.add_argument("--rebuild-bbox", action="store_true",
                    help="recompute the frozen bounding box. Invalidates every "
                         "grid file already written -- use --overwrite too.")
    ap.add_argument("--only", nargs="*", default=list(TARGETS),
                    choices=list(TARGETS))
    ap.add_argument("--time-slice", nargs=2, default=list(TIME_SLICE),
                    metavar=("START", "END"))
    ap.add_argument("--conus404-raw", action="store_true",
                    help="store hourly components (~180 GB) instead of "
                         "reducing on the way in")
    ap.add_argument("--era5-login", action="store_true",
                    help="run the Arraylake interactive login first")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=TARGET_CHUNK_MB)
    a = ap.parse_args()

    dask.config.set(scheduler="threads", num_workers=a.workers)

    root = Path(a.root)
    out = root / "grids"
    out.mkdir(parents=True, exist_ok=True)
    tslice = (a.time_slice[0], a.time_slice[1])

    print(f"destination: {out}")
    print(f"time slice : {tslice[0]} -> {tslice[1]}")
    print(f"targets    : {a.only}")
    print(f"mode       : {'DRY RUN' if a.dry_run else 'write'}"
          f"{' (overwrite)' if a.overwrite else ''}")
    print(f"free space : {shutil.disk_usage(out).free / 1024 ** 3:.0f} GB\n")

    bpath = out / "bbox.json"
    if bpath.exists() and not a.rebuild_bbox:
        b = json.loads(bpath.read_text())
        bbox = (b["lat_min"], b["lat_max"], b["lon_min"], b["lon_max"])
        print(f"bounding box: frozen in {bpath.name} "
              f"({bbox[0]:.2f}..{bbox[1]:.2f} N, {bbox[2]:.2f}..{bbox[3]:.2f} E)"
              f" from {len(b.get('from_stores', []))} store(s)")
        print("  --rebuild-bbox to recompute (invalidates existing grid files)")
    else:
        print(f"bounding box from {root}"
              f"{'/' + a.variable if a.variable else ''}:")
        bbox, used = product_bbox(root, a.variable)
        if not a.dry_run:
            bpath.write_text(json.dumps(
                {"lat_min": bbox[0], "lat_max": bbox[1],
                 "lon_min": bbox[2], "lon_max": bbox[3],
                 "pad_deg": BBOX_PAD, "from_stores": used,
                 "frozen": True}, indent=1))
            print(f"  frozen in {bpath}")
    print()

    manifest, failures = [], []
    jobs = [
        ("era5", lambda: do_era5(out, bbox, tslice, a.dry_run, a.overwrite,
                                 a.era5_login)),
        ("elevation", lambda: do_elevation(out, a.dry_run, a.overwrite)),
        ("conus404", lambda: do_conus404(out, bbox, tslice, a.dry_run,
                                         a.overwrite, a.conus404_raw)),
    ]
    for name, fn in jobs:
        if name not in a.only:
            continue
        print(f"[{name}]")
        try:
            manifest += fn() or []
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
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
