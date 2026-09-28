#!/usr/bin/env python
"""
Establish the project geometry: the bounding box every other fetcher crops to,
and the WRF terrain field the analysis needs.

    python fetch_geometry.py --dry-run          # show the box, write nothing
    python fetch_geometry.py                    # freeze the box, fetch terrain
    python fetch_geometry.py --rebuild-bbox --overwrite

Output:

    /shared/data/grids/bbox.json            the crop, frozen
    /shared/data/grids/wrf_elevation.nc     d03 terrain, for sub-grid relief

Run AFTER fetch_local.py (which supplies the footprint) and BEFORE fetch_era5.py
and fetch_conus404.py (which read the box). Without bbox.json those two fall
through to their own hardcoded fallback rectangle, which has never been checked
against the real d03 footprint.

VARIABLE-AGNOSTIC. d03 and the LOCA2 grid are fixed, so one box and one terrain
file serve every variable. Nothing here is written per variable, and nothing
needs refetching when a second variable is downloaded.

THE BOX IS THE UNION OF EVERY PRODUCT FOOTPRINT. Products cover different areas
-- WRF fills its rotated d03 domain including ocean, LOCA2 is land-masked -- so
cropping to either alone would clip the other.

AND IT IS FROZEN. Once written it is reused rather than recomputed. If a later
variable arrived on a marginally different footprint, silently recomputing the
union would crop ERA5 to a different set of cells, and metrics from before and
after would look comparable while not being. --rebuild-bbox recomputes
deliberately, which invalidates every grid, ERA5 and CONUS404 store already
written.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_ROOT = "/shared/data"
BBOX_PAD = 0.25
WRF_ELEVATION_URL = "s3://cadcat/wrf/derived-vars/elevation_wrf.nc"

# Folders written by the other fetchers, not by fetch_local.py.
NOT_VARIABLES = ("grids", "era5", "conus404", "hdp")

LAT_NAMES = (("lat", "lon"), ("latitude", "longitude"),
             ("XLAT", "XLONG"), ("south_north", "west_east"))


def get_latlon(ds) -> tuple[xr.DataArray, xr.DataArray]:
    """lat/lon coords whatever they are called. Works for 1D and 2D alike."""
    for la, lo in LAT_NAMES:
        if la in ds.coords and lo in ds.coords:
            return ds[la], ds[lo]
    raise KeyError(f"no lat/lon coords in {list(ds.coords)}")


def product_bbox(root: Path, variable: str | None = None
                 ) -> tuple[tuple[float, ...], list[str]]:
    """Union bounding box of every local product store under `root`.

    Every variable folder is scanned unless one is named. The grids are the same
    for all variables, so more stores only constrains the union better; it does
    not make the answer variable-specific.
    """
    if variable:
        dirs = [root / variable]
    else:
        dirs = sorted(d for d in root.iterdir()
                      if d.is_dir() and d.name not in NOT_VARIABLES)
    stores = sorted(p for d in dirs for p in d.glob("*_*.zarr"))
    if not stores:
        raise FileNotFoundError(
            f"no product stores under {root}. Run fetch_local.py first.")

    lat_min = lon_min = np.inf
    lat_max = lon_max = -np.inf
    used = []
    for p in stores:
        try:
            lat, lon = get_latlon(xr.open_zarr(p, consolidated=True))
        except Exception as e:
            print(f"  {p.parent.name}/{p.name}: skipped ({type(e).__name__})")
            continue
        lat_min, lat_max = min(lat_min, float(lat.min())), max(lat_max, float(lat.max()))
        lon_min, lon_max = min(lon_min, float(lon.min())), max(lon_max, float(lon.max()))
        used.append(f"{p.parent.name}/{p.name}")
        print(f"  {p.parent.name}/{p.name:<26} "
              f"{float(lat.min()):6.2f}..{float(lat.max()):6.2f} N, "
              f"{float(lon.min()):8.2f}..{float(lon.max()):8.2f} E")

    if not used:
        raise RuntimeError("no store yielded lat/lon coordinates")
    print(f"\n  union: {lat_min:.2f}..{lat_max:.2f} N, "
          f"{lon_min:.2f}..{lon_max:.2f} E   "
          f"(consumers apply a {BBOX_PAD} deg pad)")
    return (lat_min, lat_max, lon_min, lon_max), used


def fetch_elevation(path: Path, url: str, dry_run: bool,
                    overwrite: bool) -> None:
    """WRF d03 terrain. One small netCDF, copied rather than converted.

    Needed for elevation_on_era5, whose sub-grid relief -- the spread of 3 km
    terrain inside each ERA5 cell -- measures what a 31 km reanalysis cannot
    resolve, and predicts the discrepancy far better than mean height does.
    """
    if path.exists() and not overwrite:
        print(f"  exists, skipping {path.name}")
        return
    print(f"  reading {url}")
    ds = xr.open_dataset(url, engine="h5netcdf" if url.endswith(".nc") else None)
    mb = sum(ds[v].nbytes for v in ds.data_vars) / 1024 ** 2
    print(f"  {dict(ds.sizes)}, {list(ds.data_vars)}, ~{mb:.1f} MB")
    if dry_run:
        print(f"  would write {path}")
        return
    for v in ds.variables:
        ds[v].encoding = {}
    ds.load().to_netcdf(path)
    print(f"  wrote {path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--variable", default=None,
                    help="restrict the footprint scan to one variable folder "
                         "(default: every variable under --root)")
    ap.add_argument("--elevation-url", default=WRF_ELEVATION_URL)
    ap.add_argument("--no-elevation", action="store_true")
    ap.add_argument("--rebuild-bbox", action="store_true",
                    help="recompute the frozen box. Invalidates every grid, "
                         "ERA5 and CONUS404 store already written.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    root = Path(a.root)
    out = root / "grids"
    out.mkdir(parents=True, exist_ok=True)
    bpath = out / "bbox.json"

    print(f"destination: {out}")
    print(f"mode       : {'DRY RUN' if a.dry_run else 'write'}"
          f"{' (overwrite)' if a.overwrite else ''}\n")

    if bpath.exists() and not a.rebuild_bbox:
        b = json.loads(bpath.read_text())
        print(f"bounding box: frozen in {bpath.name}")
        print(f"  {b['lat_min']:.2f}..{b['lat_max']:.2f} N, "
              f"{b['lon_min']:.2f}..{b['lon_max']:.2f} E "
              f"from {len(b.get('from_stores', []))} store(s)")
        print("  --rebuild-bbox to recompute (invalidates downstream stores)")
    else:
        print(f"bounding box from {root}"
              f"{'/' + a.variable if a.variable else ''}:")
        bbox, used = product_bbox(root, a.variable)
        if a.dry_run:
            print(f"\n  would freeze in {bpath}")
        else:
            bpath.write_text(json.dumps(
                {"lat_min": bbox[0], "lat_max": bbox[1],
                 "lon_min": bbox[2], "lon_max": bbox[3],
                 "pad_deg": BBOX_PAD, "from_stores": used,
                 "frozen": True}, indent=1))
            print(f"\n  frozen in {bpath}")

    if not a.no_elevation:
        print("\nWRF terrain:")
        fetch_elevation(out / "wrf_elevation.nc", a.elevation_url,
                        a.dry_run, a.overwrite)

    print("\nnext:  fetch_era5.py u10 v10   |   fetch_conus404.py U10 V10")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
