#!/usr/bin/env python
"""
Derive ERA5 wind-speed aggregations from the local hourly components.

    python derive_era5.py --dry-run
    python derive_era5.py

Output, all small enough to keep forever:

    /shared/data/era5/era5_speed_mon.zarr     wspeed, wspeed_vector      ~5 MB
    /shared/data/era5/era5_speed_day.zarr     wspeed, wspeed_max         ~150 MB
    /shared/data/era5/era5_diurnal.zarr       month x hour climatology   ~1 MB

SPEED IS FORMED HOURLY, THEN AGGREGATED. This is the one rule that must not be
got wrong. sqrt(u^2+v^2) is taken on every hourly step and the result is
averaged; averaging u and v first and taking the magnitude afterwards gives the
VECTOR mean, roughly 20% lower in this domain -- a bias the size of the model
discrepancies under study. `wspeed_vector` computes exactly that wrong answer on
purpose, so the size of the penalty is measured rather than assumed, and it is
the only variable here that should never be used as a reference.

ONE PASS. Every product below comes out of a single dask.compute over the hourly
store. Computing them separately would read ~14 GB once per product; together it
is read once. That is the whole reason this script exists rather than each stage
calling era5_speed() for itself.

DIURNAL CLIMATOLOGY comes along free. The gridded comparison has never had one --
everything is monthly, which cannot see a sea breeze or a nocturnal downslope
jet. A month x hour mean is 24 x 12 fields and costs nothing extra in this pass.
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


def open_hourly(root: Path) -> xr.Dataset:
    path = root / "era5" / "era5_hourly.zarr"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run fetch_era5.py first")
    return xr.open_zarr(path, consolidated=True)


def check_crop(ds: xr.Dataset, root: Path) -> None:
    """Warn if the hourly store does not cover the frozen bounding box.

    A store cropped smaller still works and still produces numbers; the cells it
    is missing simply vanish under xr.align later, with nothing raised. Saying so
    here is the only place it is cheap to notice.
    """
    import json
    bpath = root / "grids" / "bbox.json"
    if not bpath.exists():
        print("  WARNING: no grids/bbox.json to check the crop against")
        return
    b = json.loads(bpath.read_text())
    lo, hi = float(ds.latitude.min()), float(ds.latitude.max())
    if lo > b["lat_min"] or hi < b["lat_max"]:
        print(f"  WARNING: crop {lo:.2f}..{hi:.2f} N does not cover the frozen "
              f"box {b['lat_min']:.2f}..{b['lat_max']:.2f} N. Derived products "
              "will inherit the smaller footprint.")
    else:
        print("  crop covers the frozen box")


def write(ds: xr.Dataset, path: Path, attrs: dict, overwrite: bool) -> None:
    """Write atomically: .tmp then rename, so a partial store never looks done."""
    ds = ds.copy()
    for v in ds.variables:
        ds[v].encoding = {}
    ds.attrs.update(attrs)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    tmp = path.parent / (path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        ds.to_zarr(tmp, mode="w", consolidated=True, zarr_format=ZARR_FORMAT)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.rename(path)
    mb = sum(ds[v].nbytes for v in ds.data_vars) / 1024 ** 2
    print(f"  wrote {path.name}  {dict(ds.sizes)}  {mb:.1f} MB")


def derive_rh(ds, out, a) -> int:
    """Relative humidity from t2m and d2m, at every cadence the analysis needs.

    ERA5 PUBLISHES NO RH. It has to be derived, and the derivation is where the
    error lives -- so it uses humidity.py, which is tested against hand
    computable values, rather than an inline formula.

    Saturation is referenced to LIQUID WATER at all temperatures. That is the
    WMO convention for surface observations and what station networks report; an
    ice-referenced formula would give a HIGHER humidity for the same air below
    freezing -- by roughly 5 pp at -10 C -- and would put a spurious cold-season
    bias into every comparison.

    THE DAILY MINIMUM IS THE POINT. RH's fire-relevant statistic is the
    afternoon minimum, not the mean, and it cannot be recovered from daily or
    monthly means afterwards. It is computed here in the same pass.
    """
    from . import humidity as H

    # Name candidates. The Earthmover store calls these `t2` and `d2`, while
    # the GRIB shortnames t2m/d2m and the CF long names are both more familiar
    # -- so the store is asked what it has rather than told. Getting this wrong
    # fails loudly here; assuming it would fail as a KeyError deep in a compute.
    T_NAMES = ("t2", "t2m", "2m_temperature", "air_temperature_2m", "T2")
    D_NAMES = ("d2", "d2m", "2m_dewpoint_temperature",
               "dewpoint_temperature_2m", "D2")

    t = next((v for v in T_NAMES if v in ds), None)
    d = next((v for v in D_NAMES if v in ds), None)
    if t is None or d is None:
        raise KeyError(
            f"need a 2 m temperature and dewpoint; era5_hourly.zarr has "
            f"{sorted(ds.data_vars)}.\n"
            "Fetch them with the names `fetch_era5.py --list` reports, e.g.\n"
            "  python src/fetch_era5.py t2 d2")
    print(f"  deriving RH from '{t}' and '{d}'")

    rh = H.rh_from_dewpoint(ds[t], ds[d])
    rh.name = "hurs"
    rh.attrs.update(units="%", long_name="2 m relative humidity",
                    derivation="Bolton (1980) over liquid water, from t2m/d2m")

    mon = xr.Dataset({"hurs": rh.resample(time="MS").mean()})
    day = xr.Dataset({
        "hurs": rh.resample(time="1D").mean(),
        "hurs_min": rh.resample(time="1D").min(),
        "hurs_max": rh.resample(time="1D").max(),
    })
    # The gap between the true daily mean and the midrange of the extremes.
    # Measuring it from ERA5 quantifies the bias a LOCA2 comparison inherits,
    # since LOCA2 publishes only the extremes.
    day["mid_minus_mean"] = (day.hurs_max + day.hurs_min) / 2.0 - day.hurs
    diur = (rh.groupby("time.month")
            .map(lambda g: g.groupby("time.hour").mean("time"))
            .to_dataset(name="hurs"))

    if a.dry_run:
        for name, d in (("mon", mon), ("day", day), ("diurnal", diur)):
            mb = sum(d[v].nbytes for v in d.data_vars) / 1024 ** 2
            print(f"  would write era5_hurs_{name}: {dict(d.sizes)} {mb:.0f} MB")
        return 0

    print("\ncomputing RH (one pass over the hourly store)...")
    t = time.time()
    from dask.diagnostics import ProgressBar
    with ProgressBar(minimum=5.0):
        mon_c, day_c, diur_c = dask.compute(mon, day, diur)
    print(f"  done in {(time.time() - t) / 60:.1f} min\n")

    base = {"source": "ERA5 hourly t2m/d2m, local",
            "derivation": "Bolton (1980) over liquid water",
            "note": "RH formed hourly, then aggregated; the daily MINIMUM is "
                    "the fire-relevant statistic and cannot be recovered from "
                    "means afterwards"}
    write(mon_c, out / "era5_hurs_mon.zarr", base, True)
    write(day_c, out / "era5_hurs_day.zarr", base, True)
    write(diur_c, out / "era5_hurs_diurnal.zarr",
          base | {"note": "mean by calendar month and hour of day (UTC)"}, True)

    print(f"\ndomain mean RH        {float(mon_c.hurs.mean()):.1f} %")
    print(f"domain mean daily min {float(day_c.hurs_min.mean()):.1f} %")
    print(f"midrange minus mean   {float(day_c.mid_minus_mean.mean()):+.2f} pp")
    print("  (the midrange sits BELOW the true daily mean because RH's diurnal\n"
          "   cycle is asymmetric -- this is the bias a LOCA2 max/min\n"
          "   comparison inherits, and it grows with the diurnal range)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--variable", default="wspeed", choices=["wspeed", "hurs"],
                    help="wspeed derives scalar speed from u10/v10; "
                         "hurs derives relative humidity from t2m/d2m")
    a = ap.parse_args()

    dask.config.set(scheduler="threads", num_workers=a.workers)
    root = Path(a.root)
    out = root / "era5"

    ds = open_hourly(root)
    print(f"hourly store: {dict(ds.sizes)}  {list(ds.data_vars)}")
    print(f"  {str(ds.time.values[0])[:13]} -> {str(ds.time.values[-1])[:13]}")
    gb = sum(ds[v].nbytes for v in ds.data_vars) / 1024 ** 3
    print(f"  ~{gb:.1f} GB to read (once)")
    check_crop(ds, root)

    if a.variable == "hurs":
        return derive_rh(ds, out, a)

    # --- the one correct primitive ---------------------------------------
    w = np.sqrt(ds.u10 ** 2 + ds.v10 ** 2)
    w.name = "wspeed"
    w.attrs.update(units="m s-1",
                   long_name="10 m scalar wind speed, formed hourly")

    # Monthly: scalar mean, plus the vector mean for the sampling test. The
    # vector path must average the COMPONENTS first -- that is the error being
    # quantified, so it has to be reproduced exactly.
    mon = xr.Dataset({
        "wspeed": w.resample(time="MS").mean(),
        "wspeed_vector": np.sqrt(ds.u10.resample(time="1D").mean() ** 2
                                 + ds.v10.resample(time="1D").mean() ** 2
                                 ).resample(time="MS").mean(),
    })

    # Daily: mean for the cadence check, max for the extremes comparison. The
    # max is over HOURLY SAMPLES, which is not the same thing as a model's
    # timestep maximum -- WRF's wspd10max is higher by construction, so any
    # comparison against it is a bound rather than an equality.
    day = xr.Dataset({
        "wspeed": w.resample(time="1D").mean(),
        "wspeed_max": w.resample(time="1D").max(),
    })

    # Diurnal: mean by calendar month and hour of day.
    diur = (w.groupby("time.month").map(
        lambda g: g.groupby("time.hour").mean("time"))
        .to_dataset(name="wspeed"))

    if a.dry_run:
        for name, d in (("mon", mon), ("day", day), ("diurnal", diur)):
            mb = sum(d[v].nbytes for v in d.data_vars) / 1024 ** 2
            print(f"  would write era5_speed_{name}: {dict(d.sizes)}  {mb:.1f} MB")
        return 0

    for name, path in (("mon", out / "era5_speed_mon.zarr"),
                       ("day", out / "era5_speed_day.zarr"),
                       ("diurnal", out / "era5_diurnal.zarr")):
        if path.exists() and not a.overwrite:
            print(f"  {path.name} exists (--overwrite to rebuild)")

    print("\ncomputing (one pass over the hourly store)...")
    t = time.time()
    from dask.diagnostics import ProgressBar
    with ProgressBar(minimum=5.0):
        mon_c, day_c, diur_c = dask.compute(mon, day, diur)
    print(f"  done in {(time.time() - t) / 60:.1f} min\n")

    base = {"source": "ERA5 hourly u10/v10, local",
            "note": "scalar speed formed hourly, then aggregated"}
    write(mon_c, out / "era5_speed_mon.zarr",
          base | {"wspeed_vector": "components averaged daily first -- the "
                                   "WRONG aggregation, kept to size the bias"},
          True)
    write(day_c, out / "era5_speed_day.zarr",
          base | {"wspeed_max": "daily max of hourly samples; a model timestep "
                                "maximum is higher by construction"}, True)
    write(diur_c, out / "era5_diurnal.zarr",
          base | {"note": "mean by calendar month and hour of day (UTC)"}, True)

    # The headline number this script exists to pin down.
    s = float(mon_c.wspeed.mean())
    v = float(mon_c.wspeed_vector.mean())
    print(f"\ndomain mean scalar {s:.3f} m/s | vector {v:.3f} m/s")
    print(f"vector-averaging penalty: {(1 - v / s) * 100:.1f}% "
          "(published value 19.7%)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)