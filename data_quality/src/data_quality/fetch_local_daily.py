#!/usr/bin/env python
"""
Fetch an hourly AE product once and write two reductions from the same read.

    python fetch_local_daily.py hurs --products wrf-era5 --dry-run
    python fetch_local_daily.py hurs --products wrf-gcm wrf-era5

Output, per product:

    /shared/data/{variable}/{product}_dayagg.zarr
        daily min, max (and mean) over the WHOLE DOMAIN
    /shared/data/{variable}/{product}_stations_1hr.zarr
        the FULL HOURLY series, at station locations only

WHY TWO OUTPUTS. They answer different questions and neither substitutes for the
other.

  The domain-wide daily extremes support the gridded comparison. WRF publishes
  hourly `rh` and no daily minimum, so the fire-weather variable -- the
  afternoon minimum -- exists only if it is computed here. A daily MEAN cannot
  yield it: the afternoon dip is precisely what averaging removes.

  The station-point hourly series supports the diurnal comparison, which is
  where humidity carries most of its signal. RH swings tens of points between a
  moist night and a dry afternoon, and every operationally relevant statistic
  lives in that swing. Reducing to daily first would destroy it.

WHY THE STATION SUBSET IS CHEAP. Six WRF drivers at 35 years hourly is ~1.6 TB
over the full domain. The same data at ~900 station points is a few GB, because
a point is one cell out of ~120,000. So sub-daily structure is kept exactly
where it will be compared against observations, and discarded everywhere it
would only be read once and averaged.

ONE PASS. Both reductions come from a single dask.compute, so the hourly field
streams through once. Computing them separately would pay the read twice, and
the read is essentially the entire cost.
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
import pandas as pd
import xarray as xr

DEFAULT_ROOT = "/shared/data"
ZARR_FORMAT = 2
GRID_LABEL = "d03"
TARGET_CHUNK_MB = 128
TIME_SLICE = ("1980-01-01", "2014-12-31")

# Raw variable id per activity, mirroring fetch_local.VARIABLES.
VARIABLES = {
    "hurs":   {"WRF": "rh"},
    "wspeed": {"WRF": "wspd10mean"},
    "tas":    {"WRF": "t2"},
}

PRODUCTS = {
    "wrf-gcm":  dict(activity="WRF", experiment="historical",
                     institution="UCLA", filter="yes"),
    "wrf-era5": dict(activity="WRF", experiment="reanalysis",
                     institution="UCLA", filter="no"),
}

MEMBER_DIMS = ("sim", "simulation", "member_id", "member")

# Which HDP store holds the stations to sample at, per variable.
HDP_STORES = {"hurs": "hdp_hurs_hourly.zarr", "sfcWind": "hdp_wind_hourly.zarr"}


def member_dim(da):
    return next((d for d in MEMBER_DIMS if d in da.dims), None)


def station_points(root: Path, variable: str, meta_csv: str | None = None):
    """(ids, lat, lon, network) for the stations this variable is compared against.

    EVERY station carrying the variable is taken, not the quality-screened
    subset. Screening is a downstream choice that may well change; re-running it
    is free, while refetching a terabyte to recover a station is not.

    COORDINATES COME FROM THE STATION LIST WHERE THE STORE LACKS THEM. The
    zarr's lat/lon are promoted from each station's own store during retrieval,
    so a station whose fetch failed, or which returned nothing inside the
    window, carries NaN. The CSV has coordinates for every station regardless of
    whether its data arrived, which makes it the better source for this. It also
    has to be fixed here rather than left to the caller: cKDTree rejects
    non-finite input outright and fails deep inside scipy with no indication of
    which station was at fault.
    """
    hdp = root / "hdp"
    path = hdp / HDP_STORES.get(variable, f"hdp_{variable}_hourly.zarr")
    if not path.exists():
        path = hdp / "hdp_hourly.zarr"
    if not path.exists():
        raise FileNotFoundError(
            f"no HDP store under {hdp}; fetch stations first so the sampling "
            "locations are known")
    ds = xr.open_zarr(path, consolidated=True)
    if variable in ds:
        has = (ds[variable].notnull().sum("time") > 0).compute().values
        ds = ds.isel(station_id=np.where(has)[0])

    ids = [str(s) for s in ds.station_id.values]
    lat = np.asarray(ds.lat.values, dtype=float)
    lon = np.asarray(ds.lon.values, dtype=float)
    net = np.array([str(n) for n in ds.network.values], dtype=object)

    miss = ~(np.isfinite(lat) & np.isfinite(lon))
    if miss.any() and meta_csv:
        try:
            m = pd.read_csv(meta_csv, index_col=0)
            m = m.rename(columns={"era-id": "station_id",
                                  "latitude": "lat", "longitude": "lon"})
            if "station_id" not in m.columns:
                m = m.reset_index().rename(
                    columns={m.index.name or "index": "station_id"})
            look = (m.dropna(subset=["lat", "lon"])
                     .drop_duplicates("station_id")
                     .set_index(m.dropna(subset=["lat", "lon"])
                                .drop_duplicates("station_id")
                                .station_id.astype(str)))
            n_before = int(miss.sum())
            for i in np.where(miss)[0]:
                if ids[i] in look.index:
                    lat[i] = float(look.at[ids[i], "lat"])
                    lon[i] = float(look.at[ids[i], "lon"])
            miss = ~(np.isfinite(lat) & np.isfinite(lon))
            print(f"    backfilled {n_before - int(miss.sum())} of {n_before} "
                  "missing coordinate(s) from the station list")
        except Exception as e:
            print(f"    could not read {meta_csv} ({type(e).__name__}); "
                  "dropping stations without coordinates instead")

    if miss.any():
        print(f"    {int(miss.sum())} station(s) still have no coordinates "
              "and cannot be sampled:")
        for n, c in pd.Series(net[miss]).value_counts().items():
            print(f"      {n:<12} {c}")
        keep = np.where(~miss)[0]
        ids = [ids[i] for i in keep]
        lat, lon, net = lat[keep], lon[keep], net[keep]

    if not ids:
        raise FileNotFoundError(
            f"{path.name} has no stations with usable coordinates")
    print(f"    sampling at {len(ids)} stations from {path.name}")
    return ids, lat, lon, net


def fetch(spec, variable_id, table_id, time_slice) -> xr.Dataset:
    from climakitae import ClimateData

    processes = {"time_slice": time_slice}
    if spec["filter"] is not None:
        processes["filter_unadjusted_models"] = spec["filter"]
    q = (ClimateData().catalog("cadcat")
         .activity_id(spec["activity"]).experiment_id(spec["experiment"])
         .table_id(table_id).grid_label(GRID_LABEL)
         .variable(variable_id).processes(processes))
    if spec["institution"]:
        q = q.institution_id(spec["institution"])
    ds = q.get()
    if ds is None:
        raise RuntimeError(f"query returned nothing for {variable_id}/{table_id}")
    return ds.to_dataset(name=variable_id) if isinstance(ds, xr.DataArray) else ds


def rechunk(ds: xr.Dataset, target_mb: int = TARGET_CHUNK_MB) -> xr.Dataset:
    """Uniform, time-contiguous chunks sized by bytes.

    Zarr requires uniform chunks except the last and refuses any buffer over
    2 GiB, so the size comes from a byte budget rather than a step count: a
    fixed count silently violates the second constraint on a finer grid.
    """
    if "time" not in ds.dims or not ds.data_vars:
        return ds
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
                  max(1, int(TARGET_CHUNK_MB * 1024 ** 2) // max(per_step, 1)))
    md = member_dim(ds[list(ds.data_vars)[0]])
    chunks = {"time": n_steps}
    chunks |= {d: (1 if d == md else -1) for d in ds.dims if d != "time"}
    return ds.chunk(chunks)


def write(ds: xr.Dataset, path: Path, attrs: dict, overwrite: bool) -> dict:
    ds = rechunk(ds)
    gb = sum(ds[v].nbytes for v in ds.data_vars) / 1024 ** 3
    print(f"    {dict(ds.sizes)}  {list(ds.data_vars)}  ~{gb:.2f} GB")
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists():
        print("    exists, skipping (--overwrite to rebuild)")
        return {"path": str(path), "skipped": True}

    ds = ds.copy()
    for v in ds.variables:
        ds[v].encoding = {}
        if ds[v].dtype.kind in ("T", "U"):
            ds[v] = ds[v].astype(object)
    ds.attrs.update(attrs)

    tmp = path.parent / (path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        ds.to_zarr(tmp, mode="w", consolidated=True, zarr_format=ZARR_FORMAT)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.rename(path)
    print(f"    wrote {path.name}")
    return {"path": str(path), "gb": round(gb, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variable", help=f"one of {', '.join(VARIABLES)}")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--products", nargs="*", default=["wrf-era5"],
                    choices=list(PRODUCTS))
    ap.add_argument("--time-slice", nargs=2, default=list(TIME_SLICE),
                    metavar=("START", "END"))
    ap.add_argument("--wrf-var", default=None)
    ap.add_argument("--no-mean", action="store_true",
                    help="write only daily min and max. The mean is kept by "
                         "default because it makes the midrange bias "
                         "measurable in the model, mirroring the observational "
                         "result, and it costs one field.")
    ap.add_argument("--no-stations", action="store_true")
    ap.add_argument("--meta-csv",
                    default="s3://wecc-historical-wx/4_merge_wx_v2/"
                            "all_network_stationlist_merge.csv",
                    help="station list, used to backfill coordinates the HDP "
                         "store is missing")
    ap.add_argument("--max-dist-deg", type=float, default=0.05,
                    help="tolerance matching a station to a cell. d03 is ~3 km "
                         "(0.027 deg), so the default allows about two cells.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    dask.config.set(scheduler="threads", num_workers=a.workers)
    root = Path(a.root)
    var = a.variable
    vid = a.wrf_var or VARIABLES.get(var, {}).get("WRF")
    if vid is None:
        ap.error(f"no WRF variable_id for '{var}'; pass --wrf-var")

    out = root / var
    out.mkdir(parents=True, exist_ok=True)
    tslice = (a.time_slice[0], a.time_slice[1])

    print(f"variable   : {var} (WRF id '{vid}')")
    print(f"time slice : {tslice[0]} -> {tslice[1]}")
    print(f"products   : {a.products}")
    print(f"destination: {out}")
    print(f"mode       : {'DRY RUN' if a.dry_run else 'write'}\n")

    ids = lat = lon = net = None
    if not a.no_stations:
        try:
            ids, lat, lon, net = station_points(root, var, a.meta_csv)
        except FileNotFoundError as e:
            print(f"  {e}\n  continuing without the station output\n")
            a.no_stations = True

    manifest, failures = [], []
    for product in a.products:
        agg_path = out / f"{product}_dayagg.zarr"
        stn_path = out / f"{product}_stations_1hr.zarr"
        print(f"[{product}] querying hourly")
        try:
            ds = fetch(PRODUCTS[product], vid, "1hr", tslice)
            da = ds[vid] if vid in ds else ds[list(ds.data_vars)[0]]
            md = member_dim(da)
            n_mem = int(da.sizes[md]) if md else 1
            gb_in = da.nbytes / 1024 ** 3
            print(f"    {n_mem} member(s), {dict(da.sizes)}")
            print(f"    ~{gb_in:.0f} GB hourly to READ, reduced on the way in")

            todo = []

            # --- 1. daily extremes, whole domain --------------------------
            agg = xr.Dataset({f"{var}_min": da.resample(time="1D").min(),
                              f"{var}_max": da.resample(time="1D").max()})
            if not a.no_mean:
                agg[f"{var}_mean"] = da.resample(time="1D").mean()
                # The gap between the true daily mean and the midrange of the
                # extremes. Measuring it in the MODEL is what allows the
                # observational finding to be checked rather than assumed.
                agg["mid_minus_mean"] = (
                    (agg[f"{var}_max"] + agg[f"{var}_min"]) / 2.0
                    - agg[f"{var}_mean"])
            todo.append((agg_path, agg))

            # --- 2. full hourly, station points only ----------------------
            if not a.no_stations:
                from .grids import sample_grid_at_points
                at, dist = sample_grid_at_points(
                    da, lat, lon, ids=ids, max_dist_deg=a.max_dist_deg)
                far = int((np.asarray(dist) > a.max_dist_deg).sum())
                if far:
                    print(f"    {far} station(s) beyond {a.max_dist_deg} deg of "
                          "a cell centre -- outside the domain, left as NaN")
                stn = at.to_dataset(name=var).assign_coords(
                    network=("station_id", np.asarray(net, dtype=object)),
                    lat=("station_id", lat), lon=("station_id", lon))
                todo.append((stn_path, stn))

            if a.dry_run:
                keep = 0.0
                for p, d in todo:
                    gb = sum(d[v].nbytes for v in d.data_vars) / 1024 ** 3
                    keep += gb
                    print(f"    would write {p.name}: {dict(d.sizes)} "
                          f"~{gb:.2f} GB")
                print(f"    read {gb_in:.0f} GB -> keep {keep:.1f} GB "
                      f"({gb_in / max(keep, 1e-9):.0f}x reduction)")
                print()
                continue

            # ONE compute for both outputs: the hourly read is the entire cost,
            # and computing them separately would pay it twice.
            print(f"    reducing (one pass over {gb_in:.0f} GB)...")
            t0 = time.time()
            from dask.diagnostics import ProgressBar
            with ProgressBar(minimum=10.0):
                computed = dask.compute(*[d for _, d in todo])
            print(f"    computed in {(time.time() - t0) / 60:.1f} min")

            attrs = {"source": f"AE {product} hourly '{vid}', reduced on ingest",
                     "time_start": tslice[0], "time_end": tslice[1],
                     "hourly_gb_read": round(gb_in, 1),
                     "note": "the hourly field was NOT retained for the full "
                             "domain. A daily mean cannot yield a minimum, "
                             "which is why the reduction happens at ingest; "
                             "sub-daily structure is kept only at station "
                             "points, where it is compared."}
            for (p, _), d in zip(todo, computed):
                manifest.append({"product": product,
                                 **write(d, p, attrs, a.overwrite)})
        except Exception as e:
            failures.append(f"{product}: {type(e).__name__}: {e}")
            print(f"[{product}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
        print()

    print("=" * 62)
    for f in failures:
        print(f"  FAILED {f}")
    if manifest and not a.dry_run:
        mp = out / "_dayagg_manifest.json"
        old = json.loads(mp.read_text()) if mp.exists() else []
        mp.write_text(json.dumps(old + manifest, indent=1))
        print(f"manifest: {mp}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
