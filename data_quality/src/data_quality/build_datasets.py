#!/usr/bin/env python
"""
Build every dataset the wind-comparison analysis needs, writing zarr directly
to S3 (or a local path). Headless equivalent of wind_speed_match.ipynb.

    python build_datasets.py --list
    python build_datasets.py --dry-run
    python build_datasets.py --stages monthly elevation
    python build_datasets.py                              # everything
    python build_datasets.py --dest /tmp/out              # local rehearsal

Stages are skipped when their output already exists, so a partial run resumes
where it stopped. --overwrite forces a rebuild.

Ordering matters: later stages read earlier ones back from the store rather
than holding them in memory, so a long run can be split across invocations.

Requires: climakitae, arraylake, xarray, zarr, s3fs, dask, flox
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from .ae_era5_comparison import (
    PRESETS,
    load_source, load_era5, load_wrf_elevation,
    crop_era5_to_source, coverage_fraction,
    era5_monthly_wspeed, source_to_era5_grid,
    agreement_metrics, elevation_on_era5,
    seasonal_bias, trend_comparison,
    metrics_over_members, loca2_members, wrf_members,
    wrf_model_list, group_by_model,
    build_labels, bin_to_era5, member_dim, sample_grid_at_points,
    regrid_nearest, get_latlon, progress,
    pattern_metrics, cos_weights, extreme_metrics,
    FREQ_FOR_TABLE,
)

DEFAULT_DEST = "s3://cadcat-tmp/data_quality"
ZARR_FORMAT = 2

PRODUCTS = ["loca2", "wrf-gcm", "wrf-era5"]
T0_MON, T1_MON = "1980-01-01", "2010-12-01"
T0_DAY, T1_DAY = "1980-01-01", "2010-12-31"   
SIM = 0
COVERAGE_THRESHOLD = 0.10

# --- HDP station observations ------------------------------------------------
# The station store lives alongside every other output, so it follows --dest and
# the Store skip/exists machinery works on it unchanged:
#   {dest}/hdp_all_1980_2010.zarr
HDP_NAME = "hdp_all_1980_2010"

# Station-list CSV: lat/lon/elevation, record start/end and per-variable
# observation counts for all ~15,900 stations. Filtering it is a dataframe
# operation; the catalog alternative (screen_stations) costs one network round
# trip per station and takes hours for the same answer.
HDP_META_CSV = "s3://wecc-historical-wx/4_merge_wx_v2/all_network_stationlist_merge.cs"

# Selection window and record-span threshold. 25% keeps RAWS (the only network
# with real mountain coverage, records starting ~1997), which is where the
# model-vs-ERA5 discrepancy is largest. Stations are therefore NOT
# contemporaneous with each other -- fine for a climatological comparison
# against free-running models, whose specific years were never comparable.
HDP_TARGET = ("1/1/1980", "12/31/2010")
HDP_MIN_COVERAGE = 0.25
HDP_FREQ = "h"          # canonical time axis resolution
HDP_BATCH_SIZE = 1      # one station per fetch: a failure costs one station,
                        # and disjoint records cannot collapse a batch to time=0

# Back-compat alias. Prefer store.path(HDP_NAME).
HDP_STORE = f"{DEFAULT_DEST}/{HDP_NAME}.zarr"


# ----------------------------------------------------------------------------
# Store helpers
# ----------------------------------------------------------------------------


class Store:
    """Named zarr datasets under one prefix, local or S3."""

    def __init__(self, dest: str, overwrite: bool = False,
                 dry_run: bool = False, profile: str | None = None):
        self.dest = dest.rstrip("/")
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.is_s3 = self.dest.startswith("s3://")
        self._profile = profile
        self.storage_options = ({"profile": profile}
                                if self.is_s3 and profile else None)
        if self.is_s3:
            import s3fs
            self.fs = (s3fs.S3FileSystem(profile=profile) if profile
                       else s3fs.S3FileSystem())
        else:
            import fsspec
            import pathlib
            self.fs = fsspec.filesystem("file")
            pathlib.Path(self.dest).mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> str:
        return f"{self.dest}/{name}.zarr"

    def exists(self, name: str) -> bool:
        try:
            return self.fs.exists(self.path(name))
        except Exception:
            return False

    def _refresh(self):
        """Drop cached s3fs instances and their aiohttp sessions.

        s3fs caches filesystem objects, and a session left idle through a long
        read can fail when its SSL context is rebuilt -- surfacing as
        "load_verify_locations: No such file or directory" from
        aiobotocore's connector, despite the CA bundle being present.
        """
        if not self.is_s3:
            return
        try:
            import s3fs
            s3fs.S3FileSystem.clear_instance_cache()
            self.fs = s3fs.S3FileSystem(**({"profile": self._profile}
                                           if self._profile else {}))
            print("    refreshed S3 session")
        except Exception as e:
            print(f"    session refresh failed: {type(e).__name__}: {e}")

    def write(self, name: str, ds: xr.Dataset, retries: int = 2,
              local_fallback: str = "data_fallback") -> None:
        """Write, retrying on a stale session, with a local fallback.

        The fallback matters because some stages spend tens of minutes
        computing what they write: losing that to a transient S3 error and
        having to recompute is far more expensive than a local copy.
        """
        target = self.path(name)
        if self.dry_run:
            print(f"    would write {target}")
            return
        ds = _clean(ds)
        ds = _unify(ds)

        # Catch an all-NaN write before it reaches the store: silently empty
        # datasets are far more expensive to discover later than to reject now.
        # NB: loop variable is `vname`, not `name` -- `name` is the dataset
        # name parameter and shadowing it corrupts the fallback path.
        for vname, da in ds.data_vars.items():
            if da.dtype.kind == "f" and da.size and not bool(da.notnull().any()):
                raise ValueError(
                    f"refusing to write '{target}': variable '{vname}' is "
                    "entirely NaN")
        if not self.is_s3:
            import pathlib
            pathlib.Path(target).parent.mkdir(parents=True, exist_ok=True)

        last = None
        for attempt in range(retries + 1):
            try:
                ds.to_zarr(target, mode="w", consolidated=True,
                           zarr_format=ZARR_FORMAT,
                           storage_options=self.storage_options)
                print(f"    wrote {target}")
                return
            except Exception as e:
                last = e
                print(f"    write attempt {attempt + 1} failed: "
                      f"{type(e).__name__}: {e}")
                if attempt < retries:
                    self._refresh()
                    time.sleep(5)

        # Every retry failed. Keep the result on local disk rather than
        # discarding an expensive computation.
        if local_fallback:
            import pathlib
            p = pathlib.Path(local_fallback)
            p.mkdir(parents=True, exist_ok=True)
            lp = str(p / f"{name}.zarr")
            ds.to_zarr(lp, mode="w", consolidated=True,
                       zarr_format=ZARR_FORMAT)
            print(f"    S3 write failed; saved locally to {lp}\n"
                  f"    upload later with:  python publish_to_s3.py "
                  f"--src {local_fallback} --only {name}.zarr")
        raise last

    def read(self, name: str) -> xr.Dataset:
        return xr.open_zarr(self.path(name), consolidated=True,
                            storage_options=self.storage_options)


def _unify(ds: xr.Dataset) -> xr.Dataset:
    """One chunk per dimension, tolerating inconsistent input chunking.

    `ds.chunks` RAISES when variables disagree along a dimension, which happens
    whenever arrays from different sources are merged -- station observations
    read from a zarr store alongside model series from the catalog. Checking it
    to decide whether to rechunk is therefore the wrong test; rechunk whenever
    anything is dask-backed and let xarray reconcile.
    """
    dask_backed = any(hasattr(v.data, "chunks") for v in ds.variables.values())
    if not dask_backed:
        return ds
    return ds.chunk({d: -1 for d in ds.dims})


def _clean(ds: xr.Dataset) -> xr.Dataset:
    """Encoding and dtypes that the zarr writer will accept.

    Inherited netCDF encoding (compressors, chunking, fill values) conflicts
    with the zarr writer, and numpy 2 StringDType is not writable at all.
    """
    ds = ds.copy()
    for name in list(ds.variables):
        ds[name].encoding = {}
        if ds[name].dtype.kind in ("T", "U"):
            ds[name] = ds[name].astype(object)
    return ds


def era5_agg_name(preset: str, cadence: str) -> str:
    """Store name for a cached ERA5 aggregation.

    Keyed by preset because the crop follows the source footprint: LOCA2 and
    WRF have different domains, so their aggregations differ even at the same
    cadence.
    """
    return f"_era5_{preset}_{cadence}"


def cached_era5(store, sess, preset: str, cadence: str, crop, t0, t1, freq):
    """ERA5 reduced to the source cadence, read from the store if present.

    This is the expensive half of every run - roughly 13 GB of hourly data per
    product per cadence. Caching it means the multi-model stages, which reuse
    the same crop as the single-run stages, cost one source read instead of a
    second ERA5 pass.
    """
    name = era5_agg_name(preset, cadence)
    if store.exists(name) and not store.overwrite:
        print(f"    reusing cached ERA5 aggregation: {name}")
        return store.read(name)["era5"]

    era5_r = era5_monthly_wspeed(crop, time_start=t0, time_end=t1, freq=freq)
    with_progress = era5_r.compute()
    store.write(name, xr.Dataset({"era5": with_progress}))
    return with_progress


def context_ds(src, era5, keep, frac, cfg: dict) -> xr.Dataset:
    """Pack a run's aligned arrays into one Dataset."""
    ds = xr.Dataset({"src": src, "era5": era5,
                     "keep": keep.astype("int8"), "frac": frac})
    ds.attrs.update(variable=cfg["variable"], label=cfg["label"],
                    synchronized=int(bool(cfg["synchronized"])))
    return ds


# ----------------------------------------------------------------------------
# Shared setup, built lazily so stages that do not need it stay cheap
# ----------------------------------------------------------------------------


class Session:
    def __init__(self):
        self._era5 = None
        self._grids: dict[tuple, dict] = {}
        self._elev = None

    @property
    def era5(self) -> xr.Dataset:
        if self._era5 is None:
            print("  opening ERA5 (temporal group)...")
            self._era5 = load_era5()[["u10", "v10"]]
        return self._era5

    @property
    def elevation(self) -> xr.Dataset:
        if self._elev is None:
            self._elev = load_wrf_elevation()
        return self._elev

    def grid(self, preset: str, table_id: str | None = None, **overrides):
        """Source dataset, ERA5 crop, coverage mask and bin labels for a preset.

        Cached: LOCA2 monthly is needed by four stages, and rebuilding the
        catalog query each time is wasteful even though it is metadata-only.
        """
        key = (preset, table_id, tuple(sorted(overrides.items())))
        if key in self._grids:
            return self._grids[key]

        cfg = dict(PRESETS[preset]) | overrides
        if table_id is not None:
            cfg["table_id"] = table_id
        variable = cfg["variable"]

        ds_source = load_source(**cfg)
        crop = crop_era5_to_source(self.era5, ds_source)
        frac, keep, kind, labels = coverage_fraction(
            ds_source, crop, variable=variable,
            threshold=COVERAGE_THRESHOLD)

        out = dict(cfg=cfg, ds_source=ds_source, era5_crop=crop, frac=frac,
                   keep=keep, kind=kind, labels=labels, variable=variable)
        self._grids[key] = out
        return out


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------

STAGES: dict[str, dict] = {}


def stage(name: str, outputs: list[str], desc: str):
    def deco(fn: Callable):
        STAGES[name] = {"fn": fn, "outputs": outputs, "desc": desc}
        return fn
    return deco


def _run_one(sess: Session, store: Store, preset: str, cadence: str):
    """One product at one cadence: metrics + context."""
    table_id = "mon" if cadence == "monthly" else "day"
    t0, t1 = ((T0_MON, T1_MON) if cadence == "monthly" else (T0_DAY, T1_DAY))
    freq = FREQ_FOR_TABLE[table_id]

    g = sess.grid(preset, table_id=table_id)
    with ProgressBar():
        era5_r = cached_era5(store, sess, preset, cadence, g["era5_crop"],
                             t0, t1, freq).load()
        src_r = source_to_era5_grid(g["ds_source"], g["kind"], g["labels"],
                                    g["era5_crop"], g["variable"],
                                    sim=SIM, time_start=t0, time_end=t1).load()
    src_r, era5_r = xr.align(src_r, era5_r, join="inner")
    if src_r.sizes["time"] == 0:
        raise RuntimeError(
            f"{preset} {cadence}: no overlapping timestamps at freq={freq}. "
            "Daily AE data is sometimes stamped at 12:00 while resample('1D') "
            "gives 00:00; floor the source time axis if so.")
    print(f"    aligned on {src_r.sizes['time']} steps")

    s_m = src_r.where(g["keep"]).compute()
    e_m = era5_r.where(g["keep"])
    metrics = agreement_metrics(s_m, e_m,
                                synchronized=g["cfg"]["synchronized"]).compute()

    # Fail loudly rather than storing an empty dataset. A small non-zero time
    # overlap passes the check above but can still yield NaN everywhere.
    n_valid = int(metrics["rel_bias"].notnull().sum())
    if n_valid == 0:
        raise RuntimeError(
            f"{preset} {cadence}: metrics are entirely NaN despite "
            f"{src_r.sizes['time']} aligned steps. Check the coverage mask "
            "and the time alignment.")
    print(f"    {n_valid} valid cells, median ratio "
          f"{float(np.nanmedian(metrics['ratio'].values)):.3f}")

    metrics["coverage_fraction"] = g["frac"]
    metrics.attrs["source"] = g["cfg"]["label"]
    metrics.attrs["variable"] = g["variable"]

    store.write(f"metrics_{preset}_{cadence}", metrics)
    store.write(f"ctx_{preset}_{cadence}",
                context_ds(s_m, e_m, g["keep"], g["frac"], g["cfg"]))


@stage("monthly", [f"metrics_{p}_monthly" for p in PRODUCTS]
                  + [f"ctx_{p}_monthly" for p in PRODUCTS],
       "Per-product monthly metrics and aligned series")
def s_monthly(sess, store):
    for p in PRODUCTS:
        print(f"  {p}")
        _run_one(sess, store, p, "monthly")


@stage("daily", [f"metrics_{p}_daily" for p in PRODUCTS]
                + [f"ctx_{p}_daily" for p in PRODUCTS],
       "Per-product daily metrics and aligned series")
def s_daily(sess, store):
    for p in PRODUCTS:
        print(f"  {p}")
        _run_one(sess, store, p, "daily")


@stage("elevation", ["grids"],
       "WRF terrain and sub-grid relief on the ERA5 grid")
def s_elevation(sess, store):
    g = sess.grid("wrf-gcm")
    h_mean, h_std = elevation_on_era5(sess.elevation, g["era5_crop"])
    store.write("grids", xr.Dataset({"elev_mean": h_mean, "elev_std": h_std}))


@stage("era5_agg", ["era5_aggregations"],
       "ERA5 aggregated three ways, for the sub-daily sampling test")
def s_era5_agg(sess, store):
    g = sess.grid("wrf-gcm")
    crop = g["era5_crop"]
    u = crop.u10.sel(time=slice(T0_DAY, T1_DAY))
    v = crop.v10.sel(time=slice(T0_DAY, T1_DAY))
    wspd = np.sqrt(u ** 2 + v ** 2)

    # Only the vector path needs the daily intermediate; the scalar control
    # equals the hourly path up to month-length weighting.
    scalar_hourly = wspd.resample(time="MS").mean().mean("time")
    vector_daily = np.sqrt(u.resample(time="1D").mean() ** 2
                           + v.resample(time="1D").mean() ** 2
                           ).resample(time="MS").mean().mean("time")

    import dask
    h, vv = dask.compute(scalar_hourly, vector_daily)
    store.write("era5_aggregations",
                xr.Dataset({"scalar_hourly": h, "vector_daily": vv,
                            "penalty_pct": (1 - vv / h) * 100}))


@stage("seasonal", [f"seasonal_{p}" for p in PRODUCTS],
       "Monthly and seasonal ratio, per product")
def s_seasonal(sess, store):
    for p in PRODUCTS:
        c = store.read(f"ctx_{p}_monthly")
        sb = seasonal_bias(c["src"], c["era5"], c["keep"].astype(bool))
        store.write(f"seasonal_{p}", sb)


@stage("trends", [f"trends_{p}" for p in PRODUCTS],
       "Per-cell decadal trends, product and ERA5")
def s_trends(sess, store):
    for p in PRODUCTS:
        c = store.read(f"ctx_{p}_monthly")
        tr = trend_comparison(c["src"], c["era5"], c["keep"].astype(bool),
                              synchronized=bool(int(c.attrs["synchronized"])))
        store.write(f"trends_{p}", tr)


@stage("multimodel_loca2", ["multimodel_loca2_monthly"],
       "All 46 LOCA2 members, one ERA5 read")
def s_mm_loca2(sess, store):
    g = sess.grid("loca2")
    with ProgressBar():
        era5_mon = cached_era5(store, sess, "loca2", "monthly", g["era5_crop"],
                               T0_MON, T1_MON, "MS").load()
        mm = metrics_over_members(
            loca2_members(g["ds_source"].load(), g["variable"],
                          time_start=T0_MON, time_end=T1_MON),
            g["kind"], g["labels"], g["era5_crop"], era5_mon, g["keep"],
            synchronized=False, dim="model").load()
    store.write("multimodel_loca2_monthly", mm)
    store.write("multimodel_loca2_bygcm", group_by_model(mm))


@stage("multimodel_wrf", ["multimodel_wrf_monthly"],
       "The bias-adjusted WRF models, one ERA5 read")
def s_mm_wrf(sess, store, bias_adjusted: bool = True):
    models = wrf_model_list(bias_adjusted=bias_adjusted)
    gen, kind, labels, crop, keep, frac = wrf_members(
        models, sess.era5, variable="wspd10mean",
        time_start=T0_MON, time_end=T1_MON)
    # The WRF multi-model crop matches the wrf-gcm single-run crop (same d03
    # domain), so the cached aggregation applies.
    with ProgressBar():
        era5_mon = cached_era5(store, sess, "wrf-gcm", "monthly", crop,
                               T0_MON, T1_MON, "MS").load()
        mm = metrics_over_members(gen, kind, labels, crop, era5_mon, keep,
                                  synchronized=False, dim="model").load()
    mm.attrs["bias_adjusted_only"] = int(bias_adjusted)
    store.write("multimodel_wrf_monthly", mm)


@stage("conus404", ["metrics_conus404_monthly", "ctx_conus404_monthly"],
       "CONUS404 4 km reanalysis vs ERA5 (the like-resolution comparator)")
def s_conus404(sess, store, years=None):
    """CONUS404 against ERA5 on the same grid as every other product.

    Why this is the sharpest comparator available: CONUS404 is 4 km WRF driven
    by ERA5, so against the AE WRF-ERA5 run (3 km, also ERA5-driven) the only
    difference is model configuration -- not resolution, not driver. If both
    sit ~60% above ERA5, that is a property of km-scale WRF against a 31 km
    reanalysis rather than of the Cal-Adapt configuration.

    Cost warning: CONUS404 chunks are spatial tiles (175x175 cells, 144 hours),
    so a regional multi-decade query reads far more than ERA5's temporal
    layout does. Restrict the window with `years`.
    """
    from .ae_era5_comparison import load_conus404, conus404_wspeed

    t0, t1 = years or (T0_DAY, T1_DAY)      # default to the decade, not 30 yr
    print(f"  window {t0} -> {t1}")

    c404 = load_conus404("hourly")

    # Crop to the WRF domain's bounding box before anything else.
    g_wrf = sess.grid("wrf-gcm")
    wlat, wlon = g_wrf["ds_source"].lat, g_wrf["ds_source"].lon
    bbox = (float(wlat.min()) - 0.25, float(wlat.max()) + 0.25,
            float(wlon.min()) - 0.25, float(wlon.max()) + 0.25)

    src_h = conus404_wspeed(c404, t0, t1, freq="MS", bbox=bbox)
    src_ds = src_h.to_dataset(name="wspeed")

    crop = crop_era5_to_source(sess.era5, src_ds)
    frac, keep, kind, labels = coverage_fraction(src_ds, crop,
                                                 variable="wspeed",
                                                 threshold=COVERAGE_THRESHOLD)
    era5_r = cached_era5(store, sess, "conus404", "monthly", crop, t0, t1, "MS")
    src_r = bin_to_era5(src_h, kind, labels, crop, how="mean")
    src_r, era5_r = xr.align(src_r, era5_r, join="inner")
    print(f"    aligned on {src_r.sizes['time']} steps")

    with progress("reading CONUS404..."):
        s_m = src_r.where(keep).compute()
    e_m = era5_r.where(keep)

    metrics = agreement_metrics(s_m, e_m, synchronized=True).compute()
    n_valid = int(metrics["rel_bias"].notnull().sum())
    if n_valid == 0:
        raise RuntimeError("CONUS404 metrics are entirely NaN")
    print(f"    {n_valid} valid cells, median ratio "
          f"{float(np.nanmedian(metrics['ratio'].values)):.3f}")
    metrics["coverage_fraction"] = frac
    metrics.attrs.update(source="CONUS404 (4 km, ERA5-driven)", variable="wspeed")

    store.write("metrics_conus404_monthly", metrics)
    store.write("ctx_conus404_monthly",
                context_ds(s_m, e_m, keep, frac,
                           {"variable": "wspeed",
                            "label": "CONUS404 (4 km)",
                            "synchronized": True}))


@stage("conus404_native", ["conus404_native_monthly", "native_c404_vs_ae"],
       "CONUS404 vs the AE products on CONUS404's own 4 km grid")
def s_conus404_native(sess, store, years=None):
    """Compare km-scale products to each other WITHOUT going through ERA5.

    Why not reuse the ERA5-grid metrics: binning two ~4 km products down to
    0.25 deg discards exactly the structure that distinguishes them. The right
    common frame is the coarser of the two native grids, which is CONUS404's.

    Nearest-neighbour resampling rather than block-mean binning: at 3 km
    against 4 km the two grids are close enough that binning would leave some
    target cells empty and others doubled.

    Saves the CONUS404 native monthly field as well, so this read serves any
    later native comparison.
    """
    from .ae_era5_comparison import load_conus404, conus404_wspeed

    t0, t1 = years or (T0_MON, T1_MON)
    print(f"  window {t0} -> {t1}")

    # --- CONUS404 on its own grid, cached so the read happens once ---------
    if store.exists("conus404_native_monthly") and not store.overwrite:
        print("    reusing stored CONUS404 native field")
        c404_m = store.read("conus404_native_monthly")["wspeed"]
    else:
        g_wrf = sess.grid("wrf-gcm")
        wlat, wlon = g_wrf["ds_source"].lat, g_wrf["ds_source"].lon
        bbox = (float(wlat.min()) - 0.25, float(wlat.max()) + 0.25,
                float(wlon.min()) - 0.25, float(wlon.max()) + 0.25)

        c404 = load_conus404("hourly")
        c404_h = conus404_wspeed(c404, t0, t1, freq="MS", bbox=bbox)
        with progress("reading CONUS404 (native grid)..."):
            c404_m = c404_h.compute()
        store.write("conus404_native_monthly", c404_m.to_dataset(name="wspeed"))

    tlat = c404_m["lat"].values
    tlon = c404_m["lon"].values
    tgt_dims = tuple(c404_m["lat"].dims)
    tgt_coords = {"lat": (tgt_dims, tlat), "lon": (tgt_dims, tlon)}
    print(f"  target grid: {tlat.shape} ({tgt_dims})")

    # --- each AE product resampled onto that grid --------------------------
    out = {"conus404": c404_m}
    dists = {}
    for preset in ("wrf-era5", "wrf-gcm", "loca2"):
        g = sess.grid(preset)
        da = g["ds_source"][g["variable"]].sel(time=slice(t0, t1))
        if member_dim(da) is not None:
            da = da.isel({member_dim(da): SIM})
        print(f"  {preset}: resampling {da.shape[-2:]} -> {tlat.shape}")
        with progress(f"    reading {preset}..."):
            da = da.compute()
        r, d = regrid_nearest(da, tlat, tlon, tgt_dims, tgt_coords=tgt_coords)
        out[preset] = r
        dists[preset] = d

    # Keep only cells every product covers.
    keep = xr.ones_like(c404_m.isel(time=0), dtype=bool)
    for preset, d in dists.items():
        keep = keep & (d <= 0.15) & out[preset].isel(time=0).notnull()
    print(f"  {int(keep.sum())} of {keep.size} target cells common to all")

    # --- metrics, each AE product against CONUS404 -------------------------
    ref = out["conus404"].where(keep)
    res = []
    for preset in ("wrf-era5", "wrf-gcm", "loca2"):
        a, b = xr.align(out[preset].where(keep), ref, join="inner")
        m = agreement_metrics(a, b,
                              synchronized=(preset == "wrf-era5")).compute()
        m = m.expand_dims(product=[preset])
        res.append(m)
        print(f"    {preset:<10} ratio to CONUS404: "
              f"{float(np.nanmedian(m.ratio.values)):.3f}")
    combined = xr.concat(res, dim="product")
    combined.attrs["reference"] = "CONUS404 4 km"
    combined.attrs["note"] = ("nearest-neighbour resampled onto the CONUS404 "
                              "grid; ERA5 is not involved")
    store.write("native_c404_vs_ae", combined)


@stage("extremes", ["extremes_wrf-gcm", "extremes_wrf-era5"],
       "Upper-tail agreement from WRF wspd10max vs ERA5 daily maxima")
def s_extremes(sess, store, threshold: float = 10.0):
    """Extreme wind, for the products that can support it.

    LOCA2 is absent by necessity: it publishes only `wspeed`, a mean, and no
    maximum-wind variable at any cadence. That is a data-availability verdict
    on the extreme-wind use case, independent of any bias.

    WRF's wspd10max is a maximum over model timesteps; the ERA5 side is a
    maximum over hourly samples. WRF is therefore higher by construction, so
    read the result as a bound rather than an exact comparison.
    """
    for preset in ("wrf-gcm", "wrf-era5"):
        print(f"  {preset}")
        g = sess.grid(preset, table_id="day", variable="wspd10max")

        # ERA5 daily maxima of hourly scalar speed, then onto the source grid.
        crop = g["era5_crop"]
        u = crop.u10.sel(time=slice(T0_MON, T1_MON))
        v = crop.v10.sel(time=slice(T0_MON, T1_MON))
        era5_dmax = np.sqrt(u ** 2 + v ** 2).resample(time="1D").max()

        src = g["ds_source"]["wspd10max"].sel(time=slice(T0_MON, T1_MON))
        if member_dim(src) is not None:
            src = src.isel({member_dim(src): 0})
        src_b = bin_to_era5(src, g["kind"], g["labels"], crop, how="mean")

        src_b, era5_dmax = xr.align(src_b, era5_dmax, join="inner")
        print(f"    aligned on {src_b.sizes['time']} days")
        if src_b.sizes["time"] == 0:
            raise RuntimeError(f"{preset}: no overlapping daily timestamps")

        with progress(f"    reading {preset} maxima..."):
            s_m = src_b.where(g["keep"]).compute()
        with progress("    reading ERA5 maxima..."):
            e_m = era5_dmax.where(g["keep"]).compute()

        ex = extreme_metrics(s_m, e_m, quantiles=(0.9, 0.99),
                             threshold=threshold, keep=g["keep"]).compute()
        ex["coverage_fraction"] = g["frac"]
        ex.attrs["source"] = preset
        store.write(f"extremes_{preset}", ex)


@stage("pattern", ["pattern_fidelity"],
       "Spatial pattern fidelity, level removed, on one common mask")
def s_pattern(sess, store):
    """Do the products put the wind maxima in the right places?

    Computed on fields with the domain mean removed, so a product 50% too
    strong everywhere can still score 1.0 -- which is the question worth
    asking once the level offset is already characterised.

    EVERY comparison uses the SAME cell set. This matters more than it looks:
    the AE WRF footprint includes ocean while LOCA2 is land-masked, and the
    land-sea contrast is the largest spatial signal in the domain. Comparing a
    WRF-vs-ERA5 correlation computed with ocean against a LOCA2-vs-ERA5
    correlation computed without it would mostly measure which mask was used.
    Both masks are reported so the difference can be seen rather than hidden.
    """
    ctxs = {}
    for p in PRODUCTS:
        c = store.read(f"ctx_{p}_monthly")
        ctxs[p] = (c["src"].mean("time"), c["era5"].mean("time"),
                   c["keep"].astype(bool))

    have_c404 = (store.exists("ctx_conus404_monthly")
                 and store.exists("metrics_conus404_monthly"))
    if have_c404:
        cc = store.read("ctx_conus404_monthly")
        ctxs["conus404"] = (cc["src"].mean("time"), cc["era5"].mean("time"),
                            cc["keep"].astype(bool))

    # One mask for everything: the intersection of every product's coverage.
    names = list(ctxs)
    common = None
    for n in names:
        k = ctxs[n][2]
        common = k if common is None else (xr.align(common, k, join="inner")[0]
                                           & xr.align(common, k, join="inner")[1])
    print(f"  common mask: {int(common.sum())} cells "
          f"(per-product: {', '.join(f'{n}={int(ctxs[n][2].sum())}' for n in names)})")

    era5_ref = ctxs[PRODUCTS[0]][1]
    rows = []

    def add(src_name, src_field, ref_name, ref_field, mask, tag):
        st = pattern_metrics(src_field, ref_field, mask, cos_weights(src_field))
        st.update(product=src_name, reference=ref_name, mask=tag)
        rows.append(st)

    for n in names:
        src = ctxs[n][0]
        # ERA5 as reference, on the common mask and on the product's own mask,
        # so the effect of the mask itself is visible.
        add(n, src, "ERA5", era5_ref, common, "common")
        add(n, src, "ERA5", era5_ref, ctxs[n][2], "own")

    if have_c404:
        c404_field = ctxs["conus404"][0]
        for p in PRODUCTS:
            add(p, ctxs[p][0], "CONUS404", c404_field, common, "common")

    df = pd.DataFrame(rows)
    print()
    print(df[df["mask"] == "common"].round(3).to_string(index=False))
    store.write("pattern_fidelity",
                xr.Dataset.from_dataframe(
                    df.set_index(["product", "reference", "mask"])))


@stage("station_pattern", ["station_pattern"],
       "Spatial pattern fidelity across station locations")
def s_station_pattern(sess, store):
    """Does a product explain the spatial variation of wind BETWEEN sites?

    The gridded pattern analysis asks whether products agree with each other
    about where wind maxima sit. This asks whether they agree with the
    observations, which is the question that matters for use-case suitability.

    Two things shape the interpretation and are computed accordingly:

    1. ERA5 and CONUS404 are sampled at the same station points, so the
       products can be compared against a coarse and a fine reference on
       identical footing. Both come from fields already in the store, so this
       costs no new read.

    2. Station spatial variance includes SITING -- a ridge-top mast and a
       sheltered field differ for reasons no gridded model can reproduce, since
       both may fall in the same cell. The achievable correlation is therefore
       well below 1, and the honest comparison is RELATIVE: does a km-scale
       product explain more of the between-site variance than ERA5 does?
       Per-network figures partly control for siting, since stations within a
       network are sited alike.
    """
    stn = store.read("station_vs_models").compute()
    lat = stn["lat"].values
    lon = stn["lon"].values
    ids = stn["station_id"].values
    net = stn["network"].values

    fields = {}
    for p in PRODUCTS:
        k = f"model_{p}"
        if k in stn:
            fields[p] = stn[k].mean("time")

    # ERA5 at the station points, from the stored monthly field. No new read.
    #
    # The distance tolerance must scale with the target grid: on a 0.25 deg
    # grid the nearest cell centre is up to 0.25*sqrt(2)/2 = 0.18 deg away even
    # for a point squarely inside the domain, so a fixed 0.1 deg would flag
    # most stations as outside.
    era5_grid = store.read(f"ctx_{PRODUCTS[0]}_monthly")["era5"].mean("time").compute()
    dlat = float(np.abs(np.diff(era5_grid["latitude"].values)).mean())
    tol_era5 = dlat * 0.75
    e_at, e_dist = sample_grid_at_points(era5_grid, lat, lon, ids=ids,
                                         max_dist_deg=tol_era5)
    fields["era5"] = e_at.drop_vars(
        [c for c in e_at.coords if c != "station_id"], errors="ignore")

    # CONUS404 likewise, if built. Curvilinear, so this goes through the
    # KD-tree path inside sample_grid_at_points.
    if store.exists("conus404_native_monthly"):
        c404 = store.read("conus404_native_monthly")["wspeed"].mean("time").compute()
        # 4 km ~ 0.04 deg, so a much tighter tolerance is appropriate here.
        c_at, c_dist = sample_grid_at_points(c404, lat, lon, ids=ids,
                                             max_dist_deg=0.08)
        fields["conus404"] = c_at.drop_vars(
            [c for c in c_at.coords if c != "station_id"], errors="ignore")

    obs = stn["station"].mean("time")

    # Usable stations: inside every product's domain and with observations.
    usable = np.isfinite(obs.values)
    for f in fields.values():
        usable &= np.isfinite(f.values)
    print(f"  {int(usable.sum())} of {len(ids)} stations usable across "
          f"{len(fields)} fields")

    keep = xr.DataArray(usable, dims="station_id", coords={"station_id": ids})

    rows = []
    for name, f in fields.items():
        st = pattern_metrics(f, obs, keep)
        st.update(source=name, network="ALL")
        rows.append(st)
        # Per network: stations within a network are sited alike, so this
        # partly removes siting as a confound.
        for n in pd.unique(net):
            m = keep & xr.DataArray(net == n, dims="station_id",
                                    coords={"station_id": ids})
            if int(m.sum()) < 10:
                continue
            st = pattern_metrics(f, obs, m)
            st.update(source=name, network=str(n))
            rows.append(st)

    df = pd.DataFrame(rows)
    print()
    print(df[df["network"] == "ALL"].round(3).to_string(index=False))
    store.write("station_pattern",
                xr.Dataset.from_dataframe(df.set_index(["source", "network"])))


@stage("hdp_stations", [HDP_NAME],
       "Fetch HDP station wind observations into a zarr store "
       "(SLOW: one network round trip per station)")
def s_hdp_stations(sess, store, meta_csv: str = HDP_META_CSV,
                   min_coverage: float = HDP_MIN_COVERAGE,
                   n_per_network: int | None = None, seed: int = 42,
                   resume: bool = True):
    """Build the station archive. Headless equivalent of
    notebooks/hdp_stations_to_zarr.ipynb, sections 1-5.

    NOT in the default run: retrieval cost scales linearly with station count
    because each HDP asset is one station's zarr store, so ~610 stations is a
    couple of hours of wall time and ~1.3 GB. Name it explicitly.

    Selection is metadata-driven, in three filters: carries wind
    (`sfcwind_nobs > 0`), inside the WRF d03 perimeter, and record span
    overlapping the target window by at least `min_coverage`.

    The domain test is point-in-polygon against the TRUE perimeter, not a
    lat/lon bounding box. The d03 domain is rotated, so its bounding box
    encloses large parts of Nevada and Arizona the model never covers -- on an
    early sample the box over-counted by about a third.

    Writes append one station at a time, so progress is durable. `resume=True`
    reads the station_ids already present and fetches only the remainder, which
    matters because a mid-run failure would otherwise cost the whole build.

    Two things this stage does NOT do, by design:

      * No QC. The store is written raw; `qc_wind` is applied at read time by
        the `stations` stage, so the masking threshold stays a downstream
        choice rather than being baked into an expensive artefact.
      * No skip-list persistence. `sfcwind_nobs` counts observations across a
        station's WHOLE record, not within the target window, so a RAWS station
        spanning 1997-2022 can pass the filter and still return nothing before
        2010. Those stations are skipped by `build_zarr` and will be retried on
        every resume -- cheap relative to the stations that do return data.
    """
    from .hdp_stations import (
        load_station_metadata, metadata_summary, select_from_metadata,
        domain_polygon, build_zarr, per_network_coverage, wind_outlier_report,
        WIND_VAR, EXCLUDE_NETWORKS, ANEMOMETER_HEIGHT_M,
    )

    path = store.path(HDP_NAME)
    print(f"  target window : {HDP_TARGET[0]} - {HDP_TARGET[1]}")
    print(f"  anemometer    : {ANEMOMETER_HEIGHT_M} m (HDP standard; siting is NOT)")
    print(f"  excluded      : {list(EXCLUDE_NETWORKS)}")
    print(f"  destination   : {path}")

    # --- 1. Domain polygon --------------------------------------------------
    poly = domain_polygon(sess.elevation)
    print(f"  WRF d03 perimeter: {poly.shape[0]} points")

    # --- 2. Metadata and the selection funnel -------------------------------
    print(f"\n  reading {meta_csv}")
    meta = load_station_metadata(meta_csv, target=HDP_TARGET, region=poly,
                                 exclude=EXCLUDE_NETWORKS)
    metadata_summary(meta, min_coverage=min_coverage)

    # How the count varies with the threshold, so the choice stays visible in
    # the log rather than being an unexplained constant.
    ok = meta[meta.has_wind & meta.in_region]
    print(f"\n  coverage ladder ({len(ok)} stations with wind, inside domain):")
    for thr in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90):
        mark = "  <-- selected" if abs(thr - min_coverage) < 1e-9 else ""
        print(f"    span >= {thr:>4.0%}: {int((ok.coverage >= thr).sum()):>5}{mark}")

    selected = select_from_metadata(meta, min_coverage=min_coverage,
                                    n_per_network=n_per_network, seed=seed)
    for net, ids in sorted(selected.items(), key=lambda kv: -len(kv[1])):
        print(f"    {net:<12} {len(ids):>4}")

    # --- 3. Resume ----------------------------------------------------------
    already: set[str] = set()
    if resume and store.exists(HDP_NAME):
        try:
            ex = xr.open_zarr(path, consolidated=True,
                              storage_options=store.storage_options)
            already = {str(v) for v in ex.station_id.values}
            print(f"\n  resuming: {len(already)} stations already present")
        except Exception as e:
            print(f"\n  existing store unreadable ({type(e).__name__}: {e})"
                  " -- rebuilding from scratch")

    todo = {net: [i for i in ids if str(i) not in already]
            for net, ids in selected.items()}
    todo = {net: ids for net, ids in todo.items() if ids}
    n_todo = sum(len(v) for v in todo.values())
    print(f"\n  {n_todo} stations to fetch across {len(todo)} networks "
          f"({sum(len(v) for v in selected.values())} selected, "
          f"{len(already)} already stored)")

    if store.dry_run:
        print(f"    would write {path}")
        return
    if n_todo == 0:
        print("  nothing to fetch; store is complete")
        return

    # --- 4. Fetch and append ------------------------------------------------
    # `append` is True from the outset when resuming, so the first network of
    # this invocation extends the store instead of truncating it.
    append = bool(already)
    for net, ids in sorted(todo.items(), key=lambda kv: -len(kv[1])):
        print("\n" + "=" * 60 + f"\n{net} ({len(ids)} stations)\n" + "=" * 60)
        try:
            build_zarr(net, ids, path, time_slice=HDP_TARGET, var=WIND_VAR,
                       freq=HDP_FREQ, batch_size=HDP_BATCH_SIZE, append=append)
            append = True
        except RuntimeError as e:
            # Every station in the network was empty or failed. Not fatal: the
            # other networks still carry the analysis.
            print(f"  {net} produced nothing: {e}")

    if not append:
        raise RuntimeError("no network produced any data")

    # --- 5. Verify ----------------------------------------------------------
    ds = xr.open_zarr(path, consolidated=True,
                      storage_options=store.storage_options)
    print(f"\n  store: {ds.sizes['station_id']} stations, "
          f"{ds.sizes['time']} hourly steps")
    per_network_coverage(ds)
    print()
    wind_outlier_report(ds, top=10)


@stage("stations", ["station_vs_models"],
       "Station observations and each product sampled at station locations")
def s_stations(sess, store):
    from .hdp_stations import open_store, qc_wind, WIND_VAR

    ds_hdp = open_store(store.path(HDP_NAME))
    ds_hdp = qc_wind(ds_hdp, drop_stations_above=0.01)

    lat = ds_hdp.lat.compute().values
    lon = ds_hdp.lon.compute().values
    ids = ds_hdp.station_id.values
    net = ds_hdp.network.values

    stn = ds_hdp[WIND_VAR].resample(time="MS").mean().compute()
    counts = ds_hdp[WIND_VAR].notnull().resample(time="MS").sum().compute()
    stn = stn.where(counts >= 24 * 10)          # >= 10 days of hours

    data = {"station": stn}
    for p in PRODUCTS:
        g = sess.grid(p)
        da = g["ds_source"][g["variable"]]
        if member_dim(da) is not None:
            da = da.isel({member_dim(da): SIM})
        sampled, dist = sample_grid_at_points(
            da.sel(time=slice(T0_MON, T1_MON)), lat, lon, ids=ids)
        # Drop coordinates inherited from the source: they differ between
        # products and would collide on merge.
        sampled = sampled.drop_vars(
            [k for k in sampled.coords if k not in ("time", "station_id")],
            errors="ignore")
        sampled.attrs = {}
        data[f"model_{p}"] = sampled
        print(f"    {p}: max distance {dist.max():.3f} deg")

    out = xr.Dataset(data, coords={
        "network": ("station_id", np.asarray(net, dtype=object)),
        "lat": ("station_id", lat), "lon": ("station_id", lon),
        "station_id": np.asarray(ids, dtype=object)})
    store.write("station_vs_models", out)


@stage("remetrics", [], "Recompute metrics from stored contexts (no ERA5 read)")
def s_remetrics(sess, store):
    """Rebuild every metrics dataset from its stored context.

    The contexts hold the aligned, masked source and ERA5 series, which is
    everything agreement_metrics needs. Use this when the metrics are wrong or
    the metric definitions change: it costs seconds, where rerunning `monthly`
    or `daily` costs an ERA5 read per product.
    """
    for cadence in ("monthly", "daily"):
        for p in PRODUCTS:
            cname = f"ctx_{p}_{cadence}"
            if not store.exists(cname):
                print(f"    {cname} absent, skipping")
                continue
            c = store.read(cname)
            src = c["src"]
            era5 = c["era5"]
            sync = bool(int(c.attrs.get("synchronized", 0)))

            m = agreement_metrics(src, era5, synchronized=sync).compute()
            n_valid = int(m["rel_bias"].notnull().sum())
            if n_valid == 0:
                raise RuntimeError(
                    f"{p} {cadence}: recomputed metrics are entirely NaN. "
                    "The stored context is empty or masked out.")
            m.attrs.update(source=c.attrs.get("label", p),
                           variable=c.attrs.get("variable", ""))
            store.write(f"metrics_{p}_{cadence}", m)
            print(f"    {p} {cadence}: {n_valid} valid cells, "
                  f"median ratio {float(np.nanmedian(m.ratio.values)):.3f}")


ORDER = ["monthly", "daily", "elevation", "era5_agg", "seasonal", "trends",
         "multimodel_loca2", "multimodel_wrf", "stations"]

# Not in a default run: CONUS404 is a much heavier read than the AE products.
ORDER = ORDER + ["extremes", "pattern", "station_pattern"]
# Not in a default run either: hdp_stations is hours of network round trips,
# and the station archive changes far less often than the analysis does.
ORDER_OPTIONAL = ["hdp_stations", "conus404", "conus404_native", "remetrics"]




# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--stages", nargs="*", default=None,
                    help=f"subset of: {' '.join(ORDER)}")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--profile", default=None, help="AWS profile")
    ap.add_argument("--workers", type=int, default=16,
                    help="dask threads; the ERA5 read is latency-bound, so "
                         "oversubscribing past the core count helps")
    ap.add_argument("--no-era5-cache", action="store_true",
                    help="recompute ERA5 aggregations instead of reusing the "
                         "cached _era5_* datasets")
    ap.add_argument("--c404-years", nargs=2, default=None,
                    metavar=("START", "END"),
                    help="window for the conus404 stage, e.g. 2000-01-01 "
                         "2010-12-31 (default: the daily window)")
    ap.add_argument("--all-wrf", action="store_true",
                    help="include WRF models without a-priori bias adjustment")
    ap.add_argument("--hdp-meta", default=HDP_META_CSV,
                    help="station-list CSV for the hdp_stations stage")
    ap.add_argument("--hdp-min-coverage", type=float, default=HDP_MIN_COVERAGE,
                    help="minimum record-span overlap with the target window")
    ap.add_argument("--hdp-n-per-network", type=int, default=None,
                    help="cap stations per network (default: keep all)")
    ap.add_argument("--hdp-no-resume", action="store_true",
                    help="rebuild the station store from scratch instead of "
                         "fetching only the stations not already in it")
    a = ap.parse_args()

    if a.list:
        print(f"{'stage':<20} {'outputs':<44} description")
        print("-" * 100)
        for name in ORDER:
            sp = STAGES[name]
            print(f"{name:<20} {','.join(sp['outputs'])[:42]:<44} {sp['desc']}")
        print("\noptional (name explicitly with --stages):")
        for name in ORDER_OPTIONAL:
            sp = STAGES[name]
            print(f"{name:<20} {','.join(sp['outputs'])[:42]:<44} {sp['desc']}")
        return 0

    import dask
    dask.config.set(scheduler="threads", num_workers=a.workers)

    store = Store(a.dest, overwrite=a.overwrite, dry_run=a.dry_run,
                  profile=a.profile)
    sess = Session()

    wanted = a.stages or ORDER
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        print(f"unknown stages: {unknown}\nchoose from: {ORDER}")
        return 1
    # Dependency order, not display order: hdp_stations must precede the
    # `stations` stage that reads its output, even though it is optional.
    order = ["hdp_stations"] + ORDER + [o for o in ORDER_OPTIONAL
                                        if o != "hdp_stations"]
    wanted = [s for s in order if s in wanted and s not in a.skip]

    print(f"destination : {store.dest}")
    print(f"stages      : {wanted}")
    print(f"mode        : {'DRY RUN' if a.dry_run else 'write'}"
          f"{' (overwrite)' if a.overwrite else ''}\n")

    results = {}
    for name in wanted:
        spec = STAGES[name]
        # Skip only when EVERY output is present. Skipping on a partial set
        # would strand a stage that failed part-way -- which is exactly what
        # happens when an expensive intermediate is written and the cheaper
        # products after it are not.
        have = [o for o in spec["outputs"] if store.exists(o)]
        missing = [o for o in spec["outputs"] if o not in have]
        if spec["outputs"] and not missing and not a.overwrite:
            print(f"[{name}] SKIP - all {len(have)} outputs present "
                  "(--overwrite to rebuild)")
            results[name] = "skipped"
            continue
        if have and missing:
            print(f"[{name}] resuming - have {have}, missing {missing}")

        print(f"[{name}] {spec['desc']}")
        t = time.time()
        try:
            if name == "multimodel_wrf":
                spec["fn"](sess, store, bias_adjusted=not a.all_wrf)
            elif name == "hdp_stations":
                spec["fn"](sess, store, meta_csv=a.hdp_meta,
                           min_coverage=a.hdp_min_coverage,
                           n_per_network=a.hdp_n_per_network,
                           resume=not a.hdp_no_resume)
            elif name in ("conus404", "conus404_native"):
                spec["fn"](sess, store, years=tuple(a.c404_years)
                           if a.c404_years else None)
            else:
                spec["fn"](sess, store)
            results[name] = "ok"
            print(f"[{name}] done in {time.time() - t:.0f}s\n")
        except Exception as e:
            results[name] = f"FAILED: {type(e).__name__}"
            print(f"[{name}] FAILED after {time.time() - t:.0f}s: "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
            print()

    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<20} {v}")
    failed = [k for k, v in results.items() if str(v).startswith("FAILED")]
    if not a.dry_run:
        print(f"\nlist with:  aws s3 ls {store.dest}/"
              if store.is_s3 else f"\nlist with:  ls {store.dest}/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
