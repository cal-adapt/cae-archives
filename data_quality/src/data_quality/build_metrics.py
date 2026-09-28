#!/usr/bin/env python
"""
Build every metrics dataset the wind comparison needs, from the local archive.

    python build_metrics.py --list
    python build_metrics.py --dry-run
    python build_metrics.py --stages monthly pattern
    python build_metrics.py                        # everything, resumable

Reads /shared/data (fetch_* + derive_*), writes /shared/data/metrics.

WHAT CHANGED FROM THE S3 PIPELINE. Every read is now a local zarr, so the
machinery that existed to survive the network is gone: no lazy ERA5 session, no
cached aggregations, no S3 retry-and-fallback, no scheduling stages across
terminals to share bandwidth. One sequential run.

The cached ERA5 aggregations are worth a note because they were a correctness
problem, not just a speed one. They were keyed on (product, cadence) but NOT on
the time window, so changing the window silently reused a stale object:
align(join="inner") then intersected the source down to it and the run finished
with no error and the wrong answer. Recomputing from local hourly data removes
the failure mode rather than guarding it.

ALIGNMENT IS PER PAIR. Products cover different spans -- LOCA2 1980-01 to
2014-12, WRF 1980-09 to 2014-08 -- and two of the three daily products use a
365-day calendar while wrf-era5 does not. Each comparison aligns on its own
overlap rather than everything being cut to a global intersection, so each uses
all the data it has. The span actually used is written into every output's
attrs, because it differs between pairs and that has to be visible rather than
assumed.

MEMBER SELECTION FOR THE REFERENCE RUNS. Section 3 of the review needs one
simulation per product, chosen so the contrasts isolate one thing at a time:
LOCA2 and WRF-GCM share a driving GCM (CNRM-ESM2-1 where available), so their
difference isolates the DOWNSCALING METHOD; WRF-GCM and WRF-ERA5 share a model,
so theirs isolates the DRIVER.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import dask
import numpy as np
import pandas as pd
import xarray as xr

from . import localdata as L
from .grids import (
    bin_to_era5, coverage_fraction, crop_era5_to_source, elevation_on_era5,
    land_fraction_on_era5, member_dim, progress, region_masks, regrid_nearest,
    sample_grid_at_points, cos_weights,
)
from .metrics import (
    agreement_metrics, elevation_summary, extreme_metrics,
    metrics_over_members, group_by_model, pattern_metrics, region_summary,
    seasonal_bias, trend_comparison,
)
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/shared/data")
OUT = ROOT / "metrics"
ZARR_FORMAT = 2

VARIABLE = "wspeed"
PRODUCTS = ["loca2", "wrf-gcm", "wrf-era5"]

# Same-timestep metrics are meaningful only for products driven by the observed
# atmosphere. A free-running GCM makes its own weather, so sync_corr would
# compare unrelated sequences and score near zero however good the model is.
SYNCHRONIZED = {"loca2": False, "wrf-gcm": False, "wrf-era5": True,
                "conus404": True}

# Driving GCM preferred for the single-run reference comparisons, so LOCA2 and
# WRF-GCM share one and their difference isolates the downscaling method.
REFERENCE_GCM = "cnrm"

COVERAGE_THRESHOLD = 0.5
EXCEED_THRESHOLD = 10.0        # m/s, for the exceedance-frequency metric


# ----------------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------------


class Store:
    """Named zarr datasets under one local prefix."""

    def __init__(self, dest: Path, overwrite=False, dry_run=False):
        self.dest = Path(dest)
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.dest.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dest / f"{name}.zarr"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def read(self, name: str) -> xr.Dataset:
        return xr.open_zarr(self.path(name), consolidated=True)

    def write(self, name: str, ds: xr.Dataset) -> None:
        target = self.path(name)
        if self.dry_run:
            print(f"    would write {target}")
            return

        # Refuse an all-NaN write. A silently empty dataset is far more
        # expensive to discover three stages later than to reject now.
        for v, da in ds.data_vars.items():
            if da.dtype.kind == "f" and da.size and not bool(da.notnull().any()):
                raise ValueError(f"refusing to write '{name}': "
                                 f"variable '{v}' is entirely NaN")

        ds = ds.copy()
        for v in ds.variables:
            ds[v].encoding = {}
            if ds[v].dtype.kind in ("T", "U"):
                ds[v] = ds[v].astype(object)
        ds = ds.chunk({d: -1 for d in ds.dims}) if any(
            hasattr(ds[v].data, "chunks") for v in ds.variables) else ds

        tmp = target.parent / (target.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            ds.to_zarr(tmp, mode="w", consolidated=True, zarr_format=ZARR_FORMAT)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
        print(f"    wrote {target.name}")


def span_attrs(*arrays) -> dict:
    """Record the window each comparison actually used.

    Pairs align on their own overlap, so this differs between outputs. Writing
    it down is the difference between a documented choice and a silent one.
    """
    a = arrays[0]
    t = a.time.values
    return {"time_start": str(t[0])[:10], "time_end": str(t[-1])[:10],
            "n_steps": int(a.sizes["time"]),
            "span_years": round(float((t[-1] - t[0])
                                      / np.timedelta64(365, "D")), 2)}


# ----------------------------------------------------------------------------
# Session: grid objects, built once per (product, cadence)
# ----------------------------------------------------------------------------


class Session:
    def __init__(self, variable=VARIABLE, root=ROOT, time_slice=None,
                 station_networks="analysis", min_months=None):
        self.variable = variable
        self.root = root
        # A window applied to EVERY source, so a run is internally consistent.
        # Alignment is still per pair on top of it: products cover different
        # spans, and intersecting each pair separately uses all the data each
        # has rather than cutting everything to the shortest.
        self.time_slice = time_slice
        self.station_networks = station_networks
        self.min_months = min_months
        self._grids: dict = {}
        self._era5: dict = {}
        self._elev = None

    def clip(self, da):
        return da if self.time_slice is None else da.sel(
            time=slice(*self.time_slice))

    def stations(self, **kw):
        """Observations under this run's window and network policy.

        Both matter for a short window: record length is measured against the
        window, and the network exclusions are themselves window-dependent.
        """
        opts = dict(root=self.root, time_slice=self.time_slice,
                    networks=self.station_networks)
        if self.min_months is not None:
            opts["min_months"] = self.min_months
        opts.update(kw)
        return L.open_hdp(**opts)

    def era5(self, cadence: str, how: str = "mean") -> xr.DataArray:
        """Reference speed at a cadence, from the derived stores if present.

        derive_era5.py writes these in one pass over the hourly components;
        falling back to computing from hourly keeps the pipeline runnable
        without that step, just slower.
        """
        key = (cadence, how)
        if key in self._era5:
            return self._era5[key]
        name = {"monthly": "era5_speed_mon", "daily": "era5_speed_day"}[cadence]
        path = self.root / "era5" / f"{name}.zarr"
        if path.exists():
            ds = xr.open_zarr(path, consolidated=True)
            var = "wspeed_max" if how == "max" else "wspeed"
            if var not in ds:
                raise KeyError(f"{path.name} has no '{var}' "
                               "-- rerun derive_era5.py")
            da = ds[var]
        else:
            print(f"  {name} not found; computing from hourly "
                  "(run derive_era5.py to avoid this)")
            freq = "MS" if cadence == "monthly" else "1D"
            da = L.era5_speed(freq=freq, how=how, root=self.root)
        da = self.clip(da)
        self._era5[key] = da
        return da

    @property
    def elevation(self) -> xr.Dataset:
        if self._elev is None:
            self._elev = L.open_elevation(self.root)
        return self._elev

    def grid(self, product: str, cadence: str = "monthly",
             variable: str | None = None) -> dict:
        """Source array plus the objects needed to put it on the ERA5 grid.

        Cached: LOCA2 monthly is needed by four stages, and rebuilding the
        labels each time is wasteful even though it is only metadata.
        """
        variable = variable or self.variable
        key = (product, cadence, variable)
        if key in self._grids:
            return self._grids[key]

        tid = {"monthly": "mon", "daily": "day"}[cadence]
        src = self.clip(L.open_product(variable, product, tid, self.root))
        ref = self.era5(cadence)
        crop = ref.to_dataset(name="wspeed")
        frac, keep, kind, labels = coverage_fraction(
            src.to_dataset(name=variable), crop, variable=variable,
            threshold=COVERAGE_THRESHOLD)
        out = dict(src=src, ref=ref, crop=crop, frac=frac, keep=keep,
                   kind=kind, labels=labels, variable=variable,
                   synchronized=SYNCHRONIZED.get(product, False))
        self._grids[key] = out
        return out


def reference_member(da: xr.DataArray, prefer: str = REFERENCE_GCM):
    """One member, preferring a named driving GCM so contrasts stay clean."""
    md = member_dim(da)
    if md is None:
        return da, "single"
    names = [str(m) for m in da[md].values]
    pick = next((n for n in names if prefer in n.lower()), names[0])
    return da.sel({md: pick}), pick


def to_ref_grid(g: dict, da: xr.DataArray) -> xr.DataArray:
    return bin_to_era5(da, g["kind"], g["labels"], g["crop"], how="mean")


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------

STAGES: dict[str, dict] = {}


def stage(name: str, outputs: list[str], desc: str):
    def deco(fn: Callable):
        STAGES[name] = {"fn": fn, "outputs": outputs, "desc": desc}
        return fn
    return deco


def _one(sess: Session, store: Store, product: str, cadence: str) -> None:
    g = sess.grid(product, cadence)
    src, _ = reference_member(g["src"])
    binned = to_ref_grid(g, src)

    # Natural alignment: each pair uses its own overlap.
    b, r = xr.align(binned, g["ref"], join="inner")
    if b.sizes["time"] == 0:
        raise RuntimeError(
            f"{product} {cadence}: no overlapping timestamps. Daily model data "
            "is sometimes stamped at 12:00 while a resample gives 00:00; floor "
            "the source time axis if so.")
    print(f"    aligned on {b.sizes['time']} steps "
          f"({str(b.time.values[0])[:10]} -> {str(b.time.values[-1])[:10]})")

    with progress(f"    reading {product} {cadence}..."):
        s_m = b.where(g["keep"]).compute()
    e_m = r.where(g["keep"]).compute()

    m = agreement_metrics(s_m, e_m, synchronized=g["synchronized"]).compute()
    n_valid = int(m["rel_bias"].notnull().sum())
    if n_valid == 0:
        raise RuntimeError(f"{product} {cadence}: metrics are entirely NaN "
                           f"despite {b.sizes['time']} aligned steps")
    print(f"    {n_valid} valid cells, median ratio "
          f"{float(np.nanmedian(m.ratio.values)):.3f}")

    m["coverage_fraction"] = g["frac"]
    m.attrs.update(product=product, cadence=cadence, variable=g["variable"],
                   synchronized=int(g["synchronized"]), **span_attrs(s_m))

    # Land / ocean / pooled, so the headline number is never quoted without the
    # composition that produced it.
    if store.exists("grids"):
        grids = store.read("grids")
        if "land_fraction" in grids:
            lf, _ = xr.align(grids.land_fraction, m.ratio, join="right")
            df = region_summary(m, region_masks(lf))
            for region in df.index:
                for col in df.columns:
                    if col != "n_cells":
                        m.attrs[f"{region}_{col}"] = round(
                            float(df.loc[region, col]), 4)
    store.write(f"metrics_{product}_{cadence}", m)

    ctx = xr.Dataset({"src": s_m, "era5": e_m,
                      "keep": g["keep"].astype("int8"), "frac": g["frac"]})
    ctx.attrs.update(m.attrs)
    store.write(f"ctx_{product}_{cadence}", ctx)


@stage("monthly", [f"metrics_{p}_monthly" for p in PRODUCTS]
                  + [f"ctx_{p}_monthly" for p in PRODUCTS],
       "Per-product monthly metrics and aligned series")
def s_monthly(sess, store):
    for p in PRODUCTS:
        print(f"  {p}")
        _one(sess, store, p, "monthly")


@stage("daily", [f"metrics_{p}_daily" for p in PRODUCTS]
                + [f"ctx_{p}_daily" for p in PRODUCTS],
       "Per-product daily metrics -- does cadence change the answer?")
def s_daily(sess, store):
    for p in PRODUCTS:
        print(f"  {p}")
        _one(sess, store, p, "daily")


@stage("elevation", ["grids"],
       "Terrain, sub-grid relief and land fraction on the reference grid")
def s_elevation(sess, store):
    """Sub-grid relief is the better predictor of a resolution-driven gap.

    A flat plateau and a rugged range can share a mean height but not a spread
    of terrain inside the cell, and it is the spread that measures what a 31 km
    reanalysis cannot resolve.

    Land fraction is written alongside because the resolution effect only exists
    where there IS terrain. On the full d03 union footprint a large part of the
    domain is open Pacific, where a km-scale model and a coarse reanalysis have
    nothing sub-grid to disagree about -- so a pooled median is diluted by cells
    carrying no signal. Measured here: CONUS404 pooled 1.124, land-only 1.545.
    """
    g = sess.grid("wrf-gcm", "monthly")
    h_mean, h_std = elevation_on_era5(sess.elevation, g["crop"])
    out = xr.Dataset({"elev_mean": h_mean, "elev_std": h_std})

    # Prefer a real land mask over a product's NaN footprint, which is that
    # product's own masking decision rather than geography.
    static = sess.root / "conus404" / "conus404_static.zarr"
    if static.exists():
        sm = xr.open_zarr(static, consolidated=True)
        if "LANDMASK" in sm:
            lf = land_fraction_on_era5(sm, g["crop"])
            out["land_fraction"] = lf
            r = region_masks(lf)
            for k, v in r.items():
                print(f"    {k:<8} {int(v.sum()):>5} cells")
        else:
            print("    conus404_static has no LANDMASK; skipping land fraction")
    else:
        print(f"    {static.name} not found -- no land fraction. "
              "Run fetch_conus404.py (statics are written first and are cheap).")
    store.write("grids", out)


@stage("era5_agg", ["era5_aggregations"],
       "ERA5 scalar vs vector averaging -- the sampling control")
def s_era5_agg(sess, store):
    """How much does averaging components before taking the magnitude cost?

    The answer bounds one candidate explanation for the model-ERA5 offset. It
    is comparable in size (~20%) but WRONG IN SIGN: vector averaging biases a
    product low, and the products run high.
    """
    path = sess.root / "era5" / "era5_speed_mon.zarr"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run derive_era5.py")
    ds = xr.open_zarr(path, consolidated=True)
    if "wspeed_vector" not in ds:
        raise KeyError("era5_speed_mon has no 'wspeed_vector'")
    h = ds.wspeed.mean("time")
    v = ds.wspeed_vector.mean("time")
    out = xr.Dataset({"scalar_hourly": h, "vector_daily": v,
                      "penalty_pct": (1 - v / h) * 100}).compute()
    # Compute first: dask has no full-array nanmedian ("only works along an
    # axis"), and these are two 2D fields, so materialising them is free.
    print(f"    vector-averaging penalty: "
          f"{float(np.nanmedian(out.penalty_pct.values)):.1f}% "
          "(published value 19.7%)")
    store.write("era5_aggregations", out)


@stage("seasonal", [f"seasonal_{p}" for p in PRODUCTS],
       "Monthly and seasonal ratio, per product")
def s_seasonal(sess, store):
    for p in PRODUCTS:
        c = store.read(f"ctx_{p}_monthly")
        store.write(f"seasonal_{p}",
                    seasonal_bias(c["src"], c["era5"], c["keep"].astype(bool)))


@stage("trends", [f"trends_{p}" for p in PRODUCTS],
       "Per-cell decadal trends, product and reference")
def s_trends(sess, store):
    for p in PRODUCTS:
        c = store.read(f"ctx_{p}_monthly")
        store.write(f"trends_{p}", trend_comparison(
            c["src"], c["era5"], c["keep"].astype(bool),
            synchronized=bool(int(c.attrs.get("synchronized", 0)))))


@stage("multimodel", ["multimodel_loca2", "multimodel_loca2_bygcm",
                      "multimodel_wrf"],
       "Every member of LOCA2 and WRF against the reference")
def s_multimodel(sess, store):
    """Across-member spread against distance from the reference.

    If the offset is far larger than the disagreement between models, it is set
    by the downscaling chain rather than by the driving climate model -- which
    is what a high signal-to-noise ratio here would mean.
    """
    for product, name in (("loca2", "multimodel_loca2"),
                          ("wrf-gcm", "multimodel_wrf")):
        # Per-product skip. The stage declares three outputs, so the driver's
        # all-or-nothing check would rerun LOCA2's 46 sequential members just to
        # reach WRF. Checking here makes the stage resumable at product level.
        done = store.exists(name) and (
            product != "loca2" or store.exists("multimodel_loca2_bygcm"))
        if done and not store.overwrite:
            print(f"  {product}: {name} present, skipping")
            continue
        g = sess.grid(product, "monthly")
        da = g["src"]
        md = member_dim(da)
        names = [str(m) for m in da[md].values] if md else ["single"]
        print(f"  {product}: {len(names)} members")
        members = ((n, da.sel({md: n})) for n in names) if md \
            else [("single", da)]
        mm = metrics_over_members(members, g["kind"], g["labels"], g["crop"],
                                  g["ref"], g["keep"],
                                  synchronized=g["synchronized"], dim="model")
        mm.attrs.update(product=product, n_members=len(names))
        store.write(name, mm)
        if product == "loca2":
            # A flat mean over members weights a GCM contributing ten of them
            # ten times more than one contributing a single run.
            store.write("multimodel_loca2_bygcm", group_by_model(mm))


@stage("conus404", ["metrics_conus404_monthly", "ctx_conus404_monthly"],
       "CONUS404 vs the reference -- an independent km-scale comparator")
def s_conus404(sess, store):
    """The one comparison that tests whether the offset is about resolution.

    CONUS404 is 4 km WRF driven by ERA5, built by NCAR/USGS -- same resolution
    class and same driver as the AE reanalysis run, differing only in model
    configuration. If two independently configured km-scale models agree with
    each other while both sit far above a 31 km reanalysis, the offset is a
    property of the comparison rather than of either model.
    """
    path = sess.root / "conus404" / "conus404_speed_mon.zarr"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run derive_conus404.py")
    ds = sess.clip(xr.open_zarr(path, consolidated=True))
    if "wspeed" not in ds:
        raise KeyError(
            "conus404_speed_mon holds no 'wspeed'. It was derived from a "
            "non-hourly source, so its speed is the VECTOR mean (~20% low) and "
            "cannot support a level comparison. Fetch the hourly product.")

    src = ds.wspeed
    ref = sess.era5("monthly")
    crop = ref.to_dataset(name="wspeed")
    frac, keep, kind, labels = coverage_fraction(
        src.to_dataset(name="wspeed"), crop, variable="wspeed",
        threshold=COVERAGE_THRESHOLD)
    binned = bin_to_era5(src, kind, labels, crop, how="mean")
    b, r = xr.align(binned, ref, join="inner")
    print(f"    aligned on {b.sizes['time']} steps")

    with progress("    reading CONUS404..."):
        s_m = b.where(keep).compute()
    e_m = r.where(keep).compute()
    m = agreement_metrics(s_m, e_m, synchronized=True).compute()
    m["coverage_fraction"] = frac
    m.attrs.update(product="conus404", cadence="monthly", **span_attrs(s_m))
    print(f"    median ratio {float(np.nanmedian(m.ratio.values)):.3f}")
    store.write("metrics_conus404_monthly", m)
    store.write("ctx_conus404_monthly", xr.Dataset(
        {"src": s_m, "era5": e_m, "keep": keep.astype("int8"), "frac": frac}))


@stage("conus404_native", ["native_c404_vs_ae"],
       "AE products vs CONUS404 on CONUS404's own grid, reference not involved")
def s_conus404_native(sess, store):
    """Binning two km-scale products down to 0.25 deg discards exactly the
    structure that distinguishes them, so the right common frame is the coarser
    of the two native grids. Nearest-neighbour rather than block-mean: at 3 km
    against 4 km, binning would leave some target cells empty and others doubled.
    """
    path = sess.root / "conus404" / "conus404_speed_mon.zarr"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run derive_conus404.py")
    c4 = xr.open_zarr(path, consolidated=True)
    var = "wspeed" if "wspeed" in c4 else "wspeed_vector"
    if var == "wspeed_vector":
        print("    WARNING: CONUS404 speed is the vector mean; level metrics "
              "here are biased ~20% low. Pattern metrics are unaffected.")
    tgt = c4[var]
    tlat, tlon = tgt.lat.values, tgt.lon.values

    rows = []
    for p in PRODUCTS:
        src, name = reference_member(
            L.open_product(sess.variable, p, "mon", sess.root))
        res = regrid_nearest(src, tlat, tlon, tgt.dims[-2:],
                             tgt_coords={"lat": (tgt.dims[-2:], tlat),
                                         "lon": (tgt.dims[-2:], tlon)})
        a, b = xr.align(res, tgt, join="inner")
        m = agreement_metrics(a, b, synchronized=False).compute()
        m = m.expand_dims(product=[p])
        m.attrs.update(member=name)
        rows.append(m)
        print(f"    {p} ({name}): median ratio "
              f"{float(np.nanmedian(m.ratio.values)):.3f}")
    out = xr.concat(rows, dim="product")
    out.attrs["reference"] = f"CONUS404 native grid ({var})"
    store.write("native_c404_vs_ae", out)


@stage("extremes", ["extremes_wrf-gcm", "extremes_wrf-era5"],
       "Upper-tail agreement from WRF daily maxima")
def s_extremes(sess, store, threshold: float = EXCEED_THRESHOLD):
    """LOCA2 is absent by necessity: it publishes only a mean wind and no
    maximum at any cadence. That is a data-availability verdict on the
    extreme-wind use case, independent of any bias.

    WRF's wspd10max is a maximum over MODEL TIMESTEPS; the reference side is a
    maximum over HOURLY SAMPLES. WRF is higher by construction, so read the
    result as a bound rather than an equality.
    """
    ref = sess.era5("daily", how="max")
    for product in ("wrf-gcm", "wrf-era5"):
        try:
            src = sess.clip(L.open_product("wspd10max", product, "day",
                                           sess.root))
        except FileNotFoundError:
            print(f"  {product}: no wspd10max store -- run "
                  "`fetch_local.py wspd10max --cadence day`; skipping")
            continue
        print(f"  {product}")
        src, _ = reference_member(src)
        crop = ref.to_dataset(name="wspeed")
        frac, keep, kind, labels = coverage_fraction(
            src.to_dataset(name="wspd10max"), crop, variable="wspd10max",
            threshold=COVERAGE_THRESHOLD)
        b, r = xr.align(bin_to_era5(src, kind, labels, crop, how="mean"),
                        ref, join="inner")
        print(f"    aligned on {b.sizes['time']} days")
        with progress("    reading maxima..."):
            s_m = b.where(keep).compute()
        e_m = r.where(keep).compute()
        ex = extreme_metrics(s_m, e_m, quantiles=(0.9, 0.99),
                             threshold=threshold, keep=keep).compute()
        ex["coverage_fraction"] = frac
        ex.attrs.update(product=product, **span_attrs(s_m))
        store.write(f"extremes_{product}", ex)


@stage("pattern", ["pattern_fidelity"],
       "Spatial pattern fidelity, level removed, on one common mask")
def s_pattern(sess, store):
    """Are the maxima in the right PLACES, independent of level?

    EVERY comparison uses the SAME cell set. The WRF footprint includes ocean
    while LOCA2 is land-masked, and the land-sea contrast is the largest spatial
    signal in the domain -- so comparing a WRF correlation computed with ocean
    against a LOCA2 one computed without it would mostly measure which mask was
    used. Both masks are reported so the difference is visible rather than
    hidden.
    """
    ctxs = {}
    for p in PRODUCTS:
        c = store.read(f"ctx_{p}_monthly")
        ctxs[p] = (c["src"].mean("time"), c["era5"].mean("time"),
                   c["keep"].astype(bool))
    if store.exists("ctx_conus404_monthly"):
        c = store.read("ctx_conus404_monthly")
        ctxs["conus404"] = (c["src"].mean("time"), c["era5"].mean("time"),
                            c["keep"].astype(bool))

    common = None
    for _, (_, _, k) in ctxs.items():
        common = k if common is None else (common & k)
    print(f"  common mask: {int(common.sum())} cells")

    rows = []
    for name, (s, r, own) in ctxs.items():
        w = cos_weights(r)
        for mask_name, mask in (("common", common), ("own", own)):
            st = pattern_metrics(s, r, mask, weights=w)
            st.update(product=name, reference="ERA5", mask=mask_name,
                      n_cells=int(mask.sum()))
            rows.append(st)
    df = pd.DataFrame(rows)
    for c in ("product", "reference", "mask"):
        df[c] = df[c].map(str)
    print(df[df["mask"] == "common"][
        ["product", "pattern_corr", "std_ratio", "mean_ratio"]]
        .round(3).to_string(index=False))
    store.write("pattern_fidelity", xr.Dataset.from_dataframe(
        df.set_index(["product", "reference", "mask"])))


@stage("stations", ["station_vs_models"],
       "Station observations and each product sampled at station locations")
def s_stations(sess, store):
    """Stations are the independent arbiter: whether the products or the
    reanalysis are closer to reality cannot be settled by comparing them to each
    other.

    Monthly means with a >= 10 days of hourly values guard, so a sparse month
    cannot contribute a mean built from a handful of hours.
    """
    obs = sess.stations()                    # window, networks, QC, quality
    ds = L._open(sess.root / "hdp" / "hdp_hourly.zarr")
    ids = [str(i) for i in obs.station_id.values]
    keep_idx = np.isin([str(i) for i in ds.station_id.values], ids)
    lat = ds.lat.values[keep_idx]
    lon = ds.lon.values[keep_idx]
    net = np.asarray([str(n) for n in ds.network.values])[keep_idx]

    with progress("  aggregating stations to monthly..."):
        counts = obs.notnull().resample(time="MS").sum().compute()
        stn = obs.resample(time="MS").mean().compute().where(counts >= 24 * 10)

    data = {"station": stn}
    for p in PRODUCTS:
        da, _ = reference_member(L.open_product(sess.variable, p, "mon",
                                                sess.root))
        sampled, dist = sample_grid_at_points(da, lat, lon, ids=ids)
        sampled = sampled.drop_vars(
            [c for c in sampled.coords if c not in ("time", "station_id")],
            errors="ignore")
        sampled.attrs = {}
        data[f"model_{p}"] = sampled
        print(f"    {p}: max distance {float(dist.max()):.3f} deg")

    out = xr.Dataset(data, coords={
        "network": ("station_id", np.asarray(net, dtype=object)),
        "exposure": ("station_id", np.asarray(
            [L.EXPOSURE.get(n, "unclassified") for n in net], dtype=object)),
        "lat": ("station_id", lat), "lon": ("station_id", lon),
        "station_id": np.asarray(ids, dtype=object)})
    store.write("station_vs_models", out)


@stage("station_daily", ["station_daily"],
       "Daily means and maxima at stations, observed and modelled")
def s_station_daily(sess, store):
    """One pass over the hourly station store, producing everything the daily
    and extreme comparisons need.

    This exists because the analysis notebook was reading the same ~1.4 GB
    hourly archive three separate times -- once for daily means, once for
    maxima, once for the diurnal cycle -- and persisting none of it. Worse, it
    computed a private view that the stored station_vs_models never saw, so the
    station-quality problem could sit in one and not the other. Computing it
    here means one pass, one answer, and something verify.py can check.

    A >= 18 hour guard per day. A daily mean built from three hours is not a
    daily mean, and a daily MAXIMUM from three hours understates the true one
    and would bias the extreme comparison low.
    """
    obs = sess.stations()
    sids = [str(s) for s in obs.station_id.values]
    full = L._open(sess.root / "hdp" / "hdp_hourly.zarr")
    idx = np.isin([str(s) for s in full.station_id.values], sids)
    lat, lon = full.lat.values[idx], full.lon.values[idx]
    net = np.asarray([str(n) for n in full.network.values])[idx]

    def clean(da):
        """Strip everything but the shared dims.

        A member selection leaves a scalar `sim` coord behind, and two products
        carrying different scalar `sim` values cannot be merged into one Dataset
        -- xarray raises "conflicting values for variable 'sim'". Attributes go
        too, since they describe the source rather than the sample.
        """
        keep = {"station_id", "time", "month", "hour"}
        out = da.drop_vars([c for c in da.coords if c not in keep],
                           errors="ignore")
        out.attrs = {}
        return out

    cnt = obs.notnull().resample(time="1D").sum()
    with progress("  daily station means and maxima..."):
        o_mean, o_max = dask.compute(
            obs.resample(time="1D").mean().where(cnt >= 18),
            obs.resample(time="1D").max().where(cnt >= 18))
    print(f"    {o_mean.sizes['station_id']} stations x "
          f"{o_mean.sizes['time']} days")

    data = {"obs_mean": clean(o_mean), "obs_max": clean(o_max)}

    # Products at the same points. The distance tolerance scales with the grid:
    # on 0.25 deg the nearest centre is up to 0.18 deg from a point squarely
    # inside its cell, so a tighter fixed value would reject most stations.
    for p in PRODUCTS:
        if not store.exists(f"ctx_{p}_daily"):
            print(f"    ctx_{p}_daily absent; skipping")
            continue
        c = store.read(f"ctx_{p}_daily")
        at, dist = sample_grid_at_points(c["src"], lat, lon, ids=sids,
                                         max_dist_deg=0.20)
        data[f"model_{p}"] = clean(at.compute())
        if "era5" not in data:
            e, _ = sample_grid_at_points(c["era5"], lat, lon, ids=sids,
                                         max_dist_deg=0.20)
            data["era5"] = clean(e.compute())
        print(f"    {p}: max distance {float(dist.max()):.3f} deg")

    # Daily maxima from the products that publish one. LOCA2 never appears:
    # it has no maximum-wind variable at any cadence, which is itself the
    # verdict for the extreme-wind use case.
    for p in ("wrf-gcm", "wrf-era5"):
        try:
            da = sess.clip(L.open_product("wspd10max", p, "day", sess.root))
        except FileNotFoundError:
            print(f"    no wspd10max for {p} -- run "
                  "`fetch_local.py wspd10max --cadence day`")
            continue
        one, _ = reference_member(da)
        at, _ = sample_grid_at_points(one, lat, lon, ids=sids,
                                      max_dist_deg=0.20)
        data[f"max_{p}"] = clean(at.compute())

    out = xr.Dataset(data, coords={
        "network": ("station_id", np.asarray(net, dtype=object)),
        "exposure": ("station_id", np.asarray(
            [L.EXPOSURE.get(n, "unclassified") for n in net], dtype=object)),
        "lat": ("station_id", lat), "lon": ("station_id", lon)})
    out.attrs.update(min_hours_per_day=18, max_dist_deg=0.20,
                     note="observations already network- and quality-filtered "
                          "by localdata.open_hdp")
    store.write("station_daily", out)


@stage("station_diurnal", ["station_diurnal"],
       "Diurnal cycles at stations, observed and from gridded references")
def s_station_diurnal(sess, store):
    """Observed hour-of-day climatology per station, plus what the gridded
    references say at the same points.

    Split by exposure downstream rather than pooled: ridge-top and valley
    stations have OPPOSITE cycles -- valleys peak in the afternoon as mixing
    brings momentum down, ridges often peak at night when the surface layer
    decouples and the mast sits in the residual flow above it. A pooled cycle
    averages two physically different signals into one describing neither.

    No AE product is available hourly, so ERA5 and CONUS404 stand in for what a
    gridded field says here. That is the largest remaining gap in the review.
    """
    obs = sess.stations()
    sids = [str(s) for s in obs.station_id.values]
    full = L._open(sess.root / "hdp" / "hdp_hourly.zarr")
    idx = np.isin([str(s) for s in full.station_id.values], sids)
    lat, lon = full.lat.values[idx], full.lon.values[idx]
    net = np.asarray([str(n) for n in full.network.values])[idx]

    def clean(da):
        keep = {"station_id", "time", "month", "hour"}
        out = da.drop_vars([c for c in da.coords if c not in keep],
                           errors="ignore")
        out.attrs = {}
        return out

    with progress("  observed hour x month climatology..."):
        by_hm = obs.groupby("time.month").map(
            lambda g: g.groupby("time.hour").mean("time")).compute()
    data = {"obs": clean(by_hm)}
    print(f"    {dict(by_hm.sizes)}")

    for k, pth in (("era5", sess.root / "era5" / "era5_diurnal.zarr"),
                   ("conus404",
                    sess.root / "conus404" / "conus404_diurnal.zarr")):
        if not pth.exists():
            print(f"    {pth.name} absent -- run the derive script")
            continue
        d = xr.open_zarr(pth, consolidated=True).wspeed
        # A looser tolerance than elsewhere: CONUS404's diurnal store is on its
        # own 4 km curvilinear grid, and a station near the edge can sit further
        # from a centre than on the reference grid.
        at, _ = sample_grid_at_points(d, lat, lon, ids=sids, max_dist_deg=0.30)
        data[k] = clean(at.compute())

    # The AE products, if an hourly fetch exists. This is the ONLY way they can
    # join a diurnal comparison: everything else in the archive is monthly or
    # daily, and a daily mean has no hour-of-day structure left in it.
    for prod in PRODUCTS:
        try:
            src = sess.clip(L.open_product(sess.variable, prod, "1hr",
                                           sess.root))
        except FileNotFoundError:
            continue
        one, _ = reference_member(src)
        at, _ = sample_grid_at_points(one, lat, lon, ids=sids,
                                      max_dist_deg=0.20)
        with progress(f"  {prod} hour x month climatology..."):
            data[prod] = clean(
                at.groupby("time.month").map(
                    lambda g: g.groupby("time.hour").mean("time")).compute())
        print(f"    {prod}: hourly diurnal added")
    if not any(p in data for p in PRODUCTS):
        print("    no hourly AE product -- fetch one to let the products join:\n"
              "      fetch_local.py wspeed --cadence 1hr --products wrf-era5")

    out = xr.Dataset(data, coords={
        "network": ("station_id", np.asarray(net, dtype=object)),
        "exposure": ("station_id", np.asarray(
            [L.EXPOSURE.get(n, "unclassified") for n in net], dtype=object)),
        "lat": ("station_id", lat), "lon": ("station_id", lon)})
    out.attrs["hours"] = "UTC; local solar time is roughly UTC-8"
    store.write("station_diurnal", out)


@stage("station_pattern", ["station_pattern"],
       "Spatial pattern fidelity across station locations")
def s_station_pattern(sess, store):
    """Does a product explain the spatial variation of wind BETWEEN sites?

    Station spatial variance includes SITING -- a ridge-top mast and a sheltered
    field differ for reasons no gridded model can reproduce, and both may fall in
    the same cell. The achievable correlation is therefore well below 1, and the
    meaningful comparison is RELATIVE: does a km-scale product explain more of
    the between-site variance than the coarse reanalysis does? Per-network and
    per-exposure figures partly control for siting, since stations within a
    group are sited alike.
    """
    if not store.exists("station_vs_models"):
        raise FileNotFoundError(
            "station_vs_models not built -- run the `stations` stage first")
    stn = store.read("station_vs_models").compute()
    lat, lon = stn.lat.values, stn.lon.values
    ids = [str(i) for i in stn.station_id.values]
    net = np.asarray([str(n) for n in stn.network.values])

    fields = {p: stn[f"model_{p}"].mean("time") for p in PRODUCTS
              if f"model_{p}" in stn}

    # The reference at the same points, from a stored field. The distance
    # tolerance must scale with the target grid: on a 0.25 deg grid the nearest
    # centre is up to 0.18 deg away even for a point squarely inside it, so a
    # fixed 0.1 deg would flag most stations as outside.
    ref = store.read(f"ctx_{PRODUCTS[0]}_monthly")["era5"].mean("time").compute()
    dlat = float(np.abs(np.diff(ref.latitude.values)).mean())
    e_at, _ = sample_grid_at_points(ref, lat, lon, ids=ids,
                                    max_dist_deg=dlat * 0.75)
    fields["era5"] = e_at.drop_vars(
        [c for c in e_at.coords if c != "station_id"], errors="ignore")

    obs = stn["station"].mean("time")
    rows = []
    groups = [("ALL", np.ones(len(ids), bool))]
    groups += [(n, net == n) for n in pd.unique(net)
               if (net == n).sum() >= L.MIN_NETWORK_N]
    for gname, mask in groups:
        for src, fld in fields.items():
            st = pattern_metrics(fld.where(xr.DataArray(mask, dims="station_id")),
                                 obs.where(xr.DataArray(mask, dims="station_id")))
            st.update(source=src, network=gname, n_cells=int(mask.sum()))
            rows.append(st)
    df = pd.DataFrame(rows)
    # Coerce the index levels to plain str. pd.unique over a numpy string array
    # yields np.str_, and mixing those with the literal "ALL" gives an object
    # array of mixed native types that zarr cannot encode.
    for c in ("source", "network"):
        df[c] = df[c].map(str)
    print(df[df.network == "ALL"][["source", "pattern_corr", "std_ratio",
                                  "mean_ratio"]].round(3).to_string(index=False))
    store.write("station_pattern", xr.Dataset.from_dataframe(
        df.set_index(["source", "network"])))


ORDER = ["elevation", "era5_agg", "monthly", "daily", "seasonal", "trends",
         "multimodel", "conus404", "conus404_native", "extremes",
         "pattern", "stations", "station_daily", "station_diurnal",
         "station_pattern"]


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dest", default=None)
    ap.add_argument("--variable", default=VARIABLE)
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--time-slice", nargs=2, default=None,
                    metavar=("START", "END"),
                    help="restrict every source to this window. Use with "
                         "--dest to keep a short-window run beside the full one.")
    ap.add_argument("--station-networks", default="analysis",
                    choices=["analysis", "all"],
                    help="'all' lets the window-dependent exclusions back in "
                         "(CWOP records are contemporaneous after ~2005)")
    ap.add_argument("--min-months", type=int, default=None,
                    help="minimum station record length within the window "
                         f"(default {L.MIN_MONTHS})")
    a = ap.parse_args()

    if a.list:
        print(f"{'stage':<18} {'outputs':<48} description")
        print("-" * 100)
        for n in ORDER:
            sp = STAGES[n]
            print(f"{n:<18} {','.join(sp['outputs'])[:46]:<48} {sp['desc']}")
        return 0

    import dask
    dask.config.set(scheduler="threads", num_workers=a.workers)

    root = Path(a.root)
    dest = Path(a.dest) if a.dest else root / "metrics"
    tslice = tuple(a.time_slice) if a.time_slice else None
    if tslice:
        yrs = (pd.Timestamp(tslice[1]) - pd.Timestamp(tslice[0])).days / 365.25
        if yrs < 10:
            print(f"NOTE: a {yrs:.0f}-year window. agreement_metrics warns "
                  "below ten years -- with few realizations per calendar month, "
                  "seasonal and interannual metrics largely measure noise. "
                  "Level and distribution metrics are unaffected.\n")
        if dest.name == "metrics":
            print("WARNING: writing a windowed run into the default `metrics` "
                  "directory will overwrite the full-period results. "
                  "Use --dest.\n")
    store = Store(dest, overwrite=a.overwrite, dry_run=a.dry_run)
    sess = Session(a.variable, root, time_slice=tslice,
                   station_networks=a.station_networks,
                   min_months=a.min_months)

    wanted = a.stages or ORDER
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        print(f"unknown stages: {unknown}\nchoose from: {ORDER}")
        return 1
    wanted = [s for s in ORDER if s in wanted and s not in a.skip]

    print(f"source      : {root}")
    print(f"destination : {dest}")
    print(f"variable    : {a.variable}")
    print(f"window      : {tslice[0]} .. {tslice[1]}" if tslice
          else "window      : full record")
    print(f"networks    : {a.station_networks}")
    print(f"stages      : {wanted}")
    print(f"mode        : {'DRY RUN' if a.dry_run else 'write'}"
          f"{' (overwrite)' if a.overwrite else ''}\n")

    results = {}
    for name in wanted:
        spec = STAGES[name]
        missing = [o for o in spec["outputs"] if not store.exists(o)]
        if spec["outputs"] and not missing and not a.overwrite:
            print(f"[{name}] SKIP - all outputs present (--overwrite to rebuild)")
            results[name] = "skipped"
            continue
        print(f"[{name}] {spec['desc']}")
        t = time.time()
        try:
            spec["fn"](sess, store)
            results[name] = "ok"
            print(f"[{name}] done in {time.time() - t:.0f}s\n")
        except FileNotFoundError as e:
            # An input that has not been fetched yet is a normal state of an
            # incrementally built archive, not a broken pipeline.
            results[name] = "skipped (input missing)"
            print(f"[{name}] SKIP - {e}\n")
        except Exception as e:
            results[name] = f"FAILED: {type(e).__name__}"
            print(f"[{name}] FAILED after {time.time() - t:.0f}s: "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
            print()

    print("=" * 62)
    for k, v in results.items():
        print(f"  {k:<18} {v}")
    return 1 if any(str(v).startswith("FAILED") for v in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())