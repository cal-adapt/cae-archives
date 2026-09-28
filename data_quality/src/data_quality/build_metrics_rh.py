#!/usr/bin/env python
"""
Build relative-humidity metrics from the local archive.

    python build_metrics_rh.py --list
    python build_metrics_rh.py                       # all stages
    python build_metrics_rh.py --stages stations thresholds --overwrite

Reads /shared/data, writes /shared/data/metrics_hurs. Resumable: a stage whose
outputs exist is skipped unless --overwrite.

WHY THIS IS SEPARATE FROM build_metrics.py. Not duplication for its own sake --
four things differ, and each would have become a branch inside every stage:

  METRIC. Wind speed is unbounded and its error is multiplicative, so `ratio` is
    the right summary. RH is bounded on [0, 100] and its error is additive, so a
    ratio is constrained by the observation itself: at observed 90% no model can
    exceed 1.11 however wrong it is. Everything here reports BIAS IN PERCENTAGE
    POINTS via rh_analysis.bounded_metrics.

  TAIL. For wind the extremes that matter are the strong days; for humidity they
    are the DRY hours -- the afternoon minimum that sets fire danger. Quantile
    comparisons use p10, and threshold counts use `<=`.

  PRODUCT COVERAGE. LOCA2 publishes hursmax and hursmin and NO mean RH at any
    cadence, so it cannot enter a comparison of means at all -- but it supports
    the fire-weather variable directly. WRF is the reverse: hourly `rh`, no
    published extremes, so its daily minimum is computed at ingest by
    fetch_local_daily.py. Stages declare which products they can serve, and skip
    for that REASON rather than reporting a missing fetch that can never arrive.

  STATION SCREENING. The wind screen's central test is a low-mean floor, aimed
    at an anemometer stuck at zero. A humidity sensor fails the other way --
    condensation on the element reads ~100% indefinitely -- so the screen is a
    mean RANGE plus a variance test. See rh_analysis.station_quality.

Requires: numpy, pandas, xarray, zarr, dask.
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

from . import humidity as H
from . import localdata as L
from . import rh_analysis as R
from .grids import (crop_era5_to_source, coverage_fraction, build_labels,
                   bin_to_era5, member_dim, progress, sample_grid_at_points)

ROOT = Path("/shared/data")
DEST = ROOT / "metrics_hurs"
ZARR_FORMAT = 2
VARIABLE = "hurs"

# WRF publishes hourly `rh`; its daily extremes are computed at ingest by
# fetch_local_daily.py. LOCA2 publishes the extremes directly and no mean.
GRIDDED = ["wrf-era5", "wrf-gcm"]
STAGES: dict[str, dict] = {}


def stage(name: str, outputs: list[str], desc: str):
    def deco(fn: Callable):
        STAGES[name] = {"fn": fn, "outputs": outputs, "desc": desc}
        return fn
    return deco


class Store:
    """Named zarr datasets under one local prefix."""

    def __init__(self, dest: Path, overwrite=False, dry_run=False):
        self.dest = Path(dest)
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.dest.mkdir(parents=True, exist_ok=True)

    def path(self, name): return self.dest / f"{name}.zarr"
    def exists(self, name): return self.path(name).exists()
    def read(self, name): return xr.open_zarr(self.path(name), consolidated=True)

    def write(self, name: str, ds: xr.Dataset) -> None:
        target = self.path(name)
        if self.dry_run:
            print(f"    would write {target}")
            return
        # Refuse an all-NaN write: a silently empty dataset costs far more to
        # discover three stages later than to reject now.
        for v, da in ds.data_vars.items():
            if da.dtype.kind == "f" and da.size and not bool(da.notnull().any()):
                raise ValueError(f"refusing to write '{name}': "
                                 f"variable '{v}' is entirely NaN")
        ds = ds.copy()
        for v in ds.variables:
            ds[v].encoding = {}
            if ds[v].dtype.kind in ("T", "U"):
                ds[v] = ds[v].astype(object)
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


def strip_coords(da, keep=("station_id", "time", "month", "hour")):
    """Drop every coordinate except the shared dimensions.

    Arrays sampled from a grid carry the GRIDCELL's lat/lon; arrays built from
    observations carry the STATION's. They differ by up to a grid spacing, so a
    Dataset holding both is ambiguous and xarray refuses to merge it. Station
    coordinates are attached once at the end, which is correct because every
    variable here is indexed by station.

    Scalar leftovers go too -- a member selection leaves a `sim` coord behind,
    and two products with different values cannot be merged.
    """
    out = da.drop_vars([c for c in da.coords if c not in keep],
                       errors="ignore")
    out.attrs = {}
    return out


def span_attrs(a) -> dict:
    t = a.time.values
    return {"time_start": str(t[0])[:10], "time_end": str(t[-1])[:10],
            "n_steps": int(a.sizes["time"]),
            "span_years": round(float((t[-1] - t[0])
                                      / np.timedelta64(365, "D")), 2)}


class Session:
    """Shared readers, opened once."""

    def __init__(self, root=ROOT, time_slice=None, networks="analysis"):
        self.root = root
        self.time_slice = time_slice
        self.networks = networks
        self._obs = None
        self._grids = {}

    def clip(self, da):
        return da if self.time_slice is None else da.sel(
            time=slice(*self.time_slice))

    def obs(self):
        """Screened hourly station RH, opened once -- it is a 1.4 GB read."""
        if self._obs is None:
            self._obs = L.open_hdp_rh(root=self.root, networks=self.networks,
                                      time_slice=self.time_slice)
        return self._obs

    def station_meta(self):
        """Per-station metadata, looked up BY ID.

        Not by boolean mask. `np.isin` returns a mask in the FULL store's
        order, while the screened station list is in whatever order survived
        the network, QC and quality filters -- so masking gives a correctly
        sized array whose entries belong to different stations. The failure is
        silent and total: every class-level result is then attributed to the
        wrong station type, with airports appearing at 997 m and ridge-top RAWS
        at 155 m.
        """
        obs = self.obs()
        sids = [str(s) for s in obs.station_id.values]
        full = L._open(L._hdp_store(self.root, "hurs"))

        fid = [str(s) for s in full.station_id.values]
        pos = {sid: i for i, sid in enumerate(fid)}
        missing = [s for s in sids if s not in pos]
        if missing:
            raise KeyError(f"{len(missing)} screened stations absent from the "
                           f"store, e.g. {missing[:3]}")
        take = np.array([pos[s] for s in sids])

        lat = np.asarray(full.lat.values)[take]
        lon = np.asarray(full.lon.values)[take]
        net = np.asarray([str(n) for n in full.network.values])[take]
        elev = (np.asarray(full.elevation.values)[take]
                if "elevation" in full.coords
                else np.full(len(sids), np.nan))
        return dict(
            ids=sids, lat=lat, lon=lon, elevation=elev,
            network=np.asarray(net, dtype=object),
            exposure=np.asarray([L.EXPOSURE.get(n, "unclassified")
                                 for n in net], dtype=object))

    def daily_obs(self):
        """Observed daily min / max / mean, with an 18-hour guard.

        The guard matters more than it does for wind: RH's daily MINIMUM falls
        in a narrow afternoon window, so a day sampled only at night yields a
        "minimum" tens of points too high -- and it looks like a plausible value
        rather than an obvious gap.
        """
        return R.daily_stats(self.obs(), min_hours=18)

    def product_daily(self, product: str) -> xr.Dataset | None:
        """Daily min/max/mean for one gridded product, from whichever source
        publishes them.

        LOCA2 publishes hursmax/hursmin directly. WRF publishes only hourly, so
        fetch_local_daily.py reduces it at ingest -- reading 268 GB to keep 22.
        """
        if product == "loca2":
            try:
                mx = self.clip(L.open_product("hursmax", "loca2", "day",
                                              self.root))
                mn = self.clip(L.open_product("hursmin", "loca2", "day",
                                              self.root))
            except FileNotFoundError:
                return None
            return xr.Dataset({"hurs_min": mn, "hurs_max": mx})
        p = self.root / "hurs" / f"{product}_dayagg.zarr"
        if p.exists():
            return self.clip(xr.open_zarr(p, consolidated=True))
        return None

    def product_daily_at_stations(self, product: str) -> xr.Dataset | None:
        """Daily min/max/mean already indexed by station_id.

        Reduced from the STATION-POINT hourly store. For a station comparison
        this is the same number as reducing the full domain and then sampling --
        the cells are identical, only the ones nobody asks about are missing --
        and it is ~21 GB rather than 1.6 TB.

        Returned separately from product_daily because the two have DIFFERENT
        DIMENSIONS: this is (station_id, time), that is a grid. A caller must
        not sample this one, and conflating them would produce a confident
        KeyError deep in a KDTree.
        """
        sp = self.root / "hurs" / f"{product}_stations_1hr.zarr"
        if not sp.exists():
            return None
        d = self.clip(xr.open_zarr(sp, consolidated=True)["hurs"])
        md = member_dim(d)
        if md:
            d = d.isel({md: 0})
        print(f"    {product}: reducing station-point hourly "
              f"({d.sizes['station_id']} stations, no domain fetch needed)")
        return xr.Dataset({"hurs_min": d.resample(time="1D").min(),
                           "hurs_max": d.resample(time="1D").max(),
                           "hurs_mean": d.resample(time="1D").mean()})


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------


@stage("era5_ref", ["era5_daily_ref"],
       "Derived ERA5 daily RH: the reference the gridded stages use")
def s_era5_ref(sess, store):
    """ERA5 publishes no RH, so it is derived from t2 and d2.

    Referenced to LIQUID WATER at all temperatures -- the WMO convention for
    surface observations and what the station networks report. An ice-referenced
    formula gives a HIGHER humidity for the same air below freezing, about 5 pp
    at -10 C, and would put a spurious cold-season bias into every comparison.
    """
    try:
        mn = sess.clip(L.era5_rh("day", "min", sess.root))
        mx = sess.clip(L.era5_rh("day", "max", sess.root))
        me = sess.clip(L.era5_rh("day", "mean", sess.root))
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(
            f"{e}\nERA5 publishes no RH. Derive it:\n"
            "  python src/fetch_era5.py t2 d2\n"
            "  python src/derive_era5.py --variable hurs") from None

    out = xr.Dataset({"hurs_min": mn, "hurs_max": mx, "hurs_mean": me})
    out["mid_minus_mean"] = (mx + mn) / 2.0 - me
    out.attrs.update(span_attrs(mn), derivation=H.__doc__.splitlines()[1][:80]
                     if H.__doc__ else "see humidity.py")
    print(f"    domain mean daily min {float(mn.mean()):.1f} %")
    print(f"    midrange minus mean   {float(out.mid_minus_mean.mean()):+.2f} pp")
    store.write("era5_daily_ref", out)


@stage("daily_min", [f"metrics_{p}_daily" for p in GRIDDED + ["loca2"]],
       "Daily minimum RH against ERA5 -- the fire-weather variable")
def s_daily_min(sess, store):
    """Per-cell agreement on the DAILY MINIMUM, the statistic fire danger uses.

    Bias in percentage points, never a ratio: see the module docstring.
    """
    if not store.exists("era5_daily_ref"):
        raise FileNotFoundError("run the era5_ref stage first")
    ref = store.read("era5_daily_ref")

    for product in ["loca2"] + GRIDDED:
        print(f"  {product}")
        # Deliberately the GRIDDED source only. This stage bins onto the ERA5
        # grid, so a station-indexed array is not a substitute however cheap it
        # was to produce -- the station-point store covers ~1,900 cells of
        # ~120,000 and says nothing about the rest of the domain.
        src = sess.product_daily(product)
        if src is None:
            print(f"    no gridded daily store for {product}. LOCA2 needs "
                  "hursmax/hursmin;\n    WRF needs fetch_local_daily.py "
                  "(the station-point store cannot serve a gridded stage)")
            continue
        if "hurs_min" not in src:
            print(f"    {product} has {list(src.data_vars)}, no hurs_min")
            continue

        s = src["hurs_min"]
        md = member_dim(s)
        if md:
            s = s.isel({md: 0})
        crop = crop_era5_to_source(ref["hurs_min"], s)
        kind, labels = build_labels(s, crop)
        binned = bin_to_era5(s, kind, labels, crop, how="mean")
        a, b = xr.align(binned, crop, join="inner")
        if a.sizes["time"] == 0:
            print("    no temporal overlap")
            continue

        with progress(f"    {product} daily minimum..."):
            m = R.bounded_metrics(a, b,
                                  synchronized=(product == "wrf-era5")).compute()
        m.attrs.update(product=product, statistic="daily minimum",
                       **span_attrs(a))
        print(f"    median bias {float(np.nanmedian(m.bias)):+.2f} pp, "
              f"p10 {float(np.nanmedian(m.p10_diff)):+.2f} pp")
        store.write(f"metrics_{product}_daily", m)
        store.write(f"ctx_{product}_daily",
                    xr.Dataset({"src": a, "era5": b}))


@stage("stations", ["station_rh"],
       "Observed and modelled daily RH extremes at station locations")
def s_stations(sess, store):
    """Everything the station comparisons need, in one pass.

    The STATISTIC has to match on both sides. A daily mean RH sits 15-25 pp above
    a daily minimum in this domain, so mixing them would read as a large moist
    bias that is purely an artefact of comparing different quantities. Only
    minima are put beside minima here.
    """
    meta = sess.station_meta()
    with progress("  observed daily statistics..."):
        obs = sess.daily_obs().compute()
    print(f"    {obs.sizes['station_id']} stations x {obs.sizes['time']} days")

    data = {f"obs_{v}": strip_coords(obs[v]) for v in obs.data_vars}
    for product in ["loca2"] + GRIDDED:
        # Two routes, and they differ in DIMENSION rather than in value.
        #
        #   station-point   already indexed by station_id -- reduced from the
        #                   hourly fetch at station locations. Nothing to
        #                   sample, and it needs no domain-wide store.
        #   gridded         a grid that has to be sampled at the points.
        #
        # The station-point route is preferred because it exists for WRF while
        # the gridded one requires a 268 GB per-member fetch. Sampling an
        # already-station-indexed array would fail inside the KDTree with an
        # error that says nothing about the cause.
        at_stn = sess.product_daily_at_stations(product)
        if at_stn is not None and "hurs_min" in at_stn:
            # All three daily statistics, not only the minimum. The minimum is
            # the fire-weather variable, but the mean and maximum answer
            # different questions -- whether the model is wet or dry on average,
            # and whether it saturates at night -- and they cost nothing extra
            # once the hourly has been read.
            common = [i for i in meta["ids"]
                      if i in {str(x) for x in at_stn.station_id.values}]
            if not common:
                print(f"    {product}: station-point store shares no stations")
                continue
            got = []
            for stat, prefix in (("hurs_min", "min"), ("hurs_max", "max"),
                                 ("hurs_mean", "mean")):
                if stat not in at_stn:
                    continue
                v = (at_stn[stat].sel(station_id=common)
                     .reindex(station_id=meta["ids"]).compute())
                data[f"{prefix}_{product}"] = strip_coords(v)
                got.append(f"{prefix} {float(v.mean()):.1f}%")
            print(f"    {product}: {len(common)} of {len(meta['ids'])} "
                  f"stations, " + ", ".join(got))
            continue

        src = sess.product_daily(product)
        if src is None or "hurs_min" not in src:
            print(f"    {product}: no daily minimum available. LOCA2 needs "
                  "hursmax/hursmin;\n      WRF needs either "
                  "fetch_stations_1hr.py (cheap) or fetch_local_daily.py")
            continue
        got, dist = [], None
        for stat, prefix in (("hurs_min", "min"), ("hurs_max", "max"),
                             ("hurs_mean", "mean")):
            if stat not in src:
                continue
            s = src[stat]
            md = member_dim(s)
            if md:
                s = s.isel({md: 0})
            at, dist = sample_grid_at_points(s, meta["lat"], meta["lon"],
                                             ids=meta["ids"], max_dist_deg=0.20)
            v = at.compute()
            data[f"{prefix}_{product}"] = strip_coords(v)
            got.append(f"{prefix} {float(v.mean()):.1f}%")
        print(f"    {product}: sampled from grid ({', '.join(got)}), "
              f"max distance {float(dist.max()):.3f} deg")

    try:
        e5 = sess.clip(L.era5_rh("day", "min", sess.root))
        at, _ = sample_grid_at_points(e5, meta["lat"], meta["lon"],
                                      ids=meta["ids"], max_dist_deg=0.30)
        data["min_era5"] = strip_coords(at.compute())
    except (FileNotFoundError, KeyError):
        print("    ERA5 daily minimum absent")

    out = xr.Dataset(data, coords={
        "network": ("station_id", meta["network"]),
        "exposure": ("station_id", meta["exposure"]),
        "lat": ("station_id", meta["lat"]), "lon": ("station_id", meta["lon"])})
    out.attrs.update(min_hours_per_day=18, statistic="daily minimum",
                     note="observed and modelled daily MINIMA only; a daily "
                          "mean is a different statistic and sits 15-25 pp "
                          "higher in this domain")
    store.write("station_rh", out)


@stage("thresholds", ["fire_thresholds"],
       "Fire-threshold frequencies and detection skill at stations")
def s_thresholds(sess, store):
    """Does the model flag the right days, not just carry the right mean?

    A bias says how far the average is off. This says whether a threshold-based
    product driven by the field would have raised the alarm on the days that
    mattered. The two can disagree completely: a model with a small moist bias
    and too little variance can miss most critical hours while looking almost
    unbiased.
    """
    if not store.exists("station_rh"):
        raise FileNotFoundError("run the stations stage first")
    stn = store.read("station_rh").compute()
    expo = np.asarray([str(e) for e in stn.exposure.values])
    obs = stn["obs_hurs_min"]

    rows = []
    for k in [v for v in stn.data_vars if v.startswith("min_")]:
        name = k.replace("min_", "")
        a, b = xr.align(obs, stn[k], join="inner")
        for group in ["all"] + [c for c in ("airport", "vegetated", "irrigated")
                                if (expo == c).sum() >= 10]:
            sel = (np.ones(a.sizes["station_id"], bool) if group == "all"
                   else expo == group)
            ov = a.isel(station_id=np.where(sel)[0]).values.ravel()
            mv = b.isel(station_id=np.where(sel)[0]).values.ravel()
            for thr in (R.THRESHOLDS["critical"], R.THRESHOLDS["elevated"]):
                sk = R.dry_hours_skill(mv, ov, thr)
                rows.append({"source": name, "group": group, **sk})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no modelled minima to score")
    print(df[["source", "group", "threshold", "obs_rate_%", "mod_rate_%",
              "POD", "FAR"]].round(3).to_string(index=False))
    store.write("fire_thresholds",
                xr.Dataset.from_dataframe(df.set_index(["source", "group",
                                                        "threshold"])))


@stage("station_diurnal", ["station_diurnal"],
       "Diurnal cycles at stations, observed and modelled")
def s_station_diurnal(sess, store):
    """Hour-of-day climatology per station.

    For humidity this is not a supplementary section. RH swings tens of points
    between a moist night and a dry afternoon, and every operationally relevant
    statistic lives in that swing -- the daily minimum is a point on this curve.

    The AE products can join because fetch_local_daily.py kept the full hourly
    series AT STATION POINTS, which is 0.76% of the domain's cells and a few GB,
    while discarding it everywhere it would only be read once and averaged.
    """
    meta = sess.station_meta()
    with progress("  observed hour x month climatology..."):
        by_hm = (sess.obs().groupby("time.month")
                 .map(lambda g: g.groupby("time.hour").mean("time")).compute())
    data = {"obs": strip_coords(by_hm)}
    print(f"    {dict(by_hm.sizes)}")

    for product in GRIDDED:
        p = sess.root / "hurs" / f"{product}_stations_1hr.zarr"
        if not p.exists():
            print(f"    {product}: no station-point hourly. Run\n"
                  f"      python src/fetch_local_daily.py hurs --products {product}")
            continue
        d = xr.open_zarr(p, consolidated=True)["hurs"]
        md = member_dim(d)
        if md:
            d = d.isel({md: 0})
        d = sess.clip(d)
        common = [s for s in meta["ids"]
                  if s in {str(x) for x in d.station_id.values}]
        d = d.sel(station_id=common)
        with progress(f"  {product} hour x month..."):
            got = (d.groupby("time.month")
                   .map(lambda g: g.groupby("time.hour").mean("time"))
                   .compute())
        # Drop the sampled array's own lat/lon before it joins the Dataset.
        #
        # They are the GRIDCELL's coordinates, not the station's, and the two
        # differ by up to a grid spacing -- 0.026 deg here, which is exactly
        # what cell_distance_deg records. Carrying both makes the merge
        # ambiguous and xarray refuses. The station coordinates are the ones
        # attached below, since every other variable is indexed by station;
        # compat="override" would have silently kept whichever came first.
        data[product] = strip_coords(got)

    for k, pth in (("era5", sess.root / "era5" / "era5_hurs_diurnal.zarr"),):
        if not pth.exists():
            print(f"    {pth.name} absent")
            continue
        dd = xr.open_zarr(pth, consolidated=True).wspeed \
            if "wspeed" in xr.open_zarr(pth, consolidated=True) \
            else xr.open_zarr(pth, consolidated=True).hurs
        at, _ = sample_grid_at_points(dd, meta["lat"], meta["lon"],
                                      ids=meta["ids"], max_dist_deg=0.30)
        data[k] = strip_coords(at.compute())

    out = xr.Dataset(data, coords={
        "network": ("station_id", meta["network"]),
        "exposure": ("station_id", meta["exposure"]),
        "elevation": ("station_id", meta["elevation"]),
        "lat": ("station_id", meta["lat"]), "lon": ("station_id", meta["lon"])})
    out.attrs["hours"] = "UTC; local solar time is roughly UTC-8"
    store.write("station_diurnal", out)


@stage("midrange", ["midrange_bias"],
       "What the LOCA2 midrange costs, measured rather than assumed")
def s_midrange(sess, store):
    """(max + min)/2 against the true daily mean, observed and modelled.

    LOCA2 publishes only the daily extremes, so any use of it as a mean rests on
    the midrange -- which is NOT one. RH's diurnal cycle is asymmetric: a long
    moist night and a short dry afternoon, so the time-mean sits ABOVE the
    midpoint by an amount that grows with the diurnal range. The bias is
    therefore largest in dry inland regimes, exactly where RH matters most, and
    would read as a regional dry bias in LOCA2 that is an artefact of the
    statistic.

    Quantified here from observations AND from a model that publishes all three,
    so the finding is checked rather than asserted.
    """
    with progress("  observed midrange bias..."):
        obs = sess.daily_obs().compute()
    mm = obs.mid_minus_mean.values.ravel()
    mm = mm[np.isfinite(mm)]
    rng = obs.hurs_range.values.ravel()
    rng = rng[np.isfinite(rng)]
    print(f"    observations: median {np.median(mm):+.2f} pp, "
          f"p10 {np.quantile(mm, .10):+.2f}, p90 {np.quantile(mm, .90):+.2f}")

    rows = [{"source": "stations", "median_pp": float(np.median(mm)),
             "mean_pp": float(mm.mean()),
             "p10_pp": float(np.quantile(mm, .10)),
             "p90_pp": float(np.quantile(mm, .90)),
             "median_range_pp": float(np.median(rng))}]

    for product in GRIDDED:
        src = sess.product_daily(product)
        if src is None or "mid_minus_mean" not in src:
            continue
        v = src["mid_minus_mean"]
        md = member_dim(v)
        if md:
            v = v.isel({md: 0})
        with progress(f"  {product} midrange bias..."):
            v = v.compute().values.ravel()
        v = v[np.isfinite(v)]
        rows.append({"source": product, "median_pp": float(np.median(v)),
                     "mean_pp": float(v.mean()),
                     "p10_pp": float(np.quantile(v, .10)),
                     "p90_pp": float(np.quantile(v, .90)),
                     "median_range_pp": np.nan})
        print(f"    {product}: median {np.median(v):+.2f} pp")

    df = pd.DataFrame(rows).set_index("source")
    print(df.round(2).to_string())
    store.write("midrange_bias", xr.Dataset.from_dataframe(df))


ORDER = ["era5_ref", "daily_min", "stations", "thresholds",
         "station_diurnal", "midrange"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dest", default=None)
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--time-slice", nargs=2, default=None,
                    metavar=("START", "END"))
    ap.add_argument("--networks", default="analysis", choices=["analysis", "all"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for n in ORDER:
            s = STAGES[n]
            print(f"{n:<18} {s['desc']}")
            print(f"{'':18} -> {', '.join(s['outputs'])}")
        return 0

    dask.config.set(scheduler="threads", num_workers=a.workers)
    root = Path(a.root)
    dest = Path(a.dest) if a.dest else root / "metrics_hurs"
    tslice = tuple(a.time_slice) if a.time_slice else None

    print(f"source      : {root}")
    print(f"destination : {dest}")
    print(f"variable    : {VARIABLE}  (bias in percentage points, not ratio)")
    print(f"window      : {tslice[0] + ' .. ' + tslice[1] if tslice else 'full record'}")
    print(f"stages      : {a.stages or ORDER}")
    print(f"mode        : {'DRY RUN' if a.dry_run else 'write'}\n")

    store = Store(dest, overwrite=a.overwrite, dry_run=a.dry_run)
    sess = Session(root, time_slice=tslice, networks=a.networks)
    results = {}

    for name in (a.stages or ORDER):
        if name not in STAGES:
            print(f"unknown stage '{name}'; see --list")
            return 1
        spec = STAGES[name]
        done = all(store.exists(o) for o in spec["outputs"])
        if done and not a.overwrite:
            print(f"[{name}] up to date")
            results[name] = "up to date"
            continue
        print(f"[{name}] {spec['desc']}")
        t0 = time.time()
        try:
            spec["fn"](sess, store)
            results[name] = "ok"
            print(f"[{name}] done in {time.time() - t0:.0f}s\n")
        except FileNotFoundError as e:
            results[name] = "skipped (input missing)"
            print(f"[{name}] SKIP - {e}\n")
        except Exception as e:
            results[name] = f"FAILED {type(e).__name__}"
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            print()

    print("=" * 62)
    for n, r in results.items():
        print(f"  {n:<18} {r}")
    return 1 if any(str(r).startswith("FAILED") for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
