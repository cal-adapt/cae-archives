#!/usr/bin/env python
"""
Read the local archive under /shared/data. The only module that knows the
on-disk layout; nothing downstream should open a path itself.

    from data_quality.localdata import (
        open_product, open_era5, era5_speed, open_conus404, conus404_speed,
        open_hdp, open_stations, open_elevation, open_bbox, members, describe,
    )

    wrf  = open_product("wspeed", "wrf-gcm", "mon")      # all 5 models
    e5   = era5_speed(freq="MS")                          # scalar, hourly then averaged
    obs  = open_hdp(qc=True).resample(time="MS").mean()

Layout:

    /shared/data/
        {variable}/{product}_{cadence}.zarr    fetch_local.py
        grids/bbox.json, wrf_elevation.nc      fetch_geometry.py
        era5/era5_hourly.zarr                  fetch_era5.py
        conus404/conus404_hourly.zarr          fetch_conus404.py
        hdp/hdp_hourly.zarr, stations.csv      fetch_hdp.py

SPEED IS FORMED HOURLY, THEN AVERAGED. This is the one rule that must not be got
wrong, and it is why the reference stores are kept as raw hourly components.
Averaging u and v first and taking the magnitude afterwards gives the VECTOR
mean, about 20% lower in this domain -- a bias comparable in size to the model
discrepancies under study. `era5_speed` and `conus404_speed` always take the
magnitude before any resampling.

NO CACHING. A monthly mean over 14 GB of local hourly data takes about a minute,
so there is nothing worth caching. The pipeline's old `_era5_*` aggregation
caches were keyed on (product, cadence) but not on the time window, which made a
changed window invisible: the stale object was reused, xr.align(join="inner")
quietly intersected the source down to it, and the run finished with no error
and the wrong answer. Recomputing removes that failure mode rather than
guarding it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/shared/data")
MEMBER_DIMS = ("sim", "simulation", "member_id", "member")

# Plausibility bounds for 10 m scalar wind speed, matching hdp_stations.qc_wind.
#   lower: a scalar magnitude cannot be negative
#   upper: the strongest reliably measured surface wind on Earth is ~113 m/s;
#          75 is far beyond anything California produces, so it removes
#          sentinels without touching real extremes
WIND_MIN, WIND_MAX = 0.0, 75.0

# Station networks by SITING REGIME. The station comparison rests on agreement
# being ordered by exposure rather than by model -- airports near 1.0, ridge-top
# RAWS high, sheltered irrigated CIMIS high, with the SIGN of the error flipping
# between them. That argument needs stations grouped by how they are sited, not
# by who operates them.
# Classes are named for SHELTER, not for altitude. That is a correction from an
# earlier version, and it was the data that forced it: measured observed means
# are airports 2.90 m/s, RAWS 2.32, CIMIS 1.96. RAWS records LESS wind than flat
# open ground, so it cannot be treated as an exposed class however many of its
# masts stand on ridges.
#
# The reason is what RAWS is FOR. It exists to represent fuel conditions for
# fire behaviour, so masts go where the vegetation is representative -- clearings
# within forest, on slopes -- rather than on bare crests. Radio line-of-sight
# puts some high, but canopy shelter dominates the measurement. Its native
# anemometer height is 6.1 m, standardised upward to 10 m by HDP, which is a
# modelled adjustment and may contribute too.
#
# The consequence for interpretation is sharp: only `airport` is unobstructed,
# so only `airport` can support a statement about model quality. Every other
# class confounds model error with shelter, and nothing in this archive
# separates them.
EXPOSURE = {
    "ASOSAWOS": "airport",      # flat, open, the closest thing to a cell average
    "OtherISD": "airport",
    "NCAWOS": "airport",
    "RAWS": "vegetated",        # fire weather: fuel-representative, canopy-sheltered
    "HPWREN": "vegetated",
    "CIMIS": "irrigated",       # agricultural: irrigated fields, crop-sheltered
    "NOS-PORTS": "sea_level",
    "NOS-NWLON": "sea_level",
    "CRN": "reference",         # climate reference network, sited to a standard
    "HADS": "mixed",            # hydrologic, often valley bottoms
    "CNRFC": "mixed",
    "CAHYDRO": "mixed",
    "CDEC": "mixed",
}

# Networks large enough to carry a per-network statistic. Below roughly 20
# stations a median is one or two sites, and the spread across networks -- which
# is the signal -- cannot be separated from sampling noise.
MIN_NETWORK_N = 20

# --- Station quality thresholds ---------------------------------------------
# A station whose observed mean is a fraction of a m/s across years of record is
# a FAILED ANEMOMETER, not a calm site: a real location still has windy days.
# qc_wind cannot catch this -- it masks values outside [0, 75] m/s, and a stuck
# sensor reporting 0.0 sits inside that range.
#
# These matter far more than their number suggests, because a station is the
# DENOMINATOR of every model/observation ratio. One station observing
# 0.004 m/s produces a ratio of 745 and dominates any mean. Medians survive it,
# which is why summary tables can look sensible while scatter plots do not --
# an argument for excluding them rather than trusting the median to hide them.
MIN_OBS_MEAN = 0.5      # m/s; below this the record is not wind
MIN_MONTHS = 60         # 5 years; below this a climatology is mostly noise
MIN_UNIQUE = 20         # distinct values; a stuck sensor has almost none

# Station-id prefixes excluded as a GROUP rather than one at a time.
#   RAWS_TR -- 73 stations with a non-standard id scheme (TR###, against
#              NWS-style LYEC1/FMOC1 elsewhere) and a median observed mean of
#              0.83 m/s against 2.16 for the rest of RAWS. Ridge-top masts
#              should read HIGHER than average, not a third; a 2.6x gap across
#              a whole subgroup is a systematic difference in convention, unit
#              or sensor generation, not siting.
EXCLUDE_PREFIXES = ("RAWS_TR",)

# Excluded from analysis, not from the archive.
#   CWOP  -- citizen weather. Every station starts 2003-2011, so against a
#            1980-2014 window each contributes a few years, below the ten-year
#            floor agreement_metrics warns at. Siting is uncontrolled and has no
#            coherent exposure class, and at ~380 stored it is large enough to
#            dominate any pooled median while adding scatter rather than a
#            category.
#   *WFO  -- NWS forecast-office collections (SGXWFO, LOXWFO, HNXWFO, MTRWFO):
#            heterogeneous, no documented siting standard.
EXCLUDE_ANALYSIS = ("CWOP", "SGXWFO", "LOXWFO", "HNXWFO", "MTRWFO",
                    "VCAPCD", "SHASAVAL")


def _open(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- has its fetcher been run?")
    return xr.open_zarr(path, consolidated=True)


def member_dim(da) -> str | None:
    return next((d for d in MEMBER_DIMS if d in da.dims), None)


def members(da) -> list[str]:
    md = member_dim(da)
    return [str(m) for m in da[md].values] if md else []


# ----------------------------------------------------------------------------
# Downscaled products
# ----------------------------------------------------------------------------


def open_product(variable: str, product: str, cadence: str = "mon",
                 root: Path = ROOT) -> xr.DataArray:
    """One product at one cadence, every member along its ensemble dim.

    product is 'loca2', 'wrf-gcm' or 'wrf-era5'; cadence is 'mon', 'day' or
    '1hr'. The returned array is named by the canonical variable, not by the
    per-product variable_id -- LOCA2 publishes `wspeed` where WRF publishes
    `wspd10mean`, and callers should not have to care.
    """
    ds = _open(root / variable / f"{product}_{cadence}.zarr")
    name = variable if variable in ds else list(ds.data_vars)[0]
    return ds[name].rename(variable)


def list_products(variable: str, root: Path = ROOT) -> pd.DataFrame:
    """What has been downloaded for a variable."""
    rows = []
    for p in sorted((root / variable).glob("*_*.zarr")):
        product, cadence = p.stem.rsplit("_", 1)
        try:
            ds = xr.open_zarr(p, consolidated=True)
            v = list(ds.data_vars)[0]
            rows.append({"product": product, "cadence": cadence,
                         "members": len(members(ds[v])) or 1,
                         "steps": int(ds.sizes.get("time", 0)),
                         "start": str(ds.time.values[0])[:10],
                         "end": str(ds.time.values[-1])[:10]})
        except Exception as e:
            rows.append({"product": product, "cadence": cadence,
                         "members": f"ERR {type(e).__name__}"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# ERA5
# ----------------------------------------------------------------------------


def open_era5(variables=None, root: Path = ROOT) -> xr.Dataset:
    """Hourly ERA5, cropped. Components, not speed."""
    ds = _open(root / "era5" / "era5_hourly.zarr")
    return ds[list(variables)] if variables else ds


def era5_speed(freq: str | None = None, how: str = "mean",
               root: Path = ROOT) -> xr.DataArray:
    """10 m scalar wind speed. Magnitude first, resample second.

    freq=None keeps the hourly series. how='max' with freq='1D' gives the daily
    maxima the extremes comparison needs.
    """
    ds = open_era5(("u10", "v10"), root)
    w = np.sqrt(ds.u10 ** 2 + ds.v10 ** 2)
    w.name = "wspeed"
    w.attrs.update(units="m s-1",
                   long_name="10 m scalar wind speed, formed hourly")
    return w if freq is None else getattr(w.resample(time=freq), how)()


def era5_vector_speed(freq: str = "1D", root: Path = ROOT) -> xr.DataArray:
    """The WRONG aggregation, on purpose: components averaged, then magnitude.

    Only for the sampling test, which quantifies how much the vector mean
    understates the scalar mean (~20% here). Never use it as a reference.
    """
    ds = open_era5(("u10", "v10"), root)
    u, v = ds.u10.resample(time=freq).mean(), ds.v10.resample(time=freq).mean()
    out = np.sqrt(u ** 2 + v ** 2)
    out.name = "wspeed_vector"
    return out


# ----------------------------------------------------------------------------
# CONUS404
# ----------------------------------------------------------------------------


def open_conus404(variables=None, root: Path = ROOT) -> xr.Dataset:
    ds = _open(root / "conus404" / "conus404_hourly.zarr")
    return ds[list(variables)] if variables else ds


def conus404_speed(freq: str | None = None, how: str = "mean",
                   root: Path = ROOT, allow_vector: bool = False
                   ) -> xr.DataArray:
    """10 m scalar wind speed from CONUS404, formed hourly.

    REFUSES A NON-HOURLY STORE. CONUS404 publishes no wind-speed variable, so a
    daily or monthly product holds mean COMPONENTS -- and the magnitude of those
    is the vector mean, about 20% below the scalar mean. That is a bias the size
    of the discrepancies under study, and it lands on the one number the "ERA5
    was the wrong yardstick" argument rests on. Pass allow_vector=True only for
    spatial-pattern work, where a uniform scaling cancels out.

    U10/V10 are relative to the rotated projection rather than true north. That
    is irrelevant for magnitude, which is invariant under the rotation, but it
    would matter for direction -- use COSALPHA/SINALPHA from the static store.
    """
    ds = open_conus404(("U10", "V10"), root)
    step = np.median(np.diff(ds.time.values[:400])).astype("timedelta64[h]")
    if step != np.timedelta64(1, "h") and not allow_vector:
        raise ValueError(
            f"CONUS404 store has a {step} step, not hourly. Its components are "
            "already time-averaged, so their magnitude is the VECTOR mean "
            "(~20% low), not the scalar mean. Fetch the hourly product, or "
            "pass allow_vector=True if you only need spatial pattern.")
    w = np.sqrt(ds.U10 ** 2 + ds.V10 ** 2)
    w.name = "wspeed"
    w.attrs.update(units="m s-1",
                   long_name="10 m scalar wind speed, formed hourly")
    w = w.assign_coords(lat=ds.lat, lon=ds.lon) if "lat" in ds.coords else w
    return w if freq is None else getattr(w.resample(time=freq), how)()


def open_conus404_static(root: Path = ROOT) -> xr.Dataset:
    """HGT, LANDMASK, VAR_SSO, COSALPHA, SINALPHA on the cropped window."""
    return _open(root / "conus404" / "conus404_static.zarr")


# ----------------------------------------------------------------------------
# Stations
# ----------------------------------------------------------------------------


def analysis_networks(ds, min_n: int = MIN_NETWORK_N,
                      exclude=EXCLUDE_ANALYSIS) -> list[str]:
    """Networks kept for analysis: not excluded, and big enough to be a group.

    `exclude` is a parameter rather than a constant because the exclusions are
    window-dependent. CWOP is excluded from a 1980-2014 analysis because every
    station starts 2003-2011 and contributes a few years of a 35-year window --
    but over 2005-2014 those same records are contemporaneous and complete.
    Pass exclude=() to let them back in for a short-window run.
    """
    counts = pd.Series([str(n) for n in ds.network.values]).value_counts()
    return sorted(n for n, c in counts.items()
                  if n not in exclude and c >= min_n)


def _hdp_store(root, variable: str) -> Path:
    """Path to the archive holding `variable`.

    ONE STORE PER VARIABLE. A shared store makes station_id the key for every
    bookkeeping decision, and that is wrong once a station can carry one
    variable and not another -- it produced three separate silent failures:
    resume skipping stations that had wind but not humidity, a merge discarding
    an entire humidity fetch as "already seen", and duplicate station_id labels
    that only surfaced later inside xr.align.

    The legacy shared store is accepted as a fallback so existing archives keep
    working.
    """
    per = {"sfcWind": "hdp_wind_hourly.zarr", "hurs": "hdp_hurs_hourly.zarr"}
    cand = root / "hdp" / per.get(variable, f"hdp_{variable}_hourly.zarr")
    return cand if cand.exists() else root / "hdp" / "hdp_hourly.zarr"


def station_quality(da: xr.DataArray, min_obs_mean: float = MIN_OBS_MEAN,
                    min_months: int = MIN_MONTHS, min_unique: int = MIN_UNIQUE,
                    exclude_prefixes=EXCLUDE_PREFIXES, verbose: bool = True):
    """Boolean mask of stations whose record is usable as a reference.

    Four tests, each catching a failure the others miss:

      mean      a stuck-at-zero sensor. The single most damaging case, because
                the station is a ratio denominator.
      months    a record too short for a climatology. Distinct from the above:
                a three-month record can have a perfectly plausible mean.
      unique    a stuck sensor with a spurious spike. One real station recorded
                4,436 zeros and a single 17.43 m/s value -- its MAXIMUM looks
                entirely plausible, so any range- or extreme-based check passes
                it. Only the distinct-value count exposes it.
      prefix    a whole subgroup differing systematically, which no per-station
                threshold should be asked to catch one at a time.
    """
    vals = da.values
    ids = np.array([str(s) for s in da.station_id.values])

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(vals, axis=1)
    n_hours = np.isfinite(vals).sum(axis=1)
    months = n_hours / (24 * 30.4)
    uniq = np.array([len(np.unique(v[np.isfinite(v)])) for v in vals])
    pref = np.array([any(i.startswith(x) for x in exclude_prefixes)
                     for i in ids])

    tests = {"observed mean < %.2f m/s" % min_obs_mean: mean < min_obs_mean,
             "record < %d months" % min_months: months < min_months,
             "< %d distinct values" % min_unique: uniq < min_unique,
             "excluded prefix %s" % (exclude_prefixes,): pref}
    bad = np.zeros(len(ids), bool)
    for t in tests.values():
        bad |= np.nan_to_num(t, nan=True).astype(bool)
    keep = ~bad

    if verbose and bad.any():
        print(f"  station quality: dropping {int(bad.sum())} of {len(ids)}")
        for name, t in tests.items():
            n = int(np.nan_to_num(t, nan=True).astype(bool).sum())
            if n:
                print(f"    {n:>4}  {name}")
    return keep


def open_hdp(variable: str = "sfcWind", qc: bool = True,
             drop_stations_above: float | None = 0.01,
             networks: str | list[str] | None = "analysis",
             min_n: int = MIN_NETWORK_N, quality: bool = True,
             min_obs_mean: float = MIN_OBS_MEAN, min_months: int = MIN_MONTHS,
             exclude_prefixes=EXCLUDE_PREFIXES,
             time_slice: tuple[str, str] | None = None,
             root: Path = ROOT) -> xr.DataArray:
    """Hourly station observations, QC'd and quality-screened.

    The store is written raw so the thresholds stay a choice rather than being
    frozen into a two-hour artefact. Three filters apply here, in order:

      networks   keep the analysis networks (see EXPOSURE / EXCLUDE_ANALYSIS)
      qc         mask values outside [0, 75] m/s, drop stations whose bad-value
                 fraction exceeds `drop_stations_above` -- a station producing
                 many impossible values is not trustworthy for the ones that
                 happen to land in range either
      quality    drop stations whose record is not wind at all: see
                 station_quality. Set quality=False to inspect what is removed.
    """
    ds = _open(_hdp_store(root, variable))
    da = ds[variable] if variable in ds else ds[list(ds.data_vars)[0]]

    # Window FIRST. Record length and observed mean are properties of the window
    # being analysed, not of the archive: a station running 2007-2014 is short
    # against 1980-2014 and complete against 2005-2014, and screening it on the
    # wrong window would discard exactly the stations a shorter run exists to
    # include.
    if time_slice is not None:
        da = da.sel(time=slice(*time_slice))
        ds = ds.sel(time=slice(*time_slice))
        print(f"  window {time_slice[0]} .. {time_slice[1]}: "
              f"{da.sizes['time']:,} steps")

    # Subset BEFORE QC, so the dropped-station count reports on what is kept.
    if networks is not None:
        want = (analysis_networks(ds, min_n) if networks == "analysis"
                else analysis_networks(ds, min_n, exclude=())
                if networks == "all"
                else list(networks))
        sel = np.isin([str(n) for n in ds.network.values], want)
        print(f"  networks: {', '.join(want)} "
              f"({int(sel.sum())} of {ds.sizes['station_id']} stations)")
        da = da.isel(station_id=np.where(sel)[0])
        ds = ds.isel(station_id=np.where(sel)[0])

    if not qc:
        return da

    bad = ((da < WIND_MIN) | (da > WIND_MAX)) & da.notnull()
    da = da.where(~bad)
    if drop_stations_above is not None:
        frac = (bad.sum("time") / ds[da.name].notnull().sum("time")).compute()
        keep = (frac.fillna(0) <= drop_stations_above).values
        if not keep.all():
            print(f"  dropped {int((~keep).sum())} stations above "
                  f"{drop_stations_above:.1%} bad values")
            da = da.isel(station_id=np.where(keep)[0])

    if quality:
        # Needs the values, so it comes last -- after the cheap filters have
        # already reduced how much has to be read.
        da = da.compute()
        keep = station_quality(da, min_obs_mean=min_obs_mean,
                               min_months=min_months,
                               exclude_prefixes=exclude_prefixes)
        da = da.isel(station_id=np.where(keep)[0])
        print(f"  {da.sizes['station_id']} stations retained")
    return da


# Physical bounds for the domain: Badwater Basin is -86 m, Mount Whitney
# 4,421 m. A station outside this is a data error, not a location.
ELEV_MIN, ELEV_MAX = -120.0, 4500.0
FT_PER_M = 3.280839895


def station_elevation(root: Path = ROOT, verbose: bool = True) -> xr.DataArray:
    """Station elevation in METRES, from the retrieved stores.

    NOT from stations.csv. That column mixes units between networks -- ASOSAWOS
    and NOS-PORTS read as metres (medians 774 and 0) while RAWS, HADS, CRN,
    SNOTEL and SCAN read as feet (medians 4,400-4,900, impossible in metres
    here). Subtracting metres of model terrain from feet of station elevation
    inflates the residual by 3.28x for exactly the networks that sit on ridges,
    which manufactures the ridge-versus-airport ordering a siting analysis is
    meant to TEST. The elevation that arrives with each station's own store is
    the one to use.

    A per-network unit check runs anyway, because the store is not guaranteed
    consistent either and the failure is silent: feet look like plausible
    metres for any station above about 300 m.
    """
    ds = _open(_hdp_store(root, "sfcWind"))
    if "elevation" not in ds.coords:
        raise KeyError("hdp_hourly.zarr carries no elevation coordinate")
    elev = ds.elevation.compute().astype("float64")
    net = np.array([str(n) for n in ds.network.values])

    vals = elev.values.copy()
    for n in np.unique(net):
        m = net == n
        v = vals[m]
        finite = v[np.isfinite(v)]
        if not finite.size:
            continue
        if np.nanpercentile(finite, 95) > ELEV_MAX:
            vals[m] = v / FT_PER_M
            if verbose:
                print(f"  {n}: p95 {np.nanpercentile(finite, 95):.0f} exceeds "
                      f"{ELEV_MAX:.0f} m -- converting from feet")
    bad = np.isfinite(vals) & ((vals < ELEV_MIN) | (vals > ELEV_MAX))
    if bad.any():
        if verbose:
            print(f"  {int(bad.sum())} elevations outside "
                  f"[{ELEV_MIN:.0f}, {ELEV_MAX:.0f}] m set to NaN")
        vals[bad] = np.nan
    return xr.DataArray(vals, dims="station_id",
                        coords={"station_id": ds.station_id},
                        name="elevation_m")


def exposure_of(ds) -> pd.Series:
    """Siting class per station, indexed by station_id. 'unclassified' if unknown."""
    nets = [str(n) for n in ds.network.values]
    return pd.Series([EXPOSURE.get(n, "unclassified") for n in nets],
                     index=[str(s) for s in ds.station_id.values],
                     name="exposure")


# ----------------------------------------------------------------------------
# Relative humidity
# ----------------------------------------------------------------------------


def era5_rh(cadence: str = "mon", stat: str = "mean",
            root: Path = ROOT) -> xr.DataArray:
    """Derived ERA5 relative humidity.

    stat is 'mean', 'min' or 'max'. The minimum is only available at daily
    cadence, and it is the one that matters: RH's fire-relevant statistic is the
    afternoon minimum, which a monthly or daily MEAN cannot recover.

    Built by `derive_era5.py --variable hurs`, which needs t2m and d2m fetched.
    """
    name = {"mon": "era5_hurs_mon", "day": "era5_hurs_day"}[cadence]
    path = root / "era5" / f"{name}.zarr"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. ERA5 publishes no RH -- derive it:\n"
            "  python src/fetch_era5.py t2m d2m\n"
            "  python src/derive_era5.py --variable hurs")
    ds = _open(path)
    var = {"mean": "hurs", "min": "hurs_min", "max": "hurs_max"}[stat]
    if var not in ds:
        raise KeyError(f"{name} has no '{var}' "
                       f"(available: {list(ds.data_vars)})")
    return ds[var]


def open_loca2_rh(cadence: str = "day", how: str = "min",
                  root: Path = ROOT) -> xr.DataArray:
    """LOCA2 relative humidity from its published daily extremes.

    LOCA2 HAS NO MEAN RH at any cadence -- only `hursmax` and `hursmin`. So this
    cannot answer a question about mean humidity, and it can answer the
    fire-weather question directly, which the wind archive could not: there
    LOCA2 had no maximum-wind variable at all.

    how='mid' returns the midrange and is NOT the daily mean; see
    rh_analysis.loca2_daily for why the difference is largest in exactly the dry
    regions where RH matters most.
    """
    from .rh_analysis import loca2_daily

    tid = {"day": "day", "mon": "mon"}[cadence]
    hmax = open_product("hursmax", "loca2", tid, root)
    hmin = open_product("hursmin", "loca2", tid, root)
    return loca2_daily(hmax, hmin, how=how)


def open_hdp_rh(qc: bool = True, networks="analysis",
                min_n: int = MIN_NETWORK_N, quality: bool = True,
                time_slice=None, root: Path = ROOT) -> xr.DataArray:
    """Hourly station relative humidity, screened for RH-specific failures.

    Deliberately NOT `open_hdp(variable="hurs")`. That path applies the wind
    screen, whose central test is a low-mean floor aimed at an anemometer stuck
    at zero. The characteristic humidity failure is the opposite -- a sensor
    wetted by condensation reading ~100% indefinitely -- which a low-mean floor
    never sees. See rh_analysis.station_quality.
    """
    from . import rh_analysis as R

    # The SHARED store, holding hurs alongside sfcWind. Same stations carrying
    # both variables is what a joint fire-weather analysis needs -- low humidity
    # and strong wind have to be evaluated at the same site and hour -- and
    # resume is variable-aware, so the two coexist without one masking the other.
    path = _hdp_store(root, "hurs")
    ds = _open(path)
    if "hurs" not in ds:
        raise KeyError(
            f"{path.name} holds {sorted(ds.data_vars)} but no 'hurs'.\n"
            "  python src/fetch_hdp.py hurs --name hdp_hurs_hourly.zarr "
            "--batch-size 48")
    da = ds["hurs"]

    if time_slice is not None:
        da = da.sel(time=slice(*time_slice))
        ds = ds.sel(time=slice(*time_slice))
        print(f"  window {time_slice[0]} .. {time_slice[1]}: "
              f"{da.sizes['time']:,} steps")

    if networks is not None:
        want = (analysis_networks(ds, min_n) if networks == "analysis"
                else analysis_networks(ds, min_n, exclude=())
                if networks == "all" else list(networks))
        sel = np.isin([str(n) for n in ds.network.values], want)
        print(f"  networks: {', '.join(want)} "
              f"({int(sel.sum())} of {ds.sizes['station_id']} stations)")
        da = da.isel(station_id=np.where(sel)[0])

    if qc:
        da = R.qc(da)
    if quality:
        da = da.compute()
        da = da.isel(station_id=np.where(R.station_quality(da))[0])
        print(f"  {da.sizes['station_id']} stations retained")
    return da


def open_hdp_joint(variables=("sfcWind", "hurs"), root=ROOT, **kw):
    """Stations carrying ALL the named variables, aligned on station and time.

    Separate per-variable stores make each fetch independent, at the cost of
    needing this for a joint question. Fire danger is a SIMULTANEOUS condition
    -- low humidity and strong wind at the same site and hour -- so a joint
    analysis cannot use the union of two populations, only the intersection.

    An inner join makes that intersection explicit and shrinking, which is the
    honest behaviour: a shared store silently produced NaN-filled rows that look
    like data until something reduces over them.
    """
    readers = {"sfcWind": open_hdp, "hurs": open_hdp_rh}
    parts = {}
    for v in variables:
        if v not in readers:
            raise KeyError(f"no reader for '{v}' (have {sorted(readers)})")
        parts[v] = readers[v](root=root, **kw)
    aligned = xr.align(*parts.values(), join="inner")
    out = xr.Dataset(dict(zip(parts, aligned)))
    print(f"  joint: {out.sizes['station_id']} stations carry all of "
          f"{list(variables)}, {out.sizes['time']:,} steps")
    for v, pp in parts.items():
        print(f"    {v}: {pp.sizes['station_id']} alone")
    return out


def open_stations(root: Path = ROOT) -> pd.DataFrame:
    """Selection metadata, indexed by station_id.

    Carries `coverage` -- record-span overlap with the window -- which does not
    survive into the zarr. A store built at 10% coverage is subset to 25% by
    joining on this rather than refetching.
    """
    df = pd.read_csv(root / "hdp" / "stations.csv")
    return df.set_index(df.station_id.astype(str))


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------


def open_bbox(root: Path = ROOT) -> dict:
    return json.loads((root / "grids" / "bbox.json").read_text())


def open_elevation(root: Path = ROOT) -> xr.Dataset:
    path = root / "grids" / "wrf_elevation.nc"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run fetch_geometry.py")
    return xr.open_dataset(path)


# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------


def describe(variable: str = "wspeed", root: Path = ROOT) -> None:
    """Print what the archive currently holds. Cheap: metadata only."""
    print(f"root: {root}\n")
    print(f"--- products ({variable})")
    try:
        print(list_products(variable, root).to_string(index=False))
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")

    for label, fn in (("era5", lambda: open_era5(root=root)),
                      ("conus404", lambda: open_conus404(root=root)),
                      ("hdp", lambda: _open(_hdp_store(root, "sfcWind")))):
        print(f"\n--- {label}")
        try:
            ds = fn()
            print(f"  {dict(ds.sizes)}  {list(ds.data_vars)}")
            print(f"  {str(ds.time.values[0])[:13]} -> "
                  f"{str(ds.time.values[-1])[:13]}")
        except Exception as e:
            print(f"  {type(e).__name__}: {e}")

    print("\n--- geometry")
    try:
        b = open_bbox(root)
        print(f"  bbox {b['lat_min']:.2f}..{b['lat_max']:.2f} N, "
              f"{b['lon_min']:.2f}..{b['lon_max']:.2f} E")
    except Exception as e:
        print(f"  bbox: {type(e).__name__}")
    try:
        e = open_elevation(root)
        print(f"  elevation {dict(e.sizes)}")
    except Exception as e:
        print(f"  elevation: {type(e).__name__}")


if __name__ == "__main__":
    import sys
    describe(sys.argv[1] if len(sys.argv) > 1 else "wspeed")