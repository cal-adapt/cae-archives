#!/usr/bin/env python
"""
Agreement metrics. Given a product and a reference already on a common grid,
how well do they agree, and in what respect?

Nothing here knows where data came from, and nothing here reads a file.

WHY THE METRICS SPLIT BY DRIVER

  Free-running products (LOCA2, GCM-driven WRF) make their own weather. It is
  statistically consistent with the real climate but not synchronised to it, so
  a same-timestep correlation compares unrelated weather sequences and scores
  near zero regardless of model quality.

  Reanalysis-driven products (ERA5-driven WRF, CONUS404) downscale the OBSERVED
  atmosphere, so same-timestep metrics are meaningful for them alone.

`agreement_metrics(synchronized=...)` enforces that split rather than leaving it
to the caller to remember.

WHAT EACH METRIC ANSWERS

  ratio / bias / rel_bias   is the LEVEL right?
  seas_corr                 does it agree about WHEN it is windy? Silent about
                            magnitude: a product 70% too strong still scores ~1.
  var_ratio                 is the year-to-year variability the right SIZE?
                            When var_ratio tracks ratio the discrepancy is a
                            scale factor, not a difference in variability.
  pattern_metrics           are the maxima in the right PLACES? Computed with
                            the domain mean removed, so level and shape are
                            separated -- a product 50% too strong everywhere but
                            spatially perfect scores 1.0.
  extreme_metrics           does the upper TAIL agree? Quantile ratios, annual
                            maxima, exceedance frequency.
  sync_corr / rmse          same-timestep skill; reanalysis-driven only.

> Plotting rel_bias against the reference mean has a built-in 1/E dependence and
> produces a hyperbola whether or not any relationship exists. Use absolute bias
> against the reference mean, or ratio against terrain.

Requires: numpy, pandas, xarray, matplotlib.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .grids import progress, bin_to_era5, member_dim

TIME_START = "1980-01-01"
TIME_END = "2014-12-31"

# Where plot_agreement writes when no path is given.
OUTPUT_PNG = "agreement.png"



SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
           "JJA": [6, 7, 8], "SON": [9, 10, 11]}



# WRF simulations WITHOUT a-priori bias adjustment, from climakitae's
# NON_WRF_BA_MODELS. The remaining GCM-driven runs are the bias-adjusted set.
WRF_NON_BA = ("CESM2", "CNRM-ESM2-1", "FGOALS-g3", "ensmean")



WRF_BA_MODELS = ("EC-Earth3", "EC-Earth3-Veg", "MIROC6",
                 "MPI-ESM1-2-HR", "TaiESM1")



def agreement_metrics(srcda: xr.DataArray, era5: xr.DataArray,
                      synchronized: bool = False) -> xr.Dataset:
    """Per-cell agreement.

    Climatological metrics always. Same-timestep metrics ONLY when the source is
    driven by observed boundary conditions (ERA5-driven WRF) -- for free-running
    GCM runs those would be comparing unrelated weather sequences.
    """
    n = srcda.sizes["time"]
    t = srcda.time.values
    span_yr = (t[-1] - t[0]) / np.timedelta64(365, "D")
    if span_yr < 10:
        warnings.warn(
            f"Record spans only {span_yr:.1f} years ({n} steps). Seasonal and "
            "interannual metrics need many years: with a ~1-year sample each "
            "calendar month has one realization, so they measure noise. "
            "Use >= 10 years.", stacklevel=2,
        )

    srcda = srcda.chunk({"time": -1})
    era5 = era5.chunk({"time": -1})

    s_mean, e_mean = srcda.mean("time"), era5.mean("time")
    sc, ec = srcda.groupby("time.month").mean(), era5.groupby("time.month").mean()
    sa = srcda.groupby("time.month") - sc
    ea = era5.groupby("time.month") - ec

    out = xr.Dataset({
        "bias": s_mean - e_mean,
        "rel_bias": (s_mean - e_mean) / e_mean * 100.0,
        "ratio": s_mean / e_mean,
        "seas_corr": xr.corr(sc, ec, dim="month"),
        "seas_amp_ratio": (sc.max("month") - sc.min("month"))
                          / (ec.max("month") - ec.min("month")),
        "var_ratio": sa.std("time") / ea.std("time"),
        "q90_diff": srcda.quantile(0.9, "time").drop_vars("quantile")
                    - era5.quantile(0.9, "time").drop_vars("quantile"),
        "source_mean": s_mean,
        "era5_mean": e_mean,
    })

    if synchronized:
        # Legitimate only for ERA5-driven runs: same weather, downscaled.
        out["sync_corr"] = xr.corr(sa, ea, dim="time")
        out["rmse"] = np.sqrt(((srcda - era5) ** 2).mean("time"))
        out.attrs["synchronized"] = "yes - same-timestep metrics are meaningful"
    else:
        out.attrs["synchronized"] = (
            "no - free-running driver; same-timestep metrics omitted by design"
        )

    out.bias.attrs["units"] = "m s-1"
    out.rel_bias.attrs["units"] = "%"
    out.attrs["n_steps"] = n
    out.attrs["span_years"] = round(float(span_yr), 2)
    out.attrs["time_start"] = str(srcda.time.values[0])[:10]
    out.attrs["time_end"] = str(srcda.time.values[-1])[:10]
    return out



def seasonal_bias(srcda: xr.DataArray, era5: xr.DataArray,
                  keep: xr.DataArray | None = None) -> xr.Dataset:
    """Bias, ratio and correlation resolved by month and by season.

    `seas_corr` says whether the seasonal SHAPE matches; this says whether the
    offset itself is seasonal. A bias concentrated in one season points at a
    process (summer sea breeze, winter storm track) rather than a uniform
    transfer-function difference.
    """
    if keep is not None:
        srcda, era5 = srcda.where(keep), era5.where(keep)

    sc = srcda.groupby("time.month").mean()
    ec = era5.groupby("time.month").mean()

    out = xr.Dataset({
        "month_bias": sc - ec,
        "month_ratio": sc / ec,
        "source_clim": sc,
        "era5_clim": ec,
    })

    # Season aggregates, computed on the monthly climatology so each season is
    # weighted by its months rather than by record length.
    seas = {}
    for name, months in SEASONS.items():
        s = sc.sel(month=months).mean("month")
        e = ec.sel(month=months).mean("month")
        seas[name] = (s / e).expand_dims(season=[name])
    out["season_ratio"] = xr.concat(list(seas.values()), dim="season")
    return out



def seasonal_table(seasonal: xr.Dataset, label: str = "source") -> pd.DataFrame:
    """Median monthly and seasonal ratio, as a readable table."""
    mr = seasonal.month_ratio
    rows = {int(m): float(np.nanmedian(mr.sel(month=m).values))
            for m in mr.month.values}
    df = pd.DataFrame({label: rows})
    df.index.name = "month"
    sr = seasonal.season_ratio
    seas = pd.DataFrame({label: {str(s.values): float(np.nanmedian(
        sr.sel(season=s).values)) for s in sr.season}})
    seas.index.name = "season"
    return df, seas



def linear_trend(da: xr.DataArray, dim: str = "time",
                 per: str = "decade") -> xr.DataArray:
    """Least-squares slope per cell, in units per decade (or per year).

    Applied to ANNUAL means: monthly data would let the seasonal cycle leak
    into the fit unless it is removed first, and annual means also damp the
    month-to-month noise that dominates a 30-year record.
    """
    ann = da.groupby(f"{dim}.year").mean()
    yrs = ann["year"].values.astype("float64")
    x = yrs - yrs.mean()
    denom = (x ** 2).sum()

    xda = xr.DataArray(x, dims="year", coords={"year": ann["year"]})
    slope = (ann * xda).sum("year") / denom          # units per year
    if per == "decade":
        slope = slope * 10.0
        slope.attrs["units"] = f"{da.attrs.get('units', '')} per decade"
    else:
        slope.attrs["units"] = f"{da.attrs.get('units', '')} per year"
    return slope



def trend_comparison(srcda: xr.DataArray, era5: xr.DataArray,
                     keep: xr.DataArray | None = None,
                     synchronized: bool = False) -> xr.Dataset:
    """Per-cell trends in both fields, and their difference.

    IMPORTANT for free-running products: a GCM-driven run reproduces the
    *forced* response, not the observed sequence, so its trend over any
    particular 30 years also contains internal variability that has no reason
    to match ERA5's. Treat a trend difference as suggestive unless the run is
    ERA5-driven (`synchronized=True`), where the comparison is meaningful.
    """
    if keep is not None:
        srcda, era5 = srcda.where(keep), era5.where(keep)

    ts, te = linear_trend(srcda), linear_trend(era5)
    out = xr.Dataset({
        "source_trend": ts,
        "era5_trend": te,
        "trend_diff": ts - te,
        # relative trend removes the mean-state offset, so a product that is
        # uniformly high can still be checked for matching *fractional* change
        "source_trend_pct": ts / srcda.mean("time") * 100,
        "era5_trend_pct": te / era5.mean("time") * 100,
    })
    out.attrs["interpretation"] = (
        "meaningful" if synchronized else
        "suggestive only - free-running driver, internal variability unmatched")
    return out



def ensemble_ratios(src: xr.Dataset, kind, labels, era5_crop, variable: str,
                    era5_mon: xr.DataArray, keep: xr.DataArray,
                    time_start=TIME_START, time_end=TIME_END,
                    members=None, freq: str = "MS") -> xr.Dataset:
    """Per-member ratio to ERA5, reusing one ERA5 aggregation.

    The expensive part of a run is the ERA5 hourly read; the source side is
    comparatively cheap. Passing an already-computed `era5_mon` means an
    N-member ensemble costs one ERA5 read instead of N.

    Returns member_ratio (member x lat x lon) plus the ensemble mean and spread,
    which answer a question the single-member runs cannot: is the offset a
    property of the product, or of the member that happened to be chosen?
    """
    da = src[variable].sel(time=slice(time_start, time_end))
    md = member_dim(da)
    if md is None:
        raise ValueError("source has no ensemble dimension")

    names = list(da[md].values) if members is None else list(members)
    print(f"{len(names)} members along '{md}'")

    e_mean = era5_mon.where(keep).mean("time")
    ratios = []
    for i, name in enumerate(names, 1):
        one = da.sel({md: name})
        binned = bin_to_era5(one, kind, labels, era5_crop, how="mean")
        binned, _ = xr.align(binned, era5_mon, join="inner")
        with progress(f"[{i}/{len(names)}] {name}"):
            s_mean = binned.where(keep).mean("time").compute()
        ratios.append((s_mean / e_mean).expand_dims({md: [name]}))

    mr = xr.concat(ratios, dim=md)
    out = xr.Dataset({
        "member_ratio": mr,
        "ens_mean_ratio": mr.mean(md),
        "ens_std_ratio": mr.std(md),
        "ens_min_ratio": mr.min(md),
        "ens_max_ratio": mr.max(md),
    })
    med = float(np.nanmedian(out.ens_mean_ratio.values))
    spread = float(np.nanmedian(out.ens_std_ratio.values))
    rng = float(np.nanmedian((out.ens_max_ratio - out.ens_min_ratio).values))
    print(f"\nensemble mean ratio {med:.3f}, median across-member std {spread:.3f}, "
          f"median min-max range {rng:.3f}")
    print("If the spread is small next to the offset, the bias is a property of "
          "the product rather than of the chosen member.")
    return out



def region_summary(metrics: xr.Dataset, masks: dict,
                   var: str = "ratio", extra=("var_ratio", "rel_bias"),
                   quiet: bool = False) -> pd.DataFrame:
    """Median metric by region, with the pooled value for contrast.

    The pooled row is included deliberately. A domain-wide median is the number
    most often quoted and the one most sensitive to where the box was drawn --
    seeing it beside the land and ocean values shows how much of it is
    composition rather than signal.

    Over open water a km-scale model and a coarse reanalysis have nothing
    sub-grid to disagree about, so an ocean median near 1.0 alongside a land
    median well above it is a direct test of the resolution mechanism: a generic
    model-wide high bias would show over water too.
    """
    cols = [var] + [c for c in extra if c in metrics]
    rows = []
    for name, m in list(masks.items()) + [("pooled", None)]:
        sel = metrics if m is None else metrics.where(m)
        row = {"region": name,
               "n_cells": int(sel[var].notnull().sum())}
        for c in cols:
            row[c] = float(np.nanmedian(sel[c].values))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("region")
    if not quiet:
        print(df.round(3).to_string())
    return df


def elevation_summary(metrics, keep, elev_mean, elev_std=None,
                      bins=(0, 250, 500, 1000, 1500, 2000, 5000),
                      var="rel_bias"):
    """Stratify a metric by elevation band. More diagnostic than a lon split.

    If the discrepancy is resolution-driven it should scale with terrain; if it
    is a uniform transfer-function offset it should not.
    """
    metrics = metrics.compute()
    keep = keep.compute() if hasattr(keep.data, "compute") else keep
    m = metrics[var].where(keep)
    h = elev_mean.where(keep)

    print(f"\n elevation band      n   median {var}" +
          ("   median relief" if elev_std is not None else ""))
    print(" " + "-" * (38 + (16 if elev_std is not None else 0)))
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (h >= lo) & (h < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        row = f" {lo:>5}-{hi:<5} m {n:>7}   {float(np.nanmedian(m.where(sel).values)):>12.1f}"
        if elev_std is not None:
            row += f"   {float(np.nanmedian(elev_std.where(keep).where(sel).values)):>12.0f}"
        print(row)



def regional_summary(metrics: xr.Dataset, keep: xr.DataArray):
    metrics = metrics.compute()
    keep = keep.compute() if hasattr(keep.data, "compute") else keep
    coastal = (metrics.longitude < -120) & keep
    interior = (metrics.longitude >= -120) & keep

    def med(da, m):
        return float(np.nanmedian(da.where(m).values))

    cols = ["rel_bias", "seas_corr", "var_ratio"]
    if "sync_corr" in metrics:
        cols.append("sync_corr")
    print("\n region     n    " + "".join(f"{c:>12}" for c in cols))
    print(" " + "-" * (17 + 12 * len(cols)))
    for name, m in (("coastal", coastal), ("interior", interior)):
        vals = "".join(f"{med(metrics[c], m):>12.2f}" for c in cols)
        print(f" {name:<9} {int(m.sum()):>5}{vals}")

    good = ((np.abs(metrics.rel_bias) < 15) & (metrics.seas_corr > 0.8)
            & (np.abs(metrics.var_ratio - 1) < 0.25)) & keep
    print(f"\n cells agreeing on all three criteria: {int(good.sum())} / {int(keep.sum())}")



def model_of(sim_name: str) -> str:
    """Driving GCM from a LOCA2 sim label.

    'loca2_ucsd_hadgem3-gc31-ll_historical_r1i1p1f3' -> 'hadgem3-gc31-ll'.
    Grouping members by GCM matters: a model contributing 10 members would
    otherwise dominate a flat across-member mean.
    """
    parts = str(sim_name).split("_")
    for i, tok in enumerate(parts):
        if tok in ("historical", "ssp245", "ssp370", "ssp585"):
            return "_".join(parts[2:i]) if i > 2 else parts[i - 1]
    return str(sim_name)



def metrics_over_members(members, kind, labels, era5_crop, era5_mon,
                         keep, synchronized: bool = False,
                         dim: str = "model") -> xr.Dataset:
    """Agreement metrics for many source arrays sharing one grid and one ERA5.

    `members` is an iterable of (name, DataArray). The ERA5 side is computed
    once and reused, which is the whole point: it is the expensive half.
    """
    out = []
    names = []
    members = list(members)
    for i, (name, da) in enumerate(members, 1):
        binned = bin_to_era5(da, kind, labels, era5_crop, how="mean")
        b, e = xr.align(binned, era5_mon, join="inner")
        if b.sizes["time"] == 0:
            warnings.warn(f"{name}: no overlapping timestamps, skipped")
            continue
        with progress(f"[{i}/{len(members)}] {name}"):
            m = agreement_metrics(b.where(keep), e.where(keep),
                                  synchronized=synchronized).compute()
        out.append(m)
        names.append(name)

    if not out:
        raise RuntimeError("no member produced metrics")
    ds = xr.concat(out, dim=pd.Index(names, name=dim))
    ds.attrs["n_members"] = len(names)
    print(f"\n{len(names)} members along '{dim}'")
    return ds



def group_by_model(ds: xr.Dataset, dim: str = "model",
                   mapper=model_of) -> xr.Dataset:
    """Collapse members to one entry per driving GCM.

    A flat mean over LOCA2's 46 members weights IPSL-CM6A-LR (10 members) five
    times more than ACCESS-CM2 (1). Averaging within model first gives each GCM
    equal weight, which is what a model-mean should mean.
    """
    groups = pd.Index([mapper(v) for v in ds[dim].values], name="gcm")
    out = ds.groupby(xr.DataArray(groups, dims=dim, coords={dim: ds[dim]})).mean(dim)
    out = out.rename({"gcm": "model"}) if "gcm" in out.dims else out
    print(f"{ds.sizes[dim]} members -> {out.sizes.get('model', 0)} models")
    return out



def multi_model_summary(ds: xr.Dataset, dim: str = "model",
                        metrics=("ratio", "rel_bias", "seas_corr",
                                 "var_ratio")) -> pd.DataFrame:
    """Domain-median of each metric, per member/model, sorted by ratio."""
    rows = []
    for i, name in enumerate(ds[dim].values):
        one = ds.isel({dim: i})
        rows.append({dim: str(name),
                     **{m: float(np.nanmedian(one[m].values))
                        for m in metrics if m in one}})
    df = pd.DataFrame(rows).set_index(dim).sort_values("ratio")
    return df



def pattern_metrics(src_mean: xr.DataArray, ref_mean: xr.DataArray,
                    keep: xr.DataArray | None = None,
                    weights: xr.DataArray | None = None) -> dict:
    """Centred pattern correlation, spatial-std ratio and centred RMS error.

    All three are computed on the field with its domain mean removed, so they
    describe SHAPE independently of level. `pattern_corr` near 1 with
    `std_ratio` near 1 means the product places wind maxima and minima where
    the reference does, with the right spatial contrast, whatever the bias.

    `weights` should be cos(latitude) on a regular lat/lon grid, so that cells
    are weighted by area rather than counted equally.
    """
    # Align BEFORE masking. `.where()` with a mask on a different grid
    # broadcasts to the union of coordinates rather than masking, so masking
    # first silently changes the shape when the two fields come from different
    # crops -- which is exactly the case when comparing CONUS404, on its own
    # ERA5 crop, against a product on the LOCA2 crop.
    # Weights are aligned alongside the fields: broadcast_like expands to the
    # UNION of coordinates rather than trimming, so weights built on the
    # pre-alignment grid would not match the aligned field.
    pieces = [src_mean, ref_mean]
    if keep is not None:
        pieces.append(keep)
    if weights is not None:
        pieces.append(weights)
    aligned = xr.align(*pieces, join="inner")

    a, b = aligned[0], aligned[1]
    i = 2
    if keep is not None:
        a, b = a.where(aligned[i]), b.where(aligned[i]); i += 1
    weights = aligned[i] if weights is not None else None
    if a.size == 0:
        raise ValueError(
            "no overlapping cells after alignment: the two fields are on "
            "different grids with no shared coordinates")

    av, bv = np.asarray(a.values, float).ravel(), np.asarray(b.values, float).ravel()
    if weights is None:
        w = np.ones_like(av)
    else:
        # cos(lat) is 1D on a regular grid while the field is 2D, and it must
        # be aligned to the post-alignment field, not the original.
        w = np.asarray(weights.broadcast_like(a).values, float).ravel()

    ok = np.isfinite(av) & np.isfinite(bv) & np.isfinite(w) & (w > 0)
    av, bv, w = av[ok], bv[ok], w[ok]
    if av.size < 10:
        raise ValueError(f"only {av.size} valid cells; pattern metrics need more")

    wsum = w.sum()
    am, bm = (w * av).sum() / wsum, (w * bv).sum() / wsum
    aa, bb = av - am, bv - bm                       # centred: level removed

    sa = np.sqrt((w * aa ** 2).sum() / wsum)
    sb = np.sqrt((w * bb ** 2).sum() / wsum)
    corr = (w * aa * bb).sum() / wsum / (sa * sb)
    crmse = np.sqrt((w * (aa - bb) ** 2).sum() / wsum)

    return {
        "n_cells": int(av.size),
        "pattern_corr": float(corr),
        "std_ratio": float(sa / sb),          # spatial contrast, relative
        "crmse_norm": float(crmse / sb),      # centred RMS error, normalised
        "mean_src": float(am),
        "mean_ref": float(bm),
        "mean_ratio": float(am / bm),
    }



def extreme_metrics(src: xr.DataArray, ref: xr.DataArray,
                    quantiles=(0.9, 0.99), threshold: float | None = None,
                    keep: xr.DataArray | None = None) -> xr.Dataset:
    """Upper-tail agreement, per cell.

    * quantile ratios -- does the product get the tail shape right, not just
      the mean?
    * annual-maximum statistics -- the quantity a design-load calculation uses
    * exceedance frequency above `threshold`, if given -- the quantity a fire
      weather or grid-resilience threshold uses
    """
    if keep is not None:
        src, ref = src.where(keep), ref.where(keep)
    src = src.chunk({"time": -1}) if hasattr(src.data, "chunks") else src
    ref = ref.chunk({"time": -1}) if hasattr(ref.data, "chunks") else ref

    out = {}
    for q in quantiles:
        s = src.quantile(q, "time").drop_vars("quantile")
        r = ref.quantile(q, "time").drop_vars("quantile")
        tag = f"q{int(q * 100)}"
        out[f"{tag}_src"] = s
        out[f"{tag}_ref"] = r
        out[f"{tag}_ratio"] = s / r
        out[f"{tag}_diff"] = s - r

    s_amax = src.groupby("time.year").max()
    r_amax = ref.groupby("time.year").max()
    out["amax_src"] = s_amax.mean("year")
    out["amax_ref"] = r_amax.mean("year")
    out["amax_ratio"] = out["amax_src"] / out["amax_ref"]
    # Year-to-year spread of the annual maximum: a design calculation depends
    # on this as much as on its mean.
    out["amax_sd_ratio"] = s_amax.std("year") / r_amax.std("year")

    if threshold is not None:
        fs = (src > threshold).mean("time")
        fr = (ref > threshold).mean("time")
        out["exceed_src"] = fs
        out["exceed_ref"] = fr
        # Ratio of frequencies, guarded where the reference never exceeds.
        out["exceed_ratio"] = fs / fr.where(fr > 0)
        out.update(threshold=threshold)

    ds = xr.Dataset({k: v for k, v in out.items() if isinstance(v, xr.DataArray)})
    ds.attrs["quantiles"] = list(quantiles)
    if threshold is not None:
        ds.attrs["threshold"] = threshold
    ds.attrs["sampling_caveat"] = (
        "If one side is a model-timestep maximum (WRF wspd10max) and the other "
        "is derived from hourly samples, the model side is higher by "
        "construction. Bounded comparison, not exact.")
    return ds



def extreme_summary(ds: xr.Dataset, keep: xr.DataArray | None = None,
                    label: str = "product") -> pd.DataFrame:
    """Domain medians of the extreme metrics."""
    def m(name):
        if name not in ds:
            return np.nan
        da = ds[name].where(keep) if keep is not None else ds[name]
        return float(np.nanmedian(da.values))

    row = {"q90_ratio": m("q90_ratio"), "q99_ratio": m("q99_ratio"),
           "amax_ratio": m("amax_ratio"), "amax_sd_ratio": m("amax_sd_ratio"),
           "mean_amax_src": m("amax_src"), "mean_amax_ref": m("amax_ref")}
    if "exceed_ratio" in ds:
        row["exceed_ratio"] = m("exceed_ratio")
        row["exceed_pct_src"] = m("exceed_src") * 100
        row["exceed_pct_ref"] = m("exceed_ref") * 100
    return pd.DataFrame([row], index=[label]).round(3)



def taylor_diagram(stats: dict, ax=None, title: str = "",
                   colors: dict | None = None):
    """Taylor diagram from {label: pattern_metrics(...) dict}.

    Radius is the spatial-standard-deviation ratio, angle the pattern
    correlation; distance to the reference point at (1, 0) is the centred RMS
    error. A product sitting on the arc at radius 1 has perfect spatial
    structure regardless of its mean bias.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))

    rmax = max(1.6, max(s["std_ratio"] for s in stats.values()) * 1.15)

    # correlation arcs
    for c in (0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99):
        th = np.arccos(c)
        ax.plot([0, rmax * np.cos(th)], [0, rmax * np.sin(th)],
                color="0.85", lw=0.6, zorder=0)
        ax.text(rmax * np.cos(th) * 1.02, rmax * np.sin(th) * 1.02, f"{c}",
                fontsize=7, color="0.4", ha="left", va="bottom")
    # std ratio arcs
    th = np.linspace(0, np.pi / 2, 200)
    for r in (0.5, 1.0, 1.5):
        if r <= rmax:
            ax.plot(r * np.cos(th), r * np.sin(th), color="0.85", lw=0.6,
                    ls="--" if r != 1.0 else "-", zorder=0)
    # centred-RMSE arcs about the reference point
    for r in (0.25, 0.5, 0.75, 1.0):
        c = 1 + r * np.cos(th + np.pi / 2) * 0 + r * np.cos(np.linspace(0, 2*np.pi, 200))
        s = r * np.sin(np.linspace(0, 2*np.pi, 200))
        m = (np.hypot(c, s) <= rmax) & (s >= 0)
        ax.plot(c[m], s[m], color="0.92", lw=0.6, zorder=0)

    ax.plot(1, 0, "k*", ms=14, label="reference", zorder=5)
    for label, s in stats.items():
        th = np.arccos(np.clip(s["pattern_corr"], -1, 1))
        r = s["std_ratio"]
        ax.plot(r * np.cos(th), r * np.sin(th), "o", ms=10,
                color=(colors or {}).get(label), label=label, zorder=5)

    ax.set_xlim(0, rmax); ax.set_ylim(0, rmax)
    ax.set_aspect("equal")
    ax.set_xlabel("spatial standard deviation, normalised")
    ax.set_title(title or "Pattern fidelity (level removed)")
    ax.legend(fontsize=8, loc="upper right")
    return ax



def plot_agreement(metrics: xr.Dataset, path=OUTPUT_PNG, label="source"):
    panels = [
        ("rel_bias", f"Mean bias, {label} - ERA5 [%]", "RdBu_r", dict(vmin=-40, vmax=40)),
        ("seas_corr", "Seasonal cycle correlation", "viridis", dict(vmin=0, vmax=1)),
        ("seas_amp_ratio", "Seasonal amplitude ratio", "RdBu_r", dict(vmin=0.5, vmax=1.5)),
        ("var_ratio", "Interannual std ratio", "RdBu_r", dict(vmin=0.5, vmax=1.5)),
        ("q90_diff", "90th percentile diff [m/s]", "RdBu_r", dict(vmin=-2, vmax=2)),
        ("era5_mean", "ERA5 mean wind [m/s]", "viridis", {}),
    ]
    if "sync_corr" in metrics:
        panels[5] = ("sync_corr", "Same-timestep anomaly corr", "viridis",
                     dict(vmin=0, vmax=1))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (name, title, cmap, kw) in zip(axes.flat, panels):
        metrics[name].plot(ax=ax, cmap=cmap, **kw)
        ax.set_title(title)
        ax.set_xlabel(""); ax.set_ylabel("")

    n = metrics.attrs.get("n_steps", "?")
    span = metrics.attrs.get("span_years", 99)
    warn = "  <-- TOO SHORT FOR CLIMATOLOGY" if span < 10 else ""
    fig.suptitle(
        f"{label} vs ERA5 10 m wind, {metrics.attrs.get('time_start','?')} to "
        f"{metrics.attrs.get('time_end','?')} ({n} months, ERA5 0.25 deg){warn}"
    )
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")