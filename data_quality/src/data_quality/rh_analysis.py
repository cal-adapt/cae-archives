#!/usr/bin/env python
"""
Relative humidity analysis: the parts that cannot reuse the wind machinery.

`humidity.py` converts fields into RH. This module handles what to do with RH
once you have it, and exists because three of the wind pipeline's assumptions
are wrong for a bounded variable.

--- 1. RATIO IS NOT INTERPRETABLE -----------------------------------------

Wind speed is unbounded above and its error is naturally multiplicative, so
`model / observation` is the right summary. RH is bounded on [0, 100] and its
error is additive, so a ratio is constrained by the observation itself: where
observed RH is 90%, no model can score above 1.11 however wrong it is, while at
observed 20% a ratio of 3 is easy. A map of RH ratio is largely a map of where
the observations are high.

So everything here reports BIAS IN PERCENTAGE POINTS. A +8 pp bias means the
same thing at 20% and at 90%; a ratio of 1.1 does not.

--- 2. THE TAIL OF INTEREST IS THE LOW ONE --------------------------------

For wind the extremes that matter are the strong days. For humidity they are
the DRY hours -- the afternoon minimum that sets fire danger. Every quantile
comparison here is oriented that way, and the threshold counts use `<=` rather
than `>=`. This is easy to invert by accident and the result still looks
plausible.

--- 3. LOCA2 PUBLISHES NO MEAN RH -----------------------------------------

Only `hursmax` and `hursmin`. Two consequences, and they point opposite ways:

  * A comparison of mean RH cannot include LOCA2 at all.
  * But daily MINIMUM RH -- the fire-weather variable -- is published directly,
    so LOCA2 supports the fire use case here in a way it could not for wind,
    where it had no maximum-wind variable at any cadence.

The midrange (hursmax + hursmin) / 2 is NOT the daily mean; see `loca2_daily`.

Requires: numpy, pandas, xarray.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .humidity import RH_MIN, RH_MAX

# Instruments and models both overshoot slightly near saturation -- calibration
# drift on one side, grid-box supersaturation on the other. Values between 100
# and this are clipped; beyond it they are a sentinel or a unit error.
RH_QC_MAX = 105.0

# Thresholds in operational use for California. Agencies differ and these
# interact with wind and fuel state, so they are a framing device rather than a
# standard -- but they make the distributional comparisons concrete.
THRESHOLDS = {"critical": 15.0, "elevated": 25.0,
              "moist": 60.0, "saturated": 95.0}
DRY_LABELS = ("critical", "elevated")


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------


def bounded_metrics(srcda: xr.DataArray, refda: xr.DataArray,
                    synchronized: bool = False,
                    lo: float = RH_MIN, hi: float = RH_MAX) -> xr.Dataset:
    """Per-cell agreement for a variable bounded on [lo, hi].

    Structurally `metrics.agreement_metrics`, with `ratio` and `rel_bias`
    removed and bias in the variable's own units in their place.

    Three additions that only make sense for a bounded variable:

      dryness_diff      difference in the percentage of time at or below 20% --
                        the fire-relevant tail, and the number an operational
                        user would actually act on.
      saturation_diff   difference in the percentage of time at or above 95%.
                        Fog and marine layer are a distinct regime, and a model
                        can carry the right mean while never saturating at all.
      p10_diff          difference at the 10th percentile: the DRY tail. For
                        wind the equivalent diagnostic is q90; inverting it here
                        would silently measure the wrong end.
    """
    n = srcda.sizes["time"]
    t = srcda.time.values
    span_yr = (t[-1] - t[0]) / np.timedelta64(365, "D")
    if span_yr < 10:
        warnings.warn(
            f"Record spans only {span_yr:.1f} years ({n} steps). Seasonal and "
            "interannual metrics need many years. Use >= 10.", stacklevel=2)

    srcda = srcda.chunk({"time": -1})
    refda = refda.chunk({"time": -1})

    s_mean, r_mean = srcda.mean("time"), refda.mean("time")
    sc = srcda.groupby("time.month").mean()
    rc = refda.groupby("time.month").mean()
    sa = srcda.groupby("time.month") - sc
    ra = refda.groupby("time.month") - rc

    span = hi - lo
    dry, sat = lo + 0.20 * span, hi - 0.05 * span

    out = xr.Dataset({
        "bias": s_mean - r_mean,
        "abs_bias": abs(s_mean - r_mean),
        "seas_corr": xr.corr(sc, rc, dim="month"),
        "seas_amp_diff": (sc.max("month") - sc.min("month"))
                         - (rc.max("month") - rc.min("month")),
        "var_ratio": sa.std("time") / ra.std("time"),
        "p10_diff": (srcda.quantile(0.10, "time").drop_vars("quantile")
                     - refda.quantile(0.10, "time").drop_vars("quantile")),
        "p90_diff": (srcda.quantile(0.90, "time").drop_vars("quantile")
                     - refda.quantile(0.90, "time").drop_vars("quantile")),
        "dryness_diff": ((srcda <= dry).mean("time")
                         - (refda <= dry).mean("time")) * 100.0,
        "saturation_diff": ((srcda >= sat).mean("time")
                            - (refda >= sat).mean("time")) * 100.0,
        "source_mean": s_mean,
        "ref_mean": r_mean,
    })

    if synchronized:
        out["sync_corr"] = xr.corr(sa, ra, dim="time")
        out["rmse"] = np.sqrt(((srcda - refda) ** 2).mean("time"))
        out.attrs["synchronized"] = "yes - same-timestep metrics are meaningful"
    else:
        out.attrs["synchronized"] = (
            "no - free-running driver; same-timestep metrics omitted by design")

    for v in ("bias", "abs_bias", "p10_diff", "p90_diff", "seas_amp_diff"):
        out[v].attrs["units"] = "percentage points"
    for v in ("dryness_diff", "saturation_diff"):
        out[v].attrs["units"] = "percentage points of time"
    out.attrs.update(
        n_steps=n, span_years=round(float(span_yr), 2),
        time_start=str(t[0])[:10], time_end=str(t[-1])[:10],
        bounds=f"[{lo}, {hi}]", dry_threshold=dry, sat_threshold=sat,
        note="ratio and rel_bias omitted: a ratio is constrained by the "
             "reference value for a bounded variable and is not comparable "
             "between wet and dry cells")
    return out


# ----------------------------------------------------------------------------
# LOCA2's max/min pair
# ----------------------------------------------------------------------------


def loca2_daily(hursmax: xr.DataArray, hursmin: xr.DataArray,
                how: str = "min", quiet: bool = False) -> xr.DataArray:
    """A daily RH series from LOCA2's published max/min.

      how="min"    daily minimum -- the fire-weather variable, published
                   directly, no approximation
      how="max"    daily maximum, likewise direct
      how="range"  diurnal range: a measure of continentality that is often more
                   discriminating than either endpoint
      how="mid"    midrange, offered as a daily-mean proxy and warned about
                   every time, because it is not one

    WHY THE MIDRANGE IS BIASED, AND BIASED WHERE IT MATTERS. RH's diurnal cycle
    is asymmetric: humidity sits near its maximum through the long night and
    dips sharply for a few afternoon hours. The time-mean therefore lies ABOVE
    the midpoint of max and min, by an amount that grows with the diurnal range.
    So the error is largest in dry inland regimes -- exactly where RH matters
    most -- and smallest at the coast. Comparing a LOCA2 midrange against an
    observed mean would read as a regional dry bias that is an artefact of the
    statistic, not a property of the model.
    """
    # ALIGN ON THE MEMBER COORDINATE, NEVER ON POSITION.
    #
    # hursmax and hursmin are fetched as separate stores and their `sim` axes
    # come back in DIFFERENT ORDERS -- observed: hursmin starts CNRM-ESM2-1
    # while hursmax starts KACE-1-0-G, same 51 members either way. Pairing by
    # position therefore takes the maximum of one model and the minimum of
    # another, which produces max < min and a range that is meaningless. It is
    # silent: both arrays are the right shape and both means look plausible.
    md = next((d for d in ("sim", "simulation", "member_id", "member")
               if d in hursmax.dims and d in hursmin.dims), None)
    if md is not None:
        a = [str(x) for x in hursmax[md].values]
        b = [str(x) for x in hursmin[md].values]
        if a != b:
            common = [x for x in a if x in set(b)]
            if not common:
                raise ValueError(
                    f"hursmax and hursmin share no '{md}' values; they are not "
                    "the same ensemble")
            print(f"  aligning {len(common)} members by '{md}' "
                  f"(the two stores are in different order)")
            hursmax = hursmax.sel({md: common})
            hursmin = hursmin.sel({md: common})

    # A physical constraint, checked on the ALIGNED arrays.
    #
    # Checking before alignment would fire on precisely the case the alignment
    # exists to fix -- which is worse than not checking at all, because a
    # warning that always appears stops being read. What is worth catching is a
    # pair that is still inconsistent AFTER alignment: that means the stores
    # hold the wrong variables, not merely the wrong order.
    if not quiet:
        try:
            sub = {"time": slice(0, 200)}
            sub |= {d: 0 for d in hursmax.dims if d not in ("time",)}
            diff = (hursmax.isel(**sub) - hursmin.isel(**sub)).values
            diff = diff[np.isfinite(diff)]
            if diff.size and (diff < 0).mean() > 0.001:
                warnings.warn(
                    f"hursmax < hursmin on {100 * (diff < 0).mean():.1f}% of "
                    "sampled points AFTER aligning members. The stores are not "
                    "a matched max/min pair -- check each store's "
                    "`variable_id` attribute against its directory name.",
                    stacklevel=2)
        except Exception:
            pass

    if how == "min":
        out = hursmin.rename("hurs_min")
    elif how == "max":
        out = hursmax.rename("hurs_max")
    elif how == "range":
        out = (hursmax - hursmin).rename("hurs_range")
        out.attrs["long_name"] = "diurnal range of relative humidity"
    elif how == "mid":
        if not quiet:
            warnings.warn(
                "(hursmax + hursmin) / 2 is the MIDRANGE, not the daily mean. "
                "RH's diurnal cycle is asymmetric, so this sits below the true "
                "mean by an amount that scales with the diurnal range -- "
                "biasing dry regions more than moist ones. Compare it only "
                "against another midrange, never against an observed mean.",
                stacklevel=2)
        out = ((hursmax + hursmin) / 2.0).rename("hurs_mid")
        out.attrs["long_name"] = "midrange relative humidity (NOT the mean)"
    else:
        raise ValueError(f"how must be min/max/range/mid, got {how!r}")
    out.attrs.setdefault("units", "%")
    return out


def daily_stats(hourly: xr.DataArray, min_hours: int = 18) -> xr.Dataset:
    """Daily min, max, mean and range from an hourly series.

    The `min_hours` guard matters more here than for wind. RH's daily minimum
    occurs in a narrow afternoon window, so a day sampled only at night would
    contribute a "minimum" tens of points too high -- and it would look like a
    plausible value rather than an obvious gap.
    """
    cnt = hourly.notnull().resample(time="1D").sum()
    ok = cnt >= min_hours
    d = xr.Dataset({
        "hurs_min": hourly.resample(time="1D").min().where(ok),
        "hurs_max": hourly.resample(time="1D").max().where(ok),
        "hurs_mean": hourly.resample(time="1D").mean().where(ok),
    })
    d["hurs_range"] = d.hurs_max - d.hurs_min
    # The gap between the true daily mean and the midrange, which is what a
    # LOCA2 comparison inherits. Quantifying it from observations turns an
    # argument into a measurement.
    d["mid_minus_mean"] = (d.hurs_max + d.hurs_min) / 2.0 - d.hurs_mean
    for v in d.data_vars:
        d[v].attrs["units"] = "%"
    d.attrs["min_hours_per_day"] = min_hours
    return d


# ----------------------------------------------------------------------------
# Station screening
# ----------------------------------------------------------------------------

# A record whose mean falls outside this is not humidity. The floor is not zero:
# even the Mojave in July averages well above 10% over a month, so a station
# below it is stuck or mis-scaled. The ceiling catches the commoner failure --
# a sensor wetted by condensation reads ~100% indefinitely, which is physically
# possible for a day in coastal fog and not for a climatology.
MEAN_MIN, MEAN_MAX = 10.0, 98.0
# RH swings by tens of points daily everywhere in this domain, so a near
# constant record is an instrument fault regardless of its mean.
STD_MIN = 3.0
MIN_MONTHS = 60
MIN_UNIQUE = 20


def station_quality(da: xr.DataArray, mean_range=(MEAN_MIN, MEAN_MAX),
                    min_std: float = STD_MIN, min_months: int = MIN_MONTHS,
                    min_unique: int = MIN_UNIQUE, exclude_prefixes=(),
                    verbose: bool = True) -> np.ndarray:
    """Boolean mask of stations whose humidity record is usable.

    THE WIND SCREEN DOES NOT TRANSFER. Its central test is a low-mean floor,
    aimed at an anemometer stuck at zero. A humidity sensor fails differently:

      stuck high    condensation on the element, or a failed capacitive sensor,
                    reads ~100% indefinitely. A low-mean floor never sees it,
                    and this is the most common RH failure in coastal networks.
      stuck low     a dead or disconnected element reads ~0%.
      no variance   any constant output, at any level.

    Hence a mean RANGE rather than a floor, and an explicit variance test.
    """
    vals = da.values
    ids = np.array([str(s) for s in da.station_id.values])

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(vals, axis=1)
        std = np.nanstd(vals, axis=1)
    months = np.isfinite(vals).sum(axis=1) / (24 * 30.4)
    uniq = np.array([len(np.unique(v[np.isfinite(v)])) for v in vals])
    pref = np.array([any(i.startswith(x) for x in exclude_prefixes)
                     for i in ids])

    tests = {
        f"mean outside [{mean_range[0]:.0f}, {mean_range[1]:.0f}] %":
            (mean < mean_range[0]) | (mean > mean_range[1]),
        f"std < {min_std:.0f} pp (stuck sensor)": std < min_std,
        f"record < {min_months} months": months < min_months,
        f"< {min_unique} distinct values": uniq < min_unique,
    }
    if exclude_prefixes:
        tests[f"excluded prefix {exclude_prefixes}"] = pref

    bad = np.zeros(len(ids), bool)
    for t in tests.values():
        bad |= np.nan_to_num(t, nan=True).astype(bool)

    if verbose and bad.any():
        print(f"  RH station quality: dropping {int(bad.sum())} of {len(ids)}")
        for name, t in tests.items():
            k = int(np.nan_to_num(t, nan=True).astype(bool).sum())
            if k:
                print(f"    {k:>4}  {name}")
    return ~bad


def qc(da: xr.DataArray, qc_max: float = RH_QC_MAX) -> xr.DataArray:
    """Mask impossible values, clip mild overshoot.

    Values between 100 and `qc_max` are CLIPPED rather than dropped. Both
    instruments and models overshoot near saturation, and discarding those
    samples would preferentially remove fog and marine-layer conditions -- the
    regime a humidity analysis most wants to keep.
    """
    bad = ((da < RH_MIN) | (da > qc_max)) & da.notnull()
    return da.where(~bad).clip(RH_MIN, RH_MAX)


# ----------------------------------------------------------------------------
# Distributional summaries
# ----------------------------------------------------------------------------


def threshold_table(pool: dict, thresholds=THRESHOLDS,
                    below=DRY_LABELS) -> pd.DataFrame:
    """Percentage of time each source spends beyond each threshold.

    Dry thresholds counted as `<=`, moist ones as `>=`. For humidity the
    operationally interesting tail is the LOW one, the reverse of wind.
    """
    rows = []
    for name, v in pool.items():
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        row = {"source": name, "n": v.size, "mean": v.mean(),
               "p10": np.quantile(v, 0.10), "p50": np.median(v)}
        for label, thr in thresholds.items():
            if label in below:
                row[f"<={thr:.0f}%"] = 100.0 * float((v <= thr).mean())
            else:
                row[f">={thr:.0f}%"] = 100.0 * float((v >= thr).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("source")


def dry_hours_skill(model: np.ndarray, obs: np.ndarray,
                    threshold: float = 15.0) -> dict:
    """Contingency scores for "was it below the fire threshold?".

    A bias comparison says how far off the mean is; this says whether the model
    would have raised the alarm on the right days. They can disagree completely:
    a model with a +2 pp bias and too little variance can miss most critical
    hours while looking almost unbiased.
    """
    m, o = np.asarray(model), np.asarray(obs)
    ok = np.isfinite(m) & np.isfinite(o)
    m, o = m[ok] <= threshold, o[ok] <= threshold
    hits = int((m & o).sum())
    misses = int((~m & o).sum())
    false_alarms = int((m & ~o).sum())
    correct_neg = int((~m & ~o).sum())
    pod = hits / max(hits + misses, 1)             # probability of detection
    far = false_alarms / max(hits + false_alarms, 1)
    return {"threshold": threshold, "n": int(ok.sum()),
            "obs_rate_%": 100.0 * o.mean(), "mod_rate_%": 100.0 * m.mean(),
            "POD": pod, "FAR": far,
            "bias_ratio": m.mean() / max(o.mean(), 1e-12),
            "hits": hits, "misses": misses, "false_alarms": false_alarms,
            "correct_negatives": correct_neg}
