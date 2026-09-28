#!/usr/bin/env python
"""
Check the local archive before anything is built on it.

    python verify.py                     # everything
    python verify.py --variable wspeed
    python verify.py --deep              # also read data: NaN fractions, ranges

Metadata-only by default, so it runs in seconds and can be used to watch a fetch
in progress. --deep touches values and takes a few minutes.

WHAT IT IS LOOKING FOR. Every failure this project has hit was silent: a stale
aggregation reused because its cache key omitted the time window, a store cropped
to a different box that then intersected away under xr.align, chunks that looked
fine until an append refused. None raised at the point of the mistake. The checks
below are the cheap version of noticing.

Exit code is 0 only when nothing is marked FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import localdata as L

OK, WARN, FAIL = "ok  ", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(level: str, what: str, detail: str = "") -> None:
    _results.append((level, what, detail))
    print(f"  [{level}] {what}" + (f" -- {detail}" if detail else ""))


def var_chunks(ds: xr.Dataset, var: str) -> dict:
    """Chunk sizes of the DATA VARIABLE only, keyed by dim.

    DataArray.chunksizes folds in the coords, and coords are often stored
    unchunked while the data is not -- which makes it raise "Object has
    inconsistent chunks". Reading the underlying Variable avoids that.
    """
    ch = ds[var].variable.chunks
    if ch is None:
        return {}
    return {d: c[0] for d, c in zip(ds[var].dims, ch)}


def chunk_report(ds: xr.Dataset) -> str:
    """Chunk shape and whether the time axis is uniformly chunked.

    A short final chunk is normal and fine. A short chunk anywhere else means
    nothing can be appended to the store cleanly, which is how the CONUS404
    calendar-year append failed.
    """
    var = list(ds.data_vars)[0]
    full = ds[var].variable.chunks
    if full is None:
        return "not chunked"
    ch = dict(zip(ds[var].dims, full))
    if "time" not in ch:
        return str({k: v[0] for k, v in ch.items()})
    tc = ch["time"]
    uniform = len(set(tc[:-1])) <= 1 and (len(tc) < 2 or tc[-1] <= tc[0])
    return (f"time chunks {tc[0]}"
            + (f" x{len(tc)}" if len(tc) > 1 else "")
            + (f", last {tc[-1]}" if len(tc) > 1 and tc[-1] != tc[0] else "")
            + ("" if uniform else "  NON-UNIFORM"))


def span(ds: xr.Dataset) -> tuple[str, str, float]:
    t = ds.time.values
    yrs = (t[-1] - t[0]) / np.timedelta64(365, "D")
    return str(t[0])[:13], str(t[-1])[:13], float(yrs)


def infer_cadence(ds: xr.Dataset) -> str:
    """'1hr', 'day', 'mon' or 'other', from the median step."""
    d = np.diff(ds.time.values[:400])
    if not len(d):
        return "other"
    h = float(np.median(d.astype("timedelta64[h]").astype(int)))
    if h == 1:
        return "1hr"
    if h == 24:
        return "day"
    if 27 * 24 <= h <= 32 * 24:
        return "mon"
    return "other"


def check_time(ds: xr.Dataset, label: str, expect_freq: str | None = None):
    t0, t1, yrs = span(ds)
    cad = infer_cadence(ds)
    check(OK, f"{label} time", f"{t0} -> {t1}  ({ds.sizes['time']:,} steps, "
                               f"{yrs:.1f} yr, {cad})")
    if yrs < 10:
        check(WARN, f"{label} span",
              f"{yrs:.1f} yr. Seasonal and interannual metrics need many years; "
              "under ten they largely measure noise.")

    if cad == "day":
        # A 365-day model calendar has no Feb 29, so its daily count is an exact
        # multiple of 365 and its steps are not evenly spaced across leap years.
        # Benign under align(join="inner"), which drops the missing days -- but
        # it matters for day-of-year and block-maxima work, so name it.
        n = int(ds.sizes["time"])
        yrs_i = max(1, round(yrs))
        if n == yrs_i * 365:
            check(OK, f"{label} calendar",
                  f"{n:,} steps = {yrs_i} x 365 -- noleap (no Feb 29)")
        elif abs(n - yrs_i * 365.25) < 5:
            check(OK, f"{label} calendar", "standard (leap days present)")
    elif cad == "other":
        check(WARN, f"{label} cadence", "steps are not a recognised cadence")

    if expect_freq and cad != expect_freq:
        check(WARN, f"{label} cadence",
              f"{cad}, not {expect_freq} -- check this is intended")


def verify_products(variable: str, root: Path) -> None:
    print(f"\n=== products ({variable})")
    try:
        df = L.list_products(variable, root)
    except Exception as e:
        check(FAIL, "products", f"{type(e).__name__}: {e}")
        return
    if df.empty:
        check(FAIL, "products", f"nothing under {root / variable}")
        return
    print(df.to_string(index=False))

    for _, r in df.iterrows():
        if not isinstance(r.get("members"), (int, np.integer)):
            check(FAIL, f"{r['product']}/{r['cadence']}", str(r["members"]))
            continue
        p = root / variable / f"{r['product']}_{r['cadence']}.zarr"
        ds = xr.open_zarr(p, consolidated=True)
        label = f"{r['product']}/{r['cadence']}"
        check(OK, label, chunk_report(ds))
        check_time(ds, label)

        # Member chunking: metrics_over_members walks members one at a time, so
        # a chunk spanning several makes each read pull its neighbours too.
        v = list(ds.data_vars)[0]
        md = L.member_dim(ds[v])
        mc = var_chunks(ds, v).get(md)
        if md and mc and mc != 1:
            check(WARN, f"{label} member chunks",
                  f"{md} chunked at {mc}, not 1 -- "
                  "single-member reads will pull neighbours")

    # Every cadence of a product should agree on its member set.
    for cad in df.cadence.unique():
        sets = {}
        for prod in df[df.cadence == cad]["product"]:
            try:
                sets[prod] = set(L.members(
                    L.open_product(variable, prod, cad, root)))
            except Exception:
                pass
        for prod, s in sets.items():
            other = {c for c in df.cadence.unique() if c != cad}
            for oc in other:
                try:
                    s2 = set(L.members(L.open_product(variable, prod, oc, root)))
                except Exception:
                    continue
                if s and s2 and s != s2:
                    check(FAIL, f"{prod} members",
                          f"{cad} has {len(s)}, {oc} has {len(s2)} -- "
                          "the two cadences are not the same runs")


def verify_reference(name: str, opener, expect_freq: str | None, root: Path,
                     bbox: dict | None) -> xr.Dataset | None:
    print(f"\n=== {name}")
    try:
        ds = opener()
    except FileNotFoundError as e:
        check(WARN, name, str(e))
        return None
    except Exception as e:
        check(FAIL, name, f"{type(e).__name__}: {e}")
        return None

    check(OK, f"{name} dims", f"{dict(ds.sizes)}  {list(ds.data_vars)}")
    check(OK, f"{name} chunks", chunk_report(ds))
    check_time(ds, name, expect_freq)

    # The crop must match the frozen box, or metrics computed against different
    # stores are silently comparing different regions.
    if bbox and "latitude" in ds.coords:
        pad = bbox.get("pad_deg", 0.25)
        lo, hi = float(ds.latitude.min()), float(ds.latitude.max())
        if lo > bbox["lat_min"] or hi < bbox["lat_max"]:
            hint = ""
            # 31.0/43.8 is fetch_era5's FALLBACK_BBOX: the store was built
            # before bbox.json existed, so it used the hardcoded rectangle.
            if abs(lo - (31.0 - pad)) < 0.1 and abs(hi - (43.8 + pad)) < 0.3:
                hint = (" This is the built-in fallback box, so the fetch ran "
                        "before fetch_geometry.py. Refetch with --overwrite.")
            check(FAIL, f"{name} crop",
                  f"latitude {lo:.2f}..{hi:.2f} does not cover the frozen box "
                  f"{bbox['lat_min']:.2f}..{bbox['lat_max']:.2f}.{hint}")
        elif abs(lo - (bbox["lat_min"] - pad)) > 0.5:
            check(WARN, f"{name} crop",
                  f"latitude starts {lo:.2f}, expected near "
                  f"{bbox['lat_min'] - pad:.2f}")
        else:
            check(OK, f"{name} crop", "covers the frozen box")
    return ds


def verify_hdp(root: Path) -> None:
    print("\n=== hdp")
    try:
        ds = xr.open_zarr(root / "hdp" / "hdp_hourly.zarr", consolidated=True)
    except Exception as e:
        check(WARN, "hdp", f"{type(e).__name__}: {e}")
        return
    check(OK, "hdp dims", f"{ds.sizes['station_id']} stations x "
                          f"{ds.sizes['time']:,} steps")
    check_time(ds, "hdp", "1hr")

    nets = pd.Series([str(n) for n in ds.network.values])
    print(nets.value_counts().to_string())

    if len({str(s) for s in ds.station_id.values}) != ds.sizes["station_id"]:
        check(FAIL, "hdp station_id", "duplicates present -- a shard merge "
                                      "combined overlapping selections")
    for c in ("lat", "lon"):
        if c not in ds.coords:
            check(FAIL, f"hdp {c}", "missing; stations cannot be located")
        elif not np.isfinite(ds[c].values).all():
            check(WARN, f"hdp {c}",
                  f"{int((~np.isfinite(ds[c].values)).sum())} stations lack it")

    try:
        meta = L.open_stations(root)
        got = {str(s) for s in ds.station_id.values}
        sel = set(meta.index)
        check(OK, "hdp metadata", f"{len(sel)} selected, {len(got)} stored")
        lost = sel - got
        if lost:
            # sfcwind_nobs counts a station's WHOLE record, not the window, so a
            # 1997-2022 station can pass the filter and still return nothing.
            # But a whole-network shortfall is a different animal: climakitae
            # validates all-or-nothing, so one unserviceable station empties its
            # entire batch. Per-network delivery separates the two.
            check(WARN, "hdp delivery",
                  f"{len(lost)} of {len(sel)} selected returned no data "
                  f"({len(lost) / max(len(sel), 1):.0%})")
            stored = pd.Series([str(n) for n in ds.network.values]).value_counts()
            picked = meta.network.value_counts()
            tab = pd.DataFrame({"selected": picked, "stored": stored}).fillna(0)
            tab["delivered"] = (tab.stored / tab.selected).replace(
                [np.inf, -np.inf], np.nan)

            # Two very different causes look identical in a delivery count, so
            # bring the evidence that separates them. A network whose records
            # start late cannot fill the window however well retrieval goes:
            # sfcwind_nobs counts a station's WHOLE record, so a 2007-2023
            # station passes a 10% span filter against a 1980-2014 window and
            # then returns little or nothing inside it.
            if "coverage" in meta:
                tab["med_cov"] = meta.groupby("network").coverage.median()
            if "start" in meta:
                yr = pd.to_datetime(meta.start, errors="coerce").dt.year
                tab["med_start"] = yr.groupby(meta.network).median()
            tab = tab.sort_values("selected", ascending=False).astype(
                {"selected": int, "stored": int})
            print(tab.to_string(float_format=lambda v: f"{v:.2f}"))

            for net, r in tab[(tab.selected >= 20)
                              & (tab.delivered < 0.6)].iterrows():
                late = ("med_start" in tab and pd.notnull(r.get("med_start"))
                        and r["med_start"] >= 2000)
                thin = ("med_cov" in tab and pd.notnull(r.get("med_cov"))
                        and r["med_cov"] < 0.35)
                if late or thin:
                    check(OK, f"hdp {net}",
                          f"{int(r.stored)} of {int(r.selected)} "
                          f"({r.delivered:.0%}); records start ~"
                          f"{int(r.get('med_start', 0))}, median span coverage "
                          f"{r.get('med_cov', float('nan')):.2f} -- consistent "
                          "with short records, not retrieval failure")
                else:
                    check(WARN, f"hdp {net}",
                          f"only {int(r.stored)} of {int(r.selected)} "
                          f"({r.delivered:.0%}) despite long records -- "
                          "check for batch-level retrieval failure")
        if got - sel:
            check(FAIL, "hdp metadata",
                  f"{len(got - sel)} stored stations are absent from "
                  "stations.csv -- the two disagree")
    except FileNotFoundError:
        check(WARN, "hdp metadata", "stations.csv missing; coverage-based "
                                    "subsetting will not be possible")


def deep_checks(variable: str, root: Path) -> None:
    """Read values. Slower, and the only way to catch an all-NaN store."""
    print("\n=== deep (reading values)")
    for prod in ("loca2", "wrf-gcm", "wrf-era5"):
        try:
            da = L.open_product(variable, prod, "mon", root)
        except Exception:
            continue
        one = da.isel({L.member_dim(da): 0}) if L.member_dim(da) else da
        v = one.compute()
        finite = np.isfinite(v.values)
        frac = finite.mean()
        if frac == 0:
            check(FAIL, f"{prod} values", "entirely NaN")
        else:
            check(OK, f"{prod} values",
                  f"{frac:.1%} finite, mean {np.nanmean(v.values):.2f}, "
                  f"range {np.nanmin(v.values):.2f}..{np.nanmax(v.values):.2f}")

    try:
        w = L.era5_speed(freq="MS", root=root).compute()
        check(OK, "era5 speed", f"mean {float(w.mean()):.2f} m/s "
                                f"(scalar, formed hourly)")
        # The vector mean is the classic wrong aggregation; quantify the gap so
        # its size is on record rather than assumed.
        wv = L.era5_vector_speed("1D", root=root).resample(time="MS").mean().compute()
        pen = float(1 - wv.mean() / w.mean()) * 100
        check(OK, "era5 vector penalty", f"{pen:.1f}% below the scalar mean")
    except Exception as e:
        check(WARN, "era5 speed", f"{type(e).__name__}: {e}")

    try:
        obs = L.open_hdp(root=root, qc=True)
        vals = obs.values.ravel()
        vals = vals[np.isfinite(vals)]
        check(OK, "hdp values",
              f"{vals.size:,} valid, mean {vals.mean():.2f} m/s, "
              f"{(vals == 0).mean():.1%} exact zeros")
        if (vals == 0).mean() > 0.05:
            # ASOS reports 0 below roughly 3 kt. Models never produce exactly
            # zero wind, so a station reference sits structurally low.
            check(WARN, "hdp calm floor",
                  f"{(vals == 0).mean():.1%} exact zeros -- station means are "
                  "structurally below any model field")
    except Exception as e:
        check(WARN, "hdp values", f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(L.ROOT))
    ap.add_argument("--variable", default="wspeed")
    ap.add_argument("--deep", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)

    print(f"verifying {root}")

    print("\n=== geometry")
    bbox = None
    try:
        bbox = L.open_bbox(root)
        check(OK, "bbox", f"{bbox['lat_min']:.2f}..{bbox['lat_max']:.2f} N, "
                          f"{bbox['lon_min']:.2f}..{bbox['lon_max']:.2f} E "
                          f"from {len(bbox.get('from_stores', []))} store(s)")
    except Exception as e:
        check(FAIL, "bbox", f"{type(e).__name__} -- run fetch_geometry.py. "
                            "Without it the fetchers fall through to a "
                            "hardcoded rectangle.")
    try:
        e = L.open_elevation(root)
        check(OK, "elevation", f"{dict(e.sizes)} {list(e.data_vars)}")
    except Exception as ex:
        check(WARN, "elevation", str(ex))

    verify_products(a.variable, root)
    verify_reference("era5", lambda: L.open_era5(root=root), "1hr", root, bbox)
    c4 = verify_reference("conus404", lambda: L.open_conus404(root=root),
                          None, root, None)
    if c4 is not None and infer_cadence(c4) != "1hr" and "U10" in c4:
        # CONUS404 publishes no wind-speed variable, so a non-hourly product
        # holds mean COMPONENTS. Their magnitude is the vector mean, ~20% below
        # the scalar mean -- a bias the size of the discrepancies under study,
        # and in CONUS404 specifically it would move the one number the "ERA5
        # was the wrong yardstick" argument rests on.
        check(FAIL, "conus404 aggregation",
              f"store is {infer_cadence(c4)}, holding mean components. "
              "sqrt(U10^2+V10^2) on these is the VECTOR mean, ~20% low. "
              "Use it for spatial pattern only, never for ratio, var_ratio or "
              "extremes; fetch hourly for those.")
    verify_hdp(root)

    if a.deep:
        deep_checks(a.variable, root)

    n_fail = sum(1 for lv, _, _ in _results if lv == FAIL)
    n_warn = sum(1 for lv, _, _ in _results if lv == WARN)
    print("\n" + "=" * 62)
    print(f"{len(_results)} checks: {n_fail} FAIL, {n_warn} WARN")
    for lv, what, detail in _results:
        if lv != OK:
            print(f"  [{lv}] {what} -- {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())