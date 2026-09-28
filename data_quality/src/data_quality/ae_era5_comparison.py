"""
Regional agreement between a Cal-Adapt Analytics Engine product and ERA5.

Supports both AE grid types:
  * LOCA2  -- statistically downscaled, RECTILINEAR 1D lat/lon, has a `sim` dim
  * WRF    -- dynamically downscaled, CURVILINEAR 2D lat/lon over x/y (Lambert
              conformal), driven either by ERA5 (reanalysis) or by a GCM

Compares a variable cell-by-cell on the ERA5 0.25 deg grid and maps *where* the
two datasets agree.

WHICH METRICS ARE LEGITIMATE DEPENDS ON THE DRIVER:
  * GCM-driven (LOCA2 historical, WRF historical/ssp) is FREE-RUNNING. Internal
    variability is not synchronised with the observed atmosphere, so same-timestep
    correlation against ERA5 is meaningless. Only climatological statistics apply.
  * ERA5-driven WRF (experiment_id="reanalysis") IS synchronised -- it is the same
    weather, dynamically downscaled. Same-timestep correlation and RMSE are
    meaningful and are computed additionally when synchronized=True.

Requires: climakitae >= 1.5, arraylake, xarray, dask, flox, numpy, matplotlib
Optional: cartopy

    python ae_era5_comparison.py --preset wrf-era5
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

TIME_START = "1980-01-01"
TIME_END = "2010-12-01"

SIM = 0                      # int index, member-name string, or "mean"
COVERAGE_THRESHOLD = 0.5
BBOX_PAD = 0.25
PROGRESS = True

# --- ERA5 (Arraylake) ---
ERA5_REPO = "earthmover-public/era5-surface-aws"
ERA5_BRANCH = "main"
ERA5_GROUP = "temporal"      # long time series in 12x12 tiles; right for this query
ERA5_U = "u10"
ERA5_V = "v10"

OUTPUT_NC = "agreement.nc"
OUTPUT_PNG = "agreement.png"

# Candidate names for the ensemble-member dimension, in priority order.
MEMBER_DIMS = ("sim", "simulation", "member_id", "member")

# ERA5 resample frequency to match each AE table_id. ERA5 is hourly natively,
# so it must be aggregated DOWN to whatever cadence the source uses.
FREQ_FOR_TABLE = {"mon": "MS", "day": "1D", "1hr": "h"}

# --- Source presets -------------------------------------------------------
# `synchronized` marks datasets whose weather is the real observed sequence.
PRESETS = {
    "loca2": dict(
        catalog="cadcat", activity_id="LOCA2", experiment_id="historical",
        table_id="mon", grid_label="d03", variable="wspeed",
        institution_id=None, source_id=None, synchronized=False,
        processes=None,          # LOCA2 IS bias-adjusted; keep the default filter
        label="LOCA2 historical",
    ),
    "wrf-era5": dict(
        catalog="cadcat", activity_id="WRF", experiment_id="reanalysis",
        table_id="mon", grid_label="d03", variable="wspd10mean",
        institution_id="UCLA", source_id="ERA5", synchronized=True,
        # Dynamically downscaled WRF is not a-priori bias-adjusted, so the
        # default filter_unadjusted_models drops the only matching entry and
        # concat then fails with "No valid datasets found for concatenation".
        processes={"filter_unadjusted_models": "no"},
        label="WRF ERA5-driven reconstruction",
    ),
    "wrf-gcm": dict(
        catalog="cadcat", activity_id="WRF", experiment_id="historical",
        table_id="mon", grid_label="d03", variable="wspd10mean",
        institution_id="UCLA", source_id="CNRM-ESM2-1", synchronized=False,
        processes={"filter_unadjusted_models": "no"},
        label="WRF CNRM-ESM2-1 historical",
    ),
}


def progress(label: str = ""):
    """Dask progress bar for local schedulers. No-op under dask.distributed."""
    if not PROGRESS:
        return nullcontext()
    if label:
        print(label)
    return ProgressBar(minimum=1.0)


# ----------------------------------------------------------------------------
# 1. Grid introspection -- the core of the generalization
# ----------------------------------------------------------------------------


def get_latlon(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Return the (lat, lon) coordinate arrays, whatever they are called."""
    for la, lo in (("lat", "lon"), ("latitude", "longitude"),
                   ("XLAT", "XLONG"), ("south_north", "west_east")):
        if la in ds.coords and lo in ds.coords:
            return ds[la], ds[lo]
    raise KeyError(f"No lat/lon coords found. Have: {list(ds.coords)}")


def grid_kind(ds: xr.Dataset) -> str:
    """'rectilinear' if lat/lon are 1D dimension coords, else 'curvilinear'.

    LOCA2 is rectilinear (lat/lon are their own dims). WRF is curvilinear:
    lat/lon are 2D fields over projected x/y dims, so a lat slice is not a
    rectangle in index space and per-dimension groupby does not work.
    """
    lat, lon = get_latlon(ds)
    return "rectilinear" if (lat.ndim == 1 and lon.ndim == 1) else "curvilinear"


def horizontal_dims(ds: xr.Dataset) -> tuple[str, ...]:
    """The two dims that span space."""
    lat, _ = get_latlon(ds)
    if lat.ndim == 1:
        la, lo = get_latlon(ds)
        return (la.dims[0], lo.dims[0])
    return tuple(lat.dims)          # e.g. ("y", "x")


def member_dim(da: xr.DataArray) -> str | None:
    """Name of the ensemble dim if present."""
    for d in MEMBER_DIMS:
        if d in da.dims:
            return d
    return None


def select_member(da: xr.DataArray, sim) -> xr.DataArray:
    """Pick one member, or average them. No-op when there is no member dim.

    WRF ERA5-driven runs have no ensemble dim at all; LOCA2 has 46.
    """
    md = member_dim(da)
    if md is None:
        print("no ensemble dimension (single realization)")
        return da
    if sim == "mean":
        print(f"averaging over {da.sizes[md]} members along '{md}'")
        return da.mean(md, keep_attrs=True)
    if isinstance(sim, int):
        print(f"member {sim} of {da.sizes[md]} along '{md}'")
        return da.isel({md: sim})
    print(f"member '{sim}' along '{md}'")
    return da.sel({md: sim})


# ----------------------------------------------------------------------------
# 2. Data loading
# ----------------------------------------------------------------------------


def load_source(
    catalog: str = "cadcat",
    activity_id: str = "LOCA2",
    experiment_id: str | list[str] = "historical",
    table_id: str = "mon",
    grid_label: str = "d03",
    variable: str = "wspeed",
    institution_id: str | None = None,
    source_id: str | None = None,
    processes: dict | None = None,
    **_ignored,
) -> xr.Dataset:
    """Retrieve gridded data from the Analytics Engine.

    Pass **PRESETS["wrf-era5"] to fill every key at once; extra preset keys
    (synchronized, label) are ignored here.
    """
    from climakitae import ClimateData

    q = (
        ClimateData()
        .catalog(catalog)
        .activity_id(activity_id)
        .experiment_id(experiment_id)
        .table_id(table_id)
        .grid_label(grid_label)
        .variable(variable)
    )
    if institution_id is not None:
        q = q.institution_id(institution_id)
    if source_id is not None:
        q = q.source_id(source_id)
    if processes:
        q = q.processes(processes)

    q.show_query()
    ds = q.get()
    if ds is None:
        raise RuntimeError(
            "Query returned nothing. Check the printed query against "
            "show_variable_options() / show_table_id_options()."
        )
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=variable)
    print(f"grid: {grid_kind(ds)}, horizontal dims {horizontal_dims(ds)}")
    return ds


def load_era5(
    repo_name: str = ERA5_REPO,
    branch: str = ERA5_BRANCH,
    group: str = ERA5_GROUP,
    chunks: dict | None = {},
    login: bool = False,
) -> xr.Dataset:
    """Open the Earthmover ERA5 surface store via Arraylake.

    chunks={} keeps on-disk chunking as dask arrays. chunks=None (as in the
    Arraylake docs) skips dask entirely -- fine for a peek, wrong for a
    multi-decade reduction.
    """
    from arraylake import Client

    client = Client()
    if login:
        client.login()
    repo = client.get_repo(repo_name)
    session = repo.readonly_session(branch)
    return xr.open_dataset(
        session.store, engine="zarr", consolidated=False,
        zarr_format=3, chunks=chunks, group=group,
    )


# ----------------------------------------------------------------------------
# 3. Spatial crop
# ----------------------------------------------------------------------------


def crop_era5_to_source(era5: xr.Dataset, src: xr.Dataset) -> xr.Dataset:
    """Bounding-box crop of ERA5 to the source footprint.

    Works for both grid kinds: min/max over a 2D lat field gives the same
    bounding box it gives over a 1D one.
    """
    lat, lon = get_latlon(src)
    lat_min, lat_max = float(lat.min()) - BBOX_PAD, float(lat.max()) + BBOX_PAD
    lon_min, lon_max = float(lon.min()) - BBOX_PAD, float(lon.max()) + BBOX_PAD

    if float(era5.longitude.max()) > 180:
        lon_lo, lon_hi = lon_min % 360, lon_max % 360
        if lon_lo > lon_hi:
            raise ValueError("Bounding box wraps the prime meridian.")
    else:
        lon_lo, lon_hi = lon_min, lon_max

    descending = float(era5.latitude[0]) > float(era5.latitude[-1])
    lat_slice = slice(lat_max, lat_min) if descending else slice(lat_min, lat_max)
    out = era5.sel(latitude=lat_slice, longitude=slice(lon_lo, lon_hi))

    # Step slice, not sortby: sortby is a fancy index and builds one task per
    # chunk, which is ruinous on the spatial layout.
    if descending:
        out = out.isel(latitude=slice(None, None, -1))

    if float(era5.longitude.max()) > 180:
        out = out.assign_coords(longitude=(((out.longitude + 180) % 360) - 180))
    if not np.all(np.diff(out.longitude.values) > 0):
        warnings.warn("longitude not monotonic; falling back to sortby")
        out = out.sortby("longitude")

    print(
        f"ERA5 cropped to {out.sizes['latitude']} x {out.sizes['longitude']} cells "
        f"({float(out.latitude.min()):.2f}..{float(out.latitude.max()):.2f} N, "
        f"{float(out.longitude.min()):.2f}..{float(out.longitude.max()):.2f} E)"
    )
    return out


# ----------------------------------------------------------------------------
# 4. Binning source cells onto the ERA5 grid
# ----------------------------------------------------------------------------


def _era5_axes(era5_crop: xr.Dataset) -> tuple[np.ndarray, np.ndarray, float, float]:
    e_lat = era5_crop.latitude.values
    e_lon = era5_crop.longitude.values
    return e_lat, e_lon, float(np.diff(e_lat).mean()), float(np.diff(e_lon).mean())


def nearest_index(vals: np.ndarray, axis: np.ndarray, d: float) -> np.ndarray:
    """Index of the nearest cell centre on a REGULAR axis. O(n), no broadcast."""
    idx = np.rint((vals - axis[0]) / d).astype(int)
    return np.clip(idx, 0, len(axis) - 1)


def build_labels(src: xr.Dataset, era5_crop: xr.Dataset):
    """Assign every source cell to an ERA5 cell.

    Rectilinear -> two 1D label arrays (fast two-step groupby).
    Curvilinear -> one 2D flat label array (single groupby over stacked points),
                   because a 2D lat field cannot be reduced per-dimension.
    """
    lat, lon = get_latlon(src)
    e_lat, e_lon, dla, dlo = _era5_axes(era5_crop)
    kind = grid_kind(src)

    if kind == "rectilinear":
        la = xr.DataArray(e_lat[nearest_index(lat.values, e_lat, dla)],
                          dims=lat.dims[0], coords={lat.dims[0]: lat})
        lo = xr.DataArray(e_lon[nearest_index(lon.values, e_lon, dlo)],
                          dims=lon.dims[0], coords={lon.dims[0]: lon})
        return kind, (la, lo)

    ilat = nearest_index(lat.values, e_lat, dla)
    ilon = nearest_index(lon.values, e_lon, dlo)
    flat = xr.DataArray(
        (ilat * len(e_lon) + ilon).astype("int64"),
        dims=lat.dims, name="cell",
    )
    return kind, (flat,)


def bin_to_era5(da: xr.DataArray, kind, labels, era5_crop, how="mean") -> xr.DataArray:
    """Aggregate a source DataArray onto the ERA5 grid."""
    e_lat, e_lon, _, _ = _era5_axes(era5_crop)

    if kind == "rectilinear":
        la, lo = labels
        g = da.assign_coords(_la=la, _lo=lo)
        g = getattr(g.groupby("_la"), how)(la.dims[0])
        g = getattr(g.groupby("_lo"), how)(lo.dims[0])
        return g.rename({"_la": "latitude", "_lo": "longitude"})

    (flat,) = labels
    hdims = flat.dims

    # Attach the labels as a plain coordinate on the stacked dim rather than
    # passing a separate DataArray to groupby. A separate array gets ALIGNED
    # against the data's stacked index, and if the source carries y/x coordinate
    # values (WRF does) while the label array does not, the intersection is empty
    # and xarray raises "the group variable's length does not match".
    da = da.transpose(..., *hdims)
    lab_vals = flat.transpose(*hdims).values.ravel()

    stacked = da.stack(_pt=hdims)
    # Strip every coord on the stacked dim, including the MultiIndex itself, so
    # nothing is left to align against.
    stacked = stacked.drop_vars(
        [c for c in stacked.coords if "_pt" in stacked[c].dims], errors="ignore"
    )
    stacked = stacked.assign_coords(cell=("_pt", lab_vals))
    g = getattr(stacked.groupby("cell"), how)("_pt")

    # groupby drops empty groups; reindex to the full grid, then unstack.
    g = g.reindex(cell=np.arange(len(e_lat) * len(e_lon)))
    mi = pd.MultiIndex.from_product([e_lat, e_lon], names=["latitude", "longitude"])
    coords = xr.Coordinates.from_pandas_multiindex(mi, "cell")
    return g.drop_vars("cell").assign_coords(coords).unstack("cell")


def coverage_fraction(
    src: xr.Dataset,
    era5_crop: xr.Dataset,
    variable: str,
    threshold: float = COVERAGE_THRESHOLD,
):
    """Fraction of each ERA5 cell backed by valid source data.

    Captures every NaN source at once: rotated domain boundary, land/ocean mask,
    and (for curvilinear grids) the projected-grid corners.
    """
    da = src[variable]
    md = member_dim(da)
    first = {"time": 0} | ({md: 0} if md else {})
    last = {"time": -1} | ({md: -1} if md else {})
    valid = da.isel(**first).notnull()

    with progress("checking valid-data mask is static..."):
        same = bool((valid == da.isel(**last).notnull()).all().compute())
    if not same:
        warnings.warn("valid-data mask varies across time/member; using first slice.")
    print(f"source valid fraction of its own bbox: {float(valid.mean()):.1%}")

    kind, labels = build_labels(src, era5_crop)
    e_lat, e_lon, _, _ = _era5_axes(era5_crop)

    counts = bin_to_era5(valid.astype("float32"), kind, labels, era5_crop, how="sum")

    lat, lon = get_latlon(src)
    if kind == "rectilinear":
        dla = float(np.abs(np.diff(lat.values)).mean())
        dlo = float(np.abs(np.diff(lon.values)).mean())
    else:
        # Mean spacing along each projected axis, in degrees.
        dla = float(np.abs(np.diff(lat.values, axis=0)).mean())
        dlo = float(np.abs(np.diff(lon.values, axis=1)).mean())
    _, _, e_dla, e_dlo = _era5_axes(era5_crop)
    expected = abs(e_dla / dla) * abs(e_dlo / dlo)
    print(f"~{expected:.0f} source cells per ERA5 cell at full coverage")

    frac = (counts / expected).clip(0, 1)
    frac = frac.reindex(latitude=e_lat, longitude=e_lon, fill_value=0.0)
    if hasattr(frac.data, "compute"):
        with progress("materializing coverage mask..."):
            frac = frac.compute()
    frac.attrs = {"long_name": "source coverage fraction", "units": "1"}

    keep = frac >= threshold
    print(f"Keeping {int(keep.sum())} of {frac.size} ERA5 cells at threshold {threshold}")
    for t in (0.0, 0.25, 0.5, 0.9):
        print(f"   frac > {t:>4}: {int((frac > t).sum()):>5} cells")
    return frac, keep, kind, labels


# ----------------------------------------------------------------------------
# 5. Common grid and cadence
# ----------------------------------------------------------------------------


def era5_monthly_wspeed(era5_crop, u=ERA5_U, v=ERA5_V,
                        time_start=TIME_START, time_end=TIME_END, freq="MS"):
    """Hourly u10/v10 -> scalar speed -> resampled mean.

    Speed BEFORE averaging: the mean of the components is not the mean of the
    scalar speed, and both AE products store the latter.
    """
    with ProgressBar():
        sub = era5_crop.sel(time=slice(time_start, time_end)).load()
    n = sub.sizes["time"]
    gb = n * sub.sizes["latitude"] * sub.sizes["longitude"] * 4 * 2 / 1e9
    print(f"ERA5 read: {n:,} hourly steps, ~{gb:.1f} GB before reduction")

    w = np.sqrt(sub[u] ** 2 + sub[v] ** 2)
    w.name = "wspeed"
    w.attrs["units"] = "m s-1"
    return w.resample(time=freq).mean()


def source_to_era5_grid(src, kind, labels, era5_crop, variable,
                        sim=SIM, time_start=TIME_START, time_end=TIME_END):
    """Block-average the source onto the ERA5 cells.

    Coarsening up, never interpolating ERA5 down: a 3 km field interpolated from
    31 km data would be fabricated structure.
    """
    da = src[variable].sel(time=slice(time_start, time_end))
    da = select_member(da, sim)
    return bin_to_era5(da, kind, labels, era5_crop, how="mean")


# ----------------------------------------------------------------------------
# 6. Metrics
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# 7. Plots and summary
# ----------------------------------------------------------------------------


def load_wrf_elevation(
    url: str = "s3://cadcat/wrf/derived-vars/elevation_wrf.nc",
    anon: bool = True,
) -> xr.Dataset:
    """Load the WRF-grid elevation field from the public cadcat bucket.

    Curvilinear, same projection as the WRF runs, so it bins to the ERA5 grid
    with the same labels. Inspect dims/coords before use: the file may carry
    more than one nested domain.
    """
    import fsspec

    with fsspec.open(url, anon=anon) as f:
        elev = xr.open_dataset(f).load()
    print("elevation vars:", list(elev.data_vars))
    print("elevation dims:", dict(elev.sizes))
    return elev


def elevation_on_era5(elev, era5_crop, var: str | None = None,
                      kind=None, labels=None):
    """Bin elevation to the ERA5 grid, returning mean height AND sub-grid relief.

    The elevation file need NOT be on the same WRF nest as the wind data -- pass
    the elevation Dataset and labels are derived from its own lat/lon. The ERA5
    grid is the common frame. Passing kind/labels explicitly is only correct when
    the elevation shares the wind grid exactly.

    The standard deviation of native-resolution terrain inside each 0.25 deg cell
    measures what ERA5 cannot resolve, and predicts a resolution-driven
    discrepancy far better than mean height: a flat plateau and a rugged range
    can share an elevation but not a relief.
    """
    if isinstance(elev, xr.Dataset):
        if var is None:
            var = list(elev.data_vars)[0]
            print(f"using elevation variable '{var}'")
        elev_da, src = elev[var], elev
    else:
        elev_da, src = elev, elev.to_dataset(name=elev.name or "elevation")

    if labels is None:
        kind, labels = build_labels(src, era5_crop)
        print(f"elevation grid: {kind}, {elev_da.shape}")

    hmean = bin_to_era5(elev_da, kind, labels, era5_crop, how="mean")
    hstd = bin_to_era5(elev_da, kind, labels, era5_crop, how="std")
    hmean.name, hstd.name = "elev_mean", "elev_std"
    hmean.attrs["units"] = hstd.attrs["units"] = "m"
    return hmean, hstd


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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(preset: str = "loca2", sim=SIM, coverage_threshold=COVERAGE_THRESHOLD,
         time_start=TIME_START, time_end=TIME_END, ds_source=None, ds_era5=None,
         output_nc=OUTPUT_NC, output_png=OUTPUT_PNG, freq=None,
         return_context=False, **overrides):
    """Run the pipeline for a named preset. Any preset key can be overridden.

    return_context=True also returns a dict holding the aligned source and ERA5
    series plus the grid objects, which the seasonality, trend, ensemble and
    station analyses all need and which the metrics alone cannot provide.
    """
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset '{preset}'. Choose from {list(PRESETS)}")
    cfg = dict(PRESETS[preset]) | overrides
    variable, label = cfg["variable"], cfg["label"]
    # ERA5 must be aggregated to the SOURCE cadence, not always monthly.
    if freq is None:
        freq = FREQ_FOR_TABLE.get(cfg["table_id"], "MS")
    print(f"=== {label}: {variable}, {cfg['table_id']} ({freq}), "
          f"{time_start} to {time_end} ===")

    if ds_source is None:
        ds_source = load_source(**cfg)
    if ds_era5 is None:
        ds_era5 = load_era5()[["u10", "v10"]]

    era5_crop = crop_era5_to_source(ds_era5, ds_source)
    frac, keep, kind, labels = coverage_fraction(
        ds_source, era5_crop, variable=variable, threshold=coverage_threshold)

    era5_mon = era5_monthly_wspeed(era5_crop, time_start=time_start,
                                   time_end=time_end, freq=freq)
    src_mon = source_to_era5_grid(ds_source, kind, labels, era5_crop, variable,
                                  sim=sim, time_start=time_start, time_end=time_end)

    src_mon, era5_mon = xr.align(src_mon, era5_mon, join="inner")
    print(f"aligned on {src_mon.sizes['time']} steps")
    if src_mon.sizes["time"] == 0:
        raise ValueError(
            f"No overlapping timestamps at freq='{freq}'. Daily AE data is often "
            "stamped at 12:00 while resample('1D') gives 00:00 -- if so, floor the "
            "source time axis before aligning.")

    with progress("reading ERA5..."):
        era5_m = era5_mon.where(keep).compute()
    with progress("reading source..."):
        src_m = src_mon.where(keep).compute()

    with progress("computing metrics..."):
        metrics = agreement_metrics(src_m, era5_m,
                                    synchronized=cfg["synchronized"]).compute()
    metrics["coverage_fraction"] = frac
    metrics.attrs["source"] = label
    metrics.attrs["variable"] = variable

    plot_agreement(metrics, path=output_png, label=label)
    regional_summary(metrics, keep)
    metrics.to_netcdf(output_nc)
    print(f"wrote {output_nc}")

    if return_context:
        ctx = {
            "src": src_m, "era5": era5_m,      # aligned, masked, computed
            "keep": keep, "frac": frac,
            "kind": kind, "labels": labels,
            "era5_crop": era5_crop, "ds_source": ds_source,
            "variable": variable, "label": label,
            "synchronized": cfg["synchronized"], "freq": freq,
        }
        return metrics, ctx
    return metrics


def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="loca2", choices=list(PRESETS))
    for k in ("catalog", "activity-id", "experiment-id", "table-id",
              "grid-label", "variable", "institution-id", "source-id"):
        p.add_argument(f"--{k}", default=None)
    p.add_argument("--sim", default=str(SIM))
    p.add_argument("--coverage-threshold", type=float, default=COVERAGE_THRESHOLD)
    p.add_argument("--time-start", default=TIME_START)
    p.add_argument("--time-end", default=TIME_END)
    p.add_argument("--output-nc", default=OUTPUT_NC)
    p.add_argument("--output-png", default=OUTPUT_PNG)
    a = p.parse_args()

    overrides = {k.replace("-", "_"): getattr(a, k.replace("-", "_"))
                 for k in ("catalog", "activity-id", "experiment-id", "table-id",
                           "grid-label", "variable", "institution-id", "source-id")
                 if getattr(a, k.replace("-", "_")) is not None}
    sim = int(a.sim) if a.sim.lstrip("-").isdigit() else a.sim

    main(preset=a.preset, sim=sim, coverage_threshold=a.coverage_threshold,
         time_start=a.time_start, time_end=a.time_end,
         output_nc=a.output_nc, output_png=a.output_png, **overrides)


if __name__ == "__main__":
    _cli()


# ----------------------------------------------------------------------------
# 8. Seasonality, trends, ensembles, stations, conservative regridding
# ----------------------------------------------------------------------------

SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
           "JJA": [6, 7, 8], "SON": [9, 10, 11]}


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


def sample_grid_at_points(da: xr.DataArray, lats, lons,
                          dim: str = "station_id", ids=None,
                          max_dist_deg: float = 0.1):
    """Sample a gridded field at scattered points (e.g. station locations).

    Handles both grid kinds. Rectilinear uses xarray's nearest selection;
    curvilinear needs nearest-neighbour in index space, because 2D lat/lon
    cannot be used as selection indexes.

    Returns (sampled, distance_deg). The distance doubles as a domain test:
    a point far from any cell centre is outside the grid, which is more
    honest than a polygon test at the boundary.
    """
    lats = np.asarray(lats, dtype="float64")
    lons = np.asarray(lons, dtype="float64")
    ids = np.arange(len(lats)) if ids is None else np.asarray(ids)

    ds_wrap = da.to_dataset(name=da.name or "var")
    if grid_kind(ds_wrap) == "rectilinear":
        la, lo = get_latlon(ds_wrap)
        ila = xr.DataArray(lats, dims=dim, coords={dim: ids})
        ilo = xr.DataArray(lons, dims=dim, coords={dim: ids})
        out = da.sel({la.dims[0]: ila, lo.dims[0]: ilo}, method="nearest")
        d = np.hypot(out[la.name].values - lats, out[lo.name].values - lons)
    else:
        from scipy.spatial import cKDTree

        la, lo = get_latlon(ds_wrap)
        tree = cKDTree(np.column_stack([la.values.ravel(), lo.values.ravel()]))
        d, idx = tree.query(np.column_stack([lats, lons]))
        iy, ix = np.unravel_index(idx, la.shape)
        hd = horizontal_dims(ds_wrap)
        out = da.isel({hd[0]: xr.DataArray(iy, dims=dim, coords={dim: ids}),
                       hd[1]: xr.DataArray(ix, dims=dim, coords={dim: ids})})

    far = int((d > max_dist_deg).sum())
    if far:
        print(f"{far} of {len(lats)} points are >{max_dist_deg} deg from any "
              "cell centre (likely outside the grid)")
    return out, d


def cell_edges_1d(centres: np.ndarray) -> np.ndarray:
    """Cell edges from centres on a 1D axis (n -> n+1).

    Midpoints between centres, with the two outer edges extrapolated by half a
    spacing. Exact for a uniform axis, which ERA5 and LOCA2 both are.
    """
    c = np.asarray(centres, dtype="float64")
    mid = (c[:-1] + c[1:]) / 2
    return np.concatenate([[c[0] - (c[1] - c[0]) / 2],
                           mid,
                           [c[-1] + (c[-1] - c[-2]) / 2]])


def cell_corners_2d(lat2d: np.ndarray, lon2d: np.ndarray):
    """Corner arrays (ny+1, nx+1) from 2D cell centres (ny, nx).

    Conservative regridding needs cell BOUNDARIES, and a curvilinear grid does
    not carry them. They are reconstructed by padding the centre array one cell
    in every direction by linear extrapolation, then averaging each 2x2 block:
    an interior corner is the mean of the four centres surrounding it.

    Accurate to well under a cell width for a smooth projected grid like WRF's
    Lambert conformal at 3-4 km. It would NOT be safe across a pole or the
    dateline, neither of which this domain touches.
    """
    def pad(a):
        a = np.asarray(a, dtype="float64")
        ny, nx = a.shape
        p = np.empty((ny + 2, nx + 2))
        p[1:-1, 1:-1] = a
        p[1:-1, 0] = 2 * a[:, 0] - a[:, 1]            # left
        p[1:-1, -1] = 2 * a[:, -1] - a[:, -2]         # right
        p[0, 1:-1] = 2 * a[0, :] - a[1, :]            # bottom
        p[-1, 1:-1] = 2 * a[-1, :] - a[-2, :]         # top
        p[0, 0] = 2 * p[1, 0] - p[2, 0]               # corners
        p[0, -1] = 2 * p[1, -1] - p[2, -1]
        p[-1, 0] = 2 * p[-2, 0] - p[-3, 0]
        p[-1, -1] = 2 * p[-2, -1] - p[-3, -1]
        return p

    pl, po = pad(lat2d), pad(lon2d)
    lat_b = 0.25 * (pl[:-1, :-1] + pl[:-1, 1:] + pl[1:, :-1] + pl[1:, 1:])
    lon_b = 0.25 * (po[:-1, :-1] + po[:-1, 1:] + po[1:, :-1] + po[1:, 1:])
    return lat_b, lon_b


def add_bounds(ds: xr.Dataset) -> xr.Dataset:
    """Attach lat_b/lon_b corner coordinates, for either grid kind."""
    lat, lon = get_latlon(ds)
    if lat.ndim == 1:
        return ds.assign_coords(lat_b=cell_edges_1d(lat.values),
                                lon_b=cell_edges_1d(lon.values))
    lat_b, lon_b = cell_corners_2d(lat.values, lon.values)
    dims = tuple(d + "_b" for d in lat.dims)
    return ds.assign_coords(lat_b=(dims, lat_b), lon_b=(dims, lon_b))


def regrid_conservative(src: xr.Dataset, target: xr.Dataset, variable: str,
                        method: str = "conservative",
                        mask: xr.DataArray | None = None,
                        reuse_weights: bool = False,
                        filename: str | None = None):
    """Area-weighted regridding with xesmf, for either grid kind.

    The nearest-centre binning used elsewhere assigns each source cell wholly
    to one target cell. Conservative regridding weights by true overlap area,
    which matters most where a target cell is only partly covered -- the
    coastline and the rotated domain edge, which are also where the terrain
    mismatch is worst.

    Corner arrays are built by `add_bounds`, so curvilinear sources work
    without externally supplied bounds.

    Requires xesmf (conda-forge). `reuse_weights` with a `filename` caches the
    weight matrix, which is the expensive part when regridding many fields.
    """
    try:
        import xesmf as xe
    except ImportError as e:
        raise ImportError(
            "conservative regridding needs xesmf:\n"
            "  conda install -c conda-forge xesmf\n"
            "The nearest-centre path (bin_to_era5) and the area-weighted path "
            "(bin_to_era5_weighted) need no extra dependency."
        ) from e

    if method.startswith("conservative"):
        src = add_bounds(src)
        target = add_bounds(target)

    rg = xe.Regridder(src, target, method=method, periodic=False,
                      reuse_weights=reuse_weights, filename=filename)
    out = rg(src[variable], keep_attrs=True)
    return out.where(mask) if mask is not None else out


def bin_to_era5_weighted(da: xr.DataArray, kind, labels, era5_crop,
                         src_lat=None):
    """Area-weighted block mean onto the ERA5 grid, without xesmf.

    A middle ground between nearest-centre binning and full conservative
    regridding: each source cell still belongs wholly to one target cell, but
    contributes in proportion to its area (cos(latitude), since the cells are
    equal in angle rather than in area). Removes the latitude bias of a plain
    count-weighted mean; does not fix cells straddling a target boundary.
    """
    if src_lat is None:
        raise ValueError("src_lat (the source latitude field) is required")

    w = np.cos(np.deg2rad(src_lat))
    w = xr.DataArray(w, dims=src_lat.dims,
                     coords={d: src_lat[d] for d in src_lat.dims
                             if d in src_lat.coords})
    num = bin_to_era5(da * w, kind, labels, era5_crop, how="sum")
    den = bin_to_era5(xr.ones_like(da) * w, kind, labels, era5_crop, how="sum")
    return num / den



# ----------------------------------------------------------------------------
# 9. Persisting intermediate results
# ----------------------------------------------------------------------------
#
# The expensive part of every run is the ERA5 hourly read. Saving the aligned
# arrays means a reviewer can reproduce every figure and table without it.
#
# A context holds two kinds of thing:
#   * computed arrays (src, era5, keep, frac) -- worth saving
#   * lazy grid objects (ds_source, era5_crop, labels) -- metadata only, cheap
#     to rebuild, and not serialisable as-is
# save_context stores the first; attach_grid rebuilds the second on demand.


def save_context(ctx: dict, path: str) -> str:
    """Write a run's aligned arrays to netCDF."""
    ds = xr.Dataset({
        "src": ctx["src"],
        "era5": ctx["era5"],
        "keep": ctx["keep"].astype("int8"),   # netCDF has no bool type
        "frac": ctx["frac"],
    })
    ds.attrs.update(
        variable=ctx["variable"],
        label=ctx["label"],
        kind=ctx["kind"],
        freq=ctx.get("freq", "MS"),
        synchronized=int(bool(ctx["synchronized"])),
    )
    ds = clear_encoding_ds(ds)
    ds.to_netcdf(path)
    print(f"wrote {path}  ({ds.sizes.get('time', 0)} steps)")
    return path


def load_context(path: str) -> dict:
    """Read back what save_context wrote. Grid objects come from attach_grid."""
    ds = xr.open_dataset(path)
    return {
        "src": ds["src"], "era5": ds["era5"],
        "keep": ds["keep"].astype(bool), "frac": ds["frac"],
        "variable": ds.attrs["variable"], "label": ds.attrs["label"],
        "kind": ds.attrs["kind"], "freq": ds.attrs.get("freq", "MS"),
        "synchronized": bool(int(ds.attrs["synchronized"])),
    }


def attach_grid(ctx: dict, preset: str, ds_era5: xr.Dataset,
                table_id: str | None = None, **overrides) -> dict:
    """Rebuild the lazy grid objects a loaded context lacks.

    Needed only by the ensemble and station sections, which index the full
    source dataset. Catalog and crop operations are metadata-only, so this is
    seconds rather than minutes -- but it does need network access.
    """
    cfg = dict(PRESETS[preset]) | overrides
    if table_id is not None:
        cfg["table_id"] = table_id
    ds_source = load_source(**cfg)
    era5_crop = crop_era5_to_source(ds_era5, ds_source)
    kind, labels = build_labels(ds_source, era5_crop)
    ctx = dict(ctx)
    ctx.update(ds_source=ds_source, era5_crop=era5_crop,
               kind=kind, labels=labels)
    return ctx


def clear_encoding_ds(ds: xr.Dataset) -> xr.Dataset:
    """Drop inherited per-variable encoding before writing."""
    ds = ds.copy()
    for v in list(ds.variables):
        ds[v].encoding = {}
    return ds


# ----------------------------------------------------------------------------
# 10. Multi-model analysis
# ----------------------------------------------------------------------------
#
# WRF models are separate catalog entries (one query each); LOCA2 members live
# on a `sim` dimension inside one dataset. Both reduce to "iterate over named
# DataArrays sharing a grid", which is what metrics_over_members consumes.
#
# The ERA5 hourly read dominates every run, so it is done ONCE per product
# family and reused across every model. 51 runs then cost 2 ERA5 reads.

# WRF simulations WITHOUT a-priori bias adjustment, from climakitae's
# NON_WRF_BA_MODELS. The remaining GCM-driven runs are the bias-adjusted set.
WRF_NON_BA = ("CESM2", "CNRM-ESM2-1", "FGOALS-g3", "ensmean")

WRF_BA_MODELS = ("EC-Earth3", "EC-Earth3-Veg", "MIROC6",
                 "MPI-ESM1-2-HR", "TaiESM1")


def wrf_model_list(bias_adjusted: bool = True, variable: str = "wspd10mean",
                   table_id: str = "mon", grid_label: str = "d03",
                   experiment_id: str = "historical") -> list[str]:
    """WRF GCM-driven source_ids available for a variable.

    Reads the catalog CSV shipped with climakitae, so no network access.
    """
    import importlib.util
    import pathlib

    # Locate the shipped catalog without importing climakitae, so this works
    # even when the package cannot be imported (e.g. a source checkout).
    spec = importlib.util.find_spec("climakitae")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("climakitae not found on the path")
    root = pathlib.Path(list(spec.submodule_search_locations)[0])
    df = pd.read_csv(root / "data" / "catalogs.csv")
    sel = df[(df.activity_id == "WRF") & (df.variable_id == variable)
             & (df.table_id == table_id) & (df.grid_label == grid_label)
             & (df.experiment_id == experiment_id)]
    models = sorted(m for m in sel.source_id.unique() if m != "ERA5")
    if bias_adjusted:
        models = [m for m in models if m not in WRF_NON_BA]
    print(f"{len(models)} WRF models"
          f"{' (bias-adjusted only)' if bias_adjusted else ''}: {models}")
    return models


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


def loca2_members(ds_source: xr.Dataset, variable: str,
                  time_start=TIME_START, time_end=TIME_END, members=None):
    """Yield (sim_name, DataArray) for LOCA2 members."""
    da = ds_source[variable].sel(time=slice(time_start, time_end))
    md = member_dim(da)
    if md is None:
        yield "single", da
        return
    names = list(da[md].values) if members is None else list(members)
    for n in names:
        yield str(n), da.sel({md: n})


def wrf_members(models, ds_era5, variable="wspd10mean", table_id="mon",
                grid_label="d03", experiment_id="historical",
                time_start=TIME_START, time_end=TIME_END, **overrides):
    """Yield (model, DataArray) for WRF models, plus the shared grid objects.

    All WRF GCM-driven runs share the d03 domain, so the crop, labels and
    coverage mask are built once from the first model and reused. Returns
    (generator, kind, labels, era5_crop, keep, frac).
    """
    first = load_source(catalog="cadcat", activity_id="WRF",
                        experiment_id=experiment_id, table_id=table_id,
                        grid_label=grid_label, variable=variable,
                        institution_id="UCLA", source_id=models[0],
                        processes={"filter_unadjusted_models": "no"},
                        **overrides)
    era5_crop = crop_era5_to_source(ds_era5, first)
    frac, keep, kind, labels = coverage_fraction(first, era5_crop,
                                                 variable=variable)

    def gen():
        for m in models:
            ds = (first if m == models[0] else
                  load_source(catalog="cadcat", activity_id="WRF",
                              experiment_id=experiment_id, table_id=table_id,
                              grid_label=grid_label, variable=variable,
                              institution_id="UCLA", source_id=m,
                              processes={"filter_unadjusted_models": "no"},
                              **overrides))
            da = ds[variable].sel(time=slice(time_start, time_end))
            if member_dim(da) is not None:
                da = da.isel({member_dim(da): 0})
            yield m, da

    return gen(), kind, labels, era5_crop, keep, frac


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


# ----------------------------------------------------------------------------
# 11. CONUS404
# ----------------------------------------------------------------------------
#
# CONUS404 is 4 km WRF over CONUS, ERA5-driven, water years 1980-2024, produced
# by NCAR with the USGS. It is the sharpest available comparator for this
# analysis: same resolution class as the AE WRF (4 km vs 3 km) and the same
# driver as WRF-ERA5, so a comparison against it isolates model configuration
# rather than resolution.
#
# Access is anonymous zarr on the USGS Open Storage Network pod -- no
# credentials, no egress fees.
#
# TWO THINGS THAT SHAPE THE IMPLEMENTATION:
#
# 1. There is no wind-speed variable, only U10 and V10. Speed must therefore be
#    formed hourly and averaged afterwards. The conus404_monthly product holds
#    monthly-mean COMPONENTS; taking their magnitude yields the vector average,
#    which is biased low by roughly 20% in this domain -- the same penalty
#    measured for ERA5. Do not use it for wind speed.
#
# 2. U10/V10 are grid-relative ("with respect to model grid"), not
#    earth-relative. For wind SPEED this is irrelevant: the magnitude is
#    invariant under the rotation, so no COSALPHA/SINALPHA correction is
#    needed. It would be needed for direction, or for u/v separately.

CONUS404_ENDPOINT = "https://usgs.osn.mghpcc.org/"
CONUS404_URLS = {
    "hourly": "s3://hytest/conus404/conus404_hourly.zarr",
    "daily": "s3://hytest/conus404/conus404_daily.zarr",
    "monthly": "s3://hytest/conus404/conus404_monthly.zarr",
}


def load_conus404(product: str = "hourly", variables=("U10", "V10"),
                  keep_static=("HGT", "LANDMASK", "VAR_SSO"),
                  chunks: dict | None = {}) -> xr.Dataset:
    """Open CONUS404 from the USGS OSN pod (anonymous, no egress fees).

    `keep_static` retains the terrain height, land mask and sub-grid orographic
    variance, which are useful for the terrain analysis and cost nothing (they
    are 2D fields).
    """
    import fsspec

    url = CONUS404_URLS[product]
    fs = fsspec.filesystem(
        "s3", anon=True, client_kwargs={"endpoint_url": CONUS404_ENDPOINT})
    ds = xr.open_dataset(fs.get_mapper(url), engine="zarr",
                         consolidated=True, chunks=chunks)

    wanted = [v for v in list(variables) + list(keep_static) if v in ds]
    missing = set(variables) - set(ds.data_vars)
    if missing:
        raise KeyError(f"CONUS404 {product} lacks {missing}. "
                       f"Available wind-like: "
                       f"{[v for v in ds.data_vars if 'U' in v or 'V' in v][:10]}")
    out = ds[wanted]
    print(f"CONUS404 {product}: {dict(out.sizes)}")
    print(f"  time {str(out.time.values[0])[:10]} -> {str(out.time.values[-1])[:10]}")
    print(f"  chunks {out[variables[0]].chunksizes}")
    return out


def conus404_wspeed(ds: xr.Dataset, time_start: str, time_end: str,
                    freq: str = "MS", bbox=None) -> xr.DataArray:
    """10 m scalar wind speed from CONUS404, formed hourly then averaged.

    `bbox` is (lat_min, lat_max, lon_min, lon_max). Subsetting in projected
    x/y before reading is what makes this affordable: the store is CONUS-wide
    with 175x175 spatial chunks, so a California box touches a handful of
    chunks instead of all 48.
    """
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        inside = ((ds.lat >= lat_min) & (ds.lat <= lat_max)
                  & (ds.lon >= lon_min) & (ds.lon <= lon_max)).compute()
        ys = np.where(inside.any("x"))[0]
        xs = np.where(inside.any("y"))[0]
        ds = ds.isel(y=slice(ys.min(), ys.max() + 1),
                     x=slice(xs.min(), xs.max() + 1))
        print(f"  subset to y={ds.sizes['y']}, x={ds.sizes['x']}")

    sub = ds.sel(time=slice(time_start, time_end))
    n = sub.sizes["time"]
    gb = n * sub.sizes["y"] * sub.sizes["x"] * 4 * 2 / 1e9
    print(f"  CONUS404 read: {n:,} steps, ~{gb:.1f} GB before reduction")
    if gb > 100:
        warnings.warn(
            f"~{gb:.0f} GB read. CONUS404 hourly is far heavier than ERA5 "
            "because its chunks are spatial tiles rather than time-runs. "
            "Consider a shorter window.")

    # Speed BEFORE averaging. Magnitude is invariant under the grid rotation,
    # so grid-relative components are fine here.
    w = np.sqrt(sub.U10 ** 2 + sub.V10 ** 2)
    w.name = "wspeed"
    w.attrs.update(units="m s-1",
                   long_name="10 m scalar wind speed, hourly then averaged")
    return w.resample(time=freq).mean()


def regrid_nearest(src_da: xr.DataArray, tgt_lat: np.ndarray,
                   tgt_lon: np.ndarray, tgt_dims: tuple,
                   tgt_coords: dict | None = None,
                   max_dist_deg: float = 0.15):
    """Nearest-neighbour resample of a gridded field onto another grid.

    For each TARGET cell, take the nearest SOURCE cell. This is the right
    operation when source and target resolutions are comparable (3 km AE
    against 4 km CONUS404): block-mean binning would leave some target cells
    with no contributing source cell and others with two, which aliases.
    Block-mean binning remains correct when the target is much coarser, as
    with the 0.25 deg ERA5 grid.

    Returns (resampled, distance_deg) with the target's dims and coords.
    """
    flat_lat = np.asarray(tgt_lat).ravel()
    flat_lon = np.asarray(tgt_lon).ravel()

    sampled, dist = sample_grid_at_points(src_da, flat_lat, flat_lon,
                                          dim="_pt",
                                          ids=np.arange(flat_lat.size),
                                          max_dist_deg=max_dist_deg)

    shape = np.asarray(tgt_lat).shape
    other = [d for d in sampled.dims if d != "_pt"]
    out = sampled.transpose(*other, "_pt")
    out = out.data.reshape(*[sampled.sizes[d] for d in other], *shape)

    dims = tuple(other) + tuple(tgt_dims)
    coords = {d: sampled[d] for d in other if d in sampled.coords}
    if tgt_coords:
        coords.update(tgt_coords)
    da = xr.DataArray(out, dims=dims, coords=coords, name=src_da.name)

    d = dist.reshape(shape)
    far = int((d > max_dist_deg).sum())
    if far:
        print(f"    {far} of {d.size} target cells are >{max_dist_deg} deg "
              "from any source cell (outside the source domain)")
    return da, xr.DataArray(d, dims=tgt_dims, coords=tgt_coords or {})


def compare_regridding(src: xr.Dataset, variable: str, era5_crop: xr.Dataset,
                       kind, labels, keep: xr.DataArray,
                       frac: xr.DataArray | None = None,
                       time_start=None, time_end=None):
    """Nearest-centre binning against xesmf conservative, quantified.

    Answers the question the caveat raises: does the regridding choice change
    any conclusion? Reports the difference overall and split by coverage
    fraction, because whole-cell assignment is exact for a fully covered target
    cell and wrong only where a cell is partly covered -- the coastline and the
    rotated domain edge.
    """
    da = src[variable]
    if member_dim(da) is not None:
        da = da.isel({member_dim(da): 0})
    if time_start is not None:
        da = da.sel(time=slice(time_start, time_end))

    near = bin_to_era5(da, kind, labels, era5_crop, how="mean").mean("time")
    near = near.where(keep).compute()

    target = era5_crop[[list(era5_crop.data_vars)[0]]].isel(time=0, drop=True)
    cons = regrid_conservative(da.mean("time").to_dataset(name=variable),
                               target, variable)
    cons = cons.where(keep).compute()

    diff = cons - near
    rel = diff / near * 100

    print(f"nearest-centre mean : {float(np.nanmedian(near.values)):.4f} m/s")
    print(f"conservative   mean : {float(np.nanmedian(cons.values)):.4f} m/s")
    print(f"median difference   : {float(np.nanmedian(diff.values)):+.4f} m/s "
          f"({float(np.nanmedian(rel.values)):+.2f} %)")
    print(f"max |difference|    : {float(np.nanmax(np.abs(diff.values))):.4f} m/s")

    if frac is not None:
        print("\n by coverage fraction:")
        print(f"   {'band':<12}{'n':>6}{'median diff':>14}{'max |diff|':>13}")
        for lo, hi in ((0.5, 0.9), (0.9, 0.99), (0.99, 1.01)):
            sel = keep & (frac >= lo) & (frac < hi)
            n = int(sel.sum())
            if n == 0:
                continue
            d = diff.where(sel).values
            print(f"   {lo:.2f}-{hi:<7.2f}{n:>6}"
                  f"{float(np.nanmedian(d)):>+14.4f}"
                  f"{float(np.nanmax(np.abs(d))):>13.4f}")
        print("\n Whole-cell assignment is exact for a fully covered cell, so"
              "\n any real difference should concentrate in the low-coverage band.")

    return xr.Dataset({"nearest": near, "conservative": cons,
                       "diff": diff, "rel_diff_pct": rel})


# ----------------------------------------------------------------------------
# 12. Spatial pattern fidelity
# ----------------------------------------------------------------------------
#
# Every metric so far is computed per cell and then summarised, which answers
# "is the level right" but never "are the maxima in the right places". Pattern
# metrics answer the second question, and deliberately remove the mean offset
# first -- a product can be 50% too strong everywhere and still reproduce the
# spatial structure perfectly, which is exactly the situation here.


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


def cos_weights(da: xr.DataArray) -> xr.DataArray:
    """cos(latitude) weights, for either grid kind."""
    lat = da["lat"] if "lat" in da.coords else da["latitude"]
    return np.cos(np.deg2rad(lat))


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


# ----------------------------------------------------------------------------
# 13. Extremes
# ----------------------------------------------------------------------------
#
# Availability decides a lot here, and is worth stating plainly:
#
#   WRF (AE)     wspd10max at day and mon  -- a genuine maximum-wind variable
#   LOCA2        NOTHING. Only `wspeed` (a mean). LOCA2 cannot support any
#                extreme-wind use case, whatever its bias.
#   ERA5         no max variable, but hourly -> daily max is derivable
#   CONUS404     no max variable, but hourly -> daily max is derivable
#   HDP stations hourly -> daily max is derivable
#
# SAMPLING CAVEAT, which matters more for maxima than for means: WRF's
# wspd10max is a maximum over MODEL TIMESTEPS (tens of seconds), while a
# maximum derived from hourly ERA5 or station data samples 24 times a day. The
# WRF value is therefore higher by construction, and the gap widens for gusty,
# convective or terrain-driven conditions. Treat WRF-vs-reanalysis maxima
# comparisons as bounded, not exact.


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