"""
event_helpers.py

Shared code for the extreme-event notebooks (heat_event.ipynb, precip_event.ipynb):
config loading + every output filename, HDP station pull, WRF warming-level pull,
event counting, census-tract mapping, CRAI hazard/index scores, and Plotly figures.

Only code used by more than one notebook lives here — hazard-specific metrics stay
in their own notebook. The statewide CRAI inputs these functions read are built
once by crai_base.ipynb (see crai_cache/).
"""

import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

import crai_helpers

# This file lives in <framework>/helpers/ — the framework root is one level up.
FRAMEWORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CRAI_CACHE = os.path.join(FRAMEWORK_DIR, "crai_cache")


# ===========================================================================
# Config
# ===========================================================================
def gwl_slug(level):
    """CRAI file-naming convention for a warming level, e.g. 0.8 -> 'gwl08', 2.0 -> 'gwl2'."""
    digits = f"{level:.2f}".replace(".", "").rstrip("0") or "0"
    return f"gwl{digits}"


def _require(raw, path, *keys):
    node = raw
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing required config key: {'.'.join(keys)} (loaded from {path})")
        node = node[key]
    return node


class EventConfig:
    """Everything read from event_config.yaml, plus every path the notebooks write to.

    Relative paths in the config (zip_shapefile, crai.cache_dir) resolve against
    the config file's own folder — the post folder — not the kernel's cwd.
    """

    def __init__(self, raw, path):
        self.raw = raw
        self.path = os.path.abspath(path)
        self.post_dir = os.path.dirname(self.path)

        def req(*keys):
            return _require(raw, path, *keys)

        def in_post_dir(p):
            return p if os.path.isabs(p) else os.path.join(self.post_dir, p)

        self.hazard = req("hazard")
        self.event_name = req("event_name")

        self.station_label = req("location", "station_label")
        self.place_name = req("location", "place_name")
        self.clip = req("location", "clip")
        self.zip_shapefile = in_post_dir(req("location", "zip_shapefile"))
        self.place_slug = self.place_name.lower().replace(" ", "_")

        self.event_year = req("event", "year")
        self.event_windows = [tuple(w) for w in req("event", "windows")]

        # Hazard-specific — each hazard notebook validates its own keys.
        self.thresholds = req("thresholds")

        proj = req("projection")
        self.warming_levels = req("projection", "warming_levels")
        self.warming_level_window = req("projection", "warming_level_window")
        self.baseline_warming_level = req("projection", "baseline_warming_level")
        self.percentile = req("projection", "percentile")
        self.window_years = req("projection", "window_years")
        self.sublabels = proj.get("sublabels") or None
        self.exclude_simulations = proj.get("exclude_simulations") or []

        crai = raw.get("crai") or {}
        self.crai_cache_dir = in_post_dir(crai["cache_dir"]) if crai.get("cache_dir") else DEFAULT_CRAI_CACHE
        self.crai_exposure_metrics = crai.get("exposure_metrics")  # None -> notebook default
        self.crai_loss_metrics = crai.get("loss_metrics")

        self.plot_extent = raw.get("plot_extent") or {}

        # Derived from projection.warming_levels/sublabels so every part of the
        # notebook references the same warming levels.
        self.gwl_order = [gwl_slug(wl) for wl in self.warming_levels]
        subs = (
            self.sublabels
            if self.sublabels and len(self.sublabels) == len(self.warming_levels)
            else [None] * len(self.warming_levels)
        )
        self.gwl_labels = {
            slug: (f"{sub} ({wl}°C)" if sub else f"{wl}°C")
            for slug, wl, sub in zip(self.gwl_order, self.warming_levels, subs)
        }

        # Output folders — all inside the post folder.
        self.figures_html_dir = os.path.join(self.post_dir, "figures", "html")
        self.figures_static_dir = os.path.join(self.post_dir, "figures", "static")
        self.crai_out_dir = os.path.join(self.post_dir, "crai_layers")

    # ---- every output filename comes from these methods -------------------
    def make_output_dirs(self):
        for d in (self.figures_html_dir, self.figures_static_dir,
                  os.path.join(self.crai_out_dir, "tract_metrics"),
                  os.path.join(self.crai_out_dir, "hazard_scores"),
                  os.path.join(self.crai_out_dir, "crai_scores")):
            os.makedirs(d, exist_ok=True)

    def figure_paths(self, name):
        """(png_path, html_path) for a figure name like 'return_period_by_gwl'."""
        stem = f"fig_{name}"
        return (os.path.join(self.figures_static_dir, f"{stem}.png"),
                os.path.join(self.figures_html_dir, f"{stem}.html"))

    def _layer_csv(self, kind, metric, gwl):
        return os.path.join(self.crai_out_dir, kind,
                            f"{self.hazard}_{metric}_{self.place_slug}_{gwl}.csv")

    def tract_metric_csv(self, metric, gwl):
        return self._layer_csv("tract_metrics", metric, gwl)

    def hazard_score_csv(self, metric, gwl):
        return self._layer_csv("hazard_scores", metric, gwl)

    def crai_score_csv(self, metric, gwl):
        return self._layer_csv("crai_scores", metric, gwl)


def load_config(path=None):
    """Load the event config: `path`, else $EVENT_CONFIG, else ./event_config.yaml."""
    path = path or os.environ.get("EVENT_CONFIG", "event_config.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = EventConfig(raw, path)

    # Validate the clip region now — a bad key would otherwise only surface deep
    # inside the slow WRF pull.
    from climakitae.new_core.param_validation.clip_param_validator import validate_clip_param
    if not validate_clip_param(cfg.clip):
        raise ValueError(
            f"location.clip '{cfg.clip}' is not a recognized climakitae boundary key. "
            f"See the warning logged above for close matches, or call "
            f"DataCatalog().list_clip_boundaries() to browse all valid options."
        )
    print(f"Loaded '{cfg.event_name}' ({cfg.hazard}) from {cfg.path}")
    return cfg


# ===========================================================================
# Observed station data (HDP)
# ===========================================================================
HDP_NETWORK_ID = "ASOSAWOS"


def resolve_hdp_station(station_label, network_id=HDP_NETWORK_ID):
    """Resolve an ICAO label (e.g. 'KSAC') to its HDP station_id, or raise."""
    from climakitae.new_core.data_access.data_access import DataCatalog
    from climakitae.new_core.processors.processor_utils import resolve_airport_code_to_hdp_station_id

    # The resolver returns the identifier unchanged on no match rather than
    # raising, so confirm the result exists in the live HDP catalog.
    stations = DataCatalog()["stations"]
    station_id = resolve_airport_code_to_hdp_station_id(station_label, stations)
    hdp_df = DataCatalog().hdp.df
    if not ((hdp_df["network_id"] == network_id) & (hdp_df["station_id"] == station_id)).any():
        raise ValueError(
            f"station_label '{station_label}' did not resolve to a valid HDP station "
            f"(got '{station_id}'). Available labels: {sorted(stations['ID'].unique())}."
        )
    print(f"Resolved '{station_label}' -> network_id={network_id!r}, station_id={station_id!r}")
    return station_id


def pull_hdp_station(station_id, network_id=HDP_NETWORK_ID):
    """Full hourly record for one HDP station, computed into memory."""
    import climakitae as ck
    from dask.diagnostics import ProgressBar

    cd = ck.ClimateData(verbosity=-1)
    ds = cd.catalog("hdp").network_id(network_id).station_id(station_id).get()
    with ProgressBar():
        return ds.squeeze().compute()


# ===========================================================================
# Projections (WRF at global warming levels)
# ===========================================================================
def pull_wrf_warming_levels(variable, cfg, units):
    """Daily WRF `variable` for cfg.clip at cfg.warming_levels, converted to `units`,
    computed into memory, with cfg.exclude_simulations removed.

    Returns a DataArray with dims (sim, warming_level, time_delta, y, x).
    """
    import climakitae as ck
    from dask.diagnostics import ProgressBar

    cd = ck.ClimateData(verbosity=-1)
    ds = (
        cd.catalog("cadcat")
        .activity_id("WRF")
        .institution_id("UCLA")
        .table_id("day")
        .grid_label("d03")
        .variable_id(variable)
        .processes({
            "warming_level": {
                "warming_levels": cfg.warming_levels,
                "warming_level_window": cfg.warming_level_window,
            },
            "clip": cfg.clip,
            "convert_units": units,
        })
        .get()
    )
    # Drop excluded simulations while still lazy, so they're never downloaded.
    da = filter_simulations(ds[variable], cfg.exclude_simulations)

    # One retry: S3 reads occasionally time out on large pulls.
    for attempt in (1, 2):
        try:
            with ProgressBar():
                return da.compute()
        except Exception as e:
            if attempt == 2 or "Timeout" not in type(e).__name__:
                raise
            print(f"{type(e).__name__} pulling {variable} — retrying once")


def filter_simulations(da, exclude):
    """Drop simulations whose name contains any string in `exclude`.

    Raises if `exclude` is non-empty but matched nothing, so a typo or a renamed
    simulation can't turn the filter into a silent no-op.
    """
    if "simulation" in da.dims and "sim" not in da.dims:
        da = da.rename(simulation="sim")
    if not exclude:
        return da

    names = [str(s) for s in da["sim"].values]
    keep = [s for s in names if not any(e in s for e in exclude)]
    if len(keep) == len(names):
        raise ValueError(
            f"projection.exclude_simulations {exclude} matched none of the simulations: {names}"
        )
    print(f"Excluded {len(names) - len(keep)} simulation(s): "
          f"{sorted(set(names) - set(keep))} — {len(keep)} remain")
    return da.sel(sim=keep)


# ===========================================================================
# Event counting (shared by station obs, projections, and tract maps)
# ===========================================================================
def count_events(mask, dim, min_length=1):
    """Number of separate events along `dim`: runs of consecutive True values
    lasting at least `min_length` steps. Back-to-back qualifying days are one
    event, so a single long event is never counted more than once."""
    m = mask.fillna(False).astype(bool)
    if min_length > 1:
        # True at the last step of every full `min_length`-long qualifying stretch;
        # each run of these marks one run of the original mask that was long enough.
        m = m.rolling({dim: min_length}).sum() == min_length
    starts = m & ~m.shift({dim: 1}, fill_value=False)
    return starts.sum(dim)


# ===========================================================================
# Census tracts
# ===========================================================================
CENSUS_TRACTS_S3 = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/"
TRACTS_CACHE_FILE = "ca_tracts.gpkg"


def load_zip_codes(path):
    return gpd.read_file(path).to_crs("EPSG:4326")


def tracts_in_area(cfg, zip_codes):
    """Statewide tracts (from the CRAI cache) that intersect the zip-code area."""
    tracts = gpd.read_file(os.path.join(cfg.crai_cache_dir, TRACTS_CACHE_FILE))
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    tracts = tracts.to_crs("EPSG:4326")
    area = zip_codes.union_all()
    selected = tracts[tracts.intersects(area)][["GEOID", "geometry"]].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No census tracts intersect {cfg.zip_shapefile} — does it cover the area of interest?")
    print(f"{len(selected)} census tracts intersect the zip-code area")
    return selected


def geoid_to_zip_label(tracts, zip_codes):
    """GEOID -> 'ZIP (Place Name)' using the zip code each tract's centroid falls in."""
    centroids = tracts.copy()
    utm = tracts.estimate_utm_crs()
    centroids["geometry"] = centroids.geometry.to_crs(utm).centroid.to_crs("EPSG:4326")
    joined = gpd.sjoin(centroids, zip_codes[["ZIP5", "PO_NAME", "geometry"]], how="left", predicate="within")
    labels = (joined["ZIP5"].fillna("Unknown") + " ("
              + joined["PO_NAME"].fillna("Unknown").str.title() + ")")
    return dict(zip(joined["GEOID"], labels))


def _wrf_crs():
    import cartopy.crs as ccrs
    # WRF's native grid projection — x/y coordinates are in this CRS.
    return ccrs.LambertConformal(
        central_longitude=-70, central_latitude=38,
        false_easting=0.0, false_northing=0.0,
        standard_parallels=[30, 60], globe=None,
    )


def reproject_to_tracts(da, tracts):
    """Assign each tract the value of its nearest WRF grid point (equal-area CRS).
    Tracts that end up without data take the value of the nearest tract that has
    data. Returns a GeoDataFrame indexed by GEOID with column `da.name`.
    """
    name = da.name
    tracts_3310 = tracts.to_crs(3310).set_index("GEOID")

    df = da.to_dataframe().reset_index()
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y)).set_crs(_wrf_crs())
    points = points.to_crs(3310)

    joined = gpd.sjoin_nearest(tracts_3310, points[[name, "geometry"]], how="left")
    joined = joined.drop(columns=["index_right"]).reset_index()[["GEOID", name, "geometry"]]
    by_tract = joined.dissolve(by="GEOID", aggfunc="mean")

    missing = by_tract[by_tract[name].isna()][["geometry", name]]
    valid = by_tract[by_tract[name].notna()][["geometry", name]]
    if missing.empty:
        return valid

    missing_c = missing.assign(centroid=missing.centroid).set_geometry("centroid")
    valid_c = valid.assign(centroid=valid.centroid).set_geometry("centroid")
    filled = missing_c.sjoin_nearest(valid_c, how="left")
    filled = filled[["geometry_left", f"{name}_right"]].rename(
        columns={"geometry_left": "geometry", f"{name}_right": name})
    return pd.concat([valid, gpd.GeoDataFrame(filled, geometry="geometry", crs=valid.crs)])


def tract_metric_layers(gridded, metric, tracts, cfg):
    """Map a gridded (warming_level, y, x) metric onto tracts for every GWL,
    min-max standardize it within the area, write one CSV per GWL, and return
    {gwl: GeoDataFrame(GEOID, metric, metric_min_max_standardized, geometry)}.
    """
    layers = {}
    for wl, gwl in zip(cfg.warming_levels, cfg.gwl_order):
        da = gridded.sel(warming_level=wl).rename(metric)
        by_tract = reproject_to_tracts(da, tracts).reset_index()
        std = crai_helpers.min_max_standardize(by_tract[["GEOID", metric]].copy(), [metric])
        std[metric] = by_tract[metric].values  # min_max_standardize drops the raw column
        std.to_csv(cfg.tract_metric_csv(metric, gwl), index=False)
        layers[gwl] = tracts.merge(std, on="GEOID", how="inner")
        print(f"Wrote {cfg.tract_metric_csv(metric, gwl)} ({len(std)} tracts)")
    return layers


# ===========================================================================
# CRAI (hazard score + index) — reads what crai_base.ipynb cached
# ===========================================================================
ADAPTIVE_CAPACITY_FILE = "adaptive_capacity.csv"
CLIMATE_METRICS_FILE = "climate_metrics_standardized.csv"


def load_crai_cache(cfg):
    """(adaptive_capacity, climate_metrics) DataFrames from crai_base.ipynb's cache."""
    paths = [os.path.join(cfg.crai_cache_dir, f) for f in (ADAPTIVE_CAPACITY_FILE, CLIMATE_METRICS_FILE)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"CRAI cache incomplete — run crai_base.ipynb first. Missing: {missing}")
    adaptive, climate = (pd.read_csv(p, dtype={"GEOID": str}) for p in paths)
    for df in (adaptive, climate):
        df["GEOID"] = df["GEOID"].str.zfill(11)
    return adaptive, climate


def hazard_scores(event_layer, metric, climate_metrics, exposure_metrics, loss_metrics):
    """Climate hazard score per tract (statewide) with the event metric injected as an
    exposure metric. Method copied from era-scratch UPDATED_DE_tool_hazard_layer.ipynb:
    flip every metric (1 - x), exposure = sum, loss = sum, score = exposure * loss
    (falling back to whichever exists), outlier-clip, then min-max standardize.

    Tracts outside the event area have no event metric (NaN), so their exposure is
    the remaining exposure metrics only — same as the original notebook.
    """
    suffix = "_min_max_standardized"
    event_col = f"{metric}{suffix}"
    exposure_cols = [event_col] + [f"{m}{suffix}" for m in exposure_metrics]
    loss_cols = [f"{m}{suffix}" for m in loss_metrics]

    df = climate_metrics[["GEOID"] + exposure_cols[1:] + loss_cols].copy()
    df[event_col] = df["GEOID"].map(event_layer.set_index("GEOID")[event_col])

    flip = exposure_cols + loss_cols
    df[flip] = 1 - df[flip]

    loss = df[loss_cols].sum(axis=1, skipna=True)
    exposure = df[exposure_cols].sum(axis=1, skipna=True)
    df["hazard_score"] = np.select(
        [loss.isna() & exposure.notna(), exposure.isna() & loss.notna(),
         loss.eq(0) & exposure.notna(), exposure.eq(0) & loss.notna()],
        [exposure, loss, exposure, loss],
        default=exposure * loss,
    )

    scored = df[["GEOID", "hazard_score"]].copy()
    scored = crai_helpers.handle_outliers(scored, domain_prefix="climate", summary_stats=False)
    scored = crai_helpers.min_max_standardize(scored, ["hazard_score"])
    return scored[["GEOID", "hazard_score_min_max_standardized"]].rename(
        columns={"hazard_score_min_max_standardized": "hazard_score"})


def crai_scores(hazard, adaptive_capacity):
    """CRAI index = adaptive capacity / hazard score (zero hazard guarded)."""
    df = adaptive_capacity.merge(hazard, on="GEOID", how="left")
    df["hazard_score"] = df["hazard_score"].replace(0.0, 0.00001)
    return crai_helpers.calculate_CRAI_index(df, clim_col="hazard_score")


def deltas_vs_baseline(layers, col, gwl_order):
    """{gwl: GeoDataFrame with 'delta' = layer[col] - baseline[col]} for every
    non-baseline GWL (baseline = gwl_order[0])."""
    base_key = gwl_order[0]
    base = layers[base_key][["GEOID", col, "geometry"]]
    out = {}
    for key in gwl_order[1:]:
        merged = base.merge(layers[key][["GEOID", col]], on="GEOID", suffixes=("_base", "_cmp"))
        merged["delta"] = merged[f"{col}_cmp"] - merged[f"{col}_base"]
        out[key] = gpd.GeoDataFrame(merged, geometry="geometry", crs=base.crs)
    return out


# ===========================================================================
# Plotting
# ===========================================================================
PLOTLY_CONFIG = dict(
    responsive=True,
    displaylogo=False,
    scrollZoom=False,
    modeBarButtonsToRemove=[
        "zoom2d", "pan2d", "select2d", "lasso2d",
        "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        "zoomInGeo", "zoomOutGeo", "resetGeo", "hoverClosestGeo",
    ],
)

# ~5m geojson simplification: invisible at web-map scale, ~4x smaller embedded
# geojson (each Choropleth trace embeds its own copy).
SIMPLIFY_TOL = 0.00005


def save_figure(fig, cfg, name):
    """Write `fig` as figures/static/fig_<name>.png (needs kaleido) and figures/html/fig_<name>.html."""
    png, html = cfg.figure_paths(name)
    fig.write_image(png, scale=4.0)
    fig.write_html(html, include_plotlyjs="cdn", full_html=False, config=PLOTLY_CONFIG)


def plot_annual_bars(years, values, event_year, title, yaxis_title, unit):
    """Annual bars (warm/positive red, negative blue) with the event year outlined
    in gold and a centered 5-year rolling mean."""
    years, values = list(years), np.asarray(values)
    rolling = pd.Series(values, index=years).rolling(5, center=True).mean()
    colors = ["firebrick" if v >= 0 else "steelblue" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=values, width=0.8,
        marker=dict(color=colors, opacity=0.75,
                    line=dict(color=["gold" if y == event_year else "rgba(0,0,0,0)" for y in years],
                              width=[2.5 if y == event_year else 0 for y in years])),
        showlegend=False,
        hovertemplate=f"%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=rolling.values, mode="lines",
        line=dict(color="black", width=2), name="5-year rolling mean",
        hovertemplate=f"5-yr mean<br>%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="gray", width=0.8, dash="dash"))
    if event_year in years:
        v = values[years.index(event_year)]
        offset = 0.05 * (np.nanmax(values) - np.nanmin(values))
        fig.add_annotation(x=event_year, y=v + (offset if v >= 0 else -offset),
                           text=str(event_year), showarrow=False, font=dict(size=12, color="goldenrod"))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Year", showgrid=False, fixedrange=True),
        yaxis=dict(title=yaxis_title, showgrid=False, fixedrange=True),
        width=815, height=600,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
        hovermode="closest", dragmode=False,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    fig.show(config=PLOTLY_CONFIG)
    return fig


def plot_return_periods(rates, cfg, title):
    """Return-period bars (1 / events-per-year) across cfg.warming_levels."""
    rates = np.asarray(rates, dtype=float)
    with np.errstate(divide="ignore"):
        return_periods = 1 / rates
    labels = [f"{wl}°C" for wl in cfg.warming_levels]
    subs = cfg.sublabels if cfg.sublabels and len(cfg.sublabels) == len(labels) else labels
    ticks = [f"{s}<br>({l})" for l, s in zip(labels, subs)]
    colors = ["#4393c3", "#f4a582", "#d6604d", "#b2182b"][:len(labels)]  # cool -> warm
    finite = return_periods[np.isfinite(return_periods)]

    fig = go.Figure(go.Bar(
        x=ticks, y=np.where(np.isfinite(return_periods), return_periods, 0),
        marker=dict(color=colors, line=dict(color="white", width=0.8)),
        text=[f"1-in-{rp:.0f} yrs" if np.isfinite(rp) else "not simulated" for rp in return_periods],
        textposition="outside", textfont=dict(size=11),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Global Warming Level (°C)", showgrid=False, fixedrange=True),
        yaxis=dict(title="Return period (years)", showgrid=False, fixedrange=True,
                   range=[0, (finite.max() if finite.size else 1) * 1.25]),
        width=700, height=450,
        hovermode="closest", dragmode=False,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    fig.show(config=PLOTLY_CONFIG)
    return fig, return_periods


def _zip_boundary_trace(zip_codes, linewidth=1.5, simplify_tol=SIMPLIFY_TOL):
    """Scattergeo trace outlining zip-code boundaries, for overlay on a choropleth."""
    lons, lats = [], []
    for geom in zip_codes.geometry.simplify(simplify_tol, preserve_topology=True).boundary:
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            x, y = part.xy
            lons.extend([round(v, 5) for v in x] + [None])
            lats.extend([round(v, 5) for v in y] + [None])
    return go.Scattergeo(lon=lons, lat=lats, mode="lines",
                         line=dict(color="#404040", width=linewidth),
                         hoverinfo="skip", showlegend=False)


def _tabbed_choropleth(traces, tab_labels, zip_codes, bounds, title):
    """Shared layout for tabbed choropleths: one trace per tab + zip overlay."""
    fig = go.Figure()
    for i, tr in enumerate(traces):
        tr.visible = (i == 0)
        fig.add_trace(tr)
    fig.add_trace(_zip_boundary_trace(zip_codes))

    n = len(traces)
    buttons = [dict(label=lbl, method="update",
                    args=[{"visible": [j == i for j in range(n)] + [True]}])
               for i, lbl in enumerate(tab_labels)]

    # Tight tract bounding box + mercator — fitbounds pads too much, and the
    # default equirectangular stretches shapes east-west at CA latitudes.
    minx, miny, maxx, maxy = bounds
    lon_pad, lat_pad = (maxx - minx) * 0.03, (maxy - miny) * 0.03
    fig.update_geos(visible=False, bgcolor="white", projection_type="mercator",
                    lonaxis_range=[minx - lon_pad, maxx + lon_pad],
                    lataxis_range=[miny - lat_pad, maxy + lat_pad])
    fig.update_layout(
        title=dict(text=title, x=0.043, xanchor="left", y=0.99, yanchor="top",
                   font=dict(size=20), pad=dict(t=20)),
        updatemenus=[dict(type="buttons", direction="right", buttons=buttons,
                          x=0.02, xanchor="left", y=1, yanchor="bottom", showactive=True,
                          bgcolor="#f8f9fa", bordercolor="#999", font=dict(size=14),
                          pad=dict(t=8, b=8, l=10, r=10))],
        width=815, height=810, paper_bgcolor="white",
        margin=dict(l=10, r=80, t=100, b=10), hovermode="closest",
    )
    return fig


def _simplified_geojson(gdf, simplify_tol=SIMPLIFY_TOL):
    g = gdf[["GEOID", "geometry"]].copy()
    g["geometry"] = g.geometry.simplify(simplify_tol, preserve_topology=True)
    return json.loads(g.to_json())


def tabbed_choropleth(layers, value_col, title, name, cfg, tracts, zip_labels, zip_codes,
                      hover_label=None, colorbar_title="# of Days"):
    """One tab per GWL showing `value_col` per tract; saved via save_figure(name)."""
    vmax = max(layers[g][value_col].max() for g in cfg.gwl_order)
    geojson = _simplified_geojson(tracts)
    traces = [go.Choropleth(
        geojson=geojson, featureidkey="properties.GEOID",
        locations=layers[g]["GEOID"], z=layers[g][value_col],
        zmin=0, zmax=vmax, colorscale="YlOrRd",
        marker=dict(line=dict(color="#cccccc", width=0.4)),
        colorbar=dict(title=dict(text=colorbar_title, font=dict(size=16)), tickfont=dict(size=16),
                      thickness=20, len=0.7, x=1.0, xanchor="left"),
        name=cfg.gwl_labels[g],
        customdata=layers[g]["GEOID"].map(zip_labels),
        hovertemplate="%{customdata}<br>" + (hover_label or value_col) + ": %{z:.2f}<extra></extra>",
    ) for g in cfg.gwl_order]
    fig = _tabbed_choropleth(traces, [cfg.gwl_labels[g] for g in cfg.gwl_order],
                             zip_codes, tracts.total_bounds, title)
    save_figure(fig, cfg, name)
    fig.show(config=PLOTLY_CONFIG)
    return fig


def tabbed_delta_choropleth(deltas, title, name, cfg, zip_labels, zip_codes):
    """One tab per non-baseline GWL showing the change vs. the baseline GWL."""
    keys = list(deltas)
    vmax = max(np.nanmax(np.abs(deltas[k]["delta"])) for k in keys) or 1
    geojson = _simplified_geojson(deltas[keys[0]])
    traces = [go.Choropleth(
        geojson=geojson, featureidkey="properties.GEOID",
        locations=deltas[k]["GEOID"], z=deltas[k]["delta"],
        zmin=-vmax, zmax=vmax, colorscale="RdBu", reversescale=True,
        marker=dict(line=dict(color="#cccccc", width=0.4)),
        colorbar=dict(tickmode="array", tickvals=[-0.75 * vmax, 0.75 * vmax],
                      ticktext=["Lower <br>Risk", "Higher <br>Risk"], tickfont=dict(size=14),
                      thickness=20, len=0.7, x=1.0, xanchor="left"),
        name=cfg.gwl_labels[k],
        customdata=deltas[k]["GEOID"].map(zip_labels),
        hovertemplate="%{customdata}<br>Δ: %{z:.2f}<extra></extra>",
    ) for k in keys]
    fig = _tabbed_choropleth(traces, [cfg.gwl_labels[k] for k in keys],
                             zip_codes, deltas[keys[0]].total_bounds, title)
    save_figure(fig, cfg, name)
    fig.show(config=PLOTLY_CONFIG)
    return fig
