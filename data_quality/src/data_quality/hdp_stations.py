"""
Read HDP weather-station observations from the Cal-Adapt Analytics Engine and
write them to a tidy zarr store with lat, lon and one requested variable.

VARIABLE-AGNOSTIC, WITH WIND AS THE DEFAULT. The station-list CSV counts
observations per variable in separate `*_nobs` columns, so a fetch must select
on the column matching what it is fetching -- see obs_column(). Filtering a
humidity fetch on `sfcwind_nobs` returns stations that measure wind, produces a
store that looks entirely normal, and is undetectable downstream.

The HDP catalog is station-based, not gridded: each asset is one station's zarr
store, and there is NO variable_id column to filter on. Consequences:

  * `network_id` is REQUIRED and must be a single value. Networks differ in
    instrumentation, record length and reporting convention, so mixing them in
    one query is blocked by the validator.
  * You cannot ask for "all stations with variable X" -- open the stores and
    inspect `data_vars`.
  * Several processors are invalid for HDP: localize, clip, convert_units,
    bias_adjust_model_to_station, filter_unadjusted_models, metric_calc,
    warming_level. `time_slice` DOES work and is the main cost control.

Retrieval cost scales linearly with station count (one store per station), so
this module fetches in batches and appends each batch to the zarr store rather
than holding everything in memory.

Requires: climakitae >= 1.5, xarray, zarr, numpy, pandas
"""

from __future__ import annotations

import warnings
from dask.diagnostics import ProgressBar

import numpy as np
import pandas as pd
import xarray as xr

# Wind-speed variable name in HDP. Station networks use CF-style names, unlike
# the WRF-native (wspd10mean) or LOCA2 (wspeed) gridded products.
WIND_VAR = "sfcWind"

# Candidate column names for station coordinates in the catalog dataframe.
LAT_COLS = ("lat", "latitude", "station_lat", "LAT", "station_latitude")
LON_COLS = ("lon", "longitude", "station_lon", "LON", "station_longitude")
ELEV_COLS = ("elevation", "elev", "station_elevation", "altitude", "alt", "height")

# Station coordinates are often per-store ATTRIBUTES rather than coords or
# catalog columns, so every plausible attr name is checked too.
LAT_ATTRS = ("latitude", "lat", "LATITUDE", "station_latitude", "y")
LON_ATTRS = ("longitude", "lon", "LONGITUDE", "station_longitude", "x")
ELEV_ATTRS = ("elevation", "elev", "ELEVATION", "station_elevation",
              "altitude", "height", "station_height")

# Per-station metadata that HDP stores as data variables. Promoted to
# coordinates on retrieval so they ride along with the wind data.
COORD_VARS = ("lat", "lon", "elevation")

# Networks skipped by default.
#   CW3E -- sub-hourly reporting. It does not fit the hourly canonical axis:
#           reindexing would discard most of its samples, and matching it would
#           force every other network onto a needlessly fine axis.
#EXCLUDE_NETWORKS = ("CW3E",)
EXCLUDE_NETWORKS = ()

# HDP is QA/QC'd: every station it exposes reports wind at 10 m, whatever the
# parent network's native convention (CIMIS is natively 2 m, RAWS 6.1 m). Height
# is therefore NOT a confound here, and stations may be pooled across networks
# on that account -- unlike raw network data, where the 2 m to 10 m spread would
# manufacture a ~25-30% offset comparable to the model biases under study.
#
# Height is common; SITING is not. Networks still differ sharply in where their
# masts stand relative to the surrounding grid cell (see build_multi_network's
# `network` coordinate, and section 8.1 of the notebook).
ANEMOMETER_HEIGHT_M = 10.0


# ----------------------------------------------------------------------------
# 1. Station discovery and sampling
# ----------------------------------------------------------------------------


def catalog_df() -> pd.DataFrame:
    """The raw HDP intake-esm dataframe (metadata only, no data reads)."""
    from climakitae.new_core.data_access.data_access import DataCatalog

    df = DataCatalog().hdp.df
    print(f"HDP catalog: {len(df)} rows, columns: {list(df.columns)}")
    return df


def list_stations(network_id: str, df: pd.DataFrame | None = None):
    """All station_ids in one network, plus that network's catalog rows."""
    if df is None:
        df = catalog_df()
    sub = df[df.network_id == network_id]
    if sub.empty:
        raise ValueError(
            f"No rows for network_id={network_id!r}. "
            f"Available: {sorted(df.network_id.unique())}"
        )
    stations = sorted(sub.station_id.unique())
    print(f"{len(stations)} stations in {network_id}")
    return stations, sub


def sample_stations(stations, n: int = 100, seed: int = 42) -> list[str]:
    """Reproducible random subsample. Seeded, or the 'test' run is not a test."""
    rng = np.random.default_rng(seed)
    n = min(n, len(stations))
    out = sorted(rng.choice(stations, size=n, replace=False).tolist())
    print(f"sampled {n} of {len(stations)} stations (seed={seed})")
    return out


def stratified_sample(sub: pd.DataFrame, n: int = 100, seed: int = 42,
                      bands: int = 10) -> list[str]:
    """Latitude-stratified sample -- better spatial spread than uniform random.

    Falls back to uniform if the catalog has no usable latitude column.
    """
    lat_col = next((c for c in LAT_COLS if c in sub.columns), None)
    if lat_col is None:
        warnings.warn("no latitude column in catalog; falling back to uniform sample")
        return sample_stations(sorted(sub.station_id.unique()), n, seed)

    uniq = sub.drop_duplicates("station_id").dropna(subset=[lat_col]).copy()
    uniq["_band"] = pd.qcut(uniq[lat_col], bands, labels=False, duplicates="drop")
    per = max(1, n // uniq["_band"].nunique())
    picked = (uniq.groupby("_band", group_keys=False)
                  .apply(lambda g: g.sample(min(per, len(g)), random_state=seed)))
    out = sorted(picked.station_id.tolist())
    print(f"stratified sample: {len(out)} stations across {uniq['_band'].nunique()} bands")
    return out


# ----------------------------------------------------------------------------
# 2. Retrieval
# ----------------------------------------------------------------------------

def fetch_batch(network_id: str, station_ids, time_slice=None, verbosity=-1,
                bisect: bool = True, _depth: int = 0):
    """Retrieve one batch. Returns a Dataset or None. Never raises.

    BISECTS ON FAILURE. climakitae validates a query all-or-nothing: one station
    it cannot serve empties the whole request. At batch_size=48 that costs 48
    stations for one bad id, and a network whose every batch contains one comes
    back empty entirely -- which is how ASOSAWOS delivered 38 of ~500.

    So an empty result on a multi-station batch is retried as two halves. A bad
    station is isolated in log2(n) extra queries and costs only itself; a batch
    that was fine simply never enters this path.
    """
    from climakitae import ClimateData

    ids = list(station_ids)
    ds = None
    try:
        q = (ClimateData(verbosity=verbosity)
             .catalog("hdp").network_id(network_id).station_id(ids))
        if time_slice is not None:
            q = q.processes({"time_slice": time_slice})
        ds = q.get()
    except Exception as e:
        if not (bisect and len(ids) > 1):
            warnings.warn(f"station failed ({type(e).__name__}): {ids}")
            return None
        ds = None

    if ds is not None and getattr(ds, "sizes", {}).get("station_id", 1) > 0:
        return ds.to_dataset() if isinstance(ds, xr.DataArray) else ds

    if not (bisect and len(ids) > 1):
        warnings.warn(f"no data for station: {ids}")
        return None

    mid = len(ids) // 2
    print(f"{'  ' * _depth}    batch of {len(ids)} empty; splitting")
    parts = [fetch_batch(network_id, half, time_slice, verbosity,
                         bisect, _depth + 1)
             for half in (ids[:mid], ids[mid:])]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return xr.concat(parts, dim="station_id", coords="minimal",
                     compat="override")


def canonical_time(time_slice, freq: str = "h") -> pd.DatetimeIndex:
    """Fixed time axis every batch is reindexed onto.

    Stations have different record periods, so batches would otherwise have
    different time axes and could not be appended along station_id. Pinning the
    axis up front makes the appends valid and the store's shape predictable.
    """
    t0, t1 = time_slice
    return pd.date_range(pd.Timestamp(t0), pd.Timestamp(t1), freq=freq)


def tidy_batch(ds: xr.Dataset, var: str, times: pd.DatetimeIndex | None,
               coords_df: pd.DataFrame | None = None) -> xr.Dataset | None:
    """Keep one variable + lat/lon, reindex onto the canonical time axis."""
    # Resolve the alias. A network that measures RH natively stores `hurs`;
    # one that measures dewpoint stores HDP's `hurs_derived`. Both are the same
    # quantity, so the first present is renamed to the canonical name and
    # nothing downstream needs to know which route a station took.
    found = next((n for n in VAR_ALIASES.get(var, (var,))
                  if n in ds.data_vars), None)
    if found is None:
        warnings.warn(
            f"'{var}' not in this batch (tried "
            f"{list(VAR_ALIASES.get(var, (var,)))}). "
            f"Present: {list(ds.data_vars)}. Skipping.")
        return None
    if found != var:
        ds = ds.rename({found: var})

    # Keep the station metadata variables alongside the wind variable. HDP
    # stores carry lat/lon/elevation as DATA VARIABLES, so subsetting to the
    # wind variable alone silently drops them.
    extra = [c for c in COORD_VARS if c in ds.data_vars or c in ds.coords]
    out = ds[[var] + [c for c in extra if c in ds.data_vars]]
    out[var] = out[var].astype("float32")

    # Promote them to coordinates so they survive the reindex and the append,
    # and so they do not get treated as another field to analyse.
    out = promote_station_coords(out, extra)

    # Strip encoding inherited from the source stores. HDP assets are zarr v2,
    # so their variables carry a numcodecs.Blosc compressor; passing that to a
    # zarr v3 writer raises "Expected a BytesBytesCodec". Clearing it lets the
    # writer choose codecs appropriate to the output format.
    out = clear_encoding(out)

    # Coordinate provenance, in order of reliability:
    #   1. already on the dataset as coords
    #   2. the station store's own attrs (works when batch_size == 1)
    #   3. the catalog dataframe, if it happens to carry lat/lon columns
    have_lat = "lat" in out.coords and bool(out.lat.notnull().any())
    if not have_lat:
        out.attrs = dict(ds.attrs)          # carry attrs through for step 2
        out = attach_coords_from_attrs(out)
        have_lat = "lat" in out.coords
    if not have_lat and coords_df is not None:
        out = attach_coords_from_catalog(out, coords_df)

    if times is not None:
        out = out.reindex(time=times)
    return out


def promote_station_coords(ds: xr.Dataset, names) -> xr.Dataset:
    """Turn lat/lon/elevation data variables into station_id coordinates.

    Handles the awkward shapes these arrive in: a time dimension they should
    not have (take the first valid value -- station position is static), or no
    station_id dimension at all when a single store is fetched.
    """
    for n in names:
        if n not in ds.data_vars:
            continue
        da = ds[n]
        if "time" in da.dims:
            if da.sizes["time"] == 0:
                # Station has no data in the requested window, so there is no
                # value to collapse. Reducing an empty axis raises
                # "zero-size array to reduction operation fmax".
                da = xr.DataArray(
                    np.full(ds.sizes.get("station_id", 1), np.nan),
                    dims="station_id", coords={"station_id": ds.station_id},
                )
            else:
                # Station position is static; any non-null value will do.
                # (max skips NaN and needs no optional accelerator, unlike ffill.)
                da = da.max("time", skipna=True)
        if "station_id" not in da.dims:
            da = da.expand_dims(station_id=ds.station_id)
        ds = ds.drop_vars(n).assign_coords({n: da})
    return ds


def coords_from_attrs(ds: xr.Dataset) -> dict:
    """Pull lat/lon/elevation out of a single station's attributes.

    HDP assets are per-station stores, so their global attrs describe that
    station. This only makes sense for a ONE-station dataset -- after a
    multi-station concat the attrs belong to whichever store came first.
    """
    def find(names):
        for n in names:
            if n in ds.attrs:
                try:
                    return float(ds.attrs[n])
                except (TypeError, ValueError):
                    continue
        return np.nan

    return {"lat": find(LAT_ATTRS), "lon": find(LON_ATTRS),
            "elevation": find(ELEV_ATTRS)}


def attach_coords_from_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Attach lat/lon/elevation from attrs onto the station_id dimension."""
    if ds.sizes.get("station_id", 1) != 1:
        warnings.warn("attrs describe one station; skipping (batch_size > 1)")
        return ds
    c = coords_from_attrs(ds)
    if np.isnan(c["lat"]) and np.isnan(c["lon"]):
        return ds
    return ds.assign_coords(
        lat=("station_id", [c["lat"]]),
        lon=("station_id", [c["lon"]]),
        elevation=("station_id", [c["elevation"]]),
    )


def clear_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Drop per-variable encoding carried over from the source store."""
    for v in list(ds.variables):
        ds[v].encoding = {}
    return ds


def attach_coords_from_catalog(ds: xr.Dataset, coords_df: pd.DataFrame) -> xr.Dataset:
    """Attach lat/lon per station from the catalog dataframe."""
    lat_col = next((c for c in LAT_COLS if c in coords_df.columns), None)
    lon_col = next((c for c in LON_COLS if c in coords_df.columns), None)
    if lat_col is None or lon_col is None:
        warnings.warn(
            f"no lat/lon columns in catalog (have {list(coords_df.columns)}); "
            "station coordinates will be missing."
        )
        return ds

    look = coords_df.drop_duplicates("station_id").set_index("station_id")
    ids = ds.station_id.values
    new = {
        "lat": ("station_id", look.reindex(ids)[lat_col].to_numpy(dtype="float64")),
        "lon": ("station_id", look.reindex(ids)[lon_col].to_numpy(dtype="float64")),
    }
    elev_col = next((c for c in ELEV_COLS if c in coords_df.columns), None)
    if elev_col is not None:
        new["elevation"] = ("station_id",
                            look.reindex(ids)[elev_col].to_numpy(dtype="float64"))
    return ds.assign_coords(**new)


# ----------------------------------------------------------------------------
# 3. Batched fetch -> zarr
# ----------------------------------------------------------------------------


def build_zarr(
    network_id: str,
    station_ids,
    path: str,
    time_slice=("1/1/2000", "12/31/2000"),
    var: str = WIND_VAR,
    freq: str = "h",
    batch_size: int = 20,
    time_chunk: int = 8760,
    station_chunk: int = 20,
    coords_df: pd.DataFrame | None = None,
    zarr_format: int = 2,
    skip_empty: bool = True,
    append: bool = False,
) -> str:
    """Fetch stations in batches and append each to a zarr store.

    Appending along station_id keeps peak memory at one batch rather than the
    whole network. Every batch is reindexed onto the same time axis first,
    which is what makes the append well defined.
    """
    times = canonical_time(time_slice, freq) if time_slice else None
    if times is not None:
        print(f"canonical time axis: {len(times)} steps at '{freq}' "
              f"({times[0]} -> {times[-1]})")

    batches = [station_ids[i:i + batch_size]
               for i in range(0, len(station_ids), batch_size)]
    print(f"{len(station_ids)} stations in {len(batches)} batches of <= {batch_size}")

    written, empty = 0, []
    for i, batch in enumerate(batches, 1):
        print(f"\n[{i}/{len(batches)}] fetching {len(batch)} stations...")
        ds = fetch_batch(network_id, batch, time_slice)
        if ds is None:
            empty.extend(batch)
            continue

        # A station whose record does not reach into the requested window comes
        # back with time=0. Writing it would add a column of pure NaN, so by
        # default it is skipped and reported at the end.
        if skip_empty and ds.sizes.get("time", 0) == 0:
            print(f"    no data in window -- skipping {list(batch)}")
            empty.extend(batch)
            continue
        ds = tidy_batch(ds, var, times, coords_df)
        if ds is None:
            continue

        # Tag every station with its network so a multi-network store stays
        # separable -- networks differ in instrumentation and reporting
        # convention, and should usually be analysed separately.
        #
        # Both string coords are cast to object (variable-length) dtype. Fixed
        # -width numpy strings take their width from the longest value in the
        # BATCH, so 'HPWREN' gives <U6 while 'ASOSAWOS' gives <U8, and zarr
        # refuses to append mismatched dtypes. Station IDs have the same problem
        # across networks.
        ds = ds.assign_coords(
            network=("station_id",
                     np.array([network_id] * ds.sizes["station_id"], dtype=object)),
            station_id=ds.station_id.astype(object),
        )

        ds = ds.chunk({"time": time_chunk, "station_id": 1})

        # zarr v2 by default: v3 does not support consolidated metadata and is
        # pickier about codecs. Pass zarr_format=3 if you need it.
        kw = {"consolidated": True, "zarr_format": zarr_format}
        if written == 0 and not append:
            ds.to_zarr(path, mode="w", encoding={var: {"dtype": "float32"}}, **kw)
        else:
            ds.to_zarr(path, mode="a", append_dim="station_id", **kw)

        written += ds.sizes["station_id"]
        print(f"    wrote {ds.sizes['station_id']} stations "
              f"(total {written}), valid {float(ds[var].notnull().mean()):.1%}")

    if written == 0:
        raise RuntimeError(
            "nothing written -- every station failed, lacked the variable, or "
            "had no data in the requested window. Check the native record range "
            "with fetch_batch(network, [station], None)."
        )
    print(f"\nwrote {written} stations to {path}")
    if empty:
        print(f"{len(empty)} stations skipped (no data in window): {empty[:5]}"
              + (" ..." if len(empty) > 5 else ""))
    return path


def open_store(path: str) -> xr.Dataset:
    """Reopen the store and report coverage per station."""
    ds = xr.open_zarr(path, consolidated=True)  # add zarr_format= if needed
    var = list(ds.data_vars)[0]
    # One value per station, so computing eagerly is cheap -- and necessary,
    # since dask cannot do a full-array nanmedian.
    valid = ds[var].notnull().mean("time").compute()
    print(f"{ds.sizes['station_id']} stations, {ds.sizes['time']} time steps")
    print(f"median per-station coverage: {float(np.nanmedian(valid.values)):.1%}")
    print(f"stations with <10% coverage: {int((valid < 0.1).sum())}")
    return ds


# ----------------------------------------------------------------------------
# 4. Repairing an existing store
# ----------------------------------------------------------------------------


def collect_station_coords(network_id: str, station_ids,
                           time_slice=("1/1/2020", "1/2/2020")) -> pd.DataFrame:
    """Fetch each station's lat/lon/elevation from its store attributes.

    Use this to add coordinates to a store built before attrs were captured,
    without re-downloading the data. The tiny time_slice keeps each call cheap:
    only the metadata is wanted.
    """
    rows = []
    for i, sid in enumerate(station_ids, 1):
        ds = fetch_batch(network_id, [sid], time_slice)
        if ds is None:
            rows.append({"station_id": sid, "lat": np.nan,
                         "lon": np.nan, "elevation": np.nan})
            continue
        rows.append({"station_id": sid, **coords_from_attrs(ds)})
        if i % 20 == 0:
            print(f"  {i}/{len(station_ids)}")
    out = pd.DataFrame(rows).set_index("station_id")
    print(f"got coordinates for {int(out.lat.notnull().sum())}/{len(out)} stations")
    return out


def add_coords_to_store(path: str, coords: pd.DataFrame, out_path: str) -> str:
    """Write a copy of the store with lat/lon/elevation attached."""
    ds = xr.open_zarr(path, consolidated=True)
    ids = ds.station_id.values
    look = coords.reindex(ids)
    ds = ds.assign_coords(
        lat=("station_id", look["lat"].to_numpy(dtype="float64")),
        lon=("station_id", look["lon"].to_numpy(dtype="float64")),
        elevation=("station_id", look["elevation"].to_numpy(dtype="float64")),
    )
    ds = clear_encoding(ds)
    ds.to_zarr(out_path, mode="w", consolidated=True, zarr_format=2)
    print(f"wrote {out_path} with coordinates")
    return out_path


# ----------------------------------------------------------------------------
# 5. Multiple networks
# ----------------------------------------------------------------------------


def network_summary(df: pd.DataFrame | None = None,
                    exclude=EXCLUDE_NETWORKS) -> pd.DataFrame:
    """Station count per network -- the first thing to look at when scaling up."""
    if df is None:
        df = catalog_df()
    out = (df.groupby("network_id")["station_id"].nunique()
             .sort_values(ascending=False).to_frame("n_stations"))
    out["excluded"] = out.index.isin(exclude or ())
    print(out.to_string())
    if exclude:
        print(f"\nexcluded by default: {list(exclude)}")
    return out


def probe_network_range(network_id: str, df: pd.DataFrame | None = None,
                        n_probe: int = 5, seed: int = 0,
                        target: tuple | None = None) -> pd.DataFrame:
    """Native record extent for a few stations, WITHOUT a time filter.

    Networks differ enormously in period, and so do stations WITHIN a network:
    ASOSAWOS probes have shown starts anywhere from 1980 to 2006. Probing before
    a big fetch stops you from requesting a window the data does not cover,
    which returns time=0 and silently skips every station.

    Pass `target=("1/1/1980", "12/31/2010")` to see what fraction of that window
    each probed station actually covers -- more useful than a single common
    window, which often does not exist.

    n_probe of 5 is a small sample of a large network. Raise it to 15-20 before
    committing to an expensive fetch.
    """
    stations, sub = list_stations(network_id, df)
    rng = np.random.default_rng(seed)
    picks = rng.choice(stations, size=min(n_probe, len(stations)), replace=False)

    rows = []
    for sid in picks:
        ds = fetch_batch(network_id, [sid], None)
        if ds is None or ds.sizes.get("time", 0) == 0:
            rows.append({"station_id": sid, "start": pd.NaT, "end": pd.NaT, "n": 0})
            continue
        rows.append({"station_id": sid,
                     "start": pd.Timestamp(ds.time.values[0]),
                     "end": pd.Timestamp(ds.time.values[-1]),
                     "n": ds.sizes["time"]})
    out = pd.DataFrame(rows)
    ok = out.dropna(subset=["start"])
    if len(ok):
        lo, hi = ok.start.max(), ok.end.min()
        print(f"{network_id}: starts {ok.start.min().date()} to {ok.start.max().date()}, "
              f"ends {ok.end.min().date()} to {ok.end.max().date()}")
        if lo < hi:
            print(f"   common window: {lo.date()} -> {hi.date()} "
                  f"({(hi - lo).days / 365.25:.1f} yr)")
        else:
            # start-of-latest is after end-of-earliest: the probed records do
            # not all overlap. Not a failure -- just means a shared window does
            # not exist, and stations should be filtered on coverage instead.
            print("   NO common window across probed stations "
                  "(records do not all overlap)")
        if target is not None:
            t0, t1 = pd.Timestamp(target[0]), pd.Timestamp(target[1])
            frac = [(min(e, t1) - max(s, t0)).days / max((t1 - t0).days, 1)
                    for s, e in zip(ok.start, ok.end)]
            frac = np.clip(frac, 0, 1)
            print(f"   coverage of {t0.date()}..{t1.date()}: "
                  f"median {np.median(frac):.0%}, "
                  f"{int((frac > 0.5).sum())}/{len(frac)} stations >50%")
    else:
        print(f"{network_id}: no usable records in probe")
    return out


def build_multi_network(
    networks,
    path: str,
    time_slice=("1/1/2010", "12/31/2020"),
    n_per_network: int | None = 100,
    seed: int = 42,
    var: str = WIND_VAR,
    freq: str = "h",
    batch_size: int = 1,
    df: pd.DataFrame | None = None,
    exclude=EXCLUDE_NETWORKS,
    **kwargs,
) -> str:
    """Fetch several networks into ONE store, appending along station_id.

    The validator allows one network per query, so this loops. Because every
    batch is reindexed onto the same canonical time axis, the appends are valid
    across networks as well as within them. Station IDs are network-prefixed,
    so they stay unique, and each carries a `network` coordinate.

    n_per_network=None takes every station in each network.
    """
    if df is None:
        df = catalog_df()

    dropped = [n for n in networks if n in (exclude or ())]
    if dropped:
        print(f"excluded: {dropped}")
        networks = [n for n in networks if n not in dropped]

    results = {}
    first = True
    for net in networks:
        print(f"\n{'=' * 60}\n{net}\n{'=' * 60}")
        try:
            stations, sub = list_stations(net, df)
        except ValueError as e:
            print(f"  skipping: {e}")
            continue

        ids = (stations if n_per_network is None
               else sample_stations(stations, n=n_per_network, seed=seed))
        try:
            build_zarr(net, ids, path, time_slice=time_slice, var=var, freq=freq,
                       batch_size=batch_size, coords_df=sub,
                       append=not first, **kwargs)
            results[net] = len(ids)
            first = False
        except RuntimeError as e:
            print(f"  {net} produced nothing: {e}")

    if first:
        raise RuntimeError("no network produced any data")

    print(f"\n{'=' * 60}")
    ds = xr.open_zarr(path, consolidated=True)
    counts = pd.Series(ds.network.values).value_counts()
    print(f"store: {ds.sizes['station_id']} stations, {ds.sizes['time']} time steps")
    print(counts.to_string())
    return path


def per_network_coverage(ds: xr.Dataset, var: str | None = None) -> pd.DataFrame:
    """Coverage and per-station statistics broken down by network.

    `var` defaults to the store's single data variable rather than to sfcWind.
    A store holding two variables must name one: guessing there would silently
    report humidity coverage under a wind heading.
    """
    if var is None:
        dv = list(ds.data_vars)
        if len(dv) != 1:
            raise ValueError(f"store holds {dv}; pass var= to choose one")
        var = dv[0]
    cov = ds[var].notnull().mean("time").compute()
    mean = ds[var].mean("time").compute()
    net = pd.Series(ds.network.values, name="network")

    out = (pd.DataFrame({"network": net,
                         "coverage": cov.values,
                         "mean_value": mean.values})
             .groupby("network")
             .agg(n_stations=("coverage", "size"),
                  median_coverage=("coverage", "median"),
                  median_value=("mean_value", "median")))
    print(out.to_string())
    return out


# ----------------------------------------------------------------------------
# 6. Screening stations by record coverage
# ----------------------------------------------------------------------------


def domain_polygon(ds: xr.Dataset | xr.DataArray):
    """Perimeter of a curvilinear grid, as an (N, 2) lon/lat array.

    Pass the WRF elevation Dataset or any WRF field. Use with `region=` in
    screen_stations: a lat/lon bounding box is NOT equivalent, because the WRF
    domain is rotated and its bounding box encloses large areas the model never
    covers -- on the first station sample the box over-counted by about a third.
    """
    lat, lon = get_latlon_any(ds)
    la, lo = lat.values, lon.values
    if la.ndim == 1:                       # rectilinear: build a rectangle
        return np.array([[lo.min(), la.min()], [lo.max(), la.min()],
                         [lo.max(), la.max()], [lo.min(), la.max()],
                         [lo.min(), la.min()]])
    plon = np.concatenate([lo[0, :], lo[:, -1], lo[-1, ::-1], lo[::-1, 0]])
    plat = np.concatenate([la[0, :], la[:, -1], la[-1, ::-1], la[::-1, 0]])
    return np.column_stack([plon, plat])


def get_latlon_any(ds):
    """lat/lon coords from a Dataset or DataArray, whatever they are named."""
    src = ds.coords
    for a, b in (("lat", "lon"), ("latitude", "longitude"), ("XLAT", "XLONG")):
        if a in src and b in src:
            return src[a], src[b]
    raise KeyError(f"no lat/lon found in {list(src)}")


def _in_region(lat, lon, region) -> bool:
    """Point-in-region test. `region` is an (N,2) lon/lat polygon or a
    (lon_min, lon_max, lat_min, lat_max) bounding box."""
    if lat is None or lon is None or not np.isfinite(lat) or not np.isfinite(lon):
        return False
    region = np.asarray(region, dtype=float)
    if region.ndim == 1 and region.size == 4:
        lo0, lo1, la0, la1 = region
        return bool(lo0 <= lon <= lo1 and la0 <= lat <= la1)
    from matplotlib.path import Path as _Path
    return bool(_Path(region).contains_point((lon, lat)))


def _station_coords(ds) -> tuple:
    """First valid lat/lon from a single-station dataset."""
    out = []
    for n in ("lat", "lon"):
        if n not in ds.variables:
            out.append(np.nan)
            continue
        da = ds[n]
        if "time" in da.dims:
            da = da.isel(time=0)
        try:
            out.append(float(np.ravel(da.values)[0]))
        except (IndexError, ValueError, TypeError):
            out.append(np.nan)
    return tuple(out)


def screen_stations(
    network_id: str,
    station_ids=None,
    target=("1/1/1980", "12/31/2010"),
    df: pd.DataFrame | None = None,
    var: str = WIND_VAR,
    limit: int | None = None,
    verbose_every: int = 25,
    region=None,
) -> pd.DataFrame:
    """Record extent and target-window coverage for many stations.

    Opens each store WITHOUT a time filter and reads only the time coordinate,
    which is metadata rather than data -- cheap per station, though the round
    trips add up. Use this instead of random sampling when the window matters:
    a network whose MEDIAN coverage is 16% can still hold a third of its
    stations above 50%, and those are the ones worth fetching.

    Pass `region=domain_polygon(elev_ds)` to also record whether each station
    falls inside the model domain. The store is already open at this point, so
    lat/lon come for free -- and since most of the catalog lies outside
    California, this is usually the filter that shrinks the fetch list most.

    Returns a frame indexed by station_id with start, end, n_steps, has_var,
    coverage (fraction of `target` the record spans), lat, lon and in_region.
    """
    if station_ids is None:
        station_ids, _ = list_stations(network_id, df)
    if limit is not None:
        station_ids = station_ids[:limit]

    t0, t1 = pd.Timestamp(target[0]), pd.Timestamp(target[1])
    span = max((t1 - t0).days, 1)

    rows = []
    for i, sid in enumerate(station_ids, 1):
        ds = fetch_batch(network_id, [sid], None)
        if ds is None or ds.sizes.get("time", 0) == 0:
            rows.append({"station_id": sid, "start": pd.NaT, "end": pd.NaT,
                         "n_steps": 0, "has_var": False, "coverage": 0.0,
                         "lat": np.nan, "lon": np.nan,
                         "in_region": region is None})
        else:
            s0 = pd.Timestamp(ds.time.values[0])
            e0 = pd.Timestamp(ds.time.values[-1])
            overlap = (min(e0, t1) - max(s0, t0)).days
            slat, slon = _station_coords(ds)
            rows.append({
                "station_id": sid, "start": s0, "end": e0,
                "n_steps": ds.sizes["time"],
                "has_var": var in ds.data_vars,
                "coverage": float(np.clip(overlap / span, 0, 1)),
                "lat": slat, "lon": slon,
                "in_region": True if region is None
                             else _in_region(slat, slon, region),
            })
        if verbose_every and i % verbose_every == 0:
            print(f"  screened {i}/{len(station_ids)}")

    out = pd.DataFrame(rows).set_index("station_id")
    ok = out[out.has_var]
    print(f"{network_id}: {len(out)} screened, {int(out.has_var.sum())} carry '{var}'")
    if region is not None:
        print(f"   in region: {int(out.in_region.sum())} stations")
        ok = ok[ok.in_region]
    for thr in (0.10, 0.25, 0.5, 0.75):
        print(f"   coverage > {thr:.0%}{' (in region)' if region is not None else ''}"
              f": {int((ok.coverage > thr).sum())} stations")
    return out


def select_by_coverage(screen: pd.DataFrame, min_coverage: float = 0.5,
                       require_var: bool = True, n: int | None = None,
                       seed: int = 42, require_region: bool = True) -> list[str]:
    """Station IDs passing coverage, variable and region filters."""
    sel = screen[screen.coverage >= min_coverage]
    if require_var:
        sel = sel[sel.has_var]
    if require_region and "in_region" in sel.columns:
        sel = sel[sel.in_region]
    ids = sorted(sel.index.tolist())
    if n is not None and len(ids) > n:
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(ids, size=n, replace=False).tolist())
    print(f"selected {len(ids)} stations with coverage >= {min_coverage:.0%}")
    return ids


def wind_variable_audit(networks, df: pd.DataFrame | None = None,
                        var: str = WIND_VAR,
                        exclude=EXCLUDE_NETWORKS) -> pd.DataFrame:
    """Does each network report wind at all, and what else does it carry?

    The cheapest possible filter: one station per network. Hydrology and
    snow-telemetry networks often carry no wind variable, so screening them
    station by station would be wasted effort.
    """
    if df is None:
        df = catalog_df()
    rows = []
    for net in networks:
        if net in (exclude or ()):
            print(f"{net}: excluded")
            continue
        try:
            stations, _ = list_stations(net, df)
        except ValueError:
            rows.append({"network": net, "has_wind": False, "vars": "no rows"})
            continue
        ds = fetch_batch(net, stations[:1], None)
        if ds is None:
            rows.append({"network": net, "has_wind": False, "vars": "no data"})
            continue
        dv = list(ds.data_vars)
        rows.append({"network": net, "n_stations": len(stations),
                     "has_wind": var in dv, "vars": ", ".join(dv[:8])})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def anemometer_audit(networks, df: pd.DataFrame | None = None,
                     exclude=EXCLUDE_NETWORKS) -> pd.DataFrame:
    """Confirm the 10 m anemometer height HDP standardises to.

    HDP is QA/QC'd to a common 10 m height, so this is a verification step
    rather than a correction step: it should report 10 m (or nothing, if the
    attrs are silent) for every network. A network reporting something else is
    worth investigating before including it.
    """
    if df is None:
        df = catalog_df()
    keys = ("height", "anem", "sensor", "level", "agl", "z")

    rows = []
    for net in networks:
        if net in (exclude or ()):
            continue
        try:
            stations, _ = list_stations(net, df)
        except ValueError:
            continue
        ds = fetch_batch(net, stations[:1], None)
        found = {}
        if ds is not None:
            found = {k: v for k, v in ds.attrs.items()
                     if any(s in k.lower() for s in keys)}
            if WIND_VAR in ds.data_vars:
                found.update({f"var:{k}": v for k, v in ds[WIND_VAR].attrs.items()
                              if any(s in k.lower() for s in keys)})
        rows.append({"network": net,
                     "expected_m": ANEMOMETER_HEIGHT_M,
                     "attrs_found": "; ".join(f"{k}={v}" for k, v in found.items())
                                    or "none"})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(f"\nHDP standardises to {ANEMOMETER_HEIGHT_M} m -- this is a check, not a "
          "correction. Anything else is worth investigating.")
    return out


# ----------------------------------------------------------------------------
# 7. Screened multi-network build
# ----------------------------------------------------------------------------


def build_screened_multi_network(
    path: str,
    networks=None,
    target=("1/1/1980", "12/31/2010"),
    min_coverage: float = 0.10,
    n_per_network: int | None = None,
    n_screen_per_network: int | None = 400,
    df: pd.DataFrame | None = None,
    exclude=EXCLUDE_NETWORKS,
    audit_wind: bool = True,
    var: str = WIND_VAR,
    freq: str = "h",
    batch_size: int = 1,
    seed: int = 42,
    dry_run: bool = False,
    region=None,
    **kwargs,
) -> pd.DataFrame:
    """Audit -> screen -> select -> build, across every network.

    Pipeline:
      1. drop excluded networks (CW3E: sub-hourly)
      2. drop networks that carry no wind variable (one fetch each)
      3. screen stations for record overlap with `target`, and for location
         inside `region` if given (metadata only -- lat/lon come free, since
         the store is already open)
      4. keep those with coverage >= min_coverage AND inside the region
      5. fetch and append to one zarr store

    Cost control matters here: the full catalog is ~15,000 stations, and
    screening costs roughly one round trip each. `n_screen_per_network` caps
    how many are screened per network; `n_per_network` caps how many are
    actually fetched. Set dry_run=True to run steps 1-4 and stop, which shows
    the yield before committing to the fetch.

    Note `coverage` is record SPAN overlap, not data density within the record.
    A station spanning the window with 40% missing hours still scores 1.0;
    per-station density is reported by open_store() afterwards.
    """
    if df is None:
        df = catalog_df()
    if networks is None:
        networks = sorted(df.network_id.unique())
    networks = [n for n in networks if n not in (exclude or ())]
    print(f"{len(networks)} networks after exclusions {list(exclude or ())}\n")

    # --- step 2: which networks report wind at all -------------------------
    if audit_wind:
        print("=" * 60 + "\nAUDIT: which networks report wind\n" + "=" * 60)
        audit = wind_variable_audit(networks, df, var=var, exclude=exclude)
        networks = audit.loc[audit.has_wind, "network"].tolist()
        print(f"\n{len(networks)} networks carry '{var}'\n")

    # --- step 3-4: screen and select ---------------------------------------
    rows, selected = [], {}
    for net in networks:
        print("=" * 60 + f"\nSCREEN: {net}\n" + "=" * 60)
        stations, _ = list_stations(net, df)
        if n_screen_per_network is not None and len(stations) > n_screen_per_network:
            rng = np.random.default_rng(seed)
            stations = sorted(rng.choice(stations, size=n_screen_per_network,
                                         replace=False).tolist())
            print(f"  screening a random {len(stations)} of the network")
        sc = screen_stations(net, stations, target=target, df=df, var=var,
                             region=region)
        ids = select_by_coverage(sc, min_coverage=min_coverage,
                                 n=n_per_network, seed=seed)
        selected[net] = ids
        rows.append({"network": net, "screened": len(sc),
                     "with_var": int(sc.has_var.sum()),
                     "in_region": int(sc.in_region.sum()),
                     "selected": len(ids),
                     "median_coverage": float(sc.coverage.median())})
        print()

    summary = pd.DataFrame(rows).set_index("network")
    print("=" * 60 + "\nSELECTION SUMMARY\n" + "=" * 60)
    print(summary.to_string())
    total = int(summary.selected.sum())
    print(f"\n{total} stations selected with coverage >= {min_coverage:.0%} "
          f"of {target[0]}..{target[1]}")

    if dry_run:
        print("\ndry_run=True -- stopping before the fetch")
        summary.attrs["selected"] = selected
        return summary

    # --- step 5: build ------------------------------------------------------
    first = True
    for net, ids in selected.items():
        if not ids:
            continue
        print("\n" + "=" * 60 + f"\nBUILD: {net} ({len(ids)} stations)\n" + "=" * 60)
        try:
            build_zarr(net, ids, path, time_slice=target, var=var, freq=freq,
                       batch_size=batch_size, append=not first, **kwargs)
            first = False
        except RuntimeError as e:
            print(f"  {net} produced nothing: {e}")

    if first:
        raise RuntimeError("no network produced any data")

    ds = xr.open_zarr(path, consolidated=True)
    print(f"\nstore: {ds.sizes['station_id']} stations, {ds.sizes['time']} steps")
    print(pd.Series(ds.network.values).value_counts().to_string())
    summary.attrs["selected"] = selected
    return summary


def drop_low_coverage(path: str, out_path: str, min_frac: float = 0.10,
                      var: str = WIND_VAR) -> str:
    """Rewrite a store keeping only stations above a DATA-density threshold.

    Complements the span-based screening: this measures how many of the window's
    time steps actually carry a value, which span overlap cannot see.
    """
    ds = xr.open_zarr(path, consolidated=True)
    frac = ds[var].notnull().mean("time").compute()
    keep = (frac >= min_frac).values
    print(f"keeping {int(keep.sum())} of {ds.sizes['station_id']} stations "
          f"with >= {min_frac:.0%} of steps reporting")
    if keep.sum() == 0:
        raise RuntimeError("threshold removes every station")
    out = ds.sel(station_id=ds.station_id[keep])
    out = clear_encoding(out)
    out.to_zarr(out_path, mode="w", consolidated=True, zarr_format=2)
    print(f"wrote {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# 8. Station metadata CSV  (replaces per-station screening entirely)
# ----------------------------------------------------------------------------
#
# The HDP station list CSV carries lat/lon/elevation, record start/end and
# per-variable observation counts for every station. Selecting from it is a
# dataframe operation, where screen_stations() costs one network round trip per
# station -- for ~15,900 stations that is the difference between seconds and
# hours. Prefer this whenever the CSV is available.

META_ID_COL = "era-id"

# Per-variable observation-count column in the station list CSV. The naming is
# lowercased and suffixed, so most variables resolve by rule; the exceptions are
# listed because a silent fallback to the wrong column is the worst outcome
# here -- selecting on `sfcwind_nobs` for a humidity fetch returns stations that
# measure WIND, and the resulting archive looks plausible and is useless.
OBS_COL_OVERRIDES = {
    "sfcWind": ["sfcwind_nobs"],
    "sfcWind_dir": ["sfcwind_dir_nobs"],
    # RH reaches a station by one of two routes, and which one depends on the
    # instrument. RAWS and CIMIS measure RH natively (`hurs`); ASOS-based
    # networks measure DEWPOINT and HDP derives RH from it (`hurs_derived`).
    # ASOSAWOS has 117 M dewpoint observations and exactly zero RH, so
    # selecting on hurs_nobs alone excludes every airport in the archive --
    # and with them the only unobstructed exposure class there is.
    #
    # There is no `hurs_derived_nobs` column, so dewpoint count stands in for
    # it: a station with temperature and dewpoint is one HDP can serve RH for.
    "hurs": ["hurs_nobs", "tdps_nobs"],
    "tdps": ["tdps_nobs", "tdps_derived_nobs"],
    "ps": ["ps_nobs", "ps_derived_nobs"],
}
WIND_OBS_COL = OBS_COL_OVERRIDES["sfcWind"][0]      # back-compat alias

# Data-variable names a canonical variable may appear under inside a station
# store. Checked in order; the first present is used and renamed to the
# canonical name, so downstream code never has to know which route it took.
VAR_ALIASES = {
    "hurs": ("hurs", "hurs_derived"),
    "tdps": ("tdps", "tdps_derived"),
    "ps": ("ps", "ps_derived"),
}


def obs_columns(var: str, columns=None) -> list:
    """The `*_nobs` column(s) that indicate `var` is available.

    Returns a LIST because a variable can be served by more than one route --
    natively measured or derived from another field -- and a station qualifies
    on any of them. Raises rather than guessing when none is present: a wrong
    column selects the wrong stations, and nothing downstream can detect it.
    """
    cands = OBS_COL_OVERRIDES.get(var, [f"{var.lower()}_nobs"])
    if columns is None:
        return list(cands)
    present = [c for c in cands if c in columns]
    if present:
        return present
    nobs = sorted(c for c in columns if c.endswith("_nobs"))
    raise KeyError(
        f"no observation-count column for '{var}' (looked for {cands}).\n"
        f"Available: {nobs}\n"
        "Add an entry to OBS_COL_OVERRIDES if the naming differs.")


def obs_column(var: str, columns=None) -> str:
    """First matching column. Kept for callers that expect a single name."""
    return obs_columns(var, columns)[0]


def load_station_metadata(
    csv_path: str,
    target=("1/1/1980", "12/31/2010"),
    region=None,
    exclude=EXCLUDE_NETWORKS,
    max_end: str = "2023-12-31",
    var: str = WIND_VAR,
) -> pd.DataFrame:
    """Load and clean the station list, adding coverage / wind / region flags.

    Cleaning applied, each for a reason found in the file:
      * end dates beyond `max_end` are clipped -- 1108 rows carry sentinels
        running to 2100
      * elevations below -1000 m are set NaN (a -30479.6952 sentinel, exactly
        -100,000 ft, appears in ASOSAWOS)
      * `elevation_suspect` flags networks whose values are plainly NOT metres

    ELEVATION UNITS ARE INCONSISTENT ACROSS NETWORKS in this file: ASOSAWOS and
    NOS-PORTS read as metres (medians 774 and 0), while RAWS, HADS and CRN read
    as feet (medians 4420, 4875, 4550 -- impossible in metres for California).
    Do NOT use this column for terrain analysis without resolving that. The
    elevation promoted from the retrieved station stores is the safer source.
    """
    d = pd.read_csv(csv_path, index_col=0)
    d = d.rename(columns={META_ID_COL: "station_id",
                          "latitude": "lat", "longitude": "lon"})

    # Make the loader idempotent. stations.csv is written AFTER this rename, so
    # on re-read there is no `era-id` to rename and the id has landed in the
    # index via index_col=0. That file is the natural fallback when S3 is
    # unreachable, so the function must be able to read what it writes.
    if "station_id" not in d.columns:
        idx_name = d.index.name
        if idx_name in (META_ID_COL, "station_id"):
            d = d.reset_index().rename(columns={idx_name: "station_id"})
        elif d.index.dtype == object:
            # Unnamed object index: almost certainly the id, since the file was
            # written with the id as index.
            d = d.reset_index().rename(columns={"index": "station_id"})
        else:
            raise KeyError(
                f"no '{META_ID_COL}' or 'station_id' column and the index "
                f"({idx_name!r}, dtype {d.index.dtype}) does not look like an "
                f"id. Columns: {list(d.columns)[:12]}")

    # Date columns keep their original names on a first read and the cleaned
    # names on a re-read, so accept either rather than requiring the raw file.
    for src, dst in (("start-date", "start"), ("end-date", "end")):
        col = src if src in d.columns else dst
        if col not in d.columns:
            raise KeyError(f"no '{src}' or '{dst}' column in {csv_path}")
        d[dst] = pd.to_datetime(d[col], format="ISO8601", utc=True,
                                errors="coerce").dt.tz_localize(None)

    n_bad = int((d.end > pd.Timestamp(max_end)).sum())
    if n_bad:
        print(f"clipped {n_bad} end dates beyond {max_end}")
        d.loc[d.end > pd.Timestamp(max_end), "end"] = pd.Timestamp(max_end)

    n_elev = int((d.elevation < -1000).sum())
    if n_elev:
        print(f"nulled {n_elev} sentinel elevations below -1000")
        d.loc[d.elevation < -1000, "elevation"] = np.nan

    if exclude:
        n = int(d.network.isin(exclude).sum())
        d = d[~d.network.isin(exclude)]
        print(f"excluded {n} stations from {list(exclude)}")

    # Select on the requested VARIABLE's observation count, not on wind's.
    # A humidity fetch that filters on sfcwind_nobs returns stations that
    # measure wind -- which is exactly as wrong as it sounds and completely
    # invisible in the resulting store.
    cols = obs_columns(var, d.columns)
    d["has_var"] = (d[cols] > 0).any(axis=1)
    d["has_wind"] = d["has_var"]          # back-compat for wind-era callers
    # Which route each station qualifies by, so the split is visible rather
    # than inferred from which networks show up.
    d["var_route"] = np.where(d[cols[0]] > 0, cols[0].replace("_nobs", ""),
                              "derived" if len(cols) > 1 else "none")
    d.attrs["variable"] = var
    d.attrs["obs_columns"] = cols
    if len(cols) > 1:
        by = {c: int((d[c] > 0).sum()) for c in cols}
        print(f"'{var}' available via {by} "
              "(a station qualifies on any route)")

    t0, t1 = pd.Timestamp(target[0]), pd.Timestamp(target[1])
    span = max((t1 - t0).days, 1)
    overlap = (d.end.clip(upper=t1) - d.start.clip(lower=t0)).dt.days
    d["coverage"] = (overlap / span).clip(0, 1)

    if region is not None:
        from matplotlib.path import Path as _Path
        region = np.asarray(region, dtype=float)
        pts = np.column_stack([d.lon.values, d.lat.values])
        ok = np.isfinite(pts).all(axis=1)
        inside = np.zeros(len(d), bool)
        if region.ndim == 1 and region.size == 4:
            lo0, lo1, la0, la1 = region
            inside[ok] = ((pts[ok, 0] >= lo0) & (pts[ok, 0] <= lo1) &
                          (pts[ok, 1] >= la0) & (pts[ok, 1] <= la1))
        else:
            inside[ok] = _Path(region).contains_points(pts[ok])
        d["in_region"] = inside
    else:
        d["in_region"] = True

    # Flag networks whose elevations cannot be metres in this domain.
    med = d.groupby("network").elevation.median()
    suspect = set(med[med > 4500].index)          # > Mt Whitney in metres
    d["elevation_suspect"] = d.network.isin(suspect)
    if suspect:
        print(f"elevation likely in FEET (median > 4500) for: {sorted(suspect)}")

    print(f"\n{len(d)} stations | {int(d.has_var.sum())} with '{var}' "
          f"(columns {cols}) | {int(d.in_region.sum())} in region | "
          f"{int((d.has_var & d.in_region & (d.coverage >= 0.1)).sum())} "
          f"passing all three at coverage>=10%")
    return d


def metadata_summary(meta: pd.DataFrame, min_coverage: float = 0.10) -> pd.DataFrame:
    """Per-network yield after each successive filter."""
    var = meta.attrs.get("variable", WIND_VAR)
    rows = []
    for net, g in meta.groupby("network"):
        wind = g[g.has_var]
        reg = wind[wind.in_region]
        sel = reg[reg.coverage >= min_coverage]
        rows.append({"network": net, "total": len(g), f"with_{var}": len(wind),
                     "in_region": len(reg), "selected": len(sel),
                     "median_coverage": round(float(reg.coverage.median()), 3)
                                        if len(reg) else np.nan})
    out = (pd.DataFrame(rows).set_index("network")
             .sort_values("selected", ascending=False))
    print(out.to_string())
    print(f"\nTOTAL selected: {int(out.selected.sum())}")
    return out


def select_from_metadata(meta: pd.DataFrame, min_coverage: float = 0.10,
                         n_per_network: int | None = None,
                         seed: int = 42) -> dict:
    """{network: [station_id, ...]} passing wind, region and coverage filters."""
    sel = meta[meta.has_var & meta.in_region & (meta.coverage >= min_coverage)]
    rng = np.random.default_rng(seed)
    out = {}
    for net, g in sel.groupby("network"):
        ids = sorted(g.station_id.tolist())
        if n_per_network is not None and len(ids) > n_per_network:
            ids = sorted(rng.choice(ids, size=n_per_network, replace=False).tolist())
        out[net] = ids
    print(f"{sum(len(v) for v in out.values())} stations across {len(out)} networks")
    return out


def build_from_selection(selected: dict, path: str,
                         time_slice=("1/1/1980", "12/31/2010"),
                         var: str = WIND_VAR, freq: str = "h",
                         batch_size: int = 1, **kwargs) -> str:
    """Fetch a {network: ids} selection into one store."""
    first = True
    for net, ids in selected.items():
        if not ids:
            continue
        print("\n" + "=" * 60 + f"\n{net} ({len(ids)} stations)\n" + "=" * 60)
        try:
            build_zarr(net, ids, path, time_slice=time_slice, var=var, freq=freq,
                       batch_size=batch_size, append=not first, **kwargs)
            first = False
        except RuntimeError as e:
            print(f"  {net} produced nothing: {e}")
    if first:
        raise RuntimeError("no network produced any data")
    ds = xr.open_zarr(path, consolidated=True)
    print(f"\nstore: {ds.sizes['station_id']} stations, {ds.sizes['time']} steps")
    print(pd.Series(ds.network.values).value_counts().to_string())
    return path


# ----------------------------------------------------------------------------
# 9. Quality control
# ----------------------------------------------------------------------------
#
# HDP is QA/QC'd for metadata (notably the 10 m anemometer height) but the wind
# values still contain physically impossible entries: negative scalar speeds and
# values in the thousands of m/s, both of which are sentinels or unit errors
# rather than observations.

# Plausibility bounds for 10 m scalar wind speed.
#   lower: a scalar magnitude cannot be negative
#   upper: the highest reliably measured surface wind on Earth is ~113 m/s
#          (Barrow Island, 1996). 75 m/s is already far beyond anything the
#          California domain produces, so it is a conservative ceiling that
#          removes sentinels without touching real extremes.
WIND_MIN, WIND_MAX = 0.0, 75.0


def outlier_report(ds: xr.Dataset, var: str | None = None,
                   lo: float | None = None, hi: float | None = None,
                   top: int = 15) -> pd.DataFrame:
    """Which stations produce out-of-range values, and how many.

    Bounds default to the wind range only when the variable is wind. For any
    other variable they must be given: [0, 75] is meaningless for relative
    humidity and would pass every impossible value through.

    Run before masking: if bad values concentrate in a few stations, dropping
    those stations is cleaner than masking values across the whole store.
    """
    if var is None:
        dv = list(ds.data_vars)
        if len(dv) != 1:
            raise ValueError(f"store holds {dv}; pass var= to choose one")
        var = dv[0]
    if lo is None or hi is None:
        if var != WIND_VAR:
            raise ValueError(
                f"no default bounds for '{var}'. Pass lo= and hi= -- the wind "
                f"range [{WIND_MIN}, {WIND_MAX}] is not meaningful here.")
        lo, hi = WIND_MIN, WIND_MAX

    da = ds[var]
    bad = ((da < lo) | (da > hi)) & da.notnull()
    n_bad = bad.sum("time").compute()
    n_val = da.notnull().sum("time").compute()

    out = pd.DataFrame({
        "network": ds.network.values,
        "n_bad": n_bad.values,
        "n_valid": n_val.values,
        "frac_bad": np.where(n_val.values > 0, n_bad.values / n_val.values, 0.0),
        "max_value": da.max("time").compute().values,
        "min_value": da.min("time").compute().values,
    }, index=ds.station_id.values)

    tot_bad, tot_val = int(out.n_bad.sum()), int(out.n_valid.sum())
    print(f"{tot_bad:,} of {tot_val:,} values outside [{lo}, {hi}] "
          f"({tot_bad / max(tot_val, 1):.4%})")
    print(f"affected stations: {int((out.n_bad > 0).sum())} of {len(out)}\n")
    print("worst offenders:")
    print(out.sort_values("n_bad", ascending=False).head(top).to_string())
    print("\nby network:")
    print(out.groupby("network").agg(
        stations=("n_bad", "size"),
        affected=("n_bad", lambda s: int((s > 0).sum())),
        bad_values=("n_bad", "sum"),
        worst_max=("max_value", "max"),
    ).to_string())
    return out


# Back-compat alias; new code should call outlier_report with explicit bounds.
wind_outlier_report = outlier_report


def qc_wind(ds: xr.Dataset, var: str = WIND_VAR,
            lo: float = WIND_MIN, hi: float = WIND_MAX,
            drop_stations_above: float | None = 0.01) -> xr.Dataset:
    """Mask out-of-range wind values, optionally dropping the worst stations.

    `drop_stations_above` removes stations whose bad-value fraction exceeds the
    threshold: a station producing many impossible values is not trustworthy for
    the values that happen to fall in range either. Set None to mask only.
    """
    da = ds[var]
    bad = ((da < lo) | (da > hi)) & da.notnull()
    n_bad = int(bad.sum().compute())
    print(f"masking {n_bad:,} values outside [{lo}, {hi}] m/s")

    out = ds.copy()
    out[var] = da.where(~bad)

    if drop_stations_above is not None:
        frac = (bad.sum("time") / da.notnull().sum("time")).compute()
        keep = (frac.fillna(0) <= drop_stations_above).values
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"dropping {n_drop} stations with >{drop_stations_above:.1%} "
                  "bad values")
            out = out.sel(station_id=out.station_id[keep])

    v = out[var].values.ravel()
    v = v[np.isfinite(v)]
    print(f"\nafter QC: {v.size:,} values, "
          f"min {v.min():.2f} / max {v.max():.2f} m/s, "
          f"mean {v.mean():.2f} / median {np.median(v):.2f}")
    return out
