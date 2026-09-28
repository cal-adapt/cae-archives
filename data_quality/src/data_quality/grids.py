#!/usr/bin/env python
"""
Grid handling: putting two differently-gridded products onto one common frame.

Nothing here knows where data came from. Every function takes arrays and
returns arrays, so the same code serves a catalog query, a local zarr or a
synthetic test. Retrieval lives in the fetch_* scripts and localdata.py.

THE CENTRAL PROBLEM. LOCA2 is rectilinear at 3 km, WRF is curvilinear on a
rotated 3 km projection, CONUS404 is curvilinear at 4 km, ERA5 is rectilinear at
0.25 deg. Comparing them means choosing a target grid and aggregating onto it,
and the aggregation has to be honest about partial coverage at the edges --
which `coverage_fraction` measures and every metric then masks on.

WHAT THE PIECES DO

  build_labels / bin_to_era5   assign each source cell to a target cell, then
                               reduce. Handles both grid kinds.
  coverage_fraction            what fraction of each target cell is backed by
                               valid source data; the mask everything uses.
  crop_era5_to_source          bounding-box crop, handling 0-360 longitude and
                               descending latitude.
  elevation_on_era5            terrain mean AND sub-grid relief per target cell.
                               Relief is the better predictor of a
                               resolution-driven discrepancy: a flat plateau and
                               a rugged range can share a mean height.
  sample_grid_at_points        nearest grid value at station locations, for
                               both grid kinds.
  regrid_nearest               resample onto another model grid, for
                               comparisons that must not pass through ERA5.

Requires: numpy, pandas, xarray, dask. `flox` is effectively required -- without
it xarray's groupby falls back to a Python loop and binning takes minutes
instead of seconds. `xesmf` is optional and only for regrid_conservative.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

# Default window. Every product in the archive spans at least this, and callers
# that care pass an explicit slice -- see COMMON_* in build_metrics.py.
TIME_START = "1980-01-01"
TIME_END = "2014-12-31"


SIM = 0                      # int index, member-name string, or "mean"
COVERAGE_THRESHOLD = 0.5


COVERAGE_THRESHOLD = 0.5
BBOX_PAD = 0.25


BBOX_PAD = 0.25
PROGRESS = True



# Candidate names for the ensemble-member dimension, in priority order.
MEMBER_DIMS = ("sim", "simulation", "member_id", "member")



def progress(label: str = ""):
    """Dask progress bar for local schedulers. No-op under dask.distributed."""
    if not PROGRESS:
        return nullcontext()
    if label:
        print(label)
    return ProgressBar(minimum=1.0)



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



def source_to_era5_grid(src, kind, labels, era5_crop, variable,
                        sim=SIM, time_start=TIME_START, time_end=TIME_END):
    """Block-average the source onto the ERA5 cells.

    Coarsening up, never interpolating ERA5 down: a 3 km field interpolated from
    31 km data would be fabricated structure.
    """
    da = src[variable].sel(time=slice(time_start, time_end))
    da = select_member(da, sim)
    return bin_to_era5(da, kind, labels, era5_crop, how="mean")



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



def land_fraction_on_era5(landmask, era5_crop, var: str = "LANDMASK",
                          kind=None, labels=None) -> xr.DataArray:
    """Fraction of each target cell that is land, from a native-resolution mask.

    A GEOGRAPHIC mask, not a product artifact. LOCA2's NaN footprint happens to
    be land, so it can stand in for one -- but it is that product's own masking
    decision, and using it silently makes every stratification depend on which
    product was loaded. Binning a real land mask separates the two.

    This matters because the resolution effect only exists where there is
    terrain: over open water a km-scale model and a 31 km reanalysis have
    nothing sub-grid to disagree about and their ratio sits near 1.0, so a
    median pooled over a domain with a large ocean fraction is diluted by cells
    that carry no signal. Reporting land and ocean separately turns that from a
    confound into the measurement.

    Returns a fraction in [0, 1]: 1 is all land, 0 all water, intermediate
    values are coastal cells -- which is a category worth keeping distinct
    rather than thresholding away, since a coastal cell mixes two regimes.
    """
    if isinstance(landmask, xr.Dataset):
        if var not in landmask:
            var = list(landmask.data_vars)[0]
        lm, src = landmask[var], landmask
    else:
        lm, src = landmask, landmask.to_dataset(name=landmask.name or var)

    if labels is None:
        kind, labels = build_labels(src, era5_crop)
        print(f"land mask grid: {kind}, {lm.shape}")

    frac = bin_to_era5(lm.astype("float32"), kind, labels, era5_crop,
                       how="mean")
    frac.name = "land_fraction"
    frac.attrs.update(units="1", long_name="fraction of cell that is land",
                      source=f"binned from native {var}")
    return frac


def region_masks(land_fraction: xr.DataArray, land_min: float = 0.9,
                 ocean_max: float = 0.1) -> dict:
    """land / ocean / coastal masks from a land fraction.

    The thresholds are deliberately asymmetric about 0.5: a cell that is 60%
    land is not a land cell for this purpose, because the 40% of water in it
    still has no terrain to resolve and still pulls the ratio toward 1. Cells
    between the two cuts are COASTAL and reported separately rather than
    assigned to whichever side is closer.
    """
    return {"land": land_fraction >= land_min,
            "ocean": land_fraction <= ocean_max,
            "coastal": (land_fraction > ocean_max) & (land_fraction < land_min)}


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



def cos_weights(da: xr.DataArray) -> xr.DataArray:
    """cos(latitude) weights, for either grid kind."""
    lat = da["lat"] if "lat" in da.coords else da["latitude"]
    return np.cos(np.deg2rad(lat))