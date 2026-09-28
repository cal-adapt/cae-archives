#!/usr/bin/env python
"""
Fetch the FULL HOURLY series at HDP station locations only.

    python fetch_stations_1hr.py hurs --probe
    python fetch_stations_1hr.py hurs --products wrf-era5 --dry-run
    python fetch_stations_1hr.py hurs --products wrf-era5 wrf-gcm

    /shared/data/{variable}/{product}_stations_1hr.zarr
        hourly {variable} at every HDP station, with station_id, network,
        lat/lon and the distance to the cell that was sampled

WHY POINTS RATHER THAN THE DOMAIN. The diurnal cycle is where humidity carries
most of its signal -- RH swings tens of points between a moist night and a dry
afternoon, and the fire-relevant statistic is a point on that curve. But keeping
hourly data for the whole domain costs 268 GB per member and every cell away
from a station would be read once and averaged. At ~2,200 station points the
same information is a couple of GB.

THE CLIP PROCESSOR IS THE FAST PATH, IF IT SELECTS SERVER-SIDE. climakitae
accepts a list of (lat, lon) with `separated: True`, which asks the catalog for
those points rather than the grid. Whether that avoids fetching whole chunks is
an empirical question -- if clip is a post-open .sel, dask still pulls every
chunk containing a station and the saving is only the KDTree.

    python fetch_stations_1hr.py hurs --probe

times one month at three points against the same query without clip, and prints
which path to use. The answer decides whether five WRF members are affordable.

--method controls it explicitly: `clip` uses the processor, `sample` opens the
grid and samples with a KDTree (the always-works fallback), `auto` probes once
and picks. Both paths write the SAME output, so the choice is performance only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_ROOT = "/shared/data"
DEFAULT_META = ("s3://wecc-historical-wx/4_merge_wx_v2/"
                "all_network_stationlist_merge.csv")
ZARR_FORMAT = 2
GRID_LABEL = "d03"
TIME_SLICE = ("1980-01-01", "2014-12-31")
TARGET_CHUNK_MB = 128

VARIABLES = {"hurs": {"WRF": "rh"},
             "wspeed": {"WRF": "wspd10mean"},
             "tas": {"WRF": "t2"}}

PRODUCTS = {
    "wrf-gcm":  dict(activity="WRF", experiment="historical",
                     institution="UCLA", filter="yes"),
    "wrf-era5": dict(activity="WRF", experiment="reanalysis",
                     institution="UCLA", filter="no"),
}

HDP_STORES = {"hurs": "hdp_hurs_hourly.zarr", "sfcWind": "hdp_wind_hourly.zarr"}
MEMBER_DIMS = ("sim", "simulation", "member_id", "member")


def member_dim(da):
    return next((d for d in MEMBER_DIMS if d in da.dims), None)


# ----------------------------------------------------------------------------
# Station locations
# ----------------------------------------------------------------------------


def station_points(root: Path, variable: str, meta_csv: str | None = None):
    """(ids, lat, lon, network) for every station carrying `variable`.

    EVERY station is taken, not the quality-screened subset. Screening is a
    downstream choice that has already moved several times; re-running it is
    free, while refetching to recover a station is not.

    COORDINATES COME FROM THE STATION LIST WHERE THE STORE LACKS THEM. The
    zarr's lat/lon are promoted from each station's own store during retrieval,
    so a station whose fetch failed carries NaN. The CSV has coordinates
    regardless of whether the data arrived.
    """
    hdp = root / "hdp"
    path = hdp / HDP_STORES.get(variable, f"hdp_{variable}_hourly.zarr")
    if not path.exists():
        path = hdp / "hdp_hourly.zarr"
    if not path.exists():
        raise FileNotFoundError(f"no HDP store under {hdp}")

    ds = xr.open_zarr(path, consolidated=True)
    if variable in ds:
        has = (ds[variable].notnull().sum("time") > 0).compute().values
        ds = ds.isel(station_id=np.where(has)[0])

    ids = [str(s) for s in ds.station_id.values]
    lat = np.asarray(ds.lat.values, dtype=float)
    lon = np.asarray(ds.lon.values, dtype=float)
    net = np.array([str(n) for n in ds.network.values], dtype=object)

    miss = ~(np.isfinite(lat) & np.isfinite(lon))
    if miss.any() and meta_csv:
        try:
            m = pd.read_csv(meta_csv, index_col=0)
            m = m.rename(columns={"era-id": "station_id",
                                  "latitude": "lat", "longitude": "lon"})
            if "station_id" not in m.columns:
                m = m.reset_index().rename(
                    columns={m.index.name or "index": "station_id"})
            look = m.dropna(subset=["lat", "lon"]).drop_duplicates("station_id")
            look = look.set_index(look.station_id.astype(str))
            n0 = int(miss.sum())
            for i in np.where(miss)[0]:
                if ids[i] in look.index:
                    lat[i] = float(look.at[ids[i], "lat"])
                    lon[i] = float(look.at[ids[i], "lon"])
            miss = ~(np.isfinite(lat) & np.isfinite(lon))
            print(f"    backfilled {n0 - int(miss.sum())} of {n0} missing "
                  "coordinate(s) from the station list")
        except Exception as e:
            print(f"    could not read {meta_csv} ({type(e).__name__})")

    if miss.any():
        print(f"    dropping {int(miss.sum())} station(s) without coordinates:")
        for n, c in pd.Series(net[miss]).value_counts().items():
            print(f"      {n:<12} {c}")
        keep = np.where(~miss)[0]
        ids = [ids[i] for i in keep]
        lat, lon, net = lat[keep], lon[keep], net[keep]

    if not ids:
        raise FileNotFoundError(f"{path.name}: no usable coordinates")
    print(f"    {len(ids)} stations from {path.name}")
    return ids, lat, lon, net


# ----------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------


def restrict_to_domain(ids, lat, lon, net, root: Path, variable: str):
    """Drop stations outside the model domain.

    THE HDP LIST IS NOT A CALIFORNIA LIST. It reaches from Phoenix to Boise, and
    a station 800 km outside d03 contributes nothing but an enormous bounding
    box -- which is exactly what makes the clip processor useless at full
    extent, since clip narrows a REGION rather than selecting points.

    The polygon comes from a fetched product's own footprint rather than a
    lat/lon box: d03 is rotated, so its bounding box over-counts substantially.
    """
    # domain_polygon lives in hdp_stations, where it was written for the
    # station screening. grids.py is checked too so a later move does not break
    # this, and a bounding box is the last resort -- it OVER-counts on a rotated
    # grid, but a loose filter still removes the Phoenix and Boise stations,
    # which is the point.
    from matplotlib.path import Path as MplPath
    domain_polygon = None
    for mod in ("hdp_stations", "grids"):
        try:
            domain_polygon = getattr(__import__(mod), "domain_polygon")
            break
        except (ImportError, AttributeError):
            continue

    src = None
    for cand in sorted((root / variable).glob("*_*.zarr")):
        try:
            src = xr.open_zarr(cand, consolidated=True)
            break
        except Exception:
            continue
    if src is None:
        print("    no fetched product to take a domain from; keeping all")
        return ids, lat, lon, net

    if domain_polygon is not None:
        inside = MplPath(domain_polygon(src)).contains_points(
            np.column_stack([lon, lat]))
    else:
        from .grids import get_latlon_any
        gl, go = get_latlon_any(src)
        la, lo = np.asarray(gl.values), np.asarray(go.values)
        print("    domain_polygon unavailable; using a bounding box "
              "(over-counts on a rotated grid)")
        inside = ((lat >= np.nanmin(la)) & (lat <= np.nanmax(la))
                  & (lon >= np.nanmin(lo)) & (lon <= np.nanmax(lo)))
    if inside.all():
        print(f"    all {len(ids)} stations inside the domain")
        return ids, lat, lon, net

    print(f"    {int((~inside).sum())} of {len(ids)} stations OUTSIDE the "
          "model domain:")
    for n, c in pd.Series(net[~inside]).value_counts().head(6).items():
        print(f"      {n:<12} {c}")
    k = np.where(inside)[0]
    ids = [ids[i] for i in k]
    lat, lon, net = lat[k], lon[k], net[k]
    print(f"    {len(ids)} inside ({lat.min():.1f}..{lat.max():.1f} N, "
          f"{lon.min():.1f}..{lon.max():.1f} E)")
    return ids, lat, lon, net


def query(spec, variable_id, table_id, time_slice, clip_points=None,
          clip_key="points"):
    from climakitae import ClimateData

    processes = {"time_slice": time_slice}
    if spec["filter"] is not None:
        processes["filter_unadjusted_models"] = spec["filter"]
    if clip_points is not None:
        # `points` vs `boundaries` is the whole question. The docstring lists
        # both keys, and the earlier attempt used `boundaries`, which the
        # processor logged as `extract_points=False` -- a bbox crop with the
        # cells masked, returning y=243, x=23 for three cities. `points` is the
        # closest-gridcell extraction path, which is what a station comparison
        # actually wants.
        #
        # Both are tried, because the accepted key differs between versions and
        # the failure is a validation error rather than anything subtle.
        # Valid keys in this climakitae version are {boundaries, separated,
        # persist}; location_based_naming is rejected outright.
        #
        # Boundaries must also be UNIQUE -- the validator refuses repeats, and
        # many stations share a location to the reported precision.
        # Deduplicating is safe because clip returns a bbox-clipped REGION
        # rather than one series per point (it logs extract_points=False), so
        # stations are matched by sampling the result, not by position here.
        uniq = sorted({(round(float(a), 5), round(float(b), 5))
                       for a, b in clip_points})
        print(f"    clip: {len(uniq)} unique locations from "
              f"{len(clip_points)} stations")
        processes["clip"] = {clip_key: uniq, "separated": True}
    q = (ClimateData().catalog("cadcat")
         .activity_id(spec["activity"]).experiment_id([spec["experiment"]])
         .table_id(table_id).grid_label(GRID_LABEL)
         .variable_id(variable_id).processes(processes))
    if spec["institution"]:
        q = q.institution_id(spec["institution"])
    out = q.get()
    if out is None:
        raise RuntimeError(f"query returned nothing for {variable_id}/{table_id}")
    return out


def probe(spec, variable_id, points=None, month=("2010-01-01", "2010-01-31")):
    """Time clip against plain open, on one month at a few points.

    The question is whether clip selects SERVER-SIDE. If it does, the fetch
    scales with the number of points and five WRF members become affordable. If
    it is a post-open .sel, dask still pulls every chunk containing a station
    and clip saves only the KDTree -- a few seconds against hours.
    """
    # Probe with the ACTUAL station coordinates where available. Three
    # clustered cities gave clip a tiny bounding box and a 5.8x speedup that
    # does not survive a station list spanning several states: clip narrows a
    # REGION, so its benefit depends entirely on how spread out the points are.
    if points is not None and len(points) >= 3:
        pts = list(points)
        print(f"  probing with {len(pts)} real station coordinates")
    else:
        pts = [(34.05, -118.25), (37.77, -122.42), (32.72, -117.16)]
        print("  probing with 3 sample cities -- NOT representative of a "
              "domain-wide station list")

    print(f"  probing one month ({month[0]}..{month[1]}) at {len(pts)} points")
    timings = {}
    for label, cp in (("clip", pts), ("no clip", None)):
        try:
            t0 = time.time()
            d = query(spec, variable_id, "1hr", month, clip_points=cp)
            da = d if isinstance(d, xr.DataArray) else d[list(d.data_vars)[0]]
            t_open = time.time() - t0
            t0 = time.time()
            v = da.compute()
            t_read = time.time() - t0
            timings[label] = t_read
            print(f"    {label:<8} open {t_open:5.1f}s  compute {t_read:6.1f}s  "
                  f"-> {dict(v.sizes)}")
        except Exception as e:
            print(f"    {label:<8} FAILED: {type(e).__name__}: {e}")
            timings[label] = None

    if timings.get("clip") and timings.get("no clip"):
        r = timings["no clip"] / timings["clip"]
        print(f"\n  clip is {r:.1f}x faster on this sample")
        if r > 3:
            print("  -> clip selects server-side; use --method clip")
            return "clip"
        print("  -> clip is not avoiding the chunk reads; use --method sample,\n"
              "     which is predictable and needs no processor support")
        return "sample"
    if timings.get("clip"):
        print("\n  only clip succeeded; use --method clip")
        return "clip"
    print("\n  clip unavailable; use --method sample")
    return "sample"


def by_clip(spec, vid, tslice, ids, lat, lon, net, tol) -> xr.Dataset:
    """Clip to the stations' bounding region, then sample at the points.

    CLIP IS A REGION FILTER, NOT A POINT SELECTOR. The processor logs
    `extract_points=False` and returns a bbox-clipped grid with the requested
    cells masked -- three California cities came back as y=243, x=23. So it
    narrows the read when the points are CLUSTERED and does nothing when they
    span the domain, and either way the per-station series still has to be
    sampled out of the grid it returns.

    Which makes it worth using only after the station list has been restricted
    to the model domain: an unrestricted HDP list reaches from Phoenix to Boise,
    and that bounding box is larger than d03.
    """
    from .grids import sample_grid_at_points

    pts = list(zip(lat, lon))
    out = None
    for key in ("points", "boundaries"):
        try:
            out = query(spec, vid, "1hr", tslice, clip_points=pts, clip_key=key)
            print(f"    clip accepted the '{key}' key")
            break
        except Exception as e:
            print(f"    clip '{key}' failed: {type(e).__name__}: "
                  f"{str(e)[:90]}")
    if out is None:
        raise RuntimeError("clip rejected both 'points' and 'boundaries'; "
                           "use --method sample")

    da = out if isinstance(out, xr.DataArray) else out[list(out.data_vars)[0]]
    print(f"    returned {dict(da.sizes)}, ~{da.nbytes / 1024 ** 3:.2f} GB")

    # Detect the extraction path by SHAPE, not by count.
    #
    # The returned point dimension is NOT the number requested: clip
    # deduplicates to unique GRIDCELLS, so 2,171 stations -> 2,155 unique
    # coordinates -> 1,928 cells, because several stations share a 3 km cell.
    # Matching on a count would therefore miss the very case it was meant to
    # catch. What identifies the extraction is that the result is
    # ONE-DIMENSIONAL in space -- a mask keeps the grid's two spatial dims.
    spatial = [d for d in da.dims if d not in ("time",) + MEMBER_DIMS]
    if len(spatial) == 1:
        pdim = spatial[0]
        print(f"    point extraction: '{pdim}' has {da.sizes[pdim]} cells for "
              f"{len(pts)} stations (several stations share a 3 km cell)")
        return _match_extracted(da, pdim, ids, lat, lon, net, tol)

    at, dist = sample_grid_at_points(da, lat, lon, ids=ids, max_dist_deg=tol)
    far = int((np.asarray(dist) > tol).sum())
    if far:
        print(f"    {far} station(s) beyond {tol} deg of a cell centre")
    return at.to_dataset(name="value").assign_coords(
        network=("station_id", net), lat=("station_id", lat),
        lon=("station_id", lon),
        cell_distance_deg=("station_id", np.asarray(dist, dtype="float32")))


def _match_extracted(da, pdim, ids, lat, lon, net, tol) -> xr.Dataset:
    """Map extracted gridcells back to stations by coordinate.

    NOT by position. The processor documents that it "automatically removes
    duplicate gridcells when multiple points map to the same location", so the
    returned series can be fewer than the points requested and in an order that
    is not the request order. Matching on coordinates survives both; matching on
    position would silently mislabel every station after the first duplicate.

    A second documented behaviour needs watching: point clipping "searches for
    the nearest gridcell with valid data within expanding radii (0.01, 0.05,
    0.1, 0.2, 0.5 deg)". At 0.5 deg that is about 55 km, so a coastal station
    over water could be silently assigned an inland cell. The distance is
    recorded per station so that substitution is visible rather than assumed.
    """
    from scipy.spatial import cKDTree

    # The extracted cells carry their own lat/lon, which is what makes the
    # match possible. Several names are tried because the processor does not
    # document which it uses, and getting this wrong would silently pair every
    # station with the wrong cell.
    plat = plon = None
    for a, b in (("lat", "lon"), ("latitude", "longitude"),
                 ("XLAT", "XLONG"), ("y", "x")):
        if a in da.coords and b in da.coords:
            ca, cb = np.asarray(da[a].values), np.asarray(da[b].values)
            if ca.size == da.sizes[pdim] and cb.size == da.sizes[pdim]:
                plat, plon = ca.ravel(), cb.ravel()
                print(f"    matching on coords '{a}'/'{b}'")
                break
    if plat is None:
        raise RuntimeError(
            f"extracted dim '{pdim}' carries no usable lat/lon "
            f"(coords: {list(da.coords)}). Use --method sample.")

    tree = cKDTree(np.column_stack([plat, plon]))
    dist, idx = tree.query(np.column_stack([lat, lon]))
    far = int((dist > tol).sum())
    if far:
        print(f"    {far} station(s) matched to a cell more than {tol} deg "
              "away -- the processor's expanding-radius search")

    out = da.isel({pdim: idx}).rename({pdim: "station_id"})
    return out.to_dataset(name="value").assign_coords(
        station_id=np.array(ids, dtype=object),
        network=("station_id", net), lat=("station_id", lat),
        lon=("station_id", lon),
        cell_distance_deg=("station_id", dist.astype("float32")))


def by_sample(spec, vid, tslice, ids, lat, lon, net, tol) -> xr.Dataset:
    """Open the grid and sample with a KDTree. Always works."""
    from .grids import sample_grid_at_points

    out = query(spec, vid, "1hr", tslice)
    da = out if isinstance(out, xr.DataArray) else out[list(out.data_vars)[0]]
    print(f"    grid {dict(da.sizes)}, ~{da.nbytes / 1024 ** 3:.0f} GB to read")
    at, dist = sample_grid_at_points(da, lat, lon, ids=ids, max_dist_deg=tol)
    far = int((np.asarray(dist) > tol).sum())
    if far:
        print(f"    {far} station(s) beyond {tol} deg of a cell centre -- "
              "outside the domain, left as NaN")
    return at.to_dataset(name="value").assign_coords(
        network=("station_id", net), lat=("station_id", lat),
        lon=("station_id", lon),
        cell_distance_deg=("station_id", np.asarray(dist, dtype="float32")))


def write(ds: xr.Dataset, path: Path, attrs: dict, overwrite: bool):
    md = member_dim(ds[list(ds.data_vars)[0]])
    per_step = sum(ds[v].dtype.itemsize * int(np.prod(
        [ds.sizes[d] for d in ds[v].dims if d != "time"]))
        for v in ds.data_vars if "time" in ds[v].dims)
    n_steps = min(int(ds.sizes["time"]),
                  max(1, int(TARGET_CHUNK_MB * 1024 ** 2) // max(per_step, 1)))
    chunks = {"time": n_steps}
    chunks |= {d: (1 if d == md else -1) for d in ds.dims if d != "time"}
    ds = ds.chunk(chunks)

    gb = sum(ds[v].nbytes for v in ds.data_vars) / 1024 ** 3
    print(f"    {dict(ds.sizes)}  ~{gb:.2f} GB")
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists():
        print("    exists, skipping (--overwrite to rebuild)")
        return
    ds = ds.copy()
    for v in ds.variables:
        ds[v].encoding = {}
        if ds[v].dtype.kind in ("T", "U"):
            ds[v] = ds[v].astype(object)
    ds.attrs.update(attrs)
    tmp = path.parent / (path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        ds.to_zarr(tmp, mode="w", consolidated=True, zarr_format=ZARR_FORMAT)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.rename(path)
    print(f"    wrote {path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variable", help=f"one of {', '.join(VARIABLES)}")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--products", nargs="*", default=["wrf-era5"],
                    choices=list(PRODUCTS))
    ap.add_argument("--time-slice", nargs=2, default=list(TIME_SLICE),
                    metavar=("START", "END"))
    ap.add_argument("--wrf-var", default=None)
    ap.add_argument("--meta-csv", default=DEFAULT_META)
    ap.add_argument("--method", default="auto",
                    choices=["auto", "clip", "sample"])
    ap.add_argument("--max-dist-deg", type=float, default=0.05,
                    help="sample-method tolerance. d03 is ~3 km (0.027 deg), "
                         "so this allows about two cells.")
    ap.add_argument("--no-domain-filter", action="store_true",
                    help="keep stations outside the model domain. They sample "
                         "as NaN and inflate the clip bounding box.")
    ap.add_argument("--probe", action="store_true",
                    help="time clip against plain open and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    dask.config.set(scheduler="threads", num_workers=a.workers)
    root = Path(a.root)
    var = a.variable
    vid = a.wrf_var or VARIABLES.get(var, {}).get("WRF")
    if vid is None:
        ap.error(f"no WRF variable_id for '{var}'; pass --wrf-var")
    out_dir = root / var
    out_dir.mkdir(parents=True, exist_ok=True)
    tslice = (a.time_slice[0], a.time_slice[1])

    print(f"variable   : {var} (WRF id '{vid}')")
    print(f"time slice : {tslice[0]} -> {tslice[1]}")
    print(f"products   : {a.products}")
    print(f"method     : {a.method}")
    print(f"destination: {out_dir}\n")

    print("stations:")
    ids, lat, lon, net = station_points(root, var, a.meta_csv)
    if not a.no_domain_filter:
        ids, lat, lon, net = restrict_to_domain(ids, lat, lon, net, root, var)

    if a.probe:
        # Probe AFTER resolving stations, so it measures the real geometry
        # rather than three convenient cities.
        probe(PRODUCTS[a.products[0]], vid, points=list(zip(lat, lon)))
        return 0

    method = a.method
    if method == "auto":
        print("\nprobing to choose a method:")
        method = probe(PRODUCTS[a.products[0]], vid,
                       points=list(zip(lat, lon)))
        print()

    failures = []
    for product in a.products:
        path = out_dir / f"{product}_stations_1hr.zarr"
        if path.exists() and not a.overwrite:
            print(f"[{product}] exists, skipping")
            continue
        print(f"[{product}] fetching hourly at {len(ids)} points via {method}")
        try:
            t0 = time.time()
            if method == "clip":
                ds = by_clip(PRODUCTS[product], vid, tslice, ids, lat,
                             lon, net, a.max_dist_deg)
            else:
                ds = by_sample(PRODUCTS[product], vid, tslice, ids, lat, lon,
                               net, a.max_dist_deg)
            ds = ds.rename({"value": var})

            if a.dry_run:
                gb = sum(ds[v].nbytes for v in ds.data_vars) / 1024 ** 3
                print(f"    would write {path.name}: {dict(ds.sizes)} "
                      f"~{gb:.2f} GB")
                continue

            from dask.diagnostics import ProgressBar
            with ProgressBar(minimum=10.0):
                ds = ds.compute()
            print(f"    computed in {(time.time() - t0) / 60:.1f} min")
            write(ds, path, {
                "source": f"AE {product} hourly '{vid}' at HDP station points",
                "method": method, "time_start": tslice[0], "time_end": tslice[1],
                "note": "hourly at station locations only. The domain-wide "
                        "field was not retained: away from a station it would "
                        "be read once and averaged."}, a.overwrite)
        except Exception as e:
            failures.append(f"{product}: {type(e).__name__}: {e}")
            print(f"[{product}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
        print()

    for f in failures:
        print(f"  FAILED {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())