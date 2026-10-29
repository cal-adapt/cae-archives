"""Offline tests for helpers/event_helpers.py on small synthetic data (no network).

Run from guidance-website/:
    conda run -n climakitae pytest relevant-hazard-template/tests
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "helpers"))
import event_helpers as eh  # noqa: E402


def series(values, start="2000-01-01", dim="time"):
    """1-D daily DataArray."""
    return xr.DataArray(np.asarray(values, dtype=float), dims=dim,
                        coords={dim: pd.date_range(start, periods=len(values), freq="D")})


# ---------------------------------------------------------------------------
# count_events
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("min_length, expected", [(1, 4), (2, 3), (3, 2), (5, 0)])
def test_count_events_runs(min_length, expected):
    #           run of 3   run of 2   run of 4    run of 1
    mask = [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    da = xr.DataArray(np.array(mask, bool), dims="t")
    assert int(eh.count_events(da, "t", min_length)) == expected


def test_count_events_run_at_start_and_end():
    da = xr.DataArray(np.array([1, 1, 1, 0, 0, 1, 1, 1], bool), dims="t")
    assert int(eh.count_events(da, "t", 3)) == 2


def test_count_events_nan_is_not_an_event():
    da = xr.DataArray(np.array([np.nan, 1, np.nan, 1, 1]), dims="t")
    assert int(eh.count_events(da, "t", 1)) == 2


def test_count_events_gridded_keeps_other_dims():
    data = np.zeros((2, 6), bool)
    data[0, [1, 2, 4]] = True   # 2 events in cell 0
    data[1, :] = True           # 1 long event in cell 1
    da = xr.DataArray(data, dims=("cell", "t"))
    assert eh.count_events(da, "t").values.tolist() == [2, 1]


# ---------------------------------------------------------------------------
# precip helpers
# ---------------------------------------------------------------------------
def test_hdp_daily_precip_in_converts_and_keeps_missing_days():
    hours = pd.date_range("2021-10-24", periods=72, freq="h")
    pr = np.full(72, np.nan)
    pr[:24] = 25.4 / 24          # day 1: exactly 1 inch in total
    pr[24:48] = 0.0              # day 2: reported, dry
    station = xr.Dataset({"pr": ("time", pr)}, coords={"time": hours})   # day 3: no reports
    daily = eh.hdp_daily_precip_in(station)
    assert daily.values[0] == pytest.approx(1.0)
    assert daily.values[1] == 0.0
    assert np.isnan(daily.values[2])


def test_nday_totals_labels_window_end():
    totals = eh.nday_totals(series([1, 2, 3, 4]), 3, "time")
    assert np.isnan(totals.values[:2]).all()
    assert totals.values[2:].tolist() == [6, 9]


def test_wet_day_nday_threshold_uses_only_windows_ending_on_wet_days():
    # 3-day totals ending on days 3..6: [3, 3, 3, 30]; day 5 is dry, so its
    # window (total 3) is excluded -> wet-ending totals [3, 3, 30].
    daily = series([1, 1, 1, 1, 0, 29])
    thr = eh.wet_day_nday_threshold(daily, 3, 0.5, 0.01, "time")
    assert float(thr) == pytest.approx(3.0)
    thr_max = eh.wet_day_nday_threshold(daily, 3, 1.0, 0.01, "time")
    assert float(thr_max) == pytest.approx(30.0)


def test_wet_day_nday_threshold_gridded_shape():
    data = np.random.default_rng(0).gamma(0.5, 0.2, size=(2, 40, 3, 4))
    da = xr.DataArray(data, dims=("sim", "time_delta", "y", "x"))
    thr = eh.wet_day_nday_threshold(da, 3, 0.99, 0.01, "time_delta")
    assert thr.dims == ("sim", "y", "x")


def test_nday_events_counts_a_multiday_storm_once():
    # One 4-day storm: several overlapping 3-day windows exceed 5 in -> 1 event.
    daily = series([0, 0, 3, 3, 3, 3, 0, 0, 0, 6, 0, 0])
    assert int(eh.nday_events(daily, 3, 5.0, "time")) == 2   # the storm + the 6-in day


def test_nday_events_all_nan_cell_is_nan_not_zero():
    # Cell 0 has data; cell 1 is outside the clip area (all NaN) -> NaN, not 0 events.
    data = np.array([[0, 6, 0, 0, 0], [np.nan] * 5])
    da = xr.DataArray(data, dims=("cell", "time"))
    out = eh.nday_events(da, 1, 5.0, "time")
    assert out.values[0] == 1
    assert np.isnan(out.values[1])
    assert float(out.mean()) == 1.0   # the empty cell doesn't drag the area mean down


def test_water_year():
    t = xr.DataArray(pd.to_datetime(["2021-09-30", "2021-10-01", "2022-01-15"]), dims="time")
    assert eh.water_year(t).values.tolist() == [2021, 2022, 2022]


# ---------------------------------------------------------------------------
# config + simulations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("level, slug", [(0.8, "gwl08"), (1.5, "gwl15"), (2.0, "gwl2"), (3.0, "gwl3")])
def test_gwl_slug(level, slug):
    assert eh.gwl_slug(level) == slug


def minimal_config(**projection):
    proj = {"warming_levels": [0.8, 1.5], "warming_level_window": 15,
            "baseline_warming_level": 0.8, **projection}
    return {
        "hazard": "precip", "event_name": "test",
        "location": {"station_label": "KSAC", "place_name": "Some Place", "clip": "x", "zip_shapefile": "z.zip"},
        "event": {"year": 2021, "windows": [["2021-10-23", "2021-10-26"]]},
        "thresholds": {"wet_day_in": 0.01},
        "projection": proj,
    }


def test_event_config_percentile_optional_and_paths(tmp_path):
    cfg = eh.EventConfig(minimal_config(), str(tmp_path / "event_config.yaml"))
    assert cfg.percentile is None
    assert cfg.gwl_order == ["gwl08", "gwl15"]
    assert cfg.place_slug == "some_place"
    assert cfg.zip_shapefile == str(tmp_path / "z.zip")
    assert cfg.tract_metric_csv("3day_p99_events", "gwl08").endswith(
        os.path.join("crai_layers", "tract_metrics", "precip_3day_p99_events_some_place_gwl08.csv"))


def test_event_config_analysis_layout(tmp_path):
    # <post>/_analysis/event_config.yaml: figures next to index.qmd, the rest in _analysis/
    analysis = tmp_path / "my-post" / "_analysis"
    cfg = eh.EventConfig(minimal_config(), str(analysis / "event_config.yaml"))
    assert cfg.post_dir == str(tmp_path / "my-post")
    assert cfg.zip_shapefile == str(analysis / "z.zip")
    png, html = cfg.figure_paths("x")
    assert png == str(tmp_path / "my-post" / "figures" / "static" / "fig_x.png")
    assert html == str(tmp_path / "my-post" / "figures" / "html" / "fig_x.html")
    assert cfg.tract_metric_csv("m", "gwl08").startswith(str(analysis / "crai_layers"))


def test_event_config_station_defaults(tmp_path):
    cfg = eh.EventConfig(minimal_config(), str(tmp_path / "event_config.yaml"))
    assert cfg.network_id == "ASOSAWOS"
    raw = minimal_config()
    raw["location"].update(station_label=None, network_id="OtherISD")
    cfg = eh.EventConfig(raw, str(tmp_path / "event_config.yaml"))
    assert cfg.station_label is None and cfg.network_id == "OtherISD"


def test_isd_candidates_by_icao_and_area():
    from shapely.geometry import box
    isd = pd.DataFrame({
        "number": ["72483993225", "72483999999", "72483023232", "11111111111"],
        "ICAO": ["KSMF", "KSMF", "KSAC", np.nan],
        "LAT": [38.70, 38.68, 38.51, 40.0], "LON": [-121.60, -121.58, -121.50, -120.0],
    })
    hdp = pd.DataFrame({
        "network_id": ["OtherISD", "OtherISD", "ASOSAWOS", "OtherISD"],
        "station_id": ["OtherISD_72483993225", "OtherISD_72483999999",
                       "ASOSAWOS_72483023232", "OtherISD_11111111111"],
    })
    assert eh.isd_candidates(isd, hdp, "OtherISD", icao="ksmf") == [
        "OtherISD_72483993225", "OtherISD_72483999999"]
    assert eh.isd_candidates(isd, hdp, "OtherISD", icao="KSAC") == []      # KSAC is ASOSAWOS only
    sac = box(-122.0, 38.0, -121.0, 39.0)
    assert eh.isd_candidates(isd, hdp, "OtherISD", area=sac) == [
        "OtherISD_72483993225", "OtherISD_72483999999"]
    assert eh.isd_candidates(isd, hdp, "ASOSAWOS", area=sac) == ["ASOSAWOS_72483023232"]


def test_event_config_missing_key_raises(tmp_path):
    raw = minimal_config()
    del raw["location"]["clip"]
    with pytest.raises(KeyError, match="location.clip"):
        eh.EventConfig(raw, str(tmp_path / "event_config.yaml"))


def sims_array(names):
    return xr.DataArray(np.zeros(len(names)), dims="sim", coords={"sim": names})


def test_filter_simulations_drops_matches():
    da = sims_array(["WRF_UCLA_EC-Earth3_x", "WRF_UCLA_EC-Earth3-Veg_x", "WRF_UCLA_MIROC6_x"])
    out = eh.filter_simulations(da, ["EC-Earth3-Veg"])
    assert list(out.sim.values) == ["WRF_UCLA_EC-Earth3_x", "WRF_UCLA_MIROC6_x"]


def test_filter_simulations_raises_when_nothing_matches():
    with pytest.raises(ValueError, match="matched none"):
        eh.filter_simulations(sims_array(["WRF_UCLA_MIROC6_x"]), ["EC-Earth3-Veg"])


def test_filter_simulations_renames_simulation_dim():
    da = xr.DataArray(np.zeros(2), dims="simulation", coords={"simulation": ["a", "b"]})
    assert "sim" in eh.filter_simulations(da, []).dims


# ---------------------------------------------------------------------------
# tracts
# ---------------------------------------------------------------------------
def test_tracts_inside_area_drops_edge_tracts():
    import geopandas as gpd
    from shapely.geometry import box
    zips = gpd.GeoDataFrame(geometry=[box(-121.5, 38.5, -121.4, 38.6)], crs="EPSG:4326")
    tracts = gpd.GeoDataFrame({"GEOID": ["inside", "edge", "outside"]}, crs="EPSG:4326", geometry=[
        box(-121.48, 38.52, -121.42, 38.58),   # fully inside
        box(-121.42, 38.52, -121.30, 38.58),   # touches the area, centroid outside
        box(-121.0, 38.0, -120.9, 38.1),       # far away
    ])
    assert eh.tracts_inside_area(tracts, zips)["GEOID"].tolist() == ["inside"]
