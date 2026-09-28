"""Exercise the check engine against synthetic stores.

These do not hit S3. They build small in-memory datasets that are deliberately
correct or deliberately broken, and assert the right codes fire.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data_audit import checks
from data_audit import standards as S
from data_audit.findings import Level


def codes(findings):
    return {f.code for f in findings}


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def make_wrf(
    variable_id="t2max",
    units="K",
    n_time=48,
    freq="D",
    start="1980-01-01",
    grid_mapping="Lambert_Conformal",
    conventions="CF-1.7",
    resolution_m=3000.0,
    time=None,
):
    time = pd.date_range(start, periods=n_time, freq=freq) if time is None else time
    n_time = len(time)
    y = np.arange(4) * resolution_m
    x = np.arange(5) * resolution_m
    lat2d = 34.0 + np.outer(np.arange(4), np.ones(5)) * 0.03
    lon2d = -118.0 + np.outer(np.ones(4), np.arange(5)) * 0.03

    data = np.random.default_rng(0).normal(290, 5, size=(n_time, 4, 5))
    array = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": time, "y": y, "x": x},
        attrs={
            "units": units,
            "long_name": "Maximum air temperature at 2m",
            "standard_name": "air_temperature",
            "grid_mapping": grid_mapping,
            "_FillValue": np.float32(1e20),
        },
    )
    dataset = xr.Dataset({variable_id: array})
    dataset = dataset.assign_coords(lat=(("y", "x"), lat2d), lon=(("y", "x"), lon2d))
    dataset["lat"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    dataset["lon"].attrs = {"units": "degrees_east", "long_name": "longitude"}
    dataset["x"].attrs = {"units": "m", "long_name": "x coordinate of projection"}
    dataset["y"].attrs = {"units": "m", "long_name": "y coordinate of projection"}
    dataset["Lambert_Conformal"] = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "lambert_conformal_conic",
            "standard_parallel": [30.0, 60.0],
            "longitude_of_central_meridian": -70.0,
            "latitude_of_projection_origin": 38.0,
        },
    )
    dataset.attrs = {
        "Conventions": conventions,
        "title": "WRF downscaled projection",
        "institution": "UCLA",
        "source": "WRF 4.x",
        "references": "Rahimi 2024",
        "history": "created",
        "comment": "test fixture",
    }
    return dataset


def make_loca2(
    variable_id="tasmax", units="K", n_time=60, start="1950-01-01", time=None
):
    time = pd.date_range(start, periods=n_time, freq="D") if time is None else time
    n_time = len(time)
    lat = 34.0 + np.arange(6) * S.LOCA2_NOMINAL_DEG
    lon = -119.0 + np.arange(7) * S.LOCA2_NOMINAL_DEG
    data = np.random.default_rng(1).normal(295, 4, size=(n_time, 6, 7))
    array = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": units, "long_name": "Maximum air temperature at 2m"},
    )
    dataset = xr.Dataset({variable_id: array})
    dataset["lat"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    dataset["lon"].attrs = {"units": "degrees_east", "long_name": "longitude"}
    dataset["spatial_ref"] = xr.DataArray(
        0, attrs={"crs_wkt": 'GEOGCS["WGS 84",DATUM["WGS_1984"]]'}
    )
    dataset = dataset.set_coords("spatial_ref")
    dataset.attrs = {
        "Conventions": "CF-1.9",
        "title": "LOCA2-Hybrid",
        "institution": "UCSD",
        "source": "LOCA v2",
        "references": "Pierce 2023",
        "history": "created",
        "comment": "test fixture",
    }
    return dataset


WRF_CONTEXT = {
    "activity_id": "WRF",
    "institution_id": "UCLA",
    "source_id": "MIROC6",
    "experiment_id": "historical",
    "table_id": "day",
    "variable_id": "t2max",
    "grid_label": "d03",
}
LOCA2_CONTEXT = {
    "activity_id": "LOCA2",
    "institution_id": "UCSD",
    "source_id": "ACCESS-CM2",
    "experiment_id": "historical",
    "member_id": "r1i1p1f1",
    "table_id": "day",
    "variable_id": "tasmax",
    "grid_label": "d03",
}


# --------------------------------------------------------------------------
# happy paths
# --------------------------------------------------------------------------


# WRF runs September to August, not calendar years; the fixture has to
# match or it trips the bounds check.
FULL_WRF_HISTORICAL = pd.date_range("1980-09-01", "2014-08-31", freq="D")
FULL_LOCA2_HISTORICAL = pd.date_range("1950-01-01", "2014-12-31", freq="D")


def test_clean_wrf_dataset_has_no_errors():
    found = checks.check_dataset(make_wrf(time=FULL_WRF_HISTORICAL), WRF_CONTEXT)
    bad = [f for f in found if f.level >= Level.ERROR]
    assert not bad, [f"{f.code}: {f.message}" for f in bad]
    assert "units.match" in codes(found)
    assert "grid.resolution.match" in codes(found)


def test_clean_loca2_dataset_has_no_errors():
    found = checks.check_dataset(make_loca2(time=FULL_LOCA2_HISTORICAL), LOCA2_CONTEXT)
    bad = [f for f in found if f.level >= Level.ERROR]
    assert not bad, [f"{f.code}: {f.message}" for f in bad]


# --------------------------------------------------------------------------
# variable naming and units
# --------------------------------------------------------------------------


def test_renamed_variable_is_flagged():
    dataset = make_wrf(variable_id="T2MAX")
    found = checks.check_variable_present(dataset, WRF_CONTEXT)
    assert "var.name_mismatch" in codes(found)


def test_wrong_units_are_an_error():
    dataset = make_wrf(units="degC")
    found = checks.check_variable_attrs(dataset["t2max"], "WRF", "day", "t2max")
    assert "units.mismatch" in codes(found)


def test_missing_units_is_an_error():
    dataset = make_wrf()
    del dataset["t2max"].attrs["units"]
    found = checks.check_variable_attrs(dataset["t2max"], "WRF", "day", "t2max")
    assert "attr.var.units.missing" in codes(found)


@pytest.mark.parametrize(
    "expected,actual,verdict",
    [
        ("K", "K", "match"),
        ("m s-1", "m/s", "match"),
        ("W/m2", "W m-2", "match"),
        ("[0 to 100]", "percent", "match"),
        ("mm", "kg m-2", "substitution"),
        ("kg m-2 s-1", "mm", "mismatch"),
        ("K", "degC", "mismatch"),
    ],
)
def test_unit_normalisation(expected, actual, verdict):
    assert S.compare_units(expected, actual) == verdict


# --------------------------------------------------------------------------
# CRS
# --------------------------------------------------------------------------


def test_wrf_without_grid_mapping_is_an_error():
    dataset = make_wrf(grid_mapping=None)
    del dataset["t2max"].attrs["grid_mapping"]
    found = checks.check_crs(dataset, dataset["t2max"], "WRF")
    assert "crs.grid_mapping.missing" in codes(found)


def test_loca2_without_spatial_ref_is_an_error():
    dataset = make_loca2()
    dataset = dataset.drop_vars("spatial_ref")
    found = checks.check_crs(dataset, dataset["tasmax"], "LOCA2")
    assert "crs.spatial_ref.missing" in codes(found)


def test_incomplete_lambert_parameters_warn():
    dataset = make_wrf()
    dataset["Lambert_Conformal"].attrs = {
        "grid_mapping_name": "lambert_conformal_conic"
    }
    found = checks.check_crs(dataset, dataset["t2max"], "WRF")
    assert "crs.lambert.incomplete" in codes(found)


# --------------------------------------------------------------------------
# global attributes
# --------------------------------------------------------------------------


def test_missing_conventions_is_an_error():
    dataset = make_wrf()
    dataset.attrs.pop("Conventions")
    found = checks.check_global_attrs(dataset)
    assert "attr.global.conventions.missing" in codes(found)


def test_sparse_global_attrs_warn():
    dataset = make_wrf()
    dataset.attrs = {"Conventions": "CF-1.7"}
    found = checks.check_global_attrs(dataset)
    assert "attr.global.recommended.missing" in codes(found)


# --------------------------------------------------------------------------
# time axis
# --------------------------------------------------------------------------


def test_wrong_temporal_extent_is_an_error():
    dataset = make_wrf(start="2005-01-01", n_time=30)
    found = checks.check_time(dataset, WRF_CONTEXT)
    assert "time.extent.mismatch" in codes(found)


def test_frequency_mismatch_is_an_error():
    dataset = make_wrf(freq="h", n_time=48, start="1980-01-01")
    found = checks.check_time(dataset, WRF_CONTEXT)  # context says table_id=day
    assert "time.frequency.mismatch" in codes(found)


def test_duplicate_timestamps_are_an_error():
    dataset = make_wrf(n_time=10)
    time = dataset["time"].values.copy()
    time[5] = time[4]
    dataset = dataset.assign_coords(time=time)
    found = checks.check_time(dataset, WRF_CONTEXT)
    assert "time.duplicates" in codes(found)


def test_time_gap_is_flagged():
    dataset = make_wrf(n_time=20)
    time = pd.to_datetime(dataset["time"].values).to_list()
    time = time[:10] + [t + pd.Timedelta(days=30) for t in time[10:]]
    dataset = dataset.assign_coords(time=pd.DatetimeIndex(time))
    found = checks.check_time(dataset, WRF_CONTEXT)
    assert "time.gaps" in codes(found)


def test_leap_day_expectations():
    # MIROC6 is documented as retaining leap days; a 2000 window without Feb 29
    # should be flagged.
    time = pd.date_range("1999-12-01", periods=120, freq="D")
    time = pd.DatetimeIndex([t for t in time if not (t.month == 2 and t.day == 29)])
    dataset = make_wrf(n_time=len(time))
    dataset = dataset.assign_coords(time=time)
    context = dict(WRF_CONTEXT, source_id="MIROC6")
    found = checks._check_leap_days(dataset["time"].values, context, "standard")
    # 'standard' has leap days, so the axis contradicts its own calendar; that
    # is reported instead of the weaker documentation mismatch.
    assert "time.leap.calendar_contradiction" in codes(found)

    # With a no-leap calendar there is no contradiction, only the documented
    # expectation that MIROC6 retains leap days.
    found = checks._check_leap_days(dataset["time"].values, context, "noleap")
    assert "time.leap.missing" in codes(found)


def test_undecoded_time_is_an_error():
    dataset = make_wrf()
    dataset = dataset.assign_coords(time=np.arange(dataset.sizes["time"]))
    found = checks.check_time(dataset, WRF_CONTEXT)
    assert "time.not_decoded" in codes(found)


# --------------------------------------------------------------------------
# spatial
# --------------------------------------------------------------------------


def test_resolution_mismatch_is_an_error():
    dataset = make_wrf(resolution_m=9000.0)  # d02 spacing declared as d03
    found = checks.check_spatial_grid(dataset, "WRF", "d03")
    assert "grid.resolution.mismatch" in codes(found)


def test_out_of_range_coordinates_are_an_error():
    dataset = make_loca2()
    dataset = dataset.assign_coords(lon=dataset["lon"].values + 200)
    found = checks.check_spatial_grid(dataset, "LOCA2", "d03")
    assert "grid.lon.out_of_range" in codes(found)


def test_nan_in_coordinate_is_an_error():
    dataset = make_loca2()
    lat = dataset["lat"].values.astype(float).copy()
    lat[2] = np.nan
    dataset = dataset.assign_coords(lat=lat)
    found = checks.check_coordinate_variables(dataset)
    assert "coord.lat.nan" in codes(found)


# --------------------------------------------------------------------------
# catalog rows
# --------------------------------------------------------------------------


def test_valid_catalog_row_is_clean():
    row = {
        "activity_id": "LOCA2",
        "institution_id": "UCSD",
        "source_id": "ACCESS-CM2",
        "experiment_id": "historical",
        "member_id": "r1i1p1f1",
        "table_id": "day",
        "variable_id": "tasmax",
        "grid_label": "d03",
        "path": "s3://cadcat/loca2/ucsd/access-cm2/historical/r1i1p1f1/day/tasmax/d03/",
    }
    found = checks.check_catalog_row(row)
    bad = [f for f in found if f.level >= Level.ERROR]
    assert not bad, [f"{f.code}: {f.message}" for f in bad]


def test_loca2_on_d01_is_an_error():
    row = {
        "activity_id": "LOCA2",
        "institution_id": "UCSD",
        "source_id": "ACCESS-CM2",
        "experiment_id": "historical",
        "member_id": "r1i1p1f1",
        "table_id": "day",
        "variable_id": "tasmax",
        "grid_label": "d01",
        "path": "s3://cadcat/loca2/ucsd/access-cm2/historical/r1i1p1f1/day/tasmax/d01/",
    }
    assert "catalog.grid_label.invalid" in codes(checks.check_catalog_row(row))


def test_path_not_matching_facets_is_an_error():
    row = {
        "activity_id": "WRF",
        "institution_id": "UCLA",
        "source_id": "CESM2",
        "experiment_id": "historical",
        "member_id": None,
        "table_id": "day",
        "variable_id": "t2max",
        "grid_label": "d03",
        "path": "s3://cadcat/wrf/ucla/cesm2/historical/day/t2min/d03/",  # wrong variable
    }
    # A differing variable segment is reported specifically, not as a generic
    # mismatch, because which segment differs changes what it means.
    assert "catalog.path.variable_mismatch" in codes(checks.check_catalog_row(row))


def test_missing_member_id_for_loca2_is_an_error():
    row = {
        "activity_id": "LOCA2",
        "institution_id": "UCSD",
        "source_id": "ACCESS-CM2",
        "experiment_id": "historical",
        "member_id": None,
        "table_id": "day",
        "variable_id": "tasmax",
        "grid_label": "d03",
        "path": None,
    }
    assert "catalog.member_id.missing" in codes(checks.check_catalog_row(row))


# --------------------------------------------------------------------------
# reference table
# --------------------------------------------------------------------------


def test_variable_table_loads_and_maps_activities():
    table = S.variable_table()
    assert not table.empty
    assert set(table["activity_id"].dropna()) == {"WRF", "LOCA2"}


def test_documented_variables_cover_the_obvious_cases():
    assert "t2max" in S.documented_variables("WRF", "day")
    assert "tasmax" in S.documented_variables("LOCA2", "day")
    assert "prec" in S.documented_variables("WRF", "1hr")
    # derived variables are excluded because they are computed, not stored
    assert not any(v.endswith("_derived") for v in S.documented_variables("WRF", "1hr"))


def test_expected_s3_path_includes_member_only_for_loca2():
    wrf = S.expected_s3_path(
        {
            "activity_id": "WRF",
            "institution_id": "UCLA",
            "source_id": "CESM2",
            "experiment_id": "historical",
            "table_id": "day",
            "variable_id": "t2max",
            "grid_label": "d03",
        }
    )
    assert wrf == "s3://cadcat/wrf/ucla/cesm2/historical/day/t2max/d03/"
    loca = S.expected_s3_path(
        {
            "activity_id": "LOCA2",
            "institution_id": "UCSD",
            "source_id": "ACCESS-CM2",
            "experiment_id": "ssp370",
            "member_id": "r1i1p1f1",
            "table_id": "mon",
            "variable_id": "pr",
            "grid_label": "d03",
        }
    )
    assert loca == "s3://cadcat/loca2/ucsd/access-cm2/ssp370/r1i1p1f1/mon/pr/d03/"


# --------------------------------------------------------------------------
# path namespace conventions (regression cases from the live catalog)
# --------------------------------------------------------------------------


def _wrf_row(**overrides):
    row = {
        "activity_id": "WRF",
        "institution_id": "UCLA",
        "source_id": "CESM2",
        "experiment_id": "historical",
        "member_id": "r11i1p1f1",
        "table_id": "1hr",
        "variable_id": "u10",
        "grid_label": "d01",
        "path": "s3://cadcat/wrf/ucla/cesm2/historical/1hr/u10/d01/",
    }
    row.update(overrides)
    return row


def test_staging_prefix_is_an_error():
    row = _wrf_row(path="s3://cadcat/tmp/wrf/ucla/cesm2/historical/1hr/u10/d01/")
    assert "catalog.path.staging_prefix" in codes(checks.check_catalog_row(row))


def test_earth_relative_wind_qualifier_is_an_error():
    """u10_earth indexed as u10 pools rotated and unrotated winds silently."""
    row = _wrf_row(path="s3://cadcat/wrf/ucla/cesm2/historical/1hr/u10_earth/d01/")
    found = checks.check_catalog_row(row)
    assert "catalog.path.variable_qualifier_dropped" in codes(found)
    message = next(
        f.message for f in found if f.code == "catalog.path.variable_qualifier_dropped"
    )
    assert "earth-relative" in message


def test_staging_and_qualifier_are_reported_separately():
    row = _wrf_row(path="s3://cadcat/tmp/wrf/ucla/cesm2/historical/1hr/u10_earth/d01/")
    found = codes(checks.check_catalog_row(row))
    assert "catalog.path.staging_prefix" in found
    assert "catalog.path.variable_qualifier_dropped" in found
    assert "catalog.path.mismatch" not in found  # fully explained


def test_derived_vars_namespace_is_a_warning_not_an_error():
    row = _wrf_row(
        institution_id="CAE",
        source_id="EC-Earth3",
        member_id="r1i1p1f1",
        variable_id="ffwi",
        grid_label="d03",
        path="s3://cadcat/wrf/derived-vars/ec-earth3/historical/1hr/ffwi/d03/",
    )
    found = checks.check_catalog_row(row)
    assert "catalog.path.pseudo_institution" in codes(found)
    assert "catalog.path.mismatch" not in codes(found)
    path_errors = [
        f
        for f in found
        if f.code.startswith("catalog.path.") and f.level >= Level.ERROR
    ]
    assert not path_errors


def test_unexplained_path_difference_stays_an_error():
    row = _wrf_row(path="s3://cadcat/wrf/ucla/cesm2/ssp999/1hr/u10/d01/")
    assert "catalog.path.mismatch" in codes(checks.check_catalog_row(row))


def test_matching_path_produces_no_path_findings():
    found = codes(checks.check_catalog_row(_wrf_row()))
    assert not {c for c in found if c.startswith("catalog.path.")}


def test_ffwi_is_not_treated_as_read_time_derived():
    assert not S.is_derived("ffwi")
    assert S.is_derived("rh_derived")
    assert S.is_derived("dew_point_derived_hrly")


# --------------------------------------------------------------------------
# wind reference frame (regression cases from live cadcat stores)
# --------------------------------------------------------------------------


def _wind(attrs):
    return xr.DataArray([1.0, 2.0], dims=("time",), attrs=attrs)


def test_earth_relative_wind_is_detected_and_flagged():
    """Attributes copied from s3://cadcat/tmp/wrf/ucla/cesm2/.../u10_earth/d01/"""
    array = _wind(
        {
            "description": "u10 rotated from WRF grid-relative to Earth-relative coordinates",
            "long_name": "Eastward wind component (Earth-relative)",
            "standard_name": "eastward_wind",
            "units": "m s-1",
        }
    )
    found = codes(checks.check_wind_rotation(array, {"variable_id": "u10"}))
    assert "wind.rotation.earth_relative" in found
    # catalog indexes it as bare u10, so it pools with unrotated records
    assert "wind.rotation.hidden_by_catalog" in found


def test_undeclared_wind_frame_is_flagged():
    """Attributes copied from s3://cadcat/wrf/ucla/era5/reanalysis/1hr/u10/d01/"""
    array = _wind({"description": "u at 10 m", "units": "m s-1"})
    found = codes(checks.check_wind_rotation(array, {"variable_id": "u10"}))
    assert "wind.rotation.undeclared" in found
    assert "wind.rotation.earth_relative" not in found


def test_non_wind_variable_is_not_checked_for_rotation():
    array = _wind({"long_name": "Air Temperature at 2m", "units": "K"})
    assert not checks.check_wind_rotation(array, {"variable_id": "t2max"})


def test_lowercase_conventions_value_is_accepted_with_a_note():
    dataset = make_wrf()
    dataset.attrs.pop("Conventions")
    dataset.attrs["conventions"] = "cf-1.7"  # as published by the ERA5 stores
    found = codes(checks.check_global_attrs(dataset))
    assert "attr.global.conventions.not_cf" not in found  # it *is* CF
    assert "attr.global.conventions.case" in found


def test_empty_global_attribute_is_flagged():
    """cadcat ships `bias_correction: ''` and `variant_label: ''` on real stores."""
    dataset = make_wrf()
    dataset.attrs["bias_correction"] = ""
    dataset.attrs["variant_label"] = ""
    found = checks.check_global_attrs(dataset)
    assert "attr.global.empty" in codes(found)
    actual = next(f.actual for f in found if f.code == "attr.global.empty")
    assert "bias_correction" in actual and "variant_label" in actual


# --------------------------------------------------------------------------
# regressions from the first real store sweep
# --------------------------------------------------------------------------


def test_d01_continental_extent_is_plausible():
    """The 45 km parent domain really does span 9.5N..67.3N, 157W..84W.

    A single California-shaped box applied to every domain produced 20 false
    positives on the first live sweep.
    """
    lat_bounds, lon_bounds = S.plausible_extent("d01")
    assert lat_bounds[0] <= 9.476 and lat_bounds[1] >= 67.329
    assert lon_bounds[0] <= -156.823 and lon_bounds[1] >= -84.187


def test_d01_wide_grid_produces_no_range_error():
    dataset = make_loca2()
    lat = np.linspace(9.476, 67.329, dataset.sizes["lat"])
    lon = np.linspace(-156.823, -84.187, dataset.sizes["lon"])
    dataset = dataset.assign_coords(lat=lat, lon=lon)
    found = codes(checks.check_spatial_grid(dataset, "WRF", "d01"))
    assert "grid.lat.out_of_range" not in found
    assert "grid.lon.out_of_range" not in found


def test_genuinely_broken_coordinates_still_error_on_d01():
    dataset = make_loca2()
    dataset = dataset.assign_coords(lon=dataset["lon"].values + 200)  # 0-360 style
    assert "grid.lon.out_of_range" in codes(
        checks.check_spatial_grid(dataset, "WRF", "d01")
    )


def test_description_substitutes_for_long_name_as_a_warning():
    """Older WRF stores carry `description: 'u at 10 m'` and no long_name."""
    array = xr.DataArray(
        [1.0], dims=("time",), attrs={"units": "m s-1", "description": "u at 10 m"}
    )
    found = checks.check_variable_attrs(array, "WRF", "1hr", "u10")
    assert "attr.var.long_name.aliased" in codes(found)
    assert "attr.var.long_name.missing" not in codes(found)


def test_no_descriptor_at_all_is_still_an_error():
    array = xr.DataArray([1.0], dims=("time",), attrs={"units": "m s-1"})
    assert "attr.var.long_name.missing" in codes(
        checks.check_variable_attrs(array, "WRF", "1hr", "u10")
    )


def test_reanalysis_extent_matches_published_stores():
    assert S.EXPECTED_TIME_RANGE[("WRF", "reanalysis")] == (1980, 2020)


def test_wrf_regridded_to_latlon_is_accepted():
    """UCSD daily/monthly WRF products are geographic, not Lambert-projected."""
    dataset = make_loca2(variable_id="u10")  # (time, lat, lon)
    found = codes(checks.check_dims(dataset, "u10", "WRF"))
    assert "dims.missing" not in found
    assert "grid.convention" in found
    assert "coords.wrf_aux.missing" not in found  # not curvilinear


def test_wrf_projected_store_still_requires_lambert():
    dataset = make_wrf()
    del dataset["t2max"].attrs["grid_mapping"]
    assert "crs.grid_mapping.missing" in codes(
        checks.check_crs(dataset, dataset["t2max"], "WRF")
    )


def test_wrf_geographic_store_is_not_asked_for_lambert():
    dataset = make_loca2(variable_id="u10")
    found = codes(checks.check_crs(dataset, dataset["u10"], "WRF"))
    assert "crs.grid_mapping.missing" not in found


def test_wrf_geographic_store_without_any_crs_is_flagged():
    dataset = make_loca2(variable_id="u10").drop_vars("spatial_ref")
    assert "crs.geographic.undeclared" in codes(
        checks.check_crs(dataset, dataset["u10"], "WRF")
    )


def test_store_missing_all_spatial_dims_still_errors():
    dataset = xr.Dataset({"u10": xr.DataArray([1.0, 2.0], dims=("time",))})
    assert "dims.missing" in codes(checks.check_dims(dataset, "u10", "WRF"))


# --------------------------------------------------------------------------
# data presence (a cadcat daily store is entirely NaN)
# --------------------------------------------------------------------------


def test_all_nan_variable_is_an_error():
    dataset = make_loca2(variable_id="u10", n_time=40)
    dataset["u10"].values[:] = np.nan
    found = checks.check_data_presence(dataset, "u10")
    assert "data.all_missing" in codes(found)


def test_populated_variable_passes():
    dataset = make_loca2(variable_id="u10", n_time=40)
    found = checks.check_data_presence(dataset, "u10")
    assert "data.coverage" in codes(found)
    assert "data.all_missing" not in codes(found)


def test_partially_masked_field_is_reported_as_coverage():
    dataset = make_loca2(variable_id="u10", n_time=40)
    dataset["u10"].values[:, :5, :] = np.nan  # ~17% finite, a plausible mask
    found = codes(checks.check_data_presence(dataset, "u10"))
    assert "data.coverage" in found
    assert "data.sparse" not in found


def test_coverage_that_varies_across_time_is_flagged():
    dataset = make_loca2(variable_id="u10", n_time=40)
    dataset["u10"].values[0, :, :] = np.nan  # first step empty, others full
    assert "data.coverage_varies" in codes(checks.check_data_presence(dataset, "u10"))


def test_probe_is_off_by_default():
    dataset = make_loca2(variable_id="u10", n_time=40)
    dataset["u10"].values[:] = np.nan
    context = dict(LOCA2_CONTEXT, variable_id="u10")
    assert "data.all_missing" not in codes(checks.check_dataset(dataset, context))
    assert "data.all_missing" in codes(
        checks.check_dataset(dataset, context, probe_values=True)
    )


def test_specific_humidity_units_are_equivalent():
    """CF gives huss as '1'; the reference table says 'kg/kg'. Same quantity."""
    assert S.compare_units("kg/kg", "1") == "match"
    assert S.compare_units("kg kg-1", "1") == "match"


@pytest.mark.parametrize(
    "expected,actual",
    [("W m-2", "W/m2"), ("kg m-2 s-1", "kg/m2/s"), ("m s-1", "m/s")],
)
def test_udunits_compound_spellings_parse(expected, actual):
    assert S.compare_units(expected, actual) == "match"


def test_gregorian_calendar_without_feb29_is_an_error():
    """LOCA2 daily stores declare proleptic_gregorian yet omit 29 February."""
    time = pd.date_range("1950-01-01", "1960-12-31", freq="D")
    time = pd.DatetimeIndex([t for t in time if not (t.month == 2 and t.day == 29)])
    context = {"activity_id": "LOCA2", "source_id": "EC-Earth3", "table_id": "day"}
    found = checks._check_leap_days(time.values, context, "proleptic_gregorian")
    assert "time.leap.calendar_contradiction" in codes(found)


def test_noleap_calendar_without_feb29_is_not_contradictory():
    time = pd.date_range("1950-01-01", "1960-12-31", freq="D")
    time = pd.DatetimeIndex([t for t in time if not (t.month == 2 and t.day == 29)])
    context = {"activity_id": "WRF", "source_id": "CESM2", "table_id": "day"}
    found = codes(checks._check_leap_days(time.values, context, "noleap"))
    assert "time.leap.calendar_contradiction" not in found


def test_masked_domain_coverage_is_informational():
    """LOCA2 over California is ~31% finite inside its lat/lon rectangle."""
    dataset = make_loca2(variable_id="tasmax", n_time=40)
    dataset["tasmax"].values[:, :4, :] = np.nan  # ~33% finite on a 6-row grid
    found = codes(checks.check_data_presence(dataset, "tasmax"))
    assert "data.coverage" in found
    assert "data.sparse" not in found


def test_nearly_empty_field_is_still_a_warning():
    dataset = make_loca2(variable_id="tasmax", n_time=40)
    dataset["tasmax"].values[:] = np.nan
    dataset["tasmax"].values[:, 0, 0] = 1.0  # 1 of 42 cells
    assert "data.sparse" in codes(checks.check_data_presence(dataset, "tasmax"))


# --------------------------------------------------------------------------
# aggregated results (ClimateData has no member_id setter, so LOCA2 queries
# match every ensemble member and intake-esm combines them)
# --------------------------------------------------------------------------


def _aggregated(n_members=3, drop_feb29=True):
    time = pd.date_range("1950-01-01", "1960-12-31", freq="D")
    if drop_feb29:
        time = pd.DatetimeIndex([t for t in time if not (t.month == 2 and t.day == 29)])
    data = np.zeros((n_members, len(time), 2, 2))
    ds = xr.Dataset(
        {"tasmax": (("member_id", "time", "lat", "lon"), data)},
        coords={
            "member_id": [f"r{i}i1p1f1" for i in range(1, n_members + 1)],
            "time": time,
            "lat": [34.0, 34.1],
            "lon": [-119.0, -118.9],
        },
    )
    ds["tasmax"].attrs = {"units": "K", "long_name": "Maximum air temperature at 2m"}
    return ds


def test_aggregated_dataset_is_detected():
    ds = _aggregated()
    assert checks.aggregation_dims(ds) == {"member_id": 3}
    assert "dataset.aggregated" in codes(checks.check_aggregation(ds, {}))


def test_single_member_result_is_not_flagged():
    ds = _aggregated(n_members=1)
    assert not checks.aggregation_dims(ds)
    assert not checks.check_aggregation(ds, {})


def test_leap_findings_are_demoted_on_an_aggregated_axis():
    """Inner-join alignment across mixed calendars drops Feb 29; that is not a
    defect in any store, so it must not be reported as one."""
    ds = _aggregated()
    context = {"activity_id": "LOCA2", "source_id": "EC-Earth3", "table_id": "day"}

    single = checks._check_leap_days(ds.time.values, context, "proleptic_gregorian")
    assert max(f.level for f in single) >= Level.ERROR

    agg = checks._check_leap_days(
        ds.time.values, context, "proleptic_gregorian", aggregated=True
    )
    assert agg  # still reported
    assert max(f.level for f in agg) <= Level.INFO  # but not as a data defect


def test_gaps_are_demoted_on_an_aggregated_axis():
    ds = _aggregated()
    context = {"activity_id": "LOCA2", "table_id": "day", "experiment_id": "historical"}
    gaps = [
        f
        for f in checks.check_time(ds, context, aggregated=True)
        if f.code == "time.gaps"
    ]
    assert gaps and all(f.level <= Level.INFO for f in gaps)


# --------------------------------------------------------------------------
# structural authority: only the raw-store reader can judge a time axis
# --------------------------------------------------------------------------


def test_backends_declare_structural_authority():
    from data_audit.backend_climakitae import ClimakitaeBackend
    from data_audit.backend_zarr import ZarrBackend

    assert ZarrBackend.authoritative_structure is True
    assert ClimakitaeBackend.authoritative_structure is False


def test_time_findings_demoted_for_a_non_authoritative_reader():
    """HadGEM3-GC31-LL/ssp245 wind stores are complete on disk (n=31411,
    feb29=21) yet climakitae returned an axis missing 29 February. Structural
    findings from that reader must not be reported as archive defects."""
    time = pd.date_range("2015-01-01", "2024-12-31", freq="D")
    time = pd.DatetimeIndex([t for t in time if not (t.month == 2 and t.day == 29)])
    dataset = make_loca2(variable_id="uas", time=time)
    context = dict(LOCA2_CONTEXT, variable_id="uas", experiment_id="ssp245")

    strict = checks.check_dataset(dataset, context, authoritative_structure=True)
    assert any(
        f.code == "time.leap.calendar_contradiction" and f.level >= Level.ERROR
        for f in strict
    )

    lenient = checks.check_dataset(dataset, context, authoritative_structure=False)
    structural = [
        f
        for f in lenient
        if f.code in {"time.leap.calendar_contradiction", "time.gaps"}
    ]
    assert structural
    assert all(f.level <= Level.INFO for f in structural)


def test_content_findings_are_not_demoted_for_a_non_authoritative_reader():
    """Only *structure* is the reader's doing. Missing units stay an error."""
    dataset = make_loca2(variable_id="uas")
    del dataset["uas"].attrs["units"]
    context = dict(LOCA2_CONTEXT, variable_id="uas")
    found = checks.check_dataset(dataset, context, authoritative_structure=False)
    assert any(
        f.code == "attr.var.units.missing" and f.level >= Level.ERROR for f in found
    )


# --------------------------------------------------------------------------
# reconciliation: documentation vs store vs delivered product
# --------------------------------------------------------------------------


class _FakeBackend:
    """Returns a prepared dataset, standing in for a real reader."""

    def __init__(self, name, dataset, authoritative_structure=True):
        self.name = name
        self._dataset = dataset
        self.authoritative_structure = authoritative_structure

    def open(self, row):
        return self._dataset, {"backend": self.name}


def _reconcile_row():
    return {
        "activity_id": "LOCA2",
        "institution_id": "UCSD",
        "source_id": "ACCESS-CM2",
        "experiment_id": "historical",
        "member_id": "r1i1p1f1",
        "table_id": "day",
        "variable_id": "tasmax",
        "grid_label": "d03",
        "path": None,
    }


def test_store_defect_repaired_by_reader_is_classified():
    """LOCA2 stores lack spatial_ref; climakitae attaches a CRS on read."""
    from data_audit import reconcile as R  # the module

    store = make_loca2().drop_vars("spatial_ref")  # defective artifact
    delivered = make_loca2()  # repaired on read
    plan = pd.DataFrame([_reconcile_row()])

    table = R.reconcile(
        plan,
        _FakeBackend("zarr", store),
        _FakeBackend("climakitae", delivered, authoritative_structure=False),
        check_time_axis=False,
        max_workers=1,
    )
    row = table[table.code == "crs.spatial_ref.missing"]
    assert len(row) == 1
    assert row.verdict.iloc[0] == "store_defect_repaired"
    assert bool(row.in_store.iloc[0]) and not bool(row.in_delivered.iloc[0])
    # still reported, not dropped just because users cannot see it
    assert row.level.iloc[0] == "WARN"


def test_reader_introduced_defect_is_an_error():
    """A clean store whose delivered form loses units affects every user."""
    from data_audit import reconcile as R  # the module

    store = make_loca2()
    delivered = make_loca2()
    del delivered["tasmax"].attrs["units"]
    plan = pd.DataFrame([_reconcile_row()])

    table = R.reconcile(
        plan,
        _FakeBackend("zarr", store),
        _FakeBackend("climakitae", delivered, authoritative_structure=False),
        check_time_axis=False,
        max_workers=1,
    )
    row = table[table.code == "attr.var.units.missing"]
    assert len(row) == 1
    assert row.verdict.iloc[0] == "introduced_by_reader"
    assert row.level.iloc[0] == "ERROR"


def test_defect_on_both_surfaces_is_attributed_to_the_source():
    from data_audit import reconcile as R  # the module

    store = make_loca2()
    del store["tasmax"].attrs["units"]
    delivered = make_loca2()
    del delivered["tasmax"].attrs["units"]
    plan = pd.DataFrame([_reconcile_row()])

    table = R.reconcile(
        plan,
        _FakeBackend("zarr", store),
        _FakeBackend("climakitae", delivered, authoritative_structure=False),
        check_time_axis=False,
        max_workers=1,
    )
    row = table[table.code == "attr.var.units.missing"]
    assert row.verdict.iloc[0] == "both"


def test_every_finding_carries_a_documentation_requirement():
    from data_audit import reconcile as R  # the module

    for code in (
        "attr.var.units.missing",
        "attr.global.conventions.missing",
        "crs.spatial_ref.missing",
        "time.extent.mismatch",
        "units.mismatch",
        "grid.resolution.mismatch",
        "store.not_consolidated",
        "wind.rotation.undeclared",
        "data.all_missing",
        "var.name_mismatch",
    ):
        requirement, doc, url = R.requirement_for(code)
        assert requirement and doc and url.startswith("http"), code


def test_markdown_report_names_both_surfaces():
    from data_audit import reconcile as R  # the module

    store = make_loca2().drop_vars("spatial_ref")
    delivered = make_loca2()
    plan = pd.DataFrame([_reconcile_row()])
    table = R.reconcile(
        plan,
        _FakeBackend("zarr", store),
        _FakeBackend("climakitae", delivered, authoritative_structure=False),
        check_time_axis=False,
        max_workers=1,
    )
    text = R.render_markdown(table)
    assert "Repaired by climakitae" in text
    assert "s3://cadcat" in text
    assert "Metadata Standards" in text


def test_reader_introduced_time_defect_reaches_the_reconciliation():
    """A store with a complete axis, delivered with 29 February missing.

    In a single-backend sweep this is demoted to INFO so the archive is not
    blamed. The reconciliation must still see it at full severity, because the
    delivered product is what users get.
    """
    from data_audit import reconcile as R

    full = pd.date_range("2015-01-01", "2024-12-31", freq="D")
    gappy = pd.DatetimeIndex([t for t in full if not (t.month == 2 and t.day == 29)])

    store = make_loca2(variable_id="tasmax", time=full)
    delivered = make_loca2(variable_id="tasmax", time=gappy)
    plan = pd.DataFrame([dict(_reconcile_row(), experiment_id="ssp245")])

    table = R.reconcile(
        plan,
        _FakeBackend("zarr", store),
        _FakeBackend("climakitae", delivered, authoritative_structure=False),
        max_workers=1,
    )
    leap = table[table.code == "time.leap.calendar_contradiction"]
    assert len(leap) == 1, "reader-introduced time defect must not be filtered out"
    assert leap.verdict.iloc[0] == "introduced_by_reader"
    assert leap.level.iloc[0] == "ERROR"


def test_checks_passing_on_both_surfaces_are_recorded_as_compliant():
    from data_audit import reconcile as R

    dataset = make_loca2(time=FULL_LOCA2_HISTORICAL)
    plan = pd.DataFrame([_reconcile_row()])
    table = R.reconcile(
        plan,
        _FakeBackend("zarr", dataset),
        _FakeBackend("climakitae", dataset, authoritative_structure=False),
        max_workers=1,
    )
    compliant = table[table.verdict == "compliant"]
    assert not compliant.empty, "compliant verdict must be reachable"
    assert "units.match" in set(compliant.code)
    assert (compliant.level == "OK").all()


# --------------------------------------------------------------------------
# unwritten chunks and calendar convention (cadcat daily wind product)
# --------------------------------------------------------------------------


def _chunked(n_time=32, chunk=8, empty_chunks=()):
    """A store whose time axis is chunked, with some chunks left unwritten."""
    time = pd.date_range("2015-01-01", periods=n_time, freq="D")
    data = np.ones((n_time, 3, 3))
    for c in empty_chunks:
        data[c * chunk : (c + 1) * chunk, :, :] = np.nan
    ds = xr.Dataset(
        {"u10": (("time", "lat", "lon"), data)},
        coords={
            "time": time,
            "lat": [34.0, 34.1, 34.2],
            "lon": [-119.0, -118.9, -118.8],
        },
    )
    ds["u10"].encoding["chunks"] = (chunk, 3, 3)
    ds["u10"].attrs = {"units": "m s-1", "long_name": "Eastward wind"}
    return ds


def test_unwritten_chunks_are_an_error():
    """Chunks 0 and 1 absent is what the cadcat wind stores actually show."""
    ds = _chunked(empty_chunks=(0, 1))
    found = checks.check_unwritten_chunks(ds, "u10")
    assert "data.unwritten_chunks" in codes(found)
    assert max(f.level for f in found) >= Level.ERROR


def test_complete_store_passes():
    assert "data.complete" in codes(checks.check_unwritten_chunks(_chunked(), "u10"))


def test_gaps_not_on_chunk_edges_are_reported_differently():
    """A gap that is not chunk-aligned is missing data, not a failed write."""
    ds = _chunked()
    ds["u10"].values[5:9, :, :] = np.nan  # straddles a chunk edge
    found = codes(checks.check_unwritten_chunks(ds, "u10"))
    assert "data.gaps" in found
    assert "data.unwritten_chunks" not in found


def test_calendar_year_axis_on_wrf_is_an_error():
    """WRF runs Sep-Aug; the daily wind stores carry calendar years instead.

    Both round to "2015-2100", so the year-granularity check passed them.
    """
    time = pd.date_range("2015-01-01", "2099-12-31", freq="D")
    ds = make_wrf(time=time)
    context = dict(WRF_CONTEXT, experiment_id="ssp370")
    found = checks.check_time(ds, context)
    assert "time.bounds.mismatch" in codes(found)


def test_september_august_axis_on_wrf_passes():
    time = pd.date_range("2014-09-01", "2100-08-31", freq="D")
    ds = make_wrf(time=time)
    context = dict(WRF_CONTEXT, experiment_id="ssp370")
    found = codes(checks.check_time(ds, context))
    assert "time.bounds.match" in found
    assert "time.bounds.mismatch" not in found
