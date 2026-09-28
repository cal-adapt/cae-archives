"""
Machine-readable version of the Cal-Adapt data standards.

Everything a check compares against lives here, so that when the documentation
changes you edit one module rather than hunting through check code.

Sources
-------
- Climate Model Simulations
  https://analytics.cal-adapt.org/data-tools/data-documentation/climate-model-sims.html
- Data Structure and Format
  https://analytics.cal-adapt.org/data-tools/data-documentation/data-structure-and-format.html
- Metadata Standards
  https://analytics.cal-adapt.org/data-tools/data-documentation/metadata-standards.html
- ``climakitae/data/variable_descriptions.csv`` (the shipped variable table that
  the Analytics Engine itself treats as authoritative for names and units)
"""

from __future__ import annotations

import functools
import re

import pandas as pd

# --------------------------------------------------------------------------
# Catalog vocabulary
# --------------------------------------------------------------------------

CADCAT_CATALOG = "cadcat"
DATA_CATALOG_URL = "https://cadcat.s3.amazonaws.com/cae-collection.json"
S3_BUCKET = "cadcat"

#: intake-esm facet columns, in hierarchy order (see "Core Concepts").
FACETS: tuple[str, ...] = (
    "activity_id",
    "institution_id",
    "source_id",
    "experiment_id",
    "member_id",
    "table_id",
    "variable_id",
    "grid_label",
)

#: Facets that actually change the *metadata* of a store. Two datasets that
#: differ only by source_id/member_id should be metadata-identical, so a QC
#: sweep samples one dataset per unique combination of these.
METADATA_KEY_FACETS: tuple[str, ...] = (
    "activity_id",
    "institution_id",
    "table_id",
    "grid_label",
    "variable_id",
)

ACTIVITIES = ("WRF", "LOCA2")  # Can add more later

#: Which grid labels each downscaling method is documented to publish.
#: WRF: 45 km (d01), 9 km WECC (d02), 3 km CA (d03). LOCA2-Hybrid: 3 km CA only.
VALID_GRID_LABELS: dict[str, frozenset[str]] = {
    "WRF": frozenset({"d01", "d02", "d03"}),
    "LOCA2": frozenset({"d03"}),
}

#: Documented temporal resolutions. WRF is natively hourly with pre-aggregated
#: day/mon; LOCA2 is natively daily with pre-aggregated mon plus a `yrmax`
#: annual-maximum product.
VALID_TABLE_IDS: dict[str, frozenset[str]] = {
    "WRF": frozenset({"1hr", "day", "mon"}),
    "LOCA2": frozenset({"day", "mon", "yrmax"}),
}

VALID_EXPERIMENT_IDS: dict[str, frozenset[str]] = {
    "WRF": frozenset({"historical", "reanalysis", "ssp245", "ssp370", "ssp585"}),
    "LOCA2": frozenset({"historical", "ssp245", "ssp370", "ssp585"}),
}

#: Documented producing institutions.
EXPECTED_INSTITUTIONS: dict[str, frozenset[str]] = {
    "WRF": frozenset({"UCLA", "ERA", "CAE", "UCSD"}),
    "LOCA2": frozenset({"UCSD"}),
}

#: GCMs listed in the documentation (Climate Model Simulations, Table 1).
#: Note the doc prose writes "HadGEM-GC31-LL" and "MIROC" while the catalog uses
#: the CMIP6 registered names below; the doc list is a known typo, not a data
#: defect, so DOCUMENTED_MODELS holds the catalog-correct spellings.
DOCUMENTED_MODELS: dict[str, frozenset[str]] = {
    "WRF": frozenset(
        {
            "CESM2",
            "CNRM-ESM2-1",
            "EC-Earth3",
            "EC-Earth3-Veg",
            "FGOALS-g3",
            "MIROC6",
            "MPI-ESM1-2-HR",
            "TaiESM1",
            "ERA5",  # reanalysis driver
            "ensmean",  # derived ensemble mean product
        }
    ),
    "LOCA2": frozenset(
        {
            "ACCESS-CM2",
            "CESM2-LENS",
            "CNRM-ESM2-1",
            "EC-Earth3",
            "EC-Earth3-Veg",
            "FGOALS-g3",
            "GFDL-ESM4",
            "HadGEM3-GC31-LL",
            "INM-CM5-0",
            "IPSL-CM6A-LR",
            "KACE-1-0-G",
            "MIROC6",
            "MPI-ESM1-2-HR",
            "MRI-ESM2-0",
            "TaiESM1",
        }
    ),
}

MEMBER_ID_RE = re.compile(r"^r\d+i\d+p\d+f\d+$")

# --------------------------------------------------------------------------
# Temporal expectations
# --------------------------------------------------------------------------

#: (first_year, last_year) each experiment is documented to cover.
#: "The WRF simulations have a historical period of 1980-2014, and the LOCA2
#: historical simulations extend from 1950-2014."
EXPECTED_TIME_RANGE: dict[tuple[str, str], tuple[int, int]] = {
    ("WRF", "historical"): (1980, 2014),
    ("WRF", "ssp245"): (2015, 2100),
    ("WRF", "ssp370"): (2015, 2100),
    ("WRF", "ssp585"): (2015, 2100),
    # Verified against published stores, which end in 2020 rather than the
    # 2022 this table originally guessed at.
    ("WRF", "reanalysis"): (1980, 2020),
    ("LOCA2", "historical"): (1950, 2014),
    ("LOCA2", "ssp245"): (2015, 2100),
    ("LOCA2", "ssp370"): (2015, 2100),
    ("LOCA2", "ssp585"): (2015, 2100),
}

#: Exact first and last timestamp each experiment is documented to cover.
#:
#: Year granularity is not enough. The cadcat daily wind stores carry a
#: calendar-year axis (2015-01-01 to 2099-12-31) where every other daily
#: variable uses the simulation's September-August convention (2014-09-01 to
#: 2100-08-31). Both round to "2015-2100", so a year-level check passes a store
#: that is four months and 344 days wrong.
#:
#: WRF runs September to August; LOCA2 runs on calendar years.
EXPECTED_TIME_BOUNDS: dict[tuple[str, str], tuple[str, str]] = {
    ("WRF", "historical"): ("1980-09-01", "2014-08-31"),
    ("WRF", "ssp245"): ("2014-09-01", "2100-08-31"),
    ("WRF", "ssp370"): ("2014-09-01", "2100-08-31"),
    ("WRF", "ssp585"): ("2014-09-01", "2100-08-31"),
    ("LOCA2", "historical"): ("1950-01-01", "2014-12-31"),
    ("LOCA2", "ssp245"): ("2015-01-01", "2100-12-31"),
    ("LOCA2", "ssp370"): ("2015-01-01", "2100-12-31"),
    ("LOCA2", "ssp585"): ("2015-01-01", "2100-12-31"),
}

#: Days of slack on those bounds. Generous enough for a monthly product
#: timestamped mid-month, tight enough to catch a four-month offset.
TIME_BOUND_TOLERANCE_DAYS = 45

#: Tolerance, in years, before a start/end mismatch is reported. One year of
#: slack absorbs the usual "runs end 2100-12-31 vs 2101-01-01" boundary noise.
TIME_RANGE_TOLERANCE_YEARS = 1

#: Expected spacing between consecutive time steps, per table_id.
#: (min_hours, max_hours); month is a range because month lengths vary.
EXPECTED_STEP_HOURS: dict[str, tuple[float, float]] = {
    "1hr": (1.0, 1.0),
    "day": (24.0, 24.0),
    "mon": (28.0 * 24, 31.0 * 24),
    "yrmax": (365.0 * 24, 366.0 * 24),
}

#: Leap-day behaviour documented per WRF driving model. LOCA2 models were all
#: interpolated to *include* leap days.
WRF_NO_LEAP_MODELS = frozenset({"CESM2", "FGOALS-g3", "TaiESM1"})
WRF_LEAP_MODELS = frozenset(
    {"EC-Earth3-Veg", "CNRM-ESM2-1", "EC-Earth3", "MIROC6", "MPI-ESM1-2-HR"}
)

# --------------------------------------------------------------------------
# Spatial expectations
# --------------------------------------------------------------------------

#: Nominal horizontal resolution in metres, and the fractional tolerance used
#: when comparing against actual projected-coordinate spacing.
NOMINAL_RESOLUTION_M: dict[str, float] = {
    "d01": 45_000.0,
    "d02": 9_000.0,
    "d03": 3_000.0,
}
RESOLUTION_TOLERANCE = 0.15

#: LOCA2 is on a geographic grid; 3 km over California is ~1/32 degree.
LOCA2_NOMINAL_DEG = 0.03125
LOCA2_DEG_TOLERANCE = 0.30

#: Plausible coordinate extent per domain, as ``(lat_range, lon_range)``.
#:
#: These differ by nearly a hemisphere and a single box does not fit them. The
#: 45 km ``d01`` parent domain spans western North America from roughly 9.5N to
#: 67.3N and 157W to 84W (verified against published stores); ``d02`` covers the
#: WECC region and ``d03`` California. LOCA2 publishes only on ``d03``.
#:
#: Bounds are deliberately generous. The purpose is catching coordinates that
#: are genuinely broken — sign errors, degree/radian confusion, 0-360 longitude,
#: NaNs — not policing the exact domain footprint.
PLAUSIBLE_EXTENT: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "d01": ((0.0, 75.0), (-175.0, -70.0)),
    "d02": ((20.0, 62.0), (-148.0, -88.0)),
    "d03": ((25.0, 50.0), (-132.0, -105.0)),
}

#: Used when the grid label is unknown or unrecognised.
DEFAULT_PLAUSIBLE_EXTENT: tuple[tuple[float, float], tuple[float, float]] = (
    (0.0, 75.0),
    (-175.0, -70.0),
)


def plausible_extent(
    grid_label: str | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``(lat_range, lon_range)`` for a grid label.

    Parameters
    ----------
    grid_label : str or None
        Domain name, e.g. ``"d03"``.

    Returns
    -------
    tuple
        ``(lat_range, lon_range)``, falling back to
        :data:`DEFAULT_PLAUSIBLE_EXTENT` for an unknown domain.
    """
    return PLAUSIBLE_EXTENT.get(str(grid_label), DEFAULT_PLAUSIBLE_EXTENT)


#: Non-CF attributes that still supply a human-readable descriptor. A store
#: carrying one of these but no ``long_name`` is imperfect rather than opaque.
LONG_NAME_ALIASES: tuple[str, ...] = ("description", "title", "standard_name")

#: CRS conventions, quoting "Coordinate Reference Systems and Grid Conventions":
#: WRF carries a `grid_mapping` attribute pointing at a `Lambert_Conformal`
#: coordinate variable; LOCA2 instead carries a `spatial_ref` coordinate and is
#: documented *not* to have `grid_mapping`.
EXPECTED_GRID_MAPPING: dict[str, str | None] = {
    "WRF": "Lambert_Conformal",
    "LOCA2": None,
}
EXPECTED_CRS_COORD: dict[str, str] = {
    "WRF": "Lambert_Conformal",
    "LOCA2": "spatial_ref",
}

#: Coordinate conventions a store may legitimately use, per activity.
#:
#: WRF is published on two different grids by two different pipelines: UCLA's
#: native hourly output is Lambert-projected with ``(time, y, x)`` dims and 2-D
#: lat/lon aux coordinates, while UCSD's daily and monthly products are
#: regridded to geographic ``(time, lat, lon)``. Both are valid; assuming only
#: the first flags every UCSD store as broken.
PROJECTED_DIMS = frozenset({"time", "x", "y"})
GEOGRAPHIC_DIMS = frozenset({"time", "lat", "lon"})

ACCEPTED_DIM_SETS: dict[str, tuple[frozenset[str], ...]] = {
    "WRF": (PROJECTED_DIMS, GEOGRAPHIC_DIMS),
    "LOCA2": (GEOGRAPHIC_DIMS,),
}

#: Retained for callers that want the canonical set for an activity.
EXPECTED_DIMS: dict[str, frozenset[str]] = {
    "WRF": PROJECTED_DIMS,
    "LOCA2": GEOGRAPHIC_DIMS,
}

#: Auxiliary 2-D coordinates expected on the curvilinear WRF grid.
WRF_AUX_COORDS = frozenset({"lat", "lon"})

# --------------------------------------------------------------------------
# Metadata expectations (Metadata Standards page)
# --------------------------------------------------------------------------

#: "At a minimum, variable attributes should contain units and a long name
#: descriptor (e.g. descriptive enough to label plots)."
REQUIRED_VAR_ATTRS = ("units", "long_name")

#: CF-recommended but not required by the Cal-Adapt minimum spec.
RECOMMENDED_VAR_ATTRS = ("standard_name", "cell_methods")

#: Recommended global attributes, Table 1 of Metadata Standards.
RECOMMENDED_GLOBAL_ATTRS = (
    "title",
    "institution",
    "source",
    "references",
    "history",
    "comment",
)

#: "List the standard convention used to organize the data. CF Convention is
#: preferred."
CONVENTIONS_ATTR_CANDIDATES = ("Conventions", "conventions", "Convention")
CONVENTIONS_EXPECTED_PREFIX = "CF-"

#: Coordinate variables must be self-describing: units + a naming attribute.
EXPECTED_COORD_UNITS: dict[str, tuple[str, ...]] = {
    "lat": ("degrees_north", "degree_north", "degrees_N", "degreeN"),
    "lon": ("degrees_east", "degree_east", "degrees_E", "degreeE"),
    "x": ("m", "meter", "metre", "meters", "metres"),
    "y": ("m", "meter", "metre", "meters", "metres"),
}

# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

#: Canonical spellings for udunits-style comparison. Keys are lowercase, with
#: whitespace stripped. Anything not listed falls through to a normalised
#: string compare.
_UNIT_ALIASES: dict[str, str] = {
    # temperature
    "k": "K",
    "kelvin": "K",
    "degk": "K",
    "degreesk": "K",
    "degc": "degC",
    "celsius": "degC",
    "degreesc": "degC",
    "c": "degC",
    "degf": "degF",
    "fahrenheit": "degF",
    # speed
    "ms-1": "m s-1",
    "m/s": "m s-1",
    "ms^-1": "m s-1",
    "meterssecond-1": "m s-1",
    "mps": "m s-1",
    # flux
    "wm-2": "W m-2",
    "w/m2": "W m-2",
    "w/m^2": "W m-2",
    "wm2": "W m-2",
    "watt/m2": "W m-2",
    # mixing ratio / specific humidity
    "kgkg-1": "kg kg-1",
    "kg/kg": "kg kg-1",
    "g/kg": "g kg-1",
    "gkg-1": "g kg-1",
    # precipitation
    "kgm-2s-1": "kg m-2 s-1",
    "kg/m2/s": "kg m-2 s-1",
    "kgm-2": "kg m-2",
    "kg/m2": "kg m-2",
    "mm": "mm",
    "millimeter": "mm",
    "mm/d": "mm d-1",
    "mm/day": "mm d-1",
    "mmd-1": "mm d-1",
    "mm/h": "mm h-1",
    "mm/hr": "mm h-1",
    "mm/s": "mm s-1",
    "mms-1": "mm s-1",
    # pressure
    "pa": "Pa",
    "pascal": "Pa",
    "hpa": "hPa",
    "mb": "hPa",
    "millibar": "hPa",
    # dimensionless / percent
    "%": "percent",
    "percent": "percent",
    "[0to100]": "percent",
    "1": "1",
    "": "1",
    "none": "1",
    "dimensionless": "1",
    "unitless": "1",
    "fraction": "1",
    # misc
    "j/kg": "J kg-1",
    "jkg-1": "J kg-1",
    "kg/m2": "kg m-2",
    "m": "m",
    "meter": "m",
    "metre": "m",
    "degrees": "degrees",
    "degree": "degrees",
}

#: Pairs that are *not* identical but are an accepted physical restatement.
#: These become INFO rather than ERROR so a reviewer still sees them.
_ACCEPTABLE_UNIT_SUBSTITUTIONS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"kg m-2", "mm"}),  # water depth over unit area
        frozenset({"kg m-2 s-1", "mm s-1"}),
        frozenset({"percent", "1"}),  # fraction vs percent, magnitude differs
        frozenset({"kg kg-1", "g kg-1"}),  # differ by 1000
        frozenset({"Pa", "hPa"}),  # differ by 100
    }
)


def normalize_unit(unit: str | None) -> str | None:
    """
    Reduce a units string to a canonical spelling for comparison.

    Parameters
    ----------
    unit : str or None
        Units string as published.

    Returns
    -------
    str or None
        Canonical spelling, ``"1"`` for an empty string, ``None`` for ``None``.
    """
    if unit is None:
        return None
    text = str(unit).strip()
    if not text:
        return "1"
    key = re.sub(r"[\s\*]", "", text).lower()
    key = key.replace("**", "^")
    if key in _UNIT_ALIASES:
        return _UNIT_ALIASES[key]
    # collapse whitespace but otherwise leave the author's spelling intact
    return re.sub(r"\s+", " ", text)


def compare_units(expected: str | None, actual: str | None) -> str:
    """
    Return one of ``"match"``, ``"substitution"``, ``"mismatch"``, ``"unknown"``.

    Three tiers, cheapest first: the alias table above, then the explicit
    substitution set, then — if pint is importable, which it is wherever
    climakitae is installed — a dimensional comparison. The pint tier catches
    spellings nobody thought to alias (``mm hr-1`` vs ``mm/h``) while still
    calling ``K`` vs ``degC`` a mismatch, since those are dimensionally equal but
    numerically offset.

    Parameters
    ----------
    expected : str or None
        Unit the reference table documents.
    actual : str or None
        Unit the store declares.

    Returns
    -------
    str
        ``"match"``, ``"substitution"`` when the two differ by a constant
        factor, ``"mismatch"``, or ``"unknown"`` when either side is absent.
    """
    if expected is None or actual is None:
        return "unknown"
    exp, act = normalize_unit(expected), normalize_unit(actual)
    if exp == act:
        return "match"
    if frozenset({exp, act}) in _ACCEPTABLE_UNIT_SUBSTITUTIONS:
        return "substitution"
    return _compare_units_pint(exp, act)


#: Units that are dimensionally identical but not interchangeable without an
#: offset or scale factor. pint would call these equal; we must not.
_OFFSET_SENSITIVE = frozenset({"K", "degC", "degF"})


def _to_pint_syntax(unit: str) -> str:
    """
    Rewrite udunits spelling into something pint can parse.

    udunits writes products as spaces and exponents as bare digits — ``kg kg-1``,
    ``W m-2``, ``kg m-2 s-1``. pint wants ``kg * kg**-1``. Without this the
    fallback fails to parse every compound unit in the archive and reports a
    mismatch, which is how ``huss`` in ``kg/kg`` came to disagree with the same
    quantity written as ``1``.

    Parameters
    ----------
    unit : str
        udunits-style units string.

    Returns
    -------
    str
        The same units in a form pint can parse.
    """
    tokens = str(unit).strip().split()
    converted = []
    for token in tokens:
        # kg-1 -> kg**-1, m2 -> m**2; leave things like "1" or "degC" alone
        converted.append(re.sub(r"^([A-Za-z]+)(-?\d+)$", r"\1**\2", token))
    return " * ".join(converted) if converted else str(unit)


def _compare_units_pint(expected: str, actual: str) -> str:
    """
    Compare two units dimensionally, using pint when it is importable.

    Parameters
    ----------
    expected : str
        Normalised expected unit.
    actual : str
        Normalised actual unit.

    Returns
    -------
    str
        ``"match"``, ``"substitution"`` or ``"mismatch"``. Temperature units are
        always a mismatch unless identical, since they differ by an offset that
        dimensional analysis does not see.
    """
    try:
        pass
    except Exception:
        return "mismatch"

    if {expected, actual} & _OFFSET_SENSITIVE and expected != actual:
        return "mismatch"

    try:
        registry = _pint_registry()
        left = registry.Unit(_to_pint_syntax(expected))
        right = registry.Unit(_to_pint_syntax(actual))
    except Exception:
        return "mismatch"

    if left.dimensionality != right.dimensionality:
        return "mismatch"
    try:
        factor = registry.Quantity(1.0, left).to(right).magnitude
    except Exception:
        return "mismatch"
    return "match" if abs(factor - 1.0) < 1e-9 else "substitution"


@functools.lru_cache(maxsize=1)
def _pint_registry() -> pint.UnitRegistry:
    """
    Build the pint registry, with the udunits spellings pint lacks.

    Returns
    -------
    pint.UnitRegistry
        Cached registry, with ``percent`` defined.
    """
    import pint

    registry = pint.UnitRegistry()
    # udunits spellings that pint does not know out of the box
    registry.define("percent = 0.01 = %")
    return registry


# --------------------------------------------------------------------------
# The variable table
# --------------------------------------------------------------------------

_TIMESCALE_TO_TABLE_IDS = {
    "hourly": {"1hr"},
    "daily": {"day"},
    "monthly": {"mon"},
}


@functools.lru_cache(maxsize=1)
def variable_table() -> pd.DataFrame:
    """
    Load ``variable_descriptions.csv`` shipped inside climakitae.

    Returns a frame with the original columns plus ``table_ids`` (a set) and
    ``activity_id`` mapped from ``downscaling_method``.

    Raises
    ------
    RuntimeError
        If climakitae is not importable. The table is the reference standard for
        names and units, so there is no sensible fallback.
    """
    try:
        import climakitae  # noqa: F401
    except Exception:  # pragma: no cover - exercised only without the dep
        pass

    path = _variable_csv_path()
    frame = pd.read_csv(path)
    frame["variable_id"] = frame["variable_id"].astype(str).str.strip()
    frame["activity_id"] = frame["downscaling_method"].map(
        {"Dynamical": "WRF", "Statistical": "LOCA2"}
    )
    frame["table_ids"] = frame["timescale"].map(_timescale_to_table_ids)
    return frame


def _variable_csv_path() -> str:
    """
    Locate variable_descriptions.csv inside the installed climakitae.

    Returns
    -------
    str
        Absolute path to the CSV.

    Raises
    ------
    RuntimeError
        If climakitae is not installed; the table is the reference standard for
        names and units, so there is no sensible fallback.
    """
    import importlib.util
    import os

    spec = importlib.util.find_spec("climakitae")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "climakitae is not installed; data_audit needs its "
            "data/variable_descriptions.csv as the reference variable table."
        )
    root = list(spec.submodule_search_locations)[0]
    return os.path.join(root, "data", "variable_descriptions.csv")


def _timescale_to_table_ids(timescale: str) -> frozenset[str]:
    """
    Map a reference-table timescale string to catalog table_ids.

    Parameters
    ----------
    timescale : str
        Reference-table timescale, e.g. ``"daily, monthly"``.

    Returns
    -------
    frozenset of str
        Matching catalog ``table_id`` values.
    """
    out: set = set()
    for token in str(timescale).split(","):
        out |= _TIMESCALE_TO_TABLE_IDS.get(token.strip().lower(), set())
    return frozenset(out)


@functools.lru_cache(maxsize=1)
def _variable_index() -> dict[tuple[str, str, str], pd.Series]:
    """
    (activity_id, table_id, variable_id) -> row of the variable table.

    Returns
    -------
    dict
        ``(activity_id, table_id, variable_id)`` to the reference table row.
    """
    index: dict[tuple[str, str, str], pd.Series] = {}
    for row in variable_table().itertuples():
        if not isinstance(row.activity_id, str):
            continue
        for table_id in row.table_ids:
            index[(row.activity_id, table_id, row.variable_id)] = row
    return index


def documented_variable(
    activity_id: str, table_id: str, variable_id: str
) -> pd.Series | None:
    """
    Look up a variable in the reference table, or ``None`` if undocumented.

    ``mon`` falls back to ``day`` because the table writes "daily, monthly" as a
    single timescale and a few rows only list one of the two.

    Parameters
    ----------
    activity_id : str
        Downscaling method.
    table_id : str
        Temporal resolution.
    variable_id : str
        Variable name.

    Returns
    -------
    pandas.Series or None
        The reference table row, or ``None`` when the variable is undocumented
        at this resolution.
    """
    index = _variable_index()
    row = index.get((activity_id, table_id, variable_id))
    if row is not None:
        return row
    if table_id == "mon":
        return index.get((activity_id, "day", variable_id))
    if table_id == "yrmax":
        # yrmax is an annual reduction of the daily product
        return index.get((activity_id, "day", variable_id))
    return None


def expected_unit(activity_id: str, table_id: str, variable_id: str) -> str | None:
    """
    Look up the documented unit for a variable.

    Parameters
    ----------
    activity_id : str
        Downscaling method.
    table_id : str
        Temporal resolution.
    variable_id : str
        Variable name.

    Returns
    -------
    str or None
        The documented unit, or ``None`` when the variable is undocumented.
    """
    row = documented_variable(activity_id, table_id, variable_id)
    return None if row is None else getattr(row, "unit", None)


def expected_long_name(activity_id: str, table_id: str, variable_id: str) -> str | None:
    """
    Look up the documented display name for a variable.

    Parameters
    ----------
    activity_id : str
        Downscaling method.
    table_id : str
        Temporal resolution.
    variable_id : str
        Variable name.

    Returns
    -------
    str or None
        The documented display name, or ``None`` when undocumented.
    """
    row = documented_variable(activity_id, table_id, variable_id)
    return None if row is None else getattr(row, "display_name", None)


def documented_variables(activity_id: str, table_id: str) -> frozenset[str]:
    """
    Every variable the reference table says should exist at this resolution.

    Derived variables (``*_derived``) are computed by climakitae at read time and
    are deliberately excluded, since they never appear as stores in S3.

    Parameters
    ----------
    activity_id : str
        Downscaling method.
    table_id : str
        Temporal resolution.

    Returns
    -------
    frozenset of str
        Variable names, excluding read-time derived variables and the
        renewables-catalog variables that share this table.
    """
    names = {
        variable_id
        for (act, tab, variable_id) in _variable_index()
        if act == activity_id
        and tab == table_id
        and not is_derived(variable_id)
        and variable_id not in RENEWABLES_SCOPE_VARIABLES
    }
    return frozenset(names)


def documented_table_ids(activity_id: str, variable_id: str) -> frozenset[str]:
    """
    Every table_id the reference table associates with this variable.

    Parameters
    ----------
    activity_id : str
        Downscaling method.
    variable_id : str
        Variable name.

    Returns
    -------
    frozenset of str
        Every ``table_id`` the reference table associates with the variable.
    """
    return frozenset(
        tab
        for (act, tab, var) in _variable_index()
        if act == activity_id and var == variable_id
    )


#: Derived variables and indices that climakitae computes at read time. They
#: appear in the reference variable table but never as stores in S3, so they
#: must not count as coverage gaps.
#:
#: ``ffwi`` is deliberately absent: the Fosberg index *is* derived, but CAE
#: materialises it to ``wrf/derived-vars/`` rather than computing it on read, so
#: it should be checked like any other published store.
DERIVED_VARIABLE_NAMES: frozenset[str] = frozenset(
    {
        "HDD_wrf",
        "HDD_loca",
        "CDD_wrf",
        "CDD_loca",
        "heat_index",
        "noaa_heat_index",
        "fosberg_fire_weather_index",
        "specific_humidity_2m",
        "effective_temp",
    }
)


def is_derived(variable_id: str) -> bool:
    """
    True for computed variables, e.g. ``rh_derived``, ``dew_point_derived_hrly``.

    Parameters
    ----------
    variable_id : str
        Variable name.

    Returns
    -------
    bool
        True for variables climakitae computes at read time, which therefore
        never appear as stores.
    """
    return "_derived" in variable_id or variable_id in DERIVED_VARIABLE_NAMES


# --------------------------------------------------------------------------
# Store paths
# --------------------------------------------------------------------------


#: Path segments that mean "this is not the canonical published location".
#: Anything served from here may be provisional or may disappear.
STAGING_PATH_SEGMENTS: frozenset[str] = frozenset({"tmp", "temp", "staging", "scratch"})

#: Segments that occupy the institution slot in a path without being an
#: institution. ``derived-vars`` namespaces computed products (e.g. the Fosberg
#: index) separately from the model output they are derived from, while the
#: catalog still attributes them to the producing institution.
PSEUDO_INSTITUTION_SEGMENTS: frozenset[str] = frozenset({"derived-vars"})

#: Variable-name suffixes that denote a physically distinct quantity from the
#: base name. ``u10_earth`` is earth-relative wind; bare ``u10`` from WRF is
#: grid-relative, aligned to the Lambert projection axes. Pooling the two
#: without rotating is a silent scientific error, and no amount of units or CRS
#: checking will catch it, because both are ``m s-1`` on the same grid.
MEANINGFUL_VARIABLE_SUFFIXES: dict[str, str] = {
    "_earth": (
        "earth-relative rather than grid-relative; WRF's native u10/v10 are "
        "aligned to the projection axes, so the two are not interchangeable "
        "without rotation"
    ),
    "_grid": "explicitly grid-relative",
    "_raw": "pre-bias-correction",
    "_bc": "bias-corrected",
}


#: Wind component variables whose values depend on the reference frame. Speed is
#: rotation-invariant; these are not.
WIND_COMPONENT_VARIABLES: frozenset[str] = frozenset(
    {"u10", "v10", "u", "v", "uas", "vas"}
)

#: Attribute text that declares an earth-relative (rotated) reference frame.
EARTH_RELATIVE_MARKERS: tuple[str, ...] = (
    "earth-relative",
    "earth relative",
    "earth_relative",
)

#: Attribute text that explicitly declares a grid-relative frame.
GRID_RELATIVE_MARKERS: tuple[str, ...] = (
    "grid-relative",
    "grid relative",
    "grid_relative",
)

#: Variable attributes searched for a reference-frame declaration.
PROVENANCE_ATTRS: tuple[str, ...] = (
    "long_name",
    "description",
    "standard_name",
    "history",
    "comment",
)


#: Activities whose store path carries a Variant/member_id level.
#: "LOCA2-Hybrid has an additional level (Variant/member_id)" — Data Structure
#: and Format. WRF assigns a member_id facet in the catalog but does *not* put
#: it in the path, which is one of the documented internal inconsistencies.
ACTIVITIES_WITH_MEMBER_IN_PATH: frozenset[str] = frozenset({"LOCA2"})


def expected_s3_path(row: dict[str, str]) -> str | None:
    """
    Rebuild the intake-ESM store path implied by a row's facets.

    The directory layout is
    ``<activity>/<institution>/<source>/<experiment>/[<member>/]<table>/<var>/<grid>/``
    all lowercased, with the member level present only for the activities in
    :data:`ACTIVITIES_WITH_MEMBER_IN_PATH`. Comparing this against the catalog's
    own ``path`` column is what catches records whose facets and location have
    drifted apart.

    Parameters
    ----------
    row : dict
        Catalog record.

    Returns
    -------
    str or None
        The implied ``s3://`` path, or ``None`` when a required facet is
        missing.
    """
    required = [
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "table_id",
        "variable_id",
        "grid_label",
    ]
    if any(not row.get(k) or pd.isna(row.get(k)) for k in required):
        return None

    activity_id = str(row["activity_id"])
    parts = [
        activity_id,
        row["institution_id"],
        row["source_id"],
        row["experiment_id"],
    ]
    member = row.get("member_id")
    if activity_id in ACTIVITIES_WITH_MEMBER_IN_PATH:
        if isinstance(member, str) and member and member.lower() != "nan":
            parts.append(member)
    parts += [row["table_id"], row["variable_id"], row["grid_label"]]
    tail = "/".join(str(p).lower() for p in parts)
    return f"s3://{S3_BUCKET}/{tail}/"


#: Variables that live in the *renewable energy generation* catalog rather than
#: cadcat, but which share ``variable_descriptions.csv``. Excluded from cadcat
#: coverage expectations so they do not read as missing WRF data.
RENEWABLES_SCOPE_VARIABLES: frozenset[str] = frozenset({"cf", "gen"})
