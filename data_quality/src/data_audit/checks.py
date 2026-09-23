"""
Individual QC checks.

Every check takes a context dict (the catalog facets) plus whatever object it
inspects, and returns a :class:`~data_audit.findings.FindingList`. Checks are
pure and never raise: an unopenable dataset produces a FATAL finding, not a
traceback.

Two families:

``check_catalog_row``
    Needs no data access. Validates the facets themselves against the documented
    vocabulary and rebuilds the expected S3 path.

``check_dataset``
    Needs a lazily-opened :class:`xarray.Dataset`. Validates variable naming,
    units, attributes, CRS, coordinates and the time axis. Reads coordinates
    (small) but never the data array itself.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from . import standards as S
from .findings import FindingList, Level

# -----------------------------------------------------------------------
# Catalog-level checks
# -----------------------------------------------------------------------


def check_catalog_row(row: dict[str, Any]) -> FindingList:
    """
    Validate one catalog record without touching S3.

    Parameters
    ----------
    row : dict
        One catalog record.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    activity = row.get("activity_id")
    table_id = row.get("table_id")
    grid_label = row.get("grid_label")
    variable_id = row.get("variable_id")

    if activity not in S.ACTIVITIES:
        out.info(
            "catalog.activity.unknown",
            f"activity_id {activity!r} is outside the WRF/LOCA2 scope of this sweep.",
            expected=list(S.ACTIVITIES),
            actual=activity,
        )
        return out

    #  vocabulary -------------------------------------------------------
    valid_grids = S.VALID_GRID_LABELS[activity]
    if grid_label not in valid_grids:
        out.error(
            "catalog.grid_label.invalid",
            f"{activity} is documented only on {sorted(valid_grids)}, "
            f"but this record is on {grid_label!r}.",
            expected=sorted(valid_grids),
            actual=grid_label,
        )

    valid_tables = S.VALID_TABLE_IDS[activity]
    if table_id not in valid_tables:
        out.error(
            "catalog.table_id.invalid",
            f"{activity} is documented at {sorted(valid_tables)}, "
            f"but this record is at {table_id!r}.",
            expected=sorted(valid_tables),
            actual=table_id,
        )

    experiment = row.get("experiment_id")
    valid_experiments = S.VALID_EXPERIMENT_IDS[activity]
    if experiment not in valid_experiments:
        out.warn(
            "catalog.experiment_id.unexpected",
            f"experiment_id {experiment!r} is not in the documented set for {activity}.",
            expected=sorted(valid_experiments),
            actual=experiment,
        )

    institution = row.get("institution_id")
    if institution not in S.EXPECTED_INSTITUTIONS[activity]:
        out.warn(
            "catalog.institution_id.unexpected",
            f"institution_id {institution!r} is not documented for {activity}.",
            expected=sorted(S.EXPECTED_INSTITUTIONS[activity]),
            actual=institution,
        )

    source_id = row.get("source_id")
    if source_id not in S.DOCUMENTED_MODELS[activity]:
        out.warn(
            "catalog.source_id.undocumented",
            f"source_id {source_id!r} is not in the documented model list for {activity}.",
            expected=sorted(S.DOCUMENTED_MODELS[activity]),
            actual=source_id,
        )

    #  member_id
    member = row.get("member_id")
    has_member = isinstance(member, str) and member and member.lower() != "nan"
    if activity == "LOCA2":
        if not has_member:
            out.error(
                "catalog.member_id.missing",
                "LOCA2 records carry a Variant/member_id level; this one has none.",
                expected="rXiYpZfW",
                actual=member,
            )
        elif not S.MEMBER_ID_RE.match(member):
            out.warn(
                "catalog.member_id.malformed",
                f"member_id {member!r} does not match the CMIP6 rXiYpZfW pattern.",
                expected="rXiYpZfW",
                actual=member,
            )

    #  variable documented?
    if isinstance(table_id, str) and isinstance(variable_id, str):
        documented = S.documented_variable(activity, table_id, variable_id)
        if documented is None:
            elsewhere = S.documented_table_ids(activity, variable_id)
            if elsewhere:
                out.warn(
                    "catalog.variable.timescale_mismatch",
                    f"{variable_id!r} is published at {activity}/{table_id} but "
                    f"variable_descriptions.csv lists it only at {sorted(elsewhere)}, "
                    "so no unit or display name resolves at this resolution.",
                    expected=sorted(elsewhere),
                    actual=table_id,
                )
            else:
                out.warn(
                    "catalog.variable.undocumented",
                    f"{variable_id!r} at {activity}/{table_id} is not in "
                    "variable_descriptions.csv at any resolution, so climakitae "
                    "has no unit or display name for it.",
                    actual=variable_id,
                )

    #  path convention
    actual_path = row.get("path")
    expected_path = S.expected_s3_path(row)
    if expected_path and isinstance(actual_path, str):
        out += _check_path(
            actual_path, expected_path, activity, institution, member, has_member
        )
    elif expected_path and not isinstance(actual_path, str):
        out.warn(
            "catalog.path.missing",
            "Catalog record has no path.",
            expected=expected_path,
        )

    return out


def _path_segments(path: str) -> list[str]:
    """
    ``s3://cadcat/wrf/ucla/...`` -> ``['wrf', 'ucla', ...]`` (bucket dropped).

    Parameters
    ----------
    path : str
        Store path, with or without the ``s3://`` scheme.

    Returns
    -------
    list of str
        Path segments with the bucket removed.
    """
    text = str(path).strip().rstrip("/").lower()
    if text.startswith("s3://"):
        text = text[len("s3://") :]
    parts = [p for p in text.split("/") if p]
    return parts[1:] if parts else []


def _check_path(
    actual_path: str,
    expected_path: str,
    activity: str | None,
    institution: str | None,
    member: Any,
    has_member: bool,
) -> FindingList:
    """
    Compare a store path to the one its facets imply, naming the difference.

    A flat "mismatch" is nearly useless on this archive: most differences are
    deliberate namespacing, and the one that matters scientifically looks
    identical to the ones that do not. So each recognised kind of difference
    gets its own code and severity, and only genuinely unexplained differences
    stay at ERROR.

    Parameters
    ----------
    actual_path : str
        Path recorded in the catalog.
    expected_path : str
        Path implied by the record's facets.
    activity : str or None
        Downscaling method, used in messages.
    institution : str or None
        Producing institution, used in messages.
    member : Any
        Ensemble member id, if any.
    has_member : bool
        Whether ``member`` is a usable string.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    actual_segments = _path_segments(actual_path)
    expected_segments = _path_segments(expected_path)

    if actual_segments == expected_segments:
        return out

    residual = list(actual_segments)

    #  staging prefix
    if residual and residual[0] in S.STAGING_PATH_SEGMENTS:
        staging = residual.pop(0)
        out.error(
            "catalog.path.staging_prefix",
            f"Store is served from a {staging!r} prefix rather than the canonical "
            "location. Data under a staging prefix carries no persistence "
            "guarantee, so anything pinned to this path can break without notice.",
            expected=expected_path,
            actual=actual_path,
        )

    #  pseudo-institution
    if len(residual) > 1 and residual[1] in S.PSEUDO_INSTITUTION_SEGMENTS:
        segment = residual[1]
        out.warn(
            "catalog.path.pseudo_institution",
            f"Path uses {segment!r} in the institution slot while the catalog "
            f"attributes this record to {institution!r}. A deliberate namespace "
            "for computed products, but it means path reconstruction from facets "
            "cannot be purely mechanical.",
            expected=institution,
            actual=segment,
        )
        residual[1] = str(institution).lower() if institution else residual[1]

    #  variable name
    if len(residual) >= 2 and len(expected_segments) >= 2:
        actual_var, expected_var = residual[-2], expected_segments[-2]
        if actual_var != expected_var:
            suffix = next(
                (s for s in S.MEANINGFUL_VARIABLE_SUFFIXES if actual_var.endswith(s)),
                None,
            )
            if suffix and actual_var[: -len(suffix)] == expected_var:
                out.error(
                    "catalog.path.variable_qualifier_dropped",
                    f"The store holds {actual_var!r} but the catalog indexes it as "
                    f"{expected_var!r}. The {suffix!r} qualifier means "
                    f"{S.MEANINGFUL_VARIABLE_SUFFIXES[suffix]}. Because the catalog "
                    "drops it, these records pool silently with ordinary "
                    f"{expected_var!r} and no metadata check will notice.",
                    expected=expected_var,
                    actual=actual_var,
                )
            else:
                out.error(
                    "catalog.path.variable_mismatch",
                    f"Path variable segment is {actual_var!r} but the catalog says "
                    f"{expected_var!r}.",
                    expected=expected_var,
                    actual=actual_var,
                )
            residual[-2] = expected_var

    #  member level
    if has_member and residual != expected_segments:
        token = str(member).lower()
        if [s for s in residual if s != token] == expected_segments:
            out.warn(
                "catalog.path.member_level_differs",
                f"{activity} paths from {institution} carry a member_id level that "
                "the rest of the activity omits. The layout is not consistent "
                "within the activity, so anything rebuilding paths from facets has "
                "to special-case this institution.",
                expected=expected_path,
                actual=actual_path,
            )
            residual = [s for s in residual if s != token]

    #  anything left unexplained
    if residual != expected_segments and not out:
        out.error(
            "catalog.path.mismatch",
            "Store path does not match the path implied by its own facets, in a "
            "way that matches no known naming convention.",
            expected=expected_path,
            actual=actual_path,
        )
    elif residual != expected_segments:
        out.info(
            "catalog.path.residual_difference",
            "Path still differs from its facets after accounting for the known "
            "naming conventions above.",
            expected="/".join(expected_segments),
            actual="/".join(residual),
        )
    return out


def _normalize_path(path: str) -> str:
    """
    Lower-case a path and strip its trailing slash for comparison.

    Parameters
    ----------
    path : str
        Store path.

    Returns
    -------
    str
        Lower-cased path without a trailing slash.
    """
    return path.strip().rstrip("/").lower()


# -----------------------------------------------------------------------
# Dataset-level checks
# -----------------------------------------------------------------------


#: Dimensions that indicate a dataset is an aggregation over several stores
#: rather than a single published store. intake-esm adds these when a query
#: matches more than one simulation.
AGGREGATION_DIMS = ("member_id", "sim", "simulation", "source_id", "scenario")


def aggregation_dims(dataset: xr.Dataset) -> dict[str, int]:
    """
    Return aggregation dimensions of length > 1, with their sizes.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to inspect.

    Returns
    -------
    dict of str to int
        Aggregation dimension names and sizes, for dimensions longer than one.
        Empty when the dataset represents a single store.
    """
    return {
        name: int(dataset.sizes[name])
        for name in AGGREGATION_DIMS
        if name in dataset.sizes and dataset.sizes[name] > 1
    }


def check_aggregation(dataset: xr.Dataset, context: dict[str, Any]) -> FindingList:
    """
    Flag a result that combines several stores into one object.

    ``ClimateData`` exposes no ``member_id`` setter, so a LOCA2 query can pin
    every available facet and still match every ensemble member. intake-esm then
    aggregates them, aligning time across members. Where the members' parent
    calendars differ, the alignment drops dates that any one member lacks —
    29 February, typically — producing a time axis with gaps that exists in no
    published store.

    Time-axis findings on such a dataset describe the aggregation, not the data,
    so :func:`check_time` demotes them when this fires.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.
    context : dict
        Catalog facets identifying the dataset.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    dims = aggregation_dims(dataset)
    if not dims:
        return out

    described = ", ".join(f"{name}={size}" for name, size in dims.items())
    out.warn(
        "dataset.aggregated",
        f"This result combines several stores ({described}) rather than being a "
        "single published dataset. Time alignment across members can drop dates "
        "that any one member lacks, so gaps and calendar findings here are "
        "properties of the aggregation, not of the archive.",
        expected="one store per query",
        actual=described,
    )
    return out


def check_dataset(
    dataset: xr.Dataset,
    context: dict[str, Any],
    check_time_axis: bool = True,
    probe_values: bool = False,
    authoritative_structure: bool = True,
) -> FindingList:
    """
    Run every dataset check against a lazily-opened store.

    ``probe_values`` additionally reads a few 2-D slices to confirm the variable
    is not empty. It is the only part of this module that touches data.

    ``authoritative_structure`` says whether this object's *shape* — its time
    axis in particular — is a property of a published store. It is False for any
    reader that aggregates or post-processes. climakitae aligns and combines
    matching stores before handing back a result, so gaps and missing leap days
    in what it returns describe that alignment rather than the archive; three
    separate investigations of such findings all traced back to the reader.
    Structural findings are demoted to INFO when this is False.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.
    context : dict
        Catalog facets identifying the dataset.
    check_time_axis : bool, optional
        Run the time-axis checks. Default True.
    probe_values : bool, optional
        Read a few slices to confirm the variable is not empty. Default False.
    authoritative_structure : bool, optional
        Whether this object's shape reflects a published store. Default True.

    Returns
    -------
    FindingList
        Every finding from every applicable check.
    """
    out = FindingList()
    activity = context.get("activity_id")
    table_id = context.get("table_id")
    variable_id = context.get("variable_id")

    aggregated = bool(aggregation_dims(dataset)) or not authoritative_structure
    out += check_aggregation(dataset, context)
    out += check_variable_present(dataset, context)
    data_var = _find_data_var(dataset, variable_id)

    if data_var is not None:
        out += check_variable_attrs(dataset[data_var], activity, table_id, variable_id)
        out += check_fill_value(dataset[data_var])
        out += check_wind_rotation(dataset[data_var], context)
        if probe_values:
            out += check_data_presence(dataset, data_var)
            out += check_unwritten_chunks(dataset, data_var)
        out += check_dims(dataset, data_var, activity)
        out += check_crs(dataset, dataset[data_var], activity)

    out += check_global_attrs(dataset)
    out += check_coordinate_variables(dataset)
    out += check_spatial_grid(dataset, activity, context.get("grid_label"))
    if check_time_axis:
        out += check_time(dataset, context, aggregated=aggregated)
    return out


def _find_data_var(dataset: xr.Dataset, variable_id: str | None) -> str | None:
    """
    Locate the payload variable, tolerating renames.

    Prefers an exact ``variable_id`` match, then falls back to the single
    highest-dimensional data variable so that downstream checks still run on a
    store whose variable was renamed.

    Parameters
    ----------
    dataset : xarray.Dataset
        Store to search.
    variable_id : str or None
        Name the catalog gives the payload variable.

    Returns
    -------
    str or None
        The resolved variable name, or ``None`` when the store holds no
        multi-dimensional candidate.
    """
    if variable_id and variable_id in dataset.data_vars:
        return variable_id
    candidates = [
        name
        for name, da in dataset.data_vars.items()
        if da.ndim >= 2 and name not in {"Lambert_Conformal", "spatial_ref", "crs"}
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return max(candidates, key=lambda n: dataset[n].ndim)
    return None


def check_variable_present(dataset: xr.Dataset, context: dict[str, Any]) -> FindingList:
    """
    Confirm the store holds the variable the catalog names.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.
    context : dict
        Catalog facets identifying the dataset.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    variable_id = context.get("variable_id")
    present = list(dataset.data_vars)

    if not present:
        out.fatal("var.none", "Store contains no data variables.", actual=present)
        return out

    if variable_id and variable_id in dataset.data_vars:
        out.ok(
            "var.present", f"Variable {variable_id!r} present as named in the catalog."
        )
        return out

    resolved = _find_data_var(dataset, variable_id)
    if resolved is None:
        out.error(
            "var.missing",
            f"Catalog says variable_id={variable_id!r} but the store has no "
            "variable with that name and no unambiguous substitute.",
            expected=variable_id,
            actual=present,
        )
    else:
        out.error(
            "var.name_mismatch",
            f"Catalog says variable_id={variable_id!r}; the store instead holds "
            f"{resolved!r}. Anything keying on the catalog name will miss this store.",
            expected=variable_id,
            actual=resolved,
        )

    extras = [n for n in present if n != resolved and dataset[n].ndim >= 2]
    if extras:
        out.info(
            "var.extra",
            "Store carries additional multi-dimensional variables.",
            actual=extras,
        )
    return out


def check_variable_attrs(
    array: xr.DataArray,
    activity_id: str | None,
    table_id: str | None,
    variable_id: str | None,
) -> FindingList:
    """
    Units and long_name are the documented minimum for variable attributes.

    Parameters
    ----------
    array : xarray.DataArray
        The payload variable.
    activity_id : str or None
        Downscaling method, used to look up the documented unit.
    table_id : str or None
        Temporal resolution, likewise.
    variable_id : str or None
        Catalog variable name, likewise.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    attrs = dict(array.attrs)

    for name in S.REQUIRED_VAR_ATTRS:
        if str(attrs.get(name, "")).strip():
            continue
        alias = next(
            (a for a in S.LONG_NAME_ALIASES if str(attrs.get(a, "")).strip()), None
        )
        if name == "long_name" and alias:
            # A descriptor exists, just not under the CF-conventional key. Plot
            # labellers keying on long_name will still come up empty, but the
            # store is not opaque.
            out.warn(
                "attr.var.long_name.aliased",
                f"No 'long_name', but {alias!r} carries a descriptor "
                f"({attrs[alias]!r}). CF tooling looks for long_name.",
                expected="long_name",
                actual=alias,
            )
        else:
            out.error(
                f"attr.var.{name}.missing",
                f"Required variable attribute {name!r} is missing or empty. "
                "The metadata standard sets units + long_name as the minimum.",
                expected=name,
            )

    for name in S.RECOMMENDED_VAR_ATTRS:
        if not str(attrs.get(name, "")).strip():
            out.info(
                f"attr.var.{name}.missing",
                f"CF-recommended attribute {name!r} is absent.",
                expected=name,
            )

    # units against the reference table
    if activity_id and table_id and variable_id:
        expected = S.expected_unit(activity_id, table_id, variable_id)
        actual = attrs.get("units")
        verdict = S.compare_units(expected, actual)
        if verdict == "match":
            out.ok("units.match", f"units {actual!r} match the reference table.")
        elif verdict == "substitution":
            out.warn(
                "units.substitution",
                f"units {actual!r} are a physical restatement of the documented "
                f"{expected!r}, not the same string. Values may differ by a "
                "constant factor; confirm before combining datasets.",
                expected=expected,
                actual=actual,
            )
        elif verdict == "mismatch":
            out.error(
                "units.mismatch",
                f"units {actual!r} disagree with the documented {expected!r}.",
                expected=expected,
                actual=actual,
            )

        expected_name = S.expected_long_name(activity_id, table_id, variable_id)
        actual_name = attrs.get("long_name") or attrs.get("description")
        if expected_name and actual_name:
            if _slug(expected_name) != _slug(actual_name):
                out.info(
                    "long_name.differs",
                    "long_name differs from the reference table display name.",
                    expected=expected_name,
                    actual=actual_name,
                )
    return out


def _slug(text: str) -> str:
    """
    Reduce a display name to comparable letters and digits.

    Parameters
    ----------
    text : str
        Display name.

    Returns
    -------
    str
        Lower-cased letters and digits only, so spacing and punctuation do not
        make two equivalent names compare unequal.
    """
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def check_fill_value(array: xr.DataArray) -> FindingList:
    """
    A fill value or valid_range should be declared on anything with gaps.

    Parameters
    ----------
    array : xarray.DataArray
        The payload variable.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    attrs = dict(array.attrs)
    encoding = dict(array.encoding)
    has_fill = any(k in attrs or k in encoding for k in ("_FillValue", "missing_value"))
    has_range = "valid_range" in attrs or {"valid_min", "valid_max"} <= set(attrs)
    if not (has_fill or has_range):
        out.info(
            "attr.var.fill.missing",
            "Neither _FillValue/missing_value nor valid_range is declared. The "
            "standard asks for one on any variable with missing data.",
            expected="_FillValue or valid_range",
        )
    dtype = str(array.dtype)
    if dtype.startswith("int") and not has_fill:
        out.warn(
            "dtype.int_without_fill",
            f"Integer dtype {dtype} with no _FillValue leaves gaps unrepresentable.",
            actual=dtype,
        )
    return out


def check_wind_rotation(array: xr.DataArray, context: dict[str, Any]) -> FindingList:
    """
    Record which reference frame a wind component is in.

    WRF's native ``u10``/``v10`` are grid-relative, aligned to the projection
    axes. Rotating them to earth-relative changes the values by the projection's
    convergence angle. Both frames carry identical units, grid, CRS and calendar,
    so no conventional metadata check distinguishes them — but mixing them
    corrupts wind direction and any u/v vector operation. Speed is unaffected.

    The finding is INFO because neither frame is wrong. Pivot a sweep on this
    code to see how the archive splits, which is the actionable output.

    Parameters
    ----------
    array : xarray.DataArray
        The payload variable.
    context : dict
        Catalog facets identifying the dataset.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    variable_id = context.get("variable_id")
    if variable_id not in S.WIND_COMPONENT_VARIABLES:
        return out

    attrs = dict(array.attrs)
    text = " ".join(str(attrs.get(name, "")) for name in S.PROVENANCE_ATTRS).lower()

    is_earth = any(marker in text for marker in S.EARTH_RELATIVE_MARKERS)
    is_grid = any(marker in text for marker in S.GRID_RELATIVE_MARKERS)

    if is_earth:
        out.info(
            "wind.rotation.earth_relative",
            f"{variable_id!r} is earth-relative (rotated). Not interchangeable "
            "with grid-relative winds from the same archive.",
            actual="earth_relative",
        )
        if variable_id in {"u10", "v10"}:
            out.warn(
                "wind.rotation.hidden_by_catalog",
                f"The store documents itself as earth-relative, but the catalog "
                f"indexes it under the bare name {variable_id!r}, so it pools with "
                "grid-relative records in any variable_id query. The stores are "
                "self-describing; the catalog is what needs fixing.",
                expected=f"{variable_id}_earth or a distinguishing facet",
                actual=variable_id,
            )
    elif is_grid:
        out.info(
            "wind.rotation.grid_relative",
            f"{variable_id!r} explicitly declares a grid-relative frame.",
            actual="grid_relative",
        )
    else:
        out.warn(
            "wind.rotation.undeclared",
            f"{variable_id!r} declares no reference frame in its attributes. WRF "
            "output is grid-relative by default, but a consumer has no way to "
            "confirm that from the file, and rotated companions exist in this "
            "same archive.",
            expected="a reference frame in long_name/description",
            actual=attrs.get("description") or attrs.get("long_name"),
        )
    return out


def check_data_presence(
    dataset: xr.Dataset, data_var: str, n_samples: int = 3
) -> FindingList:
    """
    Read a few 2-D slices and confirm the variable actually holds data.

    Every other check in this module reads metadata only, which means a store
    can pass all of them while containing nothing: a regridded product whose
    pipeline wrote coordinates and attributes but no values looks perfectly
    healthy from the outside. cadcat serves at least one such store.

    Cost is bounded — three horizontal slices, a megabyte or so — but it is a
    genuine data read, so the sweep only runs it when asked.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.
    data_var : str
        Name of the payload variable.
    n_samples : int, optional
        Timesteps to sample. Default 3, taken from the start, middle and end.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    array = dataset[data_var]
    if "time" not in array.dims:
        return out

    n_time = array.sizes["time"]
    if n_time == 0:
        out.error("data.no_timesteps", "Variable has zero timesteps.")
        return out

    indices = sorted({0, n_time // 2, n_time - 1})[:n_samples]
    fractions: list[float] = []
    for index in indices:
        try:
            values = np.asarray(array.isel(time=index).values, dtype="float64")
        except Exception as exc:
            out.error(
                "data.read_failed",
                f"Could not read timestep {index}: {type(exc).__name__}: {exc}",
                actual=str(exc)[:200],
            )
            return out
        if values.size == 0:
            fractions.append(0.0)
        else:
            fractions.append(float(np.isfinite(values).mean()))

    best = max(fractions) if fractions else 0.0
    detail = ", ".join(f"t={i}: {f * 100:.1f}%" for i, f in zip(indices, fractions))

    if best == 0.0:
        out.error(
            "data.all_missing",
            f"Every sampled timestep is entirely missing ({detail}). The store "
            "has coordinates and attributes but no values, so it passes metadata "
            "checks while being unusable.",
            expected="finite values",
            actual="0% finite",
        )
    elif best < 0.01:
        out.error(
            "data.almost_all_missing",
            f"Sampled timesteps are almost entirely missing ({detail}).",
            expected="finite values",
            actual=f"{best * 100:.2f}% finite",
        )
    elif best < 0.05:
        out.warn(
            "data.sparse",
            f"Under 5% of each sampled field is finite ({detail}), which is too "
            "little to be a land mask.",
            actual=f"{best * 100:.1f}% finite",
        )
    else:
        # A masked domain inside a lat/lon rectangle is legitimately far under
        # half finite: LOCA2 over California sits around 31%. Record the
        # footprint rather than judging it, so a change in coverage between
        # variables or releases is visible.
        out.info(
            "data.coverage",
            f"Finite coverage {best * 100:.1f}% ({detail}).",
            actual=f"{best * 100:.1f}% finite",
        )

    if len(set(round(f, 3) for f in fractions)) > 1:
        out.warn(
            "data.coverage_varies",
            f"Finite coverage differs between sampled timesteps ({detail}), so "
            "the field is not uniformly populated across the record.",
            actual=detail,
        )
    return out


def check_unwritten_chunks(
    dataset: xr.Dataset, data_var: str, max_steps: int = 40_000
) -> FindingList:
    """
    Detect time chunks that were never written.

    Zarr returns the fill value for a chunk that was never written, so an
    incompletely-written store reads as NaN with no error. Its metadata,
    coordinates and consolidated index all look correct; nothing declares the
    absence. The giveaway is that the empty regions align exactly to chunk
    boundaries rather than to anything in the calendar.

    Found in the cadcat daily wind product, where every store was missing time
    chunks 0 and 1 and the projections were truncated part way, leaving 50-73%
    populated behind healthy-looking metadata.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.
    data_var : str
        Name of the payload variable.
    max_steps : int, optional
        Skip the scan above this many timesteps; it reads one reduction per
        step. Default 40000, enough for a daily record to 2100.

    Returns
    -------
    FindingList
        ``data.unwritten_chunks`` when the empty regions align to chunk
        boundaries, ``data.gaps`` when they do not.
    """
    out = FindingList()
    array = dataset[data_var]
    if "time" not in array.dims:
        return out
    n_time = array.sizes["time"]
    if n_time == 0 or n_time > max_steps:
        return out

    spatial = [d for d in array.dims if d != "time"]
    if not spatial:
        return out
    try:
        empty = (array.isnull().mean(dim=spatial) == 1.0).values
    except Exception as exc:
        out.info("data.scan_failed", f"Could not scan for gaps: {exc}")
        return out

    n_empty = int(empty.sum())
    if n_empty == 0:
        out.ok("data.complete", "Every timestep carries data.")
        return out

    chunk = _time_chunk(array)
    populated = float(1.0 - n_empty / n_time)

    if chunk:
        boundaries_ok = _runs_align_to_chunks(empty, chunk)
        missing = sorted({i // chunk for i in np.flatnonzero(empty)})
        n_chunks = -(-n_time // chunk)
        if boundaries_ok:
            out.error(
                "data.unwritten_chunks",
                f"{len(missing)} of {n_chunks} time chunks were never written "
                f"({populated * 100:.0f}% of the record populated). The empty "
                "regions align exactly to chunk boundaries, so this is an "
                "incomplete write rather than missing source data. Readers get "
                "fill values with no error.",
                expected=f"{n_chunks} chunks written",
                actual="missing chunks " + ", ".join(str(int(m)) for m in missing[:12]),
            )
            return out

    out.error(
        "data.gaps",
        f"{n_empty} of {n_time} timesteps are entirely missing "
        f"({populated * 100:.0f}% populated).",
        expected="no empty timesteps",
        actual=f"{n_empty} empty",
    )
    return out


def _time_chunk(array: xr.DataArray) -> int | None:
    """
    Return the stored chunk length along time.

    Parameters
    ----------
    array : xarray.DataArray
        The payload variable.

    Returns
    -------
    int or None
        Chunk length, or ``None`` when it cannot be determined.
    """
    chunks = array.encoding.get("chunks")
    if chunks:
        try:
            return int(chunks[list(array.dims).index("time")])
        except Exception:
            pass
    if array.chunks:
        try:
            return int(array.chunks[list(array.dims).index("time")][0])
        except Exception:
            pass
    return None


def _runs_align_to_chunks(empty: np.ndarray, chunk: int) -> bool:
    """
    Test whether every empty/populated transition sits on a chunk edge.

    Parameters
    ----------
    empty : numpy.ndarray
        Boolean mask, True where a timestep is entirely missing.
    chunk : int
        Chunk length along time.

    Returns
    -------
    bool
        True when every transition is a multiple of ``chunk``. The final
        boundary is exempt, since the last chunk is usually short.
    """
    edges = np.flatnonzero(np.diff(empty.astype(np.int8)) != 0) + 1
    if edges.size == 0:
        return True
    return all(int(e) % chunk == 0 for e in edges if int(e) < len(empty))


def check_dims(
    dataset: xr.Dataset, data_var: str, activity_id: str | None
) -> FindingList:
    """
    Validate dims against any grid convention the activity legitimately uses.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.
    data_var : str
        Name of the payload variable.
    activity_id : str or None
        Downscaling method, which determines the accepted grid conventions.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    accepted = S.ACCEPTED_DIM_SETS.get(activity_id)
    if not accepted:
        return out

    actual = set(dataset[data_var].dims)
    matched = next((expected for expected in accepted if expected <= actual), None)

    if matched is None:
        best = min(accepted, key=lambda e: len(e - actual))
        out.error(
            "dims.missing",
            f"Data variable is missing expected dimension(s) {sorted(best - actual)}.",
            expected=" or ".join(sorted(str(sorted(e)) for e in accepted)),
            actual=sorted(actual),
        )
        return out

    is_projected = matched == S.PROJECTED_DIMS
    out.info(
        "grid.convention",
        f"Store is on a {'projected' if is_projected else 'geographic'} grid "
        f"({sorted(matched)}).",
        actual="projected" if is_projected else "geographic",
    )

    extra = actual - matched
    if extra:
        out.info(
            "dims.extra", "Data variable has extra dimensions.", actual=sorted(extra)
        )

    if is_projected:
        missing_aux = S.WRF_AUX_COORDS - set(dataset.coords)
        if missing_aux:
            out.warn(
                "coords.wrf_aux.missing",
                f"Projected WRF stores are on a curvilinear grid and should carry "
                f"2-D {sorted(S.WRF_AUX_COORDS)} coordinates; missing "
                f"{sorted(missing_aux)}.",
                expected=sorted(S.WRF_AUX_COORDS),
            )
    return out


def _is_projected(dataset: xr.Dataset, array: xr.DataArray) -> bool:
    """
    True when the payload sits on x/y rather than lat/lon.

    Parameters
    ----------
    dataset : xarray.Dataset
        Store the array belongs to.
    array : xarray.DataArray
        The payload variable.

    Returns
    -------
    bool
        True when the payload sits on ``x``/``y`` rather than ``lat``/``lon``.
    """
    return {"x", "y"} <= set(array.dims)


def check_crs(
    dataset: xr.Dataset, array: xr.DataArray, activity_id: str | None
) -> FindingList:
    """
    WRF should carry grid_mapping -> Lambert_Conformal; LOCA2 a spatial_ref.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.
    array : xarray.DataArray
        The payload variable.
    activity_id : str or None
        Downscaling method, which determines the expected CRS convention.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    if activity_id not in S.EXPECTED_GRID_MAPPING:
        return out

    grid_mapping = array.attrs.get("grid_mapping")
    expected_mapping = S.EXPECTED_GRID_MAPPING[activity_id]
    crs_coord = S.EXPECTED_CRS_COORD[activity_id]
    names = set(dataset.coords) | set(dataset.data_vars) | set(dataset.variables)

    # WRF appears on two grids. Only the projected one is documented to carry a
    # Lambert grid_mapping; the regridded UCSD products are geographic and
    # should be judged by the geographic rule instead.
    projected = _is_projected(dataset, array)
    if activity_id == "WRF" and not projected:
        crs_coord = S.EXPECTED_CRS_COORD["LOCA2"]
        if crs_coord not in names and not grid_mapping:
            out.warn(
                "crs.geographic.undeclared",
                "WRF store is on a geographic grid but declares neither a "
                "spatial_ref coordinate nor a grid_mapping, so its datum is "
                "undocumented.",
                expected=crs_coord,
            )
        return out

    if activity_id == "WRF":
        if not grid_mapping:
            out.error(
                "crs.grid_mapping.missing",
                "WRF data is documented to carry a grid_mapping attribute "
                "referencing a Lambert_Conformal coordinate variable.",
                expected=expected_mapping,
            )
        elif grid_mapping != expected_mapping:
            out.warn(
                "crs.grid_mapping.unexpected",
                f"grid_mapping points at {grid_mapping!r}.",
                expected=expected_mapping,
                actual=grid_mapping,
            )
        if crs_coord not in names and grid_mapping not in names:
            out.error(
                "crs.variable.missing",
                f"No {crs_coord!r} variable holding the projection parameters.",
                expected=crs_coord,
                actual=sorted(n for n in names if "conformal" in str(n).lower())
                or None,
            )
        else:
            target = grid_mapping if grid_mapping in names else crs_coord
            out += _check_lambert_params(dataset, target)
    else:  # LOCA2
        if grid_mapping:
            out.info(
                "crs.grid_mapping.unexpected_present",
                "LOCA2 is documented as *not* carrying grid_mapping; CRS lives in "
                "a spatial_ref coordinate instead. Harmless, but a convention drift.",
                expected=None,
                actual=grid_mapping,
            )
        if crs_coord not in names:
            out.error(
                "crs.spatial_ref.missing",
                "LOCA2 stores should carry a spatial_ref coordinate holding the "
                "WGS84 definition.",
                expected=crs_coord,
            )
        else:
            attrs = dataset[crs_coord].attrs
            wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
            if not wkt:
                out.warn(
                    "crs.spatial_ref.empty",
                    "spatial_ref exists but carries no crs_wkt/spatial_ref text.",
                    expected="crs_wkt",
                )
            elif "WGS" not in str(wkt) and "4326" not in str(wkt):
                out.warn(
                    "crs.datum.unexpected",
                    "spatial_ref does not mention WGS84/EPSG:4326.",
                    expected="WGS 84",
                    actual=str(wkt)[:120],
                )
    return out


_LAMBERT_KEYS = (
    "grid_mapping_name",
    "standard_parallel",
    "longitude_of_central_meridian",
    "latitude_of_projection_origin",
)


def _check_lambert_params(dataset: xr.Dataset, name: str) -> FindingList:
    """Check a Lambert_Conformal variable carries its projection parameters.

    Parameters
    ----------
    dataset : xarray.Dataset
        Store holding the projection variable.
    name : str
        Name of that variable.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    try:
        attrs = dict(dataset[name].attrs)
    except KeyError:
        return out
    missing = [k for k in _LAMBERT_KEYS if k not in attrs]
    if missing:
        out.warn(
            "crs.lambert.incomplete",
            f"{name} is missing projection parameter(s) {missing}; the store is "
            "not self-describing enough to reproject without external knowledge.",
            expected=list(_LAMBERT_KEYS),
            actual=sorted(attrs),
        )
    if attrs.get("grid_mapping_name") not in (None, "lambert_conformal_conic"):
        out.warn(
            "crs.lambert.name",
            "grid_mapping_name is not lambert_conformal_conic.",
            expected="lambert_conformal_conic",
            actual=attrs.get("grid_mapping_name"),
        )
    return out


def check_global_attrs(dataset: xr.Dataset) -> FindingList:
    """
    Conventions plus the six recommended global attributes.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    attrs = dict(dataset.attrs)
    lowered = {str(k).lower(): v for k, v in attrs.items()}

    conventions = next(
        (attrs[k] for k in S.CONVENTIONS_ATTR_CANDIDATES if k in attrs), None
    )
    if conventions is None:
        out.error(
            "attr.global.conventions.missing",
            "No Conventions attribute. The standard asks every file to name the "
            "convention it follows, with CF preferred.",
            expected="CF-1.x",
        )
    elif not str(conventions).upper().startswith(S.CONVENTIONS_EXPECTED_PREFIX.upper()):
        out.warn(
            "attr.global.conventions.not_cf",
            f"Conventions is {conventions!r}, not a CF version.",
            expected="CF-1.x",
            actual=conventions,
        )
    else:
        # The value is CF, but capitalisation drifts across the archive
        # ("CF-1.7" vs "cf-1.7", attribute spelled "Conventions" vs
        # "conventions"). Case-sensitive tooling will miss one of them.
        attribute_name = next(k for k in S.CONVENTIONS_ATTR_CANDIDATES if k in attrs)
        if attribute_name != "Conventions" or not str(conventions).startswith("CF-"):
            out.info(
                "attr.global.conventions.case",
                f"Conventions recorded as {attribute_name}={conventions!r}; CF spells "
                "the attribute 'Conventions' and the value 'CF-x.y'.",
                expected="Conventions=CF-1.x",
                actual=f"{attribute_name}={conventions}",
            )

    missing = [a for a in S.RECOMMENDED_GLOBAL_ATTRS if a not in lowered]
    if missing:
        level = Level.WARN if len(missing) >= 4 else Level.INFO
        out.add(
            "attr.global.recommended.missing",
            level,
            f"Missing {len(missing)} of {len(S.RECOMMENDED_GLOBAL_ATTRS)} recommended "
            "global attributes; datasets are supposed to be self-describing.",
            expected=list(S.RECOMMENDED_GLOBAL_ATTRS),
            actual=missing,
        )

    # Present-but-empty is worse than absent: a human reads the blank as a
    # meaningful negative ("not bias-corrected") while a script reads the key as
    # populated. cadcat ships `bias_correction: ''` and `variant_label: ''` this
    # way on stores where the answer is genuinely not blank.
    empty = sorted(
        str(key)
        for key, value in attrs.items()
        if isinstance(value, str) and not value.strip()
    )
    if empty:
        out.warn(
            "attr.global.empty",
            f"{len(empty)} global attribute(s) are present but empty. A reader "
            "cannot tell an empty value from an unanswered question.",
            expected="omit the attribute or give it a value",
            actual=empty,
        )
    return out


def check_coordinate_variables(dataset: xr.Dataset) -> FindingList:
    """
    Coordinates need units + a naming attribute, and no missing values.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    for name in ("lat", "lon", "x", "y"):
        if name not in dataset.coords:
            continue
        coord = dataset[name]
        attrs = dict(coord.attrs)

        units = attrs.get("units")
        accepted = S.EXPECTED_COORD_UNITS.get(name, ())
        if not units:
            out.warn(
                f"coord.{name}.units.missing",
                f"Coordinate {name!r} has no units attribute.",
                expected=accepted[0] if accepted else None,
            )
        elif accepted and str(units) not in accepted:
            out.warn(
                f"coord.{name}.units.unexpected",
                f"Coordinate {name!r} has units {units!r}.",
                expected=list(accepted),
                actual=units,
            )

        if not (attrs.get("long_name") or attrs.get("standard_name")):
            out.info(
                f"coord.{name}.name.missing",
                f"Coordinate {name!r} has neither long_name nor standard_name.",
            )

        try:
            values = np.asarray(coord.values)
            if np.issubdtype(values.dtype, np.floating) and np.isnan(values).any():
                out.error(
                    f"coord.{name}.nan",
                    f"Coordinate {name!r} contains missing values. Coordinate "
                    "variables are required not to.",
                    actual=int(np.isnan(values).sum()),
                )
        except Exception:  # coordinate not readable; reported elsewhere
            pass
    return out


def check_spatial_grid(
    dataset: xr.Dataset, activity_id: str | None, grid_label: str | None
) -> FindingList:
    """
    Grid spacing against the nominal resolution, and a sanity bounding box.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.
    activity_id : str or None
        Downscaling method.
    grid_label : str or None
        Domain, which sets the nominal resolution and plausible extent.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()

    lat_bounds, lon_bounds = S.plausible_extent(grid_label)
    for name, bounds in (("lat", lat_bounds), ("lon", lon_bounds)):
        if name not in dataset.coords:
            continue
        try:
            values = np.asarray(dataset[name].values, dtype="float64")
        except Exception:
            continue
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        low, high = float(finite.min()), float(finite.max())
        if low < bounds[0] or high > bounds[1]:
            out.error(
                f"grid.{name}.out_of_range",
                f"{name} spans {low:.3f}..{high:.3f}, outside the plausible "
                f"extent {bounds} for domain {grid_label}.",
                expected=f"{bounds[0]}..{bounds[1]}",
                actual=f"{low:.3f}..{high:.3f}",
            )

    if activity_id == "WRF" and grid_label in S.NOMINAL_RESOLUTION_M:
        spacing = _uniform_spacing(dataset, "x") or _uniform_spacing(dataset, "y")
        nominal = S.NOMINAL_RESOLUTION_M[grid_label]
        if spacing is not None:
            if abs(spacing - nominal) / nominal > S.RESOLUTION_TOLERANCE:
                out.error(
                    "grid.resolution.mismatch",
                    f"Projected grid spacing is {spacing:,.0f} m but {grid_label} "
                    f"is documented as {nominal:,.0f} m.",
                    expected=f"{nominal:,.0f} m",
                    actual=f"{spacing:,.0f} m",
                )
            else:
                out.ok(
                    "grid.resolution.match",
                    f"Grid spacing {spacing:,.0f} m matches {grid_label}.",
                )
    elif activity_id == "LOCA2":
        spacing = _uniform_spacing(dataset, "lat") or _uniform_spacing(dataset, "lon")
        if spacing is not None:
            nominal = S.LOCA2_NOMINAL_DEG
            if abs(spacing - nominal) / nominal > S.LOCA2_DEG_TOLERANCE:
                out.warn(
                    "grid.resolution.mismatch",
                    f"Geographic grid spacing is {spacing:.5f} deg; LOCA2 3 km is "
                    f"about {nominal:.5f} deg.",
                    expected=f"{nominal:.5f} deg",
                    actual=f"{spacing:.5f} deg",
                )
    return out


def _uniform_spacing(dataset: xr.Dataset, name: str) -> float | None:
    """
    Median absolute step of a 1-D coordinate, or None if not applicable.

    Parameters
    ----------
    dataset : xarray.Dataset
        Store to inspect.
    name : str
        Coordinate name.

    Returns
    -------
    float or None
        Median absolute step, or ``None`` when the coordinate is absent, not
        one-dimensional, or too short to difference.
    """
    if name not in dataset.coords:
        return None
    coord = dataset[name]
    if coord.ndim != 1 or coord.size < 3:
        return None
    try:
        values = np.asarray(coord.values, dtype="float64")
    except Exception:
        return None
    diffs = np.abs(np.diff(values))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


# -----------------------------------------------------------------------
# Time axis
# -----------------------------------------------------------------------


def check_time(
    dataset: xr.Dataset, context: dict[str, Any], aggregated: bool = False
) -> FindingList:
    """
    Decode status, extent, frequency, monotonicity, duplicates, gaps.

    ``aggregated`` marks a dataset that combines several stores. Its time axis
    is the result of aligning members, so gap and leap-day findings are demoted
    to INFO: they describe the alignment rather than any published store.

    Parameters
    ----------
    dataset : xarray.Dataset
        Lazily-opened store.
    context : dict
        Catalog facets identifying the dataset.
    aggregated : bool, optional
        Whether the dataset combines several stores. Default False.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    if "time" not in dataset.coords and "time" not in dataset.dims:
        out.error("time.missing", "No time coordinate.", expected="time")
        return out

    time = dataset["time"]
    values = np.asarray(time.values)
    if values.size == 0:
        out.error("time.empty", "Time coordinate has zero length.")
        return out

    #  decoding
    decoded = np.issubdtype(values.dtype, np.datetime64) or _is_cftime(values)
    if not decoded:
        out.error(
            "time.not_decoded",
            f"Time values are raw {values.dtype} rather than datetimes. Check the "
            "units attribute follows the 'since <reference>' convention.",
            actual=str(values.dtype),
        )
        return out

    calendar = _calendar_of(time)
    if calendar:
        out.info(
            "time.calendar", f"Calendar reported as {calendar!r}.", actual=calendar
        )
    else:
        out.info("time.calendar.unknown", "No calendar attribute on the time axis.")

    years = _years(values)
    first_year, last_year = int(min(years)), int(max(years))

    #  documented bounds, to the day
    key = (context.get("activity_id"), context.get("experiment_id"))
    bounds = S.EXPECTED_TIME_BOUNDS.get(key)
    if bounds and np.issubdtype(values.dtype, np.datetime64):
        first, last = pd.Timestamp(values[0]), pd.Timestamp(values[-1])
        want_first, want_last = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        tol = S.TIME_BOUND_TOLERANCE_DAYS
        off_first = abs((first - want_first).days)
        off_last = abs((last - want_last).days)
        if off_first > tol or off_last > tol:
            out.error(
                "time.bounds.mismatch",
                f"Axis runs {first.date()} to {last.date()}; this activity's "
                f"convention is {want_first.date()} to {want_last.date()} "
                f"(off by {off_first} and {off_last} days). A store on a "
                "different calendar convention from its siblings may also have "
                "its values offset from their timestamps.",
                expected=f"{want_first.date()}..{want_last.date()}",
                actual=f"{first.date()}..{last.date()}",
            )
        else:
            out.ok(
                "time.bounds.match",
                f"Axis {first.date()}..{last.date()} matches the convention.",
            )

    #  documented extent, year granularity
    expected = S.EXPECTED_TIME_RANGE.get(key)
    if expected:
        exp_first, exp_last = expected
        tol = S.TIME_RANGE_TOLERANCE_YEARS
        if abs(first_year - exp_first) > tol or abs(last_year - exp_last) > tol:
            out.error(
                "time.extent.mismatch",
                f"Time axis covers {first_year}-{last_year}; documentation says "
                f"{exp_first}-{exp_last} for {key[0]}/{key[1]}.",
                expected=f"{exp_first}-{exp_last}",
                actual=f"{first_year}-{last_year}",
            )
        else:
            out.ok(
                "time.extent.match",
                f"Time axis {first_year}-{last_year} matches the documented extent.",
            )
    else:
        out.info(
            "time.extent",
            f"Time axis covers {first_year}-{last_year}; no documented extent to "
            "compare against.",
            actual=f"{first_year}-{last_year}",
        )

    #  ordering
    ordinals = _ordinals(values)
    diffs = np.diff(ordinals)
    if np.any(diffs < 0):
        out.error(
            "time.not_monotonic",
            "Time axis is not monotonically increasing.",
            actual=int(np.sum(diffs < 0)),
        )
    n_duplicate = int(np.sum(diffs == 0))
    if n_duplicate:
        out.error(
            "time.duplicates",
            f"Time axis contains {n_duplicate} duplicated timestamp(s).",
            actual=n_duplicate,
        )

    #  frequency and gaps
    table_id = context.get("table_id")
    positive = diffs[diffs > 0]
    if positive.size and table_id in S.EXPECTED_STEP_HOURS:
        low, high = S.EXPECTED_STEP_HOURS[table_id]
        step_hours = positive / 3600.0
        median_step = float(np.median(step_hours))
        if not (low - 1e-6 <= median_step <= high + 1e-6):
            out.error(
                "time.frequency.mismatch",
                f"Median time step is {median_step:.2f} h, which does not match "
                f"table_id={table_id!r}.",
                expected=f"{low}-{high} h",
                actual=f"{median_step:.2f} h",
            )
        else:
            out.ok(
                "time.frequency.match",
                f"Median step {median_step:.2f} h is consistent with {table_id!r}.",
            )

        irregular = int(np.sum((step_hours < low - 1e-6) | (step_hours > high + 1e-6)))
        if irregular:
            biggest = float(step_hours.max())
            out.add(
                "time.gaps",
                Level.INFO if aggregated else Level.WARN,
                f"{irregular} interval(s) fall outside the expected step for "
                f"{table_id!r}; largest is {biggest:.1f} h."
                + (
                    " This dataset came from a reader that aligns and combines "
                    "stores, so the gap describes that processing rather than "
                    "any published store. Re-check with the zarr backend."
                    if aggregated
                    else " Expect missing periods."
                ),
                expected=f"{low}-{high} h",
                actual=f"{irregular} irregular, max {biggest:.1f} h",
            )

    #  leap days
    out += _check_leap_days(values, context, calendar, aggregated=aggregated)
    return out


def _check_leap_days(
    values: np.ndarray,
    context: dict[str, Any],
    calendar: str | None,
    aggregated: bool = False,
) -> FindingList:
    """
    Compare observed Feb 29 presence against calendar and documentation.

    Parameters
    ----------
    values : numpy.ndarray
        Decoded time values.
    context : dict
        Catalog facets identifying the dataset.
    calendar : str or None
        Calendar reported by the time axis.
    aggregated : bool, optional
        Whether the dataset combines several stores. Default False.

    Returns
    -------
    FindingList
        Findings from this check; empty when it does not apply.
    """
    out = FindingList()
    activity = context.get("activity_id")
    source_id = context.get("source_id")
    table_id = context.get("table_id")
    if table_id not in {"1hr", "day"}:
        return out

    try:
        months = np.array([_month_of(v) for v in values])
        days = np.array([_day_of(v) for v in values])
        years = _years(values)
    except Exception:
        return out
    has_leap_day = bool(np.any((months == 2) & (days == 29)))

    # A leap-capable calendar with no 29 February is self-contradictory. The
    # axis and its own metadata disagree, so software that trusts the calendar
    # attribute silently misaligns from every leap year onward.
    leap_capable = calendar is not None and not any(
        marker in str(calendar).lower() for marker in ("noleap", "365", "360")
    )
    spans_leap_year = bool(np.any(_is_leap_year(years)))

    if leap_capable and spans_leap_year and not has_leap_day:
        expected_count = int(np.unique(years[_is_leap_year(years)]).size)
        out.add(
            "time.leap.calendar_contradiction",
            Level.INFO if aggregated else Level.ERROR,
            f"Calendar is {calendar!r}, which has leap days, but the axis contains "
            f"no 29 February across {expected_count} leap year(s) in range. The "
            "time coordinate and its own calendar attribute disagree; resampling "
            "or day-of-year alignment will drift by a day after each leap year.",
            expected=f"29 Feb present ({calendar})",
            actual="no 29 Feb",
        )
        return out

    if activity == "WRF" and source_id in S.WRF_NO_LEAP_MODELS and has_leap_day:
        out.add(
            "time.leap.unexpected",
            Level.INFO if aggregated else Level.WARN,
            f"{source_id} is documented as a no-leap model but the axis contains "
            "29 February.",
            expected="no leap days",
            actual="Feb 29 present",
        )
    elif activity == "WRF" and source_id in S.WRF_LEAP_MODELS and not has_leap_day:
        out.add(
            "time.leap.missing",
            Level.INFO if aggregated else Level.WARN,
            f"{source_id} is documented to retain leap days but none appear.",
            expected="leap days present",
            actual="no Feb 29",
        )
    elif activity == "LOCA2" and not has_leap_day and spans_leap_year:
        out.add(
            "time.leap.missing",
            Level.INFO if aggregated else Level.WARN,
            "LOCA2 models were all interpolated to include leap days, but none "
            "appear on this axis.",
            expected="leap days present",
            actual="no Feb 29",
        )
    if calendar and "360" in str(calendar):
        out.info(
            "time.calendar.360day",
            "360-day calendar; downstream resampling needs explicit handling.",
            actual=calendar,
        )
    return out


def _is_leap_year(years: np.ndarray) -> np.ndarray:
    """
    Test which years are leap years under the Gregorian rules.

    Parameters
    ----------
    years : numpy.ndarray
        Calendar years.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True for leap years.
    """
    years = np.asarray(years)
    return ((years % 4 == 0) & (years % 100 != 0)) | (years % 400 == 0)


#  time helpers


def _is_cftime(values: np.ndarray) -> bool:
    """
    Test whether a decoded time array holds cftime objects.

    Parameters
    ----------
    values : numpy.ndarray
        Decoded time values.

    Returns
    -------
    bool
        True when the array holds cftime objects rather than ``datetime64``.
    """
    if values.dtype != object or values.size == 0:
        return False
    return hasattr(values.flat[0], "calendar") or type(
        values.flat[0]
    ).__module__.startswith("cftime")


def _calendar_of(time: xr.DataArray) -> str | None:
    """
    Read the calendar from a time coordinate's attrs, encoding or values.

    Parameters
    ----------
    time : xarray.DataArray
        Time coordinate.

    Returns
    -------
    str or None
        Calendar name, or ``None`` when none can be determined. A
        ``datetime64`` axis reports ``proleptic_gregorian``, which is what
        xarray decoded it as.
    """
    for source in (time.attrs, time.encoding):
        if "calendar" in source:
            return str(source["calendar"])
    values = np.asarray(time.values)
    if _is_cftime(values):
        return str(getattr(values.flat[0], "calendar", None))
    if np.issubdtype(values.dtype, np.datetime64):
        return "proleptic_gregorian"
    return None


def _years(values: np.ndarray) -> np.ndarray:
    """
    Extract the year of each timestamp, for numpy or cftime arrays.

    Parameters
    ----------
    values : numpy.ndarray
        Decoded time values.

    Returns
    -------
    numpy.ndarray
        Calendar year of each timestamp.
    """
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[Y]").astype(int) + 1970
    return np.array([v.year for v in values])


def _month_of(value: Any) -> int:
    """
    Extract the month of one timestamp.

    Parameters
    ----------
    value : Any
        One timestamp, ``datetime64`` or cftime.

    Returns
    -------
    int
        Month, 1-12.
    """
    if isinstance(value, np.datetime64):
        return int(pd.Timestamp(value).month)
    return int(value.month)


def _day_of(value: Any) -> int:
    """
    Extract the day of month of one timestamp.

    Parameters
    ----------
    value : Any
        One timestamp, ``datetime64`` or cftime.

    Returns
    -------
    int
        Day of month.
    """
    if isinstance(value, np.datetime64):
        return int(pd.Timestamp(value).day)
    return int(value.day)


def _ordinals(values: np.ndarray) -> np.ndarray:
    """
    Seconds since epoch, for both numpy and cftime axes.

    Parameters
    ----------
    values : numpy.ndarray
        Decoded time values, numpy or cftime.

    Returns
    -------
    numpy.ndarray
        Seconds since an epoch, as float64, so consecutive differences are
        comparable across both time representations.
    """
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[s]").astype("int64").astype("float64")
    import cftime  # noqa: F401  (only needed on this branch)

    reference = values.flat[0]
    unit = "seconds since 1900-01-01"
    import cftime as _cftime

    return np.asarray(
        _cftime.date2num(
            list(values), unit, calendar=getattr(reference, "calendar", "standard")
        ),
        dtype="float64",
    )
