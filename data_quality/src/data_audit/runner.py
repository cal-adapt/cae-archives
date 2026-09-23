"""
Orchestration: turn a catalog subset into a table of findings.

Three entry points, roughly in order of cost:

``audit_catalog``
    No network beyond loading the catalog. Vocabulary, path conventions,
    coverage gaps. Run this first; it is seconds, and it finds a surprising
    amount.

``sweep``
    Opens each sampled store with one backend and runs the dataset checks.
    Minutes to hours depending on the plan size.

``compare_backends``
    Opens each sampled store with *both* backends and reports where the metadata
    climakitae presents differs from what is actually in S3.
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd

from . import catalog as catalog_module
from . import checks
from . import standards as S
from .findings import Finding, FindingList, Level

logger = logging.getLogger(__name__)

CONTEXT_COLUMNS = list(S.FACETS) + ["path"]


# --------------------------------------------------------------------------
# Catalog audit
# --------------------------------------------------------------------------


def audit_catalog(frame: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    """
    Validate the catalog without opening any data.

    Parameters
    ----------
    frame : pandas.DataFrame
        Catalog records, from :func:`data_audit.catalog.load_catalog`.
    max_rows : int, optional
        Cap the number of records validated row by row. Coverage findings are
        always computed over the whole frame, since they are about what is
        absent.

    Returns
    -------
    pandas.DataFrame
        Findings table: per-record vocabulary and path checks, plus
        catalog-wide coverage and facet-consistency findings.
    """
    rows = frame if max_rows is None else frame.head(max_rows)
    records: list[dict[str, Any]] = []

    for row in rows.to_dict("records"):
        for finding in checks.check_catalog_row(row):
            finding.check = "check_catalog_row"
            finding.context = _context(row)
            records.append(finding.as_row())

    for finding in _coverage_findings(frame):
        records.append(finding.as_row())

    return _as_frame(records)


def _coverage_findings(frame: pd.DataFrame) -> list[Finding]:
    """
    Derive the catalog-wide findings that no single record can show.

    Parameters
    ----------
    frame : pandas.DataFrame
        Catalog records.

    Returns
    -------
    list of Finding
        Documented variables absent everywhere, variables published at an
        undocumented resolution, variables undocumented at any resolution,
        grid-coverage asymmetry, and partially-populated facets.
    """
    out: list[Finding] = []

    missing = catalog_module.documented_but_absent(frame)
    for row in missing.to_dict("records"):
        out.append(
            Finding(
                code="coverage.variable.absent",
                level=Level.WARN,
                message=(
                    f"{row['variable_id']!r} is documented for "
                    f"{row['activity_id']}/{row['table_id']} but no store exists at "
                    "any grid label."
                ),
                expected=row["variable_id"],
                actual=None,
                check="coverage",
                context={
                    "activity_id": row["activity_id"],
                    "table_id": row["table_id"],
                    "variable_id": row["variable_id"],
                },
            )
        )

    asymmetric = catalog_module.grid_coverage_asymmetry(frame)
    for row in asymmetric.to_dict("records"):
        out.append(
            Finding(
                code="coverage.grid.asymmetric",
                level=Level.INFO,
                message=(
                    f"{row['variable_id']!r} at {row['activity_id']}/{row['table_id']} "
                    f"exists at {row['grids_present']} but not {row['grids_missing']}. "
                    "Often intentional; confirm it is."
                ),
                expected=row["grids_present"],
                actual=row["grids_missing"],
                check="coverage",
                context={
                    "activity_id": row["activity_id"],
                    "table_id": row["table_id"],
                    "variable_id": row["variable_id"],
                },
            )
        )

    for row in catalog_module.facet_consistency(frame).to_dict("records"):
        out.append(
            Finding(
                code="catalog.facet.partial",
                level=Level.INFO,
                message=(
                    f"{row['activity_id']}: facet {row['facet']!r} is blank on "
                    f"{row['n_blank']:,} of {row['n_blank'] + row['n_populated']:,} "
                    f"records ({row['pct_blank']}%). Queries that filter on it will "
                    "silently drop the blank ones."
                ),
                expected="populated on every record",
                actual=f"{row['pct_blank']}% blank",
                check="coverage",
                context={"activity_id": row["activity_id"]},
            )
        )

    for row in catalog_module.timescale_mismatch(frame).to_dict("records"):
        out.append(
            Finding(
                code="coverage.variable.timescale_mismatch",
                level=Level.WARN,
                message=(
                    f"{row['variable_id']!r} is published at "
                    f"{row['activity_id']}/{row['published_at']} but the reference "
                    f"variable table lists it only at {row['documented_at']}. "
                    "climakitae will find no unit or display name for it here."
                ),
                expected=row["documented_at"],
                actual=row["published_at"],
                check="coverage",
                context={
                    "activity_id": row["activity_id"],
                    "table_id": row["published_at"],
                    "variable_id": row["variable_id"],
                },
            )
        )

    extra = catalog_module.undocumented_in_catalog(frame)
    for row in extra.to_dict("records"):
        out.append(
            Finding(
                code="coverage.variable.undocumented",
                level=Level.INFO,
                message=(
                    f"{row['n_stores']} store(s) hold {row['variable_id']!r} at "
                    f"{row['activity_id']}/{row['table_id']}, which the reference "
                    "variable table does not describe."
                ),
                actual=row["variable_id"],
                check="coverage",
                context={
                    "activity_id": row["activity_id"],
                    "table_id": row["table_id"],
                    "variable_id": row["variable_id"],
                },
            )
        )
    return out


# --------------------------------------------------------------------------
# Store sweep
# --------------------------------------------------------------------------


def sweep(
    plan: pd.DataFrame,
    backend: Any,
    inspect_store: bool = False,
    check_time_axis: bool = True,
    probe_values: bool = False,
    max_workers: int = 4,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Open every row in ``plan`` with ``backend`` and collect findings.

    Parameters
    ----------
    plan
        Catalog rows, typically from :func:`catalog.build_sample_plan`.
    backend : Any
        Anything with ``.open(row) -> (Dataset | None, diagnostics)``.
        Optionally ``.inspect_store(row) -> FindingList``.
    inspect_store : bool, optional
        Also run the backend's raw store inspection, if it has one.
    check_time_axis : bool, optional
        Run the time-axis checks. Default True.
    probe_values : bool, optional
        Read a few slices per store to catch empty variables. Default False.
    max_workers : int, optional
        Threads. Opening a Zarr store is I/O bound, so several help; going much
        above 8 tends to get throttled by S3.
    progress : bool, optional
        Log progress every 25 stores. Default True.

    Returns
    -------
    pandas.DataFrame
        One row per finding, with facet columns, ``code``, ``level``,
        ``message``, ``expected``, ``actual`` and ``elapsed_s``.
    """
    rows = plan.to_dict("records")
    records: list[dict[str, Any]] = []
    total = len(rows)

    def work(row: dict[str, Any]) -> list[dict[str, Any]]:
        return _inspect_one(row, backend, inspect_store, check_time_axis, probe_values)

    if max_workers <= 1:
        iterator: Iterable[list[dict[str, Any]]] = (work(r) for r in rows)
        for index, result in enumerate(iterator, start=1):
            records.extend(result)
            if progress:
                _log_progress(index, total)
    else:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for index, result in enumerate(pool.map(work, rows), start=1):
                records.extend(result)
                if progress:
                    _log_progress(index, total)

    return _as_frame(records)


def _log_progress(index: int, total: int) -> None:
    """
    Log sweep progress at a readable interval.

    Parameters
    ----------
    index : int
        Stores inspected so far, 1-based.
    total : int
        Stores in the plan.
    """
    if index == total or index % 25 == 0:
        logger.info("inspected %d/%d stores", index, total)


def _inspect_one(
    row: dict[str, Any],
    backend: Any,
    inspect_store: bool,
    check_time_axis: bool,
    probe_values: bool = False,
) -> list[dict[str, Any]]:
    """
    Open and check one store, converting every outcome into findings.

    Parameters
    ----------
    row : dict
        Catalog record.
    backend : Any
        Reader with an ``open(row)`` method.
    inspect_store : bool
        Also run the backend's raw store inspection, if it has one.
    check_time_axis : bool
        Run the time-axis checks.
    probe_values : bool, optional
        Read a few slices to catch empty variables. Default False.

    Returns
    -------
    list of dict
        Finding records, each carrying ``elapsed_s`` for the whole inspection.
        A store that cannot be opened yields one FATAL record rather than
        raising, so a sweep survives a malformed store.
    """
    context = _context(row)
    context["backend"] = getattr(backend, "name", type(backend).__name__)
    started = time.perf_counter()
    found = FindingList()

    if inspect_store and hasattr(backend, "inspect_store"):
        try:
            found += backend.inspect_store(row)
        except Exception as exc:
            found.error("store.inspect.failed", f"{type(exc).__name__}: {exc}")

    dataset = None
    try:
        dataset, diagnostics = backend.open(row)
    except Exception as exc:
        diagnostics = {"error": f"{type(exc).__name__}: {exc}"}

    if dataset is None:
        found.fatal(
            "open.failed",
            "Could not open this store: "
            + str(diagnostics.get("error", "no dataset returned")),
            actual=diagnostics.get("error"),
        )
    else:
        n_returned = diagnostics.get("n_returned")
        if isinstance(n_returned, int) and n_returned > 1:
            found.warn(
                "query.ambiguous",
                f"A fully-specified query returned {n_returned} datasets, so the "
                "catalog holds duplicate records for these facets.",
                expected=1,
                actual=diagnostics.get("returned_keys"),
            )
        try:
            found += checks.check_dataset(
                dataset,
                context,
                check_time_axis=check_time_axis,
                probe_values=probe_values,
                authoritative_structure=getattr(
                    backend, "authoritative_structure", True
                ),
            )
        except Exception as exc:
            found.error("checks.failed", f"{type(exc).__name__}: {exc}")
        finally:
            try:
                dataset.close()
            except Exception:
                pass

    elapsed = time.perf_counter() - started
    out: list[dict[str, Any]] = []
    for finding in found:
        finding.check = finding.check or "check_dataset"
        finding.context = context
        record = finding.as_row()
        record["elapsed_s"] = round(elapsed, 3)
        out.append(record)
    return out


# --------------------------------------------------------------------------
# Backend comparison
# --------------------------------------------------------------------------

#: What to compare between the two access paths.
_COMPARED_FIELDS = (
    "units",
    "long_name",
    "standard_name",
    "grid_mapping",
    "dtype",
    "shape",
    "time_start",
    "time_end",
    "n_time",
    "conventions",
    "n_global_attrs",
)


def compare_backends(
    plan: pd.DataFrame,
    backend_a: Any,
    backend_b: Any,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Open each row with both backends and diff the metadata they expose.

    Differences are expected and informative rather than automatically wrong:
    climakitae attaches a CRS to stores that lack one, so ``grid_mapping``
    present in A and absent in B means the *store* is missing it. That is the
    single most useful column in this table.

    Parameters
    ----------
    plan : pandas.DataFrame
        Catalog rows to compare.
    backend_a : Any
        First reader, conventionally ``ClimakitaeBackend``.
    backend_b : Any
        Second reader, conventionally ``ZarrBackend``.
    max_workers : int, optional
        Threads. Each row opens two stores. Default 4.

    Returns
    -------
    pandas.DataFrame
        One row per (dataset, field) with ``value_a``, ``value_b`` and
        ``agree``.

    Notes
    -----
    For a verdict that attributes a divergence to one surface or the other, use
    :func:`data_audit.reconcile.reconcile` instead; this function only reports
    that the two views differ.
    """
    rows = plan.to_dict("records")

    def work(row: dict[str, Any]) -> list[dict[str, Any]]:
        context = _context(row)
        summary_a = _summarize(backend_a, row)
        summary_b = _summarize(backend_b, row)
        out: list[dict[str, Any]] = []
        for field in _COMPARED_FIELDS:
            value_a, value_b = summary_a.get(field), summary_b.get(field)
            record = dict(context)
            record.update(
                {
                    "field": field,
                    "backend_a": getattr(backend_a, "name", "a"),
                    "backend_b": getattr(backend_b, "name", "b"),
                    "value_a": _text(value_a),
                    "value_b": _text(value_b),
                    "agree": _text(value_a) == _text(value_b),
                }
            )
            out.append(record)
        return out

    records: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(work, rows):
            records.extend(result)
    return pd.DataFrame(records)


def _summarize(backend: Any, row: dict[str, Any]) -> dict[str, Any]:
    """
    Pull the comparable fields out of whatever the backend opened.

    Parameters
    ----------
    backend : Any
        Reader with an ``open(row)`` method.
    row : dict
        Catalog record.

    Returns
    -------
    dict
        The fields named in :data:`_COMPARED_FIELDS`, or a single ``error`` key
        when the store could not be opened.
    """
    try:
        dataset, _ = backend.open(row)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if dataset is None:
        return {"error": "open returned None"}

    try:
        variable_id = row.get("variable_id")
        name = checks._find_data_var(dataset, variable_id)
        summary: dict[str, Any] = {}
        if name is not None:
            array = dataset[name]
            summary.update(
                {
                    "units": array.attrs.get("units"),
                    "long_name": array.attrs.get("long_name"),
                    "standard_name": array.attrs.get("standard_name"),
                    "grid_mapping": array.attrs.get("grid_mapping"),
                    "dtype": str(array.dtype),
                    "shape": tuple(array.shape),
                }
            )
        if "time" in dataset.coords:
            time = dataset["time"].values
            summary["n_time"] = int(time.size)
            if time.size:
                summary["time_start"] = str(time[0])[:19]
                summary["time_end"] = str(time[-1])[:19]
        attrs = dataset.attrs
        summary["conventions"] = next(
            (attrs[k] for k in S.CONVENTIONS_ATTR_CANDIDATES if k in attrs), None
        )
        summary["n_global_attrs"] = len(attrs)
        return summary
    finally:
        try:
            dataset.close()
        except Exception:
            pass


def _text(value: Any) -> str | None:
    """Render a value for comparison, preserving ``None``.

    Parameters
    ----------
    value : Any
        Value to render.

    Returns
    -------
    str or None
        ``None`` passes through so that "absent" and the string ``"None"`` stay
        distinguishable.
    """
    return None if value is None else str(value)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _context(row: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the facet columns that identify a dataset.

    Parameters
    ----------
    row : dict
        Catalog record.

    Returns
    -------
    dict
        The columns in :data:`CONTEXT_COLUMNS`, missing ones set to ``None``.
    """
    return {column: row.get(column) for column in CONTEXT_COLUMNS}


def _as_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble finding records into a tidy table with stable column order.

    Parameters
    ----------
    records : list of dict
        Finding records.

    Returns
    -------
    pandas.DataFrame
        Facet columns first, then the finding columns. An empty input yields an
        empty frame with the full column set, so downstream code can rely on
        the schema.
    """
    if not records:
        columns = CONTEXT_COLUMNS + [
            "check",
            "code",
            "level",
            "level_num",
            "message",
            "expected",
            "actual",
        ]
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records)
    ordered = [c for c in CONTEXT_COLUMNS if c in frame.columns]
    tail = [c for c in frame.columns if c not in ordered]
    return frame[ordered + tail]
