"""
Reconcile the two surfaces against the documentation.

The archive has two faces and they are not the same artifact:

* the **published store** in ``s3://cadcat``, read directly;
* the **delivered product**, what ``climakitae.ClimateData`` hands a user.

climakitae is not merely a viewer. It repairs some things on read — it attaches
a CRS to cadcat stores that lack one — and it changes others, aligning and
combining matching stores so the result's time axis may differ from any store's.
Both surfaces are real. A user of the Analytics Engine never sees the store, so
a defect climakitae repairs does not reach them; equally, a defect climakitae
*introduces* reaches every one of them regardless of how clean the store is.

Judging either surface alone gives a misleading answer, so this module runs the
same checks against both and classifies every divergence:

``compliant``
    Neither surface trips the check.
``store_defect_repaired``
    The store trips it, the delivered product does not. Invisible to users;
    still wrong in the artifact, and still a problem for anyone reading the
    Zarr directly or for any future reader that does not apply the same repair.
``introduced_by_reader``
    The store is clean, the delivered product is not. Reaches every user. The
    store is not the thing to fix.
``both``
    Both surfaces trip it. Fix at the source.

Every row carries the documentation requirement the check enforces, so a
reviewer can go from a finding to the sentence it came from.
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
from typing import Any

import pandas as pd

from . import checks
from . import standards as S
from .findings import Level

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Which documented requirement each check enforces
# --------------------------------------------------------------------------

DOCS = {
    "metadata": (
        "Metadata Standards",
        "https://analytics.cal-adapt.org/data-tools/data-documentation/metadata-standards.html",
    ),
    "structure": (
        "Data Structure and Format",
        "https://analytics.cal-adapt.org/data-tools/data-documentation/data-structure-and-format.html",
    ),
    "sims": (
        "Climate Model Simulations",
        "https://analytics.cal-adapt.org/data-tools/data-documentation/climate-model-sims.html",
    ),
    "variables": (
        "climakitae variable_descriptions.csv",
        "https://github.com/cal-adapt/climakitae",
    ),
}

#: Longest matching code prefix wins, so specific entries may precede general.
REQUIREMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "attr.var.units",
        "metadata",
        "Variable attributes must contain units at minimum.",
    ),
    (
        "attr.var.long_name",
        "metadata",
        "Variable attributes must contain a long name descriptor, descriptive "
        "enough to label plots.",
    ),
    (
        "attr.var.standard_name",
        "metadata",
        "CF conventions are preferred; standard_name is part of CF.",
    ),
    (
        "attr.var.cell_methods",
        "metadata",
        "CF conventions are preferred; cell_methods documents temporal reduction.",
    ),
    (
        "attr.var.fill",
        "metadata",
        "A fill value or valid_range must be applied to any variable with "
        "missing data.",
    ),
    (
        "attr.global.conventions",
        "metadata",
        "Each file must list the standard convention used to organize the data; "
        "CF Convention is preferred.",
    ),
    (
        "attr.global.recommended",
        "metadata",
        "Recommended global attributes: title, institution, source, references, "
        "history, comment.",
    ),
    (
        "attr.global.empty",
        "metadata",
        "Datasets should be self-describing; an attribute present but empty "
        "describes nothing.",
    ),
    (
        "units",
        "variables",
        "Units should use industry standard (udunits) names for interoperability.",
    ),
    (
        "long_name",
        "variables",
        "Variable display names should match the published variable table.",
    ),
    (
        "crs",
        "sims",
        "WRF carries a grid_mapping attribute referencing a Lambert_Conformal "
        "coordinate variable; LOCA2-Hybrid stores CRS in a spatial_ref "
        "coordinate at the global level.",
    ),
    (
        "coord",
        "metadata",
        "Dimensions of time and space must be standalone variables, each "
        "self-described with units and a naming attribute, and must not contain "
        "missing values.",
    ),
    (
        "grid.resolution",
        "sims",
        "Spatial resolution: 45 km (d01), 9 km (d02, WECC), 3 km (d03, CA).",
    ),
    (
        "grid.convention",
        "sims",
        "Coordinate reference systems and grid conventions per downscaling method.",
    ),
    ("grid", "sims", "Grid conventions and documented domain extents."),
    (
        "time.extent",
        "sims",
        "WRF historical 1980-2014, LOCA2 historical 1950-2014, projections "
        "through 2100.",
    ),
    (
        "time.leap",
        "sims",
        "Documented leap-day behaviour per model; LOCA2 models were interpolated "
        "to include leap days.",
    ),
    (
        "time.frequency",
        "structure",
        "Temporal resolution must match the table_id the store is catalogued " "under.",
    ),
    (
        "time",
        "metadata",
        "The time variable must represent time elapsed since a reference date; "
        "Gregorian is preferred.",
    ),
    (
        "var.",
        "variables",
        "The variable in the store must carry the name it is catalogued under.",
    ),
    (
        "wind.rotation",
        "sims",
        "Wind components must declare their reference frame; grid-relative and "
        "earth-relative are not interchangeable.",
    ),
    (
        "data.",
        "metadata",
        "Datasets should be self-describing and usable; a store with no values "
        "is neither.",
    ),
    (
        "store.",
        "structure",
        "Zarr stores should be cloud-optimised, with consolidated metadata for "
        "efficient random access.",
    ),
    (
        "dims",
        "sims",
        "Expected dimensions for the downscaling method and grid.",
    ),
    (
        "dataset.aggregated",
        "structure",
        "A query pinning every available facet should identify one store.",
    ),
)


def requirement_for(code: str) -> tuple[str, str, str]:
    """
    Look up the documented requirement a check enforces.

    Parameters
    ----------
    code : str
        Check code, e.g. ``"crs.spatial_ref.missing"``. The longest matching
        prefix in :data:`REQUIREMENTS` wins, so specific entries override
        general ones.

    Returns
    -------
    requirement : str
        The documented rule, in prose. Empty when the code is unrecognised.
    doc_title : str
        Title of the documentation page.
    doc_url : str
        Link to that page.
    """
    best = ""
    for prefix, _, _ in REQUIREMENTS:
        if code.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    for prefix, doc_key, text in REQUIREMENTS:
        if prefix == best:
            title, url = DOCS[doc_key]
            return text, title, url
    return "", "", ""


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------

VERDICTS = (
    "introduced_by_reader",
    "both",
    "store_defect_repaired",
    "compliant",
)

#: Severity floor per verdict, so a repaired store defect is never silently
#: dropped just because the user-facing surface looks fine.
_VERDICT_LEVEL = {
    "introduced_by_reader": "ERROR",
    "both": "ERROR",
    "store_defect_repaired": "WARN",
    "compliant": "OK",
}


def reconcile(
    plan: pd.DataFrame,
    store_backend: Any,
    delivered_backend: Any,
    check_time_axis: bool = True,
    probe_values: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Run the same checks on both surfaces and classify every divergence.

    Parameters
    ----------
    plan
        Catalog rows, from :func:`catalog.build_sample_plan`.
    store_backend : Any
        Reads published stores directly, normally ``ZarrBackend``.
    delivered_backend : Any
        Reads what users receive, normally ``ClimakitaeBackend``.
    check_time_axis : bool, optional
        Run the time-axis checks. Default True.
    probe_values : bool, optional
        Read a few slices per store to catch empty variables. Default False.
    max_workers : int, optional
        Threads. Each row opens two stores, so this doubles the concurrent
        requests. Default 4.

    Returns
    -------
    pandas.DataFrame
        One row per (dataset, check code) that fired on either surface, plus
        one per check that passed on both. Carries ``verdict``, ``in_store``,
        ``in_delivered``, the severity each surface reported, and the
        documentation requirement behind the check. ``verdict`` is an ordered
        categorical so sorting puts reader-introduced defects first.
    """
    rows = plan.to_dict("records")

    def work(row: dict[str, Any]) -> list[dict[str, Any]]:
        return _reconcile_one(
            row, store_backend, delivered_backend, check_time_axis, probe_values
        )

    records: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(work, rows):
            records.extend(result)

    if not records:
        return pd.DataFrame(
            columns=list(S.FACETS)
            + [
                "code",
                "verdict",
                "in_store",
                "in_delivered",
                "level",
                "store_level",
                "delivered_level",
                "requirement",
                "doc",
                "doc_url",
                "store_message",
                "delivered_message",
            ]
        )
    frame = pd.DataFrame(records)
    frame["verdict"] = pd.Categorical(frame["verdict"], VERDICTS, ordered=True)
    return frame.sort_values(["verdict", "code"]).reset_index(drop=True)


def _reconcile_one(
    row: dict[str, Any],
    store_backend: Any,
    delivered_backend: Any,
    check_time_axis: bool,
    probe_values: bool,
) -> list[dict[str, Any]]:
    """
    Compare one dataset across both surfaces.

    Parameters
    ----------
    row : dict
        Catalog record.
    store_backend : Any
        Reader for the published store.
    delivered_backend : Any
        Reader for the delivered product.
    check_time_axis : bool
        Run the time-axis checks.
    probe_values : bool
        Read a few slices to catch empty variables.

    Returns
    -------
    list of dict
        One record per divergence, plus one per check passing on both surfaces.
        A single ``open.failed`` record when neither surface could be opened.
    """
    context = {column: row.get(column) for column in list(S.FACETS) + ["path"]}

    store_result = _findings_for(
        store_backend, row, context, check_time_axis, probe_values
    )
    delivered_result = _findings_for(
        delivered_backend, row, context, check_time_axis, probe_values
    )

    if store_result is None and delivered_result is None:
        return [
            dict(
                context,
                code="open.failed",
                verdict="both",
                in_store=True,
                in_delivered=True,
                level="FATAL",
                store_level="FATAL",
                delivered_level="FATAL",
                requirement="Data must be readable from both surfaces.",
                doc=DOCS["structure"][0],
                doc_url=DOCS["structure"][1],
                store_message="could not open",
                delivered_message="could not open",
            )
        ]

    store, store_passes = store_result or ({}, set())
    delivered, delivered_passes = delivered_result or ({}, set())

    records: list[dict[str, Any]] = []
    for code in sorted(set(store) | set(delivered)):
        in_store = code in store
        in_delivered = code in delivered
        if in_store and in_delivered:
            verdict = "both"
        elif in_delivered:
            verdict = "introduced_by_reader"
        else:
            verdict = "store_defect_repaired"

        requirement, doc, doc_url = requirement_for(code)
        records.append(
            dict(
                context,
                code=code,
                verdict=verdict,
                in_store=in_store,
                in_delivered=in_delivered,
                level=_VERDICT_LEVEL[verdict],
                store_level=store.get(code, (None, None))[0],
                delivered_level=delivered.get(code, (None, None))[0],
                requirement=requirement,
                doc=doc,
                doc_url=doc_url,
                store_message=store.get(code, (None, None))[1],
                delivered_message=delivered.get(code, (None, None))[1],
            )
        )

    # Checks that passed on both surfaces. Without these the report has no
    # denominator and reads as though nothing in the archive complies.
    for code in sorted((store_passes & delivered_passes) - set(store) - set(delivered)):
        requirement, doc, doc_url = requirement_for(code)
        records.append(
            dict(
                context,
                code=code,
                verdict="compliant",
                in_store=False,
                in_delivered=False,
                level="OK",
                store_level="OK",
                delivered_level="OK",
                requirement=requirement,
                doc=doc,
                doc_url=doc_url,
                store_message=None,
                delivered_message=None,
            )
        )
    return records


def _findings_for(
    backend: Any,
    row: dict[str, Any],
    context: dict[str, Any],
    check_time_axis: bool,
    probe_values: bool,
) -> tuple[dict[str, tuple[str, str]], set[str]] | None:
    """
    Run the dataset checks against one surface.

    Parameters
    ----------
    backend : Any
        Reader with an ``open(row)`` method.
    row : dict
        Catalog record.
    context : dict
        Facets identifying the dataset, passed to the checks.
    check_time_axis : bool
        Run the time-axis checks.
    probe_values : bool
        Read a few slices to catch empty variables.

    Returns
    -------
    tuple of (dict, set) or None
        ``({code: (level, message)}, passing_codes)``, or ``None`` when the
        store could not be opened or the checks raised.

    Notes
    -----
    Note ``authoritative_structure=True`` for *both* backends. The demotion that
    :func:`checks.check_dataset` applies to non-authoritative readers exists to
    stop a single-backend sweep blaming the archive for the reader's own
    aggregation. Here both surfaces are examined together and the verdict does
    that attribution properly, so demoting first would only hide reader-
    introduced defects — which is the class this module exists to surface.
    """
    try:
        dataset, _ = backend.open(row)
    except Exception as exc:
        logger.debug("%s failed to open: %s", getattr(backend, "name", "?"), exc)
        return None
    if dataset is None:
        return None

    try:
        found = checks.check_dataset(
            dataset,
            context,
            check_time_axis=check_time_axis,
            probe_values=probe_values,
            authoritative_structure=True,
        )
    except Exception as exc:
        logger.debug("checks failed: %s", exc)
        return None
    finally:
        try:
            dataset.close()
        except Exception:
            pass

    problems = {
        finding.code: (finding.level.label, finding.message)
        for finding in found
        if finding.level >= Level.WARN
    }
    # Codes that explicitly passed, so "complies on both surfaces" is countable
    # rather than merely inferred from the absence of a complaint.
    passes = {finding.code for finding in found if finding.level == Level.OK} - set(
        problems
    )
    return problems, passes


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def summarize(reconciliation: pd.DataFrame) -> pd.DataFrame:
    """
    Count rows per verdict and code, with the requirement attached.

    Parameters
    ----------
    reconciliation : pandas.DataFrame
        Output of :func:`reconcile`.

    Returns
    -------
    pandas.DataFrame
        Columns ``verdict``, ``code``, ``n``, ``requirement`` and ``doc``,
        ordered by verdict severity then count. Empty in, empty out.
    """
    if reconciliation.empty:
        return pd.DataFrame()
    grouped = reconciliation.groupby(["verdict", "code"], observed=True)
    return (
        grouped.agg(
            n=("code", "size"),
            requirement=("requirement", "first"),
            doc=("doc", "first"),
        )
        .reset_index()
        .sort_values(["verdict", "n"], ascending=[True, False])
        .reset_index(drop=True)
    )


_VERDICT_PROSE = {
    "introduced_by_reader": (
        "Introduced by climakitae",
        "The published store satisfies the requirement; what users receive does "
        "not. Every Analytics Engine user is affected regardless of the store's "
        "state, and fixing the store would change nothing.",
    ),
    "both": (
        "Non-compliant on both surfaces",
        "Neither the store nor the delivered product meets the documented "
        "requirement. Fix at the source.",
    ),
    "store_defect_repaired": (
        "Repaired by climakitae",
        "The store does not meet the requirement, but climakitae compensates on "
        "read, so users do not see it. Still a defect in the published artifact: "
        "it affects anyone reading the Zarr directly, and it depends on the "
        "repair continuing to exist.",
    ),
}


def render_markdown(
    reconciliation: pd.DataFrame,
    title: str = "cadcat: documentation compliance across both surfaces",
) -> str:
    """
    Render a reviewer-facing report organised by verdict.

    Parameters
    ----------
    reconciliation : pandas.DataFrame
        Output of :func:`reconcile`.
    title : str, optional
        Heading for the report.

    Returns
    -------
    str
        Markdown with a verdict tally, a section per non-compliant verdict
        naming the documented requirement behind each code, and links to the
        documentation pages. Compliant checks are counted but not listed.
    """
    lines: list[str] = [f"# {title}", ""]
    if reconciliation.empty:
        lines.append("No divergences recorded.")
        return "\n".join(lines)

    n_datasets = reconciliation[list(S.FACETS)].astype(str).drop_duplicates().shape[0]
    lines += [
        f"Compared **{n_datasets:,}** dataset(s) across two surfaces: the Zarr "
        "stores published in `s3://cadcat`, and the objects "
        "`climakitae.ClimateData` returns to users.",
        "",
        "Both are judged against the Cal-Adapt documentation. A requirement can "
        "fail on one surface and pass on the other, which is why each row records "
        "where it failed rather than a single verdict.",
        "",
        "| verdict | rows |",
        "| --- | ---: |",
    ]
    counts = reconciliation["verdict"].value_counts()
    for verdict in VERDICTS:
        if verdict in counts:
            lines.append(f"| {verdict} | {counts[verdict]:,} |")
    lines.append("")

    n_compliant = int((reconciliation["verdict"] == "compliant").sum())
    if n_compliant:
        lines += [
            f"{n_compliant:,} check(s) passed identically on both surfaces and "
            "are listed in the CSV with verdict `compliant`; the sections below "
            "cover only divergences and shared failures.",
            "",
        ]

    summary = summarize(reconciliation)
    for verdict in ("introduced_by_reader", "both", "store_defect_repaired"):
        subset = summary[summary["verdict"] == verdict]
        if subset.empty:
            continue
        heading, prose = _VERDICT_PROSE[verdict]
        lines += [f"## {heading}", "", prose, ""]
        lines += [
            "| code | n | documented requirement | source |",
            "| --- | ---: | --- | --- |",
        ]
        for record in subset.to_dict("records"):
            requirement = str(record["requirement"]).replace("|", "\\|")
            lines.append(
                f"| `{record['code']}` | {record['n']:,} | {requirement} | "
                f"{record['doc']} |"
            )
        lines.append("")

    lines += ["## Documentation", ""]
    for _, (doc_title, url) in DOCS.items():
        lines.append(f"- [{doc_title}]({url})")
    lines.append("")
    return "\n".join(lines)
