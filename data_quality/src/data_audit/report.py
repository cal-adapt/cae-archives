"""
Turn a findings table into something a person will actually read.
"""

from __future__ import annotations

import datetime as dt
import html
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

SEVERITY_ORDER = ["FATAL", "ERROR", "WARN", "INFO", "OK"]


def summarize_by_code(findings: pd.DataFrame) -> pd.DataFrame:
    """
    Count findings per check code: what is wrong, and how often.

    Parameters
    ----------
    findings : pandas.DataFrame
        Findings table, as returned by the runner.

    Returns
    -------
    pandas.DataFrame
        Columns ``code``, ``level``, ``n``, ``n_datasets`` and ``example``,
        ordered worst-severity first and then by count. ``level`` is an ordered
        categorical, so sorting it groups FATAL before ERROR before WARN.
    """
    if findings.empty:
        return pd.DataFrame(columns=["code", "level", "n", "n_datasets", "example"])
    keys = [
        c
        for c in ("activity_id", "table_id", "grid_label", "variable_id")
        if c in findings
    ]
    grouped = findings.groupby(["code", "level"], dropna=False)
    summary = grouped.agg(
        n=("code", "size"),
        n_datasets=(keys[0] if keys else "code", "nunique"),
        example=("message", "first"),
    ).reset_index()
    summary["level"] = pd.Categorical(summary["level"], SEVERITY_ORDER, ordered=True)
    return summary.sort_values(["level", "n"], ascending=[True, False]).reset_index(
        drop=True
    )


def summarize_by_dataset(
    findings: pd.DataFrame,
    keys: Sequence[str] = ("activity_id", "table_id", "grid_label", "variable_id"),
) -> pd.DataFrame:
    """
    Summarise findings one row per dataset.

    Parameters
    ----------
    findings : pandas.DataFrame
        Findings table.
    keys : sequence of str, optional
        Facet columns identifying a dataset. Columns absent from ``findings``
        are skipped.

    Returns
    -------
    pandas.DataFrame
        The key columns plus ``worst``, ``n_findings`` and ``codes``, ordered by
        finding count descending. Only FATAL, ERROR and WARN rows are counted;
        a dataset with nothing wrong does not appear.
    """
    if findings.empty:
        return pd.DataFrame()
    keys = [k for k in keys if k in findings.columns]
    problems = findings[findings["level"].isin(["FATAL", "ERROR", "WARN"])]
    grouped = problems.groupby(keys, dropna=False)
    out = grouped.agg(
        worst=("level_num", "max"),
        n_findings=("code", "size"),
        codes=("code", lambda s: ", ".join(sorted(set(s)))),
    ).reset_index()
    out["worst"] = out["worst"].map(
        {40: "FATAL", 30: "ERROR", 20: "WARN", 10: "INFO", 0: "OK"}
    )
    return out.sort_values(["n_findings"], ascending=False).reset_index(drop=True)


def severity_counts(findings: pd.DataFrame) -> pd.Series:
    """
    Count findings by severity.

    Parameters
    ----------
    findings : pandas.DataFrame
        Findings table.

    Returns
    -------
    pandas.Series
        Counts indexed by severity name, ordered worst first. Empty when there
        are no findings.
    """
    if findings.empty:
        return pd.Series(dtype=int)
    counts = findings["level"].value_counts()
    return counts.reindex([s for s in SEVERITY_ORDER if s in counts.index])


def write_outputs(
    findings: pd.DataFrame,
    outdir: str | Path,
    stem: str = "qc",
    formats: Sequence[str] = ("csv", "md"),
    title: str = "cadcat data quality report",
) -> list[Path]:
    """
    Write the findings table and its summaries to disk.

    Parameters
    ----------
    findings : pandas.DataFrame
        Findings table.
    outdir : str or pathlib.Path
        Directory to write into; created if absent.
    stem : str, optional
        Filename prefix. Default ``"qc"``.
    formats : sequence of str, optional
        Any of ``"csv"``, ``"parquet"``, ``"md"``, ``"html"``. ``"csv"`` also
        writes a by-code summary alongside the findings.
    title : str, optional
        Heading used by the Markdown and HTML reports.

    Returns
    -------
    list of pathlib.Path
        The files written, in the order they were produced.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if "csv" in formats:
        path = outdir / f"{stem}_findings.csv"
        findings.to_csv(path, index=False)
        written.append(path)

        summary_path = outdir / f"{stem}_summary_by_code.csv"
        summarize_by_code(findings).to_csv(summary_path, index=False)
        written.append(summary_path)

    if "parquet" in formats:
        path = outdir / f"{stem}_findings.parquet"
        findings.to_parquet(path, index=False)
        written.append(path)

    if "md" in formats:
        path = outdir / f"{stem}_report.md"
        path.write_text(render_markdown(findings, title=title), encoding="utf-8")
        written.append(path)

    if "html" in formats:
        path = outdir / f"{stem}_report.html"
        path.write_text(render_html(findings, title=title), encoding="utf-8")
        written.append(path)

    return written


def render_markdown(
    findings: pd.DataFrame, title: str = "cadcat data quality report"
) -> str:
    """
    Render a short Markdown report.

    Parameters
    ----------
    findings : pandas.DataFrame
        Findings table.
    title : str, optional
        Heading for the report.

    Returns
    -------
    str
        Markdown with severity counts, the top 40 problem codes, and the 25
        datasets carrying the most findings.
    """
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [f"# {title}", "", f"Generated {stamp}.", ""]

    if findings.empty:
        lines.append("No findings recorded.")
        return "\n".join(lines)

    n_datasets = _count_datasets(findings)
    lines += [
        f"Inspected **{n_datasets:,}** dataset(s); recorded "
        f"**{len(findings):,}** finding(s).",
        "",
        "## Severity",
        "",
    ]
    counts = severity_counts(findings)
    lines.append("| level | count |")
    lines.append("| --- | ---: |")
    for level, count in counts.items():
        lines.append(f"| {level} | {count:,} |")
    lines.append("")

    lines += ["## Findings by code", ""]
    summary = summarize_by_code(findings)
    problems = summary[summary["level"].isin(["FATAL", "ERROR", "WARN"])]
    if problems.empty:
        lines.append("No FATAL, ERROR or WARN findings.")
    else:
        lines.append("| level | code | count | example |")
        lines.append("| --- | --- | ---: | --- |")
        for row in problems.head(40).itertuples():
            example = str(row.example).replace("|", "\\|")[:160]
            lines.append(f"| {row.level} | `{row.code}` | {row.n:,} | {example} |")
    lines.append("")

    worst = summarize_by_dataset(findings)
    if not worst.empty:
        lines += ["## Datasets with the most findings", ""]
        keys = [c for c in worst.columns if c not in {"worst", "n_findings", "codes"}]
        header = " | ".join(keys + ["worst", "n", "codes"])
        divider = " | ".join(["---"] * (len(keys) + 3))
        lines.append(f"| {header} |")
        lines.append(f"| {divider} |")
        for row in worst.head(25).to_dict("records"):
            cells = [str(row.get(k, "")) for k in keys]
            cells += [
                str(row["worst"]),
                str(row["n_findings"]),
                f"`{row['codes'][:120]}`",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines)


def render_html(
    findings: pd.DataFrame, title: str = "cadcat data quality report"
) -> str:
    """
    Render a standalone HTML report.

    Parameters
    ----------
    findings : pandas.DataFrame
        Findings table. At most the first 5000 rows are tabulated.
    title : str, optional
        Heading and document title.

    Returns
    -------
    str
        A complete HTML document: the Markdown summary followed by the findings
        table, with styles inlined so the file stands alone.
    """
    body = render_markdown(findings, title=title)
    rows = findings.head(5000)
    table = rows.to_html(index=False, escape=True, classes="findings", border=0)
    style = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 2rem auto; max-width: 1100px; line-height: 1.5; color: #1a1a1a; }
    table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
    th, td { border-bottom: 1px solid #e3e3e3; padding: 0.35rem 0.5rem;
             text-align: left; vertical-align: top; }
    th { background: #f6f6f6; position: sticky; top: 0; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    pre { background: #f6f6f6; padding: 1rem; overflow-x: auto; white-space: pre-wrap; }
    """
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{style}</style></head><body>"
        f"<pre>{html.escape(body)}</pre>"
        "<h2>All findings</h2>"
        f"{table}"
        "</body></html>"
    )


def _count_datasets(findings: pd.DataFrame) -> int:
    """
    Count the distinct datasets represented in a findings table.

    Parameters
    ----------
    findings : pandas.DataFrame
        Findings table.

    Returns
    -------
    int
        Distinct facet combinations, or the row count when no facet columns are
        present.
    """
    keys = [
        c
        for c in (
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "member_id",
            "table_id",
            "variable_id",
            "grid_label",
        )
        if c in findings.columns
    ]
    if not keys:
        return len(findings)
    return int(findings[keys].astype(str).drop_duplicates().shape[0])
