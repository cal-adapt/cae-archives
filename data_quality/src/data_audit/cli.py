"""
Command line interface: ``python -m data_audit <command>``.

Commands
--------
``audit``    catalog-only checks, no data access
``sweep``    open sampled stores and check their metadata
``compare``  diff climakitae's view against the raw Zarr view
``coverage`` write the variable x resolution presence matrices
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from . import catalog as catalog_module
from . import report, runner
from . import standards as S

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``audit``, ``sweep``, ``compare``, ``reconcile`` and
        ``coverage`` subcommands, each carrying the shared catalog filters.
    """
    parser = argparse.ArgumentParser(
        prog="data_audit",
        description="Data quality checks for the Cal-Adapt cadcat WRF and LOCA2 archives.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--catalog-source",
            choices=["auto", "live", "bundled"],
            default="auto",
            help="where to read the catalog from (default: auto)",
        )
        sub.add_argument(
            "--activity",
            nargs="+",
            default=list(S.ACTIVITIES),
            help="activity_id values to include (default: WRF LOCA2)",
        )
        sub.add_argument(
            "--grid-label", nargs="+", default=None, help="e.g. d01 d02 d03"
        )
        sub.add_argument("--table-id", nargs="+", default=None, help="e.g. 1hr day mon")
        sub.add_argument(
            "--variable", nargs="+", default=None, help="variable_id filter"
        )
        sub.add_argument(
            "--experiment", nargs="+", default=None, help="experiment_id filter"
        )
        sub.add_argument("--outdir", default="qc_output", help="output directory")
        sub.add_argument(
            "--formats",
            nargs="+",
            default=["csv", "md"],
            choices=["csv", "md", "html", "parquet"],
        )

    audit = subparsers.add_parser(
        "audit", help="catalog-only checks (fast, no data reads)"
    )
    add_common(audit)

    sweep = subparsers.add_parser(
        "sweep", help="open sampled stores and check metadata"
    )
    add_common(sweep)
    sweep.add_argument(
        "--backend",
        choices=["climakitae", "zarr"],
        default="climakitae",
        help="how to open the data (default: climakitae)",
    )
    sweep.add_argument(
        "--per-key", type=int, default=1, help="stores per facet combination"
    )
    sweep.add_argument(
        "--per-model",
        action="store_true",
        help="add source_id to the sampling key, so every driving model is "
        "probed rather than one per variable (much larger plan)",
    )
    sweep.add_argument(
        "--no-experiments",
        action="store_true",
        help="collapse experiments into one sample (smaller plan, skips extent checks)",
    )
    sweep.add_argument("--limit", type=int, default=None, help="cap the plan size")
    sweep.add_argument("--workers", type=int, default=4)
    sweep.add_argument(
        "--inspect-store",
        action="store_true",
        help="also run raw Zarr structural checks (zarr backend only)",
    )
    sweep.add_argument("--no-time", action="store_true", help="skip time-axis checks")
    sweep.add_argument(
        "--probe-values",
        action="store_true",
        help="read a few slices per store to catch empty variables (slower)",
    )

    compare = subparsers.add_parser(
        "compare", help="diff climakitae vs raw Zarr metadata"
    )
    add_common(compare)
    compare.add_argument("--per-key", type=int, default=1)
    compare.add_argument("--limit", type=int, default=50)
    compare.add_argument("--workers", type=int, default=4)

    reconcile = subparsers.add_parser(
        "reconcile",
        help="compare store vs delivered product, both against the documentation",
    )
    add_common(reconcile)
    reconcile.add_argument("--per-key", type=int, default=1)
    reconcile.add_argument("--limit", type=int, default=40)
    reconcile.add_argument("--workers", type=int, default=4)
    reconcile.add_argument("--probe-values", action="store_true")
    reconcile.add_argument("--no-time", action="store_true")

    coverage = subparsers.add_parser(
        "coverage", help="variable x resolution presence matrices"
    )
    add_common(coverage)

    return parser


def _configure_logging(verbose: bool) -> None:
    """
    Keep our own output visible and everyone else's out of the way.

    Two sources of noise worth naming: climakitae logs progress to the *root*
    logger at INFO ("Choice not found. Ignoring: ..."), and pint emits a dozen
    unit-redefinition warnings the moment climakitae imports it. Neither says
    anything about data quality.

    So the root logger sits at WARNING while ``data_audit`` sits at INFO, and the
    known-chatty third parties are pinned above their noise. ``--verbose`` opens
    everything back up, because when something is actually broken you want the
    library's own trace.

    Parameters
    ----------
    verbose : bool
        Raise every logger to DEBUG instead of quieting the third parties.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        return

    logging.getLogger("data_audit").setLevel(logging.INFO)
    for noisy, level in (
        ("pint", logging.ERROR),
        ("climakitae", logging.WARNING),
        ("intake", logging.WARNING),
        ("s3fs", logging.WARNING),
        ("botocore", logging.WARNING),
        ("aiobotocore", logging.WARNING),
        ("fsspec", logging.WARNING),
        ("asyncio", logging.CRITICAL),
    ):
        logging.getLogger(noisy).setLevel(level)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run a data_audit subcommand.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status: 0 clean, 1 warnings only or no rows matched, 2
        when errors were found. Suitable for CI.
    """
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    frame = catalog_module.load_catalog(args.catalog_source)
    frame = catalog_module.filter_catalog(
        frame,
        activities=args.activity,
        grid_labels=args.grid_label,
        table_ids=args.table_id,
        variable_ids=args.variable,
        experiment_ids=args.experiment,
    )
    logger.info("catalog subset: %d store(s)", len(frame))
    if frame.empty:
        logger.error("No catalog rows matched those filters.")
        return 1

    outdir = Path(args.outdir)

    if args.command == "audit":
        findings = runner.audit_catalog(frame)
        _emit(findings, outdir, "catalog_audit", args.formats, "cadcat catalog audit")
        return _exit_code(findings)

    if args.command == "coverage":
        outdir.mkdir(parents=True, exist_ok=True)
        for activity in sorted(set(frame["activity_id"].dropna())):
            matrix = catalog_module.coverage_matrix(frame, activity)
            path = outdir / f"coverage_{activity}.csv"
            matrix.to_csv(path)
            print(f"wrote {path}")
        missing = catalog_module.documented_but_absent(frame)
        path = outdir / "documented_but_absent.csv"
        missing.to_csv(path, index=False)
        print(f"wrote {path} ({len(missing)} row(s))")
        return 0

    key_facets = list(S.METADATA_KEY_FACETS)
    if getattr(args, "per_model", False):
        # Metadata is per-variable, but *data* is per-simulation: one model's
        # store can be empty while its siblings are fine. Sampling one model
        # per variable cannot see that.
        key_facets.append("source_id")
    plan = catalog_module.build_sample_plan(
        frame,
        key_facets=key_facets,
        per_key=args.per_key,
        include_experiments=not getattr(args, "no_experiments", False),
    )
    if args.limit:
        plan = plan.head(args.limit)
    logger.info("sample plan: %d store(s) to open", len(plan))

    if args.command == "sweep":
        backend = _make_backend(args.backend)
        try:
            findings = runner.sweep(
                plan,
                backend,
                inspect_store=args.inspect_store and args.backend == "zarr",
                check_time_axis=not args.no_time,
                probe_values=args.probe_values,
                max_workers=args.workers,
            )
        finally:
            _close(backend)
        _emit(
            findings,
            outdir,
            f"sweep_{args.backend}",
            args.formats,
            f"cadcat metadata sweep ({args.backend})",
        )
        return _exit_code(findings)

    if args.command == "reconcile":
        from . import reconcile as reconcile_module
        from .backend_climakitae import ClimakitaeBackend
        from .backend_zarr import ZarrBackend

        store = ZarrBackend()
        delivered = ClimakitaeBackend()
        try:
            table = reconcile_module.reconcile(
                plan,
                store,
                delivered,
                check_time_axis=not args.no_time,
                probe_values=args.probe_values,
                max_workers=args.workers,
            )
        finally:
            _close(store)
            _close(delivered)

        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / "reconciliation.csv"
        table.to_csv(csv_path, index=False)
        md_path = outdir / "reconciliation.md"
        md_path.write_text(reconcile_module.render_markdown(table), encoding="utf-8")
        print(f"wrote {csv_path}")
        print(f"wrote {md_path}")
        if not table.empty:
            print()
            print(table["verdict"].value_counts().to_string())
        return 2 if (not table.empty and (table["level"] == "ERROR").any()) else 0

    if args.command == "compare":
        from .backend_climakitae import ClimakitaeBackend
        from .backend_zarr import ZarrBackend

        zarr_backend = ZarrBackend()
        try:
            diff = runner.compare_backends(
                plan, ClimakitaeBackend(), zarr_backend, max_workers=args.workers
            )
        finally:
            _close(zarr_backend)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "backend_comparison.csv"
        diff.to_csv(path, index=False)
        disagreements = diff[~diff["agree"]]
        print(f"wrote {path}")
        print(f"{len(disagreements)} disagreement(s) across {len(diff)} comparison(s)")
        if not disagreements.empty:
            print(disagreements["field"].value_counts().to_string())
        return 0

    return 1


def _close(backend: Any) -> None:
    """
    Release a backend's network resources, if it holds any.

    Parameters
    ----------
    backend : Any
        Reader that may expose a ``close()`` method. Anything else is ignored,
        and a failing ``close()`` is swallowed: cleanup must not mask the
        result of the run.
    """
    closer = getattr(backend, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def _make_backend(name: str) -> Any:
    """
    Construct a reader by name.

    Parameters
    ----------
    name : str
        Either ``"climakitae"`` or ``"zarr"``.

    Returns
    -------
    Any
        The backend instance. Imports are deferred so the CLI starts without
        the optional dependencies installed.
    """
    if name == "climakitae":
        from .backend_climakitae import ClimakitaeBackend

        return ClimakitaeBackend()
    from .backend_zarr import ZarrBackend

    return ZarrBackend()


def _emit(
    findings: pd.DataFrame,
    outdir: Path,
    stem: str,
    formats: Sequence[str],
    title: str,
) -> None:
    """
    Write the findings and print a severity tally.

    Parameters
    ----------
    findings : pandas.DataFrame
        Findings table.
    outdir : pathlib.Path
        Directory to write into.
    stem : str
        Filename prefix.
    formats : sequence of str
        Output formats, passed to :func:`data_audit.report.write_outputs`.
    title : str
        Heading for the rendered reports.
    """
    written = report.write_outputs(
        findings, outdir, stem=stem, formats=formats, title=title
    )
    for path in written:
        print(f"wrote {path}")
    counts = report.severity_counts(findings)
    if not counts.empty:
        print()
        print(counts.to_string())


def _exit_code(findings: pd.DataFrame) -> int:
    """
    Derive a process exit status from the worst finding.

    Parameters
    ----------
    findings : pandas.DataFrame
        Findings table.

    Returns
    -------
    int
        0 when clean or informational only, 1 when the worst is a warning,
        2 when any error or fatal finding is present.
    """
    if findings.empty:
        return 0
    worst = int(findings["level_num"].max())
    if worst >= 30:
        return 2
    if worst >= 20:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
