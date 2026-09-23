"""Data quality checks for the Cal-Adapt ``cadcat`` WRF and LOCA2 archives.

Two access paths, one check engine:

- :class:`~data_audit.backend_climakitae.ClimakitaeBackend` reads through
  ``climakitae.new_core.user_interface.ClimateData``, so it sees the data as a
  user of the Analytics Engine does.
- :class:`~data_audit.backend_zarr.ZarrBackend` reads the Zarr stores straight
  from ``s3://cadcat``, so it sees what was actually published.

Running both and diffing them is the point: anything climakitae repairs on read
is still a defect in the delivered artifact.

Quick start
-----------
>>> from data_audit import load_catalog, audit_catalog
>>> cat = load_catalog()
>>> findings = audit_catalog(cat)          # no data access
>>> findings.query("level == 'ERROR'").head()
"""

from typing import Any

from .catalog import (
    build_sample_plan,
    compare_catalog_sources,
    coverage_matrix,
    documented_but_absent,
    facet_consistency,
    filter_catalog,
    grid_coverage_asymmetry,
    load_catalog,
    timescale_mismatch,
    undocumented_in_catalog,
)
from .checks import check_catalog_row, check_dataset
from .findings import Finding, FindingList, Level

# `reconcile` stays the submodule; its main entry point is exported under a
# distinct name so `data_audit.reconcile.render_markdown` remains reachable.
from .reconcile import reconcile as reconcile_surfaces
from .reconcile import requirement_for
from .report import (
    render_markdown,
    severity_counts,
    summarize_by_code,
    summarize_by_dataset,
    write_outputs,
)
from .runner import audit_catalog, compare_backends, sweep

__version__ = "0.1.0"

__all__ = [
    "load_catalog",
    "filter_catalog",
    "build_sample_plan",
    "coverage_matrix",
    "documented_but_absent",
    "grid_coverage_asymmetry",
    "facet_consistency",
    "timescale_mismatch",
    "undocumented_in_catalog",
    "compare_catalog_sources",
    "check_catalog_row",
    "check_dataset",
    "audit_catalog",
    "sweep",
    "compare_backends",
    "reconcile_surfaces",
    "requirement_for",
    "summarize_by_code",
    "summarize_by_dataset",
    "severity_counts",
    "render_markdown",
    "write_outputs",
    "Finding",
    "FindingList",
    "Level",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Import a backend on first attribute access.

    The backends import climakitae and s3fs, so deferring them keeps
    ``import data_audit`` working in an environment that has neither.

    Parameters
    ----------
    name : str
        Attribute being looked up on the package.

    Returns
    -------
    Any
        The backend class.

    Raises
    ------
    AttributeError
        If ``name`` is not a known backend.
    """
    if name == "ClimakitaeBackend":
        from .backend_climakitae import ClimakitaeBackend

        return ClimakitaeBackend
    if name == "ZarrBackend":
        from .backend_zarr import ZarrBackend

        return ZarrBackend
    raise AttributeError(name)
