"""Wind speed and relative humidity analysis for the Cal-Adapt archive.

Companion to :mod:`data_audit`, which checks the archive against its
documentation. This package answers the next question: do the downscaled
products agree with independent references?

**Wind speed** — do the downscaled 10 m winds agree with ERA5, with an
independent km-scale reanalysis (CONUS404), and with station observations?

**Relative humidity** — the same, for RH, where CMIP6 carries a known moist
bias over the arid Southwest.

Layers
------
Data access is layered, and the layers are meant to stay separate:

``fetch_*``
    Pull raw data from source (ERA5, CONUS404, HDP stations, the local archive)
    into the working store. Network-bound, run once.
``derive_*``, :mod:`~data_quality.humidity`
    Compute what the sources do not ship — scalar wind speed from components,
    relative humidity from temperature and specific humidity.
``localdata``
    The only module that knows the on-disk layout. Everything downstream opens
    data through it rather than touching paths.
``grids``, :mod:`~data_quality.metrics`
    Regridding, binning, coverage, and the comparison statistics.
``build_*``
    Headless pipelines that assemble the analysis store, resumable stage by
    stage.
``ae_era5_comparison``, :mod:`~data_quality.rh_analysis`
    The comparisons themselves, used by the review notebooks.

Notes
-----
Submodules are imported lazily, so ``import data_quality`` does not pull in
xarray, dask or the cloud clients and works without credentials.
"""

from __future__ import annotations

import importlib
from typing import Any, List

__version__ = "0.1.0"

#: Submodules reachable as ``data_quality.<name>`` without an explicit import.
SUBMODULES: tuple[str, ...] = (
    "ae_era5_comparison",
    "apply_cache_stamp",
    "build_datasets",
    "build_metrics",
    "build_metrics_rh",
    "derive_conus404",
    "derive_era5",
    "fetch_conus404",
    "fetch_era5",
    "fetch_geometry",
    "fetch_grids",
    "fetch_hdp",
    "fetch_local",
    "fetch_local_daily",
    "fetch_stations_1hr",
    "grids",
    "hdp_stations",
    "humidity",
    "localdata",
    "metrics",
    "publish_to_s3",
    "rh_analysis",
    "split_hdp_stores",
    "verify",
)


def __getattr__(name: str) -> Any:
    """Import a submodule on first attribute access.

    Parameters
    ----------
    name : str
        Attribute being looked up on the package.

    Returns
    -------
    Any
        The imported submodule.

    Raises
    ------
    AttributeError
        If ``name`` is not one of :data:`SUBMODULES`.
    """
    if name in SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """List public attributes, including the lazily-imported submodules.

    Returns
    -------
    list of str
        Sorted attribute names.
    """
    return sorted(set(globals()) | set(SUBMODULES))


__all__ = ["__version__", "SUBMODULES", *SUBMODULES]
