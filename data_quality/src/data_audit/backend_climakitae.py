"""
Backend A: reach the data the way a user would, through ``ClimateData``.

This exercises the whole climakitae read path, so it sees the metadata *after*
the library has post-processed it (``add_crs_to_downscaled_data`` attaches a CRS
to stores that lack one, for instance). That makes it the right backend for
answering "is the data usable as delivered?" and the wrong backend for "is the
store on S3 correct?" — for the latter use :mod:`data_audit.backend_zarr`.

Nothing here calls ``.load()``. ``.get()`` returns lazily-opened datasets and we
only ever touch coordinates.
"""

from __future__ import annotations

import logging
from typing import Any

import xarray as xr

from . import standards as S

logger = logging.getLogger(__name__)

#: Facets that ClimateData exposes as query methods.
_QUERY_METHODS = {
    "activity_id": "activity_id",
    "institution_id": "institution_id",
    "source_id": "source_id",
    "experiment_id": "experiment_id",
    "table_id": "table_id",
    "grid_label": "grid_label",
    "variable_id": "variable_id",
}


class ClimakitaeBackend:
    """
    Open catalog rows via the ``ClimateData`` fluent interface.

    Parameters
    ----------
    verbosity
        Passed to ``ClimateData``. Defaults to ``-2`` (silent), because a sweep
        over thousands of stores otherwise drowns in INFO logging.
    """

    name = "climakitae"

    #: climakitae aligns and combines matching stores before returning, so the
    #: shape of what it hands back — the time axis above all — is a property of
    #: that processing, not of any published store. Structural findings from
    #: this backend are reported at INFO; use the zarr backend to judge the
    #: archive itself.
    authoritative_structure = False

    def __init__(self, verbosity: int = -2) -> None:
        """
        Open a ``ClimateData`` session.

        Parameters
        ----------
        verbosity : int, optional
            Passed to ``ClimateData``. Defaults to ``-2`` (silent), because a
            sweep over thousands of stores otherwise drowns in INFO logging.
        """
        from climakitae.new_core.user_interface import ClimateData

        self._client = ClimateData(verbosity=verbosity)

    def open(self, row: dict[str, Any]) -> tuple[xr.Dataset | None, dict[str, Any]]:
        """
        Retrieve one catalog row through the fluent interface.

        Parameters
        ----------
        row : dict
            Catalog record. Facets present in :data:`_QUERY_METHODS` are pinned
            on the query; anything else is ignored.

        Returns
        -------
        dataset : xarray.Dataset or None
            The lazily-opened result, or ``None`` when the query matched
            nothing.
        diagnostics : dict
            Always carries ``n_returned`` and ``returned_keys`` so the runner
            can report over-broad queries: getting three datasets back for a
            query that pinned every facet means the catalog holds duplicate
            records.
        """
        query = self._build_query(row)
        client = self._client.reset()
        for facet, method_name in _QUERY_METHODS.items():
            value = row.get(facet)
            if isinstance(value, str) and value and value.lower() != "nan":
                getattr(client, method_name)(value)
        client.catalog(S.CADCAT_CATALOG)

        result = client.get()
        diagnostics: dict[str, Any] = {"query": query, "backend": self.name}

        datasets = _normalize_result(result)
        diagnostics["n_returned"] = len(datasets)
        diagnostics["returned_keys"] = list(datasets)

        if not datasets:
            return None, diagnostics

        key = next(iter(datasets))
        diagnostics["selected_key"] = key
        return datasets[key], diagnostics

    @staticmethod
    def _build_query(row: dict[str, Any]) -> dict[str, str]:
        """
        Extract the queryable facets from a catalog row.

        Parameters
        ----------
        row : dict
            Catalog record.

        Returns
        -------
        dict
            Facet name to value, omitting blanks and non-strings.
        """
        return {
            facet: row[facet]
            for facet in _QUERY_METHODS
            if isinstance(row.get(facet), str) and row[facet].lower() != "nan"
        }


def _normalize_result(result: Any) -> dict[str, xr.Dataset]:
    """
    Coerce whatever ``.get()`` handed back into ``{key: Dataset}``.

    Parameters
    ----------
    result : Any
        Return value of ``ClimateData.get()``. A dict of datasets keyed by
        catalog id in the common case, but the documentation also shows a bare
        ``Dataset`` or ``DataArray`` depending on the processors applied, so all
        three are handled.

    Returns
    -------
    dict of str to xarray.Dataset
        Empty when the result held no datasets.
    """
    if result is None:
        return {}
    if isinstance(result, dict):
        out: dict[str, xr.Dataset] = {}
        for key, value in result.items():
            dataset = _as_dataset(value)
            if dataset is not None:
                out[str(key)] = dataset
        return out
    dataset = _as_dataset(result)
    return {"result": dataset} if dataset is not None else {}


def _as_dataset(value: Any) -> xr.Dataset | None:
    """
    Promote a DataArray to a Dataset, passing Datasets through.

    Parameters
    ----------
    value : Any
        Candidate object.

    Returns
    -------
    xarray.Dataset or None
        ``None`` when the value is neither a ``Dataset`` nor a ``DataArray``.
    """
    if isinstance(value, xr.Dataset):
        return value
    if isinstance(value, xr.DataArray):
        name = value.name or "data"
        return value.to_dataset(name=name)
    return None


def show_options(verbosity: int = 0) -> None:
    """
    Print every catalog option, mirroring ``ClimateData.show_all_options()``.

    Handy at the start of an exploratory session. The sweep itself uses the
    catalog DataFrame instead, because the ``show_*`` methods log rather than
    return values.

    Parameters
    ----------
    verbosity : int, optional
        Passed to ``ClimateData``. Default 0, which is the INFO level the
        ``show_*`` methods write to.
    """
    from climakitae.new_core.user_interface import ClimateData

    ClimateData(verbosity=verbosity).show_all_options()
