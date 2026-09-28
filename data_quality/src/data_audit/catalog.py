"""
Load and slice the cadcat intake-ESM catalog.

Two sources, in preference order:

1. **Live** — ``climakitae.new_core.data_access.DataCatalog().data.df``, which
   reads ``cae-collection.json`` from S3. This is what ``ClimateData`` actually
   queries, so it is the truth for a QC sweep.
2. **Bundled** — ``climakitae/data/catalogs.csv``, a static snapshot that ships
   inside the package. Useful offline, and useful as a *second* opinion: a
   difference between the two is itself a finding worth reporting.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from . import standards as S

logger = logging.getLogger(__name__)


def load_catalog(source: str = "auto") -> pd.DataFrame:
    """
    Return the cadcat catalog as a DataFrame.

    Parameters
    ----------
    source
        ``"live"``, ``"bundled"``, or ``"auto"`` (live, falling back to bundled).

    Returns
    -------
    DataFrame
        One row per Zarr store, with the columns in :data:`standards.FACETS`
        plus ``path`` and a ``catalog_source`` column recording where it came
        from.
    """
    if source not in {"auto", "live", "bundled"}:
        raise ValueError(f"source must be auto/live/bundled, got {source!r}")

    if source in {"auto", "live"}:
        try:
            frame = _load_live()
            frame["catalog_source"] = "live"
            return _tidy(frame)
        except Exception as exc:
            if source == "live":
                raise
            logger.warning(
                "Live catalog unavailable (%s); using bundled snapshot.", exc
            )

    frame = _load_bundled()
    frame["catalog_source"] = "bundled"
    return _tidy(frame)


def _load_live() -> pd.DataFrame:
    """
    Read the live intake-ESM collection through climakitae.

    Returns
    -------
    pandas.DataFrame
        The cadcat collection as intake-esm exposes it.
    """
    from climakitae.new_core.data_access.data_access import DataCatalog

    catalog = DataCatalog()
    return catalog.data.df.copy()


def _load_bundled() -> pd.DataFrame:
    """
    Read the catalog snapshot bundled inside climakitae.

    Returns
    -------
    pandas.DataFrame
        The snapshot, filtered to the cadcat rows.

    Raises
    ------
    RuntimeError
        If climakitae is not installed.
    """
    import importlib.util
    import os

    spec = importlib.util.find_spec("climakitae")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("climakitae is not installed; cannot load bundled catalog.")
    root = list(spec.submodule_search_locations)[0]
    path = os.path.join(root, "data", "catalogs.csv")
    frame = pd.read_csv(path)
    # The bundled file mixes cadcat with the renewables catalog; keep cadcat.
    if "catalog" in frame.columns:
        frame = frame[frame["catalog"].astype(str).str.lower().isin({"data", "cadcat"})]
    return frame.copy()


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise columns and dtypes so both catalog sources match.

    Parameters
    ----------
    frame : pandas.DataFrame
        Raw catalog frame from either source.

    Returns
    -------
    pandas.DataFrame
        The facet columns plus ``path`` and ``catalog_source``, with missing
        facets added as nulls so both sources share a schema.
    """
    for column in S.FACETS:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = frame[column].astype("object")
    if "path" not in frame.columns:
        frame["path"] = pd.NA
    keep = list(S.FACETS) + ["path", "catalog_source"]
    keep = [c for c in keep if c in frame.columns]
    out = frame[keep].reset_index(drop=True)
    return out


def filter_catalog(
    frame: pd.DataFrame,
    activities: Sequence[str] = S.ACTIVITIES,
    grid_labels: Sequence[str] | None = None,
    table_ids: Sequence[str] | None = None,
    variable_ids: Sequence[str] | None = None,
    experiment_ids: Sequence[str] | None = None,
    source_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Subset the catalog. ``None`` means "no constraint on this facet".

    Parameters
    ----------
    frame : pandas.DataFrame
        Catalog records.
    activities : sequence of str, optional
        ``activity_id`` values to keep. Default WRF and LOCA2.
    grid_labels, table_ids, variable_ids, experiment_ids, source_ids : sequence of str, optional
        Further facet filters. ``None`` means no constraint on that facet.

    Returns
    -------
    pandas.DataFrame
        The matching rows, re-indexed.
    """
    mask = frame["activity_id"].isin(list(activities))
    for column, values in (
        ("grid_label", grid_labels),
        ("table_id", table_ids),
        ("variable_id", variable_ids),
        ("experiment_id", experiment_ids),
        ("source_id", source_ids),
    ):
        if values is not None:
            mask &= frame[column].isin(list(values))
    return frame[mask].reset_index(drop=True)


def build_sample_plan(
    frame: pd.DataFrame,
    key_facets: Sequence[str] = S.METADATA_KEY_FACETS,
    per_key: int = 1,
    include_experiments: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Pick a representative subset of stores to open.

    Opening every store is possible but wasteful: metadata is a property of
    (activity, institution, table_id, grid_label, variable_id), not of which GCM
    or ensemble member produced the run. So take ``per_key`` stores per unique
    combination of ``key_facets``.

    ``include_experiments`` adds ``experiment_id`` to the key, which roughly
    quadruples the plan but is what you want when checking temporal extent,
    since expected start/end years are per-experiment.

    Returns the sampled rows plus an ``n_in_group`` column recording how many
    stores each sampled row stands for.

    Parameters
    ----------
    frame : pandas.DataFrame
        Catalog records.
    key_facets : sequence of str, optional
        Facets that define a distinct metadata combination.
    per_key : int, optional
        Stores to sample per combination. Default 1.
    include_experiments : bool, optional
        Add ``experiment_id`` to the key. Default True, because expected
        start and end years are per-experiment.
    seed : int, optional
        Random seed, so a plan is reproducible. Default 0.
    """
    keys = list(key_facets)
    if include_experiments and "experiment_id" not in keys:
        keys.append("experiment_id")

    grouped = frame.groupby(keys, dropna=False, sort=True)
    sizes = grouped.size().rename("n_in_group")

    sampled = (
        grouped.apply(
            lambda g: g.sample(n=min(per_key, len(g)), random_state=seed),
            include_groups=False,
        )
        .reset_index(level=list(range(len(keys))))
        .reset_index(drop=True)
    )
    sampled = sampled.merge(sizes.reset_index(), on=keys, how="left")
    return sampled


def coverage_matrix(frame: pd.DataFrame, activity_id: str) -> pd.DataFrame:
    """
    Variables x (grid_label, table_id) presence table for one activity.

    Rows are variables, columns are ``grid_label/table_id``, values are the
    number of stores. Reading down a column shows which documented variables are
    missing at that resolution.

    Parameters
    ----------
    frame : pandas.DataFrame
        Catalog records.
    activity_id : str
        Downscaling method to tabulate.

    Returns
    -------
    pandas.DataFrame
        Variables as rows, ``grid_label/table_id`` as columns, store counts as
        values. Empty when the activity is absent.
    """
    subset = frame[frame["activity_id"] == activity_id]
    if subset.empty:
        return pd.DataFrame()
    subset = subset.assign(
        combo=subset["grid_label"].astype(str) + "/" + subset["table_id"].astype(str)
    )
    table = subset.pivot_table(
        index="variable_id",
        columns="combo",
        values="path",
        aggfunc="count",
        fill_value=0,
    )
    return table.sort_index()


def documented_but_absent(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Documented variables with no store anywhere at that activity/table.

    This is the strong signal: the reference table says the variable exists at
    this temporal resolution and the catalog has nothing at any grid. For the
    softer "present at some grids but not others" case see
    :func:`grid_coverage_asymmetry`.

    Parameters
    ----------
    frame : pandas.DataFrame
        Catalog records.

    Returns
    -------
    pandas.DataFrame
        One row per (``activity_id``, ``table_id``, ``variable_id``) that the
        reference table documents and the catalog does not carry at any grid.
    """
    rows: list[dict] = []
    for activity_id in sorted(set(frame["activity_id"].dropna())):
        subset = frame[frame["activity_id"] == activity_id]
        for table_id in sorted(set(subset["table_id"].dropna())):
            expected = S.documented_variables(activity_id, table_id)
            if not expected:
                continue
            present = set(
                subset[subset["table_id"] == table_id]["variable_id"].dropna()
            )
            published_anywhere = set(subset["variable_id"].dropna())
            for variable_id in sorted(expected - present):
                if variable_id in published_anywhere:
                    continue  # exists at another table_id -> timescale_mismatch
                rows.append(
                    {
                        "activity_id": activity_id,
                        "table_id": table_id,
                        "variable_id": variable_id,
                    }
                )
    return pd.DataFrame(rows)


def grid_coverage_asymmetry(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Variables published at some grid labels but not others.

    Asymmetry is often intentional (the 3 km domain carries variables the 45 km
    domain does not), so this is reported for review rather than as a defect.
    One row per (activity_id, table_id, variable_id) with the grids present and
    missing.

    Parameters
    ----------
    frame : pandas.DataFrame
        Catalog records.

    Returns
    -------
    pandas.DataFrame
        One row per variable published at some grids and not others, with
        ``grids_present`` and ``grids_missing``.
    """
    rows: list[dict] = []
    for activity_id in sorted(set(frame["activity_id"].dropna())):
        subset = frame[frame["activity_id"] == activity_id]
        all_grids = set(subset["grid_label"].dropna())
        if len(all_grids) < 2:
            continue
        for table_id in sorted(set(subset["table_id"].dropna())):
            at_table = subset[subset["table_id"] == table_id]
            grids_here = set(at_table["grid_label"].dropna())
            if len(grids_here) < 2:
                continue
            for variable_id in sorted(set(at_table["variable_id"].dropna())):
                present = set(
                    at_table[at_table["variable_id"] == variable_id][
                        "grid_label"
                    ].dropna()
                )
                missing = grids_here - present
                if missing:
                    rows.append(
                        {
                            "activity_id": activity_id,
                            "table_id": table_id,
                            "variable_id": variable_id,
                            "grids_present": ",".join(sorted(present)),
                            "grids_missing": ",".join(sorted(missing)),
                        }
                    )
    return pd.DataFrame(rows)


def facet_consistency(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Facets that are populated for some records of a group and blank for others.

    The documentation warns that "parameters are not always internally
    consistent"; this quantifies it. One row per (activity_id, facet) where the
    facet is partially populated.

    Parameters
    ----------
    frame : pandas.DataFrame
        Catalog records.

    Returns
    -------
    pandas.DataFrame
        One row per (``activity_id``, facet) that is populated on some records
        and blank on others, with the blank count and percentage.
    """
    rows: list[dict] = []
    for activity_id in sorted(set(frame["activity_id"].dropna())):
        subset = frame[frame["activity_id"] == activity_id]
        for facet in S.FACETS:
            if facet == "activity_id":
                continue
            values = subset[facet]
            blank = values.isna() | values.astype(str).str.lower().isin({"nan", ""})
            n_blank = int(blank.sum())
            if 0 < n_blank < len(subset):
                rows.append(
                    {
                        "activity_id": activity_id,
                        "facet": facet,
                        "n_populated": int(len(subset) - n_blank),
                        "n_blank": n_blank,
                        "pct_blank": round(100.0 * n_blank / len(subset), 1),
                    }
                )
    return pd.DataFrame(rows)


def timescale_mismatch(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Variables published at a temporal resolution the docs do not list.

    Distinct from "undocumented": the reference table *does* describe the
    variable, just at different ``table_id``s. Either the data or the docs is
    wrong, and which one is a question for the data producer.

    Parameters
    ----------
    frame : pandas.DataFrame
        Catalog records.

    Returns
    -------
    pandas.DataFrame
        One row per variable published at a ``table_id`` the reference table
        does not list for it, with ``published_at`` and ``documented_at``.
    """
    rows: list[dict] = []
    seen = frame[["activity_id", "table_id", "variable_id"]].drop_duplicates()
    for row in seen.itertuples():
        if not isinstance(row.activity_id, str) or not isinstance(row.table_id, str):
            continue
        documented = S.documented_table_ids(row.activity_id, row.variable_id)
        if documented and row.table_id not in documented:
            rows.append(
                {
                    "activity_id": row.activity_id,
                    "variable_id": row.variable_id,
                    "published_at": row.table_id,
                    "documented_at": ",".join(sorted(documented)),
                }
            )
    return pd.DataFrame(rows)


def undocumented_in_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Catalog variables the reference table does not describe at all.

    Variables described at a *different* table_id are excluded here and reported
    by :func:`timescale_mismatch` instead, so each problem is stated once.

    Parameters
    ----------
    frame : pandas.DataFrame
        Catalog records.

    Returns
    -------
    pandas.DataFrame
        One row per variable the reference table does not describe at any
        resolution, with the number of stores holding it.
    """
    rows: list[dict] = []
    seen = frame[["activity_id", "table_id", "variable_id"]].drop_duplicates()
    for row in seen.itertuples():
        if not isinstance(row.activity_id, str) or not isinstance(row.table_id, str):
            continue
        if (
            S.documented_variable(row.activity_id, row.table_id, row.variable_id)
            is not None
        ):
            continue
        if S.documented_table_ids(row.activity_id, row.variable_id):
            continue  # known variable, wrong table_id -> timescale_mismatch
        n = len(
            frame[
                (frame["activity_id"] == row.activity_id)
                & (frame["table_id"] == row.table_id)
                & (frame["variable_id"] == row.variable_id)
            ]
        )
        rows.append(
            {
                "activity_id": row.activity_id,
                "table_id": row.table_id,
                "variable_id": row.variable_id,
                "n_stores": n,
            }
        )
    return pd.DataFrame(rows)


def compare_catalog_sources() -> pd.DataFrame:
    """
    Diff the live catalog against the bundled snapshot.

    Returns a frame with a ``presence`` column of ``live_only`` / ``bundled_only``.
    An empty frame means the snapshot is current.
    """
    live = load_catalog("live")
    bundled = load_catalog("bundled")
    keys = list(S.FACETS)
    live_keys = set(map(tuple, live[keys].astype(str).values))
    bundled_keys = set(map(tuple, bundled[keys].astype(str).values))

    rows: list[dict] = []
    for tup in sorted(live_keys - bundled_keys):
        rows.append(dict(zip(keys, tup), presence="live_only"))
    for tup in sorted(bundled_keys - live_keys):
        rows.append(dict(zip(keys, tup), presence="bundled_only"))
    return pd.DataFrame(rows)
