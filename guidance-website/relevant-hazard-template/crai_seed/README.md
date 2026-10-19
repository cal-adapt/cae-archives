# crai_seed — local snapshot of the statewide CRAI inputs

Used by `crai_base.ipynb` when `configs/crai_base.yaml` has `source: local`. Copied 2026-10-19.

| File | Source | Notes |
|---|---|---|
| `adaptive_capacity_index_DE.csv` | `era-scratch/developers/victoria/data/adaptive_capacity_index_DE.csv` (dated 2026-08-28) | 9,129 tracts. Domain scores match the ones inside the July 31 `RECENT_HAZARDS_crai_*` files behind the Sacramento 2020 post, to within 0.00001. |
| `concatenate_climate_metrics_DE.csv` | `era-scratch/developers/victoria/data/concatenate_climate_metrics_DE.csv` (dated 2026-08-28) | Raw (unstandardized) Cal-CRAI climate metrics, 9,129 tracts. |
| `cava_calcrai_metrics.csv` | `era-scratch/developers/victoria/data/cava_calcrai_metrics.csv` | Cal-CRAI metric list (domain; whether a high value is vulnerable or resilient). Only used in `source: s3` mode. |
| `ca_tracts.gpkg` | `geometry` column of `guidance-website/blog/2026-08-13-sac-extreme-heat/RECENT-HAZARDS_crai_layers_extreme_heat_sacramento/RECENT_HAZARDS_crai_heat_counts_gwl2.csv` | 9,129 statewide tracts, EPSG:4326. Git-ignored (58 MB). |

These are snapshots, not recomputed — switch `configs/crai_base.yaml` to `source: s3` for a real refresh.
