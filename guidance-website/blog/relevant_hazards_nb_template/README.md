# Extreme-event post framework

One config + one notebook per post. Replaces `blog/_templates/` and the era-scratch
CRAI notebook chain (both left untouched for reference).

## Files

| File | Role |
|---|---|
| `heat_event.ipynb` | Per-event notebook for extreme heat (copied into each post) |
| `precip_event.ipynb` | Per-event notebook for extreme precipitation — *not built yet* |
| `crai_base.ipynb` | One-time statewide CRAI inputs → `crai_cache/` (rerun only when Cal-CRAI source data changes) |
| `event_helpers.py` | Code shared by the event notebooks: config + output filenames, station and WRF pulls, event counting, tract mapping, CRAI scores, figures |
| `crai_helpers.py` | Cal-CRAI calculation functions (merged from era-scratch `file_helpers.py` + `calc_*.py`) |
| `heat_index_template.qmd` | Starting `index.qmd` for a heat post, with every notebook figure already embedded (HTML + mobile/PDF/DOCX PNGs) |
| `run_event.sh` | Runs a post's notebook under the `climakitae` env; `--publish` moves its figures to the guidance site |
| `cava_calcrai_metrics.csv` | Cal-CRAI metric list (copied from era-scratch `data/`) used by `crai_base.ipynb` |
| `configs/` | Worked example configs |
| `crai_cache/` | Built by `crai_base.ipynb` (git-ignored) |

## Start a new post

1. Once per machine (or when the Cal-CRAI data changes): run `crai_base.ipynb` in this
   folder. Needs AWS credentials.
2. Make the post folder, e.g. `blog/2027-07-01-fresno-extreme-heat/`.
3. Copy `configs/heat_sacramento_2020.yaml` into it as `event_config.yaml` and edit it.
4. Put a zip-code shapefile covering the map area in the post folder (`location.zip_shapefile`),
   and set `publish.guidance_post_dir` to the post's folder in `cal-adapt-guidance/blog/posts/`.
5. From `blog/`: `relevant_hazards_nb_template/run_event.sh 2027-07-01-fresno-extreme-heat --publish`
   — runs the notebook, then publishes to the guidance post folder:
   - **moves** every `figures/{html,static}/fig_*` file (an empty source folder means
     nothing new to publish)
   - **copies** the post's `index.qmd` and `images/` (overwriting the old copy)
6. Write the post in **this repo's** `<post>/index.qmd` — it's the source of truth, started
   from `heat_index_template.qmd` on the first run. Put photos in `<post>/images/`.
   After edits: `run_event.sh <post> --publish-only`, then preview in `cal-adapt-guidance`
   (`quarto preview blog/posts/<post>/index.qmd`). Never edit the guidance copy — every
   publish overwrites it.

Figures are named `fig_<name>.{html,png}` — the same names the template embeds:
`daily_temp_by_year`, `annual_anomaly`, `return_period_by_gwl`, `hot_days_hot_nights_by_gwl`,
`heatwave_events_by_gwl`, `hot_days_hot_nights_delta_by_gwl`, `heatwave_events_delta_by_gwl`.

Outputs land in the post folder:

```
<post>/
├── event_config.yaml
├── index.qmd                   post text — source of truth, copied out by --publish
├── images/                     optional photos, copied out by --publish
├── heat_event.ipynb            (copied in on first run, executed in place)
├── figures/static/*.png, figures/html/*.html   (moved out by --publish)
└── crai_layers/
    ├── tract_metrics/          event metric per tract, per warming level
    ├── hazard_scores/          CRAI climate hazard score per tract
    └── crai_scores/            CRAI index per tract
```

## Data flow

```
crai_base.ipynb (rarely)                  heat_event.ipynb (per post)
  S3 Cal-CRAI metrics                       Part 1  Observed event (HDP station)
  statewide tracts                          Part 2  Historical context
  adaptive-capacity domain scores           Part 3  Return period (WRF, pulled once)
  standardized climate metrics              Part 4  Event metrics → census tracts
        └──▶ crai_cache/ ──────────────▶    Part 5  Hazard score + CRAI index
                                            Part 6–7 Narrative (TODO)
```
