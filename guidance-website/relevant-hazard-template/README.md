# Extreme-event post framework

`cae-archives/guidance-website/relevant-hazard-template/`

One config + one notebook per post. Replaces `guidance-website/blog/_templates/` and the
era-scratch CRAI notebook chain (both left untouched for reference).

## Layout

```
guidance-website/
├── relevant-hazard-template/          ← this folder (the framework)
│   ├── heat_event.ipynb               per-event notebook for extreme heat (copied into each post)
│   ├── crai_base.ipynb                one-time statewide CRAI inputs → crai_cache/
│   ├── heat_index_template.qmd        starting index.qmd for a heat post
│   ├── run_event.sh                   runs a post's notebook; --publish sends it to the guidance site
│   ├── helpers/
│   │   ├── event_helpers.py           shared by the event notebooks
│   │   └── crai_helpers.py            Cal-CRAI calculation functions
│   ├── configs/
│   │   ├── crai_base.yaml             local/S3 toggle for crai_base.ipynb
│   │   └── heat_sacramento_2020.yaml  worked example event config
│   ├── crai_seed/                     local snapshot of the statewide CRAI inputs (see its README)
│   └── crai_cache/                    built by crai_base.ipynb (git-ignored)
└── blog/
    └── <post>/                        one folder per post (see "Start a new post")
```

| File | Role |
|---|---|
| `heat_event.ipynb` | Per-event notebook for extreme heat (copied into each post) |
| `precip_event.ipynb` | Per-event notebook for extreme precipitation — *not built yet* |
| `crai_base.ipynb` | One-time statewide CRAI inputs → `crai_cache/` (rerun only when Cal-CRAI source data changes) |
| `helpers/event_helpers.py` | Code shared by the event notebooks: config + output filenames, station and WRF pulls, event counting, tract mapping, CRAI scores, figures |
| `helpers/crai_helpers.py` | Cal-CRAI calculation functions (merged from era-scratch `file_helpers.py` + `calc_*.py`) |
| `heat_index_template.qmd` | Starting `index.qmd` for a heat post, with every notebook figure already embedded (HTML + mobile/PDF/DOCX PNGs) |
| `run_event.sh` | Runs a post's notebook under the `climakitae` env; `--publish` moves its figures and copies its `index.qmd` to the guidance site |
| `configs/crai_base.yaml` | `source: local` (build from `crai_seed/`) or `source: s3` (real refresh from S3) |
| `configs/heat_sacramento_2020.yaml` | Worked example event config — copy it to start a post |
| `crai_seed/` | Snapshot CRAI inputs + the Cal-CRAI metric list (`cava_calcrai_metrics.csv`) |
| `crai_cache/` | Built by `crai_base.ipynb` (git-ignored) |

## Start a new post

1. Once per machine (or when the Cal-CRAI data changes): run
   `guidance-website/relevant-hazard-template/crai_base.ipynb`. With
   `configs/crai_base.yaml` at `source: local` it builds `crai_cache/` from `crai_seed/` in
   seconds; `source: s3` re-downloads and recomputes everything (needs AWS credentials).
2. Make the post folder, e.g. `guidance-website/blog/2027-07-01-fresno-extreme-heat/`.
3. Copy `relevant-hazard-template/configs/heat_sacramento_2020.yaml` into it as
   `event_config.yaml` and edit it.
4. Put a zip-code shapefile covering the map area in the post folder (`location.zip_shapefile`),
   and set `publish.guidance_post_dir` to the post's folder in
   `cal-adapt-guidance/blog/posts/` (relative to the post folder:
   `../../../../cal-adapt-guidance/blog/posts/<post>`).
5. From `guidance-website/blog/`:
   `../relevant-hazard-template/run_event.sh 2027-07-01-fresno-extreme-heat --publish`
   — runs the notebook, then publishes to the guidance post folder:
   - **moves** every `figures/{html,static}/fig_*` file (an empty source folder means
     nothing new to publish)
   - **copies** the post's `index.qmd` and `images/` (overwriting the old copy)

   To watch it run cell by cell instead, open `guidance-website/blog/<post>/heat_event.ipynb`
   in VS Code with the `climakitae` kernel and Run All (copy it in from this folder first if
   `run_event.sh` hasn't yet).
6. Write the post in `guidance-website/blog/<post>/index.qmd` — it's the source of truth,
   started from `heat_index_template.qmd` on the first run. Put photos in
   `guidance-website/blog/<post>/images/`. After edits, from `guidance-website/blog/`:
   `../relevant-hazard-template/run_event.sh <post> --publish-only`, then preview from
   `cal-adapt-guidance/`: `quarto preview blog/posts/<post>/index.qmd`. Never edit the
   guidance copy — every publish overwrites it.

Figures are named `fig_<name>.{html,png}` — the same names the template embeds:
`daily_temp_by_year`, `annual_anomaly`, `return_period_by_gwl`, `hot_days_hot_nights_by_gwl`,
`heatwave_events_by_gwl`, `hot_days_hot_nights_delta_by_gwl`, `heatwave_events_delta_by_gwl`.

Outputs land in the post folder:

```
guidance-website/blog/<post>/
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
  crai_seed/ (local) or S3 (refresh)        Part 1  Observed event (HDP station)
  statewide tracts                          Part 2  Historical context
  adaptive-capacity domain scores           Part 3  Return period (WRF, pulled once)
  standardized climate metrics              Part 4  Event metrics → census tracts
        └──▶ crai_cache/ ──────────────▶    Part 5  Hazard score + CRAI index
                                            Part 6–7 Narrative (TODO)
```
