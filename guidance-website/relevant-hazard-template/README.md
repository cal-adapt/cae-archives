# Relevant Hazard Post Generation
(last updated: 11/2/26)

Hi there! This README walks you through how to generate a new relevant hazard post for a specific event of interest. This currently supports two types of events: **extreme heat and extreme precipitation.**

## Layout

```
guidance-website/
├── relevant-hazard-template/          ← this folder (the framework)
│   ├── heat_event.ipynb               per-event notebook for extreme heat (copied into <post>/_analysis/)
│   ├── heat_index_template.qmd        starting index.qmd for a heat post
│   ├── precip_event.ipynb             per-event notebook for extreme precipitation
│   ├── precip_index_template.qmd      starting index.qmd for a precip post
│   ├── crai_base.ipynb                one-time statewide CRAI inputs → crai_cache/
│   ├── run_event.sh                   creates a post folder (--new heat|precip)
│   ├── helpers/
│   │   ├── event_helpers.py           shared by the event notebooks
│   │   └── crai_helpers.py            Cal-CRAI calculation functions
│   ├── configs/
│   │   ├── crai_base.yaml             local/S3 toggle for crai_base.ipynb
│   │   ├── heat_sacramento_2020.yaml  worked example heat config (Aug and Sept 2020 Sac heatwaves)
│   │   ├── precip_sacramento_2021.yaml  worked example precip config (Oct 2021 Sac atmospheric river)
│   │   └── sac_zipcodes.zip           example zip-code shapefile (Sacramento), copied by --new
│   ├── tests/                         offline tests for helpers/event_helpers.py
│   ├── crai_seed/                     local snapshot of the statewide CRAI inputs (see its README)
│   └── crai_cache/                    built by crai_base.ipynb (git-ignored)
└── blog/
    └── <post>/                        one folder per post — copy it whole into cal-adapt-guidance/blog/posts/
```

| File | Role |
|---|---|
| `heat_event.ipynb` | Per-event notebook for extreme heat (copied into each post's `_analysis/`) |
| `precip_event.ipynb` | Per-event notebook for extreme precipitation (station precip is HDP `pr` used as-is) |
| `crai_base.ipynb` | One-time statewide CRAI inputs → `crai_cache/` (rerun only when Cal-CRAI source data changes) |
| `helpers/event_helpers.py` | Code shared by the event notebooks: config + output filenames, station and WRF pulls, event counting, tract mapping, CRAI scores, figures |
| `helpers/crai_helpers.py` | Cal-CRAI calculation functions (merged from era-scratch `file_helpers.py` + `calc_*.py`) |
| `heat_index_template.qmd`, `precip_index_template.qmd` | Starting `index.qmd` for a post, with every notebook figure already embedded (HTML + mobile/PDF/DOCX PNGs) |
| `run_event.sh` | `--new heat\|precip` creates a post folder |
| `configs/crai_base.yaml` | `source: local` (build from `crai_seed/`) or `source: s3` (real refresh from S3) |
| `configs/heat_sacramento_2020.yaml`, `configs/precip_sacramento_2021.yaml` | Worked example event configs. `--new` copies one into the post |
| `configs/sac_zipcodes.zip` | Sacramento zip codes used by both worked examples — `--new` copies it into the post |
| `tests/` | Tests to confirm accuracy and runnability of helper functions |
| `crai_seed/` | Snapshot CRAI inputs + the Cal-CRAI metric list (`cava_calcrai_metrics.csv`) |
| `crai_cache/` | Built by `crai_base.ipynb` (git-ignored) |

## Start a new post

1. **Build the CRAI cache** (once per machine, or when the Cal-CRAI data changes)

   Run `guidance-website/relevant-hazard-template/crai_base.ipynb`. With
   `configs/crai_base.yaml` at `source: local` it builds `crai_cache/` from `crai_seed/` in
   seconds; `source: s3` re-downloads and recomputes everything (needs AWS credentials).

2. **Create the post folder**

   From `guidance-website/blog/`:
   ```bash
   ../relevant-hazard-template/run_event.sh 2027-07-01-fresno-extreme-heat --new heat
   ```
   Use `--new precip` for a precip post. This sets up `index.qmd`, an empty `images/`
   folder, and an `_analysis/` folder holding `event_config.yaml`, the notebook, and the
   example `sac_zipcodes.zip`.

3. **Edit the config**

   Edit `_analysis/event_config.yaml` for your event. See "Setting up a new event" below.

4. **Run the notebook**

   Open `_analysis/heat_event.ipynb` (or `precip_event.ipynb`) in VS Code with the
   `climakitae` kernel and Run All. It needs an internet connection (HDP station data, WRF
   projections, census tracts and NOAA's station list are all read remotely). The figures
   are saved to `<post>/figures/`, where `index.qmd` already expects them.

5. **Write the post**

   Write the post in `index.qmd` and put any photos in `images/`.

6. **Copy the folder to cal-adapt-guidance**

   Copy the whole post folder into `cal-adapt-guidance/blog/posts/`, then preview it from
   `cal-adapt-guidance/`:
   ```bash
   quarto preview blog/posts/<post>/index.qmd
   ```
   Quarto skips `_analysis/`, so it's safe to copy (or delete it from the copy to save space).

Figures are named `fig_<name>.{html,png}`. Figure names by event type are:
- **heat:** `daily_temp_by_year`, `annual_anomaly`, `return_period_by_gwl`, `hot_days_hot_nights_by_gwl`,
  `heatwave_events_by_gwl`, `hot_days_hot_nights_delta_by_gwl`, `heatwave_events_delta_by_gwl`
- **precip** (per event definition `<id>`, e.g. `3day_p99`): `cumulative_precip_by_year`,
  `annual_precip_anomaly`, `return_period_<id>`, `<id>_events_by_gwl`, `<id>_events_delta_by_gwl`

<br>

**What a finished post folder looks like:**

```
guidance-website/blog/<post>/
├── index.qmd                   post text
├── images/                     photos
├── figures/
│   ├── html/fig_*.html         interactive figures
│   └── static/fig_*.png        mobile / PDF / DOCX versions
└── _analysis/                  skipped by Quarto
    ├── event_config.yaml
    ├── heat_event.ipynb        (or precip_event.ipynb), executed in place
    ├── <zip codes>.zip
    └── crai_layers/
        ├── tract_metrics/      event metric per tract, per warming level
        ├── hazard_scores/      CRAI climate hazard score per tract
        └── crai_scores/        CRAI index per tract
```

## Setting up a new event

Go through `_analysis/event_config.yaml` top to bottom. The worked example it was copied from is filled in for Sacramento, so change every entry that doesn't fit your event.

- **`event_name`, `place_name`**: free text used in titles. `place_name` also goes into output file names.
- **`station_label`**: the airport's 4-letter code (`K` + its 3-letter code, e.g. `KFAT` for Fresno). Leave it empty to use the station inside `clip` with the longest record. If the code isn't in HDP, Part 1 stops with an error.
- **`network_id`**: `ASOSAWOS` (default) or `OtherISD`. Some airports are only in one of them. If your code isn't found, try the other.
- **`clip`**: the county, written `"<Name> County"` (e.g. `"Fresno County"`). The WRF
  projections are cut to it, and an empty `station_label` searches it for a station.
- **`zip_shapefile`**: a zip-code shapefile (`.zip`) covering the area to map, placed in
  `_analysis/`. Counties' GIS or open-data portals usually publish one. It needs a `ZIP5`
  (zip code) and `PO_NAME` (place name) column; if yours names them differently, rename
  them first:
  ```python
  import geopandas as gpd
  z = gpd.read_file("county_zips.zip").rename(columns={"ZIP_CODE": "ZIP5", "CITY": "PO_NAME"})
  z[["ZIP5", "PO_NAME", "geometry"]].to_file("my_zipcodes.zip", driver="ESRI Shapefile")
  ```
  The maps cover the census tracts whose center falls inside these zip codes.
- **`event.year`, `event.windows`**: the event year and a `[start, end]` pair per
  heat wave or storm (both dates included).
- **Thresholds**: what counts as an event at this place (see "Config files" below for each
  setting). Run Part 1 first with a first guess. It prints the station's record dates. Check that they cover your event's dates and each window's peak values (heat: peak max temperature and the number of days and nights past the thresholds; precip: total, max 1-day and max N-day rainfall). Adjust the thresholds, then Run All.
- **`projection`**: the example values work anywhere in California.
  - `warming_levels`: up to four, lowest first.
  - `baseline_warming_level`: your baseline WL to compare from.
  - `sublabels`: one per warming level, or they're dropped. Update `index.qmd` to match.
  - `exclude_simulations`: keep `EC-Earth3-Veg` while 0.8°C is the baseline (no data there).

**Precip only:** `index.qmd` embeds the figures of the example event `3day_p99`. If your
config uses a different `id`, replace `3day_p99` with it throughout `index.qmd`. For each
extra entry in `thresholds.events`, copy the return-period, events-map and delta-map
blocks and change the `id` in the copies.

## Config files

Each post's `_analysis/event_config.yaml` starts as a copy of one of the two worked
examples in `configs/`. Both share the settings below and differ in their `thresholds:`
sections (heat also sets `projection.percentile`).

### Shared settings

```yaml
hazard: heat                       # heat | precip — picks the notebook and post template
event_name: "Sacramento 2020"

location:
  station_label: "KSAC"            # airport code of the HDP weather station (may be empty)
  network_id: ASOSAWOS             # optional: ASOSAWOS (default) | OtherISD
  place_name: "Sacramento"         # used in titles and file names
  clip: "Sacramento County"        # area the WRF projections are cut to
  zip_shapefile: "sac_zipcodes.zip"  # zip codes for the maps (in _analysis/)

event:
  year: 2020
  windows:                         # [start, end] of each event, any number
    - ["2020-08-14", "2020-08-26"]
    - ["2020-09-05", "2020-09-08"]

projection:
  warming_levels: [0.8, 1.5, 2.0]  # °C
  warming_level_window: 15         # years either side of each warming level
  baseline_warming_level: 0.8
  sublabels: ["Historical", "Today", "Mid-Century"]
  exclude_simulations: ["EC-Earth3-Veg"]   # no 0.8°C coverage
```

### Heat (`configs/heat_sacramento_2020.yaml`)

```yaml
thresholds:
  tmax_f: 105                      # hot day: daily max above this (°F)
  tmin_f: 75                       # hot night: daily min at or above this (°F)
  consecutive_days: 1              # days in a row that make one heatwave event

projection:
  percentile: 0.999                # heat only, see below
```

A qualifying day has both a hot day and a hot night. `tmax_f` and `tmin_f` are used directly
for the station record (Parts 1–2) and for the tract maps and CRAI scores (Parts 4–5). The
return period (Part 3) instead gives each simulation and grid cell its own max and min
thresholds: the `percentile` of its temperatures at `baseline_warming_level`. This way, each
model is judged against its own historical climate.

### Precip (`configs/precip_sacramento_2021.yaml`)

`thresholds.events` is a list of entries, each with its own return period, tract map, CRAI scores, and delta map:

```yaml
thresholds:
  wet_day_in: 0.01                 # wet day = at least this much precipitation (in)
  events:
    - id: 3day_p99                 # used in figure/output filenames
      label: "3-day rainfall above the 99th percentile"
      accumulation_days: 3         # rolling window (days)
      percentile: 0.99             # of N-day totals ending on a wet day
    - id: 3day_6in                 # or a fixed amount instead of a percentile
      label: "3-day rainfall above 6 inches"
      accumulation_days: 3
      threshold_in: 6
```

An event is a stretch of days whose N-day total exceeds the threshold; back-to-back
qualifying days count once. Each entry uses one of:
- `percentile`: the station threshold comes from the HDP record, and the projections use a
  threshold calibrated per simulation and grid cell at `baseline_warming_level`. This removes
  model wet/dry bias, so the baseline map is near-uniform and the change across warming
  levels is the signal.
- `threshold_in`: the same amount for the station and every grid cell. Maps show where that
  amount is reached most often, but model bias is kept.

The precip example uses Sacramento International (`KSMF`) from the `OtherISD` network. One
airport code can have several station IDs in a network (they get renumbered when a station
moves); the one with the longest record is used.

Station precipitation is HDP `pr` used as-is. HDP double-counts some reports, so station
totals run high.


## Tests

Offline tests (synthetic data, no network) for `helpers/event_helpers.py`. From
`guidance-website/`:
```bash
conda run -n climakitae pytest relevant-hazard-template/tests
```

## Data flow

```
crai_base.ipynb (rarely)                  heat_event.ipynb / precip_event.ipynb (per post)
  crai_seed/ (local) or S3 (refresh)        Part 1  Observed event (HDP station)
  adaptive-capacity domain scores           Part 2  Historical context
  standardized climate metrics              Part 3  Return period (WRF, pulled once)
                                            Part 4  Event metrics → census tracts (census.gov)
        └──▶ crai_cache/ ──────────────▶    Part 5  Hazard score + CRAI index
                                            Part 6–7 Narrative (TODO)
                                                    │
                                                    ▼
                                  <post>/figures/ → embedded in <post>/index.qmd
```
