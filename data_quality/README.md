# data_audit

Documentation-compliance audit for the **Cal-Adapt: Analytics Engine** `cadcat`
archive — WRF and LOCA2-Hybrid, all domains (`d01`/`d02`/`d03`) and all temporal
resolutions (`1hr`/`day`/`mon`/`yrmax`).

Lives at `cae-archives/data_quality/`. The wind-speed and relative-humidity
analysis packages will sit alongside it later; this folder is the audit only.

## What it checks, and against what

Three sources define the standard, and every finding names the one it came from:

- [Climate Model Simulations](https://analytics.cal-adapt.org/data-tools/data-documentation/climate-model-sims.html)
- [Data Structure and Format](https://analytics.cal-adapt.org/data-tools/data-documentation/data-structure-and-format.html)
- [Metadata Standards](https://analytics.cal-adapt.org/data-tools/data-documentation/metadata-standards.html)

plus `variable_descriptions.csv` shipped inside `climakitae`, which is the
machine-readable variable table the Analytics Engine itself uses.

The same checks run over two independent access paths:

| Backend | Reads via | Answers |
| --- | --- | --- |
| `ZarrBackend` | `xarray.open_zarr` on `s3://cadcat` | Is the published artifact correct? |
| `ClimakitaeBackend` | `climakitae.ClimateData` | Is the data usable as delivered? |

Running both and diffing them is the point. climakitae repairs some metadata on
read — it attaches a CRS to stores that lack one — so a defect can exist in the
store without reaching users, or reach users without existing in the store.
`reconcile` classifies which.

## Install

From `cae-archives/data_quality/`:

```bash
pip install -e ".[stores]"      # audit + direct Zarr reads
pip install -e ".[all]"         # + pint, parquet, dev tools
```

The catalog audit runs on the base install alone, from the snapshot bundled
inside `climakitae`. Everything past it needs `[stores]` and network access to
`s3://cadcat` (anonymous — no credentials).

## Run the complete audit

```bash
./scripts/run_audit.sh --outdir audit_$(date +%Y_%m)
```

Eight stages, cheapest first, each writing its own findings CSV:

| stage | what | cost |
| --- | --- | --- |
| 0 | version check — aborts if the installed copy is stale | instant |
| 1 | clear the output directory | instant |
| 2 | catalog audit — vocabulary, path conventions, coverage | seconds, no data reads |
| 3 | coverage matrices — variable x domain x resolution | seconds |
| 4 | empty-store scan — reads values, catches unwritten stores | minutes |
| 5 | metadata sweep, raw Zarr, with store structure | minutes |
| 6 | metadata sweep, climakitae | minutes |
| 6b | reconciliation — both surfaces against the documentation | minutes |
| 7 | backend comparison | minutes |
| 8 | consolidated `all_findings.csv` and `REPORT.md` | instant |

`--html` adds `REPORT.html` and a per-stage HTML report: the same summary plus a
sortable table of every finding, styles inlined and no external assets, so it
can be emailed or dropped on a share as one file.

Options:

```bash
./scripts/run_audit.sh --full            # every facet combination, hours
./scripts/run_audit.sh --workers 8       # threads for stages 4-7 (default 6)
./scripts/run_audit.sh --outdir NAME     # output directory
./scripts/run_audit.sh --html            # also write standalone HTML reports
./scripts/run_audit.sh --skip 4          # skip a stage (repeatable)
./scripts/run_audit.sh --only 2 --only 3 # run only these stages
```

Stages 0, 1 and 8 (version check, clean, summary) always run.

**Skipping stage 4** turns the run into a metadata-only pass and cuts most of
the wall time — it is the only stage that reads values rather than metadata.
The cost is that nothing left in the run can see an empty or partly-written
store, which is the defect class metadata checks are blind to by construction.
Worth skipping for a quick re-check; not worth skipping before reporting to a
data producer.

Nothing is incremental. The output directory is deleted at the start, so a run
is reproducible.

## Individual commands

```bash
python -m data_audit audit --outdir out        # catalog only, seconds
python -m data_audit coverage --outdir out     # presence matrices
python -m data_audit sweep --backend zarr --inspect-store --outdir out
python -m data_audit sweep --backend climakitae --outdir out
python -m data_audit reconcile --limit 40 --outdir out
python -m data_audit compare --limit 60 --outdir out
```

Filters apply to all of them: `--activity WRF`, `--grid-label d03`,
`--table-id day mon`, `--variable u10 v10`, `--experiment ssp370`, `--limit N`,
`--workers N`, `--formats csv md html`.

Exit codes suit CI: `0` clean, `1` warnings only, `2` errors or worse.

### Reading values

Metadata checks cannot see an empty store. `--probe-values` reads a few slices
per store and scans for unwritten Zarr chunks:

```bash
python -m data_audit sweep --backend zarr --probe-values --per-model \
  --activity WRF --table-id day --variable u10 v10 --outdir out
```

`--per-model` matters here. By default the sampler takes one store per
`(activity, institution, table_id, grid_label, variable_id)`, because *metadata*
is a property of those facets. **Data is not** — one driving model's store can be
empty while its siblings are fine, and the default plan will not see it. Always
pass `--per-model` with `--probe-values`.

## Python

```bash
python -c "
from data_audit import load_catalog, filter_catalog, audit_catalog, summarize_by_code
cat = filter_catalog(load_catalog())
print(summarize_by_code(audit_catalog(cat)))
"
```

Every function returns a tidy DataFrame: one row per finding, catalog facets as
columns, plus `code`, `level`, `message`, `expected`, `actual`. Group on `code`;
never parse `message`.

For reading stores, `ZarrBackend` is a context manager and should be used as one
— it holds an s3fs session:

```python
from data_audit import ZarrBackend, build_sample_plan, sweep

with ZarrBackend() as backend:
    findings = sweep(plan, backend, probe_values=True, max_workers=8)
```

## Layout

```
data_quality/
  src/data_audit/
    standards.py           the expectations, encoded from the three doc pages
    catalog.py             load / filter / sample; coverage and consistency
    checks.py              the checks; return findings, never raise
    findings.py            Finding + Level
    backend_zarr.py        direct store reads, plus raw structural inspection
    backend_climakitae.py  the delivered-product view
    runner.py              audit / sweep / compare
    reconcile.py           both surfaces judged against the documentation
    report.py              summaries, markdown, HTML
    cli.py                 python -m data_audit
  scripts/
    run_audit.sh           the full staged run
    audit_docs.py          type-hint and docstring coverage
  tests/test_checks.py     88 tests
  notebooks/               exploratory walkthrough
```

## Tuning

`standards.py` is the single place expectations live — time bounds, nominal
resolutions, required attributes, unit aliases, CRS conventions per activity.
Checks read from it and hard-code nothing.

Two entries were derived from observation rather than documentation, so confirm
them before relying on the findings they produce:

- `EXPECTED_TIME_BOUNDS` — WRF runs September to August (1980-09-01 to
  2014-08-31; 2014-09-01 to 2100-08-31), LOCA2 on calendar years. Compared to
  the day, with `TIME_BOUND_TOLERANCE_DAYS` of slack.
- `EXPECTED_TIME_RANGE` — the older year-granularity check, kept alongside it
  for the coarse case.

## Development

```bash
pytest -q                                          # 88 tests
python scripts/audit_docs.py                       # type-hint and docstring coverage
python scripts/audit_docs.py --min-coverage 100    # CI gate; currently passes
ruff check src tests
```

### Adding a check

Two conventions worth knowing before you write one.

**Checks emit findings; they never raise.** A store that cannot be opened yields
a `FATAL` finding, so a sweep over thousands of stores survives one malformed
store instead of dying on it. Every finding carries a stable `code` — group and
count on that, never parse `message`.

**Expectations live in `standards.py`, not in the check.** Nothing in
`checks.py` should hard-code a threshold, a date, a unit or a vocabulary. That
way the answer to "why did this fail?" is always one file, and adjusting the
standard does not mean touching the logic.