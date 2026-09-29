# AE Data Quality Guidance

Quarto site for the Data Quality & Documentation Pillar: variable
characterization and usage guidance for the Cal-Adapt Analytics Engine downscaled
data.

```bash
quarto preview      # live reload while editing
quarto render       # build to _site/
```

## Structure

```
index.qmd                        overview — what is covered, and the method in brief
method/
  scope-and-definitions.qmd      the method: six dimensions, reference hierarchy, metric choice
  provenance.qmd                 the shared backbone, plus one branch per variable
  extending.qmd                  what is fixed, what is filled in, checklist
variables/
  wind-speed.qmd                 characterization
  relative-humidity.qmd          characterization
guidance/
  good-bad-matrix.qmd            verdicts, all variables
  variable-cards.qmd             one card per variable name
references.qmd                   R## / Q## / DQ## entries
data/                            the CSV and YAML that drive the above
```

## Adding a variable

One page under `variables/`, one branch in `method/provenance.qmd`, rows in
`data/goodbad_matrix.csv`, an entry in `data/variables.yml`, and a navbar line in
`_quarto.yml`. Nothing else moves.

See `method/extending.qmd` for the checklist and for the two things that are
**not** the same across variables — the metric (ratio vs difference) and the tail
of interest.

## Figures

Figures are Plotly exports embedded as iframes. The `.qmd` names the figure; the
height comes from the export itself via a generated stylesheet, so the notebook's
`style(pf, height=...)` is the single source of truth.

```bash
python figures/write_heights_css.py    # after re-exporting figures
python figures/check_figures.py        # before rendering
```

See `figures/README.md`.

## Dependencies

The rendered pages execute Python: `pandas`, `pyyaml`. `itables` is optional and
the matrix page falls back gracefully without it. Mermaid diagrams render natively
in Quarto.

## Notebooks

The analysis notebooks live outside this folder and write into `figures/`. Every
figure destined for the site is produced by a cell marked:

```python
# Export: dist_daily_stations
save_png(fig, "dist_daily_stations")
save_html(pf, "dist_daily_stations")
```

`save_png` writes `figures/png/`, `save_html` writes `figures/html/`. Nothing
writes to `figures/` directly.
