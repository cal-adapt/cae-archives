# Figures

```
figures/
  png/                    fig_name.png    static — slides, print, PDF
  html/                   fig_name.html   interactive — what the site embeds
  heights.css             GENERATED — do not edit
  heights.json            the heights, for the checker
  write_heights_css.py    regenerates heights.css from the exports
  check_figures.py        consistency check
```

**Nothing is written to `figures/` itself.** A loose PNG beside the two
subfolders is how the tree ended up duplicated once; `check_figures.py` now fails
if one appears.

## How a figure is embedded

The `.qmd` carries the figure **name and nothing else**:

```markdown
::: {#fig-dist-daily .column-page-inset}
```{=html}
<div class="fig-frame" data-fig="dist_daily_stations">
  <iframe src="../figures/html/dist_daily_stations.html" loading="lazy" scrolling="no"></iframe>
</div>
```
Caption, which can run to several lines.
:::
```

No height, no aspect ratio, no inline style.

## Where the height comes from

The notebook already decides it:

```python
return style(pf, height=620)
```

Plotly writes that into the export, and `write_heights_css.py` reads it back out
into one rule per figure:

```css
.fig-frame[data-fig="dist_daily_stations"] { height: 716px; }
```

So **the notebook is the single source of truth**. Change `height=` there,
re-export, re-run the script, and the site follows. No scrollbars, because the
frame is exactly as tall as the figure asked to be.

## After re-exporting

```bash
python figures/write_heights_css.py    # regenerate heights.css
python figures/check_figures.py        # verify
quarto render
```

## Breakout classes

`.column-page` / `.column-page-inset` let a wide figure extend past the text
column. Applied per figure in the `.qmd`, by how wide it is relative to its
height: a multi-panel figure gets ~280px per panel in the default column and
~400px at `.column-page`.

Tall figures are left in the default column — breaking them out only floats them
in whitespace.

## What the export must do

`style()` in the notebook already handles this. For reference:

```python
pf.update_layout(height=620, width=None, autosize=True)
pf.update_xaxes(automargin=True)
pf.update_yaxes(automargin=True)
pf.write_html(path, include_plotlyjs="cdn", full_html=True,
              config={"responsive": True, "displaylogo": False})
```

- **`width=None`** so the figure fills the frame horizontally.
- **`automargin=True`** expands the margin to fit the axis furniture, which is
  what stops labels being clipped.
- **`height=`** is what the site reads. Leaving it unset makes the figure
  responsive in both directions, and `write_heights_css.py` then falls back to
  450px.

## Checking

```bash
python figures/check_figures.py
```

| exit | meaning |
|---|---|
| 2 | an embedded HTML is missing, or a loose PNG is sitting in `figures/` |
| 1 | a PNG is missing, or a figure has no height rule |
| 0 | clean |
