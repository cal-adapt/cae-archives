# Data files

These drive the rendered pages. Edit the file, re-render, done.

| file | drives |
|---|---|
| `goodbad_matrix.csv` | `guidance/good-bad-matrix.qmd` — one row per variable × use case × region |
| `variables.yml` | `guidance/variable-cards.qmd` — one entry per variable name a user has in hand |
| `lit_review.csv` | `references.qmd` — exported from the internal lit-review database |
| `open_questions.csv` | `references.qmd` — the DQ## entries |
| `audit/` | `guidance/catalog-audit.qmd` — output of one `cadcat_qc` run |

## Expected columns

**`goodbad_matrix.csv`** — `Variable`, `Use case`, `Region / terrain`,
`Dimensions assessed`, `Verdict`, `Evidence`, `Open questions`, `Status`.

`Verdict` takes `Good for` / `Use with caution` / `Not recommended` / `TBD`;
those strings are colour-coded on the variable cards, so keep them exact.

Set `Status` to *IOU-vetted* only after engagement confirms a verdict with a real
user.

## Refreshing the catalog audit

`audit/` holds one run's output, copied in whole. To refresh:

```bash
python -m cadcat_qc audit   --outdir qc_output
python -m cadcat_qc sweep   --backend climakitae --outdir qc_output
python -m cadcat_qc sweep   --backend zarr --inspect-store --outdir qc_output
python -m cadcat_qc compare --outdir qc_output
cp qc_output/{all_findings,reconciliation,documented_but_absent}.csv data/audit/
```

The page reads those three; the rest are kept for reference.

**Update the generation date in the page footer when you refresh.** The findings
are a snapshot and the catalog moves — a page reporting 276 errors with no date
on it will be wrong and look authoritative.
