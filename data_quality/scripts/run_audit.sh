#!/usr/bin/env bash
#
# Full cadcat QC run, from scratch.
#
#   ./scripts/run_audit.sh                      quick pass, a few minutes
#   ./scripts/run_audit.sh --full               every facet combination, hours
#   ./scripts/run_audit.sh --outdir NAME        name the output directory
#   ./scripts/run_audit.sh --workers 8          threads for the network stages
#   ./scripts/run_audit.sh --html               also write standalone HTML reports
#   ./scripts/run_audit.sh --skip 4             skip a stage (repeatable)
#   ./scripts/run_audit.sh --only 2 --only 3    run only these stages
#
# Stages: 2 catalog audit, 3 coverage, 4 empty-store scan, 5 zarr sweep,
# 6 climakitae sweep, 6b reconciliation, 7 backend comparison. Stages 0, 1
# and 8 (version check, clean, summary) always run.
#
# Stage 4 is the slow one: it reads values rather than metadata, so it costs
# more than every other stage combined. --skip 4 is reasonable for a metadata
# pass, but then nothing in the run can see an empty or partly-written store.
#
# --workers defaults to 6 and applies to stages 4-7. Raising it much past 8
# does not help: S3 throttles, and the quick pass only opens 40 stores, so the
# plan runs out before the threads do.
#
# Nothing here is incremental. Each stage rebuilds from the catalog, so the
# output directory is deleted at the start and the run is reproducible.

set -euo pipefail

OUTDIR="audit_output"
MODE="quick"
WORKERS=6
SKIP=""
ONLY=""
FORMATS="csv md"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)    MODE="full"; shift ;;
    --outdir)  OUTDIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --skip)    SKIP="$SKIP $2"; shift 2 ;;
    --only)    ONLY="$ONLY $2"; shift 2 ;;
    --html)    FORMATS="csv md html"; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Stage 3+ opens stores over the network. In quick mode cap the plan; in full
# mode let it run over every facet combination.
if [[ "$MODE" == "quick" ]]; then
  LIMIT=(--limit 40)
else
  LIMIT=()
fi

# Stages 2 and 3 are catalog-only; 4-7 read from S3. Stage 4 is by far the
# slowest, because it reads values rather than metadata.
want_stage() {
  local stage="$1"
  if [[ -n "$ONLY" ]]; then
    [[ " $ONLY " == *" $stage "* ]] || return 1
  fi
  [[ " $SKIP " != *" $stage "* ]] || return 1
  return 0
}

echo "=== data_audit: mode=$MODE outdir=$OUTDIR workers=$WORKERS${SKIP:+ skip:$SKIP}${ONLY:+ only:$ONLY}"
echo

# --- 0. confirm the installed package is the current one -------------------
echo "--- 0. version check"
python - <<'EOF'
import data_audit
from data_audit import standards as S

print(f"data_audit {data_audit.__version__}")

# Each of these was added or corrected after a live finding. If any is missing,
# an older copy is installed and the run will reproduce old false positives.
required = {
    "per-domain coordinate extent": lambda: S.plausible_extent("d01"),
    "reanalysis window 1980-2020": lambda: S.EXPECTED_TIME_RANGE[("WRF", "reanalysis")],
    "projected + geographic grids": lambda: S.ACCEPTED_DIM_SETS["WRF"],
    "staging path segments": lambda: S.STAGING_PATH_SEGMENTS,
    "wind component vocabulary": lambda: S.WIND_COMPONENT_VARIABLES,
}
missing = []
for label, probe in required.items():
    try:
        probe()
    except Exception:
        missing.append(label)

if missing:
    raise SystemExit(
        "Installed data_audit is out of date; missing: "
        + ", ".join(missing)
        + "\nReinstall with: pip install -e ."
    )
print("all post-finding corrections present")
EOF
echo

# --- 1. clean --------------------------------------------------------------
echo "--- 1. clearing $OUTDIR"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
echo

# --- 2. catalog audit: no data reads, seconds ------------------------------
echo "--- 2. catalog audit"
python -m data_audit audit --outdir "$OUTDIR" --formats $FORMATS || true
echo

# --- 3. coverage matrices --------------------------------------------------
echo "--- 3. coverage"
python -m data_audit coverage --outdir "$OUTDIR"
echo

# --- 4. empty-store scan ---------------------------------------------------
# Runs first among the network stages because an empty store outranks every
# metadata problem, and this is the cheapest way to find one.
if want_stage 4; then
  echo "--- 4. empty-store scan (reads data)"
  # --per-model is mandatory here. Metadata is a property of the variable, but
  # DATA is a property of the simulation: one driving model's store can be
  # entirely unwritten while its siblings are fine. Without this the sampler
  # takes one model per variable and cannot see that by construction.
  python -m data_audit sweep --backend zarr --probe-values --no-time --per-model \
    --workers "$WORKERS" "${LIMIT[@]}" \
    --outdir "$OUTDIR/empty" --formats csv || true
fi
echo

# --- 5. metadata sweep, raw Zarr ------------------------------------------
if want_stage 5; then
  echo "--- 5. metadata sweep (zarr, with store structure)"
  python -m data_audit sweep --backend zarr --inspect-store \
    --workers "$WORKERS" "${LIMIT[@]}" \
    --outdir "$OUTDIR/zarr" --formats $FORMATS || true
fi
echo

# --- 6. metadata sweep, climakitae ----------------------------------------
if want_stage 6; then
  echo "--- 6. metadata sweep (climakitae)"
  python -m data_audit sweep --backend climakitae \
    --workers "$WORKERS" "${LIMIT[@]}" \
    --outdir "$OUTDIR/ckae" --formats $FORMATS || true
fi
echo

# --- 6b. documentation compliance across both surfaces --------------------
# The two sweeps above each judge one surface. This judges both against the
# documentation and says which surface a divergence belongs to.
if want_stage 6b; then
  echo "--- 6b. reconciliation (store vs delivered, both vs the docs)"
  python -m data_audit reconcile --limit 40 --workers "$WORKERS" \
    --outdir "$OUTDIR" || true
fi
echo

# --- 7. backend comparison -------------------------------------------------
if want_stage 7; then
  echo "--- 7. backend comparison"
  python -m data_audit compare --limit 60 --workers "$WORKERS" \
    --outdir "$OUTDIR" || true
fi
echo

# --- 8. consolidated summary ----------------------------------------------
echo "--- 8. summary"
python - "$OUTDIR" "${FORMATS##* }" <<'EOF'
import sys
from pathlib import Path

import pandas as pd

outdir = Path(sys.argv[1])
frames = []
for path in sorted(outdir.rglob("*_findings.csv")):
    frame = pd.read_csv(path)
    stage = path.parent.name if path.parent != outdir else "catalog"
    frame["stage"] = stage
    if stage == "empty":
        # Stage 4 uses the same backend and sample plan as stage 5, so its
        # metadata findings are the same rows twice. Keep only the value-read
        # findings, which no other stage produces.
        frame = frame[frame["code"].astype(str).str.startswith("data.")]
    frames.append(frame)

if not frames:
    print("no findings files produced")
    raise SystemExit(0)

findings = pd.concat(frames, ignore_index=True)
findings.to_csv(outdir / "all_findings.csv", index=False)

print(f"{len(findings):,} finding(s) across {findings.stage.nunique()} stage(s)")
print()
print(findings.groupby(["level", "code"]).size().to_string())

serious = findings[findings.level.isin(["FATAL", "ERROR"])]
if not serious.empty:
    print("\n--- errors by dataset")
    keys = [k for k in ("activity_id", "institution_id", "table_id",
                        "grid_label", "variable_id") if k in serious.columns]
    print(serious.groupby(keys).size().sort_values(ascending=False).head(25).to_string())
else:
    print("\nno errors")

from data_audit.report import render_html, render_markdown

TITLE = "cadcat WRF + LOCA2 quality report"
(outdir / "REPORT.md").write_text(
    render_markdown(findings, title=TITLE), encoding="utf-8"
)
if len(sys.argv) > 2 and sys.argv[2] == "html":
    # Standalone: styles inlined, no assets, safe to email or drop on a share.
    (outdir / "REPORT.html").write_text(
        render_html(findings, title=TITLE), encoding="utf-8"
    )
    print(f"wrote {outdir / 'REPORT.html'}")
print(f"\nwrote {outdir / 'all_findings.csv'}")
print(f"wrote {outdir / 'REPORT.md'}")
EOF