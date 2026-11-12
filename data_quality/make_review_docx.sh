#!/usr/bin/env bash
# Build a single reviewable .docx from the rendered Quarto site.
# Usage:  ./make_review_docx.sh  [path/to/data_quality_guidance]  [output.docx]
# Prereqs: quarto, pandoc, python3 (bs4), and mmdc (mermaid-cli, via npm) on PATH.
set -euo pipefail

GUID="${1:-data_quality_guidance}"
OUT="${2:-Data_Quality_Review.docx}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"

# 1. Make sure the site is rendered (produces _site/ with all tables/figures)
if [ ! -d "$GUID/_site" ]; then
  echo "Rendering Quarto site..."
  ( cd "$GUID" && quarto render )
fi

# 2. Render the provenance Mermaid diagram to PNG (it is client-side on the web)
python3 -I - "$GUID" "$WORK" <<'PY'
import sys, re, os
guid, work = sys.argv[1], sys.argv[2]
qmd = os.path.join(guid, "method", "provenance.qmd")
m = re.search(r"```\{mermaid\}(.*?)```", open(qmd, encoding="utf-8").read(), re.DOTALL)
open(os.path.join(work, "prov.mmd"), "w").write(m.group(1).strip() if m else "")
PY
if [ -s "$WORK/prov.mmd" ]; then
  mmdc -i "$WORK/prov.mmd" -o "$WORK/prov.png" -b white -w 1400 >/dev/null 2>&1 || true
fi

# 3. Consolidate the rendered HTML pages (swap figure iframes -> PNG, mermaid -> PNG)
python3 -I "$HERE/build_docx.py" "$GUID/_site" "$GUID" "$WORK/prov.png" "$WORK/combined.html"

# 4. One pandoc call -> docx (resource-path lets it find the figure PNGs)
pandoc "$WORK/combined.html" -o "$OUT" --resource-path="$GUID:$GUID/_site"
echo "Wrote $OUT"
rm -rf "$WORK"
