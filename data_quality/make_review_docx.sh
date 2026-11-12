#!/usr/bin/env bash
# Build a single reviewable .docx from the rendered Quarto site.
# Usage:  ./make_review_docx.sh  [path/to/data_quality_guidance]  [output.docx]
# Prereqs: quarto, pandoc, python3 (with beautifulsoup4). mmdc (mermaid-cli) is
#          optional — if absent/failing, a bundled provenance.png is used instead.
set -euo pipefail

GUID="${1:-data_quality_guidance}"
OUT="${2:-Data_Quality_Review.docx}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"

# 1. Render the Quarto site if needed (produces _site/ with all tables/figures)
if [ ! -d "$GUID/_site" ]; then
  echo "Rendering Quarto site..."
  ( cd "$GUID" && quarto render )
fi

# 2. Provenance Mermaid diagram -> PNG (it is client-side on the web).
python3 -I - "$GUID" "$WORK" <<'PY'
import sys, re, os
guid, work = sys.argv[1], sys.argv[2]
qmd = os.path.join(guid, "method", "provenance.qmd")
m = re.search(r"```\{mermaid\}(.*?)```", open(qmd, encoding="utf-8").read(), re.DOTALL)
open(os.path.join(work, "prov.mmd"), "w").write(m.group(1).strip() if m else "")
PY

if command -v mmdc >/dev/null 2>&1 && [ -s "$WORK/prov.mmd" ]; then
  # mmdc drives headless Chromium; on most Linux boxes it needs --no-sandbox,
  # otherwise it dies with "No usable sandbox!". Give it a puppeteer config.
  printf '{"args":["--no-sandbox","--disable-setuid-sandbox"]}' > "$WORK/pptr.json"
  echo "Rendering provenance diagram with mmdc..."
  if ! mmdc -i "$WORK/prov.mmd" -o "$WORK/prov.png" -b white -p "$WORK/pptr.json"; then
    echo "  mmdc failed (see error above); will try bundled provenance.png"
    rm -f "$WORK/prov.png"
  fi
fi

# Fallback: use a pre-rendered provenance.png if mmdc didn't produce one.
# Looks next to the script, in the current dir, and in the project folder.
if [ ! -s "$WORK/prov.png" ]; then
  for CAND in "$HERE/provenance.png" "./provenance.png" "$GUID/provenance.png"; do
    if [ -f "$CAND" ]; then
      echo "Using bundled provenance.png: $CAND"
      cp "$CAND" "$WORK/prov.png"; break
    fi
  done
fi
[ -s "$WORK/prov.png" ] || echo "WARNING: provenance flowchart will be a placeholder (no mmdc output, no provenance.png found)."

# 3. Consolidate rendered HTML (figure iframes -> PNG, mermaid -> PNG).
python3 -I "$HERE/build_docx.py" "$GUID/_site" "$GUID" "$WORK/prov.png" "$WORK/combined.html"

# 4. One pandoc call -> docx.
pandoc "$WORK/combined.html" -o "$OUT" --resource-path="$GUID:$GUID/_site"
echo "Wrote $OUT"
rm -rf "$WORK"
