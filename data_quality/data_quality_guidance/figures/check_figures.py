#!/usr/bin/env python
"""Check every figure a page embeds exists and has a height rule.

    python figures/check_figures.py

Run before rendering. An iframe pointing at nothing renders as a blank frame --
silent, and the kind of thing that survives review unnoticed.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PNG, HTML = ROOT / "figures" / "png", ROOT / "figures" / "html"
CSS, JSON = ROOT / "figures" / "heights.css", ROOT / "figures" / "heights.json"

embedded = set()
for q in sorted(ROOT.rglob("*.qmd")):
    embedded |= set(re.findall(r'data-fig="([^"]+)"', q.read_text()))

styled = set(re.findall(r'data-fig="([^"]+)"', CSS.read_text())) if CSS.exists() else set()
heights = json.loads(JSON.read_text()) if JSON.exists() else {}

missing_html = sorted(s for s in embedded if not (HTML / f"{s}.html").exists())
missing_png = sorted(s for s in embedded if not (PNG / f"{s}.png").exists())
# A figure with no rule falls back to the generic 466px frame, which will
# either clip it or leave a band of white below.
no_rule = sorted(embedded - styled)
stale = sorted(styled - embedded)
# PNGs written by the notebook that no page shows -- usually an exploratory
# figure, or one whose section was cut.
orphan_png = sorted({f.stem for f in PNG.glob("*.png")} - embedded)
# A loose PNG beside the subfolders means something bypassed save_png().
loose = sorted(f.name for f in (ROOT / "figures").glob("*.png"))

print(f"embedded: {len(embedded)} figures | {len(styled)} height rules")
for label, items in (
        ("MISSING html (renders as a blank frame)", missing_html),
        ("MISSING png (portable copy absent)", missing_png),
        ("NO HEIGHT RULE (falls back to 466px)", no_rule),
        ("height rule with no page using it", stale),
        ("png on disk that no page embeds", orphan_png),
        ("LOOSE png in figures/ -- should be in figures/png/", loose)):
    if items:
        print(f"\n{label} ({len(items)}):")
        for i in items:
            print(f"   {i}")

if missing_html or loose:
    sys.exit(2)
if missing_png or no_rule:
    sys.exit(1)
print("\nall good")
