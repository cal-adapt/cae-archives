import sys, os, re
from bs4 import BeautifulSoup

SITE = sys.argv[1]            # .../_site
PROJ = sys.argv[2]            # project root (for figures/png)
PROV_PNG = sys.argv[3]        # rendered provenance png
OUT_HTML = sys.argv[4]

PAGES = [
    ("index.html", "Overview"),
    ("method/scope-and-definitions.html", "Method — Scope & Definitions"),
    ("method/provenance.html", "Method — Provenance"),
    ("method/extending.html", "Method — Extending to a new variable"),
    ("variables/wind-speed.html", "Variable — Wind speed"),
#    ("variables/relative-humidity.html", "Variable — Relative humidity"),
    ("guidance/recommendation.html", "Guidance — Recommendation"),
    ("guidance/variable-cards.html", "Guidance — Variable Cards"),
    ("guidance/catalog-audit.html", "Guidance — Catalog Audit"),
    ("references.html", "References"),
]

PNG_DIR = os.path.join(PROJ, "figures", "png")
parts = []

for rel, _label in PAGES:
    path = os.path.join(SITE, rel)
    if not os.path.exists(path):
        print("MISSING", rel); continue
    soup = BeautifulSoup(open(path, encoding="utf-8"), "html.parser")
    main = soup.select_one("#quarto-document-content") or soup.select_one("main")
    if main is None:
        print("no main in", rel); continue

    # strip page-level TOC, scripts, styles, nav artifacts
    for sel in ["nav#TOC", "#quarto-margin-sidebar", "script", "style",
                ".quarto-title-meta", "div.toc-actions", "#quarto-sidebar"]:
        for el in main.select(sel):
            el.decompose()

    # swap figure iframes -> PNG
    for d in main.select("div.fig-frame[data-fig]"):
        name = d["data-fig"]
        png = os.path.join(PNG_DIR, name + ".png")
        if os.path.exists(png):
            img = soup.new_tag("img", src=os.path.abspath(png))
            img["style"] = "width:100%;"
            d.replace_with(img)
        else:
            box = soup.new_tag("p")
            em = soup.new_tag("em")
            em.string = f"[Figure \u201c{name}\u201d not in this export \u2014 interactive-only; see the live site]"
            box.append(em)
            d.replace_with(box)

    # swap mermaid -> provenance png
    for pre in main.select("pre.mermaid, pre.mermaid-js, div.mermaid"):
        if os.path.exists(PROV_PNG):
            img = soup.new_tag("img", src=os.path.abspath(PROV_PNG))
            img["style"] = "width:100%;"
            pre.replace_with(img)
        else:
            note = soup.new_tag("p")
            em = soup.new_tag("em")
            em.string = ("[Provenance flowchart not rendered \u2014 mermaid-cli (mmdc) was not "
                         "found and no fallback provenance.png was supplied. Install mmdc or place "
                         "provenance.png next to the script.]")
            note.append(em); pre.replace_with(note)

    # drop any leftover bare iframes
    for ifr in main.select("iframe"):
        ifr.decompose()

    parts.append('<div style="page-break-before:always"></div>')
    parts.append(str(main))

html = ('<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
        + "\n".join(parts) + "</body></html>")
open(OUT_HTML, "w", encoding="utf-8").write(html)
print("wrote", OUT_HTML, "pages:", len(parts)//2)
