#!/usr/bin/env python
"""
Two fixes to src/build_datasets.py, applied in place.

1. `cached_era5` stamps time_start / time_end / freq onto every ERA5 cache it
   writes, and refuses to reuse a cache whose stamp does not match the window
   being asked for. Keying only on (preset, cadence) made a changed T0_DAY
   invisible: the stale object was reused, xr.align(join="inner") silently
   intersected the source down to the cache's span, and the run completed with
   no error and the wrong answer.

   Caches written before this change carry no attributes and are rebuilt once.

2. `multimodel_loca2` declares both datasets it writes. It also produces
   `multimodel_loca2_bygcm`, which wind_speed_review section 1 reads, but only
   declared `multimodel_loca2_monthly` -- so a run with `_bygcm` missing and
   `_monthly` present reported SKIP and never rebuilt it.

Idempotent and edit-safe: two exact string replacements, not a context diff.

    python apply_cache_stamp.py
    python apply_cache_stamp.py --check
    python apply_cache_stamp.py --path <file>
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import shutil
import sys
import zlib

PAYLOAD = (
    "eNrtVt9v2zYQ/lcOfokMyErcoS8GPKxLgq0YkABpgDzYhkFLZ4mtRKokFScL8r/3jqT8o4mDonsY"
    "BsxJHNM83n333XcnzmaDAteQi7zCYolGvE+s0wZTsGhtCq1Bi24C1pmUrApUOfYro9sU3Bn9jVNY"
    "G/w6nMwV0Gs+4J/Lmw/vwWDR5ViA0+AqBKs7k2PvKKVtUdBR3YRdjgxyHaIql81VcHhbSQv0y0b4"
    "0KKy8h6hEvUa9BrwHs0jmE7BCIzuyqp+hPEv8MfvvFlRRFoXwglo0QR/rdEEy/EXPZYMzokDqUqQ"
    "DhoUKkRrutrJUaMLrAmeKJE42VQyrwh6ZzG489hFg54SEOGkJV81jhhWfzDX1oFWWxp89lJZx/8J"
    "qwjuLOZaFeD5a4W12T6r4bPiaFPgei1FWS55nYRabcs0DKZEp+c1wwdpnU3YdAiCAijt4pYmBjdG"
    "OowFDBxJ5ZL1fMALTpa5CToJ0CiuwVI4qdUEntjt83ww3Dkw6DqjYgRONYSezQcMez5Y9NX1WZg+"
    "nUYrRxVcbmyLWCRRZbLBJdFo3NQLjpeoimmvvKmXX3C3ka5aUoUJnbW9V5Plumk7h0m0CrB80h5X"
    "Cg8muyCVEIXJU49xcujteRhPx9wONgcp/N9L/4FeijT8eQnnH87p/eby/Prm4hN8vP0E13dXcPfx"
    "6uL6LoO/8JFT0IpS1gpetBc04gvaPlBeCVVScW7PTm/HBOReWrmqqb6hGKJG0KvPSEQR9z7fwktO"
    "1LJUyWct1XQ+kEoRrX0Tfe0kOgpOfYiGEnF2v+6F3qheCl5yJxZspQ3Zgm2FSn2P8y7RFjFSC9To"
    "CDQLl/qfesNoszXcGE0JU7k2aDISCpIiT934lPUIwjkjVx2fZu331RKO8WkQYWAQAw2JnWrkvQpK"
    "ddXJutjyzspgANR4DhWscM06dSxK2si/UC6GFEjYdgGpgXyej7SpeGqtolgo30BCIxzJyEc0GGOy"
    "PFiLtx5j4C7XxnANdOeIDKoOeQtCjfBs5/30tSJjbQobZbqRilhnUec8Mf/hYN4I5ciUZoRJ3Nlw"
    "NhmfLVLwq/H+yk+DYc/fz47zOLmnL+bxzqTUHk+wzJh+m5U0DeeD3fglcaZwzIImMu/vPB7EPrDn"
    "pMh2Lzol5gFMPTGTQyc/8zCCviaHL3KSPFGk2dniGUa/gv88Xjyn4dO7xfPw4DG2N+5DwP0H2Evw"
    "yRUNpRR278O3UokpRMCkE5qtvj290IjxpuVMj2YyGvUNxmbUSvDE7G1z8wtK7iAjrC3+OKggdXiN"
    "MmrR49BOejpPUlDI/l5Fxj5Owoptj/j7LlOfzr94eaDxQXs/dGHYHojy71oucbKHKdKyByxyE9FF"
    "btLXePGl4+Ey/W7GTPtrwZFTioJja6dc9gPEmZV/o52FjiaND4/elyint25DBH82+M0/rqnf/bPf"
    "P/qXtc7FuzndlmYvv+4LSIFTvk+9dX6b2Zt+Unhld/VY5o2PsVh8A5NEQlA="
)

SENTINEL = "THE CACHE RECORDS ITS OWN WINDOW"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="src/build_datasets.py")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()

    pairs = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)).decode())

    try:
        src = open(a.path).read()
    except OSError as e:
        print(f"cannot read {a.path}: {e}")
        return 1

    if SENTINEL in src:
        print(f"{a.path}: already patched, nothing to do")
        return 0

    missing = [i for i, (old, _) in enumerate(pairs, 1) if old not in src]
    if missing:
        print(f"{a.path}: cannot patch -- anchors {missing} not found.")
        for i in missing:
            print("\n--- anchor %d ---\n%s" % (i, pairs[i - 1][0]))
        return 1

    out = src
    for old, new in pairs:
        if out.count(old) != 1:
            print(f"anchor appears {out.count(old)} times, expected 1:"
                  f"\n{old[:80]}")
            return 1
        out = out.replace(old, new)

    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"result does not parse ({e}); refusing to write")
        return 1

    if a.check:
        print(f"{a.path}: both anchors found, result parses. "
              "Re-run without --check to write.")
        return 0

    if not a.no_backup:
        shutil.copy(a.path, a.path + ".bak2")
        print(f"backup: {a.path}.bak2")
    open(a.path, "w").write(out)
    print(f"patched {a.path}: ERA5 caches now record their window; "
          "multimodel_loca2 declares both outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
