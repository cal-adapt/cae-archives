#!/usr/bin/env bash
# Create a relevant-hazard post folder.
#
# Usage (from guidance-website/blog/):
#   ../relevant-hazard-template/run_event.sh <post_folder> --new heat|precip
#
# A post folder is ready to copy as-is into cal-adapt-guidance/blog/posts/:
#
#   <post>/
#   ├── index.qmd           post text (started from <hazard>_index_template.qmd)
#   ├── images/             photos for the post
#   ├── figures/            written by the notebook — the paths index.qmd embeds
#   └── _analysis/          Quarto ignores folders starting with "_"
#       ├── event_config.yaml
#       ├── <hazard>_event.ipynb
#       ├── <zip-code shapefile>.zip
#       └── crai_layers/    written by the notebook
#
# Then open _analysis/<hazard>_event.ipynb in VS Code (climakitae kernel) and Run All.
set -euo pipefail

FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USAGE="Usage: run_event.sh <post_folder> --new heat|precip"
POST_ARG="${1:?$USAGE}"
shift

new_post() {
  local hazard="$1"
  local example
  example="$(ls "$FRAMEWORK_DIR/configs/${hazard}_"*.yaml 2>/dev/null | head -1 || true)"
  if [[ ! -f "$FRAMEWORK_DIR/${hazard}_event.ipynb" || -z "$example" ]]; then
    echo "Unknown hazard '$hazard' — expected heat or precip" >&2
    exit 1
  fi
  if [[ -e "$POST_ARG/index.qmd" || -e "$POST_ARG/_analysis" ]]; then
    echo "$POST_ARG already has index.qmd or _analysis/ — not overwriting" >&2
    exit 1
  fi

  mkdir -p "$POST_ARG/_analysis" "$POST_ARG/images"
  cp "$FRAMEWORK_DIR/${hazard}_index_template.qmd" "$POST_ARG/index.qmd"
  cp "$example" "$POST_ARG/_analysis/event_config.yaml"
  cp "$FRAMEWORK_DIR/${hazard}_event.ipynb" "$POST_ARG/_analysis/"

  echo "Created $POST_ARG/ for a $hazard post:"
  echo "  index.qmd                       write the post here"
  echo "  images/                         put photos here"
  echo "  _analysis/event_config.yaml     edit for your event (copied from $(basename "$example"))"
  echo "  _analysis/${hazard}_event.ipynb     run after editing the config"
  echo "Next: add a zip-code shapefile to _analysis/ and set location.zip_shapefile to its name."
}

case "${1:-}" in
  --new)
    new_post "${2:?$USAGE}" ;;
  *)
    echo "$USAGE" >&2; exit 1 ;;
esac
