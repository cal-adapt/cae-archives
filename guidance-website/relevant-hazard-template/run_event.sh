#!/usr/bin/env bash
# Create or run a relevant-hazard post folder.
#
# Usage (from guidance-website/blog/):
#   ../relevant-hazard-template/run_event.sh <post_folder> --new heat|precip   create the post folder
#   ../relevant-hazard-template/run_event.sh <post_folder>                     run its notebook
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
# Running executes _analysis/<hazard>_event.ipynb in place (outputs saved into it) under
# the climakitae conda env. You can also just open that notebook in VS Code and Run All.
set -euo pipefail

FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USAGE="Usage: run_event.sh <post_folder> [--new heat|precip]"
POST_ARG="${1:?$USAGE}"
shift
CONDA_ENV="${CONDA_ENV:-climakitae}"

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

run_notebook() {
  local analysis="$POST_ARG/_analysis"
  local config="$analysis/event_config.yaml"
  if [[ ! -f "$config" ]]; then
    echo "No $config — create the post first with: run_event.sh $POST_ARG --new heat|precip" >&2
    exit 1
  fi
  analysis="$(cd "$analysis" && pwd)"
  config="$analysis/event_config.yaml"

  local hazard
  hazard="$(sed -n 's/^hazard:[[:space:]]*\([a-z]*\).*/\1/p' "$config")"
  if [[ -z "$hazard" ]]; then
    echo "No 'hazard:' key found in $config" >&2
    exit 1
  fi
  local notebook="$analysis/${hazard}_event.ipynb"
  if [[ ! -f "$notebook" ]]; then
    cp "$FRAMEWORK_DIR/${hazard}_event.ipynb" "$notebook"
    echo "Copied ${hazard}_event.ipynb into $analysis"
  fi
  if [[ ! -f "$FRAMEWORK_DIR/crai_cache/adaptive_capacity.csv" ]]; then
    echo "Warning: $FRAMEWORK_DIR/crai_cache/ is empty — run crai_base.ipynb first or Part 5 will fail." >&2
  fi

  export EVENT_CONFIG="$config"
  export EVENT_FRAMEWORK_DIR="$FRAMEWORK_DIR"
  export PYTHONPATH="$FRAMEWORK_DIR/helpers${PYTHONPATH:+:$PYTHONPATH}"

  echo "Running $notebook (conda env: $CONDA_ENV)"
  (cd "$analysis" && conda run -n "$CONDA_ENV" jupyter execute --inplace "$notebook")
}

case "${1:-}" in
  --new)
    new_post "${2:?$USAGE}" ;;
  "")
    run_notebook ;;
  *)
    echo "Unknown option: $1" >&2; echo "$USAGE" >&2; exit 1 ;;
esac
