#!/usr/bin/env bash
# Run an event notebook for one post, and optionally publish it to the guidance site.
#
# Usage:
#   ./run_event.sh <post_folder | event_config.yaml>                  run the notebook
#   ./run_event.sh <post_folder | event_config.yaml> --publish        run, then publish
#   ./run_event.sh <post_folder | event_config.yaml> --publish-only   publish without running
#
# Run: picks <hazard>_event.ipynb from the config's `hazard:` key. If the post folder has
# no copy of that notebook yet, copies it in from this framework folder first, then runs it
# in place (outputs saved into the post's copy) under the climakitae conda env.
#
# The post folder's index.qmd (+ optional images/) is the SOURCE OF TRUTH for the post's
# text. It's started from <hazard>_index_template.qmd the first time it's needed.
#
# Publish, into the guidance-site folder named by `publish.guidance_post_dir`:
#   - MOVES figures/html/fig_* and figures/static/fig_* (an empty source folder means
#     nothing new was exported)
#   - COPIES index.qmd (with a "don't edit here" note) and images/, always overwriting
#     the old copy — edits made directly to the guidance copy are lost
set -euo pipefail

FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USAGE="Usage: run_event.sh <post_folder | event_config.yaml> [--publish | --publish-only]"
TARGET="${1:?$USAGE}"
shift
CONDA_ENV="${CONDA_ENV:-climakitae}"

DO_RUN=1 DO_PUBLISH=0
for arg in "$@"; do
  case "$arg" in
    --publish) DO_PUBLISH=1 ;;
    --publish-only) DO_PUBLISH=1; DO_RUN=0 ;;
    *) echo "Unknown option: $arg" >&2; echo "$USAGE" >&2; exit 1 ;;
  esac
done

if [[ -d "$TARGET" ]]; then
  CONFIG="$TARGET/event_config.yaml"
else
  CONFIG="$TARGET"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

POST_DIR="$(cd "$(dirname "$CONFIG")" && pwd)"
CONFIG="$POST_DIR/$(basename "$CONFIG")"

HAZARD="$(sed -n 's/^hazard:[[:space:]]*\([a-z]*\).*/\1/p' "$CONFIG")"
if [[ -z "$HAZARD" ]]; then
  echo "No 'hazard:' key found in $CONFIG" >&2
  exit 1
fi
TEMPLATE_QMD="$FRAMEWORK_DIR/${HAZARD}_index_template.qmd"

ensure_source_qmd() {
  if [[ ! -f "$POST_DIR/index.qmd" ]]; then
    cp "$TEMPLATE_QMD" "$POST_DIR/index.qmd"
    echo "Started $POST_DIR/index.qmd from $(basename "$TEMPLATE_QMD") — edit the post text there"
  fi
}

run_notebook() {
  local notebook="$POST_DIR/${HAZARD}_event.ipynb"
  if [[ ! -f "$notebook" ]]; then
    if [[ ! -f "$FRAMEWORK_DIR/${HAZARD}_event.ipynb" ]]; then
      echo "No notebook for hazard '$HAZARD' in $FRAMEWORK_DIR" >&2
      exit 1
    fi
    cp "$FRAMEWORK_DIR/${HAZARD}_event.ipynb" "$notebook"
    echo "Copied ${HAZARD}_event.ipynb into $POST_DIR"
  fi

  if [[ ! -f "$FRAMEWORK_DIR/crai_cache/adaptive_capacity.csv" ]]; then
    echo "Warning: $FRAMEWORK_DIR/crai_cache/ is empty — run crai_base.ipynb first or Part 5 will fail." >&2
  fi

  export EVENT_CONFIG="$CONFIG"
  export EVENT_FRAMEWORK_DIR="$FRAMEWORK_DIR"
  export PYTHONPATH="$FRAMEWORK_DIR/helpers${PYTHONPATH:+:$PYTHONPATH}"

  echo "Running $notebook with $CONFIG (conda env: $CONDA_ENV)"
  (cd "$POST_DIR" && conda run -n "$CONDA_ENV" jupyter execute --inplace "$notebook")
}

publish() {
  # publish.guidance_post_dir, resolved relative to the post folder
  local dest
  dest="$(conda run -n "$CONDA_ENV" python -c '
import os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
p = (cfg.get("publish") or {}).get("guidance_post_dir")
print(os.path.normpath(os.path.join(os.path.dirname(sys.argv[1]), p)) if p else "")
' "$CONFIG")"
  if [[ -z "$dest" ]]; then
    echo "No publish.guidance_post_dir in $CONFIG — nothing to publish to." >&2
    exit 1
  fi
  mkdir -p "$dest"
  local dest_qmd="$dest/index.qmd"

  # --- figures: move -----------------------------------------------------------
  local moved=0 sub f
  for sub in html static; do
    mkdir -p "$dest/figures/$sub"
    for f in "$POST_DIR/figures/$sub"/fig_*; do
      [[ -e "$f" ]] || continue
      mv -f "$f" "$dest/figures/$sub/"
      moved=$((moved + 1))
    done
  done
  echo "Moved $moved figure file(s) to $dest/figures/"

  # --- index.qmd + images/: copy (source of truth stays in the post folder) -----
  ensure_source_qmd
  awk -v src="$POST_DIR/index.qmd" '
    { print }
    /^---[[:space:]]*$/ && ++n == 2 {
      print ""
      print "<!-- COPY — do not edit here. Source of truth: " src
      print "     Edit that file, then run: run_event.sh <post> --publish-only -->"
    }
  ' "$POST_DIR/index.qmd" > "$dest_qmd"
  echo "Copied index.qmd to $dest_qmd"

  if [[ -d "$POST_DIR/images" ]]; then
    mkdir -p "$dest/images"
    cp -R "$POST_DIR/images/." "$dest/images/"
    echo "Copied images/ to $dest/images/"
  fi
}

if [[ "$DO_RUN" -eq 1 ]]; then
  ensure_source_qmd
  run_notebook
fi
if [[ "$DO_PUBLISH" -eq 1 ]]; then
  publish
fi
