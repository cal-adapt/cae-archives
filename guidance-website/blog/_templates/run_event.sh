#!/usr/bin/env bash
# Run the extreme-heat-event template notebook against a given event config.
#
# Usage:
#   ./run_event.sh <event_config.yaml> [notebook.ipynb]
#
# Run this from the post folder that holds your event_config.yaml, the notebook
# copy, and any assets the config points to (zip shapefile, CRAI CSVs) — figures
# are written to figures/static and figures/html relative to that folder.
set -euo pipefail

CONFIG="${1:?Usage: run_event.sh <event_config.yaml> [notebook.ipynb]}"
NOTEBOOK="${2:-extreme_heat_event_template.ipynb}"
CONDA_ENV="${CONDA_ENV:-climakitae}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$NOTEBOOK" ]]; then
  echo "Notebook not found: $NOTEBOOK" >&2
  exit 1
fi

export EXTREME_HEAT_CONFIG
EXTREME_HEAT_CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

echo "Running $NOTEBOOK with config $EXTREME_HEAT_CONFIG (conda env: $CONDA_ENV)"
conda run -n "$CONDA_ENV" jupyter execute --inplace "$NOTEBOOK"
