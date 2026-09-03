#!/usr/bin/env bash
set -euo pipefail

# run.sh — launch the model-bench CLI from the component venv.
#
# Usage: model-bench/run.sh <command> [options]   — e.g. `run.sh compare --pack <id>`
#
# Everything is resolved from this script's own location, so the working
# directory the caller happens to be in does not matter.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "model-bench/run.sh: no venv at $HERE/.venv — run model-bench/setup.sh first." >&2
  exit 1
fi

exec "$PY" -m modelbench "$@"
