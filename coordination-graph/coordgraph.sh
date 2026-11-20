#!/usr/bin/env bash
set -euo pipefail

# coordgraph.sh — entry point for the coordination-graph CLI.
#
# Named `.sh` because the Python package occupies coordination-graph/coordgraph/.
# Resolved from this script's own location, so the working directory does not
# matter. Runs from the component venv (setup.sh).
#
# Usage: coordination-graph/coordgraph.sh <subcommand> [args]

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "coordgraph.sh: no venv at $HERE/.venv — run coordination-graph/setup.sh first." >&2
  exit 1
fi

exec "$PY" -m coordgraph "$@"
