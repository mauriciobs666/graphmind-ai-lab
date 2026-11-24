#!/usr/bin/env bash
set -euo pipefail

# run.sh — run the component venv's Python with `src/` on the import path.
#
# Usage: llm-gateway/run.sh -m llmgateway.<module> [args]   (modules land per docs/plans/llm-gateway.md §4)
#        llm-gateway/run.sh -c 'import llmgateway; print(llmgateway.__version__)'
#
# This is the development entry; the key holder and the service run from the root-owned
# code root, never from here. Paths are resolved from this script's own location.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "llm-gateway/run.sh: no venv at $HERE/.venv — run llm-gateway/setup.sh first." >&2
  exit 1
fi

PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" "$@"
