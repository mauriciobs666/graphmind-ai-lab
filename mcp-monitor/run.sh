#!/usr/bin/env bash
set -euo pipefail

# run.sh — launch mcp-monitor from the host venv.
#
# Usage: mcp-monitor/run.sh --config PATH [--log-level LEVEL]
#
# Everything is resolved from this script's own location, so the working
# directory the caller happens to be in does not matter.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "mcp-monitor/run.sh: no venv at $HERE/.venv — run mcp-monitor/setup.sh first." >&2
  exit 1
fi

exec "$PY" -m mcp_monitor "$@"
