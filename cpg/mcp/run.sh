#!/usr/bin/env bash
set -euo pipefail

# run.sh — launch the `cpg` MCP server over stdio.
#
# This is the only path that appears in a harness config (.mcp.json): everything
# else is resolved from this script's own location, so the working directory the
# harness happens to start in does not matter.
#
# Usage: cpg/mcp/run.sh          (speaks MCP on stdin/stdout — not interactive)
#
# Env (all optional, defaults in server.py):
#   FALKORDB_HOST / FALKORDB_PORT   where FalkorDB listens (127.0.0.1 / 6379)
#   CPG_MCP_MAX_ROWS / _MAX_CELL / _MAX_CHARS   display-only truncation caps
#   CPG_MCP_TIMEOUT_MS              per-query server-side timeout
#
# Nothing here may write to stdout: the transport owns it. Diagnostics go to
# stderr, which the harness surfaces in its MCP log.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "cpg/mcp/run.sh: no venv at $HERE/.venv — run cpg/mcp/setup.sh first." >&2
  exit 1
fi

exec "$PY" "$HERE/server.py"
