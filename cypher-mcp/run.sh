#!/usr/bin/env bash
set -euo pipefail

# run.sh — launch the `cypher` MCP server over stdio, from the host venv.
#
# This is the Docker-less variant. The repo-root .mcp.json names docker-run.sh
# (the container path); this script stays as the test loop, the fallback when the
# image or the Docker daemon is broken, and what ports to a Docker-less host —
# see the container section of cypher-mcp/README.md. Everything is resolved from this
# script's own location, so the working directory the harness happens to start in
# does not matter.
#
# Usage: cypher-mcp/run.sh          (speaks MCP on stdin/stdout — not interactive)
#
# Env (all optional, defaults in server.py):
#   FALKORDB_HOST / FALKORDB_PORT   where FalkorDB listens (127.0.0.1 / 6379)
#   CYPHER_MCP_MAX_ROWS / _MAX_CELL / _MAX_CHARS   display-only truncation caps
#   CYPHER_MCP_TIMEOUT_MS              per-query server-side timeout
#
# Nothing here may write to stdout: the transport owns it. Diagnostics go to
# stderr, which the harness surfaces in its MCP log.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "cypher-mcp/run.sh: no venv at $HERE/.venv — run cypher-mcp/setup.sh first." >&2
  exit 1
fi

exec "$PY" "$HERE/server.py"
