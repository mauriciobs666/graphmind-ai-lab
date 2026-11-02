#!/usr/bin/env bash
set -euo pipefail
export MCP_URL=http://localhost:8200/mcp
mcp-call reindex
