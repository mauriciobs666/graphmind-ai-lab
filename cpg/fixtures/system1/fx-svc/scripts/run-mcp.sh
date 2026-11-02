#!/usr/bin/env bash
set -euo pipefail
exec python -m fx_svc.mcp.server "$@"
