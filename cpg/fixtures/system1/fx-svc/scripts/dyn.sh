#!/usr/bin/env bash
set -euo pipefail
TOOL="${1:?tool name}"
eval "mcp-call $TOOL"
