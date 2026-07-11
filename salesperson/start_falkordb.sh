#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper: the canonical FalkorDB start script lives in
# ../falkor-chat/scripts/start_falkordb.sh (both components share the one
# FalkorDB container on 6379/3000, so there is a single way to start it).
exec "$(dirname "$(readlink -f "$0")")/../falkor-chat/scripts/start_falkordb.sh" "$@"
