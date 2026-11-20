#!/usr/bin/env bash
# reembed_unembedded_chunks.sh — K-102 one-off repair: list (default, dry run) or re-embed
# (--apply) a workspace's Chunks with no embedding. Never changes Document.status.
# Thin wrapper over scripts/reembed_unembedded_chunks.py; preflight like migrate_embeddings.sh.
#
# Usage: ./scripts/reembed_unembedded_chunks.sh <workspaceId> [--apply]
#
# Env: FALKORDB_HOST (127.0.0.1)  FALKORDB_PORT (6379)
#      FALKORCHAT_OPENCODE_CONFIG (default: resolved by scripts/shlib/opencode_config.sh —
#        opencode.local.json first — i.e. the same endpoint the running server embeds through)
usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help|"") usage; exit 0 ;; esac
set -euo pipefail
FALKORDB_HOST="${FALKORDB_HOST:-127.0.0.1}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=shlib/opencode_config.sh
. "$REPO_DIR/scripts/shlib/opencode_config.sh"
resolve_opencode_config
PY="$REPO_DIR/server/.venv/bin/python"
redis-cli -h "$FALKORDB_HOST" -p "$FALKORDB_PORT" PING 2>/dev/null | grep -q PONG || {
  echo "ERROR: FalkorDB not reachable at ${FALKORDB_HOST}:${FALKORDB_PORT}" >&2; exit 1; }
[ -x "$PY" ] || { echo "ERROR: venv missing at $REPO_DIR/server/.venv" >&2; exit 1; }
[ -f "$FALKORCHAT_OPENCODE_CONFIG" ] || {
  echo "ERROR: FALKORCHAT_OPENCODE_CONFIG file not found: ${FALKORCHAT_OPENCODE_CONFIG}" >&2; exit 1; }
export FALKORDB_HOST FALKORDB_PORT FALKORCHAT_OPENCODE_CONFIG
exec "$PY" "$REPO_DIR/scripts/reembed_unembedded_chunks.py" "$@"
