#!/usr/bin/env bash
# backfill_document_lsh_bands.sh — K-091 one-off: list (default, dry run) or stamp (--apply)
# Document.lshBand0..7 on documents ingested before the ingest write stored them.
# Idempotent — a second run lists 0. No model calls. Thin wrapper over
# scripts/backfill_document_lsh_bands.py.
#
# Rollout order: restart every server serving the workspace on K-091 code FIRST,
# then --apply; a document ingested by an old server in between stays band-less
# until a re-run, which is safe. Exit 1 if any workspace failed (e.g. mistyped id).
#
# Usage: ./scripts/backfill_document_lsh_bands.sh <workspaceId> [<workspaceId> ...] [--apply]
#
# Env: FALKORDB_HOST (127.0.0.1)  FALKORDB_PORT (6379)
usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help|"") usage; exit 0 ;; esac
set -euo pipefail
FALKORDB_HOST="${FALKORDB_HOST:-127.0.0.1}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO_DIR/server/.venv/bin/python"
redis-cli -h "$FALKORDB_HOST" -p "$FALKORDB_PORT" PING 2>/dev/null | grep -q PONG || {
  echo "ERROR: FalkorDB not reachable at ${FALKORDB_HOST}:${FALKORDB_PORT}" >&2; exit 1; }
[ -x "$PY" ] || { echo "ERROR: venv missing at $REPO_DIR/server/.venv" >&2; exit 1; }
export FALKORDB_HOST FALKORDB_PORT
exec "$PY" "$REPO_DIR/scripts/backfill_document_lsh_bands.py" "$@"
