#!/usr/bin/env bash
# migrate_embeddings.sh — FR-3/FR-4/FR-5/FR-8/FR-10: re-embed one workspace's
# Message/Chunk rows to a new embedding model, rebuild both vector indexes at
# the new dimension, and move the workspace's embeddingModelOverride
# (`scripts/embedding_migration.py`'s `migrate` operation,
# docs/plans/embedding-migration.md §3.3/§3.4/§5 step 4).
#
# Thin wrapper, same preflight shape as pin_workspace_embedding_model.sh
# (FalkorDB PING, venv check, FALKORCHAT_OPENCODE_CONFIG file check) — unlike
# `pin`, a missing/invalid FALKORCHAT_OPENCODE_CONFIG IS fatal here: `migrate`
# has no best-effort degrade path (it must resolve a real embedder to do
# anything at all).
#
# ⚠️ §3.4 step 0: this does NOT stop traffic for you. Before running this
# script, stop the falkorchat.app process whose FALKORCHAT_WS_ID equals the
# target workspace (§2.6) — the only process that could originate a live write
# against it in this deployment. Pass --i-have-stopped-traffic once it is
# down, or answer the interactive prompt this script's Python module prints
# when the flag is omitted and a terminal is attached.
#
# ⚠️ After this script reports success: RESTART that same server process
# before resuming traffic (§2.7) — it clears EmbeddingWorker's process-lifetime
# index-dimension cache, which otherwise still compares new writes against the
# stale pre-migration dimension.
#
# Usage:
#   ./scripts/migrate_embeddings.sh <workspaceId> <targetModelRef> \
#       [--batch-size N] [--i-have-stopped-traffic]
#
# Idempotent-resume (FR-10) — re-running after any interruption (including a
# crash between the vector-index DROP and CREATE) only processes/repairs what
# is left; a fully-migrated re-run reports zero rows processed.
#
# Env vars:
#   FALKORDB_HOST (127.0.0.1)  FALKORDB_PORT (6379)
#   FALKORCHAT_OPENCODE_CONFIG (default: $HOME/.config/opencode/opencode.json)
#                              — the shared, pristine OpenCode providers file.
#                              Required (fatal if missing/invalid).
#   FALKORCHAT_MODEL_CONFIG   (default: falkor-chat/config/models.json, set by
#                              config.py itself — no need to export it here).
#                              Must declare models.<targetModelRef>.dim before
#                              running this script (§3.3's precondition).

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0 ;; esac

set -euo pipefail

FALKORDB_HOST="${FALKORDB_HOST:-127.0.0.1}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
FALKORCHAT_OPENCODE_CONFIG="${FALKORCHAT_OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <workspaceId> <targetModelRef> [--batch-size N] [--i-have-stopped-traffic]" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$REPO_DIR/server"
VENV="$SERVER_DIR/.venv"
PY="$VENV/bin/python"

echo "Checking FalkorDB at ${FALKORDB_HOST}:${FALKORDB_PORT}..."
redis-cli -h "$FALKORDB_HOST" -p "$FALKORDB_PORT" PING 2>/dev/null | grep -q PONG || {
  echo "ERROR: FalkorDB not reachable at ${FALKORDB_HOST}:${FALKORDB_PORT} — start it with ./scripts/start_falkordb.sh -d" >&2
  exit 1
}

[ -x "$PY" ] || {
  echo "ERROR: venv missing at $VENV — run: cd server && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
}

[ -f "$FALKORCHAT_OPENCODE_CONFIG" ] || {
  echo "ERROR: FALKORCHAT_OPENCODE_CONFIG file not found at ${FALKORCHAT_OPENCODE_CONFIG} — migrate must resolve a real embedder, unlike pin's best-effort degrade" >&2
  exit 1
}

export FALKORDB_HOST FALKORDB_PORT FALKORCHAT_OPENCODE_CONFIG
exec "$PY" "$REPO_DIR/scripts/embedding_migration.py" migrate "$@"
