#!/usr/bin/env bash
# pin_workspace_embedding_model.sh — FR-1/FR-2: idempotently pin one or more
# workspaces' `embeddingModelOverride` to the current global default, preserving
# any existing agent/guard/responder override untouched
# (`scripts/embedding_migration.py`'s `pin` operation,
# docs/plans/embedding-migration.md §3.2/§5 step 1).
#
# Thin wrapper, mirrors scripts/seed_eval_corpus.sh's preflight (FalkorDB PING,
# venv check) with one deliberate divergence: a missing/invalid
# FALKORCHAT_OPENCODE_CONFIG is NOT fatal here (unlike seed_eval_corpus.sh) —
# `pin`'s own design (§3.2 Option B mitigation) is to degrade to a printed
# WARNING and no-op per workspace, so this script only warns and lets the
# Python module handle it.
#
# Usage:
#   ./scripts/pin_workspace_embedding_model.sh <workspaceId> [<workspaceId> ...]
#
# Idempotent — a workspace that already has an explicit embeddingModelOverride
# is left untouched (no-op, reported as such by embedding_migration.py).
#
# Env vars:
#   FALKORDB_HOST (127.0.0.1)  FALKORDB_PORT (6379)
#   FALKORCHAT_OPENCODE_CONFIG (default: $HOME/.config/opencode/opencode.json)
#                              — the shared, pristine OpenCode providers file.
#                              Needed for ModelGateway.from_env() to resolve the
#                              real configured embedding default; missing/invalid
#                              degrades to a WARNING + no-op per workspace rather
#                              than failing this script (see above).
#   FALKORCHAT_MODEL_CONFIG   (default: falkor-chat/config/models.json, set by
#                              config.py itself — no need to export it here)

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0 ;; esac

set -euo pipefail

FALKORDB_HOST="${FALKORDB_HOST:-127.0.0.1}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
FALKORCHAT_OPENCODE_CONFIG="${FALKORCHAT_OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <workspaceId> [<workspaceId> ...]" >&2
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
  echo "WARNING: FALKORCHAT_OPENCODE_CONFIG file not found at ${FALKORCHAT_OPENCODE_CONFIG} — pin will print its own WARNING and no-op for each workspace below until a valid config is available" >&2
}

export FALKORDB_HOST FALKORDB_PORT FALKORCHAT_OPENCODE_CONFIG
exec "$PY" "$REPO_DIR/scripts/embedding_migration.py" pin "$@"
