#!/usr/bin/env bash
set -euo pipefail

# One-shot script: ensure FalkorDB is up, bootstrap schema, start the M1 server.
#
# Override defaults with env vars:
#   FALKORCHAT_WS_ID   (default: acme)
#   FALKORCHAT_USER_ID (default: u1)
#   FALKORDB_PORT      (default: 6379)
#   FALKORDB_HOST      (default: 127.0.0.1)
#   EMBEDDING_DIM      (default: 1536)
#   UVICORN_ARGS       (default: --reload)
#
# Example (custom workspace + headless FalkorDB):
#   FALKORCHAT_WS_ID=myws ./scripts/start_server.sh

usage() {
  cat <<EOF
Usage: start_server.sh [-h|--help]

Starts everything in one terminal:
  1. Starts FalkorDB (detached) if not already running
  2. Creates/updates Python venv in server/.venv
  3. Bootstraps schema for the configured workspace
  4. Starts uvicorn (reload by default)

Stop with Ctrl+C; FalkorDB keeps running in the background.
Stop FalkorDB: docker stop falkordb-dev

Env overrides: FALKORCHAT_WS_ID, FALKORCHAT_USER_ID,
               FALKORDB_HOST, FALKORDB_PORT, EMBEDDING_DIM, UVICORN_ARGS
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) echo "start_server.sh: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ── defaults ──────────────────────────────────────────────────────────────────
FALKORCHAT_WS_ID="${FALKORCHAT_WS_ID:-acme}"
FALKORCHAT_USER_ID="${FALKORCHAT_USER_ID:-u1}"
FALKORDB_HOST="${FALKORDB_HOST:-127.0.0.1}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
EMBEDDING_DIM="${EMBEDDING_DIM:-1536}"
UVICORN_ARGS="${UVICORN_ARGS:---reload}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$REPO_DIR/server"
VENV_DIR="$SERVER_DIR/.venv"

# ── 1. FalkorDB ───────────────────────────────────────────────────────────────
if docker inspect falkordb-dev --format '{{.State.Status}}' 2>/dev/null | grep -q running; then
  echo "[1/4] FalkorDB already running — ok"
else
  echo "[1/4] Starting FalkorDB (detached)..."
  "$REPO_DIR/scripts/start_falkordb.sh" -d
  echo "      Waiting for FalkorDB to be ready..."
  for i in $(seq 1 30); do
    if redis-cli -h "$FALKORDB_HOST" -p "$FALKORDB_PORT" ping 2>/dev/null | grep -q PONG; then
      echo "      Ready."
      break
    fi
    if [ "$i" -eq 30 ]; then
      echo "ERROR: FalkorDB did not respond after 30s" >&2
      exit 1
    fi
    sleep 1
  done
fi

# ── 2. venv + deps ────────────────────────────────────────────────────────────
echo "[2/4] Setting up Python venv..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install -q -e "$SERVER_DIR[dev]"

# ── 3. Bootstrap schema ───────────────────────────────────────────────────────
echo "[3/4] Bootstrapping schema for workspace '$FALKORCHAT_WS_ID'..."
EMBEDDING_DIM="$EMBEDDING_DIM" "$REPO_DIR/scripts/bootstrap_schema.sh" "$FALKORCHAT_WS_ID"

# ── 4. Start uvicorn ──────────────────────────────────────────────────────────
echo "[4/4] Starting uvicorn on http://localhost:8000..."
echo "      Workspace: $FALKORCHAT_WS_ID  |  User: $FALKORCHAT_USER_ID"
echo "      MCP endpoint: http://localhost:8000/mcp"
echo "      Web UI:       http://localhost:8000/"
echo "      Stop with Ctrl+C (FalkorDB keeps running in background)"
echo ""
export FALKORCHAT_WS_ID FALKORCHAT_USER_ID FALKORDB_HOST FALKORDB_PORT
exec "$VENV_DIR/bin/uvicorn" falkorchat.app:app $UVICORN_ARGS
