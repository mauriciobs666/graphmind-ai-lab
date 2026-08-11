#!/usr/bin/env bash
set -euo pipefail

# One-shot script: ensure FalkorDB is up, bootstrap schema, seed the demo agent,
# and start the server (REST + MCP + AI responder).
#
# Override defaults with env vars:
#   FALKORCHAT_WS_ID       (default: acme)
#   FALKORCHAT_USER_ID     (default: u1)
#   FALKORDB_PORT          (default: 6379)
#   FALKORDB_HOST          (default: 127.0.0.1)
#   EMBEDDING_DIM          (default: 1024)   — MUST match the workspace's vector
#                          index; ws:acme is bootstrapped at 1024 (Qwen3-Embedding).
#                          Exported to the app as FALKORCHAT_EMBEDDING_DIM so a
#                          wrong-dim embedding can't silently drop out of ANN.
#   FALKORCHAT_ENABLE_AGENT(default: 1)      — wire the live embedder + LLM + AI
#                          responder (@mention the agent to get a reply). Set 0 to
#                          serve the UI/REST without the AI loop.
#   FALKORCHAT_OPENCODE_CONFIG(default: $HOME/.config/opencode/opencode.json)
#                          — K-042: the pristine, unmodified shared OpenCode file
#                          (providers only). Required whenever ENABLE_AGENT/
#                          WORKFLOW_ENABLED is on; this script supplies the dev
#                          convenience default (the product itself has none — see
#                          config/opencode.example.json for the expected shape).
#   FALKORCHAT_MODEL_CONFIG(default: falkor-chat/config/models.json, set by
#                          config.py itself — no need to export it here) — K-042:
#                          falkor-chat's own overlay (per-kind defaults/settings).
#   FALKORCHAT_WORKFLOW_ENABLED(default: 1)  — wire the M3 LLM-native workflow executor
#                          + trigger (@mention starts the triage run; a plain reply
#                          resumes a waiting one). Seeds the triage def first (a def MUST
#                          be published or @mention-to-start is a silent no-op). Set 0 to
#                          keep the M2 direct-reply wiring only.
#   FALKORCHAT_AGENT_ID    (default: assistant)
#   FALKORCHAT_AGENT_NAME  (default: Assistant)
#   FALKORCHAT_TRIGGER_DEF_KEY     (default: triage) — which def an @mention resolves/starts
#   FALKORCHAT_TRIGGER_DEF_VERSION (default: v1)       (config.TRIGGER_DEF_KEY/_VERSION).
#                          Override for a demo/QA session (e.g. `access-request`/`v1`) to
#                          point the trigger at a different published def. K-037: this is
#                          fully decoupled from `seed_workflows.sh`'s own
#                          FALKORCHAT_TRIAGE_DEF_KEY/_VERSION — overriding this pair no
#                          longer affects what that script publishes.
#   UVICORN_ARGS           (default: --reload)
#
# Example (custom workspace + headless FalkorDB):
#   FALKORCHAT_WS_ID=myws ./scripts/start_server.sh

usage() {
  cat <<EOF
Usage: start_server.sh [-h|--help]

Starts everything in one terminal:
  1. Starts FalkorDB (detached) if not already running
  2. Creates/updates Python venv in server/.venv
  3. Bootstraps schema for the configured workspace (EMBEDDING_DIM)
  4. Seeds the AI agent + a demo channel/thread (idempotent)
  5. Seeds the M3 triage workflow def (idempotent)
  6. Starts uvicorn with the AI responder + workflow engine enabled (reload by default)

Stop with Ctrl+C; FalkorDB keeps running in the background.
Stop FalkorDB: docker stop falkordb-dev

Env overrides: FALKORCHAT_WS_ID, FALKORCHAT_USER_ID, FALKORDB_HOST,
               FALKORDB_PORT, EMBEDDING_DIM (default 1024), UVICORN_ARGS,
               FALKORCHAT_ENABLE_AGENT (default 1), FALKORCHAT_WORKFLOW_ENABLED
               (default 1), FALKORCHAT_AGENT_ID, FALKORCHAT_AGENT_NAME,
               FALKORCHAT_TRIGGER_DEF_KEY (default triage), FALKORCHAT_TRIGGER_DEF_VERSION
               (default v1), FALKORCHAT_OPENCODE_CONFIG (default
               \$HOME/.config/opencode/opencode.json), FALKORCHAT_MODEL_CONFIG
               (default falkor-chat/config/models.json)
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
EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
FALKORCHAT_ENABLE_AGENT="${FALKORCHAT_ENABLE_AGENT:-1}"
FALKORCHAT_WORKFLOW_ENABLED="${FALKORCHAT_WORKFLOW_ENABLED:-1}"
FALKORCHAT_AGENT_ID="${FALKORCHAT_AGENT_ID:-assistant}"
FALKORCHAT_AGENT_NAME="${FALKORCHAT_AGENT_NAME:-Assistant}"
# K-037/Finding 2: defaults match config.py's own TRIGGER_DEF_KEY/_VERSION defaults, so an
# unoverridden run's banner (stage 6 below) reflects reality. Declared here (not just left
# to config.py's own default) so the banner can interpolate the actual configured value
# instead of a hardcoded literal.
FALKORCHAT_TRIGGER_DEF_KEY="${FALKORCHAT_TRIGGER_DEF_KEY:-triage}"
FALKORCHAT_TRIGGER_DEF_VERSION="${FALKORCHAT_TRIGGER_DEF_VERSION:-v1}"
UVICORN_ARGS="${UVICORN_ARGS:---reload}"
# K-042 §4.1: the product carries no home-directory default (a default pointing into
# one specific user's home is the "works on my box" failure mode) — this dev script
# supplies the convenience default instead, so the stakeholder's day-to-day command
# stays unchanged while a bare `uvicorn` run must be explicit.
FALKORCHAT_OPENCODE_CONFIG="${FALKORCHAT_OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$REPO_DIR/server"
VENV_DIR="$SERVER_DIR/.venv"

# ── 1. FalkorDB ───────────────────────────────────────────────────────────────
if docker inspect falkordb-dev --format '{{.State.Status}}' 2>/dev/null | grep -q running; then
  echo "[1/6] FalkorDB already running — ok"
else
  echo "[1/6] Starting FalkorDB (detached)..."
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
echo "[2/6] Setting up Python venv..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install -q -e "$SERVER_DIR[dev]"

# ── 3. Bootstrap schema ───────────────────────────────────────────────────────
echo "[3/6] Bootstrapping schema for workspace '$FALKORCHAT_WS_ID' (dim $EMBEDDING_DIM)..."
EMBEDDING_DIM="$EMBEDDING_DIM" "$REPO_DIR/scripts/bootstrap_schema.sh" "$FALKORCHAT_WS_ID"

# ── 4. Seed the agent + demo channel/thread ──────────────────────────────────
echo "[4/6] Seeding agent '$FALKORCHAT_AGENT_ID' + demo channel/thread..."
FALKORCHAT_WS_ID="$FALKORCHAT_WS_ID" FALKORCHAT_USER_ID="$FALKORCHAT_USER_ID" \
FALKORCHAT_AGENT_ID="$FALKORCHAT_AGENT_ID" FALKORCHAT_AGENT_NAME="$FALKORCHAT_AGENT_NAME" \
FALKORDB_HOST="$FALKORDB_HOST" FALKORDB_PORT="$FALKORDB_PORT" \
  "$REPO_DIR/scripts/seed_demo.sh" "$FALKORCHAT_WS_ID"

# ── 5. Seed the M3 triage workflow def ────────────────────────────────────────
# Must run before uvicorn: the trigger's @mention-to-start resolves the def by
# TRIGGER_DEF_KEY/VERSION; without a published def, a WORKFLOW_ENABLED @mention is a
# silent no-op. Idempotent (publish/materialize MERGE on the fixed key/version).
# Only FALKORCHAT_WS_ID/FALKORDB_HOST/FALKORDB_PORT are explicitly forwarded below, but
# seed_workflows.sh's own FALKORCHAT_TRIAGE_DEF_KEY/_VERSION (K-037) still reach it if the
# caller exported them before invoking this script — ordinary bash env inheritance, the
# same mechanism the original K-037 bug relied on, now pointed at the harmless var.
case "$(printf '%s' "$FALKORCHAT_WORKFLOW_ENABLED" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) WF_ON=1 ;;
  *) WF_ON=0 ;;
esac
if [ "$WF_ON" = "1" ]; then
  echo "[5/6] Seeding the triage workflow def..."
  FALKORCHAT_WS_ID="$FALKORCHAT_WS_ID" \
  FALKORDB_HOST="$FALKORDB_HOST" FALKORDB_PORT="$FALKORDB_PORT" \
    "$REPO_DIR/scripts/seed_workflows.sh" "$FALKORCHAT_WS_ID"
else
  echo "[5/6] Workflow engine disabled — skipping triage def seed."
fi

# ── 6. Start uvicorn ──────────────────────────────────────────────────────────
echo "[6/6] Starting uvicorn on http://localhost:8000..."
echo "      Workspace: $FALKORCHAT_WS_ID  |  User: $FALKORCHAT_USER_ID  |  Dim: $EMBEDDING_DIM"
echo "      AI agent:  enabled=$FALKORCHAT_ENABLE_AGENT  id=$FALKORCHAT_AGENT_ID (@mention to trigger)"
echo "      Model config: opencode=$FALKORCHAT_OPENCODE_CONFIG  overlay=${FALKORCHAT_MODEL_CONFIG:-<falkor-chat>/config/models.json}"
echo "      Workflow:  enabled=$FALKORCHAT_WORKFLOW_ENABLED (triage def ${FALKORCHAT_TRIGGER_DEF_KEY}@${FALKORCHAT_TRIGGER_DEF_VERSION})"
echo "      MCP endpoint: http://localhost:8000/mcp"
echo "      Web UI:       http://localhost:8000/"
echo "      Stop with Ctrl+C (FalkorDB keeps running in background)"
echo ""
# FALKORCHAT_EMBEDDING_DIM MUST match the workspace's vector index (a wrong-dim
# vecf32 write is silently accepted then drops out of ANN — DEF in config.py).
export FALKORCHAT_WS_ID FALKORCHAT_USER_ID FALKORDB_HOST FALKORDB_PORT
export FALKORCHAT_EMBEDDING_DIM="$EMBEDDING_DIM"
export FALKORCHAT_ENABLE_AGENT FALKORCHAT_AGENT_ID FALKORCHAT_AGENT_NAME
export FALKORCHAT_WORKFLOW_ENABLED
export FALKORCHAT_TRIGGER_DEF_KEY FALKORCHAT_TRIGGER_DEF_VERSION
# K-042: the shared, pristine OpenCode file (providers only). FALKORCHAT_MODEL_CONFIG
# is left unexported — config.py's own default (falkor-chat/config/models.json) is
# right for the vast majority of runs; export it yourself to point at a different
# overlay.
export FALKORCHAT_OPENCODE_CONFIG
exec "$VENV_DIR/bin/uvicorn" falkorchat.app:app $UVICORN_ARGS
