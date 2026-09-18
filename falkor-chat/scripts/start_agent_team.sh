#!/usr/bin/env bash
set -euo pipefail

# start_agent_team.sh — DESIGN DRAFT, NOT YET WIRED FOR USE.
#
# Written for `claude/docs/plans/agent-knowledge-base-strategy.md` §4.3/§3 Stage 3
# (Track 1, raw-capture migration). Do NOT run this yet:
#   1. It calls `seed_agent_team.sh`, Stage 2's deliverable — that script does not
#      exist yet (Stage 2 is not merged).
#   2. It must run a falkor-chat build that includes Stage 1's `produced_by`
#      patch to `ingest_document`/`ingest_documents` — not yet merged either.
# Running this today against the current tree would stand up a process that
# looks healthy but cannot deliver FR-8/AC-6's per-agent attribution, against a
# workspace nothing has bootstrapped correctly yet. Once Stages 1-2 land, drop
# this notice and the script is otherwise ready to run as-is.
#
# Bring-up for the dedicated falkor-chat server process serving ws:agent-team —
# the shared, always-on knowledge-base substrate every Claude Code agent in
# this repo's `claude/` team reads/writes via MCP (raw kaizen capture now,
# distilled-knowledge retrieval later, §3 Track 2). Mirrors start_server.sh's
# shape (plain dev/API deployment: FalkorDB -> venv -> bootstrap -> seed ->
# uvicorn), NOT start_demo.sh's shape — this process serves no storefront, no
# workflow engine, and no human-facing chat UI, so those steps are dropped
# rather than disabled via flags.
#
# **Pins FALKORCHAT_WS_ID=agent-team, overridable but never left to config.py's
# "acme" default** — the exact convention start_demo.sh already uses for its
# own separate `demo` workspace (falkor-chat/AGENTS.md's `Key scripts` table).
#
# **Runs on its own port (default 8200), distinct from every other falkor-chat
# deployment's** (start_server.sh/start_demo.sh both default to uvicorn's own
# 8000; load_test.sh uses 8100 for its own throwaway runs) — this process must
# coexist with whichever of those a developer also has running locally.
#
# **Must NOT diverge the model-config overlay.** Do not export
# FALKORCHAT_MODEL_CONFIG here, and do not add a `ws:agent-team`-specific entry
# to that overlay's workspace-override section. An unconfigured workspace
# falls through to the shared default correctly by itself (`ModelGateway`) —
# adding an override here would silently invalidate the retrieval calibration
# work planned for Track 2 (plan §4.3, `analyst`'s Pass 2 finding).
#
# Deliberately NOT run: seed_workflows.sh (no workflow engine needed — this
# process only serves `ingest_document`/`search_documents`/`list_documents`/
# `delete_document`/`get_document`/`get_document_history`), seed_catalog.sh /
# seed_salesperson.sh (storefront-only), any SPA build step.
#
# Override defaults with env vars:
#   FALKORCHAT_WS_ID       (default: agent-team)  — see the pin note above
#   FALKORCHAT_USER_ID     (default: u1)           — unused by this workspace's
#                          actual traffic (no chat happens here) but still
#                          required by config.py/get_context(); left at the
#                          repo's usual dev default rather than invented anew.
#   FALKORDB_HOST          (default: 127.0.0.1)
#   FALKORDB_PORT          (default: 6379)
#   EMBEDDING_DIM          (default: 1024)   — MUST match this repo's standing
#                          convention (Qwen3-Embedding); exported to the app as
#                          FALKORCHAT_EMBEDDING_DIM.
#   AGENT_TEAM_PORT        (default: 8200)   — this process's own port; folded
#                          into UVICORN_ARGS's default below.
#   FALKORCHAT_ENABLE_AGENT(default: 1)      — REQUIRED on: search_documents
#                          needs the embedder wired (app._build_default_app).
#                          Do not set this to 0 for this process.
#   FALKORCHAT_OPENCODE_CONFIG(default: $HOME/.config/opencode/opencode.json)
#                          — same convention as start_server.sh/start_demo.sh.
#   FALKORCHAT_MODEL_CONFIG — deliberately NOT set/defaulted here. Leave unset
#                          so config.py's own shared default overlay applies
#                          unchanged. See the model-config note above.
#   UVICORN_ARGS           (default: --port ${AGENT_TEAM_PORT}) — deliberately
#                          NOT `--reload` (§ lifecycle note below: this is an
#                          always-on process many independent agent sessions
#                          read from at unpredictable times; a file write
#                          anywhere under falkor-chat/ must not kill it
#                          mid-read, same reasoning start_demo.sh already
#                          documents for its own always-on-during-a-demo case).
#
# Example (custom port):
#   AGENT_TEAM_PORT=8201 ./scripts/start_agent_team.sh

usage() {
  cat <<EOF
Usage: start_agent_team.sh [-h|--help]

Brings up the dedicated falkor-chat server process for ws:agent-team:
  1. Starts FalkorDB (detached) if not already running
  2. Creates/updates the server's Python venv
  3. Bootstraps schema for the agent-team workspace (EMBEDDING_DIM)
  4. Seeds the Agent roster (seed_agent_team.sh — Stage 2 deliverable)
  5. Starts uvicorn on its own port (default 8200), --reload OFF

Stop with Ctrl+C; FalkorDB keeps running in the background.
Stop FalkorDB: docker stop falkordb-dev

Every env override is documented in this script's own header comment. The
workspace defaults to 'agent-team', never falkorchat.config's 'acme' default.

Health check, once up:
  curl -sf http://127.0.0.1:\${AGENT_TEAM_PORT:-8200}/health
(503 until the workspace is bootstrapped and FalkorDB answers; 200 once ready
— see the plan's §4.3/closing-item discussion for the fuller pinning check.)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) echo "start_agent_team.sh: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ── defaults ──────────────────────────────────────────────────────────────────
FALKORCHAT_WS_ID="${FALKORCHAT_WS_ID:-agent-team}"
FALKORCHAT_USER_ID="${FALKORCHAT_USER_ID:-u1}"
FALKORDB_HOST="${FALKORDB_HOST:-127.0.0.1}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
AGENT_TEAM_PORT="${AGENT_TEAM_PORT:-8200}"
FALKORCHAT_ENABLE_AGENT="${FALKORCHAT_ENABLE_AGENT:-1}"
if [ "$FALKORCHAT_ENABLE_AGENT" != "1" ]; then
  echo "ERROR: FALKORCHAT_ENABLE_AGENT must be 1 for the agent-team process —" >&2
  echo "       search_documents needs the embedder wired (app._build_default_app)." >&2
  exit 1
fi
FALKORCHAT_OPENCODE_CONFIG="${FALKORCHAT_OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"
if [ -n "${FALKORCHAT_MODEL_CONFIG:-}" ]; then
  echo "WARNING: FALKORCHAT_MODEL_CONFIG is set. This process must use the same," >&2
  echo "         default model-config overlay as the rest of the deployment —" >&2
  echo "         a ws:agent-team-specific override would silently invalidate the" >&2
  echo "         retrieval calibration work planned for Track 2. Unset it unless" >&2
  echo "         you are certain this override applies repo-wide, not just here." >&2
fi
# `:-` substitutes on unset OR empty. Deliberately not `--reload` — see the
# lifecycle note in the header comment.
UVICORN_ARGS="${UVICORN_ARGS:---port ${AGENT_TEAM_PORT}}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$REPO_DIR/server"
VENV_DIR="$SERVER_DIR/.venv"

TOTAL_STEPS=5

# ── 1. FalkorDB ───────────────────────────────────────────────────────────────
echo "[1/$TOTAL_STEPS] FalkorDB"
if docker inspect falkordb-dev --format '{{.State.Status}}' 2>/dev/null | grep -q running; then
  echo "      Already running — ok"
else
  echo "      Starting FalkorDB (detached)..."
  "$REPO_DIR/scripts/start_falkordb.sh" -d
fi
echo "      Waiting for FalkorDB to be ready..."
falkordb_ready=0
for i in $(seq 1 30); do
  if redis-cli -h "$FALKORDB_HOST" -p "$FALKORDB_PORT" ping 2>/dev/null | grep -q PONG; then
    echo "      Ready."
    falkordb_ready=1
    break
  fi
  sleep 1
done
if [ "$falkordb_ready" -ne 1 ]; then
  echo "ERROR: FalkorDB did not respond at ${FALKORDB_HOST}:${FALKORDB_PORT} after 30s" >&2
  exit 1
fi

# ── 2. venv + deps ────────────────────────────────────────────────────────────
echo "[2/$TOTAL_STEPS] Python venv (server)"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install -q -e "$SERVER_DIR[dev]"

# ── 3. Bootstrap schema ───────────────────────────────────────────────────────
echo "[3/$TOTAL_STEPS] Bootstrapping schema for workspace '$FALKORCHAT_WS_ID' (dim $EMBEDDING_DIM)..."
EMBEDDING_DIM="$EMBEDDING_DIM" FALKORDB_HOST="$FALKORDB_HOST" FALKORDB_PORT="$FALKORDB_PORT" \
  "$REPO_DIR/scripts/bootstrap_schema.sh" "$FALKORCHAT_WS_ID"

# ── 4. Seed the Agent roster (Stage 2 deliverable) ───────────────────────────
echo "[4/$TOTAL_STEPS] Seeding the agent-team Agent roster..."
if [ ! -x "$REPO_DIR/scripts/seed_agent_team.sh" ]; then
  echo "ERROR: $REPO_DIR/scripts/seed_agent_team.sh not found or not executable." >&2
  echo "       This is Stage 2's deliverable (claude/docs/plans/agent-knowledge-base-strategy.md" >&2
  echo "       §3 Track 1 Stage 2) — bootstrap it before running this script." >&2
  exit 1
fi
FALKORCHAT_WS_ID="$FALKORCHAT_WS_ID" \
FALKORDB_HOST="$FALKORDB_HOST" FALKORDB_PORT="$FALKORDB_PORT" \
  "$REPO_DIR/scripts/seed_agent_team.sh" "$FALKORCHAT_WS_ID"

# ── 5. Start uvicorn ──────────────────────────────────────────────────────────
echo "[5/$TOTAL_STEPS] Starting uvicorn on http://127.0.0.1:${AGENT_TEAM_PORT}..."
echo ""
echo "      ══════════════════════════════════════════════════════════════"
echo "      Workspace: ws:${FALKORCHAT_WS_ID}  (pinned — never config.py's 'acme' default)"
echo "      Dim: $EMBEDDING_DIM  |  AI/embedder enabled=$FALKORCHAT_ENABLE_AGENT"
echo "      Workflow engine: OFF (not needed)  |  Storefront: OFF (not needed)"
echo "      Model config: opencode=$FALKORCHAT_OPENCODE_CONFIG  overlay=${FALKORCHAT_MODEL_CONFIG:-<falkor-chat>/config/models.json (shared default — do not override)}"
echo "      MCP endpoint: http://127.0.0.1:${AGENT_TEAM_PORT}/mcp"
echo "      Health:       http://127.0.0.1:${AGENT_TEAM_PORT}/health"
echo "      UVICORN_ARGS: $UVICORN_ARGS  (--reload OFF — always-on process, see header note)"
echo "      Stop with Ctrl+C (FalkorDB keeps running in background)"
echo "      ══════════════════════════════════════════════════════════════"
echo ""

export FALKORCHAT_WS_ID FALKORCHAT_USER_ID FALKORDB_HOST FALKORDB_PORT
export FALKORCHAT_EMBEDDING_DIM="$EMBEDDING_DIM"
export FALKORCHAT_ENABLE_AGENT
export FALKORCHAT_OPENCODE_CONFIG
# FALKORCHAT_WORKFLOW_ENABLED / FALKORCHAT_STOREFRONT_ENABLED intentionally NOT
# exported — config.py's own defaults (both off) are exactly right here.
# FALKORCHAT_MODEL_CONFIG intentionally NOT exported — see the header note.
exec "$VENV_DIR/bin/uvicorn" falkorchat.app:app $UVICORN_ARGS
