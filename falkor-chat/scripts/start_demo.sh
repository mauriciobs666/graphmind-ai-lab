#!/usr/bin/env bash
set -euo pipefail

# start_demo.sh — from-cold-box bring-up of the salesperson storefront demo
# (docs/plans/salesperson-ui.md S11). Stands up FalkorDB, bootstraps schema,
# seeds the demo agent/catalog/salesperson defs, preflight-verifies them,
# builds the SPA, and launches uvicorn as the storefront deployment — never
# the dev/API deployment `start_server.sh` produces.
#
# **Pins FALKORCHAT_WS_ID=demo, overridable but never left to config.py's
# `"acme"` default.** `ws:acme` is this repo's populated dev/demo workspace
# (seed_demo.sh's channel/thread, the M2/M5 hand-verification transcript) —
# serving the storefront there by omission would mix demo traffic into it.
# Every seed AND verify script below gets the workspace explicitly, even
# though the pin already makes their own default correct: defence in depth,
# not a load-bearing requirement (docs/plans/salesperson-ui.md §4.9 move 2).
#
# Deliberately NOT run: seed_workflows.sh — this demo needs neither `triage`
# nor `access-request`, only `salesperson`/`order-fulfillment`.
#
# Override defaults with env vars:
#   FALKORCHAT_WS_ID       (default: demo)   — see the pin note above
#   FALKORCHAT_USER_ID     (default: u1)
#   FALKORDB_HOST          (default: 127.0.0.1)
#   FALKORDB_PORT          (default: 6379)
#   EMBEDDING_DIM          (default: 1024)   — MUST match the workspace's vector
#                          index (Qwen3-Embedding). Exported to the app as
#                          FALKORCHAT_EMBEDDING_DIM.
#   FALKORCHAT_AGENT_ID    (default: assistant)
#   FALKORCHAT_AGENT_NAME  (default: Assistant)
#   FALKORCHAT_TRIGGER_DEF_KEY     (default: salesperson) — pinned so an @mention
#   FALKORCHAT_TRIGGER_DEF_VERSION (default: v7)           starts the demo agent,
#                          not the dev deployment's `triage`.
#   FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH (default: 0) — OFF: the M2 responder's
#                          workspace-wide retrieval must stay structurally
#                          unreachable in the storefront (§4.3 part 4).
#   FALKORCHAT_ENABLE_AGENT    (default: 1)
#   FALKORCHAT_WORKFLOW_ENABLED(default: 1)
#   FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S (default: 30)
#   FALKORCHAT_STOREFRONT_ENABLED (default: 1) — pinned on: this IS the storefront
#                          deployment, so the legacy REST/`/`/`/mcp` surfaces are
#                          not registered (§4.9 move 1).
#   FALKORCHAT_STOREFRONT_DIR (default: <repo-root>/salesperson/dist) — the BUILT
#                          SPA output, never the source tree. Overriding this to a
#                          directory with no `index.html` is a loud failure below,
#                          not a silent 404 once uvicorn is up.
#   FALKORCHAT_STOREFRONT_PRESENTER_KEY (default: unset — no presenter surface;
#                          export it yourself, this script sets no default so a
#                          demo secret is never invented on the operator's behalf)
#   FALKORCHAT_OPENCODE_CONFIG(default: $HOME/.config/opencode/opencode.json)
#                          — required whenever ENABLE_AGENT/WORKFLOW_ENABLED is on
#                          (K-042 §4.1); this script supplies the dev convenience
#                          default, same as start_server.sh.
#   FALKORCHAT_MODEL_CONFIG(default: falkor-chat/config/models.json, config.py's
#                          own default — no need to export it here)
#   SALESPERSON_DIR        (default: <repo-root>/salesperson) — the SPA component
#   SALESPERSON_BUILD_ARGS (default: empty) — extra args forwarded to
#                          salesperson/build.sh, e.g. `--skip-install`
#   UVICORN_ARGS           (default: --host 0.0.0.0) — MUST be non-empty: `:-`
#                          substitutes the default on unset OR empty, so
#                          UVICORN_ARGS="" does NOT disable --reload. The default
#                          here is deliberately not `--reload` (R7): a live demo
#                          must not have its in-flight background work silently
#                          killed by a file write under falkor-chat/ mid-session.
#
# Example (custom workspace):
#   FALKORCHAT_WS_ID=myws ./scripts/start_demo.sh

usage() {
  cat <<EOF
Usage: start_demo.sh [-h|--help]

From a cold box, brings up the full salesperson storefront demo in one
terminal:
  1. Starts FalkorDB (detached) if not already running
  2. Creates/updates the server's Python venv
  3. Bootstraps schema for the demo workspace (EMBEDDING_DIM)
  4. Seeds the demo agent + channel/thread, then verifies the Agent exists
  5. Seeds the product catalog, then preflight-verifies it
  6. Seeds the salesperson + order-fulfillment workflow defs, then
     preflight-verifies both against 'reference'
  7. Builds the SPA (salesperson/build.sh) and checks the built bundle exists
  8. Starts uvicorn as the storefront deployment (--reload OFF)

Stop with Ctrl+C; FalkorDB keeps running in the background.
Stop FalkorDB: docker stop falkordb-dev

Every env override is documented in this script's own header comment.
The workspace defaults to 'demo', never falkorchat.config's 'acme' default —
see the pin note at the top of this file.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) echo "start_demo.sh: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ── defaults ──────────────────────────────────────────────────────────────────
FALKORCHAT_WS_ID="${FALKORCHAT_WS_ID:-demo}"
FALKORCHAT_USER_ID="${FALKORCHAT_USER_ID:-u1}"
FALKORDB_HOST="${FALKORDB_HOST:-127.0.0.1}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
FALKORCHAT_AGENT_ID="${FALKORCHAT_AGENT_ID:-assistant}"
FALKORCHAT_AGENT_NAME="${FALKORCHAT_AGENT_NAME:-Assistant}"
FALKORCHAT_TRIGGER_DEF_KEY="${FALKORCHAT_TRIGGER_DEF_KEY:-salesperson}"
FALKORCHAT_TRIGGER_DEF_VERSION="${FALKORCHAT_TRIGGER_DEF_VERSION:-v7}"
FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH="${FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH:-0}"
FALKORCHAT_ENABLE_AGENT="${FALKORCHAT_ENABLE_AGENT:-1}"
FALKORCHAT_WORKFLOW_ENABLED="${FALKORCHAT_WORKFLOW_ENABLED:-1}"
FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S="${FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S:-30}"
FALKORCHAT_STOREFRONT_ENABLED="${FALKORCHAT_STOREFRONT_ENABLED:-1}"
# K-042 §4.1: no product default pointing into one user's home — this dev/demo
# script supplies the convenience default (mirrors start_server.sh).
FALKORCHAT_OPENCODE_CONFIG="${FALKORCHAT_OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"
# `:-` substitutes on unset OR empty — see the header note. Default is
# deliberately not `--reload`.
UVICORN_ARGS="${UVICORN_ARGS:---host 0.0.0.0}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"          # .../falkor-chat
ROOT_DIR="$(cd "$REPO_DIR/.." && pwd)"                # monorepo root
SERVER_DIR="$REPO_DIR/server"
VENV_DIR="$SERVER_DIR/.venv"
SALESPERSON_DIR="${SALESPERSON_DIR:-$ROOT_DIR/salesperson}"
FALKORCHAT_STOREFRONT_DIR="${FALKORCHAT_STOREFRONT_DIR:-$SALESPERSON_DIR/dist}"

TOTAL_STEPS=8

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

# ── 4. Seed the agent + demo channel/thread, verify the Agent exists ─────────
echo "[4/$TOTAL_STEPS] Seeding agent '$FALKORCHAT_AGENT_ID' + demo channel/thread..."
FALKORCHAT_WS_ID="$FALKORCHAT_WS_ID" FALKORCHAT_USER_ID="$FALKORCHAT_USER_ID" \
FALKORCHAT_AGENT_ID="$FALKORCHAT_AGENT_ID" FALKORCHAT_AGENT_NAME="$FALKORCHAT_AGENT_NAME" \
FALKORDB_HOST="$FALKORDB_HOST" FALKORDB_PORT="$FALKORDB_PORT" \
  "$REPO_DIR/scripts/seed_demo.sh" "$FALKORCHAT_WS_ID"

# No standing verify_demo.sh exists (unlike catalog/salesperson/workflows) —
# this is this script's own read-only check for the sixth failure mode the
# plan's done-condition names ("the demo Agent"), GRAPH.RO_QUERY only.
echo "      Verifying Agent '${FALKORCHAT_AGENT_ID}' exists in ws:${FALKORCHAT_WS_ID}..."
agent_check="$(redis-cli -h "$FALKORDB_HOST" -p "$FALKORDB_PORT" GRAPH.RO_QUERY "ws:${FALKORCHAT_WS_ID}" \
  "CYPHER agentId='${FALKORCHAT_AGENT_ID}' MATCH (a:Agent {agentId: \$agentId}) RETURN a.agentId" --compact 2>&1)" || {
  echo "ERROR: could not query ws:${FALKORCHAT_WS_ID} for Agent '${FALKORCHAT_AGENT_ID}':" >&2
  echo "$agent_check" >&2
  exit 1
}
if ! printf '%s\n' "$agent_check" | grep -q "$FALKORCHAT_AGENT_ID"; then
  echo "ERROR: demo Agent '${FALKORCHAT_AGENT_ID}' not found in ws:${FALKORCHAT_WS_ID} after seeding." >&2
  echo "       @mention-to-start will silently no-op without it. Re-run:" >&2
  echo "         ./scripts/seed_demo.sh ${FALKORCHAT_WS_ID}" >&2
  exit 1
fi
echo "      Agent present — ok"

# ── 5. Seed + verify the product catalog ─────────────────────────────────────
echo "[5/$TOTAL_STEPS] Seeding product catalog..."
FALKORDB_HOST="$FALKORDB_HOST" FALKORDB_PORT="$FALKORDB_PORT" \
  "$REPO_DIR/scripts/seed_catalog.sh" "$FALKORCHAT_WS_ID"
echo "      Preflight: verify_catalog.sh"
if ! FALKORDB_HOST="$FALKORDB_HOST" FALKORDB_PORT="$FALKORDB_PORT" \
     "$REPO_DIR/scripts/verify_catalog.sh"; then
  echo "ERROR: catalog verification failed — see output above." >&2
  exit 1
fi

# ── 6. Seed + verify the salesperson + order-fulfillment defs ────────────────
echo "[6/$TOTAL_STEPS] Seeding salesperson + order-fulfillment defs..."
FALKORCHAT_WS_ID="$FALKORCHAT_WS_ID" \
FALKORDB_HOST="$FALKORDB_HOST" FALKORDB_PORT="$FALKORDB_PORT" \
  "$REPO_DIR/scripts/seed_salesperson.sh" "$FALKORCHAT_WS_ID"
echo "      Preflight: verify_salesperson.sh"
if ! FALKORCHAT_WS_ID="$FALKORCHAT_WS_ID" \
     FALKORDB_HOST="$FALKORDB_HOST" FALKORDB_PORT="$FALKORDB_PORT" \
     "$REPO_DIR/scripts/verify_salesperson.sh" "$FALKORCHAT_WS_ID"; then
  echo "ERROR: salesperson/order-fulfillment def verification failed — see output above." >&2
  exit 1
fi

# ── 7. Build the SPA ──────────────────────────────────────────────────────────
echo "[7/$TOTAL_STEPS] Building the SPA ($SALESPERSON_DIR)..."
if [ ! -d "$SALESPERSON_DIR" ]; then
  echo "ERROR: salesperson component not found at $SALESPERSON_DIR" >&2
  echo "       Set SALESPERSON_DIR if the component lives elsewhere." >&2
  exit 1
fi
if [ ! -x "$SALESPERSON_DIR/build.sh" ]; then
  echo "ERROR: $SALESPERSON_DIR/build.sh not found or not executable." >&2
  echo "       The SPA scaffold (plan step S5) is expected to have landed it." >&2
  exit 1
fi
# build.sh itself fails loudly and specifically when Node is missing, wrong,
# or resolves to the WSL2 Windows shim (see salesperson/AGENTS.md) — no need
# to duplicate that check here. It also verifies its own dist/index.html.
# shellcheck disable=SC2086
( cd "$SALESPERSON_DIR" && ./build.sh ${SALESPERSON_BUILD_ARGS:-} )

# Redundant top-level guard: FALKORCHAT_STOREFRONT_DIR may have been overridden
# to point somewhere other than what build.sh just produced — fail loudly here
# rather than let uvicorn come up and 404 every /shop request silently.
if [ ! -f "$FALKORCHAT_STOREFRONT_DIR/index.html" ]; then
  echo "ERROR: no built bundle at $FALKORCHAT_STOREFRONT_DIR/index.html" >&2
  echo "       (FALKORCHAT_STOREFRONT_DIR=$FALKORCHAT_STOREFRONT_DIR)" >&2
  exit 1
fi
echo "      Bundle present at $FALKORCHAT_STOREFRONT_DIR — ok"

# ── 8. Start uvicorn (storefront deployment) ─────────────────────────────────
echo "[8/$TOTAL_STEPS] Starting uvicorn on http://localhost:8000..."
echo ""
echo "      ══════════════════════════════════════════════════════════════"
echo "      Workspace: ws:${FALKORCHAT_WS_ID}  (pinned — never config.py's 'acme' default)"
echo "      User:      $FALKORCHAT_USER_ID   |  Dim: $EMBEDDING_DIM"
echo "      AI agent:  enabled=$FALKORCHAT_ENABLE_AGENT  id=$FALKORCHAT_AGENT_ID"
echo "      Workflow:  enabled=$FALKORCHAT_WORKFLOW_ENABLED  trigger=${FALKORCHAT_TRIGGER_DEF_KEY}@${FALKORCHAT_TRIGGER_DEF_VERSION}  fallthrough=${FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH}"
echo "      Storefront: enabled=$FALKORCHAT_STOREFRONT_ENABLED  dir=$FALKORCHAT_STOREFRONT_DIR"
echo "      Model config: opencode=$FALKORCHAT_OPENCODE_CONFIG  overlay=${FALKORCHAT_MODEL_CONFIG:-<falkor-chat>/config/models.json}"
echo "      Shop:      http://localhost:8000/shop"
echo "      UVICORN_ARGS: $UVICORN_ARGS  (--reload OFF)"
echo "      Stop with Ctrl+C (FalkorDB keeps running in background)"
echo "      ══════════════════════════════════════════════════════════════"
echo ""

export FALKORCHAT_WS_ID FALKORCHAT_USER_ID FALKORDB_HOST FALKORDB_PORT
export FALKORCHAT_EMBEDDING_DIM="$EMBEDDING_DIM"
export FALKORCHAT_ENABLE_AGENT FALKORCHAT_AGENT_ID FALKORCHAT_AGENT_NAME
export FALKORCHAT_WORKFLOW_ENABLED FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S
export FALKORCHAT_TRIGGER_DEF_KEY FALKORCHAT_TRIGGER_DEF_VERSION
export FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH
export FALKORCHAT_STOREFRONT_ENABLED FALKORCHAT_STOREFRONT_DIR
export FALKORCHAT_OPENCODE_CONFIG
# FALKORCHAT_STOREFRONT_PRESENTER_KEY is intentionally NOT defaulted here —
# it is exported only if the caller already set it in their environment.
exec "$VENV_DIR/bin/uvicorn" falkorchat.app:app $UVICORN_ARGS
