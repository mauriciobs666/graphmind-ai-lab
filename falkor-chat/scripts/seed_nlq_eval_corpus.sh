#!/usr/bin/env bash
set -euo pipefail

# seed_nlq_eval_corpus.sh — seed (or re-seed) the dedicated NL-query-generation
# evaluation workspace `ws:nlq-eval` (K-055 M6 unit U29,
# docs/plans/workflow-nl-query-generation-ml.md §4).
#
# Thin wrapper, mirrors scripts/seed_eval_corpus.sh's role: checks FalkorDB is
# reachable, checks the server/.venv exists, execs the real logic
# (scripts/seed_nlq_eval_corpus.py) with $@ passed through.
#
# UNLIKE seed_eval_corpus.sh, this wrapper also checks that a REAL server is
# already running and reachable at NLQ_EVAL_BASE_URL, wired to ws:nlq-eval
# with FALKORCHAT_ENABLE_AGENT=1 — document ingestion's background
# embed+extract scheduling lives only in the REST/MCP transport handlers
# (background._schedule_chunk_processing), never in Services.ingest_document
# itself, so a bare script cannot drive extraction on its own. This script
# does NOT start that server for you (spawning/managing a long-running
# uvicorn from inside a wrapper script is its own can of worms — reload
# behavior, port conflicts, log capture); it fails loudly with the exact
# command to start one if it isn't up yet.
#
# `ws:nlq-eval` is a PERSISTENT workspace, same posture as `ws:eval` (not
# rebuilt every run). Re-running this script against an already-seeded
# ws:nlq-eval is a safe no-op — see seed_nlq_eval_corpus.py's own docstring
# for the exact idempotency strategy (fixed per-document `title`, terminal
# status skip, FORCE_REINGEST escape hatch).
#
# Usage:
#   # 1. Bootstrap the workspace once (idempotent):
#   EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh nlq-eval
#
#   # 2. Start a dedicated server for it (separate terminal / background):
#   FALKORCHAT_WS_ID=nlq-eval FALKORCHAT_USER_ID=nlq-author \
#   EMBEDDING_DIM=1024 FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_WORKFLOW_ENABLED=0 \
#   UVICORN_ARGS='--port 8010' ./scripts/start_server.sh
#
#   # 3. Seed the corpus against it:
#   NLQ_EVAL_BASE_URL=http://127.0.0.1:8010 ./scripts/seed_nlq_eval_corpus.sh
#
#   FORCE_REINGEST=1 ./scripts/seed_nlq_eval_corpus.sh   # deliberate full re-post
#                                                          # (only after a fresh
#                                                          # GRAPH.DELETE ws:nlq-eval
#                                                          # + re-bootstrap; see the
#                                                          # .py docstring)
#
# Env overrides:
#   FALKORDB_HOST (127.0.0.1)  FALKORDB_PORT (6379)
#   NLQ_EVAL_WS (nlq-eval)       — graph key is ws:${NLQ_EVAL_WS}
#   NLQ_EVAL_BASE_URL (http://127.0.0.1:8010) — the running server to POST/GET against
#   FORCE_REINGEST (unset)       — bypass the by-title idempotency check, re-POST everything
#   NLQ_EVAL_POLL_ATTEMPTS (30) / NLQ_EVAL_POLL_INTERVAL_S (2) — status-poll tuning

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0 ;; esac

FALKORDB_HOST="${FALKORDB_HOST:-127.0.0.1}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
NLQ_EVAL_BASE_URL="${NLQ_EVAL_BASE_URL:-http://127.0.0.1:8010}"

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

echo "Checking ${NLQ_EVAL_BASE_URL}/health..."
curl -sf -m 5 "${NLQ_EVAL_BASE_URL}/health" >/dev/null 2>&1 || {
  echo "ERROR: ${NLQ_EVAL_BASE_URL} is not reachable. This corpus needs a REAL" >&2
  echo "running server wired to ws:nlq-eval with FALKORCHAT_ENABLE_AGENT=1" >&2
  echo "(document ingestion's background extraction is wired only in the" >&2
  echo "REST/MCP transport, never in Services.ingest_document alone). Start one:" >&2
  echo "" >&2
  echo "  FALKORCHAT_WS_ID=nlq-eval FALKORCHAT_USER_ID=nlq-author \\" >&2
  echo "  EMBEDDING_DIM=1024 FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_WORKFLOW_ENABLED=0 \\" >&2
  echo "  UVICORN_ARGS='--port 8010' ./scripts/start_server.sh" >&2
  exit 1
}

export FALKORDB_HOST FALKORDB_PORT NLQ_EVAL_BASE_URL
exec "$PY" "$REPO_DIR/scripts/seed_nlq_eval_corpus.py" "$@"
