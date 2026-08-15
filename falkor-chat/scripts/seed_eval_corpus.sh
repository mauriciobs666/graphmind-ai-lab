#!/usr/bin/env bash
set -euo pipefail

# seed_eval_corpus.sh — seed (or re-seed) the persistent GraphRAG evaluation
# workspace `ws:eval` (K-026, docs/plans/graphrag-eval.md §5 Unit 1).
#
# Thin wrapper, mirrors scripts/load_test.sh's role: checks FalkorDB is
# reachable, checks the server/.venv exists, execs the real logic
# (scripts/seed_eval_corpus.py) with $@ passed through.
#
# `ws:eval` is a PERSISTENT workspace (plan §3 D2) — unlike ws:test/ws:live/
# ws:load, it is not rebuilt every run. Re-running this script against an
# already-seeded ws:eval is a safe no-op (idempotent messages via dupMsg,
# embeddings skipped when already present).
#
# Usage:
#   ./scripts/seed_eval_corpus.sh                # seed/verify ws:eval
#   RESEED=1 ./scripts/seed_eval_corpus.sh        # force GRAPH.DELETE + rebootstrap
#                                                  # (the deliberate reset path when
#                                                  # the embedding model/dim changes)
#
# Env overrides:
#   FALKORDB_HOST (127.0.0.1)  FALKORDB_PORT (6379)
#   EVAL_WS (eval)             — graph key is ws:${EVAL_WS}
#   RESEED (unset)             — force a full reset before seeding
#   FALKORCHAT_OPENCODE_CONFIG (default: $HOME/.config/opencode/opencode.json)
#                              — K-042: the shared, pristine OpenCode providers
#                              file. Required — ModelGateway.from_env() needs it
#                              to resolve the real configured embedding model.
#   FALKORCHAT_MODEL_CONFIG   (default: falkor-chat/config/models.json, set by
#                              config.py itself — no need to export it here)
#   EVAL_USER_ID/EVAL_USER_NAME/EVAL_AGENT_ID/EVAL_AGENT_NAME — the fixed
#                              corpus-author ids (defaults: eval-author /
#                              "Corpus Author" / eval-assistant / "Eval
#                              Assistant")

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0 ;; esac

FALKORDB_HOST="${FALKORDB_HOST:-127.0.0.1}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
# K-042 §4.1: the product carries no home-directory default (a default pointing
# into one specific user's home is the "works on my box" failure mode) — this
# dev script supplies the convenience default instead, same as start_server.sh.
FALKORCHAT_OPENCODE_CONFIG="${FALKORCHAT_OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"

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
  echo "ERROR: FALKORCHAT_OPENCODE_CONFIG file not found at ${FALKORCHAT_OPENCODE_CONFIG} — set it to the shared, pristine opencode.json path (see config/opencode.example.json)" >&2
  exit 1
}

export FALKORDB_HOST FALKORDB_PORT FALKORCHAT_OPENCODE_CONFIG
exec "$PY" "$REPO_DIR/scripts/seed_eval_corpus.py" "$@"
