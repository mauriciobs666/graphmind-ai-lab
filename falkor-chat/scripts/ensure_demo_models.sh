#!/usr/bin/env bash
# ensure_demo_models.sh — K-066: make sure LM Studio has the storefront demo's
# models loaded, with the demo settings, before the storefront serves.
# start_demo.sh runs it before uvicorn; it is also safe to run on its own.
#
# The demo set is derived from the code that owns it (never a copy):
#   chat model — SALESPERSON_DEF's `agent` step config.model (falkorchat/proof_defs.py),
#                which query_graph_data's internal call inherits
#   embedder   — the overlay's `embedding` default (FALKORCHAT_MODEL_CONFIG)
# resolved through ModelGateway, so the LM Studio URL it checks is the lmstudio
# baseURL in FALKORCHAT_OPENCODE_CONFIG — the one the server will call.
#
# Behaviour: reads GET <lmstudio>/api/v0/models, then
#   - stops (exit 1) if any model OUTSIDE the set is loaded (e.g. qwen, which the
#     :8200 agent-team server may be using), or the chat model is loaded with a
#     context other than 8192 — it never unloads or reloads anything;
#   - loads whichever demo model is missing: `lms load <id> -c 8192 --gpu max
#     --identifier <id> -y`, then re-reads the API to confirm (waiting up to 30 s
#     while a just-loaded model still reports `loading`);
#   - does nothing when both are already loaded (idempotent; no CLI needed).
# The embedder's context is not checked (the API misreports it as 2048) and GPU
# offload is not observable — see scripts/ensure_demo_models.py's docstring.
#
# Env vars:
#   LMS_CLI                    path to the lms CLI (lms or lms.exe). Unset: `lms`, then
#                              `lms.exe`, on PATH. Only needed when a load is needed;
#                              absent then -> stop with load-it-yourself instructions.
#                              WSL example: LMS_CLI=/mnt/c/Users/<you>/.lmstudio/bin/lms.exe
#   FALKORCHAT_OPENCODE_CONFIG (default: $HOME/.config/opencode/opencode.json) — gives the
#                              LM Studio baseURL; must match what the server gets.
#   FALKORCHAT_MODEL_CONFIG    (default: falkor-chat/config/models.json, via config.py)
#
# Options: --print-set   print the derived set and LM Studio URL as JSON, then exit.
# Workspace model overrides on ws:demo are not read (see the .py docstring's LIMIT).
#
# Exit: 0 ready · 1 refused/failed (the message says what to do) · 2 usage ·
#       3 the URL answered but is not LM Studio — HTTP 404/405, or a 200 that is not
#         LM Studio's JSON (only then is DEMO_ENSURE_MODELS=0 apt; any other HTTP error is 1).

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0 ;; esac

set -euo pipefail

FALKORCHAT_OPENCODE_CONFIG="${FALKORCHAT_OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO_DIR/server/.venv/bin/python"

[ -x "$PY" ] || {
  echo "ERROR: venv missing at $REPO_DIR/server/.venv — run: cd server && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
}

export FALKORCHAT_OPENCODE_CONFIG
exec "$PY" "$REPO_DIR/scripts/ensure_demo_models.py" "$@"
