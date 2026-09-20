#!/usr/bin/env bash
# create_workspace.sh — canonical entry point for creating a *real*, live-served
# workspace (FR-2's enforcement mechanism, §3.2 Option B,
# docs/plans/embedding-migration.md §3.2/§5 step 2).
#
# bootstrap_schema.sh's DDL alone leaves a new workspace's embedding-model
# override unset — FR-1/FR-2's decision-log requirement is that this NOT be a
# manual convention an operator can forget. This wrapper makes the pin
# structural: `bootstrap_schema.sh "$@"`, then FR-2's pin operation
# (`pin_workspace_embedding_model.sh`) for every given workspace id.
#
# Usage:
#   ./scripts/create_workspace.sh <wsId> [<wsId> ...]
#   EMBEDDING_DIM=1024 ./scripts/create_workspace.sh acme
#
# Forwards EMBEDDING_DIM (and any other bootstrap_schema.sh env vars already in
# the environment) unchanged to bootstrap_schema.sh "$@" — this script sets none
# of its own, so whatever the caller exported flows through as-is. Then runs
# `pin_workspace_embedding_model.sh <wsId>` once per given workspace id
# (bash-to-bash — no Python of its own; the actual pin logic lives in
# `embedding_migration.py`'s `pin()`).
#
# NOT a blanket replacement for bootstrap_schema.sh — a throwaway/offline-suite
# workspace (ws:test, ws:nlq-eval) is a deliberate carve-out (§3.2): it has no
# need for FR-1/FR-2's safety net and must not gain a new hard dependency on
# FALKORCHAT_OPENCODE_CONFIG parsing successfully. Those call sites keep calling
# bootstrap_schema.sh directly — do not route them through this script.
#
# The pin step is best-effort by design, not by anything this script adds:
# `pin()` itself treats a missing/invalid FALKORCHAT_OPENCODE_CONFIG
# (ModelConfigError) as non-fatal — printed as a WARNING, not raised — which is
# what preserves FALKORCHAT_ENABLE_AGENT=0's documented no-config mode (§3.2's
# regression mitigation). This script deliberately adds no error handling
# around the pin call that would turn that WARNING back into a hard failure;
# `set -e` here only ever fires on a genuinely unexpected error (e.g. FalkorDB
# unreachable, missing venv), the same failure modes bootstrap_schema.sh itself
# already treats as fatal.
#
# Env vars: same as bootstrap_schema.sh (FALKORDB_HOST, FALKORDB_PORT,
# EMBEDDING_DIM) plus pin_workspace_embedding_model.sh's own
# (FALKORCHAT_OPENCODE_CONFIG).

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help) usage; exit 0 ;; esac

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <wsId> [<wsId> ...]" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"$REPO_DIR/scripts/bootstrap_schema.sh" "$@"

for wid in "$@"; do
  "$REPO_DIR/scripts/pin_workspace_embedding_model.sh" "$wid"
done
