#!/usr/bin/env bash
# compose-up.sh <slug> — bring up one environments.json-registered environment
# and, only on a successful cold start, write its ownership marker. Inner
# script: `tank` invokes this by exact, absolute path (plan §3.2); it never
# builds a compose invocation from the model's own words, and it owns the
# marker read/write directly in its own code (plan §3.3), never via an
# OpenCode `write` tool call.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$SCRIPT_DIR/lib.sh"

require_one_arg "compose-up.sh" "$@"
SLUG="$1"

if ! RESOLVED="$(resolve_slug "$SLUG")"; then
  echo "REFUSED: '${SLUG}' is not a known environment (see environments.json)" >&2
  exit 1
fi

COMPOSE_FILE="${RESOLVED%%$'\t'*}"
PROJECT_DIR="${RESOLVED#*$'\t'}"
COMPOSE_FILE_ABS="$REPO_ROOT/$COMPOSE_FILE"
PROJECT_DIR_ABS="$REPO_ROOT/$PROJECT_DIR"

# Already up? Refuse to (re)claim ownership of something a human or another
# agent started — never rewrite the marker in that case (plan §3.3 step 1).
RUNNING="$(docker compose -f "$COMPOSE_FILE_ABS" --project-directory "$PROJECT_DIR_ABS" ps --status running -q)"
if [ -n "$RUNNING" ]; then
  echo "REFUSED: '${SLUG}' is already running, not started by me" >&2
  exit 1
fi

if ! docker compose -f "$COMPOSE_FILE_ABS" --project-directory "$PROJECT_DIR_ABS" up -d --build; then
  status=$?
  echo "ERROR: compose up failed for '${SLUG}' (exit ${status}); marker not written" >&2
  exit "$status"
fi

write_marker "$SLUG" "$(marker_path "$SLUG")"
echo "OK: '${SLUG}' brought up"
