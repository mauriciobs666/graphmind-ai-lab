#!/usr/bin/env bash
# compose-status.sh <slug> — read-only `docker compose ps` for one
# environments.json-registered environment. Inner script: `tank` invokes this
# by exact, absolute path (opencode/docs/plans/devops-opencode-headless.md
# §3.2); it never builds a compose invocation from the model's own words.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$SCRIPT_DIR/lib.sh"

require_one_arg "compose-status.sh" "$@"
SLUG="$1"

if ! RESOLVED="$(resolve_slug "$SLUG")"; then
  echo "REFUSED: '${SLUG}' is not a known environment (see environments.json)" >&2
  exit 1
fi

COMPOSE_FILE="${RESOLVED%%$'\t'*}"
PROJECT_DIR="${RESOLVED#*$'\t'}"

exec docker compose \
  -f "$REPO_ROOT/$COMPOSE_FILE" \
  --project-directory "$REPO_ROOT/$PROJECT_DIR" \
  ps
