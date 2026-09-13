#!/usr/bin/env bash
# compose-down.sh <slug> — tear down one environments.json-registered
# environment, but only if this script's own marker says it brought that
# environment up, and only if that marker is still fresh. Inner script:
# `tank` invokes this by exact, absolute path (plan §3.2).
#
# Security property (plan §3.3, security review Pass 2 nit): the compose
# file/project-directory values used below are ALWAYS re-resolved fresh from
# environments.json via resolve_slug, never read out of the marker — the
# marker schema does not even carry those fields. This file intentionally
# never spells out those two JSON key names anywhere in its own source; the
# test suite greps for that as a standing regression check.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$SCRIPT_DIR/lib.sh"

TANK_MARKER_MAX_AGE_SECONDS="${TANK_MARKER_MAX_AGE_SECONDS:-21600}" # 6h default

require_one_arg "compose-down.sh" "$@"
SLUG="$1"

# Unknown slug refuses immediately, before the marker is even looked at.
if ! RESOLVED="$(resolve_slug "$SLUG")"; then
  echo "REFUSED: '${SLUG}' is not a known environment (see environments.json)" >&2
  exit 1
fi

MARKER="$(marker_path "$SLUG")"
if [ ! -f "$MARKER" ]; then
  echo "REFUSED: '${SLUG}' — not something I brought up (no marker present)" >&2
  exit 1
fi

AGE="$(marker_age_seconds "$MARKER")"
if [ "$AGE" -gt "$TANK_MARKER_MAX_AGE_SECONDS" ]; then
  echo "REFUSED: '${SLUG}' — marker present but stale (${AGE}s old, max ${TANK_MARKER_MAX_AGE_SECONDS}s), treating as not mine" >&2
  exit 1
fi

COMPOSE_FILE="${RESOLVED%%$'\t'*}"
PROJECT_DIR="${RESOLVED#*$'\t'}"

if ! docker compose -f "$REPO_ROOT/$COMPOSE_FILE" --project-directory "$REPO_ROOT/$PROJECT_DIR" down; then
  status=$?
  echo "ERROR: compose down failed for '${SLUG}' (exit ${status}); marker left in place" >&2
  exit "$status"
fi

rm -f "$MARKER"
echo "OK: '${SLUG}' torn down"
