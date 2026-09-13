#!/usr/bin/env bash
# tear-down.sh <slug> — ask tank to tear down one environment it itself
# brought up (it refuses anything else — see compose-down.sh / the state
# marker in README.md).
#
# Outer script (see health-check.sh's header note for the outer/inner
# distinction). Same non-mutating relationship to <slug> as bring-up.sh.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $(basename "$0") <slug>" >&2
  echo "  <slug> is a key in opencode/agents/tank/environments.json" >&2
  exit 1
fi
SLUG="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TANK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$TANK_DIR"
# --title avoids OpenCode's background auto-title call, which errors against
# this model's chat template — see health-check.sh's header note for detail.
exec opencode run --agent tank --title "tank tear-down: ${SLUG}" "Tear down the '${SLUG}' environment."
