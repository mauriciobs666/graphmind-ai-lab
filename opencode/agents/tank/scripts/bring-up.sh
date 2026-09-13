#!/usr/bin/env bash
# bring-up.sh <slug> — ask tank to bring up one environment registered in
# environments.json.
#
# Outer script (see health-check.sh's header note for the outer/inner
# distinction). <slug> only selects *which environment to ask tank about* in
# a plain-text message — it is never turned into a docker/compose flag by
# this script or by tank: tank's own compose-up.sh does that lookup, by
# exact key match against environments.json.
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
exec opencode run --agent tank --title "tank bring-up: ${SLUG}" "Bring up the '${SLUG}' environment."
