#!/usr/bin/env bash
# lib.sh — shared helpers for compose-status.sh / compose-up.sh / compose-down.sh.
#
# Sourced (not executed) by each wrapper script. Owns the two pieces of logic
# that must behave identically across all three scripts: the exact-match
# environments.json lookup (resolve_slug) and the state-marker read/write
# (write_marker / marker_field / marker_path). Nothing here ever builds a
# `docker compose` invocation from anything but the values resolve_slug
# returns — see plan §3.2 (opencode/docs/plans/devops-opencode-headless.md).
set -euo pipefail

# REPO_ROOT: this file lives at opencode/agents/tank/scripts/lib.sh, four
# directories below the repo root (opencode -> agents -> tank -> scripts) —
# do not copy opencode.json's shallower {file:...} path depth here.
LIB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LIB_SCRIPT_DIR/../../../.." && pwd)"
ENV_JSON="$REPO_ROOT/opencode/agents/tank/environments.json"
STATE_DIR="$REPO_ROOT/opencode/agents/tank/state"

# require_one_arg SCRIPT_NAME ARG...
# Refuses (message to stderr, exit 1) unless exactly one argument was given.
# Must run before any other logic in every wrapper script — an argc mismatch
# is refused before environments.json is even consulted.
require_one_arg() {
  local script_name="$1"
  shift
  if [ "$#" -ne 1 ]; then
    echo "REFUSED: ${script_name} takes exactly one argument (an environments.json slug), got $#" >&2
    return 1
  fi
}

# resolve_slug SLUG [ENV_JSON_PATH]
# Prints "composeFile<TAB>projectDirectory" on an exact match and returns 0;
# prints nothing and returns 1 on any miss. This is a Python dict `.get()`
# lookup — true key equality, never substring/prefix/glob — so an unknown
# slug, a slug with trailing garbage, and a `../`-shaped value are all
# refused identically, the same way a wrong key would be.
resolve_slug() {
  local slug="$1"
  local env_json="${2:-$ENV_JSON}"
  python3 - "$slug" "$env_json" <<'PYEOF'
import json
import sys

slug, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)

entry = data.get(slug)
if entry is None:
    sys.exit(1)

print(f"{entry['composeFile']}\t{entry['projectDirectory']}")
PYEOF
}

# marker_path SLUG
marker_path() {
  printf '%s/%s.json' "$STATE_DIR" "$1"
}

# marker_field MARKER_PATH FIELD_NAME
# Reads exactly one field out of a marker JSON file. The marker schema is
# {slug, broughtUpAt, sessionId, pid, host} — it never stores composeFile or
# projectDirectory (§3.3): compose-down.sh always re-resolves those fresh via
# resolve_slug, never from here.
marker_field() {
  local marker="$1" field="$2"
  python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
value = data.get(sys.argv[2])
if value is None:
    sys.exit(1)
print(value)
' "$marker" "$field"
}

# write_marker SLUG MARKER_PATH
# Writes the ownership marker for SLUG. Called only by compose-up.sh on a
# successful cold bring-up. Never includes composeFile/projectDirectory.
write_marker() {
  local slug="$1" marker="$2"
  local now session pid host
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  pid="$$"
  session="${pid}-$(date +%s%N)"
  host="$(hostname 2>/dev/null || echo unknown)"
  mkdir -p "$(dirname "$marker")"
  python3 -c '
import json, sys
slug, now, session, pid, host, path = sys.argv[1:7]
with open(path, "w") as f:
    json.dump(
        {
            "slug": slug,
            "broughtUpAt": now,
            "sessionId": session,
            "pid": int(pid),
            "host": host,
        },
        f,
    )
    f.write("\n")
' "$slug" "$now" "$session" "$pid" "$host" "$marker"
}

# marker_age_seconds MARKER_PATH
# Prints the marker's age in seconds, from its broughtUpAt field, as of now.
marker_age_seconds() {
  local marker="$1"
  local brought_up_at now_epoch brought_up_epoch
  brought_up_at="$(marker_field "$marker" broughtUpAt)"
  now_epoch="$(date -u +%s)"
  brought_up_epoch="$(date -u -d "$brought_up_at" +%s)"
  echo $(( now_epoch - brought_up_epoch ))
}
