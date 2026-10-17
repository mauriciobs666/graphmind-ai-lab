#!/usr/bin/env bash
# Shared PreToolUse core for a repository-bounded Write|Edit guard (frontmatter
# `hooks:`, matcher `Write|Edit`), thin-wrapped per agent: `teco`
# (teco/hooks/guard-repo-writes.sh, since 2026-10-14) and `tico`
# (tico/hooks/guard-tico-repo-writes.sh, since 2026-10-17). Not doc-scoped:
# the agent may Write/Edit any file of any kind inside the git repository its
# session was started in; a write anywhere else escalates for a one-time human
# approval. An orchestrator's frontmatter hooks also gate its delegates' calls,
# so an in-repo allow here is what spares a delegate spurious asks (its own
# guard still applies; the strictest decision wins) — and why the outside case
# is `ask`, never `deny`. What the agent may *author* stays prompt-held.
#
# Usage: guard-repo-writes.sh <agent-name>   (hook input JSON on stdin)
#
# Decisions (stdout JSON, exit 0 always — the decision is in the JSON):
#   allow — the resolved path is inside the repo root, or under /tmp (the
#           session scratchpad).
#   ask   — the resolved path is outside both, or inside the repo's .git/
#           directory (git internals: hooks there execute code, config there
#           changes every git command — not a "file of the project").
#   (none)— fail-open: no file_path could be extracted (needs jq or python3);
#           the ambient permission flow decides.
#
# Repo root: `git rev-parse --show-toplevel` from $CLAUDE_PROJECT_DIR (the dir
# the session was started in; hook input `cwd` as fallback); if that dir is not
# in a git repo, the dir itself is the boundary. A relative file_path resolves
# against the hook input's `cwd`. Both sides go through `realpath -m`, so `..`
# segments and symlinks are resolved before comparing: a symlink inside the
# repo that points outside it is treated as outside.
#
# The agents are user-scoped (deployed via ~/.claude/agents/<name> symlinks), so
# "the repo" is whichever project the agent was launched in, not graphmind-ai-lab.

set -uo pipefail

agent="${1:-agent}"
input="$(cat)"

field() { # $1 = jq path, $2 = python expression over d
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$input" | jq -r "$1 // empty" 2>/dev/null || true
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s' "$input" | python3 -c "import sys,json
try:
    d=json.load(sys.stdin); print($2 or '')
except Exception: pass" 2>/dev/null || true
  fi
}

path="$(field '.tool_input.file_path' '(d.get("tool_input") or {}).get("file_path")')"
cwd="$(field '.cwd' 'd.get("cwd")')"

# Fail-open: no extractable path, let it through (prompt guardrail backstops).
[ -z "$path" ] && exit 0

start="${CLAUDE_PROJECT_DIR:-${cwd:-$PWD}}"
root="$(git -C "$start" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$start")"
root="$(realpath -m -- "$root")"

case "$path" in
  /*) abs="$path" ;;
  *) abs="${cwd:-$root}/$path" ;;
esac
abs="$(realpath -m -- "$abs")"

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
decide() { # $1 = allow|ask, $2 = reason
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"%s","permissionDecisionReason":"%s"}}\n' \
    "$1" "$(json_escape "$2")"
  exit 0
}

case "$abs" in
  "$root"/.git|"$root"/.git/*)
    decide ask "$agent guardrail: Write/Edit targets '$abs', inside the repository's .git/ directory (git internals, not a project file). Approve only if you explicitly asked $agent to change git configuration or hooks." ;;
  "$root"|"$root"/*)
    decide allow "in-repo write — inside the repository $agent was started in ($root), auto-approved by guard, path: $abs" ;;
  /tmp/*)
    decide allow "scratch write under /tmp — auto-approved by guard, path: $abs" ;;
esac

decide ask "$agent guardrail: Write/Edit targets '$abs', which is outside the repository $agent was started in ($root). ${agent^} may change any file inside that repository; a write outside it needs your explicit approval. Approve only if you asked for this specific out-of-repo change."
