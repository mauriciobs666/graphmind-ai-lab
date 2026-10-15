#!/usr/bin/env bash
# PreToolUse guard for the `teco` agent (frontmatter `hooks:`, matcher
# `Write|Edit`). Repository-bounded, not doc-scoped: teco may Write/Edit any
# file of any kind inside the git repository its session was started in; a
# write anywhere else escalates for a one-time human approval. Replaced the
# docs/plans/-only allowlist (guard-coordination-doc-writes.sh, and with it the
# Status->archived flip carve-out, which an in-repo allow now subsumes) on
# 2026-10-14 — history in claude/teco/kaizen/history.md.
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
# The agent is user-scoped (deployed via ~/.claude/agents/teco symlink), so
# "the repo" is whichever project teco was launched in, not graphmind-ai-lab.

set -uo pipefail

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
    decide ask "teco guardrail: Write/Edit targets '$abs', inside the repository's .git/ directory (git internals, not a project file). Approve only if you explicitly asked teco to change git configuration or hooks." ;;
  "$root"|"$root"/*)
    decide allow "in-repo write — inside the repository teco was started in ($root), auto-approved by guard, path: $abs" ;;
  /tmp/*)
    decide allow "scratch write under /tmp — auto-approved by guard, path: $abs" ;;
esac

decide ask "teco guardrail: Write/Edit targets '$abs', which is outside the repository teco was started in ($root). Teco may change any file inside that repository; a write outside it needs your explicit approval. Approve only if you asked for this specific out-of-repo change."
