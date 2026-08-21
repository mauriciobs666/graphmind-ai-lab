#!/usr/bin/env bash
# guard-doc-writes.sh — shared PreToolUse core for the doc-scoped write guards.
#
# The doc-scoped agents (architect, analyst, data-scientist, teco, tico,
# cobb, qa-engineer, security-expert's review guard) each keep a thin wrapper
# in <agent>/hooks/ (wired via the agent's frontmatter `hooks:`, matcher
# `Write|Edit`) that execs this script with its parameters:
#
#   guard-doc-writes.sh '<glob>|<glob>...' '<escalation message template>' [<on_mismatch>]
#
#   $1  pipe-separated allowed-path globs; the /tmp scratchpad ('/tmp/*') is
#       always allowed and needn't be listed
#   $2  message shown to the human on escalation (only used when $3 is "ask",
#       the default); the literal token __PATH__ is replaced with the
#       (JSON-escaped) offending path. Keep templates free of double quotes
#       and backslashes — the message is spliced into JSON verbatim.
#   $3  optional, "ask" (default, omit for today's behavior) or "pass".
#       "ask": a non-matching write escalates to the human (unchanged).
#       "pass": a non-matching write silently falls through instead — for an
#       agent whose remit is genuinely wider than its doc-scoped allowlist
#       (e.g. qa-engineer, which also authors source/test files as part of
#       execution; escalating those would be new friction, not a fix).
#
# Behavior: a Write/Edit whose target MATCHES an allowed glob is explicitly
# APPROVED (PreToolUse permissionDecision "allow") — changed 2026-08-21
# (agent-permission-friction FR-1/FR-2/FR-3). Previously this was a silent
# `exit 0`, which left an in-remit write's fate to whatever ambient
# permission mode governed the session — the fresh evidence showed that is
# NOT reliable (see claude/docs/plans/agent-permission-friction.md §1: a
# subagent's frontmatter permissionMode is silently ignored/overridden by
# the parent session's mode in documented cases, including the Pro/Max/Team
# default "auto" mode). An explicit hook "allow" skips the permission prompt
# unconditionally per code.claude.com/docs/en/hooks, independent of ambient
# mode — that's what actually closes the gap. A non-matching write still
# escalates to "ask" by default ($3 unset) — unchanged, and still what AC-4
# relies on.
#
# Deliberately NOT covered: Bash. Mutating the tree via Bash would be a
# deliberate guardrail violation (prompt-guarded), whereas drifting into code
# edits via the editing tools is the realistic *accidental* failure mode this
# guard closes. See architect kaizen K-003 resolution (2026-07-08).
#
# Contract (verified 2026-08-21 against code.claude.com/docs/en/hooks):
#   - stdin: JSON with .tool_input.file_path (matcher already restricts to
#     Write/Edit).
#   - stdout JSON on a match:
#       {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#         "permissionDecision":"allow","permissionDecisionReason":"..."}}
#   - stdout JSON on a mismatch (on_mismatch="ask", default):
#       {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#         "permissionDecision":"ask","permissionDecisionReason":"..."}}
#   - on_mismatch="pass": no output, exit 0 (normal permission flow decides).
#   - exit 0 always (the decision is carried in the JSON, not the exit code).
#
# No hard dependency on jq: extraction tries jq, then python3. Fail-open by
# design — if the path can't be extracted, the call proceeds and the
# prompt-level guardrail backstops.

set -uo pipefail
set -f # no filename expansion — the allowed globs must reach `case` literally

allowed_globs="${1:?usage: guard-doc-writes.sh '<globs>' '<message template>' [ask|pass]}"
msg_template="${2:?usage: guard-doc-writes.sh '<globs>' '<message template>' [ask|pass]}"
on_mismatch="${3:-ask}"

input="$(cat)"

path=""
if command -v jq >/dev/null 2>&1; then
  path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
  path="$(printf '%s' "$input" | python3 -c 'import sys,json;
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: pass' 2>/dev/null || true)"
fi

# Fail-open: no extractable path, let it through (prompt guardrail backstops).
[ -z "$path" ] && exit 0

esc_path=""
json_escape_path() {
  esc_path="$(printf '%s' "$path" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  shopt -u patsub_replacement 2>/dev/null || true # keep '&' in paths literal
}

IFS='|'
for glob in $allowed_globs '/tmp/*'; do
  case "$path" in
    $glob)
      json_escape_path
      printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"in-remit write (matches an allowed path) — auto-approved by guard, path: %s"}}\n' "$esc_path"
      exit 0
      ;;
  esac
done
unset IFS

if [ "$on_mismatch" = "pass" ]; then
  exit 0
fi

json_escape_path
msg="${msg_template//__PATH__/$esc_path}"

printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$msg"
exit 0
