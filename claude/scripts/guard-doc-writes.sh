#!/usr/bin/env bash
# guard-doc-writes.sh — shared PreToolUse core for the doc-scoped write guards.
#
# The doc-scoped agents (architect, analyst, data-scientist, teco, tico) each
# keep a thin wrapper in <agent>/hooks/ (wired via the agent's frontmatter
# `hooks:`, matcher `Write|Edit`) that execs this script with its parameters:
#
#   guard-doc-writes.sh '<glob>|<glob>...' '<escalation message template>'
#
#   $1  pipe-separated allowed-path globs; the /tmp scratchpad ('/tmp/*') is
#       always allowed and needn't be listed
#   $2  message shown to the human on escalation; the literal token __PATH__
#       is replaced with the (JSON-escaped) offending path. Keep templates
#       free of double quotes and backslashes — the message is spliced into
#       JSON verbatim.
#
# Behavior: any Write/Edit whose target does not match an allowed glob is
# escalated to the human for approval (PreToolUse permissionDecision "ask")
# instead of silently proceeding. Conforming writes pass straight through.
#
# Deliberately NOT covered: Bash. Mutating the tree via Bash would be a
# deliberate guardrail violation (prompt-guarded), whereas drifting into code
# edits via the editing tools is the realistic *accidental* failure mode this
# guard closes. See architect kaizen K-003 resolution (2026-07-08).
#
# Contract (same as devops/hooks/guard-destructive-ops.sh, verified 2026-07-02
# against code.claude.com/docs/en/hooks):
#   - stdin: JSON with .tool_input.file_path (matcher already restricts to
#     Write/Edit).
#   - stdout JSON on a hit:
#       {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#         "permissionDecision":"ask","permissionDecisionReason":"..."}}
#   - exit 0 always (the decision is carried in the JSON, not the exit code).
#
# No hard dependency on jq: extraction tries jq, then python3. Fail-open by
# design — if the path can't be extracted, the call proceeds and the
# prompt-level guardrail backstops.

set -uo pipefail
set -f # no filename expansion — the allowed globs must reach `case` literally

allowed_globs="${1:?usage: guard-doc-writes.sh '<globs>' '<message template>'}"
msg_template="${2:?usage: guard-doc-writes.sh '<globs>' '<message template>'}"

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

IFS='|'
for glob in $allowed_globs '/tmp/*'; do
  case "$path" in
    $glob) exit 0 ;;
  esac
done
unset IFS

# Escape the two JSON-hostile characters a path could plausibly contain.
esc_path="$(printf '%s' "$path" | sed 's/\\/\\\\/g; s/"/\\"/g')"
shopt -u patsub_replacement 2>/dev/null || true # keep '&' in paths literal
msg="${msg_template//__PATH__/$esc_path}"

printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$msg"
exit 0
