#!/usr/bin/env bash
# guard-plan-doc-writes.sh
#
# PreToolUse guard for the `architect` subagent (wired via its frontmatter
# `hooks:`, matcher `Write|Edit`). The architect's contract is read-only on
# code: its Write/Edit exist for one purpose — authoring/revising plan docs.
# This hook enforces that contract in the harness: any Write/Edit whose target
# is NOT under a `docs/plans/` directory (or the session scratchpad in /tmp)
# is escalated to the human for approval (PreToolUse permissionDecision "ask")
# instead of silently proceeding. Conforming writes pass straight through.
#
# Deliberately NOT covered: Bash. Mutating the tree via Bash would be a
# deliberate guardrail violation (prompt-guarded), whereas drifting into code
# edits via the editing tools is the realistic *accidental* failure mode this
# hook closes. See architect kaizen K-003 resolution (2026-07-08).
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

# Allowed targets: any docs/plans/ directory (repo-root or per-component), or
# the session scratchpad / temp space.
case "$path" in
  docs/plans/*|*/docs/plans/*|/tmp/*)
    exit 0
    ;;
esac

# Escape the two JSON-hostile characters a path could plausibly contain.
esc_path="$(printf '%s' "$path" | sed 's/\\/\\\\/g; s/"/\\"/g')"
msg="architect guardrail: Write/Edit targets '${esc_path}', which is outside a docs/plans/ directory (or /tmp scratchpad). The architect is read-only on code — its Write/Edit are for plan documents only. Approve only if this is genuinely a plan/design artifact; otherwise the agent should put the change in the plan for an implementer."

printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$msg"
exit 0
