#!/usr/bin/env bash
# guard-destructive-ops.sh — shared core for the destructive-ops PreToolUse guards.
#
# Called by thin per-agent wrappers (devops, graph-dba, qa-engineer) wired via
# each agent's frontmatter `hooks:` (matcher `Bash`). Runs before every Bash
# tool call while the guarded agent is active and escalates destructive /
# shared-state operations to the human for approval (PreToolUse
# permissionDecision "ask") instead of letting the agent run them unattended.
# Non-matching commands pass straight through.
#
# Usage: guard-destructive-ops.sh <agent-name>
#   <agent-name> personalizes the escalation message shown to the human.
#
# Contract (verified 2026-07-02 against code.claude.com/docs/en/hooks):
#   - stdin: JSON with .tool_input.command (the frontmatter matcher already
#     restricts this hook to the Bash tool).
#   - stdout JSON on a hit:
#       {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#         "permissionDecision":"ask","permissionDecisionReason":"..."}}
#   - exit 0 always (the decision is carried in the JSON, not the exit code).
#
# No hard dependency on jq: extraction tries jq, then python3, then falls back
# to scanning the raw JSON payload. Match patterns use non-alphanumeric token
# boundaries so they work correctly on either a clean command string or the raw
# payload (where a token like `-v` is followed by `"`). Fail-open by design —
# if nothing parses, the call proceeds and the prompt-level guardrail backstops.

set -uo pipefail

agent="${1:-agent}"

input="$(cat)"

# Best-effort extract of the command; fall back to the raw payload as haystack.
haystack="$input"
if command -v jq >/dev/null 2>&1; then
  c="$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"
  [ -n "$c" ] && haystack="$c"
elif command -v python3 >/dev/null 2>&1; then
  c="$(printf '%s' "$input" | python3 -c 'import sys,json;
try: print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception: pass' 2>/dev/null || true)"
  [ -n "$c" ] && haystack="$c"
fi

norm="$(printf '%s' "$haystack" | tr '\n' ' ' | tr -s ' ')"

# Token boundary that also matches JSON punctuation (", }, \) and end-of-string.
B='([^[:alnum:]]|$)'

reason=""
if printf '%s' "$norm" | grep -qiE "docker[[:space:]]+(volume[[:space:]]+(rm|prune)|system[[:space:]]+prune)"; then
  reason="Docker volume/system prune or removal — destroys persisted data in named volumes"
elif printf '%s' "$norm" | grep -qiE "docker[[:space:]]+(container[[:space:]]+)?rm[[:space:]]+(-[[:alnum:]]*f|--force)"; then
  reason="force-removal of a running Docker container (docker rm -f) — may evict a service others use"
elif printf '%s' "$norm" | grep -qiE "docker[- ]compose[[:space:]].*down.*(-v${B}|--volumes)"; then
  reason="compose down -v/--volumes — removes named volumes and their data"
elif printf '%s' "$norm" | grep -qiE "(^|[^[:alnum:]])(FLUSHALL|FLUSHDB)${B}|GRAPH\.DELETE${B}"; then
  reason="flush/delete of a shared Redis/FalkorDB datastore — wipes data other components depend on"
fi

[ -z "$reason" ] && exit 0

msg="${agent} guardrail: ${reason}. This is a destructive/shared-state operation — approve only if you are sure of the blast radius. The agent should otherwise return to the caller for confirmation."

# Emit the PreToolUse decision. Escape the message for JSON by hand (no quotes/
# backslashes/newlines are introduced above, so only \" is a concern — none present).
printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$msg"
exit 0
