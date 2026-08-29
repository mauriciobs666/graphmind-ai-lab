#!/usr/bin/env bash
# guard-agent-dispatch.sh — shared PreToolUse core for the Agent-dispatch
# guards (frontmatter `hooks:`, matcher `Agent|Task`).
#
# An `Agent` dispatch that omits `subagent_type` silently runs as
# `general-purpose` — no error, no warning — even when the brief says "you
# are <named-agent>": that delegate never loads the real agent's system
# prompt, tool restrictions, or PreToolUse hooks, and a later `SendMessage`
# resume inherits the wrong identity for the rest of the thread (verified
# live 2026-08-21, teco kaizen history).
#
# Extracted 2026-08-29 from `teco`'s original standalone script into this
# shared core, when `tico` gained the same `Agent`+`SendMessage` dispatch
# shape (`claude/docs/requirements/tico-specialist-collaboration.md`) and
# so the same hazard. `teco`'s own script header had anticipated exactly
# this: "extract into a shared core only if a second orchestrator-shaped
# agent ever needs it" — the actual trigger is the `Agent`+`SendMessage`
# combination, not orchestrator status: `tico` stays single-topic per FR-9,
# never a second orchestrator.
#
# Each caller keeps a thin wrapper in <agent>/hooks/ that execs this script
# with its own agent name:
#
#   guard-agent-dispatch.sh '<agent-name>'
#
# Behavior: subagent_type present and non-empty → exit 0 silently (normal
# flow). Missing/empty → permissionDecision "ask". Unparseable input →
# fail-open exit 0 (same contract as the other shared cores; the prompt
# guardrail backstops).

set -uo pipefail

agent_name="${1:?usage: guard-agent-dispatch.sh '<agent-name>'}"

input="$(cat)"

parsed=""
sat=""
if command -v jq >/dev/null 2>&1; then
  parsed="$(printf '%s' "$input" | jq -r 'if (.tool_input | type) == "object" then "yes" else "no" end' 2>/dev/null || true)"
  sat="$(printf '%s' "$input" | jq -r '.tool_input.subagent_type // empty' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
  parsed="$(printf '%s' "$input" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print("yes" if isinstance(d.get("tool_input"),dict) else "no")
except Exception: pass' 2>/dev/null || true)"
  sat="$(printf '%s' "$input" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print(d.get("tool_input",{}).get("subagent_type",""))
except Exception: pass' 2>/dev/null || true)"
fi

# Fail-open: input not parseable as a tool call — let it through.
[ "$parsed" != "yes" ] && exit 0

# Explicit subagent_type present — normal flow.
[ -n "$sat" ] && exit 0

printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s guardrail: this Agent dispatch has no subagent_type — it would silently run as a general-purpose delegate with none of the named agent'"'"'s system prompt, tool restrictions, or hooks (verified live 2026-08-21), and a SendMessage resume cannot fix the identity afterwards. Approve only if a general-purpose delegate is genuinely intended; otherwise deny and have %s re-dispatch with an explicit subagent_type."}}\n' "$agent_name" "$agent_name"
exit 0
