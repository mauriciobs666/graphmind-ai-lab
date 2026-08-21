#!/usr/bin/env bash
# PreToolUse guard for the `teco` subagent (frontmatter `hooks:`, matcher
# `Agent|Task`). An `Agent` dispatch that omits `subagent_type` silently runs
# as `general-purpose` — no error, no warning — even when the brief says "you
# are <named-agent>": that delegate never loads the real agent's system
# prompt, tool restrictions, or PreToolUse hooks, and a later SendMessage
# resume inherits the wrong identity for the rest of the thread (verified
# live 2026-08-21, teco kaizen history). Prompt-level discipline already
# failed twice, so this hook makes the omission escalate to the human
# instead of landing silently.
#
# Behavior: subagent_type present and non-empty → exit 0 silently (normal
# flow). Missing/empty → permissionDecision "ask". Unparseable input →
# fail-open exit 0 (same contract as the shared guard cores; the prompt
# guardrail backstops). Standalone agent-owned script, same mechanics as the
# shared cores (jq→python3 extraction, ask-only, fail-open) — extract into a
# shared core only if a second orchestrator-shaped agent ever needs it.

set -uo pipefail

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

printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"teco guardrail: this Agent dispatch has no subagent_type — it would silently run as a general-purpose delegate with none of the named agent'"'"'s system prompt, tool restrictions, or hooks (verified live 2026-08-21), and a SendMessage resume cannot fix the identity afterwards. Approve only if a general-purpose delegate is genuinely intended; otherwise deny and have teco re-dispatch with an explicit subagent_type."}}\n'
exit 0
