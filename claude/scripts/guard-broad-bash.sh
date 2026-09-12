#!/usr/bin/env bash
# guard-broad-bash.sh — shared PreToolUse core for a "broad implementer" Bash
# guard: the Bash-tool counterpart to guard-broad-write.sh, for an agent whose
# remit is "the whole codebase, this task" and needs to run ordinary Bash
# (pytest, git, npm, etc.) constantly as part of normal work.
#
# Root cause this closes: `permissionMode: acceptEdits` only auto-approves
# in-working-directory file EDITS — it has no effect on Bash. An implementer
# agent with only a Write|Edit guard (guard-broad-write.sh) therefore still
# hits the plain confirm-before-Bash prompt on every single command, moving
# the friction from Write/Edit onto Bash rather than removing it (observed
# 2026-09-11: switching the session to acceptEdits stopped the Write/Edit
# prompts but "any command asks" — every Bash call still confirmed).
#
# Deliberately REUSES claude/scripts/guard-destructive-ops.sh's own pattern
# matching rather than duplicating it: this script pipes the same stdin JSON
# through that core (same <agent-name> arg) and inspects its output instead
# of re-implementing the destructive-command patterns. That keeps the
# destructive-pattern catalog in exactly one place — this script's only new
# behavior is emitting an explicit "allow" when the destructive-ops core
# stays silent, instead of leaving a non-destructive command to fall through
# to whatever ambient permission mode governs (unreliable — see
# claude/docs/plans/agent-permission-friction.md §1.3, the same reasoning
# that motivated guard-broad-write.sh's explicit allow on the Write/Edit
# side).
#
#   guard-broad-bash.sh <agent-name>
#
# Same stdin/stdout contract as the other shared cores (verified 2026-08-21
# against code.claude.com/docs/en/hooks): stdin is the PreToolUse JSON
# payload (matcher already restricts this hook to Bash); stdout on a
# destructive match is guard-destructive-ops.sh's own "ask" JSON, relayed
# unchanged; stdout otherwise is this script's own explicit "allow" JSON.
# Exit 0 always — the decision lives in the JSON, not the exit code.
#
# NOTE (same caveat as guard-broad-write.sh, from the 2026-08-24 root-cause
# finding, promoted to skills/agent-standards/claude-code.md): a
# subagent-delegated (Task/Agent) Bash call can still hit a human
# confirmation prompt AFTER this guard emits "allow" — the auto-mode
# classifier reviews Task-delegated tool calls independently of PreToolUse
# hook output. This guard closes the gap for a top-level interactive session
# (or any session not governed by auto mode); it does not, and per the
# 2026-08-24 finding cannot, close the Task/auto-mode gap — that is a
# settled, permanent, team-wide limitation, not something this script failed
# to do.
#
# Tracing (debug only, opt-in): if $GUARD_BROAD_BASH_TRACE names a file,
# every invocation appends one line per checkpoint (timestamp, the raw
# stdin, and the final decision emitted) to that file. Unset/empty (the
# default) → zero tracing overhead, nothing written. Never write elsewhere
# and never fail the guard's own decision if the trace write fails.

set -uo pipefail

agent="${1:-agent}"
core_dir="$(dirname "$(readlink -f "$0")")"

trace_file="${GUARD_BROAD_BASH_TRACE:-}"
_trace() {
  [ -n "$trace_file" ] || return 0
  printf '[%s] %s\n' "$(date -Iseconds 2>/dev/null || date)" "$*" >>"$trace_file" 2>/dev/null || true
}

input="$(cat)"
_trace "invoked for agent=${agent}; stdin: ${input}"

result="$(printf '%s' "$input" | "${core_dir}/guard-destructive-ops.sh" "$agent")"

if [ -n "$result" ]; then
  _trace "destructive-ops core matched; relaying its ask decision: ${result}"
  printf '%s\n' "$result"
  exit 0
fi

_trace "decision: allow (in-remit implementer Bash call — no destructive pattern matched)"
printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"in-remit implementer Bash call — auto-approved by guard"}}\n'
exit 0
