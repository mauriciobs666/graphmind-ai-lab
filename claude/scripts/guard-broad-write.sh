#!/usr/bin/env bash
# guard-broad-write.sh — shared PreToolUse core for a "broad implementer"
# write guard: the INVERSE shape from guard-doc-writes.sh.
#
# guard-doc-writes.sh is an ALLOW-LIST: escalate everything except a small
# set of paths that ARE the whole remit (a doc-scoped specialist like
# architect/analyst/tico/cobb/qa-engineer, whose Write/Edit legitimately
# touches nothing else). This core is a DENY-LIST: allow everything except a
# small set of paths KNOWN to belong to a DIFFERENT specialist's documented
# deliverable-path convention (or a genuinely unresolved team-governance
# ambiguity) — for an agent whose own remit is "the whole codebase, this
# task" and has no single folder/kind to allowlist. See
# claude/docs/plans/agent-permission-friction.md §6.
#
#   guard-broad-write.sh '<glob>|<glob>...' '<escalation message template>'
#
#   $1  pipe-separated globs that are OUT of this agent's remit — a match
#       escalates. No implicit /tmp/* handling needed here (nothing to
#       protect a scratchpad from for a broad-remit agent).
#   $2  message shown to the human on escalation; __PATH__ is replaced with
#       the (JSON-escaped) offending path. Keep templates free of double
#       quotes and backslashes.
#
# Behavior: a Write/Edit whose target MATCHES a listed glob escalates
# (PreToolUse permissionDecision "ask"); everything else — the overwhelming
# common case, source/test/any-other-in-task-file work — is explicitly
# APPROVED (permissionDecision "allow"), skipping the ambient permission-mode
# prompt regardless of which mode governs the session (see root-cause
# finding, agent-permission-friction.md §1).
#
# A SEPARATE core from guard-doc-writes.sh rather than a third mode bolted
# onto it: same precedent as security-expert's two independent hook cores
# (claude/AGENTS.md "Hook machinery") — a genuinely different truth table
# earns its own script.
#
# Same stdin/stdout contract, jq->python3 extraction, and fail-open behavior
# as guard-doc-writes.sh (contract verified 2026-08-21 against
# code.claude.com/docs/en/hooks).
#
# Tracing (debug only, opt-in): if $GUARD_BROAD_WRITE_TRACE names a file, every
# invocation appends one line per checkpoint (timestamp, wrapper args, raw
# stdin, extracted path, matched glob if any, and the final decision emitted)
# to that file. Unset/empty (the default) → zero tracing overhead, nothing
# written. Never write elsewhere and never fail the guard's own decision if
# the trace write fails (e.g. the directory doesn't exist) — tracing is
# strictly observational. NOTE (from the 2026-08-24 root-cause finding in
# cobb/kaizen/history.md, promoted to skills/agent-standards/claude-code.md):
# a subagent-delegated write can still hit a human confirmation prompt AFTER
# this guard emits "allow" — the auto-mode classifier reviews
# Task/Agent-delegated tool calls independently of PreToolUse hook output.
# A trace showing "decision: allow" is proof the guard did its job; it is
# not proof the human won't still be asked.

trace_file="${GUARD_BROAD_WRITE_TRACE:-}"

_trace() {
  [ -n "$trace_file" ] || return 0
  printf '[%s] %s\n' "$(date -Iseconds 2>/dev/null || date)" "$*" >>"$trace_file" 2>/dev/null || true
}

set -uo pipefail
set -f

denied_globs="${1:?usage: guard-broad-write.sh '<globs>' '<message template>'}"
msg_template="${2:?usage: guard-broad-write.sh '<globs>' '<message template>'}"

_trace "invoked; denied_globs=${denied_globs}"

input="$(cat)"
_trace "stdin: ${input}"

path=""
if command -v jq >/dev/null 2>&1; then
  path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
  path="$(printf '%s' "$input" | python3 -c 'import sys,json;
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: pass' 2>/dev/null || true)"
fi
_trace "extracted path: ${path:-<empty>}"

# Fail-open: no extractable path, let it through (prompt guardrail backstops).
if [ -z "$path" ]; then
  _trace "decision: allow (fail-open — no path extracted)"
  exit 0
fi

IFS='|'
for glob in $denied_globs; do
  case "$path" in
    $glob)
      esc_path="$(printf '%s' "$path" | sed 's/\\/\\\\/g; s/"/\\"/g')"
      shopt -u patsub_replacement 2>/dev/null || true
      msg="${msg_template//__PATH__/$esc_path}"
      _trace "matched glob: ${glob}; decision: ask; message: ${msg}"
      printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$msg"
      exit 0
      ;;
  esac
done
unset IFS

_trace "decision: allow (in-remit implementer write — no glob matched)"
printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"in-remit implementer write — auto-approved by guard"}}\n'
exit 0
