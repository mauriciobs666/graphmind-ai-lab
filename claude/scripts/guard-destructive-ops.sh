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
# Left-side counterpart: start-of-string or non-alnum. Used to anchor a basename
# match so it can't fire on an unrelated name that merely ends the same way
# (see the pipeline.sh branch below).
LB='(^|[^[:alnum:]])'

reason=""
if printf '%s' "$norm" | grep -qiE "docker[[:space:]]+(volume[[:space:]]+(rm|prune)|system[[:space:]]+prune)"; then
  reason="Docker volume/system prune or removal — destroys persisted data in named volumes"
elif printf '%s' "$norm" | grep -qiE "docker[[:space:]]+(container[[:space:]]+)?rm[[:space:]]+(-[[:alnum:]]*f|--force)"; then
  reason="force-removal of a running Docker container (docker rm -f) — may evict a service others use"
elif printf '%s' "$norm" | grep -qiE "docker[- ]compose[[:space:]].*down.*(-v${B}|--volumes)"; then
  reason="compose down -v/--volumes — removes named volumes and their data"
elif printf '%s' "$norm" | grep -qiE "(^|[^[:alnum:]])(FLUSHALL|FLUSHDB)${B}|GRAPH\.DELETE${B}"; then
  reason="flush/delete of a shared Redis/FalkorDB datastore — wipes data other components depend on"
elif printf '%s' "$norm" | grep -qiE "${LB}pipeline\.sh${B}" \
     && printf '%s' "$norm" | grep -qiE "${LB}--reset${B}"; then
  # Ad-hoc wrapper match (C-311, 2026-08-08; tightened 2026-08-08 per analyst
  # review docs/reviews/safety-net-guard-fixes.md): skills/joern-cpg/scripts/
  # pipeline.sh --reset runs `redis-cli ... GRAPH.DELETE` INSIDE the script, so
  # the literal string never appears in the Bash command text this guard
  # inspects — match the wrapper invocation itself instead.
  #
  # Deliberately anchored on the basename only (LB = start-of-string or
  # non-alnum immediately before "pipeline.sh"), NOT the full
  # skills/joern-cpg/scripts/ path: SKILL.md's own documented usage
  # (`scripts/pipeline.sh <source> ...`) is written cwd-relative, so a real
  # invocation may legitimately appear in the command text as
  # `scripts/pipeline.sh`, `./pipeline.sh`, or even bare `pipeline.sh`
  # depending on the caller's cwd — anchoring on the full path would silently
  # reopen C-311 for those forms. The basename boundary still rejects the
  # concrete false positive the review found (`mypipeline.sh --reset`, no
  # separator before "pipeline") without risking a false negative on any real
  # invocation shape (path-prefixed, bash/sh-prefixed, or bare). A prose/
  # argument mention (e.g. `grep ... pipeline.sh` or `echo '...pipeline.sh
  # --reset...'`) can still trip this, same as the pre-existing GRAPH.DELETE/
  # FLUSHALL branches already do on their own literal strings — accepted,
  # not a regression: over-asking is this guard's safe failure direction,
  # under-asking is the exact gap C-311 exists to close.
  #
  # Two INDEPENDENT greps ANDed, not one alternation requiring a left-to-right
  # order (Pass-2 review regression, 2026-08-08): the original single-regex
  # form had "--reset${B}" and "${LB}pipeline\.sh" sharing one separator
  # character when only a single space stood between them, so `--reset
  # pipeline.sh` (bare basename, flag before the name) silently stopped
  # matching. Splitting into two greps means each boundary consumes its own
  # separator, independent of where the other token sits in the string — so
  # match order no longer matters at all.
  #
  # This repo has exactly one such wrapper today (verified by grepping
  # skills/*/scripts/ for other destructive flags); if a second wrapper
  # appears, replace this one-off pattern with a documented wrapper-registry
  # convention rather than accreting more special cases here.
  reason="skills/joern-cpg/scripts/pipeline.sh --reset — wraps a GRAPH.DELETE the guard can't see in the script's own command text"
fi

[ -z "$reason" ] && exit 0

msg="${agent} guardrail: ${reason}. This is a destructive/shared-state operation — approve only if you are sure of the blast radius. The agent should otherwise return to the caller for confirmation."

# Emit the PreToolUse decision. Escape the message for JSON by hand (no quotes/
# backslashes/newlines are introduced above, so only \" is a concern — none present).
printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$msg"
exit 0
