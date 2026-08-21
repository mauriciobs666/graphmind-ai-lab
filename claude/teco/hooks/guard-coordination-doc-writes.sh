#!/usr/bin/env bash
# PreToolUse guard for the `teco` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). Teco coordinates and mostly doesn't implement — its Write/Edit
# are for its coordination/work-breakdown document (convention:
# docs/plans/<slug>-coordination.md, co-located with the architect's plan),
# and (2026-07-25) a genuinely trivial single-file no-brainer fix it judges
# safe to make directly instead of delegating. (Learning capture moved to the
# `kaizen_team` graph via `mcp__cypher__query` back on 2026-08-20 — it was
# never a Write/Edit path — and teco's frozen `kaizen/inbox.md` was itself
# deleted 2026-08-21, so no inbox allowance belongs in this glob any more.)
# Either of those two land silently (path matches an allowed glob); everything
# else — including a trivial fix, since source/config paths aren't in the
# allowlist — escalates here for a one-time human approval. Thin wrapper: the
# shared logic lives in claude/scripts/guard-doc-writes.sh (resolved through
# this file's real path, so it also works via the ~/.claude/agents/ symlink).
#
# Status-flip carve-out (2026-08-21, stakeholder-approved): before deferring
# to the shared core, this wrapper auto-ALLOWS one mechanically-verifiable
# edit shape — an Edit to a docs/**.md file whose old/new strings differ ONLY
# in the canonical `**Status:**` field, flipping it to `archived`. That is the
# milestone-close archival flip (root AGENTS.md lifecycle), which previously
# cost one delegated agent spawn per one-token edit because teco's allowlist
# reaches docs/plans/* only. The check masks the Status field on both strings
# and requires byte-equality of everything else; anything wider than the
# one-token flip still escalates exactly as before.

set -uo pipefail

input="$(cat)"

# The status field is '**Status:** <token and free text>' up to the ' · '
# separator (middle dot) or end of line, per root AGENTS.md's header grammar.
verdict="NO"
if command -v python3 >/dev/null 2>&1; then
  verdict="$(printf '%s' "$input" | python3 -c '
import sys, json, re
out = "NO"
try:
    d = json.load(sys.stdin)
    ti = d.get("tool_input") or {}
    old = ti.get("old_string") or ""
    new = ti.get("new_string") or ""
    path = ti.get("file_path") or ""
    ok_path = path.endswith(".md") and ("/docs/" in path or path.startswith("docs/"))
    pat = re.compile(r"(\*\*Status:\*\*\s*)([^·\n]*)")
    mo = pat.findall(old)
    mn = pat.findall(new)
    if (
        d.get("tool_name") == "Edit"
        and ok_path
        and len(mo) == 1
        and len(mn) == 1
        and mn[0][1].strip() == "archived"
        and mo[0][1].strip() != "archived"
        and pat.sub(r"\1@", old) == pat.sub(r"\1@", new)
    ):
        out = "FLIP"
except Exception:
    out = "NO"
print(out)
' 2>/dev/null)" || verdict="NO"
fi

if [ "$verdict" = "FLIP" ]; then
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"milestone-close archival flip — the edit changes only the canonical Status: field to archived on a docs/ markdown file (mechanically verified by the guard; stakeholder-approved carve-out, 2026-08-21)"}}\n'
  exit 0
fi

printf '%s' "$input" | "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/plans/*|*/docs/plans/*' \
  "teco guardrail: Write/Edit targets '__PATH__', which is outside a docs/plans/ directory, a mechanically-verified Status-archived flip, or the /tmp scratchpad. Teco coordinates and normally delegates implementation, but may make a genuinely trivial single-file no-brainer fix directly (a typo, an obvious one-liner, a config value, a rename) instead of spinning up a specialist. Approve only if this is that kind of trivial, obviously-safe fix (or a coordination artifact); if it needs design judgment, spans multiple files, or touches anything security/data-model/test-critical, deny and let it route to a delegated specialist instead."
exit $?
