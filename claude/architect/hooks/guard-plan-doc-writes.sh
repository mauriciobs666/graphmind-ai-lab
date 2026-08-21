#!/usr/bin/env bash
# PreToolUse guard for the `architect` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). The architect is read-only on code — its Write/Edit exist for
# one purpose: authoring/revising plan docs under docs/plans/. See architect
# kaizen K-003 (2026-07-08). (Learning capture moved to the `kaizen_team`
# graph via `mcp__cypher__query` back on 2026-08-20 — it was never a
# Write/Edit path — and architect's frozen `kaizen/inbox.md` was itself
# deleted 2026-08-21, so no inbox allowance belongs in this glob any more.)
# Thin wrapper: the shared logic lives in claude/scripts/guard-doc-writes.sh
# (resolved through this file's real path, so it also works when invoked via
# the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/plans/*|*/docs/plans/*' \
  "architect guardrail: Write/Edit targets '__PATH__', which is outside a docs/plans/ directory or the /tmp scratchpad. The architect is read-only on code — its Write/Edit are for plan documents only. Approve only if this is genuinely a plan/design artifact; otherwise the agent should put the change in the plan for an implementer."
