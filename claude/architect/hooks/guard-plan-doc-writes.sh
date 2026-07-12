#!/usr/bin/env bash
# PreToolUse guard for the `architect` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). The architect is read-only on code — its Write/Edit exist for
# two purposes: authoring/revising plan docs under docs/plans/, and appending
# to its own learnings inbox (kaizen/inbox.md — the learning-capture loop). See
# architect kaizen K-003 (2026-07-08). Thin wrapper: the shared logic lives in
# claude/scripts/guard-doc-writes.sh (resolved through this file's real path,
# so it also works when invoked via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/plans/*|*/docs/plans/*|architect/kaizen/inbox.md|*/architect/kaizen/inbox.md' \
  "architect guardrail: Write/Edit targets '__PATH__', which is outside a docs/plans/ directory, the agent's own kaizen/inbox.md, or the /tmp scratchpad. The architect is read-only on code — its Write/Edit are for plan documents and its learnings inbox only. Approve only if this is genuinely a plan/design artifact; otherwise the agent should put the change in the plan for an implementer."
