#!/usr/bin/env bash
# PreToolUse guard for the `teco` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). Teco coordinates, never implements — its Write/Edit exist for
# two purposes: authoring/revising its coordination/work-breakdown document
# (convention: docs/plans/<slug>-coordination.md, co-located with the
# architect's plan), and appending to its own learnings inbox
# (kaizen/inbox.md — the learning-capture loop). Thin wrapper: the shared
# logic lives in claude/scripts/guard-doc-writes.sh (resolved through this
# file's real path, so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/plans/*|*/docs/plans/*|teco/kaizen/inbox.md|*/teco/kaizen/inbox.md' \
  "teco guardrail: Write/Edit targets '__PATH__', which is outside a docs/plans/ directory, the agent's own kaizen/inbox.md, or the /tmp scratchpad. Teco coordinates — its Write/Edit are for the coordination/work-breakdown document and its learnings inbox only, never source, tests, or config. Approve only if this is genuinely a coordination artifact; otherwise the work belongs to a delegated specialist."
