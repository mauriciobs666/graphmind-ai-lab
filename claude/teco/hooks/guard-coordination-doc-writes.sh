#!/usr/bin/env bash
# PreToolUse guard for the `teco` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). Teco coordinates, never implements — its Write/Edit exist for
# one purpose: authoring/revising its coordination/work-breakdown document
# (convention: docs/plans/<slug>-coordination.md, co-located with the
# architect's plan). Thin wrapper: the shared logic lives in
# claude/scripts/guard-doc-writes.sh (resolved through this file's real path,
# so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/plans/*|*/docs/plans/*' \
  "teco guardrail: Write/Edit targets '__PATH__', which is outside a docs/plans/ directory (or /tmp scratchpad). Teco coordinates — its Write/Edit are for the coordination/work-breakdown document only, never source, tests, or config. Approve only if this is genuinely a coordination artifact; otherwise the work belongs to a delegated specialist."
