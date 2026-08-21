#!/usr/bin/env bash
# PreToolUse guard for the `tico` agent (frontmatter `hooks:`, matcher
# `Write|Edit` — fires in main-session mode too). Tico owns two doc kinds —
# its Write/Edit exist for: authoring/advancing the feature requirements
# document under docs/requirements/, and authoring/maintaining user manuals
# under docs/manuals/. (Learning capture moved to the `kaizen_team` graph via
# `mcp__cypher__query` back on 2026-08-20 — it was never a Write/Edit path —
# and tico's frozen `kaizen/inbox.md` was itself deleted 2026-08-21, so no
# inbox allowance belongs in this glob any more.) Thin wrapper: the shared
# logic lives in claude/scripts/guard-doc-writes.sh (resolved through this
# file's real path, so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/requirements/*|*/docs/requirements/*|docs/manuals/*|*/docs/manuals/*' \
  "tico guardrail: Write/Edit targets '__PATH__', which is outside a docs/requirements/ or docs/manuals/ directory or the /tmp scratchpad. Tico owns requirements documents and user manuals only — no source, tests, config, or design docs. Approve only if this is genuinely one of those artifacts; otherwise the need belongs in the requirements doc (or a downstream agent's deliverable), not a tico write."
