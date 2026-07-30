#!/usr/bin/env bash
# PreToolUse guard for the `tico` agent (frontmatter `hooks:`, matcher
# `Write|Edit` — fires in main-session mode too). Tico owns two doc kinds plus
# its own inbox — its Write/Edit exist for: authoring/advancing the feature
# requirements document under docs/requirements/, authoring/maintaining user
# manuals under docs/manuals/, and appending to its own learnings inbox
# (kaizen/inbox.md — the learning-capture loop). Thin wrapper: the shared
# logic lives in claude/scripts/guard-doc-writes.sh (resolved through this
# file's real path, so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/requirements/*|*/docs/requirements/*|docs/manuals/*|*/docs/manuals/*|tico/kaizen/inbox.md|*/tico/kaizen/inbox.md' \
  "tico guardrail: Write/Edit targets '__PATH__', which is outside a docs/requirements/ or docs/manuals/ directory, the agent's own kaizen/inbox.md, or the /tmp scratchpad. Tico owns requirements documents and user manuals only — no source, tests, config, or design docs; its learnings inbox is the one other allowed target. Approve only if this is genuinely one of those artifacts; otherwise the need belongs in the requirements doc (or a downstream agent's deliverable), not a tico write."
