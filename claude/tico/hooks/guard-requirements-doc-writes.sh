#!/usr/bin/env bash
# PreToolUse guard for the `tico` agent (frontmatter `hooks:`, matcher
# `Write|Edit` — fires in main-session mode too). Tico owns requirements only —
# its Write/Edit exist for two purposes: authoring/advancing the feature
# requirements document under docs/requirements/, and appending to its own
# learnings inbox (kaizen/inbox.md — the learning-capture loop). Thin wrapper:
# the shared logic lives in claude/scripts/guard-doc-writes.sh (resolved
# through this file's real path, so it also works via the ~/.claude/agents/
# symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/requirements/*|*/docs/requirements/*|tico/kaizen/inbox.md|*/tico/kaizen/inbox.md' \
  "tico guardrail: Write/Edit targets '__PATH__', which is outside a docs/requirements/ directory, the agent's own kaizen/inbox.md, or the /tmp scratchpad. Tico owns requirements documents only — no source, tests, config, or design docs; its learnings inbox is the one other allowed target. Approve only if this is genuinely a requirements artifact; otherwise the need belongs in the requirements doc for a downstream agent."
