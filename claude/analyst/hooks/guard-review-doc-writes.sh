#!/usr/bin/env bash
# PreToolUse guard for the `analyst` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). The analyst is review-only — it never edits source, tests,
# config, or the artifact under review; its Write/Edit exist for one purpose:
# authoring/revising review documents under docs/reviews/. Thin wrapper: the
# shared logic lives in claude/scripts/guard-doc-writes.sh (resolved through
# this file's real path, so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/reviews/*|*/docs/reviews/*' \
  "analyst guardrail: Write/Edit targets '__PATH__', which is outside a docs/reviews/ directory (or /tmp scratchpad). The analyst is review-only — its Write/Edit are for review documents only, never source, tests, config, or the artifact under review. Approve only if this is genuinely a review artifact; otherwise the agent should put the change in its findings for the artifact's owner."
