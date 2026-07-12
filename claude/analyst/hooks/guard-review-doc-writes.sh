#!/usr/bin/env bash
# PreToolUse guard for the `analyst` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). The analyst is review-only — it never edits source, tests,
# config, or the artifact under review; its Write/Edit exist for two purposes:
# authoring/revising review documents under docs/reviews/, and appending to its
# own learnings inbox (kaizen/inbox.md — the learning-capture loop). Thin
# wrapper: the shared logic lives in claude/scripts/guard-doc-writes.sh
# (resolved through this file's real path, so it also works via the
# ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/reviews/*|*/docs/reviews/*|analyst/kaizen/inbox.md|*/analyst/kaizen/inbox.md' \
  "analyst guardrail: Write/Edit targets '__PATH__', which is outside a docs/reviews/ directory, the agent's own kaizen/inbox.md, or the /tmp scratchpad. The analyst is review-only — its Write/Edit are for review documents and its learnings inbox only, never source, tests, config, or the artifact under review. Approve only if this is genuinely a review artifact; otherwise the agent should put the change in its findings for the artifact's owner."
