#!/usr/bin/env bash
# PreToolUse guard for the `teco` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). Teco coordinates and mostly doesn't implement — its Write/Edit
# are for its coordination/work-breakdown document (convention:
# docs/plans/<slug>-coordination.md, co-located with the architect's plan),
# its own learnings inbox (kaizen/inbox.md — the learning-capture loop), and
# (2026-07-25) a genuinely trivial single-file no-brainer fix it judges safe
# to make directly instead of delegating. Any of those three land silently
# (path matches an allowed glob); everything else — including a trivial fix,
# since source/config paths aren't in the allowlist — escalates here for a
# one-time human approval. Thin wrapper: the shared logic lives in
# claude/scripts/guard-doc-writes.sh (resolved through this file's real path,
# so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/plans/*|*/docs/plans/*|teco/kaizen/inbox.md|*/teco/kaizen/inbox.md' \
  "teco guardrail: Write/Edit targets '__PATH__', which is outside a docs/plans/ directory, the agent's own kaizen/inbox.md, or the /tmp scratchpad. Teco coordinates and normally delegates implementation, but may make a genuinely trivial single-file no-brainer fix directly (a typo, an obvious one-liner, a config value, a rename) instead of spinning up a specialist. Approve only if this is that kind of trivial, obviously-safe fix (or a coordination artifact); if it needs design judgment, spans multiple files, or touches anything security/data-model/test-critical, deny and let it route to a delegated specialist instead."
