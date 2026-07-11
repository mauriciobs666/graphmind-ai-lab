#!/usr/bin/env bash
# PreToolUse guard for the `data-scientist` subagent (frontmatter `hooks:`,
# matcher `Write|Edit`). The data-scientist is advisory-only — its Write/Edit
# exist for one purpose: authoring/revising its method notes
# (docs/plans/<slug>-ml.md) and methodology reviews (docs/reviews/<slug>-ml.md).
# Thin wrapper: the shared logic lives in claude/scripts/guard-doc-writes.sh
# (resolved through this file's real path, so it also works via the
# ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/plans/*|*/docs/plans/*|docs/reviews/*|*/docs/reviews/*' \
  "data-scientist guardrail: Write/Edit targets '__PATH__', which is outside a docs/plans/ or docs/reviews/ directory (or /tmp scratchpad). The data-scientist is advisory-only — its Write/Edit are for method notes and methodology reviews only, never source, tests, config, or data. Approve only if this is genuinely an advisory document; otherwise the agent should put the change in its recommendations for the implementers."
