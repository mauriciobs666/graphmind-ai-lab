#!/usr/bin/env bash
# PreToolUse guard for the `security-expert` subagent (frontmatter `hooks:`,
# matcher `Write|Edit`). The security expert is review-only across all four
# of its capabilities (code-security, agent/prompt-safety, secrets/infra,
# compliance) — FR-11 requires every review to land as a written findings
# report at <component>/docs/reviews/<slug>.md, the same convention as
# analyst's reviews; it never edits the artifact under review, source, tests,
# config, or another agent's files (agent/skill/prompt/kaizen changes stay
# `cobb`'s; infra/secrets changes stay `devops`'s — this agent only writes
# its own opinion). Thin wrapper: shared logic lives in
# claude/scripts/guard-doc-writes.sh (resolved through this file's real path,
# so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/reviews/*|*/docs/reviews/*' \
  "security-expert guardrail: Write/Edit targets '__PATH__', which is outside a docs/reviews/ directory or the /tmp scratchpad. The security expert is review-only — its Write/Edit are for findings reports only, never source, tests, config, infra, or the artifact under review. Approve only if this is genuinely a review report; otherwise the agent should put the change in its findings for the artifact's owner (cobb for agent/prompt material, devops for infra/secrets, the code owner otherwise)."
