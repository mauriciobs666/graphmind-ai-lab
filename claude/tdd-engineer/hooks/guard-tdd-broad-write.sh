#!/usr/bin/env bash
# PreToolUse guard for the `tdd-engineer` subagent (frontmatter `hooks:`,
# matcher `Write|Edit`). tdd-engineer's remit is genuinely "the whole
# codebase, this task" (red-green-refactor over whatever source/test files
# the task needs) — no single folder to allowlist, so this uses the INVERSE
# shape (claude/scripts/guard-broad-write.sh): allow by default, escalate
# only on a path that's known to belong to a DIFFERENT specialist's
# documented deliverable-path convention, or that the stakeholder left
# genuinely unresolved. See
# claude/docs/requirements/agent-permission-friction.md FR-2 (instances 7,8;
# AC-3's tdd-engineer example) and U1 below.
#
# Deny-list (escalates). Every entry doubled — bare + "*/"-prefixed — for
# the same reason as guard-doc-writes.sh's callers (Claude Code's
# tool_input.file_path can arrive absolute; a leading "*" is what absorbs an
# arbitrary absolute prefix ahead of the literal directory — analyst review
# 2026-08-21 Finding 1, same fix as §4 above). Also folds in Finding 5
# (skills/agent-maintenance/*, skills/agent-standards/*, cypher-mcp/README.md
# were named in §4 as cobb's topic-remit but missing from this deny-list):
#   docs/plans/*, */docs/plans/*             architect / teco-coordination /
#                                             data-scientist-ml
#   docs/reviews/*, */docs/reviews/*         analyst / security-expert /
#                                             data-scientist-ml
#   docs/requirements/*, */docs/requirements/*   tico
#   docs/manuals/*, */docs/manuals/*         tico
#   docs/test-plans/*, */docs/test-plans/*   qa-engineer
#   docs/test-reports/*, */docs/test-reports/*   qa-engineer
#   claude/*/*.md, */claude/*/*.md           agent definitions / kaizen — cobb
#   claude/*/kaizen/*, */claude/*/kaizen/*   (same claude/*/*.md caveat as §4
#                                             Finding 2 applies here too: also
#                                             catches kaizen/inbox.md, accepted
#                                             the same way — frozen, no-op)
#   claude/README.md, */claude/README.md     team catalog/context — cobb
#   claude/AGENTS.md, */claude/AGENTS.md
#   claude/CLAUDE.md, */claude/CLAUDE.md
#   skills/README.md, */skills/README.md
#   skills/agent-maintenance/*, */skills/agent-maintenance/*   cobb's own
#   skills/agent-standards/*, */skills/agent-standards/*       skill packages
#   cypher-mcp/README.md, */cypher-mcp/README.md   cobb's topic-remit (FR-1
#                                             instance 6) — a tdd-engineer
#                                             write here should escalate too
#   docs/BACKLOG.md, */docs/BACKLOG.md       U1 (agent-permission-friction.md):
#                                             the stakeholder was explicitly
#                                             unsure whether a tdd-engineer ->
#                                             BACKLOG.md write is in-remit —
#                                             left here so it keeps asking,
#                                             same as today, deliberately NOT
#                                             resolving U1 either way
#
# Everything else -- source code, test files, any other in-task file --
# is explicitly allowed. Thin wrapper: shared logic lives in
# claude/scripts/guard-broad-write.sh (resolved through this file's real
# path, so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-broad-write.sh" \
  'docs/plans/*|*/docs/plans/*|docs/reviews/*|*/docs/reviews/*|docs/requirements/*|*/docs/requirements/*|docs/manuals/*|*/docs/manuals/*|docs/test-plans/*|*/docs/test-plans/*|docs/test-reports/*|*/docs/test-reports/*|claude/*/*.md|*/claude/*/*.md|claude/*/kaizen/*|*/claude/*/kaizen/*|claude/README.md|*/claude/README.md|claude/AGENTS.md|*/claude/AGENTS.md|claude/CLAUDE.md|*/claude/CLAUDE.md|skills/README.md|*/skills/README.md|skills/agent-maintenance/*|*/skills/agent-maintenance/*|skills/agent-standards/*|*/skills/agent-standards/*|cypher-mcp/README.md|*/cypher-mcp/README.md|docs/BACKLOG.md|*/docs/BACKLOG.md' \
  "tdd-engineer guardrail: Write/Edit targets '__PATH__', which looks like another specialist's documented deliverable path (a plan/review/requirements/manual/test-plan/test-report doc, an agent definition or kaizen file, a team catalog or skill package, an MCP-standards doc, or the project backlog). Approve only if tdd-engineer genuinely owns this write for the current task; otherwise it belongs to whichever agent normally authors that doc kind."
