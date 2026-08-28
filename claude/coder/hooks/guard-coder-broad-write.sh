#!/usr/bin/env bash
# PreToolUse guard for the `coder` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). Like tdd-engineer, coder's remit is genuinely "the whole
# codebase, this task" — no single folder to allowlist, so this uses the
# INVERSE shape (claude/scripts/guard-broad-write.sh): every in-task write
# (source, tests, scripts, component docs it maintains like QUERIES.md) gets
# an explicit hook "allow" — the one mode-independent way to suppress the
# per-write confirmation prompt for a Task-spawned subagent (see
# claude/docs/plans/agent-permission-friction.md §1.3; evidence 2026-08-27/28:
# every hook-free coder Write/Edit in a teco/auto session prompted, while
# guard-carrying agents wrote silently). Escalate ("ask") only on a path
# that belongs to a DIFFERENT specialist's documented deliverable-path
# convention — same deny-list as tdd-engineer's guard-tdd-broad-write.sh,
# kept in lockstep (both agents share the implementer altitude; if one list
# changes, change the other or note why not).
# Thin wrapper: shared logic lives in claude/scripts/guard-broad-write.sh
# (resolved through this file's real path, so it also works via the
# ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-broad-write.sh" \
  'docs/plans/*|*/docs/plans/*|docs/reviews/*|*/docs/reviews/*|docs/requirements/*|*/docs/requirements/*|docs/manuals/*|*/docs/manuals/*|docs/test-plans/*|*/docs/test-plans/*|docs/test-reports/*|*/docs/test-reports/*|claude/*/*.md|*/claude/*/*.md|claude/*/kaizen/*|*/claude/*/kaizen/*|claude/README.md|*/claude/README.md|claude/AGENTS.md|*/claude/AGENTS.md|claude/CLAUDE.md|*/claude/CLAUDE.md|skills/README.md|*/skills/README.md|skills/agent-maintenance/*|*/skills/agent-maintenance/*|skills/agent-standards/*|*/skills/agent-standards/*|cypher-mcp/README.md|*/cypher-mcp/README.md|docs/BACKLOG.md|*/docs/BACKLOG.md' \
  "coder guardrail: Write/Edit targets '__PATH__', which looks like another specialist's documented deliverable path (a plan/review/requirements/manual/test-plan/test-report doc, an agent definition or kaizen file, a team catalog or skill package, an MCP-standards doc, or the project backlog). Approve only if coder genuinely owns this write for the current task; otherwise it belongs to whichever agent normally authors that doc kind."
