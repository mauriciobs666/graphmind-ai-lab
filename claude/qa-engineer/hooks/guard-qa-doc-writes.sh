#!/usr/bin/env bash
# PreToolUse guard for the `qa-engineer` subagent (frontmatter `hooks:`,
# second matcher entry alongside the existing Bash destructive-ops guard).
# FR-3 (claude/docs/requirements/agent-permission-friction.md): qa-engineer's
# two versioned deliverable-doc kinds must not require a manual confirmation
# — evidenced by instance 4 (docs/test-plans/generic-cypher-mcp2.md,
# docs/test-reports/generic-cypher-mcp2-report.md).
#
# on_mismatch="pass" (NOT the shared core's "ask" default): qa-engineer also
# authors automated functional tests and drives the running app as part of
# its own execution phase (qa-engineer.md §3) — those Write/Edit calls are
# squarely in-remit too, just not doc-scoped. Escalating them would be new
# friction this FR never evidenced; "pass" leaves them to the ambient
# permission flow exactly as today, only the two doc-kind paths below change
# behavior (silent exit 0 -> explicit allow).
#
# Thin wrapper: shared logic lives in claude/scripts/guard-doc-writes.sh
# (resolved through this file's real path, so it also works via the
# ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/test-plans/*|*/docs/test-plans/*|docs/test-reports/*|*/docs/test-reports/*' \
  "unused — on_mismatch is pass, no escalation message is ever rendered" \
  pass
