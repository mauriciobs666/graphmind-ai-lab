#!/usr/bin/env bash
# PreToolUse guard for the `tdd-engineer` subagent (frontmatter `hooks:`,
# second matcher entry alongside the existing Write|Edit guard, this one on
# `Bash`). tdd-engineer's `permissionMode: acceptEdits` auto-approves file
# edits but has no effect on Bash — without this guard every Bash call
# (pytest, git, npm...) still hits the plain confirm prompt, which is exactly
# the friction observed 2026-09-11: fixing Write/Edit alone just moved the
# interruption onto Bash instead of removing it — and the TDD loop's own
# red-green-refactor rhythm runs a test command after nearly every step, so
# this guard's absence compounds worse here than almost anywhere else on the
# team. Thin wrapper: shared logic lives in
# claude/scripts/guard-broad-bash.sh, which itself reuses
# claude/scripts/guard-destructive-ops.sh's pattern matching rather than
# duplicating it (resolved through this file's real path, so it also works
# via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-broad-bash.sh" tdd-engineer
