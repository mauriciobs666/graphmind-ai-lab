#!/usr/bin/env bash
# PreToolUse guard for the `devops` subagent (frontmatter `hooks:`, matcher
# `Bash`). Escalates destructive / shared-state operations (volume wipes,
# system prune, docker rm -f, compose down -v, Redis/FalkorDB flush/delete)
# to the human for approval. Thin wrapper: the shared logic lives in
# claude/scripts/guard-destructive-ops.sh (resolved through this file's real
# path, so it also works when invoked via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-destructive-ops.sh" devops
