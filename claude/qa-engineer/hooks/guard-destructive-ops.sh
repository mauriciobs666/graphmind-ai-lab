#!/usr/bin/env bash
# PreToolUse guard for the `qa-engineer` subagent (frontmatter `hooks:`, matcher
# `Bash`). The qa-engineer drives running apps against the lab's shared live
# FalkorDB — destructive / shared-state operations (GRAPH.DELETE,
# FLUSHALL/FLUSHDB, volume wipes, container force-removal) escalate to the
# human for approval, backstopping its "never mutate the environment" rule.
# See cobb kaizen K-011 (2026-07-11). Thin wrapper: the shared logic lives in
# claude/scripts/guard-destructive-ops.sh (resolved through this file's real
# path, so it also works when invoked via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-destructive-ops.sh" qa-engineer
