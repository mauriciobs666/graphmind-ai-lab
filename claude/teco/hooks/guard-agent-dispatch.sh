#!/usr/bin/env bash
# PreToolUse guard for the `teco` subagent (frontmatter `hooks:`, matcher
# `Agent|Task`). Thin wrapper: the shared logic lives in
# claude/scripts/guard-agent-dispatch.sh (resolved through this file's real
# path, so it also works via the ~/.claude/agents/ symlink). Extracted out of
# this file into that shared core 2026-08-29, when `tico` gained the same
# `Agent`+`SendMessage` dispatch shape and so the same hazard
# (`claude/docs/requirements/tico-specialist-collaboration.md`); see the
# core script's own header for why.
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-agent-dispatch.sh" "teco"
