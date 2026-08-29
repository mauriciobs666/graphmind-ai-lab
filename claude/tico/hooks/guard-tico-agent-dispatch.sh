#!/usr/bin/env bash
# PreToolUse guard for the `tico` agent (frontmatter `hooks:`, matcher
# `Agent|Task` — fires in main-session mode too). Same hazard `teco`'s
# dispatch guard exists for: an `Agent` dispatch that omits `subagent_type`
# silently runs as `general-purpose` — no error — and a later `SendMessage`
# resume (tico's multi-turn specialist-consult mechanism) inherits the wrong
# identity for the rest of the thread. Thin wrapper: the shared logic lives
# in claude/scripts/guard-agent-dispatch.sh (resolved through this file's
# real path, so it also works via the ~/.claude/agents/ symlink). Added
# 2026-08-29 alongside the `SendMessage` grant
# (`claude/docs/requirements/tico-specialist-collaboration.md`).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-agent-dispatch.sh" "tico"
