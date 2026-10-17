#!/usr/bin/env bash
# PreToolUse guard for the `teco` agent (frontmatter `hooks:`, matcher
# `Write|Edit`). Repository-bounded: any file inside the git repo the session
# was started in (or /tmp) is allowed; outside it, or inside its .git/, asks.
# Thin wrapper: the logic and full contract live in
# claude/scripts/guard-repo-writes.sh (resolved through this file's real path,
# so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-repo-writes.sh" teco
