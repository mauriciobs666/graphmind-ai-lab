#!/usr/bin/env bash
# PreToolUse guard for the `tico` agent (frontmatter `hooks:`, matcher
# `Write|Edit` — fires in main-session mode too, and for tico's delegates'
# calls). Repository-bounded: any file inside the git repo the session was
# started in (or /tmp) is allowed; outside it, or inside its .git/, asks.
# Replaced the docs/requirements|manuals|plans/*-coordination.md allowlist
# (guard-tico-doc-writes.sh) on 2026-10-17 — history in
# claude/tico/kaizen/history.md. What tico may author stays prompt-held
# (tico.md, Guardrails). Thin wrapper: the logic and full contract live in
# claude/scripts/guard-repo-writes.sh (resolved through this file's real path,
# so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-repo-writes.sh" tico
