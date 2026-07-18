#!/usr/bin/env bash
# PreToolUse guard for the `joern` subagent (frontmatter `hooks:`, matcher
# `Bash`). joern loads Code Property Graphs into the lab's shared live FalkorDB
# and typically resets the target graph first — destructive / shared-state
# operations (GRAPH.DELETE, FLUSHALL/FLUSHDB, volume wipes, container
# force-removal) escalate to the human for approval. Thin wrapper: the shared
# logic lives in claude/scripts/guard-destructive-ops.sh (resolved through this
# file's real path, so it also works when invoked via the ~/.claude/agents/
# symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-destructive-ops.sh" joern
