#!/usr/bin/env bash
# PreToolUse guard for the `tico` agent (frontmatter `hooks:`, matcher
# `Write|Edit` — fires in main-session mode too). Tico owns three doc kinds —
# its Write/Edit exist for: authoring/advancing the feature requirements
# document under docs/requirements/, authoring/maintaining user manuals under
# docs/manuals/, and (since 2026-09-01) its own docs-only coordination ledger
# at docs/plans/<slug>-coordination.md — same convention teco uses, opened
# only when tico is sequencing/gating a multi-unit chain that never touches
# code (see tico.md, "Coordinating a docs-only chain"). (Learning capture
# moved to the `kaizen_team` graph via `mcp__cypher__query` back on
# 2026-08-20 — it was never a Write/Edit path — and tico's frozen
# `kaizen/inbox.md` was itself deleted 2026-08-21, so no inbox allowance
# belongs in this glob any more.) Thin wrapper: the shared logic lives in
# claude/scripts/guard-doc-writes.sh (resolved through this file's real
# path, so it also works via the ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'docs/requirements/*|*/docs/requirements/*|docs/manuals/*|*/docs/manuals/*|docs/plans/*-coordination.md|*/docs/plans/*-coordination.md' \
  "tico guardrail: Write/Edit targets '__PATH__', which is outside a docs/requirements/ or docs/manuals/ directory, a docs/plans/*-coordination.md ledger, or the /tmp scratchpad. Tico owns requirements documents, user manuals, and its own docs-only coordination ledger only — no source, tests, config, or design docs (a plan itself is architect's to write, not tico's). Approve only if this is genuinely one of those artifacts; otherwise the need belongs in the requirements doc (or a downstream agent's deliverable), not a tico write."
