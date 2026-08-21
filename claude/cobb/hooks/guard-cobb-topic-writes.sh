#!/usr/bin/env bash
# PreToolUse guard for the `cobb` subagent (frontmatter `hooks:`, matcher
# `Write|Edit`). Cobb's remit is TOPIC-bounded, not folder-bounded — the team
# maintainer's job cuts across every agent's own folder (definitions, kaizen
# curation) plus a small, explicitly maintained set of cross-cutting
# MCP/agent-standards docs that live outside claude/ and skills/ entirely
# (e.g. a component README documenting MCP wiring). See
# claude/docs/requirements/agent-permission-friction.md FR-1 for the
# evidence trail (instances 1-3,5,6,9) and counter-example C2
# (docs/BACKLOG.md — genuinely out of remit, still escalates below).
#
# Allowed-path union (every entry doubled — bare + "*/"-prefixed — because
# tool_input.file_path can arrive absolute, not just repo-relative; bash
# `case` lets a leading `*` cross `/`, which is what lets the doubled sibling
# absorb an arbitrary absolute prefix ahead of the literal directory. Every
# existing guard-doc-writes.sh caller already relies on this
# (claude/architect/kaizen/history.md:342, "absolute + relative docs/plans/
# -> pass") — analyst review 2026-08-21 caught this plan's first draft
# omitting the doubled form here, which would have meant the guard never
# actually matches on the delivery shape every evidenced FR-1 instance has
# (a Task-delegated subagent write, file_path absolute):
#   claude/*/*.md, */claude/*/*.md   any agent's own top-level docs, incl.
#                                     cobb's own (<name>/<name>.md,
#                                     TESTING.md, *-notes.md, *-quirks.md,
#                                     ...) — NOTE (analyst review Finding 2):
#                                     because `case` lets a bare `*` cross
#                                     `/`, this also matches
#                                     claude/<agent>/kaizen/inbox.md (and any
#                                     other .md file at any depth under
#                                     claude/<agent>/). Scoping this to
#                                     "exactly one path segment" would need
#                                     an extglob pattern (`+([^/])`) — a
#                                     single bracket-negation `[^/]` only
#                                     constrains ONE character, not the whole
#                                     run, so it does NOT actually work in
#                                     plain `case` matching (verified: `case
#                                     "kaizen/inbox.md" in [^/]*.md) ...`
#                                     still matches) — and the shared core
#                                     doesn't use extglob today. ACCEPTED
#                                     DELIBERATELY instead: inbox.md is
#                                     frozen and nobody writes to it (FR-1's
#                                     evidence trail), so silently allowing a
#                                     write there costs nothing; the plain,
#                                     already-battle-tested glob form is kept
#                                     rather than introducing a new pattern
#                                     dialect for one low-risk edge case.
#                                     (2026-08-21: all 12 frozen inbox.md
#                                     files were deleted outright once fully
#                                     distilled — the caveat above is now
#                                     "matches a path that doesn't exist" in
#                                     addition to "matches a path nobody
#                                     writes to"; same conclusion, costs
#                                     nothing either way, no glob change
#                                     needed.)
#   claude/*/kaizen/history.md, */claude/*/kaizen/history.md
#   claude/*/kaizen/plan.md, */claude/*/kaizen/plan.md
#                                     kaizen curation for any agent — cobb
#                                     curates, not the agent itself (FR-1
#                                     instance 5); kaizen/inbox.md is not
#                                     listed here either — redundant with the
#                                     claude/*/*.md entry above in any case
#   claude/README.md, */claude/README.md
#   claude/AGENTS.md, */claude/AGENTS.md
#   claude/CLAUDE.md, */claude/CLAUDE.md
#                                     team catalog + agent-context files —
#                                     cobb's own maintenance duty
#                                     (claude/AGENTS.md "Maintenance rules")
#   skills/agent-maintenance/*, */skills/agent-maintenance/*
#   skills/agent-standards/*, */skills/agent-standards/*
#                                     cobb's own skill packages
#   skills/README.md, */skills/README.md
#                                     skills catalog (shared; cobb updates its
#                                     own entries here)
#   cypher-mcp/README.md, */cypher-mcp/README.md
#                                     MCP/agent-standards doc outside
#                                     claude/skills/ (FR-1 instance 6) — a
#                                     path-only hook can't detect "documents
#                                     MCP wiring" by content, so this line is
#                                     a small, EXPLICITLY MAINTAINED list:
#                                     extend it, don't broaden the globs
#                                     above, when a new such doc surfaces.
#
# Deliberately NOT allowed (still escalates — AC-4, counter-example C2): a
# general project doc with no agent/skill/MCP relevance, e.g. docs/BACKLOG.md.
# "cobb can edit anything on the agents" was corrected by the stakeholder to
# "topic-bounded, not folder-bounded," not path-unrestricted.
#
# Thin wrapper: shared logic lives in claude/scripts/guard-doc-writes.sh
# (resolved through this file's real path, so it also works via the
# ~/.claude/agents/ symlink).
exec "$(dirname "$(readlink -f "$0")")/../../scripts/guard-doc-writes.sh" \
  'claude/*/*.md|*/claude/*/*.md|claude/*/kaizen/history.md|*/claude/*/kaizen/history.md|claude/*/kaizen/plan.md|*/claude/*/kaizen/plan.md|claude/README.md|*/claude/README.md|claude/AGENTS.md|*/claude/AGENTS.md|claude/CLAUDE.md|*/claude/CLAUDE.md|skills/agent-maintenance/*|*/skills/agent-maintenance/*|skills/agent-standards/*|*/skills/agent-standards/*|skills/README.md|*/skills/README.md|cypher-mcp/README.md|*/cypher-mcp/README.md' \
  "cobb guardrail: Write/Edit targets '__PATH__', which is outside cobb's agentic-development topic-remit (any agent's own definition file, kaizen curation for the team, MCP/agent-standards documentation) or the /tmp scratchpad. Approve only if this is genuinely agent/skill/MCP-standards work; otherwise it belongs to whichever agent actually owns that doc kind (e.g. a general project backlog item is not cobb's job — see counter-example C2)."
