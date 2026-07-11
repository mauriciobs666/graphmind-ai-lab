# Claude agents — context for AI agents working here

This directory (`claude/`) holds custom Claude Code subagents. Each agent is a folder: `<name>/<name>.md` (Markdown + YAML frontmatter) plus `<name>/kaizen/{plan,history}.md`. **Skills no longer live here** — they were unified into the repo-root [`skills/`](../skills/) home (see [`skills/README.md`](../skills/README.md)); cobb's `agent-maintenance` and `agent-standards` skills are there.

**The full agent catalog — what each does, when to use it, handoff contracts, hook enforcement — lives once, in [`README.md`](./README.md).** Each agent's frontmatter `description` is its routing contract and is auto-injected into sessions; each `<name>/<name>.md` is the source of truth for its behavior. This file keeps only the index plus directory-level conventions.

## Agents

Roster — behavior source is always `<name>/<name>.md`, kaizen at `<name>/kaizen/`; what each
does and when to use it lives in the injected descriptions and [`README.md`](./README.md):

`teco` (coordinator) · `tico` (product owner; **first-order**: `claude --agent tico`) ·
`architect` · `coder` · `tdd-engineer` · `frontend-engineer` · `qa-engineer` · `analyst` ·
`data-scientist` · `graph-dba` (carries two on-demand knowledge bases: `falkordb-quirks.md`,
live-verified and perishable — re-verify on upgrades — and `falkordb-reference.md`) · `devops` (user-scoped —
runs in every project) · `cobb` (team maintainer: `agent-maintenance`/`agent-standards` skills,
`scripts/audit-team.sh`, testing standards in `cobb/TESTING.md`).

## Hook machinery

The five doc-scoped write guards (architect, analyst, data-scientist, teco, tico) are thin
wrappers over one shared core, **`scripts/guard-doc-writes.sh`** — each wrapper passes its
allowed-path globs and escalation message; the core does jq→python3 path extraction, fail-open,
and the `permissionDecision: "ask"` escalation. `devops/hooks/guard-destructive-ops.sh` is
standalone (it matches Bash command patterns, not write paths). Frontmatter wires every hook via
`$HOME/.claude/agents/<name>/hooks/<script>.sh`, which resolves through the deployment symlink.

## Maintenance rules

- Adding/editing/renaming/removing an agent → update the agent source, its `kaizen/{plan,history}.md`, the full catalog entry in [`README.md`](./README.md), and the name rosters here and in the repo-root `AGENTS.md`, in the same change.
- Skills live in the repo-root [`skills/`](../skills/) home, not here. Their catalog is [`skills/README.md`](../skills/README.md); cobb's kaizen logs changes to `agent-maintenance`/`agent-standards`.
- Don't paste full system prompts or duplicate the README catalog here — point to the source.
