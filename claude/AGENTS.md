# Claude agents — context for AI agents working here

This directory (`claude/`) holds custom Claude Code subagents. Each agent is a folder: `<name>/<name>.md` (Markdown + YAML frontmatter) plus `<name>/kaizen/{plan,history,inbox}.md` — `inbox.md` is the agent's own append-only learnings capture (durable environment facts discovered during runs), with one exception: `graph-dba`'s raw capture now writes directly into the `kaizen_graph_dba` FalkorDB graph instead (`:KaizenEntry` nodes, attributed to itself via `mcp__cypher__query`), and its own `kaizen/inbox.md` is a frozen historical snapshot, no longer written to (`docs/plans/generic-cypher-mcp.md`). `cobb` distills inboxes — and `graph-dba`'s graph entries alongside them — periodically per the `agent-maintenance` skill §5 (verify → route to prompt/knowledge base/project docs → log in `history.md` → clear, a curator-scoped graph delete for `graph-dba`). **Skills no longer live here** — they were unified into the repo-root [`skills/`](../skills/) home (see [`skills/README.md`](../skills/README.md)); cobb's `agent-maintenance` and `agent-standards` skills are there.

**The full agent catalog — what each does, when to use it, handoff contracts, hook enforcement — lives once, in [`README.md`](./README.md).** Each agent's frontmatter `description` is its routing contract and is auto-injected into sessions; each `<name>/<name>.md` is the source of truth for its behavior. This file keeps only the index plus directory-level conventions.

## Agents

Roster — behavior source is always `<name>/<name>.md`, kaizen at `<name>/kaizen/`; what each
does and when to use it lives in the injected descriptions and [`README.md`](./README.md):

`teco` (coordinator) · `tico` (product owner, project explainer, user-manual curator; **interactive**: `claude --agent tico`) ·
`architect` · `coder` · `tdd-engineer` · `frontend-engineer` ·
`qa-engineer` (carries an on-demand knowledge base: `qa-testing-techniques.md` — environment/
tooling techniques such as the WSL2 browser-automation fallback, driving an interactive TUI, and
CLI health-check gotchas) ·
`analyst` (carries an on-demand knowledge base: `review-techniques.md` — review-methodology
techniques consulted on demand) ·
`data-scientist` (carries an on-demand knowledge base: `lm-studio-model-notes.md` —
live-verified small-model realism notes for this lab's LM Studio stack) ·
`graph-dba` (carries two on-demand knowledge bases: `falkordb-quirks.md`,
live-verified and perishable — re-verify on upgrades — and `falkordb-reference.md`; also drives
Joern via the `joern-cpg` skill, on demand, to build a repo's CPG and export/load it into
FalkorDB as Cypher — a rare capability, not a proactive default) ·
`devops` (user-scoped — runs in every project; carries an on-demand knowledge base:
`ops-quirks.md` — live-verified Docker/BuildKit/Bash-scripting traps) ·
`cobb` (team maintainer: `agent-maintenance`/`agent-standards` skills,
`scripts/audit-team.sh`, testing standards in `cobb/TESTING.md`).

## Hook machinery

The five doc-scoped write guards (architect, analyst, data-scientist, teco, tico) are thin
wrappers over one shared core, **`scripts/guard-doc-writes.sh`** — each wrapper passes its
allowed-path globs and escalation message; the core does jq→python3 path extraction, fail-open,
and the `permissionDecision: "ask"` escalation. The three destructive-ops guards (devops,
graph-dba, qa-engineer) are likewise thin wrappers over **`scripts/guard-destructive-ops.sh`**
(each passes its agent name; the core matches Bash command patterns — `GRAPH.DELETE`,
`FLUSHALL`/`FLUSHDB`, volume wipes, `docker rm -f`, and (since 2026-08-08, C-311)
`pipeline.sh ... --reset` — a wrapper invocation matched ad hoc because the script runs
`GRAPH.DELETE` internally, where the literal string never reaches the guard — not write paths).
Frontmatter wires every
hook via `$HOME/.claude/agents/<name>/hooks/<script>.sh`, which resolves through the deployment
symlink.

**Git-commit authority is prompt-level, not hook-enforced.** Only `tico` and `teco` document
`git add`/`git commit` authority — `tico` for its own doc kinds (requirements, manuals; mirrors
its Write/Edit guard exactly), `teco` for a coordinated specialist's already-verified deliverable
by explicit path (its integrator role, deliberately wider than its own Write/Edit guard). No
`PreToolUse` hook matches `git commit` (the destructive-ops guards match Bash command patterns
like `GRAPH.DELETE`, not versioning commands), so this is self-discipline backstopped only by
`scripts/audit-team.sh` check 8, which fails if any agent other than `tico`/`teco` claims the
same authority. Stakeholder decision, 2026-07-30: no proliferation of commit rights beyond these
two — see `claude/teco/kaizen/history.md` and `claude/cobb/kaizen/history.md` for the reasoning.

## Maintenance rules

- Adding/editing/renaming/removing an agent → update the agent source, its `kaizen/{plan,history}.md` (and seed `kaizen/inbox.md` on creation), the full catalog entry in [`README.md`](./README.md), and the name rosters here and in the repo-root `AGENTS.md`, in the same change.
- Skills live in the repo-root [`skills/`](../skills/) home, not here. Their catalog is [`skills/README.md`](../skills/README.md); cobb's kaizen logs changes to `agent-maintenance`/`agent-standards`.
- Don't paste full system prompts or duplicate the README catalog here — point to the source.
