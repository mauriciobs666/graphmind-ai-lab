# Claude agents — context for AI agents working here

This directory (`claude/`) holds custom Claude Code subagents. Each agent is a folder: `<name>/<name>.md` (Markdown + YAML frontmatter) plus `<name>/kaizen/{plan,history}.md`. Every agent's raw learnings capture (durable environment facts discovered during runs) writes directly into one **shared** working-memory FalkorDB graph, `kaizen_team`, `author`-partitioned, as `:KaizenEntry` nodes attributed to itself via `mcp__cypher__query` — a pattern piloted on `graph-dba` as its own graph (`kaizen_graph_dba`), migrated team-wide onto one graph per agent 2026-08-20, then consolidated the same day onto this single shared `kaizen_team` graph (`claude/cobb/kaizen/history.md`; `docs/plans/generic-cypher-mcp2.md`) so one query reaches every agent's raw learnings. The 12 agents that existed at the migration each carry a `kaizen/inbox.md` — now a **permanent frozen historical snapshot**, never written to again but never deleted; an agent created since the consolidation gets **no `inbox.md` at all** (FR-12/AC-9) — `audit-team.sh` check 1 requires only `plan.md`+`history.md`, not a triad. `cobb` distills the shared graph periodically per the `agent-maintenance` skill §5 (verify → route to prompt/knowledge base/project docs → log in the entry's own agent's `history.md` → clear, a curator-scoped `DETACH DELETE` through `mcp__cypher__query` against `kaizen_team`). **Skills no longer live here** — they were unified into the repo-root [`skills/`](../skills/) home (see [`skills/README.md`](../skills/README.md)); cobb's `agent-maintenance` and `agent-standards` skills are there.

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
`security-expert` (on-demand deep security reviewer: code/app security, agent/prompt-safety,
secrets/infra-hardening, compliance checklists — advisory to `analyst`/`cobb`/`devops`; the only
agent on this team gated to attempt active exploitation, local/dev targets only, fresh explicit
approval every time) ·
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

`security-expert` carries **two** `PreToolUse` hooks under one frontmatter `hooks:` block — a
departure from the one-hook-per-agent pattern above. Its `Write|Edit` guard
(`security-expert/hooks/guard-review-doc-writes.sh`) is a normal thin wrapper over the shared
`guard-doc-writes.sh` core, scoped to `docs/reviews/*` only. Its `Bash` guard
(`security-expert/hooks/guard-exploitation-approval.sh`), enforcing FR-10's "active exploitation
needs a fresh, explicit approval every time, local/dev targets only"
(`docs/requirements/security-expert.md`), is deliberately **not** layered on the shared
`guard-destructive-ops.sh` core — that core's catalog is shared-state-destruction literals
(`GRAPH.DELETE`, `FLUSHALL`, volume wipes), a different hazard class from the offensive-tool/
network-exploitation patterns this guard matches (named tools like `sqlmap`/`nmap`/`msfconsole`,
listener setups, or a network-reaching command with no visible local/dev marker). It's a
standalone, agent-owned script with the same mechanics/contract as the shared cores (fail-open,
`ask`-only, jq→python3 extraction) — extract it into a shared core only if a second
exploitation-shaped agent is ever added (`security-expert/kaizen/plan.md` K-003).

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

- **A stakeholder proposal for a new team member is a `tico` requirements interview, not a
  straight-to-`cobb` request.** It's an ordinary Mode 1 interview with `claude/` treated as the
  component — WHAT/WHY only (the new agent's remit, its boundaries with existing agents, any
  destructive-shaped capability), landing at `claude/docs/requirements/<slug>.md`. Only once that
  doc reaches **Ready for design** does `cobb` design the actual agent (name, prompt, tools,
  hooks) from it. Precedent: `claude/docs/requirements/security-expert.md`.
- Adding/editing/renaming/removing an agent → update the agent source, its `kaizen/{plan,history}.md` (no `inbox.md` is created for a new agent — FR-12/AC-9), the full catalog entry in [`README.md`](./README.md), and the name rosters here and in the repo-root `AGENTS.md`, in the same change.
- Skills live in the repo-root [`skills/`](../skills/) home, not here. Their catalog is [`skills/README.md`](../skills/README.md); cobb's kaizen logs changes to `agent-maintenance`/`agent-standards`.
- Don't paste full system prompts or duplicate the README catalog here — point to the source.
