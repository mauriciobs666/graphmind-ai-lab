# Claude agents — context for AI agents working here

This directory (`claude/`) holds custom Claude Code subagents. Each agent is a folder: `<name>/<name>.md` (Markdown + YAML frontmatter) plus `<name>/kaizen/{plan,history}.md`. **Skills no longer live here** — they were unified into the repo-root [`skills/`](../skills/) home (see [`skills/README.md`](../skills/README.md)); cobb's `agent-maintenance` and `agent-standards` skills are there.

**The full agent catalog — what each does, when to use it, handoff contracts, hook enforcement — lives once, in [`README.md`](./README.md).** Each agent's frontmatter `description` is its routing contract and is auto-injected into sessions; each `<name>/<name>.md` is the source of truth for its behavior. This file keeps only the index plus directory-level conventions.

## Agents

- **teco** — technical coordinator/orchestrator: decomposes goals, delegates to specialists via `Agent`, integrates; documentation curator + independent-review-by-default. Hybrid: pauses to the user at decision points. Source: `teco/teco.md`. Kaizen: `teco/kaizen/`. Hook: `teco/hooks/guard-coordination-doc-writes.sh` (Write/Edit → `docs/plans/` only).
- **tico** — conversational product owner, **first-order agent** (`claude --agent tico`; as a subagent it degrades to one interview round per invocation). Owns the requirements doc at `docs/requirements/<slug>.md` — WHAT/WHY, never HOW. Source: `tico/tico.md`. Kaizen: `tico/kaizen/`. Hook: `tico/hooks/guard-requirements-doc-writes.sh` (fires in main-session mode too).
- **architect** — investigates the codebase and writes step-by-step implementation plans to `docs/plans/<slug>.md`, handed off by path; read-only on code. Source: `architect/architect.md`. Kaizen: `architect/kaizen/`. Hook: `architect/hooks/guard-plan-doc-writes.sh`.
- **coder** — implements an approved plan/spec end-to-end (reads the plan file by path as source of truth); tests alongside. Routing is by efficiency: detailed plan → coder; test-first shapes → tdd-engineer; UI depth → frontend-engineer. Source: `coder/coder.md`. Kaizen: `coder/kaizen/`.
- **tdd-engineer** — implements strictly via TDD (red → green → refactor): bug fixes, safety-net refactors, test work, clear-contract features. Source: `tdd-engineer/tdd-engineer.md`. Kaizen: `tdd-engineer/kaizen/`.
- **frontend-engineer** — UI-depth implementer (web platform, React & peers, Streamlit); orients on the actual UI stack first, verifies in the running UI. Source: `frontend-engineer/frontend-engineer.md`. Kaizen: `frontend-engineer/kaizen/`.
- **qa-engineer** — behavior/acceptance QA: risk-based strategy → versioned test plan (`docs/test-plans/`) → execution (suites + black-box) → test report (`docs/test-reports/`). Source: `qa-engineer/qa-engineer.md`. Kaizen: `qa-engineer/kaizen/`.
- **analyst** — static reviewer/diagnostician of plans and code + RCA; severity-ranked findings and verdicts to `docs/reviews/<slug>.md` (RCA: `<slug>-rca.md`), never changes the artifact under review. Source: `analyst/analyst.md`. Kaizen: `analyst/kaizen/`. Hook: `analyst/hooks/guard-review-doc-writes.sh`.
- **data-scientist** — advisory AI/ML/DS scientist: method notes (`docs/plans/<slug>-ml.md`, with architect) and methodology reviews (`docs/reviews/<slug>-ml.md`, with analyst); never implements. Source: `data-scientist/data-scientist.md`. Kaizen: `data-scientist/kaizen/`. Hook: `data-scientist/hooks/guard-ds-doc-writes.sh`.
- **graph-dba** — FalkorDB DBA & data architect: modeling, Cypher, indexes/constraints, deployment sizing, GraphRAG layers; carries the live-verified quirks base `graph-dba/falkordb-quirks.md` (perishable — re-verify on upgrades). Source: `graph-dba/graph-dba.md`. Kaizen: `graph-dba/kaizen/`.
- **devops** — environments, containers, delivery lifecycle; **user-scoped** (symlinked into `~/.claude/agents/`, runs in every project) and orients per-repo before acting. Source: `devops/devops.md`. Kaizen: `devops/kaizen/`. Hook: `devops/hooks/guard-destructive-ops.sh` (destructive/shared-state ops → human approval).
- **cobb** — agentic-development expert (Claude Code, Kiro, OpenCode formats + cross-tool standards); relies on the `agent-maintenance` and `agent-standards` skills in [`skills/`](../skills/) and `claude/scripts/audit-team.sh` for team certification. Source: `cobb/cobb.md`. Kaizen: `cobb/kaizen/`. Testing standards: `cobb/TESTING.md`.

## Hook machinery

The five doc-scoped write guards (architect, analyst, data-scientist, teco, tico) are thin
wrappers over one shared core, **`scripts/guard-doc-writes.sh`** — each wrapper passes its
allowed-path globs and escalation message; the core does jq→python3 path extraction, fail-open,
and the `permissionDecision: "ask"` escalation. `devops/hooks/guard-destructive-ops.sh` is
standalone (it matches Bash command patterns, not write paths). Frontmatter wires every hook via
`$HOME/.claude/agents/<name>/hooks/<script>.sh`, which resolves through the deployment symlink.

## Maintenance rules

- Adding/editing/renaming/removing an agent → update the agent source, its `kaizen/{plan,history}.md`, the full catalog entry in [`README.md`](./README.md), and the one-line indexes here and in the repo-root `AGENTS.md`, in the same change.
- Skills live in the repo-root [`skills/`](../skills/) home, not here. Their catalog is [`skills/README.md`](../skills/README.md); cobb's kaizen logs changes to `agent-maintenance`/`agent-standards`.
- Don't paste full system prompts or duplicate the README catalog here — point to the source.
