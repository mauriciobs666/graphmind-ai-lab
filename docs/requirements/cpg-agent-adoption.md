# CPG agent adoption — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD) · **Last updated:** 2026-08-15

## Intent
_To be captured through the interview._

## Problem & current state
The `cpg-analysis` skill (reads a Joern Code Property Graph loaded in FalkorDB) was
scoped at M2 (`docs/plans/m2-cpg-analysis-skill.md`, delivered ✅) to exactly three
consumer agents — `analyst`, `architect`, `qa-engineer` — plus `graph-dba` as the
producer. `docs/requirements/cpg-query-access.md` (M3, delivered ✅, archived) later
fixed the read path onto the `mcp__cpg__query` MCP tool for that same three-agent
set; its own `Scope` section named `analyst`/`architect`/`qa-engineer` as **in**, and
"general agent access to FalkorDB" as **out**.

**Prior-decision note (surfaced 2026-08-15, before this interview's first question):**
this feature request — widening which agents benefit from the CPG — touches ground
already ruled on at M2/M3. It is not a reversal of what M2/M3 *built* (the MCP tool,
the skill's recipes), but it does revisit the *consumer-scope* boundary those
milestones drew deliberately. Whatever this interview settles must say explicitly
whether it extends that boundary (new consumers added, M2/M3 untouched) or supersedes
it — the resulting requirements doc and any downstream plan need to reconcile with
`m2-cpg-analysis-skill.md` and `cpg-query-access.md` rather than silently diverge from
them (`docs/BACKLOG.md` M2/M3 rows are the canonical status log for both).

Investigation ahead of this interview (tico, 2026-08-15) found, concretely:
- Only `analyst`, `architect`, `qa-engineer` (consumers) and `graph-dba` (producer) are
  wired to `cpg-analysis` / `mcp__cpg__query` today. `coder`/`tdd-engineer` were
  considered for a similar skill extension once before (`python-web-quirks`, 2026-08-09)
  but `cpg-analysis` was not carried along at that time.
- Routing language is conditional ("when a CPG is loaded") — there is no proactive
  discovery step, and `graph-dba`'s own description frames CPG generation as
  deliberately rare/on-demand, not something to suggest.
- Only two components have a CPG actually built and loaded: `cpg_falkorchat` (2,037
  methods) and `cpg_salesperson` (359 methods), confirmed live via `GRAPH.QUERY` on
  2026-08-15. No other component (`cpg/`, `claude/`, `skills/`, `opencode/`,
  `mcp-monitor/`, `kiro/`) has one.
- No evidence of the CPG being consulted in real (non-CPG-meta) task work since it was
  built — e.g. the most recent `analyst` review, `falkor-chat/docs/reviews/graphrag-eval.md`,
  makes no mention of it.
- No documented freshness/rebuild convention — `cpg_falkorchat` traces to M3-era work;
  falkor-chat has had substantial commits since with no recorded CPG refresh tied to them.

## User stories
_To be captured._

## Functional requirements
_To be captured._

## Out of scope
_To be captured._

## Acceptance criteria
_To be captured._

## Open questions
_To be captured._

## Decision log
- 2026-08-15 — Session opened from a Mode-2 explanation ("why aren't the Claude agents
  benefiting from the CPG, how can we improve them") that surfaced enough of a gap to
  warrant a requirements interview. Stakeholder chose to proceed with the interview
  (Mode 1) over going straight to a `cobb` design pass.
- 2026-08-15 — What triggered this now? → **All three** surfaced reasons apply at once:
  a specific review that took longer than it should have, a bug impact-analysis
  would plausibly have caught, and the standing discomfort of an expensive capability
  sitting dormant. Stakeholder did not single one out as primary.
- 2026-08-15 — What does "solved" look like? → **All four** candidate outcomes matter:
  (1) more agents consult the CPG beyond the current three, (2) the already-wired
  agents actually reach for it in real tasks (not just carry the wiring), (3) staleness
  is knowable/managed, (4) coverage extends past `falkor-chat`/`salesperson`. This is a
  broad feature — the FR list will need to cover all four without conflating them.
