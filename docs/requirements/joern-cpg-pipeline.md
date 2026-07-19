# Joern CPG Component — Feature Requirements
> Status: M1 (producer pipeline) **delivered ✅** · M2 (CPG consumer skill) **specified, in progress** ·
> Last updated: 2026-07-18

## Intent
The stakeholder wants to run **Joern** to extract a **Code Property Graph (CPG)** from source code,
represent that CPG **in FalkorDB**, and let the **agent team** query it to do real
software-development work. Two phases:

- **M1 — Produce the CPG** (delivered): a pipeline builds a CPG from any repo and loads it into
  FalkorDB so the code graph is traversable with Cypher.
- **M2 — Consume the CPG** (specified): a **`cpg-analysis` skill** (one skill, per-task recipes)
  teaches existing agents to query the loaded CPG for **impact analysis**, **root-cause analysis**,
  **code review**, and **test-gap** analysis.

**RAG indexing of code is explicitly not a goal for now.** The CPG serves the **agent team**
(`analyst`, `architect`, `qa-engineer`) first, and later the **workflows and agent nodes** being
built in `falkor-chat`.

## Milestones & delivery status
| Milestone | Scope | Status | Where |
|---|---|---|---|
| **M1 — Producer pipeline** | FR-1…FR-8 — build a CPG and load it into FalkorDB; the stored graph answers caller/callee, transitive-impact, data-flow, and symbol-reference queries | **✅ delivered 2026-07-17** (commit `b2b9a6e`) | `joern` agent + `joern-cpg` skill; see [`../HISTORY.md`](../HISTORY.md) |
| **M2 — CPG consumer skill** | FR-9…FR-14 — a `cpg-analysis` skill with impact / RCA / code-review / test-gap recipes over the loaded CPG | 🔵 specified | [`../BACKLOG.md`](../BACKLOG.md) C-201…C-208 |

## Problem & current state
- Before M1 the agents (e.g. `analyst`, which does impact analysis and RCA) reasoned about code by
  reading and grepping files — there was **no structured call-graph / data-flow representation** to
  query. **M1 closed that gap on the producer side**: a CPG for a repo now exists in FalkorDB.
- **M2 is the remaining gap**: the loaded CPG is only usable today by someone who already knows the
  Joern→FalkorDB schema and hand-writes Cypher. The agents that would benefit (`analyst`,
  `architect`, `qa-engineer`) need packaged, task-shaped query recipes to actually use it.
- Note: `falkor-chat` is a hybrid human+AI chat platform over FalkorDB. This CPG capability is a
  **distinct new component** (decided 2026-07-12) — hence this doc lives at repo-root
  `docs/requirements/`. It reuses FalkorDB but is not part of the chat platform.

## User stories
**M1 — producer**
- As an **operator**, I want to run one pipeline over a repo and get a queryable CPG in FalkorDB, so
  that a code graph exists to query at all.

**M2 — consumers (via the `cpg-analysis` skill)**
- As the **`analyst`/`architect` agent**, I want to query the code graph for callers/callees and
  dependency paths, so that I can do **impact analysis** without reading every file by hand.
- As the **`analyst` agent**, I want to trace a value's data-flow back from a symptom and find every
  definition/reference of a symbol, so that I can do **root-cause analysis** on a structured graph.
- As the **`analyst` agent**, I want to find tainted paths from inputs to risky sinks, so that I can
  do **code review** for security-relevant patterns.
- As the **`qa-engineer` agent**, I want to find code reachable from production entrypoints but from
  no test entrypoint, so that I can target **test gaps**.
- As a **falkor-chat workflow / agent node** (later), I want the same recipes, so that automated
  workflows can reason about code structure.

## Functional requirements

### M1 — producer pipeline (delivered ✅)
- **FR-1** ✅ — A pipeline extracts a CPG from source code using **Joern** and loads it into
  **FalkorDB** as a queryable graph.
- **FR-2** ✅ — The stored CPG can answer **caller/callee** queries: who calls a given function, and
  what a given function calls.
- **FR-3** ✅ — The stored CPG can answer **transitive-impact** queries: the up/downstream reach of a
  change across call/dependency chains ("what could break if I change X").
- **FR-4** ✅ — The stored CPG can answer **data-flow** queries: how a value propagates through the
  code (Joern's data-flow edges, `REACHING_DEF`).
- **FR-5** ✅ — The stored CPG can answer **symbol reference** queries: where a symbol is defined and
  all places it is referenced, across files.
- **FR-6** — The CPG is queryable by the **agent team** first (see FR-9); the same queries are later
  reachable by **falkor-chat workflows / agent nodes**.
- **FR-7** ✅ — Extraction is **on-demand / snapshot-based**: a run produces a fresh CPG for the
  target code; no requirement to keep it continuously in sync (auto-sync is a possible later
  extension).
- **FR-8** ✅ — The pipeline targets **this monorepo initially** (Python and JS/TS) but is designed
  to be **generic**, able to extract from arbitrary repositories later.

### M2 — CPG consumer skill (`cpg-analysis`)
- **FR-9** — Agents access the loaded CPG through a **`cpg-analysis` skill** (a lean core plus
  per-task recipes), querying FalkorDB with Cypher (`redis-cli GRAPH.QUERY`). *(Resolves former OQ1;
  chosen over MCP tool / raw Cypher.)*
- **FR-10** — An **impact-analysis** recipe packages FR-2/FR-3 for `analyst` and `architect`
  (callers/callees + transitive up/downstream reach over `CALL`).
- **FR-11** — An **RCA** recipe packages FR-4/FR-5 for `analyst` (data-flow back from a symptom over
  `REACHING_DEF` + cross-file symbol definition/reference).
- **FR-12** — A **code-review** recipe finds tainted paths from inputs to risky sinks (data-flow to
  suspicious calls/patterns) for `analyst`.
- **FR-13** — A **test-gap** recipe finds code reachable from production entrypoints but from no test
  entrypoint (structural reachability — **not** runtime coverage) for `qa-engineer`.
- **FR-14** — The recipes cite a **single canonical CPG schema reference**
  (`skills/joern-cpg/references/cpg-model.md`: node/edge labels, UPPER_CASE property keys, `id`,
  real booleans) rather than duplicating the schema per recipe.

## Out of scope
- **RAG / vector indexing of code** — not a goal for now (may come later).
- **Continuous auto-sync** with a live repo — snapshots only for the first version.
- **Runtime code coverage** — the test-gap recipe (FR-13) reasons about *structural reachability*
  from the CPG, not executed line/branch coverage (that needs a coverage tool, out of scope here).
- **falkor-chat workflow/agent-node consumption** — the same recipes serve it later, but wiring the
  CPG into falkor-chat workflows is a separate future milestone.

## Acceptance criteria
- **AC-1** ✅ — Given a target repo, when the pipeline is run, then a CPG for that code exists in
  FalkorDB and is queryable.
- **AC-2** — Given a function in the CPG, when the analyst asks (via the impact recipe) for its
  callers and callees, then both are returned correctly.
- **AC-3** — Given a proposed change to a symbol, when the analyst/architect asks for its transitive
  impact, then the up/downstream reach is returned across call/dependency chains.
- **AC-4** — Given a value/parameter, when the RCA recipe's data-flow query is run, then the
  propagation path(s) are returned.
- **AC-5** — Given a symbol, when the analyst asks where it is defined and referenced, then all
  cross-file definitions and references are returned.
- **AC-6** — Given a loaded CPG, when an agent invokes the `cpg-analysis` skill, then it can run the
  recipe's Cypher against FalkorDB and get correct results without hand-knowing the schema.
- **AC-7** — Given code where an input reaches a risky sink, when the code-review recipe runs, then
  the tainted path(s) are reported (and clean code reports none).
- **AC-8** — Given production and test entrypoints, when the test-gap recipe runs, then code
  reachable from production but from no test entrypoint is listed.

## Open questions
- OQ1 — ~~How does the analyst *access* the CPG?~~ **RESOLVED 2026-07-18** — a **skill**
  (`cpg-analysis`); now **FR-9**.
- OQ4 — ~~Do the code-review and test-gap recipes (and `qa-engineer` as a consumer) belong in
  scope?~~ **RESOLVED 2026-07-18 (user)** — **yes, both are in scope**; now **FR-12/FR-13** with
  **AC-7/AC-8**. `qa-engineer` added as a named consumer.
- OQ2 — Naming/structure of the component (e.g. a `code-graph/` code dir vs. living entirely as the
  `joern` agent + skills) — for the architect.
- OQ3 — Which Joern language frontends are needed for the initial monorepo targets (Python, JS/TS)
  and how well they cover those languages — design-time verification.

## Decision log
2026-07-12 — Idea → run Joern to extract a CPG and represent it in FalkorDB.
2026-07-12 — Purpose → **code navigation & impact analysis** for software development; **RAG indexing
of code is out of scope for now**. Consumers → **`analyst` agent first**, then **falkor-chat
workflows / agent nodes**.
2026-07-12 — Required queries → all four: callers/callees, transitive impact, data-flow, cross-file
symbol references (FR-2..FR-5).
2026-07-12 — Scope → **this monorepo (Python + JS/TS) initially, but built generic** for arbitrary
repos later. Freshness → **snapshot / on-demand to start; auto-sync a possible later extension**.
2026-07-12 — Placement → **new top-level component** (serves the agent team *and* falkor-chat, distinct
from the chat platform). Doc relocated from `falkor-chat/docs/requirements/` to repo-root
`docs/requirements/` accordingly.
2026-07-17 — **M1 delivered ✅** — producer pipeline (`joern` agent + `joern-cpg` skill), live-load
verified; commit `b2b9a6e`. Satisfies FR-1..FR-8 / AC-1. See [`../HISTORY.md`](../HISTORY.md).
2026-07-18 — Access mechanism (OQ1) → **skill** `cpg-analysis` (one skill, per-task recipes), approved
by user → **FR-9**.
2026-07-18 — **M2 scope confirmed (user)** → all four recipes are in scope: impact-analysis (FR-10)
and RCA (FR-11) were already covered by FR-2..FR-5; **code-review (FR-12) and test-gap (FR-13),
previously flagged as scope extensions under OQ4, are now approved** — `qa-engineer` added as a named
consumer, runtime coverage explicitly excluded. Tracked as **M2 / C-201…C-208** in
[`../BACKLOG.md`](../BACKLOG.md); the blocking requirements pass (C-200) is thereby resolved.
