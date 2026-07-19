# Change History — CPG code-graph component

> Dated log of actual changes to the repo-root **CPG / code-graph** component (Joern → FalkorDB).
> Most recent first. Forward-looking work lives in [`BACKLOG.md`](./BACKLOG.md); requirements in
> [`requirements/joern-cpg-pipeline.md`](./requirements/joern-cpg-pipeline.md).

## 2026-07-19 — M2: CPG consumer skill (`cpg-analysis`) ✅

The **consumer** side of the component: one `cpg-analysis` skill teaches the agent team to
query a loaded CPG in FalkorDB with Cypher (`redis-cli GRAPH.QUERY`), closing the M2 gap.

- **`cpg-analysis` skill** (`skills/cpg-analysis/`) — lean `SKILL.md` core (connection idiom,
  silent-failure gotchas, shared traversal idioms: `CONTAINS`→`CALL`, `REACHING_DEF`,
  interprocedural bridge) plus four on-demand `references/` recipes: **impact-analysis**
  (callers/callees + transitive reach), **rca** (data-flow slice + cross-file symbol def/ref),
  **code-review** (taint to risky sinks), **test-gap** (production methods outside the
  test-reach closure). Cites the single canonical schema
  `skills/joern-cpg/references/cpg-model.md` (FR-14) — no duplicated schema; C-201 added a
  "Consumer-query facts" section there.
- **Consumers wired** (C-207): CPG-capability lines added to the `analyst`, `architect`, and
  `qa-engineer` routing descriptions (skill owned by `graph-dba`).
- **Satisfies FR-9…FR-14 / AC-2…AC-8.** Live-verified against `cpg_falkorchat` (79,581 nodes /
  522,182 edges — a Python CPG of `falkor-chat/server/{falkorchat,tests}` via `pysrc2cpg`):
  AC-2 callers=21; AC-3 transitive reach; AC-4 `REACHING_DEF` backward slice; AC-5
  `hybrid_search` cross-file def/ref; **AC-6 independent cold invocation by `analyst` passed on
  all four recipes** (correct results without hand-knowing the schema); AC-7 taint both
  directions (clean=none is a true clean with a documented coverage caveat); AC-8 test-gap =
  **39 untested-method sites / 32 distinct names**.
- **Reviews:** plan Gate-1 (`docs/reviews/m2-cpg-analysis.md`) and skill Gate-2a
  (`docs/reviews/m2-cpg-analysis-skill.md`) both **approve with suggestions**; cobb standards
  Gate-2b **accept**. All suggestions folded in.
- **Known limits:** verification is **Python-only** (JS/TS frontends not exercised);
  `REACHING_DEF` is intraprocedural in this CPG; deep interprocedural taint routes to the
  `joern` agent's `reachableBy`.

Delivers M2 (FR-9…FR-14 / AC-2…AC-8). Producer pipeline was M1 (2026-07-17).

## 2026-07-17 — M1: Producer pipeline (CPG build → FalkorDB load) ✅

First milestone: the **producer** side of the component — turn any source repository into a Code
Property Graph and materialize it in FalkorDB so the code graph is traversable with Cypher.
Delivered as commit `b2b9a6e` and **live-load verified**.

- **`joern` agent** (`claude/joern/`) — CPG specialist that operates the Joern toolset in the local
  Linux environment: builds CPGs with `joern-parse`, queries via the REPL/CPGQL (AST·CFG·CDG·DDG·PDG,
  call graphs, data-flow & taint), exports (neo4jcsv), transforms to FalkorDB-dialect Cypher, and
  ingests end-to-end.
- **`joern-cpg` skill** (`skills/joern-cpg/`) — the scripts and contract the agent drives:
  `pipeline.sh` (build → export → transform → optional load), the CPG→FalkorDB model (shared
  `:CpgNode` label + `CpgNode(id)` index, UPPER_CASE property keys, real booleans), and a CPGQL
  cheat-sheet. Schema/model reference: `skills/joern-cpg/references/cpg-model.md`.
- **Satisfies FR-1** (extract a CPG and load it into FalkorDB) and **AC-1** (a run yields a
  queryable CPG in FalkorDB). Verified against `falkordb v4.18.11`, Joern v4.0.579, JDK 21.

Consumer-side querying (letting `analyst`/`architect`/`qa-engineer` use the loaded CPG) is the next
milestone — **M2**, tracked in [`BACKLOG.md`](./BACKLOG.md) (C-200…C-208).
