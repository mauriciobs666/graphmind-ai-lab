# Change History — CPG code-graph component

> Dated log of actual changes to the repo-root **CPG / code-graph** component (Joern → FalkorDB).
> Most recent first. Forward-looking work lives in [`BACKLOG.md`](./BACKLOG.md); requirements in
> [`requirements/joern-cpg-pipeline.md`](./requirements/joern-cpg-pipeline.md) and, for the read
> path, [`requirements/cpg-query-access.md`](./requirements/cpg-query-access.md).

## 2026-07-25 — M3: CPG query access — the MCP read path ✅

Asking the code graph a question is now **one tool call**, not a hand-assembled shell command.
`mcp__cpg__query(graph, cypher)` replaces `redis-cli GRAPH.QUERY` on the CPG **read** path:
the graph key and the Cypher text are parameters, so nothing has to survive a shell layer.

- **`cpg` MCP server** (`cpg/mcp/`) — a Python **FastMCP** stdio server exposing **exactly one**
  read-only tool over `GRAPH.RO_QUERY`, with `setup.sh`, `run.sh`, a README and a pytest suite
  (**53 offline / 7 live** — the component's only regression signal). Semantics: read-only;
  **`EXPLAIN`-only, `PROFILE` removed** (decision D4 — `GRAPH.PROFILE` *executes* the query
  including writes, so routing to it from a `readOnlyHint=True` tool was a read-only hole;
  `graph-dba` keeps `PROFILE` via `redis-cli`); the `PROFILE` refusal is comment-blind, because
  `/* c */ PROFILE …` through raw `GRAPH.RO_QUERY` really does return results; a typo'd graph name
  returns a curated not-found listing the loaded graphs and **does not materialise an empty key**
  (closing the known FalkorDB quirk); truncation is **display-only** (200 rows / 300-char cells /
  30,000 chars) with the notice repeated as the first *and* last line.
- **Wiring** — repo-root `.mcp.json` (`bash -c 'exec "$CLAUDE_PROJECT_DIR/cpg/mcp/run.sh"'`, no
  absolute paths) plus `enabledMcpjsonServers` in `.claude/settings.json`. This is the repo's
  **first MCP wiring, and it is Claude-Code-only** — OpenCode and Kiro configure MCP through their
  own files and neither is wired (backlog **C-310**), so `redis-cli GRAPH.QUERY` remains their only
  path and stays documented as the fallback everywhere.
- **Consumers** — `mcp__cpg__query` added to the `analyst` and `architect` `tools:` allowlists
  (without which the tool is invisible to them; `qa-engineer` declares none and inherits) and to
  `skills/cpg-analysis/SKILL.md` `allowed-tools`, with §1 rewritten around the tool.
  `skills/agent-standards/claude-code.md` §MCP was rewritten and an **OpenCode MCP** section added,
  recording the divergences and the cross-tool rule that **MCP wiring does not port**.
- **`joern-cpg-pipeline.md` FR-9 reversed** — it had chosen `redis-cli` *"over MCP tool"*; it now
  routes through `mcp__cpg__query` and points at `docs/requirements/cpg-query-access.md`, with
  `redis-cli` as the documented fallback (**AC-4**).
- **Build, not buy** — the official `@falkordb/mcpserver` v1.3.0 exposes 7 tools including
  `delete_graph` with no tool filtering (a flat FR-2 violation) and needs Node ≥18, absent on the
  Linux side; **reversal trigger:** an upstream server that can be filtered to one read-only tool.
- **CPG rebuilt** (stakeholder-authorised destructive rebuild, decision D1) from
  `falkor-chat/server/{falkorchat,tests}`. **New baseline for `cpg_falkorchat`: 110,048 nodes ·
  734,929 edges · 1,968 METHODs · 1,019 test-file METHODs (512 `test_*`) · direct callers of
  `post_message` = 21 · test-gap = 50 rows / 43 distinct names** (the pair does not collapse to one
  number).
  ⚠ **These figures supersede the M2 numbers below** (79,581 nodes / 522,182 edges; test-gap 39
  rows / 32 distinct names). Those describe a specific build of a *moving* source tree — 8 commits
  have landed in `falkor-chat/server` since — not a property of the access mechanism. They are not
  a target and must not be iterated toward.
  The M2 entry stays as written; it was true when written.
- **Acceptance: PASS WITH DEFECTS** (`docs/test-reports/cpg-query-access-report.md`, 23 cases,
  22 pass / 1 fail). **AC-1** (one tool call, zero shell quoting; 1 tool / 2 parameters at protocol
  level), **AC-2** (multi-line ≡ single-line, byte-identical row bodies) and **AC-4** pass.
  The one failing case (TP-010) was **DEF-1**, a conflict between two approved specs — AC-3's
  *"byte-identical value sets"* vs plan §4.4's `repr` rendering for list/map cells, which cannot
  both hold for any query projecting a non-scalar. 5 of 6 tool-vs-`redis-cli` pairs were
  byte-identical; the sixth (RCA data-flow, projecting `labels()`) returned the same 44 rows in the
  same order with identical values and differed only in list syntax.
- **DEF-1 ruled the same day (stakeholder decision D5, Option A) → C-313 closed.** **AC-3 is
  narrowed to values + row counts + ordering**, excluding the display rendering of non-scalar cells,
  with plan §4.4 named as the authority for how a cell is rendered — a **specification
  reconciliation, not a code fix**: the alternative (re-rendering lists `redis-cli`-style) was
  rejected and **no source changed**. **AC-3 passes** under the reconciled wording, so
  **AC-1…AC-4 are all met**. The test report keeps its original results and verdict as the dated
  execution record, with the ruling appended as an addendum. DEF-2/DEF-3/DEF-5 remain low-severity
  cleanups (C-314/C-315/C-316).
- **Known limits:** Claude-Code-only wiring; read-only; `EXPLAIN`-only; display-only truncation;
  non-scalar cell rendering diverges from `redis-cli`; the transitive upward call-closure query is
  deferred to **C-308** (D3 — this feature changed how Cypher is *transmitted*, not how powerful it
  is). Also learned, and bigger than this feature: `FILENAME` is **relative to the Joern parse
  root**, so the parse root alone silently decides whether every `STARTS WITH 'tests/'` recipe
  filter works — and the failure is invisible in node/edge counts. That, not the missing test
  sources, is why the pre-rebuild graph was useless; a post-load check is filed as **C-312**.

Delivers M3 (FR-1…FR-6 / AC-1…AC-4 of `docs/requirements/cpg-query-access.md`, superseding FR-9 of
`joern-cpg-pipeline.md`) — items **C-301…C-307**, follow-ups **C-308…C-319** in
[`BACKLOG.md`](./BACKLOG.md). Consumer skill was M2 (2026-07-19); producer pipeline M1 (2026-07-17).

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
