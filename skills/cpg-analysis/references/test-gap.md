# Recipe: test-gap analysis

> Back to [`../SKILL.md`](../SKILL.md) · schema in
> [`../../joern-cpg/references/cpg-model.md`](../../joern-cpg/references/cpg-model.md).
> **Consumer:** qa-engineer. **Covers:** FR-13 / AC-8.

**Purpose.** List production methods that **no test structurally reaches** — a
lower bound on untested code. **Change the classification parameters** for the
codebase: the **test-file prefix** (`'tests/'`), the **test-method prefix**
(`'test_'`), and the **production prefix** (`'falkorchat/'`).

> **How this satisfies AC-8 (complement, not forward-reach).** The plan framed
> AC-8 as "reachable from prod entrypoints minus reachable from test entrypoints."
> This recipe instead computes the **complement of the test-reach closure**: all
> first-party production methods that the `test_*`-`NAME`-closure does not reach.
> That is a deliberate adaptation — framework entrypoints (routes, MCP tools) are
> **not statically connected** to their handlers in this frontend, so a
> forward-reach-from-prod-entrypoint design would be crippled. The trade-off: the
> complement also includes prod code reachable from *no* entrypoint at all (dead
> code), so this is strictly "prod code no test reaches," a **lower bound** on
> untested code — not "prod-entrypoint-forward minus test."

## Why name-based reachability here (not the resolved `CALL` edge)

Tests drive the app the way a framework does — over an HTTP `TestClient`, via
pytest collection, through fixtures — so there is **no static `CALL` edge** from a
`test_*` method to a route handler or MCP tool. The resolved `CALL` edge is also
too sparse for cross-object calls (see cpg-model.md). So "reachable from a test"
is computed by a **transitive `CALL.NAME` closure**: a method named `X` is
test-reached if a test calls `X`, or calls something that (transitively) calls `X`.
Transitivity matters — a helper with **no direct** test caller can still be
reached *through* a tested method, and must not be flagged.

## The query — production methods outside the test-reach closure (AC-8)

Bounded to 3 expansion levels (extend by copying the middle block for more depth;
3 was sufficient on `cpg_falkorchat`). Each level collects the `CALL.NAME`s
invoked by methods already in the closure. `<operator>.*` synthetics are excluded.

```cypher
// L1: names called directly by any test method
MATCH (t:METHOD) WHERE t.FILENAME STARTS WITH 'tests/' AND t.NAME STARTS WITH 'test_'
MATCH (t)-[:CONTAINS]->(c1:CALL) WHERE NOT c1.NAME STARTS WITH '<operator>'
WITH collect(DISTINCT c1.NAME) AS L1
// L2: names called by production methods whose NAME is in L1
MATCH (m2:METHOD) WHERE m2.NAME IN L1 AND m2.FILENAME STARTS WITH 'falkorchat/'
MATCH (m2)-[:CONTAINS]->(c2:CALL) WHERE NOT c2.NAME STARTS WITH '<operator>'
WITH L1, collect(DISTINCT c2.NAME) AS L2
WITH [x IN L2 WHERE NOT x IN L1] + L1 AS L12
// L3: one more expansion
MATCH (m3:METHOD) WHERE m3.NAME IN L12 AND m3.FILENAME STARTS WITH 'falkorchat/'
MATCH (m3)-[:CONTAINS]->(c3:CALL) WHERE NOT c3.NAME STARTS WITH '<operator>'
WITH L12, collect(DISTINCT c3.NAME) AS L3
WITH L12 + [x IN L3 WHERE NOT x IN L12] AS reached
// gap = first-party production methods whose NAME is not in the closure
MATCH (g:METHOD)
  WHERE g.FILENAME STARTS WITH 'falkorchat/' AND g.IS_EXTERNAL = false
    AND NOT g.NAME CONTAINS '<' AND NOT g.NAME STARTS WITH '__'
    AND NOT g.NAME IN reached
RETURN DISTINCT g.NAME AS untestedMethod, g.FILENAME AS file, g.LINE_NUMBER AS line
ORDER BY file, line
```

> **FalkorDB idiom note.** Do **not** fold an aggregation and a reference to a
> prior list into the same `WITH` (e.g. `WITH acc + [x IN collect(...) …]`) — this
> build raises `_AR_EXP_UpdateEntityIdx: Unable to locate a value with alias`.
> Split into two `WITH` steps (`WITH acc, collect(...) AS lvl` then
> `WITH acc + [x IN lvl WHERE NOT x IN acc]`), as above.

## Sanity-check the prod/test split (counts only — not a closure seed)

```cypher
MATCH (t:METHOD) WHERE t.FILENAME STARTS WITH 'tests/' AND t.NAME STARTS WITH 'test_'
RETURN count(*) AS testEntrypoints;      // verified: 336
MATCH (m:METHOD) WHERE m.FULL_NAME CONTAINS 'build_router.' AND m.FILENAME = 'falkorchat/api.py'
RETURN count(*) AS routeHandlers;        // verified: 17
MATCH (m:METHOD) WHERE m.FILENAME = 'falkorchat/mcp.py' RETURN count(*) AS mcpMethods;  // verified: 10
```

## Expected shape & verification

One row per untested production method (file + line). The `RETURN` yields one row
per **file:line**, so read two numbers: **Verified** (`cpg_falkorchat`): **39 rows**
(`count(*)`) across **32 distinct method names** (`count(DISTINCT g.NAME)`). The gap
between them is real — a `NAME` can flag at several sites (e.g. `ping` at both
`repository.py:115` and `services.py:152`; `record` ×3 in `executor.py`), and each
site is its own row. Quote **39** as the untested-method-site count and **32** as
the distinct-name count; do not collapse them to one figure. Anchor checks (all as
specified):

| Method | Flagged as gap? | Why |
|---|---|---|
| `Services.ping` | **yes** | no test reaches it, even transitively — a true prod-only path |
| `_safe_respond` (api.py L48) | **yes** | prod-only background responder |
| `_safe_run_workflow` (api.py L71) | **yes** | prod-only background worker |
| `_serialize_opaque` | **no** (correctly excluded) | transitively test-reached: `test_* → publish_workflow_def → _serialize_opaque` |

The `_serialize_opaque` case is the reason the closure must be transitive: a
one-hop "no direct test caller" check would wrongly flag it.

## Limits

- **Structural reachability, not runtime coverage.** A method *on* a test-reached
  path may still be functionally untested (asserted-through but never exercised
  for the branch that matters). Runtime line/branch coverage is out of scope —
  use a coverage tool for that.
- **Name-based ⇒ possible false negatives from `NAME` collisions.** If a prod
  method shares a `NAME` with something a test calls, it is treated as reached
  even if it is a different method. Spot-check flagged/expected methods whose
  `NAME` is generic; disambiguate by `FULL_NAME` when it matters.
- **Depth-bounded.** 3 levels caught the anchors here; a deep prod call chain
  could need more. Add expansion blocks and confirm the closure size stabilises
  (`RETURN size(reached)`), or the gap count stops shrinking.
- **`__dunder__` and synthetic (`<…>`) methods are excluded** as non-targets;
  adjust the `WHERE` if your codebase tests dunders directly.
