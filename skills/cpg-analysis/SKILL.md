---
name: cpg-analysis
description: >-
  Query an already-loaded Joern Code Property Graph (CPG) in FalkorDB with Cypher
  (redis-cli GRAPH.QUERY) to answer structured code questions without reading files:
  impact analysis (callers/callees + transitive reach), root-cause analysis
  (data-flow slices + cross-file symbol def/ref), code review (input to risky-sink
  taint), and test-gap analysis (prod code no test reaches). Use when analyst,
  architect, or qa-engineer need call-graph or data-flow answers over a codebase.
  Each task is a copy-adaptable recipe under references/ — change one parameter
  (the target FULL_NAME or NAME) and run. The single CPG schema lives in
  skills/joern-cpg/references/cpg-model.md; this skill does not restate it.
  Requires a CPG already built and loaded by the joern pipeline; building or
  loading one routes to the joern agent.
allowed-tools: Bash, Read
---

# cpg-analysis — query a loaded CPG in FalkorDB

You are reading an **already-loaded** Code Property Graph (produced by the `joern`
pipeline, ingested into FalkorDB as Cypher). This skill teaches the shared
connection + traversal idioms; each of the four analyses is a self-contained
recipe you open on demand and adapt by changing **one parameter**.

**Schema is not repeated here.** Node labels, edge types, property keys, and the
topology gotchas live once in
[`../joern-cpg/references/cpg-model.md`](../joern-cpg/references/cpg-model.md)
(read its **"Consumer-query facts"** section before writing traversals). This
skill carries only the query idioms that stand on that schema.

## 1. Connect and run a query

The graph is a named Redis key. **The graph name is caller-supplied** — never
hardcode it. Get it from whoever loaded the CPG (`redis-cli GRAPH.LIST` shows
the loaded graphs). Examples below use `$GRAPH`.

```bash
GRAPH=cpg_yourrepo            # <-- the caller's graph key; do NOT assume a value
HOST=${FALKORDB_HOST:-127.0.0.1}
PORT=${FALKORDB_PORT:-6379}

# read-only query; --no-raw makes multi-column output legible
redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$GRAPH" \
  "MATCH (m:METHOD) WHERE m.NAME = 'post_message' RETURN m.FULL_NAME, m.FILENAME" \
  --no-raw
```

- Confirm the graph exists first: `redis-cli -p "$PORT" GRAPH.LIST`.
- **No graph listed, or FalkorDB is down?** This skill only *queries* an
  already-loaded CPG — it does not build or load one. Building a CPG from source
  and loading it into FalkorDB is the **`joern` agent's** job (the `joern-cpg`
  pipeline). Route the "there is no CPG yet" case there; do not attempt to
  parse/export/load here.
- All recipes are **read-only** (`MATCH … RETURN`). Never issue `CREATE`/`SET`/
  `DELETE` against a shared analysis graph.
- Heavy variable-length traversals: prepend `GRAPH.EXPLAIN` (plan) or
  `GRAPH.PROFILE` (measured) instead of `GRAPH.QUERY` to inspect cost.

> **Coverage boundary — verified Python-only.** Every recipe here was
> live-verified against a **Python** CPG (`pysrc2cpg`, the `cpg_falkorchat` graph);
> the JS/TS frontends were **not** exercised. The queries are label/property-driven
> and therefore language-agnostic in principle, but the recorded "Verified" results
> and any file-prefix / naming assumptions reflect Python. Treat correctness claims
> against a JS/TS CPG as unverified until re-run, and re-check the schema in
> `cpg-model.md` for that frontend.

## 2. The five gotchas that silently return wrong/empty results

Full detail in `cpg-model.md`; the minimum you trip on:

1. **Property keys are UPPER_CASE**, only `id` is lowercase. `m.name` returns
   `null` silently — use `m.NAME`, `m.FULL_NAME`, `m.FILENAME`, `m.LINE_NUMBER`,
   `m.IS_EXTERNAL`, `c.CODE`.
2. **Booleans are real booleans**: `WHERE m.IS_EXTERNAL = false` (not `'false'`).
3. **`CALL` is a call-*site* node, not a method→method edge.** Callee =
   `(:CALL)-[:CALL]->(:METHOD)`; caller = `(:METHOD)-[:CONTAINS]->(:CALL)`.
4. **`FILENAME` is reliable only on `METHOD`/`TYPE_DECL`.** `CALL`, `IDENTIFIER`,
   `LOCAL` carry empty `FILENAME` — resolve a node's file via its enclosing
   method: `(owner:METHOD)-[:CONTAINS]->(n) RETURN owner.FILENAME`.
5. **`REACHING_DEF` (data flow) is intraprocedural** — it stops at call-site
   arguments and does not cross into a callee. Crossing calls is a deliberate,
   sparser step (see the interprocedural note below and the rca/code-review recipes).

## 3. Shared traversal idioms (the building blocks every recipe reuses)

All use a parameterized target. Substitute `$fn` (a short `NAME`) or `$full`
(a `FULL_NAME`) for your target and run.

> **`$fn` / `$full` are literal text you paste into the query string, not bound
> Cypher parameters.** Replace the token with the actual quoted value before you
> send the query (e.g. edit `$full` to `'falkorchat/services.py:<module>.Services.post_message'`).
> `redis-cli GRAPH.QUERY` has no `--param`-style binding, so do **not** try to pass
> them as parameters — an un-substituted `$full` left in the string will error or
> match nothing. Always quote the substituted value and keep it a fixed literal
> (these are analysis queries over trusted inputs, not user-supplied strings).

**Anchor a target method** (short name may collide across classes; disambiguate
by `FILENAME`):
```cypher
MATCH (m:METHOD) WHERE m.NAME = 'post_message' AND m.FILENAME = 'falkorchat/services.py'
RETURN m.FULL_NAME, m.LINE_NUMBER, m.IS_EXTERNAL
```

**Callers of a method — match call sites by NAME, caller is the container**
(the reliable direction; inbound `CALL`-edge resolution is too sparse to trust):
```cypher
MATCH (caller:METHOD)-[:CONTAINS]->(c:CALL {NAME: 'post_message'})
RETURN DISTINCT caller.FULL_NAME, caller.FILENAME, caller.LINE_NUMBER
ORDER BY caller.FILENAME, caller.LINE_NUMBER
```

**Callees of a method — resolved `CALL` edge** (clean, first-party; misses
dynamic/cross-object dispatch, which is unresolved):
```cypher
MATCH (m:METHOD {FULL_NAME: $full})-[:CONTAINS]->(:CALL)-[:CALL]->(callee:METHOD)
RETURN DISTINCT callee.FULL_NAME
```

**Transitive downstream reach** — what a change to `$full` could break. Only
`:METHOD` nodes are reachable across a `CALL` edge, so terminating at `:METHOD`
over the mixed `CONTAINS|CALL` walk yields true call reach (bound the depth):
```cypher
MATCH (m:METHOD {FULL_NAME: $full})-[:CONTAINS|CALL*1..8]->(reached:METHOD)
WHERE reached.IS_EXTERNAL = false AND reached <> m
RETURN DISTINCT reached.FULL_NAME
```

**Data-flow slice within a method** (`REACHING_DEF`, intraprocedural) — forward
from a parameter, or backward from a symptom node:
```cypher
MATCH (m:METHOD {FULL_NAME: $full})-[:AST]->(p:METHOD_PARAMETER_IN {NAME: 'body'})
MATCH (p)-[:REACHING_DEF*1..12]->(use)
RETURN DISTINCT use.LINE_NUMBER, labels(use), use.CODE ORDER BY use.LINE_NUMBER
```

**Cross-file symbol def & references** — definitions carry `FILENAME`;
references (`IDENTIFIER`/`CALL`) do not, so resolve each to its enclosing method:
```cypher
// definitions
MATCH (d) WHERE d.NAME = 'get_context' AND (d:METHOD OR d:TYPE_DECL OR d:LOCAL)
MATCH (owner:METHOD)-[:CONTAINS|AST]->(d)
RETURN DISTINCT labels(d), owner.FILENAME, d.LINE_NUMBER
```

**Interprocedural boundary (read before rca/code-review).** `REACHING_DEF` stays
inside one method. To follow flow across a call you bridge:
`(callSite:CALL)-[:CALL]->(callee:METHOD)-[:AST]->(param:METHOD_PARAMETER_IN)`,
matching the call-site argument's `ARGUMENT_INDEX` to the param's `INDEX`, then
continue `REACHING_DEF` inside the callee. This is only as complete as the sparse
call resolution: **same-object `self.x()` calls resolve; cross-object dispatch
(e.g. a service calling a repository it holds) does not.** For high-fidelity
interprocedural taint, escalate to Joern's `reachableBy` in the REPL (the `joern`
agent) — pure Cypher here is a documented approximation.

## 4. Navigation — open the recipe for your task

| You need to… | Consumer | Open |
|---|---|---|
| Find callers/callees of a function and what a change transitively reaches | analyst, architect | [`references/impact-analysis.md`](references/impact-analysis.md) |
| Trace a bad value back to its definitions; find a symbol's defs + cross-file refs | analyst | [`references/rca.md`](references/rca.md) |
| Check whether external input can reach a risky sink (taint) | analyst | [`references/code-review.md`](references/code-review.md) |
| List production code no test structurally reaches | qa-engineer | [`references/test-gap.md`](references/test-gap.md) |

Each recipe states its purpose, the one parameter to change, the parameterized
Cypher, the expected shape of results, and its known limits. Recipes assume the
schema in `cpg-model.md` and the idioms above; they do not restate them.
