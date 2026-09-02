---
name: cpg-analysis
description: >-
  Query an already-loaded Joern Code Property Graph (CPG) in FalkorDB with read-only
  Cypher — through the mcp__cypher__query MCP tool in Claude Code, or redis-cli
  GRAPH.QUERY as the documented fallback — to answer structured code questions
  without reading files: impact analysis (callers/callees + transitive reach),
  root-cause analysis (data-flow slices + cross-file symbol def/ref), code review
  (input to risky-sink taint), and test-gap analysis (prod code no test reaches).
  Use when analyst, architect, qa-engineer, coder, tdd-engineer, frontend-engineer, or
  security-expert need call-graph or data-flow answers over a codebase. Each task is a copy-adaptable
  recipe under references/ — change
  one parameter (the target FULL_NAME or NAME) and run. The single CPG schema lives
  in skills/joern-cpg/references/cpg-model.md; this skill does not restate it.
  Requires a CPG already built and loaded by the joern-cpg pipeline; building or
  loading one routes to graph-dba (an on-demand capability, not routine).
allowed-tools: mcp__cypher__query, Bash, Read
---

# cpg-analysis — query a loaded CPG in FalkorDB

You are reading an **already-loaded** Code Property Graph (produced by the
`joern-cpg` pipeline, ingested into FalkorDB as Cypher). This skill teaches the query surface
and the shared traversal idioms; each of the four analyses is a self-contained
recipe you open on demand and adapt by changing **one parameter**.

**Schema is not repeated here.** Node labels, edge types, property keys, and the
topology gotchas live once in
[`../joern-cpg/references/cpg-model.md`](../joern-cpg/references/cpg-model.md)
(read its **"Consumer-query facts"** section before writing traversals). This
skill carries only the query idioms that stand on that schema.

## 1. Run a query

Send Cypher through the **`mcp__cypher__query`** MCP tool. It takes exactly two
parameters and no shell is involved:

| Parameter | What you pass |
|---|---|
| `graph` | the FalkorDB graph key — **caller-supplied, never hardcoded** |
| `cypher` | the Cypher text itself, verbatim; multi-line and indentation are fine, and nothing needs quoting or escaping |

```
mcp__cypher__query(
  graph  = "cpg_yourrepo",        # <-- the caller's graph key; do NOT assume a value
  cypher = "MATCH (m:METHOD) WHERE m.NAME = 'post_message'
            RETURN m.FULL_NAME, m.FILENAME"
)
```

- **The tool is read-only** (`GRAPH.RO_QUERY`): FalkorDB rejects a write
  server-side, and a mistyped graph name cannot silently create an empty graph.
  All recipes here are `MATCH … RETURN` anyway — never issue `CREATE`/`SET`/
  `DELETE` against a shared analysis graph.
- **Finding the graph name.** It comes from whoever loaded the CPG. There is
  deliberately **no `list_graphs` tool** (the surface is one tool, two
  parameters). Before falling back to discovery, try a **first guess**: a
  loaded CPG here follows the pattern **`cpg_<component>`**, the
  component-directory name with hyphens stripped (`falkor-chat` →
  `cpg_falkorchat`). Confirm the graph you land on actually covers the code
  you mean — a component can be renamed or retired while a graph built from
  its old contents keeps the old name. Send a cheap query
  against that guessed name — e.g. `MATCH (n) RETURN count(n)`, or the
  freshness recipe itself, which doubles as an existence probe (see
  [`references/freshness.md`](references/freshness.md)). A **hit** means the
  CPG exists and hands you its freshness in the same call. A **miss** falls
  back to the two remaining paths: the `redis-cli GRAPH.LIST` fallback below,
  or sending a query with the wrong name so the tool's not-found error lists
  the graphs currently loaded — check that list for a differently named match
  before concluding there is none.
- **No graph listed, or FalkorDB is down?** This skill only *queries* an
  already-loaded CPG — it does not build or load one. Building a CPG from source
  and loading it into FalkorDB is **`graph-dba`'s** job, on demand, via the
  `joern-cpg` pipeline. Route the "there is no CPG yet" case there; do not
  attempt to parse/export/load here. The tool's error text says the same thing.

**Plans: `EXPLAIN` yes, `PROFILE` no.**

- Prefix `cypher` with `EXPLAIN ` to get the query plan instead of results —
  worth doing before a heavy variable-length traversal.
- **`PROFILE` is refused by the tool**, deliberately. FalkorDB *silently ignores*
  an `EXPLAIN`/`PROFILE` prefix inside `GRAPH.QUERY`/`GRAPH.RO_QUERY` and runs
  the query for real, so passing it through would hand you **results where you
  asked for a profile** — a wrong answer, not an error. Measured profiling stays
  on `redis-cli … GRAPH.PROFILE` (`graph-dba`'s territory), which also *executes*
  the query, writes included.
- ⚠ **Divergence to remember:** the `EXPLAIN` prefix is a convention of this
  tool. Copy the same string into `redis-cli GRAPH.QUERY` and it **executes**.

**Truncation is display-only — with one FalkorDB-level exception.** Long results are capped for
rendering — a maximum number of rows, a per-cell character cap, and a total-size cap (the current
defaults and their env overrides live in the repo's `cypher-mcp/README.md`, next to the server). The
query itself always runs in full, and below 10,000 true rows the reported row count is exact; when
one of this tool's own caps binds, the output says which one and how many rows of how many are
shown. **But** FalkorDB's own server-side `RESULTSET_SIZE` (default 10000) silently caps the result
*beneath* this tool, even against an explicit larger `LIMIT` — at or above 10k rows, treat `rows=`
as "at least this many," not the true total, and re-query with a narrowing predicate to get an
exact count. Because a truncated sample of an **unordered** result set is arbitrary, narrow
deliberately — add `ORDER BY`, a projection, a `LIMIT`, or an aggregate — rather than reasoning from
the first N rows.

**Fallback — `redis-cli`.** Use it outside Claude Code or when the tool is
unavailable. The MCP wiring is **Claude-Code-only today** (`.mcp.json` at the
repo root; OpenCode and Kiro configure MCP through their own files and this repo
wires neither), so under those harnesses this is *the* path, not a degraded one:

```bash
GRAPH=cpg_yourrepo            # <-- the caller's graph key; do NOT assume a value
HOST=${FALKORDB_HOST:-127.0.0.1}
PORT=${FALKORDB_PORT:-6379}

redis-cli -p "$PORT" GRAPH.LIST          # which graphs are loaded
# read-only query; --no-raw makes multi-column output legible
redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$GRAPH" \
  "MATCH (m:METHOD) WHERE m.NAME = 'post_message' RETURN m.FULL_NAME, m.FILENAME" \
  --no-raw
```

Two shell-path cautions the tool does not have: Cypher lives inside a shell
argument, so quotes/`$`/newlines must be defended; and `GRAPH.QUERY` against a
**non-existent** graph *materialises* that key — check `GRAPH.LIST` first, or use
`GRAPH.RO_QUERY`, which does not.

**Parsing `--no-raw` output yourself (e.g. to diff it against the MCP tool's rendering):**
`--no-raw` prints a **flat, one-scalar-per-line stream** — header names, then every row's cells in
order, no RESP nesting or per-row grouping — followed by two stats lines. Regroup by dropping the
first *N* lines (N = column count) and the last 2, then chunking the remainder into N-tuples. The
trap: a `null` cell and an empty-string cell **both render as a blank line**, so filtering blank
lines out before regrouping silently drops cells and shifts every later row into the wrong column
— don't filter; count positionally instead.

> **Coverage boundary — verified Python-only.** Every recipe here was
> live-verified against a **Python** CPG (`pysrc2cpg`, the `cpg_falkorchat` graph);
> the JS/TS frontends were **not** exercised. The queries are label/property-driven
> and therefore language-agnostic in principle, but the recorded "Verified" results
> and any file-prefix / naming assumptions reflect Python. Treat correctness claims
> against a JS/TS CPG as unverified until re-run, and re-check the schema in
> `cpg-model.md` for that frontend.
>
> **The `Verified` figures in the recipes are dated evidence, not targets.** Each
> was measured against a specific build of `cpg_falkorchat` (M2, 2026-07-19); the
> source tree has moved since, so counts and names legitimately differ on a
> rebuilt graph. Use them as shape/sanity signals and confirm against the source
> — never iterate a query until it reproduces an old number.

## 2. The five gotchas that silently return wrong/empty results

Full detail in `cpg-model.md`; the minimum you trip on:

1. **Property keys are UPPER_CASE**, only `id` is lowercase. `m.name` returns
   `null` silently — use `m.NAME`, `m.FULL_NAME`, `m.FILENAME`, `m.LINE_NUMBER`,
   `m.IS_EXTERNAL`, `c.CODE`.
2. **Booleans are real booleans**: the hazard is **quoting**, not case — `WHERE m.IS_EXTERNAL =
   false` and `= False` both work (FalkorDB accepts Cypher boolean literals case-insensitively);
   `= 'false'` is what breaks, because it's a string comparison, never true against a real boolean.
3. **`CALL` is a call-*site* node, not a method→method edge.** Callee =
   `(:CALL)-[:CALL]->(:METHOD)`; caller = `(:METHOD)-[:CONTAINS]->(:CALL)`.
4. **`FILENAME` is reliable only on `METHOD`/`TYPE_DECL`.** `CALL`, `IDENTIFIER`,
   `LOCAL` carry empty `FILENAME` — resolve a node's file via its enclosing
   method: `(owner:METHOD)-[:CONTAINS]->(n) RETURN owner.FILENAME`.
5. **`REACHING_DEF` (data flow) is intraprocedural** — it stops at call-site
   arguments and does not cross into a callee. Crossing calls is a deliberate,
   sparser step (see the interprocedural note below and the rca/code-review recipes).
6. **`rows=` in the `cypher` MCP tool's own accounting is exact below 10,000 true rows, but
   FalkorDB's server-side `RESULTSET_SIZE` (default 10000) silently caps it above that** —
   even against an explicit larger `LIMIT`, with no marker distinguishing "the true count" from
   "the cap." At/above 10k, re-query with a narrowing predicate rather than trusting the figure.
   Also: `METHOD.CODE` holds only short signatures — the wide source text lives on `LITERAL`/
   `BLOCK`/`CALL` nodes (docstrings are `LITERAL`), so a payload-size probe against `METHOD.CODE`
   binds far too early to exercise a char-cap.

**A query that returns nothing may mean "no graph loaded" or "can't reach FalkorDB at all," and
those look identical from inside a subagent** — a connectivity failure (container down/restarting)
and an absent graph both surface as an error with no data. Probe reachability explicitly (a
short-`timeout` `GRAPH.LIST`/`RETURN 1`) before concluding a graph is simply missing, and report
which one it actually was rather than silently treating "no data" as the answer.

## 3. Shared traversal idioms (the building blocks every recipe reuses)

All use a parameterized target. Substitute `$fn` (a short `NAME`) or `$full`
(a `FULL_NAME`) for your target and run.

> **`$fn` / `$full` are literal text you paste into the query string, not bound
> Cypher parameters.** Replace the token with the actual quoted value before you
> send the query (e.g. edit `$full` to `'falkorchat/services.py:<module>.Services.post_message'`).
> **Neither path binds Cypher parameters** — `redis-cli GRAPH.QUERY` has no
> `--param`-style binding, and `mcp__cypher__query` takes only `graph` and `cypher`
> (a `params` argument would be a third parameter the tool deliberately does not
> have). So do **not** try to pass them as parameters — an un-substituted `$full`
> left in the string will error or match nothing. Always quote the substituted
> value and keep it a fixed literal (these are analysis queries over trusted
> inputs, not user-supplied strings).

**Anchor a target method** (short name may collide across classes; disambiguate
by `FILENAME`):
```cypher
MATCH (m:METHOD) WHERE m.NAME = 'post_message' AND m.FILENAME = 'falkorchat/services.py'
RETURN m.FULL_NAME, m.LINE_NUMBER, m.IS_EXTERNAL
```

**Already have a short, finite list of candidate names? Batch it into one query.**
When the caller already knows the small set of symbol names it needs (e.g. every
function named in a plan's "files to read" list), check all of them in a single
`WHERE m.NAME IN [...]` rather than issuing one query per name — one round trip
returns every match's `FILENAME`/`LINE_NUMBER` together, letting production vs.
test-file definitions be told apart immediately:
```cypher
MATCH (m:METHOD) WHERE m.NAME IN ['_run_agent_node', '_handle_tool_call', '_execute_step']
RETURN m.NAME, m.FILENAME, m.LINE_NUMBER
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
interprocedural taint, escalate to Joern's `reachableBy` in the REPL
(`graph-dba`, driving the `joern-cpg` skill) — pure Cypher here is a documented
approximation.

## 4. Navigation — open the recipe for your task

| You need to… | Consumer | Open |
|---|---|---|
| Find callers/callees of a function and what a change transitively reaches | analyst, architect, coder, tdd-engineer, frontend-engineer | [`references/impact-analysis.md`](references/impact-analysis.md) |
| Trace a bad value back to its definitions; find a symbol's defs + cross-file refs | analyst | [`references/rca.md`](references/rca.md) |
| Check whether external input can reach a risky sink (taint) | analyst | [`references/code-review.md`](references/code-review.md) |
| List production code no test structurally reaches | qa-engineer | [`references/test-gap.md`](references/test-gap.md) |
| Judge how current a loaded CPG is before trusting it | `teco`, at dispatch (2026-08-19: centralized there — a standalone-invoked consumer no longer runs this check itself) | [`references/freshness.md`](references/freshness.md) |

Each recipe states its purpose, the one parameter to change, the parameterized
Cypher, the expected shape of results, and its known limits. Recipes assume the
schema in `cpg-model.md` and the idioms above; they do not restate them.
