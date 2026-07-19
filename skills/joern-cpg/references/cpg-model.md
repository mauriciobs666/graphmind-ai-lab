# CPG schema & FalkorDB model — reference

Deeper notes behind `SKILL.md`. Load when you need the node/edge vocabulary, the
neo4jcsv column contract, or the reasoning behind the FalkorDB mapping.

## The Code Property Graph, layer by layer

A CPG merges several program representations onto **shared nodes**:

| Layer | Edge type(s) | Answers |
|-------|--------------|---------|
| **AST** — abstract syntax tree | `AST` | structure: what contains what |
| **CFG** — control-flow graph | `CFG` | execution order within a method |
| **CDG** — control dependence | `CDG` | which predicate guards a statement |
| **DDG/PDG** — data / program dependence | `REACHING_DEF`, `CDG` | which definition reaches a use |
| **Call graph** | `CALL` | which method calls which |
| **Dominance** | `DOMINATE`, `POST_DOMINATE` | dominator relationships |
| **Structure** | `CONTAINS`, `ARGUMENT`, `RECEIVER`, `PARAMETER_LINK` | membership, call args, method receivers |

`joern-export --repr <X>` extracts one layer (`ast`, `cfg`, `cdg`, `ddg`, `pdg`)
or the whole thing (`cpg` = the modern full graph, `cpg14` = the legacy schema,
`all`). Default in `export-cpg.sh` is `cpg`.

## Node labels you'll see most

`METHOD`, `METHOD_PARAMETER_IN`, `METHOD_PARAMETER_OUT`, `METHOD_RETURN`,
`BLOCK`, `CALL`, `IDENTIFIER`, `LITERAL`, `LOCAL`, `MEMBER`, `FIELD_IDENTIFIER`,
`CONTROL_STRUCTURE`, `RETURN`, `TYPE`, `TYPE_DECL`, `NAMESPACE`,
`NAMESPACE_BLOCK`, `FILE`, `META_DATA`, `METHOD_REF`, `TYPE_REF`, `MODIFIER`.

Common properties: `CODE` (source text; `<empty>` is Joern's placeholder for an
empty string, kept verbatim), `NAME`, `FULL_NAME`, `LINE_NUMBER`,
`COLUMN_NUMBER`, `ORDER`, `ARGUMENT_INDEX`, `TYPE_FULL_NAME`,
`DISPATCH_TYPE`, `SIGNATURE`.

## neo4jcsv column contract (what the transformer parses)

Header row columns:

- `:ID` — globally-unique node id (a large integer). Becomes property `id`.
- `:LABEL` — the node label. Becomes the type label (dropped as a property).
- `:START_ID`, `:END_ID`, `:TYPE` — edge endpoints + relationship type.
- `NAME:int` / `NAME:string` / `NAME:string[]` / `NAME` — a property. Suffix
  gives the type: `int` → integer, `…[]` → array (semicolon-delimited in the
  data), anything else (or no suffix) → string. Empty data cells are dropped.

Data is standard CSV (quoted fields may contain commas and newlines — `CODE`
often does; the transformer uses Python's `csv` reader, which handles this).

Node ids recur across the per-method export files (a boundary node like a
`METHOD` appears in several method dumps); the transformer dedups by id, so a
plain `CREATE` is safe.

## Why the shared `:CpgNode` label

FalkorDB indexes are **per-label**. Edges reference nodes by id only, without
knowing the target's type label, so a label-agnostic lookup is needed. Giving
every node a second, shared `:CpgNode` label lets one index —
`CREATE INDEX FOR (n:CpgNode) ON (n.id)` — back every edge `MATCH (a:CpgNode {id:…})`.
Without it, each edge match is a full scan and a repo-scale load is unusable.

This is the default that ships with the transformer. When the query workload is
known (e.g. "walk call graphs", "trace data flow from parameters"), hand the
model to **`graph-dba`** to add label/edge-type-specific indexes, decide whether
dense layers (`AST`, `CFG`) belong in the same graph, and size the RAM. Write the
resulting design to `<component>/docs/plans/<slug>-graph.md` (graph-dba's
convention) if an implementer will build on it.

## FalkorDB dialect reminders (verify against the pinned build via `graph-dba`)

- FalkorDB is **OpenCypher, not Neo4j** — no APOC/GDS, no `LOAD CSV` runner
  (hence the transformer generates `GRAPH.QUERY`-able statements rather than
  Neo4j's `*_cypher.csv` `LOAD CSV` scripts).
- Index DDL: `CREATE INDEX FOR (n:Label) ON (n.prop)` — **verified accepted** on
  `falkordb v4.18.11` (2026-07-17). If a pinned build wants the older
  `CREATE INDEX ON :Label(prop)` form, adjust the first statement the transformer
  emits. A failing index statement degrades load speed only (the loader
  continues), so this is safe to iterate on.
- **Property names are UPPER_CASE.** The transformer preserves Joern's schema
  property keys verbatim, so on the loaded graph you query `m.NAME`, `m.CODE`,
  `m.FULL_NAME`, `m.IS_EXTERNAL`, `m.LINE_NUMBER` — **not** CPGQL's lowercase
  `.name`/`.code`. A lowercase key silently returns null (no error, just blank
  results). Only `id` (from `:ID`) is lowercase. (Verified 2026-07-17.)
- **Booleans are real booleans.** `IS_EXTERNAL` and other `:boolean` columns load
  as Cypher `true`/`false`, so predicate them as `WHERE m.IS_EXTERNAL = false`,
  not `= 'false'`. (Transformer fixed 2026-07-17 — earlier builds stored them as
  strings.)
- Relationship types cannot be parameterized in Cypher; the transformer inlines
  each Joern edge type (they are safe identifiers like `AST`, `REACHING_DEF`).

## Consumer-query facts (topology gotchas for reading the loaded graph)

> Additive section for **consumers** who query an already-loaded CPG with Cypher
> (the `cpg-analysis` skill). These are **schema/topology facts**, not producer
> semantics — nothing above changes. The Cypher *idioms* that use them live in
> `skills/cpg-analysis/SKILL.md`. **Verified 2026-07-19** against a `pysrc2cpg`
> CPG of `falkor-chat/server` (graph `cpg_falkorchat`, 79.6k nodes / 522k edges,
> falkordb `v4.18.11`). `pysrc2cpg`-specific; re-verify for other frontends.

- **`CALL` is a call-*site* node, not a method→method edge.** The table row
  "Call graph | `CALL`" refers to the call-site node. There are **two** distinct
  `CALL` things: the **`:CALL` node** (one per call expression) and the **`CALL`
  *edge*** from a call-site node to the callee `METHOD`. Direct
  `(:METHOD)-[:CALL]->(:METHOD)` does **not** exist (count 0 on the verified graph).
- **Resolve callees via the `CALL` edge; resolve callers via `CONTAINS`.**
  - Callee of a call site: `(:CALL)-[:CALL]->(callee:METHOD)`.
  - Caller (the method a call site sits in): `(caller:METHOD)-[:CONTAINS]->(:CALL)`.
    So "callers of M" and "callees of M" are **not** one symmetric edge.
- **Inbound call resolution is sparse (frontend limit).** On `pysrc2cpg` the
  `(:CALL)-[:CALL]->(:METHOD)` edge is present for only a minority of call sites
  (~1.3k of ~20k) — reliably for **same-object `self.x()` / same-file** dispatch,
  plus synthetic `…<metaClassAdapter>` wrappers. **Cross-object dispatch**
  (a value holding another class's instance, e.g. a service calling into a
  repository it was handed) is **not** resolved. Consequence: **downstream/callee**
  traversal over the `CALL` edge is trustworthy; **upstream/caller** traversal over
  it is not — match callers by `CALL.NAME` instead.
- **`METHOD_FULL_NAME` (on `CALL` nodes) resolves inconsistently** for the same
  callee: you may see a short form (`Services.post_message`), a full path
  (`falkorchat/services.py:<module>.Services.post_message`), a phantom
  `…<returnValue>.post_message`, or `<unknownFullName>` / `<empty>`. Do **not**
  key joins on it; match call sites by `CALL.NAME` (+ disambiguate by the callee
  `METHOD`'s `FILENAME`). It **is** a real key on `CALL` (confirmed present).
- **`IS_EXTERNAL` is a real boolean on `METHOD`** (confirmed key). `WHERE
  m.IS_EXTERNAL = false` selects first-party methods; library/builtin stubs are
  `true`. Filter with it to keep transitive traversals first-party and bounded.
- **`FILENAME` is only reliable on `METHOD` and `TYPE_DECL`.** `CALL`,
  `IDENTIFIER`, `LOCAL`, `LITERAL`, etc. carry **empty/absent `FILENAME`** (and
  usually empty `CODE`-level file context). To get the file of any such node,
  hop to its enclosing method: `(owner:METHOD)-[:CONTAINS]->(n)` and read
  `owner.FILENAME`. `LINE_NUMBER` **is** present on most nodes (including `CALL`).
- **`METHOD` structural children attach via `AST`, not `CONTAINS`.** Params
  (`METHOD_PARAMETER_IN`), the `BLOCK`, `METHOD_RETURN`, and `MODIFIER` are `AST`
  children. `CONTAINS` from a `METHOD` reaches its nested **`CALL` sites** (used
  for the call graph). Reach a parameter with
  `(m:METHOD)-[:AST]->(:METHOD_PARAMETER_IN)`; its position is the node property
  `INDEX` (call-site arguments carry the matching `ARGUMENT_INDEX`).
- **`REACHING_DEF` (data flow) is intraprocedural and carries no edge
  properties.** It links def→use **within one method** and stops at call-site
  arguments — it does **not** cross into a callee. Interprocedural flow must be
  reconstructed by bridging over the `CALL` edge (call-site → callee → matching
  `METHOD_PARAMETER_IN` by `INDEX`), which is only as complete as the sparse call
  resolution above. For high-fidelity interprocedural taint, run Joern's
  `reachableBy` in the REPL (the `joern` agent) rather than pure Cypher.
