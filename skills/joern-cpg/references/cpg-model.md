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
