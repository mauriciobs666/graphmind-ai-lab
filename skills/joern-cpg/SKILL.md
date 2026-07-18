---
name: joern-cpg
description: Operate the Joern toolset to turn a source repository into a Code Property Graph (CPG) and export/load it into FalkorDB as Cypher. Use when building a CPG for a codebase, querying it with CPGQL (AST/CFG/DDG, call graphs, data-flow & taint), or exporting/ingesting a repo's code graph into FalkorDB. Carries scripts that pin JOERN_HOME/JAVA_HOME and run parse → export (neo4jcsv) → transform → FalkorDB load, plus the CPG→FalkorDB model and a CPGQL cheat-sheet. Primarily driven by the `joern` agent.
---

# Joern CPG → FalkorDB

Turn source code into a **Code Property Graph** with [Joern](https://docs.joern.io)
and materialize it in **FalkorDB** so the code graph is traversable with Cypher.
The scripts encode the environment and the export→transform→load contract — use
them rather than hand-running `joern-*` from memory.

## Prerequisites (the scripts check these)

- **Joern** under `$HOME/joern/joern-cli` (override with `JOERN_HOME`). Verified
  build here: **v4.0.579**.
- **JDK 21** (`java -version` → 21). `scripts/joern-env.sh` resolves `JAVA_HOME`
  from `java` on PATH, falling back to the system JVM — no shell config assumed.
- **FalkorDB** reachable at `localhost:6379` (override `FALKORDB_HOST`/`FALKORDB_PORT`),
  loaded via `redis-cli` (start it with `falkor-chat/scripts/start_falkordb.sh`).
  The Python `falkordb`/`redis` packages are **not** required — the loader uses
  `redis-cli GRAPH.QUERY`.

## The pipeline

One command, end to end (build → export → transform → optional load):

```bash
scripts/pipeline.sh <source> --graph cpg_myrepo --workdir ./joern-work --load
# omit --load to stop at the .cypher artifact (./joern-work/load.cypher)
```

Or run the stages individually:

```bash
# 1. Parse source -> CPG binary (overlays applied: call graph, control/data flow)
scripts/build-cpg.sh <source-dir> cpg.bin
#    JOERN_LANGUAGE=python forces a frontend; joern-parse auto-detects otherwise.

# 2. (optional) Query in the REPL before/instead of exporting — see CPGQL below
joern cpg.bin

# 3. Export the CPG to neo4jcsv (the format the transformer consumes)
scripts/export-cpg.sh cpg.bin cpg-export cpg neo4jcsv
#    --repr choices: all|ast|cdg|cfg|cpg|cpg14|ddg|pdg ; export only what you need.
#    NOTE: joern-export requires the outdir to NOT pre-exist; the script clears it.

# 4. Transform the export into FalkorDB Cypher, and load it
python3 scripts/cpg-to-falkordb.py cpg-export -o load.cypher --graph cpg_myrepo --load
#    without --load: writes load.cypher only (the "export to Cypher" artifact)
```

### What the export looks like

`joern-export --format neo4jcsv` writes, **nested per method** under the outdir:

- `nodes_<LABEL>_header.csv` / `_data.csv` — header `:ID,:LABEL,<PROP>[:type],…`
- `edges_<TYPE>_header.csv` / `_data.csv` — header `:START_ID,:END_ID,:TYPE`
- `*_cypher.csv` — Neo4j `LOAD CSV` scripts; **ignored** (not FalkorDB-usable).

The transformer walks the tree recursively, so the per-method nesting is handled.

## The FalkorDB model (default — `graph-dba` owns real tuning)

The transformer maps the CPG onto FalkorDB like this:

- Every node gets a **shared `:CpgNode` label** *plus* its Joern type label
  (`CREATE (n:CpgNode:CALL) …`), so edges can be matched by `id` without knowing
  the node's type label.
- The Joern `:ID` becomes an integer property **`id`**; other columns become
  properties (`:int` → integer, `:string[]` → array split on `;`, else string;
  empty cells dropped).
- **`CpgNode(id)` is indexed first** (`CREATE INDEX FOR (n:CpgNode) ON (n.id)`),
  so the edge `MATCH (a:CpgNode {id:…})` is cheap. *(Confirm this DDL against the
  pinned FalkorDB build / `graph-dba` — a wrong index only degrades load speed,
  not correctness, since the loader tolerates a failing index statement.)*
- Nodes/edges are created with **UNWIND-batched `CREATE`** (default 500/statement,
  `--batch`), deduped by `id` (nodes) and `(start,end,type)` (edges) so the
  per-method export overlap doesn't double-create.

Edge relationship types are the Joern edge types verbatim (`AST`, `CFG`, `CALL`,
`ARGUMENT`, `REACHING_DEF`, `DOMINATE`, `CONTAINS`, `RECEIVER`, …). `AST`/`CFG`/
`REACHING_DEF` dominate the edge count — a whole-repo load is large and lives in
FalkorDB's RAM; size it with `graph-dba` and export only the `--repr` you need.

### Reloading is deliberate (destructive)

The loader **refuses a non-empty graph**. For a clean reload, reset it first:

```bash
redis-cli GRAPH.DELETE cpg_myrepo   # destructive, shared-state — escalates via joern's guard
```

This keeps the reset an explicit, guard-visible command rather than a hidden
side effect. Use `--append` to add into an existing graph instead.

## CPGQL cheat-sheet (in the `joern` REPL)

CPGQL is a Scala traversal DSL over the CPG. Common starting points:

```scala
cpg.method.name("main").l                    // methods named main
cpg.method.name("run").parameter.l           // its parameters
cpg.call.name("system|exec.*").l             // calls to risky sinks (regex)
cpg.call.name("os.system").argument.code.l   // argument source text
cpg.literal.code(".*password.*").l           // suspicious literals

// data-flow / taint: does user input reach a sink?
val src = cpg.call.name("input").argument
val sink = cpg.call.name("os.system").argument
sink.reachableBy(src).l                       // non-empty => tainted path exists

cpg.method.size ; cpg.call.size               // sanity counts after a build
```

Export a query result to a file with `... .l |> "out.txt"` or run non-interactively
with `joern --script <file.sc> --params cpgFile=cpg.bin`.

## Gotchas

- **JVM startup is slow** (~30–60s per `joern-*` invocation). A full
  parse+export of a real repo takes minutes; expect it and run long jobs in the
  background.
- **`--out` must not pre-exist** for `joern-export` — `export-cpg.sh` clears it
  (with a guard against unsafe targets like `/` or `$HOME`).
- **Overlays matter:** taint/data-flow queries need the default overlays (call
  graph, control/data flow) that `joern-parse` applies — don't pass `--nooverlays`
  if you'll query flow.
- **Scale:** deduped in memory by the transformer — fine for moderate repos; for
  very large codebases this is a streaming-loader concern (tracked in the `joern`
  agent's kaizen). Prefer a narrower `--repr` (e.g. `ast` or `cpg14`) when you
  don't need every edge layer.
- **Deeper schema & model notes:** see [`references/cpg-model.md`](./references/cpg-model.md).
