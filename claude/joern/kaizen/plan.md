# Kaizen — Improvement Plan: joern

> Forward-looking backlog for the `joern` agent (CPG generation + Joern toolset + CPG→FalkorDB).
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-17

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-002 | 2026-07-16 | med | 🔵 | Confirm/tune the CPG→FalkorDB model with graph-dba (index DDL, constraint, dense layers) |
| K-003 | 2026-07-16 | med | 🔵 | Streaming loader for large repos (transformer currently holds all nodes/edges in memory) |
| K-004 | 2026-07-16 | low | 🔵 | Reusable CPGQL script library (common security/taint/call-graph queries) |

> K-001 (live FalkorDB load test) — ✅ done 2026-07-17, see history.md. The `--load`
> path round-tripped a real CPG (107 nodes/462 edges) and the run surfaced + fixed two
> transformer bugs (`graph_nonempty` false-positive; boolean-as-string).

### K-002 — Bless the FalkorDB model with graph-dba
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** the shipped model (shared `:CpgNode` label + `id` index, type labels, edge types verbatim) is a sensible default, not a tuned one. Real query workloads (call-graph walks, taint traces) may want label/edge-specific indexes, a uniqueness constraint on `CpgNode(id)`, or splitting dense layers (`AST`/`CFG`) out.
- **Proposed change:** hand the model to graph-dba once a concrete query workload exists; capture the design at `<component>/docs/plans/<slug>-graph.md`.

### K-003 — Streaming loader for large codebases
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** `cpg-to-falkordb.py` collects all nodes/edges into memory to dedup before emitting. Fine for moderate repos; a repo-scale CPG (millions of AST/CFG edges) could exhaust RAM.
- **Proposed change:** stream per-file with an on-disk/rocksdb-style seen-id set, or dedup via `MERGE` at load time; benchmark against a large real repo first.

### K-004 — CPGQL script library
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** common queries (dangerous-sink reachability, unsanitized-input flow, call-chain to a function) get rewritten each run.
- **Proposed change:** add `skills/joern-cpg/scripts/queries/*.sc` runnable via `joern --script`, referenced from the CPGQL cheat-sheet.

## Parking lot / ideas
- Int-typed array columns (`prop:int[]`) are stored as arrays of *strings* by the transformer
  (element type isn't inspected). Harmless for the string arrays Joern actually emits
  (type hints, overlays, modifiers); revisit only if an int-array property surfaces in a query.
- Alternative export path via `graphml` (single file) for tools that prefer it over neo4jcsv.
- Incremental re-CPG: only re-parse changed files and patch the FalkorDB graph.
- `--repr` presets in the skill (e.g. `--profile callgraph` → `cfg`+`call` only) to shrink loads.
- Model choice (opus) is a revisit point — much of the pipeline is mechanical; sonnet may suffice once the model/queries stabilize.
