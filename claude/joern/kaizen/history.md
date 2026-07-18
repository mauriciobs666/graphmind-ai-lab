# Kaizen — Change History: joern

> Dated log of actual changes to the `joern` agent. Most recent first.

## 2026-07-17 — K-001 closed: live FalkorDB load verified; two transformer bugs fixed
- **What:** exercised the full `--load` path against a live FalkorDB `v4.18.11`
  (`falkor-chat/scripts/start_falkordb.sh -d`). A real Python-sample CPG round-tripped:
  build → export (147 nodes/657 edges) → transform (deduped 107 nodes/462 edges → 30 stmts) →
  **loaded 30/30 statements, 0 failed**. Verified in-graph: 107 nodes / 462 edges, the
  `CREATE INDEX FOR (n:CpgNode) ON (n.id)` DDL **is accepted** by this build (index present via
  `db.indexes()`), every node carries `:CpgNode` + its Joern type label, all CPG edge layers present
  (AST/CFG/CDG/REACHING_DEF/DOMINATE/CALL/…), and the call graph traverses correctly
  (`main` → `read_input,print,greet`; `read_input` → `input`).
- **Two bugs the live test surfaced, both fixed in `skills/joern-cpg/scripts/cpg-to-falkordb.py`:**
  1. **`graph_nonempty` false-positive** — it regex-scanned the whole `redis-cli` reply for any
     digit `>0`, so the `Query internal execution time: 0.179153 milliseconds` stat line always
     read as "non-empty" and the loader refused *every* graph, even truly empty ones. Fixed: probe
     with `GRAPH.RO_QUERY` (a missing graph returns `ERR ... empty key` and, unlike `GRAPH.QUERY`,
     does **not** materialize an empty key), and parse the count from the one pure-integer output line.
  2. **`:boolean` columns stored as strings** — `parse_header` had no boolean branch, so
     `IS_EXTERNAL:boolean` fell through to string and stored `"false"`; predicates like
     `WHERE m.IS_EXTERNAL = false` matched nothing. Fixed: `boolean`→`bool` kind emitting real
     Cypher `true`/`false` (also mapped `long`→int). Re-load confirmed the boolean predicate works.
- **Why:** K-001 — confirm ingestion against the real engine, not just the transform step.
- **Learnings routed (this run, cobb acting):** Joern neo4jcsv property names are **UPPER_CASE**
  (`NAME`, `CODE`, `FULL_NAME`, `IS_EXTERNAL`, `ORDER`) — CPGQL's lowercase `.name`/`.code` do
  **not** work on the loaded FalkorDB graph → documented in `skills/joern-cpg/references/cpg-model.md`.
  The two FalkorDB engine quirks (`GRAPH.QUERY` read materializes an empty key; `GRAPH.RO_QUERY`
  errors + creates nothing on a missing graph) → added to `claude/graph-dba/falkordb-quirks.md`.
- **Plan items:** closed **K-001**.

## 2026-07-16 — Created
- **What:** New `joern` subagent (CPG specialist) + companion `joern-cpg` skill created together.
  - Agent `claude/joern/joern.md` (model `opus`): operates the Joern toolset to build CPGs, query
    them via CPGQL, and export/load a repository's code graph into FalkorDB end-to-end. Frontmatter
    wires the shared destructive-ops `PreToolUse` guard (`joern/hooks/guard-destructive-ops.sh` →
    `claude/scripts/guard-destructive-ops.sh joern`), since loading resets a shared FalkorDB graph.
  - Skill `skills/joern-cpg/` (SKILL.md + `references/cpg-model.md` + scripts): `joern-env.sh`
    (pins `JOERN_HOME` default `$HOME/joern/joern-cli`, auto-detects `JAVA_HOME`), `build-cpg.sh`,
    `export-cpg.sh` (neo4jcsv), `cpg-to-falkordb.py` (neo4jcsv → FalkorDB-dialect UNWIND-batched
    Cypher, deduped; optional `redis-cli` load, refuses a non-empty graph), `pipeline.sh`.
  - FalkorDB model: shared `:CpgNode` label + Joern type label, integer `id` property,
    `CpgNode(id)` indexed first, edge types verbatim. Flagged for graph-dba tuning (K-002).
- **Why:** user asked for an agent that knows CPGs deeply and can drive the local Joern toolset to
  represent a repo and export it to Cypher/FalkorDB.
- **Verification:** end-to-end pipeline run on a Python sample succeeded (exit 0):
  build (pysrc2cpg, 104K cpg.bin) → export (122 nodes/538 edges neo4jcsv) → transform (deduped to
  91 nodes/373 edges → 29 Cypher statements). Transformer independently validated against a real
  export. Live FalkorDB `--load` NOT exercised (server down) → K-001.
- **Environment facts folded into the prompt/skill at creation (not inbox):** Joern v4.0.579 under
  `$HOME/joern/joern-cli`; requires JDK 21 (installed this session via `apt install openjdk-21-jdk`);
  `joern-export` neo4jcsv output is nested per method with `:ID/:LABEL` node headers and
  `:START_ID/:END_ID/:TYPE` edge headers; the `*_cypher.csv` files are Neo4j `LOAD CSV` (not FalkorDB).
- **Bookkeeping:** seeded kaizen plan/history/inbox; added catalog entries (claude/README.md,
  skills/README.md), roster/index entries (claude/AGENTS.md, root AGENTS.md, teco roster);
  added reciprocal graph-dba↔joern boundary clause + `joern:graph-dba` to audit-team.sh
  BOUNDARY_PAIRS; deployed `~/.claude/agents/joern` symlink; added joern+java to shell PATH.
- **Plan items:** opened K-001…K-004.
