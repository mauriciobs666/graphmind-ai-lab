# Kaizen — Learnings Inbox: graph-dba

> Append-only capture of durable, non-obvious environment facts the `graph-dba` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-08-16 — `META_DATA` (and `FILE`/`TYPE`/`NAMESPACE`) are absent from both live pysrc2cpg-built graphs, despite being listed in cpg-model.md's node-label vocabulary
- **Evidence:** `MATCH (n:META_DATA) RETURN n` → 0 rows on both `cpg_falkorchat` and `cpg_salesperson`, via `mcp__cpg__query`. `CALL db.labels()` on `cpg_falkorchat` returned exactly 20 labels (`CpgNode`, `METHOD`, `CALL`, `LOCAL`, `MODIFIER`, `LITERAL`, `IDENTIFIER`, `FIELD_IDENTIFIER`, `BLOCK`, `METHOD_RETURN`, `METHOD_PARAMETER_OUT`, `CONTROL_STRUCTURE`, `METHOD_PARAMETER_IN`, `UNKNOWN`, `METHOD_REF`, `TYPE_DECL`, `RETURN`, `IMPORT`, `TYPE_REF`, `MEMBER`) — none of `META_DATA`, `FILE`, `TYPE`, `NAMESPACE`/`NAMESPACE_BLOCK` appear, though `skills/joern-cpg/references/cpg-model.md`'s "Node labels you'll see most" list includes all of them.
- **Context:** designing the cpg-agent-adoption freshness-marker feature (`docs/plans/cpg-agent-adoption-graph.md`) — was evaluating whether to hook a build-timestamp property onto the existing `META_DATA` node vs. a dedicated node; had to rule out `META_DATA` empirically before designing around it.
- **Suggested home:** knowledge base (`skills/joern-cpg/references/cpg-model.md`) — the "Node labels you'll see most" list reads as "always present" but is apparently frontend/export-configuration-dependent (likely `pysrc2cpg` + default `--repr cpg` specific); worth a one-line caveat there ("confirmed absent on `pysrc2cpg`/`cpg_falkorchat`/`cpg_salesperson`, 2026-08-16") so a future producer doesn't assume `META_DATA` is a safe anchor without checking first, the way this task almost did.

## 2026-08-16 — `GRAPH.EXPLAIN` (unlike `GRAPH.QUERY`/`GRAPH.RO_QUERY`) refuses to run against a graph key that doesn't exist yet — it cannot be used to syntax-check a write query with no prior graph
- **Evidence:** `redis-cli GRAPH.EXPLAIN cpg_stamp_syntax_check "MERGE (b:CpgBuildInfo) SET ..."` against a never-created graph key → `ERR Invalid graph operation on empty key`, for a syntactically valid Cypher statement. Confirmed the key was never materialized by the attempt (`GRAPH.LIST` afterward doesn't show it) — so this isn't the same "querying a non-existent graph materializes it" gotcha `cpg-analysis/SKILL.md` §1 already documents for `GRAPH.QUERY`; `EXPLAIN` just hard-fails instead.
- **Context:** implementing the `cpg-agent-adoption` freshness-marker stamping step in `skills/joern-cpg/scripts/pipeline.sh` (U4a) — wanted to syntax-verify the generated `MERGE (b:CpgBuildInfo) SET ...` Cypher string without running a real pipeline build or mutating a live graph. `GRAPH.EXPLAIN` was the first thing tried and doesn't work for this without an existing graph; `GRAPH.RO_QUERY` against an *existing* graph worked instead — it parses the write query first and only then rejects it for being a write ("graph.RO_QUERY is to be executed only on read-only queries"), which is enough to prove the Cypher parses without persisting anything.
- **Suggested home:** knowledge base (`falkordb-quirks.md`) — a reusable technique worth recording: to syntax-check a *write* Cypher string with zero side effects, run it via `GRAPH.RO_QUERY` against any existing graph (not necessarily the target one) and confirm the rejection reason is "read-only queries only," not a parse error; `GRAPH.EXPLAIN` is not a substitute unless the target graph already exists.
