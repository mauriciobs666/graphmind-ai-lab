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

## 2026-07-19 — `pysrc2cpg` call-graph is directionally asymmetric in FalkorDB: trust callees over the `CALL` edge, match callers by `CALL.NAME`
- **Evidence:** On `cpg_falkorchat` (79.6k/522k): direct `(:METHOD)-[:CALL]->(:METHOD)` count 0; `(:CALL)-[:CALL]->(:METHOD)` present for only ~1,334 of ~20,488 call sites (same-object `self.x()` + synthetic `…<metaClassAdapter>`). Reverse transitive reach into `Services.post_message` over the resolved edge returned 3 synthetic nodes; the name-based idiom `(caller:METHOD)-[:CONTAINS]->(:CALL {NAME:'post_message'})` returned 21 real callers (incl. `api.py`, `mcp.py`). No production entrypoint reaches a `query` sink over the resolved call graph (cross-object dispatch + `.query()` on an external handle are unresolved).
- **Context:** Authoring/verifying the `cpg-analysis` skill recipes (impact, rca, code-review, test-gap) against a loaded CPG.
- **Suggested home:** knowledge base — full topology facts now live in `skills/joern-cpg/references/cpg-model.md` "Consumer-query facts"; worth a back-pointer from `falkordb-reference.md` (GraphRAG/CPG section) since falkor-chat + joern both consume this shape.

## 2026-07-19 — In this CPG, `FILENAME` is reliable only on `METHOD`/`TYPE_DECL`; params attach via `AST` not `CONTAINS`; `REACHING_DEF` is intraprocedural with no edge props
- **Evidence:** `keys()` on a `CALL` = [ARGUMENT_INDEX, CODE, COLUMN_NUMBER, DISPATCH_TYPE, LINE_NUMBER, METHOD_FULL_NAME, NAME, …] (no FILENAME); `IDENTIFIER`/`LOCAL` FILENAME empty. `(METHOD)-[:AST]->(METHOD_PARAMETER_IN)` holds; `(METHOD)-[:CONTAINS]->(METHOD_PARAMETER_IN)` returns nothing (CONTAINS reaches CALL sites). `keys(r)` on a `REACHING_DEF` edge = []. Forward slice of a param ends at the outbound call site, never entering the callee. Resolve a node's file via `(owner:METHOD)-[:CONTAINS]->(n)`.
- **Context:** same skill build; these drive the recipes' file-resolution and interprocedural-boundary handling.
- **Suggested home:** knowledge base (already captured in `cpg-model.md`); logged here for cross-project visibility.
