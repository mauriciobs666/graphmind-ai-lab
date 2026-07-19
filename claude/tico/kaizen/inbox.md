# Kaizen — Learnings Inbox: tico

> Append-only capture of durable, non-obvious environment facts the `tico` agent
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

## 2026-07-19 — In this repo a new feature request often *reverses a decision already recorded as an FR* in an existing requirements doc; grep `docs/requirements/` + the status logs in `docs/plans/` before the first question.
- **Evidence:** `docs/requirements/joern-cpg-pipeline.md:77-78` records FR-9 "querying FalkorDB with Cypher (`redis-cli GRAPH.QUERY`) … *chosen over MCP tool / raw Cypher*" — the exact option the stakeholder then asked for. `docs/plans/m2-cpg-analysis-coordination.md:57-63` carries a dated "Status log (resume)" with live-verified numbers (callers=21, test-gap=30) reusable as acceptance criteria.
- **Context:** interview for `docs/requirements/cpg-query-access.md` (MCP access to a loaded CPG); the prior FR turned an innocuous request into an explicit supersession requirement plus a doc-consistency AC.
- **Suggested home:** prompt (tico's "do your homework silently" step — name prior-decision provenance as a thing to check for)

## 2026-07-19 — `claude --agent tico` sessions start on `main` with a dirty tree; tico cannot honour a "please commit" request itself.
- **Evidence:** session git status: branch `main`, ~16 modified files unrelated to the interview. tico's guardrail is "Bash is for investigation only … never mutate the tree", while the global git instruction says to branch first when on the default branch.
- **Context:** stakeholder closed the interview with "confirmed please commit"; resolved by handing back a ready-to-run branch+commit command.
- **Suggested home:** prompt (handoff section — state that closing a doc hands the commit back to the human, with the command)
