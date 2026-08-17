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

## 2026-08-17 — `AskUserQuestion` hard-rejects any question with fewer than 2 options — there is no way to route a stakeholder's free-text "let me describe it" answer back through the tool as a single-choice follow-up
- **Evidence:** Calling `AskUserQuestion` with a question offering one option (`"Let me describe it in free text"`) raised `InputValidationError: array to have >=2 items` before the user ever saw it, with explicit guidance not to invent a filler second option and to fall back to plain conversation instead.
- **Context:** `generic-cypher-mcp` requirements interview — the stakeholder picked "Something else — I'll describe it" on a multiSelect `AskUserQuestion`, and the natural next move (a single-option follow-up question) was rejected by the tool; had to drop back to a plain conversational prompt to get the free-text answer.
- **Suggested home:** prompt (a one-line addition to Tico's own interview-craft guidance: when a stakeholder's answer needs open-ended free-text follow-up, ask in plain conversation, not via `AskUserQuestion`).

## 2026-08-17 — FalkorDB's bundled web console is already live on `:3000` wherever `falkordb-dev` runs — no new work needed for "can I just look at the graph"
- **Evidence:** `falkor-chat/scripts/start_falkordb.sh` publishes `-p "${FALKORDB_WEB_PORT}:3000"` (default `3000`) alongside the `:6379` Redis/Cypher port, and prints `Web console: http://localhost:${FALKORDB_WEB_PORT}` on every start.
- **Context:** `generic-cypher-mcp` requirements interview — stakeholder asked "can I use the web tool to monitor created nodes/relationships," which resolved a whole access-tier open question (deferring their own read/write access to a later phase) once confirmed this already exists today with zero new work.
- **Suggested home:** unsure (candidate: a line in `tico`'s own explanation reflexes, or `cpg/mcp/README.md`'s troubleshooting/observability section, since agents debugging graph state might not know this console exists either).
