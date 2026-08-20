# Kaizen — Learnings Inbox: devops

> **FROZEN — 2026-08-20.** This file is a historical snapshot only (no entries had accumulated at
> migration time). `claude/cobb/kaizen/history.md`'s 2026-08-20 entry records the team-wide
> switch; `devops` no longer appends here. New raw learnings are written directly into the
> `kaizen_devops` FalkorDB graph and are immediately queryable by any agent:
> `mcp__cypher__query(graph='kaizen_devops', cypher='MATCH (e:KaizenEntry) RETURN e.date,
> e.fact, e.evidence, e.context, e.suggestedHome, e.author ORDER BY e.date')`. Content below is
> preserved for historical reference and will not change.

> Append-only capture of durable, non-obvious environment facts the `devops` agent
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

*(empty — no unprocessed learnings)*
