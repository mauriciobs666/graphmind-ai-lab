# Kaizen — Learnings Inbox: architect

> Append-only capture of durable, non-obvious environment facts the `architect` agent
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

## 2026-08-19 — `CLAUDE_CODE_SESSION_ID` is a real, live environment variable holding the actual Claude Code session UUID, readable via a plain `Bash env` call in a subagent
- **Evidence:** during the `generic-cypher-mcp2` (M7) plan, `env | grep CLAUDE_CODE_SESSION_ID` in this session returned `CLAUDE_CODE_SESSION_ID=7315f2d5...`, which matched — character for character — the UUID segment of this same session's own scratchpad path (`/tmp/claude-1000/.../7315f2d5-1ceb-48b3-836e-dbf764c16fe0/scratchpad`), confirming it is the real session identifier, not an unrelated value. `CLAUDE_CODE_CHILD_SESSION=1` was also present alongside it (plausibly a boolean flag marking this process as a spawned subagent). Not found in Claude Code's published docs — discovered only by direct environment inspection.
- **Context:** resolving `docs/requirements/generic-cypher-mcp2.md` FR-8a's open question ("how does an agent obtain its own Claude Code session ID at write time") — this became the concrete mechanism recorded in `docs/plans/generic-cypher-mcp2.md` §3.3.
- **Suggested home:** knowledge base — a candidate fact for wherever this team keeps live-verified-but-undocumented Claude Code harness specifics (e.g. `cobb`'s `agent-standards` skill), since it's exactly the kind of perishable, version-sensitive detail that convention exists to hold, and it's now load-bearing for eleven agents' Learning-capture recipes if the FR-8a design in `generic-cypher-mcp2.md` ships as planned.

*(otherwise empty — no other unprocessed learnings)*

