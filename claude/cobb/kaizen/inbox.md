# Kaizen — Learnings Inbox: cobb

> Append-only capture of durable, non-obvious environment facts the `cobb` agent
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

## 2026-08-10 — A subagent's runtime tool set can be narrower than its frontmatter `tools:`, silently
- **Evidence:** read-only probe of a live `teco` run (asked it to list every tool defined in its own context, no inference). Frontmatter declares `Read, Grep, Glob, Bash, Agent, SendMessage, Write, Edit`; the run reported exactly `Read, Bash, Agent, SendMessage, Write, Edit`. **`Grep` and `Glob` were absent** — no error, no warning, no deferred-tool notice. Repeated in a second probe with the same result. Consistent with this build steering search/read work to `Bash`.
- **Context:** graphmind-ai-lab, the 2026-08-10 teco coordination rework — the finding invalidated a planned `ListAgents` grant as self-evidently sufficient.
- **Suggested home:** knowledge base (`skills/agent-standards/claude-code.md`, subagents section) — a frontmatter `tools:` entry is a *request*, not a guarantee; verify by probing a live run before writing prompt logic that depends on a tool. Needs a `Verified:` stamp and a re-check on the next harness update.

## 2026-08-10 — Custom agent definitions load at parent-session start; editing an agent mid-session does not affect subagents spawned later in that same session
- **Evidence:** added `ListAgents` to `claude/teco/teco.md`'s `tools:`, then spawned `teco` from the *same* session to verify — it reported `ListAgents` absent. The session's own `/context` had listed "Custom agents: 12 agents" at start, i.e. the definitions were already resolved into context before the edit. Inconclusive-by-construction: the probe could only ever have seen the pre-edit definition. (Contrast with `Grep`/`Glob`, which predate the session and so are a sound negative.)
- **Context:** same rework; cost one probe spawn before the confound was spotted.
- **Suggested home:** knowledge base (`skills/agent-standards/claude-code.md`) — and a practical corollary for cobb's own workflow: **any verification of an agent-definition edit must run from a fresh session**, never from the session that made the edit.

## 2026-08-10 — The user-memory (AutoMem) *index* reaches a subagent; the entry *bodies* do not
- **Evidence:** probed a live `teco` run for four strings from `~/.claude/projects/<flattened-repo>/memory/MEMORY.md`. All four (`Memory Index`, `no-commit-footer`, `teco-process-lessons`, `quality-and-efficiency`) were present in its injected context inside a system-reminder labelled "user's auto-memory, persists across conversations" — but teco reported seeing **one line and a short gloss per entry, not the entry contents**. So a subagent knows a memory file exists and roughly what it covers, and cannot act on what is in it unless it reads the file.
- **Context:** deciding whether five standing coordination practices could stay in the user's memory file instead of being promoted into `teco.md`. They were promoted — a teaser the agent can't act on is worse than either alternative.
- **Suggested home:** knowledge base (`skills/agent-standards/claude-code.md`, "what loads where") — plus the general rule it implies: **behavior an agent must exhibit belongs in its committed prompt**, never in user-scoped memory, which is untracked, index-only to subagents, and invisible to the audit script.

