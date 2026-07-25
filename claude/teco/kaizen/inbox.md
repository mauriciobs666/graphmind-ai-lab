# Kaizen — Learnings Inbox: teco

> Append-only capture of durable, non-obvious environment facts the `teco` agent
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

## 2026-07-25 — A `.mcp.json`-wired stdio MCP server cannot be verified by the session that delivers it; the *next* session gets the verification for free

- **Evidence:** the session that delivered S3 wrote `.mcp.json` + `enabledMcpjsonServers`, and
  `claude mcp list` reported `✔ Connected` — but no in-session `mcp__cpg__query` tool existed, because
  Claude Code materialises `.mcp.json` servers only at **session start**. Three done-conditions were
  logged as "user-only actions" and the dependent unit (S9 acceptance) was blocked. The following
  session simply *started* with the file present: the server's `instructions` block appeared in the
  system prompt unprompted, and S9 ran immediately with no user action at all.
- **Context:** coordinating the CPG-query-access feature; S9 (`qa-engineer`) reaches `mcp__cpg__query`
  only by inheriting it from a parent session that has the server connected.
- **Suggested home:** prompt — when a unit's done-condition needs a harness restart, don't park it as
  a user action; sequence it as *"first act of the next session"*. Subagents inherit MCP tools from
  the parent session, so the coordinator's session state gates every delegate's tool access.

## 2026-07-25 — Verifying "no *new* audit failures" needs a diff against the last commit, not a re-read of the gate's verdict

- **Evidence:** `claude/scripts/audit-team.sh` was already `RESULT: FAIL` on pre-existing leaks, so the
  unit's done-condition was "no new failures". Two agents reported that condition met. It was not:
  `git show f2d55f7:docs/plans/cpg-query-access-coordination.md | grep -c "/home/<user>"` → **0**, but
  the working tree had **1** — a genuinely new leak, in teco's own coordination doc, invisible to
  anyone comparing only the FAIL/PASS verdict (which is identical either way, since the check was
  already red). Compounding it: the *fix note* describing the leak reintroduced it by quoting the
  literal path, and had to be fixed twice.
- **Context:** integration-checking S10; the leak was authored by teco itself in a prior session.
- **Suggested home:** prompt — on a gate that is already red, "no new failures" must be verified by
  diffing hit *paths* against the base commit, never by re-reading the verdict. Corollary: prose about
  a PII audit can fail that PII audit; write the pattern generically the first time.
