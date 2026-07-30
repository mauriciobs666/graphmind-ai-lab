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

## 2026-07-29 — Resuming a background subagent after a platform error means a fresh `Agent` call with a self-contained state-recovery brief, not a true session continuation
- **Evidence:** two consecutive background `Agent` calls for the same frontend unit (K-036 Wave 3+4, graphmind-ai-lab) failed with `API Error: 500 Internal server error` mid-run. Task-notification text references resuming via "SendMessage," but no `SendMessage` tool was present in this session's tool list — only `Agent` (which starts a fresh context). Recovery worked by launching a new `Agent` call whose prompt described exactly what the failed agent's own `git status`/`git diff` would show (files already written, what was in progress) and instructed it to verify that state itself before continuing — not by literally resuming the old session.
- **Context:** falkor-chat K-036 web-api-coverage Wave 3+4 delivery, two transient 500s in a row before the third attempt succeeded.
- **Suggested home:** prompt (teco.md's Guardrails/Delegate-with-complete-briefs section) — note explicitly that "continue this agent via SendMessage" in task-notification text is not always backed by a tool actually available to teco; the fallback is a fresh `Agent` call whose brief tells the agent to inspect on-disk/git state and resume from there.

## 2026-07-29 — This dev box has no `node` on `PATH`; subagents needed a workaround to run bare-`node` JS tests
- **Evidence:** `falkor-chat/`'s web test convention (its own architect plan, `docs/plans/web-api-coverage.md` §5.2) requires bare-`node` unit tests for dependency-free JS logic. Two independent subagent sessions (frontend-engineer, then analyst re-reviewing) both had to discover a workaround — one used a Playwright-bundled Node binary, the other found `node.exe` reachable from WSL — before they could run `node web/tests/run-select.test.js`. Neither workaround was mentioned in `falkor-chat/AGENTS.md`.
- **Context:** K-036 Wave 3+4 delivery and its Gate 2 review, graphmind-ai-lab/falkor-chat.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md`'s "Key scripts"/environment section) — worth a one-line pointer to a working `node` invocation for this box, so future frontend/analyst briefs in this repo don't re-spend turns rediscovering it.
