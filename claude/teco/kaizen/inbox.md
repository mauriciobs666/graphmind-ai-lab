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

## 2026-08-12 — A review-gated unit with no coordination doc is nearly unrecoverable after a mid-session credit crash; a self-sufficient deliverable is the only thing that saved it
- **Evidence:** Resumed a session where `cobb` had run a full-team kaizen distillation (39 files, uncommitted) and `analyst` had gated it "needs changes," but credits ran out before the fix pass was dispatched and no `docs/plans/*-coordination.md` ledger existed for the unit (the review's own header read "no backlog id; stakeholder-triggered `cobb` sweep," suggesting `cobb`/`analyst` were invoked directly, not routed through `teco`). Reconstructing state required reading the full 340-line review (`docs/reviews/kaizen-distillation-2026-08.md`) cold, cross-checking `git status`/`git log`, and manually re-deriving facts the review didn't settle (e.g. grepping `falkor-chat/docs/BACKLOG.md` by hand to discover K-041 already covered an "open question" the review had flagged as unresolved). `ListAgents` showed no path to the original `cobb`/`analyst` runs — only other live sessions, not subagents spawned inside a now-dead one — so recovery meant cold-respawning with a brief rebuilt entirely from the written review, not resuming anything.
- **Context:** Resuming a kaizen-inbox-distillation fix-and-regate after the prior session's credits ran out mid-coordination.
- **Suggested home:** prompt (`teco.md` step 2/3) — open the coordination doc *at first dispatch*, not once a 3-unit/complexity threshold is felt, for any sequence carrying a review gate; record each delegate's `agentId` in the ledger the instant it returns. The ledger is the only thing that survives a crash — a review/report deliverable being evidence-dense and self-sufficient is what made cold-respawn viable here, but that was the producing agent's habit, not a guarantee.
