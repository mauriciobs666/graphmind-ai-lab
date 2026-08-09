# Kaizen — Learnings Inbox: analyst

> Append-only capture of durable, non-obvious environment facts the `analyst` agent
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

## 2026-08-09 — A "held, do not clear until X lands" kaizen-inbox note can go stale silently when X lands via a *different* agent's kaizen files than the one holding the note

- **Evidence:** Reviewing the cross-agent kaizen-inbox-distillation batch (`docs/reviews/kaizen-inbox-distillation.md`), `claude/architect/kaizen/inbox.md` carried two entries explicitly gated "do not clear until [a consolidated `skills/agent-standards/kiro.md` follow-up] lands." The follow-up *had* landed (confirmed via `git diff -- skills/agent-standards/kiro.md`, part of the same uncommitted batch) — but only `claude/analyst/kaizen/history.md`'s own entry ("Held entry 28 promoted: consolidated Kiro-facts edit landed") documented the closure. Neither `claude/architect/kaizen/inbox.md`/`history.md` nor the owning agent's (`cobb`'s) `kaizen/history.md` were updated to match, even though the same physical edit closed out both sides. Reading only the architect-side files would have made this look like a still-pending, not-yet-due item (the brief's own framing anticipated exactly this ambiguity).
- **Context:** cross-checking a "held pending a shared follow-up" note during a multi-agent kaizen-bookkeeping review — the note lived in one agent's inbox but the actual close-out evidence lived in a *sibling* agent's history file for the same batched edit.
- **Suggested home:** prompt (Guardrails — "Evidence over vibes" bullet, or a dedicated multi-agent-bookkeeping clause): when a document under review claims "held/pending until a shared/coordinated follow-up lands," don't take the holding document's own state as authoritative — grep sibling agents'/owners' kaizen history for the same target file/date to check whether the follow-up already landed elsewhere and just wasn't mirrored back.
