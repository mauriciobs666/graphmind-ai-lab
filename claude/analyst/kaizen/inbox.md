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

## 2026-08-11 — Reviewing a kaizen-inbox distillation: count removed `## ` entries in each inbox diff and reconcile against the history entry's own claimed count — that arithmetic is where silent drops surface

- **Evidence:** reviewing `cobb`'s 39-file 2026-08-11 distillation, `git diff claude/coder/kaizen/inbox.md | grep -c '^-## '` → 8 while `coder/kaizen/history.md`'s new entry claimed "8 entries routed (6 to …, 1 discarded)" — 6+1≠8, and mapping each removed entry to a stated disposition found two with none at all (both `Suggested home: prompt`, one of them a still-real gap: the skip-count rule exists at `tdd-engineer.md:42` but not at `coder.md:22`). The same reconciliation caught four more unlogged dispositions and four wrong header counts across `graph-dba` (7 removed vs. "5"), `devops` (13 vs. "9"), `tico` (4 vs. "3"), `qa-engineer` (sub-counts don't close). Second trap it exposes: a history entry can list promotions that came from a *different* agent's inbox (coder's listed analyst's/architect's urllib + LM Studio entries), which makes an incomplete list read as complete — so map entry→disposition from the **diff**, never from the history's prose. Third: an inbox entry can be *headless* (a stray `- **Evidence:**` with no `## ` heading, as in `teco`'s 458k one), so a pure heading count under-counts by one there.
- **Context:** diff-scoped review of a full-team kaizen distillation (`docs/reviews/kaizen-distillation-2026-08.md`); this check produced the review's only blocker, and nothing else in the diff would have revealed it.
- **Suggested home:** knowledge base (`claude/analyst/review-techniques.md`) — a short "auditing a kaizen distillation" technique: per inbox, `grep -c '^-## '` the diff, enumerate each removed entry's disposition from the diff text, reconcile against the history header, and check each claimed promotion's *source* inbox.

## 2026-08-11 — Agent prompts deploy by symlink into the working tree, so an *uncommitted* prompt edit under review is already live for the running team — including the reviewing agent itself

- **Evidence:** `ls -la ~/.claude/agents` shows one symlink per agent into `/home/<user>/prg/graphmind-ai-lab/claude/<name>` (and `~/.claude/skills → <repo>/skills`). Reviewing an uncommitted diff that edited `claude/analyst/analyst.md`, the new clause was already present verbatim in this review run's own system prompt. The symlink layout itself is documented (`claude/README.md:61-67`, `claude/AGENTS.md:34`); the review-relevant corollary is not stated anywhere I found.
- **Context:** diff-scoped review of a working-tree-only change to six agent prompts and three new knowledge bases.
- **Suggested home:** prompt or knowledge base (`claude/analyst/review-techniques.md`) — when the artifact under review is an agent prompt/skill under `claude/` or `skills/`, "uncommitted" does not mean "not yet in effect": findings ship immediately rather than at commit, which raises the urgency of a blocker and makes "restore it now, it's still recoverable via `git diff`" a real, time-boxed remedy.

