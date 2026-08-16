# Kaizen — Learnings Inbox: qa-engineer

> Append-only capture of durable, non-obvious environment facts the `qa-engineer` agent
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

## 2026-08-16 — a live `pytest -m live` run doing ~50 sequential local-LLM round trips against LM Studio takes ~3 minutes wall-clock, exceeding Bash's default 120s foreground timeout
- **Evidence:** `falkor-chat`'s K-026 eval harness (`server/tests/eval/test_judge_live.py`) issued ~50 sequential `.complete()` calls (20 generation + 20 judge-of-generation + 10 calibration-judge) against a local `qwen/qwen3-4b-2507` via LM Studio; the foreground Bash call auto-moved to background after 120s, final run time `175.92s (0:02:55)` per `time`'s own output. Resolved cleanly with `Monitor` polling `kill -0 <pid>` on the backgrounded process, no manual sleep-looping needed.
- **Context:** K-026 GraphRAG eval harness QA acceptance pass (TP-005, `docs/test-plans/graphrag-eval.md`).
- **Suggested home:** knowledge base (`qa-engineer/qa-testing-techniques.md`) — a live-marked suite with many sequential local-LLM calls should be launched expecting >120s and either use `run_in_background` proactively or be ready to hand off to `Monitor`/an until-loop rather than assuming a single foreground Bash call will return in time.
