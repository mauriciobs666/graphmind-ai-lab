# Kaizen — Learnings Inbox: data-scientist

> Append-only capture of durable, non-obvious environment facts the `data-scientist` agent
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

## 2026-07-16 — `repository.read_thread` returns `authorType` as a LIST (`["User"]`), not a string

- **Evidence:** `falkor-chat/server/falkorchat/repository.py:585-596` — the Cypher projects
  `labels(author) AS authorType` and the row dict passes it straight through (`"authorType": row[6]`).
  FalkorDB's `labels()` returns a list. Plan prose in `docs/plans/m3-guard-thread-context.md:241`
  describes the row shape as `{msgId, text, role, createdAt, authorId, displayName, authorType}`
  without noting the list — easy to build a fixture/stub with `"authorType": "User"` and have it
  silently diverge from live rows.
- **Context:** authoring `server/tests/eval/golden_guards.jsonl`; golden turn rows must mirror
  `read_thread` shape exactly to be a faithful stand-in for a live thread.
- **Suggested home:** project docs (`falkor-chat/AGENTS.md` schema conventions, or a `QUERIES.md` §4
  read-path note)

## 2026-07-16 — falkor-chat's LLM guard judges are bias-to-suspend BY DESIGN, so symmetric agreement metrics (Cohen's κ, plain accuracy) mis-gate them

- **Evidence:** `falkor-chat/server/falkorchat/guards.py:109-132` (`_coerce_verdict`) resolves every
  ambiguity — non-mapping output, non-bool `decision`, a rationale tripping `_NEGATION_CUES` — to
  `decision=False`, deliberately pinning specificity near its ceiling. Simulation over an
  11-advance/10-suspend golden set (20k trials, run 2026-07-16): false-advance rate is a function of
  specificity *only* and reads ~0–1% for ANY sensitivity, so an always-suspend judge scores a perfect
  0% FAR; κ is dominated by sensitivity once specificity is high (sens 0.95/0.80/0.60/0.40 at
  spec 0.99 → E[κ] 0.94/0.78/0.58/0.38). The two gate arms in `docs/plans/m3-executor-ml.md`
  §"Evaluation design" thus measure different classes and decouple. κ also moves with hand-picked case
  mix (same judge: E[κ] 0.70 at 11/10, 0.55 at 18/3 — prevalence effect).
- **Context:** K-022 U15 guard-judge calibration; the DS note's own Q1 bias-to-suspend decision
  contradicted the κ gate written two sections later to protect it. Recorded because this lab keeps
  building deliberately asymmetric LLM judges (bias-to-suspend / abstention designs), so the reflex to
  reach for κ/accuracy will recur.
- **Suggested home:** prompt (standing rule: for a deliberately asymmetric judge, gate on
  class-conditional rates — false-advance + advance-recall — and demote κ/accuracy to reported
  diagnostics with marginals)
