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

## 2026-08-15 — When judge and agent-under-test collapse onto the same model, the self-preference-bias caveat must be split per sub-pass, not applied blanket

- **Evidence:** `falkor-chat/docs/plans/graphrag-eval.md` v4 D1 collapses Unit 3's judge onto the
  same `qwen/qwen3-4b-2507` instance as the agent-under-test (stakeholder-directed hardware
  constraint) and writes one blanket self-preference-bias caveat covering "any faithfulness/
  relevance numbers Unit 3 reports." On inspection this conflates two sub-passes with different
  exposure: the *calibration* sub-pass judges fixed, independently-authored triples (not
  self-generated — self-preference bias barely applies, it's still a legitimate rubric-following
  signal), while the *generation* sub-pass judges the model's own live output (classic
  self-preference exposure — the judge may favor its own phrasing/reasoning). A reader who sees a
  passing calibration number and extends that trust to the generation numbers is making exactly the
  error the caveat exists to prevent, and a single undifferentiated caveat doesn't stop that read.
  This codebase already had one prior instance of the identical structural issue, independently
  named: `falkor-chat/docs/archive/plans/m3-guard-calibration.md` risk #4, "self-preference
  (inherited, DS risk #3)... unmeasurable with this set alone" for the intake/research guard judge.
- **Context:** methodology sign-off on K-026's Unit 3 judge-layer plan
  (`falkor-chat/docs/reviews/graphrag-eval-ml.md`), where the stakeholder deliberately traded away
  the "never the model-under-test judging itself" guidance for a real hardware constraint.
- **Suggested home:** prompt (data-scientist's LLM-as-judge validity section) — the general rule
  "when judge collapses onto agent-under-test, the caveat must distinguish content the judge
  generated itself from content it didn't; only the former carries self-preference risk" is reusable
  well beyond this one K-026 baseline and beyond falkor-chat.
