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

## 2026-08-16 — A zero-tolerance "current >= baseline" regression gate on a golden-set metric at n≈38 is stricter than the sample supports, and the arithmetic to show it is cheap

- **Evidence:** `falkor-chat/server/tests/eval/test_retrieval_eval.py`
  (`test_retrieval_metrics_meet_or_beat_baseline`) hard-fails on any `recall_at_10` drop below the
  committed baseline (0.9737, n=38), no slack, while allowing MRR a 5%-relative floor. At n=38 a
  single golden pair flipping hit→miss moves recall@10 by exactly 1/38 ≈ 2.6 points — about a fifth
  of the metric's own Wilson 95% CI width (≈13 points) at that n, and the baseline's sum-of-scores
  (0.97368... × 38 = 37.0 exactly) shows it is literally one pair-flip from both the ceiling (1.0)
  and the gate's failure floor. A metric already near ceiling with zero slack below it will fail on
  ANN tie-breaking noise or index-rebuild nondeterminism as readily as on a genuine regression — and
  it also cannot register a genuinely better change (no room to move up). Cheap diagnostic: compute
  `1/n` as the "one-pair delta" and compare it against a naive Wilson/Wald CI at the reported n
  *before* signing off on any hard pass/fail gate built on a golden set this size — if the one-pair
  delta is a large fraction of the CI width, the gate needs either a tolerance band (mirroring
  whatever slack the paired metric already gets) or an explicit "route first failure to manual
  triage" policy instead of an unconditional hard fail. Also worth checking whether the golden
  pairs are drawn independently or clustered (e.g., k pairs per source thread/document) — clustering
  means the true variance is understated by treating n as the pair count, not the corrected count.
- **Context:** M-4 baseline sign-off on K-026's `retrieval_baseline.json`
  (`falkor-chat/docs/reviews/graphrag-eval-ml.md`, "Baseline sign-off" section).
- **Suggested home:** prompt (data-scientist's evaluation-engineering / golden-set section) — the
  "compute the one-unit delta and compare it to the CI width before blessing a hard gate" check is a
  reusable pre-sign-off habit for any small-golden-set regression gate, not specific to retrieval or
  to this repo.
