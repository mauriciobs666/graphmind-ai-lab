# GraphRAG retrieval + generation evaluation harness — same-model judge methodology review

> **Status:** archived · **Owner:** `data-scientist` · **Tracks:** K-026 (M2.5-quality)

## Baseline sign-off (retrieval_baseline.json, n=38) — 2026-08-16

**Scope, per the brief and D6/M-4 (`docs/plans/graphrag-eval.md` §3 D6, coordinator-accepted
`docs/reviews/graphrag-eval.md` M-4):** methodology/statistical sign-off on the *first* committed
`server/tests/eval/retrieval_baseline.json` — `recall@10=0.9737, recall@5=0.8947, MRR=0.6259, n=38`
— before it is treated as gating for future regressions (`test_retrieval_metrics_meet_or_beat_baseline`,
`test_retrieval_eval.py`). Read in full: my own method note (`docs/plans/graphrag-eval-ml.md` v2),
the implementation plan v4 §5 Unit 2b and D6, the coordination ledger's U3a/U3b/U3c rows and the
M-4/M-5 teco-decisions block, the Unit 1 corpus review and Unit 2a golden-set content review
sections of `docs/reviews/graphrag-eval.md`, and the actual code — `metrics.py`,
`test_retrieval_eval.py`, `golden_retrieval.jsonl` (38 pairs, confirmed gr-31 reworded per the Unit
2a review's suggested fix — no longer near-verbatim clause reuse), and `test_golden_set_integrity.py`
(confirms the self-retrieval-inflation guard from the method note's risk #2 is mechanically
enforced, not just aspirational).

### Findings

**F1 (informational — metric choice and computation are correct).** `recall_at_k`/`mrr` in
`metrics.py` implement the standard formulas exactly as specified in the method note
(`|top-k ∩ relevant| / |relevant|`; 1-indexed reciprocal rank of first hit, 0.0 on no hit), handle
the multi-relevant case (`gr-15`, `gr-34`) and ANN's documented "may return fewer than k" case
correctly (slicing/iterating simply stops short, no special-casing needed, no silent
divide-by-zero). `test_retrieval_eval.py` aggregates by unweighted mean across all 38 pairs, one
`hybrid_search` round-trip per query with recall@5 sliced from the same ordered recall@10 result
(no double-counting risk, no second network round-trip to introduce nondeterminism between the
two). recall@10 is the right primary metric (bounds what the LLM actually sees, since `k=10` is fed
raw with no relevance filtering — method note finding 2); recall@5 anchors the still-open
seed-cutoff question; MRR is the right secondary rank-quality signal. No finding here — this is
the one part of the deliverable I have no reservations about.

**F2 (Major — the recall@10 gate's zero-tolerance comparison is stricter than n=38 supports; this
is the finding the sign-off exists to catch).** `test_retrieval_metrics_meet_or_beat_baseline`
enforces `current["recall_at_10"] >= baseline["recall_at_10"]` with **no slack at all**, while MRR
gets a 5%-relative floor (D6). At n=38, a single golden pair flipping from hit to miss moves
recall@10 by exactly 1/38 = **2.6 percentage points** (2.7% relative) — for a single-relevant pair;
a partial flip on one of the two multi-relevant pairs (`gr-15`, `gr-34`) moves it by half that. A
Wilson 95% CI around 0.9737 at n=38 is **(0.865, 0.995)** — 13 points wide. One pair's worth of
movement is roughly a fifth of that interval's width, i.e., well inside the noise the sample size
can't resolve. Two facts compound this:
- **Sum-of-scores arithmetic pins the shortfall at exactly 1.0 point out of 38** (0.9736842... × 38
  = 37.0 exactly) — this baseline is one pair-flip away from a perfect 1.0 on the high side and one
  pair-flip away from 0.9474 on the low side. There is essentially **no slack left above** the
  baseline (recall@10 is already 1 flip from ceiling) and **zero tolerance below** it. A retrieval
  change that is genuinely neutral-to-better but happens to nudge one borderline query the wrong
  way (a near-tied cosine distance, or HNSW's own approximate-search non-determinism after a
  corpus/index change) fails the gate exactly as loudly as a real regression would.
- **The 38 pairs are not independent draws for variance purposes.** They're clustered 2–4 per
  thread across 12 threads (Unit 2a content review); if a thread's messages share retrieval
  difficulty (topic density, distractor overlap — the Unit 1 corpus review's own flagged
  "incident-report template" distractor bleed touches several threads at once), the effective
  degrees of freedom for the CI above sit somewhere between 38 (pair-level, what I computed) and 12
  (thread-level) — meaning the true sampling uncertainty is understated by the number above, not
  overstated. I have not modeled the intra-thread correlation directly (would need per-pair
  hit/miss labels, not just the aggregate); flagging the direction of the bias, not a corrected
  number.

**Recommendation:** don't hard-fail CI on any recall@10 move below baseline as currently coded.
Either (a) add a small tolerance band mirroring MRR's (e.g., allow recall@10 to drop up to one
pair's worth, ~2.6 points absolute, before failing — matching the metric's own resolution limit at
this n), or (b) keep the zero-tolerance check but route a first-time failure to manual triage
("did a specific golden pair flip, and does it look like noise or a real regression" — cheap to
answer by diffing the per-pair retrieved-id lists, which the harness doesn't currently print) rather
than an unconditional hard CI fail. (a) is the lower-maintenance fix and is what I'd default to;
(b) is better if the team wants zero-tolerance preserved deliberately as a "never regress even by
one pair, investigate every time" policy — that's a legitimate product choice, but should be a
stated policy, not an artifact of not having done this arithmetic.

**F3 (Minor — MRR's 5%-relative tolerance is closer to right-sized, but only just).** A single
pair swinging from a rank-1 hit to a total miss moves aggregate MRR by 1/38 = 0.0263 absolute ≈
**4.2% relative** to the 0.6259 baseline — just under the 5% floor on its own, but two independent
noise-driven flips (8.4% relative) would already trip it. A milder swing (rank 1 → rank 2 on one
pair) is only ~2.1% relative and wouldn't trip it alone. This tolerance is defensible as-is (it's
in the right order of magnitude relative to the metric's own single-pair resolution, unlike
recall@10's), but it is not a large margin either — worth knowing that "MRR regressed >5%" in
practice means "roughly 2 pairs' worth of rank movement," not a large or dramatic shift.

**F4 (Minor — near-ceiling recall@10 leaves little headroom to detect future improvement).**
recall@10=0.9737 is already close to 1.0; recall@5=0.8947 and MRR=0.6259 have materially more room
to move. Future retrieval work (entity expansion, hybrid fusion, a seed-relevance cutoff) is
therefore more likely to show its signal in recall@5/MRR than in recall@10 — worth setting that
expectation now so a flat recall@10 on a genuinely-improved retrieval change isn't misread as "no
improvement." This doesn't change the gate's validity, just how the three numbers should be read
together.

**F5 (informational — golden-set validity chain, inherited not re-verified here).** The golden set
behind this baseline was **`analyst`-reviewed, not human-verified**, per the coordinator's M-5
decision ("proceeding with `analyst`-review as the human-verification stand-in... Auto Mode bias to
proceed on a non-blocking methodology opinion" — coordination ledger). My own method note states
human verification as "the validity anchor, not optional," and D6 itself already names this exact
gap ("the corpus behind them is `analyst`-reviewed, not human-verified... a mediocre first-run
number would otherwise silently block a genuinely better future retrieval change") as part of why
this sign-off step exists. I am accepting the substitution as the coordinator already made that
call and it's out of scope to relitigate here, but it means this baseline's numbers rest on one
fewer layer of independent verification than the method note originally specified — a second-order
compounding factor on top of F2's statistical fragility, not a new independent risk. The `analyst`
pass itself was thorough on the dimension it checked (all 38 `target_text` values byte-verified
against the corpus, gr-31's near-verbatim reuse caught and fixed, self-retrieval guard confirmed
mechanically enforced) — this is a real, if partial, substitute, not a rubber stamp.

**F6 (informational — corpus scale, already-flagged risk, not new).** 121 messages / 12 threads is
within the method note's own recommended range and was `analyst`-approved as structurally
representative (message/thread counts, near-miss pairs, orthogonal topics), with two non-blocking
caveats already on record (narrative uniformity; wider-than-labeled distractor bleed around the
incident-report template). This baseline characterizes retrieval quality on *this* corpus — a
synthetic, stylistically-uniform, single-generation eval fixture — not on production-scale or
real-usage traffic. That gap was already the method note's #1-ranked risk before this baseline
existed; I'm not discovering it, only confirming it still applies to the numbers as measured and
should stay attached to them (e.g., in any future report text) rather than being read as "retrieval
quality is 97%" in an unqualified sense.

### Verdict: Approve with suggestions

**n=38 is adequate to *establish and record* this baseline** — it's within the method note's own
30–50 target range, the metrics are the right ones and are computed correctly, and the golden set
behind it is well-constructed (Unit 2a's exhaustive `target_text` check, gr-31 fixed, self-retrieval
mechanically guarded). **These numbers are a reasonable floor** in D6's sense and I sign off on
`retrieval_baseline.json` being treated as the frozen reference point.

What should not happen without the F2 fix: treating the recall@10 comparison as a silent,
unconditional CI hard-fail on any future drop, as currently coded. At this sample size, a
single-pair movement (2.6 points) is inside the metric's own noise floor (Wilson 95% CI width ≈13
points, and likely wider still given the thread-clustered, non-independent pair design), so the
zero-tolerance gate as written will produce false "regression" failures indistinguishable from a
real one, and gives essentially no room for a genuinely better change to register (the baseline is
already one flip from ceiling). Recommend F2's tolerance-band fix (or the explicit manual-triage
policy alternative) before the first real future retrieval change is measured against this file —
this is a one-line threshold change to `test_retrieval_eval.py`, not a re-measurement. F3/F4/F5/F6
are attach-as-caveat items for whoever reads or reports these numbers next (the QA pass,
`generate_report.py`, or a future retrieval-change PR), not blockers to committing the baseline as
recorded.

**Sign-off statement for the coordination ledger:** `retrieval_baseline.json`
(recall@10=0.9737/recall@5=0.8947/MRR=0.6259, n=38) is **approved as the recorded baseline**
(D6/M-4 satisfied); its use as an **automatic pass/fail CI gate** should be treated as provisional
pending F2's tolerance-band adjustment — the numbers themselves are sound, the zero-tolerance
comparison built on top of them is not yet right-sized to the sample.

## Scope & verdict

Methodology sign-off on `docs/plans/graphrag-eval.md` v4's D1 (§3), specifically the v4 revision
that collapses Unit 3's judge model onto the same model as the agent-under-test
(`qwen/qwen3-4b-2507`), replacing the prior distinct-judge default (`openai/gpt-oss-20b`). This
is **not** a re-review of the plan's engineering (`analyst` already gated that through Pass 3,
Approve) and it does **not** relitigate the stakeholder's hardware-constraint decision itself —
that call is accepted as given, per the brief. The question in scope: does D1's framing of the
resulting limitation hold up methodologically, and is Unit 3 safe to build against it as written.

Read in full: `docs/plans/graphrag-eval-ml.md` (my own method note, now v2 — I added the addendum
described below), `docs/plans/graphrag-eval.md` v4 (D1 §3, its citations in §5 Unit 3/§6/§7 items
2/8, and the v4 revision note), `docs/reviews/graphrag-eval.md` (all three analyst passes, for the
sign-off-gate precedent D1 mirrors), and `docs/archive/plans/m3-guard-calibration.md` (this
codebase's one prior same-model self-preference precedent — its own risk register already flags
"self-preference (inherited, DS risk #3)... unmeasurable with this set alone" for a different
judge, which is directly on point here).

**Verdict: Approve with suggestions.** D1's framing is honest and correctly gated — it names the
deviation as a deliberate, stakeholder-directed exception rather than absorbing it silently, wires
a machine-readable flag (`sameModelAsAgentUnderTest`) so the report can't lose track of it, and
routes the "is this number meaningful at all" call to `data-scientist` before trusting it,
mirroring D6's existing baseline sign-off pattern rather than inventing new machinery. Unit 3 is
**safe to build** against D1 as currently written. One finding (below) should land before
`generate_report.py`'s caveat text is finalized, because it changes what the caveat has to say,
not just how the mechanism works — but it doesn't block starting the unit, and the plan's own
"sign-off before numbers are trusted" gate already covers the risk of it being missed.

---

## Findings

### Major

**M-1. D1's caveat is correctly triggered but insufficiently precise — it treats "Unit 3's judge
numbers" as one undifferentiated risk class, when the two sub-passes carry structurally different
self-preference exposure.**

Unit 3 (§5) runs two sub-passes against the same judge client, and D1's caveat currently applies
uniformly to "any faithfulness/relevance numbers Unit 3 reports" (§3 D1) and to `generate_report.py`'s
planned output ("a mandatory... self-preference-bias caveat verbatim... alongside... whether the
required `data-scientist` sign-off... has happened yet," §5 Unit 3). That's the right instinct, but
it conflates two claims with very different validity:

- **Calibration sub-pass** (`golden_judge_calibration.jsonl`, ~10 fixed `question/context/answer`
  triples, judge-vs-human agreement per axis). The judge is scoring content it did **not**
  generate — the triples are authored independently (and, per the coordinator-accepted M-5, should
  get a real human spot-check). Self-preference bias is a judge favoring outputs that look like its
  own — there's no self-generated output here for it to favor. This sub-pass is not clean (small-N,
  D4's own caveat; a general judge-quality risk if this exact model has systematic rubric-following
  weaknesses unrelated to authorship), but it is **not** compromised by the specific mechanism D1
  names, and it remains a legitimate signal of whether this model can follow the faithfulness/
  relevance rubric at all on content it didn't write.
- **Generation sub-pass** (~15–20 items, live-generated by the agent-under-test, then judged by the
  same model instance). This is where self-preference bias is squarely in play in its classic
  form — a model rating its own output. A high faithfulness/relevance score here cannot be
  distinguished from "the judge likes how it would have said this itself."

**The dangerous reading D1 doesn't foreclose:** a future reader sees `judge_calibration.json`
report an acceptable agreement number on the calibration sub-pass, concludes "the judge is
calibrated," and then extends that trust to the generation-sub-pass faithfulness/relevance scores —
treating a passing calibration as if it had validated the judge's independence, when calibration
only validates its rubric-following on someone else's content. That is exactly the "must never be
claimed" line the brief asked me to police, and D1's current single blanket-caveat framing does not
make the distinction sharp enough to stop it. This isn't a defect in D1's judgment (the deviation
*was* correctly named and gated) — it's a completeness gap in *what the caveat says*, which matters
because `generate_report.py`'s caveat text is exactly what a future reader will actually see.

**Recommendation — the exact caveat language, adjacent to the numbers, not a footnote** (mirroring
`m3-guard-calibration.md`'s own report-template rule: "the caveat sentence, verbatim, adjacent to
the verdict — not in a footnote," §8 item 3, a precedent already in this codebase for exactly this
situation):

> **Same-model judge limitation.** The judge model (`qwen/qwen3-4b-2507`) is identical to the
> agent-under-test model for this run.
> - *Calibration numbers* (judge-vs-human agreement on fixed, independently-authored triples) are
>   largely unaffected by self-preference bias — the judge did not generate the content it's
>   scoring. Read subject to the existing small-N caveat (D4), not this one.
> - *Generation-sub-pass faithfulness/relevance numbers* (the judge scoring its own model's live
>   output) are structurally exposed to self-preference bias. A high score here is a same-model
>   directional signal, **not independent validation**, and must not be read as if a distinct judge
>   produced it. **A passing calibration number does not license trusting these — they are two
>   different validity claims.**
> - Gross/obvious failures (flat contradiction of the retrieved context, answering the wrong
>   question) likely remain catchable even here — the bias risk concentrates in borderline/
>   subjective calls, where it becomes indistinguishable from genuine quality.
> - `data-scientist` sign-off status on these numbers: **[pending / not yet reviewed]**.

This is a text change to `generate_report.py`'s caveat, not a design change to Unit 3 — no new
field, no new fixture, no new dependency. `judge_calibration.json`'s existing
`sameModelAsAgentUnderTest` boolean is sufficient to gate emitting it; the report just needs to say
the more precise thing when it does.

### Minor

**N-1. The plan doesn't specify what the future `data-scientist` sign-off will actually check —
worth pinning now, mirroring D6's specificity ("is this a reasonable floor," not just "the harness
ran green").** D1 correctly defers the numbers-review to a future `data-scientist` pass ("that call
belongs to `data-scientist`, not to this plan"), but leaves the sign-off's own criteria unstated. So
that the coordinator and Unit 3's implementer know what they're building toward, and so this
doesn't drift into a rubber-stamp when the numbers eventually land, the sign-off should verify at
minimum: (a) the report's caveat correctly distinguishes calibration vs. generation-sub-pass numbers
per M-1; (b) the calibration sub-pass's ~10 labels did in fact get the human spot-check §7 item 3/
M-5 recommends (if it didn't happen, the calibration number itself is on shakier ground, compounding
the generation-sub-pass risk rather than offsetting it); (c) the generation-sub-pass numbers are
reported and read as directional/same-model, never presented as a pass/fail gate the way D6's
retrieval baseline is — this layer should stay descriptive, not gating, for as long as judge and
agent-under-test share a model. **This review does not itself constitute that future sign-off** —
`judge_calibration.json` doesn't exist yet; there are no numbers to sign off on. This review approves
the *plan's framing* of the limitation and specifies what the later numbers-review will require.

**N-2. Worth citing the codebase's own prior precedent for this exact failure mode, for continuity
with prior methodology decisions rather than treating this as a novel question.**
`docs/archive/plans/m3-guard-calibration.md` risk #4 already named "self-preference (inherited, DS
risk #3). If the same 4B both emits the `understanding` and judges it, it is grading its own
homework. Unmeasurable with this set alone" for the intake/research guard judge — a different
judge, same structural issue, same conclusion (report it, don't let it silently inflate a gating
number). D1 reaches the same place independently; citing the precedent in the report or in D1
itself would show this is a recognized, recurring pattern in this lab rather than a one-off call,
and would help a future reader who's seen one of the two documents recognize the other.

**N-3 (open question, not a finding).** If sequential model loading (not concurrent) is compatible
with the stated hardware constraint — i.e., the constraint is about running two models
*simultaneously*, not about ever loading a larger model at all — running the judge pass as a
separate step after generation, against a distinct model loaded only for that step, would fully
restore judge independence for the generation sub-pass at no additional *concurrent* resource cost,
only added wall-clock time (a live, opt-in, offline test already tolerates that). I'm not asking
this be adopted or the hardware constraint reopened as a fact question — per the brief, that's the
stakeholder's call and I'm not second-guessing it. Flagging it once, here, as an option the
coordinator may not have had in front of them when the constraint was framed as binary
(same-instance vs. nothing); if sequential loading isn't viable either, this is moot and no action
is needed.

---

## What's solid

- **The deviation is named, not absorbed.** D1 states plainly that it drops the method note's
  safeguard, why (a real hardware constraint, not an oversight), and that the plan doesn't
  relitigate the trade-off — exactly the right posture for a stakeholder-directed methodology
  exception.
- **The sign-off gate mirrors an existing pattern (D6) instead of inventing new process.** Routing
  "is a same-model judge number meaningful at all" to `data-scientist`, the same way D6 routes "is
  this baseline a reasonable floor," keeps the plan's gating vocabulary consistent and makes both
  gates discoverable the same way.
- **`sameModelAsAgentUnderTest` is the right machine-readable carrier.** Deriving it from the
  resolved model refs (rather than hand-maintaining a flag) means it can't silently go stale if an
  operator overrides `FALKORCHAT_LIVE_JUDGE_MODEL` back to a distinct model later — the caveat and
  the sign-off requirement disappear automatically when the condition that triggers them does.
  §5 Unit 3 confirms this is derived at run time from the two env-var-literal model refs, not a
  static default baked into the report generator.
- **The config-wiring mechanism (D1's option (b), env-var literal) is untouched by this change** —
  only the *default value* moved. An operator who can run two distinct models still gets full
  judge independence for free by overriding the env var; nothing about this exception forecloses
  that path for a future run on different hardware.
- **The judge layer stays non-gating regardless.** Nothing in D1/D6 makes Unit 3's numbers block a
  CI run the way Unit 2b's `retrieval_baseline.json` does — the generation-sub-pass risk this review
  flags (M-1) is about *misreading* a descriptive number, not about it silently failing a build.
  That containment already limits the blast radius of the finding above.

---

## Recommended report caveat (for the implementer of `generate_report.py`)

See M-1's block-quoted text above — that is the concrete language recommended, to be emitted
whenever `judge_calibration.json`'s `sameModelAsAgentUnderTest` is `true`, placed adjacent to the
judge numbers themselves (not in a trailing footnote), per the `m3-guard-calibration.md` precedent
this plan already borrows its report-honesty posture from.

## Method note addendum

`docs/plans/graphrag-eval-ml.md` bumped to **Version: 2** with a dated addendum under "Layer 2"
(2026-08-15): names this as an accepted, sign-off-gated exception for K-026's Unit 3 baseline only,
explicitly not a retraction of the general "never the 4B-under-test judging itself" guidance for
future work, and states the same calibration-vs-generation-sub-pass distinction as M-1 above so the
method note and this review stay consistent with each other.

## Open questions

- N-3 above (sequential vs. concurrent model loading) — not blocking, surfaced once for the
  coordinator's awareness.
- Whether the coordinator wants N-1's sign-off criteria folded into Unit 3's own "Done when" text
  (mirroring how D6's sign-off gate is already written into Unit 2b's done-condition) or left as
  this review's standalone specification, referenced when the numbers-review actually happens. Not
  my call to make unilaterally since it's plan-text, not method-note text — flagging for `architect`/
  `teco` to fold in if they agree it's clearer that way.
