# Typed Decision Models: U6f Fit Rulings (method note)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — · **Extends:** `docs/plans/typed-decision-models-ml.md` · **Reviews:** `docs/reviews/typed-decision-models4-impl.md` (Pass 38)

Rulings for plan4 §3.11/§4.4 (`docs/plans/typed-decision-models4.md`) that note §Q2 leaves open. The
note's ladders, margins and outcome order (§Q2.0–§Q2.3) stand; this note states only what they do not.
Where it says "the family", it means the decision-only candidate's empty-band search space.

## 1. The decision-only family

**Definition (confirmed).** A predicate's family is `yesAtOrAbove == noBelow == t`, with `t` drawn from
the distinct calibration `p` plus `{0, 1}`; a choice's family is the single point `minConfidence = 0`.
- `apply_policy` reads `p >= yesAtOrAbove` as yes and `p < noBelow` as no, so the band is empty and
  `handoffModel: null` loads. `t` ranges over every partition of the calibration `p` that the policy
  language can express, so nothing is lost by the choice of candidate values.
- `s = 0` always. A choice has no search: it is the model's `choice`, and the new-ability ladder (§Q2.2)
  reports its single point, rung 4's max-min agreement included.
- The hand-off calls are still collected for every item: the guard's `D_c` needs the LLM's answer
  (calls are shared, §3.11). An over-limit item is concordant (guard) or fail-closed (new abilities),
  per note §Q4.3's "no hand-off model" row. It adds 0 to every `D_c`, has no score, and is no hand-off.
- **Latency terms.** With no band there is no `t̄_{L|H}` over band items. Take it over every calibration
  hand-off call. Then `s_2` (guard) and `s_350` (new abilities) are the ceilings the fixed `s = 0` is
  tested against: `s = 0 ≤ s_x − 0.05` selects the margin rungs, `s = 0 ≤ s_x` the next. If `s_x < 0` even
  `s = 0` misses the latency ceiling.

**Reading of the guard ladder on the family.** Rungs 1 and 3 coincide unless `0 ≤ s_2 < 0.05`, and rung 2
and rung 3 likewise; the ladder runs as written with `s` fixed. Rung 4 (`D_c ≤ 0`, latency ignored,
`fit-infeasible: latency`) is attainable only if some `t` gives `D_c ≤ 0` in all four cells; "rung 4
always exists" holds for the hand-off family alone (`s = 1`), not here.

## 2. No quality rung reached, guard decision-only (P38-M1)

**The candidate carries the `t` that minimises the McNemar sort key `(max_c D_c, max_c m_c)` over the whole
family, the non-rejection constraint dropped.** Exact ties go to the smaller `t`; the sweep is ascending, so
this is the first minimum.
- **Why this key.** It is the key the McNemar rungs already sort by (§Q2.1): the least worst-cell excess,
  then the least discordance. The fallback is that rung with its constraint relaxed, so the carried `t`
  is the one that comes closest to the requirement value and the report's "unmet" claim is about the best
  setting, not an arbitrary one. A `t` chosen for latency or for a mean error would be a different
  objective from the bar's.
- **Rejected:** minimising total errors across cells (FR-1 is cell-wise, so a good sum can hide one failing
  cell); the `t` nearest 0.5 (no calibration basis for the model's natural boundary).
- **Outcome (confirmed).** `fit-infeasible: agreement` with `qualityTarget` `mcnemar-nonreject`. The name
  fits: the requirement value (FR-1's calibration McNemar not rejecting) is what is unmet, the closed set
  has no parity-specific name, and `_fit_problem` (`model-bench/modelbench/decisions.py:463`) tests
  outcome and target independently, so the pair loads. The §Q2.0 order still applies: `ranking` (section 3,
  when no text is eligible) outranks `agreement`, and `agreement` outranks `latency`.
- **Latency-only miss.** If a quality target is attainable and only the latency ceiling is not
  (`s_2 < 0`), the family has no rung 4 in §Q2.1's literal form. The outcome is `fit-infeasible: latency`
  with the first quality target attainable: `D_c<=0`, else `mcnemar-nonreject`; `t` is the §Q2.1 rung-1
  minimiser of the sort key at that target.
- **Reading.** The carried `t` is a least-bad setting, not a claim to pass. The candidate is evaluated once
  as any `fit-infeasible` candidate is (§Q2.0), so the verdict is measured.

**Evaluation (U6f.6).** Fixtures: (a) a family where no `t` is non-rejecting, with three `t` having keys
(2,5), (2,3), (3,1): the (2,3) one is carried, `fit-infeasible: agreement`, `mcnemar-nonreject`; (b) two
`t` with identical keys: the smaller is carried; (c) a non-rejecting `t` exists: the ordinary McNemar rung
wins and the fallback is not reached; (d) the `ranking` outcome beats `agreement` when no text is eligible (section 3).

## 3. Selection among instruction texts

**Eligibility comes first: a text whose calibration AUC is at or below 0.5 leaves the selection while any
text still in the selection has AUC above 0.5.** A text that left on a decision-call failure
(section 4) is not in the selection, so it does not count: it cannot be shipped, and counting it would
let `noEligibleText` differ between a fit that dropped it and one that never had it. The AUC is §Q2.3's (over-limit items excluded; a constant score is 0.5, so it
leaves). A rung reached by an exhaustive threshold search on a reversed or uninformative score is a
calibration artifact, and the AUC, a rank statistic over every item, is a steadier signal than the
winning rung. So a lower-rung text with AUC above 0.5 beats a higher-rung text with AUC at or below 0.5,
and the outcome is the chosen text's rung. When no text is eligible, selection runs over all texts as
below and the fit records `fit-infeasible: ranking`; the candidate is still evaluated (§Q2.0). The
control is subject to the same filter. Ties are decided among eligible texts only.

**Evaluation (U6f.8b).** (a) A at rung 1 with AUC 0.48, B at rung 3 with AUC 0.70: B, `below-margin`, no
`ranking`; (b) every text AUC ≤ 0.5: the literal rule picks the best rung, outcome `ranking`; (c) the
control at AUC 0.5 and a longer text at AUC 0.6 on a lower rung: the longer text; (d) the control
dropped and two eligible texts tied: the earlier in `questions.json`; (e) a text at AUC exactly 0.5
leaves; (f) a text that left on a decision-call failure and alone has AUC above 0.5: `noEligibleText`
true, outcome `ranking`.

**Selection is per candidate, and eligibility is not.** The hand-off and the decision-only candidate each
ship the text their own family selects. Eligibility is a property of the text, not the family: the AUC
is of `p` (or `confidence`) against the label over the same calibration items, and a text dropped for a
decision-call failure is dropped for both candidates, since both read the same decision calls. So
`noEligibleText` is true for both candidates or for neither, `ranking` is recorded on both or on neither,
and the two candidates can differ only in rung and key among the same eligible texts, hence possibly in the
shipped text. That is acceptable: each is a complete artifact, and no criterion ranks a text across
families. Fixture: with `noEligibleText`, both selections carry it and both outcomes are `ranking`.

**Then** the best rung, then that rung's key, then the control; a tie is exact equality of rung and
key, compared as integers and exact fractions, never floats. Among tied non-control texts, file order in
`questions.json`.
- **Why the control.** The best of up to five texts on one calibration split carries a winner's curse; the
  control is the one text whose quality does not depend on rubric wording (§Q2.3), so it is the
  conservative default when calibration cannot separate the texts.
- **No margin.** A rule that makes a longer text beat the control only by some margin would be a tuning
  knob with no data to set it. The check against the selection effect is the eval split, which no fit
  touches, plus the split-half shift (§Q2.4).
- **Evaluation (U6f.8).** A fixture with the control and a longer text at the same rung and key selects the
  control; one with a key better by a single count selects the longer text; the float form of a tie (an
  equal `s` computed as 0.3 versus 3/10) still ties.

## 4. Abort on any chat failure

**Sound for the fit, with three additions.** A fit scores a frozen calibration set on the served artifact.
Dropping a failed item would change the set's `n`, its cell floors (§Q2.4) and its composition after the
data was seen; the bench's withhold-and-count accounting is right for a run because a run measures a
verdict. Abort keeps the fit's set whole, and a rerun costs minutes. A parse failure is data and scored
on its fallback exactly as the bench scores it (§Q4.3's last row), counted per kind.

**What the fit cannot see.** It sees no decision-call timeout rate; the bench keeps and counts them (a
concordant pair for the guard, a fail-closed answer for new abilities, which lowers agreement). Bound it
rather than ignore it. Three report fields, descriptive and never gating:
- per call kind, the count `k` of calls that are a listed disposition (0 when none are listed, since any
  other failure aborts), the call count `n`, and the Wilson 95% upper bound for `k` of `n`,
  `(p̂ + z²/2n + z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)` with `p̂ = k/n`, `z = 1.96`: the most the fit's
  agreement can overstate the bench's. At `k = 0` it is `z²/(n + z²)` (about 0.76% at `n = 500`, 1.26% at
  `n = 300`);
- the number of calls above half of the call's timeout budget (`decisionTimeoutSeconds` for decisions,
  180 s for hand-offs), beside the existing median and maximum.

**The abort record.** An abort writes one small record and no candidate, manifest or report: the
operator needs the evidence, and "nothing is written" means no artifact. Plan4 §3.11's `AbortRecord` row
is the sole home of its fields.

**Operator procedure when a call persistently fails.**
1. Read the record. A `connection` failure, or a `timeout`, is not evidence about the item (server load
   and the network decide it): check residency, rerun the whole fit once. An `http` or `malformed`
   failure on a resident pair is treated as the item's until a rerun shows otherwise, so the same rerun
   is the test.
2. A decision-call timeout: the lever is `--decision-timeout-seconds`, which is also the value the
   candidate ships and the bench will apply. Rerun with a larger value only if it is a value the runtime
   can accept; if the model needs more than the runtime's budget on calibration items, that is a finding
   about the model and the point stops there (report to `architect`).
3. The same item fails the same call twice: the operator does not edit, drop or reorder the item, the
   prompt or the constants (parity-bound; an item changed after a fit has seen it is a selection on the
   outcome). Stop and take the dispositions below, which are the architect's rule in plan4 §3.11.

**Recorded item disposition: method-sound, with these rulings.** The preconditions are plan4 §3.11's
(two matching records, `inputs` equal to the run's, a ledger line first). Method rulings on them:
- **Evidence kinds.** `http` and `malformed` are evidence when both models are resident in both records;
  `timeout` and `connection` never are. Reason: at temperature 0 the same item failing the same call twice
  with the same answer is the item's property. A hand-off 5xx on an oversized input repeats exactly,
  while a transient 5xx on one item twice running, with both models resident, is improbable. The
  records must also agree on `status` (plan4 §3.11 check (b)), so a 500 and a 503 are not pooled. A
  `timeout` stays ineligible even though a degenerate generation could repeat it: it depends on load, the
  lever is `--decision-timeout-seconds`, and an item that still cannot run goes to `architect`.

The disposition follows the bench's row for that call (§Q4.3), so the fit sees what the bench
will:
- **Guard, hand-off chat failure:** the item leaves the calibration set, for every text and both
  candidates. This is consistent, not a distortion: both candidates share one calibration set and one set
  of hand-off calls (§3.11); the `only` candidate's `D_c` needs the LLM's answer, which this item lacks;
  and at a run the same call is the incumbent's judge call (same prompt, same budget), so the pair is
  dropped from both candidates' evaluations too. A cell under the §Q2.4 floor refuses the fit for both.
- **Guard, decision-call failure:** the item is not excluded. For a non-control text, that text leaves
  the selection (a text the served artifact cannot answer on a calibration item within its budget is not
  a text to ship; the other texts keep the item, so no text gains from a hard item). If the control fails,
  or every text does, the item stays and is concordant for that call, the bench's row.
- **Join-in and message type, hand-off or decision failure:** the item stays and the failed call is
  fail-closed (the answer "no" for join-in, "no label" for message type). This estimates the bench's
  agreement rather than biasing it: the failure belongs to the item, and the eval split holds such items at
  the same rate, so dropping them would be the optimistic bias. It acts only where the item is handed
  off (or decided by that text), so the ladder sees the cost as a price on a larger `s`. The failed
  call's elapsed time enters `t̄_{L|H}` (§Q4.3, FR-10 row); a 180 s timeout can lower `s_350` visibly,
  which is the correct, conservative reading.
- **Effect on the T ladder.** An item fail-closed in a class lowers that class's agreement by
  `1/n_class` wherever it is handed off. At 60 per class one item costs 1.7 pp, so one item can move the
  chosen rung (0.94 against 0.93). The report prints, per class, the fail-closed count `k` and `n_class`,
  and the agreement ceiling `1 − k/n_class` for a band that hands them off.
- **No numeric cap: sound.** The floor re-check bounds the guard, fail-closed scoring bounds the
  new-ability points (the loss is measured, not hidden), and each exclusion costs two aborted fits and a
  ledger line. The report prints exclusions as `k/n` with the Wilson 95% upper bound for the rate.

**Evaluation.** Dispositions: a guard hand-off failure removes the item for both candidates and refuses
under a floor; a guard decision failure on a non-control text drops that text, on the control keeps the
item concordant; a new-ability failure keeps the item fail-closed with the failed call's elapsed time in
`t̄_{L|H}`; two `http` records with statuses 500 and 503 are refused. Summaries: the Wilson bound is a
literal for `k = 0` and for `k = 1` at `n = 300`; the half-budget count is exact on a fake clock. The
abort-record fixtures are plan4 §5's U6f.8a row.

## 5. Risks and open points

- Observed `p` values as thresholds put a boundary on an item; a replicate's tiny numerical drift can flip
  that item. This is the hand-off family's behaviour too and the note's, so it is kept; a midpoint
  between adjacent distinct `p` would be robust and is an amendment to §Q2.1 for both families, not made here.
