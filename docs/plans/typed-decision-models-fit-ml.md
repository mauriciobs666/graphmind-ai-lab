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
  hand-off call (an over-limit item adds its hand-off call's time only, never the failed decision
  call's, which sits in `t̄_d`). Then `s_2` (guard) and `s_350` (new abilities) are the ceilings the fixed `s = 0` is
  tested against: `s = 0 ≤ s_x − 0.05` selects the margin rungs, `s = 0 ≤ s_x` the next. If `s_x < 0` even
  `s = 0` misses the latency ceiling.

**Reading of the guard ladder on the family.** Rungs 1 and 3 coincide unless `0 ≤ s_2 < 0.05`, and rung 2
and rung 3 likewise; the ladder runs as written with `s` fixed. Rung 4 (`D_c ≤ 0`, latency ignored,
`fit-infeasible: latency`) is attainable only if some `t` gives `D_c ≤ 0` in all four cells; it is
never reported here (§2 controls), and on the hand-off family it can be missing too (§2).

## 2. No quality rung reached, guard (P38-M1, P47-M1)

**Hand-off family.** A band hands off `l ≤ p < h` with `h ≤ 1` (plan3 §3.2), so an item at `p = 1.0` is
decided *yes* in every band; with label *no* and the LLM right its cell has `D_c ≥ 1` everywhere and
rung 4 can be missing (`p = 0.0` is not stuck). Whether a 4B model returns exactly 1.0 is unverified (the client accepts the closed
`[0, 1]`). Ruling: a missing rung 4 is an outcome, not an error. After rung 4 the family takes the
decision-only steps below, latency ignored: `mcnemar-nonreject` if attainable, with `rung` `None`,
`fit-infeasible: latency` and the rung-2 minimiser; else `fit-infeasible: agreement`, `mcnemar-nonreject`, and the `(max D_c,
max m_c, s)` minimiser (first on an exact tie). `LadderResult` carries that setting's `policy` and
`key`; `rung` `None` is valid for `family="handoff"` only with `qualityTarget` `mcnemar-nonreject`.
The fit never raises on it. The report prints one number, the *stuck* count: items (after `--exclude-item`, with an `ok` decision call) with `p == 1.0`, label *no* and the LLM's answer correct. The bounds stay (plan3 §3.2);
`h > 1` would change the runtime contract.

**Decision-only family.** **The candidate carries the `t` that minimises the McNemar sort key `(max_c D_c, max_c m_c)` over the whole
family, the non-rejection constraint dropped.** Exact ties go to the smaller `t`; the sweep is ascending, so
this is the first minimum.
- **Why this key.** It is the McNemar rungs' own key (§Q2.1) with the constraint relaxed, so the carried
  `t` comes closest to the requirement value and "unmet" is claimed about the best setting.
- **Rejected:** total errors across cells (FR-1 is cell-wise; a good sum can hide a failing cell); the `t`
  nearest 0.5 (no calibration basis).
- **Outcome (confirmed).** `fit-infeasible: agreement` with `qualityTarget` `mcnemar-nonreject`. The name
  fits: the requirement value (FR-1's calibration McNemar not rejecting) is what is unmet, the closed set
  has no parity-specific name, and `_fit_problem` (`model-bench/modelbench/decisions.py:463`) tests
  outcome and target independently, so the pair loads. The §Q2.0 order still applies: `ranking` (section 3,
  when no text is eligible) outranks `agreement`, and `agreement` outranks `latency`.
- **Latency-only miss.** If a quality target is attainable and only the latency ceiling is not
  (`s_2 < 0`), the family has no rung 4 in §Q2.1's literal form. The outcome is `fit-infeasible: latency`
  with the first quality target attainable: `D_c<=0`, else `mcnemar-nonreject`; `t` is the §Q2.1 rung-1
  minimiser of the sort key at that target.
- **Reading.** The carried setting is least-bad, not a claim to pass; it is evaluated once (§Q2.0).

**Evaluation (U6f.6).** Fixtures: (a) a family where no `t` is non-rejecting, with three `t` having keys
(2,5), (2,3), (3,1): the (2,3) one is carried, `fit-infeasible: agreement`, `mcnemar-nonreject`; (b) two
`t` with identical keys: the smaller is carried; (c) a non-rejecting `t` exists: the ordinary McNemar rung
wins and the fallback is not reached; (d) the `ranking` outcome beats `agreement` when no text is eligible (section 3).

## 3. Selection among instruction texts

**Eligibility comes first: a text whose calibration AUC is at or below 0.5 leaves the selection while any
text still in the selection has AUC above 0.5.** A text that left on a decision-call failure
(section 4) is not in the selection and does not count, so `noEligibleText` cannot differ between a fit
that dropped it and one that never had it. The AUC is §Q2.3's (over-limit items excluded; a constant score is 0.5, so it
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

**Selection is per candidate, and eligibility is not.** Each candidate ships the text its own family
selects. Eligibility is a property of the text: the AUC is of `p` (or `confidence`) against the label over
the same items, and a text dropped for a decision-call failure is dropped for both candidates. So
`noEligibleText` and `ranking` hold for both candidates or neither; they can differ only in rung and key
among the same eligible texts, hence possibly in the shipped text (each is a complete artifact). Fixture:
with `noEligibleText`, both selections carry it and both outcomes are `ranking`.

**Then** the best rung, then that rung's key, then the control; a tie is exact equality of rung and
key, compared as integers and exact fractions, never floats. **A `rung` `None` text ranks below every
rung 1–4 text** (the fallback has not reached a quality bar plus latency), and among `None` texts by
`agreementUnmet` false before true (a latency-only miss meets the quality requirement), then
`qualityTarget` `D_c<=0` before `mcnemar-nonreject`, then the key (same family and target, so
comparable), then the control, then file order. A `None` text stays eligible (eligibility is AUC only), so
a candidate whose every text is `None` still selects one and reports `noEligibleText` only by the AUC rule.
Fixtures: a rung-4 text beats a `None`-latency text; a `None`-latency text beats a `None`-agreement text
with a better key; two `None`-agreement texts rank by `(max D_c, max m_c, s)`; all-`None` with AUC above
0.5 gives no `noEligibleText`. Among tied non-control texts, file order in
`questions.json`.
- **Why the control.** The best of up to five texts on one calibration split carries a winner's curse; the
  control is the one text whose quality does not depend on rubric wording (§Q2.3), so it is the
  conservative default when calibration cannot separate the texts.
- **No margin.** A margin would be a tuning knob with no data to set it; the check against the selection
  effect is the untouched eval split plus the split-half shift (§Q2.4).
- **Evaluation (U6f.8).** A fixture with the control and a longer text at the same rung and key selects the
  control; one with a key better by a single count selects the longer text; the float form of a tie (an
  equal `s` computed as 0.3 versus 3/10) still ties.

## 4. Abort on any chat failure

**Sound for the fit, with three additions.** A fit scores a frozen calibration set on the served artifact.
Dropping a failed item would change the set's `n`, cell floors (§Q2.4) and composition after the data was
seen; withhold-and-count is right for a run, which measures a verdict. Abort keeps the set whole. A parse failure is data and scored
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
1. Read the record. A `connection` failure or `timeout` is not evidence about the item: check residency,
   rerun the whole fit once. An `http` or `malformed` failure on a resident pair is treated as the item's
   until a rerun shows otherwise, so the same rerun is the test.
2. A decision-call timeout: the lever is `--decision-timeout-seconds`, also the value the candidate ships
   and the bench applies. Rerun with a larger value only if the runtime can accept it; if the model needs
   more than the runtime's budget, that is a finding about the model (report to `architect`).
3. The same item fails the same call twice: the operator does not edit, drop or reorder the item, the
   prompt or the constants (an item changed after a fit has seen it is a selection on the outcome). Stop
   and take the dispositions below (plan4 §3.11).

**Recorded item disposition: method-sound, with these rulings.** The preconditions are plan4 §3.11's
(two matching records, `inputs` equal to the run's, a ledger line first). Method rulings on them:
- **Evidence kinds.** `http` and `malformed` are evidence when both models are resident in both records;
  `timeout` and `connection` never are. At temperature 0 the same item failing the same call twice with
  the same answer is the item's property (a hand-off 5xx on an oversized input repeats exactly; a
  transient 5xx twice running is improbable). The records must agree on `status` (plan4 §3.11 check
  (b)), so a 500 and a 503 are not pooled. A `timeout` depends on load: its lever is
  `--decision-timeout-seconds`, and an item that still cannot run goes to `architect`.

- **The exclusion must fire.** If an excluded call succeeds in the rerun, the fit refuses before writing
  any artifact (report-and-refuse, naming the exclusion): the disposition's premise, a failure that is
  the item's property, is falsified, and applying it anyway would drop an item on a call that can succeed,
  a selection on the outcome. The operator reruns without the flag and the success is ordinary data. The
  call's elapsed time stays out of the refused fit. `summarize_calls` takes every call, before any
  exclusion pruning, so the failed calls count in `k` and `n`.

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
  the same rate; dropping them would be optimistic. It acts only where the item is handed
  off (or decided by that text), so the ladder sees the cost as a price on a larger `s`. The failed
  call's elapsed time enters `t̄_{L|H}` (§Q4.3, FR-10 row); a 180 s timeout can lower `s_350` visibly,
  which is the correct, conservative reading.
- **Effect on the T ladder.** An item whose fail-closed answer is wrong lowers its class's agreement by
  `1/n_class` wherever it is handed off: every message-type item, and a join-in item only when its label
  is *yes* (the fail-closed *no* is right on a label-*no* item). At 60 per class one item costs 1.7 pp,
  so one item can move the chosen rung. The report prints, per class, `n_class`, `k` (distinct items
  with at least one disposition) and `kWrong` (those whose fail-closed answer is wrong), and the agreement
  ceiling `(n_class − kWrong)/n_class` for a band that hands them off. A ceiling below the attainable
  agreement would not be a ceiling, so `k` alone is not used. The guard has no per-class figure; it
  prints the exclusion `k/n` and the floor re-check. With more than two matching records, the first valid
  pair in file-name order is used.
- **No numeric cap: sound.** The floor re-check bounds the guard, fail-closed scoring bounds the
  new-ability points, and each exclusion costs two aborted fits and a ledger line. The report prints
  exclusions as `k/n` with the Wilson 95% upper bound.

**Evaluation.** Dispositions: a guard hand-off failure removes the item for both candidates and refuses
under a floor; a guard decision failure on a non-control text drops that text, on the control keeps the
item concordant; a new-ability failure keeps the item fail-closed with the failed call's elapsed time in
`t̄_{L|H}`; two `http` records with statuses 500 and 503 are refused. Summaries: the Wilson bound is a
literal for `k = 0` and for `k = 1` at `n = 300`; the half-budget count is exact on a fake clock. The
abort-record fixtures are plan4 §5's U6f.8a row.

## 5. New-ability ladders are total (P47-M1)

An item decided wrongly at every setting (predicate `p = 1.0` against a *no* label; a choice with
`confidence = 1.0` and the wrong value, since `minConfidence ≤ 1`) lowers its class's agreement by
`1/n_class` wherever it sits, but rung 4 is a maximiser, so §Q2.2's ladder always returns a setting and
never raises. U6f.7 carries no fallback; it prints per true-answer class the *unfixable* count: items (after `--exclude-item`, with an `ok` decision call) decided wrongly at every setting, i.e. predicate `p == 1.0` with label *no*, choice `confidence == 1.0` with a `choice` other than the label, beside the §4 agreement ceiling.

## 6. Selection and report figures (U6f.8b)

- **AUC.** Predicate: score `p`, target the label. Choice: score `confidence`, target "the choice equals the label"
  (§Q2.5's reliability ranking). An AUC undefined (no `ok` call, or one target class absent) is 0.5, so the
  text leaves like a constant score; the report flags it `aucUndefined` so 0.5 is not read as measured.
  `TextScore.auc` is a property of the text, identical in both candidates' records.
- **Rank order** `(rung, None as 5, agreementUnmet, target, key, not control, file order)`: `target` puts
  `mcnemar-nonreject` after `D_c<=0` within a rung (guard rung 3 can mix them with incomparable keys); new-ability `T` targets tie on it and the key decides.
- **Leavers.** `leftSelection="decision-failure"` is the guard's only (§4); new abilities stay fail-closed with no leaver.
- **Over-limit `(k, n)`:** `k` over-limit decision calls, `n` all of the text's decision calls, failed included.
- **`max(probabilities)` AUC** (choice, exploratory, gating nothing): a call with no entry is dropped from its pairs
  and counted in `maxProbDropped`; `None` when a class is absent.
- **Split-half** (§Q2.4): the ladder fitted on one half, the fitted policy scored on both, both directions; the shift
  is held-out minus fit-half in `s` and each `D_c` (guard) or class agreement (new abilities). Print each
  half's per-cell (class) `n`; a cell absent from a half is reported absent, never as 0.

## 7. Risks and open points

- Observed `p` values as thresholds put a boundary on an item; a replicate's numerical drift can flip it.
  Both families share this and it is kept; a midpoint between adjacent distinct `p` would be an amendment
  to §Q2.1 for both, not made here.
