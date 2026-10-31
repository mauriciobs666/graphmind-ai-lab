# Typed Decision Models — Method Note (statistics, data, fitting)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — · **Version:** 1.6 · **Extends:** `docs/plans/typed-decision-models.md` · **Reviews:** `docs/reviews/typed-decision-models.md` (Pass 8)

*2026-10-31 — v1.6: answers review Pass 8; over-limit hand-off is the stakeholder's decision. The probe facts cited here are in `ws:agent-team` document 480ef825… (the U1a probe).*

## 0. The question and the decision it serves

This note answers plan §4 U0's Q1–Q7. `modelbench/decision_stats.py` implements §Q3–§Q6 and no
other source (§3.6). `decision-models/` implements §Q1, §Q2 and §Q7. The decision served: does a
point's bar (FR-1 + FR-2, or FR-10) hold on evidence that none of these can explain?
- too few items (R-6);
- selection on a reused set (R-5);
- the generator agreeing with itself (R-4);
- one lucky model load (§Q6).

**Conventions.**
- The analysis unit is the item, and there is one item per seed (§Q1.3).
- Rates print as `k/n` with a Wilson 95% interval (`stats.wilson_interval`).
- Percentiles come from `stats.percentile` only.
- A rate threshold is evaluated in integers (`10·k ≥ 9·n`).
- Decision-shaping arguments take no default.
- "pp" means percentage points.

**Governing principles.**
- Where the runtime takes a decision on an item, the item enters the bar at that decision.
- Where the runtime takes none (its run fails), the item is withheld and counted (§Q4.3).
- No correction, floor, margin or rule may make a bar easier to meet (§Q3.2, §Q2.0).

## Q1 — Data generation

### Q1.1 Scenario taxonomy

Each item comes from one **seed**, drawn by stratified sampling.

| Point | Seed axes |
|---|---|
| `guard-judge` | condition text × domain × evidence kind (`understanding`/`turns`) × intended answer × difficulty (`clear`/`near-miss`) |
| `join-in-judge` | thread purpose × addressee cue (agent named but not @mentioned; question to the room; human-to-human; reply to the agent's last turn; small talk) × intended answer × difficulty |
| `message-type-classifier` | intended label (5) × domain × thread position (`opening`/`follow-up`) × difficulty |

- **Guard conditions.** A third of guard seeds use production's one condition ("the user has
  provided enough information to research their request"). The rest spread over ≥ 8
  generator-authored conditions of the same shape, so a candidate is not fitted to one sentence.
  `turns` seeds vary from 1 to 10 turns, which exercises the renderer's last-6 cut.
- **Near-miss** seeds hold back one material fact, or state it ambiguously. They are a third of
  every point's seeds and the synthetic counterpart of the side check's `boundary` tier. Without
  them, the eval set is easy and hides differences.
- **Intent is a sampling variable, never a label.** No item is kept or dropped for agreeing with
  its intent.

### Q1.2 Blind labeling (R-4)

- **Generator ≠ labeler.** Both are hosted frontier models (FR-6, FR-8), but with two distinct
  model ids, so no model labels inputs it wrote. `generation.json` records both. `labels.labeler`
  is the labeler (AC-4).
- **Blind.** The labeler sees only the rendered decision input (§3.2) plus a fixed rubric. It
  never sees the seed, the intent, the difficulty or the generator prompt.
- **Majority of three** independent calls at the provider's default temperature. Calls at 0 would
  be one call three times.
  - A choice item with three different labels has no label. It is dropped and counted.
  - If more than 3% of a point's items drop, fix the rubric before drawing eval.
  - Nothing else is filtered: items are kept whatever their vote split or intent agreement.
- **Diagnostics** (exploratory, per answer class):
  - unanimity (the 3–0 share), the labeler's own consistency ceiling; near 0.90, FR-10's bar
    sits at label-noise level.
  - intent-label agreement.
- `labelVotes` is stored per item for §Q7. It maps every answer in the point's answer space to its
  vote count, and the counts sum to 3. Predicate answers are `yes` and `no`; a choice uses all five
  labels, zeros included. It is not an input field, so it never enters `input_hash`.

### Q1.3 Clusters, splits and sizes

- **One item per seed, in every split.** Cluster = seed, and splits are drawn by seed with a
  recorded RNG seed. One item per seed is what makes items independent and keeps
  `designEffect = 1.0` true (§Q6). Minimal pairs (two variants of one seed) are a U8-only option,
  priced separately if U8 wants them.
- **Shingle check.** An eval item that shares an 8-token span with any train or calibration item
  is dropped and counted. Condition sentences are exempt. Tokens are produced by:
  1. NFKC-normalizing and lowercasing the text;
  2. replacing every character that is not a Unicode letter or digit with a space;
  3. splitting on whitespace.

  The check covers the rendered evidence fields only.
- **Filling cells** by random subsampling within (label × stratum). Generate about 1.6× the
  target, since labels will not match intents one for one.
- **Length is reported, not stratified.** An input over the 512-token limit is a forced hand-off
  (§Q4.3). The question's own tokens come out of that budget: about 44 for a short question, by
  the probe's arithmetic. That leaves at most about 1,870 characters of evidence, at an estimated
  ~4 characters per token (Laya's tokenizer is unmeasured), and less with longer instructions.
  Every point's render cap exceeds that: about 2,800 characters of turn text for join-in, 2,400
  for message type and 6,000 for the guard (`JUDGE_USER_MAX_CHARS`). The 85 side-check items
  render at 134–436 characters, far inside the limit. There
  is no length quota, because the generator's mix is not production's. Sizes do not change,
  because over-limit items are scored (FR-11). Production's share comes from `record_decision`'s
  `reason=over-limit` (FR-3).

| Point | Cell | Eval per cell | Eval total | Calibration | Train (U8 only) |
|---|---|---|---|---|---|
| `guard-judge` | kind × label (4) | **100** (floor 50) | 400 | = eval (floor 50/cell) | ≥ 1,000 |
| `join-in-judge` | label (2) | **100** (FR-11 binds) | 200 | = eval | ≥ 1,000 |
| `message-type-classifier` | label (5) | **60** (floor 52) | 300 | = eval | ≥ 1,500 |

These sizes **replace the plan's prior of ~50 per cell**. FR-11's 200 is a floor, not a target.
- **Guard: 100 per cell.** At 50 per cell, parity can only detect a 13.1–19.9 pp worsening
  (§Q3.3), against stored incumbent error rates of 10–13%. At 100 per cell that falls to
  6.7–14.1 pp. The cost is about 5 min per chat arm (§Q4.4).
- **Message type: 60 per label.** "About 40 per label" is just 200 / 5. The criterion is that one
  class truly at 0.85 passes with probability ≤ 0.20 (§Q5.3).
  - Under `10·k ≥ 9·n` that probability is a sawtooth in n, with local worsts at multiples of 10:
    0.263 at 40, 0.219 at 50, 0.185 at 60.
  - Every n ≥ 52 meets the criterion, so **52 is the floor** and 60 the round design size.
  - A class withheld below 52 shows in its printed n (`docs/plans/typed-decision-models2.md` §3.6).

**Labelling volume (stakeholder-accepted 2026-10-30).**
- **Kept items.** Eval is 900 (400 + 200 + 300), and calibration matches it: 1,800 kept items.
- **Labelling.** With 1.6× over-generation that is about 2,880 items labelled, × 3 votes ≈ 8,640
  labeller calls. That is about 14× the plan's prior reading of 600 single-labelled items. There
  is also one generator call per seed.
- **If U8 runs,** its train split (≥ 3,500 kept) adds about 16,800 calls, about 42× in total.
- `fr10.py` prints this arithmetic.

### Q1.4 Side-check items are never generation material

- The 85 human-curated items (`model-bench/packs/guard-judge-understanding/items.jsonl`) are never
  seeds or few-shot examples, and never appear in any prompt. `decision-models/` cannot read them,
  under the FR-8 path test (§3.4).
- A paraphrase would evade the exact-hash check, so U5 also drops and counts any generated item
  that shares an 8-token span (§Q1.3 tokenization) with a side-check item's
  `understanding`/`turns` text.

### Q1.5 Message type takes the preceding turn

**Yes.** The classifier sees every non-assistant message, and most of those are mid-thread. "Yes,
the blue one" is an `order`, and "third time I ask" is an `escalation`, only in context. Without
context, labels are noisy and the eval rewards guessing.
- Input fields: `text` and `priorTurn` (`{speaker, text}` or `null`). `priorTurn` is the row
  immediately before the message, taken as join-in takes `prior_turns` (§3.2.1).
- `speaker` follows `guards._recent_turns`' rule: the row's `displayName`, else `authorId`, else
  `member`. Join-in uses the same rule. Generated items carry display names in that form.
- Rendered as
  `PREVIOUS MESSAGE:\n{speaker}: {priorTurn.text[:400]}\n\nMESSAGE:\n{text[:2000]}`. The first
  block is omitted when `priorTurn` is `null`.
- Eval is about 40% `opening` and 60% `follow-up`. Agreement by position is exploratory.

## Q2 — Fitting objective

Fitting uses the **calibration split only**, on the **served artifact** (the GGUF via
`/v1/decisions`, with `decisionTimeoutSeconds` as a `fit.py` input, written into the candidate,
§Q4.3). `fit.py` runs both models over calibration:
- the decision model gives `p` (or `choice` + `confidence`);
- the hand-off LLM gives its parsed answer and wall-clock under §3.2.

### Q2.0 Fitting margins and outcomes

**`T_fit` and the 0.05 hand-off-share cushion are fitting targets, never bars.** The bars stay
FR-1's test (§Q3), FR-2's 2.0, and FR-10's 0.90 and 350 ms. One rule covers all three bars:
- **Latency.** Both latency bars become a hand-off-share ceiling, computed from calibration
  latencies with `t̄_{L|H}` taken over the band's own hand-off items:
  - FR-2: `s_2 = (t̄_L/2 − t̄_d) / t̄_{L|H}`;
  - FR-10: `s_350 = (350 − t̄_d) / t̄_{L|H}`.

  Each gets the same cushion: aim for `s ≤ s_x − 0.05`.
- **Quality.** The target is FR-1's no-worse point form (`D_c ≤ 0`, §Q2.1) or FR-10's
  `T_fit = 0.94` (§Q2.2). Each ladder below relaxes it to the requirement value (FR-1's calibration
  McNemar not rejecting; FR-10's 0.90) when it cannot be met, never past it.
- **Outcome.** The fit records the quality target used, and the rung reached as `fitOutcome`:
  - `within-margin`;
  - `below-margin`: requirement values met on calibration, a margin not;
  - `fit-infeasible: <bar>`: a requirement value is unmet on calibration. When several are, the
    first in the order `ranking`, `agreement`, `latency` is named; the fit report holds every
    bar's calibration value.
- **Every candidate, `fit-infeasible` included, is evaluated once** (FR-4). One evaluation is the
  two-load replicate pair (§Q6), so "does not meet its bar" is always a measured verdict.
  Replicate 2 may be skipped when replicate 1 already fails, since the conjunction is decided. The
  pair counts once toward K (§Q6).

### Q2.1 Speed-up point (guard)

- **Band.** `(l, h)` decides *yes* if `p ≥ h`, *no* if `p < l`, and hands off otherwise. A
  hand-off item takes the LLM's answer.
- **Cell excess.** Per FR-13 cell `c`, the composite's error count minus the incumbent's is
  `D_c(l,h) = Σ_{decided items in c} (model wrong − LLM wrong)`.
- **Objective.** The fit takes the first rung attainable on calibration:
  1. `D_c ≤ 0` in all four cells with `s ≤ s_2 − 0.05`, giving `within-margin`;
  2. every cell's calibration one-sided exact McNemar (§Q3.1) not rejecting, with
     `s ≤ s_2 − 0.05`, giving `below-margin`;
  3. rungs 1–2 with `s ≤ s_2`, giving `below-margin`;
  4. `D_c ≤ 0` at the smallest `s`, latency ignored, giving `fit-infeasible: latency`.

  Rungs 1 and 4 minimise the hand-off share `s(l,h)`, breaking ties by larger `min_c(−D_c)`, then
  by the narrower band. The McNemar rungs keep non-rejection as the constraint and sort by
  `(max_c D_c ↑, max_c m_c ↑, s ↑)`: least worst-cell excess, then least discordance, never a
  noisier band. With one cell differing and worst, `(b_w, c_w)` = (2,0) beats (3,1) and (19,16).
  - The search is exhaustive over `l ≤ h`, from the distinct calibration `p` values plus `{0, 1}`.
  - `s = 1` gives every `D_c = 0`, so rung 4 always exists.
  - An over-limit calibration item has no `p`, so it is a hand-off in every band (§Q4.3), here and
    in §Q2.2. It adds 0 to every `D_c`, it raises the floor of `s`, and it enters `t̄_{L|H}` with
    the failed call's time included.

### Q2.2 New-ability points

The fit takes the first rung that is attainable on calibration:
1. the highest `T ∈ {0.94, 0.93, 0.92, 0.91, 0.90}` with `s ≤ s_350 − 0.05`, giving
   `within-margin` at T = 0.94 and `below-margin` otherwise;
2. the same `T` ladder with `s ≤ s_350`, giving `below-margin`;
3. the highest attainable `T ≥ 0.90` at the smallest `s`, with latency ignored, giving
   `fit-infeasible: latency`;
4. otherwise, the setting that maximises the minimum per-class agreement, with ties going to the
   smaller `s`, giving `fit-infeasible: agreement`. This happens, for example, when the hand-off
   LLM itself agrees below 0.90 in some class.

At each rung, `T` means calibration agreement ≥ T within every true answer, and `s` is minimised.
For a predicate the search is §Q2.1's band search; for a choice it is a 1-D search over
`minConfidence`.

**Why 0.94:** §Q5.3's conjunction figures.

### Q2.3 Question instructions and the ranking diagnostic

- **At most 5 instruction texts per point,** scored by the objective on calibration. The cap
  limits selection on calibration. One text is a minimal control: bare label names for a choice,
  a one-line question for a predicate. A longer text forces more hand-offs (§Q1.3), a cost the
  fit sees on the served artifact.
- **Why the control.** arXiv 2609.26758 ("Type-Safe Is Not Error-Free…") studies Jev and two
  open-weight decision models. Its abstract reports two findings:
  - yes/no option names raise decision-flip rates by up to 70.4 pp over a 0/1 control;
  - reassigning option names against their definitions drops mean AUC from 93.8% to a
    below-chance 23.2%.

  The abstract names only Jev; whether Laya was studied is unverified. So rubric text may carry
  little weight, and a predicate (yes/no), the guard's `advance`, is the most exposed shape.
- **Diagnostic.** `fit.py`'s fit report (not the fit manifest) records each instruction text's
  calibration ROC-AUC of `p` (or `confidence`) against the label. An AUC ≤ 0.5 is a reversed or
  uninformative ranking that no threshold repairs. Over-limit items have no score and are
  excluded. If the chosen text's AUC is ≤ 0.5, the fit records
  `fit-infeasible: ranking`, and the candidate is still evaluated (§Q2.0).

### Q2.4 Minimum calibration size

- **At least the eval floor per cell, recommended equal to eval** (§Q1.3).
- The fitted constraint is a calibration point estimate. If calibration cells are smaller than eval
  cells, fit noise exceeds what eval resolves, and eval failures then come from the fit rather than
  the model.
- A design rule, not a simulated figure (no per-item Laya `p` distribution exists). U6 checks it
  by split-half: fit on one half of calibration, score the other, report the shift in `s` and
  each `D_c`.

### Q2.5 No calibration map for choice confidence

- A threshold policy needs a **ranking** of items by reliability, not calibrated probabilities.
  Laya's over-confidence is absorbed exactly by a fitted predicate threshold (§3.4), and
  approximately by a choice threshold.
- Temperature scaling can reorder choice confidences, but it is fitted to log loss, not this
  objective, so a reorder may as easily hurt. No map.
- Use `confidence` as returned. It is not `max(probabilities)` (probe document, one choice
  answer). Its definition is undocumented, so the fit report also gives `max(probabilities)`'s
  AUC (§Q2.3), as an exploratory figure.
- **Next lever,** if one `minConfidence` is infeasible: one threshold per predicted label. The
  likely failure is one weak class. That needs a new policy kind (`docs/plans/typed-decision-models2.md` §3.8).

## Q3 — FR-1 parity statistics

### Q3.1 Instrument

- **Cells and pairing.** Each of the four FR-13 cells is tested separately:
  `falseAdvanceRateBy{Understanding,Turns}` and `falseSuspendRateBy{Understanding,Turns}`. The
  candidate and incumbent runs are paired by `itemId` over items scored in both (§Q4.3).
- **Errors** are graded against the majority label (§Q1.2).
- **Counts.** `b_w` = items where the candidate errs and the incumbent does not. `c_w` = the
  reverse. `m = b_w + c_w`, and `n_c` = paired items in the cell.
- **Test.** One-sided exact McNemar for "candidate's error rate is higher":
  `p_worse = Σ_{i=b_w}^{m} C(m,i) / 2^m` (and `p_worse = 1` when `m = 0`).
- **α = 0.05,** one-sided, per cell, unadjusted. A cell is **worse** iff `p_worse ≤ 0.05`, and
  otherwise **holds**.
- **FR-1 holds iff all four cells hold,** in every replicate (§Q6). Pooled and side-check rates
  print with Wilson intervals as `exploratory — no significance claim`.
- `decision_stats` computes `p_worse` itself. It never calls `stats.verdict()` or
  `stats.holm_steps()` (§Q3.2).

### Q3.2 Why: nothing may make `holds` easier

FR-1 claims the **absence** of a detected worsening, so the harmful error is a false `holds`.
- **Holm** divides α, so fewer cells reject as the family grows. Its stop rule also leaves later
  members "not tested", which would read as `holds`. Excluded.
- **Rule 7's floor demotion** (on `verdict()`'s envelope path) turns a significant worsening into
  `holds`. Excluded. On the exact test the floor is implied anyway: rejection needs `b_w ≥ 5`.
- **One-sided, not two-sided.** Two-sided 0.05 spends half of α on "better", which FR-1 does not
  ask about. It would test the worse direction at 0.025.
- **Monotone.** Unadjusted cells and replicates combined as a conjunction mean that adding a
  cell or a replicate can only add chances to reject.
- **The price.** A candidate exactly as good as the incumbent fails FR-1 with probability
  **1.05% / 4.86% / 9.04%** in one run, for four cells of 50 at discordance ψ = 0.05 / 0.10 /
  0.20. Two load replicates (§Q6) raise it to at most **2.08% / 9.49% / 17.27%**, the
  independent-replicate bound `1 − (1 − p)²`; replicates share items, so the real rate is lower.
  That cost falls on the candidate, where FR-1 puts it.
- **Clustering cannot loosen the decision either.** Under clustering McNemar over-rejects, the
  safe side here. Only the detection limit takes `designEffect` (§Q3.3).

### Q3.3 The detectable difference on every `holds` line

**MDD₈₀** is the smallest true worsening `δ` this test detects with probability 0.80, at the
cell's `n = floor(n_c / designEffect)` and its observed discordance `ψ̂ = m / n_c`.
- **Power, exact.** Candidate-worse and candidate-better discordance probabilities are
  `(ψ ± δ)/2`, with `ψ = max(ψ̂, δ)`, since a worsening of δ needs at least δ discordance. Then:
  - `power(δ) = Σ_{m=0}^{n} Bin(m; n, ψ) · P(Bin(m, q) ≥ crit(m))`;
  - `q = (ψ+δ)/(2ψ)`;
  - `crit(m)` is the smallest `b` with `Σ_{i≥b} C(m,i)/2^m ≤ 0.05` (none for `m < 5`).

  Terms are computed in log space (`math.lgamma`).
- **Search.** Step δ up a 0.1 pp grid and print the first point with power ≥ 0.80. That is a
  ceiling, the direction that keeps "detects ≥ N" true (SMB `-ml` §3.4 Rule 3).
- **Never below best case.** `stats.min_detectable_difference` always assumes `π_c = 0`. The
  plug-in `ψ̂` equals that best case whenever `ψ̂` is below the MDD, as in the n = 50 row at
  ψ̂ ≤ 0.10. It is larger, and more honest, when the observed discordance exceeds the MDD.
- **Unattainable.** Below `n = 5` no worsening can reach significance, so `mdd` returns `None`, as
  with `stats.UnattainablePower`. The line then prints
  `holds (no worsening is detectable at n = …: fewer than 5 items)`.

MDD₈₀ in pp, one-sided α = 0.05 (`parity.py`):

| n per cell | ψ̂ ≤ 0.05 | ψ̂ = 0.10 | ψ̂ = 0.20 | ψ̂ = 0.30 |
|---|---|---|---|---|
| 50 | 13.1 | 13.1 | 16.4 | 19.9 |
| 75 | 8.8 | 9.4 | 13.3 | 16.3 |
| 100 | 6.7 | 8.2 | 11.6 | 14.1 |
| 150 | 4.5–4.8 | 6.7 | 9.5 | 11.6 |

### Q3.4 The printed line

`<metric>: candidate e₁/n [Wilson] vs incumbent e₂/n [Wilson]; diff +x.x pp; b=…, c=…, p_worse=…; worse by at most U pp (one-sided 95%) — holds (could only detect ≥ N pp at n = …, ψ̂ = …)`

Otherwise the line ends `— not met: candidate worse`.
- **`U`** is the upper end of `stats.mover_d_interval(a, b_w, c_w, d, z=1.6448536269514722)`,
  oriented candidate minus incumbent (`a`/`d` = both err / neither errs). It is AC-1's interval,
  and it reads as "the candidate is at most U pp worse".
- `U` is descriptive. `p_worse` decides.

### Q3.5 Reading the reuse count K (FR-4, R-5)

A `holds` from the (K+1)-th candidate on one eval set is a survivor of K+1 tries.
- **Inflated miss rate.** If all K+1 tries were worse by the printed N, at least one passes with
  probability at most `1 − 0.8^{K+1}`: **0.20 / 0.36 / 0.49 / 0.59** for K = 0…3.
  - This bound holds for **non-adaptive** tries, where positive dependence only lowers it.
  - Re-tunes informed by eval results are adaptive (FR-4 allows them). Under them the printed
    figure bounds the inflation in neither direction, so it stays advice.
- **Reuse-adjusted limit, printed when K ≥ 1.** This is §Q3.3's search with power target
  `0.8^{1/(K+1)}`: the worsening still ruled out at 80% across all tries. At ψ̂ = 0.10, K = 0…3:
  - 50 per cell: 13.1 → 15.2 → 16.4 → 17.2 pp;
  - 100 per cell: 8.2 → 9.1 → 9.4 → 9.6 pp.
- **Reader's rule.** Quote the adjusted figure. Past K = 3, a fresh dataset version costs less than
  the lost resolution. That is advice only; FR-4 does not require it.

### Q3.6 Worked example (guard, 50 per cell, K = 0)

| Cell | b_w | c_w | diff | p_worse | Reading |
|---|---|---|---|---|---|
| falseAdvance · understanding | 4 | 1 | +6.0 pp | 0.1875 | holds (could only detect ≥ 13.1 pp, ψ̂ = 0.10) |
| falseAdvance · turns | 6 | 2 | +8.0 pp | 0.1445 | holds (could only detect ≥ 14.5 pp, ψ̂ = 0.16) |
| falseSuspend · understanding | 5 | 0 | +10.0 pp | 0.0312 | **not met** |
| falseSuspend · turns | 7 | 1 | +12.0 pp | 0.0352 | **not met** |

FR-1 is not met. Row 1 is R-6 made visible: 6 pp worse still prints `holds`, but beside a 13.1 pp
limit. At 100 per cell, the same discordance (8 vs 2) gives `p_worse = 0.055`. That still holds,
but now it rules out ≥ 8.2 pp. This is why §Q1.3 sets 100 per cell.

## Q4 — Mean-latency estimator

### Q4.1 FR-2: paired ratio of means

- **Inputs.**
  - The incumbent arm and the candidate arm, with the same top-level `run.sessionId`.
  - `x_i` and `y_i` = each arm's `ItemTiming.wallClockMs`. For the candidate, that covers the
    decision call plus any hand-off call.
- **Estimator.** `R = Σx_i / Σy_i` over the paired set `P` (§Q4.3). FR-2 asks about time "on
  average across the whole mix", which is a ratio of means, not a mean of per-item ratios.
- **Interval.** A paired percentile bootstrap:
  - sort `P` by `itemId`, so the result depends only on the data and the seed;
  - draw B = 10,000 index resamples with `random.Random(seed)`, where `seed` is the pack's
    declared seed;
  - take `stats.percentile` at `Fraction(1,40)` and `Fraction(39,40)`.

  Unadjusted: FR-2 is one bar. Index drawing is not a second percentile
  estimator.
- **Bar.** `bar_holds(R, 2.0, direction="at-least")` on the measured `R` (FR-11a), with the
  interval printed beside it.

### Q4.2 FR-10: single-arm mean

The estimator is `ȳ` over the candidate's paired-set items. The interval is
`stats.mean_bootstrap_interval(values sorted by itemId, B=10_000, seed=<pack seed>,
levels=(LEVEL_CI95_LO, LEVEL_CI95_HI), support=None)`. The bar is
`bar_holds(ȳ, 350, direction="at-most")`.

### Q4.3 Withheld, failed and timed-out items

The rows follow what the runtime does with each item:
- **A guard chat-call failure** (timeout or HTTP error) **fails the run**: the guard step raises
  and `_drive`'s `except Exception` calls `_fail_with_note` (`executor.py:691-693`); only a
  *malformed reply* maps to suspend. With no runtime decision to imitate, the bench keeps today's
  withholding: `withheldFor="timeout"`/`"no_response"`, `wallClockMs=None` (`runner.py`).
- **A new-ability hand-off chat failure fails closed.** Join-in's hand-off runs inside
  `background._safe_respond`, whose `except Exception` swallows it (`background.py:358-369`): no
  reply, so `no`. `_safe_classify` mirrors it (§3.7): no label.
- **A decision-call failure** also has a runtime decision (§3.7): the guard's `llm-fallback`; the
  new abilities fail closed.
- **An over-limit input hands off to the LLM at every point** (stakeholder decision,
  2026-10-31). LM Studio does not truncate: a question whose tokenized input, question text
  included, exceeds 512 tokens gets an HTTP 500 (boundary and body text: the probe document).
  The outcome depends on the input alone, so these items are a known class, not a random fault.
  The runtime treats one as an unsure item: `decidedBy="llm-handoff"`, reason `over-limit`. The
  client recognises the stable substring `too large to process`; any other 500 stays a failure.
  - **No hand-off model.** On a candidate with no `handoffModel`, no hand-off is possible, so the
    item takes the decision-call-failure row's outcomes. It still counts in `overLimit`.
  - **Rejected:** failing closed (systematically blind to long messages); caller-side truncation
    (no Laya tokenizer, and a character cap guarantees nothing); chunking (unvalidated).
- **Timeout sources.** The decision timeout is the candidate's `decisionTimeoutSeconds`, which
  §3.7 copies into `timeoutSeconds`. The chat timeout, both arms and the hand-off, must equal the
  runtime's for that kind: **180 s** (`falkor-chat/config/models.json` `timeouts.guard` and
  `timeouts.agent`; no retry wrapper exists on the guard, responder or decision paths). Equal arms are not enough: a
  shorter shared timeout still shortens kept hand-off failures (FR-10's 350 ms is absolute) and
  drops more guard pairs, moving R. A run at another value gives no bar verdict.

| Item state | FR-2 latency | FR-10 latency | FR-1 quality | FR-10 quality |
|---|---|---|---|---|
| `withheldFor="load"` in either arm | dropped from both arms | dropped | dropped pairwise | dropped |
| Guard chat call fails or times out, in the incumbent run or a candidate hand-off | dropped from both arms (today's withholding, no elapsed time) | — | dropped pairwise | — |
| New-ability hand-off chat call fails or times out | — | `y_i` = elapsed (decision + hand-off to its failure) | — | scored fail-closed, as below |
| Decision call over the 512-token limit (§Q4.3 above) | `y_i` = failed call + hand-off call | `y_i` = failed call + hand-off call | the hand-off LLM's answer, as for any hand-off | the hand-off LLM's answer |
| Candidate decision call fails or times out (over-limit excluded) | `y_i` = elapsed (decision timeout) + incumbent's `x_i` | `y_i` = elapsed | kept as a **concordant pair**: the candidate's answer is the incumbent's | scored as the fail-closed answer: join-in `no`; message type "no label", a disagreement |
| Reply parses as malformed (any chat call) | as measured | as measured | §3.2's parse fallback (guard: suspend) | §3.2's parse fallback |

- **Counts.** Every dropped row is counted per arm and per reason, and printed beside each bar.
  Kept failures and forced hand-offs print per replicate beside `handoffShare`:
  `decisionCallFailures k/n` (over-limit items excluded), `overLimit k/n` (a floor on `s`) and,
  for new abilities, `handoffCallFailures k/n`.
- **Why a concordant pair.** It leaves `p_worse` unchanged and keeps `n_c` and ψ̂ at the
  runtime's values.
- **Elapsed time needs capture** on the failed-call rows and on over-limit items; `ItemTiming` does not record it today
  (`docs/plans/typed-decision-models2.md` §3.3).
- **Not assessable.** If the paired scored n of eval items falls below FR-11's 200, through load
  or chat-call withholding, the bar line prints `not assessable (scored n < 200)` instead of a
  verdict (a new state, `docs/plans/typed-decision-models2.md` §3.6).

### Q4.4 Break-even hand-off share (recomputing R-2's "~45%")

`R = t̄_L / (t̄_d + s·t̄_{L|H})`, so FR-2 holds iff `s ≤ (t̄_L/2 − t̄_d) / t̄_{L|H}`. Inputs:
the two stored incumbent runs (`qwen/qwen3-4b-2507`, 85 items each), and `t̄_d = 20 ms` (the
requirements' warm figure, until U7 measures it).

| Run | t̄_L | median | s_max, hand-offs typical | s_max, hand-offs as slow as the slowest tier |
|---|---|---|---|---|
| 2026-09-11 | 832 ms | 812 ms | 0.476 | 0.405 (`clear_advance` mean 977 ms) |
| 2026-09-19 | 682 ms | 653 ms | 0.471 | 0.393 (816 ms) |

- **R-2's ~45% → 0.39–0.48,** depending on whether hand-offs (uncertain items, which may draw
  longer replies) are slower than average.
- **FR-10's 350 ms** allows `s ≤ 330 / t̄_{L|H}`: 0.40–0.48, or 0.34–0.40 for slow-tier hand-offs,
  with a `qwen3-4b` hand-off model.
- **Same-session only.** The stored mean moved 18% across sessions (832 → 682 ms), hence FR-2's
  within-session comparison.

**Worked example** (constructed, not measured): 400 latencies resampled from the 85 stored ones;
each candidate item costs 20 ms, plus the LLM's latency in a random hand-off subset; B = 10,000.
- s = 0.30: `R = 3.02 [2.65, 3.51]`.
- s = 0.45: `R = 2.10 [1.90, 2.35]`. That is met on the measured value, and the interval shows
  the margin.

## Q5 — FR-10 statistics

### Q5.1 Per-answer agreement

- **Metric.** For each true answer `j`, `a_j = k_j/n_j` over scored eval items labeled `j` (§Q4.3).
- **Bar.** `bar_holds(k_j/n_j, 0.90, direction="at-least")`, evaluated as `10·k_j ≥ 9·n_j`. FR-10
  holds iff every class meets it, in every replicate, and §Q4.2 also holds.
- **Interval.** Wilson 95% per class, printed beside it.

### Q5.2 The five-label family: no correction

- The bar is a conjunction of measured-value thresholds (FR-11a), not a test, so there is no α to
  divide. Adding classes only makes it harder, the same monotone property as §Q3.2.
- The intervals print as `per-class 95% Wilson, not simultaneous`. A joint reading would need
  Bonferroni 99%, which is not printed: two interval sets invite choosing between them.

### Q5.3 What the bar does at these sizes

P(a class with true agreement `a` measures ≥ 0.90), exact binomial (`fr10.py`):

| n per class | a = 0.80 | 0.85 | 0.88 | 0.90 | 0.92 | 0.95 |
|---|---|---|---|---|---|---|
| 40 | 0.076 | 0.263 | 0.467 | 0.629 | 0.787 | 0.952 |
| 60 | 0.031 | 0.185 | 0.409 | 0.606 | 0.798 | 0.970 |
| 100 | 0.006 | 0.099 | 0.334 | 0.583 | 0.824 | 0.989 |

- **False pass comes from one weak class among strong ones.** The conjunction then reduces to that
  class's row. This is §Q1.3's basis for 60 per class.
- **The conjunction is harsh at the bar.** With all five classes truly at 0.90, it passes 0.082 of
  the time at 60 per class; at 0.92, 0.324; at 0.94, 0.706 (0.926 for two classes at 100). Hence
  the fitting target `T_fit = 0.94` (§Q2.0). This follows from the stakeholder's measured-value
  rule (SQ-2) and is not a defect.
- **Edge half-widths.** The Wilson half-width at `k = ⌈0.9n⌉` is 0.077 at n = 60 and 0.060 at
  n = 100.

### Q5.4 Worked example (message type, 60 per label)

| Label | k/60 | Wilson 95% | Bar |
|---|---|---|---|
| order | 58 (0.967) | [0.886, 0.991] | met |
| question | 56 (0.933) | [0.841, 0.974] | met |
| complaint | 54 (0.900) | [0.799, 0.953] | met (`10·54 = 9·60`) |
| escalation | 53 (0.883) | [0.778, 0.942] | **not met** |
| other | 55 (0.917) | [0.819, 0.964] | met |

FR-10 is not met, on `escalation`. AC-15's always-majority arm scores 0/60 on four classes.

## Q6 — Determinism and replicate runs

**Method.** `designEffect = 1.0, basis = "by-construction"` stands for candidate runs. A bar claim
needs **two load replicates** of the candidate arm, and a bar is met only if it is met in both
(stakeholder-confirmed 2026-10-30).

**The cost.** One extra candidate run per point, about 5 min at 400 items. It also raises the false
fail of a candidate exactly as good as the incumbent (§Q3.2 gives the figures).

- **The basis stands:** it concerns unit independence. Within a run:
  - each eval item is its own cluster (§Q1.3);
  - every item runs on one load (the load rule below);
  - Laya is deterministic within a load.

  So inference is exact for "this candidate under this load": the load is one shared draw the run
  is conditional on, not clustering, and no within-run design effect represents it.
- **Replicates are needed.** The deployed model runs under another load. The requirements record
  14/30 vs 12/30 across two loads at a 0.5 cut-off, so at least 6.7% of those items flipped. A band
  exposes only the items near `l` or `h`, so the effect should be smaller, but it is unmeasured.
- **Protocol (U7, one session).**
  1. Run the incumbent arm.
  2. Run candidate replicate 1.
  3. Reload the **decision model only**, then run replicate 2. The hand-off LLM's load, shared with
     the incumbent by FR-3's pairing, is not under test.
- **Rule.** FR-1 (each cell), FR-2 and FR-10 are evaluated per replicate, each against the one
  incumbent run. A split result prints `unstable across loads — not met`. The conjunction only
  makes bars harder, as §Q3.2 requires.
- **Diagnostic.** Print the replicate disagreement in final answer and in `decidedBy`, each as
  `k/n` with Wilson.
- **Load rule.** A candidate run is one load replicate only if nothing shows a reload. Under JIT
  loading a `withheldFor="load"` item means the next call reloaded, so the run mixes two loads and
  is exact for neither. At run end, probe residency, then:
  - model absent, or any candidate item withheld for `"load"`: no re-check call (it would JIT-load
    and compare across loads); basis `"assumed"`;
  - otherwise re-issue the warm-up's fixed `decide` call; a different `p`, or a raise, gives
    `"assumed"`.

  An `"assumed"` run's bars print `not assessable (load changed mid-run)` and the replicate is
  rerun; same hash, so K is unchanged. The fixed-input check misses a reload that leaves that one
  `p` unchanged. Under default JIT, LM Studio held one JIT model, and a hand-off evicted the
  decision model (probe document). So both models must be explicitly loaded, in the bench and in
  a falkor-chat deployment. Otherwise every hand-off is followed by a cold reload, and the measured
  latency does not transfer.
- **Interactions.** Fitting (§Q2) happens under a third load, and the replicate rule covers it.
  Replicate 2 shares replicate 1's `candidateContentHash`, so it does not raise K (§3.5).

## Q7 — Fine-tuning method

Q7 is used only by U8, after a point misses its bar despite §Q2's refit.

**Source.** The model facts come from a Hugging Face card, `huggingface.co/aaroncool9/laya`, read
2026-10-30. Its provenance relative to LM Studio's `ggml-org` GGUF is unverified, and U8's first
step confirms that the weights match. The card states:
- a ModernBERT-large backbone, plus a decision head trained from scratch, 421M parameters in all;
- a 512-token per-question context (the U1a probe confirms the limit as a hard error, §Q4.3);
- "ships over-confident";
- training with a strictly proper scoring rule.

**Method: supervised fine-tuning of backbone and head with a proper scoring rule, then §Q2's
refit on the served GGUF.**
1. **Data.** Use the train split only. The early-stopping dev slice (about 15%) is cut from train
   by cluster, never from calibration. Question text and choice names are fixed to the candidate's
   (§Q2.3).
2. **Loss.** Log loss on the soft label `labelVotes / 3` (§Q1.2: a vote vector over the answer space). It is proper, so it does not teach
   over-confidence, and a 2–1 item counts as uncertain. Classes are weighted to balance within
   each FR-13 cell (guard) or each answer, because the bars are class-conditional.
3. **Training.** Learning rate about 1e-5 to 3e-5, at most 3 epochs, early-stopped on dev log
   loss: a conventional start, not a measured optimum.
4. **Learning curve.** Train at 25 / 50 / 100% of the split. If dev loss is still falling, generate
   more data. §Q1.3's train sizes are estimates that this replaces.
5. **Export acceptance (go/no-go).** On 200 calibration items, compare training-framework `p` with
   the served GGUF's `p`. Pass needs both:
   - median |Δp| ≤ 0.02;
   - ≥ 99% agreement of decided answers at a 0.5 cut-off.

   A failure means the export changes the model, and U8 reports "no".
6. **Refit.** Refit thresholds (§Q2) on the **served** model, because Q8_0 moves `p`. The fit
   manifest's `purposes` gains `training`, and the leakage check covers the train hashes.
7. **Context.** Nothing truncates silently (§Q4.3). The 500 names LM Studio's physical batch, not
   the model. Raising it would serve inputs beyond the card's 512-token context, which is
   untested, and it would be a different served artifact, so U8 does not raise it without
   measuring first.

## Handed to architect

Items 1–6 are the v1.5/v1.6 additions to `docs/plans/typed-decision-models2.md`. Earlier items
are absorbed there, and this list is removed once the plan absorbs these.
1. **§3.3 / §3.7: over-limit detection (§Q4.3).** `LMStudioCallFailed` gains a structured `body`
   field; falkor-chat's `ProviderCallError.detail` already carries it. Both clients match
   `too large to process`, with the verbatim probe body pinned as a fixture beside
   `lmStudioAppVersion`. The bench scores the item per §Q4.3's over-limit row, and
   `reason="over-limit"` goes into `ItemResult.detail` and `record_decision`.
2. **§3.6: counts.** `overLimit k/n` per replicate and per cell. `decisionCallFailures` excludes
   over-limit items.
3. **§3.2 / §3.3: chat timeouts** at the runtime's 180 s in both arms (§Q4.3).
4. **§3.7 / §6: explicit loading** of both models, in the bench and in deployment (§Q6). R-13
   ("silent context truncation") is false, and becomes "forced hand-off, a floor on `s`".
5. **§3.5: fit.** Over-limit calibration items are handled per §Q2.1/§Q2.3. The fit report adds
   `max(probabilities)`'s AUC (§Q2.5).
6. **§5: an offline test (U1e or U2)** feeds the pinned 500 body through `decide` and the runner,
   since eval may hold no over-limit item.

## Reproducing the figures

Every figure in this note is printed by one of three stdlib scripts in
`docs/plans/typed-decision-models-ml/`, run from the repo root with `python3 -I`:
- `parity.py` covers §Q3.
- `latency.py model-bench/results/runs` covers §Q4.4, including the slow-tier columns.
- `fr10.py` covers §Q1.3's sawtooth and labelling volume, and §Q5.

`decision_stats.py`'s tests re-derive the figures from the formulas, not from these scripts.
