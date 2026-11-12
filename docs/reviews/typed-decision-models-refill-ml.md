# Methodology review: eval refill from cell surplus and the floor gate

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — · **Reviews:** `docs/plans/typed-decision-models4.md` v1.4 §3.6 "Cells, then splits", U5.10c

**Verdict: approve with suggestions.** The refill is valid; the floor gate is right. For message-type `escalation` the refill alone will almost certainly not reach the floor (52), so the gate will refuse and a drop-aware top-up is the better path than accepting the printed n.

Method basis: note `docs/plans/typed-decision-models-ml.md` §Q1.3 (shingle check, cells, floors) and §Q5.3. Evidence read from `decision-models/data/message-type-classifier/v1/` (`splits.json`, `items.jsonl`, `diagnostics.json`, `generation.json`); no runs, nothing edited.

## 1. Is the refill biased? No new selection

- Subsampling to the cell target is random within (label x design stratum), so the surplus is a random remainder, exchangeable with the kept items. The walk order is a random shuffle, so a stopping rule ("until full") does not select.
- Pre-refill eval is "random eval-assigned items that survive the overlap test against calibration/train". Post-refill eval is "random items from the same cell that survive the same test", against the same fixed calibration/train set. Same conditioning, same population. The test uses text only, never a label or a model score, so there is no outcome-dependent selection (verified by reading §3.6; the candidate has no scores yet).
- Calibration and train are not read or changed, so the held-out purity argument (no eval item shares a span with a fitting item) holds exactly as before. Mutants listed in U5.10c (refill from calibration, skip the test on a refill, draw before the drop is known, use side-check-dropped surplus) are the right ones.

## 2. Comparability and independence: findings

**F1 (minor, observed) — The overlap filter is asymmetric and heavy for `escalation`.** From v1: the 42 `shingleSplit` drops split by class as escalation 20 of 60 eval-assigned (33%), complaint 8, other 7, order 6, question 1 (derived from kept counts 40/52/53/54/59 against 60). Within the kept v1 sets I recomputed 8-token overlap (note's tokenization): calibration items sharing a span with another calibration item, escalation 15 of 60 (25%), complaint 7, question 6; eval-vs-eval escalation 2 of 40 (5%); eval-vs-calibration 0 (as designed). So `escalation` text is templated, calibration holds near-duplicate clusters (effective calibration n below the printed 60), and eval is by construction the more diverse side. The refill keeps this property and adds no new bias, but eval and calibration are not drawn from identical populations. If agreement correlates with prototypical phrasing, eval reads conservatively (false-fail more than false-pass); unmeasured. Suggest: print per cell the drop rate and the post-refill eval-vs-calibration agreement as exploratory figures, and flag a cell whose drop rate is above a stated level (say 25%) as a generation-diversity signal rather than only a sizing problem.

**F2 (minor) — Eval-eval overlap is not tested, and is not a leakage.** Refill candidates are tested against calibration/train only, same as today. A refill can add items near-duplicate to an eval item (v1 shows 2 of 40). That lowers effective n slightly, not validity. The per-cell design effect (§Q6) is computed on the final eval, so it needs no change.

**F3 (nit) — `dropped.shingleSplit` changes meaning.** Refill candidates that overlap are counted there too, although they were never eval items. Keys do not change, but the number is no longer "eval-assigned items dropped". Say so in the printed per-cell line (assigned drops vs candidate drops), or the v1/v2 counts will not reconcile.

## 3. Interaction with design-effect, top-up and cell-proportion claims

- **Cell proportions:** the refill is within the cell, so label x design-stratum counts are preserved. Intent mix inside a cell can shift slightly, because the filter removes templated items; the top-up claim ("no intent conditioning") concerns the sampler, not this, and is unaffected. Round 1 vs top-up composition of the refill follows the random surplus; the existing round-1-vs-top-up diagnostics still apply.
- **Design effect:** unaffected as above (computed post-hoc on the final eval).
- **Top-up claim "surplus unused":** the refill makes surplus a reserve with a purpose. The planner should size a top-up for the expected drop (F5), not for the target.

## 4. Eval >= floor with calibration unchanged: defensible

Equal eval and calibration sizes are a design nicety, not a requirement; §Q2.4 asks calibration only for the floor. Refilling only eval to its size keeps calibration at its own floor. Defensible.

**F4 (minor) — fresh seed for `v2`.** A re-finalize with a new RNG seed redraws cells and splits wholesale. That is fine because `v1` is unconsumed, provided the seed is declared before any model score is seen and recorded in `splits.json`. A seed retried until the floors pass is selection on sample size only, harmless, but state it so it never becomes selection on results.

## 5. Open case: `escalation` eval 40 vs floor 52

**Expected refill (derived, inferred inputs).** Labelled escalation items 132 (`diagnostics.json`), target 60 + 60, so surplus is 12 (inferred; the surplus is not stored). Eval needs 20 refills. With the v1 pass rate 40/60 = 0.667, refill adds Binomial(12, 0.667): eval 44 or more with probability 0.996, 46+ 0.93, 48+ 0.63, 49+ 0.39, 50+ 0.18, 52 (all 12 pass) 0.008. At pass rates 0.55 or 0.75 the 48+ figure is 0.30 or 0.84. So expect about 48, and the floor gate will refuse in about 99% of re-finalizations.

**What an eval cell under 52 costs** (exact binomial, bar `10*k >= 9*n`, my recomputation, matches note §Q5.3 at 40, 50, 60):

| n | P(pass \| a=0.80) | 0.85 | 0.88 | 0.94 |
|---|---|---|---|---|
| 40 | 0.076 | 0.263 | 0.467 | 0.910 |
| 44 | 0.044 | 0.190 | 0.379 | 0.878 |
| 48 | 0.025 | 0.134 | 0.302 | 0.841 |
| 50 | 0.048 | 0.219 | 0.435 | 0.922 |
| 52 | 0.037 | 0.188 | 0.396 | 0.910 |
| 60 | 0.031 | 0.185 | 0.409 | 0.933 |

- The note's criterion (false pass at a=0.85 at most 0.20) is a sawtooth, not a monotone floor: n = 41-43 and 50-51 exceed 0.20 (0.243 to 0.203), n = 44-49 meet it (0.19 down to 0.123). 52 is the smallest n after which every n meets it; it is conservative, not a cliff. Between 44 and 51 the false-pass risk is within about 0.03 of the design's.
- The real cost of a smaller n is sensitivity and width: a class truly at 0.94 passes with 0.841 at 48 against 0.933 at 60, and with the other four classes at n = 60 the five-class pass probability at 0.94 falls from 0.706 to 0.637. Wilson half-width at the bar edge is 0.081 at 48 against 0.077 at 60 (0.082 at 52). So 48 to 51 costs roughly 7 to 9 points of pass probability on a good model, not extra false passes. At 40 it is 0.263 false pass and the clearest violation.
- An eval below floor also falls behind equal calibration (60), and escalation's effective n is lower still (F1).

**Opinion.** Keep the floor gate refusing under 52; do not accept the printed n for `escalation` by default. Accepting at about 48 is statistically tolerable (above), but it is a decision for the existing `--short`/stakeholder path with n printed, not a silent default. The principled fix is cheap: require the top-up planner to size shortfalls for the expected drop, `candidates = shortfall / (1 - d)` with d the cell's observed eval-assigned drop rate (here 0.33), plus the refill's own drop. For escalation, the need after an expected refill is about 4 more eval items, so about 6 to 7 more labelled escalation items; with escalation about 0.2 of labelled items (132 of 660) that is roughly one to two 25-seed runs, well under the spend halt (half of round 1). Join-in's 186/200 is the same case; whether it has surplus is unverified (not stored), and with none the refill is a no-op and the gate refuses until a top-up.

## 6. Suggestions (ordered)

1. Make the planner drop-aware as in §5 (one sentence in §3.6 plus a U5.9 row: a fixture cell with d = 0.33 sizes the draw as shortfall / 0.67).
2. Per-cell print: assigned, dropped-assigned, refill candidates, refill drops, refilled n, floor (F1, F3).
3. Pre-declare the `v2` seed (F4).
4. Optionally restate the note's floor as the set of n meeting the 0.20 criterion plus the monotone bound 52; no change in behavior.

### Proposed REVIEW.md lines

None.
