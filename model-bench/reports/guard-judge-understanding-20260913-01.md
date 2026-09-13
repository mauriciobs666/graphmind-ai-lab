# Comparison — guard-judge-understanding@1.0.0 (guard-judge)

## Arms

| arm | metric | k/n | rate | 95% Wilson |
|---|---|---|---|---|
| qwen2.5-3b-instruct | falseAdvanceRate | 2/40 | 0.050 | [0.014, 0.165] |
| qwen2.5-3b-instruct | falseSuspendRate | 28/30 | 0.933 | [0.787, 0.982] |
| qwen2.5-3b-instruct | advanceRecall | 2/30 | 0.067 | [0.018, 0.213] |
| qwen2.5-3b-instruct | falseAdvanceRateBoundary | 0/15 | 0.000 | [0.000, 0.204] |
| qwen2.5-3b-instruct | falseAdvanceRateByUnderstanding | 0/24 | 0.000 | [0.000, 0.138] |
| qwen2.5-3b-instruct | falseAdvanceRateByTurns | 2/16 | 0.125 | [0.035, 0.360] |
| qwen2.5-3b-instruct | falseSuspendRateByUnderstanding | 18/18 | 1.000 | [0.824, 1.000] |
| qwen2.5-3b-instruct | falseSuspendRateByTurns | 10/12 | 0.833 | [0.552, 0.953] |
| qwen/qwen3-4b-2507 | falseAdvanceRate | 4/40 | 0.100 | [0.040, 0.231] |
| qwen/qwen3-4b-2507 | falseSuspendRate | 4/30 | 0.133 | [0.053, 0.297] |
| qwen/qwen3-4b-2507 | advanceRecall | 26/30 | 0.867 | [0.703, 0.947] |
| qwen/qwen3-4b-2507 | falseAdvanceRateBoundary | 5/15 | 0.333 | [0.152, 0.583] |
| qwen/qwen3-4b-2507 | falseAdvanceRateByUnderstanding | 0/24 | 0.000 | [0.000, 0.138] |
| qwen/qwen3-4b-2507 | falseAdvanceRateByTurns | 4/16 | 0.250 | [0.102, 0.495] |
| qwen/qwen3-4b-2507 | falseSuspendRateByUnderstanding | 3/18 | 0.167 | [0.058, 0.392] |
| qwen/qwen3-4b-2507 | falseSuspendRateByTurns | 1/12 | 0.083 | [0.015, 0.354] |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

## Verdicts

Comparison kind: **paired, cross-session** (§3.7).

### falseAdvanceRate

Not distinguishable at this sample size. Observed difference -5.0 pp, 95% CI [-15.9, 3.6] pp covers zero (b=0, c=2, McNemar exact p=0.500). This pack resolves differences of >=21.9 pp with 80% power at n=40 effective items (40 units, design effect 1.00, by-construction, alpha=0.025); the observed 5.0 pp is below that. Neither model is ranked above the other.

- paired n: 40 of 85 items (`asymmetry`: 0 scoreable for qwen2.5-3b-instruct only, 0 scoreable for qwen/qwen3-4b-2507 only; 45 unscoreable in both; 0 present in qwen2.5-3b-instruct only, 0 in qwen/qwen3-4b-2507 only) — §4.3
- marginal Wilson intervals overlap: yes
- decided by: mcnemar-exact

This pack resolves differences of >=21.9 pp with 80% power at n=40 effective items (40 units, design effect 1.00, by-construction, alpha=0.025). Differences below 15.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every item the models differ on. Inference is conditional on the 40 items in guard-judge-understanding@1.0.0; generalization to unwritten items is not certified by any interval in this report.

### falseSuspendRate

qwen2.5-3b-instruct is better than qwen/qwen3-4b-2507 on falseSuspendRate: +80.0 pp (95% CI [59.2, 88.9] pp), n=30 paired items (unit: item, design effect 1.00), McNemar exact p=0.000 (b=24, c=0).

- paired n: 30 of 85 items (`asymmetry`: 0 scoreable for qwen2.5-3b-instruct only, 0 scoreable for qwen/qwen3-4b-2507 only; 55 unscoreable in both; 0 present in qwen2.5-3b-instruct only, 0 in qwen/qwen3-4b-2507 only) — §4.3
- marginal Wilson intervals overlap: no
- decided by: mcnemar-exact

This pack resolves differences of >=28.7 pp with 80% power at n=30 effective items (30 units, design effect 1.00, by-construction, alpha=0.025). Differences below 20.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every item the models differ on. Inference is conditional on the 30 items in guard-judge-understanding@1.0.0; generalization to unwritten items is not certified by any interval in this report.

### Family-wise error control

Holm–Bonferroni across the 2 pre-registered verdict metrics, applied: the smallest p is tested at alpha/2, the next at alpha/1, and the first non-rejection stops the procedure. Every **MDD** above is computed at the family-adjusted alpha=0.025; every **observable floor** is computed at the unadjusted alpha=0.05, the loosest step a member can face, because that is the only alpha at which the floor's own sentence is true (§7.1).

| metric | McNemar p | Holm-adjusted threshold | decision |
|---|---|---|---|
| falseAdvanceRate | 0.500 | 0.0500 | not distinguishable |
| falseSuspendRate | 0.000 | 0.0250 | distinguishable |

_This pack declares no headline metric: its verdict metrics are co-equal and are printed side by side, in the manifest's declared order, with no summary line above them and no arithmetic combining them (§3.3)._

### Exploratory metrics

- `advanceRecall` — exploratory — no significance claim
- `falseAdvanceRateBoundary` — exploratory — no significance claim
- `falseAdvanceRateByUnderstanding` — exploratory — no significance claim
- `falseAdvanceRateByTurns` — exploratory — no significance claim
- `falseSuspendRateByUnderstanding` — exploratory — no significance claim
- `falseSuspendRateByTurns` — exploratory — no significance claim

_The marginal-overlap line is a **diagnostic**, never the verdict: at this lab's sample sizes two marginal intervals overlapping is a far stronger condition than their difference covering zero, and the literal rule cannot fire at all at n<=40 with a baseline >=0.90 (`-ml` §3.1)._

