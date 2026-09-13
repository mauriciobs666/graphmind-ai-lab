# Comparison — nlq-structured-query@1.2.0 (nlq-generator)

## Arms

| arm | metric | k/n | rate | 95% Wilson |
|---|---|---|---|---|
| qwen2.5-3b-instruct | layer1ExactMatchRate | 30/30 | 1.000 | [0.886, 1.000] |
| qwen2.5-3b-instruct | exactMatchByAggregation | 8/8 | 1.000 | [0.676, 1.000] |
| qwen2.5-3b-instruct | exactMatchByCompoundFilter | 2/2 | 1.000 | [0.342, 1.000] |
| qwen2.5-3b-instruct | exactMatchByConflictingFacts | 0/1 | 0.000 | [0.000, 0.793] |
| qwen2.5-3b-instruct | exactMatchByFilterList | 7/7 | 1.000 | [0.646, 1.000] |
| qwen2.5-3b-instruct | exactMatchByNotFound | 5/5 | 1.000 | [0.566, 1.000] |
| qwen2.5-3b-instruct | exactMatchByRelationshipTraversal | 0/2 | 0.000 | [0.000, 0.658] |
| qwen2.5-3b-instruct | exactMatchBySingleFact | 8/8 | 1.000 | [0.676, 1.000] |
| qwen2.5-3b-instruct | unanswerableAbstainRate | 0/3 | 0.000 | [0.000, 0.561] |
| qwen/qwen3-4b-2507 | layer1ExactMatchRate | 34/34 | 1.000 | [0.898, 1.000] |
| qwen/qwen3-4b-2507 | exactMatchByAggregation | 8/8 | 1.000 | [0.676, 1.000] |
| qwen/qwen3-4b-2507 | exactMatchByCompoundFilter | 4/4 | 1.000 | [0.510, 1.000] |
| qwen/qwen3-4b-2507 | exactMatchByConflictingFacts | 0/1 | 0.000 | [0.000, 0.793] |
| qwen/qwen3-4b-2507 | exactMatchByFilterList | 7/7 | 1.000 | [0.646, 1.000] |
| qwen/qwen3-4b-2507 | exactMatchByNotFound | 6/6 | 1.000 | [0.610, 1.000] |
| qwen/qwen3-4b-2507 | exactMatchByRelationshipTraversal | 0/3 | 0.000 | [0.000, 0.561] |
| qwen/qwen3-4b-2507 | exactMatchBySingleFact | 9/9 | 1.000 | [0.701, 1.000] |
| qwen/qwen3-4b-2507 | unanswerableAbstainRate | 1/4 | 0.250 | [0.046, 0.699] |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

## Verdicts

Comparison kind: **paired, cross-session** (§3.7).

### layer1ExactMatchRate

Not distinguishable at this sample size. Observed difference +0.0 pp, 95% CI [-11.4, 11.4] pp covers zero (b=0, c=0, McNemar exact p=1.000). This pack resolves differences of >=25.1 pp with 80% power at n=30 effective items (30 units, design effect 1.00, by-construction, alpha=0.05); the observed 0.0 pp is below that. Neither model is ranked above the other.

- paired n: 30 of 40 items (`asymmetry`: 0 scoreable for qwen2.5-3b-instruct only, 4 scoreable for qwen/qwen3-4b-2507 only; 6 unscoreable in both; 0 present in qwen2.5-3b-instruct only, 0 in qwen/qwen3-4b-2507 only) — §4.3
- marginal Wilson intervals overlap: yes
- decided by: mcnemar-exact

This pack resolves differences of >=25.1 pp with 80% power at n=30 effective items (30 units, design effect 1.00, by-construction, alpha=0.05). Differences below 20.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every item the models differ on. Inference is conditional on the 30 items in nlq-structured-query@1.2.0; generalization to unwritten items is not certified by any interval in this report.

**Headline (layer1ExactMatchRate):** Not distinguishable at this sample size. Observed difference +0.0 pp, 95% CI [-11.4, 11.4] pp covers zero (b=0, c=0, McNemar exact p=1.000). This pack resolves differences of >=25.1 pp with 80% power at n=30 effective items (30 units, design effect 1.00, by-construction, alpha=0.05); the observed 0.0 pp is below that. Neither model is ranked above the other.

### Exploratory metrics

- `exactMatchByAggregation` — exploratory — no significance claim
- `exactMatchByCompoundFilter` — exploratory — no significance claim
- `exactMatchByConflictingFacts` — exploratory — no significance claim
- `exactMatchByFilterList` — exploratory — no significance claim
- `exactMatchByNotFound` — exploratory — no significance claim
- `exactMatchByRelationshipTraversal` — exploratory — no significance claim
- `exactMatchBySingleFact` — exploratory — no significance claim
- `unanswerableAbstainRate` — exploratory — no significance claim

_The marginal-overlap line is a **diagnostic**, never the verdict: at this lab's sample sizes two marginal intervals overlapping is a far stronger condition than their difference covering zero, and the literal rule cannot fire at all at n<=40 with a baseline >=0.90 (`-ml` §3.1)._

