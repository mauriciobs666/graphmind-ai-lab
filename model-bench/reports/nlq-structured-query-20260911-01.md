# Comparison — nlq-structured-query@1.2.0 (nlq-generator)

## Arms

| arm | metric | k/n | rate | 95% Wilson |
|---|---|---|---|---|
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

_None: fewer than two arms were selected, so there is nothing to compare. Check `--models` and `--session` against `model-bench models --tested`; a comparison needs two stored runs for this pack._

