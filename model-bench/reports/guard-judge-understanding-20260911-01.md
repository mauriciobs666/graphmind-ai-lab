# Comparison — guard-judge-understanding@1.0.0 (guard-judge)

## Arms

| arm | metric | k/n | rate | 95% Wilson |
|---|---|---|---|---|
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

_None: fewer than two arms were selected, so there is nothing to compare. Check `--models` and `--session` against `model-bench models --tested`; a comparison needs two stored runs for this pack._

