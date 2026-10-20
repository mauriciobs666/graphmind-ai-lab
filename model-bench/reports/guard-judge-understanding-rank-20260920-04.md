# Ranked comparison — guard-judge-understanding@1.0.0 (guard-judge)

### falseAdvanceRate

> Two co-equal class-conditional error rates, no single headline: floor 15.0/20.0 pp, MDD80 21.9/28.7 pp for falseAdvanceRate/falseSuspendRate at the two-member alpha_mdd=0.025 (`-ml` §7.3).

| rank | model | k/n | rate | 95% Wilson | latency p95 | footprint |
|---|---|---|---|---|---|---|
| 1 | google/gemma-4-e2b | 0/40 | 0.000 | [0.000, 0.088] | 3253 | — |
| 2 | mistralai/ministral-3-3b | 0/40 | 0.000 | [0.000, 0.088] | 1296 | — |
| 3 | mistralai_ministral-3-3b-instruct-2512 | 0/40 | 0.000 | [0.000, 0.088] | 1411 | — |
| 4 | prism-ml/bonsai-27b | 0/40 | 0.000 | [0.000, 0.088] | 12167 | — |
| 5 | qwen/qwen3-4b-thinking-2507 | 0/40 | 0.000 | [0.000, 0.088] | 4252 | — |
| 6 | smollm3-3b | 1/40 | 0.025 | [0.004, 0.129] | 4898 | — |
| 7 | qwen2.5-3b-instruct | 2/40 | 0.050 | [0.014, 0.165] | 884 | — |
| 8 | google/gemma-3-4b | 3/40 | 0.075 | [0.026, 0.199] | 663 | — |
| 9 | nvidia/nemotron-3-nano-4b | 4/40 | 0.100 | [0.040, 0.231] | 3792 | — |
| 10 | qwen3.5-2b-claude-4.6-opus-reasoning-distilled | 4/40 | 0.100 | [0.040, 0.231] | 2578 | — |
| 11 | qwen/qwen3-4b-2507 | 4/40 | 0.100 | [0.040, 0.231] | 1079 | — |
| 12 | stable-code-instruct-3b | 5/40 | 0.125 | [0.055, 0.261] | 4975 | — |
| 13 | qwen2.5-coder-3b-instruct | 8/40 | 0.200 | [0.105, 0.348] | 1172 | — |
| 14 | llama-3.2-3b-instruct | 13/40 | 0.325 | [0.201, 0.480] | 645 | — |
| 15 | gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | 38/40 | 0.950 | [0.835, 0.986] | 599 | — |
| 16 | stablelm-zephyr-3b | 38/40 | 0.950 | [0.835, 0.986] | 1400 | — |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

<!-- rank-report: pack=guard-judge-understanding metric=falseAdvanceRate top=google/gemma-4-e2b value=0.0000 ci=[0.0000,0.0876] -->

This pack resolves differences of >=21.9 pp with 80% power at n=40 effective items (40 units, design effect 1.00, by-construction, alpha=0.025). Differences below 15.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every item the models differ on. Inference is conditional on the 40 items in guard-judge-understanding@1.0.0; generalization to unwritten items is not certified by any interval in this report.

If this pack's optional reference-anchored family (FR-8) were run — any one of the 16 models here designated as the reference, the other 15 compared against it jointly across both verdict metrics, — that family of 30 tests would resolve differences of >=32.6 pp with 80% power (alpha_mdd = 0.05/30).

#### Reference-anchored family — falseAdvanceRate vs `stablelm-zephyr-3b`

_exploratory — no significance claim outside this family_

| candidate | diff | 95% CI | Holm-adjusted threshold | decision |
|---|---|---|---|---|
| gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | +0.0 pp | [-10.3, 10.3] pp | 0.0250 | not tested (Holm stops here) |
| google/gemma-3-4b | +87.5 pp | [71.2, 93.4] pp | 0.0022 | distinguishable |
| google/gemma-4-e2b | +95.0 pp | [80.5, 98.6] pp | 0.0017 | distinguishable |
| llama-3.2-3b-instruct | +62.5 pp | [44.7, 74.9] pp | 0.0042 | distinguishable |
| mistralai/ministral-3-3b | +95.0 pp | [80.5, 98.6] pp | 0.0017 | distinguishable |
| mistralai_ministral-3-3b-instruct-2512 | +95.0 pp | [80.5, 98.6] pp | 0.0018 | distinguishable |
| nvidia/nemotron-3-nano-4b | +85.0 pp | [68.3, 91.8] pp | 0.0023 | distinguishable |
| prism-ml/bonsai-27b | +95.0 pp | [80.5, 98.6] pp | 0.0019 | distinguishable |
| qwen2.5-3b-instruct | +90.0 pp | [74.2, 95.0] pp | 0.0021 | distinguishable |
| qwen2.5-coder-3b-instruct | +75.0 pp | [57.4, 84.8] pp | 0.0028 | distinguishable |
| qwen3.5-2b-claude-4.6-opus-reasoning-distilled | +85.0 pp | [65.1, 92.9] pp | 0.0026 | distinguishable |
| qwen/qwen3-4b-2507 | +85.0 pp | [68.3, 91.8] pp | 0.0024 | distinguishable |
| qwen/qwen3-4b-thinking-2507 | +95.0 pp | [80.5, 98.6] pp | 0.0019 | distinguishable |
| smollm3-3b | +92.5 pp | [77.3, 96.6] pp | 0.0020 | distinguishable |
| stable-code-instruct-3b | +82.5 pp | [65.5, 90.1] pp | 0.0025 | distinguishable |

### falseSuspendRate

> Two co-equal class-conditional error rates, no single headline: floor 15.0/20.0 pp, MDD80 21.9/28.7 pp for falseAdvanceRate/falseSuspendRate at the two-member alpha_mdd=0.025 (`-ml` §7.3).

| rank | model | k/n | rate | 95% Wilson | latency p95 | footprint |
|---|---|---|---|---|---|---|
| 1 | gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | 0/30 | 0.000 | [0.000, 0.114] | 599 | — |
| 2 | stablelm-zephyr-3b | 0/30 | 0.000 | [0.000, 0.114] | 1400 | — |
| 3 | llama-3.2-3b-instruct | 4/30 | 0.133 | [0.053, 0.297] | 645 | — |
| 4 | qwen/qwen3-4b-2507 | 4/30 | 0.133 | [0.053, 0.297] | 1079 | — |
| 5 | nvidia/nemotron-3-nano-4b | 5/30 | 0.167 | [0.073, 0.336] | 3792 | — |
| 6 | qwen2.5-coder-3b-instruct | 8/30 | 0.267 | [0.142, 0.444] | 1172 | — |
| 7 | google/gemma-3-4b | 9/30 | 0.300 | [0.167, 0.479] | 663 | — |
| 8 | smollm3-3b | 12/30 | 0.400 | [0.246, 0.577] | 4898 | — |
| 9 | mistralai/ministral-3-3b | 20/30 | 0.667 | [0.488, 0.808] | 1296 | — |
| 10 | mistralai_ministral-3-3b-instruct-2512 | 20/30 | 0.667 | [0.488, 0.808] | 1411 | — |
| 11 | stable-code-instruct-3b | 23/30 | 0.767 | [0.591, 0.882] | 4975 | — |
| 12 | qwen3.5-2b-claude-4.6-opus-reasoning-distilled | 26/30 | 0.867 | [0.703, 0.947] | 2578 | — |
| 13 | google/gemma-4-e2b | 28/30 | 0.933 | [0.787, 0.982] | 3253 | — |
| 14 | qwen2.5-3b-instruct | 28/30 | 0.933 | [0.787, 0.982] | 884 | — |
| 15 | qwen/qwen3-4b-thinking-2507 | 29/30 | 0.967 | [0.833, 0.994] | 4252 | — |
| 16 | prism-ml/bonsai-27b | 30/30 | 1.000 | [0.886, 1.000] | 12167 | — |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

<!-- rank-report: pack=guard-judge-understanding metric=falseSuspendRate top=gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf value=0.0000 ci=[0.0000,0.1135] -->

This pack resolves differences of >=28.7 pp with 80% power at n=30 effective items (30 units, design effect 1.00, by-construction, alpha=0.025). Differences below 20.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every item the models differ on. Inference is conditional on the 30 items in guard-judge-understanding@1.0.0; generalization to unwritten items is not certified by any interval in this report.

If this pack's optional reference-anchored family (FR-8) were run — any one of the 16 models here designated as the reference, the other 15 compared against it jointly across both verdict metrics, — that family of 30 tests would resolve differences of >=42.7 pp with 80% power (alpha_mdd = 0.05/30).

#### Reference-anchored family — falseSuspendRate vs `stablelm-zephyr-3b`

_exploratory — no significance claim outside this family_

| candidate | diff | 95% CI | Holm-adjusted threshold | decision |
|---|---|---|---|---|
| gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | +0.0 pp | [-11.4, 11.4] pp | 0.0500 | not tested (Holm stops here) |
| google/gemma-3-4b | -30.0 pp | [-47.9, -12.5] pp | 0.0071 | distinguishable |
| google/gemma-4-e2b | -93.3 pp | [-98.2, -74.8] pp | 0.0033 | distinguishable |
| llama-3.2-3b-instruct | -13.3 pp | [-29.7, 0.6] pp | 0.0125 | not tested (Holm stops here) |
| mistralai/ministral-3-3b | -66.7 pp | [-80.8, -45.5] pp | 0.0050 | distinguishable |
| mistralai_ministral-3-3b-instruct-2512 | -66.7 pp | [-80.8, -45.5] pp | 0.0056 | distinguishable |
| nvidia/nemotron-3-nano-4b | -16.7 pp | [-33.6, -2.0] pp | 0.0100 | not distinguishable |
| prism-ml/bonsai-27b | -100.0 pp | [-100.0, -83.9] pp | 0.0029 | distinguishable |
| qwen2.5-3b-instruct | -93.3 pp | [-98.2, -74.8] pp | 0.0036 | distinguishable |
| qwen2.5-coder-3b-instruct | -26.7 pp | [-44.4, -9.8] pp | 0.0083 | distinguishable |
| qwen3.5-2b-claude-4.6-opus-reasoning-distilled | -86.7 pp | [-94.7, -66.8] pp | 0.0038 | distinguishable |
| qwen/qwen3-4b-2507 | -13.3 pp | [-29.7, 0.6] pp | 0.0167 | not tested (Holm stops here) |
| qwen/qwen3-4b-thinking-2507 | -96.7 pp | [-99.4, -79.2] pp | 0.0031 | distinguishable |
| smollm3-3b | -40.0 pp | [-57.7, -20.9] pp | 0.0063 | distinguishable |
| stable-code-instruct-3b | -76.7 pp | [-88.2, -55.7] pp | 0.0045 | distinguishable |

