# Ranked comparison — guard-judge-understanding@1.0.0 (guard-judge)

### falseAdvanceRate

> Two co-equal class-conditional error rates, no single headline: floor 15.0/20.0 pp, MDD80 21.9/28.7 pp for falseAdvanceRate/falseSuspendRate at the two-member alpha_mdd=0.025 (`-ml` §7.3).

| rank | model | k/n | rate | 95% Wilson | latency p95 | footprint |
|---|---|---|---|---|---|---|
| 1 | google/gemma-4-e2b | 0/40 | 0.000 | [0.000, 0.088] | 3253 | ~2.7-3.0 GB (Q4_K_M) |
| 2 | mistralai/ministral-3-3b | 0/40 | 0.000 | [0.000, 0.088] | 1296 | ~3.0 GB (Q8_0) |
| 3 | mistralai_ministral-3-3b-instruct-2512 | 0/40 | 0.000 | [0.000, 0.088] | 1411 | ~3.0 GB (Q8_0) |
| 4 | prism-ml/bonsai-27b | 0/40 | 0.000 | [0.000, 0.088] | 12167 | ~3.5-5.5 GB (Q1_0) |
| 5 | qwen/qwen3-4b-thinking-2507 | 0/40 | 0.000 | [0.000, 0.088] | 4252 | ~2.2-2.5 GB (Q4_K_M) |
| 6 | smollm3-3b | 1/40 | 0.025 | [0.004, 0.129] | 4898 | ~3.0 GB (Q8_0) |
| 7 | qwen2.5-3b-instruct | 2/40 | 0.050 | [0.014, 0.165] | 884 | ~3.0 GB (Q8_0) |
| 8 | google/gemma-3-4b | 3/40 | 0.075 | [0.026, 0.199] | 663 | ~2.2-2.5 GB (Q4_K_M) |
| 9 | nvidia/nemotron-3-nano-4b | 4/40 | 0.100 | [0.040, 0.231] | 3792 | ~2.2-2.5 GB (Q4_K_M) |
| 10 | qwen3.5-2b-claude-4.6-opus-reasoning-distilled | 4/40 | 0.100 | [0.040, 0.231] | 2578 | ~1.3-1.5 GB (Q5_K_S) |
| 11 | qwen/qwen3-4b-2507 | 4/40 | 0.100 | [0.040, 0.231] | 1079 | ~2.2-2.5 GB (Q4_K_M) |
| 12 | stable-code-instruct-3b | 5/40 | 0.125 | [0.055, 0.261] | 4975 | ~3.0 GB (Q8_0) |
| 13 | qwen2.5-coder-3b-instruct | 8/40 | 0.200 | [0.105, 0.348] | 1172 | ~3.0 GB (Q8_0) |
| 14 | llama-3.2-3b-instruct | 13/40 | 0.325 | [0.201, 0.480] | 645 | ~3.2 GB (Q8_0) |
| 15 | gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | 38/40 | 0.950 | [0.835, 0.986] | 599 | ~2.2-2.5 GB (Q4_K_M) |
| 16 | stablelm-zephyr-3b | 38/40 | 0.950 | [0.835, 0.986] | 1400 | ~3.0 GB (Q8_0) |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

<!-- rank-report: pack=guard-judge-understanding metric=falseAdvanceRate top=google/gemma-4-e2b value=0.0000 ci=[0.0000,0.0876] -->

This pack resolves differences of >=21.9 pp with 80% power at n=40 effective items (40 units, design effect 1.00, by-construction, alpha=0.025). Differences below 15.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every item the models differ on. Inference is conditional on the 40 items in guard-judge-understanding@1.0.0; generalization to unwritten items is not certified by any interval in this report.

If this pack's optional reference-anchored family (FR-8) were run — any one of the 16 models here designated as the reference, the other 15 compared against it jointly across both verdict metrics, — that family of 30 tests would resolve differences of >=32.6 pp with 80% power (alpha_mdd = 0.05/30).

#### Reference-anchored family — falseAdvanceRate vs `qwen/qwen3-4b-2507`

_exploratory — no significance claim outside this family_

| candidate | diff | 95% CI | Holm-adjusted threshold | decision |
|---|---|---|---|---|
| gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | -85.0 pp | [-91.8, -68.3] pp | 0.0017 | distinguishable |
| google/gemma-3-4b | +2.5 pp | [-10.0, 15.4] pp | 0.0083 | not tested (Holm stops here) |
| google/gemma-4-e2b | +10.0 pp | [-0.6, 23.1] pp | 0.0028 | not tested (Holm stops here) |
| llama-3.2-3b-instruct | -22.5 pp | [-37.3, -7.5] pp | 0.0026 | not tested (Holm stops here) |
| mistralai/ministral-3-3b | +10.0 pp | [-0.6, 23.1] pp | 0.0029 | not tested (Holm stops here) |
| mistralai_ministral-3-3b-instruct-2512 | +10.0 pp | [-0.6, 23.1] pp | 0.0031 | not tested (Holm stops here) |
| nvidia/nemotron-3-nano-4b | +0.0 pp | [-9.6, 9.6] pp | 0.0100 | not tested (Holm stops here) |
| prism-ml/bonsai-27b | +10.0 pp | [-0.6, 23.1] pp | 0.0033 | not tested (Holm stops here) |
| qwen2.5-3b-instruct | +5.0 pp | [-3.6, 15.9] pp | 0.0071 | not tested (Holm stops here) |
| qwen2.5-coder-3b-instruct | -10.0 pp | [-21.6, -0.2] pp | 0.0036 | not tested (Holm stops here) |
| qwen3.5-2b-claude-4.6-opus-reasoning-distilled | +0.0 pp | [-15.0, 15.0] pp | 0.0125 | not tested (Holm stops here) |
| qwen/qwen3-4b-thinking-2507 | +10.0 pp | [-0.6, 23.1] pp | 0.0038 | not tested (Holm stops here) |
| smollm3-3b | +7.5 pp | [-1.7, 19.7] pp | 0.0063 | not tested (Holm stops here) |
| stable-code-instruct-3b | -2.5 pp | [-16.7, 11.5] pp | 0.0167 | not tested (Holm stops here) |
| stablelm-zephyr-3b | -85.0 pp | [-91.8, -68.3] pp | 0.0017 | distinguishable |

### falseSuspendRate

> Two co-equal class-conditional error rates, no single headline: floor 15.0/20.0 pp, MDD80 21.9/28.7 pp for falseAdvanceRate/falseSuspendRate at the two-member alpha_mdd=0.025 (`-ml` §7.3).

| rank | model | k/n | rate | 95% Wilson | latency p95 | footprint |
|---|---|---|---|---|---|---|
| 1 | gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | 0/30 | 0.000 | [0.000, 0.114] | 599 | ~2.2-2.5 GB (Q4_K_M) |
| 2 | stablelm-zephyr-3b | 0/30 | 0.000 | [0.000, 0.114] | 1400 | ~3.0 GB (Q8_0) |
| 3 | llama-3.2-3b-instruct | 4/30 | 0.133 | [0.053, 0.297] | 645 | ~3.2 GB (Q8_0) |
| 4 | qwen/qwen3-4b-2507 | 4/30 | 0.133 | [0.053, 0.297] | 1079 | ~2.2-2.5 GB (Q4_K_M) |
| 5 | nvidia/nemotron-3-nano-4b | 5/30 | 0.167 | [0.073, 0.336] | 3792 | ~2.2-2.5 GB (Q4_K_M) |
| 6 | qwen2.5-coder-3b-instruct | 8/30 | 0.267 | [0.142, 0.444] | 1172 | ~3.0 GB (Q8_0) |
| 7 | google/gemma-3-4b | 9/30 | 0.300 | [0.167, 0.479] | 663 | ~2.2-2.5 GB (Q4_K_M) |
| 8 | smollm3-3b | 12/30 | 0.400 | [0.246, 0.577] | 4898 | ~3.0 GB (Q8_0) |
| 9 | mistralai/ministral-3-3b | 20/30 | 0.667 | [0.488, 0.808] | 1296 | ~3.0 GB (Q8_0) |
| 10 | mistralai_ministral-3-3b-instruct-2512 | 20/30 | 0.667 | [0.488, 0.808] | 1411 | ~3.0 GB (Q8_0) |
| 11 | stable-code-instruct-3b | 23/30 | 0.767 | [0.591, 0.882] | 4975 | ~3.0 GB (Q8_0) |
| 12 | qwen3.5-2b-claude-4.6-opus-reasoning-distilled | 26/30 | 0.867 | [0.703, 0.947] | 2578 | ~1.3-1.5 GB (Q5_K_S) |
| 13 | google/gemma-4-e2b | 28/30 | 0.933 | [0.787, 0.982] | 3253 | ~2.7-3.0 GB (Q4_K_M) |
| 14 | qwen2.5-3b-instruct | 28/30 | 0.933 | [0.787, 0.982] | 884 | ~3.0 GB (Q8_0) |
| 15 | qwen/qwen3-4b-thinking-2507 | 29/30 | 0.967 | [0.833, 0.994] | 4252 | ~2.2-2.5 GB (Q4_K_M) |
| 16 | prism-ml/bonsai-27b | 30/30 | 1.000 | [0.886, 1.000] | 12167 | ~3.5-5.5 GB (Q1_0) |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

<!-- rank-report: pack=guard-judge-understanding metric=falseSuspendRate top=gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf value=0.0000 ci=[0.0000,0.1135] -->

This pack resolves differences of >=28.7 pp with 80% power at n=30 effective items (30 units, design effect 1.00, by-construction, alpha=0.025). Differences below 20.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every item the models differ on. Inference is conditional on the 30 items in guard-judge-understanding@1.0.0; generalization to unwritten items is not certified by any interval in this report.

If this pack's optional reference-anchored family (FR-8) were run — any one of the 16 models here designated as the reference, the other 15 compared against it jointly across both verdict metrics, — that family of 30 tests would resolve differences of >=42.7 pp with 80% power (alpha_mdd = 0.05/30).

#### Reference-anchored family — falseSuspendRate vs `qwen/qwen3-4b-2507`

_exploratory — no significance claim outside this family_

| candidate | diff | 95% CI | Holm-adjusted threshold | decision |
|---|---|---|---|---|
| gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | +13.3 pp | [-0.6, 29.7] pp | 0.0042 | not tested (Holm stops here) |
| google/gemma-3-4b | -16.7 pp | [-33.2, -0.0] pp | 0.0045 | not tested (Holm stops here) |
| google/gemma-4-e2b | -80.0 pp | [-88.9, -59.2] pp | 0.0019 | distinguishable |
| llama-3.2-3b-instruct | +0.0 pp | [-17.2, 17.2] pp | 0.0250 | not tested (Holm stops here) |
| mistralai/ministral-3-3b | -53.3 pp | [-67.5, -32.7] pp | 0.0022 | distinguishable |
| mistralai_ministral-3-3b-instruct-2512 | -53.3 pp | [-67.5, -32.7] pp | 0.0023 | distinguishable |
| nvidia/nemotron-3-nano-4b | -3.3 pp | [-19.3, 12.4] pp | 0.0500 | not tested (Holm stops here) |
| prism-ml/bonsai-27b | -86.7 pp | [-94.7, -66.8] pp | 0.0018 | distinguishable |
| qwen2.5-3b-instruct | -80.0 pp | [-88.9, -59.2] pp | 0.0020 | distinguishable |
| qwen2.5-coder-3b-instruct | -13.3 pp | [-29.4, 2.4] pp | 0.0056 | not tested (Holm stops here) |
| qwen3.5-2b-claude-4.6-opus-reasoning-distilled | -73.3 pp | [-83.8, -52.1] pp | 0.0021 | distinguishable |
| qwen/qwen3-4b-thinking-2507 | -83.3 pp | [-91.6, -63.0] pp | 0.0019 | distinguishable |
| smollm3-3b | -26.7 pp | [-42.2, -10.5] pp | 0.0025 | not distinguishable |
| stable-code-instruct-3b | -63.3 pp | [-78.9, -36.5] pp | 0.0024 | distinguishable |
| stablelm-zephyr-3b | +13.3 pp | [-0.6, 29.7] pp | 0.0050 | not tested (Holm stops here) |

