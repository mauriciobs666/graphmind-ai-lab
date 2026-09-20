# Ranked comparison — tool-caller-shop-assistant@0.2.0 (tool-caller)

### cleanThroughTurnH

> The analysis unit is scripts, n=12 (3 shapes x 4 scripts): floor 50.0 pp, MDD80 57.8 pp at this n (`-ml` §7.2/§4.5).

| rank | model | k/n | rate | 95% Wilson | latency p95 | footprint |
|---|---|---|---|---|---|---|
| 1 | qwen/qwen3-4b-thinking-2507 | 6/12 | 0.500 | [0.254, 0.746] | — | ~2.2-2.5 GB (Q4_K_M) |
| 2 | google/gemma-4-e2b | 5/12 | 0.417 | [0.193, 0.680] | — | ~2.7-3.0 GB (Q4_K_M) |
| 3 | mistralai/ministral-3-3b | 4/12 | 0.333 | [0.138, 0.609] | — | ~3.0 GB (Q8_0) |
| 4 | mistralai_ministral-3-3b-instruct-2512 | 4/12 | 0.333 | [0.138, 0.609] | — | ~3.0 GB (Q8_0) |
| 5 | nvidia/nemotron-3-nano-4b | 4/12 | 0.333 | [0.138, 0.609] | — | ~2.2-2.5 GB (Q4_K_M) |
| 6 | prism-ml/bonsai-27b | 1/12 | 0.083 | [0.015, 0.354] | — | ~3.5-5.5 GB (Q1_0) |
| 7 | qwen2.5-3b-instruct | 1/12 | 0.083 | [0.015, 0.354] | — | ~3.0 GB (Q8_0) |
| 8 | gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | 0/12 | 0.000 | [0.000, 0.242] | — | ~2.2-2.5 GB (Q4_K_M) |
| 9 | google/gemma-3-4b | 0/12 | 0.000 | [0.000, 0.242] | — | ~2.2-2.5 GB (Q4_K_M) |
| 10 | llama-3.2-3b-instruct | 0/3 | 0.000 | [0.000, 0.561] | — | ~3.2 GB (Q8_0) |
| 11 | qwen2.5-coder-3b-instruct | 0/12 | 0.000 | [0.000, 0.242] | — | ~3.0 GB (Q8_0) |
| 12 | qwen3.5-2b-claude-4.6-opus-reasoning-distilled | 0/12 | 0.000 | [0.000, 0.242] | — | ~1.3-1.5 GB (Q5_K_S) |
| 13 | qwen/qwen3-4b-2507 | 0/12 | 0.000 | [0.000, 0.242] | — | ~2.2-2.5 GB (Q4_K_M) |
| 14 | smollm3-3b | 0/12 | 0.000 | [0.000, 0.242] | — | ~3.0 GB (Q8_0) |
| 15 | stable-code-instruct-3b | 0/12 | 0.000 | [0.000, 0.242] | — | ~3.0 GB (Q8_0) |
| 16 | stablelm-zephyr-3b | 0/12 | 0.000 | [0.000, 0.242] | — | ~3.0 GB (Q8_0) |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

<!-- rank-report: pack=tool-caller-shop-assistant metric=cleanThroughTurnH top=qwen/qwen3-4b-thinking-2507 value=0.5000 ci=[0.2538,0.7462] -->

This pack resolves differences of >=57.8 pp with 80% power at n=12 effective conversations (12 units, design effect 1.00, by-construction, alpha=0.05). Differences below 50.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every conversation the models differ on; if it loses one for every two it wins, 80% power is not reached at any effect size at this n. Inference is conditional on the 12 scripts in tool-caller-shop-assistant@0.2.0; generalization to unwritten scripts is not certified by any interval in this report.

If this pack's optional reference-anchored family (FR-8) were run — any one of the 16 models here designated as the reference, the other 15 compared against it — that family of 15 tests would resolve differences of >=87.0 pp with 80% power (alpha_mdd = 0.05/15).

#### Reference-anchored family — cleanThroughTurnH vs `qwen/qwen3-4b-2507`

_exploratory — no significance claim outside this family_

| candidate | diff | 95% CI | Holm-adjusted threshold | decision |
|---|---|---|---|---|
| gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | +-0.0 pp | [-24.2, 24.2] pp | 0.0050 | not tested (Holm stops here) |
| google/gemma-3-4b | +-0.0 pp | [-24.2, 24.2] pp | 0.0056 | not tested (Holm stops here) |
| google/gemma-4-e2b | +41.7 pp | [8.7, 68.0] pp | 0.0036 | not tested (Holm stops here) |
| llama-3.2-3b-instruct | +-0.0 pp | [-56.1, 56.1] pp | 0.0063 | not tested (Holm stops here) |
| mistralai/ministral-3-3b | +33.3 pp | [2.2, 60.9] pp | 0.0038 | not tested (Holm stops here) |
| mistralai_ministral-3-3b-instruct-2512 | +33.3 pp | [2.2, 60.9] pp | 0.0042 | not tested (Holm stops here) |
| nvidia/nemotron-3-nano-4b | +33.3 pp | [2.2, 60.9] pp | 0.0045 | not tested (Holm stops here) |
| prism-ml/bonsai-27b | +8.3 pp | [-16.9, 35.4] pp | 0.0071 | not tested (Holm stops here) |
| qwen2.5-3b-instruct | +8.3 pp | [-16.9, 35.4] pp | 0.0083 | not tested (Holm stops here) |
| qwen2.5-coder-3b-instruct | +-0.0 pp | [-24.2, 24.2] pp | 0.0100 | not tested (Holm stops here) |
| qwen3.5-2b-claude-4.6-opus-reasoning-distilled | +-0.0 pp | [-24.2, 24.2] pp | 0.0125 | not tested (Holm stops here) |
| qwen/qwen3-4b-thinking-2507 | +50.0 pp | [15.4, 75.0] pp | 0.0033 | not distinguishable |
| smollm3-3b | +-0.0 pp | [-24.2, 24.2] pp | 0.0167 | not tested (Holm stops here) |
| stable-code-instruct-3b | +-0.0 pp | [-24.2, 24.2] pp | 0.0250 | not tested (Holm stops here) |
| stablelm-zephyr-3b | +-0.0 pp | [-24.2, 24.2] pp | 0.0500 | not tested (Holm stops here) |

