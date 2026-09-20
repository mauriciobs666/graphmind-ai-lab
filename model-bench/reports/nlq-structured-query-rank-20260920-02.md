# Ranked comparison — nlq-structured-query@1.2.0 (nlq-generator)

### layer1ExactMatchRate

> The true denominator is 34, not 40 — 6 items are structurally unanswerable and excluded (`-ml` §7.2, v1.25 note).

> **EXCLUDED — n=0 for `layer1ExactMatchRate`**
>
> Ran, and the stored record is internally consistent, but the declared aggregate honestly reports zero scoreable observations for this metric — excluded from the ranking below, never ranked "worst" and never silently absent either:
> - `qwen/qwen3-4b-thinking-2507` — 40 item(s) attempted, n=0 scored
> - `stable-code-instruct-3b` — 40 item(s) attempted, n=0 scored

| rank | model | k/n | rate | 95% Wilson | latency p95 | footprint |
|---|---|---|---|---|---|---|
| 1 | mistralai/ministral-3-3b | 32/32 | 1.000 | [0.893, 1.000] | 2635 | ~3.0 GB (Q8_0) |
| 2 | mistralai_ministral-3-3b-instruct-2512 | 32/32 | 1.000 | [0.893, 1.000] | 2796 | ~3.0 GB (Q8_0) |
| 3 | prism-ml/bonsai-27b | 8/8 | 1.000 | [0.676, 1.000] | 24422 | ~3.5-5.5 GB (Q1_0) |
| 4 | qwen2.5-3b-instruct | 30/30 | 1.000 | [0.886, 1.000] | 1265 | ~3.0 GB (Q8_0) |
| 5 | qwen/qwen3-4b-2507 | 34/34 | 1.000 | [0.898, 1.000] | 1408 | ~2.2-2.5 GB (Q4_K_M) |
| 6 | nvidia/nemotron-3-nano-4b | 30/31 | 0.968 | [0.838, 0.994] | 8536 | ~2.2-2.5 GB (Q4_K_M) |
| 7 | google/gemma-4-e2b | 16/17 | 0.941 | [0.730, 0.990] | 6528 | ~2.7-3.0 GB (Q4_K_M) |
| 8 | smollm3-3b | 29/34 | 0.853 | [0.699, 0.936] | 1383 | ~3.0 GB (Q8_0) |
| 9 | qwen2.5-coder-3b-instruct | 22/27 | 0.815 | [0.633, 0.918] | 2344 | ~3.0 GB (Q8_0) |
| 10 | llama-3.2-3b-instruct | 13/16 | 0.812 | [0.570, 0.934] | 1380 | ~3.2 GB (Q8_0) |
| 11 | google/gemma-3-4b | 2/3 | 0.667 | [0.208, 0.939] | 2073 | ~2.2-2.5 GB (Q4_K_M) |
| 12 | gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | 15/23 | 0.652 | [0.449, 0.812] | 1620 | ~2.2-2.5 GB (Q4_K_M) |
| 13 | qwen3.5-2b-claude-4.6-opus-reasoning-distilled | 5/8 | 0.625 | [0.306, 0.863] | 4991 | ~1.3-1.5 GB (Q5_K_S) |
| 14 | stablelm-zephyr-3b | 2/9 | 0.222 | [0.063, 0.547] | 2173 | ~3.0 GB (Q8_0) |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

<!-- rank-report: pack=nlq-structured-query metric=layer1ExactMatchRate top=mistralai/ministral-3-3b value=1.0000 ci=[0.8928,1.0000] -->

This pack resolves differences of >=22.3 pp with 80% power at n=34 effective items (34 units, design effect 1.00, by-construction, alpha=0.05). Differences below 17.6 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every item the models differ on. Inference is conditional on the 34 items in nlq-structured-query@1.2.0; generalization to unwritten items is not certified by any interval in this report.

If this pack's optional reference-anchored family (FR-8) were run — any one of the 16 models here designated as the reference, the other 15 compared against it — that family of 15 tests would resolve differences of >=34.9 pp with 80% power (alpha_mdd = 0.05/15).

#### Reference-anchored family — layer1ExactMatchRate vs `qwen/qwen3-4b-2507`

_exploratory — no significance claim outside this family_

| candidate | diff | 95% CI | Holm-adjusted threshold | decision |
|---|---|---|---|---|
| gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | -34.8 pp | [-55.1, -13.3] pp | 0.0033 | not distinguishable |
| google/gemma-3-4b | -33.3 pp | [-79.2, 29.1] pp | 0.0056 | not tested (Holm stops here) |
| google/gemma-4-e2b | -5.9 pp | [-27.0, 13.2] pp | 0.0063 | not tested (Holm stops here) |
| llama-3.2-3b-instruct | -18.8 pp | [-43.0, 4.1] pp | 0.0045 | not tested (Holm stops here) |
| mistralai/ministral-3-3b | +-0.0 pp | [-10.7, 10.7] pp | 0.0071 | not tested (Holm stops here) |
| mistralai_ministral-3-3b-instruct-2512 | +-0.0 pp | [-10.7, 10.7] pp | 0.0083 | not tested (Holm stops here) |
| nvidia/nemotron-3-nano-4b | -3.2 pp | [-16.2, 8.1] pp | 0.0100 | not tested (Holm stops here) |
| prism-ml/bonsai-27b | +-0.0 pp | [-32.4, 32.4] pp | 0.0125 | not tested (Holm stops here) |
| qwen2.5-3b-instruct | +-0.0 pp | [-11.4, 11.4] pp | 0.0167 | not tested (Holm stops here) |
| qwen2.5-coder-3b-instruct | -18.5 pp | [-36.7, -2.3] pp | 0.0038 | not tested (Holm stops here) |
| qwen3.5-2b-claude-4.6-opus-reasoning-distilled | -37.5 pp | [-69.4, 2.7] pp | 0.0050 | not tested (Holm stops here) |
| qwen/qwen3-4b-thinking-2507 | — | — | 0.0250 | no verdict — no paired data |
| smollm3-3b | -14.7 pp | [-30.1, -1.6] pp | 0.0042 | not tested (Holm stops here) |
| stable-code-instruct-3b | — | — | 0.0500 | no verdict — no paired data |
| stablelm-zephyr-3b | -77.8 pp | [-93.7, -33.6] pp | 0.0036 | not tested (Holm stops here) |

