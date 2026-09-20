# Ranked comparison — chat-responder-grounded-answers@0.1.0 (chat-responder)

### groundingRate

> **Reply quality is not measured by this pack.** `groundingRate` is a deterministic containment check against the retrieved context, never a judgement of how good, helpful, or well-written a reply is (FR-21a — the judged-quality layer is deferred, `docs/BACKLOG.md`).

| rank | model | k/n | rate | 95% Wilson | latency p95 | footprint |
|---|---|---|---|---|---|---|
| 1 | qwen2.5-coder-3b-instruct | 24/30 | 0.800 | [0.627, 0.905] | 1180 | ~3.0 GB (Q8_0) |
| 2 | qwen/qwen3-4b-thinking-2507 | 24/30 | 0.800 | [0.627, 0.905] | 7796 | ~2.2-2.5 GB (Q4_K_M) |
| 3 | google/gemma-4-e2b | 23/30 | 0.767 | [0.591, 0.882] | 4902 | ~2.7-3.0 GB (Q4_K_M) |
| 4 | llama-3.2-3b-instruct | 23/30 | 0.767 | [0.591, 0.882] | 1918 | ~3.2 GB (Q8_0) |
| 5 | qwen/qwen3-4b-2507 | 23/30 | 0.767 | [0.591, 0.882] | 1038 | ~2.2-2.5 GB (Q4_K_M) |
| 6 | qwen2.5-3b-instruct | 22/30 | 0.733 | [0.556, 0.858] | 1480 | ~3.0 GB (Q8_0) |
| 7 | stablelm-zephyr-3b | 22/30 | 0.733 | [0.556, 0.858] | 1810 | ~3.0 GB (Q8_0) |
| 8 | smollm3-3b | 19/30 | 0.633 | [0.455, 0.781] | 2151 | ~3.0 GB (Q8_0) |
| 9 | google/gemma-3-4b | 18/30 | 0.600 | [0.423, 0.754] | 1872 | ~2.2-2.5 GB (Q4_K_M) |
| 10 | stable-code-instruct-3b | 14/30 | 0.467 | [0.302, 0.639] | 8587 | ~3.0 GB (Q8_0) |
| 11 | gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | 13/30 | 0.433 | [0.274, 0.608] | 1656 | ~2.2-2.5 GB (Q4_K_M) |
| 12 | mistralai/ministral-3-3b | 9/30 | 0.300 | [0.167, 0.479] | 2132 | ~3.0 GB (Q8_0) |
| 13 | mistralai_ministral-3-3b-instruct-2512 | 9/30 | 0.300 | [0.167, 0.479] | 2100 | ~3.0 GB (Q8_0) |
| 14 | nvidia/nemotron-3-nano-4b | 8/30 | 0.267 | [0.142, 0.444] | 3791 | ~2.2-2.5 GB (Q4_K_M) |
| 15 | prism-ml/bonsai-27b | 3/30 | 0.100 | [0.035, 0.256] | 23737 | ~3.5-5.5 GB (Q1_0) |
| 16 | qwen3.5-2b-claude-4.6-opus-reasoning-distilled | 0/30 | 0.000 | [0.000, 0.114] | 5129 | ~1.3-1.5 GB (Q5_K_S) |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

<!-- rank-report: pack=chat-responder-grounded-answers metric=groundingRate top=qwen2.5-coder-3b-instruct value=0.8000 ci=[0.6269,0.9049] -->

This pack resolves differences of >=25.1 pp with 80% power at n=30 effective items (30 units, design effect 1.00, by-construction, alpha=0.05). Differences below 20.0 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every item the models differ on. Inference is conditional on the 30 items in chat-responder-grounded-answers@0.1.0; generalization to unwritten items is not certified by any interval in this report.

If this pack's optional reference-anchored family (FR-8) were run — any one of the 16 models here designated as the reference, the other 15 compared against it — that family of 15 tests would resolve differences of >=39.3 pp with 80% power (alpha_mdd = 0.05/15).

#### Reference-anchored family — groundingRate vs `qwen/qwen3-4b-2507`

_exploratory — no significance claim outside this family_

| candidate | diff | 95% CI | Holm-adjusted threshold | decision |
|---|---|---|---|---|
| gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf | -33.3 pp | [-53.0, -8.6] pp | 0.0050 | not distinguishable |
| google/gemma-3-4b | -16.7 pp | [-37.5, 6.3] pp | 0.0063 | not tested (Holm stops here) |
| google/gemma-4-e2b | +-0.0 pp | [-18.4, 18.4] pp | 0.0083 | not tested (Holm stops here) |
| llama-3.2-3b-instruct | +-0.0 pp | [-10.6, 10.6] pp | 0.0100 | not tested (Holm stops here) |
| mistralai/ministral-3-3b | -46.7 pp | [-64.2, -21.8] pp | 0.0042 | distinguishable |
| mistralai_ministral-3-3b-instruct-2512 | -46.7 pp | [-64.2, -21.8] pp | 0.0045 | distinguishable |
| nvidia/nemotron-3-nano-4b | -50.0 pp | [-65.6, -27.0] pp | 0.0038 | distinguishable |
| prism-ml/bonsai-27b | -66.7 pp | [-78.8, -45.4] pp | 0.0036 | distinguishable |
| qwen2.5-3b-instruct | -3.3 pp | [-22.5, 16.1] pp | 0.0125 | not tested (Holm stops here) |
| qwen2.5-coder-3b-instruct | +3.3 pp | [-8.9, 15.9] pp | 0.0167 | not tested (Holm stops here) |
| qwen3.5-2b-claude-4.6-opus-reasoning-distilled | -76.7 pp | [-88.2, -55.7] pp | 0.0033 | distinguishable |
| qwen/qwen3-4b-thinking-2507 | +3.3 pp | [-8.9, 15.9] pp | 0.0250 | not tested (Holm stops here) |
| smollm3-3b | -13.3 pp | [-32.2, 6.8] pp | 0.0071 | not tested (Holm stops here) |
| stable-code-instruct-3b | -30.0 pp | [-51.1, -4.0] pp | 0.0056 | not tested (Holm stops here) |
| stablelm-zephyr-3b | -3.3 pp | [-24.3, 18.0] pp | 0.0500 | not tested (Holm stops here) |

