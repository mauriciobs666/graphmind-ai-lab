# Comparison — embedder-graphrag-retrieval@1.1.0 (embedder)

## Arms

| arm | metric | k/n | rate | 95% Wilson |
|---|---|---|---|---|
| bm25 — reference arm (deterministic given pack version) | recallAt5 | 29/38 | 0.763 | [0.608, 0.870] |
| bm25 — reference arm (deterministic given pack version) | recallAt10 | 34/38 | 0.895 | [0.759, 0.958] |
| bm25 — reference arm (deterministic given pack version) | precisionAt1 | 15/38 | 0.395 | [0.256, 0.553] |
| bm25 — reference arm (deterministic given pack version) | mrr | n=38 | 0.5734 | — |
| bm25 — reference arm (deterministic given pack version) | separationRaw | n=38 | p50 -2.1578, p10 -16.0779 | — |
| bm25 — reference arm (deterministic given pack version) | separationZ | n=38 | p50 -0.9478, p10 -6.5980 | — |
| text-embedding-qwen3-embedding-0.6b | recallAt5 | 34/38 | 0.895 | [0.759, 0.958] |
| text-embedding-qwen3-embedding-0.6b | recallAt10 | 37/38 | 0.974 | [0.865, 0.995] |
| text-embedding-qwen3-embedding-0.6b | precisionAt1 | 17/38 | 0.447 | [0.301, 0.603] |
| text-embedding-qwen3-embedding-0.6b | mrr | n=38 | 0.6278 | — |
| text-embedding-qwen3-embedding-0.6b | separationRaw | n=38 | p50 -0.0436, p10 -0.1845 | — |
| text-embedding-qwen3-embedding-0.6b | separationZ | n=38 | p50 -0.4879, p10 -2.0352 | — |

_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._

## Verdicts

Comparison kind: **paired, same session** (§3.7).

### mrr

Not distinguishable at this sample size. Observed difference -0.054, 95% CI [-0.166, +0.055] covers zero, n=38 paired querys (unit: query, design effect 1.00, by-construction), decided by paired bootstrap on per-query differences (B=10000, seed=20260902). Neither model is ranked above the other.

- paired n: 38 of 38 querys (`asymmetry`: 0 scoreable for bm25 only, 0 scoreable for text-embedding-qwen3-embedding-0.6b only; 0 unscoreable in both; 0 present in bm25 only, 0 in text-embedding-qwen3-embedding-0.6b only) — §4.3

**Headline (mrr):** Not distinguishable at this sample size. Observed difference -0.054, 95% CI [-0.166, +0.055] covers zero, n=38 paired querys (unit: query, design effect 1.00, by-construction), decided by paired bootstrap on per-query differences (B=10000, seed=20260902). Neither model is ranked above the other.

### Exploratory metrics

- `recallAt5` — exploratory — no significance claim
- `recallAt10` — exploratory — no significance claim
- `precisionAt1` — exploratory — no significance claim
- `separationRaw` — exploratory — no significance claim
- `separationZ` — exploratory — no significance claim

_The marginal-overlap line is a **diagnostic**, never the verdict: at this lab's sample sizes two marginal intervals overlapping is a far stronger condition than their difference covering zero, and the literal rule cannot fire at all at n<=40 with a baseline >=0.90 (`-ml` §3.1)._

