# Ranked comparison — embedder-graphrag-retrieval@1.1.0 (embedder)

### mrr

> recall@10 = 37/38 at this pack's own item set: only 1 item is available to win, and McNemar needs 6 — this ranking can detect a materially worse embedder but cannot certify a better one (`-ml` §7.4).

| rank | model | n | mean | 95% CI | latency p95 | footprint |
|---|---|---|---|---|---|---|
| 1 | text-embedding-qwen3-embedding-4b | n=38 | 0.6610 | [0.5491, 0.7684] | 55 | — |
| 2 | text-embedding-nomic-embed-text-v1.5 | n=38 | 0.6407 | [0.5238, 0.7547] | 17 | — |
| 3 | text-embedding-granite-embedding-278m-multilingual | n=38 | 0.6317 | [0.5266, 0.7365] | 12 | — |
| 4 | text-embedding-qwen3-embedding-0.6b | n=38 | 0.6278 | [0.5148, 0.7387] | 35 | — |
| 5 | bm25 | n=38 | 0.5734 | [0.4573, 0.6916] | — | — |

_This interval describes this model's own mean; it is not a comparison, and two such intervals overlapping or not overlapping is not itself a basis for a verdict — see FR-8's optional reference-anchored family for an actual test, when one was run._

<!-- rank-report: pack=embedder-graphrag-retrieval metric=mrr top=text-embedding-qwen3-embedding-4b value=0.6610 ci=[0.5491,0.7684] -->

#### Reference-anchored family — mrr vs `bm25`

_exploratory — no significance claim outside this family_

_`alpha_family=0.05, k=4, alpha_used=0.0125 (98.75% CI)`_

| candidate | diff | CI | decision |
|---|---|---|---|
| text-embedding-granite-embedding-278m-multilingual | +0.058 | [-0.012, +0.131] | not distinguishable |
| text-embedding-nomic-embed-text-v1.5 | +0.067 | [-0.022, +0.163] | not distinguishable |
| text-embedding-qwen3-embedding-0.6b | +0.054 | [-0.086, +0.192] | not distinguishable |
| text-embedding-qwen3-embedding-4b | +0.088 | [-0.026, +0.212] | not distinguishable |

