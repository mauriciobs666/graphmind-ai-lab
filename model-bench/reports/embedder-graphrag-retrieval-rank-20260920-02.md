# Ranked comparison — embedder-graphrag-retrieval@1.1.0 (embedder)

### mrr

> recall@10 = 37/38 at this pack's own item set: only 1 item is available to win, and McNemar needs 6 — this ranking can detect a materially worse embedder but cannot certify a better one (`-ml` §7.4).

| rank | model | n | mean | 95% CI | latency p95 | footprint |
|---|---|---|---|---|---|---|
| 1 | text-embedding-qwen3-embedding-4b | n=38 | 0.6610 | [0.5491, 0.7684] | 55 | ~2.2-2.5 GB (Q4_K_M) |
| 2 | text-embedding-nomic-embed-text-v1.5 | n=38 | 0.6407 | [0.5238, 0.7547] | 17 | ~0.08 GB (Q4_K_M) |
| 3 | text-embedding-granite-embedding-278m-multilingual | n=38 | 0.6317 | [0.5266, 0.7365] | 12 | ~0.28 GB (Q8_0) |
| 4 | text-embedding-qwen3-embedding-0.6b | n=38 | 0.6278 | [0.5148, 0.7387] | 35 | ~0.6 GB (Q8_0) |
| 5 | bm25 | n=38 | 0.5734 | [0.4573, 0.6916] | — | — |

_This interval describes this model's own mean; it is not a comparison, and two such intervals overlapping or not overlapping is not itself a basis for a verdict — see FR-8's optional reference-anchored family for an actual test, when one was run._

<!-- rank-report: pack=embedder-graphrag-retrieval metric=mrr top=text-embedding-qwen3-embedding-4b value=0.6610 ci=[0.5491,0.7684] -->

This pack resolves differences of >=20.1 pp with 80% power at n=38 effective querys (38 units, design effect 1.00, by-construction, alpha=0.05). Differences below 15.7 pp cannot reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case — assumes the candidate wins every query the models differ on. Inference is conditional on the 38 queries in embedder-graphrag-retrieval@1.1.0; generalization to unwritten queries is not certified by any interval in this report.

If this pack's optional reference-anchored family (FR-8) were run — any one of the 5 models here designated as the reference, the other 4 compared against it — that family of 4 tests would resolve differences of >=25.8 pp with 80% power (alpha_mdd = 0.05/4).

