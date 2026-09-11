# Embedder Harness Self-Check — Test Report

> **Status:** active · **Owner:** `coder` · **Tracks:** S3 Step 2 — `docs/plans/small-model-benchmarking-s3-spec.md` §8 Step 2 item 3, `docs/plans/small-model-benchmarking-ml.md` §5.4

Live execution of the `embedder-graphrag-retrieval` pack's harness self-check (`-ml` §5.4 point 1),
run 2026-09-11 against a reachable local LM Studio serving `text-embedding-qwen3-embedding-0.6b`
(catalog quantization `Q8_0`). **Diagnostic, never a gate** — the stakeholder ruling `-ml` §5.4
records (2026-09-02, decision 3) is unchanged by this run's result: S3 proceeds to done-conditions 1
and 2 regardless of what this number reads.

## The number

Exact brute-force cosine retrieval, the same 121-document corpus and 38 golden queries as
`retrieval_baseline.json`, the live model re-embedding all 38 queries fresh (no cached/fixture
vectors on this side — that is done-condition 4's job, not this one's):

| Metric | Live harness self-check | `retrieval_baseline.json` (pinned) |
|---|---|---|
| recall@10 | **0.9736842105263158** (37/38) | 0.9736842105263158 (37/38) |
| recall@5 | 0.8947368421052632 (34/38) | 0.8947368421052632 (34/38) |
| MRR | 0.6277568922305764 | 0.6258771929824561 |
| n | 38 | 38 |

`prime()` reported a cache HIT against the just-written `corpus.embeddings.json` (0.038 s, no live
call for the corpus side — the live cost here is 38 per-query embed calls, one per golden query).

## Reading this number

The raw self-check recall@10 is **0.9736842105263158**, well above the `-ml` §5.4 point 1 floor of
"at or above the ANN-based 0.974, and certainly not below ~0.85." **The below-~0.85 investigation
(wrong prefix, unnormalized vectors, truncated corpus) was NOT triggered** — nothing in this run
crossed that threshold, so none of the three causes needed checking.

Per `-ml` §5.4's own explicit ruling, this report does **not** treat the closeness of this run's
numbers to `retrieval_baseline.json`'s pinned figures as evidence of anything beyond "no harness
defect crossed the sanity floor." The pinned baseline came from falkor-chat's `hybrid_search`
(approximate in-graph ANN **plus** full-text); this harness is exact brute-force, vector-only. Two
differences that point in **opposite** directions on recall — exact ≥ ANN (no approximation loss),
vector-only ≤ hybrid (loses the keyword contribution) — mean a disagreement of either sign, *and*
an agreement of either sign, is uninterpretable as a quality signal here. This run's numbers landing
this close to the pinned baseline is **not** "explained" by either factor, and is not offered as
such; it is reported only as a sanity-floor pass, which is this diagnostic's entire job.

## Outcome

No harness defect indicated. S3 proceeded to done-conditions 1 (the real `run` invocation) and 2
(the BM25 deterministic arm, stored under the same `sessionId`) unconditionally, per the
diagnostic-never-a-gate ruling this report itself is bound by.
