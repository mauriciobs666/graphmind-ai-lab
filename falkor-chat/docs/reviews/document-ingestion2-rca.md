# Document ingestion — update & delete — RCA: live-suite ANN search flakiness around Stage C

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M5, `document-ingestion2` Stage C)

CPG: considered, not relevant — `cpg_falkorchat` (`cpg/.cpg-artifacts/MANIFEST.txt`) models Python
call/data-flow structure; this defect is a live FalkorDB HNSW-engine recall behavior triggered by
node churn, not a code-reachability or call-graph question a CPG traversal would answer.

## 1. Symptom & impact

`pytest -q tests/test_repository.py tests/test_services.py tests/test_api.py tests/test_mcp.py
tests/test_graphrag.py` (and the full `pytest -q`) intermittently — but **deterministically per
exact test-order/count** — fails 2-6 tests whose assertions all reduce to the same shape:
`repository.search_chunks`'s `CALL db.idx.vector.queryNodes(...)` returns **fewer rows than the
just-written, exact-match corpus contains**, sometimes zero rows for a single freshly-embedded
chunk. Every failing test passes in isolation. Impact: the Stage C diff (uncommitted,
`falkor-chat/docs/plans/document-ingestion2.md` §3.3/§4) cannot be judged against a clean CI signal
until this is understood — which is what items 1-4 below settle.

## 2. Reproduction & evidence

Ran the exact command from the brief against the current working tree — reproduced verbatim (4
failed, all "expected chunk missing from `search_chunks` result", not a wrong value):

```
FAILED tests/test_graphrag.py::test_set_chunk_embedding_writes_and_chunk_is_ann_retrievable
FAILED tests/test_graphrag.py::test_search_chunks_ranks_by_cosine_distance_asc
FAILED tests/test_graphrag.py::test_search_chunks_returns_denormalized_document_id_and_seq
FAILED tests/test_graphrag.py::test_search_chunks_still_returns_current_chunks_alongside_a_superseded_one
4 failed, 739 passed
```

Post-mortem on the live `ws:test` graph immediately after that run (state left as of the last
executed test, `conn` only wipes at test *setup*, never teardown): exactly one `Chunk` node
existed (`c1`, `documentCurrent:false`, confirmed `embedding IS NOT NULL`), and a raw
`GRAPH.QUERY` `CALL db.idx.vector.queryNodes('Chunk','embedding',10, vecf32([1,0,0,0]))` — **no
`WHERE`, not routed through any application code** — returned **zero rows** for it. This already
falsifies "the new filter discards it": `ProcedureCall` itself is the empty step (Appendix A).

**Baseline check (the "single most important question" per the brief).** Built a real,
independent venv against a `git worktree add --detach <tmp> aa1c9be` checkout (not the earlier
invalid `PYTHONPATH`-override attempt — confirmed clean this time:
`falkorchat.__file__` resolved into the worktree, and
`'documentCurrent' in inspect.getsource(Repository.search_chunks)` → `False`). Ran the **identical**
5-file command against baseline:

```
FAILED tests/test_graphrag.py::test_search_chunks_ranks_by_cosine_distance_asc
FAILED tests/test_graphrag.py::test_search_chunks_returns_denormalized_document_id_and_seq
2 failed, 736 passed
```

**Two of the four current failures reproduce byte-for-byte on the pre-Stage-C baseline, with zero
Stage-C code present.** This is dispositive: whatever is failing is not caused by the `WHERE
seed.documentCurrent = true` filter or the `k = limit * SEARCH_DOCUMENTS_OVERFETCH` change — both
are entirely absent from the code that already fails this way.

**`GRAPH.PROFILE` on the new query** (item 2 of the brief), first on the degenerate zero-row state
(Appendix A), then reconstructed with real matching data (one `documentCurrent:true`, one `false`,
both scoring `0`, `k=10`) to see the filter actually discard a row:

```
--- WITH WHERE ---                              --- WITHOUT WHERE ---
Results  (1 row)                                Results  (2 rows)
  Sort  (1 row)                                    Sort  (2 rows)
    Project  (1 row)                                 Project  (2 rows)
      Filter  (1 row)                                  ProcedureCall  (2 rows)
        ProcedureCall  (2 rows)
```

`ProcedureCall` yields the same 2 candidates in both plans; `Filter` sits strictly **after** it and
discards the non-current row. **The plan's stated assumption in §3.3 — "this build's vector index
has no pre-filter predicate support demonstrated anywhere in this codebase's usage so far" —
holds exactly for this query shape.** Item 2 confirmed: no planner surprise, no pushdown.

**Churn-vs-recall characterization (item 3)**: a throwaway `ws:rca_churn_probe` graph, dim-4
`Chunk.embedding` index (mirrors `ws:test`/`TEST_EMBEDDING_DIM`), bootstrapped fresh, then
create+`DETACH DELETE` cycles of randomly-vectored `Chunk`s in increasing batches, checking recall
of a single exact-match target at `k∈{4,10,50}` after each batch (script + full log: Appendix B).
Result, from a clean index: **k=4 recall starts failing around 150-200 cumulative create/delete
cycles; k=10 stays reliable through 1000+ in that run, but the earlier post-mortem above shows
k=10 also failing once `ws:test`'s real session churn (hundreds of embedding writes across the
whole 739-test run) is high enough.** Recall degradation is a **monotonic function of cumulative
churn on the same live index**, reproduced with zero relation to `documentCurrent`/`WHERE`/Stage C
code — pure `db.idx.vector.queryNodes` + this build's HNSW behavior under churn.

## 3. Causal chain

1. `tests/conftest.py`'s `_schema` fixture is **session**-scoped: it drops and rebuilds `ws:test`
   (fresh vector index) exactly once per `pytest` process invocation. `conn` (function-scoped)
   only wipes **node data** (`MATCH (n) DETACH DELETE n`) before each test — the HNSW index
   structure itself is never rebuilt mid-session.
2. Every test in the session that embeds a `Chunk`/`Message` (hundreds, across
   `test_repository.py`/`test_services.py`/`test_api.py`/`test_mcp.py`/`test_graphrag.py`/
   `test_responder.py`/`test_tools.py`) contributes one more create+delete cycle to that same
   persistent index.
3. This FalkorDB build's HNSW recall for small `k` degrades monotonically with that accumulated
   churn (confirmed, §2/Appendix B) — a pre-existing, already-documented-in-spirit hazard
   (`claude/graph-dba/falkordb-quirks.md`: "ANN kNN returns *up to* `k`, not exactly `k`... may
   return fewer than `k`"; `falkor-chat/docs/SERVER.md` §1.7: "`ws:test`'s vector indexes are dim
   4... never point a real-embedder test at it"). Neither doc had this exact
   churn-count-vs-`k` curve measured before; this RCA adds that.
4. Because degradation is a function of **exact cumulative churn count at the moment a given test
   runs**, and `pytest` collects tests in **file order**, the failure set is a deterministic
   function of (a) total churn contributed by every test *before* a given ANN assertion in that
   invocation, and (b) that test's own `k`. This is exactly why the brief's own repro was
   "identical twice" — it is not flaky in the random sense, it is order-and-count-deterministic
   for a fixed test selection.
5. Stage C's diff adds 4-5 real chunk-churn events across `test_graphrag.py`/`test_api.py` (its
   new `test_services.py` case uses `FakeRepo`, contributing zero live churn) — all landing
   **before** `test_graphrag.py`'s tail in file order. That nudges the cumulative-churn count at
   every subsequent assertion up by a small, fixed amount, which is enough, at this suite's current
   size, to push 2 more pre-existing-shape assertions (`test_set_chunk_embedding_writes_and_
   chunk_is_ann_retrievable`, and the new `test_search_chunks_still_returns_current_chunks_
   alongside_a_superseded_one`, itself just another small-`k` ANN read) over the same recall cliff
   that already claims 2 *other* pre-existing tests on baseline.

**Hypotheses ruled out:**
- **(b) Planner pre-filter pushdown into the vector scan** — refuted directly by `GRAPH.PROFILE`
  (§2): `Filter` never fuses with or precedes `ProcedureCall`.
- **The filter's correctness on real data** — the `documentCurrent` exclusion behaves exactly as
  designed when ANN actually yields candidates (§2's real-data profile: the `false` row is
  correctly dropped, the `true` row correctly kept).

## 4. Root cause

**Confirmed** (reproduced end-to-end, both on the current tree and on an independently-built
pre-Stage-C baseline venv, plus an isolated clean-room churn experiment): the failure is
**pre-existing FalkorDB HNSW recall degradation under accumulated node churn on a long-lived,
never-mid-session-rebuilt, tiny-dimension (4) vector index** — a property of `tests/conftest.py`'s
session-scoped `ws:test` schema fixture combined with this build's ANN behavior, not of
`repository.search_chunks`'s new `WHERE seed.documentCurrent = true` filter or
`services.search_documents`'s new over-fetch multiplier.

**Trigger:** the *exact* test-order/count in a given `pytest` invocation, which determines how
much cumulative churn has landed on `ws:test`'s single Chunk vector index by the time each ANN
assertion runs. Stage C's diff is a trigger only in the weak sense of "adds a few more churn
events earlier in file order," not a mechanism-level cause.

**Contributing factors:** (1) `conftest.py`'s schema-rebuild boundary is session-, not
test-module-, scoped — a deliberate speed trade-off (`_schema`'s own docstring) that was never
load-bearing against ANN-recall correctness until the suite's total churn crossed this build's
degradation curve; (2) `TEST_EMBEDDING_DIM=4` sharpens the effect (already flagged, `SERVER.md`
§1.7, as a source of "ANN recall oddities," but never quantified); (3) no existing test or
assertion protects against "recall gets worse as the suite grows" — every new ANN-touching test
added anywhere in the suite silently spends down the same shared headroom.

## 5. Suggested fix & prevention

**Stage C's diff needs no change** — commit it as designed. The `WHERE seed.documentCurrent =
true` filter and the `k = limit * SEARCH_DOCUMENTS_OVERFETCH` over-fetch are both correct and
uninvolved in this failure (§2/§3).

**The real defect to fix is test infrastructure, not `search_chunks`/`search_documents`:**
suggest a function- or module-scoped vector-index rebuild boundary for ANN-sensitive test modules
(`test_graphrag.py`, `test_tools.py`, `test_responder.py`, and any `test_api.py`/`test_services.py`
node that embeds a `Chunk`/`Message` against a live graph) — e.g. a `fresh_vector_index` fixture
that reruns `_schema`'s drop+rebuild before each such module (or, cheaper, before each such test
class) rather than once per session, trading some suite runtime for eliminating cross-test churn
contamination. This bounds the failure at its actual cause instead of chasing the symptom test by
test as the suite keeps growing. A reproduction test for `tdd-engineer`/`qa-engineer` to hand this
to: the Appendix B churn script, adapted to assert recall failure at a specific churn count against
a **freshly-bootstrapped** probe graph — that is the guard that would have caught this class of
defect (a monotonic-in-churn regression test), not a fixed-`k` bump in any one test.

**As an immediate, narrower stop-gap** (buys headroom, does not fix the structural growth
problem): raise `k` in the specific currently-failing small-`k` ANN assertions
(`test_set_chunk_embedding_writes_and_chunk_is_ann_retrievable`,
`test_search_chunks_ranks_by_cosine_distance_asc`,
`test_search_chunks_returns_denormalized_document_id_and_seq`,
`test_search_chunks_still_returns_current_chunks_alongside_a_superseded_one`) — but flag this in
the commit/PR as a stop-gap: at this suite's current growth rate, some other small-`k` ANN test
will cross the same cliff again once more `Chunk`/`Message`-embedding tests are added, wherever
they land in file order.

**Also worth routing to `graph-dba`**: the churn-vs-recall curve characterized in Appendix B
(k=4 failing ~150-200 cycles, k=10 failing further out but confirmed reachable within a single
739-test session) is a new, precisely-measured data point for `claude/graph-dba/falkordb-quirks.md`
next to the existing "ANN kNN returns up to k, not exactly k" entry — that entry names the
phenomenon but had no churn-count curve before this investigation.

## What's solid

The Stage C design itself: the post-`YIELD` filter placement, the `exists()`-avoidance rationale,
and the over-fetch idiom mirroring `hybrid_search`'s own pattern are all sound and, per `GRAPH.
PROFILE`, behave exactly as the plan describes. The plan's own dropped-live-E2E-test note
(`test_api.py`, next to `test_search_documents_excludes_a_superseded_documents_chunks`) already
names "this build's HNSW ANN recall is not a simple function of k... at small scale" as a reason to
avoid asserting exact ANN counts — this RCA is a direct, deeper confirmation of exactly that
instinct, not a contradiction of it.

## Open questions

- Whether to fix the test-infra churn issue as part of landing Stage C, or file it as a separate
  follow-on ticket — the caller's call, not diagnosed here since it depends on release timing
  priorities outside this RCA's remit.
- Whether `efRuntime` (currently the FalkorDB default, `10`, per `db.indexes()`'s reported vector
  options) is tunable higher at index-creation time for `ws:test` specifically as an alternative
  mitigation to a rebuild-boundary fixture — not probed here; routes to `graph-dba` if the
  rebuild-boundary fix is judged too costly on suite runtime.

## Appendix A — post-mortem `GRAPH.PROFILE`/raw-query evidence (degenerate zero-row state)

```
$ redis-cli -p 6379 GRAPH.QUERY ws:test "MATCH (c:Chunk) RETURN c.chunkId, c.documentCurrent, c.documentId"
c1 | false | d1
$ redis-cli -p 6379 GRAPH.QUERY ws:test "MATCH (c:Chunk {chunkId:'c1'}) RETURN c.embedding IS NOT NULL"
true
$ redis-cli -p 6379 GRAPH.PROFILE ws:test "CALL db.idx.vector.queryNodes('Chunk','embedding',10,vecf32([1,0,0,0])) YIELD node AS seed, score WHERE seed.documentCurrent = true RETURN seed.chunkId, score"
Results (0)
  Project (0)
    Filter (0)
      ProcedureCall (0)
$ redis-cli -p 6379 GRAPH.PROFILE ws:test "CALL db.idx.vector.queryNodes('Chunk','embedding',10,vecf32([1,0,0,0])) YIELD node AS seed, score RETURN seed.chunkId, score"
Results (0)
  Project (0)
    ProcedureCall (0)
```
`ProcedureCall` alone returns 0 rows for a real, correctly-embedded, exact-match single node —
proves the empty result predates and is independent of the `WHERE` clause.

## Appendix B — churn-vs-recall probe (clean-room, `ws:rca_churn_probe`, dim 4)

Script: create N random-vector `Chunk`s then `DETACH DELETE` each immediately (simulating the
create/embed/delete cycle every embedding test performs), in increasing cumulative batches on the
**same never-rebuilt** index; after each batch, insert one exact-match target chunk and query ANN
at `k∈{4,10,50}`, then remove it again.

```
fresh index, zero churn        : k=4 found  k=10 found  k=50 found
after   10 cycles (cumulative) : k=4 found  k=10 found  k=50 found
after   50 cycles              : k=4 found  k=10 found  k=50 found
after  100 cycles              : k=4 found  k=10 found  k=50 found
after  200 cycles              : k=4 MISSED k=10 found  k=50 found
after  300 cycles              : k=4 MISSED k=10 found  k=50 found
after  500 cycles              : k=4 MISSED k=10 found  k=50 found
after 1000 cycles              : k=4 MISSED k=10 found  k=50 found
```
A second, longer run continuing churn on the *same* persistent index (not rebuilt between runs,
deliberately mirroring `ws:test`'s session-scoped rebuild boundary) reached k=10 failures by
500-1000 additional cycles, and k=50 stayed reliable throughout every run attempted — confirming
the effect is real, monotonic in cumulative churn, and exactly what a long `pytest` session
(hundreds of embedding writes against one never-mid-session-rebuilt index) reproduces.
