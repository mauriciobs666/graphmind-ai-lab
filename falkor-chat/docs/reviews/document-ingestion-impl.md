# Document Ingestion — Stage 1 Implementation Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-050 (M5)

**Scope.** Diff-scoped code gate (distinct from the plan gate at `docs/reviews/document-ingestion.md`)
against the uncommitted working-tree changes implementing Stage 1 (chunking + `Document`/`Chunk`
write path) of `docs/plans/document-ingestion.md`. Baseline: the locked plan (§3.2, §2.4, §3.5, §3.6,
§4 Stage 1) and `docs/plans/document-ingestion-graph.md` §2.1/§2.4 (live-verified Cypher). Files
reviewed: `server/falkorchat/{chunking.py (new), repository.py, services.py, schemas.py, mcp.py,
api.py}`, `server/tests/{test_chunking.py (new), test_repository.py, test_services.py, test_api.py,
test_mcp.py}`, `docs/QUERIES.md` §14, `docs/HISTORY.md`. Stage 2+ (embedding, extraction, fusion,
chat-grounding) is out of scope — none of it exists in this diff, and this review does not comment on
it. `docs/plans/document-ingestion-coordination.md`'s own U9/U10 entries were read for context but
not treated as authoritative for the verification claimed below — every executed claim in this
review (test runs, mutation-test spot check, DDL/no-change confirmation) was independently re-run
in this session, not inherited from that log.

**CPG:** considered, not relevant — `cpg_falkorchat` is stale relative to `server/` (7 commits since
its 2026-08-17 build, none ingestion-related, confirmed in the brief) and this diff is net-new code;
read the files directly instead, per the brief's own note.

**Verdict: approve with suggestions.** One MAJOR finding (a real, verified behavioral gap — an empty/
whitespace-only document is silently accepted and produces a permanently-`processing`, zero-chunk
`Document`, contradicting `chunking.py`'s own docstring claim about where that's rejected); everything
else is solid. Not a blocker: it doesn't corrupt data, doesn't violate a tested AC, and is cheap to
fix in a follow-up before Stage 6 (or now, implementer's call).

---

## Findings

### MAJOR — empty/whitespace-only input is silently accepted, not rejected as claimed

`chunking.py`'s own docstring states: *"Empty or whitespace-only input yields `[]` — chunking is not
the layer that rejects that (the service validates non-empty input upstream)"*
(`server/falkorchat/chunking.py:26-28`). **This claim is false as implemented.**
`services.ingest_document` (`server/falkorchat/services.py:970-1004`) only checks
`len(text) > MAX_DOCUMENT_CHARS` — there is no check for empty or whitespace-only text anywhere in
the service. `IngestDocumentIn.text = Field(min_length=1, ...)` (`schemas.py:59`) blocks a literal
`""` at the REST boundary only, but `min_length=1` does **not** block a single space or any other
whitespace-only string, and the MCP tool (`mcp.py`) has no schema layer at all, so this is completely
unguarded on that transport.

Verified live, both at the service layer and end-to-end over REST against the running server
(`ws:test`, via a scripted `TestClient` call, not just an inspection):

```
POST /documents {"text": "   \n\n\t  "}  →  201 {"chunkCount": 0, "status": "processing"}
GET /documents/{id}  →  {"text": "   \n\n\t  ", "chunkCount" absent (chunks=[]), "status": "processing", ...}
```

A `Document` node is created with zero `Chunk`s, `HAS_CHUNK` edges are never written, and
`status` reports `'processing'` — a state nothing in the pipeline will ever advance, since a
chunkless document has nothing for any later stage to act on. This is silently reachable by any
connected MCP agent (FR-5's own stated audience) with no feedback that the call did nothing useful.

This is not a new deviation from existing convention — `PostMessageIn.text` has the identical
`min_length=1`-only posture, so a whitespace-only chat message is already tolerated today — but a
`Message` has no `status` field for that to strand, so the consequence is materially different for a
`Document`.

**Suggested fix:** either (a) reject empty/whitespace-only `text` in `services.ingest_document`
itself, mirroring the `MAX_DOCUMENT_CHARS` check's shape (a small `if not text.strip(): raise
<SomeError>` ahead of the chunking call — a new `EmptyDocumentError` or reuse of an existing
validation-error convention, implementer's call), which is what the `chunking.py` docstring already
claims happens; or (b) if an empty document is actually meant to be a legal degenerate case, fix the
docstring to stop claiming otherwise and decide what `chunkCount: 0`/`status: 'processing'` should
mean for a document nothing will ever advance out of `'processing'`. (a) is almost certainly the
right call — nothing in the plan or requirements doc contemplates a zero-content document as
meaningful, and the docstring shows the implementer's own intent was to reject it.

### What's solid

- **Cypher fidelity — exact match.** `repository.create_document`'s Cypher
  (`repository.py:960-989`) is byte-identical (up to inert string-literal concatenation) to
  `document-ingestion-graph.md` §2.1's live-verified query, including the actor-guard `FOREACH`
  wrapper, the `sourceKind` derivation, and the `$chunks` list-of-maps parameter shape. No silent
  drift.
- **`get_document`'s explicit-field-projection deviation is the right call, not a lapse.**
  The plan/graph note's illustrative Cypher does `RETURN d, ...` (a whole node); the implementation
  instead projects explicit properties, matching the pattern that `get_message` already uses
  (`repository.py:1016-1035`) — this is actually *more* consistent with the codebase's own existing
  convention than the graph note's own snippet was, not a deviation worth flagging back to
  `graph-dba`.
- **`title` falling back to `source_label` and the extra REST-layer `MAX_DOCUMENT_CHARS` check** are
  both reasonable, low-stakes implementation judgment calls squarely inside "how", not "what" — the
  plan left the title-default unspecified and the REST-layer bound duplicates rather than contradicts
  the service-layer one (same constant, imported from `schemas.py`), mirroring how `PostMessageIn`
  already double-binds message-length caps at both REST and (implicitly) nowhere else needed. Neither
  should have gone back to `architect`/`graph-dba`.
- **Actor guard / DDL claims confirmed live, not just trusted.** Re-ran
  `git diff -- scripts/bootstrap_schema.sh` (empty) and read the live script: `Document`/`Chunk`
  range index + uniqueness constraint and the `Chunk.embedding` vector index all predate this diff;
  `Chunk.seq`/`Chunk.documentId` are genuinely plain unindexed properties, exactly as claimed.
- **Scope discipline is clean.** No `background.py`/`embedding.py`/`extraction.py`/`fusion.py`/
  `ingestion.py`/`responder.py`/`config/models.json` changes anywhere in the diff; `grep`-verified no
  `'ready'` status flip exists outside doc comments describing a *future* stage. `Document.status`
  genuinely sits at `'processing'` unconditionally.
- **Test quality is good.** Re-ran the full offline suite myself: `1563 passed, 3 deselected`
  (matches the reported count) and `./scripts/test_queries.sh`: `320/320` (matches, unchanged as
  expected — no new DDL/Cypher shape entered that suite). Did an independent mutation-test spot
  check beyond trusting the report: broke the actor guard in `repository.create_document`
  (hardcoded `ingestor_found=True`) and confirmed exactly the two tests the report claims would catch
  it actually fail (`test_repository.py::test_create_document_unknown_actor_nothing_written`,
  `test_mcp.py::test_ingest_document_unknown_actor_errors`), then reverted — `git diff` afterward
  shows the file back to its original diff state. The AC-9 round trip, actor-kind derivation
  (`User`→`document`, `Agent`→`agent`), chunk ordering/denorm, and the non-idempotent-retry test are
  all real integration tests against the live `ws:test` graph, not mocks dressed up as coverage.
  `chunking.py`'s own unit tests are thorough on the stated boundary rule, including a byte-exact
  reconstruction proof for the sentence-split path (`test_long_paragraph_splits_on_sentence_
  boundaries`) and explicit acknowledgment that a chunk may run up to `size + overlap` chars at a
  hard-cut/overlap boundary (`test_long_single_sentence_hard_cuts`) — a real, intentional soft-cap
  nuance, correctly tested rather than silently left undocumented.
- **Conventions fit is native.** Layering (`repository.py` owns Cypher 1:1 with `QUERIES.md`,
  `services.py` owns id/timestamp minting + the one cross-transport invariant, `api.py`/`mcp.py` are
  thin) mirrors the existing message/channel write paths exactly; `DocumentTooLargeError` and
  `UnknownActorError` both map through the same generic `ServiceError`→HTTP-status handler every
  other service error already uses (`api.py:151,175`), not a bespoke path.
- **`QUERIES.md` §14 and `HISTORY.md`** are accurate, complete, and correctly disclose the
  `get_document` projection deviation and the scope boundary (Stage 1 only). Both restored the shared
  `reference` graph after the test runs that wiped it — a housekeeping step this review also had to
  redo (`bootstrap_schema.sh acme && seed_workflows.sh acme`, confirmed back in sync via
  `verify_workflows.sh acme`) since running the same suites myself wiped it again.

---

## Open questions

- Should the empty/whitespace-only rejection (MAJOR finding above) land before Stage 6's QA
  acceptance pass, or is a `chunkCount: 0` document an acceptable (if unintended) degenerate case for
  v1? The plan/requirements doc don't address it either way — worth a quick coordinator call rather
  than assuming.

---

## Pass 2 (2026-08-24) — Stage 2 diff-scoped code gate

**Scope.** Diff-scoped code gate against the (then-)uncommitted working-tree changes implementing
Stage 2 (chunk embeddings + standalone search, FR-3) of `docs/plans/document-ingestion.md`. This is
a **new** diff, layered on top of Stage 1's already-gated (and by now committed, `167eeac`) changes —
not a re-review of Pass 1's findings, which are unaffected (the Stage 1 MAJOR finding was fixed
separately, U11 in `docs/plans/document-ingestion-coordination.md`, verified present at
`services.py:1001-1014` and out of scope here). Baseline: the locked plan (§3.7 FR-3, §4 Stage 2) and
`docs/plans/document-ingestion-graph.md` (consulted; Stage 2's Cypher — `set_chunk_embedding`,
`search_chunks`, `list_document_chunks` — has no dedicated graph-dba section since it's a direct
analogue of already-verified §6 `Message` patterns, confirmed by reading `QUERIES.md` §14.3/§14.4
directly against the implementation rather than assuming). Files reviewed: `server/falkorchat/
{embedding,repository,background,services,api,mcp,app}.py` and their 7 corresponding test files
(`test_embedding.py`, `test_background.py`, `test_repository.py`, `test_graphrag.py`, `test_services.py`,
`test_api.py`, `test_mcp.py`), `docs/QUERIES.md` §14.3/§14.4, `docs/HISTORY.md`'s new Stage 2 entry.
Per the brief, suite counts (1566→1597 offline, +31; 320/320 query suite unchanged) and doc
completeness/accuracy were independently pre-verified by the dispatching session and are taken as
given here — this pass focuses on code correctness, Cypher fidelity, FR-19/scope-gating correctness,
and spot-checking the claimed mutation tests against their actual assertions.

**Verdict: approve.** No blockers, no majors. One minor (a real, newly-introduced resource-usage
concern, not a correctness bug) and one nit.

**CPG:** considered, not relevant — `cpg_falkorchat` is stale relative to this diff (built
2026-08-17, 9+ commits since including both Stage 1 and this Stage 2 diff, per the brief's own
freshness note) and this is a small, self-contained diff over freshly-read current files, not a
call-graph impact-analysis task; read the files directly instead, as directed.

### Findings

**MINOR — MCP's chunk-embed scheduling amplifies an already-accepted unbounded-thread trade-off by
up to ~500x per call, undocumented at the new scale.**

`mcp.py`'s `_default_schedule` (`mcp.py:47`) spawns one raw, untracked daemon `threading.Thread`
per scheduled call — a trade-off the codebase already knowingly accepts for M1's lab-scale posture
(its own docstring cites `docs/reviews/mcp-background-scheduling-impl.md` Minor 1/2). Before this
diff, that trade-off was exercised at most once per MCP call (`send_message` → one `_safe_embed`
thread). This diff's `ingest_document` chunk-scheduling loop (`mcp.py:235-237`) calls `_schedule`
once per chunk, synchronously, inside the tool handler, before the call returns — at the plan's own
stated bounds (`MAX_DOCUMENT_CHARS = 500_000` at the §3.2 1,000-char target ≈ 500-600 chunks), a
single `ingest_document` MCP call now spawns on the order of 500 raw OS threads in a tight loop
before the tool returns. This is a materially different order of magnitude from the precedent the
existing docstring's risk framing describes (1 thread/call), even though the *general* fan-out risk
was already named in the plan (`document-ingestion.md` §3.5, in the context of Stage 3's extraction
fan-out, not Stage 2's embedding). REST's equivalent path (`api.py:174-183`) does not have this
amplification — `BackgroundTasks.add_task` is a cheap list append per chunk, bounded execution
happens later via anyio's ~40-worker capacity limiter — so the two transports now diverge more
sharply in resource behavior than they did before this diff, and neither `HISTORY.md`'s new entry
nor `QUERIES.md` §14.4 mentions the amplification. Not a blocker: it's lab-scale, self-contained
(daemon threads that do a bounded amount of I/O and exit), and consistent with an already-accepted
codebase posture, not a new kind of risk. **Suggested improvement:** either cap/batch the MCP
chunk-embed scheduling (e.g. one thread that iterates the chunk list internally, rather than one
thread per chunk), or — cheaper — add a line to `mcp.py`'s `_default_schedule` docstring and/or
`QUERIES.md` §14.4 noting that `ingest_document` can fan this out to hundreds of threads per call,
so a future reader of the "accepted for M1 lab-scale" reasoning isn't reasoning about 1x when the
real number is now ~500x.

**NIT — `search_documents`'s MCP tool has no `limit` upper bound, consistent with existing precedent
but worth naming.** `mcp.py`'s `search_documents(query, limit=20)` (no REST-style `Query(..., le=200)`
cap) mirrors the pre-existing `search_messages(query, limit=50)` MCP tool
(`mcp.py:175-182`), which has the identical gap — not a regression this diff introduces, and matching
established convention is the right call here, not a lapse. Flagged only because the REST sibling
(`api.py:192`, `Query(20, ge=1, le=200)`) does bound it, so the two transports diverge on this one
input the same way `search_messages`'s REST/MCP pair already does — a pre-existing, not new,
inconsistency; no action needed for this diff specifically.

### Verified claims (evidence, not trust)

- **`EmbeddingWorker._resolve_and_embed` factor-out is correct and independently gated.** Read
  `embedding.py:86-239` (`class EmbeddingWorker`, `_resolve_and_embed` at :139, `embed_message` at
  :208, `embed_chunk` at :222) in full: `embed_message`/`embed_chunk` each pass their own `write_label`
  (`"Message"`/`"Chunk"`) into the shared helper, which is the only place `_index_dimension(ws,
  write_label)` is called — a `Chunk` embed genuinely never consults `Message`'s index and vice
  versa. Confirmed against the tests, not just the prose:
  `test_worker_gates_chunk_embed_on_chunk_index_only_never_message` and the pre-existing
  `test_worker_never_queries_or_considers_chunk_when_writing_a_message`
  (`test_embedding.py:293-421`) are the mirror-image pair that would catch a label mix-up in either
  direction.
- **Cypher fidelity — exact match against `QUERIES.md` §14.3.** `repository.set_chunk_embedding`
  (`repository.py:1041`) and `search_chunks` (`repository.py:1071`) are byte-identical (up
  to inert string concatenation) to the documented Cypher, including the pre-write dimension guard
  ported verbatim from `set_embedding` and `search_chunks`'s explicit `ORDER BY score ASC` (no
  scope traversal, no `ABOUT` expansion — correctly deferred, not a gap: `ABOUT` stays unpopulated
  until Stage 3, so an `OPTIONAL MATCH` on it today would just no-op).
- **`GET /documents/search` route-ordering fix is real, correctly placed, and behaviorally
  verified — not just claimed.** Grepped `api.py` for every `@router.get("/documents...")`
  registration (`api.py:161,189,197`): `POST /documents` → `GET /documents/search` → `GET
  /documents/{document_id}`, in that literal source order. This is the right fix for the right
  reason — Starlette/FastAPI's `APIRouter` matches routes in registration order, so a static path
  registered ahead of a dynamic sibling wins; a `{document_id}` route registered first would swallow
  `/documents/search` as `document_id="search"`. `test_api.py`'s
  `test_search_documents_route_not_shadowed_by_document_id_route` is a real regression guard for this
  specific claim, not a restatement of it — it exercises the live route table via `TestClient`, not
  a unit-level assertion about registration order.
- **`SearchNotAvailableError` → 503 wiring is complete and consistent.** `services.py`'s new
  exception class, `app.py`'s `_register_error_handlers` registration (`app.py:150-155`), and the
  “deployment gap, not caller mistake” posture all mirror `WorkflowEngineDisabledError` exactly —
  confirmed by reading all three sites together, not inferring from one.
- **Mutation-test claims spot-checked against actual test assertions, not just labels — all six
  hold up:**
  - *Label mix-up in `_resolve_and_embed`* — would be caught by
    `test_worker_gates_chunk_embed_on_chunk_index_only_never_message`'s exact-call assertion
    (`repo.index_dim_calls == [("acme", "Chunk", "embedding")]`), which fails outright on a swapped
    label.
  - *`_safe_embed_chunk`'s try/except removed* — `test_safe_embed_chunk_swallows_failure_logs_error_
    never_raises` (`test_background.py`) calls it with a worker whose `embed_chunk` always raises;
    removing the try/except means the test itself raises `RuntimeError` instead of passing.
  - *`search_chunks`'s `ORDER BY` flipped to DESC* — `test_search_chunks_ranks_by_cosine_distance_asc`
    asserts `scores == sorted(scores)` (ascending by `sorted`'s default) against three seeded chunks
    at three different cosine distances from the query vector; a DESC flip fails this directly.
  - *`set_chunk_embedding`'s dimension check dropped* — `test_set_chunk_embedding_rejects_wrong_
    dimension_loudly` asserts `pytest.raises(EmbeddingDimensionError)` on a wrong-length vector;
    removing the check means the write silently succeeds and the test fails on the missing raise.
  - *`search_documents` skips the query-embed step* — `test_search_documents_embeds_the_query_then_
    searches_chunks` asserts `models.embedded == ["hello"]`, which is empty if the embed call is
    skipped (and the mock gateway's stub vector would also then be undefined input to
    `search_chunks`, likely a `TypeError` before the assertion even runs).
  - *Chunk-scheduling loop removed from `api.py`/`mcp.py`* — `test_ingesting_a_document_schedules_
    every_chunk_for_embedding` (REST) and `test_ingest_document_schedules_every_chunk_for_embedding`
    (MCP) both assert `len(worker.chunk_calls/calls) == posted["chunkCount"]` against a
    multi-paragraph (non-trivial `chunkCount`) document — a removed loop leaves the count at 0 and
    fails directly.
- **Scope discipline confirmed.** `git diff --stat` matches the brief's file list exactly (7 server
  files + 7 test files + 3 docs); `document-ingestion.md`/`-graph.md`/`-ml.md`/`BACKLOG.md` are
  untouched, matching `HISTORY.md`'s own disclosure; `git diff -- scripts/bootstrap_schema.sh` is
  empty, confirming the "no new DDL" claim (`Chunk.embedding`'s vector index predates this stage).
- **Layering/conventions fit is native.** `repository.py` still owns 100% of the new Cypher 1:1 with
  `QUERIES.md`; `services.py` owns the one new invariant (`SearchNotAvailableError` gating) and the
  query→vector translation; `api.py`/`mcp.py` stay thin, differing only in how each transport
  schedules background work — exactly the Stage 1 precedent this stage extends, not a new pattern.

### What's solid (beyond the verified claims above)

- The two genuinely new implementation judgment calls the implementer self-disclosed (`search_chunks`
  omitting Entity co-occurrence expansion; the internal `list_document_chunks` seam not named in the
  plan's Stage 2 file list) are both the right call, correctly reasoned, and correctly disclosed
  rather than silently introduced — same posture Pass 1 credited Stage 1's `get_document` projection
  deviation for.
- Test coverage genuinely exercises new behavior, not just new code paths — e.g.
  `test_worker_caches_message_and_chunk_index_dimensions_independently` pins a real, easy-to-regress
  interaction (two independent per-label caches on one worker instance) that a shallower "it embeds a
  chunk" test would miss.

### Open questions

- None new. Pass 1's open question (whether the whitespace-only-document degenerate case needs a
  product decision) is resolved as of U11 (fixed) and is out of scope for this pass.

---

## Pass 3 (2026-08-24) — Stage 3 diff-scoped code gate

**Scope.** Diff-scoped code gate against the (then-)uncommitted working-tree changes implementing
Stage 3 (Extraction, FR-7a) of `docs/plans/document-ingestion.md`. A **new** diff, layered on top of
Stage 1+2's already-gated and committed changes (`167eeac`, `0404dcf`) — not a re-review of Pass 1/2's
findings. Baseline: the locked plan (§3.1, §3.3, §4 Stage 3), `docs/plans/document-ingestion-ml.md`
§3.1/§3.2/§3.3 (extraction technique/taxonomy/schema-validation/stub-repair, plan-gated and approved),
`docs/plans/document-ingestion-graph.md` §1.8/§2.2/§2.3/§4/§5 (exact Cypher and DDL, live-verified
there — including the post-review note that the eventual fusion call site is
`create_entity_with_auto_match`, confirmed to be Stage 4's concern, not this diff's: Stage 3's own file
list only names plain `create_entity`, and `ingestion.py`'s own docstring is explicit — "Do not add any
matching/dedup logic here"), and `falkor-chat/AGENTS.md`'s conventions (parameterized Cypher,
`{label}Id` index+constraint, opaque-string `RELATES_TO.label`). Files reviewed: new
`server/falkorchat/{extraction.py, ingestion.py}`; modified `server/falkorchat/{repository.py,
background.py, api.py, mcp.py, app.py}`; `config/models.json` + `server/tests/data/models.json`;
`scripts/bootstrap_schema.sh`; new `server/tests/{test_extraction.py, test_ingestion.py}`; extended
`server/tests/{test_background.py, test_repository.py, test_api.py, test_mcp.py}`; `docs/QUERIES.md`
§14.5; `docs/HISTORY.md`'s new Stage 3 entry. `docs/plans/document-ingestion-coordination.md`'s own
diff was out of scope per the brief (the coordinator's ledger). Every claim below — Cypher fidelity,
test counts, mutation-test spot-checks, the two new findings — was independently verified in this
session (code read, tests read and in two cases executed directly against `extraction.extract`/
`IngestionPipeline`, full suites re-run), not inherited from the implementer's self-report.

**CPG:** considered, not relevant — `cpg_falkorchat` is stale (built 2026-08-17T00:40:42Z, no
`sourceCommit`, 7+ commits behind including all three ingestion stages, per the brief's own freshness
note) and this diff is small and self-contained over freshly-read current files; read the files
directly instead, as directed.

**Verdict: approve with suggestions.** No blockers. Two MAJOR findings — both real, evidence-backed
correctness/robustness gaps, neither a build-breaker and both cheap to fix — plus the mutation-test
claims all independently confirmed and one escalated-severity note on an already-known, deferred
concern. Everything else (Cypher fidelity, shared-normalizer discipline, enum coercion, stage-boundary
discipline, scheduling independence, docs accuracy) is clean.

### Findings

**MAJOR — the entity cap can silently truncate away a stub-repaired entity a relationship depends on,
silently dropping the relationship fact — the exact failure FR-6/the ML note built stub-repair to
prevent.**

`extraction.py:212-214`:

```python
relationships = relationships[:MAX_RELATIONSHIPS_PER_CHUNK]
entities = _repair_stub_entities(entities, relationships)
entities = entities[:MAX_ENTITIES_PER_CHUNK]
```

`_repair_stub_entities` (`extraction.py:158-176`) appends synthesized stubs to the **end** of the
entities list. When the model's raw entity list already has `MAX_ENTITIES_PER_CHUNK` (20) items and a
relationship references a 21st name absent from that list, the stub for that name is appended as item
#21 — and the very next line, `entities[:MAX_ENTITIES_PER_CHUNK]`, slices it back off. `extract()`
then returns an `ExtractionResult` that is internally inconsistent: `result.relationships` still
carries the fact, but `result.entities` no longer contains its endpoint. Downstream,
`IngestionPipeline.extract_chunk` (`ingestion.py:112-120`) builds `entity_ids_by_name` only from
`result.entities`, fails to resolve the truncated endpoint, and silently `continue`s past the
relationship — never writing it. Verified by direct execution, not just reading:

```python
entities = [{"name": f"e{i}", "type": "Other"} for i in range(MAX_ENTITIES_PER_CHUNK)]
relationships = [{"subject": "e0", "predicate": "relatesTo", "object": "StubOnly"}]
result = extract(chunk_text, LLM(json.dumps({"entities": entities, "relationships": relationships})))
# result.entities: 20 items, "StubOnly" NOT among them
# result.relationships: still contains the "relatesTo StubOnly" fact
```

and end-to-end through `IngestionPipeline` with a `SpyRepo`: `repo.relationships == []` — the fact
is written nowhere, with no error, no log line, nothing to indicate it happened. This directly
contradicts the ML note's own stated rationale for building stub-repair in the first place
(`document-ingestion-ml.md` §3.2: *"this is cheap, deterministic, and avoids silently dropping a
relationship fact (the costlier failure — FR-6 is about not losing facts) over a model's bookkeeping
lapse"*) — the mechanism exists specifically to prevent this outcome, and the cap's ordering (entities
truncated **after** repair, by design per the module's own docstring at `extraction.py:48-54`, "the
cap's purpose is bounding the actual Entity nodes... written per chunk... the post-repair count")
reopens exactly the gap it was meant to close, for any chunk dense enough to hit the entity cap. This
isn't a remote corner case: the cap's whole purpose is bounding *dense* chunks (plan §3.3, §6 —
"RAM/supernode risk"), and dense/structured content (CSV rows, JSON blobs — FR-1's own named formats)
is exactly the shape most likely to both hit 20 raw entities and reference one more by name in a
relationship. No test in `test_extraction.py` covers this interaction — the two closest tests
(`test_extract_truncates_entities_at_the_cap_after_repair`, entities-only, no relationships;
`test_extract_a_relationship_dropped_by_the_cap_gets_no_stub_entity`, tests the *opposite* direction —
a relationship dropped by the relationship cap correctly gets no stub) don't exercise "entities at cap
+ a relationship needing a not-yet-listed stub."

**Suggested fix:** don't let the entity truncation cut into stub-repair's own output. The cleanest
shape: truncate the *raw* entities to `MAX_ENTITIES_PER_CHUNK` **before** calling
`_repair_stub_entities`, then let stub-repair add on top, uncapped (or capped separately, e.g. one
stub per truncated relationship, which is already implicitly bounded by `MAX_RELATIONSHIPS_PER_CHUNK`
— so the true worst case becomes `MAX_ENTITIES_PER_CHUNK + 2 × MAX_RELATIONSHIPS_PER_CHUNK`, still a
small, RAM-safe bound, not the unbounded raw model output the cap exists to guard against). This
preserves the cap's actual purpose (bounding hallucinated/runaway raw entity counts) without
re-orphaning a relationship stub-repair just created. At minimum, add the missing test case so this
doesn't regress silently even if the coordinator decides the current behavior is an acceptable v1
trade-off (which would need a documented decision — right now it's an unacknowledged gap, not a
considered trade-off).

**MAJOR — Stage 3 doubles the already-flagged MCP per-chunk thread fan-out (Pass 2's deferred MINOR),
and the doubling introduces a new, uncaught failure mode, not just a bigger number.**

Pass 2 flagged (MINOR, deferred to Stage 6) that `mcp.py`'s `ingest_document` spawns one raw
`threading.Thread` per chunk for embedding — up to ~500-600 threads for a max-size document
(`MAX_DOCUMENT_CHARS = 500_000` at `schemas.py:54`, ~1000-char chunks minus 150-char overlap ≈ 850
effective chars/chunk). Confirmed by reading `mcp.py:243-256`: this diff schedules `_safe_extract`
**inside the same per-chunk loop**, as a second, independent `_schedule()` call alongside
`_safe_embed_chunk` — when both `_embed_worker` and `_ingestion_pipeline` are wired (the production
default, per `app.py`'s `_build_default_app`), a single `ingest_document` MCP call now spawns on the
order of **1,000-1,200 raw OS threads**, sequentially, synchronously, inside the tool handler, before
it returns — double Pass 2's own citation, in the very next stage, with no mitigation added. Neither
`HISTORY.md`'s new entry nor `QUERIES.md` §14.5 mentions the amplification (both restate the
"scheduled independently, alongside, never chained" framing without a resource-usage note).

Beyond the raw multiplier, I want to flag something Pass 2 didn't have occasion to check because only
one `_schedule()` call existed per chunk then: **the per-chunk scheduling loop
(`mcp.py:244-256`) has no try/except around `_schedule()` itself.** `_default_schedule`
(`mcp.py:78`) is a bare `threading.Thread(target=fn, args=args, daemon=True).start()` — if thread
creation itself fails (`RuntimeError: can't start new thread`, a real OS-level failure once a
process's thread count approaches a `ulimit -u` / cgroup `pids.max` ceiling — commonly in the low
thousands in constrained/containerized environments, and now within a small constant factor of a
single call's fan-out), that exception propagates straight out of the `ingest_document` tool handler
— unhandled, unlike every `_safe_*` function it's scheduling (which are all try/except-wrapped by
design). This is a genuinely new failure mode this diff introduces relative to what Pass 2 examined:
not just "more threads, same accepted trade-off," but "the tool call itself can now raise on resource
exhaustion where before it couldn't" (the underlying `Document`/`Chunk`s are already committed by this
point, so no data is lost, but the caller gets an unhandled 500-equivalent on an otherwise-successful
ingest). Independent judgment, as asked: the doubling is not a mechanical non-issue — it moves this
from "an accepted trade-off, exercised at a scale within normal OS headroom" to "measurably closer to
a real resource ceiling, with a new caller-visible failure mode attached," in the same commit that was
warned the trade-off was already at its edge. I'd still call this MAJOR rather than a blocker (it's
lab-scale, daemon threads that do bounded I/O and exit, and REST's path is unaffected — `api.py`'s
`BackgroundTasks.add_task` stays a cheap list append), but I don't think it should keep sliding to
Stage 6 unexamined a second time. **Suggested improvement**, in order of cost: (a) cheapest — wrap the
per-chunk `_schedule()` calls in a try/except that logs and continues rather than lets a thread-start
failure abort the whole loop (and the tool call) partway through; (b) better — batch each transport's
fan-out into a small fixed pool (e.g. one thread that iterates the chunk list internally per job type,
or a bounded `ThreadPoolExecutor`) instead of one-thread-per-(chunk × job); (c) at minimum, update
`_default_schedule`'s docstring and `QUERIES.md` §14.5/§14.4 to state the doubled number explicitly,
so the "accepted for M1 lab-scale" reasoning isn't silently reasoning about half the real fan-out.

### Verified claims (evidence, not trust)

- **Cypher fidelity — exact match against `document-ingestion-graph.md` §2.2/§2.3/§5.**
  `repository.create_entity`/`link_chunk_about_entity`/`create_entity_relationship`
  (`repository.py:1112-1194`) are byte-identical (up to inert string concatenation) to the graph
  doc's designed shapes — property names, parameterization, no interpolation, all three plain
  `CREATE`s with no guard-and-status-row contract (correctly, per the graph doc's own reasoning: both
  endpoints are freshly minted by the same pipeline run). `bootstrap_schema.sh`'s two new DDL lines
  (`Entity.name` fulltext, `Entity.nameNormalized` plain RANGE index, no constraint) match §2.3/§5
  exactly; `Entity.entityId`'s index+constraint (lines 48-49/65-66, 103-104/199-200) predate this
  diff, confirmed by `git diff` scoping — the brief's claim holds.
- **Shared-normalizer discipline — genuinely one function, not two.** `extraction.normalize_name`
  (`extraction.py:61-72`) is the only name-normalization logic in the diff; `repository.create_entity`
  takes `name_normalized` as a caller-computed parameter rather than computing it itself, and the only
  caller (`IngestionPipeline.extract_chunk`, `ingestion.py:100,110,114,117`) calls
  `extraction.normalize_name` for every one of its own uses, both the write and the in-memory
  relationship-endpoint match. Grepped the whole `falkorchat/` package for any other normalization
  helper touching entity names (`_normalize_opaque`, `_normalize_base_url`, `_normalize_tool_call` are
  all unrelated domains — config/guard-comparison, base-URL, tool-call parsing) — no second,
  independently-written normalizer exists anywhere.
- **Closed-enum coercion is correctly scoped — coerces `type`, drops on missing/non-string `name`.**
  `_coerce_entities` (`extraction.py:111-134`): a missing/blank/non-string `name` drops the whole item
  (`continue`, line 128-129); an out-of-enum or missing `type` coerces to `"Other"` (lines 130-132),
  never rejects the item. Matches the ML note's F1 distinction exactly ("a wrong type is recoverable
  via fusion review; a missing key is not something to guess at" — applied per-item here for `name`
  specifically, not conflated with the top-level-shape rejection `extract()` does separately for a
  non-list `entities`/`relationships`). Confirmed by both reading and
  `test_extract_skips_an_entity_item_missing_a_name_but_keeps_the_rest` /
  `test_extract_coerces_an_out_of_enum_type_to_other` /
  `test_extract_coerces_a_missing_type_to_other`.
- **Stub-repair correctness and truncation order — matches the docstring's claimed order exactly**
  (relationships truncated first at `extraction.py:212`, off the raw parsed list; stub-repair runs
  next at :213 against the truncated relationships; entities truncated last at :214). This order is
  real, not just claimed — but see the MAJOR finding above: matching the claimed order is exactly
  what produces the silent-fact-loss bug, since "entities truncated last" is what lets the truncation
  cut into stub-repair's own just-added output. `_repair_stub_entities` (`extraction.py:158-176`)
  itself is correct in isolation: it only synthesizes a stub for a `subject`/`object` name not already
  present (normalized compare via the shared helper), confirmed via
  `test_extract_stub_repair_matches_case_and_whitespace_insensitively` (an already-declared entity,
  differently cased/spaced, does NOT get a duplicate stub) and
  `test_extract_repairs_a_dangling_relationship_object_with_a_stub_entity`.
- **`RELATES_TO` never-deduplicated — confirmed a plain `CREATE`, and the test correctly uses
  `count(r)`, not `count(*)`.** `create_entity_relationship` (`repository.py:1182-1194`) is a bare
  `CREATE`, no `MERGE` anywhere in its Cypher. `test_create_entity_relationship_is_never_deduplicated`
  (`test_repository.py:918-944`) calls it twice with identical arguments and asserts
  `MATCH (...)-[r:RELATES_TO]->(...) RETURN count(r)` (bound relationship variable) `== [[2]]` —
  correctly avoiding the live-verified FalkorDB `count(*)` parallel-edge-undercounting quirk
  (per the brief, already captured for `graph-dba`'s knowledge base; not re-verified here, only
  confirmed this test's query shape accounts for it). This is a real, live integration test against
  `ws:test`, not a mock.
- **`_safe_extract` mirrors `_safe_embed`/`_safe_embed_chunk` exactly.** `background.py:53-74`: same
  try/except-log-never-raise shape, same `# noqa: BLE001` comment, same `_log.exception(...)` call
  pattern. `test_safe_extract_swallows_failure_logs_error_never_raises`
  (`test_background.py:147-158`) calls it with a pipeline whose `extract_chunk` always raises and
  asserts no exception propagates, one ERROR log record, `exc_info` populated — a real regression
  guard, not a restatement.
- **No fusion logic anywhere in this diff — the Stage 3/4 boundary holds.** `IngestionPipeline`
  (`ingestion.py`) and the three new `repository.py` methods contain no lookup-before-create,
  no `MATCH ... nameNormalized`, no `SAME_AS` reference outside doc comments describing the future
  stage. `test_extract_chunk_never_looks_up_or_reuses_an_existing_entity`
  (`test_ingestion.py:184-195`) is a real regression guard for this, not just an assertion of intent:
  `SpyRepo` has no lookup method at all, so a fusion lookup attempt would fail with `AttributeError`
  before the test's own assertions even run. `git diff -- server/falkorchat/repository.py | grep
  -i "SAME_AS\|match"` confirms the only two hits are doc-comment mentions of a *future* stage, not
  code.
- **Independent extraction/embedding scheduling, both transports, synchronous return path unaffected.**
  `api.py:176-192` and `mcp.py:243-256`: both schedule `_safe_embed_chunk` and `_safe_extract` as two
  separate calls per chunk inside the same loop, each independently gated on its own worker/pipeline
  being non-`None` — neither chains off the other. `services.ingest_document`'s receipt is built and
  returned (REST) / the receipt dict is returned (MCP) before either scheduling loop runs, so
  extraction genuinely cannot affect the synchronous return path — confirmed by reading the call
  order, not inferring it.
- **`config/models.json`'s `extraction` kind resolves through the existing generic path — no
  special-casing needed, and the deliberately-skipped workspace-override wiring degrades gracefully.**
  `Overlay.default_for(kind)` (`modelconfig.py:441-443`) reads `self.defaults.get(kind)` with no
  membership check against any closed set — `ModelGateway.resolve()`/`.llm()` (`modelconfig.py:
  729-784`) never special-cases a kind string. The one place a closed set (`KINDS`,
  `frozenset[str]` at `modelconfig.py:87`, still `{"agent", "step", "embedding", "guard"}` —
  untouched by this diff, confirmed via `git diff --stat -- server/falkorchat/modelconfig.py`, empty)
  is consulted is `_KIND_TO_OVERRIDE_KEY.get(kind)` inside `_workspace_override_ref`
  (`modelconfig.py:708-727`) and `GraphWorkspaceOverrides.get` (`modelconfig.py:272-277`) — both
  return `None`/no-override on a miss rather than raising, so `resolve("extraction", ws=...)` simply
  never finds a workspace override and falls straight through to `self._overlay.default_for(kind)`,
  which now resolves via the newly-added `config/models.json` entry. Confirmed by reading the fall-
  through logic directly, not inferring it from the absence of an exception. `KINDS`'s own docstring
  ("The four closed consumer kinds... adding a fifth means adding its own override property") is now
  slightly stale in spirit (there are 5 defined kinds in `defaults`/`timeouts` as of this diff) but
  `KINDS` itself is purely a test-parametrization/documentation constant — `grep`-confirmed it's
  referenced nowhere at runtime as a validation gate — so this is a documentation nit, not a defect;
  noted below, not raised as a finding.
- **`docs/QUERIES.md` §14.5 and `docs/HISTORY.md`'s new entry are accurate against the diff.** Read
  both against the actual code: the Cypher blocks match `repository.py` verbatim; the described
  file/test list, scope boundary, and DDL-inertness claims ("neither queried yet") are all correct.
  Both restored `reference` (bootstrap + seed + verify, confirmed back in sync) after the test runs
  this pass also had to redo.

### Mutation-test claims — spot-checked against actual assertions (all six confirmed, plus item 2's
own claimed detection gap independently reproduced in spirit)

- **Stub-repair removed** → caught by 3 tests, confirmed:
  `test_extract_repairs_a_dangling_relationship_object_with_a_stub_entity` asserts the stub `Acme
  Corp` entity exists with `type == "Other"`; `test_extract_never_silently_drops_a_relationship_fact_
  for_a_missing_entity` asserts both the relationship AND both endpoint entities are present;
  `test_extract_chunk_writes_relationships_for_stub_repaired_entities_too` (integration, via
  `IngestionPipeline`) asserts a written `RELATES_TO`. Removing `_repair_stub_entities`'s call would
  fail all three directly (the stub entity simply wouldn't exist).
- **Enum-coercion removed** → caught by 2 tests: `test_extract_coerces_an_out_of_enum_type_to_other`
  and `test_extract_coerces_a_missing_type_to_other` both assert `type == "Other"` on the returned
  entity; without coercion the item would either keep the invalid raw type string or (depending on the
  mutation's exact shape) be dropped — either way these exact-equality assertions fail.
- **`_safe_extract`'s try/except removed** → caught by
  `test_safe_extract_swallows_failure_logs_error_never_raises`: calls `_safe_extract` with a pipeline
  whose `extract_chunk` always raises `RuntimeError`; with the try/except gone, the `RuntimeError`
  propagates out of the test body itself (no `pytest.raises` wraps the call), failing the test with an
  uncaught exception rather than an assertion failure — genuinely fatal to the mutant, not a shallow
  check.
- **`RELATES_TO`'s `CREATE` flipped to a guarded `MERGE`** → caught by
  `test_create_entity_relationship_is_never_deduplicated`: two identical calls, asserts `count(r) ==
  2`; a `MERGE` would produce `count(r) == 1`, failing the equality assertion directly. Real, live
  `ws:test` integration test, not a mock.
- **Shared normalizer swapped for an independently-written one** → the report's own account (initially
  NOT caught, two new tests added) checks out under direct reasoning: a naive `.lower()`-only
  normalizer (no `.strip()`, no whitespace-collapse) applied to `"  Acme   CORP  "` would produce
  `"  acme   corp  "` ≠ `normalize_name`'s `"acme corp"` —
  `test_extract_chunk_writes_name_normalized_via_the_shared_helper`'s exact-equality assertion against
  `normalize_name(...)` directly would fail. `test_extract_chunk_resolves_a_relationship_across_
  internal_whitespace_variance` independently fails the same mutant a second way: without whitespace
  collapse, `"Acme  Corp"` (double space, from the entity list) and `"Acme Corp"` (single space, from
  the relationship's `object`) would normalize to different strings, the relationship-endpoint lookup
  would miss, and `len(repo.relationships) == 1` would fail. Both are genuine, not restatements of the
  mutation's own description.
- **Extraction-scheduling loop removed from `api.py`/`mcp.py`** → caught by
  `test_ingesting_a_document_schedules_every_chunk_for_extraction` (REST) and
  `test_ingest_document_schedules_every_chunk_for_extraction` (MCP), both asserting
  `len(pipeline.calls) == chunkCount` against a real multi-paragraph document; a removed loop leaves
  `pipeline.calls` empty while `chunkCount > 0`, failing directly.

### What's solid (beyond the verified claims above)

- **Test counts match exactly, independently re-run.** `git diff` line counts for `^def test_` per
  file (`test_extraction.py` +24, `test_ingestion.py` +11, `test_background.py` +2,
  `test_repository.py` +7, `test_api.py` +3, `test_mcp.py` +3 — sums to the claimed +50) match
  `HISTORY.md`'s claim exactly. Re-ran the full offline suite: `1647 passed, 3 deselected` (matches
  1597→1647 claimed). Re-ran `./scripts/test_queries.sh`: `320/320` (unchanged, as expected — no new
  Cypher shape entered that suite, same precedent Stage 2 set).
- **Scope discipline is clean.** `git diff --stat` matches the brief's file list exactly (2 new server
  modules, 5 modified server modules, 2 config files, 6 test files [2 new, 4 extended], 1 script,
  2 docs — `document-ingestion.md`/`-graph.md`/`-ml.md`/`BACKLOG.md` untouched, matching `HISTORY.md`'s
  own disclosure). `modelconfig.py` genuinely untouched (confirmed via empty `git diff --stat`),
  matching the implementer's self-reported "workspace-override wiring left out of scope" claim.
- **Layering fit is native.** `IngestionPipeline` is a clean structural peer of `EmbeddingWorker`
  (same `llm=`/`models=` FR-4 sugar, same `id_gen`/`clock` injection seam,
  `repo` accessed directly rather than through `Services` — reasoned explicitly in the module
  docstring against the one real alternative, and the reasoning holds); `background._safe_extract`,
  `api.py`'s and `mcp.py`'s scheduling additions all mirror the Stage 2 precedent exactly, not a new
  pattern.
- **The prompt design and closed-enum framing match the ML note's guidance precisely** — no
  `confidence` field requested, one combined call not split, the explicit empty-result instruction,
  the seven-type enum given verbatim with one-line definitions — all present in `_SYSTEM_PROMPT`
  (`extraction.py:75-93`).

### Open questions

- **The entity-cap/stub-repair interaction (MAJOR finding above)** needs a decision either way: fix
  the truncation order, or explicitly document the trade-off (and add the missing test) if the
  coordinator decides silently dropping a relationship fact in this specific cap-collision case is an
  acceptable v1 gap. Right now it's neither — it's an unacknowledged gap that contradicts the ML
  note's own stated rationale for the mechanism it undermines.
- **The doubled MCP thread fan-out (second MAJOR finding)** — my independent read is that continuing
  to defer this to Stage 6 is riskier now than it was after Pass 2, given the new uncaught-exception
  path. Worth a quick coordinator call on whether the cheap mitigation (wrap `_schedule()` in
  try/except so a thread-start failure can't abort an otherwise-successful `ingest_document` call)
  should land now rather than wait three more stages, even if the fuller fix (bounded pool/batching)
  still waits for Stage 6.
- **`modelconfig.KINDS`'s "four closed consumer kinds" docstring** is now stale in spirit (5 kinds are
  defined in `config/models.json`'s `defaults`/`timeouts`) but not touched by this diff and not
  functionally load-bearing (confirmed nothing at runtime checks membership against it) — worth a
  one-line note or a `KINDS` update whenever `extraction`'s workspace-override wiring is eventually
  built (Stage 4+ or later), not urgent enough to hold this gate on.

### Re-gate (2026-08-24) — both MAJORs fixed, independently re-verified

The `coder` fixed both Pass 3 MAJOR findings. Re-verified independently (not relying on the
coordinator's own pre-check, per this codebase's standing practice that the gate closes a finding, not
a self-report): read both diffs in full, re-ran my own original reproduction against the fixed code,
mutation-spot-checked both new regression tests, re-ran the full offline and query suites myself, and
confirmed no other file changed beyond the two fixes.

**MAJOR 1 (entity-cap/stub-repair truncation order) — confirmed fixed.** `extraction.py:218-224` now
caps raw entities (`entities[:MAX_ENTITIES_PER_CHUNK]`) **before** `_repair_stub_entities` runs, so a
stub the repair pass adds can no longer be sliced back off by a truncation that used to run after it;
the module docstring (`extraction.py:48-60`) was updated to match, correctly naming the new worst-case
bound (`MAX_ENTITIES_PER_CHUNK + 2 × MAX_RELATIONSHIPS_PER_CHUNK`). Re-ran my own Pass 3 reproduction
script verbatim against the fixed code — same input (20 raw entities at the cap, one relationship
referencing a 21st, not-yet-listed name) — and it now behaves correctly: `extract()` returns 21
entities including the stub, and `IngestionPipeline.extract_chunk` against a `SpyRepo` now writes the
relationship (`repo.relationships == [('id0', 'id20', 'relatesTo')]`, empty before the fix). The new
regression test, `test_extract_stub_repair_is_not_truncated_away_by_the_entity_cap`
(`test_extraction.py:311-334`), reproduces the exact same scenario and asserts the fact, the stub's
presence, and the exact post-repair count (`MAX_ENTITIES_PER_CHUNK + 1`) — genuinely mine, not a
paraphrase. **Mutation-spot-checked**: reconstructed the pre-fix ordering in memory (repair-then-cap,
without touching `extraction.py` on disk) and ran the new test's own assertions against it — the
stub-presence and count assertions both fail on the mutant (the relationship-survives assertion alone
does not, since truncation only removes the entity, not the already-separately-capped relationship
list — but the test fails outright on the very next assertion, so the mutant is genuinely caught).

**MAJOR 2 (doubled MCP thread fan-out, no failure isolation) — confirmed fixed, disposition matches
what was asked.** `mcp.py`'s `_default_schedule` (`mcp.py:52-113`) now wraps
`threading.Thread(target=fn, args=args, daemon=True).start()` in `try/except Exception`, logging via
`_log.exception(...)` and swallowing rather than propagating — the identical shape every other
`_safe_*` isolation point in this codebase already uses. The docstring now states the doubled fan-out
explicitly (~1,000-1,200 threads for a max-size document, up from Pass 2's ~500-600) and the fix's
scope (thread-start failures caught; the fuller bounded-pool redesign still deferred to Stage 6, matching
the coordinator's explicit direction and Pass 2's original disposition — I did not treat "fix (b) not
built" as a gap, since that was this pass's own stated middle-ground suggestion, not a hard requirement).
`docs/QUERIES.md` §14.5 gained a "Resource note" stating the same numbers and disposition — read
against the code, it's accurate. **Mutation-spot-checked** `test_default_schedule_swallows_a_thread_
start_failure_and_logs` (`test_mcp.py:305-330`): reconstructed the pre-fix (bare, unwrapped) call
shape in memory against the test's own monkeypatched `_FailingThread`, confirmed it raises
`RuntimeError` uncaught — since the real test has no `pytest.raises` around the call (only a `# must
not raise` comment), that would fail the test directly, not just miss an assertion. The second new
test, `test_default_schedule_still_runs_the_job_on_the_happy_path` (`test_mcp.py:333-352`), is a
real-thread (not monkeypatched) regression guard confirming the try/except doesn't also swallow the
happy path — joins the spawned thread before asserting, so it isn't racing the background call.

**No new issues from either fix.** `git status` shows exactly the two touched source files
(`extraction.py`, `mcp.py`), their two test files, and the `QUERIES.md` note — `repository.py`,
`ingestion.py`, `background.py`, `api.py`, `app.py` are untouched since Pass 3, confirming the fix
stayed scoped to the two findings with no drive-by changes elsewhere.

**Suites re-run myself, both match the coordinator's counts exactly.** Offline: `1650 passed, 3
deselected` (1647→1650, the three new regression tests: 1 in `test_extraction.py`, 2 in `test_mcp.py`).
`./scripts/test_queries.sh`: `320/320`, unchanged. Re-seeded `reference` after the query suite's
teardown wipe (`bootstrap_schema.sh acme` → `seed_demo.sh acme` → `seed_workflows.sh acme` →
`verify_workflows.sh acme`, confirmed back in sync).

**Updated verdict: approve.** Both MAJOR findings from Pass 3 are closed — genuinely fixed, not just
present, confirmed by direct re-execution of my own reproductions and by mutation-spot-checking both
new regression tests against the pre-fix behavior. No blockers, no majors, no new issues. The two
open questions above that were about *whether/when* to fix are resolved (both landed now); the third
(`modelconfig.KINDS`'s stale docstring) remains open, unchanged, not urgent.
