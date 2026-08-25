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

---

## Pass 4 (2026-08-24) — Stage 4 diff-scoped code gate

**Scope.** Diff-scoped code gate against the (then-)uncommitted working-tree changes implementing
Stage 4 (Fusion, FR-6/7/8/9/10, OQ-1/2/3) of `docs/plans/document-ingestion.md`. A **new** diff,
layered on top of Stages 1-3's already-gated changes — not a re-review of Pass 1-3's findings.
Baseline: the locked plan (§4 Stage 4, §3.4 "Concurrency note", §5 test-strategy table's AC-1..AC-4/
AC-7/AC-8 rows plus the "Additional, non-AC-mapped test coverage" section) and
`docs/plans/document-ingestion-graph.md` §1.5-§1.8 (final `SAME_AS` schema, exact Cypher for
`create_or_reopen_match`/`confirm_match`/`reject_match`/`recheck_match`/`list_pending_matches`/
`list_matches`/`create_entity_with_auto_match`, including the MAJOR-finding two-query-string
`list_matches` fix and the live-verified atomic-ordering proof). Files reviewed: new
`server/falkorchat/fusion.py`; modified `server/falkorchat/{repository.py, ingestion.py,
background.py, services.py, mcp.py, api.py, app.py}`; `scripts/bootstrap_schema.sh`; new
`server/tests/test_fusion.py`; extended `server/tests/{test_repository.py, test_ingestion.py,
test_background.py, test_services.py, test_api.py, test_mcp.py}`. Per the brief, `docs/HISTORY.md`,
`docs/DESIGN.md`, `docs/SERVER.md`, `docs/test-reports/capacity-report.md`, and the two `claude/`
agent-doc files were unrelated concurrent work in the shared tree and are explicitly out of scope —
not reviewed, not commented on below. Every claim below (Cypher fidelity, the self-match-filter
necessity, the mutation-test spot-check, the test-count arithmetic, the concurrency regression test,
the shipped-Cypher quirk-shape check) was independently verified in this session — code read line by
line against both plan documents, the full offline suite and `./scripts/test_queries.sh` re-run
myself, the concurrency test re-run 5x, one mutation planted/observed/reverted live, and one live
FalkorDB probe run to settle a factual question the brief posed rather than assumed — not inherited
from `coder`'s self-report.

**CPG:** considered, not relevant — this diff is net-new/modified application code in a fast-moving
stage of an active feature; reading the files directly (as this pass did, against two authoritative,
live-verified design documents) is more precise here than a CPG built at an earlier commit would be,
and no freshness claim for `cpg_falkorchat` was supplied in the brief to rely on instead.

**Verdict: approve with suggestions.** No blockers, no majors. Cypher fidelity is essentially
byte-identical to the graph note's live-verified shapes across all seven new repository methods —
correct unlabeled-endpoint discipline everywhere a `SAME_AS`-anchored query needs it, correct
`list_matches` two-branch (not null-guarded) structure, correct atomic ordering in
`create_entity_with_auto_match`. The self-match filter in `fuse_entity` is not just plausible but
live-confirmed necessary and correctly placed. The concurrency regression test is genuine (real
threads, a barrier, separate connections) and passed 5/5 independent re-runs. The mutation-testing
and test-count claims both check out under independent verification. Two MINOR findings, both
documentation/test-altitude gaps rather than correctness defects.

### Findings

**MINOR — no `docs/QUERIES.md` section for Stage 4's seven new repository methods, even though the
diff's own code comments reference one that doesn't exist.**

Every stage so far has added its Cypher to `docs/QUERIES.md` as the module's own "canonical query
library, verified against the live instance" (`falkor-chat/AGENTS.md`'s Key Documents table) —
Stage 1 got §14.1/§14.2, Stage 2 got §14.3/§14.4, Stage 3 got §14.5 (confirmed via `grep -n "^### 14\."
docs/QUERIES.md`). This diff's own section-marker comments (`repository.py:1196`, `services.py`,
`api.py`, `bootstrap_schema.sh`) all say `# ── §14.6 Entity fusion — SAME_AS …` — the code is written
as if `docs/QUERIES.md` §14.6 already exists, but `git diff --stat -- docs/QUERIES.md` is empty: no
such section was ever added. Every one of the seven new Cypher blocks (`create_entity_with_auto_match`,
`find_fuzzy_candidates`, `create_or_reopen_match`, `confirm_match`, `reject_match`, `recheck_match`,
`list_pending_matches`, `list_matches`) currently exists only in `repository.py` and in the graph note
— not in the document this codebase's own convention designates as the source of truth for verified
queries, and not reachable from a `§14.6` citation anyone follows off the code comments themselves.
Not a correctness defect (the Cypher is verified correct against the graph note either way, confirmed
above), but a real conformance gap against an established, three-stages-running convention, and a
dangling internal cross-reference. **Suggested fix:** add `docs/QUERIES.md` §14.6, mirroring §14.5's
shape — the seven query blocks plus a short "verified against the live instance" note, sourced
directly from `document-ingestion-graph.md` §1.6-§1.8 (already exact-shape-verified, so this is a
copy-and-cite job, not new verification work).

**MINOR — AC-8's specific test-strategy shape (two full `ingest_document` calls / background-
completion-aware) isn't demonstrated end-to-end; the underlying mechanism is thoroughly proven at a
lower altitude instead.**

Plan §5's test-strategy table specifies AC-8 as: *"`ingest_documents([doc_a, doc_b])` where both
mention the same entity; assert both processed and a `SAME_AS` edge links their extracted entities,
not just doc_a's — service integration, background-completion-aware (poll `Document.status` or run
the pipeline synchronously in the test)."* There is no batch `ingest_documents` entry point in this
codebase (confirmed: `grep -rn "def ingest_document" falkorchat/*.py` finds only the singular,
per-document `ingest_document` in `services.py`/`api.py`/`mcp.py`) — so the plan's literal scenario
name doesn't map onto a real call, and "two documents" in practice means two sequential/independent
`ingest_document`/`extract_chunk` calls sharing a workspace graph. That said, no test in this diff
exercises the *pipeline-level* two-call scenario the AC row is actually testing for — the cross-
document linking behavior is instead proven only at the repository primitive's own altitude:
`test_create_entity_with_auto_match_links_an_existing_candidate` (`test_repository.py:966`, two
sequential calls, same `(nameNormalized, type)`, asserts the link) and the concurrent variant
(`test_repository.py:1038`, two REAL concurrent calls). Both are genuine and sufficient to prove the
*mechanism* is correct — including under the exact race condition AC-8's own footnote calls out — but
neither goes through `IngestionPipeline.extract_chunk` twice (once per "document"), so nothing in this
diff demonstrates the *pipeline* wiring correctly produces a cross-document `SAME_AS` edge the way a
reader of the plan's test-strategy row would expect to find proven. **Suggested fix:** add one test —
either at `test_ingestion.py`'s altitude (call `pipeline.extract_chunk` twice with different
`document_id`s but an entity of the same normalized name/type, assert the `SpyRepo` recorded a link
call spanning both) or a real repository-backed integration test — closing the gap between what the
plan's AC-8 row names and what's actually exercised. Low severity: the primitive this would exercise
is already proven correct and race-safe; this is about closing an explicit, named test-strategy row's
literal shape, not about an unverified behavior.

### Verified claims (evidence, not trust)

- **Cypher fidelity — essentially byte-identical to `document-ingestion-graph.md` §1.6-§1.8 across
  all seven new methods.** Read `repository.py:1196-1497` line by line against the graph note's
  Cypher blocks: `create_entity_with_auto_match` (§1.8) matches the `OPTIONAL MATCH ... ORDER BY
  createdAt ASC LIMIT 1 ... CREATE ... FOREACH (CASE WHEN candidate IS NOT NULL ...)` shape exactly,
  including the no-reopen-branch simplification and the exact property set on the auto-created
  `SAME_AS` edge; `create_or_reopen_match` (§1.6) matches the guarded double-`FOREACH` shape exactly,
  including the id-anchored `MATCH (a:Entity {entityId: ...})`/`MATCH (b:Entity {entityId: ...})`
  labels (correct here — real per-node predicates, not the §1.4 bystander-label trap) alongside the
  unlabeled undirected `OPTIONAL MATCH (a)-[existing:SAME_AS]-(b)` lookup; `confirm_match`/
  `reject_match`/`recheck_match`/`list_pending_matches` (§1.7) all use bare `(a)-[r:SAME_AS {...}]-
  (b)`/`(a)-[r:SAME_AS {...}]->(b)`, never a bare-labeled endpoint, matching §1.4's planner-trap note
  exactly; `list_matches` (§1.7's post-review MAJOR fix) is genuinely **two separate query strings**
  branched in Python (`repository.py:1466-1489`), not a `WHERE $status IS NULL OR r.status = $status`
  null-guard — confirmed by reading the actual `if status is not None: ... else: ...` branch, not
  inferring it from a docstring claim. `find_fuzzy_candidates` (§2.3) matches the
  `db.idx.fulltext.queryNodes` + post-`YIELD` `WHERE candidate.type = $type` shape exactly. All eight
  Cypher strings are fully parameterized — no interpolation anywhere in the diff.
- **`bootstrap_schema.sh`'s new DDL matches §1.5/§5 exactly, index-before-constraint.** Two new
  relationship-scoped `CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.matchId | r.status)` lines land before
  the `UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1 matchId` constraint later in the same function
  (`scripts/bootstrap_schema.sh:115-123` vs. `:214-215`), same ordering discipline every other
  identity in this script already follows.
- **The self-match filter in `fuse_entity` is live-confirmed necessary, not just plausible.** Ran a
  direct probe against a live, properly-bootstrapped `ws:test` (`CREATE INDEX`/fulltext index applied
  first — the same fulltext lookup against an unindexed graph silently returns nothing, which is *not*
  the shipped condition since `bootstrap_schema.sh` always runs before real traffic): created one
  entity via `create_entity_with_auto_match`, then immediately called `find_fuzzy_candidates` against
  its own exact name. Result: the just-created entity **was** returned as the (only) fuzzy hit, score
  2.0 — confirming a same-connection write is synchronously visible to the next RediSearch fulltext
  query on this build, and that `fuse_entity`'s exclusion filter
  (`ingestion.py:186-189`: `[c for c in fusion.find_fuzzy_candidates(...) if c["entityId"] != entity_id]`)
  is load-bearing, not defensive-only. **Placement is correct**: the filter runs on `candidates`
  *before* `fusion.classify_fuzzy(candidates)` is called (`ingestion.py:189-190`) — filtering after
  would risk `classify_fuzzy` seeing a self-only candidate list and returning `'suggested'` when the
  correct classification (after removing the self-hit) is `'none'`; filtering before, as shipped,
  means `classify_fuzzy` only ever sees genuinely-other candidates. Also confirmed by direct test read:
  `test_extract_chunk_excludes_the_just_created_entity_from_its_own_fuzzy_candidates`
  (`test_ingestion.py:315-330`) scripts exactly this scenario (a `SpyRepo` fuzzy result containing only
  the entity's own id) and asserts `repo.reopen_calls == []`.
- **The atomic ordering guarantee holds — re-verified by independent re-execution, not just reading.**
  Re-ran `test_create_entity_with_auto_match_concurrent_calls_produce_exactly_one_edge`
  (`test_repository.py:1038-1085`) 5 times independently (real `threading.Thread`s, a
  `threading.Barrier(2)` forcing both to fire together, separate `db.connect()` connections per
  thread) — passed 5/5, `entity_count == 2`, exactly one `confirmed` `SAME_AS` edge each time, no
  errors from either thread. This is a genuine regression test for the plan-gate BLOCKER, not a
  sequential-calls-dressed-as-concurrent shape.
- **`_safe_fuse`'s inline-not-scheduled granularity is a reasoned, correctly-argued deviation, not an
  oversight.** Confirmed both sides of the claim: `background.py`'s own docstring
  (`background.py:77-99`) states the rationale (a fuzzy lookup can only run once the entity exists,
  and `api.py`/`mcp.py` schedule per-*chunk*, before any entity is known) and `grep -n "_safe_fuse"
  api.py mcp.py` returns nothing — `_safe_fuse` is called exactly once, from inside
  `IngestionPipeline.extract_chunk`'s own per-entity loop (`ingestion.py:135-141`), never scheduled
  separately by either transport's per-chunk loop the way `_safe_extract`/`_safe_embed_chunk` are.
  This gives per-ENTITY failure isolation, one level finer than `_safe_extract`'s per-chunk isolation,
  matching the plan file-list note's own framing of this as "the implementer's call on granularity."
- **The shipped, non-test Cypher has no instance of the undirected + inline-relationship-property-
  filter quirk shape `coder` reported to `kaizen_team`.** Grepped every `SAME_AS`-touching query in
  `repository.py` for an undirected pattern (`-[...]-`  with no arrow) carrying an inline property
  filter on the relationship variable itself. The only undirected pattern in the shipped code is
  `create_or_reopen_match`'s `OPTIONAL MATCH (a)-[existing:SAME_AS]-(b)` (`repository.py:1320`) —
  genuinely undirected, but the relationship variable carries **no** inline property filter (no
  `{status: ...}` or similar on `existing`); every query that *does* filter inline on a `SAME_AS`
  property (`{matchId: $matchId}`, `{status: 'pending'}`, `{status: $status}`) is directed (`->`),
  never undirected. So the exact dangerous shape doesn't appear in shipped code — confirmed, not
  assumed.
- **Mutation-testing spot-check (FR-8 type-filter) — reproduced independently.** Mutated
  `create_entity_with_auto_match`'s candidate `OPTIONAL MATCH` to drop `type: $type` from the pattern
  (leaving only `nameNormalized`), ran `test_create_entity_with_auto_match_requires_matching_type_too`
  alone — failed exactly as `coder` reported (`assert True is False` on `exactMatched is False`),
  then reverted the file and re-ran the same test to confirm it passes again on the restored code
  (`git diff --stat -- server/falkorchat/repository.py` showed zero changes after restore, confirming
  no residue). One mutation is the brief's stated minimum; this one exercises the exact-tier's second
  join key, the same shape `coder`'s report claims to have covered.
- **Test-count arithmetic checks out exactly, independently recomputed.** Re-ran the full offline
  suite: `1711 passed, 3 deselected` (matches `coder`'s claim exactly). Independently counted new
  `def test_` lines per file via `git diff`: `test_repository.py` +23, `test_ingestion.py` +7,
  `test_background.py` +2, `test_services.py` +10, `test_api.py` +8, `test_mcp.py` +5, plus the new
  `test_fusion.py`'s 7 — **sums to 62**, matching the "7 files summing to 62" claim exactly. The
  reported **net** of +61 (not +62) reconciles too: `test_ingestion.py` also **removes** one test,
  `test_extract_chunk_never_looks_up_or_reuses_an_existing_entity` (the Stage-3 "no fusion, ever"
  degenerate-case guard, correctly superseded now that Stage 4 does look up/reuse via fusion) — 62
  added, 1 removed, net +61, and the baseline this stage started from was Pass 3's own
  independently-confirmed `1650 passed, 3 deselected` (not the brief's illustrative "1640"), so
  1650+61 = 1711 ties out exactly. No arithmetic issue once the correct baseline is used.
- **`./scripts/test_queries.sh`: 320/320, unchanged** (no new query shape entered that suite —
  consistent with Stage 2/3's own precedent that this suite doesn't grow for repository-layer-only
  Cypher additions). Both this run and the earlier full-suite `pytest` run wiped `reference` at
  teardown, per this codebase's known hazard (`falkor-chat/AGENTS.md`); re-seeded
  (`bootstrap_schema.sh acme` → `seed_demo.sh acme` → `seed_workflows.sh acme` →
  `verify_workflows.sh acme`) after each, confirmed back in sync both times.

### What's solid (beyond the verified claims above)

- **Layering and wiring are clean and symmetric.** `services.py`'s five new passthroughs
  (`confirm_match`/`reject_match`/`recheck_match`/`list_pending_matches`/`list_matches`) are
  correctly thin, `MatchNotFoundError` is correctly wired into `app.py`'s 404 set, and both
  `api.py`/`mcp.py` expose the identical five-operation surface with matching test coverage on both
  transports (`test_api.py`/`test_mcp.py` both seed via the repository directly, correctly noting
  match creation isn't itself a REST/MCP-reachable write).
- **`recheck_match`'s can't-distinguish-unknown-from-not-rejected posture is honestly propagated
  end-to-end**, not smoothed over at a higher layer — `Repository.recheck_match` returns `None` for
  both cases, `Services.recheck_match` documents and preserves that (does not raise), and both
  `api.py`/`mcp.py` return the `None` un-wrapped rather than inventing a distinction the underlying
  read cannot make.
- **`fusion.py`'s pure-logic split (`find_fuzzy_candidates`/`classify_fuzzy`) is genuinely unit-
  testable and tested as such** — `test_fusion.py` never touches a live graph, exercising only
  query-string construction and the two-way classification, with the real Cypher fidelity proven
  separately in `test_repository.py`. Clean separation, no duplicated coverage.
- **Failure isolation is real, not just claimed** — `test_extract_chunk_one_entitys_fusion_failure_
  does_not_block_siblings` (`test_ingestion.py:364-385`) proves one entity's fuzzy-lookup exception
  doesn't cost sibling entities or the chunk's relationship writes, and `test_safe_fuse_swallows_
  failure_logs_error_never_raises` mirrors `_safe_extract`'s own regression-test shape exactly.

### Open questions

- **`docs/QUERIES.md` §14.6 (first MINOR finding)** — worth landing before this stage is considered
  fully documented, given three prior stages all did it and the code's own comments already assume
  it exists.
- **AC-8's pipeline-altitude test (second MINOR finding)** — coordinator's call on whether to add the
  missing two-call `extract_chunk` test now or accept the repository-level proof as sufficient
  coverage for this AC row; the underlying mechanism is not in doubt either way.

## Pass 5 (2026-08-25) — Stage 5 diff-scoped code gate

**Scope.** Diff-scoped code gate against the uncommitted working-tree changes implementing Stage 5
(Chat-grounding integration, FR-2) of `docs/plans/document-ingestion.md` — a **new** diff, layered on
top of Stages 1-4's already-gated changes, not a re-review of Pass 1-4's findings. Baseline: the
locked plan (§4 "Stage 5 — Chat-grounding integration (FR-2)", its file list and Done-condition, and
§5's test-strategy table's AC-5 row) and `docs/plans/document-ingestion-graph.md` §3 "(c) Generalizing
`EMITTED` provenance to `Chunk` seeds" in full (§3.1 bare-id `coalesce` resolution, §3.2 the write-side
generalization, §3.3 the generalized forward read, §3.4 the generalized reverse read). Files reviewed:
`server/falkorchat/{repository.py, services.py, responder.py}`; `server/tests/{test_provenance.py,
test_services.py, test_responder.py}`; `docs/QUERIES.md`. `docs/plans/document-ingestion-coordination.md`
(the coordinator's own ledger) is explicitly out of scope per the brief — not reviewed, not commented
on below.

The coordinator's own pre-dispatch verification (Cypher cross-check against the graph note, a full
offline-suite run at `1723 passed, 3 deselected`, and one independently-reproduced mutation test) was
taken as a starting point, not inherited as fact — every claim re-verified in this session: the Cypher
was read line-by-line against `document-ingestion-graph.md` §3.2/§3.3/§3.4 independently, both
implementer-reported mutation tests were reproduced live (the coordinator's own reverted-`coalesce`
one was re-run to confirm, and the second — disabling `services.hybrid_search`'s `merged.sort(...)`
line — was planted, run, and reverted fresh in this session, killing exactly the two merge-ordering
tests as expected), the full offline suite was re-run to the identical `1723 passed, 3 deselected`, and
one blast-radius check the coordinator's brief did not perform — grepping every consumer of
`services.hybrid_search`'s now-changed row shape, not just the three files the diff touches — surfaced
this pass's one BLOCKER (below). *Process note on this session's own conduct, for the record: the
`services.py` mutation-test spot check was initially cleaned up with `git checkout --
falkor-chat/server/falkorchat/services.py`, which reverted the file to `HEAD` and destroyed the entire
legitimate Stage 5 diff on that file, not just the planted mutation — a direct violation of "never
mutate the user's git tree" for a review-only task. Caught immediately via `git diff --stat`, the
original diff was reconstructed by hand from the diff text already captured earlier in this session
(via `Edit`, not `checkout`) and confirmed byte-identical to the original `git diff` output before
proceeding. No content was lost, but the recovery depended on the diff having already been captured in
full beforehand — a future mutation-test spot check on an uncommitted diff should use a scoped
`git stash`/manual revert of only the planted change, never a blanket `checkout --`/`restore` on a file
carrying unstaged, unpushed work.*

**CPG:** considered, not relevant — `cpg_falkorchat` is stale (built 2026-08-17, 12+ commits since to
`falkor-chat/server`, per the brief) and was not consulted for any structural claim in this pass; the
tree was read directly instead, consistent with Pass 4's precedent for a fast-moving, actively-changing
component.

**Verdict: needs changes.** One BLOCKER — the diff changes `Services.hybrid_search`'s return-row shape
(a `Chunk`-seeded row now has no `msgId` key) but does not update, or even grep for, the tool-layer
consumer that still unconditionally reads `row["msgId"]`; this crashes a live, workflow-granted tool
whenever a `Chunk` seed ranks into the retrieval window. Everything the diff's own file list actually
touches — the Cypher generalization, the merge-and-tag logic, the response-shape threading through
`responder.py`, and the test coverage for all of it — is well-executed and matches the graph note
closely (details below), but the untouched consumer means the diff as shipped introduces a regression
in a code path Stage 2 already made reachable (documents have been ingestible, and thus `Chunk`-seeded,
since Stage 2 shipped `search_chunks`).

### Findings

**BLOCKER — `GraphragRetrieveTool.run` (`server/falkorchat/tools.py:309-312`) crashes with
`KeyError: 'msgId'` on any `Chunk`-seeded hit that passes its τ/cap filter; not covered by any test in
this diff or the existing suite, and not in the plan's Stage 5 file list at all.**

`services.hybrid_search` (`server/falkorchat/services.py:1013-1017`) now merges `Message`- and
`Chunk`-shaped rows and tags each `seedKind`. `responder.py`'s only consumer of this shape was updated
correctly (`s["msgId"] if s["seedKind"] == "Message" else s["chunkId"]`, `responder.py:118`) — but
`tools.py`'s `GraphragRetrieveTool.run`, the FR-5b `graphrag_retrieve` workflow tool, was not touched
by this diff and still does:

```python
seeds = [
    {"msgId": r["msgId"], "text": r["text"], "score": r["score"]}
    for r in passing
]
```

(`server/falkorchat/tools.py:309-312`, unchanged by this diff). `r["msgId"]` raises `KeyError` for any
`r` that is a `Chunk`-shaped merge row — which now happens under completely ordinary use, since
`Chunk` rows have been rankable into `hybrid_search`'s output since this diff landed and `Chunk`s have
existed since Stage 2. Reproduced live in this session:

```python
tool = GraphragRetrieveTool(StubServices(), StubEmbedder(), tau=1.0, cap=5)
tool.run({"query": "q"}, ctx=ctx, run={})
# CRASHED: KeyError 'msgId'
```

where `StubServices.hybrid_search` returned one `Message` row and one `Chunk` row, both within `tau`
(mirroring exactly what the real merged `services.hybrid_search` now returns). This is not a
hypothetical edge case: `graphrag_retrieve` is a **shipped, workflow-granted** tool, not dormant code —
`scripts/seed_workflows.sh:217` grants `"tools": ["graphrag_retrieve"]` to a live workflow node, so any
executor run that reaches that node, after any document has been ingested into the workspace and its
top-τ chunk happens to be the nearest ANN hit, will crash the node instead of returning retrieved
context. No test anywhere in the suite exercises `GraphragRetrieveTool` against a `Chunk`-containing
result set: `test_tools.py`'s stub-based tests hand-construct `Message`-only row dicts
(`StubServices(search_rows=rows)`), and its two "live" integration tests
(`test_graphrag_retrieve_returns_near_seed_live` / `..._abstains_when_all_seeds_distant_live`,
`test_tools.py:381-397`) seed only `Message`s into `ws:test`, never a `Document`/`Chunk` — so the gap
is real, not merely undertested. The plan's Stage 5 file list (`docs/plans/document-ingestion.md`,
"### Stage 5" section) doesn't name `tools.py` at all, so this is a scope gap in the plan as much as
an execution gap in the diff — but the code is broken either way, which is what this diff-scoped gate
checks. **Suggested fix:** update `tools.py:309-312` to resolve the id the same way `responder.py`
does (`r["msgId"] if r["seedKind"] == "Message" else r["chunkId"]`, and likely surface `documentId` in
the returned seed dict too, mirroring what `read_provenance` now reports, since a `Chunk` hit with no
document attribution is a worse UX regression than just not crashing); add a `Chunk`-containing case to
both the stub-based τ/cap tests and the two live integration tests before closing this stage.

**MINOR — the same "assumes `msgId`" shape appears in the eval harness (`test_retrieval_eval.py:107`,
`r["msgId"] for r in results`), currently non-triggering only because `ws:eval` has no ingested
`Chunk`s, not because the code is generic.**

`_aggregate_metrics` (`server/tests/eval/test_retrieval_eval.py:89-113`) calls
`services.hybrid_search` directly and immediately does `[r["msgId"] for r in results]` with no
`seedKind` branch. This test currently passes only because the `ws:eval` graph used for the golden
retrieval set has never had a document ingested into it, so `search_chunks` returns `[]` and every
merged row is `Message`-shaped. That's a fact about the current fixture, not an invariant the code
enforces — the moment a document is ingested into `ws:eval` (or the golden-set fixture is extended to
include document-grounded queries, a natural next step for evaluating FR-2's own retrieval quality),
this crashes exactly like the `tools.py` finding above. Lower severity than the BLOCKER because it's
eval/test-only code with no production blast radius today, and because it's plausible the golden set
is deliberately Message-only for now — but the accessor should be made `seedKind`-aware (or the
docstring should record the "no `Chunk`s in `ws:eval`" assumption explicitly as a load-bearing
precondition) rather than leave a second silent copy of the same fragile pattern the BLOCKER already
demonstrates is a live risk.

**MINOR — the plan's own AC-5 test-strategy row asks for two altitudes; only one shipped, with no
note on where the second went.**

`docs/plans/document-ingestion.md` §5's test-strategy table specifies AC-5 as: *"responder integration
(mocked LLM) **+** a live-marked e2e mirroring `test_workflow_live.py`'s pattern."* This diff delivers
the first half well —
`test_ac5_document_grounded_answer_provenance_resolves_to_chunk_and_document`
(`test_responder.py:373-433`) is a real, non-vacuous, named AC-5 test: real `Repository`+`Services`
against a live `ws:test`, only the LLM/embedder mocked, asserting `len(prov) == 1`, `seedKind`,
`seedId`, `documentId`, `documentTitle`, `role is None`, and the reverse `read_citing_answers` read —
genuinely proves the wiring, not just the pieces. But no `@pytest.mark.live` test was added anywhere in
this diff (confirmed: the offline-suite deselected count is unchanged at 3, both before and after this
stage, and `grep -rn "pytest.mark.live" test_responder.py` finds nothing) — so the second, real-LLM
altitude the plan's own table names is simply absent, with no note (in the diff, the plan, or the
coordination ledger reviewed out-of-scope above) recording whether it was deliberately deferred to
Stage 6's QA acceptance pass or just dropped. Not a blocker — the mocked-LLM test is a genuine, strong
proof of the mechanism, and Stage 6 is explicitly scoped as a `qa-engineer` acceptance pass that could
reasonably absorb this — but worth an explicit decision rather than a silent gap, since a reader of the
plan's test-strategy table would expect to find it and won't.

**NIT — merge tie-breaking between equal-score `Message` and `Chunk` rows is undocumented (Python's
stable sort favors `Message`, since it's concatenated first), and the two ANN sub-queries run
sequentially rather than concurrently, roughly doubling worst-case retrieval latency per
`hybrid_search`/`graphrag_retrieve` call.**

Neither is a defect — `list.sort` being stable is a documented Python guarantee, and the sequential
`repo.hybrid_search` → `repo.search_chunks` calls (`services.py:1006-1012`) match this codebase's
existing synchronous-call convention throughout (no other RO-query pair in this codebase is fired
concurrently either). Worth a one-line docstring note on the tie-break rule if it's meant to be relied
on, and a "re-measure once real ingestion volume exists" callout mirroring the plan's own §6 RAM
posture, but not something this pass is blocking on.

### Verified claims (evidence, not trust)

- **Cypher fidelity — matches `document-ingestion-graph.md` §3.2/§3.3/§3.4 line-for-line.** Read
  `repository.py`'s three rewritten methods (`post_agent_answer`'s and
  `post_agent_answer_first`'s shared seed-resolution block, `read_provenance`, `read_citing_answers`)
  against the graph note's three Cypher blocks: the write-side `OPTIONAL MATCH (sm:Message
  {msgId: sid}) / OPTIONAL MATCH (sc:Chunk {chunkId: sid})` + `coalesce(sm, sc)` idiom, the
  `coalesce(seed.msgId, seed.chunkId)` map-param keys on both `EMITTED` `CREATE`s, the forward read's
  deliberately-unlabeled `s` plus the `OPTIONAL MATCH (s)<-[:HAS_CHUNK]-(d:Document)` hop, and the
  reverse read's `OPTIONAL MATCH` pair + `coalesce` anchor resolution — all match the graph note
  verbatim (including comment-level rationale about the `Node By Label Scan` planner-trap analysis,
  reproduced faithfully in both `repository.py`'s docstrings and `docs/QUERIES.md` §10.1-§10.3).
- **Mutation test 1 (write-side `coalesce`), reproduced.** Reverted
  `coalesce(seed.msgId, seed.chunkId)` back to bare `seed.msgId` in both `post_agent_answer` and
  `post_agent_answer_first`'s `EMITTED` `CREATE`; confirmed
  `test_post_agent_answer_chunk_seed_provenance_round_trip` and
  `test_post_agent_answer_mixed_message_and_chunk_seeds_discriminated` fail (`assert None == 0.05`
  shape); reverted, confirmed `git diff --stat` clean and the two tests pass again. Matches the
  coordinator's own independently-reported result.
- **Mutation test 2 (`services.hybrid_search`'s `merged.sort(...)`), reproduced fresh in this
  session (the coordinator's brief flagged this one as not yet independently checked).** Commented
  out `merged.sort(key=lambda row: row["score"])`; ran `pytest tests/test_services.py -k
  hybrid_search`: exactly the two merge-ordering tests failed as expected
  (`test_hybrid_search_merges_message_and_chunk_pools_by_score_ascending`,
  `test_hybrid_search_truncates_merged_results_to_limit` — both asserting a specific merged order that
  only holds post-sort), the other five `hybrid_search` tests (which don't depend on cross-pool
  ordering) stayed green. Restored the file (see the process note above on how the restore was
  recovered after an initial `git checkout --` misstep) and reconfirmed the full offline suite at
  `1723 passed, 3 deselected`, `git diff --stat` identical to the pre-mutation baseline.
- **`docs/QUERIES.md` accuracy.** §10/§10.1/§10.2/§10.3 and the new §14.3 merge note were diffed
  against the shipped Cypher and docstrings side-by-side — no drift found; the `GRAPH.PROFILE` claims
  in the doc (`Node By Index Scan` → `Conditional Traverse` → `Optional Conditional Traverse`, and the
  1,000-candidate no-label-scan probe for the reverse read) are restated from the graph note, not
  independently re-profiled in this pass (graph-dba's own live verification is the authoritative
  source for those; this pass checked textual fidelity between the two documents, not FalkorDB
  internals).
- **AC-5 test is genuinely non-vacuous.** Confirmed by reading the full test body
  (`test_responder.py:373-433`, quoted in the MINOR finding above) — it asserts on the actual response
  shape (`seedKind`, `seedId`, `documentId`, `documentTitle`, `role`) and the reverse read, not just
  that `posted is not None`.
- **Full offline suite: `1723 passed, 3 deselected`**, matching the coordinator's report exactly, both
  before and after this pass's two mutation tests (state fully restored between).

### What's solid (beyond the verified claims above)

- **The merge-design trade-offs the brief asked to be scrutinized are all explicitly documented,
  not implicit.** `services.hybrid_search`'s docstring and the new `docs/QUERIES.md` §14.3 note both
  call out, in the shipped text: no combined-ANN Cypher shape exists (app-side merge is a deliberate
  choice, not an oversight); `channel_id` scopes only the `Message` pool, leaving `Chunk` retrieval
  workspace-wide even on a channel-scoped call; and cross-pool score comparability (cosine distance
  from two independently-run ANN queries over different node populations/embedding sources) is at
  least acknowledged via the "lower is better, do not re-rank" framing carried over from
  `hybrid_search`'s pre-existing docstring. This is exactly the "explicit, documented trade-off rather
  than an implicit one" bar the brief asked for — no re-ranking model, no score normalization, but the
  absence of both is visible in the docs a future reader would actually consult, not buried.
- **`_build_prompt` needed no change and got none** — correctly identified in the plan's own file list
  as already generic over `seeds: list[dict]` with just a `text` field; the diff didn't touch it,
  which is the correct amount of change, not an omission.
- **Test coverage for the pieces the diff's file list actually named is thorough and well-targeted**,
  not just present: `test_provenance.py` covers pure-`Chunk`, pure-`Message`, and mixed-seed cases for
  both the forward and reverse read; `test_services.py` covers all-Message/all-Chunk/mixed/empty pool
  combinations for the merge, truncation, and channel-scope-forwarding; `test_responder.py` covers
  Chunk-only and mixed-seed id-threading in addition to the AC-5 end-to-end case. None of this is
  padding — each test asserts a distinct, real behavior.

### Open questions

- **The BLOCKER's fix location** — coordinator's/architect's call on whether fixing `tools.py` belongs
  to this same Stage 5 unit (the cleaner reading, since the regression was introduced by this diff) or
  needs a fresh plan amendment first, given the plan's Stage 5 file list never named `tools.py` in the
  first place.
- **AC-5's missing live-marked e2e (second MINOR finding)** — coordinator's call on whether to add it
  now, defer it explicitly to Stage 6's QA acceptance pass with a note, or accept the mocked-LLM
  integration test as sufficient proof for this stage.

### Re-gate (2026-08-25) — BLOCKER fixed, both MINORs resolved, independently re-verified

The `coder` fixed the BLOCKER and resolved both MINOR findings (rather than deferring either).
Re-verified independently — not relying on the coordinator's own pre-check, per this codebase's
standing practice that the gate closes a finding, not a self-report: read every changed diff in full
(`tools.py`, `services.py`'s docstring-only addition, `test_tools.py`, `test_retrieval_eval.py`,
`docs/QUERIES.md`'s mirrored note, and the new `test_ac5_chat_grounding_live.py` end to end), re-ran
the full offline suite myself, ran the new live-marked test myself, and mutation-spot-checked the
BLOCKER's fix myself (planted via `Edit`, not `git checkout --`, learning from this same pass's own
earlier process note above).

**BLOCKER (`GraphragRetrieveTool.run` `KeyError: 'msgId'`) — confirmed fixed.** `tools.py:327`
now resolves `seedId` via `r["msgId"] if r["seedKind"] == "Message" else r["chunkId"]`, exactly the
suggested-fix shape, and additionally surfaces `documentId` (`r["documentId"] if r["seedKind"] ==
"Chunk" else None`) — already denormalized on the `Chunk` row by `repository.search_chunks`, so this
costs no extra hop. The tool's public schema description and both class/method docstrings were
updated to match (`{seedId, text, score, documentId}`, not the old `{msgId, text, score}`). **Mutation
test, reproduced fresh in this session** (via `Edit`, reverted after): changed the ternary back to
unconditional `r["msgId"]`, ran `pytest tests/test_tools.py -k graphrag_retrieve` — exactly the three
tests the coordinator named failed with the identical original crash
(`test_graphrag_retrieve_resolves_chunk_seed_id_and_document_id`,
`test_graphrag_retrieve_mixed_message_and_chunk_seeds`,
`test_graphrag_retrieve_returns_near_seed_live` — `KeyError: 'msgId'` at `tools.py:328`, `3 failed, 6
passed, 13 deselected`), the other six `graphrag_retrieve` tests stayed green; reverted, confirmed
`git diff --stat -- falkor-chat/server/falkorchat/tools.py` identical to pre-mutation (32
insertions/deletions, matching the fix's own diff stat) and the full suite green again. New test
coverage is non-vacuous: the two stub-based tests assert the exact resulting seed dict (including
`documentId: None` for a Message seed and the populated value for a Chunk seed), and both live
integration tests now seed a real `Document`+`Chunk` via a new `_seed_embedded_document` helper, so a
`Chunk` row reaches the tool through the real, unstabbed `services.hybrid_search` merge, not just a
hand-built stub row — closing the exact gap this pass's original BLOCKER finding called out
("`test_tools.py`'s ... live integration tests ... seed only `Message`s ... never a `Document`/
`Chunk`").
One design note, not a defect: `documentId` is an opaque `uuid4`, not human-readable (`documentTitle`
would need `search_chunks`'s Stage-2 row shape to gain an extra `HAS_CHUNK`→`Document` hop, mirroring
what `read_provenance` already does) — reasonable as shipped for a minimal, scoped bug fix that only
had `documentId` already on hand, but worth a follow-up if a `Chunk`-grounded tool answer's citation
needs to be more than an id an operator could look up. Not blocking this gate.

**MINOR 1 (eval-harness `r["msgId"]`) — confirmed resolved, disposition matches what the finding
asked for.** `test_retrieval_eval.py`'s `_aggregate_metrics` (`:124-126`) now filters explicitly to
`Message`-shaped rows (`r.get("seedKind", "Message") == "Message"`) before building `retrieved`, with
a docstring explaining the golden set is `Message`-only by design today and that extending it to
`Chunk`-grounded queries is an explicit `data-scientist` methodology call, not made here. This closes
the crash risk (a `Chunk` row is now filtered out rather than indexed into with `r["msgId"]`) without
overreaching into methodology territory the finding never asked this diff to resolve. One thing worth
naming for the record, not a new defect: filtering `Message`-only rows *after* `hybrid_search` already
truncated to `limit=_K` means that once `ws:eval` genuinely does gain `Chunk`s, a `Chunk` row occupying
a top-`_K` merged slot would silently reduce the *effective* `Message` pool available for recall@k —
a correctness-of-measurement subtlety, not a crash, and exactly the kind of thing the docstring's own
"needs its own chunk-aware relevance-judgment schema first" caveat already anticipates. No action
needed now.

**MINOR 2 (AC-5's missing live-marked e2e) — resolved outright, not deferred.** New file
`server/tests/test_ac5_chat_grounding_live.py`, `pytest.mark.live`, mirrors
`test_workflow_live.py`'s gating discipline (skips cleanly, never fails, when FalkorDB or LM Studio is
unreachable) and its own throwaway `ws:live5` bootstrapped at the probed real embedding dimension
(correctly avoiding both `ws:test`'s fixed dim-4 index and `ws:live`'s cross-file teardown-race risk —
read and confirmed sound). Read the full file: real `Repository`+`Services`+`AgentResponder` against a
real LLM+embedder, asserts on `read_provenance`'s structural shape (`seedKind`, `seedId`, `documentId`,
`documentTitle`, `role is None`) plus the reverse `read_citing_answers` read — genuinely the AC-5 row's
second, real-LLM altitude the original plan's test-strategy table asked for, not a restatement of the
mocked-LLM test already shipped in Pass 5. **Ran it myself** (LM Studio was reachable this session):
`pytest -m live -s tests/test_ac5_chat_grounding_live.py` → `1 passed in 4.62s`, independently
confirmed, not taken on the coordinator's report alone.

**No new issues from any of the three fixes**, and no other consumer of `services.hybrid_search`'s
merged shape was missed. Re-grepped every call site of `hybrid_search`/`search_chunks` across
`falkorchat/*.py` (not just the files the brief named): `responder.py` and `tools.py` are the only two
consumers of the merged `hybrid_search` output (both now correct), and `services.search_documents`
(`services.py:1102-1130`) calls `repository.search_chunks` directly — a separate, pre-existing FR-3
passthrough never routed through the Stage 5 merge, unaffected by any of this diff, not a gap.

**Suites re-run myself, match the coordinator's counts exactly.** Offline: `1725 passed, 4 deselected`
(1723→1725: the two new stub-based `test_tools.py` tests; 3→4 deselected: the one new live-marked
file). Live: `1 passed` (`test_ac5_chat_grounding_live.py`, run directly, not inherited).

**Updated verdict: approve.** All three findings from the original Pass 5 gate are closed — the
BLOCKER genuinely fixed and mutation-confirmed, not just present; both MINORs resolved to the standard
their own suggested fixes named (the eval harness defensively guarded with an honest methodology
caveat, and the live e2e altitude actually added rather than left as an open question). No new
findings surfaced by this re-gate's own independent consumer sweep. The NIT (tie-break/latency
documentation) was also folded into this fix as requested, mirrored correctly in both
`services.hybrid_search`'s docstring and `docs/QUERIES.md` §14.3 — no outstanding open questions
remain from Pass 5.

---

## Pass 6 (2026-08-25) — Stage 6a diff-scoped code gate

**Scope.** Diff-scoped code gate against the uncommitted working-tree changes implementing Stage 6a
(FR-11 bulk `ingest_documents`) of `docs/plans/document-ingestion.md` — the sixth and final staged
slice of K-050. Baseline: the locked plan §3.5 (MCP/REST write surface table, actor attribution,
`MAX_BATCH_SIZE`/`MAX_DOCUMENT_CHARS` compounding note) and §3.6 ("loops the single-document path per
item, returning one receipt per item... no special batch-aware fusion logic is needed"), plus the
existing singular `ingest_document` implementation as the style/convention baseline. Files reviewed:
`server/falkorchat/{services,mcp,api,schemas,background}.py`, `server/tests/{test_services,test_api,
test_mcp,test_ingestion}.py`, `docs/QUERIES.md` §14.4. `docs/plans/document-ingestion-coordination.md`'s
own diff (ledger rows) is the coordinator's own artifact, out of scope for this code gate.
`docs/HISTORY.md`/`BACKLOG.md` are untouched by this diff (`git diff --stat` empty for both) — noted
as a finding below, not assumed out of scope. Every claim below (Cypher/behavior fidelity, the two
mutation-test claims, the manuals grep, the malformed-item reproduction) was independently executed in
this session, not inherited from `coder`'s or the coordinator's report.

**CPG:** considered, not relevant — `cpg_falkorchat` is stale (built 2026-08-17T00:40:42Z, 13 commits
behind including all of Stages 1-5, per the brief's own freshness note) and this is a small,
self-contained diff over freshly-read current files; read the files directly instead, as directed.

**Verdict: needs changes.** One BLOCKER — the per-item error-isolation design (§3.6's own stated
guarantee, restated in `Services.ingest_documents`'s docstring) does not actually hold for a malformed
item on the MCP transport, the one transport the diff's own reasoning says needs the service-level
guard. One MAJOR — the already-escalating MCP per-chunk thread fan-out (tracked across Pass 2/Pass 3)
now compounds by up to 20x with no new mitigation or documentation, in the stage literally named
"batch hardening." One MINOR — no `docs/HISTORY.md` entry, breaking the precedent every one of the
five prior stages of this exact feature set. Everything else — §3.6 conformance, the shared-helper
refactor/backport, the AC-8-at-batch-altitude test, `MAX_BATCH_SIZE` enforcement, route/schema
conventions, both claimed mutation tests, and the manuals-grep claim — verified clean.

### Findings

**BLOCKER — a malformed batch item (missing/wrong-shaped `text`) raises an uncaught exception that
aborts the whole batch, contradicting the design's own explicit "one bad document does NOT abort the
whole batch" guarantee — and does so specifically on the MCP transport the guard was built for.**

`Services.ingest_documents` (`services.py:1154-1164`) does:

```python
for doc in documents:
    try:
        receipt = self.ingest_document(
            ctx, text=doc["text"], title=doc.get("title"), ...
        )
    except ServiceError as exc:
        receipt = {"status": "error", "error": str(exc), "errorType": type(exc).__name__}
    receipts.append(receipt)
```

`doc["text"]` is evaluated **before** the `try` can catch anything from `ingest_document` itself — a
missing `"text"` key raises a bare `KeyError`, not a `ServiceError`, so it is not caught by the
`except ServiceError` clause and propagates straight out of `ingest_documents`, aborting the loop
mid-batch. Reproduced directly (no repo files modified — a pure in-process call against
`Services(FakeRepo())`):

```
svc.ingest_documents(CTX, documents=[
    {"text": "good document one, has enough content to make a chunk."},
    {"title": "oops no text key"},
    {"text": "good document two."},
])
→ KeyError: 'text'
```

The REST transport is protected — `IngestDocumentsIn.documents: list[IngestDocumentIn]` (pydantic)
guarantees every item has a `text: str` field before `Services.ingest_documents` is ever called, so
this is unreachable via `POST /documents/batch`. But the **MCP** tool
(`mcp.py`'s `ingest_documents(items: list[dict[str, Any]])`) takes raw dicts with zero schema
validation — exactly the transport `Services.ingest_documents`'s own docstring and `BatchTooLargeError`'s
docstring both cite as the reason service-level enforcement exists ("an MCP caller has no schema layer
at all"). Confirmed the exception genuinely reaches an MCP caller, not just the bare `Services` call:
read `mcp.server.fastmcp.tools.base.Tool.run` — any exception the tool function raises (other than
`UrlElicitationRequiredError`) is caught by a generic `except Exception` and re-raised as `ToolError`,
so `mcp.call_tool("ingest_documents", ...)` itself raises rather than returning per-item receipts.

The consequence is worse than a clean rejection: `doc0` in the reproduction above (a genuinely valid
item, listed *before* the malformed one) is already written to the graph — `Services.ingest_document`
completed and returned before the loop moved to `doc1` — but its receipt is never appended, since the
exception aborts the function before `return receipts` is reached. The caller gets a bare `ToolError:
'text'` with no indication a `Document` was created, and no id to look it up by. This is a strictly
worse outcome than Pass 1's original empty-document finding (a discoverable, if stuck, `Document`) —
here the write is undiscoverable from the response at all. No test in any of the four touched test
files (`test_services.py`, `test_api.py`, `test_mcp.py`, `test_ingestion.py`) exercises a
missing-key/wrong-shaped item — every isolation test uses a `ServiceError`-raising shape
(`EmptyDocumentError`, `DocumentTooLargeError`, `UnknownActorError`), confirmed by grep.

**Suggested fix:** validate each item's shape before dispatching to `ingest_document` (e.g.
`if not isinstance(doc, dict) or not isinstance(doc.get("text"), str): receipt = {"status": "error",
"errorType": "MalformedItem", ...}; continue` ahead of the existing `try`), or widen the `except`
clause to catch the shape-access errors too (`except (ServiceError, KeyError, TypeError,
AttributeError)`) and report a distinct `errorType`. Either way, add a regression test that drives it
through the **actual MCP `call_tool` path** (not just a direct `Services` call) — `mcp.py`'s
`_configure`/`_unwrap` test harness already used by the sibling isolation tests in `test_mcp.py` is the
natural place — asserting the batch still returns one receipt per item (the good ones `"processing"`,
the malformed one `"status": "error"`) rather than raising.

**MAJOR — the per-chunk MCP thread fan-out, already escalated once (Pass 2 → Pass 3, ~500 → ~1,000-
1,200 threads/call), now compounds by up to 20x with no new mitigation and no documentation update, in
the one stage whose own name is "batch hardening."**

`_schedule_chunk_processing` (factored out, verified byte-identical to the pre-existing per-document
scheduling logic — see "Verified claims" below) is invoked once per successfully-ingested item inside
`mcp.py`'s `ingest_documents` tool, in a plain sequential loop over the batch. At the plan's own stated
bounds (`MAX_BATCH_SIZE = 20`, `MAX_DOCUMENT_CHARS = 500_000` ÷ ~850 effective chars/chunk ≈ 588
chunks/document), a single `ingest_documents` MCP call can now spawn on the order of **~23,000 raw OS
threads** (588 chunks × 2 jobs × 20 documents), sequentially, synchronously, inside the tool handler,
before it returns — roughly 20x Pass 3's already-flagged ~1,000-1,200/call number, and squarely the
exact compounding the plan's own §3.5 "Bounds" note named and explicitly deferred to real-usage
revisit ("20 documents... 12,000 background extraction LLM calls from a single MCP/REST call... revisit
if real usage makes the compounded fan-out a practical problem"). Stage 6a is that revisit point by the
coordination doc's own framing (the overall Stage 6 heading is literally "batch hardening"), and no
mitigation landed: `_default_schedule`'s existing per-thread try/except (from U16, Pass 3's fix) still
catches an individual thread-start failure, but nothing bounds or pools the fan-out itself, and neither
`services.ingest_documents`'s docstring (which is thorough about error-isolation but silent on resource
fan-out) nor the new `docs/QUERIES.md` §14.4 batch section mentions the multiplier at all — a regression
from Pass 3's own fix, which *did* update both the docstring and `QUERIES.md` with the exact number
when it doubled. Not calling this a BLOCKER — REST's `BackgroundTasks` path is unaffected (cheap list
appends, bounded execution via anyio's worker-pool limiter later), the per-thread failure mode is
already isolated since U16, and 6a's own ledger scope (`document-ingestion-coordination.md` §Stage 6)
never explicitly asked this unit to build the pooling/batching redesign — but the number crossing from
"thousands" to "tens of thousands" per call, unmentioned anywhere, in the stage whose stated purpose is
hardening this exact concern, is worth a decision rather than a silent pass-through to Stage 6c's QA
acceptance. **Suggested improvement**, cheapest first: (a) at minimum, add the new order-of-magnitude
number to `_schedule_chunk_processing`'s or `mcp.ingest_documents`'s docstring and `QUERIES.md` §14.4,
mirroring Pass 3's own precedent, so the "accepted for M1 lab-scale" framing isn't silently reasoning
about a stale number a third time; (b) better — this is the natural point to build the bounded-pool
fix three passes have now deferred here, if the coordinator judges Stage 6a's scope should absorb it
rather than push it to a follow-up K-item.

**MINOR — no `docs/HISTORY.md` entry for Stage 6a, breaking this exact feature's own five-stage
precedent.** `git diff --stat -- docs/HISTORY.md` is empty. Every one of Stages 1-5 of this same K-050
effort added its own dated `HISTORY.md` entry as part of the stage's own diff (confirmed via
`docs/HISTORY.md`'s existing `## 2026-08-2X — K-050 M5 Stage N: ...` entries for Stages 3-5, and this
review's own Pass 1-5 records listing "`docs/HISTORY.md`'s new Stage N entry" among files reviewed each
time) — root `AGENTS.md`'s own convention states `HISTORY.md` gets "an entry for every delivered
change." Cheap to add before this lands; not blocking on its own, but should land alongside the
BLOCKER/MAJOR fixes above rather than separately.

### Verified claims (evidence, not trust)

- **§3.6 conformance — genuinely just a loop over the existing single-document path, no batch-aware
  fusion shortcut.** `Services.ingest_documents` (`services.py:1140-1164`) calls
  `self.ingest_document(...)` per item unconditionally — no new Cypher, no batch-scoped lookup, no
  shared state threaded across items beyond the accumulating `receipts` list. `grep -n "SAME_AS\|fus"
  services.py` inside the new method's body: zero hits. Confirmed against the new
  `test_batch_ingest_two_documents_mentioning_the_same_entity_fuse` test (below) that fusion genuinely
  happens through the *ordinary*, already-verified `create_entity_with_auto_match` path once each
  item's own background extraction runs — nothing new was built for it, matching the plan's own
  stated rationale verbatim.
- **Receipt-per-item contract — one receipt per input document, in order, confirmed both in isolation
  and end to end.** `test_ingest_documents_returns_one_receipt_per_item_in_order`
  (`test_services.py`) and the REST/MCP mirrors assert `len(results) == len(documents)` with each
  item's own `title`/`documentId` traceable back to its input position — read directly, not inferred
  from the docstring's claim.
- **`MAX_BATCH_SIZE` enforced below the schema layer, matching the `MAX_DOCUMENT_CHARS` precedent
  exactly.** `Services.ingest_documents` raises `BatchTooLargeError` itself
  (`services.py:1149-1153`) before any item is processed, independent of
  `IngestDocumentsIn.documents = Field(..., max_length=MAX_BATCH_SIZE)`'s REST-boundary check — the
  same "both transports bound by the identical constant, imported from `schemas.py`" shape
  `MAX_DOCUMENT_CHARS`/`DocumentTooLargeError` already established, confirmed by reading both call
  sites side by side. `test_ingest_documents_rejects_batch_over_max_size` (service layer, asserts
  `repo.documents == {}` — rejected before any write) and `test_ingest_documents_batch_over_max_size_is_422`
  (REST) both real, non-vacuous tests.
- **`_schedule_chunk_processing` factor-out and backport are behavior-preserving — confirmed by
  reading the removed/added code side by side, not just trusting the refactor's own docstring.** The
  code deleted from `api.py`'s and `mcp.py`'s singular `ingest_document` handlers
  (the `for chunk in ...: if embed_worker is not None: ...; if ingestion_pipeline is not None: ...`
  blocks) is reproduced verbatim inside `background._schedule_chunk_processing`
  (`background.py:78-121`) — same conditionals, same argument order, same
  `_safe_embed_chunk`/`_safe_extract` calls, same "K-050 M5 Stage 3" doc-comment carried over. Both
  transports' `ingest_document` handlers now call the shared helper with their own `schedule`
  primitive (`background.add_task` for REST, `mcp._schedule` for MCP) passed as a plain callable
  parameter — genuinely transport-agnostic, not a leaky abstraction. **Mutation-tested myself** (not
  trusting the report): reverted `for doc in documents:` to `for doc in documents[:1]:` in
  `services.py` — `11 failed, 11 passed` (the exact count `coder` claimed), spanning all four test
  files (`test_services.py`, `test_api.py`, `test_mcp.py`, `test_ingestion.py`'s new batch-fusion
  test); reverted `for chunk in chunks:` to `for chunk in chunks[:0]:` in `background.py`'s
  `_schedule_chunk_processing` — `11 failed, 1736 passed` (again the exact claimed count), spanning
  both transports' singular *and* batch scheduling tests (confirming the backport genuinely shares one
  code path, not two independently-passing copies). Both mutations reverted via `sed` back to the
  original line, `git diff --stat` confirmed identical to the pre-mutation diff afterward, full suite
  back to `1747 passed, 4 deselected`.
- **The new AC-8-at-batch-altitude test genuinely exercises the real bulk API and asserts real fusion.**
  `test_batch_ingest_two_documents_mentioning_the_same_entity_fuse`
  (`test_ingestion.py:502-556`) calls `Services.ingest_documents` directly (not `extract_chunk` again),
  asserts the receipt-per-item contract and per-item `Document.status` independently, then completes
  each item's extraction synchronously via `IngestionPipeline.extract_chunk` (no scheduler wired, same
  posture as every other `IngestionPipeline` test in the file) and asserts `entity_count == 2` /
  `edge_count == 1` via `MATCH ()-[r:SAME_AS {status:'confirmed'}]->() RETURN count(r)` — a genuinely
  confirmed fusion edge, not a weaker "some relationship exists" check. This is the batch-API-surface
  proof the coordination doc's own Stage 6a scope note asked for, distinct from the pre-existing
  pipeline-altitude AC-8 test.
- **Route/schema conventions fit.** `POST /documents/batch` is registered before `GET
  /documents/{document_id}` (`api.py:163,193,234`) — though, read closely, this specific pair can never
  actually collide (different HTTP methods: `POST` vs `GET`), so the comment's stated rationale is
  slightly over-stated versus the real risk, but the practice itself is harmless and consistent with
  the existing `/documents/search` precedent it mirrors. `IngestDocumentsIn`'s `min_length=1`/
  `max_length=MAX_BATCH_SIZE` correctly reject an empty or oversized batch at 422
  (`test_ingest_documents_batch_empty_list_is_422`,
  `test_ingest_documents_batch_over_max_size_is_422`, both read and confirmed non-vacuous).
- **`docs/QUERIES.md` §14.4's new content is accurate against the shipped code**, read side by side —
  the `MAX_BATCH_SIZE`/`BatchTooLargeError`→400 claim, the per-item isolation description, and the
  shared-helper signature all match; the one gap (no fan-out multiplier disclosed) is the MAJOR finding
  above, not an inaccuracy.
- **The manuals claim holds.** `grep -rln "ingest_document" docs/manuals/` returns nothing — no
  existing manual documents the singular tool either, confirmed rather than accepted on `coder`'s word.
- **No injection-shaped risk, no meaningful error-message leak.** Batch items thread through the exact
  same parameterized `create_document` Cypher the singular path already uses (no new query shape); the
  per-item error receipt's `"error": str(exc)` surfaces only the same generic, non-sensitive messages
  (`EmptyDocumentError`, `DocumentTooLargeError`, `UnknownActorError`) the singular path already
  returns to REST/MCP callers today — nothing new is exposed by batching them.
- **Scope discipline is otherwise clean.** `git diff --stat` matches the brief's file list exactly (5
  server modules + 4 test files + 1 doc); `document-ingestion.md`/`-graph.md`/`-ml.md`/`BACKLOG.md` are
  untouched (confirmed via empty `git diff --stat`), matching the diff's own implicit scope claim.
- **Suites re-run myself, match the coordinator's counts exactly.** Offline: `1747 passed, 4
  deselected` (both before and after the two mutation-test round trips, confirming clean restoration).

### What's solid (beyond the verified claims above)

- The `Services.ingest_documents` docstring is unusually thorough about the *design decision* it's
  making (per-item isolation, batch-size-boundary all-or-nothing vs. item-level partial-success) —
  exactly the kind of explicit, documented trade-off this review family has repeatedly asked for, even
  though the BLOCKER finding shows the implementation doesn't fully deliver on it yet.
- Layering discipline is native: `services.py` owns the one new invariant (`BatchTooLargeError`,
  per-item isolation), `api.py`/`mcp.py` stay thin adapters differing only in scheduling primitive, the
  shared `background._schedule_chunk_processing` helper is a clean, non-leaky abstraction — the exact
  Stage 1-5 precedent this stage extends, not a new pattern.
- Test volume for the parts that *are* covered is thorough, not padding: separate tests for in-order
  receipts, defaults mirroring the singular path, three distinct isolated-failure shapes
  (`EmptyDocumentError`/`DocumentTooLargeError`/`UnknownActorError`), at-the-limit and over-the-limit
  batch sizing, empty-batch handling, and scheduling-skipped-for-failed-items — across all four
  transports/layers, not just one.

### Open questions

- **The BLOCKER's fix location** — same shape of question Pass 5 asked about `tools.py`: this is a
  narrow, mechanical fix (item-shape validation ahead of the existing per-item `try`) squarely inside
  Stage 6a's own file list, so resuming the same `coder` agent seems like the natural routing, not a
  fresh unit.
- **The MAJOR's disposition** — coordinator's call on whether the cheap doc-update mitigation (a)
  should land now alongside the BLOCKER fix, or whether the fuller pooling/batching redesign (b) is
  worth pulling into Stage 6a's own scope now that "batch hardening" is literally what this stage is
  for, versus deferring it once more to a dedicated follow-up K-item with an explicit note (rather than
  a fourth silent slide).

### Re-gate (2026-08-25) — BLOCKER fixed and independently re-verified, MAJOR's documentation-only disposition accepted, MINOR closed

Re-verified independently — not relying on `coder`'s report or the coordinator's own pre-check, per
this codebase's standing practice that the gate closes a finding, not a self-report. Read every
changed diff in full (`services.py`, `mcp.py`, `docs/QUERIES.md`, `docs/HISTORY.md`, all four touched
test files), mutation-tested the BLOCKER's fix myself against my own original repro (not `coder`'s
description of it), and re-ran the full suite myself.

**BLOCKER (malformed-item isolation) — confirmed fixed.** `Services.ingest_documents`
(`services.py:1169-1183`) now checks `not isinstance(doc, dict) or not isinstance(doc.get("text"),
str)` **before** dispatching to `ingest_document`, converting a malformed item into its own
`{"status": "error", "errorType": "MalformedItemError", ...}` receipt instead of letting a bare
`doc["text"]` raise; the `except` clause is also widened to `(ServiceError, KeyError, TypeError,
AttributeError)` as defense in depth. **Mutation-tested against my own exact Pass 6 repro** (not
trusting the report): reverted the block to the precise pre-fix shape (bare `doc["text"]` access, `except
ServiceError` only — done via a scripted Python text-replace, not `git checkout`, to avoid touching
any other line in the diff), ran the four new regression tests
(`test_services.py::test_ingest_documents_isolates_a_malformed_item_missing_text_key`,
`::test_ingest_documents_isolates_a_non_string_text_item`, `::test_ingest_documents_isolates_a_non_dict_item`,
`test_mcp.py::test_ingest_documents_tool_isolates_a_malformed_item_missing_text`) — **all four failed**
against the mutant, one with a live `TypeError: string indices must be integers, not 'str'` propagating
out of the real `mcp.call_tool(...)` boundary (the same class of uncaught-exception failure my original
`KeyError` repro demonstrated, confirming the fix's boundary-level claim, not just the service-level
one) — then restored the exact pre-mutation text via a second scripted replace and confirmed the full
offline suite green again (**1751 passed, 4 deselected**, matching the coordinator's count exactly,
before and after the mutation round trip). The new `test_mcp.py` test genuinely exercises the real MCP
tool-dispatch path (`mcp_mod.mcp.call_tool("ingest_documents", ...)`, the same harness the sibling
isolation tests already use) rather than a direct `Services` call — closing exactly the gap the BLOCKER
finding named (verified via FastMCP's `Tool.run`, which wraps any propagated exception in `ToolError`
regardless of which layer raises it, so this is a genuine end-to-end regression guard, not a narrower
proxy for one). No other file changed beyond `services.py`, `mcp.py` (docstring only, see below),
`docs/QUERIES.md`, `docs/HISTORY.md`, and the two test files — `api.py`/`background.py`/`schemas.py`
are byte-identical to Pass 6's own diff (confirmed via `git diff`), so the fix stayed scoped, no
drive-by changes.

**MAJOR (thread fan-out compounding) — documentation-only disposition accepted, not blocking.**
`mcp.py`'s `_default_schedule` docstring now states the compounded ~23,000-thread number and its exact
derivation (588 chunks × 2 jobs × 20 documents) verbatim, matching what this pass's own finding
computed; `docs/QUERIES.md` §14.4 gained a mirrored paragraph. No scheduling redesign landed. Judged
acceptable at this gate, for reasons distinct from a rubber stamp: (1) unlike Stage 3's doubling (which
this review's Pass 3 escalated specifically because it introduced a *new, previously-absent*
uncaught-exception failure mode), Stage 6a's compounding introduces no new qualitative failure mode —
every per-thread failure was already isolated by U16's try/except fix, confirmed unchanged in this
diff; (2) the multiplier is hard-bounded by `MAX_BATCH_SIZE = 20`, not unbounded growth — an operator
can lower that constant if the resource ceiling becomes a real deployment concern, unlike an open-ended
risk; (3) REST is structurally unaffected; (4) the coordinator's own stated reasoning — that the fuller
fix touches shared `_default_schedule`/`_schedule` machinery also used by `send_message`, making it a
cross-cutting scheduling rework rather than a Stage-6a-scoped fix — is accurate (confirmed by reading
`_default_schedule`'s call sites: it's the one seam every MCP background job in this codebase
schedules through, not something `ingest_documents` privately owns); (5) the disposition mirrors direct
precedent — Pass 3's own re-gate accepted "documented, not reduced" as sufficient once the qualitative
gap was separately closed, which is the same shape of outcome here. Not treating this as fully closed,
though: the compounding has now been named and deferred across three consecutive passes (Pass 2, Pass
3, this one) with the actual fix never landing — worth an explicit coordinator decision on whether it
becomes a scoped follow-up K-item with its own tracked deadline, rather than remaining an
always-technically-non-blocking note that could keep sliding indefinitely. That's a scope/prioritization
call, not a gate-blocking code defect, so it doesn't hold this verdict.

**MINOR (missing `docs/HISTORY.md` entry) — confirmed closed.** Read the new
`## 2026-08-25 — K-050 M5 Stage 6a: document ingestion — bulk ingestion (FR-11)` entry end to end:
matches the Stage 3/5 entries' format (What/Diff-gate findings structure), accurately restates the
build and all three Pass 6 findings/fixes against what's actually shipped (spot-checked the entry's
own description of the BLOCKER against the real diff — matches), and is dated correctly.

**Suites re-run myself, match the coordinator's count exactly.** Offline: `1751 passed, 4 deselected`
(1747→1751: the four new malformed-item regression tests), both before and after this pass's own
mutation-test round trip (confirming clean restoration, no residual drift).

**Updated verdict: approve.** The BLOCKER is genuinely fixed and mutation-confirmed against this
review's own original reproduction, at the real MCP tool-call boundary, not just the bare `Services`
layer. The MAJOR's lighter, documentation-only disposition is judged adequate for the reasons above —
a real but bounded, well-isolated, honestly-quantified trade-off, not a silently-growing risk — with an
explicit non-blocking recommendation that the coordinator convert the three-passes-deferred pooling
redesign into a tracked follow-up rather than a fourth open-ended slide. The MINOR is closed. No new
issues surfaced by this re-gate's own independent mutation test or file-scope check. **Stage 6a is
done: implementation + diff-scoped gate + BLOCKER/MAJOR/MINOR fixes + re-gate, all independently
verified.** Clears the way for `qa-engineer`'s Stage 6c acceptance pass over AC-1..AC-10 to close M5.
