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
