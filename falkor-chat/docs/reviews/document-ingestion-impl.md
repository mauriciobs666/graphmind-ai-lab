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
