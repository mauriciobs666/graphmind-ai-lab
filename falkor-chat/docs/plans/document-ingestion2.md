# Document ingestion — update & delete — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** —

Turns `docs/requirements/document-ingestion2.md` (FR-1..FR-8, AC-1..AC-8, Status: Ready for
design) into an ordered, staged build. Resolves Open Question 1 (entity/relationship cascade on
update/delete) as a concrete design decision. Open Question 2 (update-detection technique) and
part of Open Question 1's confidence-tiering were delegated to `data-scientist`, exactly as the
original feature's OQ-1 (entity-match technique) was delegated — **the note has landed**
(`docs/plans/document-ingestion2-ml.md`) and this plan is written against its conclusions, not a
guess at them. Open Question 3 (where a pending suggestion surfaces) is resolved by direct reuse
of the original feature's OQ-2 answer (a dedicated review surface, not chat) — nothing about
document granularity changes that reasoning.

## 0. Delegation summary

| Follow-on note | Owner | What it settled |
|---|---|---|
| `docs/plans/document-ingestion2-ml.md` | `data-scientist` | **Deterministic, uncalibrated-threshold-free detection at both tiers — no embeddings, no LLM, in v1.** Auto tier ("very-high confidence," FR-2/AC-1): exact equality of `Document.textNormalizedHash` (a hash of case-folded, whitespace-collapsed full text) against an existing `currentVersion` document — a definitional identity check, not a score, `confidence=1.0`. Suggested tier (FR-2/AC-2): shingled-Jaccard content-overlap (5-word shingles) over a cheaply-narrowed candidate shortlist, `confidence` = raw Jaccard ratio, stored as an explicitly non-probabilistic audit value with an implementer-tunable noise-floor cutoff. Candidate scope: workspace-wide among `currentVersion` documents only, **never** actor-scoped (would miss the motivating cross-actor-edit case). Pipeline placement: the auto tier runs **synchronously**, folded into one atomic write (mirrors `create_entity_with_auto_match`); the suggested tier runs **asynchronously** (cost/latency of the Jaccard computation over up to 500,000-char documents, not an embedding dependency — neither tier needs `Chunk.embedding`). A false auto-supersede is judged **more consequential** than a false entity auto-merge (it hides an independent document from default search, not just adds a spurious link) — reflected in an even narrower auto-tier criterion than the entity precedent's name+type match. Embeddings/LLM comparison both explicitly deferred to a scoped, evaluation-gated v2 (§6 of the note), for the same "zero calibration data" reason `document-ingestion-ml.md` gave for entity matching, sharpened by the larger blast-radius argument above. The note also recommends (§7) that the `SUPERSEDES` confirm/reject/recheck/list surface mirror `SAME_AS`'s **exactly**, including the OQ-3 reopen-on-corroboration/manual-recheck pattern — adopted below (§3.4), reversing this plan's own first-draft instinct to scope that out, because the edge already carries `resuggestCount`/`lastResuggestedAt` (per the note's F6) and leaving those fields permanently unused would be a design inconsistency, not a simplification. Two schema/RAM specifics (the `textNormalizedHash` index shape, and the suggested-tier candidate-generation index) were explicitly left to `graph-dba` to finalize — both now resolved by the live-verification pass below. |
| `graph-dba` live-verification pass, 2026-09-11 (throwaway `ws:docprobe`, deleted after) | `graph-dba` | Dispatched by `teco` against this plan's design, not a separate written note. **4 of 5 items confirmed exactly as designed, no correction:** the unlabeled-`SUPERSEDES`-endpoint planner-trap discipline (§3.2, `GRAPH.PROFILE`-confirmed for the match+`SET` shape and the bare status-filter shape alike); the atomic auto-supersede write's concurrency fix (§3.4, `threading.Barrier` probe — exactly one confirmed edge, never zero/duplicated — plus one **technique note, folded in below**: build it as two separate `FOREACH`-guarded blocks sharing one `WITH`, not one `FOREACH` mixing `SET`+`CREATE`, the form actually verified and the closer mirror of `create_or_reopen_match`'s own idiom); the single-query delete shape (§3.5 — ran verbatim, succeeded outright; **the two-query fallback is dropped below**, no longer needed as a hedge); the `Document.textNormalizedHash` RANGE index (§7 — index-anchored, no constraint, exactly as designed); the `SUPERSEDES` DDL (§3.2/§4 Stage B — both indexes + the `UNIQUE RELATIONSHIP` constraint reach `OPERATIONAL`, duplicate-`matchId` rejection confirmed live). **One item needed a real redesign, folded into §3.4/§4.1/§6/§7 below:** a raw RediSearch fulltext index on `Document.text` at the plan's own 500,000-char ceiling measured **2-5x the raw text size in RAM** (0.27 MB/doc on a low-entropy 8k-token vocabulary probe, 2.45 MB/doc on a higher-entropy 60k-token one closer to real prose) — not negligible, stacking on the already-dominant `Chunk.embedding` line workspace-wide. Replaced with an app-side MinHash/LSH-banding fingerprint computed off the same shingle set `update_detection.shingles()` already builds — a handful of small indexed properties per document, negligible RAM, the standard technique for this exact near-duplicate-detection shape. `Document.title`'s fulltext index (unaffected by the RAM finding, stays as the cheap complementary booster it already was) and the K-049-family oversized-indexed-value crash risk (confirmed **not reachable** here — `RANGE`-only, no `UNIQUE` constraint on any of `Document.text`/`textNormalizedHash`, so the crash family this plan already avoided by design stays avoided) needed no change. This is an index-design correction on `graph-dba`'s own call, not a reversal of anything `data-scientist` recommended — the underlying deterministic-Jaccard technique is unchanged; only how candidates are cheaply narrowed changes. |
| (future) `docs/plans/document-ingestion2-coordination.md` | `teco` | Sequencing/gating log once implementation starts — not authored here. |

---

## 1. Goal & scope

**Goal.** Add real `update`/`delete`/`list` capability to `falkor-chat`'s document-ingestion
pipeline: automatic, confidence-tiered detection of "this new submission is an edited version of
an already-ingested document" (auto-supersede at very-high confidence, pending suggestion
otherwise); true, explicit-only hard deletion with an audit trail; version history retained and
inspectable; all reachable via the same MCP/REST surface ingestion itself already uses.

**In scope:** FR-1..FR-8, AC-1..AC-8; resolving Open Question 1 (entity/relationship cascade).

**Out of scope** (per the requirements doc): the detection technique's ML specifics (delegated,
§0); where a pending suggestion surfaces (resolved by direct reuse of the original feature's OQ-2
answer, §3.4); any change to `agent-knowledge-base-strategy.md`'s own substrate decision — that
document is a downstream consumer of this capability, revisited once this ships, not decided here.
Also out of scope, by the requirements doc's own decision log: an explicit
`update_document(document_id, new_text)` tool — see §3.3's "no explicit-ID update path" note.

---

## 2. Context & findings

**CPG:** considered, not relevant — this is new-code design for `falkor-chat/server`, extending an
already-designed pipeline via direct reading of the current tree (`repository.py`, `services.py`,
`mcp.py`, `api.py`, `schemas.py`, `background.py`, `config.py`, `app.py`,
`scripts/bootstrap_schema.sh`) rather than an impact-analysis question a call graph would answer
faster than reading the (relatively small, already-well-documented) source directly.

### 2.1 What already exists, precisely (file:line, verified by reading the source)

- `repository.py:1013` `create_document` — plain guarded `CREATE`, non-idempotent by design
  (`Document{documentId,title,text,sourceFormat,sourceKind,status,pendingJobs,createdAt}` +
  `HAS_CHUNK` → `Chunk{chunkId,text,seq,documentId}`).
- `:1059` `get_document`, `:1080` `start_document_progress`, `:1113` `report_document_job_done`
  (both guard on `d.status='processing'`, first-terminal-write-wins), `:1146`
  `list_document_chunks` (internal-only), `:1171` `set_chunk_embedding`, `:1201` `search_chunks`
  (`Chunk`-only ANN, `ORDER BY score ASC`, no `Entity` expansion yet).
- `:1242` `create_entity` (always a new node), `:1276` `link_chunk_about_entity`, `:1296`
  `create_entity_relationship` (`RELATES_TO`, never deduplicated — FR-6's provenance-first axis).
- `:1336` `create_entity_with_auto_match` — **the load-bearing precedent this plan reuses at
  document granularity.** One atomic `GRAPH.QUERY` folding (1) an `OPTIONAL MATCH` for the oldest
  exact candidate sharing `(nameNormalized, type)`, (2) the new `Entity`'s `CREATE`, (3) a
  `FOREACH`-conditional `CREATE` of a `SAME_AS{status:'confirmed', decidedBy:'system',
  confidence:1.0}` edge — all in one round trip, closing a plan-gate-review BLOCKER (a
  check-then-act race across concurrent calls) by construction, because FalkorDB/Redis serializes
  `GRAPH.QUERY` execution.
- `:1401` `find_fuzzy_candidates` (RediSearch fuzzy full-text on `Entity.name`, type-filtered),
  `:1426` `create_or_reopen_match` — the guarded find-or-create-or-reopen write on an `OPTIONAL
  MATCH`-ed **undirected** `SAME_AS` edge (not a bare `MERGE`, which would miss an edge written in
  the opposite discovery order). `:1480` `confirm_match`, `:1504` `reject_match` (never deletes
  the edge), `:1528` `recheck_match` (`rejected`→`pending` only; a no-op for any other status,
  including "no such id" — both cases are indistinguishable and equally "nothing to do"). `:1554`
  `list_pending_matches`, `:1579` `list_matches` (status-filterable **via two separate query
  strings, not a null-guarded `WHERE`** — that idiom silently drops the index even when `$status`
  is bound, a live-verified quirk).
- **Critical planner trap** (`repository.py:1326-1334`'s own comment, `graph-dba`'s finding): every
  `SAME_AS` query matches its endpoints **unlabeled** (`(a)-[r:SAME_AS {...}]->(b)`, never
  `(a:Entity)-[r:SAME_AS ...`) — a bare label on either endpoint of a relationship-index-anchored
  query forces a full `Node By Label Scan` on this FalkorDB build even though the
  relationship-property scan alone is fully selective. **This applies identically to any new
  relationship-indexed edge type** — carried forward to `SUPERSEDES` below (§3.2).
- `services.py:1081` `ingest_document` (validates non-empty text + `MAX_DOCUMENT_CHARS`, splits via
  `chunking`, calls `create_document`, returns `{documentId, chunkCount, status:'processing'}`),
  `:1125` `ingest_documents` (loops, per-item error isolation, `MAX_BATCH_SIZE`), `:1216`
  `get_document`, `:1229` `search_documents` (embeds `query` via `ModelGateway`'s `embedding` kind,
  calls `search_chunks`), `:1266`-`:1307` `confirm_match`/`reject_match`/`recheck_match`/
  `list_pending_matches`/`list_matches` (thin passthroughs; `decided_by=ctx.actor`, never
  `'system'`, on the human/agent-driven path).
- `config.py:250` `CallContext(ws, actor)`; `:262` `get_context()` — **the single hardcoded-tenant
  auth/tenancy seam for M1.** Both REST and MCP resolve every call through this one function; MCP
  ignores any client-supplied `from`. **This is why FR-6/AC-7 access parity is already structural,
  not a design task**: there is no separate authorization tier anywhere in this codebase today. A
  new `Services` method, reached identically from `api.py`/`mcp.py` (each transport's own docstring:
  "no business logic lives here"), is parity-safe by construction — exactly as true today for every
  existing document/match method. Confirmed as a **finding**, below (§3.6), not designed as new
  machinery.
- `background.py` — `_safe_embed_chunk`/`_safe_extract`/`_safe_fuse`: established
  try/except-log-never-propagate discipline. `_report_document_job` is **`MATCH`-anchored** and
  silently no-ops if the `Document` is already gone — i.e. **deleting a document mid-background-
  processing degrades safely by construction**, because every background repository write is
  anchored on an id it does not assume still exists. **One exception, a genuine new race this plan
  must name (§7):** `ingestion.py:129`'s `create_entity_with_auto_match` call is an *unconditional*
  `CREATE`, not `MATCH`-anchored — so if a chunk is hard-deleted mid-extraction, the `Entity` still
  gets created, but the immediately-following `link_chunk_about_entity` (`MATCH`-anchored on
  `chunk_id`) silently no-ops. Net effect of a delete racing an in-flight extraction: a rare,
  harmless-but-real orphaned `Entity` with no `ABOUT` edge from any surviving chunk, extracted from
  content that is supposed to be gone.
- `mcp.py`/`api.py` — thin adapters (`@mcp.tool()` / `@router.<verb>(...)`); route-registration
  order only matters for literal-vs-dynamic collisions on the **same segment count**
  (`/documents/search` before `/documents/{document_id}`, `/documents/batch` likewise) — a new
  `DELETE /documents/{document_id}` (different verb) or `/documents/{document_id}/history` (a
  longer path) needs no such care. `app.py:82` `_register_error_handlers` — a `ServiceError`
  subclass maps to 404 only if it's in the `(ChannelNotFoundError, ThreadNotFoundError,
  MatchNotFoundError)` tuple, else 400; other error classes get their own explicit handler.
  `schemas.py:30` `MAX_ID_LEN=200` — the existing defensive path-param bound, already used for
  `match_id` (`Path(..., min_length=1, max_length=MAX_ID_LEN)`) — reused for `document_id`.
- `scripts/bootstrap_schema.sh` — exact DDL idioms to mirror: `CREATE INDEX FOR ()-[r:SAME_AS]-()
  ON (r.matchId)` / `(r.status)`, `UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1 matchId`.

### 2.2 The storefront reset feature's real hard-delete precedent (`docs/QUERIES.md` §18)

`DETACH DELETE` is an **established, live-verified, atomic (one `GRAPH.QUERY`), safe pattern in
this codebase already** — not a novel mechanism this plan invents. `docs/QUERIES.md` §18.6's
"keep/delete inventory" table for the storefront reset feature already documents, for an unrelated
feature, that `Document`/`Chunk`/`Entity` are "survivors of both [resets]; never matched" — i.e.
this exact feature's node types are already reasoned about by that design as untouched by a
different real-delete path. §18.6/§18.7 also establish the precedent this plan leans on directly
for the entity/relationship cascade decision (§3.5): **an edge disappears when its endpoint is
deleted, but a surviving node/edge may keep a property recording an id that no longer resolves** —
documented there as an accepted residual (`ReadCursor`'s own case), not silently swept under the
rug. §18.7 also documents the "client-side timeout ≠ nothing changed, re-read and report" posture
for a reset whose failure boundary is client-side — the same posture this plan adopts if
`delete_document` ever needs two round trips instead of one (§3.5).

---

## 3. Design & rationale

### 3.1 Two orthogonal axes, again (this plan's key design insight, mirroring the original feature's §3.1)

`Document.status` (`processing`/`ready`/`failed`) is **background-pipeline completion state** —
untouched by this feature. **Version-lifecycle state is a new, orthogonal axis**:
`Document.currentVersion: bool` (`true` at creation, flipped `false` only when superseded). These
are deliberately **not** conflated into one enum — "a `failed` document that later gets superseded"
must stay independently expressible, and a third state bolted onto `status` would make it not.
**Deletion has no property state at all.** No third boolean value, no enum member — the node's
non-existence *is* "deleted." The audit trail (FR-8) lives on a separate node, not a property on
the (now-gone) `Document` (§3.5).

### 3.2 Versioning shape — reuse the `SAME_AS` idiom exactly, at `Document` granularity

```
(:Document {new})-[:SUPERSEDES {
  matchId, status, confidence, technique,
  createdAt, decidedAt, decidedBy,
  resuggestCount, lastResuggestedAt
}]->(:Document {old})
```

Direction `new → old` (mirrors `SAME_AS`'s `new → existing` convention exactly). `status ∈
{pending, confirmed, rejected}`. **Every `SUPERSEDES` query matches its endpoints unlabeled**
(`(a)-[r:SUPERSEDES {...}]->(b)`, never `(a:Document)-[r:SUPERSEDES...`) — the same planner-trap
discipline `SAME_AS` queries already follow (§2.1), which applies identically here since it's the
same FalkorDB build and the same relationship-index-anchored query shape.

Denormalized onto the **old** `Document` directly, for the hot single-row read (`get_document`)
with no traversal: `Document.supersededAt`, `Document.supersededBy` — mirrors the existing
`decidedAt`/`decidedBy`-on-the-edge convention (audit fields) *and* the `Thread.updatedAt`-style
hot-read denormalization precedent. **Deliberately not** denormalizing "supersedes `<oldDocId>`"
onto the *new* Document — no hot-read need for it (`get_document_history`, §3.7, already has to
traverse the edge chain regardless of any denormalized pointer) — contrast this explicitly with
why `Chunk.documentCurrent` (§3.3) *does* need denormalizing: that one backs a genuinely hot,
per-row ANN-result read; this one would only ever back an already-traversal-bound history read. A
deliberate asymmetry, not an oversight.

### 3.3 Default search must exclude superseded content (AC-4)

`Chunk.documentCurrent: bool` — denormalized (mirrors `Chunk.documentId`'s existing "navigation
metadata, not a lookup key" precedent exactly), set `true` at chunk creation, flipped `false` in
the **same atomic write** that supersedes the owning `Document` (one `FOREACH` bulk `SET` across
`(:Document)-[:HAS_CHUNK]->(:Chunk)`). `search_chunks` gains `WHERE seed.documentCurrent = true`,
evaluated on the already-ANN-yielded (bounded, `k`-sized) rows post-`YIELD` — a plain property
equality filter, **not** an `exists()` pattern (flagged as buggy on this build in
`claude/graph-dba/falkordb-quirks.md`) and not a pre-filter (this build's vector index has no
pre-filter predicate support demonstrated anywhere in this codebase's usage so far).

**A necessary follow-on change, not just a filter bolt-on:** `services.search_documents` today
calls `search_chunks` with `k=limit` (no over-fetch — its own docstring: "there is no downstream
scope traversal to over-fetch for... the ANN fan-out IS the result set"). Once a post-filter can
discard rows, that reasoning no longer holds — a workspace with many superseded chunks ranking
highly could under-fill below `limit` with no over-fetch. **`search_documents` must over-fetch**
(e.g. `k = limit * 2`, an implementer-tunable multiplier, not load-bearing) exactly the same
"over-fetch, then filter" idiom `hybrid_search` already uses for its own scope filtering — a
concrete, testable behavior change this plan is naming explicitly, not leaving implicit.

`get_document(old_id)` is **unchanged** for a superseded version — the node still exists
(`currentVersion=false`), so direct-by-id lookup still returns the full row (FR-5/AC-5). Only
default *search* excludes non-current content, never direct lookup.

### 3.4 Update mechanics — auto tier (FR-2/AC-1) and suggested tier (FR-2/AC-2/FR-3/AC-3)

Per the ML note (§0): the auto tier is **synchronous**, folded into one new atomic repository
call, mirroring `create_entity_with_auto_match` precisely:

```python
def create_document_with_auto_supersede(
    ws, *, document_id, title, text, text_normalized_hash, source_format,
    ingested_by, created_at, chunks, match_id,
) -> dict:
    """One atomic GRAPH.QUERY: (1) OPTIONAL MATCH the existing currentVersion
    Document sharing textNormalizedHash, (2) CREATE the new Document + Chunks
    (documentCurrent: true per chunk, currentVersion: true), (3) two SEPARATE
    FOREACH-conditional blocks sharing one WITH — the exact shape graph-dba
    live-verified (§0), not one FOREACH mixing SET+CREATE: block one flips
    the candidate's currentVersion to false + supersededAt/supersededBy=
    'system' + bulk-flips all its Chunks' documentCurrent to false; block two
    CREATEs the SUPERSEDES{status:'confirmed', decidedBy:'system',
    confidence:1.0, technique:'exact_normalized_text_hash'} edge new->old.
    This is the form that most closely mirrors create_or_reopen_match's own
    idiom (separate guarded FOREACH blocks per concern, not one doing double
    duty). Returns {documentId, chunkCount, autoSuperseded,
    supersededDocumentId, matchId}.
    """
```

**Live-verified** (`graph-dba`, §0): a `threading.Barrier` concurrency probe against this exact
shape — two near-simultaneous calls sharing the same `textNormalizedHash` — produced exactly one
confirmed `SUPERSEDES` edge, never zero, never duplicated, mirroring
`test_create_entity_with_auto_match_concurrent_calls_produce_exactly_one_edge`'s own result.

**Why one atomic write, not three round trips** — this is a *direct, deliberate* application of
the lesson `document-ingestion.md` §3.4's concurrency note already paid for once: two
near-simultaneous ingests of the same edited content (a batch resubmitting duplicates, or two
concurrent MCP calls) could each run the hash-equality lookup before either sibling's `Document`
commits, silently defeating the auto tier's "no confirmation needed" guarantee — the exact race
shape `create_entity_with_auto_match` was built to close. Designing this atomically **from the
start** avoids reproducing that mistake and needing a second fix pass.

**`services.ingest_document`'s new flow:**
1. Compute `text_normalized_hash = update_detection.content_hash(update_detection.normalize_text(text))`
   app-side (mirrors how `nameNormalized` is computed app-side by the caller, not inside
   `repository.py`).
2. Call `repository.create_document_with_auto_supersede(...)` — replaces today's plain
   `repository.create_document` at this call site (mirrors `create_entity`/
   `create_entity_with_auto_match`'s exact relationship: the plain primitive stays in
   `repository.py`, unmodified beyond the two baseline properties in §3.5.3, but is no longer this
   call site once this stage lands).
3. If **not** `autoSuperseded`: schedule the suggested-tier detection job (below) alongside
   embed/extract scheduling.
4. Return the receipt — extended with `autoSuperseded`/`supersededDocumentId` so a caller sees
   immediately, in the same response, when their ingest just superseded something (a nicety beyond
   AC-1's literal requirement, satisfying it transparently rather than only as a side effect a
   caller must separately discover).

**Suggested tier — asynchronous, per the ML note's cost/latency (not embedding-dependency)
reasoning:** a new background peer, `_safe_detect_update` (mirrors `_safe_extract`/
`_safe_embed_chunk`'s try/except-log-never-raise discipline), scheduled once per document (not per
chunk, unlike embed/extract) right after the synchronous write returns. It:
1. Narrows candidates cheaply first — never an O(n) pairwise scan (per the ML note §4.1), via
   **two unioned, independent candidate-generation signals**, each a small shortlist (e.g. limit
   5, mirroring the entity tier's own candidate cap), scoped to `currentVersion` documents only,
   **never** actor-scoped (per the note §4.1), de-duplicated before ranking:
   - **LSH/MinHash-banding fingerprint match** (`graph-dba`-redesigned, §0/§7 — replaces this
     plan's original raw-fulltext-on-`Document.text` recommendation, measured too RAM-heavy):
     `update_detection.minhash_signature`/`lsh_bands` (below) turn the same shingle set the
     Jaccard computation itself uses into a small, fixed number of banded fingerprint values,
     each stored as its own short indexed `Document.lshBand0..bandB` property (§4 Stage D DDL).
     Candidate generation becomes an index-anchored `OR` across those band-equality predicates —
     orders of magnitude cheaper than a raw full-text scan of up to 500,000-char documents, the
     standard technique for this exact near-duplicate-detection shape.
   - **Title-fuzzy match** (RediSearch on `Document.title`, when non-empty) — a cheap
     complementary booster, not a replacement for the fingerprint signal, since title alone is too
     often absent/generic to carry candidate generation by itself (the note's F4). Unaffected by
     the RAM finding above (titles are short, `MAX_NAME_LEN=200`).
2. Computes `update_detection.jaccard(shingles, shingles)` in Python against the (unioned,
   de-duplicated) shortlist only (bounded, since the shortlist is small even though individual
   documents can be large) — the LSH bands only ever narrow *candidates*; the stored
   `SUPERSEDES.confidence` is still the precise raw Jaccard ratio the ML note specifies (§0), not
   an LSH estimate.
3. For the top-ranked candidate above an implementer-tunable noise floor: calls
   `repository.create_or_reopen_supersede_suggestion(...)` — **the exact same idiom as
   `create_or_reopen_match`**, applied verbatim at `Document` granularity (guarded find-or-create-
   or-reopen on an undirected `OPTIONAL MATCH`, not a bare `MERGE`). This gets the ML note's §7
   recommendation — auto-reopen-on-corroboration for a previously-`rejected` suggestion — **for
   free**, with zero new logic: it is the identical write shape `SAME_AS` already uses, so a later
   re-derivation of the same pair reopens the existing edge (bumping
   `resuggestCount`/`lastResuggestedAt`) rather than duplicating it, exactly as OQ-3 already
   resolved for entities.

**`Document.status`/`pendingJobs` is explicitly NOT touched by this job** (per the ML note's own
scope note): a detection failure is a *soft* failure — the document is still valid and
independently searchable — and must not flip the whole document to `'failed'` the way an
extraction/embedding failure does. `_schedule_update_detection` is therefore a **separate**
scheduling call from `_schedule_chunk_processing` (which owns the `pendingJobs` counter), not
folded into its per-chunk loop — detection is per-*document*, and its completion signal is simply
the presence/absence of a pending `SUPERSEDES` edge, not a `Document.status` value.

**FR-3/AC-3 (confirm/reject) — full `SAME_AS`-lifecycle parity, not a subset:** `confirm_document_
update`/`reject_document_update`/`recheck_document_update` mirror `confirm_match`/`reject_match`/
`recheck_match` verbatim (same guard-and-stamp shape, `decidedBy=ctx.actor` never `'system'` on
this path). **Reversing this plan's own first-draft instinct to omit `recheck_document_update`
for "no FR literally asks for it"** — adopted per the ML note's §7 recommendation: since the edge
already carries `resuggestCount`/`lastResuggestedAt` (inherited by directly copying `SAME_AS`'s
property set, per the note's F6), leaving those fields permanently unset/unused would be an
inconsistency in the shipped schema, not a simplification. Confirming a `SUPERSEDES` edge flips the
old document non-current (mirrors the auto-tier's own flip, §3.3) and bulk-flips its chunks;
rejecting leaves both documents independent and current (AC-3's own wording).

### 3.5 Delete (FR-4/AC-6/AC-8) — real hard delete, one atomic write where possible

```
MATCH (d:Document {documentId: $documentId})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE d, c
```

mirrors the storefront reset's exact "atomic, one query, `DETACH DELETE`" precedent (§2.2). This
structurally removes `Chunk`→`ABOUT`→`Entity` edges (they die with the `Chunk`) but leaves `Entity`
nodes and `RELATES_TO` fact edges completely untouched — not a policy choice at that level, it
falls directly out of the graph shape (§3.6 covers the actual policy decision, which is at the
entity/relationship level, not the chunk/document level).

**Audit trail (FR-8/AC-8):** a **separate** node, since nothing can point at a `DETACH DELETE`d
node — `(:DocumentDeletion {documentId, deletedBy, deletedAt})`, keyed by the (now-gone)
`documentId` value, not a graph reference. `get_document(deleted_id)` returns `None` (unchanged
codepath — the `MATCH` simply finds nothing); a new `get_document_deletion(document_id)` answers
FR-8/AC-8 directly. **Recommended single-query shape** (capture `d.documentId` into a plain scalar
via `WITH` *before* the delete, then `CREATE` the audit node referencing that scalar, not the
now-deleted node):

```
MATCH (d:Document {documentId: $documentId})
WITH d, d.documentId AS did
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
WITH d, did, collect(c) AS chunks
FOREACH (ch IN chunks | DETACH DELETE ch)
DETACH DELETE d
CREATE (:DocumentDeletion {documentId: did, deletedBy: $deletedBy, deletedAt: $deletedAt})
RETURN did AS documentId
```

**Live-verified** (`graph-dba`, §0): this exact single-query shape — `MATCH` → capture
`documentId` via `WITH` → collect+`FOREACH`-delete the chunks → `DETACH DELETE d` → `CREATE` the
`DocumentDeletion` audit node → `RETURN` — ran verbatim and succeeded outright, one round trip,
`Document`+`Chunk`s confirmed gone and the audit node correct. **Build this single-query form
directly — no two-query fallback is needed** (this plan's earlier draft hedged with one as an
accepted degradation path; that hedge is now moot and dropped). If `delete_document` ever needed a
second round trip for some unrelated reason, the storefront reset's own §18.7 still establishes how
to reason about a client-side-timeout failure boundary between two such steps ("re-read and
report," never assume "nothing changed") — noted for completeness, not because this plan needs it.

`delete_document(ctx, document_id)` raises a new `DocumentNotFoundError` (added to `app.py`'s
404-mapped tuple) when nothing matched — mirrors `MatchNotFoundError`'s existing shape/posture.

### 3.6 Entity/relationship cascade (Open Question 1) — resolved

**On UPDATE (supersede): zero cascade, by construction.** The superseded document's `Chunk`s still
exist (only `documentCurrent` flips) — nothing about extraction or fusion changes at all. This
falls directly out of "never destroy, only reflag" and needs no new code whatsoever. It is a direct
extension of the original feature's already-settled axis (`document-ingestion.md` §3.1): fact/
relationship provenance never merges or retracts.

**On DELETE: recommend no cascade either.** `Entity` nodes and `RELATES_TO` edges survive a hard
delete untouched, by construction (§3.5). `RELATES_TO` edges whose `sourceChunkId`/
`sourceDocumentId` named the now-deleted subtree become historical, orphaned provenance pointers —
an **accepted, named residual**, directly precedented by `docs/QUERIES.md` §18.6/§18.7's own
"an edge disappears when its endpoint dies, a surviving node/edge keeps recording an id that no
longer resolves" pattern (the storefront reset's documented `ReadCursor` residual).

**Rejected alternative: cascade-prune orphaned entities** (delete an `Entity` if, after the chunk
delete, it has zero remaining `ABOUT` edges). Rejected because: (a) it is new destructive graph
surgery this codebase's fusion design deliberately avoids everywhere else (never physically
merge/delete fused knowledge, `document-ingestion.md` §3.4); (b) it needs an extra per-entity
existence check per deleted chunk — real added cost for a deliberately rare, low-frequency action;
(c) an `Entity` that is `SAME_AS`-confirmed with a survivor sourced from *another* document would
need special-casing to avoid wrongly being pruned even though only one of its sourcing chunks
died — meaningful extra complexity with no FR forcing it. **Named as an explicit, revisitable v2
"purge orphaned knowledge" maintenance operation, not a silently-dropped idea** — flagged in §7 for
stakeholder visibility, mirroring how embedding-based entity matching was named-and-deferred rather
than silently dropped in the original ML note.

### 3.7 Supporting capabilities: list + history

Not literal FRs, but needed to exercise FR-4/FR-5/FR-8 at all — precedented by how the original
plan added `list_matches` beyond FR text to close a discoverability gap the plan-gate review found.

- `list_documents(ws, *, current_only=True, limit=50) -> list[dict]` — summaries
  (`documentId, title, sourceFormat, sourceKind, status, currentVersion, createdAt,
  ingestedByKind, ingestedById`), mirrors `list_matches`'s `ORDER BY createdAt`/`LIMIT` shape,
  `current_only` toggled via the same "two separate query strings, not a null-guarded `WHERE`"
  index-preserving idiom.
- `get_document_history(ws, *, document_id) -> list[dict]` — every version in the confirmed
  `SUPERSEDES` chain (oldest → newest), each row's own `{documentId, title, createdAt,
  currentVersion, supersededAt, supersededBy}`. Works from **any** version's id in the chain, not
  only the current tip — traverses `SUPERSEDES` in both directions, **confirmed edges only** (a
  `pending`/`rejected` suggestion is not "history," only ever a resolved supersession is).

### 3.8 MCP/REST/Services surface

| MCP tool | REST | Service method | Notes |
|---|---|---|---|
| `delete_document(document_id)` | `DELETE /documents/{id}` | `delete_document` | FR-4, explicit-id only |
| `list_documents(current_only=True, limit=50)` | `GET /documents` | `list_documents` | supporting, §3.7 |
| `get_document_history(document_id)` | `GET /documents/{id}/history` | `get_document_history` | FR-5 |
| `get_document_deletion(document_id)` | `GET /documents/{id}/deletion` | `get_document_deletion` | FR-8 |
| `list_pending_document_updates(limit=50)` | `GET /document-updates/pending` | `list_pending_document_updates` | OQ-3 (reuses original OQ-2 answer) |
| `list_document_updates(status=None, limit=50)` | `GET /document-updates` | `list_document_updates` | audit parity w/ `list_matches` |
| `confirm_document_update(match_id)` | `POST /document-updates/{id}/confirm` | `confirm_document_update` | FR-3/AC-3 |
| `reject_document_update(match_id)` | `POST /document-updates/{id}/reject` | `reject_document_update` | FR-3/AC-3 |
| `recheck_document_update(match_id)` | `POST /document-updates/{id}/recheck` | `recheck_document_update` | §3.4's reversed scope call |

**No explicit `update_document(document_id, new_text)` tool.** The requirements doc's own decision
log forecloses explicit-ID addressing as the primary update path ("System detects it automatically,
not explicit-ID addressing"), and FR-4's own contrast ("unlike FR-2's update path" for delete being
explicit) confirms update is never explicit-ID-based. The existing `ingest_document`/
`ingest_documents` tools are **unchanged in signature** — detection is new server-side behavior
riding the same call.

**Where a pending suggestion surfaces (Open Question 3) — resolved by direct reuse of the original
feature's OQ-2 answer.** A dedicated review surface (`list_pending_document_updates`, reachable via
both MCP and REST), not a channel message — the original feature's rejected alternative reasoning
transfers unchanged: a pending document-update suggestion has no natural channel/thread anchor
(document ingestion happens per-workspace, not per-conversation), and conflating knowledge-base
curation with chat would undermine the "ingested-content search stays separate from chat search"
posture (FR-14 of the original feature) this plan has no reason to revisit.

**Route ordering:** `GET /documents` (list) is a distinct literal path from `GET
/documents/{document_id}` — no ambiguity. `GET /documents/search` stays registered before `GET
/documents/{id}` (unchanged, existing rule). `DELETE /documents/{id}` is a different HTTP verb —
no ordering concern with the existing `GET`. `/documents/{id}/history` and `/documents/{id}/deletion`
are two segments past the prefix — no collision with the existing one-segment `/documents/{id}`.

**New error classes:** `DocumentNotFoundError(ServiceError)` (get/delete/history/deletion lookups
on an unknown or deleted `documentId`) and `DocumentUpdateNotFoundError(ServiceError)` (confirm/
reject/recheck on an unknown `SUPERSEDES` `matchId` — a distinct class from `MatchNotFoundError`
rather than reused, since "match" in that class's existing error message would be a confusing,
wrong term for a document-update suggestion). Both added to `app.py`'s 404-mapped tuple.

### 3.9 Access parity (FR-6/AC-7) — a finding, not a design task

Already covered in §2.1: `get_context()` is the single hardcoded-tenant seam every call resolves
through, on both transports, with no separate authorization tier anywhere in this codebase today.
Every method above just needs to be an ordinary `Services` method, invoked identically (no
business logic) from `api.py`/`mcp.py` — parity is automatic by construction, exactly as it already
is for every existing document/match method. No new mechanism is proposed or needed.

---

## 4. Step-by-step implementation

Staged so the tree stays buildable and each stage is independently testable; only Stage D needs
the ML note's conclusions (already landed, §0).

### Stage A — schema baseline + delete + list + deletion audit (zero dependency on detection)

**Files:**
- `scripts/bootstrap_schema.sh` — add: `Document.currentVersion` RANGE index; `DocumentDeletion`
  RANGE index + `UNIQUE NODE DocumentDeletion PROPERTIES 1 documentId` on `documentId`.
  `Chunk.documentCurrent` needs **no** index (same "navigation metadata, only ever read on
  bounded ANN-yielded rows" posture as `Chunk.seq`/`Chunk.documentId`).
- `server/falkorchat/repository.py` — `create_document` gains two baseline properties every
  document needs regardless of this feature's ML tiering: `currentVersion: true` on the `Document`
  `CREATE`, `documentCurrent: true` per `Chunk` in its `FOREACH`. New: `delete_document(ws, *,
  document_id, deleted_by, deleted_at) -> bool` (the §3.5 query, live-verified single-query form);
  `get_document_deletion(ws, *, document_id) -> dict | None`; `list_documents(ws, *,
  current_only=True, limit=50) -> list[dict]` (§3.7).
- `server/falkorchat/services.py` — `delete_document(ctx, *, document_id) -> dict` (raises
  `DocumentNotFoundError` on no-op); `get_document_deletion(ctx, *, document_id) -> dict | None`;
  `list_documents(ctx, *, current_only=True, limit=50) -> list[dict]`.
- `server/falkorchat/mcp.py`/`api.py`/`schemas.py` — the three tools/routes above; `app.py` gains
  `DocumentNotFoundError` in the 404-mapped tuple.

**Done:** a document can be listed, deleted (content genuinely gone, `get_document` → `None`), and
its deletion is auditable (`get_document_deletion`) — AC-6/AC-8 provable in isolation, before any
detection code exists.

### Stage B — `SUPERSEDES` DDL + version history (still no detection dependency)

**Files:**
- `scripts/bootstrap_schema.sh` — `CREATE INDEX FOR ()-[r:SUPERSEDES]-() ON (r.matchId)` / `(r.status)`;
  `UNIQUE RELATIONSHIP SUPERSEDES PROPERTIES 1 matchId`.
- `server/falkorchat/repository.py` — `get_document_history(ws, *, document_id) -> list[dict]`
  (§3.7 — the confirmed-`SUPERSEDES`-chain traversal); `confirm_document_update`/
  `reject_document_update`/`recheck_document_update`/`list_pending_document_updates`/
  `list_document_updates` (verbatim mirrors of the `SAME_AS` equivalents, §3.4).
- `server/falkorchat/services.py`/`mcp.py`/`api.py`/`schemas.py` — the corresponding thin
  passthroughs + tools/routes (§3.8); `DocumentUpdateNotFoundError` added to `app.py`'s tuple.

**Done:** the confirm/reject/recheck/list plumbing for document-update suggestions is buildable
and testable with **directly-injected `matchId`s** (a test fixture writing a `SUPERSEDES` edge by
hand), before any detection code exists to produce one organically — the same "build the
mechanism, test it standalone, wire detection in last" sequencing the original entity-fusion plan
itself used (Stage 3 before Stage 4).

### Stage C — default-search filtering (AC-4)

**Files:**
- `server/falkorchat/repository.py` — `search_chunks` gains `WHERE seed.documentCurrent = true`
  (§3.3).
- `server/falkorchat/services.py` — `search_documents` over-fetches (`k = limit * 2` or similar,
  §3.3) before filtering/limiting.

**Done:** AC-4 provable with a raw test fixture that flips one document's `currentVersion` to
`false` directly (no detection needed yet) — search excludes it, `get_document` on it directly
still succeeds.

### Stage D — update-detection (FR-2/AC-1/AC-2) — needs the ML note (landed, §0)

**Files:**
- `server/falkorchat/update_detection.py` — new, mirrors `fusion.py`'s shape: `normalize_text(text)
  -> str` (case-fold + whitespace-collapse — the **one** shared normalizer for both the hash and
  the shingling, per the ML note's own instruction not to write two independently-drifting
  normalizers, mirroring `extraction.normalize_name`'s existing "one shared helper" precedent);
  `content_hash(normalized_text) -> str`; `shingles(normalized_text, n=5) -> set[str]` (5-word
  n-grams); `jaccard(a: set[str], b: set[str]) -> float`. **Two additions per `graph-dba`'s
  redesign of the candidate-generation index (§0/§3.4/§7):** `minhash_signature(shingles: set[str],
  k: int = 32) -> list[int]` (a standard MinHash signature over the shingle set — the same set
  `shingles()` already builds, no second document representation); `lsh_bands(signature: list[int],
  b: int = 8) -> list[str]` (bands the signature into `b` short fingerprint strings, one per
  `Document.lshBand<i>` property — `k`/`b` are an implementer-tunable RAM/recall trade-off,
  mirroring this plan's existing posture toward heuristic constants like `MAX_DOCUMENT_CHARS`, not
  load-bearing).
- `scripts/bootstrap_schema.sh` — `CREATE INDEX FOR (n:Document) ON (n.lshBand<i>)` for each of the
  `b` band properties (plain RANGE/equality indexes, per `graph-dba`'s recommendation, §0); `CALL
  db.idx.fulltext.createNodeIndex('Document', 'title')` (new — titles are short, `MAX_NAME_LEN=200`,
  unaffected by the fulltext-on-`text` RAM finding that ruled out the original candidate-index
  design, §0).
- `server/falkorchat/repository.py` — new `create_document_with_auto_supersede` (§3.4, two separate
  `FOREACH`-guarded blocks per graph-dba's verified shape, replaces `create_document` at
  `ingest_document`'s call site only — `create_document` itself stays, unmodified beyond Stage A's
  two baseline properties, as the underlying plain-create primitive, mirroring `create_entity`/
  `create_entity_with_auto_match`'s exact relationship); new `find_update_shortlist(ws, *, bands:
  list[str], title, limit=5) -> list[dict]` — an index-anchored `OR` across the `b` band-equality
  predicates (`WHERE d.lshBand0 = $band0 OR d.lshBand1 = $band1 OR ...`), unioned app-side with a
  separate title-fuzzy full-text lookup, **not yet independently live-verified** (§7 — the general
  "`OR`-as-scan-anchor" quirk category is already named in `falkor-chat/AGENTS.md`'s live-verified-
  facts list, so this specific multi-property-`OR` shape needs its own check, not an assumption
  that it behaves like the single-predicate cases already verified elsewhere in this plan);
  `create_or_reopen_supersede_suggestion(ws, *, new_document_id, candidate_document_id, match_id,
  status, confidence, technique, created_at) -> dict` (verbatim mirror of `create_or_reopen_match`,
  also not yet independently live-verified at `Document` granularity, §7 — only the entity-level
  original has been).
- `server/falkorchat/services.py` — `ingest_document`'s new flow (§3.4): compute the hash, call
  the atomic method, schedule suggested-tier detection only when not auto-superseded, return the
  extended receipt.
- `server/falkorchat/background.py` — `_safe_detect_update` (mirrors `_safe_extract`'s isolation
  discipline, **without** `_report_document_job` — a detection failure is a soft failure, §3.4);
  `_schedule_update_detection` (a **separate** scheduling call from `_schedule_chunk_processing`,
  since detection is per-document, not per-chunk).
- `server/falkorchat/mcp.py`/`api.py` — schedule `_safe_detect_update` alongside the existing
  per-chunk embed/extract scheduling, right after `ingest_document`'s synchronous write returns.

**Done:** AC-1 provable synchronously (two ingests of byte-identical-modulo-whitespace content →
immediate auto-supersede, no confirmation); AC-2 provable once the background job runs (a
plausible-but-not-identical edit → a pending `SUPERSEDES` suggestion, old document still current
and searchable).

### Stage E — QA acceptance pass

- Confirm the full AC-1..AC-8 matrix against a real multi-version, multi-actor fixture (an edit
  cycle: ingest → auto-supersede → history lookup → a separate suggested-tier edit → confirm →
  reject-then-recheck → delete).
- `qa-engineer` acceptance pass, mirroring the K-050 pattern (`docs/test-plans/document-
  ingestion2.md`, `docs/test-reports/document-ingestion2-report.md`) — including the ML note's own
  named v1 limitation (a heavily-rewritten edit that also changes its title enough to escape both
  candidate-generation signals silently becomes an independent document, today's exact behavior)
  as an explicit known-gap test case, not an undiscovered defect.

---

## 5. Test strategy

| AC | What proves it | Altitude |
|---|---|---|
| AC-1 (auto-supersede, no confirmation) | Ingest, then re-ingest byte-identical-modulo-whitespace content; assert old `currentVersion=false`+`supersededAt`/`supersededBy='system'`, new `currentVersion=true`, `SUPERSEDES{status:'confirmed',decidedBy:'system',confidence:1.0}`, search returns only the new version | repository/service integration |
| AC-2 (suggested, not superseded) | Ingest a plausible-but-not-identical edit; assert `SUPERSEDES{status:'pending'}` after the background job runs, old document still `currentVersion=true` and searchable | repository/service integration, background-completion-aware |
| AC-3 (confirm/reject) | `confirm_document_update`/`reject_document_update` on a pending suggestion; assert the confirm branch supersedes (old non-current, chunks flipped) and the reject branch leaves both independent+current+searchable | service/API contract |
| AC-4 (search excludes superseded) | Direct `search_chunks`/`search_documents` assertion after a supersession — never returns the old version's chunks | repository/service integration |
| AC-5 (history retrievable) | `get_document_history` and direct `get_document(oldId)` both return the prior version's full text after supersession | repository/service integration |
| AC-6 (deleted content gone by any means) | `get_document` → `None`, `search_documents`/`list_documents` exclude it (chunks/node genuinely gone, not merely filtered) | repository/service integration |
| AC-7 (access parity) | Per-transport coverage in both `test_api.py` and `test_mcp.py` for each new method (mirrors the existing pattern — e.g. `test_confirm_match_tool`/`test_confirm_match_route` — there is no single cross-transport test in this suite today; parity is proven by both transports' independent suites asserting identical `Services`-layer behavior, not a bespoke parity harness) | REST + MCP integration |
| AC-8 (delete audit trail) | `get_document_deletion` returns `{deletedBy, deletedAt}` after `delete_document`, even though `get_document` on the same id is `None` | repository/service integration |

**Additional, non-AC-mapped test coverage:**
- `update_detection.normalize_text`/`content_hash`/`shingles`/`jaccard` — pure unit tests (empty
  text, whitespace-only variance, identical-modulo-case-and-whitespace pairs, two genuinely
  unrelated documents sharing heavy boilerplate — a hard-negative case named directly in the ML
  note §6).
- **Concurrency regression test for the atomic auto-supersede write** — real threads, a
  `threading.Barrier`, mirroring `test_create_entity_with_auto_match_concurrent_calls_produce_
  exactly_one_edge` exactly: two near-simultaneous `create_document_with_auto_supersede` calls
  against the same content-hash must produce exactly one confirmed `SUPERSEDES` edge, never zero,
  never duplicated.
- **Delete-during-in-flight-background-processing race** — delete a document while its
  extraction/embedding jobs are still scheduled; assert no crash, no corrupted `Document` (it's
  simply gone), and name the one accepted residual explicitly (§2.1's orphaned-`Entity` race —
  assert it stays rare/harmless, not that it cannot occur, since this plan does not propose fixing
  it, §7).
- **Cascade-non-effect regression test (Open Question 1)** — explicitly assert `Entity`/
  `RELATES_TO` survive both an update (trivially — nothing touches them) and a delete (the
  accepted-residual case, §3.6) — a **regression** test for the design decision itself, not just
  design-note prose, so a future change doesn't silently start cascading without a deliberate
  decision to do so.
- **`SUPERSEDES` reopen-on-corroboration** — mirrors the existing `create_or_reopen_match` test
  shape: reject a suggestion, then re-derive the same pair; assert reopen to `pending` (never
  straight to `confirmed`), `resuggestCount` bumped on the **original** `matchId`, no duplicate
  edge.
- **`test_queries.sh` additions** — every new Cypher shape (the atomic auto-supersede write, the
  `SUPERSEDES` find-or-reopen write, the delete query, the history traversal, the candidate
  shortlist query) raises the enumerated baseline — exact queries are `graph-dba`'s to author/
  verify (§7), not enumerated here.

---

## 6. RAM/scale considerations (rule 6)

- **`Document.textNormalizedHash`** — one short fixed-length string property per `Document` (e.g.
  a 64-char hex SHA-256 digest) plus a RANGE index entry — negligible against the `Chunk.embedding`
  line that already dominates this feature's RAM budget (`document-ingestion.md` §6).
- **`Chunk.documentCurrent`** — one boolean property per `Chunk`, no index — negligible.
- **`SUPERSEDES` edges** — same per-edge shape/cost as `SAME_AS` (`document-ingestion-graph.md`
  §1.2/§6 measured ~840 bytes/edge) — document-update volume is bounded by ingestion frequency, not
  extraction fan-out, so this line grows far more slowly than `Entity`/`SAME_AS` already does.
- **`DocumentDeletion` nodes** — one small node per deletion, a deliberately rare action — negligible.
- **The suggested tier's candidate-generation index — measured, not assumed, and redesigned as a
  result (`graph-dba`, §0).** A raw RediSearch full-text index on `Document.text` at this plan's
  own 500,000-char ceiling was live-measured at **2-5x the raw text size in RAM** (0.27 MB/doc on a
  low-entropy 8k-token vocabulary, 2.45 MB/doc on a higher-entropy 60k-token one closer to real
  prose) — a real, non-negligible line stacking on the already-dominant `Chunk.embedding` line,
  workspace-wide. **Ruled out for that reason** and replaced with the LSH/MinHash-banding
  fingerprint (§3.4/§4 Stage D): a handful of short indexed properties per `Document`
  (`lshBand0..bandB`), each a small int/short string — negligible RAM, the standard technique for
  this exact near-duplicate-detection shape. `Document.title`'s fulltext index (new, §4 Stage D) is
  cheap and unaffected — titles are short (`MAX_NAME_LEN=200`), nowhere near the RAM-costly ceiling
  that ruled out indexing `Document.text` directly.
- **No new vector index of any kind** — the ML note's entire v1 recommendation is embedding-free at
  both tiers, so this feature adds zero vector-RAM growth, unlike the original feature's dominant
  `Chunk.embedding` line.

---

## 7. Risks & open questions

- **Verification gap — largely closed by `graph-dba`'s live-verification pass (§0), narrowed to
  what's actually left.** Four of this plan's five originally-by-analogy Cypher/index designs are
  now **confirmed live** against a real FalkorDB instance (throwaway `ws:docprobe`): the unlabeled-
  `SUPERSEDES`-endpoint planner-trap discipline (§3.2, for the match+`SET` shape *and* the
  status-filter/list shape), the atomic auto-supersede write including its concurrency fix (§3.4),
  the single-query delete (§3.5), the `Document.textNormalizedHash` index, and the `SUPERSEDES` DDL
  (§4 Stage B). **What genuinely remains unverified, narrowed from the original broad hedge:** the
  new `find_update_shortlist` band-equality `OR`-lookup (§4 Stage D — a brand-new query shape this
  plan is introducing in response to item 4b's redesign, not yet run against a real instance; the
  general "`OR`-as-scan-anchor" quirk category is already named in `falkor-chat/AGENTS.md`'s
  live-verified-facts list, so this specific multi-property-`OR` shape should not be assumed to
  behave like the single-predicate cases already confirmed elsewhere in this plan) and
  `create_or_reopen_supersede_suggestion` (a verbatim mirror of `create_or_reopen_match`, which
  *is* live-verified at entity granularity, but this plan's `Document`-granularity instance of it
  has not independently been). **Recommend a second, narrower `graph-dba` (or `coder`-run)
  live-verification pass covering just these two, before Stage D implementation** — not the full
  Stage-0-sized gate the first pass already closed.
- **The suggested tier's candidate-generation index — resolved, not merely flagged.** Originally
  left to `graph-dba` to finalize (per the ML note's §4.1/§7); now concretely redesigned as
  LSH/MinHash banding after the live RAM measurement ruled out a raw full-text index on
  `Document.text` (§0/§6) — folded into §3.4/§4 Stage D above, not an open item anymore.
  `Document.textNormalizedHash`'s indexing shape (RANGE, no uniqueness constraint — two genuinely
  different documents could, vanishingly unlikely, share a hash before either is superseded, so
  this deliberately does **not** follow the "every entity gets a uniqueness constraint" convention
  `falkor-chat/AGENTS.md` states — `textNormalizedHash` is a content fingerprint, not an identity
  property, so that convention doesn't apply to it, named here so a reviewer doesn't mistake the
  omission for a violation) is confirmed as designed. The K-049-family oversized-indexed-property
  crash concern this plan flagged for a direct check is **confirmed not reachable** (§0 — live
  500,000-char writes into a `RANGE`-indexed, unconstrained field, zero crash; this plan already
  proposed no `UNIQUE` constraint on either `Document.text` or `textNormalizedHash`, so that crash
  family was never reachable here by construction, now empirically confirmed rather than merely
  argued).
- **A genuine new race, named (§2.1): delete racing an in-flight extraction can leave a rare,
  orphaned `Entity` node** with no `ABOUT` edge from any surviving chunk — `create_entity_with_
  auto_match` is an unconditional `CREATE`, not `MATCH`-anchored, so a chunk deleted between that
  call and the following (`MATCH`-anchored, silently-no-op-safe) `link_chunk_about_entity` leaves
  the entity created but unlinked. Small blast radius (one extra node, rare timing window),
  consistent with this codebase's existing accepted-risk posture for other timing races
  (non-idempotent document creation on retry, `document-ingestion.md` §7) — not proposed as a
  blocker, named so it isn't mistaken for an undiscovered defect later.
- **The entity/relationship cascade decision (§3.6) is a considered, revisitable call, not a
  certainty.** The rejected alternative (cascade-prune orphaned entities on delete) is a real,
  legitimate v2 "purge orphaned knowledge" maintenance operation if the stakeholder later decides a
  deliberately-deleted document's extracted (and un-corroborated) knowledge should not linger
  disconnected in the graph — named explicitly, not silently foreclosed.
- **Auto-supersede's accidental side benefit** (per the ML note §3.1): because the auto tier is an
  *exact* content-hash match, a retried `ingest_document` call after a client-side timeout — the
  non-idempotent-creation risk `document-ingestion.md` §7 already names and accepts — now
  auto-supersedes its own accidental duplicate rather than leaving two independent, byte-identical
  documents in the graph forever. Not a complete fix for that risk (a client-side idempotency key
  would be more direct and is not designed here, since it wasn't asked for), but a real,
  unplanned-for improvement worth recording.
- **The ML note's named v1 limitation stands as an accepted gap, not a defect**: a substantively
  rewritten edit that also changes its title enough to escape both the title-fuzzy and
  content-fingerprint candidate-generation nets is not detected at all — it silently becomes an
  independent `Document`, exactly today's behavior. Flagged for `qa-engineer`'s test plan as a named
  known-gap scenario (§4 Stage E), not a regression to chase in v1.
- **No web UI work is planned** — every FR/AC here is reachable via MCP/REST, mirroring the original
  feature's own "no `web/` changes" scope decision. Flag to the coordinator if the stakeholder
  expects UI in this milestone.
