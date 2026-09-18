# Agent-team ingestion — `produced_by` attribution — graph design

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

Graph-side design companion to `claude/docs/plans/agent-knowledge-base-strategy.md` §4.1 (Track 1
Stage 1) — that plan's interface spec is settled (two independent review passes at the plan level);
this note is the concrete implementation design underneath it: exact signatures, exact Cypher
predicate change, the new exception class, and what test coverage the change needs. **Design only —
no code changes in this note.** `coder` implements directly from it.

Read against the current source, not the plan's own line-number citations (which may have drifted):
`falkor-chat/server/falkorchat/{services,repository,mcp,api,app,schemas}.py`, verified 2026-09-18.
`cpg_falkorchat` is stale for this task (predates `document-ingestion2`'s ship, per the coordinating
brief) — every finding below is from a direct read of current source, plus one live `EXPLAIN`
against `ws:demo` (§2.3).

---

## 1. What actually calls what today (the part the plan's prose compresses)

`Services.ingest_document` does **not** call `Repository.create_document` — it calls
`Repository.create_document_with_auto_supersede` (`services.py:1190`), the FR-2/AC-1 atomic
auto-supersede variant. The plain `create_document` (`repository.py:1036`) has **no production
caller** — grep confirms every call site left is a test fixture (`tests/test_repository.py`,
`test_services.py`'s `FakeRepo`, `test_api.py`, `test_mcp.py`, `test_graphrag.py`,
`test_provenance.py`, `test_tools.py`), none of them exercising `produced_by`.

**Scope decision:** extend only `create_document_with_auto_supersede` (and, one layer up,
`Services.ingest_document`/`ingest_documents` and their MCP/REST surface). Leave plain
`create_document` untouched — it is dead code from any Service/MCP/REST path, so there is no
`produced_by` caller for it to serve, and touching it would be unreviewable scope creep against a
method nothing in this feature (or any other shipped feature) reaches. If a future caller ever
needs `create_document` directly with `produced_by`, that is a small same-shape follow-up when it
actually has a caller, not now.

---

## 2. `Repository.create_document_with_auto_supersede` — exact change

### 2.1 Signature

```python
def create_document_with_auto_supersede(
    self, ws: str, *, document_id: str, title: str, text: str,
    text_normalized_hash: str, source_format: str, ingested_by: str,
    created_at: int, chunks: list[dict[str, Any]], match_id: str,
    produced_by: str | None = None,
) -> dict[str, Any]:
```

One new keyword-only parameter, appended last, defaulting to `None` — every existing call site
(production and test) compiles and behaves identically unchanged.

### 2.2 The resolution branch — two literal prefixes, not a null-parameter match

**Do not** resolve `produced_by` with an `OPTIONAL MATCH (pa:Agent {agentId: $producedBy})` inside
the *same* query text used when `produced_by` is `None` (i.e. never pass a possibly-`NULL`
`$producedBy` into one shared query). Live-verified today against `ws:demo` (§2.3 below): the
identical pattern-property match plans `Node By Index Scan` when the parameter carries a real
value and falls back to `Node By Label Scan` + `Filter` when the parameter is `NULL` — a different
failure shape from the already-documented `$param IS NULL OR prop = $param` idiom
(`claude/graph-dba/falkordb-quirks.md`, 2026-09-18 entry, added by this note), but the same fix:
branch in Python into two literal query strings, mirroring `hybrid_search`'s/`list_documents`'s
existing conditional-clause-text convention in this same file. `Agent` is a small label here so the
label-scan fallback would cost little regardless — but there is no reason to accept even a small,
avoidable regression when the two-branch shape is free and already this file's own convention.

Module-level constants (place beside the method, same file):

```python
_INGESTOR_RESOLVE_BY_ACTOR = (
    "OPTIONAL MATCH (u:User  {userId:  $ingestedBy}) "
    "OPTIONAL MATCH (a:Agent {agentId: $ingestedBy}) "
    "WITH coalesce(u, a) AS ingestor, (coalesce(u, a) IS NOT NULL) AS ok, "
    "     (u IS NOT NULL) AS ingestorIsUser "
)
_INGESTOR_RESOLVE_BY_PRODUCER = (
    "OPTIONAL MATCH (pa:Agent {agentId: $producedBy}) "
    "WITH pa AS ingestor, (pa IS NOT NULL) AS ok, false AS ingestorIsUser "
)
```

`ingestorIsUser` is the key move: it replaces carrying the raw `u`/`a` node bindings through the
rest of the query (the original does — see `WITH u, a, ingestor, ok, candidate` and the `sourceKind:
CASE WHEN u IS NOT NULL …` line two blocks later). Reducing "which node resolved" to one bool that
both branches can produce means **everything after the resolution prefix is byte-identical text in
both branches** — only the prefix and one `WITH` clause's carried-variable list change; the
`FOREACH`/`CREATE` blocks, the auto-supersede logic, and the `RETURN` are untouched.

### 2.3 Live-verified — the index-anchoring claim, not asserted

```
EXPLAIN CYPHER producedBy='demo' MATCH (d:Document {documentId:'x'})
  OPTIONAL MATCH (pa:Agent {agentId: $producedBy}) RETURN pa
→ ... Optional → Node By Index Scan | (pa:Agent) → Argument

EXPLAIN CYPHER producedBy=null MATCH (d:Document {documentId:'x'})
  OPTIONAL MATCH (pa:Agent {agentId: $producedBy}) RETURN pa
→ ... Optional → Filter → Node By Label Scan | (pa:Agent) → Argument
```

Run against `ws:demo` (2026-09-18), which already carries the demo `Agent` and a confirmed `RANGE`
index on `Agent.agentId` (`CALL db.indexes() … → {'agentId': ['RANGE']}`). Confirms both halves of
the design: (a) **no new index needed** — `produced_by` resolution reuses the exact same
`Agent.agentId` index every other agent-identity lookup in this codebase already hits
(`ensure_agent`, `post_first_message`'s author resolution, the pre-existing `ingested_by` branch);
(b) the two-branch shape in §2.2 is not just stylistically consistent but the only way to keep that
index scan on the `produced_by`-bound call.

### 2.4 Full method body

```python
def create_document_with_auto_supersede(
    self, ws: str, *, document_id: str, title: str, text: str,
    text_normalized_hash: str, source_format: str, ingested_by: str,
    created_at: int, chunks: list[dict[str, Any]], match_id: str,
    produced_by: str | None = None,
) -> dict[str, Any]:
    """[... existing docstring, plus:]

    **`produced_by`** (optional, additive — `agent-knowledge-base-strategy.md`
    §4.1): when given, the ingestor-resolution clause matches ONLY
    `(a:Agent {agentId: $producedBy})` — `ingested_by`/`$ingestedBy` plays no
    role in this branch's query text or params at all, so a resolved
    `produced_by` always sources `sourceKind: 'agent'`, never `'document'`.
    When omitted (`None`, the default), the query text, params, and result are
    byte-identical to this method's pre-existing behavior. See §2.2/§2.3 of
    `docs/plans/agent-team-ingestion-graph.md` for why this is two literal
    query-text branches, not one shared text with a possibly-`NULL` param.
    """
    resolve = (
        _INGESTOR_RESOLVE_BY_PRODUCER if produced_by is not None
        else _INGESTOR_RESOLVE_BY_ACTOR
    )
    params: dict[str, Any] = {
        "documentId": document_id, "title": title, "text": text,
        "textNormalizedHash": text_normalized_hash,
        "sourceFormat": source_format,
        "createdAt": created_at, "chunks": chunks, "matchId": match_id,
    }
    if produced_by is not None:
        params["producedBy"] = produced_by
    else:
        params["ingestedBy"] = ingested_by

    res = self._graph(ws).query(
        resolve +
        "OPTIONAL MATCH (candidate:Document {"
        "  textNormalizedHash: $textNormalizedHash, currentVersion: true"
        "}) "
        "WITH ingestor, ok, ingestorIsUser, candidate "
        "ORDER BY candidate.createdAt ASC "
        "LIMIT 1 "
        "FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | "
        "  CREATE (d:Document {"
        "    documentId: $documentId, title: $title, text: $text, "
        "    sourceFormat: $sourceFormat, "
        "    sourceKind: CASE WHEN ingestorIsUser THEN 'document' ELSE 'agent' END, "
        "    status: 'processing', pendingJobs: 0, createdAt: $createdAt, "
        "    currentVersion: true, textNormalizedHash: $textNormalizedHash"
        "  }) "
        "  CREATE (d)-[:INGESTED_BY]->(ingestor) "
        "  FOREACH (ch IN $chunks | "
        "    CREATE (d)-[:HAS_CHUNK]->(:Chunk {"
        "      chunkId: ch.chunkId, text: ch.text, seq: ch.seq, documentId: $documentId, "
        "      documentCurrent: true"
        "    })"
        "  )"
        ") "
        "WITH ok, candidate "
        "OPTIONAL MATCH (d:Document {documentId: $documentId}) "
        "WITH ok, d, candidate, (ok AND candidate IS NOT NULL) AS doSupersede "
        "FOREACH (_ IN CASE WHEN doSupersede THEN [1] ELSE [] END | "
        "  SET candidate.currentVersion = false, candidate.supersededAt = $createdAt, "
        "      candidate.supersededBy = 'system' "
        ") "
        "WITH ok, d, candidate, doSupersede "
        "OPTIONAL MATCH (candidate)-[:HAS_CHUNK]->(c:Chunk) "
        "WITH ok, d, candidate, doSupersede, collect(c) AS candidateChunks "
        "FOREACH (ch IN CASE WHEN doSupersede THEN candidateChunks ELSE [] END | "
        "  SET ch.documentCurrent = false"
        ") "
        "WITH ok, d, candidate, doSupersede "
        "FOREACH (_ IN CASE WHEN doSupersede THEN [1] ELSE [] END | "
        "  CREATE (d)-[:SUPERSEDES {"
        "    matchId: $matchId, status: 'confirmed', confidence: 1.0, "
        "    technique: 'exact_normalized_text_hash', createdAt: $createdAt, "
        "    decidedAt: $createdAt, decidedBy: 'system', "
        "    resuggestCount: 0, lastResuggestedAt: null"
        "  }]->(candidate) "
        ") "
        "RETURN ok AS ingestorFound, d.documentId AS documentId, "
        "       doSupersede AS autoSuperseded, "
        "       CASE WHEN doSupersede THEN candidate.documentId ELSE null END "
        "         AS supersededDocumentId, "
        "       CASE WHEN doSupersede THEN $matchId ELSE null END AS matchId",
        params,
    )
    row = res.result_set[0]
    return {
        "documentId": row[1], "chunkCount": len(chunks),
        "autoSuperseded": bool(row[2]), "supersededDocumentId": row[3],
        "matchId": row[4], "ingestorFound": bool(row[0]),
    }
```

**Diff shape against today's method, precisely:** the `resolve`/`params` branching above, plus two
textual changes inside the previously-fixed query body — `WITH u, a, ingestor, ok, candidate` →
`WITH ingestor, ok, ingestorIsUser, candidate`, and `sourceKind: CASE WHEN u IS NOT NULL THEN
'document' ELSE 'agent' END` → `sourceKind: CASE WHEN ingestorIsUser THEN 'document' ELSE 'agent'
END`. Everything else in the query body — the candidate lookup, both `FOREACH` blocks, the
`SUPERSEDES` write, the `RETURN` — is unchanged text. The return-value construction in Python is
unchanged entirely.

**No repository-level status field is added.** `ingestorFound` keeps its existing single-boolean
meaning ("did some identity resolve") in both branches; the service layer (§3) decides which
exception to raise from the *branch it knows it took*, not from a new repository flag.

---

## 3. `Services.ingest_document` — exact change

```python
def ingest_document(
    self, ctx: CallContext, *, text: str, title: str | None = None,
    source_format: str = "text", source_label: str | None = None,
    produced_by: str | None = None,
) -> dict[str, Any]:
    """[... existing docstring, plus:]

    **`produced_by`** (optional — `agent-knowledge-base-strategy.md` §4.1):
    when given, attribution resolves ONLY against an existing `Agent` (never
    the `ctx.actor` `User`/`Agent` coalesce) and raises `AgentNotFoundError`
    — loudly, no silent fallback to `ctx.actor` — if no such `Agent` exists
    yet. Omitted, behavior is unchanged: `INGESTED_BY` resolves `ctx.actor`
    exactly as before.
    """
    if not text.strip():
        raise EmptyDocumentError("document text must not be empty or whitespace-only")
    if len(text) > MAX_DOCUMENT_CHARS:
        raise DocumentTooLargeError(...)
    chunk_texts = chunking.split_into_chunks(text)
    document_id = self._id()
    now = self._clock()
    chunks = [...]
    text_normalized_hash = update_detection.content_hash(
        update_detection.normalize_text(text)
    )
    result = self._repo.create_document_with_auto_supersede(
        ctx.ws, document_id=document_id, title=title or source_label or "",
        text=text, text_normalized_hash=text_normalized_hash,
        source_format=source_format, ingested_by=ctx.actor,
        created_at=now, chunks=chunks, match_id=self._id(),
        produced_by=produced_by,
    )
    if not result["ingestorFound"]:
        if produced_by is not None:
            raise AgentNotFoundError(produced_by)
        raise UnknownActorError(ctx.actor)
    return {
        "documentId": document_id, "chunkCount": len(chunks),
        "status": "processing",
        "autoSuperseded": result["autoSuperseded"],
        "supersededDocumentId": result["supersededDocumentId"],
    }
```

Only three lines change: the new parameter, `produced_by=produced_by` threaded into the repository
call, and the `if produced_by is not None: raise AgentNotFoundError(produced_by)` branch ahead of
the existing `raise UnknownActorError(ctx.actor)`. `ingested_by=ctx.actor` is still passed
unconditionally (harmless — the producer branch's query text never references `$ingestedBy`, and
§2.4's `params` construction doesn't even include the key in that branch).

**Why this guarantees "no silent fallback," structurally, not just by convention:** when
`produced_by` is given and unresolvable, the repository's producer branch never attempts an
`ingested_by`/`ctx.actor` lookup at all (§2.2) — there is no code path left that could silently
attribute to `ctx.actor` even by accident. The loud failure is a property of the query shape, not
only of the `if` above.

### 3.1 New exception — `AgentNotFoundError`

Add to `services.py`, next to `DocumentNotFoundError` (mirrors its shape/posture exactly, per the
coordinating brief):

```python
class AgentNotFoundError(ServiceError):
    """Raised when `ingest_document`/`ingest_documents`'s optional `produced_by`
    names no existing `Agent` (agent-knowledge-base-strategy plan §4.1, FR-8's
    per-producer attribution).

    Mirrors `DocumentNotFoundError`'s shape/posture: an explicit-id lookup that
    resolves nothing is a loud 404, never a silent no-op or a fallback to
    `ctx.actor` — `produced_by`'s whole point is differentiating the writing
    agent from the single, process-pinned configured actor `get_context()`
    always returns, so silently attributing to `ctx.actor` instead would
    defeat exactly what it exists to fix. `repository.
    create_document_with_auto_supersede`'s `produced_by` branch guards the
    entire `Document`/`Chunk` `CREATE` on the same `ok` flag
    `UnknownActorError`'s check already relies on, so no partial `Document`
    is ever written when this fires.
    """
```

**Wiring, `app.py`:** import `AgentNotFoundError` alongside the existing `DocumentNotFoundError`
import (line ~44), and add it to `_handle_service_error`'s `not_found` isinstance tuple
(line ~130) so it maps to 404 (matching `DocumentNotFoundError`'s own mapping) rather than the
generic `ServiceError` 400 default. No new `@app.exception_handler` needed — `AgentNotFoundError`
is a `ServiceError` subclass and the existing generic handler already dispatches on it; only the
`not_found` tuple membership needs the one addition.

---

## 4. `Services.ingest_documents` — per-item `produced_by` (resolves the plan's open question)

**Decision: yes, per-item, not a batch-level default.** Every other optional field
(`title`/`source_format`/`source_label`) is already per-item in this method's batch-item dict
shape — a batch-level `produced_by` covering every item uniformly would be an inconsistent shape
relative to that convention, and this feature has no known caller needing one constant producer
across a heterogeneous batch that a per-item field can't already express (a caller that *does* want
one producer for every item just repeats the same string in each item's dict — no expressiveness
lost). No new `Services.ingest_documents` parameter at all; the change is entirely inside the loop:

```python
for doc in documents:
    try:
        if (
            not isinstance(doc, dict)
            or not isinstance(doc.get("text"), str)
            or not isinstance(doc.get("produced_by"), (str, type(None)))
        ):
            receipt = {
                "status": "error",
                "error": "each batch item must be a dict with a string 'text' "
                         "key and an optional string 'produced_by' key",
                "errorType": "MalformedItemError",
            }
        else:
            receipt = self.ingest_document(
                ctx, text=doc["text"], title=doc.get("title"),
                source_format=doc.get("source_format", "text"),
                source_label=doc.get("source_label"),
                produced_by=doc.get("produced_by"),
            )
    except (ServiceError, KeyError, TypeError, AttributeError) as exc:
        receipt = {"status": "error", "error": str(exc), "errorType": type(exc).__name__}
    receipts.append(receipt)
```

**The added `isinstance(doc.get("produced_by"), (str, type(None)))` check is a deliberate,
slightly-stricter addition versus `title`/`source_format`/`source_label`'s existing untyped
`.get(...)` calls** — those rely purely on the wide `except` clause for defense in depth. This
feature's entire point is "fail loudly on misattribution, never silently" (§3.1); a wrong-typed
`produced_by` (e.g. a list) reaching the repository call risks surfacing as some driver-level
exception type outside the wide `except` tuple, aborting the whole batch rather than isolating one
item's receipt — a worse failure mode here specifically than it would be for a cosmetic field like
`source_label`. Cheap to add, so add it explicitly rather than leaning on the existing net.

**REST surface gets this for free.** `IngestDocumentsIn.documents` is already `list[IngestDocumentIn]`
(§5) — adding `producedBy` to `IngestDocumentIn` covers the batch schema automatically, no separate
`IngestDocumentsIn` change needed.

---

## 5. MCP and REST surface

### 5.1 MCP (`mcp.py`)

```python
@mcp.tool()
def ingest_document(
    text: str, title: str | None = None, source_format: str = "text",
    source_label: str | None = None, produced_by: str | None = None,
) -> dict[str, Any]:
    """[... existing docstring, plus a produced_by paragraph mirroring §3's.]"""
    ctx = _get_context()
    receipt = _svc().ingest_document(
        ctx, text=text, title=title, source_format=source_format,
        source_label=source_label, produced_by=produced_by,
    )
    # unchanged below this line
```

`ingest_documents(items: list[dict[str, Any]])` needs **no signature change** — each `items` dict
already flows straight into `Services.ingest_documents` (§4), so `produced_by` is simply a new
optional key an MCP caller may include per item. Only the docstring needs a line naming it.

### 5.2 REST (`schemas.py`, `api.py`)

`schemas.py` — add one field to `IngestDocumentIn` (covers both `/documents` and, via the existing
`list[IngestDocumentIn]` composition, `/documents/batch`):

```python
class IngestDocumentIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    title: str | None = Field(None, max_length=MAX_NAME_LEN)
    sourceFormat: str = Field("text", min_length=1, max_length=MAX_SOURCE_FORMAT_LEN)
    sourceLabel: str | None = Field(None, max_length=MAX_NAME_LEN)
    producedBy: str | None = Field(None, max_length=MAX_NAME_LEN)
```

(`MAX_NAME_LEN` — already defined earlier in the file and already the bound `title`/`sourceLabel`
use — reused rather than introducing a new constant; an `agentId` is a short slug, well inside it.
`MAX_KEY_LEN` is defined later in the file, after this class, so it isn't usable here without a
reorder this change doesn't need.)

`api.py` — thread `body.producedBy` through both routes:

```python
# POST /documents
receipt = services.ingest_document(
    ctx, text=body.text, title=body.title,
    source_format=body.sourceFormat, source_label=body.sourceLabel,
    produced_by=body.producedBy,
)

# POST /documents/batch
receipts = services.ingest_documents(
    ctx,
    documents=[
        {
            "text": item.text, "title": item.title,
            "source_format": item.sourceFormat,
            "source_label": item.sourceLabel,
            "produced_by": item.producedBy,
        }
        for item in body.documents
    ],
)
```

No other line in either route changes — the scheduling/`autoSuperseded` logic below is untouched.

---

## 6. Index implication — confirmed none, live-verified (not assumed)

**No new index or constraint.** `Agent.agentId` already carries a `RANGE` index and a uniqueness
constraint (`falkor-chat/AGENTS.md`'s "every entity node has a stable `{label}Id` property, a range
index, and a uniqueness constraint" convention) and is already the exact index every other
agent-identity lookup in this codebase hits (`ensure_agent`, message-author resolution). §2.3's live
`EXPLAIN` confirms the new `produced_by` branch's `OPTIONAL MATCH (pa:Agent {agentId:
$producedBy})` reuses that same index, `Node By Index Scan`, when a real value is bound — the one
condition §2.2's two-branch design exists to guarantee. `ws:agent-team`'s own bootstrap
(`agent-knowledge-base-strategy.md` §4.2, Track 1 Stage 2) creates this index the same way every
other workspace's does — nothing new to add to `bootstrap_schema.sh` for this change specifically.

---

## 7. Required test coverage

§7 of the coordinating plan names three floor cases; the ones below add the malformed-item and
sourceKind cases the floor didn't spell out, and point at the exact existing tests each new test
sits beside.

**`tests/test_repository.py`** (live, against `ws:test`) — extend the `_auto_supersede` helper
(`tests/test_repository.py:1472`) to accept an optional `produced_by` kwarg, forwarded through:

- *Existing Agent succeeds, `INGESTED_BY` resolves to it, not `ctx.actor`* — register an Agent
  (`repo.ensure_agent(...)`) **and** a different, also-valid `User` (`repo.ensure_user(...)`); call
  with `ingested_by=<user>` **and** `produced_by=<agent>` simultaneously; assert
  `repo.get_document(...)["ingestedByKind"] == "Agent"` and `ingestedById == <agent>` — proving
  resolution took the producer branch, not the actor branch, even though the actor would have
  resolved fine on its own. Also assert `sourceKind == "agent"`.
- *Missing Agent raises loudly, no partial `Document`* — mirror
  `test_create_document_with_auto_supersede_unknown_actor_nothing_written`
  (`tests/test_repository.py:1505`) exactly, but with `produced_by="ghost-agent"` (no such `Agent`)
  and a *valid* `ingested_by`: assert `result["ingestorFound"] is False` and
  `repo.get_document(ws, document_id=…) is None`. This is the repository-level half of the
  `AgentNotFoundError` guarantee — the loud raise itself is a `Services` behavior (below).
- *Omitted `produced_by` is byte-for-byte unchanged* — every existing test in this file's
  `create_document_with_auto_supersede` block (lines ~1487–1735) must pass **unmodified**. Do not
  edit any of them; their continuing to pass with zero changes is the regression proof, not a
  separate new test.

**`tests/test_services.py`** — extend `FakeRepo.create_document_with_auto_supersede`
(`tests/test_services.py:257`) to accept `produced_by=None`; when given, resolve against
`self.agents` **only** (ignore `ingested_by`/`self.members` entirely in that branch, mirroring
§2.2's real-repository behavior), returning `ingestorFound=False` when absent — same shape the real
method returns.

- *Success* — sibling of `test_ingest_document_known_agent_actor_source_kind_agent`
  (`tests/test_services.py:997`): `svc.ingest_document(ctx, text="hello", produced_by="bot1")` where
  `ctx.actor` is a **different**, unresolvable id — succeeds, proving `produced_by` alone drove
  resolution.
- *Missing Agent raises `AgentNotFoundError`, not `UnknownActorError`* — sibling of
  `test_ingest_document_unknown_actor_raises_instead_of_silent_write`
  (`tests/test_services.py:987`): `svc.ingest_document(ctx, text="hello", produced_by="ghost")` with
  a **valid** `ctx.actor` — must raise `AgentNotFoundError`, and `repo.documents == {}` (no partial
  write). The valid-`ctx.actor` setup is the load-bearing part of this test: it is what proves there
  is no silent fallback, not merely that *some* error is raised.
- *Omitted `produced_by` regression* — `test_ingest_document_unknown_actor_raises_instead_of_silent_write`
  and `test_ingest_document_known_agent_actor_source_kind_agent` must both pass unmodified (same
  proof pattern as the repository layer).
- *Batch, per-item independence* — sibling of `test_ingest_documents_isolates_an_unknown_actor_failure`
  (`tests/test_services.py:1140`): a two-item batch, one item carrying a resolvable `produced_by`
  and one carrying none (falls back to `ctx.actor`), assert each receipt reflects its own resolution
  independently.
- *Malformed per-item `produced_by`* — sibling of
  `test_ingest_documents_isolates_a_non_string_text_item` (`tests/test_services.py:1098`): an item
  with `"produced_by": 123` (non-string, non-`None`) isolates to a `MalformedItemError` receipt; the
  batch's other items still process.

**`tests/test_api.py`** — `POST /documents` with `producedBy` in the body → 201, then
`GET /documents/{id}` confirms `ingestedByKind: "Agent"`/`ingestedById`; an unresolvable
`producedBy` → **404** with `{"error": "AgentNotFoundError", ...}` (proves the `app.py`
`not_found`-tuple wiring, §3.1); omitted `producedBy` → existing tests at
`tests/test_api.py:269`/`:274` region unmodified. Mirror for `POST /documents/batch`'s per-item
`producedBy`.

**`tests/test_mcp.py`** — same three cases via the MCP tool call, sibling to whatever existing
`ingest_document`/`ingest_documents` MCP tests already do around `tests/test_mcp.py:358`/`:363`.

**Not this unit's test responsibility** (named so nobody re-derives it): the coordinating plan's
§7 "attribution correctness, end-to-end" spot check (`list_documents`'s `ingestedByKind`/
`ingestedById` actually varying across real agents in `ws:agent-team`) needs Track 1 Stages 2–4
(workspace bootstrap, dedicated process, rollout) to exist first — out of scope here, flagged for
whoever picks up Stage 4.

---

## 8. Explicitly out of scope (named, not silently dropped)

- **`delete_document`'s `deletedBy` still stamps from `ctx.actor` only.** The coordinating brief
  names this gap explicitly as *not* this unit's job (§8 of the parent plan, `analyst`'s minor
  finding 2) — every `delete_document` call the parent plan names is `cobb`'s own, so it doesn't
  undermine FR-8's producer-attribution goal. A same-shaped `produced_by`-style follow-up for
  `delete_document` is a small, separate unit if it's ever needed — not designed here.
- **`ws:agent-team` bootstrap and the `Agent`-roster seed script** (parent plan §4.2, Track 1
  Stage 2) — a separate, already-named `graph-dba`/`devops` unit. This note's `produced_by`
  resolution has zero forward dependency on it (the Cypher shape is workspace-agnostic — it works
  identically against `ws:test`, `ws:demo`, or `ws:agent-team` once that graph exists and carries
  the expected `Agent` nodes).
- **The dedicated `ws:agent-team` server process** (parent plan §4.3, Track 1 Stage 3) — `devops`
  scope, unaffected by this note.
