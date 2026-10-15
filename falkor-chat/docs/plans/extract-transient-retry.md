# Transient provider-error retry for document-ingestion background jobs

> **Status:** active · **Owner:** `architect` · **Tracks:** K-067 · **Version:** 3 · **Reviews:** `falkor-chat/docs/reviews/extract-transient-retry.md`

*Revision 2026-10-15 (v2): folds in Pass 1 plan review findings M-1…M-4 and m-1…m-5 (dispositions in §8.1) and the stakeholder's confirmation that embed is in scope.*

*Revision 2026-10-15 (v3): targeted amendment for Pass 2 findings N-1…N-5, N-7 and the two nits (dispositions in §8.2); scope unchanged.*

## 1. Goal & scope

**Goal.** A transient provider failure during a document chunk's background model call no longer
marks the whole `Document` as `failed` forever. Such failures include LM Studio's HTTP 400
`{"error":"terminated"}`, an HTTP 5xx, a failed JIT model load, and a dropped connection. The
call is retried in place with bounded, jittered backoff. A permanent error still fails on its
first attempt: a context-size overflow, a malformed request, a bad response shape, or any graph
error. An exhausted budget still ends in `failed`. The K-051 contract is unchanged: each job
reports exactly once, the first terminal write wins, and one failed job fails the document.

**In scope.**
- Extract job (`background._safe_extract` → `IngestionPipeline.extract_chunk`): this is K-067
  itself.
- Chunk-embed job (`background._safe_embed_chunk` → `EmbeddingWorker.embed_chunk`). The
  stakeholder confirmed this on 2026-10-15; the rationale is in §3.4.
- A structured classification seam on `transport.ProviderCallError`. Today the HTTP status lives
  only inside the message string.

**Out of scope.**
- Retrying the graph-write phase of `extract_chunk`. It is not idempotent, and §3.3 explains why
  this design never re-runs it.
- `EmbeddingWorker.embed_message` and every request-path or agent-path model call
  (`AgentResponder`, `Services.search_documents`, `GraphragRetrieveTool`, the executor). Each has
  its own latency contract and failure surface.
- An automatic re-drive of documents that are already `failed`. The 12 currently `failed`
  documents in `ws:agent-team` (§2.6) still need the manual delete-then-re-ingest.
- Reordering `_schedule_chunk_processing` so that it schedules all embeds before all extracts.
  This is a plausible mitigation for the model-eviction thrash, filed as a follow-up candidate in
  §6.
- A bounded MCP thread pool. That is the existing deferred item in `mcp._default_schedule`'s
  docstring.

## 2. Context & findings

CPG: used cpg_falkorchat — caller sweep of `extract_chunk`/`embed_chunk`/`embed_message`/`_resolve_and_embed`/`embed` and every `ProviderCallError(...)` construction site, to fix the retry seam and prove the new exception fields are additive.

### 2.1 Where the failure happens: call chain and threads

```
api.py ingest_document(s) ─┐                       (REST: starlette BackgroundTasks.add_task)
mcp.py ingest_document(s) ─┴─ background._schedule_chunk_processing   (MCP: mcp._schedule → daemon Thread)
      ├─ repo.start_document_progress(total_jobs = chunks × wired workers)
      ├─ _safe_embed_chunk(worker, ws, document_id, chunk_id, text)
      │     └─ EmbeddingWorker.embed_chunk → _resolve_and_embed → embedder.embed(text) → transport
      │        then repo.set_chunk_embedding           (MATCH … SET, idempotent)
      └─ _safe_extract(pipeline, ws, chunk_id, document_id, text)
            └─ IngestionPipeline.extract_chunk
                 llm = self._models.llm("extraction", ws=ws)          ← read-only (workspace-override ro_query), no write
                 result = extraction.extract(text, llm)              ← the ONLY provider call
                 for entity …: create_entity_with_auto_match / link_chunk_about_entity / _safe_fuse
                 for rel …:    create_entity_relationship
      each _safe_* → _report_document_job → repo.report_document_job_done(success=…)   (once)
```

- **Both job functions are synchronous.**
  - REST: Starlette 1.3.1 `BackgroundTask.__call__` sends a sync callable through
    `run_in_threadpool`. That means an anyio worker thread under the default limiter, which
    `app.py`'s lifespan raises to `config.THREAD_LIMIT` (100). **`BackgroundTasks.__call__` awaits
    its tasks one after another** (`for task in self.tasks: await task()`; verified in
    `server/.venv/.../starlette/background.py`). So for one REST ingest request, every chunk's
    embed and extract job runs **sequentially**. For `ingest_documents` that means one chain
    across **every document in the batch** (`api.py` `ingest_documents`; up to
    `schemas.MAX_BATCH_SIZE` = 20). A sleeping retry delays that request's later
    jobs. It does not block the event loop or other requests.
  - MCP: `mcp._default_schedule` starts one unbounded daemon `threading.Thread` per job. All of a
    document's jobs run **concurrently**, so a provider hiccup hits every in-flight job at once
    (a thundering herd on recovery).
- **`extract_chunk` makes exactly one provider call, and it comes before any graph write.**
  `extraction.extract` calls `llm.complete(...)` once and returns `ExtractionResult`. An
  unparseable or invalid reply is **not** an exception: it returns `_EMPTY`. `_safe_fuse` →
  `fuse_entity` → `fusion.find_fuzzy_candidates` is a RediSearch read with no provider call, and
  it is already wrapped per entity.
- **`embed_chunk`'s pre-write region** is `_resolve_and_embed`: gateway resolution, the
  `read_index_dimension` read, `embedder.embed`, and a length check. It writes nothing. The one
  write, `set_chunk_embedding`, comes after it.
- `embed_message` is also called **synchronously from `AgentResponder.maybe_respond`**
  (CPG-verified), so `_resolve_and_embed` is shared with a latency-sensitive path. That is why
  the embed retry wraps the call inside `embed_chunk`, not `_resolve_and_embed` itself (§3.4).

### 2.2 What a provider failure actually surfaces as at the call site

`extraction` resolves through `ModelGateway.llm("extraction")`. The shipped `config/models.json`
default is the direct ref `lmstudio/qwen/qwen3-4b-2507`, a length-1 chain, so
`ModelGateway._client_for_chain` returns a bare `OpenAICompatibleLLM`, not a `FallbackClient`.
The embedder is always a bare `OpenAICompatibleEmbedder` (`embedder()` never wraps). Both POST
through `transport.make_http_transport._transport`. Its ladder converts failures as follows; the
"verified by" column says how I know.

| Provider event | Python path | Surfaces as | Verified by |
|---|---|---|---|
| HTTP 400 `{"error":"terminated"}` | `urlopen` raises `HTTPError(400)` → rung 1 | `ProviderCallError("<p/m> @ <url>: HTTP 400 Bad Request: {\"error\":\"terminated\"}")`, `__cause__` = `HTTPError` | field log, `falkor-chat/docs/test-reports/salesperson-language-salience-report.md` "Engine-stability observation" (exact rung-1 message format) |
| HTTP 500 (HTML body) | rung 1 | `ProviderCallError("…: HTTP 500 Internal Server Error: <!DOCTYPE html>…")` | same field log |
| HTTP 400 `Failed to load model … Engine protocol startup was aborted` / `… Error loading model` (JIT cold load failed, or eviction) | rung 1 | `ProviderCallError("…: HTTP 400 …: …Failed to load model…")` | `docs/reviews/document-ingestion-ml.md` (HTTP 400 during concurrent extract+embed); `claude/data-scientist/lm-studio-model-notes.md` (400 on the embedding model, an immediate retry succeeded) |
| HTTP 400 `exceed_context_size_error` (permanent: the input is too large) | rung 1 | `ProviderCallError("…: HTTP 400 …: {\"error\":{…\"type\":\"exceed_context_size_error\"…}}")` | `opencode/docs/test-reports/devops-opencode-headless-report.md` (body shape) |
| HTTP 400 role-alternation or Jinja template error (permanent) | rung 1 | same, different body | ws:agent-team KB entries (LM Studio 400 on Mistral templates) |
| Connection refused (LM Studio restarting or down) | `h.request` raises `OSError` → urllib wraps it as `URLError` → rung 2 | `ProviderCallError("…: connection failed: [Errno 111] …")` | read `urllib.request.AbstractHTTPHandler.do_open` (py 3.12.3) |
| Server drops the socket before the status line | `getresponse` raises `RemoteDisconnected` (an `OSError` subclass, **not** wrapped by urllib) → rung 3 | `ProviderCallError("…: RemoteDisconnected: …")` | read `do_open` + MRO check |
| Read timeout (180 s for `extraction`, 30 s for `embedding`) | `TimeoutError` → rung 3 (or rung 2 if it fires during send) | `ProviderCallError("…: TimeoutError: …")` | `transport.py` ladder |
| Connection reset or truncated body **while reading the body** | `resp.read()` sits **outside** the ladder's `try` | **raw** `ConnectionResetError` / `http.client.IncompleteRead` / `TimeoutError`, not a `ProviderCallError` | read `make_http_transport._transport`: `resp.read()` is under `try/finally` with no `except` |
| 200 with `{"error": …}` | rung 5 | `ProviderCallError("…: provider error: …")` | `transport.py` |
| 200, `choices`/`data` missing | client layer | `ProviderCallError("… response missing choices/data …")` | `llm.py` / `embedding.py` |

**Consequence for the classifier.** Every provider failure that matters is a single exception
class, `ProviderCallError`. Its only structured trace of the HTTP status is `__cause__.code`, and
only on rung 1: the body has already been consumed by `_read_error_body` and survives only inside
the message string. So **400 "terminated" and 400 "context overflow" are distinguishable today
only by parsing the message.** That is the fragility §3.1 removes. A classifier keyed on
`urllib.error.HTTPError`, `requests` exceptions, or `openai.APIError` would match nothing,
because none of those ever reaches `_safe_extract`.

### 2.3 The K-051 status contract (unchanged by this design)

- `Repository.start_document_progress` sets `pendingJobs = total_jobs` once, synchronously,
  before scheduling.
- `Repository.report_document_job_done` is one atomic query:
  - it decrements `pendingJobs`;
  - on `success=False` with `status='processing'`, it sets `'failed'`;
  - on success with `pendingJobs <= 0` and `status='processing'`, it sets `'ready'`.
- The first terminal write wins.
- The Cypher is documented in `falkor-chat/docs/QUERIES.md` §14.1a.

### 2.4 What `extract_chunk` writes, and whether that is repeatable

All of these are plain `CREATE`s keyed on freshly minted `uuid4` ids. None is a `MERGE`:

- `create_entity_with_auto_match`:
  - `CREATE (e:Entity {entityId: <new uuid>…})`;
  - plus `CREATE (e)-[:SAME_AS {status:'confirmed'…}]->(candidate)` when an older `Entity` shares
    `(nameNormalized, type)`.
- `link_chunk_about_entity`: `CREATE (c)-[:ABOUT]->(e)`. Its docstring says "a chunk is
  extracted exactly once per ingestion".
- `create_entity_relationship`: `CREATE (subj)-[:RELATES_TO {…}]->(obj)`, "never deduplicated".
- `_safe_fuse` → `create_or_reopen_match`: find-or-create by entity pair. Since the pair includes
  a fresh entity, it is effectively always a create.

**Re-running the write phase is therefore not idempotent.** A second pass over the same
`ExtractionResult` would leave:
- a second `Entity` per mention;
- a confirmed `SAME_AS` from each new duplicate to the first pass's twin, because the exact tier
  finds it;
- a doubled set of `ABOUT` and `RELATES_TO` edges.

A partial first pass, for example a FalkorDB error after entity 3 of 7, followed by a full re-run
leaves duplicates of entities 1–3. Making these writes idempotent would need deterministic ids
plus `MERGE`, and each `MERGE` would need a uniqueness constraint. That is a schema and RAM
change. §3.3 shows it is unnecessary, because this design never re-runs this phase.

`set_chunk_embedding` is `MATCH (c:Chunk {chunkId}) SET c.embedding = vecf32(...)`, which is
idempotent by construction.

### 2.5 Concurrency fact that shapes the backoff budget

On this lab's LM Studio, **exactly one model is resident at a time**: loading a second one evicts
the first (`claude/devops/kaizen/history.md`, citing data-scientist's live notes). The extract
job (chat model) and the embed job (embedding model) for the same document alternate, so they
evict each other. A request that lands during a JIT reload can fail with HTTP 400 "Failed to load
model", and three measured cold loads span **~3.5 s to ~21 s**
(`claude/data-scientist/lm-studio-model-notes.md`). The K-067 crash pattern is 400 "terminated",
then 500, then recovery "within seconds". One earlier session saw "terminated" errors persist
for **several minutes** under concurrent extract+embed load (`docs/reviews/document-ingestion-ml.md`).

### 2.6 Field evidence: which job actually failed

A read-only query against `ws:agent-team` on 2026-10-15 found **12 `failed` documents** with
`pendingJobs = 0` (every job reported) and **27 chunks** across them:

| chunk state | chunks |
|---|---|
| no embedding, no `ABOUT` edge | 12 |
| no embedding, entities extracted | 9 |
| embedded, no entities | 1 |
| embedded and extracted | 5 |

21 of the 27 chunks lack an embedding. **Every one of the 12 failed documents has at least one
un-embedded chunk.**

- **What a missing `ABOUT` edge means.** It is ambiguous: extract may have failed, or it may have
  returned `_EMPTY`. The review measured 116 of 563 chunks (20.6 %) in `ready` documents with no
  `ABOUT` edge, so `_EMPTY` is common.
- **What a missing embedding means.** An un-embedded chunk under `pendingJobs = 0` is a reported
  embed failure. The review confirmed this independently on four counts:
  - `_safe_*` has no status check, so every job runs;
  - `set_chunk_embedding` always writes when the chunk exists;
  - nothing clears `Chunk.embedding`;
  - `EmbeddingDimensionError` is ruled out by the 337 `ready` documents in the same workspace.
- **What this shows.** Embed jobs demonstrably failed in the field.
- **What it does not show.** *Which exception* failed them. No server log from 2026-09-19/20
  survives. A 30 s `embedding` read timeout under contention is plausible, and v1 never retries
  timeouts. This evidence therefore cannot say whether the change would have fixed these
  incidents. Live leg (g) (§5) exists to capture the real shape, and §3.2.1 pre-states the
  response if it turns out to be a timeout.

## 3. Design & rationale

**Shape in one paragraph.**
1. `ProviderCallError` gains structured, keyword-only fields that its raisers fill in.
2. A new module, `falkorchat/retry.py`, owns:
   - the classifier, `is_transient(exc) -> bool`;
   - the policy, `RetryPolicy` / `DEFAULT_PROVIDER_RETRY`;
   - one helper, `call_with_retry(...)`.
3. `IngestionPipeline.extract_chunk` wraps **only** `extraction.extract(text, llm)` in
   `call_with_retry`.
4. `EmbeddingWorker.embed_chunk` wraps **only** its `_resolve_and_embed(...)` call.
5. Before each retry sleep, a document-liveness gate asks whether the chunk's owning `Document`
   is still `processing`. If it is not, the job stops retrying.
6. `background.py`'s `_safe_*` wrappers and `report_document_job_done` are **unchanged in
   code**. Their docstrings gain one sentence each.

### 3.1 Decision 1: transient vs permanent, and how the code tells them apart

**Mechanism: structured fields on the exception, with classification in exactly one place.**

`transport.ProviderCallError` gains keyword-only constructor arguments, each defaulting to
`None` or empty:

| field | meaning | set by |
|---|---|---|
| `kind: str \| None` | one of `"http"`, `"connection"`, `"network"`, `"timeout"`, `"read"`, `"malformed_request"`, `"invalid_json"`, `"body_error"`, `"body_shape"`, `"response_shape"`, `"fallback_exhausted"` | each raise site (table below) |
| `status: int \| None` | HTTP status code (rung 1 only) | `make_http_transport._transport` |
| `detail: str \| None` | the provider's error text: the truncated HTTP body for rung 1, the rendered `error` value for rung 5 | `make_http_transport._transport` |
| `causes: tuple[ProviderCallError, ...]` | per-element errors of an exhausted fallback chain | `FallbackClient.chat` / `.complete` |

`str(exc)` (the message) is **byte-for-byte unchanged** at every site, so every existing log
line, test assertion, and `WorkflowRun.ctx` error string stays identical. The fields are
additive. Every construction site keeps working without edits: the CPG finds them in 6
production methods and 5 test fakes. A
bare `ProviderCallError(msg)` gets `kind=None`, which classifies as permanent.

**Raise-site mapping** (in `transport.make_http_transport._transport` unless noted):

| rung | `kind` | extra fields |
|---|---|---|
| 1 `HTTPError` | `"http"` | `status=exc.code`, `detail=body` |
| 2 `URLError` | `"timeout"` if `isinstance(exc.reason, TimeoutError)`, else `"connection"` | — |
| **new rung 3a** `http.client.InvalidURL` (v3, review N-1; must precede rung 3) | `"malformed_request"` | — |
| 3 `(TimeoutError, OSError, http.client.HTTPException)` (v2: `HTTPException` added, review m-3) | `"timeout"` if `isinstance(exc, TimeoutError)`, else `"network"` | — |
| 4 `ValueError` | `"malformed_request"` | — |
| **new rung 3b: `resp.read()` failure** | `"timeout"` if `TimeoutError`, else `"read"` | — |
| JSON decode failure | `"invalid_json"` | — |
| 5 body `error` key | `"body_error"` | `detail=_render_body_error(...)` |
| 6 non-object body | `"body_shape"` | — |
| `OpenAICompatibleLLM.complete`/`.chat`, `OpenAICompatibleEmbedder.embed`: missing `choices`/`data` | `"response_shape"` | — |
| `FallbackClient.chat`/`.complete`: all elements failed | `"fallback_exhausted"` | `causes=tuple(the caught errors)` |

**New rung 3a (v3, review N-1).** `http.client.InvalidURL` is an `HTTPException`, **not** a
`ValueError` (MRO `InvalidURL → HTTPException → Exception`, re-verified in the server venv,
CPython 3.12.3). `urlopen` raises it at call time for a URL `modelconfig._validate_url` lets
through, for example `http://localhost:abc/…` ("nonnumeric port") or a space in the host ("can't
contain control characters"). Rung 4, the existing "malformed-URL shapes" rung, never sees it;
without rung 3a, v2's widened rung 3 would classify it `"network"` and retry a permanent
misconfiguration for a full budget. Rung 3a is a dedicated `except http.client.InvalidURL` clause
placed **immediately before rung 3**, raising `ProviderCallError(f"{label}: malformed request:
{exc}", kind="malformed_request")`, the same message format as rung 4. Today the exception
escapes raw; this is the same class of contract change as rung 3b (R-6). Rung 4 stays where it is.

**New rung 3b** wraps `raw = resp.read()` in `except (TimeoutError, OSError,
http.client.HTTPException)`, keeping the existing `finally: close()`, and raises
`ProviderCallError(f"{label}: reading response failed: {type(exc).__name__}: {exc}")`. This
closes the gap found in §2.2, where a reset mid-body escapes as a raw exception. It changes
behaviour for every caller: they now see a `ProviderCallError` where they used to see a raw
`OSError`/`IncompleteRead`, and `FallbackClient` now falls through on it. Every caller already
treats `ProviderCallError` as "the provider failed", so this is strictly a fix. It is called out
here because it is a contract change on a shared seam.

**The classifier, `retry.is_transient(exc: BaseException, *, retry_timeouts: bool = False) -> bool`:**

```python
TRANSIENT_HTTP_STATUSES = frozenset({408, 429})        # plus every 5xx
TRANSIENT_DETAIL_MARKERS = (                           # case-insensitive substring match on .detail
    "terminated",                    # LM Studio engine crash mid-request (K-067)
    "failed to load model",          # JIT (re)load failed — eviction / cold load
    "error loading model",
    "engine protocol startup was aborted",
    "channel error",                 # LM Studio serving-layer error seen alongside "terminated"
)

def is_transient(exc, *, retry_timeouts=False):
    if not isinstance(exc, ProviderCallError):
        return False                               # graph errors, EmbeddingDimensionError,
                                                   # ModelResolutionError, bugs: never retried
    if exc.kind == "fallback_exhausted":
        return bool(exc.causes) and all(
            is_transient(c, retry_timeouts=retry_timeouts) for c in exc.causes)
    if exc.kind in {"connection", "network", "read"}:
        return True
    if exc.kind == "timeout":
        return retry_timeouts                      # off in every v2 default policy (§3.2.1)
    if exc.kind == "http":
        return (exc.status in TRANSIENT_HTTP_STATUSES
                or (exc.status is not None and 500 <= exc.status <= 599)
                or _has_marker(exc.detail))
    if exc.kind == "body_error":
        return _has_marker(exc.detail)
    return False   # "malformed_request", "invalid_json", "body_shape",
                   # "response_shape", None
```

**Rationale, call by call.**
- **Status alone cannot decide a 400.** LM Studio uses 400 both for the transient "terminated"
  and model-load failures and for the permanent context-overflow and template errors (§2.2).
  So a 4xx is permanent **unless** its body carries a known transient marker. This is an
  allow-list: an unknown 400 fails fast. Failing fast is the safe default, because the result is
  today's behaviour, never a silent loop.
- **Every 5xx is transient**, along with 408 and 429. These are standard. LM Studio never sends
  429, but a cloud provider configured through the same seam would.
- **Connection-level failures are transient.** Examples: refused, reset, `RemoteDisconnected`,
  mid-body read failure. They are what an LM Studio restart or crash looks like from the client.
  A wrong host or port, a permanent misconfiguration, also lands here and burns one budget
  before failing (§3.2 gives the figures). That cost is bounded, and the gate in §3.5 limits it to
  the first job per document on the sequential REST path.
- **Timeouts are not retried by default.**
  - For `extraction`, the read timeout is 180 s. Retrying it three times would hold a worker for
    up to ~13 min per job (§3.2) against a server that is either hung or genuinely too slow for
    this chunk.
  - The review is right that this argument does not carry over to `embedding`: there the timeout
    is 30 s, and 4×30 + 60 = 180 s.
  - So timeout retry is a **per-policy switch** (`RetryPolicy.retry_timeouts`, passed through to
    `is_transient`), built and unit-tested now but **off in both shipped policies**.
  - Whether to turn it on for embed is decided by live leg (g). The rule for that is pre-stated in
    §3.2.1.
- **`"channel error"` is inferred, not verified.** The salience report lists it among LM Studio's
  hard 400/500 errors under contention, but its exact status and body were not captured. If it
  arrives as a 5xx, the status rule already covers it. Keeping it as a marker costs nothing,
  because no permanent error says "channel error".
- **Model-output problems are not exceptions at all.** `extraction.extract` returns `_EMPTY` for
  an unparseable reply, so a "bad chunk" in the model-output sense already never fails the job.
  The genuinely bad chunk that must fail fast is the provider-rejected one, a 4xx with no marker
  such as context overflow. The allow-list makes it fail on attempt 1.

**Rejected alternative: regex over `str(exc)`** (`r"HTTP (\d{3})"` plus a marker search). It
needs zero changes to `transport.py`, but it couples the retry policy to a message format that is
documented as human-facing (FR-10), and an innocent rewording would silently turn every retry
off. The structured fields cost one keyword argument per raise site.

**Rejected alternative: a `transient: bool` computed inside `transport.py`.** It would spread the
policy across `transport.py` (one decision per rung), `llm.py` (fallback aggregation), and
whoever else raises. With the fields above, `transport.py` reports facts and `retry.py` alone
decides, so it is the only file a future policy change touches.

### 3.2 Decision 2: backoff, attempts, and budget

**Policy.** At most **4 attempts**: the initial call plus 3 retries. The nominal delays before
retries 1, 2, and 3 are **2 s, 8 s, 30 s**. Each delay is multiplied by `1 + 0.5·U[0,1)`, upward
"positive jitter".

| before attempt | nominal cumulative wait | jittered cumulative wait |
|---|---|---|
| 2 | 2 s | 2–3 s |
| 3 | 10 s | 10–15 s |
| 4 | 40 s | 40–60 s |

**Worst-case time per job (v2, recomputed; review m-1).** Sleep is < 60 s. But each attempt is
bounded only by its read timeout, not by "fails fast". A 400 "terminated" or a 500 can arrive
after almost a full timeout's worth of generation, and a timeout is not retried, so an attempt
that hits it ends the job. The bounds below use "<" because U < 1 and a retried attempt must end
before the timeout fires.

| job | attempts × per-attempt bound | + sleep | worst case | typical (fast-failing 400/500/refused, < 1 s each) |
|---|---|---|---|---|
| extract (`timeouts.extraction` = 180 s) | 4 × < 180 s | < 60 s | **< 780 s (13 min)** | 40–60 s |
| embed (`timeouts.embedding` = 30 s) | 4 × < 30 s | < 60 s | **< 180 s** | 40–60 s |
| embed with `retry_timeouts` on (§3.2.1) | 4 × ≤ 30 s | < 60 s | **≤ 180 s** (unchanged) | — |

The cumulative-wait table above is unaffected. Every other figure in this plan that says
"per job" cites this table rather than restating it.

**Derivation from the evidence (§2.5), not from the backlog's prior:**
- **Attempt 2 at ~2–3 s** catches the "within seconds" crash recovery from K-067, and the cold-load
  flake that "re-running the exact same call immediately after passed" (data-scientist notes).
- **Attempt 3 at ~10–15 s** lands after a typical cold reload, at the ~3.5 s low end and partway
  up the range.
- **Attempt 4 at ~40–60 s** clears the **slowest measured cold load (~21 s) by about 2×**. It also
  absorbs a "terminated → 500 → recovery" sequence that takes a few cycles. An earlier final
  attempt, for example at 10 s, would sit inside the reload window. That is exactly the failure
  K-067 warns about: retrying against a still-crashed instance.
- **Nothing longer.** The "several minutes" sustained-contention episode
  (`document-ingestion-ml.md`) is not something an inline per-job retry should absorb. Holding
  anyio worker threads (REST) or OS threads (MCP) for minutes per job, multiplied across chunks,
  costs more than re-ingesting. Such a document still ends in `failed`, which is the correct,
  visible outcome, and that case belongs to the follow-ups in §6.
- **Upward jitter** gives a ≥ 2 s / ≥ 10 s / ≥ 40 s floor, so no attempt fires earlier than its
  derived point, while spreading the MCP path's concurrent jobs (§2.1). The MCP path is where all
  of a document's jobs fail together and would otherwise retry in lockstep against a model that is
  still reloading.

**Configurability: module constants plus constructor injection. No new runtime knob.**
- `retry.DEFAULT_PROVIDER_RETRY = RetryPolicy(max_attempts=4, delays_s=(2.0, 8.0, 30.0),
  jitter=0.5)`.
- `IngestionPipeline` and `EmbeddingWorker` each accept `retry_policy: RetryPolicy =
  DEFAULT_PROVIDER_RETRY` and `sleep: Callable[[float], None] = time.sleep`. These are
  keyword-only and serve as the test seam, the same pattern as their existing
  `id_gen=`/`clock=` injection.
- `app.py` passes nothing, so production uses the defaults.
- **Rejected: an overlay key in `config/models.json`.** `modelconfig.Overlay` knows six
  top-level keys and logs and ignores any others. Its subject is models and providers, and a
  retry policy belongs to the background job, not to a model. Adding a key would mean new parsing,
  validation, and per-kind resolution in `ModelGateway`, all for a value no one has needed to tune
  yet. If tuning is needed, a `retries.<kind>` overlay key is the natural home, because it would
  follow `timeouts.<kind>` exactly. That is OQ-2 in §6.
- **Rejected: an env var.** The repo has explicitly moved model-related knobs out of env vars
  (`config.assert_no_legacy_model_env`).
- Consequence: no `falkor-chat/AGENTS.md` change (§4, step 3).

**`RetryPolicy` fields (v2):**
- `max_attempts`, `delays_s`, `jitter = 0.5`, and **`retry_timeouts: bool = False`**.
- Two shipped constants:
  - `DEFAULT_PROVIDER_RETRY`, used by `IngestionPipeline`;
  - `EMBED_CHUNK_RETRY`, used by `EmbeddingWorker.embed_chunk`.
- The two are equal in v2, both with `retry_timeouts=False`. They are separate constants so that
  §3.2.1 can flip one of them without touching the other.

#### 3.2.1 Pre-stated outcome if live leg (g) shows embed read timeouts (OQ-B)

**Decision: a bounded, embed-only timeout retry, pre-authorized.** It is not a return for
redesign.

- **Trigger.** Leg (g) (§5) records at least one embed job whose first-attempt failure is
  `kind="timeout"`.
- **Change.** In `retry.py`, set `EMBED_CHUNK_RETRY = RetryPolicy(max_attempts=4, delays_s=(2.0,
  8.0, 30.0), jitter=0.5, retry_timeouts=True)`.
  - That is one constant. No other production code changes: the switch, its plumbing, and its
    tests (U-5b, U-17) land in steps 1 and 3 regardless. The flip turns U-5b's shipped-policy row
    red **by design**, so the same edit updates that row (done-condition 2 below).
  - The extract policy stays `retry_timeouts=False`.
- **Numbers.** Worst case per embed job ≤ 180 s (§3.2 table). That equals one extract attempt
  bound, so it does not change any per-request worst case already stated for extract.
- **Done when** all four hold (v3, review N-5):
  1. U-17 is parametrized over the flipped constant and passes.
  2. U-5b's shipped-policy row is updated to `DEFAULT_PROVIDER_RETRY.retry_timeouts is False`
     and `EMBED_CHUNK_RETRY.retry_timeouts is True`, and passes.
  3. Leg (g) is re-run once (MCP then REST, fresh nonces), and **for every embed job whose
     attempt 1 failed with `kind=timeout`**, the server log shows, for that job's label
     (`embed chunkId=…`), either the retry WARNING `transient provider error (… kind=timeout …),
     attempt 1/4` or the gate's INFO `abandoning retry` line; **and at least one such job shows
     the WARNING.** Both lines are absent unless the flip took effect: an unflipped timeout is
     non-transient and re-raises with no log line (§3.6), before the gate is ever read. The INFO
     alternative exists because a job stopped by the gate (a sibling already failed the
     document, likely on MCP) never logs a WARNING; the "at least one WARNING" clause stops an
     all-gated run from passing vacuously. The REST run usually supplies it: its jobs run
     sequentially, so a timed-out embed job is gated only if an earlier job of that document
     already gave up. If no embed job times
     out on attempt 1 in the re-run, condition 3 is **not exercised**: record that, and the flip
     still stands on U-17 alone.
  4. The result is tabulated in the QA report.
- **What does not route back.** If the re-run still ends `failed` on timeouts, the change is still
  done: the classifier fired as designed. The remaining failure is the contention in R-8, and it
  routes to follow-up F-4. It does not reopen this design.
- **K-067 closure rule (applies to every final (g) outcome, flipped or not; v3, review N-5).**
  This change's definition of done is the retry *mechanism*; a (g) that ends `failed` with
  transient kinds in its give-up line proves the mechanism fired, **not** that K-067's
  user-visible goal (a contended ingest reaches `ready`) is met. Therefore:
  - If the final (g) document is `failed`, K-067 closes as **"mechanism delivered"**, and F-4 is
    filed in `falkor-chat/docs/BACKLOG.md` **before K-067 leaves it**, carrying (g)'s per-job
    table as the continuation of K-067's goal.
  - `falkor-chat/docs/HISTORY.md` must not say the field failure is cured; it says the retry
    mechanism shipped and names F-4 as the open continuation.
  - If (g) ends `ready`, the report says **"converged in this run"** (n = 1, nondeterministic),
    never "the MCP path converges".
- **Why pre-authorize instead of redesign.**
  - The 30 s embed timeout retried is cheap and bounded (≤ 180 s).
  - The switch is a reversible one-line constant.
  - A timeout on a ~1 k-character embedding call is far more plausibly queuing behind the
    resident chat model than a "too-slow chunk". The 180 s extract argument is about a chunk that
    is genuinely slow to generate, and it does not apply.
- **A new transient 4xx body.** If leg (g) captures a 4xx whose body names a load, crash, or
  serving failure and matches no marker, add it to `TRANSIENT_DETAIL_MARKERS` with a U-5 row
  (R-1). This is pre-authorized.
- **What does trigger a redesign.** Leg (g) shows a raw non-`ProviderCallError` exception reaching
  `_safe_*`, or a transient shape that needs anything beyond a marker or this one flag. Those are
  ladder or classifier gaps, so they return to `architect` with the captured strings.

**`RetryPolicy` invariant:** `len(delays_s) == max_attempts - 1`, `max_attempts >= 1`, and
`0 <= jitter`. Enforce it in `__post_init__` with `ValueError`. `RetryPolicy(max_attempts=1,
delays_s=())` is the "no retry" policy that tests and any future opt-out use.

### 3.3 Decision 3: idempotency, proven from the write Cypher

**The claim.** Under this design, N failed attempts followed by one success leave the graph in
exactly the state a single first-try success leaves.

**The proof is structural; it does not depend on the writes being idempotent.**
1. The retried region in `extract_chunk` is `extraction.extract(text, llm)` and nothing else.
   Its only side effect is the HTTP POST to the model. It performs **zero graph writes** (§2.1,
   read from `extraction.extract`).
2. `call_with_retry` returns only on success, and only once. The write loop runs after it
   returns, so it runs **at most once per job**, consuming the single successful
   `ExtractionResult`. Failed attempts contribute nothing to the graph: no node, no edge, no
   property.
3. If every attempt fails, or a permanent error occurs, `call_with_retry` raises the last
   exception. The write loop never runs, `_safe_extract` reports `success=False`, and the graph
   is untouched by this job. That is the same as today's failure.
4. "A retry that fires twice" cannot happen. The attempts run in a plain sequential loop on one
   thread, and nothing re-schedules a job. `_schedule_chunk_processing` schedules each
   `(chunk, job)` pair exactly once.
5. A failure **inside** the write loop (FalkorDB error) is not a `ProviderCallError`, so it is
   never retried. It ends in today's partial-write-then-`failed` outcome, unchanged and out of
   scope.
6. The embed retry wraps `_resolve_and_embed`. It writes nothing, and the subsequent
   `set_chunk_embedding` is a `MATCH … SET` that would be idempotent even if repeated.

**This is a structural property, so it is pinned by a test, not just argued.**
- Test I-1 (§5): a real-FalkorDB run with two transient failures, then success, produces exactly
  the counts of a clean run.
- Test U-9 asserts that the stub LLM's `complete` is called N times while the `SpyRepo` write
  calls happen exactly once.

**Rejected: retrying the whole `extract_chunk` from `_safe_extract`.** This is what the brief's
file pointers would suggest. It is idempotent today only because of an unstated invariant, "the
only `ProviderCallError` source precedes every write". The first future edit that adds a
provider call after a write would break that silently. An example is a per-entity embedding,
which is a plausible next feature for fusion. Wrapping only the pure provider region makes
idempotency hold by construction.

**Rejected: making the write loop idempotent** (deterministic `entityId = f(chunkId, index)` plus
`MERGE`). `Entity.entityId` does have a uniqueness constraint (`bootstrap_schema.sh`), so
`MERGE (e:Entity {entityId})` would be legal. But the edges have no key to `MERGE` on:
- `ABOUT` has no properties;
- `RELATES_TO` is "never deduplicated" by contract;
- the exact-tier `SAME_AS` is created inside the same atomic query.

Making all of them retry-safe means redesigning the §14.5/§14.6 write queries, for a phase this
design never repeats. If write-phase failures ever matter, that is a separate design question.

### 3.4 Decision 4: scope — include the embed job, in this change

**Scope: `_safe_embed_chunk`'s path (`EmbeddingWorker.embed_chunk`) is included.** The
stakeholder confirmed this on 2026-10-15 (review OQ-A), and the reviewer's own view agrees.

**Why it is included.**
1. **The evidence shows reported embed failures** (§2.6). All 12 currently `failed` documents in
   `ws:agent-team` have at least one un-embedded chunk, and 21 of 27 chunks lack an embedding.
   - The failing exception's shape was never captured. So this plan makes **no claim** about how
     many of those incidents it would have prevented.
   - Leg (g) captures the shape, and §3.2.1 pre-states the response if it is a timeout.
2. **Same mechanism, same contract.** The embed job reports through the same
   `report_document_job_done`, where one failure fails the document. It calls the same transport,
   which raises the same exception shapes. The documented cold-load HTTP 400 was observed **on
   the embedding model specifically**.
3. **Eviction couples the two jobs.** Extract and embed evict each other's model (§2.5). Retrying
   only one would leave the other job, the one most often hit by a reload, failing on the very
   reload the first job's retry is waiting out.
4. **Marginal cost is small.** The shared `retry.py` helper, one wrap in `embed_chunk`, two
   constructor kwargs, and a few mirrored tests.

**Why the wrap goes inside `embed_chunk`, not `_resolve_and_embed`.** `_resolve_and_embed` is
shared with `embed_message`, which `AgentResponder.maybe_respond` calls synchronously in the
`@mention` reply path (CPG). A 60 s retry budget there would change the responder's latency
contract, and that is out of scope. Within `embed_chunk`, the pre-flight that is now re-run per
attempt is cheap:
- gateway resolution is read-only: in production, `GraphWorkspaceOverrides.get` makes one
  `ro_query` per `resolve`, so there are two graph reads per attempt, with no writes;
- `_index_dimension` is cached per `(ws, label)` after its first successful read;
- `EmbeddingDimensionError` and `ModelResolutionError` are not `ProviderCallError`, so they fail
  fast on attempt 1.

**Not included:** `_safe_embed` → `embed_message` for chat messages. It feeds no document status,
a failed message embed is already silent and non-terminal, and widening it would touch the
responder path.

### 3.5 Decision 5: `Document.status` and `pendingJobs` during retries

**What does not change.**
- **The job still reports exactly once, after its final outcome.** The retry loop lives inside
  `extract_chunk`/`embed_chunk`, below `_safe_*`. `_report_document_job` runs only after that
  method returns (`success=True`) or raises (`success=False`). Its code is unchanged.
- **While a job is retrying,** `pendingJobs` still counts it as outstanding and `status` stays
  `'processing'`. Unless a *sibling* job reports a failure, which is K-051's existing "one failure
  fails the document" rule, unchanged.
- No new status value. No counter changes. `start_document_progress` is unchanged.

**The one addition is a liveness gate, read twice per retry: before the sleep and again after
it, immediately before the next attempt** (v2; review M-1).
- New method: `Repository.chunk_document_is_processing(ws, *, chunk_id) -> bool`.
- Cypher: `MATCH (c:Chunk {chunkId: $chunkId})<-[:HAS_CHUNK]-(d:Document) RETURN d.status =
  'processing'`.
  - `ro_query`.
  - EXPLAIN verified on `ws:agent-team`: `Node By Index Scan (c:Chunk)` → `Conditional Traverse`.
  - Returns `False` when no row is found, meaning the chunk or document was deleted.
- `call_with_retry` takes an optional `keep_going: Callable[[], bool]`. Both workers pass
  `lambda: self._repo.chunk_document_is_processing(ws, chunk_id=chunk_id)`.
- **Pre-sleep read.** If the gate returns `False`, the helper stops and re-raises the last
  transient error without sleeping. This is the read that saves time on the sequential REST
  chain.
- **Post-sleep read.** After `sleep(delay)` returns, the gate is read again. If it is now `False`,
  the helper re-raises the same last transient error without making the next attempt. This is the
  read that sees a delete or sibling failure landing **during** the sleep.
  - v1 read only before the sleep, so it missed both. The review's M-1 is correct. I checked it
    against my own §3.6 contract, where the only gate check sat in the `except` branch before
    `sleep`.
- If the gate *itself* raises, it is treated as `True` (keep retrying) and logged. A FalkorDB
  blip must not turn a recoverable provider error into an early failure.

**Why the gate is needed.**
1. **The sequential REST path would otherwise multiply the budget by the job count during a
   sustained outage.**
   - Concrete example: a 20-chunk document makes 40 jobs, so ×60 s ≈ 40 min of a request's
     background chain sleeping against a dead LM Studio.
   - At ~588 chunks (`MAX_DOCUMENT_CHARS`), it is 1,176 jobs, ≈ 19.6 h.
   - Once the first job exhausts its budget and the document is `failed`, no later job's success
     can revert it (first-terminal-wins). Every further retry is pure waste. With the gate, each
     later job **of that document** costs one attempt plus one indexed read.
   - **The bound is per document, not per request** (review m-2, recomputed). `api.ingest_documents`
     puts every document's jobs on one sequential chain. During an outage, each document's first
     faulting job burns a full budget before that document goes `failed`, so the chain pays one
     budget per document.

     | | outage shape | cost of the chain |
     |---|---|---|
     | 19-document batch (the K-067 field size) | fast-failing | 19 × 40–60 s ≈ **13–19 min** |
     | 19-document batch | worst case, every attempt near the 180 s timeout before a 5xx | < 19 × 780 s ≈ **4.1 h** |
     | 20-document batch (`schemas.MAX_BATCH_SIZE`) | fast-failing | ≈ 13–20 min |

     During this time one limiter token is held, and later documents in the batch sit at
     `processing`.
   - It is bounded and accepted for lab load. A per-request "provider down" breaker shared
     across one chain is follow-up F-3.
2. **A delete during the retry window.** A delete that lands during a sleep would otherwise let a
   late success create orphan `Entity` nodes: `create_entity_with_auto_match` is a bare `CREATE`,
   and the `ABOUT` link's `MATCH` would find no chunk.
   - With the post-sleep read, a delete during any sleep is seen before the next attempt.
   - What remains is the pre-existing race: a delete that lands **during an attempt, or between
     the successful attempt and the end of the write loop**. That window is one attempt (≤ the
     timeout) plus the write loop, the same as without retry for a first-try call.
   - **I rejected the optional third read, just before the write loop.** It would run on the happy
     path for every chunk. It would change semantics: entities of a document already failed by a
     sibling are written today, and they would be skipped. And it still would not close the race,
     because a delete can land between that read and the writes. The race predates this change and
     stays out of scope (R-5).
3. Both reads happen **only on the failure path**, after a transient error, so the happy path pays
   nothing.

**Where the retry sleeps** (the brief's explicit question).
- `call_with_retry` calls the injected `sleep` (`time.sleep` in production) **on the job's own
  thread**:
  - REST: an anyio worker thread under the process-wide limiter (100 tokens). The sleep holds
    one token. It does **not** block the event loop or other requests. It **does** delay the
    rest of that same request's `BackgroundTasks` chain, because the chain runs sequentially.
  - MCP: the job's own daemon thread, which blocks nothing else.
- **Never `asyncio.sleep`.** The `_safe_*` functions and both workers are synchronous by design.
  An async variant would need an event loop that the MCP daemon threads do not have.
- **Token pressure.** Each concurrently running REST background chain holds one token while it
  retries, for up to the §3.2 per-job worst case per faulting document:
  - < 780 s for extract, or < 180 s for embed;
  - typically 40–60 s.

  Sync REST *endpoints* share the same limiter, so 100 simultaneously retrying chains would starve
  them. That needs ≥ 100 concurrent ingest requests, each mid-outage, which is far beyond this
  lab's load. The gate limits each document to one full budget. This is accepted and recorded in
  §6, R-3.

### 3.6 `retry.call_with_retry`: the one helper

```python
def call_with_retry(
    fn: Callable[[], T], *,
    policy: RetryPolicy = DEFAULT_PROVIDER_RETRY,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
    keep_going: Callable[[], bool] | None = None,
    label: str = "",
) -> T:
```

**Contract.**
- Calls `fn()`. Returns its value on the first success.
- On an exception:
  - if `not is_transient(exc, retry_timeouts=policy.retry_timeouts)`, re-raise **the same
    exception object** (`raise`, so the traceback is preserved), with no extra log line;
  - if it was the last attempt, log `WARNING "giving up after %d attempts (%s, kind=%s,
    status=%s): %s"` and re-raise the same object;
  - if `keep_going()` (the pre-sleep read) returns `False`, re-raise the same object;
  - otherwise:
    1. compute `delay = policy.delays_s[i] * (1 + policy.jitter * rand())`;
    2. log `WARNING "transient provider error (%s, kind=%s, status=%s), attempt %d/%d,
       retrying in %.1fs: %s"`;
    3. call `sleep(delay)`;
    4. **then call `keep_going()` again (the post-sleep read)**: if `False`, re-raise the same
       last exception object, with no further attempt;
    5. loop.
- `kind`/`status` are logged via `getattr(exc, "kind", None)` / `getattr(exc, "status", None)`,
  so the next field failure diagnoses itself (review M-2).
- When the gate is the reason to stop: one `INFO` line `"%s: owning document no longer
  processing; abandoning retry (kind=%s, status=%s)"` with `label`, whichever read tripped it.
  (v3: label and kind made explicit so live legs (c′)/(g) can attribute it to one job.)
- No catch of `BaseException`. `KeyboardInterrupt`/`SystemExit` propagate immediately.

It sits in its own module rather than in `background.py` for two reasons:
- `ingestion.py` already imports `background` (`_safe_fuse`), but `embedding.py` must not grow a
  dependency on the scheduling module.
- `transport.py` is the HTTP seam and should not carry retry policy.

## 4. Step-by-step implementation

Three steps. The tree stays green after each one. Steps 1 → 2 → 3 run strictly in order: step 2
needs `retry.py`, and step 3 touches `embedding.py`, which step 1 also edits. All paths are
relative to `falkor-chat/server/` unless rooted otherwise.

| Step | What | Files | Done when |
|---|---|---|---|
| **1** | Classification seam + retry helper (no behaviour change for any caller yet, except the rung-3b fix) | `falkorchat/transport.py` · `falkorchat/llm.py` · `falkorchat/embedding.py` (`OpenAICompatibleEmbedder.embed` only) · **new** `falkorchat/retry.py` · **new** `tests/test_retry.py` · `tests/test_transport.py` · `tests/test_llm.py` · `tests/test_embedding.py` (one U-4 row) | U-1…U-8c (incl. U-5b) green; every existing `test_transport.py`/`test_llm.py`/`test_embedding.py` test unchanged and green (message strings byte-identical) |
| **2** | Extract-job retry + liveness gate | `falkorchat/ingestion.py` · `falkorchat/repository.py` · `tests/test_ingestion.py` · `tests/test_repository.py` · `tests/test_background.py` | U-9, U-9b, U-9c, U-10…U-13, I-1, I-2, I-4 green |
| **3** | Chunk-embed-job retry + docs | `falkorchat/embedding.py` (`EmbeddingWorker`) · `tests/test_embedding.py` · **`tests/test_background.py`** (I-3; v2, review m-5) · `falkorchat/background.py` (docstrings only) · `falkorchat/repository.py` (docstring only) · `falkor-chat/docs/QUERIES.md` §14.1a | U-14…U-17, I-3 green; docs updated as listed below |

Step 1 lists 8 files, over the ~5 target. The 8th is one test row. The `llm.py`/`embedding.py` edits are one keyword
argument per raise site, and `test_llm.py` gets two tests. Splitting them out would leave the
classifier unable to see fallback chains. The review agrees that this need not be forced into a
1a/1b split.

Step 3 lists 6 files. The two `falkorchat/` docstring-only entries carry no logic.

### Step 1: classification seam and `retry.py`

1. **`transport.ProviderCallError`**:
   - Add `__init__(self, message: str, *, kind: str | None = None, status: int | None = None,
     detail: str | None = None, causes: tuple["ProviderCallError", ...] = ())`.
   - Call `super().__init__(message)` and store all four as attributes of the same names.
   - Keep the class docstring and add one paragraph that names the fields and points to
     `retry.is_transient` as their only consumer of record.
2. **`transport.make_http_transport._transport`**: pass the fields at each existing raise, per
   the §3.1 raise-site table.
   - Rung 1 already computes `body`. Pass `status=exc.code, detail=body`.
   - Rung 2: `kind="timeout" if isinstance(exc.reason, TimeoutError) else "connection"`.
   - **Add rung 3a** immediately before rung 3 (review N-1): `except http.client.InvalidURL as
     exc: raise ProviderCallError(f"{label}: malformed request: {exc}",
     kind="malformed_request") from exc`. Order is load-bearing: `InvalidURL` is an
     `HTTPException`, so placed after rung 3 it would be caught there as `"network"`.
   - Rung 3: widen the `except` to `(TimeoutError, OSError, http.client.HTTPException)` (review
     m-3: `BadStatusLine`/`LineTooLong` from `getresponse()` are not `OSError` and escape raw
     today; verified in CPython 3.12.3 `do_open`). Set `kind="timeout" if isinstance(exc,
     TimeoutError) else "network"`. The message format is unchanged.
   - **Add rung 3b:**
     - wrap `raw = resp.read()` with `except (TimeoutError, OSError, http.client.HTTPException)
       as exc: raise ProviderCallError(f"{label}: reading response failed:
       {type(exc).__name__}: {exc}", kind="timeout" if isinstance(exc, TimeoutError) else
       "read") from exc`;
     - keep the `finally` close;
     - add `import http.client`.
   - Update the module docstring's ladder list with rungs 3a and 3b (one item each).
3. **`llm.OpenAICompatibleLLM.complete`/`.chat` and `embedding.OpenAICompatibleEmbedder.embed`**:
   add `kind="response_shape"` to their missing-`choices`/`data` raise.
4. **`llm.FallbackClient.chat`/`.complete`**:
   - collect the caught exceptions alongside the `errors` strings;
   - raise `ProviderCallError(self._errors_message(errors), kind="fallback_exhausted",
     causes=tuple(caught))`;
   - leave the message unchanged.
5. **New `falkorchat/retry.py`**:
   - module docstring stating the §3.1 rules and the §3.2 derivation, briefly; cite this plan
     by path;
   - `TRANSIENT_HTTP_STATUSES`, `TRANSIENT_DETAIL_MARKERS`, `is_transient(exc, *,
     retry_timeouts=False)`;
   - `@dataclass(frozen=True) class RetryPolicy(max_attempts: int, delays_s: tuple[float, ...],
     jitter: float = 0.5, retry_timeouts: bool = False)` with the §3.2 `__post_init__` validation;
   - `DEFAULT_PROVIDER_RETRY` and `EMBED_CHUNK_RETRY` (equal in v2; §3.2);
   - `call_with_retry(...)` exactly per §3.6;
   - logger `logging.getLogger(__name__)`;
   - imports `ProviderCallError` from `.transport` only. No import of `background`,
     `ingestion`, or `embedding`.

### Step 2: extract-job retry and liveness gate

1. **`Repository.chunk_document_is_processing(self, ws: str, *, chunk_id: str) -> bool`**:
   - `self._graph(ws).ro_query("MATCH (c:Chunk {chunkId: $chunkId})<-[:HAS_CHUNK]-(d:Document)
     RETURN d.status = 'processing'", {"chunkId": chunk_id})`;
   - return `bool(rows[0][0])` if there is a row, else `False`.
   - Docstring: what it is for (§3.5); `False` on a missing chunk or document; read-only; and a
     note that it is the K-067 retry gate.
   - Place it next to `report_document_job_done`.
2. **`IngestionPipeline.__init__`**: add keyword-only `retry_policy: RetryPolicy =
   retry.DEFAULT_PROVIDER_RETRY` and `sleep: Callable[[float], None] = time.sleep`, stored
   privately. `app.py` is untouched.
3. **`IngestionPipeline.extract_chunk`**: replace `result = extraction.extract(text, llm)` with:
   ```python
   result = retry.call_with_retry(
       lambda: extraction.extract(text, llm),
       policy=self._retry_policy, sleep=self._sleep,
       keep_going=lambda: self._repo.chunk_document_is_processing(ws, chunk_id=chunk_id),
       label=f"extract chunkId={chunk_id}",
   )
   ```
   - `llm = self._models.llm(...)` stays **outside** the lambda: resolve once, and a
     `ModelResolutionError` fails fast anyway.
   - Add one docstring paragraph: the retried region is the provider call only; the write loop
     runs at most once (§3.3); the K-067 cross-reference.
4. **Test fakes**: `SpyRepo` in `tests/test_ingestion.py` gains `chunk_document_is_processing`,
   returning a scriptable bool, default `True`.

### Step 3: chunk-embed-job retry and documentation

1. **`EmbeddingWorker.__init__`**: add the same two keyword-only args as step 2.2. The
   `retry_policy` default is `retry.EMBED_CHUNK_RETRY`.
2. **`EmbeddingWorker.embed_chunk`**: wrap the `self._resolve_and_embed(...)` call in
   `retry.call_with_retry(...)` with the same `keep_going` gate keyed on `chunk_id`, and
   `label=f"embed chunkId={chunk_id}"`.
   - `set_chunk_embedding` stays outside.
   - **`embed_message` and `_resolve_and_embed` are not touched.**
   - Docstring: one paragraph, as in step 2.3, plus the note that `embed_message` is deliberately
     not retried (§3.4).
   - The test file's `SpyRepo` (`tests/test_embedding.py`) gains the gate method.
3. **Docs (part of this step's definition of done):**
   - `background._safe_extract` and `background._safe_embed_chunk` docstrings: one sentence each.
     A transient provider error is retried inside the worker per `retry.DEFAULT_PROVIDER_RETRY`
     (extract) or `retry.EMBED_CHUNK_RETRY` (embed), and the job still reports exactly once,
     with its final outcome.
   - `background._report_document_job`: one sentence. A retrying job is not reported until it
     finishes.
   - `Repository.report_document_job_done`: one sentence appended to the K-051 paragraph.
     "Failure" here means the job's final outcome after the worker's own bounded transient retry
     (K-067), never one attempt.
   - `falkor-chat/docs/QUERIES.md` §14.1a:
     - add the gate query as a short sub-block with its EXPLAIN shape (`Node By Index Scan
       (c:Chunk)` → `Conditional Traverse`);
     - add one sentence: a reported failure is post-retry.
     - This is the canonical query library, and a new query belongs there (falkor-chat AGENTS
       rule: QUERIES.md is the source of truth for queries).
   - **`falkor-chat/docs/SERVER.md`: no change.** Grep shows it does not describe the
     `pendingJobs`/`Document.status` lifecycle. That lives in QUERIES §14.1a and the docstrings.
     The implementer re-greps (`pendingJobs|report_document_job|K-051`) and reports.
   - **`falkor-chat/AGENTS.md`: no change.** No new knob, env var, or always-loaded invariant
     lands.
   - `BACKLOG.md` (remove K-067) and `HISTORY.md` (one entry) are **`teco`'s at close**, per
     the coordination doc, not this step's.

**Test-run hazard for the implementer.**
- `tests/test_repository.py` contains `wf_repo` users, and that fixture wipes the global
  `reference` graph at setup (`falkor-chat/docs/SERVER.md` §1.7). Run new repository tests by
  node id or `-k chunk_document_is_processing`, never the whole file, unless the coordinator says
  otherwise.
- `test_retry.py`, `test_transport.py`, and `test_llm.py` need no FalkorDB.
- **`tests/test_background.py` changes how it must be run** (review m-5). Today it is fully
  offline (fakes only). I-1/I-3/I-4 add tests that request the `repo` fixture, so the file now
  needs FalkorDB up, and without it those tests **skip** (read `N passed, M skipped`, `SERVER.md`
  §1.7). `repo` wipes only `ws:test`, never `reference`, so running the whole file stays safe for
  shared state.
- Do not run `./scripts/test_queries.sh`. That is the coordinator's job, and step 3 adds a read
  query to QUERIES.md, so the coordinator decides whether a `test_queries.sh` case is wanted.

## 5. Test strategy

The cases below are ordered for red → green, `tdd-engineer`-style, within each step. "Fake
sleep" means a list-appending callable. "Stub LLM" is the existing `_ReplyLLM` pattern, extended
to a `_ScriptedLLM(outcomes)` that raises or returns per call. No test sleeps for real.

### Step 1: unit tests (no FalkorDB)

`tests/test_transport.py` extends the existing per-rung tests, which already drive each rung
through an injected `opener`:

- **U-1:** for each rung, assert the new `kind` (and `status`/`detail` where applicable).
  - Rung 1 with a 400 body `{"error":"terminated"}` → `kind=="http"`, `status==400`, `"terminated"
    in detail`.
  - Rung 2 with `URLError(TimeoutError())` → `"timeout"`.
  - Rung 2 with `URLError(ConnectionRefusedError())` → `"connection"`.
  - Rung 3 with `RemoteDisconnected` → `"network"`.
  - Rung 3 with `http.client.BadStatusLine("x")` raised by the opener → `ProviderCallError`,
    `"network"` (m-3; today this escapes raw).
  - Rung 3a: an opener raising `http.client.InvalidURL("nonnumeric port: 'abc'")` →
    `ProviderCallError`, `kind=="malformed_request"`, and `retry.is_transient(exc) is False`
    (review N-1). This row fails under v2's ladder (rung 3 catches it as `"network"`), so it pins
    the clause order.
  - Rung 5 → `"body_error"` with `detail`.
  - **Every existing message assertion stays as it is**. That is the "message unchanged" pin.
- **U-2 (rung 3b):**
  - a fake response whose `read()` raises `ConnectionResetError` → `ProviderCallError`,
    `kind=="read"`, `close()` still called;
  - `http.client.IncompleteRead(b"")` → `"read"`;
  - `TimeoutError` → `"timeout"`.

`tests/test_llm.py`:

- **U-3:** `FallbackClient` with two scripted elements both raising `ProviderCallError(kind=…)`
  → the raised error has `kind=="fallback_exhausted"` and `causes` holding both, in order. The
  message is unchanged.
- **U-4:** `OpenAICompatibleLLM.complete` on a body missing `choices` → `kind=="response_shape"`.
  `tests/test_embedding.py` adds the same row for `OpenAICompatibleEmbedder.embed` on a body
  missing `data` (the review nit). That makes `test_embedding.py` an 8th step-1 file, a
  one-test addition.

`tests/test_retry.py` (new):

- **U-5, `is_transient` truth table.** A parametrized table over constructed
  `ProviderCallError`s.

  Transient:
  - `http` 500, 502, 503, 408, 429;
  - `http` 400 + `{"error":"terminated"}`;
  - `http` 400 + `Failed to load model "qwen/qwen3-4b-2507". Error: Engine protocol startup was
    aborted.`;
  - `http` 400 + `Error loading model`;
  - case-insensitive `TERMINATED`;
  - `connection`, `network`, `read`;
  - `body_error` with "terminated";
  - `fallback_exhausted` whose causes are all transient.

  Permanent:
  - `http` 400 + `exceed_context_size_error` body (the verbatim shape from §2.2);
  - `http` 400 + Jinja "Only user, assistant and tool roles are supported";
  - `http` 401, 404, 422;
  - `http` 400 with `detail=None`;
  - `timeout`, `malformed_request`, `invalid_json`, `body_shape`, `response_shape`;
  - `kind=None` (a bare `ProviderCallError("x")`);
  - `body_error` without a marker;
  - `fallback_exhausted` with one permanent cause;
  - `fallback_exhausted` with an empty `causes`.

  **Non-`ProviderCallError`** cases (`RuntimeError`, `ConnectionError` raw, `ValueError`,
  `EmbeddingDimensionError`, `ModelResolutionError`) are all permanent.
- **U-5b, timeout switch.**
  - `kind="timeout"` → `is_transient(...)` is `False` by default and `True` with
    `retry_timeouts=True`.
  - `fallback_exhausted` with all-timeout causes follows the same switch.
  - **Both shipped policies have `retry_timeouts is False`.** This pins the v2 default, so a §3.2.1
    flip is a visible, deliberate test edit: on a flip, this row becomes
    `DEFAULT_PROVIDER_RETRY.retry_timeouts is False` and `EMBED_CHUNK_RETRY.retry_timeouts is
    True` (§3.2.1 done-condition 2).
- **U-6, happy-path retry.**
  - `fn` raises transient, transient, then returns `"ok"`. Result `"ok"`, `fn` called 3×.
  - Fake sleep records `[2.0, 8.0]` with `rand=lambda: 0.0`.
  - With `rand=lambda: 0.999…`, the delays are `< [3.0, 12.0]` and `>= [2.0, 8.0]`. This pins the
    upward-jitter floor.
  - One WARNING per retry in `caplog`, naming `attempt 1/4`, `2/4`, and the exception's `kind`
    and `status`.
- **U-7, budget exhausted.**
  - `fn` always raises a transient error. It is called exactly `max_attempts` (4) times, sleep is
    called 3×, and the **last** exception object is re-raised (`is`-identity check on the
    instance from attempt 4).
  - One `giving up after 4 attempts` WARNING carrying `kind`/`status`.
- **U-8, permanent: no retry.**
  - `fn` raises `ProviderCallError(kind="http", status=400, detail='{"error":{"type":
    "exceed_context_size_error"}}')`. Called once, sleep never called, the same object is
    re-raised.
  - Also `RuntimeError`: called once.
- **U-8b, gate.**
  - **Pre-sleep read.** `keep_going` returns `False` after the first transient failure → `fn`
    called once, sleep never called, the transient error is re-raised, and one INFO line is
    logged.
  - **Post-sleep read** (review M-1). `keep_going` scripted `[True, False]` → `fn` called
    **once**, sleep called **once**, and the attempt-1 error object re-raised. v1's single-read
    design fails this test, so it pins the fix.
  - The gate is read in the order pre, sleep, post. Assert on a shared event log that records
    `"gate"` and `"sleep"` entries.
  - `keep_going` raises → treated as `True`: retries continue and an error is logged.
  - `keep_going` is never called on success, on a permanent error, or after the final attempt
    (count its calls).
- **U-8c, `RetryPolicy` validation.**
  - mismatched `delays_s` length → `ValueError`;
  - `max_attempts=0` → `ValueError`;
  - `max_attempts=1, delays_s=()` is valid, and `call_with_retry` then calls `fn` once on a
    transient error.

### Step 2: extract job

`tests/test_ingestion.py` uses `SpyRepo` and `_ScriptedLLM`:

- **U-9, write phase runs once after retries.**
  - LLM: transient ×2, then a valid reply (1 entity, 1 relationship, 2 entities after
    stub-repair).
  - `complete` called 3×.
  - `SpyRepo` records exactly the same `create_entity_with_auto_match`/`link_chunk_about_entity`/
    `create_entity_relationship` calls as a single-success run (compare to a control pipeline fed
    the reply once).
  - **Both pipelines get the same deterministic `id_gen`/`clock` sequence, from a fresh counter
    per pipeline**: construct each with its own `ids = itertools.count(); id_gen=lambda:
    f"id-{next(ids)}"` (and likewise for `clock`). Never share one `itertools.count` instance
    between the two pipelines: the control pipeline would continue the sequence, get different
    ids, and the comparison would go falsely red (Pass 2 nit). With the default `uuid4`, the
    literal comparison could never be equal (review m-4).
  - Fake sleep called 2×.
- **U-9b, the retry boundary itself** (review m-4).
  - The LLM returns a valid reply on the first call.
  - `SpyRepo.create_entity_with_auto_match` raises `ProviderCallError(kind="http", status=503)`.
    This is transient-classified and stands in for a future provider call inside the write loop.
  - Expected: `extract_chunk` raises it with `complete` called **once**, exactly **one** write
    attempt, and **zero** sleeps.
  - This fails if the retry is ever widened to all of `extract_chunk`, which is the regression
    §3.3 rejects and which U-9/I-1 cannot see. It calls `extract_chunk` directly, so it cannot see
    a retry loop added one level up in `background._safe_extract`; U-9c covers that (review N-2).
- **U-10, exhausted.** LLM is always transient → `extract_chunk` raises `ProviderCallError`,
  **zero** `SpyRepo` write calls, `complete` called 4×.
- **U-11, permanent.** LLM raises a 400 `exceed_context_size_error` → raises after 1 call, zero
  writes, zero sleeps.
- **U-12, gate wiring.** `SpyRepo.chunk_document_is_processing` scripted `False` → one attempt,
  then re-raise. Assert the gate was called with the job's own `chunk_id`.
- **U-13, unparseable reply is still not an error.**
  - A garbage reply → no exception, no retry (`complete` called once), zero writes.
  - This guards against someone "helpfully" classifying `_EMPTY` as retryable.

`tests/test_repository.py`, live (`repo`/`conn` fixtures, `ws:test`):

- **I-2, `chunk_document_is_processing` against real FalkorDB.** Create a document via the
  existing repository helpers the other §14 tests use.
  - `True` while `processing`.
  - `False` after `report_document_job_done(success=False)` flips it to `failed`.
  - `False` after it reaches `ready`.
  - `False` for an unknown `chunkId`.
  - `False` after `delete_document`.

`tests/test_background.py` (I-1 and I-4 live; U-9c offline):

- **I-1, idempotency against real FalkorDB.** This is the stated requirement, pinned end to end
  at the `_safe_extract` altitude.
  - Set up two identical documents, `A` and `B`, ingested through `Services.ingest_document` with
    the same one-chunk text.
  - Use the `repo` fixture, and call `repo.start_document_progress(total_jobs=1)` the way
    `_schedule_chunk_processing` would.
  - Run `_safe_extract` on `A` with an `IngestionPipeline` whose LLM is transient ×2 then reply
    `R`. Run it on `B` with an LLM that returns `R` at once. Both use fake sleep.
  - Assert, per document:
    - `status == 'ready'`, `pendingJobs == 0`;
    - the count of `Entity` reached via `(:Chunk)-[:ABOUT]->` from that document's chunk;
    - the count of `RELATES_TO {sourceDocumentId}`;
    - the count of confirmed `SAME_AS` **originating from that document's entities**.
  - The entity, `ABOUT`, and `RELATES_TO` counts must be equal between `A` and `B`.
  - Use **distinct entity names per document** (two replies `R_A`, `R_B` with the same shape) so
    that cross-document exact-tier fusion does not make the `SAME_AS` counts order-dependent.
  - Also assert that the **whole** graph has no `Entity` with zero `ABOUT` in-edges, meaning no
    orphan from a failed attempt.
- **U-9c, the retry boundary at the `_safe_extract` altitude** (offline, fakes; v3, review N-2).
  This is the sibling of U-9b one level up, and the reason it lives here rather than in
  `tests/test_ingestion.py`.
  - A local recording repo in `tests/test_background.py` (same shape as `_RecordingProgressRepo`,
    plus `chunk_document_is_processing → True` and a `create_entity_with_auto_match` that records
    the call and raises `ProviderCallError("x", kind="http", status=503)`).
  - A real `IngestionPipeline(repo, llm, sleep=fake_sleep)` whose LLM returns a valid one-entity
    reply on every call and counts `complete` calls; `fake_sleep` records calls.
  - Call `background._safe_extract(pipeline, "ws", chunk_id, document_id, text)`.
  - Assert: `complete` called **once**; **one** `create_entity_with_auto_match` attempt;
    **zero** sleeps; `report_document_job_done` called **exactly once**, with `success=False`.
  - This fails if a retry is ever added in `_safe_extract` (or anywhere that re-invokes
    `extract_chunk`), since the 503 from the write phase is transient-classified.

- **I-4, `pendingJobs`/`status` during and after retries, two jobs.**
  - `start_document_progress(total_jobs=2)`.
  - Job 1 (extract) retries via a fake `sleep` that **asserts, at the moment it is called**, that
    the live document reads `status == 'processing'` and `pendingJobs == 2`. That pins "a retrying
    job is not counted as reported".
  - Job 1 then succeeds → `pendingJobs == 1`, still `processing`.
  - Job 2 (a `_FailingChunkWorker`-style embed that reports failure) → `failed`.
  - Separately, the **exhausted** variant: an always-transient extract with `total_jobs=1` →
    `failed`, `pendingJobs == 0`, exactly one decrement.

### Step 3: embed job

`tests/test_embedding.py` uses `SpyRepo` and a scripted embedder:

- **U-14:** transient ×2, then a vector → `set_chunk_embedding` called once with that vector, and
  `embed` called 3×.
- **U-15:** `EmbeddingDimensionError` from pre-flight (index dimension mismatch) → raised after
  one pass, zero sleeps, `embed` never called.
- **U-17, embed timeout switch** (§3.2.1).
  - An embedder raising `ProviderCallError(kind="timeout")` once, then succeeding.
  - Under a worker built with the shipped `EMBED_CHUNK_RETRY`: raises after 1 call, zero sleeps.
  - Under `dataclasses.replace(EMBED_CHUNK_RETRY, retry_timeouts=True)`: succeeds after 2 calls,
    1 sleep.
  - The second case is the done-condition test a §3.2.1 flip reuses.
- **U-16:** `embed_message` with a transient-failing embedder → raises on the **first** call, with
  no retry. This pins the §3.4 boundary that keeps the responder path unchanged.

`tests/test_background.py`, live:

- **I-3:** `_safe_embed_chunk` with an embedder that fails transient ×1, then succeeds, on a real
  `ws:test` chunk → the chunk has an embedding, and the document reaches `ready` when
  `total_jobs=1`. Use `ws:test`'s dim-4 index (`SERVER.md` §1.7) with a 4-dim stub vector.

### Live proof run (for the U5 QA unit)

The live proof has two halves, and they prove different things:

| Half | Legs | What it proves | What it does not prove |
|---|---|---|---|
| **A. Proxy legs** | (a)–(d′) | Wiring, timing, the status contract, the gate, rung 3b | Classifier fit. The injected bodies are copies of the strings the allow-list was built from (review M-3). |
| **B. Real-fault legs** | (e)–(g) | Whether the classifier fires on what LM Studio *actually* returns today | — |

Both halves are scaffolding owned by QA. They live in the QA unit's scratch or test-plan
directory, never in `server/`.

**Environment.**
- LM Studio runs on **Windows**. falkor-chat runs in **WSL2** and reaches it at
  `http://localhost:1234` (mirrored networking). The shared `opencode.json` LAN address is known
  to be unreachable from WSL (ws:agent-team KB), so always use a scratch copy pointing at
  `localhost`.
- **The `lms` CLI is reachable from WSL.** It is not on `PATH`, but it runs at
  `/mnt/c/Users/mauri/.lmstudio/bin/lms.exe`, at about 0.3 s per call. I verified on 2026-10-15
  that these subcommands exist: `lms unload [id] / --all`, `lms load`, `lms ps`,
  `lms server start|stop|status`, and `lms log stream`. So legs (e)–(g) need **no human at the
  Windows side**, provided the LM Studio app or daemon is running.
- **Human fallback, only if an `lms.exe` call errors or hangs > 20 s.** The human at the Windows
  desktop, in the LM Studio app's **Developer** tab:
  - **(e):** click **Eject** on the loaded chat model at the moment QA says "now".
  - **(f):** toggle the server **Status: Running** switch off, wait the stated seconds, then
    toggle it on.
  - QA records the wall-clock time the human acted, to correlate with the server log.
- **Shared-instance preconditions (v3, review N-7).** Legs (e)/(f)/(g) (`lms unload`,
  `server stop`) disrupt **every** consumer of the one LM Studio. The likeliest concurrent
  consumer is another agent's learning-capture `ingest_document` into `ws:agent-team`, whose
  server on port 8200 is up on every teco-coordinated session (teco's `SessionStart` hook), so a
  listener check (`ss`) or a loaded-model check (`lms ps`) says nothing about usage. The other
  known consumer is a model-bench sweep (`model-bench/run-sweep.sh`).
  - **Coordinator obligation (addressed to `teco`, before U5 starts legs (e)–(g)).** Either tell
    every in-flight agent not to call `ingest_document` (or anything else that reaches LM Studio)
    until QA reports the window closed, or schedule U5 when no other agent is in flight. QA does
    not start (e) until the coordinator confirms one of the two.
  - **Pre-leg checks: all three, mechanical, immediately before each of (e), (f), (f-12 s), (g)
    and any (g) re-run.** Any miss → do not start the leg; wait and re-check, or ask the
    coordinator.
    1. On `ws:agent-team`, `MATCH (d:Document {status:'processing'}) RETURN count(d)` returns
       **0** (via `GRAPH.RO_QUERY` or the `cypher` MCP). Also record
       `MATCH (d:Document {status:'failed'}) RETURN count(d)` as the **failed baseline**; it was
       12 (339 `ready`) on 2026-10-15.
    2. `pgrep -af '[r]un-sweep|[m]odel-bench'` returns **nothing**. (The bracketed pattern keeps
       a wrapping shell's own command line from matching itself.) Any match is inspected; a
       non-sweep match (an editor on a model-bench file) needs the coordinator's call, not QA's.
    3. `lms.exe log stream`, started at least 60 s before the leg, shows **no request for the
       last 60 s**. I verified only that the subcommand exists, not its output format; QA notes
       in the report what a request line looks like in it.
  - **Post-leg re-check, after each leg and again at teardown.** Re-run check 1's two counts.
    - `processing` must return to **0** within a few minutes; one stuck in `processing` is
      reported.
    - A **non-zero `failed` delta** against the baseline is a reportable side effect: the report
      lists each new failed `Document`'s `documentId`/`title` and the leg it coincided with, and
      the coordinator owns its re-drive (the manual delete-then-re-ingest, §1). QA never re-drives
      a `ws:agent-team` document itself.
    - The report states the delta, including when it is 0.
  - Restore the loaded-model state (`lms ps` before = after) at the end.

**Common wiring (all legs).**
- Throwaway workspace: `EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh retryprobe`, and
  `FALKORCHAT_WS_ID=retryprobe`.
- Uvicorn with `--reload` **off** (`SERVER.md` QA gotchas), on a port other than 8200.
- `FALKORCHAT_OPENCODE_CONFIG` points at a scratch `opencode.json` copy:
  - `baseURL` is `http://localhost:1299/v1` (the proxy) for half A;
  - it is `http://localhost:1234/v1` (LM Studio direct) for half B.
- Capture the server log at INFO. Every retry WARNING now carries `kind`/`status` (§3.6). That is
  the evidence column for half B.

**Timing rule** (review M-3). Any leg that asserts **per-job retry spacing** uses a **1-chunk
document**, so it has exactly one extract job and one embed job.
- The proxy keys its fault script **per request path, counting per distinct request body**
  (sha256 of the body). A retried attempt resends the identical body, so "first N requests"
  means "first N attempts of this job" even on the concurrent MCP path.
- Multi-chunk documents are used only in legs whose assertions are about the final outcome or
  counts, never about spacing.
- **Per-run nonce, not a proxy reset (v3, review N-3).** Every leg-run (each leg × {REST, MCP},
  and every re-run) ingests text whose **first line is a fresh nonce**, e.g. `Probe run
  <uuid4-hex[:8]>.`, generated by QA's driver at ingest time. Within a multi-chunk document the
  chunks must also differ from each other (the proxy keys per body). The proxy is **not**
  restarted or reset between leg-runs; its counters simply never see a repeated body.
  - Why the nonce: without it the REST and MCP runs of one leg send byte-identical chat and
    embedding bodies, and the second run finds the fault counter exhausted and is forwarded with
    zero faults. A reset would fix only that. The nonce also makes each run's text distinct, so a
    re-ingest never triggers `create_document_with_auto_supersede` against the previous run's
    document, and each run's body hashes are self-evidently distinct in the proxy log. It is
    stateless, so a forgotten reset cannot silently void a run.
  - QA records `nonce → documentId → body hashes` per leg-run in the report, so each proxy log
    line maps to one leg-run.

#### Half A: proxy legs (wiring, timing, contract)

- **Proxy.**
  - A ~80-line stdlib `http.server.ThreadingHTTPServer` on `localhost:1299`. It forwards
    verbatim to `localhost:1234`, except on the chosen path, where it first serves the leg's
    scripted faults per body hash.
  - It logs `timestamp, path, body-hash, response` per request.
- **Each leg runs via both REST `POST` ingest and the MCP `ingest_document` tool**, because their
  thread models differ (§2.1).
- **The consumer half is mandatory.** A leg is not done until `search_documents` (or
  `get_document` for a `failed` one) returns the ingested content through the public surface.
- Legs:
  - **(a) `terminated`, 1-chunk, on chat then separately on embeddings.**
    - Script: `400 {"error":"terminated"}` ×2, then `500` HTML ×1, then forward. This is the
      exact K-067 field sequence.
    - Expect: the document is `ready`, with entities or an embedding present.
    - The log shows attempts 1/4, 2/4, 3/4 with `kind=http status=400/400/500`.
    - Proxy gaps are ≥ 2 s, ≥ 8 s, ≥ 30 s.
  - **(b) `permanent`, 1-chunk.**
    - Script: `400` plus the verbatim `exceed_context_size_error` body.
    - Expect: `failed` within ~1 s, exactly **one** proxy hit for that job, and no retry WARNING.
  - **(c) `outage`, 3-chunk, REST.**
    - Script: always `500`.
    - Expect: the first job hits 4 times and the document is `failed` after 40–60 s. Every later
      job of that document makes **exactly one** hit (the gate, §3.5).
    - Record the total drain time.
  - **(c′) gate during sleep, 1-chunk** (v3: re-scripted per review N-4).
    - Script: `500` ×2 on chat, then **forward**. Embeddings are forwarded throughout.
    - Timing: attempt 1 fails, sleep ≈ 2–3 s, attempt 2 fails, then the second sleep, which is
      8–12 s (8 s × jitter `[1, 1.5)`). QA `delete_document`s the document through the public
      surface **3 s after the proxy logs the second chat hit** for this body hash: after the
      pre-sleep gate read (which runs immediately after attempt 2 fails), and well inside the 8 s
      minimum of the sleep.
    - Expect, both assertions discriminating:
      - **Proxy hit count:** exactly **2** chat hits for this body hash, never a third. Without
        the post-sleep read, attempt 3 is forwarded and logged as hit 3.
      - **No orphan:** `MATCH (e:Entity) WHERE NOT ()-[:ABOUT]->(e) RETURN count(e)` on
        `ws:retryprobe` returns **0**. Without the post-sleep read, attempt 3 succeeds and its
        write loop `CREATE`s entities whose `ABOUT` link cannot land on the deleted chunk, so the
        count is > 0. (The per-leg orphan check below must read 0 before this leg starts, so a
        non-zero here is attributable.)
      - The server log shows the INFO "abandoning retry" line for this chunk's label and no
        attempt-3 WARNING.
    - If the delete lands outside the window (the proxy log shows it before hit 2 or after the
      sleep could have ended), the run is void and is repeated with a fresh nonce; it is not
      scored either way.
  - **(d) `reset`, 1-chunk.**
    - Script: accept the connection, close it without a status line, ×1, then forward.
    - Expect: `kind=network`, retried, then `ready`.
  - **(d′) truncated body, 1-chunk** (review M-3; rung 3b).
    - Script: send a `200` status and headers with `Content-Length: 4096`, write 10 bytes, then
      close, ×1, then forward.
    - Expect: `kind=read`, retried, then `ready`.
    - Without this leg, rung 3b is never exercised live.
- **In every retrying leg:**
  - A Cypher read of `Document.status`/`pendingJobs` inside the 8 s window shows `processing`
    with the not-yet-decremented count.
  - After each leg, `MATCH (e:Entity) WHERE NOT ()-[:ABOUT]->(e) RETURN count(e)` returns 0 on
    `ws:retryprobe`.

#### Half B: real-fault legs (classifier fit; no synthetic bodies)

These run direct to LM Studio. **Record, per job, the first-attempt `kind`/`status`/`detail`
prefix, the attempts used, and the final status** in a table in the QA report. Optionally run
`lms log stream` in parallel to capture LM Studio's side.

- **(e) Unload mid-generation.**
  - Ingest a 1-chunk document with a long chunk (~1,000 chars, entity-dense) via REST.
  - About 1–2 s after the request returns, while extract is generating (`lms ps` shows the chat
    model), run `lms.exe unload qwen/qwen3-4b-2507`.
  - Record what the in-flight request surfaced as, and whether the retry recovered after the JIT
    reload.
  - This is the closest reproducible relative of "terminated".
- **(f) Server bounce.**
  - Same document shape. While extract is in flight, run `lms.exe server stop`, wait **5 s**, then
    `lms.exe server start`.
  - Expect `kind=connection` or `network` on the attempts during the bounce, then recovery within
    the 2/10/40 s cumulative schedule.
  - Repeat once with a **12 s** wait, which lands between attempts 3 and 4.
- **(g) Unproxied contention. This is the actual K-067 trigger and the acceptance evidence for
  the MCP path (R-8).**
  - Run `lms.exe unload --all`, so both models start cold.
  - **MCP**-ingest one 8-chunk document, which schedules 16 concurrent jobs. Then repeat via REST.
  - Record:
    - every first-attempt failure's `kind`/`status`/`detail` prefix;
    - attempts per job;
    - the final document status;
    - wall time.
  - **Pre-stated readings:**
    - **Any embed job with `kind=timeout`** → apply §3.2.1. Flip `EMBED_CHUNK_RETRY.retry_timeouts`,
      U-17 passes, and (g) is re-run once.
    - **A transient-looking 4xx body that matches no marker** → add the marker per §3.2.1 and
      re-run.
    - **A raw non-`ProviderCallError` in the log** → return to `architect`.
    - **Document `ready`** → "converged in this run" (n = 1); not a convergence claim for the
      MCP path.
    - **Document `failed` with the classifier firing**, meaning WARNING lines show transient
      kinds and the final give-up names a transient kind → this change is still done. R-8 is
      confirmed, and the §3.2.1 **K-067 closure rule** applies: F-4 filed with (g)'s table
      before K-067 leaves `BACKLOG.md`, and no "cured" claim in `HISTORY.md`.

#### Teardown

- `GRAPH.DELETE ws:retryprobe`.
- Stop the proxy.
- Restore the loaded-model state.
- Final `ws:agent-team` re-check and failed-count delta (Environment, post-leg re-check).
- Leave the shared `opencode.json` untouched.

### Acceptance mapping (coordination definition of done → evidence)

| DoD item | Evidence |
|---|---|
| transient retried with bounded backoff | U-6, U-9, U-14, I-1, I-3, live (a)/(d)/(d′), (e)/(f) if they surface transient kinds |
| bad chunk fails fast | U-8, U-11, U-15, live (b) |
| exhausted budget → `failed` | U-7, U-10, I-4 (exhausted), live (c) |
| retried extract idempotent on chunk + document state | I-1, U-9, U-9b, U-9c, live orphan query, (c′) |
| classifier fires on real LM Studio faults | live (e)/(f)/(g) table, read per the pre-stated rules above |
| docs updated | step 3 docs list |

## 6. Risks & open questions

**Risks**

- **R-1: an unknown transient shape is not retried.**
  - The marker list is an allow-list built from captured field strings. A transient LM Studio
    error with a new body text and a 4xx status fails fast, which is today's behaviour, so this
    is not a regression.
  - Mitigation:
    - the retry and give-up WARNINGs carry `kind`/`status`, and `_safe_*`'s exception log
      carries the full message;
    - adding a marker is a one-line change to `retry.TRANSIENT_DETAIL_MARKERS` plus one row in
      U-5, pre-authorized by §3.2.1;
    - legs (e)–(g) are the first real-fault sample.
- **R-2: a permanent error shaped like a transient one burns one budget.**
  - Examples: a wrong host or port (`connection`), or a missing model that LM Studio reports as
    "Failed to load model".
  - Cost: one per-job budget (§3.2 table) before the same `failed` outcome. That is typically
    40–60 s, because refused connections fail instantly.
  - The gate caps it to the first job per document on REST. On MCP, all of a document's jobs run
    concurrently, so wall time is one budget.
  - Accepted.
- **R-3: anyio limiter occupancy.**
  - A retrying REST chain holds one of the 100 worker-thread tokens for up to the per-job worst
    case per faulting document:
    - < 780 s for extract, or < 180 s for embed, from §3.2;
    - typically 40–60 s;
    - for a batch, per document, which §3.5 point 1 quantifies.
  - Sync REST endpoints share that limiter. Starving it needs ~100 concurrent chains, each
    mid-outage. That is far outside lab load.
  - Accepted, and recorded so a future load test reads the number correctly.
- **R-4: the sequential REST chain gets slower under transient faults, by design.**
  - Later jobs wait behind an earlier job's backoff, so a document stays `processing` longer.
  - In a batch, later **documents** also wait, at one budget per faulting document (§3.5 point 1:
    13–19 min for a 19-document batch in a fast-failing outage).
  - Previously the document went `failed` fast and was lost, so this is the intended trade.
    F-3 is the follow-up if batch latency matters.
- **R-5: the residual delete race.**
  - A `delete_document` that lands **during an attempt, or between the successful attempt and the
    end of the write loop**, can still leave orphan `Entity` nodes, because
    `create_entity_with_auto_match` is a bare `CREATE`.
  - That window is the same one a first-try call has today.
  - With the post-sleep gate read (v2), no sleep window widens it.
  - I rejected the optional third, pre-write read (§3.5 point 2).
  - Pre-existing, out of scope.
- **R-6: rungs 3a, 3 and 3b change a shared contract.**
  - Callers that today see raw `ConnectionResetError`/`IncompleteRead`/`BadStatusLine`/`InvalidURL` will see
    `ProviderCallError`. Every caller already handles `ProviderCallError` as "the provider failed".
    `FallbackClient` now falls through on these errors, which is an improvement.
  - **One observable change** (v3: Pass 2 nit; I re-read `executor.py:1066-1097` and agree):
    1. **Workflow start.** `Services`' drive-fault catch (`services.py`, the `except
       (NotImplementedError, WorkflowConfigError, ModelResolutionError, ProviderCallError)`
       block) now catches these. A drive that hits one returns a `failed` envelope instead of
       propagating as a 500.
  - **No executor-visible change.** The tool dispatch is `try: … except ServiceError as exc:`,
    and both the `tool %s failed: %r` WARNING and the tool-trace line live inside that branch.
    Neither the old raw exceptions nor `ProviderCallError` (a `RuntimeError`) is a
    `ServiceError`, so neither enters it, before or after this change: no log line, no trace
    text, no control-flow difference. (v2 claimed the `%r` line changed; it does not.)
  - The check is the step-1 gate: U-1/U-2 pass, and the existing `test_llm.py`/executor tests
    stay green.
- **R-7: no schema, RAM, or topology change.** The gate is up to two indexed reads per retry, on
  the failure path only. Rollback is a plain revert. There is no data migration and no stored
  format change.
- **R-8: possible non-convergence on the MCP path** (review M-4). **This is inferred, not run.**
  - **The concern.** MCP runs all of a document's jobs concurrently, and LM Studio holds one model
    at a time. So a document's failed embed and extract jobs retry in the **same** jittered
    windows, and each window again mixes chat and embedding requests against the single slot.
    The gate cannot trip early here, because no job reports until its budget ends. Upward jitter
    spreads requests within a kind; it does not separate the two kinds. A multi-chunk MCP
    document can therefore exhaust its budget **because of its own retry traffic**. The worst
    outcome is today's `failed`, reached up to one budget later.
  - **Decision: no design mitigation in this change.**
    - The only mitigation I would trust is to separate the two kinds in time, for example per
      document, embeds then extracts on MCP. That is a scheduling change to `mcp._default_schedule`
      / `_schedule_chunk_processing`, a behaviour change in its own right, and it is the same
      family as F-1.
    - A cheaper timing trick (different delay schedules per kind) cannot guarantee separation,
      because attempt durations vary from under a second to many seconds of generation. It also
      cannot be validated offline, so it would be a guess shipped as a fix.
  - **Acceptance.** Live leg (g) is the acceptance evidence for MCP. Its pre-stated readings (§5)
    route a "classifier fires, still `failed`" result to **F-4**, not to a defect in this change,
    under the §3.2.1 K-067 closure rule (F-4 filed before K-067 leaves `BACKLOG.md`; no "cured"
    claim in `HISTORY.md`).
  - REST is less exposed, because its chain is sequential, which is why F-1 helps it directly.

**Follow-up candidates** (for `teco` to file, not part of this change):

- **F-1: reduce model-eviction thrash at the source on REST.**
  - `_schedule_chunk_processing` interleaves `embed(c1), extract(c1), embed(c2), …`. On the
    sequential REST path, with the one-resident-model behaviour (§2.5), that forces a model swap
    on **every** job.
  - Scheduling all embeds, then all extracts, would cut swaps per document from ~2N to ~2.
  - It is a behaviour change to scheduling order, so it needs its own small design and a live
    measurement.
- **F-2: re-drive the 12 already-`failed` `ws:agent-team` documents** (§2.6), using the existing
  delete-then-re-ingest procedure, once this lands.
- **F-3: per-request "provider down" breaker on the REST chain** (review m-2).
  - One shared flag per `BackgroundTasks` chain: after one job exhausts its budget on a transient
    kind, later documents in the same chain make one attempt and no sleeps.
  - This turns a batch's outage cost from one budget per document into one per request.
- **F-4: per-document job serialization on MCP** (R-8), so that a document's embeds and extracts
  never contend with each other.
  - Filed only if leg (g) shows MCP non-convergence.
  - Carry (g)'s measured table as the evidence.

**Open questions** (none blocks implementation; each has a default already taken)

- **OQ-1: retry timeouts.**
  - **Embed:** settled in advance by §3.2.1. It flips to on if leg (g) shows embed timeouts.
  - **Extract:** stays off. Revisit only if the field shows `kind="timeout"` extract failures that
    succeed on a manual retry. The change would then be a per-kind attempt cap, not a plain flip,
    because 180 s × 4 is too long.
- **OQ-2: operator-tunable policy?** The default is module constants (§3.2). If an operator needs
  to tune it without a code change, add an overlay `retries.<kind>` key next to
  `timeouts.<kind>`, and give `falkor-chat/AGENTS.md` its one-line knob mention at that time.
- **OQ-3: the `"channel error"` marker.** It is inferred, not captured (§3.1). Leg (g) may surface
  it. If so, record its status and body in U-5's table.

## 7. Ready to implement

**Implement the plan at `falkor-chat/docs/plans/extract-transient-retry.md` (v3), steps 1 → 2 → 3,
test cases per §5.**

Digest:
- **Exception fields.** `ProviderCallError` gains additive `kind`/`status`/`detail`/`causes`
  fields. Messages are unchanged.
- **Transport ladder.** Rung 3 now also catches `http.client.HTTPException`, a new rung 3a ahead
  of it maps `http.client.InvalidURL` to permanent `malformed_request`, and a new rung 3b
  covers `resp.read()`. Together they close the raw-exception escapes.
- **Classifier.** A new `falkorchat/retry.py` provides an allow-list classifier: every 5xx, 408,
  429, connection-level failures, and a 4xx only when its body carries an LM Studio transient
  marker. Timeouts are behind a per-policy switch that is **off** in both shipped policies.
  §3.2.1 pre-states when the embed policy flips it.
- **Backoff.**
  - 4 attempts at 2, 8, 30 s with upward jitter: 40–60 s of sleep.
  - The per-job worst case is < 780 s for extract and < 180 s for embed.
  - Module constants (`DEFAULT_PROVIDER_RETRY`, `EMBED_CHUNK_RETRY`) plus constructor injection.
    No new knob.
- **Retry scope.** The retry wraps **only** the provider call inside
  `IngestionPipeline.extract_chunk` and `EmbeddingWorker.embed_chunk`. The write phase runs at most
  once, so idempotency holds by construction. U-9b and U-9c pin the boundary.
- **Gate.** A `Repository.chunk_document_is_processing` gate is read before **and after** each
  sleep. It stops retrying once the document is terminal or gone.
- **Jobs covered.** Both the extract and the chunk-embed jobs are in scope: stakeholder-confirmed,
  sharing the mechanism, and with reported embed failures in the field. The failing shape was
  never captured, and live leg (g) captures it. `embed_message` and the responder path are
  untouched.
- **K-051.** The reporting code is unchanged: each job reports once, with its final outcome.
- **Docs.** Docstrings plus `QUERIES.md` §14.1a. No change to `SERVER.md` or `AGENTS.md`.
- **Live proof.** Proxy legs (a)–(d′) and real-fault legs (e)–(g). `lms.exe` is reachable from
  WSL, so no human is needed unless it fails.

## 8. Review dispositions

### 8.1 Pass 1 (v2)

| Finding | Disposition |
|---|---|
| **M-1** gate read only before the sleep | **Fixed.** The review is correct. I confirmed it against my own v1 §3.6 contract, where the only `keep_going` call was in the exception branch before `sleep`. There is now a post-sleep read before the next attempt (§3.5, §3.6), plus a U-8b `[True, False]` test and live leg (c′). The optional pre-write third read is **declined**: it runs on the happy path, changes what a sibling-failed document gets written, and still cannot close the race (§3.5 point 2, R-5). |
| **M-2** embed premise overstated; failure shape unknown | **Fixed in wording** (§2.6, §3.4, §7). No claim about incidents prevented. Retry and give-up logs now carry `kind`/`status`. Timeout handling is per policy, and the embed outcome is pre-stated (§3.2.1, answering OQ-B). |
| **M-3** live proof cannot catch classifier misfit | **Fixed.** Real-fault legs (e)/(f)/(g), truncated-body leg (d′), gate-during-sleep leg (c′), the single-chunk timing rule, and proxy fault scripts keyed per body hash (§5). `lms.exe` reachability from WSL is verified, and a human fallback is named. |
| **M-4** possible MCP non-convergence | **Accepted as risk R-8. No design mitigation**, with reasons (R-8). Leg (g) is the acceptance evidence, with pre-stated routing to F-4. |
| **m-1** worst case understated | **Fixed.** Recomputed in §3.2: extract < 4×180 + 60 = 780 s, embed < 4×30 + 60 = 180 s, typical 40–60 s. R-2/R-3 now cite this table. |
| **m-2** gate bounds per document, not per request | **Fixed.** Recomputed in §3.5 point 1: a 19-document batch costs 13–19 min in a fast-failing outage, and < 4.1 h worst case. Per-request breaker filed as F-3. |
| **m-3** `HTTPException` from `getresponse()` escapes rung 3 | **Fixed.** Rung 3 is widened; this is step 1.2, with a new U-1 row. |
| **m-4** U-9/I-1 do not pin the boundary; uuid comparison | **Fixed.** U-9b added, and U-9 uses a deterministic `id_gen`/`clock`. |
| **m-5** I-3's file missing from step 3; `test_background.py` now needs FalkorDB | **Fixed.** Step table updated, and a test-run hazard bullet added. |
| Nit: "config only, no I/O" | **Fixed:** now says read-only (§2.1, §3.4). |
| Nit: embedder `response_shape` untested | **Fixed:** a U-4 row in `test_embedding.py`, added to the step 1 file list. |
| Nit: R-6 observable changes | **Partly disputed.** The `services.py` drive-fault change is confirmed and stated. The executor **tool-trace** text does not change; only the `%r` log line does (R-6). |
| OQ-A | Settled by the stakeholder: embed is in scope. |
| OQ-B | **Pre-authorized:** a bounded embed-only timeout retry (§3.2.1), with numbers and a done-condition. A redesign is triggered only by ladder or classifier gaps. |

### 8.2 Pass 2 (v3)

Every Pass 2 finding is accepted; I verified each against the code or runtime before applying it.
One fix was tightened beyond the review's wording (N-5), and one log-line precision was added in
support of it.

| Finding | Disposition |
|---|---|
| **N-1** rung 3 classifies `http.client.InvalidURL` as `"network"` | **Fixed.** Re-verified in `falkor-chat/server/.venv` (CPython 3.12.3): MRO is `InvalidURL → HTTPException → Exception → BaseException → object`, not `ValueError`; `urlopen` raises it for `http://localhost:abc/…` and for a space in the host. New **rung 3a**, a dedicated `except http.client.InvalidURL` placed immediately before rung 3, → `kind="malformed_request"` with rung 4's message format (§3.1 raise-site table and rung-3a paragraph; step 1.2). I chose the dedicated clause over moving rung 4 above rung 3, so rung 4 and its existing tests are untouched. U-1 gains the `InvalidURL("nonnumeric port: 'abc'")` row asserting `malformed_request` and `is_transient is False`; it fails under v2's ladder. R-6 now lists `InvalidURL` among the raw escapes this closes. |
| **N-2** U-9b cannot see a retry in `_safe_extract` | **Fixed by adding U-9c**, not by narrowing: `_safe_extract` is a plausible place for a future "helpful" retry, and an offline test there is cheap. U-9c (§5 step 2, `tests/test_background.py`, offline) drives a real `IngestionPipeline` through `background._safe_extract` with a write-phase 503 and asserts `complete` ×1, one write attempt, zero sleeps, exactly one `report_document_job_done(success=False)`. U-9b's claim now names `extract_chunk` only. Step-2 table and acceptance mapping updated. |
| **N-3** per-body-hash counters collide across REST/MCP runs | **Fixed with a per-run nonce** (§5 "Timing rule"), not a proxy reset: the nonce also stops same-text auto-supersede between runs, makes run separation visible in the proxy log, and has no state to forget. |
| **N-4** (c′) orphan assertion vacuous under always-500 | **Fixed** (§5 leg (c′)): `500` ×2 then forward, delete 3 s after the second chat hit (inside the second sleep's 8 s minimum, after the pre-sleep read). Hit-count (exactly 2) and orphan (0) each fail without the post-sleep read. Added a void-run rule for a delete that misses the window. |
| **N-5** OQ-B done-condition tautological; U-5b edit; closure rule | **Fixed, with one correction to the review's proposed condition.** The review's "every timeout-at-attempt-1 embed job shows the `attempt 1/4 … kind=timeout` WARNING" keeps the false-fail it identifies: a job stopped by the pre-sleep gate logs INFO, not WARNING (§3.6 reads the gate before the WARNING). §3.2.1 condition 3 therefore accepts WARNING **or** the gate's INFO per job, and requires **at least one** WARNING; both lines are absent without the flip, so it stays falsifiable. To make the INFO attributable per job, §3.6's INFO line now carries `label`/`kind`/`status`. Condition 2 is the U-5b row edit (also noted at U-5b). The **K-067 closure rule** is stated once in §3.2.1 and cited from (g) and R-8; (g)'s `ready` reading says "converged in this run". |
| **N-7** shared-instance check not mechanical | **Fixed** (§5 "Environment"): coordinator obligation addressed to `teco`; three pre-leg checks with pass values (0 `processing` on `ws:agent-team`, empty `pgrep`, 60 s with no request in `lms log stream`), before each of (e)/(f)/(g) and re-runs; post-leg and teardown re-check with the `failed`-count delta always reported (baseline 12 `failed` / 339 `ready`, re-read 2026-10-15); QA never re-drives an `ws:agent-team` document. Two refinements over the review: the `pgrep` pattern is bracketed so a wrapping shell cannot self-match, and the `lms log stream` output format is flagged as unverified. |
| **N-6** | Named in the review's verdict ("N-1, N-2 and N-6") but Pass 2 contains no N-6 body. I read it as the U-9 fresh-counter nit below; if something else was meant, it is not addressed here. |
| Nit: R-6 item 2 overcorrects | **Accepted.** Re-read `executor.py:1066-1097`: the `tool %s failed: %r` WARNING sits inside `except ServiceError`, so it fires for neither the old raw exceptions nor `ProviderCallError`. R-6 now states one observable change (the `services.py` drive-fault catch) and **no executor-visible change**; the 8.1 "Partly disputed" row is superseded by this one. |
| Nit: U-9 "same deterministic `id_gen`" | **Accepted.** U-9 now requires a fresh counter per pipeline, never a shared `itertools.count`. |
