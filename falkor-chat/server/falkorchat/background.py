"""Out-of-band post-message policy, shared by both transports (K-041).

`_safe_embed`/`_safe_respond`/`_safe_run_workflow` encode the M3 one-handler
guarantee — exactly one of {trigger, responder} handles a posted message,
embedding is independent of both — and must run failure-isolated (never raise
into the caller's scheduling mechanism). Both `api.py` (FastAPI
`BackgroundTasks`) and `mcp.py` (a `threading.Thread` fire-and-forget, since
MCP tools have no per-call background-task object) schedule these same three
functions after a successful `post_message`, so the policy is defined exactly
once here instead of drifting between two copies.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from . import update_detection
from .config import CallContext

_log = logging.getLogger(__name__)

# Suggested-tier noise floor (document-ingestion2 Stage D, ML note §3.2/§5):
# an implementer-tunable minimum Jaccard ratio below which a candidate never
# reaches the pending queue at all — a usability floor against alert fatigue,
# not a correctness gate (mirrors `MAX_DOCUMENT_CHARS`'s "implementer-tunable,
# not load-bearing" posture). NOT the same knob as `create_or_reopen_
# supersede_suggestion`'s own semantics — this only decides whether to call
# it at all.
_UPDATE_DETECTION_NOISE_FLOOR = 0.1


def _default_id() -> str:
    return uuid.uuid4().hex


def _default_clock() -> int:
    """Server clock in milliseconds since the epoch — mirrors `ingestion.
    _default_clock`/`services._default_clock`'s identical shape."""
    return int(time.time() * 1000)


def _safe_embed(embed_worker: Any, ws: str, msg_id: str, text: str) -> None:
    """Embed a posted message out-of-band, swallowing+logging any failure.

    Runs off-band (DECISION 3, K-008 posture): EVERY posted message is
    embedded so the retrievable corpus grows, but a message is readable
    before its embedding lands and an embedder hiccup must never surface to
    the poster. Embedding is a pure write — it never triggers a response.
    """
    try:
        embed_worker.embed_message(ws, msg_id=msg_id, text=text)
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception("background embed failed (msgId=%s)", msg_id)


def _report_document_job(
    worker_or_pipeline: Any, ws: str, document_id: str, chunk_id: str,
    *, success: bool, kind: str,
) -> None:
    """Report one chunk-level background job's outcome back onto its owning
    `Document` (K-051 — closes the "`Document.status` never reaches a
    terminal state" defect).

    `worker_or_pipeline.repo` is the injected repository `EmbeddingWorker`/
    `IngestionPipeline` already carry (K-051 adds the public `.repo`
    accessor to both) — sourced this way rather than threaded through every
    `_schedule_chunk_processing` call site, since both objects are already
    constructed against the very same `Repository` instance in production
    (`app.py`). A worker/pipeline with no `.repo` at all (every pre-K-051
    test fake in this module and in `test_api.py`/`test_mcp.py`) silently
    skips the report instead of raising — those fakes exist to pin
    scheduling behavior, not document-status bookkeeping, and must keep
    working unmodified. The report call itself is guarded the same
    failure-isolation way as the job it reports on: it must never raise into
    the caller's scheduling mechanism either.
    """
    repo = getattr(worker_or_pipeline, "repo", None)
    if repo is None:
        return
    try:
        repo.report_document_job_done(ws, document_id=document_id, success=success)
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception(
            "failed to report background %s completion (chunkId=%s)", kind, chunk_id
        )


def _safe_embed_chunk(
    embed_worker: Any, ws: str, document_id: str, chunk_id: str, text: str
) -> None:
    """Embed an ingested document's chunk out-of-band, swallowing+logging any
    failure (K-050 M5 Stage 2 — mirrors `_safe_embed` exactly).

    Runs off-band, decoupled from `services.ingest_document`'s write path: a
    chunk is readable before its embedding lands, and an embedder hiccup for
    one chunk must never surface to the ingesting caller nor corrupt the
    `Document` or block sibling chunks (same failure-isolation discipline as
    every other `_safe_*` wrapper in this module). K-051: either way, the
    outcome is reported back onto `document_id` via `_report_document_job` —
    this is one of the two paths (`_safe_extract` is the other) whose
    completion `Document.status` needs to reach a terminal state.
    """
    try:
        embed_worker.embed_chunk(ws, chunk_id=chunk_id, text=text)
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception("background chunk embed failed (chunkId=%s)", chunk_id)
        _report_document_job(
            embed_worker, ws, document_id, chunk_id, success=False, kind="embed"
        )
        return
    _report_document_job(
        embed_worker, ws, document_id, chunk_id, success=True, kind="embed"
    )


def _safe_extract(
    ingestion_pipeline: Any, ws: str, chunk_id: str, document_id: str, text: str
) -> None:
    """Extract an ingested document's chunk into entities/relationships
    out-of-band, swallowing+logging any failure (K-050 M5 Stage 3 — mirrors
    `_safe_embed_chunk` exactly).

    Runs off-band, independent of `_safe_embed_chunk` for the same chunk (both
    are scheduled side by side from `api.py`/`mcp.py`'s `ingest_document`, not
    chained) and decoupled from `services.ingest_document`'s write path: a
    chunk is readable — and its embedding may already be searchable — before
    its extraction lands. An extraction failure for one chunk must never
    surface to the ingesting caller, corrupt the `Document`, or block sibling
    chunks (same failure-isolation discipline as every other `_safe_*`
    wrapper in this module). K-051: either way, the outcome is reported back
    onto `document_id` via `_report_document_job`.
    """
    try:
        ingestion_pipeline.extract_chunk(
            ws, chunk_id=chunk_id, document_id=document_id, text=text
        )
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception("background extract failed (chunkId=%s)", chunk_id)
        _report_document_job(
            ingestion_pipeline, ws, document_id, chunk_id, success=False, kind="extract"
        )
        return
    _report_document_job(
        ingestion_pipeline, ws, document_id, chunk_id, success=True, kind="extract"
    )


def _schedule_chunk_processing(
    schedule: Callable[..., None],
    ws: str,
    document_id: str,
    chunks: list[dict[str, Any]],
    *,
    embed_worker: Any | None,
    ingestion_pipeline: Any | None,
) -> None:
    """Schedule the per-chunk embed+extract background jobs for one ingested
    document (K-050 M5 Stage 6a).

    Factored out of `api.py`'s and `mcp.py`'s `ingest_document` handlers
    (previously duplicated inline, one copy per transport) so the batch path
    (`ingest_documents`, both transports) doesn't triple it into a third
    copy — every caller now shares this one loop.

    `schedule` is either transport's own scheduling primitive: REST passes
    `starlette.BackgroundTasks.add_task`, MCP passes its own `_schedule`
    module seam (itself swappable in tests for synchronous execution,
    `mcp._default_schedule`'s docstring) — both match the `(fn, *args) ->
    None` shape every `_safe_*` function above expects, so this helper stays
    transport-agnostic.

    `embed_worker`/`ingestion_pipeline` are each independently optional,
    mirroring `ingest_document`'s existing posture exactly: neither wired is
    a no-op, either wired schedules only that job per chunk, both wired
    schedules both jobs per chunk, independently (neither blocks the other,
    same as before this was factored out).

    K-051: before scheduling anything, initializes `document_id`'s
    outstanding-job count (`repository.start_document_progress`) to however
    many `_safe_embed_chunk`/`_safe_extract` calls this method is about to
    schedule (one per chunk per wired worker) — synchronously, on the
    calling thread, strictly before any of those jobs can start running, so
    `_report_document_job`'s later decrements never race an uninitialized
    counter. Sourced via whichever of `embed_worker`/`ingestion_pipeline` is
    wired (`.repo`, K-051's new accessor on both) — a no-op when neither
    exposes one (pre-K-051 test fakes).

    **Always calls `start_document_progress`, even when `total_jobs == 0`**
    (K-051 review MINOR 2 — a document with zero chunks): `total_jobs == 0`
    is exactly what `start_document_progress` itself treats as "nothing to
    wait for" and flips straight to `'ready'`. Skipping the call here would
    leave such a document parked at `'processing'` forever, safe today only
    because `Services.ingest_document` never produces a zero-chunk document
    (`EmptyDocumentError` two layers up) — an invariant this method no
    longer needs to trust.
    """
    total_jobs = (len(chunks) if embed_worker is not None else 0) + (
        len(chunks) if ingestion_pipeline is not None else 0
    )
    repo = getattr(embed_worker, "repo", None) or getattr(
        ingestion_pipeline, "repo", None
    )
    if repo is not None:
        repo.start_document_progress(
            ws, document_id=document_id, total_jobs=total_jobs
        )
    for chunk in chunks:
        if embed_worker is not None:
            schedule(
                _safe_embed_chunk, embed_worker, ws, document_id,
                chunk["chunkId"], chunk["text"],
            )
        # K-050 M5 Stage 3: extraction is scheduled independently of embedding
        # for the same chunk — neither blocks the other.
        if ingestion_pipeline is not None:
            schedule(
                _safe_extract, ingestion_pipeline, ws,
                chunk["chunkId"], document_id, chunk["text"],
            )


def _safe_detect_update(repo: Any, ws: str, document_id: str) -> None:
    """FR-2/AC-2 suggested-tier update detection for a just-ingested document
    that was NOT auto-superseded (document-ingestion2 Stage D, plan §3.4) —
    mirrors `_safe_extract`'s try/except-log-never-raise isolation
    discipline, but **deliberately without `_report_document_job`**: per the
    ML note's own scope note, a detection failure is a *soft* failure (the
    document is still valid and independently searchable) and must never
    flip `Document.status` to `'failed'` the way an extraction/embedding
    failure does — detection's completion signal is simply the presence/
    absence of a pending `SUPERSEDES` edge, never folded into `pendingJobs`.

    Re-fetches the document's own `title`/`text` via `repo.get_document`
    rather than having every call site thread them through — both transports
    already have them at the point they'd schedule this, but re-deriving the
    *effective* title (the `title or source_label or ""` fallback
    `services.ingest_document` applies) would duplicate that logic in two
    transport layers; a single indexed-`documentId` lookup is cheap and
    avoids that drift risk. Returns silently (no-op) if the document is
    already gone by the time this runs (a delete raced the background job —
    same "MATCH-anchored, degrades safely" posture as every other background
    write in this module, §2.1 of the plan).

    Narrows candidates via `repo.find_update_shortlist` (two unioned
    signals — LSH/MinHash band-equality + title-fuzzy full text, both
    already scoped to `currentVersion` documents only inside that method),
    then computes the precise `update_detection.jaccard` ratio in Python
    against the (small, bounded) shortlist only — the bands only ever narrow
    *candidates*, never substitute for the exact metric (plan §3.4 step 2).
    For the top-ranked candidate above `_UPDATE_DETECTION_NOISE_FLOOR`:
    records a `pending` suggestion via `repo.create_or_reopen_supersede_
    suggestion` — the exact `create_or_reopen_match` idiom, ML note §7 — which
    gets OQ-3's reopen-on-corroboration behavior for free, zero new logic.
    """
    try:
        document = repo.get_document(ws, document_id=document_id)
        if document is None:
            return  # deleted before this job ran — nothing to detect
        text = document.get("text") or ""
        title = document.get("title") or ""
        doc_shingles = update_detection.shingles(update_detection.normalize_text(text))
        signature = update_detection.minhash_signature(doc_shingles)
        bands = update_detection.lsh_bands(signature)
        candidates = repo.find_update_shortlist(ws, bands=bands, title=title)

        best_ratio = 0.0
        best_candidate_id: str | None = None
        for candidate in candidates:
            if candidate["documentId"] == document_id:
                continue  # never a candidate for itself
            candidate_shingles = update_detection.shingles(
                update_detection.normalize_text(candidate.get("text") or "")
            )
            ratio = update_detection.jaccard(doc_shingles, candidate_shingles)
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate_id = candidate["documentId"]

        if best_candidate_id is None or best_ratio < _UPDATE_DETECTION_NOISE_FLOOR:
            return

        repo.create_or_reopen_supersede_suggestion(
            ws, new_document_id=document_id, candidate_document_id=best_candidate_id,
            match_id=_default_id(), status="pending", confidence=best_ratio,
            technique="shingled_jaccard_overlap", created_at=_default_clock(),
        )
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception("background update-detection failed (documentId=%s)", document_id)


def _schedule_update_detection(
    schedule: Callable[..., None], repo: Any, ws: str, document_id: str,
) -> None:
    """Schedule the suggested-tier update-detection job for one just-ingested
    document (document-ingestion2 Stage D, plan §3.4) — a **separate**
    scheduling call from `_schedule_chunk_processing`, since detection is
    per-*document*, not per-chunk, and must never touch `Document.status`/
    `pendingJobs` (that counter belongs solely to the embed/extract jobs).
    `repo` is the same `Repository` instance already threaded through
    `app.py` — independently optional, same posture as `embed_worker`/
    `ingestion_pipeline`: a deployment with no repository reference wired
    for this purpose (should not happen in practice, since `repo` always
    exists once the app is built, but mirrors the other schedulers'
    defensive `is None` guard for symmetry and testability) schedules
    nothing.
    """
    if repo is None:
        return
    schedule(_safe_detect_update, repo, ws, document_id)


def _safe_fuse(
    ingestion_pipeline: Any, ws: str, entity_id: str, name: str, type: str
) -> None:
    """Fuse one newly-created entity out-of-band, swallowing+logging any
    failure (K-050 M5 Stage 4 — FR-9 suggested-tier isolation).

    Unlike every other `_safe_*` wrapper in this module, this one is called
    **inline, from inside `IngestionPipeline.extract_chunk`'s own per-entity
    loop** (`ingestion.py`), not scheduled as a separate background task by
    `api.py`/`mcp.py` — the fuzzy-tier lookup/write for one entity can only
    run once that entity exists (`create_entity_with_auto_match` already
    ran), and `api.py`/`mcp.py` schedule per-CHUNK, before any entity is
    known. Granularity here is per-ENTITY, one level finer than
    `_safe_extract`'s per-chunk isolation: a fusion failure for one entity
    (a RediSearch syntax error, a transient connection hiccup) must not
    corrupt the `Document`, and — same failure-isolation discipline as every
    other wrapper here — must not block sibling entities or that chunk's
    relationship writes, which `extract_chunk`'s loop continues past this
    call regardless of outcome.
    """
    try:
        ingestion_pipeline.fuse_entity(ws, entity_id, name, type)
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception("background fuse failed (entityId=%s)", entity_id)


def _safe_respond(responder: Any, ctx: CallContext, posted: dict[str, Any]) -> None:
    """Fire the AI responder out-of-band, swallowing+logging any failure.

    Runs off-band, off the guarded write path. The responder owns the
    trigger policy (@mention + non-agent-authored) and self-no-ops
    otherwise, so each transport stays a thin adapter and never
    re-implements the trigger check. `channel_id` is not carried by the
    message-post path; channel-scoped retrieval in the wired path is a K-014
    follow-up (needs a thread→channel read).
    """
    try:
        responder.maybe_respond(
            ctx,
            thread_id=posted["threadId"],
            msg_id=posted["msgId"],
            text=posted["text"],
            role=posted["role"],
            channel_id=None,
            mentions=posted.get("mentions", []),
        )
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception("background responder failed (msgId=%s)", posted.get("msgId"))


def _safe_run_workflow(trigger: Any, ctx: CallContext, posted: dict[str, Any]) -> None:
    """Fire the workflow trigger out-of-band, swallowing+logging any failure.

    Runs off-band, off the guarded write path, mirroring `_safe_respond`. The
    trigger applies the §6 ordered rule (resume/start/responder fall-through)
    and **owns** the responder, so this is the *single* handler when a
    trigger is wired — exactly one of trigger-vs-responder is scheduled per
    posted message (the M3 one-handler guarantee). Its latency never blocks
    the poster.
    """
    try:
        trigger.maybe_trigger(
            ctx,
            thread_id=posted["threadId"],
            msg_id=posted["msgId"],
            text=posted["text"],
            role=posted["role"],
            mentions=posted.get("mentions", []),
        )
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception(
            "background workflow trigger failed (msgId=%s)", posted.get("msgId")
        )
