"""REST transport — the browser front door (DESIGN §14.4).

A thin router over the shared `Services`. `get_context` is a FastAPI dependency
(the §14.3 seam) so tests can override the tenant. Business logic stays in
`services.py`; this layer only maps HTTP <-> service calls and errors <-> status.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query

from .background import (
    _safe_embed,
    _safe_respond,
    _safe_run_workflow,
    _schedule_chunk_processing,
)
from .config import CallContext
from .config import get_context as _resolve_context
from .schemas import (
    MAX_ID_LEN,
    MAX_KEY_LEN,
    CreateChannelIn,
    CreateThreadIn,
    IngestDocumentIn,
    IngestDocumentsIn,
    PostMessageIn,
    PublishWorkflowDefIn,
    StartWorkflowRunIn,
    SubmitWorkflowInputIn,
    SweepDueWorkflowRunsIn,
    WorkflowDefStructureOut,
    WorkflowDiffOut,
)
from .services import Services

_log = logging.getLogger(__name__)


def get_context() -> CallContext:
    """The auth/tenancy seam as a FastAPI dependency (overridable in tests)."""
    return _resolve_context()


def build_router(
    services: Services, *, responder: Any | None = None,
    embed_worker: Any | None = None, trigger: Any | None = None,
    ingestion_pipeline: Any | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health(ctx: CallContext = Depends(get_context)):
        # Liveness for probes/CI: proves the process is up AND FalkorDB answers.
        try:
            ok = services.ping(ctx)
        except Exception:
            ok = False
        if not ok:
            raise HTTPException(status_code=503, detail="FalkorDB unreachable")
        return {"status": "ok"}

    @router.post("/channels", status_code=201)
    def create_channel(body: CreateChannelIn, ctx: CallContext = Depends(get_context)):
        return services.create_channel(ctx, name=body.name)

    @router.get("/channels")
    def list_channels(
        limit: int = Query(50, ge=1, le=200),
        ctx: CallContext = Depends(get_context),
    ):
        return services.list_channels(ctx, limit=limit)

    @router.post("/channels/{channel_id}/threads", status_code=201)
    def create_thread(
        channel_id: str, body: CreateThreadIn,
        ctx: CallContext = Depends(get_context),
    ):
        return services.create_thread(ctx, channel_id=channel_id, title=body.title)

    @router.get("/channels/{channel_id}/threads")
    def list_threads(
        channel_id: str,
        limit: int = Query(50, ge=1, le=200),
        ctx: CallContext = Depends(get_context),
    ):
        return services.list_threads(ctx, channel_id=channel_id, limit=limit)

    @router.post("/threads/{thread_id}/messages", status_code=201)
    def post_message(
        thread_id: str, body: PostMessageIn, background: BackgroundTasks,
        ctx: CallContext = Depends(get_context),
    ):
        posted = services.post_message(
            ctx, thread_id=thread_id, text=body.text, mentions=body.mentions
        )
        # Out-of-band, off the guarded write path (failure-isolated):
        #  * embed EVERY posted message (DECISION 3) so it joins the corpus;
        #  * fire the responder, which self-decides whether to answer.
        # Both are scheduled AFTER the write returns — the message is readable
        # first (DESIGN §9). Embedding and triggering are separate paths.
        if embed_worker is not None:
            background.add_task(
                _safe_embed, embed_worker, ctx.ws, posted["msgId"], posted["text"]
            )
        # Exactly ONE of {trigger, responder} handles the message (M3 one-handler
        # guarantee). When a trigger is wired it owns the responder fall-through
        # (§6), so `_safe_respond` is NOT also scheduled — an @mention can never
        # fire both a workflow and a direct reply.
        if trigger is not None:
            background.add_task(_safe_run_workflow, trigger, ctx, posted)
        elif responder is not None:
            background.add_task(_safe_respond, responder, ctx, posted)
        return posted

    @router.get("/search")
    def search_messages(
        q: str = Query(..., min_length=1),
        limit: int = Query(50, ge=1, le=200),
        ctx: CallContext = Depends(get_context),
    ):
        return services.search_messages(ctx, query=q, limit=limit)

    @router.get("/threads/{thread_id}/messages")
    def read_thread(
        thread_id: str,
        since: int | None = Query(None, ge=0),
        limit: int | None = Query(None, ge=1, le=1000),
        ctx: CallContext = Depends(get_context),
    ):
        # No params → the full §4 thread read (M1 web client contract). Either
        # param → the paginated §9.1 since-read as a pure read: `since`
        # defaults to epoch 0 explicitly so the member's cursor is never
        # consulted or advanced by a browser poll (cursors stay agent-owned).
        # Explicit `since` is plain `>` (OQ3): it may re-deliver or skip rows
        # within that exact millisecond — lossless catch-up is the cursor
        # path's job (MCP `read_messages` without `since`).
        if since is None and limit is None:
            return services.read_thread(ctx, thread_id=thread_id)
        return services.read_messages(
            ctx, thread_id=thread_id, since=since or 0, limit=limit or 50
        )

    @router.get("/messages/{msg_id}")
    def get_message(msg_id: str, ctx: CallContext = Depends(get_context)):
        # msgId is workspace-unique; resolution is workspace-global by design,
        # so the route stays flat (spec §5 / fork 4) — the body carries
        # `threadId` (denormalized, K-007) for navigation.
        msg = services.get_message(ctx, msg_id=msg_id)
        if msg is None:
            raise HTTPException(status_code=404, detail="message not found")
        return msg

    # ── §14 Documents & Chunks (K-050 M5 Stages 1-2) ──────────────────────────
    # `DocumentTooLargeError` maps to 400 via the generic `ServiceError`
    # handler; `UnknownActorError` (an unresolvable `get_context()` actor) also
    # maps to 400 the same way. `SearchNotAvailableError` (no `ModelGateway`
    # wired) maps to 503 (`app._register_error_handlers`).

    @router.post("/documents", status_code=201)
    def ingest_document(
        body: IngestDocumentIn, background: BackgroundTasks,
        ctx: CallContext = Depends(get_context),
    ):
        receipt = services.ingest_document(
            ctx, text=body.text, title=body.title,
            source_format=body.sourceFormat, source_label=body.sourceLabel,
        )
        # Out-of-band, off the guarded write path (K-050 M5 Stage 2, mirrors
        # the message embed scheduling above): every chunk of a just-ingested
        # document is embedded so it joins the standalone-KB-searchable corpus.
        # The document is readable before any chunk's embedding lands.
        if embed_worker is not None or ingestion_pipeline is not None:
            chunks = services.list_document_chunks(
                ctx, document_id=receipt["documentId"]
            )
            _schedule_chunk_processing(
                background.add_task, ctx.ws, receipt["documentId"], chunks,
                embed_worker=embed_worker, ingestion_pipeline=ingestion_pipeline,
            )
        return receipt

    # K-050 M5 Stage 6a (FR-11): bulk variant of the route above — loops the
    # same `Services.ingest_document` path per item (`Services.ingest_documents`
    # docstring), one receipt per item, an item's failure isolated to its own
    # receipt rather than aborting the batch. Registered BEFORE
    # `/documents/{document_id}` for the same static-before-dynamic reason as
    # `/documents/search` below — `/documents/batch` would otherwise be
    # swallowed by `{document_id}` treating "batch" as an id.
    @router.post("/documents/batch", status_code=201)
    def ingest_documents(
        body: IngestDocumentsIn, background: BackgroundTasks,
        ctx: CallContext = Depends(get_context),
    ):
        receipts = services.ingest_documents(
            ctx,
            documents=[
                {
                    "text": item.text, "title": item.title,
                    "source_format": item.sourceFormat,
                    "source_label": item.sourceLabel,
                }
                for item in body.documents
            ],
        )
        if embed_worker is not None or ingestion_pipeline is not None:
            for receipt in receipts:
                if receipt.get("status") != "processing":
                    continue  # this item errored — nothing to schedule for it
                chunks = services.list_document_chunks(
                    ctx, document_id=receipt["documentId"]
                )
                _schedule_chunk_processing(
                    background.add_task, ctx.ws, receipt["documentId"], chunks,
                    embed_worker=embed_worker, ingestion_pipeline=ingestion_pipeline,
                )
        return receipts

    # Registered BEFORE `/documents/{document_id}`: Starlette matches routes in
    # registration order, and `{document_id}` would otherwise swallow a literal
    # `/documents/search` path (treating "search" as the id) — the same static-
    # before-dynamic ordering this router already relies on elsewhere.
    @router.get("/documents/search")
    def search_documents(
        q: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=200),
        ctx: CallContext = Depends(get_context),
    ):
        return services.search_documents(ctx, query=q, limit=limit)

    @router.get("/documents/{document_id}")
    def get_document(document_id: str, ctx: CallContext = Depends(get_context)):
        doc = services.get_document(ctx, document_id=document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        return doc

    # ── §14.6 Entity fusion review surface (K-050 M5 Stage 4, FR-10/OQ-2) ─────
    # `MatchNotFoundError` maps to 404 via the generic `ServiceError` handler.
    # Registered BEFORE `/matches/{match_id}/...`: no ambiguity here (no
    # `/matches/search`-shaped route exists), but kept adjacent to `/matches`
    # reads for readability, same static-then-scoped grouping as `/documents`.

    @router.get("/matches/pending")
    def list_pending_matches(
        limit: int = Query(50, ge=1, le=200),
        ctx: CallContext = Depends(get_context),
    ):
        return services.list_pending_matches(ctx, limit=limit)

    @router.get("/matches")
    def list_matches(
        status: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
        ctx: CallContext = Depends(get_context),
    ):
        return services.list_matches(ctx, status=status, limit=limit)

    @router.post("/matches/{match_id}/confirm")
    def confirm_match(
        match_id: str = Path(..., min_length=1, max_length=MAX_ID_LEN),
        ctx: CallContext = Depends(get_context),
    ):
        return services.confirm_match(ctx, match_id=match_id)

    @router.post("/matches/{match_id}/reject")
    def reject_match(
        match_id: str = Path(..., min_length=1, max_length=MAX_ID_LEN),
        ctx: CallContext = Depends(get_context),
    ):
        return services.reject_match(ctx, match_id=match_id)

    @router.post("/matches/{match_id}/recheck")
    def recheck_match(
        match_id: str = Path(..., min_length=1, max_length=MAX_ID_LEN),
        ctx: CallContext = Depends(get_context),
    ):
        return services.recheck_match(ctx, match_id=match_id)

    # ── K-036 web-api-coverage: thread-scoped reads (FR-2/FR-8, Wave 2) ──────
    # New read paths for the web UI's inline run cue + participants list; both
    # wrap GRAPH.PROFILE-verified queries (QUERIES.md §12.14, §2 "List thread
    # participants") — see docs/plans/web-api-coverage.md §3.1a/§3.1b. Neither
    # declares a `response_model` (matches the surface's convention — see the
    # note at the K-031 structure routes below, the one place that does).
    # `ThreadNotFoundError` maps to 404 via the generic `ServiceError` handler.

    @router.get("/threads/{thread_id}/workflow-runs")
    def list_workflow_runs_for_thread(
        thread_id: str,
        limit: int = Query(10, ge=1, le=50),
        ctx: CallContext = Depends(get_context),
    ):
        return services.list_workflow_runs_for_thread(
            ctx, thread_id=thread_id, limit=limit
        )

    @router.get("/threads/{thread_id}/participants")
    def list_thread_participants(
        thread_id: str, ctx: CallContext = Depends(get_context)
    ):
        return services.list_thread_participants(ctx, thread_id=thread_id)

    # ── §11 Workflow definitions & snapshots (M3 Slice 1) ────────────────────
    # Def authoring/reading is GLOBAL (the `reference` graph); only materialize +
    # snapshot list consume the tenant workspace via `get_context`. Spec/​not-found
    # errors map to 400/404 through the app-level exception handlers; a topology-
    # differing re-publish/re-materialize maps to 409 (`WorkflowDefConflictError`,
    # K-034).

    @router.post("/workflow-defs", status_code=201)
    def publish_workflow_def(
        body: PublishWorkflowDefIn, ctx: CallContext = Depends(get_context)
    ):
        return services.publish_workflow_def(
            ctx,
            key=body.key, version=body.version, name=body.name, kind=body.kind,
            steps=[s.model_dump() for s in body.steps],
            transitions=[t.model_dump(by_alias=True) for t in body.transitions],
        )

    @router.get("/workflow-defs")
    def list_workflow_defs(
        limit: int = Query(50, ge=1, le=200),
        ctx: CallContext = Depends(get_context),
    ):
        return services.list_workflow_defs(ctx, limit=limit)

    @router.get("/workflow-defs/{key}")
    def get_workflow_def(
        key: str,
        version: str | None = Query(None),
        ctx: CallContext = Depends(get_context),
    ):
        got = services.get_workflow_def(ctx, key=key, version=version)
        if got is None:
            raise HTTPException(status_code=404, detail="workflow def not found")
        return got

    # ── §11 def/snapshot STRUCTURE reads + diff (K-031 observability) ─────────
    # The version-qualified def is its own resource, so it gets its own path
    # rather than an `?expand=` on `GET /workflow-defs/{key}` — that route
    # resolves **latest** when no version is given, and structure-reading a
    # version the caller never named is the worst possible affordance in a tool
    # whose purpose is confirming *which* version is live. There is deliberately
    # no `latest` alias here for the same reason.
    #
    # These three routes are the only ones in the surface that declare a
    # `response_model` — see the `schemas.py` module docstring for why the rest
    # are not retrofitted.

    @router.get(
        "/workflow-defs/{key}/versions/{version}",
        response_model=WorkflowDefStructureOut,
        # Keeps `startKeys` **absent** (not `null`) in the ordinary single-START
        # case, without `exclude_none` also swallowing a null `startKey`.
        response_model_exclude_unset=True,
    )
    def get_workflow_def_structure(
        key: str = Path(..., min_length=1, max_length=MAX_KEY_LEN),
        version: str = Path(..., min_length=1, max_length=MAX_KEY_LEN),
        ctx: CallContext = Depends(get_context),
    ):
        return services.get_workflow_def_structure(ctx, key=key, version=version)

    @router.post("/workflow-defs/{key}/versions/{version}/materialize", status_code=201)
    def materialize_def(
        key: str, version: str, ctx: CallContext = Depends(get_context)
    ):
        return services.materialize_def(ctx, key=key, version=version)

    # ── §12 Workflow-run start + human/signal input (K-024 U3) ────────────────
    # The non-chat front door for a `kind:'process'` run: start it from a snapshot
    # with no trigger `Message` (§12.12), then advance each parked `human`/`wait`
    # step with structured input (§12.13). Driving is **synchronous** — a process
    # drive is pure graph work with no LLM, so it is fast and deterministically
    # testable; D-G is what makes that safe at the HTTP boundary.
    #
    # Error map (all app-level handlers): 404 unknown run · 409 not parked / lost
    # the resume CAS · 400 rejected input or malformed guard · 503 engine unwired.
    # A fault *during* the drive is NOT an error status: the run is already
    # correctly terminal in the graph, so it comes back 201/200 carrying
    # `{"status": "failed", "error": …}`.

    @router.post("/workflow-runs", status_code=201)
    def start_workflow_run(
        body: StartWorkflowRunIn, ctx: CallContext = Depends(get_context)
    ):
        return services.start_workflow_run(
            ctx, def_key=body.defKey, version=body.version, run_ctx=body.ctx,
            trace=body.trace, max_steps=body.maxSteps,
        )

    @router.post("/workflow-runs/{run_id}/input")
    def submit_workflow_input(
        body: SubmitWorkflowInputIn,
        run_id: str = Path(..., min_length=1, max_length=MAX_ID_LEN),
        ctx: CallContext = Depends(get_context),
    ):
        return services.submit_workflow_input(ctx, run_id=run_id, input=body.input)

    @router.post("/workflow-runs/due")
    def sweep_due_workflow_runs(
        body: SweepDueWorkflowRunsIn, ctx: CallContext = Depends(get_context)
    ):
        return services.sweep_due_workflow_runs(ctx, limit=body.limit)

    # ── §12 Workflow-run inspection (U12 — AC-5 observability seam) ───────────
    # Thin, size-bounded RO pass-throughs so QA can read a run / its step-runs /
    # its trace black-box (no reaching into Cypher). A missing run → 404 (step-runs
    # and trace return [] for an unknown run, matching the RO query semantics).

    @router.get("/workflow-runs/{run_id}")
    def get_workflow_run(
        run_id: str = Path(..., min_length=1, max_length=MAX_ID_LEN),
        ctx: CallContext = Depends(get_context),
    ):
        run = services.get_workflow_run(ctx, run_id=run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="workflow run not found")
        return run

    @router.get("/workflow-runs/{run_id}/step-runs")
    def read_workflow_step_runs(
        run_id: str = Path(..., min_length=1, max_length=MAX_ID_LEN),
        ctx: CallContext = Depends(get_context),
    ):
        return services.read_workflow_step_runs(ctx, run_id=run_id)

    @router.get("/workflow-runs/{run_id}/trace")
    def read_workflow_trace(
        run_id: str = Path(..., min_length=1, max_length=MAX_ID_LEN),
        ctx: CallContext = Depends(get_context),
    ):
        return services.read_workflow_trace(ctx, run_id=run_id)

    @router.get("/workspaces/{ws}/snapshots")
    def list_snapshots(
        ws: str,
        limit: int = Query(50, ge=1, le=200),
        ctx: CallContext = Depends(get_context),
    ):
        # The path `ws` is descriptive; tenancy is resolved by `get_context`
        # (the M1 single-tenant seam), mirroring how MCP ignores a client-supplied
        # `from`. When real multi-tenant auth lands, the seam authorizes the path.
        return services.list_snapshots(ctx, limit=limit)

    @router.get(
        "/workspaces/{ws}/snapshots/{key}/versions/{version}",
        response_model=WorkflowDefStructureOut,
        response_model_exclude_unset=True,
    )
    def get_snapshot_structure(
        ws: str,
        key: str = Path(..., min_length=1, max_length=MAX_KEY_LEN),
        version: str = Path(..., min_length=1, max_length=MAX_KEY_LEN),
        ctx: CallContext = Depends(get_context),
    ):
        # `ws` is descriptive here too — tenancy comes from `get_context`.
        # Same response shape as the def route apart from `source`, so a human
        # or `jq` can diff the two bodies directly.
        got = services.get_snapshot_structure(ctx, key=key, version=version)
        if got is None:
            raise HTTPException(status_code=404, detail="workflow snapshot not found")
        return got

    @router.get(
        "/workspaces/{ws}/snapshots/{key}/versions/{version}/diff",
        response_model=WorkflowDiffOut,
    )
    def diff_def_snapshot(
        ws: str,
        key: str = Path(..., min_length=1, max_length=MAX_KEY_LEN),
        version: str = Path(..., min_length=1, max_length=MAX_KEY_LEN),
        ctx: CallContext = Depends(get_context),
    ):
        # Lives under `/workspaces/…` because the comparison is inherently
        # workspace-scoped (the snapshot side is the workspace's).
        #
        # ⚠️ **Version-qualified**: this compares *content within one named
        # version*. It cannot detect "`reference` now has v2, the workspace only
        # ever materialized v1" — the shape a `key`/`version` bump produces. To
        # detect a stale *version*, compare `GET /workflow-defs` against
        # `GET /workspaces/{ws}/snapshots` first, or run
        # `./scripts/verify_workflows.sh <wsId>` which checks both defs at once.
        return services.diff_def_snapshot(ctx, key=key, version=version)

    # ── FR-10 workspace readiness (web-api-coverage plan §3.1c / U2) ──────────
    # The HTTP form of `scripts/verify_workflows.sh` — same expected defs, same
    # checks, reachable without a shell + venv. Always 200: readiness is a
    # *report*, never a 404/error condition, so a caller (or the web UI) never
    # has to special-case a not-ready workspace as a failed request.
    @router.get("/workspaces/{ws}/readiness")
    def check_demo_readiness(ws: str, ctx: CallContext = Depends(get_context)):
        # `ws` is descriptive here too — tenancy comes from `get_context`, same
        # convention as `list_snapshots`/`diff_def_snapshot` above.
        return services.check_demo_readiness(ctx)

    return router
