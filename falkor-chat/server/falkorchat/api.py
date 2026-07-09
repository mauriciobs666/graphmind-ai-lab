"""REST transport — the browser front door (DESIGN §14.4).

A thin router over the shared `Services`. `get_context` is a FastAPI dependency
(the §14.3 seam) so tests can override the tenant. Business logic stays in
`services.py`; this layer only maps HTTP <-> service calls and errors <-> status.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from .config import CallContext
from .config import get_context as _resolve_context
from .schemas import (
    CreateChannelIn,
    CreateThreadIn,
    PostMessageIn,
    PublishWorkflowDefIn,
)
from .services import Services

_log = logging.getLogger(__name__)


def get_context() -> CallContext:
    """The auth/tenancy seam as a FastAPI dependency (overridable in tests)."""
    return _resolve_context()


def _safe_embed(embed_worker: Any, ws: str, msg_id: str, text: str) -> None:
    """Embed a posted message out-of-band, swallowing+logging any failure.

    Runs on `BackgroundTasks` (DECISION 3, K-008 posture): EVERY posted message
    is embedded so the retrievable corpus grows, but a message is readable before
    its embedding lands and an embedder hiccup must never surface to the poster.
    Embedding is a pure write — it never triggers a response.
    """
    try:
        embed_worker.embed_message(ws, msg_id=msg_id, text=text)
    except Exception:  # noqa: BLE001 — background isolation: log, never propagate
        _log.exception("background embed failed (msgId=%s)", msg_id)


def _safe_respond(responder: Any, ctx: CallContext, posted: dict[str, Any]) -> None:
    """Fire the AI responder out-of-band, swallowing+logging any failure.

    Runs on `BackgroundTasks`, off the guarded write path. The responder owns the
    trigger policy (@mention + non-agent-authored) and self-no-ops otherwise, so
    the API stays a thin adapter and never re-implements the trigger check.
    `channel_id` is not carried by the message-post route; channel-scoped
    retrieval in the wired path is a K-014 follow-up (needs a thread→channel read).
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


def build_router(
    services: Services, *, responder: Any | None = None,
    embed_worker: Any | None = None,
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
        if responder is not None:
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

    # ── §11 Workflow definitions & snapshots (M3 Slice 1) ────────────────────
    # Def authoring/reading is GLOBAL (the `reference` graph); only materialize +
    # snapshot list consume the tenant workspace via `get_context`. Spec/​not-found
    # errors map to 400/404 through the app-level exception handlers.

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

    @router.post("/workflow-defs/{key}/versions/{version}/materialize", status_code=201)
    def materialize_def(
        key: str, version: str, ctx: CallContext = Depends(get_context)
    ):
        return services.materialize_def(ctx, key=key, version=version)

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

    return router
