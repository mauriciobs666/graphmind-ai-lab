"""REST transport — the browser front door (DESIGN §14.4).

A thin router over the shared `Services`. `get_context` is a FastAPI dependency
(the §14.3 seam) so tests can override the tenant. Business logic stays in
`services.py`; this layer only maps HTTP <-> service calls and errors <-> status.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from .config import CallContext
from .config import get_context as _resolve_context
from .schemas import CreateChannelIn, CreateThreadIn, PostMessageIn
from .services import Services


def get_context() -> CallContext:
    """The auth/tenancy seam as a FastAPI dependency (overridable in tests)."""
    return _resolve_context()


def build_router(services: Services) -> APIRouter:
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
        thread_id: str, body: PostMessageIn,
        ctx: CallContext = Depends(get_context),
    ):
        return services.post_message(
            ctx, thread_id=thread_id, text=body.text, mentions=body.mentions
        )

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

    return router
