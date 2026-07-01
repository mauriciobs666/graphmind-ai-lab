"""REST transport — the browser front door (DESIGN §14.4).

A thin router over the shared `Services`. `get_context` is a FastAPI dependency
(the §14.3 seam) so tests can override the tenant. Business logic stays in
`services.py`; this layer only maps HTTP <-> service calls and errors <-> status.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from .config import CallContext, get_context as _resolve_context
from .schemas import CreateChannelIn, CreateThreadIn, PostMessageIn
from .services import Services


def get_context() -> CallContext:
    """The auth/tenancy seam as a FastAPI dependency (overridable in tests)."""
    return _resolve_context()


def build_router(services: Services) -> APIRouter:
    router = APIRouter()

    @router.post("/channels", status_code=201)
    def create_channel(body: CreateChannelIn, ctx: CallContext = Depends(get_context)):
        return services.create_channel(ctx, name=body.name)

    @router.get("/channels")
    def list_channels(ctx: CallContext = Depends(get_context)):
        return services.list_channels(ctx)

    @router.post("/channels/{channel_id}/threads", status_code=201)
    def create_thread(
        channel_id: str, body: CreateThreadIn,
        ctx: CallContext = Depends(get_context),
    ):
        return services.create_thread(ctx, channel_id=channel_id, title=body.title)

    @router.get("/channels/{channel_id}/threads")
    def list_threads(channel_id: str, ctx: CallContext = Depends(get_context)):
        return services.list_threads(ctx, channel_id=channel_id)

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
    def read_thread(thread_id: str, ctx: CallContext = Depends(get_context)):
        return services.read_thread(ctx, thread_id=thread_id)

    @router.get("/threads/{thread_id}/messages/{msg_id}")
    def get_message(
        thread_id: str, msg_id: str, ctx: CallContext = Depends(get_context)
    ):
        msg = services.get_message(ctx, msg_id=msg_id)
        if msg is None:
            raise HTTPException(status_code=404, detail="message not found")
        return msg

    return router
