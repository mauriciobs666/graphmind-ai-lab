"""MCP (Streamable-HTTP) transport — a peer of `api.py` (plan §3, §6).

A thin adapter that translates MCP tool calls into the same `Services` methods the
REST router calls. No business logic lives here. The `Services` instance and the
actor/context provider are injected via `configure(...)` at app-build time (and in
tests), so this module never hardcodes the tenant.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import config
from .background import (
    _safe_embed,
    _safe_embed_chunk,
    _safe_respond,
    _safe_run_workflow,
)
from .config import CallContext
from .services import Services

mcp = FastMCP("falkor-chat")
# Serve the Streamable-HTTP handler at the mount root so that mounting the app
# under "/mcp" in app.py yields a clean "/mcp" endpoint (Appendix A contract),
# not "/mcp/mcp".
mcp.settings.streamable_http_path = "/"

# Injected at app-build time via `configure`. Kept module-level because FastMCP
# tools are registered at import; the injection swaps the backing service/context
# without re-registering tools.
_services: Services | None = None
_get_context: Callable[[], CallContext] = config.get_context
# K-041: the same opt-in out-of-band handlers `api.build_router` accepts
# (`responder`/`embed_worker`/`trigger`), mirrored here so `send_message` can
# schedule the identical background policy a REST post gets. All default to
# `None` so an unconfigured/test-default MCP tool set keeps posting inert.
_responder: Any | None = None
_embed_worker: Any | None = None
_trigger: Any | None = None


def _default_schedule(fn: Callable[..., None], *args: Any) -> None:
    """Fire-and-forget a background call on an unbounded daemon thread.

    A plain `@mcp.tool()` function has no per-call object like FastAPI's
    `BackgroundTasks` to schedule work on. `fn` is always one of
    `background._safe_*`, which is already synchronous and failure-isolated
    (try/except + log, never raises) — a daemon thread runs it off-band
    without blocking the tool call's return. This is **not** the same
    throttling posture as the REST path: Starlette's `BackgroundTasks` runs a
    sync callable via `anyio.to_thread.run_sync`, which is bounded by a
    default capacity limiter (~40 concurrent worker threads); this spins one
    new OS thread per call with no ceiling. A burst of concurrent
    `send_message` calls therefore spawns unboundedly, unlike REST's
    throttled threadpool — accepted for M1's lab-scale, unauthenticated,
    single-tenant posture (`docs/DESIGN.md` §15.3), not because the two are
    equivalent (`docs/reviews/mcp-background-scheduling-impl.md` Minor 1).
    Overridable as `mcp._schedule` — tests swap it for an inline call to
    assert deterministically instead of racing a background thread.

    **No shutdown/drain awareness.** Unlike REST, where Starlette awaits
    `BackgroundTasks` as part of the same ASGI request coroutine (so the
    request isn't "done" until the background task completes), a thread
    spawned here is untracked by `app.py`'s `_lifespan` and is `daemon=True`
    — killed outright, silently, with no log line, if the process exits
    before it finishes. A chosen trade-off for M1's scale, not an oversight
    (`docs/reviews/mcp-background-scheduling-impl.md` Minor 2); closing it
    would mean tracking spawned threads in a small registry and joining them
    with a timeout in `_lifespan`'s shutdown branch.
    """
    threading.Thread(target=fn, args=args, daemon=True).start()


_schedule: Callable[..., None] = _default_schedule


def configure(
    services: Services,
    *,
    context_provider: Callable[[], CallContext] | None = None,
    responder: Any | None = None,
    embed_worker: Any | None = None,
    trigger: Any | None = None,
) -> FastMCP:
    """Wire the MCP tools to a `Services` instance (and optional context seam).

    `responder`/`embed_worker`/`trigger` mirror `api.build_router`'s same-named
    parameters (K-041) — pass the exact same objects `create_app` wires into
    the REST router so both transports run the identical post-message policy.
    """
    global _services, _get_context, _responder, _embed_worker, _trigger
    _services = services
    if context_provider is not None:
        _get_context = context_provider
    _responder = responder
    _embed_worker = embed_worker
    _trigger = trigger
    return mcp


def _svc() -> Services:
    if _services is None:  # pragma: no cover - guards against unconfigured use
        raise RuntimeError("MCP tools used before configure() was called")
    return _services


def _schedule_background(ctx: CallContext, posted: dict[str, Any]) -> None:
    """Schedule the same out-of-band work `api.py`'s `post_message` route does.

    Mirrors that route's ordering exactly (K-041): embed always-if-configured;
    then exactly ONE of {trigger, responder} — the M3 one-handler guarantee —
    so an @mention posted via MCP can never fire both a workflow and a direct
    reply, same as an @mention posted via REST.
    """
    if _embed_worker is not None:
        _schedule(_safe_embed, _embed_worker, ctx.ws, posted["msgId"], posted["text"])
    if _trigger is not None:
        _schedule(_safe_run_workflow, _trigger, ctx, posted)
    elif _responder is not None:
        _schedule(_safe_respond, _responder, ctx, posted)


@mcp.tool()
def send_message(
    body: str, re: str, mentions: list[str] | None = None, frm: str | None = None
) -> dict[str, Any]:
    """Post `body` into thread `re`, optionally mentioning members.

    `frm` is reserved/ignored in M1 — the author is the configured actor (Q#1).
    Schedules the same out-of-band embed/trigger/responder work the REST route
    does (K-041), off-band so the write itself is never delayed.
    """
    ctx = _get_context()
    posted = _svc().post_message(ctx, thread_id=re, text=body, mentions=mentions)
    _schedule_background(ctx, posted)
    return posted


@mcp.tool()
def read_messages(
    re: str | None = None,
    since: int | None = None,
    limit: int = 50,
    advance: bool = True,
) -> list[dict[str, Any]]:
    """Catch up on messages.

    With `re` (thread id): read that thread since your cursor (or explicit
    `since`), advancing the cursor unless `since` is given. Cursor-driven reads
    are tie-safe — the composite `(lastReadAt, lastReadMsgId)` cursor never
    skips or re-delivers, even across same-millisecond messages. An explicit
    `since` is a plain `>` timestamp read and may re-deliver or skip within
    that exact millisecond. Without `re`: workspace-wide read since `since`
    (default epoch 0); no cursor is advanced. Rows carry `threadId`.
    """
    ctx = _get_context()
    return _svc().read_messages(
        ctx, thread_id=re, since=since, limit=limit, advance=advance
    )


@mcp.tool()
def create_thread(channel_id: str, title: str) -> dict[str, Any]:
    """Create a thread in an existing channel so an agent is self-sufficient."""
    ctx = _get_context()
    return _svc().create_thread(ctx, channel_id=channel_id, title=title)


@mcp.tool()
def search_messages(query: str, limit: int = 50) -> list[dict[str, Any]]:
    """Full-text keyword search over this workspace's messages (QUERIES.md §5).

    Returns matches newest-relevance first (RediSearch score). Invalid query
    syntax (unbalanced quotes, stray operators) surfaces as a tool error.
    """
    ctx = _get_context()
    return _svc().search_messages(ctx, query=query, limit=limit)


@mcp.tool()
def create_channel(name: str) -> dict[str, Any]:
    """Create a channel so an agent is self-sufficient (can set up its own space)."""
    ctx = _get_context()
    return _svc().create_channel(ctx, name=name)


@mcp.tool()
def list_channels(limit: int = 50) -> list[dict[str, Any]]:
    """List this workspace's channels, newest first.

    The navigation entry point: without it an agent cannot discover an existing
    channel to join — only create its own.
    """
    ctx = _get_context()
    return _svc().list_channels(ctx, limit=limit)


@mcp.tool()
def list_threads(channel_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """List a channel's threads, most recently active first.

    Pairs with `list_channels` so an agent can find the thread id `send_message`
    and `read_messages` need.
    """
    ctx = _get_context()
    return _svc().list_threads(ctx, channel_id=channel_id, limit=limit)


@mcp.tool()
def ingest_document(
    text: str, title: str | None = None, source_format: str = "text",
    source_label: str | None = None,
) -> dict[str, Any]:
    """Ingest a text document: split into chunks, retained verbatim (K-050).

    Attributed to the configured `get_context()` actor (FR-4) — same posture
    as `send_message`, MCP ignores any notion of a client-supplied author.
    Chunks are embedded out-of-band right after this call returns (K-050 M5
    Stage 2) — readable and full-text-round-trippable via `get_document`
    immediately, ranked-searchable via `search_documents` once the background
    embed lands (same eventually-consistent posture as a posted message).
    Extraction/fusion are later stages; this call only returns
    `{documentId, chunkCount, status: 'processing'}`.
    """
    ctx = _get_context()
    receipt = _svc().ingest_document(
        ctx, text=text, title=title, source_format=source_format,
        source_label=source_label,
    )
    if _embed_worker is not None:
        for chunk in _svc().list_document_chunks(ctx, document_id=receipt["documentId"]):
            _schedule(_safe_embed_chunk, _embed_worker, ctx.ws, chunk["chunkId"], chunk["text"])
    return receipt


@mcp.tool()
def get_document(document_id: str) -> dict[str, Any] | None:
    """Fetch a document (verbatim `text`, plus its ingestion status/actor)."""
    ctx = _get_context()
    return _svc().get_document(ctx, document_id=document_id)


@mcp.tool()
def search_documents(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Rank ingested document chunks by similarity to `query` (K-050 M5 Stage 2,
    FR-3 standalone KB search — independent of chat/`search_messages`).

    Returns chunks ordered most-similar-first (`score` is cosine distance —
    lower is more similar), each carrying its source `documentId` so a caller
    can `get_document` the full text. Raises a tool error if no embedding
    model is configured for this deployment.
    """
    ctx = _get_context()
    return _svc().search_documents(ctx, query=query, limit=limit)
