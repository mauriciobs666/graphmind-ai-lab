"""MCP (Streamable-HTTP) transport — a peer of `api.py` (plan §3, §6).

A thin adapter that translates MCP tool calls into the same `Services` methods the
REST router calls. No business logic lives here. The `Services` instance and the
actor/context provider are injected via `configure(...)` at app-build time (and in
tests), so this module never hardcodes the tenant.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import config
from .background import (
    _safe_embed,
    _safe_respond,
    _safe_run_workflow,
    _schedule_chunk_processing,
)
from .config import CallContext
from .services import Services

_log = logging.getLogger(__name__)

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
_ingestion_pipeline: Any | None = None


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
    single-tenant posture (`docs/SERVER.md` §2.3), not because the two are
    equivalent (`docs/reviews/mcp-background-scheduling-impl.md` Minor 1).
    Overridable as `mcp._schedule` — tests swap it for an inline call to
    assert deterministically instead of racing a background thread.

    **The fan-out is doubled as of K-050 M5 Stage 3** (`docs/reviews/
    document-ingestion-impl.md` Pass 3 MAJOR 2): `ingest_document` now
    schedules TWO independent per-chunk jobs (embed + extract, both via this
    function) instead of one, so a single call against a max-size document
    (`MAX_DOCUMENT_CHARS = 500_000`, ~1,000-char chunks) now spawns on the
    order of **~1,000-1,200 raw OS threads**, sequentially and synchronously,
    inside the tool handler, before it returns — up from Pass 2's ~500-600.
    Still accepted for M1's lab-scale posture, not silently re-reasoning about
    half the real number; a bounded thread pool (batching each transport's
    fan-out instead of one-thread-per-(chunk × job)) is deferred to Stage 6,
    the same disposition Pass 2's original finding already had.

    **The fan-out compounds by up to `MAX_BATCH_SIZE` (20x) as of K-050 M5
    Stage 6a** (`docs/reviews/document-ingestion-impl.md` Pass 6 MAJOR):
    `mcp.ingest_documents` calls `background._schedule_chunk_processing`
    once per successfully-ingested item in a plain sequential loop over the
    batch, so at the plan's own stated bounds (`MAX_BATCH_SIZE = 20`,
    `MAX_DOCUMENT_CHARS = 500_000` ÷ ~850 effective chars/chunk ≈ 588
    chunks/document) a single `ingest_documents` MCP call can now spawn on
    the order of **~23,000 raw OS threads** (588 chunks × 2 jobs × 20
    documents), sequentially and synchronously, inside the one tool handler,
    before it returns — roughly 20x Stage 3's already-flagged ~1,000-1,200/
    call number above. Still accepted for M1's lab-scale posture (same
    reasoning as Stage 3's own doubling: REST is unaffected — cheap list
    appends, bounded execution via anyio's worker-pool limiter later — and
    every per-thread failure is already isolated by the try/except below),
    but explicitly NOT re-mitigated here: `MAX_BATCH_SIZE = 20` bounds the
    multiplier (it cannot grow further without either raising that cap or a
    caller issuing more `ingest_documents` calls, which is no worse than the
    pre-batch world of one `ingest_document` call per document) but does
    nothing to shrink the per-call number itself. A bounded thread pool
    (batching each transport's fan-out instead of one-thread-per-(chunk ×
    job) — the same fix Stage 3 deferred, now 20x more overdue) remains the
    real fix, deferred again pending a coordinator scope decision on whether
    it belongs in this feature or a standalone follow-up K-item.

    **Thread-creation failures are caught here, not propagated (Pass 3 MAJOR
    2 fix (a)).** If OS thread creation itself fails
    (`RuntimeError: can't start new thread` — a real failure mode once a
    process's thread count nears a `ulimit -u`/cgroup `pids.max` ceiling,
    commonly in the low thousands in constrained/containerized environments,
    now within a small constant factor of the doubled fan-out above), the
    failure is logged and swallowed rather than propagating out of the
    calling `@mcp.tool()` handler as an unhandled exception — one job's
    thread-start failure no longer aborts the rest of a per-chunk scheduling
    loop or turns an otherwise-successful `ingest_document` call into a
    caller-visible error (the underlying `Document`/`Chunk`s are already
    committed by the time this runs, so no data is lost either way — only the
    one scheduled background job is skipped).

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
    try:
        threading.Thread(target=fn, args=args, daemon=True).start()
    except Exception:  # noqa: BLE001 — see docstring: log, never propagate
        _log.exception(
            "mcp background schedule failed to start a thread (fn=%s)",
            getattr(fn, "__name__", fn),
        )


_schedule: Callable[..., None] = _default_schedule


def configure(
    services: Services,
    *,
    context_provider: Callable[[], CallContext] | None = None,
    responder: Any | None = None,
    embed_worker: Any | None = None,
    trigger: Any | None = None,
    ingestion_pipeline: Any | None = None,
) -> FastMCP:
    """Wire the MCP tools to a `Services` instance (and optional context seam).

    `responder`/`embed_worker`/`trigger`/`ingestion_pipeline` mirror
    `api.build_router`'s same-named parameters (K-041) — pass the exact same
    objects `create_app` wires into the REST router so both transports run
    the identical post-message policy.
    """
    global _services, _get_context, _responder, _embed_worker, _trigger
    global _ingestion_pipeline
    _services = services
    if context_provider is not None:
        _get_context = context_provider
    _responder = responder
    _embed_worker = embed_worker
    _trigger = trigger
    _ingestion_pipeline = ingestion_pipeline
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
    Entity/relationship extraction is scheduled independently, per chunk,
    right alongside the embed (K-050 M5 Stage 3) — no fusion yet, every
    extracted entity is a fresh node (plan §3.1). Fusion is a later stage;
    this call only returns `{documentId, chunkCount, status: 'processing'}`.
    """
    ctx = _get_context()
    receipt = _svc().ingest_document(
        ctx, text=text, title=title, source_format=source_format,
        source_label=source_label,
    )
    if _embed_worker is not None or _ingestion_pipeline is not None:
        chunks = _svc().list_document_chunks(ctx, document_id=receipt["documentId"])
        _schedule_chunk_processing(
            _schedule, ctx.ws, receipt["documentId"], chunks,
            embed_worker=_embed_worker, ingestion_pipeline=_ingestion_pipeline,
        )
    return receipt


@mcp.tool()
def ingest_documents(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bulk-ingest multiple documents in one call (FR-11, K-050 M5 Stage 6a).

    Each entry in `items` takes the same fields as `ingest_document`'s own
    parameters: `text` (required), `title`, `source_format` (defaults
    `"text"`), `source_label` (all optional except `text`). Returns **one
    receipt per item**, in the same order as `items` — a per-item failure
    (empty text, oversized text, unknown actor) does not abort the batch; it
    comes back as that item's own `{"status": "error", "error": ...,
    "errorType": ...}` receipt instead (`Services.ingest_documents`
    docstring has the full reasoning). Chunk embedding/extraction is
    scheduled the same as `ingest_document`, per chunk, for every
    successfully-ingested item in the batch — never for an item that errored.
    Cross-document fusion (AC-8) happens naturally once each item's
    independent background extraction runs; no batch-local fusion logic is
    needed (plan §3.6).
    """
    ctx = _get_context()
    receipts = _svc().ingest_documents(ctx, documents=items)
    if _embed_worker is not None or _ingestion_pipeline is not None:
        for receipt in receipts:
            if receipt.get("status") != "processing":
                continue  # this item errored — nothing to schedule for it
            chunks = _svc().list_document_chunks(
                ctx, document_id=receipt["documentId"]
            )
            _schedule_chunk_processing(
                _schedule, ctx.ws, receipt["documentId"], chunks,
                embed_worker=_embed_worker, ingestion_pipeline=_ingestion_pipeline,
            )
    return receipts


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


# ── §14.6 Entity fusion review surface (K-050 M5 Stage 4, FR-10/OQ-2) ────────


@mcp.tool()
def list_pending_matches(limit: int = 50) -> list[dict[str, Any]]:
    """List `SAME_AS` suggestions awaiting confirm/reject (OQ-2's review
    surface) — oldest first, each carrying both entities' id + name."""
    ctx = _get_context()
    return _svc().list_pending_matches(ctx, limit=limit)


@mcp.tool()
def list_matches(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """List `SAME_AS` matches, optionally filtered by `status` (`pending` /
    `confirmed` / `rejected`) — with no filter, includes the auto-merged tier
    (`status='confirmed', decidedBy='system'`), otherwise undiscoverable."""
    ctx = _get_context()
    return _svc().list_matches(ctx, status=status, limit=limit)


@mcp.tool()
def confirm_match(match_id: str) -> dict[str, Any]:
    """Confirm a `SAME_AS` suggestion (FR-10) — the two entities are "the
    same," recorded as an edge, never physically merged."""
    ctx = _get_context()
    return _svc().confirm_match(ctx, match_id=match_id)


@mcp.tool()
def reject_match(match_id: str) -> dict[str, Any]:
    """Reject a `SAME_AS` suggestion (FR-10) — the edge stays as a `rejected`
    record, reversible via `recheck_match` or automatic re-derivation (OQ-3)."""
    ctx = _get_context()
    return _svc().reject_match(ctx, match_id=match_id)


@mcp.tool()
def recheck_match(match_id: str) -> dict[str, Any] | None:
    """Manually reopen a `rejected` `SAME_AS` match back to `pending` (OQ-3).
    A no-op (returns `None`) if `match_id` is unknown or not `rejected`."""
    ctx = _get_context()
    return _svc().recheck_match(ctx, match_id=match_id)
