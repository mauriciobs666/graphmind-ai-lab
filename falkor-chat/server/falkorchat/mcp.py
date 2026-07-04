"""MCP (Streamable-HTTP) transport — a peer of `api.py` (plan §3, §6).

A thin adapter that translates MCP tool calls into the same `Services` methods the
REST router calls. No business logic lives here. The `Services` instance and the
actor/context provider are injected via `configure(...)` at app-build time (and in
tests), so this module never hardcodes the tenant.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import config
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


def configure(
    services: Services,
    *,
    context_provider: Callable[[], CallContext] | None = None,
) -> FastMCP:
    """Wire the MCP tools to a `Services` instance (and optional context seam)."""
    global _services, _get_context
    _services = services
    if context_provider is not None:
        _get_context = context_provider
    return mcp


def _svc() -> Services:
    if _services is None:  # pragma: no cover - guards against unconfigured use
        raise RuntimeError("MCP tools used before configure() was called")
    return _services


@mcp.tool()
def send_message(
    body: str, re: str, mentions: list[str] | None = None, frm: str | None = None
) -> dict[str, Any]:
    """Post `body` into thread `re`, optionally mentioning members.

    `frm` is reserved/ignored in M1 — the author is the configured actor (Q#1).
    """
    ctx = _get_context()
    return _svc().post_message(ctx, thread_id=re, text=body, mentions=mentions)


@mcp.tool()
def read_messages(
    re: str | None = None,
    since: int | None = None,
    limit: int = 50,
    advance: bool = True,
) -> list[dict[str, Any]]:
    """Catch up on messages.

    With `re` (thread id): read that thread since your cursor (or explicit
    `since`), advancing the cursor unless `since` is given. Without `re`:
    workspace-wide read since `since` (default epoch 0); no cursor is advanced.
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
