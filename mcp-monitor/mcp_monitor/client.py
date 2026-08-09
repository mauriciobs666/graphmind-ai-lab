"""Per-server MCP client connection lifecycle (plan §4).

One `ServerConnection` per **named server** (`[server.*]`), shared by every
watch that references it — not one connection per watch — so two watches
against the same server don't open two redundant sessions (FR-6: "different
patterns against the same tool").

Two review findings are addressed here:

- **Major** — a stdio connection is built from `ServerConfig.command`
  (already split to a bare executable `str` by `config.py`) and
  `ServerConfig.args` (the rest, a `list[str]`), matching the installed `mcp`
  SDK's `StdioServerParameters(command: str, args: list[str] = ...)`. The
  plan's own worked example passed the whole argv array as `command`, which
  raises a Pydantic `ValidationError` at construction.
- **Moderate** — `_lock` (an `asyncio.Lock`) guards open/discard/reopen so two
  watches polling the same shared connection concurrently (independent
  `asyncio.Task`s, plan §3) never race on the same session object: one
  watch's failure-triggered discard-and-reopen can no longer happen while
  another watch's call is in flight on the same object.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .config import ServerConfig


class ToolCallError(RuntimeError):
    """Raised when a tool call completes but reports `isError = True`.

    Distinct from a transport-level exception (broken pipe, unreachable
    server): the session is still healthy, so `ServerConnection` does not
    discard it for this case — only watch.py's poll-failure handling reacts.
    """


def _content_to_text(result: Any) -> str:
    """Flatten a `CallToolResult` into the raw text a watch's regex runs against.

    Mirrors `falkor-chat/server/falkorchat/tools.py: _content_to_text` — prefer
    `structuredContent` (JSON-encoded) when the server returned it, else
    concatenate the text content blocks. This is FR-5's "full raw tool
    result": the whole flattened string, not a parsed/filtered subset.
    """
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, default=str)
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


@contextlib.asynccontextmanager
async def _stdio_session(config: ServerConfig) -> AsyncIterator[ClientSession]:
    params = StdioServerParameters(command=config.command, args=config.args, env=config.env or None)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@contextlib.asynccontextmanager
async def _http_session(config: ServerConfig) -> AsyncIterator[ClientSession]:
    async with streamablehttp_client(config.url) as (read, write, *_rest):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _default_session_cm(config: ServerConfig):
    if config.transport == "stdio":
        return _stdio_session(config)
    return _http_session(config)


class ServerConnection:
    """A lazily-opened, reused-on-success, discard-on-failure MCP session.

    `session_cm_factory` is the test seam: it must return a fresh async
    context manager yielding an initialized `ClientSession` each time it is
    called. The default builds one from `config` per transport; tests can pass
    `lambda: mcp.shared.memory.create_connected_server_and_client_session(stub)`
    directly, since that helper's own shape (an async CM yielding a connected,
    initialized `ClientSession`) is exactly what this class expects — no stdio
    or HTTP transport involved.
    """

    def __init__(self, name: str, config: ServerConfig, *, session_cm_factory=None) -> None:
        self._name = name
        self._config = config
        self._session_cm_factory = session_cm_factory or (lambda: _default_session_cm(config))
        self._lock = asyncio.Lock()
        self._cm: Any = None
        self._session: ClientSession | None = None

    @property
    def name(self) -> str:
        return self._name

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> str:
        """Call `tool`, returning the flattened result text.

        Two distinct failure shapes both surface as a raised exception here,
        matching plan §3 step 2's "network error, tool error" — both are a
        poll failure a watch logs and retries past (FR-10):

        - A **transport-level** failure (unreachable server, broken pipe) is
          whatever `session.call_tool` itself raises. The session is then
          discarded (not retried mid-call) so the *next* poll that needs this
          connection lazily reopens it (plan §4) — this is what lets a poll
          failure survive a server restart rather than replaying the same
          dead session forever.
        - A **tool-level** failure is the MCP protocol's own error shape: the
          call returns normally with `CallToolResult.isError = True` (verified
          against the installed SDK — a tool body raising does *not* propagate
          client-side, FastMCP packages it as an error result instead). That
          is raised here as `ToolCallError` so watch.py's single
          except-and-log path covers it too, without discarding a connection
          that is otherwise healthy.

        The whole open-call-discard sequence runs under `self._lock`, so a
        concurrent watch sharing this connection never observes (or races) a
        half-discarded session.
        """
        async with self._lock:
            session = await self._ensure_open()
            try:
                result = await session.call_tool(tool, arguments)
            except Exception:
                await self._discard_locked()
                raise
            text = _content_to_text(result)
            if getattr(result, "isError", False):
                raise ToolCallError(
                    f"tool {tool!r} on server {self._name!r} returned an error: {text}"
                )
            return text

    async def aclose(self) -> None:
        async with self._lock:
            await self._discard_locked()

    async def _ensure_open(self) -> ClientSession:
        if self._session is not None:
            return self._session
        cm = self._session_cm_factory()
        session = await cm.__aenter__()
        self._cm = cm
        self._session = session
        return session

    async def _discard_locked(self) -> None:
        cm, self._cm = self._cm, None
        self._session = None
        if cm is not None:
            with contextlib.suppress(Exception):
                await cm.__aexit__(None, None, None)
