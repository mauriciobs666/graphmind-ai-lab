"""Domain logic — the invariants live here (DESIGN §14.2).

`services.py` is the only layer that:
  * generates ids and timestamps (server clock — never client-supplied),
  * picks the first-vs-subsequent §4 message write variant,
  * validates that mentions resolve to known members before writing,
  * constructs `cursorId` and decides read-only vs read-write for `read_messages`.

Both front doors (`api.py` REST, `mcp.py` MCP tools) are thin adapters over
these methods; they carry no business logic.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from redis.exceptions import ResponseError

from .config import CallContext
from .repository import Repository

# ── errors ─────────────────────────────────────────────────────────────────────


class ServiceError(Exception):
    """Base class for service-layer validation errors."""


class ChannelNotFoundError(ServiceError):
    pass


class ThreadNotFoundError(ServiceError):
    pass


class UnknownMemberError(ServiceError):
    """Raised when a mention does not resolve to a known member."""


class UnknownActorError(ServiceError):
    """Raised when the context actor does not resolve to a known member.

    Guards the silent-no-op failure mode: the §4 write queries anchor on
    `MATCH (author …)`, and a missing author makes the whole write a no-op
    while the transport would still report success.
    """


class InvalidSearchQueryError(ServiceError):
    """Raised when the full-text query is rejected by RediSearch syntax."""


def _default_id() -> str:
    return uuid.uuid4().hex


def _default_clock() -> int:
    """Server clock in milliseconds since the epoch."""
    return int(time.time() * 1000)


def _dedup(items: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    return list(dict.fromkeys(items))


class Services:
    def __init__(
        self,
        repo: Repository,
        *,
        clock: Callable[[], int] = _default_clock,
        id_gen: Callable[[], str] = _default_id,
    ) -> None:
        self._repo = repo
        self._clock = clock
        self._id = id_gen

    # ── health ──────────────────────────────────────────────────────────────────

    def ping(self, ctx: CallContext) -> bool:
        """True when the workspace graph answers a trivial read."""
        return self._repo.ping(ctx.ws)

    # ── members ─────────────────────────────────────────────────────────────────

    def ensure_actor(self, ctx: CallContext) -> None:
        """Project the context actor into the workspace as a `User` (idempotent).

        Called at app startup so the M1 hardcoded actor exists before the first
        write — the §4 write paths anchor on the author node. The actor is a
        `User` in M1; agent actors arrive with real per-client auth.
        """
        self._repo.ensure_user(ctx.ws, user_id=ctx.actor)

    # ── channels ────────────────────────────────────────────────────────────────

    def create_channel(self, ctx: CallContext, *, name: str) -> dict[str, Any]:
        channel_id = self._id()
        now = self._clock()
        self._repo.create_channel(
            ctx.ws, channel_id=channel_id, name=name, created_at=now
        )
        return {"channelId": channel_id, "name": name, "createdAt": now}

    def list_channels(self, ctx: CallContext, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._repo.list_channels(ctx.ws, limit=limit)

    # ── threads ─────────────────────────────────────────────────────────────────

    def create_thread(
        self, ctx: CallContext, *, channel_id: str, title: str
    ) -> dict[str, Any]:
        if not self._repo.channel_exists(ctx.ws, channel_id=channel_id):
            raise ChannelNotFoundError(channel_id)
        thread_id = self._id()
        now = self._clock()
        self._repo.create_thread(
            ctx.ws, channel_id=channel_id, thread_id=thread_id,
            title=title, created_at=now,
        )
        return {
            "threadId": thread_id, "channelId": channel_id,
            "title": title, "createdAt": now,
        }

    def list_threads(
        self, ctx: CallContext, *, channel_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self._repo.list_threads(ctx.ws, channel_id=channel_id, limit=limit)

    # ── messages ────────────────────────────────────────────────────────────────

    def post_message(
        self, ctx: CallContext, *, thread_id: str, text: str,
        mentions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Post a message into an existing thread.

        Picks the first-vs-subsequent §4 write variant, validates mentions, and
        attributes the message to the context actor. The actor is a `User` in M1
        (role ``user``); agent authorship arrives with real per-client auth.
        """
        if not self._repo.thread_exists(ctx.ws, thread_id=thread_id):
            raise ThreadNotFoundError(thread_id)

        wanted = _dedup(list(mentions or []))
        # One membership lookup covers the author and every mention. The author
        # check is load-bearing: the §4 write anchors on the author node, and a
        # missing author would silently no-op the entire write.
        known = self._repo.existing_members(ctx.ws, ids=[ctx.actor, *wanted])
        if ctx.actor not in known:
            raise UnknownActorError(ctx.actor)
        unknown = [m for m in wanted if m not in known]
        if unknown:
            raise UnknownMemberError(unknown)

        msg_id = self._id()
        now = self._clock()
        write = (
            self._repo.post_subsequent_message
            if self._repo.thread_has_head(ctx.ws, thread_id=thread_id)
            else self._repo.post_first_message
        )
        write(
            ctx.ws, thread_id=thread_id, msg_id=msg_id, author_id=ctx.actor,
            text=text, role="user", created_at=now, mentions=wanted,
        )
        return {
            "msgId": msg_id, "threadId": thread_id, "authorId": ctx.actor,
            "text": text, "role": "user", "createdAt": now, "mentions": wanted,
        }

    def read_messages(
        self, ctx: CallContext, *, thread_id: str | None = None,
        since: int | None = None, limit: int = 50, advance: bool = True,
    ) -> list[dict[str, Any]]:
        """Read messages since a cursor/timestamp.

        Modes:
          * explicit ``since`` → pure read; the cursor is never touched.
          * no ``since`` + ``thread_id`` → read from the member's per-thread
            cursor (or epoch 0 if none), then, when ``advance`` is set, move the
            cursor forward to the newest ``createdAt`` actually delivered (a
            write). Never the server clock — that would permanently skip rows a
            ``limit`` truncated. An empty page advances nothing.
          * no ``since`` + no ``thread_id`` → room-wide read from epoch 0. There
            is no room-wide cursor in M1, so nothing is advanced.
        """
        explicit_since = since is not None

        if thread_id is not None:
            cursor_id = f"{ctx.actor}:{thread_id}"
            eff_since = since if explicit_since else (
                self._repo.get_cursor(ctx.ws, cursor_id=cursor_id) or 0
            )
            rows = self._repo.read_thread_since(
                ctx.ws, thread_id=thread_id, me_id=ctx.actor,
                since=eff_since, limit=limit,
            )
            if advance and not explicit_since and rows:
                self._repo.advance_cursor(
                    ctx.ws, me_id=ctx.actor, thread_id=thread_id,
                    cursor_id=cursor_id,
                    now=max(r["createdAt"] for r in rows),
                )
            return rows

        # room-wide: no cursor, defaults to epoch 0, never advances
        eff_since = since if explicit_since else 0
        return self._repo.read_ws_since(
            ctx.ws, me_id=ctx.actor, since=eff_since, limit=limit
        )

    def search_messages(
        self, ctx: CallContext, *, query: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Workspace-wide full-text keyword search. QUERIES.md §5.

        RediSearch parses the query string; its syntax errors (unbalanced
        quotes, stray operators) are a caller problem, not a server fault.
        """
        try:
            return self._repo.search_messages(ctx.ws, query=query, limit=limit)
        except ResponseError as exc:
            raise InvalidSearchQueryError(str(exc)) from exc

    # ── reads (thin passthroughs) ───────────────────────────────────────────────

    def read_thread(self, ctx: CallContext, *, thread_id: str) -> list[dict[str, Any]]:
        return self._repo.read_thread(ctx.ws, thread_id=thread_id)

    def get_message(self, ctx: CallContext, *, msg_id: str) -> dict[str, Any] | None:
        return self._repo.get_message(ctx.ws, msg_id=msg_id)
