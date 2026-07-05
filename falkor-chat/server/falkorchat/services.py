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

import threading
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
        self._ts_lock = threading.Lock()
        self._last_ts = 0

    def _next_ts(self) -> int:
        """Monotonic per-process ms clock — makes same-ms message ties impossible
        (K-007 item 4a). Used only for message `createdAt`; channel/thread stamps
        keep the plain clock (ties there are harmless). Lock-guarded because
        FastAPI runs sync endpoints on a threadpool."""
        with self._ts_lock:
            ts = max(self._clock(), self._last_ts + 1)
            self._last_ts = ts
            return ts

    # ── health ──────────────────────────────────────────────────────────────────

    def ping(self, ctx: CallContext) -> bool:
        """True when the workspace graph answers a trivial read."""
        return self._repo.ping(ctx.ws)

    # ── members ─────────────────────────────────────────────────────────────────

    def ensure_actor(self, ctx: CallContext) -> None:
        """Project the context actor into the workspace as a `User` (idempotent).

        Called at app startup so the configured actor exists before the first
        write — the §4 write paths refuse an unknown author. The configured
        actor is projected as a `User`; Agent actors (seeded via
        `repo.ensure_agent`) post with role `assistant` — real per-client agent
        auth is still to come.
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

        Validates the actor and mentions, derives `role` from the actor's node
        label (`User → user`, `Agent → assistant` — agents can author), then
        dispatches on the §4 v2 status row (QUERIES.md §4 contract):
        `dupMsg` = idempotent retry success; `hadHead` = lost the first-post
        race → re-dispatch as subsequent; subsequent with no TAIL → `None` →
        re-dispatch as first. The loop bound is a tripwire — ping-pong is
        impossible by contract (a headed thread always has a TAIL).
        """
        if not self._repo.thread_exists(ctx.ws, thread_id=thread_id):
            raise ThreadNotFoundError(thread_id)

        wanted = _dedup(list(mentions or []))
        # One member-kind lookup covers the author and every mention. The author
        # check is load-bearing: an unknown author makes the v2 write refuse
        # (authorFound=false) — validate before writing.
        kinds = self._repo.resolve_member_kinds(ctx.ws, ids=[ctx.actor, *wanted])
        actor_kind = kinds.get(ctx.actor)
        if actor_kind is None:
            raise UnknownActorError(ctx.actor)
        role = "user" if actor_kind == "User" else "assistant"
        unknown = [m for m in wanted if kinds.get(m) is None]
        if unknown:
            raise UnknownMemberError(unknown)

        msg_id, now = self._id(), self._next_ts()
        use_first = not self._repo.thread_has_head(ctx.ws, thread_id=thread_id)
        for _attempt in range(4):
            write = (
                self._repo.post_first_message
                if use_first
                else self._repo.post_subsequent_message
            )
            st = write(
                ctx.ws, thread_id=thread_id, msg_id=msg_id, author_id=ctx.actor,
                text=text, role=role, created_at=now, mentions=wanted,
            )
            if st is None:
                if use_first:                    # thread anchor vanished (TOCTOU)
                    raise ThreadNotFoundError(thread_id)
                use_first = True                 # no TAIL yet — retry as first-post
                continue
            if st.written or st.dup_msg:         # dup_msg = idempotent success (OQ2)
                break
            if not st.author_found:              # belt-and-suspenders vs the pre-check
                raise UnknownActorError(ctx.actor)
            if st.had_head:                      # lost the first-post race
                use_first = False
                continue
            raise RuntimeError(f"unexpected write status {st!r} (thread={thread_id!r})")
        else:
            raise RuntimeError(
                "message write dispatch did not converge "
                f"(thread={thread_id!r}, msg={msg_id!r})"
            )
        return {
            "msgId": msg_id, "threadId": thread_id, "authorId": ctx.actor,
            "text": text, "role": role, "createdAt": now, "mentions": wanted,
        }

    def read_messages(
        self, ctx: CallContext, *, thread_id: str | None = None,
        since: int | None = None, limit: int = 50, advance: bool = True,
    ) -> list[dict[str, Any]]:
        """Read messages since a cursor/timestamp.

        Modes:
          * explicit ``since`` → pure read with plain ``>`` timestamp semantics;
            the cursor is never touched. May re-deliver or skip messages within
            that exact millisecond (OQ3 contract) — agents that need lossless
            catch-up use cursor mode.
          * no ``since`` + ``thread_id`` → read from the member's per-thread
            composite cursor ``(lastReadAt, lastReadMsgId)`` (or the epoch base
            ``(0, '')``), then, when ``advance`` is set, move the cursor forward
            to the newest ``(createdAt, msgId)`` pair actually delivered (a
            write). Never the server clock — that would permanently skip rows a
            ``limit`` truncated. An empty page advances nothing. Cursor-driven
            reads never skip or re-deliver, even across millisecond ties.
          * no ``since`` + no ``thread_id`` → room-wide read from epoch 0. There
            is no room-wide cursor in M1, so nothing is advanced.
        """
        explicit_since = since is not None

        if thread_id is not None:
            cursor_id = f"{ctx.actor}:{thread_id}"
            if explicit_since:
                return self._repo.read_thread_since(
                    ctx.ws, thread_id=thread_id, me_id=ctx.actor,
                    since=since, since_msg_id=None, limit=limit,  # plain `>`
                )
            pair = self._repo.get_cursor(ctx.ws, cursor_id=cursor_id)
            eff_since, eff_msg = pair if pair is not None else (0, None)
            rows = self._repo.read_thread_since(
                ctx.ws, thread_id=thread_id, me_id=ctx.actor,
                since=eff_since or 0, since_msg_id=eff_msg or "", limit=limit,
            )
            if advance and rows:
                last = rows[-1]  # rows are ORDER BY (createdAt, msgId) — the max pair
                self._repo.advance_cursor(
                    ctx.ws, me_id=ctx.actor, thread_id=thread_id,
                    cursor_id=cursor_id,
                    now=last["createdAt"], now_msg_id=last["msgId"],
                )
            return rows

        # room-wide: no cursor, defaults to epoch 0, never advances (plain `>`)
        eff_since = since if explicit_since else 0
        return self._repo.read_ws_since(
            ctx.ws, me_id=ctx.actor, since=eff_since, since_msg_id=None, limit=limit
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
