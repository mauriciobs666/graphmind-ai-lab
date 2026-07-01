"""Unit tests for the service layer.

Services own the invariants (write-variant dispatch, mention validation, RO/RW
read dispatch, `cursorId` construction, id/clock generation). They are tested
against a fake repository so the logic is pinned without a live database.
"""

from __future__ import annotations

import itertools

import pytest

from falkorchat.config import CallContext
from falkorchat.services import (
    ChannelNotFoundError,
    Services,
    ThreadNotFoundError,
    UnknownMemberError,
)

CTX = CallContext(ws="test", actor="u1")


class FakeRepo:
    """Records calls and simulates the small amount of state services depend on."""

    def __init__(self):
        self.channels: set[str] = set()
        self.threads: set[str] = set()
        self.heads: set[str] = set()          # threads that already have a HEAD
        self.members: set[str] = set()        # known member ids
        self.cursors: dict[str, int] = {}     # cursorId -> lastReadAt
        self.calls: list[tuple] = []
        self.since_rows: list[dict] = []

    # writes / lookups used by services
    def create_channel(self, ws, *, channel_id, name, created_at):
        self.channels.add(channel_id)
        self.calls.append(("create_channel", ws, channel_id, name, created_at))

    def channel_exists(self, ws, *, channel_id):
        return channel_id in self.channels

    def create_thread(self, ws, *, channel_id, thread_id, title, created_at):
        self.threads.add(thread_id)
        self.calls.append(("create_thread", ws, channel_id, thread_id, title, created_at))

    def thread_exists(self, ws, *, thread_id):
        return thread_id in self.threads

    def thread_has_head(self, ws, *, thread_id):
        return thread_id in self.heads

    def existing_members(self, ws, *, ids):
        return {i for i in ids if i in self.members}

    def post_first_message(self, ws, **kw):
        self.heads.add(kw["thread_id"])
        self.calls.append(("post_first_message", kw))

    def post_subsequent_message(self, ws, **kw):
        self.calls.append(("post_subsequent_message", kw))

    def get_cursor(self, ws, *, cursor_id):
        return self.cursors.get(cursor_id)

    def advance_cursor(self, ws, *, me_id, thread_id, cursor_id, now):
        prev = self.cursors.get(cursor_id, 0)
        self.cursors[cursor_id] = max(prev, now)
        self.calls.append(("advance_cursor", cursor_id, now))
        return self.cursors[cursor_id]

    def read_thread_since(self, ws, *, thread_id, me_id, since, limit=50):
        self.calls.append(("read_thread_since", thread_id, me_id, since, limit))
        return self.since_rows

    def read_ws_since(self, ws, *, me_id, since, limit=50):
        self.calls.append(("read_ws_since", me_id, since, limit))
        return self.since_rows


def make_service(repo, *, now=1000):
    ids = (f"id{n}" for n in itertools.count(1))
    return Services(repo, clock=lambda: now, id_gen=lambda: next(ids))


# ── create_channel / create_thread ─────────────────────────────────────────────


def test_create_channel_generates_id_and_time():
    repo = FakeRepo()
    svc = make_service(repo, now=1000)

    ch = svc.create_channel(CTX, name="general")

    assert ch["channelId"] == "id1"
    assert ch["name"] == "general"
    assert ch["createdAt"] == 1000
    assert "id1" in repo.channels


def test_create_thread_requires_existing_channel():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(ChannelNotFoundError):
        svc.create_thread(CTX, channel_id="missing", title="hi")


def test_create_thread_creates_when_channel_exists():
    repo = FakeRepo()
    repo.channels.add("c1")
    svc = make_service(repo, now=1000)

    th = svc.create_thread(CTX, channel_id="c1", title="hi")

    assert th["threadId"] == "id1"
    assert th["channelId"] == "c1"
    assert th["createdAt"] == 1000


# ── post_message: dispatch + validation ─────────────────────────────────────────


def test_post_message_missing_thread_errors():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(ThreadNotFoundError):
        svc.post_message(CTX, thread_id="nope", text="hi")


def test_post_message_first_uses_first_write_path():
    repo = FakeRepo()
    repo.threads.add("t1")  # exists, no head yet
    svc = make_service(repo, now=1000)

    msg = svc.post_message(CTX, thread_id="t1", text="hello")

    assert repo.calls[-1][0] == "post_first_message"
    assert msg["msgId"] == "id1"
    assert msg["authorId"] == "u1"
    assert msg["role"] == "user"
    assert msg["createdAt"] == 1000


def test_post_message_subsequent_uses_append_path():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.heads.add("t1")  # already has a head
    svc = make_service(repo)

    svc.post_message(CTX, thread_id="t1", text="second")

    assert repo.calls[-1][0] == "post_subsequent_message"


def test_post_message_rejects_unknown_mention():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.members.add("u2")
    svc = make_service(repo)

    with pytest.raises(UnknownMemberError):
        svc.post_message(CTX, thread_id="t1", text="hi", mentions=["u2", "ghost"])

    # nothing written when validation fails
    assert not any(c[0].startswith("post_") for c in repo.calls)


def test_post_message_dedups_mentions_before_write():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.members.update({"u2"})
    svc = make_service(repo)

    msg = svc.post_message(CTX, thread_id="t1", text="hi", mentions=["u2", "u2"])

    kw = repo.calls[-1][1]
    assert kw["mentions"] == ["u2"]
    assert msg["mentions"] == ["u2"]


# ── read_messages: RO/RW dispatch ───────────────────────────────────────────────


def test_read_messages_explicit_since_is_pure_read_no_advance():
    repo = FakeRepo()
    svc = make_service(repo)

    svc.read_messages(CTX, thread_id="t1", since=50, advance=True)

    assert ("read_thread_since", "t1", "u1", 50, 50) in repo.calls
    assert not any(c[0] == "advance_cursor" for c in repo.calls)


def test_read_messages_thread_uses_cursor_and_advances():
    repo = FakeRepo()
    repo.cursors["u1:t1"] = 200
    svc = make_service(repo, now=1000)

    svc.read_messages(CTX, thread_id="t1", advance=True)

    assert ("read_thread_since", "t1", "u1", 200, 50) in repo.calls
    assert ("advance_cursor", "u1:t1", 1000) in repo.calls


def test_read_messages_no_cursor_defaults_since_zero():
    repo = FakeRepo()
    svc = make_service(repo, now=1000)

    svc.read_messages(CTX, thread_id="t1", advance=False)

    assert ("read_thread_since", "t1", "u1", 0, 50) in repo.calls
    assert not any(c[0] == "advance_cursor" for c in repo.calls)


def test_read_messages_room_wide_requires_no_thread_and_no_advance():
    repo = FakeRepo()
    svc = make_service(repo)

    svc.read_messages(CTX, since=None, advance=True)  # no thread_id → room-wide, since 0

    assert ("read_ws_since", "u1", 0, 50) in repo.calls
    assert not any(c[0] == "advance_cursor" for c in repo.calls)
