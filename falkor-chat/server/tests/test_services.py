"""Unit tests for the service layer.

Services own the invariants (write-variant dispatch, mention validation, RO/RW
read dispatch, `cursorId` construction, id/clock generation). They are tested
against a fake repository so the logic is pinned without a live database.
"""

from __future__ import annotations

import itertools

import pytest
from redis.exceptions import ResponseError

from falkorchat.config import CallContext
from falkorchat.repository import MessageWriteStatus
from falkorchat.services import (
    ChannelNotFoundError,
    InvalidSearchQueryError,
    MemberIdCollisionError,
    Services,
    ThreadNotFoundError,
    UnknownActorError,
    UnknownMemberError,
)

CTX = CallContext(ws="test", actor="u1")

OK = MessageWriteStatus(written=True, had_head=False, dup_msg=False, author_found=True)
DUP = MessageWriteStatus(written=False, had_head=False, dup_msg=True, author_found=True)
HAD_HEAD = MessageWriteStatus(written=False, had_head=True, dup_msg=False, author_found=True)


class FakeRepo:
    """Records calls and simulates the small amount of state services depend on."""

    def __init__(self):
        self.channels: set[str] = set()
        self.threads: set[str] = set()
        self.heads: set[str] = set()          # threads that already have a HEAD
        self.members: set[str] = set()        # known User ids
        self.agents: set[str] = set()         # known Agent ids
        # cursorId -> (lastReadAt, lastReadMsgId) composite pair
        self.cursors: dict[str, tuple[int, str | None]] = {}
        self.calls: list[tuple] = []
        self.since_rows: list[dict] = []
        # scripted §4 v2 status rows (popped first); default behavior otherwise
        self.first_status: list[MessageWriteStatus | None] = []
        self.subseq_status: list[MessageWriteStatus | None] = []
        self.agent_first_status: list[MessageWriteStatus | None] = []
        self.agent_subseq_status: list[MessageWriteStatus | None] = []

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

    def resolve_member_kinds(self, ws, *, ids):
        def kind(i):
            if i in self.agents:
                return "Agent"
            if i in self.members:
                return "User"
            return None

        return {i: kind(i) for i in ids}

    def post_first_message(self, ws, **kw):
        self.calls.append(("post_first_message", kw))
        if self.first_status:
            return self.first_status.pop(0)
        self.heads.add(kw["thread_id"])
        return OK

    def post_subsequent_message(self, ws, **kw):
        self.calls.append(("post_subsequent_message", kw))
        if self.subseq_status:
            return self.subseq_status.pop(0)
        return OK

    def post_agent_answer(self, ws, **kw):
        self.calls.append(("post_agent_answer", kw))
        if self.agent_subseq_status:
            return self.agent_subseq_status.pop(0)
        return OK

    def post_agent_answer_first(self, ws, **kw):
        self.calls.append(("post_agent_answer_first", kw))
        if self.agent_first_status:
            return self.agent_first_status.pop(0)
        self.heads.add(kw["thread_id"])
        return OK

    def get_cursor(self, ws, *, cursor_id):
        return self.cursors.get(cursor_id)

    def advance_cursor(self, ws, *, me_id, thread_id, cursor_id, now, now_msg_id):
        prev = self.cursors.get(cursor_id, (0, ""))
        self.cursors[cursor_id] = max(prev, (now, now_msg_id))
        self.calls.append(("advance_cursor", cursor_id, now, now_msg_id))
        return self.cursors[cursor_id]

    def read_thread_since(self, ws, *, thread_id, me_id, since, since_msg_id=None,
                          limit=50):
        self.calls.append(
            ("read_thread_since", thread_id, me_id, since, since_msg_id, limit)
        )
        return self.since_rows

    def read_ws_since(self, ws, *, me_id, since, since_msg_id=None, limit=50):
        self.calls.append(("read_ws_since", me_id, since, since_msg_id, limit))
        return self.since_rows

    def search_messages(self, ws, *, query, limit=50):
        self.calls.append(("search_messages", query, limit))
        if isinstance(self.since_rows, Exception):
            raise self.since_rows
        return self.since_rows

    def hybrid_search(self, ws, *, q_vec, k, limit, channel_id=None, timeout=None):
        self.calls.append(("hybrid_search", ws, tuple(q_vec), k, limit, channel_id, timeout))
        return self.since_rows

    def ensure_user(self, ws, *, user_id, display_name=None, email=None):
        # mirrors the §2 v2 guarded ensure: an id held by an Agent is refused
        if user_id in self.agents:
            raise MemberIdCollisionError(user_id)
        self.members.add(user_id)
        self.calls.append(("ensure_user", ws, user_id))


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


def test_post_message_unknown_actor_errors_instead_of_silent_noop():
    repo = FakeRepo()
    repo.threads.add("t1")  # thread exists but the actor u1 is not a member
    svc = make_service(repo)

    with pytest.raises(UnknownActorError):
        svc.post_message(CTX, thread_id="t1", text="hi")

    assert not any(c[0].startswith("post_") for c in repo.calls)


def test_ensure_actor_projects_context_actor_as_user():
    repo = FakeRepo()
    svc = make_service(repo)

    svc.ensure_actor(CTX)

    assert ("ensure_user", "test", "u1") in repo.calls


def test_ensure_actor_propagates_member_id_collision():
    """DEF-1: an actor id held by an Agent must surface, never silently shadow."""
    repo = FakeRepo()
    repo.agents.add("u1")
    svc = make_service(repo)

    with pytest.raises(MemberIdCollisionError):
        svc.ensure_actor(CTX)


def test_post_message_first_uses_first_write_path():
    repo = FakeRepo()
    repo.threads.add("t1")  # exists, no head yet
    repo.members.add("u1")
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
    repo.members.add("u1")
    svc = make_service(repo)

    svc.post_message(CTX, thread_id="t1", text="second")

    assert repo.calls[-1][0] == "post_subsequent_message"


def test_post_message_rejects_unknown_mention():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.members.update({"u1", "u2"})
    svc = make_service(repo)

    with pytest.raises(UnknownMemberError):
        svc.post_message(CTX, thread_id="t1", text="hi", mentions=["u2", "ghost"])

    # nothing written when validation fails
    assert not any(c[0].startswith("post_") for c in repo.calls)


def test_post_message_dedups_mentions_before_write():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.members.update({"u1", "u2"})
    svc = make_service(repo)

    msg = svc.post_message(CTX, thread_id="t1", text="hi", mentions=["u2", "u2"])

    kw = repo.calls[-1][1]
    assert kw["mentions"] == ["u2"]
    assert msg["mentions"] == ["u2"]


# ── post_message: v2 status dispatch + role derivation (K-007) ──────────────────


def test_post_message_agent_actor_gets_assistant_role():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.agents.add("a1")
    svc = make_service(repo)

    msg = svc.post_message(CallContext(ws="test", actor="a1"), thread_id="t1", text="hi")

    assert msg["role"] == "assistant"
    assert repo.calls[-1][1]["role"] == "assistant"  # derived, never trusted


def test_post_message_dup_msg_is_idempotent_success():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.heads.add("t1")
    repo.members.add("u1")
    repo.subseq_status = [DUP]  # retry replay of our own write
    svc = make_service(repo, now=1000)

    msg = svc.post_message(CTX, thread_id="t1", text="hi")

    assert msg["msgId"] == "id1"  # returned as success, no error
    assert msg["role"] == "user"


def test_post_message_had_head_redispatches_as_subsequent():
    repo = FakeRepo()
    repo.threads.add("t1")  # no HEAD seen at dispatch time…
    repo.members.add("u1")
    repo.first_status = [HAD_HEAD]  # …but another writer won the first-post race
    svc = make_service(repo)

    svc.post_message(CTX, thread_id="t1", text="hi")

    kinds = [c[0] for c in repo.calls if c[0].startswith("post_")]
    assert kinds == ["post_first_message", "post_subsequent_message"]


def test_post_message_tailless_subsequent_redispatches_as_first():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.heads.add("t1")  # HEAD seen at dispatch time…
    repo.members.add("u1")
    repo.subseq_status = [None]  # …but the anchor missed (no TAIL yet)
    svc = make_service(repo)

    svc.post_message(CTX, thread_id="t1", text="hi")

    kinds = [c[0] for c in repo.calls if c[0].startswith("post_")]
    assert kinds == ["post_subsequent_message", "post_first_message"]


def test_post_message_dispatch_loop_is_bounded():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.members.add("u1")
    # impossible-by-contract ping-pong: first says "had head", subsequent says "no TAIL"
    repo.first_status = [HAD_HEAD] * 10
    repo.subseq_status = [None] * 10
    svc = make_service(repo)

    with pytest.raises(RuntimeError):
        svc.post_message(CTX, thread_id="t1", text="hi")


def test_post_message_created_at_is_strictly_increasing_under_fixed_clock():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.members.add("u1")
    svc = make_service(repo, now=1000)  # frozen wall clock

    m1 = svc.post_message(CTX, thread_id="t1", text="one")
    m2 = svc.post_message(CTX, thread_id="t1", text="two")

    # monotonic per-process clock: same-ms ties are impossible at the source
    assert m1["createdAt"] == 1000
    assert m2["createdAt"] == 1001
    assert m2["createdAt"] > m1["createdAt"]


# ── read_messages: RO/RW dispatch ───────────────────────────────────────────────


def test_read_messages_explicit_since_is_pure_read_no_advance():
    repo = FakeRepo()
    svc = make_service(repo)

    svc.read_messages(CTX, thread_id="t1", since=50, advance=True)

    # explicit since → plain `>` semantics (since_msg_id=None), cursor untouched
    assert ("read_thread_since", "t1", "u1", 50, None, 50) in repo.calls
    assert not any(c[0] == "advance_cursor" for c in repo.calls)


def test_read_messages_thread_uses_cursor_and_advances_to_last_returned():
    repo = FakeRepo()
    repo.cursors["u1:t1"] = (200, "m0")
    repo.since_rows = [
        {"msgId": "m1", "createdAt": 300},
        {"msgId": "m2", "createdAt": 450},
    ]
    svc = make_service(repo, now=1000)

    svc.read_messages(CTX, thread_id="t1", advance=True)

    assert ("read_thread_since", "t1", "u1", 200, "m0", 50) in repo.calls
    # cursor moves to the newest row actually delivered — NOT the server clock,
    # which would skip rows a `limit` truncated (and race concurrent posts)
    assert ("advance_cursor", "u1:t1", 450, "m2") in repo.calls


def test_read_messages_empty_page_does_not_advance():
    repo = FakeRepo()
    repo.cursors["u1:t1"] = (200, "m0")
    svc = make_service(repo, now=1000)

    svc.read_messages(CTX, thread_id="t1", advance=True)  # since_rows is []

    assert not any(c[0] == "advance_cursor" for c in repo.calls)


def test_read_messages_no_cursor_defaults_since_zero():
    repo = FakeRepo()
    svc = make_service(repo, now=1000)

    svc.read_messages(CTX, thread_id="t1", advance=False)

    # no cursor yet → epoch base with the composite '' msgId convention
    assert ("read_thread_since", "t1", "u1", 0, "", 50) in repo.calls
    assert not any(c[0] == "advance_cursor" for c in repo.calls)


def test_read_messages_room_wide_requires_no_thread_and_no_advance():
    repo = FakeRepo()
    svc = make_service(repo)

    svc.read_messages(CTX, since=None, advance=True)  # no thread_id → room-wide, since 0

    assert ("read_ws_since", "u1", 0, None, 50) in repo.calls
    assert not any(c[0] == "advance_cursor" for c in repo.calls)


def test_read_messages_cursor_pair_round_trips_as_composite_since():
    repo = FakeRepo()
    repo.since_rows = [
        {"msgId": "m2", "createdAt": 300},
        {"msgId": "m3", "createdAt": 300},  # millisecond tie — last row is max pair
    ]
    svc = make_service(repo)

    svc.read_messages(CTX, thread_id="t1", advance=True)

    # advanced to the last returned (createdAt, msgId) pair
    assert repo.cursors["u1:t1"] == (300, "m3")

    repo.since_rows = []
    svc.read_messages(CTX, thread_id="t1", advance=True)

    # the stored pair is fed back as the composite since — tied siblings with a
    # larger msgId would still be delivered, nothing re-delivered
    assert ("read_thread_since", "t1", "u1", 300, "m3", 50) in repo.calls


# ── search: thin passthrough ────────────────────────────────────────────────────


def test_search_messages_passes_query_and_limit_through():
    repo = FakeRepo()
    repo.since_rows = [{"msgId": "m1", "text": "hello", "createdAt": 120, "score": 1.5}]
    svc = make_service(repo)

    hits = svc.search_messages(CTX, query="hello", limit=10)

    assert ("search_messages", "hello", 10) in repo.calls
    assert hits == repo.since_rows


def test_search_messages_maps_syntax_error_to_service_error():
    repo = FakeRepo()
    repo.since_rows = ResponseError("RediSearch: Syntax error at offset 6")
    svc = make_service(repo)

    with pytest.raises(InvalidSearchQueryError):
        svc.search_messages(CTX, query='hello"unbalanced')


# ── hybrid_search (GraphRAG) ────────────────────────────────────────────────────


def test_hybrid_search_applies_rag_timeout_constant():
    from falkorchat.services import RAG_QUERY_TIMEOUT_MS

    repo = FakeRepo()
    repo.since_rows = [
        {"msgId": "m1", "text": "cats", "role": "user", "score": 0.0, "relatedContext": []}
    ]
    svc = make_service(repo)

    hits = svc.hybrid_search(CTX, q_vec=[1.0, 0.0], k=10, limit=5)

    assert hits == repo.since_rows
    call = next(c for c in repo.calls if c[0] == "hybrid_search")
    # (name, ws, q_vec, k, limit, channel_id, timeout)
    assert call[1] == "test"
    assert call[3] == 10 and call[4] == 5
    assert call[5] is None
    assert call[6] == RAG_QUERY_TIMEOUT_MS


def test_hybrid_search_forwards_channel_scope():
    repo = FakeRepo()
    repo.since_rows = []
    svc = make_service(repo)

    svc.hybrid_search(CTX, q_vec=[1.0], k=3, limit=3, channel_id="c1")

    call = next(c for c in repo.calls if c[0] == "hybrid_search")
    assert call[5] == "c1"


# ── post_agent_answer: agent-authored answer + EMITTED provenance (K-013) ───────

CTX_AGENT = CallContext(ws="test", actor="bot1")


def _agent_svc(repo, *, now=1000):
    repo.threads.add("t1")
    repo.heads.add("t1")  # realistic path: trigger message is the HEAD → subsequent
    repo.agents.add("bot1")
    return make_service(repo, now=now)


def test_post_agent_answer_posts_as_agent_with_role_assistant_and_seeds():
    repo = FakeRepo()
    svc = _agent_svc(repo)

    out = svc.post_agent_answer(
        CTX_AGENT, thread_id="t1", text="the answer",
        seeds=[("s1", 0.1), ("s2", 0.2)],
    )

    assert out["role"] == "assistant"       # derived from the Agent actor label
    assert out["authorId"] == "bot1"
    assert out["text"] == "the answer"
    assert out["seeds"] == [("s1", 0.1), ("s2", 0.2)]
    call = next(c for c in repo.calls if c[0] == "post_agent_answer")
    assert call[1]["role"] == "assistant"
    assert call[1]["author_id"] == "bot1"
    assert call[1]["seeds"] == [("s1", 0.1), ("s2", 0.2)]


def test_post_agent_answer_missing_thread_errors():
    repo = FakeRepo()
    repo.agents.add("bot1")
    svc = make_service(repo)

    with pytest.raises(ThreadNotFoundError):
        svc.post_agent_answer(CTX_AGENT, thread_id="nope", text="hi", seeds=[])


def test_post_agent_answer_unknown_actor_errors():
    repo = FakeRepo()
    repo.threads.add("t1")
    repo.heads.add("t1")
    svc = make_service(repo)  # bot1 not registered

    with pytest.raises(UnknownActorError):
        svc.post_agent_answer(CTX_AGENT, thread_id="t1", text="hi", seeds=[])
    assert not any(c[0].startswith("post_agent") for c in repo.calls)


def test_post_agent_answer_validates_mentions():
    repo = FakeRepo()
    svc = _agent_svc(repo)

    with pytest.raises(UnknownMemberError):
        svc.post_agent_answer(
            CTX_AGENT, thread_id="t1", text="hi", mentions=["ghost"], seeds=[]
        )


def test_post_agent_answer_retries_as_first_when_no_tail():
    repo = FakeRepo()
    svc = _agent_svc(repo)
    repo.agent_subseq_status = [None]  # subsequent path: no TAIL → dispatch to first

    out = svc.post_agent_answer(CTX_AGENT, thread_id="t1", text="hi", seeds=[])

    assert out["role"] == "assistant"
    names = [c[0] for c in repo.calls if c[0].startswith("post_agent")]
    assert names == ["post_agent_answer", "post_agent_answer_first"]


def test_post_agent_answer_dup_msg_is_idempotent_success():
    repo = FakeRepo()
    svc = _agent_svc(repo)
    repo.agent_subseq_status = [DUP]

    out = svc.post_agent_answer(CTX_AGENT, thread_id="t1", text="hi", seeds=[])
    assert out["msgId"]  # returned cleanly, no raise
