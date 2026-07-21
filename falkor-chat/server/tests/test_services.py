"""Unit tests for the service layer.

Services own the invariants (write-variant dispatch, mention validation, RO/RW
read dispatch, `cursorId` construction, id/clock generation). They are tested
against a fake repository so the logic is pinned without a live database.
"""

from __future__ import annotations

import itertools
import json

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
    WorkflowRunNotFoundError,
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
        # §11 workflow state
        self.published: list[dict] = []       # publish_def kwargs, in order
        self.materialized: list[dict] = []     # materialize_snapshot kwargs, in order
        self.defs: dict[tuple, dict] = {}      # (key, version) -> def subgraph/meta
        self.snapshots: dict[tuple, dict] = {}  # (key, version) -> snapshot subgraph/meta
        # §12 workflow-run state
        self.messages: dict[str, dict] = {}    # msgId -> message (for get_message)
        self.started_runs: list[dict] = []     # start_run kwargs, in order
        self.start_run_result = _UNSET        # override to None to simulate a miss
        self.runs: dict[str, dict] = {}        # runId -> run state (for get_run)
        self.step_runs: dict[str, list] = {}   # runId -> step-run trail
        self.trace: dict[str, list] = {}       # runId -> trace events
        self.waiting_runs: dict[str, dict] = {}  # threadId -> waiting run (resume lookup)

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

    # ── §11 workflow defs (reference) + snapshots (workspace) ────────────────────

    def publish_def(self, *, key, version, name, kind, start_key, steps, transitions):
        self.published.append({
            "key": key, "version": version, "name": name, "kind": kind,
            "start_key": start_key, "steps": steps, "transitions": transitions,
        })
        return {
            "key": key, "version": version,
            "stepCount": len(steps), "transitionCount": len(transitions),
        }

    def read_def_subgraph(self, *, key, version):
        self.calls.append(("read_def_subgraph", key, version))
        return self.defs.get((key, version))

    def get_def(self, *, key, version=None):
        self.calls.append(("get_def", key, version))
        return self.defs.get((key, version))

    def list_defs(self, *, limit=50):
        self.calls.append(("list_defs", limit))
        return list(self.defs.values())

    def materialize_snapshot(self, ws, *, key, version, name, kind, start_key,
                             steps, transitions):
        self.materialized.append({
            "ws": ws, "key": key, "version": version, "name": name, "kind": kind,
            "start_key": start_key, "steps": steps, "transitions": transitions,
        })
        return {
            "key": key, "version": version,
            "stepCount": len(steps), "transitionCount": len(transitions),
        }

    def get_snapshot(self, ws, *, key, version):
        self.calls.append(("get_snapshot", ws, key, version))
        return self.snapshots.get((key, version))

    def list_snapshots(self, ws, *, limit=50):
        self.calls.append(("list_snapshots", ws, limit))
        return list(self.snapshots.values())

    # ── §12 workflow runs ────────────────────────────────────────────────────

    def get_message(self, ws, *, msg_id):
        self.calls.append(("get_message", ws, msg_id))
        return self.messages.get(msg_id)

    def start_run(self, ws, *, run_id, def_key, def_version, started_at,
                  trigger_msg_id, ctx, trace, max_steps):
        self.started_runs.append({
            "ws": ws, "run_id": run_id, "def_key": def_key,
            "def_version": def_version, "started_at": started_at,
            "trigger_msg_id": trigger_msg_id, "ctx": ctx, "trace": trace,
            "max_steps": max_steps,
        })
        if self.start_run_result is _UNSET:
            return {"runId": run_id, "startKey": "intake", "status": "running",
                    "stepCount": 0}
        return self.start_run_result

    def get_run(self, ws, *, run_id):
        self.calls.append(("get_run", ws, run_id))
        return self.runs.get(run_id)

    def read_step_runs(self, ws, *, run_id):
        self.calls.append(("read_step_runs", ws, run_id))
        return self.step_runs.get(run_id, [])

    def read_trace(self, ws, *, run_id):
        self.calls.append(("read_trace", ws, run_id))
        return self.trace.get(run_id, [])

    def find_waiting_run_for_thread(self, ws, *, thread_id):
        self.calls.append(("find_waiting_run_for_thread", ws, thread_id))
        return self.waiting_runs.get(thread_id)


_UNSET = object()


class StubExecutor:
    """Records `run`/`resume` calls and returns scripted statuses (U5 tests the
    service orchestration in isolation; the real engine is covered in U4)."""

    def __init__(self, *, step_budget=12, run_status="waiting", resume_status="done"):
        self.step_budget = step_budget
        self._run_status = run_status
        self._resume_status = resume_status
        self.run_calls: list[str] = []
        self.resume_calls: list[str] = []

    def run(self, ctx, *, run_id):
        self.run_calls.append(run_id)
        return self._run_status

    def resume(self, ctx, *, run_id):
        self.resume_calls.append(run_id)
        return self._resume_status


def make_service(repo, *, now=1000, executor=None):
    ids = (f"id{n}" for n in itertools.count(1))
    return Services(repo, clock=lambda: now, id_gen=lambda: next(ids),
                    executor=executor)


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


# ── §11 Workflow definitions & snapshots (M3 Slice 1) ───────────────────────────
#
# publish_workflow_def validates the spec BEFORE any write (plan §B5): unknown
# kind/step-type, duplicate step keys, a start marker that isn't exactly one
# declared step, and dangling transition endpoints all raise WorkflowDefSpecError
# and write nothing. A step marks itself the start with `start: True` (exactly one
# required — that step's key becomes the repo `start_key`). config/guard are
# serialized to opaque strings. materialize_def is two-phase: read the def from
# `reference`, write the snapshot into ctx.ws. Def authoring/reading is global
# (repo omits ws); only materialization + snapshot reads consume ctx.ws.

from falkorchat.services import (  # noqa: E402
    WorkflowDefNotFoundError,
    WorkflowDefSpecError,
)

VALID_STEPS = [
    # `waitsForHuman` is mandatory on a `human`/`wait` step (K-024 U2 publish invariant).
    {"key": "start", "type": "human", "config": {"a": 1, "waitsForHuman": True},
     "start": True},
    {"key": "review", "type": "decision", "config": "raw-string"},
    {"key": "done", "type": "message"},  # no config → serializes to ""
]
VALID_TRANSITIONS = [
    {"from": "start", "to": "review", "on": "submitted", "order": 0},  # no guard → ""
    {"from": "review", "to": "done", "on": "approved",
     "guard": {"expr": "x>0"}, "order": 0},
]


def _publish(svc, repo, *, kind="process", steps=None, transitions=None):
    return svc.publish_workflow_def(
        CTX, key="onboarding", version="1", name="Onboarding", kind=kind,
        steps=steps if steps is not None else VALID_STEPS,
        transitions=transitions if transitions is not None else VALID_TRANSITIONS,
    )


def test_publish_workflow_def_derives_start_and_serializes_config_and_guard():
    repo = FakeRepo()
    svc = make_service(repo)

    _publish(svc, repo)

    assert len(repo.published) == 1
    pub = repo.published[0]
    assert pub["start_key"] == "start"                 # derived from start:True
    # steps handed to the repo carry only {key,type,config}; config is a string
    by_key = {s["key"]: s for s in pub["steps"]}
    # dict → compact JSON, stable key order
    assert by_key["start"]["config"] == '{"a":1,"waitsForHuman":true}'
    assert by_key["review"]["config"] == "raw-string"  # str passthrough
    assert by_key["done"]["config"] == ""              # missing → ""
    assert "start" not in by_key["start"]              # start flag stripped
    # transition guards serialized to strings
    trs = {(t["from"], t["to"]): t for t in pub["transitions"]}
    assert trs[("start", "review")]["guard"] == ""         # missing → ""
    assert trs[("review", "done")]["guard"] == '{"expr":"x>0"}'


def test_publish_workflow_def_returns_repo_result():
    repo = FakeRepo()
    svc = make_service(repo)

    out = _publish(svc, repo)

    assert out["key"] == "onboarding"
    assert out["version"] == "1"
    assert out["stepCount"] == 3
    assert out["transitionCount"] == 2


def test_publish_workflow_def_invalid_kind_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)

    with pytest.raises(WorkflowDefSpecError):
        _publish(svc, repo, kind="chatbot")  # not conversation|process

    assert repo.published == []


def test_publish_workflow_def_conversation_kind_allowed():
    repo = FakeRepo()
    svc = make_service(repo)

    _publish(svc, repo, kind="conversation")

    assert repo.published[0]["kind"] == "conversation"


def test_publish_workflow_def_invalid_step_type_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    bad = [
        {"key": "start", "type": "bogus", "start": True},
        {"key": "done", "type": "message"},
    ]

    with pytest.raises(WorkflowDefSpecError):
        _publish(svc, repo, steps=bad, transitions=[])

    assert repo.published == []


def test_publish_workflow_def_duplicate_step_key_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    dup = [
        {"key": "start", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "start", "type": "message"},
    ]

    # `match=` is load-bearing (B-2): the K-024 U2 invariants were added AFTER this
    # check, so a type-only assertion would pass even if the duplicate-key check were
    # deleted and some later invariant raised instead. Asserting the message keeps this
    # test a real regression net for the rule it names.
    with pytest.raises(WorkflowDefSpecError, match="duplicate step key"):
        _publish(svc, repo, steps=dup, transitions=[])

    assert repo.published == []


def test_publish_workflow_def_no_start_step_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    no_start = [
        {"key": "a", "type": "human", "config": {"waitsForHuman": True}},
        {"key": "b", "type": "message"},
    ]

    with pytest.raises(WorkflowDefSpecError, match="exactly one start step"):
        _publish(svc, repo, steps=no_start, transitions=[])

    assert repo.published == []


def test_publish_workflow_def_multiple_start_steps_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    two_starts = [
        {"key": "a", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "b", "type": "message", "start": True},
    ]

    with pytest.raises(WorkflowDefSpecError, match="exactly one start step"):
        _publish(svc, repo, steps=two_starts, transitions=[])

    assert repo.published == []


def test_publish_workflow_def_dangling_transition_from_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    bad_tr = [{"from": "ghost", "to": "done", "on": "x", "order": 0}]
    steps = [
        {"key": "start", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "done", "type": "message"},
    ]

    with pytest.raises(WorkflowDefSpecError, match="from 'ghost' is not a declared"):
        _publish(svc, repo, steps=steps, transitions=bad_tr)

    assert repo.published == []


def test_publish_workflow_def_dangling_transition_to_raises_nothing_written():
    repo = FakeRepo()
    svc = make_service(repo)
    bad_tr = [{"from": "start", "to": "ghost", "on": "x", "order": 0}]
    steps = [
        {"key": "start", "type": "human", "config": {"waitsForHuman": True},
         "start": True},
        {"key": "done", "type": "message"},
    ]

    with pytest.raises(WorkflowDefSpecError, match="to 'ghost' is not a declared"):
        _publish(svc, repo, steps=steps, transitions=bad_tr)

    assert repo.published == []


def test_materialize_def_two_phase_reads_reference_then_writes_workspace():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.defs[("onboarding", "1")] = {
        "name": "Onboarding", "kind": "process", "start_key": "start",
        "steps": [{"key": "start", "type": "human", "config": ""}],
        "transitions": [],
    }

    out = svc.materialize_def(CTX, key="onboarding", version="1")

    assert len(repo.materialized) == 1
    mat = repo.materialized[0]
    assert mat["ws"] == "test"                 # writes into ctx.ws
    assert mat["key"] == "onboarding"
    assert mat["name"] == "Onboarding"
    assert mat["start_key"] == "start"
    assert out["key"] == "onboarding"


def test_materialize_def_not_found_raises_nothing_materialized():
    repo = FakeRepo()
    svc = make_service(repo)  # repo.defs is empty

    with pytest.raises(WorkflowDefNotFoundError):
        svc.materialize_def(CTX, key="ghost", version="1")

    assert repo.materialized == []


def test_get_workflow_def_passthrough_is_global_no_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.defs[("onboarding", "1")] = {
        "key": "onboarding", "version": "1", "name": "Onboarding", "kind": "process",
    }

    got = svc.get_workflow_def(CTX, key="onboarding", version="1")

    assert got["name"] == "Onboarding"
    assert ("get_def", "onboarding", "1") in repo.calls


def test_list_workflow_defs_passthrough():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.defs[("a", "1")] = {"key": "a", "version": "1", "name": "A", "kind": "process"}

    out = svc.list_workflow_defs(CTX)

    assert out and out[0]["key"] == "a"


def test_get_snapshot_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.snapshots[("a", "1")] = {
        "name": "A", "kind": "process", "start_key": "s", "steps": [], "transitions": [],
    }

    got = svc.get_snapshot(CTX, key="a", version="1")

    assert got["name"] == "A"
    assert ("get_snapshot", "test", "a", "1") in repo.calls


def test_list_snapshots_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.snapshots[("a", "1")] = {"key": "a", "version": "1", "name": "A", "kind": "process"}

    out = svc.list_snapshots(CTX)

    assert out and out[0]["key"] == "a"
    assert ("list_snapshots", "test", 50) in repo.calls


# ── §12 workflow-run orchestration (U5) ─────────────────────────────────────────
#
# The service mints the run id + start timestamp (server clock), resolves the
# trigger message's thread into the run `ctx` (so a suspend can denorm it for the
# resume lookup), starts the run via the repository, then hands off to the injected
# executor. Reads are thin, ctx.ws-scoped pass-throughs. The engine itself is U4.


def test_start_workflow_run_mints_run_seeds_thread_ctx_and_drives():
    repo = FakeRepo()
    repo.messages["trig1"] = {"msgId": "trig1", "threadId": "t1"}
    ex = StubExecutor(step_budget=12, run_status="waiting")
    svc = make_service(repo, executor=ex)

    out = svc.start_workflow_run(
        CTX, def_key="triage", version="1", trigger_msg_id="trig1", trace=True
    )

    # minted a run id + started it at the snapshot, using the executor's budget
    started = repo.started_runs[0]
    assert started["def_key"] == "triage"
    assert started["def_version"] == "1"
    assert started["trigger_msg_id"] == "trig1"
    assert started["trace"] is True
    assert started["max_steps"] == 12
    # the trigger's thread is seeded into ctx for the resume denorm (§2.4)
    assert json.loads(started["ctx"]) == {"threadId": "t1"}
    # then drove the engine and returned its status
    assert ex.run_calls == [out["runId"]]
    assert out["status"] == "waiting"


def test_start_workflow_run_missing_anchor_raises_nothing_driven():
    repo = FakeRepo()
    repo.messages["trig1"] = {"msgId": "trig1", "threadId": "t1"}
    repo.start_run_result = None  # snapshot/trigger anchor missed
    ex = StubExecutor()
    svc = make_service(repo, executor=ex)

    with pytest.raises(WorkflowRunNotFoundError):
        svc.start_workflow_run(
            CTX, def_key="ghost", version="1", trigger_msg_id="trig1"
        )
    assert ex.run_calls == []  # never handed to the engine


def test_start_workflow_run_without_executor_raises():
    repo = FakeRepo()
    svc = make_service(repo)  # no executor wired
    with pytest.raises(RuntimeError):
        svc.start_workflow_run(
            CTX, def_key="triage", version="1", trigger_msg_id="trig1"
        )


def test_resume_workflow_run_delegates_to_executor():
    repo = FakeRepo()
    ex = StubExecutor(resume_status="done")
    svc = make_service(repo, executor=ex)

    out = svc.resume_workflow_run(CTX, run_id="r1")

    assert ex.resume_calls == ["r1"]
    assert out == {"runId": "r1", "status": "done"}


def test_get_workflow_run_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.runs["r1"] = {"runId": "r1", "status": "running", "atStepKey": "intake"}

    got = svc.get_workflow_run(CTX, run_id="r1")

    assert got["status"] == "running"
    assert ("get_run", "test", "r1") in repo.calls


def test_read_workflow_step_runs_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.step_runs["r1"] = [{"stepRunId": "sr1", "stepKey": "intake"}]

    out = svc.read_workflow_step_runs(CTX, run_id="r1")

    assert out and out[0]["stepKey"] == "intake"
    assert ("read_step_runs", "test", "r1") in repo.calls


def test_read_workflow_trace_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.trace["r1"] = [{"traceId": "te1", "kind": "guard_judgment"}]

    out = svc.read_workflow_trace(CTX, run_id="r1")

    assert out and out[0]["kind"] == "guard_judgment"
    assert ("read_trace", "test", "r1") in repo.calls


def test_find_waiting_run_for_thread_passthrough_uses_ctx_ws():
    repo = FakeRepo()
    svc = make_service(repo)
    repo.waiting_runs["t1"] = {"runId": "r1", "status": "waiting"}

    got = svc.find_waiting_run_for_thread(CTX, thread_id="t1")

    assert got["runId"] == "r1"
    assert ("find_waiting_run_for_thread", "test", "t1") in repo.calls


def test_find_waiting_run_for_thread_returns_none_when_nothing_parked():
    repo = FakeRepo()
    svc = make_service(repo)
    assert svc.find_waiting_run_for_thread(CTX, thread_id="t1") is None


def test_agent_step_type_is_accepted_by_publish_validation():
    # the LLM-native node kind (§3): STEP_TYPES gains 'agent' so a triage def
    # (type:'agent' steps) validates and publishes
    repo = FakeRepo()
    svc = make_service(repo)

    svc.publish_workflow_def(
        CTX, key="triage", version="1", name="Triage", kind="conversation",
        steps=[{"key": "intake", "type": "agent", "start": True,
                "config": {"waitsForHuman": True}}],
        transitions=[],
    )

    assert repo.published[0]["key"] == "triage"
    assert repo.published[0]["steps"][0]["type"] == "agent"
