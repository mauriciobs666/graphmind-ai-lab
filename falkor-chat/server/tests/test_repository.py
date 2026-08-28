"""Integration tests for the repository layer against a live `ws:test` graph.

Each test wraps one repository method 1:1 with a verified `QUERIES.md` query.
A few structural regression tests probe the graph directly (`_probe`) — the
write-path defects they pin (NEXT self-loops, duplicate HEADs) are invisible
through the public read methods by design. Two tests take no fixture at all and
drive `Repository._read_structure` (a `@staticmethod`) over a `_FakeGraph`: the
graph states they pin cannot be produced without depending on publish semantics
K-034 is chartered to change.
"""

from __future__ import annotations

import json

import pytest
from redis.exceptions import ResponseError

from falkorchat import db
from falkorchat.repository import MemberIdCollisionError, Repository


def _probe(conn, cypher: str):
    """Raw structural read against ws:test (test-only; app Cypher stays in repository.py)."""
    return db.workspace_graph(conn, "test").ro_query(cypher).result_set


def _add_to_channel(conn, *, member_id: str, channel_id: str):
    """Raw MEMBER_OF write (test-only; no repository method exists yet — QUERIES.md
    §2 "Add user to channel" is a documented, verified query but not yet wrapped
    by a repository method, same gap `seed_demo.sh` fills with raw Cypher). Anchors
    on `mem` via the same label-agnostic `userId OR agentId` pattern
    `advance_cursor` already uses, so one helper covers both User and Agent members.
    """
    db.workspace_graph(conn, "test").query(
        "MATCH (mem) WHERE mem.userId = $memberId OR mem.agentId = $memberId "
        "MATCH (c:Channel {channelId: $channelId}) "
        "MERGE (mem)-[:MEMBER_OF]->(c)",
        {"memberId": member_id, "channelId": channel_id},
    )

# ── §3 Channels ────────────────────────────────────────────────────────────────


def test_create_channel_then_list_returns_it(repo):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)

    channels = repo.list_channels("test")

    assert [c["channelId"] for c in channels] == ["c1"]
    assert channels[0]["name"] == "general"
    assert channels[0]["createdAt"] == 100


def test_list_channels_empty_when_none(repo):
    assert repo.list_channels("test") == []


def test_list_channels_orders_by_createdAt_desc(repo):
    repo.create_channel("test", channel_id="c1", name="first", created_at=100)
    repo.create_channel("test", channel_id="c2", name="second", created_at=200)

    channels = repo.list_channels("test")

    assert [c["channelId"] for c in channels] == ["c2", "c1"]


# ── §3 Threads ─────────────────────────────────────────────────────────────────


def test_create_thread_then_list_returns_it(repo):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread(
        "test", channel_id="c1", thread_id="t1", title="hello", created_at=110
    )

    threads = repo.list_threads("test", channel_id="c1")

    assert [t["threadId"] for t in threads] == ["t1"]
    assert threads[0]["title"] == "hello"
    assert threads[0]["updatedAt"] == 110


def test_list_threads_orders_by_updatedAt_desc(repo):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="old", created_at=110)
    repo.create_thread("test", channel_id="c1", thread_id="t2", title="new", created_at=120)

    threads = repo.list_threads("test", channel_id="c1")

    assert [t["threadId"] for t in threads] == ["t2", "t1"]


def test_create_thread_missing_channel_raises_not_silent_noop(repo):
    # the service pre-validates the channel; this raise is the repository
    # tripwire — a missing anchor must never be a silent no-op (K-007 §2.5)
    with pytest.raises(RuntimeError):
        repo.create_thread(
            "test", channel_id="ghost", thread_id="t1", title="x", created_at=110
        )

    assert repo.thread_exists("test", thread_id="t1") is False


def test_thread_has_head_false_before_first_message(repo):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x", created_at=110)

    assert repo.thread_has_head("test", thread_id="t1") is False


# ── §2/§7 guarded member ensures (DEF-1: cross-label id namespace) ─────────────


def test_ensure_user_fresh_creates_then_reensure_is_quiet_noop(repo, conn):
    repo.ensure_user("test", user_id="u1", display_name="Alice", email="a@x.io")
    repo.ensure_user("test", user_id="u1", display_name="Changed", email="c@x.io")

    # exactly one node; re-ensure never updates properties (old ON CREATE-only behavior)
    rows = _probe(
        conn, "MATCH (u:User {userId:'u1'}) RETURN count(u), collect(u.displayName)"
    )
    assert rows == [[1, ["Alice"]]]


def test_ensure_agent_fresh_creates_then_reensure_is_quiet_noop(repo, conn):
    repo.ensure_agent("test", agent_id="a1", name="Bot", model="m-1", created_at=100)
    repo.ensure_agent("test", agent_id="a1", name="Renamed", model="m-2", created_at=200)

    rows = _probe(conn, "MATCH (a:Agent {agentId:'a1'}) RETURN count(a), collect(a.name)")
    assert rows == [[1, ["Bot"]]]


def test_ensure_user_refuses_id_held_by_agent_nothing_written(repo, conn):
    """DEF-1 repro direction: a User ensure with an Agent's id must refuse —
    the old MERGE silently created a shadow User that eclipsed the Agent in
    every coalesce(u, a) lookup."""
    repo.ensure_agent("test", agent_id="qabot", name="Bot")

    with pytest.raises(MemberIdCollisionError, match="held by an Agent"):
        repo.ensure_user("test", user_id="qabot", display_name="Shadow")

    # nothing written — and the Agent is still what the id resolves to
    [[shadow]] = _probe(conn, "OPTIONAL MATCH (u:User {userId:'qabot'}) RETURN u IS NOT NULL")
    assert shadow is False
    assert repo.resolve_member_kinds("test", ids=["qabot"]) == {"qabot": "Agent"}


def test_ensure_agent_refuses_id_held_by_user_nothing_written(repo, conn):
    repo.ensure_user("test", user_id="u1", display_name="Alice")

    with pytest.raises(MemberIdCollisionError, match="held by a User"):
        repo.ensure_agent("test", agent_id="u1", name="Impostor")

    [[shadow]] = _probe(conn, "OPTIONAL MATCH (a:Agent {agentId:'u1'}) RETURN a IS NOT NULL")
    assert shadow is False
    assert repo.resolve_member_kinds("test", ids=["u1"]) == {"u1": "User"}


def test_ensure_refuses_pre_guard_corruption_with_alarm(repo, conn):
    """existed AND collided — both labels hold the id (pre-guard shadow state).
    Both ensures must raise the distinguishable corruption alarm."""
    # seed the corruption directly: the guarded ensures refuse to create it
    db.workspace_graph(conn, "test").query(
        "CREATE (:User {userId:'x1'}), (:Agent {agentId:'x1'})"
    )

    with pytest.raises(MemberIdCollisionError, match="corrupted"):
        repo.ensure_user("test", user_id="x1")
    with pytest.raises(MemberIdCollisionError, match="corrupted"):
        repo.ensure_agent("test", agent_id="x1")


# ── §4 Messages ────────────────────────────────────────────────────────────────


def _seed_thread(repo, *, with_author="u1"):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x", created_at=110)
    if with_author:
        repo.ensure_user("test", user_id=with_author, display_name="Alice")


def test_post_first_message_is_readable(repo):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="hello", role="user", created_at=120,
    )

    msgs = repo.read_thread("test", thread_id="t1")

    assert [m["msgId"] for m in msgs] == ["m1"]
    assert msgs[0]["text"] == "hello"
    assert msgs[0]["role"] == "user"
    assert msgs[0]["authorId"] == "u1"


def test_first_message_sets_head_and_updates_thread(repo):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="hi", role="user", created_at=120,
    )
    assert repo.thread_has_head("test", thread_id="t1") is True


def test_post_first_message_unknown_author_reports_status_nothing_written(repo):
    _seed_thread(repo, with_author=None)  # thread exists, author does not

    st = repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="ghost",
        text="hello", role="user", created_at=120,
    )

    assert st is not None
    assert st.written is False
    assert st.author_found is False
    assert repo.read_thread("test", thread_id="t1") == []


def test_post_subsequent_message_unknown_author_reports_status_nothing_written(repo):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="one", role="user", created_at=120,
    )

    st = repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m2", author_id="ghost",
        text="two", role="user", created_at=130,
    )

    assert st is not None
    assert st.written is False
    assert st.author_found is False
    assert [m["msgId"] for m in repo.read_thread("test", thread_id="t1")] == ["m1"]


def test_read_thread_returns_tools_used_when_stamped(repo, conn):
    # K-056: `read_thread` must surface the `toolsUsed` audit property
    # `link_step_emission` stamps (pure audit trail — `executor._assemble_messages`
    # does not read it back into a replayed prompt; that path was reverted).
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="hello", role="user", created_at=120,
    )
    db.workspace_graph(conn, "test").query(
        "MATCH (m:Message {msgId:'m1'}) SET m.toolsUsed = $toolsUsed",
        {"toolsUsed": ["lookup_product_fact"]},
    )

    msgs = repo.read_thread("test", thread_id="t1")

    assert msgs[0]["toolsUsed"] == ["lookup_product_fact"]


def test_read_thread_defaults_tools_used_to_empty_list_when_absent(repo):
    # A Message written before this change (no `toolsUsed` property at all) must
    # degrade to `[]`, not `None` — `_assemble_messages` treats both as falsy, but the
    # repository contract should be explicit either way.
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="hello", role="user", created_at=120,
    )

    msgs = repo.read_thread("test", thread_id="t1")

    assert msgs[0]["toolsUsed"] == []


def test_subsequent_message_appends_in_order(repo):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="one", role="user", created_at=120,
    )
    repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m2", author_id="u1",
        text="two", role="user", created_at=130,
    )

    msgs = repo.read_thread("test", thread_id="t1")

    assert [m["msgId"] for m in msgs] == ["m1", "m2"]
    assert [m["text"] for m in msgs] == ["one", "two"]


# ── §4 v2 write-path guards (K-007 defect regressions) ──────────────────────────


def test_replay_of_subsequent_write_is_structural_noop(repo, conn):
    """Defect A regression: a retried subsequent write must not corrupt the chain."""
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="one", role="user", created_at=120,
    )
    repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m2", author_id="u1",
        text="two", role="user", created_at=130,
    )

    st = repo.post_subsequent_message(  # exact replay (client-timeout retry)
        "test", thread_id="t1", msg_id="m2", author_id="u1",
        text="two", role="user", created_at=130,
    )

    # chain intact: one NEXT edge, no self-loop, one POSTED_BY from m2
    [[next_count, self_loops]] = _probe(
        conn,
        "MATCH (a:Message)-[r:NEXT]->(b:Message) "
        "RETURN count(r), sum(CASE WHEN a.msgId = b.msgId THEN 1 ELSE 0 END)",
    )
    [[posted_by]] = _probe(
        conn, "MATCH (:Message {msgId:'m2'})-[r:POSTED_BY]->() RETURN count(r)"
    )
    assert (next_count, self_loops, posted_by) == (1, 0, 1)
    assert [m["msgId"] for m in repo.read_thread("test", thread_id="t1")] == ["m1", "m2"]
    assert st is not None
    assert st.written is False
    assert st.dup_msg is True  # idempotent success signal


def test_replay_of_first_write_reports_dup_and_had_head(repo, conn):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="one", role="user", created_at=120,
    )

    st = repo.post_first_message(  # exact replay
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="one", role="user", created_at=120,
    )

    [[head_count]] = _probe(conn, "MATCH (:Thread)-[r:HEAD]->() RETURN count(r)")
    assert head_count == 1
    assert st.written is False
    assert st.dup_msg is True
    assert st.had_head is True


def test_first_post_on_headed_thread_refuses_two_heads(repo, conn):
    """Defect B regression: a lost first-post race must not create a second HEAD."""
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="one", role="user", created_at=120,
    )

    st = repo.post_first_message(  # fresh msgId — the racing loser's write
        "test", thread_id="t1", msg_id="m9", author_id="u1",
        text="racer", role="user", created_at=121,
    )

    [[head_count]] = _probe(conn, "MATCH (:Thread)-[r:HEAD]->() RETURN count(r)")
    assert head_count == 1
    assert repo.get_message("test", msg_id="m9") is None  # nothing created
    assert st.written is False
    assert st.had_head is True
    assert st.dup_msg is False


def test_subsequent_on_tailless_thread_returns_none(repo):
    _seed_thread(repo)  # thread exists but has no messages → no TAIL anchor

    st = repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="orphan", role="user", created_at=120,
    )

    assert st is None  # dispatch signal: retry as first-post
    assert repo.read_thread("test", thread_id="t1") == []


def test_agent_author_subsequent_write_commits(repo, conn):
    """K-007 item 1 regression: Agents (agentId, no userId) can author messages."""
    _seed_thread(repo)
    repo.ensure_agent("test", agent_id="a1", name="Bot")
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="question", role="user", created_at=120,
    )

    st = repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m2", author_id="a1",
        text="answer", role="assistant", created_at=130,
    )

    assert st.written is True
    assert st.author_found is True
    [[author_labels, author_id]] = _probe(
        conn,
        "MATCH (:Message {msgId:'m2'})-[:POSTED_BY]->(a) "
        "RETURN labels(a), coalesce(a.userId, a.agentId)",
    )
    assert (author_labels, author_id) == (["Agent"], "a1")
    msgs = repo.read_thread("test", thread_id="t1")
    assert msgs[1]["authorId"] == "a1"
    assert msgs[1]["role"] == "assistant"  # stored as passed (service derives it)


def test_thread_id_stamped_by_both_write_paths(repo, conn):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="one", role="user", created_at=120,
    )
    repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m2", author_id="u1",
        text="two", role="user", created_at=130,
    )

    rows = _probe(
        conn, "MATCH (m:Message) RETURN m.msgId, m.threadId ORDER BY m.createdAt"
    )
    assert rows == [["m1", "t1"], ["m2", "t1"]]


# ── §9.1 Read a thread since a cursor/timestamp (mention-aware) ─────────────────


def test_read_thread_since_flags_mention_of_reader(repo):
    _seed_thread(repo)
    repo.ensure_user("test", user_id="u2", display_name="Bob")
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="hey @bob", role="user", created_at=120, mentions=["u2"],
    )

    rows = repo.read_thread_since("test", thread_id="t1", me_id="u2", since=0)

    assert [r["msgId"] for r in rows] == ["m1"]
    assert rows[0]["isMention"] is True


def test_read_thread_since_no_mention_is_false(repo):
    _seed_thread(repo)
    repo.ensure_user("test", user_id="u2", display_name="Bob")
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="plain", role="user", created_at=120,  # no mentions → no-op block
    )

    rows = repo.read_thread_since("test", thread_id="t1", me_id="u2", since=0)

    assert rows[0]["isMention"] is False


def test_read_thread_since_is_chronological_with_mention_flag(repo):
    _seed_thread(repo)
    repo.ensure_user("test", user_id="u2", display_name="Bob")
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="plain-earlier", role="user", created_at=120,
    )
    repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m2", author_id="u1",
        text="mentions-bob-later", role="user", created_at=130, mentions=["u2"],
    )

    rows = repo.read_thread_since("test", thread_id="t1", me_id="u2", since=0)

    # chronological order — the cursor-pagination invariant; the mention is
    # flagged, not resorted (a mention-first sort + LIMIT loses messages)
    assert [r["msgId"] for r in rows] == ["m1", "m2"]
    assert [r["isMention"] for r in rows] == [False, True]


def test_read_thread_since_limit_returns_earliest_page(repo):
    _seed_thread(repo)
    repo.ensure_user("test", user_id="u2", display_name="Bob")
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="one", role="user", created_at=120,
    )
    repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m2", author_id="u1",
        text="two", role="user", created_at=130,
    )
    repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m3", author_id="u1",
        text="mentions bob", role="user", created_at=140, mentions=["u2"],
    )

    rows = repo.read_thread_since("test", thread_id="t1", me_id="u2", since=0, limit=2)

    # a truncated page must be the earliest messages so the caller can resume
    # from the last returned createdAt without skipping anything
    assert [r["msgId"] for r in rows] == ["m1", "m2"]


def test_read_thread_since_filters_by_since(repo):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="old", role="user", created_at=120,
    )
    repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m2", author_id="u1",
        text="new", role="user", created_at=130,
    )

    rows = repo.read_thread_since("test", thread_id="t1", me_id="u1", since=125)

    assert [r["msgId"] for r in rows] == ["m2"]


def test_mentions_dedup_and_skip_unknown(repo):
    _seed_thread(repo)
    repo.ensure_user("test", user_id="u2", display_name="Bob")
    # duplicate u2 and an unknown 'nope' — dedup to one edge, unknown skipped, no error
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="x", role="user", created_at=120, mentions=["u2", "u2", "nope"],
    )

    rows = repo.read_thread_since("test", thread_id="t1", me_id="u2", since=0)

    assert rows[0]["isMention"] is True  # single edge is enough; no crash on 'nope'


# ── §9.2 Read workspace-wide since a timestamp ─────────────────────────────────


def test_read_ws_since_spans_threads_and_filters(repo):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.ensure_user("test", user_id="u2", display_name="Bob")
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="a", created_at=110)
    repo.create_thread("test", channel_id="c1", thread_id="t2", title="b", created_at=110)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="old", role="user", created_at=120,
    )
    repo.post_first_message(
        "test", thread_id="t2", msg_id="m2", author_id="u1",
        text="mentions bob", role="user", created_at=130, mentions=["u2"],
    )

    rows = repo.read_ws_since("test", me_id="u2", since=125)

    assert [r["msgId"] for r in rows] == ["m2"]  # m1 filtered out by since
    assert rows[0]["isMention"] is True


# ── §9.3/§9.4 Read-cursor advance (composite monotonic) & read ──────────────────


def test_get_cursor_none_when_absent(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    assert repo.get_cursor("test", cursor_id="u1:t1") is None


def test_advance_cursor_then_get_returns_pair(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.advance_cursor(
        "test", me_id="u1", thread_id="t1", cursor_id="u1:t1", now=300, now_msg_id="k2"
    )

    assert repo.get_cursor("test", cursor_id="u1:t1") == (300, "k2")


def test_advance_cursor_unknown_member_is_noop_returning_none(repo):
    # nothing seeded — the member does not exist; must not raise (was IndexError)
    assert (
        repo.advance_cursor("test", me_id="ghost", thread_id="t1",
                            cursor_id="ghost:t1", now=300, now_msg_id="k1")
        is None
    )
    assert repo.get_cursor("test", cursor_id="ghost:t1") is None


def test_advance_cursor_is_monotonic(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")

    def adv(now, msg_id):
        return repo.advance_cursor(
            "test", me_id="u1", thread_id="t1", cursor_id="u1:t1",
            now=now, now_msg_id=msg_id,
        )

    adv(300, "k2")
    adv(200, "k9")  # stale timestamp — refused

    assert repo.get_cursor("test", cursor_id="u1:t1") == (300, "k2")

    adv(400, "k4")
    assert repo.get_cursor("test", cursor_id="u1:t1") == (400, "k4")


def test_advance_cursor_composite_tie_break(repo):
    """K-007 item 4: within one millisecond the msgId breaks the tie —
    a larger id advances, a stale replay of a smaller id is refused."""
    repo.ensure_user("test", user_id="u1", display_name="Alice")

    def adv(now, msg_id):
        return repo.advance_cursor(
            "test", me_id="u1", thread_id="t1", cursor_id="u1:t1",
            now=now, now_msg_id=msg_id,
        )

    assert adv(2000, "k2") == (2000, "k2")   # create
    assert adv(2000, "k3") == (2000, "k3")   # tie, larger id → advances
    assert adv(2000, "k2") == (2000, "k3")   # tie, smaller id → stale replay refused
    assert adv(1500, "k9") == (2000, "k3")   # backward → refused
    assert adv(3000, "k4") == (3000, "k4")   # forward


def test_get_cursor_pre_k007_cursor_reads_pair_with_none_msg_id(repo, conn):
    """A cursor written before K-007 has no lastReadMsgId property — the pair
    read must surface (ts, None), and `coalesce(…, '')` in the advance guard
    covers it without any backfill."""
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    # seed legacy state directly: lastReadAt only (pre-K-007 shape)
    db.workspace_graph(conn, "test").query(
        "MATCH (mem:User {userId:'u1'}) "
        "MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId:'u1:t1'}) "
        "ON CREATE SET rc.memberId = 'u1', rc.threadId = 't1' "
        "SET rc.lastReadAt = 250"
    )

    assert repo.get_cursor("test", cursor_id="u1:t1") == (250, None)

    # composite advance still works over the legacy cursor
    got = repo.advance_cursor(
        "test", me_id="u1", thread_id="t1", cursor_id="u1:t1", now=250, now_msg_id="k1"
    )
    assert got == (250, "k1")


# ── §9.1 keyset paging (millisecond-tie page-boundary regression) ───────────────


def _seed_tied_thread(repo):
    """m1@120, m2@130, m3@130 (tie), m4@140 — repo-level explicit timestamps."""
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="one", role="user", created_at=120,
    )
    for msg_id, created_at in (("m2", 130), ("m3", 130), ("m4", 140)):
        repo.post_subsequent_message(
            "test", thread_id="t1", msg_id=msg_id, author_id="u1",
            text=msg_id, role="user", created_at=created_at,
        )


def test_keyset_paging_delivers_all_rows_across_tie_boundary(repo):
    _seed_tied_thread(repo)

    page1 = repo.read_thread_since(
        "test", thread_id="t1", me_id="u1", since=0, since_msg_id="", limit=2
    )
    assert [r["msgId"] for r in page1] == ["m1", "m2"]  # boundary lands on the tie

    last = page1[-1]
    page2 = repo.read_thread_since(
        "test", thread_id="t1", me_id="u1",
        since=last["createdAt"], since_msg_id=last["msgId"], limit=50,
    )

    # the tied sibling m3 is delivered, nothing skipped (defect item 4 regression)
    assert [r["msgId"] for r in page2] == ["m3", "m4"]


def test_plain_since_read_keeps_exclusive_timestamp_semantics(repo):
    _seed_tied_thread(repo)

    rows = repo.read_thread_since(
        "test", thread_id="t1", me_id="u1", since=130, since_msg_id=None
    )

    # explicit-since (plain `>`) excludes the whole boundary millisecond — the
    # documented OQ3 contract; lossless catch-up is the cursor path's job
    assert [r["msgId"] for r in rows] == ["m4"]


def test_since_reads_return_thread_id(repo):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="one", role="user", created_at=120,
    )

    thread_rows = repo.read_thread_since("test", thread_id="t1", me_id="u1", since=0)
    ws_rows = repo.read_ws_since("test", me_id="u1", since=0)

    assert thread_rows[0]["threadId"] == "t1"
    assert ws_rows[0]["threadId"] == "t1"


# ── §4 Get a single message ────────────────────────────────────────────────────


def test_get_message_returns_fields(repo):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="hello", role="user", created_at=120,
    )

    msg = repo.get_message("test", msg_id="m1")

    assert msg["msgId"] == "m1"
    assert msg["text"] == "hello"
    assert msg["authorId"] == "u1"
    assert msg["threadId"] == "t1"  # denormalized navigation metadata (K-007)


def test_get_message_none_when_absent(repo):
    assert repo.get_message("test", msg_id="nope") is None


# ── §14 Documents & Chunks (K-050 M5 Stage 1) ────────────────────────────────────


def _chunks(*texts: str) -> list[dict]:
    return [
        {"chunkId": f"c{i}", "text": t, "seq": i} for i, t in enumerate(texts)
    ]


def test_create_document_then_get_round_trips_full_text(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")

    status = repo.create_document(
        "test", document_id="d1", title="My Doc", text="full original text",
        source_format="text", ingested_by="u1", created_at=100,
        chunks=_chunks("full ", "original text"),
    )

    assert status.written is True
    assert status.ingestor_found is True

    doc = repo.get_document("test", document_id="d1")
    assert doc["documentId"] == "d1"
    assert doc["title"] == "My Doc"
    assert doc["text"] == "full original text"  # AC-9: byte-identical
    assert doc["sourceFormat"] == "text"
    assert doc["status"] == "processing"
    assert doc["createdAt"] == 100
    assert doc["ingestedByKind"] == "User"
    assert doc["ingestedById"] == "u1"


def test_create_document_known_user_actor_source_kind_document(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.create_document(
        "test", document_id="d1", title="t", text="x",
        source_format="text", ingested_by="u1", created_at=100,
        chunks=_chunks("x"),
    )
    assert repo.get_document("test", document_id="d1")["sourceKind"] == "document"


def test_create_document_known_agent_actor_source_kind_agent(repo):
    repo.ensure_agent("test", agent_id="bot1", name="Bot")
    repo.create_document(
        "test", document_id="d1", title="t", text="x",
        source_format="text", ingested_by="bot1", created_at=100,
        chunks=_chunks("x"),
    )
    assert repo.get_document("test", document_id="d1")["sourceKind"] == "agent"


def test_create_document_unknown_actor_nothing_written(repo):
    status = repo.create_document(
        "test", document_id="d1", title="t", text="x",
        source_format="text", ingested_by="ghost", created_at=100,
        chunks=_chunks("x"),
    )

    assert status.written is False
    assert status.ingestor_found is False
    assert repo.get_document("test", document_id="d1") is None


def test_create_document_writes_chunks_in_order_with_denormalized_documentId(repo, conn):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.create_document(
        "test", document_id="d1", title="t", text="abcdef",
        source_format="text", ingested_by="u1", created_at=100,
        chunks=_chunks("abc", "def"),
    )

    rows = _probe(
        conn,
        "MATCH (d:Document {documentId:'d1'})-[:HAS_CHUNK]->(c:Chunk) "
        "RETURN c.chunkId, c.text, c.seq, c.documentId ORDER BY c.seq",
    )
    assert rows == [
        ["c0", "abc", 0, "d1"],
        ["c1", "def", 1, "d1"],
    ]


def test_get_document_none_when_absent(repo):
    assert repo.get_document("test", document_id="nope") is None


def test_list_document_chunks_returns_chunkid_and_text_in_order(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.create_document(
        "test", document_id="d1", title="t", text="abcdef",
        source_format="text", ingested_by="u1", created_at=100,
        chunks=_chunks("abc", "def"),
    )

    rows = repo.list_document_chunks("test", document_id="d1")

    assert rows == [
        {"chunkId": "c0", "text": "abc"},
        {"chunkId": "c1", "text": "def"},
    ]


def test_list_document_chunks_empty_for_unknown_document(repo):
    assert repo.list_document_chunks("test", document_id="nope") == []


def test_create_document_is_non_idempotent_on_retry(repo, conn):
    """§2.4's deliberate posture: a retried call mints a second Document."""
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.create_document(
        "test", document_id="d1", title="t", text="x",
        source_format="text", ingested_by="u1", created_at=100,
        chunks=_chunks("x"),
    )
    repo.create_document(
        "test", document_id="d2", title="t", text="x",
        source_format="text", ingested_by="u1", created_at=101,
        chunks=[{"chunkId": "c99", "text": "x", "seq": 0}],
    )

    [[count]] = _probe(conn, "MATCH (d:Document) RETURN count(d)")
    assert count == 2


# ── §14 Document.status terminal state (K-051) ───────────────────────────────
#
# `start_document_progress` initializes the outstanding-job counter
# `_schedule_chunk_processing` uses (K-051's suggested fix); `report_document_
# job_done` decrements it and flips `Document.status` to `'ready'`/`'failed'`.
# Pinned here at the repository/Cypher altitude — background.py's own unit
# tests (test_background.py) and the REST-level integration tests
# (test_api.py) exercise the full call chain these primitives sit under.


def test_start_document_progress_then_report_all_done_reaches_ready(repo):
    _document_with_chunk(repo, document_id="d1", chunk_id="c0")

    repo.start_document_progress("test", document_id="d1", total_jobs=2)
    assert repo.get_document("test", document_id="d1")["status"] == "processing"

    repo.report_document_job_done("test", document_id="d1", success=True)
    assert repo.get_document("test", document_id="d1")["status"] == "processing"

    repo.report_document_job_done("test", document_id="d1", success=True)
    assert repo.get_document("test", document_id="d1")["status"] == "ready"


def test_report_document_job_done_one_failure_marks_failed_even_with_others_pending(repo):
    _document_with_chunk(repo, document_id="d1", chunk_id="c0")

    repo.start_document_progress("test", document_id="d1", total_jobs=2)
    repo.report_document_job_done("test", document_id="d1", success=False)

    assert repo.get_document("test", document_id="d1")["status"] == "failed"


def test_report_document_job_done_late_success_does_not_revert_failed(repo):
    """A failure's terminal write wins — a job that finishes successfully
    after the document is already 'failed' must not flip it back to
    'ready', even once the outstanding count reaches zero."""
    _document_with_chunk(repo, document_id="d1", chunk_id="c0")

    repo.start_document_progress("test", document_id="d1", total_jobs=2)
    repo.report_document_job_done("test", document_id="d1", success=False)
    repo.report_document_job_done("test", document_id="d1", success=True)

    assert repo.get_document("test", document_id="d1")["status"] == "failed"


def test_report_document_job_done_late_failure_does_not_revert_ready(repo):
    """Symmetric guard: once every job has succeeded and the document is
    'ready', a stray late failure report must not flip it back."""
    _document_with_chunk(repo, document_id="d1", chunk_id="c0")

    repo.start_document_progress("test", document_id="d1", total_jobs=1)
    repo.report_document_job_done("test", document_id="d1", success=True)
    assert repo.get_document("test", document_id="d1")["status"] == "ready"

    repo.report_document_job_done("test", document_id="d1", success=False)
    assert repo.get_document("test", document_id="d1")["status"] == "ready"


def test_start_document_progress_zero_total_jobs_flips_straight_to_ready(repo):
    """K-051 review MINOR 2: a document with nothing scheduled (`total_jobs
    == 0` — e.g. zero chunks) must not park at 'processing' forever waiting
    for a `report_document_job_done` call that will never come; there is
    nothing outstanding, so it's 'ready' the instant progress is
    (non-)started."""
    _document_with_chunk(repo, document_id="d1", chunk_id="c0")

    repo.start_document_progress("test", document_id="d1", total_jobs=0)

    assert repo.get_document("test", document_id="d1")["status"] == "ready"


def test_start_document_progress_zero_total_jobs_does_not_revert_an_already_failed_document(
    repo,
):
    """Same first-terminal-write-wins guard applies here: a zero-total start
    must not resurrect a document some other job already failed."""
    _document_with_chunk(repo, document_id="d1", chunk_id="c0")
    repo.start_document_progress("test", document_id="d1", total_jobs=1)
    repo.report_document_job_done("test", document_id="d1", success=False)

    repo.start_document_progress("test", document_id="d1", total_jobs=0)

    assert repo.get_document("test", document_id="d1")["status"] == "failed"


# ── §14.5 Entities & RELATES_TO (K-050 M5 Stage 3) ────────────────────────────


def _document_with_chunk(repo, *, document_id="d1", chunk_id="c0"):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.create_document(
        "test", document_id=document_id, title="t", text="x",
        source_format="text", ingested_by="u1", created_at=100,
        chunks=[{"chunkId": chunk_id, "text": "x", "seq": 0}],
    )


def test_create_entity_writes_all_fields(repo, conn):
    repo.create_entity(
        "test", entity_id="e1", name="Acme Corp", name_normalized="acme corp",
        type="Organization", created_at=100,
    )

    rows = _probe(
        conn,
        "MATCH (e:Entity {entityId:'e1'}) "
        "RETURN e.name, e.nameNormalized, e.type, e.createdAt",
    )
    assert rows == [["Acme Corp", "acme corp", "Organization", 100]]


def test_create_entity_returns_entity_id(repo):
    result = repo.create_entity(
        "test", entity_id="e1", name="Bob", name_normalized="bob",
        type="Person", created_at=100,
    )

    assert result["entityId"] == "e1"


def test_create_entity_always_creates_a_new_node_never_reuses(repo, conn):
    # Fusion (a future stage) never blocks creation here — two mentions with
    # the identical normalized name + type still produce two Entity nodes.
    repo.create_entity(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e2", name="Acme", name_normalized="acme",
        type="Organization", created_at=101,
    )

    [[count]] = _probe(
        conn, "MATCH (e:Entity {nameNormalized:'acme', type:'Organization'}) RETURN count(e)"
    )
    assert count == 2


def test_link_chunk_about_entity_writes_the_edge(repo, conn):
    _document_with_chunk(repo)
    repo.create_entity(
        "test", entity_id="e1", name="Bob", name_normalized="bob",
        type="Person", created_at=100,
    )

    repo.link_chunk_about_entity("test", chunk_id="c0", entity_id="e1")

    rows = _probe(
        conn,
        "MATCH (c:Chunk {chunkId:'c0'})-[r:ABOUT]->(e:Entity {entityId:'e1'}) RETURN count(r)",
    )
    assert rows == [[1]]


def test_link_chunk_about_entity_is_never_deduplicated(repo, conn):
    _document_with_chunk(repo)
    repo.create_entity(
        "test", entity_id="e1", name="Bob", name_normalized="bob",
        type="Person", created_at=100,
    )

    repo.link_chunk_about_entity("test", chunk_id="c0", entity_id="e1")
    repo.link_chunk_about_entity("test", chunk_id="c0", entity_id="e1")

    rows = _probe(
        conn,
        "MATCH (c:Chunk {chunkId:'c0'})-[r:ABOUT]->(e:Entity {entityId:'e1'}) RETURN count(r)",
    )
    assert rows == [[2]]  # a plain CREATE, never a guarded MERGE


def test_create_entity_relationship_writes_the_edge_with_provenance(repo, conn):
    _document_with_chunk(repo)
    repo.create_entity(
        "test", entity_id="e1", name="Alice", name_normalized="alice",
        type="Person", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e2", name="Acme", name_normalized="acme",
        type="Organization", created_at=100,
    )

    repo.create_entity_relationship(
        "test", subject_id="e1", object_id="e2", label="works at",
        source_chunk_id="c0", source_document_id="d1", created_at=200,
    )

    rows = _probe(
        conn,
        "MATCH (:Entity {entityId:'e1'})-[r:RELATES_TO]->(:Entity {entityId:'e2'}) "
        "RETURN r.label, r.sourceChunkId, r.sourceDocumentId, r.createdAt",
    )
    assert rows == [["works at", "c0", "d1", 200]]


def test_create_entity_relationship_is_never_deduplicated(repo, conn):
    # FR-6: conflicting/repeated facts are always kept, never merged.
    _document_with_chunk(repo)
    repo.create_entity(
        "test", entity_id="e1", name="Alice", name_normalized="alice",
        type="Person", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e2", name="Acme", name_normalized="acme",
        type="Organization", created_at=100,
    )

    repo.create_entity_relationship(
        "test", subject_id="e1", object_id="e2", label="works at",
        source_chunk_id="c0", source_document_id="d1", created_at=200,
    )
    repo.create_entity_relationship(
        "test", subject_id="e1", object_id="e2", label="works at",
        source_chunk_id="c0", source_document_id="d1", created_at=200,
    )

    rows = _probe(
        conn,
        "MATCH (:Entity {entityId:'e1'})-[r:RELATES_TO]->(:Entity {entityId:'e2'}) "
        "RETURN count(r)",
    )
    assert rows == [[2]]


# ── §14.6 Entity fusion — SAME_AS (K-050 M5 Stage 4) ──────────────────────────


def test_create_entity_with_auto_match_no_prior_candidate(repo, conn):
    result = repo.create_entity_with_auto_match(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=100, match_id="m1",
    )

    assert result == {
        "entityId": "e1", "exactMatched": False,
        "candidateEntityId": None, "matchId": None,
    }
    [[count]] = _probe(conn, "MATCH (:Entity {entityId:'e1'}) RETURN count(*)")
    assert count == 1
    [[edges]] = _probe(conn, "MATCH ()-[r:SAME_AS]->() RETURN count(r)")
    assert edges == 0


def test_create_entity_with_auto_match_links_an_existing_candidate(repo, conn):
    repo.create_entity_with_auto_match(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=100, match_id="m0",
    )

    result = repo.create_entity_with_auto_match(
        "test", entity_id="e2", name="Acme Inc", name_normalized="acme",
        type="Organization", created_at=200, match_id="m1",
    )

    assert result == {
        "entityId": "e2", "exactMatched": True,
        "candidateEntityId": "e1", "matchId": "m1",
    }
    rows = _probe(
        conn,
        "MATCH (:Entity {entityId:'e2'})-[r:SAME_AS {matchId:'m1'}]"
        "->(:Entity {entityId:'e1'}) "
        "RETURN r.status, r.confidence, r.technique, r.decidedBy, r.decidedAt, "
        "       r.resuggestCount, r.lastResuggestedAt",
    )
    assert rows == [["confirmed", 1.0, "exact_normalized_name_type", "system", 200, 0, None]]


def test_create_entity_with_auto_match_never_self_matches_on_the_first_call(repo, conn):
    # If the query's own CREATE were somehow visible to its own preceding
    # MATCH, a brand-new (nameNormalized, type) pair would incorrectly report
    # exactMatched=True against itself. Sharpest possible check: zero prior
    # entities of this (nameNormalized, type) exist at all.
    result = repo.create_entity_with_auto_match(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=100, match_id="m1",
    )

    assert result["exactMatched"] is False
    [[edges]] = _probe(conn, "MATCH ()-[r:SAME_AS]->() RETURN count(r)")
    assert edges == 0


def test_create_entity_with_auto_match_picks_the_oldest_candidate_on_a_tie(repo, conn):
    repo.create_entity_with_auto_match(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=300, match_id="m0",
    )
    repo.create_entity_with_auto_match(
        "test", entity_id="e2", name="Acme", name_normalized="acme",
        type="Organization", created_at=100, match_id="m1",
    )  # e2 is OLDER than e1 despite being created second (createdAt=100 < 300)

    result = repo.create_entity_with_auto_match(
        "test", entity_id="e3", name="Acme", name_normalized="acme",
        type="Organization", created_at=500, match_id="m2",
    )

    assert result["candidateEntityId"] == "e2"  # the older of the two, not e1


def test_create_entity_with_auto_match_requires_matching_type_too(repo, conn):
    repo.create_entity_with_auto_match(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=100, match_id="m0",
    )

    result = repo.create_entity_with_auto_match(
        "test", entity_id="e2", name="Acme", name_normalized="acme",
        type="Product", created_at=200, match_id="m1",
    )

    assert result["exactMatched"] is False  # same normalized name, different type


def test_create_entity_with_auto_match_concurrent_calls_produce_exactly_one_edge(repo, conn):
    """The plan-gate review's BLOCKER regression test (`document-ingestion.md`
    §3.4/§5's "Exact-tier auto-merge race"): two entities sharing the same
    `(nameNormalized, type)`, created via two REAL concurrent
    `create_entity_with_auto_match` calls on separate connections/threads,
    must produce exactly one `SAME_AS{status:'confirmed'}` edge — never zero
    (both calls missing each other's not-yet-committed sibling) and never
    duplicated.
    """
    import threading

    from falkorchat import db

    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def _create(entity_id, created_at, match_id):
        try:
            barrier.wait(timeout=5)  # force both threads to fire together
            thread_repo = Repository(db.connect())
            thread_repo.create_entity_with_auto_match(
                "test", entity_id=entity_id, name="Acme", name_normalized="acme",
                type="Organization", created_at=created_at, match_id=match_id,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors` below
            errors.append(exc)

    t1 = threading.Thread(target=_create, args=("e1", 100, "m1"))
    t2 = threading.Thread(target=_create, args=("e2", 200, "m2"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert errors == []
    [[entity_count]] = _probe(conn, "MATCH (n:Entity {nameNormalized:'acme'}) RETURN count(n)")
    assert entity_count == 2
    # Unlabeled, directed endpoints — per `document-ingestion-graph.md` §1.4's
    # planner-trap note (never `(a:Entity)` on a SAME_AS-anchored query) and
    # §1.5 (the canonical write direction is `new -> candidate`); only two
    # entities exist in this test, so any confirmed SAME_AS edge is the one
    # between them.
    [[edge_count]] = _probe(
        conn, "MATCH ()-[r:SAME_AS {status:'confirmed'}]->() RETURN count(r)",
    )
    assert edge_count == 1  # never zero (the race), never duplicated


def test_find_fuzzy_candidates_matches_a_typo_and_filters_by_type(repo, conn):
    repo.create_entity(
        "test", entity_id="e1", name="Acme Corporation", name_normalized="acme corporation",
        type="Organization", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e2", name="Acme Corporation", name_normalized="acme corporation",
        type="Product", created_at=100,
    )  # same name, wrong type -> must NOT come back

    results = repo.find_fuzzy_candidates(
        "test", fuzzy_query="%Acmee%", type="Organization", limit=5,
    )

    assert [r["entityId"] for r in results] == ["e1"]


def test_find_fuzzy_candidates_no_match_returns_empty(repo):
    results = repo.find_fuzzy_candidates(
        "test", fuzzy_query="%zzzznomatch%", type="Organization", limit=5,
    )
    assert results == []


def test_create_or_reopen_match_creates_a_pending_edge(repo, conn):
    repo.create_entity(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e2", name="Acme Co", name_normalized="acme co",
        type="Organization", created_at=100,
    )

    result = repo.create_or_reopen_match(
        "test", new_entity_id="e1", candidate_entity_id="e2", match_id="m1",
        status="pending", confidence=2.5, technique="fuzzy_fulltext", created_at=100,
    )

    assert result == {"created": True, "reopened": False, "matchId": "m1", "status": "pending"}
    rows = _probe(
        conn,
        "MATCH (:Entity{entityId:'e1'})-[r:SAME_AS {matchId:'m1'}]->(:Entity{entityId:'e2'}) "
        "RETURN r.status, r.confidence, r.technique, r.decidedBy, r.decidedAt",
    )
    assert rows == [["pending", 2.5, "fuzzy_fulltext", None, None]]


def test_create_or_reopen_match_is_idempotent_for_the_same_pair(repo):
    repo.create_entity(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e2", name="Acme Co", name_normalized="acme co",
        type="Organization", created_at=100,
    )
    repo.create_or_reopen_match(
        "test", new_entity_id="e1", candidate_entity_id="e2", match_id="m1",
        status="pending", confidence=2.5, technique="fuzzy_fulltext", created_at=100,
    )

    result = repo.create_or_reopen_match(
        "test", new_entity_id="e1", candidate_entity_id="e2", match_id="m2",
        status="pending", confidence=3.0, technique="fuzzy_fulltext", created_at=200,
    )

    assert result == {"created": False, "reopened": False, "matchId": "m1", "status": "pending"}


def test_create_or_reopen_match_finds_the_pair_regardless_of_argument_order(repo):
    repo.create_entity(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e2", name="Acme Co", name_normalized="acme co",
        type="Organization", created_at=100,
    )
    repo.create_or_reopen_match(
        "test", new_entity_id="e1", candidate_entity_id="e2", match_id="m1",
        status="pending", confidence=2.5, technique="fuzzy_fulltext", created_at=100,
    )

    # swapped argument order (candidate discovered "first" this time)
    result = repo.create_or_reopen_match(
        "test", new_entity_id="e2", candidate_entity_id="e1", match_id="m2",
        status="pending", confidence=3.0, technique="fuzzy_fulltext", created_at=200,
    )

    assert result == {"created": False, "reopened": False, "matchId": "m1", "status": "pending"}


def test_create_or_reopen_match_reopens_a_rejected_edge(repo, conn):
    repo.create_entity(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e2", name="Acme Co", name_normalized="acme co",
        type="Organization", created_at=100,
    )
    repo.create_or_reopen_match(
        "test", new_entity_id="e1", candidate_entity_id="e2", match_id="m1",
        status="pending", confidence=2.5, technique="fuzzy_fulltext", created_at=100,
    )
    repo.reject_match("test", match_id="m1", decided_by="u1", decided_at=150)

    result = repo.create_or_reopen_match(
        "test", new_entity_id="e1", candidate_entity_id="e2", match_id="m2",
        status="pending", confidence=3.0, technique="fuzzy_fulltext", created_at=200,
    )

    assert result == {"created": False, "reopened": True, "matchId": "m1", "status": "pending"}
    rows = _probe(
        conn,
        "MATCH ()-[r:SAME_AS {matchId:'m1'}]->() "
        "RETURN r.status, r.resuggestCount, r.lastResuggestedAt",
    )
    assert rows == [["pending", 1, 200]]
    [[count]] = _probe(conn, "MATCH ()-[r:SAME_AS]->() RETURN count(r)")
    assert count == 1  # no duplicate edge


def _seeded_match(repo, *, status="pending"):
    repo.create_entity(
        "test", entity_id="e1", name="Acme", name_normalized="acme",
        type="Organization", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e2", name="Acme Co", name_normalized="acme co",
        type="Organization", created_at=100,
    )
    repo.create_or_reopen_match(
        "test", new_entity_id="e1", candidate_entity_id="e2", match_id="m1",
        status=status, confidence=2.5, technique="fuzzy_fulltext", created_at=100,
    )


def test_confirm_match_flips_status_and_stamps_audit_fields(repo, conn):
    _seeded_match(repo)

    result = repo.confirm_match("test", match_id="m1", decided_by="u1", decided_at=500)

    assert result == {"matchId": "m1", "status": "confirmed", "entityA": "e1", "entityB": "e2"}
    rows = _probe(
        conn, "MATCH ()-[r:SAME_AS {matchId:'m1'}]->() RETURN r.decidedBy, r.decidedAt",
    )
    assert rows == [["u1", 500]]


def test_confirm_match_returns_none_for_unknown_match_id(repo):
    assert repo.confirm_match("test", match_id="nope", decided_by="u1", decided_at=500) is None


def test_reject_match_flips_status_and_stamps_audit_fields(repo, conn):
    _seeded_match(repo)

    result = repo.reject_match("test", match_id="m1", decided_by="u1", decided_at=500)

    assert result == {"matchId": "m1", "status": "rejected", "entityA": "e1", "entityB": "e2"}
    rows = _probe(conn, "MATCH ()-[r:SAME_AS {matchId:'m1'}]->() RETURN r.status")
    assert rows == [["rejected"]]


def test_reject_match_returns_none_for_unknown_match_id(repo):
    assert repo.reject_match("test", match_id="nope", decided_by="u1", decided_at=500) is None


def test_recheck_match_flips_a_rejected_edge_back_to_pending(repo, conn):
    _seeded_match(repo)
    repo.reject_match("test", match_id="m1", decided_by="u1", decided_at=200)

    result = repo.recheck_match("test", match_id="m1", at=300)

    assert result == {"matchId": "m1", "status": "pending", "entityA": "e1", "entityB": "e2"}
    rows = _probe(
        conn,
        "MATCH ()-[r:SAME_AS {matchId:'m1'}]->() "
        "RETURN r.status, r.resuggestCount, r.lastResuggestedAt",
    )
    assert rows == [["pending", 1, 300]]


def test_recheck_match_is_a_noop_for_a_pending_match(repo):
    _seeded_match(repo, status="pending")

    assert repo.recheck_match("test", match_id="m1", at=300) is None


def test_recheck_match_is_a_noop_for_an_unknown_match_id(repo):
    assert repo.recheck_match("test", match_id="nope", at=300) is None


def test_list_pending_matches_returns_only_pending_oldest_first(repo):
    repo.create_entity(
        "test", entity_id="e1", name="A", name_normalized="a",
        type="Organization", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e2", name="A2", name_normalized="a2",
        type="Organization", created_at=100,
    )
    repo.create_entity(
        "test", entity_id="e3", name="B", name_normalized="b",
        type="Organization", created_at=100,
    )
    repo.create_or_reopen_match(
        "test", new_entity_id="e1", candidate_entity_id="e2", match_id="m2",
        status="pending", confidence=1.0, technique="fuzzy_fulltext", created_at=200,
    )
    repo.create_or_reopen_match(
        "test", new_entity_id="e1", candidate_entity_id="e3", match_id="m1",
        status="pending", confidence=1.0, technique="fuzzy_fulltext", created_at=100,
    )
    # a confirmed edge must never show up in the pending-only listing
    repo.create_entity_with_auto_match(
        "test", entity_id="e4", name="A", name_normalized="a",
        type="Organization", created_at=300, match_id="m3",
    )

    rows = repo.list_pending_matches("test", limit=50)

    assert [r["matchId"] for r in rows] == ["m1", "m2"]  # oldest first


def test_list_matches_filtered_by_status(repo):
    _seeded_match(repo)  # m1, pending
    repo.create_entity_with_auto_match(
        "test", entity_id="e3", name="Acme", name_normalized="acme",
        type="Organization", created_at=300, match_id="m4",
    )  # auto-confirms against e1 (oldest 'acme'/Organization entity)

    pending = repo.list_matches("test", status="pending", limit=50)
    confirmed = repo.list_matches("test", status="confirmed", limit=50)

    assert [r["matchId"] for r in pending] == ["m1"]
    assert [r["matchId"] for r in confirmed] == ["m4"]


def test_list_matches_unfiltered_includes_every_status(repo):
    _seeded_match(repo)  # m1, pending
    repo.reject_match("test", match_id="m1", decided_by="u1", decided_at=200)
    repo.create_entity_with_auto_match(
        "test", entity_id="e3", name="Acme", name_normalized="acme",
        type="Organization", created_at=300, match_id="m4",
    )  # auto-confirms

    rows = repo.list_matches("test", limit=50)

    assert {r["matchId"] for r in rows} == {"m1", "m4"}
    statuses = {r["matchId"]: r["status"] for r in rows}
    assert statuses == {"m1": "rejected", "m4": "confirmed"}


def test_list_matches_respects_limit(repo):
    repo.create_entity(
        "test", entity_id="e1", name="A", name_normalized="a",
        type="Organization", created_at=100,
    )
    for i in range(3):
        repo.create_entity(
            "test", entity_id=f"cand{i}", name=f"A{i}", name_normalized=f"a{i}",
            type="Organization", created_at=100,
        )
        repo.create_or_reopen_match(
            "test", new_entity_id="e1", candidate_entity_id=f"cand{i}",
            match_id=f"m{i}", status="pending", confidence=1.0,
            technique="fuzzy_fulltext", created_at=100 + i,
        )

    rows = repo.list_matches("test", status="pending", limit=2)

    assert len(rows) == 2


# ── §5 Full-text search ────────────────────────────────────────────────────────


def test_search_messages_finds_by_keyword(repo):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="hello world", role="user", created_at=120,
    )
    repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="m2", author_id="u1",
        text="goodbye moon", role="user", created_at=130,
    )

    hits = repo.search_messages("test", query="hello")

    assert [h["msgId"] for h in hits] == ["m1"]
    assert hits[0]["text"] == "hello world"
    assert hits[0]["createdAt"] == 120
    assert hits[0]["threadId"] == "t1"  # denormalized navigation metadata (K-007)
    assert "score" in hits[0]


def test_search_messages_empty_when_no_match(repo):
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="m1", author_id="u1",
        text="hello world", role="user", created_at=120,
    )

    assert repo.search_messages("test", query="nonexistentterm") == []


# ── validation reads (used by services) ────────────────────────────────────────


def test_channel_exists(repo):
    assert repo.channel_exists("test", channel_id="c1") is False
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    assert repo.channel_exists("test", channel_id="c1") is True


def test_resolve_member_kinds_maps_label_or_none_across_labels(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.ensure_agent("test", agent_id="a1", name="Bot")

    got = repo.resolve_member_kinds("test", ids=["u1", "a1", "ghost"])

    assert got == {"u1": "User", "a1": "Agent", "ghost": None}


def test_resolve_member_kinds_empty_input(repo):
    assert repo.resolve_member_kinds("test", ids=[]) == {}


def test_thread_exists(repo):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    assert repo.thread_exists("test", thread_id="t1") is False
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x", created_at=110)
    assert repo.thread_exists("test", thread_id="t1") is True


# ── §2 list_thread_participants (K-036 — web-api-coverage FR-8) ────────────────


def test_list_thread_participants_returns_both_kinds(repo, conn):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x", created_at=110)
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.ensure_agent("test", agent_id="a1", name="Bot")
    _add_to_channel(conn, member_id="u1", channel_id="c1")
    _add_to_channel(conn, member_id="a1", channel_id="c1")

    rows = repo.list_thread_participants("test", thread_id="t1")

    assert {(r["memberId"], r["displayName"], tuple(r["type"])) for r in rows} == {
        ("u1", "Alice", ("User",)),
        ("a1", None, ("Agent",)),
    }


def test_list_thread_participants_only_human(repo, conn):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x", created_at=110)
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _add_to_channel(conn, member_id="u1", channel_id="c1")

    rows = repo.list_thread_participants("test", thread_id="t1")

    assert [(r["memberId"], r["type"]) for r in rows] == [("u1", ["User"])]


def test_list_thread_participants_only_agent(repo, conn):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x", created_at=110)
    repo.ensure_agent("test", agent_id="a1", name="Bot")
    _add_to_channel(conn, member_id="a1", channel_id="c1")

    rows = repo.list_thread_participants("test", thread_id="t1")

    assert [(r["memberId"], r["type"]) for r in rows] == [("a1", ["Agent"])]


def test_list_thread_participants_empty_when_channel_has_no_members(repo):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x", created_at=110)

    assert repo.list_thread_participants("test", thread_id="t1") == []


def test_list_thread_participants_empty_when_thread_unknown(repo):
    assert repo.list_thread_participants("test", thread_id="ghost") == []


# ── §11 Workflow definitions & snapshots (M3 Slice 1) ───────────────────────────
#
# Reference-scoped methods take NO `ws` (defs are global, plan F3); workspace-
# scoped methods take `ws`. The `wf_repo` fixture wipes BOTH ws:test and the
# global `reference` graph so reference-def tests stay isolated (plan F8).

# A small canonical def: 3 steps, 2 transitions. `config`/`guard` are opaque
# strings stored verbatim (rule 8) — one non-empty, one empty, to pin round-trip.
DEF_STEPS = [
    {"key": "start", "type": "human", "config": "{}"},
    {"key": "review", "type": "decision", "config": "cfg-review"},
    {"key": "done", "type": "message", "config": ""},
]
DEF_TRANSITIONS = [
    {"from": "start", "to": "review", "on": "submitted", "guard": "", "order": 0},
    {"from": "review", "to": "done", "on": "approved", "guard": "score>0", "order": 0},
]


def _publish_sample(repo, *, key="onboarding", version="1"):
    return repo.publish_def(
        key=key, version=version, name="Onboarding", kind="process",
        start_key="start", steps=DEF_STEPS, transitions=DEF_TRANSITIONS,
    )


def _sorted_steps(steps):
    return sorted(steps, key=lambda s: s["key"])


def _sorted_transitions(trs):
    return sorted(trs, key=lambda t: (t["from"], t["to"], t["on"], t["order"]))


def test_publish_def_reports_step_and_transition_counts(wf_repo):
    res = _publish_sample(wf_repo)

    assert res["key"] == "onboarding"
    assert res["version"] == "1"
    assert res["stepCount"] == 3
    assert res["transitionCount"] == 2


def test_publish_def_then_read_subgraph_returns_full_def(wf_repo):
    _publish_sample(wf_repo)

    sub = wf_repo.read_def_subgraph(key="onboarding", version="1")

    assert sub["name"] == "Onboarding"
    assert sub["kind"] == "process"
    assert sub["start_key"] == "start"
    assert _sorted_steps(sub["steps"]) == _sorted_steps(DEF_STEPS)
    assert _sorted_transitions(sub["transitions"]) == _sorted_transitions(DEF_TRANSITIONS)


def test_read_def_subgraph_none_when_absent(wf_repo):
    assert wf_repo.read_def_subgraph(key="ghost", version="1") is None


def test_publish_def_is_idempotent_no_new_nodes_on_republish(wf_repo):
    _publish_sample(wf_repo)
    before = wf_repo.read_def_subgraph(key="onboarding", version="1")

    res2 = _publish_sample(wf_repo)  # re-publish same key@version

    # MERGE-backed: structural no-op. Counts still reflect the def's shape,
    # but the subgraph is unchanged (immutability per version).
    after = wf_repo.read_def_subgraph(key="onboarding", version="1")
    assert res2["stepCount"] == 3
    assert _sorted_steps(after["steps"]) == _sorted_steps(before["steps"])
    assert _sorted_transitions(after["transitions"]) == _sorted_transitions(
        before["transitions"]
    )


def test_publish_def_direct_call_is_unsafe_mints_parallel_transition_on_retarget(
    wf_repo,
):
    # K-034: the guard lives in services.py, not here — see
    # services._check_no_structural_conflict. Do not "fix" this by changing
    # _PUBLISH_CYPHER without a graph-dba re-PROFILE gate. This pins that a raw
    # `Repository.publish_def` call is unsafe on its own: the safety is a
    # services.py contract, not a repository-layer guarantee.
    _publish_sample(wf_repo)

    # same (from, on, order) as the stored start->review transition, but
    # retargeted to "done" — a *different* MERGE pattern, not an update (§2.1)
    wf_repo.publish_def(
        key="onboarding", version="1", name="Onboarding", kind="process",
        start_key="start", steps=DEF_STEPS,
        transitions=[
            {"from": "start", "to": "done", "on": "submitted", "guard": "",
             "order": 0},
            DEF_TRANSITIONS[1],
        ],
    )

    sub = wf_repo.read_def_subgraph(key="onboarding", version="1")
    from_start = [
        t for t in sub["transitions"]
        if t["from"] == "start" and t["on"] == "submitted" and t["order"] == 0
    ]
    assert len(from_start) == 2  # parallel edge minted, the old one never removed
    assert {t["to"] for t in from_start} == {"review", "done"}


def test_get_def_specific_version(wf_repo):
    _publish_sample(wf_repo, version="1")

    got = wf_repo.get_def(key="onboarding", version="1")

    assert got == {
        "key": "onboarding", "version": "1", "name": "Onboarding", "kind": "process",
    }


def test_get_def_latest_version_when_version_none(wf_repo):
    _publish_sample(wf_repo, version="1")
    _publish_sample(wf_repo, version="2")

    got = wf_repo.get_def(key="onboarding")  # latest

    assert got["version"] == "2"


def test_get_def_none_when_absent(wf_repo):
    assert wf_repo.get_def(key="ghost") is None


def test_list_defs_returns_published(wf_repo):
    _publish_sample(wf_repo, key="a", version="1")
    _publish_sample(wf_repo, key="b", version="1")

    keys = {(d["key"], d["version"]) for d in wf_repo.list_defs()}

    assert ("a", "1") in keys and ("b", "1") in keys


def test_list_defs_empty_when_none(wf_repo):
    assert wf_repo.list_defs() == []


def _materialize_sample(repo, *, key="onboarding", version="1"):
    return repo.materialize_snapshot(
        "test", key=key, version=version, name="Onboarding", kind="process",
        start_key="start", steps=DEF_STEPS, transitions=DEF_TRANSITIONS,
    )


def test_materialize_snapshot_reports_counts(wf_repo):
    res = _materialize_sample(wf_repo)

    assert res["key"] == "onboarding"
    assert res["version"] == "1"
    assert res["stepCount"] == 3
    assert res["transitionCount"] == 2


def test_materialize_then_get_snapshot_returns_full_subgraph(wf_repo):
    _materialize_sample(wf_repo)

    snap = wf_repo.get_snapshot("test", key="onboarding", version="1")

    assert snap["name"] == "Onboarding"
    assert snap["kind"] == "process"
    assert snap["start_key"] == "start"
    assert _sorted_steps(snap["steps"]) == _sorted_steps(DEF_STEPS)
    assert _sorted_transitions(snap["transitions"]) == _sorted_transitions(
        DEF_TRANSITIONS
    )


def test_get_snapshot_none_when_absent(wf_repo):
    assert wf_repo.get_snapshot("test", key="ghost", version="1") is None


def test_materialize_snapshot_is_idempotent_on_rematerialize(wf_repo):
    _materialize_sample(wf_repo)
    before = wf_repo.get_snapshot("test", key="onboarding", version="1")

    _materialize_sample(wf_repo)  # re-materialize same key@version

    after = wf_repo.get_snapshot("test", key="onboarding", version="1")
    assert _sorted_steps(after["steps"]) == _sorted_steps(before["steps"])
    assert _sorted_transitions(after["transitions"]) == _sorted_transitions(
        before["transitions"]
    )


def test_read_def_structure_returns_full_structure(wf_repo):
    _publish_sample(wf_repo)

    st = wf_repo.read_def_structure(key="onboarding", version="1")

    assert st["name"] == "Onboarding"
    assert st["kind"] == "process"
    assert st["start_keys"] == ["start"]
    assert _sorted_steps(st["steps"]) == _sorted_steps(DEF_STEPS)
    assert _sorted_transitions(st["transitions"]) == _sorted_transitions(
        DEF_TRANSITIONS
    )


def test_read_snapshot_structure_mirrors_the_def_structure(wf_repo):
    _publish_sample(wf_repo)
    _materialize_sample(wf_repo)

    ref = wf_repo.read_def_structure(key="onboarding", version="1")
    snap = wf_repo.read_snapshot_structure("test", key="onboarding", version="1")

    # Same query constants, different root label. Steps/transitions come back
    # unordered from the graph (F6) — canonical ordering lives in `services.py`.
    assert (snap["name"], snap["kind"], snap["start_keys"]) == (
        ref["name"], ref["kind"], ref["start_keys"]
    )
    assert _sorted_steps(snap["steps"]) == _sorted_steps(ref["steps"])
    assert _sorted_transitions(snap["transitions"]) == _sorted_transitions(
        ref["transitions"]
    )


def test_read_structure_none_when_absent(wf_repo):
    assert wf_repo.read_def_structure(key="ghost", version="1") is None
    assert wf_repo.read_snapshot_structure("test", key="ghost", version="1") is None


def test_read_snapshot_structure_none_when_graph_key_fully_deleted(conn):
    """K-005 reproduction (live): `test_queries.sh`'s teardown doesn't just empty
    `reference`'s node data, it `GRAPH.DELETE`s the graph key entirely
    (`falkor-chat/AGENTS.md` `test_queries.sh` row). A read against a
    fully-deleted key raises FalkorDB's `ERR Invalid graph operation on empty
    key` `ResponseError` — a different failure mode from "key exists, root node
    absent" (`test_read_structure_none_when_absent` above), and one
    `_read_structure` (repository.py) previously let escape uncaught, which is
    what let it defeat `Services.diff_def_snapshot`'s both-sides-checked
    contract (services.py:1748) — see `claude/coder/kaizen/plan.md` K-005 for
    the full trace.

    Uses a throwaway `ws:<probe>` key (not the shared `ws:test`/`reference`)
    per `falkor-chat/AGENTS.md`'s "probing shared graph state without mutating
    it" guidance — genuinely `GRAPH.DELETE`s it, the same op `test_queries.sh`
    performs, so `ro_query` genuinely raises rather than being mocked.
    """
    probe_ws = "k005probe"
    graph = db.workspace_graph(conn, probe_ws)
    graph.query("RETURN 1")  # materialize the key (write-mode query side effect)
    graph.delete()  # the exact op test_queries.sh's teardown performs

    repo = Repository(conn)

    assert repo.read_snapshot_structure(probe_ws, key="ghost", version="1") is None


class _FakeRes:
    def __init__(self, rows):
        self.result_set = rows


class _FakeGraph:
    """Minimal stand-in for a FalkorDB graph handle, replaying canned result sets.

    `_read_structure` is a `@staticmethod` whose only collaborator is the injected
    `graph` (two positional `ro_query` calls), so its row handling is reachable
    with **no publish, no Cypher and no FalkorDB** — which is what lets the
    multi-`START` branch below be pinned without asserting publish-additivity
    (that is K-034's to change).
    """

    def __init__(self, *result_sets):
        self._queued = [_FakeRes(rows) for rows in result_sets]
        self.calls: list[tuple[str, dict]] = []

    def ro_query(self, cypher, params):
        self.calls.append((cypher, params))
        return self._queued.pop(0)


def test_read_structure_unions_multi_start_meta_rows_pinning_v1s_live_shape():
    # Pins the exact shape K-031 V-1 observed live (falkordb v4.18.11, throwaway
    # `ws:k031probe`) and QUERIES.md §11.2's multi-`START` note records: two
    # `START` edges on one root ⇒ 11.2a returns **two rows, one per distinct
    # `startKey`, each carrying the full `steps` collection**. This is the only
    # coverage of `_read_structure`'s union loop — every graph-backed structure
    # test feeds it a single meta row, so a refactor back to `result_set[0]`
    # would otherwise leave the suite green while silently un-wiring `startKeys`
    # and `scripts/verify_workflows.sh`'s finding-3 tripwire.
    steps = [
        {"key": "b", "type": "decision", "config": ""},
        {"key": "a", "type": "human", "config": '{"waitsForHuman": true}'},
    ]
    graph = _FakeGraph(
        [["Probe", "process", "a", steps], ["Probe", "process", "b", steps]],
        [[[{"from": "a", "to": "b", "on": "go", "guard": "", "order": 0}]]],
    )

    st = Repository._read_structure(
        graph, label="WorkflowDefSnapshot", key="probe", version="v1"
    )

    # Both start keys survive; `_read_structure` preserves first-seen order and
    # `services._canonical_structure` is what sorts them for the REST boundary.
    assert st["start_keys"] == ["a", "b"]
    # The `steps` collection repeated on every row is UNIONED, not duplicated.
    assert [s["key"] for s in st["steps"]] == ["b", "a"]
    assert (st["name"], st["kind"]) == ("Probe", "process")
    assert st["transitions"] == [
        {"from": "a", "to": "b", "on": "go", "guard": "", "order": 0}
    ]
    # Both reads ran the UNMODIFIED §11.2 constants — no new/edited Cypher.
    assert [c[0] for c in graph.calls] == [
        Repository._READ_META_CYPHER.format(label="WorkflowDefSnapshot"),
        Repository._READ_TRANSITIONS_CYPHER.format(label="WorkflowDefSnapshot"),
    ]
    assert all(c[1] == {"key": "probe", "version": "v1"} for c in graph.calls)


def test_read_structure_tolerates_a_root_with_no_start_edge_and_no_steps():
    # The other branch of the same loop: both `OPTIONAL MATCH`es miss, so 11.2a
    # returns ONE row whose `startKey` is null and whose `collect(DISTINCT …)`
    # holds a single all-null map. Neither may leak into the structure — an empty
    # `start_keys` becomes an explicit `startKey: null` at the REST boundary,
    # which is the anomaly the observability surface exists to show.
    graph = _FakeGraph(
        [["Orphan", "process", None, [{"key": None, "type": None, "config": None}]]],
        [],
    )

    st = Repository._read_structure(
        graph, label="WorkflowDef", key="orphan", version="1"
    )

    assert st["start_keys"] == []
    assert st["steps"] == []
    assert st["transitions"] == []


class _RaisingFakeGraph:
    """Raises a genuine `redis.exceptions.ResponseError` from `ro_query`, exactly
    the shape live FalkorDB raises for a fully `GRAPH.DELETE`d graph key (K-005)
    — not a mocked `None` return. Reused for both `_read_structure` callers
    (`read_def_structure`'s `reference` side, `read_snapshot_structure`'s
    `ws:{id}` side already covered live above) since both delegate to the same
    static helper with only the `graph` handle differing.
    """

    def __init__(self, message: str):
        self._message = message

    def ro_query(self, cypher, params):
        raise ResponseError(self._message)


def test_read_structure_none_when_graph_key_fully_deleted():
    # The `reference`-side path (label=WorkflowDef) — mirrors the live
    # `read_snapshot_structure` reproduction above without touching the shared
    # `reference` graph (AGENTS.md: `reference` has no isolatable graph seam).
    graph = _RaisingFakeGraph("ERR Invalid graph operation on empty key")

    st = Repository._read_structure(
        graph, label="WorkflowDef", key="onboarding", version="1"
    )

    assert st is None


def test_read_structure_reraises_response_errors_that_are_not_empty_key():
    # The fix must not swallow every `ResponseError` — only the specific "empty
    # key" (fully-deleted-graph) case, matching the existing `_read_or_absent`
    # (services.py:649) and vector-index-probe (repository.py:807) precedent.
    graph = _RaisingFakeGraph("RediSearch: Syntax error at offset 6")

    with pytest.raises(ResponseError, match="Syntax error"):
        Repository._read_structure(
            graph, label="WorkflowDef", key="onboarding", version="1"
        )


def test_snapshot_structurally_matches_reference_def(wf_repo):
    # publish → read def subgraph → materialize with that subgraph → parity
    _publish_sample(wf_repo)
    ref = wf_repo.read_def_subgraph(key="onboarding", version="1")
    wf_repo.materialize_snapshot(
        "test", key="onboarding", version="1", name=ref["name"], kind=ref["kind"],
        start_key=ref["start_key"], steps=ref["steps"], transitions=ref["transitions"],
    )

    snap = wf_repo.get_snapshot("test", key="onboarding", version="1")

    assert snap == ref  # structurally identical (both label-agnostic subgraphs)


def test_list_snapshots_returns_materialized(wf_repo):
    _materialize_sample(wf_repo, key="a", version="1")
    _materialize_sample(wf_repo, key="b", version="1")

    keys = {(s["key"], s["version"]) for s in wf_repo.list_snapshots("test")}

    assert ("a", "1") in keys and ("b", "1") in keys


# ── §12 Workflow execution — runs, step-runs & traces (M3 executor, K-022) ───────
#
# Integration tests against `ws:test`; each method wraps a verified QUERIES §12
# query 1:1. A run + its trace are a workspace-local subgraph anchored on a
# materialized snapshot (§11) and a trigger `Message`. A 3-step conversation def
# (intake→research→answer) is the fixture: intake `waitsForHuman`, research→answer
# unconditional (D5), answer terminal.

from falkorchat.repository import (  # noqa: E402
    StepBudgetExceededError,
    WorkflowRunNotFoundError,
)

RUN_STEPS = [
    {"key": "intake", "type": "agent", "config": '{"waitsForHuman":true}'},
    {"key": "research", "type": "agent", "config": "{}"},
    {"key": "answer", "type": "agent", "config": "{}"},
]
RUN_TRANSITIONS = [
    {"from": "intake", "to": "research", "on": "ready",
     "guard": '{"kind":"llm","text":"enough info?"}', "order": 0},
    {"from": "research", "to": "answer", "on": "done", "guard": "", "order": 0},
]


def _uid(step_key, *, key="triage", version="1"):
    return f"{key}:{version}:{step_key}"


def _seed_run_fixtures(repo, *, trigger="trig1"):
    """Materialize the triage snapshot + a thread with one trigger message."""
    repo.materialize_snapshot(
        "test", key="triage", version="1", name="Triage", kind="conversation",
        start_key="intake", steps=RUN_STEPS, transitions=RUN_TRANSITIONS,
    )
    _seed_thread(repo)  # channel c1 + thread t1 + user u1
    repo.post_first_message(
        "test", thread_id="t1", msg_id=trigger, author_id="u1",
        text="please help", role="user", created_at=120,
    )


def _start(repo, *, run_id="r1", trigger="trig1", trace=False, max_steps=12):
    return repo.start_run(
        "test", run_id=run_id, def_key="triage", def_version="1",
        started_at=1000, trigger_msg_id=trigger,
        ctx='{"threadId":"t1"}', trace=trace, max_steps=max_steps,
    )


# ── §12.1 start_run ──────────────────────────────────────────────────────────

def test_start_run_creates_the_run_subgraph(wf_repo):
    _seed_run_fixtures(wf_repo)

    res = _start(wf_repo)

    assert res["runId"] == "r1"
    assert res["startKey"] == "intake"
    assert res["status"] == "running"
    assert res["stepCount"] == 0

    got = wf_repo.get_run("test", run_id="r1")
    assert got["status"] == "running"
    assert got["atStepKey"] == "intake"
    assert got["defKey"] == "triage"
    assert got["defVersion"] == "1"
    assert got["maxSteps"] == 12
    assert got["trace"] is False
    assert got["waitingThreadId"] == ""


def test_start_run_zero_rows_when_trigger_missing(wf_repo):
    _seed_run_fixtures(wf_repo)
    assert _start(wf_repo, trigger="ghost") is None


def test_start_run_zero_rows_when_snapshot_missing(wf_repo):
    _seed_thread(wf_repo)
    wf_repo.post_first_message(
        "test", thread_id="t1", msg_id="trig1", author_id="u1",
        text="hi", role="user", created_at=120,
    )
    assert _start(wf_repo) is None


# ── §12.2 record_step_and_advance ────────────────────────────────────────────

def _advance(repo, *, run_id="r1", step_run_id, to_step, status="done",
             output="out"):
    return repo.record_step_and_advance(
        "test", run_id=run_id, step_run_id=step_run_id, step_status=status,
        started_at=1001, ended_at=1002, input="in", output=output,
        to_step_uid=_uid(to_step),
    )


def test_first_advance_seeds_tail_bumps_count_relinks_at_step(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)

    res = _advance(wf_repo, step_run_id="sr1", to_step="research")

    assert res["stepCount"] == 1
    assert res["stepRunId"] == "sr1"
    assert res["ranStepKey"] == "intake"          # records the step LEFT
    assert wf_repo.get_run("test", run_id="r1")["atStepKey"] == "research"
    # exactly one StepRun, no NEXT yet
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert [s["stepKey"] for s in trail] == ["intake"]


def test_second_advance_appends_next_in_order(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    _advance(wf_repo, step_run_id="sr1", to_step="research")

    res = _advance(wf_repo, step_run_id="sr2", to_step="answer")

    assert res["stepCount"] == 2
    assert res["ranStepKey"] == "research"
    assert wf_repo.get_run("test", run_id="r1")["atStepKey"] == "answer"
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert [s["stepKey"] for s in trail] == ["intake", "research"]  # NEXT order


def test_advance_to_self_records_a_step_run(wf_repo):
    # the self-loop / suspend / terminal record shape: to == cur
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)

    res = _advance(wf_repo, step_run_id="sr1", to_step="intake")

    assert res["stepCount"] == 1
    assert wf_repo.get_run("test", run_id="r1")["atStepKey"] == "intake"


def test_advance_zero_rows_when_run_missing(wf_repo):
    _seed_run_fixtures(wf_repo)
    assert _advance(wf_repo, run_id="ghost", step_run_id="x", to_step="research") is None


def test_advance_round_trips_input_and_output_verbatim(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    _advance(wf_repo, step_run_id="sr1", to_step="research", output="verbatim-out")

    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert trail[0]["input"] == "in"
    assert trail[0]["output"] == "verbatim-out"
    assert trail[0]["status"] == "done"


# ── K-042 Landing 2 (FR-8, `-graph.md` §1.4/§1.7): resolvedModel/modelSource/
# modelFallback ride the same atomic CREATE and the same read projection ──────────

def test_advance_with_no_model_params_omits_the_three_properties(wf_repo):
    # The default (non-LLM step) case: no branching, no extra bytes — a NULL param
    # omits the property entirely rather than writing a literal null.
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    _advance(wf_repo, step_run_id="sr1", to_step="research")

    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert trail[0]["resolvedModel"] is None
    assert trail[0]["modelSource"] is None
    assert trail[0]["modelFallback"] is None


def test_advance_writes_resolved_model_source_and_fallback_when_given(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    wf_repo.record_step_and_advance(
        "test", run_id="r1", step_run_id="sr1", step_status="done",
        started_at=1001, ended_at=1002, input="in", output="out",
        to_step_uid=_uid("research"),
        resolved_model="lmstudio/qwen3-4b", model_source="step", model_fallback=True,
    )

    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert trail[0]["resolvedModel"] == "lmstudio/qwen3-4b"
    assert trail[0]["modelSource"] == "step"
    assert trail[0]["modelFallback"] is True


def test_advance_writes_model_fallback_false_is_never_produced_by_none_param(wf_repo):
    # The omission contract: modelFallback is either True (a fallback fired) or
    # absent/None (unknown/not-applicable) — never a written `false`.
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    wf_repo.record_step_and_advance(
        "test", run_id="r1", step_run_id="sr1", step_status="done",
        started_at=1001, ended_at=1002, input="in", output="out",
        to_step_uid=_uid("research"),
        resolved_model="lmstudio/qwen3-4b", model_source="default", model_fallback=None,
    )

    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert trail[0]["resolvedModel"] == "lmstudio/qwen3-4b"
    assert trail[0]["modelSource"] == "default"
    assert trail[0]["modelFallback"] is None


# ── §12.3 / §12.4 suspend / resume CAS ───────────────────────────────────────

def test_suspend_run_flips_running_to_waiting_and_denorms_thread(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)

    res = wf_repo.suspend_run("test", run_id="r1", thread_id="t1")

    assert res["status"] == "waiting"
    got = wf_repo.get_run("test", run_id="r1")
    assert got["status"] == "waiting"
    assert got["waitingThreadId"] == "t1"


def test_suspend_of_non_running_run_is_zero_rows_no_write(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    wf_repo.suspend_run("test", run_id="r1", thread_id="t1")  # now waiting

    # a second suspend can't apply — the CAS WHERE status='running' fails
    assert wf_repo.suspend_run("test", run_id="r1", thread_id="t1") is None


def test_resume_run_flips_waiting_to_running_single_flight(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    wf_repo.suspend_run("test", run_id="r1", thread_id="t1")

    first = wf_repo.resume_run("test", run_id="r1")
    second = wf_repo.resume_run("test", run_id="r1")  # loser of the race

    assert first["status"] == "running"
    assert second is None                              # single-flight CAS
    assert wf_repo.get_run("test", run_id="r1")["waitingThreadId"] == ""


# ── §12.16 find_due_wait_candidates (K-028) ───────────────────────────────────

TIMER_STEPS = [
    {"key": "park", "type": "wait",
     "config": '{"waitsForHuman":true,"waitForSeconds":60,"signal":"provisioned"}'},
    {"key": "park_human", "type": "human",
     "config": '{"waitsForHuman":true,"fields":["provisioned"]}'},
    {"key": "park_no_timer", "type": "wait",
     "config": '{"waitsForHuman":true,"signal":"provisioned"}'},
    {"key": "activate", "type": "decision", "config": "{}"},
]
TIMER_TRANSITIONS = [
    {"from": "park", "to": "activate", "on": "provisioned",
     "guard": '{"kind":"cmp","path":"ctx.provisioned","op":"truthy"}', "order": 0},
    {"from": "park_human", "to": "activate", "on": "provisioned",
     "guard": '{"kind":"cmp","path":"ctx.provisioned","op":"truthy"}', "order": 0},
    {"from": "park_no_timer", "to": "activate", "on": "provisioned",
     "guard": '{"kind":"cmp","path":"ctx.provisioned","op":"truthy"}', "order": 0},
]


def _timers_uid(step_key, *, key="timers-q", version="1"):
    return f"{key}:{version}:{step_key}"


def _seed_timer_snapshot(repo):
    repo.materialize_snapshot(
        "test", key="timers-q", version="1", name="Timers", kind="process",
        start_key="park", steps=TIMER_STEPS, transitions=TIMER_TRANSITIONS,
    )
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="trig-timers", author_id="u1",
        text="start", role="user", created_at=120,
    )


def _timer_run(repo, *, run_id, step_key="park", parked_at=1000, suspend=True):
    """Start a run and advance `AT_STEP` to `step_key`, recording a `StepRun`
    with `startedAt = parked_at` — the read half of what OUTCOME B writes.
    `suspend=False` leaves the run `running` (for the exclusion tests)."""
    repo.start_run(
        "test", run_id=run_id, def_key="timers-q", def_version="1",
        started_at=parked_at, trigger_msg_id="trig-timers", ctx="{}",
        trace=False, max_steps=12,
    )
    repo.record_step_and_advance(
        "test", run_id=run_id, step_run_id=f"{run_id}-sr0", step_status="done",
        started_at=parked_at, ended_at=parked_at, input="{}", output="{}",
        to_step_uid=_timers_uid(step_key),
    )
    if suspend:
        repo.suspend_run("test", run_id=run_id, thread_id="t1")


def test_find_due_wait_candidates_returns_a_waiting_wait_step_with_config_and_parked_at(
    wf_repo,
):
    _seed_timer_snapshot(wf_repo)
    _timer_run(wf_repo, run_id="r1", step_key="park", parked_at=5000)

    candidates = wf_repo.find_due_wait_candidates("test", limit=10)

    assert len(candidates) == 1
    row = candidates[0]
    assert row["runId"] == "r1"
    assert row["stepKey"] == "park"  # v3: the new RETURN-clause projection
    assert row["stepType"] == "wait"
    assert row["parkedAt"] == 5000
    assert json.loads(row["stepConfig"]) == {
        "waitsForHuman": True, "waitForSeconds": 60, "signal": "provisioned",
    }


def test_find_due_wait_candidates_is_due_agnostic_human_and_no_timer_steps_too(
    wf_repo,
):
    # The query itself never filters on timer keys or dueness — that is the
    # app-side job (`services._wait_due_at`/`sweep_due_workflow_runs`).
    _seed_timer_snapshot(wf_repo)
    _timer_run(wf_repo, run_id="r_human", step_key="park_human", parked_at=1000)
    _timer_run(wf_repo, run_id="r_no_timer", step_key="park_no_timer", parked_at=1000)

    candidates = wf_repo.find_due_wait_candidates("test", limit=10)

    by_run = {row["runId"]: row for row in candidates}
    assert set(by_run) == {"r_human", "r_no_timer"}
    assert by_run["r_human"]["stepType"] == "human"
    assert by_run["r_human"]["stepKey"] == "park_human"
    assert by_run["r_no_timer"]["stepType"] == "wait"
    assert by_run["r_no_timer"]["stepKey"] == "park_no_timer"


def test_find_due_wait_candidates_excludes_non_waiting_runs(wf_repo):
    _seed_timer_snapshot(wf_repo)
    _timer_run(wf_repo, run_id="r_running", parked_at=1000, suspend=False)
    _timer_run(wf_repo, run_id="r_done", parked_at=1000, suspend=False)
    wf_repo.complete_run("test", run_id="r_done", ended_at=2000)
    _timer_run(wf_repo, run_id="r_failed", parked_at=1000, suspend=False)
    wf_repo.fail_run("test", run_id="r_failed", ended_at=2000, ctx="{}")
    _timer_run(wf_repo, run_id="r_waiting", parked_at=1000)

    candidates = wf_repo.find_due_wait_candidates("test", limit=10)

    assert [row["runId"] for row in candidates] == ["r_waiting"]


def test_find_due_wait_candidates_respects_limit(wf_repo):
    _seed_timer_snapshot(wf_repo)
    for n in range(3):
        _timer_run(wf_repo, run_id=f"r{n}", parked_at=1000 + n)

    candidates = wf_repo.find_due_wait_candidates("test", limit=2)

    assert len(candidates) == 2


# ── §12.5 complete / fail ────────────────────────────────────────────────────

def test_complete_run_clears_at_step_and_sets_done(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)

    res = wf_repo.complete_run("test", run_id="r1", ended_at=2000)

    assert res["status"] == "done"
    got = wf_repo.get_run("test", run_id="r1")
    assert got["status"] == "done"
    assert got["atStepKey"] is None
    assert got["endedAt"] == 2000


def test_fail_run_clears_at_step_stamps_ctx_note(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)

    res = wf_repo.fail_run(
        "test", run_id="r1", ended_at=2000, ctx='{"error":"step budget exceeded"}'
    )

    assert res["status"] == "failed"
    got = wf_repo.get_run("test", run_id="r1")
    assert got["status"] == "failed"
    assert got["atStepKey"] is None
    assert got["ctx"] == '{"error":"step budget exceeded"}'


def test_complete_missing_run_zero_rows(wf_repo):
    _seed_run_fixtures(wf_repo)
    assert wf_repo.complete_run("test", run_id="ghost", ended_at=2000) is None


def test_step_runs_retained_after_fail(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    _advance(wf_repo, step_run_id="sr1", to_step="research")
    wf_repo.fail_run("test", run_id="r1", ended_at=2000, ctx="{}")

    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert [s["stepKey"] for s in trail] == ["intake"]  # audit trail preserved


# ── §12.6 link_step_emission (PRODUCED) ──────────────────────────────────────

def test_link_step_emission_creates_produced_edge(wf_repo, conn):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    _advance(wf_repo, step_run_id="sr1", to_step="research")

    res = wf_repo.link_step_emission("test", step_run_id="sr1", msg_id="trig1")

    assert res == {"stepRunId": "sr1", "msgId": "trig1"}
    produced = _probe(
        conn,
        "MATCH (sr:StepRun {stepRunId:'sr1'})-[:PRODUCED]->(m:Message) "
        "RETURN m.msgId",
    )
    assert produced == [["trig1"]]


def test_link_step_emission_is_idempotent(wf_repo, conn):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    _advance(wf_repo, step_run_id="sr1", to_step="research")
    wf_repo.link_step_emission("test", step_run_id="sr1", msg_id="trig1")
    wf_repo.link_step_emission("test", step_run_id="sr1", msg_id="trig1")  # retry

    count = _probe(
        conn,
        "MATCH (:StepRun {stepRunId:'sr1'})-[e:PRODUCED]->(:Message) RETURN count(e)",
    )
    assert count == [[1]]  # MERGE → exactly one edge


# ── K-056 — `link_step_emission` stamps the `toolsUsed` audit property ───────
#
# A durable, per-Message signal for "did the step that produced this reply actually
# dispatch a domain tool?" — persisted here (not just traced) so it survives long
# after the debug trace (if any) is gone. Pure audit/observability: the
# replayed-history breadcrumb this was originally built to feed
# (`_assemble_messages`) was reverted — see that function's docstring and
# `docs/reviews/salesperson-tool-reliability-impl.md`, MAJOR 1.

def test_link_step_emission_stamps_tools_used_on_the_message(wf_repo, conn):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    _advance(wf_repo, step_run_id="sr1", to_step="research")

    wf_repo.link_step_emission(
        "test", step_run_id="sr1", msg_id="trig1",
        tools_used=["lookup_product_fact"],
    )

    stamped = _probe(conn, "MATCH (m:Message {msgId:'trig1'}) RETURN m.toolsUsed")
    assert stamped == [[["lookup_product_fact"]]]


def test_link_step_emission_defaults_tools_used_to_empty_list(wf_repo, conn):
    # Backward-compatible default: a caller that doesn't pass tools_used (or a node
    # that dispatched nothing) stamps an empty list, not a missing/null property —
    # `read_thread`'s `coalesce(m.toolsUsed, [])` would degrade fine either way, but
    # the write path should be explicit rather than relying on that fallback.
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    _advance(wf_repo, step_run_id="sr1", to_step="research")

    wf_repo.link_step_emission("test", step_run_id="sr1", msg_id="trig1")

    stamped = _probe(conn, "MATCH (m:Message {msgId:'trig1'}) RETURN m.toolsUsed")
    assert stamped == [[[]]]


# ── §12.7 / §12.8 reads ──────────────────────────────────────────────────────

def test_get_run_none_when_absent(wf_repo):
    _seed_run_fixtures(wf_repo)
    assert wf_repo.get_run("test", run_id="ghost") is None


def test_read_step_runs_empty_before_any_advance(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    assert wf_repo.read_step_runs("test", run_id="r1") == []


def test_read_step_runs_returns_next_ordered_trail(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    _advance(wf_repo, step_run_id="sr1", to_step="research")
    _advance(wf_repo, step_run_id="sr2", to_step="answer")

    trail = wf_repo.read_step_runs("test", run_id="r1")

    assert [s["stepRunId"] for s in trail] == ["sr1", "sr2"]
    assert [s["stepKey"] for s in trail] == ["intake", "research"]


# ── §12.9 find_waiting_run_for_thread ────────────────────────────────────────

def test_find_waiting_run_for_thread_finds_parked_run(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    wf_repo.suspend_run("test", run_id="r1", thread_id="t1")

    res = wf_repo.find_waiting_run_for_thread("test", thread_id="t1")

    assert res["runId"] == "r1"
    assert res["status"] == "waiting"


def test_find_waiting_run_none_when_not_parked(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)  # running, not waiting
    assert wf_repo.find_waiting_run_for_thread("test", thread_id="t1") is None


def test_find_waiting_run_none_for_other_thread(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    wf_repo.suspend_run("test", run_id="r1", thread_id="t1")
    assert wf_repo.find_waiting_run_for_thread("test", thread_id="other") is None


# ── §12.14 find_runs_for_thread ──────────────────────────────────────────────

def _start_at(repo, *, run_id, trigger, started_at):
    return repo.start_run(
        "test", run_id=run_id, def_key="triage", def_version="1",
        started_at=started_at, trigger_msg_id=trigger,
        ctx='{"threadId":"t1"}', trace=False, max_steps=12,
    )


def test_find_runs_for_thread_empty_when_none(wf_repo):
    _seed_run_fixtures(wf_repo)
    assert wf_repo.find_runs_for_thread("test", thread_id="t1") == []


def test_find_runs_for_thread_returns_run_fields(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)

    runs = wf_repo.find_runs_for_thread("test", thread_id="t1")

    assert len(runs) == 1
    run = runs[0]
    assert run["runId"] == "r1"
    assert run["status"] == "running"
    assert run["defKey"] == "triage"
    assert run["defVersion"] == "1"
    assert run["startedAt"] == 1000
    assert run["endedAt"] is None


def test_find_runs_for_thread_orders_newest_first(wf_repo):
    _seed_run_fixtures(wf_repo)
    wf_repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="trig2", author_id="u1",
        text="another", role="user", created_at=130,
    )
    _start_at(wf_repo, run_id="r1", trigger="trig1", started_at=1000)
    _start_at(wf_repo, run_id="r2", trigger="trig2", started_at=2000)

    runs = wf_repo.find_runs_for_thread("test", thread_id="t1")

    assert [r["runId"] for r in runs] == ["r2", "r1"]


def test_find_runs_for_thread_respects_limit(wf_repo):
    _seed_run_fixtures(wf_repo)
    wf_repo.post_subsequent_message(
        "test", thread_id="t1", msg_id="trig2", author_id="u1",
        text="another", role="user", created_at=130,
    )
    _start_at(wf_repo, run_id="r1", trigger="trig1", started_at=1000)
    _start_at(wf_repo, run_id="r2", trigger="trig2", started_at=2000)

    runs = wf_repo.find_runs_for_thread("test", thread_id="t1", limit=1)

    assert [r["runId"] for r in runs] == ["r2"]


def test_find_runs_for_thread_ignores_other_threads(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo)
    assert wf_repo.find_runs_for_thread("test", thread_id="other") == []


# ── §12.15 read_recent_post_success ──────────────────────────────────────────

def _seed_trigger(repo, msg_id, *, created_at=130):
    """A second (or later) trigger message in thread t1, for a second run."""
    repo.post_subsequent_message(
        "test", thread_id="t1", msg_id=msg_id, author_id="u1",
        text="another", role="user", created_at=created_at,
    )


def _post_success_run(
    repo, *, run_id, trigger, started_at, terminal, posted,
):
    """Seed one WorkflowRun for the post-success sample.

    `posted=True` advances the run one step (creating a StepRun linked to it
    via HAS_STEP_RUN, reusing the existing `_advance` helper) and links a
    `StepRun -[:PRODUCED]-> Message` edge (D2) before reaching `terminal`
    status; `posted=False` reaches `terminal` with no StepRun/PRODUCED edge at
    all. `terminal` is one of `"done"`, `"failed"`, `"waiting"`, or `"running"`
    (the run is simply left as `_start_at` leaves it).
    """
    _start_at(repo, run_id=run_id, trigger=trigger, started_at=started_at)
    if posted:
        _advance(repo, run_id=run_id, step_run_id=f"{run_id}-sr", to_step="research")
        repo.link_step_emission("test", step_run_id=f"{run_id}-sr", msg_id=trigger)
    if terminal == "done":
        repo.complete_run("test", run_id=run_id, ended_at=started_at + 10)
    elif terminal == "failed":
        repo.fail_run("test", run_id=run_id, ended_at=started_at + 10, ctx="{}")
    elif terminal == "waiting":
        repo.suspend_run("test", run_id=run_id, thread_id="t1")
    # terminal == "running": nothing further — _start_at already leaves it running.


def test_read_recent_post_success_all_posted(wf_repo):
    _seed_run_fixtures(wf_repo)
    _seed_trigger(wf_repo, "trig2")
    _post_success_run(
        wf_repo, run_id="r1", trigger="trig1", started_at=1000,
        terminal="done", posted=True,
    )
    _post_success_run(
        wf_repo, run_id="r2", trigger="trig2", started_at=2000,
        terminal="failed", posted=True,
    )

    res = wf_repo.read_recent_post_success(
        "test", def_key="triage", def_version="1", limit=20
    )

    assert res == {"sampleSize": 2, "postedCount": 2}
    # the query's raw `postedCount` is a Python float (sum() over a CASE) —
    # the repository must cast it to a clean int before returning.
    assert isinstance(res["postedCount"], int)
    assert isinstance(res["sampleSize"], int)


def test_read_recent_post_success_some_posted(wf_repo):
    _seed_run_fixtures(wf_repo)
    _seed_trigger(wf_repo, "trig2")
    _post_success_run(
        wf_repo, run_id="r1", trigger="trig1", started_at=1000,
        terminal="done", posted=True,
    )
    _post_success_run(
        wf_repo, run_id="r2", trigger="trig2", started_at=2000,
        terminal="done", posted=False,
    )

    res = wf_repo.read_recent_post_success(
        "test", def_key="triage", def_version="1", limit=20
    )

    assert res == {"sampleSize": 2, "postedCount": 1}


def test_read_recent_post_success_none_posted(wf_repo):
    _seed_run_fixtures(wf_repo)
    _seed_trigger(wf_repo, "trig2")
    _post_success_run(
        wf_repo, run_id="r1", trigger="trig1", started_at=1000,
        terminal="done", posted=False,
    )
    _post_success_run(
        wf_repo, run_id="r2", trigger="trig2", started_at=2000,
        terminal="failed", posted=False,
    )

    res = wf_repo.read_recent_post_success(
        "test", def_key="triage", def_version="1", limit=20
    )

    assert res == {"sampleSize": 2, "postedCount": 0}


def test_read_recent_post_success_zero_runs_is_no_data_not_an_error(wf_repo):
    _seed_run_fixtures(wf_repo)  # thread/snapshot exist, but no run was ever started

    res = wf_repo.read_recent_post_success(
        "test", def_key="triage", def_version="1", limit=20
    )

    assert res == {"sampleSize": 0, "postedCount": 0}


def test_read_recent_post_success_respects_limit(wf_repo):
    _seed_run_fixtures(wf_repo)
    _seed_trigger(wf_repo, "trig2")
    _seed_trigger(wf_repo, "trig3")
    # oldest run is unposted; the two newest are posted — a limit of 2 must
    # truncate to just the newest two and drop the unposted oldest one.
    _post_success_run(
        wf_repo, run_id="r1", trigger="trig1", started_at=1000,
        terminal="done", posted=False,
    )
    _post_success_run(
        wf_repo, run_id="r2", trigger="trig2", started_at=2000,
        terminal="done", posted=True,
    )
    _post_success_run(
        wf_repo, run_id="r3", trigger="trig3", started_at=3000,
        terminal="done", posted=True,
    )

    res = wf_repo.read_recent_post_success(
        "test", def_key="triage", def_version="1", limit=2
    )

    assert res == {"sampleSize": 2, "postedCount": 2}


def test_read_recent_post_success_excludes_waiting_and_running(wf_repo):
    _seed_run_fixtures(wf_repo)
    _seed_trigger(wf_repo, "trig2")
    _seed_trigger(wf_repo, "trig3")
    _post_success_run(
        wf_repo, run_id="r1", trigger="trig1", started_at=1000,
        terminal="done", posted=True,
    )
    _post_success_run(
        wf_repo, run_id="r2", trigger="trig2", started_at=2000,
        terminal="waiting", posted=False,
    )
    _post_success_run(
        wf_repo, run_id="r3", trigger="trig3", started_at=3000,
        terminal="running", posted=False,
    )

    res = wf_repo.read_recent_post_success(
        "test", def_key="triage", def_version="1", limit=20
    )

    # only r1 (done) counts — r2 (waiting) and r3 (running) are non-terminal
    assert res == {"sampleSize": 1, "postedCount": 1}


# ── §12.10 / §12.11 trace write & read ───────────────────────────────────────

def test_append_trace_event_then_read_trace_round_trips(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo, trace=True)
    _advance(wf_repo, step_run_id="sr1", to_step="research")

    wf_repo.append_trace_event(
        "test", step_run_id="sr1", trace_id="te1", seq=0,
        kind="guard_judgment", at=1500, payload="verdict=true; because ...",
    )
    wf_repo.append_trace_event(
        "test", step_run_id="sr1", trace_id="te2", seq=1,
        kind="node_rationale", at=1501, payload="posted a clarifying question",
    )

    events = wf_repo.read_trace("test", run_id="r1")

    assert [e["kind"] for e in events] == ["guard_judgment", "node_rationale"]
    assert events[0]["payload"] == "verdict=true; because ..."
    assert events[0]["stepKey"] == "intake"
    assert [e["seq"] for e in events] == [0, 1]


def test_append_trace_event_zero_rows_when_step_run_missing(wf_repo):
    _seed_run_fixtures(wf_repo)
    _start(wf_repo, trace=True)
    assert wf_repo.append_trace_event(
        "test", step_run_id="ghost", trace_id="te1", seq=0,
        kind="node_rationale", at=1500, payload="x",
    ) is None


def test_read_trace_empty_for_non_debug_run(wf_repo):
    # AC-5 negative half at the query level: a run with no TRACED edges reads empty
    _seed_run_fixtures(wf_repo)
    _start(wf_repo, trace=False)
    _advance(wf_repo, step_run_id="sr1", to_step="research")
    assert wf_repo.read_trace("test", run_id="r1") == []


# ── typed errors exist and are re-exported by services ───────────────────────

def test_workflow_run_errors_are_exception_subclasses():
    assert issubclass(WorkflowRunNotFoundError, Exception)
    assert issubclass(StepBudgetExceededError, Exception)


def test_list_snapshots_empty_when_none(wf_repo):
    assert wf_repo.list_snapshots("test") == []


# ── §15 Product catalog (K-052 M6) ────────────────────────────────────────────
#
# `Product` lives in the global `reference` graph (plan §3.1) — no repository
# write method exists (the catalog is seed-script-only, `scripts/seed_catalog.sh`,
# not built in this cluster), so fixtures are seeded with a raw, test-only write,
# the same posture `_add_to_channel` already takes for an un-wrapped write. The
# `wf_repo` fixture wipes `reference`'s node data (schema — the Product indexes +
# UNIQUE constraint from `bootstrap_schema.sh` — survives), so each test starts
# from an empty-but-schemaed catalog.

# `category` is seeded Title-Case (matching seed_catalog.sh's real catalog, e.g.
# "Audio") while `categoryNormalized` is its case-folded form — deliberately
# mismatched casing so a test that queries with a differently-cased `category`
# argument (normalized internally by `Repository.filter_products`) proves the
# match runs against `categoryNormalized`, not a case-sensitive compare on the
# raw `category` (K-052 M6 live-discovered fix, 2026-08-28).
_CATALOG_FIXTURE = [
    {"productId": "prod1", "name": "Wireless Mouse", "nameNormalized": "wireless mouse",
     "category": "Accessories", "categoryNormalized": "accessories", "price": 25.0},
    {"productId": "prod2", "name": "Bluetooth Speaker", "nameNormalized": "bluetooth speaker",
     "category": "Audio", "categoryNormalized": "audio", "price": 89.99},
    {"productId": "prod3", "name": "4K Monitor", "nameNormalized": "4k monitor",
     "category": "Displays", "categoryNormalized": "displays", "price": 349.0},
    {"productId": "prod4", "name": "USB-C Hub", "nameNormalized": "usb-c hub",
     "category": "Accessories", "categoryNormalized": "accessories", "price": 45.5},
    {"productId": "prod5", "name": "Noise Cancelling Headphones",
     "nameNormalized": "noise cancelling headphones", "category": "Audio",
     "categoryNormalized": "audio", "price": 199.99},
]


def _seed_products(conn, rows=_CATALOG_FIXTURE):
    """Raw Product write against `reference` (test-only; no repository method
    exists — the catalog is seed-script-write-only, QUERIES.md §15).
    """
    db.reference_graph(conn).query(
        "UNWIND $rows AS row "
        "CREATE (:Product {productId: row.productId, name: row.name, "
        "                  nameNormalized: row.nameNormalized, "
        "                  category: row.category, "
        "                  categoryNormalized: row.categoryNormalized, "
        "                  price: row.price})",
        {"rows": rows},
    )


def test_lookup_product_exact_name_hit(conn, wf_repo):
    _seed_products(conn)

    row = wf_repo.lookup_product(name_normalized="bluetooth speaker")

    assert row == {"name": "Bluetooth Speaker", "category": "Audio", "price": 89.99}


def test_lookup_product_abstains_when_absent(conn, wf_repo):
    _seed_products(conn)

    assert wf_repo.lookup_product(name_normalized="nonexistent gadget") is None


def test_lookup_product_productid_constraint_blocks_duplicate(conn, wf_repo):
    _seed_products(conn)

    with pytest.raises(ResponseError, match="unique constraint violation"):
        db.reference_graph(conn).query(
            "CREATE (:Product {productId: 'prod1', name: 'Imposter'})"
        )


def test_filter_products_unfiltered_lists_whole_catalog_price_ascending(conn, wf_repo):
    _seed_products(conn)

    rows = wf_repo.filter_products(
        category=None, min_price=None, max_price=None, limit=20,
    )

    assert [r["name"] for r in rows] == [
        "Wireless Mouse", "USB-C Hub", "Bluetooth Speaker",
        "Noise Cancelling Headphones", "4K Monitor",
    ]


def test_filter_products_category_only(conn, wf_repo):
    _seed_products(conn)

    rows = wf_repo.filter_products(
        category="accessories", min_price=None, max_price=None, limit=20,
    )

    assert {r["name"] for r in rows} == {"Wireless Mouse", "USB-C Hub"}


def test_filter_products_category_is_case_and_whitespace_insensitive(conn, wf_repo):
    """K-052 M6 live-discovered fix (2026-08-28): an LLM tool call lowercased a
    category argument ("audio") against the seeded Title-Case "Audio" and the
    old exact `p.category = $category` comparison silently returned zero rows.
    `filter_products` now normalizes the caller's `category` (case-fold +
    whitespace-collapse, `extraction.normalize_name`) and matches it against
    `Product.categoryNormalized` instead.
    """
    _seed_products(conn)

    for variant in ("audio", "Audio", "AUDIO", "  audio  ", "  AUDIO"):
        rows = wf_repo.filter_products(
            category=variant, min_price=None, max_price=None, limit=20,
        )
        assert {r["name"] for r in rows} == {
            "Bluetooth Speaker", "Noise Cancelling Headphones",
        }, f"category={variant!r} did not match the seeded 'Audio' rows"


def test_filter_products_price_range_only(conn, wf_repo):
    _seed_products(conn)

    rows = wf_repo.filter_products(
        category=None, min_price=50.0, max_price=250.0, limit=20,
    )

    assert {r["name"] for r in rows} == {"Bluetooth Speaker", "Noise Cancelling Headphones"}


def test_filter_products_category_and_price_combined(conn, wf_repo):
    _seed_products(conn)

    rows = wf_repo.filter_products(
        category="audio", min_price=100.0, max_price=None, limit=20,
    )

    assert [r["name"] for r in rows] == ["Noise Cancelling Headphones"]


def test_filter_products_abstains_when_nothing_matches(conn, wf_repo):
    _seed_products(conn)

    assert wf_repo.filter_products(
        category="nonexistent", min_price=None, max_price=None, limit=20,
    ) == []


def test_filter_products_respects_limit(conn, wf_repo):
    _seed_products(conn)

    rows = wf_repo.filter_products(
        category=None, min_price=None, max_price=None, limit=2,
    )

    assert len(rows) == 2
    assert [r["name"] for r in rows] == ["Wireless Mouse", "USB-C Hub"]  # cheapest 2


def test_filter_products_empty_catalog_returns_empty(wf_repo):
    # No _seed_products call — `reference` has no Product nodes at all.
    assert wf_repo.filter_products(
        category=None, min_price=None, max_price=None, limit=20,
    ) == []


# ── §16 Cart / Order (K-053 M6) ──────────────────────────────────────────────
#
# Workspace-scoped (`ws:test`, via `repo`/`conn`) — no `reference` graph is
# touched by anything in this section, so `repo`/`conn` (not `wf_repo`) is the
# right fixture throughout. Each test wraps one `docs/plans/
# workflow-cart-and-totals-graph.md` (v2) Cypher shape 1:1, named per test.


def test_ensure_customer_fresh_creates_then_reensure_is_quiet_noop(repo, conn):
    first = repo.ensure_customer("test", customer_id="cust1", now=100)
    assert first == {"customerId": "cust1", "createdAt": 100}

    second = repo.ensure_customer("test", customer_id="cust1", now=200)
    assert second == {"customerId": "cust1", "createdAt": 100}  # createdAt unchanged

    rows = _probe(conn, "MATCH (c:Customer {customerId: 'cust1'}) RETURN count(c)")
    assert rows[0][0] == 1


def test_ensure_cart_after_ensure_customer_creates_cart(repo, conn):
    repo.ensure_customer("test", customer_id="cust1", now=100)

    first = repo.ensure_cart("test", customer_id="cust1", now=100)
    assert first == {"customerId": "cust1", "createdAt": 100}

    second = repo.ensure_cart("test", customer_id="cust1", now=200)
    assert second == {"customerId": "cust1", "createdAt": 100}  # createdAt unchanged

    rows = _probe(
        conn,
        "MATCH (:Customer {customerId: 'cust1'})-[:HAS_CART]->(cart:Cart) "
        "RETURN count(cart)",
    )
    assert rows[0][0] == 1


def test_ensure_cart_missing_customer_is_a_noop_returning_none(repo):
    assert repo.ensure_cart("test", customer_id="ghost", now=100) is None


def test_add_to_cart_without_a_cart_yet_is_a_noop_returning_none(repo):
    # Regression for `analyst`'s MAJOR finding (`docs/reviews/
    # workflow-cart-and-totals.md`): a brand-new customerId with no prior
    # Customer/Cart node — this repository method alone must not silently
    # write nothing while looking like success; it must report the no-op.
    assert repo.add_to_cart(
        "test", customer_id="cust1", product_id="prod1", qty=2, now=100
    ) is None


def test_add_to_cart_merges_and_increments_not_duplicates(repo, conn):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)

    first = repo.add_to_cart(
        "test", customer_id="cust1", product_id="prod1", qty=2, now=100
    )
    assert first == {"productId": "prod1", "quantity": 2}

    second = repo.add_to_cart(
        "test", customer_id="cust1", product_id="prod1", qty=2, now=200
    )
    assert second == {"productId": "prod1", "quantity": 4}

    rows = _probe(conn, "MATCH (i:CartItem {productId: 'prod1'}) RETURN count(i)")
    assert rows[0][0] == 1


def test_add_to_cart_second_product_added_cleanly_alongside_first(repo):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod1", qty=1, now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod2", qty=3, now=100)

    cart = repo.read_cart("test", customer_id="cust1")
    assert {(row["productId"], row["quantity"]) for row in cart} == {
        ("prod1", 1), ("prod2", 3),
    }


def test_adjust_cart_item_decrement_updates_in_place(repo, conn):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod1", qty=4, now=100)

    result = repo.adjust_cart_item(
        "test", customer_id="cust1", product_id="prod1", qty=1, now=200
    )
    assert result == {"quantity": 3, "removed": False}

    rows = _probe(conn, "MATCH (i:CartItem {productId: 'prod1'}) RETURN i.quantity")
    assert rows[0][0] == 3


def test_adjust_cart_item_decrement_to_zero_deletes_node_and_edge(repo, conn):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod1", qty=3, now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod2", qty=1, now=100)

    result = repo.adjust_cart_item(
        "test", customer_id="cust1", product_id="prod1", qty=3, now=200
    )
    assert result == {"quantity": 0, "removed": True}

    rows = _probe(conn, "MATCH (i:CartItem {productId: 'prod1'}) RETURN count(i)")
    assert rows[0][0] == 0
    # the other product's line is untouched
    rows = _probe(conn, "MATCH (i:CartItem {productId: 'prod2'}) RETURN i.quantity")
    assert rows[0][0] == 1


def test_adjust_cart_item_over_removal_deletes_rather_than_going_negative(repo):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod1", qty=2, now=100)

    result = repo.adjust_cart_item(
        "test", customer_id="cust1", product_id="prod1", qty=99, now=200
    )
    assert result == {"quantity": -97, "removed": True}
    assert repo.read_cart("test", customer_id="cust1") == []


def test_adjust_cart_item_no_such_line_is_a_noop_returning_none(repo):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)

    assert repo.adjust_cart_item(
        "test", customer_id="cust1", product_id="never-added", qty=1, now=100
    ) is None


def test_read_cart_empty_when_no_cart_yet(repo):
    assert repo.read_cart("test", customer_id="ghost") == []


def test_read_cart_orders_by_added_at(repo):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod2", qty=1, now=200)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod1", qty=1, now=100)

    cart = repo.read_cart("test", customer_id="cust1")
    assert [row["productId"] for row in cart] == ["prod1", "prod2"]


def test_clear_cart_deletes_all_items(repo, conn):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod1", qty=1, now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod2", qty=1, now=100)

    repo.clear_cart("test", customer_id="cust1")

    assert repo.read_cart("test", customer_id="cust1") == []
    rows = _probe(conn, "MATCH (:Cart {customerId: 'cust1'}) RETURN count(*)")
    assert rows[0][0] == 1  # the Cart node itself survives, only its items go


def test_clear_cart_on_empty_cart_is_a_plain_noop(repo):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)
    repo.clear_cart("test", customer_id="cust1")  # must not raise
    assert repo.read_cart("test", customer_id="cust1") == []


_ORDER_LINES = [
    {"productId": "prod1", "name": "Widget", "unitPrice": 10.0, "quantity": 2,
     "lineTotal": 20.0},
    {"productId": "prod2", "name": "Gadget", "unitPrice": 5.0, "quantity": 3,
     "lineTotal": 15.0},
]


def test_place_order_snapshots_lines_and_clears_cart(repo, conn):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod1", qty=2, now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod2", qty=3, now=100)

    result = repo.place_order(
        "test", customer_id="cust1", order_id="order1", now=300, lines=_ORDER_LINES,
    )
    assert result == {"created": True, "lineCount": 2}

    order = repo.get_order("test", order_id="order1")
    assert order["orderId"] == "order1"
    assert order["status"] == "placed"
    assert order["total"] == 35.0
    assert {(l["productId"], l["unitPrice"], l["quantity"], l["lineTotal"])
            for l in order["lines"]} == {
        ("prod1", 10.0, 2, 20.0), ("prod2", 5.0, 3, 15.0),
    }
    assert repo.read_cart("test", customer_id="cust1") == []


def test_place_order_retry_with_same_order_id_is_idempotent_noop(repo, conn):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod1", qty=2, now=100)

    lines = [_ORDER_LINES[0]]
    first = repo.place_order(
        "test", customer_id="cust1", order_id="order1", now=300, lines=lines,
    )
    assert first == {"created": True, "lineCount": 1}

    # retry: cart already cleared, but the same orderId must not duplicate
    retry = repo.place_order(
        "test", customer_id="cust1", order_id="order1", now=400, lines=lines,
    )
    assert retry == {"created": False, "lineCount": 1}

    rows = _probe(conn, "MATCH (o:Order {orderId: 'order1'}) RETURN count(o)")
    assert rows[0][0] == 1
    rows = _probe(
        conn, "MATCH (:Order {orderId: 'order1'})-[:HAS_LINE]->(l) RETURN count(l)"
    )
    assert rows[0][0] == 1


def test_place_order_with_empty_lines_creates_a_zero_line_order(repo):
    repo.ensure_customer("test", customer_id="cust1", now=100)

    result = repo.place_order(
        "test", customer_id="cust1", order_id="order-empty", now=300, lines=[],
    )
    assert result == {"created": True, "lineCount": 0}

    order = repo.get_order("test", order_id="order-empty")
    assert order["status"] == "placed"
    assert order["lines"] == []
    assert order["total"] == 0.0  # sum() over an empty/all-null set is 0, not null


def test_place_order_missing_customer_raises_not_silent_noop(repo):
    with pytest.raises(RuntimeError):
        repo.place_order(
            "test", customer_id="ghost", order_id="order1", now=300,
            lines=_ORDER_LINES,
        )


def test_get_order_none_when_absent(repo):
    assert repo.get_order("test", order_id="ghost") is None


def _place_test_order(repo, *, order_id="order1"):
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.place_order(
        "test", customer_id="cust1", order_id=order_id, now=300, lines=_ORDER_LINES,
    )


def test_fulfill_order_from_placed_succeeds(repo):
    _place_test_order(repo)

    result = repo.fulfill_order("test", order_id="order1", now=400)
    assert result == {"orderId": "order1", "status": "fulfilled"}
    assert repo.get_order("test", order_id="order1")["status"] == "fulfilled"


def test_fulfill_order_not_placed_is_a_noop_returning_none(repo):
    _place_test_order(repo)
    repo.fulfill_order("test", order_id="order1", now=400)

    # already fulfilled — a second fulfill attempt matches zero rows
    assert repo.fulfill_order("test", order_id="order1", now=500) is None
    assert repo.get_order("test", order_id="order1")["status"] == "fulfilled"


def test_deliver_order_from_fulfilled_succeeds(repo):
    _place_test_order(repo)
    repo.fulfill_order("test", order_id="order1", now=400)

    result = repo.deliver_order("test", order_id="order1", now=500)
    assert result == {"orderId": "order1", "status": "delivered"}
    assert repo.get_order("test", order_id="order1")["status"] == "delivered"


def test_deliver_order_not_yet_fulfilled_is_a_noop_returning_none(repo):
    _place_test_order(repo)

    # still just 'placed' — deliver requires 'fulfilled' first
    assert repo.deliver_order("test", order_id="order1", now=500) is None
    assert repo.get_order("test", order_id="order1")["status"] == "placed"


def test_cancel_order_from_placed_succeeds(repo):
    _place_test_order(repo)

    result = repo.cancel_order("test", order_id="order1", now=400)
    assert result == {"orderId": "order1", "status": "cancelled"}
    assert repo.get_order("test", order_id="order1")["status"] == "cancelled"


def test_cancel_order_after_fulfilled_is_blocked_ac8(repo):
    _place_test_order(repo)
    repo.fulfill_order("test", order_id="order1", now=400)

    # AC-8: cannot cancel once fulfilled — matches zero rows, nothing changes
    assert repo.cancel_order("test", order_id="order1", now=500) is None
    assert repo.get_order("test", order_id="order1")["status"] == "fulfilled"


def test_order_lifecycle_end_to_end_placed_fulfilled_delivered(repo):
    _place_test_order(repo)

    assert repo.get_order("test", order_id="order1")["status"] == "placed"
    repo.fulfill_order("test", order_id="order1", now=400)
    assert repo.get_order("test", order_id="order1")["status"] == "fulfilled"
    repo.deliver_order("test", order_id="order1", now=500)
    assert repo.get_order("test", order_id="order1")["status"] == "delivered"


def test_customer_and_cart_persist_across_repository_instances_ac3(repo):
    """AC-3 (same-customer cross-conversation persistence), at the repository
    layer: a second `Repository` built over a fresh connection sees the first
    one's writes — durability is the graph, not any in-process object (the
    same fact two separate `Thread`s/service calls will rely on at the
    service layer, later cluster).
    """
    repo.ensure_customer("test", customer_id="cust1", now=100)
    repo.ensure_cart("test", customer_id="cust1", now=100)
    repo.add_to_cart("test", customer_id="cust1", product_id="prod1", qty=1, now=100)

    repo2 = Repository(db.connect())
    cart = repo2.read_cart("test", customer_id="cust1")
    assert cart == [{"productId": "prod1", "quantity": 1, "addedAt": 100}]
