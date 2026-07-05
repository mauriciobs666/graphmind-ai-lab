"""Integration tests for the repository layer against a live `ws:test` graph.

Each test wraps one repository method 1:1 with a verified `QUERIES.md` query.
A few structural regression tests probe the graph directly (`_probe`) — the
write-path defects they pin (NEXT self-loops, duplicate HEADs) are invisible
through the public read methods by design.
"""

from __future__ import annotations

import pytest

from falkorchat import db
from falkorchat.repository import MemberIdCollisionError


def _probe(conn, cypher: str):
    """Raw structural read against ws:test (test-only; app Cypher stays in repository.py)."""
    return db.workspace_graph(conn, "test").ro_query(cypher).result_set

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
