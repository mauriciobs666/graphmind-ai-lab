"""Integration tests for the repository layer against a live `ws:test` graph.

Each test wraps one repository method 1:1 with a verified `QUERIES.md` query.
"""

from __future__ import annotations


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


def test_thread_has_head_false_before_first_message(repo):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x", created_at=110)

    assert repo.thread_has_head("test", thread_id="t1") is False


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


def test_read_thread_since_orders_mentions_first(repo):
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

    # mention sorts first despite being chronologically later
    assert [r["msgId"] for r in rows] == ["m2", "m1"]


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


# ── §9.3/§9.4 Read-cursor advance (monotonic) & read ───────────────────────────


def test_get_cursor_none_when_absent(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    assert repo.get_cursor("test", cursor_id="u1:t1") is None


def test_advance_cursor_then_get(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.advance_cursor("test", me_id="u1", thread_id="t1", cursor_id="u1:t1", now=300)

    assert repo.get_cursor("test", cursor_id="u1:t1") == 300


def test_advance_cursor_is_monotonic(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.advance_cursor("test", me_id="u1", thread_id="t1", cursor_id="u1:t1", now=300)
    repo.advance_cursor("test", me_id="u1", thread_id="t1", cursor_id="u1:t1", now=200)  # stale

    assert repo.get_cursor("test", cursor_id="u1:t1") == 300  # never moved backward

    repo.advance_cursor("test", me_id="u1", thread_id="t1", cursor_id="u1:t1", now=400)
    assert repo.get_cursor("test", cursor_id="u1:t1") == 400


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


def test_existing_members_filters_unknown_across_labels(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.ensure_agent("test", agent_id="a1", name="Bot")

    got = repo.existing_members("test", ids=["u1", "a1", "ghost"])

    assert got == {"u1", "a1"}


def test_existing_members_empty_input(repo):
    assert repo.existing_members("test", ids=[]) == set()


def test_thread_exists(repo):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    assert repo.thread_exists("test", thread_id="t1") is False
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x", created_at=110)
    assert repo.thread_exists("test", thread_id="t1") is True
