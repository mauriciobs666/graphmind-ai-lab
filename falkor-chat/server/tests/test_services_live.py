"""A few live service checks (real repository, live `ws:test`).

Unit coverage of the dispatch logic lives in `test_services.py`; these verify
the service composes correctly against real Cypher.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor

from falkorchat import db
from falkorchat.config import CallContext
from falkorchat.services import Services

CTX = CallContext(ws="test", actor="u1")


def _svc(repo, ids=None):
    # A monotonically increasing clock — mirrors real server time so that
    # sequential posts get distinct, ordered `createdAt` values.
    clock = itertools.count(1000)
    ids = iter(ids or [f"g{n}" for n in range(1, 50)])
    return Services(repo, clock=lambda: next(clock), id_gen=lambda: next(ids))


def test_post_then_read_roundtrip_and_cursor_advance(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _svc(repo, ids=["c1", "t1", "m1", "m2"])

    svc.create_channel(CTX, name="general")
    svc.create_thread(CTX, channel_id="c1", title="hi")
    svc.post_message(CTX, thread_id="t1", text="first")
    svc.post_message(CTX, thread_id="t1", text="second")

    # first read (no cursor) sees both, then advances the cursor to now=1000
    rows = svc.read_messages(CTX, thread_id="t1", advance=True)
    assert [r["text"] for r in rows] == ["first", "second"]

    # a second read starts from the advanced cursor → nothing new
    rows2 = svc.read_messages(CTX, thread_id="t1", advance=True)
    assert rows2 == []


def test_mention_prioritised_and_validated_live(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.ensure_user("test", user_id="u2", display_name="Bob")
    ctx_bob = CallContext(ws="test", actor="u2")
    svc = _svc(repo, ids=["c1", "t1", "m1", "m2"])

    svc.create_channel(CTX, name="general")
    svc.create_thread(CTX, channel_id="c1", title="hi")
    svc.post_message(CTX, thread_id="t1", text="plain")
    svc.post_message(CTX, thread_id="t1", text="hey bob", mentions=["u2"])

    # Bob reads from epoch: chronological order, his mention flagged
    rows = svc.read_messages(ctx_bob, thread_id="t1", since=0, advance=False)
    assert [r["text"] for r in rows] == ["plain", "hey bob"]
    assert [r["isMention"] for r in rows] == [False, True]


def test_cursor_pagination_is_lossless_under_limit(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _svc(repo, ids=["c1", "t1", "m1", "m2", "m3"])

    svc.create_channel(CTX, name="general")
    svc.create_thread(CTX, channel_id="c1", title="hi")
    for text in ("one", "two", "three"):
        svc.post_message(CTX, thread_id="t1", text=text)

    # page 1: the two earliest; the cursor must advance only to what was delivered
    page1 = svc.read_messages(CTX, thread_id="t1", limit=2, advance=True)
    assert [r["text"] for r in page1] == ["one", "two"]

    # page 2: the truncated remainder — not lost to a clock-advanced cursor
    page2 = svc.read_messages(CTX, thread_id="t1", limit=2, advance=True)
    assert [r["text"] for r in page2] == ["three"]


def test_concurrent_posts_keep_one_head_one_tail_contiguous_chain(repo, conn):
    """K-007 concurrency hammer (scaled for CI): 8 workers race post_message on a
    fresh thread through one shared Services — the v2 status dispatch must yield
    exactly one HEAD/TAIL and a contiguous chain, with no errors."""
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = Services(repo)  # real clock + uuid ids; monotonic stamp is per-process
    ch = svc.create_channel(CTX, name="general")
    th = svc.create_thread(CTX, channel_id=ch["channelId"], title="race")
    thread_id = th["threadId"]

    workers = 8
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(svc.post_message, CTX, thread_id=thread_id, text=f"msg {n}")
            for n in range(workers)
        ]
        results = [f.result() for f in futures]  # raises if any worker errored

    assert len(results) == workers
    g = db.workspace_graph(conn, "test")
    [[heads, tails]] = g.ro_query(
        "MATCH (t:Thread {threadId: $tid}) "
        "OPTIONAL MATCH (t)-[h:HEAD]->() OPTIONAL MATCH (t)-[tl:TAIL]->() "
        "RETURN count(DISTINCT h), count(DISTINCT tl)",
        {"tid": thread_id},
    ).result_set
    assert (heads, tails) == (1, 1)
    chain = svc.read_thread(CTX, thread_id=thread_id)
    assert len(chain) == workers  # contiguous chain — every racer's write landed
    assert {r["msgId"] for r in chain} == {r["msgId"] for r in results}


def test_agent_actor_posts_as_assistant_end_to_end(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.ensure_agent("test", agent_id="a1", name="Bot")
    ctx_agent = CallContext(ws="test", actor="a1")
    svc = _svc(repo, ids=["c1", "t1", "m1"])

    svc.create_channel(CTX, name="general")
    svc.create_thread(CTX, channel_id="c1", title="hi")
    svc.post_message(ctx_agent, thread_id="t1", text="I can help with that")

    rows = svc.read_messages(CTX, thread_id="t1", since=0, advance=False)
    assert [r["authorId"] for r in rows] == ["a1"]
    assert [r["role"] for r in rows] == ["assistant"]  # derived from the Agent label
