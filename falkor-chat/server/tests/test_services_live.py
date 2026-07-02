"""A few live service checks (real repository, live `ws:test`).

Unit coverage of the dispatch logic lives in `test_services.py`; these verify
the service composes correctly against real Cypher.
"""

from __future__ import annotations

import itertools

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
