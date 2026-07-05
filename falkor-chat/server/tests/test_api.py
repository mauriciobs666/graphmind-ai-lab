"""REST contract tests via FastAPI TestClient against the live `ws:test` graph.

The app is built with `mount_mcp=False` (REST is what's under test; keeping the
FastMCP session manager out avoids its run-once-per-instance constraint) and a
context override pinning the tenant to `ws:test`.
"""

from __future__ import annotations

import itertools

import pytest
from fastapi.testclient import TestClient

from falkorchat.app import create_app
from falkorchat.config import CallContext
from falkorchat.repository import Repository
from falkorchat.services import Services


@pytest.fixture()
def client(conn):
    services = Services(Repository(conn))
    Repository(conn).ensure_user("test", user_id="u1", display_name="Alice")
    app = create_app(
        services,
        context_provider=lambda: CallContext(ws="test", actor="u1"),
        mount_mcp=False,
    )
    return TestClient(app)


def _new_channel(client, name="general") -> str:
    return client.post("/channels", json={"name": name}).json()["channelId"]


def _new_thread(client, channel_id, title="hi") -> str:
    return client.post(
        f"/channels/{channel_id}/threads", json={"title": title}
    ).json()["threadId"]


def test_create_and_list_channels(client):
    r = client.post("/channels", json={"name": "general"})
    assert r.status_code == 201
    assert r.json()["name"] == "general"

    listed = client.get("/channels").json()
    assert [c["name"] for c in listed] == ["general"]


def test_create_and_list_threads(client):
    cid = _new_channel(client)
    r = client.post(f"/channels/{cid}/threads", json={"title": "topic"})
    assert r.status_code == 201

    listed = client.get(f"/channels/{cid}/threads").json()
    assert [t["title"] for t in listed] == ["topic"]


def test_post_and_read_messages(client):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    r = client.post(f"/threads/{tid}/messages", json={"text": "hello"})
    assert r.status_code == 201
    mid = r.json()["msgId"]

    msgs = client.get(f"/threads/{tid}/messages").json()
    assert [m["text"] for m in msgs] == ["hello"]

    # msgId is workspace-global; the flat lookup resolves it without a thread scope.
    one = client.get(f"/messages/{mid}")
    assert one.status_code == 200
    assert one.json()["text"] == "hello"


def test_post_message_mention_parity(client, conn):
    # seed a mention target in the same live ws:test graph
    Repository(conn).ensure_user("test", user_id="u2", display_name="Bob")

    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    ok = client.post(f"/threads/{tid}/messages", json={"text": "hi", "mentions": ["u2"]})
    assert ok.status_code == 201

    bad = client.post(f"/threads/{tid}/messages", json={"text": "x", "mentions": ["ghost"]})
    assert bad.status_code == 400
    assert bad.json()["error"] == "UnknownMemberError"


def test_create_thread_unknown_channel_404(client):
    r = client.post("/channels/nope/threads", json={"title": "x"})
    assert r.status_code == 404


def test_post_to_missing_thread_404(client):
    r = client.post("/threads/ghost/messages", json={"text": "x"})
    assert r.status_code == 404


def test_get_missing_message_404(client):
    r = client.get("/messages/nope")
    assert r.status_code == 404


def test_search_returns_matching_messages(client):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)
    client.post(f"/threads/{tid}/messages", json={"text": "hello world"})
    client.post(f"/threads/{tid}/messages", json={"text": "goodbye moon"})

    r = client.get("/search", params={"q": "hello"})

    assert r.status_code == 200
    hits = r.json()
    assert [h["text"] for h in hits] == ["hello world"]


def test_thread_id_present_in_since_search_and_get_message(client):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)
    mid = client.post(
        f"/threads/{tid}/messages", json={"text": "hello navigation"}
    ).json()["msgId"]

    since_rows = client.get(f"/threads/{tid}/messages", params={"since": 0}).json()
    assert [m["threadId"] for m in since_rows] == [tid]

    hits = client.get("/search", params={"q": "navigation"}).json()
    assert [h["threadId"] for h in hits] == [tid]

    one = client.get(f"/messages/{mid}").json()
    assert one["threadId"] == tid  # route stays flat; the body carries the thread


def test_search_requires_q(client):
    r = client.get("/search")
    assert r.status_code == 422


def test_search_syntax_error_is_400_not_500(client):
    r = client.get("/search", params={"q": 'hello"unbalanced'})
    assert r.status_code == 400
    assert r.json()["error"] == "InvalidSearchQueryError"


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_input_size_bounds_are_422(client):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    assert client.post("/channels", json={"name": ""}).status_code == 422
    assert client.post("/channels", json={"name": "x" * 201}).status_code == 422
    assert client.post(
        f"/channels/{cid}/threads", json={"title": "x" * 201}
    ).status_code == 422
    assert client.post(
        f"/threads/{tid}/messages", json={"text": "x" * 8001}
    ).status_code == 422
    assert client.post(f"/threads/{tid}/messages", json={"text": ""}).status_code == 422


def test_list_limit_bounds_are_422(client):
    cid = _new_channel(client)
    assert client.get("/channels", params={"limit": 0}).status_code == 422
    assert client.get("/channels", params={"limit": 201}).status_code == 422
    assert client.get(
        f"/channels/{cid}/threads", params={"limit": 0}
    ).status_code == 422


def test_read_thread_since_limit_paginates(conn):
    # deterministic clock: same-ms createdAt ties would make `since >` pagination
    # ambiguous (the known ms-tie caveat) — not what this test is about
    clock = itertools.count(1000)
    services = Services(Repository(conn), clock=lambda: next(clock))
    Repository(conn).ensure_user("test", user_id="u1", display_name="Alice")
    client = TestClient(create_app(
        services,
        context_provider=lambda: CallContext(ws="test", actor="u1"),
        mount_mcp=False,
    ))

    cid = _new_channel(client)
    tid = _new_thread(client, cid)
    for text in ("one", "two", "three"):
        client.post(f"/threads/{tid}/messages", json={"text": text})

    # first page: earliest rows (chronological — the cursor-safe order)
    page = client.get(f"/threads/{tid}/messages", params={"limit": 2}).json()
    assert [m["text"] for m in page] == ["one", "two"]

    # next page: strictly after the last delivered createdAt
    rest = client.get(
        f"/threads/{tid}/messages",
        params={"since": page[-1]["createdAt"], "limit": 2},
    ).json()
    assert [m["text"] for m in rest] == ["three"]

    # no params keeps the full-read contract the web client relies on
    full = client.get(f"/threads/{tid}/messages").json()
    assert [m["text"] for m in full] == ["one", "two", "three"]
