"""REST contract tests via FastAPI TestClient against the live `ws:test` graph.

The app is built with `mount_mcp=False` (REST is what's under test; keeping the
FastMCP session manager out avoids its run-once-per-instance constraint) and a
context override pinning the tenant to `ws:test`.
"""

from __future__ import annotations

import itertools

import pytest
from fastapi.testclient import TestClient

from falkorchat import db
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


# ── K-013 out-of-band wiring: every-message embedding + responder trigger ──────


class RecordingWorker:
    """Records embed_message calls scheduled on BackgroundTasks."""

    def __init__(self):
        self.calls: list[tuple] = []

    def embed_message(self, ws, *, msg_id, text):
        self.calls.append((ws, msg_id, text))
        return [0.0]


class RecordingResponder:
    """Records maybe_respond calls scheduled on BackgroundTasks."""

    def __init__(self):
        self.calls: list[dict] = []

    def maybe_respond(self, ctx, *, thread_id, msg_id, text, role, channel_id, mentions):
        self.calls.append(
            {
                "thread_id": thread_id, "msg_id": msg_id, "text": text,
                "role": role, "channel_id": channel_id, "mentions": mentions,
            }
        )
        return None


@pytest.fixture()
def wired(conn):
    """App wired with a recording embed-worker + responder (BackgroundTasks paths)."""
    services = Services(Repository(conn))
    Repository(conn).ensure_user("test", user_id="u1", display_name="Alice")
    Repository(conn).ensure_agent("test", agent_id="bot1", name="Bot")
    worker = RecordingWorker()
    responder = RecordingResponder()
    app = create_app(
        services,
        context_provider=lambda: CallContext(ws="test", actor="u1"),
        mount_mcp=False,
        embed_worker=worker,
        responder=responder,
    )
    return TestClient(app), worker, responder


def test_every_posted_message_is_scheduled_for_embedding(wired):
    client, worker, _ = wired
    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    r = client.post(f"/threads/{tid}/messages", json={"text": "plain user message"})
    assert r.status_code == 201
    mid = r.json()["msgId"]

    # BackgroundTasks run before the TestClient response returns
    assert (("test", mid, "plain user message")) in worker.calls


def test_posting_schedules_responder_with_posted_message(wired):
    client, _, responder = wired
    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    r = client.post(
        f"/threads/{tid}/messages", json={"text": "hey @bot", "mentions": ["bot1"]}
    )
    assert r.status_code == 201
    mid = r.json()["msgId"]

    assert len(responder.calls) == 1
    call = responder.calls[0]
    assert call["msg_id"] == mid
    assert call["thread_id"] == tid
    assert call["text"] == "hey @bot"
    assert call["role"] == "user"
    assert call["mentions"] == ["bot1"]


def test_plain_message_still_scheduled_but_responder_decides(wired):
    # The API delegates the trigger decision to the responder (it owns agent_id):
    # a non-mention post still schedules maybe_respond, which self-no-ops.
    client, _, responder = wired
    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    client.post(f"/threads/{tid}/messages", json={"text": "just chatting"})

    assert len(responder.calls) == 1
    assert responder.calls[0]["mentions"] == []


def test_embedding_a_message_never_posts_a_response(wired):
    # Embedding path and trigger path are separate: the worker is not the responder.
    client, worker, responder = wired
    cid = _new_channel(client)
    tid = _new_thread(client, cid)
    client.post(f"/threads/{tid}/messages", json={"text": "hi"})
    # one embed schedule, one responder schedule — neither crosses into the other
    assert len(worker.calls) == 1
    assert len(responder.calls) == 1


def test_default_app_has_no_wiring_and_posts_normally(client):
    # No embed_worker/responder configured → posting works, nothing scheduled.
    cid = _new_channel(client)
    tid = _new_thread(client, cid)
    r = client.post(f"/threads/{tid}/messages", json={"text": "hi"})
    assert r.status_code == 201


# ── K-023 trigger wiring: exactly one handler per request (trigger XOR responder) ─


class RecordingTrigger:
    """Records maybe_trigger calls scheduled on BackgroundTasks."""

    def __init__(self):
        self.calls: list[dict] = []

    def maybe_trigger(self, ctx, *, thread_id, msg_id, text, role, mentions):
        self.calls.append(
            {"thread_id": thread_id, "msg_id": msg_id, "text": text,
             "role": role, "mentions": mentions}
        )
        return None


@pytest.fixture()
def wired_wf(conn):
    """App wired with an embed-worker, a trigger AND a responder — the M3 shape.

    The trigger holds the responder for its fall-through, so the API must schedule the
    trigger and NOT the responder (exactly one handler per request).
    """
    services = Services(Repository(conn))
    Repository(conn).ensure_user("test", user_id="u1", display_name="Alice")
    worker = RecordingWorker()
    trigger = RecordingTrigger()
    responder = RecordingResponder()
    app = create_app(
        services,
        context_provider=lambda: CallContext(ws="test", actor="u1"),
        mount_mcp=False,
        embed_worker=worker,
        trigger=trigger,
        responder=responder,
    )
    return TestClient(app), worker, trigger, responder


def test_trigger_wired_schedules_trigger_not_responder(wired_wf):
    client, _, trigger, responder = wired_wf
    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    r = client.post(
        f"/threads/{tid}/messages", json={"text": "@bot help", "mentions": []}
    )
    assert r.status_code == 201
    mid = r.json()["msgId"]

    # exactly one handler fired — the trigger, never the responder (no double-response)
    assert len(trigger.calls) == 1
    assert trigger.calls[0]["msg_id"] == mid
    assert trigger.calls[0]["text"] == "@bot help"
    assert responder.calls == []


def test_trigger_wired_still_embeds_every_message(wired_wf):
    client, worker, trigger, _ = wired_wf
    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    r = client.post(f"/threads/{tid}/messages", json={"text": "hello"})
    mid = r.json()["msgId"]

    # embedding path is independent of the trigger path
    assert ("test", mid, "hello") in worker.calls
    assert len(trigger.calls) == 1


# ── §11 Workflow definitions & snapshots REST surface (M3 Slice 1) ──────────────


@pytest.fixture()
def wf_client(wf_repo):
    """TestClient whose repo has BOTH ws:test and `reference` wiped (plan F8)."""
    wf_repo.ensure_user("test", user_id="u1", display_name="Alice")
    app = create_app(
        Services(wf_repo),
        context_provider=lambda: CallContext(ws="test", actor="u1"),
        mount_mcp=False,
    )
    return TestClient(app)


DEF_BODY = {
    "key": "onboarding",
    "version": "1",
    "name": "Onboarding",
    "kind": "process",
    "steps": [
        # config is a **string** here — the REST shape (`schemas.py` types it `str`).
        # A `human` step must declare `waitsForHuman` (K-024 U2), and the publish
        # validator normalizes the string before checking it (M-7).
        {"key": "start", "type": "human", "config": '{"waitsForHuman": true}',
         "start": True},
        {"key": "done", "type": "message"},
    ],
    "transitions": [
        {"from": "start", "to": "done", "on": "submitted", "order": 0},
    ],
}


def test_publish_workflow_def_returns_201_and_counts(wf_client):
    r = wf_client.post("/workflow-defs", json=DEF_BODY)

    assert r.status_code == 201
    body = r.json()
    assert body["key"] == "onboarding"
    assert body["stepCount"] == 2
    assert body["transitionCount"] == 1


def test_publish_workflow_def_invalid_kind_is_400(wf_client):
    bad = {**DEF_BODY, "kind": "chatbot"}

    r = wf_client.post("/workflow-defs", json=bad)

    assert r.status_code == 400
    assert r.json()["error"] == "WorkflowDefSpecError"


def test_list_and_get_workflow_def(wf_client):
    wf_client.post("/workflow-defs", json=DEF_BODY)

    listed = wf_client.get("/workflow-defs").json()
    assert any(d["key"] == "onboarding" for d in listed)

    got = wf_client.get("/workflow-defs/onboarding").json()
    assert got["version"] == "1"
    assert got["name"] == "Onboarding"


def test_get_workflow_def_specific_version(wf_client):
    wf_client.post("/workflow-defs", json=DEF_BODY)

    got = wf_client.get("/workflow-defs/onboarding", params={"version": "1"}).json()

    assert got["version"] == "1"


def test_get_workflow_def_missing_is_404(wf_client):
    r = wf_client.get("/workflow-defs/ghost")

    assert r.status_code == 404


def test_materialize_def_creates_snapshot_and_lists_it(wf_client):
    wf_client.post("/workflow-defs", json=DEF_BODY)

    r = wf_client.post("/workflow-defs/onboarding/versions/1/materialize")
    assert r.status_code == 201
    assert r.json()["stepCount"] == 2

    snaps = wf_client.get("/workspaces/test/snapshots").json()
    assert any(s["key"] == "onboarding" and s["version"] == "1" for s in snaps)


def test_materialize_missing_def_is_404(wf_client):
    r = wf_client.post("/workflow-defs/ghost/versions/1/materialize")

    assert r.status_code == 404
    assert r.json()["error"] == "WorkflowDefNotFoundError"


# ── U12 run-inspection REST reads (AC-5 observability seam) ─────────────────────


def _seed_run(conn):
    """Seed a debug run with one StepRun + one TraceEvent directly in ws:test."""
    g = db.workspace_graph(conn, "test")
    g.query(
        "CREATE (r:WorkflowRun {runId:'r1', status:'done', stepCount:1, maxSteps:12, "
        "trace:true, ctx:'{}', startedAt:1, endedAt:9, waitingThreadId:''})"
    )
    g.query(
        "MATCH (r:WorkflowRun {runId:'r1'}) "
        "CREATE (r)-[:HAS_STEP_RUN]->(sr:StepRun {stepRunId:'sr1', stepKey:'intake', "
        "status:'done', startedAt:1, endedAt:2, input:'', output:'asked a question'})"
    )
    g.query(
        "MATCH (sr:StepRun {stepRunId:'sr1'}) "
        "CREATE (sr)-[:TRACED]->(te:TraceEvent {traceId:'te1', seq:0, "
        "kind:'node_rationale', at:1, payload:'asked a question'})"
    )


def test_get_workflow_run(client, conn):
    _seed_run(conn)
    r = client.get("/workflow-runs/r1")
    assert r.status_code == 200
    body = r.json()
    assert body["runId"] == "r1"
    assert body["status"] == "done"


def test_get_workflow_run_missing_is_404(client):
    r = client.get("/workflow-runs/ghost")
    assert r.status_code == 404


def test_get_workflow_step_runs(client, conn):
    _seed_run(conn)
    r = client.get("/workflow-runs/r1/step-runs")
    assert r.status_code == 200
    rows = r.json()
    assert [s["stepKey"] for s in rows] == ["intake"]


def test_get_workflow_trace(client, conn):
    _seed_run(conn)
    r = client.get("/workflow-runs/r1/trace")
    assert r.status_code == 200
    events = r.json()
    assert events and events[0]["kind"] == "node_rationale"
