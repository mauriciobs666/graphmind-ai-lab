"""REST contract tests via FastAPI TestClient against the live `ws:test` graph.

The app is built with `mount_mcp=False` (REST is what's under test; keeping the
FastMCP session manager out avoids its run-once-per-instance constraint) and a
context override pinning the tenant to `ws:test`.
"""

from __future__ import annotations

import itertools

import pytest
from conftest import TEST_EMBEDDING_DIM
from fastapi.testclient import TestClient

from falkorchat import config, db
from falkorchat.app import create_app
from falkorchat.config import CallContext
from falkorchat.repository import Repository
from falkorchat.services import DEMO_EXPECTED_DEFS, Services


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


# ── §14 Documents & Chunks (K-050 M5 Stage 1) ────────────────────────────────────


def test_ingest_and_get_document_round_trips_full_text(client):
    r = client.post("/documents", json={"text": "hello world", "title": "My Doc"})
    assert r.status_code == 201
    body = r.json()
    assert body["chunkCount"] == 1
    assert body["status"] == "processing"
    doc_id = body["documentId"]

    got = client.get(f"/documents/{doc_id}")
    assert got.status_code == 200
    doc = got.json()
    assert doc["text"] == "hello world"  # AC-9: byte-identical round trip
    assert doc["title"] == "My Doc"
    assert doc["sourceKind"] == "document"  # actor "u1" is a known User
    assert doc["ingestedById"] == "u1"


def test_ingest_document_defaults_source_format_to_text(client):
    r = client.post("/documents", json={"text": "hello"})
    assert r.status_code == 201
    doc = client.get(f"/documents/{r.json()['documentId']}").json()
    assert doc["sourceFormat"] == "text"


def test_get_missing_document_404(client):
    r = client.get("/documents/nope")
    assert r.status_code == 404


def test_ingest_document_oversized_text_is_422(client):
    r = client.post("/documents", json={"text": "x" * 500_001})
    assert r.status_code == 422


def test_ingest_document_empty_text_is_422(client):
    r = client.post("/documents", json={"text": ""})
    assert r.status_code == 422


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
    """Records embed_message/embed_chunk calls scheduled on BackgroundTasks."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.chunk_calls: list[tuple] = []

    def embed_message(self, ws, *, msg_id, text):
        self.calls.append((ws, msg_id, text))
        return [0.0]

    def embed_chunk(self, ws, *, chunk_id, text):
        self.chunk_calls.append((ws, chunk_id, text))
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


# ── K-050 M5 Stage 2: chunk embedding + search_documents ────────────────────────


def test_ingesting_a_document_schedules_every_chunk_for_embedding(wired):
    client, worker, _ = wired
    text = "First paragraph.\n\nSecond paragraph, a little longer than the first."
    r = client.post("/documents", json={"text": text, "title": "Doc"})
    assert r.status_code == 201
    body = r.json()

    assert len(worker.chunk_calls) == body["chunkCount"]
    assert {ws for ws, _cid, _text in worker.chunk_calls} == {"test"}
    # every scheduled chunk's text is a non-empty slice of the ingested document
    for _ws, _chunk_id, chunk_text in worker.chunk_calls:
        assert chunk_text
        assert chunk_text in text


def test_default_app_ingest_document_has_no_chunk_embed_wiring(client):
    # No embed_worker configured (the plain `client` fixture, mirrors
    # `test_default_app_has_no_wiring_and_posts_normally`) → ingest still
    # succeeds with nothing scheduled, no crash from a None embed_worker.
    r = client.post("/documents", json={"text": "hello"})
    assert r.status_code == 201


class _StubQueryEmbedder:
    """Returns a fixed TEST_EMBEDDING_DIM vector for any query text."""

    def embed(self, text):
        return [1.0] + [0.0] * (TEST_EMBEDDING_DIM - 1)


class _StubEmbeddingGateway:
    def embedder(self, kind, *, ws=None):
        return _StubQueryEmbedder()


@pytest.fixture()
def search_client(conn):
    """App wired with a stub `ModelGateway` so `search_documents` can embed a
    query — the other fixtures above never wire `models=` (K-050 M5 Stage 2 is
    the first REST surface that needs it). Returns `(client, repo)` — the repo
    is used to write a chunk embedding directly, since this fixture has no
    `embed_worker` (the search path itself is what's under test).
    """
    repo = Repository(conn)
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    services = Services(repo, models=_StubEmbeddingGateway())
    app = create_app(
        services,
        context_provider=lambda: CallContext(ws="test", actor="u1"),
        mount_mcp=False,
    )
    return TestClient(app), repo


def test_search_documents_returns_ranked_chunks(search_client):
    client, repo = search_client
    r = client.post("/documents", json={"text": "about cats", "title": "Doc"})
    assert r.status_code == 201
    doc_id = r.json()["documentId"]
    chunk_id = repo.list_document_chunks("test", document_id=doc_id)[0]["chunkId"]
    repo.set_chunk_embedding(
        "test", chunk_id=chunk_id,
        embedding=[1.0] + [0.0] * (TEST_EMBEDDING_DIM - 1),
        expected_dim=TEST_EMBEDDING_DIM,
    )

    hits = client.get("/documents/search", params={"q": "cats"})

    assert hits.status_code == 200
    body = hits.json()
    assert chunk_id in [h["chunkId"] for h in body]
    hit = next(h for h in body if h["chunkId"] == chunk_id)
    assert hit["documentId"] == doc_id
    assert hit["text"] == "about cats"


def test_search_documents_503_when_no_models_wired(client):
    # The plain `client` fixture builds `Services(repo)` with `models=None` —
    # `search_documents` refuses cleanly (SearchNotAvailableError -> 503)
    # rather than an AttributeError on a None gateway.
    r = client.get("/documents/search", params={"q": "cats"})
    assert r.status_code == 503
    assert r.json()["error"] == "SearchNotAvailableError"


def test_search_documents_route_not_shadowed_by_document_id_route(search_client):
    # Registration-order regression guard: `/documents/search` must resolve to
    # the search route, not `/documents/{document_id}` treating "search" as an id.
    client, _repo = search_client
    r = client.get("/documents/search", params={"q": "anything"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


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


# ── §11 def/snapshot STRUCTURE reads + diff (K-031 observability) ───────────────
#
# Black-box answers to "is what I think is published actually published", "is the
# workspace running the same thing", and "have `reference` and `ws:{id}` gone
# stale independently". Read-only: this surface makes the current publish
# semantics observable, it never changes them.

_STRUCTURE_KEYS = {
    "source", "key", "version", "name", "kind", "startKey",
    "stepCount", "transitionCount", "steps", "transitions",
}
# The same anti-drift guard for the third route: `response_model=` FILTERS
# undeclared fields, so a field added to the service dict but not to
# `WorkflowDiffOut` — or dropped from it — is invisible to field-by-field
# assertions.
_DIFF_KEYS = {
    "key", "version", "defPresent", "snapshotPresent", "inSync",
    "differences", "differenceCount",
}
_DIFF_ENTRY_KEYS = {"path", "def", "snapshot"}


def _wipe_reference(conn):
    """Wipe the global `reference` graph mid-test — the documented live trap.

    A plain helper, deliberately **not** a fixture: it must only ever run inside
    a test that already owns `reference` (i.e. under `wf_repo`/`wf_client`, whose
    fixture wipes it at setup anyway), never autouse or session-scoped.
    """
    db.reference_graph(conn).query("MATCH (n) DETACH DELETE n")


def test_def_structure_read_returns_the_published_spec_exactly(wf_client):
    wf_client.post("/workflow-defs", json=DEF_BODY)

    r = wf_client.get("/workflow-defs/onboarding/versions/1")

    assert r.status_code == 200
    body = r.json()
    # The anti-drift assertion: exact key set, no `startKeys` for one START edge.
    assert set(body) == _STRUCTURE_KEYS
    assert body["source"] == "reference"
    assert (body["key"], body["version"]) == ("onboarding", "1")
    assert (body["name"], body["kind"]) == ("Onboarding", "process")
    assert body["startKey"] == "start"
    assert body["stepCount"] == 2
    assert body["transitionCount"] == 1
    # `config` comes back byte-identical to what was published (rule 8).
    assert body["steps"] == [
        {"key": "done", "type": "message", "config": ""},
        {"key": "start", "type": "human", "config": '{"waitsForHuman": true}'},
    ]
    assert body["transitions"] == [
        {"from": "start", "to": "done", "on": "submitted", "order": 0, "guard": ""},
    ]


def test_republish_is_create_only_on_properties_structure_read_unchanged(wf_client):
    # Pins a **decision**, not a bug: publish is `MERGE … ON CREATE SET`, so an
    # edited re-publish of the same (key, version) leaves stored properties
    # alone. K-031 deliberately does not change that — it makes it observable.
    # The *structural* (additive) half of re-publish is **K-034's**, not tested
    # here.
    wf_client.post("/workflow-defs", json=DEF_BODY)
    before = wf_client.get("/workflow-defs/onboarding/versions/1").json()

    edited = {
        **DEF_BODY,
        "name": "Onboarding EDITED",
        # must stay inside WORKFLOW_KINDS or the re-publish 400s and the test
        # would pin the wrong thing
        "kind": "conversation",
        "steps": [
            # the edited config must keep `waitsForHuman` on the `human` step —
            # dropping it makes the re-publish a 400 (`_validate_def_spec`)
            {"key": "start", "type": "human",
             "config": '{"waitsForHuman": true, "note": "edited"}', "start": True},
            {"key": "done", "type": "message"},
        ],
        "transitions": DEF_BODY["transitions"],
    }
    r = wf_client.post("/workflow-defs", json=edited)
    assert r.status_code == 201

    after = wf_client.get("/workflow-defs/onboarding/versions/1").json()
    assert after == before


# ── K-034 — topology-conflict gate (409) ─────────────────────────────────────
#
# `WorkflowDefConflictError` → 409, same envelope shape as the other workflow
# error handlers. This first test isolates the app-level wiring (handler
# registration) from the gate logic itself — it doesn't matter yet *how* the
# service raises the error, only that the app maps it correctly when it does.


def test_workflow_def_conflict_error_maps_to_409(wf_client, monkeypatch):
    from falkorchat.services import Services, WorkflowDefConflictError

    def _boom(self, ctx, **kwargs):
        raise WorkflowDefConflictError("onboarding v1 topology differs")

    monkeypatch.setattr(Services, "publish_workflow_def", _boom)

    r = wf_client.post("/workflow-defs", json=DEF_BODY)

    assert r.status_code == 409
    assert r.json()["error"] == "WorkflowDefConflictError"
    assert "topology differs" in r.json()["detail"]


def test_republish_with_changed_transition_to_is_409_and_leaves_structure_unchanged(
    wf_client,
):
    # end-to-end proof against the real `_PUBLISH_CYPHER`: a retargeted
    # transition would otherwise MERGE a *parallel* edge beside the old one
    # (§2.1) — the gate must reject before that write happens.
    wf_client.post("/workflow-defs", json=DEF_BODY)
    before = wf_client.get("/workflow-defs/onboarding/versions/1").json()

    conflicting = {
        **DEF_BODY,
        "steps": [*DEF_BODY["steps"], {"key": "escalate", "type": "message"}],
        "transitions": [
            {"from": "start", "to": "escalate", "on": "submitted", "order": 0},
        ],
    }
    r = wf_client.post("/workflow-defs", json=conflicting)

    assert r.status_code == 409
    assert r.json()["error"] == "WorkflowDefConflictError"

    after = wf_client.get("/workflow-defs/onboarding/versions/1").json()
    assert after == before  # exactly the original — no parallel structure minted


def test_republish_with_changed_start_key_is_409_and_leaves_one_start(wf_client):
    # end-to-end proof: a moved start step would otherwise MERGE a *second*
    # `START` edge beside the old one (§2.1) — the gate must reject first.
    wf_client.post("/workflow-defs", json=DEF_BODY)
    before = wf_client.get("/workflow-defs/onboarding/versions/1").json()

    conflicting = {
        **DEF_BODY,
        "steps": [
            {"key": "start", "type": "human", "config": '{"waitsForHuman": true}'},
            {"key": "done", "type": "message", "start": True},  # start moved here
        ],
    }
    r = wf_client.post("/workflow-defs", json=conflicting)

    assert r.status_code == 409
    assert r.json()["error"] == "WorkflowDefConflictError"

    after = wf_client.get("/workflow-defs/onboarding/versions/1").json()
    assert after == before
    assert after["startKey"] == "start"
    assert "startKeys" not in after  # exactly one START edge, not two


def test_materialize_conflict_after_seeded_drift_is_409_and_leaves_snapshot_unchanged(
    wf_client, wf_repo,
):
    # Reproduce the live trap the way it actually arises (§2.2): the reference
    # def stays clean, but the workspace snapshot independently drifts (e.g. via
    # a caller that bypasses `services.py` — `Repository` is a thin, non-
    # validating primitive by design, §3.3). The subsequent service-layer
    # materialize call must reject, not silently mint parallel structure.
    wf_client.post("/workflow-defs", json=DEF_BODY)
    wf_client.post("/workflow-defs/onboarding/versions/1/materialize")

    wf_repo.materialize_snapshot(
        "test", key="onboarding", version="1", name="Onboarding", kind="process",
        start_key="start",
        steps=[
            {"key": "start", "type": "human", "config": '{"waitsForHuman": true}'},
            {"key": "done", "type": "message", "config": ""},
            {"key": "escalate", "type": "message", "config": ""},
        ],
        transitions=[
            {"from": "start", "to": "escalate", "on": "submitted", "order": 0,
             "guard": ""},
        ],
    )
    before = wf_client.get("/workspaces/test/snapshots/onboarding/versions/1").json()

    r = wf_client.post("/workflow-defs/onboarding/versions/1/materialize")

    assert r.status_code == 409
    assert r.json()["error"] == "WorkflowDefConflictError"

    after = wf_client.get("/workspaces/test/snapshots/onboarding/versions/1").json()
    assert after == before  # nothing written — still the seeded, differing content


def test_snapshot_structure_route_404_before_and_200_after_materialize(wf_client):
    wf_client.post("/workflow-defs", json=DEF_BODY)

    missing = wf_client.get("/workspaces/test/snapshots/onboarding/versions/1")
    assert missing.status_code == 404

    wf_client.post("/workflow-defs/onboarding/versions/1/materialize")
    r = wf_client.get("/workspaces/test/snapshots/onboarding/versions/1")

    assert r.status_code == 200
    body = r.json()
    assert set(body) == _STRUCTURE_KEYS
    assert body["source"] == "workspace"
    # identical to the def body apart from `source` — hand-diffable with `jq`
    def_body = wf_client.get("/workflow-defs/onboarding/versions/1").json()
    assert {**body, "source": "reference"} == def_body


def test_def_structure_404_for_unknown_key_and_unknown_version(wf_client):
    wf_client.post("/workflow-defs", json=DEF_BODY)

    assert wf_client.get("/workflow-defs/ghost/versions/1").status_code == 404
    r = wf_client.get("/workflow-defs/onboarding/versions/99")
    assert r.status_code == 404
    assert r.json()["error"] == "WorkflowDefNotFoundError"


def test_structure_response_model_serializes_from_not_from_underscore(wf_client):
    wf_client.post("/workflow-defs", json=DEF_BODY)

    tr = wf_client.get("/workflow-defs/onboarding/versions/1").json()["transitions"][0]

    assert set(tr) == {"from", "to", "on", "order", "guard"}
    assert tr["from"] == "start"


def test_structure_response_model_start_keys_omission_all_three_directions():
    # Pins the `response_model_exclude_unset=True` decision (K-031). `exclude_none`
    # would be the obvious mechanism and is WRONG here: `startKey` is itself
    # nullable, so `exclude_none` would silently drop it for a root with no `START`
    # edge — an observability endpoint hiding exactly the anomaly it exists to show.
    # `exclude_unset` keys off which fields the service actually put in the dict,
    # so "absent" and "explicitly null" stay distinguishable. No graph needed, and
    # deliberately no dependency on how a two-`START` root arises (that is K-034's).
    from fastapi import FastAPI

    from falkorchat.schemas import WorkflowDefStructureOut

    base = {
        "source": "reference", "key": "a", "version": "1", "name": "A",
        "kind": "process", "stepCount": 1, "transitionCount": 1,
        "steps": [{"key": "s", "type": "human", "config": "{}"}],
        "transitions": [
            {"from": "s", "to": "s", "on": "go", "order": 0, "guard": ""}
        ],
    }
    app = FastAPI()

    def route(payload):
        return lambda: payload

    for path, payload in (
        ("/one", {**base, "startKey": "s"}),
        ("/two", {**base, "startKey": "a", "startKeys": ["a", "b"]}),
        ("/none", {**base, "startKey": None}),
    ):
        app.get(
            path,
            response_model=WorkflowDefStructureOut,
            response_model_exclude_unset=True,
        )(route(payload))
    c = TestClient(app)

    one = c.get("/one").json()
    assert "startKeys" not in one and one["startKey"] == "s"

    two = c.get("/two").json()
    assert two["startKeys"] == ["a", "b"] and two["startKey"] == "a"

    # the anomaly case: a root with no START edge keeps an explicit null `startKey`
    no_start = c.get("/none").json()
    assert "startKey" in no_start and no_start["startKey"] is None
    assert "startKeys" not in no_start


def test_diff_identical_def_and_snapshot_is_in_sync(wf_client):
    wf_client.post("/workflow-defs", json=DEF_BODY)
    wf_client.post("/workflow-defs/onboarding/versions/1/materialize")

    body = wf_client.get(
        "/workspaces/test/snapshots/onboarding/versions/1/diff"
    ).json()

    assert set(body) == _DIFF_KEYS
    assert body["inSync"] is True
    assert body["differenceCount"] == 0
    assert body["differences"] == []
    assert (body["defPresent"], body["snapshotPresent"]) == (True, True)


def test_diff_reports_divergence_after_the_documented_reseed_trap(wf_client, conn):
    # Reproduce the live trap exactly: publish + materialize, then a `pytest` /
    # `test_queries.sh` run wipes `reference` while `ws:{id}` survives, and a
    # naive re-seed republishes an *edited* def into the now-empty `reference`.
    # Every edit lands because that re-publish is a fresh CREATE — this fixture
    # depends on create semantics only, never on K-034's additive semantics.
    wf_client.post("/workflow-defs", json=DEF_BODY)
    wf_client.post("/workflow-defs/onboarding/versions/1/materialize")
    _wipe_reference(conn)

    edited = {
        **DEF_BODY,
        "name": "Onboarding v2",
        "steps": [
            {"key": "start", "type": "human",
             "config": '{"waitsForHuman": true, "note": "edited"}', "start": True},
            {"key": "done", "type": "message"},
            {"key": "escalate", "type": "message"},
        ],
        "transitions": [
            {"from": "start", "to": "done", "on": "submitted", "order": 0,
             "guard": '{"kind":"cmp","op":"eq","path":"ctx.status","value":"ok"}'},
        ],
    }
    assert wf_client.post("/workflow-defs", json=edited).status_code == 201

    body = wf_client.get(
        "/workspaces/test/snapshots/onboarding/versions/1/diff"
    ).json()

    assert body["inSync"] is False
    # entry-level anti-drift: `def_` must serialize under its `def` alias, and no
    # entry field may appear or vanish unnoticed.
    assert set(body["differences"][0]) == _DIFF_ENTRY_KEYS
    seen = {d["path"]: (d["def"], d["snapshot"]) for d in body["differences"]}
    assert seen["meta.name"] == ("Onboarding v2", "Onboarding")
    assert seen["steps[escalate]"] == ("present", "absent")
    assert seen["steps[start].config"] == (
        '{"waitsForHuman": true, "note": "edited"}', '{"waitsForHuman": true}'
    )
    assert seen["transitions[start->done@submitted#0].guard"] == (
        '{"kind":"cmp","op":"eq","path":"ctx.status","value":"ok"}', ""
    )
    assert body["differenceCount"] == len(body["differences"]) == 4


def test_diff_def_missing_snapshot_present_is_200_with_presence_flags(
    wf_client, conn
):
    wf_client.post("/workflow-defs", json=DEF_BODY)
    wf_client.post("/workflow-defs/onboarding/versions/1/materialize")
    _wipe_reference(conn)

    r = wf_client.get("/workspaces/test/snapshots/onboarding/versions/1/diff")

    assert r.status_code == 200
    body = r.json()
    assert (body["defPresent"], body["snapshotPresent"]) == (False, True)
    assert body["inSync"] is False
    assert body["differences"] == []


def test_diff_both_absent_is_404(wf_client):
    r = wf_client.get("/workspaces/test/snapshots/ghost/versions/1/diff")

    assert r.status_code == 404
    assert r.json()["error"] == "WorkflowDefNotFoundError"


# ── FR-10 workspace readiness (web-api-coverage plan §3.1c / U2) ────────────────
#
# The HTTP form of `scripts/verify_workflows.sh`. Always 200 — readiness is a
# report, never a 404/error condition. The service layer (`test_services.py`)
# exhaustively covers the presence/sync/tripwire logic against a fake repo;
# this is the contract test that the route is wired and shaped correctly.

_READINESS_KEYS = {"ready", "defs", "postSuccess"}
_READINESS_ENTRY_KEYS = {
    "key", "version", "defPresent", "snapshotPresent", "inSync", "problems",
}
_POST_SUCCESS_KEYS = {
    "defKey", "defVersion", "sampleSize", "postedCount", "rate", "status",
}


def test_readiness_route_not_ready_when_nothing_seeded(wf_client):
    r = wf_client.get("/workspaces/test/readiness")

    assert r.status_code == 200
    body = r.json()
    assert set(body) == _READINESS_KEYS
    assert body["ready"] is False
    assert [(d["key"], d["version"]) for d in body["defs"]] == list(DEMO_EXPECTED_DEFS)
    for entry in body["defs"]:
        assert set(entry) == _READINESS_ENTRY_KEYS
        assert (entry["defPresent"], entry["snapshotPresent"]) == (False, False)
        label = f"{entry['key']}@{entry['version']}"
        assert entry["problems"] == [
            f"{label}: not published in `reference` at this version",
            f"{label}: not materialized into ws:test at this version",
        ]
    # no WorkflowRuns exist in this fresh ws:test graph — "no data", not 0%
    post_success = body["postSuccess"]
    assert set(post_success) == _POST_SUCCESS_KEYS
    assert post_success["defKey"] == config.TRIGGER_DEF_KEY
    assert post_success["defVersion"] == config.TRIGGER_DEF_VERSION
    assert post_success["sampleSize"] == 0
    assert post_success["postedCount"] == 0
    assert post_success["rate"] is None
    assert post_success["status"] == "no-data"


def test_readiness_route_ready_when_both_demo_defs_published_and_synced(wf_client):
    for key, version in DEMO_EXPECTED_DEFS:
        body = {**DEF_BODY, "key": key, "version": version}
        assert wf_client.post("/workflow-defs", json=body).status_code == 201
        r = wf_client.post(f"/workflow-defs/{key}/versions/{version}/materialize")
        assert r.status_code == 201

    r = wf_client.get("/workspaces/test/readiness")

    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    for entry in body["defs"]:
        assert (entry["defPresent"], entry["snapshotPresent"], entry["inSync"]) == (
            True, True, True,
        )
        assert entry["problems"] == []
    # publishing/materializing the demo defs starts no WorkflowRuns — still "no data"
    assert body["postSuccess"]["status"] == "no-data"
    assert body["postSuccess"]["sampleSize"] == 0


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


# ── K-036 web-api-coverage: thread-scoped reads (FR-2/FR-8, Wave 2) ──────────


def _seed_run_for_thread(
    conn, *, thread_id, run_id, started_at, status="running",
    ended_at=None, def_key="triage", def_version="1",
):
    """Seed a minimal WorkflowRun -[:TRIGGERED_BY]-> Message directly in ws:test.

    The trigger message need not sit on the thread's actual HEAD/TAIL chain —
    `find_runs_for_thread` only needs `Message.threadId`, not thread structure.
    """
    g = db.workspace_graph(conn, "test")
    g.query(
        "CREATE (m:Message {msgId: $msgId, text: 'trigger', role: 'user', "
        "                    createdAt: $startedAt, threadId: $threadId}) "
        "CREATE (r:WorkflowRun {runId: $runId, status: $status, defKey: $defKey, "
        "                       defVersion: $defVersion, startedAt: $startedAt, "
        "                       endedAt: $endedAt, stepCount: 0, maxSteps: 12, "
        "                       trace: false, ctx: '{}', waitingThreadId: ''}) "
        "CREATE (r)-[:TRIGGERED_BY]->(m)",
        {
            "msgId": f"trig-{run_id}", "runId": run_id, "status": status,
            "defKey": def_key, "defVersion": def_version, "startedAt": started_at,
            "endedAt": ended_at, "threadId": thread_id,
        },
    )


def _add_to_channel(conn, *, member_id, channel_id):
    """Raw MEMBER_OF write — no repository method exists yet (test_repository.py's
    `_add_to_channel` docstring explains the gap); anchors label-agnostically on
    `userId OR agentId`, same pattern `advance_cursor` uses in production."""
    db.workspace_graph(conn, "test").query(
        "MATCH (mem) WHERE mem.userId = $memberId OR mem.agentId = $memberId "
        "MATCH (c:Channel {channelId: $channelId}) "
        "MERGE (mem)-[:MEMBER_OF]->(c)",
        {"memberId": member_id, "channelId": channel_id},
    )


def test_thread_workflow_runs_empty_when_none(client):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    r = client.get(f"/threads/{tid}/workflow-runs")

    assert r.status_code == 200
    assert r.json() == []


def test_thread_workflow_runs_populated_newest_first(client, conn):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)
    _seed_run_for_thread(conn, thread_id=tid, run_id="r1", started_at=1000)
    _seed_run_for_thread(conn, thread_id=tid, run_id="r2", started_at=2000)

    r = client.get(f"/threads/{tid}/workflow-runs")

    assert r.status_code == 200
    assert [x["runId"] for x in r.json()] == ["r2", "r1"]


def test_thread_workflow_runs_respects_limit_query_param(client, conn):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)
    _seed_run_for_thread(conn, thread_id=tid, run_id="r1", started_at=1000)
    _seed_run_for_thread(conn, thread_id=tid, run_id="r2", started_at=2000)

    r = client.get(f"/threads/{tid}/workflow-runs", params={"limit": 1})

    assert r.status_code == 200
    assert [x["runId"] for x in r.json()] == ["r2"]


def test_thread_workflow_runs_limit_bounds_are_422(client):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    assert (
        client.get(f"/threads/{tid}/workflow-runs", params={"limit": 0}).status_code
        == 422
    )
    assert (
        client.get(f"/threads/{tid}/workflow-runs", params={"limit": 51}).status_code
        == 422
    )


def test_thread_workflow_runs_unknown_thread_404(client):
    r = client.get("/threads/ghost/workflow-runs")
    assert r.status_code == 404


def test_thread_participants_both_kinds(client, conn):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)
    Repository(conn).ensure_agent("test", agent_id="a1", name="Bot")
    _add_to_channel(conn, member_id="u1", channel_id=cid)  # u1: seeded by `client`
    _add_to_channel(conn, member_id="a1", channel_id=cid)

    r = client.get(f"/threads/{tid}/participants")

    assert r.status_code == 200
    rows = r.json()
    assert {(row["memberId"], row["kind"]) for row in rows} == {
        ("u1", "User"), ("a1", "Agent"),
    }


def test_thread_participants_only_human(client, conn):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)
    _add_to_channel(conn, member_id="u1", channel_id=cid)

    r = client.get(f"/threads/{tid}/participants")

    assert r.status_code == 200
    assert [row["kind"] for row in r.json()] == ["User"]


def test_thread_participants_only_agent(client, conn):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)
    Repository(conn).ensure_agent("test", agent_id="a1", name="Bot")
    _add_to_channel(conn, member_id="a1", channel_id=cid)

    r = client.get(f"/threads/{tid}/participants")

    assert r.status_code == 200
    assert [row["kind"] for row in r.json()] == ["Agent"]


def test_thread_participants_empty_when_channel_has_no_members(client):
    cid = _new_channel(client)
    tid = _new_thread(client, cid)

    r = client.get(f"/threads/{tid}/participants")

    assert r.status_code == 200
    assert r.json() == []


def test_thread_participants_unknown_thread_404(client):
    r = client.get("/threads/ghost/participants")
    assert r.status_code == 404
