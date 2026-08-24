"""MCP transport tests — in-memory via FastMCP `list_tools`/`call_tool`.

The MCP tools are thin adapters; these confirm discovery and that a tool call
round-trips through the real service layer against the live `ws:test` graph.
No HTTP server is started — FastMCP is exercised directly.
"""

from __future__ import annotations

import asyncio
import itertools
import json

import pytest

from falkorchat import mcp as mcp_mod
from falkorchat.config import CallContext
from falkorchat.services import Services

TEST_CTX = CallContext(ws="test", actor="u1")


def _configure(
    repo, *, actor="u1", responder=None, embed_worker=None, trigger=None, models=None
):
    clock = itertools.count(1000)
    ids = (f"id{n}" for n in itertools.count(1))
    svc = Services(
        repo, clock=lambda: next(clock), id_gen=lambda: next(ids), models=models
    )
    mcp_mod.configure(
        svc,
        context_provider=lambda: CallContext(ws="test", actor=actor),
        responder=responder,
        embed_worker=embed_worker,
        trigger=trigger,
    )
    return svc


def _unwrap(result):
    """call_tool returns ``(content_blocks, structured)``.

    Prefer the structured payload: dict tools return the dict directly; list
    tools are wrapped as ``{"result": [...]}``. Fall back to parsing the first
    text block when no structured content is present.
    """
    if isinstance(result, tuple):
        structured = result[1]
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        if structured is not None:
            return structured
        result = result[0]
    return json.loads(result[0].text)


def test_tool_discovery_lists_all_tools(repo):
    _configure(repo)
    tools = asyncio.run(mcp_mod.mcp.list_tools())
    assert {t.name for t in tools} == {
        "send_message", "read_messages", "create_thread",
        "search_messages", "create_channel", "list_channels", "list_threads",
        "ingest_document", "get_document", "search_documents",
    }


def test_list_tools_let_agent_navigate_to_existing_thread(repo):
    """An agent must be able to discover an existing conversation, not just
    create its own: list_channels → list_threads → send_message."""
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _configure(repo)
    ch = svc.create_channel(TEST_CTX, name="general")
    th = svc.create_thread(TEST_CTX, channel_id=ch["channelId"], title="standup")

    async def scenario():
        channels = _unwrap(await mcp_mod.mcp.call_tool("list_channels", {}))
        cid = channels[0]["channelId"]
        threads = _unwrap(await mcp_mod.mcp.call_tool(
            "list_threads", {"channel_id": cid}
        ))
        tid = threads[0]["threadId"]
        await mcp_mod.mcp.call_tool("send_message", {"body": "found you", "re": tid})
        return channels, threads, _unwrap(await mcp_mod.mcp.call_tool(
            "read_messages", {"re": tid, "since": 0, "advance": False}
        ))

    channels, threads, rows = asyncio.run(scenario())
    assert [c["channelId"] for c in channels] == [ch["channelId"]]
    assert [t["threadId"] for t in threads] == [th["threadId"]]
    assert [r["text"] for r in rows] == ["found you"]


def test_search_messages_tool_finds_posted_text(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _configure(repo)
    ch = svc.create_channel(TEST_CTX, name="general")

    async def scenario():
        th = _unwrap(await mcp_mod.mcp.call_tool(
            "create_thread", {"channel_id": ch["channelId"], "title": "hi"}
        ))
        tid = th["threadId"]
        await mcp_mod.mcp.call_tool("send_message", {"body": "hello world", "re": tid})
        await mcp_mod.mcp.call_tool("send_message", {"body": "goodbye moon", "re": tid})
        return _unwrap(await mcp_mod.mcp.call_tool(
            "search_messages", {"query": "hello"}
        ))

    hits = asyncio.run(scenario())
    assert [h["text"] for h in hits] == ["hello world"]


def test_create_channel_tool_enables_full_agent_flow(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    async def scenario():
        ch = _unwrap(await mcp_mod.mcp.call_tool("create_channel", {"name": "general"}))
        th = _unwrap(await mcp_mod.mcp.call_tool(
            "create_thread", {"channel_id": ch["channelId"], "title": "hi"}
        ))
        await mcp_mod.mcp.call_tool("send_message", {"body": "hi", "re": th["threadId"]})
        return _unwrap(await mcp_mod.mcp.call_tool(
            "read_messages", {"re": th["threadId"], "since": 0, "advance": False}
        ))

    rows = asyncio.run(scenario())
    assert [r["text"] for r in rows] == ["hi"]


def test_create_thread_send_and_read_roundtrip(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _configure(repo)
    # channel is REST-only; seed it directly through the service
    ch = svc.create_channel(TEST_CTX, name="general")

    async def scenario():
        th = _unwrap(await mcp_mod.mcp.call_tool(
            "create_thread", {"channel_id": ch["channelId"], "title": "hi"}
        ))
        await mcp_mod.mcp.call_tool(
            "send_message", {"body": "hello world", "re": th["threadId"]}
        )
        return _unwrap(await mcp_mod.mcp.call_tool(
            "read_messages", {"re": th["threadId"], "since": 0, "advance": False}
        ))

    rows = asyncio.run(scenario())
    assert [r["text"] for r in rows] == ["hello world"]


def test_send_message_mention_flagged_in_chronological_read(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.ensure_user("test", user_id="u2", display_name="Bob")
    svc = _configure(repo)
    ch = svc.create_channel(TEST_CTX, name="general")

    async def scenario():
        th = _unwrap(await mcp_mod.mcp.call_tool(
            "create_thread", {"channel_id": ch["channelId"], "title": "hi"}
        ))
        tid = th["threadId"]
        await mcp_mod.mcp.call_tool("send_message", {"body": "plain", "re": tid})
        await mcp_mod.mcp.call_tool(
            "send_message", {"body": "hey bob", "re": tid, "mentions": ["u2"]}
        )
        # read as Bob
        mcp_mod.configure(svc, context_provider=lambda: CallContext(ws="test", actor="u2"))
        return _unwrap(await mcp_mod.mcp.call_tool(
            "read_messages", {"re": tid, "since": 0, "advance": False}
        ))

    rows = asyncio.run(scenario())
    assert [r["text"] for r in rows] == ["plain", "hey bob"]
    assert [r["isMention"] for r in rows] == [False, True]


def test_read_messages_rows_carry_thread_id(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _configure(repo)
    ch = svc.create_channel(TEST_CTX, name="general")

    async def scenario():
        th = _unwrap(await mcp_mod.mcp.call_tool(
            "create_thread", {"channel_id": ch["channelId"], "title": "hi"}
        ))
        await mcp_mod.mcp.call_tool("send_message", {"body": "hi", "re": th["threadId"]})
        rows = _unwrap(await mcp_mod.mcp.call_tool(
            "read_messages", {"re": th["threadId"], "since": 0, "advance": False}
        ))
        return th["threadId"], rows

    tid, rows = asyncio.run(scenario())
    assert [r["threadId"] for r in rows] == [tid]


def test_send_message_unknown_mention_errors(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _configure(repo)
    ch = svc.create_channel(TEST_CTX, name="general")

    async def scenario():
        th = _unwrap(await mcp_mod.mcp.call_tool(
            "create_thread", {"channel_id": ch["channelId"], "title": "hi"}
        ))
        await mcp_mod.mcp.call_tool(
            "send_message",
            {"body": "x", "re": th["threadId"], "mentions": ["ghost"]},
        )

    with pytest.raises(Exception):
        asyncio.run(scenario())


# ── §14 Documents & Chunks (K-050 M5 Stage 1) ────────────────────────────────────


def test_ingest_document_then_get_document_round_trips(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    async def scenario():
        posted = _unwrap(await mcp_mod.mcp.call_tool(
            "ingest_document", {"text": "hello world", "title": "My Doc"}
        ))
        got = _unwrap(await mcp_mod.mcp.call_tool(
            "get_document", {"document_id": posted["documentId"]}
        ))
        return posted, got

    posted, got = asyncio.run(scenario())
    assert posted["chunkCount"] == 1
    assert posted["status"] == "processing"
    assert got["text"] == "hello world"  # AC-9 round trip
    assert got["sourceKind"] == "document"


def test_ingest_document_unknown_actor_errors(repo):
    _configure(repo, actor="ghost")  # not a known User or Agent

    with pytest.raises(Exception):
        asyncio.run(mcp_mod.mcp.call_tool("ingest_document", {"text": "hello"}))


def test_get_document_missing_returns_none(repo):
    _configure(repo)

    got = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("get_document", {"document_id": "nope"})
    ))
    assert got is None


# ── K-050 M5 Stage 2: chunk embedding + search_documents ────────────────────────


class RecordingChunkWorker:
    """Records embed_chunk calls scheduled off-band (mirrors RecordingWorker
    below, but for `ingest_document`'s chunk-embed scheduling rather than
    `send_message`'s message embed)."""

    def __init__(self):
        self.calls: list[tuple] = []

    def embed_chunk(self, ws, *, chunk_id, text):
        self.calls.append((ws, chunk_id, text))
        return [0.0]


def test_ingest_document_schedules_every_chunk_for_embedding(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    worker = RecordingChunkWorker()
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *args: fn(*args)  # synchronous, like sync_schedule
    try:
        _configure(repo, embed_worker=worker)
        posted = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
            "ingest_document", {"text": "First para.\n\nSecond, longer paragraph."}
        )))
    finally:
        mcp_mod._schedule = original

    assert len(worker.calls) == posted["chunkCount"]
    assert {ws for ws, _cid, _text in worker.calls} == {"test"}


def test_ingest_document_with_no_embed_worker_schedules_nothing(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)  # no embed_worker

    posted = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
        "ingest_document", {"text": "hello"}
    )))
    assert posted["status"] == "processing"  # succeeds; nothing to assert-not-crash on


class _StubQueryEmbedder:
    def embed(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class _StubEmbeddingGateway:
    def embedder(self, kind, *, ws=None):
        return _StubQueryEmbedder()


def test_search_documents_tool_returns_ranked_chunks(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _configure(repo, models=_StubEmbeddingGateway())

    posted = svc.ingest_document(TEST_CTX, text="about cats")
    chunk_id = svc.list_document_chunks(
        TEST_CTX, document_id=posted["documentId"]
    )[0]["chunkId"]
    repo.set_chunk_embedding(
        "test", chunk_id=chunk_id, embedding=[1.0, 0.0, 0.0, 0.0], expected_dim=4
    )

    hits = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("search_documents", {"query": "cats"})
    ))
    assert chunk_id in [h["chunkId"] for h in hits]


def test_search_documents_tool_errors_when_no_models_wired(repo):
    _configure(repo)  # no models gateway

    with pytest.raises(Exception):
        asyncio.run(mcp_mod.mcp.call_tool("search_documents", {"query": "cats"}))


# ── K-041: MCP send_message must schedule the same background work the REST ────
# route does (out-of-band embed + trigger XOR responder, the M3 one-handler
# guarantee) — see `falkorchat/background.py` for the shared policy. Before this
# fix, `mcp.py`'s `send_message` posted via `Services.post_message` directly and
# returned, so no reply was ever scheduled for an MCP-posted message (D-1).


class RecordingWorker:
    """Records embed_message calls scheduled off-band."""

    def __init__(self):
        self.calls: list[tuple] = []

    def embed_message(self, ws, *, msg_id, text):
        self.calls.append((ws, msg_id, text))
        return [0.0]


class RecordingResponder:
    """Records maybe_respond calls scheduled off-band."""

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


class RecordingTrigger:
    """Records maybe_trigger calls scheduled off-band."""

    def __init__(self):
        self.calls: list[dict] = []

    def maybe_trigger(self, ctx, *, thread_id, msg_id, text, role, mentions):
        self.calls.append(
            {"thread_id": thread_id, "msg_id": msg_id, "text": text,
             "role": role, "mentions": mentions}
        )
        return None


@pytest.fixture()
def sync_schedule():
    """Make MCP background scheduling synchronous so tests can assert deterministically.

    Production fires a daemon thread (a plain MCP tool function has no per-call
    `BackgroundTasks` object the way a FastAPI route does); tests swap the
    `mcp_mod._schedule` seam for an inline call so the scheduled work has
    already happened by the time `call_tool` returns.
    """
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *args: fn(*args)
    yield
    mcp_mod._schedule = original


def _thread(svc, name="general"):
    ch = svc.create_channel(TEST_CTX, name=name)
    return svc.create_thread(TEST_CTX, channel_id=ch["channelId"], title="hi")


def test_send_message_schedules_responder_with_posted_message(repo, sync_schedule):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.ensure_agent("test", agent_id="bot1", name="Bot")
    responder = RecordingResponder()
    svc = _configure(repo, responder=responder)
    th = _thread(svc)

    asyncio.run(mcp_mod.mcp.call_tool(
        "send_message",
        {"body": "hey @bot", "re": th["threadId"], "mentions": ["bot1"]},
    ))

    assert len(responder.calls) == 1
    call = responder.calls[0]
    assert call["thread_id"] == th["threadId"]
    assert call["text"] == "hey @bot"
    assert call["role"] == "user"
    assert call["mentions"] == ["bot1"]


def test_send_message_trigger_wired_schedules_trigger_not_responder(repo, sync_schedule):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    trigger = RecordingTrigger()
    responder = RecordingResponder()
    svc = _configure(repo, trigger=trigger, responder=responder)
    th = _thread(svc)

    asyncio.run(mcp_mod.mcp.call_tool(
        "send_message", {"body": "@bot help", "re": th["threadId"]}
    ))

    # Exactly one handler fires — the trigger, never the responder (M3 one-handler
    # guarantee: an @mention can never fire both a workflow and a direct reply).
    assert len(trigger.calls) == 1
    assert trigger.calls[0]["text"] == "@bot help"
    assert responder.calls == []


def test_send_message_embeds_independently_of_trigger_or_responder(repo, sync_schedule):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    worker = RecordingWorker()
    trigger = RecordingTrigger()
    svc = _configure(repo, embed_worker=worker, trigger=trigger)
    th = _thread(svc)

    asyncio.run(mcp_mod.mcp.call_tool(
        "send_message", {"body": "hello", "re": th["threadId"]}
    ))

    assert len(worker.calls) == 1
    assert worker.calls[0][2] == "hello"
    assert len(trigger.calls) == 1


def test_send_message_with_no_wiring_posts_normally(repo, sync_schedule):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _configure(repo)
    th = _thread(svc)

    result = asyncio.run(mcp_mod.mcp.call_tool(
        "send_message", {"body": "hi", "re": th["threadId"]}
    ))

    assert _unwrap(result)["text"] == "hi"


def test_send_message_default_scheduling_runs_off_a_background_thread(repo):
    """Without the `sync_schedule` override, scheduling must not block the tool call
    (mirrors the REST route's off-band intent) — verified by observing that the
    responder call lands on a different thread than the one that ran `send_message`."""
    import threading

    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.ensure_agent("test", agent_id="bot1", name="Bot")
    seen_threads: list[int] = []
    done = threading.Event()

    class ThreadRecordingResponder(RecordingResponder):
        def maybe_respond(self, *args, **kwargs):
            seen_threads.append(threading.get_ident())
            result = super().maybe_respond(*args, **kwargs)
            done.set()
            return result

    responder = ThreadRecordingResponder()
    svc = _configure(repo, responder=responder)
    th = _thread(svc)

    caller_thread = threading.get_ident()
    asyncio.run(mcp_mod.mcp.call_tool(
        "send_message",
        {"body": "hey @bot", "re": th["threadId"], "mentions": ["bot1"]},
    ))

    assert done.wait(timeout=2), "responder was never scheduled"
    assert len(responder.calls) == 1
    assert seen_threads[0] != caller_thread
