"""MCP transport tests — in-memory via FastMCP `list_tools`/`call_tool`.

The MCP tools are thin adapters; these confirm discovery and that a tool call
round-trips through the real service layer against the live `ws:test` graph.
No HTTP server is started — FastMCP is exercised directly.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import threading

import pytest

from falkorchat import mcp as mcp_mod
from falkorchat.config import CallContext
from falkorchat.services import Services

TEST_CTX = CallContext(ws="test", actor="u1")


def _configure(
    repo, *, actor="u1", responder=None, embed_worker=None, trigger=None,
    models=None, ingestion_pipeline=None, detection_repo=None,
):
    """`detection_repo` (document-ingestion2 Stage D) is a SEPARATE opt-in
    from the real `repo` this helper's callers already pass for `Services` —
    most existing tests wire fake `embed_worker`/`ingestion_pipeline` objects
    with no real repo behind them and never call `ingest_document`, so wiring
    `_safe_detect_update`'s repo by default here would schedule a real
    background thread touching `ws:test` in tests that don't expect it. Pass
    `detection_repo=repo` explicitly in a test that actually wants the
    suggested-tier job to run (mirrors `api.py`'s `wired_real_ingestion`
    fixture, which only ever wires a real repo through `create_app`'s own
    `embed_worker.repo`/`ingestion_pipeline.repo` derivation)."""
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
        ingestion_pipeline=ingestion_pipeline,
        repo=detection_repo,
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
        "ingest_document", "ingest_documents", "get_document", "search_documents",
        "delete_document", "list_documents", "get_document_deletion",
        "get_document_history", "list_pending_document_updates",
        "list_document_updates", "confirm_document_update",
        "reject_document_update", "recheck_document_update",
        "list_pending_matches", "list_matches", "confirm_match",
        "reject_match", "recheck_match",
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


def _thread_with_sync_schedule(repo, worker):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    svc = _configure(repo, embed_worker=worker)
    ch = svc.create_channel(TEST_CTX, name="general")
    return svc.create_thread(TEST_CTX, channel_id=ch["channelId"], title="hi")["threadId"]


def test_send_message_over_max_text_len_is_rejected_naming_the_limit(repo):
    """K-099: an 8001-character post is refused loudly, with the limit in the
    error, writing nothing and scheduling no embed job. Since K-101 the MCP
    signature's own bound refuses it before the service's cap is reached
    (the service cap is pinned in `test_services.py`). Literal 8001/8000,
    never `MAX_TEXT_LEN` (root REVIEW.md)."""
    from mcp.server.fastmcp.exceptions import ToolError

    worker = RecordingWorker()
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *args: fn(*args)  # synchronous
    try:
        tid = _thread_with_sync_schedule(repo, worker)
        with pytest.raises(ToolError, match="body\n  String should have at most 8000 characters"):
            asyncio.run(mcp_mod.mcp.call_tool(
                "send_message", {"body": "x" * 8001, "re": tid}
            ))
        rows = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
            "read_messages", {"re": tid, "since": 0, "advance": False}
        )))
    finally:
        mcp_mod._schedule = original

    assert rows == []
    assert worker.calls == []


def test_send_message_at_exactly_max_text_len_is_accepted_and_embedded(repo):
    worker = RecordingWorker()
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *args: fn(*args)  # synchronous
    try:
        tid = _thread_with_sync_schedule(repo, worker)
        posted = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
            "send_message", {"body": "x" * 8000, "re": tid}
        )))
    finally:
        mcp_mod._schedule = original

    assert len(posted["text"]) == 8000
    assert worker.calls == [("test", posted["msgId"], "x" * 8000)]


# ── K-101: stored-value bounds reach MCP through the service (no database) ──────


@pytest.mark.parametrize(
    "tool,args,match",
    [
        ("send_message", {"body": "   ", "re": "t1"}, "message text"),
        ("send_message", {"body": "\n\t ", "re": "t1"}, "message text"),
    ],
    ids=["body-blank", "body-newline-tab"],
)
def test_mcp_tool_refuses_out_of_bounds_stored_value_before_any_repo_call(
    tool, args, match,
):
    """Whitespace-only text passes the signature's `min_length=1` (as it
    passes REST's) and is refused by the service's non-blank check
    (`InputBoundError`, K-101 stakeholder decision): the untouchable repo
    proves nothing was read or written, and no embed job was scheduled. Every
    other stored-value bound is now refused earlier, by the signature — see
    `test_client_sees_a_tool_error_naming_the_field_and_the_limit`."""
    from mcp.server.fastmcp.exceptions import ToolError
    from test_services import _UntouchableRepo

    worker = RecordingWorker()
    _configure(_UntouchableRepo(), embed_worker=worker)
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *a: fn(*a)  # synchronous
    try:
        with pytest.raises(ToolError, match=match):
            asyncio.run(mcp_mod.mcp.call_tool(tool, args))
    finally:
        mcp_mod._schedule = original
    assert worker.calls == []


def test_send_message_tool_description_states_the_current_max_text_len(repo):
    """The description is what an MCP client reads; its stated limit must be
    the real one. Literal, never `MAX_TEXT_LEN` (root REVIEW.md)."""
    _configure(repo)
    tools = asyncio.run(mcp_mod.mcp.list_tools())
    desc = next(t.description for t in tools if t.name == "send_message")
    assert "at most 8000 characters" in desc


# ── K-101: bounds declared on the MCP tool signatures (no database) ─────────────
# Every expected value below is a literal, never a `schemas.MAX_*` constant: the
# parity test compares MCP against REST, and both read the same constants, so a
# drifted constant would move both sides together and pass it (root REVIEW.md).


def _tool_schemas() -> dict[str, dict]:
    return {t.name: t.inputSchema for t in asyncio.run(mcp_mod.mcp.list_tools())}


def _arg(schema: dict, arg: str) -> dict:
    """The constraint-bearing node for `arg`: the non-null `anyOf` branch for
    an Optional argument, else the property itself."""
    prop = schema["properties"][arg]
    branches = [b for b in prop.get("anyOf", []) if b.get("type") != "null"]
    return branches[0] if branches else prop


_KEYWORDS = ("minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum")

_LIST_TOOLS_WITH_200_LIMIT = (
    "search_messages", "list_channels", "list_threads", "search_documents",
    "list_documents", "list_pending_document_updates", "list_document_updates",
    "list_pending_matches", "list_matches",
)

EXPECTED_ARG_BOUNDS = {
    ("send_message", "body"): {"minLength": 1, "maxLength": 8000},
    ("send_message", "mentions"): {"maxItems": 50},
    ("read_messages", "since"): {"minimum": 0},
    ("read_messages", "limit"): {"minimum": 1, "maximum": 1000},
    ("create_thread", "title"): {"minLength": 1, "maxLength": 200},
    ("create_channel", "name"): {"minLength": 1, "maxLength": 200},
    ("search_messages", "query"): {"minLength": 1},
    ("search_documents", "query"): {"minLength": 1},
    ("ingest_document", "title"): {"maxLength": 200},
    ("ingest_document", "source_format"): {"minLength": 1, "maxLength": 50},
    ("ingest_document", "source_label"): {"maxLength": 200},
    **{(tool, "limit"): {"minimum": 1, "maximum": 200} for tool in _LIST_TOOLS_WITH_200_LIMIT},
}


@pytest.mark.parametrize(
    "tool,arg", sorted(EXPECTED_ARG_BOUNDS), ids=lambda v: v if isinstance(v, str) else "",
)
def test_tool_input_schema_advertises_the_bound(tool, arg):
    node = _arg(_tool_schemas()[tool], arg)
    advertised = {k: node[k] for k in _KEYWORDS if k in node}
    assert advertised == EXPECTED_ARG_BOUNDS[(tool, arg)]


@pytest.mark.parametrize(
    "tool,arg",
    [
        # RCA §5.2 item 3 "not copied": lookup-only, no stored-value cost.
        ("ingest_document", "produced_by"),
        ("delete_document", "document_id"),
        ("confirm_match", "match_id"),
        # Service-only by RCA Appendix A: the service's own blank refusal is stricter.
        ("ingest_document", "text"),
    ],
)
def test_not_copied_arguments_advertise_no_bound(tool, arg):
    node = _arg(_tool_schemas()[tool], arg)
    assert {k for k in _KEYWORDS if k in node} == set()


def test_ingest_documents_items_stays_untyped_so_per_item_receipts_survive():
    """Typing `items` would make one bad item fail the whole call (RCA §5.1 B)."""
    node = _tool_schemas()["ingest_documents"]["properties"]["items"]
    assert node["type"] == "array"
    assert node["items"] == {"additionalProperties": True, "type": "object"}
    assert {k for k in _KEYWORDS if k in node} == set()


@pytest.mark.parametrize(
    "tool,args,field,message",
    [
        ("read_messages", {"limit": 0}, "limit", "greater than or equal to 1"),
        ("read_messages", {"limit": 1001}, "limit", "less than or equal to 1000"),
        ("read_messages", {"since": -1}, "since", "greater than or equal to 0"),
        ("list_channels", {"limit": -1}, "limit", "greater than or equal to 1"),
        ("search_documents", {"query": "q", "limit": 201}, "limit",
         "less than or equal to 200"),
        ("send_message", {"body": "", "re": "t1"}, "body", "at least 1 character"),
        ("send_message", {"body": "x" * 8001, "re": "t1"}, "body",
         "at most 8000 characters"),
        ("search_messages", {"query": ""}, "query", "at least 1 character"),
        ("create_channel", {"name": "x" * 201}, "name", "at most 200 characters"),
        ("ingest_document", {"text": "doc", "source_format": "x" * 51}, "source_format",
         "at most 50 characters"),
        ("send_message",
         {"body": "hi", "re": "t1", "mentions": [f"m{i}" for i in range(51)]},
         "mentions", "at most 50 items"),
        ("create_channel", {"name": ""}, "name", "at least 1 character"),
        ("create_thread", {"channel_id": "c1", "title": ""}, "title", "at least 1 character"),
        ("create_thread", {"channel_id": "c1", "title": "x" * 201}, "title",
         "at most 200 characters"),
        ("ingest_document", {"text": "doc", "title": "x" * 201}, "title",
         "at most 200 characters"),
        ("ingest_document", {"text": "doc", "source_format": ""}, "source_format",
         "at least 1 character"),
        ("ingest_document", {"text": "doc", "source_label": "x" * 201}, "source_label",
         "at most 200 characters"),
    ],
    ids=["read-limit-0", "read-limit-1001", "since-neg", "channels-limit-neg",
         "docsearch-limit-201", "body-empty", "body-8001", "query-empty", "name-201",
         "format-51", "mentions-51", "name-empty", "title-empty", "title-201",
         "doc-title-201", "format-empty", "label-201"],
)
def test_client_sees_a_tool_error_naming_the_field_and_the_limit(tool, args, field, message):
    """What an MCP client receives over the wire for an annotated bound: a
    `CallToolResult` with `isError` set and one text block naming the field and
    the limit (FastMCP's pydantic argument model, `mcp` 1.28.1). The refusal
    happens before the tool body runs, so the untouchable repo proves nothing
    was read, written or scheduled."""
    from mcp.shared.memory import create_connected_server_and_client_session
    from test_services import _UntouchableRepo

    worker = RecordingWorker()
    _configure(_UntouchableRepo(), embed_worker=worker)

    async def call():
        async with create_connected_server_and_client_session(
            mcp_mod.mcp._mcp_server,
        ) as client:
            return await client.call_tool(tool, args)

    result = asyncio.run(call())
    assert result.isError is True
    [block] = result.content
    assert f"validation error for {tool}Arguments\n{field}\n" in block.text
    assert message in block.text
    assert worker.calls == []


class _RecordingServices:
    """Answers every `Services` method with an empty result and records the
    keyword arguments it was called with: isolates the MCP signature layer, so
    an accepted call proves the annotation let the value through unchanged."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __getattr__(self, name):
        def method(ctx, **kwargs):
            self.calls.append((name, kwargs))
            if name == "ingest_document":
                return {"documentId": "d1", "status": "processing", "autoSuperseded": True}
            if name.startswith(("list_", "search_", "read_")) or name == "ingest_documents":
                return []
            return {"ok": True}
        return method


@pytest.mark.parametrize(
    "tool,args,method,expected",
    [
        ("read_messages", {"limit": 1000}, "read_messages", {"limit": 1000}),
        ("read_messages", {"limit": 1, "since": 0}, "read_messages",
         {"limit": 1, "since": 0}),
        ("read_messages", {"re": "t1", "since": None}, "read_messages", {"since": None}),
        ("list_channels", {"limit": 200}, "list_channels", {"limit": 200}),
        ("list_matches", {"limit": 1}, "list_matches", {"limit": 1}),
        ("search_documents", {"query": "q", "limit": 200}, "search_documents",
         {"query": "q", "limit": 200}),
        ("search_messages", {"query": "q"}, "search_messages", {"query": "q"}),
        ("send_message", {"body": "x" * 8000, "re": "t1"}, "post_message",
         {"text": "x" * 8000}),
        ("send_message",
         {"body": "hi", "re": "t1", "mentions": [f"m{i}" for i in range(50)]},
         "post_message", {"mentions": [f"m{i}" for i in range(50)]}),
        ("create_channel", {"name": "x" * 200}, "create_channel", {"name": "x" * 200}),
        ("create_thread", {"channel_id": "c1", "title": "x"}, "create_thread",
         {"title": "x"}),
        ("ingest_document",
         {"text": "doc", "title": "x" * 200, "source_format": "x" * 50,
          "source_label": "x" * 200},
         "ingest_document",
         {"title": "x" * 200, "source_format": "x" * 50, "source_label": "x" * 200}),
        ("ingest_document", {"text": "doc", "source_format": "x"}, "ingest_document",
         {"source_format": "x"}),
        # Not copied: a 201-char producer still reaches the service, which owns it.
        ("ingest_document", {"text": "doc", "produced_by": "x" * 201}, "ingest_document",
         {"produced_by": "x" * 201}),
        # Untyped: a 21-item batch reaches the service, whose BatchTooLargeError owns it.
        ("ingest_documents", {"items": [{"text": "d"}] * 21}, "ingest_documents",
         {"documents": [{"text": "d"}] * 21}),
    ],
    ids=["read-limit-1000", "read-limit-1-since-0", "since-none", "channels-200",
         "matches-1", "docsearch-200", "query-1", "body-8000", "mentions-50",
         "name-200", "title-1", "doc-fields-at-max", "format-1", "produced-by-201",
         "batch-21"],
)
def test_value_exactly_at_the_bound_reaches_the_service(tool, args, method, expected):
    """No annotation is tighter than REST/the service: the boundary value
    itself passes the MCP layer and reaches the service unchanged."""
    svc = _RecordingServices()
    mcp_mod.configure(svc, context_provider=lambda: TEST_CTX)
    asyncio.run(mcp_mod.mcp.call_tool(tool, args))
    [(called, kwargs)] = svc.calls
    assert called == method
    assert {k: kwargs[k] for k in expected} == expected


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


def test_ingest_document_produced_by_resolves_agent_via_mcp_tool(repo):
    """Sibling of `test_ingest_document_then_get_document_round_trips` — the
    same round trip, but attributed via `produced_by` to a registered Agent
    rather than the configured actor."""
    repo.ensure_agent("test", agent_id="bot1", name="Bot One")
    _configure(repo, actor="ghost")  # actor is unresolvable; produced_by must still work

    async def scenario():
        posted = _unwrap(await mcp_mod.mcp.call_tool(
            "ingest_document", {"text": "hello world", "produced_by": "bot1"}
        ))
        got = _unwrap(await mcp_mod.mcp.call_tool(
            "get_document", {"document_id": posted["documentId"]}
        ))
        return posted, got

    posted, got = asyncio.run(scenario())
    assert posted["status"] == "processing"
    assert got["sourceKind"] == "agent"
    assert got["ingestedByKind"] == "Agent"
    assert got["ingestedById"] == "bot1"


def test_ingest_document_produced_by_unresolvable_agent_errors(repo):
    """Sibling of `test_ingest_document_unknown_actor_errors` — a valid actor
    must not rescue an unresolvable `produced_by` (`AgentNotFoundError`, no
    silent fallback)."""
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)  # actor "u1" IS resolvable

    with pytest.raises(Exception):
        asyncio.run(mcp_mod.mcp.call_tool(
            "ingest_document", {"text": "hello", "produced_by": "ghost-agent"}
        ))


def test_get_document_missing_returns_none(repo):
    _configure(repo)

    got = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("get_document", {"document_id": "nope"})
    ))
    assert got is None


# ── §14.7 Delete + list (document-ingestion2 Stage A, FR-4/FR-8) ────────────


def test_list_documents_tool(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    async def scenario():
        posted = _unwrap(await mcp_mod.mcp.call_tool(
            "ingest_document", {"text": "hello"}
        ))
        rows = _unwrap(await mcp_mod.mcp.call_tool("list_documents", {}))
        return posted, rows

    posted, rows = asyncio.run(scenario())
    assert posted["documentId"] in [d["documentId"] for d in rows]


def test_delete_document_tool_removes_content(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    async def scenario():
        posted = _unwrap(await mcp_mod.mcp.call_tool(
            "ingest_document", {"text": "hello"}
        ))
        await mcp_mod.mcp.call_tool(
            "delete_document", {"document_id": posted["documentId"]}
        )
        got = _unwrap(await mcp_mod.mcp.call_tool(
            "get_document", {"document_id": posted["documentId"]}
        ))
        return got

    got = asyncio.run(scenario())
    assert got is None


def test_delete_document_tool_errors_for_unknown_document_id(repo):
    _configure(repo)

    with pytest.raises(Exception):
        asyncio.run(
            mcp_mod.mcp.call_tool("delete_document", {"document_id": "nope"})
        )


def test_get_document_deletion_tool_after_delete(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    async def scenario():
        posted = _unwrap(await mcp_mod.mcp.call_tool(
            "ingest_document", {"text": "hello"}
        ))
        await mcp_mod.mcp.call_tool(
            "delete_document", {"document_id": posted["documentId"]}
        )
        deletion = _unwrap(await mcp_mod.mcp.call_tool(
            "get_document_deletion", {"document_id": posted["documentId"]}
        ))
        return posted, deletion

    posted, deletion = asyncio.run(scenario())
    assert deletion["documentId"] == posted["documentId"]
    assert deletion["deletedBy"] == "u1"


def test_get_document_deletion_tool_none_when_never_deleted(repo):
    _configure(repo)

    got = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("get_document_deletion", {"document_id": "nope"})
    ))
    assert got is None


# ── §14.8 SUPERSEDES review surface (document-ingestion2 Stage B) ───────────


def _seed_document_update(repo, *, status="pending"):
    repo.ensure_user("test", user_id="u2", display_name="Bob")
    repo.create_document(
        "test", document_id="d1", title="new", text="hello v2",
        source_format="text", ingested_by="u2", created_at=100,
        chunks=[{"chunkId": "dc1", "text": "hello v2", "seq": 0}],
    )
    repo.create_document(
        "test", document_id="d2", title="old", text="hello v1",
        source_format="text", ingested_by="u2", created_at=50,
        chunks=[{"chunkId": "dc2", "text": "hello v1", "seq": 0}],
    )
    repo.create_or_reopen_supersede_suggestion(
        "test", new_document_id="d1", candidate_document_id="d2", match_id="dm1",
        status=status, confidence=0.8, technique="shingled_jaccard", created_at=100,
    )


def test_list_pending_document_updates_tool(repo):
    _seed_document_update(repo, status="pending")
    _configure(repo)

    rows = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("list_pending_document_updates", {})
    ))

    assert [m["matchId"] for m in rows] == ["dm1"]


def test_list_document_updates_tool_filters_by_status(repo):
    _seed_document_update(repo, status="pending")
    _configure(repo)

    pending = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("list_document_updates", {"status": "pending"})
    ))
    confirmed = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("list_document_updates", {"status": "confirmed"})
    ))

    assert [m["matchId"] for m in pending] == ["dm1"]
    assert confirmed == []


def test_confirm_document_update_tool(repo):
    _seed_document_update(repo, status="pending")
    _configure(repo)

    result = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("confirm_document_update", {"match_id": "dm1"})
    ))

    assert result["status"] == "confirmed"


def test_confirm_document_update_tool_errors_for_unknown_match_id(repo):
    _configure(repo)

    with pytest.raises(Exception):
        asyncio.run(
            mcp_mod.mcp.call_tool("confirm_document_update", {"match_id": "nope"})
        )


def test_reject_then_recheck_document_update_tool_round_trips(repo):
    _seed_document_update(repo, status="pending")
    _configure(repo)

    async def scenario():
        rejected = _unwrap(await mcp_mod.mcp.call_tool(
            "reject_document_update", {"match_id": "dm1"}
        ))
        rechecked = _unwrap(await mcp_mod.mcp.call_tool(
            "recheck_document_update", {"match_id": "dm1"}
        ))
        return rejected, rechecked

    rejected, rechecked = asyncio.run(scenario())
    assert rejected["status"] == "rejected"
    assert rechecked["status"] == "pending"


def test_get_document_history_tool_after_confirm(repo):
    _seed_document_update(repo, status="pending")
    _configure(repo)

    async def scenario():
        await mcp_mod.mcp.call_tool(
            "confirm_document_update", {"match_id": "dm1"}
        )
        return _unwrap(await mcp_mod.mcp.call_tool(
            "get_document_history", {"document_id": "d1"}
        ))

    rows = asyncio.run(scenario())
    assert [row["documentId"] for row in rows] == ["d2", "d1"]


# ── §14.6 Entity fusion — SAME_AS review surface (K-050 M5 Stage 4) ──────────


def _seed_match(repo, *, status="pending"):
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
        status=status, confidence=2.0, technique="fuzzy_fulltext", created_at=100,
    )


def test_list_pending_matches_tool(repo):
    _seed_match(repo, status="pending")
    _configure(repo)

    rows = _unwrap(asyncio.run(mcp_mod.mcp.call_tool("list_pending_matches", {})))

    assert [m["matchId"] for m in rows] == ["m1"]


def test_list_matches_tool_filters_by_status(repo):
    _seed_match(repo, status="pending")
    _configure(repo)

    pending = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("list_matches", {"status": "pending"})
    ))
    confirmed = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("list_matches", {"status": "confirmed"})
    ))

    assert [m["matchId"] for m in pending] == ["m1"]
    assert confirmed == []


def test_confirm_match_tool(repo):
    _seed_match(repo, status="pending")
    _configure(repo)

    result = _unwrap(asyncio.run(
        mcp_mod.mcp.call_tool("confirm_match", {"match_id": "m1"})
    ))

    assert result["status"] == "confirmed"


def test_confirm_match_tool_errors_for_unknown_match_id(repo):
    _configure(repo)

    with pytest.raises(Exception):
        asyncio.run(mcp_mod.mcp.call_tool("confirm_match", {"match_id": "nope"}))


def test_reject_then_recheck_match_tool_round_trips(repo):
    _seed_match(repo, status="pending")
    _configure(repo)

    async def scenario():
        rejected = _unwrap(await mcp_mod.mcp.call_tool(
            "reject_match", {"match_id": "m1"}
        ))
        rechecked = _unwrap(await mcp_mod.mcp.call_tool(
            "recheck_match", {"match_id": "m1"}
        ))
        return rejected, rechecked

    rejected, rechecked = asyncio.run(scenario())
    assert rejected["status"] == "rejected"
    assert rechecked["status"] == "pending"


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


# ── Pass 3 MAJOR 2 fix (a): a thread-start failure must not escape a tool ────


def test_default_schedule_swallows_a_thread_start_failure_and_logs(monkeypatch, caplog):
    # Simulates the real OS failure mode (`RuntimeError: can't start new
    # thread`, near a ulimit/cgroup ceiling) without actually exhausting OS
    # threads — `threading.Thread` itself is monkeypatched to raise on start.
    class _FailingThread:
        def __init__(self, target, args, daemon):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(mcp_mod.threading, "Thread", _FailingThread)
    calls: list[str] = []

    def _fn(marker):
        calls.append(marker)  # pragma: no cover — never reached, start() raises first

    with caplog.at_level(logging.ERROR):
        mcp_mod._default_schedule(_fn, "c1")  # must not raise

    assert calls == []  # the job never ran — thread creation itself failed
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "mcp background schedule failed to start a thread" in error_records[0].getMessage()
    assert error_records[0].exc_info is not None
    assert "can't start new thread" in str(error_records[0].exc_info[1])


def test_default_schedule_still_runs_the_job_on_the_happy_path(monkeypatch):
    # Regression guard: the try/except must not swallow the real dispatch —
    # confirmed against the REAL threading.Thread (no monkeypatch here),
    # joined so the assertion isn't racing the background thread.
    calls: list[str] = []
    threads: list[threading.Thread] = []
    original_thread = mcp_mod.threading.Thread

    def _tracking_thread(target, args, daemon):
        t = original_thread(target=target, args=args, daemon=daemon)
        threads.append(t)
        return t

    monkeypatch.setattr(mcp_mod.threading, "Thread", _tracking_thread)

    mcp_mod._default_schedule(calls.append, "ran")
    for t in threads:
        t.join(timeout=5)

    assert calls == ["ran"]


# ── K-050 M5 Stage 3: extraction scheduling ──────────────────────────────────


class RecordingIngestionPipeline:
    """Records extract_chunk calls scheduled off-band (mirrors
    RecordingChunkWorker above, but for extraction rather than embedding)."""

    def __init__(self):
        self.calls: list[tuple] = []

    def extract_chunk(self, ws, *, chunk_id, document_id, text):
        self.calls.append((ws, chunk_id, document_id, text))


def test_ingest_document_schedules_every_chunk_for_extraction(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    pipeline = RecordingIngestionPipeline()
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *args: fn(*args)  # synchronous
    try:
        _configure(repo, ingestion_pipeline=pipeline)
        posted = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
            "ingest_document", {"text": "First para.\n\nSecond, longer paragraph."}
        )))
    finally:
        mcp_mod._schedule = original

    assert len(pipeline.calls) == posted["chunkCount"]
    for ws, _chunk_id, document_id, _text in pipeline.calls:
        assert ws == "test"
        assert document_id == posted["documentId"]


def test_ingest_document_schedules_extraction_and_embedding_independently(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    embed_worker = RecordingChunkWorker()
    pipeline = RecordingIngestionPipeline()
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *args: fn(*args)  # synchronous
    try:
        _configure(repo, embed_worker=embed_worker, ingestion_pipeline=pipeline)
        posted = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
            "ingest_document", {"text": "One paragraph of text."}
        )))
    finally:
        mcp_mod._schedule = original

    assert len(embed_worker.calls) == posted["chunkCount"]
    assert len(pipeline.calls) == posted["chunkCount"]
    assert {cid for _ws, cid, _text in embed_worker.calls} == {
        cid for _ws, cid, _doc, _text in pipeline.calls
    }


def test_ingest_document_with_no_ingestion_pipeline_schedules_nothing(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)  # no ingestion_pipeline

    posted = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
        "ingest_document", {"text": "hello"}
    )))
    assert posted["status"] == "processing"  # succeeds; nothing to assert-not-crash on


# ── document-ingestion2 Stage D (FR-2/AC-1/AC-2) — update detection ─────────
# MCP-side parity for the REST integration tests in test_api.py
# (`wired_real_ingestion`) — same scenarios, same assertions, via `call_tool`.


def test_ingest_document_tool_byte_identical_modulo_whitespace_auto_supersedes(repo):
    """AC-1 via MCP — provable synchronously, no scheduling override needed
    at all (the auto tier is folded into the synchronous write, not a
    background job)."""
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    async def scenario():
        first = _unwrap(await mcp_mod.mcp.call_tool(
            "ingest_document",
            {"text": "Hello   World\nThis is the content.", "title": "Doc"},
        ))
        second = _unwrap(await mcp_mod.mcp.call_tool(
            "ingest_document",
            {"text": "hello world\nthis is the content.", "title": "Doc v2"},
        ))
        updates = _unwrap(await mcp_mod.mcp.call_tool("list_document_updates", {}))
        return first, second, updates

    first, second, updates = asyncio.run(scenario())
    assert second["autoSuperseded"] is True
    assert second["supersededDocumentId"] == first["documentId"]
    match = next(u for u in updates if u["documentB"] == first["documentId"])
    assert match["status"] == "confirmed"
    assert match["confidence"] == 1.0


def test_ingest_document_tool_plausible_edit_produces_pending_suggestion(repo):
    """AC-2 via MCP — the suggested-tier background job, run synchronously
    via the `mcp_mod._schedule` override (mirrors `test_ingest_document_
    schedules_every_chunk_for_extraction`'s own override pattern)."""
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    base_text = " ".join(f"paragraph{i} about the quarterly report" for i in range(60))
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *args: fn(*args)  # synchronous
    try:
        _configure(repo, detection_repo=repo)

        async def scenario():
            first = _unwrap(await mcp_mod.mcp.call_tool(
                "ingest_document", {"text": base_text, "title": "Report"}
            ))
            edited_text = (
                base_text + " with one new closing sentence added at the very end"
            )
            second = _unwrap(await mcp_mod.mcp.call_tool(
                "ingest_document", {"text": edited_text, "title": "Report"}
            ))
            pending = _unwrap(await mcp_mod.mcp.call_tool(
                "list_pending_document_updates", {}
            ))
            return first, second, pending

        first, second, pending = asyncio.run(scenario())
    finally:
        mcp_mod._schedule = original

    assert second["autoSuperseded"] is False
    match = next(
        (
            u for u in pending
            if u["documentA"] == second["documentId"]
            and u["documentB"] == first["documentId"]
        ),
        None,
    )
    assert match is not None, f"expected a pending suggestion, got {pending}"
    assert match["technique"] == "shingled_jaccard_overlap"


def test_ingest_document_tool_with_no_detection_repo_schedules_nothing(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *args: fn(*args)  # synchronous
    try:
        _configure(repo)  # detection_repo defaults to None

        posted = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
            "ingest_document", {"text": "hello there general text"}
        )))
    finally:
        mcp_mod._schedule = original

    assert posted["status"] == "processing"  # succeeds; nothing scheduled to crash on


# ── K-050 M5 Stage 6a: bulk ingestion (FR-11) ────────────────────────────────────


def test_ingest_documents_tool_returns_one_receipt_per_item(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    async def scenario():
        receipts = _unwrap(await mcp_mod.mcp.call_tool(
            "ingest_documents",
            {"items": [
                {"text": "first document", "title": "First"},
                {"text": "second document", "title": "Second"},
            ]},
        ))
        got0 = _unwrap(await mcp_mod.mcp.call_tool(
            "get_document", {"document_id": receipts[0]["documentId"]}
        ))
        got1 = _unwrap(await mcp_mod.mcp.call_tool(
            "get_document", {"document_id": receipts[1]["documentId"]}
        ))
        return receipts, got0, got1

    receipts, got0, got1 = asyncio.run(scenario())
    assert len(receipts) == 2
    assert all(rec["status"] == "processing" for rec in receipts)
    assert got0["text"] == "first document"
    assert got1["text"] == "second document"


def test_ingest_documents_tool_isolates_a_per_item_failure(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    receipts = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
        "ingest_documents",
        {"items": [{"text": "good document"}, {"text": "   "}]},
    )))

    assert receipts[0]["status"] == "processing"
    assert receipts[1]["status"] == "error"
    assert receipts[1]["errorType"] == "EmptyDocumentError"


def test_ingest_documents_tool_isolates_a_malformed_item_missing_text(repo):
    # Pass 6 review BLOCKER regression: MCP's `items: list[dict]` has no
    # schema layer (unlike REST's `IngestDocumentIn`), so a missing "text"
    # key used to raise a bare KeyError that aborted the whole batch call —
    # driven here through the REAL mcp.call_tool path (not a direct
    # Services call), asserting the tool returns receipts rather than
    # raising a ToolError, and that the good sibling ahead of the malformed
    # item still succeeds with its own receipt intact.
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    receipts = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
        "ingest_documents",
        {"items": [
            {"text": "good document one, has enough content."},
            {"title": "oops no text key"},
            {"text": "good document two."},
        ]},
    )))

    assert len(receipts) == 3
    assert receipts[0]["status"] == "processing"
    assert receipts[1]["status"] == "error"
    assert receipts[1]["errorType"] == "MalformedItemError"
    assert receipts[2]["status"] == "processing"
    # the two good documents are independently retrievable — not just
    # "not crashed," genuinely still written and returned correctly
    assert receipts[0]["documentId"] != receipts[2]["documentId"]


def test_ingest_documents_tool_produced_by_per_item(repo):
    """Sibling of `test_ingest_documents_tool_returns_one_receipt_per_item` —
    one item carrying a resolvable `produced_by`, one carrying none (falls
    back to the configured actor), each receipt reflecting its own
    resolution independently."""
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.ensure_agent("test", agent_id="bot1", name="Bot One")
    _configure(repo)

    async def scenario():
        receipts = _unwrap(await mcp_mod.mcp.call_tool(
            "ingest_documents",
            {"items": [
                {"text": "from the agent", "produced_by": "bot1"},
                {"text": "from the actor"},
            ]},
        ))
        got0 = _unwrap(await mcp_mod.mcp.call_tool(
            "get_document", {"document_id": receipts[0]["documentId"]}
        ))
        got1 = _unwrap(await mcp_mod.mcp.call_tool(
            "get_document", {"document_id": receipts[1]["documentId"]}
        ))
        return got0, got1

    got0, got1 = asyncio.run(scenario())
    assert got0["ingestedByKind"] == "Agent"
    assert got0["ingestedById"] == "bot1"
    assert got1["ingestedByKind"] == "User"
    assert got1["ingestedById"] == "u1"


def test_ingest_documents_tool_isolates_a_non_string_produced_by_item(repo):
    """Sibling of `test_ingest_documents_tool_isolates_a_malformed_item_missing_text`
    — a non-string, non-`None` `produced_by` isolates to a `MalformedItemError`
    receipt; the batch's other items still process."""
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)

    receipts = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
        "ingest_documents",
        {"items": [
            {"text": "good document"}, {"text": "bad producer", "produced_by": 123},
        ]},
    )))

    assert receipts[0]["status"] == "processing"
    assert receipts[1]["status"] == "error"
    assert receipts[1]["errorType"] == "MalformedItemError"


def test_ingest_documents_tool_schedules_every_chunk_of_every_item_for_embedding(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    worker = RecordingChunkWorker()
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *args: fn(*args)  # synchronous
    try:
        _configure(repo, embed_worker=worker)
        receipts = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
            "ingest_documents",
            {"items": [
                {"text": "First para.\n\nSecond, longer paragraph."},
                {"text": "Another document entirely."},
            ]},
        )))
    finally:
        mcp_mod._schedule = original

    total_chunks = sum(rec["chunkCount"] for rec in receipts)
    assert len(worker.calls) == total_chunks
    assert {ws for ws, _cid, _text in worker.calls} == {"test"}


def test_ingest_documents_tool_does_not_schedule_for_a_failed_item(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    worker = RecordingChunkWorker()
    pipeline = RecordingIngestionPipeline()
    original = mcp_mod._schedule
    mcp_mod._schedule = lambda fn, *args: fn(*args)  # synchronous
    try:
        _configure(repo, embed_worker=worker, ingestion_pipeline=pipeline)
        receipts = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
            "ingest_documents",
            {"items": [{"text": "good document"}, {"text": "   "}]},
        )))
    finally:
        mcp_mod._schedule = original

    assert receipts[0]["status"] == "processing"
    assert receipts[1]["status"] == "error"
    assert len(worker.calls) == receipts[0]["chunkCount"]
    assert len(pipeline.calls) == receipts[0]["chunkCount"]


def test_ingest_documents_tool_with_no_wiring_schedules_nothing(repo):
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    _configure(repo)  # no embed_worker/ingestion_pipeline

    receipts = _unwrap(asyncio.run(mcp_mod.mcp.call_tool(
        "ingest_documents", {"items": [{"text": "hello"}]}
    )))
    assert receipts[0]["status"] == "processing"  # succeeds, nothing to crash on


class _StubQueryEmbedder:
    def embed(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class _StubEmbeddingGateway:
    def embedder(self, kind, *, ws=None):
        return _StubQueryEmbedder()


def test_search_documents_tool_returns_ranked_chunks(repo, fresh_vector_index):
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
