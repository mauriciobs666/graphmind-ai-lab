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


def _configure(repo, *, actor="u1"):
    clock = itertools.count(1000)
    ids = (f"id{n}" for n in itertools.count(1))
    svc = Services(repo, clock=lambda: next(clock), id_gen=lambda: next(ids))
    mcp_mod.configure(
        svc, context_provider=lambda: CallContext(ws="test", actor=actor)
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


def test_tool_discovery_lists_three_tools(repo):
    _configure(repo)
    tools = asyncio.run(mcp_mod.mcp.list_tools())
    assert {t.name for t in tools} == {"send_message", "read_messages", "create_thread"}


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


def test_send_message_prioritises_mention(repo):
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
    assert rows[0]["text"] == "hey bob"
    assert rows[0]["isMention"] is True


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
