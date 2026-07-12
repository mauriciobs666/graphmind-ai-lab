"""Tests for the MCP-client seam (U10 / FR-5c) against a stub in-memory MCP server.

`McpToolClient` lists + calls tools on an external MCP server and registers each into a
`ToolRegistry` as an `McpTool`, so an MCP-exposed tool dispatches through the **same**
`registry.dispatch(name, args, *, ctx, run)` path as a built-in — indistinguishable to
`executor._run_agent_node`. No real external server is started: a stub `FastMCP` server is
wired over the library's in-memory transport (`create_connected_server_and_client_session`).
Wiring a real external server is deferred (§4).
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from falkorchat.config import CallContext
from falkorchat.tools import McpToolClient, ToolRegistry

CTX = CallContext(ws="test", actor="u1")

# ── a stub external MCP server (two tools) ───────────────────────────────────
stub_server = FastMCP("stub-external-server")


@stub_server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@stub_server.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"hello {name}"


def _connect():
    return create_connected_server_and_client_session(stub_server)


# ── list + call directly on the client ───────────────────────────────────────

def test_client_lists_tools_as_function_schemas():
    with McpToolClient(_connect) as client:
        schemas = client.list_tools()

    by_name = {s["function"]["name"]: s for s in schemas}
    assert {"add", "greet"} <= set(by_name)
    add_schema = by_name["add"]
    assert add_schema["type"] == "function"
    assert add_schema["function"]["description"] == "Add two integers."
    # the server's JSON input schema rides through as the offered parameters
    assert "a" in add_schema["function"]["parameters"]["properties"]


def test_client_calls_a_tool_and_returns_a_string():
    with McpToolClient(_connect) as client:
        out = client.call_tool("add", {"a": 2, "b": 3})
        greeting = client.call_tool("greet", {"name": "Ada"})

    # FastMCP wraps scalar returns as structured {"result": ...}; flattened to a string
    assert json.loads(out)["result"] == 5
    assert "hello Ada" in greeting


# ── registry integration: MCP tool dispatches through the same path as a built-in ─

class _BuiltinEcho:
    name = "echo"
    schema = {"type": "function", "function": {"name": "echo", "parameters": {}}}

    def run(self, arguments, *, ctx, run):
        return f"echo:{arguments.get('v')}"


def test_mcp_tool_registers_and_dispatches_like_a_builtin():
    registry = ToolRegistry([_BuiltinEcho()])
    with McpToolClient(_connect) as client:
        registered = client.register_into(registry)

        # both the built-in and the MCP-exposed tools are now uniform in the registry
        assert set(registered) == {"add", "greet"}
        assert {"echo", "add", "greet"} <= set(registry.names())

        # schema(name) works for the MCP tool exactly as for a built-in
        assert registry.schema("add")["function"]["name"] == "add"

        # dispatch(name, args, *, ctx, run) — same path for built-in and MCP tool
        assert registry.dispatch("echo", {"v": 1}, ctx=CTX, run={}) == "echo:1"
        mcp_out = registry.dispatch("add", {"a": 4, "b": 6}, ctx=CTX, run={})
        assert json.loads(mcp_out)["result"] == 10
