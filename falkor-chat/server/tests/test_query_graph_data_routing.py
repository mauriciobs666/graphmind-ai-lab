"""K-066 — `query_graph_data` runs on the same model as the agent step that called it.

`docs/plans/salesperson-single-model.md` §3.1/§5 T7: an in-process integration test
through the real dispatch path — a real `WorkflowExecutor`, a real `ToolRegistry`
holding a real `QueryGraphDataTool`, both wired to **one** gateway double. The
double hands out a distinct query client per `requested` ref, so the test pins
*which client answered* the tool's internal structured-completion call, not just
the arguments it was resolved with. Network-free; no FalkorDB.
"""

from __future__ import annotations

import json

from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.llm import ChatResult, ToolCall
from falkorchat.tools import QueryGraphDataTool, ToolRegistry

CTX = CallContext(ws="test", actor="u1")
RUN = {"runId": "r1", "defKey": "salesperson", "defVersion": "8"}
STEP = {"key": "assistant", "type": "agent"}

MINISTRAL = "lmstudio/mistralai/ministral-3-3b"
# The `Repository.read_model_overrides` shape: all four keys, all unset.
NO_OVERRIDES = {
    "agentModel": None, "guardModel": None,
    "embeddingModel": None, "responderModel": None,
}

# A valid `QueryRequest` JSON line (same reply as `test_tools.py`'s
# `_CATALOG_FILTER_REPLY`).
CATALOG_FILTER_REPLY = (
    '{"matches": [{"var": "p", "label": "Product", '
    '"filters": [{"property": "category", "op": "=", "value": "Audio"}]}], '
    '"returns": ["p.name", "p.price"]}'
)
ROWS = [{"p.name": "Bluetooth Speaker", "p.price": 49.99}]


# ── doubles ──────────────────────────────────────────────────────────────────

class _FakeResolution:
    """Just enough of `modelconfig.Resolution` for `_run_agent_node` to read `.source`."""

    def __init__(self, *, source: str) -> None:
        self.source = source


class ScriptedChatLLM:
    """The agent step's chat client: one `query_graph_data` call, then final text."""

    def __init__(self) -> None:
        self._results = [
            ChatResult(tool_calls=[ToolCall(
                id="c1", name="query_graph_data",
                arguments={"question": "Which products cost less than $50?",
                           "dataset": "catalog"},
            )]),
            ChatResult(text="We have the Bluetooth Speaker."),
        ]
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self._results.pop(0) if self._results else ChatResult(text="(exhausted)")


class StubQueryLLM:
    """The tool's structured-completion client: one canned `QueryRequest` reply."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self._text


class RoutingGateway:
    """One `ModelGateway`-shaped double behind both the executor and the tool.

    `resolve_llm` (the node's seam) records the call and returns the scripted chat
    client with a `_FakeResolution`, the tuple `_run_agent_node` unpacks. `llm`
    (the tool's seam) records the call and returns a `StubQueryLLM` keyed by the
    `requested` ref, created on first use — so a tool that resolved anything other
    than the node's model would show up as an extra key, `None` included.
    """

    def __init__(self) -> None:
        self.chat_client = ScriptedChatLLM()
        self.resolve_llm_calls: list[dict] = []
        self.llm_calls: list[dict] = []
        self.query_clients: dict[str | None, StubQueryLLM] = {}

    def has_chat(self) -> bool:
        return True

    def resolve_llm(self, kind, *, requested=None, ws=None, overrides=None):
        self.resolve_llm_calls.append(
            {"kind": kind, "requested": requested, "ws": ws, "overrides": overrides}
        )
        source = "requested" if requested is not None else "default"
        return self.chat_client, _FakeResolution(source=source)

    def llm(self, kind, *, requested=None, ws=None, overrides=None):
        self.llm_calls.append(
            {"kind": kind, "requested": requested, "ws": ws, "overrides": overrides}
        )
        if requested not in self.query_clients:
            self.query_clients[requested] = StubQueryLLM(CATALOG_FILTER_REPLY)
        return self.query_clients[requested]

    def embedder(self, kind, *, requested=None, ws=None, overrides=None):
        raise AssertionError("this path must never resolve an embedder")


class StubQueryServices:
    """Only the `run_structured_query`/`catalog_categories` seams
    `QueryGraphDataTool` needs."""

    def __init__(self, rows) -> None:
        self._rows = rows
        self.structured_query_calls: list[dict] = []

    def run_structured_query(self, ctx, graph_key, compiled, *, timeout=None):
        self.structured_query_calls.append({"graph_key": graph_key, "compiled": compiled})
        return list(self._rows)

    def catalog_categories(self, ctx):
        # K-086 D6: a catalog call reads the category list first; none here.
        return []


# ── T7 ───────────────────────────────────────────────────────────────────────

def test_query_graph_data_is_answered_by_the_calling_steps_model_client():
    gateway = RoutingGateway()
    svc = StubQueryServices(ROWS)
    ex = WorkflowExecutor(
        None, None, models=gateway,
        tool_registry=ToolRegistry([QueryGraphDataTool(svc, models=gateway)]),
        guard_judge=None,
    )
    run = {**RUN, "modelOverrides": NO_OVERRIDES}
    config = {"systemPrompt": "You are the salesperson.", "model": MINISTRAL,
              "tools": ["query_graph_data"], "maxIterations": 4}

    result = ex._run_agent_node(CTX, run, STEP, config, {"threadId": "t1"})

    # (1) the ministral-keyed query client answered the tool's internal call, once
    assert MINISTRAL in gateway.query_clients
    assert len(gateway.query_clients[MINISTRAL].calls) == 1
    # (2) no client was ever created for any other key — `None` (the default) included
    assert set(gateway.query_clients) == {MINISTRAL}
    # (3) the tool resolved with the very override snapshot the node itself used
    [node_call] = gateway.resolve_llm_calls
    [tool_call] = gateway.llm_calls
    assert tool_call["kind"] == "step"
    assert tool_call["overrides"] is node_call["overrides"]
    # and the tool's rows made it back to the agent as the tool turn
    assert result.output == "We have the Bluetooth Speaker."
    tool_turn = gateway.chat_client.calls[1][-1]
    assert tool_turn["role"] == "tool"
    assert json.loads(tool_turn["content"]) == {"items": ROWS}
