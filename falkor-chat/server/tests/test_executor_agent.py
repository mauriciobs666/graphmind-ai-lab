"""Unit tests for the bounded, tool-scoped agent-node loop (U8).

`executor.WorkflowExecutor._run_agent_node` runs a `type:'agent'` step as an agent loop:
it offers **only the node's scoped tool schemas** (AC-6), dispatches granted tool calls via
the injected `tool_registry`, loops bounded by `config.maxIterations`, and ends on a final
text. The security-critical property is AC-6: a call naming a tool **not** in the granted set
is rejected by the dispatcher (defensive — not merely omitted from the offered schemas), and a
malformed call triggers a bounded re-prompt, never a dispatch. On `maxIterations` exhaustion the
node terminates gracefully (it does **not** fail the run).

Both collaborators are injected stubs — no LLM, no network, no graph.
"""

from __future__ import annotations

from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.llm import ChatResult, ToolCall

CTX = CallContext(ws="test", actor="u1")
RUN = {"runId": "r1", "defKey": "triage", "defVersion": "1"}
STEP = {"key": "research", "type": "agent"}

RETRIEVE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "graphrag_retrieve",
        "description": "Retrieve workspace context.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


# ── stubs ────────────────────────────────────────────────────────────────────

class StubChatLLM:
    """Returns the next scripted `ChatResult`; a final text once the script is spent."""

    def __init__(self, results):
        self._results = list(results)
        self.calls: list[dict] = []

    def chat(self, messages, tools):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": list(tools)})
        if self._results:
            return self._results.pop(0)
        return ChatResult(text="(script exhausted)")


class AlwaysToolLLM:
    """Always asks for the same tool call — never emits a final text (drives exhaustion)."""

    def __init__(self, call):
        self._call = call
        self.calls: list[dict] = []

    def chat(self, messages, tools):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": list(tools)})
        return ChatResult(text="thinking", tool_calls=[self._call])


class StubRegistry:
    """Records dispatches; returns a canned result per tool."""

    def __init__(self, schemas, results=None):
        self._schemas = schemas
        self._results = results or {}
        self.dispatched: list[tuple[str, dict]] = []

    def schema(self, name):
        return self._schemas[name]

    def dispatch(self, name, arguments, *, ctx, run):
        self.dispatched.append((name, arguments))
        return self._results.get(name, f"result:{name}")


class StubThreadServices:
    """Records `read_thread` calls; returns a scripted thread transcript."""

    def __init__(self, thread_msgs=None):
        self._thread_msgs = thread_msgs or []
        self.read_calls: list[str] = []

    def read_thread(self, ctx, *, thread_id):
        self.read_calls.append(thread_id)
        return list(self._thread_msgs)


def _executor(*, llm, registry, services=None):
    return WorkflowExecutor(
        services, None, llm=llm, tool_registry=registry, guard_judge=None
    )


def _config(**over):
    cfg = {"systemPrompt": "You are the research agent.",
           "tools": ["graphrag_retrieve"], "maxIterations": 4}
    cfg.update(over)
    return cfg


# ── final-text termination + scoped offering (AC-6) ──────────────────────────

def test_ends_on_final_text_and_offers_only_granted_tools():
    llm = StubChatLLM([ChatResult(text="here is the answer")])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {"threadId": "t1"})

    assert result.output == "here is the answer"
    assert result.on == "done"
    assert reg.dispatched == []
    # AC-6 offering: only the node's granted tool schema is offered to the model
    assert llm.calls[0]["tools"] == [RETRIEVE_SCHEMA]


# ── dispatch a granted tool, loop, then finish ───────────────────────────────

def test_dispatches_granted_tool_then_loops_to_final_text():
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "graphrag_retrieve", {"query": "reset password"})]),
        ChatResult(text="grounded answer"),
    ])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA},
                       results={"graphrag_retrieve": "seed: reset via settings"})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {})

    assert reg.dispatched == [("graphrag_retrieve", {"query": "reset password"})]
    assert result.output == "grounded answer"
    # the tool result is fed back to the model on the next turn
    assert any(m["role"] == "tool" for m in llm.calls[1]["messages"])
    # the loop traces the llm + tool aspects (debug)
    kinds = {k for k, _ in result.trace}
    assert {"llm_response", "tool_call", "tool_result"} <= kinds


# ── AC-6 — an ungranted tool call is rejected by the dispatcher ───────────────

def test_ungranted_tool_call_is_rejected_and_never_dispatched():
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[ToolCall("c9", "delete_everything", {})]),
        ChatResult(text="done safely"),
    ])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {})

    # the ungranted tool is NEVER dispatched (defensive AC-6, not just un-offered)
    assert reg.dispatched == []
    # the rejection is surfaced back to the model as a re-prompt, then it finishes
    assert result.output == "done safely"
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and "not granted" in tool_msgs[0]["content"]


# ── a malformed call re-prompts within the cap, never dispatches ─────────────

def test_malformed_call_reprompts_within_cap_then_dispatches_valid():
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "graphrag_retrieve", {})]),            # missing required "query"
        ChatResult(text="", tool_calls=[
            ToolCall("c2", "graphrag_retrieve", {"query": "vpn"})]),
        ChatResult(text="answer"),
    ])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {})

    # only the well-formed call is dispatched; the malformed one is re-prompted
    assert reg.dispatched == [("graphrag_retrieve", {"query": "vpn"})]
    assert result.output == "answer"


# ── maxIterations exhaustion → graceful termination, not a run failure ────────

def test_max_iterations_exhaustion_terminates_gracefully_with_trace_note():
    call = ToolCall("c1", "graphrag_retrieve", {"query": "loop"})
    llm = AlwaysToolLLM(call)
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(maxIterations=2), {})

    # terminates with its best current text + a trace note (does NOT hard-fail)
    assert result.on == "done"
    assert result.output == "thinking"
    assert len(reg.dispatched) == 2                 # bounded by maxIterations=2
    assert any(k == "node_note" for k, _ in result.trace)


# ── thread-message context folded into the agent-node prompt (AC-2 prereq) ────

def test_agent_node_folds_thread_messages_into_prompt():
    # AC-2: intake must SEE the human's thread turns to judge "enough info". The node
    # reads the thread via services.read_thread and folds role-mapped turns ahead of
    # the CONTEXT block.
    thread = [
        {"role": "user", "text": "reset my password", "authorId": "u1",
         "displayName": "Alice"},
        {"role": "assistant", "text": "what is your username?", "authorId": "assistant",
         "displayName": "Bot"},
    ]
    svc = StubThreadServices(thread)
    llm = StubChatLLM([ChatResult(text="ok")])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg, services=svc)

    ex._run_agent_node(CTX, RUN, STEP, _config(), {"threadId": "t1"})

    assert svc.read_calls == ["t1"]
    msgs = llm.calls[0]["messages"]
    # the human turn maps to role user, the agent turn to role assistant; the speaker
    # is named in the content so the model sees who spoke.
    user_turns = [m for m in msgs if m["role"] == "user"]
    assert any("Alice: reset my password" in m["content"] for m in user_turns)
    assistant_turns = [m for m in msgs if m["role"] == "assistant"]
    assert any("what is your username?" in m["content"] for m in assistant_turns)


def test_agent_node_skips_thread_read_when_no_thread_id():
    # offline unit path: no threadId → no read (network-free stub path preserved).
    svc = StubThreadServices([{"role": "user", "text": "hi", "authorId": "u1"}])
    llm = StubChatLLM([ChatResult(text="ok")])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg, services=svc)

    ex._run_agent_node(CTX, RUN, STEP, _config(), {})

    assert svc.read_calls == []


# ── Option B — emitted msgIds are captured on the StepResult for later linking ─

def test_agent_node_captures_posted_msg_ids_as_emissions():
    # post_message dispatch returns a JSON envelope carrying the posted msgId (the
    # tool no longer links inline — Option B). _run_agent_node buffers those ids on
    # StepResult.emissions so _drive can link StepRun→PRODUCED→Message after _record.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "post_message", {"text": "here you go"})]),
        ChatResult(text="done"),
    ])
    reg = StubRegistry(
        {"post_message": {"type": "function",
                          "function": {"name": "post_message", "parameters": {}}}},
        results={"post_message": '{"posted": "m42", "threadId": "t1"}'},
    )
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(tools=["post_message"]), {})

    assert result.emissions == ["m42"]


def test_agent_node_emissions_empty_when_no_post():
    # a node that only retrieves (no post) emits nothing to link.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "graphrag_retrieve", {"query": "q"})]),
        ChatResult(text="answer"),
    ])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA},
                       results={"graphrag_retrieve": '{"seeds": []}'})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {})

    assert result.emissions == []


# ── the loop is reached through the public step-execution seam ───────────────

def test_execute_step_routes_agent_type_through_the_agent_loop():
    llm = StubChatLLM([ChatResult(text="node output")])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._execute_step(CTX, RUN, STEP, _config(), {})

    assert result.output == "node output"
    assert llm.calls  # the llm was actually driven
