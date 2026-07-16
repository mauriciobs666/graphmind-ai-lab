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

import pytest

from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.llm import ChatResult, ToolCall
from falkorchat.services import UnknownMemberError
from falkorchat.tools import HumanHandoffSignal

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


class RaisingRegistry(StubRegistry):
    """Dispatch always raises `exc` — the tool-failure seam (Defect B)."""

    def __init__(self, schemas, exc):
        super().__init__(schemas)
        self._exc = exc

    def dispatch(self, name, arguments, *, ctx, run):
        self.dispatched.append((name, arguments))
        raise self._exc


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


# ── Defect B — a failing tool re-prompts the model, it does NOT kill the run ──

def test_tool_level_error_is_reprompted_not_raised():
    # Defect B (K-022 U14, reproduced live 2/3 runs): the model hallucinated
    # `mentions: ["alice"]` (a displayName it read off the folded thread context), the
    # §4 write raised UnknownMemberError, and — with no try/except around dispatch —
    # the error escaped the node, hit the M-1 fault net and failed the WHOLE run.
    # A tool-level error is a bad *argument*, not an engine fault: it must come back to
    # the model as a bounded re-prompt, exactly like the ungranted/malformed cases.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "post_message", {"text": "hi", "mentions": ["alice"]})]),
        ChatResult(text="posted without the bogus mention"),
    ])
    reg = RaisingRegistry(
        {"post_message": {"type": "function",
                          "function": {"name": "post_message", "parameters": {}}}},
        UnknownMemberError(["alice"]),
    )
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(tools=["post_message"]), {})

    # the node survives and finishes on the model's next turn
    assert result.output == "posted without the bogus mention"
    # the failure is surfaced back to the model as a tool message (a re-prompt)
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and "error" in tool_msgs[0]["content"]
    assert "alice" in tool_msgs[0]["content"]
    # and it is traced — the diagnostic must not vanish with the exception
    assert any(k == "tool_result" and p.startswith("ERROR:")
               for k, p in result.trace)


def test_repeated_tool_errors_are_bounded_by_max_iterations():
    # An error/re-prompt cycle must not spin forever: it burns the SAME per-node
    # iteration budget as any other turn, then terminates gracefully (§7 — only
    # maxSteps hard-fails a run, never a node).
    llm = AlwaysToolLLM(ToolCall("c1", "post_message", {"text": "x"}))
    reg = RaisingRegistry(
        {"post_message": {"type": "function",
                          "function": {"name": "post_message", "parameters": {}}}},
        UnknownMemberError(["ghost"]),
    )
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(
        CTX, RUN, STEP, _config(tools=["post_message"], maxIterations=3), {}
    )

    assert result.on == "done"
    assert len(reg.dispatched) == 3                 # bounded by maxIterations
    assert any(k == "node_note" for k, _ in result.trace)


def test_engine_fault_in_a_tool_still_escapes_to_the_m1_net():
    # The M-1 net must NOT be neutered: only *tool-level* (service-validation) errors
    # become re-prompts. An unexpected engine fault still propagates so the run fails
    # loudly rather than the model being told to "retry" a broken database.
    llm = AlwaysToolLLM(ToolCall("c1", "post_message", {"text": "x"}))
    reg = RaisingRegistry(
        {"post_message": {"type": "function",
                          "function": {"name": "post_message", "parameters": {}}}},
        RuntimeError("engine exploded"),
    )
    ex = _executor(llm=llm, registry=reg)

    with pytest.raises(RuntimeError, match="engine exploded"):
        ex._run_agent_node(CTX, RUN, STEP, _config(tools=["post_message"]), {})


def test_human_handoff_signal_escapes_the_tool_loop_to_the_suspend_path():
    # HumanHandoffSignal is CONTROL FLOW raised *through* dispatch, not an error — a
    # blanket `except Exception` around dispatch would swallow it and break the suspend
    # contract (§2.4). It must pass straight through the node loop.
    llm = AlwaysToolLLM(ToolCall("c1", "human_handoff", {"reason": "need a human"}))
    reg = RaisingRegistry(
        {"human_handoff": {"type": "function",
                           "function": {"name": "human_handoff", "parameters": {}}}},
        HumanHandoffSignal("need a human"),
    )
    ex = _executor(llm=llm, registry=reg)

    with pytest.raises(HumanHandoffSignal):
        ex._run_agent_node(CTX, RUN, STEP, _config(tools=["human_handoff"]), {})


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


# ── Defect A — the node's thread window rides out on StepResult.thread ────────
#
# `_run_agent_node` already reads the recent thread turns to build its own prompt, then
# dropped them on the floor — so `_select_transition` had nothing to hand the guard judge
# and passed the literal `None` (executor.py, `thread=None`). The judge was asked to rule
# on an empty state every turn and correctly biased to suspend forever. These pin the
# restored seam: the turns ride out on the StepResult, at ZERO extra graph reads (m-C).

def _thread_rows(n):
    return [
        {"msgId": f"m{i}", "text": f"turn {i}", "role": "user",
         "createdAt": 1000 + i, "authorId": "u1", "displayName": "Alice",
         "authorType": "User"}
        for i in range(n)
    ]


def test_agent_node_carries_its_thread_window_out_on_the_step_result():
    # T7 — the Defect-A regression pin. The turns the node read are the evidence the
    # guard needs; they must leave the node, not die in it.
    rows = _thread_rows(8)
    svc = StubThreadServices(rows)
    llm = StubChatLLM([ChatResult(text="Thank you for the details, Alice.")])
    ex = _executor(llm=llm, registry=StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA}),
                   services=svc)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {"threadId": "t1"})

    assert result.thread == rows
    assert svc.read_calls == ["t1"]          # T8 (m-C): exactly ONE read, not two


def test_the_thread_window_also_rides_out_when_max_iterations_are_exhausted():
    # The graceful-exhaustion return path is the one a chatty node actually takes; it
    # must carry the same evidence, or the guard goes blind exactly when it matters.
    rows = _thread_rows(3)
    svc = StubThreadServices(rows)
    # never emits a final text → always a tool call → exhausts maxIterations
    llm = StubChatLLM([
        ChatResult(text="thinking", tool_calls=[ToolCall("c1", "graphrag_retrieve", {})])
    ] * 4)
    ex = _executor(llm=llm, registry=StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA}),
                   services=svc)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(maxIterations=2), {"threadId": "t1"})

    assert result.thread == rows


def test_a_node_with_no_thread_context_carries_an_empty_window():
    # The offline stub path / a node with no threadId degrades to `[]` — guards then
    # takes the understanding-only path (never a crash).
    svc = StubThreadServices([])
    llm = StubChatLLM([ChatResult(text="done")])
    ex = _executor(llm=llm, registry=StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA}),
                   services=svc)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {})

    assert result.thread == []
