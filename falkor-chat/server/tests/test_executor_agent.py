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

import json

import pytest

from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.llm import ChatResult, ToolCall
from falkorchat.services import UnknownActorError, UnknownMemberError
from falkorchat.tools import AddToCartTool, HumanHandoffSignal, ToolRegistry

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


def test_non_model_correctable_service_error_propagates_to_the_m1_net(caplog):
    # D16 (`m3-executor.md` §2.2): only the model's own bad *arguments* are absorbed as a
    # re-prompt. `UnknownActorError` comes from the DEPLOYMENT (a misconfigured
    # FALKORCHAT_AGENT_ID) — the model cannot fix it, so re-prompting would burn
    # maxIterations and let the run reach `done` having posted nothing (the AC-4 failure
    # signature). It must reach the M-1 fault net instead, and be logged on the way out.
    llm = AlwaysToolLLM(ToolCall("c1", "post_message", {"text": "x"}))
    reg = RaisingRegistry(
        {"post_message": {"type": "function",
                          "function": {"name": "post_message", "parameters": {}}}},
        UnknownActorError("assistant"),
    )
    ex = _executor(llm=llm, registry=reg)

    with caplog.at_level("WARNING", logger="falkorchat.executor"):
        with pytest.raises(UnknownActorError):
            ex._run_agent_node(CTX, RUN, STEP, _config(tools=["post_message"]), {})

    # it escaped on the FIRST failure — no bounded re-prompt loop swallowed it
    assert len(reg.dispatched) == 1
    # and the diagnostic is logged unconditionally (not tracer-gated)
    assert any("post_message" in r.getMessage() and "UnknownActorError" in r.getMessage()
               for r in caplog.records)


def test_a_model_correctable_tool_error_is_logged_even_without_a_tracer(caplog):
    # M-2(a): the trace record is not enough — `_trace_step` uses `_NULL_TRACER` unless
    # `run["trace"]` is set, so on a normal run the only durable diagnostic is this log line.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "post_message", {"text": "hi", "mentions": ["alice"]})]),
        ChatResult(text="recovered"),
    ])
    reg = RaisingRegistry(
        {"post_message": {"type": "function",
                          "function": {"name": "post_message", "parameters": {}}}},
        UnknownMemberError(["alice"]),
    )
    ex = _executor(llm=llm, registry=reg)

    with caplog.at_level("WARNING", logger="falkorchat.executor"):
        result = ex._run_agent_node(CTX, RUN, STEP, _config(tools=["post_message"]), {})

    assert result.output == "recovered"          # still absorbed as a re-prompt
    assert any("UnknownMemberError" in r.getMessage() for r in caplog.records)


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


# ── K-056 — StepResult.toolsUsed carries this node's own dispatched domain tools ──
#
# The breadcrumb (`_assemble_messages`, above) is only as good as the signal it reads:
# `StepResult.toolsUsed` must name every non-`post_message` tool this node execution
# actually dispatched successfully, so `_link_emissions` can persist it onto the
# `Message`(s) this step posted (`repository.link_step_emission`).

def test_step_result_carries_a_dispatched_domain_tool_as_toolsused():
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "graphrag_retrieve", {"query": "reset password"})]),
        ChatResult(text="grounded answer"),
    ])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA},
                       results={"graphrag_retrieve": "seed: reset via settings"})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {})

    assert result.toolsUsed == frozenset({"graphrag_retrieve"})


def test_step_result_toolsused_is_empty_when_only_post_message_was_dispatched():
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

    assert result.toolsUsed == frozenset()


def test_step_result_toolsused_is_empty_when_no_tool_was_dispatched():
    llm = StubChatLLM([ChatResult(text="plain answer, no tool")])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {})

    assert result.toolsUsed == frozenset()


def test_step_result_toolsused_rides_out_on_max_iterations_exhaustion():
    call = ToolCall("c1", "graphrag_retrieve", {"query": "loop"})
    llm = AlwaysToolLLM(call)
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(maxIterations=2), {})

    assert result.toolsUsed == frozenset({"graphrag_retrieve"})


# ── K-048 — _assemble_messages must never emit two consecutive same-role turns ─
#
# A strict-alternation chat template (live-confirmed: LM Studio's Ministral-3B) hard-rejects
# a message list with two consecutive `user` or `assistant` entries. `_assemble_messages`
# unconditionally appended a trailing `user`-role CONTEXT block after the thread turns, which
# crashes whenever the thread's last turn is also `user` — structurally guaranteed on
# `intake`'s first call and every `research`→`answer` handoff (`research` never posts, so
# `answer` always sees a `user`-terminated thread). See docs/plans/assemble-messages-alternation.md.

def test_assemble_messages_thread_ending_in_assistant_is_unchanged():
    # Characterization test — today's already-correct, already-alternating shape must stay
    # byte-for-byte identical after the fix (the coalescing helper is a no-op here since no
    # two adjacent appends ever share a role).
    config = {"systemPrompt": "You are the bot."}
    run_ctx = {"foo": "bar"}
    thread_msgs = [
        {"role": "user", "text": "hi", "authorId": "u1", "displayName": "Alice"},
        {"role": "assistant", "text": "reply", "authorId": "a1", "displayName": "Bot"},
    ]

    messages = WorkflowExecutor._assemble_messages(config, run_ctx, thread_msgs)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    context = json.dumps(run_ctx, separators=(",", ":"), sort_keys=True)
    assert messages[-1]["content"] == f"CONTEXT:\n{context}"
    assert messages[1]["content"] == "Alice: hi"
    assert messages[2]["content"] == "Bot: reply"


def test_assemble_messages_thread_ending_in_user_merges_context_into_it():
    # The confirmed crash shape: thread's last turn is `user` (intake's first call /
    # research→answer handoff), so the trailing CONTEXT block would land right after it —
    # two consecutive `user` messages. Must now merge into one alternating `user` turn.
    config = {"systemPrompt": "You are the bot."}
    run_ctx = {"foo": "bar"}
    thread_msgs = _thread_rows(1)

    messages = WorkflowExecutor._assemble_messages(config, run_ctx, thread_msgs)

    assert [m["role"] for m in messages] == ["system", "user"]
    context = json.dumps(run_ctx, separators=(",", ":"), sort_keys=True)
    assert "Alice: turn 0" in messages[-1]["content"]
    assert f"CONTEXT:\n{context}" in messages[-1]["content"]


def test_assemble_messages_coalesces_consecutive_same_role_thread_turns():
    # Sibling shape (not live-verified, but closed algorithmically by the same helper):
    # two consecutive `user` thread turns with no assistant reply between them — reuses the
    # existing `_thread_rows` fixture (already produces consecutive `role: "user"` rows).
    config = {"systemPrompt": "You are the bot."}
    run_ctx = {"foo": "bar"}
    thread_msgs = _thread_rows(2)

    messages = WorkflowExecutor._assemble_messages(config, run_ctx, thread_msgs)

    assert [m["role"] for m in messages] == ["system", "user"]
    context = json.dumps(run_ctx, separators=(",", ":"), sort_keys=True)
    merged = messages[-1]["content"]
    assert "Alice: turn 0" in merged
    assert "Alice: turn 1" in merged
    assert f"CONTEXT:\n{context}" in merged


# ── K-056 — tool-use breadcrumb reverted (analyst review, MAJOR 1) ────────────
#
# An earlier pass attempted D-1's mitigation (docs/reviews/salesperson-tool-reliability-ml.md
# §4.1/§4.3) by folding a `[verified via <tool>]` breadcrumb into a replayed assistant
# turn backed by `Message.toolsUsed`. Live verification showed it did not reduce
# fabrication and, worse, the model began imitating the breadcrumb's own surface text
# in *fabricated* replies with no tool ever called — a self-authored false-verification
# claim that then replays into the next turn's history
# (docs/reviews/salesperson-tool-reliability-impl.md, MAJOR 1). Reverted here.
# `Message.toolsUsed` remains a pure audit/logging property (StepResult.toolsUsed →
# _link_emissions → repository.link_step_emission → read_thread) — never fed back into
# anything the model reads. These tests pin that absence as a regression guard against
# the tagging being silently reintroduced.

def test_assemble_messages_does_not_tag_an_assistant_turn_even_when_toolsused_is_present():
    config = {"systemPrompt": "You are the bot."}
    run_ctx = {}
    thread_msgs = [
        {"role": "user", "text": "what's the price of the Wireless Mouse Pro?",
         "authorId": "u1", "displayName": "Alice"},
        {"role": "assistant", "text": "It's $29.99.", "authorId": "assistant",
         "displayName": "Bot", "toolsUsed": ["lookup_product_fact"]},
    ]

    messages = WorkflowExecutor._assemble_messages(config, run_ctx, thread_msgs)

    assistant_turns = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_turns) == 1
    assert assistant_turns[0]["content"] == "Bot: It's $29.99."
    assert "verified via" not in assistant_turns[0]["content"]


def test_assemble_messages_ignores_toolsused_regardless_of_shape():
    # `toolsUsed` on a thread row (empty list, or absent entirely — the pre-existing-data
    # shape from before the property existed) never affects assembled content either way.
    config = {"systemPrompt": "You are the bot."}
    run_ctx = {}
    thread_msgs = [
        {"role": "assistant", "text": "hello there!", "authorId": "assistant",
         "displayName": "Bot", "toolsUsed": []},
        {"role": "assistant", "text": "no key here", "authorId": "assistant",
         "displayName": "Bot"},
    ]

    messages = WorkflowExecutor._assemble_messages(config, run_ctx, thread_msgs)

    # consecutive same-role turns coalesce into one (K-048); both texts land in it
    assistant_turns = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_turns) == 1
    assert "hello there!" in assistant_turns[0]["content"]
    assert "no key here" in assistant_turns[0]["content"]
    assert "verified via" not in assistant_turns[0]["content"]


def test_assemble_messages_never_tags_a_user_turn_even_if_toolsused_is_present():
    # Defensive: a malformed row carrying toolsUsed on a `user` turn must still never
    # produce a tag (toolsUsed is meaningless there regardless).
    config = {"systemPrompt": "You are the bot."}
    run_ctx = {}
    thread_msgs = [
        {"role": "user", "text": "hi", "authorId": "u1", "displayName": "Alice",
         "toolsUsed": ["should_never_apply"]},
    ]

    messages = WorkflowExecutor._assemble_messages(config, run_ctx, thread_msgs)

    user_turns = [m for m in messages if m["role"] == "user"]
    assert user_turns[0]["content"].startswith("Alice: hi")
    assert "verified via" not in user_turns[0]["content"]


# ── K-056 — observability: warn on a fact-bearing answer with no tool dispatched ─
#
# The cheap, model-independent signal (ml note §4.1/§5, generalized): when an agent-node
# turn's final answer *looks* fact-bearing (references a price-like token) but none of
# the step's own granted domain tools (its `config.tools`, minus the always-present
# `post_message`) were actually dispatched this execution, warn — loudly, always (not
# gated on `run["trace"]`), the same posture `_note_must_post_violation` already uses.
# Driven entirely off the step's own tool grant set, never a hardcoded tool name, so it
# stays useful once K-053/K-054/K-055 add their own tools to this scaffold.

def test_warns_when_fact_bearing_answer_has_no_domain_tool_dispatched(caplog):
    llm = StubChatLLM([ChatResult(text="It's $29.99.")])
    reg = StubRegistry({"lookup_product_fact": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)
    config = _config(tools=["lookup_product_fact"])

    with caplog.at_level("WARNING", logger="falkorchat.executor"):
        result = ex._run_agent_node(CTX, RUN, STEP, config, {})

    assert any("possible fabrication" in r.getMessage() for r in caplog.records)
    assert any(k == "possible_fabrication" for k, _ in result.trace)


def test_no_warning_when_the_granted_domain_tool_was_actually_dispatched(caplog):
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "lookup_product_fact", {"query": "mouse"})]),
        ChatResult(text="It's $29.99."),
    ])
    reg = StubRegistry({"lookup_product_fact": RETRIEVE_SCHEMA},
                       results={"lookup_product_fact": '{"price": 29.99}'})
    ex = _executor(llm=llm, registry=reg)
    config = _config(tools=["lookup_product_fact"])

    with caplog.at_level("WARNING", logger="falkorchat.executor"):
        result = ex._run_agent_node(CTX, RUN, STEP, config, {})

    assert not any("possible fabrication" in r.getMessage() for r in caplog.records)
    assert not any(k == "possible_fabrication" for k, _ in result.trace)


def test_no_warning_when_the_answer_does_not_look_fact_bearing(caplog):
    llm = StubChatLLM([ChatResult(text="Sure, happy to help with anything else!")])
    reg = StubRegistry({"lookup_product_fact": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)
    config = _config(tools=["lookup_product_fact"])

    with caplog.at_level("WARNING", logger="falkorchat.executor"):
        result = ex._run_agent_node(CTX, RUN, STEP, config, {})

    assert not any("possible fabrication" in r.getMessage() for r in caplog.records)
    assert not any(k == "possible_fabrication" for k, _ in result.trace)


def test_no_warning_when_the_step_grants_no_domain_tools_at_all():
    # A step with nothing but post_message granted (or no tools at all) has nothing it
    # could have dispatched — the signal is meaningless there, not a violation.
    llm = StubChatLLM([ChatResult(text="It's $29.99.")])
    reg = StubRegistry(
        {"post_message": {"type": "function",
                          "function": {"name": "post_message", "parameters": {}}}},
    )
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(tools=["post_message"]), {})

    assert not any(k == "possible_fabrication" for k, _ in result.trace)


def test_fabrication_warning_also_fires_on_max_iterations_exhaustion(caplog):
    # The exhaustion path builds its own StepResult from `last_text` — the check must
    # run there too, not only on the early non-tool-call return.
    llm = StubChatLLM([
        ChatResult(text="It's $29.99.",
                   tool_calls=[ToolCall("c1", "lookup_product_fact", {"query": "x"})]),
    ] * 2)
    reg = StubRegistry({"lookup_product_fact": RETRIEVE_SCHEMA},
                       results={"lookup_product_fact": "not really dispatched cleanly"})

    class _NeverSatisfiesRegistry(StubRegistry):
        def dispatch(self, name, arguments, *, ctx, run):
            self.dispatched.append((name, arguments))
            raise UnknownMemberError(["ghost"])

    reg = _NeverSatisfiesRegistry({"lookup_product_fact": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)
    config = _config(tools=["lookup_product_fact"], maxIterations=2)

    with caplog.at_level("WARNING", logger="falkorchat.executor"):
        result = ex._run_agent_node(CTX, RUN, STEP, config, {})

    assert any("possible fabrication" in r.getMessage() for r in caplog.records)
    assert any(k == "possible_fabrication" for k, _ in result.trace)


# ── K-039 / mention-reply-delivery RCA #1 — implicit post_message fallback ────
#
# A node granted `post_message` that ends its turn via the non-tool-call branch (plain
# prose, or a "recovery" after an earlier call was rejected) must not have that text
# silently discarded — the executor dispatches `post_message` with it as a fallback
# (an implicit call, not a silent no-op).

POST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "post_message",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


def test_plain_text_with_granted_post_message_is_posted_as_implicit_fallback():
    # The RCA's primary failure shape: the model never calls its granted post_message
    # tool at all, just writes prose. Before the fix this text is returned as
    # `StepResult.output` and discarded — no dispatch, no emission.
    llm = StubChatLLM([ChatResult(text="2 + 2 equals 4.")])
    reg = StubRegistry(
        {"post_message": POST_SCHEMA},
        results={"post_message": '{"posted": "m99", "threadId": "t1"}'},
    )
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(tools=["post_message"]), {})

    assert reg.dispatched == [("post_message", {"text": "2 + 2 equals 4."})]
    assert result.emissions == ["m99"]
    assert result.output == "2 + 2 equals 4."


def test_recovery_after_mention_rejection_still_posts_via_implicit_fallback():
    # The RCA's second failure shape: turn 1's post_message call is rejected (a
    # hallucinated `mentions` arg), the model "recovers" on turn 2 by dropping the tool
    # call and just writing plain text — that funnels through the SAME non-tool-call
    # branch, so the same fix must cover it.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "post_message", {"text": "hi", "mentions": ["alice"]})]),
        ChatResult(text="here is your answer, no mention"),
    ])

    class _RecoveringRegistry:
        """`post_message` rejects a call carrying `mentions`, succeeds without it —
        mirroring the real §4 write rejecting a hallucinated displayName mention."""

        def __init__(self):
            self.dispatched: list[tuple[str, dict]] = []

        def schema(self, name):
            return POST_SCHEMA

        def dispatch(self, name, arguments, *, ctx, run):
            self.dispatched.append((name, arguments))
            if "mentions" in arguments:
                raise UnknownMemberError(["alice"])
            return '{"posted": "m7", "threadId": "t1"}'

    reg = _RecoveringRegistry()
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(tools=["post_message"]), {})

    assert reg.dispatched == [
        ("post_message", {"text": "hi", "mentions": ["alice"]}),
        ("post_message", {"text": "here is your answer, no mention"}),
    ]
    assert result.emissions == ["m7"]


def test_no_implicit_post_when_post_message_not_granted():
    # A node that was never granted post_message must never have one dispatched on its
    # behalf — the fallback is scoped strictly to nodes that already hold the tool.
    llm = StubChatLLM([ChatResult(text="grounded answer")])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {})

    assert reg.dispatched == []
    assert result.emissions == []
    assert result.output == "grounded answer"


def test_no_implicit_post_when_final_text_is_empty():
    # An empty final text has nothing to post — the fallback must not dispatch a blank
    # message.
    llm = StubChatLLM([ChatResult(text="")])
    reg = StubRegistry({"post_message": POST_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(tools=["post_message"]), {})

    assert reg.dispatched == []
    assert result.emissions == []


# ── must-post engine contract (K-027 item 2, `docs/plans/must-post-engine-contract.md`) ──
#
# A `type:'agent'` node may declare `config.requiredTools`, a subset of `config.tools`
# that must be successfully dispatched at least once before the node ends its turn. A
# violation never fails or parks the node — it is always logged (`_log.warning`,
# unconditional, not tracer-gated) and, on a debug run, appended to `trace` as a
# `must_post_violation` entry. This generalizes past the K-039 fallback above: it covers
# any declared tool name (not just `post_message`), and it also covers the
# `maxIterations`-exhaustion return path, which K-039 never touches.

NOTIFY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "notify_owner",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
}


def test_compliant_node_dispatching_required_tool_leaves_no_violation_trace():
    # The common/compliant path: the model calls its required tool directly (the
    # tool-call branch, not the K-039 fallback). Zero behavior change from today, and no
    # must_post_violation trace entry.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[ToolCall("c1", "post_message", {"text": "hi"})]),
        ChatResult(text="done"),
    ])
    reg = StubRegistry(
        {"post_message": POST_SCHEMA},
        results={"post_message": '{"posted": "m1", "threadId": "t1"}'},
    )
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(
        CTX, RUN, STEP,
        _config(tools=["post_message"], requiredTools=["post_message"]), {},
    )

    assert result.output == "done"
    assert not any(k == "must_post_violation" for k, _ in result.trace)


def test_compliant_node_dispatching_a_non_post_message_required_tool_leaves_no_violation_trace():
    # Supplementary to the plan's 8 specified tests (found via this unit's own
    # mutation-testing pass): the compliant-path test above only exercises
    # `post_message`, whose satisfaction is read off `emissions`, never `satisfied` — so
    # it cannot catch a regression in the `satisfied` threading at the MAIN dispatch-loop
    # `_handle_tool_call` call site specifically (as opposed to the K-039 implicit-call
    # site). This pins that a non-post_message required tool, dispatched directly via a
    # real tool call, satisfies the contract with no violation.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "notify_owner", {"reason": "escalate"})]),
        ChatResult(text="done, owner notified"),
    ])
    reg = StubRegistry({"notify_owner": NOTIFY_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(
        CTX, RUN, STEP,
        _config(tools=["notify_owner"], requiredTools=["notify_owner"]), {},
    )

    assert result.output == "done, owner notified"
    assert not any(k == "must_post_violation" for k, _ in result.trace)


def test_plain_text_ending_with_required_post_message_recovers_via_k039_no_violation_logged():
    # Mirrors test_plain_text_with_granted_post_message_is_posted_as_implicit_fallback
    # above, but with requiredTools=["post_message"] declared: the K-039 implicit
    # dispatch still fires exactly as before, and its success satisfies the contract via
    # emissions — no violation.
    llm = StubChatLLM([ChatResult(text="2 + 2 equals 4.")])
    reg = StubRegistry(
        {"post_message": POST_SCHEMA},
        results={"post_message": '{"posted": "m99", "threadId": "t1"}'},
    )
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(
        CTX, RUN, STEP,
        _config(tools=["post_message"], requiredTools=["post_message"]), {},
    )

    assert reg.dispatched == [("post_message", {"text": "2 + 2 equals 4."})]
    assert result.emissions == ["m99"]
    assert not any(k == "must_post_violation" for k, _ in result.trace)


def test_required_non_post_message_tool_never_dispatched_logs_and_traces_a_visible_violation(
    caplog,
):
    # The core new-behavior test (RCA §5 item 3's ask): a required tool with a shape
    # K-039 cannot help with (not post_message), never called. The run is not failed
    # (on == "done"), a must_post_violation names the missing tool, and the warning is
    # logged even with no tracer configured — visibility does not depend on debug-run
    # status.
    llm = StubChatLLM([ChatResult(text="all done, nothing to escalate")])
    reg = StubRegistry({"notify_owner": NOTIFY_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    with caplog.at_level("WARNING", logger="falkorchat.executor"):
        result = ex._run_agent_node(
            CTX, RUN, STEP,
            _config(tools=["notify_owner"], requiredTools=["notify_owner"]), {},
        )

    assert result.on == "done"
    assert any(
        k == "must_post_violation" and "notify_owner" in p for k, p in result.trace
    )
    assert any("notify_owner" in r.getMessage() for r in caplog.records)


def test_required_post_message_whose_own_implicit_dispatch_declines_still_logs_a_violation():
    # The concrete gap found while reading tools.py: a registry double whose
    # post_message dispatch returns an error string WITHOUT raising (mirroring
    # PostMessageTool.run's "no thread bound" path) — the K-039 fallback "succeeds" at
    # the dispatch layer but produces no "posted" envelope. Satisfaction is read off
    # emissions, not off "the dispatch didn't raise".
    llm = StubChatLLM([ChatResult(text="here is your answer")])
    reg = StubRegistry(
        {"post_message": POST_SCHEMA},
        results={"post_message": "error: no thread is bound to this run; cannot post a message"},
    )
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(
        CTX, RUN, STEP,
        _config(tools=["post_message"], requiredTools=["post_message"]), {},
    )

    assert result.emissions == []
    assert any(
        k == "must_post_violation" and "post_message" in p for k, p in result.trace
    )


def test_required_tool_never_dispatched_across_max_iterations_logs_and_traces_a_violation():
    # The maxIterations-exhaustion exit point, which K-039 never covered at all: the
    # model only ever calls a DIFFERENT granted tool every turn, never the required one,
    # until the node's iteration budget exhausts. Existing exhaustion behavior is
    # unchanged (on == "done", best-current-text output, the node_note exhaustion trace
    # entry) and a must_post_violation is also present.
    call = ToolCall("c1", "graphrag_retrieve", {"query": "loop"})
    llm = AlwaysToolLLM(call)
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA, "notify_owner": NOTIFY_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(
        CTX, RUN, STEP,
        _config(tools=["graphrag_retrieve", "notify_owner"],
                requiredTools=["notify_owner"], maxIterations=2), {},
    )

    assert result.on == "done"
    assert result.output == "thinking"
    assert len(reg.dispatched) == 2
    assert any(k == "node_note" for k, _ in result.trace)
    assert any(
        k == "must_post_violation" and "notify_owner" in p for k, p in result.trace
    )


def test_undeclared_required_tools_is_fully_backward_compatible():
    # No requiredTools declared (every currently-shipped def and existing test fixture)
    # — `required` is always the empty set, so `_missing_required_tools` always returns
    # empty. Re-run a representative slice of the existing K-039/plain-tool-loop
    # scenarios and pin zero new trace entries — the explicit compatibility pin for
    # §7's claim, not just an implicit consequence of not touching those tests' own
    # assertions.
    llm = StubChatLLM([ChatResult(text="grounded answer")])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)
    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {})
    assert not any(k == "must_post_violation" for k, _ in result.trace)

    loop_call = ToolCall("c1", "graphrag_retrieve", {"query": "loop"})
    llm2 = AlwaysToolLLM(loop_call)
    reg2 = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex2 = _executor(llm=llm2, registry=reg2)
    result2 = ex2._run_agent_node(CTX, RUN, STEP, _config(maxIterations=2), {})
    assert not any(k == "must_post_violation" for k, _ in result2.trace)

    llm3 = StubChatLLM([ChatResult(text="2 + 2 equals 4.")])
    reg3 = StubRegistry(
        {"post_message": POST_SCHEMA},
        results={"post_message": '{"posted": "m99", "threadId": "t1"}'},
    )
    ex3 = _executor(llm=llm3, registry=reg3)
    result3 = ex3._run_agent_node(CTX, RUN, STEP, _config(tools=["post_message"]), {})
    assert not any(k == "must_post_violation" for k, _ in result3.trace)


def test_required_tool_absent_from_granted_tools_is_silently_dropped_at_drive_time():
    # Review finding M2.1: a node whose config.requiredTools names a tool that is NOT in
    # config.tools (the hand-crafted-graph-write shape publish-time validation is meant
    # to catch — this test bypasses it deliberately, exercising `_run_agent_node`
    # directly). The `& granted_set` intersection silently drops it: no exception, no
    # must_post_violation, and the node's output/on are unaffected.
    llm = StubChatLLM([ChatResult(text="grounded answer")])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(
        CTX, RUN, STEP, _config(requiredTools=["notify_owner"]), {},
    )

    assert not any(k == "must_post_violation" for k, _ in result.trace)
    assert result.output == "grounded answer"
    assert result.on == "done"


def test_required_post_message_node_ending_on_empty_text_still_logs_a_violation():
    # Review finding M2.2: a node with requiredTools=["post_message"] whose model ends
    # its turn via the non-tool-call branch with EMPTY result.text. K-039's implicit
    # dispatch is gated on result.text being truthy, so it never even attempts here —
    # this contract's check is the SOLE defense on this path, not a redundant
    # restatement of an existing K-039 assertion.
    llm = StubChatLLM([ChatResult(text="")])
    reg = StubRegistry({"post_message": POST_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(
        CTX, RUN, STEP,
        _config(tools=["post_message"], requiredTools=["post_message"]), {},
    )

    assert reg.dispatched == []
    assert result.emissions == []
    assert any(
        k == "must_post_violation" and "post_message" in p for k, p in result.trace
    )


# ── K-042 code review Major 2: the `step`-kind gateway resolution wiring ──────────
#
# Every test above injects `llm=<stub>` — the pre-K-042 `StaticModelGateway` sugar
# path, which proves backward compatibility but never exercises `_run_agent_node`'s
# own `self._models.resolve_llm("step", requested=config.get("model"), ws=ctx.ws,
# overrides=run.get("modelOverrides"))` call. These use a small recording `models=`
# double instead, so a regression at that one call site (a swapped kind string, a
# dropped `ws=`/`overrides=`, a `requested=` source that silently stops reading
# `config.get("model")`) would fail here even though it is invisible to every other
# test in this file.


class _FakeResolution:
    """Just enough of `modelconfig.Resolution` for `_run_agent_node` to read
    `.source` off — no real chain/provider machinery needed for this double."""

    def __init__(self, *, source: str) -> None:
        self.source = source


class RecordingGateway:
    """A minimal `ModelGateway`-shaped double: records every `.resolve_llm()` call's
    `(kind, requested, ws, overrides)` and returns a distinct `StubChatLLM` per
    resolved ref (so two differently-modeled steps are provably answered by two
    different clients — the executor-level half of AC-4/§10 item 8, without any real
    `ModelGateway`/config file/network).

    K-042 L2-3: mirrors the real gateway's three-rung precedence just enough to drive
    `_run_agent_node`'s `_MODEL_SOURCE_LABEL` mapping — a workspace override (read off
    `overrides["agentModel"]`, the `step`-kind crosswalk key, `-graph.md` §8.4) beats
    `requested`, which beats the (fixed) default ref.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self._clients: dict[str | None, StubChatLLM] = {}

    def has_chat(self) -> bool:
        return True

    def resolve_llm(self, kind, *, requested=None, ws=None, overrides=None):
        self.calls.append(
            {"kind": kind, "requested": requested, "ws": ws, "overrides": overrides}
        )
        override = (overrides or {}).get("agentModel") if kind == "step" else None
        if override:
            ref, source = override, "workspace"
        elif requested is not None:
            ref, source = requested, "requested"
        else:
            ref, source = requested, "default"
        if ref not in self._clients:
            self._clients[ref] = StubChatLLM([ChatResult(text=f"answer from {ref}")])
        return self._clients[ref], _FakeResolution(source=source)

    def embedder(self, kind, *, requested=None, ws=None, overrides=None):
        raise AssertionError("_run_agent_node must never resolve an embedder")


def test_run_agent_node_resolves_step_kind_with_the_steps_own_model_and_ws():
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    gateway = RecordingGateway()
    ex = WorkflowExecutor(None, None, models=gateway, tool_registry=reg, guard_judge=None)

    ex._run_agent_node(
        CTX, RUN, STEP, _config(model="lmstudio/qwen3-4b"), {"threadId": "t1"}
    )

    assert gateway.calls == [
        {
            "kind": "step", "requested": "lmstudio/qwen3-4b", "ws": CTX.ws,
            "overrides": None,
        }
    ]


def test_run_agent_node_requests_no_model_when_the_step_names_none():
    # `config.get("model")` is absent — the gateway must be asked with
    # `requested=None`, letting it fall back to the kind default rather than the
    # executor inventing a ref.
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    gateway = RecordingGateway()
    ex = WorkflowExecutor(None, None, models=gateway, tool_registry=reg, guard_judge=None)

    ex._run_agent_node(CTX, RUN, STEP, _config(), {"threadId": "t1"})

    assert gateway.calls == [
        {"kind": "step", "requested": None, "ws": CTX.ws, "overrides": None}
    ]


def test_two_steps_naming_different_models_resolve_to_different_clients_through_the_executor():
    # §10 item 8 / AC-4 (Landing-1 half), driven through the executor rather than
    # `ModelGateway` in isolation (`test_modelconfig.py` covers the gateway alone).
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    gateway = RecordingGateway()
    ex = WorkflowExecutor(None, None, models=gateway, tool_registry=reg, guard_judge=None)

    result_a = ex._run_agent_node(
        CTX, RUN, {"key": "research", "type": "agent"},
        _config(model="lmstudio/model-a"), {"threadId": "t1"},
    )
    result_b = ex._run_agent_node(
        CTX, RUN, {"key": "answer", "type": "agent"},
        _config(model="lmstudio/model-b"), {"threadId": "t1"},
    )

    assert [c["requested"] for c in gateway.calls] == [
        "lmstudio/model-a", "lmstudio/model-b",
    ]
    assert result_a.output == "answer from lmstudio/model-a"
    assert result_b.output == "answer from lmstudio/model-b"
    assert result_a.output != result_b.output


def test_run_agent_node_threads_run_model_overrides_to_the_gateway():
    # L2-3's other half of the same call site: `run["modelOverrides"]` (stamped by
    # `_drive`, §4.10/§2.6) must reach `resolve_llm` as `overrides=`, not just `ws=`.
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    gateway = RecordingGateway()
    ex = WorkflowExecutor(None, None, models=gateway, tool_registry=reg, guard_judge=None)
    run = {**RUN, "modelOverrides": {"agentModel": None, "guardModel": None,
                                      "embeddingModel": None, "responderModel": None}}

    ex._run_agent_node(CTX, run, STEP, _config(model="lmstudio/qwen3-4b"), {"threadId": "t1"})

    assert gateway.calls[0]["overrides"] == {
        "agentModel": None, "guardModel": None, "embeddingModel": None,
        "responderModel": None,
    }


# ── K-042 Landing 2 (L2-2, FR-8): resolvedModel/modelSource/modelFallback ─────────
#
# `_run_agent_node` must carry the answering model, which precedence rung named it,
# and whether it came from a fallback, out on `StepResult` — for `_record` to forward
# to `repo.record_step_and_advance(...)` (§8.1/§1.5 `-graph.md`: the value must ride
# `StepResult`, the only carrier the SHA-locked `_drive_loop` call site passes to
# `_record`). This unit's reachable outcomes are only `modelSource ∈ {'step',
# 'default'}` — `'workspace'` is L2-3, not yet built.

def test_step_naming_its_own_model_records_step_as_the_source():
    llm = StubChatLLM([ChatResult(text="answer", model="lmstudio/qwen3")])
    ex = _executor(llm=llm, registry=StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA}))

    result = ex._run_agent_node(
        CTX, RUN, STEP, _config(model="lmstudio/qwen3"), {"threadId": "t1"}
    )

    assert result.resolvedModel == "lmstudio/qwen3"
    assert result.modelSource == "step"
    assert result.modelFallback is None


def test_step_naming_no_model_records_default_as_the_source():
    llm = StubChatLLM([ChatResult(text="answer", model="lmstudio/kind-default")])
    ex = _executor(llm=llm, registry=StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA}))

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {"threadId": "t1"})

    assert result.resolvedModel == "lmstudio/kind-default"
    assert result.modelSource == "default"
    assert result.modelFallback is None


def test_fallback_flag_carried_from_chat_result_onto_step_result():
    llm = StubChatLLM([ChatResult(text="answer", model="lmstudio/b", fallback=True)])
    ex = _executor(llm=llm, registry=StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA}))

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {"threadId": "t1"})

    assert result.resolvedModel == "lmstudio/b"
    assert result.modelFallback is True


def test_offline_stub_with_no_model_on_chat_result_leaves_all_three_fields_none():
    # The pre-K-042 `llm=<stub>` pattern used by every other test in this file never
    # sets `ChatResult.model` — a non-answer, not an unset default rung. Confirms the
    # three new fields are additive/default-safe against every pre-existing test.
    llm = StubChatLLM([ChatResult(text="here is the answer")])
    ex = _executor(llm=llm, registry=StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA}))

    result = ex._run_agent_node(CTX, RUN, STEP, _config(), {"threadId": "t1"})

    assert result.resolvedModel is None
    assert result.modelSource is None
    assert result.modelFallback is None


def test_max_iterations_exhaustion_still_carries_the_last_resolved_model():
    # AlwaysToolLLM (never emits a final text, drives exhaustion) never sets
    # ChatResult.model — a small local variant does, so the exhaustion path (the
    # OTHER StepResult return statement) is covered too.
    class _ModelledAlwaysToolLLM:
        def __init__(self, call):
            self._call = call

        def chat(self, messages, tools):
            return ChatResult(
                text="thinking", tool_calls=[self._call], model="lmstudio/looping",
            )

    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    call = ToolCall("c1", "graphrag_retrieve", {"query": "x"})
    ex = _executor(llm=_ModelledAlwaysToolLLM(call), registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(maxIterations=2), {"threadId": "t1"})

    assert result.resolvedModel == "lmstudio/looping"
    assert result.modelSource == "default"


def test_last_answering_model_wins_across_iterations():
    # -graph.md §1.6 "last answering model wins" (m-6): a node that calls the LLM more
    # than once must report the LAST successful call's resolvedModel/modelSource/
    # modelFallback, not the first. Iteration 1 answers on model A with no fallback
    # (a tool call, so the loop continues); iteration 2 answers on model B WITH a
    # fallback flag, and ends the node on a final text.
    llm = StubChatLLM([
        ChatResult(
            text="thinking", tool_calls=[ToolCall("c1", "graphrag_retrieve", {"query": "x"})],
            model="lmstudio/model-a", fallback=None,
        ),
        ChatResult(text="final answer", model="lmstudio/model-b", fallback=True),
    ])
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(maxIterations=4), {"threadId": "t1"})

    assert result.output == "final answer"
    assert result.resolvedModel == "lmstudio/model-b"
    assert result.modelFallback is True


# ── K-042 Landing 2 (L2-3, FR-16/FR-17): the workspace override hard cap ──────────
#
# These drive `_run_agent_node` through a REAL `ModelGateway` (not the recording/
# static doubles above) because the precedence itself — and specifically the
# resolver-sourced `modelSource` reshape — lives in `ModelGateway.resolve()`, not in
# the executor. `RecordingGateway`'s tests above only pin the *call site* (kind/
# requested/ws/overrides reach the gateway); these pin the *behavior* the gate
# required as an explicit done-condition for this unit.

def _real_gateway(*, providers, overlay_doc):
    from falkorchat.modelconfig import ModelGateway, Overlay, ProviderCatalog

    catalog = ProviderCatalog(providers, path="opencode.json")
    overlay = Overlay(overlay_doc, path="models.json")
    return ModelGateway(catalog, overlay)


def test_workspace_override_beats_the_steps_own_explicit_choice(monkeypatch):
    # AC-10, driven at the executor level: the step names its own model, the
    # workspace overrides kind `step` to a different one — the workspace's model
    # runs (never even calling the step's declared one) and `modelSource` reports
    # `'workspace'`, not `'step'`.
    gateway = _real_gateway(
        providers={"lmstudio": {"options": {"baseURL": "http://host:1/v1"}}},
        overlay_doc={"defaults": {"step": "lmstudio/kind-default"}},
    )
    calls: list[str] = []

    def fake_make_http_transport(*, timeout, headers=None, opener=None, provider="?", model="?"):
        def _transport(url, payload):
            calls.append(payload.get("model"))
            return {"choices": [{"message": {"content": "answer"}}]}
        return _transport

    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", fake_make_http_transport
    )
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = WorkflowExecutor(None, None, models=gateway, tool_registry=reg, guard_judge=None)
    # `agentModel` is the workspace-override crosswalk key for kind `step`
    # (`-graph.md` §8.4 — `modelconfig._KIND_TO_OVERRIDE_KEY`), stamped by `_drive`
    # in production; supplied directly here as `_run_agent_node` is unit-tested
    # below `_drive`.
    run = {**RUN, "modelOverrides": {"agentModel": "lmstudio/workspace-model"}}

    result = ex._run_agent_node(
        CTX, run, STEP, _config(model="lmstudio/step-declared-model"), {"threadId": "t1"},
    )

    assert calls == ["workspace-model"]  # the step's declared model was never called
    assert result.resolvedModel == "lmstudio/workspace-model"
    assert result.modelSource == "workspace"


def test_workspace_override_naming_a_role_that_falls_back_reports_workspace_and_fallback_true(
    monkeypatch,
):
    # The exact bug class the U8 gate's "naive fix" warning named: a workspace
    # override targets a ROLE, and that role's first element fails and falls back to
    # its second. A local `config.get('model')`-truthiness mirror (or a bolted-on
    # `if run.get("modelOverrides",{}).get("step"): "workspace"` patch) cannot see
    # this interaction — only the resolver, which is already walking the fallback
    # chain, knows both facts at once. `modelSource == 'workspace'` AND
    # `modelFallback is True` must hold on the SAME row.
    from falkorchat.transport import ProviderCallError

    gateway = _real_gateway(
        providers={
            "lmstudio": {"options": {"baseURL": "http://host:1/v1"}},
            "second": {"options": {"baseURL": "http://host:2/v1"}},
        },
        overlay_doc={
            "defaults": {"step": "lmstudio/never-used"},
            "roles": {"cheap": {"models": ["lmstudio/model-a", "second/model-b"]}},
        },
    )

    def fake_make_http_transport(*, timeout, headers=None, opener=None, provider="?", model="?"):
        def _transport(url, payload):
            if provider == "lmstudio":
                raise ProviderCallError(f"{provider}/{model}: simulated outage")
            return {"choices": [{"message": {"content": "from second"}}]}
        return _transport

    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", fake_make_http_transport
    )
    reg = StubRegistry({"graphrag_retrieve": RETRIEVE_SCHEMA})
    ex = WorkflowExecutor(None, None, models=gateway, tool_registry=reg, guard_judge=None)
    run = {**RUN, "modelOverrides": {"agentModel": "cheap"}}  # a ROLE, not a direct ref

    result = ex._run_agent_node(
        CTX, run, STEP, _config(model="lmstudio/step-declared-model"), {"threadId": "t1"},
    )

    assert result.modelSource == "workspace"
    assert result.modelFallback is True
    assert result.resolvedModel == "second/model-b"


# ── AC-9 (workflow-cart-and-totals.md §3.1/§5, K-053 M6) ───────────────────────
#
# "No LLM/model call is made solely to perform" a cart computation. §3.1's own
# verification method: on a debug run, dispatching a cart-mutating tool costs
# exactly its own turn's routing llm_prompt/llm_response pair — no *extra*
# llm_prompt is squeezed in between the tool_call and its tool_result, because
# the arithmetic (pricing.compute_line_total, reached through services.
# add_cart_item) never touches self._models. Uses the REAL `AddToCartTool`
# (not a stub `Tool`), so this proves the property for the actual production
# dispatch path — a stub tool could never demonstrate an extra LLM call either
# way, since Tool.run()'s call signature never carries an LLM client at all.


class FakeCartServices:
    """The one method `AddToCartTool.run` calls — no LLM client anywhere in
    reach, mirroring the real `Services.add_cart_item`'s LLM-free contract."""

    def __init__(self):
        self.calls: list[dict] = []

    def add_cart_item(self, ctx, *, product_name, quantity):
        self.calls.append({"product_name": product_name, "quantity": quantity})
        return {
            "productId": "p1", "name": product_name, "price": 9.99,
            "quantity": quantity,
        }


def test_cart_tool_dispatch_emits_no_extra_llm_call_between_tool_call_and_result():
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "add_to_cart", {"productName": "Widget", "quantity": 2})]),
        ChatResult(text="Added 2 Widgets to your cart."),
    ])
    services = FakeCartServices()
    reg = ToolRegistry([AddToCartTool(services)])
    ex = _executor(llm=llm, registry=reg)

    result = ex._run_agent_node(CTX, RUN, STEP, _config(tools=["add_to_cart"]), {})

    assert result.output == "Added 2 Widgets to your cart."
    assert services.calls == [{"product_name": "Widget", "quantity": 2}]

    kinds = [k for k, _ in result.trace]
    # exactly the two routing turns' own llm_prompt/llm_response pairs (one
    # iteration dispatches the tool, the next produces the final text) —
    # never more than one pair per iteration.
    assert kinds.count("llm_prompt") == 2
    assert kinds.count("llm_response") == 2
    tool_call_i = kinds.index("tool_call")
    tool_result_i = kinds.index("tool_result")
    between = kinds[tool_call_i + 1:tool_result_i]
    assert "llm_prompt" not in between
    assert "llm_response" not in between


# ── K-058 (`docs/reviews/salesperson-tool-reliability-ml.md` §9.2/§9.4) ────────
#
# `mistralai/ministral-3-3b` sometimes re-issues an earlier, already-completed
# write-mutating tool call (e.g. `add_to_cart`) within a *later* turn's own
# multi-iteration tool loop, even though that turn's own text never mentions the
# re-issued call's target. The dispatch-time guard: immediately before executing
# a write-mutating call whose schema names a resolved single-string target
# argument (`add_to_cart`/`remove_from_cart`'s `productName`), check whether that
# argument's value appears — case/whitespace-insensitive, `extraction.
# normalize_name`, the same shared helper `nameNormalized`/`categoryNormalized`
# are built with — in the *current turn's own* raw text (its triggering user
# message plus whatever the model has already said this same turn). If not,
# hold the call (never dispatch) and surface it via a dedicated observability
# signal, `_note_off_turn_write_held` — deliberately distinct from `_note_
# possible_fabrication`, since that signal targets a different failure class
# (a fact-bearing reply with no domain tool dispatched at all) and must keep
# meaning "no grounding tool ran" rather than being overloaded for "a tool ran
# but its target was ungrounded in this turn's text".

ADD_TO_CART_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_to_cart",
        "description": "Add a product to the cart.",
        "parameters": {
            "type": "object",
            "properties": {
                "productName": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["productName"],
        },
    },
}

REMOVE_FROM_CART_SCHEMA = {
    "type": "function",
    "function": {
        "name": "remove_from_cart",
        "description": "Remove a product from the cart.",
        "parameters": {
            "type": "object",
            "properties": {
                "productName": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["productName"],
        },
    },
}


def test_off_turn_write_is_held_not_dispatched_on_confirmed_repro_shape(caplog):
    # ml note §9.2's exact confirmed reproduction: turn 2 ("Also add 1 Mechanical
    # Keyboard K200") correctly adds the keyboard on its first tool-calling
    # iteration, then spontaneously re-issues add_to_cart for the mouse — a
    # product turn 2's own text never mentions — on its very next iteration.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "add_to_cart",
                     {"productName": "Mechanical Keyboard K200", "quantity": 1})]),
        ChatResult(text="", tool_calls=[
            ToolCall("c2", "add_to_cart",
                     {"productName": "Wireless Mouse Pro", "quantity": 1})]),
        ChatResult(text="Your cart now includes: Wireless Mouse Pro x2, "
                        "Mechanical Keyboard K200 x1. Total: $149.97"),
    ])
    reg = StubRegistry(
        {"add_to_cart": ADD_TO_CART_SCHEMA},
        results={"add_to_cart": '{"found": true}'},
    )
    services = StubThreadServices(thread_msgs=[
        {"role": "assistant", "text": "Added 1 Wireless Mouse Pro to your cart.",
         "authorId": "assistant", "displayName": "Bot"},
        {"role": "user", "text": "Also add 1 Mechanical Keyboard K200",
         "authorId": "u1", "displayName": "Alice"},
    ])
    ex = _executor(llm=llm, registry=reg, services=services)
    config = _config(tools=["add_to_cart"])

    with caplog.at_level("WARNING", logger="falkorchat.executor"):
        result = ex._run_agent_node(CTX, RUN, STEP, config, {"threadId": "t1"})

    # only the legitimately-requested keyboard reaches dispatch — the mouse
    # (mentioned nowhere in turn 2's own text) is held, not dispatched
    assert reg.dispatched == [
        ("add_to_cart", {"productName": "Mechanical Keyboard K200", "quantity": 1}),
    ]
    assert any(k == "off_turn_write_held" for k, _ in result.trace)
    assert any("off-turn write held" in r.getMessage() for r in caplog.records)

    # the held call is fed back to the model as a clean rejection, not a crash —
    # the same "never crash, never fabricate" posture as QueryGraphDataTool's
    # abstention shape, just for this different failure class
    tool_msgs = [m for m in llm.calls[-1]["messages"] if m["role"] == "tool"]
    held_msg = next(m for m in tool_msgs if m["tool_call_id"] == "c2")
    assert "held" in held_msg["content"].lower()
    assert "Wireless Mouse Pro" not in "".join(
        f"{k}{v}" for k, v in reg.dispatched
    )


def test_legitimate_repeat_mentioned_in_turn_text_still_dispatches():
    # Explicitly NOT the ruled-out fix (§9.4): a genuine later repeat of the same
    # product, when THIS turn's own text names it, must still dispatch normally —
    # "add another mouse" three turns later is a real, intended repeat.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "add_to_cart",
                     {"productName": "Wireless Mouse Pro", "quantity": 1})]),
        ChatResult(text="Added another Wireless Mouse Pro."),
    ])
    reg = StubRegistry(
        {"add_to_cart": ADD_TO_CART_SCHEMA},
        results={"add_to_cart": '{"found": true}'},
    )
    services = StubThreadServices(thread_msgs=[
        {"role": "user", "text": "Add another Wireless Mouse Pro please",
         "authorId": "u1", "displayName": "Alice"},
    ])
    ex = _executor(llm=llm, registry=reg, services=services)
    config = _config(tools=["add_to_cart"])

    result = ex._run_agent_node(CTX, RUN, STEP, config, {"threadId": "t1"})

    assert reg.dispatched == [
        ("add_to_cart", {"productName": "Wireless Mouse Pro", "quantity": 1}),
    ]
    assert not any(k == "off_turn_write_held" for k, _ in result.trace)
    assert result.output == "Added another Wireless Mouse Pro."


def test_target_mention_matching_is_case_and_whitespace_insensitive():
    # Mirrors the nameNormalized/categoryNormalized precedent (extraction.
    # normalize_name): casefold + whitespace-collapse, not a byte-exact match.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "add_to_cart",
                     {"productName": "Wireless Mouse Pro", "quantity": 1})]),
        ChatResult(text="Added."),
    ])
    reg = StubRegistry(
        {"add_to_cart": ADD_TO_CART_SCHEMA},
        results={"add_to_cart": '{"found": true}'},
    )
    services = StubThreadServices(thread_msgs=[
        {"role": "user", "text": "please add 1   wireless MOUSE   pro",
         "authorId": "u1", "displayName": "Alice"},
    ])
    ex = _executor(llm=llm, registry=reg, services=services)
    config = _config(tools=["add_to_cart"])

    result = ex._run_agent_node(CTX, RUN, STEP, config, {"threadId": "t1"})

    assert reg.dispatched == [
        ("add_to_cart", {"productName": "Wireless Mouse Pro", "quantity": 1}),
    ]
    assert not any(k == "off_turn_write_held" for k, _ in result.trace)


def test_off_turn_remove_from_cart_is_also_held():
    # The same guard applies to remove_from_cart (both take a resolved
    # productName target) — ml note §9's remove-retrigger condition names the
    # same mechanism for removal, even though it wasn't observed at n=4.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "remove_from_cart",
                     {"productName": "Mechanical Keyboard K200"})]),
        ChatResult(text="Removed."),
    ])
    reg = StubRegistry(
        {"remove_from_cart": REMOVE_FROM_CART_SCHEMA},
        results={"remove_from_cart": '{"found": true, "removed": true}'},
    )
    services = StubThreadServices(thread_msgs=[
        {"role": "user", "text": "Remove the Wireless Mouse Pro",
         "authorId": "u1", "displayName": "Alice"},
    ])
    ex = _executor(llm=llm, registry=reg, services=services)
    config = _config(tools=["remove_from_cart"])

    result = ex._run_agent_node(CTX, RUN, STEP, config, {"threadId": "t1"})

    assert reg.dispatched == []
    assert any(k == "off_turn_write_held" for k, _ in result.trace)


def test_write_tool_with_no_resolved_target_argument_is_unaffected():
    # place_order/clear_cart carry no product-name-shaped target argument, so
    # the guard has nothing to check and never holds them regardless of turn text.
    schema = {
        "type": "function",
        "function": {"name": "place_order", "parameters": {
            "type": "object", "properties": {}, "required": [],
        }},
    }
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[ToolCall("c1", "place_order", {})]),
        ChatResult(text="Order placed."),
    ])
    reg = StubRegistry({"place_order": schema},
                       results={"place_order": '{"orderId": "o1"}'})
    services = StubThreadServices(thread_msgs=[
        {"role": "user", "text": "that's everything, thanks",
         "authorId": "u1", "displayName": "Alice"},
    ])
    ex = _executor(llm=llm, registry=reg, services=services)
    config = _config(tools=["place_order"])

    result = ex._run_agent_node(CTX, RUN, STEP, config, {"threadId": "t1"})

    assert reg.dispatched == [("place_order", {})]
    assert not any(k == "off_turn_write_held" for k, _ in result.trace)


def test_no_thread_context_available_does_not_hold_write_calls():
    # Backward compatibility: when no thread text is obtainable at all (the
    # offline-stub path several pre-existing tests use — no `services` wired,
    # or no `threadId` in run_ctx), the guard has no information to act on and
    # must not block — failing open on missing data, not closed. Mirrors
    # `test_cart_tool_dispatch_emits_no_extra_llm_call_between_tool_call_and_
    # result` above, which relies on exactly this default.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "add_to_cart",
                     {"productName": "Widget", "quantity": 2})]),
        ChatResult(text="Added."),
    ])
    reg = StubRegistry({"add_to_cart": ADD_TO_CART_SCHEMA},
                       results={"add_to_cart": '{"found": true}'})
    ex = _executor(llm=llm, registry=reg)  # no services — no thread context

    result = ex._run_agent_node(CTX, RUN, STEP, _config(tools=["add_to_cart"]), {})

    assert reg.dispatched == [
        ("add_to_cart", {"productName": "Widget", "quantity": 2}),
    ]
    assert not any(k == "off_turn_write_held" for k, _ in result.trace)


# ── K-061 (`docs/reviews/salesperson-tool-reliability-ml.md` §12) ─────────────
#
# Distinct from K-058 above by design, not an oversight: K-058 holds an *off-turn*
# re-fire of a *previous* turn's already-completed write, checked by asking
# whether the target is mentioned anywhere in *this* turn's own text. This is a
# *same-turn* re-fire of the model's *own current-turn* successful write, on a
# target genuinely mentioned in this turn's text — exactly the case K-058's
# check does not (and structurally cannot) hold. §12.3's confirmed repro shape:
# the model's own multi-iteration tool loop dispatches `add_to_cart` for a
# product once, successfully, then re-issues the identical call again later in
# that same loop, doubling the cart quantity the customer never asked to double.

def test_same_turn_exact_repeat_of_own_successful_write_is_held(caplog):
    # ml.md §12.3's confirmed shape: turn 2's own loop dispatches
    # add_to_cart(Mechanical Keyboard K200, qty=1) successfully on iteration 1,
    # then re-issues the IDENTICAL call again on iteration 2. The target is
    # genuinely mentioned in this turn's own text, so K-058's guard would not
    # (and should not) hold it — this is a different mechanism.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "add_to_cart",
                     {"productName": "Mechanical Keyboard K200", "quantity": 1})]),
        ChatResult(text="", tool_calls=[
            ToolCall("c2", "add_to_cart",
                     {"productName": "Mechanical Keyboard K200", "quantity": 1})]),
        ChatResult(text="Mechanical Keyboard K200 x2 added. Total: $179.98"),
    ])
    reg = StubRegistry(
        {"add_to_cart": ADD_TO_CART_SCHEMA},
        results={"add_to_cart": '{"found": true}'},
    )
    services = StubThreadServices(thread_msgs=[
        {"role": "user", "text": "Also add 1 Mechanical Keyboard K200",
         "authorId": "u1", "displayName": "Alice"},
    ])
    ex = _executor(llm=llm, registry=reg, services=services)
    config = _config(tools=["add_to_cart"])

    with caplog.at_level("WARNING", logger="falkorchat.executor"):
        result = ex._run_agent_node(CTX, RUN, STEP, config, {"threadId": "t1"})

    # only the first dispatch reaches the tool — the identical repeat is held,
    # not dispatched, so the cart quantity is never doubled
    assert reg.dispatched == [
        ("add_to_cart", {"productName": "Mechanical Keyboard K200", "quantity": 1}),
    ]
    assert any(k == "same_turn_write_held" for k, _ in result.trace)
    assert not any(k == "off_turn_write_held" for k, _ in result.trace)
    assert any("same-turn write held" in r.getMessage() for r in caplog.records)

    # the held repeat is fed back to the model as a clean rejection, not a
    # crash and not a second dispatch
    tool_msgs = [m for m in llm.calls[-1]["messages"] if m["role"] == "tool"]
    held_msg = next(m for m in tool_msgs if m["tool_call_id"] == "c2")
    assert "held" in held_msg["content"].lower()


def test_same_turn_different_args_for_same_target_still_dispatches_both():
    # A customer who says "add 1 wireless mouse, then actually make that 2" in
    # one message can legitimately produce two add_to_cart calls for the same
    # product with DIFFERENT quantities in the same turn — the guard must key
    # on the full argument set, not just the resolved target, so both go
    # through rather than the second being wrongly held.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "add_to_cart",
                     {"productName": "Wireless Mouse Pro", "quantity": 1})]),
        ChatResult(text="", tool_calls=[
            ToolCall("c2", "add_to_cart",
                     {"productName": "Wireless Mouse Pro", "quantity": 2})]),
        ChatResult(text="Updated to 2 Wireless Mouse Pro."),
    ])
    reg = StubRegistry(
        {"add_to_cart": ADD_TO_CART_SCHEMA},
        results={"add_to_cart": '{"found": true}'},
    )
    services = StubThreadServices(thread_msgs=[
        {"role": "user", "text": "add 1 wireless mouse pro, actually make that 2",
         "authorId": "u1", "displayName": "Alice"},
    ])
    ex = _executor(llm=llm, registry=reg, services=services)
    config = _config(tools=["add_to_cart"])

    result = ex._run_agent_node(CTX, RUN, STEP, config, {"threadId": "t1"})

    assert reg.dispatched == [
        ("add_to_cart", {"productName": "Wireless Mouse Pro", "quantity": 1}),
        ("add_to_cart", {"productName": "Wireless Mouse Pro", "quantity": 2}),
    ]
    assert not any(k == "same_turn_write_held" for k, _ in result.trace)


def test_same_turn_dedup_does_not_hold_an_off_turn_held_call():
    # A call that was itself HELD by K-058 (off-turn) must not be treated as
    # "already successfully dispatched" — only a genuinely successful dispatch
    # seeds the same-turn dedup set. Turn text never mentions the mouse at
    # all, so both mouse calls are off-turn holds (K-058), not K-061 holds.
    # (This covers only the K-058-held branch, which returns before
    # `dispatch_key` is even computed; the sibling test below covers the
    # separate "a failed dispatch attempt must not poison the set either"
    # branch, per `docs/reviews/salesperson-tool-reliability2-impl.md` MAJOR 1.)
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "add_to_cart",
                     {"productName": "Wireless Mouse Pro", "quantity": 1})]),
        ChatResult(text="", tool_calls=[
            ToolCall("c2", "add_to_cart",
                     {"productName": "Wireless Mouse Pro", "quantity": 1})]),
        ChatResult(text="Done."),
    ])
    reg = StubRegistry(
        {"add_to_cart": ADD_TO_CART_SCHEMA},
        results={"add_to_cart": '{"found": true}'},
    )
    services = StubThreadServices(thread_msgs=[
        {"role": "user", "text": "Also add 1 Mechanical Keyboard K200",
         "authorId": "u1", "displayName": "Alice"},
    ])
    ex = _executor(llm=llm, registry=reg, services=services)
    config = _config(tools=["add_to_cart"])

    result = ex._run_agent_node(CTX, RUN, STEP, config, {"threadId": "t1"})

    assert reg.dispatched == []
    kinds = [k for k, _ in result.trace]
    assert kinds.count("off_turn_write_held") == 2
    assert "same_turn_write_held" not in kinds


class _FailOnceThenSucceedRegistry(StubRegistry):
    """`add_to_cart`'s FIRST dispatch attempt raises a model-correctable
    `ServiceError` (a transient dispatch failure, e.g. `UnknownMemberError`);
    every dispatch after that succeeds normally, mirroring the model's own
    bounded re-prompt-and-retry convention (`_handle_tool_call`'s
    `MODEL_CORRECTABLE_TOOL_ERRORS` path). Used to prove a FAILED dispatch
    attempt does not seed the K-061 same-turn dedup set — only a genuinely
    successful one may (MAJOR 1, `docs/reviews/
    salesperson-tool-reliability2-impl.md`)."""

    def __init__(self, schemas, results, *, exc):
        super().__init__(schemas, results)
        self._exc = exc
        self._raised = False

    def dispatch(self, name, arguments, *, ctx, run):
        if not self._raised:
            self._raised = True
            self.dispatched.append((name, arguments))
            raise self._exc
        return super().dispatch(name, arguments, ctx=ctx, run=run)


def test_same_turn_dedup_is_not_seeded_by_a_failed_dispatch_attempt():
    # MAJOR 1 (`docs/reviews/salesperson-tool-reliability2-impl.md`): the code
    # is already correct — `dispatched_writes.add(dispatch_key)` sits strictly
    # after the try/except around `self._tools.dispatch`, so a call that RAISES
    # a model-correctable `ServiceError` must not seed the same-turn dedup set.
    # This pins that down: the model's own bounded re-prompt has it retry with
    # the IDENTICAL arguments after the first attempt fails, and that retry
    # must actually reach the tool — not be wrongly held as "already
    # dispatched this turn" just because an earlier, failed attempt used the
    # same arguments.
    llm = StubChatLLM([
        ChatResult(text="", tool_calls=[
            ToolCall("c1", "add_to_cart",
                     {"productName": "Wireless Mouse Pro", "quantity": 1})]),
        ChatResult(text="", tool_calls=[
            ToolCall("c2", "add_to_cart",
                     {"productName": "Wireless Mouse Pro", "quantity": 1})]),
        ChatResult(text="Added 1 Wireless Mouse Pro to your cart."),
    ])
    reg = _FailOnceThenSucceedRegistry(
        {"add_to_cart": ADD_TO_CART_SCHEMA},
        {"add_to_cart": '{"found": true}'},
        exc=UnknownMemberError(["n/a"]),
    )
    services = StubThreadServices(thread_msgs=[
        {"role": "user", "text": "Add 1 Wireless Mouse Pro please",
         "authorId": "u1", "displayName": "Alice"},
    ])
    ex = _executor(llm=llm, registry=reg, services=services)
    config = _config(tools=["add_to_cart"])

    result = ex._run_agent_node(CTX, RUN, STEP, config, {"threadId": "t1"})

    # the failed first attempt, then the identical-argument retry — BOTH
    # reach the tool; the retry is not wrongly held as an already-dispatched
    # same-turn repeat
    assert reg.dispatched == [
        ("add_to_cart", {"productName": "Wireless Mouse Pro", "quantity": 1}),
        ("add_to_cart", {"productName": "Wireless Mouse Pro", "quantity": 1}),
    ]
    assert not any(k == "same_turn_write_held" for k, _ in result.trace)
    assert result.output == "Added 1 Wireless Mouse Pro to your cart."
