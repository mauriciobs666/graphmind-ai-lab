"""`modelbench.convo` — prompt assembly and the scripted-conversation driver (plan §3.3, §3.8.4,
§4 S2). Offline throughout: a hand-built stub LLM callable and a hand-built `ToolEnvironment`, no
pack loader, no LM Studio, no network — matching `tests/test_lmstudio.py`'s "stub everything at the
boundary" convention and `tests/conftest.py`'s "built by hand" fixture convention.

Every `Turn`/`Conversation`/`PromptConfig` here is constructed directly (dataclass constructors),
never read from a pack file: `convo.py` is never imported by a pack (only `modelbench.tooling` is on
the AST allowlist, plan §3.3), so its types have no on-disk fixture format of their own to load.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_args

import pytest

from modelbench import convo
from modelbench.convo import (
    ChatMessage,
    Conversation,
    ConversationTrace,
    PromptConfig,
    Turn,
    assemble,
    drive,
)
from modelbench.lmstudio import ChatResult
from modelbench.tooling import DispatchRecord

# --------------------------------------------------------------------------------------------
# Shared fixtures — hand-built, offline
# --------------------------------------------------------------------------------------------


def make_cfg(**overrides: Any) -> PromptConfig:
    """A complete, valid `PromptConfig` — the §3.3 `pack.json` example's own literal scalars
    (`historyReplay: "structured"`, `representToolSchemasEachTurn: true`, `historyTurns: 0`,
    `temperature: 0.0`, `maxTokens: 1024`) as the baseline, overridable per test."""
    base: dict[str, Any] = {
        "systemPrompt": "You are a helpful shop assistant.",
        "toolSchemas": ({"name": "lookup_product_fact", "parameters": {}},),
        "historyReplay": "structured",
        "representToolSchemasEachTurn": True,
        "historyTurns": 0,
        "temperature": 0.0,
        "maxTokens": 1024,
    }
    base.update(overrides)
    return PromptConfig(**base)


def turn(seq: int, user: str, **expect: Any) -> Turn:
    return Turn(seq=seq, user=user, expect=dict(expect))


def conversation(
    script_id: str, turns: tuple[Turn, ...], shape: str = "A", replicate: int = 1
) -> Conversation:
    return Conversation(scriptId=script_id, shape=shape, replicate=replicate, turns=turns)


class StubEnvironment:
    """A hand-built `ToolEnvironment` mirroring what a pack's `tools/sim.py` would build — it
    records its own `DispatchRecord`s, the way the real plugin contract requires (`dispatch`'s
    recording is the environment's own job, not `drive`'s or `modelbench.tooling`'s)."""

    def __init__(
        self,
        *,
        schemas: tuple[Mapping[str, Any], ...] = (),
        results: Mapping[str, Any] | None = None,
    ) -> None:
        self._schemas = schemas
        self._results = dict(results or {})
        self._trace: list[DispatchRecord] = []
        self._state: dict[str, Any] = {}

    def schemas(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self._schemas]

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
        return_value = self._results.get(name, {"ok": True})
        self._trace.append(
            DispatchRecord(
                name=name,
                rawArguments=arguments,
                parsedArguments=arguments,
                returnValue=return_value,
                timestamp="2026-09-09T00:00:00Z",
            )
        )
        self._state[f"lastCall_{name}"] = dict(arguments)
        self._state["callCount"] = len(self._trace)
        return return_value

    def trace(self) -> list[DispatchRecord]:
        return list(self._trace)

    def state(self) -> dict[str, Any]:
        return dict(self._state)


def stub_llm(responses: list[ChatResult]):
    """A stub `llm` callable in `drive`'s expected shape: `llm(messages, *, tools, temperature,
    max_tokens) -> ChatResult`, `model=`/`timeout_s=` already pre-bound by (a stand-in for) the
    caller — `drive` never supplies either. `.calls` records every invocation's keyword arguments
    for assertion, in call order."""
    calls: list[dict[str, Any]] = []

    def _llm(messages, *, tools=None, temperature, max_tokens):
        index = len(calls)
        calls.append(
            {
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return responses[index]

    _llm.calls = calls  # type: ignore[attr-defined]
    return _llm


def chat_result(
    *,
    content: str | None = "ok",
    tool_calls: tuple[Mapping[str, Any], ...] = (),
) -> ChatResult:
    form = "native" if tool_calls else "prose"
    return ChatResult(
        message={"role": "assistant", "content": content},
        tool_calls=tool_calls,
        toolCallForm=form,
        stats=None,
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=5.0,
    )


def native_call(call_id: str, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(dict(arguments))},
    }


# --------------------------------------------------------------------------------------------
# assemble — historyReplay's three axes (§5 test 10)
# --------------------------------------------------------------------------------------------


def test_assemble_out_of_range_turn_index_raises() -> None:
    history = (turn(1, "hi", toolRequired=False),)
    with pytest.raises(ValueError, match="out of range"):
        assemble(1, history, make_cfg())
    with pytest.raises(ValueError, match="out of range"):
        assemble(-1, history, make_cfg())


def test_assemble_unknown_history_replay_raises() -> None:
    history = (turn(1, "hi", toolRequired=False),)
    cfg = make_cfg(historyReplay="verbose")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="historyReplay"):
        assemble(0, history, cfg)


def test_assemble_current_turn_user_message_is_always_last() -> None:
    history = (
        turn(1, "first", toolRequired=False, finalReplyMustContain=["ok"]),
        turn(2, "second", toolRequired=False, finalReplyMustContain=["ok"]),
    )
    for mode in ("structured", "plaintext", "none"):
        messages = assemble(1, history, make_cfg(historyReplay=mode))
        assert messages[-1] == {"role": "user", "content": "second"}


def test_assemble_none_mode_produces_no_history_messages_at_all() -> None:
    history = (
        turn(1, "first", toolRequired=True, tool="lookup_product_fact", args={"name": "Pad"}),
        turn(2, "second", toolRequired=False, finalReplyMustContain=["ok"]),
    )
    cfg = make_cfg(historyReplay="none")
    messages = assemble(1, history, cfg)
    roles = [m["role"] for m in messages]
    # system prompt + tool-schema block + current user only — no trace of turn 1 at all.
    assert roles == ["system", "system", "user"]
    assert "first" not in json.dumps(messages)


def test_assemble_no_system_prompt_omits_the_system_message() -> None:
    history = (turn(1, "hi", toolRequired=False, finalReplyMustContain=["ok"]),)
    cfg = make_cfg(systemPrompt=None, toolSchemas=())
    messages = assemble(0, history, cfg)
    assert messages == [{"role": "user", "content": "hi"}]


def test_assemble_plaintext_mode_flattens_prior_turns_into_a_single_message() -> None:
    history = (
        turn(1, "what is the price of the Pad?", toolRequired=True, tool="lookup_product_fact",
             args={"name": "Pad"}, finalReplyMustContain=["24.99"]),
        turn(2, "and the Mug?", toolRequired=True, tool="lookup_product_fact",
             args={"name": "Mug"}, finalReplyMustContain=["9.99"]),
        turn(3, "add the Pad to my cart", toolRequired=False, finalReplyMustContain=["ok"]),
    )
    cfg = make_cfg(historyReplay="plaintext", representToolSchemasEachTurn=False)
    messages = assemble(2, history, cfg)
    # system + schema (turn 0 only, but we're assembling turn 2 with representToolSchemasEachTurn
    # False) + one flattened history message + current user.
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "user"]
    flattened = messages[1]["content"]
    assert "what is the price of the Pad?" in flattened
    assert "and the Mug?" in flattened
    assert "24.99" in flattened
    assert "9.99" in flattened
    assert messages[-1] == {"role": "user", "content": "add the Pad to my cart"}


def test_assemble_structured_mode_replays_native_assistant_tool_message_pairs() -> None:
    history = (
        turn(1, "price of the Pad?", toolRequired=True, tool="lookup_product_fact",
             args={"name": "Pad"}, finalReplyMustContain=["24.99"]),
        turn(2, "thanks, anything else?", toolRequired=False, finalReplyMustContain=["no"]),
    )
    cfg = make_cfg(historyReplay="structured", toolSchemas=())
    messages = assemble(1, history, cfg)
    # turn 2 (index 1) is the turn being assembled; only turn 1 (index 0) is replayed as history:
    # system, user(t1), assistant(tool_calls), tool, user(current).
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assistant_msg = messages[2]
    assert assistant_msg["content"] is None
    call = assistant_msg["tool_calls"][0]
    assert call["function"]["name"] == "lookup_product_fact"
    assert json.loads(call["function"]["arguments"]) == {"name": "Pad"}
    tool_msg = messages[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == call["id"]
    assert tool_msg["name"] == "lookup_product_fact"
    assert "24.99" in tool_msg["content"]
    assert messages[-1] == {"role": "user", "content": "thanks, anything else?"}


def test_assemble_structured_mode_prior_turn_without_a_tool_call_has_no_tool_message() -> None:
    history = (
        turn(1, "hi", toolRequired=False, finalReplyMustContain=["hello!"]),
        turn(2, "bye", toolRequired=False, finalReplyMustContain=["goodbye!"]),
    )
    cfg = make_cfg(historyReplay="structured", toolSchemas=())
    messages = assemble(1, history, cfg)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assistant_msg = messages[2]
    assert "tool_calls" not in assistant_msg
    assert assistant_msg["content"] == "hello!"


def test_assemble_represent_tool_schemas_each_turn_false_drops_schemas_after_turn_1() -> None:
    """The literal §5 test 10 requirement: `representToolSchemasEachTurn=False` really does drop
    the tool-schema message after the first assembled turn."""
    history = (
        turn(1, "t1", toolRequired=False, finalReplyMustContain=["a"]),
        turn(2, "t2", toolRequired=False, finalReplyMustContain=["b"]),
        turn(3, "t3", toolRequired=False, finalReplyMustContain=["c"]),
    )
    cfg = make_cfg(representToolSchemasEachTurn=False)

    def has_schema_message(messages: list[ChatMessage]) -> bool:
        return any(
            "lookup_product_fact" in json.dumps(m) for m in messages if m["role"] == "system"
        )

    assert has_schema_message(assemble(0, history, cfg)) is True
    assert has_schema_message(assemble(1, history, cfg)) is False
    assert has_schema_message(assemble(2, history, cfg)) is False


def test_assemble_represent_tool_schemas_each_turn_true_keeps_schemas_every_turn() -> None:
    """"Every turn" is checked at all three of this fixture's turns, matching the thoroughness of
    the `False` counterpart above (which checks all three of its own) — not just the first two, so
    the name's "every" is not a claim the assertions under-cover."""
    history = (
        turn(1, "t1", toolRequired=False, finalReplyMustContain=["a"]),
        turn(2, "t2", toolRequired=False, finalReplyMustContain=["b"]),
        turn(3, "t3", toolRequired=False, finalReplyMustContain=["c"]),
    )
    cfg = make_cfg(representToolSchemasEachTurn=True)

    def has_schema_message(messages: list[ChatMessage]) -> bool:
        return any(
            "lookup_product_fact" in json.dumps(m) for m in messages if m["role"] == "system"
        )

    assert has_schema_message(assemble(0, history, cfg)) is True
    assert has_schema_message(assemble(1, history, cfg)) is True
    assert has_schema_message(assemble(2, history, cfg)) is True


@pytest.mark.parametrize("represent_each_turn", [True, False])
def test_assemble_no_tool_schemas_means_no_schema_message_regardless_of_the_flag(
    represent_each_turn: bool,
) -> None:
    history = (turn(1, "t1", toolRequired=False, finalReplyMustContain=["a"]),)
    cfg = make_cfg(toolSchemas=(), representToolSchemasEachTurn=represent_each_turn)
    messages = assemble(0, history, cfg)
    assert [m["role"] for m in messages] == ["system", "user"]


def test_assemble_history_turns_zero_replays_every_prior_turn() -> None:
    history = (
        turn(1, "t1", toolRequired=False, finalReplyMustContain=["a"]),
        turn(2, "t2", toolRequired=False, finalReplyMustContain=["b"]),
        turn(3, "t3", toolRequired=False, finalReplyMustContain=["c"]),
        turn(4, "t4", toolRequired=False, finalReplyMustContain=["d"]),
    )
    cfg = make_cfg(historyReplay="plaintext", historyTurns=0, toolSchemas=())
    messages = assemble(3, history, cfg)
    flattened = messages[-2]["content"]
    assert "t1" in flattened and "t2" in flattened and "t3" in flattened


@pytest.mark.parametrize(
    ("window", "expected_present", "expected_absent"),
    [
        (1, ("t3",), ("t1", "t2")),
        (2, ("t2", "t3"), ("t1",)),
    ],
)
def test_assemble_history_turns_windows_to_only_the_last_n_prior_turns(
    window: int, expected_present: tuple[str, ...], expected_absent: tuple[str, ...]
) -> None:
    """Two window sizes, not just `N=1` — "the last N" is a claim about an arbitrary N, so a
    second value (`N=2`) is what confirms the windowing is a general slice rather than a
    special-cased "keep exactly one" branch that happens to pass at `N=1`."""
    history = (
        turn(1, "t1", toolRequired=False, finalReplyMustContain=["a"]),
        turn(2, "t2", toolRequired=False, finalReplyMustContain=["b"]),
        turn(3, "t3", toolRequired=False, finalReplyMustContain=["c"]),
        turn(4, "t4", toolRequired=False, finalReplyMustContain=["d"]),
    )
    cfg = make_cfg(historyReplay="plaintext", historyTurns=window, toolSchemas=())
    messages = assemble(3, history, cfg)
    flattened = messages[-2]["content"]
    for present in expected_present:
        assert present in flattened
    for absent in expected_absent:
        assert absent not in flattened


def test_assemble_transcribed_from_the_plans_own_conversation_row_literal() -> None:
    """§3.8.4's own `conversations.jsonl` row example, transcribed verbatim (including
    `argChecks`/`terminal`, which `assemble` does not read but must tolerate) — a check-input
    derived from the plan's literal, not from this implementation's own idea of what a turn looks
    like."""
    row = json.loads(
        """
        {"scriptId": "A-02", "shape": "A", "replicate": 1,
         "description": "read-only catalog lookups",
         "turns": [{"seq": 1, "user": "What's the price of the Wireless Charging Pad?",
                    "expect": {"toolRequired": true, "tool": "lookup_product_fact",
                               "args": {"name": "Wireless Charging Pad"},
                               "argChecks": [{"kind": "boundary", "arg": "maxPrice", "value": 50}],
                               "terminal": false,
                               "finalReplyMustContain": ["24.99"]}}],
         "provenance": {"draftedBy": "x", "verifiedBy": "y", "basedOn": "z"}}
        """
    )
    script_turn = Turn(seq=row["turns"][0]["seq"], user=row["turns"][0]["user"],
                        expect=row["turns"][0]["expect"])
    next_turn = turn(2, "thanks", toolRequired=False, finalReplyMustContain=["you're welcome"])
    history = (script_turn, next_turn)
    cfg = make_cfg(historyReplay="structured", toolSchemas=())

    messages = assemble(1, history, cfg)

    assistant_msg = next(m for m in messages if m["role"] == "assistant")
    call = assistant_msg["tool_calls"][0]
    assert call["function"]["name"] == "lookup_product_fact"
    assert json.loads(call["function"]["arguments"]) == {"name": "Wireless Charging Pad"}
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert "24.99" in tool_msg["content"]


# --------------------------------------------------------------------------------------------
# drive — one LLM call per scripted turn, dispatched against a real (stub) environment
# --------------------------------------------------------------------------------------------


def test_drive_calls_the_llm_exactly_once_per_scripted_turn() -> None:
    script = conversation(
        "S-01",
        (
            turn(1, "t1", toolRequired=False, finalReplyMustContain=["a"]),
            turn(2, "t2", toolRequired=False, finalReplyMustContain=["b"]),
            turn(3, "t3", toolRequired=False, finalReplyMustContain=["c"]),
        ),
    )
    llm = stub_llm([chat_result() for _ in range(3)])
    env = StubEnvironment()
    trace = drive(env, script, llm, make_cfg(toolSchemas=()))
    assert len(llm.calls) == 3
    assert len(trace.turns) == 3


def test_drive_returns_a_conversation_trace_named_for_the_script() -> None:
    script = conversation("S-42", (turn(1, "t1", toolRequired=False, finalReplyMustContain=["a"]),))
    llm = stub_llm([chat_result()])
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    assert isinstance(trace, ConversationTrace)
    assert trace.scriptId == "S-42"


def test_drive_dispatches_a_native_tool_call_with_parsed_arguments() -> None:
    script = conversation(
        "S-02", (turn(1, "price of Pad?", toolRequired=True, tool="lookup_product_fact",
                       args={"name": "Pad"}),)
    )
    call = native_call("call_1", "lookup_product_fact", {"name": "Pad"})
    llm = stub_llm([chat_result(content=None, tool_calls=(call,))])
    env = StubEnvironment(results={"lookup_product_fact": {"price": 24.99}})
    trace = drive(env, script, llm, make_cfg(toolSchemas=()))
    dispatches = trace.turns[0].dispatches
    assert len(dispatches) == 1
    assert dispatches[0].name == "lookup_product_fact"
    assert dispatches[0].parsedArguments == {"name": "Pad"}
    assert dispatches[0].returnValue == {"price": 24.99}


def test_drive_turn_without_a_tool_call_dispatches_nothing() -> None:
    script = conversation("S-03", (turn(1, "hi", toolRequired=False),))
    llm = stub_llm([chat_result(content="hello!")])
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    assert trace.turns[0].dispatches == ()


def test_drive_isolates_each_turns_dispatches_from_every_other_turn() -> None:
    script = conversation(
        "S-04",
        (
            turn(1, "price of Pad?", toolRequired=True, tool="lookup_product_fact",
                 args={"name": "Pad"}),
            turn(2, "price of Mug?", toolRequired=True, tool="lookup_product_fact",
                 args={"name": "Mug"}),
        ),
    )
    calls = [
        native_call("c1", "lookup_product_fact", {"name": "Pad"}),
        native_call("c2", "lookup_product_fact", {"name": "Mug"}),
    ]
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(calls[0],)),
            chat_result(content=None, tool_calls=(calls[1],)),
        ]
    )
    env = StubEnvironment()
    trace = drive(env, script, llm, make_cfg(toolSchemas=()))
    assert len(trace.turns[0].dispatches) == 1
    assert trace.turns[0].dispatches[0].parsedArguments == {"name": "Pad"}
    assert len(trace.turns[1].dispatches) == 1
    assert trace.turns[1].dispatches[0].parsedArguments == {"name": "Mug"}
    # The environment's own full trace grew to 2 — drive's per-turn slicing, not the environment,
    # is what keeps each TurnTrace to its own share.
    assert len(env.trace()) == 2


def test_drive_env_state_is_captured_after_each_turns_own_dispatches() -> None:
    script = conversation(
        "S-05",
        (
            turn(1, "add Pad", toolRequired=True, tool="add_to_cart", args={"name": "Pad"}),
            turn(2, "add Mug", toolRequired=True, tool="add_to_cart", args={"name": "Mug"}),
        ),
    )
    calls = (
        native_call("c1", "add_to_cart", {"name": "Pad"}),
        native_call("c2", "add_to_cart", {"name": "Mug"}),
    )
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(calls[0],)),
            chat_result(content=None, tool_calls=(calls[1],)),
        ]
    )
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    assert trace.turns[0].envState["callCount"] == 1
    assert trace.turns[1].envState["callCount"] == 2


def test_drive_passes_native_tools_param_every_turn_even_when_schemas_not_restated() -> None:
    """The design decision documented in `convo.py`'s module docstring:
    `representToolSchemasEachTurn=False` withholds the *textual* schema block `assemble`
    builds, never the native `tools=`
    parameter `llm` receives — otherwise native tool calls would be structurally impossible from
    turn 2 onward on every pack that sets the flag `False`."""
    script = conversation(
        "S-06",
        (
            turn(1, "t1", toolRequired=False, finalReplyMustContain=["a"]),
            turn(2, "t2", toolRequired=False, finalReplyMustContain=["b"]),
        ),
    )
    llm = stub_llm([chat_result(), chat_result()])
    cfg = make_cfg(representToolSchemasEachTurn=False)
    drive(StubEnvironment(), script, llm, cfg)
    assert llm.calls[0]["tools"] == [dict(s) for s in cfg.toolSchemas]
    assert llm.calls[1]["tools"] == [dict(s) for s in cfg.toolSchemas]


def test_drive_replays_history_from_the_script_never_from_this_runs_own_model_output() -> None:
    """Pins the module's central design decision. Turn 1's *scripted* `expect` names
    `lookup_product_fact("Pad")`, but the stub model actually calls `lookup_product_fact("Different
    Item")` this run. Turn 2's assembled messages must replay turn 1 using the *scripted* args
    ("Pad"), never the model's own real ones ("Different Item") — because the determinism probe
    (plan §3.8.4) re-runs a script and diffs outcome vectors turn by turn, which is only a clean
    comparison if every run sees an identical context regardless of what the model under test
    actually did."""
    script = conversation(
        "S-07",
        (
            turn(1, "price of the Pad?", toolRequired=True, tool="lookup_product_fact",
                 args={"name": "Pad"}, finalReplyMustContain=["24.99"]),
            turn(2, "anything else?", toolRequired=False, finalReplyMustContain=["no"]),
        ),
    )
    real_call = native_call("real", "lookup_product_fact", {"name": "Different Item"})
    llm = stub_llm([chat_result(content=None, tool_calls=(real_call,)), chat_result()])
    drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    turn_2_messages = llm.calls[1]["messages"]
    serialized = json.dumps(turn_2_messages)
    assert "Pad" in serialized
    assert "Different Item" not in serialized


def test_drive_malformed_tool_call_missing_a_function_name_is_skipped_not_raised() -> None:
    script = conversation("S-08", (turn(1, "hi", toolRequired=False),))
    broken_call = {"id": "x", "type": "function", "function": {"arguments": "{}"}}
    llm = stub_llm([chat_result(content=None, tool_calls=(broken_call,))])
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    assert trace.turns[0].dispatches == ()


def test_drive_never_catches_an_error_the_llm_callable_raises() -> None:
    class _Boom(RuntimeError):
        pass

    def failing_llm(messages, *, tools=None, temperature, max_tokens):
        raise _Boom("simulated LM Studio failure")

    script = conversation("S-09", (turn(1, "hi", toolRequired=False),))
    with pytest.raises(_Boom):
        drive(StubEnvironment(), script, failing_llm, make_cfg(toolSchemas=()))


def test_drive_records_a_nonnegative_wall_clock_per_turn() -> None:
    """Checked on every turn of a 3-turn script, not just the first — "per turn" names a property
    of each turn's own record, not only turn 0's."""
    script = conversation(
        "S-10",
        (
            turn(1, "t1", toolRequired=False),
            turn(2, "t2", toolRequired=False),
            turn(3, "t3", toolRequired=False),
        ),
    )
    llm = stub_llm([chat_result(), chat_result(), chat_result()])
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    assert len(trace.turns) == 3
    for turn_trace in trace.turns:
        assert turn_trace.wallClockMs is not None
        assert turn_trace.wallClockMs >= 0.0


def test_drive_raises_type_error_before_any_llm_call_when_env_is_not_a_tool_environment() -> None:
    """Pins the fix for the defect `tooling.py`'s own docstring named: `ToolEnvironment` is
    `@runtime_checkable` *so that* `drive` can assert conformance — a claim that must be backed by
    `drive` actually calling `isinstance`, not merely by the Protocol supporting the call.
    "Before any LLM call" is checked via `llm.calls` staying empty, not just via the raise."""

    class _NotAnEnvironment:
        pass

    script = conversation("S-11", (turn(1, "hi", toolRequired=False),))
    llm = stub_llm([chat_result()])
    with pytest.raises(TypeError, match="ToolEnvironment"):
        drive(_NotAnEnvironment(), script, llm, make_cfg(toolSchemas=()))
    assert llm.calls == []


# --------------------------------------------------------------------------------------------
# `TURN_DISPOSITIONS` — plan §3.8.4's four-row `turnDisposition` table, asserted (plan §4 S2)
# --------------------------------------------------------------------------------------------
#
# §4 S2 requires a **three-way** probe: `set(get_args(TurnDisposition))`, the set of dispositions
# the S5 scorer branches on, and `TURN_DISPOSITIONS` must **each** equal a constant transcribed
# into this test from §3.8.4's table — each against the transcript, never against each other,
# because two sets authored in one unit agree by construction and a probe that cannot redden is
# not a guard (`AGENTS.md`, "A guard's reach lives in an asserted constant", form (i)).
#
# THE THIRD LEG IS ABSENT ON PURPOSE, AND IT IS OWED. The S5 scorer that branches on these
# dispositions is not built yet — §4 S2 says this row splits across stages — so only two of the
# three declarations exist today. The absence is *blocked on unbuilt work*, not a choice, and
# `test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5` below is the tripwire that
# refuses to let it be forgotten: it reddens the moment a scorer package appears.

#: Transcribed by hand from plan §3.8.4's four-row table (v1.26), one row per mechanism:
#: `replied` (a response with no tool calls ended the loop) · `cap-hit` (`maxIterationsPerTurn`
#: reached) · `no-response` (the call did not complete — `LMStudioCallTimeout`, or
#: `LMStudioCallFailed` with no HTTP status) · `server-rejected` (the server answered and
#: refused — `LMStudioCallFailed` carrying an HTTP status). This literal is the *independent*
#: declaration the module's two are each bound to; never derive it from either of them.
_DISPOSITIONS_PER_PLAN_3_8_4 = {"replied", "cap-hit", "no-response", "server-rejected"}


def test_turn_disposition_literal_is_exactly_the_plan_table() -> None:
    """Leg 1 of §4 S2's three-way probe: the type annotation `TurnTrace.turnDisposition` will
    carry, against the transcript above."""
    assert set(get_args(convo.TurnDisposition)) == _DISPOSITIONS_PER_PLAN_3_8_4


def test_turn_dispositions_constant_is_exactly_the_plan_table() -> None:
    """Leg 2: the runtime constant a consumer validates against, against the same transcript.
    Deliberately not asserted against leg 1 — the two module-level declarations are written out
    separately so that a fifth member added to either one alone reddens here."""
    assert convo.TURN_DISPOSITIONS == frozenset(_DISPOSITIONS_PER_PLAN_3_8_4)


def test_turn_dispositions_is_an_immutable_frozenset() -> None:
    """Appendix A types it `frozenset[str]`, and it is about to be imported by a module this one
    does not own (S5's scorer, the rework unit's `drive`). A mutable set shared that way is a
    constant only by convention."""
    assert isinstance(convo.TURN_DISPOSITIONS, frozenset)


def test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5() -> None:
    """Not a test of `convo`, and deliberately so: it is the named placeholder for the leg that
    cannot be written yet, so that its absence above cannot be read as an oversight.

    §4 S2's probe is three-way. The third set — the dispositions S5's scorer actually branches on
    — has no declaration to bind while `modelbench/scoring/` does not exist. When it does, this
    test fails, and the fix is to add the third assertion against
    `_DISPOSITIONS_PER_PLAN_3_8_4` above and delete this one. (If S5's scorer lands somewhere
    other than that package, this tripwire will not fire and the leg is still owed — which is
    why the reason is written out here rather than left to the assertion.)"""
    scoring_pkg = Path(convo.__file__).parent / "scoring"
    assert not scoring_pkg.exists(), (
        f"{scoring_pkg} now exists: wire S5's branch set into this file as the third leg of "
        "§4 S2's disposition probe, then delete this tripwire."
    )
