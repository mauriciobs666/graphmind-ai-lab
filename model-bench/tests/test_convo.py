"""`modelbench.convo` — prompt assembly and the scripted-conversation driver (plan §3.3, §3.8.4,
§4 S2). Offline throughout: a hand-built stub LLM callable and a hand-built `ToolEnvironment`, no
pack loader, no LM Studio, no network — matching `tests/test_lmstudio.py`'s "stub everything at the
boundary" convention and `tests/conftest.py`'s "built by hand" fixture convention.

Every `Turn`/`Conversation`/`PromptConfig` here is constructed directly (dataclass constructors),
never read from a pack file: `convo.py` is never imported by a pack (only `modelbench.tooling` is on
the AST allowlist, plan §3.3), so its types have no on-disk fixture format of their own to load.

**`assemble` is driven from `observed` throughout, never from a script's `expect`** — §3.8.4's
*"Prompt assembly"* ruling. Several fixtures below deliberately give the model output that shares no
string with the script's oracle, so a replay that reached for `expect` would redden rather than
coincide.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import pytest

from modelbench import convo
from modelbench.convo import (
    ChatMessage,
    Conversation,
    ConversationTrace,
    PromptConfig,
    ToolDispatchFailed,
    TraceContractViolated,
    Turn,
    TurnTrace,
    assemble,
    drive,
)
from modelbench.lmstudio import (
    ChatResult,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    LMStudioError,
    LMStudioUnreachable,
    ToolCallingIneligible,
)
from modelbench.tooling import DispatchRecord

# --------------------------------------------------------------------------------------------
# Shared fixtures — hand-built, offline
# --------------------------------------------------------------------------------------------


def make_cfg(**overrides: Any) -> PromptConfig:
    """A complete, valid `PromptConfig` — the §3.3 `pack.json` example's own literal scalars
    (`historyReplay: "structured"`, `representToolSchemasEachTurn: true`, `historyTurns: 0`,
    `maxIterationsPerTurn: 8`, `temperature: 0.0`, `maxTokens: 1024`) as the baseline, overridable
    per test. The example declares `historyReplay: "structured-replies-only"`; `"structured"` is
    kept as this helper's baseline because it is the mode with the most structure to assert, and
    every mode-specific test names its own."""
    base: dict[str, Any] = {
        "systemPrompt": "You are a helpful shop assistant.",
        "toolSchemas": ({"name": "lookup_product_fact", "parameters": {}},),
        "historyReplay": "structured",
        "representToolSchemasEachTurn": True,
        "historyTurns": 0,
        "maxIterationsPerTurn": 8,
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


class ResettingEnvironment(StubEnvironment):
    """A plausible **pack** defect: an environment that clears its own trace inside `dispatch`, so
    `drive`'s trace diff yields an *empty* slice and a real, state-mutating call is recorded as
    zero dispatches (plan gate P15-5).

    Deliberately its own fixture rather than an assertion bolted onto `StubEnvironment`, which
    conforms by construction and so could never redden."""

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
        return_value = super().dispatch(name, arguments)
        self._trace.clear()
        return return_value


class SkewEnvironment(StubEnvironment):
    """A **pack** defect no per-*iteration* aggregate can see: the environment's own bookkeeping is
    off by one in both directions **inside a single iteration** — it records nothing for the first
    call it is handed and two entries for the second. Two calls, two new entries, so any check that
    compares the iteration's growth against the iteration's dispatch count balances exactly.

    What it costs is the thing `tooling.py` states as a per-**call** contract and
    `TraceContractViolated` calls a checked fact: `assemble` reads the tool-message pairing
    positionally out of the slice, so call 1's `tool` message would carry call 2's return value.

    Its own fixture rather than a flag on `StubEnvironment` for `ResettingEnvironment`'s reason:
    an environment that conforms by construction can never redden.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._dispatch_count = 0

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
        self._dispatch_count += 1
        return_value = self._results.get(name, {"ok": True})
        # The call really runs and really mutates state either way — that is what makes losing
        # its record a defect rather than bookkeeping.
        self._state[f"lastCall_{name}"] = dict(arguments)
        self._state["callCount"] = self._dispatch_count
        if self._dispatch_count == 1:
            return return_value  # executed, recorded nowhere
        for suffix in ("A", "B"):
            self._trace.append(
                DispatchRecord(
                    name=name,
                    rawArguments=arguments,
                    parsedArguments=arguments,
                    returnValue={"from": f"{name}-{suffix}"},
                    timestamp="2026-09-09T00:00:00Z",
                )
            )
        return return_value


class ReadMutatingEnvironment(StubEnvironment):
    """The **pack** defect the per-iteration check is designed to catch: `trace()` drops its
    oldest entry on every *read*, so the entry's visibility depends on how many reads occur
    between two comparison points — which is why the per-iteration and per-turn re-takes of the
    prefix check are needed even when the per-call check is present. The per-call and per-iteration
    checks are not independently reachable via deletion (impl review Pass 18, P18-1): whether the
    per-call check's extra reads occur affects when the drop happens, changing which check fires
    first. This is a test-fixture coupling only — the production guard logic is unaffected and
    all detectable defects are still caught (`TraceContractViolated` docstring for the full
    explanation).
    """

    def trace(self) -> list[DispatchRecord]:
        entries = list(self._trace)
        if self._trace:
            del self._trace[0]
        return entries


def stub_llm(responses: list[Any]):
    """A stub `llm` callable in `drive`'s expected shape: `llm(messages, *, tools, temperature,
    max_tokens) -> ChatResult`, `model=`/`timeout_s=` already pre-bound by (a stand-in for) the
    caller — `drive` never supplies either. `.calls` records every invocation's keyword arguments
    for assertion, in call order.

    An entry of `responses` that is a `BaseException` **instance** is raised instead of returned,
    which is how the disposition cases below drive `drive`'s exception path."""
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
        response = responses[index]
        if isinstance(response, BaseException):
            raise response
        return response

    _llm.calls = calls  # type: ignore[attr-defined]
    return _llm


def chat_result(
    *,
    content: str | None = "ok",
    tool_calls: tuple[Mapping[str, Any], ...] = (),
    wall_clock_ms: float = 5.0,
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
        wallClockMs=wall_clock_ms,
    )


def native_call(call_id: str, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(dict(arguments))},
    }


def observed_turn(
    *,
    chat_results: tuple[ChatResult, ...] = (),
    dispatches: tuple[DispatchRecord, ...] = (),
    disposition: str = "replied",
    final_reply: str | None = "observed reply",
    env_state: Mapping[str, Any] | None = None,
) -> TurnTrace:
    """A hand-built `TurnTrace` standing in for what `drive` recorded on a prior turn — what
    `assemble` replays *from*. `finalReplyText` is `None` iff the disposition is not `replied`, the
    invariant §3.8.4's table owns; this helper keeps the two consistent by construction so a fixture
    cannot accidentally assert against an impossible record."""
    reply = final_reply if disposition == "replied" else None
    return TurnTrace(
        messagesSent=(),
        chatResults=chat_results,
        dispatches=dispatches,
        envState=dict(env_state or {}),
        iterations=len(chat_results),
        turnDisposition=disposition,  # type: ignore[arg-type]
        finalReplyText=reply,
        wallClockMs=1.0,
    )


def dispatch_record(name: str, arguments: Mapping[str, Any], return_value: Any) -> DispatchRecord:
    return DispatchRecord(
        name=name,
        rawArguments=arguments,
        parsedArguments=arguments,
        returnValue=return_value,
        timestamp="2026-09-09T00:00:00Z",
    )


#: A prior turn that really did dispatch a tool: one iteration emitting a native call, one
#: dispatch record, then a final reply. Case (d) of §5 test 10 turns on this being a *fact about
#: the fixture* — the two reply-text modes must show no tool evidence for a turn that has some.
def tool_calling_prior_turn(reply: str = "That one is 24.99.") -> TurnTrace:
    call = native_call("c1", "lookup_product_fact", {"name": "Pad"})
    return observed_turn(
        chat_results=(
            chat_result(content=None, tool_calls=(call,)),
            chat_result(content=reply),
        ),
        dispatches=(dispatch_record("lookup_product_fact", {"name": "Pad"}, {"price": 24.99}),),
        final_reply=reply,
    )


class _FakeClock:
    """A stub monotonic clock in milliseconds, advanced explicitly by the stubs that consume time —
    so a turn's wall clock is an exact arithmetic fact about the fixture rather than a measurement
    of how fast this box happened to run the test."""

    def __init__(self, start_s: float = 1000.0) -> None:
        self._now_s = start_s

    def __call__(self) -> float:
        return self._now_s

    def advance_ms(self, ms: float) -> None:
        self._now_s += ms / 1000.0


class _TickingEnvironment(StubEnvironment):
    """A conforming `StubEnvironment` whose `dispatch` also advances the stub clock — the tool
    dispatches *between* a turn's iterations are real time the operator waits through, and §3.8.4
    puts them inside the turn's wall clock."""

    def __init__(self, clock: _FakeClock, *, dispatch_ms: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._clock = clock
        self._dispatch_ms = dispatch_ms

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
        self._clock.advance_ms(self._dispatch_ms)
        return super().dispatch(name, arguments)


ALL_MODES = ("structured", "structured-replies-only", "plaintext", "none")
REPLY_TEXT_MODES = ("structured-replies-only", "plaintext")


# --------------------------------------------------------------------------------------------
# `PromptConfig` — Appendix A's field list, in order (plan gate P15-9)
# --------------------------------------------------------------------------------------------
#
# These names are `pack.json`'s `prompt` keys, so a drift is a silently-unreadable manifest rather
# than a type error. Transcribed by hand from Appendix A's `PromptConfig` row, in the same shape as
# `tests/test_tooling.py`'s `DispatchRecord` pin, which is the working template.

_APPENDIX_A_PROMPT_CONFIG_FIELDS = (
    "systemPrompt",
    "toolSchemas",
    "historyReplay",
    "representToolSchemasEachTurn",
    "historyTurns",
    "maxIterationsPerTurn",
    "temperature",
    "maxTokens",
)


def test_prompt_config_fields_match_the_plans_appendix_a_literal_in_order() -> None:
    fields = tuple(f.name for f in dataclasses.fields(PromptConfig))
    assert fields == _APPENDIX_A_PROMPT_CONFIG_FIELDS


# --------------------------------------------------------------------------------------------
# `historyReplay` — §3.3's four values, bound and behaviourally distinguished (plan gate P15-3)
# --------------------------------------------------------------------------------------------
#
# Two declarations of one set, written out separately in `convo.py` on purpose. Binding them here
# is form (i) of the guard-reach convention: non-tautological *because* neither is derived from the
# other. The "accepted set equals the constant" form is deliberately **not** used — with a pure
# membership guard it is true of any constant and reddens on nothing (Pass 15 §4).


def test_history_replay_literal_and_mode_constant_are_the_same_four_values() -> None:
    assert convo._HISTORY_REPLAY_MODES == set(get_args(convo.HistoryReplay))
    assert len(convo._HISTORY_REPLAY_MODES) == 4


def test_every_history_replay_mode_renders_a_distinct_message_list() -> None:
    """A member of the constant with no branch in `assemble` would be *accepted* and silently
    degrade to `none` — which is what a membership-only guard cannot see. One fixture with a prior
    tool-calling turn, four modes, four different renderings."""
    script = (turn(1, "price of the Pad?"), turn(2, "anything else?"))
    observed = (tool_calling_prior_turn(),)
    rendered = {
        mode: json.dumps(assemble(1, script, observed, make_cfg(historyReplay=mode)))
        for mode in ALL_MODES
    }
    assert len(set(rendered.values())) == 4


def test_assemble_refuses_a_mode_outside_the_four() -> None:
    script = (turn(1, "hi"),)
    cfg = make_cfg(historyReplay="verbose")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="historyReplay"):
        assemble(0, script, (), cfg)


# --------------------------------------------------------------------------------------------
# assemble — preconditions (§4 S2's replay contract)
# --------------------------------------------------------------------------------------------


def test_assemble_out_of_range_turn_index_raises() -> None:
    script = (turn(1, "hi"),)
    with pytest.raises(ValueError, match="out of range"):
        assemble(1, script, (observed_turn(),), make_cfg())
    with pytest.raises(ValueError, match="out of range"):
        assemble(-1, script, (), make_cfg())


@pytest.mark.parametrize("observed_count", [0, 2])
def test_assemble_requires_exactly_turn_index_observed_turns(observed_count: int) -> None:
    """§5 test 10(b), on **both** sides of the equality. This precondition is the structural
    guarantee of §3.8.4's ruling: turn *n* cannot be assembled from anything but *n* observations,
    so there is no argument through which a textbook prefix could re-enter."""
    script = (turn(1, "t1"), turn(2, "t2"), turn(3, "t3"))
    observed = tuple(observed_turn() for _ in range(observed_count))
    with pytest.raises(ValueError, match="observed"):
        assemble(1, script, observed, make_cfg())


def test_assemble_accepts_the_matching_observed_count() -> None:
    """The control for the pair above: at `len(observed) == turn_index` it does not raise, so the
    two negatives are about the equality rather than about the argument existing."""
    script = (turn(1, "t1"), turn(2, "t2"))
    messages = assemble(1, script, (observed_turn(),), make_cfg())
    assert messages[-1] == {"role": "user", "content": "t2"}


# --------------------------------------------------------------------------------------------
# assemble — the replay contract's central ruling: observed output, never the script's `expect`
# --------------------------------------------------------------------------------------------


def test_assemble_replays_what_the_model_produced_and_never_the_scripts_expect() -> None:
    """§5 test 10(a) — the negative that would have caught the scripted-`expect` replay. The
    script's oracle says the model should have called `lookup_product_fact(name="Pad")` and replied
    with `24.99`; what the model actually did was call it with `"Different Item"` and reply
    `"It is 99.99."`. The assembled history must carry the observed strings and **none** of the
    oracle's."""
    script = (
        turn(
            1,
            "what does that one cost?",
            toolRequired=True,
            tool="lookup_product_fact",
            args={"name": "Pad"},
            finalReplyMustContain=["24.99"],
        ),
        turn(2, "anything else?", toolRequired=False, finalReplyMustContain=["no"]),
    )
    real_call = native_call("real", "lookup_product_fact", {"name": "Different Item"})
    observed = (
        observed_turn(
            chat_results=(
                chat_result(content=None, tool_calls=(real_call,)),
                chat_result(content="It is 99.99."),
            ),
            dispatches=(
                dispatch_record(
                    "lookup_product_fact", {"name": "Different Item"}, {"price": 99.99}
                ),
            ),
            final_reply="It is 99.99.",
        ),
    )
    serialized = json.dumps(assemble(1, script, observed, make_cfg(toolSchemas=())))
    assert "Different Item" in serialized
    assert "99.99" in serialized
    assert "Pad" not in serialized
    assert "24.99" not in serialized


def test_assemble_tolerates_the_plans_own_conversation_row_literal() -> None:
    """§3.8.4's own `conversations.jsonl` row example, transcribed verbatim — a check-input derived
    from the plan's literal, not from this implementation's idea of what a turn looks like. Under
    the v1.25 ruling the whole `expect` block (`argChecks` and `terminal` included) is data
    `assemble` must **tolerate and never read**, so this drives the row through a replay and
    asserts the oracle's own strings are absent from what is assembled."""
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
    script_turn = Turn(
        seq=row["turns"][0]["seq"],
        user=row["turns"][0]["user"],
        expect=row["turns"][0]["expect"],
    )
    script = (script_turn, turn(2, "thanks"))
    observed = (
        observed_turn(
            chat_results=(chat_result(content="Sorry, I could not find that."),),
            final_reply="Sorry, I could not find that.",
        ),
    )

    serialized = json.dumps(assemble(1, script, observed, make_cfg(toolSchemas=())))

    # The scripted *user* text is always the script's, at every mode (`-ml` §4.1).
    assert "What's the price of the Wireless Charging Pad?" in serialized
    assert "Sorry, I could not find that." in serialized
    # Nothing from the oracle: neither its reply fragment nor its argument value.
    assert "24.99" not in serialized
    assert "maxPrice" not in serialized


# --------------------------------------------------------------------------------------------
# assemble — order, the whole-list contract, and the schema block
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ALL_MODES)
def test_assemble_current_turn_user_message_is_always_last(mode: str) -> None:
    script = (turn(1, "first"), turn(2, "second"))
    observed = (tool_calling_prior_turn(),)
    messages = assemble(1, script, observed, make_cfg(historyReplay=mode))
    assert messages[-1] == {"role": "user", "content": "second"}


def test_assemble_structured_emits_the_documented_role_order_for_the_whole_list() -> None:
    """The order §4 S2 states — system · schema block · replayed history · current user — asserted
    as the **whole** sequence rather than by membership, which is what catches an `insert(0, ...)`
    or a block emitted in the wrong place."""
    script = (turn(1, "price of the Pad?"), turn(2, "anything else?"))
    observed = (tool_calling_prior_turn(),)
    messages = assemble(1, script, observed, make_cfg())
    assert [m["role"] for m in messages] == [
        "system",  # system prompt
        "system",  # tool-schema text block
        "user",  # replayed turn 1's scripted user text
        "assistant",  # iteration 1: the model's own message, tool_calls verbatim
        "tool",  # its one dispatched call's real return value
        "assistant",  # the turn's final reply (iteration 2's content, replayed once)
        "user",  # the current turn, always last
    ]


def test_assemble_none_mode_produces_no_history_messages_at_all() -> None:
    script = (turn(1, "first"), turn(2, "second"))
    observed = (tool_calling_prior_turn(reply="the first answer"),)
    messages = assemble(1, script, observed, make_cfg(historyReplay="none"))
    assert [m["role"] for m in messages] == ["system", "system", "user"]
    serialized = json.dumps(messages)
    assert "first" not in serialized
    assert "the first answer" not in serialized


def test_assemble_no_system_prompt_omits_the_system_message() -> None:
    script = (turn(1, "hi"),)
    cfg = make_cfg(systemPrompt=None, toolSchemas=())
    assert assemble(0, script, (), cfg) == [{"role": "user", "content": "hi"}]


def test_assemble_represent_tool_schemas_each_turn_false_drops_schemas_after_turn_1() -> None:
    """The literal §5 test 10 requirement: `representToolSchemasEachTurn=False` really does drop
    the tool-schema message after the first assembled turn."""
    script = (turn(1, "t1"), turn(2, "t2"), turn(3, "t3"))
    cfg = make_cfg(representToolSchemasEachTurn=False)
    observed = [observed_turn(), observed_turn()]

    def has_schema_message(messages: list[ChatMessage]) -> bool:
        return any(
            "lookup_product_fact" in json.dumps(m) for m in messages if m["role"] == "system"
        )

    assert has_schema_message(assemble(0, script, (), cfg)) is True
    assert has_schema_message(assemble(1, script, observed[:1], cfg)) is False
    assert has_schema_message(assemble(2, script, observed[:2], cfg)) is False


def test_assemble_represent_tool_schemas_each_turn_true_keeps_schemas_every_turn() -> None:
    """"Every turn" is checked at all three of this fixture's turns, matching the thoroughness of
    the `False` counterpart above (which checks all three of its own) — not just the first two, so
    the name's "every" is not a claim the assertions under-cover."""
    script = (turn(1, "t1"), turn(2, "t2"), turn(3, "t3"))
    cfg = make_cfg(representToolSchemasEachTurn=True)
    observed = [observed_turn(), observed_turn()]

    def has_schema_message(messages: list[ChatMessage]) -> bool:
        return any(
            "lookup_product_fact" in json.dumps(m) for m in messages if m["role"] == "system"
        )

    assert has_schema_message(assemble(0, script, (), cfg)) is True
    assert has_schema_message(assemble(1, script, observed[:1], cfg)) is True
    assert has_schema_message(assemble(2, script, observed[:2], cfg)) is True


@pytest.mark.parametrize("represent_each_turn", [True, False])
def test_assemble_no_tool_schemas_means_no_schema_message_regardless_of_the_flag(
    represent_each_turn: bool,
) -> None:
    script = (turn(1, "t1"),)
    cfg = make_cfg(toolSchemas=(), representToolSchemasEachTurn=represent_each_turn)
    assert [m["role"] for m in assemble(0, script, (), cfg)] == ["system", "user"]


# --------------------------------------------------------------------------------------------
# assemble — `historyTurns`, windowed over the (script, observed) **pair**
# --------------------------------------------------------------------------------------------


def _windowing_fixture() -> tuple[tuple[Turn, ...], tuple[TurnTrace, ...]]:
    """Four prior turns whose scripted user text and observed reply are *distinguishable per
    index* — `u1..u4` against `a1..a4`. That is what makes a drift between the two sequences
    visible: a window that took the last N of one and the last N of the other would pair `u3` with
    `a4` and still produce a perfectly plausible transcript."""
    script = tuple(turn(i, f"u{i}") for i in range(1, 6))
    observed = tuple(observed_turn(final_reply=f"a{i}") for i in range(1, 5))
    return script, observed


def test_assemble_history_turns_zero_replays_every_prior_turn() -> None:
    script, observed = _windowing_fixture()
    cfg = make_cfg(historyReplay="plaintext", historyTurns=0, toolSchemas=())
    flattened = assemble(4, script, observed, cfg)[-2]["content"]
    for i in range(1, 5):
        assert f"u{i}" in flattened
        assert f"a{i}" in flattened


@pytest.mark.parametrize(
    ("window", "expected_present", "expected_absent"),
    [
        (1, (4,), (1, 2, 3)),
        (2, (3, 4), (1, 2)),
    ],
)
def test_assemble_history_turns_windows_the_pair_to_only_the_last_n_prior_turns(
    window: int, expected_present: tuple[int, ...], expected_absent: tuple[int, ...]
) -> None:
    """Two window sizes, not just `N=1` — "the last N" is a claim about an arbitrary N, so a
    second value (`N=2`) is what confirms the windowing is a general slice rather than a
    special-cased "keep exactly one" branch that happens to pass at `N=1`.

    Both sequences are asserted at every index (Pass 15 §6): the window now spans two sequences,
    and applying it to one of them alone is silent."""
    script, observed = _windowing_fixture()
    cfg = make_cfg(historyReplay="plaintext", historyTurns=window, toolSchemas=())
    flattened = assemble(4, script, observed, cfg)[-2]["content"]
    for i in expected_present:
        assert f"u{i}" in flattened
        assert f"a{i}" in flattened
    for i in expected_absent:
        assert f"u{i}" not in flattened
        assert f"a{i}" not in flattened


def test_assemble_history_turns_pairs_each_replayed_user_with_its_own_observed_reply() -> None:
    """The drift the window makes possible, asserted directly rather than through absence: under
    `structured-replies-only` the replayed messages alternate `u3, a3, u4, a4` — never `u3, a4`."""
    script, observed = _windowing_fixture()
    cfg = make_cfg(historyReplay="structured-replies-only", historyTurns=2, toolSchemas=())
    messages = assemble(4, script, observed, cfg)
    replayed = [(m["role"], m["content"]) for m in messages[1:-1]]
    assert replayed == [
        ("user", "u3"),
        ("assistant", "a3"),
        ("user", "u4"),
        ("assistant", "a4"),
    ]


# --------------------------------------------------------------------------------------------
# assemble — per-mode message shapes (§5 test 10 cases (c), (d), (e))
# --------------------------------------------------------------------------------------------


def test_assemble_structured_replays_an_iterations_tool_calls_and_its_return_value() -> None:
    """`structured` replays the whole real exchange: per in-turn iteration the model's own
    assistant message with its `tool_calls` **verbatim, ids included**, then one `tool` message per
    call carrying the environment's real `DispatchRecord.returnValue`.

    One tool-calling iteration, which is what this fixture has — the *across*-iterations half of
    the claim (each iteration getting its **own** return value) is the sibling test below, on the
    only fixture shape in which it is visible."""
    script = (turn(1, "price of the Pad?"), turn(2, "anything else?"))
    observed = (tool_calling_prior_turn(),)
    messages = assemble(1, script, observed, make_cfg(toolSchemas=()))

    assistant = messages[2]
    call = assistant["tool_calls"][0]
    assert assistant["content"] is None
    assert call == native_call("c1", "lookup_product_fact", {"name": "Pad"})
    tool_msg = messages[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"
    assert tool_msg["name"] == "lookup_product_fact"
    assert json.loads(tool_msg["content"]) == {"price": 24.99}
    assert messages[-2] == {"role": "assistant", "content": "That one is 24.99."}


def test_assemble_structured_replays_every_iteration_with_its_own_return_value() -> None:
    """`_replay_structured` threads **one** dispatch cursor across a turn's iterations, and a turn
    with **two** tool-calling iterations is the only shape in which that threading is observable:
    on a one-iteration turn a cursor reset per iteration is indistinguishable from the real thing,
    and every other `structured` fixture in this file has exactly one.

    Under a reset, iteration 2's `tool` message carries iteration 1's return value — *"a plausible
    transcript nobody would read as wrong"*, which is `TraceContractViolated`'s own phrase for the
    failure it closes on the *dispatch* side, arriving here on the *replay* side instead. So the
    two return values are deliberately distinct: with equal ones the pairing assertion is vacuous.
    """
    call_1 = native_call("c1", "lookup_product_fact", {"name": "Pad"})
    call_2 = native_call("c2", "lookup_product_fact", {"name": "Case"})
    reply = "The Pad is 24.99 and the Case is 9.99."
    script = (turn(1, "price of the Pad and the Case?"), turn(2, "anything else?"))
    observed = (
        observed_turn(
            chat_results=(
                chat_result(content=None, tool_calls=(call_1,)),
                chat_result(content=None, tool_calls=(call_2,)),
                chat_result(content=reply),
            ),
            dispatches=(
                dispatch_record("lookup_product_fact", {"name": "Pad"}, {"price": 24.99}),
                dispatch_record("lookup_product_fact", {"name": "Case"}, {"price": 9.99}),
            ),
            final_reply=reply,
        ),
    )
    messages = assemble(1, script, observed, make_cfg(toolSchemas=()))

    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert [m["tool_calls"] for m in (messages[2], messages[4])] == [[call_1], [call_2]]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
    assert [json.loads(m["content"]) for m in tool_messages] == [
        {"price": 24.99},
        {"price": 9.99},
    ]


@pytest.mark.parametrize(
    ("bad_call", "reason"),
    [
        ({"id": "x", "type": "function", "function": {"arguments": "{}"}}, "missing-function-name"),
        (
            {"id": "y", "type": "function", "function": {"name": "view_cart", "arguments": "{{"}},
            "unparseable-arguments",
        ),
    ],
)
def test_assemble_structured_emits_one_tool_message_per_call_including_undispatchable_ones(
    bad_call: Mapping[str, Any], reason: str
) -> None:
    """§5 test 10(c) — **every** `tool_calls` entry gets exactly one `tool` message, without
    exception, ids matching pairwise. An unanswered tool call is rejected by OpenAI-shaped servers,
    so omitting one would convert a model failure into a *transport* failure at the next turn. For
    an entry that could not be dispatched the content is a JSON object naming the failure."""
    good_call = native_call("good", "lookup_product_fact", {"name": "Pad"})
    script = (turn(1, "t1"), turn(2, "t2"))
    observed = (
        observed_turn(
            chat_results=(
                chat_result(content=None, tool_calls=(good_call, bad_call)),
                chat_result(content="done"),
            ),
            dispatches=(
                dispatch_record("lookup_product_fact", {"name": "Pad"}, {"price": 24.99}),
            ),
            final_reply="done",
        ),
    )
    messages = assemble(1, script, observed, make_cfg(toolSchemas=()))

    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert [m["tool_call_id"] for m in tool_messages] == ["good", bad_call["id"]]
    assert json.loads(tool_messages[0]["content"]) == {"price": 24.99}
    assert json.loads(tool_messages[1]["content"])["reason"] == reason


def test_assemble_names_a_dispatchable_call_with_no_record_instead_of_reusing_one() -> None:
    """`_iteration_exchange`'s third undispatchable reason, and the only one whose cause is the
    *record* rather than the call: an iteration emitting two dispatchable calls whose turn slice
    holds one `DispatchRecord`.

    `drive` cannot produce that record any more — the per-call trace check refuses such an
    environment mid-run — but `assemble` replays a `TurnTrace` it is *handed*, and one rebuilt by a
    later unit or read back off disk carries no such guarantee. The entry still owes a `tool`
    message, exactly like the other two reasons, and nothing here may invent a return value for it:
    reusing the previous record's is the plausible wrong answer and is the substitution this pins.
    """
    call_1 = native_call("c1", "lookup_product_fact", {"name": "Pad"})
    call_2 = native_call("c2", "lookup_product_fact", {"name": "Case"})
    script = (turn(1, "price of both?"), turn(2, "anything else?"))
    observed = (
        observed_turn(
            chat_results=(
                chat_result(content=None, tool_calls=(call_1, call_2)),
                chat_result(content="done"),
            ),
            dispatches=(dispatch_record("lookup_product_fact", {"name": "Pad"}, {"price": 24.99}),),
            final_reply="done",
        ),
    )
    messages = assemble(1, script, observed, make_cfg(toolSchemas=()))

    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
    assert json.loads(tool_messages[0]["content"]) == {"price": 24.99}
    assert json.loads(tool_messages[1]["content"]) == {
        "error": "tool-call-not-dispatched",
        "reason": "no-dispatch-record",
    }


@pytest.mark.parametrize("mode", REPLY_TEXT_MODES)
def test_assemble_reply_text_modes_show_no_tool_evidence_for_a_tool_calling_prior_turn(
    mode: str,
) -> None:
    """§5 test 10(d) — the tool-evidence axis §3.3's table declares, asserted as an **absence** on
    a fixture whose prior turn really did dispatch a tool, so the absence is a fact about the mode
    and not about the fixture."""
    script = (turn(1, "price of the Pad?"), turn(2, "anything else?"))
    observed = (tool_calling_prior_turn(),)
    messages = assemble(1, script, observed, make_cfg(historyReplay=mode, toolSchemas=()))

    assert not any(m["role"] == "tool" for m in messages)
    assert not any("tool_calls" in m for m in messages)
    serialized = json.dumps(messages)
    assert "lookup_product_fact" not in serialized
    assert "That one is 24.99." in serialized  # the reply text itself is still replayed


def test_the_two_reply_text_modes_differ_on_role_ownership_alone() -> None:
    """§3.3 (P13-8) rules `historyReplay`'s four values across **two** axes — role ownership and
    tool evidence — not one ladder, and rules that `plaintext` and `structured-replies-only` differ
    on **ownership alone**: both carry the prior turns' text, neither carries tool evidence, but
    `structured-replies-only` gives each prior reply its own `assistant` message while `plaintext`
    quotes the whole transcript inside somebody else's `user` message. §6 R-3's bisect inference
    turns on exactly that difference, so ownership is the canonical axis and not a rendering
    detail.

    Nothing pinned it (impl review Pass 17, P17-7):
    `test_every_history_replay_mode_renders_a_distinct_message_list` separates the four modes by
    JSON inequality, which is blind to *which* role differs — retagging `plaintext`'s flattened
    message `system` keeps all four renderings distinct and leaves the suite green (measured)."""
    script = (turn(1, "u1"), turn(2, "u2"), turn(3, "u3"))
    observed = (observed_turn(final_reply="a1"), observed_turn(final_reply="a2"))
    flattened = assemble(2, script, observed, make_cfg(historyReplay="plaintext", toolSchemas=()))
    native = assemble(
        2, script, observed, make_cfg(historyReplay="structured-replies-only", toolSchemas=())
    )

    # `plaintext`: the whole history is one `user`-owned quotation, never the model's own voice.
    assert [m["role"] for m in flattened] == ["system", "user", "user"]
    assert "a1" in flattened[-2]["content"] and "a2" in flattened[-2]["content"]
    # `structured-replies-only`: the same two replies, owned by `assistant`.
    assert [m["role"] for m in native] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert [m["content"] for m in native[1:-1]] == ["u1", "a1", "u2", "a2"]


def test_assemble_structured_replies_only_emits_exactly_one_assistant_per_prior_turn() -> None:
    """The fourth value's own shape (§4 S2's replay contract): `{"role": "user", ...}` then
    **exactly one** `{"role": "assistant", "content": <that turn's observed final reply text>}` —
    not one per iteration, which is what a copy of `structured`'s loop would produce."""
    script = (turn(1, "price of the Pad?"), turn(2, "anything else?"))
    observed = (tool_calling_prior_turn(),)  # two iterations, one dispatch
    messages = assemble(1, script, observed, make_cfg(historyReplay="structured-replies-only",
                                                      toolSchemas=()))
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2] == {"role": "assistant", "content": "That one is 24.99."}


@pytest.mark.parametrize(
    "disposition", ["cap-hit", "timed-out", "no-response", "server-rejected"]
)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_assemble_replays_a_reply_less_prior_turn_in_every_mode(
    mode: str, disposition: str
) -> None:
    """§5 test 10(e) — a prior turn with `turnDisposition != "replied"` is replayed and never
    omitted, in every mode: the assembled list never has one fewer turn than the observed prefix.
    Omitting it would shorten the visible history, and history length is the covariate this whole
    pack measures against."""
    script = (turn(1, "u1"), turn(2, "u2"), turn(3, "u3"))
    call = native_call("c1", "view_cart", {})
    reply_less = observed_turn(
        chat_results=(chat_result(content=None, tool_calls=(call,)),),
        dispatches=(dispatch_record("view_cart", {}, {"items": []}),),
        disposition=disposition,
    )
    observed = (observed_turn(final_reply="a1"), reply_less)
    cfg = make_cfg(historyReplay=mode, toolSchemas=())

    messages = assemble(2, script, observed, cfg)
    serialized = json.dumps(messages)

    if mode == "none":
        assert [m["role"] for m in messages] == ["system", "user"]
        return
    # Both prior turns' scripted user text is present — the reply-less one has not been dropped.
    assert "u1" in serialized
    assert "u2" in serialized


@pytest.mark.parametrize("mode", REPLY_TEXT_MODES)
def test_assemble_reply_less_prior_turn_contributes_captured_and_empty_text(mode: str) -> None:
    """§5 test 10(e), the reply-text half: `content: ""` — *captured and empty*, which is what the
    conversation contained — and never the script's `expect`, which for a turn with no reply is
    exactly where the repair looks natural and is wrong."""
    script = (
        turn(1, "u1", toolRequired=False, finalReplyMustContain=["THE ORACLE STRING"]),
        turn(2, "u2"),
    )
    observed = (observed_turn(chat_results=(chat_result(content=None),), disposition="timed-out"),)
    messages = assemble(1, script, observed, make_cfg(historyReplay=mode, toolSchemas=()))
    serialized = json.dumps(messages)

    assert "THE ORACLE STRING" not in serialized
    if mode == "structured-replies-only":
        assert messages[1] == {"role": "user", "content": "u1"}
        assert messages[2] == {"role": "assistant", "content": ""}
    else:
        assert "User: u1\nAssistant: " in messages[1]["content"]


def test_assemble_structured_reply_less_prior_turn_has_no_trailing_assistant_message() -> None:
    """§5 test 10(e), the `structured` half (v1.26, P13-9): such a turn contributes **its
    iterations** and no trailing assistant message, so the prefix ends on a `tool` message
    immediately before the next `user` turn."""
    script = (turn(1, "u1"), turn(2, "u2"))
    call = native_call("c1", "view_cart", {})
    observed = (
        observed_turn(
            chat_results=(chat_result(content=None, tool_calls=(call,)),),
            dispatches=(dispatch_record("view_cart", {}, {"items": []}),),
            disposition="cap-hit",
        ),
    )
    messages = assemble(1, script, observed, make_cfg(toolSchemas=()))
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]


# --------------------------------------------------------------------------------------------
# drive — the bounded per-turn iteration loop (§5 test 10b)
# --------------------------------------------------------------------------------------------


def test_drive_returns_a_conversation_trace_carrying_the_scripts_full_identity() -> None:
    """`scriptId`, `shape` and `replicate` all three: §3.3's `pairingKey` is
    `["scriptId", "replicate", "turnIndex"]` and `shape` is §3.8.4's reporting stratum, so a trace
    missing either is one the pairing key and the stratum cannot be derived from (P15-8)."""
    script = conversation("S-42", (turn(1, "t1"),), shape="B", replicate=2)
    trace = drive(StubEnvironment(), script, stub_llm([chat_result()]), make_cfg(toolSchemas=()))
    assert isinstance(trace, ConversationTrace)
    assert (trace.scriptId, trace.shape, trace.replicate) == ("S-42", "B", 2)


def test_drive_loops_within_a_turn_until_a_response_carries_no_tool_calls() -> None:
    """§5 test 10b's first case: tool calls twice, then text — `iterations == 3`, disposition
    `"replied"`, and the **third** response's text as `finalReplyText`."""
    script = conversation("S-01", (turn(1, "price of the Pad?"),))
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(native_call("c1", "lookup_product_fact", {}),)),
            chat_result(content=None, tool_calls=(native_call("c2", "view_cart", {}),)),
            chat_result(content="Here is the answer."),
        ]
    )
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    turn_trace = trace.turns[0]
    assert len(llm.calls) == 3
    assert turn_trace.iterations == 3
    assert turn_trace.turnDisposition == "replied"
    assert turn_trace.finalReplyText == "Here is the answer."


def test_drive_stops_at_the_cap_when_the_model_never_stops_emitting_tool_calls() -> None:
    """§5 test 10b's second case: `iterations == cfg.maxIterationsPerTurn`, `"cap-hit"`, and
    `finalReplyText is None`."""
    cfg = make_cfg(toolSchemas=(), maxIterationsPerTurn=4)
    script = conversation("S-02", (turn(1, "loop forever"),))
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(native_call(f"c{i}", "view_cart", {}),))
            for i in range(10)
        ]
    )
    turn_trace = drive(StubEnvironment(), script, llm, cfg).turns[0]
    assert len(llm.calls) == 4
    assert turn_trace.iterations == 4
    assert turn_trace.turnDisposition == "cap-hit"
    assert turn_trace.finalReplyText is None


def test_drive_records_a_null_content_termination_as_captured_and_empty() -> None:
    """§5 test 10b's third case: a response that terminates the loop with `content: None` records
    `"replied"` and `finalReplyText == ""`, **not `None`** — *captured and empty* is a different
    fact from *no reply at all*, and `None` there would collapse them."""
    script = conversation("S-03", (turn(1, "hi"),))
    turn_trace = drive(
        StubEnvironment(), script, stub_llm([chat_result(content=None)]), make_cfg(toolSchemas=())
    ).turns[0]
    assert turn_trace.turnDisposition == "replied"
    assert turn_trace.finalReplyText == ""


@pytest.mark.parametrize("cap", [None, 0, -1])
def test_drive_refuses_an_unusable_iteration_cap_before_any_llm_call(cap: int | None) -> None:
    """`maxIterationsPerTurn` is pack data with **no default** (§3.3) and `drive` is its only
    consumer, so it raises rather than substituting a value. A non-positive cap is refused on the
    same principle one step along: §3.8.4's five-member disposition set has no member for *the cap
    forbade the first call*, so a silently undefined record is the alternative."""
    script = conversation("S-04", (turn(1, "hi"),))
    llm = stub_llm([chat_result()])
    with pytest.raises(ValueError, match="maxIterationsPerTurn"):
        drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=(), maxIterationsPerTurn=cap))
    assert llm.calls == []


def test_drive_raises_type_error_before_any_llm_call_when_env_is_not_a_tool_environment() -> None:
    """Pins the fix for the defect `tooling.py`'s own docstring named: `ToolEnvironment` is
    `@runtime_checkable` *so that* `drive` can assert conformance — a claim that must be backed by
    `drive` actually calling `isinstance`, not merely by the Protocol supporting the call.
    "Before any LLM call" is checked via `llm.calls` staying empty, not just via the raise."""

    class _NotAnEnvironment:
        pass

    script = conversation("S-05", (turn(1, "hi"),))
    llm = stub_llm([chat_result()])
    with pytest.raises(TypeError, match="ToolEnvironment"):
        drive(_NotAnEnvironment(), script, llm, make_cfg(toolSchemas=()))
    assert llm.calls == []


def test_drive_iterations_equals_the_number_of_completed_chat_results() -> None:
    """`iterations == len(chatResults)` — one number, asserted rather than assumed, because two
    independently maintained counts of the same thing is how several of this component's defects
    started (§3.8.4, P14-6; `-ml` §11.4 binds the third, `callCount`, at the runner). Checked on a
    turn that **completed** and on one that raised on its third call, where the two would differ if
    the count were maintained separately."""
    script = conversation("S-06", (turn(1, "t1"), turn(2, "t2")))
    tool_call = native_call("c", "view_cart", {})
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(tool_call,)),
            chat_result(content="done"),
            chat_result(content=None, tool_calls=(tool_call,)),
            chat_result(content=None, tool_calls=(tool_call,)),
            LMStudioCallFailed("boom", status=None),
        ]
    )
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    for turn_trace in trace.turns:
        assert turn_trace.iterations == len(turn_trace.chatResults)
    assert trace.turns[0].iterations == 2
    assert trace.turns[1].iterations == 2  # the third call raised and is not an iteration


def test_drive_turn_wall_clock_brackets_the_whole_turn_assembly_included(monkeypatch) -> None:
    """`wallClockMs` brackets the **whole turn** — the message assembly, every iteration, and the
    tool dispatches between them — so it is strictly greater than any one of that turn's
    `ChatResult.wallClockMs` and at least their sum. Redefining it as one call of the turn would
    make a model that loops eight times report a *smaller* latency than one that answers in a
    single call (§3.8.4).

    Driven against a **stub clock** the stubs advance themselves, so the figures are exact rather
    than timing-dependent — and, deliberately, the three per-call figures are **distinct and
    non-zero**: with equal or zero ones the plan's own `>=` assertion is satisfied by the last
    call's figure too, and the substitution it exists to catch ships green (measured: it does).

    **`assemble` is on the clock too, and it is a third of what §5 test 10b measures** — that
    test's own words are *"the difference being the harness's own dispatch **and message
    assembly**"* (impl review Pass 17, P17-8). Modelling the calls and the dispatches but not the
    assembly left the exact `==` blind to the stopwatch starting one statement too late, since a
    fake clock does not advance on its own during a function that never touches it."""
    clock = _FakeClock()
    monkeypatch.setattr(convo, "time", SimpleNamespace(monotonic=clock))
    assemble_ms = 3.0
    real_assemble = convo.assemble

    def ticking_assemble(*args: Any, **kwargs: Any):
        clock.advance_ms(assemble_ms)
        return real_assemble(*args, **kwargs)

    monkeypatch.setattr(convo, "assemble", ticking_assemble)

    script = conversation("S-07", (turn(1, "t1"),))
    tool_call = native_call("c", "view_cart", {})
    per_call_ms = (11.0, 22.0, 33.0)
    responses = [
        chat_result(content=None, tool_calls=(tool_call,), wall_clock_ms=per_call_ms[0]),
        chat_result(content=None, tool_calls=(tool_call,), wall_clock_ms=per_call_ms[1]),
        chat_result(content="done", wall_clock_ms=per_call_ms[2]),
    ]
    inner = stub_llm(responses)

    def ticking_llm(messages, **kwargs):
        result = inner(messages, **kwargs)
        clock.advance_ms(result.wallClockMs or 0.0)
        return result

    env = _TickingEnvironment(clock, dispatch_ms=5.0)
    turn_trace = drive(env, script, ticking_llm, make_cfg(toolSchemas=())).turns[0]

    calls_total = sum(per_call_ms)
    assert turn_trace.wallClockMs == pytest.approx(assemble_ms + calls_total + 2 * 5.0)
    assert turn_trace.wallClockMs >= calls_total
    # And strictly above any single call's figure, which is what the substitution would report.
    assert turn_trace.wallClockMs > max(c.wallClockMs or 0.0 for c in turn_trace.chatResults)


def test_drive_records_a_wall_clock_figure_on_every_turn_including_one_that_raised() -> None:
    """Checked on every turn of a 3-turn script, not just the first — "per turn" names a property
    of each turn's own record, not only turn 0's.

    Turn 2's **first call raises**, which is the one shape a reader might expect to withhold a
    figure and is why `wallClockMs` is typed `float` rather than `float | None`: the turn still
    took time and still records it, and §4 S2 puts the withholding on `ItemTiming.withheldFor`,
    which is the runner's (impl review Pass 17, P17-9). Without a raised turn in the fixture the
    claim is checked three times on the one disposition that could never have been `None`."""
    script = conversation("S-08", (turn(1, "t1"), turn(2, "t2"), turn(3, "t3")))
    llm = stub_llm([chat_result(), LMStudioCallTimeout("slow"), chat_result()])
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))

    assert [t.iterations for t in trace.turns] == [1, 0, 1]
    assert [t.turnDisposition for t in trace.turns] == ["replied", "timed-out", "replied"]
    for turn_trace in trace.turns:
        assert isinstance(turn_trace.wallClockMs, float)
        assert turn_trace.wallClockMs >= 0.0


def test_drive_messages_sent_is_the_working_list_of_the_turns_last_call() -> None:
    """`messagesSent` is the stored record of the context the model saw, and the ruling this rework
    implements is entirely about what is in that list — a run whose `messagesSent` did not describe
    its own prompt could not be audited for it (P15-7). Under the loop the field holds the in-turn
    working list **as sent at the turn's last model call**, which has iteration 1's list as a
    prefix; both halves are asserted here so the choice is pinned rather than documented."""
    script = conversation("S-09", (turn(1, "t1"),))
    tool_call = native_call("c1", "view_cart", {})
    llm = stub_llm(
        [chat_result(content=None, tool_calls=(tool_call,)), chat_result(content="done")]
    )
    turn_trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=())).turns[0]
    assert turn_trace.messagesSent == tuple(llm.calls[-1]["messages"])
    assert list(turn_trace.messagesSent)[: len(llm.calls[0]["messages"])] == list(
        llm.calls[0]["messages"]
    )


def test_drive_messages_sent_on_a_turn_whose_first_call_raised_is_the_assembled_list() -> None:
    """The same definition, at the disposition where the "last call" is the only call: the field is
    exactly `assemble`'s output, so it is total over all five mechanisms rather than undefined on
    four of them."""
    script = conversation("S-10", (turn(1, "t1"),))
    llm = stub_llm([LMStudioCallTimeout("slow")])
    cfg = make_cfg(toolSchemas=())
    turn_trace = drive(StubEnvironment(), script, llm, cfg).turns[0]
    assert turn_trace.messagesSent == tuple(assemble(0, script.turns, (), cfg))


def test_drive_feeds_each_iterations_tool_results_back_into_the_next_call() -> None:
    """The loop's whole point: iteration 2 sees iteration 1's assistant message and the
    environment's **real** return value, so the model can state a final reply grounded in what the
    tool actually returned (`-ml` §4.2(g))."""
    script = conversation("S-11", (turn(1, "price of the Pad?"),))
    call = native_call("c1", "lookup_product_fact", {"name": "Pad"})
    llm = stub_llm([chat_result(content=None, tool_calls=(call,)), chat_result(content="24.99")])
    env = StubEnvironment(results={"lookup_product_fact": {"price": 24.99}})
    drive(env, script, llm, make_cfg(toolSchemas=()))

    second_call_messages = llm.calls[1]["messages"]
    assert second_call_messages[:-2] == llm.calls[0]["messages"]
    assert second_call_messages[-2]["tool_calls"] == [call]
    assert second_call_messages[-1]["role"] == "tool"
    assert json.loads(second_call_messages[-1]["content"]) == {"price": 24.99}


def test_drive_replays_a_prior_turn_from_this_runs_own_output_never_from_the_script() -> None:
    """The inverse of the test this file used to carry. Turn 1's *scripted* `expect` names
    `lookup_product_fact("Pad")` and a `24.99` reply; the model actually calls it with
    `"Different Item"` and replies `"It is 99.99."`. Turn 2's assembled messages must carry the
    model's own strings and none of the oracle's (§3.8.4's ruling)."""
    script = conversation(
        "S-12",
        (
            turn(
                1,
                "what does that one cost?",
                toolRequired=True,
                tool="lookup_product_fact",
                args={"name": "Pad"},
                finalReplyMustContain=["24.99"],
            ),
            turn(2, "anything else?"),
        ),
    )
    real_call = native_call("real", "lookup_product_fact", {"name": "Different Item"})
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(real_call,)),
            chat_result(content="It is 99.99."),
            chat_result(content="Nothing else."),
        ]
    )
    drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))

    serialized = json.dumps(llm.calls[-1]["messages"])
    assert "Different Item" in serialized
    assert "It is 99.99." in serialized
    assert "Pad" not in serialized
    assert "24.99" not in serialized


# --------------------------------------------------------------------------------------------
# drive — dispatching (FR-10's ground truth)
# --------------------------------------------------------------------------------------------


def test_drive_dispatches_a_native_tool_call_with_parsed_arguments() -> None:
    script = conversation("S-13", (turn(1, "price of Pad?"),))
    call = native_call("call_1", "lookup_product_fact", {"name": "Pad"})
    llm = stub_llm([chat_result(content=None, tool_calls=(call,)), chat_result(content="ok")])
    env = StubEnvironment(results={"lookup_product_fact": {"price": 24.99}})
    dispatches = drive(env, script, llm, make_cfg(toolSchemas=())).turns[0].dispatches
    assert len(dispatches) == 1
    assert dispatches[0].name == "lookup_product_fact"
    assert dispatches[0].parsedArguments == {"name": "Pad"}
    assert dispatches[0].returnValue == {"price": 24.99}


def test_drive_turn_without_a_tool_call_dispatches_nothing() -> None:
    script = conversation("S-14", (turn(1, "hi"),))
    llm = stub_llm([chat_result(content="hello!")])
    assert drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=())).turns[0].dispatches == ()


@pytest.mark.parametrize("call_count", [0, 1, 2, 5])
def test_drive_dispatches_every_dispatchable_call_in_a_turn_in_emission_order(
    call_count: int,
) -> None:
    """A coverage probe over the fan-out axis rather than one two-call test (P15-6): `-ml` §4.2(e)
    needs `duplicate_turn_rate`'s within-turn variant and `spurious_turn_rate`, both over
    `|E(t)| >= 1`, and neither is scoreable if only the first call of a turn reaches the trace."""
    calls = tuple(
        native_call(f"c{i}", "add_to_cart", {"name": f"item{i}"}) for i in range(call_count)
    )
    script = conversation("S-15", (turn(1, "add them all"),))
    responses: list[Any] = []
    if calls:
        responses.append(chat_result(content=None, tool_calls=calls))
    responses.append(chat_result(content="done"))
    turn_trace = drive(
        StubEnvironment(), script, stub_llm(responses), make_cfg(toolSchemas=())
    ).turns[0]
    assert len(turn_trace.dispatches) == call_count
    assert [d.parsedArguments["name"] for d in turn_trace.dispatches] == [
        f"item{i}" for i in range(call_count)
    ]


def test_drive_isolates_each_turns_dispatches_from_every_other_turn() -> None:
    script = conversation("S-16", (turn(1, "price of Pad?"), turn(2, "price of Mug?")))
    calls = [
        native_call("c1", "lookup_product_fact", {"name": "Pad"}),
        native_call("c2", "lookup_product_fact", {"name": "Mug"}),
    ]
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(calls[0],)),
            chat_result(content="a"),
            chat_result(content=None, tool_calls=(calls[1],)),
            chat_result(content="b"),
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


def test_drive_env_state_is_captured_after_the_turns_last_iterations_dispatches() -> None:
    """Under the loop a turn has several rounds of dispatches, and `envState` is read after the
    **last** one — a read taken after iteration 1 would record turn 1's state as `callCount == 1`
    where the turn really made two calls."""
    script = conversation("S-17", (turn(1, "add two things"), turn(2, "add one more")))
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(native_call("c1", "add_to_cart", {"n": 1}),)),
            chat_result(content=None, tool_calls=(native_call("c2", "add_to_cart", {"n": 2}),)),
            chat_result(content="done"),
            chat_result(content=None, tool_calls=(native_call("c3", "add_to_cart", {"n": 3}),)),
            chat_result(content="done"),
        ]
    )
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    assert trace.turns[0].envState["callCount"] == 2
    assert trace.turns[1].envState["callCount"] == 3


def test_drive_passes_native_tools_param_every_turn_even_when_schemas_not_restated() -> None:
    """The design decision documented in `convo.py`'s module docstring:
    `representToolSchemasEachTurn=False` withholds the *textual* schema block `assemble`
    builds, never the native `tools=`
    parameter `llm` receives — otherwise native tool calls would be structurally impossible from
    turn 2 onward on every pack that sets the flag `False`."""
    script = conversation("S-18", (turn(1, "t1"), turn(2, "t2")))
    llm = stub_llm([chat_result(), chat_result()])
    cfg = make_cfg(representToolSchemasEachTurn=False)
    drive(StubEnvironment(), script, llm, cfg)
    assert llm.calls[0]["tools"] == [dict(s) for s in cfg.toolSchemas]
    assert llm.calls[1]["tools"] == [dict(s) for s in cfg.toolSchemas]


def test_drive_malformed_tool_call_missing_a_function_name_is_skipped_not_raised() -> None:
    script = conversation("S-19", (turn(1, "hi"),))
    broken_call = {"id": "x", "type": "function", "function": {"arguments": "{}"}}
    llm = stub_llm(
        [chat_result(content=None, tool_calls=(broken_call,)), chat_result(content="done")]
    )
    assert drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=())).turns[0].dispatches == ()


@pytest.mark.parametrize("raw_arguments", ["not json at all", "[1, 2]", '"a string"', "17"])
def test_drive_never_dispatches_a_call_whose_arguments_are_not_a_json_object(
    raw_arguments: str,
) -> None:
    """P15-4's fix. Degrading every unreadable `arguments` value to `{}` dispatched a call the
    model never validly made — mutating FR-10 ground-truth state, and under the loop feeding that
    dispatch's return value back to the model as a `tool` message, so a *harness-side* parse
    failure shaped the model's next iteration. `{}` now has one meaning: the model sent one."""
    call = {
        "id": "x",
        "type": "function",
        "function": {"name": "add_to_cart", "arguments": raw_arguments},
    }
    script = conversation("S-20", (turn(1, "add it"),))
    llm = stub_llm([chat_result(content=None, tool_calls=(call,)), chat_result(content="done")])
    env = StubEnvironment()
    turn_trace = drive(env, script, llm, make_cfg(toolSchemas=())).turns[0]
    assert turn_trace.dispatches == ()
    assert env.trace() == []
    # The call still owes a `tool` message on the next iteration, naming the failure.
    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0]["content"])["reason"] == "unparseable-arguments"


def test_drive_does_dispatch_a_genuinely_empty_arguments_object() -> None:
    """The control for the parametrized refusal above: `"{}"` on the wire is a real, well-formed
    call with no arguments and is dispatched. Without this the refusal could be "never dispatch
    anything" and still pass."""
    call = native_call("c1", "view_cart", {})
    script = conversation("S-21", (turn(1, "show my cart"),))
    llm = stub_llm([chat_result(content=None, tool_calls=(call,)), chat_result(content="done")])
    turn_trace = drive(
        StubEnvironment(), script, llm, make_cfg(toolSchemas=())
    ).turns[0]
    assert [d.name for d in turn_trace.dispatches] == ["view_cart"]
    assert turn_trace.dispatches[0].parsedArguments == {}


def test_drive_refuses_an_environment_that_loses_a_dispatch_it_executed() -> None:
    """P15-5. A `ToolEnvironment` that clears its own trace inside `dispatch` makes `drive`'s
    trace diff yield an **empty** slice, so a real, state-mutating `add_to_cart` is recorded as
    zero dispatches — FR-10 ground truth silently lost in the module whose whole thesis is that
    the trace and the state *are* the ground truth. This raises mid-run and can abort a
    conversation, which is right: it is a **pack** defect, and pack defects fail closed (§3.3),
    unlike the model failures `-ml` §4.1 requires to be recorded and driven past.

    Note which check fires and why the finding's literal one would not: on the conversation's very
    first call nothing had been reported yet, so no prefix can have been dropped and no length can
    have shrunk. What is wrong is that the environment executed one call and reported none."""
    script = conversation("S-22", (turn(1, "add it"),))
    call = native_call("c1", "add_to_cart", {"name": "Pad"})
    llm = stub_llm([chat_result(content=None, tool_calls=(call,)), chat_result(content="done")])
    env = ResettingEnvironment()
    with pytest.raises(TraceContractViolated, match="grew by 0 entries"):
        drive(env, script, llm, make_cfg(toolSchemas=()))
    # The call really did run and really did mutate state — which is exactly why losing it is
    # a defect worth failing closed on rather than a bookkeeping nicety.
    assert env.state()["callCount"] == 1


def test_drive_refuses_an_environment_that_drops_an_entry_it_had_already_reported() -> None:
    """The other half of the contract: `trace()` must *"never drop or reorder an earlier entry"*,
    and a turn's dispatch slice is the tail beyond a prefix `drive` read before dispatching. An
    environment that rewrites that prefix makes the slice describe some other turn's calls."""

    class _RewritingEnvironment(StubEnvironment):
        def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
            return_value = super().dispatch(name, arguments)
            if len(self._trace) > 1:
                del self._trace[0]
            return return_value

    script = conversation("S-22b", (turn(1, "add one"), turn(2, "add another")))
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(native_call("c1", "add_to_cart", {"n": 1}),)),
            chat_result(content="ok"),
            chat_result(content=None, tool_calls=(native_call("c2", "add_to_cart", {"n": 2}),)),
            chat_result(content="ok"),
        ]
    )
    with pytest.raises(TraceContractViolated, match="no longer starts with"):
        drive(_RewritingEnvironment(), script, llm, make_cfg(toolSchemas=()))


def test_drive_refuses_an_environment_whose_per_call_record_count_is_skewed() -> None:
    """The contract `tooling.py` states and says `drive` **enforces** is *"exactly one
    `DispatchRecord` per call"* — per **call**, which is also the granularity `assemble`'s
    positional pairing is read at. A per-*iteration* aggregate is a weaker claim than the prose:
    an environment recording 0 entries for one call and 2 for the next balances it exactly, and the
    turn completes clean with `add_to_cart`'s replayed `tool` message carrying `view_cart`'s return
    value (measured: it did).

    The refusal names the offending **call**, not just the iteration, which is what distinguishes
    this check from the aggregate one it replaces."""
    script = conversation("S-22d", (turn(1, "add it and show me the cart"),))
    calls = (
        native_call("c1", "add_to_cart", {"name": "Pad"}),
        native_call("c2", "view_cart", {}),
    )
    llm = stub_llm([chat_result(content=None, tool_calls=calls), chat_result(content="done")])
    env = SkewEnvironment()

    with pytest.raises(
        TraceContractViolated, match=r"grew by 0 entries.*dispatch of 'add_to_cart'"
    ):
        drive(env, script, llm, make_cfg(toolSchemas=()))
    # It really ran, which is why a lost record is a defect and not a bookkeeping nicety.
    assert env.state()["lastCall_add_to_cart"] == {"name": "Pad"}


def test_drive_refuses_an_environment_that_loses_an_entry_between_two_calls() -> None:
    """The per-**iteration** re-take, on the only path that reaches it now that the per-call check
    holds: an environment that mutates itself on `trace()` *read* moves the record between the
    reads a per-call check brackets, so that check sees a balanced 1-for-1 pair and passes.

    The refusal names the **iteration** and not a call, which is what shows it was this layer and
    not the per-call one that fired."""
    script = conversation("S-22e", (turn(1, "show me the cart"),))
    call = native_call("c1", "view_cart", {})
    llm = stub_llm([chat_result(content=None, tool_calls=(call,)), chat_result(content="done")])

    with pytest.raises(
        TraceContractViolated, match=r"grew by 0 entries across turn 0's iteration 1, which"
    ):
        drive(ReadMutatingEnvironment(), script, llm, make_cfg(toolSchemas=()))


def test_drive_refuses_a_rewritten_trace_on_a_turn_that_dispatched_nothing() -> None:
    """The per-**turn** re-take, on the only path that reaches *it*: a turn whose response carries
    no tool calls runs neither the per-call nor the per-iteration check, so the prefix `drive` was
    shown at the start of the turn is checked exactly once, at the end of it. Without that re-take
    a `trace()` that rewrites history on read silently reattributes every later turn's slice.

    The two prior entries are dispatched **before** `drive` is called, so there is a prefix for the
    environment to lose — on an empty trace the check has nothing to compare and cannot fire."""
    env = ReadMutatingEnvironment()
    env.dispatch("view_cart", {})
    env.dispatch("view_cart", {})
    script = conversation("S-22f", (turn(1, "just answer, no tools"),))
    llm = stub_llm([chat_result(content="done")])

    with pytest.raises(TraceContractViolated, match=r"no longer starts with the 2 entries"):
        drive(env, script, llm, make_cfg(toolSchemas=()))


class RaisingEnvironment(StubEnvironment):
    """A pack whose `dispatch` raises — the shape a `tools/sim.py` author writes without thinking
    about it (`raise KeyError(...)` on an unknown product), and one the model can *trigger*, since
    the model chooses the arguments."""

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
        raise KeyError(f"no such product: {dict(arguments)}")


def test_drive_names_the_exception_a_packs_raising_dispatch_reaches_the_runner_as() -> None:
    """A `ToolEnvironment.dispatch` that raises used to arrive at the runner as whatever the pack
    happened to raise, and `drive`'s own docstring rules *"anything else"* to be §3.6 clause (iv)'s
    **server went away** — so a `KeyError` from a pack's `tools/sim.py` bought a re-probe and an
    exit `3` under a **false cause**, which is the signature defect §3.6 names (impl review Pass 17,
    P17-3). Naming it is what makes the two tellable apart, and it is unconditionally right
    regardless of how the record-versus-refuse question below is settled.

    The original is preserved as `__cause__` rather than swallowed, and the tool and turn travel on
    the exception so the runner can attribute the fault to a pack without re-parsing a message."""
    script = conversation("S-30", (turn(1, "add the Ghost"), turn(2, "and then?")))
    call = native_call("c1", "add_to_cart", {"name": "Ghost"})
    llm = stub_llm([chat_result(content=None, tool_calls=(call,)), chat_result(content="done")])

    with pytest.raises(ToolDispatchFailed) as excinfo:
        drive(RaisingEnvironment(), script, llm, make_cfg(toolSchemas=()))

    assert type(excinfo.value) is ToolDispatchFailed
    assert isinstance(excinfo.value.__cause__, KeyError)
    assert excinfo.value.toolName == "add_to_cart"
    assert excinfo.value.turnIndex == 0
    assert "add_to_cart" in str(excinfo.value)
    assert "KeyError" in str(excinfo.value)
    # The raise is on turn 0's very first call, so no turn has completed yet, and the parsed
    # arguments are the dict the model's tool call actually carried (dispatch-failure note §4(b)).
    assert excinfo.value.completedTurns == ()
    assert excinfo.value.parsedArguments == {"name": "Ghost"}


def test_dispatch_failure_carries_the_turns_completed_before_it() -> None:
    """`ToolDispatchFailed.completedTurns` is the turns `drive` already finished before the raise
    — the runner's only way to build the censored `ConversationTrace` it stores, since `drive`
    builds `turn_traces` as a private local and returns nothing on a raise (dispatch-failure note
    §4(b); `docs/plans/small-model-benchmarking-runner-spec.md` §2.2 item 2)."""
    script = conversation(
        "S-30b", (turn(1, "just answer"), turn(2, "add the Ghost"), turn(3, "and then?"))
    )
    call = native_call("c1", "add_to_cart", {"name": "Ghost"})
    llm = stub_llm(
        [
            chat_result(content="ok"),
            chat_result(content=None, tool_calls=(call,)),
            chat_result(content="done"),
        ]
    )

    with pytest.raises(ToolDispatchFailed) as excinfo:
        drive(RaisingEnvironment(), script, llm, make_cfg(toolSchemas=()))

    assert excinfo.value.turnIndex == 1
    assert len(excinfo.value.completedTurns) == 1
    assert excinfo.value.completedTurns[0].turnDisposition == "replied"
    assert excinfo.value.completedTurns[0].finalReplyText == "ok"
    assert excinfo.value.parsedArguments == {"name": "Ghost"}


def test_drive_fails_closed_on_a_raising_dispatch_and_drives_no_further_turn() -> None:
    """The record-versus-refuse half, pinned **as it stands** so the ruling is visible rather than
    implied. Today a raising `dispatch` is treated the way `TraceContractViolated` is — a pack
    defect that fails closed (§3.3) — so the conversation is abandoned and no `ConversationTrace`
    is returned at all.

    It is deliberately its own test: §4 S2's *"or the dispatch raised"* clause is evidence that
    *record the call as undispatchable and drive past* was the original intent, and the trigger is
    partly model-chosen, so the question is genuinely two-sided and is owed to whoever writes
    `tools/sim.py` (S5). Whichever way it is settled, this is the test that changes — the naming
    above does not."""
    script = conversation("S-31", (turn(1, "add the Ghost"), turn(2, "and then?")))
    call = native_call("c1", "add_to_cart", {"name": "Ghost"})
    llm = stub_llm([chat_result(content=None, tool_calls=(call,)), chat_result(content="done")])

    with pytest.raises(ToolDispatchFailed):
        drive(RaisingEnvironment(), script, llm, make_cfg(toolSchemas=()))
    # Turn 2 was never reached: one call issued out of a two-turn script.
    assert len(llm.calls) == 1


def test_drive_accepts_a_conforming_environment_across_several_turns() -> None:
    """The control for the two refusals above: a `StubEnvironment` that appends one record per
    call and never rewrites earlier ones drives a two-turn script clean, so the guard is shown to
    discriminate rather than to refuse everything."""
    script = conversation("S-22c", (turn(1, "add one"), turn(2, "add another")))
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(native_call("c1", "add_to_cart", {"n": 1}),)),
            chat_result(content="ok"),
            chat_result(content=None, tool_calls=(native_call("c2", "add_to_cart", {"n": 2}),)),
            chat_result(content="ok"),
        ]
    )
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    assert [len(t.dispatches) for t in trace.turns] == [1, 1]


def test_drive_records_a_turn_whose_only_calls_were_undispatchable_with_an_empty_trace() -> None:
    """The case that discriminates the emission form (§5 test 10b; P13-5): `toolCallForm` at the
    transport boundary reads `"native"` while `|E(t)| == 0`, so a scorer that reads the form from
    iteration 1 would partition this turn `native` where `-ml` §4.2's own predicate over the
    dispatch trace makes it `no_attempt`. `drive`'s job is to leave both facts on the record; the
    partition itself is S5's."""
    nameless = {"id": "x", "type": "function", "function": {"arguments": "{}"}}
    script = conversation("S-23", (turn(1, "do it"),))
    llm = stub_llm([chat_result(content=None, tool_calls=(nameless,)), chat_result(content="hm")])
    turn_trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=())).turns[0]
    assert turn_trace.chatResults[0].toolCallForm == "native"
    assert turn_trace.dispatches == ()


# --------------------------------------------------------------------------------------------
# drive — the five dispositions, and never abandoning the script (§5 test 10c)
# --------------------------------------------------------------------------------------------


def _five_turn_script(script_id: str) -> Conversation:
    return conversation(script_id, tuple(turn(i, f"t{i}") for i in range(1, 6)))


@pytest.mark.parametrize(
    ("failure", "expected_disposition"),
    [
        (LMStudioCallFailed("dropped", status=None), "no-response"),
        (LMStudioCallFailed("refused", status=400), "server-rejected"),
        (LMStudioCallTimeout("slow"), "timed-out"),
    ],
)
def test_drive_records_a_failed_turn_and_runs_the_rest_of_the_script(
    failure: Exception, expected_disposition: str
) -> None:
    """§5 test 10c, and the blocker Pass 15 raised: `-ml` §4.1's hard rule is that a turn is never
    skipped because a previous turn failed, so an error propagating out of `drive` would abandon
    the script and destroy every later turn's denominator — and `turnIndex` is **positional** in
    `ConversationTrace.turns`, so a dropped turn silently shifts every later pairing key."""
    script = _five_turn_script("S-24")
    responses: list[Any] = [chat_result(content=f"reply {i}") for i in range(5)]
    responses[2] = failure
    llm = stub_llm(responses)

    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))

    assert len(trace.turns) == 5
    assert len(llm.calls) == 5
    assert trace.turns[2].turnDisposition == expected_disposition
    assert trace.turns[2].finalReplyText is None
    assert trace.turns[2].chatResults == ()
    assert trace.turns[2].iterations == 0
    for index in (0, 1, 3, 4):
        assert trace.turns[index].turnDisposition == "replied"
        assert trace.turns[index].finalReplyText == f"reply {index}"


def test_drive_final_reply_text_is_none_exactly_when_the_turn_did_not_reply() -> None:
    """The two-declaration invariant `finalReplyText is None` **iff**
    `turnDisposition != "replied"`, driven over **all five** mechanisms in one conversation. A
    docstring will not hold this, and the plan records eleven instances of that being true (Pass 15
    §6)."""
    script = conversation("S-25", tuple(turn(i, f"t{i}") for i in range(1, 6)))
    tool_call = native_call("c", "view_cart", {})
    cfg = make_cfg(toolSchemas=(), maxIterationsPerTurn=2)
    llm = stub_llm(
        [
            chat_result(content="a real reply"),  # turn 1 -> replied
            chat_result(content=None, tool_calls=(tool_call,)),  # turn 2 -> cap-hit
            chat_result(content=None, tool_calls=(tool_call,)),
            LMStudioCallTimeout("slow"),  # turn 3 -> timed-out
            LMStudioCallFailed("dropped", status=None),  # turn 4 -> no-response
            LMStudioCallFailed("refused", status=503),  # turn 5 -> server-rejected
        ]
    )
    trace = drive(StubEnvironment(), script, llm, cfg)

    observed = [t.turnDisposition for t in trace.turns]
    assert observed == ["replied", "cap-hit", "timed-out", "no-response", "server-rejected"]
    assert set(observed) == convo.TURN_DISPOSITIONS
    for turn_trace in trace.turns:
        assert (turn_trace.finalReplyText is None) == (turn_trace.turnDisposition != "replied")


def test_drive_tests_the_exception_path_before_the_cap() -> None:
    """§3.8.4's precedence (P14-5): a call that raises at the **cap-th** iteration records the
    transport mechanism, never `"cap-hit"` — `cap-hit` requires a *completed* response still
    carrying tool calls."""
    cfg = make_cfg(toolSchemas=(), maxIterationsPerTurn=3)
    script = conversation("S-26", (turn(1, "t1"),))
    tool_call = native_call("c", "view_cart", {})
    llm = stub_llm(
        [
            chat_result(content=None, tool_calls=(tool_call,)),
            chat_result(content=None, tool_calls=(tool_call,)),
            LMStudioCallTimeout("slow at the cap"),
        ]
    )
    turn_trace = drive(StubEnvironment(), script, llm, cfg).turns[0]
    assert len(llm.calls) == 3
    assert turn_trace.turnDisposition == "timed-out"
    assert turn_trace.iterations == 2


class _SynthesizedFailure(Exception):
    """An exception class `modelbench.convo` has never seen and shares no base with the two it
    catches — the member of the axis below that a catch narrowed to any *named* type must let
    through."""


@pytest.mark.parametrize(
    "raised",
    [
        LMStudioCallFailed("dropped", status=None),
        LMStudioCallFailed("refused", status=429),
        LMStudioCallTimeout("slow"),
    ],
)
def test_drive_catches_exactly_the_two_transport_classes_and_continues(raised: Exception) -> None:
    """A coverage probe over the catch axis, not a list of exception shapes. Every member of the
    caught side leaves a recorded turn and a completed script."""
    script = conversation("S-27", (turn(1, "t1"), turn(2, "t2")))
    llm = stub_llm([raised, chat_result(content="second")])
    trace = drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))
    assert len(trace.turns) == 2
    assert trace.turns[0].turnDisposition in convo.TURN_DISPOSITIONS
    assert trace.turns[1].finalReplyText == "second"


#: The `LMStudioError` subclasses `drive` deliberately does **not** catch — the two the narrowing
#: exists for. Written out here and bound to `lmstudio.py`'s own tree below, since "two further
#: subclasses" is a **reach** claim and the axis is what would silently go on driving two of three.
_UNCAUGHT_LMSTUDIO_SUBCLASSES = (LMStudioUnreachable, ToolCallingIneligible)


def test_the_lmstudio_subclasses_drive_declines_to_catch_are_exactly_the_two_it_names() -> None:
    """`drive`'s docstring gives *"that base has two further subclasses carrying no `.status`"* as
    the **whole** reason its `except` is narrowed, and the propagate axis below is what drives
    them. That is a claim about `lmstudio.py`'s exception tree, so it is bound to the tree rather
    than restated: a third `.status`-less subclass added there reddens here instead of quietly
    joining the set the docstring counts while the axis keeps driving two of three."""
    caught = {LMStudioCallTimeout, LMStudioCallFailed}  # transcribed from `drive`'s two `except`s
    assert set(LMStudioError.__subclasses__()) - caught == set(_UNCAUGHT_LMSTUDIO_SUBCLASSES)
    assert not any(hasattr(cls("x"), "status") for cls in _UNCAUGHT_LMSTUDIO_SUBCLASSES)


@pytest.mark.parametrize(
    "raised",
    [
        _SynthesizedFailure("a class this module has never seen"),
        RuntimeError("a bare runtime error"),
        ValueError("something else entirely"),
        LMStudioUnreachable("no server"),
        ToolCallingIneligible("not eligible"),
    ],
)
def test_drive_lets_every_other_exception_propagate(raised: Exception) -> None:
    """The other half of the same axis, and the reason the catch is narrowed rather than written
    against `LMStudioError` (v1.28, P15-6): that base has two further subclasses carrying no
    `.status`, so a handler reading `exc.status` off the base would raise `AttributeError` *inside
    the handler* and abandon the script — the one thing `-ml` §4.1 forbids absolutely. Anything
    outside the two classes is §3.6 clause (iv)'s *server went away* and is the runner's, not a
    turn disposition.

    **Both of those subclasses are driven here, and that is the whole point of the case list**
    (impl review Pass 17, P17-4). Without them the widening this docstring warns about leaves the
    suite green: `RuntimeError` is `LMStudioError`'s **parent** and propagates under any narrowing,
    and the two hand-picked non-`LMStudio` classes never reach the handler at all — so the only
    members that can see the defect are the ones the reason names."""
    script = conversation("S-28", (turn(1, "t1"), turn(2, "t2")))
    llm = stub_llm([raised, chat_result()])
    with pytest.raises(type(raised)):
        drive(StubEnvironment(), script, llm, make_cfg(toolSchemas=()))


def test_drive_replays_a_failed_prior_turn_into_the_next_turns_context() -> None:
    """The two rulings meeting: a reply-less turn is recorded *and* replayed, so turn 3's context
    still contains turn 2's scripted user text. Omitting it would shorten the visible history,
    which is the covariate the whole pack measures against."""
    script = conversation("S-29", (turn(1, "t1"), turn(2, "t2"), turn(3, "t3")))
    llm = stub_llm(
        [
            chat_result(content="first reply"),
            LMStudioCallFailed("dropped", status=None),
            chat_result(content="third reply"),
        ]
    )
    cfg = make_cfg(historyReplay="structured-replies-only", toolSchemas=())
    drive(StubEnvironment(), script, llm, cfg)

    third_turn_messages = llm.calls[2]["messages"]
    assert [(m["role"], m["content"]) for m in third_turn_messages[1:]] == [
        ("user", "t1"),
        ("assistant", "first reply"),
        ("user", "t2"),
        ("assistant", ""),
        ("user", "t3"),
    ]


# --------------------------------------------------------------------------------------------
# `TURN_DISPOSITIONS` — plan §3.8.4's five-row `turnDisposition` table, asserted (plan §4 S2)
# --------------------------------------------------------------------------------------------
#
# §4 S2 requires a **three-way** probe: `set(get_args(TurnDisposition))`, the set of dispositions
# the S5 scorer branches on, and `TURN_DISPOSITIONS` must **each** equal a constant transcribed
# into this test from §3.8.4's table — each against the transcript, never against each other,
# because two declarations bound to one another agree without either being checked against the
# plan (`AGENTS.md`, "A guard's reach lives in an asserted constant", form (i)).
#
# What that buys is **cross-unit** protection, and the narrower claim is the correct one (plan
# gate P14-3): the transcript below is authored in the same unit as the two module declarations,
# so in the round that introduces a member all three agree by construction and these legs cannot
# redden then either. They redden when a **later** unit widens the vocabulary and leaves the
# transcript behind — which is what the `timed-out` split did, on the round after this guard
# landed, and is the whole return on having built it early.
#
# THE THIRD LEG IS ABSENT ON PURPOSE, AND IT IS OWED. The S5 scorer that branches on these
# dispositions is not built yet — §4 S2 says this row splits across stages — so only two of the
# three declarations exist today. The absence is *blocked on unbuilt work*, not a choice, and
# `test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5` below is the tripwire that
# refuses to let it be forgotten: it reddens the moment a scorer package appears.

#: Transcribed by hand from plan §3.8.4's five-row table (v1.27), one row per mechanism:
#:
#:   * `replied` — a response with no tool calls ended the loop.
#:   * `cap-hit` — `maxIterationsPerTurn` reached with tool calls still being emitted.
#:   * `timed-out` — a call hit `requestTimeoutSeconds`, raising `LMStudioCallTimeout`.
#:   * `no-response` — the server did not answer, or answered unusably: `LMStudioCallFailed`
#:     with no HTTP status.
#:   * `server-rejected` — the server answered and refused: `LMStudioCallFailed` carrying one.
#:
#: Mechanisms only. What each one *scores* is `-ml` §4.3 rule 4's and is deliberately not
#: transcribed here — a second home for that mapping is what the plan gate's P14-1 was. This
#: literal is the *independent* declaration the module's two are each bound to; never derive it
#: from either of them.
_DISPOSITIONS_PER_PLAN_3_8_4 = {
    "replied",
    "cap-hit",
    "timed-out",
    "no-response",
    "server-rejected",
}


def test_turn_disposition_literal_is_exactly_the_plan_table() -> None:
    """Leg 1 of §4 S2's three-way probe: the type annotation `TurnTrace.turnDisposition` carries,
    against the transcript above."""
    assert set(get_args(convo.TurnDisposition)) == _DISPOSITIONS_PER_PLAN_3_8_4


def test_turn_dispositions_constant_is_exactly_the_plan_table() -> None:
    """Leg 2: the runtime constant a consumer validates against, against the same transcript.
    Deliberately not asserted against leg 1 — the two module-level declarations are written out
    separately so that a member added to either one alone reddens here."""
    assert convo.TURN_DISPOSITIONS == frozenset(_DISPOSITIONS_PER_PLAN_3_8_4)


def test_turn_dispositions_is_an_immutable_frozenset() -> None:
    """Appendix A types it `frozenset[str]`, and it is imported by modules this one does not own
    (S5's scorer; `-ml` §4.2(f)'s two iteration-summary constants bind against it). A mutable set
    shared that way is a constant only by convention."""
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
