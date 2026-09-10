"""Prompt assembly and the scripted-conversation driver (§3.3, §3.8.4, §4 S2).

Design: `docs/plans/small-model-benchmarking.md` §3.3 ("`prompt` is the FR-9a carrier" — the three
axes below are pack configuration, never hardcoded) and §3.8.4 (`tool-caller`'s
`conversations.jsonl` shape and its simulated tool environment). Appendix A names this module's
types; §4 S2's own code sketch names exactly two functions, `assemble` and `drive`, and this file
ships nothing beyond them
and their supporting types — unlike `modelbench.tooling`, no pack ever imports this module (the AST
allowlist names only `modelbench.tooling`), so there is no external contract to hold stable here,
only `modelbench.runner`'s (a later, separate unit).

**The three FR-9a axes**, read from a pack's `prompt` manifest block (§3.3) and carried on
`PromptConfig`:

* **`historyReplay`** (`"structured" | "plaintext" | "none"`) — how turns before the one being
  assembled are re-presented: native `assistant`/`tool` message scaffolding, a single flattened
  text transcript, or omitted entirely.
* **`representToolSchemasEachTurn`** — whether the tool-schema *text* block `assemble` embeds is
  repeated on every assembled turn or only the first (`turn_index == 0`). This is independent of
  whether the model's *native* tool-calling mechanism is available: `drive` passes `tools=` to the
  LLM on every turn regardless, because withholding it after turn 1 would make native tool calls
  structurally impossible from turn 2 onward on every pack that sets this `false` — a knob about
  restated prose, not about disabling the API mechanism (v1: reasoned from the plan's own "answer
  'is the replay style what breaks at turn 4?'" framing, which presupposes native calling still
  works at turn 4).
* **`historyTurns`** — a trailing window over the turns being replayed (`0` = unbounded, replay
  every prior turn).

**What `assemble` replays a prior turn *from*.** A script's `Turn.expect` (`toolRequired`, `tool`,
`args`, `finalReplyMustContain`, per the `conversations.jsonl` row shape, §3.8.4) is a **scoring
oracle**, not a transcript — it names what a correct agent should have done, never what the model
under test actually said. `assemble` replays prior turns *from that oracle*, i.e. the
"textbook" conversation, never from this run's own model output. That is a design choice, not an
oversight, made for three reasons all present in the reading: (1) "the harness never carries hidden
state between turns beyond what the configuration says it carries" (§3.8.4) reads most literally as
`assemble` being a pure function of `(turn_index, history, cfg)` with nothing threaded in from a
live run; (2) the determinism probe (§3.8.4) re-runs two scripts and diffs outcome vectors turn by
turn — that comparison is only clean if both runs see an *identical* context at every turn, which a
model's own (possibly non-deterministic) prior output cannot guarantee and a fixed script can; (3)
the per-turn hazard `P(first failure at t | clean through t-1)` (§3.8.4) is computed from the
*scored* outcome at each turn, not from what was fed as context, so feeding a fixed "clean" context
throughout is what isolates a turn-`t`-specific failure from an accumulated-error confound, matching
"gradual degradation" vs "deterministic collapse" being the distinction FR-9 exists to draw. `drive`
therefore always calls `assemble(index, script.turns, cfg)` — the *script's* fixed turns, never a
list mutated with this run's real replies.

**What `drive` does *not* do, because it is a later unit's**: no timing discipline (warm-up,
budgets, `coldLoadSeconds`, timeouts — `RunResult.latency`/`LatencyBlock` are the runner's, §4 S2);
no scoring (the nine rules under (i)-(vi) are the runner/report's); no multi-step tool loop within a
turn — `drive` issues exactly one LLM call per scripted turn, dispatches every native tool call that
response carries, and records the turn from that one call. A model that needs a second, dispatch-
result-informed call before stating its final reply is a refinement §5's `tool-caller` scoring unit
either needs or does not; nothing here presupposes an answer. `drive` never catches an LLM error —
`llm(...)` is a plain callable a caller pre-binds with `model=`/`timeout_s=` (§3.6's two budgets are
sized and enforced by the runner, not here), and an error from it propagates unchanged.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from modelbench.lmstudio import ChatResult
from modelbench.tooling import DispatchRecord, ToolEnvironment

#: One chat message, the same shape `LMStudio.chat`'s `messages` parameter takes
#: (`{"role": ..., "content": ...}`, plus `tool_calls`/`tool_call_id`/`name` as OpenAI's
#: tool-calling convention adds them) — a plain mapping, not a new type, since nothing in this
#: module or its caller needs more structure than that.
ChatMessage = Mapping[str, Any]

HistoryReplay = Literal["structured", "plaintext", "none"]

_HISTORY_REPLAY_MODES: frozenset[str] = frozenset({"structured", "plaintext", "none"})

#: How a turn ended — plan §3.8.4's four-row `turnDisposition` table (v1.26), which is the only
#: home of the mapping from mechanism to what scores it: `replied` (a response carrying no tool
#: calls terminated the turn's loop) · `cap-hit` (`maxIterationsPerTurn` reached with tool calls
#: still being emitted) · `no-response` (the call did not complete — `LMStudioCallTimeout`, or
#: `LMStudioCallFailed` carrying **no** HTTP status) · `server-rejected` (the server answered and
#: refused — `LMStudioCallFailed` carrying one; `LMStudioCallFailed.status` is the field that
#: partition reads, and the adapter, not a caller, decides it).
#:
#: The mechanism is recorded **separately from the reply field**: `finalReplyText is None` iff
#: `turnDisposition != "replied"`, so no consumer keys §4 S5's *absent-not-failed* rule on the
#: reply text and converts a §3.6 `fail` into an `n_a`.
TurnDisposition = Literal["replied", "cap-hit", "no-response", "server-rejected"]

#: The same four, as runtime data a consumer can validate against — Appendix A types it
#: `frozenset[str]`. **Declared and gated ahead of every consumer, on purpose** (plan §4 S2): a
#: probe authored in the same step that introduces a member can never redden against it, so
#: `TURN_DISPOSITIONS` lands in its own unit and the rework unit that builds `drive`'s bounded
#: per-turn loop consumes a constant it did not write. A fifth mechanism arriving later reddens
#: `tests/test_convo.py`'s probe instead of silently joining the enum.
#:
#: Written out rather than derived from `TurnDisposition` for that same reason: the two are
#: *independent* declarations, each bound in the probe to the plan table transcribed there, so a
#: member added to either one alone is caught. Nothing in this module consumes it yet.
TURN_DISPOSITIONS: frozenset[str] = frozenset(
    {"replied", "cap-hit", "no-response", "server-rejected"}
)


@dataclass(frozen=True)
class PromptConfig:
    """A pack's `prompt` manifest block, parsed (Appendix A). `systemPrompt` and `toolSchemas` are
    the **resolved content** (the text `prompt.systemPrompt` names, the JSON Schema list
    `prompt.toolSchemas` names) — resolving those two manifest paths against a pack's root is the
    caller's job (§3.3's `data.*` paths are `Pack.data_path`'s; `prompt.*` paths have no equivalent
    in `modelbench.packs`, so nothing here presumes one exists yet), never `assemble`'s: `assemble`
    takes no pack root and cannot read a file.
    """

    systemPrompt: str | None
    toolSchemas: tuple[Mapping[str, Any], ...]
    historyReplay: HistoryReplay
    representToolSchemasEachTurn: bool
    historyTurns: int
    temperature: float
    maxTokens: int


@dataclass(frozen=True)
class Turn:
    """One scripted turn — `(seq, user, expect)`, one entry of a `conversations.jsonl` row's
    `turns` array (§3.8.4, Appendix A). `expect` is read only by `assemble`'s replay of *prior*
    turns (never by the harness for control flow) and only by `scoring/toolcalls.py` (S5) for the
    turn's real scored outcome."""

    seq: int
    user: str
    expect: Mapping[str, Any]


@dataclass(frozen=True)
class Conversation:
    """One row of `conversations.jsonl` (§3.8.4) — a fixed, versioned script. `description` and
    `provenance` are carried for self-containment (a script row loaded once should not need a
    second read of the file to be printed or audited) but drive no behaviour here, so they default
    rather than forcing every hand-built test fixture to restate them."""

    scriptId: str
    shape: str
    replicate: int
    turns: tuple[Turn, ...]
    description: str = ""
    provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TurnTrace:
    """One turn's record from `drive` — Appendix A's `ConversationTrace` row, per turn:
    `messagesSent` (what `assemble` built for this turn), `chatResult` (the one LLM call `drive`
    makes per turn), `dispatches` (this turn's own slice of `env.trace()` — never the whole
    conversation's), `envState` (`env.state()` read immediately after this turn's dispatches), and
    `wallClockMs` (this turn's own stopwatch, call plus dispatch — never `chatResult.wallClockMs`
    substituted for it, since that measures only the HTTP call)."""

    messagesSent: tuple[ChatMessage, ...]
    chatResult: ChatResult
    dispatches: tuple[DispatchRecord, ...]
    envState: Mapping[str, Any]
    wallClockMs: float | None


@dataclass(frozen=True)
class ConversationTrace:
    """A whole scripted conversation's record — `scriptId` plus one `TurnTrace` per scripted turn,
    in script order."""

    scriptId: str
    turns: tuple[TurnTrace, ...]


def _system_message(cfg: PromptConfig) -> dict[str, Any] | None:
    if not cfg.systemPrompt:
        return None
    return {"role": "system", "content": cfg.systemPrompt}


def _tool_schema_message(schemas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "role": "system",
        "content": "Available tools (JSON Schema):\n" + json.dumps(list(schemas)),
    }


def _expected_exchange(turn: Turn) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The (assistant, tool) message pair standing in for `turn`'s *expected* behaviour — see the
    module docstring for why `expect`, not this run's real output, is what `structured` replay
    uses. Returns `(assistant_message, None)` when no tool call was expected that turn."""
    expect = turn.expect or {}
    reply_text = "; ".join(expect.get("finalReplyMustContain") or []) or None
    if not expect.get("toolRequired"):
        return {"role": "assistant", "content": reply_text}, None
    call_id = f"call_{turn.seq}"
    tool_name = expect.get("tool")
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(dict(expect.get("args") or {})),
                },
            }
        ],
    }
    tool = {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": reply_text or "",
    }
    return assistant, tool


def _flatten_turn(turn: Turn) -> str:
    expect = turn.expect or {}
    if expect.get("toolRequired"):
        action = f"called {expect.get('tool')}({dict(expect.get('args') or {})})"
    else:
        action = "replied directly"
    reply = "; ".join(expect.get("finalReplyMustContain") or []) or "(no stated content)"
    return f"User: {turn.user}\nAssistant: {action} -> {reply}"


def assemble(turn_index: int, history: Sequence[Turn], cfg: PromptConfig) -> list[ChatMessage]:
    """Build turn `turn_index`'s complete request message list (§3.3, §3.8.4): `history` is the
    *whole* script's turns (or at least a prefix through `turn_index`) — `history[turn_index]` is
    the turn being asked, `history[:turn_index]` are its predecessors, replayed per `cfg`'s three
    axes (module docstring). The harness resends the whole conversation from scratch every turn (no
    hidden state), so this is the entire message list a caller passes to `llm.chat`/`llm(...)`, not
    an increment.

    Order: system prompt (if any) · tool-schema text block (turn 0, or every turn when
    `representToolSchemasEachTurn`) · replayed history (per `historyReplay`) · the current turn's
    own `{"role": "user", "content": ...}` message, always last.
    """
    if not (0 <= turn_index < len(history)):
        raise ValueError(
            f"turn_index {turn_index} is out of range for a {len(history)}-turn history"
        )
    if cfg.historyReplay not in _HISTORY_REPLAY_MODES:
        raise ValueError(
            f"unknown historyReplay {cfg.historyReplay!r}; must be one of "
            f"{sorted(_HISTORY_REPLAY_MODES)!r}"
        )

    current = history[turn_index]
    prior = history[:turn_index]
    if cfg.historyTurns > 0:
        prior = prior[-cfg.historyTurns :]

    messages: list[dict[str, Any]] = []
    system = _system_message(cfg)
    if system is not None:
        messages.append(system)
    if cfg.toolSchemas and (turn_index == 0 or cfg.representToolSchemasEachTurn):
        messages.append(_tool_schema_message(cfg.toolSchemas))

    if cfg.historyReplay == "structured":
        for turn in prior:
            messages.append({"role": "user", "content": turn.user})
            assistant_message, tool_message = _expected_exchange(turn)
            messages.append(assistant_message)
            if tool_message is not None:
                messages.append(tool_message)
    elif cfg.historyReplay == "plaintext":
        if prior:
            transcript = "\n\n".join(_flatten_turn(turn) for turn in prior)
            messages.append({"role": "user", "content": f"Prior conversation:\n{transcript}"})
    # "none": no history messages at all.

    messages.append({"role": "user", "content": current.user})
    return messages


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """A native tool call's `function.arguments` is a JSON *string* on the wire (OpenAI's
    convention, which LM Studio's `/api/v0/chat/completions` follows) — `ToolEnvironment.dispatch`
    takes an already-parsed mapping, so `drive` parses here, once, at the boundary. Tolerant like
    `lmstudio.py`'s own transport-boundary helpers: an unparseable or wrong-shaped value degrades to
    `{}` rather than raising, since a malformed tool call is the model's failure to score, not the
    harness's to crash on."""
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def drive(
    env: ToolEnvironment,
    script: Conversation,
    llm: Callable[..., ChatResult],
    cfg: PromptConfig,
) -> ConversationTrace:
    """Execute `script` turn by turn against `env`, calling `llm` once per turn, and return the
    full trace (module docstring: no timing discipline, no scoring, one call per turn).

    `llm` is called as `llm(messages, tools=tools_or_None, temperature=cfg.temperature,
    max_tokens=cfg.maxTokens)` — `model=`/`timeout_s=` are pre-bound by the caller (the runner unit
    sizes and enforces §3.6's two budgets; this function has no timeout parameter to size one with).
    `tools` is passed on **every** turn when `cfg.toolSchemas` is non-empty, regardless of
    `cfg.representToolSchemasEachTurn` — that knob governs only the *textual* schema block
    `assemble` embeds (module docstring); withholding the native `tools` parameter after turn 1
    would make native tool calls impossible from turn 2 on every pack that sets it `false`.

    Every scripted turn runs unconditionally, in order — `drive` never stops early on a bad turn,
    matching the funnel-table design elsewhere in this pack's scoring (§3.8.4): every turn position
    must be attempted so an early collapse cannot silently improve every later conditional count.

    Raises `TypeError` immediately, before issuing any LLM call, when `env` does not structurally
    satisfy `ToolEnvironment` (`modelbench.tooling`'s `@runtime_checkable` is what makes this a real
    check rather than a trust-blind call to a pack's `build_environment()` return value).
    """
    if not isinstance(env, ToolEnvironment):
        raise TypeError(
            f"drive: env {env!r} does not implement ToolEnvironment "
            "(schemas/dispatch/trace/state, plan §3.3/§4 S2)"
        )
    tools = list(cfg.toolSchemas) or None
    turn_traces: list[TurnTrace] = []
    for index in range(len(script.turns)):
        start = time.monotonic()
        messages = assemble(index, script.turns, cfg)
        chat_result = llm(
            messages,
            tools=tools,
            temperature=cfg.temperature,
            max_tokens=cfg.maxTokens,
        )
        trace_before = len(env.trace())
        for call in chat_result.tool_calls:
            function = call.get("function") if isinstance(call, Mapping) else None
            name = function.get("name") if isinstance(function, Mapping) else None
            if not name:
                continue
            arguments = _parse_tool_arguments(function.get("arguments"))
            env.dispatch(name, arguments)
        dispatches_this_turn = tuple(env.trace()[trace_before:])
        env_state = env.state()
        wall_clock_ms = (time.monotonic() - start) * 1000.0
        turn_traces.append(
            TurnTrace(
                messagesSent=tuple(messages),
                chatResult=chat_result,
                dispatches=dispatches_this_turn,
                envState=env_state,
                wallClockMs=wall_clock_ms,
            )
        )
    return ConversationTrace(scriptId=script.scriptId, turns=tuple(turn_traces))
