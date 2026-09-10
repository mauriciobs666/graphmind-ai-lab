"""Prompt assembly and the scripted-conversation driver (§3.3, §3.8.4, §4 S2).

Design: `docs/plans/small-model-benchmarking.md` §3.3 ("`prompt` is the FR-9a carrier" — the four
axes below are pack configuration, never hardcoded) and §3.8.4 (`tool-caller`'s
`conversations.jsonl` shape, its simulated tool environment, and the bounded per-turn iteration
loop). Appendix A names this module's types; §4 S2's own code sketch names exactly two functions,
`assemble` and `drive`, and this file ships nothing beyond them
and their supporting types — unlike `modelbench.tooling`, no pack ever imports this module (the AST
allowlist names only `modelbench.tooling`), so there is no external contract to hold stable here,
only `modelbench.runner`'s (a later, separate unit).

**The four FR-9a axes**, read from a pack's `prompt` manifest block (§3.3) and carried on
`PromptConfig`:

* **`historyReplay`** — how turns before the one being assembled are re-presented, over four values
  spanning §3.3's **two** axes, role ownership and tool evidence (v1.26, P13-8 — not one ladder):
  `structured` replays the whole real exchange in native roles (each in-turn iteration's own
  `assistant` message with its `tool_calls` verbatim, then one `tool` message per call carrying the
  environment's real return value, then the turn's final `assistant` reply);
  `structured-replies-only` keeps the native roles but contributes **exactly one** `assistant`
  message per prior turn carrying that turn's **final reply text only** — no `tool_calls`, no
  `tool` messages, no breadcrumb; `plaintext` quotes a flattened transcript inside a single `user`
  message, likewise with no tool evidence; `none` replays nothing.
* **`representToolSchemasEachTurn`** — whether the tool-schema *text* block `assemble` embeds is
  repeated on every assembled turn or only the first (`turn_index == 0`). This is independent of
  whether the model's *native* tool-calling mechanism is available: `drive` passes `tools=` to the
  LLM on every turn regardless, because withholding it after turn 1 would make native tool calls
  structurally impossible from turn 2 onward on every pack that sets this `false` — a knob about
  restated prose, not about disabling the API mechanism (v1: reasoned from the plan's own "answer
  'is the replay style what breaks at turn 4?'" framing, which presupposes native calling still
  works at turn 4).
* **`historyTurns`** — a trailing window over the turns being replayed (`0` = unbounded, replay
  every prior turn). The window is taken over the **pair** of sequences below, never over one of
  them, since a drift between the two produces a plausible transcript nobody would read as wrong.
* **`maxIterationsPerTurn`** — `-ml` §4.1's `I(t)` cap, bounding the per-turn loop `drive` runs.
  Pack data with no default: `drive` **raises** on `None` rather than substituting a value (§3.3).

**What `assemble` replays a prior turn *from*: what the model actually produced in *this* run,
never the script's `expect`** (§3.8.4's *"Prompt assembly"* ruling, plan v1.25). A `Turn.expect`
block is a **scoring oracle** — it names what a correct agent should have done — and
`scoring/toolcalls.py` (S5) is its only reader. `assemble` never reads it, and the structural
guarantee is the `len(observed) == turn_index` precondition: turn *n*'s message list cannot be
built from anything but *n* observations, so no caller can reintroduce a textbook prefix. A prior
turn's replayed content therefore comes from that turn's own `TurnTrace` — its `ChatResult`s and
the environment's `DispatchRecord`s — while the `user` text is always the script's, at every mode,
because the harness drives the full script (`-ml` §4.1).

**A prior turn with no final reply is replayed and never omitted, in every mode.** The two
reply-text modes contribute an `assistant` message with `content: ""` — *captured and empty*, which
is what the conversation contained — and `structured` contributes that turn's iterations with no
trailing assistant message (v1.26, P13-9). Omitting the turn would shorten the visible history, and
history length is the covariate this whole pack measures against. What such a turn never
contributes is the script's `expect`: for a reply-less turn the repair looks natural and is wrong
three ways — hidden state §3.8.4 forbids, a silent un-contamination of the trajectory `-ml` v1.22
§4.3 rule 5 censors, and a falsification of the pack's declared `historyReplay` on exactly the
conversations under study.

**What `drive` does *not* do, because it is a later unit's**: no timing discipline (warm-up,
budgets, `coldLoadSeconds`, timeouts — `RunResult.latency`/`LatencyBlock` are the runner's, §4 S2);
no scoring (`TurnTrace.turnDisposition` is a mechanism vocabulary, and what a mechanism scores is
`-ml` §4.3 rule 4's, decided on the pair `(D(t), E(t))`). `llm(...)` is a plain callable a caller
pre-binds with `model=`/`timeout_s=` (§3.6's two budgets are sized and enforced by the runner, not
here); `drive` catches **`LMStudioCallTimeout` and `LMStudioCallFailed`** from it, records the
turn's disposition, and **continues the script**, because `-ml` §4.1 forbids skipping a turn
because a previous one failed. Any other `LMStudioError` propagates to the runner, which treats it
as §3.6 clause (iv)'s *server went away* rather than as a turn disposition (v1.28, P15-6).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from modelbench.lmstudio import ChatResult, LMStudioCallFailed, LMStudioCallTimeout
from modelbench.tooling import DispatchRecord, ToolEnvironment

#: One chat message, the same shape `LMStudio.chat`'s `messages` parameter takes
#: (`{"role": ..., "content": ...}`, plus `tool_calls`/`tool_call_id`/`name` as OpenAI's
#: tool-calling convention adds them) — a plain mapping, not a new type, since nothing in this
#: module or its caller needs more structure than that.
ChatMessage = Mapping[str, Any]

#: §3.3's four `historyReplay` values, in the manifest's own vocabulary. Written out here and
#: again in `_HISTORY_REPLAY_MODES` below as two **independent** declarations, bound to each other
#: only in `tests/test_convo.py` — deriving one from the other would make that binding a tautology
#: (`AGENTS.md`, "A guard's reach lives in an asserted constant", form (i)), which is how a member
#: added to one alone shipped green before (plan gate P15-3).
HistoryReplay = Literal["structured", "structured-replies-only", "plaintext", "none"]

#: The same four, as the runtime set `assemble` refuses against. See `HistoryReplay` above for why
#: this is spelled out rather than derived from it.
_HISTORY_REPLAY_MODES: frozenset[str] = frozenset(
    {"structured", "structured-replies-only", "plaintext", "none"}
)

#: How a turn ended — the **mechanism**, transcribed from plan §3.8.4's five-row
#: `turnDisposition` table (v1.27, which split `timed-out` out of `no-response`):
#:
#: * `replied` — a response carrying no tool calls terminated the turn's loop.
#: * `cap-hit` — `maxIterationsPerTurn` reached with tool calls still being emitted, every one
#:   of those calls having returned.
#: * `timed-out` — a call hit `requestTimeoutSeconds`, raising `LMStudioCallTimeout`.
#: * `no-response` — the server did not answer, or answered unusably: `LMStudioCallFailed`
#:   with `status is None` (a dropped connection, a socket error, or a 2xx whose body is not a
#:   usable response).
#: * `server-rejected` — the server answered and refused: `LMStudioCallFailed` carrying an HTTP
#:   status. `LMStudioCallFailed.status` is the field that partition reads, and the adapter,
#:   not a caller, decides it.
#:
#: **This is a mechanism vocabulary and it maps to no scored outcome here.** What a mechanism
#: scores is `-ml` §4.3 rule 4's, decided on the pair `(D(t), E(t))` rather than on the
#: mechanism alone; two homes for one mapping is what produced the plan gate's P14-1, so this
#: block states none of it and claims no ownership of it. What the mechanism does decide is the
#: **record**, §3.8.4's own columns: `finalReplyText`, `ItemTiming.withheldFor`, and whether the
#: runner re-probes and may exit `3` — which it does after `timed-out` alone (§3.6 clause (iv)),
#: never after the other two transport rows.
#:
#: The mechanism is recorded **separately from the reply field**: `finalReplyText is None` iff
#: `turnDisposition != "replied"`, so the four mechanisms that share an absent reply stay
#: distinguishable from one another, and no consumer can key §4 S5's *absent-not-failed* rule on
#: the reply text.
TurnDisposition = Literal["replied", "cap-hit", "timed-out", "no-response", "server-rejected"]

#: The same five, as runtime data a consumer can validate against — Appendix A types it
#: `frozenset[str]`. **Declared and gated ahead of every consumer, on purpose** (plan §4 S2),
#: and what that buys is **cross-unit** protection specifically — the narrower claim, correcting
#: this block's own earlier one (plan gate P14-3). `tests/test_convo.py`'s transcript is authored
#: in the same unit as these two declarations, so in the round that introduces a member all three
#: agree by construction and the probe cannot redden then either. It reddens when a **later**
#: unit widens the vocabulary and leaves the transcript behind, which is what earned this
#: constant a round of its own ahead of the rework unit that builds `drive`'s bounded per-turn
#: loop and consumes a constant it did not write.
#:
#: Written out rather than derived from `TurnDisposition`: the two are *independent*
#: declarations, each bound in the probe to the plan table transcribed there, so a member added
#: to either one alone is caught. A sixth mechanism arriving later reddens that probe instead of
#: silently joining the enum.
TURN_DISPOSITIONS: frozenset[str] = frozenset(
    {"replied", "cap-hit", "timed-out", "no-response", "server-rejected"}
)


@dataclass(frozen=True)
class PromptConfig:
    """A pack's `prompt` manifest block, parsed (Appendix A). `systemPrompt` and `toolSchemas` are
    the **resolved content** (the text `prompt.systemPrompt` names, the JSON Schema list
    `prompt.toolSchemas` names) — resolving those two manifest paths against a pack's root is the
    caller's job (§3.3's `data.*` paths are `Pack.data_path`'s; `prompt.*` paths have no equivalent
    in `modelbench.packs`, so nothing here presumes one exists yet), never `assemble`'s: `assemble`
    takes no pack root and cannot read a file.

    `maxIterationsPerTurn` is `int | None` and the `None` is a **role** fact, not a default: §3.3
    requires the field iff `roles.MULTI_CALL_TURN_BY_ROLE[role]` and forbids it otherwise, so the
    four item-level roles carry `None` here and `drive` — the one consumer — refuses it rather than
    inventing a number.

    The field **order** is Appendix A's, pinned by a test: these names are manifest keys, so a
    drift is a silently-unreadable `pack.json` rather than a type error (plan gate P15-9).
    """

    systemPrompt: str | None
    toolSchemas: tuple[Mapping[str, Any], ...]
    historyReplay: HistoryReplay
    representToolSchemasEachTurn: bool
    historyTurns: int
    maxIterationsPerTurn: int | None
    temperature: float
    maxTokens: int


@dataclass(frozen=True)
class Turn:
    """One scripted turn — `(seq, user, expect)`, one entry of a `conversations.jsonl` row's
    `turns` array (§3.8.4, Appendix A).

    `expect` is a **scoring oracle and `scoring/toolcalls.py` (S5) is its only reader**: it names
    what a correct agent should have done that turn, never what the model under test actually said.
    Nothing in this module reads it — `assemble` replays a prior turn from that turn's own
    `TurnTrace` (§3.8.4's *"Prompt assembly"* ruling), and `drive` never branches on it.
    """

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
    """One turn's record from `drive` — Appendix A's `ConversationTrace` row, per turn, under
    §3.8.4's bounded per-turn iteration loop:

    * `messagesSent` — the in-turn working list **as it stood when this turn's last model call was
      issued**: `assemble`'s output plus every in-turn `assistant`/`tool` message appended before
      that call. One definition, total over all five dispositions — on a turn whose first call
      raised it is exactly `assemble`'s output, and on every turn it has iteration 1's list as a
      prefix (plan gate P15-7, which asks for the choice to be *stated*).
    * `chatResults` — **every completed** iteration's `ChatResult`, in order; empty for a turn
      whose first call raised.
    * `dispatches` — this turn's own slice of `env.trace()`, spanning every iteration, never the
      whole conversation's.
    * `envState` — `env.state()` read immediately after this turn's **last** iteration's
      dispatches.
    * `iterations` — the calls in this turn that **completed**, so `iterations == len(chatResults)`
      (§3.8.4, P14-6; `-ml` §11.4 binds it to `callCount` and `len(ItemTiming.calls)`). A turn
      whose first call raised records `0`, and that `0` means *no iteration was observed*, never
      *the model used no iterations* — which is why `-ml` §4.2(f) keeps such a turn out of the
      `I(t)` summary and on the record all the same.
    * `turnDisposition` — the **mechanism** that ended the turn (see `TurnDisposition`).
    * `finalReplyText` — `None` **iff** `turnDisposition != "replied"`; §3.8.4's table is the only
      home of **that** mapping (the `finalReplyText`↔disposition one, and no other — what a
      mechanism *scores* is `-ml` §4.3 rule 4's). A terminating response whose `content` is `null`
      or `""` records `""`, *captured and empty*, never `None`.
    * `wallClockMs` — this turn's own stopwatch, **bracketing the whole turn**: the message
      assembly, every iteration, and the tool dispatches between them, which is the response time
      an operator waits through. Never one `ChatResult.wallClockMs` substituted for it, which would
      make a model that loops eight times report a *smaller* latency than one that answers in a
      single call. **`float`, never `None`, on every one of the five dispositions** — a turn whose
      first call raised still took time and still records it; §4 S2 puts the *withholding* on
      `ItemTiming.withheldFor`, which is the runner's, so an optional annotation here would be the
      only statement to the contrary and would invite the runner to key withholding on a value that
      never arrives (impl review Pass 17, P17-9).

    **There is deliberately no singular `chatResult` property over `chatResults[0]`.** The
    temptation is real — the plan gives iteration 1 no privileged role, and P13-5 ruled that a
    turn's *emission form* is `-ml` §4.2's predicate over `E(t)` (`native` iff `|E(t)| >= 1`) and is
    **never** read from iteration 1, the two diverging exactly when iteration 1's tool calls are all
    undispatchable. A convenience accessor named for the singular would be the ready-made way to
    reach for the wrong one, so consumers index `chatResults` explicitly.
    """

    messagesSent: tuple[ChatMessage, ...]
    chatResults: tuple[ChatResult, ...]
    dispatches: tuple[DispatchRecord, ...]
    envState: Mapping[str, Any]
    iterations: int
    turnDisposition: TurnDisposition
    finalReplyText: str | None
    wallClockMs: float


@dataclass(frozen=True)
class ConversationTrace:
    """A whole scripted conversation's record — one `TurnTrace` per scripted turn, in script order,
    under the script's own identity.

    `scriptId`, `shape` and `replicate` are all three carried, sourced from the driven
    `Conversation`: §3.3's `pairingKey` is `["scriptId", "replicate", "turnIndex"]` and §3.8.4 makes
    `shape` the reporting stratum, so a trace omitting either is one from which the pairing key and
    the stratum cannot be derived — and §3.8.4's reason for `replicate` existing at all is that
    raising `replicatesPerScript` later must not change the record shape, which a trace that drops
    it *is* (plan gate P15-8). `turnIndex`, the third pairing component, stays **positional** in
    `turns`, which is what makes `-ml` §4.1's never-skip-a-turn rule load-bearing here: a dropped
    turn would silently shift every later pairing key.
    """

    scriptId: str
    shape: str
    replicate: int
    turns: tuple[TurnTrace, ...]


class TraceContractViolated(RuntimeError):
    """A pack's `ToolEnvironment` broke `trace()`'s own contract — the Protocol states it *"must
    return calls in a stable order and never drop or reorder an earlier entry"* and that an
    implementation *"append[s] a `DispatchRecord` to their own internal trace on every call"*
    (`modelbench.tooling`).

    `drive` isolates a turn's dispatches as the tail of `env.trace()` beyond a prefix it read
    before dispatching, and `assemble` pairs those records back onto the model's `tool_calls`
    positionally. Both rest on that contract, and when it breaks the failure is **silent**: an
    environment that clears its trace inside `dispatch` executes a real, state-mutating
    `add_to_cart` that `drive` records as **zero** dispatches, in the module whose entire thesis is
    that FR-10's ground truth is the trace and the state (plan gate P15-5). It raises mid-run and
    so can abort a conversation, which is right — it is a **pack** defect and pack defects fail
    closed (§3.3), unlike the *model* failure `-ml` §4.1 requires to be recorded and driven past.

    **Two checks, and neither is the finding's own literal prescription** (*"refuse when
    `len(after) < trace_before`"*), because that one cannot see the shape the finding describes: a
    trace cleared and re-appended inside `dispatch` leaves the length exactly where it was, so the
    comparison never fires while the slice is empty and the call is lost. So `drive` checks (1) the
    entries it had already been shown are still there, unchanged and in order, and (2) the number
    of new entries equals the number of calls it actually dispatched. The second is what catches
    the clearing environment on its very first call — where nothing had been reported yet, so no
    prefix can have been dropped — and it is also what makes `assemble`'s positional pairing a
    checked fact rather than an assumption: a mismatch there would replay the *n*-th call's message
    carrying the *m*-th call's return value, which is a plausible transcript nobody would read as
    wrong.

    **Check (2) is taken per *call*, which is the granularity `tooling.py` states the contract at
    and the granularity the pairing is read at** (impl review Pass 17, P17-2). Taken per
    *iteration* it is a strictly weaker claim than the prose it enforces: an environment recording
    **0** entries for one call and **2** for the next balances the aggregate exactly, and the turn
    then completes clean with the first call's `tool` message carrying the second call's return
    value — the very substitution the paragraph above says is checked. Check (1) is additionally
    re-taken over the whole iteration and again over the whole turn. The **per-iteration and
    per-turn re-takes are independent of each other**: a `trace()` that mutates the environment on
    *read* moves the record **between** two calls, where a check reading its own `before`
    afterwards cannot see it, and on a turn that dispatches nothing at all only the turn-level
    re-take is left. The **per-iteration check has a known coupling with the per-call check**, via
    the test fixture's read-count behavior: deleting the per-call check alone does not cause the
    per-iteration test to redden, because the fixture's drop timing depends on when that read
    occurs; the production guard logic is unaffected and all detectable defects are still caught
    (impl review Pass 18, P18-1).
    """


class ToolDispatchFailed(RuntimeError):
    """A pack's `ToolEnvironment.dispatch` raised. A **pack** defect, like `TraceContractViolated`
    above, and deliberately a *sibling* of it rather than a subclass: one says the environment
    misreported what it did, the other that it could not do it at all.

    **It exists to be a name.** `drive`'s catch is narrowed to two transport classes and everything
    else propagates to the runner, which reads a propagated exception as §3.6 clause (iv)'s *server
    went away* and re-probes before exiting `3`. A bare `KeyError` out of a pack's `tools/sim.py`
    therefore bought a re-probe and an exit under a **false cause** — a live server diagnosed as
    dead — which is exactly the mis-attribution §3.6 exists to prevent (impl review Pass 17,
    P17-3). With a name the runner can tell the two apart before deciding anything.

    `toolName` and `turnIndex` travel on the exception so a caller can attribute the fault without
    re-parsing a message, and the pack's own exception is preserved as `__cause__`, never swallowed.

    **What is deliberately *not* decided here is whether such a call should abort the conversation
    at all.** Today it does: `drive` lets this propagate, so the whole `ConversationTrace` is lost,
    which is `TraceContractViolated`'s precedent and defensible as a pack defect failing closed
    (§3.3). But plan §4 S2's replay contract has a category for *"the dispatch raised"* — a `tool`
    message naming the failure, the call recorded as undispatchable and the script driven past, the
    treatment `-ml` §4.1 gives a *model* failure — and the trigger is partly model-chosen, since the
    model picks the arguments a sim raises on. That is a design decision owed to whoever writes
    `tools/sim.py` (S5) and it is left open on purpose; if it is settled the other way, the change
    is a `try`/`except ToolDispatchFailed` at this exception's one raise site plus the
    `_undispatchable_tool_content` reason to go with it, and this class is unaffected either way.
    """

    def __init__(self, message: str, *, toolName: str, turnIndex: int) -> None:
        super().__init__(message)
        self.toolName = toolName
        self.turnIndex = turnIndex


def _system_message(cfg: PromptConfig) -> dict[str, Any] | None:
    if not cfg.systemPrompt:
        return None
    return {"role": "system", "content": cfg.systemPrompt}


def _tool_schema_message(schemas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "role": "system",
        "content": "Available tools (JSON Schema):\n" + json.dumps(list(schemas)),
    }


def _tool_call_name(call: Any) -> str | None:
    """The `function.name` of one native tool-call entry, or `None` when the entry does not carry
    a usable one — a call with no name cannot be dispatched at all."""
    function = call.get("function") if isinstance(call, Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    return name if isinstance(name, str) and name else None


def _parse_tool_arguments(raw: Any) -> dict[str, Any] | None:
    """A native tool call's `function.arguments` is a JSON *string* on the wire (OpenAI's
    convention, which LM Studio's `/api/v0/chat/completions` follows) — `ToolEnvironment.dispatch`
    takes an already-parsed mapping, so `drive` parses here, once, at the boundary.

    **`None` means *this call's arguments could not be read as an object*, and `{}` recovers its
    single meaning: the model really sent one** (plan gate P15-4). Degrading unparseable JSON, a
    JSON array and a JSON scalar all to `{}` made four different wire values arrive at `dispatch`
    as the same four bytes — and worse, it *dispatched* them, mutating FR-10 ground-truth state on
    a call the model never validly made and feeding the return value back into the model's next
    iteration. A `None` here is not dispatched; §4 S2's replay contract already has the category
    for it, the `tool` message whose content names the failure.

    Still tolerant in the sense that matters, like `lmstudio.py`'s own transport-boundary helpers:
    nothing raises. A malformed tool call is the model's failure to score, not the harness's to
    crash on.
    """
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _undispatchable_tool_content(reason: str) -> str:
    """The `tool` message body for a `tool_calls` entry that could not be dispatched — §4 S2's
    *"a JSON object naming the failure"*. **Every `tool_calls` entry gets exactly one `tool`
    message, without exception**: an unanswered tool call is rejected by OpenAI-shaped servers, so
    omitting it would convert a model failure into a transport failure at the next turn."""
    return json.dumps({"error": "tool-call-not-dispatched", "reason": reason})


def _iteration_exchange(
    chat_result: ChatResult,
    dispatches: Sequence[DispatchRecord],
    cursor: int,
) -> tuple[list[dict[str, Any]], int]:
    """One in-turn iteration's own messages: the model's `assistant` message with `content` exactly
    as returned (`None` permitted) and its `tool_calls` **verbatim, ids included**, then one `tool`
    message per entry in emission order.

    `dispatches` is the turn's dispatch slice and `cursor` the index of the next unconsumed record;
    a dispatchable entry consumes one, an undispatchable one consumes none and gets the failure
    body instead. Returns the messages and the advanced cursor.

    **This is the single producer of that shape, used by `drive` to extend the in-turn working list
    and by `assemble` to replay the turn under `structured`.** Two implementations of one message
    layout would drift, and the drift is silent: a replayed prefix that plausibly differs from what
    was actually sent.
    """
    message = chat_result.message if isinstance(chat_result.message, Mapping) else {}
    assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    tool_calls = list(chat_result.tool_calls)
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    messages: list[dict[str, Any]] = [assistant]
    for call in tool_calls:
        call_id = call.get("id") if isinstance(call, Mapping) else None
        function = call.get("function") if isinstance(call, Mapping) else None
        name = _tool_call_name(call)
        arguments = _parse_tool_arguments(
            function.get("arguments") if isinstance(function, Mapping) else None
        )
        if name is None:
            content = _undispatchable_tool_content("missing-function-name")
        elif arguments is None:
            content = _undispatchable_tool_content("unparseable-arguments")
        elif cursor >= len(dispatches):
            # The environment recorded fewer entries than the harness dispatched. Nothing here can
            # invent a return value, and the entry still owes a `tool` message.
            content = _undispatchable_tool_content("no-dispatch-record")
        else:
            content = json.dumps(dispatches[cursor].returnValue, default=str)
            cursor += 1
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}
        )
    return messages, cursor


def _replay_structured(script_turn: Turn, observed: TurnTrace) -> list[dict[str, Any]]:
    """One prior turn under `structured`: its scripted `user` message, then every **tool-calling**
    iteration's own exchange, then the turn's final `assistant` reply — **which a turn with no
    final reply simply does not have** (v1.26, P13-9). Such a turn ends on a `tool` message
    immediately before the next `user` turn, and is present rather than omitted.

    A `ChatResult` carrying **no** tool calls is the response that terminated the loop, and its
    content *is* `finalReplyText` — so it is replayed once, as the trailing assistant reply, and
    not a second time as an iteration. The loop only continues on a tool-calling response, so at
    most one `ChatResult` per turn is in that position and the two branches are exhaustive."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": script_turn.user}]
    cursor = 0
    for chat_result in observed.chatResults:
        if not chat_result.tool_calls:
            continue
        iteration_messages, cursor = _iteration_exchange(
            chat_result, observed.dispatches, cursor
        )
        messages.extend(iteration_messages)
    if observed.finalReplyText is not None:
        messages.append({"role": "assistant", "content": observed.finalReplyText})
    return messages


def _final_reply_for_replay(observed: TurnTrace) -> str:
    """The reply text the two reply-text modes replay. A turn with no final reply contributes
    `""` — *captured and empty*, which is what the conversation contained — and **never** the
    script's `expect`, which is a scoring oracle and not a transcript (§3.8.4)."""
    return observed.finalReplyText or ""


def _flatten_turn(script_turn: Turn, observed: TurnTrace) -> str:
    return f"User: {script_turn.user}\nAssistant: {_final_reply_for_replay(observed)}"


def assemble(
    turn_index: int,
    script: Sequence[Turn],
    observed: Sequence[TurnTrace],
    cfg: PromptConfig,
) -> list[ChatMessage]:
    """Build turn `turn_index`'s complete request message list (§3.3, §3.8.4, §4 S2's replay
    contract). `script` is the whole conversation's scripted turns and supplies **user text only**;
    `observed` is what this run actually produced for turns `0 .. turn_index - 1`. The harness
    resends the conversation from scratch every turn (no hidden state), so this is the entire
    message list a caller passes to `llm(...)`, not an increment.

    Order: system prompt (if any) · the tool-schema text block on turn 0, or every turn when
    `representToolSchemasEachTurn` · the replayed history (per `historyReplay`) · the current
    turn's own `{"role": "user", "content": script[turn_index].user}` message, **always last**.

    **Two preconditions, both raising.** `0 <= turn_index < len(script)`, and
    `len(observed) == turn_index`. The second is the whole mechanism of §3.8.4's ruling: turn *n*
    cannot be assembled from anything but *n* observations, so there is no argument through which a
    textbook prefix could re-enter.

    `historyTurns > 0` windows the replayed prefix from the tail — over the **pair**
    `(script[i], observed[i])`, never over one sequence, since an index drift between the two is
    silent and produces a plausible transcript.
    """
    if not (0 <= turn_index < len(script)):
        raise ValueError(
            f"turn_index {turn_index} is out of range for a {len(script)}-turn script"
        )
    if len(observed) != turn_index:
        raise ValueError(
            f"assemble(turn_index={turn_index}) needs exactly {turn_index} observed turn(s), "
            f"got {len(observed)}: a turn is replayed from what the model actually produced "
            "(plan §3.8.4), so the two sequences cannot disagree"
        )
    if cfg.historyReplay not in _HISTORY_REPLAY_MODES:
        raise ValueError(
            f"unknown historyReplay {cfg.historyReplay!r}; must be one of "
            f"{sorted(_HISTORY_REPLAY_MODES)!r}"
        )

    prior = list(zip(script[:turn_index], observed, strict=True))
    if cfg.historyTurns > 0:
        prior = prior[-cfg.historyTurns :]

    messages: list[dict[str, Any]] = []
    system = _system_message(cfg)
    if system is not None:
        messages.append(system)
    if cfg.toolSchemas and (turn_index == 0 or cfg.representToolSchemasEachTurn):
        messages.append(_tool_schema_message(cfg.toolSchemas))

    if cfg.historyReplay == "structured":
        for script_turn, observed_turn in prior:
            messages.extend(_replay_structured(script_turn, observed_turn))
    elif cfg.historyReplay == "structured-replies-only":
        for script_turn, observed_turn in prior:
            messages.append({"role": "user", "content": script_turn.user})
            messages.append(
                {"role": "assistant", "content": _final_reply_for_replay(observed_turn)}
            )
    elif cfg.historyReplay == "plaintext":
        if prior:
            transcript = "\n\n".join(_flatten_turn(s, o) for s, o in prior)
            messages.append({"role": "user", "content": f"Prior conversation:\n{transcript}"})
    # "none": no history messages at all.

    messages.append({"role": "user", "content": script[turn_index].user})
    return messages


def _final_reply_text(chat_result: ChatResult) -> str:
    """A terminating response's reply text: `str(content or "")`, **never `None`** — a response
    with `content: null` or `""` is *captured and empty*, which is a different fact from a turn
    that produced no reply at all (§3.8.4's table)."""
    message = chat_result.message if isinstance(chat_result.message, Mapping) else {}
    return str(message.get("content") or "")


def drive(
    env: ToolEnvironment,
    script: Conversation,
    llm: Callable[..., ChatResult],
    cfg: PromptConfig,
) -> ConversationTrace:
    """Execute `script` turn by turn against `env` and return the full trace.

    **One turn is a bounded iteration loop, not one call** (§3.8.4). Per scripted turn: call the
    model; while the response carries native `tool_calls` and the cap is not reached, dispatch each
    call against `env`, append the assistant message and one `tool` message per call to the
    **in-turn** working list, and call again; stop when a response carries no tool calls — that
    response's text is the turn's final reply — or when `cfg.maxIterationsPerTurn` is reached.

    `llm` is called as `llm(messages, tools=tools_or_None, temperature=cfg.temperature,
    max_tokens=cfg.maxTokens)` — `model=`/`timeout_s=` are pre-bound by the caller (the runner unit
    sizes and enforces §3.6's two budgets; this function has no timeout parameter to size one with).
    `tools` is passed on **every** turn when `cfg.toolSchemas` is non-empty, regardless of
    `cfg.representToolSchemasEachTurn` — that knob governs only the *textual* schema block
    `assemble` embeds (module docstring); withholding the native `tools` parameter after turn 1
    would make native tool calls impossible from turn 2 on every pack that sets it `false`.

    **Every scripted turn runs, in order, and a turn is never skipped because a previous turn
    failed** (`-ml` §4.1's hard rule). `LMStudioCallTimeout` and `LMStudioCallFailed` are caught,
    recorded as the turn's disposition, and the script continues: an error that propagated out of
    here would abandon the script and destroy every later turn's denominator. The catch is narrowed
    to those two classes — `LMStudioError` is the base and has two further subclasses carrying no
    `.status`, so `except LMStudioError as exc: ... exc.status` would raise `AttributeError`
    *inside the handler* and abandon the script, the one thing §4.1 forbids absolutely (v1.28,
    P15-6). Anything else **from `llm`** propagates to the runner as §3.6 clause (iv)'s *server
    went away*.

    **A pack's `ToolEnvironment.dispatch` that raises is not that**, and is re-raised as
    `ToolDispatchFailed` so the runner cannot confuse the two: unnamed, it propagated bare and the
    runner's *anything else* rule diagnosed a live server as dead (impl review Pass 17, P17-3).
    That divergence from §4 S2's *"or the dispatch raised"* replay category is stated at
    `ToolDispatchFailed` itself, together with what remains open about it — this function's current
    behaviour is to fail closed and abandon the conversation, exactly as for
    `TraceContractViolated`.

    **The exception path is tested before the cap** (§3.8.4's precedence, P14-5), so a call that
    raises at the cap-th iteration is `timed-out`/`no-response`/`server-rejected` and never
    `cap-hit`: `cap-hit` requires a *completed* response still carrying tool calls.

    Raises `TypeError` immediately, before issuing any LLM call, when `env` does not structurally
    satisfy `ToolEnvironment` (`modelbench.tooling`'s `@runtime_checkable` is what makes this a real
    check rather than a trust-blind call to a pack's `build_environment()` return value), and
    `ValueError` when `cfg.maxIterationsPerTurn` is not a usable cap — `None` because §3.3 makes the
    field pack data with **no default** and this is its one consumer, so refusing is what keeps no
    number from being invented anywhere; a non-positive cap for the same reason one iteration up,
    since §3.8.4's five-member disposition set has no member for *the cap forbade the first call*
    and a silently undefined record is worse than a refusal.
    """
    if not isinstance(env, ToolEnvironment):
        raise TypeError(
            f"drive: env {env!r} does not implement ToolEnvironment "
            "(schemas/dispatch/trace/state, plan §3.3/§4 S2)"
        )
    cap = cfg.maxIterationsPerTurn
    if cap is None:
        raise ValueError(
            "drive: cfg.maxIterationsPerTurn is None; it is pack data with no default (plan §3.3) "
            "and this is its only consumer, so no value is substituted here"
        )
    if cap < 1:
        raise ValueError(
            f"drive: cfg.maxIterationsPerTurn must be at least 1, got {cap}; a turn that never "
            "issues a call has no mechanism in plan §3.8.4's disposition table"
        )

    tools = list(cfg.toolSchemas) or None
    turn_traces: list[TurnTrace] = []
    for index in range(len(script.turns)):
        turn_traces.append(_drive_turn(env, script.turns, turn_traces, llm, cfg, index, cap, tools))
    return ConversationTrace(
        scriptId=script.scriptId,
        shape=script.shape,
        replicate=script.replicate,
        turns=tuple(turn_traces),
    )


def _check_trace_contract(
    after: Sequence[DispatchRecord],
    before: Sequence[DispatchRecord],
    dispatched: int | None,
    where: str,
) -> None:
    """The two checks `drive` takes around a set of dispatches — see `TraceContractViolated` for
    why each one is needed and why a length comparison is neither. `dispatched` is the number of
    calls just handed to `env.dispatch` — **`1` at the per-call site, which is the granularity the
    count contract is stated and read at** — or `None` where only the prefix is being re-checked."""
    if list(after[: len(before)]) != list(before):
        raise TraceContractViolated(
            f"drive: env.trace() no longer starts with the {len(before)} entries it reported "
            f"before {where} (it now holds {len(after)}); a turn's dispatches are isolated as the "
            "tail beyond that prefix, so a rewritten trace silently records real calls as none "
            "(plan §3.3, §4 S2)"
        )
    if dispatched is not None and len(after) - len(before) != dispatched:
        raise TraceContractViolated(
            f"drive: env.trace() grew by {len(after) - len(before)} entries across {where}, which "
            f"dispatched {dispatched} call(s); ToolEnvironment.dispatch must record exactly one "
            "DispatchRecord per call, and both the turn's dispatch slice and the replayed "
            "tool-message pairing are read positionally from that (plan §3.3, §4 S2)"
        )


def _drive_turn(
    env: ToolEnvironment,
    script_turns: Sequence[Turn],
    observed: Sequence[TurnTrace],
    llm: Callable[..., ChatResult],
    cfg: PromptConfig,
    index: int,
    cap: int,
    tools: list[Mapping[str, Any]] | None,
) -> TurnTrace:
    """One scripted turn's bounded iteration loop — `drive`'s body, per turn. Split out so the
    loop's control flow (and its precedence: exception before cap) reads as one function."""
    start = time.monotonic()
    working: list[ChatMessage] = list(assemble(index, script_turns, observed, cfg))
    before_turn = list(env.trace())
    chat_results: list[ChatResult] = []
    disposition: TurnDisposition = "cap-hit"
    final_reply: str | None = None
    messages_sent: tuple[ChatMessage, ...] = tuple(working)

    for iteration in range(cap):
        messages_sent = tuple(working)
        try:
            chat_result = llm(
                list(working),
                tools=tools,
                temperature=cfg.temperature,
                max_tokens=cfg.maxTokens,
            )
        except LMStudioCallTimeout:
            disposition = "timed-out"
            break
        except LMStudioCallFailed as exc:
            disposition = "server-rejected" if exc.status is not None else "no-response"
            break
        chat_results.append(chat_result)
        if not chat_result.tool_calls:
            disposition = "replied"
            final_reply = _final_reply_text(chat_result)
            break
        before_iteration = list(env.trace())
        dispatched = 0
        for call in chat_result.tool_calls:
            name = _tool_call_name(call)
            if name is None:
                continue
            function = call.get("function") if isinstance(call, Mapping) else None
            arguments = _parse_tool_arguments(
                function.get("arguments") if isinstance(function, Mapping) else None
            )
            if arguments is None:
                continue
            before_call = list(env.trace())
            try:
                env.dispatch(name, arguments)
            except Exception as exc:
                raise ToolDispatchFailed(
                    f"drive: env.dispatch({name!r}, ...) raised "
                    f"{type(exc).__name__} on turn {index}'s iteration {iteration + 1}; a "
                    "ToolEnvironment.dispatch that raises is a pack defect and fails closed "
                    "(plan §3.3), and it is named rather than propagated bare so the runner does "
                    "not read it as §3.6 clause (iv)'s server went away",
                    toolName=name,
                    turnIndex=index,
                ) from exc
            _check_trace_contract(
                list(env.trace()),
                before_call,
                1,
                f"turn {index}'s iteration {iteration + 1}, dispatch of {name!r}",
            )
            dispatched += 1
        after_iteration = list(env.trace())
        _check_trace_contract(
            after_iteration,
            before_iteration,
            dispatched,
            f"turn {index}'s iteration {iteration + 1}",
        )
        iteration_messages, _ = _iteration_exchange(
            chat_result, after_iteration[len(before_iteration) :], 0
        )
        working.extend(iteration_messages)

    after = list(env.trace())
    _check_trace_contract(after, before_turn, None, f"turn {index}")
    return TurnTrace(
        messagesSent=messages_sent,
        chatResults=tuple(chat_results),
        dispatches=tuple(after[len(before_turn) :]),
        envState=env.state(),
        iterations=len(chat_results),
        turnDisposition=disposition,
        finalReplyText=final_reply,
        wallClockMs=(time.monotonic() - start) * 1000.0,
    )
