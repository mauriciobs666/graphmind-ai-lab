"""The offline workflow-executor engine (M3, K-022 — Phase 1 / U4).

`WorkflowExecutor` walks a materialized `WorkflowDefSnapshot` (§11) as a
`WorkflowRun`, recording each executed step as a `StepRun` and driving the
run-lifecycle state machine defined in `docs/plans/m3-executor.md` §2.1. It
orchestrates via `repository.py` (the §12 queries) and injected collaborators —
it holds **no Cypher** of its own (layering, AGENTS.md).

**This landing is offline (D3).** Step execution and guard judging are injected
seams so the whole engine (advance, NEXT audit trail, `AT_STEP` relink,
suspend/resume, step budget, done/fail, tracing) is unit-testable with **stub**
handlers/guards — no LLM, no network. The LLM-native agent-node loop (`type:'agent'`
execution) and the fuzzy-guard judge prompt landed in U6–U8; the deterministic typed
handlers (`decision`/`human`/`wait`, K-024 U2) are pure and offline by construction,
and the guard judge remains an injected callable. `_execute_step` documents which step
types the engine executes today and which are an explicit raising seam.

The §2.1 loop, mapped to code (`_drive`):

  * **OUTCOME A (advance)** — a guard fires → `record_step_and_advance(to=firing.to)`
    records the just-run step's `StepRun` **and** relinks `AT_STEP` in one atomic
    query, then (budget permitting) continue.
  * **OUTCOME B (suspend)** — no guard fires *and* the step declares
    `waitsForHuman` → record the execution (advance-to-self) then `suspend_run`
    (CAS `running→waiting`); return `waiting`.
  * **OUTCOME C (re-loop / terminate)** — no guard fires and not `waitsForHuman`:
    if the step has outgoing transitions, record (advance-to-self) and re-execute
    (a legitimate self-loop, bounded by the step budget); if it is terminal (no
    outgoing) record then `complete_run` → `done`.

Because `record_step_and_advance` couples *record* + *advance* into one atomic
query (§12.2, the M4 tail-anchored write), every executed step is recorded via it,
with the advance target being the firing transition's `to` (A) or the current step
itself (B / C — "advance-to-self", which records the `StepRun` and bumps the budget
without moving `AT_STEP`). The step budget is enforced on the *continue* paths
(A, C-reloop) using the `stepCount` the advance returns.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import CallContext
from .guards import CMP_KINDS, GuardVerdict, evaluate_guard, render_label
from .repository import Repository, WorkflowRunNotFoundError
from .services import InvalidSearchQueryError, ServiceError, UnknownMemberError
from .tools import HumanHandoffSignal

_log = logging.getLogger(__name__)

# Run-level step budget default (DS note Q4 / §7 — per-def default 12).
DEFAULT_STEP_BUDGET = 12
# Per-node tool-loop cap default (DS note Q4 / §7 — tool-light proof nodes = 4).
DEFAULT_MAX_ITERATIONS = 4
# Cap a trace payload at the write boundary (rule 6 — RAM). Matches MAX_CONFIG_LEN.
TRACE_PAYLOAD_MAX = 8000
# Cap the recent-thread window folded into an agent-node prompt (rule 6 — RAM/latency).
# A long thread cannot blow the prompt; the last N turns carry the live intake context.
THREAD_CONTEXT_WINDOW = 20
# D16 (`m3-executor.md` §2.2) — the *model-correctable* subset of `ServiceError`: failures
# caused by the arguments the model chose, which it can plausibly fix on a re-prompt. This is
# an **allowlist on purpose**: every other `ServiceError` — today `UnknownActorError`,
# `ThreadNotFoundError`, `ChannelNotFoundError`, and any subclass added later — propagates to
# the M-1 fault net rather than being silently re-prompted away.
MODEL_CORRECTABLE_TOOL_ERRORS: tuple[type[ServiceError], ...] = (
    UnknownMemberError,
    InvalidSearchQueryError,
)
# Guard kinds that contribute a `guard_judgment` trace line (M-6): the LLM judge's
# verdicts **and** the deterministic `cmp` family. The unconditional default guard
# judges nothing and is deliberately absent.
TRACED_GUARD_KINDS: frozenset[str] = frozenset({"llm"}) | CMP_KINDS


# ── defaults ─────────────────────────────────────────────────────────────────

def _default_id() -> str:
    return uuid.uuid4().hex


def _default_clock() -> int:
    return int(time.time() * 1000)


# ── value objects ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StepResult:
    """The outcome of executing one step: its opaque `output`, emitted `on`, the
    debug trace events, and the msgIds the node posted (empty for the offline stub).

    `trace` is a list of `(kind, payload)` the node accumulates while it runs (llm
    prompts/responses, tool calls/results, an exhaustion note). It is emitted by
    `_trace_step` **after** the StepRun exists (the record→trace order), so the events
    key to a real `stepRunId`. A `NullTracer` drops them all (AC-5).

    `emissions` are the msgIds this node posted (`post_message`), buffered during
    execution and drained by `_link_emissions` **after** the StepRun exists — the exact
    same deferred, stepRun-keyed lifecycle as `trace` (Option B, K-023). The tool no
    longer links inline (no `stepRunId` is resolvable at dispatch time); the executor
    owns the `StepRun -[:PRODUCED]-> Message` audit link once `_record` has run.

    `thread` is the recent thread window the node already read (`_read_thread_context`) —
    carried out so the transition guard can judge against the live conversation (DS §Q1
    RECENT-TURNS fallback) **without a second read** (m-C neutral). Empty for the offline
    stub path and for non-agent steps; `evaluate_guard` then degrades to the
    `understanding`-only path."""

    output: str = ""
    on: str = "done"
    trace: list[tuple[str, str]] = field(default_factory=list)
    emissions: list[str] = field(default_factory=list)
    thread: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _TransitionDecision:
    """Which transition fired (or `None`) plus the guard judgments made en route
    (for tracing). Judgments are `(transition, guard_text, verdict)` tuples."""

    firing: dict[str, Any] | None
    judgments: list[tuple[dict[str, Any], str, GuardVerdict]] = field(
        default_factory=list
    )


# ── tracing (FR-4 / AC-5) ────────────────────────────────────────────────────

class Tracer(Protocol):
    """The per-run trace seam. `GraphTracer` writes `TraceEvent`s for a debug
    instance; `NullTracer` no-ops for a lean run (AC-5 by construction)."""

    def record(
        self, ws: str, *, step_run_id: str, seq: int, kind: str, payload: str
    ) -> None: ...


class NullTracer:
    """Records nothing — the non-debug tracer (a lean run writes zero TraceEvents)."""

    def record(self, ws: str, *, step_run_id: str, seq: int, kind: str,
               payload: str) -> None:
        return None


class GraphTracer:
    """Writes one `TraceEvent` per recorded aspect via `repository.append_trace_event`
    (§12.10). Mints `traceId`s and stamps `at` from injected id/clock; caps the
    payload length at the write boundary (rule 6)."""

    def __init__(self, repo: Repository, *, id_gen: Callable[[], str] = _default_id,
                 clock: Callable[[], int] = _default_clock) -> None:
        self._repo = repo
        self._id = id_gen
        self._clock = clock

    def record(self, ws: str, *, step_run_id: str, seq: int, kind: str,
               payload: str) -> None:
        self._repo.append_trace_event(
            ws, step_run_id=step_run_id, trace_id=self._id(), seq=seq,
            kind=kind, at=self._clock(), payload=payload[:TRACE_PAYLOAD_MAX],
        )


_NULL_TRACER = NullTracer()


def _short(value: Any, limit: int = 200) -> str:
    """A compact string form of a trace payload fragment (final cap is at the write boundary)."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _describe_result(result: Any) -> str:
    """A one-line trace description of an LLM turn (tool calls named, or the text)."""
    if result.is_tool_call:
        return "tool_calls: " + ", ".join(c.name for c in result.tool_calls)
    return "text: " + _short(result.text or "")


def _assistant_turn(result: Any) -> dict[str, Any]:
    """Echo the model's tool-call turn back into the message list (OpenAI shape) so the
    tool results that follow are correlated by `tool_call_id`."""
    return {
        "role": "assistant",
        "content": result.text or "",
        "tool_calls": [
            {
                "id": c.id, "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": json.dumps(c.arguments, separators=(",", ":")),
                },
            }
            for c in result.tool_calls
        ],
    }


def _str_list(value: Any) -> list[str]:
    """A defensive list-of-strings view of an authored config list (`config.fields`).

    Config is author-supplied data, so a non-list (or a list of non-strings) must not
    raise inside a pure handler — it degrades to what it can describe."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _dumps(obj: Any) -> str:
    """Serialize a step-output envelope compactly and deterministically (stable key order,
    so a `StepRun.output` is byte-comparable across runs)."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _load_json_obj(raw: str | None) -> dict[str, Any]:
    """Deserialize an opaque `ctx`/`config` string to a dict (rule 8 is app-side).

    `None`/`""`/non-object → `{}` so callers can `.get(...)` safely.
    """
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


class WorkflowExecutor:
    """Drives a `WorkflowRun` through the §2.1 lifecycle (offline Phase-1 engine).

    Constructed with injected collaborators so the loop is testable with stubs:
      * `services` — the domain layer (post/retrieve seams for later phases; unused
        by the Phase-1 core).
      * `repo` — the §12 run/step-run/trace queries (1:1).
      * `llm` / `tool_registry` — LLM tool-calling + node capabilities (later units;
        accepted here, unused offline).
      * `guard_judge` — the fuzzy-guard decision seam handed to `guards.evaluate_guard`
        as `judge=`: a callable `(condition, *, understanding, recent_turns, ctx,
        step_output) -> {decision, rationale}` (Q1 extract-then-judge; `recent_turns` is
        the DS RECENT-TURNS fallback, non-empty only when no `understanding` was
        emitted — `guards` applies the omit rule). Only invoked for `{"kind":"llm"}`
        guards; the empty-string unconditional guard never calls it. `evaluate_guard`
        applies the bias-to-suspend policy on malformed/contradictory output.
      * `tracer` — the debug `GraphTracer` (or any `Tracer`); a `NullTracer` is used
        for a non-debug run regardless (AC-5).
      * `step_budget` — the default `maxSteps` a run starts with (the run carries its
        own `maxSteps`, which the executor enforces).

    `id_gen`/`clock` mint `stepRunId`s and monotonic `startedAt`/`endedAt` stamps;
    the clock is wrapped monotonic (like the message clock) so same-ms StepRun ties
    are impossible — the `NEXT` audit trail then coincides with `startedAt` order
    (§12.8).
    """

    def __init__(
        self,
        services: Any,
        repo: Repository,
        *,
        llm: Any = None,
        guard_judge: Callable[..., Any] | None = None,
        tool_registry: Any = None,
        tracer: Tracer | None = None,
        step_budget: int = DEFAULT_STEP_BUDGET,
        id_gen: Callable[[], str] = _default_id,
        clock: Callable[[], int] = _default_clock,
    ) -> None:
        self._services = services
        self._repo = repo
        self._llm = llm
        self._guard_judge = guard_judge
        self._tools = tool_registry
        self._tracer = tracer or _NULL_TRACER
        self.step_budget = step_budget
        self._id = id_gen
        self._raw_clock = clock
        self._ts_lock = threading.Lock()
        self._last_ts = 0

    def _clock(self) -> int:
        """Monotonic per-executor ms clock — StepRun/trace `startedAt` ties are
        impossible at source, so `NEXT` order coincides with `startedAt` (§12.8)."""
        with self._ts_lock:
            ts = max(self._raw_clock(), self._last_ts + 1)
            self._last_ts = ts
            return ts

    # ── public entry points ──────────────────────────────────────────────────

    def run(self, ctx: CallContext, *, run_id: str) -> str:
        """Drive a freshly-started run from its current `AT_STEP` to a stopping
        point. Returns the terminal status (`waiting` / `done` / `failed`). Raises
        `WorkflowRunNotFoundError` when the run does not exist."""
        run = self._repo.get_run(ctx.ws, run_id=run_id)
        if run is None:
            raise WorkflowRunNotFoundError(run_id)
        return self._drive(ctx, run)

    def resume(
        self, ctx: CallContext, *, run_id: str, run_ctx_json: str | None = None
    ) -> str | None:
        """Resume a parked (`waiting`) run on a human reply. The `waiting→running`
        CAS is single-flight (§12.4): if it does not apply (the run is not waiting,
        or a concurrent reply already resumed it), returns `None` without driving.
        Otherwise drives the loop and returns the terminal status.

        `run_ctx_json` (K-024 **D-F**) carries a FULL merged run ctx to be written
        **inside** the CAS (§12.13) instead of the plain flip (§12.4) — so only the
        CAS winner's ctx is ever persisted and a loser writes neither the flip nor
        the ctx. Omitted ⇒ the chat/trigger path, byte-identical in behaviour to
        before. The post-CAS `get_run` below is what makes the just-written ctx the
        one `_drive_loop` loads."""
        cas = (
            self._repo.resume_run(ctx.ws, run_id=run_id)
            if run_ctx_json is None
            else self._repo.resume_run_with_ctx(
                ctx.ws, run_id=run_id, ctx=run_ctx_json
            )
        )
        if cas is None:
            return None
        run = self._repo.get_run(ctx.ws, run_id=run_id)
        if run is None:  # pragma: no cover — resumed then vanished (TOCTOU tripwire)
            raise WorkflowRunNotFoundError(run_id)
        return self._drive(ctx, run)

    # ── the §2.1 loop ──────────────────────────────────────────────────────────

    def _drive(self, ctx: CallContext, run: dict[str, Any]) -> str:
        """Drive `run` to a stopping point, converting any fault into a defined terminal
        (M-1). The §2.1 loop lives in `_drive_loop`; this wrapper is the top-level fault
        net so a mid-drive exception can never leave a `running` zombie:

          * `HumanHandoffSignal` (a granted `human_handoff` tool, §2.4) → **suspend** the
            run pending a human (reusing the intake suspend/resume mechanics), return
            `waiting`. Caught **before** the generic net (it is an `Exception` subclass);
            defensive — the triage proof grants no `human_handoff`.
          * any other exception → **fail, don't zombie.** This net is for *engine* faults:
            a **tool-level** failure (a `ServiceError` — the model's arguments rejected by
            the domain) is absorbed at the node as a re-prompt (`_handle_tool_call`) and
            never reaches here, so one bad tool argument cannot end a run. Stamp `fail_run`
            with a diagnostic
            `ctx` note (the `_fail_budget` shape) then **re-raise** so `_safe_run_workflow`'s
            isolation logs the stack. The run ends `failed` with `AT_STEP` cleared —
            resumable-never, but no longer a permanent `running` orphan (m-3's named
            `WorkflowConfigError` and the M7 `NotImplementedError` reach this net)."""
        run_id = run["runId"]
        run_ctx = _load_json_obj(run["ctx"])
        try:
            return self._drive_loop(ctx, run)
        except HumanHandoffSignal:
            self._repo.suspend_run(
                ctx.ws, run_id=run_id, thread_id=run_ctx.get("threadId", "")
            )
            return "waiting"
        except Exception as exc:
            self._fail_with_note(ctx, run_id, run_ctx, f"unexpected: {exc!r}")
            raise

    def _drive_loop(self, ctx: CallContext, run: dict[str, Any]) -> str:
        snap = self._repo.get_snapshot(
            ctx.ws, key=run["defKey"], version=run["defVersion"]
        )
        if snap is None:  # pragma: no cover — run without its snapshot (tripwire)
            raise WorkflowRunNotFoundError(f"snapshot for run {run['runId']!r} missing")

        steps_by_key = {s["key"]: s for s in snap["steps"]}
        outgoing: dict[str, list[dict[str, Any]]] = {}
        for tr in snap["transitions"]:
            outgoing.setdefault(tr["from"], []).append(tr)

        run_id = run["runId"]
        max_steps = run["maxSteps"]
        run_ctx = _load_json_obj(run["ctx"])
        tracer = self._tracer if run["trace"] else _NULL_TRACER
        current_key = run["atStepKey"]
        if current_key is None:  # already terminal — nothing to drive
            return run["status"]

        while True:
            step = steps_by_key[current_key]
            config = _load_json_obj(step["config"])
            result = self._execute_step(ctx, run, step, config, run_ctx)
            decision = self._select_transition(
                outgoing.get(current_key, []), run, run_ctx, result
            )
            firing = decision.firing
            to_key = firing["to"] if firing is not None else current_key

            rec = self._record(ctx, run, current_key, to_key, result)
            self._trace_step(ctx, tracer, rec["stepRunId"], result, decision)
            self._link_emissions(ctx, rec["stepRunId"], result.emissions)

            if firing is not None:
                # OUTCOME A — a guard fired: advance, then enforce the budget
                if rec["stepCount"] > max_steps:
                    return self._fail_budget(ctx, run_id, run_ctx)
                current_key = to_key
                continue

            if config.get("waitsForHuman"):
                # OUTCOME B — suspend (guaranteed human-unblockable, §2.4). No budget
                # check here by design: the intake loop is human-paced, bounded by the
                # DS clarifying-round ceiling (a ctx-write follow-up), NOT `maxSteps` —
                # a parked run cannot self-drive (plan §7, closing review m-1).
                self._repo.suspend_run(
                    ctx.ws, run_id=run_id, thread_id=run_ctx.get("threadId", "")
                )
                return "waiting"

            if outgoing.get(current_key):
                # OUTCOME C — re-loop (a legitimate self-loop), bounded by the budget
                if rec["stepCount"] > max_steps:
                    return self._fail_budget(ctx, run_id, run_ctx)
                continue

            # OUTCOME C — terminal (no outgoing transitions): done
            self._repo.complete_run(ctx.ws, run_id=run_id, ended_at=self._clock())
            return "done"

    # ── seams ──────────────────────────────────────────────────────────────────

    def _execute_step(
        self, ctx: CallContext, run: dict[str, Any], step: dict[str, Any],
        config: dict[str, Any], run_ctx: dict[str, Any],
    ) -> StepResult:
        """Execute one step and return its `(output, on)`.

        Dispatches on `Step.type` (§2.3, `m3-process-flow.md` §3.3 / D-E). Four types are
        executed today; the rest are an explicit, named seam:

          * `agent` **with a wired LLM** → the bounded, tool-scoped agent loop
            (`_run_agent_node`, U8).
          * `agent` **without an LLM** → the deterministic empty stub. This is
            **deliberate, not a fall-through accident** (plan F-3): it is the affordance
            the offline loop-engine tests are built on — guard-based branching
            (`_select_transition`) drives the whole §2.1 loop with no network, so an
            empty result still exercises advance / suspend / re-loop / terminate. Never
            "tidy" it into a raise.
          * `decision` / `human` / `wait` → the pure typed handlers below (K-024 U2).
            They have **no side effect**: their whole job is to produce an auditable
            `StepResult` describing what the step is (or is waiting for). All branching
            stays in the outgoing guards.
          * `prompt` / `tool` / `message` and any unknown type → `NotImplementedError`,
            the documented typed-handler seam (D-E). It reaches the M-1 fault net in
            `_drive`, which stamps `fail_run` with the message and re-raises — a named
            terminal, never a silent no-op that "succeeds" doing nothing.
        """
        step_type = step.get("type")
        if step_type == "agent":
            if self._llm is None:
                return StepResult(output="", on="done")
            return self._run_agent_node(ctx, run, step, config, run_ctx)
        if step_type == "decision":
            return self._run_decision_node(step, config)
        if step_type == "human":
            return self._run_human_node(step, config)
        if step_type == "wait":
            return self._run_wait_node(step, config)
        raise NotImplementedError(
            f"step type {step_type!r} is not implemented in this cut "
            f"(typed-handler seam); see docs/plans/m3-process-flow.md §D-E"
        )

    # ── typed step handlers (K-024 U2 — pure, side-effect-free) ────────────────

    @staticmethod
    def _run_decision_node(
        step: dict[str, Any], config: dict[str, Any]
    ) -> StepResult:
        """A `decision` node: a pure branch point. It computes nothing — its semantics
        live entirely in its outgoing guards, and with **zero** outgoing transitions it is
        a terminal outcome node (the run completes `done` there, OUTCOME C).

        The envelope key is `node`, **not** `decision` (plan n-1): `decision` is already
        the name of an approval *value* in `ctx` (`ctx.decision`), and a guard path reading
        `output.decision` would then mean something quite different from `ctx.decision`.
        """
        return StepResult(
            output=_dumps({"node": {"step": step["key"]}}),
            on="done",
            trace=[("node_note", "decision node — branching in guards")],
        )

    @staticmethod
    def _run_human_node(step: dict[str, Any], config: dict[str, Any]) -> StepResult:
        """A `human` node: park until a person supplies the declared fields.

        The park itself is the existing OUTCOME B (`config.waitsForHuman` + no firing
        guard) — this handler only *describes* the wait. Because the envelope lands on the
        `StepRun.output`, `GET /workflow-runs/{id}/step-runs` tells a client exactly what
        the run is waiting for with no new query. `on` stays `"done"`: `on` is vestigial
        (plan F-1) and inventing an `"await"` value for a dead field would be inconsistent.
        """
        return StepResult(
            output=_dumps({"awaiting": {
                "kind": "human",
                "prompt": config.get("prompt", ""),
                "assignee": config.get("assignee", ""),
                "fields": _str_list(config.get("fields")),
            }}),
            on="done",
        )

    @staticmethod
    def _run_wait_node(step: dict[str, Any], config: dict[str, Any]) -> StepResult:
        """A `wait` node: park until an external actor signals back (D-C).

        **Signal-driven, not timer-driven — this system has no scheduler** (no periodic
        worker, no due-run sweep; `BackgroundTasks` are request-scoped). Real timers are
        proposed backlog item K-028. To the engine a `wait` step is therefore
        **mechanically identical to `human`** (plan m-7): same park, same publish
        invariant, same input path, same guard mechanism. The only difference is the
        `awaiting.kind` string, which exists so a client renders the right prompt.
        """
        return StepResult(
            output=_dumps({"awaiting": {
                "kind": "signal",
                "signal": config.get("signal", ""),
            }}),
            on="done",
        )

    def _run_agent_node(
        self, ctx: CallContext, run: dict[str, Any], step: dict[str, Any],
        config: dict[str, Any], run_ctx: dict[str, Any],
    ) -> StepResult:
        """Run a `type:'agent'` node as a bounded, tool-scoped agent loop (§2.2 / Q3).

        Builds the message list (node `systemPrompt` + assembled context), offers the LLM
        **only** the node's scoped tool schemas (`config.tools`, the AC-6 author fence — an
        ungranted tool is never offered), and lets the model drive: on a tool call it
        validates the name against the granted set and the args against the tool schema, then
        dispatches via the injected `tool_registry`, feeds the result back, and loops —
        **bounded by `config.maxIterations`**. A final text ends the node with its `output`.

        **AC-6 (defensive):** a call naming an *ungranted* tool is rejected by the dispatcher
        (not merely un-offered) and a *malformed* call (missing required args) is refused —
        both surface an error back to the model as a bounded re-prompt, **never a dispatch**.
        A dispatched tool that **fails** on the model's arguments (a `ServiceError` — e.g. a
        hallucinated `@mention`) is the third case of the same rule: the error goes back to
        the model, it does not kill the run. Every one of those re-prompts costs an ordinary
        iteration, so the cycle is bounded by `maxIterations` like any other turn.
        On `maxIterations` exhaustion the node terminates gracefully with its best current
        text + a trace note; it does **not** hard-fail the run (only `maxSteps` does, §7).

        Every LLM prompt/response and tool call/result is collected into the returned
        `StepResult.trace` (emitted to the tracer once the StepRun exists — debug only).
        """
        granted = list(config.get("tools", []))
        granted_set = set(granted)
        offered = [self._tools.schema(name) for name in granted]
        required_by = {
            name: self._required_params(schema)
            for name, schema in zip(granted, offered)
        }
        max_iter = int(config.get("maxIterations", DEFAULT_MAX_ITERATIONS))

        thread_msgs = self._read_thread_context(ctx, run_ctx)
        messages = self._assemble_messages(config, run_ctx, thread_msgs)
        trace: list[tuple[str, str]] = []
        emissions: list[str] = []
        last_text = ""

        for iteration in range(1, max_iter + 1):
            trace.append(
                ("llm_prompt", f"iter {iteration}/{max_iter}: {len(messages)} msgs, "
                               f"{len(offered)} tool(s)")
            )
            result = self._llm.chat(messages, offered)
            trace.append(("llm_response", _describe_result(result)))
            if result.text:
                last_text = result.text

            if not result.is_tool_call:
                return StepResult(output=result.text or "", on="done",
                                  trace=trace, emissions=emissions,
                                  thread=thread_msgs)

            messages.append(_assistant_turn(result))
            for call in result.tool_calls:
                content = self._handle_tool_call(
                    call, granted_set, required_by, ctx, run, trace, emissions
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )

        # maxIterations exhausted — graceful termination (§7, DS Q4): best text + note.
        trace.append(
            ("node_note", f"max iterations ({max_iter}) reached; terminating node "
                          f"with best current text")
        )
        return StepResult(output=last_text, on="done", trace=trace,
                          emissions=emissions, thread=thread_msgs)

    def _handle_tool_call(
        self, call: Any, granted_set: set[str],
        required_by: dict[str, list[str]], ctx: CallContext, run: dict[str, Any],
        trace: list[tuple[str, str]], emissions: list[str],
    ) -> str:
        """Validate + dispatch one tool call; return the message content fed back to the
        model. Enforces AC-6 (ungranted → reject) and arg-schema validity (malformed →
        refuse) — both a bounded re-prompt, never a dispatch. A *dispatched* tool that fails
        with a **model-correctable** `ServiceError` (`MODEL_CORRECTABLE_TOOL_ERRORS` — the
        domain rejecting the model's own arguments) takes that same re-prompt path; every
        other `ServiceError` (a misconfigured actor, a missing thread/channel), an engine
        fault, and `HumanHandoffSignal` propagate — D16, see the `except` below. Every failed
        dispatch is logged unconditionally, tracer or not. A dispatch that posts a
        message (a `"posted"` key in the returned JSON envelope) has its msgId buffered
        onto `emissions` for post-record PRODUCED linking (Option B, K-023)."""
        if call.name not in granted_set:  # AC-6 defensive rejection
            trace.append(("tool_call", f"REJECTED ungranted tool {call.name!r} (AC-6)"))
            return (f"error: tool {call.name!r} is not granted to this node "
                    f"and was not run; use only your available tools")
        missing = [p for p in required_by.get(call.name, []) if p not in call.arguments]
        if missing:
            trace.append(("tool_call", f"INVALID {call.name}: missing {missing}"))
            return (f"error: call to {call.name} is missing required argument(s) "
                    f"{missing}; fix the arguments and retry")
        trace.append(("tool_call", f"{call.name}({_short(call.arguments)})"))
        try:
            out = self._tools.dispatch(call.name, call.arguments, ctx=ctx, run=run)
        except ServiceError as exc:
            # D16 (`m3-executor.md` §2.2) — **log always, absorb only what the model can
            # fix.** Every failed dispatch is logged unconditionally here: the trace is not a
            # substitute, because `_trace_step` uses `_NULL_TRACER` unless `run["trace"]` is
            # set, so on a normal run the trace record never leaves the process.
            #
            # Then the split. A `ServiceError` in `MODEL_CORRECTABLE_TOOL_ERRORS` is a bad
            # *argument* the model chose (an unresolvable mention, a malformed search query),
            # exactly like the ungranted/malformed cases above, so it takes the same bounded
            # re-prompt: hand the error back and let the model correct itself. Everything
            # else is **not** the model's to fix — `UnknownActorError` comes from the
            # deployment's agent id, `ThreadNotFoundError`/`ChannelNotFoundError` from the
            # run's own `ctx` — and re-prompting on those would burn `maxIterations` and let
            # the run reach `done` having posted nothing (the AC-4 failure signature). Those
            # propagate to the M-1 fault net in `_drive` → `fail_run`, loudly. The allowlist
            # is deliberate: a `ServiceError` subclass added later fails loud by default.
            #
            # This is still deliberately NOT a blanket `except Exception`:
            #   * an engine fault (a driver/`Repository`/programming error) is never caught
            #     here at all — telling a model to "retry" a broken database would burn the
            #     budget and hide a real bug; and
            #   * `HumanHandoffSignal` is **control flow** raised *through* `dispatch`
            #     (§2.4) and must reach `_drive`'s suspend path — swallowing it here would
            #     silently break the handoff contract.
            # The re-prompt is bounded by `maxIterations` like any other turn (§7): a model
            # that keeps failing exhausts the node gracefully, it never spins.
            _log.warning("tool %s failed: %r", call.name, exc)
            if not isinstance(exc, MODEL_CORRECTABLE_TOOL_ERRORS):
                raise
            trace.append(("tool_result", f"ERROR: {type(exc).__name__}: {exc}"))
            return (f"error: {call.name} failed: {type(exc).__name__}: {exc}; "
                    f"fix the arguments and retry, or continue without this tool")
        trace.append(("tool_result", _short(out)))
        content = out if isinstance(out, str) else str(out)
        self._buffer_emission(content, emissions)
        return content

    @staticmethod
    def _buffer_emission(content: str, emissions: list[str]) -> None:
        """If a tool result is a `post_message` envelope (`{"posted": msgId, …}`), buffer
        the msgId for post-record PRODUCED linking. Non-post tools / non-JSON results are
        ignored — the executor owns audit linking, keeping the tool decoupled (Option B)."""
        obj = _load_json_obj(content)
        msg_id = obj.get("posted")
        if isinstance(msg_id, str) and msg_id:
            emissions.append(msg_id)

    def _read_thread_context(
        self, ctx: CallContext, run_ctx: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Fetch the recent thread transcript for the agent node (U11.3, AC-2 prereq).

        The intake node must see the human's replies to judge "enough info" — so the
        node reads the thread via the **service layer** (layering: the executor holds
        `self._services`, holds no Cypher). `read_thread` (QUERIES §4) is **thread-scoped**
        via `HEAD`/`NEXT*0..` — the workspace-wide channel caveat (K-015) does NOT apply.
        Absent `threadId` (offline unit tests) or no wired services → no read, so the
        network-free stub path is preserved. The window is capped app-side so a long
        thread cannot blow the prompt (RAM/latency hygiene, rule 6)."""
        thread_id = run_ctx.get("threadId")
        if not thread_id or self._services is None:
            return []
        msgs = self._services.read_thread(ctx, thread_id=thread_id)
        return msgs[-THREAD_CONTEXT_WINDOW:]

    @staticmethod
    def _assemble_messages(
        config: dict[str, Any], run_ctx: dict[str, Any],
        thread_msgs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build the node's opening message list (§2.2): the node `systemPrompt`, then the
        recent thread turns as conversation messages (role-mapped, speaker-named so the
        model sees who spoke), then a compact `CONTEXT` block carrying the run's serialized
        state (the prior nodes' output). Thread turns come pre-capped by
        `_read_thread_context`; an empty list (offline stub path) leaves only system+CONTEXT."""
        messages: list[dict[str, Any]] = []
        system = config.get("systemPrompt", "")
        if system:
            messages.append({"role": "system", "content": system})
        for m in thread_msgs:
            role = "assistant" if m.get("role") == "assistant" else "user"
            speaker = m.get("displayName") or m.get("authorId") or "member"
            messages.append(
                {"role": role, "content": f"{speaker}: {m.get('text', '')}"}
            )
        context = json.dumps(run_ctx, separators=(",", ":"), sort_keys=True)
        messages.append({"role": "user", "content": f"CONTEXT:\n{context}"})
        return messages

    @staticmethod
    def _required_params(schema: Any) -> list[str]:
        """The `parameters.required` list from a tool's function schema (or `[]`)."""
        if not isinstance(schema, dict):
            return []
        fn = schema.get("function", schema)
        params = fn.get("parameters", {}) if isinstance(fn, dict) else {}
        req = params.get("required", []) if isinstance(params, dict) else []
        return list(req) if isinstance(req, list) else []

    def _select_transition(
        self, transitions: list[dict[str, Any]], run: dict[str, Any],
        run_ctx: dict[str, Any], result: StepResult,
    ) -> _TransitionDecision:
        """Evaluate a step's outgoing transition guards; the first firing wins.

        Conditional guards are evaluated before the unconditional (empty-string)
        default — the empty guard is **lowest priority** (§2.5), firing only when no
        conditional guard matches. Within each class, `TRANSITION.order` breaks ties.
        Each guard is resolved by `guards.evaluate_guard` (empty → unconditional;
        `{"kind":"llm"}` → the injected judge with the bias-to-suspend policy;
        `{"kind":"cmp"|"all"|"any"|"not"}` → the deterministic comparator; any other
        `kind` → `NotImplementedError`, the M7 seam). Every *judged* guard contributes a
        `guard_judgment` to the trace (§5, M-6) — LLM and `cmp`-family alike; the
        unconditional default judges nothing and so traces nothing.

        The traced label is the guard's own `text` when it has one (LLM guards) and
        `guards.render_label` otherwise: a `cmp` guard carries no `text`, and
        `_trace_step`'s `f"{label} -> …"` payload would otherwise open with a bare
        `" -> "` and name nothing.
        """
        ordered = sorted(transitions, key=lambda t: (t["guard"] == "", t["order"]))
        judgments: list[tuple[dict[str, Any], str, GuardVerdict]] = []
        for tr in ordered:
            guard = tr["guard"]
            verdict = evaluate_guard(
                guard, ctx=run_ctx, run=run, step_output=result.output,
                thread=result.thread, judge=self._guard_judge,
            )
            parsed = _load_json_obj(guard)
            if parsed.get("kind") in TRACED_GUARD_KINDS:
                judgments.append(
                    (tr, parsed.get("text") or render_label(parsed), verdict)
                )
            if verdict.decision:
                return _TransitionDecision(firing=tr, judgments=judgments)
        return _TransitionDecision(firing=None, judgments=judgments)

    # ── write helpers ──────────────────────────────────────────────────────────

    def _record(
        self, ctx: CallContext, run: dict[str, Any], cur_key: str, to_key: str,
        result: StepResult,
    ) -> dict[str, Any]:
        """Atomically record the just-run step's `StepRun` and relink `AT_STEP` to
        `to_key` (§12.2). `to_key == cur_key` is the advance-to-self record used by
        the suspend / re-loop / terminal paths (records + bumps the budget without
        moving position). Zero rows = the run vanished mid-drive → not-found."""
        started = self._clock()
        rec = self._repo.record_step_and_advance(
            ctx.ws, run_id=run["runId"], step_run_id=self._id(),
            step_status="done", started_at=started, ended_at=self._clock(),
            input=run["ctx"], output=result.output,
            to_step_uid=self._uid(run, to_key),
        )
        if rec is None:
            raise WorkflowRunNotFoundError(run["runId"])
        return rec

    def _fail_budget(
        self, ctx: CallContext, run_id: str, run_ctx: dict[str, Any]
    ) -> str:
        """Transition the run to `failed` with a step-budget note stamped in `ctx`
        (§7 runaway guard)."""
        self._fail_with_note(ctx, run_id, run_ctx, "step budget exceeded")
        return "failed"

    def _fail_with_note(
        self, ctx: CallContext, run_id: str, run_ctx: dict[str, Any], error: str
    ) -> None:
        """Stamp the run `failed` with a diagnostic `error` note in `ctx` (§7/§12.5). Shared
        by the step-budget guard and the M-1 top-level fault net so both terminate a faulted
        run identically (AT_STEP cleared, a readable cause) — never a `running` zombie."""
        note = dict(run_ctx)
        note["error"] = error
        self._repo.fail_run(
            ctx.ws, run_id=run_id, ended_at=self._clock(),
            ctx=json.dumps(note, separators=(",", ":"), sort_keys=True),
        )

    def _trace_step(
        self, ctx: CallContext, tracer: Tracer, step_run_id: str,
        result: StepResult, decision: _TransitionDecision,
    ) -> None:
        """Emit the debug trace records for one executed step: a `node_rationale` for the
        execution, then the node's own collected events (llm prompts/responses, tool
        calls/results, any exhaustion note — U8), then one `guard_judgment` per judged
        guard (LLM **and** `cmp`-family, M-6). `seq` orders events within the StepRun
        (§12.10). A `NullTracer` no-ops (AC-5).

        The judgment payload is `"{label} -> {decision}: {rationale}"`. `_select_transition`
        guarantees a non-empty label for every judged guard; the empty-label branch here is
        the belt that keeps a trace line from ever opening with a bare `" -> "`."""
        seq = 0
        tracer.record(
            ctx.ws, step_run_id=step_run_id, seq=seq, kind="node_rationale",
            payload=result.output or "(no output)",
        )
        for kind, payload in result.trace:
            seq += 1
            tracer.record(
                ctx.ws, step_run_id=step_run_id, seq=seq, kind=kind, payload=payload,
            )
        for _tr, label, verdict in decision.judgments:
            seq += 1
            prefix = f"{label} -> " if label else ""
            tracer.record(
                ctx.ws, step_run_id=step_run_id, seq=seq, kind="guard_judgment",
                payload=f"{prefix}{verdict.decision}: {verdict.rationale}",
            )

    def _link_emissions(
        self, ctx: CallContext, step_run_id: str, msg_ids: list[str]
    ) -> None:
        """Link `StepRun -[:PRODUCED]-> Message` for each msgId the node posted (Option B,
        K-023). Mirrors `_trace_step`: emissions are buffered during execution and drained
        **after** `_record` created the StepRun, keyed to its real `stepRunId`. The link is
        the deliberately two-step, non-atomic second query (§3/§9) — a `None` return (a
        missing endpoint) is a diagnosable, retry-able gap that is logged, **never raised**:
        a missing audit link must not fail a run whose message already stands (the durable
        artifact)."""
        for msg_id in msg_ids:
            linked = self._services.link_step_emission(
                ctx, step_run_id=step_run_id, msg_id=msg_id
            )
            if linked is None:
                _log.warning(
                    "PRODUCED link gap: stepRun %s -> msg %s (endpoint missing)",
                    step_run_id, msg_id,
                )

    @staticmethod
    def _uid(run: dict[str, Any], step_key: str) -> str:
        """A `Step.stepUid` = `"{defKey}:{version}:{stepKey}"` (the §11 convention)."""
        return f"{run['defKey']}:{run['defVersion']}:{step_key}"
