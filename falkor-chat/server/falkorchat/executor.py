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
execution) and the fuzzy-guard judge prompt land in later units (U6–U8); here
`_execute_step` is a deterministic Phase-1 stub and the guard judge is an injected
callable.

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
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import CallContext
from .guards import GuardVerdict, evaluate_guard
from .repository import Repository, WorkflowRunNotFoundError

# Run-level step budget default (DS note Q4 / §7 — per-def default 12).
DEFAULT_STEP_BUDGET = 12
# Per-node tool-loop cap default (DS note Q4 / §7 — tool-light proof nodes = 4).
DEFAULT_MAX_ITERATIONS = 4
# Cap a trace payload at the write boundary (rule 6 — RAM). Matches MAX_CONFIG_LEN.
TRACE_PAYLOAD_MAX = 8000


# ── value objects ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StepResult:
    """The outcome of executing one step: its opaque `output`, emitted `on`, and the
    debug trace events the node collected during execution (empty for the offline stub).

    `trace` is a list of `(kind, payload)` the node accumulates while it runs (llm
    prompts/responses, tool calls/results, an exhaustion note). It is emitted by
    `_trace_step` **after** the StepRun exists (the record→trace order), so the events
    key to a real `stepRunId`. A `NullTracer` drops them all (AC-5)."""

    output: str = ""
    on: str = "done"
    trace: list[tuple[str, str]] = field(default_factory=list)


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

    def __init__(self, repo: Repository, *, id_gen: Callable[[], str],
                 clock: Callable[[], int]) -> None:
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


# ── defaults ─────────────────────────────────────────────────────────────────

def _default_id() -> str:
    return uuid.uuid4().hex


def _default_clock() -> int:
    return int(time.time() * 1000)


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
        as `judge=`: a callable `(condition, *, understanding, ctx, step_output) ->
        {decision, rationale}` (Q1 extract-then-judge). Only invoked for `{"kind":"llm"}`
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

    def resume(self, ctx: CallContext, *, run_id: str) -> str | None:
        """Resume a parked (`waiting`) run on a human reply. The `waiting→running`
        CAS is single-flight (§12.4): if it does not apply (the run is not waiting,
        or a concurrent reply already resumed it), returns `None` without driving.
        Otherwise drives the loop and returns the terminal status."""
        cas = self._repo.resume_run(ctx.ws, run_id=run_id)
        if cas is None:
            return None
        run = self._repo.get_run(ctx.ws, run_id=run_id)
        if run is None:  # pragma: no cover — resumed then vanished (TOCTOU tripwire)
            raise WorkflowRunNotFoundError(run_id)
        return self._drive(ctx, run)

    # ── the §2.1 loop ──────────────────────────────────────────────────────────

    def _drive(self, ctx: CallContext, run: dict[str, Any]) -> str:
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

            if firing is not None:
                # OUTCOME A — a guard fired: advance, then enforce the budget
                if rec["stepCount"] > max_steps:
                    return self._fail_budget(ctx, run_id, run_ctx)
                current_key = to_key
                continue

            if config.get("waitsForHuman"):
                # OUTCOME B — suspend (guaranteed human-unblockable, §2.4)
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

        Dispatches on `Step.type` (§2.3): a `type:'agent'` node with a wired LLM runs
        the bounded, tool-scoped agent loop (`_run_agent_node`, U8). Without a wired LLM
        (the offline loop-engine tests, D2) — or for any other type in this cut — it
        falls back to the deterministic no-op stub; the guard-based branching
        (`_select_transition`) is what drives the engine there, so an empty result
        exercises the whole loop without a network. The deterministic typed-step handler
        library (§2.3) is a later slice.
        """
        if step.get("type") == "agent" and self._llm is not None:
            return self._run_agent_node(ctx, run, step, config, run_ctx)
        return StepResult(output="", on="done")

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

        messages = self._assemble_messages(config, run_ctx)
        trace: list[tuple[str, str]] = []
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
                return StepResult(output=result.text or "", on="done", trace=trace)

            messages.append(_assistant_turn(result))
            for call in result.tool_calls:
                content = self._handle_tool_call(
                    call, granted_set, required_by, ctx, run, trace
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )

        # maxIterations exhausted — graceful termination (§7, DS Q4): best text + note.
        trace.append(
            ("node_note", f"max iterations ({max_iter}) reached; terminating node "
                          f"with best current text")
        )
        return StepResult(output=last_text, on="done", trace=trace)

    def _handle_tool_call(
        self, call: Any, granted_set: set[str],
        required_by: dict[str, list[str]], ctx: CallContext, run: dict[str, Any],
        trace: list[tuple[str, str]],
    ) -> str:
        """Validate + dispatch one tool call; return the message content fed back to the
        model. Enforces AC-6 (ungranted → reject) and arg-schema validity (malformed →
        refuse) — both a bounded re-prompt, never a dispatch."""
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
        out = self._tools.dispatch(call.name, call.arguments, ctx=ctx, run=run)
        trace.append(("tool_result", _short(out)))
        return out if isinstance(out, str) else str(out)

    @staticmethod
    def _assemble_messages(
        config: dict[str, Any], run_ctx: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Build the node's opening message list (§2.2): the node `systemPrompt` plus an
        assembled context message carrying the run's serialized state. Recent thread
        messages are folded in by the services/trigger layer when wired (a later unit); the
        offline node sees the run `ctx`."""
        messages: list[dict[str, Any]] = []
        system = config.get("systemPrompt", "")
        if system:
            messages.append({"role": "system", "content": system})
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
        `{"kind":"llm"}` → the injected judge with the bias-to-suspend policy; any other
        `kind` → `NotImplementedError`, the M7 seam). Only LLM-judged guards contribute a
        `guard_judgment` to the trace (§5) — the unconditional default judges nothing.
        """
        ordered = sorted(transitions, key=lambda t: (t["guard"] == "", t["order"]))
        judgments: list[tuple[dict[str, Any], str, GuardVerdict]] = []
        for tr in ordered:
            guard = tr["guard"]
            verdict = evaluate_guard(
                guard, ctx=run_ctx, run=run, step_output=result.output,
                thread=None, judge=self._guard_judge,
            )
            parsed = _load_json_obj(guard)
            if parsed.get("kind") == "llm":
                judgments.append((tr, parsed.get("text", ""), verdict))
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
        note = dict(run_ctx)
        note["error"] = "step budget exceeded"
        self._repo.fail_run(
            ctx.ws, run_id=run_id, ended_at=self._clock(),
            ctx=json.dumps(note, separators=(",", ":"), sort_keys=True),
        )
        return "failed"

    def _trace_step(
        self, ctx: CallContext, tracer: Tracer, step_run_id: str,
        result: StepResult, decision: _TransitionDecision,
    ) -> None:
        """Emit the debug trace records for one executed step: a `node_rationale` for the
        execution, then the node's own collected events (llm prompts/responses, tool
        calls/results, any exhaustion note — U8), then one `guard_judgment` per LLM guard
        judged. `seq` orders events within the StepRun (§12.10). A `NullTracer` no-ops (AC-5)."""
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
        for _tr, text, verdict in decision.judgments:
            seq += 1
            tracer.record(
                ctx.ws, step_run_id=step_run_id, seq=seq, kind="guard_judgment",
                payload=f"{text} -> {verdict.decision}: {verdict.rationale}",
            )

    @staticmethod
    def _uid(run: dict[str, Any], step_key: str) -> str:
        """A `Step.stepUid` = `"{defKey}:{version}:{stepKey}"` (the §11 convention)."""
        return f"{run['defKey']}:{run['defVersion']}:{step_key}"
