"""Unit/integration tests for the offline workflow executor engine (U4).

The engine loop (§2.1 A/B/C outcomes, suspend/resume, step budget, done/fail) is
driven end-to-end against a live `ws:test` graph with **stub** handlers/guards —
no LLM, no network (D3, this landing is offline). The only injected decision seam
is `guard_judge` (a scripted stub); step execution is a Phase-1 stub. Tracing is
exercised on/off (AC-5) with the real `GraphTracer`/`NullTracer`.
"""

from __future__ import annotations

import itertools

import pytest

from falkorchat.config import CallContext
from falkorchat.executor import (
    GraphTracer,
    WorkflowExecutor,
)
from falkorchat.modelconfig import ModelResolutionError
from falkorchat.repository import WorkflowRunNotFoundError

CTX = CallContext(ws="test", actor="u1")


# ── stubs ────────────────────────────────────────────────────────────────────

class StubJudge:
    """A scripted LLM-guard judge (the `guards.evaluate_guard` `judge=` seam): returns
    the next scripted `{decision, rationale}` output (False when the script is exhausted).
    Records the guard conditions it was asked about."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls: list[str] = []

    def __call__(self, condition, *, understanding, recent_turns, ctx, step_output):
        self.calls.append(condition)
        decision = self._verdicts.pop(0) if self._verdicts else False
        return {"decision": decision, "rationale": f"stub verdict={decision}"}


def _make_executor(repo, *, guard_judge, tracer=None, step_budget=12):
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(1000)
    return WorkflowExecutor(
        None, repo, guard_judge=guard_judge, tracer=tracer,
        step_budget=step_budget,
        id_gen=lambda: next(ids), clock=lambda: next(clock),
    )


# ── fixtures: a materialized triage snapshot + trigger message + started run ──

TRIAGE_STEPS = [
    {"key": "intake", "type": "agent", "config": '{"waitsForHuman":true}'},
    {"key": "research", "type": "agent", "config": "{}"},
    {"key": "answer", "type": "agent", "config": "{}"},
]
TRIAGE_TRANSITIONS = [
    {"from": "intake", "to": "research", "on": "ready",
     "guard": '{"kind":"llm","text":"enough info?"}', "order": 0},
    {"from": "research", "to": "answer", "on": "done", "guard": "", "order": 0},
]


def _seed_thread(repo):
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x",
                       created_at=110)
    repo.ensure_user("test", user_id="u1", display_name="Alice")


def _start_run(repo, *, steps=TRIAGE_STEPS, transitions=TRIAGE_TRANSITIONS,
               start_key="intake", trace=False, max_steps=12, run_id="r1"):
    repo.materialize_snapshot(
        "test", key="triage", version="1", name="Triage", kind="conversation",
        start_key=start_key, steps=steps, transitions=transitions,
    )
    _seed_thread(repo)
    repo.post_first_message(
        "test", thread_id="t1", msg_id="trig1", author_id="u1",
        text="please help", role="user", created_at=120,
    )
    repo.start_run(
        "test", run_id=run_id, def_key="triage", def_version="1",
        started_at=1000, trigger_msg_id="trig1", ctx='{"threadId":"t1"}',
        trace=trace, max_steps=max_steps,
    )


# ── OUTCOME B — suspend on waitsForHuman + guard false ───────────────────────

def test_run_suspends_at_intake_when_guard_false(wf_repo):
    _start_run(wf_repo)
    ex = _make_executor(wf_repo, guard_judge=StubJudge([False]))

    status = ex.run(CTX, run_id="r1")

    assert status == "waiting"
    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "waiting"
    assert run["atStepKey"] == "intake"            # parked on intake, not advanced
    assert run["waitingThreadId"] == "t1"          # denormed from ctx for resume
    # the suspended execution is recorded (execute → record → branch)
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert [s["stepKey"] for s in trail] == ["intake"]


# ── OUTCOME A + C — full flow intake-wait → resume → research → answer → done ─

def test_full_flow_drives_to_done_with_complete_audit_trail(wf_repo):
    _start_run(wf_repo)
    judge = StubJudge([False, True])           # 1st (initial): suspend; 2nd: advance
    ex = _make_executor(wf_repo, guard_judge=judge)

    first = ex.run(CTX, run_id="r1")           # → waiting
    second = ex.resume(CTX, run_id="r1")       # → drives to done

    assert first == "waiting"
    assert second == "done"
    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "done"
    assert run["atStepKey"] is None            # AT_STEP cleared on terminal
    # per-execution audit trail: intake ran twice (parked, then advanced),
    # then research (unconditional D5), then answer (terminal → done)
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert [s["stepKey"] for s in trail] == ["intake", "intake", "research", "answer"]
    # only the intake→research guard is LLM-judged; research→answer is unconditional
    assert judge.calls == ["enough info?", "enough info?"]


# ── OUTCOME C — step-budget abort on an autonomous self-loop ──────────────────

SPIN_STEPS = [{"key": "spin", "type": "agent", "config": "{}"}]  # no waitsForHuman
SPIN_TRANSITIONS = [
    {"from": "spin", "to": "spin", "on": "again",
     "guard": '{"kind":"llm","text":"done yet?"}', "order": 0},
]


def test_step_budget_abort_fails_the_run(wf_repo):
    # a non-waiting node whose guard never fires re-loops (outcome C) until the
    # run-level step budget trips → status failed (§7 runaway guard)
    _start_run(wf_repo, steps=SPIN_STEPS, transitions=SPIN_TRANSITIONS,
               start_key="spin", max_steps=3)
    ex = _make_executor(wf_repo, guard_judge=StubJudge([]), step_budget=3)  # always False

    status = ex.run(CTX, run_id="r1")

    assert status == "failed"
    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "failed"
    assert run["atStepKey"] is None
    assert "step budget exceeded" in run["ctx"]
    # stepCount ran one past the budget, then failed
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert len(trail) == 4  # maxSteps=3 → the 4th advance trips the guard


# ── unconditional guard is the lowest-priority default ────────────────────────

PRIORITY_STEPS = [
    {"key": "pick", "type": "agent", "config": "{}"},
    {"key": "viaDefault", "type": "agent", "config": "{}"},
    {"key": "viaJudged", "type": "agent", "config": "{}"},
]
PRIORITY_TRANSITIONS = [
    # unconditional has the LOWER order number, but must still lose to a firing
    # conditional guard (unconditional = lowest priority, §2.5)
    {"from": "pick", "to": "viaDefault", "on": "d", "guard": "", "order": 0},
    {"from": "pick", "to": "viaJudged", "on": "j",
     "guard": '{"kind":"llm","text":"go?"}', "order": 1},
]


def test_firing_conditional_guard_beats_lower_order_unconditional(wf_repo):
    _start_run(wf_repo, steps=PRIORITY_STEPS, transitions=PRIORITY_TRANSITIONS,
               start_key="pick")
    ex = _make_executor(wf_repo, guard_judge=StubJudge([True]))  # the judged guard fires

    status = ex.run(CTX, run_id="r1")

    assert status == "done"
    trail = wf_repo.read_step_runs("test", run_id="r1")
    # advanced pick → viaJudged (conditional won), never viaDefault
    assert [s["stepKey"] for s in trail] == ["pick", "viaJudged"]


def test_unconditional_fires_when_no_conditional_guard_matches(wf_repo):
    _start_run(wf_repo, steps=PRIORITY_STEPS, transitions=PRIORITY_TRANSITIONS,
               start_key="pick")
    ex = _make_executor(wf_repo, guard_judge=StubJudge([False]))  # judged guard fails

    status = ex.run(CTX, run_id="r1")

    assert status == "done"
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert [s["stepKey"] for s in trail] == ["pick", "viaDefault"]  # fell through


# ── K-034 defense-in-depth — `_select_transition`'s tie-break is deterministic ─
#
# A def with two outgoing `TRANSITION`s sharing `(from, on, order)` but different
# `to` is unreachable through the sanctioned write path after the K-034 gate, but
# stays reachable via a direct `materialize_snapshot` call (as this fixture does,
# same as `_start_run` above) or pre-existing corrupted data. `_select_transition`
# sorted only by `(guard == "", order)`, so two equal-priority candidates sorted
# equal and Python's stable sort just preserved whatever order the list arrived
# in — standing in for FalkorDB's unpinned edge-retrieval order. Supplying the
# same two transitions in both orders must land on the same `to` either way.

TIEBREAK_STEPS = [
    {"key": "pick", "type": "agent", "config": "{}"},
    {"key": "viaA", "type": "agent", "config": "{}"},
    {"key": "viaB", "type": "agent", "config": "{}"},
]
TIEBREAK_TRANSITIONS = [
    {"from": "pick", "to": "viaB", "on": "x", "guard": "", "order": 0},
    {"from": "pick", "to": "viaA", "on": "x", "guard": "", "order": 0},
]


@pytest.mark.parametrize("transitions", [TIEBREAK_TRANSITIONS, list(reversed(TIEBREAK_TRANSITIONS))])
def test_select_transition_tie_break_is_deterministic_regardless_of_order(
    wf_repo, transitions
):
    _start_run(wf_repo, steps=TIEBREAK_STEPS, transitions=transitions,
               start_key="pick")
    ex = _make_executor(wf_repo, guard_judge=StubJudge([]))

    status = ex.run(CTX, run_id="r1")

    assert status == "done"
    trail = wf_repo.read_step_runs("test", run_id="r1")
    # deterministic regardless of input order — `to` is the final tie-break
    assert [s["stepKey"] for s in trail] == ["pick", "viaA"]


# ── AC-5 — tracing on vs off ─────────────────────────────────────────────────

def test_debug_run_records_trace_events(wf_repo):
    _start_run(wf_repo, trace=True)
    tracer = GraphTracer(
        wf_repo, id_gen=(lambda c=itertools.count(1): f"te{next(c)}"),
        clock=(lambda c=itertools.count(9000): next(c)),
    )
    judge = StubJudge([False, True])
    ex = _make_executor(wf_repo, guard_judge=judge, tracer=tracer)

    ex.run(CTX, run_id="r1")
    ex.resume(CTX, run_id="r1")

    events = wf_repo.read_trace("test", run_id="r1")
    kinds = {e["kind"] for e in events}
    assert len(events) > 0
    assert "guard_judgment" in kinds        # the LLM guard verdicts are traced
    assert "node_rationale" in kinds


def test_non_debug_run_records_zero_trace_events(wf_repo):
    # same flow, trace=False → NullTracer by construction → zero TraceEvents (AC-5)
    _start_run(wf_repo, trace=False)
    tracer = GraphTracer(
        wf_repo, id_gen=(lambda c=itertools.count(1): f"te{next(c)}"),
        clock=(lambda c=itertools.count(9000): next(c)),
    )
    ex = _make_executor(wf_repo, guard_judge=StubJudge([False, True]), tracer=tracer)

    ex.run(CTX, run_id="r1")
    ex.resume(CTX, run_id="r1")

    assert wf_repo.read_trace("test", run_id="r1") == []


# ── resume / error edges ─────────────────────────────────────────────────────

def test_resume_of_non_waiting_run_returns_none_without_driving(wf_repo):
    _start_run(wf_repo)  # status = running, not waiting
    ex = _make_executor(wf_repo, guard_judge=StubJudge([True]))

    assert ex.resume(CTX, run_id="r1") is None    # CAS miss → no drive
    assert wf_repo.read_step_runs("test", run_id="r1") == []


def test_run_missing_raises_not_found(wf_repo):
    ex = _make_executor(wf_repo, guard_judge=StubJudge([]))
    with pytest.raises(WorkflowRunNotFoundError):
        ex.run(CTX, run_id="ghost")


# ── M-1 — an unexpected mid-drive exception fails the run (no zombie `running`) ─

class _BoomExecutor(WorkflowExecutor):
    """Raises inside the drive loop (at step execution) to exercise the M-1 net."""

    def _execute_step(self, ctx, run, step, config, run_ctx):
        raise RuntimeError("boom in a step handler")


def _boom_executor(repo, **over):
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(1000)
    return _BoomExecutor(
        None, repo, guard_judge=StubJudge([]),
        id_gen=lambda: next(ids), clock=lambda: next(clock), **over,
    )


def test_unexpected_exception_fails_the_run_and_reraises(wf_repo):
    # M-1: an unexpected exception mid-drive must leave the run `failed` (AT_STEP
    # cleared, a diagnostic ctx note) and re-raise — never a stuck `running` zombie.
    _start_run(wf_repo)
    ex = _boom_executor(wf_repo)

    with pytest.raises(RuntimeError):
        ex.run(CTX, run_id="r1")

    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "failed"
    assert run["atStepKey"] is None
    assert "boom in a step handler" in run["ctx"]


def test_llm_guard_without_judge_fails_the_run_with_named_error(wf_repo):
    # m-3 via the M-1 net: an llm guard reached with guard_judge=None raises the named
    # WorkflowConfigError, which the net converts into a `failed` run carrying the message.
    _start_run(wf_repo)  # intake→research guard is {kind:llm}
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(1000)
    ex = WorkflowExecutor(
        None, wf_repo, guard_judge=None,        # no judge wired
        id_gen=lambda: next(ids), clock=lambda: next(clock),
    )

    from falkorchat.guards import WorkflowConfigError
    with pytest.raises(WorkflowConfigError):
        ex.run(CTX, run_id="r1")

    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "failed"
    assert "judge" in run["ctx"].lower()


# ── K-042 Landing 2 (L2-5, FR-10): an unresolvable model at drive time ────────
#
# Mostly *pinning*: `_drive`'s existing M-1 net (`except Exception as exc:
# self._fail_with_note(...); raise`) already catches a `ModelResolutionError`
# raised anywhere inside `_drive_loop`, same as any other engine fault (the
# `NotImplementedError`/`WorkflowConfigError` cases above). These two tests prove
# it does so for THIS exception type specifically, with the unresolvable
# identifier readable in the failure and AT_STEP cleared — no production code
# change was needed for this half.

class _UnresolvableGateway:
    """A `ModelGateway`-shaped double whose `.resolve_llm()` raises
    `ModelResolutionError` naming the unresolvable ref — the same "unknown
    provider" unresolvable-thing `test_modelconfig.py`'s
    `test_unknown_provider_raises_resolution_error` already covers at the
    resolver layer. This double proves that failure reaches the drive-level M-1
    net and terminates the run, rather than duplicating the resolver-level
    assertion."""

    def has_chat(self) -> bool:
        return True

    def resolve_llm(self, kind, *, requested=None, ws=None, overrides=None):
        raise ModelResolutionError(f"unknown provider for ref {requested!r}")

    def llm(self, kind, *, requested=None, ws=None, overrides=None):
        raise AssertionError("llm() should never be reached — resolve_llm() failed first")

    def embedder(self, kind, *, requested=None, ws=None, overrides=None):
        raise AssertionError("embedder() should never be called by an agent step")


UNRESOLVABLE_STEPS = [
    {"key": "intake", "type": "agent",
     "config": '{"model":"nope/thing-that-does-not-exist"}'},
    # never reached — `intake` raises before transition selection; present only
    # because an empty `transitions` list trips the empty-UNWIND row-collapse
    # quirk (`claude/graph-dba/falkordb-quirks.md`) in the snapshot-publish query.
    {"key": "sink", "type": "agent", "config": "{}"},
]
UNRESOLVABLE_TRANSITIONS = [
    {"from": "intake", "to": "sink", "on": "done", "guard": "", "order": 0},
]


def _start_unresolvable_run(repo, **over):
    _start_run(
        repo, steps=UNRESOLVABLE_STEPS, transitions=UNRESOLVABLE_TRANSITIONS,
        start_key="intake", **over,
    )


def test_unresolvable_model_ref_at_drive_time_fails_the_run_with_identifier_in_message(
    wf_repo,
):
    _start_unresolvable_run(wf_repo)
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(1000)
    ex = WorkflowExecutor(
        None, wf_repo, guard_judge=StubJudge([]), models=_UnresolvableGateway(),
        id_gen=lambda: next(ids), clock=lambda: next(clock),
    )

    with pytest.raises(ModelResolutionError) as excinfo:
        ex.run(CTX, run_id="r1")

    assert "nope/thing-that-does-not-exist" in str(excinfo.value)

    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "failed"
    assert run["atStepKey"] is None            # AT_STEP cleared, per fail_run
    assert "nope/thing-that-does-not-exist" in run["ctx"]  # identifier readable in the failure


def test_unresolvable_model_ref_uses_no_fallback_model(wf_repo):
    # Distinct from a fallback-EXHAUSTION scenario (`test_modelconfig.py`'s
    # `FallbackClient` tests, U8/L2-1: a chain of >=1 elements IS tried and every
    # element raises `ProviderCallError` before the chain gives up). Here there is
    # no chain at all — `resolve_llm()` itself raises before naming/trying any
    # model — so nothing downstream of it ever runs, no LLM client is ever handed
    # back, and no StepRun is ever recorded (the step never completed a single
    # execution). This is the "no other model was used" half of L2-5's own
    # done-condition, kept as an explicitly separate test so it does not read as
    # a duplicate of the fallback-exhaustion coverage.
    _start_unresolvable_run(wf_repo)
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(1000)
    ex = WorkflowExecutor(
        None, wf_repo, guard_judge=StubJudge([]), models=_UnresolvableGateway(),
        id_gen=lambda: next(ids), clock=lambda: next(clock),
    )

    with pytest.raises(ModelResolutionError):
        ex.run(CTX, run_id="r1")

    # no StepRun exists at all — the failure happened before any model answered,
    # so there is nothing for a fallback (or any other model) to have produced
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert trail == []


# ── Defect B — a failing tool must not kill the run (drive level) ─────────────

class _FailingToolLLM:
    """First turn: call `post_message` with a hallucinated displayName mention.
    Every later turn: a plain final text (the model "recovers" after the re-prompt)."""

    def __init__(self):
        self.turns = 0

    def chat(self, messages, tools):
        from falkorchat.llm import ChatResult, ToolCall
        self.turns += 1
        if self.turns == 1:
            return ChatResult(text="", tool_calls=[
                ToolCall("c1", "post_message",
                         {"text": "hello", "mentions": ["alice"]})])
        return ChatResult(text="node answer")


class _MentionRejectingRegistry:
    """`post_message` rejects the hallucinated displayName mention, as the §4 write does."""

    SCHEMA = {"type": "function",
              "function": {"name": "post_message",
                           "parameters": {"type": "object", "properties": {},
                                          "required": []}}}

    def schema(self, name):
        return self.SCHEMA

    def dispatch(self, name, arguments, *, ctx, run):
        from falkorchat.services import UnknownMemberError
        raise UnknownMemberError(["alice"])


TOOL_STEPS = [
    {"key": "answer", "type": "agent",
     "config": '{"tools":["post_message"],"maxIterations":4}'},
    # terminal step the *node* loop must not derail: it was `type:"task"` until K-024 U2
    # made an unimplemented type a raising seam (D-E), which would have killed this
    # Defect-B pin. `agent` keeps the step's role (it runs no tool and posts nothing) and
    # its assertions unchanged.
    {"key": "end", "type": "agent", "config": "{}"},
]
TOOL_TRANSITIONS = [
    {"from": "answer", "to": "end", "on": "done", "guard": "", "order": 0},
]


def test_hallucinated_mention_does_not_fail_the_run(wf_repo):
    # Defect B end-to-end through the real §2.1 drive loop: the model passes a
    # displayName as a mention, the tool raises UnknownMemberError — the run must still
    # reach `done`. Before the fix this propagated to the M-1 net → status `failed`.
    _start_run(wf_repo, steps=TOOL_STEPS, transitions=TOOL_TRANSITIONS,
               start_key="answer", trace=True)
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(1000)
    tracer = GraphTracer(
        wf_repo, id_gen=(lambda c=itertools.count(1): f"te{next(c)}"),
        clock=(lambda c=itertools.count(9000): next(c)),
    )
    ex = WorkflowExecutor(
        None, wf_repo, llm=_FailingToolLLM(),
        tool_registry=_MentionRejectingRegistry(), guard_judge=StubJudge([]),
        tracer=tracer, id_gen=lambda: next(ids), clock=lambda: next(clock),
    )

    status = ex.run(CTX, run_id="r1")

    assert status == "done"
    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "done"
    # the step was recorded — and, because the node survived, its trace was NOT lost
    # (the observability gap: _trace_step runs after _record, which never happened).
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert [s["stepKey"] for s in trail] == ["answer", "end"]
    events = wf_repo.read_trace("test", run_id="r1")
    assert any(e["kind"] == "tool_result" and e["payload"].startswith("ERROR:")
               for e in events)


def test_human_handoff_signal_suspends_the_run(wf_repo):
    # A granted human_handoff tool raises HumanHandoffSignal; the M-1 net parks the run
    # as `waiting` (not `failed`), reusing the intake suspend/resume mechanics (§2.4).
    from falkorchat.tools import HumanHandoffSignal

    class _HandoffExecutor(WorkflowExecutor):
        def _execute_step(self, ctx, run, step, config, run_ctx):
            raise HumanHandoffSignal("a human is needed")

    _start_run(wf_repo)
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(1000)
    ex = _HandoffExecutor(
        None, wf_repo, guard_judge=StubJudge([]),
        id_gen=lambda: next(ids), clock=lambda: next(clock),
    )

    status = ex.run(CTX, run_id="r1")

    assert status == "waiting"
    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "waiting"
    assert run["waitingThreadId"] == "t1"       # denormed for the resume lookup


# ── K-042 Landing 2 (L2-2, FR-8): resolvedModel/modelSource/modelFallback reach the
# graph end-to-end — `_run_agent_node` → `StepResult` → `_record` →
# `record_step_and_advance` → `read_step_runs`, driven through a real `ex.run(...)`
# against a live `ws:test` graph (not a fake repo double).

_SOLO_STEPS = [
    {"key": "solo", "type": "agent", "config": "{}"},
    {"key": "end", "type": "agent", "config": "{}"},  # terminal — no outgoing
]
# A snapshot needs >=1 transition (`materialize_snapshot`'s `UNWIND` over an empty
# transitions list collapses the whole write to zero rows — the empty-UNWIND
# row-collapse quirk, `falkor-chat/AGENTS.md`), so "solo" is not itself terminal;
# "end" (no outgoing) is where the run actually completes, OUTCOME C.
_SOLO_TRANSITIONS = [{"from": "solo", "to": "end", "on": "done", "guard": "", "order": 0}]


class _ModelledLLM:
    """A minimal chat stub whose `ChatResult` carries `model`/`fallback` — the FR-8
    carriers `OpenAICompatibleLLM`/`FallbackClient` set for real."""

    def __init__(self, model: str, *, fallback: bool | None = None):
        self._model = model
        self._fallback = fallback

    def chat(self, messages, tools):
        from falkorchat.llm import ChatResult
        return ChatResult(text="node answer", model=self._model, fallback=self._fallback)


def test_resolved_model_and_fallback_reach_the_stored_step_run_end_to_end(wf_repo):
    _start_run(wf_repo, steps=_SOLO_STEPS, transitions=_SOLO_TRANSITIONS, start_key="solo")
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(1000)
    ex = WorkflowExecutor(
        None, wf_repo, llm=_ModelledLLM("lmstudio/qwen3-4b", fallback=True),
        guard_judge=StubJudge([]), id_gen=lambda: next(ids), clock=lambda: next(clock),
    )

    status = ex.run(CTX, run_id="r1")

    assert status == "done"
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert [s["stepKey"] for s in trail] == ["solo", "end"]
    assert trail[0]["resolvedModel"] == "lmstudio/qwen3-4b"
    assert trail[0]["modelSource"] == "default"     # the step named no model of its own
    assert trail[0]["modelFallback"] is True


def test_non_llm_offline_stub_step_leaves_model_fields_absent_end_to_end(wf_repo):
    # No `llm=` injected ⇒ StaticModelGateway.has_chat() is False ⇒ the deliberate
    # offline-stub path (§2.3 executor.py docstring) — no chat() call is ever made,
    # so all three fields must read back None, not some other default.
    _start_run(wf_repo, steps=_SOLO_STEPS, transitions=_SOLO_TRANSITIONS, start_key="solo")
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(1000)
    ex = WorkflowExecutor(
        None, wf_repo, guard_judge=StubJudge([]),
        id_gen=lambda: next(ids), clock=lambda: next(clock),
    )

    status = ex.run(CTX, run_id="r1")

    assert status == "done"
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert len(trail) == 2
    for step_run in trail:
        assert step_run["resolvedModel"] is None
        assert step_run["modelSource"] is None
        assert step_run["modelFallback"] is None
