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

    def __call__(self, condition, *, understanding, ctx, step_output):
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
