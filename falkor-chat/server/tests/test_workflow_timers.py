"""K-028 v3 — workflow timers / scheduled wakeups: `_wait_due_at`'s pure due-ness
computation and `Services.sweep_due_workflow_runs`'s batch-resume flow
(`docs/plans/workflow-timers.md` §5 steps 2/4, §6 tests 1/4).

Publish-time validation (the new fifth `_validate_def_spec` invariant) has its
own tests in `test_executor_process.py` (test 2, the existing home for that
validator's shape-matrix tests); `find_due_wait_candidates`'s repository-level
tests live in `test_repository.py` (test 3, next to the §12.3/§12.4 CAS
tests). This file covers what's left.

**The v3 marker-guard mechanism, in one paragraph** (`docs/plans/workflow-timers.md`
§3.3/§3.5): a timer-bearing `wait`/`human` step must declare an outgoing
transition guarded on `{"kind":"cmp","path":"ctx.timerFired","op":"eq",
"value":"<its own key>"}`. `timerFired` is reserved (`RESERVED_CTX_KEYS`), so no
human/API caller can ever set or spoof it — the guard is therefore false at
first arrival (the step correctly parks) and false on any ordinary or
"not yet" resume, becoming true only when `sweep_due_workflow_runs` itself
wins the resume CAS and writes `ctx.timerFired = step["key"]` atomically via
`resume_run_with_ctx` (§12.13). Unlike v2's defective "bare unconditional
fallback" invariant (which made a timer-bearing step never park at all — see
`services.py`'s inline comment on the check and this unit's `coder` report),
this is a **genuinely conditional** guard, so every test below drives a def
through the real, unmodified engine (`publish_workflow_def` → `materialize_def`
→ `start_workflow_run`) exactly as a production def would be authored — no
fixture needs to bypass publish validation or write `waiting` graph state
directly.
"""

from __future__ import annotations

import itertools
import json

import pytest

from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.schemas import MAX_CONFIG_LEN
from falkorchat.services import (
    WAITING_STEP_TYPES,
    Services,
    WorkflowRunNotWaitingError,
    _wait_due_at,
)

CTX = CallContext(ws="test", actor="u1")

VERSION = "v1"


def _escalation_guard(step_key: str) -> dict:
    """The canonical v3 escalation guard for `step_key`."""
    return {"kind": "cmp", "path": "ctx.timerFired", "op": "eq", "value": step_key}


def _ctx_of(services, run_id: str) -> dict:
    return json.loads(services.get_workflow_run(CTX, run_id=run_id)["ctx"])


# ── §6 test 1 — `_wait_due_at` pure-function tests (no repo, no clock) ───────

@pytest.mark.parametrize("step_type", ["decision", "agent", "prompt", "tool"])
def test_wait_due_at_none_for_a_non_waiting_step_type_regardless_of_config(
    step_type,
):
    assert _wait_due_at(
        step_type, {"waitForSeconds": 60}, 1000
    ) is None


@pytest.mark.parametrize("step_type", sorted(WAITING_STEP_TYPES))
def test_wait_due_at_none_when_neither_key_is_declared(step_type):
    # Today's forever-park def — the additive-only acceptance case.
    assert _wait_due_at(step_type, {"waitsForHuman": True}, 1000) is None
    assert _wait_due_at(step_type, {}, 1000) is None


def test_wait_due_at_computes_relative_due_time_for_a_wait_step():
    assert _wait_due_at("wait", {"waitForSeconds": 60}, 1000) == 61000


def test_wait_due_at_treats_human_and_wait_identically():
    # v2 scope (Finding 2) — would have silently regressed if the helper still
    # hard-coded `== "wait"`.
    assert _wait_due_at("human", {"waitForSeconds": 60}, 1000) == 61000


def test_wait_due_at_absolute_form_ignores_parked_at():
    assert _wait_due_at("wait", {"waitUntil": 5000}, parked_at=None) == 5000
    assert _wait_due_at("wait", {"waitUntil": 5000}, parked_at=999_999) == 5000


def test_wait_due_at_none_when_parked_at_is_missing_for_the_relative_form():
    assert _wait_due_at("wait", {"waitForSeconds": 60}, parked_at=None) is None


def test_wait_due_at_none_for_a_config_that_does_not_normalize_to_a_dict():
    assert _wait_due_at("wait", "raw-string", parked_at=1000) is None
    assert _wait_due_at("wait", None, parked_at=1000) is None


def test_wait_due_at_accepts_a_json_string_config_like_a_real_graph_read():
    # `Step.config` comes back from the graph as an opaque string (rule 8) —
    # `_wait_due_at` must normalize it exactly like every other config reader.
    cfg = json.dumps({"waitForSeconds": 30}, separators=(",", ":"))
    assert _wait_due_at("human", cfg, parked_at=1000) == 31000


# ── §6 test 4 — `sweep_due_workflow_runs`, against a real executor ──────────

class _SharedClock:
    """A single, settable clock passed to BOTH `Services` and
    `WorkflowExecutor` (v2 Finding 3, `docs/reviews/workflow-timers.md`) —
    `Services`/`WorkflowExecutor` hold two independently-defaulting clocks
    (`services.py:_default_clock`, `executor.py:_default_clock`) with nothing
    wiring one to the other in production either. `parkedAt` (minted by the
    executor's own monotonic wrapper around this clock) and `now`
    (`Services._clock()`, read directly) must be comparable numbers for the
    before/after due-time assertions below — sharing this one object is what
    makes them so. Deliberately NOT `test_executor_process.py`'s
    `_make_executor` pattern, which gives its executor an independent
    `itertools.count(1000)` clock disconnected from any `Services`.
    """

    def __init__(self, start: int) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now

    def advance(self, delta: int) -> None:
        self.now += delta


def _make_services(repo, clock, *, id_prefix="id"):
    ids = (f"{id_prefix}{n}" for n in itertools.count(1))
    services = Services(repo, clock=clock, id_gen=lambda: next(ids))
    sr_ids = (f"{id_prefix}sr{n}" for n in itertools.count(1))
    services.set_executor(WorkflowExecutor(
        services, repo, llm=None, guard_judge=None,
        id_gen=lambda: next(sr_ids), clock=clock,
    ))
    return services


def _publish_and_start(services, *, key, steps, transitions, run_ctx=None,
                        max_steps=12):
    """Publish + materialize + start an untriggered (`kind:'process'`) run —
    the real engine path, exercising the v3 publish-time invariant for real."""
    services.publish_workflow_def(
        CTX, key=key, version=VERSION, name=key, kind="process",
        steps=steps, transitions=transitions,
    )
    services.materialize_def(CTX, key=key, version=VERSION)
    return services.start_workflow_run(
        CTX, def_key=key, version=VERSION, run_ctx=run_ctx or {},
        max_steps=max_steps,
    )


@pytest.mark.parametrize("step_type", sorted(WAITING_STEP_TYPES))
def test_sweep_skips_a_candidate_before_its_due_time_and_resumes_it_after(
    wf_repo, step_type,
):
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "park", "type": step_type, "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60}},
        {"key": "activate", "type": "decision", "config": {}},
    ]
    transitions = [
        {"from": "park", "to": "activate", "on": "timeout", "order": 0,
         "guard": _escalation_guard("park")},
    ]
    started = _publish_and_start(
        services, key=f"timers-{step_type}", steps=steps, transitions=transitions,
    )
    run_id = started["runId"]
    assert started["status"] == "waiting"

    # Before due (59s later): nothing happens, and — v3 — no `timerFired` marker
    # is ever written for a candidate that isn't due yet.
    clock.advance(59_000)
    out = services.sweep_due_workflow_runs(CTX, limit=10)
    assert out["checked"] == 1
    assert out["due"] == 0
    assert out["resumed"] == []
    run = services.get_workflow_run(CTX, run_id=run_id)
    assert run["status"] == "waiting"
    assert "timerFired" not in _ctx_of(services, run_id)

    # Past due (60s+ from parkedAt): the sweep resumes it via the escalation
    # guard, and — v3 — `ctx.timerFired` now equals the parked step's own key,
    # proving the marker mechanism actually worked end to end, not just the
    # dueness comparison.
    clock.advance(2_000)
    out = services.sweep_due_workflow_runs(CTX, limit=10)
    assert out["checked"] == 1
    assert out["due"] == 1
    assert out["resumed"] == [{"runId": run_id, "status": "done"}]
    assert out["raced"] == []
    assert out["faulted"] == []
    run = services.get_workflow_run(CTX, run_id=run_id)
    assert run["status"] == "done"
    assert run["atStepKey"] is None
    assert _ctx_of(services, run_id)["timerFired"] == "park"


def test_sweep_ignores_a_waiting_run_whose_step_declares_no_timer_key(wf_repo):
    # The additive-only guarantee, end-to-end (not just at the pure-function
    # level, §6 test 1): today's forever-park def is never swept, at any `now`.
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "park", "type": "wait", "start": True,
         "config": {"waitsForHuman": True, "signal": "provisioned"}},
        {"key": "activate", "type": "decision", "config": {}},
    ]
    transitions = [
        {"from": "park", "to": "activate", "on": "signalled", "order": 0,
         "guard": {"kind": "cmp", "path": "ctx.provisioned", "op": "truthy"}},
    ]
    started = _publish_and_start(
        services, key="timers-no-key", steps=steps, transitions=transitions,
    )
    run_id = started["runId"]

    clock.advance(10 ** 9)  # far in the future — would be "due" if it had a timer
    out = services.sweep_due_workflow_runs(CTX, limit=10)

    assert out["checked"] == 1
    assert out["due"] == 0
    assert out["resumed"] == []
    assert services.get_workflow_run(CTX, run_id=run_id)["status"] == "waiting"


def test_sweep_respects_the_limit(wf_repo):
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "park", "type": "wait", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60}},
        {"key": "activate", "type": "decision", "config": {}},
    ]
    transitions = [
        {"from": "park", "to": "activate", "on": "timeout", "order": 0,
         "guard": _escalation_guard("park")},
    ]
    services.publish_workflow_def(
        CTX, key="timers-limit", version=VERSION, name="Timers", kind="process",
        steps=steps, transitions=transitions,
    )
    services.materialize_def(CTX, key="timers-limit", version=VERSION)
    run_ids = [
        services.start_workflow_run(
            CTX, def_key="timers-limit", version=VERSION, run_ctx={}, max_steps=12,
        )["runId"]
        for _ in range(3)
    ]
    clock.advance(120_000)  # all three are due

    out = services.sweep_due_workflow_runs(CTX, limit=2)

    assert out["checked"] == 2
    assert len(out["resumed"]) == 2
    remaining = [
        rid for rid in run_ids
        if services.get_workflow_run(CTX, run_id=rid)["status"] == "waiting"
    ]
    assert len(remaining) == 1

    # A second call with the same clock sweeps the rest.
    out2 = services.sweep_due_workflow_runs(CTX, limit=2)
    assert out2["checked"] == 1
    assert len(out2["resumed"]) == 1


def test_sweep_faults_a_due_run_whose_escalation_branch_reaches_an_unimplemented_step(
    wf_repo,
):
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "park", "type": "wait", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60}},
        {"key": "boom", "type": "prompt", "config": {}},  # unimplemented (D-E)
    ]
    transitions = [
        {"from": "park", "to": "boom", "on": "timeout", "order": 0,
         "guard": _escalation_guard("park")},
    ]
    started = _publish_and_start(
        services, key="timers-fault", steps=steps, transitions=transitions,
    )
    run_id = started["runId"]
    clock.advance(120_000)

    out = services.sweep_due_workflow_runs(CTX, limit=10)

    assert out["due"] == 1
    assert out["resumed"] == []
    assert out["raced"] == []
    assert len(out["faulted"]) == 1
    assert out["faulted"][0]["runId"] == run_id
    assert "NotImplementedError" in out["faulted"][0]["error"]
    run = services.get_workflow_run(CTX, run_id=run_id)
    assert run["status"] == "failed"
    assert run["atStepKey"] is None


def test_sweep_batch_isolation_a_fault_on_one_candidate_does_not_stop_the_rest(
    wf_repo,
):
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)

    # Candidate A: due, escalation branch reaches an unimplemented step (faults).
    a = _publish_and_start(
        services, key="a",
        steps=[
            {"key": "park", "type": "wait", "start": True,
             "config": {"waitsForHuman": True, "waitForSeconds": 60}},
            {"key": "boom", "type": "prompt", "config": {}},
        ],
        transitions=[
            {"from": "park", "to": "boom", "on": "timeout", "order": 0,
             "guard": _escalation_guard("park")},
        ],
    )
    # Candidate B: due, escalation branch reaches a valid terminal step.
    b = _publish_and_start(
        services, key="b",
        steps=[
            {"key": "park", "type": "wait", "start": True,
             "config": {"waitsForHuman": True, "waitForSeconds": 30}},
            {"key": "activate", "type": "decision", "config": {}},
        ],
        transitions=[
            {"from": "park", "to": "activate", "on": "timeout", "order": 0,
             "guard": _escalation_guard("park")},
        ],
    )
    clock.advance(120_000)

    out = services.sweep_due_workflow_runs(CTX, limit=10)

    assert out["due"] == 2
    assert [f["runId"] for f in out["faulted"]] == [a["runId"]]
    assert [r["runId"] for r in out["resumed"]] == [b["runId"]]
    assert services.get_workflow_run(CTX, run_id=a["runId"])["status"] == "failed"
    assert services.get_workflow_run(CTX, run_id=b["runId"])["status"] == "done"


def test_sweep_faults_a_candidate_whose_merged_ctx_would_exceed_max_config_len(
    wf_repo,
):
    # `analyst`'s diff-gate finding on U3a/U3b: `sweep_due_workflow_runs`'s ctx
    # merge (current ctx + the `timerFired` marker) must apply the same
    # `MAX_CONFIG_LEN` bound `submit_workflow_input`'s own merge already
    # enforces (D-H) — bucketed as **faulted**, not an uncaught
    # `WorkflowInputRejectedError` that would break batch isolation for every
    # remaining candidate.
    #
    # The oversized VALUE lives in the run's own `ctx` (seeded at start via
    # `run_ctx`, a large-but-boring filler string) — the exact same shape
    # `test_process_input.py::test_oversized_merged_ctx_is_rejected_by_the_
    # service_as_400` already uses safely. An earlier version of this test
    # tried an oversized STEP KEY instead (an ~8KB `Step.key`/`stepUid`,
    # indexed/constrained properties) — that write crashed the shared
    # FalkorDB dev instance outright (connection closed by server, container
    # exited) when writing it into `reference` via `publish_workflow_def`, a
    # genuine engine-crash gotcha on an oversized INDEXED string property, not
    # a bug in this feature. `ctx` is opaque and unindexed (rule 8), so an
    # 8KB value there is exactly the load-bearing shape every other opaque
    # payload in this codebase already stores at this scale, with no such
    # risk — reported separately, not re-attempted here.
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)

    oversized = _publish_and_start(
        services, key="timers-oversized",
        steps=[
            {"key": "park", "type": "wait", "start": True,
             "config": {"waitsForHuman": True, "waitForSeconds": 60}},
            {"key": "activate", "type": "decision", "config": {}},
        ],
        transitions=[
            {"from": "park", "to": "activate", "on": "timeout", "order": 0,
             "guard": _escalation_guard("park")},
        ],
        run_ctx={"seed": "y" * MAX_CONFIG_LEN},
    )
    # A second, normal due run in the same sweep call — batch isolation must
    # hold exactly as it does for a drive-time fault (the test above).
    normal = _publish_and_start(
        services, key="timers-oversized-normal",
        steps=[
            {"key": "park", "type": "wait", "start": True,
             "config": {"waitsForHuman": True, "waitForSeconds": 30}},
            {"key": "activate", "type": "decision", "config": {}},
        ],
        transitions=[
            {"from": "park", "to": "activate", "on": "timeout", "order": 0,
             "guard": _escalation_guard("park")},
        ],
    )
    clock.advance(120_000)

    out = services.sweep_due_workflow_runs(CTX, limit=10)

    assert out["due"] == 2
    assert [f["runId"] for f in out["faulted"]] == [oversized["runId"]]
    faulted_error = out["faulted"][0]["error"]
    assert faulted_error.startswith("WorkflowInputRejectedError:")
    assert "character bound" in faulted_error
    # the oversized run was never touched — no marker written, still parked
    run = services.get_workflow_run(CTX, run_id=oversized["runId"])
    assert run["status"] == "waiting"
    assert "timerFired" not in _ctx_of(services, oversized["runId"])
    # the normal run in the same call still resolves correctly
    assert [r["runId"] for r in out["resumed"]] == [normal["runId"]]
    assert services.get_workflow_run(CTX, run_id=normal["runId"])["status"] == "done"


def test_sweep_after_resuming_a_run_a_second_submit_gets_the_existing_409(
    wf_repo,
):
    # The mirror of test_repository.py's CAS-contention shape (§2's finding):
    # once the sweep wins the CAS and drives the run out of `waiting` (via the
    # escalation branch), a subsequent human `submit_workflow_input` on the SAME
    # run gets the existing 409, proving the sweep is not a distinguishable,
    # special-cased resume path from the CAS's point of view.
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "park", "type": "wait", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60,
                    "signal": "provisioned"}},
        {"key": "activate", "type": "decision", "config": {}},
    ]
    transitions = [
        {"from": "park", "to": "activate", "on": "timeout", "order": 0,
         "guard": _escalation_guard("park")},
    ]
    started = _publish_and_start(
        services, key="timers-409", steps=steps, transitions=transitions,
    )
    run_id = started["runId"]
    clock.advance(120_000)

    out = services.sweep_due_workflow_runs(CTX, limit=10)
    assert out["resumed"] == [{"runId": run_id, "status": "done"}]

    with pytest.raises(WorkflowRunNotWaitingError):
        services.submit_workflow_input(CTX, run_id=run_id, input={"provisioned": True})


def test_not_yet_resume_survives_unbroken_on_a_timer_bearing_step(wf_repo):
    # v3, resolves `analyst` Pass 2's "not yet" finding — verified, not just
    # documented (`docs/plans/workflow-timers.md` §6 test 4 new bullet). A
    # `wait` step shaped exactly like `proof_defs.py`'s shipped `provision` step
    # (deliberately no `fields`/`expects`, so `{"provisioned": false}` is an
    # accepted "not yet" input) PLUS a timer and its canonical escalation
    # transition — the two patterns must compose on the same step.
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "provision", "type": "wait", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60,
                    "signal": "provisioned"}},
        {"key": "activate", "type": "decision", "config": {}},
        {"key": "escalate", "type": "decision", "config": {}},
    ]
    transitions = [
        {"from": "provision", "to": "activate", "on": "provisioned", "order": 0,
         "guard": {"kind": "cmp", "path": "ctx.provisioned", "op": "truthy"}},
        {"from": "provision", "to": "escalate", "on": "timeout", "order": 1,
         "guard": _escalation_guard("provision")},
    ]
    started = _publish_and_start(
        services, key="timers-not-yet", steps=steps, transitions=transitions,
    )
    run_id = started["runId"]
    assert started["status"] == "waiting"

    # Before the due time: an explicit-but-negative "not yet" reply.
    clock.advance(30_000)
    out = services.submit_workflow_input(
        CTX, run_id=run_id, input={"provisioned": False},
    )
    assert out["status"] == "waiting"
    run = services.get_workflow_run(CTX, run_id=run_id)
    assert run["status"] == "waiting"
    assert run["atStepKey"] == "provision"
    ctx = _ctx_of(services, run_id)
    assert ctx["provisioned"] is False
    assert "timerFired" not in ctx

    # Past the due time: the sweep now resumes it via the escalation branch,
    # exactly as the plain timer case does.
    clock.advance(60_000)
    out = services.sweep_due_workflow_runs(CTX, limit=10)
    assert out["resumed"] == [{"runId": run_id, "status": "done"}]
    run = services.get_workflow_run(CTX, run_id=run_id)
    assert run["status"] == "done"
    assert run["atStepKey"] is None
    assert _ctx_of(services, run_id)["timerFired"] == "provision"


def test_step_scoped_marker_does_not_leak_across_different_timer_steps(wf_repo):
    # v3, closes the self-caught leakage bug (see the header revision note in
    # `docs/plans/workflow-timers.md`). Two timer-bearing steps in sequence:
    # sweeping A past due must not make B immediately escalate on arrival —
    # B's OWN escalation guard checks `ctx.timerFired == "B"`, and the stale
    # `"A"` value left over from A's own resume must not match.
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "A", "type": "wait", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 30}},
        {"key": "B", "type": "wait",
         "config": {"waitsForHuman": True, "waitForSeconds": 30}},
        {"key": "done_step", "type": "decision", "config": {}},
    ]
    transitions = [
        {"from": "A", "to": "B", "on": "timeout", "order": 0,
         "guard": _escalation_guard("A")},
        {"from": "B", "to": "done_step", "on": "timeout", "order": 0,
         "guard": _escalation_guard("B")},
    ]
    started = _publish_and_start(
        services, key="timers-leak", steps=steps, transitions=transitions,
    )
    run_id = started["runId"]
    assert started["status"] == "waiting"
    assert services.get_workflow_run(CTX, run_id=run_id)["atStepKey"] == "A"

    # Sweep A past its due time — advances into B via A's own escalation guard.
    clock.advance(31_000)
    out = services.sweep_due_workflow_runs(CTX, limit=10)
    assert out["resumed"] == [{"runId": run_id, "status": "waiting"}]
    run = services.get_workflow_run(CTX, run_id=run_id)
    # The regression pin: B correctly PARKS on arrival (the stale "A" marker
    # does not satisfy B's own "== B" guard) rather than escalating immediately.
    assert run["status"] == "waiting"
    assert run["atStepKey"] == "B"
    assert _ctx_of(services, run_id)["timerFired"] == "A"

    # B's own due time hasn't passed yet relative to when it just parked.
    out = services.sweep_due_workflow_runs(CTX, limit=10)
    assert out["due"] == 0
    assert services.get_workflow_run(CTX, run_id=run_id)["atStepKey"] == "B"

    # Now sweep B past ITS OWN due time — correctly escalates via its own guard.
    clock.advance(31_000)
    out = services.sweep_due_workflow_runs(CTX, limit=10)
    assert out["resumed"] == [{"runId": run_id, "status": "done"}]
    run = services.get_workflow_run(CTX, run_id=run_id)
    assert run["status"] == "done"
    assert _ctx_of(services, run_id)["timerFired"] == "B"


def test_cas_contention_human_resolves_first_the_sweep_never_touches_it(wf_repo):
    # v3 CAS-contention, direction 1 — proves *which branch fired*, a stronger
    # proof than asserting merely "a resume happened" (plan §6 test 4). A
    # sequential test (call A fully returns, THEN call B starts) cannot
    # reproduce a genuine mid-flight race — `find_due_wait_candidates` re-scans
    # fresh on every call, so a run the human's `submit_workflow_input` already
    # drove OUT of `waiting` is simply never returned as a candidate at all
    # (not "raced" — never scanned in the first place). This is a *stronger*
    # proof of the single-winner property than "raced": the sweep does not even
    # attempt a run someone else already resolved. The genuine read-act-gap
    # race (mid-flight, inside one sweep call) is covered separately below by
    # `_RacingRepo`.
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "park", "type": "wait", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60,
                    "signal": "provisioned"}},
        {"key": "activated_by_human", "type": "decision", "config": {}},
        {"key": "escalated", "type": "decision", "config": {}},
    ]
    transitions = [
        {"from": "park", "to": "activated_by_human", "on": "provisioned",
         "order": 0,
         "guard": {"kind": "cmp", "path": "ctx.provisioned", "op": "truthy"}},
        {"from": "park", "to": "escalated", "on": "timeout", "order": 1,
         "guard": _escalation_guard("park")},
    ]
    started = _publish_and_start(
        services, key="timers-race-human-first", steps=steps, transitions=transitions,
    )
    run_id = started["runId"]

    clock.advance(120_000)  # past due, but the human resolves it first below
    out_human = services.submit_workflow_input(
        CTX, run_id=run_id, input={"provisioned": True},
    )
    assert out_human["status"] == "done"

    out_sweep = services.sweep_due_workflow_runs(CTX, limit=10)
    assert out_sweep["checked"] == 0
    assert out_sweep["due"] == 0
    assert out_sweep["resumed"] == []
    assert out_sweep["raced"] == []

    run = services.get_workflow_run(CTX, run_id=run_id)
    assert run["status"] == "done"
    ctx = _ctx_of(services, run_id)
    assert ctx["provisioned"] is True
    assert "timerFired" not in ctx  # the sweep's merge never reached the graph


def test_cas_contention_sweep_resolves_first_a_second_human_submit_gets_409(
    wf_repo,
):
    # v3 CAS-contention, direction 2 — the sweep wins first, advancing via the
    # ESCALATION branch (not the domain one); a second human attempt on the
    # same run gets the existing 409, proving the sweep is not a
    # distinguishable, special-cased resume path from the CAS's point of view.
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "park", "type": "wait", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60,
                    "signal": "provisioned"}},
        {"key": "activated_by_human", "type": "decision", "config": {}},
        {"key": "escalated", "type": "decision", "config": {}},
    ]
    transitions = [
        {"from": "park", "to": "activated_by_human", "on": "provisioned",
         "order": 0,
         "guard": {"kind": "cmp", "path": "ctx.provisioned", "op": "truthy"}},
        {"from": "park", "to": "escalated", "on": "timeout", "order": 1,
         "guard": _escalation_guard("park")},
    ]
    started = _publish_and_start(
        services, key="timers-race-sweep-first", steps=steps, transitions=transitions,
    )
    run_id = started["runId"]
    clock.advance(120_000)

    out_sweep = services.sweep_due_workflow_runs(CTX, limit=10)
    assert out_sweep["resumed"] == [{"runId": run_id, "status": "done"}]
    run = services.get_workflow_run(CTX, run_id=run_id)
    assert run["atStepKey"] is None
    assert _ctx_of(services, run_id)["timerFired"] == "park"  # escalation, not domain

    with pytest.raises(WorkflowRunNotWaitingError):
        services.submit_workflow_input(CTX, run_id=run_id, input={"provisioned": True})


class _RacingRepo:
    """Wraps a real `Repository`; the FIRST `find_due_wait_candidates` call ALSO
    invokes `on_scan()` before returning — simulating a concurrent actor acting
    on a scanned run in the gap between the sweep's own read and its
    per-candidate act, inside ONE `sweep_due_workflow_runs` call. A fully
    sequential pytest (call A, wait for it to return, THEN call B) cannot
    otherwise reproduce this window: resume calls drive synchronously to a
    stopping point before returning, so the race has to be injected INSIDE the
    sweep's own read-then-act sequence, which is what this wrapper does.
    """

    def __init__(self, repo, *, on_scan) -> None:
        self._repo = repo
        self._on_scan = on_scan
        self._scanned = False

    def __getattr__(self, name):
        return getattr(self._repo, name)

    def find_due_wait_candidates(self, ws, *, limit):
        candidates = self._repo.find_due_wait_candidates(ws, limit=limit)
        if not self._scanned:
            self._scanned = True
            self._on_scan()
        return candidates


def test_sweep_cas_contention_a_run_resumed_in_the_read_act_gap_lands_in_raced(
    wf_repo,
):
    clock = _SharedClock(1_000_000)
    # `services`/its executor are built over the REAL `wf_repo` first, so the
    # def can be published/started normally; the racing wrapper is swapped in
    # onto `services._repo` afterward, only for the sweep call itself.
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "park", "type": "wait", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60}},
        {"key": "activate", "type": "decision", "config": {}},
    ]
    transitions = [
        {"from": "park", "to": "activate", "on": "timeout", "order": 0,
         "guard": _escalation_guard("park")},
    ]
    started = _publish_and_start(
        services, key="timers-race-gap", steps=steps, transitions=transitions,
    )
    run_id = started["runId"]
    clock.advance(120_000)

    racing_repo = _RacingRepo(
        wf_repo, on_scan=lambda: wf_repo.resume_run("test", run_id=run_id),
    )
    services._repo = racing_repo  # noqa: SLF001 — deliberate, see docstring above

    out = services.sweep_due_workflow_runs(CTX, limit=10)

    assert out["checked"] == 1
    assert out["due"] == 1
    assert out["resumed"] == []
    assert out["faulted"] == []
    assert out["raced"] == [{"runId": run_id}]
    # The competing actor's bare CAS flip landed; nothing about `ctx`/the
    # step-run trail reflects a second drive, and no `timerFired` was written.
    run = wf_repo.get_run("test", run_id=run_id)
    assert run["status"] == "running"
    assert run["atStepKey"] == "park"
    assert "timerFired" not in json.loads(run["ctx"])


def test_sweep_bucket_raced_on_a_stale_scanned_step_key_not_just_status(wf_repo):
    # v3, `docs/reviews/workflow-timers.md` Pass 3 new-minor: a candidate
    # scanned at step `park` that has since moved to a DIFFERENT waiting step
    # (still `waiting`, just not at the scanned step) must bucket as `raced`,
    # not be merged/resumed with a stale `stepKey` marker.
    clock = _SharedClock(1_000_000)
    services = _make_services(wf_repo, clock)
    steps = [
        {"key": "park", "type": "wait", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60,
                    "signal": "moveOn"}},
        {"key": "park2", "type": "wait",
         "config": {"waitsForHuman": True, "signal": "other"}},
        {"key": "escalated", "type": "decision", "config": {}},
        {"key": "final", "type": "decision", "config": {}},
    ]
    transitions = [
        {"from": "park", "to": "park2", "on": "moveOn", "order": 0,
         "guard": {"kind": "cmp", "path": "ctx.moveOn", "op": "truthy"}},
        {"from": "park", "to": "escalated", "on": "timeout", "order": 1,
         "guard": _escalation_guard("park")},
        {"from": "park2", "to": "final", "on": "other", "order": 0,
         "guard": {"kind": "cmp", "path": "ctx.other", "op": "truthy"}},
    ]
    services.publish_workflow_def(
        CTX, key="timers-stale-key", version=VERSION, name="Timers",
        kind="process", steps=steps, transitions=transitions,
    )
    services.materialize_def(CTX, key="timers-stale-key", version=VERSION)
    started = services.start_workflow_run(
        CTX, def_key="timers-stale-key", version=VERSION, run_ctx={}, max_steps=12,
    )
    run_id = started["runId"]
    clock.advance(120_000)  # `park`'s timer is due

    # Simulate a concurrent human moving the run from `park` to `park2` in the
    # gap between the sweep's scan and its per-candidate act.
    racing_repo = _RacingRepo(
        wf_repo,
        on_scan=lambda: services.submit_workflow_input(
            CTX, run_id=run_id, input={"moveOn": True},
        ),
    )
    services._repo = racing_repo  # noqa: SLF001 — deliberate, see `_RacingRepo`

    out = services.sweep_due_workflow_runs(CTX, limit=10)

    assert out["checked"] == 1
    assert out["due"] == 1
    assert out["resumed"] == []
    assert out["faulted"] == []
    assert out["raced"] == [{"runId": run_id}]
    run = wf_repo.get_run("test", run_id=run_id)
    assert run["status"] == "waiting"
    assert run["atStepKey"] == "park2"  # moved on, untouched by the sweep
    assert "timerFired" not in json.loads(run["ctx"])  # no stale marker written
