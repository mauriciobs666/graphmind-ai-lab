"""K-097 — the `ctxVersion` compare-and-swap on the resume
(`docs/plans/workflow-input-cas-graph.md` §2/§3).

A status-only resume CAS matches `waiting` again once the winner's drive re-parks
the run (the ABA, note §1): a slower `/input` submitter or timer sweep whose merge
was built on the pre-resume `ctx` would then overwrite the winner's input and get
200. The fix: `get_run` projects `coalesce(r.ctxVersion, 0)`, the service hands that
exact value to `resume_run_with_ctx`, and the CAS compares and bumps it. Every exit
from `waiting` (`resume_run`) and every ctx write to an existing run (`fail_run`)
bumps it too.

Offline and deterministic: a real `Repository` over `ws:test`, a real `Services` and
a real LLM-free `WorkflowExecutor`. A concurrent actor is injected by a proxy on
`executor._repo` (the CAS) or on `services._repo` (the service's own reads), the
`_RacingRepo` technique of `test_workflow_timers.py`. Numbered tests follow note §3.
"""

from __future__ import annotations

import json

import pytest

from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.services import Services, WorkflowRunNotWaitingError

CTX = CallContext(ws="test", actor="u1")
VERSION = "v1"


class _Clock:
    """One settable clock shared by `Services` and the executor, so a park's
    `parkedAt` and the sweep's `now` are comparable (see `test_workflow_timers`)."""

    def __init__(self, start: int) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now

    def advance(self, delta: int) -> None:
        self.now += delta


def _build(repo, clock=None):
    clock = clock or _Clock(1_000_000)
    counter = iter(range(1, 10_000))
    services = Services(repo, clock=clock, id_gen=lambda: f"id{next(counter)}")
    executor = WorkflowExecutor(
        services, repo, llm=None, guard_judge=None,
        id_gen=lambda: f"sr{next(counter)}", clock=clock,
    )
    services.set_executor(executor)
    return services, executor, clock


def _start(services, *, key, steps, transitions):
    services.publish_workflow_def(
        CTX, key=key, version=VERSION, name=key, kind="process",
        steps=steps, transitions=transitions,
    )
    services.materialize_def(CTX, key=key, version=VERSION)
    return services.start_workflow_run(
        CTX, def_key=key, version=VERSION, run_ctx={}, max_steps=12,
    )["runId"]


def _cmp(path, op, value=None):
    g = {"kind": "cmp", "path": path, "op": op}
    if value is not None:
        g["value"] = value
    return g


def _escalation(step_key):
    return _cmp("ctx.timerFired", "eq", step_key)


# `note` re-parks on itself until `ctx.finish` (harness A3).
SAME_STEP = dict(
    steps=[
        {"key": "note", "type": "human", "start": True,
         "config": {"waitsForHuman": True}},
        {"key": "end", "type": "decision", "config": {}},
    ],
    transitions=[{"from": "note", "to": "end", "on": "finish", "order": 0,
                  "guard": _cmp("ctx.finish", "truthy")}],
)
# approval --approve--> provision (parks) --ticket--> granted (harness A2).
TWO_STEP = dict(
    steps=[
        {"key": "approval", "type": "human", "start": True,
         "config": {"waitsForHuman": True, "fields": ["decision"],
                    "expects": {"decision": ["approve", "reject"]}}},
        {"key": "provision", "type": "human",
         "config": {"waitsForHuman": True, "fields": ["ticket"]}},
        {"key": "granted", "type": "decision", "config": {}},
    ],
    transitions=[
        {"from": "approval", "to": "provision", "on": "approved", "order": 0,
         "guard": _cmp("ctx.decision", "eq", "approve")},
        {"from": "provision", "to": "granted", "on": "ticketed", "order": 0,
         "guard": _cmp("ctx.ticket", "exists")},
    ],
)
# A timer-bearing bare human step that re-parks on any ordinary input.
TIMED = dict(
    steps=[
        {"key": "park", "type": "human", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60}},
        {"key": "end", "type": "decision", "config": {}},
    ],
    transitions=[{"from": "park", "to": "end", "on": "timeout", "order": 0,
                  "guard": _escalation("park")}],
)


class _BeforeCas:
    """Proxy for `executor._repo`: the FIRST `resume_run_with_ctx` (the submitter
    under test's CAS) first runs `on_cas()`, then forwards — with `ctx_version`,
    which every proxy overriding the CAS must accept and pass on (note §3)."""

    def __init__(self, inner, on_cas):
        self._inner, self._on_cas, self._fired = inner, on_cas, False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def resume_run_with_ctx(self, ws, *, run_id, ctx, ctx_version):
        if not self._fired:
            self._fired = True
            self._on_cas()
        return self._inner.resume_run_with_ctx(
            ws, run_id=run_id, ctx=ctx, ctx_version=ctx_version,
        )


class _ForcedVersion:
    """Proxy for `executor._repo` that sends a fixed (stale) `ctx_version`."""

    def __init__(self, inner, version):
        self._inner, self._version = inner, version

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def resume_run_with_ctx(self, ws, *, run_id, ctx, ctx_version):
        return self._inner.resume_run_with_ctx(
            ws, run_id=run_id, ctx=ctx, ctx_version=self._version,
        )


def _raw_version(conn, run_id):
    """The stored property, no `coalesce` — absent reads as `None`."""
    res = conn.select_graph("ws:test").ro_query(
        "MATCH (r:WorkflowRun {runId: $runId}) RETURN r.ctxVersion", {"runId": run_id},
    )
    return res.result_set[0][0]


def _trail(repo, run_id):
    return [s["stepKey"] for s in repo.read_step_runs("test", run_id=run_id)]


def _run(repo, run_id):
    return repo.get_run("test", run_id=run_id)


# ── 1. ABA, re-park at the same step (harness A3) ───────────────────────────────


def test_aba_reparked_at_the_same_step_refuses_the_stale_submitter(wf_repo):
    services, executor, _ = _build(wf_repo)
    run_id = _start(services, key="cas-same", **SAME_STEP)
    after_a = {}

    def submitter_a():
        out = services.submit_workflow_input(CTX, run_id=run_id,
                                             input={"comment": "from-A"})
        assert out["status"] == "waiting"   # A re-parked at `note`
        after_a["run"] = _run(wf_repo, run_id)
        after_a["trail"] = _trail(wf_repo, run_id)

    executor._repo = _BeforeCas(wf_repo, submitter_a)  # noqa: SLF001

    with pytest.raises(WorkflowRunNotWaitingError):
        services.submit_workflow_input(CTX, run_id=run_id, input={"other": "from-B"})

    run = _run(wf_repo, run_id)
    assert json.loads(run["ctx"]) == {"comment": "from-A"}
    assert run["status"] == "waiting"
    assert run["stepCount"] == after_a["run"]["stepCount"]   # B consumed no step
    assert _trail(wf_repo, run_id) == after_a["trail"]       # B left no StepRun


# ── 2. ABA, re-park at a different step (harness A2) ────────────────────────────


def test_aba_reparked_at_another_step_keeps_the_input_the_guard_branched_on(wf_repo):
    services, executor, _ = _build(wf_repo)
    run_id = _start(services, key="cas-two", **TWO_STEP)
    after_a = {}

    def submitter_a():
        services.submit_workflow_input(CTX, run_id=run_id,
                                       input={"decision": "approve"})
        after_a["trail"] = _trail(wf_repo, run_id)

    executor._repo = _BeforeCas(wf_repo, submitter_a)  # noqa: SLF001

    with pytest.raises(WorkflowRunNotWaitingError):
        services.submit_workflow_input(CTX, run_id=run_id,
                                       input={"decision": "reject"})

    run = _run(wf_repo, run_id)
    assert json.loads(run["ctx"])["decision"] == "approve"
    assert run["atStepKey"] == "provision"
    # A's trail: the park, A's re-execution of `approval`, then `provision`.
    assert _trail(wf_repo, run_id) == after_a["trail"] == [
        "approval", "approval", "provision",
    ]


# ── 3. A chat resume in the gap (probe B2) ──────────────────────────────────────


def test_a_chat_resume_and_repark_in_the_gap_refuses_the_submitter(wf_repo):
    services, executor, _ = _build(wf_repo)
    run_id = _start(services, key="cas-chat", **SAME_STEP)
    # The run has already taken one input (so the chat bump stacks on a CAS bump).
    services.submit_workflow_input(CTX, run_id=run_id, input={"first": 1})

    def chat_reply():
        assert wf_repo.resume_run("test", run_id=run_id) is not None
        assert wf_repo.suspend_run("test", run_id=run_id, thread_id="") is not None

    executor._repo = _BeforeCas(wf_repo, chat_reply)  # noqa: SLF001

    with pytest.raises(WorkflowRunNotWaitingError):
        services.submit_workflow_input(CTX, run_id=run_id, input={"late": 1})

    run = _run(wf_repo, run_id)
    assert json.loads(run["ctx"]) == {"first": 1}
    assert run["status"] == "waiting"


# ── 3b. The version is the one read WITH the merged ctx ─────────────────────────


def test_a_repark_during_validation_refuses_the_merge_built_on_the_earlier_read(
    wf_repo,
):
    # The gap is B's validate-and-merge, after B's `get_run`: A runs to a re-park
    # while B resolves the parked step's snapshot. The version B sends must be the
    # one that came with the ctx B merged onto — a fresher read would match A's
    # re-park and write B's stale merge over A's input.
    services, _, _ = _build(wf_repo)
    run_id = _start(services, key="cas-validate", **SAME_STEP)

    class _DuringValidation:
        def __init__(self, inner):
            self._inner, self._fired = inner, False

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def get_snapshot(self, *a, **kw):
            if not self._fired:
                self._fired = True
                services._repo = self._inner  # noqa: SLF001 — A runs unproxied
                services.submit_workflow_input(CTX, run_id=run_id,
                                               input={"comment": "from-A"})
                services._repo = self  # noqa: SLF001
            return self._inner.get_snapshot(*a, **kw)

    services._repo = _DuringValidation(wf_repo)  # noqa: SLF001

    with pytest.raises(WorkflowRunNotWaitingError):
        services.submit_workflow_input(CTX, run_id=run_id, input={"other": "from-B"})

    assert json.loads(_run(wf_repo, run_id)["ctx"]) == {"comment": "from-A"}


# ── 4. The sweep in its read-to-CAS gap ─────────────────────────────────────────


def test_a_human_repark_in_the_sweeps_read_to_cas_gap_lands_the_run_in_raced(
    wf_repo,
):
    services, executor, clock = _build(wf_repo)
    run_id = _start(services, key="cas-sweep-gap", **TIMED)
    clock.advance(120_000)

    def human():
        out = services.submit_workflow_input(CTX, run_id=run_id,
                                             input={"note": "from-human"})
        assert out["status"] == "waiting"   # re-parked at the same step

    executor._repo = _BeforeCas(wf_repo, human)  # noqa: SLF001

    out = services.sweep_due_workflow_runs(CTX, limit=10)

    assert out["raced"] == [{"runId": run_id}]
    assert out["resumed"] == [] and out["faulted"] == []
    run = _run(wf_repo, run_id)
    assert run["status"] == "waiting" and run["atStepKey"] == "park"
    assert json.loads(run["ctx"]) == {"note": "from-human"}   # no `timerFired`


# ── 4b. The sweep's version is the one its step-5.1 read returned ───────────────


def test_a_human_repark_right_after_the_sweeps_read_lands_the_run_in_raced(wf_repo):
    # The sweep counterpart of 3b: the human re-parks at the same step between the
    # sweep's fresh read (step 5.1, whose `ctx` the marker is merged onto and whose
    # `atStepKey` was just checked) and its CAS. The version the CAS carries must be
    # that read's — a re-read just before the CAS would match the re-park and write
    # the stale `timerFired` merge over the human's input.
    services, _, clock = _build(wf_repo)
    run_id = _start(services, key="cas-sweep-read", **TIMED)
    clock.advance(120_000)

    class _AfterRead:
        def __init__(self, inner):
            self._inner, self._fired = inner, False

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def get_run(self, ws, *, run_id):
            row = self._inner.get_run(ws, run_id=run_id)
            if not self._fired:
                self._fired = True
                services._repo = self._inner  # noqa: SLF001 — the human runs unproxied
                out = services.submit_workflow_input(CTX, run_id=run_id,
                                                     input={"note": "from-human"})
                assert out["status"] == "waiting"   # re-parked at `park`
                services._repo = self  # noqa: SLF001
            return row

    services._repo = _AfterRead(wf_repo)  # noqa: SLF001

    out = services.sweep_due_workflow_runs(CTX, limit=10)

    assert out["raced"] == [{"runId": run_id}]
    assert out["resumed"] == [] and out["faulted"] == []
    run = _run(wf_repo, run_id)
    assert run["status"] == "waiting" and run["atStepKey"] == "park"
    assert json.loads(run["ctx"]) == {"note": "from-human"}   # no `timerFired`


# ── 5. The sweep, no race, on a run that carries no `ctxVersion` ────────────────


def test_the_sweep_resumes_a_due_run_that_carries_no_ctx_version(wf_repo, conn):
    services, _, clock = _build(wf_repo)
    run_id = _start(services, key="cas-sweep", **TIMED)
    assert _raw_version(conn, run_id) is None
    clock.advance(120_000)

    out = services.sweep_due_workflow_runs(CTX, limit=10)

    assert out["resumed"] == [{"runId": run_id, "status": "done"}]
    assert _raw_version(conn, run_id) == 1


# ── 6. The version moves only where it should ───────────────────────────────────


def test_ctx_version_moves_on_every_resume_and_ctx_write_and_nowhere_else(
    wf_repo, conn,
):
    services, executor, clock = _build(wf_repo)
    run_id = _start(services, key="cas-version", steps=[
        {"key": "park", "type": "human", "start": True,
         "config": {"waitsForHuman": True, "waitForSeconds": 60}},
        {"key": "park2", "type": "human", "config": {"waitsForHuman": True}},
        {"key": "boom", "type": "prompt", "config": {}},   # unimplemented → fault
    ], transitions=[
        {"from": "park", "to": "park2", "on": "timeout", "order": 0,
         "guard": _escalation("park")},
        {"from": "park2", "to": "boom", "on": "go", "order": 0,
         "guard": _cmp("ctx.go", "truthy")},
    ])
    assert _raw_version(conn, run_id) is None              # a start writes none

    out = services.submit_workflow_input(CTX, run_id=run_id, input={"x": 1})
    assert out["status"] == "waiting"
    assert _raw_version(conn, run_id) == 1                 # first /input

    assert wf_repo.resume_run("test", run_id=run_id) is not None
    assert _raw_version(conn, run_id) == 2                 # chat-path flip
    assert wf_repo.suspend_run("test", run_id=run_id, thread_id="") is not None
    assert _raw_version(conn, run_id) == 2                 # a park: unchanged

    assert services.resume_workflow_run(CTX, run_id=run_id)["status"] == "waiting"
    assert _raw_version(conn, run_id) == 3                 # chat resume + re-park

    executor._repo = _ForcedVersion(wf_repo, 0)  # noqa: SLF001
    with pytest.raises(WorkflowRunNotWaitingError):
        services.submit_workflow_input(CTX, run_id=run_id, input={"y": 1})
    executor._repo = wf_repo  # noqa: SLF001
    assert _raw_version(conn, run_id) == 3                 # refused: nothing written
    assert json.loads(_run(wf_repo, run_id)["ctx"]) == {"x": 1}

    clock.advance(120_000)
    swept = services.sweep_due_workflow_runs(CTX, limit=10)
    assert swept["resumed"] == [{"runId": run_id, "status": "waiting"}]
    assert _run(wf_repo, run_id)["atStepKey"] == "park2"
    assert _raw_version(conn, run_id) == 4                 # sweep resume

    out = services.submit_workflow_input(CTX, run_id=run_id, input={"go": True})
    assert out["status"] == "failed"
    assert _raw_version(conn, run_id) == 6                 # resume +1, fail_run +1


# ── executor guard: a ctx resume without its version is a programming error ─────


def test_executor_resume_with_a_ctx_but_no_version_raises_and_writes_nothing(
    wf_repo, conn,
):
    services, executor, _ = _build(wf_repo)
    run_id = _start(services, key="cas-guard", **SAME_STEP)

    with pytest.raises(ValueError, match="ctx_version"):
        executor.resume(CTX, run_id=run_id, run_ctx_json='{"x":1}')

    run = _run(wf_repo, run_id)
    assert run["status"] == "waiting" and json.loads(run["ctx"]) == {}
    assert _raw_version(conn, run_id) is None
