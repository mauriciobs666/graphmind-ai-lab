"""Acceptance: the `access-request@v1` proof flow end-to-end (K-024 U4, plan §4.3 / §7).

**The DESIGN §6.3 proof made executable**: a real business process — file a request,
route it, approve it, wait for provisioning, activate — running on the workflow engine
with **no LLM, no network and no `live` marker**. Every step is `human` / `decision` /
`wait`, every branch is a deterministic `cmp` guard, and every advance is a REST-shaped
`submit_workflow_input` call.

The def under test is imported from `falkorchat.proof_defs` — the *same* constant
`scripts/seed_workflows.sh` publishes (plan §4.4). A copied spec would drift; this cannot.
It is published through the real `publish_workflow_def`, so the K-024 U2 publish
invariants (`waitsForHuman`, `guards.validate_cmp`) are exercised against the shipped def
on every run of this suite — under a **test-only version** (`VERSION` below), so a pytest
run never leaves the production `access-request@v1` behind in the shared `reference`
graph.

The three §4.3 paths, asserted at each stop on `status`, the `AT_STEP` step key and the
`awaiting` envelope of the newest `StepRun`, and at the end on the full `NEXT`-ordered
step-run trail (the audit property), the terminal step reached, `endedAt`, a cleared
`AT_STEP`, and the §4.3 `stepCount`:

  * **privileged** (`role: contractor`) — 8 steps, terminal `activate`;
  * **standard hire** (`role: engineer`) — 6 steps, `route`'s unconditional default beats
    nothing and skips approval entirely, terminal `activate`;
  * **rejected** (`decision: reject`) — 6 steps, terminal `rejected`, still `done`.

Plus the two budget properties: `maxSteps` really does fail a run (F-4 made visible), and
**mistakes are free** (D-H) — rejected submissions consume no budget at all.
"""

from __future__ import annotations

import itertools
import json

import pytest

from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.proof_defs import ACCESS_REQUEST_DEF, ACCESS_REQUEST_MAX_STEPS
from falkorchat.services import Services, WorkflowInputRejectedError

CTX = CallContext(ws="test", actor="u1")

KEY = ACCESS_REQUEST_DEF["key"]
#: Published under a **test-only version** (U4b m-C). `conftest.wf_repo` wipes the shared
#: global `reference` graph at fixture *setup*, so a finished pytest session leaves the
#: last workflow test's defs behind. Publishing the production `access-request@v1` from
#: here therefore made `seed_workflows.sh`'s `already present — no-op` line untrustworthy:
#: it could be reporting a *test's* publish while `ws:acme` still held the old snapshot.
#: Only the version field is overridden — the spec itself is still the one constant the
#: seed script publishes (§4.4's anti-drift property is the whole point of importing it).
VERSION = "v1-test"
TEST_DEF = {**ACCESS_REQUEST_DEF, "version": VERSION}


def _make_services(repo):
    """Real service + real engine, deterministic ids/clocks, **no LLM**."""
    ids = (f"id{n}" for n in itertools.count(1))
    services = Services(repo, clock=lambda: 1000, id_gen=lambda: next(ids))
    sr_ids = (f"sr{n}" for n in itertools.count(1))
    sr_clock = itertools.count(2000)
    services.set_executor(WorkflowExecutor(
        services, repo, llm=None, guard_judge=None,
        id_gen=lambda: next(sr_ids), clock=lambda: next(sr_clock),
    ))
    return services


@pytest.fixture()
def svc(wf_repo):
    """Services over `ws:test` with the shipped proof def published + materialized.

    Published through the *service* layer, exactly as `seed_workflows.sh` does — so the
    publish-time invariants are part of what this acceptance suite proves — but under
    `VERSION` (`v1-test`), never the production `(key, version)` pair.
    """
    services = _make_services(wf_repo)
    services.publish_workflow_def(CTX, **TEST_DEF)
    services.materialize_def(CTX, key=KEY, version=VERSION)
    return services


def _start(svc, **kw):
    kw.setdefault("max_steps", ACCESS_REQUEST_MAX_STEPS)
    return svc.start_workflow_run(CTX, def_key=KEY, version=VERSION, **kw)


def _awaiting(repo, run_id):
    """The `awaiting` envelope of the newest recorded StepRun — what a client renders."""
    trail = repo.read_step_runs("test", run_id=run_id)
    return json.loads(trail[-1]["output"])["awaiting"]


def _trail(repo, run_id):
    return [sr["stepKey"] for sr in repo.read_step_runs("test", run_id=run_id)]


def _assert_parked(repo, run_id, *, step_key):
    """A parked run: `waiting`, still sitting on its step, no `endedAt`."""
    run = repo.get_run("test", run_id=run_id)
    assert run["status"] == "waiting"
    assert run["atStepKey"] == step_key
    assert run["endedAt"] is None
    return run


def _assert_completed(repo, run_id, *, terminal, step_count):
    """A finished run: `done`, `AT_STEP` cleared, `endedAt` stamped, budget as §4.3."""
    run = repo.get_run("test", run_id=run_id)
    assert run["status"] == "done"
    assert run["atStepKey"] is None
    assert run["endedAt"] is not None
    assert run["stepCount"] == step_count
    assert _trail(repo, run_id)[-1] == terminal
    return run


# ── §4.3 path 1 — the privileged-role happy path (8 steps → `activate`) ────────


def test_privileged_request_walks_submit_route_approval_provision_to_activate(
    svc, wf_repo
):
    # ── §4.3 row 1: the start parks on `submit` — guard #1 cannot fire, there is no
    # `ctx.request` yet. This is the whole park-and-branch mechanic (§3.1).
    started = _start(svc)
    run_id = started["runId"]
    assert started["status"] == "waiting"
    _assert_parked(wf_repo, run_id, step_key="submit")
    assert _awaiting(wf_repo, run_id) == {
        "kind": "human",
        "prompt": "File the access request",
        "assignee": "requester",
        "fields": ["request"],
    }

    # ── rows 2–4: filing the request fires #1 (`exists`), then #2 (`in`) routes a
    # contractor to approval, which parks (neither #4 nor #5 can fire yet).
    out = svc.submit_workflow_input(
        CTX, run_id=run_id,
        input={"request": {"role": "contractor", "laptop": True}},
    )
    assert out["status"] == "waiting"
    # the input merged FLAT — guards read `ctx.request.role`, not `ctx.input.request`
    assert out["ctx"]["request"] == {"role": "contractor", "laptop": True}
    _assert_parked(wf_repo, run_id, step_key="approval")
    assert _awaiting(wf_repo, run_id) == {
        "kind": "human",
        "prompt": "Approve or reject this access request",
        "assignee": "manager",
        "fields": ["decision"],
    }
    assert _trail(wf_repo, run_id) == ["submit", "submit", "route", "approval"]

    # ── rows 5–6: approval fires #4 (`eq`) and provision parks on the signal.
    out = svc.submit_workflow_input(CTX, run_id=run_id, input={"decision": "approve"})
    assert out["status"] == "waiting"
    _assert_parked(wf_repo, run_id, step_key="provision")
    assert _awaiting(wf_repo, run_id) == {"kind": "signal", "signal": "provisioned"}

    # ── rows 7–8: the provisioning signal fires #6 (`truthy`) and `activate`, having no
    # outgoing transition, completes the run.
    out = svc.submit_workflow_input(CTX, run_id=run_id, input={"provisioned": True})
    assert out["status"] == "done"

    _assert_completed(wf_repo, run_id, terminal="activate", step_count=8)
    # the audit proof: the NEXT-ordered trail *is* the story of the request. The doubled
    # `submit`/`approval`/`provision` entries are the park + the re-execution that the
    # newly-supplied input unblocked.
    assert _trail(wf_repo, run_id) == [
        "submit", "submit", "route", "approval", "approval",
        "provision", "provision", "activate",
    ]
    # …and the run's own ctx explains every branch it took (D-F).
    assert json.loads(wf_repo.get_run("test", run_id=run_id)["ctx"]) == {
        "request": {"role": "contractor", "laptop": True},
        "decision": "approve",
        "provisioned": True,
    }


# ── §4.3 path 2 — the standard hire (6 steps, approval skipped) ────────────────


def test_standard_hire_skips_approval_via_the_unconditional_default(svc, wf_repo):
    started = _start(svc)
    run_id = started["runId"]

    # `engineer` is not in #2's `["contractor","exec"]`, so the conditional loses and
    # #3 — the unconditional default — routes straight to provision (the
    # conditional-beats-unconditional ordering rule, exercised from the other side).
    out = svc.submit_workflow_input(
        CTX, run_id=run_id, input={"request": {"role": "engineer"}}
    )
    assert out["status"] == "waiting"
    _assert_parked(wf_repo, run_id, step_key="provision")
    assert _awaiting(wf_repo, run_id) == {"kind": "signal", "signal": "provisioned"}
    assert "approval" not in _trail(wf_repo, run_id)

    out = svc.submit_workflow_input(CTX, run_id=run_id, input={"provisioned": True})
    assert out["status"] == "done"

    _assert_completed(wf_repo, run_id, terminal="activate", step_count=6)
    assert _trail(wf_repo, run_id) == [
        "submit", "submit", "route", "provision", "provision", "activate",
    ]


# ── §4.3 path 3 — the rejected request (6 steps, terminal `rejected`) ──────────


def test_rejected_request_ends_done_at_the_rejected_outcome_node(svc, wf_repo):
    started = _start(svc)
    run_id = started["runId"]
    svc.submit_workflow_input(
        CTX, run_id=run_id, input={"request": {"role": "exec"}}
    )
    _assert_parked(wf_repo, run_id, step_key="approval")

    out = svc.submit_workflow_input(CTX, run_id=run_id, input={"decision": "reject"})

    # A rejected request is a **completed process with a rejected outcome** — `done`,
    # never `failed`. `failed` stays reserved for engine faults and budget exhaustion.
    assert out["status"] == "done"
    _assert_completed(wf_repo, run_id, terminal="rejected", step_count=6)
    assert _trail(wf_repo, run_id) == [
        "submit", "submit", "route", "approval", "approval", "rejected",
    ]
    assert "provision" not in _trail(wf_repo, run_id)


# ── the budget properties (F-4 / D-H) ─────────────────────────────────────────


def test_a_run_started_with_a_too_small_budget_fails_with_the_step_budget_note(
    svc, wf_repo
):
    # F-4 made visible: `maxSteps` is a real runaway guard, not decoration. Two steps
    # is not enough to reach even `route`.
    started = _start(svc, max_steps=2)
    run_id = started["runId"]

    out = svc.submit_workflow_input(
        CTX, run_id=run_id, input={"request": {"role": "contractor"}}
    )

    assert out["status"] == "failed"
    run = wf_repo.get_run("test", run_id=run_id)
    assert run["status"] == "failed"
    assert run["atStepKey"] is None
    assert json.loads(run["ctx"])["error"] == "step budget exceeded"


def test_typos_are_free_rejected_submissions_never_consume_the_step_budget(
    svc, wf_repo
):
    # D-H's "mistakes are free" property, asserted rather than assumed: the def declares
    # what `approval` accepts (`fields`) and which values are legal (`expects`), so an
    # undeclared key and an out-of-range value are both rejected at the boundary —
    # before the merge, before the resume CAS, before any step runs.
    started = _start(svc)
    run_id = started["runId"]
    svc.submit_workflow_input(
        CTX, run_id=run_id, input={"request": {"role": "contractor"}}
    )
    before = wf_repo.get_run("test", run_id=run_id)
    assert before["status"] == "waiting"

    bad_inputs = [
        {"decision": "approv"},          # typo — outside `expects`
        {"decision": "yes please"},      # free text — outside `expects`
        {"verdict": "approve"},          # undeclared key — outside `fields`
    ]
    for bad in bad_inputs:
        with pytest.raises(WorkflowInputRejectedError):
            svc.submit_workflow_input(CTX, run_id=run_id, input=bad)

    after = wf_repo.get_run("test", run_id=run_id)
    assert after["stepCount"] == before["stepCount"]
    assert after["status"] == "waiting"
    assert after["atStepKey"] == "approval"
    # nothing written: not one of the three reached `ctx`
    assert json.loads(after["ctx"]) == json.loads(before["ctx"])
    assert "decision" not in json.loads(after["ctx"])

    # …and the run is still perfectly advanceable afterwards.
    out = svc.submit_workflow_input(CTX, run_id=run_id, input={"decision": "approve"})
    assert out["status"] == "waiting"
    _assert_parked(wf_repo, run_id, step_key="provision")


# ── the def itself (the seed script publishes exactly this) ───────────────────


def test_the_shipped_def_is_published_and_materialized_as_specified(svc, wf_repo):
    # §4.1/§4.2 pinned as data: six steps, **six** transitions (the §4 tables are
    # authoritative), and the def is `kind:'process'` with `submit` as its start.
    # The production `(key, version)` pair the seed script publishes — pinned here
    # because the suite itself drives a test-only version (m-C).
    assert (ACCESS_REQUEST_DEF["key"], ACCESS_REQUEST_DEF["version"]) == (
        "access-request", "v1",
    )
    snap = svc.get_snapshot(CTX, key=KEY, version=VERSION)
    assert snap["kind"] == "process"
    assert snap["start_key"] == "submit"
    assert {s["key"] for s in snap["steps"]} == {
        "submit", "route", "approval", "provision", "activate", "rejected",
    }
    # §4.1's step *types* — `human`×2, `decision`×3, `wait`×1 — read back **from the graph**
    # against a literal. Asserting the snapshot against `ACCESS_REQUEST_DEF` (or `TEST_DEF`
    # against its own definition, as this line used to) cannot fail: the snapshot is
    # materialized *from* that constant, so both sides move together. The literal is what
    # makes a `proof_defs.py` type edit — and a publish/materialize round-trip that dropped
    # `Step.type` — visible here.
    assert {s["key"]: s["type"] for s in snap["steps"]} == {
        "submit": "human", "route": "decision", "approval": "human",
        "provision": "wait", "activate": "decision", "rejected": "decision",
    }
    assert len(snap["transitions"]) == 6
    # ≥1 transition is not incidental: a zero-transition publish partially writes and
    # then raises `IndexError` (the empty-`UNWIND` collapse in `_PUBLISH_CYPHER`).
    assert ACCESS_REQUEST_DEF["transitions"]
    # the four `cmp` ops the flow is built to exercise
    assert {
        tr["guard"]["op"] for tr in ACCESS_REQUEST_DEF["transitions"]
        if isinstance(tr["guard"], dict)
    } == {"exists", "in", "eq", "truthy"}
    # the budget the def declares, with the §4.3 happy path (8) well inside it
    assert ACCESS_REQUEST_MAX_STEPS == 24
