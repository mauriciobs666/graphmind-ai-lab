"""Start-without-trigger + the human/signal input endpoint (K-024 U3, §3.4 / §7).

Offline and deterministic end-to-end: a real `Repository` over `ws:test`, a real
`Services`, a real `WorkflowExecutor` with **no LLM**, and the real FastAPI app —
no network, no `live` marker. A `kind:'process'` run needs none of that, which is
the point of the slice.

What each altitude covers:
  * `services.start_workflow_run(trigger_msg_id=None)` → §12.12, no `TRIGGERED_BY`;
  * `services.submit_workflow_input` → D-H validation, then the D-F single-query
    merge-and-resume (§12.13);
  * the REST round-trip and the full D-G error map, one case per handler.
"""

from __future__ import annotations

import itertools
import json

import pytest
from fastapi.testclient import TestClient

from falkorchat.app import create_app
from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.services import (
    Services,
    WorkflowInputRejectedError,
    WorkflowRunNotFoundError,
    WorkflowRunNotWaitingError,
)

CTX = CallContext(ws="test", actor="u1")

GUARD_FILED = json.dumps(
    {"kind": "cmp", "path": "ctx.request.role", "op": "exists"}, separators=(",", ":")
)
GUARD_APPROVED = json.dumps(
    {"kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "approve"},
    separators=(",", ":"),
)
GUARD_PROVISIONED = json.dumps(
    {"kind": "cmp", "path": "ctx.provisioned", "op": "truthy"}, separators=(",", ":")
)

# A `human` step that declares BOTH an accepted-key list and an allowed-value list
# — the two halves of D-H rule 2 / rule 3.
APPROVAL_CONFIG = json.dumps({
    "waitsForHuman": True,
    "prompt": "Approve or reject",
    "fields": ["decision"],
    "expects": {"decision": ["approve", "reject"]},
    "assignee": "manager",
}, separators=(",", ":"))
# A `wait` step: accepted key = the declared signal, no `expects` (so "not yet" —
# `{"provisioned": false}` — stays expressible and simply re-parks).
WAIT_CONFIG = json.dumps(
    {"waitsForHuman": True, "signal": "provisioned"}, separators=(",", ":")
)
# A `human` step that declares NOTHING beyond the park flag — the permissive
# fallback that makes D-H non-retroactive.
BARE_HUMAN_CONFIG = json.dumps({"waitsForHuman": True}, separators=(",", ":"))

APPROVAL_STEPS = [
    {"key": "approval", "type": "human", "config": APPROVAL_CONFIG},
    {"key": "granted", "type": "decision", "config": "{}"},
]
APPROVAL_TRANSITIONS = [
    {"from": "approval", "to": "granted", "on": "approved", "guard": GUARD_APPROVED,
     "order": 0},
]


def _make_services(repo):
    """Real service + real engine, deterministic ids/clocks, no LLM."""
    ids = (f"id{n}" for n in itertools.count(1))
    services = Services(repo, clock=lambda: 1000, id_gen=lambda: next(ids))
    sr_ids = (f"sr{n}" for n in itertools.count(1))
    sr_clock = itertools.count(2000)
    services.set_executor(WorkflowExecutor(
        services, repo, llm=None, guard_judge=None,
        id_gen=lambda: next(sr_ids), clock=lambda: next(sr_clock),
    ))
    return services


def _materialize(repo, *, steps, transitions, start_key, key="access-request"):
    # NOTE: a def with zero transitions trips the latent empty-`UNWIND` collapse in
    # `_PUBLISH_CYPHER` (out of U3's scope) — every fixture here carries ≥1
    # transition, and a terminal step is one with no *outgoing* transition.
    repo.materialize_snapshot(
        "test", key=key, version="1", name="Access request", kind="process",
        start_key=start_key, steps=steps, transitions=transitions,
    )


@pytest.fixture()
def svc(wf_repo):
    return _make_services(wf_repo)


@pytest.fixture()
def client(wf_repo):
    """The real app over `ws:test` with the engine wired (REST altitude)."""
    services = _make_services(wf_repo)
    return TestClient(create_app(
        services,
        context_provider=lambda: CTX,
        mount_mcp=False,
    ))


def _parked_run(svc, repo, **kw):
    """Materialize the approval shape and start an untriggered run that parks."""
    _materialize(repo, steps=APPROVAL_STEPS, transitions=APPROVAL_TRANSITIONS,
                 start_key="approval")
    return svc.start_workflow_run(CTX, def_key="access-request", version="1", **kw)


# ── §7 U3.1 / U3.2 — the two start paths ────────────────────────────────────────


def test_untriggered_start_creates_a_run_with_no_trigger_edge(svc, wf_repo):
    out = _parked_run(svc, wf_repo)

    # it parked on the `human` step, and it is sitting on that step (AT_STEP kept —
    # which is exactly what makes the parked step resolvable for D-H validation)
    assert out["status"] == "waiting"
    run = wf_repo.get_run("test", run_id=out["runId"])
    assert run["atStepKey"] == "approval"
    # …and there is NO TRIGGERED_BY edge: §12.12 never creates one
    edges = wf_repo._graph("test").ro_query(
        "MATCH (r:WorkflowRun {runId: $r})-[e:TRIGGERED_BY]->() RETURN count(e)",
        {"r": out["runId"]},
    ).result_set[0][0]
    assert edges == 0
    # a process run parks with an EMPTY waitingThreadId — it has no thread (F-5/F-6)
    assert run["waitingThreadId"] == ""


def test_untriggered_start_seeds_the_callers_ctx_verbatim(svc, wf_repo):
    _materialize(wf_repo, steps=APPROVAL_STEPS, transitions=APPROVAL_TRANSITIONS,
                 start_key="approval")

    out = svc.start_workflow_run(
        CTX, def_key="access-request", version="1", run_ctx={"requester": "alice"},
    )

    run = wf_repo.get_run("test", run_id=out["runId"])
    assert json.loads(run["ctx"]) == {"requester": "alice"}


def test_triggered_start_still_creates_the_trigger_edge(svc, wf_repo):
    # §7 U3.2 — regression pin: the @mention/chat start path is untouched.
    _materialize(wf_repo, steps=APPROVAL_STEPS, transitions=APPROVAL_TRANSITIONS,
                 start_key="approval")
    wf_repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    wf_repo.create_thread("test", channel_id="c1", thread_id="t1", title="x",
                          created_at=110)
    wf_repo.ensure_user("test", user_id="u1", display_name="Alice")
    wf_repo.post_first_message(
        "test", thread_id="t1", msg_id="trig1", author_id="u1", text="go",
        role="user", created_at=120,
    )

    out = svc.start_workflow_run(
        CTX, def_key="access-request", version="1", trigger_msg_id="trig1",
    )

    run = wf_repo.get_run("test", run_id=out["runId"])
    assert json.loads(run["ctx"]) == {"threadId": "t1"}
    edges = wf_repo._graph("test").ro_query(
        "MATCH (r:WorkflowRun {runId: $r})-[e:TRIGGERED_BY]->(m:Message) "
        "RETURN m.msgId",
        {"r": out["runId"]},
    ).result_set
    assert edges[0][0] == "trig1"


def test_untriggered_start_on_a_missing_snapshot_starts_nothing(svc, wf_repo):
    with pytest.raises(WorkflowRunNotFoundError):
        svc.start_workflow_run(CTX, def_key="ghost", version="1")
    assert wf_repo._graph("test").ro_query(
        "MATCH (r:WorkflowRun) RETURN count(r)"
    ).result_set[0][0] == 0


# ── §7 U3.3 — merge + resume in ONE query (D-F) ─────────────────────────────────


def test_submit_input_merges_flat_into_ctx_and_drives(svc, wf_repo):
    started = _parked_run(svc, wf_repo)

    out = svc.submit_workflow_input(
        CTX, run_id=started["runId"], input={"decision": "approve"},
    )

    # the guard fired, the run advanced past the parked step and completed
    assert out["status"] == "done"
    # merged FLAT — guards read `ctx.decision`, never `ctx.input.decision`
    assert out["ctx"] == {"decision": "approve"}
    run = wf_repo.get_run("test", run_id=started["runId"])
    assert json.loads(run["ctx"]) == {"decision": "approve"}
    # the ctx and the status flip landed together (§12.13, one query)
    assert run["status"] == "done"
    trail = [sr["stepKey"] for sr in wf_repo.read_step_runs("test",
                                                           run_id=started["runId"])]
    assert trail == ["approval", "approval", "granted"]


def test_submit_input_preserves_existing_ctx_keys(svc, wf_repo):
    _materialize(wf_repo, steps=APPROVAL_STEPS, transitions=APPROVAL_TRANSITIONS,
                 start_key="approval")
    started = svc.start_workflow_run(
        CTX, def_key="access-request", version="1", run_ctx={"requester": "alice"},
    )

    out = svc.submit_workflow_input(
        CTX, run_id=started["runId"], input={"decision": "approve"},
    )

    assert out["ctx"] == {"requester": "alice", "decision": "approve"}


def test_a_declared_value_that_fires_no_guard_re_parks(svc, wf_repo):
    # The `wait` shape: `expects` is deliberately absent so "not yet" is expressible.
    _materialize(
        wf_repo,
        steps=[
            {"key": "provision", "type": "wait", "config": WAIT_CONFIG},
            {"key": "activate", "type": "decision", "config": "{}"},
        ],
        transitions=[{"from": "provision", "to": "activate", "on": "provisioned",
                      "guard": GUARD_PROVISIONED, "order": 0}],
        start_key="provision",
    )
    started = svc.start_workflow_run(CTX, def_key="access-request", version="1")

    not_yet = svc.submit_workflow_input(
        CTX, run_id=started["runId"], input={"provisioned": False},
    )
    assert not_yet["status"] == "waiting"  # re-parked, one step consumed

    done = svc.submit_workflow_input(
        CTX, run_id=started["runId"], input={"provisioned": True},
    )
    assert done["status"] == "done"


def test_the_chat_resume_path_still_writes_no_ctx(svc, wf_repo):
    # Regression pin for the D-F parameter: `executor.resume` WITHOUT `run_ctx_json`
    # must stay the plain §12.4 flip — the trigger path is byte-identical.
    _materialize(wf_repo, steps=APPROVAL_STEPS, transitions=APPROVAL_TRANSITIONS,
                 start_key="approval")
    started = svc.start_workflow_run(
        CTX, def_key="access-request", version="1", run_ctx={"seed": 1},
    )

    svc.resume_workflow_run(CTX, run_id=started["runId"])

    run = wf_repo.get_run("test", run_id=started["runId"])
    assert json.loads(run["ctx"]) == {"seed": 1}   # unchanged by the resume
    assert run["status"] == "waiting"              # re-parked: no guard could fire


# ── §7 U3.4 / U3.5 — the run-state errors ───────────────────────────────────────


def test_submit_input_to_a_non_waiting_run_is_rejected(svc, wf_repo):
    started = _parked_run(svc, wf_repo)
    svc.submit_workflow_input(CTX, run_id=started["runId"],
                              input={"decision": "approve"})  # → done

    with pytest.raises(WorkflowRunNotWaitingError):
        svc.submit_workflow_input(CTX, run_id=started["runId"],
                                  input={"decision": "reject"})
    # the terminal run's ctx is untouched — a loser writes nothing (§12.13)
    run = wf_repo.get_run("test", run_id=started["runId"])
    assert json.loads(run["ctx"]) == {"decision": "approve"}


def test_submit_input_to_an_unknown_run_raises_not_found(svc, wf_repo):
    with pytest.raises(WorkflowRunNotFoundError):
        svc.submit_workflow_input(CTX, run_id="ghost", input={"decision": "approve"})


# ── §7 U3.6 / 6b / 6c — the reserved-key rule (M-2 / F-6) ───────────────────────


@pytest.mark.parametrize("reserved", ["threadId", "error"])
def test_reserved_key_in_submitted_input_is_rejected_and_writes_nothing(
    svc, wf_repo, reserved
):
    started = _parked_run(svc, wf_repo)
    before = wf_repo.get_run("test", run_id=started["runId"])

    with pytest.raises(WorkflowInputRejectedError):
        svc.submit_workflow_input(CTX, run_id=started["runId"],
                                  input={reserved: "t1"})

    after = wf_repo.get_run("test", run_id=started["runId"])
    assert after["ctx"] == before["ctx"]
    assert after["status"] == "waiting"
    assert after["stepCount"] == before["stepCount"]


@pytest.mark.parametrize("reserved", ["threadId", "error"])
def test_reserved_key_in_the_start_ctx_is_rejected_at_the_service_layer(
    svc, wf_repo, reserved
):
    # M-2: enforced in the SERVICE, not only in the pydantic schema — MCP and direct
    # service callers never see a schema.
    _materialize(wf_repo, steps=APPROVAL_STEPS, transitions=APPROVAL_TRANSITIONS,
                 start_key="approval")

    with pytest.raises(WorkflowInputRejectedError):
        svc.start_workflow_run(CTX, def_key="access-request", version="1",
                               run_ctx={reserved: "t1"})

    assert wf_repo._graph("test").ro_query(
        "MATCH (r:WorkflowRun) RETURN count(r)"
    ).result_set[0][0] == 0


def test_reserved_key_in_the_start_ctx_is_rejected_over_rest(client):
    r = client.post("/workflow-runs", json={
        "defKey": "access-request", "version": "1", "ctx": {"threadId": "t1"},
    })
    assert r.status_code == 400
    assert r.json()["error"] == "WorkflowInputRejectedError"


def test_a_parked_process_run_is_not_resumed_by_an_unrelated_chat_message(
    svc, wf_repo
):
    # §7 U3.6c — the F-6 scenario end-to-end. The process run parks with
    # `waitingThreadId == ''`, so the thread lookup a chat message performs must not
    # find it (and an empty lookup must not find it either — F-5).
    started = _parked_run(svc, wf_repo)
    wf_repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    wf_repo.create_thread("test", channel_id="c1", thread_id="t1", title="x",
                          created_at=110)

    assert svc.find_waiting_run_for_thread(CTX, thread_id="t1") is None
    assert svc.find_waiting_run_for_thread(CTX, thread_id="") is None
    assert wf_repo.get_run("test", run_id=started["runId"])["status"] == "waiting"


# ── §7 U3.7 — the m-5 layer split: schema bounds the input, service the merge ────


def test_oversized_input_is_rejected_by_the_schema_as_422(client):
    r = client.post("/workflow-runs/r1/input", json={"input": {"note": "x" * 9000}})
    assert r.status_code == 422  # never reaches the service


def test_oversized_merged_ctx_is_rejected_by_the_service_as_400(svc, wf_repo):
    # Pydantic cannot see the STORED ctx an input merges into, so the merged bound
    # is the service's (m-5). Each half is under the input bound on its own.
    _materialize(
        wf_repo,
        steps=[{"key": "approval", "type": "human", "config": BARE_HUMAN_CONFIG},
               {"key": "granted", "type": "decision", "config": "{}"}],
        transitions=APPROVAL_TRANSITIONS,
        start_key="approval",
    )
    started = svc.start_workflow_run(
        CTX, def_key="access-request", version="1", run_ctx={"seed": "y" * 5000},
    )

    with pytest.raises(WorkflowInputRejectedError):
        svc.submit_workflow_input(CTX, run_id=started["runId"],
                                  input={"note": "x" * 5000})

    run = wf_repo.get_run("test", run_id=started["runId"])
    assert json.loads(run["ctx"]) == {"seed": "y" * 5000}  # nothing merged
    assert run["status"] == "waiting"


# ── §7 U3.8 — the declared step budget (D-H part c) ─────────────────────────────


def test_start_honours_a_caller_supplied_max_steps(svc, wf_repo):
    _materialize(wf_repo, steps=APPROVAL_STEPS, transitions=APPROVAL_TRANSITIONS,
                 start_key="approval")

    out = svc.start_workflow_run(CTX, def_key="access-request", version="1",
                                 max_steps=24)

    assert wf_repo.get_run("test", run_id=out["runId"])["maxSteps"] == 24


def test_start_without_max_steps_falls_back_to_the_engine_default(svc, wf_repo):
    _materialize(wf_repo, steps=APPROVAL_STEPS, transitions=APPROVAL_TRANSITIONS,
                 start_key="approval")

    out = svc.start_workflow_run(CTX, def_key="access-request", version="1")

    assert wf_repo.get_run("test", run_id=out["runId"])["maxSteps"] == 12


@pytest.mark.parametrize("bad", [0, 51])
def test_rest_bounds_max_steps(client, bad):
    r = client.post("/workflow-runs", json={
        "defKey": "access-request", "version": "1", "maxSteps": bad,
    })
    assert r.status_code == 422


# ── §7 U3.10 — the REST round trip ──────────────────────────────────────────────


def test_rest_round_trip_start_then_input(client, wf_repo):
    _materialize(wf_repo, steps=APPROVAL_STEPS, transitions=APPROVAL_TRANSITIONS,
                 start_key="approval")

    started = client.post("/workflow-runs",
                          json={"defKey": "access-request", "version": "1"})
    assert started.status_code == 201
    body = started.json()
    assert body["status"] == "waiting"

    # the parked step told the client what it is waiting for, with no new query
    trail = client.get(f"/workflow-runs/{body['runId']}/step-runs").json()
    assert json.loads(trail[-1]["output"])["awaiting"]["fields"] == ["decision"]

    advanced = client.post(f"/workflow-runs/{body['runId']}/input",
                           json={"input": {"decision": "approve"}})
    assert advanced.status_code == 200
    assert advanced.json()["status"] == "done"
    assert advanced.json()["ctx"] == {"decision": "approve"}


def test_rest_input_to_an_unknown_run_is_404(client):
    r = client.post("/workflow-runs/ghost/input", json={"input": {"decision": "x"}})
    assert r.status_code == 404
    assert r.json()["error"] == "WorkflowRunNotFoundError"


def test_rest_input_to_a_non_waiting_run_is_409(client, wf_repo):
    _materialize(wf_repo, steps=APPROVAL_STEPS, transitions=APPROVAL_TRANSITIONS,
                 start_key="approval")
    run_id = client.post("/workflow-runs",
                         json={"defKey": "access-request", "version": "1"}
                         ).json()["runId"]
    client.post(f"/workflow-runs/{run_id}/input",
                json={"input": {"decision": "approve"}})  # → done

    r = client.post(f"/workflow-runs/{run_id}/input",
                    json={"input": {"decision": "reject"}})
    assert r.status_code == 409
    assert r.json()["error"] == "WorkflowRunNotWaitingError"


def test_rest_start_on_a_missing_snapshot_is_404(client):
    r = client.post("/workflow-runs", json={"defKey": "ghost", "version": "1"})
    assert r.status_code == 404
    assert r.json()["error"] == "WorkflowRunNotFoundError"


# ── §7 U3.11 / 11b — D-G: a drive fault is a failed envelope, not a 500 ─────────


TOOL_STEPS = [
    {"key": "approval", "type": "human", "config": APPROVAL_CONFIG},
    {"key": "call", "type": "tool", "config": "{}"},
    {"key": "granted", "type": "decision", "config": "{}"},
]
TOOL_TRANSITIONS = [
    {"from": "approval", "to": "call", "on": "approved", "guard": GUARD_APPROVED,
     "order": 0},
    {"from": "call", "to": "granted", "on": "done", "guard": "", "order": 0},
]
# A def whose START is the unimplemented type — the fault happens on the first drive.
TOOL_START_STEPS = [
    {"key": "call", "type": "tool", "config": "{}"},
    {"key": "granted", "type": "decision", "config": "{}"},
]


def test_drive_fault_on_input_returns_200_with_a_failed_envelope(client, wf_repo):
    _materialize(wf_repo, steps=TOOL_STEPS, transitions=TOOL_TRANSITIONS,
                 start_key="approval")
    run_id = client.post("/workflow-runs",
                         json={"defKey": "access-request", "version": "1"}
                         ).json()["runId"]

    r = client.post(f"/workflow-runs/{run_id}/input",
                    json={"input": {"decision": "approve"}})

    assert r.status_code == 200          # NOT 500 — the run is correctly terminal
    assert r.json()["status"] == "failed"
    assert "tool" in r.json()["error"]
    # and the envelope's status is the GRAPH's, not a literal
    assert wf_repo.get_run("test", run_id=run_id)["status"] == "failed"


def test_the_failed_envelope_ctx_is_the_graphs_post_fault_ctx_not_the_submission(
    svc, wf_repo
):
    # The other half of m-12's rule: status AND ctx both come from the same
    # post-fault `get_run`. Reporting graph truth for one field and the caller's
    # hoped-for input for the other is the inconsistency a reader can least afford —
    # the envelope must carry the engine's diagnostic note.
    _materialize(wf_repo, steps=TOOL_STEPS, transitions=TOOL_TRANSITIONS,
                 start_key="approval")
    started = svc.start_workflow_run(CTX, def_key="access-request", version="1")

    out = svc.submit_workflow_input(
        CTX, run_id=started["runId"], input={"decision": "approve"},
    )

    assert out["status"] == "failed"
    graph_ctx = json.loads(wf_repo.get_run("test", run_id=started["runId"])["ctx"])
    assert out["ctx"] == graph_ctx
    assert out["ctx"] != {"decision": "approve"}   # NOT the submitted dict
    assert "error" in out["ctx"]                   # the engine's diagnostic note


def test_the_clean_path_envelope_ctx_still_equals_the_persisted_merge(svc, wf_repo):
    # Guard the other direction: with no fault the envelope's ctx is the merge the
    # CAS wrote, which is the graph's ctx — the two paths agree by construction.
    started = _parked_run(svc, wf_repo)

    out = svc.submit_workflow_input(
        CTX, run_id=started["runId"], input={"decision": "approve"},
    )

    assert out["status"] == "done"
    assert out["ctx"] == json.loads(
        wf_repo.get_run("test", run_id=started["runId"])["ctx"]
    )


def test_drive_fault_on_start_returns_201_with_a_failed_envelope(client, wf_repo):
    _materialize(wf_repo, steps=TOOL_START_STEPS,
                 transitions=[{"from": "call", "to": "granted", "on": "done",
                               "guard": "", "order": 0}],
                 start_key="call")

    r = client.post("/workflow-runs",
                    json={"defKey": "access-request", "version": "1"})

    assert r.status_code == 201          # the run resource WAS created
    assert r.json()["status"] == "failed"
    assert wf_repo.get_run("test", run_id=r.json()["runId"])["status"] == "failed"


def test_the_failed_envelope_status_comes_from_a_post_fault_get_run(svc, wf_repo):
    # §7 U3.11 — pin that the reported status is re-read, not guessed: force the
    # graph to say `done` after the fault and the envelope must say `done` too.
    _materialize(wf_repo, steps=TOOL_START_STEPS,
                 transitions=[{"from": "call", "to": "granted", "on": "done",
                               "guard": "", "order": 0}],
                 start_key="call")
    real_get_run = wf_repo.get_run
    calls = {"n": 0}

    def fake_get_run(ws, *, run_id):
        calls["n"] += 1
        row = real_get_run(ws, run_id=run_id)
        if row is not None and calls["n"] > 1:
            row = {**row, "status": "done"}
        return row

    wf_repo.get_run = fake_get_run
    out = svc.start_workflow_run(CTX, def_key="access-request", version="1")

    assert out["status"] == "done"       # from get_run, not a hardcoded "failed"


def test_a_zombie_running_run_after_a_fault_re_raises_instead_of_reporting_success(
    svc, wf_repo
):
    # §7 U3.11b (m-12) — the worst possible outcome would be a 200/201 success
    # envelope describing a run that is still `running`. Re-raise instead.
    _materialize(wf_repo, steps=TOOL_START_STEPS,
                 transitions=[{"from": "call", "to": "granted", "on": "done",
                               "guard": "", "order": 0}],
                 start_key="call")
    real_get_run = wf_repo.get_run

    def fake_get_run(ws, *, run_id):
        row = real_get_run(ws, run_id=run_id)
        return None if row is None else {**row, "status": "running"}

    wf_repo.get_run = fake_get_run

    with pytest.raises(NotImplementedError):
        svc.start_workflow_run(CTX, def_key="access-request", version="1")


def test_budget_exhaustion_reaches_the_same_envelope_without_raising(svc, wf_repo):
    # m-11 — `StepBudgetExceededError` does NOT exist and must not be invented:
    # `_fail_budget` RETURNS "failed" through the normal path. Pin that a run that
    # self-loops past its budget comes back as a plain failed status, no exception
    # and no `error` key (nothing was caught).
    _materialize(
        wf_repo,
        steps=[{"key": "spin", "type": "decision", "config": "{}"},
               {"key": "other", "type": "decision", "config": "{}"}],
        # a self-loop with an unconditional guard: OUTCOME C forever, until the budget
        transitions=[{"from": "spin", "to": "spin", "on": "again", "guard": "",
                      "order": 0},
                     {"from": "other", "to": "spin", "on": "x", "guard": "",
                      "order": 1}],
        start_key="spin",
    )

    out = svc.start_workflow_run(CTX, def_key="access-request", version="1",
                                 max_steps=3)

    assert out["status"] == "failed"
    assert "error" not in out
    assert wf_repo.get_run("test", run_id=out["runId"])["status"] == "failed"


# ── §7 U3.12 / U3.13 — the remaining two handlers ───────────────────────────────


def test_a_malformed_cmp_guard_at_publish_is_400(client):
    r = client.post("/workflow-defs", json={
        "key": "bad-guard", "version": "1", "name": "Bad", "kind": "process",
        "steps": [{"key": "a", "type": "decision", "config": "{}", "start": True},
                  {"key": "b", "type": "decision", "config": "{}"}],
        "transitions": [{"from": "a", "to": "b", "on": "x", "order": 0,
                         "guard": '{"kind":"cmp","path":"ctx.x","op":"nope"}'}],
    })

    assert r.status_code == 400
    assert r.json()["error"] == "WorkflowConfigError"


def test_starting_a_run_without_a_wired_engine_is_503(wf_repo):
    # Folds the plan's OQ-1: `_require_executor` raises a NAMED RuntimeError
    # subclass, so it maps to 503 without a blanket RuntimeError handler.
    app = create_app(Services(wf_repo), context_provider=lambda: CTX,
                     mount_mcp=False)

    r = TestClient(app).post("/workflow-runs",
                             json={"defKey": "access-request", "version": "1"})

    assert r.status_code == 503
    assert r.json()["error"] == "WorkflowEngineDisabledError"


# ── §7 U3.14 — D-H: mistakes are free (no step budget consumed) ─────────────────


def test_an_undeclared_input_key_is_rejected_for_free(svc, wf_repo):
    started = _parked_run(svc, wf_repo)
    before = wf_repo.get_run("test", run_id=started["runId"])["stepCount"]

    with pytest.raises(WorkflowInputRejectedError):
        svc.submit_workflow_input(CTX, run_id=started["runId"],
                                  input={"decisionn": "approve"})

    after = wf_repo.get_run("test", run_id=started["runId"])
    assert after["stepCount"] == before      # no budget consumed
    assert json.loads(after["ctx"]) == {}    # nothing written
    assert after["status"] == "waiting"


def test_a_value_outside_config_expects_is_rejected_for_free(svc, wf_repo):
    started = _parked_run(svc, wf_repo)
    before = wf_repo.get_run("test", run_id=started["runId"])["stepCount"]

    with pytest.raises(WorkflowInputRejectedError):
        svc.submit_workflow_input(CTX, run_id=started["runId"],
                                  input={"decision": "aprove"})

    after = wf_repo.get_run("test", run_id=started["runId"])
    assert after["stepCount"] == before
    assert after["status"] == "waiting"


def test_repeated_invalid_values_never_exhaust_the_budget(svc, wf_repo):
    # The whole point of D-H part (b): four mistyped approvals used to kill a run at
    # the default budget of 12. Now they cost nothing at all.
    started = _parked_run(svc, wf_repo, max_steps=2)

    for _ in range(10):
        with pytest.raises(WorkflowInputRejectedError):
            svc.submit_workflow_input(CTX, run_id=started["runId"],
                                      input={"decision": "maybe"})

    assert wf_repo.get_run("test", run_id=started["runId"])["status"] == "waiting"
    out = svc.submit_workflow_input(CTX, run_id=started["runId"],
                                    input={"decision": "approve"})
    assert out["status"] == "done"


def test_a_wait_step_accepts_exactly_its_declared_signal(svc, wf_repo):
    _materialize(
        wf_repo,
        steps=[{"key": "provision", "type": "wait", "config": WAIT_CONFIG},
               {"key": "activate", "type": "decision", "config": "{}"}],
        transitions=[{"from": "provision", "to": "activate", "on": "provisioned",
                      "guard": GUARD_PROVISIONED, "order": 0}],
        start_key="provision",
    )
    started = svc.start_workflow_run(CTX, def_key="access-request", version="1")

    with pytest.raises(WorkflowInputRejectedError):
        svc.submit_workflow_input(CTX, run_id=started["runId"],
                                  input={"provisionned": True})

    assert svc.submit_workflow_input(
        CTX, run_id=started["runId"], input={"provisioned": True}
    )["status"] == "done"


def test_a_step_that_declares_nothing_accepts_any_non_reserved_key(svc, wf_repo):
    # The permissive fallback (D-H) — this is what makes the rule non-retroactive:
    # no pre-existing def can start failing because it never declared a field list.
    _materialize(
        wf_repo,
        steps=[{"key": "approval", "type": "human", "config": BARE_HUMAN_CONFIG},
               {"key": "granted", "type": "decision", "config": "{}"}],
        transitions=APPROVAL_TRANSITIONS,
        start_key="approval",
    )
    started = svc.start_workflow_run(CTX, def_key="access-request", version="1")

    out = svc.submit_workflow_input(
        CTX, run_id=started["runId"],
        input={"anything": 1, "decision": "approve"},
    )

    assert out["status"] == "done"
    assert out["ctx"] == {"anything": 1, "decision": "approve"}
