"""Typed step handlers + the publish-time invariants (K-024 U2, plan §3.3 / §7).

Two altitudes, both **offline and deterministic** — no LLM, no network, no `live`
marker:

  * the `decision` / `human` / `wait` handlers and the `prompt`/`tool`/`message`
    raising seam (D-E), driven through the **real** §2.1 loop against `ws:test`;
  * the two `_validate_def_spec` invariants (a parking step must declare
    `waitsForHuman`; a `cmp`-family guard is structurally validated at publish),
    including the M-7 **shape matrix** — the REST front door hands `config`/`guard`
    over as **strings**, and neither invariant may be escapable that way.

A `human`/`wait` step is only ever *parked* by the existing OUTCOME B (`waitsForHuman`
+ no firing guard); the handlers themselves are pure. So these tests make a guard fire
by seeding the run `ctx` at start — the ctx **write** path (`resume_run_with_ctx`) is
U3's, and nothing here anticipates it.
"""

from __future__ import annotations

import itertools
import json

import pytest

from falkorchat.config import CallContext
from falkorchat.executor import GraphTracer, WorkflowExecutor
from falkorchat.guards import WorkflowConfigError
from falkorchat.services import Services, WorkflowDefSpecError

CTX = CallContext(ws="test", actor="u1")

# The two guards the process shapes below branch on (plan §4.2 #1 and #4).
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

HUMAN_CONFIG = json.dumps({
    "waitsForHuman": True,
    "prompt": "File the access request",
    "fields": ["request"],
    "assignee": "requester",
}, separators=(",", ":"))
WAIT_CONFIG = json.dumps(
    {"waitsForHuman": True, "signal": "provisioned"}, separators=(",", ":")
)


def _make_executor(repo, *, tracer=None, llm=None):
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(1000)
    return WorkflowExecutor(
        None, repo, llm=llm, guard_judge=None, tracer=tracer,
        id_gen=lambda: next(ids), clock=lambda: next(clock),
    )


# NOTE (latent defect found while writing these, reported to teco — NOT fixed here):
# `repository._PUBLISH_CYPHER` ends with `UNWIND $transitions AS tr … RETURN …`, so a def
# with **zero** transitions collapses the row stream (the documented empty-UNWIND class,
# AGENTS.md) and `publish_def`/`materialize_snapshot` raise `IndexError` on
# `result_set[0]`. Every def below therefore carries at least one transition; a terminal
# step is expressed as a step with no *outgoing* transition instead.

def _start(repo, *, steps, transitions, start_key, ctx="{}", trace=False,
           max_steps=12, run_id="r1"):
    """Materialize a snapshot and start an untriggered-shaped run on it.

    `start_run` still needs a trigger `Message` anchor in this cut (plan F-2 — the
    untriggered start is U3), so a minimal thread + message is seeded. The run `ctx`
    is caller-supplied, which is how a guard is made to fire deterministically here.
    """
    repo.materialize_snapshot(
        "test", key="access-request", version="1", name="Access request",
        kind="process", start_key=start_key, steps=steps, transitions=transitions,
    )
    repo.create_channel("test", channel_id="c1", name="general", created_at=100)
    repo.create_thread("test", channel_id="c1", thread_id="t1", title="x",
                       created_at=110)
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    repo.post_first_message(
        "test", thread_id="t1", msg_id="trig1", author_id="u1",
        text="start", role="user", created_at=120,
    )
    repo.start_run(
        "test", run_id=run_id, def_key="access-request", def_version="1",
        started_at=1000, trigger_msg_id="trig1", ctx=ctx, trace=trace,
        max_steps=max_steps,
    )


def _newest_output(repo, run_id="r1", step_key=None):
    """The `output` of the newest recorded StepRun (optionally for one step key)."""
    trail = repo.read_step_runs("test", run_id=run_id)
    rows = [s for s in trail if step_key is None or s["stepKey"] == step_key]
    return json.loads(rows[-1]["output"])


# ── the typed handlers, through the real drive loop ──────────────────────────

HUMAN_STEPS = [
    {"key": "submit", "type": "human", "config": HUMAN_CONFIG},
    {"key": "route", "type": "decision", "config": "{}"},
]
HUMAN_TRANSITIONS = [
    {"from": "submit", "to": "route", "on": "filed", "guard": GUARD_FILED,
     "order": 0},
]


def test_human_step_parks_the_run_when_its_guard_cannot_fire_yet(wf_repo):
    # §7 case 1 — the park-and-branch mechanic (§3.1): a `human` step is simply a step
    # whose outgoing guard reads a ctx key that is not there yet.
    _start(wf_repo, steps=HUMAN_STEPS, transitions=HUMAN_TRANSITIONS,
           start_key="submit", ctx="{}")
    ex = _make_executor(wf_repo)

    status = ex.run(CTX, run_id="r1")

    assert status == "waiting"
    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "waiting"
    assert run["atStepKey"] == "submit"      # still parked on the step, not advanced
    assert [s["stepKey"] for s in wf_repo.read_step_runs("test", run_id="r1")] \
        == ["submit"]


def test_human_step_records_what_the_run_is_waiting_for(wf_repo):
    # §7 case 2 — the `awaiting` envelope lands on the StepRun.output, so
    # `GET /workflow-runs/{id}/step-runs` answers "what is this waiting for?" with no
    # new query.
    _start(wf_repo, steps=HUMAN_STEPS, transitions=HUMAN_TRANSITIONS,
           start_key="submit", ctx="{}")

    _make_executor(wf_repo).run(CTX, run_id="r1")

    awaiting = _newest_output(wf_repo)["awaiting"]
    assert awaiting == {
        "kind": "human",
        "prompt": "File the access request",
        "assignee": "requester",
        "fields": ["request"],
    }


WAIT_STEPS = [
    {"key": "provision", "type": "wait", "config": WAIT_CONFIG},
    {"key": "activate", "type": "decision", "config": "{}"},
]
WAIT_TRANSITIONS = [
    {"from": "provision", "to": "activate", "on": "provisioned",
     "guard": GUARD_PROVISIONED, "order": 0},
]


def test_wait_step_parks_and_records_the_signal_it_expects(wf_repo):
    # §7 case 3 — `wait` is signal-driven, not timer-driven (D-C: this system has no
    # scheduler), and is mechanically identical to `human` (m-7). Only `awaiting.kind`
    # and the signal name differ, so a client can render the right prompt.
    _start(wf_repo, steps=WAIT_STEPS, transitions=WAIT_TRANSITIONS,
           start_key="provision", ctx="{}")

    status = _make_executor(wf_repo).run(CTX, run_id="r1")

    assert status == "waiting"
    assert _newest_output(wf_repo)["awaiting"] == {
        "kind": "signal", "signal": "provisioned",
    }


def test_wait_step_advances_once_the_signalled_key_is_in_ctx(wf_repo):
    # The other half of the same mechanic: the identical step + guard advances as soon
    # as the ctx carries the signal. `activate` is terminal → the run completes.
    _start(wf_repo, steps=WAIT_STEPS, transitions=WAIT_TRANSITIONS,
           start_key="provision", ctx='{"provisioned":true}')

    status = _make_executor(wf_repo).run(CTX, run_id="r1")

    assert status == "done"
    assert [s["stepKey"] for s in wf_repo.read_step_runs("test", run_id="r1")] \
        == ["provision", "activate"]


BRANCH_STEPS = [
    {"key": "approval", "type": "human", "config": HUMAN_CONFIG},
    {"key": "provision", "type": "decision", "config": "{}"},
    {"key": "rejected", "type": "decision", "config": "{}"},
]
BRANCH_TRANSITIONS = [
    {"from": "approval", "to": "provision", "on": "approved",
     "guard": GUARD_APPROVED, "order": 0},
    {"from": "approval", "to": "rejected", "on": "rejected",
     "guard": json.dumps({"kind": "cmp", "path": "ctx.decision", "op": "eq",
                          "value": "reject"}, separators=(",", ":")), "order": 1},
]


def test_decision_step_advances_on_the_firing_guard_and_has_no_side_effect(wf_repo):
    # §7 case 4 — a `decision` node computes nothing; its semantics are its outgoing
    # guards. The envelope key is `node`, NOT `decision` (n-1): `ctx.decision` is an
    # approval *value* in this very def, and `output.decision` must not read as the same
    # thing.
    _start(wf_repo, steps=BRANCH_STEPS, transitions=BRANCH_TRANSITIONS,
           start_key="approval", ctx='{"decision":"approve"}')

    status = _make_executor(wf_repo).run(CTX, run_id="r1")

    assert status == "done"                       # `provision` is terminal here
    trail = wf_repo.read_step_runs("test", run_id="r1")
    assert [s["stepKey"] for s in trail] == ["approval", "provision"]
    assert _newest_output(wf_repo, step_key="provision") == {
        "node": {"step": "provision"}
    }
    # no side effect: the decision node posted nothing and wrote no ctx
    assert wf_repo.get_run("test", run_id="r1")["ctx"] == '{"decision":"approve"}'


def test_the_losing_branch_is_not_taken(wf_repo):
    # The two-outcome branch (§4.2 #4/#5): `reject` reaches the other terminal.
    _start(wf_repo, steps=BRANCH_STEPS, transitions=BRANCH_TRANSITIONS,
           start_key="approval", ctx='{"decision":"reject"}')

    _make_executor(wf_repo).run(CTX, run_id="r1")

    assert [s["stepKey"] for s in wf_repo.read_step_runs("test", run_id="r1")] \
        == ["approval", "rejected"]


def test_a_decision_step_with_no_outgoing_transitions_completes_the_run(wf_repo):
    # §7 case 5 — a terminal `decision` node is how a process names its outcome; the
    # run ends `done` (the *process* completed; the outcome is which terminal it
    # reached), not `failed`.
    _start(
        wf_repo,
        steps=[{"key": "route", "type": "decision", "config": "{}"},
               {"key": "activate", "type": "decision", "config": "{}"}],
        transitions=[{"from": "route", "to": "activate", "on": "auto", "guard": "",
                      "order": 0}],
        start_key="route",
    )

    status = _make_executor(wf_repo).run(CTX, run_id="r1")

    assert status == "done"
    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "done"
    assert run["atStepKey"] is None
    assert [s["stepKey"] for s in wf_repo.read_step_runs("test", run_id="r1")] \
        == ["route", "activate"]
    assert _newest_output(wf_repo) == {"node": {"step": "activate"}}


@pytest.mark.parametrize("step_type", ["prompt", "tool", "message", "bogus"])
def test_an_unimplemented_step_type_fails_the_run_with_a_named_error(
    wf_repo, step_type
):
    # §7 case 6 (D-E / R-3) — the deliberate behaviour change: an unimplemented type is
    # a named raise reaching the M-1 fault net, not the old silent no-op that let a
    # `decision` node "succeed" doing nothing. Same contract as
    # test_executor.py::test_llm_guard_without_judge_fails_the_run_with_named_error:
    # the run is stamped `failed` with a readable cause AND the exception is re-raised.
    # (The HTTP conversion to D-G's 200/201 failed envelope is U3's.)
    _start(
        wf_repo,
        steps=[{"key": "x", "type": step_type, "config": "{}"},
               {"key": "sink", "type": "decision", "config": "{}"}],
        transitions=[{"from": "x", "to": "sink", "on": "done", "guard": "",
                      "order": 0}],
        start_key="x",
    )
    ex = _make_executor(wf_repo)

    with pytest.raises(NotImplementedError, match="typed-handler seam"):
        ex.run(CTX, run_id="r1")

    run = wf_repo.get_run("test", run_id="r1")
    assert run["status"] == "failed"
    assert run["atStepKey"] is None
    assert step_type in run["ctx"] and "m3-process-flow" in run["ctx"]


def test_agent_step_without_a_wired_llm_still_returns_the_offline_stub(wf_repo):
    # §7 case 7 — F-3 case 1 regression pin. The `agent`-without-LLM stub is the
    # affordance the whole offline loop-engine estate is built on; the typed-handler
    # dispatch must not convert it into a raise.
    _start(
        wf_repo,
        steps=[{"key": "solo", "type": "agent", "config": "{}"},
               {"key": "sink", "type": "decision", "config": "{}"}],
        transitions=[{"from": "solo", "to": "sink", "on": "done", "guard": "",
                      "order": 0}],
        start_key="solo",
    )
    ex = _make_executor(wf_repo, llm=None)

    result = ex._execute_step(
        CTX, {"runId": "r1"}, {"key": "solo", "type": "agent"}, {}, {}
    )

    assert (result.output, result.on) == ("", "done")
    assert ex.run(CTX, run_id="r1") == "done"


def test_a_cmp_guard_is_traced_with_a_readable_label(wf_repo):
    # §7 case 9 (M-6) — a `cmp` guard carries no `text`, so before the
    # `_select_transition`/`_trace_step` edit the payload would have opened with a bare
    # " -> " naming nothing. It must now start with the rendered guard label.
    _start(wf_repo, steps=HUMAN_STEPS, transitions=HUMAN_TRANSITIONS,
           start_key="submit", ctx='{"request":{"role":"contractor"}}', trace=True)
    tracer = GraphTracer(
        wf_repo, id_gen=(lambda c=itertools.count(1): f"te{next(c)}"),
        clock=(lambda c=itertools.count(9000): next(c)),
    )

    _make_executor(wf_repo, tracer=tracer).run(CTX, run_id="r1")

    judgments = [e["payload"] for e in wf_repo.read_trace("test", run_id="r1")
                 if e["kind"] == "guard_judgment"]
    assert judgments, "a cmp guard must contribute a guard_judgment (M-6)"
    assert not any(p.startswith(" -> ") or p.startswith("->") for p in judgments)
    assert judgments[0].startswith("ctx.request.role exists -> True")


# ── publish-time invariants (M-4, M-7, B-2) ──────────────────────────────────

def _steps(*steps):
    return list(steps)


#: A well-formed no-op transition, so a publish that is EXPECTED TO SUCCEED never trips
#: the empty-`$transitions` collapse noted above. Accepting publishes declare their own
#: second step (`sink`).
SINK_STEP = {"key": "sink", "type": "decision", "config": "{}"}
SINK_TRANSITION = {"from": "park", "to": "sink", "on": "done", "order": 0}


def _publish(svc, *, steps, transitions, key="access-request"):
    # `transitions` is REQUIRED, with no default (m-B): a zero-transition publish is the
    # O-6 shape — now rejected by `_validate_def_spec`, so an omitted default would make
    # a future test fail on the transitions rule instead of the rule it meant to test.
    return svc.publish_workflow_def(
        CTX, key=key, version="1", name="Access request", kind="process",
        steps=steps, transitions=list(transitions),
    )


@pytest.mark.parametrize("step_type", ["human", "wait"])
def test_publish_rejects_a_parking_step_that_does_not_declare_waits_for_human(
    wf_repo, step_type
):
    # §7 case 8 — without the flag the executor's OUTCOME B never fires, so the step
    # self-loops (OUTCOME C) until the budget kills the run: a silent, expensive
    # footgun, cheap to reject at authoring time.
    svc = Services(wf_repo)

    with pytest.raises(WorkflowDefSpecError, match="waitsForHuman"):
        _publish(
            svc,
            steps=_steps(
                {"key": "park", "type": step_type, "config": {"prompt": "hi"},
                 "start": True},
                SINK_STEP,
            ),
            transitions=[SINK_TRANSITION],
        )

    assert wf_repo.get_def(key="access-request", version="1") is None


def test_publish_accepts_a_parking_step_that_declares_waits_for_human(wf_repo):
    svc = Services(wf_repo)

    out = _publish(
        svc,
        steps=_steps(
            {"key": "park", "type": "human", "config": {"waitsForHuman": True},
             "start": True},
            {"key": "sink", "type": "wait",
             "config": {"waitsForHuman": True, "signal": "provisioned"}},
        ),
        transitions=[SINK_TRANSITION],
    )

    assert out["stepCount"] == 2
    assert wf_repo.get_def(key="access-request", version="1") is not None


def test_the_new_invariant_runs_last_and_cannot_mask_an_older_check(wf_repo):
    # §7 case 8b (B-2) — ordering pin. This spec violates BOTH the duplicate-key rule
    # and the waitsForHuman rule; it must fail for the duplicate key, i.e. the older
    # checks keep failing for their own reason. Without this, the five tightened
    # `pytest.raises` tests in test_services.py could go vacuous again.
    svc = Services(wf_repo)

    with pytest.raises(WorkflowDefSpecError, match="duplicate step key"):
        _publish(
            svc,
            steps=_steps(
                {"key": "dup", "type": "human", "config": {}, "start": True},
                {"key": "dup", "type": "human", "config": {}},
                SINK_STEP,
            ),
            transitions=[{"from": "dup", "to": "sink", "on": "done", "order": 0}],
        )

    assert wf_repo.get_def(key="access-request", version="1") is None


def test_publish_rejects_a_structurally_malformed_cmp_guard(wf_repo):
    # §7 case 10 (M-4) — one validator, two call sites. A typo'd op otherwise surfaces
    # when a manager clicks approve, killing a live run, instead of at seed time.
    svc = Services(wf_repo)

    with pytest.raises(WorkflowConfigError, match="unknown guard op"):
        _publish(
            svc,
            steps=_steps({"key": "a", "type": "decision", "start": True},
                         {"key": "b", "type": "decision"}),
            transitions=[{"from": "a", "to": "b", "on": "x", "order": 0,
                          "guard": {"kind": "cmp", "path": "ctx.a",
                                    "op": "equals", "value": 1}}],
        )

    assert wf_repo.get_def(key="access-request", version="1") is None


def test_publish_still_accepts_a_kindless_guard_and_an_opaque_config(wf_repo):
    # §7 case 11 — the non-retroactivity contract. Only the cmp family is validated:
    # a guard with no `kind` is not a declaration this validator owns, and an opaque
    # non-JSON config on a non-parking step stays exactly that (the long-standing
    # test_services.py `{"expr":"x>0"}` / `"raw-string"` fixtures).
    svc = Services(wf_repo)

    _publish(
        svc,
        steps=_steps({"key": "a", "type": "decision", "config": "raw-string",
                      "start": True},
                     {"key": "b", "type": "message"}),
        transitions=[{"from": "a", "to": "b", "on": "x", "order": 0,
                      "guard": {"expr": "x>0"}}],
    )

    assert wf_repo.get_def(key="access-request", version="1") is not None


# §7 case 12 — the M-7 shape matrix. `_validate_def_spec` runs BEFORE serialization, so
# it sees `config`/`guard` heterogeneously typed: REST callers deliver **strings**,
# service/MCP callers deliver dicts. Both failure directions are real — naive dict
# access on a string is a 500 on `POST /workflow-defs`, while a bare `isinstance(...,
# dict)` skip would let **every REST-published def** escape both invariants with a green
# suite. These four cases pin that the front door cannot escape either one.

def test_shape_matrix_string_config_declaring_the_flag_is_accepted(wf_repo):
    svc = Services(wf_repo)

    _publish(
        svc,
        steps=_steps(
            {"key": "park", "type": "human", "config": '{"waitsForHuman": true}',
             "start": True},
            SINK_STEP,
        ),
        transitions=[SINK_TRANSITION],
    )

    assert wf_repo.get_def(key="access-request", version="1") is not None


def test_shape_matrix_string_config_without_the_flag_is_rejected(wf_repo):
    svc = Services(wf_repo)

    with pytest.raises(WorkflowDefSpecError, match="waitsForHuman"):
        _publish(
            svc,
            steps=_steps(
                {"key": "park", "type": "human", "config": "{}", "start": True},
                SINK_STEP,
            ),
            transitions=[SINK_TRANSITION],
        )

    assert wf_repo.get_def(key="access-request", version="1") is None


def test_shape_matrix_a_json_string_cmp_guard_is_validated_like_the_dict_form(wf_repo):
    svc = Services(wf_repo)

    with pytest.raises(WorkflowConfigError, match="unknown guard op"):
        _publish(
            svc,
            steps=_steps({"key": "a", "type": "decision", "start": True},
                         {"key": "b", "type": "decision"}),
            transitions=[{"from": "a", "to": "b", "on": "x", "order": 0,
                          "guard": '{"kind":"cmp","path":"ctx.a","op":"equals",'
                                   '"value":1}'}],
        )

    assert wf_repo.get_def(key="access-request", version="1") is None


def test_shape_matrix_an_opaque_config_on_a_parking_step_is_rejected(wf_repo):
    # A step that MUST declare `waitsForHuman` cannot carry a config that does not
    # normalize to a dict — there is nowhere for the declaration to live.
    svc = Services(wf_repo)

    with pytest.raises(WorkflowDefSpecError, match="waitsForHuman"):
        _publish(
            svc,
            steps=_steps(
                {"key": "park", "type": "human", "config": "raw-string",
                 "start": True},
                SINK_STEP,
            ),
            transitions=[SINK_TRANSITION],
        )

    assert wf_repo.get_def(key="access-request", version="1") is None
