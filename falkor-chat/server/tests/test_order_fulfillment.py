"""Acceptance: the `order-fulfillment@v1` process flow (K-053 M6, `docs/plans/
workflow-cart-and-totals.md` §3.4/§5 — AC-7/AC-8).

Mirrors `test_process_flow.py`'s own pattern exactly: the def under test is
imported from `falkorchat.proof_defs` — the *same* constant
`scripts/seed_salesperson.sh` publishes (plan §3.4/§4) — published through the
real `publish_workflow_def`/`materialize_def`, driven with real
`submit_workflow_input` calls, no LLM, no network, no `live` marker (every
step is `human`/`decision`, no `agent` step at all).

**What this flow proves, and what it deliberately does not.** The def's own
steps have no side effect on the `Order` node (DESIGN §6.1 — `human`/
`decision` steps never write outside `ctx`); advancing `Order.status` is a
*separate* guarded-CAS call (`services.advance_order`), the "two-step,
accepted" pairing `workflow-cart-and-totals-graph.md` §4 describes and
explicitly leaves to "whichever endpoint accepts the operator's fulfillment
input" — not built in this cluster (no new REST route). So every test below
drives **both** halves explicitly, exactly as that future endpoint would:
`submit_workflow_input` advances the *run*, `advance_order` advances the
*Order* — proving each half is correct and that they agree, without assuming
a single call does both.
"""

from __future__ import annotations

import itertools

import pytest

from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.proof_defs import ORDER_FULFILLMENT_DEF, ORDER_FULFILLMENT_MAX_STEPS
from falkorchat.services import Services, WorkflowInputRejectedError

CTX = CallContext(ws="test", actor="u1")

KEY = ORDER_FULFILLMENT_DEF["key"]
#: Test-only version (mirrors `test_process_flow.py`'s `v1-test` convention) —
#: `conftest.wf_repo` wipes the shared global `reference` graph at fixture
#: *setup*, so a finished pytest session leaves the last workflow test's defs
#: behind; publishing the production `order-fulfillment@v1` from here would
#: make `seed_salesperson.sh`'s "already present — no-op" line untrustworthy.
VERSION = "v1-test"
TEST_DEF = {**ORDER_FULFILLMENT_DEF, "version": VERSION}

CUSTOMER_ID = "cust-of-1"
ORDER_ID = "order-of-1"


def _make_services(repo):
    """Real service + real engine, deterministic ids/clocks, no LLM (mirrors
    `test_process_flow.py`'s `_make_services`)."""
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
    services = _make_services(wf_repo)
    services.publish_workflow_def(CTX, **TEST_DEF)
    services.materialize_def(CTX, key=KEY, version=VERSION)
    return services


def _seed_order(repo, *, order_id=ORDER_ID, customer_id=CUSTOMER_ID, now=500):
    """An already-placed `Order` (repository.place_order, K-053 cluster 1) —
    the fulfillment def manages an order's *lifecycle*, not its placement, so
    this seeds the precondition directly rather than through the cart tools."""
    repo.ensure_customer("test", customer_id=customer_id, now=now)
    repo.place_order(
        "test", customer_id=customer_id, order_id=order_id, now=now,
        lines=[{
            "productId": "p1", "name": "Widget", "unitPrice": 9.99,
            "quantity": 2, "lineTotal": 19.98,
        }],
    )


def _start(svc, **kw):
    kw.setdefault("max_steps", ORDER_FULFILLMENT_MAX_STEPS)
    return svc.start_workflow_run(CTX, def_key=KEY, version=VERSION, **kw)


def _trail(repo, run_id):
    return [sr["stepKey"] for sr in repo.read_step_runs("test", run_id=run_id)]


def _assert_parked(repo, run_id, *, step_key):
    run = repo.get_run("test", run_id=run_id)
    assert run["status"] == "waiting"
    assert run["atStepKey"] == step_key
    assert run["endedAt"] is None
    return run


def _assert_completed(repo, run_id, *, terminal, step_count):
    run = repo.get_run("test", run_id=run_id)
    assert run["status"] == "done"
    assert run["atStepKey"] is None
    assert run["endedAt"] is not None
    assert run["stepCount"] == step_count
    assert _trail(repo, run_id)[-1] == terminal
    return run


# ── the def itself publishes clean, and republishes as a no-op ─────────────


def test_order_fulfillment_def_publishes_clean_through_validate_def_spec(wf_repo):
    """`_validate_def_spec`'s invariants all pass for the real shipped
    constant: exactly one start step (`placed`), >= 1 transition, every
    parking step (`placed`/`fulfilled`) resolves via its own `config.
    waitsForHuman` — exercised through the real `publish_workflow_def`, not a
    copy of the spec (the `svc` fixture already does this at setup; this test
    asserts on the shape it produced rather than assuming it)."""
    services = _make_services(wf_repo)

    pub = services.publish_workflow_def(CTX, **TEST_DEF)

    assert (pub["key"], pub["version"]) == (KEY, VERSION)
    assert pub["stepCount"] == 4       # placed, fulfilled, delivered, cancelled
    assert pub["transitionCount"] == 3  # fulfill, cancel, deliver

    mat = services.materialize_def(CTX, key=KEY, version=VERSION)
    snap = services.get_snapshot(CTX, key=KEY, version=VERSION)
    assert mat["stepCount"] == 4
    assert snap["start_key"] == "placed"
    assert {s["key"]: s["type"] for s in snap["steps"]} == {
        "placed": "decision", "fulfilled": "human",
        "delivered": "decision", "cancelled": "decision",
    }
    assert len(snap["transitions"]) == 3


def test_republish_of_a_byte_identical_topology_is_a_clean_no_op(wf_repo):
    """Same property `test_salesperson_scaffold.py` proves for `salesperson`:
    republishing the exact same (key, version) with byte-identical content is
    a harmless, structurally-identical no-op."""
    services = _make_services(wf_repo)

    first = services.publish_workflow_def(CTX, **TEST_DEF)
    second = services.publish_workflow_def(CTX, **TEST_DEF)

    assert second == first


# ── AC-7 — the happy path: placed -> fulfilled -> delivered ────────────────


def test_fulfill_then_deliver_via_explicit_human_resumes(svc, wf_repo):
    _seed_order(wf_repo)
    started = _start(svc)
    run_id = started["runId"]
    assert started["status"] == "waiting"
    _assert_parked(wf_repo, run_id, step_key="placed")
    assert wf_repo.get_order("test", order_id=ORDER_ID)["status"] == "placed"

    # ── the operator marks it fulfilled: advance the RUN, then the ORDER ──
    out = svc.submit_workflow_input(CTX, run_id=run_id, input={"action": "fulfill"})
    assert out["status"] == "waiting"
    _assert_parked(wf_repo, run_id, step_key="fulfilled")
    order = svc.advance_order(CTX, order_id=ORDER_ID, transition="fulfill")
    assert order["status"] == "fulfilled"
    assert wf_repo.get_order("test", order_id=ORDER_ID)["status"] == "fulfilled"

    # ── the operator marks it delivered: same pairing ──
    out = svc.submit_workflow_input(CTX, run_id=run_id, input={"action": "deliver"})
    assert out["status"] == "done"
    order = svc.advance_order(CTX, order_id=ORDER_ID, transition="deliver")
    assert order["status"] == "delivered"
    assert wf_repo.get_order("test", order_id=ORDER_ID)["status"] == "delivered"

    # placed parks (1), fires fulfill (1) = 2; fulfilled parks (1), fires
    # deliver (1) = 2; delivered terminal = 1. Total 5 (§3.4's own accounting).
    _assert_completed(wf_repo, run_id, terminal="delivered", step_count=5)
    assert _trail(wf_repo, run_id) == [
        "placed", "placed", "fulfilled", "fulfilled", "delivered",
    ]


# ── AC-8 — a cancel before fulfillment blocks a later deliver ──────────────


def test_cancel_before_fulfillment_ends_the_run_and_blocks_a_later_deliver(
    svc, wf_repo
):
    _seed_order(wf_repo)
    started = _start(svc)
    run_id = started["runId"]
    _assert_parked(wf_repo, run_id, step_key="placed")

    out = svc.submit_workflow_input(CTX, run_id=run_id, input={"action": "cancel"})
    assert out["status"] == "done"
    _assert_completed(wf_repo, run_id, terminal="cancelled", step_count=3)
    assert _trail(wf_repo, run_id) == ["placed", "placed", "cancelled"]
    assert "fulfilled" not in _trail(wf_repo, run_id)

    order = svc.advance_order(CTX, order_id=ORDER_ID, transition="cancel")
    assert order["status"] == "cancelled"

    # The run is terminal (`done`) — there is no parked step left to submit
    # input against, so a later "deliver" is structurally unreachable at the
    # run layer.
    with pytest.raises(Exception):
        svc.submit_workflow_input(CTX, run_id=run_id, input={"action": "deliver"})

    # AC-8's own "cannot cancel once fulfilled" is the Order-status CAS's own
    # guard (graph note §3.4) — the complementary direction: fulfill first,
    # THEN attempt cancel, and the guarded CAS finds zero rows (no-op, `None`)
    # because it only matches from `status = 'placed'`. `advance_order`
    # operates purely on `order_id` (no workflow run involved), so a second,
    # independently-seeded `Order` is enough — no second run/def needed.
    order_id_2 = "order-of-2"
    _seed_order(wf_repo, order_id=order_id_2, customer_id="cust-of-2")
    assert svc.advance_order(
        CTX, order_id=order_id_2, transition="fulfill"
    )["status"] == "fulfilled"
    assert svc.advance_order(CTX, order_id=order_id_2, transition="cancel") is None
    assert wf_repo.get_order("test", order_id=order_id_2)["status"] == "fulfilled"


# ── AC-7 — a parked run never changes status absent an explicit resume ─────


def test_a_parked_run_never_auto_advances(svc, wf_repo):
    _seed_order(wf_repo)
    started = _start(svc)
    run_id = started["runId"]
    run_before = _assert_parked(wf_repo, run_id, step_key="placed")

    # Neither `placed` nor `fulfilled` declares a timer key (`waitForSeconds`/
    # `waitUntil`) — the K-028 sweep is due-agnostic at the read but excludes
    # any candidate without one (services.py docstring above `checked`); this
    # def is "today's forever-park def [that] is never swept."
    swept = svc.sweep_due_workflow_runs(CTX)
    assert swept["due"] == 0
    assert swept["resumed"] == []

    run_after = wf_repo.get_run("test", run_id=run_id)
    assert run_after["status"] == "waiting"
    assert run_after["atStepKey"] == "placed"
    assert run_after["stepCount"] == run_before["stepCount"]
    assert wf_repo.get_order("test", order_id=ORDER_ID)["status"] == "placed"


# ── typos are free (D-H), mirroring test_process_flow.py's own property ────


def test_an_unexpected_action_value_is_rejected_free_no_step_consumed(svc, wf_repo):
    _seed_order(wf_repo)
    started = _start(svc)
    run_id = started["runId"]
    before = wf_repo.get_run("test", run_id=run_id)

    with pytest.raises(WorkflowInputRejectedError):
        svc.submit_workflow_input(CTX, run_id=run_id, input={"action": "explode"})

    after = wf_repo.get_run("test", run_id=run_id)
    assert after["stepCount"] == before["stepCount"]
    assert after["status"] == "waiting"
    assert after["atStepKey"] == "placed"


def test_deliver_is_rejected_free_while_parked_at_fulfilled(svc, wf_repo):
    _seed_order(wf_repo)
    started = _start(svc)
    run_id = started["runId"]
    svc.submit_workflow_input(CTX, run_id=run_id, input={"action": "fulfill"})
    before = wf_repo.get_run("test", run_id=run_id)

    # `fulfilled`'s own `expects` only allows "deliver" — "cancel" is not a
    # legal value there (there is no `fulfilled -> cancelled` transition,
    # AC-8's other half, at the run layer).
    with pytest.raises(WorkflowInputRejectedError):
        svc.submit_workflow_input(CTX, run_id=run_id, input={"action": "cancel"})

    after = wf_repo.get_run("test", run_id=run_id)
    assert after["stepCount"] == before["stepCount"]
    assert after["atStepKey"] == "fulfilled"
