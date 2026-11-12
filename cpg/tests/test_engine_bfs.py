"""P1-22 the walk's rows, unit tier: ``ccg.engine.bfs.walk`` over ``MemoryReader`` (``tests/engine_walk_helpers.py``, the same columns as ``CypherReader``).

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-22 behaviours (2)-(4), (6), (7); the row table of ``docs/plans/connected-code-graph-graph.md`` §6.2 plus
``-p0-graph.md`` §3.2; ``-answer-spec2.md`` §3.3 (wiring), §3.5 (``TO_UNIT``), §5. **Every fixture sets ``Unit.flowAvailable`` explicitly** (``G(flow=True)``):
the flag gate is P1-23's, and changes no pin here.
"""
from __future__ import annotations

import pytest
from engine_walk_helpers import G, MemoryReader

from ccg.engine import bfs
from ccg.engine import state as S
from ccg.engine.reader import GraphDataError


def run(g, anchor, state=None, **kw):
    return bfs.walk(MemoryReader(g), anchor, state, **kw)


def items_by_id(w):
    return {i.id: i for i in w.items()}


def nodes(item):
    return [s["node"] for s in item.chain]


def edge_types(item):
    return [(s["edge"]["type"], s["edge"]["dir"]) for s in item.chain[1:]]


# --------------------------------------------------------------------------- (1) calls, owners, Ret: the precise rows

def call_chain(*, ret_reached: bool):
    """cB (in B) calls the anchor A; cC (in C) calls B. With ``ret_reached`` B's ``rB`` returns cB's result."""
    g = G()
    for s in "ABC":
        g.sym(s)
    g.site("cB", "B")
    g.site("cC", "C")
    g.calls("cB", "A")
    g.calls("cC", "B")
    g.ret("rB", "B")
    if ret_reached:
        g.data("Ret", "rB", "CallSite", "cB", "r", "ret")
    return g


def test_a_direct_caller_is_an_item_at_distance_one_with_its_chain_and_the_anchor_is_not():
    w = run(call_chain(ret_reached=False), ("Symbol", "A"))
    items = items_by_id(w)
    assert set(items) == {"B"}
    b = items["B"]
    assert (b.kind, b.distance, b.status, b.reason, b.possible_hops, b.length) == ("symbol", 1, "confirmed", None, 0, 2)
    assert nodes(b) == [("Symbol", "A"), ("CallSite", "cB"), ("Symbol", "B")]
    assert edge_types(b) == [("CALLS", "<-"), ("HAS_CALL", "<-")]
    assert [s["state"] for s in b.chain] == [("value", ""), ("path", "r"), ("end", "")]
    assert b.chain[0]["edge"] is None


def test_a_caller_of_a_function_whose_return_the_change_does_not_reach_is_not_listed_through_it():
    # the slice (function-level) listed C as well; the precise rows end B at the point that held the change (V2 gate)
    assert "C" not in items_by_id(run(call_chain(ret_reached=False), ("Symbol", "A")))


def test_a_caller_through_a_ret_the_change_reaches_is_listed_at_distance_two_confirmed():
    w = run(call_chain(ret_reached=True), ("Symbol", "A"))
    items = items_by_id(w)
    assert set(items) == {"B", "C"}
    c = items["C"]
    assert (c.distance, c.status, c.possible_hops, c.length) == (2, "confirmed", 0, 5)
    assert nodes(c) == [("Symbol", "A"), ("CallSite", "cB"), ("Ret", "rB"), ("Symbol", "B"), ("CallSite", "cC"), ("Symbol", "C")]
    assert edge_types(c) == [("CALLS", "<-"), ("DATA", "<-"), ("HAS_RETURN", "<-"), ("CALLS", "<-"), ("HAS_CALL", "<-")]
    assert [s["state"] for s in c.chain] == [("value", ""), ("path", "r"), ("path", "ret"), ("value", ""), ("path", "r"), ("end", "")]
    assert c.via_data is True and items["B"].via_data is False


def test_the_direct_caller_stays_confirmed_whether_or_not_a_ret_is_reached():
    for ret in (True, False):
        b = items_by_id(run(call_chain(ret_reached=ret), ("Symbol", "A")))["B"]
        assert (b.status, b.reason, b.possible_hops) == ("confirmed", None, 0)


def test_edge_direction_is_against_the_stored_edge():
    g = G()
    g.sym("A"), g.sym("B"), g.site("cB", "B"), g.calls("cB", "A")
    assert [i.id for i in run(g, ("Symbol", "A")).items()] == ["B"]
    assert run(g, ("Symbol", "B")).items() == []


def test_an_anchor_called_by_itself_is_never_its_own_item():
    g = G()
    g.sym("A"), g.site("cA", "A"), g.calls("cA", "A")
    w = run(g, ("Symbol", "A"))
    assert w.items() == []
    assert ("CallSite", "cA", "path", "r") in w.visited


def test_a_symbol_entered_through_a_ret_and_through_a_site_is_one_item_on_its_cheaper_arrival():
    g = G()
    g.sym("A"), g.sym("B"), g.site("cB", "B"), g.calls("cB", "A"), g.ret("rB", "B"), g.data("Ret", "rB", "CallSite", "cB", "r", "ret")
    w = run(g, ("Symbol", "A"))
    assert [i.id for i in w.items()] == ["B"]
    assert w.items()[0].chain[-1]["state"] == ("end", "")   # the cheaper arrival: through the call site, not the longer way through the Ret


# --------------------------------------------------------------------------- access paths: DATA is path-matched

def field_graph():
    g = G()
    g.sym("F"), g.sym("G1"), g.sym("G2")
    g.param("p0", "F", index=0, name="opts")
    g.site("c1", "G1"), g.site("c2", "G2")
    g.data("CallSite", "c1", "Param", "p0", "p0.path", "a0")
    g.data("CallSite", "c2", "Param", "p0", "p0.method", "a1.method")
    return g


@pytest.mark.parametrize("path,reached,not_reached", [
    ("p0.path", {"G1"}, {"G2"}), ("p0.method", {"G2"}, {"G1"}), ("p0", {"G1", "G2"}, set()), ("p0.query", set(), {"G1", "G2"}),
])
def test_a_path_state_follows_only_the_data_edges_whose_from_path_overlaps(path, reached, not_reached):
    ids = {i.id for i in run(field_graph(), ("Param", "p0"), S.path_state(path)).items()}
    assert reached <= ids and not (not_reached & ids)
    assert ids - {"F"} == reached


def test_a_value_state_follows_every_data_edge_of_the_point():
    assert {i.id for i in run(field_graph(), ("Param", "p0"), S.VALUE).items()} - {"F"} == {"G1", "G2"}


# --------------------------------------------------------------------------- (3) cycle-safe traversal, F-12 and C1

def test_two_point_data_cycle_terminates_and_visits_each_state_once():
    g = G()
    g.sym("F"), g.site("c1", "F"), g.site("c2", "F")
    g.data("CallSite", "c2", "CallSite", "c1", "r", "r")
    g.data("CallSite", "c1", "CallSite", "c2", "r", "r")
    w = run(g, ("CallSite", "c1"), S.VALUE)
    assert w.visited == {("CallSite", "c1", "value", ""), ("CallSite", "c2", "path", "r"), ("CallSite", "c1", "path", "r"), ("Symbol", "F", "end", "")}
    assert w.levels == len(w.frontier_sizes) and sum(w.frontier_sizes) == len(w.visited)
    assert all(w.best[k].pred_key is None or w.best[k].pred_key in w.best for k in w.best)


def test_a_data_self_edge_terminates():
    g = G()
    g.sym("F"), g.site("c", "F")
    g.data("CallSite", "c", "CallSite", "c", "r", "r")
    w = run(g, ("CallSite", "c"), S.path_state("r"))
    assert w.visited == {("CallSite", "c", "path", "r"), ("Symbol", "F", "end", "")}


def cond_cycle():
    g = G()
    g.sym("F"), g.cond("k1", "F"), g.cond("k2", "F")
    g.controlled("Cond", "k2", "k1")
    g.controlled("Cond", "k1", "k2")
    return g


def test_c1_a_cond_to_cond_controlled_by_cycle_through_the_anchor_terminates_with_no_item_equal_to_the_anchor():
    w = run(cond_cycle(), ("Cond", "k1"), S.VALUE)
    assert w.visited == {("Cond", "k1", "value", ""), ("Cond", "k2", "value", ""), ("Symbol", "F", "end", "")}
    assert [i.id for i in w.items()] == ["F"]
    assert all(i.id != "k1" for i in w.items())
    # the anchor is entered once: its best entry has no predecessor, so a chain always ends at it
    assert w.best[("Cond", "k1", "value", "")].pred is None and w.chain(("Cond", "k2", "value", ""))[0]["node"] == ("Cond", "k1")


def test_a_controlled_by_self_edge_terminates():
    g = G()
    g.sym("F"), g.cond("k", "F"), g.controlled("Cond", "k", "k")
    assert run(g, ("Cond", "k"), S.VALUE).visited == {("Cond", "k", "value", ""), ("Symbol", "F", "end", "")}


def test_controlled_by_reaches_the_points_a_changed_condition_guards_at_distance_zero():
    g = G()
    g.sym("F"), g.sym("H"), g.cond("k", "F"), g.site("cg", "F"), g.ret("rg", "F"), g.param("pg", "H", index=0), g.site("ch", "H")
    g.controlled("CallSite", "cg", "k"), g.controlled("Ret", "rg", "k"), g.controlled("Param", "pg", "k")
    w = run(g, ("Cond", "k"), S.VALUE)
    assert {k[:2] for k in w.visited} >= {("CallSite", "cg"), ("Ret", "rg"), ("Param", "pg"), ("Symbol", "F"), ("Symbol", "H")}
    assert all(i.distance == 0 for i in w.items())
    assert all(s["edge"]["type"] in ("CONTROLLED_BY", "HAS_CALL", "HAS_RETURN", "HAS_PARAM", "HAS_COND") for i in w.items() for s in i.chain[1:])
    assert nodes(items_by_id(w)["F"]) == [("Cond", "k"), ("Symbol", "F")] and edge_types(items_by_id(w)["F"]) == [("HAS_COND", "<-")]


def test_a_controlled_cond_chain_continues_to_the_callers_through_a_guarded_ret():
    g = G()
    g.sym("A"), g.sym("F"), g.sym("X")
    g.cond("k", "F"), g.ret("rg", "F"), g.site("cx", "X"), g.calls("cx", "F")
    g.controlled("Ret", "rg", "k")
    w = run(g, ("Cond", "k"), S.VALUE)
    assert items_by_id(w)["X"].distance == 1  # Cond -> guarded Ret -> F (value) -> CALLS <- cx -> X


def test_an_equal_cost_second_arrival_is_not_re_expanded_and_keeps_its_first_chain():
    """cx is reached directly (length 1) and again through a DATA hop at the same (possible, distance) but length 2: the first chain stays (``<`` not ``<=``)."""
    g = G()
    g.sym("A"), g.sym("F"), g.site("c1", "F"), g.site("cx", "F"), g.calls("c1", "A"), g.calls("cx", "A")
    g.data("CallSite", "cx", "CallSite", "c1", "r", "r")
    w = run(g, ("Symbol", "A"))
    best = w.best[("CallSite", "cx", "path", "r")]
    assert best.cost == (0, 1, 1, 0) and best.pred_key == ("Symbol", "A", "value", "") and best.edge.type == "CALLS"
    assert w.frontier_sizes == [1, 2, 1]  # A; c1, cx; F (once: no re-expansion)


def test_a_strictly_smaller_later_arrival_is_re_expanded_and_improves_everything_downstream():
    """u is first reached through a possible DATA edge (1 possible hop, level 1); a confirmed route through w arrives a level later with 0 possible hops."""
    g = G()
    g.sym("F"), g.param("p", "F", index=0)
    g.sym("G1"), g.sym("G0"), g.site("u", "G1"), g.site("w", "G0")
    g.data("CallSite", "u", "Param", "p", "p0", "a0", status="possible", reason="field-truncated")
    g.data("CallSite", "w", "Param", "p", "p0", "a9")
    g.data("CallSite", "u", "CallSite", "w", "a9", "a0")
    w_ = run(g, ("Param", "p"), S.path_state("p0"))
    u = w_.best[("CallSite", "u", "path", "a0")]
    assert u.cost[:2] == (0, 0) and u.pred_key == ("CallSite", "w", "path", "a9") and u.edge.status == "confirmed"
    g1 = items_by_id(w_)["G1"]
    assert (g1.status, g1.possible_hops, g1.length) == ("confirmed", 0, 3) and g1.reason is None
    assert nodes(g1)[-3:] == [("CallSite", "w"), ("CallSite", "u"), ("Symbol", "G1")]


def test_an_item_reached_in_two_states_at_the_same_cost_takes_the_smaller_key():
    """B is entered through a call site (state ``end``) and through a Ret (state ``value``) at the same cost: the item is the smaller key, whatever the read order."""
    g = G()
    g.sym("B"), g.sym("Q"), g.param("p", "Q", index=0), g.site("c", "B"), g.ret("r", "B")
    g.data("CallSite", "c", "Param", "p", "p0", "a0"), g.data("Ret", "r", "Param", "p", "p0", "ret")
    w = run(g, ("Param", "p"), S.path_state("p0"))
    assert w.best[("Symbol", "B", "end", "")].cost == w.best[("Symbol", "B", "value", "")].cost
    (b,) = [i for i in w.items() if i.id == "B"]
    assert b.chain[-1]["state"] == ("end", "") and nodes(b)[1] == ("CallSite", "c")


def test_the_item_reason_is_the_first_possible_hops_not_the_last():
    """The link is possible (``port-shared``) and its consumer edge is possible with another reason: the item says the nearer one."""
    g = G()
    g.sym("P"), g.sym("F"), g.site("cs", "F")
    g.link("L", provider="P", status="possible", reason="port-shared"), g.via("cs", "L", status="possible", reason="example-config")
    f = items_by_id(run(g, ("Symbol", "P")))["F"]
    assert (f.status, f.reason, f.possible_hops) == ("possible", "port-shared", 2)
    assert [s["edge"]["reason"] for s in f.chain[1:3]] == ["port-shared", "example-config"]


def test_a_confirmed_item_is_the_cheaper_even_when_a_possible_route_is_shorter():
    """F is reached by a name-matched site (possible, length 2) and by a longer confirmed route through a Ret: fewer possible hops wins over length."""
    g = G()
    g.sym("A"), g.sym("B"), g.sym("F")
    g.site("c_ok", "B"), g.calls("c_ok", "A"), g.ret("rb", "B"), g.data("Ret", "rb", "CallSite", "c_ok", "r", "ret")
    g.site("c_f1", "F"), g.calls("c_f1", "B")                         # F calls B: confirmed, reached through B's Ret (value)
    g.site("c_f2", "F", callee="a"), g.calls_name("c_f2", "A", "a")  # F also name-matches A: possible, distance 1, shorter
    f = items_by_id(run(g, ("Symbol", "A")))["F"]
    assert (f.status, f.possible_hops, f.distance) == ("confirmed", 0, 2)
    assert w_best_cost(g, ("CallSite", "c_f2", "path", "r")) == (1, 1, 1, 1)


def w_best_cost(g, key):
    return run(g, ("Symbol", "A")).best[key].cost
