"""P1-23 behaviour (1): the flow fallback (``docs/plans/connected-code-graph-p1-answers.md`` P1-23; ``-answer-spec2.md`` §5; ``-p1-ml.md`` §3.5, §6.6).

Inside a unit whose ``Unit.flowAvailable`` is true ``DATA``/``CONTROLLED_BY``/``arg(i)`` rows are followed ``confirmed``; inside a flag-false unit they are not, a
function whose callers are reached only at call level is in the ``function`` state, and only the closed list of rows leaving that state is labelled ``possible``
(``flow-unavailable:<lang>`` when the hop's own edge is confirmed). Every fixture here sets its units' flags and languages explicitly.
"""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import pytest
from engine_walk_helpers import G, MemoryReader

from ccg.engine import bfs
from ccg.engine import state as S
from test_engine_bfs import edge_types, items_by_id, nodes, run


def states(item):
    return [s["state"][0] for s in item.chain]


def edge_of(item, i):
    return item.chain[i]["edge"]


def caller_chain(*, flow: bool, lang: str = "javascript", caller_flow: bool | None = None, caller_lang: str | None = None):
    """``F`` holds the changed point ``pF``; ``G`` calls ``F`` (resolved) and ``H`` calls ``G`` (resolved). ``F`` lives in a unit of ``flow``/``lang``; the callers
    live in a unit of their own when ``caller_flow`` is given."""
    g = G(unit="uf", flow=flow, lang=lang)
    if caller_flow is not None:
        g.unit_node("uc", flow=caller_flow, lang=caller_lang or lang)
    cu = "uc" if caller_flow is not None else None
    g.sym("F"), g.param("pF", "F")
    g.sym("G", unit=cu), g.site("cG", "G", unit=cu), g.calls("cG", "F")
    g.sym("H", unit=cu), g.site("cH", "H", unit=cu), g.calls("cH", "G")
    return g


def walk_param(g, **kw):
    return run(g, ("Param", "pF"), S.VALUE, **kw)


# --------------------------------------------------------------------------- (a) the labelled rows leaving ``function``

def test_in_a_flag_false_unit_the_callers_of_a_function_holding_the_changed_point_are_possible_flow_unavailable_and_so_are_their_callers():
    items = items_by_id(walk_param(caller_chain(flow=False)))
    assert items["F"].status == "confirmed"                                   # the function that holds the changed point
    for who, dist in (("G", 1), ("H", 2)):
        assert (items[who].status, items[who].reason, items[who].distance) == ("possible", "flow-unavailable:javascript", dist)
    assert states(items["H"]) == ["value", "function", "value", "function", "value", "function"]


def test_the_labelled_hop_is_possible_and_the_hops_around_it_keep_their_own_status():
    g = caller_chain(flow=False)
    item = items_by_id(walk_param(g))["G"]
    assert [(s["node"][0], s["edge"]["type"], s["edge"]["dir"], s["edge"]["status"]) for s in item.chain[1:]] == [
        ("Symbol", "HAS_PARAM", "<-", "confirmed"),            # the point's owner: enters ``function``
        ("CallSite", "CALLS", "<-", "possible"),               # the labelled row leaving ``function``
        ("Symbol", "HAS_CALL", "<-", "confirmed")]             # the caller's own owner hop: its status is its own
    assert edge_of(item, 2)["reason"] == "flow-unavailable:javascript" and "reason" not in edge_of(item, 1)


def test_a_flag_true_unit_enters_no_function_state_and_every_hop_stays_confirmed():
    w = walk_param(caller_chain(flow=True))
    items = items_by_id(w)
    assert {i.status for i in items.values()} == {"confirmed"} and {i.reason for i in items.values()} == {None}
    assert all(k[2] != "function" for k in w.visited)


def test_the_lang_is_the_language_of_the_unit_of_the_function_hop_not_of_the_caller():
    items = items_by_id(walk_param(caller_chain(flow=False, lang="javascript", caller_flow=True, caller_lang="python")))
    assert items["G"].reason == "flow-unavailable:javascript"


def test_a_caller_in_a_flag_true_unit_is_entered_precisely_so_it_does_not_enter_function_and_its_own_callers_are_not_listed():
    """The labelled hop is the one leaving ``function`` (a flag-false unit's function); a flag-true caller is reached by the precise rows: its owner hop ends in
    ``end`` (no ``function`` state, so no callers of it), the possible hop it carries is the labelled one."""
    w = walk_param(caller_chain(flow=False, lang="javascript", caller_flow=True, caller_lang="python"))
    items = items_by_id(w)
    assert (items["G"].status, items["G"].reason) == ("possible", "flow-unavailable:javascript") and "H" not in items
    assert ("Symbol", "G", "end", "") in w.visited and ("Symbol", "G", "function", "") not in w.visited
    assert states(items["G"]) == ["value", "function", "path", "end"]


# --------------------------------------------------------------------------- the flag gate: which rows a flag-false unit does not follow

def point_rows_graph(flow: bool):
    """``F`` (flag ``flow``) holds the changed param ``pF``: a ``DATA`` use at call site ``cD`` of ``D``, a ``CONTROLLED_BY`` guard that ``cK`` (in ``K``) sits under, and
    the argument path into the callee ``E``; nothing calls ``F``, so only the precise rows reach anything."""
    g = G(unit="uf", flow=flow)
    for s in ("F", "D", "K", "E"):
        g.sym(s) if s == "F" else g.sym(s, unit="ue" if s == "E" else None)
    g.unit_node("ue", flow=True)
    g.param("pF", "F", index=0, name="x"), g.cond("kF", "F")
    g.site("cD", "D"), g.data("CallSite", "cD", "Param", "pF", "p0", "a0")        # pF flows into cD's argument
    g.site("cK", "K"), g.controlled("CallSite", "cK", "kF"), g.data("Cond", "kF", "Param", "pF", "p0", "")      # a guard that reads pF, and a site under it
    g.calls("cD", "E")
    return g


def test_in_a_flag_true_unit_data_and_controlled_by_rows_are_followed_confirmed():
    w = walk_param(point_rows_graph(True), column=bfs.IMPACT)
    items = items_by_id(w)
    assert {"F", "D", "K"} <= set(items) and {i.status for i in items.values()} == {"confirmed"}
    assert items["D"].via_data and any(s["edge"] and s["edge"]["type"] == "CONTROLLED_BY" for s in items["K"].chain)


def test_in_a_flag_false_unit_data_controlled_by_and_the_argument_rows_are_not_followed():
    w = run(point_rows_graph(False), ("Param", "pF"), S.path_state("p0"))
    assert set(items_by_id(w)) == {"F"}
    assert {k[0] for k in w.visited} == {"Param", "Symbol"}
    assert all(s["edge"] is None or s["edge"]["type"] not in ("DATA", "CONTROLLED_BY") for i in w.items() for s in i.chain)


def test_the_argument_rows_are_followed_only_between_flag_true_units():
    """``CALLS ->`` under an argument path enters a callee: the callee's unit must be flag-true as well (a flag-false callee has no ``path`` state to enter)."""
    def graph(callee_flow):
        g = G(unit="ux", flow=True)
        g.unit_node("uf2", flow=callee_flow)
        g.sym("X"), g.sym("F2", unit="uf2"), g.param("px", "X", index=0, name="o"), g.param("pf", "F2", index=0, name="o")
        g.site("cx", "X"), g.calls("cx", "F2"), g.data("CallSite", "cx", "Param", "px", "p0", "a0")
        return g
    assert "F2" in items_by_id(run(graph(True), ("Param", "px"), S.path_state("p0")))
    off = run(graph(False), ("Param", "px"), S.path_state("p0"))
    assert "F2" not in items_by_id(off) and all(k[1] != "F2" for k in off.visited)


def test_no_hop_in_a_flag_false_unit_has_state_path():
    """Not a row of the shipped table can put a ``path`` state in a flag-false unit: the call sites a labelled row reaches are in ``value`` state there."""
    g = caller_chain(flow=False)
    g.data("CallSite", "cG", "Param", "pF", "p0", "a0")     # a data edge a flag-true unit would follow
    w = walk_param(g)
    assert all(k[2] != "path" for k in w.visited if k[0] != "Param")
    assert ("CallSite", "cG", "value", "") in w.visited and ("CallSite", "cG", "path", "r") not in w.visited


def test_an_absent_flag_reads_false_and_a_unit_with_no_row_is_a_graph_data_error():
    g = caller_chain(flow=True)
    del next(n for n in g.rows.nodes["Unit"] if n["unitId"] == "uf")["flowAvailable"]
    assert items_by_id(walk_param(g))["G"].reason == "flow-unavailable:javascript"
    g2 = caller_chain(flow=True)
    g2.rows.nodes["Unit"].clear()
    with pytest.raises(bfs.GraphDataError, match="no Unit row"):
        walk_param(g2)


def test_a_point_with_no_owner_has_no_unit_and_a_gated_row_from_it_is_a_graph_data_error():
    g = G()
    g.cond("k", "S"), g.node("Cond", pointId="orphan", line=1, code="if y")
    g.sym("S")
    with pytest.raises(bfs.GraphDataError, match="no unit"):
        run(g, ("Cond", "orphan"), S.VALUE)


# --------------------------------------------------------------------------- the closed list of labelled rows (impact)

def test_a_direct_resolved_caller_of_an_anchor_symbol_is_not_labelled_but_its_callers_are():
    g = G(unit="uf", flow=False, lang="python")
    g.sym("A"), g.sym("B"), g.site("cB", "B"), g.calls("cB", "A"), g.sym("C"), g.site("cC", "C"), g.calls("cC", "B")
    items = items_by_id(run(g, ("Symbol", "A")))
    assert (items["B"].status, items["B"].reason, items["B"].possible_hops) == ("confirmed", None, 0)
    assert (items["C"].status, items["C"].reason, items["C"].possible_hops, items["C"].distance) == ("possible", "flow-unavailable:python", 1, 2)
    assert states(items["C"]) == ["value", "value", "function", "value", "function"]


def test_a_name_matched_caller_leaving_function_is_possible_with_its_own_reason_never_flow_unavailable():
    g = G(unit="uf", flow=False)
    g.sym("F"), g.param("pF", "F")
    g.sym("N"), g.site("cN", "N", callee="f"), g.calls_name("cN", "F", "f")                           # unresolved: name-match
    g.sym("M"), g.site("cM", "M", callee="f"), g.calls("cM", "Z"), g.calls_name("cM", "F", "f")       # resolved elsewhere: resolution-unverified
    g.sym("Z")
    items = items_by_id(walk_param(g))
    assert (items["N"].status, items["N"].reason) == ("possible", "name-match") and (items["M"].status, items["M"].reason) == ("possible", "resolution-unverified")
    assert edge_of(items["N"], 2) == {"type": "CALLS_NAME", "dir": "<-", "status": "possible", "reason": "name-match"}


def test_to_leaving_function_is_labelled_with_the_links_own_status_and_reason():
    for status, reason, want in (("confirmed", "key-match", "flow-unavailable:python"), ("possible", "literal-key", "literal-key")):
        g = G(unit="uf", flow=False)
        g.sym("P"), g.param("pP", "P"), g.cp("cp1"), g.sym("U"), g.site("cU", "U"), g.link("L", provider="P", cp="cp1", status=status, reason=reason), g.via("cU", "L", status=status, reason=reason)
        w = run(g, ("Param", "pP"), S.VALUE)
        link = next(s for i in w.items() for s in i.chain if s["node"] == ("Link", "L"))
        assert (link["edge"]["type"], link["edge"]["dir"], link["edge"]["status"], link["edge"]["reason"], link["state"]) == ("TO", "<-", "possible", want, ("value", ""))


def test_a_contract_consumer_through_a_flag_false_unit_stays_confirmed_because_no_labelled_row_is_on_its_chain():
    g = G(unit="uf", flow=False)
    g.sym("P"), g.cp("cp1"), g.provides("P", "cp1"), g.sym("U"), g.site("cU", "U"), g.link("L", provider="P", cp="cp1"), g.via("cU", "L")
    items = items_by_id(run(g, ("ContractPoint", "cp1")))
    assert (items["U"].status, items["U"].reason, items["U"].possible_hops) == ("confirmed", None, 0)
    assert "function" not in {k[2] for k in run(g, ("ContractPoint", "cp1")).visited}


def test_only_the_function_state_exits_are_labelled_so_a_function_reached_by_ret_enters_value_and_its_callers_stay_confirmed():
    g = G(unit="uf", flow=False)
    g.sym("F"), g.ret("rF", "F"), g.sym("G"), g.site("cG", "G"), g.calls("cG", "F")
    items = items_by_id(run(g, ("Ret", "rF"), S.VALUE))
    assert (items["F"].status, items["G"].status, items["G"].reason) == ("confirmed", "confirmed", None)


# --------------------------------------------------------------------------- the depends-on column

def callee_graph(callee_flow: bool):
    """``F`` (flag-true) calls ``K`` (``callee_flow``, lang go); ``K`` calls ``J`` and holds a param, a return and a guard that ``J``'s site sits under."""
    g = G(unit="uf", flow=True)
    g.unit_node("uk", flow=callee_flow, lang="go")
    g.sym("F"), g.sym("K", unit="uk"), g.sym("J"), g.site("cF", "F"), g.calls("cF", "K")
    g.site("cK", "K", unit="uk"), g.calls("cK", "J"), g.param("pK", "K"), g.ret("rK", "K"), g.cond("kK", "K")
    g.sym("T"), g.site("cK2", "K", unit="uk"), g.calls("cK2", "T")
    return g


def test_a_callee_in_a_flag_false_unit_is_in_function_state_and_only_has_call_leaves_it_labelled():
    w = run(callee_graph(False), ("Symbol", "F"), column=bfs.DEPENDS_ON)
    items = items_by_id(w)
    assert (items["K"].status, items["K"].distance) == ("confirmed", 1)
    for j in ("J", "T"):
        assert (items[j].status, items[j].reason, items[j].distance) == ("possible", "flow-unavailable:go", 2)
    assert states(items["J"]) == ["value", "value", "function", "value", "value"]
    assert edge_of(items["J"], 3) == {"type": "HAS_CALL", "dir": "->", "status": "possible", "reason": "flow-unavailable:go"}
    assert {k[0] for k in w.visited if k[1] in ("pK", "rK", "kK")} == set()      # HAS_PARAM/HAS_RETURN/HAS_COND leave no flag-false function


def test_a_callee_in_a_flag_true_unit_is_in_value_state_and_its_whole_containment_is_followed():
    w = run(callee_graph(True), ("Symbol", "F"), column=bfs.DEPENDS_ON)
    items = items_by_id(w)
    assert {i.status for i in items.values()} == {"confirmed"} and "function" not in {k[2] for k in w.visited}
    assert {k[1] for k in w.visited if k[0] in ("Param", "Ret", "Cond")} == {"pK", "rK", "kK"}


def test_the_language_is_the_function_hops_unit_in_the_depends_on_column_too():
    g = callee_graph(False)
    g.unit_node("uj", flow=True, lang="rust")
    next(n for n in g.rows.nodes["Symbol"] if n["symbolId"] == "J")["unitId"] = "uj"
    assert items_by_id(run(g, ("Symbol", "F"), column=bfs.DEPENDS_ON))["J"].reason == "flow-unavailable:go"


def test_a_flag_false_function_calling_into_a_flag_true_callee_leaves_nothing_labelled():
    """The labelled row is the one *leaving* ``function``: ``HAS_CALL ->`` from an anchor ``Symbol`` (value state, flag-false) and ``CALLS ->`` into a flag-true callee
    are plain rows."""
    g = G(unit="uf", flow=False)
    g.unit_node("ut", flow=True)
    g.sym("F"), g.sym("K", unit="ut"), g.site("cF", "F"), g.calls("cF", "K")
    k = items_by_id(run(g, ("Symbol", "F"), column=bfs.DEPENDS_ON))["K"]
    assert (k.status, k.reason, k.possible_hops) == ("confirmed", None, 0) and states(k)[-1] == "value"


# --------------------------------------------------------------------------- (c) the hop's labelled status is not its own edge's status

def two_site_caller(*, name_site_first: bool):
    """``C`` reaches the function ``F`` through two sites: ``c_name`` (a name match) and ``c_resolved`` (a resolved call). ``name_site_first`` gives the name-matched
    site the smaller stable key (``-answer-spec2.md`` §5: otherwise the stable key alone picks the resolved site and the test cannot see the tie-break dropped)."""
    g = G(unit="uf", flow=False)
    g.sym("F"), g.param("pF", "F"), g.sym("C"), g.sym("Z")
    first, second = ("c_a_name", "c_z_resolved") if name_site_first else ("c_z_name", "c_a_resolved")
    g.site(first, "C", callee="f"), g.calls_name(first, "F", "f")
    g.site(second, "C", callee="f"), g.calls(second, "F")
    return g, first, second


@pytest.mark.parametrize("name_site_first", [True, False])
def test_the_resolved_chain_wins_when_the_name_matched_site_sorts_first_and_both_hops_are_labelled(name_site_first):
    g, name_site, resolved_site = two_site_caller(name_site_first=name_site_first)
    assert (name_site < resolved_site) == name_site_first
    c = items_by_id(walk_param(g))["C"]
    assert [s["node"] for s in c.chain][2] == ("CallSite", resolved_site)
    assert edge_of(c, 2) == {"type": "CALLS", "dir": "<-", "status": "possible", "reason": "flow-unavailable:python"}
    assert (c.status, c.reason, c.possible_hops) == ("possible", "flow-unavailable:python", 1)


def test_both_arrivals_cost_the_same_up_to_the_own_edge_so_only_the_own_possible_term_decides():
    g, name_site, resolved_site = two_site_caller(name_site_first=True)
    w = walk_param(g)
    arrivals = w.arrivals[("Symbol", "C", "function", "")]
    assert len(arrivals) == 1                                           # the pair is equal on (possible hops, distance): one is dominated, the sort decides which
    costs = {}
    for site in (name_site, resolved_site):
        (entry,) = w.arrivals[("CallSite", site, "value", "")]
        costs[site] = entry.cost
    assert costs[name_site][:3] == costs[resolved_site][:3] and costs[name_site][3] == costs[resolved_site][3] + 1
    assert costs[name_site][0] == costs[resolved_site][0] == 1             # both hops are labelled: the possible-hops count cannot tell them apart
    assert arrivals[0].pred_key[1] == resolved_site


# --------------------------------------------------------------------------- (d) equal-cost chains resolve by the predecessor's stable key

class Shuffled:
    """``MemoryReader`` with every reply reshuffled by ``seed`` (and ``reverse``): a walk must not depend on the order a reply arrives in (FR-19)."""

    def __init__(self, inner, seed, reverse=False):
        self.inner, self.system, self.seed, self.reverse = inner, inner.system, seed, reverse

    def read(self, name, params=None):
        rows = list(self.inner.read(name, params))
        random.Random(f"{self.seed}:{name}").shuffle(rows)
        return rows[::-1] if self.reverse else rows

    def bound_of_link(self, ids):
        return self.inner.bound_of_link(ids)


def equal_cost_callers():
    """Three resolved callers' sites of ``F`` (a flag-false function) in ids that are neither reader order nor insertion order: ``G`` is reached from each at the same
    cost (possible 1, distance 1, length 3, own-possible 0)."""
    g = G(unit="uf", flow=False)
    g.sym("F"), g.param("pF", "F"), g.sym("G")
    for site in ("c_m", "c_a", "c_z"):
        g.site(site, "G"), g.calls(site, "F")
    return g


@pytest.mark.parametrize("seed,reverse", [(s, r) for s in range(6) for r in (False, True)])
def test_equal_cost_arrivals_show_the_chain_of_the_smallest_predecessor_key_whatever_order_the_replies_arrive_in(seed, reverse):
    g = equal_cost_callers()
    reader = Shuffled(MemoryReader(g), seed, reverse)
    item = items_by_id(bfs.walk(reader, ("Param", "pF"), S.VALUE))["G"]
    assert [s["node"] for s in item.chain] == [("Param", "pF"), ("Symbol", "F"), ("CallSite", "c_a"), ("Symbol", "G")]
    assert (item.possible_hops, item.distance, item.length) == (1, 1, 3)


# --------------------------------------------------------------------------- (b) the candidate walks: row-table variants (-p1-ml.md §3.5)

from ccg.engine import flow  # noqa: E402


def added_ids(name):
    return {r.id for r in flow.candidate_table(name)} - {r.id for r in bfs.ROWS}


def reach_graph():
    """``X`` (flag-false, python) holds the changed param ``px``: it flows into a call of ``F2`` (flag-true) whose own param owns ``F2``, it guards the site ``cK`` of
    ``K``, and it is used at ``cD`` of ``D``. ``F`` (a flag-false javascript function) holds ``pF``; it is called by ``C`` and calls ``Q``."""
    g = G(unit="ux", flow=False, lang="python")
    g.unit_node("ut", flow=True, lang="rust"), g.unit_node("uj", flow=False, lang="javascript")
    g.sym("X"), g.param("px", "X", index=0, name="o"), g.cond("kX", "X")
    g.sym("F2", unit="ut"), g.param("pf", "F2", index=0, name="o")
    g.site("cx", "X"), g.calls("cx", "F2"), g.data("CallSite", "cx", "Param", "px", "p0", "a0")
    g.sym("D"), g.site("cD", "D"), g.data("CallSite", "cD", "Param", "px", "p0", "a1")
    g.sym("K"), g.site("cK", "K"), g.controlled("CallSite", "cK", "kX"), g.data("Cond", "kX", "Param", "px", "p0", "")
    g.sym("F", unit="uj"), g.param("pF", "F"), g.sym("C", unit="uj"), g.site("cC", "C", unit="uj"), g.calls("cC", "F")
    g.sym("Q", unit="uj"), g.site("cQ", "F", unit="uj"), g.calls("cQ", "Q")
    return g


def test_the_as_built_table_is_the_shipped_one_and_the_candidates_are_named_and_closed():
    assert flow.CANDIDATES == ("as-built", "V-flow", "V-both", "V-union")
    assert flow.candidate_table("as-built") == bfs.ROWS
    with pytest.raises(ValueError):
        flow.candidate_table("V-nothing")


def test_v_flow_follows_data_controlled_by_and_the_argument_rows_in_a_flag_false_unit_every_added_hop_possible_flow_unavailable():
    g = reach_graph()
    shipped = items_by_id(run(g, ("Param", "px"), S.path_state("p0")))
    assert set(shipped) == {"X"}                                                       # the shipped table follows none of them
    w = run(g, ("Param", "px"), S.path_state("p0"), table=flow.candidate_table("V-flow"))
    items = items_by_id(w)
    assert {"X", "D", "K", "F2"} <= set(items)
    for who in ("D", "K", "F2"):
        assert items[who].status == "possible" and items[who].reason == "flow-unavailable:python", who
        for s in items[who].chain[1:]:
            if s["edge"]["type"] in ("DATA", "CONTROLLED_BY") or (s["edge"]["type"], s["edge"]["dir"]) == ("CALLS", "->"):
                assert (s["edge"]["status"], s["edge"]["reason"]) == ("possible", "flow-unavailable:python"), (who, s["edge"])
    assert items["F2"].possible_hops == 2 and items["D"].possible_hops == 1
    assert items["X"].status == "confirmed"


def test_v_flow_adds_nothing_in_a_flag_true_unit():
    g = reach_graph()
    for n in g.rows.nodes["Unit"]:
        n["flowAvailable"] = True
    shipped = run(g, ("Param", "px"), S.path_state("p0"))
    variant = run(g, ("Param", "px"), S.path_state("p0"), table=flow.candidate_table("V-flow"))
    assert shipped.visited == variant.visited and {i.status for i in variant.items()} == {"confirmed"}


def callee_reach_graph():
    """``F`` (a flag-false javascript function) holds ``pF`` and calls ``Q`` (resolved) and ``R`` (by name); ``C`` calls ``F``."""
    g = G(unit="uf", flow=False, lang="javascript")
    g.sym("F"), g.param("pF", "F"), g.sym("C"), g.site("cC", "C"), g.calls("cC", "F")
    g.sym("Q"), g.site("cQ", "F", callee="q"), g.calls("cQ", "Q")
    g.sym("R"), g.site("cR", "F", callee="r"), g.calls_name("cR", "R", "r")
    return g


def test_v_both_lists_the_callees_of_a_function_reached_at_call_level_as_well_every_added_hop_possible():
    g = callee_reach_graph()
    assert {"Q", "R"}.isdisjoint(items_by_id(run(g, ("Param", "pF"))))
    items = items_by_id(run(g, ("Param", "pF"), table=flow.candidate_table("V-both")))
    assert {"F", "C", "Q", "R"} <= set(items)
    q, r = items["Q"], items["R"]
    assert (q.status, q.reason, q.distance, q.possible_hops) == ("possible", "flow-unavailable:javascript", 1, 2)       # HAS_CALL -> and CALLS ->: both added hops
    assert (r.status, r.reason, r.distance) == ("possible", "flow-unavailable:javascript", 1)                          # the first possible hop is the HAS_CALL ->
    assert [(s["edge"]["type"], s["edge"]["dir"], s["edge"]["status"], s["edge"].get("reason")) for s in r.chain[2:]] == [
        ("HAS_CALL", "->", "possible", "flow-unavailable:javascript"), ("CALLS_NAME", "->", "possible", "name-match")]
    assert (items["C"].status, items["C"].reason) == ("possible", "flow-unavailable:javascript")      # the callers stay as in the as-built walk


def test_v_both_adds_no_callee_in_a_flag_true_unit():
    g = callee_reach_graph()
    g.rows.nodes["Unit"][0]["flowAvailable"] = True
    items = items_by_id(run(g, ("Param", "pF"), table=flow.candidate_table("V-both")))
    assert set(items) == {"F"}


def test_v_union_has_both_sets_of_added_rows_and_lists_at_least_what_each_lists():
    assert added_ids("V-union") == added_ids("V-flow") | added_ids("V-both") and added_ids("V-flow") and added_ids("V-both")
    assert not added_ids("V-flow") & added_ids("V-both")
    g = reach_graph()
    listed = {n: {(a, i.id) for a, st in ((("Param", "px"), S.path_state("p0")), (("Param", "pF"), S.VALUE)) for i in run(g, a, st, table=flow.candidate_table(n)).items()}
              for n in ("as-built", "V-flow", "V-both", "V-union")}
    assert listed["V-union"] >= listed["V-flow"] | listed["V-both"]
    assert listed["V-flow"] > listed["as-built"] and listed["V-both"] > listed["as-built"] and listed["V-union"] > listed["V-flow"] and listed["V-union"] > listed["V-both"]


def test_a_candidate_walk_has_no_algorithm_version_only_the_shipped_table_has():
    g = callee_reach_graph()
    for name in ("V-flow", "V-both", "V-union"):
        with pytest.raises(bfs.WalkError, match="no algorithm version"):
            run(g, ("Param", "pF"), table=flow.candidate_table(name)).version()


def test_every_added_row_of_a_candidate_walk_is_labelled_and_none_is_a_request_field():
    for name in ("V-flow", "V-both", "V-union"):
        for row in flow.candidate_table(name):
            if row.id in added_ids(name):
                assert row.labelled, row.id


def test_the_candidate_table_is_an_engine_argument_never_a_request_field():
    schema = json.loads((Path(bfs.__file__).parent.parent / "schemas" / "answer-v1.schema.json").read_text())
    request = schema["$defs"]["Request"]["properties"]
    assert not {"table", "variant", "walk", "candidate"} & set(request), sorted(request)


# --------------------------------------------------------------------------- the gates and labels of the shipped table, pinned row by row

#: row id -> (gate, labelled), written from ``-answer-spec2.md`` §5 independently of ``bfs.ROWS``; a row not named has no gate and is not labelled
GATE_PINS = {
    "data_in:Param": "src:true", "data_in:CallSite": "src:true", "controlled_in": "src:true", "arg_has_param": "src:true",
    "data_out:CallSite": "src:true", "data_out:Ret": "src:true", "data_out:Cond": "src:true",
    "controlled_out:CallSite": "src:true", "controlled_out:Ret": "src:true", "controlled_out:Cond": "src:true", "controlled_out:Param": "src:true",
    "arg_calls_out": "both:true", "arg_name_out": "both:true",
    "owner:CallSite": "dst:true", "owner:Param": "dst:true", "owner:Cond": "dst:true", "calls_out": "dst:true", "name_out": "dst:true",
    "owner:CallSite:fn": "dst:false", "owner:Param:fn": "dst:false", "owner:Cond:fn": "dst:false", "calls_out:fn": "dst:false", "name_out:fn": "dst:false",
}
LABELLED_PINS = {"calls_in:fn", "name_in:fn", "to_in:fn", "has_out:CallSite:fn"}


def test_every_shipped_row_has_the_gate_and_label_the_flow_fallback_gives_it():
    assert {r.id: (r.gate, r.labelled) for r in bfs.ROWS} == {r.id: (GATE_PINS.get(r.id, ""), r.id in LABELLED_PINS) for r in bfs.ROWS}
    assert set(GATE_PINS) | LABELLED_PINS <= {r.id for r in bfs.ROWS}


def test_the_labelled_rows_are_exactly_the_exits_of_function_in_each_column_and_every_one_leaves_function():
    """``-answer-spec2.md`` §5: impact ``CALLS <-``, ``CALLS_NAME <-``, ``TO <-``; depends-on ``HAS_CALL ->``; none leaves any other state, none is followed from ``value``."""
    labelled = [r for r in bfs.ROWS if r.labelled]
    assert {(r.column, r.edge, r.dir) for r in labelled} == {("impact", "CALLS", "<-"), ("impact", "CALLS_NAME", "<-"), ("impact", "TO", "<-"), ("depends-on", "HAS_CALL", "->")}
    assert all(r.src == "Symbol" and r.kinds == ("function",) for r in labelled)
    assert {r.id for r in bfs.ROWS if "function" in r.kinds} == LABELLED_PINS


def test_the_rows_that_enter_function_are_the_closed_list_of_the_spec_and_only_in_a_flag_false_unit():
    enter = {r.id for r in bfs.ROWS if r.to == "function"}
    assert enter == {"owner:CallSite:fn", "owner:Param:fn", "owner:Cond:fn", "calls_out:fn", "name_out:fn"}       # HAS_CALL/HAS_PARAM/HAS_COND <- and CALLS/CALLS_NAME ->
    assert {r.gate for r in bfs.ROWS if r.to == "function"} == {"dst:false"}
    assert all(r.id.startswith("owner:") for r in bfs.ROWS if r.to == "function" and r.column == "impact")          # not contract mode: owner:CallSite:contract ends in ``end``
    assert next(r for r in bfs.ROWS if r.id == "owner:CallSite:contract").to == "end"


# --------------------------------------------------------------------------- (e) a missing candidate_count row is "N unknown"

def via_graph():
    """``F`` owns three call sites: ``c1``/``c2`` carry ``VIA`` edges of key ``k`` to two links, ``c3`` a ``VIA`` edge with no ``consumerKey``; ``orphan`` is no symbol's."""
    g = G()
    g.sym("F"), g.site("c1", "F"), g.site("c2", "F"), g.site("c3", "F"), g.link("L1"), g.link("L2"), g.link("L3")
    g.via("c1", "L1", key="k"), g.via("c2", "L2", key="k")
    g.edge("CallSite", "VIA", "Link", "c3", "L3", status="confirmed", reason="key-match")
    g.node("CallSite", callSiteId="orphan", repoId="repo-a", unitId="ua", path="a.py", line=1, calleeName="f"), g.via("orphan", "L1", key="k")
    return g


def test_candidate_count_is_n_for_a_key_and_none_for_an_absent_key_or_an_ownerless_site_never_one_or_zero():
    got = bfs.candidate_counts(MemoryReader(via_graph()), [("c1", "L1"), ("c2", "L2"), ("c3", "L3"), ("orphan", "L1"), ("c1", "L3")])
    assert got == {("c1", "L1"): 2, ("c2", "L2"): 2, ("c3", "L3"): None, ("orphan", "L1"): None, ("c1", "L3"): None}
    assert got[("c3", "L3")] is None and got[("orphan", "L1")] is None


def test_candidate_counts_asks_once_and_not_at_all_for_no_pairs():
    r = MemoryReader(via_graph())
    assert bfs.candidate_counts(r, []) == {} and r.calls == []
    bfs.candidate_counts(r, [("c1", "L1"), ("c1", "L1"), ("c2", "L2")])
    assert r.calls == [("candidate_count", {"pairs": [["c1", "L1"], ["c2", "L2"]]})]


@pytest.mark.parametrize("row", [("c1", "L1", 0), ("c1", "L1", True), ("c1", "L1", "2"), ("c1", "L1"), ("zz", "L1", 1), (None, "L1", 1)])
def test_a_malformed_candidate_count_row_is_a_graph_data_error(row):
    from engine_helpers import ScriptedReader
    with pytest.raises(bfs.GraphDataError):
        bfs.candidate_counts(ScriptedReader({"candidate_count": [row]}), [("c1", "L1")])


# --------------------------------------------------------------------------- the flag and language are read strictly; a closed row is not even asked

@pytest.mark.parametrize("flag", ["true", "false", 1, "", None])
def test_only_the_boolean_true_opens_a_gate_any_other_flag_value_reads_false(flag):
    g = caller_chain(flow=True)
    next(n for n in g.rows.nodes["Unit"] if n["unitId"] == "uf")["flowAvailable"] = flag
    w = walk_param(g)
    assert w.units["uf"].flag is False and items_by_id(w)["G"].reason == "flow-unavailable:javascript"


def test_a_labelled_hop_whose_unit_has_no_language_is_a_graph_data_error_not_a_guessed_label():
    g = caller_chain(flow=False)
    del next(n for n in g.rows.nodes["Unit"] if n["unitId"] == "uf")["language"]
    with pytest.raises(bfs.GraphDataError, match="no language"):
        walk_param(g)


def test_a_row_the_flag_closes_is_not_asked_of_the_reader_at_all():
    r = MemoryReader(point_rows_graph(False))
    bfs.walk(r, ("Param", "pF"), S.path_state("p0"))
    assert not {"data_in:Param", "data_in:CallSite", "controlled_in", "calls_out", "name_out"} & set(r.names())
    r = MemoryReader(point_rows_graph(True))
    bfs.walk(r, ("Param", "pF"), S.path_state("p0"))
    assert {"data_in:Param", "data_in:CallSite"} <= set(r.names())


def test_each_unit_is_read_once_per_walk():
    r = MemoryReader(caller_chain(flow=False))
    walk = bfs.walk(r, ("Param", "pF"), S.VALUE)
    asked = [i for name, p in r.calls if name == "units" for i in p["ids"]]
    assert asked and len(asked) == len(set(asked)) == len(walk.units)


def test_a_v_flow_hop_is_labelled_with_the_language_of_the_flag_false_unit_the_row_leaves_or_enters():
    """``src:false`` rows say the source's language; the argument twin (``any:false``) says the source's when it is flag-false, else the callee's."""
    def graph(src_flow, callee_flow):
        g = G(unit="ux", flow=src_flow, lang="python")
        g.unit_node("uc", flow=callee_flow, lang="rust")
        g.sym("X"), g.param("px", "X", index=0, name="o"), g.sym("F2", unit="uc"), g.param("pf", "F2", index=0, name="o")
        g.site("cx", "X"), g.calls("cx", "F2"), g.data("CallSite", "cx", "Param", "px", "p0", "a0")
        return g

    def call_hop(src_flow, callee_flow):
        w = run(graph(src_flow, callee_flow), ("Param", "px"), S.path_state("p0"), table=flow.candidate_table("V-flow"))
        item = items_by_id(w)["F2"]
        return next(s["edge"] for s in item.chain if s["edge"] and (s["edge"]["type"], s["edge"]["dir"]) == ("CALLS", "->"))

    assert call_hop(False, False) == {"type": "CALLS", "dir": "->", "status": "possible", "reason": "flow-unavailable:python"}
    assert call_hop(False, True)["reason"] == "flow-unavailable:python"
    assert call_hop(True, False)["reason"] == "flow-unavailable:rust"
    assert call_hop(True, True)["status"] == "confirmed"       # both flag-true: the shipped row followed it, no twin adds a labelled one


# --------------------------------------------------------------------------- the candidate rows, pinned and walked row by row

#: the twins V-flow adds: shipped row id -> the gate of its twin (the twin is the shipped row, same edge/direction/to/increment, followed where the shipped one is not, labelled)
V_FLOW_TWINS = {
    "data_in:Param": "src:false", "data_in:CallSite": "src:false", "controlled_in": "src:false", "arg_has_param": "src:false", "arg_calls_out": "any:false", "arg_name_out": "any:false",
    "data_out:CallSite": "src:false", "data_out:Ret": "src:false", "data_out:Cond": "src:false",
    "controlled_out:CallSite": "src:false", "controlled_out:Ret": "src:false", "controlled_out:Cond": "src:false", "controlled_out:Param": "src:false",
}
#: the rows V-both adds, written out: (id, src, kinds, edge, dir, to, increment); every one labelled, none gated
V_BOTH_ROWS = [("has_out:CallSite:fn:both", "Symbol", ("function",), "HAS_CALL", "->", "end", 0, "", ""),
               ("has_out:CallSite:callee:both", "Symbol", ("end",), "HAS_CALL", "->", "end", 0, "callee-state", ""),
               ("calls_out:both", "CallSite", ("end",), "CALLS", "->", "end:callee", 1, "", "symbol"),
               ("name_out:both", "CallSite", ("end",), "CALLS_NAME", "->", "end:callee", 1, "", "symbol")]


def test_v_flow_adds_exactly_the_twin_of_every_gated_precise_row():
    shipped = {r.id: r for r in bfs.ROWS}
    twins = {r.id: r for r in flow.candidate_table("V-flow") if r.id not in shipped}
    assert set(twins) == {f"{i}:flow" for i in V_FLOW_TWINS}
    for rid, gate in V_FLOW_TWINS.items():
        assert twins[f"{rid}:flow"] == dataclasses.replace(shipped[rid], id=f"{rid}:flow", gate=gate, labelled=True), rid


def test_v_both_adds_exactly_the_call_site_and_callee_rows_every_one_labelled_and_ungated():
    got = [(r.id, r.src, r.kinds, r.edge, r.dir, r.to, r.inc, r.where, r.item) for r in flow.candidate_table("V-both") if r.id in added_ids("V-both")]
    assert got == V_BOTH_ROWS
    assert all(r.column == "impact" and r.gate == "" and r.labelled for r in flow.candidate_table("V-both") if r.id in added_ids("V-both"))
    assert list(flow.candidate_table("V-both")[:len(bfs.ROWS)]) == list(bfs.ROWS)


def test_the_shipped_item_rules_carry_no_v_both_arrival_and_the_shipped_rows_carry_no_item_of_their_own():
    assert ("Symbol", "CALLS", "->", "end") not in bfs._ITEM_RULES[bfs.IMPACT] and ("Symbol", "CALLS_NAME", "->", "end") not in bfs._ITEM_RULES[bfs.IMPACT]
    assert {r.item for r in bfs.ROWS} == {""}


# --------------------------------------------------------------------------- V-both walks callees to the bound (-p1-ml.md §3.5)

def callee_chain(*, depth=3, q2_flow=True):
    """``F`` (flag-false) holds ``pF`` and calls ``Q1``, which calls ``Q2``, which calls ``Q3`` ...; ``Q2`` lives in a flag-true unit (no flag gate on the callee chain)."""
    g = G(unit="uf", flow=False, lang="python")
    g.unit_node("uq", flow=q2_flow, lang="python")
    g.sym("F"), g.param("pF", "F")
    prev = "F"
    for i in range(1, depth + 1):
        u = "uq" if i == 2 else None
        g.sym(f"Q{i}", unit=u), g.site(f"c{i}", prev, unit=("uq" if prev == "Q2" else None)), g.calls(f"c{i}", f"Q{i}")
        prev = f"Q{i}"
    return g


def vboth(g, anchor=("Param", "pF"), state=S.VALUE, **kw):
    return run(g, anchor, state, table=flow.candidate_table("V-both"), **kw)


def test_v_both_lists_the_callees_to_the_depth_bound_and_no_further():
    items = items_by_id(vboth(callee_chain()))
    assert set(items) == {"F", "Q1", "Q2"}
    assert [(items[q].distance, items[q].status, items[q].reason) for q in ("Q1", "Q2")] == [(1, "possible", "flow-unavailable:python"), (2, "possible", "flow-unavailable:python")]
    assert items["Q2"].possible_hops == 4                                         # HAS_CALL -> and CALLS -> for each of the two levels
    assert [s["state"] for s in items["Q2"].chain][-1] == ("end", "callee")


def test_a_callee_chain_is_bounded_by_the_distance_of_the_function_that_opened_it():
    """``G`` calls ``F`` (function at distance 1); ``G`` is a function at distance 1 too, so its callees' ``Q1`` is at 2 and ``Q2`` past the bound."""
    g = G(unit="uf", flow=False, lang="python")
    g.sym("F"), g.param("pF", "F"), g.sym("G"), g.site("cG", "G"), g.calls("cG", "F"), g.sym("Q1"), g.site("g1", "G"), g.calls("g1", "Q1"), g.sym("Q2"), g.site("q1", "Q1"), g.calls("q1", "Q2")
    items = items_by_id(vboth(g))
    assert set(items) == {"F", "G", "Q1"} and items["G"].distance == 1 and items["Q1"].distance == 2


def test_a_callees_own_callers_are_never_walked_and_a_flag_true_function_expands_no_callees():
    g = callee_chain(depth=1)
    g.sym("Z"), g.site("cZ", "Z"), g.calls("cZ", "Q1")                                # a caller of the callee
    assert set(items_by_id(vboth(g))) == {"F", "Q1"}
    h = G(unit="uf", flow=False)
    h.unit_node("ut", flow=True)
    h.sym("F"), h.param("pF", "F"), h.sym("T", unit="ut"), h.site("cT", "T", unit="ut"), h.calls("cT", "F"), h.sym("Q", unit="ut"), h.site("tq", "T", unit="ut"), h.calls("tq", "Q")
    hi = items_by_id(vboth(h))
    assert "T" in hi and "Q" not in hi                                                 # T is a flag-true caller: ``end``, not ``function``


def test_a_callee_cycle_back_to_the_function_creates_no_duplicate_item():
    g = callee_chain(depth=1)
    g.site("back", "Q1"), g.calls("back", "F")
    w = vboth(g)
    ids = [i.id for i in w.items()]
    assert sorted(ids) == ["F", "Q1"] and len(ids) == len(set(ids))


def test_v_union_is_the_union_of_the_rows_and_lists_what_neither_list_has():
    """``D`` is reached by V-flow's ``DATA`` row, then leaves ``function`` by V-both's callee rows: ``Q`` is in V-union only."""
    g = G(unit="uf", flow=False, lang="python")
    g.sym("X"), g.param("px", "X"), g.sym("D"), g.site("cD", "D"), g.data("CallSite", "cD", "Param", "px", "p0", "a0"), g.sym("Q"), g.site("dq", "D"), g.calls("dq", "Q")
    listed = {n: {i.id for i in run(g, ("Param", "px"), S.path_state("p0"), table=flow.candidate_table(n)).items()} for n in ("as-built", "V-flow", "V-both", "V-union")}
    assert listed == {"as-built": {"X"}, "V-flow": {"D", "X"}, "V-both": {"X"}, "V-union": {"D", "Q", "X"}}
    assert [r.id for r in flow.candidate_table("V-union")] == [r.id for r in flow.candidate_table("V-flow")] + sorted(added_ids("V-both"), key=[r.id for r in flow.candidate_table("V-both")].index)


def flagfalse():
    return G(unit="uf", flow=False, lang="python")


@pytest.mark.parametrize("which", ["data_in:Param", "data_in:CallSite", "controlled_in"])
def test_each_v_flow_use_row_reaches_its_item_possible_flow_unavailable_at_distance_zero(which):
    g = flagfalse()
    g.sym("F"), g.sym("D")
    if which == "data_in:Param":
        g.param("pF", "F"), g.site("cD", "D"), g.data("CallSite", "cD", "Param", "pF", "p0", "a0")
        anchor, state, hop = ("Param", "pF"), S.path_state("p0"), ("CallSite", "path", "a0")
    elif which == "data_in:CallSite":
        g.site("cF", "F"), g.ret("rD", "D"), g.data("Ret", "rD", "CallSite", "cF", "r", "ret")
        anchor, state, hop = ("CallSite", "cF"), S.path_state("r"), ("Ret", "path", "ret")
    else:
        g.cond("kF", "F"), g.site("cD", "D"), g.controlled("CallSite", "cD", "kF")
        anchor, state, hop = ("Cond", "kF"), S.VALUE, ("CallSite", "value", "")
    assert "D" not in items_by_id(run(g, anchor, state))
    w = run(g, anchor, state, table=flow.candidate_table("V-flow"))
    d = items_by_id(w)["D"]
    assert (d.status, d.reason, d.distance, d.possible_hops) == ("possible", "flow-unavailable:python", 0, 1)
    assert (d.chain[1]["node"][0], d.chain[1]["state"][0], d.chain[1]["state"][1]) == hop
    assert d.chain[1]["edge"]["status"] == "possible"


def test_the_v_flow_argument_rows_enter_a_callee_whatever_its_unit_and_carry_the_parameter_on():
    def graph(by_name):
        g = G(unit="ux", flow=False, lang="python")
        g.unit_node("uc", flow=False, lang="rust")
        g.sym("X"), g.param("px", "X", index=0, name="o"), g.sym("F2", unit="uc"), g.param("pf", "F2", index=0, name="o")
        g.site("cx", "X", callee="f")
        g.calls_name("cx", "F2", "f") if by_name else g.calls("cx", "F2")
        g.data("CallSite", "cx", "Param", "px", "p0", "a0")
        return g

    for by_name in (False, True):
        f2 = items_by_id(run(graph(by_name), ("Param", "px"), S.path_state("p0"), table=flow.candidate_table("V-flow")))["F2"]
        assert (f2.status, f2.reason, f2.distance, f2.possible_hops) == ("possible", "flow-unavailable:python", 1, 3)
        call = ("CALLS_NAME", "->", "possible", "name-match") if by_name else ("CALLS", "->", "possible", "flow-unavailable:python")
        assert [(s["node"][0], s["state"], s["edge"]["type"], s["edge"]["dir"], s["edge"]["status"], s["edge"].get("reason")) for s in f2.chain[1:]] == [
            ("CallSite", ("path", "a0"), "DATA", "<-", "possible", "flow-unavailable:python"),
            ("Symbol", ("path", "a0"), *call),
            ("Param", ("path", "p0"), "HAS_PARAM", "->", "possible", "flow-unavailable:rust"),
            ("Symbol", ("function", ""), "HAS_PARAM", "<-", "confirmed", None)]


def dep_graph(first):
    """``F`` (flag-false) holds a ``first`` point whose ``DATA`` edge reaches the result of ``cG``, a call of ``G`` that belongs to ``Y`` (which the walk never enters)."""
    g = flagfalse()
    g.sym("F"), g.sym("Y"), g.sym("G"), g.site("cG", "Y"), g.calls("cG", "G")
    if first == "CallSite":
        g.site("c0", "F"), g.data("CallSite", "c0", "CallSite", "cG", "a0", "r")
    elif first == "Ret":
        g.ret("r0", "F"), g.data("Ret", "r0", "CallSite", "cG", "ret", "r")
    else:
        g.cond("k0", "F"), g.data("Cond", "k0", "CallSite", "cG", "", "r")
    return g


@pytest.mark.parametrize("first", ["CallSite", "Ret", "Cond"])
def test_each_v_flow_data_out_row_reaches_the_callee_behind_the_data_edge(first):
    g = dep_graph(first)
    assert "G" not in items_by_id(run(g, ("Symbol", "F"), column=bfs.DEPENDS_ON))
    got = items_by_id(run(g, ("Symbol", "F"), column=bfs.DEPENDS_ON, table=flow.candidate_table("V-flow")))["G"]
    assert (got.status, got.reason, got.distance, got.possible_hops) == ("possible", "flow-unavailable:python", 1, 1)
    assert [(s["edge"]["type"], s["edge"]["dir"], s["edge"]["status"]) for s in got.chain[-2:]] == [("DATA", "->", "possible"), ("CALLS", "->", "confirmed")]


@pytest.mark.parametrize("first", ["CallSite", "Ret", "Cond", "Param"])
def test_each_v_flow_controlled_by_out_row_reaches_the_guard_and_the_callee_behind_it(first):
    """``first`` is guarded by the cond ``k``, whose ``DATA`` edge reaches ``cG`` (the cond's own ``data_out`` twin is a second labelled hop)."""
    g = flagfalse()
    g.sym("F"), g.sym("Y"), g.sym("G"), g.site("cG", "Y"), g.calls("cG", "G"), g.cond("k", "Y"), g.data("Cond", "k", "CallSite", "cG", "", "r")
    if first == "CallSite":
        g.site("c0", "F"), g.controlled("CallSite", "c0", "k")
    elif first == "Ret":
        g.ret("r0", "F"), g.controlled("Ret", "r0", "k")
    elif first == "Cond":
        g.cond("k0", "F"), g.controlled("Cond", "k0", "k")
    else:
        g.param("p0", "F"), g.controlled("Param", "p0", "k")
    assert "G" not in items_by_id(run(g, ("Symbol", "F"), column=bfs.DEPENDS_ON))
    got = items_by_id(run(g, ("Symbol", "F"), column=bfs.DEPENDS_ON, table=flow.candidate_table("V-flow")))["G"]
    assert (got.status, got.reason, got.distance, got.possible_hops) == ("possible", "flow-unavailable:python", 1, 2)
    assert [(s["edge"]["type"], s["edge"]["dir"]) for s in got.chain[-3:]] == [("CONTROLLED_BY", "->"), ("DATA", "->"), ("CALLS", "->")]


@pytest.mark.parametrize("anchor_label,build", [("Param", lambda g: g.param("a", "F")), ("Cond", lambda g: g.cond("a", "F")), ("CallSite", lambda g: g.site("a", "F"))])
def test_every_point_kind_of_a_flag_false_function_enters_function_at_distance_zero_and_its_callers_are_possible(anchor_label, build):
    g = G(unit="uf", flow=False, lang="go")
    g.sym("F"), g.sym("C"), g.site("cC", "C"), g.calls("cC", "F")
    build(g)
    items = items_by_id(run(g, (anchor_label, "a"), S.VALUE))
    assert (items["F"].status, items["F"].distance, items["F"].chain[-1]["state"]) == ("confirmed", 0, ("function", ""))
    assert (items["C"].status, items["C"].reason, items["C"].distance) == ("possible", "flow-unavailable:go", 1)


def test_the_labelled_exits_keep_their_distance_increments_and_the_call_sites_they_reach_are_value_state_in_a_flag_false_unit():
    g, name_site, resolved_site = two_site_caller(name_site_first=True)
    items = items_by_id(walk_param(g))
    assert items["C"].distance == 1 and ("CallSite", resolved_site, "value", "") in {k for k in run(g, ("Param", "pF"), S.VALUE).visited}
    n = G(unit="uf", flow=False)
    n.sym("F"), n.param("pF", "F"), n.sym("N"), n.site("cN", "N", callee="f"), n.calls_name("cN", "F", "f")
    nm = items_by_id(walk_param(n))["N"]
    assert (nm.distance, states(nm)) == (1, ["value", "function", "value", "function"])
    link = G(unit="uf", flow=False)
    link.sym("P"), link.param("pP", "P"), link.cp("cp1"), link.sym("U"), link.site("cU", "U"), link.link("L", provider="P", cp="cp1"), link.via("cU", "L")
    assert items_by_id(run(link, ("Param", "pP"), S.VALUE))["U"].distance == 1                       # TO <- adds 0, VIA <- adds 1


def test_a_name_matched_callee_of_a_flag_false_unit_enters_function_one_call_away():
    g = G(unit="uf", flow=False, lang="go")
    g.sym("F"), g.sym("K"), g.site("cF", "F", callee="k"), g.calls_name("cF", "K", "k")
    k = items_by_id(run(g, ("Symbol", "F"), column=bfs.DEPENDS_ON))["K"]
    assert (k.status, k.reason, k.distance, states(k)) == ("possible", "name-match", 1, ["value", "value", "function"])


def test_a_name_matched_caller_in_a_flag_true_unit_is_reached_in_path_state_and_its_own_reason_is_kept():
    g = G(unit="uf", flow=False, lang="javascript")
    g.unit_node("uc", flow=True, lang="python")
    g.sym("F"), g.param("pF", "F"), g.sym("N", unit="uc"), g.site("cN", "N", unit="uc", callee="f"), g.calls_name("cN", "F", "f")
    n = items_by_id(walk_param(g))["N"]
    assert (n.status, n.reason, states(n)) == ("possible", "name-match", ["value", "function", "path", "end"])


# --------------------------------------------------------------------------- paired rows share one read

@pytest.mark.parametrize("make,anchor,state,column", [
    (lambda: caller_chain(flow=False), ("Param", "pF"), S.VALUE, bfs.IMPACT), (lambda: callee_graph(False), ("Symbol", "F"), S.VALUE, bfs.DEPENDS_ON),
    (lambda: callee_graph(True), ("Symbol", "F"), S.VALUE, bfs.DEPENDS_ON), (callee_reach_graph, ("Param", "pF"), S.VALUE, bfs.IMPACT)])
def test_a_row_pair_that_shares_a_read_issues_it_once_per_level_and_the_walk_is_unchanged(make, anchor, state, column):
    r = MemoryReader(make())
    w = bfs.walk(r, anchor, state, column=column)
    keys = [(n, json.dumps(p, sort_keys=True)) for n, p in r.calls]
    assert len(keys) == len(set(keys)), [k for k in set(keys) if keys.count(k) > 1]
    assert w.items()


# --------------------------------------------------------------------------- randomized flag-mixed worlds: the V18 (a)-(e) conditions over real walk output

def mixed_world(seed):
    """``test_engine_bfs_bound.random_world`` with its symbols spread over three units of random flags and languages (a call site lives in its owner's unit)."""
    from test_engine_bfs_bound import random_world
    g, anchors = random_world(seed)
    rnd = random.Random(f"flags:{seed}")
    units = {u: (rnd.random() < 0.4, lang) for u, lang in (("u1", "python"), ("u2", "javascript"), ("u3", "go"))}
    g.rows.nodes["Unit"] = []
    for u, (flag, lang) in units.items():
        g.unit_node(u, flow=flag, lang=lang)
    of = {}
    for s in g.rows.nodes["Symbol"]:
        s["unitId"] = of[s["symbolId"]] = rnd.choice(sorted(units))
    owner = {}
    for (src, et, dst), edges in g.rows.edges.items():
        if src == "Symbol" and et in ("HAS_CALL", "HAS_PARAM", "HAS_RETURN", "HAS_COND"):
            owner.update({e["t"]: e["s"] for e in edges})
    for c in g.rows.nodes["CallSite"]:
        c["unitId"] = of[owner[c["callSiteId"]]]
    for key in [k for k in g.rows.edges if k[1] in ("DATA", "CONTROLLED_BY")]:           # the assembler writes both within one function only
        g.rows.edges[key] = [e for e in g.rows.edges[key] if owner[e["s"]] == owner[e["t"]]]
    return g, anchors, units


def check_v18(w, tables_shipped):
    """The conditions of ``-answer-spec2.md`` §7.2 V18 (a)-(e) on every item's chain, from ``answer.FUNCTION_ENTRY``/``FLOW_ROWS``, and what §5 says of flag-false units."""
    from ccg import answer
    col = w.column
    unit_of = lambda hop: w.facts.get(hop["node"], {}).get("unit")           # noqa: E731
    for item in w.items():
        assert item.possible_hops == sum(s["edge"]["status"] == "possible" for s in item.chain[1:]) and (item.status == "possible") == bool(item.possible_hops)
        for i in range(1, len(item.chain)):
            hop, prev = item.chain[i], item.chain[i - 1]
            edge, kind = hop["edge"], hop["state"][0]
            row = (edge["type"], edge["dir"])
            if kind == "function":
                assert hop["node"][0] == "Symbol" and row in answer.FUNCTION_ENTRY[col]                                   # (a)
                assert w.units[unit_of(hop)].flag is False                                                                 # only in a flag-false unit
            if prev["state"][0] == "function" and row in answer.FLOW_ROWS[col]:
                assert edge["status"] == "possible"                                                                        # (b)
            if not tables_shipped:
                continue
            reason = edge.get("reason") or ""
            if reason.startswith("flow-unavailable:"):
                assert prev["state"][0] == "function" and row in answer.FLOW_ROWS[col] and row[0] != "CALLS_NAME"          # (c)
                assert reason == "flow-unavailable:" + w.units[unit_of(prev)].lang                                         # (d)
            if prev["state"][0] == "function":
                assert row in answer.FLOW_ROWS[col]                                                                        # (e)
            if i >= 1 and kind == "path":
                assert w.units[unit_of(hop)].flag is True                                                                  # no path hop in a flag-false unit


def test_the_v18_conditions_hold_on_the_walk_output_of_random_flag_mixed_worlds_for_the_shipped_table_and_the_three_candidates():
    walks = functions = exits = 0
    for seed in range(24):
        g, anchors, _ = mixed_world(seed)
        for anchor, state, column in anchors:
            for name in flow.CANDIDATES:
                w = bfs.walk(MemoryReader(g), anchor, state, column=column, bound=3, table=flow.candidate_table(name))
                check_v18(w, name == "as-built")
                walks += 1
                functions += sum(k[2] == "function" for k in w.visited)
                exits += sum(1 for i in w.items() for s in i.chain[1:] if (s["edge"].get("reason") or "").startswith("flow-unavailable:"))
    assert walks > 700 and functions > 100 and exits > 100, (walks, functions, exits)     # the property is not vacuous
