"""P1-22 the row table pinned row by row, and the depends-on column walked past every hop kind.

The version fingerprint protects the table, not what the table does; this file states what each row does (increment, to-state, edge, direction) from
``docs/plans/connected-code-graph-graph.md`` §6.2 (distance: inter-function hops count 1, steps inside a function 0), ``-answer-spec2.md`` §3.3, §3.5, §5 and
``-p1-answers.md`` P1-22, independently of ``bfs.ROWS``, and walks a distance-2 item behind each depends-on hop kind so that a changed increment or to-state moves an item.
"""
from __future__ import annotations

from engine_walk_helpers import G, MemoryReader

from ccg.engine import bfs
from ccg.engine import state as S
from test_engine_bfs import edge_types, items_by_id, nodes, run
from test_engine_bfs_rows import contract_graph

#: row id -> (distance increment, to-state token, edge type, direction)
PINS = {
    "calls_in": (1, "path:r", "CALLS", "<-"), "name_in": (1, "path:r", "CALLS_NAME", "<-"), "to_in:value": (0, "value", "TO", "<-"), "to_in:contract": (0, "contract", "TO", "<-"),
    "provides_out": (0, "contract", "PROVIDES", "->"), "arg_calls_out": (1, "path:arg", "CALLS", "->"), "arg_name_out": (1, "path:arg", "CALLS_NAME", "->"),
    "arg_has_param": (0, "path:param", "HAS_PARAM", "->"),
    "owner:CallSite": (0, "end", "HAS_CALL", "<-"), "owner:Param": (0, "end", "HAS_PARAM", "<-"), "owner:Ret": (0, "value", "HAS_RETURN", "<-"), "owner:Cond": (0, "end", "HAS_COND", "<-"),
    "owner:CallSite:contract": (0, "end", "HAS_CALL", "<-"),
    "data_in:Param": (0, "path:to", "DATA", "<-"), "data_in:CallSite": (0, "path:to", "DATA", "<-"), "controlled_in": (0, "value", "CONTROLLED_BY", "<-"),
    "via_in:value": (1, "path:r", "VIA", "<-"), "via_in:contract": (1, "contract", "VIA", "<-"), "provides_in": (0, "contract", "PROVIDES", "<-"), "on_in": (0, "contract", "ON", "<-"),
    "consumes_in": (1, "contract", "CONSUMES", "<-"), "bound_in": (1, "site", "BOUND_BY", "<-"), "via_in:site": (1, "path:r", "VIA", "<-"), "wiring": (1, "end", "BOUND_BY", "->"),
    "has_out:CallSite": (0, "value", "HAS_CALL", "->"), "has_out:Param": (0, "value", "HAS_PARAM", "->"), "has_out:Ret": (0, "value", "HAS_RETURN", "->"),
    "has_out:Cond": (0, "value", "HAS_COND", "->"), "calls_out": (1, "value", "CALLS", "->"), "name_out": (1, "value", "CALLS_NAME", "->"), "via_out": (1, "value", "VIA", "->"),
    "to_out": (0, "value", "TO", "->"), "to_unit_out": (0, "end", "TO_UNIT", "->"),
    "data_out:CallSite": (0, "value", "DATA", "->"), "data_out:Ret": (0, "value", "DATA", "->"), "data_out:Cond": (0, "value", "DATA", "->"),
    "controlled_out:CallSite": (0, "value", "CONTROLLED_BY", "->"), "controlled_out:Ret": (0, "value", "CONTROLLED_BY", "->"), "controlled_out:Cond": (0, "value", "CONTROLLED_BY", "->"),
    "controlled_out:Param": (0, "value", "CONTROLLED_BY", "->"),
}
#: adapter -> where the arriving hop's status and reason come from (the behaviour of each is pinned in ``test_engine_bfs_rows.py``)
STATUS_SOURCE = {
    "calls_in": "the CALLS edge", "calls_out": "the CALLS edge", "name_in": "possible; reason from count(x)", "name_out": "possible; reason from count(x)",
    "data_in": "the DATA edge", "data_out": "the DATA edge", "controlled_in": "confirmed", "controlled_out": "confirmed", "link_in": "the Link", "via_in": "the VIA edge",
    "consumes_in": "the CONSUMES edge", "provides_in": "the PROVIDES edge", "provides_out": "the PROVIDES edge", "bound_in": "the Link", "owner_of": "confirmed",
    "has_out": "confirmed", "via_out": "the VIA edge", "to_out": "the Link", "to_unit_out": "possible/target-not-mapped (literal)", "bound_of_link": "the Link",
}


def test_every_row_does_what_the_pin_table_says():
    assert {r.id: (r.inc, r.to, r.edge, r.dir) for r in bfs.ROWS} == PINS


def test_every_adapter_has_a_stated_status_source():
    assert set(bfs.ADAPTERS) == set(STATUS_SOURCE)


def test_the_distance_rule_is_inter_function_hops_only():
    one = {i for i, p in PINS.items() if p[0] == 1}
    assert {PINS[i][2] for i in one} == {"CALLS", "CALLS_NAME", "VIA", "CONSUMES", "BOUND_BY"} and all(PINS[i][2] in ("CALLS", "CALLS_NAME", "VIA", "CONSUMES", "BOUND_BY") or PINS[i][0] == 0 for i in PINS)
    assert all(p[0] == 0 for p in PINS.values() if p[2] not in ("CALLS", "CALLS_NAME", "VIA", "CONSUMES", "BOUND_BY"))
    assert PINS["wiring"][0] == PINS["bound_in"][0] == 1                                 # a config hop is inter-function (graph note §6.2's distance paragraph)


# --------------------------------------------------------------------------- the depends-on column past every hop kind

def depends_pins():
    """F's depends-on reach: every branch ends in a callee ``K`` at distance 1 whose own call reaches ``J`` at distance 2. Definition sites belong to ``Y``, which the
    walk never enters, so a branch reaches its ``K`` only through the hop kind it names."""
    g = G()
    for s in ("F", "Y", "Z"):
        g.sym(s)
    g.unit_node("un", "repo-b", flow=False)

    def tail(k, j):
        g.sym(k), g.sym(j), g.site(f"kk_{k}", k), g.calls(f"kk_{k}", j)

    def to_callee(site, k, j, owner="Y"):
        g.site(site, owner), g.calls(site, k), tail(k, j)

    # has_out:CallSite, calls_out
    g.site("c1", "F"), g.calls("c1", "K1"), tail("K1", "J1")
    # name_out (a possible hop)
    g.site("c2", "F", callee="n"), g.calls_name("c2", "K2", "n"), tail("K2", "J2")
    # via_out, to_out
    g.site("c3", "F"), g.link("L3", provider="K3"), g.via("c3", "L3"), tail("K3", "J3")
    # via_out, to_unit_out
    g.site("c4", "F"), g.link("L4", to_unit="un", status="possible", reason="target-not-mapped"), g.via("c4", "L4", status="possible", reason="target-not-mapped")
    # has_out:Ret, data_out:Ret
    g.ret("r5", "F"), to_callee("d5", "K5", "J5"), g.data("Ret", "r5", "CallSite", "d5", "r", "ret")
    # has_out:Cond, data_out:Cond
    g.cond("k6", "F"), to_callee("d6", "K6", "J6"), g.data("Cond", "k6", "CallSite", "d6", "r", "")
    # has_out:Param, controlled_out:Param, data_out:Cond
    g.param("p7", "F", index=0), g.cond("k7", "Y"), to_callee("d7", "K7", "J7"), g.controlled("Param", "p7", "k7"), g.data("Cond", "k7", "CallSite", "d7", "r", "")
    # data_out:CallSite
    g.site("c8", "F"), to_callee("d8", "K8", "J8"), g.data("CallSite", "c8", "CallSite", "d8", "r", "a0")
    # controlled_out:CallSite
    g.site("c9", "F"), g.cond("k9", "Y"), to_callee("d9", "K9", "J9"), g.controlled("CallSite", "c9", "k9"), g.data("Cond", "k9", "CallSite", "d9", "r", "")
    # controlled_out:Ret
    g.ret("r10", "F"), g.cond("k10", "Y"), to_callee("d10", "K10", "J10"), g.controlled("Ret", "r10", "k10"), g.data("Cond", "k10", "CallSite", "d10", "r", "")
    # controlled_out:Cond (a condition controlled by another)
    g.cond("k11", "F"), g.cond("k12", "Y"), to_callee("d12", "K12", "J12"), g.controlled("Cond", "k11", "k12"), g.data("Cond", "k12", "CallSite", "d12", "r", "")
    return g


EXPECT = {
    ("symbol", "K1"): (0, 1), ("symbol", "J1"): (0, 2),
    ("symbol", "K2"): (1, 1), ("symbol", "J2"): (1, 2),
    ("symbol", "K3"): (0, 1), ("symbol", "J3"): (0, 2),
    ("unit", "un"): (2, 1),     # the VIA hop and the TO_UNIT hop are both possible
    ("symbol", "K5"): (0, 1), ("symbol", "J5"): (0, 2),
    ("symbol", "K6"): (0, 1), ("symbol", "J6"): (0, 2),
    ("symbol", "K7"): (0, 1), ("symbol", "J7"): (0, 2),
    ("symbol", "K8"): (0, 1), ("symbol", "J8"): (0, 2),
    ("symbol", "K9"): (0, 1), ("symbol", "J9"): (0, 2),
    ("symbol", "K10"): (0, 1), ("symbol", "J10"): (0, 2),
    ("symbol", "K12"): (0, 1), ("symbol", "J12"): (0, 2),
}


def test_a_distance_two_item_stands_behind_each_depends_on_hop_kind_at_the_exact_distance_and_possible_count():
    w = run(depends_pins(), ("Symbol", "F"), column=bfs.DEPENDS_ON)
    assert {(i.kind, i.id): (i.possible_hops, i.distance) for i in w.items()} == EXPECT
    assert w.unexplored == 0


def test_each_depends_on_hop_kind_appears_on_a_chain_with_its_edge_and_direction():
    w = run(depends_pins(), ("Symbol", "F"), column=bfs.DEPENDS_ON)
    seen = {(s["edge"]["type"], s["edge"]["dir"]) for i in w.items() for s in i.chain[1:]}
    assert seen == {("HAS_CALL", "->"), ("HAS_RETURN", "->"), ("HAS_COND", "->"), ("HAS_PARAM", "->"), ("CALLS", "->"), ("CALLS_NAME", "->"), ("VIA", "->"), ("TO", "->"),
                    ("TO_UNIT", "->"), ("DATA", "->"), ("CONTROLLED_BY", "->")}
    j7 = items_by_id(w)["J7"]
    assert edge_types(j7) == [("HAS_PARAM", "->"), ("CONTROLLED_BY", "->"), ("DATA", "->"), ("CALLS", "->"), ("HAS_CALL", "->"), ("CALLS", "->")]


def test_the_terminal_depends_on_states_are_the_ones_the_rows_name():
    w = run(depends_pins(), ("Symbol", "F"), column=bfs.DEPENDS_ON)
    assert ("Unit", "un", "end", "") in w.visited
    assert {k[2:] for k in w.visited if k[0] != "Unit"} == {("value", "")}


def test_the_contract_rows_with_increment_zero_cost_no_distance():
    """``to_in:contract`` (a provider's link) and ``provides_out`` (a key change reaching its contract point) add nothing: the consumers stay at distance 1."""
    items = items_by_id(run(contract_graph(), ("Symbol", "P"), S.CONTRACT))
    assert items["F3"].distance == 1
    w = run(contract_graph(), ("Symbol", "P"), S.KEY)
    assert w.best[("ContractPoint", "cp1", "contract", "")].cost[1] == 0 and w.best[("Symbol", "P", "contract", "")].cost[1] == 0
    assert {i.id: i.distance for i in w.items()} == {"F3": 1, "F4": 1, "F5": 1}


def test_a_provider_in_contract_state_reaches_the_link_at_distance_zero():
    g = G()
    g.sym("P"), g.sym("F"), g.site("c", "F"), g.link("L", provider="P"), g.via("c", "L")
    w = run(g, ("Symbol", "P"), S.CONTRACT)
    assert w.best[("Link", "L", "contract", "")].cost[1] == 0 and w.best[("Link", "L", "contract", "")].edge.type == "TO"
