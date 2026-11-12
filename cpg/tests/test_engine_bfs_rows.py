"""P1-22 the walk's rows, unit tier, part 2: ``arg(i)``/``Ret``, contract mode, wiring items, ``TO_UNIT``, the name hop, the depth bound and ``unexplored``.

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-22 (2), (4), (5), (6); ``-answer-spec2.md`` §3.3, §3.5, §4. ``G(flow=True)`` everywhere: the flag gate is P1-23's.
"""
from __future__ import annotations

import pytest
from engine_helpers import ScriptedReader
from engine_walk_helpers import G, MemoryReader

from ccg.engine import bfs, config
from ccg.engine import state as S
from ccg.engine.reader import GraphDataError
from test_engine_bfs import edge_types, items_by_id, nodes, run


# --------------------------------------------------------------------------- (4) arg(i) and Ret rows

def arg_graph(*, ret_summary: bool, param_index: int = 0, arg_to: str = "a0.k"):
    """X's ``cx`` calls F with ``opts.k`` as its argument; F's parameter ``pf`` (index ``param_index``) feeds F's return when ``ret_summary``; Y also calls F."""
    g = G()
    for s in ("X", "F", "Y"):
        g.sym(s)
    g.param("px", "X", index=0, name="opts")
    g.site("cx", "X", callee="f")
    g.calls("cx", "F")
    g.data("CallSite", "cx", "Param", "px", "p0.k", arg_to)
    g.param("pf", "F", index=param_index, name="o")
    g.ret("rf", "F")
    if ret_summary:
        g.data("Ret", "rf", "Param", "pf", "p0.k", "ret")
    g.site("cy", "Y")
    g.calls("cy", "F")
    return g


def test_a_changed_argument_enters_the_callee_and_its_ret_summary_reaches_the_other_callers():
    w = run(arg_graph(ret_summary=True), ("Param", "px"), S.path_state("p0.k"))
    items = items_by_id(w)
    assert {k: (i.distance, i.status) for k, i in items.items()} == {"X": (0, "confirmed"), "F": (1, "confirmed"), "Y": (2, "confirmed")}
    assert nodes(items["F"]) == [("Param", "px"), ("CallSite", "cx"), ("Symbol", "F"), ("Param", "pf"), ("Symbol", "F")]
    assert edge_types(items["F"]) == [("DATA", "<-"), ("CALLS", "->"), ("HAS_PARAM", "->"), ("HAS_PARAM", "<-")]
    assert [s["state"] for s in items["F"].chain] == [("path", "p0.k"), ("path", "a0.k"), ("path", "a0.k"), ("path", "p0.k"), ("end", "")]
    assert edge_types(items["Y"])[-3:] == [("HAS_RETURN", "<-"), ("CALLS", "<-"), ("HAS_CALL", "<-")]
    assert items["Y"].via_data is True


def test_a_caller_of_a_callee_whose_return_the_changed_argument_does_not_reach_is_not_listed():
    items = items_by_id(run(arg_graph(ret_summary=False), ("Param", "px"), S.path_state("p0.k")))
    assert set(items) == {"X", "F"}  # F holds the changed parameter; Y, which only calls F, is no longer listed through it


def test_a_changed_argument_reaches_only_the_parameter_of_its_index():
    ok = items_by_id(run(arg_graph(ret_summary=False, param_index=0), ("Param", "px"), S.path_state("p0.k")))
    off = items_by_id(run(arg_graph(ret_summary=False, param_index=1), ("Param", "px"), S.path_state("p0.k")))
    assert "F" in ok and "F" not in off


def test_the_receiver_is_index_minus_one_and_a_keyword_argument_matches_by_parameter_name():
    g = G()
    for s in ("X", "F"):
        g.sym(s)
    g.param("px", "X", index=0, name="self_")
    g.site("cx", "X"), g.calls("cx", "F")
    g.data("CallSite", "cx", "Param", "px", "p0", "recv")
    g.param("p_recv", "F", index=-1, name="self"), g.param("p_a0", "F", index=0, name="a"), g.param("p_kw", "F", index=2, name="size")
    w = run(g, ("Param", "px"), S.path_state("p0"))
    assert {k[1] for k in w.visited if k[0] == "Param"} == {"px", "p_recv"}
    g2 = G()
    for s in ("X", "F"):
        g2.sym(s)
    g2.param("px", "X", index=0, name="n"), g2.site("cx", "X"), g2.calls("cx", "F")
    g2.data("CallSite", "cx", "Param", "px", "p0", "kw.size")
    g2.param("p_recv", "F", index=-1, name="size"), g2.param("p_a0", "F", index=0, name="size2"), g2.param("p_kw", "F", index=2, name="size")
    w2 = run(g2, ("Param", "px"), S.path_state("p0"))
    assert {k[1] for k in w2.visited if k[0] == "Param"} == {"px", "p_kw"}
    assert ("Param", "p_kw", "path", "p2") in w2.visited


def test_a_call_sites_result_path_never_enters_a_callee_and_is_never_counted_as_cut():
    g = G()
    g.sym("A"), g.sym("B"), g.site("cs", "B"), g.calls("cs", "A")
    r = MemoryReader(g)
    w = bfs.walk(r, ("Symbol", "A"), None, bound=1)
    assert "calls_out" not in r.names() and "name_out" not in r.names()
    assert w.unexplored == 0 and ("Symbol", "A", "path", "r") not in w.visited


def test_a_name_matched_callee_carries_the_argument_too_as_a_possible_hop():
    g = G()
    for s in ("X", "F"):
        g.sym(s)
    g.param("px", "X"), g.site("cx", "X", callee="f"), g.calls_name("cx", "F", "f")
    g.data("CallSite", "cx", "Param", "px", "p0", "a0")
    g.param("pf", "F", index=0)
    w = run(g, ("Param", "px"), S.path_state("p0"))
    f = items_by_id(w)["F"]
    assert (f.status, f.reason, f.distance) == ("possible", "name-match", 1)


# --------------------------------------------------------------------------- (2) contract mode: CONSUMES <-, ON <-, PROVIDES <-

def contract_graph():
    g = G()
    for s in ("P", "F1", "F2", "F3", "F4", "F5"):
        g.sym(s)
    g.cp("cp0", "GET /orphan")                       # no provider, no link
    g.site("c1", "F1"), g.site("c2", "F2")
    g.consumes("c1", "cp0"), g.consumes("c2", "cp0", status="possible", reason="runtime-built")
    g.cp("cp1", "GET /ok")
    g.provides("P", "cp1")
    g.link("L", provider="P", cp="cp1")
    g.site("c3", "F3"), g.consumes("c3", "cp1"), g.via("c3", "L")       # matched: reached through the Link
    g.site("c4", "F4"), g.consumes("c4", "cp1")                         # unmatched: no Link reaches it
    g.link("L2", cp="cp1", status="possible", reason="port-shared")
    g.site("c5", "F5"), g.consumes("c5", "cp1", status="possible", reason="port-shared"), g.via("c5", "L2", status="possible", reason="port-shared")
    return g


def test_a_contract_walk_from_a_provider_less_point_lists_its_consumers_by_consumes():
    w = run(contract_graph(), ("ContractPoint", "cp0"))
    items = items_by_id(w)
    assert {k: (i.distance, i.status, i.reason) for k, i in items.items()} == {"F1": (1, "confirmed", None), "F2": (1, "possible", "runtime-built")}
    assert nodes(items["F1"]) == [("ContractPoint", "cp0"), ("CallSite", "c1"), ("Symbol", "F1")]
    assert edge_types(items["F1"]) == [("CONSUMES", "<-"), ("HAS_CALL", "<-")]
    assert [s["state"] for s in items["F1"].chain] == [("contract", ""), ("contract", ""), ("end", "")]


def test_a_consumer_a_link_reaches_comes_through_the_link_and_only_an_unmatched_one_by_consumes():
    w = run(contract_graph(), ("ContractPoint", "cp1"))
    items = items_by_id(w)
    assert set(items) == {"F3", "F4", "F5"}
    assert nodes(items["F3"]) == [("ContractPoint", "cp1"), ("Link", "L"), ("CallSite", "c3"), ("Symbol", "F3")]
    assert edge_types(items["F3"]) == [("ON", "<-"), ("VIA", "<-"), ("HAS_CALL", "<-")]
    assert edge_types(items["F4"]) == [("CONSUMES", "<-"), ("HAS_CALL", "<-")]
    assert w.best[("CallSite", "c3", "contract", "")].edge.type == "VIA"
    assert items["F3"].distance == items["F4"].distance == 1


def test_a_possible_link_makes_its_consumers_possible_with_the_links_reason_from_the_first_possible_hop():
    f5 = items_by_id(run(contract_graph(), ("ContractPoint", "cp1")))["F5"]
    assert (f5.status, f5.reason, f5.possible_hops) == ("possible", "port-shared", 2)  # the ON <- hop and the VIA <- hop
    assert f5.chain[1]["edge"] == {"type": "ON", "dir": "<-", "status": "possible", "reason": "port-shared"}


def test_the_provider_is_listed_by_the_contract_row_but_is_not_an_item():
    w = run(contract_graph(), ("ContractPoint", "cp1"))
    assert ("Symbol", "P", "contract", "") in w.visited and w.best[("Symbol", "P", "contract", "")].edge.type == "PROVIDES"
    assert w.best[("Symbol", "P", "contract", "")].cost[1] == 0
    assert "P" not in items_by_id(w)


def test_a_changed_provider_whose_key_changed_switches_to_the_contract_walk():
    w = run(contract_graph(), ("Symbol", "P"), S.KEY)
    assert w.best[("ContractPoint", "cp1", "contract", "")].edge.dir == "->"
    assert {"F3", "F4", "F5"} <= set(items_by_id(w))


def test_a_provider_in_contract_state_reaches_its_consumers_through_the_link_it_provides():
    g = G()
    for s in ("P", "F3"):
        g.sym(s)
    g.cp("cp1"), g.provides("P", "cp1"), g.link("L", provider="P"), g.site("c3", "F3"), g.via("c3", "L")
    items = items_by_id(run(g, ("Symbol", "P"), S.CONTRACT))
    assert edge_types(items["F3"]) == [("TO", "<-"), ("VIA", "<-"), ("HAS_CALL", "<-")] and [s["state"][0] for s in items["F3"].chain] == ["contract"] * 3 + ["end"]


def test_the_matched_set_is_per_contract_point_and_per_site():
    """c3 reaches cp_a through a Link and consumes cp_b too: on cp_b it has no Link, so it is an unmatched consumer there."""
    g = G()
    g.sym("F"), g.site("c3", "F")
    g.cp("cp_a"), g.cp("cp_b"), g.link("L", cp="cp_a"), g.via("c3", "L"), g.consumes("c3", "cp_a"), g.consumes("c3", "cp_b")
    assert items_by_id(run(g, ("ContractPoint", "cp_b")))["F"].chain[1]["edge"]["type"] == "CONSUMES"
    assert items_by_id(run(g, ("ContractPoint", "cp_a")))["F"].chain[1]["edge"]["type"] == "ON"


def test_the_matched_set_is_per_contract_point_when_several_points_are_expanded_at_once():
    """P provides cp_a and cp_b (one level, one read of each query). c3 reaches cp_a through Link L and consumes cp_b with no link on cp_b: unmatched there."""
    g = G()
    g.sym("P"), g.sym("F"), g.site("c3", "F")
    g.cp("cp_a"), g.cp("cp_b"), g.provides("P", "cp_a"), g.provides("P", "cp_b")
    g.link("L", cp="cp_a"), g.via("c3", "L"), g.consumes("c3", "cp_a"), g.consumes("c3", "cp_b")
    w = run(g, ("Symbol", "P"), S.KEY)
    best = w.best[("CallSite", "c3", "contract", "")]
    assert (best.edge.type, best.pred_key) == ("CONSUMES", ("ContractPoint", "cp_b", "contract", ""))   # not silently dropped as "matched" because cp_a's link reaches it


# --------------------------------------------------------------------------- wiring items (spec2 §3.3)

def wiring_graph():
    g = G()
    for s in ("P", "F"):
        g.sym(s)
    g.link("L", provider="P", status="possible", reason="port-shared")
    g.site("cs", "F"), g.via("cs", "L", status="possible", reason="port-shared")
    g.entry("e1", key_path="services.api.url", line=3), g.entry("e2", key_path="services.api.port", line=9), g.entry("eL", launch="command", line=20)
    g.bound("L", "e1", "s_b", ord=0, port=222, portOrigin="explicit")
    g.bound("L", "e1", "s_a", ord=5, port=111, portOrigin="compose")
    g.bound("L", "e2", "s_a", ord=0)
    g.bound("L", "eL", "s_a", ord=1, port=8001, portOrigin="pack-default", packId="fastapi")
    return g


def test_each_config_entry_a_reached_link_is_bound_by_is_one_config_item_at_distance_one_with_the_links_status():
    w = run(wiring_graph(), ("Symbol", "P"))
    cfg = {i.id: i for i in w.items() if i.kind == "config"}
    assert set(cfg) == {"e1", "e2"}  # the launch entry eL is evidence, never an item
    e1 = cfg["e1"]
    assert (e1.distance, e1.status, e1.reason, e1.possible_hops) == (1, "possible", "port-shared", 2)   # the TO <- hop and the BOUND_BY hop are both the Link's
    assert nodes(e1) == [("Symbol", "P"), ("Link", "L"), ("ConfigEntry", "e1")] and edge_types(e1) == [("TO", "<-"), ("BOUND_BY", "->")]
    assert e1.chain[-1]["state"] == ("end", "") and e1.chain[-1]["edge"]["status"] == "possible"


def test_the_item_takes_its_bound_from_the_entrys_lowest_site_and_ord_edge():
    f = run(wiring_graph(), ("Symbol", "P")).facts
    assert f[("ConfigEntry", "e1")]["bound"] == {"ord": 5, "port": 111, "portOrigin": "compose"}   # (s_a, 5) < (s_b, 0)
    assert f[("ConfigEntry", "e1")]["keyPath"] == "services.api.url" and f[("ConfigEntry", "e1")]["path"] == "cfg/a.yaml"
    assert f[("ConfigEntry", "e2")]["bound"] == {"ord": 0}


def test_the_consumers_of_the_link_are_listed_beside_its_wiring():
    items = items_by_id(run(wiring_graph(), ("Symbol", "P")))
    assert items["F"].distance == 1 and edge_types(items["F"]) == [("TO", "<-"), ("VIA", "<-"), ("HAS_CALL", "<-")]


def sites_graph():
    """Link L has consumer sites s1, s2, s3 (in C1, C2, C3); entry E binds s1 and s3 only, entry E2 binds s2."""
    g = G()
    for s in ("P", "C1", "C2", "C3"):
        g.sym(s)
    g.link("L", provider="P")
    for sid, owner in (("s1", "C1"), ("s2", "C2"), ("s3", "C3")):
        g.site(sid, owner), g.via(sid, "L")
    g.entry("E", key_path="svc.url"), g.entry("E2", key_path="svc.port")
    g.bound("L", "E", "s1", ord=0), g.bound("L", "E", "s3", ord=1), g.bound("L", "E2", "s2", ord=0)
    return g


def test_a_config_change_reaches_only_the_sites_it_binds_and_lists_no_wiring_entry_back():
    w = run(sites_graph(), ("ConfigEntry", "E"))
    items = items_by_id(w)
    assert set(items) == {"C1", "C3"} and {i.distance for i in items.values()} == {2}      # C2's binding is another entry: E's change does not reach it
    assert edge_types(items["C1"]) == [("BOUND_BY", "<-"), ("VIA", "<-"), ("HAS_CALL", "<-")]
    assert {k[3] for k in w.visited if k[0] == "Link"} == {"s1", "s3"} and all(k[2] == "site" for k in w.visited if k[0] == "Link")
    assert ("CallSite", "s2", "path", "r") not in w.visited


def test_an_entry_bound_at_two_sites_of_one_link_lists_exactly_those_two_consumers():
    assert set(items_by_id(run(sites_graph(), ("ConfigEntry", "E")))) == {"C1", "C3"}
    assert set(items_by_id(run(sites_graph(), ("ConfigEntry", "E2")))) == {"C2"}


def test_a_config_entry_that_binds_a_site_the_link_has_no_via_for_lists_nothing_for_it():
    g = sites_graph()
    g.entry("E3", key_path="svc.x"), g.bound("L", "E3", "s_unknown", ord=0)
    w = run(g, ("ConfigEntry", "E3"))
    assert w.items() == [] and ("Link", "L", "site", "s_unknown") in w.visited


def test_a_bound_in_row_without_a_site_id_is_malformed():
    r = ScriptedReader({"bound_in": [("E", "L", "http", "confirmed", "key-match", None, 0)]})
    with pytest.raises(GraphDataError):
        bfs.walk(r, ("ConfigEntry", "E"))


def test_a_site_state_link_does_not_list_wiring_entries_nor_other_sites_when_the_site_id_is_not_a_via_site():
    w = run(wiring_graph(), ("ConfigEntry", "e1"))
    assert not any(i.kind == "config" for i in w.items())      # the wiring row never fires from a Link entered by BOUND_BY <-


def test_wiring_applies_in_contract_state_too():
    g = wiring_graph()
    g.cp("cp1"), g.edge("Link", "ON", "ContractPoint", "L", "cp1")
    cfg = {i.id for i in run(g, ("ContractPoint", "cp1")).items() if i.kind == "config"}
    assert cfg == {"e1", "e2"}


# --------------------------------------------------------------------------- depends-on column, TO_UNIT (spec2 §3.5)

def depends_graph():
    g = G()
    for s in ("F", "G1", "H", "K", "J", "Z"):
        g.sym(s)
    g.unit_node("un", "repo-b", flow=False, root="svc/new")
    g.site("c1", "F"), g.calls("c1", "G1")                                        # a resolved call
    g.site("c2", "F"), g.link("L", provider="H"), g.via("c2", "L")                 # a link to a provider
    g.site("c3", "F"), g.link("LU", to_unit="un", status="possible", reason="target-not-mapped"), g.via("c3", "LU", status="possible", reason="target-not-mapped")
    g.site("c4", "F", callee="k"), g.calls_name("c4", "K", "k")                    # a name-match candidate
    g.site("cg", "G1"), g.calls("cg", "J")                                         # G1 calls J: distance 2
    g.site("cj", "J"), g.calls("cj", "Z")                                          # J calls Z: distance 3 (cut)
    return g


def test_the_reverse_walk_lists_what_the_function_depends_on_by_call_link_and_unit():
    w = run(depends_graph(), ("Symbol", "F"), column=bfs.DEPENDS_ON)
    items = {(i.kind, i.id): i for i in w.items()}
    assert {k: (i.distance, i.status, i.reason) for k, i in items.items()} == {
        ("symbol", "G1"): (1, "confirmed", None), ("symbol", "H"): (1, "confirmed", None), ("unit", "un"): (1, "possible", "target-not-mapped"),
        ("symbol", "K"): (1, "possible", "name-match"), ("symbol", "J"): (2, "confirmed", None)}
    assert edge_types(items[("symbol", "H")]) == [("HAS_CALL", "->"), ("VIA", "->"), ("TO", "->")]
    assert edge_types(items[("symbol", "G1")]) == [("HAS_CALL", "->"), ("CALLS", "->")]
    assert w.cut == frozenset({("CALLS", "cj", "Z")}) and w.unexplored == 1


def test_a_link_with_no_provider_reaches_a_unit_hop_that_is_possible_target_not_mapped_whatever_the_link_says():
    g = depends_graph()
    g.rows.nodes["Link"][1]["status"], g.rows.nodes["Link"][1]["reason"] = "confirmed", "key-match"   # LU claims confirmed: the Unit hop still says what the registry row says
    for e in g.rows.edges[("CallSite", "VIA", "Link")]:
        if e["t"] == "LU":
            e["p"]["status"], e["p"]["reason"] = "confirmed", "key-match"
    unit = [i for i in run(g, ("Symbol", "F"), column=bfs.DEPENDS_ON).items() if i.kind == "unit"][0]
    assert unit.chain[-1]["edge"] == {"type": "TO_UNIT", "dir": "->", "status": "possible", "reason": "target-not-mapped"}
    assert unit.chain[-1]["state"] == ("end", "") and unit.distance == 1 and unit.id == "un"


def test_the_unit_hop_adds_no_distance_and_the_link_hop_one():
    w = run(depends_graph(), ("Symbol", "F"), column=bfs.DEPENDS_ON)
    assert w.best[("Unit", "un", "end", "")].cost[1] == 1 == w.best[("Link", "LU", "value", "")].cost[1]
    assert w.facts[("Unit", "un")] == {"repo": "repo-b", "unit": "un", "rootPath": "svc/new", "buildStatus": "fresh"}


def test_a_provider_reached_through_a_possible_link_carries_the_links_status_on_the_to_hop():
    g = G()
    for s in ("F", "H"):
        g.sym(s)
    g.site("c2", "F"), g.link("L", provider="H", status="possible", reason="port-shared"), g.via("c2", "L")
    h = items_by_id(run(g, ("Symbol", "F"), column=bfs.DEPENDS_ON))["H"]
    assert [s["edge"]["status"] for s in h.chain[1:]] == ["confirmed", "confirmed", "possible"] and (h.status, h.reason) == ("possible", "port-shared")
    assert h.chain[-1]["edge"] == {"type": "TO", "dir": "->", "status": "possible", "reason": "port-shared"}


def test_points_reached_by_the_reverse_walk_get_their_owners_repo_unit_and_path_from_the_owner_join():
    w = run(depends_graph(), ("Symbol", "F"), column=bfs.DEPENDS_ON)
    assert w.facts[("CallSite", "c1")] == {"line": 5, "name": "f", "repo": "repo-a", "unit": "ua", "path": "a.py"}


def test_the_reverse_walk_from_a_point_follows_data_and_control_to_the_functions_it_depends_on():
    g = G()
    for s in ("F", "G1", "H"):
        g.sym(s)
    g.ret("rf", "F"), g.site("c1", "F"), g.calls("c1", "G1"), g.cond("k", "F"), g.site("c2", "F"), g.calls("c2", "H")
    g.data("Ret", "rf", "CallSite", "c1", "r", "ret")
    g.controlled("Ret", "rf", "k"), g.data("Cond", "k", "CallSite", "c2", "r", "")
    items = items_by_id(run(g, ("Ret", "rf"), column=bfs.DEPENDS_ON))
    assert set(items) == {"G1", "H"} and items["G1"].distance == items["H"].distance == 1
    assert edge_types(items["H"]) == [("CONTROLLED_BY", "->"), ("DATA", "->"), ("CALLS", "->")]


# --------------------------------------------------------------------------- (6) the name hop

def name_graph():
    g = G()
    for s in ("A", "Z", "B", "C", "D", "E"):
        g.sym(s)
    g.site("cB", "B"), g.calls("cB", "A")                                     # resolved: CALLS only
    g.site("cC", "C", callee="a"), g.calls_name("cC", "A", "a")               # unresolved: a candidate, name-match
    g.site("cD", "D", callee="a"), g.calls("cD", "Z"), g.calls_name("cD", "A", "a")   # resolved elsewhere: a candidate, resolution-unverified
    g.site("cE", "E", callee="a"), g.calls("cE", "A"), g.calls_name("cE", "A", "a")   # the resolved pair is suppressed, never listed twice
    return g


def test_a_name_candidate_is_possible_and_the_reason_depends_on_whether_the_site_resolved_elsewhere():
    items = items_by_id(run(name_graph(), ("Symbol", "A")))
    assert {k: (i.status, i.reason, i.distance) for k, i in items.items()} == {
        "B": ("confirmed", None, 1), "C": ("possible", "name-match", 1), "D": ("possible", "resolution-unverified", 1), "E": ("confirmed", None, 1)}
    assert edge_types(items["C"]) == [("CALLS_NAME", "<-"), ("HAS_CALL", "<-")] and edge_types(items["E"])[0] == ("CALLS", "<-")
    assert items["C"].chain[1]["edge"] == {"type": "CALLS_NAME", "dir": "<-", "status": "possible", "reason": "name-match"}


@pytest.mark.parametrize("count,reason", [(0, "name-match"), (1, "resolution-unverified"), (2, "resolution-unverified"), (7, "resolution-unverified")])
def test_the_count_column_alone_decides_the_reason(count, reason):
    r = ScriptedReader({"name_in": [("A", "cs", "repo-a", "ua", "a.py", 5, "a", count)], "owner_of:CallSite": [("cs", "F", "repo-a", "ua", "F", "a.py", 1, 9, None)]})
    w = bfs.walk(r, ("Symbol", "A"))
    assert w.best[("CallSite", "cs", "path", "r")].edge.reason == reason and w.best[("CallSite", "cs", "path", "r")].edge.status == "possible"


@pytest.mark.parametrize("count", [-1, "1", 1.0, None, True])
def test_a_count_that_is_not_a_non_negative_integer_is_a_graph_data_error(count):
    r = ScriptedReader({"name_in": [("A", "cs", "repo-a", "ua", "a.py", 5, "a", count)]})
    with pytest.raises(GraphDataError):
        bfs.walk(r, ("Symbol", "A"))


# --------------------------------------------------------------------------- (5) the depth bound and unexplored

def caller_chain(n: int, **extra):
    """S0 <- S1 <- ... <- Sn, each caller returning its call's result so that the change continues (distance i at S_i)."""
    g = G()
    for i in range(n + 1):
        g.sym(f"S{i}")
    for i in range(1, n + 1):
        g.site(f"c{i}", f"S{i}"), g.calls(f"c{i}", f"S{i-1}"), g.ret(f"r{i}", f"S{i}")
        g.data("Ret", f"r{i}", "CallSite", f"c{i}", "r", "ret")
    return g


def test_the_default_bound_is_two_and_a_distance_two_item_is_reached_by_default():
    w = run(caller_chain(4), ("Symbol", "S0"))
    assert config.DEFAULT_DEPTH_BOUND == 2 and w.bound == 2
    assert {i.id: i.distance for i in w.items()} == {"S1": 1, "S2": 2}


@pytest.mark.parametrize("bound,reached", [(1, {"S1"}), (2, {"S1", "S2"}), (3, {"S1", "S2", "S3"}), (4, {"S1", "S2", "S3", "S4"})])
def test_items_are_exactly_those_within_the_bound(bound, reached):
    w = run(caller_chain(4), ("Symbol", "S0"), bound=bound)
    assert {i.id for i in w.items()} == reached and max(i.distance for i in w.items()) == bound


@pytest.mark.parametrize("bound,unexplored", [(1, 1), (2, 1), (3, 1), (4, 0)])
def test_the_cut_edge_is_counted_when_something_lies_beyond_the_bound(bound, unexplored):
    assert run(caller_chain(4), ("Symbol", "S0"), bound=bound).unexplored == unexplored


def test_an_increment_zero_row_is_followed_at_the_bound_and_never_counted():
    w = run(caller_chain(2), ("Symbol", "S0"), bound=1)
    # S1 is at distance 1 == bound; its Ret, and its callers' sites beyond, are cut only by the CALLS <- row (increment 1)
    assert ("Ret", "r1", "path", "ret") in w.visited and ("Symbol", "S1", "value", "") in w.visited and w.unexplored == 1
    assert w.cut == frozenset({("CALLS", "c2", "S1")})


def test_every_row_with_an_increment_of_one_counts_its_cut_edges():
    g = caller_chain(2)
    g.sym("S3"), g.site("c3", "S3"), g.calls("c3", "S2")                                # a second caller of S2 (resolved)
    g.site("c3b", "S3", callee="f"), g.calls_name("c3b", "S2", "f")                     # a name candidate of S2
    g.link("L", provider="S2"), g.site("cv", "S3"), g.via("cv", "L")                    # a consumer of a link S2 provides
    g.entry("e1"), g.bound("L", "e1", "cv")                                             # a wiring entry
    w = run(g, ("Symbol", "S0"), bound=2)
    assert w.cut == frozenset({("CALLS", "c3", "S2"), ("CALLS_NAME", "c3b", "S2"), ("VIA", "cv", "L"), ("BOUND_BY", "L", "e1")}) and w.unexplored == 4
    assert {i.id for i in w.items()} == {"S1", "S2"}


def test_a_stored_edge_two_states_reach_is_counted_once():
    """cs (the call's result, distance 1) feeds ``c`` at two arguments; ``c`` calls F: the one CALLS edge is cut from both states (a0 and a1)."""
    g = G()
    for s in ("A", "B", "C1", "F"):
        g.sym(s)
    g.site("cs", "B"), g.calls("cs", "A"), g.site("c", "C1"), g.calls("c", "F")
    g.data("CallSite", "c", "CallSite", "cs", "r", "a0"), g.data("CallSite", "c", "CallSite", "cs", "r", "a1")
    w = run(g, ("Symbol", "A"), bound=1)
    assert {("CallSite", "c", "path", "a0"), ("CallSite", "c", "path", "a1")} <= w.visited
    assert w.cut == frozenset({("CALLS", "c", "F")}) and w.unexplored == 1
    assert run(g, ("Symbol", "A"), bound=2).unexplored == 0 and "F" not in {i.id for i in run(g, ("Symbol", "A"), bound=1).items()}


def test_a_cut_edge_is_counted_whether_or_not_its_target_is_reached_some_other_way():
    g = caller_chain(2)
    # S2 is also reached directly: c2b in S3 calls S1 (distance 2) and S3 is the target of the cut edge c3 -> S2 only if S3 owns c3
    g.sym("S3"), g.site("c3", "S3"), g.calls("c3", "S2"), g.site("c3x", "S3"), g.calls("c3x", "S0")   # S3 is reached at distance 1 by c3x
    w = run(g, ("Symbol", "S0"), bound=2)
    assert "S3" in {i.id for i in w.items()} and w.cut == frozenset({("CALLS", "c3", "S2")})


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True, None.__class__])
def test_a_bound_that_is_not_a_positive_integer_is_refused(bad):
    with pytest.raises(bfs.WalkError):
        run(caller_chain(1), ("Symbol", "S0"), bound=bad)


def test_the_walk_reads_the_default_bound_from_config_at_call_time(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_DEPTH_BOUND", 3)
    assert run(caller_chain(4), ("Symbol", "S0")).bound == 3
