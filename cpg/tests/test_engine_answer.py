"""P1-39b: the answer builder (``ccg.engine.answer_build``): the walk's output as the canonical answer document, validated by ``ccg.answer``.

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-39 (0)-(3), (5), (6); ``-answer-spec2.md`` §3 (hops, wiring items, ``site``, ``Unit``), §3.5/§4.3 (staleness, ``builtAt``
outside the facts), §5, §6 (v1 text it replaces), §7 (V1-V19), §8 (one answer-level test per staleness-unit source); ``-answer-spec.md`` §4-§6 (document shapes, canonical bytes,
``facts_bytes``). Offline: worlds in readable names rewritten to 32-hex ids (``tests/engine_answer_helpers.py``); the one live round trip is ``test_engine_answer_live.py``.
"""
from __future__ import annotations

import copy
import dataclasses
import re

import pytest
from engine_answer_helpers import SNAP, T1, T2, AnswerReader, Shuffled, hexify
from engine_walk_helpers import G

from ccg import answer
from ccg.engine import answer_build as AB
from ccg.engine import anchors, bfs, config
from ccg.engine.reader import GraphDataError


def sym_anchor(g, ids_, name):
    n = next(s for s in g.rows.nodes["Symbol"] if s["symbolId"] == ids_[name])
    return anchors.SymbolAnchor(n["symbolId"], n["qname"], n["repoId"], n["path"], n["unitId"], n["lineStart"], n["lineEnd"])


def cp_anchor(g, ids_, name):
    n = next(c for c in g.rows.nodes["ContractPoint"] if c["cpId"] == ids_[name])
    return anchors.ContractAnchor(n["cpId"], n["kind"], n["keyNorm"])


def cfg_anchor(g, ids_, name):
    n = next(c for c in g.rows.nodes["ConfigEntry"] if c["entryId"] == ids_[name])
    return anchors.ConfigAnchor(n["entryId"], n["repoId"], n["path"], n["keyPath"], n["unitId"], n["line"])


def built(g, ids_, make, name, **kw):
    reader = kw.pop("reader", None) or AnswerReader(g)
    return AB.build_answer(reader, make(g, ids_, name), **kw), reader


def chain_of(doc, item_index=0):
    return doc["items"][item_index]["chain"]


def subject_id(it):
    return it["subject"].get("symbolId") or it["subject"].get("entryId") or it["subject"]["unitId"]


def item_of(doc, ids_, name):
    (it,) = [i for i in doc["items"] if subject_id(i) == ids_[name]]
    return it


def states(chain):
    return [h["state"] for h in chain]


# ------------------------------------------------------------------------------------------------------------------ worlds

def caller_world(**kw):
    return hexify(caller_world_g(**kw))


def caller_world_g(*, flow=True, lang="python"):
    """``C`` calls the anchor ``A`` from site ``cC``: A <-CALLS- cC <-HAS_CALL- C."""
    g = G(unit="u1", flow=flow, lang=lang)
    g.unit_node("u2", flow=flow, root="svc", lang=lang)
    g.sym("A", "m.py::A", path="m.py", start=3, end=9)
    g.sym("C", "svc/c.py::C", path="svc/c.py", start=10, end=30, unit="u2")
    g.site("cC", "C", line=12, callee="A", path="svc/c.py", unit="u2")
    g.calls("cC", "A")
    return g


# ------------------------------------------------------------------------------------------------------------------ (0) the document

def test_a_direct_caller_builds_a_valid_resolved_1_1_answer_whose_chain_is_the_walks():
    g, ids_ = caller_world()
    doc, _ = built(g, ids_, sym_anchor, "A")
    answer.validate(doc)
    assert doc["answerSchemaVersion"] == "1.1"
    assert doc["interpretation"] == {"status": "resolved", "request": {"useCase": "impact", "direction": "impact", "anchors": [
        {"type": "symbol", "symbolId": ids_["A"], "qname": "m.py::A", "repoId": "repo-a", "path": "m.py"}]}}
    (item,) = doc["items"]
    assert item["subject"] == {"type": "symbol", "symbolId": ids_["C"], "qname": "svc/c.py::C", "path": "svc/c.py", "lineStart": 10, "lineEnd": 30}
    assert (item["repoId"], item["unitId"], item["status"], item["directness"], item["distance"], item["viaData"]) == ("repo-a", ids_["u2"], "confirmed", "direct", 1, False)
    assert "reason" not in item
    assert doc["groups"] == []
    chain = item["chain"]
    assert [(h["kind"], h["id"]) for h in chain] == [("Symbol", ids_["A"]), ("CallSite", ids_["cC"]), ("Symbol", ids_["C"])]
    assert chain[0] == {"kind": "Symbol", "id": ids_["A"], "repo": "repo-a", "unit": ids_["u1"], "path": "m.py", "line": 3, "qname": "m.py::A", "state": "value"}
    assert chain[1] == {"kind": "CallSite", "id": ids_["cC"], "repo": "repo-a", "unit": ids_["u2"], "path": "svc/c.py", "line": 12, "calleeName": "A", "state": "path",
                        "accessPath": "r", "edgeType": "CALLS", "dir": "<-", "status": "confirmed"}
    assert (chain[2]["edgeType"], chain[2]["dir"], chain[2]["state"]) == ("HAS_CALL", "<-", "end")


def test_the_recipe_names_the_shipped_definition_the_guards_and_every_query_the_build_ran():
    g, ids_ = caller_world()
    doc, reader = built(g, ids_, sym_anchor, "A")
    assert doc["recipe"]["algorithm"] == {"id": "bfs-v1", "version": "1.1"}
    assert doc["recipe"]["guards"] == {"visitedSeedsAnchor": True, "levelGuard": config.LEVEL_GUARD}
    assert doc["recipe"]["queries"] == sorted(reader.recipe_queries(), key=lambda q: q["name"])
    names = {q["name"] for q in doc["recipe"]["queries"]}
    assert {"calls_in", "units", "snapshot", "symbol_by_qname"} <= names and not names & {"lifted_of", "bound_out", "candidate_count"}     # nothing was asked about links: no text ran


def test_an_impact_item_carries_its_span_from_the_owner_join_so_only_the_anchor_is_re_read_by_qname():
    g, ids_ = caller_world()
    _, reader = built(g, ids_, sym_anchor, "A")
    assert [c for c in reader.calls if c[0] == "symbol_by_qname"] == [("symbol_by_qname", {"repo": "repo-a", "qname": "m.py::A"})]


def test_the_depth_bound_is_recorded_on_every_resolved_answer_with_the_walks_unexplored_count():
    g, ids_ = caller_world()
    doc, _ = built(g, ids_, sym_anchor, "A")
    assert doc["depthBound"] == {"bound": config.DEFAULT_DEPTH_BOUND, "unexplored": 0}


def test_the_stamp_is_the_snapshot_nodes_with_the_p1_verdict_log_pair_and_the_staleness_lists_each_unit_on_a_chain():
    g, ids_ = caller_world()
    doc, _ = built(g, ids_, sym_anchor, "A")
    assert doc["stamp"] == {"snapshotId": SNAP, "verdictLogOffset": 0, "verdictLogHash": answer.EMPTY_VERDICT_LOG_HASH}
    assert [(u["unitId"], u["rootPath"], u["buildStatus"], u["builtCommit"], u["builtAt"]) for u in doc["staleness"]["units"]] == [
        (ids_["u1"], "", "fresh", "a" * 40, T1), (ids_["u2"], "svc", "fresh", "a" * 40, T1)]


def test_the_returned_document_is_canonical_and_round_trips_through_the_validator():
    g, ids_ = caller_world()
    doc, _ = built(g, ids_, sym_anchor, "A")
    assert doc == answer.canonicalize(doc)
    assert answer.loads(answer.dumps(doc)) == doc


# ------------------------------------------------------------------------------------------------------------------ (0) the hop state, per kind

def test_the_state_table_maps_every_walk_state_kind_to_the_hop_state_of_the_same_name_site_included():
    from ccg.engine import state as S
    assert set(AB.HOP_STATE) == set(S.KINDS) and all(AB.HOP_STATE[k] == k for k in S.KINDS)
    assert set(AB.HOP_STATE.values()) == set(answer.load_schema()["$defs"]["Hop"]["properties"]["state"]["enum"])


@pytest.mark.parametrize("kind", ["value", "contract", "key", "end", "site", "function"])
def test_hop_state_gives_the_kind_and_never_an_access_path_but_for_path(kind):
    assert AB.hop_state((kind, "ignored")) == {"state": kind}


def test_a_path_state_gives_state_path_and_its_access_path():
    assert AB.hop_state(("path", "p0.q")) == {"state": "path", "accessPath": "p0.q"}


def test_a_flag_true_call_site_is_in_state_path_r_a_flag_false_ones_is_value_and_the_function_state_is_reported_as_such():
    g, ids_ = caller_world(flow=False)
    doc, _ = built(g, ids_, sym_anchor, "A")
    chain = chain_of(doc)
    assert states(chain) == ["value", "value", "function"] and all("accessPath" not in h for h in chain)
    g, ids_ = caller_world(flow=True)
    doc, _ = built(g, ids_, sym_anchor, "A")
    assert states(chain_of(doc)) == ["value", "path", "end"] and chain_of(doc)[1]["accessPath"] == "r"


def flag_false_two_callers():
    return hexify(flag_false_two_callers_g())


def flag_false_two_callers_g():
    g = G(unit="u1", flow=False, lang="python")
    g.sym("A", "m.py::A", path="m.py"), g.sym("B", "m.py::B", path="m.py", start=11, end=19), g.sym("D", "m.py::D", path="m.py", start=21, end=29)
    g.site("cB", "B", line=12, callee="A"), g.site("cD", "D", line=22, callee="B")
    g.calls("cB", "A"), g.calls("cD", "B")
    return g


def test_a_labelled_hop_is_possible_with_the_flow_unavailable_reason_and_the_item_takes_the_first_possible_hops_reason():
    g, ids_ = flag_false_two_callers()
    doc, _ = built(g, ids_, sym_anchor, "A")
    by = {it["subject"]["symbolId"]: it for it in doc["items"]}
    assert by[ids_["B"]]["status"] == "confirmed" and "reason" not in by[ids_["B"]]
    d = by[ids_["D"]]
    assert (d["status"], d["reason"], d["distance"], d["directness"]) == ("possible", "flow-unavailable:python", 2, "indirect")
    hops = d["chain"]
    assert states(hops) == ["value", "value", "function", "value", "function"]
    assert [(h["edgeType"], h["dir"], h["status"], h.get("reason")) for h in hops[1:]] == [
        ("CALLS", "<-", "confirmed", None), ("HAS_CALL", "<-", "confirmed", None), ("CALLS", "<-", "possible", "flow-unavailable:python"), ("HAS_CALL", "<-", "confirmed", None)]


def test_a_confirmed_hop_never_carries_a_reason_and_a_possible_one_always_does():
    g, ids_ = flag_false_two_callers()
    doc, _ = built(g, ids_, sym_anchor, "A")
    for it in doc["items"]:
        for h in it["chain"][1:]:
            assert ("reason" in h) == (h["status"] == "possible")


# ------------------------------------------------------------------------------------------------------------------ links: candidateCount, lifted, bound

def link_world(**kw):
    return hexify(link_world_g(**kw))


def link_world_g(*, consumer_key="GET /x", sibling_links=0, with_lifted=True, with_bound=True, entry_unit="u_cfg"):
    """cp ``GET /x`` is provided by ``H`` (unit u_srv), linked to the consumer site ``cs`` of ``getHealth`` (unit u_ui); ``sibling_links`` more links hang on other sites of the
    consumer with the same key (candidateCount N = 1 + that); the link carries lifted evidence (from ``cs`` to a wrapper) and a bound config entry (unit ``entry_unit``)."""
    g = G(unit="u_srv", flow=False, lang="python")
    g.unit_node("u_ui", root="ui", lang="javascript", flow=False)
    g.unit_node("u_cfg", root="cfg", lang="yaml", flow=False)
    g.unit_node("u_lift", root="lib", lang="javascript", flow=False)
    g.sym("H", "srv/app.py::health", path="srv/app.py", start=10, end=20)
    g.cp("cp1", key="GET /x")
    g.provides("H", "cp1")
    g.sym("getHealth", "ui/e.ts::getHealth", path="ui/e.ts", start=16, end=23, unit="u_ui")
    g.site("cs", "getHealth", line=17, callee="apiFetch", path="ui/e.ts", unit="u_ui")
    g.link("lk", provider="H", cp="cp1")
    g.via("cs", "lk", key=consumer_key)
    for i in range(sibling_links):
        g.site(f"cx{i}", "getHealth", line=30 + i, callee="apiFetch", path="ui/e.ts", unit="u_ui")
        g.link(f"lk{i}", provider="H", cp="cp1")
        g.via(f"cx{i}", f"lk{i}", key=consumer_key)
    if with_lifted:
        g.sym("wrap", "lib/w.ts::wrap", path="lib/w.ts", start=5, end=9, unit="u_lift")
        g.edge("Link", "LIFTED_VIA", "CallSite", "lk", "cs", siteId="cs", ord=0, line=17, urlPath="a0.path", verbPath="a0.method")
        g.edge("Link", "LIFTED_VIA", "Symbol", "lk", "wrap", siteId="cs", ord=1, line=6, urlPath="p0", verbPath="p1", viaSiteId="cs", viaLine=7)
    if with_bound:
        g.entry("e1", key_path="services.api.ports[0]", path="cfg/compose.yaml", line=4, unit=entry_unit)
        g.bound("lk", "e1", "cs", 0, port=8000, portOrigin="compose")
    return g


def test_a_via_hop_carries_the_candidate_count_n_of_its_site_and_link():
    g, ids_ = link_world(sibling_links=2)
    doc, _ = built(g, ids_, cp_anchor, "cp1")
    item = item_of(doc, ids_, "getHealth")
    via = next(h for h in item["chain"] if h.get("edgeType") == "VIA")
    assert (via["kind"], via["candidateCount"]) == ("CallSite", 3) and via["id"] in {ids_["cs"], ids_["cx0"], ids_["cx1"]}
    assert [h["candidateCount"] for h in item["chain"] if "candidateCount" in h] == [3]


def test_an_unknown_candidate_count_is_a_build_error_never_one():
    g, ids_ = link_world(consumer_key="GET /x")
    for p in g.rows.edges[("CallSite", "VIA", "Link")]:
        p["p"].pop("consumerKey")      # the assembler writes the key only when it is not null: the read then has no row (graph-dba G1)
    with pytest.raises(AB.BuildError, match="candidate"):
        built(g, ids_, cp_anchor, "cp1")


def test_a_link_hop_carries_the_lifted_chain_of_the_site_it_was_entered_for_in_order():
    g, ids_ = link_world()
    doc, _ = built(g, ids_, cp_anchor, "cp1")
    link = next(h for h in item_of(doc, ids_, "getHealth")["chain"] if h["kind"] == "Link")
    assert [(x["ord"], x["kind"], x["id"], x["siteId"], x["unit"], x["repo"], x["path"], x["line"], x["urlPath"], x["name"]) for x in link["lifted"]] == [
        (0, "CallSite", ids_["cs"], ids_["cs"], ids_["u_ui"], "repo-a", "ui/e.ts", 17, "a0.path", "apiFetch"),
        (1, "Symbol", ids_["wrap"], ids_["cs"], ids_["u_lift"], "repo-a", "lib/w.ts", 6, "p0", "lib/w.ts::wrap")]
    assert link["lifted"][0]["verbPath"] == "a0.method" and "viaSiteId" not in link["lifted"][0]
    assert (link["lifted"][1]["viaSiteId"], link["lifted"][1]["viaLine"]) == (ids_["cs"], 7)


def test_a_link_hop_carries_the_bound_entries_of_its_site_with_their_port_evidence():
    g, ids_ = link_world()
    doc, _ = built(g, ids_, cp_anchor, "cp1")
    link = next(h for h in item_of(doc, ids_, "getHealth")["chain"] if h["kind"] == "Link")
    assert link["bound"] == [{"ord": 0, "entryId": ids_["e1"], "repo": "repo-a", "unit": ids_["u_cfg"], "path": "cfg/compose.yaml", "line": 4,
                              "keyPath": "services.api.ports[0]", "port": 8000, "portOrigin": "compose"}]


def test_a_launch_fact_bound_entry_carries_launch_and_no_key_path_and_the_pack_default_port_names_its_pack():
    g = link_world_g(with_bound=False)
    g.node("ConfigEntry", entryId="e_launch", repoId="repo-a", unitId="u_cfg", path=".mcp.json", line=2, launchKind="command", argv0="cypher-mcp/docker-run.sh")
    g.node("ConfigEntry", entryId="e_port", repoId="repo-a", unitId="u_cfg", path="Dockerfile", line=9, launchKind="port", protocol="http", port=8000)
    g.bound("lk", "e_launch", "cs", 0)
    g.bound("lk", "e_port", "cs", 1, port=8001, portOrigin="pack-default", packId="fastapi")     # the edge's matched port is not the entry's declared one
    g, ids_ = hexify(g)
    doc, _ = built(g, ids_, cp_anchor, "cp1")
    link = next(h for h in item_of(doc, ids_, "getHealth")["chain"] if h["kind"] == "Link")
    assert link["bound"][0] == {"ord": 0, "entryId": ids_["e_launch"], "repo": "repo-a", "unit": ids_["u_cfg"], "path": ".mcp.json", "line": 2,
                                "launch": {"launchKind": "command", "argv0": "cypher-mcp/docker-run.sh"}}
    assert link["bound"][1]["launch"] == {"launchKind": "port", "protocol": "http", "port": 8000}
    assert (link["bound"][1]["port"], link["bound"][1]["portOrigin"], link["bound"][1]["packId"]) == (8001, "pack-default", "fastapi") and "keyPath" not in link["bound"][1]


#: the launch fields each ``launchKind`` may carry (the answer schema forbids the others); ``service`` is allowed on all four. Values differ per field, so a column read off by one lands on
#: another field's value or on none.
LAUNCH_FIELDS = {
    "command": {"argv0": "run.sh", "service": "webapp"},
    "port": {"protocol": "tcp", "port": 8123, "service": "webapp"},
    "env-default": {"envVar": "API_URL", "service": "webapp"},
    "mount": {"mountPath": "/srv/data", "service": "webapp"},
}


@pytest.mark.parametrize("kind", sorted(LAUNCH_FIELDS))
def test_every_launch_field_is_read_from_its_own_column_with_distinct_values_per_field(kind):
    g = link_world_g(with_bound=False)
    g.node("ConfigEntry", entryId="e_l", repoId="repo-a", unitId="u_cfg", path="deploy/run.yaml", line=7, launchKind=kind, **LAUNCH_FIELDS[kind])
    g.bound("lk", "e_l", "cs", 0, port=9001, portOrigin="compose")      # the edge's own port differs from the entry's declared one
    g, ids_ = hexify(g)
    doc, _ = built(g, ids_, cp_anchor, "cp1")
    link = next(h for h in item_of(doc, ids_, "getHealth")["chain"] if h["kind"] == "Link")
    assert link["bound"] == [{"ord": 0, "entryId": ids_["e_l"], "repo": "repo-a", "unit": ids_["u_cfg"], "path": "deploy/run.yaml", "line": 7, "port": 9001, "portOrigin": "compose",
                              "launch": {"launchKind": kind, **LAUNCH_FIELDS[kind]}}]


def test_a_link_with_no_lifted_or_bound_rows_carries_neither_field():
    g, ids_ = link_world(with_lifted=False, with_bound=False)
    doc, _ = built(g, ids_, cp_anchor, "cp1")
    link = next(h for h in item_of(doc, ids_, "getHealth")["chain"] if h["kind"] == "Link")
    assert "lifted" not in link and "bound" not in link


# ------------------------------------------------------------------------------------------------------------------ wiring items, the site state, depends-on

def test_a_wiring_item_is_a_config_subject_whose_link_hop_carries_exactly_that_entry_and_no_lifted():
    g, ids_ = link_world()
    doc, _ = built(g, ids_, sym_anchor, "H")
    wiring = item_of(doc, ids_, "e1")
    assert wiring["subject"] == {"type": "config", "entryId": ids_["e1"], "path": "cfg/compose.yaml", "line": 4, "keyPath": "services.api.ports[0]"}
    assert (wiring["repoId"], wiring["unitId"], wiring["distance"], wiring["status"]) == ("repo-a", ids_["u_cfg"], 1, "confirmed")
    link, entry = wiring["chain"][1], wiring["chain"][2]
    assert [h["kind"] for h in wiring["chain"]] == ["Symbol", "Link", "ConfigEntry"] and "lifted" not in link
    assert link["bound"] == [{"ord": 0, "entryId": ids_["e1"], "repo": "repo-a", "unit": ids_["u_cfg"], "path": "cfg/compose.yaml", "line": 4, "keyPath": "services.api.ports[0]",
                              "port": 8000, "portOrigin": "compose"}]
    assert (entry["edgeType"], entry["dir"], entry["state"], entry["keyPath"]) == ("BOUND_BY", "->", "end", "services.api.ports[0]")


def test_an_entry_bound_at_two_sites_is_one_wiring_item_whose_bound_is_the_lowest_site_and_ord_edge():
    g = link_world_g()
    g.site("cs_z", "getHealth", line=40, callee="apiFetch", path="ui/e.ts", unit="u_ui")
    g.via("cs_z", "lk")
    for e in g.rows.edges[("Link", "BOUND_BY", "ConfigEntry")]:
        e["p"]["ord"] = 3                      # the compose edge (site cs) has ord 3, the explicit one (site cs_z) ord 0: the lowest (siteId, ord) decides, not the lowest ord
    g.bound("lk", "e1", "cs_z", 0, port=9999, portOrigin="explicit")
    g, ids_ = hexify(g)
    doc, _ = built(g, ids_, sym_anchor, "H")
    (wiring,) = [i for i in doc["items"] if i["subject"]["type"] == "config"]
    (b,) = wiring["chain"][1]["bound"]
    lowest = min((ids_["cs"], 3, 8000, "compose"), (ids_["cs_z"], 0, 9999, "explicit"))
    assert (b["ord"], b["port"], b["portOrigin"]) == lowest[1:]


def test_a_config_anchor_enters_its_link_in_state_site_and_lists_only_the_site_it_binds():
    g, ids_ = link_world()
    doc, _ = built(g, ids_, cfg_anchor, "e1")
    (item,) = [i for i in doc["items"] if i["subject"]["type"] == "symbol"]
    assert item["subject"]["symbolId"] == ids_["getHealth"] and item["distance"] == 2
    chain = item["chain"]
    assert [h["kind"] for h in chain] == ["ConfigEntry", "Link", "CallSite", "Symbol"] and states(chain) == ["value", "site", "value", "function"]
    assert "accessPath" not in chain[1] and (chain[1]["edgeType"], chain[1]["dir"]) == ("BOUND_BY", "<-")
    assert chain[0] == {"kind": "ConfigEntry", "id": ids_["e1"], "repo": "repo-a", "unit": ids_["u_cfg"], "path": "cfg/compose.yaml", "line": 4, "keyPath": "services.api.ports[0]", "state": "value"}
    assert [x["siteId"] for x in chain[1]["lifted"]] == [ids_["cs"], ids_["cs"]] and chain[1]["bound"][0]["entryId"] == ids_["e1"]
    assert doc["interpretation"]["request"]["anchors"] == [{"type": "config", "entryId": ids_["e1"], "repoId": "repo-a", "path": "cfg/compose.yaml", "keyPath": "services.api.ports[0]"}]


def depends_world():
    return hexify(depends_world_g())


def depends_world_g(**kw):
    g = link_world_g(**kw)
    g.unit_node("u_never", root="never", lang="python", flow=False)
    for n in g.rows.nodes["Unit"]:
        if n["unitId"] == "u_never":
            n.update(buildStatus="not-mapped", failReason="manifest:never-built"), n.pop("flowAvailable")
            n["flowAvailable"] = False
    g.site("cs2", "getHealth", line=25, callee="other", path="ui/e.ts", unit="u_ui")
    g.link("lk2", to_unit="u_never", status="possible", reason="target-not-mapped")
    g.via("cs2", "lk2", key="GET /y")
    return g


def test_a_depends_on_chain_walks_the_call_site_its_link_and_the_provider_and_the_link_carries_its_sites_evidence():
    g, ids_ = depends_world()
    doc, _ = built(g, ids_, sym_anchor, "getHealth", question="depends-on")
    assert doc["interpretation"]["request"]["direction"] == "depends-on"
    provider = item_of(doc, ids_, "H")
    chain = provider["chain"]
    assert [(h["kind"], h.get("edgeType"), h.get("dir")) for h in chain] == [("Symbol", None, None), ("CallSite", "HAS_CALL", "->"), ("Link", "VIA", "->"), ("Symbol", "TO", "->")]
    assert chain[2]["candidateCount"] == 1 and [x["siteId"] for x in chain[2]["lifted"]] == [ids_["cs"], ids_["cs"]] and chain[2]["bound"][0]["entryId"] == ids_["e1"]
    assert provider["subject"] == {"type": "symbol", "symbolId": ids_["H"], "qname": "srv/app.py::health", "path": "srv/app.py", "lineStart": 10, "lineEnd": 20}


def test_a_link_into_a_never_built_unit_is_a_unit_item_possible_target_not_mapped_and_the_unit_is_listed_not_mapped():
    g, ids_ = depends_world()
    doc, _ = built(g, ids_, sym_anchor, "getHealth", question="depends-on")
    item = item_of(doc, ids_, "u_never")
    assert item["subject"] == {"type": "unit", "unitId": ids_["u_never"]}
    assert (item["status"], item["reason"], item["repoId"], item["unitId"]) == ("possible", "target-not-mapped", "repo-a", ids_["u_never"])
    last = item["chain"][-1]
    assert last == {"kind": "Unit", "id": ids_["u_never"], "repo": "repo-a", "unit": ids_["u_never"], "state": "end", "edgeType": "TO_UNIT", "dir": "->", "status": "possible",
                    "reason": "target-not-mapped"}
    (entry,) = [u for u in doc["staleness"]["units"] if u["unitId"] == ids_["u_never"]]
    assert entry == {"unitId": ids_["u_never"], "repoId": "repo-a", "rootPath": "never", "lang": "python", "buildStatus": "not-mapped", "reason": "manifest:never-built"}


def test_a_symbol_item_without_a_span_in_the_walks_reads_takes_it_from_the_snapshot_and_an_owned_region_list_is_carried():
    g, ids_ = depends_world()
    next(n for n in g.rows.nodes["Symbol"] if n["qname"] == "srv/app.py::health")["ownedRegions"] = (12, 14, 16, 18)
    doc, reader = built(g, ids_, sym_anchor, "getHealth", question="depends-on")
    assert item_of(doc, ids_, "H")["subject"]["ownedRegions"] == [12, 14, 16, 18]
    assert ("symbol_by_qname", {"repo": "repo-a", "qname": "srv/app.py::health"}) in reader.calls


# ------------------------------------------------------------------------------------------------------------------ carry-over (b): a missing anchor is a build error

def drop(g, label, prop, ident):
    """Delete a node and, as the store does, every edge on it."""
    g.rows.nodes[label] = [n for n in g.rows.nodes[label] if n[prop] != ident]
    for key in list(g.rows.edges):
        g.rows.edges[key] = [e for e in g.rows.edges[key] if ident not in (e["s"], e["t"])]


def test_the_walk_alone_returns_an_empty_walk_for_an_absent_symbol_anchor_so_the_builders_re_read_is_what_makes_it_an_error():
    g, ids_ = caller_world()
    drop(g, "Symbol", "symbolId", ids_["A"])
    assert bfs.walk(AnswerReader(g), ("Symbol", ids_["A"])).items() == []     # the shipped table: no gated row leaves a Symbol in value state, so the walk itself never raises


def test_an_anchor_the_snapshot_no_longer_holds_is_a_build_error_for_each_anchor_type_never_an_empty_answer():
    g, ids_ = caller_world()
    anchor = sym_anchor(g, ids_, "A")
    drop(g, "Symbol", "symbolId", ids_["A"])
    with pytest.raises(AB.BuildError, match="anchor"):
        AB.build_answer(AnswerReader(g), anchor)
    g, ids_ = link_world()
    for make, name, label, prop in ((cp_anchor, "cp1", "ContractPoint", "cpId"), (cfg_anchor, "e1", "ConfigEntry", "entryId")):
        anchor = make(g, ids_, name)
        g2 = copy.deepcopy(g)
        drop(g2, label, prop, ids_[name])
        with pytest.raises(AB.BuildError, match="anchor"):
            AB.build_answer(AnswerReader(g2), anchor)


@pytest.mark.parametrize("field", ["unit_id", "path", "line_start"])
def test_a_symbol_anchor_whose_facts_differ_from_the_snapshots_is_stale_and_refused(field):
    g, ids_ = caller_world()
    a = sym_anchor(g, ids_, "A")
    stale = dataclasses.replace(a, **{field: {"unit_id": ids_["u2"], "path": "other.py", "line_start": a.line_start + 1}[field]})
    with pytest.raises(AB.BuildError, match="anchor"):
        AB.build_answer(AnswerReader(g), stale)


def test_an_unusable_anchor_object_a_missing_path_or_an_unknown_question_is_refused_with_a_static_message():
    g, ids_ = caller_world()
    a = sym_anchor(g, ids_, "A")
    with pytest.raises(AB.BuildError, match="path or unit"):
        AB.build_answer(AnswerReader(g), dataclasses.replace(a, path=None))
    with pytest.raises(AB.BuildError, match="resolved symbol, contract or config"):
        AB.build_answer(AnswerReader(g), object())
    with pytest.raises(ValueError, match="unknown question"):
        AB.build_answer(AnswerReader(g), a, question="hostile\u0000" * 40)


def test_a_stamp_that_is_not_the_p1_verdict_log_pair_and_a_snapshot_that_changes_mid_build_are_refused():
    g, ids_ = caller_world()
    a = sym_anchor(g, ids_, "A")
    with pytest.raises(GraphDataError):
        AB.build_answer(AnswerReader(g, [(SNAP, 7, answer.EMPTY_VERDICT_LOG_HASH, "1.1", "1.0")]), a)

    class Swapping(AnswerReader):
        def __init__(self, g):
            super().__init__(g)
            self.reads = 0

        def _x_snapshot(self):
            self.reads += 1
            return [("f" * 32 if self.reads > 1 else SNAP, 0, answer.EMPTY_VERDICT_LOG_HASH, "1.1", "1.0")]

    with pytest.raises(AB.BuildError, match="snapshot changed"):
        AB.build_answer(Swapping(g), a)


# ------------------------------------------------------------------------------------------------------------------ carry-over (e): one test per staleness-unit source (spec2 §8, F4)

def listed(doc):
    return {u["unitId"] for u in doc["staleness"]["units"]}


def test_staleness_unit_source_1_the_items_own_unit_with_no_other_hop_in_it():
    g, ids_ = depends_world()      # depends-on from getHealth (u_ui): the provider H is the only thing in u_srv, as the last hop
    doc, _ = built(g, ids_, sym_anchor, "getHealth", question="depends-on")
    provider = item_of(doc, ids_, "H")
    assert provider["unitId"] == ids_["u_srv"] and [h["unit"] for h in provider["chain"][:-1] if "unit" in h].count(ids_["u_srv"]) == 0
    assert ids_["u_srv"] in listed(doc)


def test_staleness_unit_source_2_a_hops_unit_on_a_chain_when_no_item_is_in_that_unit():
    g, ids_ = hexify(depends_world_g(with_lifted=False, with_bound=False))      # the anchor and its call site live in u_ui; no item and no evidence entry is in u_ui
    doc, _ = built(g, ids_, sym_anchor, "getHealth", question="depends-on")
    assert all(it["unitId"] != ids_["u_ui"] for it in doc["items"]) and ids_["u_ui"] in listed(doc)
    assert not any(x["unit"] == ids_["u_ui"] for it in doc["items"] for h in it["chain"] for x in (*h.get("lifted", ()), *h.get("bound", ())))
    assert any(h.get("unit") == ids_["u_ui"] for it in doc["items"] for h in it["chain"])


def test_staleness_unit_source_3_a_lifted_entrys_unit_when_it_is_on_no_other_path():
    g, ids_ = link_world(with_bound=False)
    doc, _ = built(g, ids_, cp_anchor, "cp1")
    paths = [h.get("unit") for it in doc["items"] for h in it["chain"]] + [b["unit"] for it in doc["items"] for h in it["chain"] for b in h.get("bound", [])]
    assert ids_["u_lift"] not in paths and ids_["u_lift"] in [x["unit"] for it in doc["items"] for h in it["chain"] for x in h.get("lifted", [])]
    assert ids_["u_lift"] in listed(doc)


def test_staleness_unit_source_4_a_bound_entrys_unit_when_it_is_on_no_other_path():
    g = link_world_g(with_bound=False, with_lifted=False)
    g.node("ConfigEntry", entryId="e_launch", repoId="repo-a", unitId="u_cfg", path=".mcp.json", line=2, launchKind="command", argv0="run.sh")
    g.bound("lk", "e_launch", "cs", 0)     # a launch fact is evidence, never an item
    g, ids_ = hexify(g)
    doc, _ = built(g, ids_, cp_anchor, "cp1")
    assert all(it["unitId"] != ids_["u_cfg"] for it in doc["items"]) and all(h.get("unit") != ids_["u_cfg"] for it in doc["items"] for h in it["chain"])
    assert [b["unit"] for it in doc["items"] for h in it["chain"] for b in h.get("bound", [])] == [ids_["u_cfg"]]
    assert ids_["u_cfg"] in listed(doc)


def fold_world():
    return hexify(fold_world_g())


def fold_world_g():
    """Depends-on from ``F`` (u_f, javascript): its site ``s`` resolves to ``T`` (u_t) and name-matches ``X`` (u_x, a unit nothing else is in): ``X`` is another candidate of a resolved
    site, folded when the site's language folds (``javascript`` does)."""
    g = G(unit="u_f", flow=True, lang="javascript")
    g.unit_node("u_t", root="t", lang="javascript"), g.unit_node("u_x", root="x", lang="javascript")
    g.sym("F", "f.ts::F", path="f.ts", start=1, end=9)
    g.sym("T", "t/t.ts::T", path="t/t.ts", start=2, end=8, unit="u_t")
    g.sym("X", "x/x.ts::X", path="x/x.ts", start=3, end=7, unit="u_x")
    g.site("s", "F", line=4, callee="f", path="f.ts")
    g.calls("s", "T")
    g.calls_name("s", "T", "f"), g.calls_name("s", "X", "f")
    return g


def test_staleness_unit_source_5_a_unit_reached_only_by_a_folded_group_member():
    g, ids_ = fold_world()
    doc, _ = built(g, ids_, sym_anchor, "F", question="depends-on")
    assert [i["subject"]["symbolId"] for i in doc["items"]] == [ids_["T"]]
    (grp,) = doc["groups"]
    assert (grp["kind"], grp["key"], grp["reason"], grp["count"]) == ("other-possible-callees", ids_["s"], "resolution-unverified", 1)
    (member,) = grp["members"]
    assert member["subject"]["symbolId"] == ids_["X"] and member["unitId"] == ids_["u_x"] and member["status"] == "possible" and member["reason"] == "resolution-unverified"
    assert ids_["u_x"] in listed(doc) and all(it["unitId"] != ids_["u_x"] for it in doc["items"])


def test_staleness_unit_source_2_an_interior_hop_in_a_unit_no_item_anchor_hop_or_evidence_entry_is_in():
    """Isolates an *interior* hop: the last hop's unit is source 1 (V7), the anchor's is hop 0, so only a hop strictly between them can be the sole witness of its unit."""
    g = data_reached_sites_g()
    g.unit_node("u_y", flow=True, root="y", lang="python")
    for n in g.rows.nodes["CallSite"]:
        if n["callSiteId"] == "cG":
            n["unitId"] = "u_y"          # the interior call site (and its owner Y) sit in a unit of their own
    for n in g.rows.nodes["Symbol"]:
        if n["symbolId"] == "Y":
            n["unitId"] = "u_y"
    g, ids_ = hexify(g)
    doc, _ = built(g, ids_, sym_anchor, "F", question="depends-on")
    chain = item_of(doc, ids_, "T")["chain"]
    assert [h["unit"] for h in chain[1:3]] == [ids_["u1"], ids_["u_y"]]
    assert all(it["unitId"] != ids_["u_y"] and chain_unit_ne(it, ids_["u_y"]) for it in doc["items"])
    assert ids_["u_y"] in listed(doc)


def chain_unit_ne(item, unit):
    """True when only an interior hop (neither the first nor the last) of ``item`` is in ``unit``, and no evidence entry is."""
    hops = item["chain"]
    return unit not in (hops[0].get("unit"), hops[-1].get("unit")) and not any(x["unit"] == unit for h in hops for x in (*h.get("lifted", ()), *h.get("bound", ())))


def member_link_world_g():
    """Depends-on from ``F`` (javascript, flag-false). ``F`` calls ``P`` directly (``c_direct``) and ``T`` directly (``c_dT``), and reaches ``P`` again through link ``lk`` (``c_via``),
    which carries lifted evidence in ``u_lift`` and a bound entry in ``u_cfg``. ``P``'s site ``c_in_P`` resolves to ``T`` and name-matches ``X`` (``u_x``, javascript folds): ``X`` is a folded
    group member whose chain ``F, c_via, lk, P, c_in_P, X`` holds a ``Link`` hop that is on no listed item's chain (``P`` and ``T`` are listed on their one-hop direct chains)."""
    g = G(unit="u_f", flow=False, lang="javascript")
    for u, root, lang in (("u_p", "p", "javascript"), ("u_x", "x", "javascript"), ("u_lift", "lib", "javascript"), ("u_cfg", "cfg", "yaml")):
        g.unit_node(u, root=root, lang=lang, flow=False)
    g.sym("F", "f.js::F", path="f.js", start=1, end=9)
    g.sym("P", "p/p.js::P", path="p/p.js", start=1, end=9, unit="u_p")
    g.sym("T", "p/t.js::T", path="p/t.js", start=1, end=9, unit="u_p")
    g.sym("X", "x/x.js::X", path="x/x.js", start=1, end=9, unit="u_x")
    g.sym("W", "lib/w.js::W", path="lib/w.js", start=1, end=3, unit="u_lift")
    g.site("c_direct", "F", line=2, callee="p", path="f.js"), g.calls("c_direct", "P")
    g.site("c_via", "F", line=3, callee="apiFetch", path="f.js")
    g.cp("cp1", key="GET /x"), g.provides("P", "cp1"), g.link("lk", provider="P", cp="cp1"), g.via("c_via", "lk")
    g.edge("Link", "LIFTED_VIA", "Symbol", "lk", "W", siteId="c_via", ord=0, line=2, urlPath="p0", verbPath="p1")
    g.entry("e1", key_path="a.b", path="cfg/c.yaml", line=1, unit="u_cfg"), g.bound("lk", "e1", "c_via", 0)
    g.site("c_dT", "F", line=5, callee="t", path="f.js"), g.calls("c_dT", "T")
    g.site("c_in_P", "P", line=4, callee="f", path="p/p.js", unit="u_p"), g.calls("c_in_P", "T")
    g.calls_name("c_in_P", "T", "f"), g.calls_name("c_in_P", "X", "f")
    return g


def test_a_folded_members_link_hop_carries_its_evidence_and_its_units_are_stale_listed_though_no_listed_item_has_that_link():
    g, ids_ = hexify(member_link_world_g())
    doc, _ = built(g, ids_, sym_anchor, "F", question="depends-on")
    assert sorted(i["subject"]["symbolId"] for i in doc["items"]) == sorted([ids_["P"], ids_["T"]])
    assert all(h["kind"] != "Link" for it in doc["items"] for h in it["chain"])           # the cheaper direct chains bypass the link
    (grp,) = doc["groups"]
    assert grp["key"] == ids_["c_in_P"]
    (member,) = grp["members"]
    assert member["subject"]["symbolId"] == ids_["X"]
    link = next(h for h in member["chain"] if h["kind"] == "Link")
    via = next(h for h in member["chain"] if h.get("edgeType") == "VIA")
    assert via["candidateCount"] == 1
    assert [x["unit"] for x in link["lifted"]] == [ids_["u_lift"]] and [x["unit"] for x in link["bound"]] == [ids_["u_cfg"]]
    assert {ids_["u_lift"], ids_["u_cfg"]} <= listed(doc)


# ------------------------------------------------------------------------------------------------------------------ carry-over (c): the fold's inputs

REPORT = "docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md"
LANG_KEY = {"Python": "python", "TS/JS": "javascript", "Go": "go", "Java": "java", "C#": "csharp", "C++": "cpp", "Kotlin": "kotlin"}


def report_rows():
    """The report's §a1.2 table, row by row: ``{language key: cells}``."""
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / REPORT).read_text(encoding="utf-8")
    section = text.split("### a1.2", 1)[1].split("### a1.3", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| ") and cells[0] in LANG_KEY:
            rows[LANG_KEY[cells[0]]] = cells
    return rows


def test_the_report_has_the_seven_languages_and_the_columns_the_transcription_reads():
    rows = report_rows()
    assert set(rows) == set(LANG_KEY.values()) and all(len(c) == 9 for c in rows.values())


def test_the_committed_measured_pairs_are_the_reports_headline_p_off_on_cells():
    rows = report_rows()
    for lang, cells in rows.items():
        found = tuple((int(a), int(b)) for a, b in re.findall(r"(\d+)/(\d+)", cells[5]))
        if "same" in cells[5]:
            found = found * 2
        assert AB.FOLD_MEASURED.get(lang, ()) == found, lang
    assert "kotlin" not in AB.FOLD_MEASURED


def test_the_derived_fold_set_equals_the_reports_fold_or_list_column():
    from ccg.engine import group
    rows = report_rows()
    expected = {lang for lang, cells in rows.items() if cells[7].startswith("fold")}
    assert group.fold_languages(AB.FOLD_MEASURED) == frozenset(expected) == frozenset({"javascript", "go"})
    assert {lang for lang, cells in rows.items() if cells[7].startswith("list")} == {"python", "java", "csharp", "cpp", "kotlin"}


def test_the_builder_folds_a_javascript_resolved_sites_other_candidates_and_lists_a_python_ones():
    g, ids_ = fold_world()
    doc, _ = built(g, ids_, sym_anchor, "F", question="depends-on")
    assert len(doc["groups"]) == 1
    g, ids_ = fold_world()
    for n in g.rows.nodes["Unit"]:
        n["language"] = "python"
    doc, _ = built(g, ids_, sym_anchor, "F", question="depends-on")
    assert doc["groups"] == [] and {i["subject"]["symbolId"] for i in doc["items"]} == {ids_["T"], ids_["X"]}
    x = item_of(doc, ids_, "X")
    assert (x["status"], x["reason"]) == ("possible", "resolution-unverified")


def test_measured_replaces_the_whole_mapping_and_an_empty_one_folds_nothing():
    g, ids_ = fold_world()
    doc, _ = built(g, ids_, sym_anchor, "F", question="depends-on", measured={})
    assert doc["groups"] == [] and len(doc["items"]) == 2
    doc, _ = built(g, ids_, sym_anchor, "F", question="depends-on", measured={"javascript": [(56, 60)]})
    assert doc["groups"] == []
    doc, _ = built(g, ids_, sym_anchor, "F", question="depends-on", measured={"javascript": [(57, 60)]})
    assert len(doc["groups"]) == 1


# ------------------------------------------------------------------------------------------------------------------ (3) determinism

def evidence_world():
    """One consumer site whose link carries two lifted entries and three bound ones (a keyPath entry and two launch facts): the arrays whose order the builder must not leave to the reader."""
    g = link_world_g()
    g.node("ConfigEntry", entryId="e_l1", repoId="repo-a", unitId="u_cfg", path=".mcp.json", line=2, launchKind="command", argv0="run.sh")
    g.node("ConfigEntry", entryId="e_l2", repoId="repo-a", unitId="u_cfg", path="Dockerfile", line=3, launchKind="port", protocol="http", port=8000)
    g.bound("lk", "e_l1", "cs", 1), g.bound("lk", "e_l2", "cs", 2, port=8000, portOrigin="explicit")
    return hexify(g)


def worlds():
    return {"caller": caller_world(), "flagfalse": flag_false_two_callers(), "link": link_world(sibling_links=1), "fold": fold_world(), "evidence": evidence_world()}


def request_of(name, g, ids_):
    if name in ("link", "evidence"):
        return cp_anchor(g, ids_, "cp1"), "impact"
    if name == "fold":
        return sym_anchor(g, ids_, "F"), "depends-on"
    return sym_anchor(g, ids_, "A"), "impact"


@pytest.mark.parametrize("name", ["caller", "flagfalse", "link", "fold", "evidence"])
def test_the_same_request_twice_gives_byte_identical_answers_and_facts(name):
    g, ids_ = worlds()[name]
    a, q = request_of(name, g, ids_)
    one, two = AB.build_answer(AnswerReader(g), a, question=q), AB.build_answer(AnswerReader(g), a, question=q)
    assert answer.dumps(one) == answer.dumps(two) and answer.facts_bytes(one) == answer.facts_bytes(two)
    assert one["items"] or one["groups"]


@pytest.mark.parametrize("name", ["caller", "flagfalse", "link", "fold", "evidence"])
@pytest.mark.parametrize("seed,reverse", [(s, r) for s in range(4) for r in (False, True)])
def test_every_reply_shuffled_gives_the_same_bytes(name, seed, reverse):
    g, ids_ = worlds()[name]
    a, q = request_of(name, g, ids_)
    base = AB.build_answer(AnswerReader(g), a, question=q)
    got = AB.build_answer(Shuffled(AnswerReader(g), seed, reverse), a, question=q)
    assert got == base and answer.facts_bytes(got) == answer.facts_bytes(base)      # the document itself, not only its canonical bytes: ``dumps`` would re-sort a builder that did not


@pytest.mark.parametrize("name", ["caller", "flagfalse", "link", "fold", "evidence"])
def test_the_insertion_order_of_the_stores_rows_changes_nothing(name):
    import random
    g, ids_ = worlds()[name]
    a, q = request_of(name, g, ids_)
    base = AB.build_answer(AnswerReader(g), a, question=q)
    for seed in range(4):
        rng = random.Random(seed)
        g2 = copy.deepcopy(g)
        for lst in g2.rows.nodes.values():
            rng.shuffle(lst)
        for lst in g2.rows.edges.values():
            rng.shuffle(lst)
        keys = list(g2.rows.edges)
        rng.shuffle(keys)
        g2.rows.edges = {k: g2.rows.edges[k] for k in keys}
        assert AB.build_answer(AnswerReader(g2), a, question=q) == base


def test_a_rebuild_from_the_same_commits_has_other_build_times_and_the_same_facts_bytes():
    g, ids_ = caller_world()
    a = sym_anchor(g, ids_, "A")
    first = AB.build_answer(AnswerReader(g), a)
    for n in g.rows.nodes["Unit"]:
        n["builtAt"] = T2
    second = AB.build_answer(AnswerReader(g), a)
    assert first["stamp"] == second["stamp"] and answer.dumps(first) != answer.dumps(second)
    assert answer.facts_bytes(first) == answer.facts_bytes(second)
    assert {u["builtAt"] for u in first["staleness"]["units"]} == {T1} and {u["builtAt"] for u in second["staleness"]["units"]} == {T2}


def test_a_different_commit_is_a_different_fact():
    g, ids_ = caller_world()
    a = sym_anchor(g, ids_, "A")
    first = AB.build_answer(AnswerReader(g), a)
    g.rows.nodes["Unit"][0]["builtCommit"] = "b" * 40
    assert answer.facts_bytes(AB.build_answer(AnswerReader(g), a)) != answer.facts_bytes(first)


def test_the_document_holds_no_time_but_a_units_own_built_and_failed_times():
    g, ids_ = caller_world()
    for n in g.rows.nodes["Unit"]:
        if n["rootPath"] == "svc":
            n.update(buildStatus="stale-kept", failedAt=T2, failReason="extract:failed")
    doc = AB.build_answer(AnswerReader(g), sym_anchor(g, ids_, "A"))
    stamps = []

    def walk(v, path=""):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]")
        elif isinstance(v, str) and re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", v):
            stamps.append(path)
    walk(doc)
    assert sorted(stamps) == sorted(f".staleness.units[{i}].{k}" for i, u in enumerate(doc["staleness"]["units"]) for k in ("builtAt", "failedAt") if k in u) and len(stamps) == 3
    again = copy.deepcopy(doc)
    for u in again["staleness"]["units"]:
        u.pop("failedAt", None)
        u["builtAt"] = "2030-01-01T00:00:00Z"
    assert answer.facts_bytes(again) == answer.facts_bytes(doc)


# ------------------------------------------------------------------------------------------------------------------ (6) canonicalization, and hostile strings

def two_callers(path_a, path_b, root="svc"):
    g = G(unit="u1", flow=True, lang="python")
    g.unit_node("u2", flow=True, root=root, lang="python")
    g.sym("A", "m.py::A", path="m.py")
    for name, path, line in (("C1", path_a, 10), ("C2", path_b, 10)):
        g.sym(name, f"{path}::{name}", path=path, start=line, end=line + 9, unit="u2")
        g.site(f"c{name}", name, line=line + 1, callee="A", path=path, unit="u2")
        g.calls(f"c{name}", "A")
    return hexify(g)


NFD_E = "é.rb"
NFC_E = "é.rb"


def test_the_raw_code_point_order_and_the_canonical_nfc_order_of_these_two_paths_are_opposite():
    assert sorted(["z.rb", NFD_E]) == [NFD_E, "z.rb"]
    assert sorted(["z.rb", NFC_E]) == ["z.rb", NFC_E]


def test_the_document_is_in_the_canonical_nfc_order_not_the_engines_raw_code_point_order():
    g, ids_ = two_callers("z.rb", NFD_E)
    doc, _ = built(g, ids_, sym_anchor, "A")
    assert [it["subject"]["path"] for it in doc["items"]] == ["z.rb", NFC_E]
    assert doc == answer.canonicalize(doc)


def test_every_string_is_nfc_and_every_identity_string_is_stripped_before_the_document_is_validated():
    g, ids_ = two_callers(" z.rb ", NFD_E, root=" svc ")
    doc, _ = built(g, ids_, sym_anchor, "A")
    paths = {it["subject"]["path"] for it in doc["items"]}
    assert paths == {"z.rb", NFC_E}
    assert {u["rootPath"] for u in doc["staleness"]["units"]} == {"", "svc"}
    import unicodedata
    for it in doc["items"]:
        for h in (it["subject"], *it["chain"]):
            for k in ("qname", "path", "calleeName"):
                assert k not in h or (h[k] == h[k].strip() and unicodedata.normalize("NFC", h[k]) == h[k])
    assert {it["subject"]["qname"] for it in doc["items"]} == {"z.rb ::C1", f"{NFC_E}::C2"}
    answer.validate(doc)


def test_clean_is_nfc_then_strip_on_identity_keys_only_and_leaves_numbers_and_other_strings_but_nfc():
    got = AB.clean({"path": " é ", "reason": " é ", "n": 3, "paths": [" á "], "x": [{"qname": "é "}]})
    assert got == {"path": "é", "reason": " é ", "n": 3, "paths": ["á"], "x": [{"qname": "é"}]}


HOSTILE = ["", "\x00", "a\x00b", "\ud800", "x\udc00y", "‮", "é", " pad ", "<script>alert(1)</script>", "'; DROP", "\U0001F600", "a" * 5000]


@pytest.mark.parametrize("field", ["qname", "path", "callee", "root"])
@pytest.mark.parametrize("value", HOSTILE)
def test_a_hostile_graph_string_either_validates_and_round_trips_or_raises_a_static_message_never_a_document_the_validator_rejects(field, value):
    g = G(unit="u1", flow=True, lang="python")
    g.unit_node("u2", flow=True, root="svc", lang="python")
    g.sym("A", "m.rb::A", path="m.rb")
    g.sym("C", "c.rb::C", path="c.rb", start=10, end=19, unit="u2")
    g.site("cC", "C", line=12, callee="A", path="c.rb", unit="u2")
    g.calls("cC", "A")
    g, ids_ = hexify(g)
    for n in g.rows.nodes["Symbol"]:
        if n["qname"] == "c.rb::C":
            n.update({"qname": {"qname": value}.get(field, n["qname"]), "path": {"path": value}.get(field, n["path"])})
    for n in g.rows.nodes["CallSite"]:
        if field == "path":
            n["path"] = value
        if field == "callee":
            n["calleeName"] = value
    for n in g.rows.nodes["Unit"]:
        if n["unitId"] == ids_["u2"] and field == "root":
            n["rootPath"] = value
    try:
        doc = AB.build_answer(AnswerReader(g), sym_anchor(g, ids_, "A"))
    except GraphDataError as exc:
        assert (value == "" or value not in str(exc)) and "\x00" not in str(exc) and not any(0xD800 <= ord(c) <= 0xDFFF for c in str(exc))
        return
    assert answer.loads(answer.dumps(doc)) == doc
    answer.validate(doc)


@pytest.mark.parametrize("field", ["keyPath", "entry path", "lifted urlPath", "wrapper qname", "launch argv0"])
@pytest.mark.parametrize("value", HOSTILE)
def test_hostile_strings_in_the_evidence_of_a_link_validate_or_raise_a_static_message(field, value):
    g = link_world_g()
    g.node("ConfigEntry", entryId="e_launch", repoId="repo-a", unitId="u_cfg", path=".mcp.json", line=2, launchKind="command", argv0="run.sh")
    g.bound("lk", "e_launch", "cs", 1)
    g, ids_ = hexify(g)
    for n in g.rows.nodes["ConfigEntry"]:
        if n["entryId"] == ids_["e1"]:
            n.update({"keyPath": value} if field == "keyPath" else {"path": value} if field == "entry path" else {})
        if n["entryId"] == ids_["e_launch"] and field == "launch argv0":
            n["argv0"] = value
    for n in g.rows.nodes["Symbol"]:
        if n["qname"] == "lib/w.ts::wrap" and field == "wrapper qname":
            n["qname"] = value
    if field == "lifted urlPath":
        for e in g.rows.edges[("Link", "LIFTED_VIA", "CallSite")]:
            e["p"]["urlPath"] = value
    try:
        doc = AB.build_answer(AnswerReader(g), cp_anchor(g, ids_, "cp1"))
    except GraphDataError as exc:
        assert value == "" or value not in str(exc)
        assert "\x00" not in str(exc) and not any(0xD800 <= ord(c) <= 0xDFFF for c in str(exc))
        return
    assert answer.loads(answer.dumps(doc)) == doc


# ------------------------------------------------------------------------------------------------------------------ a call site reached by DATA / CONTROLLED_BY has no calleeName in the walk's facts

def data_reached_sites_g():
    """Flag-true. ``F`` holds ``c0``, whose argument ``a0`` takes the result of ``cG`` (a call of ``T`` that belongs to ``Y``): depends-on from ``F`` walks ``HAS_CALL`` ``c0``,
    ``DATA ->`` ``cG`` and ``CALLS ->`` ``T``; impact from ``A`` walks ``CALLS <-`` ``cB`` (result ``r``), ``DATA <-`` ``cX`` (which uses it), ``HAS_CALL <-`` ``X``. ``cK`` is under the
    guard ``kB`` of ``B``: ``CONTROLLED_BY <-`` reaches it from the condition."""
    g = G(unit="u1", flow=True, lang="python")
    for n, line in (("F", 1), ("Y", 11), ("T", 21), ("A", 31), ("B", 41), ("X", 51)):
        g.sym(n, f"m.py::{n}", path="m.py", start=line, end=line + 8)
    g.site("c0", "F", line=2, callee="zero"), g.site("cG", "Y", line=12, callee="gee"), g.calls("cG", "T")
    g.data("CallSite", "c0", "CallSite", "cG", "r", "a0")
    g.site("cB", "B", line=42, callee="A"), g.calls("cB", "A"), g.site("cX", "X", line=52, callee="use")
    g.data("CallSite", "cX", "CallSite", "cB", "r", "a0")
    return g


def test_a_call_site_reached_by_data_carries_the_callee_name_the_walks_reads_do_not_give_it_in_depends_on():
    g, ids_ = hexify(data_reached_sites_g())
    doc, reader = built(g, ids_, sym_anchor, "F", question="depends-on")
    chain = item_of(doc, ids_, "T")["chain"]
    assert [(h["kind"], h.get("edgeType")) for h in chain] == [("Symbol", None), ("CallSite", "HAS_CALL"), ("CallSite", "DATA"), ("Symbol", "CALLS")]
    assert [h["calleeName"] for h in chain if h["kind"] == "CallSite"] == ["zero", "gee"]
    assert any(n == "owner_of:CallSite" for n, _ in reader.calls) and any(n == "has_out:CallSite" for n, _ in reader.calls)


def test_a_call_site_reached_by_data_carries_its_callee_name_in_impact_too_and_no_read_is_made_when_every_site_has_a_name():
    g, ids_ = hexify(data_reached_sites_g())
    doc, _ = built(g, ids_, sym_anchor, "A")
    chain = item_of(doc, ids_, "X")["chain"]
    assert [(h["kind"], h.get("edgeType"), h.get("calleeName")) for h in chain if h["kind"] == "CallSite"] == [("CallSite", "CALLS", "A"), ("CallSite", "DATA", "use")]
    g, ids_ = caller_world()
    _, reader = built(g, ids_, sym_anchor, "A")
    assert [n for n, _ in reader.calls if n == "has_out:CallSite"] == []


# ------------------------------------------------------------------------------------------------------------------ the bound, and a span read that must match by id

def test_the_depth_bound_cuts_are_counted_into_unexplored_and_the_item_beyond_is_not_listed():
    g = flag_false_two_callers_g()
    g.sym("E", "m.py::E", path="m.py", start=31, end=39), g.site("cE", "E", line=32, callee="D"), g.calls("cE", "D")
    g, ids_ = hexify(g)
    doc, _ = built(g, ids_, sym_anchor, "A")
    assert doc["depthBound"] == {"bound": 2, "unexplored": 1}
    assert {subject_id(i) for i in doc["items"]} == {ids_["B"], ids_["D"]}
    assert max(i["distance"] for i in doc["items"]) == 2


def test_the_span_read_takes_the_row_of_the_symbol_by_id_when_other_symbols_share_the_qname_and_the_repository():
    g = depends_world_g()
    qname = "srv/app.py::health"
    for path, end in (("srv/other.ts", 91), ("srv/other.go", 92), ("srv/other.sh", 93)):
        g.sym(f"decoy{end}", qname, path=path, start=1, end=end, unit="u_srv")
    g, ids_ = hexify(g)
    assert max(ids_[f"decoy{e}"] for e in (91, 92, 93)) > ids_["H"], "the fixture must put a decoy after the real row in the read's order"
    doc, _ = built(g, ids_, sym_anchor, "getHealth", question="depends-on")
    assert item_of(doc, ids_, "H")["subject"]["lineEnd"] == 20


# ------------------------------------------------------------------------------------------------------------------ the anchor re-read matches by id and by every fact the anchor carries

def test_a_symbol_anchor_builds_when_other_symbols_share_its_qname_and_repository_the_row_is_matched_by_id():
    g = caller_world_g()
    for path in ("other.ts", "other.go"):
        g.sym(f"decoy-{path}", "m.py::A", path=path, start=1, end=2)
    g, ids_ = hexify(g)
    doc, _ = built(g, ids_, sym_anchor, "A")
    assert [subject_id(i) for i in doc["items"]] == [ids_["C"]]


def test_a_contract_anchor_is_matched_by_id_and_kind_when_another_kind_shares_its_key_and_a_stale_kind_is_refused():
    g = link_world_g()
    g.node("ContractPoint", cpId="cp_decoy", kind="topic", keyNorm="GET /x")
    g, ids_ = hexify(g)
    a = cp_anchor(g, ids_, "cp1")
    assert [subject_id(i) for i in AB.build_answer(AnswerReader(g), a)["items"] if i["subject"]["type"] == "symbol"] == [ids_["getHealth"]]
    with pytest.raises(AB.BuildError, match="anchor"):
        AB.build_answer(AnswerReader(g), dataclasses.replace(a, kind="topic"))
    with pytest.raises(AB.BuildError, match="anchor"):
        AB.build_answer(AnswerReader(g), dataclasses.replace(a, cp_id=ids_["cp_decoy"]))


@pytest.mark.parametrize("change", [{"line": 99}, {"key_path": "services.other"}, {"unit_id": "f" * 32}, {"path": "cfg/other.yaml"}])
def test_a_config_anchor_whose_facts_differ_from_the_snapshots_is_refused(change):
    g, ids_ = link_world()
    a = cfg_anchor(g, ids_, "e1")
    assert AB.build_answer(AnswerReader(g), a)["items"]
    with pytest.raises(AB.BuildError, match="anchor"):
        AB.build_answer(AnswerReader(g), dataclasses.replace(a, **change))


def test_a_config_anchor_that_is_a_launch_entry_is_refused_a_launch_fact_is_never_an_anchor():
    g = link_world_g()
    g.node("ConfigEntry", entryId="e_both", repoId="repo-a", unitId="u_cfg", path="cfg/l.json", line=2, keyPath="a.b", launchKind="command")
    g, ids_ = hexify(g)
    with pytest.raises(AB.BuildError, match="anchor"):
        AB.build_answer(AnswerReader(g), cfg_anchor(g, ids_, "e_both"))


def test_a_wiring_items_bound_carries_the_pack_default_port_evidence_of_its_edge():
    g = link_world_g(with_bound=False)
    g.entry("e2", key_path="server.port", path="cfg/app.yaml", line=7, unit="u_cfg")
    g.bound("lk", "e2", "cs", 0, port=8123, portOrigin="pack-default", packId="fastapi")
    g, ids_ = hexify(g)
    doc, _ = built(g, ids_, sym_anchor, "H")
    (b,) = item_of(doc, ids_, "e2")["chain"][1]["bound"]
    assert (b["entryId"], b["ord"], b["port"], b["portOrigin"], b["packId"]) == (ids_["e2"], 0, 8123, "pack-default", "fastapi")


# ------------------------------------------------------------------------------------------------------------------ per-kind hop fields: ContractPoint, Cond

def test_a_contract_anchor_is_hop_zero_with_its_kind_and_key_in_state_contract_and_the_links_ride_the_contract_state():
    g, ids_ = link_world()
    doc, _ = built(g, ids_, cp_anchor, "cp1")
    chain = item_of(doc, ids_, "getHealth")["chain"]
    assert chain[0] == {"kind": "ContractPoint", "id": ids_["cp1"], "contractKind": "http", "keyNorm": "GET /x", "state": "contract"}
    assert [(h["kind"], h["state"], h.get("edgeType"), h.get("dir")) for h in chain[1:]] == [
        ("Link", "contract", "ON", "<-"), ("CallSite", "contract", "VIA", "<-"), ("Symbol", "end", "HAS_CALL", "<-")]
    assert doc["interpretation"]["request"]["anchors"] == [{"type": "contract", "cpId": ids_["cp1"], "kind": "http", "keyNorm": "GET /x"}]


def cond_world_g():
    """Flag-true: ``F`` holds the condition ``k`` (code ``if x``) and a second one with no code; ``k`` feeds the call site ``cG`` of ``Y`` (``DATA``), which calls ``T``."""
    g = G(unit="u1", flow=True, lang="python")
    for n, line in (("F", 1), ("Y", 21), ("T", 31)):
        g.sym(n, f"m.py::{n}", path="m.py", start=line, end=line + 8)
    g.cond("k", "F", line=3, code="if x"), g.site("cG", "Y", line=22, callee="gee"), g.calls("cG", "T")
    g.data("Cond", "k", "CallSite", "cG", "", "r")
    return g


def test_a_cond_hop_carries_its_code_when_the_graph_has_one():
    g, ids_ = hexify(cond_world_g())
    doc, _ = built(g, ids_, sym_anchor, "F", question="depends-on")
    chain = item_of(doc, ids_, "T")["chain"]
    cond = next(h for h in chain if h["kind"] == "Cond")
    assert cond == {"kind": "Cond", "id": ids_["k"], "repo": "repo-a", "unit": ids_["u1"], "path": "m.py", "line": 3, "code": "if x", "state": "value", "edgeType": "HAS_COND",
                    "dir": "->", "status": "confirmed"}


def test_a_cond_hop_with_no_code_carries_none_and_the_document_still_validates():
    g = cond_world_g()
    for n in g.rows.nodes["Cond"]:
        n.pop("code")
    g, ids_ = hexify(g)
    doc, _ = built(g, ids_, sym_anchor, "F", question="depends-on")
    assert "code" not in next(h for h in item_of(doc, ids_, "T")["chain"] if h["kind"] == "Cond")


@pytest.mark.parametrize("world,make,anchor,question", [("link", link_world, ("cp_anchor", "cp1"), "impact"), ("depends", depends_world, ("sym_anchor", "getHealth"), "depends-on"),
                                                         ("data", lambda: hexify(data_reached_sites_g()), ("sym_anchor", "F"), "depends-on"), ("fold", fold_world, ("sym_anchor", "F"), "depends-on")])
def test_the_recipe_lists_every_text_the_whole_build_ran_the_evidence_and_the_span_and_name_reads_included(world, make, anchor, question):
    g, ids_ = make()
    doc, reader = built(g, ids_, globals()[anchor[0]], anchor[1], question=question)
    assert doc["recipe"]["queries"] == sorted(reader.recipe_queries(), key=lambda q: q["name"])
    names = {q["name"] for q in doc["recipe"]["queries"]}
    assert {"snapshot", "units"} <= names
    if world == "link":
        assert {"lifted_of", "bound_out", "candidate_count"} <= names
    if world == "data":
        assert {"owner_of:CallSite", "has_out:CallSite"} <= names


def test_an_anchor_in_the_snapshot_that_nothing_reaches_is_a_valid_empty_answer_with_no_units_listed():
    g = G(unit="u1", flow=True)
    g.sym("A", "m.py::A", path="m.py")
    g, ids_ = hexify(g)
    doc, _ = built(g, ids_, sym_anchor, "A")
    assert (doc["items"], doc["groups"], doc["staleness"]) == ([], [], {"units": []})
    assert doc["depthBound"] == {"bound": 2, "unexplored": 0} and doc["stamp"]["snapshotId"] == SNAP
    answer.validate(doc)


# ------------------------------------------------------------------------------------------------------------------ malformed rows of the reads the builder adds

class Corrupt(AnswerReader):
    """``AnswerReader`` whose reply to one read is rewritten by ``fn(rows, params)`` (a short row, a wrong type, a bool); ``call`` (0-based) restricts it to one call of that read: the
    builder's own read comes after the walk's, which asks some of the same texts and would raise first."""

    def __init__(self, g, name, fn, call=None):
        super().__init__(g)
        self.name, self.fn, self.call, self.seen = name, fn, call, 0

    def read(self, name, params=None):
        rows = super().read(name, params)
        if name != self.name:
            return rows
        self.seen += 1
        return self.fn(rows, params) if self.call is None or self.seen - 1 == self.call else rows


SHORT = lambda rows, p: [r[:-1] for r in rows]                                   # noqa: E731
BOOL_COL0 = lambda rows, p: [(True, *r[1:]) for r in rows]                       # noqa: E731
BOOL_COL2 = lambda rows, p: [(*r[:2], True, *r[3:]) for r in rows]              # noqa: E731
NOT_STR = lambda rows, p: [(r[0], 5, *r[2:]) for r in rows]                      # noqa: E731
EMPTY_ID = lambda rows, p: [(r[0], "", *r[2:]) for r in rows]                    # noqa: E731
SPAN_ONLY = lambda rows, p: [r[:-1] for r in rows] if p.get("qname") == "srv/app.py::health" else rows   # noqa: E731
LAST = "last"


@pytest.mark.parametrize("world,read,fn,anchor,question", [
    ("link", "lifted_of", SHORT, ("cp_anchor", "cp1"), "impact"), ("link", "lifted_of", BOOL_COL0, ("cp_anchor", "cp1"), "impact"), ("link", "lifted_of", EMPTY_ID, ("cp_anchor", "cp1"), "impact"),
    ("link", "lifted_of", BOOL_COL2, ("cp_anchor", "cp1"), "impact"), ("link", "bound_out", BOOL_COL2, ("cp_anchor", "cp1"), "impact"),
    ("link", "bound_out", SHORT, ("cp_anchor", "cp1"), "impact"), ("link", "bound_out", NOT_STR, ("cp_anchor", "cp1"), "impact"),
    ("depends", "symbol_by_qname", SPAN_ONLY, ("sym_anchor", "getHealth"), "depends-on"),
    ("data", "owner_of:CallSite", SHORT, ("sym_anchor", "F"), "depends-on"), ("data", "has_out:CallSite", SHORT, ("sym_anchor", "F"), "depends-on"),
    ("data", "owner_of:CallSite", NOT_STR, ("sym_anchor", "F"), "depends-on"), ("data", "has_out:CallSite", NOT_STR, ("sym_anchor", "F"), "depends-on"),
])
def test_a_malformed_row_of_a_read_the_builder_adds_is_a_graph_data_error_that_names_no_value(world, read, fn, anchor, question):
    g, ids_ = {"link": link_world, "depends": depends_world, "data": lambda: hexify(data_reached_sites_g())}[world]()
    a = globals()[anchor[0]](g, ids_, anchor[1])
    clean_reader = AnswerReader(g)
    assert AB.build_answer(clean_reader, a, question=question)["items"]
    last = sum(1 for c in clean_reader.calls if c[0] == read) - 1 if world == "data" else None     # the data world's reads are also the walk's: only the builder's own, the last, is corrupted
    with pytest.raises(GraphDataError, match="malformed row") as exc:
        AB.build_answer(Corrupt(g, read, fn, last), a, question=question)
    assert not any(v in str(exc.value) for v in ids_.values())
