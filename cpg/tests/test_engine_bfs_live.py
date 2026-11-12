"""P1-22 the walk, LIVE on a throwaway instance only (``infra/ccg-falkordb.sh --throwaway up``; ``-m live``).

Needs ``CCG_FALKORDB_PORT`` (the throwaway's port; 6379 and 6390 are refused) and ``CCG_CONFIG_DIR`` (a scratch dir holding the throwaway's
``falkordb-admin.pass``); ``CCG_REQUIRE_LIVE=1`` turns a missing instance into a failure. Loads the union of the unit fixtures (``engine_walk_helpers.union``) through
``ccg.store.loader.load`` as ``ccg_build_<s>``, renames the build key to the live key as admin, and reads it as ``ccg_engine_<s>`` only (``CypherReader``).
The oracle is the unit tier's own reader: every walk is run over ``CypherReader`` and over ``MemoryReader`` on the same rows and must agree on everything the walk
produces (states, best chains, items, cut edges, facts), so the in-memory reader the unit tests rest on is proved to answer as the store does.
Contract: ``docs/plans/connected-code-graph-p1-graph.md`` §5 (the reads), ``-p1-answers.md`` P1-22."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from engine_walk_helpers import MemoryReader, union
from loader_helpers import admin_password, skip_or_fail, snapshot_row, throwaway_port

from ccg import sysname
from ccg.engine import bfs
from ccg.engine import state as S
from ccg.engine.reader import QUERIES, CypherReader
from ccg.store import acl
from ccg.store import loader as L
from test_engine_bfs import call_chain, cond_cycle, edge_types, field_graph, items_by_id
from test_engine_bfs_guards import cond_chain
from test_engine_bfs_rows import arg_graph, caller_chain, contract_graph, depends_graph, name_graph, sites_graph, wiring_graph
from test_engine_bfs_bound import planted, three_pairs
from test_engine_bfs_pins import depends_pins

HOST = "127.0.0.1"
SYSTEM = f"p122x{os.getpid() % 1000}"
pytestmark = pytest.mark.live

PARTS = {
    "chain": call_chain(ret_reached=True), "plain": call_chain(ret_reached=False), "field": field_graph(), "cycle": cond_cycle(), "arg": arg_graph(ret_summary=True),
    "contract": contract_graph(), "wiring": wiring_graph(), "dep": depends_graph(), "name": name_graph(), "deep": caller_chain(4), "conds": cond_chain(9),
    "sites": sites_graph(), "short": planted("short"), "long": planted("long"), "dpin": depends_pins(), "three": three_pairs(),
}

#: (part, anchor label, anchor id, state, column, kwargs)
WALKS = [
    ("chain", "Symbol", "A", None, bfs.IMPACT, {}), ("plain", "Symbol", "A", None, bfs.IMPACT, {}),
    ("field", "Param", "p0", S.path_state("p0.path"), bfs.IMPACT, {}), ("field", "Param", "p0", S.VALUE, bfs.IMPACT, {}),
    ("cycle", "Cond", "k1", S.VALUE, bfs.IMPACT, {}), ("arg", "Param", "px", S.path_state("p0.k"), bfs.IMPACT, {}),
    ("contract", "ContractPoint", "cp0", None, bfs.IMPACT, {}), ("contract", "ContractPoint", "cp1", None, bfs.IMPACT, {}), ("contract", "Symbol", "P", S.KEY, bfs.IMPACT, {}),
    ("wiring", "Symbol", "P", None, bfs.IMPACT, {}), ("wiring", "ConfigEntry", "e1", None, bfs.IMPACT, {}), ("dep", "Symbol", "F", None, bfs.DEPENDS_ON, {}),
    ("name", "Symbol", "A", None, bfs.IMPACT, {}), ("deep", "Symbol", "S0", None, bfs.IMPACT, {}), ("deep", "Symbol", "S0", None, bfs.IMPACT, {"bound": 3}),
    ("deep", "Symbol", "S0", None, bfs.IMPACT, {"bound": 1}), ("conds", "Cond", "k0", S.VALUE, bfs.IMPACT, {}),
    ("sites", "ConfigEntry", "E", None, bfs.IMPACT, {}), ("sites", "ConfigEntry", "E2", None, bfs.IMPACT, {}), ("short", "Symbol", "A", None, bfs.IMPACT, {}),
    ("long", "Symbol", "A", None, bfs.IMPACT, {}), ("dpin", "Symbol", "F", None, bfs.DEPENDS_ON, {}), ("three", "Param", "p", S.path_state("p0"), bfs.IMPACT, {}),
]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    port = throwaway_port()
    if port is None:
        skip_or_fail("no throwaway instance (or the docker CLI cannot start): start one with cpg/infra/ccg-falkordb.sh --throwaway up and export "
                     "CCG_FALKORDB_PORT and CCG_CONFIG_DIR (a scratch dir)")
    try:
        pw = admin_password()
    except FileNotFoundError as exc:
        skip_or_fail(f"throwaway admin password not readable ({exc})")
    admin = L.connect(HOST, port, username=None, password=pw)
    cred = tmp_path_factory.mktemp("ccg-credentials-p122")
    graph = union(PARTS)
    graph.rows.snapshot = snapshot_row(SYSTEM, "p122")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CCG_CREDENTIALS_DIR", str(cred))
        users = acl.create_users(admin, SYSTEM)
        creds = acl.read_credentials(SYSTEM)
    w = SimpleNamespace(port=port, admin=admin, graph=graph)
    try:
        build_db = L.connect(HOST, port, username=sysname.acl_user("build", SYSTEM), password=creds["build"])
        report = L.load(L.LiveStore(build_db, SYSTEM), graph.rows, SYSTEM)
        admin.connection.rename(report.build_key, sysname.live_key(SYSTEM))
        w.db = L.connect(HOST, port, username=sysname.acl_user("engine", SYSTEM), password=creds["engine"])
        yield w
    finally:
        for key in list(admin.connection.scan_iter(match=f"ccg_{SYSTEM}*")):
            admin.connection.delete(key)
        for user in users:
            admin.connection.execute_command("ACL", "DELUSER", user)


def _snapshot(w):
    return (sorted(w.visited), {k: sorted((e.cost, e.pred_key, e.edge, e.cuts) for e in lst) for k, lst in w.arrivals.items()}, [(i.kind, i.id, i.chain) for i in w.items()], w.cut,
            w.facts, w.levels, w.frontier_sizes)


def _run(reader, part, label, ident, state, column, kw):
    return bfs.walk(reader, (label, f"{part}:{ident}"), state, column=column, **kw)


@pytest.mark.parametrize("part,label,ident,state,column,kw", WALKS, ids=[f"{w[0]}-{w[2]}-{w[4]}-{sorted(w[5].items())}" for w in WALKS])
def test_a_walk_over_the_store_is_the_walk_over_the_in_memory_reader(world, part, label, ident, state, column, kw):
    live = _run(CypherReader(world.db, SYSTEM), part, label, ident, state, column, kw)
    mem = _run(MemoryReader(world.graph), part, label, ident, state, column, kw)
    assert _snapshot(live) == _snapshot(mem)
    assert live.visited, "an empty walk proves nothing"


def test_the_live_walks_are_not_all_trivial(world):
    sizes = [len(_run(CypherReader(world.db, SYSTEM), *w).visited) for w in WALKS]
    assert sum(sizes) > 100 and min(sizes) >= 2


def test_live_name_hop_labels_come_from_the_count_column(world):
    items = items_by_id(_run(CypherReader(world.db, SYSTEM), "name", "Symbol", "A", None, bfs.IMPACT, {}))
    got = {k.split(":", 1)[1]: (i.status, i.reason) for k, i in items.items()}
    assert got == {"B": ("confirmed", None), "C": ("possible", "name-match"), "D": ("possible", "resolution-unverified"), "E": ("confirmed", None)}


def test_live_contract_walk_from_a_provider_less_point_lists_its_consumers(world):
    items = items_by_id(_run(CypherReader(world.db, SYSTEM), "contract", "ContractPoint", "cp0", None, bfs.IMPACT, {}))
    assert sorted(k.split(":", 1)[1] for k in items) == ["F1", "F2"]
    assert edge_types(items["contract:F1"]) == [("CONSUMES", "<-"), ("HAS_CALL", "<-")]


def test_live_cycles_terminate_and_the_depth_bound_counts_the_cut_edge(world):
    cyc = _run(CypherReader(world.db, SYSTEM), "cycle", "Cond", "k1", S.VALUE, bfs.IMPACT, {})
    assert {k[:2] for k in cyc.visited} == {("Cond", "cycle:k1"), ("Cond", "cycle:k2"), ("Symbol", "cycle:F")}
    deep = _run(CypherReader(world.db, SYSTEM), "deep", "Symbol", "S0", None, bfs.IMPACT, {})
    assert deep.unexplored == 1 and deep.cut == frozenset({("CALLS", "deep:c3", "deep:S2")}) and {i.id.split(":")[1] for i in deep.items()} == {"S1", "S2"}


def test_every_text_the_walk_ran_is_in_the_closed_table_and_nothing_was_written(world):
    rd = CypherReader(world.db, SYSTEM)
    for w in WALKS:
        _run(rd, w[0], w[1], w[2], w[3], w[4], w[5])
    ran = rd.recipe_queries()
    assert ran and all(QUERIES[q["name"]] == q["cypher"] for q in ran)
    g = world.admin.select_graph(sysname.live_key(SYSTEM))
    assert g.ro_query("MATCH (n:Symbol {symbolId: 'chain:A'}) RETURN count(n)").result_set[0][0] == 1


def test_live_a_config_entry_reaches_only_the_sites_it_binds(world):
    items = items_by_id(_run(CypherReader(world.db, SYSTEM), "sites", "ConfigEntry", "E", None, bfs.IMPACT, {}))
    assert sorted(k.split(":", 1)[1] for k in items) == ["C1", "C3"]


def test_live_the_planted_graph_keeps_both_arrivals_and_the_chain_recomputes(world):
    w = _run(CypherReader(world.db, SYSTEM), "short", "Symbol", "A", None, bfs.IMPACT, {})
    h = items_by_id(w)["short:H"]
    edges = [s["edge"] for s in h.chain[1:]]
    assert (h.possible_hops, h.distance) == (1, 2) and sum(e["status"] == "possible" for e in edges) == 1 and w.unexplored == 1
    assert sorted(e.cost[:2] for e in w.arrivals[("CallSite", "short:c2", "path", "r")]) == [(0, 2), (1, 1)]
