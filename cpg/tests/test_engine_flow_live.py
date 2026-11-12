"""P1-23 behaviour (1), LIVE on a throwaway instance only (``infra/ccg-falkordb.sh --throwaway up``; ``-m live``).

Same rules as ``tests/test_engine_bfs_live.py``: needs ``CCG_FALKORDB_PORT`` (the throwaway's port) and ``CCG_CONFIG_DIR`` (a scratch dir holding its ``falkordb-admin.pass``);
``CCG_REQUIRE_LIVE=1`` turns a missing instance into a failure. Loads the union of the flow fixtures through ``ccg.store.loader.load`` as ``ccg_build_<s>``, renames it to the
live key as admin and reads it as ``ccg_engine_<s>`` only (``CypherReader``).

Two things only the store can show: the flow walks over ``CypherReader`` (the ``units`` read returns ``flowAvailable`` and ``language`` as the gate reads them) agree with the
walks over ``MemoryReader``; and ``candidate_count`` returns **no row**, not ``1``, for a ``VIA`` edge with no ``consumerKey`` (``docs/reviews/connected-code-graph-p1-21.md``
Pass 2 G1), which ``bfs.candidate_counts`` reads as "N unknown" (``None``).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from engine_walk_helpers import MemoryReader, union
from loader_helpers import admin_password, skip_or_fail, snapshot_row, throwaway_port

from ccg import sysname
from ccg.engine import bfs, flow
from ccg.engine import state as S
from ccg.engine.reader import CypherReader
from ccg.store import acl
from ccg.store import loader as L
from test_engine_flow import callee_chain, callee_graph, caller_chain, equal_cost_callers, point_rows_graph, reach_graph, two_site_caller, via_graph

HOST = "127.0.0.1"
SYSTEM = f"p123x{os.getpid() % 1000}"
pytestmark = pytest.mark.live

PARTS = {"callers": caller_chain(flow=False), "mixed": caller_chain(flow=False, caller_flow=True, caller_lang="python"), "points": point_rows_graph(False),
         "callee": callee_graph(False), "reach": reach_graph(), "ties": equal_cost_callers(), "two": two_site_caller(name_site_first=True)[0], "via": via_graph(), "chain": callee_chain()}

#: (part, anchor label, anchor id, state, column, table name)
WALKS = [
    ("callers", "Param", "pF", S.VALUE, bfs.IMPACT, "as-built"), ("mixed", "Param", "pF", S.VALUE, bfs.IMPACT, "as-built"),
    ("points", "Param", "pF", S.path_state("p0"), bfs.IMPACT, "as-built"), ("callee", "Symbol", "F", S.VALUE, bfs.DEPENDS_ON, "as-built"),
    ("reach", "Param", "px", S.path_state("p0"), bfs.IMPACT, "V-flow"), ("reach", "Param", "pF", S.VALUE, bfs.IMPACT, "V-both"), ("reach", "Param", "pF", S.VALUE, bfs.IMPACT, "V-union"),
    ("ties", "Param", "pF", S.VALUE, bfs.IMPACT, "as-built"), ("two", "Param", "pF", S.VALUE, bfs.IMPACT, "as-built"),
    ("chain", "Param", "pF", S.VALUE, bfs.IMPACT, "V-both"), ("chain", "Param", "pF", S.VALUE, bfs.IMPACT, "V-union"),
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
    cred = tmp_path_factory.mktemp("ccg-credentials-p123")
    graph = union(PARTS)
    graph.rows.snapshot = snapshot_row(SYSTEM, "p123")
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
            w.facts, w.levels, w.frontier_sizes, w.units)


def _run(reader, part, label, ident, state, column, table):
    return bfs.walk(reader, (label, f"{part}:{ident}"), state, column=column, table=flow.candidate_table(table))


@pytest.mark.parametrize("part,label,ident,state,column,table", WALKS, ids=[f"{w[0]}-{w[2]}-{w[5]}" for w in WALKS])
def test_a_flow_walk_over_the_store_is_the_walk_over_the_in_memory_reader(world, part, label, ident, state, column, table):
    live = _run(CypherReader(world.db, SYSTEM), part, label, ident, state, column, table)
    mem = _run(MemoryReader(world.graph), part, label, ident, state, column, table)
    assert _snapshot(live) == _snapshot(mem)
    assert live.visited and live.units, "an empty walk proves nothing"


def test_the_live_units_read_gives_the_gate_its_flag_and_language(world):
    w = _run(CypherReader(world.db, SYSTEM), "callers", "Param", "pF", S.VALUE, bfs.IMPACT, "as-built")
    assert w.units == {"callers:uf": flow.UnitInfo(False, "javascript")}
    items = {i.id.split(":", 1)[1]: i for i in w.items()}
    assert (items["G"].status, items["G"].reason) == ("possible", "flow-unavailable:javascript")


def test_live_candidate_count_has_no_row_for_a_via_edge_without_a_consumer_key_and_the_engine_reads_it_as_unknown(world):
    rd = CypherReader(world.db, SYSTEM)
    pair = ["via:c3", "via:L3"]
    assert rd.read("candidate_count", {"pairs": [pair]}) == []                                  # the absent-key row: no row, not 1
    assert rd.read("candidate_count", {"pairs": [["via:c1", "via:L1"]]}) == [("via:c1", "via:L1", 2)]
    got = bfs.candidate_counts(rd, [("via:c3", "via:L3"), ("via:c1", "via:L1"), ("via:orphan", "via:L1")])
    assert got == {("via:c3", "via:L3"): None, ("via:c1", "via:L1"): 2, ("via:orphan", "via:L1"): None}
    mem = bfs.candidate_counts(MemoryReader(world.graph), [("via:c3", "via:L3"), ("via:c1", "via:L1"), ("via:orphan", "via:L1")])
    assert mem == got
