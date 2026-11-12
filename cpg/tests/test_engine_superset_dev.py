"""P1-39c live: the entry of the measuring tool ``cpg/eval/superset_dev.py`` (``-m live``; the tool's logic is in that module, its offline proof in
``tests/test_superset_dev_tool.py``). Pinned, with the module, by the ``ccg-p1-flow-fallback-v1-tool`` record line.

Throwaway instance only (``infra/ccg-falkordb.sh --throwaway up``; ``CCG_FALKORDB_PORT`` and a scratch ``CCG_CONFIG_DIR`` exported before ``up``;
``CCG_REQUIRE_LIVE=1`` turns a missing instance into a failure).

* ``test_the_digest_and_the_candidate_measurement_run_over_the_store``: always, on the flow fixtures loaded as in ``tests/test_engine_flow_live.py``: the digest's
  read-only pages work against FalkorDB (``properties()``, the keyset paging), are stable, and ``measure`` over ``CypherReader`` is byte-equal across two runs.
* ``test_the_real_phase``: only with ``CCG_SUPERSET_PHASE=A`` or ``B`` exported (the pre-registered run: a quiet host, the Joern toolchain, the dedicated
  throwaway): runs the tool's phase and fails on a non-zero exit. Never part of an ordinary run.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from engine_walk_helpers import union
from loader_helpers import admin_password, skip_or_fail, snapshot_row, throwaway_port
from test_engine_flow import callee_chain, caller_chain

from ccg import sysname
from ccg.engine import bfs, flow
from ccg.store import acl
from ccg.store import loader as L

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
sd = importlib.import_module("superset_dev")

pytestmark = pytest.mark.live
ENV0 = dict(os.environ)                      # the conftest's autouse fixture replaces CCG_CONFIG_DIR per test; the real phase needs the exported one
SYSTEM = f"sdev{os.getpid() % 1000}"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    port = throwaway_port()
    if port is None:
        skip_or_fail("no throwaway instance: start one with cpg/infra/ccg-falkordb.sh --throwaway up and export CCG_FALKORDB_PORT and CCG_CONFIG_DIR (a scratch dir)")
    admin = L.connect(sd.HOST, port, username=None, password=admin_password())
    cred = tmp_path_factory.mktemp("ccg-credentials-sdev")
    graph = union({"callers": caller_chain(flow=False), "callee": callee_chain()})
    graph.rows.snapshot = snapshot_row(SYSTEM, "sdev")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CCG_CREDENTIALS_DIR", str(cred))
        users = acl.create_users(admin, SYSTEM)
        creds = acl.read_credentials(SYSTEM)
        try:
            build_db = L.connect(sd.HOST, port, username=sysname.acl_user("build", SYSTEM), password=creds["build"])
            report = L.load(L.LiveStore(build_db, SYSTEM), graph.rows, SYSTEM)
            admin.connection.rename(report.build_key, sysname.live_key(SYSTEM))
            yield SimpleNamespace(port=port, report=report, live=sd.LiveGraph(port, SYSTEM))
        finally:
            for key in list(admin.connection.scan_iter(match=f"ccg_{SYSTEM}*")):
                admin.connection.delete(key)
            for user in users:
                admin.connection.execute_command("ACL", "DELUSER", user)


def test_the_digest_and_the_candidate_measurement_run_over_the_store(world):
    first, counts = sd.graph_digest(world.live)
    second, _ = sd.graph_digest(world.live)
    assert first == second and len(first) == 64
    assert counts == {"nodes": world.report.node_total, "edges": world.report.edge_total}      # the pages reassembled every node and edge the loader wrote
    dev = [sd.DevItem("g", "callers:F", "callers:G", "R", "a", "R", "i.py", "q", "callee-arg", "python"),
           sd.DevItem("h", "callers:F", "callers:H", "R", "a", "R", "i.ts", "q", "callee-arg,return-use", "javascript")]
    res = sd.measure(world.live.reader, dev, walk=bfs.walk, tables=flow.candidate_table, impact=bfs.IMPACT)
    assert res["candidates"]["as-built"]["listedIds"] == ["g", "h"] and res["candidates"]["V-union"]["burdenSum"] >= res["candidates"]["as-built"]["burdenSum"]


@pytest.fixture
def restore_environ():
    """``sd.main`` writes ``os.environ`` (scratch credentials/audit dirs, the forbid-real-credentials guard): put it back."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_the_real_phase(monkeypatch, restore_environ):
    phase = ENV0.get("CCG_SUPERSET_PHASE")
    if phase not in ("A", "B"):
        pytest.skip("CCG_SUPERSET_PHASE=A|B is not exported: the pre-registered run is not part of an ordinary live run")
    for k in ("CCG_CONFIG_DIR", "CCG_FALKORDB_PORT"):
        monkeypatch.setenv(k, ENV0[k])
    assert sd.main([phase], env=ENV0) == 0
