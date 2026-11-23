"""P1-32b ``gate_run``, live tier (``pytest -m live``): the ask translation and the ``run`` verb's default wiring against the real engine reader over a
loader-built graph on a throwaway FalkorDB of this module's own (``test_gate_live.throwaway``: local image, ``--pull never``, 127.0.0.1, labels
``ccg-falkordb=1`` ``ccg-falkordb.throwaway=1``, a port from 16420 up, removed afterwards; never 6379/6390/16390/16400). No ACL users: the engine reader and
the snapshot reader are the default user's (injected; the ACL-role reader on a password-protected store is ``test_gate_rehearsal_defects_live.py``); the ``raw`` role snapshot cross-check is injected. Invented data (``engine_helpers.mini_rows``)."""
import json
import types

import pytest
from engine_answer_helpers import hexify
from engine_helpers import mini_rows
from loader_helpers import snapshot_row
from test_engine_answer import link_world_g
from test_gate_live import throwaway  # noqa: F401  (the container fixture)

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env
import gate_run as R
from ccg import sysname
from ccg.engine.reader import CypherReader
from ccg.store import loader as L
from clients.common import Client

pytestmark = pytest.mark.live
SYSTEM, SYSTEM2 = "gtlive1", "gtlive2"
HEALTH = H.contract("http", "GET /shop/api/health")
HEALTH_ID = G._mk.contract_id(HEALTH)
X = H.contract("http", "GET /x")
X_ID = G._mk.contract_id(X)


def _answer_rows():
    """The answer builder's own ``link_world`` (hexified), its contract point ``GET /x`` renamed to the contract id the gold carries."""
    g, ids = hexify(link_world_g())
    cp = ids["cp1"]
    for n in g.rows.nodes["ContractPoint"]:
        if n["cpId"] == cp:
            n["cpId"] = X_ID
    for rows in g.rows.edges.values():
        for e in rows:
            for k in ("s", "t"):
                if e[k] == cp:
                    e[k] = X_ID
    g.rows.snapshot = snapshot_row(SYSTEM2, "t")
    return g.rows, ids, g.repo


def _rows():
    r = mini_rows(SYSTEM)
    for n in r.nodes["ContractPoint"]:
        if n["cpId"] == "cp_health":
            n["cpId"] = HEALTH_ID
    for rows in r.edges.values():
        for e in rows:
            for k in ("s", "t"):
                if e[k] == "cp_health":
                    e[k] = HEALTH_ID
    # a second symbol with the qname of ``app.Outer`` in another file of the same repo, and a masked config key path
    r.nodes["Symbol"].append(dict(symbolId="s_twin", repoId="repo-a", unitId="ua", kind="function", name="Outer", nameNorm="outer", qname="app.Outer",
                                  path="app/b.py", lineStart=1, lineEnd=3, depth=0))
    r.edges.setdefault(("File", "DEFINES", "Symbol"), []).append({"s": "fa2", "t": "s_twin"})
    r.nodes["ConfigEntry"].append(dict(entryId="eM", repoId="repo-a", unitId="ua", path="cfg/app.yaml", line=30, keyPath="services.[#…].env", valueHash="v9"))
    return r


@pytest.fixture(scope="module")
def store(throwaway):
    db = throwaway
    port = db.connection.connection_pool.connection_kwargs["port"]
    rep = L.load(L.LiveStore(L.connect("127.0.0.1", port, username=None, password=None), SYSTEM), _rows(), SYSTEM)
    db.connection.rename(rep.build_key, sysname.live_key(SYSTEM))
    rd = CypherReader(db, SYSTEM)
    yield types.SimpleNamespace(db=db, port=port, client=Client(SYSTEM, host="127.0.0.1", port=port, reader_factory=lambda *a: rd))
    db.connection.delete(sysname.live_key(SYSTEM))


def test_live_each_anchor_type_resolves_to_its_own_id_by_the_translation_and_a_twin_qname_is_ambiguous_by_the_old_locator(store):
    c = store.client
    sym = H.item(1, {"type": "symbol", "repoId": "repo-a", "qname": "app.Outer", "path": "app/a.py"}, H.sym("x.py::g"))
    sym["anchorId"] = "s_outer"
    cfg = H.item(2, {"type": "config", "repoId": "repo-a", "path": "cfg/app.yaml", "keyPath": "services." + "x" * 30 + ".env"}, H.sym("x.py::g"))
    cfg["anchorId"] = "eM"
    con = H.item(3, HEALTH, H.sym("x.py::g"))
    for rec, key in ((sym, "symbolId"), (cfg, "entryId"), (con, "cpId")):
        aid = rec["anchorId"]
        res = c.resolve(R.cli_request(G.build_request(rec, aid))["anchor"])
        assert res["status"] == "resolved" and [x[key] for x in res["candidates"]] == [aid], (rec["anchor"]["type"], res)
    assert c.resolve({"kind": "symbol", "repo": "repo-a", "qname": "app.Outer"})["status"] == "ambiguous"       # the pre-round-B locator
    assert c.resolve({"kind": "config", "repo": "repo-a", "path": "cfg/app.yaml", "keyPath": "services." + "x" * 30 + ".env"})["status"] == "unresolved"


@pytest.fixture(scope="module")
def store2(throwaway):
    db = throwaway
    port = db.connection.connection_pool.connection_kwargs["port"]
    rows, ids, repo = _answer_rows()
    rep = L.load(L.LiveStore(L.connect("127.0.0.1", port, username=None, password=None), SYSTEM2), rows, SYSTEM2)
    db.connection.rename(rep.build_key, sysname.live_key(SYSTEM2))
    rd = CypherReader(db, SYSTEM2)
    yield types.SimpleNamespace(db=db, port=port, ids=ids, repo=repo, client=Client(SYSTEM2, host="127.0.0.1", port=port, reader_factory=lambda *a: rd))
    db.connection.delete(sysname.live_key(SYSTEM2))


def test_live_the_resolve_dry_run_and_the_whole_run_verb_over_the_default_wiring(store2, tmp_path):
    store = store2
    e = Env(tmp_path)
    imp = {"type": "symbol", "repoId": store.repo, "qname": "ui/e.ts::getHealth", "path": "ui/e.ts"}
    rec = H.item(1, X, imp, lines=[17])
    e.set_records([rec])
    a = types.SimpleNamespace(host="127.0.0.1", port=store.port, graph=None, system=SYSTEM2)
    deps = R._default_deps(a, client_factory=lambda system, host, port: store.client, db_factory=lambda h, p: store.db)       # this throwaway has no ACL users: the default user stands in (the role path is test_gate_rehearsal_defects_live.py)
    deps["refresh_fn"] = e.refresh                                   # the build is the throwaway's own graph above; the Target wiring is tested offline
    assert deps["live_graph_present"]() is True                      # round C (N4): the real probe sees that graph, so the guard would refuse a real build here
    deps["live_graph_present"] = lambda: False                       # this test stands the graph up itself and replaces the build with a no-op
    deps["engine_snapshot_id"] = None                                # needs the raw ACL role
    out = R.run_gate(e.repo, "ac5-gate-1", [e.lists], e.cust, home=e.home, observed={"algorithm": {"id": "bfs-v1", "version": "1.2"}, "defaultDepthBound": 2},
                     probe_map=e.probes, **deps)
    assert out["asked"] == 1
    run = json.loads(e.run_files()[0].read_text())
    snap = deps["snapshot_factory"](None)
    res = G.score_gate([rec], run, snap, ())
    assert res["harnessErrors"] == {} and res["stops"] == {}, res
    assert res["items"][rec["itemId"]]["outcome"] == "hit"
    assert run["snapshotId"] == snap.snapshot_id() and e.marker()["state"] == "asked"


def test_live_a_wrong_id_is_refused_by_the_dry_run_with_counts_only(store, tmp_path):
    e = Env(tmp_path)
    ghost = H.contract("http", "GET /shop/api/items/{}")             # stored under the id "cp_items", not the contract id the record carries
    rec = H.item(1, ghost, H.sym("x.py::g"))
    e.set_records([rec])
    a = types.SimpleNamespace(host="127.0.0.1", port=store.port, graph=None, system=SYSTEM)
    deps = R._default_deps(a, client_factory=lambda system, host, port: store.client, db_factory=lambda h, p: store.db)       # this throwaway has no ACL users: the default user stands in (the role path is test_gate_rehearsal_defects_live.py)
    snap = deps["snapshot_factory"](None)
    fake = types.SimpleNamespace(node_exists=lambda label, nid: True, symbol=lambda i: {}, config_entry=lambda i: {})
    with pytest.raises(R.ResolveRefusal) as exc:
        R.resolve_preflight([rec], {}, fake, store.client.resolve)
    assert exc.value.counts["id-mismatch"] == 1 and snap.snapshot_id()
