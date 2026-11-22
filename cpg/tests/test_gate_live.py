"""P1-32b gate harness, live tier (``pytest -m live``): ``StoreSnapshot``'s Cypher against a throwaway FalkorDB of this test's own
(local image, ``--pull never``, 127.0.0.1, labels ``ccg-falkordb=1`` ``ccg-falkordb.throwaway=1``, a port from 16420 up, removed afterwards).
It never touches 6379, 6390, 16390, 16400 or any other running service. The graph is a tiny synthetic one (invented ids)."""
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import gate_helpers as H
from gate_helpers import G

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "tools"))
from snapshot_export import RoRunner  # noqa: E402

pytestmark = pytest.mark.live
IMAGE = "falkordb/falkordb:v4.18.11"
FORBIDDEN_PORTS = {6379, 6390, 16390, 16400}


def _free_port():
    published = subprocess.run(["docker", "ps", "--format", "{{.Ports}}"], capture_output=True, text=True, check=True).stdout
    taken = {int(p) for p in re.findall(r":(\d+)->", published)} | FORBIDDEN_PORTS
    for port in range(16420, 16520):
        if port in taken:
            continue
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port in 16420-16519")


@pytest.fixture(scope="module")
def throwaway():
    try:
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"{IMAGE} is not available locally (docker missing or image not pulled; this test never pulls)")
    port = _free_port()
    name = f"ccg-falkordb-tw-{port}"
    subprocess.run(["docker", "run", "-d", "--pull", "never", "--name", name, "-p", f"127.0.0.1:{port}:6379",
                    "--label", "ccg-falkordb=1", "--label", "ccg-falkordb.throwaway=1", IMAGE], capture_output=True, text=True, check=True)
    try:
        from falkordb import FalkorDB
        db = None
        for _ in range(60):
            try:
                db = FalkorDB(host="127.0.0.1", port=port)
                db.connection.ping()
                break
            except Exception:  # noqa: BLE001 -- not up yet
                time.sleep(0.5)
        else:
            pytest.fail("throwaway FalkorDB did not come up")
        yield db
    finally:
        labels = subprocess.run(["docker", "inspect", "-f", "{{index .Config.Labels \"ccg-falkordb.throwaway\"}}", name], capture_output=True, text=True).stdout.strip()
        if labels == "1":                      # only ever remove the container this fixture started
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def load_world(db, key, w: "H.World"):
    g = db.select_graph(key)
    g.query("CREATE (:Snapshot {snapshotId: $s})", {"s": H.SNAP})
    g.query("CREATE (:ContractPoint {cpId: $c, kind: 'http', keyNorm: 'GET /a'})", {"c": w.cp})
    g.query("CREATE (:Link {linkId: $l}), (:CallSite {callSiteId: $c}), (:Symbol {symbolId: $s, qname: $q, path: $p, lineStart: 5, lineEnd: 20, ownedRegions: [28, 32]})",
            {"l": w.link, "c": w.cs, "s": w.sid, "q": w.qname, "p": w.qname.split("::")[0]})
    g.query("MATCH (l:Link {linkId: $l}), (p:ContractPoint {cpId: $c}) CREATE (l)-[:ON]->(p)", {"l": w.link, "c": w.cp})
    g.query("MATCH (c:CallSite {callSiteId: $c}), (l:Link {linkId: $l}) CREATE (c)-[:VIA]->(l)", {"l": w.link, "c": w.cs})
    g.query("MATCH (s:Symbol {symbolId: $s}), (c:CallSite {callSiteId: $c}) CREATE (s)-[:HAS_CALL]->(c)", {"s": w.sid, "c": w.cs})
    g.query("CREATE (:ConfigEntry {entryId: $e, path: 'deploy/compose.yaml', keyPath: 'services.web'})", {"e": "9" * 32})
    # a name-matched call: CallSite -CALLS_NAME-> Name -RESOLVES_TO-> Symbol
    g.query("CREATE (:CallSite {callSiteId: $c2}), (:Name {name: 'n'}), (:Symbol {symbolId: $s2, qname: 'svc/d.py::callee', path: 'svc/d.py', lineStart: 1, lineEnd: 9})",
            {"c2": "6" * 32, "s2": "7" * 32})
    g.query("MATCH (c:CallSite {callSiteId: $c2}), (n:Name {name: 'n'}), (s:Symbol {symbolId: $s2}) CREATE (c)-[:CALLS_NAME]->(n), (n)-[:RESOLVES_TO]->(s)",
            {"c2": "6" * 32, "s2": "7" * 32})
    return StoreSnapshotFor(g)


def StoreSnapshotFor(g):
    return G.StoreSnapshot(RoRunner(g))


def test_live_snapshot_reads_and_replay_match_the_in_memory_semantics(throwaway):
    w = H.World()
    snap = load_world(throwaway, "gate_tw_a", w)
    assert snap.snapshot_id() == H.SNAP
    assert snap.node_exists("ContractPoint", w.cp) and not snap.node_exists("ContractPoint", "f" * 32)
    assert snap.symbol(w.sid) == {"symbolId": w.sid, "qname": w.qname, "path": "svc/c.py", "lineStart": 5, "lineEnd": 20, "ownedRegions": [[28, 32]]}
    assert snap.symbol("f" * 32) is None and snap.config_entry("9" * 32)["keyPath"] == "services.web"
    assert [s["symbolId"] for s in snap.symbols_of_path("svc/c.py")] == [w.sid]
    assert G.replay_chain(w.chain(), snap) == ("ok", "")
    fab = [H.hop0("ContractPoint", w.cp), H.hopn("Symbol", w.sid, "PROVIDES", "<-")]            # real nodes, no PROVIDES edge
    assert G.replay_chain(fab, snap) == ("fail", "no-edge")
    rev = w.chain()
    rev[1]["dir"] = "->"
    assert G.replay_chain(rev, snap) == ("fail", "kind-pair")                                    # (ContractPoint, Link) is not an ON pair


def test_live_calls_name_two_edge_form_and_class_d_not_loaded(throwaway):
    w = H.World()
    snap = load_world(throwaway, "gate_tw_b", w)
    chain = [H.hop0("CallSite", "6" * 32), H.hopn("Symbol", "7" * 32, "CALLS_NAME", "->", status="possible")]
    assert G.replay_chain(chain, snap) == ("ok", "")
    assert snap.edge_type_loaded("CALLS_NAME") and not snap.edge_type_loaded("READS")
    d = [H.hop0("ContractPoint", w.cp), H.hopn("CallSite", w.cs, "READS", "<-")]
    assert G.replay_chain(d, snap) == ("error", "class-D-not-loaded")


def test_live_every_replayable_edge_pair_query_is_valid_cypher(throwaway):
    g = throwaway.select_graph("gate_tw_c")
    g.query("CREATE (:Snapshot {snapshotId: 'x'})")                  # RO_QUERY on a key that does not exist is an error
    snap = G.StoreSnapshot(RoRunner(g))
    for et, (cls, pairs) in G.EDGES.items():
        for sl, dl in pairs:
            assert snap.edge_joins(et, (sl, "a" * 32), (dl, "b" * 32)) is False, (et, sl, dl)


def test_live_the_hit_rule_end_to_end_including_owned_regions_and_the_stamp(throwaway):
    w = H.World(lines=(30,), span=(5, 20))                     # the line lies in an owned region, not in the span
    snap = load_world(throwaway, "gate_tw_d", w)
    ans = w.answer([H.ans_item(H.sym_subject(w.sid, w.qname, 5, 20, ownedRegions=[28, 32]), w.chain())])
    res = G.score_gate([w.rec], w.run(ans), snap, ())
    assert res["items"][w.rec["itemId"]]["outcome"] == "hit" and res["defects"] == {}
    assert G.check_stamp(ans, snap.snapshot_id()) == []


def test_live_two_snapshot_nodes_are_a_harness_error(throwaway):
    g = throwaway.select_graph("gate_tw_e")
    g.query("CREATE (:Snapshot {snapshotId: 'x'}), (:Snapshot {snapshotId: 'y'})")
    with pytest.raises(G.HarnessError):
        G.StoreSnapshot(RoRunner(g)).snapshot_id()
