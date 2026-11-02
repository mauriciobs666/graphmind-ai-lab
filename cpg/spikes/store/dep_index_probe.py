"""S20/S21 side ruling -- does the loader's ``Dependency`` edge load need a ``Dependency(depId)`` index?

The loader (S17-L1) writes ``(:Unit)-[:DEPENDS_ON]->(:Dependency)`` edges with
``MATCH (b:Dependency {depId: r.t})`` (``load.edge_statement``); ``docs/plans/connected-code-graph-p0-graph.md`` §1.6
deferred that index "to the phase that loads the label". This probe loads N ``Dependency`` nodes and N ``DEPENDS_ON``
edges in ``BATCH_MAX`` batches, once with no ``Dependency`` index and once with ``Dependency(depId)``, and records the
edge-load time, the plan operator and the index's cost.

Keys: ``ccg_p0spike__dep_*`` only. Usage: ``python dep_index_probe.py [--sizes 1000,5000,20000]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import synth as S  # noqa: E402
import load as L  # noqa: E402

N_UNITS = 50


def run_case(st: S.SpikeStore, n: int, indexed: bool, constrained: bool = False, timeout_s: int = 600) -> dict:
    key = f"ccg_p0spike__dep_{n}_{'idx' if indexed else 'noidx'}{'_uq' if constrained else ''}"
    S.check_spike_key(key)
    if st.exists(key):
        st.delete(key)
    st.query(key, "CREATE INDEX FOR (n:Unit) ON (n.unitId)")
    if indexed:
        st.query(key, "CREATE INDEX FOR (n:Dependency) ON (n.depId)")
    units = [{"unitId": f"U{i}"} for i in range(N_UNITS)]
    st.query(key, L.node_statement("Unit"), {"rows": units})
    deps = [{"depId": f"D{i:07d}", "ecosystem": "pypi", "package": f"pkg{i}"} for i in range(n)]
    t0 = time.monotonic()
    for chunk in S.stream_batches(iter(deps)):
        st.query(key, L.node_statement("Dependency"), {"rows": chunk})
    t_nodes = time.monotonic() - t0
    plan = st.db.select_graph(key).explain(
        "UNWIND $rows AS r MATCH (a:Unit {unitId: r.s}) MATCH (b:Dependency {depId: r.t}) CREATE (a)-[e:DEPENDS_ON]->(b) SET e = r.p",
        {"rows": [{"s": "U0", "t": "D0000000", "p": {}}]})
    ops = [l.strip().split(" | ")[0] for l in str(plan).splitlines()] if not isinstance(plan, list) else [str(x).strip() for x in plan]
    rows = [{"s": f"U{i % N_UNITS}", "t": f"D{i:07d}", "p": {"version": "1", "manifestPath": "pyproject.toml", "line": 1, "scope": "runtime"}}
            for i in range(n)]
    per_batch = []
    t0 = time.monotonic()
    for chunk in S.stream_batches(iter(rows)):
        tb = time.monotonic()
        res = st.query(key, L.edge_statement("Unit", "DEPENDS_ON", "Dependency", True), {"rows": chunk})
        per_batch.append(round(time.monotonic() - tb, 3))
        assert res.relationships_created == len(chunk)
        if time.monotonic() - t0 > timeout_s:
            raise S.SynthError("edge load exceeded the probe's own budget")
    t_edges = time.monotonic() - t0
    out = {"dependencies": n, "indexed": indexed, "constrained": constrained, "nodeLoadSeconds": round(t_nodes, 3),
           "edgeLoadSeconds": round(t_edges, 3), "perBatchSeconds": per_batch, "planOperators": ops}
    if indexed:
        mem = st.memory(key)
        out["indicesMb"] = mem["indices_sz_mb"]
    st.delete(key)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="1000,5000,20000")
    args = ap.parse_args(argv)
    st = S.SpikeStore("127.0.0.1", 6390)
    results = []
    for n in (int(x) for x in args.sizes.split(",")):
        for indexed in (True, False):
            r = run_case(st, n, indexed)
            print(json.dumps({k: v for k, v in r.items() if k != "perBatchSeconds"}), flush=True)
            results.append(r)
    dest = HERE / "results"
    dest.mkdir(exist_ok=True)
    (dest / "dep_index_probe.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
