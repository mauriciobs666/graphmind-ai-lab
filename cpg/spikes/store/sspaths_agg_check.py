"""S20 gate follow-up -- ``algo.SSpaths`` with the aggregation inside the query and a large ``pathCount``.

The benchmark (``bench_reach.py``) returned rows, so ``RESULTSET_SIZE`` and ``pathCount`` were both 10,000 and the
reply held 10,000 paths. Here the reply is one row (``collect(DISTINCT ...)``), ``pathCount`` is 1,000,000, and
the item set is compared to ``bfs-v1``'s on the same call-only segment (``levels = maxLen = 2 * 3``), 50 anchors,
20 repeats (set digest identical?), wall time per answer. No chain is returned in this form.
Usage: ``python sspaths_agg_check.py [--repeats 20] [--path-count 1000000]`` (keys: ``ccg_p0spike`` read-only).
A ``pathCount`` of 3,000,000 on the top hub OOM-killed the 2 GB container (see MEASUREMENTS.md): do not raise it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_reach as R  # noqa: E402
import bfs as B  # noqa: E402
import synth as S  # noqa: E402

QUERY = ("MATCH (a:Symbol {symbolId: $id}) CALL algo.SSpaths({sourceNode: a, relTypes: ['CALLS','HAS_CALL'], "
         "relDirection: 'incoming', weightProp: 'w', maxLen: $maxlen, pathCount: $pc}) YIELD path "
         "WITH last(nodes(path)) AS e WHERE e:Symbol AND e.symbolId <> $id RETURN count(*), collect(DISTINCT e.symbolId)")


def answer(graph, anchor, depth, pc):
    t0 = time.perf_counter()
    r = graph.ro_query(QUERY, {"id": anchor, "maxlen": 2 * depth, "pc": pc}, R.READ_TIMEOUT_MS).result_set[0]
    ids = sorted(r[1])
    return {"paths": r[0], "ids": ids, "seconds": time.perf_counter() - t0, "exhausted": r[0] < pc}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--path-count", type=int, default=1_000_000)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args(argv)
    st = S.SpikeStore("127.0.0.1", 6390)
    g = st.db.select_graph("ccg_p0spike")
    out = []
    for a in S.Synth(1000).anchors():
        bs = {x[0] for x in B.answer(g, a["symbolId"], mode="segment", levels=2 * args.depth, timeout=R.READ_TIMEOUT_MS)["items"]}
        runs = [answer(g, a["symbolId"], args.depth, args.path_count) for _ in range(args.repeats)]
        digests = {B.digest(r["ids"]) for r in runs}
        rec = {"anchor": a["symbolId"], "class": a["class"], "indegree": a["indegree"], "bfsItems": len(bs),
               "sspathsItems": len(runs[0]["ids"]), "setEqual": set(runs[0]["ids"]) == bs, "repeatsIdentical": len(digests) == 1,
               "paths": runs[0]["paths"], "pathCountExhausted": all(r["exhausted"] for r in runs),
               "seconds": [round(r["seconds"], 4) for r in runs]}
        out.append(rec)
        print({k: v for k, v in rec.items() if k != "seconds"}, flush=True)
    (HERE / "results" / "sspaths_aggregated.json").write_text(json.dumps({"pathCount": args.path_count, "depth": args.depth, "records": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
