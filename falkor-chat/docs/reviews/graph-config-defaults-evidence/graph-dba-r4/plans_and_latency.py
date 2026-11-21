"""K-098 r4 (U12 gate F1): EXPLAIN per form/variant and a scoped latency sweep over threads-per-channel,
plus row equality v2 vs v3 (seed channel and decoy). Throwaway container on 16483 only; ws:cov."""
import json, sys, time
from falkordb import FalkorDB
db = FalkorDB(host="127.0.0.1", port=16483); g = db.select_graph("ws:cov")
src = open("../graph-dba-r2/coverage.py").read().split("def cap(b)")[0]   # query builders only
ns = {}; exec(src, ns)
v2, v3, cur = ns["v2"], ns["v3"], ns["cur"]
scen = json.load(open("cov_scenarios.json"))
FOLD = (ns["ANN"] + "MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t:Thread {threadId: seed.threadId}) "
        + ns["BOUND"] + ns["ret"](False, tie=True))
def explain(q, p):
    res = db.connection.execute_command("GRAPH.EXPLAIN", "ws:cov", "CYPHER " + " ".join(f"{k}={json.dumps(v)}" for k, v in p.items()) + " " + q)
    return [x.decode() if isinstance(x, bytes) else x for x in res]
vec = scen["threads per channel 1M"]
p = {"qVec": vec, "k": 10, "limit": 10, "channelId": "tpc1M"}
for label, q in (("v2 unscoped", v2(False)), ("v2 scoped", v2(True)), ("v3 unscoped", v3(False)), ("v3 scoped", v3(True)),
                 ("folded-anchor scoped", FOLD)):
    print(f"## EXPLAIN {label}"); [print("   ", l) for l in explain(q, p)]
print("## scoped latency (median of 5, ms) and rows, k=10")
for lab in ("1k", "10k", "100k", "1M"):
    v = scen[f"threads per channel {lab}"]
    for chan in (f"tpc{lab}", "c1"):
        p = {"qVec": v, "k": 10, "limit": 10, "channelId": chan}
        rec = {"threads": lab, "channel": chan}
        rows = {}
        for name, q in (("v2", v2(True, True)), ("v3", v3(True, True)), ("folded", FOLD)):
            ts = []
            for _ in range(5):
                t = time.perf_counter(); rs = g.ro_query(q, p, timeout=300000).result_set; ts.append((time.perf_counter() - t) * 1000)
            rec[name + "_ms"] = round(sorted(ts)[2], 1); rows[name] = rs
        rec["rows"] = len(rows["v3"])
        rec["v3_equals_v2"] = sorted((r[3], r[0], tuple(r[4])) for r in rows["v3"]) == sorted((r[3], r[0], tuple(r[4])) for r in rows["v2"])
        print(json.dumps(rec), flush=True)
