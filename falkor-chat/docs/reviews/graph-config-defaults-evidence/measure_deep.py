"""K-098: decompose hybrid_search memory (walk vs. Entity expansion) + repeatability. Probe port only."""
import json, random, sys, time
from falkordb import FalkorDB
from redis.exceptions import ResponseError
db = FalkorDB(host="127.0.0.1", port=16479); r = db.connection; g = db.select_graph("ws:deep")
vec = lambda i: [float(x) for x in g.ro_query("MATCH (m:Message {msgId:$i}) RETURN m.embedding", {"i": i}).result_set[0][0]]
ANN = "CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
FULL = (ANN + "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) "
  "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
  "WITH seed, score, collect(DISTINCT related)[..5] AS expanded "
  "RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, score, [m IN expanded | m.text] AS relatedContext ORDER BY score ASC LIMIT $limit")
V = {
 "ann only": ANN + "RETURN seed.msgId, score ORDER BY score LIMIT $limit",
 "ann + walk": ANN + "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) RETURN seed.msgId, score ORDER BY score LIMIT $limit",
 "ann + expansion": ANN + "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) WITH seed, score, collect(DISTINCT related)[..5] AS x RETURN seed.msgId, score, size(x) ORDER BY score LIMIT $limit",
 "full": FULL,
}
def cap(b): r.execute_command("GRAPH.CONFIG", "SET", "QUERY_MEM_CAPACITY", str(b))
def ok(c, p):
    try: g.ro_query(c, p, timeout=60000); return True
    except ResponseError as e:
        assert "mem consumption" in str(e), e; return False
def bisect(c, p):
    cap(0); ok(c, p); lo, hi = 1, 1 << 30
    cap(lo)
    if ok(c, p): cap(0); return 0
    while hi - lo > 1024:
        m = (lo + hi) // 2; cap(m)
        if ok(c, p): hi = m
        else: lo = m
    cap(0); return hi
seeds = {"depth 100k + hub 50k": "d-99999", "depth 60k + hub": "d-60000", "depth 30k, no hub": "d-30000"}
out = open(sys.argv[1], "a")
for sname, sid in seeds.items():
    p = {"qVec": vec(sid), "k": 10, "limit": 10}
    for vname, c in V.items():
        t=time.perf_counter(); cap(0); g.ro_query(c, p, timeout=60000); ms=(time.perf_counter()-t)*1000; runs = [bisect(c, p)]
        rec = {"seed": sname, "variant": vname, "min_cap_kib": round(runs[0] / 1024), "uncapped_ms": round(ms, 1)}
        print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n")
