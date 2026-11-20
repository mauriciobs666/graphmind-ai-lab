"""K-098 r2: repeat timings + GRAPH.PROFILE for the scoped v2 form (worst scenario), throwaway 16483."""
import json, time
from falkordb import FalkorDB
db = FalkorDB(host="127.0.0.1", port=16483); g = db.select_graph("ws:cov")
vec = json.load(open("cov_scenarios.json"))["worst: depth 100k clustered + fan-in 100k"]
ANN = "CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
BOUND = ("CALL { WITH seed OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
         "WITH DISTINCT related LIMIT 5 RETURN collect(related) AS expanded } ")
RET = ("RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, score, [m IN expanded | m.text] AS relatedContext "
       "ORDER BY score ASC LIMIT $limit")
for label, scope in (("scoped, Channel->Thread", "MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t) "),
                     ("scoped, Thread<-Channel", "MATCH (t)<-[:HAS_THREAD]-(c:Channel {channelId: $channelId}) "),
                     ("unscoped", "")):
    q = ANN + "MATCH (t:Thread {threadId: seed.threadId}) " + scope + BOUND + RET
    p = {"qVec": vec, "k": 10, "limit": 10, "channelId": "c1"}
    ts = []
    for _ in range(5):
        t = time.perf_counter(); g.ro_query(q, p, timeout=60000); ts.append(round((time.perf_counter() - t) * 1000, 1))
    print(label, "ms x5:", ts)
    for line in g.profile(q, p): print("   ", line)
