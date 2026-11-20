"""K-098 r2: explain the k=10 `same_as_current: False` rows. Throwaway 16483 only, cap 0.
(a) is the verbatim query deterministic against itself? (b) where do current and v2 differ:
seed set, seed order, scores, related ids? Writes equivalence.jsonl. Query texts identical to coverage.py."""
import json
from falkordb import FalkorDB
db = FalkorDB(host="127.0.0.1", port=16483); g = db.select_graph("ws:cov")
db.connection.execute_command("GRAPH.CONFIG", "SET", "QUERY_MEM_CAPACITY", "0")
ANN = "CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
RET = ("RETURN seed.msgId AS msgId, score, [m IN expanded | m.msgId] AS rel ORDER BY score ASC LIMIT $limit")
CUR = (ANN + "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
       "WITH seed, score, collect(DISTINCT related)[..5] AS expanded " + RET)
BOUND = ("CALL { WITH seed OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
         "WITH DISTINCT related LIMIT 5 RETURN collect(related) AS expanded } ")
V2 = ANN + "MATCH (t:Thread {threadId: seed.threadId}) " + BOUND + RET
ANN_ONLY = ANN + "RETURN seed.msgId, score ORDER BY score ASC"
scen = json.load(open("cov_scenarios.json"))
out = open("equivalence.jsonl", "w")
for name, vec in scen.items():
    p = {"qVec": vec, "k": 10, "limit": 10}
    a = g.ro_query(CUR, p, timeout=300000).result_set
    a2 = g.ro_query(CUR, p, timeout=300000).result_set
    b = g.ro_query(V2, p, timeout=300000).result_set
    ann = g.ro_query(ANN_ONLY, p, timeout=60000).result_set
    rec = {"scenario": name,
           "current_vs_itself_identical": a == a2,
           "seed_sets_equal": sorted(r[0] for r in a) == sorted(r[0] for r in b),
           "seed_order_equal": [r[0] for r in a] == [r[0] for r in b],
           "scores_equal_as_multiset": sorted(r[1] for r in a) == sorted(r[1] for r in b),
           "tied_scores_in_top10": len(ann) - len({round(r[1], 12) for r in ann}),
           "related_equal_per_seed": all(dict((r[0], r[2]) for r in a).get(r[0]) == r[2] for r in b),
           "related_sets_equal_per_seed": all(sorted(dict((r[0], r[2]) for r in a).get(r[0]) or []) == sorted(r[2]) for r in b)}
    print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n")
