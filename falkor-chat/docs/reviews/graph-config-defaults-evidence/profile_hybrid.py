import sys
from falkordb import FalkorDB
sys.path.insert(0, '.')
db = FalkorDB(host="127.0.0.1", port=16479); g = db.select_graph("ws:synth")
v = [float(x) for x in g.ro_query("MATCH (m:Message {msgId:'tlong-9999'}) RETURN m.embedding").result_set[0][0]]
q = ("CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
     "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) "
     "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
     "WITH seed, score, collect(DISTINCT related)[..5] AS expanded "
     "RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, score, [m IN expanded | m.text] AS relatedContext "
     "ORDER BY score ASC LIMIT $limit")
for line in g.profile(q, {"qVec": v, "k": 10, "limit": 10}): print(line)
