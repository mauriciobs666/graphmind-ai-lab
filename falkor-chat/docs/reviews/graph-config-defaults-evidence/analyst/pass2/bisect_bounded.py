import json
from falkordb import FalkorDB
from redis.exceptions import ResponseError
db=FalkorDB(host="127.0.0.1",port=16595); r=db.connection
base=json.load(open("/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/aae0e7a7-4212-4b57-9541-3e1dca2fac24/scratchpad/analyst-k098-p2/base.json"))
B=("CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score MATCH (t:Thread {threadId: seed.threadId}) "
   "CALL { WITH seed OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) WITH DISTINCT related LIMIT 5 RETURN collect(related) AS expanded } "
   "RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, score, [m IN expanded | m.text] AS relatedContext ORDER BY score ASC, msgId ASC LIMIT $limit")
for ws in ("r1","r2"):
    g=db.select_graph("ws:"+ws); p={"qVec":base,"k":10,"limit":10}; lo,hi=1,1<<27
    while hi-lo>1024:
        m=(lo+hi)//2; r.execute_command("GRAPH.CONFIG","SET","QUERY_MEM_CAPACITY",str(m))
        try: g.ro_query(B,p); hi=m
        except ResponseError as e: assert "mem consumption" in str(e); lo=m
    print(json.dumps({"ws":ws,"bounded_min_cap_KiB":round(hi/1024)}))
r.execute_command("GRAPH.CONFIG","SET","QUERY_MEM_CAPACITY","67108864")
