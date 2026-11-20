import json
from falkordb import FalkorDB
from falkorchat.repository import Repository
db=FalkorDB(host="127.0.0.1",port=16595); r=db.connection; repo=Repository(db)
base=json.load(open("/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/aae0e7a7-4212-4b57-9541-3e1dca2fac24/scratchpad/analyst-k098-p2/base.json"))
B=("CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score MATCH (t:Thread {threadId: seed.threadId}) "
   "CALL { WITH seed OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) WITH DISTINCT related LIMIT 5 RETURN collect(related) AS expanded } "
   "RETURN seed.msgId AS msgId, score ORDER BY score ASC, msgId ASC LIMIT $limit")
r.execute_command("GRAPH.CONFIG","SET","QUERY_MEM_CAPACITY","0")
for ws in ("r1","r5"):
    cur=[(x["msgId"],x["score"]) for x in repo.hybrid_search(ws,q_vec=base,k=10,limit=10,timeout=600000)]
    cur2=[(x["msgId"],x["score"]) for x in repo.hybrid_search(ws,q_vec=base,k=10,limit=10,timeout=600000)]
    b=[tuple(row) for row in db.select_graph("ws:"+ws).ro_query(B,{"qVec":base,"k":10,"limit":10}).result_set]
    print(ws,"current:",cur); print(ws,"current again same:",cur==cur2); print(ws,"bounded:",b)
    # is the bounded order exactly current's order with ties broken by msgId?
    exp=sorted(cur,key=lambda x:(x[1],x[0])); print(ws,"bounded == current re-sorted by (score,msgId):",exp==b, "| current already sorted by score:", [s for _,s in cur]==sorted(s for _,s in cur))
    # test limit<k: does top-limit set differ?
for lim in (3,5):
    cur=[x["msgId"] for x in repo.hybrid_search("r5",q_vec=base,k=10,limit=lim,timeout=600000)]
    b=[row[0] for row in db.select_graph("ws:r5").ro_query(B,{"qVec":base,"k":10,"limit":lim}).result_set]
    print("r5 limit",lim,"current",cur,"bounded",b)
r.execute_command("GRAPH.CONFIG","SET","QUERY_MEM_CAPACITY","67108864")
