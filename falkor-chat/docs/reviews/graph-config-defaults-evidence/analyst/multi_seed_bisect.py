import json, sys, time
from falkordb import FalkorDB
from redis.exceptions import ResponseError
db=FalkorDB(host="127.0.0.1",port=16591); r=db.connection
base=json.load(open("base.json"))
ANN="CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
FULL=(ANN+"MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) "
 "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
 "WITH seed, score, collect(DISTINCT related)[..5] AS expanded "
 "RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, "
 "score, [m IN expanded | m.text] AS relatedContext ORDER BY score ASC LIMIT $limit")
V={"ann only":ANN+"RETURN seed.msgId, score ORDER BY score LIMIT $limit",
   "ann+walk":ANN+"MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) RETURN seed.msgId, score ORDER BY score LIMIT $limit",
   "ann+expansion":ANN+"OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) WITH seed, score, collect(DISTINCT related)[..5] AS x RETURN seed.msgId, score, size(x) ORDER BY score LIMIT $limit",
   "full (verbatim repository.hybrid_search, ws-wide)":FULL}
def cap(b): r.execute_command("GRAPH.CONFIG","SET","QUERY_MEM_CAPACITY",str(b))
def ok(g,c,p):
    try: return True, g.ro_query(c,p,timeout=60000)
    except ResponseError as e:
        assert "mem consumption" in str(e), e; return False, None
def bisect(g,c,p):
    lo,hi=1,1<<30
    while hi-lo>1024:
        m=(lo+hi)//2; cap(m)
        if ok(g,c,p)[0]: hi=m
        else: lo=m
    return hi
for G in sys.argv[1:]:
    g=db.select_graph(G); p={"qVec":base,"k":10,"limit":10}
    cap(0); seeds=[row[0] for row in g.ro_query(V["ann only"],p).result_set]
    print(G,"seeds:",seeds, flush=True)
    for n,c in V.items():
        cap(0); g.ro_query(c,p,timeout=60000); t=time.perf_counter(); g.ro_query(c,p,timeout=60000); ms=(time.perf_counter()-t)*1000
        cap(67108864); passed,_=ok(g,c,p)
        b=bisect(g,c,p)
        print(json.dumps({"graph":G,"variant":n,"uncapped_ms":round(ms,1),"min_cap_MiB":round(b/1048576,2),"passes_at_64MiB":passed}),flush=True)
    cap(67108864)
