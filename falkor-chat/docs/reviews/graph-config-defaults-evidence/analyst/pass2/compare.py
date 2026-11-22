"""analyst Pass 2: current repository.hybrid_search vs the note's §3.1 bounded text. Container analyst-k098-p2 (16595) only."""
import json, random, time, sys
from falkordb import FalkorDB
from redis.exceptions import ResponseError
from falkorchat.repository import Repository
db=FalkorDB(host="127.0.0.1",port=16595); r=db.connection; repo=Repository(db)
base=json.load(open("/tmp/claude-1000/-home-<user>-prg-graphmind-ai-lab/aae0e7a7-4212-4b57-9541-3e1dca2fac24/scratchpad/analyst-k098-p2/base.json"))
def cap(b): r.execute_command("GRAPH.CONFIG","SET","QUERY_MEM_CAPACITY",str(b))
def bounded(scoped):
    return ("CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
            "MATCH (t:Thread {threadId: seed.threadId}) "
            + ("MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t) " if scoped else "")
            + "CALL { WITH seed "
            "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
            "WITH DISTINCT related LIMIT 5 "
            "RETURN collect(related) AS expanded } "
            "RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, "
            "score, [m IN expanded | m.text] AS relatedContext "
            "ORDER BY score ASC, msgId ASC LIMIT $limit")
def ids_cur(ws, ch):
    # same query text as repository.hybrid_search but returning related msgIds (to compare WHICH 5)
    q=("CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
       "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) "+("MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t) " if ch else "")+
       "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
       "WITH seed, score, collect(DISTINCT related)[..5] AS expanded "
       "RETURN seed.msgId AS msgId, score, [m IN expanded | m.msgId] AS rel ORDER BY score ASC LIMIT $limit")
    return q
def ids_bnd(ch):
    return bounded(ch).replace("[m IN expanded | m.text] AS relatedContext","[m IN expanded | m.msgId] AS relatedContext").replace("seed.text AS text, seed.role AS role, ","")
out=[]
for ws, ch in (("r1",None),("r1","c1"),("r2",None),("r5",None),("r5","cx")):
    g=db.select_graph("ws:"+ws); p={"qVec":base,"k":10,"limit":10}
    if ch: p["channelId"]=ch
    cap(0)
    t=time.perf_counter(); cur=repo.hybrid_search(ws,q_vec=base,k=10,limit=10,channel_id=ch,timeout=600000); tc=time.perf_counter()-t
    curids=[(row[0],row[1],row[2]) for row in g.ro_query(ids_cur(ws,ch),p,timeout=600000).result_set]
    cap(67108864)
    t=time.perf_counter()
    try:
        b=g.ro_query(bounded(bool(ch)),p,timeout=5000).result_set; tb=time.perf_counter()-t; err=None
        bids=[(row[0],row[1],row[2]) for row in g.ro_query(ids_bnd(bool(ch)),p,timeout=5000).result_set]
    except ResponseError as e: b=[]; bids=[]; tb=None; err=str(e)
    curt=[(x["msgId"],x["score"],x["relatedContext"]) for x in cur]
    bt=[(row[0],row[3],row[4]) for row in b]
    rec={"ws":ws,"channel":ch,"current_s":round(tc,3),"bounded_s":round(tb,3) if tb else None,"bounded_err_at_64MiB":err,
         "n_current":len(curt),"n_bounded":len(bt),
         "same_seed_set":sorted(x[0] for x in curt)==sorted(x[0] for x in bt),
         "same_scores":sorted(x[1] for x in curt)==sorted(x[1] for x in bt),
         "same_order":[x[0] for x in curt]==[x[0] for x in bt],
         "same_relatedContext_text_per_seed":{x[0]:x[2] for x in curt}=={x[0]:x[2] for x in bt},
         "same_related_ids_per_seed":{x[0]:x[2] for x in curids}=={x[0]:x[2] for x in bids},
         "sample_related_ids_current":curids[0][2] if curids else None,"sample_related_ids_bounded":bids[0][2] if bids else None}
    print(json.dumps(rec),flush=True)
cap(67108864)
