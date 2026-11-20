import json, time
from falkordb import FalkorDB
from redis.exceptions import ResponseError
from falkorchat.repository import Repository
db=FalkorDB(host="127.0.0.1",port=16591); r=db.connection
r.execute_command("GRAPH.CONFIG","SET","QUERY_MEM_CAPACITY","67108864")
def run(g,c,p=None):
    t=time.perf_counter()
    try: res=g.query(c,p or {}); return "ok "+str(res.result_set[:1])[:120], round(time.perf_counter()-t,2)
    except ResponseError as e: return "ERR "+str(e), round(time.perf_counter()-t,2)
# 1. backfill_thread_ids.sh verbatim whole-graph query on ws:r1 (100k msgs, threadId null)
g=db.select_graph("ws:r1")
BF="MATCH (t:Thread)-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) WHERE m.threadId IS NULL SET m.threadId = t.threadId RETURN count(m) AS backfilled"
print("backfill_thread_ids @64MiB, 100k msgs:", run(g,BF), "nulls after:", g.ro_query("MATCH (m:Message) WHERE m.threadId IS NULL RETURN count(m)").result_set)
# 2. reset_all_participants verbatim
cy=Repository._RESET_ALL_PARTICIPANTS_CYPHER
for P,M in ((1000,10),(5000,10),(10000,10)):
    G=f"ws:r3-{P}"; g=db.select_graph(G)
    g.query("CREATE INDEX FOR (n:Message) ON (n.msgId)")
    for s in range(0,P,1000):
        g.query("UNWIND range($s,$e-1) AS i CREATE (u:User {userId:'p'+toString(i), tokenHash:'h'})-[:MEMBER_OF]->(c:Channel {channelId:'c'+toString(i), participantId:'p'+toString(i)})-[:HAS_THREAD]->(t:Thread {threadId:'t'+toString(i)}) "
                "WITH t, i UNWIND range(0,$m-1) AS j CREATE (m:Message {msgId:'m'+toString(i)+'-'+toString(j), seq:j, text:'hello there this is a storefront message'}) "
                "WITH t, i, collect(m) AS ms CREATE (t)-[:HEAD]->(ms[0]) WITH ms UNWIND range(0,size(ms)-2) AS k WITH ms[k] AS a, ms[k+1] AS b CREATE (a)-[:NEXT]->(b)",{"s":s,"e":min(P,s+1000),"m":M})
    before=g.ro_query("MATCH (n) RETURN count(n)").result_set[0][0]
    out=run(g,cy)
    after=g.ro_query("MATCH (n) RETURN count(n)").result_set[0][0]
    print(f"reset_all @64MiB P={P} msgs={P*M}: {out} nodes before={before} after={after}", flush=True)
