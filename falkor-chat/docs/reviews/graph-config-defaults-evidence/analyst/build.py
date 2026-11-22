"""analyst K-098 probe: throwaway container analyst-k098-a, port 16591 only."""
import random, time, json
from falkordb import FalkorDB
PORT=16591
db=FalkorDB(host="127.0.0.1",port=PORT); r=db.connection
random.seed(7)
base=[random.random()-0.5 for _ in range(1024)]
def near(): return [x+random.gauss(0,0.002) for x in base]
def far(): return [random.random()-0.5 for _ in range(1024)]
def build_deep(G, N, HUB):
    g=db.select_graph(G); q=lambda c,p=None: g.query(c,p or {},timeout=600000)
    for d in ["CREATE INDEX FOR (n:Thread) ON (n.threadId)","CREATE INDEX FOR (n:Message) ON (n.msgId)",
              "CREATE INDEX FOR (n:Channel) ON (n.channelId)","CREATE INDEX FOR (n:Entity) ON (n.entityId)",
              "CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding) OPTIONS {dimension:1024, similarityFunction:'cosine'}"]:
        q(d)
    q("CREATE (c:Channel {channelId:'c1'})-[:HAS_THREAD]->(:Thread {threadId:'tdeep'}), (:Entity {entityId:'hub'})")
    B=5000
    for s in range(0,N,B):
        e=min(N,s+B)
        q("UNWIND range($s,$e-1) AS i CREATE (:Message {msgId:'d-'+toString(i), role:'user', text:'word word word word word word'})",{"s":s,"e":e})
        q("UNWIND range($s,$e-1) AS i WITH i WHERE i < $n-1 MATCH (x:Message {msgId:'d-'+toString(i)}), (y:Message {msgId:'d-'+toString(i+1)}) CREATE (x)-[:NEXT]->(y)",{"s":max(s-1,0),"e":e,"n":N})
        q("UNWIND range($s,$e-1) AS i WITH i WHERE i >= $n-$hub MATCH (x:Message {msgId:'d-'+toString(i)}), (h:Entity {entityId:'hub'}) CREATE (x)-[:MENTIONS]->(h)",{"s":s,"e":e,"n":N,"hub":HUB})
    q("MATCH (t:Thread {threadId:'tdeep'}), (h:Message {msgId:'d-0'}) CREATE (t)-[:HEAD]->(h)")
    for i in range(N-10,N):
        q("MATCH (m:Message {msgId:$id}) SET m.embedding=vecf32($v)",{"id":f"d-{i}","v":near()})
    for t in range(50):
        q("MATCH (c:Channel {channelId:'c1'}) CREATE (c)-[:HAS_THREAD]->(th:Thread {threadId:$t}) CREATE (th)-[:HEAD]->(:Message {msgId:$t+'-0', threadId:$t, role:'user', text:'noise', embedding: vecf32($v)})",{"t":f"n{t}","v":far()})
def build_fan(G, F):
    g=db.select_graph(G); q=lambda c,p=None: g.query(c,p or {},timeout=600000)
    for d in ["CREATE INDEX FOR (n:Thread) ON (n.threadId)","CREATE INDEX FOR (n:Message) ON (n.msgId)",
              "CREATE INDEX FOR (n:Entity) ON (n.entityId)",
              "CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding) OPTIONS {dimension:1024, similarityFunction:'cosine'}"]:
        q(d)
    q("CREATE (:Entity {entityId:'hub'})")
    for s in range(0,F,5000):
        q("UNWIND range($s,$e-1) AS i MATCH (h:Entity {entityId:'hub'}) CREATE (:Message {msgId:'f-'+toString(i), role:'user', text:'word word word word word word'})-[:MENTIONS]->(h)",{"s":s,"e":min(F,s+5000)})
    for t in range(10):
        q("MATCH (h:Entity {entityId:'hub'}) CREATE (th:Thread {threadId:$t})-[:HEAD]->(m:Message {msgId:$t+'-0', role:'user', text:'seed', embedding: vecf32($v)})-[:MENTIONS]->(h)",{"t":f"s{t}","v":near()})
    for t in range(50):
        q("CREATE (th:Thread {threadId:$t})-[:HEAD]->(:Message {msgId:$t+'-0', role:'user', text:'noise', embedding: vecf32($v)})",{"t":f"n{t}","v":far()})
t0=time.time()
build_deep("ws:r1", 100000, 50000); print("r1", round(time.time()-t0,1), flush=True)
build_fan("ws:r2", 100000); print("r2", round(time.time()-t0,1), flush=True)
json.dump(base, open("/tmp/claude-1000/-home-<user>-prg-graphmind-ai-lab/aae0e7a7-4212-4b57-9541-3e1dca2fac24/scratchpad/analyst-k098/base.json","w"))
