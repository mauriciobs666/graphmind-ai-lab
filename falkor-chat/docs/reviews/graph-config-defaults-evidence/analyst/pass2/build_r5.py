"""ws:r5: 10 near seeds (2 exact ties), 5 mention a 1000-fan-in entity, 5 mention nothing; channel cx; plus 50 far noise."""
import json, random
from falkordb import FalkorDB
db=FalkorDB(host="127.0.0.1",port=16595); g=db.select_graph("ws:r5"); q=lambda c,p=None: g.query(c,p or {},timeout=600000)
base=json.load(open("base.json")); random.seed(11)
for d in ["CREATE INDEX FOR (n:Thread) ON (n.threadId)","CREATE INDEX FOR (n:Message) ON (n.msgId)","CREATE INDEX FOR (n:Entity) ON (n.entityId)","CREATE INDEX FOR (n:Channel) ON (n.channelId)",
          "CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding) OPTIONS {dimension:1024, similarityFunction:'cosine'}"]: q(d)
q("CREATE (:Entity {entityId:'e1'}), (:Channel {channelId:'cx'})")
q("UNWIND range(0,999) AS i MATCH (e:Entity {entityId:'e1'}) CREATE (:Message {msgId:'f-'+toString(i), text:'rel '+toString(i)})-[:MENTIONS]->(e)")
tie=[x+random.gauss(0,0.002) for x in base]
for s in range(10):
    v = tie if s in (3,4) else [x+random.gauss(0,0.002) for x in base]
    q("MATCH (c:Channel {channelId:'cx'}) CREATE (c)-[:HAS_THREAD]->(t:Thread {threadId:$t})-[:HEAD]->(h:Message {msgId:$t+'-0', threadId:$t, text:'head'}) "
      "CREATE (h)-[:NEXT]->(m:Message {msgId:'z'+toString(9-$s)+'-seed', threadId:$t, role:'user', text:'seed', embedding: vecf32($v)})",{"t":f"x{s}","s":s,"v":v})
    if s % 2 == 0:
        q("MATCH (m:Message {msgId:$id}), (e:Entity {entityId:'e1'}) CREATE (m)-[:MENTIONS]->(e)",{"id":f"z{9-s}-seed"})
for t in range(50):
    q("CREATE (th:Thread {threadId:$t})-[:HEAD]->(:Message {msgId:$t+'-0', threadId:$t, role:'user', text:'noise', embedding: vecf32($v)})",{"t":f"n{t}","v":[random.random()-0.5 for _ in range(1024)]})
print("r5 built")
