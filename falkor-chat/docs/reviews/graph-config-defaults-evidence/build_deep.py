"""K-098: ws:deep in the probe container only — one 100k-message thread (only the last 20 embedded),
a hub Entity mentioned by the last 50k messages, plus 2k embedded noise messages in short threads."""
import random, time
from falkordb import FalkorDB
g = FalkorDB(host="127.0.0.1", port=16479).select_graph("ws:deep")
q = lambda c, p=None: g.query(c, p or {}, timeout=600000)
for d in ["CREATE INDEX FOR (n:Thread) ON (n.threadId)", "CREATE INDEX FOR (n:Message) ON (n.msgId)",
          "CREATE INDEX FOR (n:Channel) ON (n.channelId)", "CREATE INDEX FOR (n:Entity) ON (n.entityId)",
          "CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding) OPTIONS {dimension:1024, similarityFunction:'cosine'}"]:
    q(d)
N, HUB, B = 100000, 50000, 2000
q("CREATE (c:Channel {channelId:'c1'})-[:HAS_THREAD]->(:Thread {threadId:'tdeep'}), (:Entity {entityId:'hub'})")
t0 = time.time()
for s in range(0, N, B):
    e = min(N, s + B)
    q("UNWIND range($s, $e-1) AS i CREATE (:Message {msgId:'d-'+toString(i), seq:i, threadId:'tdeep', role:'user', "
      "text: reduce(s='', x IN range(1,20) | s + 'word ')})", {"s": s, "e": e})
    q("UNWIND range($s, $e-1) AS i WITH 'd-'+toString(i) AS a, 'd-'+toString(i+1) AS b, i WHERE i < $n-1 "
      "MATCH (x:Message {msgId:a}), (y:Message {msgId:b}) CREATE (x)-[:NEXT]->(y)", {"s": max(s - 1, 0), "e": e, "n": N})
    q("UNWIND range($s, $e-1) AS i WITH 'd-'+toString(i) AS a, i WHERE i >= $n - $hub "
      "MATCH (x:Message {msgId:a}), (h:Entity {entityId:'hub'}) CREATE (x)-[:MENTIONS]->(h)", {"s": s, "e": e, "n": N, "hub": HUB})
q("MATCH (t:Thread {threadId:'tdeep'}), (h:Message {msgId:'d-0'}) CREATE (t)-[:HEAD]->(h)")
q("UNWIND range($n-20, $n-1) AS i MATCH (m:Message {msgId:'d-'+toString(i)}) SET m.embedding = vecf32([x IN range(1,1024) | rand()-0.5])", {"n": N})
for t in range(100):
    q("MATCH (c:Channel {channelId:'c1'}) CREATE (c)-[:HAS_THREAD]->(th:Thread {threadId:$t}) "
      "WITH th UNWIND range(0,19) AS i CREATE (m:Message {msgId:$t+'-'+toString(i), threadId:$t, role:'user', text:'noise', "
      "embedding: vecf32([x IN range(1,1024) | rand()-0.5])}) WITH th, collect(m) AS ms "
      "CREATE (th)-[:HEAD]->(ms[0]) WITH ms UNWIND range(0,18) AS j WITH ms[j] AS a, ms[j+1] AS b CREATE (a)-[:NEXT]->(b)", {"t": f"n{t}"})
print("built", round(time.time() - t0, 1), g.ro_query("MATCH (m:Message) RETURN count(m)").result_set,
      g.ro_query("MATCH ()-[r]->() RETURN type(r), count(*)").result_set,
      g.ro_query("MATCH (t:Thread {threadId:'tdeep'})-[:HEAD]->(h) MATCH p=(h)-[:NEXT*]->(z:Message {msgId:'d-99999'}) RETURN length(p)").result_set)
