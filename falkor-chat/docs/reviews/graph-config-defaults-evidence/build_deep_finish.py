"""K-098: finish ws:deep (probe container only): the embedded tail (if missing) + noise threads."""
from falkordb import FalkorDB
g = FalkorDB(host="127.0.0.1", port=16479).select_graph("ws:deep")
q = lambda c, p=None: g.query(c, p or {}, timeout=600000)
N = 100000
print("existing noise threads", g.ro_query("MATCH (t:Thread) WHERE t.threadId STARTS WITH 'n' RETURN count(t)").result_set)
print("HEAD", g.ro_query("MATCH (:Thread {threadId:'tdeep'})-[:HEAD]->(h) RETURN h.msgId").result_set)
print("embedded tail", g.ro_query("MATCH (m:Message) WHERE m.threadId='tdeep' AND m.embedding IS NOT NULL RETURN count(m)").result_set)
for t in range(100):
    q("MATCH (c:Channel {channelId:'c1'}) CREATE (c)-[:HAS_THREAD]->(th:Thread {threadId:$t}) "
      "WITH th UNWIND range(0,19) AS i CREATE (m:Message {msgId:$t+'-'+toString(i), threadId:$t, role:'user', text:'noise', "
      "embedding: vecf32([x IN range(1,1024) | rand()-0.5])}) WITH th, collect(m) AS ms "
      "WITH th, ms, ms[0] AS h CREATE (th)-[:HEAD]->(h) WITH ms UNWIND range(0,18) AS j WITH ms[j] AS a, ms[j+1] AS b CREATE (a)-[:NEXT]->(b)", {"t": f"n{t}"})
print(g.ro_query("MATCH (m:Message) RETURN count(m)").result_set,
      g.ro_query("MATCH ()-[r]->() RETURN type(r), count(*)").result_set,
      g.ro_query("MATCH (t:Thread {threadId:'tdeep'})-[:HEAD]->(h) MATCH p=(h)-[:NEXT*]->(z:Message {msgId:'d-99999'}) RETURN length(p)", timeout=60000).result_set)
