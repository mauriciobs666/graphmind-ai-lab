"""K-098 r4 (U12 gate F1): threads-per-channel axis for ws:cov, THROWAWAY container on port 16483 only.

Run after build_cov.py, in the same cwd (it extends that cwd's cov_scenarios.json). For each N in
1k/10k/100k/1M: a Channel tpc<N> with N Threads. Ten of them hold one seed message each (HEAD ->
seed, mentioning the 'small' entity); the rest are empty. The seeds' embeddings are near-duplicates
of the scenario's base vector, so the ANN query returns exactly those 10 seeds for k <= 10.
"""
import json, random
from falkordb import FalkorDB

g = FalkorDB(host="127.0.0.1", port=16483).select_graph("ws:cov")
q = lambda c, p=None: g.query(c, p or {}, timeout=600000)
DIM, B = 1024, 50000
scen = json.load(open("cov_scenarios.json"))
for label, n in (("1k", 1000), ("10k", 10000), ("100k", 100000), ("1M", 1000000)):
    ch = f"tpc{label}"
    q("CREATE (:Channel {channelId:$c})", {"c": ch})
    for s in range(0, n, B):
        q("MATCH (c:Channel {channelId:$c}) UNWIND range($s,$e-1) AS i "
          "CREATE (c)-[:HAS_THREAD]->(:Thread {threadId:$c+'-t'+toString(i)})", {"c": ch, "s": s, "e": min(n, s + B)})
    v = [random.random() - 0.5 for _ in range(DIM)]
    for j in range(10):
        tid = f"{ch}-t{j * (n // 10)}"          # seeds spread across the channel's threads
        q("MATCH (t:Thread {threadId:$t}), (h:Entity {entityId:'small'}) "
          "CREATE (t)-[:HEAD]->(m:Message {msgId:$m, threadId:$t, role:'user', text:'seed', embedding:vecf32($v)})-[:MENTIONS]->(h)",
          {"t": tid, "m": f"{ch}-s{j}", "v": [x + random.gauss(0, 0.001) for x in v]})
    scen[f"threads per channel {label}"] = v
    print(ch, g.ro_query("MATCH (:Channel {channelId:$c})-[:HAS_THREAD]->(t) RETURN count(t)", {"c": ch}).result_set, flush=True)
json.dump(scen, open("cov_scenarios.json", "w"))
