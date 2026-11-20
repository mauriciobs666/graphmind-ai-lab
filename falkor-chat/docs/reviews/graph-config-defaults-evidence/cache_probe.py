"""K-098 CACHE_SIZE probe — validation container (port 16482) only.
(1) plan-cache miss vs hit wall time for repository.py-shaped texts (alias suffix forces a distinct text);
(2) used_memory delta for N distinct parameterized texts pushed through every worker thread."""
import json, random, statistics, sys, threading, time
from falkordb import FalkorDB
db = FalkorDB(host="127.0.0.1", port=16482); r = db.connection; g = db.select_graph("ws:synth")
HY = ("CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
      "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
      "WITH seed, score, collect(DISTINCT related)[..5] AS expanded RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, "
      "score, [m IN expanded | m.text] AS relatedContext ORDER BY score ASC LIMIT $limit")
RT = ("MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) "
      "RETURN m.msgId AS msgId, m.text AS text, m.createdAt AS createdAt ORDER BY m.createdAt")
GET = "MATCH (m:Message {msgId: $id}) RETURN m.text AS t"
vec = [random.random() - 0.5 for _ in range(1024)]
out = {}
for name, base, p in (("get_message", GET, {"id": "t1-1"}), ("read_thread(50)", RT, {"threadId": "t3"}),
                      ("hybrid_msg", HY, {"qVec": vec, "k": 10, "limit": 10})):
    miss, hit = [], []
    for i in range(20):
        q = base.replace(" AS t", f" AS t{i}r{random.randrange(10**9)}", 1) if name == "get_message" else base.replace(" AS msgId", f" AS msgId{i}r{random.randrange(10**9)}", 1)  # new alias -> new text -> forced miss
        t = time.perf_counter(); res = g.ro_query(q, p, timeout=5000); miss.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter(); res2 = g.ro_query(q, p, timeout=5000); hit.append((time.perf_counter() - t) * 1000)
    out[name] = {"miss_ms_median": round(statistics.median(miss), 2), "hit_ms_median": round(statistics.median(hit), 2),
                 "first_cached_flag": res.cached_execution, "second_cached_flag": res2.cached_execution}
print(json.dumps(out)); sys.stdout.flush()
def used(): return r.info("memory")["used_memory"]
N = 100
texts = [RT.replace(" AS msgId", f" AS msgId{n}", 1) for n in range(N)]
assert g.ro_query(texts[0], {"threadId": "t3"}).result_set
time.sleep(1); b = used()
def worker():
    for _ in range(4):
        for t in texts: g.ro_query(t, {"threadId": "t3"}, timeout=5000)
ts = [threading.Thread(target=worker) for _ in range(16)]; [x.start() for x in ts]; [x.join() for x in ts]
time.sleep(2); a = used()
print(json.dumps({"distinct_texts": N, "used_memory_delta_bytes": a - b, "per_text_bytes": (a - b) // N,
                  "per_text_per_thread_upper_bytes_if_8_thread_caches": (a - b) // (N * 8)}))
