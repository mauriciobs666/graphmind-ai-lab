"""K-098 read-mix load generator for a THROWAWAY container (port argv[1]). Mix per iteration, repository.py texts:
hybrid_search Message half (k=10) + search_chunks (k=30) + read_thread of a 50-msg thread + search_messages-style
full-text. C client threads for D seconds. Prints JSON: ops/s, p50/p99 per op."""
import json, random, sys, threading, time
from falkordb import FalkorDB
port, C, D = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
db = FalkorDB(host="127.0.0.1", port=port, max_connections=C + 4) if False else FalkorDB(host="127.0.0.1", port=port)
g = db.select_graph("ws:synth")
HY = ("CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
      "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
      "WITH seed, score, collect(DISTINCT related)[..5] AS expanded RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, "
      "score, [m IN expanded | m.text] AS relatedContext ORDER BY score ASC LIMIT $limit")
CH = ("CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score WHERE seed.documentCurrent = true "
      "RETURN seed.chunkId AS chunkId, seed.text AS text, seed.documentId AS documentId, seed.seq AS seq, score ORDER BY score ASC LIMIT $limit")
RT = ("MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message) MATCH (first)-[:NEXT*0..]->(m:Message) "
      "RETURN m.msgId, m.text, m.createdAt ORDER BY m.createdAt")
FT = "CALL db.idx.fulltext.queryNodes('Message', $q) YIELD node RETURN node.msgId, node.text LIMIT 50"
vecs = [[random.random() - 0.5 for _ in range(1024)] for _ in range(64)]
lat = {k: [] for k in ("hybrid", "chunks", "thread", "fulltext")}; errs = []; lock = threading.Lock()
stop = time.time() + D
def worker():
    loc = {k: [] for k in lat}
    while time.time() < stop:
        for name, c, p in (("hybrid", HY, {"qVec": random.choice(vecs), "k": 10, "limit": 10}),
                           ("chunks", CH, {"qVec": random.choice(vecs), "k": 30, "limit": 10}),
                           ("thread", RT, {"threadId": f"t{random.randrange(200)}"}),
                           ("fulltext", FT, {"q": f"word{random.randrange(1, 21)}"})):
            t = time.perf_counter()
            try: g.ro_query(c, p, timeout=5000)
            except Exception as e:
                with lock: errs.append(str(e)[:80])
            loc[name].append((time.perf_counter() - t) * 1000)
    with lock:
        for k in loc: lat[k] += loc[k]
ts = [threading.Thread(target=worker) for _ in range(C)]
[t.start() for t in ts]; [t.join() for t in ts]
pct = lambda xs, p: round(sorted(xs)[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None
n = sum(len(v) for v in lat.values())
print(json.dumps({"clients": C, "secs": D, "ops": n, "ops_per_s": round(n / D, 1),
  **{k: {"p50": pct(v, .5), "p99": pct(v, .99)} for k, v in lat.items()}, "errors": len(errs), "err_sample": errs[:2]}))
