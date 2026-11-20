"""K-098: bisect the smallest QUERY_MEM_CAPACITY each GraphRAG read survives.

THROWAWAY probe container only (port 16479). The query texts are copied verbatim
from falkor-chat/server/falkorchat/repository.py (hybrid_search, search_chunks).
Each read runs as GRAPH.RO_QUERY via falkordb-py's ro_query, timeout 60000 ms.
Output: one JSON line per variant -> measure_qmc.jsonl
"""
import json, random, sys, time
from falkordb import FalkorDB
from redis.exceptions import ResponseError

PORT = 16479
db = FalkorDB(host="127.0.0.1", port=PORT)
r = db.connection
g = db.select_graph("ws:synth")

HYBRID = lambda scoped: (
    "CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) "
    "YIELD node AS seed, score "
    "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) "
    + ("MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t) " if scoped else "")
    + "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
    "WITH seed, score, collect(DISTINCT related)[..5] AS expanded "
    "RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, "
    "score, [m IN expanded | m.text] AS relatedContext "
    "ORDER BY score ASC "
    "LIMIT $limit"
)
CHUNKS = (
    "CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($qVec)) "
    "YIELD node AS seed, score "
    "WHERE seed.documentCurrent = true "
    "RETURN seed.chunkId AS chunkId, seed.text AS text, "
    "seed.documentId AS documentId, seed.seq AS seq, score "
    "ORDER BY score ASC "
    "LIMIT $limit"
)

def vec_of(msg_id):
    # read the stored embedding back as a plain list (the "query = an existing message" case)
    row = g.ro_query("MATCH (m:Message {msgId:$id}) RETURN m.embedding", {"id": msg_id}).result_set[0][0]
    return [float(x) for x in row]

rand_vec = [random.random() - 0.5 for _ in range(1024)]
deep_vec = vec_of("tlong-9999")   # last message of the 10k thread; mentions the hub entity
mid_vec = vec_of("tlong-5000")

variants = [
    ("hybrid_msg ws-wide k=10 random seed", HYBRID(False), {"qVec": rand_vec, "k": 10, "limit": 10}),
    ("hybrid_msg ws-wide k=10 seed=tail of 10k thread + hub", HYBRID(False), {"qVec": deep_vec, "k": 10, "limit": 10}),
    ("hybrid_msg channel-scoped k=10 seed=tail of 10k thread + hub", HYBRID(True), {"qVec": deep_vec, "k": 10, "limit": 10, "channelId": "c1"}),
    ("hybrid_msg ws-wide k=10 seed=mid of 10k thread + hub", HYBRID(False), {"qVec": mid_vec, "k": 10, "limit": 10}),
    ("search_chunks k=30 limit=10 (hybrid_search chunk half)", CHUNKS, {"qVec": rand_vec, "k": 30, "limit": 10}),
    ("search_chunks k=600 limit=200 (search_documents max)", CHUNKS, {"qVec": rand_vec, "k": 600, "limit": 200}),
]

def set_cap(b):
    r.execute_command("GRAPH.CONFIG", "SET", "QUERY_MEM_CAPACITY", str(b))

def run(cypher, params):
    t = time.perf_counter()
    try:
        res = g.ro_query(cypher, params, timeout=60000)
        return True, len(res.result_set), (time.perf_counter() - t) * 1000, None
    except ResponseError as e:
        return False, 0, (time.perf_counter() - t) * 1000, str(e)

out = open(sys.argv[1] if len(sys.argv) > 1 else "measure_qmc.jsonl", "a")
for name, cypher, params in variants:
    set_cap(0)
    ok, rows, ms, err = run(cypher, params)        # warm the plan cache, uncapped
    ok, rows, ms, err = run(cypher, params)
    lo, hi = 1, 1 << 30                              # fail at lo, succeed at hi (1 GiB)
    set_cap(lo); f_ok, _, _, f_err = run(cypher, params)
    if f_ok:
        rec = {"variant": name, "rows": rows, "uncapped_ms": round(ms, 1), "min_cap_bytes": "<=1 (untracked)"}
    else:
        while hi - lo > 1024:                        # 1 KiB resolution
            mid = (lo + hi) // 2
            set_cap(mid)
            if run(cypher, params)[0]:
                hi = mid
            else:
                lo = mid
        rec = {"variant": name, "rows": rows, "uncapped_ms": round(ms, 1),
               "min_cap_bytes": hi, "min_cap_kib": round(hi / 1024, 1), "fail_error": f_err}
    set_cap(0)
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
set_cap(0)
