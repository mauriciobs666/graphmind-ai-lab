"""U12: drive the REAL Repository.hybrid_search (text captured from the method, both
variants) on a hub-seed + deep-thread fixture in a throwaway container (argv[1] port).
Reports: rows under QUERY_MEM_CAPACITY 64 MiB, latency, bisected min cap; and the
pre-U12 text (M0, from git HEAD) under the same cap, for contrast."""
import subprocess, sys, time, random, pathlib, re
HERE = pathlib.Path(__file__).resolve().parent
SERVER = HERE.parents[3] / "server"
sys.path.insert(0, str(SERVER))
from falkordb import FalkorDB
from redis.exceptions import ResponseError
from falkorchat.repository import Repository

PORT = int(sys.argv[1])
db = FalkorDB(host="127.0.0.1", port=PORT); r = db.connection
assert PORT != 6379
WS = "u12"; g = db.select_graph(f"ws:{WS}")
q = lambda c, p=None: g.query(c, p or {}, timeout=600000)
DIM, B, FAN, DEPTH = 1024, 5000, 100_000, 100_000
random.seed(12)
for d in ["CREATE INDEX FOR (n:Thread) ON (n.threadId)", "CREATE INDEX FOR (n:Message) ON (n.msgId)",
          "CREATE INDEX FOR (n:Channel) ON (n.channelId)", "CREATE INDEX FOR (n:Entity) ON (n.entityId)",
          f"CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding) OPTIONS {{dimension:{DIM}, similarityFunction:'cosine'}}"]:
    q(d)
t0 = time.time()
q("CREATE (:Channel {channelId:'c1'})-[:HAS_THREAD]->(:Thread {threadId:'deep'}), (:Entity {entityId:'hub'})")
for s in range(0, DEPTH, B):   # 100k-deep thread, every message mentioning the hub (fan-in 100k)
    e = min(DEPTH, s + B)
    q("MATCH (h:Entity {entityId:'hub'}) UNWIND range($s,$e-1) AS i "
      "CREATE (m:Message {msgId:'d-'+toString(i), threadId:'deep', role:'user', text:'filler '+toString(i)})-[:MENTIONS]->(h)", {"s": s, "e": e})
    q("UNWIND range($s,$e-1) AS i MATCH (a:Message {msgId:'d-'+toString(i)}), (b:Message {msgId:'d-'+toString(i+1)}) CREATE (a)-[:NEXT]->(b)",
      {"s": max(s - 1, 0), "e": e})
q("MATCH (t:Thread {threadId:'deep'}), (h:Message {msgId:'d-0'}) CREATE (t)-[:HEAD]->(h)")
base = [random.random() - 0.5 for _ in range(DIM)]
near = lambda: [x + random.gauss(0, 0.001) for x in base]
prev = f"d-{DEPTH-1}"
for j in range(10):            # 10 clustered seeds at the deep thread's tail, each on the hub
    sid = f"s{j}"
    q("MATCH (p:Message {msgId:$p}), (h:Entity {entityId:'hub'}) CREATE (p)-[:NEXT]->(m:Message {msgId:$s, threadId:'deep', role:'user', "
      "text:'seed '+$s, embedding:vecf32($v)})-[:MENTIONS]->(h)", {"p": prev, "s": sid, "v": near()})
    prev = sid
for j in range(10):            # 10 spread seeds, own short threads in c2, same hub
    q("MERGE (c:Channel {channelId:'c2'}) CREATE (c)-[:HAS_THREAD]->(t:Thread {threadId:$t})-[:HEAD]->"
      "(m:Message {msgId:$s, threadId:$t, role:'user', text:'seed '+$s, embedding:vecf32($v)}) "
      "WITH m MATCH (h:Entity {entityId:'hub'}) CREATE (m)-[:MENTIONS]->(h)", {"t": f"sp{j}", "s": f"p{j}", "v": near()})
print(f"built ws:{WS}: depth {DEPTH}, hub fan-in {FAN + 20}, 20 seeds, {time.time()-t0:.0f}s")

repo = Repository(db)
def method_text(channel_id):
    sent = []
    class Rec:
        def __init__(s, gr): s.g = gr
        def ro_query(s, c, p=None, timeout=None): sent.append((c, p)); return s.g.ro_query(c, p, timeout=timeout)
    real = repo._graph; repo._graph = lambda ws: Rec(real(ws))
    try: repo.hybrid_search(WS, q_vec=base, k=10, limit=10, channel_id=channel_id, timeout=600000)
    finally: repo._graph = real
    return sent[0]
head_src = subprocess.run(["git", "-C", str(SERVER), "show", "HEAD:falkor-chat/server/falkorchat/repository.py"],
                          capture_output=True, text=True, check=True).stdout
assert "HEAD|NEXT*0.." in head_src
def old_text(scoped):
    return ("CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
            "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) " + ("MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t) " if scoped else "")
            + "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
            "WITH seed, score, collect(DISTINCT related)[..5] AS expanded "
            "RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, score, [m IN expanded | m.text] AS relatedContext "
            "ORDER BY score ASC LIMIT $limit")
assert "collect(DISTINCT related)[..5] AS expanded" in head_src  # M0 is HEAD's form

def setcap(v): r.execute_command("GRAPH.CONFIG", "SET", "QUERY_MEM_CAPACITY", v)
def run(c, p):
    t = time.perf_counter(); res = g.ro_query(c, p, timeout=600000); return res.result_set, (time.perf_counter() - t) * 1000
def ok(c, p, cap):
    setcap(cap)
    try: run(c, p); return True
    except ResponseError as e:
        if "mem consumption exceeded" in str(e): return False
        raise
def mincap(c, p):
    lo, hi = 64 * 1024, 256 * 1024 * 1024
    if not ok(c, p, hi): return None
    while hi - lo > 16 * 1024:
        mid = (lo + hi) // 2
        hi, lo = (mid, lo) if ok(c, p, mid) else (hi, mid)
    return hi

for label, ch, cid in [("unscoped", None, None), ("scoped c1 (clustered deep seeds)", "c1", "c1"), ("scoped c2 (spread seeds)", "c2", "c2")]:
    c, p = method_text(ch)
    print(f"\n== {label}")
    setcap(0); rows0, _ = run(c, p)
    setcap(67108864)
    lat = []
    for _ in range(5):
        rows, ms = run(c, p); lat.append(round(ms, 1))
    assert rows == rows0
    print(f"  shipped method text @64MiB: {len(rows)} rows, relatedContext sizes {[len(x[4]) for x in rows]}, latency ms x5 {lat}")
    print(f"  shipped method text min cap: {mincap(c, p)/1024:.0f} KiB")
    oc = old_text(ch is not None)
    setcap(67108864)
    try:
        orows, oms = run(oc, p); print(f"  pre-U12 text (M0) @64MiB: passed in {oms:.0f} ms")
    except ResponseError as e:
        print(f"  pre-U12 text (M0) @64MiB: FAILED: {str(e)[:80]}")
    m = mincap(oc, p); print(f"  pre-U12 text (M0) min cap: {m/1024/1024:.1f} MiB" if m else "  pre-U12 text min cap: > 256 MiB")
    setcap(0); orows, oms = run(oc, p)
    print(f"  pre-U12 text uncapped latency: {oms:.0f} ms")
    key = lambda rs: sorted((x[0], x[3]) for x in rs)
    print(f"  same seed set and scores as pre-U12: {key(orows) == key(rows)}; "
          f"same per-seed relatedContext: {dict((x[0], x[4]) for x in orows) == dict((x[0], x[4]) for x in rows)}; "
          f"shipped order == pre-U12 re-sorted by (score,msgId): {[x[0] for x in rows] == [x[0] for x in sorted(orows, key=lambda x: (x[3], x[0]))]}")
setcap(0)
print("\nQUERY_MEM_CAPACITY reset to 0 on the throwaway")
