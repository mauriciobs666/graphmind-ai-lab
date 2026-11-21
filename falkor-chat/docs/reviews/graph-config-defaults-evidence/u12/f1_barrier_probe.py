"""F1 check (throwaway port argv[1] only): the REAL method's scoped text (WITH barrier)
vs the same text without the barrier (M15), on a channel holding N threads plus a decoy
channel. Reports plan anchor, latency x5 and row equality, for N in 1k/100k/1M."""
import sys, time, random, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "server"))
from falkordb import FalkorDB
from falkorchat.repository import Repository
PORT = int(sys.argv[1]); assert PORT != 6379
db = FalkorDB(host="127.0.0.1", port=PORT); WS = "f1"; g = db.select_graph(f"ws:{WS}")
q = lambda c, p=None: g.query(c, p or {}, timeout=600000)
DIM = 16; random.seed(1)
for d in ["CREATE INDEX FOR (n:Thread) ON (n.threadId)", "CREATE INDEX FOR (n:Message) ON (n.msgId)",
          "CREATE INDEX FOR (n:Channel) ON (n.channelId)", "CREATE INDEX FOR (n:Entity) ON (n.entityId)",
          f"CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding) OPTIONS {{dimension:{DIM}, similarityFunction:'cosine'}}"]:
    q(d)
q("CREATE (:Channel {channelId:'c1'}), (:Channel {channelId:'c2'}), (:Entity {entityId:'h'})")
base = [random.random() - 0.5 for _ in range(DIM)]
for j in range(10):   # 10 seeds: even ones in c1, odd in c2 (decoys for the c1 scope), all on hub h
    q("MATCH (c:Channel {channelId:$c}), (h:Entity {entityId:'h'}) CREATE (c)-[:HAS_THREAD]->(t:Thread {threadId:$t})-[:HEAD]->"
      "(m:Message {msgId:$s, threadId:$t, role:'user', text:$s, embedding:vecf32($v)})-[:MENTIONS]->(h)",
      {"c": "c1" if j % 2 == 0 else "c2", "t": f"st{j}", "s": f"s{j}", "v": [x + random.gauss(0, 0.001) for x in base]})
repo = Repository(db)
def sent(ch):
    out = []
    class Rec:
        def __init__(s, gr): s.g = gr
        def ro_query(s, c, p=None, timeout=None): out.append((c, p)); return s.g.ro_query(c, p, timeout=timeout)
    real = repo._graph; repo._graph = lambda ws: Rec(real(ws))
    try: repo.hybrid_search(WS, q_vec=base, k=10, limit=10, channel_id=ch, timeout=600000)
    finally: repo._graph = real
    return out[0]
c_ship, p = sent("c1")
assert "WITH seed, score, t " in c_ship
c_m15 = c_ship.replace("WITH seed, score, t ", "")
have = 10
for N in [1_000, 100_000, 1_000_000]:
    for s in range(have, N, 50_000):
        q("MATCH (c:Channel {channelId:'c1'}) UNWIND range($a,$b-1) AS i CREATE (c)-[:HAS_THREAD]->(:Thread {threadId:'f'+toString(i)})",
          {"a": s, "b": min(N, s + 50_000)})
    have = N
    print(f"\n== threads in c1 ~ {N}")
    rows = {}
    for name, c in [("shipped (WITH barrier)", c_ship), ("M15 (no barrier)", c_m15)]:
        anchor = [l.strip() for l in g.explain(c, p).plan if "Index Scan" in l]
        lat = []
        for _ in range(5):
            t = time.perf_counter(); rs = g.ro_query(c, p, timeout=600000).result_set; lat.append(round((time.perf_counter() - t) * 1000, 1))
        rows[name] = rs
        print(f"  {name:24} anchors {anchor} | ms x5 {lat} | seeds {[r[0] for r in rs]}")
    a, b = rows.values()
    print(f"  rows identical: {a == b}; all seeds in c1: {all(int(r[0][1:]) % 2 == 0 for r in a)}")
