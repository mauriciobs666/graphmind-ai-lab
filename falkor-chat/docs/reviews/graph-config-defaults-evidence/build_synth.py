"""K-098: build a synthetic GraphRAG workspace in the THROWAWAY probe container only.

Shape (deliberately skewed past anything live, see note §3.5):
  - 1 Channel; 200 short threads x 50 msgs + 1 long thread of LONG msgs
  - every Message: ~200-char text, 1024-dim vecf32 embedding, threadId, createdAt
  - 2,000 Entities; each Message MENTIONS 2 random entities; one hub entity
    mentioned by HUB messages (future extraction pipeline, worst case for
    hybrid_search's collect(DISTINCT related))
  - CHUNKS Chunks: ~700-char text, 1024-dim embedding, documentCurrent (80% true)
Port is hard-wired to the probe container (16479), never the shared 6379.
"""
import random, sys, time
from falkordb import FalkorDB

PORT = 16479
G = "ws:synth"
DIM = 1024
SHORT_THREADS, SHORT_LEN = 200, 50
LONG = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
HUB = 5000
CHUNKS = 5000
ENT = 2000
B = 250

db = FalkorDB(host="127.0.0.1", port=PORT)
g = db.select_graph(G)
q = lambda c, p=None: g.query(c, p or {}, timeout=600000)

ddl = [
    "CREATE INDEX FOR (n:Channel) ON (n.channelId)",
    "CREATE INDEX FOR (n:Thread) ON (n.threadId)",
    "CREATE INDEX FOR (n:Message) ON (n.msgId)",
    "CREATE INDEX FOR (n:Message) ON (n.createdAt)",
    "CREATE INDEX FOR (n:Chunk) ON (n.chunkId)",
    "CREATE INDEX FOR (n:Entity) ON (n.entityId)",
    f"CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding) OPTIONS {{dimension:{DIM}, similarityFunction:'cosine'}}",
    f"CREATE VECTOR INDEX FOR (n:Chunk) ON (n.embedding) OPTIONS {{dimension:{DIM}, similarityFunction:'cosine'}}",
]
for d in ddl:
    q(d)
q("CALL db.idx.fulltext.createNodeIndex('Message', 'text')")

t0 = time.time()
q("CREATE (:Channel {channelId:'c1'})")
q("UNWIND range(0, $n-1) AS i CREATE (:Entity {entityId:'e'+toString(i), name:'entity '+toString(i)})", {"n": ENT})

threads = [(f"t{i}", SHORT_LEN) for i in range(SHORT_THREADS)] + [("tlong", LONG)]
ts = 0
for tid, n in threads:
    q("MATCH (c:Channel {channelId:'c1'}) CREATE (c)-[:HAS_THREAD]->(:Thread {threadId:$t})", {"t": tid})
    for s in range(0, n, B):
        e = min(n, s + B)
        rows = []
        for i in range(s, e):
            ts += 1
            ents = random.sample(range(1, ENT), 2)
            if tid == "tlong" and i >= n - HUB:  # tail of the long thread mentions the hub
                ents[0] = 0
            rows.append({"id": f"{tid}-{i}", "ts": ts, "e1": f"e{ents[0]}", "e2": f"e{ents[1]}"})
        q(
            "UNWIND $rows AS r "
            "CREATE (m:Message {msgId:r.id, threadId:$t, createdAt:r.ts, role:'user', "
            "text: reduce(s='', x IN range(1,20) | s + 'word' + toString(x) + ' '), "
            "embedding: vecf32([x IN range(1,$dim) | rand()-0.5])}) "
            "WITH m, r MATCH (e1:Entity {entityId:r.e1}), (e2:Entity {entityId:r.e2}) "
            "CREATE (m)-[:MENTIONS]->(e1), (m)-[:MENTIONS]->(e2)",
            {"rows": rows, "t": tid, "dim": DIM},
        )
        pairs = [{"a": f"{tid}-{i}", "b": f"{tid}-{i+1}"} for i in range(max(s - 1, 0), e - 1)]
        if pairs:
            q("UNWIND $p AS r MATCH (a:Message {msgId:r.a}), (b:Message {msgId:r.b}) CREATE (a)-[:NEXT]->(b)", {"p": pairs})
    q("MATCH (t:Thread {threadId:$t}), (h:Message {msgId:$h}), (z:Message {msgId:$z}) "
      "CREATE (t)-[:HEAD]->(h), (t)-[:TAIL]->(z)", {"t": tid, "h": f"{tid}-0", "z": f"{tid}-{n-1}"})
    print(tid, n, round(time.time() - t0, 1), flush=True)

for s in range(0, CHUNKS, B):
    rows = [{"id": f"ch{i}", "cur": random.random() < 0.8, "doc": f"d{i // 10}",
             "e": f"e{random.randrange(ENT)}"} for i in range(s, min(CHUNKS, s + B))]
    q(
        "UNWIND $rows AS r "
        "CREATE (c:Chunk {chunkId:r.id, documentId:r.doc, seq:0, documentCurrent:r.cur, "
        "text: reduce(s='', x IN range(1,70) | s + 'chunkword' + ' '), "
        "embedding: vecf32([x IN range(1,$dim) | rand()-0.5])}) "
        "WITH c, r MATCH (e:Entity {entityId:r.e}) CREATE (c)-[:ABOUT]->(e)",
        {"rows": rows, "dim": DIM},
    )
print("chunks", CHUNKS, round(time.time() - t0, 1))
print(g.ro_query("MATCH (m:Message) RETURN count(m)").result_set,
      g.ro_query("MATCH (c:Chunk) RETURN count(c)").result_set,
      g.ro_query("MATCH ()-[r]->() RETURN type(r), count(*)").result_set)
