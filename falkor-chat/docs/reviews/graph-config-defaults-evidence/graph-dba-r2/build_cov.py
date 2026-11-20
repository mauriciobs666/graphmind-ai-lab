"""K-098 r2 (M1): coverage graph ws:cov in the THROWAWAY container on port 16483 only.

Filler (unembedded) gives the axes their size; each scenario is 10 seed messages whose embeddings
are near-duplicates of the scenario's own base vector, so an ANN query with that base returns
exactly those seeds for k <= 10. 2,000 random embedded noise messages sit beside them.
Axes: fan-in (100/1k/10k/100k), mentions per seed (1 vs 10 x 10k hubs), depth (~20/10k/100k),
clustered (one thread) vs spread (10 threads), and the worst combination (100k depth + 100k hub).
"""
import json, random
from falkordb import FalkorDB

g = FalkorDB(host="127.0.0.1", port=16483).select_graph("ws:cov")
q = lambda c, p=None: g.query(c, p or {}, timeout=600000)
DIM, B = 1024, 5000
for d in ["CREATE INDEX FOR (n:Thread) ON (n.threadId)", "CREATE INDEX FOR (n:Message) ON (n.msgId)",
          "CREATE INDEX FOR (n:Channel) ON (n.channelId)", "CREATE INDEX FOR (n:Entity) ON (n.entityId)",
          f"CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding) OPTIONS {{dimension:{DIM}, similarityFunction:'cosine'}}"]:
    q(d)
q("CREATE (:Channel {channelId:'c1'})")
hubs = ["H100", "H1k", "H10k", "H100k"] + [f"M{i}" for i in range(10)] + ["small"]
q("UNWIND $h AS h CREATE (:Entity {entityId:h})", {"h": hubs})

def thread(tid, n, hub=None):
    """Unembedded linear thread tid of n messages (ids tid-0..tid-n-1), optionally each mentioning hub."""
    q("MATCH (c:Channel {channelId:'c1'}) CREATE (c)-[:HAS_THREAD]->(:Thread {threadId:$t})", {"t": tid})
    for s in range(0, n, B):
        e = min(n, s + B)
        q("UNWIND range($s,$e-1) AS i CREATE (:Message {msgId:$t+'-'+toString(i), threadId:$t, role:'user', text:'filler text'})",
          {"s": s, "e": e, "t": tid})
        q("UNWIND range($s,$e-2) AS i MATCH (a:Message {msgId:$t+'-'+toString(i)}), (b:Message {msgId:$t+'-'+toString(i+1)}) CREATE (a)-[:NEXT]->(b)",
          {"s": max(s - 1, 0), "e": e, "t": tid})
        if hub:
            q("MATCH (h:Entity {entityId:$h}) UNWIND range($s,$e-1) AS i MATCH (m:Message {msgId:$t+'-'+toString(i)}) CREATE (m)-[:MENTIONS]->(h)",
              {"s": s, "e": e, "t": tid, "h": hub})
    q("MATCH (t:Thread {threadId:$t}), (h:Message {msgId:$t+'-0'}) CREATE (t)-[:HEAD]->(h)", {"t": tid})

def filler(prefix, total, hub, per=50):
    for i in range(total // per):
        thread(f"{prefix}{i}", per, hub)

# fan-in sources: short filler threads mentioning each hub
filler("fA", 100, "H100"); filler("fB", 1000, "H1k"); filler("fC", 10000, "H10k")
for i in range(10):
    filler(f"fM{i}_", 10000, f"M{i}")
thread("deepHub", 100000, "H100k")      # 100k fan-in for H100k AND a 100k-deep thread
thread("deepPlain", 100000)             # 100k depth, no mentions
thread("mid", 10000)                    # 10k depth, no mentions
for i in range(10):
    thread(f"spread{i}", 5000)          # 10 threads of 5k depth, no mentions

def base():
    return [random.random() - 0.5 for _ in range(DIM)]
def near(v):
    return [x + random.gauss(0, 0.001) for x in v]

def seeds_tail(name, tid, mentions):
    """10 seeds appended to the tail of existing thread tid (clustered)."""
    v = base(); last = g.ro_query("MATCH (t:Thread {threadId:$t})-[:HEAD]->(h) MATCH (h)-[:NEXT*0..]->(z) WHERE NOT (z)-[:NEXT]->() RETURN z.msgId",
                                  {"t": tid}, timeout=600000).result_set[0][0]
    prev = last
    for j in range(10):
        sid = f"{name}-s{j}"
        q("MATCH (p:Message {msgId:$p}) CREATE (p)-[:NEXT]->(:Message {msgId:$s, threadId:$t, role:'user', text:'seed', embedding:vecf32($v)})",
          {"p": prev, "s": sid, "t": tid, "v": near(v)})
        for h in mentions:
            q("MATCH (m:Message {msgId:$s}), (h:Entity {entityId:$h}) CREATE (m)-[:MENTIONS]->(h)", {"s": sid, "h": h})
        prev = sid
    return v

def seeds_spread(name, mentions, under=None):
    """10 seeds, each the tail of its own thread: a new 20-msg thread, or the given existing threads."""
    v = base()
    for j in range(10):
        tid = under[j] if under else f"{name}-t{j}"
        if not under:
            thread(tid, 20)
        last = f"{tid}-{19 if not under else 4999}"
        sid = f"{name}-s{j}"
        q("MATCH (p:Message {msgId:$p}) CREATE (p)-[:NEXT]->(:Message {msgId:$s, threadId:$t, role:'user', text:'seed', embedding:vecf32($v)})",
          {"p": last, "s": sid, "t": tid, "v": near(v)})
        for h in mentions:
            q("MATCH (m:Message {msgId:$s}), (h:Entity {entityId:$h}) CREATE (m)-[:MENTIONS]->(h)", {"s": sid, "h": h})
    return v

scen = {}
scen["light: spread, short, 1 mention fan-in~10"] = seeds_spread("light", ["small"])
scen["fan-in 100, spread short"] = seeds_spread("f100", ["H100"])
scen["fan-in 1k, spread short"] = seeds_spread("f1k", ["H1k"])
scen["fan-in 10k, spread short"] = seeds_spread("f10k", ["H10k"])
scen["fan-in 100k, spread short"] = seeds_spread("f100k", ["H100k"])
scen["10 mentions x 10k, spread short"] = seeds_spread("m10", [f"M{i}" for i in range(10)])
scen["depth 10k clustered, no hub"] = seeds_tail("d10k", "mid", [])
scen["depth 100k clustered, no hub"] = seeds_tail("d100k", "deepPlain", [])
scen["depth 5k spread x10 threads, no hub"] = seeds_spread("d5ks", [], under=[f"spread{i}" for i in range(10)])
scen["worst: depth 100k clustered + fan-in 100k"] = seeds_tail("worst", "deepHub", ["H100k"])
# noise
for s in range(0, 2000, 200):
    q("UNWIND range($s,$s+199) AS i CREATE (:Message {msgId:'noise-'+toString(i), threadId:'noise', role:'user', text:'n', "
      "embedding: vecf32([x IN range(1,$d) | rand()-0.5])})", {"s": s, "d": DIM})
json.dump(scen, open("cov_scenarios.json", "w"))
print(g.ro_query("MATCH (m:Message) RETURN count(m)").result_set,
      g.ro_query("MATCH ()-[r]->() RETURN type(r), count(*)").result_set,
      g.ro_query("MATCH (e:Entity)<-[:MENTIONS]-(m) RETURN e.entityId, count(m) ORDER BY e.entityId").result_set)
