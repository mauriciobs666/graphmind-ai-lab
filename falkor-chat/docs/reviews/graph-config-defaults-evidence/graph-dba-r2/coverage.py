"""K-098 r2 (M1) coverage probe — throwaway container 16483 only.
For each scenario x k x query form x scope: uncapped latency, smallest passing QUERY_MEM_CAPACITY
(bisect, 64 KiB resolution), and result equality vs the verbatim query (seed ids, scores order, and
the related-message ids via id-returning diagnostic twins). Appends JSON lines to argv[1].
r4 (U12 gate F1): adds the threads-per-channel axis (build_tpc.py; scoped to the seeds' own channel,
plus a decoy channel that must return 0 rows) and the note's corrected text, form v3; env
COV_FORMS limits which forms are bisected (the reference form `current` always runs once, uncapped)."""
import os
import json, sys, time
from falkordb import FalkorDB
from redis.exceptions import ResponseError

db = FalkorDB(host="127.0.0.1", port=16483); r = db.connection; g = db.select_graph("ws:cov")
ANN = "CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec)) YIELD node AS seed, score "
SCOPE = "MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t) "
def ret(diag, tie=False):
    return ("RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, score, "
            + ("[m IN expanded | m.msgId]" if diag else "[m IN expanded | m.text]") + " AS relatedContext "
            + ("ORDER BY score ASC, msgId ASC LIMIT $limit" if tie else "ORDER BY score ASC LIMIT $limit"))
def cur(scoped, diag=False):   # verbatim repository.hybrid_search (diag: related ids instead of texts)
    return (ANN + "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) " + (SCOPE if scoped else "")
            + "OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
            "WITH seed, score, collect(DISTINCT related)[..5] AS expanded " + ret(diag))
BOUND = ("CALL { WITH seed OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message) "
         "WITH DISTINCT related LIMIT 5 RETURN collect(related) AS expanded } ")
def v2b(scoped, diag=False):   # bounded expansion, walk kept
    return ANN + "MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed) " + (SCOPE if scoped else "") + BOUND + ret(diag)
def v2(scoped, diag=False):    # bounded expansion + threadId anchor
    return ANN + "MATCH (t:Thread {threadId: seed.threadId}) " + (SCOPE if scoped else "") + BOUND + ret(diag)
def v3(scoped, diag=False):    # the note's §3.1 text as corrected after the U12 gate (WITH barrier, scoped only)
    return (ANN + "MATCH (t:Thread {threadId: seed.threadId}) " + ("WITH seed, score, t " + SCOPE if scoped else "")
            + BOUND + ret(diag, tie=True))
FORMS = {"current": cur, "v2b_bound_only": v2b, "v2_bound+anchor": v2, "v3_note_corrected": v3}
RUN = os.environ.get("COV_FORMS", ",".join(FORMS)).split(",")

def cap(b): r.execute_command("GRAPH.CONFIG", "SET", "QUERY_MEM_CAPACITY", str(b))
def run(c, p):
    try:
        return True, g.ro_query(c, p, timeout=300000).result_set
    except ResponseError as e:
        if "mem consumption" not in str(e): raise
        return False, None
def bisect(c, p):
    lo, hi = 0, 512 << 20
    while hi - lo > 64 << 10:
        m = (lo + hi) // 2; cap(m)
        if run(c, p)[0]: hi = m
        else: lo = m
    cap(0); return hi

scen = json.load(open("cov_scenarios.json"))
heavy_scoped = {"fan-in 100k, spread short", "worst: depth 100k clustered + fan-in 100k"}
tpc = {f"threads per channel {l}": f"tpc{l}" for l in ("1k", "10k", "100k", "1M")}
def rows_of(rows, tie):  # (seed id, related ids) in result order; tie: reference re-sorted by (score, msgId)
    return [(x[0], list(x[4])) for x in (sorted(rows, key=lambda x: (x[3], x[0])) if tie else rows)]
out = open(sys.argv[1], "a")
only = sys.argv[2:]  # optional scenario-name filter
for name, vec in scen.items():
    if only and name not in only: continue
    chans = [tpc[name], "c1"] if name in tpc else ["c1"]
    for scoped, chan in ([(False, None)] + [(True, c) for c in chans] if name in heavy_scoped or name in tpc else [(False, None)]):
        base_p = {"qVec": vec, "limit": 10, "channelId": chan} if scoped else {"qVec": vec, "limit": 10}
        for k in (1, 5, 10):
            p = dict(base_p, k=k)
            cap(0)
            ref = run(cur(scoped, True), p)[1]
            for fname, f in FORMS.items():
                if fname not in RUN: continue
                cap(0); t = time.perf_counter(); ok, rows = run(f(scoped), p); ms = (time.perf_counter() - t) * 1000
                same = None
                if fname != "current":
                    d = run(f(scoped, True), p)[1]
                    same = rows_of(d, False) == rows_of(ref, fname == "v3_note_corrected")
                rec = {"scenario": name, "scoped": scoped, "channel": chan, "k": k, "form": fname, "rows": len(rows),
                       "uncapped_ms": round(ms, 1), "min_cap_kib": bisect(f(scoped), p) // 1024,
                       "same_as_current": same}
                print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
cap(0)
