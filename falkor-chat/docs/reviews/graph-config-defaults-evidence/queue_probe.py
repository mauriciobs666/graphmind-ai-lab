"""K-098 queue probe — validation container (16482). C clients each fire ONE deep-walk read (~1.25 s, ws:deep)
with timeout T ms at the same instant; sample GRAPH.INFO waiting/running; report outcome per client."""
import json, sys, threading, time, collections
from falkordb import FalkorDB
C, T = int(sys.argv[1]), int(sys.argv[2])
db = FalkorDB(host="127.0.0.1", port=16482); g = db.select_graph("ws:deep"); r = db.connection
Q = "MATCH (x:Message {msgId:'d-99999'}) MATCH (t:Thread)-[:HEAD|NEXT*0..]->(x) RETURN t.threadId"
res = []; lock = threading.Lock(); go = threading.Event(); peak = {"waiting": 0, "running": 0}; done = threading.Event()
def info():
    while not done.is_set():
        try:
            raw = r.execute_command("GRAPH.INFO")
            txt = json.dumps(raw, default=str)
            w = txt.count("Received at") ; peak["raw_sample"] = txt[:300]
        except Exception as e: peak["err"] = str(e)
        time.sleep(0.1)
def w():
    go.wait(); t = time.perf_counter()
    try: g.ro_query(Q, timeout=T); out = "ok"
    except Exception as e: out = str(e)[:60]
    with lock: res.append((out, round(time.perf_counter() - t, 2)))
ts = [threading.Thread(target=w) for _ in range(C)]; [x.start() for x in ts]
it = threading.Thread(target=info); it.start(); time.sleep(0.3); go.set(); [x.join() for x in ts]; done.set(); it.join()
c = collections.Counter(o for o, _ in res)
oks = sorted(s for o, s in res if o == "ok"); bad = sorted(s for o, s in res if o != "ok")
print(json.dumps({"clients": C, "timeout_ms": T, "outcomes": c, "ok_latency_s_min_max": [oks[0], oks[-1]] if oks else None,
                  "fail_latency_s_min_max": [bad[0], bad[-1]] if bad else None, "graph_info_sample": peak.get("raw_sample")}))
