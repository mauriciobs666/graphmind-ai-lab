"""S20/S21 gate follow-up (m1) -- the claims MEASUREMENTS.md makes that had no ``results/`` file, re-run and recorded.

``digests``  re-digest every 5th recorded depth-3 anchor (BFS and SSpaths) on the current ``ccg_p0spike`` and compare with
             ``results/bench_reach.jsonl`` (the graph has been rebuilt since the benchmark; same generator and seed);
``ping``     ``redis-cli PING`` from another process while a ~4 s read runs in a child process, at several delays after the
             child signals it is about to send the read;
``timeout``  the anchor whose read hit the 1,000 ms default ``TIMEOUT`` during the benchmark, re-run 5 times with NO
             ``timeout`` argument, worst single read.
Usage: ``python recheck_claims.py [digests|ping|timeout]...`` ; writes ``results/recheck_<name>.json``.
Keys: reads ``ccg_p0spike``; ``ping`` writes ``ccg_p0spike__swap_ping`` (deleted at the end).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_reach as R  # noqa: E402
import bfs as B  # noqa: E402
import swap_probe as W  # noqa: E402
import synth as S  # noqa: E402

RESULTS = HERE / "results"


def digests(st):
    g = st.db.select_graph("ccg_p0spike")
    recs = [json.loads(l) for l in (RESULTS / "bench_reach.jsonl").read_text().splitlines()]
    seg3 = [r for r in recs if r["kind"] == "segment" and r["depth"] == 3]
    rows = []
    for r in seg3[::5]:
        a = r["anchor"]["symbolId"]
        b = B.answer(g, a, mode="segment", levels=6, timeout=R.READ_TIMEOUT_MS)["digest"]
        s = R.sspaths_answer(g, a, 3)["digest"]
        rows.append({"anchor": a, "class": r["anchor"]["class"], "bfsSame": b == r["bfs"]["digests"][0], "sspathsSame": s == r["sspaths"]["digests"][0]})
    return {"anchors": len(rows), "allSame": all(x["bfsSame"] and x["sspathsSame"] for x in rows), "rows": rows}


def ping(st):
    p = W.Probe(st)
    g = W.PREFIX + "ping"
    p.wipe(g)
    p.make(g, "c", n=5000)
    env = dict(os.environ, REDISCLI_AUTH=S.L.password())
    src = ("import sys,time\nsys.path.insert(0,%r)\nimport synth\nst=synth.SpikeStore('127.0.0.1',6390)\nprint('go',flush=True)\n"
           "t=time.time()\nst.ro_query(%r,'MATCH (a:N),(b:N) WHERE a.v+b.v>0 RETURN count(*)',None,120000)\nprint(round(time.time()-t,2))\n") % (str(HERE), g)
    rows = []
    try:
        for delay in (0.3, 0.1, 0.05, 0.02, 0.005, 0.0):
            child = subprocess.Popen([sys.executable, "-c", src], stdout=subprocess.PIPE, text=True)
            child.stdout.readline()
            time.sleep(delay)
            t = time.perf_counter()
            r = subprocess.run(["redis-cli", "-p", "6390", "PING"], env=env, capture_output=True, text=True)
            ms = (time.perf_counter() - t) * 1000
            read_s = float(child.communicate()[0].strip())
            rows.append({"delayAfterGoS": delay, "pingMs": round(ms, 1), "reply": r.stdout.strip(), "readSeconds": read_s})
    finally:
        p.wipe(g)
    return {"rows": rows, "maxPingMs": max(x["pingMs"] for x in rows)}


def timeout(st):
    g = st.db.select_graph("ccg_p0spike")
    recs = [json.loads(l) for l in (RESULTS / "bench_reach.jsonl").read_text().splitlines()]
    hubs = [r for r in recs if r["kind"] == "segment" and r["depth"] == 3 and r["anchor"]["class"] == "hub"]
    crash_anchor = hubs[11]["anchor"]["symbolId"]  # the 12th hub record: the first one written after the resume
    worst, runs = 0.0, []
    for _ in range(5):
        r = B.answer(g, crash_anchor, mode="segment", levels=6)  # no timeout argument: the engine default applies
        w = max(x[2] for x in r["tracker"].reads)
        runs.append(round(w * 1000, 1))
        worst = max(worst, w)
    return {"anchor": crash_anchor, "worstReadMsPerRun": runs, "worstReadMs": round(worst * 1000, 1)}


def main(argv):
    st = S.SpikeStore("127.0.0.1", 6390)
    for name in (argv or ["digests", "ping", "timeout"]):
        out = {"digests": digests, "ping": ping, "timeout": timeout}[name](st)
        (RESULTS / f"recheck_{name}.json").write_text(json.dumps(out, indent=1))
        print(name, json.dumps(out)[:400])


if __name__ == "__main__":
    main(sys.argv[1:])
