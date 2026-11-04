"""S23 (P0-16) -- does a single large write make ``ccg-falkordb`` stop answering ``PING``? (review §S03 "Incident")

Contract: p0-2 §3 S23 row, §2 D16, reservation R-C4. A 100k-edge single-query write once left the instance silent for
minutes; the cause was not established. This probe writes one query of N edges and samples ``PING`` from a **separate
process** on its own connection every ~5 ms (an in-process second connection is a known harness artefact, S21), and records
CPU (``INFO cpu`` deltas) and ``INFO memory`` around each write.

Two write shapes, because the incident's cause is one of them:

* ``indexed``   -- ``UNWIND $rows AS r MATCH (a:N {id: r.a}), (b:N {id: r.b}) CREATE (a)-[:E]->(b)`` over 100,000 ``N``
  nodes with a range index on ``id`` (the loader's shape);
* ``unindexed`` -- the same over ``M`` nodes with no index (a label scan per row: the ``Dependency`` defect, S21).

CEILINGS and STOP RULES (stated before running; keys ``ccg_p0spike__stall*`` only): ``indexed`` N in 20k, 50k, 100k and,
only while the previous size's worst ``PING`` stayed under 1 s, 200k and 500k; ``unindexed`` N in 1k, 2k, 5k, 10k over
20,000 ``M`` nodes. A call over 60 s, a ``PING`` over 5 s, or ``used_memory`` over 1 GB is the last step of its shape.

Usage: ``python stall_probe.py`` (from ``cpg/``; writes ``results/stall_probe.json``)
"""

from __future__ import annotations

import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lock_replay as LR  # noqa: E402

G = "ccg_p0spike__stall_g"
N_NODES, M_NODES = 100_000, 20_000
MEM_STOP = 1_000_000_000
PING_STOP_S, CALL_STOP_S = 5.0, 60.0

SAMPLER_PORT, SAMPLER_INTERVAL_S = LR.PORT, 0.005

_SAMPLER_TEMPLATE = r"""
import json, os, signal, sys, time
import redis
r = redis.Redis(host="127.0.0.1", port=%(port)d, password=os.environ["STALL_PW"], socket_timeout=300, retry=None)
out, samples, stop = sys.argv[1], [], []
signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
r.ping()
print("READY", r.info("server")["run_id"], flush=True)
while not stop:
    t0 = time.monotonic()
    try:
        r.ping()
    except redis.exceptions.ResponseError:
        pass                      # a BUSY reply is still the server answering; its latency is the datum
    samples.append((t0, time.monotonic() - t0))
    time.sleep(%(interval)r)
json.dump(samples, open(out, "w"))
"""


def sampler_source() -> str:
    return _SAMPLER_TEMPLATE % {"port": SAMPLER_PORT, "interval": SAMPLER_INTERVAL_S}


def windows(t0: float, t1: float) -> dict:
    """The three PING windows around a write that ran ``[t0, t1]`` (1 s idle before, 1 s tail after)."""
    return {"baseline": (t0 - 1.0, t0 - 0.05), "during": (t0, t1), "after": (t1 + 0.05, t1 + 1.0)}


def sample_during(work, out: Path, pw: str) -> dict:
    """Start the PING sampler in its own process, idle 1 s, run ``work()``, tail 1 s, stop it. Returns the samples, the
    window stats, the timestamps, ``work``'s result and the ``run_id`` of the server the sampler is connected to."""
    import os
    env = dict(os.environ, STALL_PW=pw)
    p = subprocess.Popen([sys.executable, "-c", sampler_source(), str(out)], stdout=subprocess.PIPE, text=True, env=env)
    ready = p.stdout.readline().split()
    assert ready[0] == "READY"
    time.sleep(1.0)
    t0 = time.monotonic()
    try:
        result = work()
    finally:
        t1 = time.monotonic()
        time.sleep(1.0)
        p.terminate()
        p.wait(30)
    samples = json.load(open(out))
    out.unlink()
    w = windows(t0, t1)
    return {"samples": samples, "t0": t0, "t1": t1, "result": result, "samplerRunId": ready[1],
            **{k: summarize_pings(samples, *v) for k, v in w.items()}}


def summarize_pings(samples: list[tuple[float, float]], t_start: float, t_end: float) -> dict:
    """Latency stats of the PINGs that were *issued* within ``[t_start, t_end]`` (a PING stuck behind the write was
    issued inside the window and returns after it). Latencies in ms."""
    lat = sorted(l * 1000 for t, l in samples if t_start <= t <= t_end)
    if not lat:
        return {"n": 0}
    pick = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]  # noqa: E731
    return {"n": len(lat), "p50_ms": round(pick(0.5), 2), "p99_ms": round(pick(0.99), 2), "max_ms": round(lat[-1], 2),
            "over100ms": sum(x > 100 for x in lat), "over1s": sum(x > 1000 for x in lat)}


def cpu(db) -> tuple[float, float]:
    i = db.connection.info("cpu")
    return float(i["used_cpu_sys"]), float(i["used_cpu_user"])


def run(out: Path | None = None) -> dict:
    import os
    admin = LR.connect()
    a = admin.connection
    if [k for k in a.scan_iter("ccg_p0slice__build_*")]:
        raise SystemExit("refusing: a ccg_p0slice build key exists (another unit may be loading)")
    LR.drop(admin, G)
    rnd = random.Random(20261104)
    rec: dict = {"shapes": {}, "setup": {}}
    from load import password
    pw = password()
    g = admin.select_graph(G)
    g.query("CREATE INDEX FOR (n:N) ON (n.id)")
    for lo in range(0, N_NODES, 20_000):
        g.query("UNWIND range($lo, $hi - 1) AS i CREATE (:N {id: i})", {"lo": lo, "hi": lo + 20_000})
    for lo in range(0, M_NODES, 20_000):
        g.query("UNWIND range($lo, $hi - 1) AS i CREATE (:M {id: i})", {"lo": lo, "hi": lo + 20_000})
    rec["setup"] = {"N": N_NODES, "M": M_NODES, "usedMemoryAfterSetup": int(a.info("memory")["used_memory"]),
                    "graphMemory": dict(zip(*[iter(a.execute_command("GRAPH.MEMORY", "USAGE", G))] * 2))}
    q_idx = "UNWIND $rows AS r MATCH (a:N {id: r.a}), (b:N {id: r.b}) CREATE (a)-[:E]->(b)"
    q_unidx = "UNWIND $rows AS r MATCH (a:M {id: r.a}), (b:M {id: r.b}) CREATE (a)-[:E]->(b)"

    def shape(name, query, universe, sizes, conditional):
        steps, prev_max = [], 0.0
        for n in sizes:
            if conditional and n > 100_000 and prev_max >= 1000:
                steps.append({"edges": n, "skipped": f"previous worst PING {prev_max} ms >= 1000 ms"})
                continue
            if int(a.info("memory")["used_memory"]) > MEM_STOP:
                steps.append({"edges": n, "skipped": "memory stop rule"})
                break
            rows = [{"a": rnd.randrange(universe), "b": rnd.randrange(universe)} for _ in range(n)]
            tmp = HERE / "results" / f"_stall_samples_{name}_{n}.json"
            box = {}

            def work():
                box["c0"], box["m0"] = cpu(admin), int(a.info("memory")["used_memory"])
                try:
                    res = admin.select_graph(G).query(q_idx if name == "indexed" else q_unidx, {"rows": rows})
                    box["created"], box["err"] = res.relationships_created, None
                except Exception as e:  # noqa: BLE001
                    box["created"], box["err"] = None, str(e).splitlines()[0][:120]
                box["c1"], box["m1"] = cpu(admin), int(a.info("memory")["used_memory"])

            sd = sample_during(work, tmp, pw)
            samples, t0, t1 = sd["samples"], sd["t0"], sd["t1"]
            created, err, c0, c1, m0, m1 = box["created"], box["err"], box["c0"], box["c1"], box["m0"], box["m1"]
            assert sd["samplerRunId"] == a.info("server")["run_id"], "sampler is not connected to ccg-falkordb"
            step = {"edges": n, "write_s": round(t1 - t0, 3), "created": created, "error": err,
                    "baseline": sd["baseline"], "during": sd["during"], "after": sd["after"],
                    "cpuSysDelta_s": round(c1[0] - c0[0], 2), "cpuUserDelta_s": round(c1[1] - c0[1], 2),
                    "usedMemoryDelta": m1 - m0}
            # the longest gap between consecutive samples inside the window is the stall as a client sees it
            inside = [s for s in samples if t0 - 0.2 <= s[0] <= t1 + 0.2]
            step["longestGapBetweenPings_ms"] = round(max((b[0] - a_[0] for a_, b in zip(inside, inside[1:])), default=0) * 1000, 1)
            steps.append(step)
            prev_max = max(step["during"].get("max_ms", 0), step["after"].get("max_ms", 0))
            if prev_max > PING_STOP_S * 1000 or step["write_s"] > CALL_STOP_S:
                steps.append({"stop": f"stop rule hit at {n}: worst PING {prev_max} ms, write {step['write_s']} s"})
                break
        rec["shapes"][name] = steps
        # positive control (gate P3-3): at the largest indexed step the PING under load must be slower than the idle baseline
        big = [x for x in steps if x.get("edges", 0) >= 200_000 and "during" in x]
        if name == "indexed" and big:
            rec["positiveControl"] = {"edges": big[-1]["edges"], "duringMaxMs": big[-1]["during"]["max_ms"], "baselineMaxMs": big[-1]["baseline"]["max_ms"]}
            assert big[-1]["during"]["max_ms"] > big[-1]["baseline"]["max_ms"], "positive control failed: the sampler does not see the write"

    try:
        shape("indexed", q_idx, N_NODES, (20_000, 50_000, 100_000, 200_000, 500_000), True)
        shape("unindexed", q_unidx, M_NODES, (1_000, 2_000, 5_000, 10_000), False)
        pass
    finally:
        LR.drop(admin, G)
        rec["usedMemoryAfterCleanup"] = int(a.info("memory")["used_memory"])
        if out:
            out.write_text(json.dumps(rec, indent=1))
    return rec


def telemetry_after_delete(out: Path | None = None) -> dict:
    """The stall run left ``telemetry{g}`` in place right after ``GRAPH.DELETE g`` of a ~1M-edge graph (gone a few seconds
    later). Measure how long it outlives the delete for a small and a larger graph.

    CEILING and STOP RULE: <= 100,000 nodes and <= 400,000 edges per graph, 3 trials each, poll <= 60 s; used_memory
    over 1 GB stops the probe. Keys: ``ccg_p0spike__stall_d`` and its stream only."""
    db = LR.connect()
    a = db.connection
    key = "ccg_p0spike__stall_d"
    tel = "telemetry{" + key + "}"
    rec = {"trials": []}
    rnd = random.Random(7)
    for nodes, edges in ((20_000, 20_000), (100_000, 400_000)):
        for trial in range(3):
            if int(a.info("memory")["used_memory"]) > MEM_STOP:
                raise SystemExit("memory stop rule")
            LR.drop(db, key)
            time.sleep(0.5)
            g = db.select_graph(key)
            g.query("CREATE INDEX FOR (n:N) ON (n.id)")
            for lo in range(0, nodes, 20_000):
                g.query("UNWIND range($lo, $hi - 1) AS i CREATE (:N {id: i})", {"lo": lo, "hi": min(nodes, lo + 20_000)})
            for lo in range(0, edges, 20_000):
                rows = [{"a": rnd.randrange(nodes), "b": rnd.randrange(nodes)} for _ in range(min(20_000, edges - lo))]
                g.query("UNWIND $rows AS r MATCH (a:N {id: r.a}), (b:N {id: r.b}) CREATE (a)-[:E]->(b)", {"rows": rows})
            for _ in range(5):
                g.ro_query("MATCH (n:N) RETURN count(n)")
            t_wait = time.monotonic()
            while not a.exists(tel) and time.monotonic() - t_wait < 8:
                g.ro_query("MATCH (n:N) RETURN count(n)")
                time.sleep(0.1)
            existed = bool(a.exists(tel))
            t0 = time.monotonic()
            g.delete()
            t_ret = time.monotonic() - t0
            gone_at = None
            while time.monotonic() - t0 < 60:
                if not a.exists(tel):
                    gone_at = time.monotonic() - t0
                    break
                time.sleep(0.005)
            rec["trials"].append({"nodes": nodes, "edges": edges, "streamExistedBeforeDelete": existed,
                                  "deleteReturnS": round(t_ret, 3), "streamGoneAtS": None if gone_at is None else round(gone_at, 3),
                                  "streamReappeared": bool(a.exists(tel)) if gone_at is not None else None})
            time.sleep(1.0)
            rec["trials"][-1]["streamExistsAfter1s"] = bool(a.exists(tel))
    LR.drop(db, key)
    if a.exists(tel):
        a.delete(tel)
    if out:
        out.write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    if sys.argv[1:] == ["delete-probe"]:
        for t in telemetry_after_delete(HERE / "results" / "stall_telemetry_delete.json")["trials"]:
            print(json.dumps(t))
        sys.exit(0)
    r = run(HERE / "results" / "stall_probe.json")
    for name, steps in r["shapes"].items():
        for s in steps:
            print(name, json.dumps(s))
