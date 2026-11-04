"""S22 (P0-16) -- what a nonzero ``TIMEOUT_MAX`` does to load writes, ``GRAPH.CONSTRAINT`` and full-text backfill.

Contract: p0-2 §4.5 S22 "Instance-wide side effects"; §2 D16; quirks "Default TIMEOUT is 1000ms" entry.
``TIMEOUT_MAX`` is instance-wide, so the window with a nonzero value is kept short (tens of seconds) and the original
value (read first) is restored in a ``finally``; the probe refuses to start if a ``ccg_p0slice__build_*`` key exists
(another unit may be loading) and touches only ``ccg_p0spike__to*`` keys.

CEILINGS and STOP RULES (stated before running): slow-write size <= 6,000 rows per query over 6,000 ``D`` nodes
(un-indexed endpoint scan; calibrated first at ``TIMEOUT_MAX`` 0); a calibration call over 20 s is the last one;
``used_memory`` is read before every step and the probe stops above 1 GB (``MEM_STOP``). Populated key for the
constraint/full-text probes: 300,000 ``Symbol`` nodes, <= 20k per batch. Nonzero-cap windows: 3,000 ms (write probes),
800 ms (index backfills), 10,000 ms (the sized P0 default; real 20k-row batches); each is short and restored in a ``finally``.

Usage: ``python timeout_probe.py`` (from ``cpg/``; writes ``results/timeout_probe.json``)
"""

from __future__ import annotations

import json
import secrets
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lock_replay as LR  # noqa: E402

G = "ccg_p0spike__to"
USER = "ccg_build"
MEM_STOP = 1_000_000_000
CAP_WRITE, CAP_INDEX, CAP_P0 = 3000, 800, 10_000
#: an un-indexed endpoint match: one label scan per row (the Dependency defect), so time grows with rows -- a controllable slow write
SLOW_Q = "UNWIND range(0, $n - 1) AS i MATCH (a:D {id: i}), (b:D {id: i + 1}) CREATE (a)-[:E]->(b)"
N_D = 6000


def mem(db) -> int:
    return int(db.connection.info("memory")["used_memory"])


def check_mem(db, log: list) -> None:
    m = mem(db)
    log.append(m)
    if m > MEM_STOP:
        raise SystemExit(f"stop rule: used_memory {m} > {MEM_STOP}")


def timed(fn):
    t0 = time.perf_counter()
    try:
        r = fn()
        return {"status": "ok", "s": round(time.perf_counter() - t0, 3), "result": r}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "s": round(time.perf_counter() - t0, 3), "error": str(e).splitlines()[0][:160]}


def count(admin, label="D") -> int:
    return admin.select_graph(G).ro_query(f"MATCH (n:{label}) RETURN count(n)", timeout=1000).result_set[0][0]


def edges(admin) -> int:
    return admin.select_graph(G).ro_query("MATCH ()-[e:E]->() RETURN count(e)", timeout=1000).result_set[0][0]


def run(out: Path | None = None) -> dict:
    admin = LR.connect()
    a = admin.connection
    if [k for k in a.scan_iter("ccg_p0slice__build_*")]:
        raise SystemExit("refusing: a ccg_p0slice build key exists (another unit may be loading)")
    original = int(a.execute_command("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")[1])
    memlog: list = []
    rec: dict = {"originalTimeoutMax": original, "engineTimeout": int(a.execute_command("GRAPH.CONFIG", "GET", "TIMEOUT")[1]),
                 "timeoutDefault": int(a.execute_command("GRAPH.CONFIG", "GET", "TIMEOUT_DEFAULT")[1]), "steps": {}}
    pw = secrets.token_urlsafe(18)
    if USER in a.execute_command("ACL", "USERS"):
        raise SystemExit(f"refusing: ACL user {USER} already exists")
    LR.drop(admin, G)
    steps = rec["steps"]

    def set_cap(v):
        a.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", v)

    try:
        a.execute_command("ACL", "SETUSER", USER, "reset", "on", ">" + pw, "resetkeys", "-@all", "~ccg_p0spike__to*",
                          "+graph.query", "+graph.constraint", "+info")
        build = LR.connect(USER, pw)

        def bq(q, p=None, t=None):
            g = build.select_graph(G)
            return g.query(q, p, t) if t else g.query(q, p)

        # -- 0. cap 0: the data, then calibration of the slow write (rows -> seconds)
        check_mem(admin, memlog)
        bq("UNWIND range(0, $n - 1) AS i CREATE (:D {id: i})", {"n": N_D})
        bq("CREATE INDEX FOR (s:Symbol) ON (s.symbolId)")
        for lo in range(0, 300_000, 20_000):
            bq("UNWIND range($lo, $hi - 1) AS i CREATE (:Symbol {symbolId: 'S' + toString(i), name: 'sym' + toString(i % 5000) + ' name' + toString(i)})",
               {"lo": lo, "hi": lo + 20_000})
        bq("CREATE INDEX FOR (s:Symbol) ON (s.name)")
        check_mem(admin, memlog)
        cal = {}
        for n in (500, 1000, 2000, 4000, 6000):
            cal[n] = timed(lambda n=n: bq(SLOW_Q, {"n": n}).relationships_created)
            check_mem(admin, memlog)
            if cal[n]["s"] > 20:
                break
        steps["calibration_cap0_slow_write"] = cal
        ok = {n: v["s"] for n, v in cal.items() if v["status"] == "ok"}

        def rows_for(target_s):
            return min(ok, key=lambda n: abs(ok[n] - target_s))

        n_mid, n_slow = rows_for(2.0), rows_for(4.0)
        rec["rows"] = {"mid(~2s)": n_mid, "slow(~4s)": n_slow}

        def constraint_and_wait():
            t0 = time.perf_counter()
            build.connection.execute_command("GRAPH.CONSTRAINT", "CREATE", G, "UNIQUE", "NODE", "Symbol", "PROPERTIES", 1, "symbolId")
            created = time.perf_counter() - t0
            st = []
            while time.perf_counter() - t0 < 60:
                st = [r[4] for r in admin.select_graph(G).ro_query("CALL db.constraints()", timeout=500).result_set]
                if st and all(x not in ("UNDER CONSTRUCTION", "PENDING") for x in st):
                    return {"command_s": round(created, 3), "operational_s": round(time.perf_counter() - t0, 3), "status": st}
                time.sleep(0.05)
            return {"command_s": round(created, 3), "operational_s": None, "status": st}

        def fulltext_and_wait():
            t0 = time.perf_counter()
            bq("CALL db.idx.fulltext.createNodeIndex('Symbol', 'name')")
            created = time.perf_counter() - t0
            rs = []
            while time.perf_counter() - t0 < 60:
                rs = admin.select_graph(G).ro_query("CALL db.indexes() YIELD label, types, status RETURN label, types, status", timeout=500).result_set
                ft = [r for r in rs if "FULLTEXT" in str(r[1])]
                if ft and all(r[2] == "OPERATIONAL" for r in ft):
                    hits = admin.select_graph(G).ro_query("CALL db.idx.fulltext.queryNodes('Symbol', 'sym17') YIELD node RETURN count(node)", timeout=500).result_set[0][0]
                    return {"command_s": round(created, 3), "operational_s": round(time.perf_counter() - t0, 3), "hits_sym17": hits}
                time.sleep(0.05)
            return {"command_s": round(created, 3), "operational_s": None, "indexes": [list(map(str, r)) for r in rs]}

        def drop_both():
            a.execute_command("GRAPH.CONSTRAINT", "DROP", G, "UNIQUE", "NODE", "Symbol", "PROPERTIES", 1, "symbolId")
            admin.select_graph(G).query("CALL db.idx.fulltext.drop('Symbol')")

        steps["constraint_cap0"] = timed(constraint_and_wait)
        steps["fulltext_cap0"] = timed(fulltext_and_wait)
        drop_both()

        # -- 1. cap 800 ms (below the uncapped backfill times): constraint and full-text backfill on 300k nodes
        check_mem(admin, memlog)
        set_cap(CAP_INDEX)
        try:
            steps["constraint_cap800"] = timed(constraint_and_wait)
            steps["fulltext_cap800"] = timed(fulltext_and_wait)
        finally:
            set_cap(original)
        drop_both()

        # -- 2. cap 3000 ms: writes with and without an explicit timeout
        set_cap(CAP_WRITE)
        w = {}
        try:
            for name, n, t in (("mid_noTimeout", n_mid, None), ("mid_timeout=cap", n_mid, CAP_WRITE),
                               ("slow_noTimeout", n_slow, None), ("slow_timeout=cap", n_slow, CAP_WRITE),
                               ("slow_timeout=1000", n_slow, 1000), ("mid_timeout=60000(>cap)", n_mid, 60000)):
                before = edges(admin)
                w[name] = timed(lambda n=n, t=t: bq(SLOW_Q, {"n": n}, t).relationships_created)
                w[name]["rows"], w[name]["committed"] = n, edges(admin) - before
                check_mem(admin, memlog)
        finally:
            steps["writes_cap3000"] = w
            set_cap(original)

        # -- 3. cap 10000 (the sized P0 default): real 20k-row node and edge batches as ccg_build
        set_cap(CAP_P0)
        p0 = {}
        try:
            edge_q = ("UNWIND $rows AS r MATCH (a:Symbol {symbolId: r.a}), (b:Symbol {symbolId: r.b}) "
                      "CREATE (a)-[:CALLS]->(b)")
            rows = [{"a": f"S{i}", "b": f"S{(i * 7 + 1) % 300_000}"} for i in range(20_000)]
            node_q = "UNWIND range(1, $n) AS i CREATE (:Z {i: i})"
            p0["edge_batch_20k_noTimeout"] = timed(lambda: bq(edge_q, {"rows": rows}).relationships_created)
            p0["edge_batch_20k_timeout=cap"] = timed(lambda: bq(edge_q, {"rows": rows}, CAP_P0).relationships_created)
            p0["node_batch_20k_noTimeout"] = timed(lambda: bq(node_q, {"n": 20_000}).nodes_created)
            p0["node_batch_20k_timeout=cap"] = timed(lambda: bq(node_q, {"n": 20_000}, CAP_P0).nodes_created)
            before = edges(admin)
            p0["slow_write_mid_noTimeout"] = timed(lambda: bq(SLOW_Q, {"n": n_mid}).relationships_created)
            p0["slow_write_mid_noTimeout"]["committed"] = edges(admin) - before
            check_mem(admin, memlog)
        finally:
            steps["loads_cap10000"] = p0
            set_cap(original)
        rec["timeoutMaxRestored"] = int(a.execute_command("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")[1])
    finally:
        try:
            set_cap(original)
        except Exception:  # noqa: BLE001
            pass
        try:
            a.execute_command("ACL", "DELUSER", USER)
        except Exception:  # noqa: BLE001
            pass
        LR.drop(admin, G)
        rec["memoryLog"] = memlog
        rec["finalTimeoutMax"] = int(a.execute_command("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")[1])
        if out:
            out.write_text(json.dumps(rec, indent=1, default=str))
    return rec


if __name__ == "__main__":
    r = run(HERE / "results" / "timeout_probe.json")
    print(json.dumps({k: v for k, v in r.items() if k != "steps"}, default=str))
    for k, v in r["steps"].items():
        print(k, json.dumps(v, default=str))
