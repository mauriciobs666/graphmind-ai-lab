"""S21 (P0-15) -- ``MULTI``/``RENAME`` rollback and the ``telemetry{<graph>}`` stream, probed on ``ccg-falkordb``.

Contract: ``docs/plans/connected-code-graph-p0.md`` §4.5 (S21), ``docs/plans/connected-code-graph-p0-2.md`` S21 row,
graph note ``docs/plans/connected-code-graph-graph.md`` Q4, V2, V10, G13.

Keys: only ``ccg_p0spike__swap_*`` graphs and their companion ``telemetry{ccg_p0spike__swap_*}`` streams are written
(:func:`guard` checks every key of every command this module sends). The telemetry config probes use
``GRAPH.CONFIG GET`` and ``CONFIG GET`` only; a ``SET`` is attempted solely on parameter names that the build does not
list, to confirm it rejects them (a rejected ``SET`` changes nothing).

Every probe returns ``{"probe", "commands", "observed"}`` and :func:`main` writes them to
``results/swap_probe.json`` next to this file; ``MEASUREMENTS.md`` quotes them.

Usage: ``python swap_probe.py [--only rollback|telemetry|delete]``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import synth as S  # noqa: E402  (SpikeStore: zero-retry connection + the spike key guard)

PREFIX = "ccg_p0spike__swap_"
_SWAP = re.compile(rf"(telemetry\{{)?{PREFIX}[A-Za-z0-9_]+(\}})?")


def guard(key: str) -> str:
    """Every key this module sends a write for: ``ccg_p0spike__swap_*`` or its ``telemetry{...}`` companion."""
    if not isinstance(key, str) or not _SWAP.fullmatch(key) or key.count("{") != key.count("}"):
        raise S.SynthError(f"refusing key {key!r}: S21 touches only {PREFIX}* and telemetry{{{PREFIX}*}}")
    return key


def tel(graph: str) -> str:
    return f"telemetry{{{graph}}}"


class Probe:
    def __init__(self, store: S.SpikeStore):
        self.st = store
        self.c = store.conn
        self.log: list[str] = []

    # ---- guarded raw commands ----------------------------------------------------------------

    def cmd(self, *args, keys: tuple = ()):
        for k in keys:
            guard(k)
        self.log.append(" ".join(str(a) for a in args))
        return self.c.execute_command(*args)

    def q(self, graph: str, cypher: str, params=None, timeout=None):
        guard(graph)
        self.log.append(f"GRAPH.QUERY {graph} {cypher!r}" + (f" params={params}" if params else ""))
        return self.st.query(graph, cypher, params, timeout)

    def exists(self, key: str) -> bool:
        return bool(self.c.exists(guard(key)))

    def wipe(self, *graphs: str) -> None:
        for g in graphs:
            guard(g)
            if self.exists(g):
                self.st.delete(g)
            if self.exists(tel(g)):
                self.c.delete(guard(tel(g)))

    def make(self, graph: str, marker: str, n: int = 10) -> None:
        self.wipe(graph)
        self.q(graph, "CREATE (:Meta {marker: $m})", {"m": marker})
        self.q(graph, "UNWIND range(1, $n) AS i CREATE (:N {v: i})", {"n": n})
        self.q(graph, "CREATE INDEX FOR (n:N) ON (n.v)")

    def marker(self, graph: str):
        if not self.exists(graph):
            return None
        return self.st.ro_query(graph, "MATCH (m:Meta) RETURN m.marker").result_set[0][0]

    def state(self, *graphs: str) -> dict:
        return {g.removeprefix(PREFIX): self.marker(g) for g in graphs}


# ---------------------------------------------------------------------------------------------
# A. MULTI / RENAME rollback.
# ---------------------------------------------------------------------------------------------


def probe_rollback(p: Probe) -> list[dict]:
    live, build, prev = PREFIX + "live", PREFIX + "build", PREFIX + "prev"
    out = []

    # (i) clean
    p.wipe(live, build, prev)
    p.make(live, "old")
    p.make(build, "new")
    p.log.clear()
    pipe = p.c.pipeline(transaction=True)
    pipe.rename(guard(live), guard(prev))
    pipe.rename(guard(build), guard(live))
    reply = pipe.execute(raise_on_error=False)
    out.append({"probe": "rollback-i-clean", "commands": ["MULTI", f"RENAME {live} {prev}", f"RENAME {build} {live}", "EXEC"],
                "observed": {"execReplies": [repr(r) for r in reply], "markersAfter": p.state(live, build, prev),
                            "buildExists": p.exists(build)}})

    # (ii) build absent: the second RENAME fails at EXEC time, after the first one ran
    p.wipe(live, build, prev)
    p.make(live, "old")
    pipe = p.c.pipeline(transaction=True)
    pipe.rename(guard(live), guard(prev))
    pipe.rename(guard(build), guard(live))
    reply = pipe.execute(raise_on_error=False)
    out.append({"probe": "rollback-ii-build-absent", "commands": ["MULTI", f"RENAME {live} {prev}", f"RENAME {build} {live}", "EXEC"],
                "observed": {"execReplies": [repr(r) for r in reply], "markersAfter": p.state(live, build, prev),
                            "liveExists": p.exists(live), "prevExists": p.exists(prev),
                            "rolledBack": p.exists(live) and not p.exists(prev)}})

    # (ii-b) the same absent-build swap guarded by one EVAL (EXISTS first): the script refuses before any RENAME
    p.wipe(live, build, prev)
    p.make(live, "old")
    script = ("if redis.call('EXISTS', KEYS[2]) == 0 then return 'refused: build absent' end "
              "redis.call('RENAME', KEYS[1], KEYS[3]) redis.call('RENAME', KEYS[2], KEYS[1]) return 'swapped'")
    try:
        r = p.cmd("EVAL", script, 3, live, build, prev, keys=(live, build, prev))
        r = r.decode() if isinstance(r, bytes) else r
    except Exception as exc:  # noqa: BLE001 -- recorded verbatim
        r = f"{type(exc).__name__}: {exc}"
    out.append({"probe": "rollback-ii-b-eval-guard-build-absent", "commands": [f"EVAL <exists-guarded 2x RENAME> 3 {live} {build} {prev}"],
                "observed": {"reply": r, "markersAfter": p.state(live, build, prev)}})
    # and on the happy path, same script
    p.wipe(live, build, prev)
    p.make(live, "old")
    p.make(build, "new")
    try:
        r = p.cmd("EVAL", script, 3, live, build, prev, keys=(live, build, prev))
        r = r.decode() if isinstance(r, bytes) else r
    except Exception as exc:  # noqa: BLE001
        r = f"{type(exc).__name__}: {exc}"
    out.append({"probe": "rollback-ii-c-eval-guard-happy", "commands": [f"EVAL <exists-guarded 2x RENAME> 3 {live} {build} {prev}"],
                "observed": {"reply": r, "markersAfter": p.state(live, build, prev)}})

    # (ii-d) the existing-prev case: RENAME live prev over an existing prev overwrites it silently
    p.wipe(live, build, prev)
    p.make(live, "old")
    p.make(build, "new")
    p.make(prev, "older")
    pipe = p.c.pipeline(transaction=True)
    pipe.rename(guard(live), guard(prev))
    pipe.rename(guard(build), guard(live))
    reply = pipe.execute(raise_on_error=False)
    out.append({"probe": "rollback-ii-d-prev-exists", "commands": ["MULTI", f"RENAME {live} {prev}", f"RENAME {build} {live}", "EXEC"],
                "observed": {"execReplies": [repr(r) for r in reply], "markersAfter": p.state(live, build, prev)}})

    # (iii) a reader in flight on live while the swap runs: inside MULTI/EXEC, then a plain RENAME for comparison
    def in_flight(name, commands, swap):
        p.wipe(live, build, prev)
        p.make(live, "old", n=7000)
        p.make(build, "new", n=100)
        slow = ("MATCH (m:Meta) WITH m.marker AS mk0 MATCH (a:N), (b:N) WHERE a.v + b.v > 0 "
                "WITH mk0, count(*) AS c MATCH (m2:Meta) RETURN mk0, m2.marker, c")
        res: dict = {}
        done = threading.Event()
        pings: list[float] = []
        slow_pings: list[list[float]] = []
        stop, warm = threading.Event(), threading.Event()
        t_ref = time.perf_counter()

        def reader():
            t0 = time.monotonic()
            try:
                r = p.st.ro_query(live, slow, None, 120000).result_set[0]
                res["reader"] = {"markers": [r[0], r[1]], "count": r[2], "seconds": round(time.monotonic() - t0, 2)}
            except Exception as exc:  # noqa: BLE001
                res["reader"] = {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.monotonic() - t0, 2)}
            done.set()

        def pinger():
            import redis
            c2 = redis.Redis(host="127.0.0.1", port=6390, password=S.L.password(), retry=S.L.zero_retry(), socket_timeout=60)
            c2.ping()  # warm: connect + AUTH happen before the reader starts
            warm.set()
            while not stop.is_set():
                t = time.perf_counter()
                c2.ping()
                ms = (time.perf_counter() - t) * 1000
                pings.append(ms)
                if ms > 100:
                    slow_pings.append([round(t - t_ref, 3), round(ms, 1)])
                time.sleep(0.01)

        th, pt = threading.Thread(target=reader), threading.Thread(target=pinger)
        pt.start()
        warm.wait(10)
        time.sleep(0.3)
        th.start()
        time.sleep(1.5)
        t0 = time.perf_counter()
        reply = swap() if swap else []
        exec_ms = (time.perf_counter() - t0) * 1000
        reader_running_at_return = not done.is_set()
        after = p.state(live, build, prev)
        th.join()
        stop.set()
        pt.join()
        res.update({"swapReplies": [repr(r) for r in reply], "swapMs": round(exec_ms, 2), "readerStillRunningWhenSwapReturned": reader_running_at_return,
                    "markersRightAfterSwap": after, "maxPingMs": round(max(pings), 2) if pings else None, "pings": len(pings), "pingsOver1s": sum(x > 1000 for x in pings),
                    "slowPings[startOffsetS, ms]": slow_pings[:6], "swapIssuedAtS": round(t0 - t_ref, 3)})
        out.append({"probe": name, "commands": [f"GRAPH.RO_QUERY {live} <7000x7000 self-join, ~12 s>", *commands], "observed": res})

    def multi():
        pipe = p.c.pipeline(transaction=True)
        pipe.rename(guard(live), guard(prev))
        pipe.rename(guard(build), guard(live))
        return pipe.execute(raise_on_error=False)

    in_flight("rollback-iii-0-control-no-swap", [], None)
    in_flight("rollback-iii-reader-in-flight-multi", ["MULTI", f"RENAME {live} {prev}", f"RENAME {build} {live}", "EXEC"], multi)
    in_flight("rollback-iii-b-reader-in-flight-plain-rename", [f"RENAME {build} {live}"],
              lambda: [p.c.rename(guard(build), guard(live))])
    in_flight("rollback-iii-c-reader-in-flight-eval", [f"EVAL <exists-guarded 2x RENAME> 3 {live} {build} {prev}"],
              lambda: [p.c.execute_command("EVAL", script, 3, live, build, prev)])
    p.wipe(live, build, prev)
    return out


# ---------------------------------------------------------------------------------------------
# B. telemetry: cap / disable / trim / rename.
# ---------------------------------------------------------------------------------------------


def probe_telemetry_config(p: Probe) -> list[dict]:
    out = []
    cfg = p.cmd("GRAPH.CONFIG", "GET", "*")
    names = [(c[0].decode() if isinstance(c[0], bytes) else c[0]) for c in cfg] if cfg and isinstance(cfg[0], (list, tuple)) else cfg
    out.append({"probe": "telemetry-graph-config-names", "commands": ["GRAPH.CONFIG GET *"], "observed": {"names": names}})
    rc = p.cmd("CONFIG", "GET", "*telemetry*")
    out.append({"probe": "telemetry-redis-config", "commands": ["CONFIG GET *telemetry*"], "observed": {"reply": [x.decode() if isinstance(x, bytes) else x for x in rc]}})
    for name in ("TELEMETRY", "TELEMETRY_ENABLED", "QUERY_TELEMETRY", "TELEMETRY_MAX_LEN", "TELEMETRY_STREAM_MAXLEN"):
        try:
            r = p.cmd("GRAPH.CONFIG", "SET", name, 0)
            out.append({"probe": f"telemetry-set-{name}", "commands": [f"GRAPH.CONFIG SET {name} 0"], "observed": {"reply": repr(r), "WARNING": "accepted"}})
        except Exception as exc:  # noqa: BLE001
            out.append({"probe": f"telemetry-set-{name}", "commands": [f"GRAPH.CONFIG SET {name} 0"], "observed": {"error": f"{type(exc).__name__}: {exc}"}})
    mods = p.cmd("MODULE", "LIST")
    out.append({"probe": "modules", "commands": ["MODULE LIST"], "observed": {"reply": repr(mods)[:400]}})
    return out


def wait_for(pred, timeout=10.0, step=0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return time.monotonic() - t0
        time.sleep(step)
    return None


def probe_telemetry_config_effect(p: Probe) -> list[dict]:
    """``MAX_INFO_QUERIES`` (1000) and ``CMD_INFO`` (1) are the only listed knobs near the stream's size. Try each at
    runtime on a swap graph; always restore the original value (instance-wide, S21 exit hygiene)."""
    g = PREFIX + "cfg"
    t = tel(g)
    out = []

    def get(k):
        return p.cmd("GRAPH.CONFIG", "GET", k)[1]

    for knob, value in (("MAX_INFO_QUERIES", 10), ("MAX_INFO_QUERIES", 1), ("MAX_INFO_QUERIES", 0), ("CMD_INFO", 0)):
        orig = get(knob)
        rec: dict = {"original": orig}
        changed = False
        try:
            try:
                rec["setReply"] = repr(p.cmd("GRAPH.CONFIG", "SET", knob, value))
                rec["valueAfterSet"] = get(knob)
                changed = True
            except Exception as exc:  # noqa: BLE001
                rec["setError"] = f"{type(exc).__name__}: {exc}"
            p.wipe(g)
            p.make(g, "x")
            for i in range(60):
                p.st.query(g, "RETURN $i", {"i": i})
            wait_for(lambda: p.exists(t), 8)
            time.sleep(4)
            rec["streamExists"] = p.exists(t)
            rec["xlenAfter60Queries"] = p.c.xlen(guard(t)) if rec["streamExists"] else None
            try:
                rec["graphInfoQueries"] = str(p.cmd("GRAPH.INFO", g, "QUERIES", keys=(g,)))[:200]
            except Exception as exc:  # noqa: BLE001
                rec["graphInfoError"] = f"{type(exc).__name__}: {exc}"
        finally:
            if changed:
                p.cmd("GRAPH.CONFIG", "SET", knob, orig)
            rec["valueAtEnd"] = get(knob)
            p.wipe(g)
        out.append({"probe": f"telemetry-config-{knob}-{value}", "commands": [f"GRAPH.CONFIG SET {knob} {value}", "60 x GRAPH.QUERY RETURN $i", f"XLEN {t}", f"GRAPH.CONFIG SET {knob} {orig}"],
                    "observed": rec})
    return out


def probe_telemetry_zero_with_existing_stream(p: Probe) -> list[dict]:
    """``MAX_INFO_QUERIES 0`` on a graph whose stream already holds entries: is the stream kept, trimmed, removed?"""
    g = PREFIX + "zero"
    t = tel(g)
    p.wipe(g)
    p.make(g, "z")
    for i in range(30):
        p.st.query(g, "RETURN $i", {"i": i})
    wait_for(lambda: p.c.xlen(guard(t)) >= 30, 10) if p.exists(t) else None
    time.sleep(3)
    rec = {"xlenBefore": p.c.xlen(guard(t))}
    orig = p.cmd("GRAPH.CONFIG", "GET", "MAX_INFO_QUERIES")[1]
    try:
        p.cmd("GRAPH.CONFIG", "SET", "MAX_INFO_QUERIES", 0)
        for i in range(10):
            p.st.query(g, "RETURN $i", {"i": i})
        time.sleep(4)
        rec["streamExistsAfter10MoreQueries"] = p.exists(t)
        rec["xlenAfter"] = p.c.xlen(t) if rec["streamExistsAfter10MoreQueries"] else None
        try:
            rec["slowlogStillWorks"] = str(p.cmd("GRAPH.SLOWLOG", g, keys=(g,)))[:120]
        except Exception as exc:  # noqa: BLE001
            rec["slowlogError"] = f"{type(exc).__name__}: {exc}"
    finally:
        p.cmd("GRAPH.CONFIG", "SET", "MAX_INFO_QUERIES", orig)
        rec["valueAtEnd"] = p.cmd("GRAPH.CONFIG", "GET", "MAX_INFO_QUERIES")[1]
        p.wipe(g)
    return [{"probe": "telemetry-zero-with-existing-stream", "commands": ["30 x GRAPH.QUERY", "GRAPH.CONFIG SET MAX_INFO_QUERIES 0", "10 x GRAPH.QUERY", "XLEN", "GRAPH.SLOWLOG", "restore"], "observed": rec}]


def probe_telemetry_stream(p: Probe) -> list[dict]:
    g = PREFIX + "tel"
    t = tel(g)
    out = []
    p.wipe(g)
    p.make(g, "t")
    appeared = wait_for(lambda: p.exists(t), 10)
    out.append({"probe": "telemetry-created-lazily", "commands": [f"GRAPH.QUERY {g} (3 writes)", f"EXISTS {t}", f"TYPE {t}", f"TTL {t}"],
                "observed": {"secondsUntilStreamExists": appeared, "type": str(p.c.type(guard(t))), "ttl": p.c.ttl(t)}})

    # parameter values land in the stream
    p.q(g, "MATCH (m:Meta) WHERE m.marker <> $secret RETURN count(m)", {"secret": "s3cr3t-value-S21"})
    seen = wait_for(lambda: any(b"s3cr3t-value-S21" in repr(e).encode() for e in p.c.xrevrange(guard(t), count=5)), 10)
    out.append({"probe": "telemetry-carries-parameters", "commands": [f"GRAPH.QUERY {g} ... CYPHER secret=..", f"XREVRANGE {t} + - COUNT 5"],
                "observed": {"secondsUntilParameterVisible": seen}})

    # does the stream cap itself? 1,200 cheap queries
    for i in range(1200):
        p.st.query(g, "RETURN $i", {"i": i})
    wait_for(lambda: p.c.xlen(guard(t)) >= 900, 10)
    time.sleep(3)
    out.append({"probe": "telemetry-cap", "commands": [f"1,200 x GRAPH.QUERY {g} 'RETURN $i'", f"XLEN {t}"], "observed": {"xlen": p.c.xlen(guard(t))}})

    # an admin XTRIM
    p.cmd("XTRIM", t, "MAXLEN", 10, keys=(t,))
    after_trim = p.c.xlen(guard(t))
    for i in range(30):
        p.st.query(g, "RETURN $i", {"i": i})
    time.sleep(3)
    out.append({"probe": "telemetry-xtrim-holds", "commands": [f"XTRIM {t} MAXLEN 10", f"XLEN {t}", f"30 x GRAPH.QUERY {g} 'RETURN $i'", f"XLEN {t}"],
                "observed": {"xlenAfterTrim": after_trim, "xlenAfter30MoreQueries": p.c.xlen(guard(t))}})

    # DEL of the stream: recreated by the next query?
    p.cmd("DEL", t, keys=(t,))
    gone = not p.exists(t)
    p.st.query(g, "RETURN 1")
    back = wait_for(lambda: p.exists(t), 10)
    out.append({"probe": "telemetry-del-recreated", "commands": [f"DEL {t}", f"EXISTS {t}", f"GRAPH.QUERY {g} 'RETURN 1'", f"EXISTS {t}"],
                "observed": {"goneRightAfterDel": gone, "secondsUntilRecreated": back}})

    # RENAME leaves it behind
    g2 = PREFIX + "tel2"
    p.wipe(g2)
    p.cmd("RENAME", g, g2, keys=(g, g2))
    out.append({"probe": "telemetry-rename", "commands": [f"RENAME {g} {g2}", f"EXISTS {t} {tel(g2)}"],
                "observed": {"oldStreamExistsRightAfter": p.exists(t), "newStreamExistsRightAfter": p.exists(tel(g2))}})
    p.st.query(g2, "RETURN 1")
    nw = wait_for(lambda: p.exists(tel(g2)), 10)
    out[-1]["observed"]["secondsUntilNewStreamAppears"] = nw
    out[-1]["observed"]["oldStreamExistsAfterQueryOnNew"] = p.exists(t)
    p.wipe(g, g2)
    return out


def probe_delete_removal(p: Probe, trials: int = 8) -> list[dict]:
    """When does ``telemetry{<graph>}`` disappear after ``GRAPH.DELETE``? Polled every 10 ms for 30 s, with and without a
    query issued just before the delete (the stream is written asynchronously, ~1-3 s after a query)."""
    out = []
    async_delete = p.cmd("GRAPH.CONFIG", "GET", "ASYNC_DELETE")
    for variant in ("settled", "query-just-before-delete", "large-graph-settled", "settled-then-query-just-before-delete"):
        rows = []
        for n in range(trials):
            g = PREFIX + f"del{n}"
            t = tel(g)
            p.wipe(g)
            p.q(g, "CREATE (:A {v: $v})", {"v": f"x{n}"})
            if variant == "large-graph-settled":
                for _ in range(3):
                    p.st.query(g, "UNWIND range(1, 100000) AS i CREATE (:B {v: i})-[:R]->(:B {v: -i})")
            if variant.endswith("settled") or variant.startswith("settled"):
                wait_for(lambda: p.exists(t), 10)
                time.sleep(2.5)
            else:
                p.st.query(g, "RETURN $v", {"v": f"y{n}"})
            if variant == "settled-then-query-just-before-delete":
                p.st.query(g, "RETURN $v", {"v": f"z{n}"})   # an entry is now in flight (written ~1-3 s later)
            existed = p.exists(t)
            t0 = time.monotonic()
            p.st.delete(g)
            graph_gone = not p.exists(g)
            ts = []
            gone_at = None
            reappeared = False
            while time.monotonic() - t0 < 30:
                e = p.exists(t)
                ts.append(e)
                if not e and gone_at is None:
                    gone_at = time.monotonic() - t0
                if e and gone_at is not None:
                    reappeared = True
                if gone_at is not None and time.monotonic() - t0 > gone_at + 8:
                    break
                time.sleep(0.01)
            rows.append({"streamExistedBeforeDelete": existed, "graphGoneRightAfter": graph_gone,
                         "secondsUntilStreamGone": None if gone_at is None else round(gone_at, 3), "reappeared": reappeared,
                         "streamStillThere30s": gone_at is None})
        out.append({"probe": f"telemetry-after-graph-delete-{variant}",
                    "commands": [f"GRAPH.QUERY ... ; (variant: {variant}) GRAPH.DELETE <g>; poll EXISTS telemetry{{<g>}} every 10 ms, up to 30 s"],
                    "observed": {"ASYNC_DELETE": async_delete, "trials": rows, "gone_after_s": sorted(r["secondsUntilStreamGone"] for r in rows if r["secondsUntilStreamGone"] is not None),
                                 "neverGone": sum(r["streamStillThere30s"] for r in rows), "reappeared": sum(r["reappeared"] for r in rows)}})
    for n in range(trials):
        p.wipe(PREFIX + f"del{n}")
    return out


def probe_ping_stall_read(p: Probe) -> list[dict]:
    """Does a long ``GRAPH.RO_QUERY`` stall the server's other clients (``PING``)? Three read shapes, same 7,000-node graph."""
    import redis
    g = PREFIX + "stall"
    p.wipe(g)
    p.make(g, "s", n=7000)
    shapes = {
        "cartesian-self-join-filter": "MATCH (a:N), (b:N) WHERE a.v + b.v > 0 RETURN count(*)",
        "unwind-range-count": "UNWIND range(1, 30000000) AS y RETURN count(y)",
        "label-scan-with-expression": "MATCH (a:N) UNWIND range(1, 3000) AS k WITH a, k WHERE a.v + k > 0 RETURN count(*)",
    }
    out = []
    for name, cypher in shapes.items():
        pings: list[tuple[float, float]] = []
        stop, started = threading.Event(), threading.Event()
        res: dict = {}

        def reader():
            started.set()
            t0 = time.monotonic()
            try:
                res["rows"] = p.st.ro_query(g, cypher, None, 120000).result_set[0][0]
            except Exception as exc:  # noqa: BLE001
                res["error"] = f"{type(exc).__name__}: {exc}"
            res["seconds"] = round(time.monotonic() - t0, 2)

        def pinger():
            c2 = redis.Redis(host="127.0.0.1", port=6390, password=S.L.password(), retry=S.L.zero_retry(), socket_timeout=60)
            c2.ping()
            t00 = time.perf_counter()
            while not stop.is_set():
                t = time.perf_counter()
                c2.ping()
                pings.append((round(t - t00, 3), (time.perf_counter() - t) * 1000))
                time.sleep(0.01)

        pt = threading.Thread(target=pinger)
        pt.start()
        time.sleep(0.3)
        th = threading.Thread(target=reader)
        th.start()
        th.join()
        stop.set()
        pt.join()
        lat = [x[1] for x in pings]
        out.append({"probe": f"ping-stall-read-{name}", "commands": [f"GRAPH.RO_QUERY {g} {cypher!r}", "PING every 10 ms from a second connection"],
                    "observed": {"read": res, "pings": len(lat), "maxPingMs": round(max(lat), 1), "pingsOver100ms": sum(x > 100 for x in lat),
                                 "pingsOver1s": sum(x > 1000 for x in lat), "slowestPingStartedAtS": max(pings, key=lambda x: x[1])[0]}})
    p.wipe(g)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=("rollback", "telemetry", "delete", "stall"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6390)
    args = ap.parse_args(argv)
    p = Probe(S.SpikeStore(args.host, args.port))
    results = []
    if args.only in (None, "rollback"):
        results += probe_rollback(p)
    if args.only in (None, "telemetry"):
        results += probe_telemetry_config(p) + probe_telemetry_config_effect(p) + probe_telemetry_zero_with_existing_stream(p) + probe_telemetry_stream(p)
    if args.only in (None, "delete"):
        results += probe_delete_removal(p)
    if args.only in (None, "stall"):
        results += probe_ping_stall_read(p)
    dest = HERE / "results"
    dest.mkdir(exist_ok=True)
    out = dest / f"swap_probe{'' if not args.only else '_' + args.only}.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
