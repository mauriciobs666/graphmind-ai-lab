"""S22 (P0-16) -- the verdict lock, the fsynced verdict log and swap/replay, probed on ``ccg-falkordb``.

Contract: proposal ``docs/plans/connected-code-graph.md`` §3.8 (lock, patch-and-advance, swap replay, post-``RENAME``
remainder replay, crash re-convergence), graph note ``docs/plans/connected-code-graph-graph.md`` §6.4 (B6), p0-2 §4.5 (S22).

* the lock is ``SET <system>:verdict_lock <token> NX PX <ttl>``; release is a compare-and-delete ``EVAL`` on the token;
* a verdict, under the lock: append + ``fsync`` the JSONL log, then (:meth:`Engine.catch_up`) replay every entry past
  the live ``:Snapshot.verdictLogOffset``, each as **one query** that patches the ``Link`` and advances the offset;
* a swap, under the lock: replay the build key past its stamped offset, ``RENAME build live``, then **replay the
  remainder** (live offset < log length) before releasing -- the P2-3 cover for a lock that expired mid-swap;
* after a kill between append and patch, the next verdict or swap replays from the offset and the log and graph converge.

Keys: only ``ccg_p0spike__*`` graphs and ``ccg_p0spike__*:verdict_lock`` lock keys (:func:`guard`). Everything here is
P0-scale: the TTL and the 20k-link graph of :func:`measure` are the S22 sizing inputs, not P1 values.

Usage: ``python lock_replay.py measure`` (writes ``results/lock_replay_timing.json``)
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # cpg/
sys.path.insert(0, str(HERE.parent / "vertical-slice"))

HOST, PORT = "127.0.0.1", 6390
_KEY = re.compile(r"ccg_p0spike__[A-Za-z0-9_]+(:verdict_lock)?")

RELEASE_LUA = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"
#: one query: patch the Link (a missing link is a no-op: OPTIONAL MATCH) and advance the offset monotonically
PATCH_QUERY = ("MATCH (s:Snapshot) OPTIONAL MATCH (l:Link {linkId: $id}) "
               "SET l.verdict = $v, l.verdictBy = $by, "
               "s.verdictLogOffset = CASE WHEN s.verdictLogOffset < $off THEN $off ELSE s.verdictLogOffset END")
OFFSET_QUERY = "MATCH (s:Snapshot) RETURN s.verdictLogOffset"


class LockError(RuntimeError):
    pass


def guard(key: str) -> str:
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        raise LockError(f"refusing key {key!r}: S22 touches only ccg_p0spike__* graphs and their :verdict_lock keys")
    return key


def connect(user: str | None = None, password: str | None = None, socket_timeout: float = 120.0):
    """A zero-retry falkordb-py client (the loader's rule); admin when ``user`` is None."""
    from falkordb import FalkorDB
    from load import password as admin_password, zero_retry
    return FalkorDB(host=HOST, port=PORT, username=user, password=password if user else admin_password(),
                    socket_timeout=socket_timeout, socket_connect_timeout=10, retry=zero_retry())


# ---------------------------------------------------------------------------------------------
# The log: append-only JSONL, fsynced, torn-tail safe.
# ---------------------------------------------------------------------------------------------


class Log:
    def __init__(self, path):
        self.path = Path(path)

    def _repair(self) -> None:
        """Drop a torn tail (a kill mid-write leaves bytes with no newline) so an append never glues onto it."""
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            with open(self.path, "r+b") as f:
                f.truncate(data.rfind(b"\n") + 1)
                f.flush()
                os.fsync(f.fileno())

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        complete = data[: data.rfind(b"\n") + 1]
        return [json.loads(ln) for ln in complete.split(b"\n") if ln]

    def append(self, entry: dict) -> int:
        """One ``write`` + ``fsync``; returns the log length including this entry."""
        self._repair()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(entry, sort_keys=True) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        return len(self.entries())


# ---------------------------------------------------------------------------------------------
# The lock.
# ---------------------------------------------------------------------------------------------


class Lock:
    def __init__(self, conn, key: str, ttl_ms: int):
        self.conn, self.key, self.ttl_ms = conn, guard(key), int(ttl_ms)

    def acquire(self, wait_s: float = 0.0, poll_s: float = 0.02, clock=time.monotonic, sleep=time.sleep) -> str | None:
        token = secrets.token_hex(8)
        deadline = clock() + wait_s
        while True:
            if self.conn.set(self.key, token, nx=True, px=self.ttl_ms):
                return token
            if clock() >= deadline:
                return None
            sleep(poll_s)

    def release(self, token: str) -> bool:
        """Compare-and-delete: True only if the token still held the key."""
        return bool(self.conn.eval(RELEASE_LUA, 1, self.key, token))


# ---------------------------------------------------------------------------------------------
# The engine: verdict() and swap().
# ---------------------------------------------------------------------------------------------


class Engine:
    """``system`` is the live graph key (``ccg_p0spike__<x>``); its lock key is ``<system>:verdict_lock``. ``hook(name)``
    is called at named points so tests can sleep or die there: ``locked``, ``after_append``, ``between_replay``,
    ``before_rename``, ``after_rename``, ``before_release``."""

    def __init__(self, db, system: str, log: Log, ttl_ms: int, hook: Callable[[str], None] | None = None):
        self.db = db
        self.conn = db.connection if db is not None else None
        self.system = guard(system)
        self.log = log
        self.lock = Lock(self.conn, system + ":verdict_lock", ttl_ms)
        self.hook = hook or (lambda name: None)

    # -- graph access (overridden by the fake in the unit tests) --------------------------------

    def _offset(self, key: str) -> int:
        rs = self.db.select_graph(guard(key)).ro_query(OFFSET_QUERY, timeout=60000).result_set
        return int(rs[0][0]) if rs and rs[0][0] is not None else 0

    def _patch(self, key: str, entry: dict, off: int) -> None:
        self.db.select_graph(guard(key)).query(
            PATCH_QUERY, {"id": entry["linkId"], "v": entry["verdict"], "by": entry["by"], "off": off}, timeout=60000)

    def _rename(self, src: str, dst: str) -> None:
        self.conn.rename(guard(src), guard(dst))

    # -- the protocol -----------------------------------------------------------------------

    def catch_up(self, key: str) -> int:
        """Replay every log entry past ``key``'s offset, one patch-and-advance query each; returns how many."""
        entries = self.log.entries()
        off = self._offset(key)
        n = 0
        for i in range(off, len(entries)):
            self._patch(key, entries[i], i + 1)
            n += 1
            self.hook("between_replay")
        return n

    def verdict(self, link_id: str, verdict: str, by: str, wait_s: float = 5.0) -> dict:
        token = self.lock.acquire(wait_s)
        if token is None:
            raise LockError("verdict lock not acquired")
        try:
            self.hook("locked")
            self.log.append({"linkId": link_id, "verdict": verdict, "by": by})
            self.hook("after_append")
            return {"replayed": self.catch_up(self.system)}
        finally:
            self.lock.release(token)

    def swap(self, build: str, wait_s: float = 5.0) -> dict:
        token = self.lock.acquire(wait_s)
        if token is None:
            raise LockError("swap lock not acquired")
        res: dict = {}
        try:
            self.hook("locked")
            t0 = time.perf_counter()
            res["replayed"] = self.catch_up(build)
            res["replay_s"] = time.perf_counter() - t0
            self.hook("before_rename")
            t0 = time.perf_counter()
            self._rename(build, self.system)
            res["rename_s"] = time.perf_counter() - t0
            self.hook("after_rename")
            t0 = time.perf_counter()
            res["remainder"] = self.catch_up(self.system)
            res["remainder_s"] = time.perf_counter() - t0
            self.hook("before_release")
            return res
        finally:
            res["released"] = self.lock.release(token)


# ---------------------------------------------------------------------------------------------
# Fixtures used by the tests and by measure().
# ---------------------------------------------------------------------------------------------


def make_graph(db, key: str, n_links: int, offset: int = 0, marker: str = "x") -> None:
    """``(:Snapshot {verdictLogOffset})`` plus ``n_links`` ``Link {linkId: 'L<i>'}`` with an index, in batches <= 20k."""
    g = db.select_graph(guard(key))
    g.query("CREATE INDEX FOR (l:Link) ON (l.linkId)")
    g.query("CREATE (:Snapshot {verdictLogOffset: $o, marker: $m})", {"o": offset, "m": marker})
    for lo in range(0, n_links, 20_000):
        hi = min(n_links, lo + 20_000)
        g.query("UNWIND range($lo, $hi - 1) AS i CREATE (:Link {linkId: 'L' + toString(i)})",
                {"lo": lo, "hi": hi}, timeout=120000)


def link_verdict(db, key: str, link_id: str):
    rs = db.select_graph(guard(key)).ro_query("MATCH (l:Link {linkId: $id}) RETURN l.verdict", {"id": link_id},
                                              timeout=60000).result_set
    return rs[0][0] if rs else "<no link>"


def drop(db, *keys: str) -> None:
    for k in keys:
        guard(k)
        if db.connection.exists(k):
            if ":" in k:
                db.connection.delete(k)
            else:
                db.select_graph(k).delete()


# ---------------------------------------------------------------------------------------------
# Sizing run (S22): worst replay of 1,000 log entries + RENAME + remainder check.
# ---------------------------------------------------------------------------------------------


def measure(reps: int = 5, n_links: int = 20_000, n_entries: int = 1_000, out: Path | None = None) -> dict:
    import random
    import tempfile
    db = connect()
    sysk, build = "ccg_p0spike__lkm", "ccg_p0spike__lkm_b"
    runs = []
    rnd = random.Random(20261104)
    for _ in range(reps):
        drop(db, sysk, build, sysk + ":verdict_lock")
        make_graph(db, sysk, n_links, 0, "old")
        make_graph(db, build, n_links, 0, "new")
        with tempfile.TemporaryDirectory() as d:
            log = Log(Path(d) / "verdicts.jsonl")
            for _i in range(n_entries):
                log.append({"linkId": f"L{rnd.randrange(n_links)}", "verdict": "dismissed", "by": "probe"})
            eng = Engine(db, sysk, log, ttl_ms=120_000)
            res = eng.swap(build)
            t0 = time.perf_counter()
            eng.verdict("L1", "confirmed", "probe")
            res["single_verdict_s"] = time.perf_counter() - t0
            res["offsetAfter"] = eng._offset(sysk)
            runs.append(res)
        drop(db, sysk, build, sysk + ":verdict_lock")
    worst = max(r["replay_s"] + r["rename_s"] + r["remainder_s"] for r in runs)
    rec = {"links": n_links, "entries": n_entries, "reps": reps, "runs": runs, "worstTotalS": worst,
           "ttlRuleS": 3 * worst}
    if out:
        out.write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    if sys.argv[1:] == ["measure"]:
        rec = measure(out=HERE / "results" / "lock_replay_timing.json")
        print(json.dumps({k: v for k, v in rec.items() if k != "runs"}, indent=1))
    else:
        print(__doc__)
