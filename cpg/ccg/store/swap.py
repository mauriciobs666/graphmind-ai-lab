"""The swap (C-1001 P1-19, dispatch 1): ``RENAME build live`` under the verdict lock, with the credentials split of ``-p1-graph.md`` §7.

Contract: ``docs/plans/connected-code-graph-p1-graph.md`` §7 (who may do what), §8 P1-19 (c); ``-p0-graph.md`` §6 rows 8, 9, 11 (lock ttl,
lock/replay protocol, plain ``RENAME``); ``docs/plans/connected-code-graph-p1-linking.md`` P1-19. A port of the protocol exercised by
``spikes/store/lock_replay.py``, with an EMPTY verdict log: the replay hook is a no-op that P4 fills.

    swap(build_conn, verdict_conn, system, build_id)

* ``build_conn`` is a falkordb-py client authenticated as ``acl_user('build', system)``: it takes the lock, reads the build key, replays
  onto the build key (the pre-``RENAME`` replay) and runs the plain ``RENAME build live``. That credential holds the live key write-only, so
  it can ``RENAME`` onto it and do nothing else with it.
* ``verdict_conn`` is authenticated as ``acl_user('verdict', system)``, the only credential that reads and writes the live key: every
  post-``RENAME`` step on live (the remainder check and replay) runs through it. There is no third, admin, connection to pass.
* the lock is ``SET <lock> <token> NX PX <ttl>`` and is released by a compare-and-delete ``EVAL``: it never deletes another holder's lock.
  A lock that lapses mid-swap is the P2-3 case the post-``RENAME`` remainder check covers: the lock is re-read right after ``RENAME`` and a
  loss is **reported** (``SwapResult.lock_held_after_rename``, and ``released`` False at the end), not raised, because live already holds the
  new graph; the remainder check and replay still run (the replay is offset-based, so it is idempotent). A loss in the window after that
  read is covered by the remainder replay alone. Before ``RENAME`` a lost lock aborts.
* ``timeout`` (milliseconds, default None) is passed to both stamp reads. None leaves them to the server's own default, so a caller that
  holds a nonzero ``TIMEOUT_MAX`` passes it (``LoadReport.timeout_max``); the lock-ttl re-derivation of P1-19 dispatch 2 counts the capped
  stamp reads in the replay and remainder phases.
* zero old snapshots are kept: the old live graph is simply replaced.
"""
from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from ccg.sysname import build_key, check_system_name, live_key, lock_key

#: P0's measured default (``-p0-graph.md`` §6 row 8: worst replay + RENAME + remainder 0.464 s, 3x = 1.39 s -> 2,000 ms). P1-19 dispatch 2
#: re-derives it from the proving repo; this is a default, not a design constant.
LOCK_TTL_MS = 2_000
LOCK_WAIT_S = 5.0

#: Compare-and-delete: True only if the token still holds the lock.
RELEASE_LUA = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"

#: One row when the key holds a complete build (``:Snapshot`` is written last), none otherwise.
STAMP_QUERY = "MATCH (s:Snapshot) RETURN s.verdictLogOffset, s.verdictLogHash"

#: P1 stores an empty verdict log's stamp (``-p1-graph.md`` §3); the engine fails on anything else before P4.
VERDICT_LOG_OFFSET = 0
EMPTY_VERDICT_LOG_HASH = "c482b1cf54631b37eb4572dc86a68fb1"   # ``ids.tuple_id(["verdictLog"])``; tested equal to it


class SwapError(RuntimeError):
    """The swap did not complete. ``renamed`` says whether ``RENAME`` had already run (the live key then holds the new graph)."""

    def __init__(self, message: str, *, renamed: bool = False):
        super().__init__(message)
        self.renamed = renamed


@dataclass
class SwapResult:
    system: str = ""
    build_key: str = ""
    live_key: str = ""
    renamed: bool = False
    #: the lock still held by this swap's token right after ``RENAME`` (a loss is reported, never fatal: the rename already happened)
    lock_held_after_rename: bool = False
    released: bool = False
    replayed_before: int = 0
    replayed_after: int = 0
    ttl_ms: int = 0
    replay_s: float = 0.0       # lock held -> end of the pre-RENAME replay (the lock-ttl inputs, ``-p0-graph.md`` §6 row 8)
    rename_s: float = 0.0
    remainder_s: float = 0.0    # RENAME done -> end of the post-RENAME check and replay
    total_s: float = 0.0


def replay_verdicts(conn, key: str, system: str) -> int:
    """Replay every verdict-log entry past ``key``'s stamped offset onto ``key``; returns how many. P1's log is empty: nothing to replay.

    P4 fills this body (``conn`` is the credential allowed to write ``key``: the build credential before ``RENAME``, the verdict credential
    after). Until then it must not touch the connection."""
    return 0


def _stamp(conn, key: str, *, read_only: bool, timeout: int | None) -> tuple | None:
    graph = conn.select_graph(key)
    res = (graph.ro_query if read_only else graph.query)(STAMP_QUERY, None, timeout)
    rows = res.result_set
    if len(rows) != 1:
        return None
    return tuple(rows[0])


def _check_stamp(stamp: tuple | None, key: str, where: str, *, renamed: bool) -> None:
    if stamp is None:
        raise SwapError(f"{where}: {key} has no single :Snapshot (an incomplete build, or not a build)", renamed=renamed)
    if stamp != (VERDICT_LOG_OFFSET, EMPTY_VERDICT_LOG_HASH):
        raise SwapError(f"{where}: {key} does not carry the empty verdict log's stamp (offset 0 and its hash); P1 expects an empty log",
                        renamed=renamed)


def _acquire(conn, key: str, token: str, ttl_ms: int, wait_s: float, clock, sleep) -> bool:
    deadline = clock() + wait_s
    while True:
        if conn.set(key, token, nx=True, px=ttl_ms):
            return True
        if clock() >= deadline:
            return False
        sleep(0.02)


def swap(build_conn, verdict_conn, system: str, build_id: str, *, ttl_ms: int = LOCK_TTL_MS, wait_s: float = LOCK_WAIT_S,
         timeout: int | None = None, replay: Callable | None = None, hook: Callable[[str], None] | None = None,
         clock=time.monotonic, sleep=time.sleep) -> SwapResult:
    """Make the complete build ``build_key(system, build_id)`` the live graph. Returns the phase times; raises :class:`SwapError`.

    ``replay(conn, key, system) -> int`` defaults to the module's :func:`replay_verdicts`, looked up at call time. ``hook(name)`` is called at
    ``locked``, ``before_rename``, ``after_rename`` and ``before_release`` so a test can stall or die there."""
    system = check_system_name(system)
    build, live, lock = build_key(system, build_id), live_key(system), lock_key(system)
    if not isinstance(ttl_ms, int) or ttl_ms <= 0:
        raise ValueError("ttl_ms must be a positive integer")
    replay_fn = replay if replay is not None else replay_verdicts
    note = hook if hook is not None else (lambda name: None)
    res = SwapResult(system=system, build_key=build, live_key=live, ttl_ms=ttl_ms)

    token = secrets.token_hex(16)
    if not _acquire(build_conn.connection, lock, token, ttl_ms, wait_s, clock, sleep):
        raise SwapError(f"the verdict lock {lock} is held; swap not attempted")
    t_start = clock()
    try:
        note("locked")
        if not build_conn.connection.exists(build):
            raise SwapError(f"build key {build} does not exist")
        _check_stamp(_stamp(build_conn, build, read_only=True, timeout=timeout), build, "before RENAME", renamed=False)
        res.replayed_before = replay_fn(build_conn, build, system)
        res.replay_s = clock() - t_start

        note("before_rename")
        held = build_conn.connection.get(lock)
        if (held.decode() if isinstance(held, bytes) else held) != token:
            raise SwapError(f"the verdict lock {lock} was lost before RENAME; nothing renamed")
        t0 = clock()
        build_conn.connection.rename(build, live)            # plain RENAME: the one write the build credential may make to live
        res.renamed = True
        res.rename_s = clock() - t0

        note("after_rename")
        held = build_conn.connection.get(lock)
        res.lock_held_after_rename = (held.decode() if isinstance(held, bytes) else held) == token
        t0 = clock()
        _check_stamp(_stamp(verdict_conn, live, read_only=False, timeout=timeout), live, "after RENAME", renamed=True)
        res.replayed_after = replay_fn(verdict_conn, live, system)
        res.remainder_s = clock() - t0
        note("before_release")
        return res
    finally:
        try:
            res.released = bool(build_conn.connection.eval(RELEASE_LUA, 1, lock, token))
        except Exception:  # noqa: BLE001 -- an unreleased lock lapses by its ttl; it must not replace the error that is propagating
            res.released = False
        res.total_s = clock() - t_start
