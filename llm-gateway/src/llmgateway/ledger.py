"""The holder-owned FR-10 ledger (docs/plans/llm-gateway.md §3.4 "FR-10 ledger").

One SQLite file in the holder's own directory. A call reserves `limit + len(body)` tokens in one
`BEGIN IMMEDIATE` transaction that checks, together: the release cap, the run's total (or the
shared `cap` bucket for the window) and the holder-wide window total. A refused reservation
raises `Refusal(code)` with a §3.3 code; nothing is clamped. After the call the reservation is
`settle`d to the reported usage (which may exceed it: recorded as reported, and the next call is
refused), `keep`t (the safe floor until the window rolls off) or `release`d (an error-storm
limiter caps releases per window). The clock is injected so the rolling window is testable.
A second process on the same ledger is refused by a lock file.
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

MAX_WINDOW_SECONDS = 30 * 24 * 3600
CAP_BUCKET = "\x00cap"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    run_id TEXT NOT NULL,          -- '' for the shared cap bucket
    bucket TEXT NOT NULL,          -- 'run' | 'cap'
    amount INTEGER NOT NULL,       -- tokens counted now (0 once released)
    state TEXT NOT NULL,           -- 'open' | 'settled' | 'kept' | 'released'
    released_ts REAL
);
CREATE INDEX IF NOT EXISTS reservations_run ON reservations(bucket, run_id);
CREATE INDEX IF NOT EXISTS reservations_ts ON reservations(ts);
"""


class Refusal(Exception):
    """A holder refusal; `code` is one of `failures.HOLDER_REFUSALS`."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class LedgerLocked(Exception):
    """Another process holds this ledger."""


def positive_int(value: object, name: str, maximum: int | None = None) -> int:
    """A positive integer (never a bool or float); raises ValueError otherwise."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class Ceilings:
    """The mandatory holder-wide window ceiling and the error-storm release cap."""

    window_seconds: int
    window_max_tokens: int
    window_max_released: int

    @classmethod
    def from_table(cls, table: dict[str, object]) -> Ceilings:
        return cls(
            positive_int(table.get("window_seconds"), "window_seconds", MAX_WINDOW_SECONDS),
            positive_int(table.get("window_max_tokens"), "window_max_tokens"),
            positive_int(table.get("window_max_released"), "window_max_released"),
        )


class Ledger:
    def __init__(self, path: str | os.PathLike[str], *, clock: Callable[[], float] = time.time):
        self._path = os.fspath(path)
        self._clock = clock
        self._mutex = threading.Lock()
        self._lock_fd = os.open(
            self._path + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
        )
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._lock_fd)
            raise LedgerLocked(f"{self._path} is held by another holder") from exc
        old = os.umask(0o077)  # the DB, -wal and -shm are created 0600 (holder only)
        try:
            self._db = sqlite3.connect(
                self._path, isolation_level=None, check_same_thread=False, timeout=30
            )
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
        finally:
            os.umask(old)
        os.chmod(self._path, 0o600)

    def close(self) -> None:
        self._db.close()
        os.close(self._lock_fd)

    # --- reads (tests and `calls`) -----------------------------------------------------------

    def window_total(self, window_seconds: int) -> int:
        row = self._db.execute(
            "SELECT COALESCE(SUM(amount),0) FROM reservations WHERE ts > ?",
            (self._clock() - window_seconds,),
        ).fetchone()
        return int(row[0])

    def run_total(self, run_id: str) -> int:
        row = self._db.execute(
            "SELECT COALESCE(SUM(amount),0) FROM reservations WHERE bucket='run' AND run_id=?",
            (run_id,),
        ).fetchone()
        return int(row[0])

    # --- the reservation transaction ---------------------------------------------------------

    def reserve(
        self,
        *,
        amount: int,
        ceilings: Ceilings,
        run_id: str = "",
        run_bound: int | None = None,
        cap_bound: int | None = None,
    ) -> int:
        """Reserve `amount` or raise `Refusal`. Exactly one of `run_bound` (a matched run id) and
        `cap_bound` (the shared bucket of an unmatched or missing id) is given."""
        if (run_bound is None) == (cap_bound is None):
            raise ValueError("exactly one of run_bound and cap_bound")
        now = self._clock()
        cutoff = now - ceilings.window_seconds
        with self._mutex:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                released = self._db.execute(
                    "SELECT COUNT(*) FROM reservations WHERE released_ts > ?", (cutoff,)
                ).fetchone()[0]
                if released >= ceilings.window_max_released:
                    raise Refusal("release_cap_reached")
                if run_bound is not None:
                    used = self._db.execute(
                        "SELECT COALESCE(SUM(amount),0) FROM reservations "
                        "WHERE bucket='run' AND run_id=?",
                        (run_id,),
                    ).fetchone()[0]
                    bucket, rid, bound = "run", run_id, run_bound
                else:
                    used = self._db.execute(
                        "SELECT COALESCE(SUM(amount),0) FROM reservations "
                        "WHERE bucket='cap' AND ts > ?",
                        (cutoff,),
                    ).fetchone()[0]
                    bucket, rid, bound = "cap", "", cap_bound
                if used + amount > bound:
                    raise Refusal("run_bound_reached")
                window_used = self._db.execute(
                    "SELECT COALESCE(SUM(amount),0) FROM reservations WHERE ts > ?", (cutoff,)
                ).fetchone()[0]
                if window_used + amount > ceilings.window_max_tokens:
                    raise Refusal("window_ceiling_reached")
                cur = self._db.execute(
                    "INSERT INTO reservations(ts, run_id, bucket, amount, state) "
                    "VALUES (?,?,?,?, 'open')",
                    (now, rid, bucket, amount),
                )
                rowid = int(cur.lastrowid)
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")
            return rowid

    # --- outcomes ----------------------------------------------------------------------------

    def settle(self, rid: int, tokens: int) -> None:
        """Count the provider-reported usage (above or below the reservation)."""
        self._finish("UPDATE reservations SET amount=?, state='settled' WHERE id=?", (tokens, rid))

    def keep(self, rid: int) -> None:
        """Keep the reservation as the floor until its window rolls off."""
        self._finish("UPDATE reservations SET state='kept' WHERE id=?", (rid,))

    def release(self, rid: int, ceilings: Ceilings) -> bool:
        """Release the whole reservation unless the window's release cap is spent (then it is
        kept). Returns whether it was released."""
        now = self._clock()
        with self._mutex:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                released = self._db.execute(
                    "SELECT COUNT(*) FROM reservations WHERE released_ts > ?",
                    (now - ceilings.window_seconds,),
                ).fetchone()[0]
                if released >= ceilings.window_max_released:
                    self._db.execute("UPDATE reservations SET state='kept' WHERE id=?", (rid,))
                    ok = False
                else:
                    self._db.execute(
                        "UPDATE reservations SET amount=0, state='released', released_ts=? "
                        "WHERE id=?",
                        (now, rid),
                    )
                    ok = True
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")
            return ok

    def _finish(self, sql: str, params: tuple[object, ...]) -> None:
        with self._mutex:
            self._db.execute(sql, params)
