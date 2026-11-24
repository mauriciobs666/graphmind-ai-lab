"""Call records: one SQLite row per attempt (docs/plans/llm-gateway.md §3.7; FR-5, AC-17,
AC-19, AC-24).

The store lives in the caller's `state_dir` (dir 0700, files 0600) and is *advisory*: the hosted
ledger is the holder's. `RecordSink` keeps it replaceable. The schema has no column for headers,
provider error bodies, request ids or organization/workspace ids, so none can be stored; the only
free-text columns are the opt-in prompt/response text, written only when the caller proves the
resolved scopes allow it (`text_allowed`), and values of the closed or pattern-checked columns
are validated on write, so an error body cannot be smuggled in through them.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from llmgateway.types import IDENT, FailureClass, GatewayError

DB_NAME = "records.sqlite3"
SCHEMA_VERSION = 1
OUTCOMES = frozenset({"ok", "failed"})
KINDS = frozenset({"first", "retry", "fallback"})
_IDENT = IDENT
UNRECORDABLE_MODEL = "invalid-reported-model"
_STATE = re.compile(r"[a-z0-9_:.-]{1,64}")


class RecordError(GatewayError):
    """An attempt that cannot be recorded (invalid field, duplicate attempt, bad store)."""


def new_call_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class Attempt:
    """One attempt of one call (FR-5). `scopes` are the effective scope ids; `kind` is how this
    attempt arose (first try, same-provider retry, next-in-chain fallback) and `reason` the
    failure class of the attempt it follows. Optional text is the opt-in payload."""

    call_id: str
    attempt: int
    component: str
    scopes: tuple[str, ...]
    call_site: str
    provider: str
    model: str
    hosted: bool
    outcome: str
    run_id: str | None = None
    kind: str = "first"
    reason: FailureClass | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int | None = None
    failure_class: FailureClass | None = None
    loaded_state: str | None = None
    request_text: str | None = None
    response_text: str | None = None

    def __post_init__(self) -> None:
        for name in ("call_id", "component", "call_site", "provider", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _IDENT.fullmatch(value):
                raise RecordError(f"attempt.{name} is not a plain identifier")
        if self.run_id is not None and not (
            isinstance(self.run_id, str) and _IDENT.fullmatch(self.run_id)
        ):
            raise RecordError("attempt.run_id is not a plain identifier")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 1:
            raise RecordError("attempt.attempt must be an int >= 1")
        if not isinstance(self.hosted, bool):
            raise RecordError("attempt.hosted must be a bool")
        if self.outcome not in OUTCOMES:
            raise RecordError(f"attempt.outcome must be one of {sorted(OUTCOMES)}")
        if self.kind not in KINDS:
            raise RecordError(f"attempt.kind must be one of {sorted(KINDS)}")
        for name in ("reason", "failure_class"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, FailureClass):
                raise RecordError(f"attempt.{name} must be a FailureClass")
        if (self.outcome == "failed") != (self.failure_class is not None):
            raise RecordError("a failed attempt has a failure class and an ok one has none")
        if (self.kind == "first") != (self.reason is None):
            raise RecordError("a first attempt has no reason and every other kind has one")
        for name in ("input_tokens", "output_tokens", "duration_ms"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise RecordError(f"attempt.{name} must be a non-negative int or None")
        if self.loaded_state is not None and not (
            isinstance(self.loaded_state, str) and _STATE.fullmatch(self.loaded_state)
        ):
            raise RecordError("attempt.loaded_state must match [a-z0-9_:.-]{1,64}")
        if not isinstance(self.scopes, tuple) or not all(
            isinstance(s, str) and 0 < len(s) <= 128 for s in self.scopes
        ):
            raise RecordError("attempt.scopes must be a tuple of scope ids")
        for name in ("request_text", "response_text"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise RecordError(f"attempt.{name} must be a string or None")


class RecordSink(Protocol):
    """Where attempts go; the SQLite store is the default implementation (§3.10)."""

    def write(self, attempt: Attempt, *, text_allowed: bool = False) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    ts            REAL    NOT NULL,
    call_id       TEXT    NOT NULL,
    attempt       INTEGER NOT NULL,
    component     TEXT    NOT NULL,
    scopes        TEXT    NOT NULL,
    call_site     TEXT    NOT NULL,
    run_id        TEXT,
    provider      TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    hosted        INTEGER NOT NULL,
    kind          TEXT    NOT NULL,
    reason        TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    duration_ms   INTEGER,
    outcome       TEXT    NOT NULL,
    failure_class TEXT,
    loaded_state  TEXT,
    request_text  TEXT,
    response_text TEXT,
    PRIMARY KEY (call_id, attempt)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS attempts_run ON attempts (run_id) WHERE run_id IS NOT NULL;
"""
SCOPE_SEPARATOR = "\n"  # scope ids are [a-z0-9:._-]: never contain a newline


class RecordStore:
    """SQLite attempt store under `state_dir`. Each call uses its own connection (WAL, busy
    timeout), so threads and processes can write concurrently."""

    def __init__(self, state_dir: str | os.PathLike[str], *, busy_timeout_s: float = 30.0):
        self._dir = os.fspath(state_dir)
        self._path = os.path.join(self._dir, DB_NAME)
        self._timeout = busy_timeout_s
        self._init()

    @property
    def path(self) -> str:
        return self._path

    def _init(self) -> None:
        os.makedirs(self._dir, mode=0o700, exist_ok=True)
        # Create the file 0600 before SQLite does: WAL/SHM siblings inherit its mode.
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise RecordError(f"{self._path}: unsupported schema version {version}")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _connect(self) -> _Closing:
        conn = sqlite3.connect(
            self._path, timeout=self._timeout, isolation_level=None, check_same_thread=True
        )
        conn.row_factory = sqlite3.Row
        return _Closing(conn)

    def write(self, attempt: Attempt, *, text_allowed: bool = False) -> None:
        """Append one attempt. Text is stored only when `text_allowed` (the resolved scopes
        opted in and none is restricted: `policy.Resolution.record_text`); otherwise it is
        dropped, not stored and not an error."""
        request_text = attempt.request_text if text_allowed else None
        response_text = attempt.response_text if text_allowed else None
        row = (
            time.time(),
            attempt.call_id,
            attempt.attempt,
            attempt.component,
            SCOPE_SEPARATOR.join(attempt.scopes),
            attempt.call_site,
            attempt.run_id,
            attempt.provider,
            attempt.model,
            int(attempt.hosted),
            attempt.kind,
            attempt.reason.value if attempt.reason else None,
            attempt.input_tokens,
            attempt.output_tokens,
            attempt.duration_ms,
            attempt.outcome,
            attempt.failure_class.value if attempt.failure_class else None,
            attempt.loaded_state,
            request_text,
            response_text,
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO attempts (ts, call_id, attempt, component, scopes, call_site,"
                    " run_id, provider, model, hosted, kind, reason, input_tokens, output_tokens,"
                    " duration_ms, outcome, failure_class, loaded_state, request_text,"
                    " response_text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
        except sqlite3.IntegrityError as exc:
            raise RecordError(
                f"attempt {attempt.attempt} of call {attempt.call_id} already recorded"
            ) from exc
        except sqlite3.Error as exc:
            raise RecordError(f"cannot write the record store: {type(exc).__name__}") from exc

    def attempts(self, call_id: str | None = None) -> list[Mapping[str, Any]]:
        """Rows (oldest first), optionally of one call; `scopes` as a tuple, `hosted` as bool."""
        sql = "SELECT * FROM attempts"
        args: tuple[Any, ...] = ()
        if call_id is not None:
            sql += " WHERE call_id = ?"
            args = (call_id,)
        sql += " ORDER BY ts, call_id, attempt"
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["scopes"] = tuple(d["scopes"].split(SCOPE_SEPARATOR)) if d["scopes"] else ()
            d["hosted"] = bool(d["hosted"])
            out.append(d)
        return out

    def columns(self) -> Sequence[str]:
        with self._connect() as conn:
            return [r["name"] for r in conn.execute("PRAGMA table_info(attempts)")]

    def tokens_used(self, run_id: str) -> int:
        """Input plus output tokens over every attempt of `run_id` (the library-side bound,
        §3.9: a cooperative early check, not the hosted guarantee)."""
        with self._connect() as conn:
            (total,) = conn.execute(
                "SELECT COALESCE(SUM(COALESCE(input_tokens,0) + COALESCE(output_tokens,0)),0)"
                " FROM attempts WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(total)


class _Closing:
    """sqlite3's own context manager commits but never closes; this one closes."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def __enter__(self) -> _Closing:
        return self

    def __exit__(self, *exc: object) -> None:
        self._conn.close()
