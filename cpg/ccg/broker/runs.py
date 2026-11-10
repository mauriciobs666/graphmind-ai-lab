"""Runs and their tokens (C-1001 P1-29). Spec: ``docs/plans/connected-code-graph-egress-spec3.md`` section 2.1.

A run token is 32 bytes from the OS CSPRNG, presented as 64 lowercase hex characters. The table keeps only the
token's SHA-256 and looks a presented token up by that hash, so the token is held nowhere but the caller's memory and
environment. A run expires at `close` or 24 hours after it opened (an injectable clock); a broker restart drops every
run (the table is memory only).
"""
from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

RUN_TTL_S = 24 * 3600
_TOKEN_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Run:
    run_id: str
    run_kind: str
    system_id: str
    codebases: tuple[str, ...]
    effective: dict[str, str]  # repoId -> effective mode, for the run's codebases
    system_local_only: bool
    expires_at: float = field(compare=False, default=0.0)


class RunTable:
    def __init__(self, clock: Callable[[], float] = time.monotonic, ttl_s: float = RUN_TTL_S):
        self._clock = clock
        self._ttl = ttl_s
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create(self, run_id: str, run_kind: str, system_id: str, codebases: list[str], effective: dict[str, str],
               system_local_only: bool) -> tuple[str, Run]:
        """Mint a token for a run whose ``policy`` record is already durable; returns ``(token, run)``."""
        token = secrets.token_bytes(32).hex()
        run = Run(run_id, run_kind, system_id, tuple(codebases), dict(effective), system_local_only,
                  self._clock() + self._ttl)
        with self._lock:
            self._runs[self._hash(token)] = run
        return token, run

    def lookup(self, token: object) -> Run | None:
        """The live run for a token, or None for no token, an unknown one or an expired one."""
        if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
            return None
        key = self._hash(token)
        with self._lock:
            run = self._runs.get(key)
            if run is not None and self._clock() >= run.expires_at:
                del self._runs[key]
                return None
            return run

    def close(self, token: object) -> None:
        """Idempotent; writes no record."""
        if isinstance(token, str) and _TOKEN_RE.fullmatch(token):
            with self._lock:
                self._runs.pop(self._hash(token), None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)
