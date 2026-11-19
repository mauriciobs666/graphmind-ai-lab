"""The interpretation cache of the plain-language front (C-1001 P1-26 behaviour (4); ``docs/plans/connected-code-graph-answer-spec2.md`` §2).

A key is ``ccg.answer.cache_key(normalized_question, snapshot_id, model_fingerprint, prompt_hash, grammar_hash, sampling_hash, resolver_version)``; :func:`key_of` is the one
place it is built. An entry **lives exactly as long as its snapshot**: the cache holds entries of one snapshot only, and the first read or write under another snapshot
drops them all, and at most :data:`MAX_ENTRIES` are held (the oldest goes first). The value is the schema-valid interpreter reply (before the fabrication guard, which runs on every use), never an answer. The cache is in this process's
memory; nothing is written to disk. Whether and where to persist it across processes is a separate decision.
"""
from __future__ import annotations

import copy
import threading
from typing import Any

from ccg import answer


def key_of(normalized_question: str, snapshot_id: str, model_fingerprint: str, prompt_hash: str, grammar_hash: str, sampling_hash: str, resolver_version: str) -> str:
    return answer.cache_key(normalized_question, snapshot_id, model_fingerprint, prompt_hash, grammar_hash, sampling_hash, resolver_version)

#: the bound on one snapshot's entries (a long-lived process, the MCP server, memoizes); past it the oldest entry goes
MAX_ENTRIES = 1024


class InterpretCache:
    """Safe to share between threads (one lock): the MCP server's one client serves every call of its process."""

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("a cache holds at least one entry")
        self._max = max_entries
        self._lock = threading.Lock()
        self._snapshot: str | None = None
        self._entries: dict[str, dict[str, Any]] = {}

    def _scope(self, snapshot_id: str) -> None:
        if self._snapshot != snapshot_id:
            self._snapshot, self._entries = snapshot_id, {}

    def get(self, key: str, snapshot_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._scope(snapshot_id)
            hit = self._entries.get(key)
            return copy.deepcopy(hit) if hit is not None else None

    def put(self, key: str, snapshot_id: str, reply: dict[str, Any]) -> None:
        with self._lock:
            self._scope(snapshot_id)
            self._entries.pop(key, None)
            self._entries[key] = copy.deepcopy(reply)
            while len(self._entries) > self._max:
                del self._entries[next(iter(self._entries))]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
