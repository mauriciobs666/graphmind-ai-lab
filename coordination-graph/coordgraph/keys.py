"""Key and indexed-value hygiene (graph note section 4.1; plan section 3.8).

A UNIQUE-indexed value over 4,096 bytes, or with leading/trailing whitespace at any
length, segfaults the shared FalkorDB; an indexed equality lookup silently misses a
whitespace-padded value. Both functions apply at write time and to every lookup
parameter. Python's `strip()`/`isspace()` cover a strict superset of the crashing set
(C `isspace()`), so no NBSP special case is needed.
"""
from __future__ import annotations

import hashlib

MAX_KEY_BYTES = 512


class KeyHygieneError(ValueError):
    """A value that must never reach an index (the failure mode is an outage)."""


def _assert_edges(value: str) -> None:
    if value and (value[0].isspace() or value[-1].isspace()):
        raise KeyHygieneError(f"whitespace edge survived normalisation: {value!r}")


def norm_value(value: str) -> str:
    """Strip an indexed non-unique value (`slug`, `scope`, `statusToken`, `itemId`, ...)."""
    out = value.strip()
    _assert_edges(out)
    return out


def safe_key(*parts: str) -> str:
    """A UNIQUE value: each part and the whole stripped, then > 512 UTF-8 bytes -> `h:<sha256>`.

    A single part is the value itself; several parts join with `#` (`<path>#<unit>`).
    """
    if not parts:
        raise KeyHygieneError("safe_key needs at least one part")
    value = "#".join(p.strip() for p in parts).strip()
    if len(value.encode("utf-8")) > MAX_KEY_BYTES:
        value = "h:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
    _assert_edges(value)
    if len(value.encode("utf-8")) > MAX_KEY_BYTES:
        raise KeyHygieneError("key still over the byte limit after hashing")
    return value
