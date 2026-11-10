"""Length-prefixed strict-JSON frames on the broker's socket (egress-spec3 section 2.2).

A frame is a 4-byte big-endian length and that many bytes of UTF-8 JSON. Parsing is strict: a duplicate object key,
``NaN``/``Infinity``, nesting deeper than 64 levels or an integer outside +-2**63 is malformed. Serialization uses
``allow_nan=False``.
"""
from __future__ import annotations

import json
import struct
from typing import Any, Callable

MAX_DEPTH = 64
INT_BOUND = 2**63
# The frame bounds (section 2.2/2.3). They live here, not in `structured`, so that `BrokerClient` needs nothing from
# the module that opens the upstream connection; `structured` reads them from here at call time.
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
UPSTREAM_TIMEOUT_S = 300
FRAME_SLACK = 64 * 1024
CLIP = 80


def clip(value: object, n: int = CLIP) -> str:
    """A short repr for a log line: caller text is never logged at its own size."""
    r = repr(value)
    return r if len(r) <= n else r[: n - 3] + "..."


def max_request_frame() -> int:
    return MAX_REQUEST_BYTES + FRAME_SLACK


def max_reply_frame() -> int:
    """The reply carries the response body base64-encoded, so the cap is the response bound after encoding plus the
    spec's 64 KiB."""
    return (MAX_RESPONSE_BYTES + 2) // 3 * 4 + FRAME_SLACK


class FrameError(ValueError):
    """A malformed or oversized frame: the receiver closes the connection and answers nothing."""


def _no_constant(name: str):
    raise FrameError(f"non-finite number {name}")


def _int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise FrameError("integer literal too long") from None
    if not -INT_BOUND <= v <= INT_BOUND:
        raise FrameError("integer outside +-2**63")
    return v


def _float(text: str) -> float:
    v = float(text)
    if v in (float("inf"), float("-inf")):
        raise FrameError("float overflows to infinity")
    return v


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise FrameError(f"duplicate object key {clip(k)}")
        out[k] = v
    return out


def _encodable(s: str) -> bool:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate: JSON admits the escape, UTF-8 does not
        return False
    return True


def _walk_ok(value: Any) -> bool:
    """Nesting within MAX_DEPTH, and every string (object keys included) encodable as UTF-8."""
    stack = [(value, 1)]
    while stack:
        v, d = stack.pop()
        if isinstance(v, str):
            if not _encodable(v):
                return False
        elif isinstance(v, (dict, list)):
            if d > MAX_DEPTH:
                return False
            if isinstance(v, dict):
                if not all(_encodable(k) for k in v):
                    return False
                children = v.values()
            else:
                children = v
            for child in children:
                stack.append((child, d + 1))
    return True


def parse_strict(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_no_constant, parse_int=_int,
                           parse_float=_float)
    except FrameError:
        raise
    except (ValueError, RecursionError) as exc:
        raise FrameError(f"malformed JSON: {exc}") from None
    if not _walk_ok(value):
        raise FrameError(f"nesting deeper than {MAX_DEPTH}, or a string that is not UTF-8 encodable")
    return value


def encode(obj: Any, max_len: int | None = None) -> bytes:
    raw = json.dumps(obj, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if max_len is not None and len(raw) > max_len:
        raise FrameError(f"frame of {len(raw)} bytes exceeds {max_len}")
    return struct.pack(">I", len(raw)) + raw


def read_frame(recv: Callable[[int], bytes], max_len: int) -> Any:
    """Read one frame through ``recv(n)`` (at most n bytes). Returns the parsed JSON; raises `EOFError` when the peer
    closed before the first byte, and `FrameError` for an oversized, truncated or malformed frame. The length is read
    first and checked before any payload byte is."""
    head = _read_exact(recv, 4, allow_eof=True)
    (n,) = struct.unpack(">I", head)
    if n > max_len:
        raise FrameError(f"frame length {n} exceeds {max_len}")
    return parse_strict(_read_exact(recv, n, allow_eof=False))


def _read_exact(recv: Callable[[int], bytes], n: int, allow_eof: bool) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = recv(min(65536, n - len(buf)))
        if not chunk:
            if allow_eof and not buf:
                raise EOFError("peer closed")
            raise FrameError("truncated frame")
        buf += chunk
    return bytes(buf)
