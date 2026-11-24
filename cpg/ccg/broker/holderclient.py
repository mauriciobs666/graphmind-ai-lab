"""The broker's key-holder client: how an R7 allow is forwarded (``docs/plans/connected-code-graph-egress-spec5.md``
section 3, items 3-5; wire: ``docs/plans/llm-gateway.md`` section 3.3/3.4).

This module imports nothing from ``llmgateway`` (the broker never does), so it re-implements the holder wire of
``llmgateway.framing`` byte for byte; ``llm-gateway/tests/holder_wire_vectors.json`` pins both copies to shared
literals. One request and one reply per connection. `HolderClient.__call__` is the broker's ``hosted_exchanger``: it
never raises for a holder outcome, it returns the `structured.Exchange` the closed table of spec5 section 3 item 4 names.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import socket
import stat
import struct
import time
from typing import Any, Callable

from ccg.broker import framing
from ccg.broker.structured import Exchange

CALLER = "broker"
REFUSAL_CONTENT_TYPE = "application/vnd.llmgateway.refusal+json"
CLOSE_REASONS = frozenset({"completed", "upstream-error", "timeout", "response-too-large"})
PRE_AUTH_REFUSALS = frozenset({"unknown_caller", "bad_credential"})  # the two refusals the holder cannot sign
_U32 = struct.Struct(">I")


class HolderUnusable(ConnectionError):
    """The socket's directory is not holder-owned (or is writable by others, or is gone)."""


class _Reject(Exception):
    """A reply that fails the closed table: the exchange is an `upstream-error`."""


class _TooLarge(Exception):
    pass


def _f(value: "str | bytes") -> bytes:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return _U32.pack(len(raw)) + raw


def request_mac(secret: bytes, *, caller: str, ts: int, nonce: str, method: str, endpoint: str, path: str,
                run_id: str, body: bytes) -> str:
    msg = b"".join((_f(caller), _f(str(ts)), _f(nonce), _f(method), _f(endpoint), _f(path), _f(run_id),
                    _f(hashlib.sha256(body).digest())))
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def response_mac(secret: bytes, *, request_nonce: str, close_reason: str, status: int, content_type: str,
                 body: bytes) -> str:
    msg = b"".join((_f("response"), _f(request_nonce), _f(close_reason), _f(str(status)), _f(content_type),
                    _f(hashlib.sha256(body).digest())))
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def _strict_loads(data: bytes) -> Any:
    def pairs(items):
        out: dict[str, Any] = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"duplicate key {k!r}")
            out[k] = v
        return out

    def no_constant(name):
        raise ValueError(f"non-finite number {name}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=no_constant)
    except RecursionError as exc:
        raise ValueError("nesting too deep") from exc


def encode_frame(header: dict[str, Any], body: bytes) -> bytes:
    head = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload = _U32.pack(len(head)) + head + body
    return _U32.pack(len(payload)) + payload


def build_request(secret: bytes, *, method: str, endpoint: str, path: str, run_id: str | None, body: bytes,
                  ts: int, nonce: str, caller: str = CALLER) -> bytes:
    rid = run_id or ""
    mac = request_mac(secret, caller=caller, ts=ts, nonce=nonce, method=method, endpoint=endpoint, path=path,
                      run_id=rid, body=body)
    header = {"caller": caller, "ts": ts, "nonce": nonce, "mac": mac, "method": method, "endpoint": endpoint,
              "path": path, "run_id": rid}
    return encode_frame(header, body)


def parse_reply(payload: bytes, secret: bytes, request_nonce: str) -> Exchange:
    """The `Exchange` for one reply payload (the frame without its length prefix); byte counts are the caller's.
    Raises `_Reject` for every row of the closed table that is an `upstream-error`."""
    if len(payload) < 4:
        raise _Reject("short frame")
    (hlen,) = _U32.unpack_from(payload)
    if hlen > len(payload) - 4:
        raise _Reject("header length exceeds frame")
    try:
        header = _strict_loads(payload[4:4 + hlen])
    except ValueError as exc:
        raise _Reject(f"header: {exc}") from None
    body = payload[4 + hlen:]
    if not isinstance(header, dict):
        raise _Reject("header is not an object")
    reason, status, ctype, mac = (header.get(k) for k in ("closeReason", "status", "contentType", "mac"))
    if (reason not in CLOSE_REASONS or isinstance(status, bool) or not isinstance(status, int)
            or not isinstance(ctype, str) or not isinstance(mac, str)):
        raise _Reject("malformed reply header")
    if mac:
        want = response_mac(secret, request_nonce=request_nonce, close_reason=reason, status=status,
                            content_type=ctype, body=body)
        if not hmac.compare_digest(mac, want):
            raise _Reject("reply MAC does not verify")
    elif not _pre_auth_refusal(ctype, body):
        raise _Reject("unsigned reply")
    if reason != "completed":
        if not mac:
            raise _Reject("unsigned reply")
        return Exchange(reason, None, None, None, 0, 0)
    if not 100 <= status <= 599:
        raise _Reject("status outside 100-599")
    return Exchange("completed", status, ctype or None, body, 0, 0)


def _pre_auth_refusal(content_type: str, body: bytes) -> bool:
    if content_type != REFUSAL_CONTENT_TYPE:
        return False
    try:
        code = _strict_loads(body).get("holderRefusal")
    except (ValueError, AttributeError):
        return False
    return code in PRE_AUTH_REFUSALS


def check_socket_dir(socket_path: str, holder_uid: int) -> None:
    """Connect only when the socket's directory is owned by the holder's uid and is not group/other-writable."""
    directory = os.path.dirname(os.path.abspath(socket_path))
    try:
        st = os.lstat(directory)
    except OSError as exc:
        raise HolderUnusable(f"socket directory unusable: {exc.strerror}") from exc
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != holder_uid or st.st_mode & 0o022:
        raise HolderUnusable("socket directory is not owned by the holder")


class HolderClient:
    """The broker's ``hosted_exchanger``: ``(endpoint, method, path, body, gateway_run_id) -> Exchange``."""

    def __init__(self, socket_path: str, holder_uid: int, secret: bytes, *,
                 now: Callable[[], float] = time.time, monotonic: Callable[[], float] = time.monotonic,
                 timeout: float = framing.UPSTREAM_TIMEOUT_S,
                 max_reply_bytes: int = framing.MAX_RESPONSE_BYTES + framing.FRAME_SLACK):
        self.socket_path, self.holder_uid, self._secret = str(socket_path), holder_uid, secret
        self.now, self.monotonic, self.timeout, self.max_reply_bytes = now, monotonic, timeout, max_reply_bytes

    def __repr__(self) -> str:  # the credential never reaches a log line or a traceback
        return f"HolderClient({self.socket_path!r})"

    def __call__(self, endpoint: str, method: str, path: str, body: bytes, gateway_run_id: "str | None") -> Exchange:
        counts = {"out": 0, "in": 0}
        sock = None
        deadline = self.monotonic() + self.timeout

        def remaining() -> float:
            left = deadline - self.monotonic()
            if left <= 0:
                raise TimeoutError("deadline")
            return left

        def take(n: int) -> bytes:
            chunks: list[bytes] = []
            while n:
                sock.settimeout(remaining())
                chunk = sock.recv(min(n, 65536))
                if not chunk:
                    raise _Reject("connection closed mid-frame")
                counts["in"] += len(chunk)
                chunks.append(chunk)
                n -= len(chunk)
            return b"".join(chunks)

        try:
            check_socket_dir(self.socket_path, self.holder_uid)
            nonce = secrets.token_hex(16)
            frame = build_request(self._secret, method=method, endpoint=endpoint, path=path,
                                  run_id=gateway_run_id, body=body, ts=int(self.now()), nonce=nonce)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(remaining())
            sock.connect(self.socket_path)
            sock.settimeout(remaining())
            sock.sendall(frame)
            counts["out"] = len(frame)
            (size,) = _U32.unpack(take(4))
            if size > self.max_reply_bytes:
                raise _TooLarge
            ex = parse_reply(take(size), self._secret, nonce)
            return ex._replace(bytes_out=counts["out"], bytes_in=counts["in"])
        except _TooLarge:
            return Exchange("response-too-large", None, None, None, counts["out"], counts["in"])
        except (TimeoutError, socket.timeout):
            return Exchange("timeout", None, None, None, counts["out"], counts["in"])
        except (OSError, _Reject, ValueError):
            return Exchange("upstream-error", None, None, None, counts["out"], counts["in"])
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
