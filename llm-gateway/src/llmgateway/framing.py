"""Holder wire framing and request/response authentication (docs/plans/llm-gateway.md §3.3, §3.4).

One request and one response per connection. A frame is `u32 total | u32 header_len | header
JSON | body bytes`; `total` counts everything after itself. The request header carries
`caller, ts, nonce, mac, method, endpoint, path, run_id`; the body bytes travel raw so their
digest is the digest of what the holder validates and forwards.

    mac = HMAC-SHA256(secret, F(caller) F(ts) F(nonce) F(method) F(endpoint) F(path)
                              F(run_id) F(sha256(body)))        F(x) = u32-be length || UTF-8

so no byte of a caller-chosen field can shift a field boundary. The response header carries
`closeReason, status, contentType, mac`; its MAC covers the request nonce, so a fake holder
that never saw the secret cannot forge one. Stdlib only; the module is import-safe for the
broker's holder client and the service transport (nothing here reads configuration).
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

REFUSAL_CONTENT_TYPE = "application/vnd.llmgateway.refusal+json"
REFUSAL_STATUS = 499
CLOSE_REASONS = frozenset({"completed", "upstream-error", "timeout", "response-too-large"})
MAX_NONCE_CHARS = 128
_U32 = struct.Struct(">I")


class FrameError(Exception):
    """A frame that is malformed, over the cap, or too slow; the holder drops the connection."""


class ReadTimeout(FrameError):
    """`read_frame` missed its total deadline."""


class HolderUnreachable(ConnectionError):
    """The holder socket cannot be used (absent, refused, or in a directory not owned by it)."""


class HolderTimeout(HolderUnreachable):
    """The holder accepted the connection but missed the read deadline (class `timeout`)."""


class ReplyRejected(Exception):
    """A reply failed authentication: a fake or tampered holder."""


def _f(value: str | bytes) -> bytes:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return _U32.pack(len(raw)) + raw


def request_mac(
    secret: bytes,
    *,
    caller: str,
    ts: int,
    nonce: str,
    method: str,
    endpoint: str,
    path: str,
    run_id: str,
    body: bytes,
) -> str:
    msg = b"".join(
        (
            _f(caller),
            _f(str(ts)),
            _f(nonce),
            _f(method),
            _f(endpoint),
            _f(path),
            _f(run_id),
            _f(hashlib.sha256(body).digest()),
        )
    )
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def response_mac(
    secret: bytes,
    *,
    request_nonce: str,
    close_reason: str,
    status: int,
    content_type: str,
    body: bytes,
) -> str:
    msg = b"".join(
        (
            _f("response"),
            _f(request_nonce),
            _f(close_reason),
            _f(str(status)),
            _f(content_type),
            _f(hashlib.sha256(body).digest()),
        )
    )
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def strict_loads(data: bytes | str) -> Any:
    """`json.loads` that refuses duplicate keys at any depth, NaN/Infinity and invalid UTF-8
    (a duplicate key is parsed differently by different readers: the holder validates what the
    provider will read)."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate key {key!r}")
            out[key] = value
        return out

    def no_constant(name: str) -> Any:
        raise ValueError(f"non-finite number {name}")

    text = data.decode("utf-8") if isinstance(data, bytes) else data
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=no_constant)
    except RecursionError as exc:
        raise ValueError("nesting too deep") from exc


# --- frames ---


def encode_frame(header: dict[str, Any], body: bytes) -> bytes:
    head = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload = _U32.pack(len(head)) + head + body
    return _U32.pack(len(payload)) + payload


def decode_payload(payload: bytes) -> tuple[dict[str, Any], bytes]:
    if len(payload) < 4:
        raise FrameError("short frame")
    (hlen,) = _U32.unpack_from(payload)
    if hlen > len(payload) - 4:
        raise FrameError("header length exceeds frame")
    try:
        header = strict_loads(payload[4 : 4 + hlen])
    except ValueError as exc:
        raise FrameError(f"header: {exc}") from exc
    if not isinstance(header, dict):
        raise FrameError("header is not an object")
    return header, payload[4 + hlen :]


def read_frame(
    sock: socket.socket,
    *,
    max_bytes: int,
    timeout: float,
    monotonic: Callable[[], float] = time.monotonic,
) -> bytes:
    """Read one frame's payload within a total deadline, refusing a declared size over the cap
    before reading any of it. Raises `FrameError` on a cap, a deadline or an early close."""
    deadline = monotonic() + timeout

    def take(n: int) -> bytes:
        chunks: list[bytes] = []
        while n:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ReadTimeout("read timeout")
            sock.settimeout(remaining)
            try:
                chunk = sock.recv(min(n, 65536))
            except TimeoutError as exc:
                raise ReadTimeout("read timeout") from exc
            except OSError as exc:
                raise FrameError(f"read failed: {type(exc).__name__}") from exc
            if not chunk:
                raise FrameError("connection closed mid-frame")
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    (size,) = _U32.unpack(take(4))
    if size > max_bytes:
        raise FrameError("frame over the cap")
    return take(size)


@dataclass(frozen=True, slots=True)
class RequestHeader:
    caller: str
    ts: int
    nonce: str
    mac: str
    method: str
    endpoint: str
    path: str
    run_id: str


def parse_request(payload: bytes) -> tuple[RequestHeader, bytes]:
    """Shape-check a request frame (types only; authenticity is the holder's next step)."""
    header, body = decode_payload(payload)
    want = {"caller", "ts", "nonce", "mac", "method", "endpoint", "path", "run_id"}
    if set(header) != want:
        raise FrameError("request header fields")
    ts = header["ts"]
    strings = {k: header[k] for k in want - {"ts"}}
    if header["run_id"] is None:
        strings["run_id"] = ""
    if (
        isinstance(ts, bool)
        or not isinstance(ts, int)
        or not all(isinstance(v, str) for v in strings.values())
        or not 0 < len(strings["nonce"]) <= MAX_NONCE_CHARS
    ):
        raise FrameError("request header types")
    return RequestHeader(ts=ts, **strings), body


def build_request(
    secret: bytes,
    *,
    caller: str,
    method: str,
    endpoint: str,
    path: str,
    run_id: str | None,
    body: bytes,
    ts: int | None = None,
    nonce: str | None = None,
) -> tuple[bytes, str]:
    """A signed request frame and its nonce. A fresh nonce every call (a retry never reuses)."""
    ts = int(time.time()) if ts is None else ts
    nonce = secrets.token_hex(16) if nonce is None else nonce
    rid = run_id or ""
    mac = request_mac(
        secret,
        caller=caller,
        ts=ts,
        nonce=nonce,
        method=method,
        endpoint=endpoint,
        path=path,
        run_id=rid,
        body=body,
    )
    header = {
        "caller": caller,
        "ts": ts,
        "nonce": nonce,
        "mac": mac,
        "method": method,
        "endpoint": endpoint,
        "path": path,
        "run_id": rid,
    }
    return encode_frame(header, body), nonce


def build_response(
    secret: bytes | None,
    *,
    request_nonce: str,
    close_reason: str,
    status: int,
    content_type: str,
    body: bytes,
) -> bytes:
    mac = ""
    if secret is not None:
        mac = response_mac(
            secret,
            request_nonce=request_nonce,
            close_reason=close_reason,
            status=status,
            content_type=content_type,
            body=body,
        )
    header = {
        "closeReason": close_reason,
        "status": status,
        "contentType": content_type,
        "mac": mac,
    }
    return encode_frame(header, body)


def refusal_body(code: str) -> bytes:
    return json.dumps({"holderRefusal": code}).encode("utf-8")


@dataclass(frozen=True, slots=True)
class HolderReply:
    close_reason: str
    status: int
    content_type: str
    body: bytes
    authenticated: bool

    @property
    def refusal_code(self) -> str | None:
        if self.content_type != REFUSAL_CONTENT_TYPE:
            return None
        try:
            code = strict_loads(self.body).get("holderRefusal")
        except (ValueError, AttributeError):
            return None
        return code if isinstance(code, str) else None


def parse_reply(
    payload: bytes, secret: bytes, request_nonce: str, *, require_mac: bool = True
) -> HolderReply:
    """Check a response against the request nonce. A signed reply must verify; an unsigned one
    is accepted (`authenticated=False`) only when `require_mac` is False, which callers allow
    for the pre-authentication refusals the holder cannot sign."""
    header, body = decode_payload(payload)
    reason, status, ctype, mac = (
        header.get(k) for k in ("closeReason", "status", "contentType", "mac")
    )
    if (
        reason not in CLOSE_REASONS
        or isinstance(status, bool)
        or not isinstance(status, int)
        or not isinstance(ctype, str)
        or not isinstance(mac, str)
    ):
        raise ReplyRejected("malformed reply header")
    if mac:
        want = response_mac(
            secret,
            request_nonce=request_nonce,
            close_reason=reason,
            status=status,
            content_type=ctype,
            body=body,
        )
        if not hmac.compare_digest(mac, want):
            raise ReplyRejected("reply MAC does not verify")
        return HolderReply(reason, status, ctype, body, True)
    if require_mac:
        raise ReplyRejected("unsigned reply")
    return HolderReply(reason, status, ctype, body, False)


def check_socket_dir(socket_path: str, holder_uid: int) -> None:
    """A caller connects only when the socket's directory is owned by the holder's uid and is
    not group/other-writable (so no other account can swap in a fake holder)."""
    directory = os.path.dirname(os.path.abspath(socket_path))
    try:
        st = os.lstat(directory)
    except OSError as exc:
        raise HolderUnreachable(f"socket directory unusable: {exc.strerror}") from exc
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != holder_uid or st.st_mode & 0o022:
        raise HolderUnreachable("socket directory is not owned by the holder")


def load_secret(path: str) -> bytes:
    """Read a credential file (hex of 32 random bytes), `O_NOFOLLOW`, regular file only."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("credential is not a regular file")
        text = os.read(fd, 4096).decode("ascii").strip()
    finally:
        os.close(fd)
    secret = bytes.fromhex(text)
    if len(secret) != 32:
        raise ValueError("credential is not 32 bytes")
    return secret


def exchange(
    socket_path: str,
    holder_uid: int,
    secret: bytes,
    *,
    caller: str,
    endpoint: str,
    path: str,
    body: bytes,
    run_id: str | None = None,
    method: str = "POST",
    timeout: float = 300.0,
    max_reply_bytes: int = 16 << 20,
    ts: int | None = None,
) -> HolderReply:
    """One authenticated call. `HolderUnreachable` for a socket that cannot be used (class
    `connection`); `ReplyRejected` for a reply that fails authentication."""
    check_socket_dir(socket_path, holder_uid)
    frame, nonce = build_request(
        secret,
        caller=caller,
        method=method,
        endpoint=endpoint,
        path=path,
        run_id=run_id,
        body=body,
        ts=ts,
    )
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        try:
            sock.connect(socket_path)
        except OSError as exc:
            raise HolderUnreachable(f"holder socket: {exc.strerror}") from exc
        sock.sendall(frame)
        try:
            payload = read_frame(sock, max_bytes=max_reply_bytes, timeout=timeout)
        except ReadTimeout as exc:
            raise HolderTimeout(f"no reply: {exc}") from exc
        except FrameError as exc:
            raise HolderUnreachable(f"no reply: {exc}") from exc
    finally:
        sock.close()
    # Pre-authentication refusals (unknown caller, bad MAC) cannot be signed; every other reply
    # must verify.
    reply = parse_reply(payload, secret, nonce, require_mac=False)
    if not reply.authenticated and reply.refusal_code not in ("unknown_caller", "bad_credential"):
        raise ReplyRejected("unsigned reply")
    return reply
