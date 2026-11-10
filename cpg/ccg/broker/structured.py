"""The structured-call transport (C-1001 P1-29): the closed ``repo-names-and-identity`` body, and the one upstream
exchange the broker makes after an allow. Spec: ``docs/plans/connected-code-graph-egress-spec3.md`` section 2.3.

The broker builds the whole upstream request: the method from the target, the path the matched entry stores, and
exactly the headers of section 2.3. The exchange is hand-written HTTP/1.1 over one connected socket so that every
bound is the broker's own: a wall-clock deadline for the whole exchange (not a per-read timeout), a byte count of
what was read from the socket with the read aborted at the bound plus one, no redirect following, no response
encoding other than ``identity``. It never resolves a name: the caller passes the address `ccg.broker.addr` validated.
"""
from __future__ import annotations

import json
import re
import socket
import ssl
import time
from typing import Any, Callable, NamedTuple

from ccg.broker import framing

MAX_HEADER_BYTES = 64 * 1024
MAX_FIELD_CHARS = 64  # the audit schema's `payloadFields` item bound
CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
USER_AGENT = "ccg-broker/1"
_SUBJECT_RE = re.compile(r"[\x21-\x7e]{1,256}")


def serialize_body(body: Any) -> bytes:
    """The UTF-8 JSON the broker sends (and whose digest and size the decision record carries)."""
    return json.dumps(body, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def valid_identity_body(body: Any, system_repo_ids: set[str], idp_endpoints: set[str]) -> bool:
    """Section 2.3: exactly ``{repoNames, identity: {subject, issuer?}}``; ``repoNames`` a non-empty array of distinct
    repository ids of the run's system; ``subject`` 1-256 printable ASCII; ``issuer`` the endpoint of an idp entry."""
    if not isinstance(body, dict) or set(body) != {"repoNames", "identity"}:
        return False
    names, ident = body["repoNames"], body["identity"]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        return False
    if len(set(names)) != len(names) or not set(names) <= system_repo_ids:
        return False
    if not isinstance(ident, dict) or "subject" not in ident or not set(ident) <= {"subject", "issuer"}:
        return False
    subject = ident["subject"]
    if not isinstance(subject, str) or not _SUBJECT_RE.fullmatch(subject):
        return False
    return "issuer" not in ident or (isinstance(ident["issuer"], str) and ident["issuer"] in idp_endpoints)


def payload_fields(body: Any, payload_class: str) -> list[str]:
    """The top-level keys sent; ``identity.*`` keys replace ``identity`` for ``repo-names-and-identity``."""
    if not isinstance(body, dict):
        return []
    out = []
    for k in sorted(body):
        if payload_class == "repo-names-and-identity" and k == "identity" and isinstance(body[k], dict):
            out.extend(f"identity.{s}" for s in sorted(body[k]))
        else:
            out.append(k)
    return out


def make_tls_context(create: Callable[..., ssl.SSLContext] = ssl.create_default_context) -> ssl.SSLContext:
    """A context that loads the fixed CA bundle only: naming ``cafile`` makes the library skip the default verify
    paths, so ``SSL_CERT_FILE`` and ``SSL_CERT_DIR`` have no effect (section 2.0)."""
    return create(cafile=CA_BUNDLE)


class Exchange(NamedTuple):
    close_reason: str  # completed | upstream-error | timeout | response-too-large
    status: int | None
    content_type: str | None
    body: bytes | None
    bytes_out: int
    bytes_in: int


class _TooLarge(Exception):
    pass


class _Bad(Exception):
    pass


def _host_header(host: str, port: int, scheme: str) -> str:
    h = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return h if port == {"http": 80, "https": 443}[scheme] else f"{h}:{port}"


def build_request(method: str, path: str, host_header: str, body: bytes | None) -> bytes:
    """Section 2.3's request line and headers, and no others."""
    lines = [f"{method} {path} HTTP/1.1", f"Host: {host_header}"]
    if body is not None:
        lines += ["Content-Type: application/json", f"Content-Length: {len(body)}"]
    lines += ["Accept: application/json", "Accept-Encoding: identity", f"User-Agent: {USER_AGENT}"]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + (body or b"")


def exchange(scheme: str, host: str, port: int, addr: str, method: str, path: str, body: bytes | None, *,
             tls_context: ssl.SSLContext | None = None, monotonic: Callable[[], float] = time.monotonic,
             connect: Callable[..., socket.socket] = socket.create_connection) -> Exchange:
    """One request to ``addr:port`` (already validated), wall-clock bounded by ``framing.UPSTREAM_TIMEOUT_S``."""
    deadline = monotonic() + framing.UPSTREAM_TIMEOUT_S
    counts = {"out": 0, "in": 0}
    sock = None

    def remaining() -> float:
        left = deadline - monotonic()
        if left <= 0:
            raise TimeoutError
        return left

    def recv_more(want: int = 65536) -> bytes:
        sock.settimeout(remaining())
        room = framing.MAX_RESPONSE_BYTES + 1 - counts["in"]
        chunk = sock.recv(max(1, min(want, room)))
        counts["in"] += len(chunk)
        if counts["in"] > framing.MAX_RESPONSE_BYTES:
            raise _TooLarge
        return chunk

    try:
        sock = connect((addr, port), timeout=remaining())
        if scheme == "https":
            sock.settimeout(remaining())
            sock = (tls_context or make_tls_context()).wrap_socket(sock, server_hostname=host)
        data = build_request(method, path, _host_header(host, port, scheme), body)
        sock.settimeout(remaining())
        sock.sendall(data)
        counts["out"] = len(data)
        status, ctype, payload = _read_response(recv_more)
        return Exchange("completed", status, ctype, payload, counts["out"], counts["in"])
    except _TooLarge:
        return Exchange("response-too-large", None, None, None, counts["out"], counts["in"])
    except (TimeoutError, socket.timeout):
        return Exchange("timeout", None, None, None, counts["out"], counts["in"])
    except (OSError, ssl.SSLError, _Bad, ValueError):
        return Exchange("upstream-error", None, None, None, counts["out"], counts["in"])
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def _read_response(recv_more: Callable[..., bytes]) -> tuple[int, str | None, bytes]:
    buf = b""
    while b"\r\n\r\n" not in buf:
        if len(buf) > MAX_HEADER_BYTES:
            raise _Bad("response headers too long")
        chunk = recv_more()
        if not chunk:
            raise _Bad("connection closed before the response headers ended")
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    m = re.fullmatch(r"HTTP/1\.[01] ([1-5][0-9]{2})(?: .*)?", lines[0])
    if not m:
        raise _Bad("bad status line")
    status = int(m.group(1))
    headers: dict[str, str] = {}
    for ln in lines[1:]:
        k, sep, v = ln.partition(":")
        if not sep:
            raise _Bad("bad header line")
        headers[k.strip().lower()] = v.strip()
    if status < 200:
        raise _Bad("informational response")
    if headers.get("content-encoding", "identity").lower() not in ("", "identity"):
        raise _Bad("response has a content encoding")
    ctype = headers.get("content-type")
    if status in (204, 304):
        return status, ctype, b""
    if "chunked" in headers.get("transfer-encoding", "").lower():
        return status, ctype, _read_chunked(rest, recv_more)
    if "content-length" in headers:
        cl = headers["content-length"]
        if not cl.isdigit():
            raise _Bad("bad content-length")
        n = int(cl)
        while len(rest) < n:
            chunk = recv_more()
            if not chunk:
                raise _Bad("body truncated")
            rest += chunk
        return status, ctype, rest[:n]
    while True:
        chunk = recv_more()
        if not chunk:
            return status, ctype, rest
        rest += chunk


def _read_chunked(buf: bytes, recv_more: Callable[..., bytes]) -> bytes:
    out = bytearray()

    def need(n_or_marker) -> None:
        nonlocal buf
        while True:
            if isinstance(n_or_marker, int):
                if len(buf) >= n_or_marker:
                    return
            elif n_or_marker in buf:
                return
            chunk = recv_more()
            if not chunk:
                raise _Bad("chunked body truncated")
            buf += chunk

    while True:
        need(b"\r\n")
        line, _, buf = buf.partition(b"\r\n")
        size_text = line.split(b";", 1)[0].strip()
        if not re.fullmatch(rb"[0-9a-fA-F]{1,8}", size_text):
            raise _Bad("bad chunk size")
        size = int(size_text, 16)
        if size == 0:
            return bytes(out)  # trailers, if any, are not read
        need(size + 2)
        out += buf[:size]
        if buf[size:size + 2] != b"\r\n":
            raise _Bad("bad chunk terminator")
        buf = buf[size + 2:]
