"""The key holder (docs/plans/llm-gateway.md §3.3, §3.4; FR-8, FR-9, FR-10, FR-18).

The only process that reads a hosted key and makes hosted exchanges. It runs under its own OS
account from the root-owned code root (`/usr/bin/python3 -I /opt/llmgateway/bin/keyholder.py`).
Per connection: one authenticated request, one response (`framing.py`). The holder validates
against root-owned configuration only (env ignored), reserves tokens in its own ledger before it
sends, and answers every refusal as a completed exchange (499, refusal content type).

It imports no provider adapter: the closed per-kind tables below are its own copy.

Run bounds (`_bound_for`): the first `[[run_bounds]]` entry, in file order, whose `match` glob
`fnmatch.fnmatchcase`s the run id supplies the bound (`*`, `?`, `[...]`, case-sensitive); the
library's `policy.check_run_bound` matches the same way (`tests/test_holder_run_bounds.py` runs
one case table against both).

Body validation (`validate_body`) fails closed and never clamps. `tools` and `cache_control` are
refused at any depth, even as a JSON-schema property name (false positives accepted). A limit alias
(`max_tokens`, `max_completion_tokens`, `num_predict`) is refused anywhere except where it names a
property of a JSON schema (a key directly under `properties`, `patternProperties`, `$defs` or
`definitions`), so a structured-output schema may have a field called `max_tokens`.

Any internal fault while serving an authenticated request is answered with a signed
`upstream-error` close reason (class `connection`: retryable), never a refusal code and never an
unsigned reply; `failures.HOLDER_REFUSALS` has no internal-error code.

`calls.log` is rotated by size (`HolderSettings.log_max_bytes`, default `LOG_MAX_BYTES`): one
previous file, `calls.log.1`, is kept.

Subcommands (`main`): `serve` (default), `check-deployment`, `add-caller`, `install-user-file`,
`log`.
"""

from __future__ import annotations

import argparse
import base64
import errno
import fnmatch
import http.client
import ipaddress
import json
import os
import pwd
import re
import secrets
import signal
import socket
import ssl
import stat
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from llmgateway import framing
from llmgateway.config import (
    DEFAULT_PATH,
    PRODUCTION_TRUSTED_UIDS,
    ConfigError,
    ConfigSource,
    GatewayConfig,
    is_hosted_config,
)
from llmgateway.failures import HOLDER_REFUSALS
from llmgateway.ledger import Ceilings, Ledger, LedgerLocked, Refusal, positive_int

DEFAULT_STATE_ROOT = "/var/lib/llmholder"
DEFAULT_SOCKET = "/run/llmholder/holder.sock"
HOLDER_ACCOUNT = "llmholder"
MAX_SKEW_SECONDS = 30
CALLER_NAME = re.compile(r"[a-z0-9_-]{1,32}")
MAX_RUN_ID_CHARS = 256
_LOCAL_KINDS = frozenset({"lmstudio", "ollama", "openai_compat"})  # config.py's local kinds
# The broker's reply cap, `ccg.broker.framing.MAX_RESPONSE_BYTES` (cpg/ccg/broker/framing.py),
# hard-coded: llm-gateway imports no ccg. A larger holder reply would be undeliverable (the broker
# closes the connection with no reply); `tests/test_holder_endpoint.py` pins the number to the
# source.
MAX_BROKER_RESPONSE_BYTES = 16 * 1024 * 1024


class HolderStartError(Exception):
    """The holder must not start (invalid ceilings, unprotected config, writable code, lock)."""


# --- closed per-kind tables (§3.4); the holder imports no adapter ---


@dataclass(frozen=True, slots=True)
class KindSpec:
    paths: frozenset[str]
    limit_paths: tuple[tuple[str, ...], ...]  # exactly one of these must be present
    body_keys: frozenset[str]
    required_usage: tuple[tuple[str, ...], ...]
    optional_usage: tuple[tuple[str, ...], ...] = ()


_OPENAI_KEYS = frozenset(
    "model messages max_tokens max_completion_tokens temperature top_p top_k repeat_penalty "
    "frequency_penalty presence_penalty seed stop response_format stream".split()
)
KINDS: dict[str, KindSpec] = {
    "anthropic": KindSpec(
        paths=frozenset({"/v1/messages"}),
        limit_paths=(("max_tokens",),),
        body_keys=frozenset(
            "model max_tokens messages system temperature top_p top_k stop_sequences "
            "output_config stream".split()
        ),
        required_usage=(("usage", "input_tokens"), ("usage", "output_tokens")),
        optional_usage=(
            ("usage", "cache_creation_input_tokens"),
            ("usage", "cache_read_input_tokens"),
        ),
    ),
    "openai_compat": KindSpec(
        paths=frozenset({"/v1/chat/completions"}),
        limit_paths=(("max_tokens",), ("max_completion_tokens",)),
        body_keys=_OPENAI_KEYS,
        required_usage=(("usage", "prompt_tokens"), ("usage", "completion_tokens")),
    ),
    "ollama": KindSpec(
        paths=frozenset({"/api/chat"}),
        limit_paths=(("options", "num_predict"),),
        body_keys=frozenset("model messages options format think stream".split()),
        required_usage=(("prompt_eval_count",), ("eval_count",)),
    ),
}
KINDS["lmstudio"] = KINDS["openai_compat"]
LIMIT_NAMES = frozenset({"max_tokens", "max_completion_tokens", "num_predict"})
GUARDED_ANYWHERE = frozenset({"cache_control", "tools"})
SCHEMA_NAME_CONTAINERS = frozenset({"properties", "patternProperties", "$defs", "definitions"})
# `calls.log` rotates at this size. The plan's `[holder]` table has no key for it, so it is a
# constant (overridable per holder through `HolderSettings.log_max_bytes`).
LOG_MAX_BYTES = 16 * 1024 * 1024
UNRELEASED_STATUSES = frozenset({408, 504, 524})


# --- request validation ---


@dataclass(frozen=True, slots=True)
class Provider:
    id: str
    kind: str
    base_url: str
    key_ref: str
    pinned_models: frozenset[str]
    max_tokens_ceiling: int
    anthropic_version: str
    allow_private: bool
    extra_body_keys: frozenset[str]

    @property
    def spec(self) -> KindSpec:
        return KINDS[self.kind]


class BodyRefused(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _walk_keys(node: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key
            yield from _walk_keys(value, path + (key,))
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item, path)


def _dig(doc: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def validate_body(provider: Provider, body: bytes) -> tuple[int, str]:
    """Check a request body against the closed table; returns `(limit, model)`.

    Raises `BodyRefused` with `body_invalid`, `model_not_pinned`, `limit_missing` or
    `limit_over_ceiling`. A limit is never clamped (FR-18)."""
    spec = provider.spec
    try:
        doc = framing.strict_loads(body)
    except ValueError:
        raise BodyRefused("body_invalid") from None
    if not isinstance(doc, dict):
        raise BodyRefused("body_invalid")
    if not set(doc) <= spec.body_keys | provider.extra_body_keys:
        raise BodyRefused("body_invalid")
    allowed_limits = {p for p in spec.limit_paths}
    for path, key in _walk_keys(doc):
        if key in GUARDED_ANYWHERE and key not in provider.extra_body_keys:
            raise BodyRefused("body_invalid")
        in_schema_names = bool(path) and path[-1] in SCHEMA_NAME_CONTAINERS
        if key in LIMIT_NAMES and not in_schema_names and path + (key,) not in allowed_limits:
            raise BodyRefused("body_invalid")  # a nested or other kind's alias
    if doc.get("stream", False) is not False:
        raise BodyRefused("body_invalid")
    model = doc.get("model")
    if not isinstance(model, str) or model not in provider.pinned_models:
        raise BodyRefused("model_not_pinned")
    present = [p for p in spec.limit_paths if _has(doc, p)]
    if not present:
        raise BodyRefused("limit_missing")
    if len(present) > 1:
        raise BodyRefused("body_invalid")
    limit = _dig(doc, present[0])
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise BodyRefused("body_invalid")
    if limit > provider.max_tokens_ceiling:
        raise BodyRefused("limit_over_ceiling")
    return limit, model


def _has(doc: Any, path: tuple[str, ...]) -> bool:
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return False
        doc = doc[key]
    return True


def parse_usage(provider: Provider, body: bytes) -> int | None:
    """Sum every reported usage field of a 2xx reply, or None when unparseable (then the
    reservation is kept)."""
    try:
        doc = framing.strict_loads(body)
    except ValueError:
        return None
    total = 0
    for path in provider.spec.required_usage:
        value = _dig(doc, path)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        total += value
    for path in provider.spec.optional_usage:
        if _has(doc, path):
            value = _dig(doc, path)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None
            total += value
    return total


def is_json_error_shape(body: bytes) -> bool:
    try:
        doc = json.loads(body)
    except ValueError:
        return False
    return isinstance(doc, dict) and "error" in doc


def releasable(status: int, body: bytes) -> bool:
    """A complete error response is releasable unless it may have been billed (§3.4)."""
    if status < 400 or status in UNRELEASED_STATUSES:
        return False
    return not (status >= 500 and not is_json_error_shape(body))


# --- secrets hygiene ---


def key_forms(key: str) -> list[bytes]:
    raw = key.encode("utf-8")
    forms = {raw, urllib.parse.quote(key, safe="").encode(), urllib.parse.quote_plus(key).encode()}
    forms.add(raw.hex().encode())
    for shift in range(3):
        padded = b"x" * shift + raw
        drop = (shift * 8) // 6 + 1 if shift else 0
        for enc in (base64.b64encode, base64.urlsafe_b64encode):
            text = enc(padded).decode().rstrip("=")
            if len(padded) % 3:
                text = text[:-1]
            forms.add(text[drop:].encode())
    return [f for f in forms if len(f) >= 8]


def scrub(data: bytes, key: str | None) -> bytes:
    if not key:
        return data
    for form in key_forms(key):
        data = data.replace(form, b"[redacted]")
    return data


def read_key(path: str) -> str | None:
    """The key, read fresh on every call; None when it is missing or unusable."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        raw = os.read(fd, 4097)
    finally:
        os.close(fd)
    try:
        key = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    if not key or len(raw) > 4096 or not key.isprintable() or " " in key:
        return None
    return key


# --- address validation and the upstream exchange ---

_NEVER = (ipaddress.ip_address("169.254.169.254"), ipaddress.ip_address("fd00:ec2::254"))


def address_allowed(ip: str, allow_private: bool) -> bool:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    if addr.is_unspecified or addr.is_multicast or addr.is_link_local or addr in _NEVER:
        return False
    if addr.is_loopback:  # (::1 sits in the reserved ::/8)
        return allow_private
    return not addr.is_reserved and (addr.is_global or allow_private)


def default_resolver(host: str, port: int) -> list[str]:
    return [i[4][0] for i in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


@dataclass(frozen=True, slots=True)
class UpstreamResult:
    wrote: bool  # any request byte may have left (TLS handshake finished)
    close_reason: str
    status: int = 0
    content_type: str = ""
    body: bytes = b""


class Upstream(Protocol):
    def __call__(
        self,
        *,
        host: str,
        port: int,
        ip: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        ca_file: str | None,
        timeout: float,
        max_response: int,
    ) -> UpstreamResult: ...


def https_upstream(
    *,
    host: str,
    port: int,
    ip: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
    ca_file: str | None,
    timeout: float,
    max_response: int,
) -> UpstreamResult:
    """Connect to the validated address with SNI and hostname verification; no redirects."""
    ctx = ssl.create_default_context(cafile=ca_file)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    deadline = time.monotonic() + timeout
    try:
        raw = socket.create_connection((ip, port), timeout=timeout)
        tls = ctx.wrap_socket(raw, server_hostname=host)
    except TimeoutError:
        return UpstreamResult(False, "timeout")
    except (OSError, ssl.SSLError):
        return UpstreamResult(False, "upstream-error")
    conn = http.client.HTTPSConnection(host, port, timeout=timeout)
    conn.sock = tls
    try:
        conn.request("POST", path, body, headers)
        resp = conn.getresponse()
        chunks: list[bytes] = []
        size = 0
        while True:
            if time.monotonic() > deadline:
                return UpstreamResult(True, "timeout")
            chunk = resp.read(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > max_response:
                return UpstreamResult(True, "response-too-large")
            chunks.append(chunk)
        return UpstreamResult(
            True,
            "completed",
            resp.status,
            resp.getheader("Content-Type", "") or "",
            b"".join(chunks),
        )
    except TimeoutError:
        return UpstreamResult(True, "timeout")
    except (OSError, http.client.HTTPException, ssl.SSLError):
        return UpstreamResult(True, "upstream-error")
    finally:
        conn.close()


# --- configuration -> policy ---

_DEFAULT_PORTS = {"http": 80, "https": 443}


def canonical_origin(url: str) -> str:
    """`scheme://host:port`: lowercase scheme and host, explicit port, no path, query or user-info,
    RFC 5952 IPv6. The same rules as the ccg broker's `manifest.canonical_endpoint` (copied, not
    imported; `tests/test_holder_endpoint.py` pins the two together), because the broker sends
    that form while an owner writes `base_url` in any spelling. Raises `ValueError`."""
    if not isinstance(url, str) or not url:
        raise ValueError("empty url")
    if any(ord(c) <= 0x1F or ord(c) == 0x7F or c == " " for c in url):
        raise ValueError("control character or space")
    try:
        parts = urllib.parse.urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError as exc:
        raise ValueError(f"unparseable url: {exc}") from None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise ValueError("scheme must be http or https")
    if not host:
        raise ValueError("no host")
    if not host.isascii() or "%" in host:
        raise ValueError("host must be plain ASCII without a zone id")
    if port is None:
        port = _DEFAULT_PORTS[scheme]
    if not 1 <= port <= 65535:
        raise ValueError("port out of range")
    if ":" in host:
        try:
            host = f"[{ipaddress.IPv6Address(host).compressed}]"
        except ValueError:
            raise ValueError("invalid IPv6 literal") from None
    return f"{scheme}://{host}:{port}"


@dataclass(frozen=True, slots=True)
class Policy:
    providers: dict[str, Provider]  # keyed by canonical_origin(base_url) (the request's `endpoint`)
    ceilings: Ceilings | None
    unmatched: str
    default_bound: int | None
    run_bounds: tuple[tuple[str, int], ...]
    ca_file: str | None
    upstream_timeout: float
    max_response: int


def _provider(pid: str, raw: dict[str, Any]) -> Provider:
    try:
        kind = raw.get("kind")
        if kind not in KINDS:
            raise ValueError("kind")
        models = raw.get("pinned_models")
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(m, str) and m for m in models)
        ):
            raise ValueError("pinned_models")
        extra = raw.get("extra_body_keys", [])
        if not isinstance(extra, list) or not all(isinstance(k, str) for k in extra):
            raise ValueError("extra_body_keys")
        for name in ("base_url", "key_ref"):
            if not isinstance(raw.get(name), str) or not raw[name]:
                raise ValueError(name)
        version = raw.get("anthropic_version", "2023-06-01")
        if not isinstance(version, str) or not re.fullmatch(r"[0-9A-Za-z._-]{1,32}", version):
            raise ValueError("anthropic_version")
        allow = raw.get("allow_private", False)
        if not isinstance(allow, bool):
            raise ValueError("allow_private")
        return Provider(
            id=pid,
            kind=kind,
            base_url=raw["base_url"],
            key_ref=raw["key_ref"],
            pinned_models=frozenset(models),
            max_tokens_ceiling=positive_int(raw.get("max_tokens_ceiling"), "max_tokens_ceiling"),
            anthropic_version=version,
            allow_private=allow,
            extra_body_keys=frozenset(extra),
        )
    except ValueError as exc:
        raise HolderStartError(f"providers.{pid}: invalid {exc}") from None


def build_policy(cfg: GatewayConfig) -> Policy:
    """The holder's view of one protected configuration; raises `HolderStartError`."""
    if not cfg.protected:
        raise HolderStartError(f"configuration is not owner-protected: {cfg.reason}")
    for pid, raw in cfg.providers.items():
        if (
            is_hosted_config(raw)
            and raw.get("kind") in _LOCAL_KINDS
            and cfg.is_local(raw.get("base_url"))
        ):
            # A keyed provider on a local origin would be routed direct, never through the holder.
            raise HolderStartError(
                f"providers.{pid}: has a key_ref but a local kind and a local_endpoints origin; "
                "remove key_ref (local) or move base_url off local_endpoints (hosted)"
            )
    providers = {
        pid: _provider(pid, dict(raw))
        for pid, raw in cfg.providers.items()
        if cfg.provider_is_hosted(pid)
    }
    table = dict(cfg.holder)
    ceilings = None
    if providers:
        try:
            ceilings = Ceilings.from_table(table)
        except ValueError as exc:
            raise HolderStartError(f"hosted providers need a window ceiling: {exc}") from None
    unmatched = table.get("unmatched", "refuse")
    if unmatched not in ("refuse", "cap"):
        raise HolderStartError("holder.unmatched must be 'refuse' or 'cap'")
    default_bound = None
    if unmatched == "cap":
        try:
            default_bound = positive_int(table.get("default_bound"), "default_bound")
        except ValueError as exc:
            raise HolderStartError(str(exc)) from None
    bounds = []
    for entry in cfg.run_bounds:
        pattern = entry.get("match")
        if not isinstance(pattern, str) or not pattern:
            raise HolderStartError("run_bounds: match must be a non-empty glob")
        try:
            bounds.append((pattern, positive_int(entry.get("max_tokens"), "run_bounds.max_tokens")))
        except ValueError as exc:
            raise HolderStartError(str(exc)) from None
    ca_file = table.get("ca_file")
    if ca_file is not None and not isinstance(ca_file, str):
        raise HolderStartError("holder.ca_file must be a path")
    timeout = table.get("upstream_timeout_seconds", 120)
    max_response = table.get("max_response_bytes", 8 << 20)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise HolderStartError("holder.upstream_timeout_seconds must be positive")
    try:
        max_response = positive_int(max_response, "max_response_bytes", MAX_BROKER_RESPONSE_BYTES)
    except ValueError as exc:
        raise HolderStartError(str(exc)) from None
    by_url: dict[str, Provider] = {}
    for p in providers.values():
        try:
            origin = canonical_origin(p.base_url)
        except ValueError as exc:
            raise HolderStartError(f"providers.{p.id}: invalid base_url ({exc})") from None
        if origin in by_url:
            raise HolderStartError(
                f"providers.{p.id} and providers.{by_url[origin].id} share the origin {origin}"
            )
        by_url[origin] = p
    return Policy(
        by_url,
        ceilings,
        unmatched,
        default_bound,
        tuple(bounds),
        ca_file,
        float(timeout),
        max_response,
    )


# --- the holder ---


@dataclass
class HolderSettings:
    config_source: ConfigSource
    state_root: str = DEFAULT_STATE_ROOT
    socket_path: str | None = None  # default: `[deployment] holder_socket`, else DEFAULT_SOCKET
    clock: Callable[[], float] = time.time
    resolver: Callable[[str, int], list[str]] = default_resolver
    upstream: Upstream = https_upstream
    code_root_check: Callable[[], list[str]] | None = None
    max_frame_bytes: int = 4 << 20
    read_timeout: float = 5.0
    max_connections: int = 64
    max_nonces: int = 100_000
    log_max_bytes: int = LOG_MAX_BYTES


class Holder:
    def __init__(self, settings: HolderSettings):
        self.s = settings
        if settings.code_root_check is not None:
            problems = settings.code_root_check()
            if problems:
                raise HolderStartError("code root is not protected: " + "; ".join(problems[:3]))
        self._policy_for: tuple[GatewayConfig, Policy] | None = None
        self._policy = self._policy_now()  # raises HolderStartError
        self._nonces: dict[tuple[str, str], float] = {}
        self._nonce_lock = threading.Lock()
        self._log_lock = threading.Lock()
        try:
            self.ledger = Ledger(
                os.path.join(settings.state_root, "ledger.sqlite3"), clock=settings.clock
            )
        except LedgerLocked as exc:
            raise HolderStartError(str(exc)) from None
        self._log_path = os.path.join(settings.state_root, "calls.log")
        self._log_fd = self._open_log()
        self._stop = threading.Event()
        self._sock: socket.socket | None = None

    def _open_log(self) -> int:
        old = os.umask(0o077)
        try:
            return os.open(
                self._log_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
        finally:
            os.umask(old)

    def _rotate_log(self) -> None:
        """Move `calls.log` to `calls.log.1` (replacing the previous one) and start a new file
        with the old file's mode and owner. Called under `_log_lock`; a failure keeps appending
        to the current file (a log that cannot rotate must not lose the call)."""
        try:
            before = os.fstat(self._log_fd)
            os.replace(self._log_path, self._log_path + ".1")
            fd = self._open_log()
            try:
                os.fchmod(fd, stat.S_IMODE(before.st_mode))
                after = os.fstat(fd)
                if (after.st_uid, after.st_gid) != (before.st_uid, before.st_gid):
                    os.fchown(fd, before.st_uid, before.st_gid)
            except OSError:
                os.close(fd)
                raise
        except OSError as exc:
            self._diag(f"log rotation failed {type(exc).__name__}")
            return
        os.close(self._log_fd)
        self._log_fd = fd

    # -- configuration, per request ----------------------------------------------------------

    def _policy_now(self) -> Policy:
        cfg = self.s.config_source.get()
        if self._policy_for is not None and self._policy_for[0] is cfg:
            return self._policy_for[1]
        policy = build_policy(cfg)
        self._policy_for = (cfg, policy)
        self._policy = policy
        return policy

    def socket_path(self) -> str:
        if self.s.socket_path:
            return self.s.socket_path
        path = self.s.config_source.get().deployment.get("holder_socket", DEFAULT_SOCKET)
        return path if isinstance(path, str) else DEFAULT_SOCKET

    # -- authentication ----------------------------------------------------------------------

    def _secret(self, caller: str) -> bytes | None:
        if not CALLER_NAME.fullmatch(caller):
            return None
        try:
            return framing.load_secret(
                os.path.join(self.s.state_root, "registry", caller + ".secret")
            )
        except (OSError, ValueError):
            return None

    def _remember_nonce(self, caller: str, nonce: str, now: float) -> bool:
        """Cache a nonce (only ever called after the MAC verified). False when seen before."""
        with self._nonce_lock:
            if len(self._nonces) >= self.s.max_nonces // 2:
                self._nonces = {k: t for k, t in self._nonces.items() if t > now}
            if self._nonces.get((caller, nonce), 0) > now:
                return False
            if len(self._nonces) >= self.s.max_nonces:
                return False  # a credential holder flooding fresh nonces: fail closed
            self._nonces[(caller, nonce)] = now + 2 * MAX_SKEW_SECONDS
            return True

    # -- one request -------------------------------------------------------------------------

    def process(self, payload: bytes) -> bytes | None:
        """One request frame payload -> one response frame (None: drop the connection)."""
        try:
            hdr, body = framing.parse_request(payload)
        except framing.FrameError:
            return None
        secret = self._secret(hdr.caller)
        if secret is None:
            return self._refuse(None, hdr.nonce, "unknown_caller")
        expected = framing.request_mac(
            secret,
            caller=hdr.caller,
            ts=hdr.ts,
            nonce=hdr.nonce,
            method=hdr.method,
            endpoint=hdr.endpoint,
            path=hdr.path,
            run_id=hdr.run_id,
            body=body,
        )
        if not _consteq(hdr.mac, expected):
            return self._refuse(None, hdr.nonce, "bad_credential")
        now = self.s.clock()
        if abs(now - hdr.ts) > MAX_SKEW_SECONDS:
            return self._answer(secret, hdr, "stale_timestamp")
        if not self._remember_nonce(hdr.caller, hdr.nonce, now):
            return self._answer(secret, hdr, "replayed_nonce")
        tokens = 0
        status = framing.REFUSAL_STATUS
        connected: list[str] = []  # the one address a connection was attempted to, if any
        try:
            reply, status, tokens = self._serve(secret, hdr, body, connected)
        except Refusal as exc:
            reply = self._answer(secret, hdr, exc.code)
        except BodyRefused as exc:
            reply = self._answer(secret, hdr, exc.code)
        except Exception as exc:  # never a traceback or message: the type name only
            self._diag(f"internal error {type(exc).__name__}")
            reply = self._upstream_failure("upstream-error", secret, hdr.nonce)
            status = 0
        self._log_call(now, hdr, status, tokens, connected[0] if connected else None)
        return reply

    def _answer(self, secret: bytes, hdr: framing.RequestHeader, code: str) -> bytes:
        return self._refuse(secret, hdr.nonce, code)

    @staticmethod
    def _refuse(secret: bytes | None, nonce: str, code: str) -> bytes:
        assert code in HOLDER_REFUSALS
        return framing.build_response(
            secret,
            request_nonce=nonce,
            close_reason="completed",
            status=framing.REFUSAL_STATUS,
            content_type=framing.REFUSAL_CONTENT_TYPE,
            body=framing.refusal_body(code),
        )

    def _serve(
        self, secret: bytes, hdr: framing.RequestHeader, body: bytes, connected: list[str]
    ) -> tuple[bytes, int, int]:
        try:
            policy = self._policy_now()
        except (HolderStartError, ConfigError):
            raise Refusal("endpoint_unknown") from None
        try:
            provider = policy.providers.get(canonical_origin(hdr.endpoint))
        except ValueError:
            raise Refusal("endpoint_unknown") from None
        if provider is None or policy.ceilings is None:
            raise Refusal("endpoint_unknown")
        if hdr.method != "POST" or hdr.path not in provider.spec.paths:
            raise Refusal("path_not_allowed")
        parts = urllib.parse.urlsplit(provider.base_url)
        if parts.scheme != "https" or not parts.hostname:
            raise Refusal("https_required")
        try:
            port = parts.port or 443
        except ValueError:
            raise Refusal("endpoint_unknown") from None
        if (
            parts.username
            or parts.password
            or parts.query
            or parts.fragment
            or parts.path not in ("", "/")
        ):
            raise Refusal("endpoint_unknown")
        limit, _model = validate_body(provider, body)
        key = read_key(provider.key_ref)
        if key is None:
            raise Refusal("key_missing")
        host = parts.hostname
        try:
            addresses = [host] if _is_ip(host) else self.s.resolver(host, port)
        except OSError:
            return self._upstream_failure("upstream-error", secret, hdr.nonce), 0, 0
        if not addresses or not all(address_allowed(a, provider.allow_private) for a in addresses):
            raise Refusal("address_forbidden")
        run_bound, cap_bound = self._bound_for(policy, hdr.run_id)
        amount = limit + len(body)
        rid = self.ledger.reserve(
            amount=amount,
            ceilings=policy.ceilings,
            run_id=hdr.run_id,
            run_bound=run_bound,
            cap_bound=cap_bound,
        )
        connected.append(addresses[0])
        try:
            result = self.s.upstream(
                host=host,
                port=port,
                ip=addresses[0],
                path=hdr.path,
                headers=_headers(provider, key, len(body), host),
                body=body,
                ca_file=policy.ca_file,
                timeout=policy.upstream_timeout,
                max_response=policy.max_response,
            )
        except Exception as exc:
            self.ledger.keep(rid)  # unknown whether anything was sent: the safe floor
            self._diag(f"upstream error {type(exc).__name__}")  # the type only, never the message
            return self._upstream_failure("upstream-error", secret, hdr.nonce), 0, amount
        return self._settle(secret, hdr, provider, policy, rid, amount, key, result)

    def _bound_for(self, policy: Policy, run_id: str) -> tuple[int | None, int | None]:
        if len(run_id) > MAX_RUN_ID_CHARS or not run_id.isprintable():
            raise Refusal("run_unmatched")
        if run_id:
            for pattern, bound in policy.run_bounds:
                if fnmatch.fnmatchcase(run_id, pattern):
                    return bound, None
        if policy.unmatched == "cap":
            return None, policy.default_bound
        raise Refusal("run_unmatched" if run_id else "run_id_missing")

    def _settle(
        self,
        secret: bytes,
        hdr: framing.RequestHeader,
        provider: Provider,
        policy: Policy,
        rid: int,
        amount: int,
        key: str,
        res: UpstreamResult,
    ) -> tuple[bytes, int, int]:
        ceilings = policy.ceilings
        assert ceilings is not None
        counted = amount
        if not res.wrote:
            if self.ledger.release(rid, ceilings):
                counted = 0
        elif res.close_reason != "completed":
            self.ledger.keep(rid)
        elif 200 <= res.status < 300:
            used = parse_usage(provider, res.body)
            if used is None:
                self.ledger.keep(rid)
            else:
                self.ledger.settle(rid, used)
                counted = used
        elif releasable(res.status, res.body):
            if self.ledger.release(rid, ceilings):
                counted = 0
        else:
            self.ledger.keep(rid)
        if res.close_reason != "completed":
            return self._upstream_failure(res.close_reason, secret, hdr.nonce), 0, counted
        body = scrub(res.body, key)
        reply = framing.build_response(
            secret,
            request_nonce=hdr.nonce,
            close_reason="completed",
            status=res.status,
            content_type=res.content_type,
            body=body,
        )
        return reply, res.status, counted

    def _upstream_failure(self, reason: str, secret: bytes | None = None, nonce: str = "") -> bytes:
        return framing.build_response(
            secret, request_nonce=nonce, close_reason=reason, status=0, content_type="", body=b""
        )

    # -- logs --------------------------------------------------------------------------------

    def _log_call(
        self,
        now: float,
        hdr: framing.RequestHeader,
        status: int,
        tokens: int,
        resolved_addr: str | None,
    ) -> None:
        """`calls.log`: ts, caller, run_id, endpoint, status, tokens, resolved_addr (the pinned
        address connected to; null when no connection was attempted); no body, no key.

        `calls.log` rotates by size (one previous file). Known limit: the ledger's `reservations`
        rows are never pruned (review P5-4), so a credential holder's call volume still grows
        `/var/lib/llmholder`; the holder's own disk, under the human's watch, is the bound."""
        line = json.dumps(
            {
                "ts": int(now),
                "caller": hdr.caller,
                "run_id": hdr.run_id[:MAX_RUN_ID_CHARS],
                "endpoint": hdr.endpoint[:256],
                "status": status,
                "tokens": tokens,
                "resolved_addr": resolved_addr,
            },
            separators=(",", ":"),
        )
        data = line.encode("utf-8") + b"\n"
        with self._log_lock:
            if os.fstat(self._log_fd).st_size + len(data) > self.s.log_max_bytes:
                self._rotate_log()
            os.write(self._log_fd, data)

    @staticmethod
    def _diag(message: str) -> None:
        print(f"keyholder: {message}", file=sys.stderr, flush=True)

    # -- the socket --------------------------------------------------------------------------

    def bind(self) -> str:
        path = self.socket_path()
        try:
            if stat.S_ISSOCK(os.lstat(path).st_mode):
                os.unlink(path)
            else:
                raise HolderStartError(f"{path} exists and is not a socket")
        except FileNotFoundError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0)
        try:
            sock.bind(path)
        finally:
            os.umask(old)
        os.chmod(path, 0o666)  # the MAC authenticates, not the mode
        sock.listen(256)
        sock.settimeout(0.2)
        self._sock = sock
        return path

    def serve_forever(self) -> None:
        assert self._sock is not None
        slots = threading.BoundedSemaphore(self.s.max_connections)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            if not slots.acquire(blocking=False):
                conn.close()  # per-connection limit
                continue
            threading.Thread(target=self._handle, args=(conn, slots), daemon=True).start()

    def _handle(self, conn: socket.socket, slots: threading.BoundedSemaphore) -> None:
        try:
            try:
                payload = framing.read_frame(
                    conn, max_bytes=self.s.max_frame_bytes, timeout=self.s.read_timeout
                )
            except framing.FrameError:
                return
            reply = self.process(payload)
            if reply is not None:
                conn.settimeout(30)
                conn.sendall(reply)
        except Exception as exc:
            self._diag(f"connection error {type(exc).__name__}")
        finally:
            conn.close()
            slots.release()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()
        if self._sock is not None:
            self._sock.close()
        self.ledger.close()
        os.close(self._log_fd)


def _consteq(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _headers(provider: Provider, key: str, length: int, host: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(length),
        "Connection": "close",
    }
    if provider.kind == "anthropic":
        headers["x-api-key"] = key
        headers["anthropic-version"] = provider.anthropic_version
    else:
        headers["Authorization"] = "Bearer " + key
    return headers


# --- check-deployment ---


class FS(Protocol):
    def lstat(self, path: str) -> os.stat_result: ...
    def listdir(self, path: str) -> list[str]: ...


class RealFS:
    def lstat(self, path: str) -> os.stat_result:
        return os.lstat(path)

    def listdir(self, path: str) -> list[str]:
        return sorted(os.listdir(path))


def _split(path: str) -> list[str]:
    return [p for p in os.path.abspath(path).split("/") if p]


def check_code_root(
    root: str,
    fs: FS | None = None,
    trusted_uids: Iterable[int] = PRODUCTION_TRUSTED_UIDS,
    anchor: str | None = None,
) -> list[str]:
    """The code root and every ancestor are trusted-owned and not group/other-writable, and every
    entry under the root is, by `lstat`, a regular file or a directory (no symlink, device,
    socket or fifo)."""
    fs = fs or RealFS()
    trusted = frozenset(trusted_uids)
    problems: list[str] = []
    parts = _split(root)
    start = len(_split(anchor)) if anchor else 0
    for i in range(start, len(parts) + 1):
        path = "/" + "/".join(parts[:i])
        try:
            st = fs.lstat(path)
        except OSError as exc:
            return problems + [f"{path}: {exc.strerror}"]
        if not stat.S_ISDIR(st.st_mode):
            problems.append(f"{path}: ancestor is not a plain directory")
        if st.st_uid not in trusted:
            problems.append(f"{path}: owned by uid {st.st_uid}")
        if st.st_mode & 0o022:
            problems.append(f"{path}: group/other-writable")

    def walk(path: str) -> None:
        for name in fs.listdir(path):
            child = os.path.join(path, name)
            st = fs.lstat(child)
            if stat.S_ISLNK(st.st_mode) or not (
                stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)
            ):
                problems.append(f"{child}: not a regular file or directory")
                continue
            if st.st_uid not in trusted:
                problems.append(f"{child}: owned by uid {st.st_uid}")
            if st.st_mode & 0o022:
                problems.append(f"{child}: group/other-writable")
            if stat.S_ISDIR(st.st_mode):
                walk(child)

    if not problems:
        walk("/" + "/".join(parts))
    return problems


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str
    kind: str  # 'file' | 'dir' | 'sock'
    uid: int
    mode: int
    required: bool = True


@dataclass
class Layout:
    code_root: str
    holder_uid: int
    etc_dir: str = "/etc/llmgateway"
    unit_path: str = "/etc/systemd/system/llmholder.service"
    state_root: str = DEFAULT_STATE_ROOT
    run_dir: str = "/run/llmholder"
    socket_path: str = DEFAULT_SOCKET
    caller_creds: Sequence[tuple[int, str]] = ()
    user_units: Sequence[tuple[int, str]] = ()
    state_dirs: Sequence[tuple[int, str]] = ()
    trusted_uids: frozenset[int] = PRODUCTION_TRUSTED_UIDS
    anchor: str | None = None


def artifact_table(layout: Layout, fs: FS) -> list[Artifact]:
    root_uid = min(layout.trusted_uids)
    h, s, e = layout.holder_uid, layout.state_root, layout.etc_dir
    rows = [
        Artifact(layout.unit_path, "file", root_uid, 0o644),
        Artifact(e, "dir", root_uid, 0o755),
        Artifact(os.path.join(e, "ccg"), "dir", root_uid, 0o755),
        Artifact(os.path.join(e, "gateway.toml"), "file", root_uid, 0o644),
        Artifact(os.path.join(e, "ccg", "system.yaml"), "file", root_uid, 0o644),
        Artifact(s, "dir", h, 0o700),
        Artifact(os.path.join(s, "keys"), "dir", h, 0o700),
        Artifact(os.path.join(s, "registry"), "dir", h, 0o700),
        Artifact(layout.run_dir, "dir", h, 0o711),
        Artifact(layout.socket_path, "sock", h, 0o666),
    ]
    for sub in ("keys", "registry"):
        try:
            names = fs.listdir(os.path.join(s, sub))
        except OSError:
            names = []
        rows += [Artifact(os.path.join(s, sub, n), "file", h, 0o600) for n in names]
    for name in (
        "ledger.sqlite3",
        "ledger.sqlite3-wal",
        "ledger.sqlite3-shm",
        "ledger.sqlite3.lock",
        "calls.log",
    ):
        rows.append(Artifact(os.path.join(s, name), "file", h, 0o600, required=name == "calls.log"))
    rows += [Artifact(p, "file", u, 0o600) for u, p in layout.caller_creds]
    rows += [Artifact(p, "file", u, 0o644) for u, p in layout.user_units]
    rows += [Artifact(p, "dir", u, 0o700) for u, p in layout.state_dirs]
    return rows


_KIND_TEST = {"file": stat.S_ISREG, "dir": stat.S_ISDIR, "sock": stat.S_ISSOCK}


def check_deployment(layout: Layout, fs: FS | None = None, *, code_only: bool = False) -> list[str]:
    """Stat every artifact against the §3.4 table; an empty list means the layout is correct."""
    fs = fs or RealFS()
    problems = check_code_root(layout.code_root, fs, layout.trusted_uids, layout.anchor)
    if code_only:
        return problems
    for art in artifact_table(layout, fs):
        try:
            st = fs.lstat(art.path)
        except FileNotFoundError:
            if art.required:
                problems.append(f"{art.path}: missing")
            continue
        except OSError as exc:
            problems.append(f"{art.path}: {exc.strerror}")
            continue
        if not _KIND_TEST[art.kind](st.st_mode):  # lstat: a symlink is never a regular file
            problems.append(f"{art.path}: not a {art.kind}")
        elif st.st_uid != art.uid:
            problems.append(f"{art.path}: owner {st.st_uid}, want {art.uid}")
        elif stat.S_IMODE(st.st_mode) != art.mode:
            problems.append(f"{art.path}: mode {stat.S_IMODE(st.st_mode):04o}, want {art.mode:04o}")
    return problems


# --- add-caller and install-user-file ---

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


class WriteRefused(Exception):
    pass


def _drop_to(uid: int, gid: int) -> None:
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    if os.geteuid() != uid:
        raise WriteRefused("cannot act as the caller uid")


def write_in_home(
    path: str, home: str, uid: int, content: bytes, *, mode: int, replace: bool
) -> None:
    """Create `path` (under `home`) as the current uid, which must be `uid`: walk it with
    `openat(O_NOFOLLOW|O_DIRECTORY)` per component, every directory from `home` down owned by
    `uid`, then write a temp file and rename it. A symlink anywhere in the chain is refused."""
    parts, home_parts = _split(path), _split(home)
    if ".." in path.split("/") or not path.startswith("/") or "\0" in path:
        raise WriteRefused("path must be absolute without '..'")
    if len(parts) <= len(home_parts) or parts[: len(home_parts)] != home_parts:
        raise WriteRefused("path is not under the caller's home")
    fd = os.open("/", _DIR_FLAGS)
    try:
        for i, comp in enumerate(parts[:-1]):
            try:
                nfd = os.open(comp, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if i < len(home_parts):
                    raise WriteRefused(f"{comp}: home path does not exist") from None
                os.mkdir(comp, 0o700, dir_fd=fd)
                nfd = os.open(comp, _DIR_FLAGS, dir_fd=fd)
            except OSError as exc:
                raise WriteRefused(f"{comp}: symlink or not a directory ({exc.strerror})") from None
            os.close(fd)
            fd = nfd
            if i >= len(home_parts) - 1 and os.fstat(fd).st_uid != uid:
                raise WriteRefused(f"{comp}: directory not owned by the caller")
        name = parts[-1]
        try:
            os.stat(name, dir_fd=fd, follow_symlinks=False)
            exists = True
        except FileNotFoundError:
            exists = False
        if exists and not replace:
            raise WriteRefused(f"{name}: exists (use --rotate)")
        tmp = f".{name}.{secrets.token_hex(8)}.tmp"
        tfd = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            mode,
            dir_fd=fd,
        )
        try:
            os.fchmod(tfd, mode)  # the child's umask is 077
            os.write(tfd, content)
            os.fsync(tfd)
        finally:
            os.close(tfd)
        os.rename(tmp, name, src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        os.close(fd)


def _write_via_child(
    path: str, home: str, uid: int, gid: int, content: bytes, *, mode: int, replace: bool
) -> None:
    """Fork a child that drops to the caller uid (groups cleared) and writes; the content
    arrives over a pipe, never argv."""
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            os.close(wfd)
            _drop_to(uid, gid)
            chunks = []
            while chunk := os.read(rfd, 65536):
                chunks.append(chunk)
            write_in_home(path, home, uid, b"".join(chunks), mode=mode, replace=replace)
            code = 0
        except WriteRefused as exc:
            print(f"keyholder: refused: {exc}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"keyholder: write failed: {type(exc).__name__}", file=sys.stderr, flush=True)
        finally:
            os._exit(code)
    os.close(rfd)
    try:
        view = memoryview(content)
        while view:
            view = view[os.write(wfd, view) :]
    finally:
        os.close(wfd)
    _, status = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        raise WriteRefused("the caller-side write failed")


def _write_registry(
    state_root: str, name: str, holder_uid: int, holder_gid: int, secret_hex: bytes, replace: bool
) -> None:
    directory = os.path.join(state_root, "registry")
    dfd = os.open(directory, _DIR_FLAGS)
    try:
        try:
            os.stat(name + ".secret", dir_fd=dfd, follow_symlinks=False)
            if not replace:
                raise WriteRefused(f"registry entry {name} exists (use --rotate)")
        except FileNotFoundError:
            pass
        tmp = f".{name}.{secrets.token_hex(8)}.tmp"
        fd = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=dfd,
        )
        try:
            os.fchown(fd, holder_uid, holder_gid)
            os.write(fd, secret_hex)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(tmp, name + ".secret", src_dir_fd=dfd, dst_dir_fd=dfd)
    finally:
        os.close(dfd)


def _account(uid: int | None, name: str | None) -> tuple[int, int, str]:
    try:
        rec = pwd.getpwuid(uid) if uid is not None else pwd.getpwnam(name or HOLDER_ACCOUNT)
    except KeyError:
        raise WriteRefused("unknown account") from None
    return rec.pw_uid, rec.pw_gid, rec.pw_dir


def cmd_add_caller(args: argparse.Namespace) -> int:
    if not CALLER_NAME.fullmatch(args.name):
        raise WriteRefused("caller name must match [a-z0-9_-]{1,32}")
    c_uid, c_gid, c_home = _account(args.caller_uid, None)
    if args.holder_uid is not None:
        h_uid = args.holder_uid
        try:
            h_gid = pwd.getpwuid(h_uid).pw_gid
        except KeyError:
            h_gid = h_uid
    else:
        h_uid, h_gid, _ = _account(None, None)
    if args.holder_gid is not None:
        h_gid = args.holder_gid
    home = args.home or c_home
    secret = os.urandom(32)
    secret_hex = secret.hex().encode() + b"\n"
    reg = os.path.join(args.state_root, "registry", args.name + ".secret")
    if not os.path.isdir(os.path.dirname(reg)):
        raise WriteRefused("the registry directory does not exist (run holder-account first)")
    if os.path.lexists(reg) and not args.rotate:
        raise WriteRefused(f"registry entry {args.name} exists (use --rotate)")
    # The caller-side copy first: a refused path changes nothing; the secret is never printed.
    _write_via_child(args.out, home, c_uid, c_gid, secret_hex, mode=0o600, replace=args.rotate)
    _write_registry(args.state_root, args.name, h_uid, h_gid, secret_hex, args.rotate)
    print(f"keyholder: caller {args.name} {'rotated' if args.rotate else 'added'}")
    return 0


def cmd_install_user_file(args: argparse.Namespace) -> int:
    c_uid, c_gid, c_home = _account(args.caller_uid, None)
    content = sys.stdin.buffer.read(1 << 20)
    _write_via_child(args.out, args.home or c_home, c_uid, c_gid, content, mode=0o644, replace=True)
    return 0


# --- log ---


def read_log(state_root: str = DEFAULT_STATE_ROOT) -> str:
    fd = os.open(os.path.join(state_root, "calls.log"), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        chunks = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks).decode("utf-8", errors="replace")


# --- entry ---


def _code_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _harden() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass
    try:
        import ctypes

        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE=0
    except (ImportError, OSError, AttributeError):
        pass


def cmd_serve(_args: argparse.Namespace) -> int:
    _harden()
    root = _code_root()
    settings = HolderSettings(
        config_source=ConfigSource(DEFAULT_PATH, PRODUCTION_TRUSTED_UIDS),
        code_root_check=lambda: check_code_root(root),
    )
    try:
        holder = Holder(settings)
        path = holder.bind()
    except (HolderStartError, ConfigError, OSError) as exc:
        detail = (
            str(exc) if isinstance(exc, (HolderStartError, ConfigError)) else type(exc).__name__
        )
        print(f"keyholder: refusing to start: {detail}", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, lambda *_: holder.stop())
    signal.signal(signal.SIGINT, lambda *_: holder.stop())
    print(f"keyholder: serving on {path}", file=sys.stderr, flush=True)
    holder.serve_forever()
    holder.close()
    return 0


def cmd_check_deployment(args: argparse.Namespace) -> int:
    holder_uid = args.holder_uid
    if holder_uid is None:
        holder_uid = -1 if args.code_only else _account(None, None)[0]
    layout = Layout(
        code_root=args.root or _code_root(),
        holder_uid=holder_uid,
        etc_dir=args.etc_dir,
        unit_path=args.unit,
        state_root=args.state_root,
        run_dir=os.path.dirname(args.socket),
        socket_path=args.socket,
        caller_creds=[_uid_path(v) for v in args.caller_cred],
        user_units=[_uid_path(v) for v in args.user_unit],
        state_dirs=[_uid_path(v) for v in args.state_dir],
        trusted_uids=frozenset(args.trusted_uid or PRODUCTION_TRUSTED_UIDS),
        anchor=args.anchor,
    )
    problems = check_deployment(layout, code_only=args.code_only)
    for line in problems:
        print(line)
    return 1 if problems else 0


def _uid_path(value: str) -> tuple[int, str]:
    uid, _, path = value.partition(":")
    return int(uid), path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="keyholder.py")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("serve")
    sub.add_parser("log")
    cd = sub.add_parser("check-deployment")
    cd.add_argument("--root")
    cd.add_argument("--anchor")
    cd.add_argument("--code-only", action="store_true")
    cd.add_argument("--trusted-uid", type=int, action="append")
    cd.add_argument("--holder-uid", type=int)
    cd.add_argument("--etc-dir", default="/etc/llmgateway")
    cd.add_argument("--unit", default="/etc/systemd/system/llmholder.service")
    cd.add_argument("--state-root", default=DEFAULT_STATE_ROOT)
    cd.add_argument("--socket", default=DEFAULT_SOCKET)
    cd.add_argument("--caller-cred", action="append", default=[], metavar="UID:PATH")
    cd.add_argument("--user-unit", action="append", default=[], metavar="UID:PATH")
    cd.add_argument("--state-dir", action="append", default=[], metavar="UID:PATH")
    ac = sub.add_parser("add-caller")
    ac.add_argument("name")
    ac.add_argument("--caller-uid", type=int, required=True)
    ac.add_argument("--out", required=True)
    ac.add_argument("--rotate", action="store_true")
    ac.add_argument("--home")
    ac.add_argument("--holder-uid", type=int)
    ac.add_argument("--holder-gid", type=int)
    ac.add_argument("--state-root", default=DEFAULT_STATE_ROOT)
    iu = sub.add_parser("install-user-file")
    iu.add_argument("--caller-uid", type=int, required=True)
    iu.add_argument("--out", required=True)
    iu.add_argument("--home")
    return p


def main(argv: Sequence[str] | None = None, *, state_root: str = DEFAULT_STATE_ROOT) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        match args.cmd:
            case None | "serve":
                return cmd_serve(args)
            case "log":
                sys.stdout.write(read_log(state_root))
                return 0
            case "check-deployment":
                return cmd_check_deployment(args)
            case "add-caller":
                return cmd_add_caller(args)
            case "install-user-file":
                return cmd_install_user_file(args)
    except WriteRefused as exc:
        print(f"keyholder: refused: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(
            f"keyholder: {errno.errorcode.get(exc.errno or 0, 'error')}: {exc.strerror}",
            file=sys.stderr,
        )
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
