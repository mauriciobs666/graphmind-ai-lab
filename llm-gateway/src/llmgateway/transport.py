"""Transports: how a wire request reaches a provider (docs/plans/llm-gateway.md §3.2, §3.4).

`Transport` is the seam: the engine hands it one `WireRequest` and gets one `WireResponse` (any
completed HTTP exchange, error statuses included) or an exception. A transport that cannot
complete an exchange raises `TransportFailure(condition)` (classified by `failures`), a
`CallFailed` it classified itself, or an `OSError`/`TimeoutError` (classified by
`failures.classify_exception`). Holder- and broker-backed transports (D8, D10) implement the
same two members and need no engine change; `reaches_hosted` tells the engine whether a hosted
step may be sent through it (FR-11).

`DirectTransport` is the caller-process transport: it connects only to origins `GatewayConfig`
calls local (loopback IP literals when the configuration is unprotected), and refuses anything
else with `HostedFromCallerProcess` before a socket is created.
"""

from __future__ import annotations

import http.client
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit

from llmgateway.config import GatewayConfig, parse_origin
from llmgateway.failures import Condition
from llmgateway.types import HostedFromCallerProcess

DEFAULT_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class WireRequest:
    """One request to one provider. `hosted` is the gateway's own judgement (`provider_is_hosted`);
    `run_id`, `component` and `scopes` are carried for holder and broker transports."""

    provider_id: str
    kind: str
    base_url: str
    path: str
    body: bytes
    hosted: bool
    method: str = "POST"
    headers: Mapping[str, str] = field(default_factory=dict)
    run_id: str | None = None
    component: str = ""
    scopes: tuple[str, ...] = ()
    timeout_s: float = 120.0


@dataclass(frozen=True, slots=True)
class WireResponse:
    status: int
    body: bytes
    content_type: str | None = None


class TransportFailure(Exception):
    """An exchange that did not complete; `condition` selects the §3.3 class."""

    def __init__(self, condition: Condition, message: str = "") -> None:
        super().__init__(message or condition.value)
        self.condition = condition


class Transport(Protocol):
    reaches_hosted: bool

    def exchange(self, request: WireRequest) -> WireResponse: ...


class DirectTransport:
    """HTTP(S) from the caller's own process to local providers only (FR-11, AC-4, AC-7).

    `config` is a `GatewayConfig` or a zero-argument callable returning the current one (the
    engine's config source), so the allow-list is evaluated on every exchange. No proxy is used
    and redirects are not followed.
    """

    reaches_hosted = False

    def __init__(
        self,
        config: GatewayConfig | Callable[[], GatewayConfig],
        *,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self._config = config if callable(config) else (lambda: config)
        self._max = max_response_bytes

    def exchange(self, request: WireRequest) -> WireResponse:
        config = self._config()
        origin = parse_origin(request.base_url)
        if origin is None or not config.is_local(request.base_url):
            raise HostedFromCallerProcess(
                f"provider {request.provider_id!r} is not a local endpoint: a hosted call "
                "goes only through the key-holding process"
            )
        scheme, host, port = origin
        prefix = urlsplit(request.base_url).path.rstrip("/")
        conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        headers = {**request.headers, "Content-Type": "application/json"}
        conn = conn_cls(host, port, timeout=request.timeout_s)
        try:
            conn.request(request.method, prefix + request.path, body=request.body, headers=headers)
            resp = conn.getresponse()
            data = resp.read(self._max + 1)
            content_type = resp.getheader("Content-Type")
            status = resp.status
        except http.client.HTTPException as exc:
            raise TransportFailure(
                Condition.CONNECTION_REFUSED_OR_RESET, type(exc).__name__
            ) from exc
        finally:
            conn.close()
        if len(data) > self._max:
            raise TransportFailure(Condition.BROKER_RESPONSE_TOO_LARGE, "response too large")
        return WireResponse(status=status, body=data, content_type=content_type)
