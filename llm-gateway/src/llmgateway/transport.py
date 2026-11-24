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

`HolderTransport` is the service's path to the key holder (§3.4): every hosted exchange is signed
with the caller credential (a fresh nonce per exchange, so a retry never replays) and sent over
the holder's Unix socket; the reply is verified against the request nonce. A completed holder
exchange (a provider reply or a refusal, status 499) comes back as a `WireResponse` for the engine
to classify; an exchange that did not complete is a `TransportFailure`.
"""

from __future__ import annotations

import http.client
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit

from llmgateway import framing
from llmgateway.config import GatewayConfig, parse_origin
from llmgateway.failures import Condition
from llmgateway.types import CallFailed, FailureClass, HostedFromCallerProcess

DEFAULT_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
# `keyholder.DEFAULT_SOCKET`, repeated so this module need not import the holder
# (`tests/test_service_holder.py` pins the two together).
DEFAULT_HOLDER_SOCKET = "/run/llmholder/holder.sock"
_HOLDER_CLOSE_CONDITIONS = {
    "upstream-error": Condition.BROKER_UPSTREAM_ERROR,
    "timeout": Condition.BROKER_TIMEOUT,
    "response-too-large": Condition.BROKER_RESPONSE_TOO_LARGE,
}


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
        if (
            request.hosted
            or (
                request.provider_id in config.providers
                and config.provider_is_hosted(request.provider_id)
            )
            or origin is None
            or not config.is_local(request.base_url)
        ):
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


def _config_failure(message: str) -> CallFailed:
    return CallFailed(FailureClass.CONFIG, message)


class HolderTransport:
    """Hosted exchanges from the service to the key holder (FR-8, FR-11, AC-5).

    `config` is a `GatewayConfig` or a callable returning the current one; `[deployment]`
    `holder_socket` and `holder_uid` are read from it on every exchange, and only a protected
    configuration is trusted for them. `secret` is the caller credential (read by the service at
    start; never logged, never in an error). Only hosted requests are accepted: a local one is a
    routing bug and fails `config` before a socket opens. The wire request's headers are ignored
    (the holder adds the key and the provider headers itself); its `base_url` is the endpoint the
    holder matches by canonical origin.
    """

    reaches_hosted = True

    def __init__(
        self,
        config: GatewayConfig | Callable[[], GatewayConfig],
        secret: bytes,
        *,
        caller: str = "service",
        own_uid: int | None = None,
    ) -> None:
        if len(secret) != 32:
            raise ValueError("the caller credential must be 32 bytes")
        self._config = config if callable(config) else (lambda: config)
        self._secret = secret
        self._caller = caller
        self._own_uid = os.getuid() if own_uid is None else own_uid

    def __repr__(self) -> str:
        return f"HolderTransport(caller={self._caller!r})"

    def _deployment(self) -> tuple[str, int]:
        config = self._config()
        if not config.protected:
            raise _config_failure(
                "the gateway configuration is not owner-protected: hosted calls are disabled"
            )
        socket_path = config.deployment.get("holder_socket", DEFAULT_HOLDER_SOCKET)
        holder_uid = config.deployment.get("holder_uid")
        if not isinstance(socket_path, str) or not socket_path.startswith("/"):
            raise _config_failure("[deployment] holder_socket must be an absolute path")
        if isinstance(holder_uid, bool) or not isinstance(holder_uid, int) or holder_uid < 0:
            raise _config_failure("[deployment] holder_uid must be set to the holder's uid")
        if holder_uid == self._own_uid:
            # a directory owned by the caller's own uid is agent-writable: any process could
            # stand in for the holder
            raise _config_failure("[deployment] holder_uid is the caller's own uid")
        return socket_path, holder_uid

    def exchange(self, request: WireRequest) -> WireResponse:
        if not request.hosted:
            raise _config_failure(f"provider {request.provider_id!r} is local: not a holder call")
        socket_path, holder_uid = self._deployment()
        try:
            reply = framing.exchange(
                socket_path,
                holder_uid,
                self._secret,
                caller=self._caller,
                endpoint=request.base_url,
                path=request.path,
                body=request.body,
                run_id=request.run_id,
                method=request.method,
                timeout=request.timeout_s,
            )
        except framing.HolderUnreachable as exc:
            # `framing.exchange` reports a missed read deadline as an unreachable holder
            condition = (
                Condition.SOCKET_TIMEOUT
                if "read timeout" in str(exc)
                else Condition.HOLDER_SOCKET_UNREACHABLE
            )
            raise TransportFailure(condition, "key holder: " + str(exc)) from exc
        except (framing.ReplyRejected, framing.FrameError) as exc:
            raise _config_failure(
                "the key holder's reply failed authentication (not the holder, or tampered)"
            ) from exc
        if reply.close_reason != "completed":
            condition = _HOLDER_CLOSE_CONDITIONS[reply.close_reason]
            raise TransportFailure(condition, f"key holder: {reply.close_reason}")
        return WireResponse(reply.status, reply.body, reply.content_type)
