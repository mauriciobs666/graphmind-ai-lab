"""Single-file stdlib client for the gateway service (docs/plans/llm-gateway.md §3.1, §3.3, §3.10).

Follow-on components that cannot import outside their own folder (model-bench, decision-models)
byte-copy THIS FILE and nothing else. It therefore imports only the standard library: not the
rest of `llmgateway` (`tests/test_client.py` imports it from an otherwise empty `sys.path` and
pins its imports and bytes). Its names are its own; the service's wire protocol is the contract.

Protocol (one request and one response per connection on the service's Unix socket): a frame is
`u32 big-endian length` + that many bytes of UTF-8 JSON. A request is `{"v": 1, "op": <op>, ...}`
with `op` one of `call`, `load`, `provider_op`, `embed`; a reply is `{"ok": true, "result": ...}`
or `{"ok": false, "error": {"type", "class", "message", "code", "call_id", "classes"}}`.

    client = Client()                       # default socket, see `default_socket_path`
    reply = client.call("answer.summarize", [{"role": "user", "content": "hi"}])
    reply.message, reply.model, reply.hosted, reply.usage

A failed call raises `CallFailed` carrying the service's failure class (a string such as
`rate_limit`, `run_bound`, `policy_refused`); an unreachable service is `connection`.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 64 * 1024 * 1024
SOCKET_ENV = "LLMGATEWAY_SERVICE_SOCKET"
_U32 = struct.Struct(">I")

TEMPORARY_CLASSES = frozenset(
    {"rate_limit", "out_of_credit", "outage", "timeout", "connection", "model_not_loaded"}
)
PERMANENT_CLASSES = frozenset(
    {"auth", "permission", "rejected", "policy_refused", "run_bound", "config"}
)


class GatewayError(Exception):
    """Base of every error this client raises."""


class EmbeddingsNotAvailable(GatewayError):
    """Embeddings are refused at first delivery."""


class HostedFromCallerProcess(GatewayError):
    """A hosted call was attempted from a caller's own process."""


class CallFailed(GatewayError):
    """A call failed with a classified cause. `code` names a key-holder refusal when one
    applies; `call_id` and `classes` (every attempt's class, in order) are set when the service
    ran the attempts."""

    def __init__(
        self,
        failure_class: str,
        message: str,
        code: str | None = None,
        call_id: str | None = None,
        classes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.code = code
        self.call_id = call_id
        self.classes = tuple(classes)

    @property
    def temporary(self) -> bool:
        return self.failure_class in TEMPORARY_CLASSES


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Response:
    """A completed call; `model` is the provider-reported identity."""

    call_id: str
    provider: str
    model: str
    hosted: bool
    message: Mapping[str, Any]
    usage: Usage = field(default_factory=Usage)
    provider_detail: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


def default_socket_path(env: Mapping[str, str] | None = None) -> str:
    """`$LLMGATEWAY_SERVICE_SOCKET`, else `$XDG_RUNTIME_DIR/llmgateway/service.sock`, else
    `/run/user/<uid>/llmgateway/service.sock`."""
    env = os.environ if env is None else env
    explicit = env.get(SOCKET_ENV)
    if explicit:
        return explicit
    runtime = env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return os.path.join(runtime, "llmgateway", "service.sock")


def _unwrap(reply: Mapping[str, Any]) -> Any:
    if reply.get("ok") is True:
        return reply.get("result")
    error = reply.get("error")
    if not isinstance(error, Mapping):
        raise CallFailed("config", "the service sent a reply this client cannot read")
    kind, message = error.get("type"), str(error.get("message", ""))
    if kind == "HostedFromCallerProcess":
        raise HostedFromCallerProcess(message)
    if kind == "EmbeddingsNotAvailable":
        raise EmbeddingsNotAvailable(message)
    classes = error.get("classes")
    raise CallFailed(
        str(error.get("class", "config")),
        message,
        error.get("code"),
        error.get("call_id"),
        classes if isinstance(classes, list) else (),
    )


def _read_exact(sock: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    while count:
        chunk = sock.recv(min(count, 1 << 20))
        if not chunk:
            raise ConnectionError("the service closed the connection")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


class Client:
    """One service socket. Every method opens one connection; the object holds no state, so it
    is safe to share between threads."""

    def __init__(self, socket_path: str | None = None, *, timeout: float = 900.0) -> None:
        self._path = socket_path or default_socket_path()
        self._timeout = timeout

    def _check_directory(self) -> None:
        directory = os.path.dirname(os.path.abspath(self._path))
        try:
            st = os.lstat(directory)
        except OSError as exc:
            raise CallFailed("connection", f"gateway service socket directory: {exc.strerror}")
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o022:
            raise CallFailed(
                "config", "the service socket directory is not private to this account"
            )

    def _roundtrip(self, payload: Mapping[str, Any]) -> Any:
        self._check_directory()
        data = json.dumps({"v": PROTOCOL_VERSION, **payload}, separators=(",", ":")).encode()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(self._timeout)
            try:
                sock.connect(self._path)
                sock.sendall(_U32.pack(len(data)) + data)
                (size,) = _U32.unpack(_read_exact(sock, 4))
                if size > MAX_FRAME_BYTES:
                    raise CallFailed("config", "the service reply is over the frame cap")
                reply = json.loads(_read_exact(sock, size))
            except TimeoutError as exc:
                raise CallFailed("timeout", "the gateway service did not answer in time") from exc
            except OSError as exc:
                raise CallFailed(
                    "connection", f"gateway service socket: {type(exc).__name__}"
                ) from exc
            except ValueError as exc:
                raise CallFailed("config", "the service sent a reply that is not JSON") from exc
        finally:
            sock.close()
        if not isinstance(reply, dict):
            raise CallFailed("config", "the service sent a reply that is not an object")
        return _unwrap(reply)

    def call(
        self,
        call_site: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        scopes: Sequence[str] = (),
        run_id: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        settings: Mapping[str, Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        retry: bool = True,
        fallback: bool = True,
        load_mode: str | None = None,
    ) -> Response:
        """A chat completion. There is no model or provider argument: the call site's owner
        configuration chooses both."""
        request: dict[str, Any] = {
            "call_site": call_site,
            "messages": list(messages),
            "scopes": list(scopes),
            "run_id": run_id,
            "settings": dict(settings or {}),
            "options": {"retry": retry, "fallback": fallback},
        }
        if tools is not None:
            request["tools"] = list(tools)
        if response_format is not None:
            request["response_format"] = dict(response_format)
        if load_mode is not None:
            request["load_mode"] = load_mode
        return self._response(self._roundtrip({"op": "call", "request": request}))

    def send(self, request: Mapping[str, Any]) -> Response:
        """A `call` whose request dict is passed as given (the service refuses unknown fields)."""
        return self._response(self._roundtrip({"op": "call", "request": dict(request)}))

    @staticmethod
    def _response(result: Any) -> Response:
        try:
            usage = result.get("usage") or {}
            return Response(
                call_id=result["call_id"],
                provider=result["provider"],
                model=result["model"],
                hosted=bool(result["hosted"]),
                message=result["message"],
                usage=Usage(usage.get("input_tokens"), usage.get("output_tokens")),
                provider_detail=result.get("provider_detail") or {},
                raw=result.get("raw") or {},
            )
        except (KeyError, AttributeError, TypeError) as exc:
            raise CallFailed(
                "config", "the service sent a call result this client cannot read"
            ) from exc

    def load(
        self, call_site: str, *, scopes: Sequence[str] = (), run_id: str | None = None
    ) -> Mapping[str, Any]:
        """Explicit-mode load of the first-step model; returns `{model, loaded_state}`."""
        return self._roundtrip(
            {"op": "load", "call_site": call_site, "scopes": list(scopes), "run_id": run_id}
        )

    def provider_op(
        self,
        call_site: str,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        scopes: Sequence[str] = (),
        run_id: str | None = None,
    ) -> Any:
        """A provider-specific operation (for example `lmstudio.models`) on the first local step
        of the call site's chain that offers it; returns the provider's JSON reply."""
        return self._roundtrip(
            {
                "op": "provider_op",
                "call_site": call_site,
                "name": name,
                "params": dict(params or {}),
                "scopes": list(scopes),
                "run_id": run_id,
            }
        )

    def embed(self, *args: object, **kwargs: object) -> None:
        """Always refused at first delivery (the service answers with the refusal)."""
        self._roundtrip({"op": "embed"})
