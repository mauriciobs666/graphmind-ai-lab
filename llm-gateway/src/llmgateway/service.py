"""The gateway service: the engine behind a Unix socket (docs/plans/llm-gateway.md §3.1, §3.4,
§4 S14; FR-1, FR-8, FR-11, FR-17, FR-20, AC-5, AC-7).

A `systemd --user` unit runs `bin/service.py` from the root-owned code root. The service routes a
call through the same `Gateway` as the library, with the same configuration (read from the
owner-protected `gateway.toml`, the path never taken from the environment), scope policy and call
records. A local step goes out from this process (`DirectTransport`); a hosted step goes only to
the key holder (`HolderTransport`), signed with the `service` caller credential read at start.

Socket: `$XDG_RUNTIME_DIR/llmgateway/service.sock` in a 0700 directory, mode 0600, peers of
another uid closed. Protocol: `llmgateway/client.py` (the file components copy). Ops: `call`,
`load`, `provider_op` (first local step of the call site's chain that offers the op; not recorded,
it is no model call), `embed` (always refused).
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import signal
import socket
import stat
import struct
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llmgateway import framing
from llmgateway.client import MAX_FRAME_BYTES, PROTOCOL_VERSION, default_socket_path
from llmgateway.config import (
    PRODUCTION_TRUSTED_UIDS,
    ConfigSource,
    GatewayConfig,
    resolve_path,
)
from llmgateway.engine import AttemptsFailed, Gateway
from llmgateway.failures import (
    Condition,
    classify_condition,
    classify_exception,
)
from llmgateway.load import LoadResidency
from llmgateway.policy import ScopeMarkerSource, resolve
from llmgateway.providers import REGISTRY, ProviderRegistry
from llmgateway.records import RecordStore
from llmgateway.transport import (
    DirectTransport,
    HolderTransport,
    Transport,
    TransportFailure,
    WireRequest,
    WireResponse,
)
from llmgateway.types import (
    CallFailed,
    EmbeddingsNotAvailable,
    FailureClass,
    GatewayError,
    HostedFromCallerProcess,
    Request,
    Response,
)

SERVICE_CALLER = "service"
READ_TIMEOUT_S = 30.0
MAX_CONNECTIONS = 32
_U32 = struct.Struct(">I")


class ServiceStartError(Exception):
    """The service must not start (socket directory not private, another service running)."""


# --- the caller credential ---


def read_credential(path: str, uid: int | None = None) -> bytes:
    """The `service` credential: `O_NOFOLLOW`, a regular file owned by `uid`, no group/other bits,
    one link, hex of 32 bytes. These checks guard against accident; the file is in the caller's
    own account, so they are not a defence against a same-uid process (plan §1, FR-8)."""
    uid = os.getuid() if uid is None else uid
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != uid or st.st_nlink != 1:
            raise ValueError("credential file is not a private regular file")
        if st.st_mode & 0o077:
            raise ValueError("credential file is readable by group or others")
        text = os.read(fd, 4096).decode("ascii").strip()
    finally:
        os.close(fd)
    secret = bytes.fromhex(text)
    if len(secret) != 32:
        raise ValueError("credential is not 32 bytes")
    return secret


def default_credential_path() -> str:
    home = pwd.getpwuid(os.getuid()).pw_dir
    return os.path.join(home, ".config", "llmgateway", "callers", SERVICE_CALLER + ".cred")


def default_state_dir() -> str:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        pwd.getpwuid(os.getuid()).pw_dir, ".local", "state"
    )
    return os.path.join(base, "llmgateway")


# --- routing ---


class RoutingTransport:
    """A local step goes out from this process; a hosted step only to the key holder. With no
    credential a hosted step fails `config` (never falls back to a direct connection)."""

    reaches_hosted = True

    def __init__(self, local: Transport, holder: Transport | None, why_no_holder: str = "") -> None:
        self._local, self._holder, self._why = local, holder, why_no_holder

    def exchange(self, request: WireRequest) -> WireResponse:
        if not request.hosted:
            return self._local.exchange(request)
        if self._holder is None:
            raise CallFailed(
                FailureClass.CONFIG, f"hosted calls are unavailable: {self._why or 'no credential'}"
            )
        return self._holder.exchange(request)


# --- the service ---


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    config_source: ConfigSource
    socket_path: str
    state_dir: str | None = None  # None: `[deployment] state_dir`, else the XDG state default
    credential_path: str | None = None  # None: the caller's default; a missing file: no hosted
    read_timeout: float = READ_TIMEOUT_S
    max_frame_bytes: int = MAX_FRAME_BYTES
    max_connections: int = MAX_CONNECTIONS
    marker_source: ScopeMarkerSource | None = None
    registry: ProviderRegistry = REGISTRY
    own_uid: int | None = None  # tests only
    holder_own_uid: int | None = None  # tests only: the uid `HolderTransport` must differ from


def _failure(
    failure_class: FailureClass,
    message: str,
    *,
    kind: str = "CallFailed",
    code: str | None = None,
    call_id: str | None = None,
    classes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": kind,
            "class": failure_class.value,
            "message": message,
            "code": code,
            "call_id": call_id,
            "classes": list(classes),
        },
    }


def _refusal(message: str) -> dict[str, Any]:
    return _failure(classify_condition(Condition.SERVICE_REFUSAL), message)


def _response_dict(resp: Response) -> dict[str, Any]:
    return {
        "call_id": resp.call_id,
        "provider": resp.provider,
        "model": resp.model,
        "hosted": resp.hosted,
        "message": resp.message,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
        "provider_detail": resp.provider_detail,
        "raw": resp.raw,
    }


class Service:
    def __init__(self, settings: ServiceSettings) -> None:
        self.s = settings
        self._uid = os.getuid() if settings.own_uid is None else settings.own_uid
        config = settings.config_source.get()
        deployed = config.deployment.get("state_dir") if config.protected else None
        state_dir = settings.state_dir or (deployed if isinstance(deployed, str) else None)
        self.records = RecordStore(state_dir or default_state_dir())
        local = DirectTransport(settings.config_source.get)
        holder, why = self._holder_transport(settings)
        self.transport = RoutingTransport(local, holder, why)
        self.gateway = Gateway(
            settings.config_source,
            self.transport,
            self.records,
            registry=settings.registry,
            marker_source=settings.marker_source,
            residency=LoadResidency(self.transport),
        )
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._slots = threading.BoundedSemaphore(settings.max_connections)

    def _holder_transport(self, settings: ServiceSettings) -> tuple[HolderTransport | None, str]:
        path = settings.credential_path or default_credential_path()
        try:
            secret = read_credential(path, self._uid)
        except (OSError, ValueError) as exc:
            reason = f"no usable caller credential ({type(exc).__name__})"
            print(f"llmgateway service: hosted calls disabled: {reason}", file=sys.stderr)
            return None, reason
        return (
            HolderTransport(
                settings.config_source.get,
                secret,
                caller=SERVICE_CALLER,
                own_uid=self._uid if settings.holder_own_uid is None else settings.holder_own_uid,
            ),
            "",
        )

    # -- ops ---------------------------------------------------------------------------------

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """One decoded request -> one reply object. Never raises."""
        try:
            if message.get("v") != PROTOCOL_VERSION:
                return _refusal(f"unsupported protocol version {message.get('v')!r}")
            op = message.get("op")
            if op == "call":
                return self._call(message)
            if op == "load":
                return self._load(message)
            if op == "provider_op":
                return self._provider_op(message)
            if op == "embed":
                return self._embed()
            return _refusal(f"unknown op {op!r}")
        except HostedFromCallerProcess as exc:
            return _failure(
                classify_condition(Condition.SERVICE_REFUSAL),
                str(exc),
                kind="HostedFromCallerProcess",
            )
        except EmbeddingsNotAvailable as exc:
            return _failure(
                classify_condition(Condition.SERVICE_REFUSAL),
                str(exc),
                kind="EmbeddingsNotAvailable",
            )
        except AttemptsFailed as exc:
            return _failure(
                exc.failure_class,
                str(exc),
                code=exc.code,
                call_id=exc.call_id,
                classes=[c.value for c in exc.classes],
            )
        except CallFailed as exc:
            return _failure(exc.failure_class, str(exc), code=exc.code)
        except GatewayError as exc:  # ConfigError, RecordError: the owner's or the host's fault
            return _failure(FailureClass.CONFIG, f"{type(exc).__name__}: {exc}")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return _refusal(f"bad request: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - never a traceback to the peer
            self._diag(f"internal error {type(exc).__name__}")
            return _failure(FailureClass.CONFIG, f"internal error ({type(exc).__name__})")

    def _call(self, message: Mapping[str, Any]) -> dict[str, Any]:
        payload = message.get("request")
        if not isinstance(payload, Mapping):
            raise ValueError("request must be an object")
        return {"ok": True, "result": _response_dict(self.gateway.call(Request.from_dict(payload)))}

    def _load(self, message: Mapping[str, Any]) -> dict[str, Any]:
        site, scopes, run_id = self._common(message)
        prepared = self.gateway.load(site, scopes=scopes, run_id=run_id)
        return {
            "ok": True,
            "result": {"model": prepared.model, "loaded_state": prepared.loaded_state},
        }

    @staticmethod
    def _common(message: Mapping[str, Any]) -> tuple[str, tuple[str, ...], str | None]:
        site, scopes, run_id = (
            message.get("call_site"),
            message.get("scopes", []),
            message.get("run_id"),
        )
        if not isinstance(site, str) or not isinstance(scopes, list):
            raise ValueError("call_site must be a string and scopes a list")
        return site, tuple(scopes), run_id

    def _embed(self) -> dict[str, Any]:
        self.gateway.embed()
        raise AssertionError("unreachable")  # pragma: no cover - embed always raises

    def _provider_op(self, message: Mapping[str, Any]) -> dict[str, Any]:
        site, scopes, run_id = self._common(message)
        name, params = message.get("name"), message.get("params", {})
        if not isinstance(name, str) or not isinstance(params, Mapping):
            raise ValueError("name must be a string and params an object")
        config = self.s.config_source.get()
        # the same policy as a call: the scope rules apply to the call site's chain
        resolution = resolve(
            config, Request(site, (), scopes=scopes, run_id=run_id), self.s.marker_source
        )
        for step in resolution.chain:
            if step.hosted:
                continue
            pconf = config.providers[step.provider]
            provider = self.s.registry.get(str(pconf.get("kind", "")))
            if name in provider.ops:
                return self._run_op(
                    config, step.provider, pconf, provider, name, params, resolution
                )
        raise CallFailed(
            classify_condition(Condition.SERVICE_REFUSAL),
            f"no local step of call site {site!r} offers operation {name!r}",
        )

    def _run_op(
        self,
        config: GatewayConfig,
        provider_id: str,
        pconf: Mapping[str, Any],
        provider: Any,
        name: str,
        params: Mapping[str, Any],
        resolution: Any,
    ) -> dict[str, Any]:
        built = provider.ops[name](dict(params))
        timeout = pconf.get("timeout_s", 120.0)
        wire = WireRequest(
            provider_id=provider_id,
            kind=provider.kind,
            base_url=str(pconf.get("base_url", "")),
            path=built.path,
            body=built.body,
            hosted=config.provider_is_hosted(provider_id),
            method=built.method,
            headers=built.headers,
            component=resolution.component,
            scopes=resolution.scopes,
            timeout_s=float(timeout) if isinstance(timeout, (int, float)) else 120.0,
        )
        try:
            reply = self.transport.exchange(wire)
        except (HostedFromCallerProcess, CallFailed):
            raise
        except TransportFailure as exc:
            raise CallFailed(classify_condition(exc.condition), f"{provider_id}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - classify_exception owns the mapping
            raise CallFailed(
                classify_exception(exc), f"{provider_id}: {type(exc).__name__}"
            ) from exc
        if not 200 <= reply.status < 300:
            fc = (
                FailureClass.REJECTED
                if reply.status < 400
                else provider.classify(reply.status, reply.content_type, reply.body, pconf)
            )
            raise CallFailed(fc, f"{provider_id}: {fc.value}, HTTP {reply.status} on {built.path}")
        try:
            body: Any = json.loads(reply.body)
        except ValueError:
            body = reply.body.decode("utf-8", errors="replace")
        return {"ok": True, "result": {"status": reply.status, "body": body}}

    # -- the socket --------------------------------------------------------------------------

    def _diag(self, text: str) -> None:
        print(f"llmgateway service: {text}", file=sys.stderr, flush=True)

    def bind(self) -> None:
        path = self.s.socket_path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, mode=0o700, exist_ok=True)
        st = os.lstat(directory)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != self._uid or st.st_mode & 0o077:
            raise ServiceStartError("socket directory is not private (0700) to this account")
        self._clear_stale(path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            listener.bind(path)
        finally:
            os.umask(old)
        listener.listen(64)
        listener.settimeout(0.2)
        self._listener = listener

    def _clear_stale(self, path: str) -> None:
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(st.st_mode) or st.st_uid != self._uid:
            raise ServiceStartError("the socket path is occupied by something that is not ours")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(1.0)
        try:
            probe.connect(path)
        except OSError:
            os.unlink(path)  # nothing listens: a stale socket from a dead service
            return
        finally:
            probe.close()
        raise ServiceStartError("another service is already listening on this socket")

    def serve_forever(self) -> None:
        assert self._listener is not None, "bind() first"
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            if not self._slots.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(target=self._serve_one, args=(conn,), daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()
        if self._listener is not None:
            self._listener.close()
            try:
                os.unlink(self.s.socket_path)
            except OSError:
                pass

    def _peer_ok(self, conn: socket.socket) -> bool:
        try:
            raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            _pid, uid, _gid = struct.unpack("3i", raw)
        except OSError:
            return False
        return uid == self._uid

    def _serve_one(self, conn: socket.socket) -> None:
        try:
            if not self._peer_ok(conn):
                return
            try:
                payload = framing.read_frame(
                    conn, max_bytes=self.s.max_frame_bytes, timeout=self.s.read_timeout
                )
                message = framing.strict_loads(payload)
            except (framing.FrameError, ValueError):
                return
            reply = (
                self.handle(message)
                if isinstance(message, dict)
                else _refusal("request is not an object")
            )
            data = json.dumps(reply, separators=(",", ":")).encode("utf-8")
            conn.settimeout(self.s.read_timeout)
            conn.sendall(_U32.pack(len(data)) + data)
        except OSError:
            pass
        except Exception as exc:  # noqa: BLE001 - a connection thread never dies loudly
            self._diag(f"connection error {type(exc).__name__}")
        finally:
            conn.close()
            self._slots.release()


# --- entry ---


def main(
    argv: Sequence[str] | None = None,
    *,
    config_source_factory: Callable[[], ConfigSource] | None = None,
) -> int:
    """`bin/service.py`'s entry. The configuration is always `/etc/llmgateway/gateway.toml`
    under root ownership (`resolve_path(library=False)`: the environment is ignored); a caller
    cannot name another file. `config_source_factory` is for tests only."""
    parser = argparse.ArgumentParser(prog="service.py", description="LLM gateway service")
    parser.add_argument("--socket", default=None, help="socket path (default: user runtime dir)")
    parser.add_argument("--credential", default=None, help="caller credential file")
    parser.add_argument("--state-dir", default=None, help="call-record directory")
    args = parser.parse_args(argv)
    source = (
        config_source_factory()
        if config_source_factory is not None
        else ConfigSource(resolve_path(library=False), PRODUCTION_TRUSTED_UIDS)
    )
    try:
        service = Service(
            ServiceSettings(
                config_source=source,
                socket_path=args.socket or default_socket_path(),
                state_dir=args.state_dir,
                credential_path=args.credential,
            )
        )
        service.bind()
    except (ServiceStartError, OSError) as exc:
        print(f"llmgateway service: {exc}", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, lambda *_: service.stop())
    signal.signal(signal.SIGINT, lambda *_: service.stop())
    print("llmgateway service: ready", file=sys.stderr, flush=True)
    service.serve_forever()
    service.close()
    return 0
