"""`BrokerClient` (C-1001 P1-29): the one module another component imports to reach the broker (I2).
Spec: ``docs/plans/connected-code-graph-egress-spec3.md`` section 2.1, 2.2, 2.5.

There is no direct-connection code here and no fallback: a broker that cannot be reached raises
`BrokerUnavailable`, a latched one (or any refusal of the call itself) raises `BrokerError`, and the caller fails
the operation with that reason. A denial is a normal `Decision`, not an exception.
"""
from __future__ import annotations

import base64
import os
import pwd
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ccg.broker import framing

STATE_REL = (".local", "state", "ccg", "broker")
CLIENT_TIMEOUT_S = framing.UPSTREAM_TIMEOUT_S + 60  # a reply waits for the whole upstream exchange


class BrokerUnavailable(RuntimeError):
    """The broker's socket cannot be reached (absent, refused, unreadable reply)."""


class BrokerError(RuntimeError):
    """The broker refused the call itself: ``code`` is ``audit-unwritable`` or ``open-run-invalid``."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


@dataclass(frozen=True)
class RunGrant:
    run_id: str
    token: str
    effective: dict[str, str]
    system_local_only: bool
    audit_id: str  # the `policy` record's auditId


@dataclass(frozen=True)
class Decision:
    decision: str
    rule: str
    reason: str | None
    audit_id: str

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    def refusal(self) -> str:
        return f"refused by egress policy ({self.rule}: {self.reason})"


@dataclass(frozen=True)
class Response:
    status: int
    content_type: str | None
    body: bytes


def default_socket_path() -> str:
    env = os.environ.get("CCG_BROKER_SOCKET")
    if env:
        return env
    return str(Path(pwd.getpwuid(os.getuid()).pw_dir).joinpath(*STATE_REL, "broker.sock"))


class BrokerClient:
    def __init__(self, socket_path: str | None = None, timeout: float = CLIENT_TIMEOUT_S):
        self.socket_path = socket_path or default_socket_path()
        self.timeout = timeout

    def _call(self, msg: dict) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(self.timeout)
            try:
                s.connect(self.socket_path)
                s.sendall(framing.encode(msg))
                reply = framing.read_frame(s.recv, framing.max_reply_frame())
            except (OSError, EOFError, framing.FrameError) as exc:
                raise BrokerUnavailable(f"broker at {self.socket_path}: {type(exc).__name__}: {exc}") from None
        finally:
            s.close()
        if not isinstance(reply, dict):
            raise BrokerUnavailable("broker sent a reply that is not an object")
        if reply.get("ok") is not True:
            raise BrokerError(str(reply.get("error")), str(reply.get("message", "")))
        return reply

    def open_run(self, run_kind: str, system_id: str, codebases: list[str] | str, override: str | None = None) -> RunGrant:
        g = self._call({"op": "open_run", "runKind": run_kind, "systemId": system_id, "codebases": codebases,
                        "override": override})["grant"]
        return RunGrant(g["runId"], g["token"], g["effective"], g["systemLocalOnly"], g["auditId"])

    def close_run(self, token: str) -> None:
        self._call({"op": "close_run", "token": token})

    def request(self, token: str, request: dict[str, Any], body: Any = None) -> tuple[Decision, Response | None]:
        r = self._call({"op": "request", "token": token, "request": request, "body": body})
        d = r["decision"]
        resp = r.get("response")
        return (Decision(d["decision"], d["rule"], d["reason"], d["auditId"]),
                Response(resp["status"], resp["contentType"], base64.b64decode(resp["bodyB64"])) if resp else None)
