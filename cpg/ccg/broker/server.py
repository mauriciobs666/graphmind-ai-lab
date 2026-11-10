"""``ccg-broker``: the egress broker process (C-1001 P1-29). Spec: ``docs/plans/connected-code-graph-egress-spec3.md``
section 2.0-2.6, 3, 5.

    python -m ccg.broker.server --system <manifest.yaml> [--system ...] [--socket <path>]

Environment: ``CCG_AUDIT_DIR`` only. Exit codes: 0 clean stop, 64 usage or configuration error, 69 precondition
failure (lock held, live socket, directory or audit-file check). `Broker` is the socket-free core (``handle`` maps one
request message to one reply); `BrokerServer` is the Unix-socket listener; `main` is the process.
"""
# The closed-environment re-exec must run before anything else is imported, so a planted module in the working
# directory or on PYTHONPATH can never be imported by the broker (section 2.0). Hence the early block, and E402 below.
import os
import pwd
import sys

STATE_REL = (".local", "state", "ccg", "broker")
_MODULE = "ccg.broker.server"


def passwd_home() -> str:
    return pwd.getpwuid(os.getuid()).pw_dir


def closed_environment(audit_dir: "str | None") -> "dict[str, str]":
    env = {"PATH": "/usr/bin:/bin", "HOME": passwd_home(), "LANG": "C.UTF-8"}
    if audit_dir is not None:
        env["CCG_AUDIT_DIR"] = audit_dir
    return env


def reexec_if_needed(argv, environ=None, *, execve=os.execve, executable=None, isolated=None) -> bool:
    """Re-execute as ``<sys.executable> -I -m ccg.broker.server ...`` under the closed environment; skipped only when
    the environment and the flags are already exactly that. Returns False when it skipped (``execve`` normally never
    returns)."""
    environ = os.environ if environ is None else environ
    executable = executable or sys.executable
    isolated = sys.flags.isolated if isolated is None else isolated
    want = closed_environment(environ.get("CCG_AUDIT_DIR"))
    if isolated and dict(environ) == want:
        return False
    execve(executable, [executable, "-I", "-m", _MODULE, *argv], want)
    return True


if __name__ == "__main__":
    reexec_if_needed(sys.argv[1:])

# ruff: noqa: E402
import argparse
import base64
import fcntl
import hashlib
import signal
import socket
import stat
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from ccg import manifest as manifest_mod
from ccg.broker import addr as addr_mod
from ccg.broker import audit as audit_mod
from ccg.broker import framing, policy, structured
from ccg.broker.audit import AuditUnwritable, AuditWriter, PreconditionError
from ccg.broker.runs import RunTable

EXIT_OK, EXIT_USAGE, EXIT_PRECONDITION = 0, 64, 69
READ_TIMEOUT_S = 30


def _default_log(msg: str) -> None:
    print(f"ccg-broker: {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ the core


class Broker:
    """Run table, rule evaluation, the upstream exchange and the audit trail. No sockets: ``handle`` maps one
    decoded message to one reply, so tests drive it with injected resolver, interface reader, TLS factory and clock."""

    def __init__(self, systems: dict[str, policy.LoadedSystem], audit: AuditWriter, *,
                 resolver: Callable[[str], list[str]] = addr_mod.default_resolver,
                 iface: Callable[[], addr_mod.IfaceData] = addr_mod.read_interfaces,
                 tls_factory: Callable[[], Any] = structured.make_tls_context,
                 clock: Callable[[], float] = time.monotonic,
                 exchanger: Callable[..., structured.Exchange] = structured.exchange,
                 log: Callable[[str], None] = _default_log):
        self.systems, self.audit = systems, audit
        self.resolver, self.iface, self.tls_factory = resolver, iface, tls_factory
        self.clock, self.exchanger, self.log = clock, exchanger, log
        self.runs = RunTable(clock)

    # -- dispatch

    def handle(self, msg: Any) -> dict:
        """The reply to one message. Raises `framing.FrameError` for a message that is not a known operation."""
        if not isinstance(msg, dict):
            raise framing.FrameError("message is not an object")
        op = msg.get("op")
        if op == "open_run":
            return self.open_run(msg)
        if op == "request":
            return self.request(msg)
        if op == "close_run":
            self.runs.close(msg.get("token"))
            return {"ok": True}
        raise framing.FrameError(f"unknown op {framing.clip(op)}")

    @staticmethod
    def _err(code: str, message: str = "") -> dict:
        return {"ok": False, "error": code, "message": message}

    # -- open_run (section 2.1)

    def open_run(self, msg: dict) -> dict:
        if self.audit.latched:
            return self._err("audit-unwritable", "the broker's audit writer is latched; restart it")
        kind, sid, cbs, override = msg.get("runKind"), msg.get("systemId"), msg.get("codebases"), msg.get("override")
        system = self.systems.get(sid) if isinstance(sid, str) else None
        problem = None
        if kind not in policy.RUN_KINDS:
            problem = "runKind"
        elif system is None:
            problem = "systemId"
        elif override not in (None, "local", "hosted"):
            problem = "override"
        elif kind == "vulndb":
            if cbs != []:
                problem = "codebases (a vulndb run has none)"
        elif cbs == "*":
            cbs = list(system.modes)
        elif (not isinstance(cbs, list) or not cbs or not all(isinstance(c, str) for c in cbs)
              or len(set(cbs)) != len(cbs) or not set(cbs) <= set(system.modes)):
            problem = "codebases"
        if problem:
            self.log(f"open-run-invalid: {problem}")
            return self._err("open-run-invalid", problem)
        rows, refused, local_only = policy.resolve_modes(system, cbs, override)
        run_id = self.runs.new_run_id()
        rec = self.audit.record(
            "policy", runId=run_id, runKind=kind, systemId=system.system, codebases=list(cbs), override=override,
            modes=rows, overrideRefused=refused, systemLocalOnly=local_only, allowListDigest=system.digest,
            auditDir=str(self.audit.audit_dir))
        try:
            self.audit.append(rec)
        except AuditUnwritable as exc:
            return self._err("audit-unwritable", str(exc))
        effective = {r["repoId"]: r["effective"] for r in rows}
        token, run = self.runs.create(run_id, kind, system.system, cbs, effective, local_only)
        return {"ok": True, "grant": {"runId": run_id, "token": token, "effective": effective,
                                      "systemLocalOnly": local_only, "auditId": rec["auditId"]}}

    # -- request (section 2.2, 2.3, 3.3, 3.4)

    def request(self, msg: dict) -> dict:
        if self.audit.latched:
            return self._err("audit-unwritable", "the broker's audit writer is latched; restart it")
        run = self.runs.lookup(msg.get("token"))
        system = self.systems[run.system_id] if run else next(iter(self.systems.values()))
        repo_count = len(system.modes) if run else max(len(s.modes) for s in self.systems.values())
        clean, rejected = policy.sanitize(msg.get("request"), repo_count)
        body = msg.get("body")
        try:
            body_bytes = structured.serialize_body(body) if body is not None else None
        except (UnicodeEncodeError, ValueError):  # a body the frame parser would have refused: no record
            raise framing.FrameError("body is not serializable as UTF-8 JSON") from None
        verdict = policy.evaluate(system, run, clean, body, body_bytes)
        validated = None
        if verdict.decision == "allow":
            try:
                validated = addr_mod.validate(system.egress, verdict.section, verdict.entry,
                                              resolver=self.resolver, iface=self.iface)
            except addr_mod.AddressDenied as exc:
                verdict = policy.Verdict("R0-INVALID", "deny", exc.reason, verdict.mode)
        allow = verdict.decision == "allow"
        fields = structured.payload_fields(body, clean["payloadClass"]) if allow and body is not None else (
            [] if allow else None)
        if fields and any(len(f) > structured.MAX_FIELD_CHARS for f in fields):
            # a body key that does not fit the record's field: null, with the digest of the list (section 5.1)
            rejected = {**rejected, "payloadFields": policy._digest(fields)}
            fields = None
        dec = self.audit.record(
            "decision", runId=run.run_id if run else None, component=clean["component"], purpose=clean["purpose"],
            payloadClass=clean["payloadClass"], target={"method": clean["method"], "path": clean["path"]},
            endpoint=clean["endpoint"], codebases=clean["codebases"], rejected=rejected,
            resolvedAddr=validated.addr if allow else None, privateAddress=bool(allow and validated.private),
            requestMode=verdict.mode, decision=verdict.decision, rule=verdict.rule, reason=verdict.reason,
            transport="structured",
            payloadFields=fields,
            bytes=len(body_bytes) if body_bytes is not None else 0,
            bodySha256=hashlib.sha256(body_bytes).hexdigest() if body_bytes is not None else None)
        try:
            self.audit.append(dec)  # durable before any upstream byte
        except AuditUnwritable as exc:
            return self._err("audit-unwritable", str(exc))
        decision = {"decision": verdict.decision, "rule": verdict.rule, "reason": verdict.reason,
                    "auditId": dec["auditId"]}
        if not allow:
            return {"ok": True, "decision": decision, "response": None}
        return self._upstream(run, clean, verdict, validated, body_bytes, decision)

    def _upstream(self, run, clean, verdict, validated, body_bytes, decision) -> dict:
        parts = urlsplit(verdict.entry["endpoint"])
        path = next(p for p in verdict.entry["paths"] if p == clean["path"])  # the entry's string, never the caller's
        t0 = self.clock()
        try:  # an allow is durable: whatever fails from here on still ends in a `close` record (section 5.2)
            ctx = self.tls_factory() if parts.scheme == "https" else None
            ex = self.exchanger(parts.scheme, parts.hostname, parts.port, validated.addr, clean["method"], path,
                                body_bytes, tls_context=ctx)
        except Exception as exc:
            self.log(f"upstream setup failed: {type(exc).__name__}")
            ex = structured.Exchange("upstream-error", None, None, None, 0, 0)
        close = self.audit.record(
            "close", decisionAuditId=decision["auditId"], runId=run.run_id, bytesOut=ex.bytes_out,
            bytesIn=ex.bytes_in, httpStatus=ex.status, durationMs=int((self.clock() - t0) * 1000),
            closeReason=ex.close_reason)
        try:
            self.audit.append(close)
        except AuditUnwritable:
            pass  # the writer is latched: the response still returns (section 2.5), later calls are refused
        response = None
        if ex.close_reason == "completed":
            response = {"status": ex.status, "contentType": ex.content_type,
                        "bodyB64": base64.b64encode(ex.body).decode("ascii")}
        return {"ok": True, "decision": decision, "response": response, "closeReason": ex.close_reason}


# ------------------------------------------------------------------ the listener


def so_peercred_uid(conn: socket.socket) -> int:
    _pid, uid, _gid = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    return uid


class BrokerServer:
    """One Unix-domain stream socket and nothing else; one request and one reply per connection."""

    def __init__(self, broker: Broker, sock_path: Path, *, peer_uid: Callable[[socket.socket], int] = so_peercred_uid,
                 log: Callable[[str], None] = _default_log):
        self.broker, self.sock_path, self.peer_uid, self.log = broker, Path(sock_path), peer_uid, log
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.sock = self._bind()

    def _bind(self) -> socket.socket:
        p = str(self.sock_path)
        if os.path.lexists(p):
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(p)
            except OSError:
                if not stat.S_ISSOCK(os.lstat(p).st_mode):
                    raise PreconditionError(f"{p} exists and is not a socket") from None
                os.unlink(p)  # nothing answered: a leftover file
            else:
                raise PreconditionError(f"a broker is already listening on {p}")
            finally:
                probe.close()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            s.bind(p)
        finally:
            os.umask(old)
        s.listen(16)
        s.settimeout(0.2)
        return s

    def serve_forever(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=self._serve_one, args=(conn,), daemon=True)
            t.start()
            self._threads = [x for x in self._threads if x.is_alive()] + [t]

    def _serve_one(self, conn: socket.socket) -> None:
        with conn:
            try:
                if self.peer_uid(conn) != os.getuid():
                    self.log("closed a connection from another uid")  # no reply, no record
                    return
                conn.settimeout(READ_TIMEOUT_S)
                try:
                    msg = framing.read_frame(conn.recv, framing.max_request_frame())
                except EOFError:
                    return
                except (framing.FrameError, OSError) as exc:
                    self.log(f"malformed frame: {str(exc)[:160]}")
                    return
                try:
                    reply = self.broker.handle(msg)
                except framing.FrameError as exc:
                    self.log(f"malformed message: {str(exc)[:160]}")
                    return
                except Exception as exc:  # a handler defect: visible in the log, the connection closes, no reply
                    self.log(f"handler failed: {type(exc).__name__}")
                    return
                conn.settimeout(None)
                try:
                    conn.sendall(framing.encode(reply, framing.max_reply_frame()))
                except framing.FrameError as exc:
                    self.log(f"reply too large: {exc}")
            except Exception as exc:  # never let a connection thread die silently
                self.log(f"connection error: {type(exc).__name__}")

    def shutdown(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
        try:
            self.sock.close()
        finally:
            try:
                os.unlink(self.sock_path)
            except OSError:
                pass


# ------------------------------------------------------------------ the process


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # argparse's own exit code is 2
        raise SystemExit(self._usage_error(message))

    @staticmethod
    def _usage_error(message: str) -> int:
        print(f"ccg-broker: usage error: {message}", file=sys.stderr)
        return EXIT_USAGE


def load_systems(paths: list[str], iface: Callable[[], addr_mod.IfaceData]) -> dict[str, policy.LoadedSystem]:
    """Each manifest through ``load_manifest`` (S1-S5), then S6; `manifest_mod.ManifestError` names the file."""
    systems: dict[str, policy.LoadedSystem] = {}
    for p in paths:
        try:
            m = manifest_mod.load_manifest(p)
            bad = addr_mod.check_local_entries(m.egress, iface)
            if bad:
                raise manifest_mod.ManifestError("; ".join(bad))
            ls = policy.LoadedSystem.from_manifest(m)
        except addr_mod.IfaceUnavailable as exc:
            raise manifest_mod.ManifestError(f"{p}: S6 needs the interface data: {exc}") from None
        except (manifest_mod.ManifestError, OSError) as exc:
            raise manifest_mod.ManifestError(f"{p}: {exc}") from None
        if ls.system in systems:
            raise manifest_mod.ManifestError(f"{p}: system {ls.system!r} is loaded twice")
        systems[ls.system] = ls
    return systems


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = _Parser(prog="ccg-broker", description=__doc__.split("\n\n")[0])
    ap.add_argument("--system", action="append", required=True, metavar="MANIFEST")
    ap.add_argument("--socket", metavar="PATH")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        systems = load_systems(args.system, addr_mod.read_interfaces)
    except manifest_mod.ManifestError as exc:
        print(f"ccg-broker: configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    lock_fd = -1
    try:
        bdir = Path(args.socket).parent if args.socket else Path(passwd_home()).joinpath(*STATE_REL)
        sock_path = Path(args.socket) if args.socket else bdir / "broker.sock"
        audit_mod.require_dir(bdir, exact_mode=0o700, create=True, check_ancestors=True)  # strict: no sticky exemption
        try:
            lock_fd = os.open(bdir / "broker.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        except OSError as exc:
            raise PreconditionError(f"{bdir / 'broker.lock'}: {exc.strerror}") from None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise PreconditionError(f"another broker holds {bdir / 'broker.lock'}") from None
        audit = AuditWriter(os.environ.get("CCG_AUDIT_DIR") or audit_mod.DEFAULT_AUDIT_DIR)
        server = BrokerServer(Broker(systems, audit), sock_path)
    except PreconditionError as exc:
        print(f"ccg-broker: precondition failed: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    signal.signal(signal.SIGTERM, lambda *_: server._stop.set())
    signal.signal(signal.SIGINT, lambda *_: server._stop.set())
    print(f"ccg-broker: listening on {sock_path}; audit {audit.audit_dir}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    finally:
        server.shutdown()
        audit.close()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
