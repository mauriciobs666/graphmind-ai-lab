"""Shared fixtures for the P1-29 broker tests (`test_broker_policy.py`, `test_broker_audit.py`,
`test_broker_server.py`): manifests, a fake interface reader and resolver, a harness around `Broker`, and a raw
upstream test server. Expected values live in the tests, written as literals; nothing here computes one."""
from __future__ import annotations

import copy
import ipaddress
import socket
import threading
from pathlib import Path

import yaml

from ccg import manifest as mf
from ccg.broker import addr as addr_mod
from ccg.broker import policy, structured
from ccg.broker.audit import AuditWriter, read_records
from ccg.broker.server import Broker

HOST_ADDR = "192.168.0.64"
GATEWAYS = ("192.168.0.1", "fe80::1")
BRIDGE_NET = "172.17.0.0/16"


def fake_iface(addresses=(HOST_ADDR, "127.0.0.1", "::1"), bridges=(BRIDGE_NET,), gateways=GATEWAYS):
    ip = ipaddress.ip_address
    data = addr_mod.IfaceData(tuple(ip(a) for a in addresses), tuple(ipaddress.ip_network(b) for b in bridges),
                              tuple(ip(g) for g in gateways))
    return lambda: data


def egress(port: int = 1234, local_host: str = "127.0.0.1") -> dict:
    return {
        "schemaVersion": 1,
        "local": [{"id": "lmstudio", "kind": "model", "endpoint": f"http://{local_host}:{port}",
                   "paths": ["/v1/chat/completions"]}],
        "outbound": [
            {"id": "vulndb", "kind": "vulndb", "endpoint": "https://vulndb.example.com:443", "paths": ["/feed"]},
            {"id": "idp", "kind": "idp", "endpoint": "https://idp.example.com:443", "paths": ["/who"]},
            {"id": "acl", "kind": "access-check", "endpoint": "https://acl.example.com:443", "paths": ["/check"]},
            {"id": "prs", "kind": "pr-addon", "endpoint": "https://prs.example.com:443", "paths": ["/pr"]},
        ],
        "hosted": [{"id": "hosted", "kind": "model", "endpoint": "https://api.hosted.example.com:443",
                    "paths": ["/v1/chat/completions"]}],
    }


def write_manifest(tmp_path: Path, *, repos=(("front", "local-only"), ("docs", "hosted"), ("lib", "hosted")),
                   eg: dict | None = None, system: str = "shop", name: str = "system.yaml") -> Path:
    doc = {"schemaVersion": 1, "system": system, "egress": eg if eg is not None else egress(),
           "repos": [{"repoId": r, "path": ".", "policy": {"egress": m}} for r, m in repos]}
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc))
    return p


def load_system(tmp_path: Path, **kw) -> policy.LoadedSystem:
    return policy.LoadedSystem.from_manifest(mf.load_manifest(write_manifest(tmp_path, **kw)))


class Resolver:
    """Records every call; answers from ``table`` (name -> list of addresses) or raises for an unknown name."""

    def __init__(self, table=None):
        self.table = {"api.hosted.example.com.": ["93.184.216.34"], "vulndb.example.com.": ["93.184.216.35"],
                      "idp.example.com.": ["93.184.216.36"], "acl.example.com.": ["93.184.216.37"],
                      "prs.example.com.": ["93.184.216.38"], **(table or {})}
        self.calls: list[str] = []

    def __call__(self, name: str):
        self.calls.append(name)
        if name not in self.table:
            raise OSError("no such host")
        return list(self.table[name])


class FakeExchange:
    """Stands in for `structured.exchange`: records its arguments, returns a canned completed response."""

    def __init__(self, result: structured.Exchange | None = None):
        self.calls: list[dict] = []
        self.result = result or structured.Exchange("completed", 200, "application/json", b'{"ok":1}', 100, 200)

    def __call__(self, scheme, host, port, addr, method, path, body, **kw):
        self.calls.append({"scheme": scheme, "host": host, "port": port, "addr": addr, "method": method,
                           "path": path, "body": body, **kw})
        return self.result


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class Harness:
    def __init__(self, tmp_path: Path, systems=None, **kw):
        self.tmp = tmp_path
        self.systems = systems or {"shop": load_system(tmp_path)}
        self.audit_dir = tmp_path / "audit"
        self.audit = AuditWriter(self.audit_dir, log=lambda m: self.logs.append(m))
        self.logs: list[str] = []
        self.resolver = kw.pop("resolver", None) or Resolver()
        self.exchange = kw.pop("exchanger", None) or FakeExchange()
        self.clock = kw.pop("clock", None) or Clock()
        self.broker = Broker(self.systems, self.audit, resolver=self.resolver, iface=kw.pop("iface", fake_iface()),
                             tls_factory=lambda: "tls", clock=self.clock, exchanger=self.exchange,
                             log=lambda m: self.logs.append(m), **kw)

    def records(self, kind: str | None = None) -> list[dict]:
        recs = list(read_records(self.audit_dir / "egress.jsonl")) if (self.audit_dir / "egress.jsonl").exists() else []
        return [r for r in recs if kind is None or r["recordType"] == kind]

    def open(self, kind="answer", codebases="*", override=None, system="shop") -> dict:
        r = self.broker.handle({"op": "open_run", "runKind": kind, "systemId": system, "codebases": codebases,
                                "override": override})
        assert r["ok"], r
        return r["grant"]

    def ask(self, token, request: dict, body=None) -> dict:
        r = self.broker.handle({"op": "request", "token": token, "request": request, "body": body})
        assert r["ok"], r
        return r

    def last_decision(self) -> dict:
        return self.records("decision")[-1]


def model_call(**over) -> dict:
    """An answer-run request: answer.interpret / model-call to the local LM Studio entry."""
    r = {"component": "answer.interpret", "purpose": "model-call", "payloadClass": "repo-derived",
         "endpoint": "http://127.0.0.1:1234", "target": {"method": "POST", "path": "/v1/chat/completions"},
         "codebases": ["front", "docs", "lib"]}
    r.update(over)
    return r


def hosted_call(**over) -> dict:
    return model_call(**{"endpoint": "https://api.hosted.example.com:443", **over})


def deepcopy(x):
    return copy.deepcopy(x)


# ------------------------------------------------------------------ a raw upstream test server


class Upstream:
    """Accepts connections on 127.0.0.1, records the raw request bytes, and answers with ``reply(conn)``."""

    def __init__(self, reply):
        self.reply = reply
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.requests: list[bytes] = []
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn):
        with conn:
            conn.settimeout(10)
            data = b""
            try:
                while b"\r\n\r\n" not in data:
                    c = conn.recv(4096)
                    if not c:
                        return
                    data += c
                head, _, rest = data.partition(b"\r\n\r\n")
                n = 0
                for ln in head.split(b"\r\n")[1:]:
                    if ln.lower().startswith(b"content-length:"):
                        n = int(ln.split(b":")[1])
                while len(rest) < n:
                    c = conn.recv(4096)
                    if not c:
                        break
                    rest += c
                self.requests.append(head + b"\r\n\r\n" + rest)
                self.reply(conn)
            except OSError:
                return

    def close(self):
        self.sock.close()


def http_reply(status=200, body=b'{"ok":true}', headers=(("Content-Type", "application/json"),)):
    def reply(conn):
        h = "".join(f"{k}: {v}\r\n" for k, v in headers)
        conn.sendall(f"HTTP/1.1 {status} X\r\n{h}Content-Length: {len(body)}\r\n\r\n".encode() + body)
    return reply
