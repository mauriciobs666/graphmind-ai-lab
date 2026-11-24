"""Shared fixtures for the key-holder suites: a temp installed layout, a logging HTTPS stand-in
provider with a self-signed cert, an in-process holder, and a real holder subprocess.

Nothing here holds a real key: `MARKER_KEY` is a planted fake whose only job is to be searched for.
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import ssl
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llmgateway import framing
from llmgateway.config import ConfigSource
from llmgateway.keyholder import Holder, HolderSettings

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
LAUNCHER = pathlib.Path(__file__).resolve().parent / "holder_launcher.py"
MODEL = "claude-sonnet-4-5-20250929"
UID = os.getuid()
CALLER_SECRET = bytes(range(32))
OTHER_SECRET = bytes(range(1, 33))


def make_cert(directory: pathlib.Path) -> tuple[str, str]:
    cert, key = directory / "server.pem", directory / "server.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "2",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return str(cert), str(key)


# --- stand-in provider ---


@dataclass
class Seen:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


def anthropic_ok(tokens_in=11, tokens_out=7, cache_c=3, cache_r=5):
    return (
        200,
        {"Content-Type": "application/json"},
        json.dumps(
            {
                "id": "msg_1",
                "model": MODEL,
                "content": [{"type": "text", "text": "hi"}],
                "usage": {
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                    "cache_creation_input_tokens": cache_c,
                    "cache_read_input_tokens": cache_r,
                },
            }
        ).encode(),
    )


class StandIn:
    """HTTPS stand-in. `responder(seen) -> (status, headers, body)`; every request is logged."""

    def __init__(self, cert: str, key: str, responder=None):
        self.seen: list[Seen] = []
        self.responder = responder or (lambda s: anthropic_ok())
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                seen = Seen(
                    "POST",
                    self.path,
                    {k.lower(): v for k, v in self.headers.items()},
                    self.rfile.read(n),
                )
                outer.seen.append(seen)
                status, headers, body = outer.responder(seen)
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        self.server.socket = ctx.wrap_socket(self.server.socket, server_side=True)
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.02), daemon=True
        )
        self.thread.start()
        self.port = self.server.server_address[1]
        self.url = f"https://127.0.0.1:{self.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


# --- the layout ---


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise TypeError(v)


def toml_text(*, providers, holder, run_bounds=(), deployment=None) -> str:
    out = []
    if deployment:
        out.append("[deployment]")
        out += [f"{k} = {toml_value(v)}" for k, v in deployment.items()]
    out.append("[holder]")
    out += [f"{k} = {toml_value(v)}" for k, v in holder.items()]
    for pid, prov in providers.items():
        out.append(f"[providers.{pid}]")
        out += [f"{k} = {toml_value(v)}" for k, v in prov.items()]
    for rb in run_bounds:
        out.append("[[run_bounds]]")
        out += [f"{k} = {toml_value(v)}" for k, v in rb.items()]
    return "\n".join(out) + "\n"


@dataclass
class Env:
    root: pathlib.Path
    standin: StandIn
    cert: str
    marker: str = field(default_factory=lambda: "sk-ant-FAKE-" + uuid.uuid4().hex)

    def __post_init__(self):
        r = self.root
        for d in ("etc", "state", "state/registry", "state/keys", "run"):
            (r / d).mkdir(parents=True, exist_ok=True)
            (r / d).chmod(0o755 if d in ("etc", "run") else 0o700)
        r.chmod(0o755)
        self.config_path = r / "etc" / "gateway.toml"
        self.state_root = str(r / "state")
        self.socket_path = str(r / "run" / "holder.sock")
        self.key_path = r / "state" / "keys" / "anth.key"
        self.key_path.write_text(self.marker + "\n")
        self.key_path.chmod(0o600)
        self.add_caller("broker", CALLER_SECRET)
        self.write_config()

    def add_caller(self, name, secret):
        p = self.root / "state" / "registry" / f"{name}.secret"
        p.write_text(secret.hex() + "\n")
        p.chmod(0o600)

    def provider(self, **over):
        prov = {
            "kind": "anthropic",
            "base_url": self.standin.url,
            "key_ref": str(self.key_path),
            "pinned_models": [MODEL],
            "max_tokens_ceiling": 4096,
            "allow_private": True,
        }
        prov.update(over)
        return prov

    def holder_table(self, **over):
        h = {
            "window_seconds": 3600,
            "window_max_tokens": 100_000,
            "window_max_released": 5,
            "unmatched": "refuse",
            "ca_file": self.cert,
            "upstream_timeout_seconds": 5,
        }
        h.update(over)
        return h

    def write_config(self, *, providers=None, holder=None, run_bounds=None, deployment=None):
        text = toml_text(
            providers=providers if providers is not None else {"anth": self.provider()},
            holder=holder if holder is not None else self.holder_table(),
            run_bounds=run_bounds
            if run_bounds is not None
            else [{"match": "eval-*", "max_tokens": 20_000}],
            deployment=deployment,
        )
        self.config_path.write_text(text)
        self.config_path.chmod(0o644)

    def source(self) -> ConfigSource:
        return ConfigSource(self.config_path, trusted_uids={UID}, anchor=self.root)

    def settings(self, **over) -> HolderSettings:
        kw = dict(
            config_source=self.source(), state_root=self.state_root, socket_path=self.socket_path
        )
        kw.update(over)
        return HolderSettings(**kw)

    def holder(self, **over) -> Holder:
        h = Holder(self.settings(**over))
        h.test_endpoint = self.endpoint
        return h

    @property
    def endpoint(self) -> str:
        return self.standin.url


class Clock:
    def __init__(self, now=1_900_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def body_for(max_tokens=100, model=MODEL, **extra) -> bytes:
    doc = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "hello"}],
    }
    doc.update(extra)
    return json.dumps(doc).encode()


def call(
    holder: Holder,
    *,
    body: bytes | None = None,
    run_id="eval-1",
    caller="broker",
    secret=CALLER_SECRET,
    endpoint=None,
    path="/v1/messages",
    method="POST",
    ts=None,
    nonce=None,
    clock=None,
):
    """Sign a request and run it through `Holder.process`; returns `HolderReply` (+ nonce)."""
    endpoint = endpoint if endpoint is not None else holder.test_endpoint
    ts = ts if ts is not None else int((clock or holder.s.clock)())
    frame, n = framing.build_request(
        secret,
        caller=caller,
        method=method,
        endpoint=endpoint,
        path=path,
        run_id=run_id,
        body=body if body is not None else body_for(),
        ts=ts,
        nonce=nonce,
    )
    raw = holder.process(frame[4:])
    if raw is None:
        return None
    return framing.parse_reply(raw[4:], secret, n, require_mac=False)


# --- a real holder subprocess ---


class HolderProc:
    def __init__(self, env: Env, *, extra: dict | None = None, extra_env: dict | None = None):
        spec = {
            "config": str(env.config_path),
            "uid": UID,
            "anchor": str(env.root),
            "state_root": env.state_root,
            "socket": env.socket_path,
            **(extra or {}),
        }
        self.proc = subprocess.Popen(
            [sys.executable, str(LAUNCHER), json.dumps(spec)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PATH": os.environ["PATH"], **(extra_env or {})},
        )
        line = self.proc.stdout.readline()
        if line.strip() != "READY":
            err = self.proc.stderr.read()
            raise RuntimeError(f"holder did not start: {line!r} {err!r}")
        self.env = env

    def exchange(self, **kw):
        kw.setdefault("secret", CALLER_SECRET)
        kw.setdefault("caller", "broker")
        kw.setdefault("endpoint", self.env.endpoint)
        kw.setdefault("path", "/v1/messages")
        kw.setdefault("body", body_for())
        kw.setdefault("run_id", "eval-1")
        return framing.exchange(self.env.socket_path, UID, **kw)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        out, err = self.proc.stdout.read(), self.proc.stderr.read()
        self.stdout, self.stderr = out, err
        return out, err


def raw_connect(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    for _ in range(100):  # a flood can briefly fill the listen backlog
        try:
            s.connect(path)
            return s
        except BlockingIOError:
            time.sleep(0.02)
    s.connect(path)
    return s


def tree_search(roots, needle: bytes) -> list[str]:
    """Every file under `roots` containing `needle` (text and binary, hidden files included)."""
    hits = []
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                p = os.path.join(dirpath, name)
                try:
                    if needle in pathlib.Path(p).read_bytes():
                        hits.append(p)
                except OSError:
                    pass
    return hits


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False
