"""Fixtures for the service suites: the holder harness's temp layout (real holder, HTTPS
stand-in provider), a local HTTP stand-in (LM Studio-shaped), a gateway configuration with a
hosted and a local call site, and a real service subprocess."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from holder_harness import CALLER_SECRET, MODEL, UID, Env, HolderProc

from llmgateway.client import Client

LAUNCHER = pathlib.Path(__file__).resolve().parent / "service_launcher.py"
LM_OK = json.dumps(
    {
        "model": "lm-reported",
        "choices": [{"message": {"role": "assistant", "content": "local"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        "stats": {"tokens_per_second": 42.5},
        "runtime": {"name": "llama.cpp"},
    }
).encode()
LM_MODELS = json.dumps(
    {"data": [{"id": "m1", "state": "loaded", "loaded_context_length": 4096}]}
).encode()
MSGS = [{"role": "user", "content": "hello"}]

SITES = """
[scopes.plain]

[scopes.secret]
marker = "restricted"

[call_sites.hosted]
component = "answer.hosted"
scopes = ["plain"]
chain = ["anth"]
settings = {max_tokens = 100}
retry = {max = 1, on = ["connection", "timeout", "outage"]}

[call_sites.hosted_fb]
component = "answer.hosted"
scopes = ["plain"]
chain = ["anth", "lm"]
settings = {max_tokens = 100}
retry = {max = 2, on = ["connection", "timeout", "outage"]}

[call_sites.unpinned]
component = "answer.hosted"
scopes = ["plain"]
chain = [{provider = "anth", model = "claude-not-pinned"}]
settings = {max_tokens = 100}

[call_sites.too_big]
component = "answer.hosted"
scopes = ["plain"]
chain = ["anth"]
settings = {max_tokens = 99999}

[call_sites.restricted]
component = "answer.hosted"
scopes = ["secret"]
chain = ["anth"]
settings = {max_tokens = 100}

[call_sites.local]
component = "answer.local"
scopes = ["plain"]
chain = ["lm"]

[call_sites.local_load]
component = "answer.local"
scopes = ["plain"]
chain = ["lm"]
load_mode = "explicit"
"""


class LocalStandIn:
    """An LM Studio-shaped local server on loopback: the models list on GET, a chat reply on
    POST; `received` logs every request (method, path, body)."""

    def __init__(self, *_ignored):
        self.received: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                outer.received.append({"method": self.command, "path": self.path, "body": body})
                data = LM_MODELS if self.path == "/api/v0/models" else LM_OK
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _serve

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()

    @property
    def chats(self) -> list[dict]:
        return [r for r in self.received if r["path"] == "/v1/chat/completions"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Stack:
    """Holder (real process) + HTTPS stand-in + local stand-in + the service config."""

    def __init__(self, env: Env, local: LocalStandIn):
        self.env, self.local = env, local
        self.holder: HolderProc | None = None
        self.service_proc: subprocess.Popen | None = None
        self.state_dir = str(env.root / "svc-state")
        self.socket = str(env.root / "svcrun" / "service.sock")
        self.cred = env.root / "service.cred"
        self.write_credential(CALLER_SECRET)
        self.configure()

    def write_credential(self, secret: bytes) -> None:
        self.cred.write_text(secret.hex() + "\n")
        self.cred.chmod(0o600)

    def configure(self, *, holder=None, extra_providers=None, deployment=None) -> None:
        env = self.env
        providers = {
            "anth": {**env.provider(), "model": MODEL},
            "lm": {"kind": "lmstudio", "base_url": self.local.url, "model": "m1"},
            **(extra_providers or {}),
        }
        env.write_config(
            providers=providers,
            holder=holder,
            deployment={
                "holder_socket": env.socket_path,
                "holder_uid": UID,
                "local_endpoints": [self.local.url],
                **(deployment or {}),
            },
        )
        with open(env.config_path, "a") as fh:
            fh.write(SITES)

    def start_holder(self, **kw) -> HolderProc:
        self.env.add_caller("service", CALLER_SECRET)
        self.holder = HolderProc(self.env, **kw)
        return self.holder

    def stop_holder(self) -> None:
        if self.holder is not None:
            self.holder.stop()
            self.holder = None

    def start_service(self, *, credential: str | None = "default") -> subprocess.Popen:
        (self.env.root / "svcrun").mkdir(mode=0o700, exist_ok=True)
        spec = {
            "config": str(self.env.config_path),
            "uid": UID,
            "anchor": str(self.env.root),
            "socket": self.socket,
            "state_dir": self.state_dir,
            "credential": str(self.cred) if credential == "default" else credential,
        }
        self.service_proc = subprocess.Popen(
            [sys.executable, str(LAUNCHER), json.dumps(spec)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PATH": os.environ["PATH"]},
        )
        line = self.service_proc.stdout.readline()
        if line.strip() != "READY":
            raise RuntimeError(
                f"service did not start: {line!r} {self.service_proc.stderr.read()!r}"
            )
        return self.service_proc

    def stop_service(self) -> tuple[str, str]:
        proc = self.service_proc
        if proc is None:
            return "", ""
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        out, err = proc.stdout.read(), proc.stderr.read()
        self.service_proc = None
        return out, err

    def client(self, **kw) -> Client:
        return Client(self.socket, **kw)

    def calls_log(self) -> list[str]:
        path = pathlib.Path(self.env.state_root) / "calls.log"
        return path.read_text().splitlines() if path.exists() else []

    def close(self) -> None:
        self.stop_service()
        self.stop_holder()
        self.local.close()
        self.env.standin.close()
