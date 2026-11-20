"""Ephemeral live-test fixtures for coordination-graph (plan S1b, note §5.4, m5).

Three pieces, each usable alone, all stdlib + the falkor-chat venv:

  EphemeralFalkorDB   `falkordb/falkordb:v4.18.11` on a free 127.0.0.1 port, no volume,
                      `--rm`, stopped by its own unique name. Never touches `falkordb-dev`.
  FakeEmbedServer     stdlib OpenAI-compatible fake: GET /v1/models, POST /v1/embeddings,
                      POST /v1/chat/completions (extraction answer: empty entities).
                      Deterministic unit-norm vectors of a fixed dim; modes ok/down/slow.
  ScratchFalkorChat   a falkor-chat uvicorn process on its own port + workspace against the
                      ephemeral container, with a throwaway opencode config + models overlay
                      pointing every model kind at the fake. Extraction is not disable-able
                      in falkor-chat (ModelGateway kind `extraction`), so it is *answered by
                      the same fake* (see `_write_configs`).

pytest: `from tests.fixtures.live import *` in a conftest.py (or add this dir to the path)
gives session fixtures `falkordb_container`, `fake_embedder`, `scratch_falkor_chat`, plus
`coordgraph_live_env` (the COORDGRAPH_FALKORDB_* env dict). CLI: `python live.py smoke`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FALKORDB_IMAGE = "falkordb/falkordb:v4.18.11"
REPO_ROOT = Path(__file__).resolve().parents[3]
FALKOR_CHAT = Path(os.environ.get("COORDGRAPH_FALKOR_CHAT_DIR", REPO_ROOT / "falkor-chat"))
FC_VENV_BIN = Path(os.environ.get("COORDGRAPH_FC_VENV", FALKOR_CHAT / "server" / ".venv")) / "bin"
SHARED_PORTS = {6379, 6390, 3000}  # never allowed as a fixture port


def free_port() -> int:
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in SHARED_PORTS:
            return port
    raise RuntimeError("no free port")


def _resp_ping(port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(b"*1\r\n$4\r\nPING\r\n")
            return s.recv(16).startswith(b"+PONG")
    except OSError:
        return False


def _wait(pred, what: str, timeout: float) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.25)
    raise TimeoutError(f"{what} not ready after {timeout}s")


# ── (1) ephemeral FalkorDB ────────────────────────────────────────────────────
class EphemeralFalkorDB:
    def __init__(self, port: int | None = None) -> None:
        self.port = port or free_port()
        self.name = f"coordgraph-test-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.host = "127.0.0.1"

    def start(self) -> "EphemeralFalkorDB":
        assert self.port not in SHARED_PORTS, "refusing a shared FalkorDB port"
        # no -v: state dies with the container. --rm: removed on stop.
        subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", self.name,
             "-p", f"127.0.0.1:{self.port}:6379", FALKORDB_IMAGE],
            check=True, capture_output=True, text=True)
        try:
            _wait(lambda: _resp_ping(self.port), f"{self.name} PING", 60)
        except Exception:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        # Own container only, by unique name; --rm removes it once stopped.
        subprocess.run(["docker", "stop", "-t", "2", self.name], capture_output=True)

    def exists(self) -> bool:
        r = subprocess.run(["docker", "ps", "-a", "-q", "--filter", f"name=^{self.name}$"],
                           capture_output=True, text=True)
        return bool(r.stdout.strip())

    def __enter__(self): return self.start()
    def __exit__(self, *exc): self.stop()


# ── (3) fake embed endpoint ───────────────────────────────────────────────────
def fake_vector(text: str, dim: int) -> list[float]:
    """Deterministic unit-norm vector: same text -> same vector, different text -> different."""
    out: list[float] = []
    counter = 0
    while len(out) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        out.extend(v / 2**31 - 1.0 for v in struct.unpack(">8i", digest[:32]))
        counter += 1
    out = out[:dim]
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


class FakeEmbedServer:
    """Modes: ok | down (HTTP 503 on every route) | slow (sleep `delay` s, then answer).
    Switch live with .set_mode(), or over HTTP: POST /_control {"mode": "slow", "delay": 3}."""

    EMBED_MODEL = "fake-embed"
    CHAT_MODEL = "fake-chat"

    def __init__(self, dim: int = 64, port: int | None = None) -> None:
        self.dim = dim
        self.port = port or free_port()
        self.mode = "ok"
        self.delay = 0.0
        self.embed_calls = 0
        self.chat_calls = 0
        self._srv: ThreadingHTTPServer | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def set_mode(self, mode: str, delay: float = 0.0) -> None:
        assert mode in ("ok", "down", "slow")
        self.mode, self.delay = mode, delay

    def start(self) -> "FakeEmbedServer":
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _send(self, code: int, body: dict) -> None:
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _body(self) -> dict:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}")

            def _gate(self) -> bool:
                if outer.mode == "down":
                    self._send(503, {"error": {"message": "fake embedder is down"}})
                    return False
                if outer.mode == "slow":
                    time.sleep(outer.delay)
                return True

            def do_GET(self):
                if self.path.rstrip("/") == "/v1/models":
                    if not self._gate(): return
                    self._send(200, {"object": "list", "data": [
                        {"id": outer.EMBED_MODEL, "object": "model"},
                        {"id": outer.CHAT_MODEL, "object": "model"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = self._body()
                if self.path == "/_control":  # never gated
                    outer.set_mode(body.get("mode", "ok"), float(body.get("delay", 0)))
                    return self._send(200, {"mode": outer.mode, "delay": outer.delay})
                if not self._gate(): return
                if self.path == "/v1/embeddings":
                    outer.embed_calls += 1
                    inp = body.get("input", "")
                    items = inp if isinstance(inp, list) else [inp]
                    self._send(200, {"object": "list", "model": body.get("model", outer.EMBED_MODEL),
                                     "data": [{"object": "embedding", "index": i,
                                               "embedding": fake_vector(str(t), outer.dim)}
                                              for i, t in enumerate(items)],
                                     "usage": {"prompt_tokens": 0, "total_tokens": 0}})
                elif self.path == "/v1/chat/completions":
                    outer.chat_calls += 1  # entity extraction: nothing to extract
                    self._send(200, {"object": "chat.completion", "model": outer.CHAT_MODEL,
                                     "choices": [{"index": 0, "finish_reason": "stop",
                                                  "message": {"role": "assistant",
                                                              "content": '{"entities": [], "relationships": []}'}}],
                                     "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
                else:
                    self._send(404, {"error": "not found"})

        self._srv = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self._srv.daemon_threads = True
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._srv:
            self._srv.shutdown(); self._srv.server_close(); self._srv = None

    def __enter__(self): return self.start()
    def __exit__(self, *exc): self.stop()


# ── (2) scratch falkor-chat ───────────────────────────────────────────────────
class ScratchFalkorChat:
    """A falkor-chat server (uvicorn, no --reload) on its own port + workspace, against
    `db`, embedding/extracting through `fake`. Needs falkor-chat/server/.venv and redis-cli."""

    def __init__(self, db: EphemeralFalkorDB, fake: FakeEmbedServer,
                 workspace: str = "coordtest", port: int | None = None,
                 embed_timeout_s: int = 5, seed_agents: bool = True) -> None:
        self.db, self.fake, self.workspace = db, fake, workspace
        self.port = port or free_port()
        self.embed_timeout_s = embed_timeout_s
        self.seed_agents = seed_agents
        self.tmp = Path(tempfile.mkdtemp(prefix="coordgraph-fc-"))
        self.proc: subprocess.Popen | None = None
        self.log_path = self.tmp / "server.log"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def mcp_url(self) -> str:
        return f"{self.base_url}/mcp"

    def _write_configs(self) -> dict[str, str]:
        # The ONLY way falkor-chat routes extraction: `defaults.extraction` in the models overlay
        # (kind `extraction`, SERVER.md §1.12). There is no off switch while FALKORCHAT_ENABLE_AGENT=1
        # (which search_documents/ingest need), so extraction is pointed at the fake's
        # /chat/completions, which answers an empty entity set.
        provider = {"$schema": "https://opencode.ai/config.schema.json", "provider": {"fake": {
            "npm": "@ai-sdk/openai-compatible", "name": "coordgraph fake",
            "options": {"baseURL": self.fake.base_url},
            "models": {self.fake.EMBED_MODEL: {"name": "fake embed"},
                       self.fake.CHAT_MODEL: {"name": "fake chat"}}}}}
        emb, chat = f"fake/{self.fake.EMBED_MODEL}", f"fake/{self.fake.CHAT_MODEL}"
        overlay = {
            "defaults": {"agent": chat, "step": chat, "guard": chat, "extraction": chat, "embedding": emb},
            "timeouts": {"agent": 10, "step": 10, "guard": 10, "extraction": 10,
                         "embedding": self.embed_timeout_s},
            "models": {emb: {"dim": self.fake.dim}, chat: {"temperature": 0}},
        }
        oc, ov = self.tmp / "opencode.json", self.tmp / "models.json"
        oc.write_text(json.dumps(provider)); ov.write_text(json.dumps(overlay))
        return {"FALKORCHAT_OPENCODE_CONFIG": str(oc), "FALKORCHAT_MODEL_CONFIG": str(ov)}

    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("FALKORCHAT_", "FALKORDB_"))}
        env.update(self._write_configs())
        env.update({
            "FALKORDB_HOST": self.db.host, "FALKORDB_PORT": str(self.db.port),
            "FALKORCHAT_WS_ID": self.workspace, "FALKORCHAT_USER_ID": "u1",
            "FALKORCHAT_EMBEDDING_DIM": str(self.fake.dim), "FALKORCHAT_ENABLE_AGENT": "1",
            "FALKORCHAT_REFERENCE_GRAPH": "reference-coordtest",
        })
        return env

    def start(self) -> "ScratchFalkorChat":
        assert self.db.port not in SHARED_PORTS
        env = self._env()
        sh = {**env, "EMBEDDING_DIM": str(self.fake.dim)}  # scripts read EMBEDDING_DIM / FALKORDB_*
        # Schema + roster go to the ephemeral container only (explicit FALKORDB_PORT).
        subprocess.run([str(FALKOR_CHAT / "scripts" / "bootstrap_schema.sh"), self.workspace],
                       env=sh, check=True, capture_output=True, text=True)
        if self.seed_agents:
            subprocess.run([str(FALKOR_CHAT / "scripts" / "seed_agent_team.sh"), self.workspace],
                           env=sh, check=True, capture_output=True, text=True)
        log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            [str(FC_VENV_BIN / "uvicorn"), "falkorchat.app:app", "--port", str(self.port),
             "--host", "127.0.0.1"],
            cwd=FALKOR_CHAT / "server", env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait(self._healthy, "falkor-chat /health", 40)
        except Exception:
            self.stop()
            raise RuntimeError("scratch falkor-chat failed to start:\n" + self.log_path.read_text()[-3000:])
        return self

    def _healthy(self) -> bool:
        if self.proc and self.proc.poll() is not None:
            raise RuntimeError("server exited:\n" + self.log_path.read_text()[-3000:])
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=1) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def __enter__(self): return self.start()
    def __exit__(self, *exc): self.stop()


# ── pytest fixtures (session scope; import into a conftest) ───────────────────
try:
    import pytest

    @pytest.fixture(scope="session")
    def falkordb_container():
        with EphemeralFalkorDB() as db:
            yield db

    @pytest.fixture(scope="session")
    def fake_embedder():
        with FakeEmbedServer() as f:
            yield f

    @pytest.fixture(scope="session")
    def scratch_falkor_chat(falkordb_container, fake_embedder):
        with ScratchFalkorChat(falkordb_container, fake_embedder) as fc:
            yield fc

    @pytest.fixture(scope="session")
    def coordgraph_live_env(falkordb_container):
        return {"COORDGRAPH_FALKORDB_HOST": falkordb_container.host,
                "COORDGRAPH_FALKORDB_PORT": str(falkordb_container.port)}
except ImportError:  # CLI use without pytest
    pass


# ── smoke ─────────────────────────────────────────────────────────────────────
def _http(method: str, url: str, body: dict | None = None, timeout: float = 15):
    req = urllib.request.Request(url, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def smoke() -> None:
    with EphemeralFalkorDB() as db, FakeEmbedServer(dim=64) as fake:
        print(f"container {db.name} on 127.0.0.1:{db.port}, PING ok={_resp_ping(db.port)}")
        print("fake /models:", [m["id"] for m in _http("GET", fake.base_url + "/models")[1]["data"]])
        s, v = _http("POST", fake.base_url + "/embeddings", {"model": "x", "input": "hello"})
        s2, v2 = _http("POST", fake.base_url + "/embeddings", {"model": "x", "input": "hello"})
        e = v["data"][0]["embedding"]
        print(f"fake /embeddings: {s} dim={len(e)} deterministic={e == v2['data'][0]['embedding']} "
              f"norm={math.sqrt(sum(x*x for x in e)):.4f}")
        fake.set_mode("down"); print("down  ->", _http("POST", fake.base_url + "/embeddings", {"input": "a"})[0])
        fake.set_mode("slow", 2); t = time.time()
        _http("POST", fake.base_url + "/embeddings", {"input": "a"}); print(f"slow 2s -> took {time.time()-t:.1f}s")
        fake.set_mode("ok")
        with ScratchFalkorChat(db, fake) as fc:
            print(f"scratch falkor-chat on :{fc.port} ws:{fc.workspace} /health =",
                  _http("GET", fc.base_url + "/health"))
            s, doc = _http("POST", fc.base_url + "/documents",
                           {"title": "smoke", "text": "Alpha beta gamma. " * 5, "source": "smoke"})
            print("POST /documents ->", s, doc)
            time.sleep(3)
            print("fake saw embed_calls =", fake.embed_calls, " chat(extraction) calls =", fake.chat_calls)
            if s in (200, 201):
                print("GET document ->", _http("GET", f"{fc.base_url}/documents/{doc.get('documentId')}")[1].get("status"))
        name = db.name
    time.sleep(1)
    print(f"after teardown: container {name} still exists = {db.exists()}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["smoke"]:
        smoke()
    else:
        print(__doc__)
