"""The hook script driving the REAL `coordgraph` Python against an EPHEMERAL FalkorDB and a scratch falkor-chat
(plan section 5 "Hook", `live`). Never the shared instance, never `ws:agent-team`, never :8200.

The hook hard-codes `--confirm coordination`, which `store.require_shared` refuses on an ephemeral port by
design (shared mode and `COORDGRAPH_EPHEMERAL_PORT` are exact complements). So the entry point here is a shim
that removes that one flag pair and records every call; everything else (the shell, `hook-sync`, `retry-failed`,
`kb-finish`) is the real code. The flag's literal and its delivery are pinned in `test_hook_script.py`."""
from __future__ import annotations

import hashlib
import http.server
import json
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from coordgraph import kbsync, store
from coordgraph.parse.kb import claim_hash
from tests.fixtures.live import (  # noqa: F401  (session fixtures)
    SHARED_PORTS,
    EphemeralFalkorDB,
    fake_embedder,
    falkordb_container,
    free_port,
    scratch_falkor_chat,
)
from tests.test_hook_script import Hook, best_vs_baseline

pytestmark = pytest.mark.live

COMPONENT = Path(__file__).resolve().parents[1]
LEDGER = ("# L\n\n> **Status:** active · **Owner:** `teco` · **Tracks:** —\n\n"
          "| Unit | Owner | Status | Deliverable |\n|---|---|---|---|\n| U1 | coder | **queued** | x |\n")
L = "a/docs/plans/x-coordination.md"
KB = "kb/a.md"
SHIM = """#!/bin/bash
echo "$*" >>"$FAKE_DIR/calls.log"
args=()
while [ $# -gt 0 ]; do
  if [ "$1" = --confirm ] && [ "${2:-}" = coordination ]; then shift 2; continue; fi
  args+=("$1"); shift
done
exec "%s" -m coordgraph "${args[@]}"
"""


class LiveHook(Hook):
    def __init__(self, tmp_path, db: EphemeralFalkorDB):
        super().__init__(tmp_path)
        self.db = db
        self.key = f"hk-{uuid.uuid4().hex[:10]}"
        self.cli.write_text(SHIM % sys.executable)
        self.extra = {"PYTHONPATH": str(COMPONENT), "COORDGRAPH_FALKORDB_HOST": db.host,
                      "COORDGRAPH_FALKORDB_PORT": str(db.port), "COORDGRAPH_EPHEMERAL_PORT": str(db.port),
                      "COORDGRAPH_KEY": self.key}
        self.put(".work-items/project.json", json.dumps({"key": "GAL", "legacy_paths": {}}))
        self.dbconn, self.graph = store.connect(db.host, db.port, self.key, env={store.EPHEMERAL_ENV: str(db.port)})
        store.bootstrap(self.dbconn, self.graph, self.key)

    def source(self, path: str):
        rows = self.graph.ro_query("MATCH (s:Source {path: $p}) RETURN s.sha256, s.rowCount", {"p": path}).result_set
        return tuple(rows[0]) if rows else None

    def wait(self, pred, what, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                return
            time.sleep(0.25)
        raise TimeoutError(what)


@pytest.fixture
def lv(falkordb_container, tmp_path):  # noqa: F811
    c = falkordb_container
    assert c.host == "127.0.0.1" and c.port not in SHARED_PORTS and c.name.startswith("coordgraph-test-")
    return LiveHook(tmp_path, c)


def payload(event: str, path: str) -> str:
    return json.dumps({"hook_event_name": event, "tool_name": "Edit", "tool_input": {"file_path": path}})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("event", ["PostToolUse", "PostToolUseFailure"])
def test_ac3_an_edit_is_in_the_graph_when_the_hook_returns(lv, event):
    p = lv.put(L, LEDGER)
    r = lv.run(payload(event, str(p)))
    assert r.returncode == 0 and r.stderr == "", r.stderr
    assert lv.source(L) == (sha(p), 1)
    p.write_text(LEDGER + "| U2 | coder | **queued** | y |\n")
    assert lv.run(payload(event, str(p))).returncode == 0
    assert lv.source(L) == (sha(p), 2)


def test_a_call_that_changed_nothing_starts_no_python_and_is_cheap_against_a_bare_bash_start(lv):
    lv.put(L, LEDGER)
    assert lv.run().returncode == 0
    n = len(lv.calls())
    base, real = best_vs_baseline(lv.run)
    assert len(lv.calls()) == n and real < 6 * base + 0.03, (base, real)


def test_an_outage_warns_on_every_edit_stays_silent_otherwise_and_the_retry_heals_it(tmp_path):
    with EphemeralFalkorDB() as db:
        lv = LiveHook(tmp_path, db)
        p = lv.put(L, LEDGER)
        assert lv.run().returncode == 0 and lv.source(L)
        db.stop()
        p.write_text(LEDGER + "| U2 | coder | **queued** | y |\n")
        r = lv.run()
        assert r.returncode == 2 and L in r.stderr and r.elapsed < 5, (r.stderr, r.elapsed)
        n = len(lv.calls())
        assert lv.stamp.stat().st_mtime >= p.stat().st_mtime
        times = []
        for _ in range(20):
            q = lv.run()
            assert q.returncode == 0 and q.stderr == ""
            times.append(q.elapsed)
        assert len(lv.calls()) == n and min(times) < 0.05, times  # best of 20: load-proof  # no foreground Python, no retry inside the window
        p.write_text(LEDGER + "| U2 | coder | **queued** | z |\n")
        assert lv.run().returncode == 2  # every edit of the failing file warns again
        failed = json.loads((lv.state / "failed.json").read_text())
        assert failed[L]["attempts"] == 1 and "unreachable" in failed[L]["error"]  # a new edit restarted the backoff
        back = EphemeralFalkorDB(port=db.port).start()
        try:
            lv2 = LiveHook.__new__(LiveHook)
            lv2.__dict__.update(lv.__dict__)
            lv2.dbconn, lv2.graph = store.connect(db.host, db.port, lv.key, env={store.EPHEMERAL_ENV: str(db.port)})
            store.bootstrap(lv2.dbconn, lv2.graph, lv.key)
            due = time.time() - 1  # the backoff window passes
            failed[L]["nextRetryAt"] = due
            (lv.state / "failed.json").write_text(json.dumps(failed))
            (lv.state / "retry-due").write_text(f"{int(due)}\n")
            assert lv.run().returncode == 0
            lv2.wait(lambda: lv2.source(L) == (sha(p), 2) and not json.loads((lv.state / "failed.json").read_text()),
                     "the detached retry syncs the file and clears failed.json")
            assert not (lv.state / "retry-due").exists()
        finally:
            back.stop()


def test_a_first_run_syncs_a_file_edited_before_it_and_creates_the_stamp(lv):
    p = lv.put(L, LEDGER)
    assert not lv.stamp.exists()
    r = lv.run()
    assert r.returncode == 0 and r.stderr == "" and lv.stamp.exists()
    assert lv.source(L) == (sha(p), 1)
    lv.stamp.unlink()
    assert lv.run().returncode == 0 and lv.source(L) == (sha(p), 1)  # hashes equal: nothing rewritten


def test_a_first_run_with_the_mirror_down_is_silent_keeps_no_stamp_and_retries_only_after_retry_due(tmp_path):
    dead = free_port()
    sock = socket.socket()
    sock.bind(("127.0.0.1", dead))  # bound, never listening: connect is refused
    try:
        class Down(Hook):
            pass

        h = Down(tmp_path)
        h.cli.write_text(SHIM % sys.executable)
        h.extra = {"PYTHONPATH": str(COMPONENT), "COORDGRAPH_FALKORDB_PORT": str(dead),
                   "COORDGRAPH_EPHEMERAL_PORT": str(dead)}
        h.put(L, LEDGER)
        r = h.run()
        assert r.returncode == 0 and r.stderr == "" and not h.stamp.exists()
        due = int((h.state / "retry-due").read_text())
        assert due >= int(time.time()) + 55
        n = len(h.calls())
        for _ in range(5):
            assert h.run().returncode == 0
        assert len(h.calls()) == n  # not again before retry-due
        (h.state / "retry-due").write_text(f"{int(time.time()) - 1}\n")
        h.run()
        assert len(h.calls()) == n + 1
    finally:
        sock.close()


# --- the KB leg ---------------------------------------------------------------------------------------------

class OkServer:
    """Answers 200 to every GET: the embed endpoint's health probe while the real fake embedder is slow."""

    def __init__(self):
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def kbl(lv, scratch_falkor_chat, fake_embedder):  # noqa: F811
    assert scratch_falkor_chat.db is lv.db and scratch_falkor_chat.base_url != "http://localhost:8200"
    fake_embedder.set_mode("ok")
    tag = uuid.uuid4().hex[:8]
    lv.tag = tag
    lv.kbfile = f"kb/{tag}.md"
    lv.put("coordination-graph/kb-files.txt", f"{lv.kbfile}\n", mtime=time.time() - 500)
    lv.put(lv.kbfile, f"# T\n\n## Fact\n\nFirst {tag}.\n")
    lv.extra.update({"COORDGRAPH_KB_URL": scratch_falkor_chat.base_url,
                     "COORDGRAPH_EMBED_URL": fake_embedder.base_url, "COORDGRAPH_KB_BUDGET_S": "1",
                     "COORDGRAPH_KB_GRAPH": "ws:coordtest"})
    lv.fake = fake_embedder
    lv.observer = kbsync.Observer(lv.dbconn.select_graph("ws:coordtest"))
    yield lv
    fake_embedder.set_mode("ok")


def test_a_kb_edit_is_searchable_when_the_hook_returns(kbl):
    r = kbl.run()
    assert r.returncode == 0, r.stderr
    want = claim_hash(f"First {kbl.tag}.")
    assert [i for i, h, cur, _ in kbl.observer.produced_by("kb-mirror") if cur and h == want] != []


def test_a_down_embed_endpoint_fails_fast_and_creates_no_document(kbl):
    before = len(kbl.observer.produced_by("kb-mirror"))
    kbl.fake.set_mode("down")
    r = kbl.run()
    assert r.returncode == 2 and kbl.kbfile in r.stderr and "embedding endpoint" in r.stderr and r.elapsed < 7
    assert len(kbl.observer.produced_by("kb-mirror")) == before


def test_a_slow_embed_returns_within_the_budget_as_pending_and_the_detached_finish_holds_no_pipe(kbl):
    assert kbl.run().returncode == 0
    probe = OkServer()
    try:
        kbl.extra["COORDGRAPH_EMBED_URL"] = probe.url  # health passes; the real embedder below is slow
        kbl.put(kbl.kbfile, f"# T\n\n## Fact\n\nSecond {kbl.tag}.\n")
        kbl.fake.set_mode("slow", 4)
        r = kbl.run()
        assert r.returncode == 2 and "pending" in r.stderr and kbl.kbfile in r.stderr, r.stderr
        assert r.elapsed < 1 + 2 + 3, r.elapsed  # budget + 2 s, plus interpreter start
        kbl.fake.set_mode("ok")
        kbl.wait(lambda: {r[0]["state"] for r in kbl.graph.query(kbsync.READ_ALL).result_set} == {"synced"},
                 "the detached kb-finish swaps the section")
    finally:
        probe.close()
        kbl.fake.set_mode("ok")


def test_first_run_with_nothing_edited_and_the_kb_services_down_is_exit_0_without_a_warning(kbl):
    assert kbl.run().returncode == 0
    kbl.stamp.unlink()
    kbl.fake.set_mode("down")
    r = kbl.run()
    assert r.returncode == 0 and r.stderr == "" and kbl.stamp.exists()
