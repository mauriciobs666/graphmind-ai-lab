"""`KbHttp` against a real local stdlib server, and the `kb-*` subcommands through `cli.main` over the fakes."""
from __future__ import annotations

import fcntl
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from coordgraph import cli, commands_kb, kbsync
from coordgraph.config import Config
from coordgraph.kbsync import KbHttp, KbHttpError
from tests.kbfakes import make_engine


class Server:
    """Records requests; answers per `plan` (a status, body, optional delay) keyed by (method, path-prefix)."""

    def __init__(self):
        self.seen: list[tuple[str, str, dict]] = []
        self.plan: dict[tuple[str, str], tuple[int, dict, float]] = {}
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.seen.append((self.command, self.path, body))
                status, payload, delay = next((v for (m, p), v in outer.plan.items()
                                               if m == self.command and self.path.startswith(p)), (200, {}, 0))
                time.sleep(delay)
                raw = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except OSError:
                    pass

            do_GET = do_POST = do_DELETE = _handle

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.srv.server_address[1]}"


@pytest.fixture
def server():
    s = Server()
    yield s
    s.srv.shutdown()
    s.srv.server_close()


def test_post_sends_the_kb_mirror_identity_and_returns_the_receipt(server):
    server.plan[("POST", "/documents")] = (201, {"documentId": "d1", "autoSuperseded": False}, 0)
    receipt = KbHttp(server.url, 2).post_document("Title", "Body")
    assert receipt["documentId"] == "d1"
    assert server.seen == [("POST", "/documents", {"text": "Body", "title": "Title", "sourceFormat": "markdown",
                                                    "producedBy": "kb-mirror"})]


def test_delete_treats_404_as_done_and_other_errors_as_failures(server):
    server.plan[("DELETE", "/documents/gone")] = (404, {"detail": "nope"}, 0)
    server.plan[("DELETE", "/documents/bad")] = (500, {"detail": "boom"}, 0)
    http = KbHttp(server.url, 2)
    http.delete_document("gone")
    with pytest.raises(KbHttpError, match="HTTP 500") as err:
        http.delete_document("bad")
    assert err.value.status == 500 and not err.value.timed_out


def test_a_client_timeout_is_flagged_outcome_unknown_and_a_refused_connection_is_not(server):
    server.plan[("POST", "/documents")] = (201, {"documentId": "d1"}, 1.0)
    with pytest.raises(KbHttpError) as slow:
        KbHttp(server.url, 0.2).post_document("T", "B")
    assert slow.value.timed_out
    server.srv.shutdown()
    server.srv.server_close()
    with pytest.raises(KbHttpError) as down:
        KbHttp(server.url, 0.5).post_document("T", "B")
    assert not down.value.timed_out


def test_a_receipt_without_a_document_id_is_an_error(server):
    server.plan[("POST", "/documents")] = (201, {"chunkCount": 1}, 0)
    with pytest.raises(KbHttpError, match="without documentId"):
        KbHttp(server.url, 2).post_document("T", "B")


def test_health(server):
    server.plan[("GET", "/health")] = (503, {}, 0)
    with pytest.raises(KbHttpError, match="503"):
        KbHttp(server.url, 2).health()


# --- the subcommands -------------------------------------------------------------------------------

ONE = "# T\n\n## Alpha\n\nAlpha body.\n"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "coordination-graph").mkdir()
    (tmp_path / "coordination-graph" / "kb-files.txt").write_text("# registry\nkb/a.md\nkb/missing.md\n")
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "a.md").write_text(ONE)
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("COORDGRAPH_STATE_DIR", str(tmp_path / "state"))
    engine, mirror, chat, _ = make_engine()
    monkeypatch.setattr(commands_kb, "_build", lambda args: (Config.from_env(), tmp_path, None, mirror))
    monkeypatch.setattr(commands_kb, "_engine", lambda cfg, db, graph: engine)
    return tmp_path, mirror, chat


def test_kb_sync_reports_each_section_and_an_unreadable_file_without_aborting(repo, capsys):
    root, mirror, chat = repo
    assert cli.main(["kb-sync"]) == 2  # kb/missing.md is registered but not on disk
    cap = capsys.readouterr()
    assert "synced\tkb/a.md § Alpha" in cap.out and "failed\tkb/missing.md" in cap.out
    assert chat.posts() == ["Alpha"]
    assert cli.main(["kb-sync", "kb/a.md"]) == 0  # named path: settled


def test_kb_sync_rejects_a_path_that_is_not_registered(repo, capsys):
    assert cli.main(["kb-sync", "kb/other.md"]) == 2
    assert "not a registered KB file" in capsys.readouterr().out


def test_kb_sync_launches_a_detached_finish_only_when_a_section_is_pending(repo, monkeypatch):
    root, mirror, chat = repo
    (root / "coordination-graph" / "kb-files.txt").write_text("kb/a.md\n")
    launched = []
    monkeypatch.setattr(commands_kb, "detach_finish", lambda cfg, confirm, graph: launched.append(1))
    assert cli.main(["kb-sync", "--detach-finish"]) == 0 and launched == []
    chat.hold_new = True
    (root / "kb" / "a.md").write_text(ONE.replace("body.", "body v2."))
    assert cli.main(["kb-sync", "--detach-finish"]) == 2 and launched == [1]


def test_detach_finish_starts_a_session_leader_with_no_inherited_pipes(tmp_path):
    cfg = Config.from_env({"COORDGRAPH_STATE_DIR": str(tmp_path / "s")})
    calls = []
    commands_kb.detach_finish(cfg, popen=lambda argv, **kw: calls.append((argv, kw)))
    argv, kw = calls[0]
    assert argv[-1] == "kb-finish" and kw["start_new_session"] is True
    assert kw["stdin"] == commands_kb.subprocess.DEVNULL and kw["stdout"].name.endswith("hook.log")


def test_kb_status_exit_codes_and_output(repo, capsys):
    root, mirror, chat = repo
    (root / "coordination-graph" / "kb-files.txt").write_text("kb/a.md\n")
    assert cli.main(["kb-status"]) == 2  # unmirrored: "not complete", as for every kb-* command
    assert "unmirrored\tkb/a.md § Alpha" in capsys.readouterr().out
    cli.main(["kb-sync"])
    capsys.readouterr()
    assert cli.main(["kb-status"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("KB mirror: last sync 19") or out.startswith("KB mirror: last sync 1970")
    (root / "kb" / "a.md").write_text(ONE.replace("body.", "edited."))
    assert cli.main(["kb-status", "--ids", chat.current()[0]["id"]]) == 2
    assert "stale\tkb/a.md § Alpha\tstale — read `kb/a.md` § `Alpha`" in capsys.readouterr().out


def test_kb_status_exits_1_only_when_it_could_not_run(repo, monkeypatch, capsys):
    monkeypatch.setattr(commands_kb.kbsync.KbSync,
                        "read_all", lambda self: (_ for _ in ()).throw(RuntimeError("graph gone")))
    assert cli.main(["kb-status"]) == 1
    assert "graph gone" in capsys.readouterr().err


def test_an_unforeseen_error_in_one_file_loses_only_that_files_outcomes(repo, monkeypatch, capsys):
    """F6: the isolation lesson of S3: one file's crash is a `failed` outcome, the other file still syncs."""
    root, mirror, chat = repo
    (root / "kb" / "b.md").write_text("# T\n\n## Beta\n\nBeta body.\n")
    (root / "coordination-graph" / "kb-files.txt").write_text("kb/a.md\nkb/b.md\n")
    real = commands_kb.kbsync.KbSync.sync_file

    def flaky(self, doc, ctx, rows):
        if doc.path == "kb/a.md":
            raise RuntimeError("boom")
        return real(self, doc, ctx, rows)

    monkeypatch.setattr(commands_kb.kbsync.KbSync, "sync_file", flaky)
    assert cli.main(["kb-sync"]) == 2
    out = capsys.readouterr().out
    assert "failed\tkb/a.md" in out and "boom" in out and "synced\tkb/b.md § Beta" in out
    assert chat.posts() == ["Beta"]


def test_kb_reconcile_exit_codes(repo, capsys):
    root, mirror, chat = repo
    (root / "coordination-graph" / "kb-files.txt").write_text("kb/a.md\n")
    cli.main(["kb-sync"])
    assert cli.main(["kb-reconcile"]) == 0
    chat.plant("stray", producer="kb-mirror", title="stray")
    assert cli.main(["kb-reconcile"]) == 2 and "orphan" in capsys.readouterr().out


def test_kb_finish_is_single_instance_and_a_second_one_exits_quietly(repo, capsys):
    root, mirror, chat = repo
    (root / "state").mkdir()
    with open(root / "state" / "kb-finish.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert cli.main(["kb-finish", "--once"]) == 0
    assert "another kb-finish is running" in capsys.readouterr().err
    assert cli.main(["kb-finish", "--once"]) == 0


def test_the_kb_commands_refuse_a_shared_falkordb_port_without_connecting(monkeypatch, capsys):
    monkeypatch.setenv("COORDGRAPH_FALKORDB_PORT", "6379")
    monkeypatch.delenv("COORDGRAPH_EPHEMERAL_PORT", raising=False)
    for cmd in ("kb-sync", "kb-status", "kb-reconcile", "kb-finish"):
        assert cli.main([cmd]) == 1
        assert "refusing" in capsys.readouterr().err


def test_a_sync_never_needs_wilib_and_the_engine_imports_no_sync_incompatible_module():
    assert "wilib" not in vars(kbsync) and "identity" not in vars(kbsync)
    assert "wilib" not in vars(commands_kb) and "identity" not in vars(commands_kb)


# --- F1: the falkor-chat endpoint is named together with the ephemeral FalkorDB ---------------------

@pytest.fixture
def real_engine(monkeypatch, tmp_path):
    """The real `_engine` over a mirror that connected (as it always does) through the ephemeral override."""
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("COORDGRAPH_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("COORDGRAPH_FALKORDB_PORT", "16399")
    monkeypatch.setenv("COORDGRAPH_EPHEMERAL_PORT", "16399")
    monkeypatch.delenv("COORDGRAPH_KB_URL", raising=False)
    made = []

    def no_request(*a, **kw):
        made.append(a)
        raise AssertionError("a KbHttp was built for a refused endpoint")

    monkeypatch.setattr(commands_kb.kbsync, "KbHttp", no_request)
    monkeypatch.setattr(commands_kb, "_build", lambda args: (Config.from_env(), tmp_path, None, None))
    return made


@pytest.mark.parametrize("url", [None, "http://localhost:8200", "http://127.0.0.1:8200/",
                                 "http://falkor-chat.example:9000", "http://10.0.0.5:9000"])
def test_an_ephemeral_mirror_refuses_an_unset_or_shared_falkor_chat_url_before_any_request(
        real_engine, monkeypatch, capsys, url):
    if url:
        monkeypatch.setenv("COORDGRAPH_KB_URL", url)
    for cmd in ("kb-sync", "kb-status", "kb-reconcile", "kb-finish"):
        assert cli.main([cmd]) == 1
        assert "refusing" in capsys.readouterr().err
    assert real_engine == []


def test_an_ephemeral_mirror_accepts_a_loopback_scratch_falkor_chat(monkeypatch):
    monkeypatch.setenv("COORDGRAPH_KB_URL", "http://127.0.0.1:18200")
    cfg = Config.from_env()
    commands_kb._engine(cfg, type("Db", (), {"select_graph": lambda s, k: None})(), None)
