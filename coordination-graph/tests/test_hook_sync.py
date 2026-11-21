"""Offline tests of `hookstate`, `hook-sync` and `retry-failed` (plan section 3.2) over the in-memory graph, the
KB fakes and tmp_path trees. The shell half is `test_hook_script.py`; the live legs are `test_hook_live.py`.

Nothing here opens a socket to a database: `commands_hook.ping` and `store.open_graph` are replaced."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import socket
import subprocess
import sys

import falkordb
import pytest

from coordgraph import cli, commands_hook, commands_kb, hookstate, kbsync, store, sync
from coordgraph.config import Config
from tests.kbfakes import make_engine
from tests.memgraph import MemGraph

LEDGER = "# L\n\n> **Status:** active · **Owner:** `teco` · **Tracks:** —\n\n| Unit | Owner | Status | Deliverable |\n|---|---|---|---|\n| U1 | coder | **queued** | x |\n"
L = "a/docs/plans/x-coordination.md"
L2 = "a/docs/plans/y-coordination.md"
KB = "kb/a.md"
KB_TEXT = "# T\n\n## Alpha\n\nAlpha body.\n"
NOW = 1_700_000_000.0


def put(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name in ("COORDGRAPH_SHARED_CONFIRM", "COORDGRAPH_EPHEMERAL_PORT", "COORDGRAPH_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("COORDGRAPH_STATE_DIR", str(tmp_path / "state"))
    graph = MemGraph()
    opened = []

    def open_graph(host, port, key, confirm=None, env=None):
        opened.append((host, port, key, confirm))
        return None, graph

    monkeypatch.setattr(store, "open_graph", open_graph)
    monkeypatch.setattr(store, "require_endpoint", lambda *a, **k: None)
    monkeypatch.setattr(commands_hook, "ping", lambda host, port, timeout=1.0: None)
    monkeypatch.setattr(commands_hook.time, "time", lambda: NOW)
    put(tmp_path, ".work-items/project.json", json.dumps({"key": "GAL", "legacy_paths": {}}))
    put(tmp_path, L, LEDGER)
    return type("E", (), {"root": tmp_path, "graph": graph, "opened": opened,
                          "state": hookstate.HookState(tmp_path / "state")})


def failed(env):
    return json.loads(env.state.failed_json.read_text())


def down(monkeypatch):
    def boom(host, port, timeout=1.0):
        raise OSError("connection refused")

    monkeypatch.setattr(commands_hook, "ping", boom)


# --- hookstate -----------------------------------------------------------------------------------------

def test_backoff_is_1_2_4_8_then_15_minutes():
    assert [hookstate.backoff_s(n) for n in (1, 2, 3, 4, 5, 6, 40)] == [60, 120, 240, 480, 900, 900, 900]


def test_relative_converts_absolute_and_ignores_paths_outside(tmp_path):
    assert hookstate.relative(tmp_path, str(tmp_path / "a" / "b.md")) == "a/b.md"
    assert hookstate.relative(tmp_path, "./a/b.md") == "a/b.md"
    assert hookstate.relative(tmp_path, "/elsewhere/x.md") is None


def test_take_pending_dedupes_and_truncates(tmp_path):
    st = hookstate.HookState(tmp_path)
    st.pending.write_text("b\na\nb\n\n")
    assert st.take_pending() == ["a", "b"] and st.pending.read_text() == ""


def test_advance_stamp_never_moves_backwards(tmp_path):
    st = hookstate.HookState(tmp_path)
    st.advance_stamp(2000.0)
    assert st.stamp.stat().st_mtime == 2000.0
    st.advance_stamp(1000.0)
    assert st.stamp.stat().st_mtime == 2000.0
    st.advance_stamp(3000.0)
    assert st.stamp.stat().st_mtime == 3000.0


def test_save_failed_writes_the_earliest_retry_due_and_removes_it_when_empty(tmp_path):
    st = hookstate.HookState(tmp_path)
    st.save_failed({"a": {"nextRetryAt": 150.2}, "b": {"nextRetryAt": 120.0}})
    assert st.retry_due.read_text().strip() == "120"
    st.save_failed({})
    assert not st.retry_due.exists()


# --- hook-sync: success ---------------------------------------------------------------------------------

def test_an_edit_is_mirrored_and_clears_a_prior_failure(env, capsys):
    env.state.ensure()
    env.state.save_failed({L: {"sha256": "old", "error": "x", "attempts": 3, "nextRetryAt": NOW + 900}})
    assert cli.main(["hook-sync", "--confirm", "coordination", L]) == 0
    assert L in env.graph.sources
    assert failed(env) == {} and not env.state.retry_due.exists()
    assert capsys.readouterr().out == ""
    assert env.opened == [("localhost", 6379, "coordination", "coordination")]


def test_no_paths_and_no_first_run_does_nothing(env):
    assert cli.main(["hook-sync"]) == 0 and env.opened == []


def test_a_path_the_mirror_does_not_cover_is_not_a_failure(env):
    put(env.root, "a/docs/plans/sub/z-coordination.md.bak", "x")
    put(env.root, "other/file.md", "x")
    assert cli.main(["hook-sync", "other/file.md"]) == 0 and failed(env) == {}


def test_an_absolute_payload_path_is_stored_repo_relative(env, monkeypatch):
    down(monkeypatch)
    assert cli.main(["hook-sync", str(env.root / L)]) == 2
    assert list(failed(env)) == [L]


# --- hook-sync: failure and backoff ---------------------------------------------------------------------

def test_a_mirror_outage_exits_2_naming_the_file_and_backs_off_one_minute(env, monkeypatch, capsys):
    down(monkeypatch)
    assert cli.main(["hook-sync", L]) == 2
    err = capsys.readouterr().err
    assert L in err and "FalkorDB unreachable" in err and "this update is not complete" in err
    entry = failed(env)[L]
    assert entry["attempts"] == 1 and entry["nextRetryAt"] == NOW + 60
    assert env.state.retry_due.read_text().strip() == str(int(NOW + 60))
    assert env.opened == []


def test_a_second_edit_of_the_same_failing_file_warns_again_and_doubles_the_backoff(env, monkeypatch):
    down(monkeypatch)
    assert cli.main(["hook-sync", L]) == 2
    assert cli.main(["hook-sync", L]) == 2
    assert failed(env)[L]["attempts"] == 2 and failed(env)[L]["nextRetryAt"] == NOW + 120


def test_a_new_edit_restarts_the_backoff(env, monkeypatch):
    down(monkeypatch)
    for _ in range(3):
        cli.main(["hook-sync", L])
    assert failed(env)[L]["attempts"] == 3
    (env.root / L).write_text(LEDGER + "| U2 | coder | **queued** | y |\n")
    cli.main(["hook-sync", L])
    assert failed(env)[L]["attempts"] == 1


def test_the_backoff_caps_at_fifteen_minutes(env, monkeypatch):
    down(monkeypatch)
    for _ in range(7):
        cli.main(["hook-sync", L])
    assert failed(env)[L]["nextRetryAt"] == NOW + 900


def test_one_failing_file_does_not_stop_the_others(env, capsys):
    put(env.root, L2, LEDGER)
    env.graph.fail_paths = {L}
    assert cli.main(["hook-sync", L, L2]) == 2
    assert L2 in env.graph.sources and list(failed(env)) == [L]
    assert L in capsys.readouterr().err


def refuse_guard(monkeypatch, reason="refusing shared access while COORDGRAPH_EPHEMERAL_PORT is set"):
    def refuse(*a, **k):
        raise store.RefusedEndpoint(reason)

    monkeypatch.setattr(store, "require_endpoint", refuse)


def test_a_refused_endpoint_is_reported_as_a_refusal_not_an_outage_and_is_never_queued(env, monkeypatch, capsys):
    refuse_guard(monkeypatch)
    assert cli.main(["hook-sync", L]) == 2
    cap = capsys.readouterr()
    assert "refused by the guard" in cap.err and "COORDGRAPH_EPHEMERAL_PORT" in cap.err and L in cap.err
    assert "retried automatically" not in cap.err and "COORDGRAPH_HOOK_DISABLED=1" in cap.err
    assert "refused by the guard" in cap.out  # stdout is the shell's hook.log
    assert not env.state.failed_json.exists() and not env.state.retry_due.exists()


def test_a_refusal_does_not_touch_entries_already_in_the_failure_state(env, monkeypatch):
    env.state.ensure()
    entry = {"sha256": "x", "error": "e", "attempts": 2, "nextRetryAt": NOW - 1}
    env.state.save_failed({L: dict(entry)})
    env.state.write_retry_due(int(NOW) + 60)  # the shell's bump before it forked this retry
    refuse_guard(monkeypatch)
    assert cli.main(["retry-failed"]) == 0
    assert failed(env) == {L: entry}
    assert env.state.retry_due.read_text().strip() == str(int(NOW) + 60)  # not rewritten to the past: no fork per call


def test_the_guard_runs_before_the_ping(env, monkeypatch):
    refuse_guard(monkeypatch)
    monkeypatch.setattr(commands_hook, "ping", lambda *a, **k: pytest.fail("pinged before the guard"))
    assert cli.main(["hook-sync", L]) == 2


def test_a_refused_run_never_connects_to_a_listener_on_a_non_shared_port(env, monkeypatch, capsys):
    monkeypatch.undo()  # real guard, real ping
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(env.root))
    monkeypatch.setenv("COORDGRAPH_STATE_DIR", str(env.root / "state"))
    for name in ("COORDGRAPH_SHARED_CONFIRM", "COORDGRAPH_EPHEMERAL_PORT", "COORDGRAPH_KEY"):
        monkeypatch.delenv(name, raising=False)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(0.5)
    monkeypatch.setenv("COORDGRAPH_FALKORDB_PORT", str(srv.getsockname()[1]))
    try:
        assert cli.main(["hook-sync", "--confirm", "coordination", L]) == 2
        with pytest.raises(socket.timeout):
            srv.accept()  # nobody connected
    finally:
        srv.close()
    assert "not a shared instance port" in capsys.readouterr().err


def test_a_first_run_refused_by_the_guard_is_loud_logged_and_asks_again_only_after_15_minutes(env, monkeypatch,
                                                                                            capsys):
    refuse_guard(monkeypatch, "nope")
    assert cli.main(["hook-sync", "--first-run"]) == 2
    cap = capsys.readouterr()
    assert "refused by the guard: nope" in cap.err and "refused by the guard: nope" in cap.out
    assert not env.state.stamp.exists() and env.state.retry_due.read_text().strip() == str(int(NOW + 900))
    assert not env.state.failed_json.exists()


def test_the_shells_pending_paths_are_merged_into_failed_json_by_the_next_run(env):
    env.state.ensure()
    env.state.pending.write_text(f"{L2}\n")
    put(env.root, L2, LEDGER)
    assert cli.main(["hook-sync", L]) == 0
    assert L2 in failed(env) and env.state.pending.read_text() == ""


def test_a_failed_file_that_is_gone_is_dropped_not_deleted_from_the_graph(env):
    env.state.ensure()
    env.state.save_failed({"a/docs/plans/gone-coordination.md": {"sha256": "x", "error": "e", "attempts": 1,
                                                                 "nextRetryAt": NOW - 1}})
    env.graph.sources["a/docs/plans/gone-coordination.md"] = ("ledger", {"sha256": "s"}, [])
    assert cli.main(["hook-sync", "a/docs/plans/gone-coordination.md"]) == 0
    assert failed(env) == {} and "a/docs/plans/gone-coordination.md" in env.graph.sources


# --- hook-sync --first-run ------------------------------------------------------------------------------

def test_first_run_syncs_what_differs_and_creates_the_stamp_at_the_call_start(env):
    assert cli.main(["hook-sync", "--first-run", "--stamp-at", "1234.5"]) == 0
    assert L in env.graph.sources and env.state.stamp.stat().st_mtime == 1234.5


def test_first_run_with_the_mirror_down_is_silent_leaves_no_stamp_and_waits_a_minute(env, monkeypatch, capsys):
    down(monkeypatch)
    assert cli.main(["hook-sync", "--first-run"]) == 0
    cap = capsys.readouterr()
    assert cap.err == "" and cap.out == ""
    assert not env.state.stamp.exists() and env.state.retry_due.read_text().strip() == str(int(NOW + 60))
    assert not env.state.failed_json.exists()


def test_first_run_with_nothing_edited_and_kb_services_down_exits_0_without_a_warning(env, monkeypatch, capsys):
    put(env.root, "coordination-graph/kb-files.txt", f"{KB}\n")
    put(env.root, KB, KB_TEXT)
    engine, mirror, chat, _ = make_engine()
    monkeypatch.setattr(commands_kb, "_engine", lambda cfg, db, graph: engine)
    monkeypatch.setattr(commands_hook, "http_ok", lambda url, timeout=2.0: None)
    assert cli.main(["hook-sync", "--first-run"]) == 0  # synced once with the services up
    assert chat.posts() == ["Alpha"]

    def dead(url, timeout=2.0):
        raise OSError("down")

    monkeypatch.setattr(commands_hook, "http_ok", dead)
    env.state.stamp.unlink()
    assert cli.main(["hook-sync", "--first-run"]) == 0  # hashes equal: nothing to health-check
    assert capsys.readouterr().err == ""


# --- hook-sync: the KB leg ------------------------------------------------------------------------------

@pytest.fixture
def kb(env, monkeypatch):
    put(env.root, "coordination-graph/kb-files.txt", f"{KB}\n")
    put(env.root, KB, KB_TEXT)
    engine, mirror, chat, clock = make_engine()
    monkeypatch.setattr(commands_kb, "_engine", lambda cfg, db, graph: engine)
    monkeypatch.setattr(commands_hook, "http_ok", lambda url, timeout=2.0: None)
    return type("K", (), {"mirror": mirror, "chat": chat, "env": env})


def test_a_kb_edit_is_mirrored_in_the_foreground(kb):
    assert cli.main(["hook-sync", KB]) == 0 and kb.chat.posts() == ["Alpha"]


def test_a_down_embed_endpoint_fails_fast_and_posts_nothing(kb, monkeypatch, capsys):
    def only_embed_down(url, timeout=2.0):
        if url.endswith("/models"):
            raise OSError("down")

    monkeypatch.setattr(commands_hook, "http_ok", only_embed_down)
    assert cli.main(["hook-sync", KB]) == 2
    err = capsys.readouterr().err
    assert "the embedding endpoint unreachable" in err and "once the embedding endpoint is back" in err
    assert kb.chat.posts() == [] and KB in failed(kb.env)


def test_a_down_falkor_chat_names_it(kb, monkeypatch, capsys):
    def chat_down(url, timeout=2.0):
        if url.endswith("/health"):
            raise OSError("down")

    monkeypatch.setattr(commands_hook, "http_ok", chat_down)
    assert cli.main(["hook-sync", KB]) == 2
    assert "falkor-chat unreachable" in capsys.readouterr().err


def test_a_pending_section_exits_2_and_launches_the_detached_finish(kb, monkeypatch, capsys):
    launched = []
    monkeypatch.setattr(commands_kb, "detach_finish", lambda cfg, confirm, graph: launched.append(confirm))
    kb.chat.hold_new = True
    assert cli.main(["hook-sync", "--confirm", "coordination", KB]) == 2
    assert "pending" in capsys.readouterr().err and launched == ["coordination"]
    assert KB in failed(kb.env)


def test_an_unchanged_kb_file_is_not_health_checked(kb, monkeypatch):
    assert cli.main(["hook-sync", KB]) == 0

    def dead(url, timeout=2.0):
        raise OSError("down")

    monkeypatch.setattr(commands_hook, "http_ok", dead)
    assert cli.main(["hook-sync", KB]) == 0  # touched, same bytes: a hash compare, no health checks


# --- retry-failed ----------------------------------------------------------------------------------------

def seed(env, entries):
    env.state.ensure()
    env.state.save_failed(entries)


def entry(due, attempts=1, sha="x"):
    return {"sha256": sha, "error": "e", "attempts": attempts, "nextRetryAt": due}


def test_retry_syncs_a_due_entry_and_leaves_a_not_yet_due_one(env):
    put(env.root, L2, LEDGER)
    seed(env, {L: entry(NOW - 1), L2: entry(NOW + 500)})
    assert cli.main(["retry-failed", "--confirm", "coordination"]) == 0
    assert L in env.graph.sources and L2 not in env.graph.sources
    assert list(failed(env)) == [L2] and env.state.retry_due.read_text().strip() == str(int(NOW + 500))


def test_retry_that_fails_again_backs_off_further(env, monkeypatch):
    seed(env, {L: entry(NOW - 1, attempts=2, sha=sha_of(env.root / L))})
    down(monkeypatch)
    assert cli.main(["retry-failed"]) == 0
    assert failed(env)[L]["attempts"] == 3 and failed(env)[L]["nextRetryAt"] == NOW + 240


def sha_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_retry_drops_an_entry_whose_file_is_gone_without_touching_the_graph(env):
    seed(env, {"a/docs/plans/gone-coordination.md": entry(NOW - 1)})
    env.graph.sources["a/docs/plans/gone-coordination.md"] = ("ledger", {"sha256": "s"}, [])
    assert cli.main(["retry-failed"]) == 0
    assert failed(env) == {} and env.graph.writes == []
    assert "a/docs/plans/gone-coordination.md" in env.graph.sources


def test_retry_drops_an_entry_whose_hash_already_equals_the_graphs(env):
    sync.sync_paths(env.graph, sync.Parser.for_root(env.root), [L])
    env.graph.writes.clear()
    seed(env, {L: entry(NOW - 1)})
    assert cli.main(["retry-failed"]) == 0
    assert failed(env) == {} and env.graph.writes == []


def test_retry_merges_the_shells_pending_paths_and_retries_them_at_once(env):
    env.state.ensure()
    env.state.pending.write_text(f"{L}\n")
    assert cli.main(["retry-failed"]) == 0
    assert L in env.graph.sources and failed(env) == {} and env.state.pending.read_text() == ""


def test_a_second_retry_failed_exits_at_once_while_one_holds_the_lock(env):
    seed(env, {L: entry(NOW - 1)})
    env.state.ensure()
    with open(env.state.retry_lock, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        assert cli.main(["retry-failed"]) == 0
    assert env.graph.writes == [] and L in failed(env)


def test_retry_leaves_a_kb_section_still_inside_its_finish_cap_to_kb_finish(kb):
    seed(kb.env, {KB: entry(NOW - 1)})
    kb.chat.hold_new = True
    # a section pending since 1 minute ago on the mirror's clock
    kb.mirror.sections["s"] = {"key": "s", "path": KB, "heading": "Alpha", "state": "pending",
                               "pendingSince": kb.mirror.now - 60_000}
    assert cli.main(["retry-failed"]) == 0
    e = failed(kb.env)[KB]
    assert kb.chat.posts() == [] and e["nextRetryAt"] == pytest.approx(NOW + (kbsync.FINISH_CAP_MS - 60_000) / 1000)


# --- the confirmation ------------------------------------------------------------------------------------

def test_the_flag_is_what_reaches_the_connection(env):
    cli.main(["hook-sync", "--confirm", "coordination", L])
    cli.main(["retry-failed", "--confirm", "coordination"])
    assert {o[3] for o in env.opened} == {"coordination"}


def test_hook_sync_and_retry_failed_are_routine_so_the_env_alone_also_authorizes(env, monkeypatch):
    monkeypatch.setenv("COORDGRAPH_SHARED_CONFIRM", "coordination")
    cli.main(["hook-sync", L])
    assert env.opened[-1][3] == "coordination"


def test_a_wrong_confirmation_is_refused_by_the_real_guard_before_any_connection(env, monkeypatch, capsys):
    monkeypatch.undo()  # the real store.open_graph, the real ping replaced below
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(env.root))
    monkeypatch.setenv("COORDGRAPH_STATE_DIR", str(env.root / "state"))
    monkeypatch.setattr(commands_hook, "ping", lambda host, port, timeout=1.0: None)
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **k: pytest.fail("connected"))
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("socket"))
    assert cli.main(["hook-sync", "--confirm", "other", L]) == 2
    assert "must equal the graph key" in capsys.readouterr().err


# --- the hook path stays offline (plan3 test 12) -----------------------------------------------------------

def test_the_hook_modules_import_neither_identity_nor_wilib_nor_falkordb():
    code = ("import sys\nimport coordgraph.commands_hook, coordgraph.hookstate, coordgraph.cli\n"
            "bad = [m for m in ('coordgraph.identity', 'wilib', 'falkordb') if m in sys.modules]\n"
            "assert not bad, bad\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parents[1],
                         env={"PYTHONPATH": ".", "PATH": "", "COORDGRAPH_WI_ROOT": "/nonexistent"})
    assert out.returncode == 0, out.stderr


def test_hook_sync_with_a_bogus_work_items_root_still_mirrors_the_file(env, monkeypatch):
    monkeypatch.setenv("COORDGRAPH_WI_ROOT", "/nonexistent")
    assert cli.main(["hook-sync", L]) == 0 and L in env.graph.sources
