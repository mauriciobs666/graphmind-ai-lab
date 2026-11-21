"""Offline tests of the deliberate shared-instance path (S8-pre): `store.connect_shared` / `open_graph`,
`commands_store.confirmation`, the KB URL counterpart, and the CLI flag.

Nothing here can reach a database: `falkordb.FalkorDB` is replaced by a recorder and every socket connect
raises, so a refusal is observable as "the constructor was never called"."""
import socket
from dataclasses import replace
from types import SimpleNamespace

import falkordb
import pytest

from coordgraph import cli, commands_kb, commands_store, store
from coordgraph.config import Config

KEY = "coordination"
ENV_CLEAN = ("COORDGRAPH_EPHEMERAL_PORT", "COORDGRAPH_SHARED_CONFIRM", "COORDGRAPH_FALKORDB_HOST",
             "COORDGRAPH_FALKORDB_PORT", "COORDGRAPH_KEY", "COORDGRAPH_KB_URL")


class FakeFalkorDB:
    built: list[tuple[str, int]] = []

    def __init__(self, host, port):
        FakeFalkorDB.built.append((host, port))

    def select_graph(self, key):
        return ("graph", key)


@pytest.fixture(autouse=True)
def sealed(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a socket connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(falkordb, "FalkorDB", FakeFalkorDB)
    for name in ENV_CLEAN:
        monkeypatch.delenv(name, raising=False)
    FakeFalkorDB.built = []


# --- connect_shared ----------------------------------------------------------------------------

def test_connect_shared_reaches_the_shared_port_when_confirm_equals_the_key():
    db, graph = store.connect_shared("localhost", 6379, KEY, KEY, {})
    assert FakeFalkorDB.built == [("localhost", 6379)] and graph == ("graph", KEY)


@pytest.mark.parametrize("confirm", [None, "", "coordinatio", "coordination ", "Coordination", "other",
                                     "ws:agent-team"])
def test_connect_shared_refuses_a_confirm_that_is_not_exactly_the_key(confirm):
    with pytest.raises(store.RefusedEndpoint, match="must equal the graph key"):
        store.connect_shared("localhost", 6379, KEY, confirm, {})
    assert FakeFalkorDB.built == []


def test_connect_shared_compares_against_the_key_it_selects_not_a_default():
    with pytest.raises(store.RefusedEndpoint):
        store.connect_shared("localhost", 6379, KEY, "other-key", {})  # the confirm names a different graph
    assert FakeFalkorDB.built == []


@pytest.mark.parametrize("key", ["cpg_falkorchat", "kaizen_team", "ws:agent-team", "coordination2", "Coordination"])
def test_connect_shared_refuses_a_matching_confirm_for_a_foreign_graph(key):
    with pytest.raises(store.RefusedEndpoint, match="coordination mirror"):
        store.connect_shared("localhost", 6379, key, key, {})
    assert FakeFalkorDB.built == []


@pytest.mark.parametrize("port", [12345, 45001, 16390, 8200])
def test_connect_shared_refuses_a_port_that_is_not_a_shared_one(port):
    with pytest.raises(store.RefusedEndpoint, match="not a shared"):
        store.connect_shared("localhost", port, KEY, KEY, {})
    assert FakeFalkorDB.built == []


@pytest.mark.parametrize("port", sorted(store.SHARED_PORTS))
def test_every_shared_port_is_reachable_in_shared_mode_only(port):
    store.connect_shared("localhost", port, KEY, KEY, {})
    with pytest.raises(store.RefusedEndpoint):
        store.connect("localhost", port, KEY, {"COORDGRAPH_EPHEMERAL_PORT": str(port)})


def test_connect_shared_refuses_an_empty_key_even_with_an_empty_confirm():
    with pytest.raises(store.RefusedEndpoint):
        store.connect_shared("localhost", 6379, "", "", {})


def test_connect_shared_is_loopback_only():
    with pytest.raises(store.RefusedEndpoint, match="non-loopback"):
        store.connect_shared("10.0.0.5", 6379, KEY, KEY, {})
    assert FakeFalkorDB.built == []


def test_connect_shared_refuses_while_an_ephemeral_port_is_declared():
    with pytest.raises(store.RefusedEndpoint, match="pick one"):
        store.connect_shared("localhost", 6379, KEY, KEY, {"COORDGRAPH_EPHEMERAL_PORT": "45001"})
    assert FakeFalkorDB.built == []


# --- open_graph: ephemeral stays the default ---------------------------------------------------

def test_the_default_path_still_refuses_the_shared_port_even_with_a_matching_declaration():
    for env in ({}, {"COORDGRAPH_EPHEMERAL_PORT": "6379"}):
        with pytest.raises(store.RefusedEndpoint, match="shared FalkorDB port|set COORDGRAPH_EPHEMERAL_PORT"):
            store.open_graph("localhost", 6379, KEY, None, env)
    assert FakeFalkorDB.built == []


def test_the_default_path_reaches_only_a_declared_ephemeral_port():
    store.open_graph("127.0.0.1", 45001, KEY, None, {"COORDGRAPH_EPHEMERAL_PORT": "45001"})
    assert FakeFalkorDB.built == [("127.0.0.1", 45001)]


def test_a_blank_confirm_does_not_fall_back_to_the_ephemeral_path():
    with pytest.raises(store.RefusedEndpoint, match="must equal the graph key"):
        store.open_graph("localhost", 6379, KEY, "", {})
    assert FakeFalkorDB.built == []


def test_a_confirm_on_a_scratch_port_is_refused_the_modes_are_exact_complements():
    with pytest.raises(store.RefusedEndpoint):
        store.open_graph("127.0.0.1", 45001, KEY, KEY, {})
    assert FakeFalkorDB.built == []


# --- confirmation(): flag, env, and which commands accept the env ------------------------------

def cfg_with(**env):
    return Config.from_env({"COORDGRAPH_REPO_ROOT": "/tmp", **env})


def test_no_flag_and_no_env_is_ephemeral_mode():
    assert commands_store.confirmation(SimpleNamespace(), cfg_with()) is None


def test_the_flag_wins_over_the_env_and_an_empty_flag_stays_a_confirmation():
    cfg = cfg_with(COORDGRAPH_SHARED_CONFIRM="from-env")
    assert commands_store.confirmation(SimpleNamespace(confirm="from-flag"), cfg) == "from-flag"
    assert commands_store.confirmation(SimpleNamespace(confirm=""), cfg) == ""


def test_the_env_confirms_routine_commands():
    assert commands_store.confirmation(SimpleNamespace(), cfg_with(COORDGRAPH_SHARED_CONFIRM=KEY)) == KEY


def test_a_blank_env_is_no_confirmation():
    assert cfg_with(COORDGRAPH_SHARED_CONFIRM="  ").shared_confirm is None


def test_a_flag_only_command_refuses_the_env_alone_but_takes_the_flag():
    cfg = cfg_with(COORDGRAPH_SHARED_CONFIRM=KEY)
    with pytest.raises(store.RefusedEndpoint, match="--confirm"):
        commands_store.confirmation(SimpleNamespace(), cfg, flag_only=True)
    assert commands_store.confirmation(SimpleNamespace(confirm=KEY), cfg, flag_only=True) == KEY
    assert commands_store.confirmation(SimpleNamespace(), cfg_with(), flag_only=True) is None


# --- the commands -------------------------------------------------------------------------------

def test_a_routine_command_with_the_flag_targets_the_shared_instance_and_the_named_graph():
    _cfg, key, _root, _db, graph = commands_store._target(SimpleNamespace(confirm=KEY))
    assert (key, graph, FakeFalkorDB.built) == (KEY, ("graph", KEY), [("localhost", 6379)])


def test_a_routine_command_with_only_the_env_targets_the_shared_instance(monkeypatch):
    monkeypatch.setenv("COORDGRAPH_SHARED_CONFIRM", KEY)
    commands_store._target(SimpleNamespace())
    assert FakeFalkorDB.built == [("localhost", 6379)]


@pytest.mark.parametrize("graph", ["cpg_falkorchat", "kaizen_team"])
def test_a_command_cannot_be_pointed_at_a_foreign_graph_by_confirming_it(graph, capsys):
    assert cli.main(["sync", "--all", "--graph", graph, "--confirm", graph]) == 1
    assert "coordination mirror" in capsys.readouterr().err and FakeFalkorDB.built == []


def test_a_confirmed_arbitrary_loopback_port_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("COORDGRAPH_FALKORDB_PORT", "12345")
    assert cli.main(["drift", "--confirm", KEY]) == 1
    assert "not a shared" in capsys.readouterr().err and FakeFalkorDB.built == []


def test_a_routine_command_with_a_wrong_flag_or_graph_mismatch_never_connects(monkeypatch):
    with pytest.raises(store.RefusedEndpoint):
        commands_store._target(SimpleNamespace(confirm="wrong"))
    with pytest.raises(store.RefusedEndpoint):
        commands_store._target(SimpleNamespace(confirm="scratch", graph=KEY))  # --confirm names another key
    monkeypatch.setenv("COORDGRAPH_SHARED_CONFIRM", KEY)
    with pytest.raises(store.RefusedEndpoint):
        commands_store._target(SimpleNamespace(graph="scratch"))  # env names coordination, --graph another
    assert FakeFalkorDB.built == []


def test_without_any_confirmation_a_command_on_the_default_port_is_refused(capsys):
    assert cli.main(["sync", "--all"]) == 1
    assert "refusing" in capsys.readouterr().err and FakeFalkorDB.built == []


@pytest.mark.parametrize("argv", [["bootstrap"], ["rebuild"], ["kb-reconcile"]])
def test_first_load_rebuild_and_kb_reseed_refuse_the_env_alone(monkeypatch, capsys, argv):
    monkeypatch.setenv("COORDGRAPH_SHARED_CONFIRM", KEY)
    assert cli.main(argv) == 1
    assert "--confirm" in capsys.readouterr().err and FakeFalkorDB.built == []


@pytest.mark.parametrize("argv", [["bootstrap", "--confirm", "wrong"], ["rebuild", "--confirm", "wrong"],
                                  ["kb-reconcile", "--confirm", "wrong"], ["kb-sync", "--confirm", "wrong"],
                                  ["kb-status", "--confirm", "wrong"], ["kb-finish", "--confirm", "wrong"],
                                  ["drift", "--confirm", "wrong"], ["query", "RETURN 1", "--confirm", "wrong"],
                                  ["inflight", "--confirm", "wrong"], ["unit", "x", "--confirm", "wrong"],
                                  ["planned", "x", "--confirm", "wrong"], ["propose", "x", "--confirm", "wrong"],
                                  ["sync", "--confirm", "wrong"]])
def test_every_store_backed_command_takes_the_flag_and_refuses_a_wrong_one(capsys, argv):
    assert cli.main(argv) == 1
    assert "must equal the graph key" in capsys.readouterr().err and FakeFalkorDB.built == []


def test_bootstrap_with_the_flag_selects_the_shared_graph(monkeypatch):
    seen = []
    monkeypatch.setattr(store, "bootstrap", lambda db, graph, key, **k: seen.append((graph, key)) or
                        store.BootstrapResult([], []))
    assert cli.main(["bootstrap", "--confirm", KEY]) == 0
    assert seen == [(("graph", KEY), KEY)] and FakeFalkorDB.built == [("localhost", 6379)]


# --- the KB endpoint counterpart ---------------------------------------------------------------

@pytest.mark.parametrize("url", ["http://localhost:8200", "http://127.0.0.1:8200/"])
def test_the_shared_mirror_accepts_the_real_falkor_chat(url):
    store.require_shared_kb_url(url)


@pytest.mark.parametrize("url", ["http://localhost:18200", "http://localhost", "http://falkor-chat.example:8200",
                                 "http://10.0.0.5:8200"])
def test_the_shared_mirror_refuses_a_scratch_or_remote_falkor_chat(url):
    with pytest.raises(store.RefusedEndpoint):
        store.require_shared_kb_url(url)


def test_the_ephemeral_mirror_still_refuses_the_real_falkor_chat():
    with pytest.raises(store.RefusedEndpoint):
        store.require_ephemeral_kb_url("http://localhost:8200")


def _engine_cfg(confirm, url):
    return replace(Config.from_env({"COORDGRAPH_REPO_ROOT": "/tmp", "COORDGRAPH_KB_URL": url}), shared_confirm=confirm)


DB = SimpleNamespace(select_graph=lambda key: ("kb", key))


def test_the_kb_engine_checks_the_endpoint_for_the_mirror_mode():
    commands_kb._engine(_engine_cfg(KEY, "http://localhost:8200"), DB, None)  # shared mirror, real falkor-chat
    commands_kb._engine(_engine_cfg(None, "http://localhost:18200"), DB, None)  # ephemeral mirror, scratch
    with pytest.raises(store.RefusedEndpoint):
        commands_kb._engine(_engine_cfg(KEY, "http://localhost:18200"), DB, None)  # shared mirror, scratch KB
    with pytest.raises(store.RefusedEndpoint):
        commands_kb._engine(_engine_cfg(None, "http://localhost:8200"), DB, None)  # scratch mirror, real KB


def test_build_puts_the_effective_confirmation_on_the_cfg(monkeypatch):
    monkeypatch.setenv("COORDGRAPH_SHARED_CONFIRM", KEY)
    cfg, _root, _db, _graph = commands_kb._build(SimpleNamespace())
    assert cfg.shared_confirm == KEY
    monkeypatch.delenv("COORDGRAPH_SHARED_CONFIRM")
    monkeypatch.setenv("COORDGRAPH_EPHEMERAL_PORT", "45001")
    monkeypatch.setenv("COORDGRAPH_FALKORDB_PORT", "45001")
    assert commands_kb._build(SimpleNamespace())[0].shared_confirm is None


def test_a_detached_finish_inherits_the_flag_and_the_graph(tmp_path):
    cfg = Config.from_env({"COORDGRAPH_STATE_DIR": str(tmp_path / "s")})
    calls = []
    commands_kb.detach_finish(cfg, KEY, KEY, popen=lambda argv, **kw: calls.append(argv))
    assert calls[0][-5:] == ["kb-finish", "--confirm", KEY, "--graph", KEY]
    commands_kb.detach_finish(cfg, popen=lambda argv, **kw: calls.append(argv))
    assert calls[1][-1] == "kb-finish"


def test_kb_sync_hands_its_confirm_and_graph_to_the_detached_finish(monkeypatch, tmp_path):
    launched = []
    out = commands_kb.kbsync.KbOutcome("kb/a.md", "H", "pending", "")
    cfg = Config.from_env({"COORDGRAPH_REPO_ROOT": str(tmp_path)})
    monkeypatch.setattr(commands_kb, "_build", lambda args: (cfg, tmp_path, None, None))
    monkeypatch.setattr(commands_kb, "_engine", lambda *a: SimpleNamespace(
        read_all=lambda: ([], 0), sync_file=lambda doc, ctx, rows: [out]))
    monkeypatch.setattr(commands_kb, "load_docs", lambda root: ([SimpleNamespace(path="kb/a.md")], []))
    monkeypatch.setattr(commands_kb.kbsync.Context, "build", classmethod(lambda cls, docs, rows: None))
    monkeypatch.setattr(commands_kb, "detach_finish", lambda cfg, confirm, graph: launched.append((confirm, graph)))
    assert cli.main(["kb-sync", "--detach-finish", "--confirm", "C", "--graph", "G"]) == 2
    assert launched == [("C", "G")]  # distinct values: a swapped pair is caught
    launched.clear()
    assert cli.main(["kb-sync", "--detach-finish"]) == 2
    assert launched == [(None, None)]
