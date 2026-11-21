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
        store.connect_shared("localhost", 6379, "other-key", KEY, {})  # the confirm names a different graph
    store.connect_shared("localhost", 6379, "other-key", "other-key", {})
    assert FakeFalkorDB.built == [("localhost", 6379)]


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


def test_a_given_confirm_never_reaches_the_ephemeral_guard_so_an_ephemeral_port_needs_no_confirm_either():
    # one mode or the other: a confirm on a scratch port is the shared path (which does not check the port)
    store.open_graph("127.0.0.1", 45001, KEY, KEY, {})
    assert FakeFalkorDB.built == [("127.0.0.1", 45001)]


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


def test_a_routine_command_with_a_wrong_flag_or_graph_mismatch_never_connects(monkeypatch):
    with pytest.raises(store.RefusedEndpoint):
        commands_store._target(SimpleNamespace(confirm="wrong"))
    with pytest.raises(store.RefusedEndpoint):
        commands_store._target(SimpleNamespace(confirm=KEY, graph="scratch"))  # --graph moved, confirm did not
    monkeypatch.setenv("COORDGRAPH_SHARED_CONFIRM", KEY)
    with pytest.raises(store.RefusedEndpoint):
        commands_store._target(SimpleNamespace(graph="scratch"))
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


def test_a_detached_finish_inherits_the_flag(tmp_path):
    cfg = Config.from_env({"COORDGRAPH_STATE_DIR": str(tmp_path / "s")})
    calls = []
    commands_kb.detach_finish(cfg, KEY, popen=lambda argv, **kw: calls.append(argv))
    assert calls[0][-3:] == ["kb-finish", "--confirm", KEY]
    commands_kb.detach_finish(cfg, popen=lambda argv, **kw: calls.append(argv))
    assert calls[1][-1] == "kb-finish"
