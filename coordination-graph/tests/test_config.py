from pathlib import Path

import pytest

from coordgraph.config import DEFAULT_EMBED_URL, LEASE_TTL_S, Config, ConfigError, package_repo_root


def test_defaults_with_an_empty_environment():
    c = Config.from_env({})
    assert c.repo_root == package_repo_root()
    assert (c.falkordb_host, c.falkordb_port, c.key) == ("localhost", 6379, "coordination")
    assert c.state_dir == c.repo_root / "coordination-graph" / ".state"
    assert c.kb_url == "http://localhost:8200"
    assert c.kb_graph == "ws:agent-team"
    assert c.embed_url == DEFAULT_EMBED_URL
    assert (c.kb_budget_s, c.kb_http_timeout_s, c.hook_disabled) == (10.0, 10.0, False)
    assert (c.kb_retry_max, c.kb_retry_backoff_s, c.kb_retry_run_cap) == (3, 1800.0, 4)


def test_package_repo_root_is_the_repo_holding_this_component():
    assert (package_repo_root() / "coordination-graph" / "coordgraph" / "config.py").is_file()


def test_every_value_is_env_overridable(tmp_path):
    env = {
        "COORDGRAPH_REPO_ROOT": str(tmp_path),
        "COORDGRAPH_FALKORDB_HOST": "db",
        "COORDGRAPH_FALKORDB_PORT": "6390",
        "COORDGRAPH_KEY": "scratch",
        "COORDGRAPH_STATE_DIR": str(tmp_path / "st"),
        "COORDGRAPH_KB_URL": "http://kb:1",
        "COORDGRAPH_KB_GRAPH": "ws:x",
        "COORDGRAPH_EMBED_URL": "http://e:2",
        "COORDGRAPH_KB_BUDGET_S": "3.5",
        "COORDGRAPH_KB_HTTP_TIMEOUT_S": "4",
        "COORDGRAPH_HOOK_DISABLED": "1",
        "COORDGRAPH_KB_RETRY_MAX": "5",
        "COORDGRAPH_KB_RETRY_BACKOFF_S": "60",
        "COORDGRAPH_KB_RETRY_RUN_CAP": "2",
    }
    c = Config.from_env(env)
    assert c.repo_root == tmp_path and c.state_dir == tmp_path / "st"
    assert (c.falkordb_host, c.falkordb_port, c.key) == ("db", 6390, "scratch")
    assert (c.kb_url, c.kb_graph, c.embed_url) == ("http://kb:1", "ws:x", "http://e:2")
    assert (c.kb_budget_s, c.kb_http_timeout_s, c.hook_disabled) == (3.5, 4.0, True)
    assert (c.kb_retry_max, c.kb_retry_backoff_s, c.kb_retry_run_cap) == (5, 60.0, 2)


def test_repo_root_falls_back_to_claude_project_dir_and_state_follows_it(tmp_path):
    c = Config.from_env({"CLAUDE_PROJECT_DIR": str(tmp_path)})
    assert c.repo_root == tmp_path
    assert c.state_dir == tmp_path / "coordination-graph" / ".state"
    explicit = Config.from_env({"COORDGRAPH_REPO_ROOT": "/a", "CLAUDE_PROJECT_DIR": "/b"})
    assert explicit.repo_root == Path("/a")


def test_embed_url_falls_back_to_the_falkor_chat_variable():
    c = Config.from_env({"FALKORCHAT_EMBEDDING_BASE_URL": "http://lm:1234/v1"})
    assert c.embed_url == "http://lm:1234/v1"
    c = Config.from_env({"COORDGRAPH_EMBED_URL": "http://own", "FALKORCHAT_EMBEDDING_BASE_URL": "x"})
    assert c.embed_url == "http://own"


def test_kb_http_timeout_must_stay_under_a_third_of_the_lease_ttl():
    Config.from_env({"COORDGRAPH_KB_HTTP_TIMEOUT_S": str(LEASE_TTL_S / 3 - 0.5)})
    for bad in (str(LEASE_TTL_S / 3), "60", "0", "-1"):
        with pytest.raises(ConfigError, match="TTL/3"):
            Config.from_env({"COORDGRAPH_KB_HTTP_TIMEOUT_S": bad})


def test_unparseable_numbers_name_the_variable():
    with pytest.raises(ConfigError, match="COORDGRAPH_FALKORDB_PORT"):
        Config.from_env({"COORDGRAPH_FALKORDB_PORT": "abc"})
    with pytest.raises(ConfigError, match="COORDGRAPH_KB_BUDGET_S"):
        Config.from_env({"COORDGRAPH_KB_BUDGET_S": "soon"})


@pytest.mark.parametrize("value,expected", [
    ("", False), ("0", False), ("false", False), ("False", False), ("no", False), ("off", False),
    ("garbage", False), ("1", True), ("yes", True), ("true", True), (" TRUE ", True), ("On", True),
])
def test_hook_disabled_flag(value, expected):
    assert Config.from_env({"COORDGRAPH_HOOK_DISABLED": value}).hook_disabled is expected
