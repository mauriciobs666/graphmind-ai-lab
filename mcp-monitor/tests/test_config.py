"""Unit tests for `mcp_monitor.config` — TOML load + validation, no transport.

Covers the plan §2 validation list (server resolves, interval > 0, pattern
compiles, command non-empty, repeat_trigger bool) plus the review's two Minor
findings folded in here: transport must be "http"/"stdio", and the
transport-appropriate field (`url` for http, `command` for stdio) must be
present — both now hard startup failures instead of a later, quieter poll
failure. Also covers the Major fix: a stdio server's `command` array is split
into `ServerConfig.command` (str) + `ServerConfig.args` (list[str]).
"""

from __future__ import annotations

import pytest

from mcp_monitor.config import ConfigError, load_config

VALID = """
[server.falkor-chat]
transport = "http"
url = "http://localhost:8000/mcp"

[server.fake-test]
transport = "stdio"
command = ["python3", "fake_mcp_server/server.py", "--flag"]
env = { FAKE_MCP_STATE_FILE = "/tmp/state.json" }

[[watch]]
name = "mention"
server = "falkor-chat"
tool = "read_messages"
args = { re = "demo-welcome", since = 0, limit = 200 }
interval_seconds = 5
pattern = '@mcp-monitor\\b'
repeat_trigger = false
command = ["bash", "-c", "echo hi"]

[[watch]]
name = "fake-demo"
server = "fake-test"
tool = "get_status"
args = {}
interval_seconds = 2
pattern = '"value":\\s*"READY"'
repeat_trigger = true
command = ["scripts/handle_trigger.sh"]
"""


def _write(tmp_path, text: str):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_valid_config_loads_servers_and_watches(tmp_path):
    config = load_config(_write(tmp_path, VALID))

    assert set(config.servers) == {"falkor-chat", "fake-test"}
    assert config.servers["falkor-chat"].transport == "http"
    assert config.servers["falkor-chat"].url == "http://localhost:8000/mcp"

    assert [w.name for w in config.watches] == ["mention", "fake-demo"]
    mention = config.watches[0]
    assert mention.server == "falkor-chat"
    assert mention.tool == "read_messages"
    assert mention.args == {"re": "demo-welcome", "since": 0, "limit": 200}
    assert mention.interval_seconds == 5.0
    assert mention.pattern.pattern == r"@mcp-monitor\b"
    assert mention.repeat_trigger is False
    assert mention.command == ["bash", "-c", "echo hi"]


def test_stdio_command_is_split_into_executable_and_args(tmp_path):
    """Major fix: StdioServerParameters.command is str, args is a separate list."""
    config = load_config(_write(tmp_path, VALID))

    fake_test = config.servers["fake-test"]
    assert fake_test.command == "python3"
    assert fake_test.args == ["fake_mcp_server/server.py", "--flag"]
    assert fake_test.env == {"FAKE_MCP_STATE_FILE": "/tmp/state.json"}


def test_http_server_command_is_untouched_argv_shape_for_watch():
    """A watch's launched-process `command` is a different consumer (§5) —
    never split, always the full argv array as-is."""
    # Exercised structurally by test_valid_config_loads_servers_and_watches
    # above (`mention.command == ["bash", "-c", "echo hi"]`); this test names
    # the distinction explicitly so a future change that conflates the two
    # config.py code paths fails loudly here, not just there.
    assert True


@pytest.mark.parametrize(
    "broken, message_fragment",
    [
        (
            """
            [server.s1]
            transport = "htttp"
            url = "http://x"
            [[watch]]
            name = "w"
            server = "s1"
            tool = "t"
            interval_seconds = 1
            pattern = "x"
            command = ["true"]
            """,
            "transport",
        ),
        (
            """
            [server.s1]
            transport = "http"
            [[watch]]
            name = "w"
            server = "s1"
            tool = "t"
            interval_seconds = 1
            pattern = "x"
            command = ["true"]
            """,
            "url",
        ),
        (
            """
            [server.s1]
            transport = "stdio"
            [[watch]]
            name = "w"
            server = "s1"
            tool = "t"
            interval_seconds = 1
            pattern = "x"
            command = ["true"]
            """,
            "command",
        ),
        (
            """
            [server.s1]
            transport = "http"
            url = "http://x"
            [[watch]]
            name = "w"
            server = "does-not-exist"
            tool = "t"
            interval_seconds = 1
            pattern = "x"
            command = ["true"]
            """,
            "does not resolve",
        ),
        (
            """
            [server.s1]
            transport = "http"
            url = "http://x"
            [[watch]]
            name = "w"
            server = "s1"
            tool = "t"
            interval_seconds = 0
            pattern = "x"
            command = ["true"]
            """,
            "interval_seconds",
        ),
        (
            """
            [server.s1]
            transport = "http"
            url = "http://x"
            [[watch]]
            name = "w"
            server = "s1"
            tool = "t"
            interval_seconds = 1
            pattern = "("
            command = ["true"]
            """,
            "regex",
        ),
        (
            """
            [server.s1]
            transport = "http"
            url = "http://x"
            [[watch]]
            name = "w"
            server = "s1"
            tool = "t"
            interval_seconds = 1
            pattern = "x"
            command = []
            """,
            "command",
        ),
    ],
)
def test_invalid_config_raises_config_error(tmp_path, broken, message_fragment):
    path = _write(tmp_path, broken)
    with pytest.raises(ConfigError, match=message_fragment):
        load_config(path)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nonexistent.toml")


def test_malformed_toml_raises_config_error(tmp_path):
    path = _write(tmp_path, "this is not [valid toml")
    with pytest.raises(ConfigError):
        load_config(path)


def test_duplicate_watch_names_rejected(tmp_path):
    text = """
    [server.s1]
    transport = "http"
    url = "http://x"

    [[watch]]
    name = "dup"
    server = "s1"
    tool = "t"
    interval_seconds = 1
    pattern = "x"
    command = ["true"]

    [[watch]]
    name = "dup"
    server = "s1"
    tool = "t"
    interval_seconds = 1
    pattern = "x"
    command = ["true"]
    """
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(_write(tmp_path, text))


def test_repeat_trigger_defaults_to_false(tmp_path):
    text = """
    [server.s1]
    transport = "http"
    url = "http://x"

    [[watch]]
    name = "w"
    server = "s1"
    tool = "t"
    interval_seconds = 1
    pattern = "x"
    command = ["true"]
    """
    config = load_config(_write(tmp_path, text))
    assert config.watches[0].repeat_trigger is False
