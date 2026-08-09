"""TOML config load + validation (plan §2).

Loads `[server.*]` connection blocks and `[[watch]]` entries into dataclasses,
failing fast — a hard startup failure naming the offending watch/server, never
a partially-running process — before any polling starts.

Two consumers read a `command` array in this config, and they need **different
shapes**, per the plan review's Major finding: a `[server.*]` stdio block's
`command` is split here into an executable (`ServerConfig.command`, a `str`)
plus the rest (`ServerConfig.args`, a `list[str]`) for
`mcp.StdioServerParameters`, which types `command` as `str`. A `[[watch]]`'s
`command` stays an untouched argv array (`WatchConfig.command`) — that one is
consumed by `asyncio.create_subprocess_exec(*command, ...)`, which wants the
full argv. Do not conflate the two.
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Supported `[server.*].transport` values (review Minor finding (b)).
_TRANSPORTS = {"http", "stdio"}


class ConfigError(Exception):
    """A malformed config file — always a hard startup failure (plan §2)."""


@dataclass(frozen=True)
class ServerConfig:
    """One `[server.<name>]` block — a named MCP connection target."""

    name: str
    transport: str  # "http" | "stdio"
    url: str | None = None
    command: str | None = None  # stdio only — executable, already split from args
    args: list[str] = field(default_factory=list)  # stdio only — the rest of argv
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WatchConfig:
    """One `[[watch]]` entry — an independent poll/match/launch loop (plan §3)."""

    name: str
    server: str
    tool: str
    args: dict[str, Any]
    interval_seconds: float
    pattern: re.Pattern[str]
    repeat_trigger: bool
    command: list[str]  # argv for the launched process — unsplit, unlike ServerConfig.command


@dataclass(frozen=True)
class Config:
    servers: dict[str, ServerConfig]
    watches: list[WatchConfig]


def load_config(path: str | Path) -> Config:
    """Load and validate a TOML config file, raising `ConfigError` on any problem.

    Every check below runs before returning, so a config with several mistakes
    reports the first one found rather than partially constructing a `Config` —
    "hard startup failure" per §2, not a lazily-discovered runtime surprise.
    """
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc

    try:
        raw = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid UTF-8: {exc}") from exc

    servers = _load_servers(raw.get("server", {}))
    watches = _load_watches(raw.get("watch", []), servers)
    return Config(servers=servers, watches=watches)


def _load_servers(raw_servers: dict[str, Any]) -> dict[str, ServerConfig]:
    if not isinstance(raw_servers, dict):
        raise ConfigError("[server.*] must be a table of named server blocks")

    servers: dict[str, ServerConfig] = {}
    for name, block in raw_servers.items():
        servers[name] = _load_one_server(name, block)
    return servers


def _load_one_server(name: str, block: Any) -> ServerConfig:
    if not isinstance(block, dict):
        raise ConfigError(f"[server.{name}] must be a table")

    transport = block.get("transport")
    if transport not in _TRANSPORTS:
        raise ConfigError(
            f"[server.{name}].transport must be one of {sorted(_TRANSPORTS)}, got {transport!r}"
        )

    env = block.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise ConfigError(f"[server.{name}].env must be a table of string -> string")

    if transport == "http":
        url = block.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ConfigError(f"[server.{name}] has transport = \"http\" but no non-empty url")
        return ServerConfig(name=name, transport="http", url=url, env=dict(env))

    # transport == "stdio"
    command = block.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(c, str) and c for c in command
    ):
        raise ConfigError(
            f"[server.{name}] has transport = \"stdio\" but command is not a non-empty "
            "array of non-empty strings"
        )
    # Major fix: split the argv array into StdioServerParameters' two separate
    # fields (`command: str`, `args: list[str]`) here, once, at load time — not
    # at every connection open in client.py.
    executable, *rest = command
    return ServerConfig(
        name=name, transport="stdio", command=executable, args=rest, env=dict(env)
    )


def _load_watches(raw_watches: Any, servers: dict[str, ServerConfig]) -> list[WatchConfig]:
    if not isinstance(raw_watches, list):
        raise ConfigError("[[watch]] must be an array of tables")

    watches: list[WatchConfig] = []
    seen_names: set[str] = set()
    for i, block in enumerate(raw_watches):
        watch = _load_one_watch(i, block, servers)
        if watch.name in seen_names:
            raise ConfigError(f"duplicate watch name {watch.name!r} ([[watch]] #{i})")
        seen_names.add(watch.name)
        watches.append(watch)
    return watches


def _load_one_watch(index: int, block: Any, servers: dict[str, ServerConfig]) -> WatchConfig:
    if not isinstance(block, dict):
        raise ConfigError(f"[[watch]] #{index} must be a table")

    name = block.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"[[watch]] #{index} has no non-empty name")
    label = f"watch {name!r}"

    server = block.get("server")
    if not isinstance(server, str) or server not in servers:
        raise ConfigError(
            f"{label}: server {server!r} does not resolve to a [server.*] block "
            f"(known servers: {sorted(servers)})"
        )

    tool = block.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        raise ConfigError(f"{label}: tool must be a non-empty string")

    args = block.get("args", {})
    if not isinstance(args, dict):
        raise ConfigError(f"{label}: args must be a table")

    interval_seconds = block.get("interval_seconds")
    if not isinstance(interval_seconds, (int, float)) or isinstance(interval_seconds, bool):
        raise ConfigError(f"{label}: interval_seconds must be a number")
    if interval_seconds <= 0:
        raise ConfigError(f"{label}: interval_seconds must be > 0, got {interval_seconds}")

    pattern_str = block.get("pattern")
    if not isinstance(pattern_str, str):
        raise ConfigError(f"{label}: pattern must be a string")
    try:
        pattern = re.compile(pattern_str)
    except re.error as exc:
        raise ConfigError(
            f"{label}: pattern {pattern_str!r} does not compile as regex: {exc}"
        ) from exc

    repeat_trigger = block.get("repeat_trigger", False)
    if not isinstance(repeat_trigger, bool):
        raise ConfigError(f"{label}: repeat_trigger must be a bool")

    command = block.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(c, str) and c for c in command
    ):
        raise ConfigError(f"{label}: command must be a non-empty array of non-empty strings")

    return WatchConfig(
        name=name,
        server=server,
        tool=tool,
        args=dict(args),
        interval_seconds=float(interval_seconds),
        pattern=pattern,
        repeat_trigger=repeat_trigger,
        command=list(command),
    )


def _main() -> None:  # pragma: no cover — manual `python -m mcp_monitor.config` smoke check
    if len(sys.argv) != 2:
        print("usage: python -m mcp_monitor.config CONFIG.toml", file=sys.stderr)
        raise SystemExit(2)
    config = load_config(sys.argv[1])
    print(f"servers: {sorted(config.servers)}")
    for w in config.watches:
        print(f"  watch {w.name!r}: server={w.server} tool={w.tool} pattern={w.pattern.pattern!r}")


if __name__ == "__main__":  # pragma: no cover
    _main()
