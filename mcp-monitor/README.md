# mcp-monitor

A standalone, generic watcher: poll a configured MCP tool on an interval, and when its result
matches a configurable regular expression, automatically launch a configured command line — so
nobody has to watch MCP tool output by hand and react manually.

The driving scenario is `falkor-chat`: today, waking up a headless agent CLI when it's
`@mention`-ed in a thread requires a human to notice and act. mcp-monitor polls falkor-chat's
message-reading MCP tool, matches an `@mention` pattern, and launches a headless CLI when it sees
one — with zero changes to falkor-chat itself. But mcp-monitor is not falkor-chat-specific: it is
its own MCP *client*, and any `[[watch]]` can point at any MCP server/tool.

Design: `docs/plans/mcp-monitor.md`. Requirements: `docs/requirements/mcp-monitor.md`.

## Quick start

```bash
mcp-monitor/setup.sh                                  # create .venv, install the package + dev extra
cp mcp-monitor/config.example.toml mcp-monitor/config.toml
$EDITOR mcp-monitor/config.toml                       # point it at your own servers/watches
mcp-monitor/run.sh --config mcp-monitor/config.toml
```

`run.sh` execs `python -m mcp_monitor` from the venv. Logs go to stdout, one line per event,
tagged `watch=`/`server=`/`tool=` — `grep watch=<name>` isolates one watch's whole history.
`Ctrl-C` (or `SIGTERM`) cancels every watch's poll loop; already-launched commands are left
running (fire-and-forget by design — what a launched command does is not mcp-monitor's concern).

## How it works

- **One `[[watch]]` per independent poll/match/launch loop**, run concurrently for the process
  lifetime. Each watch's own polling is sequential (poll, sleep `interval_seconds`, poll again);
  launched commands run fully in parallel, across watches and across matches in the same poll.
- **One MCP connection per named `[server.*]` block**, shared by every watch that references it —
  two watches against the same server don't open two redundant sessions. A poll failure discards
  and lazily reopens the connection on the next call that needs it; the whole thing is guarded by
  a per-connection lock so two watches sharing a server never race on the same session object.
- **A match is every non-overlapping regex hit** in a poll's flattened result text, not just the
  first — a single poll can legitimately contain several new items.
- **A launched command receives the full match payload on stdin**, as one JSON document:

  ```json
  {
    "watch": "falkor-chat-mention",
    "server": "falkor-chat",
    "tool": "read_messages",
    "raw_result": "<the full flattened tool result text>",
    "matched_text": "@mcp-monitor",
    "matched_at": "2026-08-08T12:34:56Z"
  }
  ```

  Plus three environment variables for commands that don't want to parse JSON:
  `MCP_MONITOR_WATCH_NAME`, `MCP_MONITOR_SERVER_NAME`, `MCP_MONITOR_TOOL_NAME`. The command itself
  runs as an argv array (`asyncio.create_subprocess_exec(*command, ...)`), never a shell string —
  put `["bash", "-c", "..."]` in `command` yourself for shell features.
- **Repeat-trigger dedupe is per watch** (`repeat_trigger` in the config): `false` (default) fires
  a given matched substring at most once for the life of the process; `true` fires every
  occurrence, every time. Dedupe state is in-memory only — lost on restart (see
  `docs/BACKLOG.md`).
- **A poll failure (server unreachable, tool error) is logged and the watch keeps going** — it
  never stops a watch or crashes the process.

## Config schema

TOML, `[server.<name>]` connection blocks plus `[[watch]]` entries referencing them by name. Full
field-by-field reference in `config.example.toml` (copy it as your starting point). Summary:

| `[server.<name>]` field | Required for | Meaning |
|---|---|---|
| `transport` | always | `"http"` or `"stdio"` |
| `url` | `transport = "http"` | the server's Streamable-HTTP endpoint |
| `command` | `transport = "stdio"` | argv of the server process, e.g. `["python3", "server.py"]` |
| `env` | optional | extra environment variables for this connection |

| `[[watch]]` field | Meaning |
|---|---|
| `name` | identifier passed to the launched command |
| `server` | must match a `[server.*]` block name |
| `tool` | the MCP tool to call on that server |
| `args` | static arguments passed to every call (`{}` for none) |
| `interval_seconds` | must be `> 0` |
| `pattern` | a Python regex checked against the poll's flattened result text |
| `repeat_trigger` | `false` (default) or `true` — see above |
| `command` | argv of the command to launch on a match |

A config error (bad transport, unresolved `watch.server`, a pattern that doesn't compile, etc.) is
a **hard startup failure** with a message naming the offending watch/server — mcp-monitor never
starts polling with a partially-valid config.

## The fake test/demo MCP server

`fake_mcp_server/` is a minimal stdio MCP server built for mcp-monitor's own automated tests and
demos (FR-12) — not an existing repo component, and deliberately a **different transport** than
falkor-chat's HTTP, to prove mcp-monitor's client layer is transport-agnostic, not just
tool-agnostic. Its one tool, `get_status`, returns whatever was last written to a small JSON state
file:

```bash
mcp-monitor/.venv/bin/python fake_mcp_server/set_state.py READY   # next get_status() poll sees it
mcp-monitor/.venv/bin/python fake_mcp_server/set_state.py idle
```

`config.example.toml`'s `[server.fake-test]` block + `fake-server-demo` watch shows it wired up.

## Testing

Three automated tiers (`docs/plans/mcp-monitor.md` §11) plus one manual runbook:

```bash
mcp-monitor/.venv/bin/pytest mcp-monitor/tests -q
```

| Tier | File(s) | What |
|---|---|---|
| 1. Unit | `test_config.py`, `test_matcher.py`, `test_launcher.py` | No network, no subprocess — config validation, regex/dedupe logic, stdin/env payload shape (subprocess exec stubbed). |
| 2. Component, in-memory | `test_watch_loop.py` | Full poll→match→dedupe→launch loop against `mcp.shared.memory.create_connected_server_and_client_session` (same technique `falkor-chat/server/tests/test_mcp_client.py` uses) — no real process/socket. |
| 3. Component, real subprocess | `test_fake_server_integration.py` | Real `fake_mcp_server/server.py` subprocess + real mcp-monitor watch loop, driven via `set_state.py`-equivalent state writes; asserts AC-4/AC-5/AC-6 via a marker file launched commands append to. |

**AC-3 (the live falkor-chat demo) is a manual runbook, not a pytest test** — no falkor-chat
server exists in the default test environment:

```bash
./falkor-chat/scripts/start_falkordb.sh -d
./falkor-chat/scripts/start_server.sh
mcp-monitor/scripts/demo_falkor_chat.sh
```

The script starts mcp-monitor against a throwaway config watching falkor-chat's seeded
`demo-welcome` thread, posts a message containing `@mcp-monitor` via `send_message`, and reports
PASS/FAIL on whether the watch fired within 30s.

## Files

```
mcp_monitor/            the package — config, client, watch loop, launcher, logging
fake_mcp_server/         FR-12's minimal stdio test/demo server + its state-flip CLI
tests/                   the three automated tiers above
scripts/demo_falkor_chat.sh   the AC-3 manual runbook
config.example.toml       the config schema, documented — copy-and-edit starting point
setup.sh / run.sh         host-venv create/refresh, and the launcher (mirrors cpg/mcp/)
```

See `AGENTS.md` for architecture pointers aimed at an agent working in this code, and
`docs/plans/mcp-monitor.md` for the full design rationale (why TOML, why one `asyncio.Task` per
watch, why stdin JSON, why the falkor-chat demo uses a literal-text regex instead of the
`isMention` flag, etc.).
