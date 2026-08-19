# mcp-monitor — agent working context

## Project in one sentence

A standalone, generic MCP tool-result watcher: poll a configured MCP tool on an interval, match
its result against a regex, launch a configured command line on match — see `README.md` for the
human-facing quickstart and `docs/plans/mcp-monitor.md` for the full design.

## Architecture pointers

- `mcp_monitor/config.py` — TOML load + validation into `Config`/`ServerConfig`/`WatchConfig`
  dataclasses. **Two different `command` shapes live in this file, on purpose**: a
  `[server.*]` stdio block's `command` array is split here into `ServerConfig.command` (`str`,
  the executable) + `ServerConfig.args` (`list[str]`, the rest) — the installed `mcp` SDK's
  `StdioServerParameters` wants those as two separate fields, not one argv array. A `[[watch]]`'s
  `command` stays untouched (`WatchConfig.command: list[str]`, full argv) because
  `asyncio.create_subprocess_exec(*command, ...)` (in `launcher.py`) wants the whole thing. Don't
  conflate the two if you touch either.
- `mcp_monitor/client.py` — `ServerConnection`: one lazily-opened session per named server, shared
  across every watch that references it, reused on success and discarded-then-lazily-reopened on
  failure. `_lock` (an `asyncio.Lock`) wraps the whole open/call/discard sequence so two watches
  polling the same shared connection as independent `asyncio.Task`s never race on the session
  object. `ToolCallError` is raised when a call completes but `CallToolResult.isError` is `True`
  (verified against the installed SDK: a tool body raising does **not** propagate client-side —
  FastMCP packages it as an error result instead) — this folds tool-level errors into the same
  poll-failure path as transport-level exceptions, without discarding an otherwise-healthy
  session. `session_cm_factory` is the test seam: pass
  `lambda: mcp.shared.memory.create_connected_server_and_client_session(stub)` directly for an
  in-memory test — its shape (an async CM yielding a connected, initialized `ClientSession`) is
  exactly what `ServerConnection` expects.
- `mcp_monitor/watch.py` — `run_watch()` is the per-`[[watch]]` infinite poll/match/dedupe loop.
  `find_matches()`/`filter_new()` are pulled out as pure functions (no transport, no asyncio)
  specifically so matcher/dedupe logic is unit-testable without a connection — see
  `tests/test_matcher.py`.
- `mcp_monitor/launcher.py` — `launch()` is the whole body of the un-awaited
  `asyncio.create_task(...)` a match schedules: spawns via the injectable
  `create_subprocess_exec` parameter (the test seam — stub it instead of monkeypatching
  `asyncio.create_subprocess_exec` globally), feeds the JSON payload on stdin, awaits completion,
  logs the exit code. A spawn `OSError` is logged at `ERROR` and swallowed, never raised — this
  runs unattached to anything that would catch it.
- `mcp_monitor/logging_setup.py` — one `LoggerAdapter` per watch, tagging every line
  `watch=`/`server=`/`tool=`. A `logging.Filter` fills in `-` for lines logged outside a watch's
  adapter (startup/shutdown), so the same formatter serves both without raising a `KeyError`.
- `mcp_monitor/__main__.py` — CLI (`--config PATH [--log-level LEVEL]`). `run(config, stop=...)`
  takes an injectable `asyncio.Event` seam so a caller (or a test) can bound a run without going
  through `SIGINT`/`SIGTERM`; production wires both signals to the same event.
- `fake_mcp_server/` — FR-12's purpose-built stdio server (one tool, `get_status`, reads a JSON
  state file) + `set_state.py` (flips it). Deliberately stdio where falkor-chat is HTTP, so the
  automated test suite proves the client layer is transport-agnostic, not merely tool-agnostic.
  Modeled on `cypher-mcp/server.py`'s mechanics (a `FastMCP` stdio server), not its content.

## Conventions

- **Host venv, no Docker.** `setup.sh`/`run.sh` mirror `cypher-mcp/`'s host-venv half exactly (create
  venv, install with dev extra, smoke-import). No container packaging — `docs/plans/mcp-monitor.md`
  §1 explains why (no Joern/JVM-toolchain concern to isolate, unlike `cypher-mcp`); it's a
  `docs/BACKLOG.md` item if a real deployment need shows up later.
- **TOML config, `tomllib`, no new dependency.** Literal strings (`pattern = '...'`) avoid
  backslash-doubling a regex the way a JSON string would need.
- **`anyio`'s pytest plugin, not `pytest-asyncio`.** `anyio` is already a transitive dependency of
  `mcp`; async tests use `@pytest.mark.anyio` + an `anyio_backend` fixture returning `"asyncio"`.
- **Test tiers mirror `docs/plans/mcp-monitor.md` §11** — unit (`test_config.py`,
  `test_matcher.py`, `test_launcher.py`), in-memory component (`test_watch_loop.py`, same
  technique as `falkor-chat/server/tests/test_mcp_client.py`), real-subprocess component
  (`test_fake_server_integration.py`). All three run in the default `pytest` invocation — nothing
  here needs a live server or a `live`-style marker. AC-3 (the falkor-chat demo) is the one
  exception: `scripts/demo_falkor_chat.sh` is a runbook, not a pytest test, because no falkor-chat
  server exists in the default test environment.
- **The falkor-chat AC-3 mechanism is a literal-text regex against the message body, not
  `isMention`.** `docs/plans/mcp-monitor.md` §8 has the full investigation (falkor-chat's
  `isMention` is keyed to one process-wide fixed actor, not an arbitrary watched name) — read that
  before changing anything about how the falkor-chat watch is configured or demoed.

## Commands

```bash
mcp-monitor/setup.sh                                    # create/refresh .venv
mcp-monitor/.venv/bin/pytest mcp-monitor/tests -q        # full suite, all three automated tiers
mcp-monitor/.venv/bin/ruff check mcp_monitor fake_mcp_server tests scripts
mcp-monitor/run.sh --config mcp-monitor/config.example.toml --log-level DEBUG
mcp-monitor/scripts/demo_falkor_chat.sh                  # AC-3 manual runbook (needs a live falkor-chat)
```

## Documentation map

`docs/requirements/mcp-monitor.md` (FR/AC/decision log) → `docs/plans/mcp-monitor.md` (design,
owner `architect`) → `docs/reviews/mcp-monitor.md` (analyst's review — Major/Moderate/Minor
findings, all folded into this implementation) → `docs/plans/mcp-monitor-coordination.md` (teco's
unit tracking). `docs/BACKLOG.md` and `docs/HISTORY.md` are the living logs (module documentation
convention, root `AGENTS.md`) — no header block, append-only.
