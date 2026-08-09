# Change History — mcp-monitor

> Dated log of actual changes to the `mcp-monitor` component. Most recent first.

## 2026-08-09 — Fix shipped `config.example.toml` (QA Findings 1 & 2)

**What:** Follow-up fix for the two Minor, non-blocking defects `qa-engineer` found in
`config.example.toml` during the M1 QA pass (`docs/test-reports/mcp-monitor-report.md`, "Findings"
— verdict was Approve; neither blocked M1). Both defects meant the shipped "copy-and-edit starting
point" didn't actually run its `fake-server-demo` watch when followed unmodified.

- **Finding 1** — `[server.fake-test]`'s `command` used bare `python3`, which lacks the `mcp` SDK
  outside `mcp-monitor/.venv` (only `setup.sh` installs it there), so every poll failed with
  `ModuleNotFoundError: No module named 'mcp'`. Fixed by pointing `command` at the venv's own
  interpreter: `["mcp-monitor/.venv/bin/python", "mcp-monitor/fake_mcp_server/server.py"]` — paths
  written relative to the repo root, matching README.md's own Quick start (every command there is
  `mcp-monitor/`-prefixed, i.e. run from the repo root, not from inside `mcp-monitor/`). Added a
  comment block explaining that stdio `command` paths resolve against mcp-monitor's own process
  working directory (wherever `run.sh` was invoked from), not the config file's own directory.
- **Finding 2** — the `fake-server-demo` watch's `command = ["scripts/handle_trigger.sh"]`
  referenced a file that didn't exist anywhere in the repo, so every match failed to spawn. Added
  `mcp-monitor/scripts/handle_trigger.sh` — a minimal example launch command (mirrors the shape of
  `scripts/demo_falkor_chat.sh`'s inline `on_trigger.py`): reads the stdin JSON payload, prints a
  one-line summary plus the raw payload to stderr, demonstrating FR-5's stdin-JSON delivery
  concretely without needing a marker file (this is a documentation example, not an automated
  test's assertion target). Config's `command` updated to
  `["mcp-monitor/scripts/handle_trigger.sh"]`, matching the same repo-root-relative convention as
  Finding 1's fix.

**Verification:** `mcp-monitor/setup.sh` (already set up, refreshed cleanly) then
`mcp-monitor/run.sh --config mcp-monitor/config.example.toml --log-level INFO`, run twice from the
repo root (matching the documented invocation) — once unmodified (~18s, several poll cycles of
`fake-server-demo` at `interval_seconds = 2`), once with the fake server's state flipped to
`READY` mid-run via `fake_mcp_server/set_state.py` to also exercise the match→launch path.
Confirmed: no `ModuleNotFoundError` or spawn-failure `ERROR` line at any point in either run;
`fake_mcp_server/server.py` logged `Processing request of type CallToolRequest` (real successful
polls); once state was `READY`, each poll logged `match found: '"value": "READY"'` followed by
`handle_trigger.sh`'s stderr output and `command [...] exited with code 0`. The `falkor-chat-mention`
watch in the same config is unrelated to either finding and was not touched.
`mcp-monitor/.venv/bin/pytest mcp-monitor/tests -q` → 34 passed (unchanged baseline).
`mcp-monitor/.venv/bin/ruff check mcp_monitor fake_mcp_server tests scripts` → clean.

## 2026-08-08 — Initial delivery (M1)

**What:** Built `mcp-monitor` end-to-end from `docs/requirements/mcp-monitor.md` and
`docs/plans/mcp-monitor.md` (both already existing, `architect`-authored and `analyst`-reviewed —
`docs/reviews/mcp-monitor.md`), per the coordination doc's unit 3. Nothing existed before this
delivery except the three upstream docs.

- **`mcp_monitor/` package** — `config.py` (TOML load/validation → `Config`/`ServerConfig`/
  `WatchConfig` dataclasses), `client.py` (`ServerConnection`: per-server lazy-open/discard/reopen
  `ClientSession` lifecycle, `_content_to_text` result flattening), `watch.py` (per-watch
  poll/match/dedupe loop, pure `find_matches`/`filter_new` helpers), `launcher.py` (stdin-JSON +
  env-var subprocess launch, exit-code logging), `logging_setup.py` (per-watch `LoggerAdapter`),
  `__main__.py` (CLI: `--config PATH [--log-level LEVEL]`, signal-driven shutdown).
- **`fake_mcp_server/`** — FR-12's purpose-built stdio server (`server.py`, one tool
  `get_status`) + `set_state.py` (state-flip CLI), modeled on `cpg/mcp/server.py`'s mechanics.
- **`tests/`** — three automated tiers, all green in the default `pytest` run (34 passed): unit
  (`test_config.py`, `test_matcher.py`, `test_launcher.py`), in-memory component
  (`test_watch_loop.py`, `mcp.shared.memory.create_connected_server_and_client_session`, same
  technique as `falkor-chat/server/tests/test_mcp_client.py`), real-subprocess component
  (`test_fake_server_integration.py`, real `fake_mcp_server` process + real watch loop, asserting
  AC-4/AC-5/AC-6 via a marker file).
- **`scripts/demo_falkor_chat.sh`** — the AC-3 manual runbook (not pytest — no falkor-chat server
  in the default test environment). Needs a human/QA run against a live `falkor-chat` +
  `FalkorDB`; not executed as part of this delivery.
- **`pyproject.toml`/`setup.sh`/`run.sh`/`config.example.toml`** — host-venv packaging mirroring
  `cpg/mcp/`'s host-venv half; no Docker (plan §1).
- **Entry docs** — `README.md` (human) + `AGENTS.md` (agent), and this component registered in
  root `AGENTS.md`'s Structure section and Component docs table.
- **`docs/BACKLOG.md`** (new) — seeded with persistent-dedupe-state-across-restart, unbounded
  dedupe-set growth, optional Docker packaging, config hot-reload, and auth/hardening; cross-
  references (not duplicates) `kiro/docs/requirements/kiro-vision-followups.md` item 4 and
  `falkor-chat/docs/BACKLOG.md` K-018.

**Review findings folded in during implementation** (`docs/reviews/mcp-monitor.md`):

- **Major (mandatory)** — the plan's `StdioServerParameters(command=[...])` sketch does not match
  the installed `mcp` 1.28.x SDK (`command` is `str`, args are a separate `args: list[str]`).
  Fixed in `config.py`: a `[server.*]` stdio block's `command` array is split at load time into
  `ServerConfig.command` (executable, `str`) + `ServerConfig.args` (rest, `list[str]`) — verified
  against the real SDK and exercised end-to-end by `test_fake_server_integration.py` (a real
  subprocess connection over stdio). A watch's own launched-process `command` (§5, consumed by
  `asyncio.create_subprocess_exec(*command, ...)`) was correctly left as an untouched argv array —
  the two are different consumers with different shape needs, and this delivery keeps them
  separate rather than reusing one code path.
- **Moderate (recommended, done)** — `ServerConnection` now guards its entire
  open/call/discard-on-failure sequence with a per-connection `asyncio.Lock`, so two watches
  sharing a `[server.*]` block as independent `asyncio.Task`s never race on the same session
  object.
- **Minor (a)** — unbounded dedupe-set growth added to `docs/BACKLOG.md`.
- **Minor (b)** — load-time config validation now also checks `[server.*].transport` is
  `"http"`/`"stdio"` and that the transport-appropriate field (`url`/`command`) is present, turning
  a typo'd transport into the "hard startup failure" §2 always promised, instead of a later, quieter
  poll failure.

**One implementation-time addition beyond the plan's own text, both consistent with its intent:**
a tool call that completes but reports `CallToolResult.isError = True` (verified: a tool body
raising does **not** propagate as a client-side exception under the installed SDK — FastMCP
packages it as an error result instead) is now raised as `client.ToolCallError`, folding it into
the same poll-failure log-and-retry path as a transport-level exception (plan §3 step 2 already
says "network error, tool error" should both be handled that way) — without discarding an
otherwise-healthy shared connection, since a tool-level error doesn't imply the session itself is
broken.

**Test results:** `mcp-monitor/.venv/bin/pytest mcp-monitor/tests -q` → 34 passed (unit tier —
`test_config.py`/`test_matcher.py`/`test_launcher.py` — 26; in-memory component tier —
`test_watch_loop.py` — 5; real-subprocess component tier — `test_fake_server_integration.py` —
3). `ruff check mcp_monitor fake_mcp_server tests scripts` clean.
