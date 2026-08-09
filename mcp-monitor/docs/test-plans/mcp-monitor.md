# mcp-monitor — Test Plan

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** — (M1)

Risk-based QA pass for `mcp-monitor`'s M1 delivery, covering AC-1..AC-8 of
`docs/requirements/mcp-monitor.md`, as the final gate before milestone close (coordination doc
unit 5). Scope, mechanisms, and the AC-3 demo design are as specified in
`docs/plans/mcp-monitor.md` (architect); `docs/reviews/mcp-monitor-impl.md` (analyst, verdict
**Approve**) already confirms 34/34 automated tests pass and are substantive, with AC-3 explicitly
left unexecuted for this unit. This plan does not re-derive unit tests the automated suite already
covers well — it (a) independently re-runs and confirms that suite, and (b) drives the one thing
nothing has executed yet: AC-3 live.

## Risk assessment

| Area | Risk if broken | Coverage strategy |
|---|---|---|
| Config load/validation (FR-1) | Silent misconfiguration, unclear startup failure | Rely on automated suite (`test_config.py`, 11 tests incl. 7-way parametrized invalid-config matrix) — re-run, not re-derived |
| Poll/match/dedupe core logic (FR-3, FR-8) | False negatives/positives, wrong repeat behavior | Rely on automated suite (`test_matcher.py`, `test_watch_loop.py`) — re-run, not re-derived |
| Parallel launch (FR-9, AC-6) | A slow command blocks the watch loop or other launches | Rely on automated suite's wall-clock overlap assertion (`test_ac6_...`) — re-run; this is the highest-value automated assertion in the suite (timing-based, not a toy count) |
| Poll-failure resilience (FR-10, AC-7) | A single bad poll crashes a watch or the process | Rely on automated suite (`test_poll_failure_is_logged_and_watch_keeps_running`) — re-run |
| **AC-3: live falkor-chat demo** | **The one thing that has never actually run.** Design (§8) rejects `isMention` and substitutes a literal-text regex mechanism reasoned about but never executed against a live server; code review explicitly declined to run it. Highest residual risk in the whole delivery — everything else has been executed at least once by a prior unit. | **Drive live**, this unit, first-hand: real FalkorDB + real falkor-chat server + real `send_message` + real mcp-monitor process. Not a review of the runbook script — an actual run of it, with the result recorded. |
| AC-4 genericity proof (fake server) | Automated test could be internally consistent but not reflect real subprocess/CLI behavior an operator would see | Automated suite already uses a *real* subprocess (not a stub) — re-run is sufficient, but also spot-check independently and manually (fresh config, fresh state file, observed via `set_state.py` + log tailing) as a second, human-eyes confirmation of the same mechanism, since this is cheap and the code review flagged this general class of risk (assertions vs. real behavior) as worth independent verification |
| Black-box first-run UX (FR-11/AC-8, README quickstart) | Something a reviewer of code/tests wouldn't see: does the shipped example actually run cleanly, are the log lines actually grep-able by a human, is the CLI rough anywhere | Not covered by any existing artifact (code review is static). Exercise `run.sh --config config.example.toml` as a first-time operator would, and read the resulting log output as an operator would, not as a test assertion |

## Test items

### 1. Automated suite — independent re-run

- **Setup:** `mcp-monitor/setup.sh` (clean/idempotent rebuild).
- **Run:** `mcp-monitor/.venv/bin/pytest mcp-monitor/tests -q`.
- **Pass condition:** 34 passed, 0 failed. (Matches the code review's own independent run —
  three independent confirmations, if this one also passes: implementer, analyst, QA.)
- **Also:** `ruff check mcp_monitor fake_mcp_server tests scripts` — clean, no findings.
- Covers, by tier (already detailed per-test in `docs/reviews/mcp-monitor-impl.md`): FR-1/FR-3/
  FR-5/FR-8/FR-9/FR-10 unit-level; AC-4/AC-5/AC-6 via real-subprocess integration tests with
  wall-clock/count assertions.

### 2. AC-3 — live falkor-chat end-to-end demo (the primary new work of this unit)

- **Precondition:** Docker available; FalkorDB startable; `falkor-chat/scripts/start_server.sh`
  reachable per `falkor-chat/AGENTS.md`.
- **Steps:**
  1. Start FalkorDB + falkor-chat server (`falkor-chat/scripts/start_server.sh`, AI
     agent/workflow engine disabled via `FALKORCHAT_ENABLE_AGENT=0 FALKORCHAT_WORKFLOW_ENABLED=0`
     since AC-3 only needs `read_messages`/`send_message`, not an LLM/embedding backend, and no
     LM Studio is available in this environment — this scopes the run to exactly what AC-3 asks
     for without a spurious external dependency).
  2. Confirm the MCP endpoint is reachable (`curl`/the runbook's own check).
  3. Run `mcp-monitor/scripts/demo_falkor_chat.sh` unmodified — it writes its own throwaway
     config, starts mcp-monitor, posts a message containing the watch's pattern into the seeded
     `demo-welcome` thread via `send_message`, and polls for the triggered command's marker file.
  4. Record PASS/FAIL as the script itself reports, plus mcp-monitor's own log excerpt.
  5. Repeat once more (fresh mcp-monitor process, fresh dedupe state) as a reproducibility check,
     not a coincidence check.
- **Pass condition:** Script reports PASS; mcp-monitor's log shows a `match found` line for the
  posted mention and a `command ... exited with code 0` line, within the 30s window (well within
  budget given a 3s `interval_seconds`).
- **If environment-blocked** (no Docker/FalkorDB available): report AC-3 as **blocked
  (environment)**, not pass/fail — do not fake or skip silently.

### 3. AC-4 — manual sanity re-check against the fake server

Automated coverage is already a real subprocess with substantive assertions (per code review);
this is a cheap, independent, human-observed confirmation of the same mechanism, not a
re-derivation of the test:

- Hand-write a throwaway config with one watch against `fake_mcp_server` (using the mcp-monitor
  venv's own Python for the stdio command — see Finding 1 below for why this matters).
- Start mcp-monitor; confirm no trigger while state is `idle`.
- Flip state to `READY` via `set_state.py`; confirm exactly one trigger fires within one poll
  interval.
- Leave state at `READY` for a further poll cycle with `repeat_trigger = false`; confirm no
  second trigger (dedupe holds).
- Inspect the log lines produced for grep-ability (`watch=`/`server=`/`tool=` tags present and
  correct).

### 4. Black-box / first-run UX pass

Not covered by the code review (static) or the unit suite (exercises internals, not the shipped
CLI/config as a new operator would encounter them):

- Run `mcp-monitor/run.sh --config mcp-monitor/config.example.toml` exactly as shipped (not a
  copy edited first) and observe whether it starts cleanly, per this unit's brief.
- Read the log output a real operator would `grep watch=<name>` for — assess whether FR-11/AC-8's
  promise ("an operator can verify after the fact that a watch is working") holds up in practice,
  not just in the log-format design.
- Note anything rough about the CLI surface (flags, error messages, first-run friction) even if
  it doesn't map to a specific FR/AC.

## Out of scope for this pass

- Re-deriving unit-level test cases the automated suite already covers substantively (confirmed
  by independent code review — see `docs/reviews/mcp-monitor-impl.md`).
- Persistent dedupe-state-across-restart and unbounded dedupe-set growth — both are explicitly
  accepted v1 trade-offs tracked in `docs/BACKLOG.md`, not defects to re-litigate here.
- Anything in `docs/requirements/mcp-monitor.md`'s Out of scope list (triggered-command behavior,
  turn-taking/backoff, server-side push, a UI, auth/hardening, config hot-reload).

## Acceptance criteria → test item mapping

| AC | Test item |
|---|---|
| AC-1 (autonomous polling) | Item 1 (automated suite) + observed directly during items 2–4 |
| AC-2 (match → launch with full payload) | Item 1 (`test_launcher.py`) + observed live in item 2 |
| AC-3 (live falkor-chat demo) | Item 2 |
| AC-4 (second-server genericity) | Item 1 (`test_ac4_...`) + Item 3 (manual re-check) |
| AC-5 (repeat-trigger on/off) | Item 1 (`test_ac5_...`, `test_watch_loop.py`) + Item 3 (repeat check) |
| AC-6 (parallel launch) | Item 1 (`test_ac6_...`, wall-clock overlap) |
| AC-7 (poll failure: log, no crash, retry) | Item 1 (`test_poll_failure_is_logged_and_watch_keeps_running`) + observed incidentally in item 4 (the example config's broken fake-test watch retries every interval without crashing mcp-monitor — a live demonstration of AC-7, not just a unit test of it) |
| AC-8 (both failures/matches visible in logs) | Item 4 (log readability) + observed throughout items 2–3 |
