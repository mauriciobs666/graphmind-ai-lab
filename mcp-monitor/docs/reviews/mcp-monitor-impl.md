# mcp-monitor — Implementation Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (M1)

Static review of the `mcp-monitor` implementation (`mcp_monitor/`, `fake_mcp_server/`, `tests/`,
`scripts/`, packaging files, entry docs) against `mcp-monitor/docs/requirements/mcp-monitor.md`
(FR-1..FR-12, AC-1..AC-8) and `mcp-monitor/docs/plans/mcp-monitor.md` (architect, Status: active),
with particular attention to whether the four findings in `mcp-monitor/docs/reviews/mcp-monitor.md`
(the plan review — verdict: approve with suggestions) were genuinely closed. This is a static
review; no implementation file was edited to produce it, and the test suite was executed but not
modified.

## Verdict: **Approve**

All four plan-review findings are genuinely fixed, not just claimed fixed — each was checked
against either the actual installed SDK source or the actual code path, not the implementer's own
account. All twelve FRs and eight ACs trace to real code with tests (or, for AC-3, a documented,
plausible runbook) that actually exercise the behavior, not merely "it ran." The one
self-flagged deviation (`ToolCallError`) is a reasonable, in-spirit extension of the plan, correctly
scoped. The test suite is genuinely green (34 passed, re-run independently) and the assertions are
substantive — AC-4/AC-5/AC-6 in particular are proven via real subprocesses and wall-clock overlap
checks, not toy assertions. One Minor gap (below) keeps this from a plain "approve, nothing to
say": the Moderate finding's concurrency guard is implemented correctly but is not exercised by any
test — only by production wiring that no test drives concurrently.

## The four plan-review findings — verified, not taken on faith

### Major — `StdioServerParameters` command/args split: **fixed, verified against the installed SDK**

Read `.venv/lib/python3.12/site-packages/mcp/client/stdio/__init__.py` directly (not the plan or
implementer's comments): `StdioServerParameters.command: str`, `args: list[str] =
Field(default_factory=list)` — confirmed, same as the plan review found.

`mcp_monitor/config.py:_load_one_server` (lines 123–138) splits a `[server.*]` stdio block's
`command` array at load time: `executable, *rest = command`, stored as `ServerConfig.command: str
| None` (executable) and `ServerConfig.args: list[str]` (the rest). `mcp_monitor/client.py:68`
constructs `StdioServerParameters(command=config.command, args=config.args, env=config.env or
None)` — the split fields land in the right places. `tests/test_config.py:74-81`
(`test_stdio_command_is_split_into_executable_and_args`) pins this exactly:
`["python3", "fake_mcp_server/server.py", "--flag"]` → `command="python3"`, `args=["fake_mcp_server/server.py", "--flag"]`.
`tests/test_fake_server_integration.py` then exercises this split for real — a genuine subprocess
connection over stdio, not a stub — and all three of its tests pass. Genuinely fixed.

The plan review also flagged that the config's *other* `command` consumer (a watch's launched
process, §5) needs the opposite shape (untouched argv array). `WatchConfig.command: list[str]`
stays unsplit and is consumed via `asyncio.create_subprocess_exec(*watch.command, ...)` in
`launcher.py:97` — the two shapes are kept genuinely separate, and `config.py`'s own docstring
(lines 6–14) calls this out explicitly so a future change doesn't conflate them. This is a good
sign for code quality generally: the fix isn't just applied, it's guarded against regression by
being named where a future editor would look.

### Moderate — shared-connection concurrency guard: **implemented correctly, but not exercised by any test**

`client.py`'s `ServerConnection.__init__` creates `self._lock = asyncio.Lock()` (line 105); `call_tool`
acquires it for the whole open→call→discard-on-failure sequence (`async with self._lock:`, line
138); `aclose()` acquires it too (line 153). This is exactly the fix the plan review recommended,
and it is correctly shaped — a call in flight on one watch's task genuinely blocks a concurrent
discard-and-reopen from a different watch's task sharing the same `ServerConnection` instance,
because both paths go through the same lock.

However: I searched the test suite for a test that runs two `run_watch()` tasks concurrently against
the *same* `ServerConnection` object (the scenario the lock exists for) and found none.
`test_watch_loop.py` and `test_fake_server_integration.py` each construct a fresh `ServerConnection`
per watch under test (`grep -c "connection = \|connection2 = \|_connection_for(stub)"` → 5 and 4
matches respectively, each paired 1:1 with a single watch task). Production wiring
(`__main__.py:run()`) does share one `ServerConnection` per `[server.*]` name across every watch
referencing it, so the code path is real and reachable, and the lock is unconditionally correct as
written (asyncio locks are cheap and safe even when uncontended) — but the specific race the
Moderate finding worried about (a failure-triggered discard racing a concurrent in-flight call) is
not exercised anywhere in the automated suite. This is a gap in test coverage of a fix, not a defect
in the fix itself. Given `asyncio.Lock`'s well-understood semantics, the residual risk is low, but a
`test_client.py` (or an addition to `test_watch_loop.py`: two watches sharing one `ServerConnection`,
one of which fails and forces a discard-and-reopen while the other has a call in flight) would close
it outright rather than leave it to code inspection. Recommend as a follow-up, not a blocker.

### Minor (a) — unbounded dedupe-set growth: **fixed**

`docs/BACKLOG.md` has an explicit entry ("Unbounded dedupe-set growth"), correctly attributed to
the plan review's Minor finding (a), correctly distinguished from the separate
persistent-dedupe-state-across-restart item. Genuinely closed at the docs level, which is the level
this finding asked for (a v1 accepted trade-off, not a code fix).

### Minor (b) — config validation for transport-shape errors: **fixed**

`config.py:_TRANSPORTS = {"http", "stdio"}`; `_load_one_server` rejects any other value
(`transport not in _TRANSPORTS`) and separately requires `url` (non-empty string) for `http` or
`command` (non-empty list of non-empty strings) for `stdio` — all at load time, all raising
`ConfigError` before `load_config` returns. `test_config.py`'s parametrized
`test_invalid_config_raises_config_error` covers a bad transport value (`"htttp"`), a missing `url`
on an `http` block, and a missing `command` on a `stdio` block — three of the parametrize cases map
directly onto this finding. Genuinely fixed, and tested, not just implemented.

## Self-flagged deviation — `ToolCallError`: reasonable, in-spirit extension

`client.py`'s `ToolCallError` (raised when `CallToolResult.isError` is `True`) is present exactly as
described: `client.py:38-44` defines it, `call_tool` (lines 145–149) raises it after a successful
`session.call_tool` call whose result reports an error, and — critically — the raise happens
*outside* the `except Exception: discard` block (lines 140–144), so a tool-level error does not
discard an otherwise-healthy session. `watch.py`'s `run_watch` catches `Exception` generically at
its single poll-failure site (`except Exception as exc:`, line 78), so `ToolCallError` flows through
the same log-and-retry path as a transport exception, satisfying FR-10/AC-7 for this failure mode
too. The reasoning documented in `client.py`'s own docstring (verified against the installed SDK: a
tool body raising does not propagate client-side, FastMCP packages it as an error result) is
correct and matches known MCP/FastMCP behavior — this is not an invented problem. Judgment: this is
a legitimate gap the plan's text left implicit ("network error, tool error" in plan §3 step 2
already names both) rather than scope creep; the implementer noticed a case where "tool error" has
no obvious code path otherwise (a tool that reports its own failure without raising) and closed it
without touching either poll semantics or the discard/reconnect logic for the (unrelated) session-
health case. No concerns.

## FR / AC coverage — checked against the code, not the plan's promise

| Req | Code | Verdict |
|---|---|---|
| FR-1 (config: server/tool/args/interval/regex/command per watch) | `config.py` `WatchConfig`/`ServerConfig` | met |
| FR-2 (mcp-monitor is its own MCP client) | `client.py` (`ClientSession` opened directly, no proxy) | met |
| FR-3 (poll checked against regex) | `watch.py:find_matches` | met, tested (`test_matcher.py`) |
| FR-4 (match → launch without a human) | `watch.py:89-96` → `launcher.launch` | met |
| FR-5 (raw result + matched text + watch/server/tool id delivered) | `launcher.py:build_payload`/`build_env` | met, tested (`test_launcher.py`) |
| FR-6 (concurrent watches) | `__main__.py:run()` — one `asyncio.Task` per `[[watch]]` | met |
| FR-7 (≥2 distinct servers/tools demonstrated) | `fake_mcp_server` (automated) + `scripts/demo_falkor_chat.sh` (runbook) | met |
| FR-8 (repeat-trigger per-watch config) | `config.py` `repeat_trigger` field, default `False` | met, tested |
| FR-9 (parallel launch, no block/skip) | `watch.py:93` — `asyncio.create_task`, not awaited | met, tested (`test_ac6_parallel_launches_do_not_block_each_other`) |
| FR-10 (poll failure logged, watch continues) | `watch.py:76-80` | met, tested |
| FR-11 (both failures and matches/triggers logged) | `logging_setup.py` + `watch.py`/`launcher.py` log calls | met |
| FR-12 (second server is a purpose-built fake) | `fake_mcp_server/server.py` — new, minimal, stdio | met |
| AC-1 (config → autonomous polling) | `__main__.py:run()`, `watch.py:run_watch` loop | met |
| AC-2 (match → launch with full payload) | `launcher.py:build_payload` (stdin JSON) + env vars | met, tested |
| AC-3 (live falkor-chat demo) | `scripts/demo_falkor_chat.sh` | plausible runbook, not independently re-run against a live falkor-chat during this review (correctly out of automated-test scope per plan §8/§11) |
| AC-4 (second server proves genericity) | `test_fake_server_integration.py::test_ac4_...` | met, tested with a real subprocess |
| AC-5 (repeat-trigger on/off) | `test_fake_server_integration.py::test_ac5_...`, `test_watch_loop.py` | met, tested both branches |
| AC-6 (parallel launch under an in-flight command) | `test_fake_server_integration.py::test_ac6_...` | met, tested via wall-clock interval overlap, not just call count |
| AC-7 (poll failure: log, no crash, retry) | `watch.py:76-80`, `test_watch_loop.py::test_poll_failure_is_logged_and_watch_keeps_running` | met, tested |
| AC-8 (both failures and matches visible in logs) | `logging_setup.py` (`LoggerAdapter`, `watch=`/`server=`/`tool=` tags) | met |

Nothing was simplified, dropped, or downgraded relative to the plan's promise. The one place the
plan itself flagged as inherently un-automatable (AC-3, no falkor-chat server in the default test
environment) is handled exactly as the plan specified: a scripted runbook
(`scripts/demo_falkor_chat.sh`), not a pytest assertion pretending otherwise.

### AC-3 runbook — read, not executed

`scripts/demo_falkor_chat.sh` was read in full but not run (this review's scope is static; running it
requires a live FalkorDB + falkor-chat server per its own prerequisites, and that live verification
belongs to QA per the coordination doc). On inspection it does what §8/AC-3 require: checks the MCP
endpoint is reachable, writes a throwaway config pointing at the real `demo-welcome` thread with
`since=0` (matching the plan's literal-text mechanism, not `isMention`), starts mcp-monitor, posts a
message containing `@mcp-monitor` via a direct `send_message` MCP call, and polls a marker file for
up to 30s before reporting PASS/FAIL. The mechanics are sound and match the design; whether it
actually passes against a live server is QA's task, not this review's.

## Test suite — re-run independently, assertions inspected

Ran `mcp-monitor/setup.sh` (idempotent, `Requirement already satisfied` for everything, clean
smoke-import) then `mcp-monitor/.venv/bin/pytest mcp-monitor/tests -q`:

```
34 passed, 1 warning in 4.09s
```

(The one warning is a `pydantic_settings` `IncompleteFieldDefinitionWarning` from the `mcp` SDK's
own dependency chain, unrelated to mcp-monitor's code.) `ruff check mcp_monitor fake_mcp_server
tests scripts` — clean, no findings. Both match `docs/HISTORY.md`'s claimed results exactly.

Spot-checked whether AC-4/AC-5/AC-6's tests assert something meaningful, not just "ran without
raising":

- **AC-4** (`test_ac4_second_server_genericity_watch_fires_on_state_flip`) asserts the marker file
  is empty while state is `idle`, then that it contains exactly one line with the matched text after
  flipping to `READY` — a real state transition observed through a real subprocess, not a mocked
  return value.
- **AC-5** (`test_ac5_repeat_trigger_off_fires_once_then_on_fires_again`) asserts `len(off_lines) ==
  1` after several poll cycles against unchanging `READY` state (proving suppression, not merely
  "no crash"), then a second watch instance with `repeat_trigger=True` against the same still-READY
  state asserts `len(on_lines) >= 3` — both branches of FR-8 genuinely exercised, with fresh dedupe
  state correctly isolated between watch instances (matching the plan's per-watch dedupe design).
- **AC-6** (`test_ac6_parallel_launches_do_not_block_each_other`) is the strongest of the three: it
  uses a launched script that sleeps 0.5s before recording its start/end timestamps, then asserts
  `overlaps(intervals[i], intervals[j])` is true for at least one pair — an actual wall-clock overlap
  check, which is the only way to actually distinguish "parallel" from "serialized but fast." This is
  a meaningfully stronger assertion than a call-count check would have been.

`test_watch_loop.py`'s in-memory tier tests are lighter-weight but appropriately so for their tier
(no real timing claims, `_RecordingLauncher` + an `asyncio.Event` for synchronization) — e.g.
`test_repeat_trigger_off_suppresses_second_identical_match` and `test_repeat_trigger_on_fires_every_time`
both assert concrete call counts rather than "the recorder was called."

## General code quality

- Idiomatic throughout: dataclasses for config, `contextlib.asynccontextmanager` for session
  lifecycle, dependency-injection seams (`session_cm_factory`, `create_subprocess_exec`, `launch=`)
  used consistently to make async code testable without real transports/subprocesses — this is why
  the test suite can be both fast (tier 1/2) and real (tier 3) without duplicating logic.
- Error handling is deliberate, not incidental: `watch.py`'s single `except Exception` at the poll
  site is documented as intentionally broad (`# noqa: BLE001` with a comment explaining why), and the
  same pattern appears in `launcher.py`'s two failure sites (spawn `OSError`, and a generic
  `communicate()` failure) — both logged, neither propagated, matching the "must never crash a watch
  loop or an unattached supervisory task" requirement. `client.py:call_tool`'s own transport-level
  `except Exception` (lines 140–144) is narrower and more careful: it re-raises after discarding the
  session, rather than swallowing — the distinction matters because that exception still needs to
  reach `watch.py`'s poll-failure log site.
- `config.py`'s validation is thorough and fails on the first problem with a message naming the
  offending watch/server, matching §2's "hard startup failure, never a partially-running process."
  One small nit, non-blocking: `interval_seconds`'s `isinstance(interval_seconds, (int, float)) or
  isinstance(interval_seconds, bool)` check correctly excludes `True`/`False` from being accepted as
  `1`/`0` — a small but real correctness detail (Python's `bool` is an `int` subclass) that a less
  careful implementation would have missed.
- `logging_setup.py`'s `_DefaultContextFilter` (fills `-` for `watch`/`server`/`tool` on lines logged
  outside a watch's `LoggerAdapter`) is a clean solution to the "one formatter, two logging call
  sites" problem — avoids a `KeyError` on startup/shutdown log lines without needing a second
  formatter or format string.
- No fragile patterns spotted: no bare `except:`, no mutable default arguments, no shared mutable
  state accessed without the lock that guards it (`ServerConnection._session`/`_cm` are only touched
  under `self._lock`).

## Repo-convention conformance

- Header block, family-slug placement (`docs/reviews/mcp-monitor-impl.md`, role suffix `-impl`,
  distinct from `docs/reviews/mcp-monitor.md` which is the plan review), and `Status`/`Owner`/`Tracks`
  fields all correctly formed per root `AGENTS.md`.
- Root `AGENTS.md`'s diff (`git diff AGENTS.md`) touches exactly two places: a new `mcp-monitor/`
  bullet in the Structure section and a new row in the Component docs table — nothing else changed.
  Both are accurate to what was actually built (correctly describes TOML config, one `asyncio.Task`
  per watch, the fake-server genericity proof, zero falkor-chat-side changes).
- `docs/BACKLOG.md`/`docs/HISTORY.md` present, flat under `docs/`, correctly cross-referencing
  (not duplicating) the turn-taking/backoff and server-side-push items tracked elsewhere
  (`kiro/docs/requirements/kiro-vision-followups.md` item 4, `falkor-chat/docs/BACKLOG.md` K-018).
- `README.md` + `AGENTS.md` entry-doc pair present and substantive (not stubs) — both explain the
  actual architecture, not just restate the plan.

## Summary

The implementation delivers what the plan and requirements promised. All three concrete plan-review
defects (Major, two Minors) are fixed and verifiably so — checked against the installed SDK's real
`StdioServerParameters` definition and the actual config-validation code paths, not the
implementer's own claims. The Moderate finding's fix (`asyncio.Lock` around the shared connection)
is correctly implemented but has no automated test exercising the actual concurrent-access scenario
it exists for — a real gap, but a narrow one (the fix's correctness doesn't depend on anything
subtle; `asyncio.Lock` semantics are well-understood), and it does not block approval. The
self-flagged `ToolCallError` addition is a sound, correctly-scoped extension of the plan's own
"network error, tool error" framing. FR/AC coverage is complete and the tests that back AC-4/AC-5/
AC-6 make real, timing- and count-based assertions rather than smoke checks. Test suite re-confirmed
green (34 passed) and lint-clean independently of the implementer's and `teco`'s own verification.

**Recommendation for the follow-up (non-blocking):** add a test that runs two `run_watch()` tasks
against one shared `ServerConnection`, with one watch's poll forcing a discard-and-reopen while the
other has a call in flight, to convert the Moderate fix from "correct by inspection" to "covered by
test."
