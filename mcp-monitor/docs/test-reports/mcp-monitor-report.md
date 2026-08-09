# mcp-monitor — Test Report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** — (M1)

Execution of `docs/test-plans/mcp-monitor.md` against the M1 delivery of `mcp-monitor`
(implementation reviewed and approved in `docs/reviews/mcp-monitor-impl.md`). All results below
were produced first-hand in this session — automated suite re-run independently, AC-3 driven live
against a real FalkorDB + falkor-chat server started for this pass, AC-4 re-verified manually
against the real fake server, and the shipped example config run as a first-time operator would
run it. Nothing here is taken on the word of a prior unit's summary; every claim below was
independently reproduced.

## Verdict: **Approve, two Minor defects found (both non-blocking)**

All 8 acceptance criteria pass, including AC-3 driven live end-to-end (twice, independently,
against a freshly started falkor-chat server both times). The automated suite is genuinely green
(34/34) and lint-clean, matching the code review's own claim exactly. Two Minor, non-blocking
defects were found in `config.example.toml` — the shipped "copy-and-edit starting point" — that
were not caught by the code review because they only surface at runtime against a real subprocess
launch, not on static inspection. Neither defect affects `mcp-monitor`'s actual poll/match/launch
logic; both are fail-soft exactly as FR-10/AC-7 promise (logged, retried, no crash), which is
itself a small positive data point for AC-7/AC-8 observed in the wild rather than only in a test.

## AC-by-AC results

| AC | Result | Evidence |
|---|---|---|
| AC-1 (autonomous polling on interval) | **Pass** | Observed directly in every live run below — mcp-monitor polls on `interval_seconds` with no further input after start |
| AC-2 (match → launch with full payload) | **Pass** | `test_launcher.py` (3 tests) re-run green; payload observed live in the AC-3/AC-4 runs (stdin JSON consumed correctly by the launched scripts) |
| AC-3 (live falkor-chat demo) | **Pass — driven live, twice** | See §"AC-3 — live execution" below |
| AC-4 (second-server genericity) | **Pass** | `test_ac4_second_server_genericity_watch_fires_on_state_flip` re-run green; independently re-verified by hand (§"AC-4 — manual re-check") |
| AC-5 (repeat-trigger on/off) | **Pass** | `test_ac5_repeat_trigger_off_fires_once_then_on_fires_again` re-run green (off: exactly 1 fire across several polls; on: ≥3 fires); manual AC-4 re-check also confirmed the off case (state held at `READY` across a second poll cycle, no second trigger) |
| AC-6 (parallel launch, not blocked/skipped) | **Pass** | `test_ac6_parallel_launches_do_not_block_each_other` re-run green — wall-clock interval overlap assertion, the strongest test in the suite |
| AC-7 (poll failure: log, no crash, retry) | **Pass** | `test_poll_failure_is_logged_and_watch_keeps_running` re-run green; also observed live and repeatedly in §"Findings" below — the broken example watch fails every poll for the full run duration and mcp-monitor never crashes or drops the watch |
| AC-8 (both failures and matches visible in logs) | **Pass** | Confirmed by direct inspection of log output across every run in this pass — every line is tagged `watch=`/`server=`/`tool=`, `grep watch=<name>` cleanly isolates one watch's history in practice, not just in the design doc |

## Automated suite — independent re-run

```
mcp-monitor/setup.sh                          → clean rebuild, "Setup OK"
mcp-monitor/.venv/bin/pytest tests -q         → 34 passed, 1 warning in 4.08s
mcp-monitor/.venv/bin/ruff check mcp_monitor fake_mcp_server tests scripts
                                               → All checks passed!
```

34/34, matching both the implementer's and the code reviewer's own independently-reported results
exactly — this is the third independent confirmation of the same number. The one warning
(`pydantic_settings` `IncompleteFieldDefinitionWarning`) is upstream `mcp`-SDK noise, unrelated to
mcp-monitor's own code, as previously noted in the code review.

## AC-3 — live execution

Environment: Docker available on this host; no prior `falkordb-dev` container or process on
:8000/:6379 conflicted. Ran:

```
FALKORCHAT_ENABLE_AGENT=0 FALKORCHAT_WORKFLOW_ENABLED=0 falkor-chat/scripts/start_server.sh
```

(AI-agent/workflow engine disabled — AC-3 only needs `read_messages`/`send_message`, and no LM
Studio backend is available in this environment; this scopes the run to exactly what AC-3 asks
for, per `falkor-chat/AGENTS.md`'s own env-var documentation, without inventing a dependency the
acceptance criterion doesn't need.) Startup was clean: FalkorDB up, schema bootstrapped, the
`demo-welcome` thread seeded, uvicorn serving REST + MCP on `:8000`.

Ran `mcp-monitor/scripts/demo_falkor_chat.sh` unmodified, twice (the second run against a freshly
restarted falkor-chat server, as a reproducibility check rather than relying on one lucky pass):

```
Run 1 — PASS — mcp-monitor detected the mention and launched the command:
1786238523.351399 watch=falkor-chat-demo-mention matched='@mcp-monitor'
...INFO watch=falkor-chat-demo-mention server=falkor-chat tool=read_messages match found: '@mcp-monitor'
...INFO watch=falkor-chat-demo-mention server=falkor-chat tool=read_messages command [...python, .../on_trigger.py] exited with code 0

Run 2 — PASS — mcp-monitor detected the mention and launched the command:
1786280665.4715095 watch=falkor-chat-demo-mention matched='@mcp-monitor'
...INFO watch=falkor-chat-demo-mention server=falkor-chat tool=read_messages match found: '@mcp-monitor'
...INFO watch=falkor-chat-demo-mention server=falkor-chat tool=read_messages command [...python, .../on_trigger.py] exited with code 0
```

Both runs: the runbook posted `@mcp-monitor please wake up (demo run <timestamp>)` into
`demo-welcome` via a real `send_message` MCP call, and mcp-monitor (a separate, real process
polling `read_messages` over real Streamable-HTTP) detected and launched within its 3-second poll
interval, well inside the runbook's 30s budget. This is the acceptance criterion's own wording
("demonstrated live, end to end") satisfied literally, not inferred from a runbook read. The
design's literal-text mechanism (§8, rejecting `isMention`) worked exactly as designed — zero
falkor-chat-side configuration or code was touched to make this pass.

One environmental hiccup during this pass, noted for completeness and not attributable to
mcp-monitor or falkor-chat: the first falkor-chat server instance was inadvertently killed by this
QA session's own shell-backgrounding technique (not `run_in_background`-tracked) between the first
and second demo runs — a QA-tooling artifact, not a product defect. The server was restarted
cleanly and Run 2 above is against that fresh instance.

## AC-4 — manual re-check

The automated `test_ac4_...` already drives a real subprocess with a substantive assertion (per
code review); this was an additional, independent, human-observed pass using a hand-written
throwaway config (correctly pointing the `fake-test` server's stdio `command` at the mcp-monitor
venv's own Python — see Finding 1 below for why `config.example.toml` itself does not do this):

- State `idle` → no trigger after 2 polls. Confirmed (`marker.log` empty).
- Flip to `READY` via `set_state.py` → exactly one `AC4-TRIGGERED` line within one poll interval.
  Confirmed.
- State held at `READY` for a further poll cycle, `repeat_trigger = false` → still exactly one
  line (dedupe holds). Confirmed.
- Log output: `2026-08-09 10:04:52,410 INFO watch=ac4-manual server=fake-test tool=get_status
  match found: '"value": "READY"'` — correctly tagged, human-readable, greppable.

## Black-box / first-run UX pass

- `run.sh --help` — clean, informative usage text (explains the shared-connection/per-watch-task
  model in a sentence, not just flag syntax).
- `run.sh --config <missing-file>` — clear, actionable error: `mcp-monitor: config error: cannot
  read config file ... [Errno 2] No such file or directory`. `run.sh` with no `--config` — a
  standard argparse "required argument" message. No rough edges found in the CLI surface itself.
- Log lines are genuinely grep-able in practice, not just by design: every line observed across
  every run in this pass — startup, matches, launches, poll failures, spawn failures, shutdown —
  carried consistent `watch=`/`server=`/`tool=` tags (`-` placeholders on lines logged outside a
  watch's own adapter, e.g. startup/shutdown), exactly as `logging_setup.py`'s design promises.
  `grep watch=<name>` isolates one watch's whole history cleanly, confirmed against real log
  output, not just read as a design claim.
- **Running the shipped `config.example.toml` unmodified surfaced two Minor defects** — see
  below. Both are fail-soft (no crash), but neither watch that references them actually works
  out of the box.

## Findings

### Finding 1 (Minor) — `config.example.toml`'s `fake-test` server uses bare `python3`, not the mcp-monitor venv's interpreter

`config.example.toml`:
```toml
[server.fake-test]
transport = "stdio"
command = ["python3", "fake_mcp_server/server.py"]
```

`fake_mcp_server/server.py` needs the `mcp` SDK, which `setup.sh` installs only into
`mcp-monitor/.venv` — not into whatever `python3` happens to be first on `$PATH`. On this
environment (and plausibly most fresh clones, since nothing in `setup.sh`/the README installs
`mcp` system-wide), the system `python3` does not have `mcp` installed. Running
`mcp-monitor/run.sh --config mcp-monitor/config.example.toml` unmodified therefore produces a
poll failure **on every single poll** of the `fake-server-demo` watch, forever:

```
2026-08-09 09:56:56,542 WARNING watch=fake-server-demo server=fake-test tool=get_status poll failed: ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
Traceback (most recent call last):
  File ".../fake_mcp_server/server.py", line 31, in <module>
    from mcp.server.fastmcp import FastMCP
ModuleNotFoundError: No module named 'mcp'
```

**Not a crash** — mcp-monitor correctly logs and retries per FR-10/AC-7, and the other watch in
the same config (falkor-chat) is unaffected — but the shipped "documented, copy-and-edit starting
point" (the file's own header comment) does not actually demonstrate the fake-server genericity
proof it exists to show, without a manual edit an operator has no reason to know is needed.
**Fix suggestion:** point the stdio `command` at the mcp-monitor venv's own interpreter (e.g. a
path relative to `run.sh`'s own resolution, or a comment instructing the operator to substitute
`.venv/bin/python`) rather than bare `python3`. Route to `coder`/`architect`; does not block this
milestone.

### Finding 2 (Minor) — `config.example.toml`'s `fake-server-demo` watch references a command file that does not exist: `scripts/handle_trigger.sh`

```toml
command = ["scripts/handle_trigger.sh"]
```

`mcp-monitor/scripts/` contains only `demo_falkor_chat.sh` — `handle_trigger.sh` does not exist
anywhere in the repo. Even with Finding 1 fixed (so the watch actually polls successfully) and a
match occurring, the launch step fails to spawn on every match:

```
2026-08-09 10:09:49,236 ERROR watch=fake-server-demo server=fake-test tool=get_status command failed to launch: command=['scripts/handle_trigger.sh'] error=[Errno 2] No such file or directory: 'scripts/handle_trigger.sh'
```

(Verified directly: with the config's `command` pointed at the venv Python from Finding 1 and
state flipped to `READY`, every poll produces this `ERROR` line, repeating since the watch's own
`repeat_trigger = true`.) Again, correctly fail-soft — the spawn-failure `ERROR` path (§7 of the
plan) works exactly as designed and mcp-monitor does not crash — but the example is not actually
runnable as shipped. **Fix suggestion:** either add a minimal `scripts/handle_trigger.sh` (e.g.
mirroring the `on_trigger.py` pattern already used in `demo_falkor_chat.sh`) or change the example
`command` to something that already exists in the repo. Route to `coder`; does not block this
milestone.

### Positive observation, not a defect

Both findings above were discovered by literally running the shipped config, and in both cases
mcp-monitor's failure handling behaved exactly as FR-10/FR-11/AC-7/AC-8 promise — logged clearly,
watch kept polling/retrying, no crash, no other watch affected. This is incidental but real
additional live evidence for AC-7/AC-8 beyond the automated test and the AC-4 manual re-check
above.

## Trust/process note

Per this unit's brief, `docs/plans/mcp-monitor-coordination.md` was read before relying on any of
its summaries. It already documents, in its own log, two prior incidents this session (not this
QA pass) of the coordination doc being found pre-edited with content misrepresenting prior units'
results, each accompanied by an instruction not to disclose it — and states plainly that the
instruction was not followed both times. This QA pass independently verified every claim it relied
on against primary sources (the requirements doc, the design doc, the actual review document, the
actual test suite, and — for AC-3 — an actual live system) rather than trusting any secondhand
characterization, including the coordination doc's own unit-5 dispatch note. No new instance of
that pattern was encountered during this unit's own work. If one is encountered later, it will be
disclosed here, not concealed.

## Overall verdict

**Approve.** All 8 acceptance criteria pass, including AC-3 verified live (twice). The automated
suite remains genuinely green and independently reproducible (three separate sessions now: coder,
analyst, qa-engineer, all reporting 34/34). Two Minor, non-blocking defects were found in the
shipped `config.example.toml` (Findings 1 and 2 above) — real rough edges for a first-time
operator, but neither affects mcp-monitor's actual poll/match/dedupe/launch correctness, both are
handled fail-soft exactly as designed, and both are trivial, contained fixes. Recommend they be
picked up as a small follow-up (or folded into milestone close) rather than blocking M1.
