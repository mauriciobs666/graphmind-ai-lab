# Server logging config — review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-069

## Scope & verdict

**Reviewed:** the uncommitted K-069 change against baseline `abe1cd1f`. That is
`git diff abe1cd1f -- falkor-chat/{Dockerfile,README.md,docs/{BACKLOG,HISTORY,SERVER}.md,scripts/start_{server,agent_team,demo}.sh}`
plus the untracked `falkor-chat/config/logging.json`. I checked it against these baselines:
- the original item (`git show abe1cd1f:falkor-chat/docs/BACKLOG.md`, `### K-069`);
- report F-1 (`falkor-chat/docs/test-reports/extract-transient-retry-report.md` §6);
- the done-conditions in `falkor-chat/docs/plans/server-logging-config-coordination.md`.

I also checked the implementer's proof artifacts in `/tmp/claude-1000/k069/` and teco's probe in
`/tmp/claude-1000/k069-teco/t.py`.

**What I ran:**
- `bash -n` and `--help` on all three scripts;
- the `case` detection on its own, with edge-case inputs;
- in-process Python with `falkor-chat/server/.venv` (uvicorn 0.49.0, mcp 1.28.1). This covered
  `dictConfig`, the app's own import side effects, and logging mutants;
- a stricter log check, re-run against the implementer's existing run logs;
- read-only greps of the shared `:8200` server's log (`~/.local/state/graphmind-ai-lab/agent-team-8200.log`).

No server was started, and no process was signalled.

**Verdict: needs changes (documentation and proof only — the config and scripts work as built).**
- The shipped config and the shell wiring are correct. I found no code defect that would block
  merge.
- The change, however, records a defect mechanism that is false for the real server, in seven
  places. The shared `:8200` server's own log contradicts it, and so does the implementer's own
  mutant log (M1).
- The pass rule cannot see duplicated or lost lines. Nothing committed guards the config (M2).

CPG: considered, not relevant — the change is a logging dictConfig and shell wiring; the one code-level question (what the app's import does to logging) was answered by reading and executing `mcp/server/fastmcp/utilities/logging.py` and `falkorchat/app.py`, not a call-graph query.

## Findings

### M1 (major) — The stated defect is false for the real server: FastMCP's import already prints every INFO line, bare

**Evidence (verified):**
- `falkorchat/app.py:35` imports `falkorchat.mcp`.
- `falkorchat/mcp.py:31` builds `FastMCP("falkor-chat")` at module level.
- FastMCP's constructor (`mcp/server/fastmcp/server.py:242`) calls `configure_logging()`, which
  runs `logging.basicConfig(level="INFO", format="%(message)s", handlers=[StreamHandler()])`
  (`mcp/server/fastmcp/utilities/logging.py:39`). `rich` is not installed, hence the plain
  `StreamHandler`.
- So under uvicorn's default config the root logger gets an INFO handler. **Every** logger's INFO
  is printed: `falkorchat.*`, mcp, httpx. Each line is bare `%(message)s`, with no time, level, or
  logger name.

**Three independent confirmations** (Appendix A):
- I applied `LOGGING_CONFIG` in-process, then imported `falkorchat.mcp`. Root is at level 20 with a
  `StreamHandler`, and `FC-INFO` prints.
- The implementer's own `mutC-naive-fold-override.log`, which runs uvicorn's default config, shows
  `model provider lan-host: …` and `StreamableHTTP session manager started` bare.
- The shared `:8200` log holds 57 such bare lines. That includes three K-067
  `… abandoning retry (kind=…)` lines, at lines 5620, 5678 and 5736 (`retry.py:139` logs at INFO).

F-1's verification applied `LOGGING_CONFIG` and logged from `falkorchat.retry` **without importing
the app**, so it missed the side effect.

**Why it matters:**
- `HISTORY.md` would record a mechanism that never existed: "every INFO line was dropped", and
  "`:8200` could not show a K-067 gate abandonment at all".
- `README.md:282`, `SERVER.md` §1.7, the three script headers and `Dockerfile:38` tell operators
  the same false thing.
- The real baseline defect is narrower: no timestamp, no level, no logger name, plus third-party
  INFO noise.

**Fix:** reword all seven sites to the true mechanism. A suggested sentence: "under uvicorn's
default config the only handler on the root logger is the one FastMCP's import installs via
`basicConfig` — INFO from every logger, printed bare (`%(message)s`: no time, level or logger
name)". Re-derive HISTORY's "Defect:" bullet from that sentence. Open question 1 covers the
upstream documents.

### M2 (major) — The pass rule cannot detect duplicated or lost lines, and nothing committed guards the config

Answer to Q1, **verified**. `check.sh` asks for at least one timestamped `falkorchat.` INFO line
and at least one timestamped `/health` access line. I applied single-key mutants of the shipped
config in-process (Appendix B), with the app import included. Four of them **pass `check.sh`'s
regexes**:

| Mutant | Effect |
|---|---|
| `falkorchat.propagate=true` | every app line printed twice |
| `uvicorn.propagate=true` | every startup and shutdown line printed twice, once as `uvicorn.error: …` |
| `uvicorn.access.propagate=true` | every access line printed twice |
| `uvicorn` with its handler removed | every `uvicorn.error` INFO line lost (startup, shutdown, reloader). This breaks done-condition 2, which the rule never measures. |

A fifth mutant passes too: deleting `root`. FastMCP's `basicConfig` then takes root back at INFO,
and third-party INFO returns bare.

**Why it matters:**
- K-067's liveness evidence was a count of `abandoning retry` lines. A duplicating handler doubles
  that count without any visible error.
- Also, stdout and stderr are merged (`2>&1` in `probe.sh`), so the rule cannot see a stream swap
  either.

**Fix: a coverage probe over the config's real axis, (logger × level) → number of handlers
reached.** Assert the whole matrix, not a list of known shapes. I wrote and ran one,
`…/scratchpad/probe_cfg.py` (Appendix B). The required matrix is:
- `falkorchat.*`: INFO 1, WARNING 1;
- `uvicorn.error`: INFO 1, WARNING 1;
- `uvicorn.access`: INFO 1;
- third-party: INFO 0, WARNING 1.

It kills all 5 mutants, and the shipped config passes.

**Where it goes:**
- A `server/tests/` test that runs the probe **in a subprocess**. `dictConfig` is process-global,
  and `falkorchat.propagate:false` would blind the suite's 144 `caplog` uses.
- The probe must import `falkorchat.mcp` after `dictConfig`, so FastMCP's side effect is part of
  what it tests.
- Add one live-check clause as well: every line after the banner is timestamped, and
  `Application startup complete` / `Finished server process` are present. The current run logs
  pass this (Appendix C).
- Do not assert exact line counts across processes. Under `--reload`, uvicorn's parent also imports
  the app (`uvicorn/main.py:115`, `config.load_app()`), so import-time lines legitimately appear
  twice with different timestamps (`run5`).

### m3 (minor) — An undocumented behaviour change: third-party INFO that used to print is now dropped

**Evidence (verified):**
- Under the shipped config root has a handler, so FastMCP's `basicConfig` does nothing.
- mcp's `Processing request of type CallToolRequest` and `StreamableHTTP session manager
  started/shutting down` lines are gone: compare `run1-agent-team-default.log` with
  `mutC-naive-fold-override.log`.
- The `:8200` log shows those request lines were its main INFO traffic.
- The HISTORY and SERVER.md wording "third-party DEBUG/INFO stays quiet", and the script headers'
  "loggers stay at WARNING", read as no change. They are a change.

**Fix:** say plainly that mcp/httpx INFO is suppressed on purpose. If a per-request MCP trace is
wanted, add `"mcp.server.lowlevel.server": {"level": "INFO"}`. It would propagate to root's
handler, but root is at WARNING and handler-level filtering won't pass it, so it would need its own
handler. This is a design choice for the implementer.

### m4 (minor) — The running `:8200` server is not using the config, and nothing says so

**Evidence (verified):** pid 220352's command line is `uvicorn falkorchat.app:app --port 8200`, with
no `--log-config`. The fix reaches `ws:agent-team` only at its next start through the hook.

**Fix:** add one HISTORY line saying the shared server picks up the config on its next restart. The
restart itself is the human's call.

### m5 (minor) — Timestamps carry no timezone, and container and host disagree

**Evidence (verified):** `docker-run.log` is stamped `22:43:31` (the container runs UTC). The host
runs were stamped `19:40`–`19:42` the same day (`date +%z` = `-0300`). `%(asctime)s` uses the
default format, with no offset.

**Why it matters:** correlating a container log, a host log and UTC graph timestamps is where the
F-1 class of diagnosis happens.

**Fix:** either document "local time, no offset", or add a `datefmt` with `%z`. Note that the
stdlib cannot put milliseconds before `%z` with `asctime` alone. The variant I tried renders
`2026-10-17T19:53:14-0300.394`, so the exact format is the implementer's call.

### m6 (minor) — `--use-colors` / `--no-use-colors` in `UVICORN_ARGS` is now silently ignored

**Evidence (verified):** `uvicorn/config.py:375-383` applies `use_colors` only when `log_config` is a
dict. A `.json` path is loaded with no override. Colour detection falls back to
`sys.stdout.isatty()` (`uvicorn/logging.py`, `ColourizedFormatter.__init__`). `--log-level` still
applies, but only to `uvicorn.error`/`uvicorn.access`/`uvicorn.asgi`, never to `falkorchat`.

**Fix:** add one sentence in SERVER.md §1.7: colour and `falkorchat` verbosity are controlled
through `UVICORN_LOG_CONFIG`, not uvicorn flags.

### n7 (nit) — A tab-separated `--log-config` in `UVICORN_ARGS` is missed, and the shipped one silently wins

**Evidence (verified):**
- `UVICORN_ARGS=$'--port 1\t--log-config x'` falls through to the shipped branch, while the
  unquoted `$UVICORN_ARGS` still word-splits on the tab.
- uvicorn then gets two `--log-config` options. Click keeps the last one, which is the shipped
  file (tested with a minimal Click command).

The other edge cases resolve as intended: `=` form, value-less, leading or multiple spaces,
`--log-configx`, `--log-level`.

**Fix:** match against `" ${UVICORN_ARGS//[[:space:]]/ } "`, or accept the gap and document it.

### n8 (nit) — Doc placement and wording

- **Placement.** The operator note sits in SERVER.md §1.7, *"Testing hazards specific to
  `server/`"*. It is a run-time note; a short subsection, or the start-script area, fits better.
- **Relative paths.** SERVER.md says to pass `--log-config config/logging.json`, which is relative
  to `falkor-chat/`. README says `../config/logging.json` from `server/`. Both are correct for their
  own working directory, but neither says which directory it assumes.
- **Port range.** HISTORY's "ports 8295–8299" does not match the artifacts: the cmdlines and logs
  show 8296–8299, plus the container's internal 8000, and `8295` appears nowhere in
  `/tmp/claude-1000/k069/`.

## What's solid

- **The config itself.** It reproduces uvicorn 0.49.0's `LOGGING_CONFIG` faithfully (Q2). The
  same formatter classes, handlers, streams (access on stdout, everything else on stderr), logger
  levels and `propagate` flags are all kept, and so is `disable_existing_loggers: false`.
  `"use_colors": null` on the access formatter matches the default. Beyond the intended
  timestamps and `falkorchat` logger, the only deltas are the root handler (m3) and the ignored
  colour flag (m6).
- **Sound design calls:**
  - **`--log-config` rather than app code.** This is better justified than the docs say: the
    suite's 144 `caplog` uses would go blind under `falkorchat.propagate:false` applied in-process.
  - **Appending after `UVICORN_ARGS`.** The caller's override keeps the config, and `mutC` proves
    the naive fold loses it.
  - **The caller's own `--log-config` wins.** `run4` confirms it, with exactly one `--log-config`
    on the cmdline.
  - **Fail-fast on a missing file before step 1.** `run6`/`run7` show it in all three scripts.
    `--help` still exits 0 with a bad `UVICORN_LOG_CONFIG`, because `--help` is parsed first.
  - **Leaving `load_test.sh` out.**
- **Shell (Q3).**
  - The block is byte-identical in all three scripts (identical md5), so `start_demo.sh` matches
    the other two by construction.
  - `${LOG_CONFIG_ARGS[@]+"${LOG_CONFIG_ARGS[@]}"}` is the correct empty-array idiom under
    `set -u`.
  - `UVICORN_ARGS` is defaulted before the `case` in each script, so `set -u` never trips.
  - The shipped path stays one argument even with spaces.
  - Word splitting of a caller's `UVICORN_ARGS` (a spaced path inside it) is a pre-existing
    limitation, not new here.
  - No script `cd`s before `exec`, so a relative `UVICORN_LOG_CONFIG` resolves the same way at the
    check and at uvicorn.
- **The hook path (Q4).** `claude/teco/hooks/ensure-environment.sh:160-163` launches
  `start_agent_team.sh` under `env -i`, with no `UVICORN_ARGS` or `UVICORN_LOG_CONFIG`, from
  `falkor-chat/`. The default resolves to the absolute `$REPO_DIR/config/logging.json`, because
  `REPO_DIR` comes from `dirname "$0"`. `run1` reproduces that exact shape and passes. The Docker
  image `COPY`s `config` (`Dockerfile:23`), and `docker-run.log` shows timestamped lines.
- **Backlog hygiene.** BACKLOG has no dangling K-069 reference.

## Open questions

1. **Upstream errata.** F-1 in `docs/test-reports/extract-transient-retry-report.md` (owner
   `qa-engineer`) carries the same false mechanism. So does the B.5 claim that the `9b2c6b30…`
   failure cause "was never visible": the `:8200` log has bare tracebacks and abandonment lines.
   Should teco route an erratum, or leave the archived record alone and let M1's corrected HISTORY
   entry carry the truth?
2. **The `mcp` per-request INFO lines (m3).** Are they wanted on the shared server, or is dropping
   them the intended noise cut?

## Appendix

### A. Baseline mechanism reproduction (M1)

`…/scratchpad/base.py`: `dictConfig(cfg)`, then `import falkorchat.mcp`, then log.

```
--- default (uvicorn LOGGING_CONFIG)
root level 20 handlers [<StreamHandler <stderr> (NOTSET)>]
FC-INFO
TP-INFO
MCP-INFO
INFO:     UV-INFO
--- shipped (config/logging.json)
root level 30 handlers [<StreamHandler <stderr> (NOTSET)>]
2026-10-17 19:47:27,104 INFO:     falkorchat.retry: FC-INFO
2026-10-17 19:47:27,104 INFO:     UV-INFO
```

Shared `:8200` log: `grep -cE "^(model provider|abandoning|StreamableHTTP|Processing request|HTTP Request)"`
gives 57. Line 5620 reads: `extract chunkId=dcf0657b…: owning document no longer processing;
abandoning retry (kind=http, status=500)`.

### B. Mutants against `check.sh`'s rule and against the structural probe (M2)

| Mutant | `check.sh` regexes | `probe_cfg.py` |
|---|---|---|
| none (shipped) | PASS | PASS |
| `falkorchat.propagate=true` | PASS (falkorchat=2) | FAIL (falkorchat INFO/WARN → 2 handlers) |
| `uvicorn.propagate=true` | PASS | FAIL (uvicorn.error → 2) |
| `uvicorn.access.propagate=true` | PASS (access=2) | FAIL (uvicorn.access → 2) |
| `uvicorn` with no handlers | PASS | FAIL (uvicorn.error INFO → 0) |
| `root` deleted | (not run) | FAIL (httpx / mcp INFO → 1, via FastMCP `basicConfig`) |

Probe matrix: `falkorchat.retry` INFO/WARN = 1; `uvicorn.error` INFO/WARN = 1; `uvicorn.access`
INFO = 1; `httpx` INFO = 0, WARN = 1; `mcp.server.lowlevel.server` INFO = 0; root WARN = 1. The
reach count walks `logger.handlers` up the `propagate` chain, and counts `lastResort` only when no
handler was found.

### C. A stricter live check over the existing run logs (M2)

`…/scratchpad/check2.sh` adds these conditions after the banner: zero bare lines, zero repeated
full lines, plus startup and shutdown lines present. It passes `run1`, `run2`, `run3`, `run5` and
`mutC-naive-fold-default`. It fails `mutA`, `mutB` and `mutC-naive-fold-override`, and fails
`run4` by design, since `run4` uses a caller config with a different prefix. A full-line
`uniq -d` catches only the `falkorchat.propagate` duplicate: the `uvicorn`/access duplicates
differ textually. That is why the structural probe, not the log grep, has to be the guard.

`…/scratchpad/` = `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/cc651677-ac22-4370-9988-76881753007e/scratchpad/`.

## Pass 2

**Verdict: approve with suggestions.** Every Pass 1 finding is fixed. The one new finding is a nit
about how a proof is recorded; it does not block the merge.

**Scope.**
- Re-reviewed the working tree against `abe1cd1f`: the new `server/falkorchat/logformat.py`, the
  new `server/tests/test_logging_config.py`, the reworked `config/logging.json`, `SERVER.md`
  §1.10, and the rewordings in HISTORY, README, the script headers and the Dockerfile.
- Checked `devops`'s round-2 artifacts in `/tmp/claude-1000/k069/`.
- The coordinator's two decisions are taken as given, not as drift: m3's noise cut is
  intentional, and the F-1 erratum lives in HISTORY while the archived report stays untouched.

CPG: considered, not relevant — logging config, a formatter module and shell wiring; the questions (import order, handler routing) were answered by executing against the tree, not by a call-graph query.

**What I ran (all against the tree):**
- `tests/test_logging_config.py`: passes.
- Four mutants, each copied aside first and restored, with `md5sum -c` OK afterwards. All four
  were killed: `root` deleted, `uvicorn.propagate=true`, access stream switched to stderr, and
  the offset stripped from `logformat.formatTime`.
- The formatter classes in-process: MRO, colour and `levelprefix`.
- A live `--reload` server on `:8294` (`ws:k069probe`, `env -i`, the shipped config). I stopped
  it by process group; `:8294` is free and no process is left.
- The suggested mcp re-enable snippet.
- Counts in the `:8200` log, read-only.

### Dispositions

- **M1 — fixed.** The FastMCP `basicConfig` mechanism is now stated correctly in all seven sites:
  HISTORY, README, `SERVER.md` §1.10, the three script headers (identical, same md5), and
  `Dockerfile:37-41`. The erratum's "18 bare `abandoning retry` lines" rechecks: the `:8200` log
  has 18, and all 18 are bare.
- **M2 — fixed.** The guard reads the shipped file (`parents[2]/config/logging.json`, not an
  inline copy). It applies the config, then imports `falkorchat.app`, so FastMCP's `basicConfig`
  really does run afterwards. The `root`-deleted mutant is killed precisely because that call then
  takes over root. The guard runs in a subprocess with `FALKORDB_PORT=1`. In `--setup-plan`, the
  only fixture beyond pytest's built-ins is conftest's autouse `_model_config_env`; there is no
  `conn`, `repo` or `wf_repo`. All 11 mutants in `mutants-r2.txt` match HISTORY's list, and I
  re-killed 4 of them myself.
- **m3 — fixed (documented as intended).** The `SERVER.md` §1.10 snippet works as written: it
  printed `… mcp.server.lowlevel.server: Processing request of type X`. It brings back the
  per-request lines only, not the `StreamableHTTP session manager` lines, and the doc's wording
  ("the per-request trace") claims exactly that.
- **m4 — fixed.** HISTORY now says the running `:8200` server is not on the config yet.
- **m5 — fixed.** Timestamps are ISO 8601 with milliseconds and offset. The `env -i` run shows
  `-03:00` and the container shows `+00:00` (`r2-docker-run.log`).
- **m6 — fixed.** `SERVER.md` §1.10 and the script headers say `--use-colors` and `--log-level`
  don't reach the file.
- **n7 — fixed, detection only.** The `case` subject is now `${UVICORN_ARGS//[[:space:]]/ }`.
  What reaches uvicorn is unchanged: the same unquoted `$UVICORN_ARGS`, which already split on
  tabs. `n7-case-check.txt` shows a tab-separated `--log-config` now detected as the caller's.
  `bash -n` passes, and the block is identical in all three scripts.
- **n8 — fixed.** The note now lives in its own §1.10, and §1.7 no longer mentions
  `--log-config`. Both relative paths now name their base directory. HISTORY now says ports
  8296–8299, which matches the logs across both rounds.

### How the same-revision fixes interact (question 1)

- **Import weight.** `falkorchat/__init__.py` is a docstring only. `logformat.py` imports only
  `logging`, `datetime` and `uvicorn.logging`, and uvicorn is already loaded in a uvicorn process.
  `-X importtime` puts the module's own cost at ~5 ms on top of uvicorn.
- **Importability.** The import is safe wherever `falkorchat` is installed:
  - the venv, where the package is installed editable;
  - the image, through `pip install -e ./server` (`r2-docker-run.log` is stamped);
  - `env -i` (`r2-run1`);
  - the `--reload` parent and its worker: my `:8294` run shows `Will watch…`, `Started reloader…`
    and `Stopping reloader…` all stamped by the new formatter.
- **Colour and `levelprefix` are kept.** The MRO is
  `DefaultFormatter → _IsoOffsetTime → uvicorn DefaultFormatter → ColourizedFormatter`. Only
  `formatTime` is overridden.
  - `use_colors=True` gives ANSI colour on the level, and access lines get coloured request lines
    and status codes.
  - `use_colors` `None` or `False` gives plain `WARNING:  `.
  - An explicit `datefmt` is still honoured.

### New findings

- **n9 (nit) — HISTORY's live-proof bullet lists a `--reload` run that predates `logformat.py`.**
  - The `--reload` run (`run5`) is from round 1, which used uvicorn's stock formatters. Round 2
    re-ran only `r2-run1` to `r2-run3` and the image, none of them with `--reload`.
  - The gap is now closed by evidence: my `:8294` `--reload` run, shipped config, `env -i`, has
    every line stamped with the offset format, with startup, shutdown and reloader lines present.
  - Optional: add "(the `--reload` case re-verified on `:8294` in review Pass 2)" so the record
    matches.
