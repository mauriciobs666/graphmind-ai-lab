# Typed Decision Models — Implementation Plan, Headless Driver and Probe-Informed Audit

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.8 · **Extends:** `docs/plans/typed-decision-models4.md` · **Reviews:** `docs/reviews/typed-decision-models6.md`

*2026-11-01 — v1.8: §3.7 fixes `runs[].attempt` as the batch's issue ordinal (4-impl review Pass 10 open question).*

## 1. Goal & scope

This plan changes five things in plan4, using the 2026-11-01 spawn-mode probe:
- **The campaign driver runs unattended and headless**, one short `claude -p` session per batch. A deterministic runner, `driver run`, launches those sessions; a scheduler starts the runner at each usage window. This answers Q-1: the stakeholder allowed unattended relaunch (coordination ledger, 2026-11-01).
- **Spawns are forced to foreground.**
- **The driver's model is pinned.**
- **The §3.3 audit rows are revised** to match the shapes the probe observed.
- **The dataset files that model-bench imports get a stated contract** (P24-m2): the `splits.json` shape and the `labelVotes` keys, pinned by one contract vector.

The plan replaces plan4 §3.5's in-session loop and the human-started window. It also replaces the coordination ledger's inline rulings on background spawns, both the halt ruling and its reconciliation with Pass 8.

**Out of scope:**
- Plan5's content. Seed identity, generation ingest, the pilot gate and the spend gate are unchanged, so this plan extends plan4 only.
- The ledger's U5.6 rulings on ingest idempotence and on `Void`-is-ingested. They stand as carried.
- Any change to the labeller, the generator, their models, or the stakeholder's split: Opus generates and Sonnet labels.

## 2. Context & findings

CPG: considered, not relevant — no `cpg_decisionmodels` graph is loaded; `decision-models/decisionmodels/audit.py` and `ingest.py` were read from source.

**Evidence.** `docs/test-reports/typed-decision-models-harness-probe-report.md` (U6.1, by `devops`) records the 2026-11-01 interactive probe on Claude Code 2.1.316. Its §1 fact index is the one home of facts F1–F7, which this plan cites by number:
- **F1** an interactive spawn is background by default (`async_launched`), and completion arrives as a `<task-notification>`;
- **F2** only `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` gives a foreground spawn (`status: "completed"`, meta `requestShape: "foreground"`, `finished: true`); `run_in_background: false` and `CLAUDE_CODE_FORK_SUBAGENT=0` do not;
- **F3** completed runs end `end_turn`, with no `SubagentHandback`;
- **F4** settings and frontmatter hooks all fire on every subagent tool call, each writing its own `hook_success`;
- **F5** a SIGKILLed driver kills its subagent: no `finished`, and the restart's `stopped` notification has no `<tool-use-id>`;
- **F6** a foreign `SubagentStop` follows every driver `Stop`;
- **F7** an Opus 5.5 safeguards stop is `stop_reason: "refusal"`, then a `user` entry with `isMeta: true` and a `refusedMessageId` key, then the blocking Model-switch dialog.

**Headless evidence** (U6.2, the report's §9), cited as H1–H6: H1 foreground under `-p` (§9.1, §9.2); H2 background under `-p` (§9.4); H3a group-kill shape (§9.4, §9.6); H4 `--setting-sources project` (§9.4); H5 transcript path (§9.5); H6 a hook `allow` is the permission decision (§9.4, §9.6). U6.7 (§10) records the `permissionDecision` of a guard deny (D1), a blocked guard failure (D2) and an undecided call (D3) (§10.2, §10.3). `requestNonInteractive` (§9.6) and F6's absence under `-p` (§9.3) are cited there.

**Hooks reference** (`code.claude.com/docs/en/hooks`, read 2026-11-01). Settings deny and ask rules are still evaluated after a hook's `allow`. A `PreToolUse` hook that cannot start, exits other than 0 or 2, times out or prints invalid output lets the call continue to the normal permission flow, unless the handler sets `"onFailure": "block"` (v2.1.295 or later), which then blocks the call.

**Not probed**, and still needed; §3.4 and §4 say where each first appears:
- what a usage-limit stop looks like under `-p`, for the driver and for a subagent;
- what a safeguards stop looks like in an Opus *subagent* under `-p`.

The local corpus of 238 runs, all `async_launched` and ending in a hand-back, comes from the SDK-hosted coordinating session and is not evidence for `claude -p`.

**CLI surface** (`claude --help`, 2.1.316, read 2026-11-01). `-p` exits after one response and skips the trust dialog. `--session-id <uuid>` fixes the transcript name. `--permission-prompts none` auto-denies anything that would prompt. `--setting-sources <list>` limits which settings files load. `--fallback-model` would switch models silently, so the runner never passes it.

## 3. Design amendments

### 3.1 Driver mode: headless, one session per batch (plan4 §3.5 replaced)

**Choice.** The driver is **non-interactive**. `driver run` is a stdlib Python loop and owns the campaign loop. For each batch it launches one `claude -p --agent dm-driver` session. That session makes exactly **one** `Agent` call and exits. The runner then ingests.

**Why.**
- An unattended interactive session is driven through a TTY. It can block on a modal dialog (F7), and it never exits by itself. Detecting both would mean scraping the screen.
- `-p` exits when its turn ends. Plus `--permission-prompts none`, nothing can wait for a human.
- One session per batch moves the loop out of an LLM and into code. Today's driver must run `next`/`ingest` through `Bash` over hundreds of turns, compacting as it goes, and stay on script. The per-batch driver only copies one prompt. Its context stays a few thousand tokens.
- The driver no longer needs `Bash`, so plan4 §3.4's `Bash` allowlist branch goes away.
- The runner picks each session's UUID (`--session-id`) and records it in the ledger before launch. §3.3's location rule then reads exactly one known transcript.

**Rejected alternatives.**
- **Interactive session in tmux, relaunched per window.** Rejected for the reasons above; U6.2's H1 showed `-p` gives the same foreground shape (§2).
- **The runner launches `claude -p --agent dm-labeler` directly, with no driver.** The labeller would then be a *main* session. Its isolation (`omitClaudeMd`, `maxTurns`, the empty `instructions` set, the effort pin) and §3.3's location rule are all built on the subagent shape. Dropping the driver would mean re-deriving and re-probing all of them.
- **One `-p` session per window, looping internally.** It keeps the in-LLM loop and its drift and gives up the per-batch session id. It saves only a session start per batch, about one short Sonnet turn.

**Spawn mode and environment.**
- Every driver session runs with `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`. That is the only switch that gave foreground spawns (F2, and H1 under `-p`). A background spawn is a halt (§3.4); the campaign never supports it.

### 3.2 `driver run`: the unattended runner

**Starting the child session.** The runner builds the child's environment from an allowlist:
- `PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`, `TZ`, `TMPDIR`;
- plus `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`.

Everything else is dropped. That includes every other `CLAUDE_CODE_*` variable (an inherited `CLAUDE_CODE_EFFORT_LEVEL` would override effort) and every `ANTHROPIC_*` variable (an API key would change billing and model routing).

The command, with `exchange/` as the working directory:

```
claude -p --agent dm-driver --session-id <uuid> --strict-mcp-config \
  --mcp-config <campaign>/exchange/empty-mcp.json --permission-mode acceptEdits \
  --permission-prompts none --output-format json "<driver message>"
```

- `<driver message>` is the fixed driver template (§3.3) with the agent type and spawn prompt substituted.
- stdout and stderr go to `store/runs/<uuid>.json` / `.err`, with the exit code. The audit classes runs from transcripts. These files are evidence only for a launch that left no transcript (§3.4), and they are recorded in that event as is: a killed child's stdout is empty (H3a), never parsed.
- `--setting-sources project` is **not** passed. The command stays H1's, the one U6.2 probed. Whether the flag silences user-level hooks is moot: the `hooks` refusal below means no user-settings hook exists at any launch, and a plugin hook can neither override a guard `deny` nor ask or deny without halting (§3.4).

**User settings.** `~/.claude/settings.json` applies to the child (m2). The user's ordinary edits to that file (`model`, `theme`, `editorMode`, `verbose`) must not stop the campaign. So `install` records only a sha256 over the canonical JSON of its `permissions` and `modelSettings` values. Before each launch, `run` refuses if any of these holds:
- an `env` or `hooks` key is present;
- `autoContinueAtUsageLimit: true` is set (M1). That setting would make a spent window wait out the limit instead of ending;
- the `permissions`/`modelSettings` hash differs from `install`'s. Settings deny and ask rules apply even to a guard-allowed call (§2), so a new rule there would refuse calls mid-campaign.

A refusal appends nothing to the ledger. It overwrites `store/refusal` with its reason and a timestamp; a successful launch deletes the file, and `status` prints it while it exists.

**Per batch:**
1. `next` reconciles, then names the pending batch, or returns `stop`.
2. Append `issued {batch, opaque, sessionId: <fresh uuid4>}`, fsync'd.
3. Launch the session in its own process group.
4. Wait, under `batchTimeout`. It is set at `init` (45 minutes by default) and recorded in the `init` event. On expiry, send SIGTERM to the group, then SIGKILL after 30 s.
5. `ingest`: the audit, then the ingester, ledger first, per the ledger's U5.6 ruling.
6. Loop.

**Exit conditions.** `run` exits 0 when `next` returns `stop`. That happens:
- when the phase is done;
- on a halt;
- on a usage stop;
- when `store/pause` exists.

**Single-flight and one ledger writer.**
- `run` takes `fcntl.flock(store/driver.lock, LOCK_EX | LOCK_NB)` and holds it for its whole life. It is the only process that appends to the ledger while it runs.
- On taking the lock, before any launch, it atomically writes `store/runner.json`: `{runnerPid, runnerStartTime}`, where the start time is field 22 of `/proc/<pid>/stat`. This file is operational state, and the ledger never reads it.
- The lock's descriptor is passed to the `claude` child (`pass_fds`). An orphan, meaning a child whose runner died, therefore keeps the campaign locked until it exits.
- When the lock is held, `run` exits 0 with one line. This covers a scheduler fire during a run, and an orphan.
- `resolve`, `install`, `finalize` and `init` append to the ledger. They take the same lock without blocking, and they refuse while it is held.
- `pause` and `resume` never touch the ledger or the lock.

**Stop switch** (M2). Pausing is operational control, not campaign state, so it lives in one file, never in the ledger.
- `driver pause` creates `store/pause`. It needs no lock, so it works while a run is active. `next` returns `stop` while the file exists, so the run finishes the batch in flight and exits. A fire while paused exits at once.
- `driver resume` deletes the file. `status` reports it.
- `driver pause --now` also creates the file, then sends SIGTERM to the runner named in `runner.json`, if that PID is alive with the recorded start time. The runner kills its child group, as on a timeout, if one is running, and appends `interrupted {batch}` (it holds the lock, so it is still the only writer). Then it exits.
  - An `interrupted` re-issue does not count toward the never-spawned/incomplete cap.
  - An orphan is not signalled. It is stopped by disabling the timer and killing it by hand (§6).
- Disabling the scheduler entry is the outer switch (U6.4).

The pause has exactly one home, this file. A `paused` ledger event copied in by the runner would keep one writer too, but it would give the pause a second home, and the two could disagree after a `resume` with no run in between.

**Resume.** A relaunch is an ordinary `run`. `next` reconciles every `issued` batch through its recorded `sessionId` with §3.4's rows, then serves the next batch. Re-issued names are retired, per the ledger's U5.6 ruling.

**Scheduling.**
- A user-level timer starts `driver run --root <campaign>`. On WSL2 this is a systemd user timer if systemd is enabled, else cron.
- It fires every 30 minutes. A fire within a spent window costs at most one short driver turn, which ends as a usage stop or a launch that left no transcript (§3.4). A fire with the campaign done, halted or paused exits at once.
- `driver run` refuses a root inside a git work tree, as `init` does (plan4 §3.5).

**Subcommands.** `init`, `install`, `run`, `pause [--now]`, `resume`, `resolve`, `finalize`, `status`. `next` and `ingest` become internal to `run` and are not listed for any session.

### 3.3 Agent definitions and guard wiring (plan4 §3.4 amended)

**`dm-driver`:**
- `model: claude-sonnet-5-5` (full id). Attempt 2's driver ran on this id with no dialog (F7).
- `tools: Agent(dm-generator, dm-labeler)`; no `Bash`.
- `maxTurns: 3`.
- No `initialPrompt`: the `-p` message is the prompt.
- Its body says: make exactly one `Agent` call with the `subagent_type` and `prompt` given in the message, copied exactly, and a fixed `description`; then reply `done`.

**The driver message template** is one fixed text with `{agent_type}` and `{spawn_prompt}` slots. Its hash is the new key `promptSha256.spawnTemplate.driver`, beside plan4 §3.7's `spawnTemplate.generator`/`.labeler`. U5.10's "matches §3.7's keys exactly" includes it.

**`dm-generator` / `dm-labeler`:** unchanged (plan4 §3.4 table).

**Guard wiring.**
- `guard_hook.py` is wired **once**, in `exchange/.claude/settings.json` `PreToolUse` with matcher `*` and `"onFailure": "block"`. Frontmatter `hooks` are removed from all three definitions.
- Settings hooks fire inside subagents with `agent_type` set (F4), and under `-p` in a folder never trusted (H1), so one wiring covers the driver and both subagents. Each tool use gets exactly one guard `hook_success`.

**The guard is the only bound** (gate (f)). Under `-p` a guard `allow` is the permission decision (H6). `acceptEdits`' working-directory limit, the frontmatter `permissionMode` and `--permission-prompts none` never see an allowed call, and bound nothing. They stay as probed (H1); `--permission-prompts none` still keeps a settings ask rule from waiting on a human. Frontmatter `tools` still limits which tools exist. Everything else that limits a call is the guard's.

**Guard decisions.** Plan4 §3.4's list holds, including its hard `deny` of a subagent `Read` outside `exchange/req/` and `Write` outside `exchange/ans/`, except:
- the `Bash` branch is deleted;
- `dm-driver` may call only `Agent`;
- **paths are canonical.** `file_path` must be a non-empty string. A relative path is joined to the hook input's `cwd`, and the result goes through `os.path.realpath`, which resolves symlinks and `..`. It must then lie strictly inside `realpath(exchange/req)` for a `Read`, or `realpath(exchange/ans)` for a `Write`;
- **fail closed.** A missing or unknown `agent_type`, a missing `tool_input`, or any exception makes the guard exit 1 with no decision, and the wiring's `onFailure: "block"` blocks the call, as when the guard cannot start or times out. Without it the permission flow would decide, and `acceptEdits` allows a `Write` in `exchange/` (D3). A blocked failure is recorded exactly like a deny (§10.2), and §3.4 halts both. The timeout and cannot-start shapes are unprobed (§10.6), but every decision other than a guard's `accept` halts;
- **`SubagentHandback`:** if U5.7 shows a hook sees it, the guard **allows** exactly that tool for `dm-generator`/`dm-labeler`. The audit's terminal rule (§3.4) limits it to one, last.

`install` records the guard's exact `command` string as a declared expectation, and refuses an installed wiring without `onFailure: "block"`. P8-F4's `Expected.guard_command` compares against it.

### 3.4 Audit rows (plan4 §3.3 amended)

These replace plan4's "no `tool_result`" row and its counted `hookDenied` row, amend its hook and bounds rows, and add rows. The other plan4 rows are unchanged.

**Guard decisions and in-flight tool uses.**
- A call's *guard decision* is the top-level `permissionDecision` on its `tool_result` entry. U6.2 and U6.7 show that an allowed call carries `accept`/`hook` plus a `hook_success` from the guard, and that a denied or blocked call carries `reject`/`hook` with **no** attachment (§10.3). A deny and a guard failure cannot be told apart by structure: only user-visible text differs. So both halt as one class. Counting a deny as the run's fault would buy an automatic retry for a call that a correctly briefed subagent never makes. It would also drain the campaign silently whenever the guard itself is broken (P3-M1). The guard's own hook log could separate the two. It is ruled out because it would make the code under suspicion the witness to its own correctness.
- A tool use's *guard entry* is a `hook_success` with its `toolUseID`, `hookEvent: "PreToolUse"` and the declared guard `command` (P8-F4), whose `stdout` parses to the guard's `allow` JSON. Only allowed calls have one.
- A tool use is *in flight* when it has no `tool_result` and the driver's `Agent` call has no matched `tool_result`. Every such tool use counts, not only the last, because parallel calls are persisted as `tool_use`, `tool_use`, then entry and result pairs (P3-m1). A killed run never persists an in-flight call's guard entry or result, even though the hook ran (H3a). An in-flight tool use is evidence only of incompleteness: no other row reads it, and plan4's "a `tool_use` but no subagent transcript" halt does not apply to an in-flight `Agent` call. So a killed or timed-out batch is `REISSUE incomplete`, never `hookMissing`, and the precedence below needs no change.
- The trade-off: a call that the guard wrongly allowed out of bounds, cut short by a kill, goes unseen in that run. Any run that completes shows such a call as `guardBreach`. The audit's bounds code does not import `guard_hook.py`, so one canonicalization defect cannot hide in both (P3-m3).

| Check | Class |
|---|---|
| **Completion, foreground (the only supported shape, F2):** the driver's `tool_result` matched to the spawn's `tool_use_id` has `toolUseResult.status: "completed"`, and the meta has `requestShape: "foreground"` and `finished: true` | completed: the remaining rows decide |
| any assistant entry in **either** transcript with `message.stop_reason: "refusal"`, or any `user` entry carrying a `refusedMessageId` key (F7) | **environmental: halt**, `safeguardStop` |
| a matched `tool_result` with `status: "async_launched"`, or a meta with `requestShape: "background"` | **environmental: halt**, `backgroundSpawn` (the environment variable did not apply). Under `-p` the process waits for the background run, whose answer may be complete (H2); it halts all the same |
| a tool use whose guard decision is `reject` with `source: "hook"` (D1 and D2) | **halt**, `guardRejected`: a guard deny or a guard failure (above) |
| a tool use whose guard decision is `accept` with `source: "hook"` and has a guard entry, but whose tool or path is outside the guard's bounds (§3.3), as the audit recomputes them from the `tool_use` input | **environmental: halt**, `guardBreach`: the only bound failed, and the call ran |
| a tool use with no guard decision, `is_error` on its `tool_result`, and no attachment | run-attributable: input validation refused the call before any hook ran, so it did nothing (P3-m2) |
| any other tool use whose decision is not `accept`/`hook` with a guard entry: no guard decision, another `source` (D3's `accept`/`config`), or `accept`/`hook` with no guard entry (plan4's row, re-keyed) | **environmental: halt**, `hookMissing`: the guard did not decide; the permission flow, a settings rule or another hook did |
| an allowed `Read`/`Write` inside the guard's bounds but on a path other than this run's own request and answer parts (plan4's bounds row, narrowed to what the guard lets through) | run-attributable |
| the driver transcript has a `<synthetic>` error other than `rate_limit`; any driver `message.model` ≠ `claude-sonnet-5-5` | environmental: halt |
| a `<synthetic>` `rate_limit` in **either** transcript | usage stop: no attempt, uncapped |
| any other `status`; `completed` with `finished` absent; a meta `requestShape` other than `foreground` | **environmental: halt**, `spawnShapeUnknown` |
| no matched `tool_result` and no error `<synthetic>` (killed, timed out, interrupted, or the driver died mid-run; in-flight tool uses are ignored, above) | incomplete: re-issue, no attempt; the 2nd for one batch halts, unless an `interrupted` event names it (§3.2) |
| a driver transcript with no `Agent` `tool_use` and no `<synthetic>`, from a child that exited 0 | never spawned (plan4's row): re-issue, no attempt; the 2nd for one batch halts |
| the runner-recorded `sessionId` has **no transcript file**, or the child exited non-zero leaving neither an `Agent` `tool_use` nor a `<synthetic>` (P2-m1: "the launch produced nothing" is one rule) | `launchFailed`: re-issue, no attempt, uncapped. The run's stdout, stderr and exit code are recorded in the event. Once launches have produced nothing continuously for longer than `launchFailBound`, the next one halts as `launchFailing`. |

**Precedence.** Rows apply in this order, and the first that matches decides:
1. the halts above, except `spawnShapeUnknown`, then plan4's other halts;
2. usage stop;
3. `spawnShapeUnknown`;
4. the no-attempt re-issues;
5. plan4's run-attributable rows.

Placing usage stop above `spawnShapeUnknown` is deliberate (M1): a subagent stopped at the usage limit may hand the driver a result of a shape never seen, and the rate-limit entry explains it. A `backgroundSpawn`, a refusal, a `guardRejected`, a `guardBreach` or a `hookMissing` is a configuration or safeguards fault and halts even then.

**Why `launchFailed` is time-bounded, not capped** (M1). The usage-limit shape under `-p` is unprobed. If a driver at the limit produces nothing, a cap of one would halt the campaign in every spent window. `launchFailBound` is a parameter recorded in the `init` event (default 6 h). It must exceed one usage window on the account in use, and the stakeholder sets it at `init`. The rule depends on no particular reset period, and a longer outage (weekly limit, broken authentication, a missing binary) halts with a clear reason. The bound runs from the first `launchFailed` with no later launch that produced a transcript with an `Agent` call or a `<synthetic>`, read from the ledger's event timestamps. **Orphan stops count** (P2-m2): an orphan is ended only by a human `kill`, which writes no `interrupted` event, so its re-issue counts toward the cap like any harness fault.

**Where the usage shape is first recorded, and its red check.** U5.7's smoke cannot reach a limit by design. The first spent window of the guard pilot is the first live case. There, `teco` appends the driver's and subagent's last entries, plus the runner's stdout and stderr, to the probe report as a new section. The red check: **a usage stop that halts the campaign means a row is wrong**. That goes to `architect` before the next window. No probe should spend a window on purpose to provoke one.

**The safeguards row keys on structure, not text** (m3). The "safety classifier" sentence in F7's `user` entry is user-visible wording, which can change in any release without notice. The same entry carries `refusedMessageId` and `isMeta: true`, and the assistant entry before it carries `stop_reason: "refusal"`. The row matches on either structured marker, so it still fires if one is missing. Neither is documented, so both are pinned by U5.5 fixtures built from the report's §7 key lists.

**Terminal shape.** A completed run is clean only if its last assistant entry is one of:
- an `end_turn` text reply (F3);
- exactly one `SubagentHandback` `tool_use`, followed only by its `tool_result`.

The hand-back is exempt from the bounds checks, `guardBreach` included, and counts against `maxTurns` as the reply. It is exempt from `hookMissing`, and a guard reject on it is `guardRejected`. Any other last entry on a completed run is the provisional `maxTurns` row, which U5.7 replaces with the smoke's recorded shape.

**Ignored as evidence:**
- `<task-notification>` user entries (F1, F5), including a `stopped` notification with no `<tool-use-id>` (report §7, session `fb3a881a`). Completion is read only from the `tool_result` and the meta.
- `hook_success` entries whose `command` ≠ the declared guard command (P8-F4).
- Every hook event other than `PreToolUse`. In particular, `SubagentStop` is never read, so F6's foreign events need no filter.
- `requestNonInteractive` (report §9.6). No row keys on it.
- `hook-log.jsonl`. Every consumer of it (U5.8 diagnostics, `status`) filters on `event == "PreToolUse"` and `agent_type ∈ {dm-driver, dm-generator, dm-labeler}`. It can name a `PreToolUse` whose `tool_use` never reached the transcript (H3a), so no consumer joins it one-to-one with transcript tool uses.

**Multiple guard entries.** Each allowed call needs at least one guard entry. §3.3 wires the guard once. A duplicate (for example a stray frontmatter wiring) still must not void a run.

### 3.5 Safeguards flags, and Opus as generator

- **The driver** is pinned to Sonnet (§3.3), so F7's dialog cannot recur there. If a driver turn is flagged anyway, `-p` cannot show a dialog. The transcript carries the `safeguardStop` markers, which halt.
- **An Opus `dm-generator` subagent** flagged mid-run may leave a partial tool call. Without `safeguardStop` above every counted row, a partial `Write` would void as `malformed` and be counted. The row halts first. If the user's `/config` holds "Switch automatically" and the run continues on Opus 4.8, the `message.model ≠ pin` row halts instead.
- **The stakeholder's rule (2026-11-01).** Each flag halts the run, and nothing from it is counted. There is **no** automatic fallback to another model. At the halt, the stakeholder chooses one of three actions, applied by `driver resolve`:
  - **retry:** `resolve --decision <text>`. The batch is re-issued under a new opaque name, with no attempt counted.
  - **skip the batch:** `resolve --skip-batch --decision <text>`. This is new. Its seeds (generation) or items (labelling) drop and are counted under the new `generation.json` key `dropped.safeguardSkipped: {seeds, items}`, beside plan4 §3.7's `retryCapExceeded`.
  - **move generation to another model:** `resolve --repin-model generator <id>`, with plan4 §3.3's `generation-restart` event.
- **Not probed:** a subagent-side flag under `-p`. The row keys on the main-session markers (F7). U5.7's smoke records any flag it meets. No probe should try to provoke one.

### 3.6 The probe evidence's home

`docs/test-reports/typed-decision-models-harness-probe-report.md` is the evidence record that every harness premise here cites, and U5.5's trimmed shape fixtures go under `decision-models/tests/fixtures/transcripts/`.

### 3.7 Dataset files model-bench imports (plan4 §3.7 amended; P24-m2)

Plan4 §3.7 defines `generation.json` and the item keys. A run's `attempt` is its batch's issue ordinal from 1, counting no-attempt re-issues too, so `runs[]` may skip numbers; nothing compares it with the retry cap. These two shapes complete the contract that `finalize` (U5.10) writes and that `model-bench/scripts/import_decision_pack.py` reads. Today they are stated only in that importer's docstring and test builder.

- **`splits.json`** is one JSON object with exactly the keys `rngSeed` (int, the split draw's seed), `eval`, `train` and `calibration`.
  - Each of the three lists holds `itemId` strings. Order carries no meaning (P2-m3).
  - Every `itemId` in `items.jsonl` appears in exactly one list, and no list repeats an id.
  - The importer reads only `eval`.
- **`labelVotes`** is a JSON object whose keys are exactly the point's answer space, written as strings, with zeros included:
  - `"true"` and `"false"` for a predicate point (guard, join-in);
  - each `MESSAGE_TYPE_LABELS` label for message type.

  Its values are non-negative ints summing to `votesPerItem` (3). A JSON boolean is never a key, so the predicate spelling is fixed here, lowercase.

**Contract vector.** One file, `model-bench/tests/fixtures/decision-contract/dataset-shape.jsonl`, owned by model-bench. It has one line per decision point (3), each line `{"role", "generation", "items", "splits"}`:
- `generation` is a complete plan4 §3.7 object, with every key including `promptSha256.spawnTemplate.driver` and `dropped.safeguardSkipped`. It also carries plan5 §3.5's `dropped.formMismatch: {<form>: n}` and `dropped.bindingMismatch`, and two `runs[]` entries. One is ingested. The other, `incomplete`, is a type exemplar rather than one real record: `voidClass` and `voidReason` are strings, and `claudeCodeVersion`, `model`, `sessionContextSha256`, `thinking` and `effort` are `null`.
- `items` holds three items, one per split, carrying every item key. Message type's show `priorTurn` both `null` and an object.
- `splits` follows the shape above.

`decision-models/tests/fixtures/decision-contract/dataset-shape.jsonl` is a byte-copy, under the existing guarded-equality test (`decision-models/tests/test_decision_contract.py`).

Who checks what:
- **The importer's tests** read the canonical file. Each line's items pass the importer's item projection and `_eval_ids`.
- **The importer** gains one refusal: `labelVotes` keys ≠ the answer space. Today it accepts any keys summing to 3.
- **U5.10** checks its `finalize` output on its fixture campaign against the copy's line for the same role.
  - **Outside the data-sized containers,** key sets must be equal, recursively for `generation.json`, and so must each value's JSON type. The types are object, array, string, integer, other number, boolean and null, as `json.loads` gives them; a boolean is never an integer.
  - **The data-sized containers** are `dropped.formMismatch` (its keys are the forms that dropped), `runs[]`, each run's `instructions`, `items.jsonl` and the three split lists. Each output entry is compared against all of the line's entries of the same container, never one chosen entry, so the vector stays the shape's one home and no key is named here. An object entry's key set must equal theirs, and each of its values must have a type that some vector entry shows for that key. A scalar entry's type must be one the vector's entries show.
  - **An entry is compared one level deep.** Below it, only each run's `tokens` key set, its `instructions` entries (by this rule) and each item's `labelVotes` key set are compared. Input-field shapes are `input_field_errors`' check.
  - **An output entry with no counterpart is refused,** never passed vacuously: its container is empty on the vector's line, as a `formMismatch` entry on a join-in line would be.
  - U5.10's fixtures include one campaign where a form dropped and one where none did.

## 4. Units

These change plan4 §4.1's U5.5–U5.7 rows and add eight units. Every code unit gets an `analyst` code gate. U5.x and U6.x serialize on `decision-models/docs/HISTORY.md` (plan4 §4.3).

| Unit | Owner | Brief | Done when |
|---|---|---|---|
| **U6.1** probe evidence record | `devops` | §3.6 | **Delivered** (`docs/test-reports/typed-decision-models-harness-probe-report.md`). |
| **U6.2** headless probe | `devops` | Rerun the U6.1 kit under §3.2's command (H1–H6). | **Delivered** (report §9). |
| **U6.7** guard-deny probe | `devops` | Plant a deny (D1), a blocked failure (D2) and an undecided call (D3) against the guard wiring. | **Delivered** (report §10). |
| U5.5 fix round | `tdd-engineer` | §3.4 in full, P8-F3–F6, and the §5 U5.5 rows. Fixture shapes come from the U6.1 report, U6.2's §9 and U6.7's §10. This replaces the ledger's two inline background-spawn rulings for U5.5. | §5 rows |
| U5.6 campaign and driver CLI | `coder` | Plan4's brief and the ledger's carried U5.6 items, with these changes:<br>- `next`/`ingest` become library entry points called by `run`, and `next` returns `stop` while `store/pause` exists;<br>- `issued` carries `sessionId`, and reconciliation locates runs by it;<br>- the `interrupted`, `launchFailed` and `launchFailing` events;<br>- `resolve --skip-batch` and `dropped.safeguardSkipped` (§3.5);<br>- `batchTimeout` and `launchFailBound` in `init`. | §5 rows |
| **U6.3** `driver run` | `coder` | §3.2: the runner loop, the child command and environment allowlist, the user-settings refusals, the `store/refusal` file, the process group and the timeout, `runner.json`, the lock with `pass_fds`, `pause [--now]`/`resume`, and the root refusal. | §5 rows |
| U5.7 agent definitions and install | `coder`; `cobb` consulted at the gate | Plan4's brief as amended by §3.3: the three definitions, the driver template and its `spawnTemplate.driver` hash, settings-only guard wiring with `onFailure: "block"`, the guard without `Bash` and with canonical paths and fail-closed decisions, and `guard_command` and the user-settings record at install. Also the ledger's carried U5.7 items (P5-F5 token smoke). | §5 rows. The **human-started smoke** runs through `driver run` (once, by hand, not the timer):<br>- its first spawn shows F2's foreground shape, recorded in `HISTORY.md`;<br>- the 50-item `pt_br_item_at_cap` batch ingests with no void, and its largest page is ≤ 22,500 tokens;<br>- the `maxTurns: 2` batch records the cut-short shape and its terminal entry;<br>- whether `SubagentHandback` appears, and whether it is hooked, is recorded;<br>- §3.3's injected-context rows pass.<br>U5.5 then swaps its provisional `maxTurns` fixture for the recorded one (a one-fixture follow-up, same owner). |
| **U6.4** unattended schedule | `devops` | §3.2 "Scheduling": install the timer (a systemd user timer, else cron) for one campaign root. Document in `decision-models/README.md` how to install and disable it, `pause [--now]`/`resume`, and how to read `status`. | On a smoke campaign:<br>- two consecutive fires: the first runs a batch, and the second, fired while the first is running, exits 0 on the lock;<br>- `pause`, issued while a run is active, lets the batch finish, and the run then exits;<br>- a fire after that exits at once;<br>- disabling the timer removes it from the scheduler's list. |
| **U6.5** requirements decision log | `tico` | Append Q-1 → "unattended relaunch per usage window OK" (stakeholder, 2026-11-01), with the FR/AC it affects, to `docs/requirements/typed-decision-models.md`'s decision log. The generator-flag rule in §3.5 gets its own entry. | The entries exist, one question → one answer each. |
| **U6.6** dataset contract vector | `coder` (model-bench, then the decision-models copy) | §3.7: `dataset-shape.jsonl`, the importer tests that read it, the importer's `labelVotes`-key refusal, and the byte-copy with its guarded-equality test. | Both suites green. A planted `labelVotes` key `"True"` is refused by the importer. Removing `rngSeed` from the canonical file reddens the decision-models equality test until the copy is re-made. |
| **U6.8** the vector's incomplete run | `coder` (model-bench, then the decision-models copy) | §3.7's second `runs[]` entry, added to each line of the canonical file: `runId` `run_0002`, `batchId` `bat_0002`, `status`, `voidClass` and `voidReason` all `"incomplete"`, `tokens` all 0, and the keys §3.7 does not name as run 1's. `test_the_vector_generation_is_complete` checks every run, not one (`(run,) =` today). Then the byte re-copy. | Both suites green, and the copies are `cmp`-equal. A new model-bench test asserts that each line's two runs differ in type exactly on §3.7's seven keys. Setting the new run's `model` to a string reddens it. |

**Order:**
- U6.7 → U5.5 fix round → U5.6 → U6.3 → U5.7 (with its smoke) → U6.4 → U5.8 onward, per plan4 §4.3.
- U6.6, then U6.8, run any time before U5.10, and U5.10's brief adds §3.7's check.
- U6.5 can run at any time.
- The guard pilot's live windows (U5p) use `driver run` under U6.4's timer, with no human start per window. Its first spent window is §3.4's usage-shape red check.

## 5. Test strategy additions

These rows replace plan4 §5's U5.5 "No `tool_result` is incomplete" clause and add to the U5.6, U5.7 and U5.10 rows. Fixtures are hand-built from the U6.1 and U6.2 shapes; the trimmed real excerpts serve as shape checks.

| Unit | Behaviour |
|---|---|
| U5.5 | **Completion.**<br>- F2's foreground shape with an `end_turn` reply → `CLEAN`. The same run ending in one `SubagentHandback` → `CLEAN`, whether hooked-allow or unhooked.<br>- H3a's kill shape (the driver ends at its `Agent` `tool_use`, with no `hook_success` and no `tool_result`) → `REISSUE incomplete`, also with F5's `stopped` `<task-notification>` present. A mutant that lets `hookMissing` read an in-flight tool use reddens (it gives `HALT`). With the driver's `Agent` result still absent, the same holds for a subagent whose last `tool_use` has no guard entry, for parallel calls killed as `tool_use`, `tool_use` with no attachments, and for an `Agent` `tool_use` with no subagent transcript.<br>- `async_launched` with a complete, valid answer, also in H2's shape (a notification, then a second driver turn) → `HALT backgroundSpawn`. A mutant that takes `async_launched` as completion reddens.<br>- `completed` with `finished` absent → `HALT spawnShapeUnknown`, and so does an unknown `status`.<br>**Safeguards.**<br>- A subagent `refusal` followed by a malformed partial `Write` → `HALT safeguardStop`, not a counted `malformed`.<br>- A `refusedMessageId` user entry with no `refusal` entry → `HALT safeguardStop`, and so does a `refusal` with no such user entry.<br>- A mutant that drops the row's precedence reddens.<br>**Usage.**<br>- A subagent `<synthetic>` `rate_limit` with an unknown `status` → `WAIT`, not `spawnShapeUnknown`. A mutant ordering `spawnShapeUnknown` first reddens.<br>- A driver `rate_limit` with no `Agent` call → `WAIT`, ten times with no halt.<br>- A `rate_limit` plus `async_launched` → `HALT backgroundSpawn`.<br>**Launch.**<br>- No transcript → `launchFailed`, uncapped. With `launchFailBound` 6 h: one more `launchFailed` at 6 h 1 min after the first, with none producing anything in between → `HALT launchFailing`; at 5 h 59 min → no halt. A launch that produced an `Agent` call or a `<synthetic>` in between resets the clock. A driver transcript with the prompt only and a non-zero exit → `launchFailed`; with exit 0 → never spawned.<br>**Interrupted.** Incomplete twice for one batch, the second named by an `interrupted` event → no halt. Without the event → halt.<br>**Guard bound.**<br>- H6's allowed shape (`accept`/`hook` with a guard entry) → no row.<br>- U6.7's D1 and D2 shapes, verbatim from report §10.3: `is_error`, `reject`/`hook`, `toolDenialKind`, no attachment → each `HALT guardRejected`. A mutant that counts either reddens.<br>- D3's shape (`accept`/`config`, no attachment) → `HALT hookMissing`, and so does `accept`/`hook` with no guard entry, not a counted `answerMissing`.<br>- A tool use with no attachment, `is_error` and no `permissionDecision` (the keys of attempt-1 `c5ce5291` line 48) → counted, not `hookMissing`.<br>- The U5.7 bounds axes, each planted against the audit's own bounds code: an allowed out-of-bounds case → `HALT guardBreach`.<br>- A guard-allowed `Write` to H6's planted path outside `exchange/ans/` → `HALT guardBreach`. A mutant that classes it as run-attributable reddens.<br>- A guard-allowed `Write` to `exchange/ans/<other>.jsonl` → counted, not a halt.<br>- Flipping `requestNonInteractive` either way changes no class.<br>**Driver.** A driver `message.model` of `claude-opus-5-5` → `HALT`.<br>**Hooks.**<br>- Two allowing guard entries → `CLEAN`.<br>- A non-guard `command` only → `HALT hookMissing` (P8-F4). So does a guard-command `hook_success` whose stdout is empty or carries no decision.<br>- In a completed run, a mid-transcript tool use with no guard entry → `HALT hookMissing`: nothing is in flight there.<br>- A foreign `SubagentStop`-shaped entry with the driver's `agent_type` changes nothing.<br>**Terminal.** Two hand-backs → `toolOutOfBounds`. A `Write` after the hand-back → counted. A rejected hand-back → `HALT guardRejected`.<br>**P8-F3** (m4(b)): the driver no longer has `Bash`, so plant a second `tool_result` with a different `tool_use_id` in the driver transcript and drop the spawn's own. That must give `REISSUE` (and `CLEAN` under the id-ignoring mutant P1). A6's pair is re-planted on a counted row, since a reject now halts: a pre-hook validation error + `rate_limit` gives `WAIT` by the precedence above, and + `authentication_failed` gives `HALT`.<br>**P8-F5, F6:** as Pass 8 states. |
| U5.6 | `issued` carries `sessionId`, and reconciliation reads only that transcript: a planted second driver session in the same directory is ignored. `next` returns `stop` while `store/pause` exists and serves again once it is gone, with the ledger byte-equal across the pause. `init` records `batchTimeout` and `launchFailBound`. `resolve --skip-batch` drops the batch's seeds or items into `dropped.safeguardSkipped` and re-issues nothing. Ledger byte-equal after each planted re-ingest, per the carried ruling. |
| U6.3 | These tests use a fake `claude` executable on `PATH`, written by the test. Nothing calls the real CLI.<br>- **Environment and argv:** the child receives exactly the allowlisted environment plus the variable, so a planted `ANTHROPIC_API_KEY` and `CLAUDE_CODE_EFFORT_LEVEL` are absent. The argv equals §3.2's command, and `--fallback-model` is absent.<br>- **User settings** (a fake `HOME`): `autoContinueAtUsageLimit: true`, an `env` key, a `hooks` key or a `permissions` edit each refuse the launch with no ledger write, and `status` prints the reason. A `theme` or `model` edit does not refuse.<br>- **Single-flight:** a second `run` against a live runner's lock exits 0 without launching.<br>- **Timeout:** a child that sleeps past `batchTimeout` is killed as a group, and the batch is ingested.<br>- **Pause (M2):** `pause` succeeds while a fake child holds the lock, and the run exits after that batch. `pause --now` stops the runner and its child, and the ledger gains exactly one `interrupted`. The next `run` reconciles the batch without counting it toward the cap.<br>- **Orphan:** a fake child outlives a SIGKILLed runner. A new `run` exits 0 on the lock without launching or signalling. Once the child ends, the next `run` reconciles. `pause --now` with a `runner.json` naming a live process with a different start time signals nothing.<br>- **Root:** a root inside a git work tree is refused. |
| U5.7 | Plan4's U5.7 row, plus: `dm-driver` pins `claude-sonnet-5-5`, has no `Bash` and has `maxTurns: 3`. No definition carries `hooks`. `settings.json` wires `guard_hook.py` once, with `onFailure: "block"`, and `install` refuses a copy without it. **Bounds axes** (P3-m3), shared with U5.5, each a planted case with its expected value written in the test: relative or absolute × `..` × a symlinked file or directory × a sibling prefix (`ans-x/`) × the bound directory itself × an empty or non-string path. An unknown or missing `agent_type`, a missing `tool_input`, and a planted exception each exit 1 with no decision. The guard allows `SubagentHandback` only for the two subagent roles, if §3.3's condition holds. `promptSha256.spawnTemplate.driver` is recorded. `install` records the user-settings hash and keys. |
| U6.6, U6.8 | As their done-conditions in §4. |
| U5.10 | Plan4's row, plus §3.7's shape check of `finalize` output against `dataset-shape.jsonl`'s copy.<br>- **Passes:** the fixture campaigns' runs come from `audit_run` on `tests/test_audit.py`-style transcripts, among them an incomplete run whose agent transcript is cut after the prompt entry (four facts `null`) and a `spawnPromptMismatch` void from an empty agent transcript (all five `null`). A message-type campaign holds an opening and a follow-up item.<br>- **Reddens:** a planted `labelVotes` key `"True"`, a missing `rngSeed`, a run's `attempt` written as `"1"`, and an item's `genRound` written as `true`.<br>- **Refused for no counterpart:** a `formMismatch` entry on a join-in output.<br>- **Mutants:** comparing against the vector's first entry only, or passing an entry with no counterpart, each redden. |

## 6. Risks & open questions

- **The guard is the only bound** (§3.3). A guard that wrongly allows a call lets it run, possibly outside `exchange/`; `guardBreach` halts only after the fact. U5.7's bounds axes and U6.7 are the prevention.
- **Plan4 R-18 (hooks depend on folder trust)** does not bite under `-p`: settings hooks fired in a folder never trusted (H1). `hookMissing` still turns a skipped guard into a halt, never a silent pass. So does a `settings.json` that `-p` silently ignores for failing validation.
- **The usage-limit shape under `-p` is unknown until the first spent pilot window.** §3.4's time-bounded `launchFailed` and the usage-over-`spawnShapeUnknown` precedence keep an unknown shape from halting in every window. A halt there is the red check.
- **Orphaned children.** An orphan holds the lock until it ends. If it hangs, the campaign stalls until a human disables the timer and kills it. `status` names the last `issued` `sessionId` to look for. This is accepted: an orphan needs the runner itself to die, which is rare.
- **The subagent-side safeguards shape is inferred** from the main-session shape (F7). The row matches either of two structured markers. A flag that leaves neither would surface as a counted void on an Opus run. U5.7's smoke and the pilot are where such a shape would first appear.
- **Generator flags** follow the stakeholder's rule (§3.5): a halt per flag, nothing counted, and a choice of retry, skip or a model move at `resolve`, with no automatic fallback.
- **A usage stop costs a driver session start** at each timer fire within a spent window: at most 48 short Sonnet turns a day, none of which costs an attempt.
- **Scheduler availability on WSL2.** Neither systemd nor cron may be running. U6.4 checks this first and records the choice.

## 7. Ready to implement

Plan: `docs/plans/typed-decision-models6.md` v1.8 (extends plan4).
- The campaign driver becomes `driver run`: a stdlib runner that launches one `claude -p --agent dm-driver` session per batch, with `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`, an environment allowlist, refusals on named user settings (shown by `status`), a runner-chosen `--session-id`, a timeout, and an `flock` lock passed to the child. Pausing is a lock-free `store/pause` file, and a 30-minute user timer starts the runner.
- `dm-driver` is pinned to `claude-sonnet-5-5` and loses `Bash`. The guard, the only bound on tool use, is wired once with `onFailure: "block"` (§3.3).
- The audit rows and their precedence are §3.4's.
- Generator flags halt, with a stakeholder choice of retry, skip or model move. `splits.json` and `labelVotes` get a contract vector (U6.6).
- **Start order:** U6.7 → U5.5 fix round → U5.6 → U6.3 → U5.7 smoke → U6.4. U6.5, U6.6 and U6.8 run at any time, with U6.6 then U6.8 before U5.10.
