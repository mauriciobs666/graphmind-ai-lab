# Typed Decision Models — Implementation Plan, Headless Driver and Probe-Informed Audit

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.2 · **Extends:** `docs/plans/typed-decision-models4.md` · **Reviews:** `docs/reviews/typed-decision-models6.md`

*2026-11-01 — v1.2: review Pass 2 (P2-M1, P2-M2, P2-m1–m3, P2-n1); runner orphan paths removed, refusals keyed on named settings, `launchFailed` bound made a parameter.*

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

**Not probed**, and still needed. U6.2 covers the first four, and §3.4 says where the last two first appear:
- spawn shape, hooks and kill shape under `claude -p`;
- whether `-p` (which skips the trust dialog, per `claude --help`) runs project-settings hooks;
- whether the runner's environment allowlist authenticates, and what `--permission-prompts none` denies;
- whether `--session-id` fixes the transcript path;
- what a usage-limit stop looks like under `-p`, for the driver and for a subagent;
- what a safeguards stop looks like in an Opus *subagent* under `-p`.

The local corpus of 238 runs, all `requestNonInteractive: true`, `async_launched` and ending in a hand-back, comes from the SDK-hosted coordinating session. It is not evidence for `claude -p`: `requestNonInteractive` is `true` for interactive background spawns too (S1).

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
- **Interactive session in tmux, relaunched per window.** This is the only mode whose spawn shape is probed (F2), so it stays the **fallback** if U6.2 fails. It is rejected as the default for the reasons above.
- **The runner launches `claude -p --agent dm-labeler` directly, with no driver.** The labeller would then be a *main* session. Its isolation (`omitClaudeMd`, `maxTurns`, the empty `instructions` set, the effort pin) and §3.3's location rule are all built on the subagent shape. Dropping the driver would mean re-deriving and re-probing all of them.
- **One `-p` session per window, looping internally.** It keeps the in-LLM loop and its drift and gives up the per-batch session id. It saves only a session start per batch, about one short Sonnet turn.

**Spawn mode and environment.**
- Every driver session runs with `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`. That is the only switch that gave foreground spawns (F2). A background spawn is a halt (§3.4); the campaign never supports it.
- **This rests on U6.2.** F2 was observed in an interactive session only. If U6.2 shows `-p` does not give foreground spawns with the variable set, §3.1 returns to `architect`, and the fallback is the tmux session.

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
- stdout and stderr go to `store/runs/<uuid>.json` / `.err`, with the exit code. The audit classes runs from transcripts. These files are evidence only for a launch that left no transcript (§3.4), and they are recorded in that event.
- `--setting-sources project` is added **only if** U6.2's H4 shows that authentication still works with it and that user-level hooks no longer fire.

**User settings.** When `--setting-sources project` is ruled out, `~/.claude/settings.json` still applies to the child (m2). The user's ordinary edits to that file (`model`, `theme`, `editorMode`, `verbose`) must not stop the campaign. So `install` records only a sha256 over the canonical JSON of its `permissions` and `modelSettings` values. Before each launch, `run` refuses if any of these holds:
- an `env` or `hooks` key is present;
- `autoContinueAtUsageLimit: true` is set (M1). That setting would make a spent window wait out the limit instead of ending;
- the `permissions`/`modelSettings` hash differs from `install`'s.

A refusal appends nothing to the ledger. It overwrites `store/refusal` with its reason and a timestamp; a successful launch deletes the file, and `status` prints it while it exists. The `autoContinueAtUsageLimit` refusal also applies when the flag is in use, until U6.2's H4 shows that the flag stops the user file from loading.

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
- `guard_hook.py` is wired **once**, in `exchange/.claude/settings.json` `PreToolUse` with matcher `*`. Frontmatter `hooks` are removed from all three definitions.
- Settings hooks fire inside subagents with `agent_type` set (F4), so one wiring covers the driver and both subagents. Each tool use gets exactly one guard `hook_success`.
- **This rests on U6.2** showing that project-settings hooks fire under `-p`.

**Guard decisions.** Plan4 §3.4's list is unchanged, except:
- the `Bash` branch is deleted;
- `dm-driver` may call only `Agent`.
- **`SubagentHandback`:** if U6.2 or U5.7 shows a hook sees it, the guard **allows** exactly that tool for `dm-generator`/`dm-labeler`. The audit's terminal rule (§3.4) limits it to one, last.

`install` records the guard's exact `command` string as a declared expectation. P8-F4's `Expected.guard_command` compares against it.

### 3.4 Audit rows (plan4 §3.3 amended)

These replace plan4's "no `tool_result`" row and add rows. The other plan4 rows are unchanged.

| Check | Class |
|---|---|
| **Completion, foreground (the only supported shape, F2):** the driver's `tool_result` matched to the spawn's `tool_use_id` has `toolUseResult.status: "completed"`, and the meta has `requestShape: "foreground"` and `finished: true` | completed: the remaining rows decide |
| any assistant entry in **either** transcript with `message.stop_reason: "refusal"`, or any `user` entry carrying a `refusedMessageId` key (F7) | **environmental: halt**, `safeguardStop` |
| a matched `tool_result` with `status: "async_launched"`, or a meta with `requestShape: "background"` | **environmental: halt**, `backgroundSpawn` (the environment variable did not apply) |
| a `tool_result` with U6.2 H6's permission-denied shape, for a tool use the guard allowed | **environmental: halt**, `permissionDenied`. The permission system, not the run, refused. |
| the driver transcript has a `<synthetic>` error other than `rate_limit`; any driver `message.model` ≠ `claude-sonnet-5-5` | environmental: halt |
| a `<synthetic>` `rate_limit` in **either** transcript | usage stop: no attempt, uncapped |
| any other `status`; `completed` with `finished` absent; a meta `requestShape` other than `foreground` | **environmental: halt**, `spawnShapeUnknown` |
| no matched `tool_result` and no error `<synthetic>` (killed, timed out, interrupted, or the driver died mid-run) | incomplete: re-issue, no attempt; the 2nd for one batch halts, unless an `interrupted` event names it (§3.2) |
| a driver transcript with no `Agent` `tool_use` and no `<synthetic>`, from a child that exited 0 | never spawned (plan4's row): re-issue, no attempt; the 2nd for one batch halts |
| the runner-recorded `sessionId` has **no transcript file**, or the child exited non-zero leaving neither an `Agent` `tool_use` nor a `<synthetic>` (P2-m1: "the launch produced nothing" is one rule) | `launchFailed`: re-issue, no attempt, uncapped. The run's stdout, stderr and exit code are recorded in the event. Once launches have produced nothing continuously for longer than `launchFailBound`, the next one halts as `launchFailing`. |

**Precedence.** Rows apply in this order, and the first that matches decides:
1. the halts above, except `spawnShapeUnknown`, then plan4's other halts;
2. usage stop;
3. `spawnShapeUnknown`;
4. the no-attempt re-issues;
5. plan4's run-attributable rows.

Placing usage stop above `spawnShapeUnknown` is deliberate (M1): a subagent stopped at the usage limit may hand the driver a result of a shape never seen, and the rate-limit entry explains it. A `backgroundSpawn`, a refusal or a denied permission is a configuration or safeguards fault and halts even then.

**Why `launchFailed` is time-bounded, not capped** (M1). The usage-limit shape under `-p` is unprobed. If a driver at the limit produces nothing, a cap of one would halt the campaign in every spent window. `launchFailBound` is a parameter recorded in the `init` event (default 6 h). It must exceed one usage window on the account in use, and the stakeholder sets it at `init`. The rule depends on no particular reset period, and a longer outage (weekly limit, broken authentication, a missing binary) halts with a clear reason. The bound runs from the first `launchFailed` with no later launch that produced a transcript with an `Agent` call or a `<synthetic>`, read from the ledger's event timestamps. **Orphan stops count** (P2-m2): an orphan is ended only by a human `kill`, which writes no `interrupted` event, so its re-issue counts toward the cap like any harness fault.

**Where the usage shape is first recorded, and its red check.** U5.7's smoke cannot reach a limit by design. The first spent window of the guard pilot is the first live case. There, `teco` appends the driver's and subagent's last entries, plus the runner's stdout and stderr, to the U6.1 report's §9. The red check: **a usage stop that halts the campaign means a row is wrong**. That goes to `architect` before the next window. No probe should spend a window on purpose to provoke one.

**The safeguards row keys on structure, not text** (m3). The "safety classifier" sentence in F7's `user` entry is user-visible wording, which can change in any release without notice. The same entry carries `refusedMessageId` and `isMeta: true`, and the assistant entry before it carries `stop_reason: "refusal"`. The row matches on either structured marker, so it still fires if one is missing. Neither is documented, so both are pinned by U5.5 fixtures built from the report's §7 key lists.

**Terminal shape.** A completed run is clean only if its last assistant entry is one of:
- an `end_turn` text reply (F3);
- exactly one `SubagentHandback` `tool_use`, followed only by its `tool_result`.

The hand-back is exempt from the bounds checks and counts against `maxTurns` as the reply. It is exempt from `hookMissing` when no guard entry exists for it, and a guard `deny` on it is `hookDenied`. Any other last entry on a completed run is the provisional `maxTurns` row, which U5.7 replaces with the smoke's recorded shape.

**Ignored as evidence:**
- `<task-notification>` user entries (F1, F5), including a `stopped` notification with no `<tool-use-id>` (report §7, session `fb3a881a`). Completion is read only from the `tool_result` and the meta.
- `hook_success` entries whose `command` ≠ the declared guard command (P8-F4).
- Every hook event other than `PreToolUse`. In particular, `SubagentStop` is never read, so F6's foreign events need no filter in the audit.
- `hook-log.jsonl`. Every consumer of it (U5.8 diagnostics, `status`) filters on `event == "PreToolUse"` and `agent_type ∈ {dm-driver, dm-generator, dm-labeler}`.

**Multiple guard entries.** Each tool use needs at least one guard `hook_success`, and every guard entry must allow. §3.3 wires the guard once, and U6.2 checks that exactly one entry appears. A duplicate (for example a stray frontmatter wiring) still must not void a run.

### 3.5 Safeguards flags, and Opus as generator

- **The driver** is pinned to Sonnet (§3.3), so F7's dialog cannot recur there. If a driver turn is flagged anyway, `-p` cannot show a dialog. The transcript carries the `safeguardStop` markers, which halt.
- **An Opus `dm-generator` subagent** flagged mid-run may leave a partial tool call. Without `safeguardStop` above every counted row, a partial `Write` would void as `malformed` and be counted. The row halts first. If the user's `/config` holds "Switch automatically" and the run continues on Opus 4.8, the `message.model ≠ pin` row halts instead.
- **The stakeholder's rule (2026-11-01).** Each flag halts the run, and nothing from it is counted. There is **no** automatic fallback to another model. At the halt, the stakeholder chooses one of three actions, applied by `driver resolve`:
  - **retry:** `resolve --decision <text>`. The batch is re-issued under a new opaque name, with no attempt counted.
  - **skip the batch:** `resolve --skip-batch --decision <text>`. This is new. Its seeds (generation) or items (labelling) drop and are counted under the new `generation.json` key `dropped.safeguardSkipped: {seeds, items}`, beside plan4 §3.7's `retryCapExceeded`.
  - **move generation to another model:** `resolve --repin-model generator <id>`, with plan4 §3.3's `generation-restart` event.
- **Not probed:** a subagent-side flag under `-p`. The row keys on the main-session markers (F7). U5.7's smoke records any flag it meets. No probe should try to provoke one.

### 3.6 The probe evidence's home

`docs/test-reports/typed-decision-models-harness-probe-report.md` is the evidence record that every harness premise here cites. U6.2 appends to its §9, and U5.5's trimmed shape fixtures go under `decision-models/tests/fixtures/transcripts/`.

### 3.7 Dataset files model-bench imports (plan4 §3.7 amended; P24-m2)

Plan4 §3.7 defines `generation.json` and the item keys. These two shapes complete the contract that `finalize` (U5.10) writes and that `model-bench/scripts/import_decision_pack.py` reads. Today they are stated only in that importer's docstring and test builder.

- **`splits.json`** is one JSON object with exactly the keys `rngSeed` (int, the split draw's seed), `eval`, `train` and `calibration`.
  - Each of the three lists holds `itemId` strings. Order carries no meaning (P2-m3).
  - Every `itemId` in `items.jsonl` appears in exactly one list, and no list repeats an id.
  - The importer reads only `eval`.
- **`labelVotes`** is a JSON object whose keys are exactly the point's answer space, written as strings, with zeros included:
  - `"true"` and `"false"` for a predicate point (guard, join-in);
  - each `MESSAGE_TYPE_LABELS` label for message type.

  Its values are non-negative ints summing to `votesPerItem` (3). A JSON boolean is never a key, so the predicate spelling is fixed here, lowercase.

**Contract vector.** One file, `model-bench/tests/fixtures/decision-contract/dataset-shape.jsonl`, owned by model-bench. It has one line per decision point (3), each line `{"role", "generation", "items", "splits"}`:
- `generation` is a complete plan4 §3.7 object, with every key including `promptSha256.spawnTemplate.driver` and `dropped.safeguardSkipped`. It also carries plan5 §3.5's `dropped.formMismatch: {<form>: n}` and `dropped.bindingMismatch`, and one `runs[]` entry.
- `items` holds three items, one per split, carrying every item key.
- `splits` follows the shape above.

`decision-models/tests/fixtures/decision-contract/dataset-shape.jsonl` is a byte-copy, under the existing guarded-equality test (`decision-models/tests/test_decision_contract.py`).

Who checks what:
- **The importer's tests** read the canonical file. Each line's items pass the importer's item projection and `_eval_ids`.
- **The importer** gains one refusal: `labelVotes` keys ≠ the answer space. Today it accepts any keys summing to 3.
- **U5.10** checks its `finalize` output on its fixture campaign against the copy, for each role. Key sets must be equal, recursively for `generation.json`, as must the JSON value types, the `labelVotes` key set and the `splits.json` key set. Three containers are the exception: `dropped.formMismatch` (its keys are the forms that dropped), `runs[]` and each run's `instructions` list (their lengths come from the data). For those, only the value type of each entry is compared, against the vector's entry. The `items.jsonl` and split lists are compared per entry in the same way. U5.10's fixtures include one campaign where a form dropped and one where none did.

## 4. Units

These change plan4 §4.1's U5.5–U5.7 rows and add six units. Every code unit gets an `analyst` code gate. U5.x and U6.x serialize on `decision-models/docs/HISTORY.md` (plan4 §4.3).

| Unit | Owner | Brief | Done when |
|---|---|---|---|
| **U6.1** probe evidence record | `devops` | §3.6 | **Delivered** (`docs/test-reports/typed-decision-models-harness-probe-report.md`). |
| **U6.2** headless probe | `devops`. The stakeholder approved the spend (2026-11-01), to run once this plan passes its gate. | Rerun the U6.1 kit (Appendix I's path rewrite) under §3.2's exact command, hook wired in project settings only, no frontmatter hooks. Launch each case as `env -i <§3.2 allowlist> claude -p …`, so no variable of the launching session reaches the child (M3).<br>**H1:** the allowlist plus `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`.<br>**H2:** without the variable; record what `-p` does to a background run when the driver's turn ends.<br>**H3:** H1 with the process group SIGKILLed mid-run; then, separately, only the `claude` parent PID SIGKILLed (does the subagent outlive it?).<br>**H4:** H1 plus `--setting-sources project`.<br>**H5:** the transcript path for the passed `--session-id`.<br>**H6:** H1 with the subagent told to `Write` once outside `exchange/` (the kit's hook allows it), to record the permission auto-deny shape.<br>Per spawn, record `read_results.py`'s columns, plus guard `hook_success` entries per tool use and whether any `SubagentHandback` is hooked. | Results are appended to the report's §9, and `teco` reads them against these gates. Each failure returns to `architect`, never to a silent fallback.<br>(a) **H1** shows F2's foreground shape, authenticates under the allowlist, has no permission-denied `tool_result`, and has **exactly one** guard `hook_success` per tool use. Zero entries fails gate (b); more than one is recorded.<br>(b) Settings hooks fire in the subagent under `-p` → §3.3's wiring stands.<br>(c) **H3:** the group kill leaves no `tool_result` (the incomplete row). If the subagent outlives its parent, the timeout model changes.<br>(d) **H4** decides §3.2's `--setting-sources` clause and whether the user-settings refusals stay.<br>(e) **H5:** the path is `<slug>/<session-id>.jsonl`.<br>(f) **H6's** shape is recorded for the `permissionDenied` row's fixture. |
| U5.5 fix round | `tdd-engineer` | §3.4 in full, P8-F3–F6, and the §5 U5.5 rows. Fixture shapes come from the U6.1 report and U6.2's §9. This replaces the ledger's two inline background-spawn rulings for U5.5. | §5 rows |
| U5.6 campaign and driver CLI | `coder` | Plan4's brief and the ledger's carried U5.6 items, with these changes:<br>- `next`/`ingest` become library entry points called by `run`, and `next` returns `stop` while `store/pause` exists;<br>- `issued` carries `sessionId`, and reconciliation locates runs by it;<br>- the `interrupted`, `launchFailed` and `launchFailing` events;<br>- `resolve --skip-batch` and `dropped.safeguardSkipped` (§3.5);<br>- `batchTimeout` and `launchFailBound` in `init`. | §5 rows |
| **U6.3** `driver run` | `coder` | §3.2: the runner loop, the child command and environment allowlist, the user-settings refusals, the `store/refusal` file, the process group and the timeout, `runner.json`, the lock with `pass_fds`, `pause [--now]`/`resume`, and the root refusal. | §5 rows |
| U5.7 agent definitions and install | `coder`; `cobb` consulted at the gate | Plan4's brief as amended by §3.3: the three definitions, the driver template and its `spawnTemplate.driver` hash, settings-only guard wiring, the guard without `Bash`, and `guard_command` and the user-settings record at install. Also the ledger's carried U5.7 items (P5-F5 token smoke). | §5 rows. The **human-started smoke** runs through `driver run` (once, by hand, not the timer):<br>- its first spawn shows F2's foreground shape, recorded in `HISTORY.md`;<br>- the 50-item `pt_br_item_at_cap` batch ingests with no void, and its largest page is ≤ 22,500 tokens;<br>- the `maxTurns: 2` batch records the cut-short shape and its terminal entry;<br>- whether `SubagentHandback` appears, and whether it is hooked, is recorded;<br>- §3.3's injected-context rows pass.<br>U5.5 then swaps its provisional `maxTurns` fixture for the recorded one (a one-fixture follow-up, same owner). |
| **U6.4** unattended schedule | `devops` | §3.2 "Scheduling": install the timer (a systemd user timer, else cron) for one campaign root. Document in `decision-models/README.md` how to install and disable it, `pause [--now]`/`resume`, and how to read `status`. | On a smoke campaign:<br>- two consecutive fires: the first runs a batch, and the second, fired while the first is running, exits 0 on the lock;<br>- `pause`, issued while a run is active, lets the batch finish, and the run then exits;<br>- a fire after that exits at once;<br>- disabling the timer removes it from the scheduler's list. |
| **U6.5** requirements decision log | `tico` | Append Q-1 → "unattended relaunch per usage window OK" (stakeholder, 2026-11-01), with the FR/AC it affects, to `docs/requirements/typed-decision-models.md`'s decision log. The generator-flag rule in §3.5 gets its own entry. | The entries exist, one question → one answer each. |
| **U6.6** dataset contract vector | `coder` (model-bench, then the decision-models copy) | §3.7: `dataset-shape.jsonl`, the importer tests that read it, the importer's `labelVotes`-key refusal, and the byte-copy with its guarded-equality test. | Both suites green. A planted `labelVotes` key `"True"` is refused by the importer. Removing `rngSeed` from the canonical file reddens the decision-models equality test until the copy is re-made. |

**Order:**
- U6.2 → U5.5 fix round → U5.6 → U6.3 → U5.7 (with its smoke) → U6.4 → U5.8 onward, per plan4 §4.3.
- U6.6 runs any time before U5.10, and U5.10's brief adds §3.7's check.
- U6.5 can run at any time.
- The guard pilot's live windows (U5p) use `driver run` under U6.4's timer, with no human start per window. Its first spent window is §3.4's usage-shape red check.

## 5. Test strategy additions

These rows replace plan4 §5's U5.5 "No `tool_result` is incomplete" clause and add to the U5.6, U5.7 and U5.10 rows. Fixtures are hand-built from the U6.1 and U6.2 shapes; the trimmed real excerpts serve as shape checks.

| Unit | Behaviour |
|---|---|
| U5.5 | **Completion.**<br>- F2's foreground shape with an `end_turn` reply → `CLEAN`. The same run ending in one `SubagentHandback` → `CLEAN`, whether hooked-allow or unhooked.<br>- H3's kill shape → `REISSUE incomplete`, also with F5's `stopped` `<task-notification>` present.<br>- `async_launched` with a complete, valid answer → `HALT backgroundSpawn`. A mutant that takes `async_launched` as completion reddens.<br>- `completed` with `finished` absent → `HALT spawnShapeUnknown`, and so does an unknown `status`.<br>**Safeguards.**<br>- A subagent `refusal` followed by a malformed partial `Write` → `HALT safeguardStop`, not a counted `malformed`.<br>- A `refusedMessageId` user entry with no `refusal` entry → `HALT safeguardStop`, and so does a `refusal` with no such user entry.<br>- A mutant that drops the row's precedence reddens.<br>**Usage.**<br>- A subagent `<synthetic>` `rate_limit` with an unknown `status` → `WAIT`, not `spawnShapeUnknown`. A mutant ordering `spawnShapeUnknown` first reddens.<br>- A driver `rate_limit` with no `Agent` call → `WAIT`, ten times with no halt.<br>- A `rate_limit` plus `async_launched` → `HALT backgroundSpawn`.<br>**Launch.**<br>- No transcript → `launchFailed`, uncapped. With `launchFailBound` 6 h: one more `launchFailed` at 6 h 1 min after the first, with none producing anything in between → `HALT launchFailing`; at 5 h 59 min → no halt. A launch that produced an `Agent` call or a `<synthetic>` in between resets the clock. A driver transcript with the prompt only and a non-zero exit → `launchFailed`; with exit 0 → never spawned.<br>**Interrupted.** Incomplete twice for one batch, the second named by an `interrupted` event → no halt. Without the event → halt.<br>**Permission.** H6's denied shape on a guard-allowed `Write` → `HALT permissionDenied`, not a counted `answerMissing`.<br>**Driver.** A driver `message.model` of `claude-opus-5-5` → `HALT`.<br>**Hooks.**<br>- Two allowing guard entries → `CLEAN`.<br>- A non-guard `command` only → `HALT hookMissing` (P8-F4).<br>- A foreign `SubagentStop`-shaped entry with the driver's `agent_type` changes nothing.<br>**Terminal.** Two hand-backs → `toolOutOfBounds`. A `Write` after the hand-back → counted. A denied hand-back → `hookDenied`.<br>**P8-F3** (m4(b)): the driver no longer has `Bash`, so plant a second `tool_result` with a different `tool_use_id` in the driver transcript and drop the spawn's own. That must give `REISSUE` (and `CLEAN` under the id-ignoring mutant P1). A6's planted `hookDenied` + `rate_limit` now gives `WAIT` by the precedence above, and the re-paired `hookDenied` + `authentication_failed` gives `HALT`.<br>**P8-F5, F6:** as Pass 8 states. |
| U5.6 | `issued` carries `sessionId`, and reconciliation reads only that transcript: a planted second driver session in the same directory is ignored. `next` returns `stop` while `store/pause` exists and serves again once it is gone, with the ledger byte-equal across the pause. `init` records `batchTimeout` and `launchFailBound`. `resolve --skip-batch` drops the batch's seeds or items into `dropped.safeguardSkipped` and re-issues nothing. Ledger byte-equal after each planted re-ingest, per the carried ruling. |
| U6.3 | These tests use a fake `claude` executable on `PATH`, written by the test. Nothing calls the real CLI.<br>- **Environment and argv:** the child receives exactly the allowlisted environment plus the variable, so a planted `ANTHROPIC_API_KEY` and `CLAUDE_CODE_EFFORT_LEVEL` are absent. The argv equals §3.2's command, and `--fallback-model` is absent.<br>- **User settings** (a fake `HOME`): `autoContinueAtUsageLimit: true`, an `env` key, a `hooks` key or a `permissions` edit each refuse the launch with no ledger write, and `status` prints the reason. A `theme` or `model` edit does not refuse.<br>- **Single-flight:** a second `run` against a live runner's lock exits 0 without launching.<br>- **Timeout:** a child that sleeps past `batchTimeout` is killed as a group, and the batch is ingested.<br>- **Pause (M2):** `pause` succeeds while a fake child holds the lock, and the run exits after that batch. `pause --now` stops the runner and its child, and the ledger gains exactly one `interrupted`. The next `run` reconciles the batch without counting it toward the cap.<br>- **Orphan:** a fake child outlives a SIGKILLed runner. A new `run` exits 0 on the lock without launching or signalling. Once the child ends, the next `run` reconciles. `pause --now` with a `runner.json` naming a live process with a different start time signals nothing.<br>- **Root:** a root inside a git work tree is refused. |
| U5.7 | Plan4's U5.7 row, plus: `dm-driver` pins `claude-sonnet-5-5`, has no `Bash` and has `maxTurns: 3`. No definition carries `hooks`. `settings.json` wires `guard_hook.py` once. The guard allows `SubagentHandback` only for the two subagent roles, if §3.3's condition holds. `promptSha256.spawnTemplate.driver` is recorded. `install` records the user-settings hash and keys. |
| U6.6 | As its done-condition in §4. |
| U5.10 | Plan4's row, plus §3.7's shape check of `finalize` output against `dataset-shape.jsonl`'s copy. A planted `labelVotes` key `"True"` reddens it, and so does a missing `rngSeed`. |

## 6. Risks & open questions

- **`-p` behaviour is unprobed** (spawn shape, hooks, kill, auth, permissions, session path). U6.2's gates settle each before any code depends on it. If a gate fails, the tmux fallback needs a short plan revision; no unit after U5.5 starts before U6.2 closes.
- **Plan4 R-18 (hooks depend on folder trust)** reads differently under `-p`, which skips the trust dialog. Whether hooks then run is U6.2's gate (b). Either way, `hookMissing` still turns a skipped guard into a halt, never a silent pass. So does a `settings.json` that `-p` silently ignores for failing validation.
- **The usage-limit shape under `-p` is unknown until the first spent pilot window.** §3.4's time-bounded `launchFailed` and the usage-over-`spawnShapeUnknown` precedence keep an unknown shape from halting in every window. A halt there is the red check.
- **Orphaned children.** An orphan holds the lock until it ends. If it hangs, the campaign stalls until a human disables the timer and kills it. `status` names the last `issued` `sessionId` to look for. This is accepted: an orphan needs the runner itself to die, which is rare.
- **The subagent-side safeguards shape is inferred** from the main-session shape (F7). The row matches either of two structured markers. A flag that leaves neither would surface as a counted void on an Opus run. U5.7's smoke and the pilot are where such a shape would first appear.
- **Generator flags** follow the stakeholder's rule (§3.5): a halt per flag, nothing counted, and a choice of retry, skip or a model move at `resolve`, with no automatic fallback.
- **A usage stop costs a driver session start** at each timer fire within a spent window: at most 48 short Sonnet turns a day, none of which costs an attempt.
- **Scheduler availability on WSL2.** Neither systemd nor cron may be running. U6.4 checks this first and records the choice.

## 7. Ready to implement

Plan: `docs/plans/typed-decision-models6.md` v1.1 (extends plan4). It goes to the `analyst` plan gate (Pass 2) first.
- The campaign driver becomes `driver run`: a stdlib runner that launches one `claude -p --agent dm-driver` session per batch, with `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`, an environment allowlist, refusals on named user settings (shown by `status`), a runner-chosen `--session-id`, a timeout, and an `flock` lock passed to the child. Pausing is a lock-free `store/pause` file, and a 30-minute user timer starts the runner.
- `dm-driver` is pinned to `claude-sonnet-5-5` and loses `Bash`. The guard is wired once, in project settings.
- The audit accepts only foreground completion and halts on background, refusal markers and permission denials. A rate limit in either transcript is a usage stop. A launch that produces nothing is uncapped for up to `launchFailBound`. A `pause --now` interruption is not counted.
- Generator flags halt, with a stakeholder choice of retry, skip or model move. `splits.json` and `labelVotes` get a contract vector (U6.6).
- **Start order:** U6.2 (gated by this plan's gate) → U5.5 fix round → U5.6 → U6.3 → U5.7 smoke → U6.4. U6.5 and U6.6 run at any time, with U6.6 before U5.10.
