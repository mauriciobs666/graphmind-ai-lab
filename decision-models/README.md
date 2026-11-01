# decision-models

Builds the data and the candidates for **typed decision models**: synthetic, labelled datasets per
decision point (guard, message type, join-in), split by seed, and later the fitted candidate
configurations that `model-bench/` measures. Nothing here runs inside falkor-chat, and nothing here
decides whether a decision model ships: `model-bench` measures, and a human edits falkor-chat's
config.

Requirements: `docs/requirements/typed-decision-models.md` (repo root).
Design: `docs/plans/typed-decision-models.md` (v1.3), extended by `typed-decision-models2.md`,
`typed-decision-models3.md` and `typed-decision-models4.md`; the later plan wins on a conflict.
Method: `docs/plans/typed-decision-models-ml.md`.

## What this component is, and is not

- **Zero runtime dependencies, local-only, human-started.** Python 3.12 stdlib, a host venv, no CI
  hook. It runs because a person typed a command, or because a user-level timer they installed
  started `driver run` (plan6 §3.2; the timer is U6.4's, not installed yet).
- **No hosted API client.** Generation and labelling are done by Claude Code subagents through
  file exchange, driven from a campaign directory (plan4 §3); nothing in `decisionmodels/` calls
  a hosted API or holds a key.
- **Synthetic inputs only (FR-8).** `decisionmodels/` imports only the standard library and
  itself, and its source names no falkor-chat path, no FalkorDB/Redis reach and no side-check
  item file. `tests/test_fr8_isolation.py` is a static tripwire for that, with a planted-red case
  per reach; its docstring lists what a source scan cannot see.
- **One-way to model-bench.** model-bench's importer copies an eval split and a candidate in;
  this component never imports model-bench.

## Quick start

```bash
decision-models/setup.sh                  # create .venv, install the package + dev extra
cd decision-models && .venv/bin/pytest -q # run the suite from the component directory
.venv/bin/ruff check .                    # lint
```

## Status

The scaffold (U5.1), the guard contract copies (U5.2: the guard input renderer and the
`input_hash` leakage key, pinned by byte-copies of model-bench's golden vectors), and the
decision points' definitions with seed generation (U5.3: `points/<role>/` rubric, generator
template and design per point; `decisionmodels/seeds.py`), and the request/answer exchange
(U5.4: `decisionmodels/exchange.py` writes closed, paged, id-blind requests;
`decisionmodels/ingest.py` checks answers), the `input_hash` rows for join-in and message
type (U5.2b-hash), and the transcript audit (U5.5: `decisionmodels/audit.py` locates a subagent
run from the driver's transcript and classes it per plan6 §3.4 over plan4 §3.3), and the
campaign ledger with its driver CLI (U5.6: `decisionmodels/campaign.py`,
`decisionmodels/driver.py`), and the unattended runner with its stop switch (U6.3:
`decisionmodels/runner.py`, `driver run`, `pause [--now]`, `resume`). Labelling requests render
guard items only until U5.2b-render.
The layout the later units fill is plan4 §3.1 over v1.3 §3.4; delivery is recorded in
`docs/HISTORY.md`.

## Running a campaign

A campaign lives in its own root, outside any git work tree and any folder under a `CLAUDE.md`
or `AGENTS.md`. Run these from `decision-models/`:

```bash
.venv/bin/python -m decisionmodels.driver init --point guard-judge --phase smoke \
    --root ~/dm-campaigns/smoke --fixture items.jsonl     # smoke: one label batch from a fixture
.venv/bin/python -m decisionmodels.driver init --point guard-judge --phase pilot \
    --root ~/dm-campaigns/pilot --launch-fail-bound 6h    # --batch-timeout defaults to 45m
.venv/bin/python -m decisionmodels.driver status --root ~/dm-campaigns/pilot
.venv/bin/python -m decisionmodels.driver resolve --root ~/dm-campaigns/pilot \
    --decision "why the halt is cleared"   # or --skip-batch, --repin-instructions,
                                           # --repin-model generator|labeler <id>
```

- **`init`** refuses, writing nothing, a root that is not empty or sits in a git work tree or
  under a `CLAUDE.md`/`AGENTS.md` (judged on its physical path, so a symlink does not hide one),
  `--phase smoke` without `--fixture`, a pilot for any point but guard, and `--phase full`
  unless the guard pilot's report (`data/guard-judge/pilot/pilot-report.json`) passes the spend
  gate, records `stakeholderDecision.cap` as 25 or 50 and `rngSeeds` as a non-empty list of
  ints, and the previous point's dataset is finalized. A full campaign holds the pilot's RNG
  seeds, so it never reuses one. `--launch-fail-bound` must exceed one usage window on the
  account in use.
- **A torn ledger** (a last line with no newline, from a crash mid-append) or a line that is not
  an event refuses every command, naming the byte offset. A torn last line is always safe to truncate,
  because every writer appends its event before acting on it; any other bad line needs a hand repair.
- **`status`** prints the batches by state, a halt and its reason, whether `store/pause` exists,
  a launch refusal (`store/refusal`), the last issued session id, and any dropped batches.
- **`resolve`** is the only way past a halt. Retry re-issues the halted batch under a new name;
  `--skip-batch` (a safeguards stop only) drops it and counts it as `safeguardSkipped`; a
  re-pin records the new model (a generation or labelling restart) or instructions set.
- `install` (U5.7) and `finalize` (U5.10) are not built yet. `run` refuses a campaign that has
  no `install` record, so no batch is spawned today.

### `run`, `pause` and `resume`

```bash
.venv/bin/python -m decisionmodels.driver run --root ~/dm-campaigns/pilot   # the timer's command
.venv/bin/python -m decisionmodels.driver pause --root ~/dm-campaigns/pilot  # after this batch
.venv/bin/python -m decisionmodels.driver pause --now --root ~/dm-campaigns/pilot  # at once
.venv/bin/python -m decisionmodels.driver resume --root ~/dm-campaigns/pilot
```

- **`run`** loops: reconcile and pick the next batch, record `issued` with a fresh session id,
  start one `claude -p --agent dm-driver` session in `exchange/` (its own process group, a
  minimal environment, `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`), wait up to `batchTimeout`
  (killing the group on expiry: SIGTERM, then, 30 s later, SIGKILL to any process of the group
  still alive), then ingest. The child's
  stdout, stderr and exit code land in `store/runs/<sessionId>.json`, `.err` and `.exit`.
- It **exits 0** when the phase is done, on a halt, on a usage stop, while `store/pause` exists,
  after a launch that produced nothing (`launchFailed`: the next fire tries again), and when
  another `run` or an orphaned child holds `store/driver.lock` (one line: "another `driver run`
  holds store/driver.lock"). Its last line says why (`stopped: done`, `stopped: halted (…)`, …).
- It **refuses** (exit 1, nothing appended to the ledger) a root inside a git work tree or under
  a `CLAUDE.md`/`AGENTS.md`, a campaign with no `install` record, and, checked before every
  launch, `~/.claude/settings.json` with a forbidden key (`env`, `hooks`, `fallbackModel`,
  `apiKeyHelper`, `disableAllHooks: true`, `autoContinueAtUsageLimit: true`; refused even if
  `install` saw it), or in which any other key outside `runner.UNRECORDED_SETTINGS` (`model`,
  `theme` and the other terminal-only keys) was added, removed or changed since `install`
  recorded its digest; an unknown key counts, and a new `permissions.allow` entry or
  `defaultMode` does not. The refusal names the keys, never a value, and ends "revert it, or run
  `driver install` to accept it". The reason, with a timestamp, goes to `store/refusal`, and
  `status` shows it until a launch succeeds:
  ``REFUSED: 2026-11-02T09:00:00+00:00 launch refused: ~/.claude/settings.json has the
  forbidden `hooks` setting; remove it``.
  Fix the cause; the next fire launches and deletes the file. The root refusal is the exception:
  `run` checks it before taking the lock and writes nothing, and `status` checks the root
  itself, printing `REFUSED: <reason>` (no timestamp) for as long as the cause is there.
- **`pause`** creates `store/pause` (no lock, no ledger): a running `run` finishes its batch and
  exits, and every fire exits 0 at once, before the settings check (so pausing to edit
  `~/.claude/settings.json` refuses nothing), until **`resume`** deletes the file. `status` shows
  `PAUSED`.
- **`pause --now`** also sends SIGTERM to the runner named in `store/runner.json`, only if that
  process is alive with the recorded start time. The runner kills its child's group, records
  `interrupted {batch}` (that batch's re-issue is not counted toward the no-attempt cap) and
  exits; the next `run` after `resume` reconciles the batch. Ctrl-C on a hand-started `run` does
  the same.
- **An orphan** (a child whose runner died) holds the lock until it exits, so every fire exits
  on the lock; `pause --now` does not signal it. Stop it by disabling the timer and killing the
  `claude` process by hand; `status`'s "last issued" names the session to look for.
