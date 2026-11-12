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
  hook. It runs because a person typed a command: campaigns are started by hand with
  `driver run --root <campaign>`, and no timer ships.
- **No hosted API client.** Generation and labelling are done by Claude Code subagents through
  file exchange, driven from a campaign directory (plan4 §3); nothing in `decisionmodels/` calls
  a hosted API or holds a key.
- **Synthetic inputs only (FR-8).** `decisionmodels/` imports only the standard library and
  itself, and its source names no falkor-chat path, no FalkorDB/Redis reach and no side-check
  item file. `tests/test_fr8_isolation.py` is a static tripwire for that, with a planted-red case
  per reach; its docstring lists what a source scan cannot see. It also scans `scripts/` and
  `agents/*.py`, under a per-file exemption map that lets only
  `scripts/import_side_check_shingles.py` name the side-check pack.
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
`decisionmodels/runner.py`, `driver run`, `pause [--now]`, `resume`), and the three agent
definitions, the guard and `driver install` (U5.7: `agents/`, `decisionmodels/install.py`; its
live smoke ran, `docs/test-reports/live-smoke-report.md`), and generation runs with the
generated-text screen, form obedience, the template halts and the generator restart (U5.8a:
`Campaign.begin_round`, `decisionmodels/screen.py`; no `driver` subcommand queues a round yet), the
votes of a settled round (U5.8b: `Campaign.queue_votes`, `decisionmodels/labels.py`), and the guard
pilot (U5.8c: `decisionmodels/pilot.py`, below), and rounds and top-up (U5.9: the message-type
tranche, the top-up planner, the spend halt and `decisionmodels/diagnostics.py`, below), and
`finalize` with the cells, splits and `generation.json` (U5.10: `decisionmodels/finalize.py`).
Labelling requests render all three points' items (U5.2b-render: the join-in and message-type
renderers, pinned by byte-copies of model-bench's render vectors and `normalize-turns.jsonl`).
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
.venv/bin/python -m decisionmodels.driver init --point guard-judge --phase full \
    --root ~/dm-campaigns/guard      # then `install`, `start` (round 1), `run` per window
.venv/bin/python -m decisionmodels.driver status --root ~/dm-campaigns/pilot
.venv/bin/python -m decisionmodels.driver resolve --root ~/dm-campaigns/pilot \
    --decision "why the halt is cleared"   # or --skip-batch, --repin-instructions,
                                           # --decline-topup, --reopen-topup, --probe-topup,
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
  re-pin records the new model (a generation or labelling restart; a labelling restart retires every
  label batch, the smoke fixture's too) or instructions set. It prints the halt it cleared and, if
  a `--skip-batch` drop closed a round or pool, the halt that raised. A `spend` halt (below) is
  let through by a plain `resolve`, or ended by `--decline-topup`; a `rubric` halt is cleared only
  once the rubric has changed; an `unboundedCell` halt refuses a plain `resolve` (below).
- **`start`** draws a `full` campaign's round 1 (⌈1.1 × the point's target⌉ seeds) and queues its
  generation runs; it spends no tokens, `run` does. It refuses, writing no ledger event (it may leave `store/driver.lock`), a smoke or pilot
  campaign (a pilot starts with `pilot start`), one with no `install` record, a held lock, and a
  second call. After a crash between the draw and the queued runs `start` says the round is
  drawn; `run` queues the stranded seeds. Votes, the message-type tranche and top-ups then start
  from `run` by themselves.
- `run` refuses a campaign that has no `install` record.
- `finalize --root <dir> --dataset-version <name> [--data-dir <dir>] [--rng-seed <n>] [--short]` (U5.10) turns a
  finished `full` campaign into `data/<role>/<name>/`: `items.jsonl`, `splits.json`,
  `generation.json` and `diagnostics.json`. It prints every cell's final n, the refill's figures
  and the `dropped` counts, and never overwrites a version (a directory with no `finalized` event, left by a crash
  between the rename and the event, is named as such in the refusal: move it aside or pick
  another name). It refuses (writing nothing) a campaign that is not
  finished, halted or unlabelled, and:
  - a message-type point whose `rubric.md` changed since the tranche was labelled (both hashes are
    named; restoring the file clears it; a rubric fix after the tranche passed means a new
    campaign);
  - a second generator-template hash unless its first run came after a `resolve` of a template
    halt (only runs after the last generator restart count);
  - a short cell: with no decline recorded the top-up is still due; after `resolve
    --decline-topup` the text names `finalize --short` and `resolve --reopen-topup`. **`--short`**
    (needs that decline) finalizes with the short cells printed. Every finalize, `--short` or not,
    holds each cell's eval and calibration n to the point's floor after the splits and the refill;
    a cell below it refuses, writes no dataset and appends a `floor-short` event to the ledger (per
    short cell its labelled items `n` and the `need` more it would take, plus the RNG seed, which
    is then held: finalize again with `--rng-seed <n>`).
  A message-type point is also rechecked over every labelled item with the tranche's 3% rule. A
  failure appends a halt (`recheck`, the other write a refusal makes); `resolve` then needs a changed
  rubric and relabels the whole point (all three votes). The second failure is cleared only by
  `resolve --decision ... --stakeholder-ref "<who decided, where recorded>"` given alone (a
  `--repin-model` or `--repin-instructions` beside it is refused, and a ref on a halt that is not
  a second recheck failure is refused too), after which `finalize` goes ahead with the ties
  counted as `voteTie` drops.

### Cells, splits and the side-check hashes (U5.10)

`finalize` calls `decisionmodels/shingles.py`, which
tokenizes (NFKC, lowercase, non-letter/digit to space) and cuts 8-token shingles over an item's
evidence text only (a guard's `condition` and the agent's name are not evidence).
`decisionmodels/split.py` `make_splits` drops, first, every generated item whose evidence shares a
shingle with a side-check item; cuts each cell (design stratum × label) at random to its target,
keeping the unused candidates; assigns eval, then calibration, then train within each cell by seed;
and drops last, and counts, any eval item that shares a shingle with a calibration or train item.
Each such slot is then refilled from the same cell's unused candidates (shuffled on the same RNG,
walked until eval is full or they run out; a candidate that shares a shingle with calibration or
train is a refill drop). Calibration and train are untouched, and a run with no drop is what it was
before the refill. The printed cell line gives eval-assigned, assigned drops, candidates walked,
refill drops, refilled, floor, remaining surplus and the drop rate `d` (flagged above 0.25: a
generation-diversity signal). A cell is keyed by the item's **majority** label.
`train` is empty (note §Q1.3's train column is U8's), so `splits.json` is `{rngSeed, eval, train:
[], calibration}`; only items in a split reach `items.jsonl` (one line per item, the U+2028, U+0085 and U+2029
line separators escaped as in `exchange.dump_line`, since model-bench splits on them), and `generation.json` is plan4 §3.7's
(`model` from the ingested runs' transcripts, `promptSha256` from the last-issued run with `rubric`
from `campaign.rubric_hash`, `runs[]` from the runs after the last restart). Its shape is checked in
`tests/test_finalize.py` against `tests/fixtures/decision-contract/dataset-shape.jsonl`.

The side-check items are never read by `decisionmodels/`. A human runs
`.venv/bin/python scripts/import_side_check_shingles.py` once; it reads
`model-bench/packs/guard-judge-understanding/items.jsonl` and writes only sha256 hashes of its
shingles, with the source file's sha256 in a header line, to
`data/guard-judge/side-check-shingles.txt` (committed; a test fails when the source changes).

### Rounds, top-up and the message-type tranche

In a `full` campaign `driver run` keeps going after round 1 is labelled:

- **Top-up.** A cell is a label within a design stratum (guard's evidence kind; none for the other
  two points) and its target is eval plus calibration (200, 120 and 200). While a cell is short,
  the driver draws the next `genRound`: per short stratum, the most any short cell needs,
  `⌈shortfall / P̂(label | stratum)⌉`, from the **same design restricted to that stratum**, never
  to an intent. All strata are drawn before a run issues, and their seeds are cut into runs
  together. Rounds repeat until every cell is full. A cell that a `floor-short` record names
  (finalize's refusal, above) is held to `max(target, n + need)` instead, so the next `driver run`
  plans the same draw for it; a cell whose `need` is null (every eval item dropped) halts as
  `unboundedCell` like a cell with no labelled item. A short cell with no labelled item cannot be
  planned: the driver halts (`unboundedCell`), naming the cell and the Wilson 95% upper bound on
  P(label | stratum) over the stratum's labelled items, and writes no `store/refusal`. A plain
  `resolve` is refused. `resolve --probe-topup --decision ...` draws one run of 25 seeds of that
  stratum (the probe; it is spend-checked like any top-up); a hit gives a normal plan, no hit halts
  again with the new bound, and a second probe is the stakeholder's choice. `--decline-topup`
  and `--repin-model generator|labeler` also clear it.
- **Spend halt.** Before a top-up round issues, its projected tokens (this point's round 1:
  top-up seeds over round-1 seeds times its live labelling tokens, plus top-up runs times its
  mean tokens per generation run; four token classes each) are compared with round 1's total
  tokens, which include a tranche relabel's or labeler re-pin's retired attempts (a generator
  re-pin discards round 1, so its tokens do not count). Over half halts the driver (`spend`); the
  reason prints both totals. Take the projection (`status`) to the stakeholder. "Yes" is a plain
  `resolve --decision ...`; "no" is `resolve --decision ... --decline-topup`, after which no
  top-up run is issued and the store stays as it is. A changed mind is `resolve --decision ...
  --reopen-topup` (valid only with no halt open and a decline in force): the next `driver run`
  plans again, and the spend halt can fire again.
- **Message-type tranche.** Round 1 of message type labels 250 items, drawn at random, with all
  three votes before the rest. More than 3% `voteTie` drops halts it (`rubric`): edit
  `points/message-type-classifier/rubric.md`, then `resolve`, which relabels the tranche under the
  changed rubric hash (and refuses while the hash is unchanged).
- **`diagnostics.json`** (`decisionmodels/diagnostics.py`, written beside `generation.json` by
  `finalize`; `python -m decisionmodels.diagnostics --root <campaign> --out <dir>` writes it for a
  campaign in flight): pair and intent-label agreement per class, position and balance effects,
  round 1 against top-up per cell, screen hits by intent, difficulty and pattern, intent-label
  agreement split by turn cut, a count of items that repeat a hand-off prompt example (over the eval split at `finalize`), guard's
  round-1 cue check (without its verdict), and per-round, per-template and retired-run totals.
  It is exploratory and gates nothing.

### The guard pilot

The pilot is a guard campaign (`init --point guard-judge --phase pilot`, then `install`) run before
guard's full spend. `pilot.py` takes the same `--root`; `report`, `recheck` and `record-decision`
read and write `data/guard-judge/pilot/pilot-report.json` (`--data-dir` for another copy).

```bash
.venv/bin/python -m decisionmodels.pilot start --root ~/dm-campaigns/pilot   # draw 100 seeds
.venv/bin/python -m decisionmodels.driver run --root ~/dm-campaigns/pilot    # per window, by hand
.venv/bin/python -m decisionmodels.pilot screen --root ~/dm-campaigns/pilot  # n, delta, the cap
.venv/bin/python -m decisionmodels.pilot report --root ~/dm-campaigns/pilot
.venv/bin/python -m decisionmodels.pilot record-decision --outcome proceed --cap 25 \
    --decided-on 2026-11-09 --ref "<ledger line>" [--stakeholder-ref "<who and when>"]
```

- **The blocks.** `start` draws the first block: 100 seeds in 4 generation runs. Its items take
  votes 1 to 3 at 50 per batch, then a screen vote at 10 per batch; the screen vote is never one
  of the three that give `labelVotes`. While fewer than 80 items carry votes 1, 2 and the screen
  vote, `driver run` draws another block of 50 seeds by itself.
- **`screen`** prints `n`, `Δ` (vote 1 against the screen vote differs, less vote 1 against vote
  2) and the cap: `10·Δ ≥ n` gives 25, else 50.
- **`report`** evaluates the pilot and writes the report: labeller `F` and `c` per token class,
  the generator's mean tokens per run, `P̂(label | evidence kind)`, `n`, `Δ`, the cap, voids,
  reruns and screen hits, reprojected tokens per point, the RNG seeds the pilot held, and
  `missingCueCheck`. Counts only: no item or seed text. It refuses a block still being voted.
- **The `missing` cue check** reads the `understanding` items by difficulty × intended answer:
  `passed`, or `failed`, which halts the pilot (`status` shows it). `driver resolve --decision
  "<the fix>"` clears it (the fix itself is an edit under `points/` or `agents/`, then `install`),
  and `recheck` draws a fresh block of 100 seeds (votes 1 to 3, no screen vote); `driver run`
  labels it and `report` then gives `recheck-passed` or `recheck-failed`. `recheck-pending` is the
  state while it runs.
- **`record-decision`** writes `stakeholderDecision` once. A `proceed` is refused unless the check
  `passed` or `recheck-passed`, or it is `recheck-failed` and `--stakeholder-ref` is given; a
  `stop` is always recordable, and is final. After a decision `report` and `recheck` refuse.
  Commit the report, then delete the pilot's store by hand: the pilot's items and seeds are
  discarded, and `init --phase full` holds the report's RNG seeds so the full campaign never
  reuses one.

### `install`, the agent definitions and the guard

```bash
.venv/bin/python -m decisionmodels.driver install --root ~/dm-campaigns/pilot
```

`agents/` holds the sources, which are dataset provenance (hashed into `promptSha256`): the three
definitions `dm-driver.md` (Sonnet, `Agent(dm-generator, dm-labeler)` only, `maxTurns: 3`),
`dm-generator.md` (`claude-opus-4-8`, effort `medium`, `Read, Write`, `maxTurns: 6`) and `dm-labeler.md`
(Sonnet, effort `high`, `Read, Write`, `maxTurns: 14`), the spawn templates
(`spawn-generator.txt`, `spawn-labeler.txt`, `spawn-driver.txt`; no trailing newline, because a
template is compared byte for byte), `guard_hook.py` and `settings.json`. `install` takes the
campaign lock (it refuses while a `run` holds it) and, before writing anything:

- refuses a `claude` older than 2.1.295 (`runner.MIN_CLAUDE_VERSION`, the first with `onFailure`),
  one with no parsable `claude --version`, or none on `PATH`; `run` makes the same check before
  every launch, after the pause check;
- refuses a definition whose frontmatter lacks a key of its role's set, has another key (Claude Code
  ignores a misspelt one), repeats a key, has a `name` other than its role's agent type, or breaks a pin (plan6 §3.3 "Definition check");
- refuses a `~/.claude/settings.json` that has a forbidden key, with the same check `run` makes
  (`runner.forbidden_settings_refusal`), and records the digest of every other recorded key
  (`runner.user_settings_digests`); it prints the recorded **key names**, never a value, so the
  person installing sees the baseline;
- byte-copies the sources into `exchange/.claude/` and compares every installed file's sha256 with
  its source: a one-byte edit of an installed file, a stray file in `.claude/agents/`, a
  `settings.local.json`, or sources that no longer hash to what the campaign's first install
  recorded refuses (remove the copy, or start a new campaign). Only the installed bytes
  (`agentDefinition`, `spawnTemplate`, `guardHook`) are compared: `requestTemplate` and `rubric` hash
  `points/` files, so each install records them fresh, and a `generator.md` or `rubric.md` edit
  between two installs appends a second `expectations` event (plan5 §3.3: a `label-answer.md` edit
  moves only `requestTemplate.labeler`, a `rubric.md` edit only `rubric`);
- appends an `expectations` event: the guard's exact command, per role its pins read from the
  installed definition (model, effort), its spawn template and an empty `instructions` set, the
  driver template, the user-settings digests and `promptSha256` (with `requestTemplate` and
  `rubric`). It is fresh on every run, which is
  how a refused user-settings edit is accepted: edit, `install` again, and the next `run`
  launches. Only the first event sets the model pins and declared `instructions`, so a re-run
  keeps a `resolve --repin-model/--repin-instructions` decision. The project `settings.json` pins `autoContinueAtUsageLimit: false` and wires the guard
  once (`PreToolUse`, matcher `*`, `"onFailure": "block"`).

`--agents-dir <dir>` installs from another directory of the same files; the smoke's `maxTurns: 2`
batch uses a copy of `agents/` with `dm-labeler.md`'s `maxTurns` edited, in a campaign of its own
(the hashes then name the copy).

**The guard** (`agents/guard_hook.py`, stdlib only) is the only bound on a tool call. It prints one
JSON decision and exits 0, or, when it cannot judge a call (a missing or unknown `agent_type`, a
missing `tool_input`, bad input, any exception, a failed log write), exits 1 with no decision, which
`onFailure: "block"` turns into a block. `dm-driver` may call only `Agent`, with the pending
request's role, no key beyond `subagent_type`, `prompt` and `description`, and the exact spawn
prompt. A subagent may `Read` strictly inside `exchange/req/` and `Write` strictly inside
`exchange/ans/` (paths resolved with `realpath`, so `..` and symlinks do not escape), and call
`SubagentHandback`; nothing else. Each call is logged to `store/hook-log.jsonl` (agent type, tool
and tool-use id, decision; never a path or prompt).

### The label smoke (human-started; run once, 2026-11-01)

```bash
.venv/bin/python -m decisionmodels.driver smoke-fixture --out ~/dm-campaigns/items.jsonl   # 50 items
.venv/bin/python -m decisionmodels.driver init --point guard-judge --phase smoke \
    --root ~/dm-campaigns/smoke --fixture ~/dm-campaigns/items.jsonl
.venv/bin/python -m decisionmodels.driver install --root ~/dm-campaigns/smoke
.venv/bin/python -m decisionmodels.driver run --root ~/dm-campaigns/smoke   # by hand
```

It spends real tokens, so a person starts it through `driver run`. The fixture is 50 guard
`understanding` items whose rendering is exactly the 6,000-character cap in double-escaped pt-BR,
the costliest line for Read's 25,000-token cap. A second batch uses a second smoke campaign
installed with `--agents-dir` from a copy whose `dm-labeler.md` has `maxTurns: 2`.

What the run showed is in `docs/test-reports/live-smoke-report.md` and `docs/HISTORY.md`: the first
spawn was foreground; the 50-item batch ingested with no void and no `Read` error (the labeler read
the pages in parallel, so the per-page token bound is judged without per-Read usage deltas, plan6
§3.3 and §4); the `maxTurns: 2` copy gave the cut-short shape that `tests/test_audit.py`'s
`max_turns_run` now holds; the guard command's `$CLAUDE_PROJECT_DIR` expanded and its
`hook_success.command` equalled `guardCommand`; `SubagentHandback` was not seen. The usage-limit
and subagent-safeguards shapes are still unprobed.

### `run`, `pause` and `resume`

```bash
.venv/bin/python -m decisionmodels.driver run --root ~/dm-campaigns/pilot   # by hand, once per window
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
  after a launch that produced nothing (`launchFailed`: the next `run` tries again), and when
  another `run` or an orphaned child holds `store/driver.lock` (one line: "another `driver run`
  holds store/driver.lock"). Its last line says why (`stopped: done`, `stopped: halted (…)`, …).
- It **refuses** (exit 1, nothing appended to the ledger) a root inside a git work tree or under
  a `CLAUDE.md`/`AGENTS.md`, a campaign with no `install` record, a `claude` below the version floor (2.1.295) or
  with no parsable version, and, checked before every launch, `~/.claude/settings.json` with a forbidden key (`env`, `hooks`, `fallbackModel`,
  `apiKeyHelper`, `disableAllHooks: true`, `autoContinueAtUsageLimit: true`; refused even if
  `install` saw it), or in which any other key outside `runner.UNRECORDED_SETTINGS` (`model`,
  `theme` and the other terminal-only keys) was added, removed or changed since `install`
  recorded its digest; an unknown key counts, and a new `permissions.allow` entry or
  `defaultMode` does not. The refusal names the keys, never a value, and ends "revert it, or run
  `driver install` to accept it". The reason, with a timestamp, goes to `store/refusal`, and
  `status` shows it until a launch succeeds:
  ``REFUSED: 2026-11-02T09:00:00+00:00 launch refused: ~/.claude/settings.json has the
  forbidden `hooks` setting; remove it``.
  Fix the cause; the next `run` launches and deletes the file. The root refusal is the exception:
  `run` checks it before taking the lock and writes nothing, and `status` checks the root
  itself, printing `REFUSED: <reason>` (no timestamp) for as long as the cause is there.
- **`pause`** creates `store/pause` (no lock, no ledger): a running `run` finishes its batch and
  exits, and every `run` exits 0 at once, before the settings check (so pausing to edit
  `~/.claude/settings.json` refuses nothing), until **`resume`** deletes the file. `status` shows
  `PAUSED`.
- **`pause --now`** also sends SIGTERM to the runner named in `store/runner.json`, only if that
  process is alive with the recorded start time. The runner kills its child's group, records
  `interrupted {batch}` (that batch's re-issue is not counted toward the no-attempt cap) and
  exits; the next `run` after `resume` reconciles the batch. Ctrl-C on a hand-started `run` does
  the same.
- **An orphan** (a child whose runner died) holds the lock until it exits, so every `run` exits
  on the lock; `pause --now` does not signal it. Stop it by killing the
  `claude` process by hand; `status`'s "last issued" names the session to look for.

### Running a campaign by hand

No timer ships: a person starts each window with `driver run --root <campaign>` (a stakeholder
decision; the cron scheduler was declined). `pause --root R` lets the batch in flight finish and
`resume --root R` re-opens; `pause --now` also stops the runner and its child. Reading `status`:
`batches: N pending/...` is progress; `PAUSED` means `store/pause`; `REFUSED:` is the last launch
refusal (cleared by the next successful launch); a halt prints its reason and needs `resolve`;
`last issued` names the session to look for after a crash.
