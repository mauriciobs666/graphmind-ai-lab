# decision-models — agent working context

## Project in one sentence

Builds synthetic, labelled datasets and fitted candidates for typed decision models, which
`model-bench/` then measures — see `README.md` for the human-facing view and
`docs/plans/typed-decision-models4.md` (repo root; extends plan3 < plan2 < v1.3, later wins) for
the design.

## Current state

`decisionmodels/` holds `render.py` (all three roles' renderers; `exchange.RENDERERS` maps each), `hashing.py` (all three roles), `points.py` (loads
`points/<role>/`), `seeds.py` (seed draws), `exchange.py` (request writer, closed schemas),
`ingest.py` (answer checks, idempotence), `audit.py` (transcript audit), `campaign.py` (ledger,
reconciliation, halts, generation runs, votes, tranche, top-up), `diagnostics.py`
(`diagnostics.json`), `screen.py` (the generated-text screen and form
obedience, pure), `labels.py` (majority of three, `labelVotes`, pure), `pilot.py` (the guard pilot's CLI and pure
screen/cue-check, below), `runner.py` (`driver run`, `pause [--now]`, `resume`), `install.py`
(`driver install`), `smoke.py` (the label smoke's fixture) and `driver.py` (CLI: `init`, `install`, `start`,
`resolve`, `run`, `pause`, `resume`, `status`, `smoke-fixture`, `finalize`), plus `shingles.py` and
`split.py` (pure: the 8-token shingle rule and cells-then-splits, plan4 §3.6;
`scripts/import_side_check_shingles.py` writes `data/guard-judge/side-check-shingles.txt`, hashes
only) and `finalize.py` (the dataset writer). `agents/` holds the three agent definitions, spawn
templates, `guard_hook.py` and `settings.json`. Fitting (U6) and what follows it are unbuilt
(`docs/HISTORY.md`). U5.7's live smoke ran; its facts are in
`docs/test-reports/live-smoke-report.md`.

## Load-bearing invariants

- **FR-8: `decisionmodels/` imports stdlib + itself only, and names no real-data reach.**
  `tests/test_fr8_isolation.py` holds the allowlist and the forbidden literals (falkor-chat
  paths/packages, FalkorDB/Redis, the `guard-judge-understanding` side-check items, port 6379).
  Never widen either to make a unit green; a new reach is a plan change. It is a static tripwire
  (its docstring lists what it cannot see). The
  scan covers `decisionmodels/`, plus `scripts/*.py` and `agents/*.py` under a per-file exemption
  map (`EXEMPT_FRAGMENTS`): `scripts/import_side_check_shingles.py` (plan2 §3.5) is the one
  sanctioned reader of the side-check items (`model-bench/packs/guard-judge-understanding/`), and
  it writes hashes only. A new exemption is a plan change.
- **Contract copies are byte-copies, never regenerated or edited here.**
  `tests/fixtures/decision-contract/*.jsonl` are copies of model-bench's canonical vectors (v1.3
  §3.2.2; six files, the renderer and `normalize-turns.jsonl` ones included); `tests/test_decision_contract.py` asserts each byte-identical (skips only when
  `model-bench/` is absent). A red equality test means re-copy from
  `model-bench/tests/fixtures/decision-contract/` in the unit adding the matching code. Reading
  model-bench from `tests/` is allowed (FR-8 scope).
- **No hosted API client.** v1.3's `labeler.py` is void (plan4 §1); generation and labelling go
  through the subagent file exchange (plan4 §3).
- **Agent definitions live in `agents/`, not `claude/` or `.claude/agents/`** (plan4 §3.1): they
  are dataset provenance, and `claude/`'s rules do not apply to them. Every file there is hashed
  into `promptSha256` and byte-copied by `install`, so an edit is a dataset-version change, and a
  spawn template has no trailing newline (the audit compares it byte for byte). `guard_hook.py` is
  the only bound on a call: stdlib only, exit 1 with no decision when it cannot judge (the wiring's
  `onFailure: "block"` blocks), never imports `decisionmodels`. Its command is shell-form with
  `$CLAUDE_PROJECT_DIR`, which the live smoke showed expands; the record keeps it unexpanded.
- **`points/<role>/` is data, reviewed as data.** `point.json` (answer space, input-field schema,
  seed bindings, seed design and its weights, cells and sizes, screen tokens), `rubric.md`,
  `generator.md`; `points/common/` holds the answer-file section appended to each
  (`Point.rubric`, `Point.generator_instructions`). A method change in the note is a data edit
  here plus a re-pin, not code. Rubrics never name seed axes, intent or difficulty; no text there
  carries a seed value, and generator templates name no `*Id` field (`tests/test_points.py`).
  `points/` is outside the FR-8 AST scan (it holds no code) but its text is scanned for the same
  forbidden fragments and for side-check shingles (note §Q1.4).
- **Seeds reproduce from the recorded RNG seed on the same Python minor version**
  (`draw_seeds(role, count, rng_seed, design_stratum=…)`; the record carries `pythonVersion`;
  `seeds.py`'s docstring says why). The recorded seeds are the source of truth. Allocation is
  unbiased systematic rounding — never swap in largest remainder (note §Q1.1). Ids are in the
  non-word form `seeds.ID_PATTERN`. `tests/test_seeds.py` pins a digest per point; a design,
  algorithm or interpreter change re-pins it deliberately.
- **`install` hashes two things differently** (plan5 §3.3, plan6 §3.8). `agentDefinition`,
  `spawnTemplate`, `guardHook` hash installed bytes and must match the first install's;
  `requestTemplate` and `rubric` (`campaign.rubric_hash`) hash `points/` files and are recorded fresh
  on every install, so a `generator.md`/`rubric.md` edit appends a second `expectations` event.
- **Splits and `finalize`** (`split.make_splits`, `finalize.py`; plan4 §3.6) run in a fixed order:
  side-check drop, random subsample per cell (keyed by the **majority** label), split by seed
  (eval first), eval-vs-calibration/train shingle drop last; `train` is empty. **A dropped eval
  slot is refilled from the same cell's unused candidates**, tested against the same
  calibration/train shingles (calibration and train never change; a run with no drop is unchanged; each deficit cell's shuffle advances the shared rng, so the walk is order-dependent across cells);
  `surplus` is what is left and `dropped.shingleSplit` counts assigned drops only. **Floors gate
  every finalize**: a cell under `evalFloorPerCell` (eval or calibration) refuses, writes no
  dataset, and `finalize()` (not the pure `build_dataset`) appends a `floor-short` event (`n` and
  `need` per cell, the RNG seed, held; `State.floor_short` folds it plainly, no halt). The other
  ledger write a refusal makes is the whole-point tranche recheck's `recheck-halt` (second failure
  only `resolve --stakeholder-ref` alone clears; a re-pin is refused). `finalize --rng-seed <n>`
  fixes the seed. `items.jsonl` goes through `exchange.dump_line` (model-bench's importer uses
  `splitlines()`); `--short` needs a recorded `--decline-topup` (the floors apply anyway);
  `generation.json` holds one model per role and the last-issued run's hashes; the one-hash rule
  reads only post-restart runs (`State.template_resolutions` against `Attempt.epoch`).
  `tests/test_finalize.py` checks the output against the contract vector (plan6 §3.7). Shingles
  cover evidence text only; a hash cannot be turned back into text.
- **Requests are closed and id-blind** (plan4 §3.2, plan5 §3.4). `exchange.request_errors` is the
  one schema check; the writer runs it before any request leaves it. No `*Id` key
  reaches a generation request, whatever `points/` declares.
- **Only `campaign.py` writes the ledger**; ingest and audit never record. The audit's classes
  beat ingest's (`audit.combine`), and only a `CLEAN`/`COUNT` audit ingests and records
  `answerSha256`. The `outcome` event is the commit point, appended before parts move to
  `store/`; a re-ingest never calls `audit_run`/`combine` and appends nothing; a re-issued name
  is retired, so a late answer raises `AnswerRefused` (`campaign.py`'s docstring).
- **`next`/`ingest` are library entry points for `driver run`**, never subcommands (plan6
  §3.2). `next` checks `store/pause` before reading anything. `run` uses **one** `Campaign` per
  process (the usage-stop memory is the instance's), holds `store/driver.lock` for its life and
  passes it to the child, and writes `store/runs/<sid>.exit` atomically before `ingest`
  (`runner.py`'s docstring). The child's environment is an allowlist; widening it is a plan change.
- **The audit locates a run only through the driver's `Agent` `tool_use` and `.meta.json`
  `toolUseId`**, never through what it checks, and fails closed (plan6 §3.4): only F2's
  foreground shape completes, and a tool use the guard did not allow with its own entry halts
  (`guardRejected`, `hookMissing`). Its bounds code never imports `guard_hook.py`. `maxTurns` is the
  smoke-recorded cut-short shape (`tests/test_audit.py`'s `max_turns_run`). Fixtures (`tests/fixtures/transcripts/`) are hand-built in the
  probe report's recorded key sets, never copies of real transcript content.
- **Join-in renders only the last 6 prior turns** (`normalize_turns`) though its `maxItems` is 8, so
  `points/join-in-judge/generator.md` tells the generator to keep the agent's job and what the new
  message depends on inside the last six; `tests/test_render_layout.py` pins the rule (not a phrase),
  the item wrappers by value, the unstripped `agentName`, and that each point's rendering matches
  its rubric's "What an item looks like" block (the first indented block only). `data-scientist`
  confirmed the rule (plan5 §3.4).
- **Generation is ledger-folded, and commits at the `outcome`** (plan6 §3.8). A `full` campaign's round 1 is drawn only by `driver start` (`run` never draws it; pilot: `pilot start`). `begin_round` records
  `round-start` (one `genRound` may have several; only `_draw_topup` (`queue_topup`'s and `_resume_topup`'s) passes `extend=True`, for a
  new named stratum of a started round, refused once a run of it has issued); `Campaign.ingest` appends a run's `generated` and
  `screen-hit`/`binding-mismatch`/`form-mismatch` events *before* its `outcome`, and `fold` applies
  them only at that ingested `outcome`. An item's `genRound` is its **seed's** round (a regeneration
  run pools rounds), its `genRunId` its batch's (`Batch.gen_run_id`, never copied by a cut). A hit
  event's `reasons` lists every reason the item shows; the first counts toward the seed's 3 hits.
  The `template` halts count first-pass hits under each reason: per `genRound` after each run
  (over 4% of its drawn seeds, two hits at least), and per generator-template hash over a pool of
  the rounds under 25 seeds (a round closes when its runs are all ingested or dropped; each
  ingested run's seeds and hits go to its own issue-time hash's pool, which is tested at 300 seeds
  and over 5%); one halt record lists the per-round part (`round`) and every pool that crossed
  (`pools`), and each pair fires once even after `resolve`, which clears the halt, not the pair. Runs
  are cut balanced (`balanced_runs`). An attempt keeps the `expectations`, pins and instructions in
  force when it was issued; the audit reads those, never the latest. A generator re-pin
  (`generation-restart`) resets the folded generation view and **retires** every batch (never
  issued, ignored by `next`); the ledger, held RNG seeds and `seedId`s, pins and expectations
  stay. A batch queued by plain `queue_batch` has no `generation` and is not checked.
- **Votes are ledger-folded too** (plan6 §3.8; `labels.py`). When nothing is pending, `next` calls
  `queue_votes`: a round whose seeds are all accepted or dropped starts vote *v* with a
  `vote-start` event (a fresh, held RNG seed; the items reshuffled on it) and queues label batches
  of at most `labelBatchCap` (`prepare` cuts `P_MAX` overflow into the same vote). Vote *v+1*
  starts when vote *v*'s batches are all ingested or dropped. A `votes` event is applied at the
  run's ingested `outcome`; `State.label_votes` / `vote_ties` give `labelVotes` and `voteTie`. A
  dropped label batch (any reason) takes its items out of the later votes. A vote is incomplete
  until its live batches hold every `vote-start` item (a crash between its `queued` events);
  `queue_votes` queues the missing ones into it. A `labelling-restart` clears the votes and
  retires **every** label batch (the smoke fixture's too), keeping the items; a generator restart
  clears both. A label batch with no `vote` records nothing. A main pilot block's fourth vote
(`labels.SCREEN_VOTE`, batches of `SCREEN_BATCH_CAP`) is a vote like the others but **never one of
the three** that `three_votes`, `label_votes` and `vote_ties` read.
- **Top-up and the tranche are ledger-folded too** (plan4 §3.6; `campaign.py`). In a `full`
  campaign, when nothing is pending, `next` calls `queue_topup`: once every round is labelled and a
  cell (design stratum × label) is below `cell_target` (eval + calibration), `topup_counts` (pure)
  gives the seeds per short stratum, `max ⌈shortfall / P̂(ℓ | s)⌉`; it names no label or intent, the
  draw is the design restricted to the stratum, and `P̂ = 0` raises `UnboundedCell(label, stratum)`,
  which `queue_topup` records as the `unbounded-halt` (`unboundedCell`, with the Wilson bound; no
  `store/refusal`). **A `floor-short` record raises its cells' targets**: `State.floor_targets()`
  gives cell → `n + need` from the newest record, `topup_counts(extra=…)` uses `max(cell_target,
  extra)` (never the sum; `need` 0 changes nothing) and `finalize._short_cells` calls the same
  function; a null `need` (every eval item dropped) raises `UnboundedCell` for that cell, the same
  halt, so the stakeholder routes (probe, decline, re-pin) are the way out. `project_topup` (pure; seeds over round-1 `seeds`, live labelling rate, four
  token classes) halts with `spend` when twice the projection exceeds round 1's total *including*
  `labelling-restart`/`tranche-restart` retirements (`Batch.retired_by`; a generation restart's are
  no round's); the reason prints both totals. A plain `resolve` lets a spend halt through and is
  refused for `unboundedCell`; `--decline-topup` (`topup-declined`, either halt) ends top-ups and
  `--reopen-topup` (`topup-reopened`, no halt open) undoes it; `--probe-topup` appends
  `probe-requested`, which `fold` keeps until a `topup-plan`/decline/generator restart, and
  `queue_topup` plans it (25 seeds of that stratum) before `topup_counts`. Every stratum is drawn
  (`topup-plan` first, `_resume_topup` after a crash) before any run issues, then cut as one pool,
  interleaved.
  Message type's round 1 votes a 250-item **tranche** first (`tranche-start`; its `vote-start`s and
  batches carry `tranche`): `tranche_failed` (`100·drops > 3·size`, `voteTie` drops) halts with
  `rubric`, and `resolve` then needs a changed `rubric_hash` and relabels it (`tranche-restart`);
  the rest votes only once it passes. A generator restart resets all of it. `diagnostics.py` is
  pure over a folded `State`; its `HANDOFF_EXAMPLES` are verbatim from model-bench's hand-off prompts.
- **The pilot is ledger-folded too** (plan4 §3.8, plan5 §3.6; `pilot.py`). A `round-start` carries
  `pilotBlock` (`main`: votes 1-3 and the screen vote; `recheck`: votes 1-3), which decides
  `State.round_votes`; only `pilot.py start`/`recheck` pass it, and the driver (`queue_pilot_block`,
  from `next`) draws each added main block while `pilot.screen(...).more_needed` (`n < 80`), at most
  `PILOT_ADDED_BLOCK_LIMIT` (3) per halt epoch: then it records the batchless `pilot-block-halt`
  (`record_pilot_block_halt`; reason `pilotBlocks`, its `mainRounds` starts the next epoch, which
  `resolve` opens; a generator restart resets it, and the cue-halt bookkeeping, with the rounds).
  Both `record_*_halt` refuse while any halt is open. `n`, `Δ` and the cap count main blocks only. `pilot.py report` is the one evaluator: it derives
  `missingCueCheck.state` from the ledger each run, appends the `cue-check-halt` event once (through
  `Campaign.record_cue_check_halt`, never `_append`), and refuses a partly voted block, and a
  report that holds a `stakeholderDecision`; its `dropped` leaves out dropped screen batches
  (`droppedScreenVote`). `record-decision` reuses `campaign.spend_gate_refusal`,
  the predicate `init --phase full` applies: never copy it. `pilot.py` and `campaign.py` import each
  other (call-time use only); the cue check's p-value is exact (`Fraction`).
- **Zero runtime dependencies** (`pyproject.toml`, pinned by `tests/test_package.py`).

## Conventions

- Run the suite **from `decision-models/`**: `.venv/bin/pytest -q`, `.venv/bin/ruff check .`.
  `pythonpath = ["."]` lets it run without the editable install.
- No network and no LM Studio in the default suite; tests write under `tmp_path`. Runner tests
  launch only the fake `claude` `tests/test_runner.py` writes, under a fake `HOME`: never the real
  CLI or `~/.claude`.
- U5.x units serialize on `docs/HISTORY.md` (plan4 §4.3): one entry per unit, most recent first.
