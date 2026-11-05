# decision-models — agent working context

## Project in one sentence

Builds synthetic, labelled datasets and fitted candidates for typed decision models, which
`model-bench/` then measures — see `README.md` for the human-facing view and
`docs/plans/typed-decision-models4.md` (repo root; extends plan3 < plan2 < v1.3, later wins) for
the design.

## Current state

`decisionmodels/` holds `render.py` (guard only), `hashing.py` (all three roles), `points.py` (loads
`points/<role>/`), `seeds.py` (seed draws), `exchange.py` (request writer, closed schemas),
`ingest.py` (answer checks, idempotence), `audit.py` (transcript audit), `campaign.py` (ledger,
reconciliation, halts, generation runs, votes), `screen.py` (the generated-text screen and form
obedience, pure), `labels.py` (majority of three, `labelVotes`, pure), `runner.py` (`driver run`, `pause [--now]`, `resume`), `install.py`
(`driver install`), `smoke.py` (the label smoke's fixture) and `driver.py` (CLI: `init`, `install`,
`resolve`, `run`, `pause`, `resume`, `status`, `smoke-fixture`). `agents/` holds the three agent
definitions, spawn templates, `guard_hook.py` and `settings.json`. The pilot and cap/screen vote,
the message-type tranche, top-up and `finalize` (plan4 §4.1 U5.8c onward) are unbuilt (`docs/HISTORY.md`). U5.7's live smoke ran; its facts are in
`docs/test-reports/live-smoke-report.md`.

## Load-bearing invariants

- **FR-8: `decisionmodels/` imports stdlib + itself only, and names no real-data reach.**
  `tests/test_fr8_isolation.py` holds the allowlist and the forbidden literals (falkor-chat
  paths/packages, FalkorDB/Redis, the `guard-judge-understanding` side-check items, port 6379).
  Never widen either to make a unit green; a new reach is a plan change. It is a static tripwire
  (its docstring lists what it cannot see). The
  scan covers `decisionmodels/` only — `scripts/import_side_check_shingles.py` (plan2 §3.5) is the
  one sanctioned reader of the side-check items (`model-bench/packs/guard-judge-understanding/`),
  and it writes hashes only.
- **Contract copies are byte-copies, never regenerated or edited here.**
  `tests/fixtures/decision-contract/*.jsonl` are copies of model-bench's canonical vectors (v1.3
  §3.2.2); `tests/test_decision_contract.py` asserts each byte-identical (skips only when
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
- **Guard-only renderers until U5.2b-render.** `exchange.RENDERERS` lacks the join-in and
  message-type renderers; strict xfails in `tests/test_exchange.py` then XPASS: delete them.
- **Generation is ledger-folded, and commits at the `outcome`** (plan6 §3.8). `begin_round` records
  `round-start` (one `genRound` may have several; only U5.9's planner passes `extend=True`, for a
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
  clears both. A label batch with no `vote` records nothing.
- **Zero runtime dependencies** (`pyproject.toml`, pinned by `tests/test_package.py`).

## Conventions

- Run the suite **from `decision-models/`**: `.venv/bin/pytest -q`, `.venv/bin/ruff check .`.
  `pythonpath = ["."]` lets it run without the editable install.
- No network and no LM Studio in the default suite; tests write under `tmp_path`. Runner tests
  launch only the fake `claude` `tests/test_runner.py` writes, under a fake `HOME`: never the real
  CLI or `~/.claude`.
- U5.x units serialize on `docs/HISTORY.md` (plan4 §4.3): one entry per unit, most recent first.
