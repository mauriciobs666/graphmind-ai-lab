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
reconciliation, halts), `runner.py` (`driver run`, `pause [--now]`, `resume`), `install.py`
(`driver install`), `smoke.py` (the label smoke's fixture) and `driver.py` (CLI: `init`, `install`,
`resolve`, `run`, `pause`, `resume`, `status`, `smoke-fixture`). `agents/` holds the three agent
definitions, spawn templates, `guard_hook.py` and `settings.json`. Later plan4 §4.1 and plan6 §4
units are unbuilt, and U5.7's live smoke has not run (`docs/HISTORY.md`).

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
  `onFailure: "block"` blocks), never imports `decisionmodels`. Its command uses an unexpanded
  `$CLAUDE_PROJECT_DIR`, unverified in a live transcript until the smoke.
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
  (`guardRejected`, `hookMissing`). Its bounds code never imports `guard_hook.py`. `maxTurns` is
  provisional until U5.7's smoke. Fixtures (`tests/fixtures/transcripts/`) are hand-built in the
  probe report's recorded key sets, never copies of real transcript content.
- **Guard-only renderers until U5.2b-render.** `exchange.RENDERERS` lacks the join-in and
  message-type renderers; strict xfails in `tests/test_exchange.py` then XPASS: delete them.
- **Zero runtime dependencies** (`pyproject.toml`, pinned by `tests/test_package.py`).

## Conventions

- Run the suite **from `decision-models/`**: `.venv/bin/pytest -q`, `.venv/bin/ruff check .`.
  `pythonpath = ["."]` lets it run without the editable install.
- No network and no LM Studio in the default suite; tests write under `tmp_path`. Runner tests
  launch only the fake `claude` `tests/test_runner.py` writes, under a fake `HOME`: never the real
  CLI or `~/.claude`.
- U5.x units serialize on `docs/HISTORY.md` (plan4 §4.3): one entry per unit, most recent first.
