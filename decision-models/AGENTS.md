# decision-models — agent working context

## Project in one sentence

Builds synthetic, labelled datasets and fitted candidates for typed decision models, which
`model-bench/` then measures — see `README.md` for the human-facing view and
`docs/plans/typed-decision-models4.md` (repo root; extends plan3 < plan2 < v1.3, later wins) for
the design.

## Current state

`decisionmodels/` holds `render.py` and `hashing.py` (guard only), `points.py` (loads
`points/<role>/`) and `seeds.py` (seed draws); `points/` holds the three points' data. The rest of
plan4 §4.1's units are not built yet (`docs/HISTORY.md`).

## Load-bearing invariants

- **FR-8: `decisionmodels/` imports stdlib + itself only, and names no real-data reach.**
  `tests/test_fr8_isolation.py` holds the allowlist and the forbidden literals (falkor-chat
  paths/packages, FalkorDB/Redis, the `guard-judge-understanding` side-check items, port 6379).
  Never widen either to make a unit green; a new reach is a plan change. It is a static tripwire;
  the test's docstring lists what it cannot see (indirect imports, run-time-assembled names). The
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
  are dataset provenance, and `claude/`'s rules do not apply to them.
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
- **`hashing.py` lacks the join-in and message-type rows until U5.2b.** `tests/test_points.py`
  holds strict xfails for them (plan5 §3.4); their XPASS means deleting those xfails and
  `PLANNED_INPUT_FIELDS`.
- **Zero runtime dependencies** (`pyproject.toml`, pinned by `tests/test_package.py`).

## Conventions

- Run the suite **from `decision-models/`**: `.venv/bin/pytest -q`, `.venv/bin/ruff check .`.
  `pythonpath = ["."]` lets it run without the editable install.
- No network and no LM Studio in the default suite; tests write under `tmp_path`.
- U5.x units serialize on `docs/HISTORY.md` (plan4 §4.3): one entry per unit, most recent first.
