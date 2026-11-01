# Change History — decision-models

> Dated log of actual changes to the `decision-models` component. Most recent first.

## 2026-11-01 — typed-decision-models U5.3 — points, rubrics, seeds

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.3, with plan5
(`typed-decision-models5.md`) §3.4–§3.5, under note `docs/plans/typed-decision-models-ml.md`
v1.11 §Q1.1–§Q1.5; reviews `docs/reviews/typed-decision-models4-impl.md` Passes 3–4 and
`docs/reviews/typed-decision-models-ml.md` Passes 1–2.

- `points/<role>/` for `guard-judge`, `message-type-classifier`, `join-in-judge`: `point.json`
  (answer kind, key and `expected` per v1.3 §3.2 and FR-10; input-field schema with v1.3 §3.5 /
  plan2 §3.1 field names; seed bindings; seed design and weights; note §Q1.3 cells and sizes;
  screen token lists), `rubric.md` (each answer, then boundary cases) and `generator.md` (no
  commentary, no `*Id` field). `points/common/` holds the two answer-file sections.
- Note v1.10–v1.11 design: guard `missingForm` drawn given (answer, difficulty) on
  `understanding` seeds, with per-item obedience as an `emptinessAxis` binding; join-in `cue`
  drawn given the answer, cue_05 narrowed; message-type rulings for off-topic questions,
  cancelling and returns; join-in small-talk ruling; the assistant speaks as `Assistant` /
  `<business> Assistant`; the guard generator adapts the situation, never the condition or the
  business. Generator text per ML Pass 2: the `ambiguous` rule ends "`missing` is `[]`"; a guard
  `understanding` near-miss follows `missingForm`; a message-type follow-up may stand alone.
- `decisionmodels/seeds.py`: `draw_seeds` — hierarchical stratified axes, balanced axes, and
  conditional axes over weighted multi-axis cases, every allocation by unbiased systematic
  rounding with an integer start (docstring); non-word ids; the record carries `pythonVersion`
  (reproduction holds on the same minor version).
- `decisionmodels/points.py`: `load_point`, the composed line-2 texts, `input_field_errors`,
  `seed_binding_errors` returning `BindingError(binding, message)` for U5.8's classification. The
  `missingForm` binding is silent when `understanding` is not a non-empty mapping, so a wrong-kind
  item reports only its `evidenceKind` errors (P4-F1).
- `tests/test_seeds.py`, `tests/test_points.py`: `allocate`'s expectation exact over evenly
  spaced starts, and constant within each integer start's interval; draw means over 1,500 RNG
  seeds at sizes 1–7 within 4 standard errors; restricted round-trip through the record;
  collision and conditional-overlap guards; the wrong-kind item; a text pin tying each form's
  "`missing` is `[]`" to its binding; strict xfails for the two roles `hashing.py` lacks until
  U5.2b (P3-F6); side-check gate fails rather than skips when the pack moves (P3-F5).
  `AGENTS.md`, `README.md`.
- **Digest pins moved**, deliberately: all three re-pinned (guard `341d8815…`, message type
  `299beb51…`, join-in `26109599…`). Message type's `point.json` is unchanged, but systematic
  rounding changes every allocation, so its pin moved too. The integer start left all three
  unchanged: it gave identical shares on 100,000 random (weights, total, RNG seed) cases.

**Verified:** `pytest -q` from `decision-models/` 364 passed, 2 xfailed, and 1 failed:
`test_the_vector_copy_is_byte_identical_to_model_benchs[input-hash.jsonl]`, because
model-bench's canonical `input-hash.jsonl` gained six rows in its working tree (U4 in flight);
this unit changed neither the copy nor the test. Deselected by that node id: 364 passed,
2 xfailed. `ruff check .` clean. Eleven mutants in a private copy, each restored `cmp`-equal and
sha256-checked: largest remainder 84 failed; start ×0.97 21, ×0.95 35, ×0.999 9, ×1.001 86;
`missingForm` guard removed 6; overlap check removed 1; the `ambiguous` rule's "`missing` is
`[]`" removed 1; undefined-form error removed 1; the guard over-silencing 7; the pre-integer
float start restored 0 (equivalent, as intended).

## 2026-10-31 — typed-decision-models U5.2 — guard contract copies

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.2, under v1.3 §3.2, §3.2.2 and §3.5.

- `decisionmodels/render.py`: `render_guard_input(condition, understanding, turns)` — the guard
  judge user message (CONDITION, then CURRENT STATE iff `understanding` is non-empty, then RECENT
  TURNS; 6,000-char cap dropping oldest turns first, base hard-truncated when no turn fits);
  `normalize_turns` (filter-then-last-6, 400-char text, `displayName`/`authorId`/`member`
  speaker); `render_guard_item(item)` composing the two (v1.3 §3.2's `normalized_turns`).
- `decisionmodels/hashing.py`: `INPUT_FIELDS_BY_ROLE` (`guard-judge` only; the other roles come
  with U5.2b), `canonical_input_json`, `input_hash` (v1.3 §3.5's formula; an unknown role is a
  `ValueError`).
- `tests/fixtures/decision-contract/render-guard.jsonl` (14 rows) and `input-hash.jsonl` (4 rows):
  byte-copies of model-bench's canonical files, `cmp`-equal.
- `tests/test_decision_contract.py`: every vector row against the implementation; the guarded
  byte-equality test per copy (skips only when `model-bench/` is absent); a row-count pin per file
  counted from raw non-blank lines; hand-derived tests for turn normalization and
  `render_guard_item` (including a ten-row turn-order and cap case, Pass 2 P2-F1), which no
  cross-component vector covers.
- `AGENTS.md` (current state; the byte-copy invariant) and `README.md` (status).

**Verified:** `pytest -q` from `decision-models/` 54 passed (26 → 54); `ruff check .` clean.
Twenty-one mutants each redden at least one test, re-run from a private driver after a failed
restore was recovered (`render.py` md5 `2ce93c54…`); the restore failed because a parallel agent and
this unit both wrote `scratchpad/mutate.py`. Independent re-sample:
`docs/reviews/typed-decision-models4-impl.md` Pass 2.

## 2026-10-31 — typed-decision-models U5.1 — component scaffold and the FR-8 isolation test

**What:** `docs/plans/typed-decision-models4.md` §4.1's U5.1 (v1.3 §4 U5's skeleton bullet and its
FR-8 test; v1.3's `labeler.py` is void and not scaffolded).

- `setup.sh` (model-bench's, renamed), `pyproject.toml` (no runtime dependencies; dev extra
  `pytest`, `ruff` at model-bench's pins; `pythonpath = ["."]`), `.gitignore`, `README.md`,
  `AGENTS.md`, and the empty `decisionmodels/` package with `__version__`.
- `tests/test_fr8_isolation.py`: an AST import allowlist (stdlib + `decisionmodels`, relative
  imports; `__import__`/`import_module` count, a non-literal module name is refused) and a scan of
  non-docstring string literals for `falkor-chat`/`falkorchat`/`falkor_chat`, `falkordb`,
  `redis` and `guard-judge-understanding`, plus the integer 6379, over `decisionmodels/` only
  (plan2 §3.5). Fourteen planted-red cases and a negative control.
- `tests/test_package.py`: `__version__` equals `pyproject.toml`'s; `dependencies` is empty.
- Root `AGENTS.md`: one Structure bullet and one Component-docs row.

**Verified:** `pytest -q` from `decision-models/` 20 passed; `ruff check .` clean. Planting
`import redis` (and, separately, `import falkorchat`, `from modelbench import packs`, a
`falkor-chat` path, the side-check items path and `6379`) in `decisionmodels/__init__.py`
reddened `test_decisionmodels_reaches_only_stdlib_and_itself`; the file was restored `cmp`-equal
each time.
- Review Pass 1 fixes (`docs/reviews/typed-decision-models4-impl.md` F1–F3, F5, F6): a string-literal planted case per fragment (dropping any one fragment reddens the suite), word-bounded `redis` with "predisposition"/"redistribuir" in the negative control, bytes literals and `from importlib import import_module as …` aliases scanned, the docstring's list of what the tripwire cannot see, `AGENTS.md`'s side-check-reader wording, and the root bullet naming the requirements doc; 26 passed.
