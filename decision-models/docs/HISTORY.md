# Change History — decision-models

> Dated log of actual changes to the `decision-models` component. Most recent first.

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
