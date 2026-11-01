# Change History — decision-models

> Dated log of actual changes to the `decision-models` component. Most recent first.

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
