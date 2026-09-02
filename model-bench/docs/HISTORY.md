# Change History — model-bench

> Dated log of actual changes to the `model-bench` component. Most recent first.

## 2026-09-02 — S0: component skeleton

**What:** Created the `model-bench/` component per stage S0 of
`docs/plans/small-model-benchmarking.md` §4 — packaging, scripts, docs skeleton and an empty
package/suite. No harness code: S0's done-condition is deliberately an empty test suite, so that
S1–S8 land against a tree that already builds and lints.

- `pyproject.toml` — `requires-python = ">=3.12"`, **no runtime dependencies** (plan §3.2, a hard
  design constraint), dev extras `pytest>=9.1,<10` + `ruff>=0.14,<0.15`, ruff `select = ["E","F","W","I"]`
  / `line-length = 100` (mcp-monitor's shape), pytest `testpaths = ["tests"]` plus falkor-chat's
  live-test convention verbatim: `addopts = '-ra -m "not live"'` and a `live` marker.
- `setup.sh` — adapted from `mcp-monitor/setup.sh`: idempotent, `--recreate`, resolves paths from the
  script's own location, ends with an import smoke test.
- `run.sh` — the mcp-monitor shape (venv check, then `exec .venv/bin/python -m modelbench "$@"`) with
  an **S0 guard**: `modelbench/__main__.py` does not exist until S1, so the script reports that in
  words and exits 1 rather than `exec`-ing into a `No module named` traceback. S1 deletes the guard.
- `.gitignore` — `.venv/`, `host.json` (the operator-attested fingerprint fields, plan §3.4),
  `results/transcripts/` (raw model output: large, and not needed for any comparison, plan §3.5).
- `README.md` — what the tool is, and the three non-features stated up front: no CI/scheduler, no
  pass/fail gate, no leaderboard or cross-role aggregate.
- `AGENTS.md` — working context: current state, the hard rules (zero runtime deps, FR-23 standalone,
  no cross-role aggregate), the `live` marker, the attested fingerprint fields, and the note that
  an empty suite exits 5.
- `docs/{BACKLOG.md,HISTORY.md}` plus empty `requirements/ plans/ reviews/ test-plans/ test-reports/`
  held by `.gitkeep` files. `BACKLOG.md` is seeded with the two items plan §7 carries forward.
- `modelbench/__init__.py` (`__version__`) and `tests/test_package.py` — one install smoke test,
  asserting `modelbench.__version__` equals the installed distribution's metadata version. The plan
  called for an empty suite at S0, but pytest exits 5 (`EXIT_NOTESTSCOLLECTED`) when nothing is
  collected, so "runs and passes with zero tests collected" cannot return 0 (plan gate finding m1).
  Resolved with this one real test rather than by configuring the exit code away: a permanent
  "no tests ran is fine" setting would still be in place at S5 and would hide a collection
  breakage. The assertion is not filler — that version string is what stamps `benchVersion` into
  every run record (plan §3.4), so a skew between `pyproject.toml` and `__init__.py` fails here.
- Root `AGENTS.md` — a `model-bench/` bullet in **Structure** and a row in **Component docs**. The
  feature's requirements and plan stay at the repo root, where they were written (plan §4 S0).

**Verification:** `model-bench/setup.sh` → venv created with Python 3.12.3, `model-bench[dev]`
installed (pytest 9.1.1, ruff 0.14.14), smoke import printed `model-bench 0.1.0`; re-run to confirm
idempotence. `.venv/bin/python -m pytest -q` from `model-bench/` → `1 passed in 0.01s`, exit 0
(0 failed / 0 skipped / 0 deselected). `.venv/bin/ruff check .` → `All checks passed!`.
`./run.sh --help` → the S0 guard's message, exit 1. Note that the test command must be run with
`model-bench/` as the working directory: the repo has no root pytest configuration, so from the repo
root pytest ignores this component's `testpaths` and walks the whole monorepo (measured: 9 collected,
8 collection errors, exit 2).
