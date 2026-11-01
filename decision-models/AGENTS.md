# decision-models — agent working context

## Project in one sentence

Builds synthetic, labelled datasets and fitted candidates for typed decision models, which
`model-bench/` then measures — see `README.md` for the human-facing view and
`docs/plans/typed-decision-models4.md` (repo root; extends plan3 < plan2 < v1.3, later wins) for
the design.

## Current state

Scaffold only: `decisionmodels/__init__.py`, the FR-8 isolation test and the version pin. The
remaining units (U5.2–U5.10, U5.2b) and their order are plan4 §4.1/§4.3; the layout they fill is
plan4 §3.1. What has landed is in `docs/HISTORY.md`.

## Load-bearing invariants

- **FR-8: `decisionmodels/` imports stdlib + itself only, and names no real-data reach.**
  `tests/test_fr8_isolation.py` holds the allowlist and the forbidden literals (falkor-chat
  paths/packages, FalkorDB/Redis, the `guard-judge-understanding` side-check items, port 6379).
  Never widen either to make a unit green; a new reach is a plan change. It is a static tripwire;
  the test's docstring lists what it cannot see (indirect imports, run-time-assembled names). The
  scan covers `decisionmodels/` only — `scripts/import_side_check_shingles.py` (plan2 §3.5) is the
  one sanctioned reader of the side-check items (`model-bench/packs/guard-judge-understanding/`),
  and it writes hashes only.
- **No hosted API client.** v1.3's `labeler.py` is void (plan4 §1); generation and labelling go
  through the subagent file exchange (plan4 §3).
- **Agent definitions live in `agents/`, not `claude/` or `.claude/agents/`** (plan4 §3.1): they
  are dataset provenance, and `claude/`'s rules do not apply to them.
- **Zero runtime dependencies** (`pyproject.toml`, pinned by `tests/test_package.py`).

## Conventions

- Run the suite **from `decision-models/`**: `.venv/bin/pytest -q`, `.venv/bin/ruff check .`.
  `pythonpath = ["."]` lets it run without the editable install.
- No network and no LM Studio in the default suite; tests write under `tmp_path`.
- U5.x units serialize on `docs/HISTORY.md` (plan4 §4.3): one entry per unit, most recent first.
