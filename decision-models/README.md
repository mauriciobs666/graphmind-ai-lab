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
  hook, no scheduler. It runs because a person typed a command.
- **No hosted API client.** Generation and labelling are done by Claude Code subagents through
  file exchange, driven from a campaign directory (plan4 §3); nothing in `decisionmodels/` calls
  a hosted API or holds a key.
- **Synthetic inputs only (FR-8).** `decisionmodels/` imports only the standard library and
  itself, and its source names no falkor-chat path, no FalkorDB/Redis reach and no side-check
  item file. `tests/test_fr8_isolation.py` is a static tripwire for that, with a planted-red case
  per reach; its docstring lists what a source scan cannot see.
- **One-way to model-bench.** model-bench's importer copies an eval split and a candidate in;
  this component never imports model-bench.

## Quick start

```bash
decision-models/setup.sh                  # create .venv, install the package + dev extra
cd decision-models && .venv/bin/pytest -q # run the suite from the component directory
.venv/bin/ruff check .                    # lint
```

## Status

The scaffold (U5.1) and the guard contract copies (U5.2): the guard input renderer and the
`input_hash` leakage key, pinned by byte-copies of model-bench's golden vectors. The layout the
later units fill is plan4 §3.1 over v1.3 §3.4; delivery is recorded in `docs/HISTORY.md`.
