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

The scaffold (U5.1), the guard contract copies (U5.2: the guard input renderer and the
`input_hash` leakage key, pinned by byte-copies of model-bench's golden vectors), and the
decision points' definitions with seed generation (U5.3: `points/<role>/` rubric, generator
template and design per point; `decisionmodels/seeds.py`), and the request/answer exchange
(U5.4: `decisionmodels/exchange.py` writes closed, paged, id-blind requests;
`decisionmodels/ingest.py` checks answers), the `input_hash` rows for join-in and message
type (U5.2b-hash), and the transcript audit (U5.5: `decisionmodels/audit.py` locates a subagent
run from the driver's transcript and classes it per plan6 §3.4 over plan4 §3.3), and the
campaign ledger with its driver CLI (U5.6: `decisionmodels/campaign.py`,
`decisionmodels/driver.py`). Labelling requests render guard items only until U5.2b-render.
The layout the later units fill is plan4 §3.1 over v1.3 §3.4; delivery is recorded in
`docs/HISTORY.md`.

## Running a campaign

A campaign lives in its own root, outside any git work tree and any folder under a `CLAUDE.md`
or `AGENTS.md`. Run these from `decision-models/`:

```bash
.venv/bin/python -m decisionmodels.driver init --point guard-judge --phase smoke \
    --root ~/dm-campaigns/smoke --fixture items.jsonl     # smoke: one label batch from a fixture
.venv/bin/python -m decisionmodels.driver init --point guard-judge --phase pilot \
    --root ~/dm-campaigns/pilot --launch-fail-bound 6h    # --batch-timeout defaults to 45m
.venv/bin/python -m decisionmodels.driver status --root ~/dm-campaigns/pilot
.venv/bin/python -m decisionmodels.driver resolve --root ~/dm-campaigns/pilot \
    --decision "why the halt is cleared"   # or --skip-batch, --repin-instructions,
                                           # --repin-model generator|labeler <id>
```

- **`init`** refuses, writing nothing, a root that is not empty or sits in a git work tree or
  under a `CLAUDE.md`/`AGENTS.md` (judged on its physical path, so a symlink does not hide one),
  `--phase smoke` without `--fixture`, a pilot for any point but guard, and `--phase full`
  unless the guard pilot's report (`data/guard-judge/pilot/pilot-report.json`) passes the spend
  gate, records `stakeholderDecision.cap` as 25 or 50 and `rngSeeds` as a non-empty list of
  ints, and the previous point's dataset is finalized. A full campaign holds the pilot's RNG
  seeds, so it never reuses one. `--launch-fail-bound` must exceed one usage window on the
  account in use.
- **A torn ledger** (a last line with no newline, from a crash mid-append) or a line that is not
  an event refuses every command, naming the byte offset. A torn last line is always safe to truncate,
  because every writer appends its event before acting on it; any other bad line needs a hand repair.
- **`status`** prints the batches by state, a halt and its reason, whether `store/pause` exists,
  a launch refusal (`store/refusal`), the last issued session id, and any dropped batches.
- **`resolve`** is the only way past a halt. Retry re-issues the halted batch under a new name;
  `--skip-batch` (a safeguards stop only) drops it and counts it as `safeguardSkipped`; a
  re-pin records the new model (a generation or labelling restart) or instructions set.
- `install` (U5.7), `run`, `pause` and `resume` (U6.3) and `finalize` (U5.10) are not built
  yet, so no batch is spawned today.
