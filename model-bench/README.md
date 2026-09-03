# model-bench

A standalone, human-started harness that measures **one local model at a time** against **one named
task pack**, stores the result with a full environment fingerprint, and compares it against
previously stored results **for the same role** — with confidence intervals and a visible flag
whenever a comparison is not apples-to-apples.

The point is not a score. The point is that a result taken today still lines up against a model
tested months ago, on this hardware and this data, instead of against a published leaderboard number
for a task nobody here runs.

Requirements: `docs/requirements/small-model-benchmarking.md` (repo root).
Design: `docs/plans/small-model-benchmarking.md` (repo root), plus the statistical method note
`docs/plans/small-model-benchmarking-ml.md`.

## Three things this tool deliberately is not

- **No CI hook, no scheduler, no timer.** `model-bench` only ever runs because a person typed a
  command. Nothing here is wired into any other component's test suite.
- **No pass/fail gate.** There is no threshold, no "bad score" and no non-zero exit for a result you
  did not like. The only non-zero exits are operational: bad arguments, unreachable LM Studio, an
  invalid pack, a missing fingerprint field.
- **No leaderboard and no cross-role aggregate number.** A "tool-caller" score and an "embedder"
  score do not add up to anything, so there is no command that adds them up. Comparison is always
  within one pack, and that is enforced by the shape of the code, not by a convention anyone has to
  remember.

It is also **standalone** (FR-23): no runtime code path reads anything outside `model-bench/` —
not `falkor-chat`'s configuration, not its model gateway, not its golden sets. Golden *data*
originating elsewhere is copied in, versioned here, and carries its provenance; that is data, not a
dependency.

## Status

**Stage S1 — the harness core is built; nothing calls a model yet.** What exists is the part that
decides whether a number may be printed: the environment fingerprint and its validation, the run
store and its quarantine-on-read, the statistics module, the markdown comparison, and three
commands. What does not exist yet is anything that produces a number: the LM Studio adapter, the
pack loader and the five task packs are stages S2–S7 of
`docs/plans/small-model-benchmarking.md` §4. The whole current suite runs offline.

## Quick start

Run everything with `model-bench/` as the working directory — the repo has no root pytest
configuration, so from the repo root pytest ignores this component's `testpaths` and walks into
other components' suites.

```bash
./setup.sh                        # create .venv, install the package + dev extra
.venv/bin/python -m pytest -q     # the suite (network-free)
.venv/bin/ruff check .            # lint
./run.sh --help                   # the CLI
```

## What the CLI does today

```bash
./run.sh compare --pack <pack-id> [--models a,b] [--session <id>] [--negative-control] [--out <path>]
./run.sh index rebuild            # regenerate results/index.csv from results/runs/
./run.sh models --tested          # models with stored results (never a deterministic arm)
```

`compare` reads `results/runs/`, renders the markdown comparison to `reports/` **and** stdout, and
never overwrites an earlier same-day comparison — the filename carries a two-digit sequence. It
exits `0` whatever the scores, including when every stored record turns out to be invalid: that is
a report, not an operational failure, and the excluded records are named in it with their reasons.

`--negative-control` puts **two copies of one stored record** in the two arms, so `b = c = 0` is
arithmetic rather than a measurement. It proves the mode is wired and nothing more, and the report
it writes says so in a banner at the top — the real negative control is two *independent* runs of
the same model, and that is an acceptance step.

`attest`, `validate` and `run` are stage S2 and are deliberately not wired yet.

Python 3.12, matching every other component. **Zero runtime dependencies** — stdlib only, on
purpose: a benchmarking tool whose own dependency tree can rot is a tool whose old results stop
being reproducible. `pytest` and `ruff` are the only dev extras.
