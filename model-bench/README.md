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

**Stage S0 — component skeleton only.** The layout, the packaging and the test/lint loop exist;
there is no CLI, no pack, and no harness code yet. `model-bench/run.sh` says so and exits non-zero
rather than pretending. Stages S1–S8 are specified in `docs/plans/small-model-benchmarking.md` §4.

## Quick start

```bash
model-bench/setup.sh                             # create .venv, install the package + dev extra
model-bench/.venv/bin/pytest model-bench -q      # the suite
model-bench/.venv/bin/ruff check model-bench     # lint
```

Python 3.12, matching every other component. **Zero runtime dependencies** — stdlib only, on
purpose: a benchmarking tool whose own dependency tree can rot is a tool whose old results stop
being reproducible. `pytest` and `ruff` are the only dev extras.
