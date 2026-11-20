# model-bench — review standing instructions

## What counts as serious

- A bar, threshold or verdict rule that a degenerate candidate can meet (one that always answers
  the majority class, or never hands off), or that no candidate can meet under the planned
  procedure, is at least major.

## Always check

- A test of a bar, threshold, verdict or report-clause comparison probes both sides of its
  boundary, including the equality case, so a `<`/`<=` or `>`/`>=` mutation turns one red.
- Every selection among runs, candidates or settings (last-wins, a re-run, a re-tune, a re-fit
  on the same pack version) is recorded and counted against the evaluation split. An
  unrecorded selection on evaluation data is a finding.
- A claim about LM Studio load state (one load per run, which model is resident, a reload between
  replicates) is checked against the code path and procedure that load the model, including JIT
  loads triggered by a call.
- A change to what a stored result, hash or fingerprint means has canonical vectors whose
  expected values were written independently of the function under test, and covers every rule
  its docstring claims (key order, non-ASCII, which fields enter the hash).
- A test of a measured duration (an item clock, a call clock) uses a fake clock that the stubbed calls advance by their own durations, and asserts the exact sum, so a mutant that starts or stops the clock at the wrong point changes the asserted value.
- A change that routes runs around `compare_report`'s `runs` (as `replicates`, or as any new argument) shows that every guard the `runs` path applies still reaches them: the pack-version and content-hash banners, aggregate consistency, and the schema banner. Each one is pinned by a test on the rerouted runs.
