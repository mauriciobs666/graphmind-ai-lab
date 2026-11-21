# model-bench defects (five fixes) — Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** GAL-F0028 (defects the live spec recorded)

**Scope.** `git show 3aafd1bb`: `model-bench/modelbench/{cli,runner}.py`, `tests/test_cli.py`, `tests/test_runner_decisions.py`. Baseline: decisions in `model-bench/docs/plans/model-bench-defects-coordination.md`; root and `model-bench/REVIEW.md` applied. Not looked at: DESIGN/FORMATS/README/manual edits (a later unit owns them; stale statements are only listed, F5).

**Verdict: approve with suggestions.** All five fixes do what they say, and each new test goes red against its mutant. Two majors: the defect-5 fix leaves the same crash reachable (F1), and one decision (defect 3) is the stakeholder's (F2).

`CPG: considered, not relevant — a ~40-line change in one stdlib-only module, fully read; no call-graph question.`

**Checks run (observed).** `cd model-bench && .venv/bin/python -m pytest -q` -> `3661 passed, 5 deselected, 2 xfailed`. Mutants were run on a copy of the tree in the scratchpad with `PYTHONDONTWRITEBYTECODE=1` and `__pycache__` cleared. A first M3 run reused a stale `.pyc` (same size, same-second mtime) and gave a wrong result; it was discarded and redone. The repo tree was not touched.

| Mutant (copy) | Result |
|---|---|
| re-add `run --reference` | `test_run_refuses_a_reference_flag_as_a_usage_error` red |
| refusal threshold `> 2` -> `> 99` | `test_plain_compare_over_more_than_two_arms_refuses...` red |
| exit-5 message restored with transcript path | `test_step_10_exit_5_message_names_no_transcript_file...` red |
| `--strict` back to `raise NotImplementedError` | `test_validate_strict_refuses_cleanly...` red |
| `_start_of_run_read` bypassed (both calls) | both parametrizations red |
| wrapper `exitCode=3` -> `4` | red |
| only `catalog` unwrapped | the `[catalog]` case red |
| **`not negative_control` dropped from the new branch** | **suite green (3658 passed): survives (F4)** |
| threshold `> 2` -> `> 1` | 9 tests red (the 2-arm path is well pinned) |

CLI exercised on the real parser: `run --pack x --model m --reference r` -> argparse `unrecognized arguments: --reference r`, exit 2; `rank --help` still lists `--reference`; `validate --pack packs/chat-responder-grounded-answers/ --strict` -> one stderr line, exit 2. A compare with 0, 1, 2 stored arms exits 0; with 3 arms it exits 2, and `--models m0,m1` over those 3 arms exits 0.

## Findings

### F1 (major, verified) — defect 5 is fixed for one exception class and two of the four call sites

`_start_of_run_read` catches only `LMStudioUnreachable`, and wraps steps 3 and 3a (`runner.py:1904,1907`). Other LM Studio reads between the step-2 probe and the first item are unwrapped:
- `runner.py:2019` step 7 `lmstudio.residency()` and `runner.py:2021` step 8 `lmstudio.catalog()`, both bare.
- `lmstudio.catalog()` raises `LMStudioCallFailed` for a non-200 status or a malformed body (`lmstudio.py:630-645`), and `residency()` goes through it. The wrapper does not catch that class, so steps 3/3a still leak it.

I reproduced all of it with a throwaway test in the scratchpad copy (removed). The stub raised `LMStudioCallFailed(status=500)` from `residency` or `catalog` at step 3/3a, and `LMStudioUnreachable` from the second `residency()` (step 7) and the second `catalog()` (step 8). Every case propagated out of `main()` as an uncaught traceback (exit 1), the same symptom the fix targets. The module's own convention treats all three of `LMStudioUnreachable`, `LMStudioCallFailed` and `TimeoutError` as one class (`runner.py:369-380`, `1095`, `1640`, `2095`). The recorded defect names only the timeout, so what shipped meets the letter but not the class.

Suggested fix: widen the wrapper's except to `(LMStudioUnreachable, LMStudioCallFailed, TimeoutError)` and route steps 7 and 8 through it (with a `what` naming "baseline residency" / "loaded-context catalog"). Extend the parametrized test over `{unreachable, callfailed} x {step 3, 3a, 7, 8}` using a stub that fails on the n-th call. Note the step-7/8 cases need a counting stub: the present stub fails every call, so step 3 always trips first. Alternatively state in DESIGN that 7/8 stay exit 1 and drop the claim "start-of-run read"; I do not recommend that.

### F2 (major, stakeholder decision) — defect 3: `> 2` versus the decision's literal "unless exactly two"

The decision text says refuse "unless exactly two arms exist". The implementation refuses only for more than two (`cli.py:248`). I ran the literal `!= 2` mutant (negative control excluded) against the full suite: it turns two existing tests red, `test_compare_session_naming_a_session_with_one_arm_reports_too_few_arms` (`test_cli.py:~428`) and `test_compare_exits_zero_even_when_every_record_is_invalid` (`test_cli.py:~139`). Both expect exit 0 and the "fewer than two arms" report for a plain compare with 1 / 0 valid arms.

- **Implementer's reading (`> 2`).** It fixes the recorded defect, an arm silently left out of the verdict, with no collateral. 0 or 1 arms are not silent: the report says "fewer than two arms were selected" (existing behavior, `report.py:782`). It keeps DESIGN §2.2 "exit 0, tool ran and reported" and plan §3.6a ("a result worth keeping, not an operational failure", `docs/plans/small-model-benchmarking.md:1635`). Cost: a deviation from the literal decision text, and an empty or one-run pack stays exit 0, which a script cannot distinguish from success.
- **Literal reading (`!= 2`).** Uniform rule, so "exit 0 means a comparison happened". Cost: an exit-2 "usage error" for a state that is a data condition (no runs yet, an all-invalid pack, a filtered session), two tests rewritten, and the `--session` case of the existing contract reversed. `--negative-control` must stay exempt either way.

My view: `> 2` is the better fit with the standing contract, but this contradicts a stakeholder decision, so it is theirs to rule. I do not call the deviation an error: the commit and ledger disclose it. Either way DESIGN §2.2/§2.3 must say which.

### F3 (minor, traced) — the refusal's advice can dead-end for repeated runs of one arm

Plain `compare` counts every valid run as an arm (`cli.py:226-253`), but `--models` resolves keys through `by_key` (newest run per `armKey`). With three stored runs of one model `m` and no other, the message reads `found 3 (m, m, m); --models a,b selects which arms`, yet `--models m` yields one arm ("fewer than two"). Neither the message nor the test (distinct keys only) covers it. Suggested: de-duplicate by `armKey` before counting (the `by_key` rule, i.e. newest), or say "stored runs" rather than "arms" in the message. Undeduplicated, 3 runs of one model is the likeliest way to hit the new refusal.

### F4 (minor, verified by mutant) — the `not negative_control` exemption is unpinned

Dropping it leaves the suite green (3658 passed). `compare --negative-control` over more than two stored runs is exactly the workflow the clause protects (it uses the first record as both arms). Add one test: three stored arms plus `--negative-control` -> exit 0 and the banner. It must fail with the clause removed.

### F5 (minor) — statements these fixes make wrong (listed, not edited)

- `docs/DESIGN.md` §2.2: lines 106-109 (a start-of-run timeout is an uncaught traceback, exit 1) and 111-112 (exit-5 message names `results/transcripts/...`); exit-code table row `3` (add the start-of-run read) and row `2`.
- `docs/DESIGN.md` §2.3 line 131 ("verdict compares the first two arms", plain compare now refuses above two) and line 128-130 ("every valid run is an arm").
- `docs/DESIGN.md` §2.8 lines 197-199 (`--strict` raises `NotImplementedError`, exit 1).
- `docs/DESIGN.md` §2.9 line 204 (`[--reference <key>]` in the `run` synopsis) and lines 216-217 ("accepted and ignored by `run`").
- `docs/manuals/small-model-benchmarking.md` lines 119-120 (`--reference` on `run`), 173-176 (first two arms, file-name order) and 457-459 (`validate --strict` "crashes with a traceback and exit 1"). `README.md:55` is the `rank` line and is correct.
- `docs/BACKLOG.md` GAL-F0028 (all five bullets) leaves with `HISTORY.md`.
- `tests/test_scoring_extraction.py:15` and `:774` carry comments about working around the `cli.py` `NotImplementedError` deferral; the comments are stale (the suite is green), the code not re-checked for dead workaround.

### F6 (nit, verified) — `referenceKey` is now a constant

`runner.py:166` declares `referenceKey: str | None`, with no default. Nothing in `modelbench/` reads it (`grep -rn referenceKey`: only the declaration and `cli.py:752`'s `None`), and about eight test constructions pass `referenceKey=None`. No reader misbehaves. The hardcoded `None` with a comment is a defensible way to avoid touching the tests; it is dead data on a record-adjacent dataclass. Remove the field in a later cleanup (the 8 sites are mechanical) rather than keep a permanent `None`.

## What's solid

- Defects 1, 2 and 4: minimal, the right exit, tested, and each test goes red against its mutant. The `--strict` test also asserts a single stderr line, no stdout and no "Traceback".
- The defect-5 wrapper itself: refuses before any write (test asserts no `results/` files), `from exc` chaining preserved, exit 3 pinned by the `4` mutant.
- The `--reference` removal leaves `rank --reference` intact (`cli.py:132`, 563).
- The commit message and ledger disclose the defect-3 deviation openly.

## Open questions

1. **Stakeholder (F2):** refuse plain `compare` for 0 and 1 arms as well (literal `!= 2`, two existing tests rewritten), or only for more than two (as shipped)?
2. Is widening defect 5 to `LMStudioCallFailed` and to steps 7/8 (F1) in scope for this wave, or a new backlog item?

## Pass 2

Scope: fix round `6c7dda13` (code + tests) against Pass 1. Rulings folded in: defect 3 stays "more than two arms" (F2 answered, no code change); defect 5 widened (Open question 2 answered). Suite re-run: `3669 passed, 5 deselected, 2 xfailed`. Mutants on a scratchpad copy, bytecode off.

**Verdict: approve with suggestions.** F1/F3/F4 are fixed. One residual (R1) is the original defect 3's own wording and still reproduces; it needs a ruling or a fix.

`CPG: considered, not relevant — a ~40-line change in one stdlib-only module, fully read.`

Dispositions:
- **F1 fixed.** The wrapper catches `(LMStudioUnreachable, LMStudioCallFailed, TimeoutError)` and wraps steps 3, 3a, 7 and 8. Mutants: dropping `CallFailed` turns 4 cases red; step 7 bare turns 2 red; step 8 bare turns 2 red. Every read between the probe and item 1 is covered: the other `lmstudio.*` reads in `run_pack` (`_between_item_residency`, `_within_load_basis`, step 10's retries, `_decision_instance_config`) carry their own handling and are correctly left alone (`grep 'lmstudio\.\(residency\|catalog\|loaded_instances\|probe\)' runner.py`).
- **TimeoutError is untested (mutant `(Unreachable, CallFailed)` only: suite green, 3666 passed), and it does not matter.** `_raw_get` catches `TimeoutError` and folds it to `None`, so `residency()`/`catalog()` cannot raise a bare `TimeoutError`; the branch is defensive and matches the module's grouping. Either drop it from the wrapper or add one parametrized case; take-or-leave (nit).
- **F2 answered** (ruling), nothing to recheck.
- **F3 fixed.** Counting is by distinct key (`dict.fromkeys`); the mutant counting runs turns `test_plain_compare_counts_distinct_arms...` red. Consistent with `--models`' one-arm-per-key resolution: the advice can now always narrow.
- **F4 fixed.** Dropping the exemption turns `test_plain_compare_with_a_negative_control_is_exempt...` red.
- **F5 open (owner: the live-spec unit).** The ruling leaves the list valid; it must now state that plain `compare` counts distinct keys. **F6 open** (nit).
- Same-revision interaction: wrapper widening vs the unwrapped reads is clean (above). Distinct-key count vs `--models` is clean. The interaction that breaks is R1.

### R1 (major, reproduced) — 2 distinct keys over 3+ runs still pairs the first two runs by file name, possibly a model against itself

Real parser: stored `aaa` (40/40), `aaa-rerun` (an `aaa` run, 10/40), `bbb` (34/40); `compare --pack guard-judge-understanding` exits 0. The arms table lists three rows, and the verdict is "aaa (run aaa) is better than aaa (run aaa-rerun) ... +75.0 pp, McNemar p=0.000". `bbb` is never compared. The "(run X)" labels disclose which runs were paired, so it is not silent about identity, but the headline is a model against itself.

This is still defect 3, not a different matter. The recorded wording is "pairs the first two arms by file-name order, not the newest": the cause is plain `compare` taking runs, not arms, in file order, and the new count-by-key check only decides whether to refuse. The "more than two arms" ruling does not cover it: here there are exactly two arms (keys) and the pairing is still wrong. The shipped test pins the opposite (`aaa`, `aaa-rerun`, `bbb` -> exit 0) without asserting what was compared, so it can't catch this.

Suggested fix (in `_select_arms`' plain branch, before the count): collapse to the newest run per `armKey`, the rule `--models` already uses (`{r.armKey: r for r in candidates}`; file-name order sorts by arm key then start time, so last is newest). Effect: the arms table lists one row per key (newest), and repeat runs drop out of a plain compare's table (they stay in `models --tested` and reachable via... nothing in `compare`; if repeat-run display matters, keep them in the table and pair only the newest of each key, a larger change in `compare_report`). I ran the collapse as a mutant on the copy: `tests/test_cli.py` stays green (83 passed), so no existing test pins the repeat-run rows. Add a test asserting the verdict names `aaa` and `bbb`, not `aaa` twice. The stakeholder should confirm that dropping repeat runs from a plain compare's table is acceptable; I can't tell from the repo whether the `(run X)` labelled rows are a designed feature (they suggest it) or a by-product.

## Pass 3

Scope: `f9773ffb` (plain `compare` keeps the newest run per `armKey`; ruling: newest run per key). Suite: `3674 passed, 5 deselected, 2 xfailed`. Mutants on a scratchpad copy.

**Verdict: approve with suggestions.** The reproduction now compares `aaa` with `bbb`; one minor remains (R2).

`CPG: considered, not relevant — a ~40-line change in one stdlib-only module, fully read.`

Dispositions: **F1 fixed** (the new `socket-timeout` cases go red when `TimeoutError` is dropped: 4 red; the Pass 2 nit is closed). **F2** ruled. **F3 fixed** (the "no collapse" mutant turns the distinct-arms test red). **F4 fixed** (Pass 2 mutant). **R1 fixed:** the real-parser rerun of `aaa`/`aaa-rerun`/`bbb` gives one arms-table row per key (aaa 40/40, bbb 34/40) and a verdict `aaa` vs `bbb`; the mutant keeping the oldest run per key turns the new test red, the no-collapse mutant turns two tests red. **F5 open** (add to the list: plain `compare` keeps one arm per key, its newest run; repeat runs are not in its table). **F6 open** (nit).

### R2 (minor, reproduced) — "newest" is the last file name, not the latest `startedAt`; both branches

`_select_arms` takes the last run in file-name order (`load_history` sorts `directory.glob("*.json")`). That is the newest only for harness-written ids (`<packId>-<armKey>-<UTC stamp>`, `runner.py:1396`). I stored `aaa-final` (started 11:00, 40/40) and `aaa-v2` (started 10:00, 20/40) plus `bbb`: both plain `compare` and `--models aaa,bbb` pick the older `aaa-v2` (20/40) and report "bbb is better than aaa +35.0 pp", exit 0, no warning. The `--models` branch has the same property (it predates this commit), so this is not a regression and not a blocker.

It is a finding rather than only a documentation point: `store()`/`load_history` accept any bare file name as `runId` (FORMATS: "must be a bare file name"; the grammar is not enforced), and the wrong run silently changes the verdict. Suggested fix, one place serving both branches: choose per key the run with the greatest `fingerprint.startedAt` (ties: file-name order). Name the assertion: the scenario above (older run's id sorts last) must pick the 40/40 run on both `--models` and plain paths. If the owner prefers to leave it, the live spec must state that "newest" means last in file-name order, valid for harness-written ids only. Either is acceptable; the stakeholder's call, and it needs no new decision before close if the spec states it.
