# ccg P1-34 reconciliation (`-v2` record) — implementation gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-34 steps 3–4)

## Pass 1

**Scope.** Commits `10655a29` (`cpg/spikes/frontend-fitness/s28_metrics.py` id constant/strings; `cpg/tests/test_flow_p1.py` tests + mutants) and `0d5ad055` (line 5 of `cpg/eval/preregistration.jsonl`), against `docs/plans/connected-code-graph-p1-flow-extract-pin-ml.md` §2–3 and root `REVIEW.md`. Not reviewed: the ruling's ML substance beyond the inertness re-check; the prereg/review text edits (§3 item 4, pending by design).

CPG: considered, not relevant — a pinned-record/constant reconciliation in an eval tool; no call-graph question.

**Verdict: approve with suggestions** (no blocker, no major).

## Findings

### F1 (minor) — `toolingCommit` and `extractRevision.commit` are 8-hex abbreviations; `-v1` carries a full 40-hex sha
Verified: `-v1` `toolingCommit` is `522b909e…` (40), `coreFreeze.commit` and `corpus.commit` are 40; `-v2` `toolingCommit` is `10655a29` and `extractRevision.commit` is `69789943` (8). The plan §3 item 2 says "full … shas verified by script" (said of the 64-hex pins, but the record's convention is full shas everywhere). No code reads either field (`grep -rIn toolingCommit` outside the record: only the plan), so nothing breaks today; both abbreviations resolve uniquely here (`git rev-parse` gives `10655a292cb440c505c8626f64c360f3fa57aaca`; `69789943` is the last commit touching `extract.sc`). But a pre-registration record is a long-lived pin, and an abbreviation can become ambiguous as the object count grows. Suggested: a one-line follow-up commit replacing the two values in line 5 with the full shas above (the `-v2` line has not been gated or executed against, so root `AGENTS.md` collision rule 5 permits in-place revision; do not touch lines 1–4). Do it before the custodian draw cites the record.

### F2 (nit) — `coreFreeze.commit` still reads `3089951f…` while `coreFreeze["extract.sc"]` is now the `69789943` blob
Copied unchanged from `-v1` as the plan prescribes, and `extractRevision` explains the extract.sc pin, so it is not wrong, only a field a reader could mis-take as the extract.sc commit. Optional: add to `extractRevision.note` that `coreFreeze.commit` dates `s28_core.py` only.

### F3 (nit) — no durable test ties the real `-v2` line to the tree
`tests/test_flow_p1.py` additions pin the tool constant and `-v1` addressability against the real file, but nothing in the suite runs `check_record` on the real record, so a later `extract.sc`/tool edit is caught only by the run-time `NOT-MEASURED:void`. That run-time refusal is the intended freeze guard (plan §3 item 3), so this is a suggestion, not a gap in the unit. Not pursued.

## Checks by execution (what was run)

1. **`s28_metrics.py` diff is id text only — verified.** `git diff --stat 10655a29^ 10655a29 -- cpg/spikes` = 4 insertions/4 deletions in that one file (constant, `load_record` docstring ×2 lines, `--record` help). Stronger: AST of the old file with `v1→v2` text-normalised equals the AST of the new file (`ast.dump` equality True); `v1` occurrences 4 → 0, `v2` 0 → 4. `flow_decide.py` / `flow_p1_audit.py` untouched (their pins are unchanged in `-v2`; their `v1` mentions are docstrings only — `grep` over `cpg/spikes/frontend-fitness/*.py` finds no code reading the id).
2. **Every pin equals the tree — verified, two ways.** Independent `sha256` of the working-tree files and of the blobs at `10655a29` equal the `-v2` pins for `extract.sc`, `s28_core.py`, `s28_run.py`, `s28_metrics.py`, `run_fitness.py`, `flow_decide.py`, `flow_p1_audit.py`, `flow-brief.template.md` (all 64-hex). `s28_metrics.check_record(load_record(real file), synthetic draw, …, runs={})` lists as mismatched only `cpgs.off/on`, `shippedFacts.off/on` (no run record supplied) — no script pin, `sample`, `corpus.commit` or `corpus.srcDigest`. Negative control: substituting the `-v1` `extract.sc` pin adds `extract.sc` to that list. Key-by-key diff `-v1` vs `-v2` (recursive): changed/added exactly `id`, `supersedes`, `coreFreeze["extract.sc"]`, `tools["s28_metrics.py"]`, `toolingCommit`, `extractRevision` (new), `tools["s28_metrics.py_note"]` (new). `corpus`, `cpgs`, `shippedFacts`, `seeds`, `frame`, `rules`, `decisionCell`, `entryValue`, `JOIN_P0_ONLY`, the other tool pins and `coreFreeze.commit`/`s28_core.py` are identical; no key removed. The added `s28_metrics.py_note` is outside the plan's list but is a note beside the pin, same shape as the existing `s28_run.py_note` — accepted.
3. **`-v1` line untouched; file only gained a line — verified.** Line 1 hashes identically (`f8a40b21…30c6`) at `96d6c267` (its introduction), `cd2fb98b`, `9cc55338`, `59bc1022` and `0d5ad055`; `cmp` of the first four lines against the file at `59bc1022` is identical; `git diff --stat 59bc1022 0d5ad055` = 1 insertion. Working tree under `cpg/` is clean.
4. **Inertness re-check — executed (independent of the ruling's scratch).** Ran `git show 3089951f:…/extract.sc` (sha `6cdb498f…`) and the tree `extract.sc` (`02001899…`) through `cpg/sandbox/offline-run.sh` offline on the two pinned CPGs (the `off` and `on` CPG shas equal the record's `cpgs.off/on`), outputs only in the session scratchpad. Each of the four runs gave 132,693 fact lines; the new version carries `"rt"` on 22,805 lines, the old on 0. Then the frozen `s28_core.load_facts` + `build_method` over all 6,254 methods per configuration: a hash of every method's full state is identical old vs new, `off` and `on` (`1142545ad185ac9a` both; my hash definition differs from the ruling's, so the digests are not comparable to its `6f279fa7…`/`cf19ae2e…`, and my off and on digests coincide where the ruling's differ — I did not chase that, the old-vs-new equality is the claim). Positive control: dropping every 20th `cs` fact changes the state hash (`efc76cd8…` → `e09b1b0e…`), so the comparison can fail. Scoring/join was not run (no labels or draw exist). Confirms ruling §1.
5. **Tests.** See "Test run" below.
6. **Tests/tools pinning `-v1` as live.** None. `test_superset_dev_tool.py:119` appends a synthetic `{"id": "ccg-p1-flow-python-v1"}` line as filler to a fixture record file (it tests a different tool and does not read `RECORD_ID`); `test_flow_p1.py` mentions `v1` only as the superseded id (`V1`, mutants). The old assertion at ~`:971` was moved to `v2`.

## What's solid

- `load_record` refusing a missing `-v2` line with no fallback is tested both ways (`test_load_record_without_a_v2_line_fails_loudly…` has a `-v1` line present to be wrongly fallen back to; mutant `load_record_falls_back_to_v1` plants exactly that). `record_id_still_v1` is checked against the literal `V2`, not the constant (`REVIEW.md`: expected values not read from the checked constant). The temp-copy helper never writes the real record.
- The plan's sequencing was followed: tool commit first, record second, pins from the committed blobs, `-v1` kept as history.

## Test run

Run from `cpg/` with `cpg/.venv`: `pytest tests/test_flow_p1.py tests/test_flow_decide.py tests/test_superset_dev_tool.py` = **382 passed, 1 skipped** (the skip is the opt-in `CCG_S28_RECOMPUTE` P0 rescore, not run). That includes every `test_mutant_is_killed[...]`, so the two new mutants (`record_id_still_v1`, `load_record_falls_back_to_v1`) are killed (each mutant re-runs the file; pattern-present assertion holds). `tests/test_ac24_smoke.py -k "prereg or heldout or registry"` (reads the real record file) = 29 passed. Rest of the `cpg` suite not run.

## Open questions

None blocking. F1 is a teco call: amend line 5 now (recommended) or accept abbreviations.

### Proposed REVIEW.md lines

None. F1 (convention drift inside one record) is a one-off, and a rule would not change what a reviewer does beyond the existing "pins verified by script" check.
