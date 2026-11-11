# RCA: `test_two_real_joern_runs_of_one_unit_give_one_bundle_and_it_is_the_stored_one` fails

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

CPG: considered, not relevant — the cause is a diff of two commits and of two bundles; no graph query was needed.

## 1. Symptom & impact
`cpg/tests/slice/test_normalize_equivalence.py::test_two_real_joern_runs_of_one_unit_give_one_bundle_and_it_is_the_stored_one` (`-m slice`, ~68 s) fails on the working tree. The two real runs agree with each other; only the comparison with the stored `salesperson` bundle (`expected_bytes("sp")`) fails. The other 21 tests in the file pass (re-run: `21 passed`). Impact: one red slice test; no product defect. The "not a regression of any of today's changes" question below is answered.

## 2. Reproduction & evidence
I ran the test's core (`run.extract_unit` once, then `assemble_bundle.build_bundle`, then compared with `expected_bytes("sp")`) through the real sandbox runner. The driver and diff scripts are `scratchpad/rca-slice/driver.py` and `cmp.py`. Each candidate commit was a `git archive` of `cpg/` into scratch, with `.ccg-work` symlinked and the real repo used only for its git objects. I used no `git checkout`, `worktree`, `stash`, `reset` or `restore`, and did not modify the real tree.

| Source tree | Result vs expected bundle |
|---|---|
| Working tree (HEAD + uncommitted) | differs: `data` 553 vs 441, `controlledBy` 456 vs 451 |
| `3089951f` (P1-09) | **identical diff to the working tree** |
| `3ee0d50b` (parent of P1-09) | **identical** to expected: 2775 lines, zero record differences (only `sandboxAuditIds`, which is exempt) |

### Characterisation of the difference (working tree), by record type and field
| Record type | Fresh vs expected |
|---|---|
| header | `counts.data` 553 vs 441, `counts.controlledBy` 456 vs 451 (consequences of the records below); `sandboxAuditIds` (exempt, new each run) |
| `data` | **112 extra records**, none missing, none changed. All `reason=intra-flow`, `status=confirmed`; `fromPath`/`toPath` are field paths such as `p1.status`, `r.kind`, `p0.product`, `r.data.ord` → `a0`. They are new def→use edges. |
| `controlledBy` | **5 extra records**, none missing, none changed |
| `symbol`, `callsite`, `cond`, `param`, `ret`, `valueSet`, `constant`, `file`, `import` | identical |
| `history` / blame fields (`lastChangedAt`, `lastChangedCommit`) | identical (`symbol` 257/257, no field diffs) |
| header tools/version fields | identical (`normalizerVersion` 0.4, script hash, tool versions) |
| ids | the extra records are new ids; no existing id changed |

The class of change is an **additive, semantic extension of the Joern data-flow projection**: more edges, with nothing removed or altered.

## 3. Causal chain
1. The live test runs a fresh extraction through `run.extract_unit`, which executes the shipped `extractors/joern/extract.sc` (`run.EXTRACT_SC`, `extractors/joern/run.py:44`). `extract_script=SLICE_SCRIPT` in the test only changes the *hashed* script file in the header; it does not change which script runs.
2. Commit `3089951f` (P1-09, "projection fix": R1 same-name earlier-use reaching defs, R2 constructor/`<init>` defs, R3 iterator-lowering transparency, 4b/4f computed-key and temp-mutation use positions, the `for` header cond) deliberately changed `extract.sc`. A diff against `spikes/vertical-slice/extract.sc` shows exactly those additions.
3. Those rules produce more `d` (data) and `cb` (controlledBy) facts. The normalizer, unchanged, turns them into 112 + 5 extra records.
4. The oracle is the slice's stored bundle, built by the old script. `DELTAS` in the test (`drop-lambda-decorators`, `p1-06-mask-cond-code`, `p1-08-skipped-files`) has no delta for P1-09's projection change. The P1-09 commit updated `tests/slice/test_extract_equivalence.py` but not `tests/slice/test_normalize_equivalence.py`. Its `git log` shows the last edit at `2c0e4f5c`, which is P1-08.
5. The stored-facts tests (`written` fixture) pass because they feed the stored raw facts and never run Joern, so they do not see the new projection. Only the live tier does.

## 4. Ruled out
- **Hardened git helper (`gitsafe`, empty HOME / config disabled / `--no-textconv`, mailmap).** Blame records are field-for-field identical, and the tree at `3ee0d50b` (before `192b3253`, which introduced `gitsafe`) has no difference. It also fails at `3089951f` with the same counts. A different blame outcome would show up as `symbol` diffs, and there are none.
- `ccg/units.py`, `ccg/atrest.py`, `stage_unit`/`checked_command`, `NORMALIZER_VERSION`, bundle schema/`HOST_LANGS`: header version/tool fields are equal and no staged-file, `file`, `import` or `symbol` difference exists.
- Gitignored runtime state: the stored bundle and facts are the same files for all three runs (sha pinned by the passing `test_the_stored_bundle...` tests). Run-to-run nondeterminism: the two real runs agree with each other.

## 5. Root cause
**Confirmed** (reproduced and bisected to a single commit): the oracle is stale relative to the intended P1-09 behaviour change. The code is right; the stored slice bundle is the pre-P1-09 baseline.

- **Trigger:** commit `3089951f` (P1-09), run through the live tier.
- **Contributing factors:** the P1-09 gate did not run `-m slice -m live` on the whole slice directory (or did not read this test as affected). The delta mechanism (`DELTAS`) covers normalizer changes but has no entry for an extraction-script change. The test passes the slice's `extract.sc` only for the hash, which suggests the author believed the run was script-equivalent. It is not.

## 6. Fix and owner
Owner: `graph-dba` (owns P1-09 and the projection golden sets) or `tdd-engineer`. A test-only change; no production change.

Choose one:
- **(a) Preferred:** add a `DELTAS` entry (version unchanged, since `NORMALIZER_VERSION` did not move) for the live tier only. Cheaper and more honest: in the live test, compare with the expectation **built from the stored raw facts through the current normalizer plus the P1-09 projection golden**, or assert the live bundle equals the stored one *plus* exactly the 112 `data` and 5 `controlledBy` ids written by hand (an allow-list of ids). That keeps the claim that the extra records are explained.
- **(b)** Re-record the oracle by regenerating the stored `sp` bundle (and `fcs`) under the P1-09 script, and re-pin `SP_BUNDLE_SHA`. This invalidates the 5b byte-equivalence tests, which depend on the stored raw facts staying the slice's, so (a) is safer.

Prevention: the live equivalence test must run in any gate that touches `extract.sc`; `tests/golden/projection/extract.sc.sha256` (added by P1-09) could be asserted in this file so a script change forces an explicit decision here.

Not done: I did not run the same test on `fcs` (Python unit); the same extra records are expected there and would be absorbed by the same fix.
