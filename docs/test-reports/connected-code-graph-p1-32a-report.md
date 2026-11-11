# C-1001 P1-32a: evaluation verbs, test report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** C-1001 P1-32a (P1)

Scope: the "32a (verbs)" half of P1-32 (`docs/plans/connected-code-graph-p1-gates.md`). 32b (`gate.py`) is out of scope.
Run date 2026-11-10, working tree on top of `6115edc7` (uncommitted; `teco` commits).
CPG: considered, not relevant — the unit adds offline evaluation tooling under `cpg/eval/tools/` over JSONL and git text; no call-graph question was needed.

**Verdict: delivered, green.** `tests/test_eval_tools_verbs.py` 84 passed; with `test_eval_tools_split.py` and `test_eval_tools.py` 602 passed (one run, one command). Every mutant that was run is killed except one equivalent mutant (removed from the code, see Mutation table). The analyst gate's B1 (P1-commit predicate) and M1 (`regate-list` manifest check) are fixed (see "Gate fixes"). No defect found in the pre-existing tools except the P0 nit (`capture` trigger), which is fixed here.

## What was built

| Verb / change | Where | Spec source |
|---|---|---|
| `capture`: `trigger` cleared once KA2 ran (`t == 3`); new `belowThreshold` keeps the "completeness below threshold" caveat | `cpg/eval/tools/capture.py` | EP 6.5, P0 report nit |
| `regate-list` (+ `route_fresh` extended with the S(u) clause and reserve = fresh; `FRESH_FLOOR = 35`; `regate_list`, `write_regate`) | `cpg/eval/tools/split_manifest.py` | EP 12.1 steps 3-5, E12 |
| `exposure` (`append_exposure`) | same | EP 9.2, 12.1 step 6 |
| `apply-corrections`, `ambiguity`, `split-ac24` (the AC-24 half of `split`) | same | EP 7.4, 10.3, 10.2 |
| `freeze-check` (+ `--probe-map`, `--positive-control`), `oracle-candidates` | same | `-p1-ml.md` 6.2, 6.4 |
| docstring updates (verb list; replay record pointer) | `split_manifest.py`, `s47_scripted_steps.py` | |
| tests | `cpg/tests/test_eval_tools_verbs.py` (new, 84 tests); one assertion in `cpg/tests/test_eval_tools_split.py` changed (below) | |

`match_keys.py` is untouched; every function used from it (`normalize_contract`, `record_probes`, `leak_check` via `check_report`, `locator_id`, `collapse`, `sha`, `canon`, `nfc_strip`, `_norm_path`, `_ids.tuple_id`) is imported. `s47_scripted_steps.py` is kept (docstring only), because it is the oracle the byte-for-byte tests run; `teco` decides whether to delete it afterwards.

**Changed existing assertion (a contract change, not a weakened test):** `test_capture_three_list_example` asserted `trigger is True` at `t = 3`. That is the P0 nit; it now asserts `belowThreshold is True` and `trigger is False`.

## Decisions taken (routine, for the record)

1. **Byte-for-byte against the replay record on synthetic inputs, not on the S47 stored inputs.** The S47 work inputs and held-out outputs live under `~/.local/share/ccg-eval/`, which is off limits. The tests run `s47_scripted_steps.py` and the three verbs on the same synthetic work dir and compare every output file and the stdout. Only the dev `cpg/eval/dev/ac24.jsonl` is in the repo (14 lines, observed with `wc -l`); the synthetic set is built to give 30 held-out and 14 dev, but that file is not byte-compared (its inputs are not available). The custodian can run the same comparison on the real inputs.
2. **Verb names:** AC-24 half of `split` is `split-ac24`; the join is `ambiguity` (writes the kept questions, gold normalized, and the `ac24-log.json`); freeze check is `freeze-check`.
3. **`regate-list` input:** it re-derives gate 1's state by calling `split` on the same `--match/--context/--contact/--eval-commit` and refuses (exit 1, nothing written) unless the re-derived lists hash to the required `--manifest` (see "Gate fixes"); `--missed` is JSONL `{anchorId}`; `--fresh` is a `match_keys scope`-shaped file (`{"records": [...]}`). The new list goes to a NEW held-out dir (refused if it already holds an `ac5-*.jsonl`, EP 12.1 step 7); item flags go to `regate-flags.jsonl`, so the key-item record schema is unchanged. Reserve items become fresh-cluster only with `--promote-reserve`. The pass / "all hits, floor not met" decision is left to 32b; the verb reports `freshCounting`, `floor`, `floorMet` and the composition shortfall per stratum.
4. **Route changes to `route_fresh`:** it now also applies the S(u) clause (EP 12.1 step 4, v1.2) and treats an anchor in a reserve cluster as a fresh cluster (never exposed). The previous behaviour called a reserve anchor "kept". The existing split suite (84 tests) still passes.
5. **`exposure` file** is rewritten sorted by `itemId` (EP 9.2 says every file is sorted); existing rows are never changed; a gate lower than one on record is refused.
6. **P1 commit test** is `re.search(r"C-1001:? P1", message)` over the full message (`%B`), not an anchored subject match (see "Gate fixes"); a path with no attributable non-merge commit in the range stays in the set; a custodian output is excluded before the non-P1 test; a path absent at the record commit is listed `deleted`, not skipped silently and not a failure.
7. **`oracle-candidates` input shape is my assumption** (`{consumer: locator, key: contract locator, lines?}`), because P1-47/P1-36 have not fixed the edge file. If P1-47 writes another shape, only this converter changes.

## Results (test items)

The test-plan row is P1T-30 (`docs/test-plans/connected-code-graph-p1.md`); this unit does not add rows. Items below are per verb.

| Item | Result | Evidence |
|---|---|---|
| capture trigger cleared at t=3, `belowThreshold` kept | pass | red first: 3 failed before the fix (observed); then green; mutants killed |
| apply-corrections byte-identical to replay | pass | `test_scripted_steps_corrections_match_and_log_are_byte_identical`: match-corrected.json, corrections.jsonl and ac24-log.json equal byte for byte; removed ids exactly the incorrect-and-union ones (an `unclear` verdict and a `single` kind are kept) |
| ambiguity join byte-identical to replay | pass | `..._ac24_files_are_byte_identical...`, `..._stdout_is_the_replay_stdout`, `..._ambiguity_drop_rules_and_gold_rewrites` (outcomes 36 / 8 / 2 / 1 / 1; gold rewrites counted; provider-form `{#}` keys, never `{?}`) |
| split-ac24 byte-identical, 30 held-out / 14 dev | pass | `..._counts_are_30_held_out_and_14_dev...`, `..._split_rule_by_rank` (6 lowest ranks per cell held out; lowest-rank ambiguous per language dev), seed includes the eval commit |
| exposure | pass | 11 test cases: first-gate append canonical and sorted, first exposure never overwritten, idempotent rerun, malformed rows / duplicate / bad gate / corrupt record each exit 4 with the file unchanged |
| regate-list | pass | 32 test cases (24 + 8 manifest/routing cases added at the gate): miss cluster to dev, others kept; discard rule for dev / held / reserve and for repeats; routing by anchor (dev, C incl. an item-less C anchor, kept, fresh, reserve anchor); S(u) clause by C, by a miss, and by the fresh unit item's own `serves[]`; kept join not counted fresh; floor 34/35/36 and non-counting items; composition shortfall; served-anchor moves not counted as misses; reserve promotion; out-of-scope fresh items; refusals; CLI counts-only, deterministic, never overwrites a list, contact/commit/missed/promote reach the library |
| freeze-check, oracle-candidates | pass | 30 test cases (freeze or oracle) on a scratch git repo (real subject shapes, body-only tag, out/probe-map guard included): path set `git diff --name-only <S47> <record>`, exclusions with reason per path (non-P1-only, custodian outputs; `cpg/eval/tools/` stays; mixed commit counts as P1; `device.py` is not `dev/`), a planted held-out probe gives exactly one hit per file (two files, two hits, none in excluded files), hashes only on stdout, probe-map maps the hash to the item, `--dev` and `--contact` remove public probes, exit 2 on an incomplete held-out list, exit 4 on a bad revision, a submodule path (real git, unreadable blob) is a failure, per-file rc 2 and rc 4 are each a failure with no hits, a failed positive control is exit 4 |
| oracle-candidates | pass | key-item schema, contract normalized, lines sorted and deduped; a non-contract key refused; candidates passed by `--heldout` hit a planted leak that the file list without them does not |

Commands (all run foreground under `ulimit -v 2000000` and `timeout -s KILL 120`, cwd `cpg/`, `.venv/bin/python -m pytest`):

| Command | Observed |
|---|---|
| `tests/test_eval_tools_verbs.py` | 84 passed in 11.35 s |
| `tests/test_eval_tools_verbs.py tests/test_eval_tools_split.py tests/test_eval_tools.py` | 602 passed in 22.34 s |
| `tests/test_eval_tools.py` (before the final test additions) | 434 passed |

No full cpg suite, no Docker, no Joern, no network, no FalkorDB. `free -m` after each batch: available memory stayed between 4.6 and 4.7 GB.

## Mutation table

Runner: `cpg/.ccg-work/p1-32a/mutate.py` (fresh copy of `cpg/eval/tools` per mutant under `.ccg-work/p1-32a/m/<id>/`, exactly-one replacement, pytest under `ulimit -v 1500000`, `timeout -s KILL 45`, `CCG_TOOLS_DIR` pointing at the copy; mutant lists `mut1.json` ... `mut5b.json` beside it). A mutant counts as killed when the targeted run reports a failure.

| Area | Mutants (all killed unless noted) |
|---|---|
| capture trigger | trigger ignores `t`; empty stratum at t=3 still triggers; `belowThreshold` lost at t=3 |
| apply-corrections | union-only rule removed; verdict `!= correct` instead of `== incorrect` (first run SURVIVED: the data had no third verdict; data added, killed) |
| ambiguity join | both-ambiguous rule loosened; annotator disagreement leniency (first run SURVIVED: only one disagreement direction in the data; reverse case added, killed) |
| split-ac24 | 5 held per cell; ambiguous dev count 2; rank order reversed; seed without eval commit; gold form `consumer` instead of `provider` (first run SURVIVED: no masked key in the data; masked key added, killed); rewrite count disabled; held/dev directories swapped; eval commit replaced by a constant |
| apply-corrections output | plain `json.dumps` instead of canonical (argument/format corruption) |
| exposure | existing row overwritten (first-exposure rule); `hit` truthy not boolean; gate order check removed; gate >= 1 check removed (first run SURVIVED: masked by the order check; fresh-file case added, killed); unsorted write; duplicate results accepted; gate written as 1; hit forced True; corrupt-record check removed; CLI gate constant; CLI `hits` counts already-recorded items (first run SURVIVED; assertion added, killed) |
| regate-list routing | C test removed; dev-cluster test removed; S(u) clause removed; S(u) clause only for C; kept join counted as fresh; kept test includes reserve |
| regate-list discard | reserve not in the discard set; dev not in it; held not in it; no dedupe inside the fresh list |
| regate-list floor | `FRESH_FLOOR` 34; 36; `>` instead of `>=`; counts non-counting items; counts kept items as fresh |
| regate-list other | served-anchor rule skipped; miss clusters not moved; missed-anchor validation removed; S(u) not recomputed with the fresh items; reserve always promoted; moved-out counts zeroed; overwrite guard removed; out-of-scope fresh items kept |
| regate-list CLI arguments (silent) | contact set dropped; eval commit replaced (first run SURVIVED, killed after the CLI-equals-library test); `--missed` ignored; `--promote-reserve` ignored (first run SURVIVED, killed by the same test) |
| freeze-check exit codes | rc 2/4 skipped; only 4; only 2 (each killed by its own parametrized case); failed files do not set exit 4; hits do not set exit 1; unreadable file skipped |
| freeze-check path set | regex without the optional colon; regex matching any phase; mixed path excluded (`all` for `any`); no-commit path excluded; `cpg/eval/tools/` excluded; `dev` prefix without trailing slash; manifest not excluded; `decisions/` not excluded; deleted file treated as clean |
| freeze-check arguments (silent) | `--out` / `--probe-map` guard removed, or only one of the two guarded (killed); `--dev` dropped; `--contact` dropped; `--s47` ignored; `--record` ignored; held-out list check bypassed; probe hash length 11 |
| positive control | always true; plants nothing; exit not set on failure |
| oracle-candidates | contract not normalized; lines not sorted; non-contract key accepted |

Equivalent mutant: dropping `and len(after) == len(before) + 1` in `positive_control` survived because the preceding set-difference check already implies it; the redundant clause was removed from the code instead.
One mutant (`fc-hash-len`) was a bad mutant on its first run (pattern did not match the source); it was corrected and run (killed).

## Coverage and gaps

- **Not reproduced on real S47 inputs** (custody): see decision 1. The custodian should run `s47_scripted_steps.py` and the verbs on the real work dir once and `cmp` the outputs; the verbs' file formats equal the replay record's by construction and by the differential test.
- **Test-first discipline:** the capture trigger and the scripted-step and exposure tests were written before their implementation (the capture tests were observed red). For the scripted-step verbs and exposure I did not capture a red run before implementing; the mutation runs show the tests bite. `regate-list` and `freeze-check` were implemented first and tested after, then mutation-tested; I say so rather than claim red-green for them.
- **Real git history:** the path-set tests run on scratch repos built in the tests; the real range was probed read-only with `HEAD` standing in for the record (numbers under "Gate fixes"). No record commit exists yet. Whether every P1 dispatch commit carries the tag is `teco`'s duty (§6.2).
- **Binary files** in the path set are decoded with replacement characters and checked as text; a leak inside a binary would only be found as text.
- **`regate-list` does not re-validate kept items against a new `EVAL_COMMIT`** (EP 12.1 step 4, last bullet; GC and the method owner do that) and does not decide the AC-5 pass; both are 32b's or the custodian's.
- **`oracle-candidates` input shape** is an assumption (decision 7).
- **Cross-run state:** closed by the manifest check (M1). Residual: the check proves the re-derived gate-1 lists equal the manifest's; it cannot prove the real S47 manifest was produced by the current `split` code (the real manifest was not re-derived here, custody). The custodian's first real run shows it; a mismatch there is a finding, not noise.

## Feedback and recommendations

1. **Proposed text for `cpg/README.md`** (not edited, other units hold uncommitted edits there): in the eval-tools row, replace "regate-list, exposure ... not built" wording, if any, with: "`split_manifest.py` verbs: `split`, `manifest`, `verify`, `contact`, `check-report`, `regate-list`, `exposure`, `apply-corrections`, `ambiguity`, `split-ac24`, `freeze-check`, `oracle-candidates` (P1-32a); `s47_scripted_steps.py` is the S47 replay record the tests compare against." I did not find such a row to edit; this is the exact text to add where the eval tools are listed.
2. **Plan wording to fix (`docs/plans/connected-code-graph-p1-gates.md` P1-32a):** it names "the AC-24 half of `split_manifest.py`" without a verb name; the verb is `split-ac24`. `-p1-ml.md` 6.2 says "C-1001:? P1" matches a commit "message"; the implementation matches `C-1001:? P1` over the whole message (`%B`), as the plan says.
3. **For 32b:** import `FRESH_FLOOR`, `regate_list`'s report keys (`freshCounting`, `floorMet`, `shortfallByStratum`, `minimumsMet`) and the `regate-flags.jsonl` item flags rather than recomputing; import `check_text`/`freeze_check` is not needed by the harness.
4. **Custodian procedure before the record:** run `freeze-check` with `--probe-map` and `--out` under the custodian root (they hold held-out-derived hashes and the map), pass the oracle candidates (from `oracle-candidates`) after the held-out files in `--heldout`, and keep `--positive-control` on; exit 2 or 4 means the check did not run and must not be recorded as clean.
5. **Learning capture:** the `ws:agent-team` MCP server was unreachable in this session (connection refused), so no `ingest_document` was made. Candidate facts for `cobb`: (a) `CCG_TOOLS_DIR` makes `test_eval_tools*.py` load the tools from a scratch directory, which is the clean way to run per-mutant copies; (b) a git submodule path in `git diff --name-only` is a real, cheap way to make `git cat-file blob` fail for a per-file-failure test.

## Files

- Changed: `/home/mauricio/prg/graphmind-ai-lab/cpg/eval/tools/capture.py`, `/home/mauricio/prg/graphmind-ai-lab/cpg/eval/tools/split_manifest.py`, `/home/mauricio/prg/graphmind-ai-lab/cpg/eval/tools/s47_scripted_steps.py` (docstring only), `/home/mauricio/prg/graphmind-ai-lab/cpg/tests/test_eval_tools_split.py` (one assertion).
- New: `/home/mauricio/prg/graphmind-ai-lab/cpg/tests/test_eval_tools_verbs.py`, this report; scratch under `/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/p1-32a/` (gitignored work area: `PROGRESS.md`, `mutate.py`, mutant lists, mutant copies).

## Gate fixes (analyst gate: B1, M1, minors)

**B1: P1-commit predicate.** Chosen: `re.search(r"C-1001:? P1", message)` over the full message (`git log --format=%B`). Reasons: `-p1-ml.md` 6.2 says "message matches", real subjects are `cpg C-1001 P1-13: ...` and `cpg C-1001 P1-48 + P1-02: ...` (an anchor drops them), and the error direction: a missed P1 commit makes the check report clean without scanning a builder file, while over-inclusion (for example a P0 commit whose body quotes `C-1001 P1`) only scans one more file. Subject-only `search` would still miss a tag that lives only in a body, so I took the message. It still rejects `C-1001 P0 close`, `C-1002 P1`, and non-C-1001 subjects. The `:798`-style assertion that enshrined `not is_p1_commit("other C-1001 P1")` is gone; the test now parametrizes real subject shapes (`C-1001 P1: ...`, `cpg C-1001 P1-13: build`, `cpg C-1001 P1-01`, `cpg C-1001 P1-48 + P1-02: ...`, `cpg C-1001 P1 N-ML: ...`, `C-1001: P1 plan set`, `C-1001 P1T-30`) which must be P1, five non-P1 subjects that must stay excluded, and two real-git commits (cpg-prefixed subject; tag only in the body) whose files must stay in the scan.
Range probe (read-only, `511b14cf..HEAD`, `HEAD` standing in for the record): 239 non-merge commits; matched 21 by the old anchored subject rule, 26 by `search` on the subject, 26 by `search` on the message. Of 414 diff paths, 275 are excluded as `non-p1-commits` and 139 are scanned (no custodian output in the range; 0 paths without an attributable commit). Under the old rule 297 were excluded, so 22 real P1 files were dropped from the scan (`ex_old - ex_new = 22`, `ex_new - ex_old = 0`). On this range `%B` finds no further path beyond what the subject search finds.
Mutants: `.search` -> `.match` killed by the real-subject cases (`test_freeze_check_real_p1_subject_shapes_are_p1_commits[cpg C-1001 P1-1...]`), which expect the right behaviour; message -> subject-only search killed by the body-tag case; `%B` -> `%s` in the git format killed by the real-git body-tag commit.

**M1: `regate-list --manifest`.** Premise checked by execution on synthetic inputs: the manifest's `files` are the five `ac5-<s>.jsonl` and `ac5-reserve.jsonl` (plus `ac24.jsonl`, which is not a split product), hashed over the same bytes `split` writes. `test_regate_manifest_premise_...` builds the manifest through the real `split` then `manifest` CLI verbs and shows the in-memory re-derivation (`split_files`, which `write_split` now also uses, so nothing is written to disk to hash held-out data) has identical sha256 per file. The verb now requires `--manifest`, compares those six hashes, bytes and item counts plus `evalCommit`, `splitSeed` and `contactAnchors`, and on any mismatch exits 1 with field names only on stderr and writes nothing. Planted-input tests: unchanged inputs accepted; dropping a contact anchor that moved a cluster to dev is refused; dropping an item-less contact anchor is caught by the count; a different eval commit, a match file missing a held-out item, and a corrupted manifest hash are each refused; a missing `--manifest` is exit 2. Mutants killed: comparison skipped; contact count check removed; evalCommit check removed; sha256 comparison removed; CLI exit mapped to 0; manifest not passed to the library; wrong contact count.

**Minors.** (m2) `freeze-check` now refuses `--out` and `--probe-map` paths inside the git toplevel of `--repo` (exit 2, nothing written), tested for both flags. (m1/m3 of the review: no code change; the oracle-edge shape stays an assumption for P1-47, and the real-input `cmp` stays an owed custodian step.) (Nit) A fresh item routed to dev does not feed `exposed()` for other fresh unit items in the same call. I did not change that: routing reads cluster sides and C only, per EP 12.1 step 4, and the only fresh items that route to dev through S(u) are unit items, which are never in another unit's served set, so the case cannot change a result; `test_regate_a_fresh_unit_item_routed_to_dev_does_not_expose_another_fresh_unit_item` pins the behaviour.
