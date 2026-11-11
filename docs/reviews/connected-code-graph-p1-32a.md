# C-1001 P1-32a (eval verbs): review gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 P1-32a (P1)

## Scope & verdict

Reviewed the uncommitted P1-32a delivery (`docs/test-reports/connected-code-graph-p1-32a-report.md`) against `docs/plans/connected-code-graph-p1-gates.md` P1-32 (32a bullet), `docs/plans/connected-code-graph-p1-ml.md` §6.2/§6.4, and `docs/plans/connected-code-graph-eval-protocol-ml.md` §6.5, §12.1: diffs of `cpg/eval/tools/{capture,split_manifest,s47_scripted_steps}.py` and `cpg/tests/test_eval_tools_split.py`, plus new `cpg/tests/test_eval_tools_verbs.py`. 32b (`gate.py`) is out of scope. Root `REVIEW.md` applied; no `cpg/REVIEW.md` exists.

**Verdict: needs changes** (one blocker, in the freeze-check P1-commit predicate). Everything else is sound.

CPG: considered, not relevant — offline eval tooling over JSONL and git text; no call-graph question.

What I executed (foreground, `ulimit -v 2000000`, `timeout -s KILL`): `tests/test_eval_tools_verbs.py` + `test_eval_tools_split.py` + `test_eval_tools.py` = **579 passed** (matches the report); 7 mutants against fresh scratch copies under `cpg/.ccg-work/gate-32a/`; one read-only probe of the real git range. Not touched: custodian root, `~/.config/ccg`, Docker, FalkorDB, network, git tree/index.

## Findings

### B1 (blocker, verified) — the P1-commit predicate is anchored at the subject start and misses real P1 commits

`split_manifest.py` `P1_COMMIT_RE.match(subject)`. Real P1 dispatch commits in `511b14cf..HEAD` are subjected `cpg C-1001 P1-13: ...`, `cpg C-1001 P1-01: ...`, `cpg C-1001 P1-48 + P1-02: ...`, `cpg C-1001 P1 N-ML: ...`. Plan §6.2: "a commit is P1 iff its message matches `C-1001:? P1`" (a search, not a prefix).
Run: over the real 239 non-merge commits, `is_p1_commit` is true for **21**, an unanchored `re.search` for **26**. Over the real path set (HEAD standing in for the record; 414 paths), **22 paths** are classified `non-p1-commits` (excluded, never checked) only because of the anchor — e.g. `cpg/ccg/atrest.py`, `cpg/eval/tools/audit.py`, `cpg/clients/__init__.py`, `cpg/infra/toolchain/build.sh`. These are P1 builder files; the freeze check would report clean without scanning them. This is the exact failure §6.2 exists to prevent (a false-clean), and no exit code flags it.
The test enshrines the wrong rule: `test_eval_tools_verbs.py:798` asserts `not sm.is_p1_commit("other C-1001 P1")`. Mutating `.match` to `.search` is killed only because of that assertion, i.e. the report's "regex matching any phase" mutant row does not cover this axis.
Fix: use `re.search` (it still rejects `C-1001 P0 close` and `C-1002 P1`), flip the assertion at :798, add a case with the real `cpg C-1001 P1-13:` subject form. Also consider reading `%B` rather than `%s` (plan says "message"; the report's decision 6 narrows it to subject) — or record the narrowing in the plan wording. Then re-run the range probe above and state the excluded-path count by reason.

### M1 (major, inferred) — `regate-list` cannot detect a different `--contact` / `--eval-commit` / `--match`; report flags it but defers

Report "Cross-run state" and decision 4. The verb re-derives gate 1 via `split`; a wrong contact file silently changes which clusters are "kept" and what is fresh. Ruling: do not leave it to 32b by comparison of an anchor count. `eval/heldout-manifest.json` already carries per-file hashes (`files`, 7 entries) and `contactAnchors`; the verb can hash the re-derived gate-1 held-out `ac5-*.jsonl` files (using the same writer `write_jsonl`/`out_record` as `split`) and refuse with exit 1/4 on a mismatch against a `--manifest` argument (reuse `verify_manifest`'s hashing). Planted-input test: same inputs pass; one contact anchor dropped is refused. I did not run this design; the assertion that would catch it wrong is that mismatch test.

### m1 (minor) — `oracle-candidates` input shape

Ruling: accept the `{consumer, key, lines?}` assumption for now. It matches `-p1-ml.md` §6.5 "Observed edge: (consumer element, contract key)", the converter is isolated, refuses a non-contract key and is mutation-tested (executed none myself). Route to P1-47's step: it must cite this shape (or change `oracle_candidates` and its test) — otherwise `--heldout` gets probes in the wrong shape and the failure is a silent miss, not an error. Ask 32b/P1-47 for a test that feeds a real P1-47 edge file through `oracle-candidates` into `freeze-check`.

### m2 (minor) — `--probe-map` / `--out` of `freeze-check` write wherever the caller points

`main()` freeze-check branch. The map holds held-out-derived hashes (§6.4 custody). Nothing refuses a path inside the repo working tree. Not a bug (custodian runs it; report recommendation 4 says so), but cheap to guard: refuse a `--probe-map` under the git toplevel of `--repo`. Take or leave.

### m3 (minor) — byte-equality evidence is real but derivative

Checked `test_eval_tools_verbs.py:95-215`: the oracle is the separate script `s47_scripted_steps.py` run as a subprocess on one synthetic work dir and the verbs on a twin; every output file and stdout is compared. It is not a self-comparison. Its limit: the verbs' functions are visibly a port of that script, so equality is expected, and the plan's "on the stored inputs" is not met (custody, disclosed in decision 1). The custodian must run the `cmp` on the real work dir once; make that an owed item on the 32b/custodian checklist rather than only a report note. Ruling: keep `s47_scripted_steps.py` as the oracle until that `cmp` is done; delete after.

## Judged items

1. **Verb conformance.** `regate-list`: routing order (C/dev anchor, S(u), kept join, fresh), discard of items in dev/held/reserve and repeats, kept-join flag `kept-cluster-fresh` excluded from `freshCounting`, `FRESH_FLOOR = 35`, composition shortfall by stratum — match EP §12.1 steps 3-5; verified by reading and by mutants (floor `>=`→`>`, S(u) clause removed, reserve test widened: all killed). Exposure keeps first exposure (mutant overwrite: killed). Freeze-check: exit 2/4 is a failure (`code = 4 if failed`; mutant removed: killed), hashes are `sha(...)[:12]`, positive control demands exactly one new hit and a failed control sets exit 4. Custodian exclusions: directory prefixes carry the trailing slash and `cpg/eval/tools/` stays in (mutant dropping `cpg/eval/dev/` etc.: killed). Capture: `trigger = below and t == 2`, `belowThreshold` retained — matches EP §6.5 ("at most one extra pass"). Only B1 fails.
2. **`route_fresh` change.** Both changes follow EP §12.1 step 4 text: the S(u) clause is "the same holds for an item on a unit whose served set S(u) has an anchor in dev or in C (v1.2)", and only a *kept* cluster is joined; a reserve cluster is "never exposed" (§9.2), so fresh is right. `route_fresh` is used only by `regate_list` and the split tests (grep), so no gate-1 `split` output changes (579 tests green, including the 84 split tests). The `test_capture_three_list_example` assertion change is the P0 nit per EP §6.5 and keeps the threshold assertion (`belowThreshold`). Justified, not a silent contract change. Caveat on one point: a fresh contract item routed to dev does not feed `exposed()` for fresh unit items' S(u) within the same call (only cluster side and C do); consistent with the plan text, but not tested — nit.
3. **Custody.** `grep` for `ccg-eval`, `.config/ccg`, `expanduser`, `Path.home` in the tests and tools: no hits. Verbs read only paths given by argument; stdout carries counts, 12-hex hashes and repo paths of hit files, never item content. No finding.
4. **Delegate decisions:** see M1, m1, m3.
5. **Test quality.** Mutants I executed against scratch copies (each `assert count==1` on the replacement; run with `CCG_TOOLS_DIR`, `-x`): floor `>=`→`>` killed; S(u) clause removed killed; exposure overwrite killed; custodian dirs dropped killed; freeze-check exit-4 on failed files removed killed; reserve joined as kept killed; `.match`→`.search` killed (only through the :798 assertion, see B1). 7 of 7 killed, matching the report's table for these rows. Red-first is claimed only for the capture trigger (stated honestly); `regate-list`/`freeze-check` tests were written after implementation, offset by mutation runs. The one gap is B1's axis.
6. **Growth-stops.** `git status docs/plans` shows no plan edit; the report only proposes plan wording (decisions 2, "Feedback" item 2). Confirmed.

## What's solid

Import-not-reimplement from `match_keys.py`; refuse-to-overwrite for lists, atomic exposure write; honest disclosure of the custody-forced test limit and of tests-after-code; scratch-git-repo freeze tests incl. a real submodule unreadable-blob case; mutation-driven additions where mutants first survived.

## Open questions

- None blocking. Plan wording: should §6.2 say "subject" or "message"? Decide with B1's fix.

### Proposed REVIEW.md lines

None.

## Pass 2

**Verdict: approve with suggestions.** B1 and M1 are fixed. I re-ran `test_eval_tools_verbs.py` + `test_eval_tools_split.py` + `test_eval_tools.py`: **602 passed** (foreground, `ulimit -v 2000000`, `timeout -s KILL`). No plan document was edited (`git status docs/plans`: clean).

- **B1: fixed.** `P1_COMMIT_RE.search(message)` over `git log --format=%B` (`split_manifest.py` `commit_subjects`/`is_p1_commit`). Own range probe over `511b14cf..HEAD` (read-only, HEAD standing in for the record): 239 non-merge commits; anchored subject rule 21; `search` on subject 26; `search` on `%B` 26 (no body-only P1 commit exists in the range). Path set 414: 275 excluded `non-p1-commits`, 139 scanned, 0 custodian — equal to the claim under both definitions (the subject and `%B` path classifications are identical here). Error direction: a `search` over the message can only add files to the scan (cost: one extra file checked; a hit there is triaged by the custodian), never drop one; the residual silent-exclusion risk is an untagged P1 commit, which is `teco`'s tagging duty (§6.2) and unchanged. The `:798` anchored assertion is gone (`is_p1_commit` is now asserted on real subject shapes, lines ~818-830, incl. body-only tag). Mutants executed against fresh scratch copies: `.search`→`.match` killed (by a real-subject case, right behaviour); `%B`→`%s` in the git format killed (body-tag real-git case). Suggestion (minor): `-p1-ml.md` §6.2 says "message"; the qa report still lists "implementation tests the subject line" as plan wording to fix (report, Feedback item 2) — stale after the fix, drop it.
- **M1: fixed.** `regate-list` now requires `--manifest`; `gate1_problems` compares schema, `evalCommit`, `splitSeed`, `contactAnchors`, and per-file sha256/bytes/items of the six split products, hashed in memory via `split_files`. Executed: `test_regate_manifest_*` (unchanged accepted; dropped contact anchor, dropped item-less anchor, different commit, missing held-out item, corrupted hash each refused with exit 1; helper asserts neither output dir exists and stderr carries field names only, e.g. `ac5-http.jsonl: sha256`). Mutants executed: sha256 comparison dropped killed; `contactAnchors` check disabled killed; write-before-refuse path removed (outputs not written) killed.
- **Same bytes as `split` writes (premise).** I ran old (`git show HEAD:` copy) and new `split_manifest.py split` on the same synthetic 200-item match file: stdout md5, held-out dir (7 files) and dev dir are **byte-identical** (`diff -r`). So the `split_files` refactor (used by `write_split` and by the M1 check) did not change `split` output; the existing split tests are green. Residual (qa report discloses it): that the real S47 manifest was produced by this `split` over these inputs cannot be shown in custody; the custodian's first real run will show a mismatch as a refusal, which is the intended behaviour.
- **m2: fixed.** `freeze-check` refuses `--out`/`--probe-map` inside the `git rev-parse --show-toplevel` of `--repo` (exit 2, before any write). Mutant (guard disabled) killed. Note the guard keys off `--repo`'s toplevel, so a custodian run with `--repo` pointing elsewhere than the working tree is not covered; acceptable.
- **m1, m3, nit:** unchanged by design (oracle-edge shape stays P1-47's to cite; real-input `cmp` stays an owed custodian step; the fresh-to-dev/`exposed()` nit is pinned by a new test). Agreed.
- **New findings:** none.

Route: `qa-engineer` to delete the stale plan-wording note (report Feedback 2); `teco` to carry the owed custodian `cmp` and the M1 first-real-run expectation into the 32b/custodian checklist.
