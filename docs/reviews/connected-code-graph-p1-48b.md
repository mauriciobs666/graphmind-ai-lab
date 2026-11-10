# Review: C-1001 P1-48b, the runner's indexer entries (diff-scoped)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

Reviewed, uncommitted: `cpg/sandbox/offline-run.sh`, `forbidden-flags.txt`, `tests/test_sandbox.sh`, `tests/mutants.sh`,
`cpg/sandbox/README.md`, `cpg/infra/README.md`, `cpg/infra/toolchain/README.md`. Baseline: the P1-48b row of
`docs/plans/connected-code-graph-p1.md`, its section in `docs/plans/connected-code-graph-p1-extraction.md`, and
`docs/plans/connected-code-graph-egress-spec3.md` v1.4 §7.4–§7.6, §10.1 tests (21)–(26). Root `REVIEW.md` applied (no
`cpg/REVIEW.md`). Not reviewed: `cpg/ccg/broker/*` (P1-29) and the deep security pass (separate advisory).

**Verdict: approve with suggestions.** No blocker. The implementation matches the spec where the spec is explicit, the
checks I attacked can fail, and the one real defect (the `--project-version` gap) was found by the delegate and is a spec
decision, not a runner bug. Two spec statements need a clarifying edit (F1, F2).

CPG: considered, not relevant — the change is Bash, Markdown and a text list, outside any loaded `cpg_<component>` graph.

Checks run, all on copies under the session scratchpad, throwaway `--rm --network none` containers only, no tree-mutating git:
- Tests 21–26 (`--only 21,22,23,24,25,26`): **230 passed, 0 failed, 0 not run, exit 0** (verified, run).
- Absent indexer image (`CCG_TEST_INDEXER_IMAGE=ccg-nonexistent:x`): tests 24 and 26 print `NOT RUN`, exit **3**, no `ok` (verified).
- `HERE=/nonexistent` copy: exit 2 (`image  missing`); with `CCG_TOOLCHAIN_IMAGE` preset, the indexer tag lookup fails and the suite
  reports `FAIL ... cannot resolve the indexer image tag` plus two `NOT RUN`, exit **1** (failure outranks 3) (verified).
- Two test-file mutants (below, F6) and a sample of runner mutants (below).

## Findings

### F1 (minor, spec): empty argv, and forbidden-over-missing precedence, are the delegate's rulings; the spec does not state them
Evidence: `offline-run.sh:sbx_indexer_grammar` (`n<1` -> `(missing index)` hit; hits checked before `SBX_IDX_MISSING`);
`:230` is the joern precedent `(missing --script)`. Spec §7.4: "A missing mandatory element is `usage`; any other element ... is
`forbidden-flag`"; §7.5 lists "a bare invocation" and "a missing mandatory indexer element" under `usage`. For `index --pnpm-workspaces`
(no `--cwd`) both sentences apply; for empty argv, `index` itself is a mandatory element or the whole argv is "not matching".
The spec is silent on both, so the delegate's reading (hostile element first; empty argv is `forbidden-flag`) is defensible and consistent
with joern, but it is a ruling. Verified: mutant I21 (usage before forbidden) is killed, and the README states the rule.
Fix: one sentence in §7.4 (architect): "a forbidden element outranks a missing mandatory one; an empty argv is `forbidden-flag` with
hit `(missing index)`". Test (22) already pins both; no code change.

### F2 (minor, spec): `/src/` with zero components is accepted; rule: keep it, and say so
Evidence: `sbx_idx_path_ok` returns 0 for `/src/`; test 22 launches `--cwd /src/`. Spec §7.4: "`/src` or `/src/` followed by
components none of which is empty" reads both ways (zero components is vacuously none-empty, or `/src/` is the prefix of at least one
component). Ruling: accept. `/src/` and `/src` name the same directory inside a read-only mount, the escape cases (`//`, `.`, `..`,
trailing `/x/`) are all refused and tested (I10–I13), and refusing it would only break a stager that appends a separator. Fix: the
architect adds "`/src/` alone is accepted" to §7.4's `<p>` line and test (22)'s launch list.

### F3 (major, spec decision): `scip-python` cannot index a staged source that has no git revision; the grammar has no `--project-version`
Verified by execution (indexer image present, `ccg-toolchain-indexer:73b9fb76bd2a`, `--network none`, read-only `/src`):
- no `.git`, no `--project-version`: `Indexing /src with version undefined`, a Fatal Error stack in `symbols.ts` `makeModuleInit`
  (`normalizeNameOrVersion`), **exit 1**, no useful index. Stderr also shows `fatal: not a git repository`.
- same source with `--project-version 1`: `Sucessfully wrote SCIP index`, exit 0.
- same source with a `git init` + one commit copied in (`.git` present in `/src`): exit 0.
The delegate's finding is confirmed. Test 22 hides it (`scip-python` exit code is "not asserted"); the code is correct to the spec.
**Recommendation: amend §7.4 to admit `--project-version <v>`** (optional, `scip-python` only, at most once, own two-element form),
`<v>` matching `^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$` (covers a 40-hex sha and a semver), and have P1-15's stager pass the source repo's
HEAD sha, which it already knows without staging `.git`. Security weighing:
- *Admit the flag:* adds no execution surface. The value is data written into SCIP metadata/symbol names; the leading-alnum rule
  blocks option smuggling (same bar as `--project-name`, I14). Cost: a grammar row, mutants and tests (flip the existing "not in the grammar" case).
- *Stage `.git`:* `scip-python` then runs `git` inside the sandbox against a target-controlled `.git/config`. Git reads repo config for
  helpers (`core.fsmonitor`, `core.pager`, `diff.external`, `core.sshCommand`/`gitProxy`) and `safe.directory` ownership checks on a
  mount owned by another uid may make it fail anyway (not tested). It widens the stager's reject-list (hooks, config, packed objects
  as symlink/hardlink targets), copies history into the mount, and makes the version depend on repo state. The sandbox bounds the
  blast radius (no network, read-only, caps dropped), but it is a new execution path for no gain over a pattern-checked string.
I recommend the flag, not a decision I can make for the plan: it is a §7.4 contract change owned by `architect`, and `merge_scip` must be
told what the version token looks like. Make it mandatory for `scip-python` only if P1-15 wants the failure to be a `usage` refusal
rather than an exit-1 tool error; I would, since the optional form lets the known failure through as a recorded launch.

### F4 (minor): forbidden-flag entries and the spec's "13 forbidden, 3 allowed" count
Evidence: `forbidden-flags.txt` now has 17 lines (13 Joern + 4 `scip-typescript`); `allowed-flags.txt` still 3. Spec §7.4 (line ~907):
"(13 forbidden, 3 allowed, for Joern v4.0.579)". **The spec sentence is the stale part, not the list.** The Joern count remains true
for Joern; the sentence has no indexer clause, while §10.1 (24) requires the indexer sweep to classify its hits and the spec's own
widened regex (`pnpm|yarn|npm|pip|workspace|infer`) cannot produce a hit that is not listed somewhere. The README line 16 already
says "13 ... plus 4". Justification per entry, checked against `scip-typescript index --help` in the image (all four flags present):
`--pnpm-workspaces`, `--yarn-workspaces`: spawn a package manager (evidence cited in the list: child_process); `--yarn-berry-workspaces`:
deprecated alias of the same path; `--infer-tsconfig`: **weakest**, "no evidence it is inert" is the stated reason, which is the
spec's own default ("forbidden unless evidence"), and it writes a file into a read-only `/src`. Acceptable fail-safe, all four are also
outside the closed grammar. Note the entries do not match the base regex, only the widened one; they are listed because the sweep
requires it, which is fine. Fix (architect): reword the sentence to "13 Joern forbidden, 3 allowed, plus 4 `scip-typescript` forbidden".
No code change; no test pins 13 (swept `cpg/sandbox` for the number).

### F5 (minor, test): `scip-python`'s launch is asserted only through its record
Evidence: `test_sandbox.sh` (test 22) prints `info: scip-python exit code N (not asserted ...)`. A scip-python-only mutant
(e.g. `sbx_frontend_path scip-python` returning a wrong path, rc 126/127) would pass: I27 changes the shared directory and is killed
only through `scip-typescript`'s rc check (observed rc 127 in the table). Fix: assert `RC != 126 && RC != 127` for `scip-python` and that
its stderr contains `Indexing /src with version` (a string the tool prints before it can fail), then drop the assertion when F3 lands
and assert exit 0 and a non-empty `.scip`.

### F6 (nit, test): the README hard-codes counts
`sandbox/README.md` "22: 167 checks in all" and "(14 keys)" will drift with the next grammar case; cite the test, not the number.

## Answers to the brief's checks

1. **Precedence / empty argv:** see F1. Spec silent; delegate's reading consistent with joern, tested, mutant I21 killed.
2. **`/src/`:** F2, accept.
3. **Indexer on `controlled` / `auto` / label last:** verified by reading `sbx_main` and `sbx_try_container`, and by test 22 (controlled
   -> `unknown-frontend` 64 for both tools; namespace -> 69 with "container backend only"; a grammar refusal outranks it, 64 not 69;
   `auto` with the default image -> 69 `backend-unavailable`, never the namespace precondition). The label check runs only when no
   earlier refusal is set (`[ -z "$SBX_BACKEND_REFUSAL" ]`), so a PE-binfmt or interop refusal keeps its own reason. One note: an
   explicit `--backend namespace` for an indexer skips `sbx_resolve_backend` when `SBX_DCFG_FAIL` is empty, so the record's
   `preconditions` are the measured baseline only; consistent with "applied after the argv checks".
4. **Forbidden entries and the count:** F4.
5. **Vacuity:** exercised as listed in the scope section. Test 26's control (`env -u PYTHONSAFEPATH`) was mutated by dropping the `-u`:
   both control checks went red (2 FAIL), so the control can fail and the main assertion cannot pass because the plant is dead. Test 24's three
   doctored copies each assert `rc==1` and the named reason in the output, plus the unwidened-regex control (`hits=0`); mutating the
   `f-missing` doctoring to a no-op turned the "unclassified" alarm red (1 FAIL). No `ok` is printed on the `NOT RUN` paths.
6. **`timeout` in `run_r`:** `timeout -k 15 ${RUN_TIMEOUT:-300}` runs the runner in its own process group. The runner's launch is
   `< /dev/null` (`offline-run.sh:696`), nothing reads a TTY or `PPID`, and the signal tests 12/13 do not go through `run_r`. A pass
   leaves the exit code unchanged; only a >300 s run becomes 124 and then fails the exact-code checks, i.e. red, not green. The earlier
   tests were not run again in this pass in full (tests 1–15 and 4 not rerun here; the delegate's full run is the evidence for those).
   No semantic change found. Residual: 300 s is a guess; a slow host could produce a spurious red on test 2's parse runs, not a false green.
7. **`scip-python` exit 1:** F3, verified.
8. **Test-file mutation:** done, F-section above (26 control; 24 doctoring), both turn red.

## What's solid
- The grammar implementation follows §7.4 point by point (two-element options, abbreviations and `=` forms refused, repeats refused,
  `--project-name` pattern, per-tool option sets), with 41 mutants covering each rule; the paths `/src//a`, `.`, `..`, `/srcfoo` all refused.
- Fixed absolute executable path, `PYTHONSAFEPATH` only on the indexer pass-list, label check last, `NOT RUN` never printing `ok` and
  exit 3 outranked by failure.
- READMEs now say what the code does; the old "no indexer path" text is removed everywhere (swept `cpg/infra`).

## Open questions
1. F3: does the architect admit `--project-version` (my recommendation) or stage `.git`? It changes spec §7.4 and a test (22) case.
2. Should `--project-version` be mandatory for `scip-python`, so the known tool failure is a `usage` refusal?

## Mutant sample
See the Appendix.

## Appendix: runner mutants run on a copy
`mutants.sh I13 I21 I31 I16` from the scratchpad copy: killed=4 survived=0, each first failing check as the README table states
(I13 `--cwd '/'`, I16 first element `npm`, I21 bare `--version`, I31 test 7/23/26 pass-list). The other 37 table rows are not re-run here;
the README's "killed" column for them is the delegate's claim.

## Pass 2

Baseline: `docs/plans/connected-code-graph-egress-spec4.md` (`317baccc`) §1–§2 against the uncommitted tree in the same seven files.
Checks, all on scratchpad copies, throwaway containers: **full `test_sandbox.sh`: 873 passed, 0 failed, 0 not run, exit 0** (run).
Not re-run: the 116 mutants (implementer's claim); I re-ran only the ones named below.

**Verdict: approve.** No blocker, no major. One nit (N1).

Dispositions:
- **F1 (precedence, empty argv):** fixed. Spec4 §1 states both; `sbx_indexer_grammar` and `sbx_main` order grammar, element audit, containment, then `usage`; test 22 precedence cases pass.
- **F2 (`/src/`):** fixed. Spec4 §1 accepts it; `sbx_idx_path_ok` and the launch `--cwd /src/` agree. Probed `/src/` through the walk: contained.
- **F3 (`--project-version`):** fixed as recommended, and mandatory. Runner: `--project-version` is accepted for `scip-python` only, once, with the `--project-name` pattern, and its absence is `usage`. Tests cover the `=` form, repeat, no value, `scip-typescript`, the empty, `-x`, `--x` and 65-char values. `scip-python` launches now exit 0 with a non-empty `.scip`. The pattern matches spec4.
- **F4 (counts):** fixed in the spec (13 + 4 = 17); the list is unchanged.
- **F5 (`scip-python` launch unasserted):** fixed. `py_launch_check` asserts not 126/127, `Indexing <cwd> with version`, exit 0 and a non-empty `.scip`, including through in-tree symlinks.
- **F6 (README counts):** fixed; no hard-coded check counts remain in `cpg/sandbox/README.md`.

Spec4 checks, by execution:
- **Containment walk.** I called `sbx_idx_path_ok` then `sbx_idx_contained` on a scratch tree, 20 values. Refused: `lnk2 -> deep/abs` (a relative link to an absolute link) and `lnk2/x`; `deep/abs`; `esc -> ../outside`; `up2 -> ../../outside/..`; dangling; a symlink loop; a host-absolute link whose target lies inside the tree; a missing file. Accepted: `/src/`, a subdirectory, a relative link, `rel/inner`, `self/self/sub`, a trailing-slash link target, `up -> sub/inner/..`. A `..` after a followed link resolves against the link's own directory, which matches the kernel.
- **Walk against the character class and `/src/`.** The two checks compose: `/src/sub/../env.json` is stopped by the class check before the walk; `/src/` has no components and reaches the tree root; a component like `..a` is a plain name and walks normally. Link targets are not class-checked, only walked, which is correct: they are tree content, not caller input.
- **Containment before `usage`:** verified by reading `sbx_main`, and by mutant (runner copy with the containment block moved after the `usage` refusal): 6 FAIL, including the three `precedence` cases. Killed.
- **Image by ID:** one `docker run` site (`offline-run.sh:518`), image element `$SBX_IMAGE_ID` (`:532`); the label check also inspects the ID (`:454`). Test imgid checks parse and indexer launches. Mutant (launch by `$SBX_IMAGE`): `22 image by ID: parse launch` and `indexer launches` FAIL. Killed.
- **Test-file mutant (new code):** the `lnk_hostabs` fixture made a relative link. The fixture self-check still passed, but the `lnk_hostabs`/`lnk_chain` refusal cases went red (8 FAIL), so the absolute-target rule is genuinely what those cases pin.

### N1 (nit): `--environment` may name a directory
`sbx_idx_contained` requires a directory for `--cwd` only; `--environment /src` or `/src/sub` passes the host check and the tool fails at read time (a recorded launch, exit non-zero). Spec4 does not require a regular file. Optional: `[ -f ]` for `--environment`, plus a case in test 22.

Open questions: none.
