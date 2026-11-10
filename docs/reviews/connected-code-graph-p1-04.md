# Connected code graph — P1-04 (Joern extractor port) review

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

Diff-scoped review of the uncommitted P1-04 delivery under `cpg/`: `extractors/joern/{__init__,run}.py`, `extract.sc`, `astgen.sc`,
`tests/test_joern_run.py`, `tests/slice/test_extract_equivalence.py` and the README Layout row. Baseline: the P1-04 row in
`docs/plans/connected-code-graph-p1.md`, its section (behaviours 1–5) in `docs/plans/connected-code-graph-p1-extraction.md`, the runner
grammar in `-egress-spec3.md` §7 and `-egress-spec4.md`, and the real `cpg/sandbox/offline-run.sh` and `forbidden-flags.txt`. Not reviewed:
`ccg/packrt/*`, `tests/test_packrt_*`, `cpg/sandbox/*` and `cpg/infra/*` (read only as the contract), `ccg/broker/*`. The 6 GB host was
shared, so I ran one JVM at a time through the runner only, with `CCG_AUDIT_DIR` and all work areas in the scratchpad. No `falkordb-dev`
contact, no tree-mutating git commands, mutants on copies.

**Verdict: approve with suggestions.** No blocker: the equivalence claim holds under every definition I tried, and the staging rule
partitions files correctly over 1,416 staged units in 150 random trees. One major (a silent stage divergence from the commit that
`git archive` causes, inherited from the slice) is worth fixing before P1-44 builds repositories outside this monorepo.

CPG: considered, not relevant — the code under review is new and uncommitted (no loaded graph reflects it), and the questions were runtime behaviour, settled by execution.

Run: `tests/test_joern_run.py` + `tests/slice/test_extract_equivalence.py` `-m "not live"` 71 passed; `-m slice` (incl. the 2 Docker
equivalence tests) 10 passed, **0 skipped**; `tests/test_joern_run.py -m live` 3 passed (the real-runner unit, JS unit, and the real
exit-0 empty CPG). The full 5,763-test component suite was not re-run.

## Findings

### M1 (major, verified) — `git archive` silently drops or rewrites files, so the stage is not the commit's tree

`run.py` `stage_unit` builds the stage with `git archive`, which honours the repo's `export-ignore` and `export-subst` attributes
(read from the tree-ish's own `.gitattributes`). Planted: `svc/tests export-ignore` plus `*.py export-subst` in a committed
`.gitattributes` made `git archive HEAD -- svc` omit `svc/tests/t.py` and replace `$Format:%H$` inside `v.py` with a hash;
`git --attr-source=<empty tree> archive` does not change that (git 2.43). `stage_unit` returns normally, `_count_sources` still passes,
and the bundle covers a file set (and text) different from `git ls-tree` at the commit that `builtCommit` names. This repo has no such
attributes, so the slice and the current system are unaffected; the manifest allows other repos. Nothing in the tests builds the
attribute case.
Fix: after extraction, compare the staged file set and blob ids against `git ls-tree -r -z <sha> -- <root>` minus the exclusions
(`git hash-object --stdin-paths` over the stage) and raise `StageError` on any difference. That catches both attributes; I did not run
this design, so the implementer should plant the same two-attribute tree as its test. The alternative is staging from `ls-tree` plus
`cat-file --batch`, which also removes the dependency on `git archive`'s attribute handling.

### m1 (minor, verified) — the CLI discovers units from the index, not from `--commit`

`run.main` calls `discover_units(a.repo_root, man, per_file=False)`, which lists `git ls-files` (the index), while staging reads
`--commit`. Planted: a staged-uncommitted `a/b` unit appears in the discovered set, and `descendant_roots(a)` returns `['a/b']`
against a commit where `a/b` does not exist; the reverse (a commit newer than the index) selects the wrong exclusion set and cannot
select a unit that exists only at the commit. `extract_unit` itself is correct, since the caller supplies `units`; the mismatch is the
"one command" in the Done line. `-p1-linking.md` §8 (8) already commissions `discover_units(commit=...)` for P1-20.
Fix: cite that in the CLI docstring and README row ("units come from the index until P1-20"), or refuse when `--commit` does not
resolve to the index's tree. Route to teco as a ruling.

### m2 (minor, verified) — a `rootPath` is a git pathspec, so glob characters over-stage

`git archive ... -- <root>` treats the root as a pathspec. Planted: root `a*` staged `ab/n.py` along with `a*/m.py` (`n=2` for a
1-file unit). `a[1]` happened to match literally. Fix: `git --literal-pathspecs archive ...`; I ran it on the same tree and it lists
only `a*/`.

### m3 (minor, verified) — resolved the commit twice

`extract_unit` calls `stage_unit` (which resolves `commit`) and then `_resolve_commit` again for `RawFacts.commit`. For a symbolic
ref (`HEAD`, a branch) that moves between the two calls, the facts are staged at one commit and labelled with another; P1-05 uses
`RawFacts.commit` for blame and `builtCommit`. Inferred from the code (race not provoked). Fix: resolve once at the top of
`extract_unit` and pass the sha down.

### m4 (minor, verified) — the SIGKILL fallback and the extension and symlink counting are unpinned

Mutants on a copy of the tree, all survived the 71 tests: `timeout-no-kill` (drop `p.kill(); p.communicate()` after the 60 s grace),
`count-pyi` (add `.pyi` to the Python set), `count-symlink-ok` (count symlinks as sources). The brief's point 6 is therefore a claim
with no test behind it. The claim itself is right: against the real runner, a stage with only `a.pyi`, and one with only `a.mts` and
`b.cts`, parse with exit 0 and `CONTENT=EMPTY` (Joern 4.0.579), so counting them would turn a failing unit into a silent R11. Consequence
worth a ruling: `discover_units` makes a unit from `.pyi`/`.mts`/`.cts` alone, and `stage_unit` then raises "no non-empty source file"
(loud, correct, but the message does not say that the unit's only files are of a kind Joern ignores).
Fix: a table-driven test with a stage of `{only .pyi, only .mts, only a symlinked .py, only an empty .py}` each expecting
`StageError`, written with literal expectations; a stub that ignores SIGTERM to pin the SIGKILL branch. Note that after SIGKILL the
runner never reaches `sbx_remove_own`, so its container can leak; the live SIGTERM path is sound (below), the fallback is not
removable from `run.py` (only the runner starts containers), so document it.

### m5 (minor, verified) — the "no forbidden flag" assertion cannot fail on the parse stage

`forbidden-flags.txt` has no entry for `joern-parse`, `pysrc2cpg` or `jssrc2cpg`, so `forbidden_hits(parse_argv, {...})` is `[]` by
construction, and the planted-flag control in `test_the_forbidden_flag_check_fires_on_a_planted_flag` covers only `joern --dep`.
Mutant: planting `--fetch-dependencies` on the parse argv is killed only by `test_the_exact_argv_of_each_stage`, not by the forbidden
check; against the real runner that argv is not refused (exit 1, the tool fails), so the runner does not backstop it either. For the
Joern script stages the control is real: `--dep` plants 7 failures and the real runner refuses it (exit 64). Fix: state in the test
that the parse stage is pinned by the exact-argv assertion, or add a control that plants a flag under a temporary entry for
`jssrc2cpg` and shows the check fires.

### m6 (minor, verified) — the comparator's unit tests exercise two record kinds

`canonical` is sound on real data (next section), but its unit tests mutate only kinds `m` and `d`. Mutants that skip `cb`, or skip
`t`/`r`/`p`/`cs`/`const`/`deco`/`f`/`c`, survive all 71 tests; only a real difference would have exposed them, and then both live
runs equal the oracle, so no harm today. Fix, per the coverage-probe rule: a parametrised test that plants one dropped record in each
kind of the oracle's own kind list (`m cs const d r p deco c cb t f`, read from the oracle, not from the comparator), so a future kind
that the comparator ignores turns it red.

### n1 (nit) — smaller points

- A relative symlink that leaves the unit (`a/l -> ../common`) fails the whole stage under the `data` filter (verified: `StageError`);
  absolute links fail the same way. The slice's `tar -x` extracted a dangling link. Loud, defensible, and no link exists in this repo
  (`git ls-files -s` shows 0 mode-120000 entries); say which is intended in the docstring.
- `_run` lets `FileNotFoundError` (a missing runner) escape as a raw `OSError`; `main` catches only `ExtractError`.
- `test_the_ported_extractor_reproduces_the_slices_raw_facts` calls `extract_unit` without `CCG_AUDIT_DIR`, so the live equivalence
  test appends to the real `cpg/.audit/sandbox.jsonl`; the unit-level live test sets it. Inferred from the code (I ran with an override).
- Files under the 0700 work directory (`*.cpg.bin`, `*.facts.jsonl`, logs, `contentcheck.txt`) are 0644 and `stage/` is 0755; the
  directory chain is what protects them (verified: `work`, `cpg` 0700). Intended, but the tests assert only the directories.

## The eight checks

1. **Equivalence — holds, re-derived independently.** I re-ran both units through `python -m extractors.joern.run` at the pinned commit
   and compared with my own script (recursive freeze into a `Counter`, no sort-and-join). Oracle sha256s match the pins (`de52bc68…`,
   `585b11df…`). Under four definitions, each unit equal: multiset of records excluding `stats`; set of distinct records
   (fcs 92,747 distinct and 38,536 duplicates on both sides; sp 3,304 and 1,779); multiset with `stats` counts kept; and raw-line
   multiset (the only differing line is the `stats` record, 1 each side). Record counts 131,284 and 5,084; kind counts equal; kind and
   key-set equal. The order differs (first differing index 194 and 86), so file order carries no information, as the docstring says.
   What the comparator drops cannot hide a difference: `stats` differs only in `loadMs` (3,980 vs 6,447), `projectionMs` and `totalMs`;
   the `histogram` buckets are timing buckets whose sum equals `methods` (6,254 and 232) on both sides, and the count keys are compared
   against literals in `SLICE`. A planted live mutant in a copy of `extract.sc` (drop every 97th `cb` emit while `cdgRaw` still counts
   it) left the stats counts equal and failed `got == want` (`5060 == 5083`), so the live tier can go red on real data.
   Count spread: fcs source files 107 (any `.py`) or 106 (non-empty), the test pins 106; sp 74 under three extension definitions
   (with and without `.mts`/`.cts`/`.d.ts`; none exist). `METHOD_COUNT` from the stored content check, `stats.methods` and the `m`
   records are 6,254 and 232.
2. **Staging — correct.** Segment boundary, same-language, fallback-over-fallback, and forced fallback-over-manifest are pinned and my
   probe agrees. My own check generated 150 random trees (manifest and fallback shapes, nested and sibling prefix names, empty
   files, TS beside Python) and, for every unit of discover_units, staged it and counted owners of each tracked `.py`/`.ts` file against
   an independent "nearest enclosing unit of that language" model: 0 mismatches over 1,416 units. Planted shapes: unicode and space
   in paths and exclusions, exec bit kept (0755), a link into an excluded directory (dangles, no error), a link inside an excluded
   directory (dropped), `a[1]`/`a*` roots (m2), links leaving the unit (n1). On the real golden's 12 units no unit has a same-language
   descendant, so P1-04 changes nothing today. Mutants killed: same-language filter dropped, segment boundary, empty and dotfile and
   large-file drops, `fully_trusted` filter, HEAD instead of the sha; survived: only a depth-4 file drop (the slice stages have none).
   M1 and m2 are the gaps.
3. **Empty-CPG refusal — right.** Real runner: an empty CPG exits 1 with `CONTENT <cpg>=EMPTY`; a missing CPG also exits 1 with no
   verdict line ("No projects loaded"). So the exit code alone cannot tell them apart, and requiring both for `EmptyCpgError` is correct;
   anything else non-zero is `RunnerError`, and exit 0 without `=ok` is a `RunnerError` too. This matches the script's header
   contract ("exit 1 if any CPG is EMPTY; consumers still read the CONTENT lines") and the plan's "refuses an empty CPG". The
   `empty-by-rc-only` and `ok-line-not-required` mutants are killed.
4. **Argv builders — good.** The exact argv per stage is pinned by literals and the live runs prove the real runner accepts each (all
   six audit records `refused:false`, `exitCode:0`, `offline`). Controls: see m5. The test's grammar regex is a reimplementation of
   `sbx_joern_grammar`, equal in effect; the mutants `mode-controlled`, `codebase-dropped`, `dep-planted-extract` are killed.
5. **`--codebase` on every stage — benign; accept.** The runner only records it (`codebases` in the audit record, nothing else reads
   it for a sandbox record); the six live records carry `["graphmind-ai-lab"]`. It deviates from "behaviour unchanged" but only in the
   audit record, and is an improvement for attribution; say so in the docstring or the delivery note.
6. **Source-extension sets — right, unpinned.** m4.
7. **Work-dir mode and timeouts — correct.** `_mkdir_private` checks, creates and re-checks, and a pre-existing 0755 or symlinked
   parent is refused (tests). Live: `_run` with a 6 s timeout on a real parse of the fcs stage raised after 6.9 s and left no
   container of mine (the one listed before and after belongs to another run), so SIGTERM reaches the runner's own trap. The SIGKILL
   path is m4.
8. **Reachability.** Every file and behaviour (1)–(5) of the P1-04 section is built; "one command" runs (used above). Behaviour (1)
   needed no regeneration (oracle present and pinned); there is no code path that regenerates it, and the plan asks only for the
   first-dispatch action. Open: m1 (the command's unit source), and the "`0 skipped` recorded" is a docstring instruction, not a
   recorded run result — for teco's delivery note.

## What's solid

The equivalence design (pinned oracle, literal counts, body-equality of the `.sc` scripts, a stage-tree comparison, and a Docker tier
that fails on real data); verdict-line parsing in the content check; the stub-runner harness with hand-written expected stage
listings; `descendant_roots` and the whole-segment rule; staged-tree equals committed-tree (not the work tree) and no `.git`; the
closed-world argv builders; the work-directory chain.

## Open questions

- m1: does teco want the CLI to carry the index-versus-commit caveat now, or wait for P1-20's `discover_units(commit=...)`?
- M1: should the stager keep `git archive` (plus the `ls-tree` tripwire) or move to `ls-tree` and `cat-file`?

## Proposed REVIEW.md lines

None. The recurring class here (an enumerating comparator or guard tested on two of its kinds) is already covered by the existing
guard-coverage line in root `REVIEW.md`.

## Pass 2

**Verdict: approve with suggestions.** M1 and m1–m6 are fixed as claimed, apart from the test for m3, which cannot fail. One new minor
(P2-1) comes from the M1 fix itself.

Re-run: `tests/test_joern_run.py` + equivalence `-m "not live"` 107 passed; `-m "slice or live"` 13 passed, 0 skipped (172 s,
`CCG_AUDIT_DIR` redirected). Mutants on a copy (24): 21 killed; the 3 survivors are discussed below.

- **M1: fixed.** `stage_unit` uses `git ls-tree -r -z --full-tree` plus `cat-file --batch` into a private spool tar under the `data`
  filter. Re-run of the two-attribute plant: `svc/tests/t.py` staged, `v.py` still `$Format:%H$`. Against `git archive|tar` on a plain
  tree (nested dirs, 0755, symlink, empty file) the stage is identical in content, mode and links. New probes, all correct: a newline,
  tab, quote, backslash, leading-dash and `--output=x.py` name, unicode, and a non-UTF-8 name (`bad\xff.py`), all staged; a gitlink
  skipped; a 150 MB random blob byte-identical (0.7 s, 325 MB peak RSS); mode 100755 kept; symlink escapes (relative out of the unit,
  absolute) fail with `StageError`. My independent nearest-enclosing-unit fuzz: 150 random trees, 1,367 staged units, 0 mismatches.
  Mutants killed: exec bit dropped, gitlink included, symlink as file, `-z` removed (40 tests), big-blob skip, depth-3 drop, segment
  boundary, same-language filter, `fully_trusted`.
- **m2: fixed; agree the "non-literal pathspecs" mutant is equivalent.** `git ls-tree` does not glob its pathspecs even without the
  option (`ls-tree -- 'cpg/ccg/uni*'` lists nothing while `units.py` exists), and the mutant survives; `--literal-pathspecs` is harmless
  defence. The `a*` probe now stages 1 file and the glob-root test is a behaviour test.
- **m3: fixed in the code, but the test cannot fail (P2-2).**
- **m4: fixed.** Mutants `timeout-no-kill`, `count-pyi`, `count-mts`, `count-symlink-ok`, `count-empty-ok` are all killed; a runner
  that ignores SIGTERM is SIGKILLed after `kill_grace`. The claim that the fallback cannot remove the runner's container is right
  (a SIGKILLed bash never reaches `sbx_remove_own`; `run.py` starts no container) and `_run`'s docstring and the test's docstring both
  say so.
- **m5: fixed.** The parse-stage control plants an entry for `jssrc2cpg` and shows the check fires and then goes quiet; the docstring
  states the bound.
- **m6: fixed.** `KINDS` is checked against the `"k" ->` literals of `extract.sc` and the oracle's kinds; per-kind dropped and changed
  tests kill `cb`, `t` and `deco` skip mutants and the set-not-multiset mutant.
- **m1: fixed** (CLI docstring, epilog and a test state the index caveat); not re-probed beyond reading the new tests.
- **Live tier:** 13 passed, 0 skipped.

### P2-1 (minor, verified) — `--full-tree` breaks a repository that is a subdirectory of its git work tree

`ccg.units.discover_units` runs `git -C <repo> ls-files`, which is relative to that directory, so a manifest repo whose `path` is a
subdirectory yields unit roots relative to it. `_tree_entries` now passes `--full-tree`, which makes the pathspec and the output
top-level-relative. Planted: top repo with `mono/svc/{pyproject.toml,m.py}` and `other/svc/o.py`, repo path `mono`: discovery
reports `svc`, `stage_unit` raises "no non-empty source file staged". (Inferred, not run: a top-level `svc` beside `mono/` would be staged in its place.) The mutant dropping `--full-tree` survives all 107 tests, so nothing pins it. With
`--full-tree` removed the same plant stages `m.py` and `pyproject.toml` (run on a copy). Fix: drop `--full-tree` and add that
subdirectory-repo test.

### P2-2 (minor, verified) — the "commit resolved once" test cannot fail

The stub advances HEAD during the parse stage, after both resolutions in the old and the new code, so `rf.commit == sha` holds in
either. The mutant that passes `commit` back into `stage_unit` survives. Fix: have the advance happen between staging and labelling
(for example wrap `run.stage_unit` in the test to commit before returning) and assert the label equals the sha that was staged.

### n1 nits, answered

Take now: the raw `OSError` from `_run` when the runner is missing (`main` catches only `ExtractError`; still unhandled, one `except`);
and `CCG_AUDIT_DIR` in `test_the_ported_extractor_reproduces_the_slices_raw_facts` (still absent, so the live tier appends to the real
`cpg/.audit/sandbox.jsonl`). Leave: file modes under the 0700 work directory. Also stale: the module docstring (`run.py` line 7) still
says `git archive`, and the P1-04 section of `docs/plans/connected-code-graph-p1-extraction.md` ("stage via `git archive`") and its
BF12 mention in P1-05 name the old mechanism — route the plan wording to `architect` through teco.
