# Review: C-1001 P1-08 (`skippedFiles`, jssrc2cpg override probe)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

Reviewed (uncommitted, graph-dba): `cpg/extractors/joern/skipped.py`, `cpg/extractors/joern/README.md`,
`cpg/tests/test_skipped_files.py`, `cpg/.ccg-work/p1-08/REPORT.md`. Baseline: `docs/plans/connected-code-graph-p1-extraction.md`
§ P1-08 (as revised: `skipped_files(unit, stage, parsed, override)`), `docs/plans/connected-code-graph-bundle-spec4.md` §1,
the S10 validator `ccg/bundle.py::_Semantics._skipped_files`, committed `assemble_bundle.py` (4cd9299f). `run.py` and
`assemble_bundle.py` are untouched (verified: absent from `git status`). Root `REVIEW.md` applied; no `cpg/REVIEW.md`.

**Verdict: approve with suggestions.** The rule logic matches spec4 §1 and the validator, and the probe conclusion
holds. Two things need a `teco` ruling before the wiring step: the signature no longer matches the revised plan (F1) and the
file walk, the riskiest part, lives outside the unit and outside any test (F1, F2).

CPG: considered, not relevant — the unit is a new pure function plus a probe of an external JVM frontend; no existing call graph to query.

Executed: pytest (165 passed, `ulimit -v 1500000`, `timeout -s KILL 100`); 10 mutants of my own (below); `jssrc2cpg.sh --help`
(one JVM, `-Xmx1200m`, `timeout -s KILL 60`, no leftover java); `javap -p -c`/`-v` on the jssrc2cpg 4.0.579 jar;
read-only `git ls-files` counts. Read only: spec4, the validator, `run.py` staging, `assemble_bundle.py`. Not re-run: the
delegate's 3-file Joern experiment and its 24 mutants (its report's `exp1_default.check` was read).

## Findings

### F1 (major, verified by reading) Signature differs from the revised plan, and the candidate walk is untested

The plan now reads `skipped_files(unit, stage, parsed, override)`: `stage` is `RawFacts.stage` (the source listing),
`parsed` is `core.lang_files(facts, lang)`, `assemble(..., skipped=None)` takes the list and `build_bundle` computes it from
`raw`. The delivered function is `skipped_files(unit, parsed: Mapping[str, bool], override)`: the walk of the stage, the
stage-relative to repo-relative mapping (`B.P`) and the parsed/unparsed join all move into "wiring note 1", i.e. into
`assemble_bundle.py` with no test of their own. Spec4 §1 itself says completeness cannot be validated, so that walk is
exactly where silent omissions would hide (report's own residual risk 1).
Fix (teco to choose, architect to align the plan): either (a) implement the plan's 4-arg form: `skipped_files` walks
`stage`, applies `unit.rootPath` mapping itself, takes `parsed` as the stage-relative names, and the coverage table test
builds a `tmp_path` stage; or (b) keep the pure 3-arg form but add a tested helper in `skipped.py`
(`candidates(stage, unit, parsed_names) -> dict[str, bool]`) and amend the plan. Either way the walk gets a test that
plants a file the walk must find in a nested directory and one it must not.
The Mapping form is derivable at the real call site (`build_bundle` has `unit`, `raw.stage`, `load_facts(raw.facts)`); the delegate's
decision is sound, it is just not where the plan put it.

### F2 (major, inferred from code, not reproduced) Walk rules the notes omit: empty files and symlinks

`run.py::_count_sources` (line ~100) treats only non-empty, non-symlink files as sources, and the spec4 baseline is "74 non-empty
`salesperson/` sources". Wiring note 1 says "every file under `stage`". An empty `.ts`/`.js` file (or a symlink; staging
keeps in-tree symlinks) would be absent from the CPG, get no `file` record, and be listed `frontend-ignore` although it
is no source file by the stage's own definition; S10 cannot catch it. I did not verify whether jssrc2cpg emits a FILE
node for an empty file. No current instance: `git ls-files salesperson` has no empty `.ts/.tsx/.js`, and `falkor-chat/server` has no
JS. Fix: the walk (F1) applies the same predicate as `_count_sources` (share it, do not copy), with a planted empty file
and a planted symlink in the test. Related nit: a file astgen fails to parse is also labelled `frontend-ignore`; say so
in the README so the label is read as "no file record, frontend", not "matched a default ignore".

### F3 (minor, verified by mutation) Extension axis is untested for non-test sources

Mutant G7 (`elif not parsed[path] or base.endswith(".tsx")`: a parsed production `.tsx` is wrongly listed) and G8
(unparsed production `.cjs/.mjs/.jsx/.cts` silently not listed) both survive 165 passes. Parsed/unparsed production
cells only use `src/app.ts` and `src/i18n/format.ts`, while the test-pattern tests are parametrized over all 8
extensions. Fix: parametrize the production parsed and unparsed cells over `JS_TEST_EXTENSIONS` (the coverage probe's
axes should include extension where the rule enumerates them, spec3 §3.3).

### F4 (minor, verified with `javap -v`) README misattributes the folder list; conclusion unaffected

README says `isIgnoredByDefault` rejects on `IgnoredFilesRegex` and that "the constant pool lists the segments
`node_modules ... i18n ...`". In 4.0.579 the folder segments (`node_modules\u0001.*`, `venv`, `flow-typed`, `i18n`, ...) are
the 18 regexes of `AstGenRunner$.AstGenDefaultIgnoreRegex` (private static, built with `makeConcatWithConstants` and so
visible only with `javap -v`, not `-c`), consumed through `SourceFiles.determine(..., ignoredDefaultRegex = Some(...))` in
`jsFiles/vueFiles/ejsFiles`. `IgnoredFilesRegex`/`IgnoredTestsRegex` are the separate `.*[.-]spec\.js`, `.*test.*\.json`
list used by `isIgnoredByDefault`. Fix: correct the two names in `extractors/joern/README.md`; it matters for the next
Joern-upgrade re-probe, which will grep by name.

### F5 (minor) Wiring notes inconsistent with the revised plan and with `assemble_bundle.py`

Compared with plan § P1-08 and the committed file: (1) notes put the call inside `assemble(...)` and rebuild a `Unit` from
`B.*`; the plan computes it in `build_bundle` (which already holds `unit`, `raw`) and passes `skipped=` into `assemble`.
(2) Note 3 "drop listed paths from `files`" omits that `files` entries are stage-relative while the list is repo-relative
(`B.P`); dropping by the wrong key removes nothing, silently. `files` feeds the file records, `blames` (line 177), the
loop at line 310 and `report["filesParsed"]`, so one filter at line 140 covers them (read, not run). (3) Missing from the
notes: bump `NORMALIZER_VERSION`, register the delta in `tests/slice/test_normalize_equivalence.py` (the plan's 33 =
28 `test` + 5 `frontend-ignore` for `salesperson`), and the wiring tests in `test_normalize_assemble.py`. (4) Order: 5c and
P1-06 edit the same function first, so line numbers in the notes (~141) will move; cite the anchor
(`files = [f for f in lang_files(...)`) rather than a line.

### F6 (nit) `override=True` raising `ValueError`

Ruling: accept. A non-test source unparsed under an asserted override has no valid answer (`frontend-ignore` would
be false, silence would violate "no file record, so listed"). It is a third table outcome, which spec4's "one reason or
not at all" wording does not state; ask architect to add the clause when the plan is aligned. Production always passes
`False`, so the guard is reached only by tests; keep it, but do not add a run.py flag for it. The validator-agreement tests build
`_Semantics` with `object.__new__` and take `vendored` from `skipped.DEFAULT_VENDORED`: acceptable, the real validator
logic is exercised, only the vendored set is circular (and F-free: `test_the_vendored_set_is_the_writers` pins it).

## Judgments asked for

**(1) Conformance to spec4 §1: holds.** Executed (165 tests) and read: test pattern (basename `.test/.spec` x 8 extensions,
segment `e2e`/`__tests__` strictly below root) applies parsed or not; `test` wins over `frontend-ignore`; `.d.ts`,
`.d.mts`, `.d.cts`, `.pyi` and vendored paths are listed in neither; sort key is UTF-8 bytes. The validator accepted the produced list and rejected a
listed path that kept a `file` record (both agreement tests pass; they call `_Semantics._skipped_files`). Reading
validator vs function: same prefix rule, same vendored rule (`dirs[:-1]`), same pattern, same `reason` iff-test check. The
72-cell table is hand-written with a completeness assertion, and expected values are not read from the module.
`.d.mts/.d.cts` as declarations: ruling accept (spec names `.d.ts`; same kind; consistent whichever way the frontend treats them since neither list is involved).
Only `javascript` returns entries: matches spec4 ("JS/TS units only", no Python frontend-ignore in v1).

**(2) Probe claim: supported for the CLI/Config surface.** Re-ran `jssrc2cpg.sh --help`: exactly `--exclude`,
`--exclude-regex`, `--enable-early-schema-checking`, `--enable-file-content`, `-o`, `--help`. `javap` of `AstGenRunner` and
`AstGenRunner$`: zero references to `defaultIgnoredFilesRegex`; `Config.ignoredFilesRegex`/`ignoredFiles` are only passed on as
exclusions (`--exclude-regex` to native astgen; as extra ignore args to `SourceFiles.determine`); `AstGenDefaultIgnoreRegex` is
a private static with no Config input. So no flag, no Config field and no `--exclude-regex` interplay can re-include `i18n/`. The
remaining route, a programmatic `withDefaultIgnoredFilesRegex(Nil)`, would not help either (that field is a different, unused
list) and a custom driver cannot alter a private static. A staged-copy rename is excluded by the plan (qnames).
Not checked: whether the folder regexes match the path relative to the input dir or absolute (would only matter for a
different input-dir choice, already ruled out). The delegate's experiment is reported, not re-run; consistent with the bytecode.

**(3) Decisions.** `parsed` as `Mapping[str, bool]`: derivable, but see F1/F2. `override` raising: F6. `.d.mts/.d.cts`: accept.
JS-only: accept. Mutants: 22 killed / 2 equivalent is credible by reading: M07 (`.pyi` unreachable behind the non-JS early return)
and M12b (UTF-8 byte order equals code point order) are both equivalent. My own mutants contradict "0 live survivors" only for the
extension axis (F3).

| mutant (mine, fresh scratch copy each) | result |
|---|---|
| G1 ARGUMENT: `_is_test(dirs[1:], base)` (drops the first segment below root) | killed (3) |
| G2 `e2e`/`__tests__` only counted above the immediate parent | killed (8) |
| G3 sort by `lower()` instead of UTF-8 bytes | killed (1) |
| G4 vendored check skips the last directory | killed (12) |
| G5 ARGUMENT: `_is_source(base, "python")` (language hard-coded) | killed (58) |
| G6 ARGUMENT: `_below_root(path, rootPath.split("/")[0])` | killed (1) |
| G7 parsed production `.tsx` listed | **survives** (F3) |
| G8 unparsed production `.cjs/.mjs/.jsx/.cts` not listed | **survives** (F3) |
| G9 test listed only if parsed or `.ts/.tsx` | killed (12) |
| G10 extra declaration suffix `.d.tsx` | survives, equivalent (not a real TS suffix) |

**(4) Risk statements.** Entries silently missing if the stage walk or `B.P` mapping is off: confirmed as the real risk
(F1, F2, F5). Recall on `falkor-chat/server`: `git ls-files falkor-chat/server` is 126 files and none is JS/TS, so the
unit is Python and returns `[]`: no delta; the plan's "gains whatever the coverage probe lists" resolves to nothing.
`salesperson`: 74 JS/TS sources (excluding `.d.*` and vendored); 28 match the test pattern under both definitions
(basename-only and basename-plus-`e2e`/`__tests__` segments; the two `tests/e2e/*.spec.ts` are also `.spec.`), 5 are
`i18n/` production files, so 33 absent matches the plan and the frontend already drops all 28. A parsed-but-test file (the report's recall
change) has zero instances in either stored slice, so the stored-bundle delta is header-only. Bound: counts are at HEAD, the
slice's stored commit is `defca60f`; ran `git ls-files`, not a checkout.

**(5) Wiring notes: incomplete.** See F5; also F2 for the walk predicate.

## What's solid

- Rule logic is small, total and matches the validator line by line; one place decides each reason.
- The table test is written independently of the module, has a completeness assertion, and a first-red record.
- The mutation work is real (24 runs, scratch copies, module path printed); the two equivalence arguments hold.
- Honest probe: negative result recorded with its limits (the REPL classpath failure is reported, not hidden), `run.py` left
  unchanged, no speculative flag.

## Open questions

1. `teco`: 4-arg plan form or 3-arg delivered form plus a tested walk helper (F1)? The plan and function must say the same thing before the wiring step.
2. Does jssrc2cpg emit a FILE node for an empty source file (F2)? One bounded run on a 2-file copy would settle it; not run here (no Joern analyses).

## Route

- `teco` -> `architect`: align plan § P1-08 with the chosen signature and add the "override with an unparsed non-test source is a contradiction" clause (F1, F6).
- `teco` -> `graph-dba` (follow-up, small): F2 walk predicate and its test, F3 extension parametrization, F4 README names, F5 notes rewrite.
- No `REVIEW.md` additions proposed.

## Pass 2

Re-gate of the graph-dba's round 2 (`cpg/.ccg-work/p1-08/REPORT.md`). Executed: `tests/test_skipped_files.py` + `tests/test_joern_run.py`
248 passed, 3 deselected (`ulimit -v 1500000`, `timeout -s KILL 100`); 9 mutants in fresh scratch copies under `cpg/.ccg-work/gate-p1-08/m/`; no Joern run.
Read: `git diff cpg/extractors/joern/run.py`, `skipped.py`, README, `exp3.check`, the new wiring notes.

**Verdict: approve with suggestions.** Nothing blocks the wiring step; two minors below.

Dispositions:
- **F1** fixed (variant b): 3-arg `skipped_files` kept, `candidate_sources` and `parsed_map` added and tested with a `tmp_path` stage (nested dir, non-source, `.cjs`, non-`.` root); the plan's 4-arg text must still be aligned by the architect (see N1).
- **F2** fixed: `is_staged_source` is one predicate; empty file and symlink planted in the walk tests. The empty-file question is settled by reading `exp3.check` (`src/empty.ts=1`, `src/real.ts=2`, `METHOD_COUNT=3`): jssrc2cpg emits a FILE for an empty file, so the walk skipping it is correct and `parsed_map` adding parsed-but-unwalked paths covers a parsed empty `*.test.ts`. Not re-run (as instructed).
- **F3** fixed: G7 killed (6 failed), G8 killed (10 failed) on re-run.
- **F4** fixed: README names `AstGenRunner$.AstGenDefaultIgnoreRegex` and `IgnoredFilesRegex`/`IgnoredTestsRegex` correctly; `frontend-ignore` is documented as "no frontend `file`".
- **F5** fixed: notes now compute in `build_bundle`, pass `skipped=`, name the stage-relative vs repo-relative trap, list the version bump, delta and wiring tests, and use symbol anchors. Consistent with committed `assemble_bundle.py` (`files = [... lang_files ...]`, `build_bundle` holds `unit`/`raw`).
- **F6** unchanged, accepted.

`run.py` edit is behaviour-preserving (read the diff): old condition `p.suffix in extensions and not p.is_symlink() and p.stat().st_size > 0`;
new `p.suffix in extensions and is_staged_source(p)` with `not p.is_symlink() and p.stat().st_size > 0`. Same operands, same
order, so a dangling symlink still short-circuits before `stat()`. Mutant H3 (reordered, `stat()` first) is killed by the existing suite (5 failed).
Identity: `skipped.is_staged_source` is a module-level import of `run.is_staged_source`, both callers look the name up in their own
module globals, and `test_the_walk_uses_the_runners_predicate_object` asserts `is` (executed, passes); a local copy in `skipped.py` would redden it.

| new mutant (fresh copy) | result |
|---|---|
| G7 / G8 re-run (extension axis) | killed (6 / 10) |
| H1 ARGUMENT: `candidate_sources` root prefix uses only the last segment of a multi-segment `unit_root` | killed (5) |
| H2 ARGUMENT: `parsed_map` marks a parsed path True only if the walk found it | killed (6) |
| H3 predicate `stat()` before `is_symlink()` (run.py) | killed (5) |
| H4 predicate allows empty files (`>= 0`, run.py) | killed (11) |

New findings:
- **N1 (minor, read)** `skipped.py` imports `extractors.joern.run` at module top (verified: importing `skipped` loads `extractors.joern.run` and
  `ccg.atrest`). `build_bundle` imports `EXTRACT_SC` from `run` inside the function with the comment "the runner is not needed to normalize".
  Wiring `skipped` into `assemble_bundle.py` with a top-level import would undo that. Either import `skipped` inside `build_bundle`
  too, or move `is_staged_source` to a neutral module that both import (keeping the identity test). Put the choice in the wiring step's row.
- **N2 (nit)** REPORT residual-risk bullet 2 still says a recall change on `falkor-chat/server` is possible; the server has no JS/TS (Pass 1, finding 4), so the
  stored-bundle delta is header-only. Also a parsed-free empty file under a frontend-ignored folder is neither walked nor parsed, hence in no list ("not found" instead of
  "not mapped"): harmless, spec4's "non-empty sources" baseline; say so in the README if wanted.

Route: `teco` -> `architect` (plan § P1-08 text still names the 4-arg form: align to `skipped_files(unit, parsed, override)` + `candidate_sources`/`parsed_map`, add the override-contradiction clause); N1 into the wiring step's brief. No `REVIEW.md` additions.

## Pass 3 (wiring)

Gate of the uncommitted wiring (`git diff` of `assemble_bundle.py`, `tests/test_normalize_assemble.py`, `tests/slice/test_normalize_equivalence.py`).
Executed: `test_normalize_assemble.py` + `test_skipped_files.py` 376 passed (`ulimit -v 1500000`); slice file `-k "hand_listed or lambda_decorator_delta or last_delta or version"`
3 passed (`ulimit -v 2500000`; the Joern-running slice tests were not run); a direct comparison of `Builder.P` against the inlined rule
(roots `.`, `web`, `a/b`; names `x.ts`, `src/deep/y.tsx`: identical); a script classifying the 33 paths from the stage; 8 mutants in fresh copies. Read: all other per-file record sources.

**Verdict: approve with suggestions.** No blocker. One minor test gap (W1).

(1) Wiring, verified:
- `assemble(..., skipped=None)` drops by `B.P(f) not in skip_paths` (repo-relative vs stage-relative, non-root unit tested); `header["skippedFiles"]` only when non-empty; `report["skippedFiles"]` always set; `NORMALIZER_VERSION` "0.4" equals the last delta.
- `build_bundle` imports `skipped` inside the function (N1 fixed: disposition fixed), computes for `javascript` units only, `override=False`.
- Record sources: read every `facts[...]` loop (`c`, `p` via `scopes`, `r`, `cs`, `const`, `d`, `cb`): each goes through `m2s`/`scopes`/`files`, all filtered by `fileset`; a call whose target sits in a dropped file loses that target (`m2s.get(t["id"])` is `None`, so it becomes unresolved/external, never a dangling `symbolId`), imports and blame loops iterate `files`. The tests assert absence for file, symbol, cond, ret and constant of a parsed `e2e/x.ts`; callsite, data, decorator, `ownedRegions` and import rest on the same `m2s`/`files` filter by reading, not by a planted record. Acceptable; plant a `d`/`cb`/`cs` fact in `extra` if you want it executed.
- The inlined `repo_rel` lambda is textually `Builder.P`'s rule (verified equal on the cases above), but is a copy; nit.

(2) Delta `p1-08-skipped-files` "0.4": independently verified by execution against `.ccg-work/vertical-slice/s14a`: staged JS/TS sources 74 (non-empty, regular),
stored `file` records 41, `stored <= staged`, difference 33; I classified each by the spec4 §1 pattern: 28 `test` (all `*.test.*`/`*.spec.*`, two under `tests/e2e/`), 5 `frontend-ignore` (`src/i18n/{LanguageChooser,Provider,config,format,useLocale}`);
my list equals `P1_08_SKIPPED_SP` exactly, in byte order. No `.d.ts` or vendored file in that stage, so the "in neither" rule is not exercised by the slice (it is by the unit tests). falkor-chat/server: delta returns 0, header unchanged (test asserts).
`test_the_stored_files_are_the_staged_sources_less_the_hand_listed_33` does not call `skipped_files`; it re-implements the stage predicate and compares to stored paths, so it is independent. The `last delta == constant` assertion (line ~428) ran and passes.

(3) Mutants (fresh copies, `assemble` + `skipped` tests):
| mutant | result |
|---|---|
| K1 ARGUMENT: drop by `f` instead of `B.P(f)` | killed (5) |
| K2 ARGUMENT: `repo_rel` without the unit-root prefix | killed (3) |
| K3 ARGUMENT: `parsed_map` marks parsed paths False | killed (4) |
| K4 header written even when empty | killed (7 failed, 51 errors) |
| K5 skipped computed for python instead of javascript | killed (4) |
| K6 last entry dropped from the list | killed (1) |
| K8 report count wrong | killed (4) |
| **K7** `repo_rel` always prefixes `unit.rootPath` (wrong for a `.` unit) | **survives** |

- **W1 (minor, executed)** K7 survives: no test builds a root-`.` JS unit through `build_bundle`. For such a unit the mutant would produce `./src/a.ts` keys, so parsed files look unparsed and get false `frontend-ignore` entries with nothing dropped. Real risk for any repo whose JS unit is the repository root. Fix: one `build_bundle` test with a `.` root unit (parsed `src/a.ts`, staged `e2e/x.ts`), expecting header `src/...` paths; or have `build_bundle` call a shared rebasing function used by `Builder.P` so the copy cannot drift.

Route: `tdd-engineer` for W1 (small). `teco`: plan § P1-08 text still needs the architect's alignment noted in Pass 2 (3-arg form, `parsed_map`, override clause), then the feature-close spec check. No `REVIEW.md` additions.
