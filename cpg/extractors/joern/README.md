# extractors/joern

The Joern extractor (`run.py` stage/parse/check/astgen/extract, `extract.sc` raw facts, `normalize/` bundle writer).
Only the P1-08 mechanism is recorded here.

## `jssrc2cpg` ignore override and `header.skippedFiles` (P1-08)

**Probe result (jssrc2cpg 4.0.579, 2026-11-10): no override exists, so the fallback holds.** The five production
files under `salesperson/src/i18n/` stay out of the CPG and are listed `frontend-ignore` (spec4 §1, R-3); the ten
empty-`targets` call sites (`formatCurrency`, `useLocale`, `formatDate`) stay unresolved.

- **CLI.** `jssrc2cpg --help` offers `--exclude`, `--exclude-regex`, `--enable-early-schema-checking`,
  `--enable-file-content`, `-o`: all can only add exclusions.
- **Config object.** `io.joern.jssrc2cpg.Config` has `defaultIgnoredFilesRegex`, `ignoredFilesRegex`, `ignoredFiles`
  (and withers), but no `AstGenRunner` bytecode reads `defaultIgnoredFilesRegex`; the default filter is
  `AstGenRunner.isIgnoredByDefault`, a private method with no config input. The ignored folder segments
  (`node_modules venv docs test tests e2e-beta examples cypress jest-cache eslint-rules codemods flow-typed i18n vendor
  dist build`) are the 18 regexes of `AstGenRunner$.AstGenDefaultIgnoreRegex` (private static, built with
  `makeConcatWithConstants`, so visible with `javap -v`, not `-c`). Separately `IgnoredFilesRegex` /
  `IgnoredTestsRegex` hold the file-name regexes (`.*[.-]spec\.js`, `.*test.*\.json`, `jest.config*`, ...) and a file that
  looks minified is rejected too. Re-probe on a `jssrc2cpg` upgrade by grepping those names.
- **Experiment** (3-file copy keeping paths: `src/routePaths.ts`, `src/i18n/format.ts`,
  `src/views/OrderPanel.test.tsx`; `JAVA_OPTS=-Xmx1500m`): CPG `FILE` nodes were `src/routePaths.ts` only.
- **Empty file** (2-file copy: a 0-byte `src/empty.ts` and a real `src/real.ts`; same caps): `FILE_METHODS` listed
  `src/empty.ts=1` (the `:program` method) and `src/real.ts=2`: **`jssrc2cpg` parses an empty source file and emits a
  `FILE` for it** (it gets a `file` record, spec4 §2 empty-file rule). So an empty file is never `frontend-ignore`; the
  stage walk skips empty files and symlinks (`run.is_staged_source`, shared with `_count_sources`), and `parsed_map` adds
  any parsed path the walk skipped.
- **Label.** `frontend-ignore` means "a source with no `file` record from the frontend": it also covers a file astgen
  failed to parse, not only a default-ignore match.
- **Consequences.** `run.py` has no override flag. The writer passes `override=False`. A staged-copy rename is
  not an escape (it changes `qname`s). Any other path under the folder list above (`docs/`, `examples/`, `dist/`,
  `build/`, `cypress/`...) is a frontend-ignored production file too: `skipped_files` lists whatever the frontend did
  not parse, it does not hard-code the i18n names. Re-run the probe on a `jssrc2cpg` upgrade.

**`skipped.py`.** `parsed_map(stage, unit_root, parsed_paths)` (walk: `candidate_sources`) builds `parsed`; `skipped_files(unit, parsed, override)`: `parsed` maps every candidate source path of the unit
(repo-relative POSIX) to whether the frontend gave it a `file`. Only `javascript` units have entries: `test`
(closed pattern, parsed or not; wins over `frontend-ignore`) or `frontend-ignore` (any other unparsed source).
Declaration files (`.d.ts`, `.d.mts`, `.d.cts`, `.pyi`), vendored paths, non-sources and paths outside `rootPath`
are listed in neither. `override=True` asserts the frontend parsed every non-test source; an unparsed one raises
`ValueError` rather than being listed. The caller drops the `file` record of every listed path and writes
`header.skippedFiles` only when the list is non-empty.

## The projection (`extract.sc`, P1-09)

`d` and `cb` records are the per-method projection; every other fact kind is the slice's, byte for byte (`tests/slice/test_extract_equivalence.py`).
Python and TS/JS only. Goldens: `tests/golden/projection/*.json` (`tests/test_projection_golden.py`); no-regression rule:
`tests/slice/test_projection_not_regressed.py`.

- **R1** `defsOf` follows a `REACHING_DEF` whose source is an earlier same-name identifier that is not an assignment target.
- **R2** a `new` lowering block (`tmp = alloc; tmp.<init>(..)` / `<operator>.new(tmp, ..)`; `tmp`) is defined by the constructor call.
- **R3** frontend-synthetic iterator-protocol calls are transparent: python `tmpN.__iter__()` / `__next__()` (receiver a temp, or the call
  assigned to one), javascript `_iterator_N.next()`. The call stays a def too (shipped records are kept); a user `next(it)`,
  `it.next()` or `it.__next__()` is not recognised. The list is closed; the source text is not available to the script.
- A lowering block's temp also depends on what is added to it (`append`/`add`/`update` of a comprehension or dict spread); a computed
  dict key flows into the whole object (`WILD`). Only the `d` projection sees these: `cs`/`deco` `args` keep the shipped positions.
- A call that is itself a CDG source also gets a `cb` record with `pk = cs` (never for its own self-edge).
- **4b synthetic-cond transparency** (python; `-p1-ml.md` §3.4): a `try`-lowering cond (a CDG-source statement directly in a try body, not an
  if/while/for condition nor a ternary/short-circuit operand) or a generator/comprehension iterator (`iteratorNonEmptyOrException` whose
  lowering is in an expression) is replaced, as the end of a `cb` edge, by the nearest real governing cond(s); none = edge dropped. A
  `for`-statement iterator is real. `stats.cbRewrites` lists `[method, pk, pid, syntheticCond, [replacements]]`; raw `cb` records otherwise
  keep their repeats and self-edges as before.
- **4f** a `for`-statement header cond (`iteratorNonEmptyOrException`, a WHILE lowering outside any expression) receives data from its
  iterable: the cond depends on what `tmp = <iterable>.__iter__()` (the call heading the lowering block) names, so the iterable's defs
  flow into it. A comprehension iterator does not (it is synthetic and stays out).
- `stats` gained `cbRewritten` (count) and `cbRewrites` (`[method, pk, pid, syntheticCond, [replacement conds]]`, the carve-out audited by
  `tests/slice/test_projection_not_regressed.py`); `cdgRaw`/`cdgSelf` still count raw CDG edges, `dataRaw` counts the (larger) `d` set.
- Not fixed: a call through a local alias (`write = repo.a if f else repo.b; write(x)`): Joern emits no `REACHING_DEF` into the callee
  identifier; a computed dict KEY is not reached either (the value is).

### Regenerating the recorded facts

Three artifacts record `extract.sc`'s output and are tied to the script by `sha256(extract.sc)`; a script edit turns their tests red until
they are regenerated and re-stamped (all runs bounded: `JAVA_OPTS=-Xmx1500m`, `timeout -s KILL 150`, one JVM at a time, no Docker):

- `tests/golden/projection/facts.{python,javascript}.jsonl` + `extract.sc.sha256`: stage the goldens' fixtures (`fixture_text` of each
  golden), `joern-parse`, `joern --script extract.sc`, keep kinds `m p r c cs d cb t` sorted as `test_live_extract_reproduces_the_stored_facts`
  compares them (`CCG_PROJ_LIVE=1` runs exactly that); then `sha256sum extract.sc > tests/golden/projection/extract.sc.sha256`.
- `.ccg-work/p1-09/fixed-{fcs,sp}.facts.jsonl` (gitignored): `joern --script extract.sc --param cpgFile=<slice cpg.bin under
  .ccg-work/vertical-slice/s14a/<tag>/cpg> --param out=<file> --param lang=python|javascript`, and write the hash to `<file>.sha256`.
