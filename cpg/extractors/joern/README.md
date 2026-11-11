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
