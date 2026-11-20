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

## Name-linked calls and export constants (`extract.sc` + `normalize/assemble_bundle.py`, P1-05 5d)

Joern links a `DYNAMIC_DISPATCH` call to every method of that name, whatever the receiver, so a call's own attributes cannot tell a name
link from a typed resolution. `extract.sc` therefore tags each such `cs` fact with **`rt`**: for a member call `p.m(x)` the `typeFullName` of
the access's base (`ANY` when untyped); for a bare-identifier call `f(x)` the label of the declaration the identifier refers to
(`METHOD_PARAMETER_IN` or `LOCAL`). A static call, or a receiver of any other shape, carries no `rt`.

`assemble` keeps a Joern target `confirmed`/`resolved-call` only if it is type-backed (`type_backed`): a `LOCAL` bare call keeps its targets
(a same-module function, a nested def, an imported or aliased name, a constructor, a builtin; limit: a rebinding `g = f; g()` is `LOCAL`
too); a parameter backs nothing; a member call keeps a target whose `fullName` is `rt` + the frontend's separator (`.` python, `:`
javascript) + the member name; a fact without `rt` is never demoted. The targets dropped go to `report["nameLinked"]` (call-site id -> symbol
list, for `merge_scip`'s `name_linked`), counted in `report["nameLinkCounts"]`; a site left with none is `unresolved`/`possible`/`name-match`.
**The `self` rule** (python): a demoted `self.m()` whose receiver is the enclosing method's own receiver parameter is `confirmed` to the
enclosing class's own `m` when the **unit defines exactly one class member named `m`** (a method or a property accessor of any class,
counted over the whole symbol table, not over Joern's candidates), it is the enclosing class's, and it is among the site's candidates
(`report["selfResolvedSites"]` lists the ids). Any other definer, a subclass override or an in-unit test double included, keeps the site
`possible`/`name-match`, whether or not Joern's `in` still lists it (Joern may have narrowed the call to the base method while a subclass
is what runs). `cls`, `this`, a nested function, a staticmethod and an inherited method are not restored. Limits, reported not fixed: a
subclass in another unit, an instance attribute shadowing the method, and same-name classes unrelated by inheritance (a correct
confirmation is lost; class-aware retention needs base-class data). Probe notes and the rows: `tests/test_normalize_name_linked.py`;
JavaScript/TypeScript rows (`MEMBER_SEP` `:`, `ANY`, module-typed `this`, bare parameter, `LOCAL`): `tests/test_normalize_name_linked_js.py`.

**Typed receivers with a short type name stay demoted: a counted limit.** Joern gives `rt` as the bare annotation name (`Campaign`, `Fx`)
when the annotated class is imported but not resolved, so `target == rt + sep + name` fails; accepting a short `rt` equal to the target's
class name would be a second `confirmed`-producing rule and would wrongly confirm an annotation naming a same-named class of another unit.
The cost is status only (candidates are listed), and `merge_scip` `name_linked` re-confirms a site whose SCIP target agrees. The planted
row is `test_a_short_type_name_receiver_stays_demoted_as_a_counted_limit` in `tests/test_normalize_name_linked.py`.

Measured figures (population, per unit, the short-`rt` count) are in `docs/plans/connected-code-graph-p1-linking.md` P1-50 and the 5d
test report, not here.

**Export-block constants.** `jssrc2cpg` writes `export const X = v` as a `BLOCK` child of `:program` (`const X = v` and `exports.X = X`);
the module-constant scan reads the assignment children of such blocks (JS/TS, `Identifier` targets only), so an imported exported constant
resolves through a `ref` instead of an `importRef`. Not read: `export let A = 1, B = 2` and destructuring exports (nested blocks).

**Oracle overlay.** The S14a oracle and stored bundles are not regenerated: the slice tests overlay `tests/data/p1_05_5d_extras.json`
(`tests/overlay_5d.py`: the hand-listed `rt` table and export-constant facts) and `NORMALIZER_VERSION` `0.7` (the 5d fix round and the Python f-string constants, P1-50) has its one delta in
`tests/slice/test_normalize_equivalence.py`.

## Constant text of Python f-strings (`normalize/core.py:Resolver.element`, P1-50)

Joern writes the pieces of a Python f-string (`formatString`) as raw source text, never quoted (`tests/golden/fstring/probe.json`: `{{lit}}` and
`\n` arrive as written, a nested format spec is one `formattedValue`, an empty f-string has no pieces). A `lit` child of a `fmt` in a Python file
is therefore `{"const": <c>}`, not parsed as a literal; the normalizer does not merge adjacent pieces and keeps escapes as source text. Not
changed: JavaScript/TypeScript templates, `concat`, every other literal, the empty-piece filter, the depth cap. Limit: an implicitly concatenated
string that holds an f-string (`"p" f"{x}"`, a multi-line message) is one `stringExpressionList` in Joern's output, so its value stays unknown.
The independent reference is `tests/fstring_reference.py` (stdlib `ast`); the opt-in `CCG_FSTRING_LIVE=1` row of `tests/test_fstring_constants.py`
re-runs the probe through live Joern.

### Regenerating the recorded facts

Three artifacts record `extract.sc`'s output and are tied to the script by `sha256(extract.sc)`; a script edit turns their tests red until
they are regenerated and re-stamped (all runs bounded: `JAVA_OPTS=-Xmx1500m`, `timeout -s KILL 150`, one JVM at a time, no Docker):

- `tests/golden/projection/facts.{python,javascript}.jsonl` + `extract.sc.sha256`: stage the goldens' fixtures (`fixture_text` of each
  golden), `joern-parse`, `joern --script extract.sc`, keep kinds `m p r c cs d cb t` sorted as `test_live_extract_reproduces_the_stored_facts`
  compares them (`CCG_PROJ_LIVE=1` runs exactly that); then `sha256sum extract.sc > tests/golden/projection/extract.sc.sha256`.
- `.ccg-work/p1-09/fixed-{fcs,sp}.facts.jsonl` (gitignored): `joern --script extract.sc --param cpgFile=<slice cpg.bin under
  .ccg-work/vertical-slice/s14a/<tag>/cpg> --param out=<file> --param lang=python|javascript`, and write the hash to `<file>.sha256`.
  Since 5d these files carry `rt` and the export-block `const` facts (the equivalence tests overlay them, see above).
