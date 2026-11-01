# S11 (P0-10) measured results — vertical-slice extraction

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** C-1001 (P0, S11 / P0-10)

Source for S14's slice report. Every figure below was read from output of the runs named here
(`cpg/.ccg-work/vertical-slice/run1`, `run2`; gitignored). Contract: `docs/plans/connected-code-graph-p0-2.md`
S11 row and §4.4; bundle spec v1.3 + `-spec2` v1.3.

## What exists

| File | Role |
|---|---|
| `run_extract.sh` | Stage (`git archive`) → parse → content check → (JS) astgen → `extract.sc` → `normalize.py` → verify. Every sandbox step goes through `cpg/sandbox/offline-run.sh` (container backend, offline). `run_extract.sh --tag <t> [--units fcs,sp] [--commit <sha>]`. |
| `extract.sc` | The Joern extractor (runner-staged script form, `@AUDIT_ID@` output stem, no `//>` directive). Emits CPG facts incl. the per-method projection (`data`) and raw CDG (`controlledBy`) candidates. |
| `astgen.sc` | Runs jssrc2cpg's bundled `astgen` in the sandbox: Babel ASTs for JS/TS imports, type end lines and the qname re-derivation. |
| `normalize.py` | Host side: qname grammar, ids, ordinals, `git blame` (`lastChanged*`), G-4, writes through `ccg.bundle.write_bundle` (sorted, canonical, validated). |
| `verify_qnames.py` | Re-derives every symbol's qname/span from source alone (Python `ast`; JS/TS Babel AST) and compares; also checks Python param records and that each callsite's line holds its callee name. Shares no code with `normalize.py`. |
| `perfile_rule.py`, `check_audit.py`, `compare_bundles.py` | Per-file rule measurement (+ the jssrc2cpg exclusion rules, finding 3); `sandboxAuditIds` ↔ audit records (+ `extractScriptHash` = `sandbox.script.sha256`; `check_audit.py <bundle> <audit dir or sandbox.jsonl>`, path required); SL-6 diff. |
| `tests/test_s11.py` | 18 unit tests (`cd cpg && .venv/bin/python -m pytest spikes/vertical-slice/tests`). |

Pinned commit: `defca60ff16534aa12a74e35856de9be98bfc710` (repo HEAD when S11 ran). Units: `falkor-chat/server`
(python, `pythonsrc`) and `salesperson` (javascript, `jssrc`). Unit ids `0fb89974…` (= GV-1) and `c2341bca…`.
`extract.sc` sha256 `0be8bf82…` was equal to the sandbox record's `script.sha256` in both units (`check_audit.py`).

## Results

**Bundles validate.** Both bundles pass `ccg.bundle` on write and on a separate `read_bundle` re-read
(fcs 85,071 records; sp 2,775 records). Record counts (run1):

| type | fcs | sp | | type | fcs | sp |
|---|---|---|---|---|---|---|
| file | 107 | 41 | | valueSet | 26,907 | 415 |
| symbol | 4,686 | 257 | | constant | 184 | 5 |
| param | 6,517 | 125 | | data | 15,983 | 441 |
| ret | 1,903 | 222 | | controlledBy | 5,699 | 451 |
| cond | 3,816 | 218 | | decorator | 213 | 0 |
| callsite | 18,129 | 462 | | import | 926 | 137 |

Not emitted (zeros in `header.counts`): `fieldAccess` ([P-slice] "if the slice emits fields"; AC-1 needs none, so
**not exercised**), `testMarker`, `dependency`, `configEntry`, `launchFact`, `schemaEntry`, `secretSighting`.

**qname re-derivation (bundle spec §4.0) — no mismatch, no SL-7 candidate from the grammar.** `verify_qnames.py`,
both runs, both units: fcs 4,686 source symbols = 4,686 bundle symbols, all qnames equal, 0 span mismatches, 0 param
mismatches; sp 257 = 257, all equal. Golden vectors reproduce from the real bundles: GV-1 (unit id), GV-11
(`salesperson/src/api/client.ts::apiFetch`), GV-14 (`…/endpoints.ts::getHealth` and its `apiFetch` call-site id
`ac74017f…`). Sensitivity: renaming one qname and moving one `lineEnd` in a copy of the sp bundle made the verifier
report `onlyInSource=1 onlyInBundle=1 spanMismatch=1`, exit 1. Caveats, stated plainly: the verifier and
`normalize.py` implement the same spec text (one author), over different inputs (source parse vs CPG scope tree);
spans for Python symbols and TS type declarations are taken from the source parse (see findings 1–2), so for those
the span comparison is not independent.

**Per-file content rule** (every non-empty source file has ≥ 1 non-external `METHOD`), measured by
`perfile_rule.py` over `cpg-content-check.sc`'s `FILE_METHODS` lines:

| frontend | non-empty source files | FILE nodes | files with ≥ 1 method | rule over non-empty source files | rule over CPG files |
|---|---|---|---|---|---|
| `pysrc2cpg` (fcs) | 106 (+1 empty `.py`) | 108 | 106 of 106 | **holds** — but a naive per-FILE-node check false-fails on the `N/A` placeholder FILE node (0 methods) | holds |
| `jssrc2cpg` (sp) | 74 | 41 | 41 of 41 | **fails**: 33 of 74 non-empty source files are absent from the CPG (frontend default exclusions: 26 `*.test.ts(x)`, 2 `*.spec.ts`, and all 5 production files of `src/i18n/`; table under finding 3) | holds |

Recommendation (the check adopts per-file only where it holds): adopt per-file for **Python**, counting only FILE
nodes whose name is a source file (exclude `N/A`); the empty `tests/eval/__init__.py` also gets a `<module>`
method. For **JS/TS** adopt it **over the FILE nodes the CPG has** (0 failures) and keep the unit-level floor for
files the frontend never parses; a literal "every source file" rule would false-fail 33 files (28 by file-name suffix, 5 production files by directory name). Both units
passed the unit-level D15 check (`CONTENT … =ok`, METHOD_COUNT 6,254 and 232) before extraction; `run_extract.sh`
fails the build with a reason if a unit with ≥ 1 non-empty source file has no non-external method (driven with a
failing input in the gate fixes, see "Content-check failure branch" below).

**Timings** (wall seconds incl. docker start; run1 / run2; one host, 2 runs, no repeat statistics):

| stage | fcs | sp |
|---|---|---|
| parse (`joern-parse`) | 12.9 / 13.1 | 7.1 / 7.6 |
| content check | 10.1 / 9.7 | 5.6 / 6.8 |
| astgen | — | 10.4 / 10.7 |
| extract (`extract.sc`, whole script) | 12.7 / 14.4 | 8.5 / 8.6 |
| of which CPG load | 3.8 s / 4.0 s | 0.76 s / 0.69 s |
| of which projection (data+CDG) | 0.37 s / 0.40 s | 0.044 s / 0.042 s |
| normalize (host, incl. `git blame`) | 40.5 / 41.6 | 2.0 / 2.2 |
| verify (re-read, qnames, audit) | 31.5 / 31.3 | 1.2 / 1.2 |
| total | 107.9 / 110.2 | 34.9 / 37.1 |

For SL-5's "projection ≤ 2× parse": whole `extract.sc` wall ÷ parse wall = 0.99× (fcs, run1) and 1.20× (sp, run1);
the projection alone is ~3% (fcs) / ~0.6% (sp) of parse. Per-method projection-time histogram (run1): fcs 6,235
methods < 1 ms, 19 in 1–10 ms; sp 226 < 1 ms, 6 in 1–10 ms. The 0.7 MB/file density half of SL-5 is S13's.

**Determinism (SL-6).** run1 vs run2, final code: `compare_bundles.py` → "identical (85071 lines)" and
"identical (2775 lines) apart from builtAt/sandboxAuditIds"; the bodies (everything after line 1) are also
byte-identical by `cmp`. The facts files differ in nothing that reaches the bundle.

**G-4 (CDG self-edges and repeated pairs) — verified against the slice CPG.** Raw CDG edges mapped to points
(run1): fcs 48,022 edges → 2,751 self-edges, 26,476 repeated (point, cond) pairs, 13,096 edges whose point is a
synthetic call that is not a call site, **5,699 written** (1,530 of them cond→cond); sp 2,242 → 5 self, 1,765
repeated, 21 unmapped, **451 written** (125 cond→cond). The rule is needed, not hypothetical: with the self-edges
kept (`normalize.py --g4-mutant self`) both bundles are rejected by the validator ("point equals cond, a CDG
self-edge", fcs line 78236, sp line 2223); with repeats kept (`--g4-mutant dedup`) both are rejected ("duplicate
controlledBy id", §8.3). With G-4 applied both validate.

**Other `-spec2` §4.1 emitter items:** `nameNorm` by §1.2 (validator recomputes it; passes); `import.ordinal`
dense per file with repeated names de-duplicated; cond→cond `controlledBy` from the CDG (1,655 written);
ASCII-`IDENT` quoting of field keys (`quote_field`, unit-tested incl. `café`, `content-type`, `""`); tool output
names use `@AUDIT_ID@` (the facts file, the astgen dir and the CPG all carry the audit id stem).

**Slice facts present** (inputs for S12; not S12's verdict): sp — `getHealth`'s `apiFetch` call has `a0.path` =
template(`{ref: …/endpoints.ts::API_PREFIX}`, `"/health"`) and `a0.method` = `GET`; in `apiFetch`, `fetch` `a0`
← `buildUrl` result and `a1.method` ← `p0.method`; `buildUrl` `ret` ← `p0` (the four AC-1 `data` shapes of spec
§5.10, all `confirmed`/`intra-flow`). fcs — `app.py::create_app` `include_router` `kw.prefix` =
`{ref: …/storefront_api.py::API_PREFIX}` controlled by the `shop is not None` cond; the dev-router
`include_router` is controlled by the `dev_surface` cond; the three `/health` handlers carry `decorator` records
(`router.get`, `app.get`, `router.get`) with `a0` = `"/health"`; `API_PREFIX` constant = `"/shop/api"`.

**Mutation testing of the added tests.** 23 deliberate breakages of `normalize.py` / `verify_qnames.py` /
`compare_bundles.py` (anon numbering, `#N` start, accessor/dup key, overload stubs, decorator span, decorator and
default-arg scope, JS binding inference, own-name precedence, ASCII IDENT, field cap, NFC, casefold, G-4 self and
dedup, blame tie-break and recency, callee order, optional-call regex, `<init>` name, synthetic-call fallback,
exempt-field handling): all killed after the two fixes below. Two first survived (one *equivalent* — a guard made the mutation unreachable,
replaced by a real variant that was killed; one real — `casefold` vs `lower`, fixed with a `Straße` assertion).

## Findings for S14 / SL-7 accounting

No qname-grammar delta. Deltas and frontend facts the slice report should carry:

1. **`pysrc2cpg` `LINE_NUMBER_END` is unusable.** On the first verification run 4,286 of 4,686 fcs symbols had a
   `lineEnd` different from the source parse (typically +2, trailing blank/comment lines). Python spans therefore
   come from `ast` (`end_lineno`, decorators included in the start); names/nesting/qnames still come from the CPG.
2. **`jssrc2cpg` gives no end line for `TYPE_DECL`** (interfaces, type aliases, classes) — Joern's `TypeDecl` has no
   `lineNumberEnd`. TS type spans come from astgen's Babel AST, whose nested nodes carry `loc.start` only plus
   `end`/`start` UTF-16 indexes (the end line is derived from the index). Type symbols are therefore the one place
   the TS bundle depends on a second parser.
3. **Frontend file selection differs: `jssrc2cpg` drops 33 of the 74 non-empty source files, and not only tests.**
   (An earlier version of this note said all 33 were `*.test.ts(x)`; that was wrong: 26 are.) Cause, measured: the
   frontend's `AstGenRunner` carries default exclusions, read from the constant pool of
   `io/joern/jssrc2cpg/utils/AstGenRunner$.class` (jssrc2cpg 4.0.579, sandbox image) and not overridable by the
   invocation here (`jssrc2cpg.sh --help` offers only `--exclude`/`--exclude-regex` as additions; `run_extract.sh`
   passes none): rule **F** = file name matching `(conf|test|spec|[.-]min|\.d)\.(js|jsx|cjs|mjs|xsjs|xsjslib|ts|tsx)$`;
   rule **D** = any directory segment in {`node_modules`, `venv`, `docs`, `test`, `tests`, `e2e`, `e2e-beta`, `examples`,
   `cypress`, `jest-cache`, `eslint-rules`, `codemods`, `flow-typed`, `i18n`, `vendor`, `www`, `dist`, `build`}.
   Confirmed on a synthetic tree (23 files) through `joern-parse` and through `astgen.sc`: only `src/intl/a.ts`,
   `src/locales/l.ts`, `src/ok.ts`, `src/q.config.ts` were parsed; excluded: `src/i18n/a.ts`, `tests/t.ts`,
   `tests/e2e/y.spec.ts`, `e2e/z.ts`, `src/{docs,examples,www,build,vendor}/*.ts`, and by name `x.test.ts`,
   `x.spec.tsx`, `xtest.ts`, `test.ts`, `spec.ts`, `conf.ts`, `x.d.ts`, `x.min.ts`, `x.mock.ts`, `x.e2e.ts` (the
   `.mock`/`.e2e` ones are unexplained by rule F as read; recorded as observed). `config.ts`/`q.config.ts` are not
   matched. Folders `venv`, `test`, `e2e-beta`, `cypress`, `jest-cache`, `eslint-rules`, `codemods`, `flow-typed`
   come from the constant pool only (not individually exercised). `astgen` applies the same exclusions (its own
   output for the unit has 41 ASTs, none of `i18n`/tests). `perfile_rule.py` now carries both rules and predicts
   the absent set: `jsRulePredictionMismatches = []` over the 74 files (predicted-absent = observed-absent = 33).
   All 33, with the rule(s) that exclude each:

   | rule | n | files |
   |---|---|---|
   | F only (`*.test.ts(x)`) | 23 | `src/api/{dispatch.test.ts,hooks.test.tsx}`; `src/components/message/{Composer,DeadTurnNotice,MessageBubble,Transcript,TurnIndicator}.test.tsx`, `{composerNotice,optimistic,welcome}.test.ts`; `src/components/sheets/{BottomSheet,ResetControl}.test.tsx`; `src/layout/{Header,Shell}.test.tsx`; `src/session/storage.test.ts`; `src/views/{CartPanel,CatalogPanel,ChatView,OrderPanel,PresenterKeyScreen,PresenterResetAllControl,PresenterRoster,ProfilePanel}.test.tsx` |
   | F + D=`i18n` | 3 | `src/i18n/{LanguageChooser.test.tsx,format.test.ts,locales.test.ts}` |
   | D=`i18n` only (**production code**) | 5 | `src/i18n/{LanguageChooser.tsx,Provider.tsx,config.ts,format.ts,useLocale.ts}` |
   | F (`*.spec.ts`) + D=`tests`,`e2e` | 2 | `tests/e2e/{mobile-shell,presenter}.spec.ts` |

   Totals: rule F matches 28 (26 `*.test.ts(x)` + 2 `*.spec.ts`), rule D matches 10 (all 8 files under `src/i18n/`
   and the 2 e2e specs); 5 files are excluded by D alone. Consequence: the locale hook/provider/formatter that
   components call have no `file`/`symbol` in the sp bundle, so call sites into `src/i18n/` cannot carry a
   `targets`. Not read by S12's AC-1 chain (`api/client.ts`, `api/endpoints.ts`). The bundle spec §4.1 rule 4
   ("every code file has exactly one file-scope symbol") is therefore not met for the JS unit **by the frontend's
   file selection**, not by the emitter: SL-7 candidate 1 below.
4. **`pysrc2cpg` adds a `N/A` FILE node** with 0 methods (breaks a naive per-FILE check).
5. **Frontend-synthesized calls are not source calls (§5.6)**, dropped by "callee name never occurs in source at/after
   the call start": fcs 4,328 (`__iter__` 1,312, `__next__` 1,312, comprehension `append` 479 / `add` 62 /
   `update` 67, `import` 1,096); sp 252 (`require` 208 for imports, `__ecma.Array.factory` 43, `next` 1). A first
   version dropped two legitimate `onReset?.()` calls (optional-call syntax); fixed and unit-tested.
6. **Python decorators** are lowered to `name = deco(...)(METHOD_REF)`; the chain calls are excluded from call
   sites and become `decorator` records; keyword arguments are call arguments with `argumentIndex = -1` +
   `argumentName` (a filter on `argumentIndex >= 0` loses every `prefix=`; found and fixed).
7. **`pysrc2cpg` drops the `*`/`**` sigil** from parameter names; recovered from the source. TS has an implicit
   `this` parameter at index 0 (not a bundle param; every TS bundle index is Joern's − 1). Python `self`/`cls` is
   the `-1` receiver by the §5.4 rule.
8. **IMPORT nodes are duplicated** (every Python import appears twice) and the entity strings are inconsistent for
   relative imports, so `import` records are derived from the source (Python `ast`, Babel `ImportDeclaration`/
   `import()`), not from Joern's IMPORT nodes. Python `import a.b as c` loses the alias (`names: []` per §5.14).
9. **REACHING_DEF crosses closure boundaries** (a lambda's free variable reaches the enclosing method's parameter).
   The projection restricts defs to the projected method (spec §5.10 "same symbol"); unrestricted, the validator
   rejected the bundle. Defs through a closure variable are therefore not projected (counted in `dataDropped`).
10. **`<operators>.…` (plural) call names exist** (`assignmentOr`); `isOp` must test both prefixes.
11. **Expression-bodied arrow functions have no RETURN node in `jssrc2cpg`**: 14 `ret` records (ordinal 0) were
    synthesized at the function's start line (no `data` can target them). Python lambdas all have one.
12. **`cond` = CDG sources**, per spec §5.5 ("Joern's CDG sources decide"). In JS this includes lowered `for…of`
    iterator conditions (`!(_result_0 = _iterator_0.next()).done`); `complexity` is 1 + the count of a symbol's
    own `cond` records (an approximation of McCabe; spec's definition says "decision points in the symbol's own CFG"; measured gap: SL-7
    candidate 2 below).
13. **Empty file** `tests/eval/__init__.py`: `git blame` has no line, so `lastChanged*` falls back to the file's last
    commit (`git log -1`). Spec §5.3 does not say what an empty span gets; SL-7 candidate 3 below.
14. **`symbol.name` for a constructor** is the type's name (rule 10); `<anon>` always carries `#N` (so the first
    anonymous scope is `<anon>#1`, never bare `<anon>` — an early normalizer bug, caught by the verifier).
15. **`fingerprint`/`complexity`/`lastChanged*`**: not independently checked by S11's own scripts (only presence and
    the validator's shape rules). The S11 review re-derived `fingerprint` (all 4,943) and `lastChanged*` (240 sampled)
    with its own code and found 0 wrong; `complexity` is re-measured below (it is not the spec's definition).
16. **`header.tools` records no host Python version** (keys observed: `extractScript`, `extractScriptHash`, `frontend`,
    `frontendVersion`, `joern`, `normalizer`, `normalizerVersion`; seven keys). The bundle depends on the host's `ast`
    grammar/`end_lineno` (Python spans, §4.0 qname derivation); this run's host is Python 3.12.3. Not fixed: the
    `tools` block is written by `normalize.py`, outside the files this gate-fix pass may edit. To do before a second
    machine: add the Python `major.minor` to `normalizerVersion` or a `tools` field.

## SL-7 candidates for the S14 slice report (no spec edited)

1. **Spec §4.1 rule 4 vs the JS frontend's file selection.** Rule 4 says every code file has exactly one file-scope
   symbol; `jssrc2cpg` never parses 33 of sp's 74 non-empty source files (finding 3), 5 of them production code
   (`src/i18n/`). Ruling needed: either the emitter adds `file` + `<module>` records for source files the frontend
   skips (derivable from source alone, as §4.0 already does for qnames), or the spec states which files may be
   absent (and D15's per-file rule for JS is then "over the files the CPG has", as recommended above). Open question
   for the owner: is `src/i18n/` meant to be in the slice (AC-1 does not touch it)?
2. **`complexity` is `1 + count(own cond records)`, not CFG McCabe (§5.3).** Re-measured in this pass (run1 fcs
   bundle vs a stdlib-`ast` count written for this check: `if`/`while`/`for`/`async for`/`except`/`IfExp`, `BoolOp`
   operands − 1, each comprehension generator + its `if`s; nested function/lambda/class scopes excluded; `with`,
   `try`, `assert`, `match` not counted), over 4,046 function/method/accessor symbols matched by (start line, name):
   3,561 equal; bundle higher by 1 in 393, by 2 in 48, by 3 in 3 (444); bundle lower by 1: 24, 2: 6, 3: 7, 4: 2,
   5: 1, 6: 1 (41). Indicative only (the `ast` count has its own choices); the same order of magnitude as the
   review's 3,532 / 443 / 41 on 4,016. Ruling needed: define `complexity = 1 + count(cond)` in §5.3, or keep McCabe
   and say how it is measured.
3. **Empty-span `lastChangedAt`/`lastChangedCommit`, and the span of an empty file.** §5.3 is silent: `git blame`
   has no line for `tests/eval/__init__.py`. Implemented fallback `git log -1 <commit> -- <path>`
   (→ `dbd2cdff…`, 2026-08-16T09:56:22Z; the file has one commit). Same file: `file.lineCount` = 0 but the
   `<module>` symbol has `lineStart` 1, `lineEnd` 1 (the only symbol whose end exceeds its file's `lineCount`;
   fingerprint = SHA-256 of the empty string `e3b0c442…`). Ruling needed on both (suggested: fall back to the path's
   last commit; span `1..max(1, lineCount)`).

## Content-check failure branch (gate fix, driven)

The step-3 failure path in `run_extract.sh` was driven, not just coded. Method: a copy of `run_extract.sh` in the
scratchpad whose `git archive` line instead writes one unparseable `bad.ts` (`export const = = ;` / `(((`), run with
`--units sp`. Result: `joern-parse` exited 0, `cpg-content-check.sc` printed `METHOD_COUNT …=0` and
`CONTENT …=EMPTY` and exited 1, and the script failed with `unit sp: content check failed -- the parse produced a
CPG with no non-external METHOD although the unit has 1 non-empty javascript source files …; see …/contentcheck.txt`,
exit 1; no `ccg-sbx*` container or `ccg` volume was left (`docker ps -a`/`docker volume ls`). Only JS/TS input can
make an empty CPG here: the reviewer found that an unparseable `bad.py` still yields a `<module>` METHOD (check says
`ok`); I did not repeat the Python case. The message used to fire on **any** non-zero exit of the runner; it now
fires only when `contentcheck.txt` holds the check's own `CONTENT …=EMPTY` line. A second driven case (the same copy
with `CHECK_SC` pointing at a nonexistent file, so the runner refuses with exit 64) now reports
`content-check run exited 64 without a CONTENT=EMPTY verdict (runner or script failure, not an empty CPG)`, exit 1.

## G-4 "repeated" has two definitions (state which when quoting)

Both sum to the raw CDG fact count (facts file, `k:"cb"` records; run1):

| definition | fcs | sp |
|---|---|---|
| raw CDG edges | 48,022 | 2,242 |
| A, edge-level, classified in the order self → unmapped → repeated (the table in "G-4" above): self + repeated + unmapped + written | 2,751 + 26,476 + 13,096 + 5,699 = 48,022 | 5 + 1,765 + 21 + 451 = 2,242 |
| B, node-level, on the fact's `(pk, pid, cond)` before any mapping: self + repeated + distinct non-self pairs | 2,751 + 37,480 + 7,791 = 48,022 | 5 + 1,773 + 464 = 2,242 |

"Repeated" is 26,476 (fcs) under A, where edges to synthetic-call points are counted as *unmapped* first, and 37,480
under B, where it is raw − self − distinct non-self pairs and the synthetic-call points are still in the pair set.
All self-edges have `pk=c` (cond-to-cond). S14 must say which definition it quotes.

## Not covered / open issues

- `fieldAccess`, `testMarker`, `dependency`, `configEntry`, `launchFact`, `schemaEntry`: not emitted (not in the S11
  row's emitter items; `launchFact`/`configEntry` are [P-later]).
- `valueSet`: one record per (call-site, argument path) that resolves to a non-`unknown` element, plus
  `point-derived`; other `unknown`s (`runtime-built`, …) are not written. `recv` values are skipped. Constants are
  module/type-level single assignments with a resolvable value; `{ref}` is only produced for same-file constants
  and imported names that resolve to an in-repo file.
- The `data` projection is a purpose-built backward walk (identifiers via REACHING_DEF to parameters / assignment
  right-hand sides / call results; operators by operand union; object-literal blocks flattened to fields, cap 3).
  It is not validated for recall/precision against a golden set — only for validity and the AC-1 shapes.
- Two sandbox records per unit are not named in `header.sandboxAuditIds` by design: the content-check run (it
  produced no bundle input). The header lists parse, (JS) astgen, extract.
- F-13: nothing in S11 normalizes a `keyNorm`/`http` key; `grep -n 'keyNorm\|%2E\|percent'` over every file S11 added
  matches nothing, so no step order is encoded here.
- Run time was ~110 s (fcs) + ~35 s (sp) per full pipeline; no run neared the 20-minute limit. No container other
  than the runner's own `ccg-sbx-<auditId>` was touched; `~/.local/state/ccg/docker-config` untouched.
