# Connected code graph — fact-bundle spec, S15 amendments (v1 freeze)

> **Status:** active · **Owner:** `architect` · **Tracks:** C-1001 (M10) · **Supersedes:** `docs/plans/connected-code-graph-bundle-spec3.md` · **Extends:** `docs/plans/connected-code-graph.md` · **Extended by:** `docs/plans/connected-code-graph-bundle-spec5.md`, `docs/plans/connected-code-graph-bundle-spec6.md` · **Reviews:** `docs/reviews/connected-code-graph-p0.md` (Pass 4, Pass 5)

This is the successor `docs/plans/connected-code-graph-bundle-spec3.md` §1 requires when the slice
forces an SL-7 delta (`docs/plans/connected-code-graph-p0-2.md` §2 D5). It states only the seven
deltas of `docs/reviews/connected-code-graph-p0.md` Pass 4 ("`-spec4` content"). **Exactly one
schema change: the optional `header.skippedFiles` (§1).** No other field, type or id changes, and
`bundleSchemaVersion` stays `"1.0"`: the freeze takes effect on this document's acceptance (§0), so
the field is part of `"1.0"` from its first frozen byte and no `"1.1"` exists. A bundle without the
field stays valid, so the stored slice bundles are unaffected.

CPG: considered, not relevant — a spec amendment; no loaded `cpg_*` graph indexes `cpg/`, and every fact below was read from the slice report, the review and `cpg/spikes/vertical-slice/normalize.py`.

## 0. Reading rule and freeze

- **Base text.** `-spec3` is not carried forward. Its text, section numbers and golden vectors
  are the base of schema v1 and are read together with this document; a sentence below that names
  a `-spec3` section replaces that sentence and nothing else. `-spec3` stays on disk as
  `superseded` (as `-spec`/`-spec2` did) because S10's tests parse its text
  (`cpg/tests/test_ids.py`, `cpg/tests/test_bundle.py`).
- **Freeze.** Schema v1 and graph model v1 (the §9 mapping, S17 rulings excepted) freeze as
  `-spec3` + this document, **when this document is accepted and the §4 schema/validator unit has
  landed** (with or before acceptance). The freeze record cites that commit, so "frozen v1" names a
  byte-exact schema. The [P-later] and [P-slice]-unexercised items freeze untested, as listed in
  Pass 4 "Frozen, untested" (cited, not restated), plus two from this document: `skippedFiles`
  (first writer P1) and the empty-file/`ownedRegions` fallback (§2, P2).

## 1. Rule 4 and the file population (replaces §4.1 rule 4's first sentence; adds to §2.2, §5.2, §7)

**Rule 4, first sentence, reads:** "Every code file **the unit's extractor parsed** has exactly one
file-scope symbol: `<module>` (code), `<script>` (shell) or `<document>` (Markdown)." The rest of
rule 4 is unchanged. The bundle's `file` records are exactly the parsed, non-vendored files of the
unit (§5.0) that are not in `skippedFiles`.

**Why.** `jssrc2cpg` omits files by its default ignore rules (S11 finding 3): 33 of 74 non-empty
`salesperson/` sources are absent at `defca60f`. D15 adopts the per-file rule "only where it
holds"; it holds for Python and not for JS over all sources.

**Which absent files are by design (stakeholder decision, 2026-11-01).**
- **JS/TS tests: excluded by design** (28 files in the slice). Python tests stay in the graph.
- **Production sources (the 5 under `salesperson/src/i18n/`): included.** A **P1-verified
  requirement**: the P1 JS/TS extractor must override `jssrc2cpg`'s default ignores, and P1 records
  the mechanism in its plan. It is **not verified** that the frontend allows it (Pass 5 open
  question 1: no documented switch found, so the fallback is the likelier case). **Fallback:**
  those files are excluded and listed with reason `frontend-ignore`; the slice's `salesperson/`
  bundle is that case, with 10 empty-`targets` call sites (`formatCurrency`, `useLocale`,
  `formatDate`). A staged-copy rename is not an escape: it changes path-anchored `qname`s.

**`header.skippedFiles` (adds a row to §2.2).**

| Field | Type | Req | Meaning |
|---|---|---|---|
| `skippedFiles` | array of `{path, reason}` | **exactly when non-empty** | the unit's source files that have no `file` record by design or by fallback. Sorted by `path` (byte order), unique. FR-7a and FR-8: the engine answers "not mapped" for a path in this list and "not found" for one in neither it nor the `file` records. |

- **`path`:** repo-relative POSIX (§8.2), under `unit.rootPath`, not under `vendoredDirs` (a
  vendored file is explained by `vendoredDirs` and never listed).
- **`reason`:** closed enum `test` | `frontend-ignore`. Both codes join the §7 registry (§7.1;
  bundle: yes; first emitted P1; on `header.skippedFiles` entries; no `status`).
- **Source file of the unit's language** = a file whose name matches the `-spec3` §3.3 "Which
  files" cell for the unit's `lang`, excluding declaration files (`.d.ts`, `.pyi`) and files
  under a vendored directory. For `config` and `markdown` units the list is not used in v1.
- **`test` pattern (closed; JS/TS units only; any other language has none in v1).** A file is
  `test` when its basename is `*.test.<ext>` or `*.spec.<ext>` with `<ext>` one of the §3.3
  JS/TS extensions, or when a directory segment below `rootPath` is `e2e` or `__tests__`. The
  writer applies it **whether or not the frontend would have parsed the file** (so overriding the
  frontend's defaults does not bring tests back). Extending the list is a registry-style change
  (§7.1), not a writer choice.
- **Precedence.** A file that is both `test` and frontend-ignored is listed once, as `test`.
- **Validator (S10) checks:** sorted, unique, POSIX, under `rootPath`, outside `vendoredDirs`,
  `reason` in the enum, no `file` record for the same path, absent when empty. It cannot check
  completeness (that needs the file system).
- **Coverage probe (normative; the P1 extractor's test).** A table test over language (JS/TS,
  Python, other) x reason (`test`, `frontend-ignore`) x overlap (file is both) x location (under
  a vendored directory, under `rootPath`, outside it) x override (on/off). For each cell it asserts
  the path appears exactly once with one reason, or not at all, and that no `file` record exists
  for it; a cell missing from the table fails the test.

## 2. Symbol and point amendments

**§5.3 `complexity` (replaces the `complexity` row's meaning).** `complexity = 1 + n`, where `n` is
the number of the symbol's own `cond` records (§5.5), owned regions included and nested symbols
excluded; for a file symbol, over its top-level code. It is **not** McCabe over the CFG. The
slice measured the gap against an `ast` count on 4,046 functions (S11 Pass 1 finding 3).

**§5.3 empty file (S11 Pass 1 finding 4; adds to the `lineStart`/`lineEnd` and `lastChanged*` rows).**
- A file with `lineCount` 0 has a `<module>` symbol with `lineStart` 1 and `lineEnd`
  `max(1, lineCount)`. Its fingerprint follows from §5.3's normalization (an empty text).
- `lastChangedAt`/`lastChangedCommit` for a symbol whose span plus `ownedRegions` has **no
  blamed line** fall back to the committer date and id of the last commit up to `builtCommit` that
  touched the path (`git log -1 --format='%ct %H' <builtCommit> -- <path>`). `header.history`
  stays `available`. The slice code tests the span only (no owned region existed); P2 must test
  span plus `ownedRegions` as stated here.

**§5.5 `ret` (adds to the first bullet).** A `lambda`-kind symbol (§5.3) with no `RETURN` from
the frontend gets a **synthesized** `ret`, ordinal 0, `line` = `max(lineStart, 1)`. This covers the
expression-bodied arrow function, which `jssrc2cpg` emits without `RETURN`, and also a block-bodied
lambda with no `return`; `function`/`method` kinds with no `return` keep zero `ret`s. The slice
synthesized 14 (`salesperson/`) and 0 (`falkor-chat/server`). A `ret` with no incoming `data` is
harmless.

**§5.10 `data` (adds to the bullets).** `use == def` is **legal**: a call site whose result feeds
its own receiver (`def` path `r`, `use` path `recv`) is a `data` record with the same point on both
ends, distinct paths and `confirmed/intra-flow`. It is **kept**, in the bundle and in the graph;
§1.4 G-4's CDG self-edge drop is a different edge and is unchanged. Traversals carry a visited set
(F-12), which stops it (the slice has 1).

## 3. §10 and writer notes

- **§10, [P-slice] bullet on rule 8:** the parenthetical "(rule 8; five in
  `falkor-chat/server/falkorchat`)" reads "(rule 8)". The count is dropped, not reconciled: a
  `Protocol`/abstract stub is a symbol by rule 8's definition, and all 7 the slice's heuristic found
  have one (Pass 4 Q10).
- **Host-Python writer note (adds to §1.13 and §4.2 Python; [writer]).** Python spans, `qname`s and
  hence ids depend on the host interpreter's `ast`. The Python normalizer runs under CPython 3.12
  (`-spec3` §3.1) and the bundle does not record the host version (`header.tools` keeps its seven
  keys). Another host minor is an untested determinism risk (the slice ran one host, one Python;
  Pass 4 SL-6). Tightenable without a bump (§1.10).

## 4. For downstream owners (not edited here)

| Owner | Item |
|---|---|
| S10 owner (`tdd-engineer`) | **Schema + validator unit, with or before acceptance (§0):** `cpg/ccg/schemas/bundle-v1.schema.json` (`header.skippedFiles`, optional), `cpg/ccg/bundle.py` (the §1 validator checks), `cpg/tests/test_bundle.py`. **Failing validator tests first** (each check of §1, "bundle without the field is valid" with the stored `s14a` bundles as positive control, and the existing header-key pin changed); the pin and the field are not authored in one step, or the pin passes by construction. The registry (`reason` codes) is updated with it. |
| Doc-only citation fix (Pass 4 m3, Pass 5 m3), after acceptance | `cpg/ccg/bundle.py` (line 3 and the `BS2` comment near 261), `cpg/ccg/keynorm.py:3`, `cpg/ccg/ids.py:3`, `cpg/spikes/vertical-slice/normalize.py:5`, `cpg/README.md` (lines 51, 70-108) and the schema's top-level `description` cite `-spec`/`-spec2`; re-point to `-spec3` + `-spec4`. Rerun the cpg suite. |
| S16 tests | The `-spec3` re-point is done; nothing re-points to `-spec4` (it carries none of the parsed sections: §1.6.3, §1.7.6, §3.6). Additive only: one test per §2 delta and the §1 coverage probe. |
| P1 extractor | Verify the `jssrc2cpg` ignore override; write `skippedFiles` by §1; pass the coverage probe. |
| S17 / graph model | `DATA` self-edge kept (§2). `skippedFiles` has one source (the header); the graph needs a lookup by path ("is X mapped?"), so a node or index, not only a `Unit` property; S17 decides. |
| Not in this document (Pass 4) | SL-4 wording (plan), `candidateCount`, G1-G8 (S17), M2 (value-mode `confirmed`; P1 gate). |
