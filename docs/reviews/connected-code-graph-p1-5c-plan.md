# Review — C-1001 P1 post-P1-05 plan edits (5c, D12, P1-06/08/09 changes)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-05 5c, P1-06, P1-08, P1-09)

## Scope & verdict

Reviewed the uncommitted edits to `docs/plans/connected-code-graph-p1-extraction.md` and `docs/plans/connected-code-graph-p1.md`
(`git diff`) against the committed normalizer (4cd9299f), `cpg/ccg/units.py`, `cpg/ccg/bundle.py`, bundle spec3/spec4, the stored
slice bundles under `cpg/.ccg-work/vertical-slice/s14a/`, and `docs/reviews/connected-code-graph-p1-05-5b.md`. Docs only; no source
edited. `REVIEW.md` (root) applied; `cpg/REVIEW.md` does not exist.

**Verdict: needs changes.** One blocker (F1, P1-09 points the join fix at the wrong file and a new resolver P1-34 will not use) and
three majors (F2 delta header bytes, F3 P1-08 signature, F4 failure-code mapping has no owner). D12 and 5c (b)/(c) are sound.

CPG: considered, not relevant — the questions are plan-vs-committed-file and stored-bundle facts, answered by reading and executing the files.

**Growth-stops check.** Word counts (`wc -w`), before → after: `-p1-extraction.md` 4,525 → 5,709 (+1,184); `-p1.md` 9,103 → 9,221 (+118).
The growth is new scope (5c, ~700 words; D12 row; 5a/5b file-list names; P1-08 wiring) plus P1-09's grounding paragraph. No
finding→resolution table or disposition list; findings are not cited by ID. Two smells, both minor (F7): D12's rationale is stated
twice (index row and 5c (a) "Why", ~225 words), and 5c's header still says it blocks P1-09 while the same section says P1-09 does
not edit the file.

## Findings

### F1 — blocker — P1-09: the join fix is attributed to the wrong file; the new resolver is a second copy that P1-34 will not use
Plan P1-09 *Files*: the two join misses "exist only in `.../adjudication/tools/flow/classify.py` (labels matched to points by line
and code text)", and the fix goes in a new `cpg/tests/projection_endpoints.py` ("the endpoint-to-point resolver the goldens and the
recall report use").

Checked, not assumed:
- `normalize/points.py` does not exist (`ls cpg/extractors/joern/normalize/`: `assemble_bundle, blame, core, javascript, mask, python`)
  and `projection_endpoints.py` does not exist yet. Both true.
- The literal `iteratorNonEmptyOrException` appears in code only at `classify.py:14` (`LOWERED` regex); the other hits are data
  (`*.adj.jsonl`, results) and docs. True, but `classify.py` only **classifies** a miss after the fact (`("join", "cond-resolved-to-wrong-node
  ... contains-join picks the operand")`, lines 106 and 112). It does not perform the join.
- The join is `cpg/spikes/frontend-fitness/s28_core.py:277` `resolve_endpoint`, cond branch: line match, then exact, then
  whitespace-insensitive, then the `contains` tier (`det["via"] = "contains"`). `s28_metrics.py` imports it (`import s28_core as C`),
  and `s28_run.py` (which P1-34 re-points to the P1 `extract.sc`, `-p1-gates.md` P1-34 (1)) scores through it. `test_s28_metrics.py`
  mutates it.

Consequences. (a) The implementer is told to look in `classify.py` for a join that is not there. (b) A new
`projection_endpoints.py` duplicates `resolve_endpoint`; P1-09 behaviour (6) says "`s28_metrics.py` reused, not copied", so the
plan contradicts itself. (c) P1-34's entry check (465/481, at most 2 of 159 fixable misses open) is measured by `s28_run.py`
through `s28_core`, which will still carry the 40 join misses unless P1-34 is changed to import the fix; the P1-09 "dev recall" and
P1-34's figure can then disagree. Neither P1-09 nor P1-34 says which resolver scores.

Fix: ground the claim in `s28_core.resolve_endpoint`. Decide one home for the two join fixes: either a P1 change to `s28_core`
(frozen P0 tool with mutants in `test_s28_metrics.py`, so name the new mutants), or a new module that `s28_core`/`s28_run.py`
import. Then add to P1-34 (1) the obligation that its run uses that resolver, and state in P1-09 which resolver its report line uses.

### F2 — major — equivalence delta: the header bytes change on every bump and no delta names it
Stored header `tools.normalizerVersion` is `"0.1"` (`head -1` of the fcs bundle) and the strict/exempt-header comparison exempts only
`builtAt`/`sandboxAuditIds` (`test_normalize_equivalence.py:143`). So `NORMALIZER_VERSION` bumped (P1-05 (2), 5c, P1-06, P1-08) makes
every stored-bundle byte comparison fail unless a delta rewrites `tools.normalizerVersion`; the plan's delta kinds are "a record
removed or a header field added, `counts` adjusted". Two further gaps:
- P1-06 says masking leaves stored `cond.code` "either all unchanged (no delta) or listed ... with `NORMALIZER_VERSION` bumped".
  Read literally the bump is only in the second branch; but a bump changes bytes, so even the "unchanged" branch needs a delta.
  One rule is needed: every step that bumps registers a `tools.normalizerVersion` delta, or a step with no byte change does not bump.
- If a delta derives the expected version from the `NORMALIZER_VERSION` constant, the bump can never fail the test. Pin the literal
  per delta, and assert the last delta's literal equals the constant.
Also checkable for 5c (b) itself (executed): the fcs bundle has 213 `decorator` records, 3 with a `lambda`-kind `targetId`
(`_read_or_absent`, `retry.call_with_retry` x2), `counts.decorator` 213 → 210; salesperson has 0. The delta is therefore well-formed.

### F3 — major — P1-08: `skipped_files(unit, stage, parsed, override)` is under-specified and diverges from the file already written
- Plan: "`stage` is `RawFacts.stage`, the listing of the unit's source files". `RawFacts.stage` is a `Path` (`run.py:305`), not a
  listing; the new `skipped.py` docstring says the function "never reads the file system". Either `stage` is a directory the
  function walks (then the no-FS contract goes) or it is a precomputed listing (then name it so and say who builds it).
- Path form: facts names are unit-relative (`src/App.tsx`, `lang_files`), bundle `file.path` and `skippedFiles.path` are
  repo-relative (`salesperson/src/...`); the plan never says where rebasing happens.
- The untracked `cpg/extractors/joern/skipped.py` (in flight) is `skipped_files(unit, parsed: Mapping[str, bool], override)`, no
  `stage`, repo-relative paths. The plan's `parsed` (`lang_files(facts, lang)`, a list of names) is a different type. Whichever is
  right, plan and delegate disagree now.
- Delta claim "`falkor-chat/server` gains whatever the coverage probe lists": wrong. Spec4 §1 gives non-JS units no reason; the
  function returns `[]` for Python; the stage has 107 `.py` files and the stored bundle 107 `file` records. The fcs delta is **none**,
  and the coverage probe is a table test, not bundle output. The salesperson figure holds (stage 74 source files, 41 `file`
  records, 33 = 28 `test` + 5 frontend-ignore; the stored bundle already lacks them, so the delta is header-only).
Fix: state the argument types and path base; say fcs has no delta.

### F4 — major — "reported under `extract:failed`" has no owning unit or test
D12 and 5c (a) route a non-UTF-8 file to `extract:failed` "via P1-44 (3)". `extract:failed` is real (`-p1-graph.md` G4 `FAIL_REASONS`).
But P1-44 (3) says only that a failure in one extractor fails that unit's bundle; no unit maps `SourceRefused` (nor
`BuildError`/`HistoryError`/`StageError`/`RunnerError`) to a `FAIL_REASONS` code, and P1-20 "sets reasons only from the constant".
Add to P1-44 (or P1-20) one row: an exception from normalization or extraction maps to `extract:failed`, value-free, with a test that
drives a Latin-1 unit through the dispatcher and asserts `failReason == "extract:failed"` and no source bytes in it.

### F5 — minor — 5c (a) describes shipped behaviour as new work
`Source.text` already raises `SourceRefused` on non-UTF-8 (`core.py:101-107`, tested in `test_normalize_core.py:88`;
5b Pass 2 F5). What 5c adds is only the `build_bundle`-level tests (Latin-1 `.py`/`.ts`, no bundle left under `bundles/<system>/`).
Drop `core.py (Source)` from 5c's file list for (a) and say "already enforced; this adds the end-to-end tests". The claim "no partial
file" holds structurally: `write_bundle` is the last call (`assemble_bundle.py:511`).

### F6 — minor — 5c (c) wording and order of edits
- Spec grounding verified: `ownedRegions` is in `bundle-v1.schema.json:1004` (`minItems 1`, line-pair items), validated strictly sorted
  (`ccg/bundle.py:822`, `_strictly_sorted`), and spec3 §5.3 says "exactly when the symbol has owned regions" and "sorted", and the
  fingerprint takes "the sorted, de-duplicated union". So "exactly when non-empty, sorted pairs" is the spec, not invented.
  The plan should add "de-duplicated" for the emitted list, since `_strictly_sorted` rejects duplicates.
- Order (question 4): the named order 5c → P1-06 → P1-08 is consistent across the three steps and P1-09 (no edit) in the index.
  The coordinator's order, P1-06 first, breaks text in three places: 5c "Blocks: P1-06", P1-06 "Needs: 5c", and P1-08 "Needs: 5c, P1-06"
  (P1-06 landing first inverts a declared dependency); P1-06 must register its delta in a delta list that 5c is stated to create
  as "its first entry"; and `NORMALIZER_VERSION` values are never named, so two out-of-order bumps collide. Working tree: 
  `assemble_bundle.py` already carries P1-06's call-site edit (`mask_cond_code`), as expected. Fix the text to "whichever of
  P1-06/5c lands first creates the delta list; versions bump 0.1 → 0.2 → 0.3 in landing order".
- 5c header says it "Blocks: ... P1-09 (they edit the same file)"; P1-09 explicitly does not. Remove P1-09 there.

### F7 — minor/nit
- D12 rationale duplicated in `-p1.md` D12 and 5c (a) "Why" (a fact has one home): keep the index row to the rule and cite 5c.
- D12 "surrogateescape is unusable" verified by execution (foreground, `ulimit -v 1000000`, `timeout -s KILL 60`): `b'x = "\xe9"'.decode('utf-8','surrogateescape')`
  gives `'\udce9'`; `ast.parse` raises `UnicodeEncodeError ('utf-8' codec can't encode ... surrogates not allowed)`; `json.dumps(..., ensure_ascii=False).encode('utf-8')`
  (the canonical writer, `ccg/bundle.py:67-68`) raises `UnicodeEncodeError`. Nuance: with `ensure_ascii=True` the dump would succeed (`\udce9`);
  the claim holds because the writer uses `False`. Say "the canonical writer".
- 0 of 382 recomputed under two definitions: (1) tracked `.py/.pyi/.ts/.tsx/.mts/.cts/.js/.jsx/.mjs/.cjs`, 718 files, outside
  `cpg/` and `deprecated/` = 382, 0 fail `decode('utf-8')`; (2) the manifest's own `units._excluder` over the same 718 = 382, 0 fail.
  Same set, both 0. (337 of the 382 are non-ASCII but valid; no BOM.) Count is over `git ls-files` of a repo with uncommitted
  changes, so it is a bound on the committed state.
- `lambda_decos` readers (question 2), grep over the repo: `core.py:120,126` (slot), `python.py:113` (assignment), `assemble_bundle.py:385`
  (`s.decos + s.lambda_decos`), and `test_normalize_assemble.py:72` (the golden that asserts the lambda decorator, which the plan says
  inverts). Nothing else reads it; `extract.sc` is untouched, so P1-04's raw-fact equivalence holds. Spec3 §5.12 targets "function,
  method or type"; it does not say "lambda", so state the ruling as an interpretation, as the plan does, with the spec text cited.
- 5b's file list omits `cpg/tests/test_normalize_final.py` (exists). Plan P1-03 text now states `discover_units(..., commit=None)` while
  committed `units.py:211` has no `commit` parameter; the text says P1-20 adds it (`-p1-linking.md` P1-20 (8), exists), which is
  fine, but read as a P1-03 signature it is stale. Say "P1-03 ships without it".
- Cited symbols: all named in P1-03/P1-05 exist as committed: `Source`, `SourceRefused`, `Scope`, `Builder`, `name_norm`, `quote_field`,
  `path_from`, `Resolver`, `py_resolve`, `py_imports`, `py_spans`, `build_scopes`, `js_resolve`, `js_imports`, `js_type_spans`,
  `git`, `parse_porcelain`, `blame`, `last_changed`, `path_last_commit`, `is_commit_id`, `HistoryError`, `assemble`, `build_bundle`,
  `NORMALIZER`, `NORMALIZER_VERSION`, `lang_files`, `point_of_use`, `cond_pid`, and the five test files named. `mask.py`,
  `skipped.py`, `test_mask_cond.py`, `test_skipped_files.py` are untracked (in flight), correctly marked "new".

## What's solid
- D12 as a decision: non-UTF-8 fails the unit, no new enum; reversal trigger named; facts verified (above).
- 5c (b): removal is a pure record deletion with a checkable delta (3 of 213, `counts.decorator` 210), no other reader.
- 5c (c): grounded in schema, validator and spec3 §5.3; the synthetic-test design (fingerprint differs; absent-field stored bundles unchanged) is right.
- P1-09's claim that the committed normalizer has no cond join and the step must not edit `assemble_bundle.py` (`point_of_use`, `cond_pid`
  join by CPG id) is correct.
- P1-08's salesperson counts (74 / 41 / 33 = 28 + 5) reproduce.

## Open questions
1. F1: does the scorer's join fix live in `s28_core` (P0 tool, mutants) or in a new module it imports? Needs `architect`/`data-scientist`.
2. F3: `stage` as directory or listing; path base for `skippedFiles`. Needs `architect`, and the P1-08 delegate must be told.

## Route
- `architect`: F1, F2, F3, F4 (add a P1-44/P1-20 row), F5, F6, F7. Re-gate after.
- `teco`: send the F3 divergence to the running P1-08 delegate before it commits `skipped.py`; the P1-06-first order needs the F6 text fix.

## Pass 2 (re-gate of the architect's in-place revision)

**Verdict: approve with suggestions.** No blocker. F1–F7 are closed; two new minors (G1, G2) and one nit (G3). Word counts (`wc -w`):
`-p1-extraction.md` 5,709 → 5,628 (pass-1 baseline 4,525), `-p1.md` 9,221 → 9,181 (baseline 9,103), `-p1-linking.md` 2,916. Pass 2 shrank both
revised documents; no finding→resolution table or disposition list in any plan (findings are not cited by ID). Growth-stops: pass.

| Finding | Disposition (evidence rechecked) |
|---|---|
| F1 | **Fixed, one misstatement (G1).** P1-09 now names `cpg/spikes/frontend-fitness/s28_core.py` `resolve_endpoint` as the one home; `projection_endpoints.py` is gone from the plan. `s28_metrics.py:27,259-260` imports `s28_core as C` and calls `C.resolve_endpoint`. `cpg/tests/test_s28_metrics.py` exists. The "`data-scientist` consulted before the `s28_core` change merges" is written as an **Owed** item in the step, and the index row gates P1-09 with `data-scientist`. |
| F2 | **Fixed.** Delta model `(version_literal, apply)` with the literal pinned, expected header taking the last literal, and a test that the **last** literal equals `NORMALIZER_VERSION`; a step bumps even with no record change. Versions 5c `0.2`, P1-06 `0.3`, P1-08 `0.4` appear in 5c, P1-06 and P1-08 and agree. 5c "Needs 5b", P1-06 "Needs 05 ... order 5c then this", P1-08 "Needs 05, 5c, P1-06", index `05 (5c)` / `04, 05 (5c), 06` agree with 5c-first; 5c no longer "blocks" P1-09. Working tree: `assemble_bundle.py` no longer has the parked `mask_cond_code` call (grep empty), so 5c starts from the committed writer; `mask.py` is untracked and not imported. |
| F3 | **Fixed.** P1-08 text matches `skipped.py:53` `skipped_files(unit, parsed: Mapping[str, bool], override)`: pure, repo-relative POSIX, `[]` for non-JS. The wiring (parsed built from `raw.stage` listing, rebased with `Builder.P`, `True` iff in `lang_files(facts, lang)`) is assigned to `assemble_bundle.py`. fcs "no delta (107 = 107)" and salesperson 33 = 28 + 5 header-only stay as I verified. |
| F4 | **Fixed in substance; G2.** Every exception named exists as committed: `HistoryError`, `require_full_history` (`blame.py:23,56`), `StageError`, `EmptyCpgError`, `RunnerError` (`run.py:61-69`), `SourceRefused`, `RuntimeRefused` (`core.py:24,72`), `BuildError` (`assemble_bundle.py:37`), `BundleError` (`bundle.py:46`). `build:shallow-clone`, `extract:failed`, `extract:timeout`, `bundle:invalid` are listed members of `FAIL_REASONS` in `-p1-graph.md` G4 (the constant itself is created by P1-18 and is not code yet). The P1-44 row carries a Latin-1 test through `build_unit`, a "second unit still builds" assertion and one planted exception per row. P1-20 (1) now takes the reason P1-44 assigned. |
| F5 | **Fixed.** 5c (a) says `Source.text` already raises and 5c adds the end-to-end tests; 5c's files no longer list `Source`. |
| F6 | **Fixed.** "de-duplicated" added to (c); the 5c/P1-09 Blocks text is gone. |
| F7 | **Fixed.** D12 is a one-line rule citing 5c (a); "canonical writer (`ensure_ascii=False`)" stated; `test_normalize_final.py` listed; the `commit=None` clause is gone from P1-03 (only P1-20 (8) states it). |

**Ruling on `RunnerTimeout(RunnerError)`.** Needed. `run.py:270` raises a plain `RunnerError("... timed out after ...")` for a timeout, so today a timeout
is distinguishable from any other runner failure only by message text, which a mapping must not parse. A subclass keeps every existing
`except RunnerError` working. The mapping must test `RunnerTimeout` before `RunnerError`; say so.

### G1 — minor — P1-09 says `s28_run.py` imports `s28_core`; it does not
`s28_run.py` imports only `s27_run`; it extracts facts (`EXTRACT`, line 24) and scores nothing. Scoring is `s28_metrics.py` -> `s28_core`.
The conclusion (P1-09's recall and P1-34's entry check share one scorer) holds only if P1-34 scores with `s28_metrics.py`; its text says it
runs "`s28_run.py`/`s28_metrics.py`", so it does. Fix the sentence: "`s28_metrics.py` imports it; `s28_run.py` only extracts".

### G2 — minor — P1-44 mapping: the `run.py` edit has no owner row, and "unlisted exception is a defect" is unspecified
- `RunnerTimeout` is an edit to committed `extractors/joern/run.py` (and `tests/test_joern_run.py` pins the kill), but P1-44's *Files* lists
  only `cpg/ccg/build.py` and two test files. Add `run.py` and the test to P1-44's files (or to P1-04's owner).
- Unlisted exceptions exist in the same path: `run.py:94,123` call `subprocess.run(..., check=True, timeout=...)` (can raise
  `CalledProcessError`/`TimeoutExpired`), and `write_bundle` raises `AtRestError` (`atrest.py:54`). Behaviour (3) says one unit's failure
  fails that unit alone; "a defect, not a code" does not say whether it propagates and aborts the whole refresh. State one rule: either
  catch `Exception` at the unit boundary and record `extract:failed` while logging the type, or let it abort loudly; test the choice.
  Mapping `ExtractError` (the common base of `StageError`/`EmptyCpgError`/`RunnerError`) rather than three names also covers a new subclass.

### G3 — nit
The `data-scientist` owed consult is inside P1-09's prose; add it to the step's *Gate* line so a gate check finds it.

## Route (Pass 2)
- `architect`: G1, G2, G3 (text only; none blocks 5c, which a tdd-engineer can implement from the current text).
- `teco`: 5c may proceed. P1-06's code stays parked until 5c lands; first delta-list literal is `0.2`.
