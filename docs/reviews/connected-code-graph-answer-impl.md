# Connected code graph: canonical answer, S34 implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P0)

**Scope.** Diff-scoped gate of S34 (commit `794eac32`): `cpg/ccg/schemas/answer-v1.schema.json`, `cpg/ccg/answer.py`, `cpg/tests/test_answer.py`, `cpg/tests/golden/answer-ac1.json`. Baseline: `docs/plans/connected-code-graph-answer-spec.md` v1.2 (§2–§9, §11), `docs/reviews/connected-code-graph-answer-spec.md` and `-ml.md`, row S34 of `docs/plans/connected-code-graph-p0-2.md`, root `REVIEW.md` (no component `REVIEW.md` exists for `cpg/`). Growth-stops: N/A. Not reviewed: the live-spec update (the feature is not closing), `cpg/spikes/frontend-fitness/` (untracked, not S34).

**Verdict: approve with suggestions.** No blocker: all 15 rules V1–V15 are implemented (Appendix A), the golden AC-1 answer reproduces from the slice, the canonical bytes are stable across hash seeds, the 499 tests pass. There are four majors (F1–F4): two real code gaps (F1, F2), one probe-coverage gap the root `REVIEW.md` names (F3), and a cluster of 13 non-equivalent surviving mutants (F4). I would close S34 only after F1–F4, or after each is recorded as a backlog item.

CPG: considered, not relevant — `cpg/ccg/answer.py` is a new contract module no loaded `cpg_*` graph indexes; every claim below was traced by running the code.

**Stopping condition (falsifiable).** Re-review answers "close" iff, running from `cd cpg`: (1) `answer.dumps`, `answer.is_canonical` and `answer.facts_bytes` on a document holding a lone surrogate raise `AnswerError` (or return False for `is_canonical`) and never `UnicodeEncodeError`; (2) a valid document with two `recipe.queries` of one `name`, two groups of one `(kind, key)`, two `lifted` of one `(siteId, ord)` or two `bound` of one `ord` is rejected, each by its own test; (3) the planted-axis table in F3 shows an empty "missed" column for `array_paths` and `derived_data_fields`; (4) each of the 13 survivors in Appendix B is killed by a named test (re-run with the harness pattern in Appendix B) or is argued equivalent in this file; (5) `normalize_question` has an owner (F7). Any one failing reopens the review.

## Findings

### F1 (major, verified): a lone surrogate passes `loads`/`validate` and crashes `dumps`, `is_canonical`, `facts_bytes`

`answer.py:229-247` (`_check_values`) checks NFC and strip only. `b'..."name":"api\ud800Fetch"...'` goes through `loads` and `validate` without error. `dumps` then raises `UnicodeEncodeError` from `canonical_line(...).encode("utf-8")`, and so do `is_canonical` and `facts_bytes`. I ran it: `validate -> OK`, `dumps -> OTHER UnicodeEncodeError`. Spec §6 and V1 say UTF-8 and "rejected, never repaired"; spec §6 also promises `AnswerError`, not a stray exception, for the other unrepresentable input. A path read with `surrogateescape` (a filename holding invalid UTF-8) lands in `Hop.path`, so the engine's own `dumps` can hit this.
**Fix.** In `_check_values`, for a `str`, try `value.encode("utf-8")` and `_fail("V1", ...)` on `UnicodeEncodeError`. Prototype, run on a copy: the 499 tests stay green and the surrogate document raises `AnswerError`. The suite does not constrain the prototype; the assertion that would catch it wrong is the new test in the stopping condition (1).

### F2 (major, verified): sort-key ties are not rejected, so two valid documents differ in bytes by array order

Spec §6 claims "ties are impossible" for `items` only, and that holds (V8 makes the subject id unique). Four other arrays sort on a key that nothing makes unique:
- `recipe.queries` by `name`: spec §4.1 says "names unique"; no rule checks it. A duplicate-name query is accepted, and `dumps` of the list reversed is **not** equal.
- `groups` by `(kind, key)`: two groups of one key with distinct members validate; reversed order gives different bytes (`probe1.py`, "bytes equal after reverse: False").
- `Hop.lifted` by `(siteId, ord)` and `Hop.bound` by `ord`: a duplicate `ord` validates; reversed lifted gives different bytes.
Each is a case of FR-19's "same facts, same bytes" failing for a valid document, which is S34's own acceptance row.
**Fix.** One uniqueness check per key in `_check_interpretation` or a new V-row (V2 is nearest): names, `(kind, key)`, `(siteId, ord)`, `ord`. Prototype on a copy: suite green (499), duplicate-query and duplicate-lifted documents raise `AnswerError`. Put the four tests in the same round that adds the rule, but write them red first (a test that builds the duplicate and expects `AnswerError`), then the rule. Do not add the uniqueness to the array-shuffle test itself: its fixtures have unique keys by construction and could never redden.

### F3 (major per root `REVIEW.md` "guard that enumerates a schema", no live defect today): the two schema probes recurse into only some composition axes

`answer.array_paths` (the §6 coverage probe) and `answer.derived_data_fields` (the F-3 derivation that `DATA_FIELDS` is tested against) walk `properties`, `oneOf/anyOf/allOf`, `if/then/else` and `items` of a named property. I planted one array (or one `x-ccg-data` string) per axis on a copy of the schema:

| Planted at | `array_paths` sees it | `unclassified_strings` (V14) |
|---|---|---|
| `additionalProperties` (map value) | missed | flags |
| `patternProperties` | missed | flags |
| `not` | missed | flags |
| `dependentSchemas` | missed | flags |
| a `$defs` entry that is itself `type: array` | missed | flags |
| inner array of an array of arrays | missed (outer seen) | flags |
| `x-ccg-data` string in an inline object inside an array's `items` | n/a; `derived_data_fields` missed it | classified, so V14 is green |

The last row is the F-3 hole in waiting: a data string inside an inline array-of-objects (as `recipe.queries[]` is today) satisfies V14 yet never reaches `DATA_FIELDS`, so the renderer would not fence it. The existing tests (`:1641`) plant five axes, all of the ones the walker already handles.
**Fix.** One recursive walker over every subschema-bearing keyword of Draft 2020-12 (derive the keyword list from the metaschema's applicator vocabulary, do not type it by hand), shared by `array_paths`, `derived_data_fields` and the classification walker. Add the planted-axis tests first, with one plant per keyword of that derived list, so the probe is red before the walker changes.

### F4 (major, test gap; the code is right, verified by direct run): 13 non-equivalent surviving mutants

My own runs on copies (Appendix B) left 25 survivors of 82; 12 are equivalent (argued in B), 13 are not:
- **Options' own arrays unpinned (2).** `canonicalize` that stops sorting an option's `anchors`/`scope` survives; so does an option sort key built on non-canonical bytes. The shuffle case for `Request.anchors` shuffles the resolved request only, and the derived array set names the path, not the instance. I checked the code directly: a multi-anchor unranked and ranked option set shuffles to equal bytes. Add the shuffle on `interpretation.options[*]`.
- **"Never by locale" is tested for scope arrays only (4).** Casefolding the item `path`, the item root path, the lookup `path` or the query name survives. Spec §6 says code point; a P1 renderer or the gate harness following a different order would disagree. Add a pair `B.py` < `a.py` to the item-order test for `rootPath`, `path`, and to the lookups and queries tests.
- **The F-4 namespace split lives twice, and the tokenizer is a copy (2).** `_screen_units` (`answer.py:302`) and `mask_identifier` (`:334`) each split on the first `:`. A mutant that splits `mask_identifier` on the last `:` survives, while the same mutant in `_screen_units` is killed by the `ns:a:b.c` vector. Add `ns:a:b` (a failing `a:b` unit) to the `mask_identifier` vectors. The tokenizer regex `_TOKEN` is also a copy of `bundle._KEYPATH_PART`; a mutant widening the index alternative to `[0-9]+` survives (`a[01]` would pass unmasked where the bundle grammar fails closed). Add `a[01]`.
- **`facts_bytes` (2).** Dropping `options[*].rank` from the facts survives (the rank is part of the did-you-mean order); retaining `provenance` for an `unresolved` answer survives (the tests cover resolved and ambiguous only). The lead's "both directions" holds for `lookups` and for resolved/ambiguous provenance, not for an unresolved answer's provenance.
- **Schema (2).** Removing `recipe.guards.visitedSeedsAnchor`'s `const: true` survives (a `false` is accepted); removing the ambiguous branch's "no `lookups`" survives (nothing else rejects it).
- **Group members tie-break (1).** Dropping the subject id from the member sort key survives; the test at `:1307` sorts members on the earlier components only.

### F5 (minor, verified): identity strings are stripped by key name, which misses array elements and some path-like fields
`IDENTITY_KEYS` (`answer.py:60`) matches `qname, path, keyPath, calleeName, keyNorm, name`. `Scope.paths[]` (key `paths`), `Unit.rootPath` and `Lifted.urlPath` are not covered: `" svc/a"` as a scope path validates. Spec §6 lists `path`; `request.scope` is compared field by field by AC-24 (§3), so an unstripped path is a silent mismatch. Add `paths` (elements) and `rootPath` to the check, or state in the spec why they are exempt.

### F6 (minor, verified): spec shape rules that no V-row names are unvalidated
`ownedRegions` `[5,3,2]` (odd, descending) and `lineEnd < lineStart` validate. Spec §5.1 says "pairs, ascending" and H1's span test reads them. Either add a row to the spec's table with a rule, or note they are the engine's attestation (spec §10.7). Spec-side first, since §8 lists V1–V15 as the closed set.

### F7 (minor): `normalize_question` has no function and no owner
Spec §3 states the rule normatively (NFC; CRLF to LF; space/tab runs to one space; ends stripped; newlines and case kept) and `cacheKey` hashes its output, but §11's public-name list has no function, and `docs/plans/connected-code-graph-p0-2.md`'s carry-over list C1–C9 has no item for it. `ids.tuple_id` already NFCs and strips the ends, so the internal-space and CRLF steps are the only ones left to two implementers to get right. I grepped the repo (`normalize_question`, `normalizedQuestion`, `normalized_question`): only the spec, the ml review and `cache_key`'s parameter. Add a small `normalize_question(raw) -> str` to `answer.py` with literal vectors (cheap and the interpreter's cache key depends on it), or add carry-over C10 owned by the P1 interpreter unit.

### F8 (minor): `facts_bytes` does not validate
`facts_bytes` canonicalizes any dict; an invalid document (`status: "bogus"`) returns bytes (verified, 5022). A caller comparing facts of an invalid answer gets a result instead of an error. Validate, or document that callers validate first.

### F9 (nit)
- The builder's tally "279 mutants, 8 survivors" includes 2 harness errors that never ran (`src_run3.txt`: `v1-validate-skips-values`, `v10-unexplored`); 277 ran. I ran the first: killed. The second's target text is not in the code (its rule is the schema's), so it was never a Python mutant.
- `v15-ext-case` is called equivalent; it is spec-silent, not equivalent (`A.PY` is skipped today; a `.lower()` would recompute). Decide in the spec, `-spec3` §3.3 does not say.
- `V12` messages for option anchors say `interpretation.request[ri]`; cosmetic.
- The golden test does not pin `lifted[].id/name/kind/viaSiteId/repo/unit`. I re-derived all of them (Appendix C), so there is no defect; a mutated golden field outside the pinned ones would still validate.

## What is solid

- **V1–V15 walked row by row against code** (Appendix A): all 15 are present; V10's "unexplored requires bound" and V2's "iff" conditions live in the schema (`DepthBound`, `interpretation` `allOf`), and tests kill their removal.
- **Golden AC-1 re-derived independently** (own reader over `assemble` → `load.build_rows` → `impact.walk` on `cpg/.ccg-work/vertical-slice/s14a`): the walk gives one item, `getHealth`, distance 1, confirmed, 5-hop chain with the same ids, edges, dirs and states; the four `lifted` hops equal field for field; `candidate_count` = 1; staleness `builtAt`/`builtCommit`/`lang`/`rootPath` equal the bundle headers; `snapshotId` equals the loader's.
- **Determinism:** `dumps` of 14 base documents (golden and the 13 test bases), each shuffled 200 times on every array except `chain`/`ownedRegions`/`rcaMarks`, was byte-equal to the unshuffled bytes under `PYTHONHASHSEED` 0, 1 and 12345, and the printed hashes were identical across the three seeds. (Within F2's limits.)
- **Test independence:** the item-order tests are built so each key component dominates the later ones, from spec text; vectors are literals; the golden is pinned by values; I found no expected value computed by the function it checks (the `answer.sampling_hash`/`cache_key` calls in `b_prov` build a base and the literals at `:1381-1401` pin the values). The deviation (code drafted before tests) did not leave mirroring tests in this sample.
- **Planted-axis controls exist** for the F-3 classification walker (five plants plus the two-classifier and list-typed cases) and for F-4 (seven field locations, the contract-point kinds, masked and unmasked forms). The fence vectors cover the spec's literal, 500/501/600, and backtick runs of 1 and 3.
- **Silent-argument mutants killed:** hash-input order in `cache_key` and `sampling_hash`, option sort direction and key, item tiebreak (id → qname), line (start → end), scope casefold, anchor `(type,id)` → id, units without root, V15 language and extension mapping, the `qname` in the V15 recompute, `facts_bytes` dropping `stamp`.

## Open questions

1. The file name `connected-code-graph-answer-impl.md` was given in the brief; the spec's family slug is `connected-code-graph-answer-spec` (so `...-answer-spec-impl.md` by the family rule). I used the brief's name; `teco` may want to rename.
2. F6 and F9 (`.PY`) need a spec ruling, not a code change; whose call is that (`architect`)?

## Proposed REVIEW.md lines

Target `REVIEW.md`, heading "Always check". Source: F2 (and `probe1.py`).
- Add: "A canonical-bytes rule is checked for ties: for every array's sort key, a valid document with two elements equal on that key, in both orders, must be rejected or give equal bytes. A key declared unique in the spec but enforced nowhere is a finding."
No prune proposed.

## Appendix A: V1–V15 walked against code

| Rule | Where in code | Result |
|---|---|---|
| V1 `loads`/NFC/strip/null/float/dup | `loads`, `_check_values`, `_no_dup_keys`, `_no_float` | implemented; gaps F1 (surrogate), F5 (key-name based strip) |
| V2 interpretation iff, empty answers, unique anchors, lookups | schema `allOf` + `_check_interpretation` | implemented; ambiguous+lookups rejected by schema only (F4) |
| V3 hop shape | `_check_hops`, `KIND_FIELDS` | implemented |
| V4 H3 anchor chaining with kind | `_check_item_rules` | implemented |
| V5 Link then VIA CallSite, `lifted.siteId` | `_check_hops` | implemented |
| V6 reason registry + status, engine codes | `registry_status` | implemented |
| V7 derivations | `_check_item_rules` | implemented |
| V8 group rules, subject unique across items/members | schema + `_check_groups` | implemented |
| V9 units set equality | `_check_staleness`, `_units_of` | implemented |
| V10 depthBound | schema `DepthBound` + `_check_depth` | implemented |
| V11 offset 0 hash | `validate` | implemented |
| V12 F-4 screen | `_check_f4`, `_check_identifier` | implemented (anchors incl. options, subject, hops, bound, launch) |
| V13 F-12 | `_check_cycles` | implemented |
| V14 F-3 classification | `unclassified_strings` | implemented; probe axes F3 |
| V15 id recompute | `_check_ids`, `EXT_LANG` | implemented |

## Appendix B: mutants (own, on copies under `scratchpad/an34/mutcopies/`)

82 mutants (1 sanity control, red as required): runs `run1.py` (32), `run2.py` (23 schema + 12 Python), `run3.py` (13), `run4.py` (2). 57 killed, 25 survived. Equivalent survivors (12, with the reason): `v15-anchor-repo-item` (a repo id is already lowercase), `mask-mount-drops-empty` (`screen_fails("")` is False), `py-loads-nonobject` (`validate` repeats the check), `py-v7-hop0` (V3 forbids a status on hop 0), `py-v9-item-unit` (V7 makes it the last hop's unit), `py-v12-placeholder` (only the boolean differs), `py-fence-suffix-order` (the suffix has no backtick), `py-opts-rank-any` (V2 rejects partial ranks first; reachable only through the non-validating `facts_bytes`, F8), `sch-int-to-number-distance` (V1 rejects floats first), `sch-item-possible-reason-dropped` (V7 requires the hop's reason), `sch-group-members-minItems-off` (`count >= 1` and V8's `count == len` cover it), `facts-lookups-also-ambiguous-mixed` (the schema forbids lookups in ambiguous). Non-equivalent (13, F4): `item-casefold-path`, `item-rootPath-casefold`, `lookups-path-casefold`, `queries-casefold`, `opts-not-canon`, `optkey-noncanon`, `mask-ns-rpartition`, `mask-index-leading-zero`, `facts-drops-rank`, `facts-prov-not-unresolved`, `sch-guards-const-removed`, `sch-ambiguous-forbids-lookups-off`, `groups-members-drop-id-tiebreak`.

Builder's survivors re-derived: of the 8 Python ones I re-ran 6 as my own mutants (`loads-nonobject`, `v7-hop0`, `v9-item-unit`, `v12-placeholder`, `fence-suffix`, `opts-rank`) and argued the other two by reading (`ord-silent-unit-root-lookup`: roots are keyed by unit id, a repo id would have to be 32 hex; `v15-ext-case`: F9). Schema: 3 of their survivor families reproduced (int-to-number, `Item` reason clause, `Group.members` minItems) and `Hop.candidateCount` minimum is killed in my run as in theirs. Their 45 reconciles with their files: `sch_run2.txt` lists 47 survivors of 625, of which two (`Item.distance` and `Hop.candidateCount` `drop-minimum`) were killed by tests added afterwards.

## Appendix C: golden re-derivation

Scripts: `scratchpad/an34/derive.py`, `cmp.py` (own `Rd` reader over `rows.edges`, not `MemoryReader.from_rows`). Result lines: `lifted equal True 4`; `cc 1`; hop 1/3/4 path, line, qname, unit, repo all True; `states [...] == [...]`; `snap True`; `ids.contract_point_id` = `a359de51...`, symbol ids = `8ae57f34...` and `a6c26487...`.

## Pass 2 (2026-11-04): re-gate of `0528c556`

**Verdict: approve. S34 is accepted**, with one residual nit (below) and the F6/`.PY` rulings routed to `architect`. Observed: `tests/test_answer.py` 577 passed; whole `cpg` 2962 passed, 1 skipped (slow leaf mutation, off by default), 10 deselected (live).

**Stopping condition.** (1) met for `loads`/`validate`/`dumps`/`is_canonical`: the lone-surrogate document raises `AnswerError`; `facts_bytes` on it still raises `UnicodeEncodeError`, because `facts_bytes` does not validate by the F8 ruling, so I waive that clause. (2) met: duplicate query name, group `(kind, key)`, `lifted` `(siteId, ord)` and `bound` `ord` each raise `AnswerError` (my `p2.py`, `probe1.py`). (3) met: every plant of the F3 table, run with my `plant.py`, now appears in the `array_paths` diff (`pa[]`, `staleness.*`, `staleness.~^x`, `staleness.pn`, `PA`, `staleness.pd`), and `derived_data_fields` returns `Group.pq[].z` for the inline-array data string. (4) met, see the mutant line. (5) met, `normalize_question` exists.

| F | Disposition |
|---|---|
| F1 | fixed: `_check_values` rejects non-UTF-8-encodable strings (verified above) |
| F2 | fixed: uniqueness in V2 (queries), V3 (`lifted`, `bound`), V8 (groups); verified by run |
| F3 | fixed: one `schema_nodes` walker feeds `array_paths` and `derived_data_fields` (see deviation 3) |
| F4 | fixed: all 13 survivors killed; see mutant line |
| F5 | fixed, stricter than spec (deviation 2); `verbPath` is still unstripped (nit) |
| F6 | not fixed; routed to `architect` as agreed |
| F7 | fixed: `normalize_question` plus vectors (deviation 5) |
| F8 | documented in the docstring, not validated (deviation 4); accepted |
| F9 | `.PY` routed to `architect`; golden-field pinning and message labels untouched, still nits |

**Mutants.** My harness re-run on the fixed code (`run1`–`run5`): 69 killed and 12 survived, the survivors being exactly the 12 equivalents of Appendix B. The only change to the harness: `optkey-noncanon` is retired (the code now *is* that form, by design, since `canonicalize` canonicalizes options first, and the matching `opts-not-canon` mutant is killed), and `opt-key-repr` was re-pointed at the new `_opt_key` text (killed). That matches the builder's 69.

**Deviations.**
1. *`unclassified_strings` kept on `_iter_schema`:* accepted. It walks every dict of the schema with no applicator logic, so it is the broader of the two; all F3 plants are flagged by it. Two walkers remain, but they answer different questions (every string vs. probe paths).
2. *F5 strip extended to `rootPath`, `urlPath`, `Scope.paths[]`:* accepted, but it is stricter than spec §6, whose identity list names only `path`; the spec text must list these at close, or a conforming engine that does not strip them is rejected. `verbPath` and `accessPath` are the same class and remain unstripped (verified: `" a0.method"` validates); either add them or state why not. Nit.
3. *F3 deriving keywords from `jsonschema_specifications`:* it is a hard dependency of `jsonschema` (installed: `jsonschema` 4.26.0 pinned in `pyproject.toml`, `jsonschema-specifications` 2025.9.1, `referencing` 0.37.0), so it is available wherever `answer.py` already imports; it is not pinned directly, so the keyword set is fixed only through the `jsonschema==4.26.0` pin. Draft 2020-12's vocabulary is frozen, so drift risk is low; a metaschema change would surface as the plant tests failing, which is the desired direction.
4. *F8 documented:* agreed. The caller is told `facts_bytes` projects an already-valid document.
5. *`normalize_question`, "the ends stripped":* I agree `str.strip()` taking edge newlines is the right reading. `ids.tuple_id` already strips the same ends before hashing (checked: leading/trailing `\n`, NBSP, U+2003, U+3000 give equal ids with and without), so the hash and the function cannot disagree; interior lines keep a single leading space, as the spec's "within a line" says (`"a b\n c"` from `"a \t b\r\n c"`). The 13 vectors match spec §3 by reading and by my own cases.

**Same-revision interaction.** The identity-strip key inheritance (`key` passed down into array elements) cannot leak into the dict elements of `lifted`/`chain` (each dict element resets the key per property); the new uniqueness checks sit in V2/V3/V8 after the schema, so the golden and all 13 bases still validate (577 green).
