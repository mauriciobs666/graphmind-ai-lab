# Review: P1-45b implementation (schema 1.2, `notices`, V20, 1.1-or-later helper)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1-45b)

## Scope & verdict

Uncommitted diff of `cpg/ccg/answer.py`, `cpg/ccg/schemas/answer-v1.schema.json`, `cpg/tests/test_answer.py`, against `docs/plans/connected-code-graph-p1-answers.md` `## P1-45b` and `docs/plans/connected-code-graph-answer-spec2.md` §1.1, §5.1, §7, §8.1, §9. Context: `docs/reviews/connected-code-graph-p1-39-notice.md`. Root `REVIEW.md` applied.

**Verdict: approve with suggestions.** No blocker. Two minor test-strength gaps (M1, M2) and two nits.

CPG: considered, not relevant — a diff-scoped review of a validator and its schema, read directly; no call-graph question arose.

Run: `cpg/.venv/bin/python -m pytest tests/test_answer.py` in `cpg/`: 673 passed (4 s). Not run by me: the engine files (the coordinator's 1381-pass run is taken as given).

## Findings

### M1 (minor) — V20 (c) coverage loop is untested for a labelled *last* hop and for *group members* (verified by mutation)
`answer.py` `_check_notices` iterates `_all_items(doc)` × `it["chain"]`. Two mutants survive all 673 tests in my copy:
- `enumerate(it["chain"][:-1])` (last hop skipped);
- `list(_all_items(doc))[:len(doc["items"])]` (group members skipped).

Cause: every `flow-unavailable` hop in `b_flow` and `_b_flow_two_langs` is a middle hop of a top-level item. The last-hop shape is real: spec2 §8.1's `B-both` has `fg` as an `end` item carrying the label. P1-23d would catch it, but P1-45b's own §8.1 list says V20 (c) is a P1-45b obligation.
Fix: two P1-45b tests with no notice, each rejected by V20 and each preceded by a validating base: (1) a `"1.2"` document whose only labelled hop is the last hop of an item chain; (2) the same labelled chain as a group member. Coverage-probe axes: hop position (first/middle/last) × container (items/groups.members) × labelled language count (1/2).

### M2 (minor) — the `notices` sort test cannot tell the key `code` from the key `langs`
Mutant `d["notices"].sort(key=lambda n: n["langs"])` survives. `test_canonicalize_sorts_notices_by_code` uses `a-code`→`[go]` and `flow-not-enumerated`→`[python]`, so both keys give one order. The mutant is unreachable through `validate`/`dumps` in P1 (one code, and two notices with one code fail V20 (b)), so this is a weak-fixture issue in a `canonicalize`-only test, not a live defect. Fix: give the earlier code the later langs, e.g. `a-code`→`[python]` and `flow-not-enumerated`→`[go]`.

### N1 (nit) — `answer_build.py:4` and `engine/staleness.py:4` docstrings still say "1.1 answer's `facts_bytes`" and "V1-V19"
The plan's file list for P1-45b excludes both; the claim is now true for 1.2 as well, and "V1-V19" is stale. P1-39d is the unit that touches `answer_build.py`; carry it there. No change needed here.

### N2 (nit) — `SHUFFLE_CANONICALIZE_ONLY` removes `notices` from the generic shuffle-then-`dumps` loop
Justified by the schema (one code, so two notices cannot validate); the replacement `test_canonicalize_sorts_notices_by_code` plus M2's fix is enough. When a second `Notice.code` lands, the exemption must go (a comment says so).

## Check-by-check

1. **Done-condition.** The builder emits `"1.1"` (`answer_build.py:409`, unchanged; not in the diff). In `test_answer.py` the only removed lines are the pins the plan allows: the `ANSWER_SCHEMA_VERSIONS` tuple, `"1.2"` moved out of the rejected-version list (now `"1.3"`), and the test rename. The schema-inventory edits (`$defs` set gains `Notice`, `INSTANCES`/`REQUIRED`, `ENUMS`, derived-array count 15→17, `SHUFFLE_CANONICALIZE_ONLY`) **do not exceed the plan**: spec2 §8.1 says the `ARRAY_RULES` coverage probe, `unclassified_strings` and `derived_data_fields` pass "with their sets extended by `notices`/`Notice.langs`/`Notice.code`". Pre-existing tests otherwise unchanged (diff read).
2. **Unfiltered "1.1" sweep** (`grep -n '1\.1'` on both files; I also grepped `"1.0"`, `answerSchemaVersion`, `ANSWER_SCHEMA`). `answer.py`: lines 5, 71, 75, 497, 547, 867-868, 873, 894, 927 are docstrings/messages, all reworded to "1.1-or-later" where the rule is 1.1-or-later; lines 876-887 are V17 messages in the `"1.0"` branch ("a 1.1 construct"), still correct; line 70 is the version tuple. No `== "1.1"` comparison remains. The two remaining `"1.2"` string comparisons (`answer.py:869` and `932`) are V17's/V20's own `"1.2"`-only rules, as intended. `answer_build.py`: line 409 is the emission (ruled: stays `"1.1"` here), line 4 is N1. The only other hit in the schema file is the description (updated). `is_1_1_or_later`: `"1.0"`→False, `"1.1"`/`"1.2"`→True (a test asserts all three); an unknown label (`"1.3"`) → False, but `validate` rejects it by schema first, and the helper is called only after `_check_values`/schema in `validate`; V17's `!= "1.2"` for notices likewise treats unknown as non-1.2. Verified by mutants (helper `== "1.1"`, `includes 1.0`, `only 1.2` all killed).
3. **V20 vs §5.1.** (a) status clause plus V17's label rule; (b) one per code; (c) the covered set is the union of `langs` over `flow-not-enumerated` notices; the sweep is over every hop of every item and group member, and item-level `reason` is the first possible hop's reason (V7), so no item-level reason escapes. Not circular: V20 reads hop reasons, which V18 (d) already ties to the departing unit's language, never `notices`. The converse (notice without labelled hop) accepted, as §5.1 says. V17 rejects `notices` on `"1.0"`/`"1.1"` (tests, mutants killed). Gaps: M1.
4. **Canonical bytes.** `facts_bytes` keeps `notices` (it only pops provenance/lookups/builtAt/failedAt; a mutant popping it is killed). `canonicalize` sorts `langs` by code point (`str.sort`; the schema pattern is ASCII) then `notices` by `code`; `dumps` validates before serializing, so malformed `notices` (missing key, non-list, mixed types) raise `AnswerError`, not `KeyError` (probed five shapes). Deterministic: no set iteration reaches output.
5. **Mutation sweep** (copy of `cpg/ccg`, `tests`, `fixtures` under the scratchpad; nothing mutated in place; each mutant run on `tests/test_answer.py`). 42 mutants: 38 killed, 4 survived. I did not re-derive the implementer's 27 individually; I wrote my own 42 covering the same areas: V20 (a)/(b)/(c)/(gate), V17's notices clause (three variants), the helper (three), each of the four `== "1.1"` call sites, canonical sort (five), `facts_bytes`, schema (nine: `1.2` enum, `minItems` x2, `uniqueItems`, closed `Notice`, pattern, `required` x2, code enum), the `validate` call. Argument-corruption mutants: covered-set `code !=` (killed), only-first-lang (killed), prefix slice off by one and none (killed), code constant (killed), first notice only (**survived**), chain minus last hop (**survived**, M1), items only (**survived**, M1). Survivors: *first notice only* is equivalent in P1 (a second same-code notice is rejected by (b) first; the enum has one code), re-derived by counter-shape: no valid P1 document distinguishes it; *sort key `langs`* is M2; the other two are M1.
6. **Same-unit interactions.** V17 runs before V20, so a `notices` on `"1.1"` reports V17, and V20 can return early for non-1.2. `ARRAY_RULES` ↔ coverage probe ↔ `unclassified_strings` ↔ `derived_data_fields`: consistent with the extended sets, tests green. No interaction defect found.
7. **Docs.** Module docstring, `facts_bytes`, `_check_labels`, `_check_flow_rows`, V19, `validate` and the schema description reworded as the plan lists; `is_1_1_or_later` added to the module's public-API list.

## What's solid
One helper, no `== "1.1"` left; mutants on every call site killed. Every rejection test starts from a validating base. The schema matches spec2 §9's table (both `notices` and `Notice`, closed, `minItems`, `uniqueItems`, pattern). Existing `"1.0"`/`"1.1"` behavior is untouched in the diff.

## Open questions
None.
