# Connected code graph, P1: V-both ship and flow notice (plan gate)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Reviewed the uncommitted `git diff` of `docs/plans/connected-code-graph-answer-spec2.md` (spec2), `-p1-answers.md`, `-p1-ml.md`, `-p1-gates.md`, `-p1.md`
(all under `docs/plans/connected-code-graph-`) against the as-built `cpg/ccg/answer.py`, `engine/{bfs,flow,answer_build}.py`, the schema, the tests that pin
versions, and `git show` at `d02de630`. Static review plus read-only runs; the suites were not run. `REVIEW.md` read first.

**Verdict (Pass 1): needs changes** (one blocker, three majors); **Pass 2 verdict below: approve with suggestions.** The version/label decision itself holds; the defects are in how the units sequence around the published `1.1` label, a test-edit instruction that would corrupt unrelated pins, a missing dependency, and under-determined notice wording.

CPG: considered, not relevant — plan documents for a component with no loaded `cpg_*` graph of its own engine; checked by reading and grepping the as-built source.

## Findings

### B1 (blocker) The unit order edits the published `"1.1"` label and leaves an inconsistent interim state

Spec2 §1.1 rests on "a published label is never edited". Yet §7.2 (V18 row) and §1.1's bullet make V18's callee-run clauses apply to `"1.1"` **and** `"1.2"`, and P1-23d
lands first (before the label exists).
- Today `FLOW_ROWS["impact"]` lacks `("HAS_CALL","->")`, so `answer.py:_check_flow_rows` (line 877, active for `"1.1"` only) rejects a V-both chain in a `"1.1"` document by (e). After P1-23d a `"1.1"` document with callee hops validates: the published label's valid set widened, which is the edit §1.1 forbids.
- Between P1-23d and P1-39d the builder (`answer_build.py:409`) still writes `"1.1"`, but P1-23d (2) makes `table=None` the V-both table with `algorithm.version` `1.2`. A `"1.1"`-labelled answer with `1.2` algorithm and callee hops is emitted on `main`, and the P1-23d test "carries `"1.2"`" (spec2 §8.1) is ambiguous between the two fields.
- If instead V18 stays `"1.1"`-strict, the builder raises on every flag-false answer in that window.
- Flipping the label inside P1-23d does not fix it: V20(c) then fails any flag-false answer until P1-39d emits `notices`.

Suggested improvement: apply the callee-run rows (`("HAS_CALL","->")`, `CALLEE_ROWS`, clauses (b)-(d), (f), (g)) to `"1.2"` only, and order 45b's label/helper/V17/V20 part before P1-23d. Since label, table and notice must switch together, move the builder's default-table flip
(P1-23d (1) `walk(table=None)` and (2)'s `table=None` version) into P1-39d, so 23d ships the `bfs` constant, the `"1.2"` pin and the validator rows only. Or merge the builder halves of 23d and 39d into one unit. The implementer decides; the plan must state an order in which no commit on `main` emits or accepts a `"1.1"` V-both document. Name the assertion that catches it: a `"1.1"` base carrying a `HAS_CALL ->` exit is rejected.

### M1 (major) P1-39d (4) / R-7 blanket "1.1" to "1.2" would corrupt unrelated pins and misses a file

P1-39d (4): "every existing `"1.1"` expected value in the engine tests becomes `"1.2"`"; spec2 R-7 names five files. Checked by grep:
- `tests/engine_answer_helpers.py:65`, `tests/test_engine_staleness.py:35,357,376-380,404,420` and `tests/test_engine_answer.py:456,465` hold `"1.1"` as the **snapshot row's graph-model version** (`(SNAP, 0, EMPTY, "1.1", "1.0")`; `loader.py:38`, `assemble/snapshot.py:18` `GRAPH_MODEL_VERSION = "1.1"`). Not answer labels, and not to change. `test_answer.py` pins are P1-45b's, not P1-39d's.
- The assertions that do move: `test_engine_answer.py:83` (label) and `:102` (algorithm version, a P1-23d effect once `table=None` is V-both), and `test_engine_answer_v18.py:105` (label) and `:161` (version line). The v18 file is named in R-7 yet absent from P1-39d's Files.

Suggested improvement: replace the blanket rule by the four named assertions, say the snapshot-row `"1.1"` stays, and add `tests/test_engine_answer_v18.py` to P1-39d's Files (and say which of `:105`/`:161` belongs to 23d versus 39d).

### M2 (major) P1-26 (and P1-25's text path) depend on P1-24b but do not need it

P1-26 behaviour (10) calls `ccg.render.common.flow_notice_lines(doc)`, which P1-24b creates. `-p1.md` row P1-26 Needs `29, 39, 24, 25, 45, -ml`; P1-25 Needs `…24…39d`. Neither reaches 24b (24b is a sibling of 39d, not upstream of 25). A scheduler may start P1-26 before the function exists.

Suggested improvement: add `24b` to P1-26 (and to P1-25 if its test (9) asserts the notice in `render_text` output), in both `-p1.md` and the unit headers.

### M3 (major) The notice wording is under-determined and can be false or off-column

Spec2 §5.1 requires saying flow-dependent impacts "are **not enumerated**", and the template is column-agnostic ("callers and callees only").
- V-both does list some flow items (9 of 28 per `-p1-flow-fallback-result-ml.md`), so "not enumerated" can be read as "none were listed". The result note's own allowed wording is "partial-recall".
- The depends-on column also triggers the notice (§5.1, P1-39d), but the sentence about "impacts" and "callers and callees" describes the impact column; a depends-on answer lists callees.

Suggested improvement: require wording of the form "flow-dependent effects in `<langs>` may be missing from this list" (stating partial, not nil), and a column-aware second sentence (or one neutral to both), with a test per direction. State the matching mode of the ban test (see m1).

### m1 (minor) The ban list is testable only once its matching mode is stated

Checked: the spec's required statements can be written without `safe`, `complete`, `exhaustive`, `full`, `guarantee`, `no impact`, `all impacts`, so the list does not forbid needed wording. But "incomplete", "unsafe", "carefully" contain banned substrings and "not evidence of no impact" is natural phrasing. State case-insensitive whole-word matching (then "incomplete" is allowed), run it against the rendered text after substituting `langs`, and keep the "no digit outside a language name" check (the `lang` pattern allows digits).

### m2 (minor) The "wiring-only answer carries no notice" fixture is likely unconstructible

Spec2 §5.1 and P1-39d (3) test "a config anchor with wiring items only carries none". By the as-built rows, `bound_in` (inc 1) then `via_in:site` (inc 1) reach the bound `CallSite` within the minimum bound of 2 (`bfs.py` rows), so a config anchor with any binding visits a `CallSite`; a symbol anchor is itself a visited `Symbol`. Only an unbound config anchor or a contract-point anchor yields no code node. Restate the test on those, or drop the sentence "so a wiring-only answer has none" from §5.1.

### m3 (minor) New fail-closed paths on answers that worked before

§5.1 makes any visited code node whose unit has no `Unit` row or no language a `GraphDataError`. Today `Units.ensure` is called only for gated reads (`bfs.py:831,842`); a `Symbol` step's `unit` may be `None` (`bfs.py:446`), and `Units.get(None)` raises. So answers that never read units (contract mode, ungated rows) gain a failure mode, and a flag-true unit with no language would raise although only flag-false units need a language. Restrict the language error to flag-false units, state the missing-unit behaviour as a decision, and note that the new `units` read adds a `recipe.queries` name to answers that had none (P1-39d already hedges this).

### m4 (minor) Stale conditionals in documents the edit did not reach

- `-p1.md:509` (backlog list): "AC-25's control half if P1-39's walk leaves it open".
- `docs/test-plans/connected-code-graph-p1.md:137` and `docs/test-plans/connected-code-graph-fixtures.md:324`: "closes only if P1-39's shipped walk lists the guarded point" (`qa-engineer`'s; route to them).
- `docs/requirements/connected-code-graph.md` FR-8 (line 123) has no decision-log entry for the 2026-11-19 waiver; the plans now say FR-8 is "not met". That entry is `tico`'s. `-p1-ml.md` §3.5's conditional "If V-flow … ships" is the pre-registered rule and correctly left.

### n1 (nit)

`answer.py` docstrings and messages still say "1.1" for what becomes 1.1-or-later (lines 5-6, 533, 852, 876, 909); P1-45b's file list covers the file, so add them to its done-condition. Spec2 §7.3's "P1-45 made these edits" and §1.2's "built … published" are past-tense narration but carry rationale; acceptable.

## Checks asked for (outcome)

- **(a) Version/label:** holds. `git merge-base --is-ancestor d02de630 origin/main` succeeds; `git show d02de630:cpg/ccg/engine/bfs.py` line 202 pins `{"1.1": "5cf047c14e43da44d0363707d3036080"}` and `answer_build.py:409/411` emits label `"1.1"` and `w.version()`. Ran `bfs.algorithm_version()` at HEAD: `1.1`; fingerprints (first 12 hex): as-built `5cf047c14e43`, V-flow `68f449379185`, V-both `39e7ab0280ea`, V-union `0325a4e3e6ab`. So V-both is a different definition and `1.2` is right. The label minor bump is right under v1 §9. Caveat: B1.
- **(b) Trigger and V20:** computable. `Walk.visited` (keys label, id, state) is a superset of every node a labelled hop leaves; the post-walk `_owner_join` gives points a `unit`; `Walk.units` only holds gate-read units, so P1-39d must `ensure` the rest, as it says (m3). V20(c) is not circular: `Hop.lang` equals the `staleness.units` language of the departing hop's unit (V18(d), `answer.py:899`), that node is visited and flag-false, hence in `langs`; the validator reads only the document. `name_out` resolves within a language (`Name.nameKey` carries `lang`), so the callee run stays in the opener's language.
- **(c) `== "1.1"` sites:** unfiltered grep of `cpg/ccg` finds exactly five comparisons: `answer.py:543` (`facts_bytes`), `:853` (V17), `:877` (V18), `:904` (V19), `:939` (`_check_hops` flag), plus the tuple at `:69`, the schema enum (`answer-v1.schema.json:12-13`) and the emitter `answer_build.py:409`. P1-45b's Files and helper clause cover all of them. Other `"1.1"` hits (`loader.py`, `bundle.py`, `snapshot.py`) are unrelated version spaces (M1).
- **(d) Sequencing:** 23d → 45b is correctly serial (`answer.py`, `test_answer.py`). 39d (`bfs.py`, `answer_build.py`, engine tests) and 24b (`render/*`, `test_render.py`, `golden/render/`) share no file; no claim collision. The defects are B1, M1, M2.
- **(e) Ban list:** m1.
- **(f) Citations and contradictions:** every cited path resolves (`-p1-flow-fallback-result-ml.md`, `cpg/eval/results/ccg-p1-flow-fallback-v1.json`, the five plans); no contradiction between the five edited documents after grepping "not closed", "V-both", "V-flow", "notices", "1.1". Stale items elsewhere are in m4.
- **(g) Growth-stops:** edits rewrite in place; the revision lines are single dated lines; no finding table or disposition list was added. Q-1 is a stakeholder question left open, as allowed.
- **Interaction defects (same revision):** B1, M1 and M2 are such interactions (23d/45b/39d ordering, R-7 versus 39d's file list, 24b versus 26).

## What's solid

- The `1.1` published-fingerprint reasoning and keeping the `5cf0…` literal untouched, with `candidate_table("as-built") == bfs.ROWS` (verified true today).
- Notice as an optional, `minItems: 1` field with one encoding, in `facts_bytes`, minted enum, no wording or figure in the document; the validator checks only what the answer carries.
- Mutant lists and the extension of the expected-value tables (`REQUIRED`, `$defs`, `ENUMS`, `ARRAY_RULES`) are specific.
- Honest recording of FR-8 / AC-25 control half as not met, consistent across the five documents.

## Open questions

1. B1: should the builder's default-table flip move to P1-39d, or should 23d/39d's builder work merge? (Architect's call.)
2. Q-1 (stakeholder): whether `--fail-on any` fails on a notice; recommendation (no) is sound.
3. Who records the FR-8 waiver in the requirements decision log (`tico`)?

## Pass 2

Scope: the architect's in-place revision of the same five documents (uncommitted `git diff`). Evidence: reading the revised text against `answer.py`, `bfs.py`, `flow.py`, `answer_build.py`; the 1058-test default subset (`test_answer`, `test_engine_answer`, `_v18`, `_bfs_guards`, `_flow`) passes on the as-built tree; and a throwaway simulation of the P1-39d flip in a scratchpad copy (V-both rows as `SHIPPED_ROWS`, `walk`/`definition` default to it, `ALGORITHM_VERSIONS` gains the V-both fingerprint `39e7ab02…`, V18 disabled; no label, V20 or notices). The simulation is crude: it shows which existing assertions a default-table flip moves, not the full unit. Failures from the copy's missing data files (`test_engine_anchors`, three report tests in `test_engine_answer`) are artifacts and excluded.

CPG: considered, not relevant — plan documents; checked by reading and by running the as-built code.

**Verdict: approve with suggestions.** B1 is resolved by design; one new major (N1) and two minors.

### Dispositions

- **B1: fixed.** New order 45b → 23d → (39d ∥ 24b); spec2 §5 states the callee run is `"1.2"`-only with `FLOW_ROWS` unchanged for `"1.1"`; R-8 states no commit emits or accepts a `"1.1"` V-both document. Walked per commit: **45b** adds `"1.2"`, the helper, V17/V20; builder still emits `"1.1"` with the caller-only table; the only existing pin it moves is `test_answer.py:2443-2444` (its own file); engine tests pin no schema version (grep). **23d** adds the constant, the `"1.2"` pin and `"1.2"`-only validator rows; `walk(table=None)`, `definition()` and the builder are untouched, so builder output and every `"1.0"`/`"1.1"` accept/reject are unchanged (a `"1.1"` `HAS_CALL ->` exit still fails (e)). **39d** flips table, label and notices in one commit, so V20(c) is satisfied when labelled hops first appear. One unnamed 23d consequence: once `ALGORITHM_VERSIONS` holds the V-both fingerprint, a V-both candidate walk has a version, so `test_engine_flow::test_a_candidate_walk_has_no_algorithm_version_only_the_shipped_table_has` fails at 23d (it failed in my simulation); see N1.
- **M1: fixed in principle, incomplete in practice** (N1). The snapshot-row `"1.1"` pins are explicitly kept and the moved assertions are named; 39d now lists `test_engine_answer_v18.py`.
- **M2: fixed.** `-p1.md` rows: P1-25 and P1-26 both Need `24b`; P1-24b "Blocks: P1-25, P1-26".
- **M3: fixed.** Wording is "may be missing from this list", partial not empty, column-neutral, tested on one impact and one depends-on document.
- **m1: fixed.** Whole-word, case-insensitive, `incomplete`/`unsafe` allowed, `langs` substituted. Interaction check against M3: the required statements ("may be missing", "only call relationships were followed", "not evidence that no such effect exists") contain no banned token; `no such effect` is not the phrase `no impact`; `only` is unbanned; `none`/`nothing` are banned from the template and not needed. No conflict.
- **m2: fixed** (unbound config anchor / contract-point anchor; bound config anchor carries the language).
- **m3: ruled below.**
- **m4: partly fixed.** `-p1.md` backlog line fixed. Still stale and outside the five documents: `docs/test-plans/connected-code-graph-p1.md:137` and `docs/test-plans/connected-code-graph-fixtures.md:324` ("unless P1-39's walk lists the guarded point"; `qa-engineer`), and no FR-8 decision-log entry in `docs/requirements/connected-code-graph.md` (`tico`).
- **n1: fixed** (45b's Files now include the docstrings and messages).

### N1 (major) The "assertions that move, by name" list is incomplete and misses a file

P1-39d (5) and spec2 §8.1 claim the moved assertions are exactly four, and P1-39d's Files omit `tests/test_engine_bfs_guards.py`. In the flip simulation these existing tests also failed, beyond the four named:
- `test_engine_bfs_guards.py`: `test_the_shipped_definition_maps_to_one_version_pinned_as_a_literal` (`:323-326`, `algorithm_version() == "1.1"` on the default definition), `test_the_definition_record_holds_the_table_the_identifiers_and_the_bound` (`:305-309`, rows equal `bfs.ROWS`), `test_the_level_guard_and_the_chunk_sizes_are_not_part_of_the_definition` (`:355-359`).
- `test_engine_answer.py::test_the_depth_bound_cuts_are_counted_into_unexplored_and_the_item_beyond_is_not_listed` (the default walk now lists callee `F2`, changing items and cuts).
- `test_engine_answer_v18.py`: `test_a_candidate_table_refuses_exactly_where_v18_does_not_define_its_added_hops_and_nowhere_else` and `test_a_candidate_table_lists_at_least_what_the_shipped_one_does_and_the_v_union_what_each_does` (only the carries-version test is named).
- `test_engine_flow.py`: `test_v_flow_follows_…`, `test_v_both_lists_the_callees_…`, `test_a_candidate_walk_has_no_algorithm_version_…` (23d), and two `test_a_row_pair_that_shares_a_read_issues_it_once…` cases (read counts 21 vs 20, 18 vs 17).

An implementer who trusts "four by name" will treat the rest as regressions or weaken them. Suggested improvement: state that P1-39d (and 23d for the candidate-version test) re-derive every assertion that takes the default table as its baseline, classified moved-by-design versus regression in the unit's commit note; name the families above as the expected set ("at least"), add `test_engine_bfs_guards.py` to 39d's Files, and keep the rule that nothing is deleted without a replacement assertion of the new behaviour.

### n2 (minor) Two small accuracy points

- §5.1 wording "only call relationships were followed": the walk in a flag-false unit also follows `TO ←` links and wiring rows (`bfs.py` impact rows `to_in:fn`, `wiring`); say "call and link relationships" or drop "only".
- P1-23d (4) skips the whole `test_engine_superset_dev.py`; its always-run test (digest and candidate measurement over the store) does not depend on the pinned-engine refusal, which belongs to the real-phase path (`superset_dev.py:190-224`, `-m live` only). Skip only the real-phase test.

### Ruling on m3 (architect's departure: fail closed on a missing `Unit` row or unit)

**Accepted.** Evidence: (1) `staleness.unit_entries` already raises `GraphDataError` for any chain unit without a `Unit` row, and `staleness.units` must hold every hop's unit (V9), so every answer already requires a `Unit` row for each unit on its chains. The new exposure is only visited nodes that are not on any listed chain. (2) The assembler writes `Symbol`/`CallSite` units and the `Unit` rows from the same bundle headers (`assemble/rows.py:427,448-456`), and G5 guarantees `Unit.flowAvailable` on every `Unit` row, so a code node without a unit or `Unit` row is a malformed graph, not a normal one. (3) Skipping instead would omit the notice silently, the one failure this feature exists to prevent. (4) The language requirement is restricted to flag-false units, as I asked. Residual: answers that never read units gain one `units` read and a new failure on a corrupt graph; the plan states the `recipe.queries` consequence. Not verified on a live graph (no instance run); the check is by code reading. One precision: `chain_units` skips nodes whose `unit` fact is `None`, while §5.1 raises for a visited code node with none; that stricter rule is correct for code nodes (`Link`/`ContractPoint` are excluded) and the engine test should plant a `Symbol` without a unit.

### Growth-stops check (root `REVIEW.md`)

Passed. Each document keeps one dated `Revised 2026-11-19:` line replacing the previous one (spec2 line 5; `-p1-answers.md` line 5; `-p1-ml.md` line 13); no finding IDs, finding→resolution table or disposition list appears in the five documents (grep for `B1`/`M1`…/`39-notice` finds only pre-existing unrelated hits); reasons are stated as design (R-8, "so the published label's valid set does not widen"). New words are new scope (§5.1, §8.1, P1-23d/45b/39d/24b). Only the stakeholder question Q-1 remains open, as allowed.

### Pass 2 open items

1. N1 (architect): widen the moved-assertion statement and add `test_engine_bfs_guards.py` to P1-39d.
2. m4 remainder: `qa-engineer` (test plans) and `tico` (FR-8 decision-log entry).
3. Q-1 (stakeholder) unchanged.
