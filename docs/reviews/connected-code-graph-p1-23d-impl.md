# Review: P1-23d implementation (V-both table, `"1.2"` pin, V18 callee run)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Uncommitted diff `git diff -- cpg/ccg cpg/tests` (7 files) against `docs/plans/connected-code-graph-p1-answers.md` `## P1-23d` and `docs/plans/connected-code-graph-answer-spec2.md` §1.2, §5 (V18 (b)-(g)), §8.1, §9 R-7/R-8. Not reviewed: P1-39d (the flip), the builder.

**Verdict: approve with suggestions.** No blocker. The done-condition holds as executed (below). One spec clause was dropped on a "shadowed by V4" claim that does not hold for the subject's type (F1), and two test gaps were found by mutation (F2, F3).

CPG: considered, not relevant — a diff-scoped review of the answer validator and walk tables, checked by running them; no call-graph question arose.

## Evidence (what I ran)

- Suites (from `cpg/`, `.venv/bin/python -m pytest`): `test_answer`, `test_engine_bfs_guards`, `test_engine_flow`, `test_engine_superset_dev`, `test_superset_dev_tool` = 998 passed, 2 deselected; `test_engine_answer_v18` + `test_engine_answer` = 257 passed. Not run: the rest of the suite, the live phases.
- (1) Done-condition: `bfs.definition()` equals `bfs.definition(bfs.ROWS)` and fingerprints to `5cf047c1...` (executed); `git diff -U0 bfs.py` shows `ROWS` and `_build_rows` untouched (comment line only); `walk`/`definition` bodies untouched; `build_answer` not in the diff. 1.1 acceptance and rejection hold: `test_v18_e_*` (a 1.1 doc with `HAS_CALL ->` out of `function`, and the relabelled `B-both`) pass, and a mutant that applies the run to `"1.1"` is killed (A5).
- (2) Fingerprint: I recomputed it from a `git archive HEAD` copy, using HEAD's `flow.candidate_table("V-both")` (53 rows) and HEAD's `bfs.definition`/`fingerprint`, code the diff does not touch: `39e7ab0280ea38fcba30754caa2595b0`, equal to the new literal. The same copy gives `5cf047c14e43da44d0363707d3036080` for the default, equal to the `"1.1"` literal. `PINNED_1_2` is a literal, not read from the constant.
- (3) Real-answer probe: every document the tests' worlds produce (`test_engine_answer_v18.raw_worlds`, both flags, both columns, every anchor; 376 per table), built with `as-built` and `V-both` tables, relabelled `"1.2"` with the python-style notice, validated by the real `answer.validate`: 376/376 and 376/376 pass. In the V-both set, 22 documents contain a run; 5 of them place a run item in a group member. Run hops: 49 impact `function -> HAS_CALL ->` hops, 0 with a state other than `end`. Depends-on: 17 `function -> HAS_CALL ->` hops, 0 in state `end` (all `value`).
- (4) `git status`: `cpg/eval/` has no change; `superset_dev.py`, `test_superset_dev_tool.py`, `preregistration.jsonl` and `results/ccg-p1-flow-fallback-v1.json` are untouched. The diff on `test_engine_superset_dev.py` is the one `@pytest.mark.skip` on `test_the_real_phase`.
- (5) Mutants, in a copy under the scratchpad (nothing reverted in place): 27 V18 mutants run against `test_answer.py` (26 killed, 1 survived: F2); 17 `bfs`/`flow` mutants run against `test_engine_bfs_guards.py` and `test_engine_flow.py` (17 killed); 1 group-member mutant against `test_answer.py` + `test_engine_answer_v18.py` (survived: F3). I could not match the delegate's "17/18 with 1 equivalent" to a mutant list, so I ran my own set.

## Findings

### F1 · minor · (g)'s symbol-subject clause was dropped as "shadowed by V4"; V4 shadows only the id, not the type

`cpg/ccg/answer.py:906-941` (`_check_flow_rows`) has no check that a run item's subject is a symbol. Spec2 §5 (g) says "every item it ends in has a `symbol` subject"; §8.1 lists "an item of the run with a `config` subject" as rejected (g). V4 (`answer.py:719`) checks only `chain[-1]["id"] == subject[<id field>]`.

Executed: in `b_both()`, I replaced `items[1].subject` with `{"type": "config", "entryId": <fg's symbol id>, "path": ..., "keyPath": ...}`. `validate` **accepted** it. The planted-subject test (`test_v18_g_an_item_a_run_ends_in_*`) uses an unrelated `entryId`, so it exercises V4's id mismatch, not (g)'s type clause. The same swap with `symbolId = <site id>` is rejected, but by V15 (the id does not recompute), not by V18.

Reach: a forged or corrupt document only; no honest builder produces it. That is why it is minor. But the spec is the contract and the delegate removed a clause of it on an argument that is true only in part.

Fix (either): (a) restore it, one line after `run` is computed: `if run == "callee" and <this is the item's last hop> and subject["type"] != "symbol": _fail("V18", w, "(g) ...")`, with a test whose config `entryId` equals the callee's id (the probe above, so V4 passes and only (g) can reject it); or (b) architect edits §5 (g) to say the symbol subject is held by V4 and V15, and drops the §8.1 "config subject" case. Take (a): it costs one line and keeps the spec as written.

### F2 · minor · test gap: the direction of the `CALLS`/`CALLS_NAME` rows in `CALLEE_ROWS` is unpinned

Mutant A4c: add `("CALLS", "<-")` to `CALLEE_ROWS` (`answer.py:87`). All 686 `test_answer.py` tests pass. It is not equivalent: in `b_both()` set `_item_g(d)["chain"][4]["dir"] = "<-"` (a `CALLS <-` hop, state `end`, after the site). The original rejects it with (f); the mutant **accepts** it. The (f) tests insert hops with `direction` default or with a state other than `end` (`other()` is state `value`; `fq` follows the callee), so none is a `CALLS <-` after a run site. `("HAS_CALL","<-")` added is caught (executed: still (f)).

Fix: in `test_v18_f_*`, add the case `chain[4]["dir"] = "<-"` (and `chain[6]` for the `CALLS_NAME` row), `rejects(..., r"V18.*\(f\)")`. Per the coverage-probe rule, derive the cases from the axes `_run_kind` reads (edge type x dir x state x previous run kind), not from the list I gave.

### F3 · minor · V18 over group members is not tested at all

Mutant: `_check_flow_rows` iterating `doc["items"]` only (members skipped). `test_answer.py` + `test_engine_answer_v18.py`: 691 passed, 3 skipped, the mutant **survives**. The loop is pre-existing, but this unit adds clauses (b)-(g) to it, spec2 §5 says "item and member hop", and the real data has run items as group members (5 documents in my probe). V20 has a member test (`test_v20_coverage_sees_a_labelled_hop_of_a_group_member`); V18 has none.

Fix: one test, `B-both` with `items[2]` moved into a `groups[0].members` (set `group.reason` to the member's reason), asserting it validates, then a (f) or (b) violation inside the member rejects.

### F4 · minor · spec text for architect: two edits (spec2 §5, "What the validator checks")

Ruling on the delegate's two open questions:

1. **Run hop requires state `end`: correct; spec text needs one edit.** The walk rows (`bfs.V_BOTH_ROWS`) all arrive in state `end`, and §5's own prose says the site and callee hops are "state `end`". The structural definition ("A hop is in a callee run when it is a `HAS_CALL ->` hop whose predecessor is ...") omits the state. Read literally, a depends-on `function -> HAS_CALL ->` hop is not a run only because the paragraph says "(impact column only)"; then (g)'s first half ("occurs only in the impact column") is vacuous. The implementation makes it live: a depends-on document with run-shaped hops in state `end` is rejected by (g), while the depends-on walk's real shape (`function -> HAS_CALL ->` state `value`, then `CALLS ->` `value`; 17 hops in my probe, all `value`) validates. Nothing a real walk produces is refused (376/376 documents above). An impact `function -> HAS_CALL ->` hop in state `value` is rejected (e), where the literal spec text would let it through as a run; stricter, and no real answer has it. Edit: `docs/plans/connected-code-graph-answer-spec2.md` §5 ("What the validator checks", the run-membership sentence) adds "in state `end`", and (g) says what it rejects (a depends-on document with a run-shaped hop in state `end`). Owner `architect`.
2. **(g) symbol-subject clause removed: see F1.** If F1 is taken as (b), the same section is edited again.

### F5 · nit

- The skip reason on `test_the_real_phase` (`test_engine_superset_dev.py:83`) says P1-23d moved the engine files the tool "now refuses". True, but the tool already refused at HEAD: I computed every pin of the record and `-tool` line against `HEAD` blobs; `cpg/ccg/answer.py`, `answer-v1.schema.json`, `connected-code-graph-answer-spec2.md`, `-p1-answers.md`, `-p1-ml.md` already differ at HEAD (P1-45b and earlier), and `bfs.py`/`flow.py` match at HEAD but not in the working tree. Editing the test file itself also breaks its own `-tool` sha line (`check_tool_line` refuses on it, executed). The skip is therefore correct and consistent, but its reason names two of the several causes.
- `bfs.py:97` (`Row` docstring) says "no shipped row has one" (an `item`); `V_BOTH_ROWS`, part of `SHIPPED_ROWS`, has two. True of `ROWS`, stale for the table P1 ships.
- `candidate_table("V-both")` is now the 1.2 table but `answer_build.build_answer(table=...)` still labels any explicit table `CANDIDATE_VERSION` (`test_engine_answer_v18.py:161` asserts it). Plan-conformant (P1-39d re-derives it); noted so that unit does not miss it.

## Rulings on the specific asks

- **(3) V-both rows vs spec2 §5.** Field by field, against the independent literal `V_BOTH_ROWS` in `test_engine_flow.py:572` and the spec prose: row 1 `function` -> `HAS_CALL ->` -> `end`, inc 0, labelled; row 2 callee-state `end` -> `HAS_CALL ->` -> `end`, inc 0, `where=callee-state`, labelled; rows 3, 4 `CallSite` `end` -> `CALLS ->`/`CALLS_NAME ->` -> `end:callee`, inc 1, labelled, `item="symbol"`, ungated. Matches §5 (not flag-gated, callee at `d` + call depth, no re-entry of `function`, no callee callers). Fingerprint-equal to HEAD's rows, so the move out of `flow.py` changed no row.
- **`CALLEE_ROWS`/`_run_kind` vs (b)-(g):** (b), (c), (d) read "in a run" correctly; (e) is exempted only where `run` is set; (f) closes both exits; 1.1 unchanged by construction (`runs` is `== "1.2"`). Open points: F1, F2, F3.
- **(4) Skip:** only `test_the_real_phase` is skipped; the pinned tool correctly refuses at HEAD (intended; see F5), and the committed result file is untouched and remains the record. A re-run needs its own `v2` tool line and record, as the plan says. `test_the_digest_and_the_candidate_measurement_run_over_the_store` still runs and passes.
- **(6) P1-45b interaction:** V20 reads every hop of every item and member (`_all_items`), so a labelled run hop is covered; a run whose callee hop keeps its own `name-match` needs no notice. `b_both` carries the notice and validates; removing the notice is not in P1-23d's tests, but V20's own tests (P1-45b) cover the coverage rule and the run's `flow-unavailable` hops go through the same loop. The labelled-last-hop shape (a run item whose last hop `fg` is `flow-unavailable`) is exactly `b_both`'s `items[1]`. No conflict found.

## What's solid

- The `"1.2"` pin is a literal and I reproduced it independently; `"1.1"` is untouched; the default walk, `definition()` and `ROWS` are unchanged, with tests that would redden if P1-39d's flip arrived early (mutants B10, B11 killed).
- The 17 `bfs`/`flow` mutants (row order, each added row's `inc`, `to`, `item`, `where`, `labelled`, `kinds`, the `SHIPPED_ROWS` order, the pin literal, `V-union` order, the dropped fourth row) all died. Most die on the single fingerprint-pin test, which is what that pin is for.
- The `_run_kind` argument and state mutants (state check dropped or inverted, `prev_run` passed as `None` or lost, run state leaking across items, each predecessor condition, 1.1 applied, the run never applied) all died.
- Real-output validation: the run hop's `state == "end"` strictness refuses nothing the walk produces.

## Open questions

- None blocking. The delegate's "1 equivalent" mutant is not identified in the ledger; if it is the (g) clause, F1 refutes it.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: F1 (this review).

- A clause of a spec'd validator that an implementation drops as "shadowed by" another check is re-derived by a planted input that passes the shadowing check and violates the dropped clause (here: equal id, different subject type). If the input is accepted, the clause is not shadowed.
