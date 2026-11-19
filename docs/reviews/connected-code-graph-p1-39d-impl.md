# Review: P1-39d implementation (engine flip to V-both `1.2`, the flow notice)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Uncommitted `git diff -- cpg/ccg/engine cpg/tests` (8 files: `answer_build.py`, `bfs.py`, `staleness.py`, `slice/test_engine_equivalence.py`, `test_engine_answer.py`, `test_engine_answer_v18.py`, `test_engine_bfs_guards.py`, `test_engine_flow.py`) against `docs/plans/connected-code-graph-p1-answers.md` `## P1-39d` and `docs/plans/connected-code-graph-answer-spec2.md` §1.1, §5, §5.1, §7, §8.1, §9 R-7/R-8. Context read: the P1-39 notice, P1-45b, P1-23d and P1-24b reviews. Not reviewed: `test_answer.py`/`test_render.py` (other hands), `docs/test-plans/`, P1-41 manifests.

**Verdict: approve with suggestions.** No blocker: the done-condition holds as executed. One spec/plan sentence contradicts the shipped trigger (F1), one test guard names the wrong table for "shipped" (F2), and the `flow_langs` default is a trap (F3).

CPG: considered, not relevant — a diff-scoped review of the engine's walk and builder, checked by running them; no call-graph question arose.

Run (cwd `cpg/`, `.venv/bin/python -m pytest`): `test_engine_flow.py test_engine_answer.py test_engine_answer_v18.py test_engine_bfs_guards.py` 456 passed; `tests/slice/test_engine_equivalence.py` 66 passed (the stored slice is present, so the five slice tests ran, not skipped); `tests/test_engine_*.py test_superset_dev_tool.py test_answer.py` 2057 passed, 130 deselected (live marker, unrun). `test_engine_*` + `test_render.py` in one process (622 passed) and `test_answer.py` + `test_render.py` (854 passed) were green at the time I ran them: **the engine diff does not contribute to the `test_render` ordering failure** (it touches neither `ccg.answer` nor `ccg.render`; the V18 population spy restores `answer.validate` in `finally`).

## Findings

### F1 (major, verified) §5.1's "a contract-point anchor yields no code node" is false for any contract anchor with a provider or consumer; the unit's test only covers the edge-less one
`-answer-spec2.md` §5.1 says "only an unbound config anchor or a contract-point anchor yields no code node, hence no notice"; plan P1-39d (4) repeats it. Executed: over `link_world` and `depends_world`, a contract anchor in the **impact** column lists 2 items and carries `notices` `[javascript, python]`; depends-on lists 0 and carries none. The engine is right by §5.1's own trigger (the visited set holds `CallSite`/`Symbol` nodes); the sentence is wrong. The test `test_an_unbound_config_anchor_and_a_contract_point_anchor_visit_no_code_node_and_carry_no_notice` uses a `cp1` with no edges, so it passes without pinning the claim, and no test pins the real behaviour.
Improvement: owner `architect` corrects the §5.1 bullet (and plan wording) to "an unbound config anchor, or a contract anchor whose walk reaches no `Symbol`/`CallSite`" before the spec-update check at close (root `REVIEW.md`: a spec contradicting shipped code is a blocker there). `tdd-engineer` adds one test: a contract anchor with a flag-false consumer, impact column, carries that language.

### F2 (major, verified by reading plus execution) `check_v18`'s "shipped" switch still points at `as-built`; V18 (c)/(d)/(e) with `CALLEE_ROWS` are never checked on random flag-mixed worlds (the implementer's flag, item 5)
`test_engine_flow.py` `check_v18(w, name == "as-built")` gates (c), (d), (e) and the no-`path`-hop rule behind `tables_shipped`; the shipped table is now `V-both`, so the shipped walk gets only (a)/(b), and the test name still says "the shipped table". A check that cannot fail for the table it claims is a finding (root `REVIEW.md`). Ruling: worth a test now, because the only builder-path populations (`test_engine_answer_v18.py`) force every unit to one flag, so mixed-flag worlds with callee runs are validated nowhere else.
No defect today: I built the default document for every anchor and both columns over 40 `mixed_world` seeds (419 documents, 138 with a `HAS_CALL ->` run, 62 with labels): all validated, with the same 96/89 `GraphDataError` "no Unit row" refusals as HEAD (pre-existing, from hexified random worlds, unrelated).
Improvement: in the same test, set `tables_shipped` for `V-both` and replace (c)/(e) with hand-written rules for it (a label's hop leaves `function` by a `FLOW_ROWS` row, or is a `CALLEE_ROWS` row of an `end` run; a hop leaving `function` is in `FLOW_ROWS`). Do not take expected values from `answer._run_kind`/`CALLEE_ROWS` alone (they would be the oracle). Keep `as-built` on the old rule (it is a subset). Name the mutant: drop `HAS_CALL ->` from `answer.CALLEE_ROWS` in a copy; the random test must redden.

### F3 (minor, verified) `Walk.flow_langs(reader, facts=None)`: the default is a trap
As a method taking the reader, it is a defensible departure from "`Walk`: new `flow_langs`, derived from `Walk.visited`" (a `Walk` holds no reader; the extra `units` read must go through the builder's reader to land in `recipe.queries`). But the default `facts=None` uses the walk's own facts, which hold no unit for the anchor unless a gated row left it: executed over the engine worlds, `w.flow_langs(reader)` raised `GraphDataError` for 70 of 89 symbol-anchor default walks. Only the builder passes its merged facts, so nothing breaks, but the signature advertises a usable default.
Improvement: make `facts` a required parameter (tests already pass it, or a walk-owned facts, in every case), or document that the default is only valid when the anchor unit is in `Walk.facts`.

Fail-closed rules, executed: flag-false unit with no language, missing facts entry, unit `None` and ghost unit (no `Unit` row) raise `GraphDataError` for each of `Symbol`/`CallSite`/`Param`/`Ret`/`Cond`; a flag-true unit with no language does not; `Link`/`ConfigEntry`/`ContractPoint`/`Unit` nodes with no unit do not. A symbol anchor always carries a unit (`_anchor_parts` fails first), so the merged-facts override cannot inject `None`.

### F4 (minor, verified by unfiltered grep over `cpg/ccg` and `cpg/tests`) stale prose that the flip made false, beyond the two carried items
The two carried docstrings are fixed (`answer_build.py` "1.1-or-later", "V1-V20"; `staleness.py` "1.1-or-later"). Still wrong after the flip:
- `ccg/engine/flow.py:20-22` ("until it does the shipped one is §5's"), `:118` and `:125` ("`as-built` is the shipped table (`bfs.ROWS`)") contradict `:128` (V-both "the shipped table") in one docstring.
- `tests/test_engine_answer.py:4` still cites "V1-V19"; `tests/test_engine_answer_v18.py` module docstring ("V17-V19", "the three candidate tables V-flow, V-both and V-union") and the name of `test_a_candidate_table_lists_at_least_what_the_shipped_one_does...` (it compares against `as-built`, now the caller-only table, not the shipped one).
- `ccg/engine/__init__.py:11` says V18 refuses "a candidate table's added hops" for P1-39c; true only for V-flow/V-union now.
Improvement: reword in the same commit; `flow.py` is in the plan's file list.

### F5 (minor, inferred) the relaxed read-count allowance is named, not pinned
`test_a_row_pair_that_shares_a_read...` now allows `has_out:CallSite` to repeat for the shipped table, impact column, each key exactly twice. The repeat is real: in `caller_chain(flow=False)` the walk reads `has_out:CallSite {"ids":["F"]}` at call 5 (F in `function`) and at call 14 (F again in `end:callee`), with no cross-level cache (`_Memo` is per expansion). It does not hide a defect today, but the allowance accepts that name for any ids; a mutant that re-reads the same ids a third time cross-level is caught only by the `== 2` cap. My within-level duplication mutants (R1-R3) survived because `_Memo` collapses them, so they say nothing about the allowance.
Improvement (optional): pin the two readers' ids (`F` twice, each in a distinct state), or accept and keep the cost note in the docstring, which already states it.

### F6 (nit, verified) redundancy in `_version` and `sorted(flow_langs)`
`_version` repeats `w.version()` in two branches; `try: return w.version() except WalkError: if table is None: raise; return CANDIDATE_VERSION` says it once. `sorted(flow_langs)` is redundant because `build_answer` returns `answer.canonicalize(doc)` and `validate` accepts any order (see the M7c row below); harmless, keep or drop.

## Executed checks

1. **Done-condition.** `test_the_builder_emits_schema_1_2...`, the V18 population (`table=None` = V-both, 1.2/1.2; `as-built` passed explicitly = 1.2 document with algorithm 1.1; nothing refused but V-flow/V-union, clause (c) only) and my own probe: every anchor of every engine world, both columns, **every per-unit flag assignment** (920 documents), all validated under V20/V18, all `"1.2"`/`"1.2"`, no `GraphDataError`. Independent oracle for the notice (a unit read straight from the graph rows, point units from owner edges, not from `facts`): `langs` equal to the expected set in all 920 (494 with a notice, 222 of them with no listed item). Slice walks: flag-true counterfactual pins unchanged under the default (so V-both adds nothing where every unit is flag-true), caller-only baseline kept with `table=bfs.ROWS`.
2. **1.0/1.1 documents.** The diff touches no validator; `test_answer.py` (goldens included) passed. The engine can no longer emit `"1.1"`; the only `"1.1"` it emits is `algorithm.version` for an explicit `bfs.ROWS`.
3. **`_version`.** `table=None`: the shipped definition's version (unmapped raises `WalkError` as before); explicit: mapped version, else `0.0`. V-flow/V-union stay refused by V18 (c); their documents carry `0.0` (population pin).
4. **Re-derived tests (from the diff).** Depth-bound: strengthened, not weakened; `unexplored` = 3 is exactly the three stored cut edges (`cD`→B, `cE`→D, `cF3`→F3, read from `Walk.cut`), listed callee F2 is `possible`, F3 absent; mutants "labelled cuts not counted" and "callee rows not counted" both killed. Read-count: see F5. Slice pins 25/96/20 and 15/59/18 are **observed values** (the docstring says so), with the (status, reason, distance) Counter alongside, and the as-built population is still guarded (`table=bfs.ROWS` keeps 19/38/6 and 10/21/9; the flag-true counterfactual pin `SHIPPED_OBSERVED` is unchanged). V-flow test: default-table assertion is `>= {"X"}` plus `D` absent, weaker than the old `== {"X"}`, but the same equality remains on `table=bfs.ROWS`. V-both test: new equality of the default walk's items against `candidate_table("V-both")` (stronger). Guards tests: fingerprint literals unchanged, default now `PINNED_1_2`.
5. **`check_v18`**: F2.
6. **Docstrings**: F4.
7. **Mutation sweep** in a scratch copy (suite baseline 1210 passed there; each mutant against the 11 engine files + `test_answer.py`, never the working tree). Killed: anchor omitted from visited; flag-true language included (`if True`); zero-item answer loses the notice; `flow_langs(reader)` without facts; `flow_langs(reader, w.facts)`; anchor facts replaced by `{}`; drop each of `Symbol`/`CallSite`/`Param`/`Ret`/`Cond`; add each of `Link`/`ConfigEntry`/`ContractPoint`/`Unit` (argument-set mutants: each by its own parametrized test); language taken from the unit id; only the first language kept; `units.info.update` removed; `ensure` skipped; flag-false no-language raise removed; raise moved to flag-true; missing facts skipped; notice code renamed; empty-list `notices`; notice only in the impact column; `facts_bytes` dropping `notices`; version `table is None` swapped; explicit-table fallback `1.1`; label `"1.1"`; walk and `definition` default back to `ROWS`; callee-state filter off; two cut-count mutants. Survived: `sorted(flow_langs)` -> `list(...)` (**equivalent**, re-derived by counter-shape: a 4-language world run under `PYTHONHASHSEED` 0-5 gave `['go','javascript','python','rust']` every time, because `canonicalize` sorts `Notice.langs`; `validate` before it accepts any order), and no-op comment control. The first sweep pass spuriously killed every `answer_build.py` mutant because my copy lacked `docs/` (a baseline failure); rerun after copying it. The claimed 24/25 was not re-derived by name; the mutants above are my own, including the argument and set-membership ones.
8. **Impacts outside scope.** P1-41 manifests and `docs/test-plans/connected-code-graph-p1.md` carry no `"1.1"` default-table pin (grep); anything recording `recipe.algorithm.version`/`answerSchemaVersion` of the default walk is now `1.2`.

## What's solid

`flow_langs` is small, set-valued, and its node-type set, anchor inclusion and fail-closed rules are each pinned by a test that dies under the matching mutant. The re-derived depth-bound test pins edges, not just a count. The explicit-table version rule is pinned both ways. The flip is one commit-sized change (default table, label, `notices`) and the shipped walk validates over every flag assignment I could generate.

## Open questions

- F1: is the intended contract that a contract anchor with providers/consumers carries the notice (the engine's behaviour, and the principle of §5.1), or that it must not? I assume the former.

## Proposed REVIEW.md lines

None: F1-F2 are instances of existing lines (a spec contradicting shipped code; a check that cannot fail for the thing it claims).
