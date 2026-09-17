# `model-bench` S4 — cumulative code + methodology-consistency gate

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (S4)

## Scope & verdict

Reviewed the whole S4 diff, `git diff 403492e..HEAD -- model-bench docs/plans/small-model-benchmarking-s4-spec.md docs/plans/small-model-benchmarking-ml.md`
(37 files, ~14.5k insertions), against `docs/plans/small-model-benchmarking-s4-spec.md` (as
corrected by U133b) and `docs/plans/small-model-benchmarking-ml.md` (as corrected by U133c). This
is the single cumulative code gate for the four units that deferred their own gate to it — U131
(`guard-judge-understanding` pack + `scoring/classification.py`), U133 (`nlq-structured-query`
pack + `scoring/extraction.py` offline build), U133d (the A3 exploratory-scoring/`luckyPass`
correction), U134 (`refresh_golden.py` CLI wiring + the live proof run, S4 close) — plus the
cross-unit interaction between U133/U133d/U134 on the shared answerability/scoring path, per the
brief. U129/U130/U132/U133a/U133b/U133c were `teco`-verified directly or are not code and are not
re-litigated here except where their rulings are load-bearing for judging the code (the 34/6 split,
the A3 contract).

**Verdict: approve.**

**CPG: considered, not relevant** — no `cpg_model-bench` graph is loaded (checked live by `teco`
per the brief), and this is greenfield feature work on new files with no CPG to cross-reference.

Verified myself, not taken on the ledger's word:
- Suite: `1450 passed, 1 pre-existing failure (`tests/test_convo.py::test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5`, the named S5 tripwire — confirmed by reading its own docstring, it is the same failure, not a second one hiding behind it), 3 deselected`. `ruff check .`: clean.
- The `nlq-structured-query` 34/6 split: `items.jsonl`'s `answerable: false` rows are exactly
  `nlq-34..nlq-39` (34 `true`, matches the mandated `docs/reviews/nlq-conflicting-facts-answerability-ml.md`
  ruling and U133c's revised `-ml` fact table, not the plan's stale "4").
- The stored live run (`results/runs/nlq-structured-query-qwen_qwen3-4b-2507-2026-09-12T02:10:01Z.json`)
  and its rendered report (`reports/nlq-structured-query-20260911-01.md`) reproduce every figure
  `HISTORY.md`'s S4 Step 4 entry and U134's ledger row claim, number for number: `layer1ExactMatchRate`
  34/34, `exactMatchByRelationshipTraversal` 0/3, `exactMatchByConflictingFacts` 0/1,
  `unanswerableAbstainRate` 1/4, `schemaViolationCount=2`, `luckyPassCount=0`, `parseFailures=0`,
  `malformedSpecCount=0`.
- `cli.py` has zero diff in this range — `--strict`'s unconditional `NotImplementedError` is
  unchanged and confirmed still the same deliberate, spec-named deferral it was before S4.
- Unfiltered grep across `model-bench/` and the four `docs/plans/small-model-benchmarking*.md`
  files for the relationship-traversal-item-count claim: every live occurrence of "4
  relationship-traversal" is now paired with "+ 2 conflicting-facts" / "6 of 40|39", i.e. it is the
  correct *sub-count* of the corrected 6, not a stray reversion to the old "4 total" claim. The one
  place that still reads as the old, uncorrected framing — `docs/plans/small-model-benchmarking.md`
  §3.8.3 itself (the top-level plan, which calls the 4 relationship-traversal items the only
  unanswerable bucket and treats `conflicting-facts` as answerable-with-an-exception) — is a
  **known, reasoned, already-disclosed** state: U133b's own ledger row records the deliberate
  decision not to touch it, because it was already executed against S1-S3 and collision rule 5
  routes the correction to the s4-spec (which `Extends` it) instead. Not a new finding.

## Findings

None at blocker or major severity. Two minor observations for the record; neither should hold up
S4's stage close.

**Minor — `luckyPassCount`/`parseFailures`/`malformedSpecCount`/`schemaViolationCount` are not
rendered anywhere in `report.py`, so a future non-zero `luckyPassCount` would surface in the Arms
table's raw `k/n` for that shape with no visual distinction from a genuine pass.**
`extraction.py:325-328`'s own docstring already names this obligation ("any future `report.py`
rendering of `luckyPassCount` must not let the shape rate stand alone unlabelled"), and
`HISTORY.md`'s S4 Step 4 entry confirms the gap is real today (`grep -n "parseFailures"
modelbench/report.py` — zero hits) and explicitly out of scope for this stage. On the one live run
that exists, `luckyPassCount=0`, so nothing is currently misrepresented. This is not a defect in
S4's diff — it is a correctly-scoped, self-documented deferral — but whoever eventually wires
`report.py`'s rendering of these four fields should add a footnote/asterisk on a shape's `k/n` line
whenever that shape's `luckyPassCount` contribution is non-zero, rather than a bare rate. Suggest
tracking it as a named `docs/BACKLOG.md` item rather than leaving it to be rediscovered.

**Minor — `tools/exec.py:376-377`'s `order_by` sort has no defensive handling for a `None`-valued
sort key mixed with non-`None` values, and no test exercises it.** `deduped.sort(key=lambda r:
r.get(order_prop), reverse=...)` raises `TypeError` in Python 3 if `order_prop` is populated on
some rows and absent/`None` on others. Checked against the real data: all three `order_by` items in
`reference_specs.json` (`nlq-16`, `nlq-17`, `nlq-20`) sort on `p.price` for `Product`, which is
never null in `packs/nlq-structured-query/tables.json`'s catalog, so this is not reachable by any
shipped golden item today. Low priority, but worth a one-line guard (`key=lambda r: (r.get(order_prop) is None, r.get(order_prop))`
or similar) plus a regression test the next time `tables.json`'s catalog half gains a sparse
property, so a future data change doesn't turn into an uncaught `TypeError` at run time instead of
a graceful `SchemaViolationError` or documented behavior.

## What's solid

- **U133/U133d/U134's three-way interaction is exactly right, and the test suite proves the
  interaction rather than each unit's isolated contract.** `test_scoring_extraction.py`'s
  `test_by_shape_pools_answerable_and_unanswerable_scores_of_the_same_shape` and
  `test_lucky_pass_count_is_additive_with_its_shape_metric_not_an_alternative` directly exercise
  the cross-unit seam the brief flagged (A3's pooling rule applied to a hypothetical mixed shape,
  and the additive-not-alternative relationship between a shape's success count and
  `luckyPassCount`) — not just each unit's own isolated fixture. The live proof run corroborates it
  on real data with `luckyPassCount=0` throughout, as expected given the six reference specs are
  all honest, subject-filtered specs (per U133a's F-4 finding and U133d's independent
  re-verification of that fact).
- **`tools/exec.py`'s Layer A/Layer B split is a careful, well-documented port** of `querygen.py`'s
  validation surface, with the de-duplication design note (`RETURN DISTINCT` parity, aggregate
  queries exempt) explicitly justified against the real `ws:nlq-eval` duplicate-row shape rather
  than asserted.
- **`refresh_golden.py`'s `run_stamp_answerability` computes answerability dynamically** by
  actually compiling and executing each item's reference spec against the live `tables.json`
  snapshot, rather than hard-coding the 34/6 split anywhere — the split is a *consequence* of
  running real specs against real data, which is what makes the reproduction in this review (and
  in U134's own re-run) a genuine check rather than a restatement.
- **Test quality across the four deferred units is consistently non-tautological.** Spot-checked
  `test_scoring_classification.py` (57 tests) and `test_tools_exec.py` (48 tests): boundary-value
  assertions (epsilon-exact numeric equality, own-line JSON parsing's line-ownership rule,
  `filters` length/`limit` range edges), not shape-only checks that a function merely returns
  without erroring.
- **`HISTORY.md`'s S4 entries are accurate against the artifacts they describe** — every number
  quoted in the Step 3/A3-correction/Step 4 entries was independently reproduced against the
  stored run JSON and rendered report in this pass.
- **`cli.py --strict` and the S5 tripwire are both exactly where S4 left them** — zero diff on the
  former, identical single failure on the latter.

## Open questions

None. One dead test helper (`test_scoring_extraction.py:34`'s `_row` function, defined but never
called) was noticed in passing; too trivial to rank as a finding, but worth a one-line cleanup
whenever that file is next touched.
