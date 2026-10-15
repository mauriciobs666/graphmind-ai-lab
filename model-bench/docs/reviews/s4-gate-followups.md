# S4 code-gate follow-ups — Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** —

Reviews the units of `model-bench/docs/plans/s4-gate-followups-coordination.md`, one `## Pass`
section per unit gate.

## Pass 1 — FX2 (report.py counters)

**Scope.** The uncommitted diff `git diff 5d106332 -- modelbench/report.py tests/test_report.py`
(+156/+259 lines), closing the `docs/BACKLOG.md` item "`report.py` never renders
`luckyPassCount`/`parseFailures`/`malformedSpecCount`/`schemaViolationCount`" from
`docs/reviews/small-model-benchmarking-s4.md`. Baseline contract: the `luckyPassCount` obligation
in `modelbench/scoring/extraction.py::aggregate()` docstring, read with `score_item`'s unanswerable
branch (`extraction.py:266-286`) and the fields in `modelbench/results.py:621-641`. Not reviewed:
`tests/test_tools_exec.py` (FX1).

**Verdict: approve with suggestions.** No blocker or major. The attribution is exact for every
record the scorer can produce, the all-zero hide rule is honest, and real stored runs render as
expected. The suggestions below cover three things: an inconsistent record whose problem goes
unreported, a latent hole that opens if a pack ever makes a shape metric a verdict metric, and a
test gap on the gloss prose.

CPG: considered, not relevant — no `cpg_model-bench` graph is loaded (coordination doc); the review
traced the two-module data flow by reading the code directly.

**What I executed** (all outside the repo, in the session scratchpad):

- `pytest -q tests/test_report.py`: 191 passed. `ruff check` on both files is clean. Teco's
  full-suite run was not repeated.
- Rendered `rank_report` (all models) and `compare_report` (first two models) for all five packs
  from the stored `results/runs/`, using the new code and the `5d106332` code (`git archive`
  extracted to scratch). Then diffed the outputs. For details, see Appendix A.
- Rendered `compare_report`/`rank_report` with a pack whose `verdictMetrics` includes
  `exactMatchByRelationshipTraversal` (the test fixtures' `_nlq_arm`). See F2.
- Fed `compare_report` a stored `luckyPassCount` smaller than its items' attribution. See F1.
- Mutated `_SCORER_COUNT_GLOSS` in-process and ran the 15 new tests. See F3.

### Premises tested

- **(a) Per-shape attribution matches the scorer: verified by reading both modules.**
  - `score_item` sets `detail["luckyPass"] = True` only when `correct` is true. In the same return
    it sets `scoreable[shape_metric] = True` and `counts[shape_metric] = int(correct) = 1`
    (`extraction.py:279-286`).
  - Each item carries exactly one `exactMatchBy*` metric. An answerable item never carries
    `luckyPass`.
  - `aggregate()` sums `counts[metric]` over the items that declare the metric. `scored_outcome`
    returns `counts[metric] > 0`, and counts are 0/1 here, so the two agree.
  - Conclusion: every lucky item is counted in exactly one shape's k and is attributed to exactly
    that shape. No success in a shape's k comes from an unanswerable item without a `luckyPass`
    flag.
  - Every `exactMatchBy*` metric that any item declares is in `byShape`, so
    `_lucky_shape_metrics` cannot miss one.
  - The `‡` path can only be reached by a record that the scorer itself never writes (a hand-built
    or corrupted record).
- **(b) Verified.** `unanswerableAbstainRate` lives in `byShape` (`extraction.py:388`) but fails
  the `exactMatchBy` prefix test, so it is never marked. `layer1ExactMatchRate` is not in
  `byShape`. A lucky item that also abstained correctly counts in the abstain k as a genuine
  abstention, and the docstring's obligation covers only the shape rate.
- **(c) Verified on real data.** Of the 10 reports rendered from stored runs, 7 are byte-identical
  to baseline. The other 3 differ only by the inserted `## Scorer counts` block (Appendix A).
  - The hide rule is honest. The block's absence makes no claim, and baseline reports omitted
    these counts entirely.
  - Grounding's `parseFailures` is a structural 0 (`grounding.py:276`, "stays `0` always"). The
    hide rule is what keeps it from rendering as if it were a measurement.
  - One limit: an extraction or classification pair whose counts are all genuinely measured zeros
    also renders nothing. See N1.
- **(d) Verified.** `rank_report` rebinds `runs` to the consistent subset (`report.py:1740`) before
  the new call at the tail.
- **† and the numeric cells.** The mark is appended to the k/n string only. The rate cell
  (`metric.successes / metric.n`) and the Wilson interval are still computed from the
  `BinaryMetric`'s numbers, so neither cell is affected (confirmed in the rendered rows, F2
  excerpt). The `not metric.n` and no-observation branches skip the mark, and a shape with n=0
  cannot hold a lucky pass anyway.

### Findings

**F1 — minor. When the stored `luckyPassCount` is below what the items attribute, the report
contradicts itself without saying so.**

- `_lucky_by_shape` returns `max(0, agg.luckyPassCount - sum(by_shape.values()))`
  (`report.py`, new code).
- An under-attributed record (stored count above the items' total) gets the named `‡` footnote.
- An over-attributed record (stored count below the items' total) is clamped to 0 with no output.
- Executed: three lucky `relationship-traversal` items with stored `luckyPassCount=1` rendered
  `| cand | exactMatchByRelationshipTraversal | 3/4 † (3 lucky) |` in the Arms table and
  `` `luckyPassCount` 1 `` in `## Scorer counts`.
- With stored `luckyPassCount=0`, the † still prints and its footnote cites `luckyPassCount`, but
  the Scorer-counts block is suppressed.
- Marking from items is the safe direction. The problem is that two numbers in one document
  disagree and nothing names the disagreement. `_aggregate_item_mismatches` is scoped to verdict
  metrics, so it does not catch this either.
- Suggestion (`modelbench/report.py`): make the `‡` condition `sum(by_shape.values()) !=
  agg.luckyPassCount` and print both numbers in that footnote. Add a test in `tests/test_report.py`
  using the over-attributed fixture above.
- Reachable only from an inconsistent record, hence minor.

**F2 — minor (latent). The lucky label stops at the descriptive tables.**

- When a pack makes an `exactMatchBy*` metric a verdict metric, every downstream use of that
  metric is unlabelled.
  - The `compare_report` `## Verdicts` section for that metric says *"The effect-size interval
    [8.5, 95.4] pp … (b=3, c=0, p=0.250)"*, and in this fixture all three discordant wins are
    lucky passes.
  - The Holm table row is unlabelled.
  - `rank_report`'s reference-anchored family prints `| cand | +75.0 pp | … |`.
  - The `<!-- rank-report: … metric=exactMatchByRelationshipTraversal top=cand value=0.7500 … -->`
    marker is unlabelled, and `scripts/consolidate_sweep_reports.py` carries it into the sweep
    consolidation with no label (Appendix B).
- **It cannot be reached with the shipped pack.** `packs/nlq-structured-query/pack.json` declares
  `verdictMetrics: ["layer1ExactMatchRate"]` and the same headline, so `rank_report` never renders
  a shape table and `compare_report` lists shapes only as exploratory.
- Nothing prevents a future pack from doing it, though. The docstring obligation says "any future
  `report.py` rendering", and a verdict whose discordant pairs are lucky passes is the case where
  a missing label matters most.
- Suggestion (`modelbench/report.py`): when a verdict or reference section renders an
  `exactMatchBy*` metric and any arm has lucky passes in it, emit one line under that section's
  heading that reuses the † footnote. Alternatively, record the scope limit (descriptive tables
  only) next to `_LUCKY_PASS_METRIC_PREFIX` and in the backlog closeout, so the next pack author
  knows. Which of the two is a design choice for the owner.

**F3 — minor. Gloss text that misstates a field's meaning stays green.**

- `test_compare_report_prints_each_counter_from_its_own_field` asserts the legend as
  `` f"`{name}`: {_SCORER_COUNT_GLOSS[name]}" in md ``. That reads the expected text out of the
  table under test (AGENTS.md: "a value the test reads back out of the table is asserted against
  itself").
- `test_every_rendered_counter_has_a_gloss_and_no_gloss_is_orphaned` pins only the keys.
- Executed: swapping the `malformedSpecCount`/`schemaViolationCount` glosses (Layer A ↔ Layer B)
  and rewriting `parseFailures` to "replies the scorer parsed fine" still gives `15 passed`.
- The values counts are pinned correctly. The fixture's 2/3/5/4 values kill any field swap, so
  this is about the prose only.
- Suggestion (`tests/test_report.py`): assert one literal discriminating fragment per field, e.g.
  ``"`malformedSpecCount`: specs that failed structural validation (Layer A)"``. That pins the
  table's values (AGENTS.md's third mutation, a value moved to another key) against a second
  declaration.

**N1 — nit. All-zero hiding is decided per report, not per kind.**

- An extraction or classification comparison where both arms measured 0 on every counter renders
  no block, which reads the same as "this harness doesn't print counters".
- This is honest (no claim is made) and matches baseline.
- The real reason for the rule is the one kind whose zero is structural, grounding. A kind-level
  rule would be more precise: always render for `classification`/`extraction`, never for
  `grounding`'s structural zero. The cost is that the all-zero byte-identity premise (c) would no
  longer hold.
- Owner's call. Leaving it as is is defensible.

**N2 — nit. The `luckyPassCount` gloss says "(marked †)" in reports where no † can appear.**

- This includes every `rank_report` on the shipped pack (headline-only, no shape rows) and the
  `no_marker` test's own output, which needs `md.replace("(marked †)", "")` to pass.
- Suggestion (`modelbench/report.py`): "(marked † where its shape row is shown)", or drop the
  parenthetical.

### What's solid

- Attribution reads only declared data, through `scored_outcome`, which is the one reader of item
  outcomes. Nothing is inferred, and the fail-safe direction (`‡` marks every shape row rather than
  none) is the right one.
- `_SCORER_COUNTS` is bound to the dataclasses' own `int` fields by equality. A shrink, a widen,
  and a key move each fail the test, and the one hand-made exemption (`ClassificationAggregates.n`)
  is named with its reason.
- The fixture builds its aggregates with the real `extraction.aggregate()` and uses distinct
  counter values, so field swaps are caught.
- Real-data renders are byte-identical wherever all counts are zero. Nothing outside the new block
  moved.
- The block and footnotes sit after the Arms table and before Speed/Verdicts, and are explicitly
  labelled "never a rate, never in a verdict".

### Open questions

- **Manual impact (for teco).** `docs/manuals/small-model-benchmarking.md` §7/§7a describe the
  `compare`/`rank` reports only at the level of what each command gives you. They name no report
  sections or k/n cells, so nothing in the manual goes wrong. A one-line mention of the new
  `## Scorer counts` block is optional and is `tico`'s call.
- **Defaulted fields.** `_aggregates_from_dict` builds with `cls(**…)`, so a record lacking
  `luckyPassCount` would load as 0 and now *render* as a measured `0`. No stored record lacks the
  field: all 18 `nlq-structured-query` records are pack 1.2.0 and carry all four counters. The
  field dates from `acfe7a8f`, the same day as the pack. This is pre-existing `results.py`
  behaviour that the new renderer makes visible for the first time, so no action is needed unless
  pre-A3 records are ever imported.

## Appendix

### A. Real-data render diff (new vs `5d106332`)

| report | result |
|---|---|
| chat-responder compare / rank | byte-identical |
| embedder compare / rank | byte-identical |
| tool-caller compare / rank | byte-identical |
| guard-judge compare | byte-identical (first two arms have `parseFailures` 0) |
| guard-judge rank | `+## Scorer counts` block only (e.g. `prism-ml/bonsai-27b: parseFailures 85`) |
| nlq-structured-query compare / rank | `+## Scorer counts` block only; all stored `luckyPassCount` = 0, so no Arms-row change |

### B. Verdict-section rendering with a shape verdict metric (F2)

Fixture: `_nlq_pack(verdicts=("layer1ExactMatchRate", "exactMatchByRelationshipTraversal"))`,
`cand` with 3 lucky `relationship-traversal` passes, `incumbent` with none.

```
| cand | exactMatchByRelationshipTraversal | 3/4 † (3 lucky) | 0.750 | [0.301, 0.954] |   <- Arms: labelled
### exactMatchByRelationshipTraversal
Not distinguishable … The effect-size interval [8.5, 95.4] pp … (b=3, c=0, p=0.250).             <- unlabelled
| exactMatchByRelationshipTraversal | 0.250 | 0.0250 | not distinguishable |                       <- Holm row, unlabelled
<!-- rank-report: … metric=exactMatchByRelationshipTraversal top=cand value=0.7500 … -->          <- unlabelled
| cand | +75.0 pp | [8.5, 95.4] pp | 0.0250 | not distinguishable |                                 <- rank reference family, unlabelled
```

## Pass 1 — FX1 (exec.py None sort key, deferred)

**Scope.** The uncommitted diff `git diff 5d106332 -- tests/test_tools_exec.py` (+44 lines). It
adds one parametrized test, `test_order_by_a_sparse_property_places_nulls_the_way_falkordb_does`,
marked `xfail(strict=True, raises=TypeError)`. The stakeholder rulings I took as given: the
`exec.py` fix is deferred to the next deliberate `packVersion` bump, and nulls are ordered the way
FalkorDB orders them rather than refused with `SchemaViolationError`. Baselines:
`packs/nlq-structured-query/tools/exec.py:312-383` and the Cypher that
`falkor-chat/server/falkorchat/querygen.py::compile` emits for the same spec (`:398-449`). Not
reviewed: the `report.py`/`test_report.py` files (FX2, edited concurrently; its 4 current
failures in the full suite belong to that in-flight work).

**Verdict: approve with suggestions.** The diff itself is correct. The one finding at major
severity, M1, is a coordination issue for FX3 rather than a defect in the test.

CPG: considered, not relevant — no `cpg_model-bench` graph is loaded, and the code path involved
is one function plus one generated Cypher string, both read directly.

### Premises tested

- **The test reaches the sort's `None` path: verified.**
  - Under `--runxfail`, both parameters raise
    `TypeError: '<' not supported between instances of 'NoneType' and 'float'` at
    `exec.py:377`, the `deduped.sort(...)` line.
  - `MalformedSpecError`/`SchemaViolationError` are `ValueError` subclasses (`exec.py:49,56`). So
    a validation failure earlier in the path is not the expected `TypeError` and would surface as
    a plain failure.
- **A correct fix makes the test XPASS, which strict mode turns into a failure: verified by
  mutation.** All mutants ran on scratch copies of `exec.py` plus the test file. The real `packs/`
  and `tests/` trees were not touched. Results (`fx1/mut.sh`):

  | mutant (sort line in `exec.py`) | ASC | DESC |
  |---|---|---|
  | unchanged | XFAIL | XFAIL |
  | correct: key `(v is None, v)`, `reverse` on DESC | `[XPASS(strict)]` FAIL | `[XPASS(strict)]` FAIL |
  | nulls first ASC / last DESC | AssertionError | AssertionError |
  | drop null rows | AssertionError | AssertionError |
  | `None` coerced to `0` | AssertionError | AssertionError |
  | nulls always last (ignores direction for nulls) | `[XPASS(strict)]` FAIL | AssertionError |
  | right null key, `order_dir` ignored | `[XPASS(strict)]` FAIL | AssertionError |
  | no sort at all | AssertionError | AssertionError |
  | an unrelated `raise TypeError` | XFAIL | XFAIL |

  Every wrong fix fails as a plain assertion rather than being hidden by the marker. The last row
  is the marker's inherent blind spot, covered in N1.
- **Pack hash unchanged: verified.** `content_hash(Path("packs/nlq-structured-query"))` is
  `8f975842302b…f3ccf`, the only `packContentHash` among the 18 stored `nlq-structured-query`
  records. `git status --short packs/` is empty after the runs. The `__pycache__` that
  `spec_from_file_location` may write is excluded from the hash by `_pack_relative_paths`.
- **"Nulls last ASC / first DESC" matches the real query for *this* projection: verified live
  (read-only).**
  - For this spec (`returns: ["p.name"]`, `order_by: "p.price"`), `querygen.compile` takes the
    `needs_tuple_distinct` branch (`querygen.py:404-440`). The order key is not in the projection,
    so the Cypher is `MATCH (p:Product) WITH DISTINCT p.name AS c0, p.price AS c1 ORDER BY c1
    <dir> LIMIT $limit RETURN c0 AS `p.name``. This differs from the plain `UNWIND … ORDER BY x`
    that teco probed.
  - That exact clause shape over the fixture's four rows, as `UNWIND`ed maps with `B`'s `price`
    key absent, returned `C, A, D, B` for ASC and `B, D, A, C` for DESC on `ws:test`. Both match
    the test's expected lists. The naive `RETURN DISTINCT … ORDER BY p.price DESC` returns the
    same order.
  - **A missing node property versus a map null:** FalkorDB cannot store a property set to null,
    so "missing" is the only real case. On `ws:nlq-eval`, `Message.toolsUsed` is present on 8 of
    15 nodes. The same `WITH DISTINCT … ORDER BY c1` shape put all 7 genuinely missing values last
    on ASC and first on DESC.
  - `exec.py` reads `row.get(prop)`, so a row lacking the key and a row with `"price": null` are
    both `None`. The fixture's absent key models the real case faithfully.

### Findings

**M1 — major (coordination, for FX3 rather than this diff). Following the coordination doc,
FX3 would delete the backlog item this test's marker cites, and the item now describes a design
that was rejected.**

- `s4-gate-followups-coordination.md` § Documentation impact says FX3 should "remove all three
  items".
- The test's `xfail` reason sends the reader to `docs/BACKLOG.md` for the deferred fix. Once the
  item is deleted, that pointer leads nowhere, and nothing records the obligation to remove the
  marker at the next `packVersion` bump. The strict XPASS still fires then, but only if somebody
  fixes `exec.py` at all.
- The item's current body (`docs/BACKLOG.md:27-32`) also says the fix is "a graceful
  `SchemaViolationError`" via "a one-line guard". Both were overruled.
- Suggestion (`docs/BACKLOG.md`, FX3): rewrite the item in place, not remove it. The rewrite
  should say:
  - the fix is deferred to the next deliberate `nlq-structured-query` `packVersion` bump, because
    editing `exec.py` changes the `contentHash` that pairs the 18 `1.2.0` runs;
  - the chosen semantics are FalkorDB's null placement (last ASC, first DESC), live-verified on
    the `WITH DISTINCT … ORDER BY` shape `querygen.compile` emits;
  - the regression test already exists as a strict `xfail` in `tests/test_tools_exec.py`, and its
    marker comes off with the fix.
- Also update the coordination doc's line ("remove all three" → "remove two, rewrite the
  `exec.py` one").

**N1 — nit. `raises=TypeError` is a type match, not a site match.**

- Any `TypeError` anywhere in `compile_and_execute` keeps the test green as XFAIL. That includes a
  later fix that crashes for a different reason (the "unrelated `raise TypeError`" mutant row).
- pytest's `xfail` takes no message `match`. Pinning the site would mean an in-body
  `pytest.raises(TypeError, match="NoneType")`, which loses strict-XPASS. That is a worse trade.
- Leave as is. The marker has to come off when the fix lands anyway, and the assertion then
  governs.

**N2 — nit, for whoever writes the eventual fix rather than for this diff. Don't assert an order
among tied rows.**

- On the live query, FalkorDB's order among tied rows was not a mirror image between ASC and
  DESC. The 7 null `Message` rows came back `117,127,132,112,122,137,143` for ASC and
  `132,137,122,143,117,127,112` for DESC.
- Python's stable sort will produce some deterministic order among nulls that FalkorDB does not
  promise.
- This fixture has exactly one null and no tied prices, so it is unaffected. A future test with
  two or more nulls, or with tied non-null keys, should compare the set within each tie group, not
  the sequence.

### What's solid

- The fixture is minimal and has a single meaning. The one null sits mid-list, and as the comment
  claims, the "no sort" and "drop null" mutants both fail.
- `strict=True` plus `raises=TypeError` is the right pair for a deferred known gap. It cannot pass
  silently once the gap is fixed, and it cannot hide a wrong fix.
- The reason string names where the deferral is recorded and what action is owed.
- The spec passes both validation layers against the file's existing `_SCHEMA`, so the
  `TypeError` comes from execution and not from the fixture.
- `ruff` is clean. The full suite shows `2 xfailed` from this test, and its only failures are
  FX2's in-flight `test_report.py` work.

### Open questions

None for the stakeholder. M1 is teco's to route to FX3.

## Pass 2 — FX2

**Scope.** The revised `git diff 5d106332 -- modelbench/report.py tests/test_report.py`, checked
against the Pass 1 FX2 findings. I executed everything below against the working tree and wrote
only to the scratchpad.

**Verdict: approve.**

**Executed:**
- The full suite gives `1817 passed, 3 deselected, 2 xfailed`, and `ruff check .` is clean.
- Re-rendered all ten real-data reports. Eight match the Pass 1 renders (seven of those still match
  `5d106332` byte for byte). The two nlq reports differ from Pass 1 only by the dropped
  "(marked †)", and no real report prints a `‡`.
- Ran the Pass 1 interaction probes (`p2.py`) and re-ran the F3 gloss mutants (`mut_gloss2.py`).

**Prior findings:**
- **F1: fixed.** My Pass 1 probe (stored 1, items place 3) now marks every `exactMatchBy*` row
  of the arm `‡`. The new footnote reads "(cand: stored `luckyPassCount` 1, items place 3)", and
  the Scorer-counts block keeps the stored 1. The two numbers disagree and the report now names the
  disagreement. The recommendation was sound, and the implementation follows it in both directions.
- **F2: superseded.** Teco ruled it a documented scope limit, filed to BACKLOG. Not re-argued.
- **F3: fixed.** Each Pass 1 mutant now reddens: the Layer A↔B swap, the `parseFailures` rewrite,
  and a new `luckyPassCount` rewrite each give `1 failed`.
- **N1: kept** per teco's ruling.
- **N2: fixed by deleting the parenthetical.** That is better than my rewording, which still
  pointed at a mark the report might never print. Conceded.

**Checking the three fixes against each other:**
- **The disagreement `‡` with a hidden block.** Stored 0, items place 3, every other counter 0:
  the Scorer-counts block is suppressed. The `‡` footnote still prints "stored 0, items place 3",
  so the disagreement is named. Pinned by
  `test_the_disagreement_is_named_even_when_the_counts_block_is_hidden`.
- **The disagreement `‡` with rank's verdict-metric-only marking.** When shape metrics are ranked
  members, both disagreeing arms are marked, and the footnote lists both arms with their two
  numbers. On a headline-only pack (the shipped `nlq-structured-query` configuration) no shape row
  is ranked, so no `‡` can print. See N3.
- **Negative control** (one record used as both arms) renders the `‡` legend once, deduplicated by
  label.
- **F2's scope limit.** Unaffected. The `‡`, like the `†`, stays on descriptive k/n cells only.

### New findings

**N3 — nit. A headline-only `rank_report` never names an inconsistent `luckyPassCount`.**

- Executed with the shipped pack's metric shape (headline `layer1ExactMatchRate`):
  - stored 0, items place 3: nothing at all is printed;
  - stored 5, items place 0: the block prints `` `luckyPassCount` 5 `` with no qualifier.
- Nothing printed is false relative to the stored record. Lucky passes never enter
  `layer1ExactMatchRate`, the only metric ranked there. The scorer cannot produce this record.
  Rank's marking is per-row by design, and this is the same boundary as F2.
- Optional fix, in `modelbench/report.py`: suffix the value in `_render_scorer_counts` with `‡`
  when it disagrees with `_lucky_by_shape`.

**N4 — nit. `_LUCKY_PASS_DISAGREEMENT_FOOTNOTE` is worded for one arm** ("The arm's stored …",
"an inconsistent record", "that arm"), but it can list several arms. Executed rank output:
"(cand: stored `luckyPassCount` 0, items place 4; incumbent: stored `luckyPassCount` 7, items place
0)". The fix is wording only, in `modelbench/report.py`.
