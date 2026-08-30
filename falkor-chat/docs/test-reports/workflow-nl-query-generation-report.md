# `workflow-nl-query-generation` — Golden-Set Evaluation Report

> **Status:** active · **Owner:** `tdd-engineer` · **Tracks:** K-055 (M6)

## Summary

**Current state (2026-08-30 re-run, post-fix, unit U29g): Layer 1 (the FR-4/AC-4 gate) still
FAILS, narrowly, and the AC-2 `knowledge_base`-subset gate still FAILS — but every remaining
point of failure is now confined to the two shapes (`relationship-traversal`,
`conflicting-facts`) the shipped `query_graph_data` v1 mechanism cannot express at all, by
design.** This is a re-run of the same 39-pair golden-set harness
(`server/tests/eval/run_nlq_golden_set_eval.py`, unit U29c,
`docs/plans/workflow-nl-query-generation-ml.md` §3-5) against the mechanism after the full RCA fix
cycle (`docs/reviews/workflow-nl-query-generation-rca.md`, both DSL fixes and the prompt fix)
landed and passed its implementation-diff review
(`docs/reviews/workflow-nl-query-generation-impl.md`, Pass 2: **approve**). Real FalkorDB
(`reference` + `ws:nlq-eval`), real local LM Studio (`qwen/qwen3-4b-2507`, temperature 0), same
unmodified golden set, same unmodified harness.

- **Overall: 33/39 = 84.6% (Wilson 95% CI [70.3%, 92.8%])** against the ≥85% target — **FAIL**,
  by exactly one pair (34/39 = 87.2% would have cleared it). Up from the pre-fix run's 18/39 =
  46.2%.
- **`knowledge_base` subset (AC-2 gate): 13/19 = 68.4% (CI [46.0%, 84.6%])** against ≥75% —
  **FAIL**. Up from 8/19 = 42.1%.
- **not-found/abstention false-answer rate: 0/6 = 0.0%** against ≤10% — **PASS** (unchanged from
  the pre-fix run — the mechanism never fabricated a specific wrong fact, before or after the
  fix).
- **Excluding `relationship-traversal`/`conflicting-facts` (the two shapes the mechanism is not
  designed to answer at all): 33/33 = 100.0% (CI [89.6%, 100.0%])**, and the `catalog` subset
  alone is 20/20 = 100.0%. Every one of the 12 specific RCA-targeted cases named in this unit's
  brief (`nlq-02, 04, 08, 16, 17, 18, 20, 21, 22, 23, 25, 26`) was individually re-checked against
  this run's actual raw tool output and **now passes** — see "RCA-targeted cases, verified" below.
  The 6 remaining misses (`nlq-34` through `nlq-39`) are exactly the `relationship-traversal`/
  `conflicting-facts` pairs, which is the same structurally-expected-0% outcome the pre-fix report
  already documented for those two shapes — not a new or surprising failure.

**What this means plainly:** the fixes worked completely on every shape the mechanism is designed
to handle. The gate still fails only because the golden set's `knowledge_base` subset includes 6
pairs (of 19) that require graph-relationship traversal or multi-fact conflict resolution — a
capability `query_graph_data` v1 structurally does not have (one single-label `MATCH`, no edge
traversal at all, per `workflow-nl-query-generation.md` §3.6 and the shipped golden-set integrity
test). Closing this gap is an architecture question (extend the DSL, or accept these shapes as
out of the v1 mechanism's scope), not a model-quality or prompt-quality question — see
"Recommendation" below.

**CPG:** not applicable — this unit re-runs an existing, approved evaluation harness and reports
its output; it involves no code-structure analysis task.

## Run context (this re-run)

- **Date:** 2026-08-30 (`server/tests/eval/nlq_eval_results.json`'s `runAt`:
  `2026-08-30T11:57:01Z`, this run).
- **Model:** `lmstudio/qwen/qwen3-4b-2507` (same model as the pre-fix run, resolved via a real
  `ModelGateway.from_env()` over `config/opencode.example.json` + `config/models.json`,
  `temperature: 0`), served by a real local LM Studio at `localhost:1234/v1`.
- **Graphs:** `reference` (catalog dataset, 15 `Product` nodes, confirmed seeded before running)
  and `ws:nlq-eval` (knowledge_base dataset, 62 `Entity` nodes, confirmed seeded before running;
  read-only throughout — every read via `GRAPH.RO_QUERY`).
- **Golden set:** `server/tests/eval/nlq_golden_set.jsonl`, all 39 rows, unmodified.
- **Harness:** `server/tests/eval/run_nlq_golden_set_eval.py`, run once, sequentially (never
  concurrent — the documented local-LM-Studio JIT-loading-thrash hazard), unmodified — read in
  full before running; no bug found in it. Raw per-pair results overwritten to
  `server/tests/eval/nlq_eval_results.json` (the prior run's file is not separately retained in
  this repo; its numbers are preserved below in "Prior run").
- **Mechanism under test:** `server/falkorchat/querygen.py` + `repository.py` + `services.py` +
  `tools.py` at the state gated `approve` by `docs/reviews/workflow-nl-query-generation-impl.md`
  Pass 2 (commit `c033b30`) — no production code was touched by this unit.

## Layer 1 — execution accuracy (the FR-4/AC-4 gate)

| Scope | n | Correct | Accuracy | Wilson 95% CI | Target | Result |
|---|---|---|---|---|---|---|
| **Overall** | 39 | 33 | 84.6% | [70.3%, 92.8%] | ≥ 85% | **FAIL** |
| `knowledge_base` subset (AC-2 gate) | 19 | 13 | 68.4% | [46.0%, 84.6%] | ≥ 75% | **FAIL** |
| `catalog` subset (context, not independently gated) | 20 | 20 | 100.0% | [83.9%, 100.0%] | — | — |
| not-found/abstention false-answer rate | 6 | 0 fabricated | 0.0% | — | ≤ 10% | **PASS** |

For context (not a substitute gate — §5 gates the two subsets above, not this number): restricting
to the five shapes the mechanism is designed to answer (excluding `relationship-traversal` and
`conflicting-facts`) yields **33/33 = 100.0% (CI [89.6%, 100.0%])** — every miss in this run sits
inside the two structurally-out-of-scope shapes; there is no remaining miss on any shape the
mechanism is supposed to handle.

### Per-shape breakdown (reported per §5's convention — not independently gated, but never averaged away)

| Shape | Dataset(s) | n | Correct | Accuracy |
|---|---|---|---|---|
| `single-fact` | catalog + knowledge_base | 9 | 9 | 100.0% |
| `filter-list` | catalog + knowledge_base | 7 | 7 | 100.0% |
| `compound-filter` | catalog only | 3 | 3 | 100.0% |
| `aggregation` | catalog + knowledge_base | 8 | 8 | 100.0% |
| `not-found` | catalog + knowledge_base | 6 | 6 | 100.0% |
| `relationship-traversal` | knowledge_base only | 4 | 0 | **0.0% — expected by design, see below** |
| `conflicting-facts` | knowledge_base only | 2 | 0 | **0.0% — expected by design, see below** |

Split by dataset × shape (the finer grain behind the table above):

| Dataset | Shape | n | Correct |
|---|---|---|---|
| catalog | single-fact | 5 | 5 |
| catalog | filter-list | 4 | 4 |
| catalog | compound-filter | 3 | 3 |
| catalog | not-found | 3 | 3 |
| catalog | aggregation | 5 | 5 |
| knowledge_base | single-fact | 4 | 4 |
| knowledge_base | filter-list | 3 | 3 |
| knowledge_base | not-found | 3 | 3 |
| knowledge_base | aggregation | 3 | 3 |
| knowledge_base | relationship-traversal | 4 | 0 |
| knowledge_base | conflicting-facts | 2 | 0 |

Every catalog and knowledge_base shape the mechanism is designed to answer is now at **100%**
(was: catalog 50.0%, knowledge_base non-relationship/conflicting shapes well below 100% pre-fix —
see "Prior run" below for the exact prior per-shape numbers).

**`relationship-traversal`/`conflicting-facts` scoring 0/4 and 0/2 is the expected, correct
outcome of this run, not a regression or a corpus/harness defect** — unchanged from the pre-fix
report's framing. Per the golden set's own integrity test (`test_nlq_golden_set_integrity.py`)
and the design (`workflow-nl-query-generation.md` §3.6), `query_graph_data` v1 supports exactly
one single-label `MATCH` pattern with no relationship traversal at all — these two shapes require
walking a `RELATES_TO` edge (who did X acquire / found / partner with; the conflicting
employee-count facts are modeled as two separate `RELATES_TO` edges, not a property on the
subject node) and are therefore structurally unreachable by the shipped mechanism, by design,
today, exactly as before the fix cycle. This is now the *only* source of gate failure in this
run — before the fix, it was one of several.

**Observed abstention behavior on these 6 pairs (evidence, not a new finding to root-cause,
included for completeness since the pre-fix report noted this mechanism did not always abstain
cleanly on them):**

| id | question | actual `toolResult` |
|---|---|---|
| `nlq-34` | Who did Marlowe Robotics acquire? | `{"items": [], "finding": "no matching data found"}` — clean abstention |
| `nlq-35` | Who founded Marlowe Robotics? | `{"items": [], "finding": "no matching data found"}` — clean abstention |
| `nlq-36` | Which company did Marlowe Robotics partner with? | `{"items": [], "finding": "no matching data found"}` — clean abstention |
| `nlq-37` | Where is Marlowe Robotics located? | `{"items": [{"e.name": "Marlowe Robotics"}]}` — returns the source node itself, not a location |
| `nlq-38` | How many employees does Marlowe Robotics have? | `{"items": [{"count(e.entityId)": 12}]}` — a compiled, executed count, wrong value |
| `nlq-39` | What is Marlowe Robotics' employee count as of March 2026? | `{"items": [], "finding": "no matching data found"}` — clean abstention |

Four of six now abstain cleanly (`nlq-34/35/36/39`); two (`nlq-37/38`) still compile and execute
to a plausible-looking but wrong single-label result. All six are still correctly scored incorrect
either way — this table is descriptive evidence for whoever looks at relationship-traversal
robustness later, not a new gate finding.

## RCA-targeted cases, verified (per this unit's brief — actual observation, no assumption)

The brief named 12 specific cases the RCA's fix categories (A/B/C/D) were expected to affect:
`nlq-08, 16, 17, 18, 20` (Priority 1, categories A/C — catalog price/aggregation exact-match and
duplicate-row fixes), `nlq-02, 04` (regression-check pairs), `nlq-21, 22, 23` (category C —
duplicate-entity `DISTINCT` fix), and `nlq-25, 26` (category D — the prompt fix, `name` vs.
`entityId` projection). Read directly from this run's actual per-pair `toolResult`, all 12 now
score Layer 1 correct:

| id | question | actual `toolResult` (this run) |
|---|---|---|
| `nlq-02` | (single-fact, catalog) | correct — Layer 1 OK |
| `nlq-04` | (single-fact, catalog) | correct — Layer 1 OK |
| `nlq-08` | Which products cost less than $50? | full correct 6-item set (`Gaming Mouse Pad XL`, `Wireless Charging Pad`, `Wireless Mouse Pro`, `Laptop Stand Aluminum`, `USB-C Hub 7-in-1`, `Bluetooth Speaker Mini`) |
| `nlq-16` | Which product is the cheapest? | `{"items": [{"p.name": "Gaming Mouse Pad XL"}]}` — matches expected |
| `nlq-17` | (aggregation, catalog) | correct — Layer 1 OK |
| `nlq-18` | How many products cost less than $100? | `{"items": [{"count(p)": 10}]}` — matches expected `10` (was `0` pre-fix) |
| `nlq-20` | (aggregation, catalog) | correct — Layer 1 OK |
| `nlq-21` | What type of entity is Marlowe Robotics? | `{"items": [{"e.type": "Organization"}]}` — one row, correct (was 12 rows pre-fix) |
| `nlq-22` | (single-fact, knowledge_base) | correct — Layer 1 OK |
| `nlq-23` | (single-fact, knowledge_base) | correct — Layer 1 OK |
| `nlq-25` | Which entities are of type Location? | full correct 8-item set of `e.name` values (was `e.entityId` pre-fix) |
| `nlq-26` | What entities are classified as Person? | `{"items": [{"e.name": "Elena Ferro"}, {"e.name": "Devon Cole"}, {"e.name": "Priya Nandakumar"}]}` — matches expected 3-item set |

No case in this list still fails — reported as directly observed, not projected from the RCA's
own expectation.

## Prompt fix (RCA Priority 2 / category D) — confirmed shipped

The RCA's §4 Priority 2 gave concrete replacement text for `tools.py`'s
`_QUERY_REQUEST_INSTRUCTIONS`. Read directly against the current file
(`server/falkorchat/tools.py:845-890`): the shipped text is **byte-identical** to the RCA's
recommended replacement, including the bare-number clarification (A), the `*Normalized`-is-
internal rule (B), the name-not-`entityId` rule + example (D), and the no-invented-filter rule
(the `nlq-16` compounding-defect fix). Priority 2 (D) was not held back — it shipped in the same
cycle as the DSL fixes, as the brief anticipated checking.

## Layer 2 — rendered-answer sanity check (non-gating)

Overall Layer 2 (does the final natural-language sentence contain the expected value(s), via
normalized-substring match): **25/39 = 64.1%** (up from the pre-fix run's 16/39 = 41.0%). Per the
ml note §3/§5, this is **not** the FR-4/AC-4 gate — a secondary AC-5 live-demo sanity signal. All
14 Layer 2 misses in this run are Layer 1 *passes* whose rendered sentence didn't literally
contain the expected text in a normalized-substring-matchable form (`nlq-11, 16, 18, 19, 25, 27,
31, 33`) or are the same 6 relationship-traversal/conflicting-facts pairs that also fail Layer 1
(`nlq-34` through `nlq-39`) — there is no case in this run where Layer 2 passes while Layer 1
fails. This is a real, reportable rendering-quality gap (worth a look if Layer 2 becomes
load-bearing for the AC-5 demo) but is out of this unit's scope to root-cause, unchanged from the
pre-fix report's framing of Layer 2 as a non-gating secondary signal.

## Scorer module and live harness

Unchanged from the prior run — `server/tests/eval/nlq_scoring.py` (32 unit tests, mutation-tested,
network/DB-free, part of the default offline suite) and
`server/tests/eval/run_nlq_golden_set_eval.py` (the bare-script harness, read in full for this
unit; no bug found, no modification made). See "Prior run" below for the original description;
not repeated here since neither changed.

## Artifacts

- Scorer module + tests: `server/tests/eval/nlq_scoring.py`, `server/tests/eval/test_nlq_scoring.py`.
- Live harness: `server/tests/eval/run_nlq_golden_set_eval.py` (read, unmodified).
- Raw per-pair results for **this** run (input to the numbers above — every question, expected
  value, raw tool JSON, Layer 1 reason string, rendered Layer 2 answer):
  `server/tests/eval/nlq_eval_results.json` (overwritten by this unit; the pre-fix run's numbers
  are preserved in "Prior run" below rather than as a separate JSON file).
- This report: `docs/test-reports/workflow-nl-query-generation-report.md`.
- Side effects, disclosed: none beyond the harness's own reads. `reference`/`ws:nlq-eval` were
  confirmed already correctly seeded before this run (15 `Product` nodes, 62 `Entity` nodes) and
  were not mutated — every read went through `GRAPH.RO_QUERY`. No offline `pytest` run was
  executed as part of this unit, so no re-seed was needed afterward.

## Recommendation

**Do not mark K-055's FR-4/AC-4/AC-2 done on this evidence — both gates still literally fail
(84.6% < 85% overall, 68.4% < 75% on the `knowledge_base` subset).** This is a materially
different situation from the pre-fix run, though: every point of failure is now confined to the 6
of 39 pairs (`relationship-traversal` ×4, `conflicting-facts` ×2) that `query_graph_data` v1
cannot express by design — one single-label `MATCH`, no edge traversal, no multi-fact conflict
resolution. On every shape the mechanism is designed to handle, this run measures **100%** (33/33,
CI [89.6%, 100.0%]). This is exactly the situation the RCA's own §5 item 2 anticipated as the
trigger to escalate past its own recommended fixes ("if `knowledge_base` still doesn't clear 75%
and overall still doesn't clear 85% [after Priority 2], that is the point to escalate to Priority
3 with real, current-run numbers instead of this RCA's projections").

That said, the RCA's Priority 3 (model routing — trying a different/larger model) is unlikely to
be the right next step on this evidence: the remaining misses are not model-comprehension
failures the way the pre-fix ones were (wrong field bound, exact-match value mismatch) — they are
requests for a capability the DSL has no way to express at all, regardless of which model fills in
the structured-completion request. A different model cannot invent a `RELATES_TO` traversal the
compiler doesn't support. Two real options, neither of which is this unit's to decide:

1. **Extend `query_graph_data`'s DSL to support relationship traversal and/or multi-fact
   resolution** — an architecture change, `architect`'s call, scoped larger than the DSL fixes
   already shipped.
2. **Revisit whether these two shapes belong in the FR-4/AC-4/AC-2 gate's scope at all** — a
   stakeholder/`data-scientist` decision, not something to resolve unilaterally by re-scoring; the
   ml note §6 already flags this exact possibility ("Aggregation/compound-filter question shapes
   may not be expressible at all under a narrow DSL... report it as a named gap... not a reason to
   drop those golden pairs from the set") for a structurally-inexpressible shape, and the same
   logic applies here.

Routed onward to `data-scientist`/`architect` per the brief — no production-code change was made
or should be inferred from this report.

---

## Prior run (pre-fix, 2026-08-29) — preserved for comparison

The section below is the original report's content, kept verbatim for historical comparison
against the numbers above. It reflects the state of the mechanism **before** the RCA fix cycle
landed and is superseded by the run reported at the top of this document.

**Verdict: Layer 1 (the FR-4/AC-4 gate) FAILS, decisively — not a borderline call.** A live,
39-pair run of the golden-set scoring harness against the pre-fix `query_graph_data` mechanism, a
real FalkorDB `reference`/`ws:nlq-eval`, and a real local LM Studio (`qwen/qwen3-4b-2507`,
temperature 0) scored **18/39 = 46.2% overall execution accuracy (Wilson 95% CI [31.6%, 61.4%])**
— the entire confidence interval sits well below the 85% target. The `knowledge_base` subset,
gated separately per AC-2, scored **8/19 = 42.1% (CI [23.1%, 63.7%])** against a 75% target — also
a clear, decisive miss. The one gate that run **did** pass: the not-found/abstention false-answer
rate was **0/6 = 0.0%** against a ≤10% ceiling.

| Scope | n | Correct | Accuracy | Wilson 95% CI | Target | Result |
|---|---|---|---|---|---|---|
| Overall | 39 | 18 | 46.2% | [31.6%, 61.4%] | ≥ 85% | FAIL |
| `knowledge_base` subset (AC-2 gate) | 19 | 8 | 42.1% | [23.1%, 63.7%] | ≥ 75% | FAIL |
| `catalog` subset (context, not independently gated) | 20 | 10 | 50.0% | [29.9%, 70.1%] | — | — |
| not-found/abstention false-answer rate | 6 | 0 fabricated | 0.0% | — | ≤ 10% | PASS |

Per-shape breakdown (pre-fix):

| Shape | Dataset(s) | n | Correct | Accuracy |
|---|---|---|---|---|
| `single-fact` | catalog + knowledge_base | 9 | 4 | 44.4% |
| `filter-list` | catalog + knowledge_base | 7 | 4 | 57.1% |
| `compound-filter` | catalog only | 3 | 0 | 0.0% |
| `aggregation` | catalog + knowledge_base | 8 | 4 | 50.0% |
| `not-found` | catalog + knowledge_base | 6 | 6 | 100.0% |
| `relationship-traversal` | knowledge_base only | 4 | 0 | 0.0% — expected by design |
| `conflicting-facts` | knowledge_base only | 2 | 0 | 0.0% — expected by design |

Layer 2 (pre-fix): **16/39 = 41.0%**.

Illustrative pre-fix failures (root-caused in full by
`docs/reviews/workflow-nl-query-generation-rca.md`):

- **Exact-match filters missing real rows.** `nlq-08`, `nlq-10`/`nlq-11`/`nlq-12`, and
  `nlq-16`/`nlq-17`/`nlq-20` all came back as `{"items": [], "finding": "no matching data
  found"}`. `nlq-18` (true answer 10) came back as `{"items": [{"count(p)": 0}]}`.
- **Wrong variable/property bound in a filter.** `nlq-21`/`nlq-22`/`nlq-23` came back with every
  entity of the expected *type* rather than the one named entity (e.g. `nlq-21` returned 12 rows
  instead of 1).
- **Extra, unrequested columns widening the result past an exact-set match.** `nlq-25`/`nlq-26`
  returned `e.entityId` instead of `e.name`.
- **`relationship-traversal`/`conflicting-facts` pairs did not cleanly abstain** — they fell back
  to a bulk, unfiltered-looking result (e.g. `nlq-34` returned 12 rows of bare `e.entityId`
  values) rather than the tool's own abstention shape.

**Report-location decision (original, unchanged):** filed as
`docs/test-reports/workflow-nl-query-generation-report.md` per the current `AGENTS.md` naming
convention (`<topic-slug>-report.md`), matching `document-ingestion-report.md` /
`workflow-catalog-lookup-report.md` precedent; not mirrored on `graphrag-eval`'s older
`<slug>-<date>.md` precedent, which predates the current strict naming convention.
