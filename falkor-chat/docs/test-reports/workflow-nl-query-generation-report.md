# `workflow-nl-query-generation` — Golden-Set Evaluation Report

> **Status:** active · **Owner:** `tdd-engineer` · **Tracks:** K-055 (M6)

## Summary

**Verdict: Layer 1 (the FR-4/AC-4 gate) FAILS, decisively — not a borderline call.** A live,
39-pair run of the golden-set scoring harness (unit U29c,
`docs/plans/workflow-nl-query-generation-ml.md` §3-5) against the real, already-`analyst`-approved
`query_graph_data` mechanism (`server/falkorchat/querygen.py` + `tools.QueryGraphDataTool`), a real
FalkorDB `reference`/`ws:nlq-eval`, and a real local LM Studio (`qwen/qwen3-4b-2507`, temperature 0)
scored **18/39 = 46.2% overall execution accuracy (Wilson 95% CI [31.6%, 61.4%])** — the entire
confidence interval sits well below the 85% target, so this is not a "85% vs 80%, can't tell at
this n" situation (§5's own caution); it is a clear miss. The `knowledge_base` subset, gated
separately per AC-2, scores **8/19 = 42.1% (CI [23.1%, 63.7%])** against a 75% target — also a
clear, decisive miss. The one gate this run **does** pass: the not-found/abstention false-answer
rate is **0/6 = 0.0%** against a ≤10% ceiling — when the mechanism doesn't know an answer, it
correctly said so every time; it never fabricated a specific-sounding wrong fact.

This report computes and states the numbers honestly, including the shapes that came out worse
than expected on shapes the mechanism is *supposed* to handle (single-fact, filter-list,
aggregation, compound-filter) — this is a real finding about the shipped mechanism + this specific
small local model combination, not a harness defect (the harness itself, `nlq_scoring.py`, is
unit-tested and mutation-tested to spec, §"Scorer module" below). Root-causing *why* the model
underperforms and deciding what to do about it is `data-scientist`/`architect` territory, out of
this unit's scope (per the dispatch brief: "No changes to `querygen.py`/`repository.py`/
`services.py`/`tools.py`") — this report's job is to compute and report correctly, which it does,
including concrete illustrative examples of what actually went wrong (§"Illustrative failures"). No
production code was touched.

**CPG:** considered, not relevant — `cpg_falkorchat` is stale relative to the K-055
querygen/repository/services/tools work (built 2026-08-26T22:27Z, predating that landing) and this
unit builds/runs a scoring harness against an already-built, already-reviewed mechanism rather than
performing structural impact analysis on it — the same assessment the dispatch brief itself made.

## Run context

- **Date:** 2026-08-30 (see `server/tests/eval/nlq_eval_results.json`'s `runAt`).
- **Model:** `lmstudio/qwen/qwen3-4b-2507` (resolved via a real `ModelGateway.from_env()` over
  `config/opencode.example.json` + the repo's real `config/models.json`, which pins
  `temperature: 0` for this model), served by a real local LM Studio at `localhost:1234/v1`.
- **Graphs:** `reference` (catalog dataset — freshly re-seeded via `bootstrap_schema.sh acme` +
  `seed_catalog.sh acme` immediately before this run, since an offline `pytest -q` run earlier in
  this session had wiped it per the documented teardown hazard) and `ws:nlq-eval` (knowledge_base
  dataset — unit U29's already-seeded corpus, 62 `Entity` nodes, untouched by this run: every read
  went through `GRAPH.RO_QUERY`).
- **Golden set:** `server/tests/eval/nlq_golden_set.jsonl`, all 39 rows, unmodified.
- **Harness:** `server/tests/eval/run_nlq_golden_set_eval.py`, run once, sequentially (39 pairs,
  never concurrent — the documented local-LM-Studio JIT-loading-thrash hazard), taking under a
  minute of live-call wall-clock time in this run. Raw per-pair results (question, expected value,
  raw tool JSON, Layer 1 verdict + reason, rendered Layer 2 answer + verdict) persisted verbatim to
  `server/tests/eval/nlq_eval_results.json`.

## Layer 1 — execution accuracy (the FR-4/AC-4 gate)

| Scope | n | Correct | Accuracy | Wilson 95% CI | Target | Result |
|---|---|---|---|---|---|---|
| **Overall** | 39 | 18 | 46.2% | [31.6%, 61.4%] | ≥ 85% | **FAIL** |
| `knowledge_base` subset (AC-2 gate) | 19 | 8 | 42.1% | [23.1%, 63.7%] | ≥ 75% | **FAIL** |
| `catalog` subset (context, not independently gated) | 20 | 10 | 50.0% | [29.9%, 70.1%] | — | — |
| not-found/abstention false-answer rate | 6 | 0 fabricated | 0.0% | — | ≤ 10% | **PASS** |

For context (not a substitute gate — §5 gates the two subsets above, not this number): restricting
to the five shapes the mechanism is actually designed to answer (excluding `relationship-traversal`
and `conflicting-facts`, both structurally unanswerable by design — see below) still yields only
**18/33 = 54.5% (CI [38.0%, 70.2%])** — the shortfall is not an artifact of the two known-hard
shapes dragging the average down; the mechanism is missing the clear majority of questions on
shapes it is supposed to handle too.

### Per-shape breakdown (reported per §5's convention — not independently gated, but never averaged away)

| Shape | Dataset(s) | n | Correct | Accuracy |
|---|---|---|---|---|
| `single-fact` | catalog + knowledge_base | 9 | 4 | 44.4% |
| `filter-list` | catalog + knowledge_base | 7 | 4 | 57.1% |
| `compound-filter` | catalog only | 3 | 0 | 0.0% |
| `aggregation` | catalog + knowledge_base | 8 | 4 | 50.0% |
| `not-found` | catalog + knowledge_base | 6 | 6 | 100.0% |
| `relationship-traversal` | knowledge_base only | 4 | 0 | **0.0% — expected by design, see below** |
| `conflicting-facts` | knowledge_base only | 2 | 0 | **0.0% — expected by design, see below** |

Split by dataset × shape (the finer grain behind the table above):

| Dataset | Shape | n | Correct |
|---|---|---|---|
| catalog | single-fact | 5 | 3 |
| catalog | filter-list | 4 | 3 |
| catalog | compound-filter | 3 | 0 |
| catalog | not-found | 3 | 3 |
| catalog | aggregation | 5 | 1 |
| knowledge_base | single-fact | 4 | 1 |
| knowledge_base | filter-list | 3 | 1 |
| knowledge_base | not-found | 3 | 3 |
| knowledge_base | aggregation | 3 | 3 |
| knowledge_base | relationship-traversal | 4 | 0 |
| knowledge_base | conflicting-facts | 2 | 0 |

**`relationship-traversal`/`conflicting-facts` scoring 0/4 and 0/2 is the expected, correct outcome
of a passing run, not a regression or a corpus/harness defect.** Per the golden set's own integrity
test (`test_nlq_golden_set_integrity.py`) and the design (`workflow-nl-query-generation.md` §3.6),
`query_graph_data` v1 supports exactly one single-label `MATCH` pattern with no relationship
traversal at all — these two shapes require walking a `RELATES_TO` edge (who did X acquire / found
/ partner with; the conflicting employee-count facts are modeled as two separate `RELATES_TO`
edges, not a property on the subject node) and are therefore structurally unreachable by the
shipped mechanism, by design, today. This run's 0% on both is exactly what §5/the integrity test's
own docstring predicts — named here explicitly so it is never mistaken for an unexplained failure
alongside the shapes that genuinely underperformed.

## Layer 2 — rendered-answer sanity check (non-gating)

Overall Layer 2 (does the final natural-language sentence contain the expected value(s), via
normalized-substring match): **16/39 = 41.0%**. Per the ml note §3/§5, this is **not** the FR-4/AC-4
gate — it is a secondary sanity signal for the AC-5 live-demo bar. It tracks Layer 1 closely in this
run (both numbers land in the same range), which is the expected relationship when the underlying
structured result is what actually determines whether a correct answer *can* be rendered — a
mechanism that never retrieved the right fact cannot render it correctly regardless of phrasing.
The two layers disagree in both directions on a handful of pairs (e.g. `nlq-22`/`nlq-26` render an
answer containing the expected value even though the raw structured result also carried extra,
unrequested rows/columns that fail Layer 1's exact-match rule; `nlq-19`/`nlq-27`/`nlq-31`/`nlq-33`
pass Layer 1 but the render step phrased the number/list in a way the containment check didn't
match) — expected given Layer 2 is a strictly different, weaker instrument, never treated here as
correcting or overriding Layer 1.

**Layer 2 method (this unit's own implementation call, per the brief).** Rather than driving a full
multi-turn `salesperson@v4` conversation through the executor (that heavier live e2e proof is a
separate, later `qa-engineer` unit's job), this harness makes one additional lightweight internal
LLM call per pair — the same `models.llm("step", ws=ctx.ws)` seam `QueryGraphDataTool` itself
already uses — asking the model to render the tool's own raw JSON result into one short sentence,
instructed to use only the JSON's data and to always use numeric digits verbatim rather than
spelling numbers out as words (to keep the containment check regex-friendly). Not an LLM-as-judge —
a deterministic normalized-substring check runs against the rendered text, exactly the ml note §3's
stated default instrument; no LLM judge was needed for any pair in this golden set.

## Illustrative failures (a sample, not exhaustive — full detail in `nlq_eval_results.json`)

These are real observations from this run's actual raw tool output, offered so a low number does
not read as an unexplained black box. Not a root-cause diagnosis (out of this unit's scope) — just
what the persisted evidence shows:

- **Exact-match filters missing real rows.** `nlq-08` ("Which products cost less than $50?"),
  `nlq-10`/`nlq-11`/`nlq-12` (compound category+price filters), and `nlq-16`/`nlq-17`/`nlq-20`
  (cheapest/most-expensive aggregation) all came back as `{"items": [], "finding": "no matching
  data found"}` — the mechanism's own abstention shape, meaning either the model's structured
  completion failed to parse/validate, or `querygen.compile` rejected an unregistered label/property,
  or the model's own filter value didn't exact-match the WHERE clause it authored. `nlq-18`
  ("How many products cost less than $100?", true answer 10) came back as `{"items":
  [{"count(p)": 0}]}` — a *compiled and executed* query that ran to a wrong count, not an
  abstention.
- **Wrong variable/property bound in a filter.** Three of four `knowledge_base` `single-fact`
  pairs (`nlq-21`/`nlq-22`/`nlq-23`, "what type of entity is X") came back with **every** entity of
  the *expected type* rather than the one named entity — e.g. `nlq-21` ("What type of entity is
  Marlowe Robotics?", expects `Organization`) returned 12 rows of `{"e.type": "Organization"}`, one
  per `Organization`-type entity in the corpus. The model's structured completion appears to have
  filtered on `type` instead of `nameNormalized`, inverting which field is the filter and which is
  the projection.
- **Extra, unrequested columns widening the result past an exact-set match.** `nlq-25`/`nlq-26`
  (list all Location-/Person-type entities) returned `e.entityId` instead of, or alongside, `e.name`
  — a technically-valid, compiled query (both properties are allowlisted for `Entity`) that simply
  didn't project the column the question asked for. Per `nlq_scoring.py`'s own documented
  set-extraction rule, this is correctly scored incorrect (an over-broad/wrong-column result is not
  the same as the requested named list), not a scoring-harness bug.
- **`relationship-traversal`/`conflicting-facts` pairs did not cleanly abstain — they fell back to
  a bulk, unfiltered-looking result** (e.g. `nlq-34` "who did Marlowe Robotics acquire" returned 12
  rows of bare `e.entityId` values; `nlq-38`'s employee-count question came back as a `count(...)`
  of 12) rather than the tool's own `{"items": [], "finding": "..."}` abstention shape. Both are
  still correctly scored 0% either way (per the expected-by-design framing above) — noted here only
  because "cleanly abstains" and "structurally can't reach the right answer but still returns
  *something*" are different behaviors worth distinguishing for whoever looks at this mechanism's
  robustness later.

## Scorer module (`server/tests/eval/nlq_scoring.py`)

Implements §3's comparison rules as pure, unit-testable functions (`score_pair`, `layer2_contains`,
`wilson_interval`, `load_golden_set`) — no FalkorDB, no LLM dependency. Test-first per this agent's
normal practice:

- **32 unit tests**, `server/tests/eval/test_nlq_scoring.py` — scalar case/whitespace-folding and
  numeric epsilon (including the exact 0.01 boundary and a never-string-equal-a-formatted-price
  case), unordered set comparison (order-independent, case-folded, rejects both a missing member
  and an extra member), `not_found` correctness, `conflicting-facts`' containment (not exact-match)
  rule with its own missing/extra/empty cases, an unknown-`expected.type` `ValueError`, and
  `wilson_interval`'s boundary/symmetry/width behavior. Runs in the default offline suite (genuinely
  network/DB-free — every test feeds hand-built rows and hand-built tool results).
- **Mutation-tested**: five deliberate single-line mutations (epsilon tolerance removed, case-fold
  removed, the `not_found` predicate inverted, `filter-list`'s exact-set-match relaxed to
  containment, `conflicting-facts`' containment tightened to exact-match) were each introduced,
  confirmed to fail the suite, then reverted — all five caught.
- Offline suite after adding this module: **2259 passed, 4 deselected** (was 2227 passed, 4
  deselected before this unit; the delta is exactly the 32 new tests).

## Live harness (`server/tests/eval/run_nlq_golden_set_eval.py`)

A bare script, **not** a `pytest.mark.live` test module — `server/tests/conftest.py`'s autouse
`_model_config_env` fixture redirects `FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_MODEL_CONFIG` to
offline dim-4 test fixtures for *every* test under `server/tests/`, the same redirection
`test_golden_set_integrity.py`'s own docstring documents and `test_guard_calibration_live.py`
deliberately works around. Rather than working around it inside a pytest module, this harness runs
standalone (the same convention `scripts/seed_eval_corpus.py` already uses for its own real
embedding-model resolution), sidestepping the question entirely. Constructs a real
`Services(Repository(db.connect()))` and `QueryGraphDataTool(services, models=gateway)` exactly as
`server/tests/test_tools.py` documents, and calls `.run(...)` once per golden pair with
`ctx.ws = "nlq-eval"` for `knowledge_base` pairs and `ctx.ws = "acme"` for `catalog` pairs (unused
by `CATALOG_SCHEMA`'s fixed `graph_key="reference"`, but still a valid, already-bootstrapped
workspace id). Fails loudly with a clear fix command if `reference`/`ws:nlq-eval` aren't seeded, or
if FalkorDB isn't reachable, rather than silently scoring against empty data.

## Report-location decision (this unit's own call, per the brief)

Filed as `docs/test-reports/workflow-nl-query-generation-report.md` — the current, dominant
`AGENTS.md` naming convention (`<topic-slug>-report.md`, same family as
`docs/requirements/workflow-nl-query-generation.md`/`docs/plans/workflow-nl-query-generation-ml.md`/
`docs/reviews/workflow-nl-query-generation*.md`), matching recent precedent
(`document-ingestion-report.md`, `workflow-catalog-lookup-report.md`). **Not** mirrored on
`graphrag-eval`'s own `<slug>-<date>.md` precedent — those two files predate the current strict
naming convention (closed role set, no date-suffix role) documented in the current `AGENTS.md`; the
`-report` role already conveys "the golden-set gate result," and a single run of this specific gate
does not need a per-date family the way `graphrag-eval`'s repeatable regression-baseline reports do.

## Artifacts

- Scorer module + tests: `server/tests/eval/nlq_scoring.py`, `server/tests/eval/test_nlq_scoring.py`.
- Live harness: `server/tests/eval/run_nlq_golden_set_eval.py`.
- Raw per-pair results (input to this report, includes every question, expected value, raw tool
  JSON, Layer 1 reason string, and rendered Layer 2 answer): `server/tests/eval/nlq_eval_results.json`.
- This report: `docs/test-reports/workflow-nl-query-generation-report.md`.
- Side effects, disclosed: `reference` was re-seeded (`bootstrap_schema.sh acme` + `seed_catalog.sh
  acme`) before this run, since an earlier offline `pytest -q` run in this same session had wiped
  it (documented hazard) — the catalog is back to its standard 15-product state, confirmed via
  `verify_catalog.sh`-equivalent direct queries during the run. `ws:nlq-eval` was read-only
  throughout (every read via `GRAPH.RO_QUERY`) and is unchanged.

## Recommendation

**Do not mark K-055's FR-4/AC-4/AC-2 done on this evidence.** The harness and its numbers are
trustworthy (unit-tested, mutation-tested, run against the real system with a real model, real
graphs); what they say is that this specific mechanism + this specific local 4B model combination
does not clear the golden set's bar today, on shapes it is designed to handle, not only on the two
shapes it is designed to fail. Next steps are outside this unit's remit (no production-code changes
were made or should be inferred from this report) — flagged for `data-scientist`/`architect`
follow-up: whether the structured-completion system prompt needs strengthening, whether this model
is simply undersized for this task shape (ml note §5's own "revisit if the chosen mechanism is
materially weaker than a 4B-class local model" clause), or whether the DSL's exact-match-only filter
semantics (no case-insensitive/fuzzy comparison) need reconsidering given how many misses in this
run look like a filter value that didn't literally match stored data.
