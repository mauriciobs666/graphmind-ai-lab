# `workflow-nl-query-generation` — root-cause analysis of the Layer 1 golden-set failure

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-055 (M6)

## 1. The question and the decision it serves

**Question:** the live golden-set run (`docs/test-reports/workflow-nl-query-generation-report.md`)
scored 18/39 = 46.2% overall / 8/19 = 42.1% on `knowledge_base` against an 85%/75% gate — a
decisive miss. *Why*, specifically, does the shipped mechanism (`querygen.py` +
`tools.QueryGraphDataTool`) + `qwen/qwen3-4b-2507` underperform this badly, and what should the
next unit actually change?

**Decision this serves:** whether the next unit is a **DSL fix** (`querygen.py`, no model
involvement), a **prompt fix** (`tools.py`'s `_QUERY_REQUEST_INSTRUCTIONS`), or a **model-routing
change** (K-056's `config.model` re-point precedent) — three different implementers, three
different risk profiles, and (per the brief) a possible genuine stakeholder fork on the third.

**Method.** Read all 39 raw records in `nlq_eval_results.json` and categorized every failure by
its actual `toolResult` shape. Where the raw evidence didn't disambiguate *why* a query returned
what it did (the tool's own docstring deliberately collapses "unparseable completion," "compile
rejection," and "compiled-but-zero-rows" into one identical `{"items": [], "finding": "no matching
data found"}` shape — see `tools.py:885-895` — so the persisted evidence alone cannot tell these
apart), I ran **7 live, sequential, single-call confirmation probes** against the real local LM
Studio (`qwen/qwen3-4b-2507`, temperature 0, exactly `_build_query_request_system_prompt`'s
system-prompt text reconstructed verbatim from `tools.py` + `querygen.DATASET_REGISTRY`, one
system+user message pair per probe, no concurrency) to recover the actual structured completion
the model produced for 7 of the ambiguous misses: `nlq-02`, `nlq-04`, `nlq-08`, `nlq-11`, `nlq-16`,
`nlq-18`, `nlq-21`, `nlq-25` (8 questions, 7 distinct pairs — `nlq-21`/`nlq-25` from the
`knowledge_base` set, the rest `catalog`). No production code, `reference`, `ws:nlq-eval`, or
`ws:acme` was touched — these calls hit LM Studio directly over HTTP, not the running
`falkor-chat` server. Scripts used: `/tmp/claude-1000/.../scratchpad/probe.py`,
`probe2.py`, `probe3.py` (scratchpad, not part of the deliverable).

This is enough to explain **21/21 failures** (100% of the misses) with named, evidenced causes —
not a sample.

## 2. Root-cause categorization (all 21 failures)

| # | Root cause | n | Pairs | Fixable at |
|---|---|---|---|---|
| A | **Numeric filter value serialized as a JSON *string*, not a number** (`"value": "50"` instead of `50`) — FalkorDB's `<`/`>` comparison between a string and the numeric `price` property matches no rows silently (no error, just an empty/zero result), never surfacing as anything but "no matching data found" | 8 | `nlq-08,10,11,12,16,17,18,20` | DSL (`querygen.py`) |
| B | **`nameNormalized` chosen as the filter property with an un-normalized (mixed-case, verbatim) value** — the stored property is lower-cased/whitespace-collapsed per `DESIGN.md` §5.1's convention; an exact `=` against `"Portable SSD 1TB"` never matches the stored `"portable ssd 1tb"` | 2 | `nlq-02, nlq-04` | DSL (`querygen.py`) |
| C | **Un-fused duplicate `Entity` nodes + no `DISTINCT` in the compiled projection** — the *filter and projection are both correct*, but the corpus (by design, per `document-ingestion-ml`'s un-fused-until-explicit-fusion model) carries multiple raw nodes per named entity, so a correct query legitimately returns N≥1 identical rows; Layer 1's scalar rule (ml note §3) requires exactly one row and rejects N>1 even when every row is the right value | 3 | `nlq-21, nlq-22, nlq-23` | DSL (`querygen.py`) |
| D | **Wrong/extra projected property for "list/classify entities of type X" questions** — the model projects `entityId` instead of, or alongside, `name`, even though the question asks to *name* the entities | 2 | `nlq-25, nlq-26` | Prompt (`tools.py`) |
| — | **Structurally excluded by design**, not a defect (v1 has no relationship traversal, per plan §3.6/FR-6) | 6 | `nlq-34..39` | N/A — expected |

8 + 2 + 3 + 2 + 6 = 21 = every miss. **None of the three candidate hypotheses in the report's
"Recommendation" section is wrong, but they are not equally weighted**: A+B+C (13/21 = 62% of all
misses, 13/33 ≈ 39% of the *entire* golden set) are DSL-level and fixable with **zero** change to
the model or its prompt; D (2/21) is a genuine prompt gap; and the evidence gives **no support**
for "model undersized for this task shape" as the dominant explanation — see §5.

## 3. Evidence — the live probes

Reconstructed system prompt = `_QUERY_REQUEST_INSTRUCTIONS.format(dataset_schema=...)` exactly as
`_build_query_request_system_prompt` builds it; user message = the golden question verbatim,
exactly as `QueryGraphDataTool.run` sends it (`tools.py:948-951`).

| Probe | Question | Raw completion (temperature 0) | Diagnosis |
|---|---|---|---|
| `nlq-02` | "What category is the Portable SSD 1TB in?" | `filters:[{"property":"nameNormalized","op":"=","value":"Portable SSD 1TB"}]` | **B** — wrong-case value against a normalized property |
| `nlq-04` | "Which category does the Action Camera 4K belong to?" | `filters:[{"property":"nameNormalized","op":"=","value":"Action Camera 4K"}]` | **B** — same pattern, confirms it generalizes |
| `nlq-08` | "Which products cost less than $50?" | `filters:[{"property":"price","op":"<","value":"50"}]` | **A** — quoted numeric string |
| `nlq-11` | "Are there any Accessories products priced above $30?" | `filters:[{"property":"category","op":"=","value":"Accessories"},{"property":"price","op":">","value":"30"}]` | **A** — quoted numeric string (compound filter; category half is correct) |
| `nlq-16` | "Which product is the cheapest?" | `filters:[{"property":"price","op":"=","value":"0"}]`, `order_by:"p.price"`, `limit:1` | **A**, plus a *second*, compounding defect: the model correctly set `order_by`/`limit` for a superlative question but also **invented an unrequested `price = 0` filter**, which alone guarantees zero rows regardless of the type bug |
| `nlq-18` | "How many products cost less than $100?" | `filters:[{"property":"price","op":"<","value":"100"}]`, `returns:["count(p)"]` | **A** — same quoted-numeric-string pattern on an aggregate query |
| `nlq-21` | "What type of entity is Marlowe Robotics?" | `filters:[{"property":"name","op":"=","value":"Marlowe Robotics"}]`, `returns:["e.type"]` | **Correct filter and projection.** This is the important negative result: it refutes the report's own leading hypothesis ("model inverting filter/projection roles") for this pair — the model got it right; the multi-row result in the actual eval is the *un-fused-duplicate-entity* effect (C), not a model mistake |
| `nlq-25` | "Which entities are of type Location?" | `filters:[{"property":"type","op":"=","value":"Location"}]`, `returns:["e.entityId"]` | **D** — correct filter, wrong projected property |

**Reading `nlq-21`'s probe result is the single most consequential finding of this RCA.** The
report's own "Recommendation" section named "the model inverting filter/projection roles" as the
most-suspected hypothesis, anchored on exactly this pair. The live probe shows the opposite: the
model's structured completion for this exact question, replayed at temperature 0, is *exactly*
the intended shape (filter on `name`, project `type`). The multi-row result the actual eval
observed is not a model error at all — it is what a **correct** query returns against a corpus
where the same named entity exists as multiple un-fused raw nodes (documented in the golden set's
own `rationale` field: "9 duplicate, un-fused nodes across the corpus" for Marlowe Robotics, "5
duplicate nodes" for Atlas-7, "3 duplicate nodes" for Devon Cole — exactly the 3 failing pairs).
`nlq-24` (NovaGrid, no documented duplicates) is the only knowledge_base single-fact pair that
passed, which is the control case confirming the mechanism: no duplicates → no problem.

This also explains why `aggregation` on the same corpus scored 3/3 (`nlq-31/32/33`, all `count(...)`
queries): a `count()` aggregate collapses duplicates into the intended raw-node count (the golden
set's own ground truth is *deliberately* the un-fused count, e.g. "17 raw entity nodes" for
`Organization`), so duplication is invisible to an aggregate projection and only breaks a
bare-property scalar projection.

## 4. Recommended fix, prioritized

### Priority 1 — DSL-level fixes in `querygen.py` (categories A, B, C; 13/21 misses; no model/prompt dependency)

These are recommended **first and independent of any prompt change** because the probes show the
model's field/operator *semantic* choices are already frequently correct (`nlq-21`'s filter+
projection, `nlq-11`'s category half, `nlq-16`'s order_by/limit) — the defects are in *value
serialization* and *result-shape handling*, both squarely the compiler's job to make robust
regardless of what a model (this one or a future one) hands it. A prompt-only fix bets on 100%
compliance from a 4B local model on a formatting convention it doesn't reliably follow today
(the wrong-case-vs-right-case split between `nlq-02`/`nlq-04` (wrong) and `nlq-21` (right, but a
different dataset/property) is itself evidence the model's field choice is not fully
prompt-steerable) — a compiler-level fix is deterministic and doesn't re-bet on that compliance.

1. **A — coerce/validate numeric filter values by declared property type.** `DatasetSchema.labels`
   today is `dict[str, frozenset[str]]` (property names only, no type). Recommend extending it to
   `dict[str, frozenset[str]] ` → per-property type info, e.g. `labels: dict[str, dict[str, type]]`
   (`Product: {"price": float, "name": str, "category": str, "nameNormalized": str}`), and in
   `compile()`, when a `QueryFilter.value` is a `str` but the registered type for that property is
   numeric, attempt `float()`/`int()` coercion before binding the parameter; if the string doesn't
   parse as a number against a numeric property, that is itself a compile-time `ValueError`
   (better than a silent zero-row false abstention). This is a genuine schema-completeness gap the
   original design didn't need until aggregation/compound-filter shapes started exercising numeric
   properties — worth doing properly rather than a same-shaped-but-untyped heuristic
   (`try: float(value)` on every string filter value with no type check) that would risk silently
   coercing a legitimately string-shaped value that happens to look numeric. No property in either
   registered schema today is like that, so the heuristic would also work — flagging the typed
   version as the more durable fix, the heuristic as the cheaper one if schema typing is judged
   out of scope for this unit.
2. **B — normalize the filter value server-side for any `*Normalized`-suffixed property**, in
   `compile()`, before binding it as a parameter: apply the exact same normalization function that
   produced the stored `nameNormalized`/`categoryNormalized` values (lowercase + whitespace
   collapse — `DESIGN.md` §5.1's own convention) to `QueryFilter.value` whenever `filt.property`
   ends in `Normalized` and the op is `=`/`<>`. This is **not** the fuzzy/case-insensitive matching
   the plan explicitly ruled out (§3 of the ml note, §rejected-alternatives) — it is enforcing the
   *existing, already-adopted* contract of a field whose entire purpose is to hold a normalized
   value; today nothing enforces that contract on the filter side, only on the write side. This
   fix is agnostic to which property the model picks (`name` vs `nameNormalized`) — it only
   activates for the latter, closing the gap regardless of which one a given phrasing steers the
   model toward.
3. **C — apply `DISTINCT` to the compiled `RETURN` clause when (and only when) none of
   `request.returns` is an aggregate expression** (i.e., every entry matches `_PROJECTION_RE`,
   none matches `_AGGREGATE_RE`). Scoping it this way is load-bearing: applying `DISTINCT`
   unconditionally would silently change `count(e.entityId)`-style aggregation results from "raw
   un-fused node count" (today's correct, golden-set-verified semantics for `nlq-31/32/33`) to
   "distinct node count" — a real regression, not a fix. Restricted to pure-projection returns,
   this is a no-op for every query that already returns one row (no behavior change for the 18
   pairs that already pass) and directly fixes the 3 duplicate-entity misses — and, as a side
   benefit, fixes the same class of degraded Layer 2 rendering seen on this run (`nlq-21`'s
   rendered answer was "The information was not found" specifically *because* the render step was
   handed 12 duplicate rows to summarize, not because the underlying fact was wrong).
   **Flag for `graph-dba` before this ships, not resolved here:** whether this pinned FalkorDB
   build enforces the openCypher/Neo4j-style constraint that `ORDER BY` after `RETURN DISTINCT`
   may only reference expressions already present in the `RETURN` list — the actual `nlq-25` probe
   produced `returns:["e.entityId"]` with `order_by:"e.name"` (a property **not** in `returns`),
   and if `DISTINCT` is added to that clause shape and the engine rejects/errors on it, the fix
   needs either an `ORDER BY` drop-if-absent-from-RETURN rule or a verified confirmation this
   build tolerates it — this is exactly the class of live-verified dialect fact this lab tracks in
   `claude/graph-dba/falkordb-quirks.md`, not something to assume.

**Projected effect (a projection from these 7 probes generalizing, not a measured result — the
acceptance test in §5 is what actually confirms it):** A+B fully resolve within `catalog` (10/10 of
that subset's misses), which would put the `catalog` subset at 20/20; C resolves 3 of
`knowledge_base`'s 5 real-bug misses, leaving D's 2 and the 6 structurally-expected zeroes,
putting `knowledge_base` at roughly 11/19 (≈58%) — still under the 75% AC-2 gate on its own, but a
large jump from 42.1%. Overall, roughly 29/39 (≈74%) — still short of the 85% overall gate, but a
28-point jump from 46.2%, using only mechanism-level changes with no model dependency. **State
this as a projection, not a promise**: it assumes the 7-probe sample's patterns hold across the
other 13 A/B/C-category pairs not individually re-probed, which the re-run in §5 verifies for
real.

### Priority 2 — prompt hardening in `tools.py` (category D, defense-in-depth for A/B; 2/21 misses directly, risk-reduction on the rest)

The report's own diagnosis that `_QUERY_REQUEST_INSTRUCTIONS` has "no examples, no few-shot, and
no explicit guidance on which field should be the filter vs. the projection" is correct as a
description of the prompt, even though the probes show it is not the dominant cause of the
*measured* failures (§2-3). It is still worth fixing, both for category D (which has no
DSL-level remedy — "should I return `name` or `entityId`" is a semantic judgment the compiler
cannot make for the model) and as defense-in-depth so a future model swap doesn't reintroduce A/B
even after the DSL guards are in place. Concrete replacement text for
`_QUERY_REQUEST_INSTRUCTIONS` (`tools.py:837-866`):

```python
_QUERY_REQUEST_INSTRUCTIONS = (
    "You translate a natural-language question into a small, structured "
    "query against ONE graph dataset. Reply with a single JSON object and "
    "nothing else, in exactly this shape:\n"
    '{{"matches": [{{"var": "<short lowercase identifier, e.g. \'p\' or \'e\'>", '
    '"label": "<one of the labels listed below>", '
    '"filters": [{{"property": "<one of the properties listed for that label>", '
    '"op": "<one of = <> < <= > >=>", "value": <a bare JSON number for a '
    'numeric property (e.g. 50, never "50"), a bare JSON string for a text '
    'property, or true/false>}}]}}], '
    '"returns": ["<var>.<property>", or "count(<var>)"/"count(<var>.<property>)"/'
    '"avg(...)"/"min(...)"/"max(...)" the same way], '
    '"order_by": "<var>.<property>" (omit if not sorting), '
    '"order_dir": "ASC" or "DESC" (default ASC), '
    '"limit": <integer between 1 and 50, default 20>}}\n\n'
    "`matches` has exactly one entry. Use ONLY the labels and properties "
    "listed below for this dataset — never invent one. Add a filter ONLY for "
    "a condition the question actually states: a superlative question "
    "(\"cheapest\", \"most expensive\") needs order_by + limit, never an "
    "invented filter with no basis in the question. When the question names "
    "a specific item or entity, filter on its plain `name` property using "
    "the exact text the question uses — never a `*Normalized` property "
    "(e.g. `nameNormalized`), those hold internal lower-cased values you do "
    "not have. When the question asks to list, identify, or classify "
    "entities, return `<var>.name` — never `<var>.entityId`, which is an "
    "internal identifier the reader cannot use. Reply with your best single "
    "JSON object even if you are unsure; never reply with prose.\n\n"
    "Examples (the schema differs per dataset — only the pattern matters):\n"
    '- "How much does the Wireless Charging Pad cost?" -> '
    '{{"matches": [{{"var": "p", "label": "Product", "filters": '
    '[{{"property": "name", "op": "=", "value": "Wireless Charging Pad"}}]}}], '
    '"returns": ["p.price"]}}\n'
    '- "Which products cost less than $50?" -> '
    '{{"matches": [{{"var": "p", "label": "Product", "filters": '
    '[{{"property": "price", "op": "<", "value": 50}}]}}], '
    '"returns": ["p.name"]}}\n'
    '- "Which entities are of type Location?" -> '
    '{{"matches": [{{"var": "e", "label": "Entity", "filters": '
    '[{{"property": "type", "op": "=", "value": "Location"}}]}}], '
    '"returns": ["e.name"]}}\n'
    '- "Which product is the cheapest?" -> '
    '{{"matches": [{{"var": "p", "label": "Product", "filters": []}}], '
    '"returns": ["p.name"], "order_by": "p.price", "order_dir": "ASC", '
    '"limit": 1}}\n\n'
    "This dataset's schema:\n{dataset_schema}"
)
```

Four targeted additions, each tied to a named failure category above: the bare-number
clarification + example (A), the `*Normalized`-is-internal rule + example (B), the
name-not-entityId rule + example (D), and the no-invented-filter rule + superlative example (the
`nlq-16` compounding defect). Not a rewrite of the whole contract — every existing constraint
(`matches` cardinality, allowed `op`s, JSON-only reply) is unchanged.

### Priority 3 (not recommended yet) — model routing

**The evidence does not support "model undersized for this task shape" as the primary cause.**
The 7 probes show the model already produces the semantically-correct filter/projection choice in
several cases studied here (`nlq-21`, the category half of `nlq-11`, the order/limit half of
`nlq-16`) — the defects are narrow, specific, and each independently reproducible (a quoted
number, a non-normalized value against a normalized field, a missing `DISTINCT`, a wrong property
name). That is a different shape of problem than a capability ceiling (e.g., the model failing to
identify *any* relevant filter at all, or ignoring the schema's allowlist). **Recommend deferring
K-056-style model-routing consideration until after Priority 1 (and, if needed, Priority 2) land
and the golden set is re-run** — routing to a larger local model (K-056's `mistralai/ministral-3-3b`
precedent, or a hosted model) changes this demo's infra footprint and cost profile, which per the
brief's own framing is a genuine stakeholder decision, not one this note should make unilaterally.
Named here as the fallback if the re-run in §5 still falls short of the 85%/75% gates after DSL +
prompt fixes, not as the first move.

**On K-056:** checked whether that defect's investigation (a different failure mode — tool-call
skipping over long multi-turn conversations) applies here. It does not, directly: `QueryGraphDataTool`'s
internal structured-completion call (`tools.py:947-951`) is a **fresh two-message call each time**
(system + user only, no accumulated conversation history) — the context-length-driven degradation
K-056 characterizes has no analog here since there is no long context to degrade. The two defects
are both real qwen3-4b-2507 weaknesses, but distinct ones; K-056's finding does not transfer as
evidence for this mechanism's failure mode, and this RCA's own findings should not be read as
confirming or extending K-056's.

## 5. Evaluation design for the fix (what re-proves this)

Per the ml note's own Layer 1 gate (§5, unchanged — this RCA does not revise the threshold):

1. **After Priority 1 (DSL fixes A/B/C) lands**, re-run the full 39-pair harness
   (`server/tests/eval/run_nlq_golden_set_eval.py`) unmodified, same model/config. Expect (not
   guaranteed — this is the actual test): `catalog` subset materially recovers toward 20/20;
   `knowledge_base` subset recovers `nlq-21/22/23` (the duplicate-entity cases) but not
   `nlq-25/26` (still Priority 2's fix). Report the new overall/subset/per-shape numbers with
   Wilson CIs exactly as the existing report does — do not hand-wave a pass from the probe-level
   projection in §4.
2. **After Priority 2 (prompt fix D) lands on top**, re-run again. If `knowledge_base` still
   doesn't clear 75% and overall still doesn't clear 85%, that is the point to escalate to
   Priority 3 with real, current-run numbers instead of this RCA's projections.
3. **No change to the golden set itself** — every pair's `expected` value is unaffected by any of
   these fixes; this RCA does not reopen the golden set's correctness (already twice
   `analyst`-reviewed, per the brief).
4. **Regression check, not just improvement check:** confirm the 18 already-passing pairs
   (including all 6 not-found pairs at their 0% false-answer rate) still pass after each fix
   round — particularly worth checking that the scoped `DISTINCT` fix (C) doesn't perturb
   `nlq-31/32/33`'s aggregate counts, and that catalog fixes don't perturb the compound-filter
   passes that already work correctly on the non-numeric half.

## 6. Risks & open questions

1. **The 7-probe sample is not the full 21-failure population.** I did not individually re-probe
   `nlq-10, nlq-12, nlq-17, nlq-20, nlq-22, nlq-23, nlq-26` — their categorization (A/A/A/A/C/C/D
   respectively) rests on (a) the actual eval's `toolResult` shape matching the exact signature of
   a category confirmed elsewhere in the same shape-family (all catalog price-filter/aggregation
   misses share `nlq-08`/`nlq-11`/`nlq-16`/`nlq-18`'s identical empty/zero-count signature; both
   remaining single-fact `knowledge_base` misses match `nlq-21`'s exact multi-row-of-one-value
   signature and the golden set's own documented duplicate-node counts for those names) and (b)
   the ml note's own "restrict live calls" posture. This is a reasoned inference from the fully
   observed pattern, not a second independent live confirmation for each — the §5 re-run is what
   actually closes this gap.
2. **The typed-`DatasetSchema` fix (A) is a real (if small) design change**, not a one-line patch
   — it touches `querygen.DatasetSchema`'s shape and both registry entries. If the implementer
   judges that out of scope for a fast follow-up, the untyped heuristic coercion is a safe
   fallback for *this* schema (no property in `CATALOG_SCHEMA`/`KNOWLEDGE_BASE_SCHEMA` is
   string-typed-but-numeric-looking today) but is a heuristic, not a contract — flagged so it
   isn't silently treated as equivalent to the typed fix if a future dataset adds one.
3. **The `DISTINCT`/`ORDER BY` dialect question (C) is unresolved and routes to `graph-dba`**, not
   this note — see §4.1.3. Do not ship the `DISTINCT` change without that confirmation; a
   compile-time change that produces a runtime engine error on `nlq-25`-shaped queries would trade
   one failure mode for a worse one (an exception instead of an abstention).
4. **This RCA does not re-litigate the golden set or the Layer 1 scoring rules** — the "exactly
   one row" scalar rule (ml note §3) is *correct as a scoring contract* for the demonstrated design
   intent (a scalar fact should be one fact); the fix is at the data/query layer (deduping via
   `DISTINCT`), not a scoring-rule relaxation, because relaxing the rule would also mask a genuine
   future regression where a query wrongly matches multiple *different* entities, not just
   duplicate copies of the same one.
5. **Layer 2 rendering anomalies noted but not root-caused here** (`nlq-19/27/31/33` pass Layer 1
   but the render step said "not found" anyway) — out of this RCA's scope (Layer 1 is the FR-4/AC-4
   gate; Layer 2 is a non-gating sanity signal per the ml note §5) but worth a follow-up look if
   Layer 2 quality becomes load-bearing for the AC-5 live demo; the `nlq-21` case in this run
   suggests at least one instance is explained by the same duplicate-row problem C addresses, which
   may resolve some of these as a side effect of the C fix.

## Artifacts

- Raw evidence read in full: `server/tests/eval/nlq_eval_results.json`.
- Golden set (ground truth, not re-litigated): `server/tests/eval/nlq_golden_set.jsonl`.
- Prior report: `docs/test-reports/workflow-nl-query-generation-report.md`.
- Mechanism read: `server/falkorchat/querygen.py`, `server/falkorchat/tools.py` (`QueryGraphDataTool`,
  `_QUERY_REQUEST_INSTRUCTIONS`, `_build_query_request_system_prompt`), `server/falkorchat/services.py`
  (`run_structured_query` — confirmed a pure pass-through, no post-processing/dedup at that layer).
- Live probe scripts (scratchpad, not shipped): `probe.py`/`probe2.py`/`probe3.py` under this
  session's scratchpad directory — reconstructed system prompts + raw completions quoted verbatim
  in §3.
- This RCA: `docs/reviews/workflow-nl-query-generation-rca.md`.
