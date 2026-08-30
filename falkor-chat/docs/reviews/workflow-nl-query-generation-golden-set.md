# NL-query-generation golden set — semantic content gate (U29b-gate)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-055 (M6)

**CPG:** considered, not relevant — this is a content/correctness review of a 39-pair JSONL golden
set and its small structural-integrity test module, checked by direct live `mcp__cypher__query`
reads against `reference`/`ws:nlq-eval` and by running the pytest suite; a CPG traversal adds
nothing over reading these two files and querying the graphs directly.

## Scope & verdict

Reviewed: `falkor-chat/server/tests/eval/nlq_golden_set.jsonl` (39 pairs) and
`falkor-chat/server/tests/eval/test_nlq_golden_set_integrity.py`, against live ground truth
(`reference` for the 20 `catalog` pairs, `ws:nlq-eval` for the 19 `knowledge_base` pairs),
`falkor-chat/server/tests/eval/nlq_corpus_provenance.json`, the prior corpus content-review gate
(`docs/reviews/workflow-nl-query-generation-corpus.md`, U29-gate), and the golden-set-construction
guidance in `docs/plans/workflow-nl-query-generation-ml.md` §4. This is the semantic content gate
on U29b's fixture, before a later unit builds the scoring harness against it. Did not review the
harness itself (not yet built) or re-derive answers via the actual `query_graph_data` mechanism
(out of scope per the brief — a plain read-only Cypher query per pair is the intended verification
method here).

**Pass 2 (2026-08-29) update: verdict revised to approve with suggestions** — see `## Pass 2`
below. The Pass 1 verdict and findings below are preserved as the original record.

**Verdict (Pass 1): needs changes.** Every one of the 39 `expected` values independently re-derives
correctly against the live graphs — no ground-truth defect, no leakage into the reversed-edge or
garbled-date extraction artifacts the corpus review flagged, no not-found pair at risk of an
accidental partial match. But finding #1 is a genuine, systemic gap against this golden set's
stated purpose (FR-1's "arbitrarily-phrased question," the explicit design risk
`workflow-nl-query-generation-ml.md` §4 itself names): roughly 40% of the 39 pairs are not
independent phrasings at all, they are the same sentence template with one slot substituted. That
does not corrupt any individual pair's correctness, but it does mean a scoring-harness run against
this fixture as-is will not actually measure what FR-4/AC-4 asks for on those pairs. This is
fixable by rewording a subset of questions (no re-deriving of `expected` values needed), not a
full rebuild — but it should happen before a later unit locks in a threshold report against this
exact fixture.

## Findings

### 1. (Major) ~40% of pairs are template-clones within their shape/dataset group, not independent phrasings — undercutting FR-1's "arbitrary phrasing" test purpose

Grouping the 39 questions by `(dataset, shape)` and comparing sentence structure (script run
against the live fixture, see Appendix) shows several shapes are covered by **one literal template
string**, varied only by the substituted entity/category/threshold — not by rewording:

- `knowledge_base`/`single-fact` (4 pairs) — all four are `"What type of entity is {X}?"` verbatim.
- `knowledge_base`/`not-found` (3 pairs) — all three reuse that exact same template.
- `knowledge_base`/`filter-list` (3 pairs) — all three are `"Which entities are of type {X}?"`.
- `knowledge_base`/`aggregation` (3 pairs) — all three are `"How many {X}-type entities are
  there?"`.
- `catalog`/`compound-filter` (3 pairs) — all three are `"Which {category} products cost {less
  than|more than} ${N}?"`.

That is 16 of 39 pairs (41%) where the "arbitrary phrasing" axis contributes nothing — a
scoring-harness pass/fail on these pairs measures slot-substitution correctness, not phrasing
robustness. Two more shapes (`catalog`/`single-fact`, `catalog`/`filter-list`) reduce to two
templates each rather than genuinely varied phrasing. Only `knowledge_base`/`conflicting-facts`
(nlq-38/39 — "How many employees does X have?" vs. "What is X's employee count as of March 2026?")
and `knowledge_base`/`relationship-traversal` (varied only because each pair asks about a
*different relation*, not because of deliberate rewording) show real phrasing diversity.

This is exactly the risk `docs/plans/workflow-nl-query-generation-ml.md` §4 names by name: "not
including [certain shapes] would let a narrow implementation pass on a golden set that doesn't
actually probe 'arbitrary phrasing'" — the same failure mode surfaces here via phrasing homogeneity
rather than shape omission. Its own format example (`"question": "<paraphrased>"`) signals each
`question` field was meant to be an independent paraphrase, not a template instantiation.

**Suggested action:** before a scoring harness is built against this exact fixture, reword a
representative subset of the 16 identically-templated pairs (rotate through 2-3 distinct phrasings
per shape — e.g., "What's the price of X?" / "Can you tell me how much X costs?" / "How much does
X cost?" for `catalog`/`single-fact`) so each shape's pairs are genuinely independent probes of
phrasing robustness, not five renderings of one template. No `expected` value, `shape`, or
`dataset` field needs to change — this is purely a `question`-text revision.

## What's solid

- **Every `expected` value re-derives correctly, independently, against the live graph** — all 20
  `catalog` pairs checked against a fresh `MATCH (p:Product) RETURN name, category, price` dump
  from `reference` (15 products); all 19 `knowledge_base` pairs checked against a fresh
  `MATCH (e:Entity)` dump and direct `RELATES_TO` edge queries from `ws:nlq-eval`. No discrepancy
  found in scalars, sets, counts, min/max, or the two structurally-unanswerable shapes.
- **`relationship-traversal`/`conflicting-facts` ground truth is honest and correctly sourced.**
  nlq-34 ("who did Marlowe Robotics acquire") correctly cites the correct-direction
  `NLQ-EVAL-06` `acquired` edge, live-confirmed, and explicitly steers clear of `NLQ-EVAL-08`'s
  reversed `acquired by` edge that `workflow-nl-query-generation-corpus.md` finding #1 flagged
  (moot in this specific case since both edges share the same object, but the pair's rationale
  correctly documents *why* only one edge is reliable). nlq-38/39's conflicting values (`62` /
  `140 employees`) are both live-confirmed still-distinct edges.
- **No garbled-extraction leakage.** No pair anywhere references the two known-garbled date
  entities (`"January 1026"`, `"September than"`) or the ambiguous CTO entity (`Concept` in one
  document, `Other` in another, no `SAME_AS` edge) as a single-fact type-lookup target — all
  correctly avoided per the corpus review's guidance.
- **`not-found` pairs are genuinely unanswerable, not typo-absent, and safe from partial-match.**
  All 6 (`nlq-13/14/15`, `nlq-28/29/30`) live-verified at 0 rows. `querygen.py:97`'s own comment
  ("match, reject anything else; never coerced/fuzzy-matched") confirms exact-match semantics, so
  the "close to a real name" design (e.g. "Bluetooth Speaker Max" vs. real "Bluetooth Speaker
  Mini") cannot accidentally partial-match.
- **Shape/dataset restrictions are correctly enforced** — `compound-filter` only in `catalog`,
  `relationship-traversal`/`conflicting-facts` only in `knowledge_base`, matching the schema
  constraint the integrity test itself encodes.
- **Structural integrity suite is green**: `pytest tests/eval/test_nlq_golden_set_integrity.py`
  passes all 249 parametrized assertions (unique ids, required fields, registered datasets, valid
  shapes, dataset-shape restriction, valid `expected.type`, ≥2-value conflicting-facts sets,
  per-dataset size range, per-shape-per-dataset minimums).

## Open questions

None that need the caller's input — finding #1 is a concrete, self-contained revision the golden
set's author (or `tico`/`data-scientist`, per the ml note's `analyst`-review posture) can act on
without further discussion.

## Pass 2 (2026-08-29) — re-gate after rewording fix

Re-reviewed after `tdd-engineer` reworded 18/39 `question` fields (commit `7cf3247`, diffed
directly with `git show` — confirmed the diff touches only `question` strings, no `id`/`dataset`/
`shape`/`expected`/`rationale` field anywhere) to close Pass 1 finding #1.

**Verdict: approve with suggestions.**

**Finding #1 disposition: substantially fixed, one residual gap (downgraded Major → Minor).**
Re-ran the same `(dataset, shape)` template-grouping method as Pass 1's Appendix. Every catalog
group (`single-fact`, `filter-list`, `compound-filter`, `not-found`) now has as many genuinely
distinct sentence structures as pairs — confirmed by rereading all 20 catalog questions.
`knowledge_base`/`filter-list` and `knowledge_base`/`aggregation` are likewise now fully
diversified (three distinct structures each: "of type X" / "classified as X" / "list all the
X-type" for filter-list; "how many X-type... there" / "what's the count of X-type" / "how many...
classified as X" for aggregation).

`knowledge_base`/`single-fact` (4 pairs) still contains one exact-duplicate template: `nlq-21`
("What type of entity is Marlowe Robotics?") and `nlq-24` ("What type of entity is NovaGrid?") are
byte-identical in structure — neither was among the 18 reworded ids, so this pair was never
touched. `nlq-22` ("What kind of entity is Atlas-7?") is only a `type`→`kind` synonym swap of the
same skeleton, not a structural rephrasing; only `nlq-23` is a genuine restructure ("Can you tell
me what type of entity Devon Cole is?"). `knowledge_base`/`not-found` (`nlq-28/29/30`) mirrors the
same three templates one-for-one (internally still three distinct strings, so it individually
passes, but it borrows rather than independently diversifies). Net: of the 7
single-fact+not-found pairs, only 2 (`nlq-23`, `nlq-30`) are genuine rephrasings; the rest are the
original template or a one-word synonym of it. This is a small, easily-fixed residual (reword
`nlq-21` or `nlq-24` — they cannot both stay verbatim-identical within one 4-pair group) — not
severe enough on its own to re-block the harness unit, since it affects one duplicate pair rather
than a systemic pattern, but worth a follow-up edit before this fixture's numbers are reported as
"tested against arbitrary phrasing."

**Spot-check (item 2): reworded questions still match their unchanged `expected`/`rationale`.**
Checked `nlq-04`, `nlq-09`, `nlq-12`, `nlq-13`, `nlq-23`, `nlq-27`, `nlq-32` — all still ask exactly
what their `expected` value answers, no drift. One observation, not a defect: `nlq-11` was
reworded to "Are there any Accessories products priced above $30?", a yes/no-shaped question,
while `expected` is still the full 2-item set. This is fine for Layer 1 (which checks the
structured query result, not the rendered sentence form — `docs/plans/workflow-nl-query-generation-ml.md`
§5 draws that line explicitly), but flagging it since a yes/no phrasing is a slightly different
information need than "which" — worth keeping in mind if a future pass reworks Layer 2 rendering
checks against this same fixture.

**Regression check (item 3): "what's solid" areas unaffected, confirmed not assumed.**
`git show 7cf3247` confirms zero changes to `id`, `dataset`, `shape`, `expected`, or `rationale`
across all 39 rows — the not-found pairs still name the same absent entities/products (so live
0-row genuineness is unaffected), `relationship-traversal`/`conflicting-facts` (`nlq-34..39`) are
untouched (so the reversed-edge/garbled-date avoidance still holds), and `compound-filter`/
`relationship-traversal`/`conflicting-facts` dataset restrictions are unchanged. Re-ran
`pytest tests/eval/test_nlq_golden_set_integrity.py`: 249/249 pass, unchanged from Pass 1.

## Appendix: template-grouping evidence for finding #1

```
---- (catalog, single-fact)           2 templates / 5 pairs
   How much does the Wireless Charging Pad cost?
   What category is the Portable SSD 1TB in?
   How much does the Smartwatch Series 5 cost?
   What category is the Action Camera 4K in?
   How much does the 27-inch 4K Monitor cost?
---- (catalog, filter-list)           2 templates / 4 pairs
   Which products are in the Peripherals category?
   Which products are in the Wearables category?
   Which products cost less than $50?
   Which products are in the Audio category?
---- (catalog, compound-filter)       1 template / 3 pairs
   Which Peripherals products cost less than $50?
   Which Accessories products cost more than $30?
   Which Wearables products cost less than $200?
---- (catalog, not-found)             reuses single-fact's 2 templates / 3 pairs
   How much does the Bluetooth Speaker Max cost?
   What category is the Wireless Earbuds Pro in?
   How much does the Gaming Keyboard RGB cost?
---- (knowledge_base, single-fact)    1 template / 4 pairs
   What type of entity is Marlowe Robotics?
   What type of entity is Atlas-7?
   What type of entity is Devon Cole?
   What type of entity is NovaGrid?
---- (knowledge_base, not-found)      same template as single-fact / 3 pairs
   What type of entity is Solstice Robotics?
   What type of entity is Griffin Aerospace?
   What type of entity is Helio Dynamics?
---- (knowledge_base, filter-list)    1 template / 3 pairs
   Which entities are of type Location?
   Which entities are of type Person?
   Which entities are of type Product?
---- (knowledge_base, aggregation)    1 template / 3 pairs
   How many Organization-type entities are there?
   How many Person-type entities are there?
   How many Concept-type entities are there?
```

`(catalog, aggregation)` (5 pairs) and `(knowledge_base, relationship-traversal)` (4 pairs) are the
two shapes with genuine template diversity, and `(knowledge_base, conflicting-facts)` (nlq-38/39)
is the one pair-of-pairs that is a deliberate same-fact reword.
