# Natural-language query generation over structured graph data — golden-set evaluation method note

> **Status:** archived · **Owner:** `data-scientist` · **Tracks:** K-055 (M6) · **Version:** 2

**2026-08-30 revision:** §5's gate formula amended to exclude shapes the shipped mechanism is
*structurally and permanently* incapable of by design (`relationship-traversal`,
`conflicting-facts`) from the Overall/AC-2 pass/fail denominators — pooling a permanent, by-design
0% into a ≥85%/≥75% target made the gate mathematically unattainable regardless of mechanism
quality; see the new paragraph after §5's table and `docs/reviews/workflow-nl-query-generation-rca.md`
for the finding that prompted this.

## 1. The question and the decision it serves

**Question:** how do we know whether the NL-query-generation mechanism (whatever `architect`
chooses — LLM-generated Cypher, a constrained query-builder DSL, or something else) actually
answers an arbitrarily-phrased question *correctly*, and how do we know it generalizes across a
structured dataset it wasn't built against?

**Decision this serves:** `workflow-nl-query-generation.md` FR-4/AC-4 require a golden-set
evaluation methodology with a defined passing bar before this capability is considered complete
— "does it actually answer correctly" is not something the FR-5/AC-5 live demo can establish on
its own (one arbitrarily-phrased question, one run, no denominator). This note is that
methodology: the pair format, how the golden set is built and how big it needs to be, the metric,
the threshold, and how it gates AC-2's schema-generality requirement. Implementation of the
harness routes to `coder`/`tdd-engineer` against this note, the same handoff shape as
`docs/plans/graphrag-eval-ml.md` → `docs/plans/graphrag-eval.md`.

**Explicitly out of scope (per the brief):** FR-3/FR-3a's adversarial safety test cases are
`security-expert`'s territory — this note does not design them. The specific query-generation
technology is `architect`'s call, made in parallel — this note is written to be evaluated against
whichever mechanism lands, and flags (§6) the one place a mechanism choice changes what the
harness needs.

## 2. Findings from the real system

**Neither candidate dataset is built yet — this evaluation has a step-0 data dependency, same
shape as `graphrag-eval-ml`'s `ws:eval`.**

- **Dataset 1 (electronics catalog, AC-1).** `workflow-catalog-lookup.md` (sibling, also Ready
  for design, no architect plan yet) specifies a flat demo catalog — name/category/price only,
  seed-script-only (FR-5/FR-7) — but it does not exist in the graph today and its concrete node
  shape (`CatalogItem{name,category,price}` or similar) is `architect`'s call to make, not fixed
  here. This note's golden set for dataset 1 must be authored **against whatever schema the
  catalog-lookup capability actually ships**, not a shape invented in this document — flagged as
  a coordination dependency in §6, not resolved here.
- **Dataset 2 (extracted-entity graph, AC-2 candidate).** `docs/DESIGN.md` §5.1/§7.1 confirm the
  shipped schema: `(:Document)-[:HAS_CHUNK]->(:Chunk)-[:ABOUT]->(:Entity{entityId,name,type})`,
  `(:Entity)-[:RELATES_TO{label,sourceChunkId,sourceDocumentId,createdAt}]->(:Entity)` (free-text
  predicate, never deduplicated), `(:Entity)-[:SAME_AS{status,confidence,technique,...}]->(:Entity)`.
  `Entity.type` is a closed 7-value enum (`Person, Organization, Location, Product, Event,
  Concept, Other` — `docs/plans/document-ingestion-ml.md` §3.1). K-050/M5 is closed
  (2026-08-25, `docs/HISTORY.md`) and its own acceptance pass
  (`docs/test-reports/document-ingestion-report.md`) drove real ingestion, extraction and fusion
  live against `ws:acme` — this is real, running infrastructure, not a stale reference.

**Volume/variety check on `ws:acme`'s existing extracted-entity data — insufficient as-is for a
golden set, verdict below.** Reading the QA report's own evidence table (TP-04 through TP-10):
the entities/relationships actually sitting in `ws:acme` today are a handful of single-sentence
synthetic fixtures purpose-built to exercise **fusion-tier mechanics** (auto-merge, pending-match,
confirm/reject/recheck), not to be a representative Q&A corpus:

| Fixture | What it produced |
|---|---|
| "Griffin Aerospace acquired Solstice Robotics in 2024." | 2 `Organization` entities, 1 `RELATES_TO{label:"acquired"}` |
| "Meridian Analytics has 40 employees." / "...400 employees." | 1 entity, 2 conflicting `RELATES_TO{label:"has"}` edges (deliberately kept both, FR-6) |
| "Northbridge Systems provides cloud infrastructure services." (×2, verbatim) | 2 duplicate entities → auto-merged |
| "Talbridge Manufacturing..." / "Talbridge Manufacturing Inc..." | 2 near-duplicate entities → pending fuzzy match |
| "Pinnacle Freight operates a nationwide trucking network." (×2, cross-document) | 2 duplicate entities → auto-merged |
| One 8,366-char/12-chunk synthetic doc (TP-01) | Content not fact-oriented (byte-identity retrieval test only) |

That is on the order of **10-12 entities total, of which only `Organization` and generic `Other`
stub types are exercised** (no `Person`, `Location`, `Product`, `Event`, `Concept` instance
anywhere in this set), and on the order of **6-8 relationships**, almost all `acquired`/`has`/
duplicate-name patterns. This is exactly the failure mode `graphrag-eval-ml` named for its own
corpus risk ("a golden set over a toy/demo corpus measures a toy problem") — reusing it as-is
would produce a golden set that is trivially small, type-homogeneous, and shaped by fusion-testing
concerns rather than by what a NL-query-generation mechanism actually needs to be stressed on
(multi-hop traversal, attribute lookup across varied entity types, conflicting-fact handling,
genuine not-found cases).

**Verdict on the document-ingestion schema as AC-2's second dataset: yes for the *schema*, no for
the *existing data*.** The schema itself is a strong, non-synthetic fit for AC-2 — genuinely
different shape from the flat catalog (graph traversal + free-text-predicate edges + duplicate/
conflicting-fact handling vs. flat attribute rows), real production infrastructure, zero
extra engineering to stand up. But it needs a **purpose-built ingestion pass**, not the QA
fixture data, before a golden set can be authored against it — see §4's "Corpus, per dataset."

**Extraction non-determinism is a distinct validity risk this dataset carries that the catalog
dataset does not.** The catalog is a deterministic seed script — a golden answer can be derived
directly from the seed data with certainty. The entity graph is populated by an **LLM extraction
step** whose own method note (`docs/plans/document-ingestion-ml.md` §6, F2/F4) explicitly flags
extraction quality on nested schema as "genuinely unmeasured" — the model may under-extract (miss
a real entity/relationship) or over-extract (fabricate one). A golden pair authored from the
*source document text* rather than the *actual post-extraction graph content* risks asserting a
"correct" answer the graph was never actually given a chance to produce — a false negative against
the query-generation mechanism for a failure that is actually upstream, in extraction. §4 makes
this a hard authoring rule, not a caveat.

**This capability's ground truth is closed-form and structured, unlike GraphRAG's.** The
`graphrag-eval-ml` precedent evaluates whether a *generated natural-language answer* is faithful
to retrieved *unstructured* chat text — an open-ended judgment that legitimately needs
LLM-as-judge, calibrated against human labels, because "is this answer grounded" has no
mechanical check. Here, the expected answer to "how much does the X cost" is a specific number
that exists verbatim in the graph; the expected answer to "which laptops are under $1000" is a
specific, enumerable set of rows. **This is a text-to-query execution-accuracy problem (the same
shape as Spider-style text-to-SQL evaluation), not a RAG-faithfulness problem** — the natural
instrument is exact comparison against ground truth, not a judge's subjective call. This is the
central methodological difference from the precedent and drives the metric choice in §5.

## 3. Recommended method

**Two-layer evaluation, primary layer deterministic execution-accuracy (no LLM judge needed for
the gate), secondary layer a deterministic end-to-end answer check** — mirroring
`graphrag-eval-ml`'s "measure two things separately" discipline, but with a different Layer 1
because the ground truth here is discrete facts, not open text.

### Layer 1 — Execution accuracy (primary; the FR-4/AC-4 gate)

For each golden question, compare the mechanism's **structured result** — whatever rows/values it
actually derived from the graph, before any natural-language rendering — against a golden expected
answer, using **exact match after canonicalization**, not LLM judgment:

- **Scalar fact** (a price, a category, an entity's type, a relationship's object) — exact value
  match; strings compared case-folded/whitespace-collapsed (mirrors the codebase's own
  `Entity.nameNormalized` convention, `docs/DESIGN.md` §5.1), numbers compared with an epsilon
  (0.01) for float rounding, never string-equal on a formatted price.
- **Set/list** (a filter/list question — "which laptops are under $1000", "what did Griffin
  Aerospace acquire") — compare as an **unordered set** of identifiers (item names / entity
  names), never sequence-equal — result ordering is an implementation detail this mechanism has no
  obligation to fix, and gating on it would fail a correct answer for the wrong reason.
- **Conflicting facts kept, not merged** (document-ingestion dataset only, per FR-6 — e.g. the
  Meridian Analytics 40-vs-400-employees case) — the golden expected answer for this category is
  itself a **set of ≥2 values**, and the mechanism is scored correct only if it surfaces all of
  them (or explicitly states the conflict) — silently picking one of two conflicting facts is a
  wrong answer for this category, not a partial credit case, because it hides information FR-6
  requires be surfaced.
- **Not-found / abstention** — a distinguished `NOT_FOUND` sentinel result; a golden pair in this
  category is scored correct only if the mechanism's structured result is empty/absent, never a
  fabricated value. See §5 for why this category is gated separately and more strictly.

**Requires a harness seam most mechanisms don't have yet — flag to `architect`/implementer, not
resolved here:** whatever mechanism ships needs to expose its raw pre-rendering result to the
harness (a debug/inspection return value, not just the final chat-facing sentence), the same class
of gap `graphrag-eval.md` D-noted for `AgentResponder` ("no side-effect-free generate-only seam,"
that plan's §2 finding on `_build_prompt`). Without it, Layer 1 cannot run and the evaluation is
forced onto Layer 2 alone, which is a strictly weaker instrument (§5 risk 1).

### Layer 2 — End-to-end natural-language answer check (secondary; the live-demo-level sanity check)

Because the golden answer is a short factual value, not open narrative, **the default instrument
is a deterministic value-extraction/containment check** (does the final rendered sentence contain
the expected number/name/set, via a small regex or normalized-substring check per answer type) —
**not LLM-as-judge by default.** This is the opposite default from `graphrag-eval-ml`'s Layer 2,
and deliberately so: an LLM judge is the right tool when "is this correct" is a subjective
faithfulness call with no mechanical check; it is the wrong, validity-risk-carrying tool when the
correct answer is a specific number or name a regex can check directly. Reserve an LLM-as-judge
fallback only for a genuinely underspecified case (e.g. a list rendered as free prose where every
item's presence can't be regex-matched reliably) — if that fallback is needed at all, it inherits
every caveat `graphrag-eval-ml` §"Layer 2" already documents (judge ≠ agent-under-test,
calibration before trust, agreement reported not assumed) rather than re-deriving them here.

Layer 2 is **not** the FR-4/AC-4 gate — it is what AC-5's live-demo bar and a coarse
sanity-check need. Layer 1 is the number that gates.

### Rejected alternatives

- **Query-text exact-match** (comparing the generated Cypher/DSL string against a golden query).
  Rejected: mechanism-specific by construction — an LLM-generated Cypher and a constrained-DSL
  query that both produce the correct answer via different valid paths would be wrongly scored
  as different, and query-text similarity says nothing about correctness (a syntactically close
  but semantically wrong query scores well; a stylistically different but correct one scores
  poorly). Execution accuracy is the standard fix for exactly this problem in text-to-SQL
  evaluation and is inherently mechanism-agnostic — the brief's own ask.
- **LLM-as-judge over the final natural-language answer as the primary/only layer.** Rejected as
  the *primary* gate: for closed-form facts, a judge is strictly worse than exact comparison —
  it adds calibration burden and self-preference/verbosity-bias risk for no accuracy benefit,
  and it obscures whether a wrong answer came from bad retrieval/query-generation (Layer 1's job)
  or bad phrasing (Layer 2's job), the same diagnose-before-blaming-either principle
  `graphrag-eval-ml` itself argues for its own two layers. Kept only as Layer 2's fallback, not
  promoted to the gate.
- **Reusing `ws:acme`'s existing QA fixture entities as the AC-2 golden set, unmodified.** Rejected
  per §2's volume/variety finding — too thin, too type-homogeneous, and shaped by fusion-testing
  concerns rather than Q&A coverage. The *schema* is reused; the *data* is not.

## 4. Golden-set construction

**Format** (one JSONL fixture, both datasets in one file, `dataset` field discriminates — see §6
for why one shared harness code path across both datasets matters):

```json
{"id": "nlq-01", "dataset": "electronics-catalog", "question": "<paraphrased>",
 "shape": "single-fact | filter-list | not-found | conflicting-facts | aggregation",
 "expected": {"type": "scalar", "value": 49.99},
 "rationale": "<why this is the correct answer, and where it lives in the seed data>"}
```
`expected.type` ∈ `scalar | set | not_found` (§3's three comparison modes); `set` carries
`{"type":"set","values":["Item A","Item B"]}`.

**Question-shape taxonomy to stratify against (not just random paraphrases of one shape):**
single-item exact fact, filter/list (single predicate and, for the catalog, a compound predicate
— category **and** price range, since FR-2/AC-2's sibling `workflow-catalog-lookup.md` only
requires single-predicate filtering, so a compound-filter question is specifically testing this
capability's claimed generality beyond the fixed-shape sibling), not-found/abstention,
aggregation (min/max/count — "which is the cheapest laptop," "how many items are under $500"),
and — document-ingestion dataset only — relationship traversal ("who did X acquire") and
conflicting-facts (§3). Aggregation and compound-filter are the shapes most likely to expose a
mechanism that only handles simple single-predicate lookups well; not including them would let a
narrow implementation pass on a golden set that doesn't actually probe "arbitrary phrasing."

**Corpus, per dataset (step 0, before golden pairs are authored — mirrors `graphrag-eval-ml`'s own
sequencing):**

- **Electronics catalog:** seeded once by `workflow-catalog-lookup.md`'s own seed script (a
  dependency this note does not build) at ≥15-20 distinct items spanning ≥3 categories with a
  real price spread, so filter/aggregation questions have genuine non-trivial answers (a 3-item
  catalog can't support a meaningful "cheapest" or "under $X" question). Golden answers are
  derived directly from the seed script's own data — no extraction step, so no non-determinism
  risk (§2's document-ingestion-specific concern doesn't apply here).
- **Document-ingestion entity graph:** a **fresh, purpose-built ingestion pass** into a dedicated
  workspace (not `ws:acme`'s QA fixtures) — on the order of 10-15 short synthetic documents,
  deliberately spanning multiple `Entity.type` values (not just `Organization`), a mix of
  relationship predicates, at least one deliberate conflicting-fact pair (mirrors the
  Meridian-Analytics precedent, needed to exercise FR-6's category), and at least one genuinely
  absent fact (for the not-found category). **Golden answers are verified against the actual
  post-ingestion graph content, read directly, not assumed from the source document text** — the
  hard rule from §2's extraction-non-determinism finding. If extraction misses or garbles a fact
  the corpus intended, that document is either rewritten to extract cleanly or the golden pair is
  dropped — never scored against a "should have been extracted" answer the graph doesn't actually
  contain, which would make Layer 1 measure extraction quality (already `document-ingestion-ml`'s
  own tracked, separate risk) rather than query-generation quality.

**Authoring workflow:** LLM-drafts-then-verifies, `analyst`-reviewed (topical/shape coverage, no
answer-key leakage into a workflow prompt), same substitution-for-human-verification posture
`graphrag-eval.md` §4 already named and flagged as the coordinator's call, not re-litigated here.

**Size.** Per-dataset, not pooled, because AC-2 gates the second dataset on its own (§5):
**catalog ≈ 20-25 pairs, document-ingestion ≈ 15-20 pairs** (smaller because the corpus itself is
smaller and each fact is authored/verified by hand, not read off an existing flat table),
stratified across the shape taxonomy above with **at least 3 pairs per shape per dataset** so a
shape-level breakdown (§5) is more than a single anecdote. Total ≈ 35-45 pairs — same order of
magnitude as `graphrag-eval-ml`'s 30-50, for the same reason: small enough to hand-verify and
re-verify when a schema or seed script changes, large enough that one flipped case doesn't swing
the headline number by double digits.

## 5. Evaluation design (metric · data · threshold)

| Layer | Metric | Data | Acceptance threshold |
|---|---|---|---|
| Layer 1 (gate) | **Execution accuracy** — exact match after canonicalization (§3), computed over **in-scope pairs only** — see the exclusion rule below the table | 35-45 golden pairs, stratified by shape, split `electronics-catalog` / `document-ingestion-entities` | **Overall ≥ 85%** (Wilson 95% CI reported alongside, never substituted for it — at n≈40 a single flipped case moves the point estimate ~2.5 pts, and the CI at that n is wide enough that 85% and 80% are not reliably distinguishable; the number is a directional pass bar, not a precision claim). **AC-2 gate: `document-ingestion-entities` subset (the `knowledge_base` dataset key in the shipped harness) ≥ 75% on its own**, evaluated separately, not folded into the pooled number — a mechanism that aces the catalog and fails the second schema must not pass on the average. |
| Layer 1, not-found/abstention shape | **False-answer rate** (fraction of not-found-shape questions where the mechanism returned a specific value instead of `NOT_FOUND`) | the not-found-shape subset (≥3 per dataset, §4) | **≤ 10%**, gated *separately and more strictly* than the general bar — a fabricated specific-sounding wrong fact is a costlier failure than a generically wrong count (mirrors this lab's own asymmetric-error-class convention for a biased judge, K-027 item 3: gate the costly error class on its own, don't let it hide inside a symmetric average). |
| Layer 1, shape breakdown | Execution accuracy per shape (single-fact, filter-list, aggregation, compound-filter, conflicting-facts, relationship-traversal) | same golden set, grouped | **Reported, not independently gated** — per the convention that a probe set's individual outcomes must appear in the summary prose even when only the aggregate is gated, so a real partial-failure pattern (e.g. aggregation failing while single-fact passes) is visible and not averaged away. `conflicting-facts`/`relationship-traversal` specifically are excluded from the *pooled* rows above (see below) whenever they are structurally out of scope — this row still reports their real per-shape score every run, at whatever it actually is. |
| Layer 2 | Deterministic value-containment/extraction match on the rendered NL answer (LLM-judge fallback only if regex genuinely can't cover a shape) | same golden set's `expected` field, applied to the live rendered answer | **Reported as a live-demo/AC-5 sanity signal, not a second independent gate** — Layer 1 is the number FR-4/AC-4 requires; Layer 2 catches a mechanism that gets the right data but renders it unusably (e.g. drops the number from the sentence). |
| Harness safety backstop (only if the chosen mechanism is LLM-generated Cypher) | Zero tolerance: no golden-set run executes a query containing a write clause | every generated query, scanned before execution | **Any occurrence fails the harness run outright** — see §6. This is a harness-level backstop, not a substitute for FR-3's structural prevention or FR-3a's adversarial suite (`security-expert`'s separate scope). |

**Exclusion rule for structurally out-of-scope shapes (added 2026-08-30, per
`docs/reviews/workflow-nl-query-generation-rca.md`).** If the shipped mechanism is, by its own
approved design, structurally and *permanently* incapable of a golden-set shape — not merely
performing poorly on it, but architecturally unable to ever answer it (e.g. a single-`MATCH`-
pattern DSL with no relationship traversal, `workflow-nl-query-generation.md` §3.6) — that shape's
pairs are (a) never dropped from the authored golden set (§6's original instruction stands
unchanged) and are run every time, (b) reported every run as a **named, permanent structural-gap
line** at their actual score (typically 0%), separately from every gated number, but (c)
**excluded from the Overall and AC-2 pooled pass/fail denominators above.** The two shapes
identified as structurally out of scope for the shipped v1 DSL are `relationship-traversal` and
`conflicting-facts` (both `document-ingestion-entities`-only, both requiring a graph traversal the
single-`MATCH` DSL cannot express by design).

This is a **correction to this note's original formula, not a new judgment call**: pooling a
permanent, by-design 0% into a ≥85%/≥75% target makes the gate mathematically unattainable no
matter how correct the mechanism is on every shape it is actually designed to handle — 6 of 39
pairs (4 `relationship-traversal` + 2 `conflicting-facts`) permanently zero caps achievable overall
accuracy at 33/39 = 84.6% (just under the original 85% bar) and caps the `knowledge_base` subset at
13/19 = 68.4% (under the 75% bar), regardless of in-scope answer quality — exactly the numbers the
real re-run against the fixed mechanism produced. A gate no spec-compliant implementation can ever
pass is not a valid acceptance instrument. **This does not retroactively bless a mechanism that
also underperforms on the shapes it is actually designed for**: the original failing run (18/39
overall, `docs/test-reports/workflow-nl-query-generation-report.md` first run) still fails under
this revised formula too (18/33 in-scope = 54.5%, still well under 85%) — the revision removes an
unattainable floor, it does not lower the bar on anything the mechanism is supposed to be able to
do.

**Why 85%/75%, not a higher or more "rigorous"-looking number.** Neither dataset nor mechanism
has a prior baseline to justify a self-referential regression gate the way `graphrag-eval-ml`'s
recall@10 does (that note had a "first run establishes the baseline, don't gate it" posture
precisely because no prior number existed). Here, FR-4/AC-4 wants a real pass/fail bar at first
build, not a baseline-record. 85% overall / 75% on the harder, thinner, extraction-dependent
second dataset is chosen as a level that (a) tolerates the real error rate expected from
arbitrary phrasing generated/interpreted by a local model, consistent with this lab's own
small-model-realism posture, while (b) still requiring the mechanism to be right on the clear
majority of questions, not merely directionally plausible. **This is a judgment call, stated as
one** — if `architect`'s chosen mechanism is materially stronger or weaker than a 4B-class local
model (e.g., a hosted frontier model, or conversely a fully deterministic DSL with no generation
step at all), revisit this number rather than treat it as fixed; it is not derived from any
measured baseline because none exists yet. **This is a different kind of gap from the exclusion
rule above, and the two should not be conflated:** a materially different *model* changes what
accuracy level is realistic and is grounds to revisit the *numeric* threshold; a mechanism
structurally incapable of an entire *shape* by design is handled by excluding that shape from the
denominator, not by lowering the threshold — treating every "this shape scores badly" case as
grounds for exclusion would let a genuine capability shortfall on a shape the mechanism *can*
attempt (e.g. aggregation/compound-filter, which turned out fully answerable once the real DSL/
prompt defects were fixed, §6) hide behind an exclusion that isn't actually warranted for it.

## 6. Mechanism-dependent flags (for `architect`, resolved here only as flags)

- **A structured-result inspection seam is required for Layer 1 regardless of mechanism.**
  Whether the mechanism is LLM-generated Cypher or a constrained DSL, the harness needs the raw
  rows/values the query produced, not only the final chat-facing sentence. If the design has no
  natural seam for this, add one (a debug-mode return value, or logging the intermediate result)
  — the harness cannot fabricate this data from the rendered text without collapsing Layer 1 into
  Layer 2's weaker instrument.
- **If the mechanism is LLM-generated Cypher specifically:** the golden-set harness should apply
  the write-clause scan in §5's last row as a defense-in-depth backstop for its *own* runs (never
  executing a generated query that contains `CREATE`/`MERGE`/`SET`/`DELETE`/`REMOVE`/
  `GRAPH.CONSTRAINT` etc.) — this is purely an evaluation-harness safety measure so a bad
  generated query during golden-set scoring can't mutate the eval workspace, and is explicitly
  **not** a replacement for FR-3's structural prevention or FR-3a's adversarial test set, which
  `security-expert` owns. If the mechanism is a constrained DSL that is structurally incapable of
  emitting a write operation at all, this row is moot — note which case applies once the
  architecture decision lands.
- **If the mechanism is a constrained query-builder DSL:** Layer 1's canonicalization rules (§3)
  still apply unchanged — execution accuracy is agnostic to how the query was produced, only to
  what it returned. No change needed to this note in that case.
- **Aggregation/compound-filter question shapes (§4) may not be expressible at all under a narrow
  DSL.** If `architect`'s chosen mechanism structurally cannot express, say, "cheapest under a
  category," that is itself a finding this evaluation should surface (a shape-level 0% rather than
  a missing test) — not a reason to drop those golden pairs from the set. Report it as a named gap
  against FR-1/FR-2's "arbitrarily-phrased" ambition rather than silently narrowing the golden set
  to only what the mechanism can do. **(2026-08-30 update.)** As it happened, the shipped v1 DSL
  turned out fully capable of both shapes — aggregation and compound-filter scored 100% once the
  real DSL/prompt defects `docs/reviews/workflow-nl-query-generation-rca.md` found (a numeric
  filter value serialized as a JSON string; an un-normalized value against a normalized property;
  a missing `DISTINCT`) were fixed. The shapes that turned out genuinely, structurally impossible
  were `relationship-traversal`/`conflicting-facts` instead (§4, document-ingestion dataset only).
  §5 now codifies the general rule this bullet anticipated: a shape structurally, *permanently*
  unanswerable by the shipped design's own scope is reported at its real score but excluded from
  the pass/fail denominator; a shape the mechanism merely performs poorly on — as aggregation/
  compound-filter did, transiently, before the fix — stays pooled and gated normally. The
  distinction is architectural incapability, not a low score.

## 7. Risks & open questions

1. **Layer 1's harness seam is not guaranteed to exist** — this is the single biggest risk to this
   evaluation running at all as designed (§3's flag, restated). If `architect`'s plan ships with
   only a natural-language-in/natural-language-out surface, Layer 2 (weaker, regex/judge-based)
   becomes the only available instrument, and this note's threshold table (calibrated for exact
   execution match) does not directly transfer — would need revisiting against whatever validity
   Layer 2 alone can support.
2. **The catalog dataset does not exist yet, and its schema is not this note's to fix.** The
   golden set's `electronics-catalog` half is written against `workflow-catalog-lookup.md`'s
   eventual seed data — a coordination dependency, not a blocker to authoring this note, but a
   real sequencing constraint on when Layer 1 can first run for real (after that sibling's seed
   script exists, mirrored by `docs/plans/workflow-nl-query-generation-ml.md`'s own step-0
   framing in §4).
3. **The document-ingestion corpus is purpose-built fresh, not `ws:acme`'s existing data (§2/§4)**
   — this is the right call given the volume/variety finding, but it means real LM Studio calls
   and an `analyst` review pass are on the critical path before that half of the golden set can be
   authored, the same sequencing cost `graphrag-eval.md` Unit 1 already paid for its own corpus.
4. **85%/75% are judgment calls with no prior baseline, named as such in §5** — revisit once
   `architect`'s mechanism choice and its real hardware/model are known; do not treat these numbers
   as derived from measurement they are not derived from. **(2026-08-30 note.)** The specific way
   this could go wrong that this note failed to anticipate up front — a permanently-0%,
   structurally-out-of-scope shape pooled into the denominator making the numeric target
   mathematically unattainable regardless of mechanism quality — is now fixed by §5's exclusion
   rule; the *numeric* values (85%/75%) are unchanged and this note takes no position on whether
   they need revisiting for any other reason.
5. **Not-found/abstention is not an explicit FR in `workflow-nl-query-generation.md`** the way it
   is in the sibling `workflow-catalog-lookup.md` (that document's FR-4/AC-3). This note includes
   it in the golden set and gates it asymmetrically anyway (§5) because a fabrication failure mode
   is real regardless of whether the requirements doc named it — flagged back as a possible
   requirements gap worth a stakeholder confirmation, not something this note unilaterally resolves
   into a new FR.
6. **Golden-set/schema drift.** If `workflow-catalog-lookup.md`'s seed data or the document-
   ingestion entity-type taxonomy changes after this golden set is authored, every affected pair
   needs re-verification (same drift risk `graphrag-eval-ml` risk 4 names for its own golden set) —
   keep both halves small enough (§4's sizing) that a full re-check by hand stays cheap.
