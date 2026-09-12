# `nlq-structured-query` — answerability of the `conflicting-facts` items (methodology review)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** S4 (`nlq-structured-query` pack)

**Verdict: needs changes.** Resolution path **(a)**, with three amendments. `nlq-38`/`nlq-39`
are unanswerable under the only reading of "answerable" that keeps `layer1ExactMatchRate`
meaningful, the unanswerable bucket is **6 of 40**, the accuracy denominator is **34**, and
`docs/plans/small-model-benchmarking.md` §3.8.3's "4" is a **regression of an already-settled,
already-gated ruling** made upstream — not a new judgment call. Option (b) is technically
constructible (exactly one candidate exists in the corpus) and should still be rejected, for
reasons given in §5.

---

## 1. The question and the decision it serves

The implementer is authoring `reference_specs.json` (40 hand-written `QueryRequest`-shaped specs,
one per golden item) whose execution result `refresh_golden.py` uses to stamp each item
`answerable: true|false`. Six specs — the 4 `relationship-traversal` items and both
`conflicting-facts` items — fail Layer B schema validation. The plan says 4. The decision at
stake: what the unanswerable bucket contains, what denominator `layer1ExactMatchRate` is computed
over, and whether the `score_pair` subset-containment branch is real or vestigial. Everything
downstream of that (the printed resolving-power line, the §7.2 adequacy judgement, the plan text)
follows mechanically.

## 2. Findings from the real system

All of the following was read or executed this session; nothing is quoted from memory.

**F-1 (decisive). The 6-item exclusion is already established upstream, and §3.8.3 reverted it.**
`falkor-chat/docs/plans/workflow-nl-query-generation-ml.md` (Status: `archived`, Version 2,
`data-scientist`, revision dated 2026-08-30) §5 carries an explicit *Exclusion rule for
structurally out-of-scope shapes*, verbatim:

> The two shapes identified as structurally out of scope for the shipped v1 DSL are
> `relationship-traversal` **and `conflicting-facts`** (both `document-ingestion-entities`-only,
> both requiring a graph traversal the single-`MATCH` DSL cannot express by design).
> … **6 of 39 pairs (4 `relationship-traversal` + 2 `conflicting-facts`) permanently zero** …

That revision was itself a *correction* prompted by an RCA
(`falkor-chat/docs/reviews/workflow-nl-query-generation-rca.md`), it was gated and executed
against (`falkor-chat/docs/test-reports/workflow-nl-query-generation-report.md` reports
`33/33 = 100.0%` on the corrected in-scope denominator, and its §"`relationship-traversal`/
`conflicting-facts` scoring 0/4 and 0/2 is the expected, correct outcome"), and the coordination
ledger records it as accepted (`falkor-chat/docs/plans/workflow-salesperson-demo-coordination.md`
U32). §3.8.3's "4 `relationship-traversal` items" re-introduces **precisely the defect that
correction removed** — pooling a permanent, by-design 0% into an accuracy denominator — on the
same golden file, one milestone later. This is not a close call between two defensible readings.

**F-2. The empirical evidence §3.8.3 cites for `nlq-34` exists identically for `nlq-38`/`nlq-39`.**
The plan justifies the relationship-traversal exclusion with falkor-chat's stored
`nlq_eval_results.json` record for `nlq-34`. The same file's records for the two conflicting-facts
items are the same class of evidence:

| id | `toolResult` | `layer1Correct` | `layer1Reason` |
|---|---|---|---|
| `nlq-34` | `{"items": [], "finding": "no matching data found"}` | `false` | `expected a single scalar row/column, got []` |
| `nlq-38` | `{"items": [{"count(e.entityId)": 12}]}` | `false` | `missing ['140 employees', '62'] from ['12']` |
| `nlq-39` | `{"items": [], "finding": "no matching data found"}` | `false` | `missing ['140 employees', '62'] from []` |

§3.8.3 applied its own answerability test to one shape and not the other.

**F-3. The structural cause is confirmed against the live graph and the declared schema.**
`KNOWLEDGE_BASE_SCHEMA` (`falkor-chat/server/falkorchat/querygen.py:213`) exposes
`Entity{entityId, name, nameNormalized, type}` only. Live against `ws:nlq-eval`, every one of the
62 `Entity` nodes carries exactly `['entityId','name','nameNormalized','type','createdAt']` — so
**no `Entity` property can ever hold an employee count**, and the two conflicting values exist as
standalone nodes:

```
e.name          | e.type | e.entityId
62              | Other  | d0464decca6d44b9b01037448a62f74b
140 employees   | Other  | 6974fc08ed2f48e4966adae1c0d87dcd
```

reachable from `Marlowe Robotics` only via `RELATES_TO{label:"has"}`, which
`QueryRequest.matches` (`min_length=1, max_length=1`, single-label pattern, no traversal) cannot
express. Mechanically identical to the relationship-traversal case. The golden set's own
`rationale` fields for `nlq-38`/`nlq-39` say exactly this, in capitals.

**F-4 (new, and it is why the fix is not a one-word edit). §3.8.3's answerability test is not
well-defined, and under its literal wording `nlq-38`/`nlq-39` are "answerable" by a degenerate
spec.** §3.8.3 defines the stamp operationally — "runs a hand-written reference spec per item
through the executor at copy time and stamps each item `answerable: true|false`" — without
requiring that the spec be a *faithful answer to the question*. `QueryMatch.filters` is
`Field(default_factory=list, max_length=4)`: **zero filters is valid**. `ws:nlq-eval` has 11
`Entity` nodes of `type = "Other"`, and both `"62"` and `"140 employees"` are among them
(F-3). So:

```json
{"dataset":"knowledge_base","matches":[{"var":"e","label":"Entity",
 "filters":[{"property":"type","op":"=","value":"Other"}]}],
 "returns":["e.name"],"limit":20}
```

is a fully schema-valid spec returning 11 names, and `score_pair`'s conflicting-facts branch —
`expected_set.issubset(actual_set)`, `nlq_scoring.py:190` — scores it **correct**. It does not
answer "How many employees does Marlowe Robotics have?" in any sense; it lists everything the
extractor typed `Other`. The asymmetry with the relationship-traversal items is real and worth
naming: those expect `type: "scalar"`, and `_extract_scalar` demands exactly one row and one
column, so no over-broad query can pass them by accident. Subset containment on a `set` item has
no such guard.

Consequence: the stamp as specified is **author-dependent**. An implementer writing the
semantically faithful spec (source entity + a plausible `employeeCount` property) gets
`answerable: false`; one writing the degenerate spec gets `answerable: true`. A classification
rule that depends on which of two valid specs the author happened to write is not a rule.

**F-5. Exactly one corpus-native property-level conflict exists, and it is extraction noise.**
Sweeping `ws:nlq-eval` for any `nameNormalized` carrying more than one `type` returns a single
row: `chief technology officer → ['Concept','Other']`. That is the *only* conflict in the corpus
reachable by a single-node property query, and it is an extractor type-assignment inconsistency,
not two sources disagreeing about a fact. Relevant to option (b) — see §5.

**F-6. `docs/plans/small-model-benchmarking-ml.md` is already stale on this pack's `n`,
independently of how this question resolves.** §7.2 carries `nlq-generator | 40 | item | McNemar
on Layer-1 exact match | 15.0 pp | 19.1 pp`, and §3's fact table carries
`40 items; 21 scalar / 13 set / 6 not_found`. Both are the *raw item count*, not the accuracy
denominator — they would already be wrong at the plan's own 4-item exclusion (n=36). This needs
correcting whichever way the conflicting-facts question goes.

---

## 3. Answer to Q1 — reclassify. There is no angle where they stay answerable.

The plan's §3.8.3 assumption is **stale and wrong**, on three independent grounds: it contradicts
the upstream note that owns this golden set (F-1), it ignores the same class of stored evidence it
relies on elsewhere (F-2), and it is refuted by the declared schema and the live data (F-3).

The one angle that *superficially* preserves answerability — F-4's degenerate spec — makes the
case for reclassification stronger, not weaker. A spec that reaches the expected set only by
returning a large slice of the table unfiltered is a **scoring-rule loophole, not a capability
demonstration**. `nlq_scoring.py`'s own module docstring already states the governing principle
for the set rule: *"FR-2's contract is to return exactly what was asked, not 'a superset that
happens to contain it.'"* The conflicting-facts branch is a narrow exception to that, intended for
a mechanism that reaches the right entity and returns its conflicting values plus incidental
extras — not a licence for an unfiltered scan.

So `nlq-38`/`nlq-39` join the unanswerable bucket: **6 of 40 unanswerable, 34 answerable.**

---

## 4. Answers to Q2 and Q3

### Q2 — the never-exercised `score_pair` branch

**The premise is fixable, and fixing it is better than documenting it.** "Build the general
mechanism even if this corpus doesn't exercise it" is ordinarily fine and is what the plan's
"the exception must be built rather than discovered" instruction correctly demands. But here the
branch does not have to go unexercised, and the upstream note already says so. Its §5
shape-breakdown row:

> `conflicting-facts`/`relationship-traversal` specifically are excluded from the *pooled* rows
> above … **this row still reports their real per-shape score every run, at whatever it actually
> is.**

and §3.8.3's own exploratory block already lists all seven shapes including
`relationship-traversal` and `conflicting-facts`. So the disposition is: **unanswerable items are
still scored with the full scorer** — they are excluded from the accuracy *denominator*, not from
*scoring*. Under that reading `score_pair`'s containment branch executes against the real pack
data on every single run (returning `False`, which is the informative outcome), and the
"unit-tested against synthetic fixtures only" concern disappears. Make that explicit in §3.8.3
rather than leaving it to the implementer: **excluded from the denominator ≠ not scored.**

Two consequences to legislate while you are there:

- **A "correct" outcome on an item stamped `answerable: false` must be labelled, not printed as a
  win.** F-4's degenerate spec would show `conflicting-facts 1/2` in the exploratory shape split
  while the same item sits in the unanswerable bucket. That is not a contradiction to hide — it is
  a *lucky pass on a structurally unanswerable item*, and it is information. The report should
  name it in those words. Without the label a reader reconciles two numbers by assuming one is a
  bug.
- **Subset containment is gameable on `set` items and should carry a guard when it is ever live
  again** (minor today, because both real items now sit in the unanswerable bucket — but it is a
  latent trap for any future golden-set refresh). The cheapest guard consistent with the existing
  rule: require the spec's `matches[0].filters` to reference the question's subject entity, or
  bound `len(actual_set)` at a small multiple of `len(expected_set)`. Recording the trigger is
  enough for S4; building the guard is not.

### Q3 — what `layer1ExactMatchRate` claims, and the numbers that move

**It measures a cleaner construct at 34 than at 40, and the statistical cost is zero.** This is the
part most likely to be misread, so, precisely:

- **The paired test does not change at all.** An item both arms score 0 on is a *concordant* pair:
  it contributes to neither `b` nor `c`, so McNemar's exact p-value is **identical** whether the
  6 items are in the denominator or out. The count floor — 6 net discordant wins, from
  `2·2^-(m-1) ≤ 0.05` — is likewise invariant. Excluding them is a **re-expression of the same
  test in units that mean something**, not a loss of power. Anyone framing 34-vs-40 as "we gave up
  15% of our sample" has it backwards.
- **What does move is the rate units the bounds print in.** Recomputed this session by exact search
  over the McNemar rejection region, using the same method as `-ml` §7.1 (verified by reproducing
  its published n=40 figure to three decimals, 19.046):

  | `n_eff` | observable floor (α=0.05, truncated) | MDD₈₀ at α=0.05, k=1 (ceilinged) | (exact) |
  |---|---|---|---|
  | 34 (**correct**) | **17.6 pp** | **22.3 pp** | 22.258 |
  | 36 (plan's current 4-item exclusion) | 16.6 pp | 21.1 pp | 21.074 |
  | 40 (`-ml` §7.2 as published) | 15.0 pp | 19.1 pp | 19.046 |

- **`-ml` §7.2's adequacy judgement survives unchanged.** The reference effects this lab has needed
  to resolve — the qwen3-4b turn-4 collapse (97.5% vs 0%) and the ~30 pp ministral
  duplicate-instruction defect — both still clear 22.3 pp. "**Yes, marginally.** Answers 'clearly
  better' only" remains the right words; only the two figures beside it change.
- **`n_effective` for this pack must be 34, and `validate` should say so.** `-ml` §7.1 mandates the
  resolving-power line be "computed from its own `n_effective` and its own α, never hardcoded". If
  `n_effective` is wired to the item count while `layer1ExactMatchRate` divides by the answerable
  count, the report prints a bound over one denominator directly beside a rate over another —
  exactly the two-denominators failure class §7.1 and §11.3 already legislate against elsewhere.
- **Do not "fix" the latency table.** `-ml` §"How the two floors divide the work" lists
  `nlq-generator | Y = 40 | X ≥ 37 | X ≥ 35`. That `Y` is the *executed-item* count for latency
  coverage; all 40 items are still run and timed. It is correct at 40 and must stay there. Two
  legitimately different denominators in the same pack — say so in the plan so nobody reconciles
  them.
- **Composition of the 34, for the shape split's interpretation** (counted from the file):

  | | count | |
  |---|---|---|
  | `single-fact` 9 · `aggregation` 8 · `filter-list` 7 · `not-found` 6 · `compound-filter` 4 | 34 | by shape |
  | `catalog` 21 · `knowledge_base` 13 | 34 | by dataset |
  | `scalar` 17 · `set` 11 · `not_found` 6 | 34 | by expected type |

  Two interpretation caveats fall out. (i) **6 of 34 (17.6%) are abstention items**, so the
  metric's floor is not 0 — a model that returns nothing for everything scores 17.6%, and the
  headline conflates "generates a correct query" with "correctly abstains". The shape split covers
  this; the report should not let the headline be read without it. (ii) **The `knowledge_base` half
  is now 13 items that can only ever query `Entity.type`/`Entity.name`** (F-3), i.e. a near-
  homogeneous type-lookup task. The pack's scope note should say the KB half tests schema
  adherence on a narrow surface, not KB question-answering breadth.
- One item is now worth **2.94 pp** (was 2.50). Wilson 95% at a plausible 29/34 = 0.853 is
  [0.699, 0.936] — 23.7 pp wide. Report it, never substitute it for the point estimate; the
  established lab convention (`nlq_scoring.wilson_interval`, `_Z_95 = 1.959963984540054`) applies
  unchanged.

---

## 5. Q4 — resolution path

### Recommended: **(a) reclassify, plus three amendments.** Reject (b).

**A1 — correct §3.8.3, citing the upstream ruling rather than re-deciding it.** Replace "the golden
set's 4 `relationship-traversal` items" with the 6 items across **both** shapes, and cite
`falkor-chat/docs/plans/workflow-nl-query-generation-ml.md` §5's exclusion rule as the governing
precedent. This matters beyond accuracy: a reader who sees model-bench decide this independently
will not know the question was already litigated, and the next refresh can regress it again.
Add `nlq-38`/`nlq-39`'s stored `nlq_eval_results.json` reasons (F-2) beside the `nlq-34` evidence
the section already carries. Owner: `architect`.

**A2 — make the answerability test well-defined (F-4). This is the amendment that actually
prevents recurrence.** §3.8.3's stamp must require the reference spec to be a *faithful* answer —
the query a competent author would write for that question, with `matches[0].filters` constraining
to the entity or predicate the question names — not merely any schema-valid spec whose output
satisfies the scorer. Without this the stamp is author-dependent and the bucket boundary is
arbitrary. Suggested wording for `refresh_golden.py`'s contract: *a reference spec must filter on
the question's own subject; a spec with zero filters, or whose filters do not reference the
question's named entity or predicate, is not a valid answerability witness.* Owner: `architect`
(plan text); the constraint is checkable by review of `reference_specs.json`, no code needed.

**A3 — state explicitly that unanswerable ≠ unscored, and label lucky passes.** Per Q2. Owner:
`architect`.

**A4 — correct `-ml` §7.2 and §3's fact table, and keep the latency `Y` at 40.** §7.2's
`nlq-generator` row becomes `n_eff = 34 · floor 17.6 pp · MDD₈₀ 21.9→22.3 pp` with the adequacy
verdict unchanged; §3's `NLQ set composition` line gains the answerable split
(`34 answerable: 17 scalar / 11 set / 6 not_found; 6 structurally unanswerable`). Owner:
`data-scientist` (me) — hand it back and I will revise the `-ml` note in place under the
same-kind/same-role rule.

### Why not (b) — replacement conflicting-facts items

Not because it is impossible. It is *nearly* impossible and, where possible, wrong:

- The corpus models conflicts as edges **by design** (FR-6). No `Entity` property can carry a
  conflicting value (F-3), and the 15-product catalog literal has 15 distinct names with one price
  each — no conflict there either. So no faithful replacement can be drawn from the data as it
  stands.
- The single exception is F-5's `chief technology officer → ['Concept','Other']`. A spec filtering
  `nameNormalized = 'chief technology officer'` and returning `e.type` would genuinely produce a
  2-element set a single-node DSL can reach, and containment would be meaningful on it. **Reject it
  anyway:** it is an extractor type-assignment defect, not two sources disagreeing about a fact, so
  an item built on it measures corpus extraction noise rather than query-generation ability; the
  question ("what type of entity is *chief technology officer*?") has no realistic user behind it;
  and one such row is a sample of one, which cannot carry a shape.
- Authoring new golden items is a **falkor-chat golden-set change**, not a model-bench one — the
  pack copies `nlq_golden_set.jsonl`, and the note that governs it is archived. Forking the set
  inside model-bench breaks the provenance chain that makes the pack auditable, for a shape whose
  0% is already the informative outcome.

**And (b) is solving a problem that does not exist.** `nlq-38` is one of the *best* unanswerable
items in the set: "How **many** employees…" actively invites a `count` aggregate, and the stored
record shows a real model taking exactly that bait — `{"count(e.entityId)": 12}`, a confident,
well-formed, entirely fabricated answer. That is precisely the fabricate-vs-abstain signal the
unanswerable bucket exists to capture. Replacing it would *cost* the pack a discriminator.

### (c), considered and rejected

Keeping the items answerable and widening the plan's tolerance (e.g. treating a 0% conflicting-facts
score as an accepted constant offset) reproduces the unattainable-gate defect the upstream RCA
already diagnosed and corrected, and would leave `layer1ExactMatchRate` with a hard ceiling of
34/40 = 85% that no compliant implementation can ever exceed. A metric whose maximum is not 100%
and whose maximum is not printed is a trap for the first reader who compares this pack's headline
to any other pack's.

---

## 6. Severity-ranked findings

| # | Severity | Finding | Suggested improvement |
|---|---|---|---|
| **B-1** | **blocker** | §3.8.3's "4 unanswerable" contradicts the archived, gated upstream ruling of 6 (F-1), which was itself a correction of this exact defect; shipping it puts a permanent 0% back into an accuracy denominator | A1 — reclassify to 6/34 and cite the upstream §5 exclusion rule |
| **B-2** | **blocker** | The answerability stamp is author-dependent: a zero-filter/broad spec passes `nlq-38`/`nlq-39` under subset containment (F-4), so "answerable" is not a property of the item | A2 — require the reference spec to filter on the question's subject |
| **M-1** | major | `-ml` §7.2 publishes floor 15.0 / MDD 19.1 pp at n=40 for a metric whose denominator is 34; the resolving-power line would print a bound over one denominator beside a rate over another (F-6) | A4 — 17.6 / 22.3 pp at `n_eff = 34`; wire `n_effective` to the answerable count and assert it in `validate` |
| **M-2** | major | "Excluded from the denominator" is ambiguous between "not counted" and "not scored"; the second reading makes `score_pair`'s containment branch dead and drops the fabricate-vs-abstain signal | A3 — unanswerable items are scored with the full scorer, reported per-shape, excluded from the denominator only |
| **m-1** | minor | A `correct` outcome on an `answerable: false` item is reportable and currently unnamed | Label it *lucky pass on a structurally unanswerable item* in the report prose |
| **m-2** | minor | Subset containment on a `set` item admits an unfiltered scan; latent once both real items move to the unanswerable bucket | Record the reversal trigger: if a future refresh makes any `conflicting-facts` item answerable, pair containment with a subject-filter or set-cardinality guard before it goes live |
| **m-3** | minor | The 34-item set is 17.6% abstention items and its KB half (13) can only query `Entity.type`/`name`; the headline reads as general query quality | Scope note in §3.8.3 + require the shape split beside the headline |
| **n-1** | nit | `-ml` §3's `NLQ set composition` fact line reports raw counts only | Add the answerable split; keep the raw counts for the latency `Y` |

## 7. Risks and open questions

- **Assumption behind "excluding costs no power" (Q3):** it holds exactly while every excluded item
  is a genuine structural zero for *both* arms. F-4's loophole is the one live exception; A2 plus
  m-1 contain it. If a future refresh admits an excluded item that some arms can pass, the
  concordance argument no longer applies and the exclusion must be re-justified.
- **`limit ≤ 50` against 62 `Entity` rows** means an unfiltered `Entity` scan truncates, so F-4's
  loophole is reachable via the `type = "Other"` filter (11 rows) but not reliably via a bare scan.
  This is incidental to the recommendation and should not be relied on as a guard — it is an
  artifact of current row counts, not a rule.
- **Open, for `architect`:** does any other model-bench pack's plan text inherit a
  denominator/exclusion claim from an upstream falkor-chat note? B-1 is a *copied-stale-claim*
  failure, and one instance rarely travels alone. Worth one grep before S4 closes.
- **Out of scope here:** whether the v1 DSL should ever gain relationship traversal. The upstream
  note flags that as a separate, future scope decision; nothing in this review bears on it.
