# Ingestion Pipeline & Entity Fusion — Extraction & Matching Method Note

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-050 (M5)

## 1. The question and the decision it serves

`architect`'s plan (`docs/plans/document-ingestion.md`) delegates two method calls, gating stage 3
and stage 4 of the build (plan §0/§4):

- **(a)** Is LLM-based extraction (plan §3.3) the right technique for FR-7a, and if so, what prompt/
  schema/taxonomy makes it buildable?
- **(b)** Is the plan's deterministic exact-match auto-merge default (plan §3.4, resolving OQ-1) —
  no numeric confidence score, "very-high confidence" = normalized-name + identical-type — defensible
  for v1, or does fusion need embedding-based semantic matching to catch non-lexical synonyms
  ("IBM" vs. "International Business Machines")? And does `SAME_AS.confidence` need a second
  gate within the suggested tier?

**Decision this serves:** whether `coder`/`tdd-engineer` build stage 3 (extraction) and stage 4
(fusion) against the plan's defaults as-is, against a firmed-up version of them, or against a
different technique entirely.

**Top-line answer:** agree with the architect's default on both axes, with one addition on (a) and
two refinements on (b). LLM-based extraction is the right call — firmed up below with a closed
entity-type taxonomy, an explicit relationship-representation rule, and a schema-validation
requirement the parser helper does not itself provide. Deterministic exact-match auto-merge is the
right v1 default — not just "acceptable," actively preferred over shipping an unvalidated numeric
threshold on a correctness-critical, silent (`decidedBy='system'`) action. Embedding-based matching
is **not recommended for v1** at any tier, including the suggested tier — deferred to a scoped v2
with its own evaluation design (§5). A **non-ML precision floor** on the suggested tier is
recommended, but framed explicitly as UX/noise control, not a calibrated confidence gate — see §4.3.

## 2. Findings from the real system

**F1 — the parse helper's two branches treat `require_key` unevenly.** Read
`server/falkorchat/llm.py:530-582` directly. `extract_own_line_json_object` has two acceptance
paths: (i) the entire (fence-stripped) reply is one JSON object — `_load_json_object(text)` at
line 561 — and (ii) exactly one object that "owns its lines" amid other prose. **`require_key` is
only checked on path (ii)** (line 578); path (i) returns whatever dict `json.loads` produces,
key or no key. Since the recommended extraction schema (§3.1) is a single top-level JSON object and
the prompt will instruct "reply with only the JSON object," almost every real reply lands on path
(i) — meaning `extract_own_line_json_object(reply, require_key="entities")` will **not** reject a
reply missing the `entities` key the way it looks like it should. This is not a defect in the
helper (it is documented, conservative-by-a-different-axis behavior, proven correct for the guard
judge's use), but a schema-validation gap for a new, structurally richer caller. **Consequence for
§3.2 below:** `extraction.py` must validate the parsed dict's shape itself (both keys present, both
lists, list items carry the right sub-keys) — do not rely on `require_key` to do this.

**F2 — no JSON-mode/structured-output request path exists today.** Grepped `llm.py`/`transport.py`
for `response_format`/`json_schema`: absent. The request payload is `{"model", "messages",
**params}` with no schema-constrained decoding. Extraction therefore rides the same "prompt
discipline + fence-tolerant parse" pattern the guard judge uses (K-027 item 1) — proven to work for
a flat `{decision, rationale}` object on the shipped `qwen/qwen3-4b-2507`, but **not yet proven**
for a nested `{entities: [...], relationships: [...]}` object, a structurally harder generation task
(open-ended list length, cross-references between two lists). Flagged as an unvalidated default,
§6.

**F3 — no entity-type taxonomy exists anywhere in the codebase today.** Grepped `docs/DESIGN.md`
and every `.py` file for `Entity.type`/`entity type`: the only occurrences are the bootstrapped-but-
dormant schema declaration (`docs/DESIGN.md:189`, `(:Entity {entityId, name, type})`) and the
`MENTIONS`-fan-out RAM note (`docs/DESIGN.md:255-256`) — neither constrains `type`'s value space.
This is a fully open design decision, not a convention to reuse (§3.1).

**F4 — the default LLM for a would-be `extraction` kind is unproven for this task shape.**
`config/models.json` defaults every existing kind (`agent`/`step`/`guard`) to
`lmstudio/qwen/qwen3-4b-2507`; absent a different call from `graph-dba`/`architect`, `extraction`
will resolve to the same local 4B model. K-027 item 3 calibrated this model as a **judge** (flat
boolean-plus-rationale output, false-advance/recall gated, `docs/test-reports/guard-judge-
calibration-2026-08-17.md`) — that calibration says nothing about its reliability as a **generator**
of nested structured output. No transfer of that result is valid here (same reasoning the guard-
calibration note itself applied to the stale K-027-item-3 BACKLOG number, `docs/plans/guard-judge-
calibration-ml.md` §3 — a result from one task/pipeline shape is not evidence for a differently-
shaped one on the same model).

**F5 — RediSearch fuzzy full-text is edit-distance matching, not semantic matching.** The plan's
§2.5/§3.4 correctly identifies RediSearch's built-in fuzzy term syntax (`%term%`/`%%term%%`/
`%%%term%%%` = Levenshtein distance 1/2/3) as already proven in this codebase (`Message.text`).
This is domain-general IR fact, not something needing a live probe: edit-distance fuzzy matching
catches **spelling variants** ("Mircosoft" → "Microsoft", "Acme Corp" → "Acme Corp." with a dropped
period) because the surface strings are close in character space. It **cannot** catch **non-lexical
synonyms** — abbreviations ("IBM" / "International Business Machines"), translations, nicknames,
or renamings — because those pairs are far apart in edit distance despite referring to the same
entity. The plan's own example ("IBM" vs. "International Business Machines") is not a corner case
of fuzzy full-text's design, it is squarely outside what the technique can do *by construction*. This
is the real, load-bearing trade-off the architect's plan asked me to judge with the RAM cost
visible (plan §6) — addressed in §4.

**F6 — the RediSearch relevance score is not a calibrated probability.** `SAME_AS.confidence`
(plan §3.4) is proposed to store "the RediSearch score" for suggested-tier candidates. A
TF‑IDF/BM25‑family relevance score is a **ranking signal**, not a probability of "same real-world
entity" — it has no fixed scale across queries, no calibration against ground truth, and mixing it
conceptually with a genuine ML confidence (the kind K-027 item 3 calibrated for the guard judge) in
the same field name risks a future reader over-trusting it. Addressed in §4.3.

## 3. Recommendation (a) — extraction technique, prompt, schema

**Verdict: agree with the plan's LLM-based-extraction default over an NLP-pipeline (spaCy-style
NER) alternative.** Rationale, weighed against the alternative:

- **Relationships are the deciding factor, not entities alone.** FR-7a requires both entities *and*
  relationships. Off-the-shelf NER (spaCy or otherwise) gives entity spans + a small closed type set
  well; it does **not** give open-predicate relation extraction out of the box — that needs either a
  separately trained/fine-tuned relation-extraction model (a genuinely new ML asset this codebase
  has no infrastructure, data, or budget for) or a rule/dependency-parse heuristic layer that
  degrades badly outside clean prose. Given FR-1's explicit non-closed format list (plain text,
  Markdown, Mermaid, CSV, JSON, "not a fixed/closed list") — several of which (CSV, JSON, Mermaid)
  are exactly the inputs a prose-trained NER pipeline handles worst — an LLM's format-agnostic
  text-in/JSON-out contract is the more robust fit, not merely the more convenient one.
- **No existing dependency, and the `ModelGateway` seam is already proven end-to-end** (parse
  robustness via K-027, per-kind config via K-042). Adding spaCy (plus a language-model download,
  plus a bespoke relation-extraction layer) is new infrastructure with no reuse; LLM extraction is
  net-new *prompt/parsing* code over infrastructure that already exists and is already tested.
- **Cost/latency is not a blocking concern** — extraction runs in the async background pipeline
  (plan §2.2), decoupled from the synchronous write path, exactly like embedding today. One call per
  ~1,000-char chunk is a bounded, per-document cost, not a per-request user-facing latency.
- **What I am *not* claiming:** that LLM extraction is more *accurate* than a well-tuned NER
  pipeline on clean English prose — it likely is not, on that narrow slice. The decision is driven by
  relationship extraction and format breadth, both of which favor the LLM path decisively enough
  that the comparison doesn't need to be closer than this.

### 3.1 Entity-type taxonomy: closed, not open-vocabulary

**Recommendation: a small closed enum, given to the model as an explicit list in the prompt, with a
mandatory `Other` catch-all.** Proposed v1 set: `Person, Organization, Location, Product, Event,
Concept, Other` (7 values). Rationale:

- **FR-8's auto-merge tier is gated on `Entity.type` equality** (plan §3.4: "identical `Entity.type`"
  is half of "very-high confidence"). An **open-vocabulary** type field defeats this at the source:
  the same real-world "IBM" mentioned in two chunks could be typed `"Company"` in one extraction and
  `"Organization"` in another purely from LLM label-choice variance — collapsing the auto-merge
  tier's recall even when the *name* string-matches exactly, for reasons that have nothing to do with
  entity identity. A closed enum removes this failure mode by construction: the model picks from a
  fixed list, so two mentions of the same entity type consistently, and the exact-match criterion
  fires as designed.
- Coarse/generic, not domain-specific, is deliberate: FR-1's source-format list is broad and
  unclosed, so a narrower taxonomy (e.g., something CRM- or org-chart-specific) would need
  domain requirements this pipeline doesn't have. `Other` is mandatory in the prompt precisely so the
  model has a safe fallback rather than either refusing to extract or inventing a new label.
- **Rejected alternative — fully open-vocabulary type strings** (the plan's schema sketch
  `{"name","type"}` doesn't specify either way): rejected because it directly undermines the one
  mechanism (§3.4's exact-type-match) the plan relies on for auto-merge, for no offsetting benefit —
  nothing downstream (repository, matching, retrieval) is shown to need finer-grained or
  LLM-invented types.

### 3.2 Output schema, parse validation, and cross-references within one chunk

Keep the plan's schema shape — `{"entities": [{"name","type"}], "relationships":
[{"subject","predicate","object"}]}` — one call per chunk, parsed via
`llm.extract_own_line_json_object(reply, require_key="entities")`. Two firmed-up requirements the
plan left open:

- **App-side schema validation is mandatory, not optional (per F1).** After parsing, `extraction.py`
  must independently verify: `entities` is a list of `{name: str, type: <one of the 7 enum values>}`;
  `relationships` is a list of `{subject: str, predicate: str, object: str}`; reject (treat as "no
  result," same posture as an unparseable reply) a payload missing either key, with `entities`/
  `relationships` not lists, or with a `type` outside the enum (coerce to `Other` rather than reject
  outright — a wrong type is recoverable via fusion review; a missing key is not something to guess
  at). This closes the F1 gap where `require_key` alone will not catch it.
- **Relationships reference entities by literal name, and that reference is not guaranteed to be
  complete — plan for it, don't assume LLM self-consistency.** `subject`/`object` are the extracted
  `name` strings, matched against the same call's `entities` list (no synthetic IDs exist yet at
  extraction time — stage 3 explicitly creates fresh entities from the list, plan §3.3). This is a
  known LLM extraction failure mode: a `subject`/`object` name absent from the `entities` list in
  the same reply is *common*, not exceptional — the model states a relationship fact and forgets to
  also enumerate one side as a standalone entity. **Recommendation:** `extraction.py` does a
  deterministic app-side repair, not a second LLM call — for any `subject`/`object` name not found
  (case-fold + whitespace-collapse compare, matching the plan's own normalization convention, §3.4)
  in `entities`, synthesize a stub entity `{name: <that name>, type: "Other"}` and append it before
  returning `ExtractionResult`. This is cheap, deterministic, and avoids silently dropping a
  relationship fact (the costlier failure — FR-6 is about *not* losing facts) over a model's
  bookkeeping lapse. Name matching is exact-normalized only here, not fuzzy — a genuinely
  under-specified reference (e.g., a pronoun) is out of scope for this pass and simply won't resolve
  to a stub, which is an acceptable, expected miss for v1.
- **Bounded output, empty-result handling:** keep the plan's 20-entities/20-relationships cap (§3.3).
  Add explicitly to the prompt: a chunk mentioning nothing extractable must still return
  `{"entities": [], "relationships": []}`, never omit the JSON object or reply with prose — this
  keeps every chunk's extraction call on the parser's "whole object" path (F1) rather than falling
  into ambiguous-prose territory that resolves to a parse failure.

### 3.3 Prompt design (structural guidance, not final copy — implementer's to word)

- System/instruction message: state the task (extract entities + relationships from the given
  passage), the closed type enum with one-line definitions, the exact JSON shape (including the
  empty-result case), and an explicit "reply with the JSON object and nothing else" instruction —
  mirroring `_JUDGE_SYSTEM_PROMPT`'s existing discipline (`app.py:320-330`) rather than inventing a
  new prompting convention for this codebase.
- Do **not** ask the model to also emit a `confidence`/`certainty` value per entity or relationship —
  small local models' self-reported confidence is well known to be poorly calibrated and there is no
  mechanism in this pipeline that would consume it (fusion's confidence, if any, comes from the
  *matching* stage against the graph, §4, not from extraction's self-assessment). Keep extraction's
  output surface exactly the two required lists.
- One call per chunk, entities + relationships combined (not split into two calls). This matches the
  plan and is the right v1 default — splitting into "extract entities" then "extract relationships
  conditioned on the entity list" is a real accuracy lever for a struggling local model, but it
  doubles background LLM load per chunk (relevant under bulk ingestion, FR-11) for a quality gain
  that is currently **unmeasured**. **Named as the first fallback if/when real extraction quality is
  assessed and found wanting** (§6) — not built preemptively against an unmeasured problem.

## 4. Recommendation (b) — matching technique and the OQ-1 threshold

**Verdict: agree with the plan's deterministic exact-match auto-merge default. Do not add
embedding-based matching in v1, at either tier.**

### 4.1 Why deterministic exact-match is the *right* v1 default, not just an acceptable one

The plan frames this default defensively ("no calibration data... an unvalidated numeric threshold
auto-linking two possibly-different real entities is a correctness risk this default avoids"). I'd
go further: given the actual failure mode at stake, exact-match is the methodologically correct
choice, not a fallback pending better data.

- **Auto-merge (`status='confirmed', decidedBy='system'`) is the one fusion action with no human/
  agent gate at all** (plan §3.4) — the suggested tier, by design, always gets reviewed before
  linking. That asymmetry means the auto-merge criterion carries all the correctness risk in this
  design; the suggested tier's criterion carries only recall/UX risk (a bad suggestion just wastes a
  reviewer's click). A numeric threshold is exactly the kind of decision that needs calibration data
  before being trusted with silent, unreviewed behavior — precisely the standard this lab already
  set for the guard judge (K-027 item 3) and precisely what does not exist here (plan §7, my
  finding F4). Shipping an *uncalibrated* numeric threshold on the *unreviewed* action would repeat
  the mistake K-027 item 3 was created to prevent, not merely accept a known gap.
- **Exact-normalized-name + identical-type is a legitimate high-precision criterion on its own
  terms**, not just "the safe thing to do because nothing else is available." Two mentions with the
  literal same normalized string and the same (now closed-taxonomy, §3.1) type are, in the
  overwhelming majority of real documents, the same entity. This needs no calibration because it is
  not a probabilistic judgment — it is a definitional identity check, same category of decision as
  a uniqueness-constrained id match.
- **Residual known risk, name it rather than let it hide:** same-name-same-type-different-entity
  collisions are real, if rare, for common `Person`/`Organization`/`Product` names (two different
  people named "John Smith," both typed `Person`, in two unrelated ingested documents). Exact-match
  auto-merge will incorrectly fuse them with no review. This is an accepted v1 risk — mitigated in
  practice by: (a) fusion never physically merges nodes (plan §3.4 — an incorrect
  `SAME_AS{status:'confirmed'}` edge is itself a reviewable, correctable record via
  `reject_match`/`recheck_match`, not a destructive migration), and (b) it is the same shape of
  risk any exact-key-based auto-linking system accepts. Not a blocker; worth one sentence in the
  eventual test plan (`qa-engineer`) as a known-gap scenario rather than an unstated assumption.

### 4.2 Why NOT to add embedding-based matching in v1 — cost and validity, both

Agree with the plan's RAM framing (§6): adding `Entity.embedding` doubles the vector-RAM growth axis
beyond the already-dominant `Chunk.embedding` line, at the same empirical per-vector cost the plan
already cites for chunks (`docs/DESIGN.md` §11: ~12.4 KB/vector at 1024 dims, ~85% index/HNSW
overhead) — `Entity` count grows with distinct real-world things mentioned across a corpus, which for
a fusion-oriented pipeline is exactly the count this feature is trying to *collapse*, not a small
fixed set. That RAM line is a real, visible cost of the alternative, cited from the plan's own
measurement rather than newly estimated here.

But the RAM cost is not even the deciding factor — **validity is.** F5 established fuzzy full-text
cannot catch "IBM"/"International Business Machines"; only semantic (embedding) matching can. That
is real and I am not dismissing it. The reason not to build it now is the same reason as §4.1: **any
embedding-based matching needs a numeric similarity threshold to be actionable at all** — even to
decide "is this candidate similar enough to bother surfacing as a suggestion" — and this pipeline has
**zero labeled data** to set one. An uncalibrated embedding threshold set by intuition
(e.g., "cosine ≥ 0.85") is not a defensible number; it is a guess wearing a decimal point,
exactly the anti-pattern this lab's own guard-judge precedent (K-027 item 3) was built to avoid.
Building the infrastructure (index, RAM cost) without a validated threshold to drive it is worse
than not building it — it invites a future author to make up a number under deadline pressure
because "the pipe is already there."

**This is a scoped v2, not an indefinite deferral.** The evaluation design that would justify
climbing this rung is in §5 — sized as the "minimal viable version" the brief asked for.

**Rejected alternative considered and set aside for now — LLM-confirmation matching** (asking the
extraction/fusion LLM itself "are these the same entity?" as a third candidate-generation technique,
between fuzzy-text and embeddings): would catch some of the same non-lexical-synonym cases as
embeddings without a new vector index, but has its own uncalibrated-threshold problem in a different
shape (an LLM's own binary "same/different" judgment needs exactly the same false-merge-rate/recall
calibration as embeddings would, per §5's metric design) and adds a synchronous or async LLM call
per candidate pair — a real throughput cost under bulk ingestion (FR-11) with no more validity than
embeddings for the same missing-data reason. Not recommended over embeddings if/when this rung is
climbed; noted only so it isn't independently proposed later without the same scrutiny.

### 4.3 `SAME_AS.confidence` — a precision floor, not a second ML gate

**Recommendation: filter the suggested tier's weakest candidates, but as a non-ML precision/UX
knob, not a calibrated confidence threshold.** Per F6, the RediSearch relevance score is not a
probability and gating on it as if it were "confidence" would misuse the number the same way an
uncalibrated symmetric metric misuses κ for a biased judge — treating a score as meaning something
it was never validated to mean.

- **Do** cut obviously-noisy hits before they reach the pending queue — an unfiltered fuzzy-text
  candidate generator will surface weak matches (e.g., a single short/common token overlap) that
  waste reviewer attention and, over time, train reviewers to stop reading the queue carefully (the
  standard alert-fatigue failure mode for any unfiltered suggestion/alert surface). This is a
  legitimate v1 concern and doesn't need a golden set to justify — it's a usability floor, not a
  correctness gate: a suggestion that gets rejected costs a click; a suggestion that never surfaces
  because it was noise costs nothing (FR-9/AC-3 require plausible matches to surface, not that every
  weak candidate does).
- **Implement it as an implementer-tunable heuristic** (e.g., a minimum RediSearch score, or a
  minimum matched-token-length rule), explicitly documented at the point of use as "noise-floor
  cutoff, not a validated match-confidence threshold" — so a future reader does not mistake it for
  the calibrated boundary FR-8's auto-merge tier needs and doesn't have. Exact value is a routine
  implementer judgment call (mirrors the plan's own "implementer-tunable, not load-bearing" framing
  for `MAX_DOCUMENT_CHARS`/`MAX_BATCH_SIZE`, plan §3.5) — not something I'm gating here.
- **Store the raw score in `SAME_AS.confidence` as the plan proposes**, for audit/UI — but
  the field's docstring/API documentation should say what it is (a retrieval relevance score) and
  what it is not (a calibrated probability of correctness), so it isn't silently promoted to a gate
  later without the calibration work §5 describes.

## 5. Evaluation design — for the v2 embedding-matching rung, if/when climbed

Not required for v1 (§4.2) — recorded now so the decision to climb this rung, when made, has a ready
protocol rather than an ad hoc one, mirroring how the guard judge's calibration was designed before
being built (`docs/plans/guard-judge-calibration-ml.md`).

- **Golden set:** entity-name(+type) **pairs** labeled `{same-entity, different-entity}`, stratified
  into: (1) exact/near-exact duplicates (trivial positives, sanity check); (2) non-lexical synonyms —
  abbreviations/acronyms, translated names, common nicknames (the "IBM" case — the real target this
  rung exists for); (3) hard negatives — same or very-similar surface string, different real entity
  (e.g., two distinct people sharing a common name, a company name that is also a place name). Class
  (3) is the one a pure similarity-threshold approach is most likely to get wrong, and is the one
  most important not to under-sample. **Minimal viable v1 size:** none is required to ship v1 as
  scoped (§4.2) — the minimal version *of the golden set itself*, when this rung is picked up, is on
  the order of the guard judge's own precedent (n≈20-30 hand-labeled pairs, `docs/archive/plans/m3-
  guard-calibration.md`'s scale) — enough for a one-sided screen, not a certification, with the same
  Wilson-interval honesty about what that sample size can and cannot detect at small n. Larger,
  ideally sourced from real post-ingestion `SAME_AS`-edge review outcomes once the exact/fuzzy
  system has been running long enough to accumulate labeled confirm/reject decisions — genuine
  production data is more representative than a hand-built set and costs nothing extra to collect
  once the suggested tier is live.
- **Metric — asymmetric, gated on the costly error, mirroring the guard judge's FAR/recall split
  (K-027 item 3), not symmetric accuracy/κ:** the costly error here is a **false merge** (declaring
  two different real-world entities the same, at the auto-merge tier), analogous to the guard's
  false-advance; a missed match (two same entities left unlinked, but still separately reachable
  since nothing is ever destructively merged, plan §3.4) is the safe-direction error, analogous to
  the guard's bias-to-suspend, and is what the existing suggested-tier/manual-recheck path already
  provides a safety net for. Report **false-merge rate** (of pairs the technique would auto-link,
  the fraction that are actually different entities) and **match-recall** (of true same-entity
  pairs, the fraction the technique finds at all) as the two numbers that matter; report any
  symmetric accuracy/κ figure as a diagnostic only, never the gate, for the same reason the guard
  judge's own calibration demoted κ (`docs/BACKLOG.md` K-027 item 3: "an always-suspend judge scores
  a perfect 0% FAR, so the original κ-based gate could be passed by a judge that never advances" —
  the same collapse-onto-the-safe-class risk applies to an always-suggest, never-auto-merge matcher).
- **No threshold number is proposed here** — that would be exactly the "a guess wearing a decimal
  point" anti-pattern §4.2 argues against. The threshold is whatever value on the labeled set
  achieves an acceptable false-merge rate (the primary gate) at the best achievable recall (the
  secondary gate) — set **after** the golden set exists, not before, following the K-027 item 3
  precedent's own sequencing (protocol authored, then run, then the number reported) rather than
  reasoning backward from a target number.
- **Where semantic matching would slot in, technique-wise, if this rung is climbed:** a new
  `Entity.embedding` vector index (per plan §6's RAM framing) used for **candidate generation only**
  at the suggested tier — never auto-merge directly off a cosine score, even a calibrated one, unless
  a second, harder-still calibration pass (with a materially lower false-merge tolerance than the
  suggested tier needs, since auto-merge has no human check) is separately run and gated. This
  two-step framing (embeddings widen candidate recall; a human/agent still confirms) keeps the
  auto-merge tier's zero-uncalibrated-numbers invariant intact even after this rung is climbed.

## 6. Risks & open questions

- **Extraction quality on the nested schema is genuinely unmeasured (F2/F4).** Unlike the guard
  judge, which had a golden set before being trusted with silent behavior, extraction has none, and
  fusion's exact-match tier only ever sees what extraction actually produces — a local model that
  under-extracts (misses real entities) or over-extracts (hallucinates spurious ones) degrades this
  feature's usefulness in ways the exact-match/fuzzy-text fusion design cannot detect or correct,
  because fusion has no notion of "this entity looks fabricated." **I am not recommending a
  pre-launch calibration pass be made a blocking precondition** — the plan's own stance (§7: "ship a
  defensible deterministic default, iterate with real data") is reasonable given this is a net-new
  capability with no existing production traffic to calibrate against yet, and extraction is a
  generation task, not a classification/screening one — there is no equivalent to the guard's
  false-advance/recall gate to run *before* any real output exists to look at. **What I do
  recommend as a firm follow-up, not optional:** once stage 3 is live end-to-end (even pre-fusion,
  per the plan's own "fusion permanently at always-create-new is a valid degenerate case," §3.3),
  `data-scientist` reviews a sample of real extraction output (e.g., 20-30 chunks across a few real
  documents) for the two visible failure modes above before stage 4 (fusion) is trusted with the
  entity population extraction produced — this is a qualitative read, not a full eval, and should be
  quick. Flagging to `teco`'s coordination as a stage-3→stage-4 gate item, not authoring the pass
  here since no real ingested content exists yet to sample.
- **The one-call-combined-entities+relationships design (§3.3) is the right v1 default but is a
  named fallback point, not a closed decision** — if the qualitative review above finds the local 4B
  struggling specifically with cross-referencing (relationships pointing at the wrong/missing
  entities, ignoring §3.2's stub-repair), the fallback is a two-stage call (entities, then
  relationships conditioned on them), at 2x background LLM cost per chunk.
- **Closed entity-type taxonomy (§3.1) is my call to make and I've made it** — 7 values
  (`Person, Organization, Location, Product, Event, Concept, Other`) — but it is a genuinely new
  piece of vocabulary this codebase has never had an opinion on before. If the stakeholder or a
  downstream consumer (e.g., a future retrieval feature that wants to filter by entity type) has
  requirements this list doesn't anticipate, it's a cheap, additive change (append a value) — not
  structurally locked the way, e.g., `RELATES_TO` predicate-as-property is.
- **Same-name-same-type-different-entity auto-merge collisions (§4.1) are an accepted, named risk,
  not a blocker** — worth one explicit line in `qa-engineer`'s eventual test plan as a known-gap
  scenario, so a future reviewer doesn't mistake an observed collision for a regression.
- **The `extraction` `ModelGateway` kind's default model choice is graph-dba/architect's call
  structurally (plan §2.3/§7)**, but methodologically I'd flag: reusing `lmstudio/qwen/qwen3-4b-2507`
  (today's default for every other kind) is a reasonable starting point precisely because it's
  already wired and config-proven, not because it's proven *for this task* (F4) — the §6 qualitative
  review above is what would surface a mismatch, not a priori reasoning.
