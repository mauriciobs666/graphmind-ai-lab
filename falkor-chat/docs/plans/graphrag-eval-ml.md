# GraphRAG retrieval + generation evaluation harness — method note (data-scientist)

> **Status:** proposed method note (M2.5-quality track). Advisory deliverable — implementation
> routes to `coder`/`tdd-engineer`; any in-graph query change routes to `graph-dba`.
> **Author:** data-scientist · **Date:** 2026-07-10
> **Slots into:** kaizen plan as a tracked **M2.5-quality** item (draft at the bottom of this doc),
> parallel to the M2.5 hardening track (K-016/K-017/K-018), sequenced behind the M3 critical path.

## The question & the decision at stake

**Question:** How do we know whether falkor-chat's GraphRAG retrieval is *good* — and how will we
know whether a future change (Entity-extraction expansion, hybrid fusion, a seed-relevance
threshold, an embedding-model swap) makes it **better or worse**?

**Decision it serves:** whether to build a retrieval/generation evaluation baseline *now*, before
the parked GraphRAG-quality work (Entity extraction, hybrid fusion) resumes. Today every such
change would ship on vibes. This note defines the harness that makes them measurable, so the
qa-engineer and implementers inherit something testable.

**What is explicitly NOT in question:** the M2 scope cuts. "M2 green = functional GraphRAG"
(embeddings + vector index + hybrid retrieval + agent + `EMITTED`) was a deliberate, user-signed-off
DoD (kaizen 2026-07-05); the dormant `Entity` expansion and vector-only retrieval are *intentional
and tracked*, not defects. This note does not reopen them — it builds the instrument that will
judge them **if/when** they're un-parked.

## Findings from the real system

Read: `server/falkorchat/responder.py`, `embedding.py`, `llm.py`, `repository.hybrid_search`,
`config.py`, `docs/plans/m2-graphrag.md`, `docs/test-reports/m2-graphrag-report.md`, `kaizen/plan.md`.

1. **Retrieval today is pure single-vector kNN.** `repository.hybrid_search` seeds via
   `db.idx.vector.queryNodes('Message','embedding',$k,…)`, cosine distance **ASC**, then a
   thread-scope `MATCH`. The `MENTIONS→Entity` co-occurrence expansion is an `OPTIONAL MATCH` that
   no-ops (no extraction pipeline — parked). Full-text (`/search`) exists but is **not fused** into
   retrieval. Model: `Qwen3-Embedding-0.6B` @ **1024-dim**; LLM `qwen/qwen3-4b-2507` via LM Studio.
2. **No relevance thresholding.** `AgentResponder` requests `k=10` seeds and feeds *all* of them raw
   into the prompt (`_build_prompt`), regardless of score. The report's own numbers show a cooking
   query pulling an orthogonal seed at cosine distance **0.786** — i.e. noise fed as "context" to a
   4B model that is especially distractor-sensitive. Score is stored in `EMITTED` provenance but
   **never used to filter**.
3. **The system prompt is grounding-permissive.** `_SYSTEM_PROMPT`: *"If the context does not help,
   answer from general knowledge."* No citation forcing, no abstention path — undercuts the
   retrieval-grounding value prop and there's no faithfulness metric to catch ungrounded answers.
4. **No evaluation exists, and none is in the backlog.** The M2 QA pass (K-015) verified the *loop
   works* — plumbing + ranking on a toy 3-topic corpus (cooking/space/biology, maximally separated).
   The report itself flags: *"retrieval quality is model-dependent… real-world overlap not
   exercised."* No golden set, no recall@k/MRR/nDCG, no faithfulness/groundedness. The kaizen
   parking lot tracks the Entity pipeline; **nothing tracks measuring retrieval quality.** This is
   the genuine gap.
5. **Self-retrieval inflation risk (methodological trap for the eval itself).** The responder
   self-embeds its trigger and answer into the corpus; the report noted the trigger surfacing as its
   own "rank-0 provenance self-hit." Any golden query that is a *verbatim* copy of its target message
   retrieves itself at distance 0 → trivially inflated recall. The golden set must use **paraphrased**
   queries (see below).

## Recommended method

Build a **two-layer offline eval harness** — retrieval and generation measured **separately**
(diagnose before blaming either) — anchored on a small, hand-maintained golden set. Establish a
baseline against today's vector-only retrieval; every future change is a measured delta against it.

### Layer 1 — Retrieval eval (primary; cheap, deterministic)

- **Golden set:** 30–50 `query → relevant_msgId(s)` pairs drawn from a **representative** corpus
  (see Risks — `seed_demo.sh` is too thin; needs a richer seeded `ws:eval` or real usage). For each,
  a natural user-phrased query whose answer lives in one (or a few) known message(s).
  - **Queries must be paraphrases, never verbatim** of the target message text (finding 5) — else
    recall is trivially inflated by self-retrieval.
  - Build assisted (LLM drafts candidate query + labels the answering message), then **human-verify
    every pair** — the human check is the validity anchor, not optional.
  - Versioned fixture (`server/tests/eval/golden_retrieval.jsonl`), **never seeded into a live
    prompt/few-shot** (leakage guard).
- **Metrics** over `hybrid_search` output (convert cosine distance → similarity `1 - d` only for
  reporting; rank on distance ASC as the code does):
  - **recall@k (k = 5, 10)** — *primary*: did the relevant message land in the seeds actually fed to
    the LLM? This is the number that bounds answer quality.
  - **MRR** — *secondary*: rank quality of the first relevant hit.
  - **nDCG@10** — *optional*, only if a query has graded (multi-level) relevance; skip for binary.
- **Baseline run establishes the numbers; it does not gate.** First execution records
  `recall@10 / recall@5 / MRR` for vector-only @1024 as the frozen baseline.

### Layer 2 — Generation / faithfulness eval (secondary; LLM-as-judge, calibrated)

- **~15–20 Q&A** over the golden set (subset is fine). Judge two axes:
  - **Faithfulness / groundedness** — is the answer supported by the retrieved seeds (not fabricated)?
    Because the prompt *permits* general-knowledge answers, score **grounded-when-context-was-relevant**
    separately from **abstained/general** — don't penalize a correct "context didn't help."
  - **Answer relevance** — does it actually address the question?
- **LLM-as-judge with its caveats made explicit:** use the **strongest available local model as
  judge** (never the 4B-under-test judging itself — self-preference bias). **Calibrate the judge
  against ~10 human-labeled examples before trusting it**; report judge-vs-human agreement. If
  agreement is poor, fall back to human spot-check for this layer — a small honest human pass beats a
  miscalibrated automated score.

### Rejected alternatives

- **Keep hand-eyeballing a toy corpus (status quo).** Rejected: proves plumbing, measures nothing;
  can't detect regression or bless a change. This is the gap we're closing.
- **Full RAGAS-style automated suite (context-precision/recall + faithfulness, all LLM-judged) up
  front.** Rejected *for now*: heavier, and every judged metric needs local-judge calibration to be
  trustworthy. Start with deterministic retrieval metrics (no judge needed) + a thin calibrated
  faithfulness check; graduate to the fuller suite only if the cheap layer proves insufficient.
- **Online A/B on real traffic.** Rejected as the *first* instrument: traffic is too thin
  pre-production ("long road before production," kaizen) — underpowered, can't answer the question
  yet. Offline eval is the right rung now; revisit A/B at real volume.

## Evaluation design (metric · data · threshold)

| Layer | Metric | Data | Acceptance threshold |
|---|---|---|---|
| Retrieval | **recall@10** (primary), recall@5, MRR | 30–50 paraphrased golden pairs, human-verified | **First run = baseline (record, don't gate).** Thereafter a retrieval change is *accepted* only if **recall@10 ≥ baseline** and **MRR not down > 5% relative**; *promising* if either improves without the other regressing. |
| Retrieval (threshold tuning) | recall@10 vs seeds-fed count | same golden set | A seed-distance cutoff is accepted only if it **holds recall@10** while cutting median seeds-fed (fewer distractors) — proves the cutoff drops noise, not signal. |
| Generation | faithfulness (grounded-when-relevant), answer-relevance | 15–20 judged Q&A | Report as baseline; **judge–human agreement ≥ ~0.7** required before the judged numbers are trusted at all. |

**Data prerequisite (flag):** the golden set is only as good as the corpus behind it. `seed_demo.sh`
is a thin demo. Recommend a dedicated **`ws:eval`** seeded with a representative multi-topic chat
corpus (or harvest from real `ws:acme` usage once it exists) — the implementer should treat "build a
representative corpus" as step 0, not assume the demo seed suffices.

**Harness placement (for the implementer):** offline, pytest-based behind a **live marker** (mirror
the existing live-LM-Studio marker convention) so the network-free baseline stays green; cache
embeddings where possible for determinism. Emit a small metrics report
(`docs/test-reports/graphrag-eval-<date>.md` or a JSON the QA pass reads). In-graph query changes
(e.g. wiring Entity expansion, fusion) go through the `graph-dba` gate; the eval harness itself is
app-layer `coder`/`tdd-engineer` work.

## Risks & open questions

1. **Corpus representativeness (highest).** A golden set over a toy/demo corpus measures a toy
   problem. Needs a realistic `ws:eval` — see data prerequisite. **Open:** harvest real `ws:acme`
   traffic vs. author a synthetic-but-realistic seed? (Recommend synthetic-realistic first; it's
   reproducible and CI-friendly; swap to real traffic when volume exists.)
2. **Self-retrieval inflation.** Enforced by the *paraphrase* rule; add a harness assertion that no
   golden query equals its target message verbatim.
3. **Local-judge validity.** The strongest local model may still be a weak judge. Mitigation:
   calibrate-before-trust + human fallback. Don't report faithfulness numbers uncalibrated.
4. **Golden-set maintenance drift.** Keep it 30–50 pairs — small enough to re-verify by hand when the
   corpus or embedding model changes. A stale golden set is worse than none.
5. **Leakage.** Golden queries/labels must never enter a prompt, few-shot, or the seeded live corpus
   as answer keys. Keep the fixture test-only.

---

## Ready-to-paste kaizen item (M2.5-quality track)

> Paste into `kaizen/plan.md`. Suggested placement: a new **"M2.5-quality track"** heading parallel
> to the deferred M2.5 hardening track, or under Active if picked up soon. Renumber `K-0xx` to the
> next free id.

```markdown
### — M2.5-quality track (retrieval evaluation; parallel to M2.5 hardening) —

### K-0xx — GraphRAG retrieval + generation evaluation harness (🔵 proposed — M2.5-quality, off the M3 critical path)

- **Owner:** **`data-scientist`** method note ✅ (`docs/plans/graphrag-eval-ml.md`) → **`coder`/`tdd-engineer`**
  (harness + golden-set fixture) → **`graph-dba`** only if a retrieval query change is later measured through it.
- **Inputs/prereqs:** M2 GraphRAG ✅ (K-008/K-013). A representative corpus — build a seeded **`ws:eval`**
  (step 0; `seed_demo.sh` is too thin). Local LM Studio for the (optional) judged generation layer.
- **Scope:** (1) 30–50 **paraphrased**, human-verified `query→relevant_msgId` golden pairs
  (`server/tests/eval/golden_retrieval.jsonl`); (2) retrieval eval over `hybrid_search` — **recall@10** (primary),
  recall@5, MRR — **establishing the vector-only @1024 baseline**; (3) thin **LLM-as-judge** faithfulness +
  answer-relevance layer over ~15–20 Q&A, **calibrated against ~10 human labels before its numbers are trusted**;
  (4) a metrics report the K-025-style QA pass can read. Behind a live marker; network-free baseline stays green.
- **Done-condition:** baseline recall@10/recall@5/MRR recorded; harness re-runnable; judge–human agreement reported;
  golden set asserts no verbatim self-retrieval; both suites green.
- **Why now:** it's the **prerequisite baseline** for un-parking Entity extraction, hybrid fusion, a seed-relevance
  threshold, or any embedding-model swap — today those would ship unmeasured. Also unblocks two cheap tracked
  quality fixes: a **seed-distance cutoff** (drop distractor seeds, finding 2) and resolving the
  **grounding-permissive system prompt** (finding 3) — each measurable against this baseline.
- **Risks/RAM:** transient `ws:eval` @1024 vector index (budget per K-008's ~12.5 KB/msg line); no production RAM.
  Corpus representativeness + local-judge validity are the methodology risks (see the method note).
- **Test strategy:** deterministic retrieval metrics (no judge) as the core; calibrated judged layer as an overlay;
  golden-set fixture versioned and test-only (leakage guard).
```
