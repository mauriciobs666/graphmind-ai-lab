# Small-Model Catalog Sweep — Feature Requirements
> **Status:** archived · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-09-19

## Intent

The stakeholder wants a comprehensive, apples-to-apples comparison across every small
(low-footprint) model currently available in the local LM Studio catalog, for every job
`model-bench` knows how to evaluate — so that picking a model for a given role (embedding,
guard-judge, NLQ generation, tool-calling, grounded chat) can be based on real, side-by-side
evidence instead of the handful of one-off runs done so far.

## Problem & current state

Today only 5 of the 68 possible (model × applicable-pack) combinations in the in-scope list have
a stored result at all, and they were produced incidentally across separate, earlier sessions —
not as one coordinated, directly-comparable batch:

| Pack | Models already tested |
|---|---|
| `embedder-graphrag-retrieval` | `text-embedding-qwen3-embedding-0.6b` |
| `guard-judge-understanding` | `google/gemma-3-4b`, `qwen2.5-3b-instruct`, `qwen/qwen3-4b-2507` |
| `nlq-structured-query` | `qwen2.5-3b-instruct`, `qwen/qwen3-4b-2507` |
| `tool-caller-shop-assistant` | `mistralai/ministral-3-3b`, `qwen/qwen3-4b-2507` |
| `chat-responder-grounded-answers` | `qwen/qwen3-4b-2507` |

No pack today has more than 3 of the in-scope models compared against each other, and none of
the existing results share a common session tag — so there is no comprehensive, trustworthy
comparison across small models for any of the five job roles yet.

## Scope: the in-scope model list

Locked in during this conversation. "Small" is judged by estimated on-disk/memory footprint at
the model's currently-configured quantization (not raw parameter count) — see the Decision log.

**Embedding models** (target pack: `embedder-graphrag-retrieval` only):
- `text-embedding-qwen3-embedding-0.6b`
- `text-embedding-nomic-embed-text-v1.5`
- `text-embedding-granite-embedding-278m-multilingual`
- `text-embedding-qwen3-embedding-4b`

**Chat/VLM models** (target packs: all four chat-role packs — `guard-judge-understanding`,
`nlq-structured-query`, `tool-caller-shop-assistant`, `chat-responder-grounded-answers`):
- `qwen/qwen3-4b-2507`
- `qwen/qwen3-4b-thinking-2507`
- `smollm3-3b`
- `stablelm-zephyr-3b`
- `qwen2.5-3b-instruct`
- `qwen2.5-coder-3b-instruct`
- `llama-3.2-3b-instruct`
- `stable-code-instruct-3b`
- `mistralai_ministral-3-3b-instruct-2512`
- `mistralai/ministral-3-3b`
- `gemma-3-4b-vl-it-gemini-pro-heretic-uncensored-thinking_gguf`
- `google/gemma-3-4b`
- `google/gemma-4-e2b`
- `qwen3.5-2b-claude-4.6-opus-reasoning-distilled`
- `nvidia/nemotron-3-nano-4b`
- `prism-ml/bonsai-27b` (27B parameters, included on estimated footprint — see Decision log)

20 models total (4 embedding + 16 chat/vlm).

## User stories

- As the stakeholder choosing a small local model for a given role, I want every in-scope
  candidate model already benchmarked against the pack(s) for that role, so that I can compare
  real evidence across candidates instead of running one-off benchmarks myself first.
- As the stakeholder reviewing the sweep afterward, I want one comparison report per pack
  covering every in-scope model that was actually run against it, so that I don't have to
  assemble the comparison by hand from individual stored runs.

## Functional requirements

- **FR-1.** Each of the 4 embedding models is run against `embedder-graphrag-retrieval`.
- **FR-2.** Each of the 16 chat/vlm models is run against all four chat-role packs:
  `guard-judge-understanding`, `nlq-structured-query`, `tool-caller-shop-assistant`,
  `chat-responder-grounded-answers`.
- **FR-3.** Every run produced by this sweep is tagged with one shared session identifier, so the
  whole batch can be retrieved and compared together after the fact.
- **FR-4.** The sweep re-runs every in-scope (model, pack) combination fresh under the shared
  session identifier — including the 5 combinations that already have a stored result from
  earlier, unrelated work — rather than reusing that existing data.
- **FR-5.** A single (model, pack) run failing for an operational reason (e.g. LM Studio
  unreachable, a type mismatch, a refused tool-caller dispatch) does not stop the rest of the
  sweep — every other in-scope combination is still attempted, and the failure is visible in the
  sweep's own record rather than silently dropped.
- **FR-6.** After the sweep completes, one comparison report is produced per pack (five reports
  total), each covering every in-scope model that has a stored result for that pack under the
  sweep's session identifier.
- **FR-7.** Each per-pack report ranks every in-scope model by that pack's headline metric (or
  each `verdictMetrics` member, for a pack with no headline metric), with each model's own
  confidence interval and the pack's resolving-power sentence recomputed for the model count
  actually in that report. It does **not** contain a full pairwise comparison matrix (120 cells
  for 16 models is both unreadable and statistically unsupportable — see the Decision log).
- **FR-8.** If pairwise significance verdicts are included, they are limited to one
  pre-registered, reference-anchored family per pack: each of the other in-scope chat/vlm models
  compared against one designated reference model (recommended: `qwen/qwen3-4b-2507`, already
  tested against all four chat-role packs and already supported by `run --reference`), corrected
  for multiple comparisons (Holm–Bonferroni) across that family. No other pair receives a
  significance verdict.
- **FR-9.** The five per-pack reports are collected into one consolidated document. A thin
  index/cover section may list or link each pack's top-ranked model(s) side by side, but performs
  no arithmetic across packs — no cell, column, or sentence in the consolidated document combines
  a score from more than one pack.
- **FR-10.** The consolidated document includes a narrative "insights and recommendations"
  section: a recommended model (or short list) per role with rationale, latency/footprint
  trade-offs, and caveats — grounded only in that role's own within-pack results, never a
  cross-role composite.
- **FR-11.** Each per-pack report restates — not merely cites — that pack's own known
  ceiling/adequacy caveat from `docs/plans/small-model-benchmarking-ml.md` (e.g. the embedder
  pack's recall@10 ceiling of 37/38, which can detect a worse model but never certify a better
  one; the guard-judge pack's two class-conditional verdicts with no single headline metric; the
  nlq-generator pack's true n=34 denominator).
- **FR-12.** Each per-pack report's ranked table includes latency (p95) and estimated footprint
  (on-disk/quantized size, from this sweep's own model list — not something `model-bench` itself
  measures) alongside each model's score.

## Out of scope

- A literal cross-role composite or aggregate score (e.g. summing or averaging a `tool-caller`
  result with an `embedder` result into one number). Confirmed by a `data-scientist` consult to
  contradict a structural design decision of the tool, not merely a formatting preference — see
  the Decision log. The consolidated document (FR-9) satisfies "one comprehensive report" without
  this.
- A full pairwise comparison matrix across all in-scope models within a pack (superseded by the
  ranked-table + reference-anchored-family design in FR-7/FR-8).

- Any model not on the 20-model list above (including the two 12B-parameter catalog models
  already discussed and excluded, `google/gemma-4-12b-qat` and `google/gemma-3-12b`, and the two
  unofficial 9B community finetunes discussed and excluded, `qwen3.5-9b-uncensored-hauhaucs-
  aggressive` and `qwen3.5-9b-claude-4.6-opus-uncensored-distilled` — see the Decision log).
- Declaring a "winning" model for any role, or any pass/fail threshold — `model-bench` has no
  gate by design, and this sweep does not add one. Interpreting the resulting reports is a
  separate, later activity.
- Turning this into a recurring or CI-triggered job — this is a one-time, human-triggered sweep.
- Any change to the packs themselves (scorers, golden data, prompts) — the sweep only exercises
  the five packs as they exist today.
- Host attestation (`./run.sh attest`) is assumed to already be current going into the sweep; the
  sweep does not add a new attestation capability.

## Acceptance criteria

- **Given** the 20-model list and the five packs, **when** the sweep completes without
  operational failures, **then** there are 4 stored runs for `embedder-graphrag-retrieval` (one
  per embedding model) and 16 × 4 = 64 stored runs across the four chat-role packs, all tagged
  with the sweep's shared session identifier — 68 stored runs in total.
- **Given** a (model, pack) combination fails for an operational reason, **when** the sweep
  finishes, **then** that failure is visible in the sweep's own summary/record (not silently
  absent from it) and every other combination was still attempted.
- **Given** the sweep is complete, **when**
  `./run.sh rank --pack <pack-id> --session <sweep-session-id>` is run for each of the five packs,
  **then** each renders a comparison report to `reports/` covering every in-scope model with a
  stored result for that pack.
- **Given** a chat-role pack's report, **when** it is rendered, **then** every number outside the
  reference-anchored family (if used) is labelled `exploratory — no significance claim`, and the
  pack's existing ceiling/adequacy caveat is restated in that report, not merely cited (FR-11).
- **Given** the five per-pack reports, **when** read together as the consolidated document,
  **then** no cell, column, or sentence combines a score from more than one pack (FR-9).
- **Given** the consolidated document, **when** read, **then** its preamble states plainly what
  this sweep's sample sizes can and cannot prove — capable of catching a large collapse, not
  fine-grained ranking among closely-matched 3-4B models — so this isn't discovered as a surprise
  in the output.

## Open questions

*(none outstanding — see Decision log)*

## Decision log

- 2026-09-18 — "Is 'small' judged by raw parameter count or by footprint?" → Footprint at the
  model's currently-configured quantization, not raw parameter count. This is why
  `prism-ml/bonsai-27b` (27B parameters, quantized to a ~3.5–5.5 GB footprint) is included, and
  why the two 12B models (~6.75–7.2 GB even quantized) are excluded despite having far fewer
  parameters than bonsai.
- 2026-09-18 — "Should already-stored combos be re-run, or reused and only gaps filled?" →
  Re-run everything fresh under one shared session identifier (FR-4), so the whole batch is
  directly comparable and shares consistent timing/fingerprint context.
- 2026-09-18 — "Is the 19-model list locked in as discussed, or does it need adjusting first?" →
  Locked in as-is (see Scope section).
- 2026-09-18 — Execution ownership: the stakeholder will have `teco` coordinate running this
  sweep (dispatching whoever executes `./run.sh run`/`./run.sh compare`) once this document is
  ready for design. This document defines what the sweep must accomplish, not how it is run.
- 2026-09-18 — "The final output should be a detailed and comprehensive report comparing all
  models" → stakeholder wants both a single consolidated document (all five per-pack comparisons
  together) and a cross-role combined view, plus narrative insights/recommendations. Consulted
  `data-scientist` (review-shaped, on the emerging FR/AC language) given the tool's documented
  never-combine-scores-across-roles design and the scale (up to C(17,2)=136 pairs per chat pack).
  Findings, folded into FR-7 through FR-12 and the matching acceptance criteria above: (1) a full
  pairwise matrix is both unreadable and statistically unsupportable at this scale — the right
  shape is a ranked table per pack with per-model confidence intervals, not pairwise cells; (2) if
  pairwise verdicts are wanted at all, they must be a pre-registered, reference-anchored family
  (≤16 comparisons per pack against one designated reference model, Holm–Bonferroni corrected),
  never all 136 naive pairs, which would produce an expected ~6.8 false "better" verdicts under
  the global null; (3) per-role separation is a **structural invariant** of the tool, confirmed in
  code (`load_history()` takes a single `packId`), not a formatting preference — a literal
  cross-role composite score is therefore out of scope (see Out of scope), but one consolidated
  *document* holding five clearly separated, non-arithmetic-linked sections is a legitimate and
  sufficient answer to "comprehensive report comparing all models"; (4) each per-pack report must
  restate its own known ceiling/adequacy caveat and include latency/footprint alongside score, and
  the consolidated document's preamble must set honest expectations about what this sample size
  can and cannot prove. This resolves the stakeholder's "both" answer: the two readings converge
  on the consolidated-document design, without a cross-role number.
- 2026-09-19 — AC-3's cited command corrected from `compare` to `rank`, per architect's plan +
  analyst's review finding that `compare` cannot support this report shape.
- 2026-09-19 — Stakeholder asked to add a third embedding model to the locked-in scope list:
  `granite-278m-multilingual` (IBM Granite Embedding, 278M params), confirmed by `teco` as
  downloaded and present in the local LM Studio catalog (`GET /api/v0/models`: publisher
  `lmstudio-community`, arch `bert`, quantization `Q8_0`, state `not-loaded`) under the
  LM-Studio-local catalog id `text-embedding-granite-embedding-278m-multilingual`. At 278M
  params, Q8_0-quantized, it is trivially within the footprint threshold already established
  above (smaller than the already-in-scope `text-embedding-qwen3-embedding-0.6b`) — no
  re-litigation of that criterion needed. Scope, FR-1, and the acceptance criteria updated from
  19/2-embedding/70-total to 20/3-embedding/71-total accordingly.
- 2026-09-19 — Stakeholder asked to add a fourth embedding model to the locked-in scope list:
  `text-embedding-qwen3-embedding-4b` (Qwen), confirmed by `teco` as present in the local LM
  Studio catalog (`GET /api/v0/models`: publisher `Qwen`, arch `qwen3`, compatibility type
  `gguf`, quantization `Q4_K_M`, state `loaded` at time of check) under the LM-Studio-local
  catalog id `text-embedding-qwen3-embedding-4b`. Applying the footprint-at-configured-
  quantization criterion already locked in above: at 4B params, Q4_K_M (~4.5 bits/weight
  effective), the estimated quantized footprint is roughly ~2.2–2.5 GB — well under the ~6 GB-ish
  line implied by the bonsai-vs-12B-models precedent, and in the same size class as several
  already-in-scope chat/vlm models at 4B params (`qwen/qwen3-4b-2507`,
  `qwen/qwen3-4b-thinking-2507`, `nvidia/nemotron-3-nano-4b`). Passes the existing footprint
  threshold cleanly — no re-litigation of the criterion itself needed. Scope, FR-1, and the
  acceptance criteria updated from 20/3-embedding/71-total to 21/4-embedding/72-total
  accordingly.
- 2026-09-19 — A `devops` execution run flagged a count discrepancy present since this document's
  very first commit (`56e2e02`, 2026-09-18), unrelated to the two embedding-count amendments
  above: the Scope section's own bulleted Chat/VLM list has always held 18 distinct models, while
  the Scope summary line, FR-2, the acceptance criteria, and the Problem & current-state section's
  combo math all said 17 (and the Problem-section's "70 possible combinations" line was never
  updated across either embedding-count-growth commit, independently stale regardless of this
  question). `tico`'s own investigation (uninfluenced by the stakeholder's parallel read) found
  three independently-derived numbers — the Problem-section's original 70-combo arithmetic
  (2×1 + 17×4), this log's own 2026-09-18 "19-model list... locked in as-is" confirmation, and
  FR-2/AC's 17×4 math — all self-consistently pointing to 17 as the number actually held at
  authoring time, and suspected `mistralai_ministral-3-3b-instruct-2512` as the likely accidental
  18th bullet given its underscore-style catalog id (inconsistent with this list's otherwise
  uniform `publisher/model` convention for officially-published entries) and its apparent overlap
  with the already-in-scope `mistralai/ministral-3-3b`. Stakeholder ruling, direct and overriding
  that inference: **18 is correct** — both Ministral catalog entries (`mistralai/ministral-3-3b`
  and `mistralai_ministral-3-3b-instruct-2512`) are confirmed genuinely distinct Mistral releases,
  not a duplicate; the `-2512` suffix denotes a real, separate release. The bulleted Chat/VLM list
  was right all along; the surrounding prose was the stale part. Scope, FR-2, the acceptance
  criteria, and the Problem & current-state section corrected from 17/68/70/72 to 18/72/76/76
  (21/4-embedding/72-total to 22/4-embedding/76-total overall) accordingly. Two more numbers
  built on the old counts, not called out by the stakeholder but caught in the same pass, were
  corrected for consistency: FR-7's pairwise-matrix-size illustration (C(17,2)=136 → C(18,2)=153)
  and the Out of scope section's "21-model list" cross-reference (→ 22-model list). The
  2026-09-18 Decision log entry above, which computed "up to C(17,2)=136 pairs per chat pack" as
  part of the `data-scientist` consult record, is left as-is — it documents what was actually
  found and decided at that time under the then-believed 17-count, and this log is append-only.
- 2026-09-19 — Stakeholder decision, direct: drop `qwen3.5-9b-uncensored-hauhaucs-aggressive` and
  `qwen3.5-9b-claude-4.6-opus-uncensored-distilled` from the locked-in scope list. Reasoning
  given: `prism-ml/bonsai-27b` stays in scope on the existing footprint criterion — its ~3.5–5.5 GB
  quantized footprint already clears the threshold, per the first Decision log entry above; the
  stakeholder additionally names its quantization scheme as Q1_0 (not previously recorded in that
  entry). The two removed 9B models are a different case: unofficial third-party "uncensored"/
  "distilled" community finetunes (publishers `HauhauCS` and `LuffyTheFox`, non-standard `qwen35`
  arch tag) at Q4_K_M, judged by the stakeholder unlikely to work reliably under this harness
  (broken chat templates or malformed tool-call output being the specific risk on unofficial
  finetunes) — a **functional-risk** judgment, distinct from the footprint criterion, and not a
  re-litigation of it. Scope, FR-2, FR-7's pairwise-matrix illustration, the Problem &
  current-state combo count, and the acceptance criteria corrected from 18/72/153/76/76 to
  16/64/120/68/68 (22/4-embedding/76-total overall to 20/4-embedding/68-total overall)
  accordingly; the two removed models were also added to the Out of scope section's named-
  exclusions list, alongside the existing 12B-parameter precedent, so a future reader sees why
  they're out rather than just that they're absent. This is the third same-day scope correction to
  this document (the two embedding additions above, then the 17-vs-18 chat/vlm correction, now
  this removal) — scope is treated as locked as of this entry pending actual sweep execution;
  any further change should re-open rather than assume the list is still moving.
