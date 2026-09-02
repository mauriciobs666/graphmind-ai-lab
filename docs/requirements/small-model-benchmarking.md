# Small-LLM Benchmarking Tool — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-09-02

## Intent

A standing, reusable capability for measuring how **small local LLMs** behave — both **inference
(chat/tool-calling) models** and **embedding models** — so that any future "which model should we
pin?" question is answered by a run of our own, on our own hardware and our own data, rather than
by published leaderboard scores that may not test the shape we care about.

Not driven by one urgent decision. The value is in results that **accumulate and stay comparable**
across runs, models, and time.

## Problem & current state

Today, a "which small model?" question is answered from **published leaderboard scores and
reasoning**, not from measurement. That has bitten before: the salesperson tool-reliability review
had to use BFCL scores as proxies while noting that no published benchmark tests the failure shape
actually observed. Each new model question restarts from scratch, and what was learned last time
survives as prose inside a review document — not as numbers that can be lined up against a new
candidate.

What already exists (found during the interview):

- **`falkor-chat/server/tests/eval/`** — a real evaluation harness: golden sets for retrieval,
  guards and NL-queries; corpus provenance files; a metrics module; a pinned `retrieval_baseline.json`.
  It is a **regression gate for falkor-chat**, wired into that component's test suite — not a
  model-comparison tool, and not usable outside falkor-chat. _(Relationship to the new tool is an
  open decision — see Open questions.)_
- `falkor-chat/docs/plans/local-model-ram-budget-ml.md` — model selection under a 16 GB shared RAM
  budget, reasoned from published figures and RAM math.
- `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` — the model survey that had to lean
  on BFCL scores as proxies.
- K-042 (`config/models.json` + `ModelGateway`) — swapping a **chat** model is a config-only change,
  so acting on a benchmark result is cheap. Swapping an **embedding** model is not: a
  wrong-dimension vector is silently accepted by FalkorDB and dropped from the ANN index, so a swap
  means re-bootstrap plus full re-embed.

Note: the "salesperson" is a **workflow definition hosted inside falkor-chat** (`salesperson@v5`),
not a separate component. Root `AGENTS.md` still lists a `salesperson/` directory as a live
component; at the time of writing another session was in the middle of moving it to
`deprecated/salesperson/`. Unrelated to this feature — noted only so the reader isn't misled about
where the salesperson lives.

## User stories

- As the person maintaining this lab, I want to **measure a candidate small model myself**, so that
  I choose what to pin based on how it behaves on my hardware and my data rather than on a published
  leaderboard score for a task I don't run.
- As the person maintaining this lab, I want a **new model's result to line up against models I
  tested months ago**, so that I can tell whether the candidate is actually better without re-running
  everything.
- As the person maintaining this lab, I want to **point the same harness at a new scenario** by
  adding a task pack, so that a future project gets benchmarking without a new tool being built.
- As the person maintaining this lab, I want to be **told when a comparison isn't apples-to-apples**,
  so that a changed test set or a changed machine state doesn't quietly turn into a wrong conclusion.

## Functional requirements

### Running and recording

- **FR-1** — A run evaluates **exactly one model** against **one named task pack** and produces a
  scored result.
- **FR-2** — Every run's result is **persisted durably**, in a form that is both readable by a person
  and comparable across runs. _(Stakeholder preference: CSV or markdown — either is fine; the exact
  format is a design choice, not a requirement.)_
- **FR-3** — A run's result can be **compared against previously stored results** for other models on
  the same task pack, **without re-running those models**.
- **FR-4** — The harness evaluates **both kinds of model**: inference (chat / tool-calling) and
  embedding.
- **FR-5** — The harness itself is **independent of any component in this repo**; the scenarios come
  from **task packs**, which can be added without changing the harness.
- **FR-6** — Each stored result records enough **provenance** to know whether two results are
  legitimately comparable. When results scored on **different versions of a task pack** are compared,
  the output **visibly flags the mismatch** and still shows the comparison — the person judges, the
  tool refuses to let them forget.
- **FR-7** — A stored result carries a full **environment fingerprint**: model id, quantization,
  context length, KV-cache setting, LM Studio version, what else was resident in memory, temperature,
  and the task pack / golden-set / corpus version. A result missing its fingerprint is **invalid**,
  not merged into history. _(Rationale: LM Studio exposes two different catalog ids for the same
  weights on this box; quantization alone changes tool-call format validity.)_

- **FR-20** — A run is always scored **for a named role**, never as one overall "best small model"
  score. There is **no global leaderboard**. _(Rationale: rankings have already been observed to flip
  by role in this lab — a smaller model beats a larger one at tool emission while losing elsewhere.)_
- **FR-20a** — The tool uses **its own role vocabulary**, descriptive of what is being tested and not
  borrowed from any component: **tool-caller, guard-judge, nlq-generator, chat-responder, embedder**.
  _(falkor-chat's configuration uses a different set of names — `agent`/`step`/`guard`/`embedding`/
  `extraction` — deliberately not adopted here; mapping a result onto a component's config is the
  reader's job, not the tool's.)_
- **FR-23** — **The tool is standalone.** It has **no runtime dependency on falkor-chat** or any other
  component: it does not read `config/models.json`, the `ModelGateway`, or any component's
  configuration to decide what to test, what the incumbent is, or what a role means. It keeps **its
  own record** of models tested, roles, task packs and results. _(A task pack may still be seeded from
  golden data that originated elsewhere — that's data, copied in and versioned by the tool, not a
  live dependency.)_

- **FR-21** — First delivery covers **all five roles that exist in falkor-chat today**: tool-caller,
  guard judge, NL-query generator, chat responder, embedder. Current state of the golden data per
  role (found during the interview):

  | Role | Golden data today |
  |---|---|
  | Embedder | exists — `golden_retrieval.jsonl` (38 items) + pinned `retrieval_baseline.json` |
  | Guard judge | exists — `golden_guards.jsonl` (85) + `golden_judge_calibration.jsonl` (10) |
  | NL-query generator | exists — `nlq_golden_set.jsonl` (40) |
  | **Tool-caller** | **none durable** — prior 40-conversation experiments ran on throwaway scripts |
  | **Chat responder** | **none** |

- **FR-22** — The tool-caller role requires **fixed, versioned multi-turn conversation scripts** as a
  durable, verified asset — the main new golden-data cost in this feature, and the prerequisite for
  FR-9's per-turn reporting.

### What gets measured — inference models

- **FR-8** — Tool-calling accuracy is reported as **separate counts, not one percentage**, covering
  at minimum: (a) whether a tool was called at all when one was required; (b) whether the call
  arrived in native tool-call form rather than prose; (c) right tool chosen; (d) argument
  correctness, split into *omitted required argument*, *wrong value*, and *boundary/unit
  translation*; (e) spurious or duplicate calls; (f) stopping when done; (g) whether the final reply
  matches what the tool actually returned.
- **FR-9** — Every inference result is reported **per turn position** across a multi-turn
  conversation. A single-turn score is never a valid result. _(Rationale: the model falkor-chat pins
  today is flawless at turns 1–3 and fails in 39/40 conversations at turn 4. A single-turn harness
  would have re-picked it.)_
- **FR-9a** — The **prompt-assembly shape is part of the task pack's own versioned configuration**,
  not hardwired into the harness and not imported from any component (FR-23): how prior turns are
  replayed (plain text vs. structured tool-call scaffolding), whether tool schemas are re-presented
  each turn, how much history is carried. A real product's shape is reproduced as **one pack's
  settings**. _(Side benefit: the replay style itself becomes testable — you can ask whether it, not
  the model, is what breaks at turn 4.)_
- **FR-10** — Scoring is against **system ground truth** — the dispatched-call trace and the
  resulting graph state — never against the model's own reply text.
- **FR-11** — Speed is reported as **end-to-end turn latency (p50 and p95)** as the headline, plus
  time-to-first-token, prefill cost per 1k prompt tokens, and cold-load time reported separately and
  never averaged into steady state. Decode tokens/sec is a diagnostic only. **Peak RAM at the
  measured settings is part of the speed result.**

### What gets measured — embedding models

- **FR-12** — Retrieval quality is reported as: **recall@k** at the k actually fed to the model,
  **MRR**, **precision@k**, and **score separation** (gap between best relevant and best irrelevant
  similarity). _(MRR is expected to be the discriminating one: the existing baseline is already
  recall@10 = 0.974 but MRR = 0.626 — the headroom is in ranking, not reach.)_
- **FR-13** — Every embedding run includes a **keyword/full-text-only comparison arm**, so a quality
  number is always read against what search without embeddings achieves.
- **FR-14** — Every embedding result reports **output dimension, RAM cost, embedding throughput, max
  input length / truncation behavior**, and the **query/document prefix convention** the model
  requires (a per-model configuration field, never an assumption).

### Trustworthiness

- **FR-15** — Every reported rate carries a **95% confidence interval**, and two models are declared
  different **only when their intervals don't overlap**. "Not distinguishable at this sample size" is
  a **valid result**, not a failure.
- **FR-16** — Models are compared **paired**: both run the same fixed scripts/queries in the same
  session, compared item by item.
- **FR-17** — A run **can include any previously-tested model as a reference arm**, re-measured in the
  same session alongside the candidate. The tool **offers** this; it is not mandatory. _(Rationale:
  the box is shared — Windows + WSL2 + Docker + FalkorDB + LM Studio — so numbers from different days
  aren't strictly comparable without an in-session anchor. Re-measuring an old model is the way to
  get one.)_
- **FR-17a** — The set of models available as a reference arm comes from **the tool's own record of
  what it has tested**, never from reading another component's configuration (see FR-23).
- **FR-18** — Temperature is **explicitly pinned and recorded** for every run.
- **FR-19** — Golden sets carry per-item **provenance** (corpus/seed version, date, drafted by,
  verified by); queries are **paraphrases, never verbatim** copies of their targets; golden data is
  **test-only** and never seeded into a live corpus or prompt; an LLM may draft labels but **a human
  verifies every one**.

## Out of scope

- **Reproducing or reporting published leaderboard scores** (BFCL, MTEB, MMLU). They may be cited as
  background in a run's notes; they are never a result of this tool and never a substitute for a run.
- **Measuring small differences.** At the sample sizes this lab can afford (~20–40 runs per arm), the
  tool answers *"is A clearly better than B for this role?"* — differences of roughly 15 percentage
  points and up. It does **not** answer *"A is 3% better"*; that needs hundreds of runs per arm.
- **Hard pass/fail regression gating.** This tool produces ranked comparisons with intervals.
  Zero-tolerance gates belong to a component's own regression suite.
- **CI integration and scheduled runs.** A run is started deliberately by a person — never fired by a
  commit, a build, or a timer. Keeps benchmarking off the critical path of any build, and off a
  shared box unpredictably.
- **Auto-applying a winner.** The tool reports; it never edits any component's configuration to
  switch a model. Acting on a result stays a human decision — which also keeps FR-23 clean.

**Deliberately _not_ ruled out** (offered as out-of-scope candidates and declined — the door stays
open, but neither is a first-delivery requirement):

- **Cloud / hosted models.** The intent names small *local* models, but support for hosted models was
  not excluded.
- **Subjective / LLM-as-judge quality scoring.** Not excluded — and likely needed for the
  chat-responder role, where "correct" isn't a ground-truth match. If used, it inherits the same
  honesty rules as any other measurement (FR-15, FR-19).

## Acceptance criteria

- **AC-1** — Given a candidate inference model and a task pack, when a run completes, then the result
  reports accuracy **broken out per failure kind and per turn position** — a single-turn-only score
  is not a valid result and must not be reported as "tool-calling accuracy".
- **AC-2** — Given a stored result missing any part of its environment fingerprint, when history is
  read, then that result is treated as **invalid** rather than merged into comparisons.
- **AC-3** — Given two results scored on **different versions** of the same task pack, when they are
  compared, then the output **visibly flags** the version mismatch.
- **AC-4** — Given two models whose confidence intervals overlap, when results are reported, then the
  output says they are **not distinguishable at this sample size** rather than ranking one above the
  other.
- **AC-5** — Given an embedding run, when results are reported, then a **keyword-only arm** appears
  alongside, and the model's **output dimension** is recorded.

## Constraints for design (not requirements — for the architect to resolve)

- **Relationship to `falkor-chat/server/tests/eval/`** — deliberately **left open for design time**,
  to be settled with the actual code in front of the architect. The stakeholder's concern to honour:
  duplicating golden sets and metric code would mean two things to keep honest. Options considered
  and neither ruled out: extract-and-generalize (one home, but touches a working regression suite),
  build-alongside-and-reuse-the-golden-data, or a clean build. **Whatever is chosen must be
  justified against the duplication risk** — and must respect **FR-23**: the dependency may only run
  *from* falkor-chat *to* the tool, never the reverse. Golden data may be **copied in and versioned by
  the tool**; it may not be read live out of falkor-chat's test tree.

## Open questions

1. Where the tool lives and what it's called (a new top-level component?).
2. Which task packs / roles exist at first delivery.

## Decision log

2026-09-02 — What decision should this tool settle first? → **None in particular.** It's a standing
capability: a reusable harness so any future "which small model?" question is a run away, with
results accumulating over time.

2026-09-02 — What should the tool measure? → **Inference models: tool-calling accuracy and speed.
Embedding models: retrieval quality** — with the stakeholder explicitly open to better indicators
on the embedding side.

2026-09-02 — Test models on our tasks or on general capability? → **Generic core with pluggable task
packs.** The harness itself knows nothing about our components; task packs point it at real
scenarios and more can be added later. Accepted the extra build cost to keep both reusability and
"will it work here" fidelity.

2026-09-02 — How does a run work, and where do results live? → **One model at a time, scored against
stored results** from earlier runs (not an all-models-at-once shootout). Durable record: **CSV or
markdown, either is fine** — recorded as a preference, not a requirement.

2026-09-02 — What happens when a task pack changes under stored results? → **Flag the mismatch and
still show the comparison.** The tool refuses to let the mismatch go unnoticed; the person judges.

2026-09-02 — How should the new tool relate to falkor-chat's existing eval harness? → **Decide at
design time.** Recorded as a design constraint rather than settled now; the architect resolves it
with the code in front of them, and must justify the choice against the duplication risk.

2026-09-02 — Score per role, or one overall "best small model"? → **Always per named role. No global
leaderboard.** (FR-20)

2026-09-02 — Which roles on day one? → **All five known roles** (tool-caller, guard judge, NL-query
generator, chat responder, embedder). Cost checked during the interview and accepted: three roles
already have verified golden sets in `falkor-chat/server/tests/eval/`; the tool-caller and chat
responder do not, and the tool-caller's versioned conversation scripts are the real new asset
(FR-21, FR-22).

2026-09-02 — Should each run re-measure the currently-pinned model as a reference arm? → **Reframed
by the stakeholder.** The tool **offers to retest any already-tested model** as an in-session
reference, but **must not depend on falkor-chat** to know what is pinned — "it is supposed to be
standalone so it should have its own tracking". Recorded as FR-17/FR-17a and, more broadly, **FR-23
(standalone, no runtime dependency on any component)**, which also constrains how the existing eval
harness may be reused.

2026-09-02 — Whose role vocabulary? → **The tool's own**: tool-caller, guard-judge, nlq-generator,
chat-responder, embedder. falkor-chat's `agent`/`step`/`guard`/`embedding`/`extraction` names are
deliberately not adopted — the harness is generic and shouldn't speak one component's dialect
(FR-20a).

2026-09-02 — FR-9 conflicted with the standalone rule (it required driving models through
falkor-chat's own prompt-assembly path). → **Resolved:** the **prompt-assembly shape becomes part of
the task pack's versioned configuration** (FR-9a), reproduced by the harness rather than imported. A
product's shape is one pack's settings, which also makes the replay style itself testable.

2026-09-02 — What's explicitly out of scope? → **CI/scheduled runs** and **auto-applying a winner**,
both ruled out. Two further candidates were offered and **declined**: cloud/hosted models, and
subjective/LLM-as-judge quality scoring — recorded as "deliberately not ruled out" rather than as
requirements.

2026-09-02 — **Consult: `data-scientist`** (fast-track methodology question). Why: the stakeholder
invited better indicators for embedding quality, and "tool-calling accuracy + speed" needed
decomposing before it could be made testable. Asked: which indicators earn their keep on both sides,
what the evaluation data must look like, and the biggest failure mode of a home-grown small-model
benchmark. **Outcome:** substantial. Accuracy decomposed into seven counts reported **per turn
position** (FR-8/FR-9); scoring against system ground truth, not reply text (FR-10); latency
percentiles over tokens/sec (FR-11); four retrieval indicators with MRR expected to discriminate,
plus a keyword-only baseline arm and five practical non-quality indicators (FR-12/13/14); confidence
intervals, paired design, in-session reference arm, golden-set honesty rules (FR-15..FR-19); and the
headline trap — **benchmarking the model instead of the job** — written into AC-1 and the out-of-scope
list. Also surfaced the pre-existing `falkor-chat/server/tests/eval/` harness and the "benchmark for
a named role" framing, both now open questions for the stakeholder.
