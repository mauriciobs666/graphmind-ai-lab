# M3 LLM-native executor — ML method note (data-scientist)

> **Status:** proposed method note. Advisory deliverable — implementation routes to
> `coder`/`tdd-engineer`; in-graph vector/Cypher changes route to `graph-dba`.
> **Author:** data-scientist · **Date:** 2026-07-10
> **Folds into:** `docs/archive/plans/m3-executor.md` at **Phase 2**; gates units U6–U8 of
> `docs/archive/plans/m3-executor-coordination.md`.
> **Builds on:** `docs/plans/graphrag-eval-ml.md` (retrieval + faithfulness harness — reused, not
> re-derived). **Reads:** `server/falkorchat/{responder,llm,config}.py` (the real seams).

## The question & the decision at stake

The executor makes every node and every transition an LLM call on a **local 4B model**
(`qwen/qwen3-4b-2507`, non-thinking Instruct, via LM Studio). Four method decisions gate whether that
is trustworthy, and the coder cannot implement the judge, the retrieval threshold, or the eval
without them. This note answers all four §10 questions concretely enough to build from:

1. Is an LLM reliable enough to judge a fuzzy transition guard, and how do we prove it? (FR-2/AC-2)
2. How does the research node stay grounded, and how do we score AC-3? (FR-5b/AC-3)
3. Native function-calling or structured-output prompting for the agent loop on a 4B model? (§2.2)
4. What `maxSteps`/`maxIterations` defaults follow from (1)–(3)'s reliability? (§7)

**Decision served:** whether Phase 2 ships the judge + agent loop as designed, and with which
concrete prompt, threshold, and safety numbers. The seams are already injectable
(`guards.evaluate_guard(judge=…)`, research node `retriever+threshold`, swappable LLM), so these
recommendations drop in without reshaping the executor.

## Findings from the real system

1. **The LLM seam is single-shot text today.** `llm.LLM` is `complete(messages) -> str`
   (`llm.py:29-32`); `LMStudioLLM` posts OpenAI `/v1/chat/completions` with **no `tools` field**
   (`llm.py:66-71`). Native tool-calling is genuinely new surface, not a config flip.
2. **The responder is the anti-pattern the research node must not copy.** `AgentResponder` requests
   `k=10` and feeds **all** seeds raw into the prompt regardless of score (`responder.py:70-80`,
   `_k=10`), under a grounding-*permissive* system prompt ("If the context does not help, answer from
   general knowledge" — `responder.py:35-39`). `graphrag-eval-ml.md` finding 2 already measured a
   distractor seed at cosine distance **0.786** fed as "context" to this exact 4B model. No
   thresholding, no citation forcing, no abstention.
3. **The model.** Qwen3-4B-Instruct-2507 *does* support tool calling and LM Studio exposes it over
   the OpenAI API, but the vendor is explicit that protocol adherence "is not guaranteed… especially
   for templates that rely more on the model itself to stay on track," and there are known LM
   Studio-side tool-call format/parse issues across Qwen3 variants (see Sources). Treat multi-tool
   free-choice on a 4B as uneven; single-tool, well-fenced calls as usually fine.
4. **The prior harness is directly reusable.** `graphrag-eval-ml.md` already specifies a two-layer
   offline eval (deterministic recall@k/MRR + a calibrated LLM-as-judge faithfulness layer), a
   paraphrased human-verified golden set, the self-retrieval-inflation guard, and pytest-behind-a-
   live-marker placement. This note extends it with a guard-judge layer and a findings-groundedness
   layer rather than inventing a new harness.

---

## Q1 — LLM-as-judge fuzzy guard reliability (FR-2/AC-2) — the core

### Recommended pattern: extract-then-judge, not judge-the-transcript

The single highest-leverage decision here. **Do not** ask the judge to read the raw thread and rule
"enough info?" — a 4B model degrades on long, noisy conversational context and its verdict variance
is high. Instead split responsibility:

- The **intake node** (already an LLM call) emits, as part of its structured output, a short
  `understanding` object: what the request is, the concrete facts gathered, and an explicit
  `missing: [...]` list of what it still needs. This is cheap — the node is already reasoning.
- The **guard judge** is a *separate, minimal* LLM call that judges the guard condition against that
  compact `understanding` (plus a short raw-turn fallback), **not** the transcript.

This decouples the verdict from conversational noise, makes the judge prompt tiny (better on a 4B),
and produces the rationale FR-4 wants for free (the `missing` list *is* the "why").

> Keep the judge call **distinct from the node call** (do not let the node that just decided it has
> enough info also score its own guard — that is textbook self-preference bias). Use the strongest
> available local model as the judge if one is loaded; fall back to the same 4B only after the
> calibration below passes.

### Judge prompt (implementable as-is)

```
SYSTEM:
You decide whether a workflow may advance past a gate. You are given a CONDITION and the
CURRENT STATE. Decide ONLY whether the CONDITION is *clearly* met by the STATE. If the state is
incomplete, ambiguous, or you are unsure, answer false. Output ONE JSON object, nothing else.

USER:
CONDITION: {guard.text}

CURRENT STATE:
{understanding_json}          # {"request": "...", "known": [...], "missing": [...]}
RECENT TURNS (context only):
{last_n_turns}                # fallback, N = 6, newest last; omit if understanding is present

Respond exactly:
{"decision": true|false, "rationale": "<one short sentence>"}
```

> **Implemented (2026-07-15, K-022 U14):** this prompt — including the RECENT-TURNS fallback and its
> omit rule — is built at `app._build_llm_judge` / `app._render_judge_user`, with the precedence and
> the N=6 window at `guards._recent_turns` / `guards.evaluate_guard`. The fallback had been specified
> here but **lost in implementation** (the judge was handed `{}` every turn and suspended forever):
> this note's own **risk #4** materialized as **Defect A**, fixed per
> [`m3-guard-thread-context.md`](m3-guard-thread-context.md). The *method* below was right and is
> unchanged.

- **Output schema:** `{"decision": bool, "rationale": str}`. Request JSON/structured output mode.
  **Do not** ask a 4B for a confidence score — its self-reported confidence is not calibrated and
  would be false precision. The bias rule below replaces it.
- **Context slice fed:** the `understanding` object (primary) + the **last 6 thread turns** as raw
  fallback. Cap the whole judge user-message at **~1500 tokens**. Never feed the full thread — Q1's
  failure mode on this model is long-context dilution.

### Calibration / agreement method (the reliability gate)

Build a small **guard golden set** — `server/tests/eval/golden_guards.jsonl`, **20–30** cases, each
`{understanding, recent_turns, condition, expected_decision}`, hand-labeled by a human. Cover the
hard middle: barely-enough, one-fact-missing, over-complete, off-topic-reply. Include both intake
("enough info to research") and any other `kind:'llm'` guard the proof flow uses.

Run the judge over it and report:

- **Accuracy** and **Cohen's κ** vs the human labels (κ corrects for the base-rate imbalance a raw
  accuracy hides).
- The **confusion matrix**, and specifically the **false-advance rate** = P(judge says advance |
  human says suspend). This is the *dangerous* error — it ships an answer grounded in too little.

**Trust thresholds (gate before the judge is wired live):** **κ ≥ 0.6** (substantial agreement)
**and false-advance rate ≤ 10%**. If either fails: first make the guard `text` more concrete
(enumerate the fields "enough information" means for *this* workflow — a specific condition beats a
vague one on a weak judge), then try a stronger judge model; only then reconsider the guard.

### Ambiguous-verdict policy — bias toward suspend (for human-unblockable guards)

The two guard classes have **opposite** safe biases, because their failure modes differ:

- **Intake guard ("enough info?") — bias to SUSPEND (decision=false).** A false-advance wastes the
  whole run on a bad answer; a false-suspend costs one more cheap clarifying question that a human
  can resolve. So on **parse failure, invalid JSON, or rationale-contradicts-decision → treat as
  `false`** (keep asking). Safe *because* a human is in the loop to unblock it.
- **Research→answer guard ("findings sufficient?") — DO NOT judge; make it unconditional.** There is
  **no human to unblock** a suspended research node, so a bias-to-suspend there is a silent infinite
  loop against `maxSteps`. Recommend the research→answer transition be **unconditional** (fires when
  reached) and push "are findings good enough?" into the *research node's own* abstention (Q2), not a
  guard. This matches the plan's "(or unconditional)" note in §8 — take the unconditional branch.

### Keeping the intake loop from over-/under-asking

Bias-to-suspend alone would let a stubborn judge ask forever. Bound it with a **clarifying-round
ceiling in `ctx`**: track `intakeRounds`; after **3** clarifying rounds, **force-advance** to research
with whatever was gathered (graceful degradation — a mediocre answer beats an endless interrogation),
and trace the forced advance. So: bias-to-suspend on ambiguity, hard ceiling at 3 rounds, `maxSteps`
as the ultimate backstop. Under-asking is prevented by the ceiling; over-asking by making the guard
`text` concrete so "enough" is judged against a real checklist, not vibes.

---

## Q2 — Research-node grounding & AC-3 evaluation (FR-5b/AC-3)

### Relevance threshold / seed count — fix the responder's raw-`k=10` anti-pattern

The research node must **not** feed all `k` seeds raw. Recommended retrieval-to-context policy:

1. Retrieve `k=10` from `hybrid_search` (cosine distance ASC, as the code ranks).
2. **Distance cutoff τ:** keep only seeds with cosine distance **≤ τ**. **Starting candidate
   τ = 0.5** (similarity ≥ 0.5) — *this is a tuning starting point, not a measured value*; it must be
   calibrated on the golden set (below), accepted only if it holds recall while cutting median
   seeds-fed, per `graphrag-eval-ml.md`'s threshold-tuning row. The measured 0.786 distractor sits
   well above any sane τ, confirming a cutoff has real signal to remove.
3. **Cap and floor:** after the cutoff, keep at most **5** seeds (fewer distractors for a
   distractor-sensitive 4B); if **zero** seeds pass τ, do **not** fall back to raw top-k — enter the
   **abstention** path.
4. **Abstention:** if no seed passes τ (or the best distance exceeds an abstain cutoff
   τ_abstain ≈ 0.7), the research node produces findings of the form *"no relevant context found for
   X"* rather than synthesizing from noise. The answer node then says so honestly. This is the
   grounding value-prop AC-3 is meant to prove.

### Grounding the node itself (prompt-side)

Replace the responder's permissive prompt for the research/answer nodes with a **grounding-strict**
one: *"Use ONLY the retrieved findings below. If they do not answer the question, say so — do not use
outside knowledge. Cite the seed you used for each claim."* Citation-forcing + a concrete abstention
path are the cheap mitigations that pay for themselves (they also make the faithfulness check below
scoreable). This is the same fix `graphrag-eval-ml.md` finding 3 flagged; apply it to the workflow
nodes now.

### Faithfulness / groundedness check for produced findings (AC-3 acceptance)

Reuse `graphrag-eval-ml.md` Layer 2, specialized to findings. For a small set (**~15**) of seeded
triage queries, judge each produced `findings` object with an LLM-as-judge on a **3-point
groundedness scale** — *fully grounded* (every claim traceable to a fed seed) / *partially* / *
ungrounded* — with the fed seeds supplied to the judge. Score **grounded-when-relevant** separately
from **correctly-abstained** (do not penalize an honest "no relevant context"). **Calibrate the judge
against ~10 human labels first** (report judge–human agreement; trust only at agreement ≥ ~0.7); human
spot-check fallback if it fails. Same judge-model discipline as Q1.

### Retrieval-quality metric for AC-3 (deterministic, primary)

This is the number that objectively judges AC-3, and it needs no judge. Extend the prior harness's
**`golden_retrieval.jsonl`** with pairs drawn from the **seeded triage corpus** (the AC-3 test
precondition data): `query → relevant_msgId(s)`, **paraphrased not verbatim** (self-retrieval
guard), human-verified. Metrics over `hybrid_search`:

- **recall@k** (k=5 and k=10) — *primary*: did the relevant seed reach the node? Bounds AC-3.
- **MRR** — secondary: rank of the first relevant hit.

**AC-3 acceptance:** on the seeded corpus, **recall@5 ≥ baseline** (baseline = first run of vector-
only @1024, recorded not gated), the produced findings score *fully/partially grounded* on the
faithfulness layer, and off-topic queries **abstain** rather than fabricate. Retrieval quality and
generation quality are measured **separately** — if AC-3 fails, this tells you whether to blame
retrieval (low recall) or the node (low groundedness at good recall).

> In-graph mechanics (the vector-index DDL, `db.idx.vector.queryNodes`, any fusion) remain a
> **graph-dba** handoff — this note owns τ, seed count, the golden set, and the metrics above them.

---

## Q3 — Function-calling vs structured-output on the 4B model

**Recommendation: native tool-calling as the primary seam, JSON-structured-output as the wired
fallback, with the proof-flow fences kept minimal — and measure format validity before trusting it.**

Rationale from findings 1 and 3: the proof-flow nodes are **tool-light** (intake → `post_message`;
research → `graphrag_retrieve`; answer → `post_message`) — 1 tool each. Single-tool, well-fenced
calls are where a 4B is *most* reliable, so native tool-calling (the `chat(messages, tools)`
extension the plan already proposes) is a reasonable primary path here. The risk is multi-tool free
choice and format/parse flakiness, not single-tool dispatch.

**Guardrails (all required, implementable):**

1. **Minimal fences on the 4B.** Grant 1–2 tools per node in the proof flow (the AC-6 fence doubles
   as a reliability lever — fewer choices, fewer malformed calls). More tools per node only after the
   validity measurement below supports it.
2. **Structured-output fallback parser.** When the model emits a "tool call" as text/JSON in
   `content` instead of the structured `tool_calls` field (a common LM Studio failure mode for
   Qwen3), parse it. Accept both shapes.
3. **Schema + scope validation with bounded re-prompt.** Validate the tool **name against the granted
   set** (AC-6 defensive reject, not just omission) and args against the tool schema; on invalid,
   re-prompt once with the error, bounded by `maxIterations`. Never dispatch an ungranted or
   malformed call.
4. **Prefer direct output where a tool isn't essential.** The intake/answer "post a message" action
   can be the node's **final text** rather than a tool round-trip — reserve tool-calling for
   `graphrag_retrieve` where a structured call is genuinely needed. Fewer agentic hops = fewer 4B
   failure points, and still honors FR-6 (the model chose to post).

**Measure before trusting (small fixture, not a full harness):** a **tool-call sanity set** of ~15
node situations. Report two rates: **format-validity** = well-formed, parseable, in-scope call when a
call is expected; **tool-selection accuracy** = right tool chosen. **Decision rule: if format-validity
< ~90% for a node, switch that node to explicit JSON-structured-output prompting** (ask for
`{"action": "...", "args": {...}}` and parse) instead of native `tools`. Structured-output prompting
is more robust on weak models precisely because it doesn't depend on the chat template's tool grammar.

---

## Q4 — Runaway-safety defaults

Derived from (1)–(3), not guessed. The proof flow is 3 nodes; the only legitimate cycle is the intake
self-loop, bounded by the Q1 clarifying-round ceiling of **3**.

- **`maxSteps` (run budget) = 12.** Worst legitimate path ≈ 3–4 intake StepRuns (up to the round
  ceiling) + research + answer ≈ 6 steps. **12** is ~2× headroom — catches a genuine runaway well
  before it gets expensive, without tripping a healthy flow. The plan's 25 is over-generous for a
  3-node flow; **recommend 12 as the per-def default with a global hard ceiling of 25** for larger
  future workflows. `maxSteps` counts every StepRun (including intake re-runs); a `waiting` run
  consumes none while parked — correct, since the human unblocks it.
- **`maxIterations` (per-node tool loop) = 4.** With 1–2 tools per node and uneven 4B tool-calling,
  more iterations mostly means the model is thrashing, not progressing — fail fast and surface it.
  4 covers retrieve → (optional) re-retrieve → synthesize with margin. The plan's 6 is fine as an
  upper bound for a wider-fenced node; **recommend 4 for the tool-light proof nodes.**
- **On `maxIterations` exhaustion → terminate the node with its best current text + a trace note**
  (graceful — do not hard-fail the run); on **`maxSteps` exceeded → `status=failed`** with a
  `TraceEvent` ("step budget exceeded"), per plan §7. Bias asymmetry: a node giving up is recoverable;
  an unbounded run is not.

These defaults are **coupled to the reliability gates above**: if the Q1 judge passes calibration and
the Q3 validity rate is high, 12/4 hold. If either is shaky, the ceilings are what keep a flaky judge
or a thrashing agent bounded — do **not** raise them to paper over a failing calibration.

---

## Evaluation design (metric · data · threshold)

| Layer | Metric | Data | Acceptance threshold |
|---|---|---|---|
| **Guard judge (Q1)** | ~~accuracy, **Cohen's κ**, **false-advance rate**~~ → **SUPERSEDED** | `golden_guards.jsonl` — authored at `server/tests/eval/golden_guards.jsonl` (26 cases) | ⚠️ **This row's gate (κ ≥ 0.6 AND false-advance ≤ 10%) is superseded by [`m3-guard-calibration.md`](m3-guard-calibration.md) §4.** κ is symmetric and this note's own Q1 bias-to-suspend decision makes the judge deliberately asymmetric; κ is retained as a **reported diagnostic**, and the gate is now **false-advance ≤ 10% (screen) AND advance-recall ≥ 0.80**. The *intent* of risk #1 below — never wire an uncalibrated judge — is unchanged. |
| **Retrieval (Q2, AC-3)** | **recall@5** (primary), recall@10, MRR | `golden_retrieval.jsonl` extended to the seeded triage corpus; paraphrased, human-verified | First run = baseline (record). AC-3: **recall@5 ≥ baseline**; no verbatim self-retrieval (assert). |
| **Findings groundedness (Q2, AC-3)** | 3-point groundedness (grounded-when-relevant vs correctly-abstained) | ~15 seeded triage Q&A | Report as baseline; **judge–human agreement ≥ ~0.7** before trusting; human spot-check fallback. |
| **Threshold τ tuning (Q2)** | recall@5 vs median seeds-fed | same retrieval golden set | τ accepted only if it **holds recall@5** while cutting median seeds-fed (drops noise, not signal). |
| **Tool-calling (Q3)** | format-validity, tool-selection accuracy | ~15 node situations | Native `tools` for a node only if **format-validity ≥ ~90%**; else structured-output prompting. |

**Harness placement (for the coder, reuse the prior note's pattern):** offline, pytest behind the
existing **live-LM-Studio marker** so the network-free baseline stays green; fixtures under
`server/tests/eval/`; cache embeddings for determinism; emit a metrics report
(`docs/test-reports/m3-executor-eval-<date>.md`). Golden fixtures are **test-only, never seeded into
a live prompt/few-shot** (leakage guard). The judge is the injected `guards.evaluate_guard(judge=…)`
callable, so calibration runs against the real judge without touching executor code.

## Risks & open questions

1. **Judge calibration is the executor's reliability gate (highest).** If `golden_guards.jsonl` fails
   the gate (**thresholds now per [`m3-guard-calibration.md`](m3-guard-calibration.md) §4/§7**, not the
   κ figure once quoted here), the fuzzy guard is not trustworthy and the intake loop should not
   ship live on the 4B — escalate to a stronger judge model or more concrete guard `text`. Do not
   wire an uncalibrated judge and rely on `maxSteps` to hide it.
2. **Seeded-corpus representativeness (inherited).** AC-3's retrieval metric is only as good as the
   seeded triage corpus. Per `graphrag-eval-ml.md`, recommend a synthetic-but-realistic multi-topic
   seed (reproducible, CI-friendly), not the thin `seed_demo.sh`. **Open:** who authors the seed — the
   AC-3 test precondition names the implementer/QA; flag that "build a representative triage corpus"
   is step 0 of Phase 5, not an afterthought.
3. **Local-judge validity (inherited).** The same 4B judging its own guard is self-preference-prone;
   prefer a distinct/stronger local judge. Report judge–human agreement for both the guard layer and
   the findings-groundedness layer; never report those numbers uncalibrated.
4. **`understanding`-object dependency.** The extract-then-judge pattern assumes the intake node
   reliably emits the `{request, known, missing}` object. If the 4B is unreliable at that structured
   emission (measure it with the Q3 validity check), fall back to judging the last-6-turns slice
   directly — degraded but functional; expect lower κ and tighten the round ceiling to compensate.
5. **τ is a starting candidate, not measured.** 0.5 / τ_abstain 0.7 are tuning seeds to calibrate on
   the golden set — the coder must run the threshold-tuning row before hard-coding any cutoff. Not a
   number to ship on faith.
6. **Decision needed from the caller:** confirm the research→answer guard is made **unconditional**
   (Q1) rather than LLM-judged — the plan's §8 offers both; this note recommends unconditional to
   avoid an unblockable suspend loop. If the caller wants it judged, it needs its own golden-set
   calibration and a non-suspend ambiguity bias.

---

**Sources (perishable model-capability facts, verified 2026-07-10):**
- [Qwen3-4B · LM Studio](https://lmstudio.ai/models/qwen/qwen3-4b-2507)
- [Function Calling — Qwen docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html) (protocol adherence "not guaranteed")
- [Qwen/Qwen3-4B-Instruct-2507 · Hugging Face](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [LM Studio tool-call format issue #825](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/825) (Qwen3 tool-call parse flakiness)
</content>
</invoke>
