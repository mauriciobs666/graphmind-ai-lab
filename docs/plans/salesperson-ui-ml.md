# Salesperson UI — DEF-6 wrong-language-under-concurrency diagnosis

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** DEF-6 (`docs/test-reports/salesperson-ui-report.md`) (M<n> TBD)

## The question and the decision it serves

`docs/test-reports/salesperson-ui-report.md`'s DEF-6: `en`-configured participants sometimes get
a fully-formed, coherent **Spanish** reply under concurrency (5/10 conversations at 40-way load,
2/10 trials even at the plan's own literal "three participants... simultaneously" wording; never
at concurrency=1). QA read `falkor-chat/server/falkorchat/executor.py` directly, found no
shared-mutable-state bug at the application layer, and left "LM Studio's own concurrent-request
serving layer" as a stated hypothesis, not a finding.

**Decision this feeds:** greenlight or gate the first live, audience-facing demo specifically
(not the current docs-closeout milestone) — mirroring how K-056 gated AC-10 in this same feature
(`falkor-chat/docs/HISTORY.md` 2026-08-30 entry, 2026-08-28 entry).

## What was investigated and how

Built a standalone harness (not committed — lives only in this session's scratchpad) that
bypasses `falkor-chat`/`salesperson` entirely and hits LM Studio's `/v1/chat/completions`
directly at `http://localhost:1234/v1`, replaying production's exact request shape:

- **`systemPrompt`** copied verbatim from `SALESPERSON_DEF` `v7`
  (`falkor-chat/server/falkorchat/proof_defs.py:363-408`) — including its one language
  instruction: *"Reply in the language named by `language` in the CONTEXT block; if no language
  is named there, reply in English."* This is the **only** per-language differentiator in the
  entire prompt — the system prompt text is byte-identical across `en`/`es`/`pt-BR` participants.
- **Message shape** copied from `executor._assemble_messages`/`_append_turn`
  (`executor.py:1243-1278`, `:233-250`): `system` message, then one coalesced `user` turn shaped
  `"{displayName}: Hi there!\n\nCONTEXT:\n{"language":"en"}"` (the two turns merge because
  `_append_turn` collapses consecutive same-role messages — Ministral's chat template
  hard-rejects non-alternating roles, per the function's own docstring).
- **Tools**: the real `post_message` schema (`tools.py:225-257`) plus stub schemas for the other
  ten granted tools, so total prompt/tool-schema size and tool-choice pressure roughly match
  production. `temperature: 0` (`falkor-chat/config/models.json` pins this for
  `lmstudio/mistralai/ministral-3-3b`). A minimal multi-round tool loop (assistant tool-call turn
  echoed back, stub tool result appended, repeat) was added after an early single-shot version
  produced false "empty reply" mismatches that were purely a harness artifact (the model
  legitimately calling `get_profile` first, per its own prompt instructions, before
  `post_message`).
- **Concurrency fired via `ThreadPoolExecutor`** sized to the batch (genuine simultaneous
  dispatch, not client-side queuing), classifying each reply's language with a small regex
  heuristic (adequate for this diagnostic; not proposed as a production classifier — see
  Mitigation C below).
- Confirmed environment: LM Studio reachable at `localhost:1234` (not the shared
  `192.168.0.69:1234` — same substitution QA's own report used), model
  `mistralai/ministral-3-3b` loaded, `Q8_0`, `loaded_context_length: 8192` (below the
  `opencode/docs/manuals/local-llm.md`-recommended 16K+ — noted, not investigated further; no
  evidence it's relevant to DEF-6 specifically).

## Findings

### 1. Confirmed: reproduces directly against LM Studio, application code entirely bypassed

At 3-way concurrent load (one `en` + one `es` + one `pt-BR` request fired simultaneously,
matching the plan's own literal wording and QA's own "literal-concurrency variant" test design),
one of 5 trials produced a fully-formed, coherent Spanish reply for the `en` participant:

```
Trial2-en  expected=en  got=es
  "Hola! ¿Cómo puedo ayudarte hoy? Por favor, dime qué necesitas saber o comprar."
```

This is **direct, application-bypassing confirming evidence** for the LM-Studio-layer hypothesis
— no `falkor-chat`/`salesperson` code was in the loop. The application-layer explanation is not
just "not found" (QA's own read of `executor.py`) but actively **ruled out** as the sole cause:
the same failure occurs with zero application code present. This resolves DEF-6's own "recorded
as measured evidence with a stated hypothesis, not a proven root cause" status: the layer
attribution (LM Studio's own concurrent-request serving/decoding path, not `falkor-chat`'s
`run_ctx` handling) is now a **finding**, not a hypothesis, for the general mechanism. The exact
internal cause inside LM Studio/llama.cpp remains open — see "What remains a hypothesis" below.

Point-estimate rate at this exact condition: 1/5 here vs. QA's own 2/10 at the same design — both
far too small an *n* to compare formally (Wilson 95% CI at 1/5 is roughly [0.02, 0.65]; at 2/10,
roughly [0.06, 0.51]) but the two point estimates (~20%) land in the same neighborhood. **QA's
own 2/10 and 5/10 figures remain the better rate estimates to plan around** — this repro's own
sample sizes (5-13 per condition, one-shot bursts rather than QA's sustained 5-turn-per-participant
load) are a characterization aid, not a replacement measurement.

### 2. The corrupted reply is not a full request/response swap

Every reproduced (and QA's own quoted) wrong-language reply keeps the **correct** customer
context — right display name, on-topic greeting, grammatically coherent in the wrong language.
This rules out the naive "concurrent request N got concurrent request M's whole response"
mechanism: that would carry the *other* participant's name/content, not just the wrong language
with everything else right. What's shared and colliding is narrower than the whole response —
consistent with something specific to the **language decision** rather than general
request/response cross-talk (full KV-cache-slot mixup, wholesale prompt-prefix swap).

### 3. Diagnostic: an all-English concurrent batch never drifted (0/8, plus every concurrency=1 baseline correct)

A batch of 8 concurrent `en`-only requests — no `es`/`pt-BR` request anywhere in the batch —
produced 8/8 correct English replies, at higher per-call latency (4-6.5s) than the 3-way mixed
runs that did show a failure. This is the single most decision-relevant new data point: it
suggests (does **not** statistically establish, at n=8) that the failure needs a **concurrently
co-batched other-language request**, not merely "any concurrency" or "a large batch." That
distinction matters for mitigation choice (§ below): a semaphore that merely caps *how many*
requests are in flight, without regard to language mix, may under-protect relative to one that's
aware of it.

Caveat stated plainly: 0/8 at one condition does not prove the necessary-condition claim — it is
consistent with it and worth a larger confirmatory run (say n=30-40 all-English) before treating
it as settled, which this session's time budget did not extend to.

### 4. No dose-response curve established from this session's own data — and a likely reason why

Unlike QA's own finding (higher concurrency → higher rate: 20% at 3-way, 50% at 40-way), this
session's heavier conditions (12-way mixed: 0/4 `en` mismatches; ~40-way mixed matching QA's own
language ratio: 0/13 `en` mismatches) did **not** reproduce at a higher rate than the 3-way
condition — if anything the opposite, though every one of these counts is small enough that this
reads as noise around a low base rate, not a contradicting trend.

The more likely explanation is a **methodology difference, not a contradicting result**: this
session fired every batch as one simultaneous burst that mostly completes and empties the queue;
QA's own heavy run drove 40 participants through **5 sequential turns each** with sustained,
evolving concurrent load (up to ~200 concurrent completions strung together over the run's
duration, with growing per-conversation message-list length as turns accumulate) — a
steady-state queueing/batch-composition regime this session's one-shot bursts don't reproduce.
**QA's own rate-vs-concurrency reading stands**; this session corroborates the mechanism and the
layer, not the curve.

## What is confirmed vs. what remains a hypothesis

**Confirmed (measured this session, LM Studio only, application code bypassed):**
- The wrong-language failure reproduces with zero `falkor-chat`/`salesperson` code in the
  request path, using production's exact prompt/message/tool shape.
- The application-layer explanation is ruled out as the *sole* cause (QA's code read already
  argued this; this adds a positive repro that doesn't need the app at all).
- The corrupted output preserves correct customer-specific content and is not a full
  response/request swap.
- An all-English concurrent batch showed no drift in this session's own (small) sample —
  suggestive that heterogeneous-language co-batching is a contributing factor, not concurrency
  alone.

**Still a hypothesis (not confirmed by this session, needs deeper LM Studio/llama.cpp visibility
than this session's tooling/budget had):**
- The exact internal mechanism — prompt-prefix/KV-cache slot cross-talk between concurrently
  batched sequences vs. batch-composition-dependent floating-point non-associativity flipping a
  near-tied greedy decision (temperature is pinned to 0, so decoding is otherwise deterministic
  per single request) vs. something else specific to LM Studio's/llama.cpp's continuous-batching
  implementation. Distinguishing these needs server-side logs/instrumentation this session did
  not have access to, or a settings/version matrix (parallel-slot count, batch size, LM Studio
  version) this session did not have budget to sweep.
- A validated concurrency-level-vs-rate curve — QA's own 2/10 → 5/10 figures are the best
  available estimate; this session's own counts are too small and methodologically different
  (burst vs. sustained load) to add a curve of its own.
- Whether this dev box's specific LM Studio version/server settings match what demo-day hardware
  will run — not checked.

## Mitigation options assessed

**A. Fully serialize chat-completion calls to LM Studio (global semaphore = 1).** Eliminates the
bug (never observed at concurrency=1, in QA's report or this session's baseline). **Rejected as
the shipped mitigation** — it defeats the ~50-participant concurrency the plan is sized for.
QA's own Run B table already shows agent-turn latency climbing from ~4s (c=1) to p50 18.1s / p95
30.3s at c=50 under whatever partial batching LM Studio does today; strict serialization turns
that into roughly the *sum* of all in-flight turns, which for a live demo audience is a much
worse, and much more visible, failure mode than an occasional wrong-language reply.

**B. Bound concurrency via a semaphore at some N > 1.** Reduces (not eliminates, per finding 3
above — a fixed small N does not stop two different-language requests from occasionally landing
in the same window) the exposure. Genuine, measurable throughput/latency cost — quantifiable by
re-running QA's own Run B sweep with the semaphore in place. **I cannot recommend a specific N**
from this session's data: no clean dose-response relationship was established, so sizing N to a
target residual rate is unsupported guesswork right now. Worth pursuing as a complementary lever,
sized by its own measurement, not as the sole fix.

**C. Post-hoc language classifier + bounded retry on the `post_message.text` argument.** Directly
targets the observed failure without sacrificing baseline throughput. Costs: latency only on the
minority of turns that fail (asymmetric, acceptable); needs a real classifier (this session's ad
hoc regex heuristic was adequate for diagnosis, not production — recommend a small local library,
e.g. `langid`/`fasttext`-style detection, no extra LLM call, fast enough not to matter next to a
multi-second chat completion); needs a bounded retry count and an explicit terminal fallback
(what happens if 2 retries are still wrong — still post it, rather than leave the customer with
no reply, given DEF-3 already shows this exact `_run_turn`/`_drive_or_fault` seam is fragile
under failure paths). The retry should be forced through a low-concurrency/serialized path
specifically (pairs naturally with B) so it isn't re-exposed to the same co-batched condition
that produced the failure.

**D. Strengthen the language signal's prompt salience.** The language cue is currently a single
JSON key (`"language":"en"`) at the tail of a long, mostly-prompt-identical-across-participants
message — a weak, easy-to-miss signal even before concurrency enters the picture, which is
consistent with concurrency-induced decoding noise being "what tips a close call" rather than
creating one from nothing. Interpolating the language directly and redundantly into the
`systemPrompt` text itself (e.g. "Respond in {language_name} for this entire reply.") rather than
relying solely on a generic "check the CONTEXT block" instruction raises the decision margin
without touching LM Studio's serving internals at all. `systemPrompt` is create-only per version
(`proof_defs.py`'s own docstring) but a `v8` bump is cheap and precedented — `v3`/`v4`/`v5`/`v7`
were each exactly this shape of change. **No throughput cost.** Does not fix the underlying
engine-level nondeterminism (if that's what it is) but plausibly reduces how often it flips the
outcome — testable cheaply with the same direct-LM-Studio harness used in this session, before
touching the application at all.

**E. Swap serving engine or model.** Rejected for the first-live-demo timeframe: out of scope
for a demo-readiness fix; the local-model reliability track is already separately owned
(K-060, `falkor-chat/docs/BACKLOG.md`).

## Recommendation

**Do not accept DEF-6 as residual risk for the first live, audience-facing demo as-is.** It
reproduces at LM Studio's own serving layer (confirmed this session, not merely QA's hypothesis),
hits the default language most attendees will pick, and even QA's own literal-minimum-concurrency
condition (three participants) shows ~20% — a rate a presenter can plausibly hit in the first few
minutes of a live run. This mirrors the K-056/AC-10 precedent: gate the live-demo milestone
specifically, not the current docs-closeout milestone.

**Minimum viable mitigation, in priority order:**
1. **D (strengthen `systemPrompt`'s language salience, `v8` bump)** — cheap (hours), low risk, no
   architecture change, no throughput cost. Validate with the same direct-LM-Studio harness
   pattern used here (no need to stand up the full app to get a first read) before deciding
   whether more is needed.
2. **C (classifier + bounded retry on `post_message.text`, retry forced serialized)** as the
   safety net — moderate effort (roughly half a day to a day), moderate risk because it touches
   the tool-loop/turn-completion contract that DEF-3 already found fragile (`_run_turn`/
   `_drive_or_fault`), so route it through whoever owns that seam with DEF-3's finding in hand,
   not as an isolated add.
3. **B (bounded semaphore)** as a complementary throughput/exposure lever, sized by its own
   Run-B-style latency sweep — not a substitute for D/C, and not confidently sizeable from this
   session's data alone.

**Do not ship A (full serialization).**

**Required evaluation before calling this closed for the demo:** re-run QA's own
`docs/test-plans/salesperson-ui.md` TP-007 literal-concurrency-variant protocol (n=10 trials,
3-way concurrent, `en`/`pt-BR`/`es`, 5 turns each) after each mitigation increment, `en` adherence
rate as the primary metric, Wilson score interval per this lab's own small-n convention. At the
~20% baseline rate, n=10 alone cannot distinguish "fixed" from "still ~10-20%" — plan for n=20-30
post-mitigation trials before treating the residual as acceptable, and report each trial's
individual outcome, not only the aggregate pass count (a 2-of-10-style partial-failure pattern
is exactly the read that matters here).

## Risks & open questions

- The exact LM Studio/llama.cpp internal mechanism is still unestablished — if D alone doesn't
  move the measured rate, that's itself evidence the cause sits deeper in the engine than prompt
  salience can reach, and B/C carry more of the load than estimated here.
- This session's "all-English batch never drifted" finding (§3) rests on n=8 — worth a
  larger confirmatory run before leaning on it to justify a language-aware (rather than
  general-purpose) concurrency control.
- Demo-day hardware/LM-Studio-version parity with this dev box was not checked.
