# Typed Decision Models (Jev/Laya-style "System 1") — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-10-30

## Intent

Benefit from the new class of non-generative "System 1" decision models (TypeSafe's Jev; its
open-weights counterpart Laya) so that falkor-chat's yes/no and pick-one decisions become
**more reliable, faster, and cheaper**, all three together. Here "more reliable" means a typed
answer that can never be unparseable, at no loss in quality (FR-1, FR-10); "cheaper" means less
local compute per decision (FR-9). Laya runs locally in LM Studio today and
is the experimentation vehicle.

## Problem & current state

- falkor-chat makes structured decisions with a general chat model. The workflow guard judge
  ("is enough known to advance?") asks `qwen/qwen3-4b-2507` for a `{decision, rationale}` text
  verdict. An unparseable verdict falls back to *suspend*. That is safe, but it is a failure mode
  of free text. Stored model-bench runs show a median of ~650–810 ms per decision on this hardware.
- A decision model answers typed questions (predicate / choice / score) with probabilities in one
  forward pass, and it cannot produce malformed output. Laya measured locally through LM Studio's
  `/v1/decisions`: ~20 ms warm and deterministic across repeats.
- **Off-the-shelf Laya is not a drop-in replacement.** An informal run (not a model-bench run) over
  the 85 guard-judge golden items, counted the way model-bench counts them: Laya at a 0.5 cut-off
  falsely advanced 14/40 clear-suspend items (plus 9/15 boundary items) and falsely suspended
  12/30, vs. `qwen3-4b`'s stored 4/40 (plus 5/15 boundary) and 4/30. Laya's model card says that base
  checkpoints are near chance on custom decisions until fine-tuned and that they ship
  over-confident.
- model-bench can only measure `chat` and `embeddings` models, so a decision model cannot be
  compared against the current judge with its honesty rules.

## User stories

- As the **lab owner**, I want structured yes/no and pick-one decisions in falkor-chat answered by
  a fast local decision model, so that workflows and replies react faster and at lower compute
  cost, without becoming less reliable.
- As the **lab owner**, I want a decision model to take over only where the measurements prove it
  isn't worse, so that I can adopt it one decision point at a time without guessing.
- As the **lab owner**, I want the cases the fast model is unsure about to fall back to today's
  LLM judge, so that I get most of the speed now, before any model is fully specialized.
- As the **lab owner**, I want to build labeled examples cheaply with a frontier LLM, so that the
  new-ability decision points, which have no data today, can be trained and measured at all.

## Functional requirements

- **FR-1 — Quality parity bar (speed-up points).** A decision model may take over a speed-up decision point only when, on
  that point's labeled set, it is **not measurably worse** than the incumbent on **either** error
  type: wrongly advancing/accepting (false positive) **and** wrongly suspending/rejecting (false
  negative). "Measurably" uses the benchmark's own resolving power at the evaluation set's size
  (`docs/requirements/small-model-benchmarking.md`).
- **FR-2 — Speed/cost win.** At parity (FR-1), the decision point must be **at least 2× faster on
  average across the whole mix of decisions** (hand-offs included, FR-3) than the incumbent
  LLM-only judge, both measured in the same session on the same hardware.
- **FR-3 — Hand-off of uncertain cases is allowed.** At a decision point, the decision model may
  decide the cases it is confident about and hand the rest to the existing LLM judge. FR-1's
  parity bar applies to the **combined** outcome, and FR-2's speed/cost win is judged over the
  whole mix of decisions, not only the fast ones. The combined configuration (decision model plus
  hand-off target) counts as **one measured candidate** in model-bench. How often each decision point hands off must be
  observable.
- **FR-4 — Specialization is in scope.** A local decision model may be trained/recalibrated on
  this lab's own labeled decisions, one decision point at a time. Each decision point therefore
  needs its own labeled examples. A specialized model is always judged on labeled examples that
  were **not** used to train it, calibrate it, or tune its hand-off cut-off (FR-3).
- **FR-5 — Labels come from an LLM, unreviewed.** This **amends**
  `docs/requirements/small-model-benchmarking.md` FR-19 for packs built under this feature only. Labeled examples for training **and** for
  evaluation may be produced by an LLM with no human review. Consequence, accepted by the
  stakeholder: for a decision point evaluated this way, FR-1's parity means **agreement with the
  labeling LLM**, not verified correctness, and every result must say so. The labeling LLM must be
  a **different model from the incumbent judge** at that decision point, so the incumbent is never
  graded against itself. This includes the guard judge: its training and evaluation examples
  also come from the frontier labeler (FR-6), and its parity no longer depends on a manually
  reviewed set.
- **FR-6 — The labeler is a hosted frontier model.** Labeling is a one-off build cost, not a
  per-decision runtime cost.
- **FR-7 — model-bench first.** model-bench must be able to measure decision models (under its
  existing honesty rules: intervals, environment fingerprint, within-role comparison), and a
  decision point's bar (FR-1 parity, or FR-10 for a new ability) is established there **before** a
  decision model is used at that point in falkor-chat.
- **FR-8 — Synthetic inputs only.** Example inputs for training and evaluation are **generated**
  (by the frontier labeler) per decision point. No real falkor-chat thread or trace leaves the
  machine for labeling. (AC-7)
- **FR-9 — Local decision models first; hosted ones later.** This effort's runtime candidates are
  local decision models only (Laya and similar, served by LM Studio). "Cheaper" means less local
  compute per decision. A hosted decision model (e.g. TypeSafe's Jev) is **deferred, not
  excluded**: nothing here may prevent comparing one later under the same bars (FR-1, FR-2), with
  per-call price then counted in "cheaper".
- **FR-10 — New-ability bar.** A new-ability decision point (scope table) has no incumbent, so
  FR-1's parity cannot apply. It is adopted only when it reaches **at least 90% agreement with the
  frontier labeler within each true answer** on its evaluation set: at least 90% of the items
  whose true answer is "yes" and at least 90% of those whose true answer is "no" (for pick-one,
  at least 90% within each type). This prevents a model from passing by always giving the most
  common answer. Hand-off of uncertain cases to a local LLM is allowed exactly as in FR-3, and the
  90% applies to the combined result. Its average decision time, hand-offs included, must be
  **350 ms or less**.
- **FR-11 — Minimum evaluation-set size.** A parity or new-ability result counts only if its
  evaluation set has **at least 200 items** for that decision point as a whole (not per evidence
  kind or per answer).
- **FR-12 — Joining in unmentioned supersedes the responder's locked trigger.** Today the
  responder replies only when @mentioned (the locked trigger in `falkor-chat/server/falkorchat/responder.py`).
  This new ability changes that rule. The rule that an agent-authored message never triggers a
  reply (loop guard) is **unchanged**, and a message that @mentions the agent still always gets a
  reply, exactly as today. The decision model decides only for messages that do not @mention it.
- **FR-13 — Guard parity per evidence kind.** The guard judge's parity (FR-1) must hold on each
  evidence kind separately ("understanding" summaries and "recent turns"), not only on the mix.
- **FR-14 — Probability-only guard trace is acceptable.** A guard decision's trace entry may carry
  the decision probability and which path decided (decision model or LLM hand-off) instead of a
  written rationale.
- **FR-15 — Human-checked side check for the guard judge.** Every guard-judge result also reports
  how the candidate does on the 85 existing human-curated items, as a side check. It does not
  affect pass/fail (FR-5), but it makes drift from human judgment visible.

## Scope — decision points

| Decision point | Kind | Question shape | Incumbent today | Hand-off target | Uses the decision | Labeled data today |
|---|---|---|---|---|---|---|
| Workflow guard judge | **speed-up** of an existing model decision | yes/no: is enough known to advance? | LLM `qwen/qwen3-4b-2507` | that LLM | workflow executor (advance vs. suspend) | 85 human-curated items (side check only, FR-5, FR-15) |
| Message-type classification | **new ability** | pick one: order / question / complaint / chit-chat … | none | a local LLM | nothing yet: recorded as a label on the message | none |
| Responder: join in unmentioned | **new ability** | yes/no: should the agent reply although not @mentioned? | none: today a fixed rule replies only when @mentioned | a local LLM | the responder | none |

Speed-up points are governed by FR-1–FR-3. New-ability points have no incumbent to match, so FR-10
governs them. "Message-type classification" is unrelated to the existing `triage@v1` workflow.

## Out of scope

- **Auto-switching a production model on a benchmark result.** Inherited from
  `docs/requirements/small-model-benchmarking.md`. Acting on a result stays a human decision.
- **Hosted decision models at runtime, for now.** Deferred, not excluded (FR-9).
- **Deciding entity matches.** Matches stay human-confirmed and never auto-linked
  (`falkor-chat/docs/requirements/document-ingestion.md` FR-9/FR-10).
- **Acting on a message's type.** Classification only records a label; nothing routes or behaves
  differently by type yet.
- **Human review of labels**, for training or evaluation (FR-5).
- **Sending real falkor-chat data off the machine** (FR-8).
- **Free-text generation tasks** (answer writing, query generation, retrieval). Decision models do
  not generate text.

## Acceptance criteria

- **AC-1** — Given the guard judge's labeled set, when the candidate and `qwen/qwen3-4b-2507` are
  compared, then neither false-advance rate nor false-suspend rate is distinguishably worse for the
  candidate, and the result is recorded with its interval (FR-1).
- **AC-2** — Given a decision point that runs with hand-off, when its decisions are reviewed,
  then the share decided by the decision model vs. handed to the LLM is reported, and the quality
  and latency figures cover both paths together (FR-3).
- **AC-3** — Given any parity or new-ability result, when it is reported, then the report
  shows that no evaluation example was used for its training, calibration, or hand-off cut-off
  tuning (FR-4).
- **AC-4** — Given any parity or new-ability result, when it is reported, then it names the
  labeler model of its evaluation set and states that the result measures agreement with that
  labeler. A set whose labeler is the incumbent's model is refused (FR-5). The labeler is a hosted
  frontier model (FR-6), and every runtime candidate measured is a local model (FR-9).
- **AC-5** — Given a local decision model loaded in LM Studio, when a person runs model-bench
  against a decision-point pack, then a stored, fingerprinted result is produced and can be
  compared with the incumbent LLM's result for that role where one exists, or checked against
  FR-10's bar for a new ability (FR-7).
- **AC-6** — Given a decision point running with hand-off, when its average decision time over
  the evaluation set is compared with the incumbent's on the same machine, then it is at most half
  (FR-2).
- **AC-7** — Given the labeled examples for any decision point, when their origin is checked, then
  every input is synthetic, with no real falkor-chat thread or trace among them (FR-8).
- **AC-8** — Given an agent-authored message, when it reaches the responder with join-in enabled,
  then no reply is produced, whatever the decision model says (FR-12).
- **AC-9** — Given FR-12 is delivered, when falkor-chat's docs describing the responder trigger are
  read, then they state the new rule and none still says "@mention only" (FR-12).
- **AC-10** — Given a guard-judge parity result, when it is reported, then it shows parity
  separately for "understanding" and "recent turns" evidence (FR-13).
- **AC-11** — Given a guard decision taken by the decision model, when the run trace is read, then
  it shows the probability and which path decided (FR-14).
- **AC-12** — Given a new-ability decision point, when its result is reported, then it shows at
  least 90% agreement with the frontier labeler and an average decision time of 350 ms or less,
  hand-offs included (FR-10).
- **AC-13** — Given any parity or new-ability result, when it is reported, then its evaluation set
  has at least 200 items, and a smaller set is refused (FR-11).
- **AC-14** — Given a message classified while the ability is enabled, when the message is read
  back, then it carries its type label (FR-10).
- **AC-15** — Given a new-ability evaluation set in which one answer dominates, when a candidate
  that always gives that answer is scored, then it fails FR-10's bar (FR-10).
- **AC-16** — Given join-in is enabled, when a human message @mentions the agent, then the agent
  replies, whatever the decision model says (FR-12).
- **AC-17** — Given any guard-judge result, when it is reported, then it includes the side check
  on the 85 human-curated items, marked as not affecting pass/fail (FR-15).

## Open questions

_(none)_

## Decision log

2026-10-30 — What is the goal: replace the LLM judge (speed/cost), make the workflow more reliable,
or both? → **More reliable, and also faster and cheaper.** (Intent)
2026-10-30 — Which decision points are in scope? → **All four:** guard judge, responder
answer-or-not, message intent triage, entity-match borderline cases. (Scope table)
2026-10-30 — What reliability bar must a decision model clear? → **No worse on either error type**
(wrongly advancing and wrongly suspending) than the incumbent, plus a speed/cost win. (FR-1, FR-2,
AC-1)
2026-10-30 — Must the decision model decide every case alone, or may it hand uncertain ones to the
LLM? → **Hand-off allowed**; parity is judged on the combined result. (FR-3, AC-2)
2026-10-30 — Is training/recalibrating a decision model on our own labeled decisions in scope? →
**Yes.** (FR-4, AC-3)
2026-10-30 — Where do labeled examples come from? → **LLM-labeled, no human review.** (FR-5)
2026-10-30 — Should the evaluation set also be LLM-labeled (FR-1 then measures agreement with the
labeling LLM, not correctness)? → **Yes, evaluation also LLM-labeled**; the labeling LLM must differ
from the incumbent judge. (FR-5, AC-4)
2026-10-30 — Which LLM produces the labels? → **A hosted frontier model.** (FR-6)
2026-10-30 — Must a candidate be measured in model-bench before falkor-chat use? → **Yes,
model-bench first.** (FR-7, AC-5)
2026-10-30 — Where do example inputs come from? → **Generated synthetically**; no real data leaves
the machine. (FR-8)
2026-10-30 — What counts as "substantially faster"? → **≥ 2× faster on average across the whole
mix**, hand-offs included. (FR-2, AC-6)
2026-10-30 — Is a hosted decision model (Jev via API) a runtime candidate? → **Start with local
models only**; keep hosted models open for a later comparison. (FR-9)
2026-10-30 — Readback of intent, FR-1–FR-9, out of scope and AC-1–AC-7 → **confirmed by the
stakeholder**; sent for independent analyst review before Ready for design.
2026-10-30 — Review found only the guard judge is a model decision today. Which points stay in
scope? → **Guard judge as a speed-up, plus two new abilities: message-type classification and
responder join-in-unmentioned.** Overturns the 2026-10-30 "all four decision points" entry; entity
matches leave scope. (Scope table, FR-10, FR-12, AC-8, AC-9)
2026-10-30 — Which parity safeguards? → **No tuning on evaluation items** and a **minimum
evaluation-set size**; no explicit tolerated margin. (FR-4, AC-3, FR-11)
2026-10-30 — Join-in: which mistake is worse? → **Both equally.** (FR-10)
2026-10-30 — Minimum evaluation-set size? → Not answered as a number. The stakeholder rejected
manual review for the guard judge too: **all labeled data, the guard's included, comes from the
frontier model.** Overturns FR-5's 2026-10-30 carve-out that kept the 85 human-curated items. (FR-5)
2026-10-30 — Is a probability-only guard trace acceptable? → **Yes.** (FR-14, AC-11)
2026-10-30 — Guard parity on each evidence kind or the mix? → **Each kind separately.** (FR-13, AC-10)
2026-10-30 — Minimum evaluation-set size? → **At least 200 items** per decision point. (FR-11)
2026-10-30 — New-ability bar? → **At least 90% agreement with the frontier labeler.** (FR-10)
2026-10-30 — May new abilities hand off uncertain cases to a local LLM? → **Yes, same as the
guard judge.** (FR-10)
2026-10-30 — What consumes message-type classification? → **Just a label for now.** (Scope table,
Out of scope, AC-14)
2026-10-30 — Latency ceiling for new abilities? → **350 ms average, hand-offs included.** (FR-10,
AC-12)
2026-10-30 — How is the 90% new-ability bar measured? → **90% within each true answer**, so
always giving the commonest answer fails. (FR-10, AC-15)
2026-10-30 — Should guard results still report the 85 human-curated items? → **Yes, as a side check
not affecting pass/fail.** (FR-15, AC-17)
2026-10-30 — Is the 200-item minimum per decision point or per slice? → **Per decision point.**
(FR-11)
2026-10-30 — With join-in on, does an @mention still always get a reply? → **Yes, always.** (FR-12,
AC-16)
2026-10-30 — Analyst review Pass 2 (`docs/reviews/typed-decision-models.md`): approve with
suggestions; suggestions folded in. Status → **Ready for design** on stakeholder confirmation.
