# Connected code graph — P1 interpreter model after a failed dev entry check

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1-35)

States only the change against `docs/plans/connected-code-graph-p1-ml.md` §4.1 (model) and §4.3
(dev entry check). Everything not named here stands.

## 1. Question and decision

The dev entry check on `qwen/qwen3-4b-2507` (Q4_K_M, context 16384) failed: unambiguous 9/12, ambiguous 1/2,
unusable 0/14 (needs >= 11/12 and 2/2). Decision: may the model be changed and the check re-run, how
often, and what if none passes.

## 2. Is the model the right lever

Mixed, and the four misses do not all point the same way (judged from miss kinds only; the result file
stores counts, not drafts, so the draft behind each miss was not read):

- **QW-47** (silent single reading, gold ambiguous) and **QW-11** (wrong reading on a direction stated
  without an example cue) are direction judgments. §4.2 already puts the burden on the model's
  reading of cues; a 4B Q4 model failing both is model-level, with the prompt frozen.
- **QW-13** (ambiguous, `unknown-name`, 0 options) and **QW-7** (`several-symbols`, 2 options) are
  over-asks on the plan's own named resolver tests. The resolver is deterministic, so the cause is
  either the draft (wrong `kindHint`, spans split across groups, or a name that is not a substring-resolvable
  form) or a resolver defect. The named resolver tests are unit tests over fixtures and should pass; I did not run them.
  Resolver defects cannot be told from model drafts without the drafts.

So a model swap is a legitimate, protocol-clean first lever for the direction misses and a weak test of
the span misses. The discriminator is cross-model agreement (§3), not argument.

**Protocol cleanliness.**
- A model swap changes nothing frozen in §4.2: prompt, grammar, sampler, resolver, and dev items stay
  byte-identical. §4.1 declares the model perishable and the gate binds what the endpoint reports.
  It is not dev tuning.
- Re-running the entry check on another model is not tuning either, because the entry check is a
  gate on a model, not a measurement for the claim. The claim is the held-out run, once, on
  items no dev try touched, so selecting a model on dev cannot inflate it.
- The cost is a weaker gate. Estimate (binomial, not measured): a model whose true unambiguous rate is 0.85 and
  ambiguous rate 0.70 passes this check about 0.44 x 0.70 = 31% of the time, and about 67% over three
  tries. A pass is therefore a smoke test, not evidence of 92%. Report it so.
- A change to the prompt, pack, cues or resolver rules after seeing these misses is tuning, and is
  barred. A resolver change is allowed only as a defect fix against §4.2 as written, found by
  a fixture test, never against a dev item.

## 3. Pre-stated tries and stop rule (commit before running)

At most three further models, in this order. Qwen3-4b-2507 is try 0; four dev runs in all.

1. `mistralai_ministral-3-3b-instruct-2512`
2. `qwen2.5-3b-instruct`
3. `nvidia/nemotron-3-nano-4b`

Rules:
- Each try loads exactly one model with `lms load <id> -c 16384 -y` (one resident at a time). It
  confirms by fingerprint that the bound model is the intended one and its loaded context is 16384.
- Each try is one dev run, repeat 1, prompt/grammar/sampler/resolver hashes equal to try 0's.
- **Stop at the first model that meets the entry check.** It is then fixed: the record's `interpreter`,
  the canary digest, and `RECOMMENDED_MODEL` / §4.1 are updated to it (coder, one change), and the
  held-out run proceeds as §4.3.
- No re-run of a model, no second list, no temperature or `max_tokens` change per model.
- A try that returns `unusable` on more than 3/14 (template 400, thinking text, truncation) is
  counted as failed, not retried with changed parameters. A 400 is recorded with its status.
- **Design-fault trigger.** If the same qid misses in the same way on at least 3 of the 4 runs,
  stop and return to `architect` with the qid and the drafts: it is a resolver, pack or cue
  question, not a model one. Those dev items are then spent for entry-check purposes, and any
  fix needs a fresh dev set before a re-check.
- Every try, passed or failed, is listed in the P1-35 record (`devAttempts`): model fingerprint and counts.

## 4. Candidates

The evidence is `model-bench` `nlq-structured-query` (n=34 effective, floor 17.6 pp, MDD80 22.3 pp:
these models are not distinguishable from each other there), a span-and-structure extraction task, not AC-24.

| # | Model | Evidence | Risks |
|---|---|---|---|
| 1 | `mistralai_ministral-3-3b-instruct-2512` (Q8_0, ~3.0 GB) | 32/32 [0.893, 1.000], different family from qwen so errors should be less correlated; documented multilingual including Portuguese | Same weights as the rejected `mistralai/ministral-3-3b`. That template rejects a second system message or a repeated role. The interpreter sends exactly one system plus one user, so the 400 should not fire; the first call confirms it. p95 ~2.7 s. |
| 2 | `qwen2.5-3b-instruct` (Q8_0) | 30/30 [0.886, 1.000], p95 1.3 s | Smaller denominator (4 items excluded); Portuguese and subtle direction cues unmeasured; weaker instruction following than the 2507 line is plausible. |
| 3 | `nvidia/nemotron-3-nano-4b` (Q4_K_M) | 30/31 [0.838, 0.994] | p95 8.5 s suggests reasoning output that `max_tokens` 384 can truncate, so a high unusable rate; English-centred, Portuguese unmeasured. Before its dev run, one `fx-*` (non-dev) probe checks that a grammar-constrained reply carries no thinking text. |

Not selected, with the deciding reason: `google/gemma-4-e2b` (16/17, small n, 6.5 s p95 and the
always-advance behaviour seen in the guard pack), `smollm3-3b` and `qwen2.5-coder-3b-instruct` (0.85 and 0.82 on the
nlq pack), `llama-3.2-3b-instruct`, `gemma-3-4b` (n=3), `stablelm-zephyr-3b` (2/9), the 9B
uncensored/distilled variants and `bonsai-27b` (24 s p95, 8/8 only), `jev-style-0.8b-decision-v3` and `laya` (not measured on a
structured-extraction pack; nothing to rank on).

Grammar: LM Studio documents that models under 7B are not all capable of strict `json_schema`; a model
whose first reply is schema-invalid shows up as `unusable` and counts as failed (§3).

## 5. AC-24 text and a "no model passes" outcome

AC-24 (`docs/requirements/connected-code-graph.md`): at least 92% of about 100 EN/PT questions
understood correctly, every ambiguous one answered with its readings; "provisional target". It applies to
"the model that actually interprets that codebase's questions ... so a local-only codebase needs a local model that
reaches it; if none does, a lower local target is set once the first phase has measured it."

A "no model meets the entry check" outcome is therefore a legitimate P1 result:
- AC-24 is reported **not met** for the local model, with the dev counts for all four tries and the
  design-fault diagnosis if triggered.
- The held-out set stays unspent (the plan forbids a held-out run after a failed entry check), so no
  contamination. The lower local target is set from a P2 measurement, a stakeholder decision, not set here.
  Note the requirement says "measured", and the dev check (14 items, one draw) is not that
  measurement. P2 owns running the held-out once with the best-available model.
- No further tries, no loop: the stop rule in §3 ends the P1 search.

## 6. Evaluation design

- Metric: the §4.3 entry thresholds, unchanged. Data: the 14 dev items. Acceptance: >= 11/12 and 2/2.
- Report unambiguous k/12 and ambiguous k/2 with Wilson intervals, over-asking and unusable counts,
  and the qids missed, per try. Intervals are wide (9/12 is [0.468, 0.911]); no ranking of tries is claimed.
- Re-record the canary digest at dev time for the model that passes (the digest is per model).

## 7. Risks and open questions

- The result file stores no drafts. For the design-fault trigger the harness needs to store the draft
  (spans, kindHint, groups) per dev item; this is a coder change to the harness only, and reading drafts
  for diagnosis is not tuning as long as §2's bar holds.
- Model capability and quantization facts here are from this lab's own packs (September 2026) and the catalog; they are perishable.
- Portuguese is unmeasured for candidates 2 and 3; held-out PT items will be the first real read.
