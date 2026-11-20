# Anthropic API adoption — Method Note (models, hosted-arm comparison, labelling, token bounds)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — · **Extends:** `docs/plans/anthropic-api-adoption.md` · **Last updated:** 2026-11-19

States only what changes or is decided against the plan (§3.2, §3.5, §3.6.5, R1). Requirements: `docs/requirements/anthropic-api-adoption.md` (FR-5, FR-6, FR-7, FR-12, FR-14). Provider behaviour is taken from the plan's 2026-11-19 docs fetch, no live call has been made; every claim resting on it is tagged **[U1-n]**, the probe in §5 that confirms it. Costs and token counts are **estimates**, labelled so, until U7's first measured runs replace them.

## 1. Decision served

The architect needs, before U7 and U9: the model pair and settings, what a hosted arm scored at provider-default sampling can be said next to temperature-0 local arms, the generator/labeller design, and the token bounds to pass. Method outcome: **pair confirmed; effort becomes one setting per model, not per pack; the hosted arm is a replicated, distribution-aware row, never a local-equivalent one; the generator is the large scored arm (stakeholder decision), so every Opus row on a decision pack is a bounded-bias row with a mandatory human check; bounds are sized by the tables in §4.**

## 2. Findings from the real system

- Every bench pack pins `temperature: 0.0` and `maxTokens` 256 (guard-judge, guard, join-in, message-type), 512 (chat-responder, nlq), 1024 (tool-caller). Scoring is deterministic: grounding containment (no judge), exact-match, label agreement, scripted tool turns. So there is **no LLM-as-judge** in the bench; affinity risk lives only in how the decision packs' labels were made.
- The decision packs already exist and are the eval data the hosted arms will be scored on: `guard-judge-synthetic` (317 eval + 85 human side-check), `join-in-judge-synthetic` (200), `message-type-classifier-synthetic` (290). Their recorded provenance: **generator `claude-opus-4-8`, labeller `claude-sonnet-5-5`**, both via Claude Code subagents (`packs/*/PROVENANCE.md`, `decision-models/data/*/generation.json`).
- Measured vote noise of that labeller is small: pair agreement 0.970-1.000 per class, 3-0 share 0.955-1.000 (`diagnostics.json`, all three points). Intent-label agreement 0.963 (guard), 0.986 (join-in).
- Measured subagent runs give real output sizes (adaptive thinking, effort high/medium): a 50-item label run outputs 2.6-2.8k tokens on average (max 4.8k); a 25-seed generation run 3.1k (message type) to 14.1k (guard) on average, max 18.4k. Their totals (23k-48k per run) are dominated by Claude Code's context floor, which the API path does not pay.
- `rank_report` keeps one run per `modelKey`, the newest (`modelbench/cli.py:_select_rank_arms`), and `DuplicateModelInReport` raises otherwise; `PairedOutcomes.from_units` raises on a repeated unit id.
- Existing 2026-11-01 decision: a safeguards stop on the Opus generator halts the campaign and the stakeholder decides; `stop_reason: "refusal"` is class `safeguardStop`, environmental HALT (`docs/plans/typed-decision-models6.md` §3.4 table, `audit.py`).

## 3. Method

### 3.1 Model pair and settings (plan §3.2)

**Confirmed: small `claude-haiku-5-5`, large `claude-opus-5-5`.** Haiku is the like-for-like tier for the 3-4B local models; Opus is the ceiling arm; `claude-sonnet-5-5` is the labeller of every decision pack and must not be a scored arm (it would score its own labels). Fable 5.2 has no matching need. This is an engineering choice, not a measurement; the decision whether a hosted tier is "worth using" comes from the rows, not from this note.

Settings, **one per model, identical across all seven packs and identical to the falkor-chat overlay**:

| Model | thinking | effort | max_tokens sent |
|---|---|---|---|
| `claude-haiku-5-5` | `disabled` **[U1-4]** | n/a | pack `maxTokens` unchanged (same budget semantics as the non-thinking local arms) |
| `claude-opus-5-5` | adaptive (cannot be disabled) | `medium` **[U1-5]** | pack `maxTokens` + 4,096 thinking allowance (tool-caller: 1,024 + 4,096) |

- **Why one setting per model.** The overlay's per-model settings are per model (`plan §3.4`). A per-pack effort would score a configuration no workspace can run. The bench must score the deployed configuration.
- **Why `medium`.** It is the arm's declared identity, "Opus 5.5 at medium effort", not a tuned value. It is fixed now, before any result, and not changed after seeing scores (that would be selection on the eval). A claim of the form "even Opus cannot do X" needs a separate, labelled `high`-effort run of that one pack; it is not part of the 28 runs.
- **Truncation is an outcome, not a wrong answer.** `stop_reason: max_tokens` is stored per item as `truncated`, counted and printed per run. If truncated calls exceed 2% of a run's calls, the run is reported "budget-limited", not ranked, and rerun with a larger recorded allowance. The cause would otherwise be the cap, not the model. The hosted `max_tokens` is recorded in the fingerprint (`maxTokens`, already in the plan).
- **Opus on `tool-caller` depends on R2 [U1-3].** If an Opus tool loop requires returning thinking blocks, the adapter that drops them runs a degraded configuration. In that case the Opus `tool-caller` run is **not run** and the gap is printed; a silently degraded arm is never scored. Haiku (thinking disabled) is unaffected.
- **Rejected:** per-pack effort (not deployable); `high` everywhere (labeller convention, adds ~2x output cost to latency-insensitive classification with no evidence it helps); Sonnet as the large arm (labeller overlap).

### 3.2 Comparing a provider-default hosted arm with temperature-0 local arms (plan §3.5)

Facts the claims rest on: local arms are deterministic within a load; the hosted arm cannot be set below temperature 1 and top-level sampling stays at provider defaults **[U1-1, U1-2]**; each item gets one draw per run.

**Can be claimed**
1. *Marginal single-run quality at default sampling.* For a k/n hosted rate, the draws are independent Bernoulli(p_i) per item, so the Wilson interval on k/n is valid for the item-averaged success probability (variance ≤ the binomial's). This is the quantity a deployed user experiences.
2. *Paired difference against a local arm on the same items*, by the existing exact paired test, as a statement about that run: discordance mixes model difference with sampling noise, which the n already carries.
3. *Run-to-run stability* from the two replicates (below), printed as item-level disagreement `k/n` with Wilson.
4. *Hosted-vs-hosted* (Haiku vs Opus): same regime, same caveats.

**Cannot be claimed**
- Reproducibility, or "the model's temperature-0 behaviour". Hosted rows are marked and footnoted (plan §3.5.5), and no hosted row enters a determinism or load-replicate claim.
- That a gap is model quality alone: sampling regime, thinking (Opus), token allowance and network differ from the local arms.
- Any latency comparison. Latency includes network and provider queueing, and the latency percentile/withholding machinery assumes local load behaviour: **do not print latency headlines or floors for hosted rows**; show raw per-call time as descriptive only. (Plan §3.5.5 says only "latency includes network"; strengthen it.)
- Any parity or new-ability bar (FR-1/FR-10): a hosted arm is never an incumbent or a candidate; the bars are defined between deterministic local arms.
- Accuracy on decision packs. Scores there are **agreement with `claude-sonnet-5-5` majority labels** on text a Claude model wrote, and for Opus the text was written by the arm itself (§3.3). A Claude arm shares errors and style with the label source, so it is biased upward relative to the local arms, and Opus cannot be shown above the labeller's own consistency. Rows read "agreement with a same-family labeller; generator = this arm" and §3.3 states the claims allowed on them.

**Two replicates per hosted (arm, pack): 28 runs, not 14.** Mirrors the local two-load rule (typed-decision-models-ml §Q6). A hosted-vs-local difference is claimed only if **each replicate** shows it on its own paired test (a conjunction only makes claims harder). Never pool the replicates into 2n; the analysis unit stays the item (the repeated measures belong to the item). Cost: Haiku is cents; the Opus second replicate is the main cost (§4). The tool-caller pack has 12 scripts, so its hosted rows carry a very wide interval at any replicate count; print it, claim nothing narrower.

### 3.3 decision-models generator/labeller (plan §3.6.5)

- **Labeller: `claude-sonnet-5-5`, confirmed.** The reference for every existing pack, so labels stay one source across versions; not a scored arm (labeller-as-arm would measure the arm against its own distribution, the worst form of leakage here). `effort: high`, adaptive thinking, matching the existing labeller's recorded `high`; temperature stays provider-default (the subagent path was also unknown, `labelerTemperature: null`).
- **Generator: `claude-opus-5-5`** (stakeholder decision), which is also the large scored arm. The risk is self-scoring: the arm is scored on text it wrote, labelled by a same-family model. It is unmeasured and its sign is not known in advance, but the likely direction is upward. Settings: adaptive thinking, `effort: medium` (as the existing generator definition pins), `max_tokens` 32,000, provider-default temperature. An Opus 5.5 safeguards stop (`stop_reason: "refusal"`, plan6 F7) halts the campaign each time and the stakeholder decides, as already decided; it never counts toward the attempt cap.
- **Bounding and reporting the self-scoring bias (mandatory on every decision pack).**
  - Every Opus row on `guard-judge-synthetic`, `join-in-judge-synthetic` and `message-type-classifier-synthetic` carries the caveat "generator = this arm; labels from a same-family model" and prints per stratum.
  - **Guard:** the 85 human-curated side-check items (a different distribution from generated text, and not written or labelled by any Claude model) are the independent stratum. Report each hosted and local arm's rate on both strata with Wilson intervals. The signal is the *difference* between the Opus gap (eval minus side-check) and the Haiku and local arms' gaps. Detectable only if large: at n=85 a 7-10 pp Wilson half-width means a gap-of-gaps under about 15 pp cannot be resolved. It detects gross self-scoring, it does not remove or estimate it, and the sample is too small for stratifying by anything finer. Do not enlarge it from the generated pool (that would be Claude-labelled again); more human items only if the owner wants a finer check.
  - **Join-in and message type have no human items** (side-check 0). Their Opus rows stay "unbounded self-scoring" unless the owner hand-labels, blind to the Claude label, a random sample from each eval split: **60 per point, stratified by label (join-in 30/30, message type 12 per label)**. Report Opus, Haiku, the Sonnet labeller and local arms against the human label on that sample. Same limits (gross bias only). The sample is used for this check only, never for fitting or tuning. This is an owner cost (about 120 items) and the sole way to lift the "unbounded" label; absent it, the rows publish with the label.
  - **Claims allowed on Opus decision-pack rows:** Opus does not beat arm X (a bias that inflates Opus can only make this claim stronger); a ranking among local arms or Haiku vs local (all share the same label source; Haiku carries the same-family caveat, not the self-scoring one); descriptive rates. **Not allowed:** an absolute accuracy or "better than the local/Haiku arms" statement for Opus, unless the human-item stratum (guard) or the human sample (others) shows the same direction. **Withheld otherwise:** any statement that a hosted arm "exceeds the labeller" on these packs.
  - The bias applies only to the decision packs; the role packs (chat-responder, tool-caller, nlq-generator, guard-judge-understanding) are authored and human-verified and carry no such caveat.
- **A dataset made this way is a new dataset version and a new pack version** (`labels.*.surface: anthropic-api`). Existing v1/v2 data is neither regenerated nor mixed into: a point never mixes two generators (existing re-pin rule). The 108 stored local runs compare only on the pack version they ran; any new-version pack needs local arms re-run before a comparison.
- **Votes: keep three, fresh keys, reshuffled batches; add no prompt or order diversity.** The measured vote noise is already small (pair agreement ≥ 0.970 per class), the shared-bias risk is not reducible by votes, and altering the prompt changes `promptSha256` and adds a confound. At provider-default sampling the three calls are independent draws, which is the property a vote needs. Keep the existing drop/tie/tranche rules unchanged.
- **Effort/thinking per role: generator `medium`, labeller `high`, both adaptive** (as the existing agent definitions pin), recorded per `api-call` event along with `stop_reason` class (`end_turn`/`max_tokens`/`refusal`): `thinking` and `effort` are not sensitive.
- **max_tokens: labeller 16,000, generator 32,000** (observed maxima 4.8k and 18.4k, the latter under `claude-opus-4-8`, so >3x and 1.7x headroom). Truncation is a sizing fault and **halts** with a recorded `resolve` to raise the cap, it is never counted toward the 3-attempt drop: items that make the model think longest would otherwise be dropped non-randomly. Large-cap non-streaming acceptance **[U1-7]**.
- **Pilot acceptance** (on top of the existing guard pilot): the pilot prints pair agreement per class beside the v1 values; a class whose Wilson 95% upper bound falls below 0.970 (the lowest existing class) goes to the stakeholder as "the API labeller differs from the subagent one". Intent-label agreement prints exploratory only (it depends on the generator, which changed).
- **Sample sizes: unchanged** (§Q1.3 of `docs/plans/typed-decision-models-ml.md`: 100/cell guard, 100/label join-in, 60/label message type). The usage-window constraint that fixed them as "pilot first" is gone, but the sizes are set by the bars' detectable differences, not by spend.

### 3.4 Token-bound sizing (plan §3.5.1 `--token-bound`, §3.6.3)

**Counting basis (both modules):** tokens used = `input_tokens + cache_creation_input_tokens + cache_read_input_tokens + output_tokens`, thinking included **[U1-8]**. The wire's `prompt_tokens` mapping must sum the three input fields, or a cached prefix undercounts the bound.

**Bench**, per run, estimated from the packs (characters ÷ 3.5 ≈ tokens, tokenizer unmeasured; Opus thinking assumed 800 tokens/call mean, no measurement). Bound = 2 × projection, rounded up:

| Pack | calls | Haiku est. | Haiku bound | Opus est. | Opus bound |
|---|---|---|---|---|---|
| chat-responder-grounded-answers | 30 | 18k | 50k | 36k | 100k |
| guard-judge-understanding | 85 | 40k | 100k | 105k | 250k |
| nlq-structured-query | 40 | 46k | 100k | 73k | 160k |
| tool-caller-shop-assistant | ~150 | 340k | 700k | 450k | 1M |
| guard-judge-synthetic | 402 | 200k | 400k | 500k | 1M |
| join-in-judge-synthetic | 200 | 180k | 350k | 330k | 700k |
| message-type-classifier-synthetic | 290 | 260k | 500k | 480k | 1M |

Per replicate ≈ 1.1M (Haiku) + 2.0M (Opus) tokens; 28 runs ≈ 6M; cost ≈ $50 at the plan's fetched prices (estimate, ~$47: Opus ≈ $23 per replicate, Haiku < $1). **Procedure:** run Haiku first, per pack; replace the table's input figure with the measured one; set the Opus bound = 2 × (measured input + calls × 800 + measured Haiku-style output), and raise 800 to the measured Opus mean after its first pack. A stop writes no record (plan §3.5.3), so a too-tight bound wastes a run; 2x makes a stop signal a runaway, not routine. Every bound must also exceed one call's `max_tokens`.

**decision-models**, per `api-run` invocation (a stop exits 0 and resumes, so the bound is a review interval, not a sizing risk). Per-call estimates: label call ≈ 10k input (rubric 4-5k chars + 50 items) + 2.8k output (measured) ≈ 13k; generation call ≈ 4k input + 3k-14k output (measured under `claude-opus-4-8`) ≈ 7k-18k. Full design (150 label + 97 generation runs, per the note's counts): ≈ 2.0M + 1.3M ≈ **3.3M tokens, an order of magnitude below the subagent figure (23M-110M)**; estimate, ±2x. Bounds: **pilot 0.6M** (4 generation + 16 label calls project ≈ 0.28M); **full points 1.0M per invocation** until the pilot's measured per-role means exist, then 2 × the invocation's projection. The existing spend-halt ("a top-up round may not project above half of round 1's total") is unchanged; its both-sides sum must use the same token basis as the bound, which holds within a point because a point never mixes surfaces.

## 4. Evaluation design (how to prove the method works)

- **Bench validity (U6, offline, stub):** a hosted fingerprint with a forbidden local field is refused; two runs of one hosted arm are both retained and both printed (see §6 M1); a `max_tokens` stop appears as `truncated`; the token total counts all four usage fields (stub returns non-zero cache fields).
- **Hosted sweep acceptance (U7):** per run, truncated ≤ 2% (else "budget-limited", rerun); both replicates stored; every hosted row carries `[hosted]`, the "same-family labeller" caveat on the three decision packs, and no latency headline; a difference vs a local arm printed only with both replicates' paired results. No pass/fail gate (the bench has none).
- **Campaign acceptance (U9):** pilot prints pair agreement per class vs v1 (rule above); truncation count is 0 or the run halted; `safeguardStop` halts, never counts; token ledger total reconciles to the sum of `api-call` events; the bound stop leaves the ledger resumable.

## 5. Provider behaviour this note depends on (add to the U1 probe)

| ID | Claim relied on | If false |
|---|---|---|
| U1-1 | `temperature < 1` is rejected for Haiku 5.5 and Opus 5.5 (omission accepted) | If Haiku accepts 0, send 0.0 to Haiku (closer to local; Opus stays default) and re-read §3.2 |
| U1-2 | Default sampling is stochastic: the same prompt twice gives different text on at least one of several items | If effectively deterministic, replicates collapse to one run |
| U1-3 | Opus tool loop with thinking blocks dropped works (or is rejected, clearly) | If it must pass blocks back: Opus `tool-caller` not run (§3.1) |
| U1-4 | Haiku `thinking:{"type":"disabled"}` accepted, also with tools | Haiku scored at its default thinking and recorded as such |
| U1-5 | `output_config.effort: "medium"` accepted for Opus 5.5; thinking counts against `max_tokens` | Use the nearest accepted value, record it; budget rule in §3.1 unchanged |
| U1-6 | A `refusal` stop on an Opus 5.5 generator prompt arrives as `stop_reason: "refusal"` in a non-error 200 response (one deliberately borderline guard near-miss prompt) | Classify by the observed shape; the halt rule in §3.3 is unchanged |
| U1-7 | A non-streaming call with `max_tokens` 32,000 is accepted | Lower the generator cap to the largest accepted and keep the truncation halt; or the wire gains streaming |
| U1-8 | `usage.output_tokens` includes thinking tokens; cache fields exist in `usage` | Token bound undercounts: add a conservative multiplier until fixed |

## 6. Methodology problems in the plan

**Blocker**
- **B1 — `refusal` is classified as truncation (plan §3.6.2).** `audit_api_call` maps "not `end_turn`" to the COUNT-class truncation refusal. An Opus safeguards stop (`stop_reason: "refusal"`) is, by the 2026-11-01 decision and the plan6 table, an environmental HALT (`safeguardStop`) for the stakeholder; counting it burns attempts and drops items. And `max_tokens` is a sizing HALT here (§3.3), not a counted void. Fix: three-way map (`end_turn` clean / `max_tokens` halt / `refusal` halt `safeguardStop`) before U8.

**Major**
- **M1 — replicates cannot be stored or ranked as planned (plan §3.5.5, U7 "14 runs").** `rank_report` dedupes to the newest run per `modelKey` and raises on a duplicate. Two replicates of one hosted arm either silently drop one or raise. Fix: the plan must give the replicate pair a representation (the existing replicate-group mechanism, or a replicate suffix on the hosted `modelKey`), and U7 becomes 28 runs.
- **M2 — per-pack effort (plan §3.2, R1).** See §3.1: replace with one setting per model matching the overlay.
- **M3 — `max_tokens` for Opus (plan §3.5.2).** The pack's 256/512 cannot hold adaptive thinking; the runner must accept a hosted override and a `truncated` outcome (§3.1), else Opus scores measure the cap.
- **M4 — U9 produces data not comparable with the stored local runs.** The 108 local runs are on v1/v2 packs generated by `opus-4-8`. API-generated data is a new version; the plan should state that local arms are re-run on it before any comparison (§3.3).
- **M5 — usage mapping and bound basis undefined (plan §3.1 `parse_response`).** `prompt_tokens` must be input + cache creation + cache read; the bound counts input + output including thinking (§3.4).

**Minor**
- **m1 — "no scored arm labelled its own eval data" (plan §3.6.5)** is true but incomplete: generator = scored arm (Opus) and same-family labels bias Claude arms on the decision packs (§3.2, §3.3). Add the caveat text to the report footnote and implement the per-stratum print.
- **m2 — the `api-call` ledger event** should also carry `thinking`, `effort` and the stop-reason class (§3.3); none is sensitive.
- **m3 — latency footnote (plan §3.5.5)** should become "no latency claim", not "includes network" (§3.2).
- **m4 — report strata.** `guard-judge-synthetic` mixes 317 eval and 85 side-check items: print per stratum for hosted arms too (§3.2).

## 7. Risks and open questions

- Effort `medium` and the 800-token thinking figure are untested guesses; the first measured packs may justify a different single setting, changed once, recorded, before the Opus sweep and never after seeing scores.
- Self-scoring and family bias on the decision packs are bounded to gross effects, not measured: the guard side-check has n=85, and join-in and message type have no human items unless the owner supplies the 60-item samples (§3.3).
- Opus generator safeguards stops on near-miss and escalation content are likely to recur; each halts and needs a stakeholder decision (existing rule), which the pilot will quantify.
- Prices and model ids are perishable (plan's fetch, 2026-11-19); re-check before U7 and U9.
