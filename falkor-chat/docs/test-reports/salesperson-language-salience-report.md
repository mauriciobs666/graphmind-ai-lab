# Salesperson `systemPrompt` language salience — Mitigation D for K-065/DEF-6 — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-065

## Summary

**What was tested:** whether Mitigation D — the `v8` bump of `SALESPERSON_DEF`
(`falkor-chat/server/falkorchat/proof_defs.py`, commit `999141f`; statically reviewed `c420ab6`,
approve with suggestions, no blockers; doc-accuracy catch-up `fe3cfe3`) adding a redundant,
more emphatic language-salience sentence to `systemPrompt` — moves the measured wrong-language
rate for K-065/DEF-6 (an `en`-configured `salesperson` storefront participant occasionally
getting a fully-formed Spanish reply under concurrent load). This is a live, black-box re-run of
QA's own `docs/test-plans/salesperson-ui.md` TP-007 "literal-concurrency variant" protocol
against the deployed `v8` config, at **n=25 trials** (the plan's own required n=20-30 range,
`docs/plans/salesperson-ui-ml.md` §"Required evaluation before calling this closed for the
demo") instead of the original n=10 — sized because at the ~20% baseline point estimate (2/10,
Wilson 95% CI 5.7-51.0%), n=10 alone cannot distinguish "fixed" from "still ~10-20%": that CI's
own width is the reason, and n=25 tightens a 0-observed-failures upper bound to ~14% (below),
materially narrower than n=10 could produce. Tested against commit `fe3cfe3` (current repo `HEAD`
is `56e2e02`, an unrelated concurrent `model-bench` docs commit landed during this pass — confirmed
`fe3cfe3` is an ancestor of `HEAD`, Mitigation D untouched).

**Test-plan decision:** no separate `docs/test-plans/` document was written for this pass. TP-007
(`docs/test-plans/salesperson-ui.md` §4) and the ml plan's own "Required evaluation" section
between them fully specify the protocol, sample-size reasoning, and required metric (Wilson
interval, per-trial reporting) — the only parameter this pass changes is `n`. Writing a new plan
document would restate those two sources rather than add anything; this report cites both
directly instead.

**Result: at n=25 trials (23 with at least one `en` reply), zero wrong-language occurrences were
observed for any of the three configured languages** — `en`, `pt-BR`, and `es` each measured
100% adherence among turns that received a reply, at every trial, with no exceptions. This is a
material drop from the recorded baseline point estimates (~20% at 3-way, QA's original pass; see
"Comparison to baseline" below) — **reported as the measured rate, not as a verdict that the
mitigation is "enough" for the live demo, which is a stakeholder call.**

**A second, separate, and more severe finding surfaced during this pass and is reported with
equal prominence**: a very high rate of turns (235/375 = 62.7% across all three languages) never
produced a reply at all — the storefront's own `turn.lastTurn == 'failed'` dead-turn signal
(the K-065-adjacent DEF-3 fix, `4cebd96`, confirmed **working correctly** — it fired accurately
on every genuine provider failure observed) fired because LM Studio's own serving layer returned
hard `400`/`500` errors (`"terminated"`, `"Channel Error"`, generic `Internal Server Error`) at a
rate far higher than this feature's own prior live-testing history recorded. This is a distinct
mechanism from DEF-6 (a **wrong-language completed reply**) — it is **failed replies**, and is
**not folded into the language-adherence count above**, in either direction. See "Engine-stability
observation" below; it is very likely inflated by this session's LM Studio instance being shared
with a second, unrelated concurrent process, not fully attributable to this pass's own load.

**CPG:** not applicable — this is live behavioral verification of an LLM's language output under
the deployed `salesperson@v8` config; no code-structure or call-graph question is in scope
(matches the original TP-007 pass's own CPG line for the same reason).

## Environment

- FalkorDB up throughout; `reference` had been wiped by this coordination's own earlier offline
  `pytest` run (expected, `falkor-chat/docs/SERVER.md` §1.7) — reseeded before any live work:
  `bootstrap_schema.sh demo` → `seed_demo.sh demo` → `seed_catalog.sh` → `seed_salesperson.sh demo`,
  then verified: `verify_salesperson.sh demo` → `RESULT: OK — 2 defs in sync`, confirming
  `salesperson@v8` (including its stored `config` against the shipped `proof_defs.py` constant,
  per that script's own strongest check) landed correctly in both `reference` and `ws:demo` before
  any trial ran; `verify_catalog.sh` → `RESULT: OK` (15/15 products).
- Demo brought up via `falkor-chat/scripts/start_demo.sh` (never `start_server.sh`, R7 — `--reload`
  off); storefront preflight logged `def=salesperson@v8` at startup, confirming the trigger target.
- Live-LLM reachability: `~/.config/opencode/opencode.json`'s `lmstudio` provider still points at
  the unreachable LAN address (`192.168.0.69:1234`, the documented gotcha,
  `claude/qa-engineer/qa-testing-techniques.md`); an existing scratch copy at
  `~/.config/opencode/opencode.local.json` (already corrected to `localhost:1234`, left by an
  earlier session in this same coordination) was reused via `FALKORCHAT_OPENCODE_CONFIG`, per the
  documented technique — the shared file was not edited.
- LM Studio: `mistralai/ministral-3-3b`, `Q8_0`, `loaded_context_length: 8192`, confirmed loaded
  and serving throughout (`/api/v0/models`).
- **A second, unrelated live server process was running on this box for this pass's entire
  duration** (PID 172892, port 8200, `FALKORCHAT_WS_ID=agent-team`) — pre-existing, not started by
  this pass, not stopped by this pass (this agent does not mutate shared environment state
  unilaterally). Its relevance to this pass's own findings is in "Engine-stability observation"
  below.
- Total wall-clock for the n=25 run: ~17 minutes (22:46:25–23:03:40 local).

## Protocol (as specified — not redesigned here)

Re-ran TP-007's literal-concurrency-variant protocol: **3-way concurrent requests, one `en` + one
`pt-BR` + one `es` participant per trial, 5 scripted turns each** (same 5 scripted turns
`salesperson/scripts/load_demo.py` uses, for parity), against the live `salesperson` storefront
demo (`falkor-chat/scripts/start_demo.sh`, `ws:demo`), at **n=25 trials** (125 turns per
language, 375 total). A new scratch harness,
`/tmp/.../scratchpad/language_salience_probe.py` (not committed — see "Feedback" for a
recommendation on this), drove it: reused `load_demo.py`'s HTTP-plumbing shape but additionally
captured every turn's actual assistant reply text (`GET /shop/api/messages?since=<cursor>`,
`role == "assistant"`) and classified its language with a small regex heuristic (same
diagnostic-grade approach the `data-scientist` repro session used, `docs/plans/salesperson-ui-ml.md`
— adequate for this purpose, not proposed for production). Every trial's full result (every turn,
full reply text, dead-turn flag) was appended to disk the moment the trial finished, per the
"land incrementally" instruction for this pass.

**A defect in the harness's own first-cut classifier was found and corrected during this pass,
by hand-reading every reply.** The original heuristic let shared Ibero-Romance filler words
("por favor", "que" — present in both Spanish and Portuguese) outvote a genuine Portuguese
diacritic signal, misclassifying 5 genuinely-Portuguese replies (e.g. *"Por favor, informe o
endereço de entrega completo para que eu possa processar seu pedido corretamente"* — unambiguously
Portuguese: "informe", "endereço", "processar", "pedido") as `es`. This was caught by manually
reading every flagged non-adherent reply's raw text (required by this agent's own evidence
standard) rather than trusting the classifier's own output. **All figures in this report use the
corrected classifier (v2)**, re-run against the same captured raw text — no new live calls were
needed to fix this, since full reply text was already on disk for every turn. Every reply
classified adherent *and* every reply classified non-adherent under v2 was independently
spot-read by hand (not just the corrected ones) before being reported — see the raw-text listing
in "Results" below.

## Results

**Turn-level adherence (of turns that received a reply — dead/no-reply turns excluded from this
denominator, reported separately below), n=25 trials, 375 total turns:**

| Language | Replied turns | Adherent | Rate | Wilson 95% CI |
|---|---|---|---|---|
| `en` | 61 | 61 | 100.0% | [94.1%, 100.0%] |
| `pt-BR` | 46 | 46 | 100.0% | [92.3%, 100.0%] |
| `es` | 33 | 33 | 100.0% | [89.6%, 100.0%] |

**Trial-level ("did this trial's participant get at least one wrong-language reply" — the same
shape as QA's original "2/10 trials affected" headline figure), of trials with at least one
reply:**

| Language | Trials w/ ≥1 reply | Affected (≥1 non-adherent turn) | Rate | Wilson 95% CI |
|---|---|---|---|---|
| `en` | 23 / 25 | 0 | 0.0% | [0.0%, 14.3%] |
| `pt-BR` | 23 / 25 | 0 | 0.0% | [0.0%, 14.3%] |
| `es` | 23 / 25 | 0 | 0.0% | [0.0%, 14.3%] |

**Every individual trial's outcome** (adherent/replied per language, dead-turn count in
parentheses — a trial with 0 replied turns for a language means every turn for that participant
died before producing a reply, distinct from a language failure):

| Trial | `en` | `pt-BR` | `es` |
|---|---|---|---|
| 0 | 3/3 (dead=2) | 3/3 (dead=3) | 2/2 (dead=4) |
| 1 | 1/1 (dead=4) | 2/2 (dead=3) | 2/2 (dead=4) |
| 2 | 3/3 (dead=2) | 3/3 (dead=2) | 1/1 (dead=4) |
| 3 | 3/3 (dead=2) | 3/3 (dead=2) | 1/1 (dead=4) |
| 4 | 4/4 (dead=1) | 1/1 (dead=4) | 1/1 (dead=4) |
| 5 | 3/3 (dead=2) | 1/1 (dead=4) | 1/1 (dead=4) |
| 6 | 3/3 (dead=2) | 1/1 (dead=4) | 3/3 (dead=2) |
| 7 | 3/3 (dead=3) | 0/0 (dead=5) | 1/1 (dead=4) |
| 8 | 3/3 (dead=2) | 2/2 (dead=4) | 1/1 (dead=4) |
| 9 | 1/1 (dead=4) | 2/2 (dead=3) | 2/2 (dead=3) |
| 10 | 0/0 (dead=5) | 1/1 (dead=5) | 0/0 (dead=5) |
| 11 | 3/3 (dead=2) | 0/0 (dead=5) | 0/0 (dead=5) |
| 12 | 3/3 (dead=2) | 2/2 (dead=3) | 1/1 (dead=4) |
| 13 | 2/2 (dead=4) | 2/2 (dead=3) | 1/1 (dead=4) |
| 14 | 3/3 (dead=2) | 2/2 (dead=3) | 1/1 (dead=4) |
| 15 | 3/3 (dead=2) | 2/2 (dead=4) | 1/1 (dead=4) |
| 16 | 3/3 (dead=3) | 3/3 (dead=2) | 1/1 (dead=4) |
| 17 | 3/3 (dead=2) | 3/3 (dead=2) | 1/1 (dead=4) |
| 18 | 1/1 (dead=4) | 3/3 (dead=2) | 2/2 (dead=3) |
| 19 | 2/2 (dead=3) | 1/1 (dead=4) | 1/1 (dead=4) |
| 20 | 0/0 (dead=5) | 3/3 (dead=2) | 2/2 (dead=3) |
| 21 | 3/3 (dead=2) | 1/1 (dead=4) | 3/3 (dead=3) |
| 22 | 3/3 (dead=3) | 1/1 (dead=4) | 2/2 (dead=4) |
| 23 | 3/3 (dead=2) | 2/2 (dead=4) | 1/1 (dead=4) |
| 24 | 2/2 (dead=4) | 2/2 (dead=3) | 1/1 (dead=4) |

**No exceptions in any trial, any language.** Every `en` reply, every `pt-BR` reply, and every
`es` reply, across all 25 trials, was in the correct configured language — confirmed both by the
v2 heuristic classifier and by manual reading of every single reply's raw text (140 replies
total: 61 `en` + 46 `pt-BR` + 33 `es`).

## Comparison to baseline

| Source | Protocol | Trial-level affected rate | Wilson 95% CI |
|---|---|---|---|
| QA original (`docs/test-reports/salesperson-ui-report.md`) | 3-way, n=10, pre-mitigation | 2/10 | [5.7%, 51.0%] |
| QA original, 40-way heavy variant | 40-way, n=10, pre-mitigation (different concurrency, not directly comparable) | 5/10 | — |
| `data-scientist` repro (`docs/plans/salesperson-ui-ml.md`) | 3-way, n=5, pre-mitigation, one-shot burst not sustained load | 1/5 | [3.6%, 62.4%] |
| **This pass** | **3-way, n=25 (23 w/ reply), post-Mitigation-D** | **0/23** | **[0.0%, 14.3%]** |

The measured point estimate dropped from ~20% to 0%, and the tighter CI this sample size buys is
exactly the improvement the ml plan's own reasoning called for (n=10 could not distinguish
"fixed" from "still ~10-20%"; at n=25 with zero observed failures, the true rate's upper bound is
confidently below 15%, which starts to genuinely separate from the baseline's ~20% point
estimate, though the two CIs still technically brush at the tails — 14.3% vs the baseline's own
5.7% floor). **This is the measured number; whether it clears the bar for a live demo is a
stakeholder risk-tolerance call, not this report's to make.**

One important asymmetry worth naming plainly: **this pass measured fewer *turns* than the
original baseline per language** (61 `en` replied turns here vs. 50 in the original heavy/literal
combined) because of the engine-stability issue below — a large fraction of turns never reached
the point where language could even be assessed. The **trial-level** comparison above is the
fairer read for that reason (it doesn't penalize a trial for turns that failed to reply at all).

## Engine-stability observation (separate from the Mitigation D result — not counted toward or
against the measured rate above)

**A very high fraction of turns across the whole run never produced a reply**: 235 of 375 turns
(62.7%) show `dead_turn: true` (the storefront's `turn.lastTurn == 'failed'` signal) and no
assistant message. A further 14 turns show `dead_turn: true` **with** a reply still delivered —
the same shape the K-065-adjacent DEF-3 fix (`4cebd96`) documented as possible (a `post_message`
tool call can succeed before a later step in the same run drives it to `failed`); those 14 are
correctly counted as replied/adherent above, since this pass classifies by reply presence, not
the dead-turn flag. **Zero turns show a missing reply with `dead_turn: false`** — every genuine
failure was correctly signaled, positive evidence the DEF-3 fix's own claim (the latch fires on
every real provider failure, not just some) holds under this pass's load. Root cause of the
failures themselves, confirmed from `falkor-chat`'s own server log during this pass:

```
falkorchat.transport.ProviderCallError: lmstudio/mistralai/ministral-3-3b @ .../v1/chat/completions:
  HTTP 400 Bad Request: {"error":"terminated"}
falkorchat.transport.ProviderCallError: lmstudio/mistralai/ministral-3-3b @ .../v1/chat/completions:
  HTTP 500 Internal Server Error: <!DOCTYPE html>...<pre>Internal Server Error</pre>...
```

Tally across the full run: 371 `ProviderCallError`s against `lmstudio/mistralai/ministral-3-3b`
(this def's own configured model) and 102 against `lmstudio/qwen/qwen3-4b-2507` (**not** this
def's model — `SALESPERSON_DEF.steps[0].config.model` is `lmstudio/mistralai/ministral-3-3b` only,
confirmed by reading `proof_defs.py` directly). The `qwen` failures ("Failed to load model
\"qwen/qwen3-4b-2507\". Error: Engine protocol startup was aborted.") are cross-process noise:
this dev box had a second, unrelated live `uvicorn` process running throughout this pass (PID
172892, port 8200, `FALKORCHAT_WS_ID=agent-team`, `claude/AGENTS.md`'s `ws:agent-team` server),
whose default `step`/`agent` role model is `qwen/qwen3-4b-2507` — this pass never touched that
process or workspace, and it was already running before this pass started.

**This does not directly explain the majority `ministral` failures** (371 of them, self-inflicted
under this pass's own 3-way concurrency to the same already-loaded model) — but it is consistent
with, and plausibly a contributing factor to, an already-contended LM Studio instance being
generally less stable under additional concurrent load, which is itself directly relevant to
K-065's own root-cause framing ("LM Studio's own concurrent-request serving/decoding path").
**Recurred throughout the entire ~17-minute test window** (97 `qwen` load-failure events, 208
`"terminated"` events, spread across nearly every one of the 25 trials — not a one-off burst),
so this reads as a persistent condition for this pass's environment, not an isolated anomaly.

**A coordinator review of this pass, mid-run, separately surfaced LM Studio's own internal log**
(not visible from this WSL2 box directly) showing, at 22:54:53–22:54:55, a **terminated tool-call
generation** whose partial (never-delivered) model output was itself in Spanish —
`{"deliveryAddress": "¿Por favor, indícame tu dirección de entrega para completar..."}` — for what
the timing places as **trial 12's `pt-BR` participant, turn 3** (`"I'd like to place my order."`,
which prompts an address-confirmation tool call; this turn is recorded as `dead_turn: true`,
`no_reply`, `post_status: 200` — trial 12 completed at 22:55:20, immediately after trial 11
finished at 22:54:39, placing the flagged window inside trial 12's execution). **This is correctly
excluded from the adherence tally above** (no reply ever reached the user — it is bucketed as a
dead turn, the same as any other provider failure), consistent with how this pass already treats
every other dead turn. It is reported here as a **distinct, narrower observation**: even under
Mitigation D, the model's *internal* tool-call-argument drafting drifted into Spanish for a
`pt-BR` participant at least this once — a related-but-different symptom from DEF-6's defined
shape (a *completed, delivered* wrong-language reply), and one this pass's language-adherence
metric is structurally blind to (it can only assess replies that were actually delivered). No
other instance of this specific sub-pattern (wrong-language partial tool-call argument) was
identified in the data this pass could inspect, but this pass's own visibility is limited to
`falkor-chat`'s server log, which does not carry LM Studio's own internal per-request log —
**this sub-pattern's true frequency is not measurable from this pass's evidence alone.**

**My recommendation on how to weigh this:** do not fold the dead-turn rate into the K-065/DEF-6
adherence measurement above — it is a different failure mechanism (no reply at all, vs. a wrong-
language reply) and the language-adherence metric is not meaningfully biased by excluding it
(a dead turn contributes no signal either way about the language question). But **do treat the
62.7% dead-turn rate itself as a serious, separate demo-readiness risk** — arguably more
disruptive to a live audience than an occasional wrong-language reply — and **do not treat this
session's specific dead-turn rate as a clean estimate of "true" demo-day reliability**, given the
confirmed shared-instance contention with an unrelated process throughout this window. A rerun
on an LM Studio instance dedicated solely to this workload, with no other process attached, would
be needed to get a trustworthy dead-turn-rate figure — this pass cannot supply one.

## Coverage & gaps

- Covers exactly TP-007's literal-concurrency-variant shape, `en`/`pt-BR`/`es`, 5 turns/
  participant, at n=25 trials (vs. the original n=10) — no other TP-007 sub-variant (e.g. the
  40-way heavy variant) was re-run; out of scope per the coordination brief.
- Does not measure C or B (unauthorized, out of scope per the coordination's own charter).
- Does not re-verify AC-8 (address-adherence) or any other AC beyond AC-9b's language question —
  out of scope for this pass.
- The dead-turn/engine-stability rate measured here is **not** a clean baseline for demo-day
  reliability, per the note above (shared LM Studio instance).
- The harness (`language_salience_probe.py`) is a scratch script in this session's scratchpad,
  not committed to the repo — see Feedback.

## Feedback & recommendations

1. **The measured en-adherence rate (0 wrong-language occurrences in 23/23 trials-with-a-reply,
   61/61 replied turns) is a clean, strong result for Mitigation D specifically** — report it as
   such; the sufficiency call is the stakeholder's.
2. **The dead-turn/engine-stability rate deserves its own attention, independent of K-065** — a
   62.7% no-reply rate, even partly attributable to a shared LM Studio instance, is a demo-risk
   in its own right. Recommend a clean (single-consumer) re-run of this same protocol before
   trusting any reliability figure from this pass, and recommend whoever owns demo-day
   infrastructure confirm the LM Studio instance used will not be shared with another workload.
3. **This pass's own classifier had a real, found-and-fixed bug** (shared Ibero-Romance filler
   words outvoting a Portuguese diacritic signal) — worth recording as a technique note: when
   building an ad hoc language classifier for pt-BR vs. es specifically, prioritize diacritics
   (ã/õ/ç) and language-exclusive vocabulary over words the two languages share, and always
   manually read every classified reply (adherent and non-adherent) before trusting the
   aggregate, not just the flagged ones — this is exactly how the bug was caught.
4. **Recommend promoting the scratch harness** (`language_salience_probe.py`) to a committed
   script (parallel to `salesperson/scripts/load_demo.py`, which it borrows its HTTP-plumbing
   shape from) if this protocol is likely to be re-run for a future mitigation increment (C or
   B) — currently it only exists in this session's scratchpad and would need to be rewritten from
   scratch for a future pass.
5. **The tool-call-level Spanish-drift observation (trial 12) is narrow, single-instance evidence**
   — worth noting to whoever owns K-065 next, but not itself actionable without LM Studio's own
   per-request logs, which this pass's tooling cannot capture from this box.
