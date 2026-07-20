# M3 capability probe — Qwen3-4B vs. Ministral 3 3B, fits-16GB (K-022 / D12→D13)

> **Type:** data-scientist method note (advisory — no code/config/seam changed here). This is a
> **ready-to-run measurement spec**, not the measurement. The run is a separate, gated step (§6).
> **Author:** data-scientist · **Date:** 2026-07-18
> **Supersedes:** the original D12 *upward* (bigger-model) capability probe. Redirected by **D13**
> (`m3-executor-coordination.md`, "Locked decisions — round 5"). D12's constraints and its "then B
> regardless" framing are otherwise **unchanged and still binding**.
> **Consumes:** `m3-executor-coordination.md` (D12 §670–683, D13 §699–739), `local-model-ram-budget-ml.md`
> (§1 budget, **§7** Mistral/Ministral evaluation — this note's parent), `m3-executor-ml.md`
> (Q1 judge, Q3 dual-shape parser, D4 4B function-calling risk), `m3-guard-calibration.md` (§4 gate).

---

## 0. TL;DR

**The probe is a redirected, fits-16GB A/B: `qwen/qwen3-4b-2507` (baseline) vs. Ministral 3 3B
(2512, Apache 2.0), config/env-only (`FALKORCHAT_LLM_MODEL` swap), S5 in tree, re-measuring the two
regressed live numbers.** It is **most likely confirmatory** — Ministral-3B is *smaller and weaker*
than the incumbent 4B, and both live blockers are "a small model is too weak" problems, so the
honest expectation is that Qwen3-4B remains the best model that fits and the blockers persist as
K-023 follow-ups. The deliverable is **the recorded comparison**, not a fix and not a green light.

**Two structural cautions decide whether the numbers mean anything:**
1. **AC-4's denominator is downstream of intake advancement** (you can only observe "does the answer
   node post?" on runs that *reach* the answer node). A weaker intake stage starves the AC-4 sample —
   so a Ministral AC-4 of "0/1" is not comparable to Qwen's "0/3"; it is *uninformative*, and that is
   itself a finding.
2. **A Ministral no-post can be a parse artifact, not a genuine no-post** (§4). Mistral's tool-call
   format differs from Qwen's; if LM Studio's OpenAI-compat layer or `llm.py`'s dual-shape parser
   doesn't recognize it, `tool_calls` arrives empty and the executor sees "no tool call" even though
   the model emitted one. **The AC-4 number is void until this is classified.**

**⛔ Blocker to running it:** Ministral 3 3B is **not currently loaded in LM Studio** (which exposes
`gemma-4-12b` / `qwen3.5-9b` / `gpt-oss-20b` as of the 2026-07-18 env check). D12's "do not
download/install models" constraint stands ⇒ **the probe cannot run until the user loads Ministral 3
3B (2512)**. Exact model identity + readiness check in §6.

---

## 1. The question and the decision it serves

**Question (D13):** on *this* task, with *this* uncalibrated judge and *this* S5 primary-path def,
does the one clean 16GB-fitting Mistral candidate — Ministral 3 3B (2512) — do better than, the same
as, or worse than the incumbent Qwen3-4B on the two numbers that regressed?

- **Intake advancement** on the S5 primary path — **4B baseline 3/10**.
- **AC-4 terminal-post reliability** (does the `answer` node actually call `post_message` → a real
  `StepRun-[:PRODUCED]->Message` edge) — **4B baseline 0/3**.

**What the caller does with the answer:** *nothing that changes the plan's direction.* Per D12/D13,
**"then B regardless"** — the executor mechanism is declared proven and live-triage reliability is
descoped to a scoped U15 (mechanism + AC-1/AC-5/AC-6 verified; AC-2b/AC-3/AC-4 recorded model-gated,
structurally-demonstrated). The probe is a **data point on record**, not a gate. The only branch it
could open: *if Ministral credibly beat Qwen on both numbers* (unlikely — §5), that would argue for
adopting Ministral as the shipped chat model before the K-023 engine/calibration work; otherwise it
confirms "Qwen3-4B is the best that fits, blockers are K-023."

**Binding constraints (from D12, carried by D13 unchanged):**
- **Config/env only** — swap `FALKORCHAT_LLM_MODEL`; **no seam change** (`guards` / `executor` /
  `tools` / `app` untouched), **no def-prompt change** beyond the S5 already in the tree.
- **S5 stays uncommitted** — it regresses the flow on the 4B (10/10 → 3/10 intake advancement); it is
  present only so the probe exercises the **primary** extract-then-judge path (D8).
- **Both arms must fit 16GB** alongside FalkorDB + the co-resident 0.6B embedder. Both do:
  Qwen3-4B ~3.5–4 GB resident, Ministral 3 3B ~3–3.5 GB (`local-model-ram-budget-ml.md` §1, §7.1).
  RAM is *why* the upward candidates were dropped — they don't co-reside with the embedder under the
  16GB pool.

---

## 2. Findings from the real system (what the baselines actually mean)

Read from `m3-executor-coordination.md` (rounds 3–4) — these are teco-verified live observations, not
my estimates:

- **Intake advancement 3/10 on S5 (was 10/10 on the degraded turns-only path).** The 4B fills
  `understanding.missing` with forensic demands (server logs, infra details) on *every* turn and never
  empties it; the **uncalibrated** judge reads `missing != []` and suspends. This is OQ-4 / the note's
  own risk #4 materializing on the primary path — a *joint* generator×judge failure (see §3).
- **AC-4 0/3.** Two measured mechanisms, both persist through the strengthened S5 prompt: (a) the
  answer node emits the answer as **plain text**, no tool call (D4's 4B function-calling risk); (b) it
  calls `post_message` with a hallucinated `mentions:["alice"]` → §4 write rejects → the 4B "recovers"
  by dropping the tool and emitting prose. **Prompting a 4B did not hold this** — the coder's verdict
  routed a "terminal node must post" engine contract to K-023.
- **The judge is the same model as the generator.** `_build_llm_judge` uses `FALKORCHAT_LLM_MODEL`.
  So swapping to Ministral swaps **both** the intake node's understanding-generation **and** the guard
  judge simultaneously. The live intake-advancement number is therefore a *joint* function of both —
  which is legitimate (it is the whole-system config) but must be read with the guard golden-set
  diagnostic (§3.3) to attribute a change to the right half.
- **The harness exists and is the vehicle.** `server/tests/test_workflow_live.py` (`-m live`,
  deselected by default) drives the real `seed_workflows.sh` (so it picks up S5 automatically) over a
  throwaway **`ws:live`** bootstrapped at the **probed** live embedding dim — never hardcoded, never
  `ws:test` (dim 4, the silent-ANN-drop trap). The chat-model swap does **not** change the embedder or
  the probed dim, so `ws:live` bootstrap is unaffected. `KEEP_WS=1` keeps the graph for inspection.

---

## 3. The comparison design

Everything from D12 held constant; the **candidate set is the only change**.

### 3.1 Arms

| Arm | `FALKORCHAT_LLM_MODEL` | Role |
|---|---|---|
| **A (baseline)** | `qwen/qwen3-4b-2507` | Incumbent; numbers largely known (3/10, 0/3) but **re-measured** for a like-for-like paired read |
| **B (candidate)** | `mistralai/ministral-3-3b-instruct-2512` *(verify exact LM Studio id at load — §6)* | The one clean 16GB-fitting Mistral (`local-model-ram-budget-ml.md` §7.1/§7.6) |

Everything else identical per arm: same S5 def (via `seed_workflows.sh`), same embedder
(`text-embedding-qwen3-embedding-0.6b`, 1024-dim, unchanged), same seams, same `ws:live` at the probed
dim, same LM Studio discipline (context 8192, KV cache Q8 — `local-model-ram-budget-ml.md` §7.6).
**Re-measure both arms in the same session** — do not carry Qwen's historical 3/10 forward as arm A;
LM Studio build / template / sampling drift since those runs would confound a cross-session compare.

### 3.2 Primary metrics, sample size, and the AC-4 coupling

Each live run walks `intake → … → research → answer`. A single run yields **one** intake-advancement
observation and — **only if it advanced** — one AC-4 observation. The two metrics are *coupled*, not
independent; this is why the 4B's AC-4 was n=3 (exactly its 3 advancing runs).

| Metric | Definition | n per arm | 4B baseline |
|---|---|---|---|
| **M1 — intake advancement** | run reaches the `answer` node (research→answer is the D5 unconditional guard, so reaching-research ⇒ reaching-answer) | **n = 10 runs** | **3/10** |
| **M2 — AC-4 terminal post** | on runs that reached `answer`, the node calls `post_message` → a real `StepRun-[:PRODUCED]->Message` edge | **conditional: denominator = M1's advancing subset** | **0/3** |

**Why n = 10 for M1.** It matches the 4B baseline exactly (enabling a paired read), and each run is a
multi-minute live LLM e2e — n=10 is the affordable coarse screen the decision needs, not a powered
test. **Small-n honesty (same discipline as the golden set §8):** at n=10, 3/10 has a 95% Wilson CI of
roughly **[0.11, 0.60]** — this instrument can only detect a *large* shift. Treat M1 as "same tier /
materially higher / materially lower," never as a precise rate. A difference of ±1–2 out of 10 is
**noise**, not a result.

**Why M2's n is honest-but-small, and how to handle it.** M2's denominator is whatever M1 delivers.
- The 4B's "0/3" is **not** "0% posting"; its 95% CI is **[0, 0.56]** — essentially uninformative. Re-state
  it that way in the report: *"no reliable terminal post observed,"* not *"0%."*
- If Ministral advances even less than the 4B (the expected direction — §5), M2's denominator shrinks
  toward 1–2 and **cannot support any AC-4 claim at all**. **That is a reportable outcome**, phrased
  exactly so: *"AC-4 not measurable for arm B — the intake stage did not deliver enough runs to the
  terminal node."* Do **not** paper over it with a single lucky/unlucky run.
- **Optional extension (only if arm B's M1 is usable and the user wants a firmer AC-4 read):** run M1
  to n=20 for arm B so M2's denominator grows. Do **not** force the flow to the answer node to fatten
  M2 — that would require a seam change, which D12/D13 forbid. The coupling is a property of the
  primary path being tested, and respecting it is part of testing that path as-is.

### 3.3 Secondary / diagnostic (reported, NOT blocking here)

- **Guard-gate arms per arm** — advance-recall ≥ 0.80 AND false-advance ≤ 10% over
  `server/tests/eval/golden_guards.jsonl` (`m3-guard-calibration.md` §4), computed with the judge
  running on each arm's model. **Reported as a diagnostic, not a blocker on this probe** (D13). Its
  value here: the golden set feeds the judge **fixed inputs**, so it isolates the *judge* half of the
  M1 joint failure — it answers "did M1 move because Ministral judges `missing` differently?" separately
  from "…because Ministral *generates* a different understanding?" A gate pass/fail on 26 cases is
  **weak evidence** (§8) and carries the D10 caveat.
- **Coercion-flip rate** on the `r1_probe` cases (`m3-guard-calibration.md` §4.3) — records whether a
  *different* model's wordier/terser rationales trip `_NEGATION_CUES` differently than Qwen's did.
  Diagnostic only.
- **Qualitative trace read** (via `KEEP_WS=1`): for each arm capture one representative
  intake-suspend rationale and one answer-node terminal turn, so the *mechanism* behind a number is on
  record, not just the count. (This is also the raw material for the §4 parse-vs-no-post classification.)

---

## 4. The tool-call-format caveat — telling a genuine no-post from a parse artifact

**This is the load-bearing methodological guard for M2.** From the outside, two very different things
produce the identical AC-4 outcome "no `PRODUCED` edge":

- **(a) Genuine no-post** — the model emitted the answer as **plain prose**, never attempting a tool
  call. This is a *model-capability* finding (D4's 4B risk), and a legitimate AC-4 = fail.
- **(b) Parse artifact** — the model **did** emit a `post_message` tool call, but in Mistral's format
  (a `[TOOL_CALLS]`-token / Mistral-JSON shape, `local-model-ram-budget-ml.md` §7.6) that either LM
  Studio's OpenAI-compat layer or `llm.py`'s dual-shape parser (`m3-executor-ml.md` Q3: native
  `tool_calls` field primary, content-embedded-JSON fallback, built and tested **against Qwen**) did
  not surface. `ChatResult.tool_calls` comes back empty; the executor sees "no tool call." This is a
  **harness/format bug**, not a model verdict — and it would **confound** an AC-4 = 0 into looking
  like a capability failure.

**A Ministral AC-4 of 0 is not trustworthy until it is classified.** Classification protocol
(read-only, no seam change):

1. **Capture ≥1 raw answer-node completion for arm B.** Either inspect the `KEEP_WS=1` trace for the
   answer step, or — cleaner — replay the answer-node request against LM Studio directly: a standalone
   `POST /v1/chat/completions` at `:1234` with `model=<ministral id>`, the **same tools schema** the
   executor offers the answer node (`post_message`), and a short prompt that should elicit a post.
   Read the **raw** response body.
2. **Classify:**
   - Raw response contains a `tool_calls` array (OpenAI-native) **or** a content-embedded JSON/`[TOOL_CALLS]`
     structure that names `post_message` → **parse artifact (b)**. The model *can* call the tool; the
     pipeline dropped it. **M2 is void** until the parser/template is fixed.
   - Raw response is pure prose with **no** tool-call structure anywhere → **genuine no-post (a)**.
     M2 = fail is real; it is the capability finding.
3. **If (b): the AC-4 number is not reportable as a capability result.** Route the parser/template gap
   to `coder` with this note + `m3-executor-ml.md` §Q3 + `local-model-ram-budget-ml.md` §7.6 as the
   brief. **Do not** count a parse artifact against Ministral's tool-calling — that would slander the
   model for a harness bug and pollute the on-record comparison. (Note also the LM Studio **prompt
   template**: a Mistral GGUF served without the correct tool-use template can emit prose where a tool
   call was intended — still class (b), still a config fix, still not a model verdict.)

**Report M2 for arm B in the form:** `k/m posted, of which <j> were genuine no-posts and <i> parse
artifacts` — never a bare "0/m."

---

## 5. Decision framing — most likely confirmatory, and the pass/interpret rules

**State it plainly (D13's honest expectation):** the two live blockers are (1) a 4B too weak to
reliably call the terminal tool and (2) an *uncalibrated* judge over-weighting `missing`. **Ministral
3 3B is smaller and weaker than the 4B on the axis that matters** (BFCL-style function-calling falls
off sharply below 4B; the 2512 family has *no* published BFCL yet — `local-model-ram-budget-ml.md`
§7.2). It is therefore **unlikely to fix either blocker.** The honest expected outcome is: *Qwen3-4B
remains the best model that fits; the blockers persist ⇒ they are K-023 engine/calibration follow-ups,
not a model choice.* **"Then B regardless"** — the probe records a number, it does not gate reaching B.

Pass/interpret rules, set so a Ministral **win is credible** and a **loss is expected, not a surprise**:

| Outcome | Reading |
|---|---|
| **Ministral loses or ties** (M1 within ±2/10 of Qwen or lower; M2 no better) | **Expected.** Confirms "Qwen3-4B is the best that fits." Records the Ministral-vs-Qwen number the user asked for. B framing proceeds; blockers → K-023. **This is the modal result — not a failure of the probe.** |
| **Ministral M1 materially higher** (e.g. ≥ +4/10, beyond the coarse-n noise band) | A *credible* signal that Ministral's smaller understanding / judge behavior suits the S5 primary path better. Corroborate with the §3.3 golden-gate diagnostic (did the *judge* half improve?) before believing it. Worth a follow-up, not an instant swap. |
| **Ministral M2 > 0 with genuine tool calls** (posts, classified NOT a parse artifact per §4) | The one result that would matter — a 3B reliably calling the terminal tool where the 4B didn't. **High bar:** requires the §4 classification to confirm genuine tool calls, an adequate M2 denominator (§3.2), and grounded content (not a hallucinated-mention "recovery"). If it clears that bar, it is a real datapoint for adopting Ministral; if it's a parse artifact or n=1, it is not. |

**A Ministral win must clear the §4 classification and the coarse-n noise band to be believed; a
Ministral loss needs neither — it is the prior.** In all cases the deliverable is the recorded
comparison folded back into `m3-executor-coordination.md` and B (scoped U15) proceeds.

---

## 6. Prerequisite / blocker — Ministral must be loaded first (probe is gated on this)

**⛔ The probe cannot run yet.** Per the 2026-07-18 env check, LM Studio at `:1234` exposes
`google/gemma-4-12b`, `qwen/qwen3.5-9b`, `openai/gpt-oss-20b` — **not** Ministral 3 3B. D12's **"do
not download/install models"** constraint stands, so this is a **user action**, not a self-provision:

**Model to load (from `local-model-ram-budget-ml.md` §7.1/§7.6):**
- **GGUF repo:** `bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF` (or `mistralai/Ministral-3-3B-Instruct-2512-GGUF`)
- **Quant:** `Q4_K_M` — **~2.0 GB weights (estimated; verify the actual file size in LM Studio at
  download — the slack math assumes ≤ ~2.3 GB, `local-model-ram-budget-ml.md` OQ-5)**
- **LM Studio settings:** context length **8192** (cap despite the model's 256k — KV budget), **KV
  cache Q8** — same discipline as the incumbent.
- **`FALKORCHAT_LLM_MODEL` value:** `mistralai/ministral-3-3b-instruct-2512` — **verify the exact id
  LM Studio reports** (the served id can differ from the repo slug).

**One-line readiness check** (run before dispatching the measurement):
```
curl -s http://localhost:1234/v1/models | grep -i ministral
```
The Ministral id must appear (equivalently: it shows in LM Studio's loaded-models list). Set
`FALKORCHAT_LLM_MODEL` to *exactly* the id this returns.

**This note is a ready-to-run spec.** The measurement run (loop the `-m live` flow n times per arm,
tally M1/M2 per §3, classify M2 per §4, compute the §3.3 diagnostics, record) is a **separate step
gated on the load**. The data-scientist owns running it once the model is loadable; it never edits
seams. If Ministral is still unavailable at run time → surface to the user, do not self-provision.

---

## 7. Evaluation design — how the recorded comparison is made trustworthy

| Element | Specification |
|---|---|
| **Instrument** | `server/tests/test_workflow_live.py` (`-m live`) over throwaway `ws:live` at the **probed** embedding dim, real `seed_workflows.sh` (S5 in tree). Measurement = **run n times per arm and tally per-run outcomes**, not a single pass/fail — the test's hard assertions are for CI; the probe reads outcomes across repeats (use `KEEP_WS=1` + trace, or a loop wrapper that records M1/M2 per run). |
| **Metrics** | **M1** intake advancement (n=10/arm); **M2** AC-4 terminal post (conditional denominator, §3.2), classified genuine-vs-parse (§4). |
| **Diagnostics (reported, not gated)** | Guard-gate arms per arm (advance-recall / false-advance, `m3-guard-calibration.md` §4); coercion-flip rate; one representative trace per arm. |
| **Acceptance / decision threshold** | **None gates reaching B** (D13 "then B regardless"). Interpretation rules per §5. A Ministral "win" is credible only if it clears the §4 classification **and** the coarse-n noise band (§3.2); a loss/tie is the expected, reportable result. |
| **Honesty caveat (mandatory, verbatim next to the numbers — mirrors D10/§8)** | *"n=10 (M1) and a conditional, single-digit denominator (M2) can only detect a large effect. A tie or small difference is noise, not a result; a bare AC-4 rate is uninformative below ~5 observations. This probe records a data point under the 16GB ceiling; it does not certify a model."* |
| **Confound controls** | Re-measure both arms same-session (§3.1); classify M2 parse-vs-no-post before trusting it (§4); read M1 with the golden-gate diagnostic to attribute generator-vs-judge (§3.3). |

---

## 8. Risks & open questions (I run isolated — confirm before/at run)

- **R1 — AC-4 sample starvation (most likely to void M2).** If arm B advances < 3/10, M2's denominator
  is 1–2 and no AC-4 claim is supportable. **Not a failure to hide — report it as "AC-4 not measurable
  for arm B."** Optional n=20 extension for arm B only (§3.2), never a seam change to force the flow.
- **R2 — parse artifact masquerading as a capability result (§4).** The single biggest threat to a
  *fair* Ministral number. The §4 classification is **mandatory** before M2 is reported; skipping it
  would put a harness bug on record as a model verdict.
- **R3 — judge and generator are the same swapped model (§2).** M1 is a joint number. The §3.3 golden
  diagnostic is how the two halves are separated; without it, a moved M1 is unattributable.
- **R4 — coarse n cuts both ways.** Do not over-read a lucky Ministral run *or* an unlucky one. The §7
  caveat is not garnish; it travels with every number to the coordination doc and any U15 mention.
- **OQ1 — exact LM Studio served id + Q4_K_M file size** for Ministral 3 3B are unverified here
  (`local-model-ram-budget-ml.md` OQ-5). Pin both at load; the fits-16GB conclusion holds across the
  plausible 1.9–2.3 GB range, the slack math should use the real number.
- **OQ2 — LM Studio Mistral tool-use template.** Whether LM Studio applies the correct tool-use
  template for this GGUF is unknown to me; a wrong/missing template is a §4-class (b) artifact, not a
  model verdict. Check it as part of the §4 classification if M2 = 0.
- **Perishability.** Model ids, quant sizes, and BFCL coverage for the ~6-week-old 2512 family drift;
  `local-model-ram-budget-ml.md` §7 is the current-as-of-2026-07-18 evaluation this note rides on.

---

## Results (run 2026-07-19)

> **Status:** RUN COMPLETE. Both arms measured same-session against live LM Studio (`:1234`)
> + FalkorDB v4.18.11, via the §7 instrument. Author: data-scientist (the run this note specified).
> **Quant (supersedes §6's Q4_K_M):** the candidate ran at **Q8_0** (~3.3–3.6 GB), a deliberate
> confounder-removal — a loss cannot be blamed on Q4 damage. **Decision rule folded in:** a
> Q4_K_M reconfirm is required **only if** Ministral wins *both* numbers at Q8_0 (Q8_0 is not
> RAM-parity with the incumbent). **It did not — so no Q4 test is needed** (see verdict).
> **Method:** a config/measurement-only wrapper drove the exact shipped live-test wiring
> (`_build_live_stack`, real `seed_workflows.sh` S5 in tree, `ws:live` at the probed 1024-dim),
> N=10 runs/arm, tallying per-run M1/M2 + full answer-node/guard traces. No seam, def, or config
> touched. Both arms same session, sequential (no LM-Studio contention).

### The numbers

| Metric | Arm A — `qwen/qwen3-4b-2507` (baseline) | Arm B — `mistralai_ministral-3-3b-instruct-2512` (Q8_0) |
|---|---|---|
| **M1 — intake advancement (reached `answer`)** | **3/10** | **0/10** |
| **M2 — AC-4 terminal post** | **2/3** (2 genuine posts, 1 genuine no-post, **0 parse artifacts**) | **not measurable** (R1 — 0 runs reached `answer`) |

Arm A re-measured **exactly its 3/10 baseline** — but M2 came in **2/3 this session**, not the
historical 0/3 (runs 3 & 5 posted via a native `post_message` tool call; run 1 reached `answer`
but emitted the answer as **plain prose, no tool call** — the D4/Defect-C genuine-no-post, §4
class (a), verified in the trace). The 0/3 baseline was small-n noise, not a 0% rate — exactly the
§3.2 caveat. Both figures carry the mandatory caveat below.

### §4 classification (MANDATORY — done; the load-bearing result)

Arm B's M2 is not directly measurable (no run reached the answer node), so the §4 question was
resolved on **both halves** of the tool-call/JSON path with read-only replays and the golden set:

- **The answer-node tool-call path is CLEAN for Ministral — no parse artifact.** A direct replay
  of the answer node's `post_message` schema against Ministral (`scratchpad/replay_answer.py`,
  3/3 draws) produced a **native OpenAI `tool_calls` `post_message`** every time, which `llm.py`'s
  parser recognized cleanly (`parser_is_tool_call=True`). LM Studio's OpenAI-compat layer surfaces
  Ministral's tool call correctly; **§4 class (b) does not apply to Ministral's terminal tool.**
  Strikingly, the **same replay on Qwen emitted plain prose 3/3 (no tool call)** — on the terminal
  tool, Ministral is the *better*-behaved model. It simply never gets there.
- **The kill is a JUDGE-side format artifact (the §3.3 / R3 analog of the §4 caveat).** Ministral
  never advanced because **every** intake→research guard judgment came back
  `"unparseable judge output"` (all 4 rounds × all 10 runs). Root cause, confirmed by raw capture
  (`scratchpad/judge_replay.py`): the fuzzy-guard judge uses `llm.complete()` (free text, no
  `response_format`) and parses with a **bare `json.loads`**; **Ministral wraps its JSON in a
  ```` ```json ```` markdown fence**, which `json.loads` rejects → `_coerce_verdict` biases to
  suspend → forever `waiting`. On golden case `ca-02` Ministral's *underlying* decision was the
  correct `true` — destroyed by the fence. This is a **harness/format bug** (the judge parser was
  built and tested against Qwen only — `m3-executor-ml.md` Q3 — and is not fence-tolerant, unlike
  `llm.py`'s own `_extract_json_object`), **not a Ministral reasoning verdict.**

### Diagnostics (reported, not gating — §3.3)

Guard-gate over the 26-case golden set (`server/tests/eval/golden_guards.jsonl`), judge running on
each arm's model, feeding the judge **fixed** inputs to isolate the judge half. Target (from
`m3-guard-calibration.md` §4): advance-recall ≥ 0.80 AND false-advance ≤ 0.10.

| Judge gate (fixed golden inputs) | Arm A (Qwen3-4B) | Arm B (Ministral-3B) |
|---|---|---|
| **advance-recall** (shipped bare-`json.loads` pipeline) | **9/11 = 0.818** ✓ | **0/11 = 0.0** ✗ (26/26 unparseable — fence) |
| **advance-recall** (fence-tolerant re-parse of same output) | 9/11 = 0.818 | **4/11 = 0.364** ✗ (still fails) |
| **false-advance** | **1/15 = 0.067** ✓ | 0/15 = 0.0 ✓ (heavy suspend bias) |
| **coercion-flip rate** (`_NEGATION_CUES` on r1_probe) | **0/5** | **0/5** |

Two attributions this cleanly separates:

1. **Qwen's live 3/10 is a GENERATOR-half problem, not a judge problem.** On clean `understanding`
   inputs the judge **passes both gate arms** (0.818 / 0.067). The live shortfall is upstream: the
   intake node emits **prose / a `missing`-loaded forensic understanding** ("has not provided backend
   logs, specific error messages…"), so the judge sees degraded or deficiency-flagged evidence and
   suspends. Fix the intake generator, not the judge (an OQ-4 / K-023 item, unchanged).
2. **Ministral is blocked on BOTH counts, and would still lose with the artifact fixed.** Even after
   fence-tolerant re-parsing, its advance-recall is only **0.364** (vs Qwen 0.818) — it genuinely
   over-suspends on clean clear-advance cases (demanding logs/traces that aren't the gate's bar), and
   several of its "false" outputs are additionally **malformed JSON**, not just fenced (e.g. `ca-02`
   truncated mid-rationale). So the fence is fatal *and* the underlying judge is weaker.

**Coercion-flip rate 0/5 both arms** — the `_NEGATION_CUES` backstop did not spuriously flip any
r1_probe case for either model; it is not implicated in either arm's numbers.

**Representative traces on record** (`scratchpad/*.json`, `KEEP_WS` not needed — traces queried live):
Qwen intake-suspend rationale: *"the user has not provided backend logs, specific error messages,
details about changes in version 4.2…"*. Ministral judge raw (golden `ca-01`, a clear-advance):
`` ```json\n{"decision": false, "rationale": "No details about request context (e.g., environment,
logs, error traces)… were provided"}\n``` `` — fenced, and over-suspending.

### Honesty caveat (mandatory — travels verbatim with every number above)

> *n=10 (M1) and a conditional, single-digit denominator (M2) can only detect a large effect. A tie
> or small difference is noise, not a result; a bare AC-4 rate is uninformative below ~5
> observations. This probe records a data point under the 16GB ceiling; it does not certify a
> model.* (At n=10, the 95% Wilson CI on Qwen's 3/10 is ≈ [0.11, 0.60]; Ministral's 0/10 is
> ≈ [0.00, 0.28]. The M1 gap 3/10 vs 0/10 is at the edge of what this instrument can call — but it
> is corroborated by the mechanism: Ministral's judge is 26/26 unparseable, a deterministic block,
> not a lucky/unlucky draw.)

### Verdict (plain language)

**Ministral 3 3B LOSES to Qwen3-4B on this task, config-and-env-as-shipped.** It advanced the
intake stage **0/10** vs Qwen's **3/10**, and never delivered a single run to the answer node, so
its terminal-post reliability (AC-4/M2) is **not measurable**. This is the **modal, expected,
confirmatory** outcome of §5: *Qwen3-4B remains the best model that fits; the two live blockers are
K-023 engine/calibration follow-ups, not a model choice.* **"Then B regardless" proceeds; the
Q4_K_M reconfirm rule does NOT trigger** (it fires only if Ministral wins *both* numbers at Q8_0 —
it won neither).

**But the loss is not a clean capability verdict, and two findings should ride to teco/K-023:**

- **(Finding 1 — the decisive nuance)** Ministral's 0/10 is **dominated by a fixable harness bug**,
  not proven weakness: the fuzzy-guard judge's `llm.complete()` + bare-`json.loads` path is
  **model-fragile** — it silently converts any fence-wrapping model's every judgment into a
  bias-to-suspend. This bit Ministral 26/26 and would bite any Mistral/Gemma-family model that
  fences JSON. **Route to coder (K-023):** make the judge parse fence/prose-tolerant (reuse
  `llm._extract_json_object`) or switch the judge to `response_format`/json-schema structured
  output. *This does not change the verdict* — even fence-fixed, Ministral's judge recall is
  0.364 « Qwen's 0.818 — but it removes a real robustness gap and would let a future candidate be
  judged fairly.
- **(Finding 2 — the pleasant surprise)** On the **terminal tool-call** (the AC-4 axis everyone
  feared for a small model), **Ministral is *more* reliable than Qwen** — native `post_message`
  tool calls 3/3 where Qwen emitted prose 3/3. If the judge artifact were fixed and Ministral could
  reach the answer node, its AC-4 might well *beat* Qwen's. That is not a datapoint this run can
  bank (M2 unmeasurable per R1), but it flips the naive prior and is worth a targeted re-probe **if**
  the judge is ever made model-robust and a small-model chat swap is reconsidered.

**Bottom line for the decision:** no model swap. Qwen3-4B stays. The blockers (weak intake
generator; uncalibrated/over-suspending judge; 4B terminal-tool prose) are engine/calibration work
(K-023), exactly as §5 predicted — plus one newly-surfaced harness robustness item (judge JSON
parsing) that the probe exposed by swapping in a model with different output conventions.

---

### Provenance

Redirected from D12's upward probe per **D13** (`m3-executor-coordination.md` §699–739). Baselines
(3/10, 0/3) are teco-verified live observations from rounds 3–4 of that doc. Budget + candidate
selection from `local-model-ram-budget-ml.md` §1/§7 (this note's parent). Guard-gate diagnostic from
`m3-guard-calibration.md` §4. §0–§8 were written **before** the run as a ready-to-run spec (every
number there is a baseline recall or a specification); the **Results (run 2026-07-19)** section above
carries the fresh same-session measurements (Ministral was loaded at Q8_0 for this run — §6 blocker
cleared). Raw artifacts (per-run JSON + traces, golden-gate output, raw judge/tool replays) live in
the run's scratchpad; the numbers and mechanisms are transcribed into the Results section.
