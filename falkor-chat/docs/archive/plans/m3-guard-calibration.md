# M3 — Guard-judge calibration protocol (K-022 / U15)

> **Deliverable pair:** `server/tests/eval/golden_guards.jsonl` (the labeled data) + this document
> (the protocol a tdd-engineer implements against).
> **Supersedes, in part:** `docs/archive/plans/m3-executor-ml.md` §"Evaluation design" — the **Guard judge
> (Q1)** row only. That row's gate (`κ ≥ 0.6 AND false-advance ≤ 10%`) is **replaced** by §4 below.
> Everything else in that note stands, including its Q1 method (extract-then-judge) and its
> risk #1 *intent* — do not wire an uncalibrated judge. This document honors that intent; it
> changes the **instrument**, not the posture.
> **Calibrates:** the **post-fix** judge (`m3-guard-thread-context.md` S3–S5: RECENT TURNS block,
> understanding primary / turns-only fallback). Not today's broken prompt.
> **Status:** advisory. Author: data-scientist. No code in this document is to be pasted as-is.

---

## 1. The question and the decision it serves

**Question:** is the LLM-as-judge behind the `intake → research` fuzzy guard good enough to wire
live on the local 4B?

**Decision it gates (D6):** whether U15 wires the production judge into the executor, or the
milestone ships with the guard unwired / escalated to a stronger judge model.

**Binding constraints:** local LM Studio 4B judge; one live guard condition exists in the whole
system; the network-free pytest baseline must stay green; a human is in the loop on the intake
node (`waitsForHuman`), and the intake self-loop is bounded by the clarifying-round ceiling of 3.

**The single condition under calibration** (verbatim, `scripts/seed_workflows.sh:137`):

```
the user has provided enough information to research their request
```

Per **D5**, `research → answer` is unconditional (`""`) — it is not judged, and the golden set
contains **no cases for it**. This protocol therefore calibrates **one condition string**. It says
nothing about a future guard text; a new `llm` guard needs its own cases before it inherits this
gate.

---

## 2. Findings from the real system

Everything here was **verified by reading the code**, not inherited from a plan (see §9 for the
verified/inferred split).

**F1 — The fixture fields are not judge arguments; they are inputs to a constructor.**
`guards.evaluate_guard`'s real signature is:

```python
evaluate_guard(guard, *, ctx, run, step_output, thread, judge) -> GuardVerdict
```

It takes **no** `understanding` and **no** `turns`. It *derives* `understanding` from `step_output`
(then `ctx`) via `_extract_understanding`, and — post-fix — derives `recent_turns` from `thread`.
So the §272 schema `{understanding, turns, condition, expected}` cannot be splatted into a call.
The harness must **synthesize** `step_output`/`ctx`/`thread` from the fixture (§5.1). This is a
feature, not a nuisance: it means the eval runs through the **real precedence logic and the real
`_coerce_verdict`**, not around them.

**F2 — The note contradicts itself on field names.** §104 says
`{understanding, recent_turns, condition, expected_decision}`; §272 says
`{understanding, turns, condition, expected}`. The code's post-fix judge kwarg is `recent_turns`.
**Recommendation:** the *fixture* keeps §272's `turns`/`expected` (shorter, and the fixture is not
a call site — see F1), and the *harness variable* it becomes is named `recent_turns` to match the
judge kwarg. The golden set as authored follows this. Flagging rather than silently picking.

**F3 — R-1 is already owned deterministically, test-first.** `tests/test_guards.py:279+` already
pins the R-1 contract in both directions (`ADVANCING_RATIONALES` vs `SUSPENDING_RATIONALES`),
including the exact `"no more info is needed"` hazard. Those 11 tests **fail today** with
`TypeError: StubJudge.__call__() missing 1 required keyword-only argument: 'recent_turns'` —
i.e. they are written against the post-fix contract that S3 has not landed yet. **Consequence for
this protocol:** the golden set is *not* the R-1 detector. Deterministic detection belongs to those
unit tests. The eval's distinct job is measuring **live incidence** — whether the real judge's real
rationales trip cues the hand-written list never anticipated. §4.3 makes that a first-class metric.

**F4 — Turn rows are `repository.read_thread` shape, and `authorType` is a LIST.**
Verified at `repository.py:590-596`: `labels(author) AS authorType` returns `["User"]` / `["Agent"]`,
not `"User"`. The plan's prose says `authorType` without noting this. The golden set uses the real
list shape. `_recent_turns` as specced doesn't read the field, so nothing breaks either way — but a
fixture that lies about the shape is a trap for the next person who does read it.

**F5 — The fallback trigger is truthiness, not presence.** `recent_turns = [] if understanding else
_recent_turns(thread)`. An emitted-but-empty `{}` understanding **falls back to turns**. Golden
`path: "turns"` cases therefore set `understanding: {}` *and* require the harness to pass a
`step_output` that does not parse to a non-empty dict, *and* `ctx` without an `understanding` key.
All three, or the case silently tests the wrong path. §5.1 specifies this.

---

## 3. The core finding — symmetric metrics are the wrong instrument for a bias-to-suspend judge

This is the methodological heart of the document. It was reached by re-deriving the §272 gate's
behavior at the judge's **actual designed operating point**, and it changes the gate.

### 3.1 The judge is asymmetric *by design*

`_coerce_verdict` is explicitly bias-to-suspend: only a clean, internally consistent
`{"decision": True}` advances; malformed output, a non-bool decision, and a contradicting rationale
all resolve to `False`. The DS note justifies this: a false-advance risks a bad answer on the whole
run; a false-suspend costs one cheap clarifying question that a human unblocks, bounded at 3 rounds.

So the judge is *engineered* to have **high specificity and sacrificed sensitivity**. Any metric
that penalizes the two error directions equally is measuring the system against a symmetry the
design deliberately rejected.

### 3.2 The two gate arms decouple — quantitatively

**FAR is a function of specificity only. κ, once specificity is high, is dominated by sensitivity.**
Simulated over the authored case mix (11 clear-advance / 10 clear-suspend, 20k trials):

| true sensitivity | true specificity | false-advance rate | E[κ] | §272 verdict |
|---|---|---|---|---|
| 0.95 | 0.99 | ~1% | 0.94 | pass |
| 0.80 | 0.99 | ~1% | 0.78 | pass |
| 0.60 | 0.99 | ~1% | 0.58 | **FAR passes, κ fails → BLOCKED** |
| 0.40 | 0.99 | ~1% | 0.38 | FAR passes, κ fails → blocked |
| 0.00 | 1.00 | **0%** | 0.00 | FAR passes perfectly, κ fails → blocked |

Read the last row: **a judge that always suspends scores a perfect 0% false-advance rate.** The FAR
arm alone would bless the most useless judge constructible. This is why the `AND` is not
ceremonial — but note *what* the κ arm is actually doing there. It is not measuring "agreement." At
high specificity it is measuring **sensitivity**, wearing a chance-correction that adds noise and a
dependence on my own case-mix choice.

That is the answer to the cut-off signal *"the FAR arm behaves very differently from κ."* It does,
and the reason is structural, not incidental: **the two arms measure different classes**, and the
design pins one of them near its ceiling. FAR is nearly information-free at the designed operating
point; κ is doing all the discriminating, indirectly and badly.

### 3.3 κ is contaminated by a case mix I chose

Cohen's κ chance-corrects against the **observed marginals of the two raters on this set**. Those
marginals are a product of my hand-picked prevalence. Same judge (sens=.85, spec=.85), different
mixes:

| case mix | E[κ] |
|---|---|
| 11 advance / 10 suspend | **0.70** |
| 15 advance / 6 suspend | 0.66 |
| 6 advance / 15 suspend | 0.66 |
| 18 advance / 3 suspend | **0.55** |

An identical judge scores 0.70 or 0.55 depending on how I built the file. A gate whose pass/fail
flips on the author's sampling choices is not measuring the judge. (This is the well-known κ
prevalence effect; it is not a quirk of this set.)

### 3.4 Position

**Symmetric accuracy and Cohen's κ are the wrong primary instruments here. Replace them with the
two class-conditional rates they are clumsily proxying, and gate on those directly.**

- **False-advance rate (specificity complement)** — the *safety* arm. Keep it; it is the right
  concept. But see §6: its threshold is not measurable at this N.
- **Advance-recall (sensitivity)** — the *usefulness / non-degeneracy* arm. This is what κ was
  covertly measuring. Measured directly it is interpretable, has a clean binomial CI, needs no
  chance correction, and **does not move when I change the case mix**.
- **Boundary cases are reported, never gated** — a suspend on a genuinely ambiguous case is the
  design working, not an error. Scoring them as errors punishes the judge for obeying its spec.
- **κ: report it, do not gate on it.** Retained for continuity with the DS note and because it is a
  compact one-number summary across releases — but it is a *reported diagnostic*, and it must be
  reported **with its marginals and prevalence**, or it is uninterpretable.

**Why this is not a downgrade of the DS note.** The note's risk #1 — "do not wire an uncalibrated
judge and rely on `maxSteps` to hide it" — is exactly right and is preserved intact. The note also
earned real credit: it predicted Defect A. What §272 got wrong is narrow and forgivable: it reached
for the standard inter-rater metric without noticing that its *own* Q1 bias-to-suspend decision, two
sections earlier, had made the raters deliberately asymmetric. The gate contradicts the design it
was written to protect. That is a fixable instrument bug, not a reason to distrust the note.

---

## 4. The metric set and the gate

Computed over `server/tests/eval/golden_guards.jsonl` (26 cases; see §5 for the mechanics).

Strata as authored:

| stratum | n | `expected` |
|---|---|---|
| `clear_advance` (8 understanding-fed + 3 turns-only) | 11 | `true` |
| `clear_suspend` (7 understanding-fed + 3 turns-only) | 10 | `false` |
| `boundary` (4 understanding-fed + 1 turns-only) | 5 | `false` (policy label) |
| **gated total (clear strata)** | **21** | 11 advance / 10 suspend |

### 4.1 Gate arms (both must hold)

| # | Metric | Definition | Denominator | Threshold |
|---|---|---|---|---|
| **G1** | **False-advance rate** | judge advanced on a case labeled `expected:false` | the **10 `clear_suspend`** cases, **counted per individual judge call**, not per case-majority (§5.2) | **≤ 10%**, as a **screen** — see §6 |
| **G2** | **Advance-recall** | judge advanced on a case labeled `expected:true` | the **11 `clear_advance`** cases, per-case majority | **≥ 0.80** |

**G1's denominator excludes boundary cases.** Advancing on a boundary case is a deviation from the
designed conservatism, not a safety failure — the information genuinely was arguably sufficient.
Report it separately as `FAR_all` (over all 15 `expected:false` cases) for continuity, but gate on
`FAR_strict`.

**Why G2 at 0.80.** Simulated on the 11 clear-advance cases: a true-0.9 judge passes **91%** of the
time; a true-0.5 judge passes **3%**. That is a well-behaved screen. Compare κ ≥ 0.6 on the same
data: a *good* true-0.85/0.85 judge **fails 20% of the time**, and a boundary true-0.8/0.8 judge
(true κ = exactly 0.60) passes only **59%** — a coin flip on the gate's own boundary.

**Why G2 matters beyond usefulness — the force-advance interaction.** A judge with poor
advance-recall does not merely annoy. It burns the 3-round clarifying ceiling and then
**force-advances to research anyway** (DS note §136). That is the *worst* reachable state: the
run advances on insufficient information **with no judgment applied at all** — a false-advance
laundered through the safety valve. So low sensitivity converts, at the ceiling, into exactly the
harm G1 exists to prevent. **G1 and G2 are both safety arms.** This is the strongest argument that
the sensitivity arm must be explicit and gated, not left implicit inside κ.

### 4.2 Reported, not gated

- **Confusion matrix** (2×2) over the 21 clear cases, plus a separate 2×2 for the 5 boundary cases.
- **Per-stratum breakdown by `path`** (`understanding` vs `turns`). The DS note's risk #4 predicts
  the fallback path is "degraded but functional… expect lower κ" — this is where that prediction
  gets tested. Cases `ca-01`/`tn-01` and `ca-02`/`tn-02` are deliberate near-pairs carrying the same
  evidence through both paths; the delta between them **is** the fallback-path cost estimate.
- **Cohen's κ** over the 21 clear cases, reported **with both raters' marginals and the set
  prevalence**, and labeled: *diagnostic, not a gate; comparable only across runs on this exact file.*
- **Conservatism rate** on the 5 boundary cases (share suspended). Expected to be high; a *low*
  value is an early warning that the bias-to-suspend posture is eroding.
- **Materiality-probe outcomes** (`ca-04`, `ca-05`, `ca-08` vs `cs-04`): does the judge assess
  sufficiency, or is it reading `len(missing)`? A judge that suspends all of `ca-04/05/08` **and**
  advances `cs-04` is doing string-length inference and should be reported as such regardless of
  whether it passes G1/G2.
- **`tn-05` in isolation** (assistant's clarifying question is the last turn, unanswered). An
  advance here is a qualitatively distinct fallback failure — the judge crediting its own question
  as if answered. Worth calling out by name in the report.

### 4.3 Coercion-flip rate (the R-1 live-incidence metric)

Per F3, the harness **must record the judge's raw output and the coerced verdict separately**:

```
raw_decision, raw_rationale, final_decision, coercion_flip := (raw_decision is True and final_decision is False)
```

**`coercion_flip_rate`** = share of calls where the judge said advance and `_coerce_verdict`
overrode it. This is the *only* thing that distinguishes an R-1 wording trip from Defect A from a
genuine judge suspend — from the outside all three look identical ("the guard never fires").
Report it overall and restricted to the 5 `r1_probe: true` cases.

**Report, do not gate.** A non-zero flip rate on `r1_probe` cases after S3 lands is a **defect
report against `_NEGATION_CUES`**, not a verdict on the judge — the judge was right and the
substring matcher was wrong. Routing it to the gate would blame the wrong component.

**Honest limitation:** `r1_probe` cases cannot *force* R-1. The fixture controls the judge's
*inputs*, not the rationale wording it chooses. These cases raise the probability that a correct
advance is phrased with a negated cue ("no more info is needed"); they do not guarantee it. A zero
flip rate is therefore **weak** evidence — it may mean R-1 is fixed, or merely that the judge
phrased things differently that day. The deterministic proof lives in `test_guards.py` (F3); this
metric only catches cue collisions the hand-written list didn't imagine.

---

## 5. Harness specification (for the tdd-engineer)

Advisory spec. **I have not written the harness and should not** — this is the contract to build against.

### 5.1 Constructing a call from a fixture row

Per F1, each row is assembled into `evaluate_guard` arguments. The **whole point** is to go through
the real `evaluate_guard`, so the eval exercises `_extract_understanding`, the S3 precedence, and
`_coerce_verdict` — not just the prompt.

| fixture field | becomes |
|---|---|
| `condition` | `guard = json.dumps({"kind": "llm", "text": row["condition"]})` |
| `understanding` (non-empty, `path: "understanding"`) | `step_output = json.dumps({"understanding": row["understanding"]})` — the envelope form the post-S5 intake node emits |
| `understanding` (`{}`, `path: "turns"`) | `step_output` = a **prose** string that does not parse as a JSON object (e.g. the node's raw text), **and** `ctx = {}` with **no** `understanding` key — per F5, all three conditions are needed or the case silently tests the primary path |
| `turns` | `thread = row["turns"]` (already `read_thread` shape, chronological, newest last) |
| — | `run = {}`, `judge = <the real production judge from app._build_llm_judge>` |
| `expected` | the gold label |

**Assert the path was actually taken.** A `path: "turns"` case that silently ran the understanding
branch is a *worthless* data point that will still score. The harness should assert, per case, that
the judge received a non-empty `recent_turns` (turns cases) or a non-empty `understanding`
(understanding cases) — a thin recording wrapper around the judge callable, in the spirit of the
existing `StubJudge.calls`. **This assertion is not optional**; without it the fallback-path arm of
the whole eval is unfalsifiable.

### 5.2 Non-determinism

Temperature 0 does **not** guarantee determinism on a local llama.cpp/LM Studio backend (batching
and kernel non-determinism both bite). Therefore:

- **Run every case `k = 3` times.** 26 × 3 = 78 judge calls — cheap enough to be routine.
- **G2, κ, and the confusion matrix use the per-case majority** (k=3 ⇒ a clean 2-of-3).
- **G1 (false-advance) is computed over ALL individual calls, not majorities.** Rationale:
  **production takes exactly one sample.** A case that advances 1-in-3 times is a real 33% chance
  of a false advance on the live path; majority-voting it to "suspend" would report a safety
  property the deployed system does not have. Majority voting is legitimate for measuring *ability*
  (G2) and wrong for measuring *risk* (G1).
- **Report `flip_rate`** = share of cases whose 3 replicates disagree. Instability on a `clear_*`
  case is itself a finding — a judge that flips on unambiguous input is not calibrated regardless of
  its aggregate scores, and this number belongs in the report even when both gates pass.
- **Fix the seed / record the model id, quantization, temperature, and prompt revision** in the
  report header. Without them the run is not reproducible and the numbers are not comparable across
  releases.
- **Replicates do not buy sample size** — see §6.

### 5.3 pytest gating (network-free baseline stays green)

Follow the existing convention (`AGENTS.md`): the eval is **offline, opt-in, behind the live-LM-Studio
marker**, exactly like `test_services_live.py` / `test_workflow_live.py`. It must **never** run in the
default `.venv/bin/python -m pytest -q` baseline — it needs a live 4B and it is slow and stochastic.

- Fixture file: `server/tests/eval/golden_guards.jsonl` (authored; test-only).
- Marker: the established `live` marker. A default run **deselects** it; no network, no LM Studio.
- Output: a metrics report at `docs/test-reports/m3-guard-calibration-<date>.md` carrying every §4
  number, the §5.2 provenance header, and the full per-case table (id, tier, path, expected, the k
  raw decisions, raw rationales, final decisions, coercion_flip).
- **Leakage guard:** these 26 cases are **test-only and must never be seeded into a prompt, a
  few-shot block, or the live corpus.** They are also, as authored, *not* drawn from the seeded
  triage corpus — they are synthetic-but-realistic. That is deliberate (no contamination path from
  the corpus the research node retrieves over), and it is also a representativeness limit (§7).

---

## 6. Statistical honesty — what 26 cases can and cannot establish

**Asked directly: can 20–30 cases support a κ ≥ 0.6 claim? No. Not as an inference.**

Numbers, simulated over the authored mix:

- **Observed κ = 0.6 at N=21 carries a 95% CI of roughly [0.24, 0.90].** A "passing" 0.6 is
  statistically compatible with a true κ of 0.24 — poor agreement. The measurement cannot
  distinguish a good judge from a mediocre one.
- **A genuinely good judge (true 0.85/0.85, true κ ≈ 0.70) fails the κ ≥ 0.6 gate 20% of the time.**
- **A judge sitting exactly on the threshold (true κ = 0.60) passes 59% of the time** — the gate is
  a coin flip at its own boundary.

And the same honesty applies to the arm I am **keeping**:

- **G1's `≤ 10%` threshold is not measurable at this N either.** With 10 clear-suspend cases, even a
  **perfect 0/10** yields a 95% CI of **[0%, 27.8%]**. Zero observed false advances is compatible
  with a true FAR of 28%. By the rule of three, bounding true FAR at ≤10% with 95% confidence needs
  **~30 clear-suspend cases with zero failures** ([0%, 11.4%]) — i.e. a golden set roughly 3× the
  suspend stratum specified, ~50 cases total.

**Do replicates rescue this? No.** 10 suspend cases × 3 reps = 30 calls, and 0/30 would *look* like
[0%, 11.4%]. That number would be a lie. Replicates are **repeated measures on the same 10 inputs**;
they reduce measurement noise, they do not sample new input space. The effective N for generalizing
to *unseen* user requests is **the number of cases (10), not the number of calls (30)**. Reporting
the call-level CI as if it were a case-level CI is the single most likely way this eval gets
misread — the report must state the case-level denominator next to every rate.

### 6.1 What the gate legitimately *is*

**A one-sided screen, not a certification.** The inference runs in exactly one direction:

- **FAIL ⇒ reject, confidently.** The gates have good power against genuinely bad judges (a
  true-0.5-sensitivity judge fails G2 **97%** of the time; an always-suspend judge fails it 100%).
  A judge that fails this set is almost certainly not fit to wire. **This is sound and it is the
  decision D6 actually needs.**
- **PASS ⇏ "the judge is calibrated."** It means, precisely: **"no blocker was found at a sample
  size that could only have found a large one."** Anyone who writes "κ = 0.64, judge calibrated ✅"
  in a report has laundered an unmeasured judge.

**This is the crux, and it is worth being blunt: D6 is not wrong, it is under-specified about the
direction of inference.** Building the set and honoring the gate is the right call — the set is
cheap, and its rejection power is real and sufficient to catch the failure modes that would
embarrass M3. What must not happen is the pass being read as evidence of quality. The protocol's
job is to make the asymmetry structural rather than a matter of the reader's care, so §8 fixes the
report's wording.

### 6.2 If a stronger claim is ever needed

To *certify* rather than screen: ~30 clear-suspend cases (FAR bound) + ~30 clear-advance cases
(sensitivity CI ±0.15) ≈ **60–70 cases**, plus a **second independent human labeler** on the
boundary stratum with inter-human agreement reported. That is a real cost (a day of labeling) and
it is **not** recommended now — the screen is the right instrument for a proof-milestone gate. It is
recommended **before** the judge is trusted on a path where no human can unblock it, which the
intake guard is not.

---

## 7. What happens when the gate fails

Stated plainly, because the DS note left it as prose and prose is where gates go to die.

- **G1 (false-advance) fails ⇒ BLOCK the live wiring.** No override. A judge that advances on
  clearly-insufficient information is the failure mode the whole extract-then-judge design exists to
  prevent. Escalate per DS risk #1: a stronger/distinct judge model, or a more concrete guard `text`.
  Do **not** compensate by lowering `maxSteps`/`maxIterations` — the note is explicit that ceilings
  must not paper over a failing calibration, and §4.1 shows the ceiling *converts* into the harm.
- **G2 (advance-recall) fails ⇒ BLOCK the live wiring.** Per §4.1, a judge that will not advance is
  not merely useless: it force-advances at the round ceiling with no judgment applied. Same
  escalation path. *(This is a deliberate strengthening: under §272 a low-sensitivity judge with
  clean FAR could have been argued through on the "conservative is safe" reading. It is not safe.)*
- **κ below 0.6 with both gates passing ⇒ do NOT block.** Report it, note it, ship. Per §3 this is
  the expected signature of a correctly conservative judge on a boundary-heavy set, and per §6 the
  measurement cannot distinguish 0.55 from 0.75 anyway. **This is the concrete behavioral change
  from §272** — under the old gate this case blocks; under this one it does not.
- **Both gates pass ⇒ wire it, and write the §6.1 sentence into the report** — "no blocker found at
  n=21; this is a screen, not a certification." The next person to read the number is the one this
  protects.
- **`coercion_flip_rate > 0` on `r1_probe` cases ⇒ a defect ticket against `_NEGATION_CUES`**,
  independent of the gate verdict. Not a judge failure (§4.3).
- **The materiality probes fail as a bloc (`ca-04/05/08` suspended AND `cs-04` advanced) ⇒ report as
  a blocker-grade finding even if G1/G2 pass.** It means the judge is pattern-matching `missing`'s
  length rather than assessing sufficiency, and its passing scores are an artifact of the set's
  correlation between `missing == []` and true sufficiency. This is the one failure mode that can
  slip both gates. **It is the reason `cs-04` exists.**

---

## 8. Report template requirements

The generated `docs/test-reports/m3-guard-calibration-<date>.md` must carry, non-negotiably:

1. **Provenance header:** model id, quantization, temperature, k, prompt revision (S4 commit),
   fixture sha256, date.
2. **The verdict line, in this form:** `G1 false-advance = X% (n=10 cases / 30 calls) · G2
   advance-recall = Y% (n=11 cases) · VERDICT: wire / block`.
3. **The §6.1 caveat sentence, verbatim, adjacent to the verdict** — not in a footnote:
   > This gate is a one-sided screen at n=21 hand-labeled cases. A failure is strong evidence the
   > judge is unfit. A pass means only that no large defect was detected at a sample size that could
   > not have detected a small one. It is not a calibration certificate.
4. κ **with both marginals and prevalence**, under a `diagnostic — not a gate` heading.
5. Per-`path` breakdown (the risk-#4 fallback-cost estimate).
6. `coercion_flip_rate`, overall and on `r1_probe` cases.
7. `flip_rate` (replicate instability).
8. The full per-case table, including **raw rationales** — the qualitative read of *why* the judge
   suspended is worth more than any aggregate at this sample size, and it is free.

---

## 9. Verified vs. inferred

**Verified by reading / executing:**

- `evaluate_guard`'s real signature, the `_NEGATION_CUES` list, `_rationale_contradicts`'s substring
  match, `_coerce_verdict`'s bias-to-suspend, `_extract_understanding`'s precedence — `guards.py`.
- The R-1 hazard is real and live: `"more info"` ∈ `_NEGATION_CUES`, matched as a bare substring
  (`guards.py:132`), so `"no more info is needed"` trips it.
- The verbatim guard condition and the `""` D5 research→answer transition — `seed_workflows.sh:130-150`.
- `test_guards.py:279+` already pins R-1 both directions; **executed** — 11 tests fail today with
  `TypeError: … missing 1 required keyword-only argument: 'recent_turns'`, confirming test-first
  against the unlanded S3 contract.
- `read_thread`'s row shape **and** that `authorType` is `labels(author)` → a **list** —
  `repository.py:590-596`.
- Every statistic in §3, §4.1 and §6 — simulated (20–40k trials) / Wilson intervals, computed during
  this run, not quoted.
- The golden set's own invariants (26 rows, unique ids, real condition string on every row, strata
  counts, turns-cases have empty `understanding`, boundary cases all `expected:false`) — asserted
  mechanically.

**Inferred / assumed (flag if wrong):**

- The **post-fix** judge behaves as `m3-guard-thread-context.md` S3–S5 specifies (`recent_turns`
  kwarg, `RECENT_TURNS_N=6`, omit-when-understanding-present). Read from the plan, **not** from
  landed code — S3 has not landed. If the landed shape differs, §5.1 needs a revision.
- The intake node post-S5 emits the `{"understanding": {...}}` **envelope**. `_extract_understanding`
  accepts both envelope and bare object, so §5.1 is robust either way.
- **Never executed:** the live judge. Per the brief, the fix is in flight; no live numbers exist in
  this document and none should be inferred from it. Every number here is a *simulated property of
  the metric*, not a measurement of the judge.
- The `live` marker's exact name/config — inferred from the existing live test files' convention.
- The 4B's actual sensitivity is **unknown**. The gate thresholds are chosen for their *screening
  behavior*, not fitted to observed performance.

---

## 10. Risks & open questions

1. **G1's threshold outruns its sample (highest).** §6: even 0/10 cannot establish FAR ≤ 10%. The
   gate is retained as a screen with eyes open. **OQ:** if M3 later puts the judge anywhere a human
   cannot unblock, the suspend stratum must grow to ~30 before that ships.
2. **Representativeness.** 26 synthetic cases written by one person (me), in one domain register
   (technical support triage), against one condition string. They are realistic, not sampled from
   traffic. When real intake transcripts exist, **re-derive the set from them** and treat this file
   as the bootstrap. The synthetic origin is what buys the leakage guarantee (§5.3) — that trade is
   deliberate.
3. **Single labeler.** All 26 labels are mine. The `clear_*` strata I consider robust — they are
   clear by construction. The 5 **boundary** labels are a **policy choice** (bias-to-suspend), not
   ground truth, which is exactly why they are excluded from both gates. **Do not let anyone compute
   a headline accuracy over all 26 cases** — it would silently gate on my policy preferences.
4. **Self-preference (inherited, DS risk #3).** If the same 4B both emits the `understanding` and
   judges it, it is grading its own homework. Unmeasurable with this set alone. **OQ:** worth a
   distinct judge model, and worth reporting the primary/fallback delta (§4.2) as a partial proxy.
5. **`bd-04` indicts the guard text, not the judge.** A two-part request where one part is
   researchable is genuinely underdetermined by "the user has provided enough information to
   research their request." If the judge scatters there, the fix is a **sharper condition string**
   (DS note's own escalation), not a better judge. The condition is a design artifact and is in
   scope for revision.
6. **The gate calibrates one string.** Any new `llm` guard inherits nothing from this run.
