# Golden-set expansion for the guard-judge calibration (K-027 item 4)

> **Status:** archived · **Owner:** `data-scientist` · **Tracks:** K-027 (item 4) · **Version:** 3
> · **Reviews:** `docs/reviews/golden-set-expansion.md` (analyst, 2026-08-20, needs changes → fixed
> this revision; Pass 3 approved unconditionally)

**2026-08-21: implementation delivered and gate-approved; K-027 closed.** `tdd-engineer` landed
this plan exactly as written (fixture 26→85, all 59 §6 rows byte-exact, the five literal-constant
edits, a new offline integrity test) and a live calibration run passed (G1=10.0%/120 calls,
G2=86.7%/30 cases, verdict "wire"); `analyst`'s Pass 3 approved the implementation diff
unconditionally. No further revision of this plan is expected.

**2026-08-20 revision (finalization pass):** the two open decisions §10 (v1) named are now
**closed** — both by the user, this date. (1) The `boundary`-tier independent-second-labeler
requirement is **dropped**: no second labeler is available, so `boundary` is now sourced the same
way as `clear_advance`/`clear_suspend` (§5(c) option (a) — LLM-drafted, human-spot-checked before
merge), a deliberate, explicit descope from the backlog item's original independence ask, not a
silent one. (2) `clear_suspend` size is **n=40** (this note's own §3.1 lead recommendation), not
the n=53 1-failure-tolerant alternative. Every section below conditional on either decision is
revised accordingly; §10 carries the full closure record. §6 is replaced in full with the 59
drafted rows this closure unblocks. Nothing in §1–§4's findings or methodology changes — this is a
finalization pass, not new research.

**2026-08-20 fix pass (v2 → v3, per `analyst`'s plan-gate review):** four items fixed, all
confined to this document, no drafted-row content changed except the two `r1_probe` corrections
below. (1) §7 step 5 no longer says boundary needs "a second independent human pass" — that
sentence was a leftover contradicting the closed descope; reworded to match §5/§8/§10. (2)
`r1_probe: true` was mismarked on `cs-10`/`cs-13` (both `clear_suspend`) — flipped to `false`;
`r1_probe` marks only `clear_advance` coercion-risk cases per its actual, narrower semantics
(`guard_calibration.py`'s `coercion_flip_rate`), now stated explicitly in §6's preamble. (3) §7
step 2 and F3's line citations for the `clear_cases`/`boundary_cases` asserts corrected from
233/234 to the actual 232/233 (re-verified against the current file). (4) §3.1's Wilson table
corrected the n=53/x=0 cell from a stale `~8.4%*` to the correct `6.8%` (and dropped the
`minimal-n-by-search` asterisk, which never applied to that cell) — does not affect the accepted
n=40 decision.

## 1. The question and the decision it serves

**Question:** what does `server/tests/eval/golden_guards.jsonl` need to become for the `intake →
research` guard-judge calibration (`docs/archive/plans/m3-guard-calibration.md`, `docs/plans/
guard-judge-calibration-ml.md`) to graduate from a **screen** ("no blocker found at a sample size
that could only have found a large one," §6.1) toward an actual **bound** on false-advance rate —
and how do we source and label the additional rows without quietly laundering the very
LLM-judgment the calibration exists to check?

**Decision this note serves, and does not make:** whether to commission the expansion at all, at
what size, sourced how, and labeled by whom. This is a design input to that decision, not the
decision — §5 states plainly what only the user can resolve.

**Constraints carried over from the calibration protocol (still binding):** the judge is
*designed* bias-to-suspend (archived §3); the gate is asymmetric — G1 (false-advance, the safety
arm) and G2 (advance-recall, the usefulness arm) are gated, κ and raw accuracy are diagnostics
only; boundary-tier labels are **policy choices**, not ground truth, and must never enter a gate.
Nothing here revisits that architecture — it revisits only the fixture's size, composition, and
provenance.

## 2. Findings from the real system

**F1 — the current strata, exactly.** Parsed `server/tests/eval/golden_guards.jsonl` directly (26
rows): `clear_advance` n=11 (8 `understanding` + 3 `turns`), `clear_suspend` n=10 (7 + 3),
`boundary` n=5 (4 + 1). Path split overall: **19 `understanding` (73%) / 7 `turns` (27%)**. This
matches `guard-judge-calibration-ml.md` F4's mechanically-verified count exactly — no drift since
2026-08-17.

**F2 — the turns/fallback path is thin in exactly the strata that matter for a bound.**
`clear_suspend` has only 3 turns-path rows, `boundary` only 1. The live 2026-08-17 report's
per-path breakdown (`docs/test-reports/guard-judge-calibration-2026-08-17.md`) already shows the
fallback path behaving *differently* from the primary path (turns: 100% accuracy, n=6;
understanding: 86.7%, n=15, `fn=2`) — on a base too small to say whether that's real or noise. A
sibling unit in this run (U1, `guards.py`/`executor.py`/`app.py` fixes, carried finding m-3) adds
the evidence tier to the execution trace, which is what will let a *future* calibration run slice
live production incidents by path — but the golden set itself is what lets a *calibration run*
measure the fallback path's FAR/recall with any power at all. Today it can't: 3-4 turns-path cases
per stratum support no defensible rate.

**F3 — the harness's tier grouping is generic; only three test-level assertions are
fixture-size-specific.** Read `server/tests/eval/guard_calibration.py` in full: every metric
function (`false_advance_rate`, `advance_recall`, `confusion_matrix`, `cohens_kappa`,
`per_path_breakdown`, …) groups dynamically by `c.tier`/`c.path` read from the fixture — nothing
hardcodes 26/21/5/11/10. The **only** place fixed counts are asserted is
`server/tests/eval/test_guard_calibration_live.py`:

```
:223  assert len(rows) == 26, f"golden_guards.jsonl has {len(rows)} rows, expected 26"
:227  assert len(cases) == 26
:228  assert sum(len(c.replicates) for c in cases) == 26 * K_REPLICATES
:232  assert len(clear_cases) == 21
:233  assert len(boundary_cases) == 5
```

(**2026-08-20 correction, analyst review:** the last two lines were previously mis-cited as 233/234
— re-read directly against the current file, they are 232/233. Content and target values were
already correct; only the two line numbers were off by one.)

`test_guard_calibration.py` (offline, stub-judge unit tests) uses small synthetic case lists built
inline (e.g. `_case("ca-01", …)`) to test the metric functions themselves — it never reads the real
fixture and needs **no** change for an expansion. This narrows the blast radius considerably: an
expanded fixture is a **content change plus five literal-constant edits**, not a harness redesign.

**F4 — no structural-integrity test exists for `golden_guards.jsonl`.** `server/tests/eval/
test_golden_set_integrity.py` performs exactly this function (unique ids, required fields,
self-retrieval guard, cache-currency) for the *retrieval* golden set, `golden_retrieval.jsonl` —
there is no analogous offline file for the guard fixture today. The only checks that exist are the
five live-gated literal assertions in F3, which (a) only run behind the `live` marker and (b) are
brittle-by-hardcoding rather than structural. §6 below recommends closing this gap as part of the
expansion, since it is exactly the kind of check that should fail fast, offline, on a malformed
fixture edit rather than surface after a live LM Studio run.

**F5 — `ws:acme` (the live dev/demo workspace) is not a usable source of realistic triage
transcripts, checked read-only.** Per the brief, I checked whether real triage-flow trace excerpts
exist and are safely readable. `ws:acme` (`mcp__cypher__query`, read-only) has 2 threads and 21
`WorkflowRun`s (7 waiting / 10 done / 4 failed), but the thread content is trivial smoke-test
chatter (`"@assistant hi"` → `"Hi! How can I assist you today?"` — the whole transcript of one
thread; the other, `"defeito 1"`, has no messages reachable via the `HEAD`/`NEXT` chain at all).
This is dev-smoke data, not triage traffic — there is no real corpus to mine here. The existing
26 rows' own framing (archived §7 risk 2: "26 synthetic cases … not sampled from traffic … when
real intake transcripts exist, re-derive the set from them") still describes the actual state:
there is no real corpus yet, anywhere in this system. **Sourcing new cases means writing more
synthetic-but-realistic scenarios, not mining live data — this is not a stopgap, it is still the
only option.**

**F6 — the Wilson-interval methodology used for statistical honesty is the archived note's own,
reproduced independently, not invented for this note.** §3 below reconstructs the archived §6
numbers (`[0%, 27.8%]` at n=10/x=0, `[0%, 11.4%]` at n=30/x=0) from the standard two-sided 95%
Wilson score interval — both reproduce to within rounding, confirming the archived note's method
and letting this note extend it with confidence rather than introduce a second, uncross-checked
statistic.

## 3. Target composition — a derivation, not a re-quote of the backlog heuristic

The backlog text ("~30 `clear_suspend` at zero failures, ≈50–60 total") is explicitly flagged in
the item itself as an estimate. I recomputed it.

### 3.1 The suspend stratum (G1 — false-advance rate, the safety arm)

Using the two-sided 95% Wilson score interval (`z=1.96`) — the same formula that reproduces the
archived note's own `[0%, 27.8%]`/`[0%, 11.4%]` figures:

| n (suspend cases) | upper bound at **0** observed failures | upper bound at **1** failure | upper bound at **2** failures |
|---|---|---|---|
| 10 (current) | 27.8% | 39.4% | 49.1% |
| 30 | 11.4% | 16.7% | 21.3% |
| **35** | **9.9%** | 14.5% | 18.6% |
| 40 | 8.8% | 12.9% | 16.5% |
| **53** | 6.8% | **9.9%*** | — |
| 69 | — | — | **9.7%*** |

(*minimal n at which that failure count still clears ≤10%, found by search)

**2026-08-20 correction (analyst review):** the n=53/x=0 cell previously read `~8.4%*`, both wrong
and mis-marked — recomputed under the same Wilson formula this note uses everywhere else, n=53 at
zero observed failures is **6.8%**, and it isn't a "minimal n found by search" result at all (that
framing only applies to the 53/x=1 and 69/x=2 cells, which do reproduce correctly) — n=35 is
already identified elsewhere in this section as the real zero-failure floor, so the asterisk never
belonged on this cell. This is a stale figure inherited from v1, corrected now; it does not touch
the accepted n=40 decision, which was independently re-verified correct at 8.8% (§3.4).

**The backlog's "~30" undershoots even its own zero-failure target once computed exactly: n=30 at
zero observed failures gives 11.4%, which *fails* the ≤10% gate.** n=35 is the real zero-failure
floor (9.9%, just under). This is not a large correction, but it is the difference between a
defensible claim and a technically-failing one, and it is exactly the kind of thing a heuristic
"~30" quietly gets wrong.

**A more important finding than the point estimate: at n=35, a single observed false-advance among
the 35 (not zero) blows the bound out to 14.5% — the gate becomes a coin flip on whether the run
happens to land 0 or 1 failures, for a judge whose true FAR is genuinely near the 5–8% range** (see
the power table below). This is a real design choice, not a bug to be sized away:

| true FAR | P(pass, n=35) | P(pass, n=53) | P(pass, n=70) |
|---|---|---|---|
| 0% | 100% | 100% | 100% |
| 2% | 49% | 71% | 84% |
| 3% | 34% | 53% | 65% |
| 5% | 17% | 25% | 31% |
| 8% | 5% | 7% | 7% |
| 10% (at threshold) | 2.5% | 2.6% | 2.4% |

Read this as: **n=35 is a zero-tolerance screen** — it only passes on a run with literally zero
observed false-advances, and it has only middling power (34–49%) to pass a judge whose *true* FAR
is a genuinely-good 2–3%. n=53 buys real tolerance for exactly one observed failure (consistent
with `evaluate_guard`'s own k=3-replicate philosophy: a single anomalous call among 35×3=105 or
53×3=159 calls is not implausible even for a good judge) and meaningfully better power at low true
FAR. n=70 buys little more power per case beyond that.

**Recommendation: n=40 `clear_suspend` cases.** This clears the zero-failure Wilson bound with a
small margin (8.8% vs. the 10% threshold, versus 35's uncomfortably tight 9.9%), stays a
zero-tolerance screen consistent with the existing no-override gate policy (archived §7: "G1
fails ⇒ BLOCK, no override"), and avoids paying for 1-failure tolerance (n=53, +13 cases, +39
live calls at k=3) unless the team has reason to expect a well-behaved judge to occasionally trip
on a single case by chance rather than by defect.

**2026-08-20: decided by the user — n=40, the zero-tolerance screen, not n=53.** This was a real
trade-off, not a foregone conclusion (the note flagged it as a live choice rather than picking it
unilaterally); it is now closed. §3.4's composition table, §7's implementation steps, and the
labeling-volume figures throughout this note were already written against n=40 — this decision
confirms rather than revises that arithmetic (checked below, §3.4). The n=53 alternative remains
recorded here for the historical rationale, not as an open option.

### 3.2 The advance stratum (G2 — advance-recall, the usefulness arm)

Wilson CI half-width at the observed 80% recall point, by n:

| n (advance cases) | 95% CI at 80% observed | half-width |
|---|---|---|
| 11 (current) | [52.3%, 94.9%] | ±21.3pp |
| 25 | [60.9%, 91.1%] | ±15.1pp |
| **30** | **[62.7%, 90.5%]** | **±13.9pp** |
| 40 | [65.2%, 89.5%] | ±12.1pp |

**Recommendation: n=30**, unchanged from `guard-judge-calibration-ml.md`'s and the archived
protocol's own §6.2 target ("~30 clear-advance … sensitivity CI ±0.15") — my independent
recomputation lands at exactly this number and confirms it was not a loose guess. Power against a
genuinely poor judge is already strong here (a true-0.5-sensitivity judge fails G2's ≥0.80 screen
97%+ of the time even at n=11; n=30 does not need to buy much more rejection power, it buys CI
tightness for the *pass* case).

### 3.3 The boundary stratum — not statistically sized, sized for coverage

Boundary cases are never gated (archived §4.2/§4.1); their labels are policy choices. Sizing here
is not a power calculation, it's a coverage question.
**Recommendation: n=15** (3× current) — enough distinct genuinely-ambiguous edge shapes to be a
meaningful coverage exercise without turning drafting and spot-check into a multi-day task.

**2026-08-20 note:** this section originally sized n=15 partly against "a meaningful
*independent*-labeling exercise" — §5's 2026-08-20 decision drops the independent-second-labeler
requirement for this tier, so n=15 is now justified purely on coverage grounds (distinct edge
shapes worth having in the fixture), not on giving a second labeler a proportionate amount of
work. The number itself is unchanged; only the justification for it is.

### 3.4 Path stratification — deliberately over-sample `turns` relative to today's 27%

Per F2, the fallback path is both thinner and (on the one live run available) showing a different
error signature than the primary path. Recommended per-stratum split:

| stratum | total | `understanding` | `turns` |
|---|---|---|---|
| `clear_advance` | 30 | 18 | 12 |
| `clear_suspend` | 40 | 24 | 16 |
| `boundary` | 15 | 9 | 6 |
| **total** | **85** | **51** | **34** |

This moves the overall turns share from 27% → 40%, and — more importantly — gives each stratum's
`turns` subgroup enough rows (6–16) to report a per-path breakdown that is more than 3 anecdotes,
even though (being a reported diagnostic, not gated per §4.2) it doesn't need its own formal power.

**Total: 85 rows, up from 26 — bigger than the backlog's "~50–60" estimate**, because that estimate
was call-count-derived (roughly a rule-of-three read) rather than the same Wilson-based method the
calibration protocol itself already established and that I'm holding this note to for consistency.
At k=3 replicates, that's 255 live judge calls against a local 4B — the existing protocol already
called 78 calls "cheap enough to be routine" (archived §5.2); 255 is the same order of magnitude
and not a meaningful cost concern for a local model. The real cost is **labeling** 59 new rows (85
− 26), which is squarely §5's question.

**2026-08-20 consistency check (n=40 finalization):** re-verified every place in this note that
states a `clear_suspend`/stratum size or total, against the now-final n=40. The table above already
read 40/24/16 — it was written against the lead recommendation, not the n=53 alternative, so no
number in it changes. Checked and confirmed consistent: the 85-row total above (30+40+15);
§7 step 1's fixture-size target (26 → 85); step 2's literal-constant edits (`== 70` i.e. 30+40,
`== 15`); step 5's cost estimate (85×3=255). Per-path split for `clear_suspend` (24 `understanding`
+ 16 `turns` = 40) was likewise derived directly against 40, not re-derived off 53 — no correction
needed. The only actual edits this decision required were narrative (§3.1, §8, §10), not numeric.

## 4. Sourcing new cases

Per F5, there is no real triage corpus in this system to mine — `ws:acme` is dev-smoke data, not
production traffic. The only viable source is the same one that produced the current 26:
**hand-drafted synthetic-but-realistic scenarios in the existing register** (technical
support/ops triage — checkout errors, deploys, rate limits, auth failures), extended to cover
under-represented edge shapes. Concretely, beyond simply padding out the existing shapes (more
`ca-*`/`cs-*` clones), the expansion should target shapes the current 26 don't exercise at all,
concentrated on the `turns` path per §3.4:

- **Multi-turn evidence aggregation** (5+ turns, information dispersed rather than delivered in one
  clean Q→A pair) — tests whether the judge aggregates across a longer window, not just the
  cleanest possible 3-turn exchange (`tn-01`/`tn-03`'s current shape).
- **Topic pivot mid-thread** — user opens one request, then switches to a different, still-under-
  specified one; tests whether the judge anchors on the *current* request rather than
  accumulated-but-stale context from the first.
- **Multi-party noise** — a thread with more than one human participant, where only one person's
  message is the live request and the others are unrelated chatter; tests context isolation on the
  fallback path specifically (§7's `tn-06` covers "no request at all," not "a request buried in
  noise from other speakers").
- **Chained unanswered clarification** — the assistant asks two clarifying questions in sequence;
  the user answers only the second, leaving the first implicitly still open. A deeper variant of
  `tn-05`'s "unanswered question" hazard, testing whether partial credit for *a* reply is
  mistaken for a complete one.
- **Boundary on the fallback path with a partial multi-part request** — the `bd-04` shape (one
  sub-request researchable, one not) replayed as a transcript rather than a pre-extracted
  `understanding` object.

§6 drafts two of these as an illustration.

## 5. The independence problem — resolved 2026-08-20 (descope), analysis kept for the record

**2026-08-20 decision, by the user: DROPPED.** No second labeler is available. The `boundary`
tier will **not** get independent-labeler validation. It is now sourced identically to
`clear_advance`/`clear_suspend` — option (a) below (LLM-drafted candidate rows with proposed
labels/rationales, flagged not-yet-verified, for a human/reviewer spot-check before merge) —
applied uniformly across all three strata, not just the two "clear by construction" ones.

**This is a deliberate, explicit descope from the backlog item's stated intent, not a quiet
scope-narrowing.** The backlog item asked for boundary-tier *independence* specifically because
(as the analysis below spells out) boundary labels are policy calls, not extractable facts — a
second same-model or single-reviewer pass over my drafted label does not test whether a human
independently agrees with the bias-to-suspend stance, it tests whether a reviewer accepts what
they were shown first (anchoring). Dropping the requirement means: **a boundary-tier row's label
now carries no more validation than a `clear_advance`/`clear_suspend` row's does** — a single
human spot-check of an LLM-drafted proposal, not the second-independent-human-labeler process the
backlog envisioned and this note's original §5(c) recommended. The fixture's boundary tier is
therefore single-labeled throughout (see §7 step 4's updated provenance language), and the
percent-agreement report §5(c) proposed as the deliverable of an independent pass will not exist —
there is no second labeling to compare against. The rest of this section is retained as the
analysis that produced that framing, for context on what the descope actually gives up.

---

The backlog's ask is "a second labeler for the boundary tier" for independence. I want to be
precise about what "independence" is actually protecting against here, because it is not the same
problem in every stratum, and collapsing them into one answer would misserve the ask.

**The `clear_advance`/`clear_suspend` tiers are fact-extraction labeling.** The archived protocol's
own position (§10 risk 3) is that these strata are "clear by construction" — a case like `cs-01`
("no service, no symptom, no timing — nothing researchable") does not require judgment calibrated
against a policy; it requires someone to correctly read what information is and isn't present. An
LLM drafting candidate rows here — including proposed labels — is a low-risk labeling task,
closer to a drafting accelerant than to the "second labeler" the backlog is asking for, **provided
a human spot-checks before the rows are treated as ground truth.**

**The `boundary` tier is a policy-judgment task, not a fact-extraction task, and this is where the
backlog's ask actually bites.** Every existing boundary row's `label_rationale` says so explicitly
(`bd-01`: "a reasonable human could argue either way … labeled false per the designed
bias-to-suspend operating point"). The disagreement the "second labeler" is meant to surface is a
disagreement about **how conservative the guard should be**, not a disagreement about what the
input contains. No LLM — mine, a different model family, or a hypothetical distinct-provider model
— resolves that disagreement, because it isn't a fact the judge got right or wrong; it's a design
stance. Recruiting a second LLM to "independently" apply the same bias-to-suspend policy the first
labeler (a human, so far) already applied by hand does not test whether humans agree with the
policy; it tests whether two LLMs can reproduce a policy statement, which is a different, less
useful question.

Laying out the brief's three options honestly, against that split:

- **(a) I draft candidate rows with proposed labels/rationales, flagged not-yet-verified, for
  human review before they enter the fixture.** Sound for `clear_advance`/`clear_suspend` (low
  ambiguity, spot-check is proportionate). **Insufficient alone for `boundary`** — a human review
  pass over my drafted labels is not the same as an *independent* second labeling; if the reviewer
  only accepts/rejects my proposed label, anchoring bias means the "independence" is compromised
  even with a human in the loop, because they saw my answer first.
- **(b) A second, distinct LLM (different provider/model family) labels independently.** Does not
  solve the boundary problem for the reason above — a second LLM does not carry policy authority
  regardless of provider distance. It could partially de-risk the *clear-tier* drafting (a
  cross-check that two different models extract the same facts), but even there only partially: if
  the "distinct" model is Ministral (the only other model already loaded in this lab's LM Studio,
  per the coordination doc's baseline table), it is a demonstrably *weaker* judge on this exact
  condition (D13: advance-recall 0.364 vs. Qwen's 0.818, pre-fix) — recruiting a weaker labeler to
  validate ground truth is backwards, not independence. A genuinely distinct, comparably-capable
  model (e.g. a cloud API model) would need provisioning: this lab currently has **no cloud API key
  available** for any component (noted in K-042's own QA report). That provisioning is itself a
  decision + cost the user would need to authorize; I am not assuming it.
- **(c, my recommendation) Split by tier.** `clear_advance`/`clear_suspend`: option (a) — I draft,
  a human spot-checks before merge, proportionate to how low-ambiguity these strata already are by
  design. `boundary`: the backlog's actual ask — a **second independent human labeler**, applying
  the same bias-to-suspend policy stance the first labeler used, **without seeing the first
  labeler's rationale**, then a plain percent-agreement report (not κ — n=15–20 is too small for a
  chance-corrected statistic to mean anything, and the archived note's own §3.3 finding about κ's
  prevalence-sensitivity applies here too). I can draft the boundary tier's candidate *scenarios*
  (the situations, not the labels) for a human to label fresh — that's a proportionate use of an
  LLM (scenario generation) that doesn't compromise the independence the labeling itself needs.

**What this section was not deciding, at the time it was written:** whether the user (or another
designated person) was willing to spend the time being that second boundary labeler, whether to
provision a cloud model for a clear-tier cross-check, and whether 15 boundary cases was the right
amount of that person's time to ask for. **2026-08-20: the first question is now answered — no
such person is available, hence the drop above.** The cloud-model-provisioning question was never
picked up (no cloud API key exists in this lab per F5/§9) and is now moot for the boundary tier
specifically, since option (a) does not require a second model at all; it remains a live question
only if the team separately wants a cross-check on the clear-tier drafting, which nobody has asked
for as part of this closure.

## 6. Full draft rows — 59 new rows, NOT committed to the fixture, unverified

**2026-08-20: this section is now the actual deliverable of the finalization pass**, replacing the
2-row illustration the earlier revision carried. All 59 rows needed to grow
`server/tests/eval/golden_guards.jsonl` from 26 → 85 per §3.4's composition table, drafted in the
existing schema (`id`, `tier`, `path`, `r1_probe`, `condition`, `understanding`, `turns`,
`expected`, `label_rationale`). **Draft only. Not independently verified. Not written to
`golden_guards.jsonl`.** Every `label_rationale` below plainly states its reasoning as a proposed
reading (the two carried-over rows, `tn-08`/`tn-09`, additionally carry the literal
`DRAFT/UNVERIFIED` tag from the earlier illustration) — none of these are ground truth until a
human spot-checks them per §5(c), now applied uniformly to all three tiers including `boundary`
(§5, §10). `tn-08` and `tn-09` are carried over unchanged from the prior revision's illustration
(they already fit the target composition below); everything else is new this pass.

**`r1_probe` semantics (2026-08-20 addition, analyst review):** this field is *not* a general
"adversarial/tricky case" flag — per the archived protocol (`docs/archive/plans/
m3-guard-calibration.md` §4.3) and `guard_calibration.py`'s `coercion_flip_rate(cases,
r1_probe_only=True)`, it marks only cases whose correct answer is **advance** and whose phrasing
risks tripping the `_NEGATION_CUES` coercion logic into a false suspend. It is **not** a substitute
for the `MATERIALITY PROBE`/`ADVERSARIAL`/`PARTIAL-ANCHOR PROBE` labels used in several
`label_rationale`s below, which describe a different, broader kind of "this case is designed to
catch a specific mistake" — those labels appear on rows across all three tiers, but `r1_probe:
true` may only appear on `clear_advance` rows (matching the real fixture's own 5 existing
`r1_probe: true` rows, all `clear_advance`). All 59 rows below were corrected to this rule before
merge (two `clear_suspend` rows, `cs-10`/`cs-13`, had been mismarked `r1_probe: true` in an earlier
draft of this section and are fixed below).

**Row count reconciliation** (existing count from `golden_guards.jsonl` F1 + new rows below =
§3.4 target):

| stratum | path | existing | new (this section) | target |
|---|---|---|---|---|
| `clear_advance` | `understanding` | 8 | 10 (`ca-09`…`ca-18`) | 18 |
| `clear_advance` | `turns` | 3 | 9 (`tn-08`, `tn-10`…`tn-17`) | 12 |
| `clear_suspend` | `understanding` | 7 | 17 (`cs-08`…`cs-24`) | 24 |
| `clear_suspend` | `turns` | 3 | 13 (`tn-09`, `tn-18`…`tn-29`) | 16 |
| `boundary` | `understanding` | 4 | 5 (`bd-05`…`bd-09`) | 9 |
| `boundary` | `turns` | 1 | 5 (`tn-30`…`tn-34`) | 6 |
| **total** | | **26** | **59** | **85** |

New rows drafted this pass (excluding the 2 carried-over `tn-08`/`tn-09`): 57. Total new rows this
section adds to the count above: 59 (10+9+17+13+5+5, with `tn-08`/`tn-09` counted once each in
their row).

### 6.1 `clear_advance` — new `understanding`-path rows (10)

```json
{"id": "ca-09", "tier": "clear_advance", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Explain why the nightly search-index rebuild is taking longer than usual", "known": ["job: nightly search-index rebuild", "normal duration: ~40 minutes", "current duration: ~3 hours", "started: 2026-07-14 02:00 UTC", "environment: production"], "missing": []}, "turns": [], "expected": true, "label_rationale": "Job, normal vs. observed duration, start time and environment are all present; nothing material is absent."}
{"id": "ca-10", "tier": "clear_advance", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Find out why users can't reset their password", "known": ["feature: password reset", "symptom: the reset email never arrives", "affected: all users", "since: 2026-07-12", "service: notifications"], "missing": []}, "turns": [], "expected": true, "label_rationale": "Feature, symptom, blast radius, onset and owning service are all present. Fully researchable."}
{"id": "ca-11", "tier": "clear_advance", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Investigate elevated 5xx errors on the payments-gateway service", "known": ["service: payments-gateway", "error class: 5xx", "rate: ~4% of requests", "started: 2026-07-16 03:00 UTC, right after deploy 9021"], "missing": []}, "turns": [], "expected": true, "label_rationale": "Service, error class, rate and a candidate trigger (deploy id + time) are all present."}
{"id": "ca-12", "tier": "clear_advance", "path": "understanding", "r1_probe": true, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Summarize what changed in the last three deploys to the inventory service", "known": ["service: inventory", "scope: last three deploys", "deploy ids: 9001, 9002, 9003"], "missing": ["whether the user wants a technical or executive-level summary"]}, "turns": [], "expected": true, "label_rationale": "MATERIALITY PROBE: the missing item shapes the ANSWER's audience/format, not what can be researched. Service and the exact deploy ids are all present."}
{"id": "ca-13", "tier": "clear_advance", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Find out why the nightly backup for the orders database failed", "known": ["job: nightly-backup", "database: orders", "error: 'disk quota exceeded'", "date: 2026-07-15"], "missing": []}, "turns": [], "expected": true, "label_rationale": "Job, target database, verbatim error and date are all present. Nothing missing."}
{"id": "ca-14", "tier": "clear_advance", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Check whether the rate limiter is misconfigured for the mobile client", "known": ["component: rate limiter", "client: mobile", "symptom: legitimate users getting 429s", "current threshold: 100 req/min per user"], "missing": []}, "turns": [], "expected": true, "label_rationale": "Component, client, symptom and the current configured threshold are all present."}
{"id": "ca-15", "tier": "clear_advance", "path": "understanding", "r1_probe": true, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Explain the retry policy the webhook delivery service uses", "known": ["service: webhook delivery", "the user wants the current documented policy, not a specific incident"], "missing": ["the specific webhook endpoint they're debugging"]}, "turns": [], "expected": true, "label_rationale": "MATERIALITY PROBE: a general documented-policy question is fully researchable from the service name alone; a specific endpoint would only personalize the answer, not enable the research."}
{"id": "ca-16", "tier": "clear_advance", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Find out why the Slack integration stopped posting alerts", "known": ["integration: Slack alerts", "channel: #incidents", "symptom: no messages posted since 2026-07-14 10:00 UTC", "last successful post: 09:58 UTC"], "missing": []}, "turns": [], "expected": true, "label_rationale": "Integration, channel, exact onset and last-known-good time are all present."}
{"id": "ca-17", "tier": "clear_advance", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Investigate why the CI pipeline is flaking on the integration-tests stage", "known": ["pipeline: CI", "stage: integration-tests", "failure rate: ~30% over the last week", "error in logs: 'connection reset by peer'"], "missing": []}, "turns": [], "expected": true, "label_rationale": "Pipeline, stage, failure rate and the recurring error text are all present."}
{"id": "ca-18", "tier": "clear_advance", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Find out why disk usage on the analytics warehouse jumped this week", "known": ["system: analytics warehouse", "metric: disk usage", "change: +40% since 2026-07-13", "no known config change reported"], "missing": []}, "turns": [], "expected": true, "label_rationale": "System, metric, magnitude, and window are all present; the absence of a known cause is itself the research question, not a missing input."}
```

### 6.2 `clear_advance` — new `turns`-path rows (9, including carried-over `tn-08`)

```json
{"id": "tn-08", "tier": "clear_advance", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn08-1", "text": "Getting a weird error on the billing page.", "role": "user", "createdAt": 1752571000000, "authorId": "u-erin", "displayName": "Erin", "authorType": ["User"]}, {"msgId": "m-tn08-2", "text": "What page exactly, and what's the error?", "role": "assistant", "createdAt": 1752571001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn08-3", "text": "The invoices tab. It's a blank white screen, no error text visible.", "role": "user", "createdAt": 1752571002000, "authorId": "u-erin", "displayName": "Erin", "authorType": ["User"]}, {"msgId": "m-tn08-4", "text": "Which browser, and does it happen every time or intermittently?", "role": "assistant", "createdAt": 1752571003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn08-5", "text": "Chrome, and yeah every time since this morning around 9am.", "role": "user", "authorId": "u-erin", "createdAt": 1752571004000, "displayName": "Erin", "authorType": ["User"]}], "expected": true, "label_rationale": "DRAFT/UNVERIFIED: multi-turn aggregation — page, symptom (blank screen, no visible error), browser, onset time and reproducibility are all present but arrive across five short turns rather than one clean exchange. Tests whether the judge aggregates dispersed evidence rather than crediting only the most recent turn."}
{"id": "tn-10", "tier": "clear_advance", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn10-1", "text": "Deploy 9021 to payments-gateway seems to have caused problems.", "role": "user", "createdAt": 1752580000000, "authorId": "u-quinn", "displayName": "Quinn", "authorType": ["User"]}, {"msgId": "m-tn10-2", "text": "What kind of problems, and roughly what error rate are you seeing?", "role": "assistant", "createdAt": 1752580001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn10-3", "text": "5xx errors, maybe 4% of requests.", "role": "user", "createdAt": 1752580002000, "authorId": "u-quinn", "displayName": "Quinn", "authorType": ["User"]}, {"msgId": "m-tn10-4", "text": "When did it start, and is it still ongoing?", "role": "assistant", "createdAt": 1752580003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn10-5", "text": "Right after the deploy this morning around 03:00 UTC, still happening now.", "role": "user", "createdAt": 1752580004000, "authorId": "u-quinn", "displayName": "Quinn", "authorType": ["User"]}], "expected": true, "label_rationale": "FALLBACK PATH / MULTI-TURN AGGREGATION: service, deploy id, error class, rate, onset and ongoing status are all present but arrive across five turns rather than one exchange. Tests whether the judge aggregates dispersed evidence."}
{"id": "tn-11", "tier": "clear_advance", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn11-1", "text": "Can you check why the search index rebuild is slow?", "role": "user", "createdAt": 1752581000000, "authorId": "u-rosa", "displayName": "Rosa", "authorType": ["User"]}, {"msgId": "m-tn11-2", "text": "Sure — do you have timing details on that?", "role": "assistant", "createdAt": 1752581001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn11-3", "text": "Actually never mind that, different thing: users can't reset their password, the reset email never arrives, been broken since Monday for everyone.", "role": "user", "createdAt": 1752581002000, "authorId": "u-rosa", "displayName": "Rosa", "authorType": ["User"]}], "expected": true, "label_rationale": "FALLBACK PATH / TOPIC PIVOT: the user abandons the first, underspecified topic mid-thread and switches to a second, fully-specified one. Tests that the judge evaluates the CURRENT request rather than anchoring on the earlier, unresolved one."}
{"id": "tn-12", "tier": "clear_advance", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn12-1", "text": "hey team", "role": "user", "createdAt": 1752582000000, "authorId": "u-grace", "displayName": "Grace", "authorType": ["User"]}, {"msgId": "m-tn12-2", "text": "sup", "role": "user", "createdAt": 1752582001000, "authorId": "u-hank", "displayName": "Hank", "authorType": ["User"]}, {"msgId": "m-tn12-3", "text": "btw the webhook delivery service stopped posting alerts to #incidents since 10:00 UTC today, last successful post was 09:58", "role": "user", "createdAt": 1752582002000, "authorId": "u-grace", "displayName": "Grace", "authorType": ["User"]}, {"msgId": "m-tn12-4", "text": "oh nice, weather's good today", "role": "user", "createdAt": 1752582003000, "authorId": "u-hank", "displayName": "Hank", "authorType": ["User"]}], "expected": true, "label_rationale": "FALLBACK PATH / MULTI-PARTY NOISE: a fully-specified request from one speaker (Grace) is surrounded by unrelated chatter from another (Hank). Tests context isolation on the advance side — the noise must not suppress a genuinely researchable request."}
{"id": "tn-13", "tier": "clear_advance", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn13-1", "text": "The rate limiter seems off for mobile users.", "role": "user", "createdAt": 1752583000000, "authorId": "u-sam", "displayName": "Sam", "authorType": ["User"]}, {"msgId": "m-tn13-2", "text": "Which endpoint, and what threshold are you seeing it trigger at?", "role": "assistant", "createdAt": 1752583001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn13-3", "text": "The /search endpoint.", "role": "user", "createdAt": 1752583002000, "authorId": "u-sam", "displayName": "Sam", "authorType": ["User"]}, {"msgId": "m-tn13-4", "text": "And is it kicking in at the normal 100 req/min limit, or lower?", "role": "assistant", "createdAt": 1752583003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn13-5", "text": "Lower — looks like it's kicking in around 40 req/min instead of 100.", "role": "user", "createdAt": 1752583004000, "authorId": "u-sam", "displayName": "Sam", "authorType": ["User"]}], "expected": true, "label_rationale": "FALLBACK PATH: two sequential clarifying questions, both fully answered by the end of the transcript. Endpoint and the observed vs. configured threshold are both present."}
{"id": "tn-14", "tier": "clear_advance", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn14-1", "text": "Can you check why the nightly backup for the orders database failed? Error was 'disk quota exceeded', happened on 2026-07-15.", "role": "user", "createdAt": 1752584000000, "authorId": "u-tom", "displayName": "Tom", "authorType": ["User"]}], "expected": true, "label_rationale": "FALLBACK PATH: a single self-contained turn that fully specifies the request, mirroring tn-02's easiest-possible-case design."}
{"id": "tn-15", "tier": "clear_advance", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn15-1", "text": "Getting logged out randomly.", "role": "user", "createdAt": 1752585000000, "authorId": "u-uma", "displayName": "Uma", "authorType": ["User"]}, {"msgId": "m-tn15-2", "text": "Which client — web or mobile — and how often does it happen?", "role": "assistant", "createdAt": 1752585001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn15-3", "text": "Mobile app, maybe every 20 minutes.", "role": "user", "createdAt": 1752585002000, "authorId": "u-uma", "displayName": "Uma", "authorType": ["User"]}, {"msgId": "m-tn15-4", "text": "Since when did you start noticing this?", "role": "assistant", "createdAt": 1752585003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn15-5", "text": "Since the update that went out yesterday around noon.", "role": "user", "createdAt": 1752585004000, "authorId": "u-uma", "displayName": "Uma", "authorType": ["User"]}], "expected": true, "label_rationale": "FALLBACK PATH / MULTI-TURN AGGREGATION: client, frequency and a candidate trigger (an update) plus its timing all arrive across five turns."}
{"id": "tn-16", "tier": "clear_advance", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn16-1", "text": "Question about the retry policy for webhooks.", "role": "user", "createdAt": 1752586000000, "authorId": "u-vic", "displayName": "Vic", "authorType": ["User"]}, {"msgId": "m-tn16-2", "text": "Sure, what would you like to know?", "role": "assistant", "createdAt": 1752586001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn16-3", "text": "Actually forget that — separate issue, the Slack integration stopped posting alerts since 10am today to #incidents, last post was 9:58am.", "role": "user", "createdAt": 1752586002000, "authorId": "u-vic", "displayName": "Vic", "authorType": ["User"]}], "expected": true, "label_rationale": "FALLBACK PATH / TOPIC PIVOT: the first, vague topic is explicitly abandoned; the second, fully-specified one is what the judge must evaluate."}
{"id": "tn-17", "tier": "clear_advance", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn17-1", "text": "morning", "role": "user", "createdAt": 1752587000000, "authorId": "u-ivy", "displayName": "Ivy", "authorType": ["User"]}, {"msgId": "m-tn17-2", "text": "morning, coffee's out again lol", "role": "user", "createdAt": 1752587001000, "authorId": "u-jack", "displayName": "Jack", "authorType": ["User"]}, {"msgId": "m-tn17-3", "text": "anyway — CI pipeline is flaking on the integration-tests stage, about 30% failure rate this week, logs show 'connection reset by peer'", "role": "user", "createdAt": 1752587002000, "authorId": "u-ivy", "displayName": "Ivy", "authorType": ["User"]}, {"msgId": "m-tn17-4", "text": "ugh annoying", "role": "user", "createdAt": 1752587003000, "authorId": "u-jack", "displayName": "Jack", "authorType": ["User"]}], "expected": true, "label_rationale": "FALLBACK PATH / MULTI-PARTY NOISE: pipeline, stage, failure rate and error text are all present from one speaker despite unrelated banter from another."}
```

### 6.3 `clear_suspend` — new `understanding`-path rows (17)

```json
{"id": "cs-08", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The dashboard looks weird", "known": [], "missing": ["which dashboard", "what specifically looks wrong", "when it started"]}, "turns": [], "expected": false, "label_rationale": "No dashboard identified and no concrete symptom. Nothing researchable."}
{"id": "cs-09", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The user wants to know if everything is okay", "known": [], "missing": ["what system or process they're asking about"]}, "turns": [], "expected": false, "label_rationale": "No target system named at all. A status check on nothing in particular is not researchable."}
{"id": "cs-10", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The deploy needs to be checked", "known": [], "missing": []}, "turns": [], "expected": false, "label_rationale": "ADVERSARIAL / TRUST PROBE: `missing` is empty but the understanding is vacuous — no deploy id, service, or symptom is named. A judge trusting `missing == []` as sufficiency advances here and is wrong, mirroring cs-04."}
{"id": "cs-11", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The user is asking about performance", "known": ["topic: performance"], "missing": ["which service", "what metric", "what the observed problem is"]}, "turns": [], "expected": false, "label_rationale": "'Performance' names a category, not a target, exactly like cs-03's 'the API'."}
{"id": "cs-12", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Can someone look into the thing from yesterday's standup", "known": ["reference: yesterday's standup"], "missing": ["what 'the thing' actually refers to"]}, "turns": [], "expected": false, "label_rationale": "An unresolved anaphoric reference, the referent itself is the research input and it is absent — same shape as cs-06."}
{"id": "cs-13", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The user has an issue with the notifications service", "known": ["service: notifications"], "missing": ["what the issue actually is"]}, "turns": [], "expected": false, "label_rationale": "PARTIAL-ANCHOR PROBE: a concrete service name is present but the request itself is absent, tests whether one confident-looking fact tips the judge into advancing, mirroring cs-07."}
{"id": "cs-14", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The user wants help but hasn't described the problem yet", "known": [], "missing": ["the problem itself"]}, "turns": [], "expected": false, "label_rationale": "Explicitly no problem description yet. The canonical suspend, restated."}
{"id": "cs-15", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Something changed after the last release and it's causing issues", "known": ["reference: the last release"], "missing": ["which service", "what the issue is", "the release version"]}, "turns": [], "expected": false, "label_rationale": "A vague causal claim with no service, symptom, or version pinned down."}
{"id": "cs-16", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The user is upset about the response times", "known": ["the user is frustrated", "topic mentioned: response times"], "missing": ["which service or endpoint", "the actual observed numbers", "when it started"]}, "turns": [], "expected": false, "label_rationale": "Affective context plus a bare topic mention, no research input. Same shape as cs-05."}
{"id": "cs-17", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The customer reported a bug in the app", "known": ["there is a bug report"], "missing": ["which app feature", "what the bug is", "reproduction steps"]}, "turns": [], "expected": false, "label_rationale": "A bug report exists but nothing about the bug itself is present."}
{"id": "cs-18", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The team wants an update on the migration", "known": ["topic: 'the migration'"], "missing": ["which migration — several are ongoing", "what kind of update is wanted"]}, "turns": [], "expected": false, "label_rationale": "'The migration' is ambiguous among several ongoing efforts; nothing disambiguates which."}
{"id": "cs-19", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Need help with the config", "known": ["topic: config"], "missing": ["which service's config", "what's wrong with it"]}, "turns": [], "expected": false, "label_rationale": "'The config' names nothing specific enough to research."}
{"id": "cs-20", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The user asked a follow-up but it's unclear which thread it continues", "known": ["it is a follow-up message"], "missing": ["which prior conversation or topic it refers to"]}, "turns": [], "expected": false, "label_rationale": "An unresolved reference to an unspecified prior context, the referent is missing."}
{"id": "cs-21", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The alert fired again", "known": ["an alert fired"], "missing": ["which alert", "which system", "whether this is new information or a repeat report"]}, "turns": [], "expected": false, "label_rationale": "'An alert' is not a research target without knowing which one or what system it covers."}
{"id": "cs-22", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The user wants the numbers", "known": [], "missing": ["which numbers", "for what system or time period"]}, "turns": [], "expected": false, "label_rationale": "'The numbers' with no referent at all."}
{"id": "cs-23", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "There's a problem with permissions", "known": ["topic: permissions"], "missing": ["which resource", "which user or role", "what the symptom is — denied vs. over-granted"]}, "turns": [], "expected": false, "label_rationale": "'Permissions' names a category; neither the resource nor the direction of the problem is present."}
{"id": "cs-24", "tier": "clear_suspend", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The job didn't run", "known": ["there is a job that failed to run"], "missing": ["which job", "scheduled for when", "any error output"]}, "turns": [], "expected": false, "label_rationale": "'The job' is unidentified; nothing to research yet."}
```

### 6.4 `clear_suspend` — new `turns`-path rows (13, including carried-over `tn-09`)

```json
{"id": "tn-09", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn09-1", "text": "The export job keeps timing out.", "role": "user", "createdAt": 1752572000000, "authorId": "u-frank", "displayName": "Frank", "authorType": ["User"]}, {"msgId": "m-tn09-2", "text": "Is that the CSV export or the PDF export, and roughly how big is the dataset?", "role": "assistant", "createdAt": 1752572001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn09-3", "text": "It's pretty large, a few hundred thousand rows I think.", "role": "user", "createdAt": 1752572002000, "authorId": "u-frank", "displayName": "Frank", "authorType": ["User"]}, {"msgId": "m-tn09-4", "text": "Got it — and which export format, CSV or PDF?", "role": "assistant", "createdAt": 1752572003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}], "expected": false, "label_rationale": "DRAFT/UNVERIFIED: chained unanswered clarification — two clarifying questions were asked; the user answered only the first (dataset size) and the second (CSV vs PDF) remains open in the last turn. A judge that credits 'the user replied' without checking which question was actually answered would wrongly advance here — the deeper variant of tn-05's single-unanswered-question hazard."}
{"id": "tn-18", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn18-1", "text": "The export job keeps failing.", "role": "user", "createdAt": 1752588000000, "authorId": "u-frank", "displayName": "Frank", "authorType": ["User"]}, {"msgId": "m-tn18-2", "text": "Is that the CSV or PDF export?", "role": "assistant", "createdAt": 1752588001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn18-3", "text": "CSV.", "role": "user", "createdAt": 1752588002000, "authorId": "u-frank", "displayName": "Frank", "authorType": ["User"]}, {"msgId": "m-tn18-4", "text": "And do you have the error message it shows?", "role": "assistant", "createdAt": 1752588003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn18-5", "text": "Not off the top of my head, I'd have to look.", "role": "user", "createdAt": 1752588004000, "authorId": "u-frank", "displayName": "Frank", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH / MULTI-TURN AGGREGATION, STILL INSUFFICIENT: five turns narrow the export type but the critical error text is explicitly not available. More turns does not automatically mean enough evidence."}
{"id": "tn-19", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn19-1", "text": "The checkout service is throwing 502s, v2.14.0, since 09:20 UTC, about 18% of requests.", "role": "user", "createdAt": 1752589000000, "authorId": "u-alice", "displayName": "Alice", "authorType": ["User"]}, {"msgId": "m-tn19-2", "text": "Got it, looking into that.", "role": "assistant", "createdAt": 1752589001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn19-3", "text": "Oh also — separate thing, something's off with the API too.", "role": "user", "createdAt": 1752589002000, "authorId": "u-alice", "displayName": "Alice", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH / TOPIC PIVOT, STALE-CONTEXT HAZARD: the first topic is fully specified, but the second ('the API', 'something's off') is not. Tests whether the judge wrongly credits the second request with the first's completeness."}
{"id": "tn-20", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn20-1", "text": "morning", "role": "user", "createdAt": 1752590000000, "authorId": "u-karen", "displayName": "Karen", "authorType": ["User"]}, {"msgId": "m-tn20-2", "text": "hey", "role": "user", "createdAt": 1752590001000, "authorId": "u-leo", "displayName": "Leo", "authorType": ["User"]}, {"msgId": "m-tn20-3", "text": "something's acting up with the service, not sure what exactly", "role": "user", "createdAt": 1752590002000, "authorId": "u-karen", "displayName": "Karen", "authorType": ["User"]}, {"msgId": "m-tn20-4", "text": "yeah I noticed too", "role": "user", "createdAt": 1752590003000, "authorId": "u-leo", "displayName": "Leo", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH / MULTI-PARTY NOISE, INSUFFICIENT: a request-shaped message exists but is itself vague — no service name, no concrete symptom. Noise isolation alone does not make this researchable."}
{"id": "tn-21", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn21-1", "text": "The export job keeps timing out.", "role": "user", "createdAt": 1752591000000, "authorId": "u-frank", "displayName": "Frank", "authorType": ["User"]}, {"msgId": "m-tn21-2", "text": "Is that the CSV or PDF export?", "role": "assistant", "createdAt": 1752591001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn21-3", "text": "Hmm, not sure offhand.", "role": "user", "createdAt": 1752591002000, "authorId": "u-frank", "displayName": "Frank", "authorType": ["User"]}, {"msgId": "m-tn21-4", "text": "No worries — roughly how many rows are we talking about, ballpark?", "role": "assistant", "createdAt": 1752591003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn21-5", "text": "A few hundred thousand, I think.", "role": "user", "createdAt": 1752591004000, "authorId": "u-frank", "displayName": "Frank", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH / CHAINED UNANSWERED CLARIFICATION: the format question (Q1) is never answered ('not sure offhand'); only the later ballpark-size question (Q2) gets a reply. A judge crediting 'the user replied' without checking WHICH question was answered advances wrongly here — the deeper variant of tn-05/tn-09's single-unanswered-question hazard, this time with the thread continuing past the gap rather than ending on it."}
{"id": "tn-22", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn22-1", "text": "Users are getting logged out randomly.", "role": "user", "createdAt": 1752592000000, "authorId": "u-uma", "displayName": "Uma", "authorType": ["User"]}, {"msgId": "m-tn22-2", "text": "Which client is this on — web or mobile — and roughly how often?", "role": "assistant", "createdAt": 1752592001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}], "expected": false, "label_rationale": "FALLBACK PATH / UNANSWERED-QUESTION PROBE: the transcript ends on the assistant's own clarifying question — the user has not replied. Direct mirror of tn-05 in a different domain."}
{"id": "tn-23", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn23-1", "text": "hi", "role": "user", "createdAt": 1752593000000, "authorId": "u-dana", "displayName": "Dana", "authorType": ["User"]}, {"msgId": "m-tn23-2", "text": "Hi! How can I help you today?", "role": "assistant", "createdAt": 1752593001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn23-3", "text": "just checking in", "role": "user", "createdAt": 1752593002000, "authorId": "u-dana", "displayName": "Dana", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH: a slightly longer exchange than tn-04 but still no request at all — 'just checking in' is explicitly not a task."}
{"id": "tn-24", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn24-1", "text": "Things feel slow lately.", "role": "user", "createdAt": 1752594000000, "authorId": "u-ben", "displayName": "Ben", "authorType": ["User"]}, {"msgId": "m-tn24-2", "text": "Slow how — page loads, API responses, something else? And which part of the product?", "role": "assistant", "createdAt": 1752594001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn24-3", "text": "Just generally sluggish I guess.", "role": "user", "createdAt": 1752594002000, "authorId": "u-ben", "displayName": "Ben", "authorType": ["User"]}, {"msgId": "m-tn24-4", "text": "Do you have any specific page or feature in mind, or a rough time it started?", "role": "assistant", "createdAt": 1752594003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn24-5", "text": "Not really, just an overall feeling.", "role": "user", "createdAt": 1752594004000, "authorId": "u-ben", "displayName": "Ben", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH / MULTI-TURN AGGREGATION THAT NEVER CONVERGES: five turns of clarification produce no scoped target, no metric, and no time window. Tests that turn count alone is not mistaken for evidence."}
{"id": "tn-25", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn25-1", "text": "anyone know why the deploys have been slow this week", "role": "user", "createdAt": 1752595000000, "authorId": "u-mia", "displayName": "Mia", "authorType": ["User"]}, {"msgId": "m-tn25-2", "text": "haha yeah no idea", "role": "user", "createdAt": 1752595001000, "authorId": "u-noah", "displayName": "Noah", "authorType": ["User"]}, {"msgId": "m-tn25-3", "text": "lol whatever, not urgent", "role": "user", "createdAt": 1752595002000, "authorId": "u-mia", "displayName": "Mia", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH / MULTI-PARTY NOISE: a request-shaped question is asked and then explicitly waved off as not urgent, with no service, deploy id, or symptom ever named. Casual banter mistaken for an active request is the hazard being tested."}
{"id": "tn-26", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn26-1", "text": "Payments gateway 5xx errors, started after deploy 9021 this morning around 03:00 UTC, about 4% of requests.", "role": "user", "createdAt": 1752596000000, "authorId": "u-quinn", "displayName": "Quinn", "authorType": ["User"]}, {"msgId": "m-tn26-2", "text": "Thanks, looking into it.", "role": "assistant", "createdAt": 1752596001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn26-3", "text": "Also can you check on the other service too, similar issue maybe?", "role": "user", "createdAt": 1752596002000, "authorId": "u-quinn", "displayName": "Quinn", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH / TOPIC PIVOT, STALE-CONTEXT HAZARD: 'the other service' names no service and 'similar issue maybe' names no symptom. The first topic's completeness must not bleed into the second."}
{"id": "tn-27", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn27-1", "text": "Getting weird behavior on the reporting dashboards.", "role": "user", "createdAt": 1752597000000, "authorId": "u-owen", "displayName": "Owen", "authorType": ["User"]}, {"msgId": "m-tn27-2", "text": "Which dashboard specifically, and what does 'weird' look like — wrong numbers, blank charts, errors?", "role": "assistant", "createdAt": 1752597001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn27-3", "text": "Not totally sure, it's just off somehow.", "role": "user", "createdAt": 1752597002000, "authorId": "u-owen", "displayName": "Owen", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH / CHAINED CLARIFICATION, NEITHER RESOLVED: two sub-questions asked in one turn, the reply answers neither concretely. Still nothing to research."}
{"id": "tn-28", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn28-1", "text": "The alert fired again.", "role": "user", "createdAt": 1752598000000, "authorId": "u-priya", "displayName": "Priya", "authorType": ["User"]}, {"msgId": "m-tn28-2", "text": "Which alert, and which system does it cover?", "role": "assistant", "createdAt": 1752598001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn28-3", "text": "The disk-usage alert, for the analytics warehouse.", "role": "user", "createdAt": 1752598002000, "authorId": "u-priya", "displayName": "Priya", "authorType": ["User"]}, {"msgId": "m-tn28-4", "text": "Got it — is this a new occurrence or something recurring, and any error details?", "role": "assistant", "createdAt": 1752598003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn28-5", "text": "Pretty sure it's fired before, don't have more details handy.", "role": "user", "createdAt": 1752598004000, "authorId": "u-priya", "displayName": "Priya", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH / MULTI-TURN AGGREGATION, STILL INSUFFICIENT: alert and system are identified across five turns, but whether this is a new, actionable occurrence and any concrete symptom detail remain unresolved."}
{"id": "tn-29", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn29-1", "text": "Actually nevermind, I figured it out.", "role": "user", "createdAt": 1752599000000, "authorId": "u-erin", "displayName": "Erin", "authorType": ["User"]}], "expected": false, "label_rationale": "FALLBACK PATH / WITHDRAWAL: explicit withdrawal of whatever request may have existed before this transcript window. No researchable content remains."}
```

### 6.5 `boundary` — new `understanding`-path rows (5)

```json
{"id": "bd-05", "tier": "boundary", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Find out why notifications are delayed", "known": ["service: notifications", "symptom: delayed delivery reported by several users"], "missing": ["how delayed — seconds vs. hours", "time window"]}, "turns": [], "expected": false, "label_rationale": "GENUINE BOUNDARY: service and a real symptom are present, but the magnitude and window that would separate 'noisy but researchable' from 'too vague to start' are absent. Policy-labeled false."}
{"id": "bd-06", "tier": "boundary", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Explain why the CI pipeline is flaky", "known": ["pipeline: CI", "symptom: intermittent failures"], "missing": ["which stage", "the error text"]}, "turns": [], "expected": false, "label_rationale": "GENUINE BOUNDARY: pipeline named, symptom real, but stage and error text may or may not be decisive for a useful investigation. Reasonable humans disagree. Policy-labeled false."}
{"id": "bd-07", "tier": "boundary", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Find the known issues for the reporting service the user is on", "known": ["service: reporting", "symptom: occasional timeouts"], "missing": ["version or deployment of the service"]}, "turns": [], "expected": false, "label_rationale": "GENUINE BOUNDARY: mirrors bd-03's shape for a different service — version is often decisive for a known-issues lookup but a general timeouts search may still be useful. Policy-labeled false."}
{"id": "bd-08", "tier": "boundary", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "The user asked two things: (a) why the nightly backup job failed, and (b) whether it affects the analytics dashboards", "known": ["job: nightly-backup", "error for (a): 'disk quota exceeded'"], "missing": ["which specific dashboards are meant in (b)"]}, "turns": [], "expected": false, "label_rationale": "GENUINE BOUNDARY / PARTIAL-REQUEST: one sub-request (a) is fully researchable, the other (b) is not, the same partial-request underdetermination bd-04 raised about the guard text itself. Policy-labeled false."}
{"id": "bd-09", "tier": "boundary", "path": "understanding", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {"request": "Find out why disk usage jumped on the analytics warehouse", "known": ["system: analytics warehouse", "symptom: disk usage increase"], "missing": ["exact percentage or amount", "time window"]}, "turns": [], "expected": false, "label_rationale": "GENUINE BOUNDARY: a real system and symptom, but no magnitude or window — the boundary companion to ca-18's fully-specified version of the same scenario."}
```

### 6.6 `boundary` — new `turns`-path rows (5)

```json
{"id": "tn-30", "tier": "boundary", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn30-1", "text": "Two things — why did deploy 8831 fail, and did it affect the reporting pipeline?", "role": "user", "createdAt": 1752600000000, "authorId": "u-cara", "displayName": "Cara", "authorType": ["User"]}, {"msgId": "m-tn30-2", "text": "For the deploy failure — do you have the error text?", "role": "assistant", "createdAt": 1752600001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn30-3", "text": "Yeah, 'migration lock timeout' on the orders table. Not sure about the reporting pipeline part though, which one specifically.", "role": "user", "createdAt": 1752600002000, "authorId": "u-cara", "displayName": "Cara", "authorType": ["User"]}], "expected": false, "label_rationale": "GENUINE BOUNDARY / PARTIAL-REQUEST on the fallback path: the deploy-failure sub-request (a) is now fully researchable (id + error text); the reporting-pipeline sub-request (b) is not. The bd-04 shape replayed as a transcript rather than a pre-extracted `understanding` object, per §4's fifth target shape. Policy-labeled false."}
{"id": "tn-31", "tier": "boundary", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn31-1", "text": "The orders endpoint has been slow.", "role": "user", "createdAt": 1752601000000, "authorId": "u-cara", "displayName": "Cara", "authorType": ["User"]}, {"msgId": "m-tn31-2", "text": "Which environment, and do you have before/after latency numbers?", "role": "assistant", "createdAt": 1752601001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn31-3", "text": "Production, and no exact numbers, but it's been noticeably worse since sometime this week.", "role": "user", "createdAt": 1752601002000, "authorId": "u-cara", "displayName": "Cara", "authorType": ["User"]}], "expected": false, "label_rationale": "GENUINE BOUNDARY: environment is pinned down but the time window ('sometime this week') is soft and no latency numbers were ever given. A reasonable human could argue either way. Policy-labeled false."}
{"id": "tn-32", "tier": "boundary", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn32-1", "text": "Search index rebuild is slow, started around 2am today, normally 40 min now taking hours.", "role": "user", "createdAt": 1752602000000, "authorId": "u-rosa", "displayName": "Rosa", "authorType": ["User"]}, {"msgId": "m-tn32-2", "text": "Thanks, on it.", "role": "assistant", "createdAt": 1752602001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn32-3", "text": "Also — the notifications feel delayed too, not sure by how much.", "role": "user", "createdAt": 1752602002000, "authorId": "u-rosa", "displayName": "Rosa", "authorType": ["User"]}], "expected": false, "label_rationale": "GENUINE BOUNDARY / TOPIC PIVOT: the first topic is fully specified; the second names a service and a real symptom (delay) but no magnitude or timeframe. Whether that's enough to start research is genuinely arguable. Policy-labeled false."}
{"id": "tn-33", "tier": "boundary", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn33-1", "text": "hey", "role": "user", "createdAt": 1752603000000, "authorId": "u-omar", "displayName": "Omar", "authorType": ["User"]}, {"msgId": "m-tn33-2", "text": "the CI pipeline's been flaky again, integration tests I think", "role": "user", "createdAt": 1752603001000, "authorId": "u-priya", "displayName": "Priya", "authorType": ["User"]}, {"msgId": "m-tn33-3", "text": "yeah noticed that too", "role": "user", "createdAt": 1752603002000, "authorId": "u-omar", "displayName": "Omar", "authorType": ["User"]}], "expected": false, "label_rationale": "GENUINE BOUNDARY / MULTI-PARTY NOISE: pipeline and stage are named ('I think') but no error text or failure rate is given. A real anchor exists but with real hedging around it. Policy-labeled false."}
{"id": "tn-34", "tier": "boundary", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn34-1", "text": "Permissions seem broken for some users.", "role": "user", "createdAt": 1752604000000, "authorId": "u-noah", "displayName": "Noah", "authorType": ["User"]}, {"msgId": "m-tn34-2", "text": "Which resource or role, and is it users being denied access or getting access they shouldn't?", "role": "assistant", "createdAt": 1752604001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn34-3", "text": "I think it's the reporting workspace, and it's more like people getting denied who shouldn't be, but only sometimes.", "role": "user", "createdAt": 1752604002000, "authorId": "u-noah", "displayName": "Noah", "authorType": ["User"]}], "expected": false, "label_rationale": "GENUINE BOUNDARY / CHAINED CLARIFICATION: resource and the direction of the issue are given, but hedged throughout ('I think', 'only sometimes'). Whether that hedged answer is researchable as stated is a real judgment call. Policy-labeled false."}
```

## 7. Downstream implementation plan (ready for `tdd-engineer` dispatch — §5's blocker is resolved)

**2026-08-20: both §10 decisions are closed; this plan is unconditional as of this revision.**

1. **Fixture content.** `server/tests/eval/golden_guards.jsonl` grows from 26 → 85 rows per §3.4's
   table, with **all three tiers** — `clear_advance`, `clear_suspend`, and now `boundary` too —
   sourced per §5(c)'s option-(a) path (my drafts + a single human spot-check before merge; no
   independent second labeler for `boundary`, per §5's 2026-08-20 descope). Every new row follows
   the existing schema exactly (`id`, `tier`, `path`, `r1_probe`, `condition`, `understanding`,
   `turns`, `expected`, `label_rationale`) — §6 now carries the full 59-row draft, not an
   illustration.
2. **`server/tests/eval/test_guard_calibration_live.py` — five literal constants to update** (F3):
   line 223 (`== 26` → `== 85`), line 227 (`== 26` → `== 85`), line 228 (`26 * K_REPLICATES` →
   `85 * K_REPLICATES`), line 232 (`== 21` → `== 70`, i.e. 30+40), line 233 (`== 5` → `== 15`). No
   other harness file needs a size-specific edit (F3) — `guard_calibration.py`'s metric functions
   already group dynamically by `tier`/`path`. (**2026-08-20 correction:** the last two line
   numbers were previously off by one — 232/233, not 233/234, re-confirmed directly against the
   current file; F3 above carries the same fix.)
3. **Recommended, not required: an offline structural-integrity test for the guard fixture**,
   mirroring `test_golden_set_integrity.py`'s pattern for the retrieval set (F4) — unique ids,
   required fields present, `tier`/`path`/`expected` enums valid, boundary rows always
   `expected: false`, and (optionally) minimum-per-stratum-and-path counts expressed as inequalities
   (`>= 40` clear_suspend, etc.) rather than exact literals, so the check doesn't need editing every
   time the fixture is extended again. This closes F4's gap and would have caught, offline and
   fast, any of the five literal-constant edits above being missed or mistyped.
4. **G1/G2 thresholds themselves: do not change them.** ≤10% FAR / ≥0.80 recall are the *design's*
   bias-to-suspend operating point (archived §3.1), not an artifact of small n — growing n makes
   the *measurement* of those thresholds trustworthy, it does not imply the thresholds should move.
   **What should change is the report template's canned §6.1 caveat sentence** (archived §8 item
   3), because at n=40/30/15 the claim it's disclaiming is a materially different claim than at
   n=10/11/5. Proposed replacement, for the report-writer to adopt or adapt:

   > This gate combines a statistically defensible bound (G1: n=40 `clear_suspend` cases; a
   > zero-observed-failure result bounds true false-advance rate ≤10% at 95% confidence via the
   > Wilson score interval — any single observed false-advance among the 40 fails the gate
   > outright, by design, not by insufficient power) with a moderate-power screen (G2: n=30
   > `clear_advance` cases, ±14pp CI at the observed point). This is a stronger claim than the
   > n=21/26 screen it replaces, but it is still not a full certification: the `boundary` tier
   > remains a policy call — single-labeled (LLM-drafted, human-spot-checked, same as the two
   > clear tiers; no independent second labeler was available — see `golden-set-expansion-ml.md`
   > §5, 2026-08-20), not ground truth and not independently cross-checked — and all three strata
   > are synthetic-but-realistic, not sampled from live traffic.

   **2026-08-20: the single-/dual-labeled branch below is resolved to single-labeled for all three
   tiers.** The fixture itself (or a small provenance file alongside it) should still record this
   plainly — "LLM-drafted candidate rows, human-spot-checked before merge; no independent second
   labeler" — for the same reason the brief's option (b) wanted a provenance header in the first
   place: so a future reader of the calibration report never mistakes the boundary tier's labels
   for anything sturdier than a spot-checked draft. There is no percent-agreement number to report,
   because there was no second independent labeling pass to compare against.
5. **Live run cost, for the user's awareness, not something I ran:** 85 cases × k=3 = 255 live
   judge calls against the local Qwen3-4B — same order of magnitude as the existing 78-call run,
   not a meaningful compute concern. The **labeling** cost (59 new rows, all sourced per §5(c)
   option (a) — LLM-drafted, single human spot-check, no second independent pass for any tier
   including `boundary`, per §5's 2026-08-20 descope) is the real cost this note is asking the user
   to authorize. (**2026-08-20 correction, analyst review:** this bullet previously read "at least
   15 of them needing a second independent human pass," a leftover from before the closure that
   directly contradicted §5/§8/§10's own record that no second labeler exists — fixed to match.)

## 8. Risks & open questions

1. **(RESOLVED 2026-08-20 — was: who labels the boundary tier, and is that person available?)**
   No second labeler is available; the user has dropped the independent-labeler requirement (§5).
   `boundary` is now sourced identically to the clear tiers (LLM-drafted, human-spot-checked). This
   no longer blocks §7 — but it is a real, recorded loss of validation strength for the tier whose
   labels are *policy* calls, not extracted facts (§5's own framing): a spot-check is not an
   independent second opinion, and the gap between the two is exactly what the original backlog
   ask was trying to close. Downstream readers of the calibration report should not read
   "spot-checked" as "independently validated."
2. **(RESOLVED 2026-08-20 — was: is n=40/30/15 the right size, vs. n=53's tolerance-for-one?)**
   The user chose n=40 (zero-tolerance screen) over n=53 (1-failure-tolerant), per §3.1's lead
   recommendation. This is a defensible, not a "safe," choice: per §3.1's own power table, at a
   genuinely good true FAR of 2–3%, n=40 only passes roughly a third to a half of the time on
   chance alone (34–49% at n=35; n=40 is marginally better) — a single unlucky replicate call
   among 120 (40×k=3) fails the gate outright, with no override, even against a fine judge. That is
   the accepted trade-off, not a defect to raise again; it is recorded here so a future reader
   doesn't mistake a G1 failure for proof the judge regressed, without first checking whether it
   was a single-case trip.
3. **Synthetic-but-realistic is still not real traffic** (F5). Nothing in this expansion closes
   that gap — it makes the synthetic set larger and better-stratified, not less synthetic. If/when
   `falkor-chat` accumulates real triage transcripts (still absent per F5's live check), the
   archived note's own recommendation stands: re-derive from real data and treat this file as the
   bootstrap.
4. **A larger fixture is a larger leakage surface to keep clean.** The existing leakage guard
   (archived §5.3: never seed these cases into a prompt, few-shot block, or live corpus) scales
   with row count; nothing about that rule changes, but 85 rows are more rows to audit than 26 if
   anyone ever proposes reusing fixture content as example text elsewhere in the codebase.
5. **My own drafted rows and rationales (§6, and any I draft for the clear tiers per §5(c)) carry
   the same construct-validity caveat the brief raised about labeling in general** — I am an LLM
   proposing labels for a system partly evaluated by an LLM judge. The spot-check in §5(c) is the
   mitigation, not a solution; it should not be skipped because the tier is "clear by construction"
   without at least a human reading each drafted row once.

## 9. Verified vs. inferred

**Verified by reading/executing/querying:**
- `golden_guards.jsonl`'s exact current composition (26 rows, strata, path split) — parsed
  directly, matches `guard-judge-calibration-ml.md` F4.
- `guard_calibration.py`'s metric functions are tier/path-generic; only
  `test_guard_calibration_live.py` lines 223/227/228/232/233 hardcode fixture size —
  read directly. (**2026-08-20 correction:** this line's own citation carried the same
  233/234 off-by-one the analyst review found in §7 step 2 and F3 — fixed here too, same root
  cause, third occurrence.)
- `test_golden_set_integrity.py` exists for `golden_retrieval.jsonl` only; no equivalent exists for
  `golden_guards.jsonl` — checked directly (`ls server/tests/eval/`).
- `ws:acme`'s actual content (2 threads, one a two-message smoke exchange, the other with no
  reachable messages; 21 `WorkflowRun`s) — queried read-only via `mcp__cypher__query`, no writes.
- No cloud API key is available in this lab (K-042's own QA report, `docs/test-reports/
  llm-provider-config2-report.md` per the AGENTS.md milestone table) — read from the milestone
  map, not independently re-verified this session.
- All Wilson-interval and binomial-power numbers in §3 — computed this session (Python, `math`
  stdlib only, no scipy available in `server/.venv`), cross-checked against the archived note's own
  `[0%, 27.8%]`/`[0%, 11.4%]` figures, which reproduce to within rounding under the same formula.

**Inferred / assumed (flag if wrong):**
- That Ministral is still the only other locally-loaded model in this lab's LM Studio at the time
  a labeler-independence decision is made — read from this run's own coordination-doc baseline
  table (`docs/plans/guard-reliability-followups-coordination.md`), not re-checked live by me.
- That a human spot-check of LLM-drafted `clear_advance`/`clear_suspend` labels is genuinely
  low-risk given "clear by construction" — inherited from the archived protocol's own §10 risk 3
  framing, not independently re-derived here.
- **2026-08-20:** both §10 closures (boundary-independence dropped; n=40 chosen) are taken as
  given, per the brief for this finalization pass — stated as user decisions, not independently
  re-derived or second-guessed by me this session. The 59 rows in §6 are my draft content produced
  under those settled decisions, not a re-opening of either.

## 10. Decisions — CLOSED 2026-08-20

Both decisions this note originally flagged as blocking are now resolved, by the user, this date:

1. **Boundary-tier independent-second-labeler requirement: DROPPED.** No second labeler is
   available. `boundary` is sourced identically to `clear_advance`/`clear_suspend` — §5(c) option
   (a) (LLM-drafted candidate rows, human-spot-checked before merge) — applied uniformly across
   all three strata. **This is an explicit, dated descope from the backlog item's stated intent**
   (K-027 item 4 asked for boundary-tier independence specifically; §5 above explains why that
   was the tier where independence actually mattered, being a policy-judgment tier rather than a
   fact-extraction one). The materiality of the descope: a boundary-tier row's label now carries
   **no more validation than a clear-tier row's does** — a single spot-check of a proposed label,
   not two independent opinions compared. Anyone reading the calibration report downstream should
   treat every `boundary` label as a spot-checked draft, not a validated ground truth, exactly as
   the clear tiers already were understood to be.
2. **`clear_suspend` stratum size: n=40.** This note's own §3.1 lead recommendation (the
   zero-tolerance screen), not the n=53 1-failure-tolerant alternative that was flagged as a live
   choice. §3.4's composition table, the 85-row total, and §7's literal-constant edits were already
   written against n=40 and required no numeric correction (re-verified this pass, §3.4).

**Resulting scope, unconditional as of this revision:** §7's implementation plan is ready for
`tdd-engineer` dispatch as written — no further decision gates it. §6 below carries the full
59-row draft (32 new understanding-path + 27 new turns-path, see the per-stratum breakdown at the
top of §6) that a human reviewer spot-checks before it is merged into
`server/tests/eval/golden_guards.jsonl` and the five literal constants in
`test_guard_calibration_live.py` are updated per §7 step 2.
