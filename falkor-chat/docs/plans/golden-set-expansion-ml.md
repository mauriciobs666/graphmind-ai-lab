# Golden-set expansion for the guard-judge calibration (K-027 item 4)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** K-027 (item 4)

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
:233  assert len(clear_cases) == 21
:234  assert len(boundary_cases) == 5
```

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
| **53** | ~8.4%* | **9.9%*** | — |
| 69 | — | — | **9.7%*** |

(*minimal n at which that failure count still clears ≤10%, found by search)

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
on a single case by chance rather than by defect. **That is a real trade-off, not a foregone
conclusion — if the team wants the gate to survive a single false-advance without going straight
to BLOCK, the number is 53, not 40, and I'm flagging that as a live choice rather than picking it
for you.**

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
is not a power calculation, it's a coverage and second-labeler-workload question.
**Recommendation: n=15** (3× current) — enough distinct genuinely-ambiguous edge shapes to be a
meaningful independent-labeling exercise (§5) without turning that exercise into a multi-day task.

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

## 5. The independence problem — this needs a human decision

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

**What I am not deciding:** whether the user (or another designated person) is willing to spend
the time being that second boundary labeler, whether to provision a cloud model for a
clear-tier cross-check, and whether 15 boundary cases is the right amount of that person's time to
ask for. Those are the open questions this note surfaces rather than resolves.

## 6. Illustrative draft rows — NOT committed to the fixture, unverified

Two rows, same JSONL shape as `golden_guards.jsonl`, illustrating §3.4's under-represented
`turns`-path shapes (multi-turn aggregation; chained unanswered clarification). **Draft only. Not
independently verified. Not written to `golden_guards.jsonl`.** Labels are my proposed reading,
exactly the kind of thing §5(c) says needs a human spot-check before being treated as ground truth.

```json
{"id": "tn-08", "tier": "clear_advance", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn08-1", "text": "Getting a weird error on the billing page.", "role": "user", "createdAt": 1752571000000, "authorId": "u-erin", "displayName": "Erin", "authorType": ["User"]}, {"msgId": "m-tn08-2", "text": "What page exactly, and what's the error?", "role": "assistant", "createdAt": 1752571001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn08-3", "text": "The invoices tab. It's a blank white screen, no error text visible.", "role": "user", "createdAt": 1752571002000, "authorId": "u-erin", "displayName": "Erin", "authorType": ["User"]}, {"msgId": "m-tn08-4", "text": "Which browser, and does it happen every time or intermittently?", "role": "assistant", "createdAt": 1752571003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn08-5", "text": "Chrome, and yeah every time since this morning around 9am.", "role": "user", "authorId": "u-erin", "createdAt": 1752571004000, "displayName": "Erin", "authorType": ["User"]}], "expected": true, "label_rationale": "DRAFT/UNVERIFIED: multi-turn aggregation — page, symptom (blank screen, no visible error), browser, onset time and reproducibility are all present but arrive across five short turns rather than one clean exchange. Tests whether the judge aggregates dispersed evidence rather than crediting only the most recent turn."}
{"id": "tn-09", "tier": "clear_suspend", "path": "turns", "r1_probe": false, "condition": "the user has provided enough information to research their request", "understanding": {}, "turns": [{"msgId": "m-tn09-1", "text": "The export job keeps timing out.", "role": "user", "createdAt": 1752572000000, "authorId": "u-frank", "displayName": "Frank", "authorType": ["User"]}, {"msgId": "m-tn09-2", "text": "Is that the CSV export or the PDF export, and roughly how big is the dataset?", "role": "assistant", "createdAt": 1752572001000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}, {"msgId": "m-tn09-3", "text": "It's pretty large, a few hundred thousand rows I think.", "role": "user", "createdAt": 1752572002000, "authorId": "u-frank", "displayName": "Frank", "authorType": ["User"]}, {"msgId": "m-tn09-4", "text": "Got it — and which export format, CSV or PDF?", "role": "assistant", "createdAt": 1752572003000, "authorId": "assistant", "displayName": "Assistant", "authorType": ["Agent"]}], "expected": false, "label_rationale": "DRAFT/UNVERIFIED: chained unanswered clarification — two clarifying questions were asked; the user answered only the first (dataset size) and the second (CSV vs PDF) remains open in the last turn. A judge that credits 'the user replied' without checking which question was actually answered would wrongly advance here — the deeper variant of tn-05's single-unanswered-question hazard."}
```

## 7. Downstream implementation plan (for a future `tdd-engineer` dispatch, contingent on §5)

**Not to be dispatched until the user resolves §5.** Once resolved:

1. **Fixture content.** `server/tests/eval/golden_guards.jsonl` grows from 26 → 85 rows per §3.4's
   table, with `clear_advance`/`clear_suspend` labels sourced per §5(c)'s option-(a) path (my
   drafts + human spot-check) and `boundary` labels sourced from the independent second human
   labeler. Every new row follows the existing schema exactly (`id`, `tier`, `path`, `r1_probe`,
   `condition`, `understanding`, `turns`, `expected`, `label_rationale`) — §6 shows the shape.
2. **`server/tests/eval/test_guard_calibration_live.py` — five literal constants to update** (F3):
   line 223 (`== 26` → `== 85`), line 227 (`== 26` → `== 85`), line 228 (`26 * K_REPLICATES` →
   `85 * K_REPLICATES`), line 233 (`== 21` → `== 70`, i.e. 30+40), line 234 (`== 5` → `== 15`). No
   other harness file needs a size-specific edit (F3) — `guard_calibration.py`'s metric functions
   already group dynamically by `tier`/`path`.
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
   > remains a policy call — one or two labelers' (see the fixture's own provenance header for
   > which), not ground truth — and both strata are synthetic-but-realistic, not sampled from live
   > traffic.

   The fixture itself (or a small provenance file alongside it) should record whether the
   boundary tier ended up single- or dual-labeled, and if dual, the plain percent-agreement number
   from §5(c) — this is exactly the "own header/provenance" caveat the brief's option (b) called
   for, generalized to whichever sourcing path the user actually picks.
5. **Live run cost, for the user's awareness, not something I ran:** 85 cases × k=3 = 255 live
   judge calls against the local Qwen3-4B — same order of magnitude as the existing 78-call run,
   not a meaningful compute concern. The **labeling** cost (59 new rows, at least 15 of them
   needing a second independent human pass) is the real cost this note is asking the user to
   authorize.

## 8. Risks & open questions

1. **(Highest — restates §5) Who labels the boundary tier, and is that person available?** This
   blocks everything downstream of §7 step 1. No default is assumed.
2. **n=40/30/15 vs. a cheaper or more ambitious target is a real choice, not a formality.** §3.1
   names n=53 explicitly as the 1-failure-tolerant alternative to n=40's zero-tolerance screen; the
   user may reasonably prefer either, or may judge 85 total rows disproportionate to a still-single
   guard condition (archived §9's own scope note: "the gate calibrates one string... any new `llm`
   guard inherits nothing from this run").
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
  `test_guard_calibration_live.py` lines 223/227/228/233/234 hardcode fixture size —
  read directly.
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

## 10. What decision is needed before this moves to implementation

**One decision blocks everything: who is the boundary tier's second independent human labeler,
and are they available to label ~15 fresh scenarios without seeing the first labeler's rationale?**
Everything else in §7 can proceed once that is answered (or once the user explicitly accepts a
narrower scope that drops the boundary-tier independence requirement, at the cost of not actually
satisfying the backlog item's stated intent). A secondary, lower-stakes decision: whether n=40
(zero-tolerance) or n=53 (1-failure-tolerant) is the right size for the `clear_suspend` stratum —
either is defensible; I've stated the trade-off in §3.1 rather than picking for you.
