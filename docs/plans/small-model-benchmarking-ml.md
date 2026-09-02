# Small-Model Benchmarking — Statistics and Metric Definitions

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — · **Version:** 1.1

2026-09-02 (`architect`, outcome only — no method changed) — §8's three flags were accepted into
the requirements (commit `afe4aef`): §3.2's paired instrument is now what FR-15/AC-4 *require*,
§4.2's nesting is now FR-8(d), §4.5's floor is now FR-22a, and §6's judged layer is **deferred** by
FR-21a, so §6.2/§6.3 describe a design to be built later rather than one in first delivery.

## 1. The question and the decision it serves

The architect is writing the implementation plan for `model-bench/` against
`docs/requirements/small-model-benchmarking.md` (Status: Ready for design). Five method
questions in that document are under-specified or in genuine tension, and an implementer would
otherwise have to invent the statistics. This note settles them so the plan can name exact
formulas, exact denominators, exact decision wording and exact sample sizes.

**Scope boundary.** This note does not redesign the requirements' measurement architecture (seven
tool-calling counts, per-turn reporting, system-ground-truth scoring, four retrieval indicators,
paired design). It resolves how those are *computed and reported*. Where a requirement is
methodologically unbuildable as written, §8 says so and names the amendment.

**Assumed design decisions (architect's, accepted here as sound):** in-process brute-force exact
cosine over a copied ~121-doc corpus with a BM25 keyword arm; simulated deterministic tools with
in-harness state as the tool-caller's ground truth; pack-versioned prompt assembly. All three are
methodologically correct for the stated object of measurement — the *model*, not a product
pipeline. Nothing below asks to change them.

**Dependency budget.** Every statistic in this note is implementable in **Python 3.12 stdlib
only** (`math.sqrt`, `math.comb`, `random`, `statistics`). **No scipy, no statsmodels.** `numpy`
is worth taking for the embedder pack alone (a 121×1024 similarity matrix per query set); the
statistics layer must not import it, so the stats module stays pure and unit-testable exactly
the way `falkor-chat/server/tests/eval/metrics.py` and `nlq_scoring.py` already are.

---

## 2. Findings from the real system

Read in full: the requirements document; `falkor-chat/server/tests/eval/{metrics.py,
nlq_scoring.py,judge.py}`; `retrieval_baseline.json`; `corpus_provenance.json`;
`golden_retrieval.jsonl` (38); `golden_guards.jsonl` (85); `golden_judge_calibration.jsonl` (10);
`nlq_golden_set.jsonl` (40); `judge_calibration.json`; and
`falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.

Measured/derived facts that drive the answers below (all computed in this session from the
committed artifacts, none quoted from memory):

| Fact | Value | Where from |
|---|---|---|
| Golden-retrieval relevance cardinality | 36 of 38 items have exactly **one** relevant id; 2 have two | counted from `golden_retrieval.jsonl` |
| Pinned baseline | recall@10 = 0.9737 (37/38), recall@5 = 0.8947 (34/38), MRR = 0.6259, n=38 | `retrieval_baseline.json` |
| Corpus | 121 messages, 12 topics, `text-embedding-qwen3-embedding-0.6b`, dim 1024 | `corpus_provenance.json` |
| Guard set composition | `clear_suspend` 40 · `clear_advance` 30 · `boundary` 15; `expected=False` 55, `expected=True` 30 — **every boundary item is an expected-suspend** | counted from `golden_guards.jsonl` |
| NLQ set composition | 40 items; 21 scalar / 13 set / 6 not_found across 7 shapes | counted from `nlq_golden_set.jsonl` |
| Judge calibration, faithfulness | raw agreement 9/10 = 0.90, Wilson95 [0.596, 0.982], **Cohen's κ = 0.833** | recomputed from `judge_calibration.json` |
| Judge calibration, relevance | raw agreement 7/10 = 0.70, Wilson95 [0.397, 0.892], **Cohen's κ = 0.211**; false-positive rate (judge says relevant when gold says not) **2/3** | recomputed from `judge_calibration.json` |
| Judge conflict of interest | `"sameModelAsAgentUnderTest": true` — judge and agent-under-test were both `qwen/qwen3-4b-2507` | `judge_calibration.json` |
| Effect sizes this lab has actually cared about | qwen3-4b turn-4 collapse 97.5% vs ministral 0/176; ministral duplicate-instruction 30% | review §8.2/§8.4 |

Three of these are decision-changing and are used repeatedly below: the **relevance-axis κ of
0.21**, the **saturation of recall@k on the 38-item set**, and the **class-conditional smallness of
the 85-item guard set**.

---

## 3. Q1 — FR-15 vs FR-16: what the tool computes and prints

### 3.1 The finding: FR-15's literal rule is not conservative, it is inert

FR-15 as written ("two models are declared different only when their intervals don't overlap")
reads as a safe, conservative rule. At the sample sizes FR's own out-of-scope section commits to
(~20–40 per arm), **it is not conservative — it is incapable of ever firing in the regime this
lab actually operates in.** Computed here, minimum second-arm score needed for two marginal
Wilson intervals to separate:

| n | baseline 0.50 | 0.70 | 0.90 | 0.95 |
|---|---|---|---|---|
| 20 | needs 0.950 (Δ 45 pp) | **impossible** | **impossible** | **impossible** |
| 30 | 0.867 (Δ 36.7 pp) | 1.000 (Δ 30 pp) | **impossible** | **impossible** |
| 40 | 0.800 (Δ 30 pp) | 0.950 (Δ 25 pp) | **impossible** | **impossible** |
| 85 | 0.706 (Δ 21.2 pp) | 0.871 (Δ 17.6 pp) | 1.000 (Δ 10.6 pp) | **impossible** |

"Impossible" means: *even a model scoring 100% cannot separate from the baseline under this rule.*
At n=40 with an incumbent at 0.90, a candidate at a flawless 40/40 still "overlaps."

The worked case to put in the plan:

> **40/40 vs 34/40, perfectly nested (the candidate gets every item the incumbent gets, plus 6).**
> Marginal Wilson: [0.912, 1.000] and [0.709, 0.929] — **overlap**, so FR-15's literal rule prints
> "not distinguishable."
> Paired: b=6, c=0. McNemar exact two-sided **p = 0.031**. Paired difference **+15.0 pp,
> 95% CI [3.2, 29.1] pp** — excludes zero.
>
> The candidate strictly dominates on every item and the literal rule cannot say so. That is not
> caution; it is a rule that discards the entire experimental design FR-16 mandates.

The reason is mechanical: two marginal intervals overlapping is a *much* stronger condition than
their difference covering zero, and the marginal intervals throw away the item-level covariance
that pairing exists to capture. FR-16 pays the cost of pairing; FR-15 as literally worded then
refuses to bank the return.

### 3.2 Recommendation

**Read "the interval" in FR-15 and AC-4 as the 95% confidence interval on the *paired
difference*, not as two marginal intervals.** This preserves AC-4's intent exactly ("don't rank
on noise") and states it correctly. §8 carries this back as a requirement amendment for `tico`.

Compute and print, in this order:

**(a) Per-arm reporting — Wilson score interval, unchanged.**
Reuse `nlq_scoring.wilson_interval` verbatim (`_Z_95 = 1.959963984540054`; this lab's convention,
not Clopper-Pearson, not rule-of-three). Every rate prints as `k/n = p̂ [lo, hi]` — never a bare
percentage, never without its denominator. These are **descriptive**: they say what each model
scored. They are explicitly **not** the comparison instrument, and the report must say so in one
line under the table.

**(b) The decision rule — McNemar's exact test on the paired 2×2, for binary metrics.**

Build the paired table over items scored by *both* models in the *same session* (FR-16):

```
                 B correct   B wrong
  A correct         a           b
  A wrong           c           d      n = a+b+c+d
```

`b` = items A wins, `c` = items B wins. Only the **discordant** pairs carry information.

```python
from math import comb
def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact conditional (binomial-sign) test. b, c = discordant counts."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(m, i) for i in range(k + 1)) * (0.5 ** m)
    return min(1.0, 2.0 * tail)
```

Exact, not the chi-square approximation — the chi-square version is invalid at the discordant
counts this lab will actually see (b+c often < 10). Stdlib only.

Its small-sample floor, computed here, is the single most useful number for the whole tool:

| discordants against (`c`) | minimum `b` to reach p ≤ 0.05 |
|---|---|
| 0 | **6** |
| 1 | 8 |
| 2 | 10 |
| 3 | 12 |
| 4 | 13 |

So **no paired binary comparison can ever be declared significant on fewer than 6 net wins**,
whatever n is. That is the honest hard floor and it should be printed.

**(c) The effect size — MOVER-D (Newcombe) confidence interval on the paired difference.**

Reuses the Wilson function the lab already has, so it is consistent with (a) by construction:

```python
def paired_diff_ci(a, b, c, d, z=1.959963984540054):
    """Newcombe's square-and-add (MOVER-D) 95% CI for p1 - p2 on paired binary data."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n          # A's rate, B's rate
    diff = p1 - p2                              # == (b - c) / n
    l1, u1 = wilson_interval(a + b, n, z=z)
    l2, u2 = wilson_interval(a + c, n, z=z)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = ((a * d - b * c) / den) if den > 0 else 0.0     # margin-zero -> phi = 0
    lo = diff - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = diff + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return max(-1.0, lo), min(1.0, hi)
```

The `max(0.0, …)` clamps are required, not cosmetic: the radicand goes slightly negative under
extreme `phi`. Worked outputs verified in this session:

| a | b | c | d | n | diff | MOVER-D 95% CI | McNemar p |
|---|---|---|---|---|---|---|---|
| 34 | 6 | 0 | 0 | 40 | +15.0 pp | [3.2, 29.1] | 0.031 |
| 30 | 6 | 0 | 4 | 40 | +15.0 pp | [3.9, 27.7] | 0.031 |
| 33 | 6 | 1 | 0 | 40 | +12.5 pp | [−1.0, 26.9] | 0.125 |
| 20 | 8 | 2 | 10 | 40 | +15.0 pp | [0.2, 28.8] | 0.109 |
| 72 | 10 | 2 | 1 | 85 | +9.4 pp | [1.5, 18.2] | 0.039 |

Use these five rows as the implementer's regression fixtures for the statistics module.

**(d) Continuous metrics — paired percentile bootstrap on the per-item difference.**

McNemar does not apply to MRR, score separation, or latency. Use a seeded paired bootstrap:
resample the *items* (with replacement, n draws), recompute the mean per-item difference each
time, B = 10 000, take the 2.5th/97.5th percentiles. ~15 lines of stdlib (`random.Random(seed)`).
The seed goes into the environment fingerprint (FR-7) so a report is reproducible. Decision:
**the CI excludes zero.** No separate significance test — for continuous metrics the CI *is* the
test, and reporting both would be redundant, not extra rigour.

For **clustered** data (tool-caller: turns nested in conversations nested in scripts), resample
the **outermost unit** — the script — not the turn. See §4.4.

**(e) The decision wording, which AC-4 must be checked against.**

Three verdicts, exactly these strings:

1. **Distinguishable.**
   `A is better than B on <metric>: +15.0 pp (95% CI [3.2, 29.1] pp), n=40 paired items, McNemar exact p=0.031 (b=6, c=0).`
2. **Not distinguishable.**
   `Not distinguishable at this sample size. Observed difference +12.5 pp, 95% CI [-1.0, 26.9] pp covers zero (b=6, c=1, McNemar exact p=0.125). This pack resolves differences of >=19.1 pp at n=40 with 80% power; the observed 12.5 pp is below that. Neither model is ranked above the other.`
3. **Instruments disagree** (MOVER-D excludes zero, McNemar does not — row 4 of the table above; real and not rare):
   `Not distinguishable at this sample size. The effect-size interval [0.2, 28.8] pp excludes zero but the exact paired test does not reach alpha=0.05 (b=8, c=2, p=0.109). Reported as not distinguishable: the exact test is the decision rule.`

**McNemar exact is the decision; MOVER-D is the effect size.** One instrument decides, one
quantifies. Do not AND them into a bloc — but *always print both individual outcomes in the
prose*, as verdict 3 does, so a reader never sees an aggregate verdict without the two components
that produced it.

**FR-15's literal marginal-overlap check should still be computed and printed as a diagnostic
line** ("marginal Wilson intervals overlap: yes/no"), because the requirement asks for it and it
costs nothing — but it must be labelled *diagnostic*, never the verdict, with a one-line footnote
naming why (§3.1's inertness). That keeps the requirement visibly honoured while the decision is
made correctly.

### 3.3 Multiple comparisons

Seven tool-calling counts × ~9 turn positions × several metrics is dozens of tests; at α=0.05 each,
a false "better" is close to certain. Cheapest correct handling, and the recommendation:

**Each task pack declares exactly one `primaryMetric` in its versioned config.** Only that metric
receives a better / not-distinguishable *verdict*. Every other number is printed with its CI and
labelled `exploratory — no significance claim`. Pre-registration in pack config also means the
primary metric cannot be chosen after seeing the results.

Recommended primary metric per role — see §7 for the n and MDD that go with each:

| Role | `primaryMetric` | Unit |
|---|---|---|
| tool-caller | clean-through-turn-`H` rate (`H` pack-declared, default 4) | conversation |
| guard-judge | false-advance rate on `clear_suspend` | item |
| nlq-generator | Layer-1 exact-match rate | item |
| chat-responder | deterministic checklist pass rate | item |
| embedder | MRR | query |

*Reversal trigger:* if the stakeholder wants verdicts on several metrics at once, apply
Holm–Bonferroni across the pre-declared family and print the adjusted threshold next to each
p-value. Do not silently test many and report the winner.

---

## 4. Q2 — FR-8/FR-9 denominators, precondition handling, per-turn reporting

### 4.1 Notation

Per turn `t` of a scripted conversation, the pack supplies ground truth and the harness records:

- `R(t)` — the required tool calls (tool name + expected arguments). Possibly **empty** (an
  abstention turn, a chit-chat turn) — those turns are first-class, not filler.
- `E(t)` — the calls actually dispatched, from the harness's own dispatch trace (FR-10).
- `P(t)` — whether the reply contained a prose-shaped pseudo-call (harness heuristic).
- `I(t)` — LLM iterations consumed in the turn, and whether the iteration cap was hit.
- `S(t)` — simulated tool state after the turn.

**Hard design rule that makes every denominator below well-defined: the harness always drives the
full script.** A turn is never skipped because a previous turn failed. This is what the prior §8.2
run did (turns 5–9 recorded after the turn-4 collapse) and it is what keeps the per-turn-position
denominator equal to `n` conversations at every `t`, rather than a selection-conditioned subset.
If a turn cannot be driven at all (LM Studio 400 / crash, as `gpt-oss-20b` produced), it is
recorded as **`unrunnable`** and reported in its own count — never as a failure, never silently
dropped. A model whose comparison rests on 8 runnable turns out of 64 is not comparable to one
with 64, and the report must make that visible rather than averaging it away.

### 4.2 The seven counts — exact denominators

FR-8's (a) and (b) should **not** be two independently-reported rates. They share a denominator and
"did the model intend a call?" is unanswerable, so reporting them separately forces the
implementer to invent an intent heuristic. Collapse them into one 3-way partition:

**(a)+(b) — Emission form.** Denominator: turns with `|R(t)| ≥ 1`. Three mutually exclusive,
exhaustive outcomes, printed as a partition that sums to the denominator:

| outcome | condition |
|---|---|
| `native` | `|E(t)| ≥ 1` |
| `prose_pseudo_call` | `|E(t)| = 0` and `P(t)` fired |
| `no_attempt` | `|E(t)| = 0` and not `P(t)` |

FR-8(a) "called a tool when required" = the `native` rate (a prose pseudo-call is *not* a
dispatched call). FR-8(b) "native form rather than prose" = the `prose_pseudo_call` vs `no_attempt`
split of the failures. Both requirements satisfied, no intent inference needed.

`P(t)` is a heuristic, so the pack must ship a small labelled set of replies (~20, human-verified)
and the report prints **the prose detector's own precision/recall** next to the partition. Prior
art: review §8.3 did exactly this for `_note_possible_fabrication` and it is what turned a shipped
signal into a trustworthy one. An uncalibrated detector's number is not a measurement.

**(c) — Right tool chosen.** Denominator: turns with `|R(t)| ≥ 1` **and** `|E(t)| ≥ 1`
(arguments/tool identity are undefined when nothing was called). Numerator: `E(t)` contains at
least one call to every tool name in `R(t)` — *coverage of the required names*. Extra calls are
(e)'s business, not (c)'s; keeping them out of (c) is what stops one failure being counted twice.

**(d) — Argument correctness. Unit shifts to the *call*, not the turn.** Denominator: dispatched
calls whose **tool name is correct** (arguments of a wrong-tool call are meaningless). Print
`n_calls` explicitly — it is a different denominator from every other count and a reader will
otherwise assume turns.

The requirement's "split three ways" is not three disjoint buckets on equal footing. The correct
structure, and the one to implement:

- **Headline:** `all_args_correct` — binary per call.
- **Failure decomposition, per *argument* not per call:**
  - `omitted_required` — a required parameter is absent.
  - `wrong_value` — present but ≠ expected after canonicalization (reuse
    `nlq_scoring._scalar_equal`'s discipline: numeric epsilon for numbers, casefold+whitespace-
    collapse for strings, **never coerce across types**).
  - `boundary_unit` — **a named subset of `wrong_value`**, not a sibling: the wrong value is
    explained by a declared boundary/unit rule (inclusive-vs-exclusive bound, `$`/cents,
    kg/g). Report it as `wrong_value: 12, of which boundary/unit: 7`.

Make it data-driven, not heuristic: each expected argument in the pack carries an optional
`boundaryRule` listing the alternate encodings that count as boundary confusions. Otherwise the
classifier is a regex nobody calibrated. (This is a live falkor-chat failure class — K-057's
inclusive-bound wording fix — so the pack should carry those cases.)

**(e) — Spurious and duplicate calls. Two separate counts, not one.** Denominator for both: turns
with `|E(t)| ≥ 1`.

- `spurious_turn_rate` — turns where `E(t)` contains a call to a tool not in `R(t)`.
- `duplicate_turn_rate` — turns where `E(t)` repeats an already-satisfied call, **either
  within-turn or re-issuing a prior turn's completed call**. Both sub-shapes are real observed
  defects (K-061 same-turn `add_to_cart`; §8.4's ministral turn-2 re-issue of turn 1). Report the
  within-turn and cross-turn variants as a named breakdown — the ministral defect was
  turn-2-specific and a pooled duplicate rate would have hidden it.

**(f) — Stopping when done.** Denominator: turns with `|E(t)| ≥ 1`. Numerator: no call dispatched
after `R(t)` was fully satisfied, **and** the loop terminated below the iteration cap. Report
alongside, over **all** turns as a diagnostic: `iteration_cap_hit_rate`, plus mean and p95 of
`I(t)`. The `gpt-oss-20b` message-spam defect (§8.4) presented purely as cap-hits; a binary
"stopped" rate alone would have scored it as a failure without saying what kind.

**(g) — Final reply matches what the tool returned.** Denominator: turns with ≥1 dispatched call
whose return value is **fact-bearing** — i.e. the pack declares at least one checkable value for
that turn. Turns where nothing checkable came back go to an explicit `unscoreable` bucket that is
printed, not silently excluded.

Scoring is deterministic, **not a judge**: normalized-substring containment of every
`mustContain` value **and** absence of every `mustNotContain` value, both listed per turn in the
pack. This is the lab's established convention (`nlq_scoring.layer2_contains`, review §8.1's
value-containment ground truth). The negative list is the half that matters: it is what catches
"successfully removed from your cart" narrated over an unchanged cart — the exact failure whose
price-shaped-token blind spot §8.3 documented.

**Plus one count the requirement omits and needs — restraint.** Denominator: turns with
`R(t) = ∅`. Numerator: `|E(t)| = 0`. Without it, a model that calls tools indiscriminately scores
perfectly on (a) and the abstention turns contribute nothing. Cheap to add, and conditions A and C
already contain abstention turns.

### 4.3 Precondition failures must never be laundered

Three rules, all mandatory:

1. **Every printed rate carries its denominator inline** — `k/n`, never a bare percentage.
2. **A turn excluded from a conditional denominator is counted in that count's own `n/a` tally**,
   printed next to the rate.
3. **The report opens with a funnel table**, not a metric table:

```
turns driven                  360
  unrunnable (harness/server)   0
  R(t) = 0 (restraint turns)   40   -> restraint rate 38/40
  R(t) >= 1                   320
    native call emitted       142   -> (a)+(b) partition over 320
    prose pseudo-call          31
    no attempt                147
  turns with >=1 call         142   -> (c), (e), (f) denominators
  dispatched calls            167   -> (d) denominator  [NOTE: calls, not turns]
  fact-bearing returns        118   -> (g) denominator
  unscoreable returns          24
```

This is the mechanism that stops "100% argument correctness" on three calls from being read as
comparable to 95% on two hundred. Without it, a model that collapses early looks *better* on every
conditional count, because it never generated the calls that could be wrong. That is the single
most likely way this harness lies, and the funnel is the fix.

**Paired-comparison corollary:** for the conditional counts (c)–(g), pairing only works on items
where *both* models produced a scoreable outcome. The paired `n` is the **intersection** and must
be printed separately from each arm's own `n`. Items scoreable for exactly one model are reported
as an `asymmetry` count — they are not missing data, they are a finding about the model that could
not produce them.

### 4.4 Per-turn-position reporting when turns are not independent

The non-independence is real but it is **not** a problem for the per-turn-position slice, and
saying why matters:

**At a fixed turn position `t`, each conversation contributes at most one observation.** Across
conversations, those observations *are* independent. So a per-turn-position rate is a clean
binomial over conversations, and a **Wilson interval over conversations (never over turns) is
meaningful and correct**. FR-9's per-position slicing is precisely the slicing that restores
independence. This is exactly what review §8.2's table did.

**Pooling across turns within a conversation is what breaks.** Any statistic that sums turns
(overall accuracy, "280 turns") has an effective sample size far below the turn count, because
turn 4's outcome is nearly determined by turn 3's. In §8.2's data the within-conversation
correlation was essentially 1.0 after onset — 121 post-onset turns produced zero independent
information. **A pooled per-turn CI at n=280 in that dataset was a fiction; the real n was 40.**

Two mandatory consequences:

1. **Never print a Wilson interval over a turn-pooled count.** Where a pooled rate is wanted,
   compute its CI by **cluster bootstrap resampling conversations** (or scripts — see §4.5),
   recomputing the pooled rate inside each resample. ~20 lines of stdlib. Print the implied
   **design effect** (`bootstrap CI width ÷ naive Wilson width`) next to it; in §8.2's data that
   would have printed ≈2.6, which is the number that tells a reader the pooled figure is soft.
2. **The per-position table is the primitive and is always printed**, with per-position `n` =
   conversations. At n=15 per condition a per-position Wilson is wide but honest (0/15 →
   [0.000, 0.204]; 15/15 → [0.796, 1.000]) and it was more than enough to establish §8.2's finding.
   At n=10 (condition C) it is wider still and should be printed with a "descriptive at this n"
   marker rather than used for a verdict.

### 4.5 The clustering hazard the requirements do not name (raise this with the stakeholder)

FR-22's conversation scripts are **fixed**. §8.2's design ran 15 replicates of the *same*
condition-A script. That is **trial replication**, not item replication: the resulting CI describes
"if I run this one script again," not "if I wrote a different 9-turn script." A tool that prints
±5 pp from 40 replicates of 3 scripts is reporting an interval that could move 40 pp if someone
wrote a fourth script. **This is the largest methodological risk in the feature**, larger than any
choice of test.

Recommendation for the FR-22 asset:

- **3 conversation shapes × 4 distinct scripts per shape × 4 replicates = 48 conversations per
  model.** Keep the A (9-turn read-only) / B (7-turn write-mutating) / C (4-turn short) shapes —
  they are empirically load-bearing and reconstructing them is already the plan.
- Cost check: ~7 turns × 48 × ~1.3 s/turn ≈ 7 min per model, ×2 for the paired arm ≈ 15 min.
  Affordable; the prior run's timing is the basis for that estimate, not a guess.
- **The CI is a two-level cluster bootstrap: resample the 12 scripts with replacement, then
  resample replicates within each drawn script.** Effective n lands between 12 and 48 and the
  bootstrap finds it rather than the analyst assuming it.
- Also pin **temperature** (FR-18 already requires this) and state the replication semantics in the
  report: at temperature 0 replicates of an identical prompt are near-duplicates and add almost no
  information — a run at temperature 0 with 4 replicates per script has an effective n closer to 12
  than 48. Print `temperature` and `replicatesPerScript` adjacent to every conversation-level n so
  the reader can see which regime they are in.

### 4.6 Aggregating across conditions of different length without a confounded headline

Do **not** pool turns across conditions — a 9-turn script contributes 2.25× the turns of a 4-turn
one and the headline becomes a weighted average of script lengths.

**Recommended headline: a survival statistic at a pack-declared turn depth.**

- Primary: **`cleanThroughTurnH`** — the fraction of conversations with zero failure of any kind
  through turn `H`, where `H = min(script length)` across all conditions in the pack (4 for the
  A/B/C set). Every conversation of every condition contributes exactly one observation, so it is
  length-independent by construction, it is a proper binomial over conversations, and it is the
  statistic that would have caught `qwen3-4b` on the first run.
- Diagnostic, always printed: the **per-turn hazard** — `P(first failure at t | clean through
  t-1)`, denominator = conversations still clean entering `t`. Hazard is what distinguishes
  "gradual degradation" (flat, low hazard) from "deterministic collapse at a fixed position"
  (hazard ≈ 0, 0, 0, 1.0 — §8.2's actual shape). A pooled accuracy number cannot tell those apart
  and the difference is the whole reason FR-9 exists.
- Per-condition tables are always printed underneath. A cross-condition headline other than
  `cleanThroughTurnH` is not reported at all.

---

## 5. Q3 — Embedder metrics, exact definitions

### 5.1 recall@k, MRR, precision@k

Preprocessing that is not optional: **L2-normalize every embedding before cosine**, for both
queries and documents, and record the distribution of raw `‖v‖` as a diagnostic (some LM Studio
embedding endpoints return unnormalized vectors; a silent scale difference would corrupt score
separation without touching ranking, so it must be visible).

- **recall@k** = `|top-k ∩ R| / |R|` — `metrics.recall_at_k` verbatim, semantics unchanged
  (standard, handles multi-relevant, correctly raises on empty `R`).
- **MRR** = reciprocal rank of the *first* relevant id — `metrics.mrr` verbatim.
- **precision@k** = `|top-k ∩ R| / k`.

**Finding on precision@k: on this golden set it carries no information.** Because
`precision@k = recall@k · |R| / k`, and `|R| = 1` for 36 of 38 items, precision@k is (to within
the two two-relevant items) a fixed rescaling of recall@k — an exact algebraic identity, not an
opinion. Its ceiling at k=10 is 0.10. Report it once for FR-12 compliance with the footnote
`precision@k = recall@k · |R|/k; |R|=1 for 36/38 items, so this is a rescaling of recall@k and
adds no discriminating information on this pack`, and treat **P@1** (= "is the top hit relevant")
as the informative member of the family. *Reversal trigger:* extend the golden set with genuinely
multi-relevant items (|R| ≥ 3) and precision@k becomes informative again.

### 5.2 Score separation

**Per query, over the full corpus (brute force gives every score anyway, so no top-k truncation):**

```
sep_raw(q)  = max_{d in R(q)} cos(q, d)  -  max_{d not in R(q)} cos(q, d)
```

Can be negative — that is the informative case (an irrelevant document outranks every relevant
one). Note the exact identity, worth printing once so nobody double-counts:
**`sep_raw(q) > 0` ⟺ the global top-1 document is relevant ⟺ P@1 = 1 for that query.** So the
*sign* is redundant with P@1; the **magnitude is the new information** — the margin by which the
ranking is right or wrong, which is what predicts whether a similarity threshold will hold up.

**Normalization is required, and this is not a judgement call.** Cosine scales genuinely differ
across embedding families — some compress almost everything into [0.6, 1.0] — so a raw gap of 0.05
from one model and 0.15 from another are not comparable quantities. Report both:

```
sep_z(q) = sep_raw(q) / sd({cos(q, d) : d in corpus})
```

Per-query z-scoring against that query's own similarity distribution over the 121-doc corpus.
Scale- and offset-free, so it is the **cross-model comparable** number; `sep_raw` stays as the
within-model, product-actionable number (it is what you would set a threshold against).
*Rejected alternative:* per-query min-max normalization — driven by the single worst document in
the corpus, unstable at N=121.

**Aggregation:** per model report **median `sep_z`**, **10th percentile `sep_z`**, and **fraction
of queries with `sep_raw` > 0**. Median over mean because the distribution is skewed and a single
catastrophic query would dominate a mean; the p10 is the tail statistic that actually predicts
retrieval failures. Comparison across models uses the **paired bootstrap on per-query
`sep_z` differences** (§3.2d).

Worth noting for the pack's own documentation: with 12 topics over 121 messages, the "irrelevant"
pool contains same-topic near-misses — genuine hard negatives. That is a feature; the separation
number is meaningfully hard rather than trivially large.

### 5.3 The BM25 arm

**Construction, all of it versioned as pack data so the arm is reproducible without a network
download:**

- Tokenization: Unicode-aware `re.findall(r"\w+", text.casefold())`.
- **No stemming.** A Porter/Snowball stemmer means a dependency and a silent behaviour change
  between versions; declaring "no stemming" as pack config makes the arm reproducible. *Reversal
  trigger:* if BM25 beats the embedder on recall and the suspicion is morphology, add a stemmer
  **as a second, separately-named arm**, never by mutating the existing one.
- Stopwords: a **small English list committed in the pack** (not an `nltk` download).
- Parameters: `k1 = 1.2`, `b = 0.75` (standard Okapi defaults), both in pack config.
- IDF: use the always-positive variant `idf(t) = ln(1 + (N - df + 0.5)/(df + 0.5))`. At N=121 a
  term appearing in >60 documents is entirely plausible, and the classic `ln((N-df+0.5)/(df+0.5))`
  goes negative there, which produces documents *penalized* for containing a query term. Use the
  `+1` form.

**Arm or reference line? Both, precisely stated.** BM25 is **deterministic** given (corpus version,
query set, parameters) — zero run-to-run variance — but it still has **sampling variance over the
query population**, so its recall/MRR on 38 queries legitimately carries a CI. Recommendation:
**report it as a full paired arm with its CI**, labelled `reference arm (deterministic given pack
version — re-running will not change it)`. It participates in the paired comparison normally
(its half of the pairing contributes no noise, which only tightens the interval). This satisfies
AC-5 and gives the "quality read against search-without-embeddings" the requirement asks for.

### 5.4 Is `retrieval_baseline.json` usable as a validation target?

**Not as a metric target. Yes as a bug detector and as an implementation cross-check.** The
reasoning, which the plan should carry so nobody re-litigates it:

The pinned numbers came from falkor-chat's `hybrid_search`: **approximate** in-graph ANN **plus**
full-text, over the 121-message corpus. The new harness is **exact brute-force, vector-only**. Two
differences pointing in **opposite** directions — exact ≥ ANN on recall (no approximation loss),
vector-only ≤ hybrid on recall (loses the keyword contribution). A disagreement of either sign is
therefore uninterpretable as a quality signal. Comparing them as if they were the same measurement
would be the classic pipeline-vs-model confound the whole tool exists to avoid.

Two narrower uses that are genuinely valuable:

1. **Sanity floor / bug detector.** Running the *same* model (`text-embedding-qwen3-embedding-0.6b`,
   dim 1024) on the *same* 121-doc corpus and the *same* 38 queries, exact brute-force recall@10
   should land at or above the ANN-based 0.974, and certainly not below ~0.85. A materially lower
   number is a harness defect — wrong prefix, unnormalized vectors, a truncated corpus — not a
   model finding. Print it as `harness self-check`, explicitly **not** a quality gate (the
   requirements rule out hard gates and are right to).
2. **Implementation cross-check on the pure metric functions.** Feed `test_metrics.py`'s existing
   fixtures through the new implementation and require byte-identical outputs. Cheap, and it
   removes "did we reimplement recall@k subtly differently" as an explanation for any future
   divergence.

**Also copy `golden_retrieval.embeddings.json` (1.1 MB) as a fixture.** It contains the query
embeddings for that exact model, so the harness can run its ranking/cosine/metric path against
known vectors and isolate "is my ranking code right" from "is my embedding call right." That is a
one-time self-test with real diagnostic value at essentially zero cost.

**Provenance at the copy boundary (FR-6/FR-19).** Every copied file must gain
`copiedFrom` (repo-root path), `copiedAt`, and the **source git SHA**. Without those the copied
data has a provenance chain that dead-ends at the copy, and FR-6's comparability check silently
degrades to "same filename."

### 5.5 FR-14 prefixes — one correctness trap

Per-model `queryPrefix` / `docPrefix` in pack config, as FR-14 requires. The trap:
**the corpus must be re-embedded per model with that model's document prefix.** A cached corpus
embedding cannot be shared across models, and a cached one shared across *prefix settings of the
same model* is equally wrong. The cache key must include `(model id, quantization, docPrefix,
corpus version)` — all four. Getting this wrong produces a plausible-looking, entirely invalid
comparison, and it is exactly the kind of error that leaves no visible trace.

---

## 6. Q4 — The chat-responder role

### 6.1 The honest position

The role is **defensibly measurable at first delivery, but not as an open-ended "chat quality"
score.** As a judge-only score with the current 10-item calibration and a same-model judge it is
not defensible, and the committed artifacts prove it rather than merely suggesting it.

**Evidence, recomputed here from `judge_calibration.json`:**

| axis | raw agreement | Wilson95 | Cohen's κ |
|---|---|---|---|
| faithfulness | 9/10 = 0.90 | [0.596, 0.982] | **0.833** |
| relevance | 7/10 = 0.70 | [0.397, 0.892] | **0.211** |

Three things follow directly:

1. **The relevance axis is close to worthless.** κ = 0.21 is "slight" agreement. The committed file
   reports `"relevanceAgreement": 0.70`, which reads far better than it is — raw agreement is
   inflated by the skewed marginals (gold 7 relevant / 3 not; judge 8 / 2). Its Wilson CI includes
   0.5. Its class-conditional failure is the informative number: the judge called **2 of 3**
   genuinely-irrelevant answers relevant.
2. **The faithfulness axis is usable.** κ = 0.833 on a comparison-against-supplied-context task —
   near-extractive, which is what a small model can actually do.
3. **The 10-item calibration cannot support a gate on either axis.** Every interval above is
   ~40 points wide.

### 6.2 Recommended minimum defensible design

**Make chat-responder ~80% ground-truth-matched and only ~20% judge-mediated.**

**Golden set: 30 items.** Each item = (conversation prefix, user turn, retrieved context or
explicitly none), with a **checklist ground truth** rather than a reference answer:

```json
{
  "id": "cr-01",
  "context": ["…"],                    // may be [] for the abstention items
  "mustContain":    ["2000ms", "p99"], // normalized-containment, deterministic
  "mustNotContain": ["5000ms"],        // fabrications the context does not support
  "mustAbstain": false,                // true for the "not in context" items
  "provenance": {"corpusVersion": "…", "draftedBy": "…", "verifiedBy": "…", "date": "…"}
}
```

Deterministic scoring, no judge: all `mustContain` present **and** no `mustNotContain` present
**and** abstention matches `mustAbstain`. Reuse `nlq_scoring.layer2_contains`'s canonicalization
and `_ABSTENTION_MARKERS`. This converts the majority of "chat quality" into ground-truth
matching — the same move `nlq_scoring` already made for NL queries, and it is why that role has a
defensible number today.

*Why 30:* the McNemar floor means 6 net wins are needed regardless of n, so at n=30 the tool
resolves 20 pp observed / 25 pp at 80% power. At n=20 that becomes 30/37 pp — too coarse to be
worth building. 30 is the smallest n where the role can distinguish anything the lab has cared
about, against a human-verification cost of 30 items.

*Drafted how:* LLM-drafted from the **copied 121-message corpus** (real, already
provenance-tracked, and the retrieval contexts are then genuine), **every item human-verified**
per FR-19. Questions must be paraphrases, never verbatim.

**Judge component: faithfulness only.** Reuse `judge.py`'s faithfulness axis — including its
conservative parse handling and its unconditional `faithfulness=None` when context is empty, both
of which are correct and hard-won. **Drop the relevance axis entirely** and let the deterministic
`mustContain`/`mustAbstain` checks measure what relevance was trying to measure. κ = 0.21 is the
justification; this is not a preference.

**Which model judges — enforce it in code.**

- `judgeModel` is pinned in pack config and recorded in the fingerprint.
- **The harness hard-errors when `judgeModel == candidateModel`.** Cheap, enforceable, and the
  committed `"sameModelAsAgentUnderTest": true` shows the collision happens by default rather than
  by accident.
- If no non-candidate judge is loadable in a given session, the run still proceeds but sets
  `judgeIsCandidate: true` and **every judge-mediated number is suppressed from the comparison** and
  printed as diagnostic-only.
- Practically on this box: judge from a different family than the candidate (qwen candidate →
  ministral judge, or vice versa). `gpt-oss-20b` is not a judge option until its LM Studio
  crash is resolved (review §8.4).

**Two distinct self-preference caveats, never one blanket one.** The judge scoring **fixed,
human-authored calibration items** carries little self-preference risk — that pass is a legitimate
rubric-following signal. The judge scoring the **candidate's own live replies** carries it fully.
One undifferentiated caveat would let a reader extend trust from the first to the second. The
report must carry them separately, worded to the sub-pass.

**Calibration: extend to 40 items, and gate on class-conditional rates, not κ.**

- 10 → **40 items**, deliberately **balanced**: ~20 gold-faithful, ~20 gold-unfaithful, plus the
  empty-context abstention cases. Balance is required so class-conditional rates are estimable at
  all; the current set has only 2 gold-`False` faithfulness items.
- `judge.py` is **conservative by design** (its own docstring: prefer `None` over a guessed `True`).
  A deliberately-biased judge is mis-gated by any symmetric statistic: specificity sits near its
  ceiling by construction, which decouples κ from the error class that matters, and κ additionally
  moves with the hand-picked case mix. **Gate on the class-conditional rates:**
  - **`false-pass rate` = P(judge says faithful | gold unfaithful)** — the error that actually
    corrupts a score.
  - **`unfaithful-recall` = 1 − false-pass rate.**
  - **`parse-failure rate`** (`judge.py` already tracks it and must never be dropped from the
    denominator).
- κ and raw agreement are **reported as diagnostics with their marginals**, never as the gate.
- **Threshold (computed here, at 20 gold-unfaithful items):** judge usable if **false-passes ≤ 2/20**
  (rate 0.10, Wilson95 [0.028, 0.301]) **and** parse-failure rate ≤ 0.05. At 3/20 the Wilson upper
  bound reaches 0.36 — a judge that could be wrong on a third of the cases that matter is not a
  measurement instrument. *Reversal trigger:* if no available local judge clears 2/20, the
  faithfulness component is dropped and chat-responder ships deterministic-checks-only.

**Report language.** Judge-mediated numbers live in their own table, **below** the deterministic
ones, never summed into a headline, each carrying:

> `judge-mediated — not ground-truth-matched. Judge <model>, calibrated <date> on <n> items:
> false-pass rate k/n [lo, hi], parse-failure rate k/n. Not comparable in strength to the
> deterministic counts above.`

### 6.3 The flag to carry to the stakeholder

**Not "this role is unmeasurable" — a costed conditional:**

> Chat-responder at first delivery costs **one new 30-item golden set** plus **30 new
> judge-calibration items** (10 → 40, balanced), all human-verified per FR-19 — roughly the same
> drafting effort as FR-22's conversation scripts, which the requirements already accept as the
> main new golden-data cost. If that is funded, the role ships with a defensible, mostly
> ground-truth-matched number. **If it is not funded, chat-responder should ship
> deterministic-checks-only (no judge, no faithfulness axis) or be deferred — it must not ship as a
> judge-only score**, because the only calibration evidence the lab has says the judge's
> open-ended axis agrees with human labels at κ = 0.21.

---

## 7. Q5 — Sample sizes and the honest minimum detectable difference

### 7.1 The two numbers every report must print

The requirements' "roughly 15 percentage points" is the **observable floor at n=40** — the smallest
difference that *could* be called significant if it landed perfectly. It is not the difference the
tool reliably *detects*. Both numbers follow from the McNemar exact floor and were computed here:

- **Observable floor = `6/n`** — below this, no paired binary result can reach α=0.05, whatever
  happens.
- **80%-power MDD ≈ `7.7/n`** (verified across n=20…120: `n·δ` ≈ 7.3–7.9). **`MDD₈₀ ≈ 8/n` is the
  rule of thumb to code.**

| n | observable floor `6/n` | 80%-power MDD |
|---|---|---|
| 20 | 30.0 pp | 36.7 pp |
| 30 | 20.0 pp | 25.1 pp |
| 38 | 15.8 pp | 20.1 pp |
| 40 | 15.0 pp | 19.1 pp |
| 60 | 10.0 pp | 12.9 pp |
| 85 | 7.1 pp | 9.2 pp |
| 120 | 5.0 pp | 6.6 pp |

**Every report prints, computed from its own actual n, not hardcoded:**

> `This pack resolves differences of >=X pp with 80% power at n=N (paired). Differences below
> Y pp (=6/N) cannot reach significance at any observed outcome.`

Reassurance worth stating once in the plan: the effects this lab has actually needed to resolve —
the qwen3-4b turn-4 collapse (97.5% vs 0%), the ministral duplicate-instruction defect (30%) — are
30–100 pp, comfortably above every MDD in the table. The tool is fit for its stated purpose and
honest about not resolving anything smaller.

### 7.2 Per role

| Role | n | Unit of n | Instrument | Observable floor | MDD₈₀ | Existing golden data adequate? |
|---|---|---|---|---|---|---|
| **tool-caller** | 48 (3 shapes × 4 scripts × 4 reps) | **conversation**, clustered in 12 scripts | cluster bootstrap over scripts + McNemar on `cleanThroughTurn4` | 12.5 pp (if unclustered) | 16 pp (unclustered) → up to ~65 pp (fully clustered) | **No — must be built (FR-22).** §4.5's 4-scripts-per-shape is the recommendation, not §8.2's single script per shape. |
| **guard-judge** | 85 total, but the decision is **class-conditional** | item | McNemar per class | see below | see below | **Partly** — see §7.3 |
| **nlq-generator** | 40 | item | McNemar on Layer-1 exact match | 15.0 pp | 19.1 pp | **Yes, marginally.** Answers "clearly better" only. |
| **chat-responder** | 30 (new) | item | McNemar on checklist pass | 20.0 pp | 25.1 pp | **No — does not exist** (§6.2) |
| **embedder** | 38 queries | query | paired bootstrap on per-query MRR | n/a (continuous) | see §7.4 | **Yes for MRR; no for recall@k** — see §7.4 |

### 7.3 Guard-judge — n=85 is misleading

The decision-relevant statistic is class-conditional (a bias-to-suspend judge is mis-gated by
pooled accuracy for the same reason §6.2's judge is), and the class slices are small:

| slice | n | statistic | observable floor | MDD₈₀ |
|---|---|---|---|---|
| `clear_suspend` | 40 | **false-advance rate — the primary** | 15.0 pp | 19.1 pp |
| `clear_advance` | 30 | advance-recall | 20.0 pp | 25.1 pp |
| `boundary` (all expected-suspend) | 15 | false-advance on near-misses | 40.0 pp | 53 pp |

So: **the naive read "n=85 → ~9 pp" is wrong.** The honest figures are 19 / 25 / 53 pp. The
`boundary` tier is **descriptive only** at n=15 and must be printed with `no significance claim` —
it is also the tier where disagreement is legitimate by construction, so pooling it into an overall
accuracy would let a model look better or worse for the wrong reason. Report the three tiers
separately, always; never a pooled 85-item accuracy as a headline. Pooled accuracy and κ stay as
diagnostics with marginals.

### 7.4 Embedder — the ceiling finding

**recall@k on the current 38-item set has zero power to certify a better model.** Computed here:

- recall@10 = 37/38. Only **1** item is available to win. McNemar needs **6**. It can **never**
  fire in the "candidate is better" direction, at any effect size.
- recall@5 = 34/38. Only **4** items available. Same conclusion.
- Marginal Wilson intervals do not separate even against a perfect 38/38 (Wilson [0.908, 1.000] vs
  [0.865, 0.995]).

The asymmetry is worth stating plainly because it is decision-relevant: **on recall, this set can
detect a materially *worse* embedder but cannot certify a *better* one.** Report recall@10 as the
harness sanity floor (§5.4) and as a regression detector, not as a comparison metric.

**MRR is the discriminating metric — the requirement already said so and the data confirm it.**
0.6259 leaves 0.374 of headroom. Its MDD is `1.96 · sd_d / √n` where `sd_d` is the standard
deviation of per-query MRR differences, which is **data-dependent and must be computed and printed
from the actual run**, not assumed:

| sd of per-query MRR difference | MDD at n=38 | MDD at n=60 |
|---|---|---|
| 0.10 | 0.032 | 0.025 |
| 0.20 | 0.064 | 0.051 |
| 0.30 | 0.095 | 0.076 |
| 0.40 | 0.127 | 0.101 |

**Verdict: 38 queries are adequate for MRR and for score separation; inadequate for recall@k
(saturated) and uninformative for precision@k (algebraically redundant, §5.1).** The cheapest
improvement, if the embedder role becomes decision-critical, is **~22 additional harder queries
(38 → 60), several with |R| ≥ 3** — which simultaneously de-saturates recall@k, makes precision@k
informative, and tightens the MRR interval by ~20%. Name it as a follow-up, not a first-delivery
requirement.

---

## 8. Places where the requirements are methodologically unbuildable as written

Three, in severity order. All three route to `tico` as requirement amendments.

**8.1 (blocker) — FR-15 / AC-4's marginal-overlap rule.** "Two models are declared different only
when their intervals don't overlap," read as two marginal Wilson intervals, **cannot fire at all**
at n ≤ 40 whenever the baseline is ≥ 0.90 — the regime this lab actually operates in (retrieval
recall@10 = 0.974; ministral 0/176). §3.1's worked case: 40/40 vs 34/40, perfectly nested, McNemar
p = 0.031, and the literal rule prints "not distinguishable." It also silently discards the paired
design FR-16 mandates and pays for. **Amendment:** FR-15/AC-4 should read *"the 95% confidence
interval on the **paired difference** must exclude zero"*. Same intent, correctly stated. §3.2's
verdict strings satisfy AC-4's wording under that reading.

**8.2 (major) — FR-8's "argument correctness, split into omitted / wrong value / boundary-unit."**
These are not three disjoint categories: boundary/unit translation is a **subset** of wrong value.
An implementer coding them as siblings will either double-count or arbitrarily prioritize, and two
runs will not be comparable. **Amendment:** state the nesting explicitly, and require each expected
argument in a pack to carry a `boundaryRule` so the classification is data-driven rather than a
regex nobody calibrated.

**8.3 (major) — FR-9/FR-22 do not name the clustering/replication problem.** Fixed scripts plus
replicates means the reported CI describes "this script run again," not "a script of this kind."
§8.2's own precedent (15 replicates of one script per condition) would produce an interval that
could move tens of points if a fourth script were written, with nothing in the report signalling it.
**Amendment:** FR-22 should require **≥3 distinct scripts per conversation shape**, and FR-15
should require the CI on any conversation-level statistic to come from a **cluster bootstrap over
scripts**, with the design effect printed.

**Two smaller items, not blockers, worth folding into the plan rather than amending:**

- **FR-12's precision@k** is algebraically redundant with recall@k on the copied golden set
  (§5.1). Keep it for compliance, footnote it.
- **FR-8 has no "restraint" count** — no measure of *not* calling a tool when none was required
  (§4.2). Add it; it is one line and without it a trigger-happy model scores perfectly.

---

## 9. Evaluation design — how to prove the harness itself is right

The harness is an instrument; an uncalibrated instrument produces confident wrong numbers. Five
checks, all cheap, all implementable as unit/integration tests:

1. **Statistics module regression fixtures.** The five worked (a,b,c,d) rows in §3.2c, with their
   expected McNemar p and MOVER-D bounds, plus the McNemar floor table (c=0→b=6, c=1→b=8,
   c=2→b=10). Pure functions, no network. *Threshold:* exact match to 3 decimal places.
2. **Metric cross-check against falkor-chat.** Run `test_metrics.py`'s existing fixtures through
   the copied recall@k/MRR. *Threshold:* byte-identical outputs.
3. **Embedding-path self-test.** Rank the 38 golden queries using the copied
   `golden_retrieval.embeddings.json` vectors and the copied corpus. *Threshold:* recall@10 ≥ 0.85
   (a bug detector, per §5.4 — not a quality gate).
4. **Prose-call detector calibration.** ~20 human-labelled replies; report the detector's own
   precision/recall in every tool-caller run. *Threshold:* the numbers are printed; no pass/fail
   gate, because the requirements rule out hard gates and §4.2's use is diagnostic.
5. **Judge calibration gate** (chat-responder only, §6.2): false-pass rate ≤ 2/20 on gold-unfaithful
   items **and** parse-failure rate ≤ 0.05, else judge-mediated numbers are suppressed.

**A negative control worth the twenty minutes:** run the paired comparison with **the same model in
both arms**. The correct output is "not distinguishable" with b ≈ c and a difference CI centred on
zero. Anything else means the pairing, the session handling, or the RNG seeding is wrong. This is
the single highest-value test in the list — it catches an entire class of harness bugs that would
otherwise present as plausible model differences.

---

## 10. Risks and open questions

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Script-level clustering treated as independent replication → confidently narrow, wrong CIs | **high** | §4.5: ≥3 scripts per shape, two-level cluster bootstrap, print the design effect |
| R2 | Conditional-denominator laundering — a model that collapses early scores *better* on (c)–(g) | **high** | §4.3: mandatory funnel table, `k/n` on every rate, printed `n/a` tallies, paired-n intersection |
| R3 | Judge-mediated chat-responder number read as equal in strength to a ground-truth one | **high** | §6.2: separate table, fixed banner, split self-preference caveats, hard error on judge==candidate |
| R4 | Corpus embedding cache shared across models or prefix settings → invalid comparison, no visible trace | medium | §5.5: cache key = (model, quantization, docPrefix, corpus version) |
| R5 | Multiple comparisons across 7 counts × turn positions → a false "better" is near-certain | medium | §3.3: one pre-registered `primaryMetric` per pack; everything else labelled exploratory |
| R6 | Temperature-0 replicates counted as independent n | medium | §4.5: print `temperature` and `replicatesPerScript` next to every conversation-level n |
| R7 | Copied golden data loses its provenance chain at the copy boundary | low | §5.4: `copiedFrom` + `copiedAt` + source git SHA on every copied artifact |
| R8 | A copied `judge.py` drifts from falkor-chat's and the two get conflated | low | Record `judgePromptVersion` in the fingerprint; state in the pack docs that the two are independent by design |

**Open questions for the architect / stakeholder:**

1. **Does the stakeholder fund the chat-responder golden data (30 items + 30 calibration items)?**
   §6.3. This is a first-delivery scope question, not a method question — the method is settled
   either way, but which of the two designs ships depends on the answer.
2. **`primaryMetric` per pack** (§3.3) — the recommendations in the table are mine; the stakeholder
   may prefer a different headline for the guard-judge in particular (false-advance rate vs.
   advance-recall) depending on which error is costlier in the product.
3. **Extending the retrieval golden set to ~60 queries** (§7.4) — named as a follow-up, not
   proposed for first delivery. Trigger: the embedder role becoming decision-critical, or a run
   where recall's saturation blocks a real comparison.
