# Small-Model Benchmarking — Statistics and Metric Definitions

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — · **Version:** 1.10

2026-09-03 (v1.10, `data-scientist`) — §11.7's renderings are **measured** (50 warm calls against
`qwen/qwen3-4b-2507`, following one cold load), and taking them reversed two of v1.9's own rulings:
LM-Studio-side TTFT **excludes** the JIT load (cold call 3 625.0 ms wall against 49.7 ms `ttft`), so
§11.4 keeps `ttftMs`/prefill/`tokensPerSecond` on a contaminated item with **their own denominator**
rather than withholding them, and slot 3 stops naming a load cost — v1.9's *"about 21 s"* is false
of a 3.625 s load. New **§11.5.1**: the same gap is a detector for R-14's in-call-reload residual
(3 485.6 ms cold against −11.3…+7.6 ms warm, 461×), with a 1 000 ms threshold and its basis.

2026-09-03 (v1.9, `data-scientist`) — plan R-13 closed on all three of its inputs: new **§11**
settles the latency percentile as **Hyndman-Fan type 1** (inverse ECDF, integer-ceiling rank, one
implementation shared by both call sites), fixes the denominator at the unnetted item count with the
withholding applied per *item* across the whole FR-11 timing block (review G3-3), and sets **two
floors** — an identity floor at X = 20 that **renames** a tail figure `max` where a p95 would be the
maximum (review G3-11), and a level floor that makes a figure **absent** where its attained level
falls more than 5 points below its nominal one. The report block is published as a verbatim clause
grammar. §3.2d gains a pointer to the estimator; nothing else changes.

2026-09-03 (v1.8, `data-scientist`) — the clustered path's four unowned or false sentences: §3.2e
publishes the cluster-path label verbatim (it was prose invented in code) and stops it asserting
clustering at DEFF 1.00, verdict 2's alternate clause becomes *"at or above that"* (equality is
reachable), §7.1's floor sentence gains its effective-unit qualifier where DEFF > 1, and §3.4
Rule 4 replaces the bare percentile interval with the **conservative envelope** — review Pass 4
(M-ML-8, m-ML-9, m-ML-10, m-ML-11).

2026-09-03 (v1.7, `data-scientist`) — four stale or unhandled corners the v1.6 fix round surfaced:
§3.4 Rule 2's dataclass sketch and Rule 4's precondition 3 now say **two** αs on `ResolvingPower`
(not three) and Rule 4 gains its fifth precondition, §3.2e verdict 2's closing clause becomes
conditional (it was false whenever the observed difference exceeded the strict-dominance MDD),
§7.1 states that the floor sentence prints independently of the MDD's attainability, and §7.2's
rendered example is brought onto §7.1's v1.6 template. §3.2a's `3.0 × 10⁻⁴ pp` becomes `3.1` (it
was below the value it bounded — review n-ML-2, the last open Pass 1 item).

2026-09-03 (v1.6, `data-scientist`) — the observable floor moves to the **unadjusted α** (review
M-ML-6), §3.4 Rule 3 is generalised into the printed-bound principle that governs it, Rule 7 splits
by path, and Rule 4 states that McNemar is permitted as a **veto**.

2026-09-03 (v1.5, `data-scientist`) — §3.2c's fixtures republished at 10 dp so the mandated 1e-9
tolerance is assertable (review m-ML-1), the floor/MDD rounding directions separated and three §7.1
floor cells corrected, and §3.4 gains Rule 7 (no `distinguishable` below the observable floor).

2026-09-03 (v1.4, `data-scientist`) — §4.6 adopts the plan's `H` semantics (manifest-declared,
validated `H ≤ min(script length)`, no longer *equal to* the minimum), closing review N-3; pairs
with plan v1.4.

2026-09-02 (v1.3, `data-scientist`) — `primaryMetrics` renamed **`verdictMetrics`** throughout
(`architect`'s naming authority, plan v1.3); semantics unchanged, `headlineMetric` unchanged.

2026-09-02 (v1.2, `data-scientist`) — tool-caller resampled to **12 distinct scripts × 1 run**,
`guard-judge` given **no headline metric**, S3's self-check pinned as diagnostic-only (three
stakeholder decisions), and the statistics-module contract fixed in **§3.4** so neither the
resolving-power line nor `verdict()` can be built from a raw *n* (gate B-1, M-1).

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

**Settled inputs this version encodes (stakeholder, 2026-09-02 — decided, not open):**

1. **Tool-caller sampling is 12 distinct conversation scripts, one run each, at temperature 0** —
   replacing 4 scripts × 4 replicates per shape. Taken on §4.5's own argument. §4.5 and §7.2 are
   rewritten around it; §4.5 also states, in full, what the lab gives up by taking it.
2. **`guard-judge` declares no headline metric.** Both class-conditional error rates carry a verdict,
   with equal weight and no single number above them (§3.3, §7.3).
3. **The S3 embedder self-check is a diagnostic, never a gate** (§5.4).

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

**The constant, settled (gate M-1).** **`z = 1.959963984540054`** is authoritative. It is
`Φ⁻¹(0.975)` — verified in this session: `statistics.NormalDist().inv_cdf(0.975)` returns
`1.9599639845400536` — and it is what `falkor-chat/server/tests/eval/nlq_scoring.py:59` already
pins as `_Z_95`. **`1.96` is a typographic rounding of that number, not a competing convention**, so
the apparent split in this lab's prose (the salesperson review says "1.96") is a split in how the
same constant is written, not in which constant is meant. Implement it as a module constant with
`z` **keyword-only**, exactly as `nlq_scoring` does.

**What the precision is worth — measured, not asserted.** Recomputing §3.2c's five regression
fixtures at both values moves every MOVER-D bound by **at most 3.1 × 10⁻⁴ pp** (measured
3.0167 × 10⁻⁴ pp, on the `34,6,0,0` row's **upper** bound; v1.2–v1.6 printed `3.0 × 10⁻⁴`, which is
*below* the value it claims to bound — an *"at most"* bound rounds **up**, §3.4 Rule 3, review
n-ML-2). That is invisible at the 0.1 pp the report prints and invisible at any
tolerance looser than ~10⁻⁵. **So the fixtures pass under either constant, and M-1 is not a
numerical defect.** The reason to pin the exact value is reproducibility of an *equality* assertion:
two modules carrying two constants disagree in the fifth decimal forever, and the day someone
tightens the fixture tolerance (§9.1 does — see below) the failure appears in a module nobody
changed. Pin one constant; pin this one.

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

| a | b | c | d | n | diff | MOVER-D lower (pp) | MOVER-D upper (pp) | McNemar p (exact rational) |
|---|---|---|---|---|---|---|---|---|
| 34 | 6 | 0 | 0 | 40 | +15.0 pp | 3.1762869443 | 29.0723243665 | 1/32 = 0.03125 |
| 30 | 6 | 0 | 4 | 40 | +15.0 pp | 3.8506738324 | 27.7026867131 | 1/32 = 0.03125 |
| 33 | 6 | 1 | 0 | 40 | +12.5 pp | −0.9864868353 | 26.8581964973 | 1/8 = 0.125 |
| 20 | 8 | 2 | 10 | 40 | +15.0 pp | 0.1708978316 | 28.7785182732 | 7/64 = 0.109375 |
| 72 | 10 | 2 | 1 | 85 | +9.4 pp | 1.4800198994 | 18.2130920778 | 79/2048 = 0.03857421875 |

Use these five rows as the implementer's regression fixtures for the statistics module. **Ten
significant decimal places in percentage points, i.e. 1e-12 as a proportion** — three orders inside
the tolerance below, which is the point (see the trap at the end of this subsection). All ten bounds
were re-derived at 60-digit precision at `z = 1.959963984540054` in this session and agree with the
independent re-derivation in `docs/reviews/small-model-benchmarking-ml.md`; the p-values are exact
rationals and are published as rationals so no decimal expansion is load-bearing.

**Assertion tolerance — settled, replacing both "exactly" and "3 decimal places" (gate m3).**
- `mcnemar_exact` is rational arithmetic over `math.comb` and `0.5**m`: assert to **1e-12
  absolute**. It is exact; a looser tolerance hides an implementation that is not.
- MOVER-D bounds: assert to **1e-9 absolute on the proportion** (i.e. 1e-7 pp). That is ~7 orders
  above double-precision noise for values of this magnitude, so operation-order differences cannot
  trip it, and it is ~4 orders tighter than the z-constant divergence above — meaning **this
  tolerance is what makes M-1's constant load-bearing.** "Exact match" is not assertable across
  platforms; 3 decimal places is loose enough to pass a genuinely wrong Wilson.

**The trap this note fell into, stated once because it generalises (review m-ML-1).** v1.2–v1.4
published these bounds at 4 dp *in pp* and mandated a 1e-9 *proportion* tolerance in the same
subsection. Those two statements are incompatible: 4 dp in pp is 1e-6 as a proportion, so the
published `3.1763` sits 1.31e-7 from the true `3.1762869443…` — **131× the tolerance it was supposed
to be asserted against**. The delivered implementation was never the problem; it agrees with the
60-digit value to 1.44e-16, about 7×10⁶ *inside* the mandate. The table was under-precise, not wrong.

> **A fixture table published for display cannot carry a tolerance tighter than its own printed
> precision.** The two numbers are one decision, not two, and they are usually written in different
> sentences by different people at different times — which is exactly how they drift apart. Whenever
> a tolerance is tightened, the fixtures must be republished at a precision that clears it, and when
> a table is published, the tolerance it can support is fixed at that moment. The safe default is to
> publish **at least 2–3 orders finer than the tolerance** so a later tightening has headroom, and
> to prefer **exact rationals** wherever the quantity has one (as the p-values above now do), since
> a rational carries no precision claim to get out of step.

**(d) Continuous metrics — paired percentile bootstrap on the per-item difference.**

McNemar does not apply to MRR, score separation, or latency. Use a seeded paired bootstrap:
resample the *items* (with replacement, n draws), recompute the mean per-item difference each
time, B = 10 000, take the 2.5th/97.5th percentiles (**the percentile estimator is §11's, and
there is one of it**). ~15 lines of stdlib (`random.Random(seed)`).
The seed goes into the environment fingerprint (FR-7) so a report is reproducible. Decision:
**the CI excludes zero.** No separate significance test — for continuous metrics the CI *is* the
test, and reporting both would be redundant, not extra rigour.

For **clustered** data, resample the **outermost independent unit, never the observation**. Under
the settled 12×1 sampling design (§4.5) conversation and script are the same unit, so the tool-caller
resample is **one-level over the 12 conversations**. The two-level (script → replicate) resample
returns the moment any pack declares `replicatesPerScript > 1`; §3.4 makes that a **validation
error** rather than a silently one-level approximation. See §4.4.

**(e) The decision wording, which AC-4 must be checked against.**

Three verdicts, exactly these strings:

1. **Distinguishable.**
   `A is better than B on <metric>: +15.0 pp (95% CI [3.2, 29.1] pp), n=40 paired items (unit: item, design effect 1.00), McNemar exact p=0.031 (b=6, c=0).`
2. **Not distinguishable.**
   `Not distinguishable at this sample size. Observed difference +12.5 pp, 95% CI [-1.0, 26.9] pp covers zero (b=6, c=1, McNemar exact p=0.125). This pack resolves differences of >=19.1 pp with 80% power at n=40 effective items (40 units, design effect 1.00, by-construction, alpha=0.05); the observed 12.5 pp is below that. Neither model is ranked above the other.`

   **The closing clause is conditional, not fixed prose (v1.7).** `the observed X pp is below that`
   is only true when `|diff| < mdd80`, and the case where it is false is not exotic — it is §7.1's
   own *normal case for a model swap*, a candidate that wins more than it loses without strictly
   dominating. At n=30 with `b=5, c=13` the difference is 26.7 pp against an MDD₈₀ of 25.1 pp and
   McNemar's p is 0.096, so the sentence as fixed prose reads *"resolves differences of >=25.1 pp
   with 80% power; the observed 26.7 pp is below that"* — two numbers in one sentence that refute
   it. When `|diff| >= mdd80`, render instead:

   > `…; the observed 26.7 pp is at or above that, but the MDD assumes strict dominance and this comparison is not strictly dominant (b=5, c=13), so the difference required for 80% power at this discordance mix is larger (§7.1).`

   The discordance counts are mandatory in that form: they are the reason the two numbers point
   opposite ways, and without them the sentence looks like the instrument contradicting itself.

   ***"at or above", not "above" (v1.8, review m-ML-9).*** The branch is `|diff| < mdd80`, so
   **equality takes this wording** — which is the right branch, since `mdd_clause` claims the pack
   resolves differences of **≥** `mdd80` and at equality the difference is resolvable. But `mdd80`
   is ceilinged onto a 0.1 pp grid while an observed difference is `(b−c)/n`, so the two coincide
   exactly whenever the grid and the lattice meet, and **that is reachable at this component's own
   sample sizes** — measured this session: `n_units = 85`, `k = 2`, `DEFF = 1.9` gives
   `mdd80 = 20.0 pp` and `|b−c| = 17` gives exactly 20.0 pp; sweeping `6 ≤ n_eff ≤ 200`, equality
   occurs at (n=90, 100, 120, 150, 180) for k=2 and (n=200) for k=1, and at n=90 it is 2.4% of
   every table that reaches this clause. A strict *"is above that"* is then false in the same way
   *"is below that"* was, one branch over. **Do not add a third wording for equality** — one
   comparative that is true across the whole branch is worth more than a third string to keep in
   step with the other two.
3. **Instruments disagree** (MOVER-D excludes zero, McNemar does not — row 4 of the table above; real and not rare):
   `Not distinguishable at this sample size. The effect-size interval [0.2, 28.8] pp excludes zero but the exact paired test does not reach alpha=0.05 (b=8, c=2, p=0.109). Reported as not distinguishable: the exact test is the decision rule.`

**McNemar exact is the decision; MOVER-D is the effect size.** One instrument decides, one
quantifies. Do not AND them into a bloc — but *always print both individual outcomes in the
prose*, as verdict 3 does, so a reader never sees an aggregate verdict without the two components
that produced it.

**One precondition on all three strings, and it is the whole of gate B-1:** McNemar exact and
MOVER-D are valid **only when each row of the paired table is one independent analysis unit**. When
the analysis unit contains correlated observations (design effect > 1), both are anti-conservative
and the decision rule becomes *"the cluster-bootstrap CI on the paired difference excludes zero"*,
with the strings rendered against the bootstrap instead of McNemar and the design effect and its
basis printed. §3.4 makes this a property the code cannot get wrong by omission.

**(f) The clustered-path label — published here, because Rule 4 required it printed and this note
published no string for it (v1.8, review m-ML-10).** It is **appended to whichever of the three
strings above was rendered**, on every verdict whose `decided_by` is `cluster-bootstrap` — not on
two of them, because a reader who sees one verdict must still be told which instrument produced it.
Two variants, and the condition is the **design effect**, never the basis:

1. **A widening was applied (`design_effect > 1.0`):**

   > `Decided by the cluster-bootstrap CI on the paired difference, widened by sqrt(DEFF)=1.41 for the declared clustering, in conjunction with McNemar's exact test (p=0.031) as a necessary condition: under clustering McNemar rejects too readily, so it may withhold a verdict but never carries one on its own.`

2. **No widening was applied (`design_effect == 1.0`), which is *every* comparison until a
   determinism probe establishes the basis:**

   > `Decided by the cluster-bootstrap CI on the paired difference — the instrument here because this comparison's design effect is assumed rather than established by construction — with no widening applied (sqrt(DEFF)=1.00), in conjunction with McNemar's exact test (p=0.031) as a necessary condition: a design effect that was never established cannot license the exact test to carry a verdict, so it may withhold one but never carries one on its own.`

   `assumed` is the `basis` field verbatim (`measured` is the other value that reaches this path).

Two things the second variant fixes, both of them the same defect this note keeps ruling on — a
clause that is true on the rare path and false on the common one:

- **The reason clause must attach to the instrument choice, not to the widening.** What displaced
  McNemar is the **basis**; what left `sqrt(DEFF)` at 1.00 is the **design effect**. A single
  *"not widened …, because the design effect is assumed"* reads as the basis explaining the
  widening, which is not the causal chain and is the nearest-attachment parse.
- **Do not say *"under clustering"* where no clustering is declared.** At `design_effect == 1.00`
  the comparison asserts none, so the rationale has to be the one that is actually true there: an
  unestablished design effect cannot license the exact test to carry a verdict. The
  clustering rationale is right in variant 1 and false in variant 2, and variant 2 is the default
  path.

**FR-15's literal marginal-overlap check should still be computed and printed as a diagnostic
line** ("marginal Wilson intervals overlap: yes/no"), because the requirement asks for it and it
costs nothing — but it must be labelled *diagnostic*, never the verdict, with a one-line footnote
naming why (§3.1's inertness). That keeps the requirement visibly honoured while the decision is
made correctly.

### 3.3 Multiple comparisons

Seven tool-calling counts × ~9 turn positions × several metrics is dozens of tests; at α=0.05 each,
a false "better" is close to certain. Cheapest correct handling, and the recommendation:

**Each task pack pre-registers, in its versioned config, a `verdictMetrics` list of one or more
metrics, and a `headlineMetric` that is either exactly one member of that list or `null`.** Only
members of `verdictMetrics` receive a better / not-distinguishable *verdict*; every other number is
printed with its CI and labelled `exploratory — no significance claim`. Pre-registration in pack
config is what stops the verdict-carrying metrics from being chosen after the results exist.

*(v1.2/v1.3 — this **retires** the singular `primaryMetric` rather than redefining it.
Stakeholder decision 2 gives `guard-judge` two co-equal verdict metrics and no headline, so a
one-metric schema cannot express the pack that most needs expressing; and re-pointing an
established name at "may now be `null`" is a trap, since the old meaning is already fixed in the
requirements, the review and two plan versions. The replacement is named `verdictMetrics`
(`architect`, plan v1.3) — a plural one character away from the retired singular would be
indistinguishable in a JSON manifest and in a diff, which is intolerable for the one field whose
entire job is pre-registration. The two fields are separable and both are needed:
**`verdictMetrics` controls inference** — `len(verdictMetrics)` *is* the multiplicity *k*, hence the
correction below — while **`headlineMetric` controls presentation**, i.e. what a reader is entitled
to read as "the" number. A pack with `headlineMetric: null` prints its verdict metrics side by side,
in a fixed declared order, with no summary line above them and no arithmetic combining them.)*

| Role | `verdictMetrics` | `headlineMetric` | Unit |
|---|---|---|---|
| tool-caller | `cleanThroughTurnH` (`H` manifest-declared, validated `≤ min(script length)`; 4 here) | same | conversation |
| **guard-judge** | **`["falseAdvanceRate", "falseSuspendRate"]`** — scored on the 40 `clear_suspend` and 30 `clear_advance` items respectively | **`null`** | item |
| nlq-generator | Layer-1 exact-match rate | same | item |
| chat-responder | deterministic checklist pass rate | same | item |
| embedder | MRR | same | query |

**When `len(verdictMetrics) > 1`, family-wise error control is mandatory, not optional.** Two
co-equal verdicts at α=0.05 each carry a ~9.75% chance of at least one false "better" under the
null — which would hand the stakeholder exactly the fishing artefact pre-registration exists to
prevent. Apply **Holm–Bonferroni across the declared family** (order the p-values; test the
smallest at α/k, the next at α/(k−1), …, stopping at the first non-rejection) and print the
adjusted threshold beside each p-value.

**And it changes what the resolving-power line must print — for the MDD only.** The
guaranteed-detectable threshold for a k-member family is computed at **α/k**, since a metric can be
required to clear that step. For `guard-judge` (k=2) that is α=0.025, and §7.3 carries the
recomputed figures — materially worse than the α=0.05 ones, which is the honest price of two verdict
metrics rather than one. §3.4 Rule 4 is the mechanism: `verdict()` raises unless
`resolving.alpha_mdd == alpha_family / len(family)`, so a k-member family cannot report its MDD at
α=0.05 by oversight.

**The observable floor takes the opposite α, on purpose — do not "fix" the two to match.** The floor
stays at the **unadjusted α=0.05** whatever *k* is. The two bounds carry different sentences, and
each takes the α that makes its own sentence true (§3.4 Rule 3): the MDD promises power *whatever
rank the member draws*, so it needs the tightest step; the floor asserts an impossibility *at any
step the member could face*, so it needs the loosest. Setting the floor to α/k understates nothing —
it **overstates** what is impossible, and prints a threshold that an ordinary Holm rejection walks
straight under. It also double-charges the multiplicity: the price of a second verdict metric is
already paid in the MDD, and paying it again in the floor collapses Holm to Bonferroni for any
difference in `[6/n, 7/n)` — exactly the band §7.3 prices.

*Reversal trigger:* if the stakeholder later ranks the two guard-judge errors, drop the loser out of
`verdictMetrics` into the exploratory block and the family collapses back to k=1 with α=0.05
(floor 15.0 pp instead of 17.5 pp on `clear_suspend`). That is a real gain in resolving power and
it is available whenever the product question "which error costs more?" becomes answerable.

### 3.4 The statistics-module contract (`stats.py`) — the B-1 guard

Gate B-1 is not "the plan chose the wrong formula". It is that **the plan's signatures made the
wrong thing the easy thing**: `min_detectable_difference(n: int)` accepts a turn count, a
conversation count and a replicate count identically, and `verdict()` accepts 48 correlated rows as
readily as 12 independent ones. A note that only says "be careful about clustering" reproduces the
defect at the next pack. So the contract below is written so that **the anti-conservative version
does not typecheck, and the honest one is the only one that runs.** Types are illustrative; the
seven rules are binding.

**Rule 1 — the paired table is constructible only from independent analysis units.**

```python
@dataclass(frozen=True)
class PairedOutcomes:
    unit_kind: str                    # pack-declared: "conversation" | "item" | "query"
    unit_ids: tuple[str, ...]         # the cluster keys, one per row
    a_correct: tuple[bool, ...]
    b_correct: tuple[bool, ...]

    @classmethod
    def from_units(cls, unit_kind: str,
                   rows: Iterable[tuple[str, bool, bool]]) -> "PairedOutcomes":
        """Raises DuplicateAnalysisUnit if any unit id appears more than once."""

    @property
    def table(self) -> tuple[int, int, int, int]: ...   # (a, b, c, d)
    @property
    def n_units(self) -> int: ...                       # == len(unit_ids)
```

There is **no other constructor**, and `from_units` raising on a repeated unit id is the mechanism:
the old 48-conversations-from-12-scripts design cannot reach `verdict()` at all, because the script
id repeats four times. A pack that legitimately buys replicates must collapse them first —
`collapse_replicates()` returning one *rate* per script — at which point the data are continuous and
McNemar no longer applies, which is the correct outcome, not an inconvenience.

**Rule 2 — resolving power is computed from effective units, and its inputs have no defaults.**

```python
@dataclass(frozen=True)
class ResolvingPower:
    n_units: int
    unit_kind: str
    alpha_family: float          # unadjusted 0.05 — the FLOOR's alpha, whatever k is (Rule 3)
    alpha_mdd: float             # 0.05/k for a k-member verdictMetrics family — the MDD's (§3.3)
    design_effect: float         # Kish; >= 1.0
    basis: Literal["by-construction", "measured", "assumed"]
    n_effective: float           # n_units / design_effect
    observable_floor: float      # b_min(alpha_family, c=0) / n_effective
    mdd80: float | None          # exact, see rule 3; None when no effect size attains the power

def resolving_power(n_units: int, *, unit_kind: str, design_effect: float, basis: str,
                    alpha_family: float, alpha_mdd: float,
                    power: float = 0.80) -> ResolvingPower: ...
```

`design_effect`, `basis`, `unit_kind` and both αs are **keyword-only with no default value**. A
default of `1.0` would rebuild B-1 by omission — the caller who forgets clustering is exactly the
caller the gate found. `min_detectable_difference` takes **`n_effective: float`** (never `n: int`),
so passing a raw observation count is a visible mislabel at the call site rather than an invisible
one inside the function. `observable_floor` still takes an `alpha` parameter — `b_min` is a function
of α in general, not the constant 6 — but **`resolving_power` must call it with `alpha_family`, the
unadjusted α, never with `alpha_mdd`** (Rule 3's α row). Keeping the parameter while fixing the
call site is deliberate: the formula stays general, and the *choice* lives in one auditable place.

**Rule 3 — the printed-bound principle. Every printed bound takes the rounding direction, the α,
and the denominator that keep *its own claim* true.**

This is one rule, not a list of special cases, and it is the rule three separate adjudications in
this document's review history each rediscovered the hard way. A printed bound is a *sentence*, and
the sentence is what has to be true — so every free parameter in computing it is fixed by asking
"which value makes this sentence true?", never by consistency with the bound printed next to it.
Two bounds sitting side by side routinely take **opposite** values of the same parameter, and that
is correct rather than sloppy. The three instances, all live in this note:

| parameter | MDD — *"resolves differences ≥ X with 80% power"* | observable floor — *"differences below Y cannot reach significance at any observed outcome"* |
|---|---|---|
| **rounding** | **up** — at the printed value the power claim must hold (19.0 pp has power 0.798) | **down** — a printed 15.8 at n=38 is false, since the attainable `6/38 = 15.789` reaches significance |
| **α** | **tightest** step, `α/k` — the claim must hold whatever rank the member takes under Holm | **loosest** step, unadjusted `α` — a member *can* be tested at 0.05, so `7/n` asserts a false impossibility |
| **denominator** | **floored** `n_effective` — flooring is what keeps X conservative | **unfloored** `n_effective` — flooring here would *raise* Y and overstate what is impossible |

Each cell is the conservative choice **for that sentence**; unifying any row would make one of the
two bounds anti-conservative, which is why the denominator nit (`n-ML-1`) was correctly declined
rather than harmonised. When a new printed bound is added, derive its three parameters from its
sentence before writing the formula.

**Rule 3a — MDD and floor are computed exactly and rounded in *opposite* directions; the `8/n` rule
of thumb is not code.**
`min_detectable_difference` bisects on δ for the smallest difference at which
`P(reject) ≥ power` under the exact McNemar rejection region, then **rounds up to the printed
precision**. Both halves matter, and this settles gate m3's "8/n gives 20.0 pp where the note says
19.1 pp":
- exact δ at n=40 is **19.046 pp**, and `8/n` = 20.0 pp — the rule of thumb is *conservative but
  wrong*, and printing two different numbers for the same quantity in the same report is the defect;
- rounding **to nearest** would print 19.0 pp, at which measured power is **0.798** — below the
  0.80 the sentence claims. Rounding up to 19.1 pp gives 0.8023.

**And the observable floor rounds the other way — *down*.** One principle, two directions: **round
each printed bound in the direction that keeps its own claim true.**
- MDD carries *"resolves differences ≥ X with 80% power"*, so X must round **up**: at the printed
  value the power claim must hold.
- The floor carries *"differences below Y cannot reach significance at any observed outcome"*, so Y
  must **truncate**: at n=38 the exact floor is `6/38 = 15.789 pp`, and printing the ceiling 15.8
  makes the sentence **false**, because the attainable observed difference `6/38 = 15.789` is below
  15.8 and *does* reach significance (b=6, c=0, p=1/32). Truncating to 15.7 keeps it true.
- The failure mode is not academic: at n=12, α=0.025 the exact floor is `7/12 = 58.333 pp`, which a
  report displays as an observed `58.3`. Ceiling the floor to 58.4 produces a report whose verdict
  line says *distinguishable* while its own honesty line says the difference cannot reach
  significance. Truncation to 58.3 removes the contradiction.
- Exact values (`6/40 = 15.0`) print exactly under either rule; only inexact ones diverge.
- **Truncate with the same expression the caller uses.** `math.floor(x/precision)` and
  `math.floor(x*1000)` are not interchangeable: the double nearest `0.001` is slightly *above* it,
  so `(7/40)/0.001 = 174.99999999999997` truncates across the bin edge to **17.4 pp** where
  `(7/40)*1000 = 175.0` gives the correct 17.5. Swept in this session over `n ≤ 2000`: with
  `b_min = 7` this bites at **n = 5, 10, 20, 40**; with `b_min = 6` it **never** bites.
  **So under v1.6's floor — always `6/n_eff` — the guard is defensive, not load-bearing on any cell
  this note prints.** It was load-bearing under the `α/k` floor that v1.6 removes (17.5 pp at n=40
  was a published cell), and the two changes arrived in the same review pass, which is precisely how
  a justification outlives the thing that justified it. Keep the guard anyway — `b_min` is a
  function of α, not the constant 6, so any future α reopens the hazard, and the cost is one
  expression — and pin it with the **code's own expression**, never an equivalent-looking one.

*(v1.5 corrects three §7.1 cells where v1.2–v1.4 ceilinged an inexact floor — 15.8→15.7 at n=38 and
7.1→7.0 at n=85 in the α=0.05 column, 46.7→46.6 at n=15 in the α=0.025 column. The note was
internally inconsistent, ceiling-rounding the floor in one column and truncating it in the other;
this rule is what makes the direction derivable rather than remembered.)*

**Rule 4 — `verdict()` asserts its own preconditions and refuses when McNemar is invalid.**

```python
def verdict(outcomes: PairedOutcomes, *, resolving: ResolvingPower,
            metric_name: str, family: Sequence[str]) -> Verdict: ...
```
Raises — never warns, never silently proceeds — unless all five hold:
1. `resolving.n_units == outcomes.n_units` (the printed resolving power belongs to *this* table);
2. `resolving.unit_kind == outcomes.unit_kind`;
3. `resolving.alpha_mdd == alpha_family / len(family)` and `metric_name in family` (§3.3's
   multiplicity). **This precondition is unchanged by v1.6 and must not be "fixed" to match the
   floor**: `alpha_mdd` is the *pre-registration* α — the tightest step, which is what makes the MDD
   claim rank-independent;
4. `resolving.design_effect >= 1.0`;
5. `alpha_step`, when supplied, lies in `[alpha_mdd, alpha_family]`. Holm's own steps are
   `α/(k−i)`, which lie in that range by construction, so a step outside it is a caller defect —
   and left unchecked it surfaces one module later as Rule 7's `mcnemar-exact` raise, which
   announces a bug in the wrong place. Checking it here is what makes Rule 7's theorem a *checked*
   premise rather than an assumed one *(v1.7: shipped with the m-ML-6 fix round and adopted here)*.

**Three αs coexist deliberately; `ResolvingPower` carries two of them.** `alpha_family`
(unadjusted, 0.05 — the **floor's**, Rule 3) and `alpha_mdd` (`α/k` — the **MDD's**) are both known
when the object is built and rendered, so both are fields. The third, `alpha_step` (`α/(k−i)`,
Holm's actual data-dependent threshold for this member at decision time), is **not** a field: it is
known only after the family's p-values are ranked, which is *after* this object is built and
printed, so a field would be `None` until it was not and the number would have two homes. It
reaches the one function that uses it as `verdict()`'s parameter. *(v1.7 corrects v1.6, which said
the dataclass carries all three in the same sentence that said the third is unknowable at
construction time. The two-field shape is the one the note requires.)*

And the decision rule branches on the design effect, which is the fix B-1 asks for:
- **`design_effect == 1.0` and `basis == "by-construction"`** → McNemar exact decides, MOVER-D
  quantifies (§3.2b/c). This is the only configuration in which McNemar is valid.
- **otherwise** → McNemar is anti-conservative and must not **decide alone**. The decision is *"the
  cluster-bootstrap CI on the paired difference excludes zero"*; McNemar's p may still be printed,
  labelled `anti-conservative under clustering — not the decision`.

**McNemar as a *veto* is permitted on that path, and is the recommended form.** Making the
non-`by-construction` decision a **conjunction** — distinguishable iff the widened CI excludes zero
**and** `mcnemar_exact(b, c) ≤ alpha_step` — does not violate this rule, and the reason is worth
stating because the rule reads as if it forbids what the fix requires. **The objection to McNemar
under clustering is that it *rejects* too readily.** A necessary condition can only ever *remove*
rejections, so the conjunction is uniformly at least as conservative as either instrument alone: at
DEFF = 1 it restores the exact test's calibration exactly, and at DEFF > 1 the widened interval
remains the binding constraint it already was. What the rule forbids is McNemar deciding *for*
distinguishability under clustering; using it to withhold that verdict is the opposite operation.

**The interval printed on that path is the conservative envelope, not the resample alone (v1.8,
review M-ML-8). This corrects v1.6–v1.7, whose *"the strings rendered against the bootstrap"*
authorised a substitution it never checked the width of.** The veto fixed the **decision** on this
path and left the **quantification** on the bare percentile interval, and that interval is the
narrower of the two available — measured this session, exactly (every paired table enumerated, and
every bootstrap resample outcome computed analytically rather than sampled):

| n, regime | MOVER-D coverage / mean width | percentile bootstrap at DEFF 1.00 | narrower than MOVER-D on |
|---|---|---|---|
| 30, strict dominance δ=15 pp | 0.983 / 30.60 pp | **0.942** / 24.57 pp | **100%** of the probability mass |
| 40, strict dominance δ=15 pp | 0.976 / 25.63 pp | **0.939** / 21.23 pp | **100%** |
| 85, strict dominance δ=12 pp | 0.969 / 15.32 pp | **0.940** / 13.56 pp | **100%** |
| 40, null | 0.962 / 29.55 pp | 0.956 / 26.80 pp | 91% |

So on the path taken **because the design effect was never established** — i.e. every comparison
until a determinism probe runs — the report prints a *tighter* effect size than the instrument
§3.2c mandates, in the strict-dominance regime this tool exists to catch. Worse, the percentile
bootstrap **degenerates at the sparse discordant counts §3.2b says this lab will actually see**:
at n=30 with `b=4, c=0` it returns `[3.3, 26.7] pp`, excluding zero, where McNemar's exact p is
**0.125** and MOVER-D is `[−0.6, 29.7]`. The mechanism is not subtle — with four non-zero rows,
`P(no +1 drawn) = (26/30)³⁰ = 1.4% < 2.5%`, so the 2.5th percentile *cannot* be zero. At n=30
that fires on **20.3%** of the probability mass under strict dominance, each time rendering
verdict 3's *"the interval excludes zero but the exact paired test does not"* on the strength of a
degenerate interval rather than a genuine disagreement.

**The rule.** On any non-`by-construction` path the printed interval is the **wider** of

- the `√DEFF`-widened percentile bootstrap (Rule 6), and
- the `√DEFF`-widened MOVER-D interval, half-widths scaled about the point estimate the same way,

which is uniformly at least as conservative as either alone, reduces to MOVER-D exactly at
`DEFF = 1.00` (restoring §3.2c's effect-size instrument on a path where *nothing about clustering
was ever declared*), keeps the interval responsive to a declared design effect where one exists,
and removes the sparse-count degeneracy because MOVER-D covers zero exactly when the counts are
too thin to support excluding it. It can only *remove* rejections, so it disturbs neither the veto
nor Rule 7.

**And the percentile itself should be computed in closed form, not resampled.** For a paired
**binary** table the per-unit differences take three values, so the resample distribution is
exactly multinomial and its quantiles are a ~30-line exact computation (verified this session
against the shipped `B=10 000` resample). The resampled version is not merely slower — **it is a
Monte-Carlo estimate of an *atomic* quantile, so whenever the target percentile lands within
Monte-Carlo error of an atom boundary the printed bound flips by a whole atom, `1/n_units`.** State
it precisely, because the loose version ("the bound moves with row order") is not reproducible on an
arbitrary table and a rationale nobody can reproduce is a defect in waiting:

- **The predictor is the exact CDF's distance from the target percentile.** At `(b=8, c=0, n=85)`
  the exact bootstrap CDF is `0.97281` at `13/85` and `0.98738` at `14/85`, so the 97.5% quantile
  is `14/85` by only **0.0022** of probability, against a Monte-Carlo standard error of
  `sqrt(0.9728·0.0272/10 000) = 0.0016` — about a 9% chance per draw that the empirical CDF crosses
  first and the bound prints `15.3` instead of `16.5`. Measured: **16 of 200 seeds** and **23 of 200
  row permutations** at a fixed seed. Where the quantile is *not* near a boundary the bound is
  perfectly stable — `(b=4, c=0, n=30)` is identical across 60 seeds.
- **Row order matters for the same reason the seed does, and the reason is not obvious.**
  `random.Random.choice` draws an **index**; a fixed seed fixes the index sequence, and a
  permutation of the same multiset re-maps those indices onto different values. So "same multiset,
  therefore same resample" is false.
- **At the tool-caller pack's own n it is a coin flip, not a tail event.** `(b=5, c=3, n=12)`:
  `paired_cluster_bootstrap(..., B=10 000)` prints a lower bound of `-25.0 pp` at seed 0 and
  `-33.3 pp` at seed 5 — **8.3 pp apart**, at **107/93 of 200 seeds** and **102/97 of 200 row
  permutations**. That is the last *two* digits of a bound printed to 0.1 pp, decided by the seed
  and by the order rows happen to sit in.

A bound at that mercy is not reproducible in the sense §3.2d's seed requirement is asking for, and
the closed form removes both dependencies at no cost.

**Rule 5 — design effect is a variance ratio, not a width ratio.** (A correction to v1.1 §4.4,
which called the width ratio "the design effect"; an implementer following v1.1 literally would have
divided by 2.6 where the truth was 7, over-stating effective *n* by ~2.7×.)

```python
def width_inflation(bootstrap_width: float, naive_width: float) -> float: ...   # the ratio
def design_effect(bootstrap_width: float, naive_width: float) -> float: ...     # ratio ** 2
def effective_n(n_observations: int, design_effect: float) -> float: ...        # n / DEFF
```
CI half-width scales as `1/√n`, so `DEFF = (bootstrap width ÷ naive Wilson width)²` and
`n_eff = n_obs / DEFF`. Check it against §4.4's real case: 280 turns in 40 conversations with
within-conversation correlation ρ ≈ 1 and m = 7 turns each gives `DEFF = 1 + (m−1)ρ = 7`, width
ratio `√7 = 2.646` — v1.1's "≈2.6" — and `n_eff = 280/7 = 40`, which is exactly the conversation
count. The identity is the check: **when ρ = 1, effective n must equal the cluster count.** Make
that a unit test; it is the one assertion that catches a squaring error in either direction.

**Rule 7 — no verdict path may return `distinguishable` when `|diff| < observable_floor`.** This is
a **required property of the verdict path itself, asserted in `verdict()`**, not a test the
implementer may or may not write. It costs one comparison and it is the only cheap check that ties
together two quantities computed by completely independent routes — the floor comes from
`b_min(α)/n_effective`, the verdict from McNemar's tail or the cluster bootstrap — so a defect in
either surfaces as a contradiction rather than as a plausible number. It is what would have caught a
bootstrap that silently resamples correlated rows i.i.d.: such a substitution changes the
instrument's name and not its interval, and an interval alone cannot report that.

Three details that decide whether it works:
- Compare against the **exact** floor, never the display-rounded one (Rule 3) — otherwise the
  invariant inherits the presentation layer's rounding and can fire or fail to fire by 0.05 pp.
- **The converse is not an invariant.** `|diff| ≥ observable_floor` does *not* imply
  `distinguishable`; the floor is necessary, never sufficient. Asserting the converse would be a
  bug that hides the discordance structure McNemar exists to read.
- **The response splits by path, because the invariant has two different statuses.**
  - On **`mcnemar-exact`** it is a **theorem, so a fire is a module bug → raise.** With the floor at
    the unadjusted α (Rule 3), `p ≤ α_step ≤ 0.05` implies `|b−c| ≥ b_min(0.05) = 6`, hence
    `|diff| ≥ 6/n`, at *every* Holm step. Verified exhaustively in this session over all `(b, c)`
    with `b + c ≤ 400`: **zero violations at α=0.05 and zero at α=0.025.** Silently demoting here
    would discard exactly the detector property this rule exists for.
  - On **`cluster-bootstrap`** it is a **guard, so a fire demotes and names** — the verdict becomes
    not-distinguishable with the floor breach printed as the reason. Raising would abort on ordinary
    clustered data: at DEFF = 2 on `(34, 6, 0, 0)` the widened interval still excludes zero at a
    15.0 pp difference while the floor has moved to 30.0 pp, which is a legitimate disagreement
    between a widened interval and a shrunken effective *n*, not a defect.
  - **This split only works with the floor at the unadjusted α.** At `α/k` the McNemar branch is
    reachable — a member rejected at a 0.05 Holm step with `|b−c| = 6` sits below a `7/n` floor —
    so the theorem is not available and `raise` would fire on correct data.

**Rule 6 — `cluster_bootstrap` is one-level, and a pack that needs two must fail validation.**

```python
def cluster_bootstrap(units: Sequence[Sequence[bool]], *, B: int = 10_000,
                      seed: int) -> BootstrapResult: ...
```
Each inner sequence is the observations belonging to one independent unit (for a turn-pooled count:
the turns of one conversation). Under 12×1 there are 12 members. `validate` **fails any pack
declaring `replicatesPerScript > 1`** while only this one-level function exists — the two-level
resample (§4.5) is then required and its absence must be an error, not an approximation.

**On the `√DEFF` widening used as the interim substitute, and why it is safe rather than merely
cheap.** Widening a percentile interval by `√DEFF` is exact for what it claims — DEFF is a variance
ratio and a CI half-width scales as `1/√n`, so `√DEFF` is precisely the factor carrying an interval
computed at `n_units` to one computed at `n_units/DEFF` — but it **rescales a variance it cannot
discover**, and in particular it does **not reproduce the few-clusters degeneracy** a genuine
cluster bootstrap has (§4.5.1(ii) rejects bootstrapping 3 shapes for exactly that reason). It only
widens; it never becomes unusable. **That gap is closed by a different mechanism, and the two
compose:** the floor is computed at `n_eff = n/DEFF`, so a too-few-clusters configuration is caught
by Rule 7 or by `UnattainablePower` rather than by the interval. Verified: at n=12, DEFF=7 the floor
is `6/1.714 = 350 pp`, i.e. no difference is resolvable at all and the run says so. **Neither
mechanism is sufficient alone** — the widening handles variance the floor cannot see, the floor
handles degeneracy the widening cannot — which is what makes the interim defensible until a pack
with `replicatesPerScript > 1` makes the structural primitive buildable against real data.
**A third mechanism was missing and v1.8 adds it:** neither the widening nor the floor sees the
*sparse-discordant-count* degeneracy of a percentile interval — at four non-zero rows in thirty the
2.5th percentile cannot be zero, whatever the design effect — and that is what Rule 4's
conservative envelope with MOVER-D closes (review M-ML-8).

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
3. **The report opens with a funnel table**, not a metric table (the counts below are an
   *illustration of the shape*, not this pack's sizing — the 12×1 design drives ~80 turns per model,
   §4.5.2 — and the funnel must additionally print, at its head, the **conversation count and the
   analysis unit**, because every rate under it is a turn or call count while every verdict above it
   is computed over 12 conversations):

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
   compute its CI by **cluster bootstrap resampling conversations** (which, under the settled 12×1
   design, *are* the scripts — §4.5), recomputing the pooled rate inside each resample. ~20 lines of
   stdlib. Print two numbers next to it: the **width inflation** (`bootstrap width ÷ naive Wilson
   width`, ≈2.6 in §8.2's data) and the **design effect**, which is that ratio **squared** (≈7
   there). *(v1.2 correction: v1.1 called the ratio itself the design effect. It is not — the Kish
   design effect is a variance ratio, and `n_eff = n_obs / DEFF`. §3.4 rule 5 carries the formula and
   the ρ=1 identity that tests it.)*
2. **The per-position table is the primitive and is always printed**, with per-position `n` =
   conversations. **Under the 12×1 design those `n`s are small and unequal by position**, and the
   report must print each one rather than a single header figure: positions 1–4 have n=12 (all three
   shapes), 5–7 have n=8 (shapes A and B), 8–9 have n=4 (shape A only). Computed Wilson widths at
   those `n` (z=1.959963984540054): `0/12 → [0.000, 0.242]`, `12/12 → [0.758, 1.000]`,
   `0/8 → [0.000, 0.324]`, `0/4 → [0.000, 0.490]`, `4/4 → [0.510, 1.000]`. **Mark every position with
   n < 10 `descriptive at this n — no significance claim`**; that is positions 5 onward, i.e. most of
   the deep-turn region. A deterministic collapse still shows (12/12 vs 0/12 at a position is
   unmistakable), but a 30-pp difference at turn 7 is not resolvable and the table must not look as
   though it were.

### 4.5 The clustering hazard, and the sampling design that answers it (settled 2026-09-02)

**The argument, unchanged from v1.1.** FR-22's conversation scripts are **fixed**. §8.2's design ran
15 replicates of the *same* condition-A script. That is **trial replication**, not item replication:
the resulting CI describes "if I run this one script again," not "if I wrote a different 9-turn
script." A tool that prints ±5 pp from 40 replicates of 3 scripts is reporting an interval that
could move 40 pp if someone wrote a fourth script.

**The decision (stakeholder, on that argument): 12 distinct conversation scripts — 4 per shape ×
the three A/B/C shapes — run once each at temperature 0.** `replicatesPerScript = 1`.
**n = 12 conversations per model per arm**, one observation per script.

#### 4.5.1 What 12×1 does to the clustering problem

**Conversation-level clustering dissolves — for conversation-level statistics only.** With one run
per script, each cluster contributes exactly one observation, so there is nothing to inflate the
variance: `DEFF = 1.00` **by construction, not by assumption**, `n_eff = 12`, the paired table for
`cleanThroughTurnH` has 12 rows, and **McNemar exact becomes a valid decision rule rather than an
anti-conservative one.** That is what the design bought. It did not buy precision — see §4.5.3.

*(The "12 rows" is 12 only while `H ≤ min(script length)` holds — §4.6. That is a validated pack
invariant, not an assumption of this section: violate it and the paired table silently loses the
conversations too short to reach turn `H`, which would make both `n_eff` and the resolving-power
line wrong in the optimistic direction. It is the only place `H` touches a denominator in this
note.)*

**Three things do not dissolve, and the implementer must handle all three.**

**(i) Turn-level dependence inside a conversation.** Every statistic that pools turns — all seven
FR-8 counts, every per-call count — still has ~7 positively-correlated observations per
conversation, for exactly the reason §4.4 gives. **The cluster bootstrap requirement survives, at
one level: resample the 12 conversations, recompute the pooled rate inside each resample** (§3.4
rule 6). The consequence worth printing: with 12 clusters, **the effective n of any turn-pooled
count is capped at 12** no matter how many turns feed it — 80 turns at ρ=1 is 12, not 80 — so a
turn-pooled count resolves ~50 pp at best and can never be a verdict metric. It isn't one (§4.6);
this is the arithmetic that says it never can be at this pack size.

**(ii) Shape-level correlation, which now has nowhere to go.** The 12 scripts are 4 per shape × 3
shapes. Scripts within a shape share tool set, length and task pattern, so they are *not*
exchangeable draws from "all conversation scripts" — a model that fails shape B's write-mutating
pattern plausibly fails all four B scripts. The tempting fix is a cluster bootstrap over shapes, and
it is wrong: **3 clusters is too few for any bootstrap** (the resample distribution is degenerate and
the resulting interval is noise). The honest handling, and the recommendation:

> Treat **shape as a fixed blocking factor, not a random cluster.** The inference is *conditional on
> these 12 fixed scripts* — which is exactly what FR-22's "fixed, versioned scripts" specifies —
> per-shape tables are always printed, and the report carries one line: `Inference is conditional on
> the 12 scripts in pack <id>@<version>; generalization to unwritten scripts is not certified by any
> interval in this report.` No number this tool prints, at any n, certifies that generalization; the
> only thing that would is more distinct scripts (§4.5.3's reversal trigger).

**(iii) Run-to-run variability becomes unmeasurable.** This is the one real loss. LM Studio at
temperature 0 is near-deterministic but not guaranteed bit-deterministic (batching and GPU
reduction order), and with one run per script the harness can no longer see the difference between
"this model is flaky" and "this script is hard". Cheap mitigation, and it should be built:
**a determinism probe — re-run 2 of the 12 scripts a second time, once per model, and report
whether the outcome vector is identical.** Two conversations of budget; **diagnostic, outside `n`,
never pooled into it.** If the probe comes back non-identical, every conversation-level statistic in
that run carries an unmeasured extra variance source and the report must say so in the same words.

Keep printing `temperature` and `replicatesPerScript` adjacent to every conversation-level `n`
(FR-18), now with `replicatesPerScript = 1` as the value that says the design effect is 1 by
construction rather than by hope.

#### 4.5.2 Cost, measured basis

12 scripts = 4×9 + 4×7 + 4×4 = **80 turns per model**. At the prior run's measured ~1.3 s/turn that
is ≈1.7 min per model, ≈3.5 min for both arms of a paired comparison. (The old 48-conversation
design was 320 turns ≈ 7 min per model, ≈14 min paired — v1.1's estimate, same basis.)

**A correction the stakeholder is owed: this is not the same run budget, it is one quarter of it.**
The authoring budget is unchanged — 12 human-verified scripts either way, which is the expensive
half — but the *inference* budget drops from ~320 to ~80 turns per model. What that freed budget can
and cannot buy is §4.5.3.

#### 4.5.3 The honest consequence, stated plainly

| | old design (12 scripts × 4 reps) | **settled design (12 × 1)** |
|---|---|---|
| nominal `n` | 48 conversations | **12 conversations** |
| honest `n_eff` at temperature 0, ρ≈1 within script | `48 / (1 + 3·1)` = **12** | **12** |
| observable floor (α=0.05) | claimed 12.5 pp / honest **50.0 pp** | **50.0 pp** |
| MDD₈₀ | claimed 16.0 pp / honest **57.8 pp** | **57.8 pp** |
| is McNemar valid? | **no** — anti-conservative over 48 correlated rows | **yes** |
| is the design effect measurable? | yes, from replicates | no — it is 1 by construction |

**So the tool did not lose resolving power. It lost a claim it could not support.** The honest
figures are identical because the old design's effective n *was* 12; all that changed is that the
report now says 12 where it used to say 48. (One v1.1 figure is corrected here: §7.2 put the fully
clustered MDD at "~65 pp", which came from the `8/n` rule of thumb. The exact value at n_eff=12,
recomputed this session, is **57.8 pp** — §3.4 rule 3. The direction of gate B-1 is unaffected; the
gap it names was 16.7 vs 57.8 pp, not 16.7 vs 65.)

**What this design genuinely cannot do, in the lab's own terms.** The effects this lab has cared
about split across the floor:
- the `qwen3-4b` turn-4 collapse (97.5% vs 0%) → at 12×1 that is b=12, c=0, McNemar exact
  p = 0.00049. **Comfortably detected.** The pack still does the job it was commissioned for.
- the ministral duplicate-instruction defect (~30 pp) → **below the 50.0 pp floor. Not resolvable,
  at any observed outcome.** It would have been *claimed* resolvable under the old nominal 48
  (MDD₈₀ 16.0 pp) and would not actually have been. The loss is of a false claim, but a reader who
  remembers the old sizing should be told the 15–50 pp band is dark.

***Reversal trigger, costed.*** The freed inference budget (36 conversations, ≈11 min paired)
converts into resolving power **only** by authoring **36 more distinct human-verified scripts** —
48 total, 16 per shape — which puts the floor at 12.5 pp and MDD₈₀ at 16.0 pp *honestly*. The
binding constraint is FR-19 human verification of scripts, not compute; the compute is already
paid for. **Trigger: the first tool-caller comparison that returns "not distinguishable" with an
observed difference in the 15–50 pp band.** That is precisely the band 48 distinct scripts would
resolve and 12 cannot, and it is the evidence that makes the authoring cost worth funding rather
than a hypothetical.

### 4.6 Aggregating across conditions of different length without a confounded headline

Do **not** pool turns across conditions — a 9-turn script contributes 2.25× the turns of a 4-turn
one and the headline becomes a weighted average of script lengths.

**Recommended headline: a survival statistic at a pack-declared turn depth.**

- Headline (`headlineMetric`): **`cleanThroughTurnH`** — the fraction of conversations with zero
  failure of any kind through turn `H`. **`H` is declared in the pack manifest
  (`metrics.cleanThroughTurnH.H`) and validated `H ≤ min(script length)` across all conditions in
  the pack**; it is *bounded by* the minimum, not *equal to* it. `H = 4` for the A/B/C set, whose
  minimum is also 4. *(v1.4: v1.1–v1.3 defined `H = min(script length)`. The plan's semantics are
  authoritative and are adopted here; the derived definition was the gate's M-11 — it let a future
  pack version that adds a 3-turn script silently redefine the headline from `cleanThroughTurn4` to
  `cleanThroughTurn3` under an unchanged name.)*

  **The `≤` is not a formality — it is exactly the precondition that makes the denominator
  honest.** Because every conversation in the pack is at least `H` turns long, every conversation
  of every condition contributes exactly one observation, so the statistic is length-independent by
  construction, is a proper binomial over the full 12 conversations, and is the statistic that
  would have caught `qwen3-4b` on the first run. **If `H > min(script length)` the denominator
  silently becomes selection-conditioned** — short scripts cannot reach turn `H`, so the headline
  quietly turns into a rate over long conversations only, which is §4.3's laundering failure in the
  one place the report calls its headline. That is the methodological reason `validate` must fail
  the pack rather than clamp `H`, and it must fail rather than warn.

  **What declaring `H` strictly below the minimum costs, since it is now allowed.** It is a
  legitimate choice — it keeps the headline's meaning stable across pack versions, which is the
  point of declaring it — but turns `H+1 … min` are then scored and excluded from the headline, so
  a failure at turn 5 does not count against `cleanThroughTurn4`. The metric gets coarser, not
  wrong. Two consequences, both cheap: **print `H` beside the metric name in every report** so
  `cleanThroughTurn4` is never read as `cleanThroughTurn7`, and treat any gap between `H` and
  `min(script length)` as discriminating information deliberately left on the table — worth a line
  in the report when the gap is non-zero, and worth nothing when it is zero (as it is today).
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

   **Settled 2026-09-02 (stakeholder decision 3), for an implementer reading only this note:
   the self-check is a diagnostic and never blocks.** If it lands below the ~0.85 reference,
   **S3 still completes** — the run is stored, the comparison renders, the exit code is unchanged —
   and the deviation plus its investigation are written into the test report. There is no
   configuration in which this number fails a build, fails a stage, or suppresses a result. The
   reason is §5.4's own argument: exact-vs-ANN and vector-only-vs-hybrid push in opposite
   directions, so a disagreement of either sign is uninterpretable as *quality*, and a number you
   cannot interpret must not hold a gate. Its job is to make a harness defect visible, and it does
   that by being printed and read, not by halting anything.
2. **Implementation cross-check on the pure metric functions.** Feed `test_metrics.py`'s existing
   fixtures through the new implementation and require byte-identical outputs — assertable here,
   unlike §3.2c's bounds, because recall@k and MRR are ratios of small integers and the values are
   exactly representable. Cheap, and it
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
resolves 20 pp observed / 25 pp at 80% power. At n=20 that becomes 30.0/36.7 pp — too coarse to be
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

- **Observable floor = `b_min(α_family)/n_eff` = `6/n_eff`, computed at the *unadjusted* α=0.05 —
  never at `α/k`** (Rule 3's α row; review M-ML-6). Under Holm a member's threshold is
  data-dependent, so no single α makes the floor's sentence true by construction and one must be
  chosen: *"below this, nothing can reach significance"* is true only at the **loosest step a member
  can face**, which is the unadjusted α. Printed at `α/k` it is `7/n_eff` and is **false** — at
  n=40, `b=6, c=0` gives p=0.031, which a rank-2 member *does* clear at its 0.05 Holm step, and
  15.0 pp sits below the printed 17.5. Same falsity class as the `15.8` withdrawn in v1.5.
  Floor-at-0.05 is additionally conservative for every member, and it is what makes Rule 7's
  McNemar branch a theorem.
- **MDD₈₀ — computed exactly, never from a rule of thumb** (§3.4 rule 3). The `n·δ ≈ 7.7` mnemonic
  holds only over n≈20–120 (recomputed: 7.33 at n=20 → 7.81 at n=120) and **breaks below it —
  6.94 at n=12** — which is exactly the range the tool-caller pack now lives in.

**All figures recomputed in this session** by exact search over the McNemar rejection region.
**MDD columns are ceilinged to 0.1 pp; the floor column is truncated** — opposite directions, for
the reason in §3.4 Rule 3. The exact MDD values are shown so nobody re-derives them; the floor is
exactly `6/n_eff`, truncated with the caller's own expression (Rule 3a's bin-edge guard):

| n_eff | **observable floor** (α=0.05, *every* k) | MDD₈₀ at α=0.05 (k=1) | (exact) | MDD₈₀ at α=0.025 (k=2) |
|---|---|---|---|---|
| **12** | **50.0 pp** | **57.8 pp** | 57.794 | 65.6 pp |
| 15 | 40.0 pp | 47.6 pp | 47.559 | 54.2 pp |
| 20 | 30.0 pp | 36.7 pp | 36.646 | — |
| 30 | 20.0 pp | 25.1 pp | 25.075 | 28.7 pp |
| 38 | 15.7 pp | 20.1 pp | 20.009 | — |
| 40 | 15.0 pp | 19.1 pp | 19.046 | 21.9 pp |
| 48 | 12.5 pp | 16.0 pp | 15.972 | — |
| 60 | 10.0 pp | 12.9 pp | 12.857 | — |
| 85 | 7.0 pp | 9.2 pp | 9.142 | — |
| 120 | 5.0 pp | 6.6 pp | 6.509 | — |

**There is one floor column and there always will be**: the floor does not move with *k*, because
its α is the unadjusted one whatever the family size. Only the MDD pays for multiplicity. *(v1.6
deletes v1.2–v1.5's second floor column — 58.3 / 46.6 / 23.3 / 17.5 pp at α=0.025 — which asserted
impossibilities that are attainable.)*

**Every report prints, computed from its own `n_effective` and its own α, never hardcoded:**

> `This pack resolves differences of >=X pp with 80% power at n=N effective <unit>s (<U> units,
> design effect D, <basis>, alpha=A_mdd). Differences below Y pp cannot reach significance at any
> observed outcome, at any Holm step (alpha <= A_family).`

Every field is mandatory, **including both αs** — they differ whenever k > 1, and a reader shown
only one cannot tell which bound it governs. The unit, the raw unit count, the design effect and its
basis are what make the line auditable — a bare `n` is the shape gate B-1 rejected, because 48 turns, 48
conversations and 12 scripts print the same sentence and mean three different things.

**The floor sentence takes an effective-unit qualifier whenever `design_effect > 1.0` (v1.8, review
m-ML-11).** The floor is `b_min/n_eff` while the McNemar p printed beside it is computed over the
`n_units` raw rows, so above DEFF 1 the two quantities live on different denominators and the
sentence as templated above reads as a flat contradiction of the number three lines from it.
Measured: at `n_units = 12, DEFF = 2` the floor prints **100.0 pp** while `b=11, c=0` is a 91.7 pp
difference at `p = 0.00098`; at `n_units = 40, DEFF = 2` the floor prints **30.0 pp** beside
`b=6, c=0` at `p = 0.031`. Both are decided correctly — Rule 7 demotes them — and both *print* a
p below α underneath a sentence saying no observed outcome can reach significance. Where
`design_effect > 1.0`, render the floor sentence as:

> `differences below 30.0 pp cannot reach significance at any observed outcome over the 20 effective items this design supports, at any Holm step (alpha <= 0.05) — the McNemar p printed beside it is computed over the 40 raw rows, where clustering makes it anti-conservative, so it can fall below that alpha without lifting the difference above this floor`

At `design_effect == 1.00` the two denominators coincide, the qualifier is noise, and the template
above stands unchanged — which is why §7.2's rendered line does not move.

**The two sentences are independent, and the second prints whatever happens to the first (v1.7).**
When no effect size attains the power the MDD sentence is replaced by the *unattainable* wording
(review M-ML-1), and it is tempting to let that one replacement speak for both bounds — *"no
difference is resolvable, so nothing can reach significance."* **That combined sentence is false in
a reachable corner**, and it is reachable for the same reason the two bounds take different αs: with
`k >= 2` and `n_eff ∈ [6, 7)`, the rejection region at `alpha_mdd = 0.025` is empty (`b_min = 7`
exceeds the 6 floored units), so `mdd80` is `None` — while at `alpha_family = 0.05` the floor is
`6/n_eff ≤ 100 pp` and is *attained*. Worked, at `n_units = 13, DEFF = 2, n_eff = 6.5`: the floor
prints 92.3 pp, and the attainable outcome `b=12, c=0` is a 92.3 pp difference at
`p = 2·2⁻¹² = 0.00049`, which clears a rank-2 member's 0.05 Holm step. So the unattainable clause
speaks **only for power**, the floor sentence prints **either way**, and where the floor itself
exceeds 100 pp it says so in words rather than quoting an unreachable threshold.

**One assumption behind every MDD above, and it is optimistic.** The power model is **strict
dominance** (`π_c = 0`): every item on which the models differ favours the candidate. That is the
most favourable case, so these are *lower bounds* on the difference actually needed. Measured
sensitivity, computed this session at α=0.05:

| discordance mix | n=12 | n=30 | n=40 | n=48 |
|---|---|---|---|---|
| strictly dominant (`π_c = 0`) | 57.8 pp | 25.1 pp | 19.1 pp | 16.0 pp |
| candidate wins 4:1 (`π_c = 0.25·π_b`) | unattainable (max power 0.56) | 45.2 pp | 34.0 pp | 28.4 pp |
| candidate wins 2:1 (`π_c = 0.5·π_b`) | unattainable (max power 0.18) | unattainable (0.43) | unattainable (0.53) | unattainable (0.58) |

"Unattainable" means 80% power is not reached **at any effect size** under that mix. So a candidate
that is genuinely better but *also* loses some items — the normal case for a model swap — needs
substantially more than the headline MDD, and at n=12 essentially needs to win every discordant
conversation. **The report prints the strict-dominance MDD** (one number, comparable across packs)
**labelled `best case — assumes the candidate wins every item the models differ on`**, and any pack
whose `n_eff < 20` additionally prints the power ceiling from the 2:1 row, because that is where the
label stops being a caveat and starts being the finding.

Reassurance, still true and now precise: the effects this lab has needed to resolve — the qwen3-4b
turn-4 collapse (97.5% vs 0%) — are 90–100 pp and clear every floor in the table including n=12's.
The ~30 pp ministral duplicate-instruction defect does **not** clear the tool-caller pack's 50.0 pp
floor (§4.5.3). The tool is fit for catching collapses and honest about not resolving anything in
the 15–50 pp band at this sample size.

### 7.2 Per role

| Role | n | Unit of n | Instrument | Observable floor | MDD₈₀ | Existing golden data adequate? |
|---|---|---|---|---|---|---|
| **tool-caller** | **12** (3 shapes × 4 distinct scripts × **1 run**) | **conversation ≡ script**, one observation per cluster, **DEFF 1.00 by construction** | McNemar exact on `cleanThroughTurn4` — **valid at this design**, plus one-level cluster bootstrap over the 12 conversations for any turn-pooled count | **50.0 pp** | **57.8 pp** | **No — must be built (FR-22/FR-22a).** 12 distinct human-verified scripts; §4.5 is the sizing and §4.5.3 the honest consequence. |
| **guard-judge** | 85 total, but the decision is **class-conditional** and the family has **two verdict metrics** (MDD at α=0.025; floor at the unadjusted α=0.05) | item | McNemar per class, Holm across the two | see below | see below | **Partly** — see §7.3 |
| **nlq-generator** | 40 | item | McNemar on Layer-1 exact match | 15.0 pp | 19.1 pp | **Yes, marginally.** Answers "clearly better" only. |
| **chat-responder** | 30 (new) | item | McNemar on checklist pass | 20.0 pp | 25.1 pp | **No — does not exist** (§6.2) |
| **embedder** | 38 queries | query | paired bootstrap on per-query MRR | n/a (continuous) | see §7.4 | **Yes for MRR; no for recall@k** — see §7.4 |

*(`chat-responder`'s row describes the design to build when FR-21a's judged layer is funded; it does
not ship in first delivery.)*

**The tool-caller pack's resolving-power line, verbatim — this is what §7.1's template renders to
under the settled design, and the string an implementer should test against:**

> `This pack resolves differences of >=57.8 pp with 80% power at n=12 effective conversations
> (12 units, design effect 1.00, by-construction, alpha=0.05). Differences below 50.0 pp cannot
> reach significance at any observed outcome, at any Holm step (alpha <= 0.05). Best case —
> assumes the candidate wins every conversation the models differ on; if it loses one for every
> two it wins, 80% power is not reached at any effect size at this n. Inference is conditional on
> the 12 scripts in <packId>@<packVersion>; generalization to unwritten scripts is not certified by
> any interval in this report.`

*(v1.7: the floor sentence gains `at any Holm step (alpha <= 0.05)`. §7.1's template mandated it
from v1.6 and this rendering was left on the pre-v1.6 wording — a stale example presenting itself as
"the string an implementer should test against" is worse than no example, so **§7.1's template is
authoritative wherever the two ever diverge again.** This pack is k=1, so both αs are 0.05 and only
the wording moves.)*

Four sentences, all mandatory, none derivable from a bare `n` — which is the whole of gate B-1's
fix. Compare what the plan's `min_detectable_difference(48)` would have printed: **16.7 pp**, a
number 3.5× more optimistic than the design supports, with no unit, no design effect and no
conditionality.

### 7.3 Guard-judge — n=85 is misleading, and this pack has two verdict metrics and no headline

The decision-relevant statistic is class-conditional (a bias-to-suspend judge is mis-gated by
pooled accuracy for the same reason §6.2's judge is), and the class slices are small.

**Stakeholder decision 2, encoded:** `guard-judge` has **no `headlineMetric`**. Both
class-conditional error rates are co-equal members of `verdictMetrics`, reported with equal weight,
in a fixed declared order, with no summary number above them and no arithmetic combining them.
**Nothing in this note depends on that pack having a single verdict-carrying metric** — v1.1's §7.3
called false-advance "the primary" and §3.3's table named it; both are corrected, and no formula,
denominator or threshold anywhere in the note ever took the retired singular `primaryMetric` as
input.

**As built (plan v1.3), and nothing here conflicts with it:**
`verdictMetrics = ["falseAdvanceRate", "falseSuspendRate"]`, `headlineMetric = null`, bare metric
names with **no `@slice` suffix** — the slice is the metric's denominator, stated once in the table
below, not part of its identity — and `advanceRecall` **printed as a labelled complement that
carries no verdict**.

**Naming, settled (gate nit).** Report **both verdict metrics as error rates in the same
direction**, so neither reads as "better is higher" beside one that reads "better is lower":

- **`falseAdvanceRate`** = P(judge advances | gold says suspend), on the 40 `clear_suspend` items.
- **`falseSuspendRate`** = P(judge suspends | gold says advance), on the 30 `clear_advance` items.

`advance-recall` (v1.1's name for the second) is the **complement**: `advanceRecall = 1 −
falseSuspendRate`. Same quantity, and it stays printed as the complement so a reader looking for
recall finds it — but it **carries no verdict**, because a metric and its own complement are one
test, not two, and counting both would inflate *k* against a difference that is by definition
identical. The verdict is rendered on the error rate, because two co-equal verdict metrics pointing
the same way is the only presentation in which "worse on one, better on the other" is readable at a
glance.

**The MDD is recomputed at α=0.025** — the tightest Holm step a two-member family can be required to
clear (§3.3) — with the k=1 figure alongside so the price of the second verdict metric is visible.
**The floor stays at the unadjusted α=0.05** and is therefore the same number it would be for a
one-metric pack:

| slice | n | verdict metric | observable floor (α=0.05) | MDD₈₀ @α=0.025 | (MDD₈₀ @α=0.05, for contrast) |
|---|---|---|---|---|---|
| `clear_suspend` | 40 | **`falseAdvanceRate`** | **15.0 pp** | **21.9 pp** | 19.1 pp |
| `clear_advance` | 30 | **`falseSuspendRate`** | **20.0 pp** | **28.7 pp** | 25.1 pp |
| `boundary` (all expected-suspend) | 15 | false-advance on near-misses — **not a verdict metric** | 40.0 pp | — | 47.6 pp |

*(v1.6: the floor column was `17.5 / 23.3 / —` at α=0.025 in v1.2–v1.5 and is now `15.0 / 20.0 /
40.0` at the unadjusted α — review M-ML-6. All three are exact `6/n`, so neither the truncation rule
nor Rule 3a's bin-edge guard has anything to do here; `17.5` at α=0.025 *was* the cell where the
guard mattered, and moving the floor to α=0.05 retired that hazard rather than the guard. MDDs are
ceilinged, the floor truncated — §3.4 Rule 3.)*

So: **the naive read "n=85 → ~9 pp" is wrong**, and the honest figures are now 21.9 / 28.7 pp
rather than v1.1's 19.1 / 25.1 — the ~2.8–3.6 pp of resolving power that a second co-equal verdict
metric costs. (v1.1's boundary-tier MDD of "53 pp" was the `8/n` rule of thumb; the exact value at n=15 is
**47.6 pp**, recomputed here.) The `boundary` tier is **descriptive only** at n=15 and must be
printed with `no significance claim` — it is also the tier where disagreement is legitimate by
construction, so pooling it into an overall accuracy would let a model look better or worse for the
wrong reason. Report the three tiers separately, always; never a pooled 85-item accuracy as a
headline. Pooled accuracy and κ stay as diagnostics with marginals.

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

*(Status at v1.2: **accepted and superseded by something stronger.** FR-22a carries the distinct-
scripts requirement, and the stakeholder's 12×1 sampling decision removes replication entirely, so
conversation-level statistics need no bootstrap at all — the design effect is 1 by construction
(§4.5.1). The bootstrap requirement survives only for **turn-pooled** counts, one level, over the
12 conversations. FR-15 does not need amending for it; §3.4 rule 4 enforces it in the code, which is
the more durable place.)*

**Two smaller items, not blockers, worth folding into the plan rather than amending:**

- **FR-12's precision@k** is algebraically redundant with recall@k on the copied golden set
  (§5.1). Keep it for compliance, footnote it.
- **FR-8 has no "restraint" count** — no measure of *not* calling a tool when none was required
  (§4.2). Add it; it is one line and without it a trigger-happy model scores perfectly.

---

## 9. Evaluation design — how to prove the harness itself is right

The harness is an instrument; an uncalibrated instrument produces confident wrong numbers. Eight
checks, all cheap, all implementable as unit/integration tests. Checks 1–4 are the statistics
module's own contract (§3.4) and are the ones that keep gate B-1 from recurring:

1. **Statistics module regression fixtures.** The five worked (a,b,c,d) rows in §3.2c, with their
   expected McNemar p and MOVER-D bounds, plus the McNemar floor table at both alphas
   (α=0.05: c=0→b=6, c=1→b=8, c=2→b=10, c=3→b=12, c=4→b=13; **α=0.025: c=0→b=7, c=1→b=9,
   c=2→b=11**). Pure functions, no network. *Threshold:* **1e-12 on p, 1e-9 on MOVER-D bounds as
   proportions** (§3.2c) — not "exactly", which is not assertable, and not 3 decimal places, which
   is loose enough to pass a wrong Wilson. **Assert against §3.2c's 10-dp table, which is the only
   published form of these fixtures precise enough to carry that tolerance** (§3.2c's trap); assert
   the p-values against the exact rationals, not their decimal expansions.
2. **The four contract tests that keep gate B-1 fixed** (§3.4), each of which must fail loudly
   rather than degrade: (a) `PairedOutcomes.from_units` raises `DuplicateAnalysisUnit` on a repeated
   unit id — feed it 48 rows drawn from 12 script ids, which is the exact shape the gate rejected;
   (b) `resolving_power` cannot be called without `design_effect` and `basis` (a `TypeError`, i.e.
   the absence of a default is itself the test); (c) `verdict` raises when
   `resolving.n_units != outcomes.n_units`, when the unit kinds differ, or when
   `alpha_mdd != alpha_family/len(family)`, or when `alpha_step` falls outside
   `[alpha_mdd, alpha_family]` (Rule 4's fifth precondition); (d) `verdict` refuses to let McNemar
   decide when `design_effect > 1.0` and renders the bootstrap decision instead.
3. **The MDD computation itself**, three assertions: it reproduces §7.1's exact column
   (19.046 pp at n_eff=40, 57.794 at 12, 47.559 at 15); it **ceilings** to the printed precision, so
   n=40 prints 19.1 and never 19.0 (measured power at 19.0 pp is 0.798 — the test is that the
   printed number's power is ≥ 0.80); and it is not a constant, not `8/n`, and rejects an `int`
   observation count passed where `n_effective` belongs.
4. **The design-effect identity** (§3.4 rule 5): for a synthetic cluster set with ρ=1 and m
   observations per cluster, `effective_n` must equal the **cluster count**, and `design_effect`
   must equal the squared width ratio. This is the one test that catches the squaring error in
   either direction; v1.1's own wording would have failed it.
5. **Metric cross-check against falkor-chat.** Run `test_metrics.py`'s existing fixtures through
   the copied recall@k/MRR. *Threshold:* byte-identical outputs.
6. **Embedding-path self-test.** Rank the 38 golden queries using the copied
   `golden_retrieval.embeddings.json` vectors and the copied corpus. *Reference line, not a
   threshold:* recall@10 is compared against ~0.85 and **printed either way** — below it, the run
   still completes and the deviation is investigated and written into the test report (§5.4,
   stakeholder decision 3). It is a bug detector; it fails no stage.
7. **Prose-call detector calibration.** ~20 human-labelled replies; report the detector's own
   precision/recall in every tool-caller run. *Threshold:* the numbers are printed; no pass/fail
   gate, because the requirements rule out hard gates and §4.2's use is diagnostic.
8. **Judge calibration gate** (chat-responder only, §6.2): false-pass rate ≤ 2/20 on gold-unfaithful
   items **and** parse-failure rate ≤ 0.05, else judge-mediated numbers are suppressed.

**A negative control worth the twenty minutes:** run the paired comparison with **the same model in
both arms — two independent runs, not two copies of one run** (two copies give b = c = 0 by
construction and cannot fail). The correct output is "not distinguishable" with b ≈ c and a
difference CI centred on zero. Anything else means the pairing, the session handling, or the RNG
seeding is wrong. It catches an entire class of harness bugs that would otherwise present as
plausible model differences.

**What the 12×1 design does to that control, and it is worth knowing before it is misread.** At
temperature 0 with one run per script, two independent runs of the same model are *expected* to
produce **b = c = 0** — not because the statistics work, but because the model is deterministic. So
on the tool-caller pack the negative control degenerates into the determinism probe of §4.5.1(iii),
and it exercises the pairing and session handling but **not** the decision arithmetic. Two
consequences: (1) the control still earns its place — `b ≠ c` here is a genuine finding, either a
harness bug or non-determinism at temperature 0, and both need to be known; (2) the decision
arithmetic must be exercised by the §9.1 fixtures and the synthetic traces instead, never by
"the negative control passed".

---

## 10. Risks and open questions

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Script-level clustering treated as independent replication → confidently narrow, wrong CIs | **high** *(largely retired by the 12×1 design; the residual is R1a)* | §4.5: one run per distinct script, so DEFF = 1 by construction; §3.4 rules 1–4 make the correlated-rows version unconstructible in code |
| R1a | The *residual* of R1: a turn-pooled count, or a future pack that buys replicates, silently re-inherits the old defect | **high** | §3.4 rule 6: one-level cluster bootstrap over conversations for turn-pooled counts, and `validate` **fails** any pack declaring `replicatesPerScript > 1` while only the one-level resample exists |
| R1b | Shape-level correlation (4 scripts per shape, 3 shapes) read as generalizing to unwritten scripts — 3 clusters is too few to bootstrap | medium | §4.5.1(ii): shape is a **fixed blocking factor**, per-shape tables always printed, and one mandatory report line saying the inference is conditional on the 12 fixed scripts |
| R2 | Conditional-denominator laundering — a model that collapses early scores *better* on (c)–(g) | **high** | §4.3: mandatory funnel table, `k/n` on every rate, printed `n/a` tallies, paired-n intersection |
| R3 | Judge-mediated chat-responder number read as equal in strength to a ground-truth one | **high** | §6.2: separate table, fixed banner, split self-preference caveats, hard error on judge==candidate |
| R4 | Corpus embedding cache shared across models or prefix settings → invalid comparison, no visible trace | medium | §5.5: cache key = (model, quantization, docPrefix, corpus version) |
| R5 | Multiple comparisons across 7 counts × turn positions → a false "better" is near-certain | medium | §3.3: pre-registered `verdictMetrics` (1..k) per pack, Holm–Bonferroni when k > 1, everything else labelled exploratory |
| R6 | Temperature-0 replicates counted as independent n | **retired** — the design no longer buys replicates | §4.5: `replicatesPerScript = 1`; still printed next to every conversation-level n (FR-18), now as the field that says DEFF = 1 by construction |
| R6a | The *inverse* of R6: with replicates gone, run-to-run variability is unmeasurable, so model flakiness at temperature 0 is invisible | medium | §4.5.1(iii): a 2-script determinism probe per model, diagnostic and outside `n`; a non-identical result is stated in the report as an unmeasured variance source |
| R9 | The 50.0 pp tool-caller floor read as "no difference exists" when it means "this pack cannot see one" | medium | §7.1's mandatory resolving-power line with unit, unit count, DEFF, basis and α; §4.5.3's named 15–50 pp dark band and its costed reversal trigger |
| R7 | Copied golden data loses its provenance chain at the copy boundary | low | §5.4: `copiedFrom` + `copiedAt` + source git SHA on every copied artifact |
| R8 | A copied `judge.py` drifts from falkor-chat's and the two get conflated | low | Record `judgePromptVersion` in the fingerprint; state in the pack docs that the two are independent by design |

**Open questions — all three of v1.1's are now closed. Closed in place, with what closed them:**

1. ~~Does the stakeholder fund the chat-responder golden data (30 items + 30 calibration items)?~~
   **Closed by FR-21a (2026-09-02): the judged layer is deferred**, and `chat-responder` ships its
   deterministic layer only. §6.2's 30-item checklist set and 40-item balanced calibration set
   describe the design to build *when* it is funded; §6.3's costed conditional stands as the
   trigger, not as an open ask.
2. ~~`primaryMetric` per pack (the retired singular) — the stakeholder may prefer a different
   guard-judge headline.~~
   **Closed by stakeholder decision 2 (2026-09-02): `guard-judge` gets no headline**; both
   class-conditional error rates are co-equal verdict metrics with equal weight. Encoded in §3.3
   (`verdictMetrics` + `headlineMetric: null`) and §7.3 (recomputed at α=0.025). The stakeholder
   declined to rank the two errors; §3.3's reversal trigger says what to do the day that changes.
3. ~~Extending the retrieval golden set to ~60 queries.~~ **Closed as a backlog follow-up**, not a
   first-delivery ask (§7.4). Trigger unchanged: the embedder role becoming decision-critical, or a
   run where recall's saturation blocks a real comparison.

**And v1.1's largest open methodological question, also closed:** ~~how the tool-caller pack should
spend its conversation budget between distinct scripts and replicates~~ — **closed by stakeholder
decision 1 (2026-09-02): 12 distinct scripts × 1 run at temperature 0** (§4.5).

**One new question this version raises, for the stakeholder rather than the architect** — it changes
data cost, not method, and nothing is blocked on it:

- **Is the freed inference budget meant to buy more distinct scripts?** The decision was recorded as
  "same total run budget", but 12 scripts × 1 run is **one quarter** of the previous inference cost
  (≈80 turns per model against ≈320; §4.5.2). If ~48 conversations of inference were genuinely
  budgeted, the statistically correct spend is **48 distinct scripts × 1 run** (16 per shape), which
  moves the floor from 50.0 pp to 12.5 pp and MDD₈₀ from 57.8 pp to 16.0 pp *honestly*. The binding
  constraint is **FR-19 human verification of 36 additional scripts**, not compute. If that authoring
  is not fundable, 12×1 is the right design and §4.5.3's trigger is the right way to revisit it —
  this question needs no answer before S1, and none before S6 either, but it should be asked out
  loud rather than settled by the phrase "same budget".


---

## 11. Q6 — The latency percentile: estimator, denominator, and the two floors (plan R-13)

### 11.1 The question and the decision it serves

Plan §6 R-13 asks for `_percentile`'s definition. Two further inputs arrived before it could be
answered, and all three are settled here because each one moves the others:

1. **The definition** — which estimator, precisely enough that two implementers cannot disagree at
   n = 12, where this component actually operates.
2. **The denominator** *(plan v1.8)*. Under LM Studio's JIT auto-load a **cold first call was
   measured at 21.068 s** (plan §2.5) against sub-second warm calls, so plan §3.6 runs a residency
   probe between items and sets **`ItemResult.latencyMs = None`** on any item a model load touched.
   The latency sample is therefore a subset of the items, **missing exactly the slow ones, because
   they were slow.**
3. **A minimum surviving-sample floor** *(review Pass 3, G3-11)*: a bad run can leave four
   latencies, and a nearest-rank p95 over four points is the maximum — "a number the report will
   print, honestly denominated, that means nothing."

What the caller does differently: S2 stores the first latency figure the day this is answered, and
FR-11's p50/p95 are compared across runs stored months apart, so the definition freezes on first
write.

**The altitude this section takes, stated once.** Latency is **descriptive**: it appears in no
pack's `verdictMetrics` (§3.3's table), it decides nothing, and no verdict rests on it. So the bar
here is deliberately *not* §3.2's bar. The paired instrument refuses because a wrong verdict is a
false claim about two models; the latency block refuses because a number labelled `p95` that is
really a `p85` is a **false label on a true measurement**, and it travels into `index.csv` as a bare
column where no qualifier can follow it. Those are different harms. The second is answered by a
floor looser than §3.2's, by conditional strings that carry the qualifier, and — for the third input
above — by **renaming rather than refusing**, which is a move the comparison instrument never has
available to it.

### 11.2 The estimator — the empirical quantile function (Hyndman–Fan type 1), one implementation

**Recommendation: nearest-rank with ceiling, no interpolation.** For a sample of `X ≥ 1` values
sorted ascending `x₍₁₎ ≤ … ≤ x₍X₎` and a percentile level `p`, the reported value is

> `P_p = x₍r₎`,  `r = ceil(p·X/100)`, clamped to `[1, X]`.

This is the **inverse of the empirical CDF** — `inf{ v : F̂(v) ≥ p/100 }` — and it is a named,
externally documented definition rather than a house convention: **Hyndman–Fan type 1**, R's
`quantile(type = 1)`, NumPy's `method = "inverted_cdf"`. Naming it is half the point: an
implementer can check the implementation against a published definition instead of against this
paragraph.

Four reasons it is the right one *here*, in the order they decided it:

1. **It returns a measurement that was actually taken.** Latency is right-tailed and the report's
   job is "how slow was a slow call". An interpolated p95 at n = 38 is a value strictly between the
   36th and 37th observed calls — a number no call took. For a descriptive statistic that is a
   fabrication with a decimal point on it, and this note has spent four review passes removing
   sentences that were not true of the numbers beside them.
2. **It is the same functional §3.4 Rule 4 already computes in closed form.** Rule 4 (v1.8) replaces
   the resampled bootstrap quantile with the exact quantile of the multinomial resample
   distribution, i.e. `inf{ v : F(v) ≥ p }`. Type 1 applied to an empirical sample is that same
   operator applied to the empirical distribution. So adopting it makes v1.8's closed form a
   **substitution of computation, not of definition** — the two agree by construction rather than by
   coincidence, and no third estimator enters the document.
3. **It is total and exactly reproducible at every X including X = 1**, in integer arithmetic, with
   no platform-dependent float rounding (§11.2.1).
4. **Its bias direction is the safe one for this component.** Type 1's p95 is the smallest observed
   value with at least 95% of the sample at or below it, so it is never *below* an interpolated
   estimate. Against a missingness mechanism that already biases the tail **low** (§11.5), an
   estimator that does not additionally shave it down is the one whose printed sentence survives.

**Rejected: linear interpolation** (`statistics.quantiles`, NumPy's default, Hyndman–Fan type 7).
It is the better estimator of a *population* quantile from a large sample, and that is not what is
being reported: FR-11 asks what the calls cost, over 12–85 of them. It also fails reason 2 — a
second quantile definition in a document whose Rule 4 already fixed one — and it makes the printed
number depend on floating-point interpolation between two observations, reopening the
reproducibility question R-13 exists to close.

**Rejected: the shipped `int(round(p/100·(X−1)))`** (both copies, `stats.py:296` and
`results.py:541`). Beyond being a third definition, `round` is half-to-even, so its tie-break
direction **alternates with X**: measured this session, at X = 4 the p50 index is 2 (the *upper* of
the two middle values) and at X = 6 it is 2 again (the *lower* of the middle pair). An estimator
whose tie-break flips with the sample size is not a definition anyone can reason about.

**One implementation, and the two call sites may not differ.** `stats.percentile` is the only copy;
`results.py` imports it and keeps no private helper. The plan's own §3.9 says why — *two copies of a
formula is one copy and one bug* — and the S1 implementation review is the evidence: a duplicated
helper is exactly what let `index.csv` compute `latencyMsP95` at the 50th percentile and stay green
(review M27). The function **sorts a copy of its input internally**; requiring a pre-sorted argument
is a precondition a caller can silently violate, and `results.py`'s copy sorted while `stats.py`'s
did not.

```python
def percentile(values: Iterable[float], *, permille: int) -> float:
    """Hyndman-Fan type 1 (inverse empirical CDF). `permille` is the level x10: p95 is 950.

    Raises ValueError on an empty input: whether a figure exists at all is decided by
    `latency_summary` (§11.6), never by returning None from here.
    """
```

#### 11.2.1 The rank is computed in integers, and this is the same hazard as Rule 3a

`math.ceil(permille * X / 1000)` is a float expression and it is **wrong on real inputs**: measured
this session, `0.28 * 25` is `7.000000000000001`, so the float form returns rank **8** where the
exact rank is **7**. Same bin-edge class as Rule 3a's `(7/40)/0.001` case, one operation over. The
binding form is integer ceiling division:

```python
r = max(1, min(X, -(-permille * X // 1000)))     # ceil(permille*X/1000), exactly
```

**Status of the guard, stated the way Rule 3a states its own.** Swept this session over the four
levels this tool actually uses (`permille ∈ {25, 500, 950, 975}`) and `X ≤ 2000`: **zero divergence**
between the float and integer forms. So on today's call sites the integer form is **defensive, not
load-bearing** — exactly as Rule 3a's guard is under the v1.6 floor. It is mandated anyway for
Rule 3a's reason: sweeping `permille = 1…999` over `X ≤ 3000` finds **1626** divergences, the first
at `(X = 25, p = 28.0)`, so the hazard is one new call site away and the cost of the guard is one
expression. **Pin the code's own expression, never an equivalent-looking one.**

### 11.3 Floor 1 — the identity floor at X = 20, which renames rather than refuses

At this component's sample sizes a "p95" is a near-maximum order statistic. Ranks computed this
session with the integer expression above:

| X (timed items) | rank of the tail figure | timed calls slower than it | rank of the p50 figure |
|---|---|---|---|
| 12 | 12 | **0** | 6 |
| 19 | 19 | **0** | 10 |
| 20 | 19 | 1 | 10 |
| 38 | 37 | 1 | 19 |
| 40 | 38 | 2 | 20 |
| 85 | 81 | 4 | 43 |
| 100 | 95 | 5 | 50 |

**`r = X` — the tail figure *is* the sample maximum — for every `X ≤ 19`** (swept `X ≤ 200`), and the
tool-caller pack's 12 conversations sit inside that range. Review G3-11 is right that printing that
number under the name `p95` is meaningless; it is wrong only in the remedy, and the difference
matters. The number itself is fine — the largest of 12 timed calls is a real measurement and a
legitimate, low-biased estimator of the population 95th percentile (`E[F̂(max)] = 12/13 = 0.923`).
What is false at `X ≤ 19` is the **label**: `p95` promises "one call in twenty was slower", and
**no** timed call was slower. Refusing the number would discard a measurement the operator wants and
can read out of the record anyway; refusing the *label* costs nothing.

> **The ruling: when `r == X`, the report prints `max`, not `p95`, and `latencyMsP95` is `None`.**
> The figure is the maximum, named as the maximum, carrying the same denominator and the same
> level clause as any other tail figure.

**This floor is derived, not chosen.** `X = 20` is not a convention: it is exactly the smallest `X`
at which `ceil(0.95X) < X`, i.e. where a 95th percentile stops being the maximum. Nothing about it
is tunable, which is what distinguishes it from §11.6's floor and is why the two are separate rules
rather than one blended threshold.

**Two consequences worth stating before someone rediscovers them.**
- **`p50` here is not `statistics.median`.** At even X, type 1 returns the **lower** of the two
  middle observations, not their mean. An implementer must not "fix" `latencyMsP50` to the median:
  the mean of two observations is again a value no call took, and it would diverge from the same
  document's Rule 4 quantile.
- **Cross-run comparability is safe within a pack and only within a pack.** `Y` is fixed by the
  pack, so at equal coverage both arms' tail figures are the *same order statistic* and the
  comparison is like-for-like. Two packs' figures are different order statistics of different item
  sets and are not comparable; nothing in the report may place them in one column.

### 11.4 The denominator — `Y` is the item count, and which fields the withholding governs

**`Y = len(run.items)`** — every item the run recorded — and **`X` = the count of items whose timing
survived**. No item is removed from `Y` for any reason.

The temptation is to net out items that were never timed, so that coverage reads better. Rule 3's
question settles it: *which denominator keeps this sentence's claim true?* The sentence the
denominator carries is a **refusal threshold** (§11.6), and a threshold is weakened by every item
netted out of its base — a conditional denominator would let a run that timed 8 of 38 items report
"8 of 8, full coverage", which is §4.3's laundering pattern with a different numerator. The
unnetted, largest denominator is the conservative one *for this sentence*, the same way §3.4 Rule 3
gives the observable floor the **unfloored** `n_effective`. It also keeps the latency line on the
same base as every other rate in the report (§4.3's `k/n` discipline).

**And which fields the withholding governs is settled by measurement, not by a conservative
default — review G3-3.** `ttftMs`, the `prefillMsPer1kPromptTokens` input and `tokensPerSecond` are
read off the **same response** as the discarded `latencyMs`, and v1.9 withheld all four on the
grounds that whether LM-Studio-side TTFT includes the JIT load was *not established*. **It is now,
and it does not.** Measured this session against **`qwen/qwen3-4b-2507` (Q4_K_M)** from a genuinely
non-resident start — `residency()` returned `[]` before the call and named the model after it:

| | client wall clock | `stats.time_to_first_token` | `stats.generation_time` | wall − (ttft + gen) |
|---|---|---|---|---|
| **cold call (the JIT load)** | **3 625.0 ms** | 49.7 ms | 89.8 ms | **3 485.6 ms** |
| 50 warm calls, same model | 55.1 / 78.2 / 114.6 ms (min/median/max) | — | — | **−11.3 … +7.6 ms** |

The load is 3 485.6 ms that LM Studio's own `stats` never sees. So:

> **Only `latencyMs` — the client wall clock — is withheld on a contaminated item.** `ttftMs`, the
> prefill figure and `tokensPerSecond` measure generation that happened *after* the load, are
> **kept**, and are each printed with **their own denominator** — which is the other half of G3-3's
> fix, and the half that survives the measurement.

**This reverses v1.9's ruling, and the order matters more than the outcome.** The conservative
default — withhold everything until a field is shown clean — was correct *while the question was
open*, and keeping it once the question closed would have been the same defect one document up:
discarding good measurements to honour a caveat that no longer describes reality. **The wall clock
is the contaminated field and `stats` is the clean one**, and their difference is a detector
(§11.5.1). Aggregates over `ttftMs` and prefill are **medians**, so they take §11.6's **p50** gate
against **their own** coverage, which will normally be `Y of Y` while the wall-clock block is short;
`tokensPerSecond` is FR-11 diagnostic-only and still prints its denominator, because a diagnostic
over an unstated subset is the same defect one severity down.

### 11.5 The missingness is informative, and its direction is known exactly

**Two producers of a withheld timing, and both are right-censoring** — which is what lets one rule
cover them:

- **Model-load contamination** (plan §3.6). The call ran with a JIT load inside it. **The load's
  cost varies by an order of magnitude** and no report may name a single figure for it: plan §2.5
  measured **21.068 s**, this session measured **3.625 s** on the same box and the same surface
  (page-cache state is the obvious difference), against warm turns of ~1.3 s in plan §2.2's
  experiment and 55–115 ms for the minimal calls of §11.4's table. That range is why §11.7's slot 3
  names a magnitude and not a number.
- **A scored call that hits `requestTimeoutSeconds`** (review G3-5, whose recommended disposition —
  scored per the pack's rule, `latencyMs = None`, timeout count printed on its own line — this
  section adopts and depends on; §11.9). A censored observation is **not a measurement**: storing
  `latencyMs = 120000` would put the timeout *constant* into the p95, where it is by construction
  the largest value, so the report would print a figure about the configuration rather than about
  the model. That is the opposite bias to contamination and it is the more dangerous of the two,
  because it arrives as a number rather than as a gap.

Both share the one property the strings need: **the withheld call was slower than every timed call,
and its true value is unknown.** Two exact consequences:

- **`p50` is robust.** All withheld points sit above the median, so the reported median moves by
  about `M/2` ranks in the densest part of the sample — the smallest displacement available anywhere
  in the distribution.
- **The tail figure is not, and the damage is computable rather than qualitative.** Since the `X`
  timed items occupy full-run ranks `1 … X`, the printed figure sits at full-run rank `r`, so it is
  — at worst — the quantile of the **whole run** at level

  > **`L = r / Y`**, the **attained level**, with `r` the integer rank of §11.2.1.

  `L` is a *lower bound* on the figure's true level (exactly `p` when nothing was withheld), so the
  clause that prints it must **truncate**, never round to nearest — §3.4 Rule 3's rounding row,
  applied to a new printed bound. The **gate compares the exact integer rank, never the
  display-truncated level** (Rule 7's precedent: an invariant that inherits the presentation layer's
  rounding fires or fails to fire by a display unit).

Worked, exactly, at `Y = 38` — including plan §3.6's own `34 of 38` sketch:

| X of 38 | r (tail) | L (tail) | r (p50) | L (p50) | printed |
|---|---|---|---|---|---|
| 38 | 37 | 97.3 | 19 | 50.0 | both |
| 37 | 36 | 94.7 | 19 | 50.0 | both |
| 36 | 35 | 92.1 | 18 | 47.3 | both |
| 35 | 34 | **89.4** | 18 | 47.3 | **p50 only** |
| **34** | 33 | **86.8** | 17 | **44.7** | **neither** |

#### 11.5.1 The in-call reload detector — R-14's residual, closed by the same measurement

Plan R-14 accepts one residual it cannot see: a reload that **begins and ends inside a single timed
call** is invisible to a between-item residency probe, which only ever looks *between* items.
§11.4's table closes it, and the separation is not marginal:

> **`latencyMs − (ttftMs + generation_time)` is the load, isolated.** Measured: **3 485.6 ms** on the
> cold call against **−11.3 … +7.6 ms** across 50 warm calls — the cold gap is **461×** the largest
> warm gap observed.

**Recommendation:** `runner` computes that gap per item and withholds `latencyMs` as a contamination
(§11.5) whenever it exceeds **1 000 ms**. That threshold sits ~130× above the largest warm gap
measured here and ~3.5× below the smallest cold load measured anywhere (this session's 3 625 ms;
plan §2.5's is 21 068 ms). **It is a starting value with a named basis, not a derived constant** — it
rests on one cold observation here plus one in the plan — so it should be re-checked against the
first real pack run, and the two quantities that would move it are the warm gap's maximum and the
cold load's minimum. The negative warm gaps are expected rather than anomalous (the client wall clock
and the server's own timers bracket different work), which is why the rule is a one-sided threshold
on a magnitude and never `gap > 0`.

This detector is **additive to** the residency probe, not a replacement: the probe catches a reload
that happened *before* an item, the gap catches one *inside* it, and neither sees what the other
does. Placing it is `architect`'s (§11.9); the metric, the threshold and its basis are here.

### 11.6 Floor 2 — the level floor, which refuses; and how the two floors divide the work

**Yes, a latency figure is refused below a coverage threshold, and the refusal is at the *value*,
not in the prose.** The reason it cannot be a prose qualifier is structural: `index.csv` carries
`latencyMsP50` / `latencyMsP95` as **bare columns** and `compare` renders them in a summary table,
so any qualifier living in a sentence is stripped the moment the number is read the way the tool
intends it to be read. This component already has the right precedent — `coldLoadSeconds` is
**absent, never `0`**, when it was not measured — and this is the same move.

> A printed figure's **attained level may fall short of its nominal level by at most 5 percentage
> points**. Below that the figure is **absent** — `None` in the record, an empty cell in
> `index.csv`, and a refusal clause in the report.

Evaluated in integers, with no float anywhere in the gate:

```python
print_p50  = (100 * r50 >= 45 * Y)      # nominal 50.0, floor 45.0
print_tail = (100 * r95 >= 90 * Y)      # nominal 95.0, floor 90.0
```

**The 5-point tolerance is a chosen convention, and it is named as one.** Its basis is the band the
headline figure's own name claims: `p95` names the top 5% of calls, so a figure whose attained level
has slipped a further 5 points is sitting at the outer edge of a band **twice as wide as its name** —
the last point at which the label still describes the number. The same 5 points serve `p50` rather
than a second, level-scaled tolerance, for §3.2c's reason: two constants written in two sentences
drift apart, and the tolerance's job — bounding how far the label may travel from the number — is
measured in the same units at every level. *Reversal trigger:* a stakeholder who wants latency
reported at any coverage moves this constant and nothing else; the strings, the ranks, the
denominator and §11.3's identity floor are all unaffected by its value.

**How the two floors divide the work — G3-11's question answered directly, with the numbers.**
A ratio floor and a count floor do fail in different regimes, and here **neither dominates; they are
complementary and only one of them refuses:**

- The level floor implies a **count** floor, because `r ≤ X` forces `100·X ≥ 90·Y`, i.e.
  **`X ≥ 0.9·Y`**. Every pack in §3.3's table has `Y ≥ 12`, so the level floor already requires
  `X ≥ 11` on the tool-caller pack, `X ≥ 36` on the embedder's 38 and `X ≥ 81` on guard-judge's 85.
  **G3-11's four-surviving-latencies case is refused outright at every declared pack** — it is not
  merely renamed. A bare count floor would have to be re-chosen per pack to achieve that; the level
  floor gets it for free because it is expressed against `Y`.
- What the level floor **cannot** see is a *clean* small sample: at `X == Y == 12` the shortfall is
  zero and the gate passes, correctly. That is exactly the regime §11.3's identity floor governs,
  and it governs it by **renaming**, because the number is sound and only the label is not.

So: **one floor for a sample that is short (refuse), one for a sample that is small (rename).** They
are checked in that order, and the identity floor is evaluated first so that a refusal message names
the figure the run would have had.

**Four properties of the level floor, all checked this session, each falsifiable:**

1. **It can never fire on a clean run.** At `X == Y`, `r/Y = ceil(p·X/100)/X ≥ p/100` by the
   definition of `ceil`, so the shortfall is never positive. A run that timed every item always
   prints both figures, at every X including X = 1.
2. **A single withheld item never suppresses either figure at `Y ≥ 10`** (swept `Y ≤ 399`,
   contiguous from 10). Every pack has `Y ≥ 12`, so the common disturbance — one TTL expiry — costs a
   printed qualifier, not a printed number. **This is what makes the floor safe against G3-4:** if
   the guard's first comparand is left as written and every cold-start run discards item 1, the floor
   still prints both figures on every pack. The floor does not depend on G3-4 being fixed — but
   G3-4 should be fixed anyway (§11.9), because a systematically absent item 1 is a missing
   measurement dressed as a contamination event.
3. **The two figures are gated independently, and the band between them is real.** `p50` survives to
   lower coverage than the tail figure — at `Y = 38`, both print at `X ≥ 36`, `p50` alone at
   `X = 35`, neither at `X ≤ 34`. That asymmetry is §11.5's robustness fact made visible rather than
   averaged away, and it is why the p50-only string exists.
4. **It bites on the plan's own example.** `latency n = 34 of 38` — §3.6's sketch — prints **no
   figure at all** under this ruling. Four withheld items in one run is a disturbed run, and the
   honest output is that it has no latency summary, not a summary of its fast calls.

**What replaces the number.** Not silence: the refusal clause names the coverage, the mechanism and
the direction of the bias, states that **the per-item timings remain in the run record — the summary
is withheld, not the data** — and names the operator's remedy. The scored outcomes are untouched in
every case and the report says so, because a reader who sees a refused latency block must not infer
that the run's quality numbers are also suspect.

### 11.7 The published block — the clause grammar, verbatim, with the condition selecting each

**Why a grammar and not one string per case.** The block varies on three independent conditions
(tail named `p95` or `max`; each of two figures printed or refused; anything withheld or not), which
is twelve monolithic strings to keep in step — more surface than the defect they prevent, and this
note has already paid once for two statements about the same thing drifting apart (§3.2c's trap). So
the **slots and their order are fixed**, each variant is published verbatim, and §11.10 pins five
fully rendered examples as test targets.

**Substitutions.** `<X>` timed count · `<Y>` item count · `<M> = Y − X` · `<ML>` withheld for model
load · `<MT>` withheld for timeout · `<TAIL>` = `max` when `r95 == X` (§11.3) else `p95` ·
`<p50>`/`<tail>` in **milliseconds, rounded to the nearest integer** · `<s> = X − r95` ·
`<L95>`/`<L50>` **truncated** to 1 dp.

**The values in the renderings below are measured, not illustrative.** Source: **50 warm
`POST /api/v0/chat/completions` calls against `qwen/qwen3-4b-2507` (Q4_K_M)** on this box,
`temperature: 0`, `max_tokens: 16`, one short prompt each, following the single cold warm-up call of
§11.4's table on a model `residency()` had just reported non-resident. The **first 38** calls are the
`Y = 38` sample and the **last 12** the `Y = 12` sample; each partial-coverage rendering withholds
the **k slowest** calls of that same sample, which is exactly what the guard does to it. These are
minimal chat calls and so run an order of magnitude faster than plan §2.2's ~1.3 s pack turns — what
is being pinned is the **shape** of the block, and the ranks and levels in it are exact.

**The unit is milliseconds and the block never prints seconds.** `ItemResult.latencyMs` is a float
in ms; `coldLoadSeconds` is the only latency field in this tool measured in seconds, it is reported
separately (plan §3.6) and it never appears inside this block. **The values round to nearest**, not
up and not down: unlike the MDD and the observable floor, a printed latency asserts no bound — it
displays a measurement — so Rule 3's question returns "nearest" for this cell, and the direction
must not be copied from the bounds beside it.

**Slot 1 — the figures.** Exactly one variant, prefixed `Latency (client wall clock): `.

- both printed → `p50 = 78 ms, p95 = 112 ms.`  *(or `max = 104 ms` when `<TAIL>` is `max`)*
- p50 only → `p50 = 76 ms; no p95 is reported.`  *(`no max is reported` under `<TAIL>` = `max`)*
- neither → `not reported for this run.`

**Slot 2 — the denominator.** Always present. Two variants, on `M == 0`:

> `latency n = 38 of 38 items; no timing was withheld.`

> `latency n = 36 of 38 items; timings withheld: 2 (model load 2, request timeout 0).`

A **second denominator line follows it whenever the two differ**, because §11.4 keeps the
server-side timings on an item the wall clock was withheld for:

> `ttft/prefill/tokens-per-second n = 38 of 38 items; these are LM-Studio-side figures and a model load is outside them (see the note's 11.4).`

The cause split is mandatory and both counts print even at zero: the two causes carry very different
operator actions — a TTL reload is a re-run, a timeout is a hung model — and a reader who sees only
the total cannot tell which run they have. Printing both at zero also makes the line's shape
constant, so a reader who has learned it once reads every run the same way.

**Slot 3 — the mechanism.** Present only when `M > 0`. One variant, true under both producers:

> `Every withheld call was slower than every timed call — a model load adds seconds to a call that otherwise takes tens to hundreds of milliseconds, and a timed-out call by definition exceeded the request budget — so the figures below are lower bounds.`

**Slot 3 names a magnitude and never a figure, and that is a correction to v1.9**, which wrote
*"a model load costs about 21 s"* into the string. §11.5's own measurements refute it: the cold load
was **3.625 s** this session and **21.068 s** in plan §2.5. A string that carries a load cost is a
string that is false on most of the runs that render it — the exact defect class four review passes
have been spent removing, reproduced by this note in the act of documenting it.

**Slot 4 — the levels.** Omitted entirely when `M == 0` (nothing was withheld, so the levels are
nominal and a clause restating that would be noise that drifts). Otherwise one variant:

- both printed →
  > `The p95 figure is at worst percentile 92.1 of the run's 38 items, and the p50 figure at worst percentile 47.3.`
- p50 only →
  > `The 95th percentile of the timed calls is at worst percentile 89.4 of the run's 38 items, more than 5 points below the level its name claims, so no p95 is reported for this run. The p50 figure stands as a lower bound: at worst percentile 47.3.`
- neither →
  > `The 95th percentile of the timed calls is at worst percentile 86.8 of the run's 38 items and the 50th at worst percentile 44.7, both more than 5 points below the level their names claim, so the 34 surviving timings are this run's faster calls rather than a sample of it. The per-item timings are in the run record — the summary is withheld, not the data. A latency summary for this model needs a re-run in which the model stays resident throughout, and the scored outcomes are unaffected either way.`
- `X == 0` → replaces slots 2–4 entirely:
  > `latency n = 0 of 38 items; no item's timing survived. The scored outcomes are unaffected; only the timing is absent.`

*"at worst percentile 92.1"*, not *"the 92.1st percentile"*, deliberately: an ordinal suffix on a
decimal is a rule an implementer has to encode and will get wrong, and the phrase carries no less.

**Slot 5 — the tail figure's own reading.** Present only when a tail figure was printed. Two
variants, and the condition is `<TAIL>`:

- `p95` (i.e. `X ≥ 20`) → `Timed calls slower than the p95 figure: 1 of 36.`
- `max` (i.e. `X ≤ 19`) → `At 12 timed calls the 95th percentile is the largest observation, so this run reports the maximum and no p95.`

**Slot 6 — the standing label, on every render whatever the coverage.**

> `Latency is descriptive: it is in no pack's verdictMetrics, it carries no verdict, no confidence interval and no Holm step, and no difference between two arms' latency figures is printed unless both arms report the same order statistic.`

**Slot 7 — `compare`, per figure, when either arm has none.** Both arms' blocks always print in
full, side by side; only the *difference* is withheld:

> `No p95 comparison: qwen/qwen3-4b-2507 has no reportable p95 (latency n = 34 of 38 items).`

**The condition on slot 6's last clause is `r_A == r_B`, not full coverage on both arms, and the
difference matters.** The reason a difference is withheld is that at unequal coverage the two arms'
figures are *different order statistics* (§11.3), so their difference is not a latency difference.
Full coverage is a sufficient condition for equal rank, not the necessary one — and under G3-4 as
written, no cold-start run reaches full coverage at all, so a full-coverage condition would suppress
every latency difference this tool ever prints. The condition must be the one its own rationale
names. **And no paired latency interval is printed on any path:** the intersection of two arms'
timed items is doubly selected by the same informative mechanism, so §3.2d's paired bootstrap has no
valid sample here even though latency is a continuous metric.

**One block assembled end to end**, the `X = 36, Y = 38` case (measured sample, two slowest calls
withheld), because a grammar is only checkable against at least one full rendering:

> `Latency (client wall clock): p50 = 76 ms, p95 = 105 ms.`
> `latency n = 36 of 38 items; timings withheld: 2 (model load 2, request timeout 0).`
> `ttft/prefill/tokens-per-second n = 38 of 38 items; these are LM-Studio-side figures and a model load is outside them (see the note's 11.4).`
> `Every withheld call was slower than every timed call — a model load adds seconds to a call that otherwise takes tens to hundreds of milliseconds, and a timed-out call by definition exceeded the request budget — so the figures below are lower bounds.`
> `The p95 figure is at worst percentile 92.1 of the run's 38 items, and the p50 figure at worst percentile 47.3.`
> `Timed calls slower than the p95 figure: 1 of 36.`
> `Latency is descriptive: it is in no pack's verdictMetrics, it carries no verdict, no confidence interval and no Holm step, and no difference between two arms' latency figures is printed unless both arms report the same order statistic.`

### 11.8 Where the figures live

- **`RunResult` / the record:** `latencyMsP50` and the tail figure are **`None` exactly when the
  gate refuses them** (and `latencyMsP95` is additionally `None` whenever `<TAIL>` is `max`, §11.3),
  never `0` and never a figure carrying a hidden qualifier. `X`, `Y`, `ML` and `MT` are stored beside
  them, so every clause in §11.7 is reconstructible from the record alone.
- **`index.csv`:** the cells are **empty exactly when the record's fields are `None`**. This is the
  property that makes a qualifier-free column honest: *a populated `latencyMsP95` cell is always a
  genuine 95th percentile whose attained level is within 5 points of its name.* That single sentence
  is the whole argument for gating at the value rather than in the prose.
- **`compare` and the per-run report:** §11.7's slots.

### 11.9 What this needs from the plan — `architect`'s, and precisely scoped

R-13 closes on the method side with this section. Four things it depends on are **plan-owned**
(§7 rule 2 — the result schema and the harness surface are the plan's). The first three are review
Pass 3 findings already routed there; they are restated here only as the *shape* this section needs,
so one plan revision can serve both.

1. **G3-5 — the timeout disposition, and it is load-bearing here, not merely adjacent.** §11.5
   depends on a timed-out scored call yielding **`latencyMs = None`** (scored per the pack's rule,
   run continues), which is G3-5's own recommendation. Under the alternative reading —
   `latencyMs = 120000` stored as a measurement — **§11.5's mechanism inverts**: the tail figure is
   then biased *high* by a configuration constant while the level clause claims it is a lower bound,
   and §11.7's slot 3 becomes a false sentence. If the architect chooses to abort the run instead,
   everything above still holds with `MT` pinned at 0.
2. **G3-3 — the three siblings need their own denominator; they must *not* be nulled.** G3-3's
   free measurement has been taken (§11.4): LM-Studio-side TTFT **excludes** the JIT load, so
   `ttftMs`, prefill and `tokensPerSecond` stay on a contaminated item and print `n = Y of Y` beside
   the wall clock's `n = X of Y`. The defect G3-3 names is real and its fix is the **denominator**,
   not the nulling. **This reverses what this section asked for at v1.9**, on the evidence G3-3
   itself proposed gathering.
2a. **New, from the same measurement: the in-call reload detector (§11.5.1).** `runner` computes
   `latencyMs − (ttftMs + generation_time)` per item and withholds `latencyMs` above **1 000 ms**.
   It closes R-14's acknowledged residual — a reload inside a single timed call, which the
   between-item probe structurally cannot see — at the cost of one subtraction per item, and it
   needs no new probe. Its placement is the plan's; the metric and the threshold are §11.5.1's.
3. **G3-4 — the guard's first comparand.** §11.6(2) shows the floor holds either way, so this is not
   a blocker for R-13. It should still be fixed as G3-4 specifies (baseline = a probe taken *after*
   the warm-up returns), because otherwise `MT`/`ML` count a systematically absent item 1 as a
   contamination event and slot 2's line reports a mechanism that did not occur.
4. **New, and this section's own ask: a `latencyMsMax` field and column.** §11.3 makes
   `latencyMsP95` `None` for every run with `X ≤ 19` — which is *every* tool-caller run, the pack
   that is the long pole. Without somewhere to put the maximum, the component's headline pack has no
   tail figure in the record or in `index.csv` at all. The field's name is the plan's; the
   requirement is that the number have a home. **Recommended alongside it:** columns carrying `X`
   and `Y`, so coverage is visible to a CSV reader. That one is a readability improvement rather
   than a correctness requirement — the gate is what makes the existing columns honest without it.

Two consequences that are the standing sweep obligation of plan §7 rather than new asks:
**§3.6's `latency n = X of Y` sketch and §5 test 15b's assertion are superseded by §11.7's slots** —
the plan cites them, as §3.9 already cites §3.2e's verdict strings, and does not restate them; and
**test 15b's scenario still renders a figure** (§11.6(2)), so the test remains satisfiable under
either G3-4 disposition.

### 11.10 Evaluation design — how to prove this implementation right

Pure functions and rendered strings; no network, no model. Asserted **exactly** — every quantity
here is an integer, a string, or a value copied from the input, so no tolerance is appropriate and
any tolerance would hide the defect it was meant to catch.

1. **Rank fixtures.** `rank(950, X)` for `X ∈ {1, 12, 19, 20, 38, 40, 85, 100}` equals
   `{1, 12, 19, 19, 37, 38, 81, 95}`, and `rank(500, X)` equals `{1, 6, 10, 10, 19, 20, 43, 50}`.
   Integer equality. This is §11.3's table and it is the one test that pins the estimator itself.
2. **The bin-edge guard.** `percentile(range(25), permille=280)` returns the **7th** value; the float
   form `math.ceil(0.28 * 25)` returns the 8th. The test asserts the integer form's answer, so a
   "simplification" back to the float expression fails. Basis: `0.28 * 25 == 7.000000000000001`,
   measured.
3. **One implementation.** `modelbench.results` exposes no private percentile helper, and its
   percentile *is* `stats.percentile` (identity, not equality of behaviour). This kills review M27's
   defect class at the import boundary rather than at a call site.
4. **The identity floor.** At `X = 19` the record's `latencyMsP95` is `None` and the rendered block
   contains `max = ` and slot 5's `max` variant; at `X = 20` it contains `p95 = ` and
   `Timed calls slower than the p95 figure: 1 of 20.` The boundary is asserted on both sides,
   because a floor tested on one side is a floor with an untested inequality.
5. **Level-floor boundaries, at `Y = 38`.** `X = 36` → both figures present; `X = 35` → `p50`
   present and the tail `None`; `X = 34` → both `None`. Exactly §11.5's table, and it is the test
   that fails if the 5-point tolerance is silently moved.
6. **The clean-run invariant.** For every `X` in `1…200`, a run with `X == Y` has both figures
   present. §11.6(1) is a theorem; a fire means the gate is wrong, not the data.
7. **The two denominators (§11.4).** An item withheld by the guard contributes to **no**
   `latencyMs` aggregate and **does** contribute to `ttftMs`, prefill and `tokensPerSecond`; the
   rendered block prints **both** denominator lines and they differ (`n = 36 of 38` against
   `n = 38 of 38`). A test asserting one denominator for all four fields pins v1.9's withdrawn
   ruling and must fail.
7a. **The in-call reload detector (§11.5.1).** An item whose
   `latencyMs − (ttftMs + generation_time)` exceeds 1 000 ms has its `latencyMs` withheld and is
   counted under model load; one at 7.6 ms — the largest warm gap measured — does not. Both sides of
   the threshold, since a threshold tested on one side is an untested inequality.
8. **String rendering.** Five blocks rendered against fixtures and asserted **verbatim**, the way
   §7.2's resolving-power line is, with §11.7's measured sample as the fixture:
   `(X=Y=38)` → `p50 = 78 ms, p95 = 112 ms`, `1 of 38` slower ·
   `(X=Y=12)` → `p50 = 77 ms, max = 104 ms`, the `max` case ·
   `(X=36, Y=38)` → `p50 = 76, p95 = 105`, levels 92.1 / 47.3 ·
   `(X=35, Y=38)` → `p50 = 76` only, levels 89.4 / 47.3 ·
   `(X=34, Y=38)` → both refused, levels 86.8 / 44.7.
   Plus one with `MT > 0`, so the cause split is exercised rather than assumed. **The fixture is the
   measured sample, so the ranks and levels are checkable by hand against it** — which is the
   property a hand-invented fixture does not have.
9. **The unit guard.** Every printed figure in the block carries ` ms`, and no latency in the block
   is printed in seconds. The ms/s boundary sits one field away from `coldLoadSeconds`, and a figure
   printed in the wrong unit is a defect no reader can detect from the report.
10. **Empty input.** `percentile([], permille=950)` raises; `latency_summary` over a run with no
    timed item renders the `X == 0` variant. The absent-or-present decision lives in exactly one
    place.

**Acceptance:** all ten pass, and no stored record carries a `latencyMsP95` whose attained level is
below 90.0 or whose rank equals its `X` — both checkable over `results/runs/` at any time, and
together they are the invariant §11.8 sells.
