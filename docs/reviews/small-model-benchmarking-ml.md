# Small-Model Benchmarking — Methodology Review of S1's Statistics

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** —

**Scope.** `model-bench/modelbench/stats.py`, `model-bench/tests/test_stats.py`, and the
statistical claims rendered by `model-bench/modelbench/report.py`, as delivered in commit
`ab91419`. Judged against `docs/plans/small-model-benchmarking-ml.md` v1.4 (the method note —
**§3.4's six binding rules**, §3.2c's fixtures, §7.1/§7.2's resolving-power figures),
`docs/plans/small-model-benchmarking.md` v1.4 §4 "S1", and `docs/reviews/small-model-benchmarking.md`
Pass 1/Pass 2 (blocker **B-1**, finding **N-1**). Engineering correctness and seam design are the
concurrent `analyst` review's; this pass is the mathematics and what the prose claims about it.

**Method.** Every figure below was **re-derived in this run from scratch**, not checked against the
note: a 60-significant-digit `decimal` implementation of Wilson and MOVER-D, exact `Fraction`
arithmetic for McNemar and the `b_min` floor tables, and an independent rational-power bisection for
MDD₈₀ — then compared three ways (independent re-derivation ⟷ the note's published table ⟷ the
delivered module's behaviour). The suite was run from `model-bench/` (`233 passed in 0.47s`).
Nothing in the working tree was modified.

**Verdict: needs changes.** 1 blocker, 4 major, 5 minor, 3 nits.

The arithmetic is right — every fixture, every floor, every MDD figure and every published Wilson
interval reproduces exactly, three ways, and Rules 1, 2, 3 and 5 are genuinely binding in code
rather than honoured by convention. What is not right is the **clustered branch of Rule 4**: it
changes the instrument's name and its label, and does not change the interval. On the exact data
gate B-1 was raised about, the delivered `verdict()` declares a 15.0 pp difference *distinguishable*
while the same report's mandatory resolving-power line says nothing below 105.0 pp can reach
significance. That is B-1's shape, one layer further in.

---

## Findings

### Blocker

**B-ML-1 — the clustered decision path does not cluster; it re-runs the anti-conservative test under
a different name.** `stats.verdict()` correctly refuses to let McNemar decide unless
`design_effect == 1.0 and basis == "by-construction"` (Rule 4's letter). The substitute is
`paired_bootstrap(outcomes.unit_diffs(), B=10_000, seed=…)` — an **i.i.d. resample of the rows of
the paired table**. Those rows are the very observations the declared design effect says are
correlated, so the resample cannot see the clustering, and `design_effect` never enters the
interval at all. Measured this run on the `(34, 6, 0, 0)` table, `n_units = 40`:

| design effect | n_effective | printed floor | printed MDD₈₀ | decided_by | CI on the difference | verdict |
|---|---|---|---|---|---|---|
| 1.0 (`by-construction`) | 40.00 | 15.0 pp | 19.1 pp | `mcnemar-exact` | [3.2, 29.1] pp | distinguishable |
| 2.0 (`measured`) | 20.00 | 30.0 pp | 36.7 pp | `cluster-bootstrap` | **[5.0, 27.5] pp** | distinguishable |
| 4.0 (`measured`) | 10.00 | 60.0 pp | 67.4 pp | `cluster-bootstrap` | **[5.0, 27.5] pp** | distinguishable |
| 7.0 (`measured`) | 5.71 | 105.0 pp | 100.0 pp | `cluster-bootstrap` | **[5.0, 27.5] pp** | distinguishable |

Three consequences, in ascending order of seriousness:

1. The interval is **invariant to the design effect** — identical to four decimal places across
   DEFF ∈ {2, 4, 7}.
2. It is **narrower than the interval it replaced**: 22.50 pp wide against MOVER-D's 25.90 pp. The
   branch that exists because McNemar is anti-conservative under clustering substitutes something
   *less* conservative than McNemar's own effect-size interval.
3. At DEFF = 7 the rendered report contains both `"…is better than…: +15.0 pp"` and
   `"Differences below 105.0 pp cannot reach significance at any observed outcome."` A measuring
   instrument must not emit two statements that contradict each other, and here the wrong one is
   the verdict.

The root cause is a **missing primitive, not a wrong formula**. `cluster_bootstrap()` exists and is
correct, but it computes a percentile CI for **one arm's pooled rate**, not for a **paired
difference over clusters** — and nothing calls it. `PairedOutcomes` carries one row per unit id and
no grouping, so there is no structure in which a paired cluster resample could be expressed.

*Suggested fix, smallest honest version at S1:* keep the branch, and make the interval respond to
the declared design effect — inflate the percentile half-widths about the point estimate by
`sqrt(design_effect)` (the Kish variance ratio is exactly the quantity that converts, Rule 5), and
say so in the prose (`"CI widened by sqrt(DEFF)=… for the declared clustering"`). The structurally
right version, when a pack that needs it exists: give `PairedOutcomes` an optional cluster grouping
and resample clusters of paired differences.

*And regardless of which fix lands, add the invariant test* — it catches this whole class in one
line and is cheap:

> **No verdict may be `distinguishable` when `|diff| < resolving.observable_floor`.**

I verified this run that the McNemar path satisfies it **by construction** — exhaustively over
n ∈ {12, 20, 30, 40, 48, 85} and every `(b, c)` with `b + c ≤ n`, there is no table where
`mcnemar_exact(b, c) ≤ 0.05` and `|b − c|/n` falls below `6/n`. The clustered path violates it in
every row of the table above. That asymmetry is the test.

---

### Major

**M-ML-1 — `min_detectable_difference` prints a false claim when no effect size attains the
power.** When `n < b_min(alpha)` the McNemar rejection region is empty, `_mcnemar_power` returns
`0.0` for every δ, and the bisection converges on its upper bracket: the function returns `1.0` and
the report renders *"This pack resolves differences of >=100.0 pp with 80% power…"*. Power at
δ = 1.0 is **zero**, not 0.80. Reachable today: `n_units = 40` with DEFF = 7 gives `n_effective =
5.71`, floored to 5, and `b_min(0.05) = 6` — exactly the table row above. The note already has the
right vocabulary for this state (§7.1's discordance-mix table says **"unattainable"**), it is just
not in the code.
*Suggested fix:* return a sentinel (or raise) when `_mcnemar_power(n, 1.0, alpha=alpha) < power`,
and have `resolving_power_line` render *"no difference is resolvable at this n_effective — fewer
than b_min = N units"* in place of both the MDD sentence and the floor sentence.

**M-ML-2 — Holm is printed, Bonferroni is applied, and the printed threshold can contradict the
verdict.** Every verdict in a k-member family is computed at `alpha = 0.05/k` (forced by Rule 4's
third precondition), which is Bonferroni. `report.py` then prints a *"Holm-adjusted threshold"*
column from `holm_thresholds(p_values, alpha=0.05)`. Computed this run: `holm_thresholds([0.01,
0.04])` → `[0.025, 0.05]`. So a guard-judge report can print

| metric | McNemar p | Holm-adjusted threshold |
|---|---|---|
| falseAdvanceRate | 0.010 | 0.0250 |
| falseSuspendRate | 0.040 | **0.0500** |

beside a `falseSuspendRate` verdict that reads *"Not distinguishable at this sample size."* A reader
comparing p = 0.040 against the printed 0.0500 concludes the opposite of the verdict — and here the
*correct* Holm answer is "distinguishable", so the tool is not merely inconsistent, it is discarding
resolving power §7.3 already told the stakeholder was expensive. Two further defects in the same
place: `holm_thresholds` carries **no step-down stopping rule**, so the larger threshold is only
valid if the smaller p rejected (`holm_thresholds([0.30, 0.04])` → `[0.05, 0.025]`, and that 0.05 is
unusable); and `verdict()` already has the `alpha_step` hook built for exactly this, which
`report.py` never passes.
*Suggested fix:* apply Holm properly — sort, test at `alpha/(k−rank)` via `alpha_step`, stop at the
first non-rejection and mark the remainder `not tested (Holm stops here)`; or drop the column, call
the correction Bonferroni in the heading, and note that Holm is available and unclaimed.

**M-ML-3 — Wilson intervals are printed over turn-pooled counts.** §4.4's first mandatory
consequence is verbatim: *"Never print a Wilson interval over a turn-pooled count."* The Arms table
in `compare_report` renders `stats.wilson_interval(metric.successes, metric.n)` for **every**
`BinaryMetric` returned by `aggregates.named_metrics()`, and `ToolCallAggregates.named_metrics()`
returns `[cleanThroughTurn, restraint, *funnel, *hazard]` — `funnel` and `restraint` are turn- and
call-denominated by §4.3's own definition. Rendered this run from a representative
`ToolCallAggregates`:

| metric | k/n | printed Wilson | width |
|---|---|---|---|
| `cleanThroughTurn4` | 9/12 | [0.468, 0.911] | 44.3 pp — **legitimate**, n is conversations |
| `restraint` | 38/40 turns | [0.835, 0.986] | 15.1 pp |
| `nativeCallEmitted` | 142/320 turns | [0.390, 0.499] | **10.8 pp** |
| honest bound at the §4.5.1(i) cap (n_eff ≤ 12 clusters, p̂ ≈ 0.44) | 5/12 | [0.193, 0.680] | **48.7 pp** |

A 10.8 pp interval where the honest one is ~48.7 pp is the "fiction" §4.4 names, understated 4.5×.
The `exploratory — no significance claim` label these metrics also receive mitigates the *verdict*
risk and does not cure the *interval*: a printed ±5 pp is read as precision whatever it is labelled.
The proximate cause is that `BinaryMetric` carries no denominator unit, so `report.py` cannot tell a
per-analysis-unit rate from a turn-pooled one.
*Suggested fix, in the same structural spirit as the closed aggregate union:* add a `unit: str`
field to `BinaryMetric` and render an interval only when `unit == pack.analysisUnit`; otherwise
print `k/n` with the cluster-bootstrap interval (Rule 6's `cluster_bootstrap`, currently unused by
any caller) or with no interval and the footnote `n_eff <= <cluster count>`.

**M-ML-4 — rows dropped from the pairing leave no printed trace.** `_paired_rows` silently skips an
item absent from arm B, and an item whose `scoreable[metric]` is false in **either** arm. §4.3 rule
2 requires *"a turn excluded from a conditional denominator is counted in that count's own `n/a`
tally, printed next to the rate"*, and §4.3's paired-comparison corollary requires the
**`asymmetry`** count — items scoreable for exactly one model — printed as a finding about the model
that could not produce them. The plan's S1 text names `asymmetry` as computed from `pairingKey` plus
`scoreable`. `grep -rn asymmetry model-bench/modelbench/` returns nothing. The paired `n` printed
inside the verdict string does shrink honestly, so the *statistic* is not laundered — but a reader
cannot see that it shrank, or which arm caused it. This is also the only place a violated
`H ≤ min(script length)` would surface at S1 (see the clean finding on `H` below).

---

### Minor

**m-ML-1 — the §3.2c/§9.1 MOVER-D tolerance, adjudicated.** The implementer's report is **correct**,
and the defect is the note's, not the implementation's. Re-derived at 60 significant digits, the
lower bound for `(34, 6, 0, 0)` is `3.1762869443…` pp = `0.031762869443…` as a proportion, against a
published `3.1763`; the gap is **1.3056 × 10⁻⁷ as a proportion — 131× the mandated 1e-9**. So the
mandate is unassertable against the table as published. But it is **not** unassertable in principle:
the delivered float implementation agrees with the 60-digit value to **≤ 1.44 × 10⁻¹⁶** on all ten
bounds, i.e. roughly **7 × 10⁶ ×** inside the 1e-9 mandate. The table is *under-precise, not wrong* —
all five rows reproduce their published 4-dp figures exactly.

*What the note should state instead:* keep the 1e-9 tolerance and **publish the fixtures at the
precision that makes it assertable.** The ten bounds, in percentage points, re-derived this run at
`z = 1.959963984540054`:

| a | b | c | d | MOVER-D lower (pp) | MOVER-D upper (pp) | McNemar p (exact rational) |
|---|---|---|---|---|---|---|
| 34 | 6 | 0 | 0 | 3.1762869443 | 29.0723243665 | 1/32 = 0.031250 |
| 30 | 6 | 0 | 4 | 3.8506738324 | 27.7026867131 | 1/32 = 0.031250 |
| 33 | 6 | 1 | 0 | −0.9864868353 | 26.8581964973 | 1/8 = 0.125000 |
| 20 | 8 | 2 | 10 | 0.1708978316 | 28.7785182732 | 7/64 = 0.109375 |
| 72 | 10 | 2 | 1 | 1.4800198994 | 18.2130920778 | 79/2048 = 0.03857421875 |

*On the substitute the implementation shipped* (`round(x * 100, 4) == published`): **adequate as a
stopgap, and its load-bearing claim holds** — I re-derived all ten bounds at `z = 1.96` and every one
of them leaves its 4-dp bin, so `test_the_pinned_z_constant_is_load_bearing_at_this_tolerance` is
not a lucky single row. But it is a **1000× looser** assertion than the mandate (it reliably detects
errors ≥ 1e-4 pp = 1e-6 as a proportion, and tolerates up to ~5e-5 pp silently), and the docstring
on `test_mover_d_reproduces_the_regression_fixtures` mis-states the margin by three orders: it says
the tolerance is *"four orders tighter than the ~3e-4 pp by which the z = 1.96 rounding moves these
bounds"*. It is not — 1e-6 as a proportion against a divergence of ~3e-6 as a proportion is a margin
of **~3×**, not 10⁴. Three orders of that sentence were inherited from the note's justification for
1e-9 and do not survive the substitution. Correct the docstring in this change; the note's fold-in
pass carries the fixture table above.

**m-ML-2 — the observable floor is rounded to nearest where Rule 3 mandates ceiling, and it errs
optimistic.** Rule 3: *"Every figure in §7.1's table is the exact value ceilinged to 0.1 pp."*
`report.py`/`verdict()` format the floor with `f"{value*100:.1f}"`, which rounds to nearest. Live
instance in the pack that most needs it — guard-judge's `falseSuspendRate` slice, n = 30, α = 0.025:
`7/30 = 23.3333…` pp printed as **23.3**, ceiling **23.4**. Same at n_eff = 12, α = 0.025:
`58.3333…` → printed **58.3**, ceiling **58.4**. Every α = 0.05 row is unaffected (nearest and
ceiling coincide at all ten). This is a **three-way agreement on the wrong rounding**, not a
divergence: §7.1's and §7.3's tables, `test_mdd_and_floor_at_the_family_adjusted_alpha`, and the
code all carry 58.3 / 23.3. Magnitude 0.03–0.07 pp; direction is the one the note exists to police
(a printed floor below the true floor invites a claim the instrument cannot support). Fix in code
and note together.

**m-ML-3 — `RunResult.designEffect` defaults to `1.0`, re-admitting the default Rule 2 bans.** Rule
2 removes the `1.0` default from `resolving_power` precisely so no caller can assert DEFF = 1 by
omission; `results.py` restores it one layer out (`designEffect: float = 1.0`, and
`from_dict` supplies `d.get("designEffect", 1.0)`). The *decision* is protected — `basis` defaults
to `"assumed"`, which moves the branch off McNemar — but the **resolving-power line is not**: a
clustered record that omits the field prints `design effect 1.00, assumed` and an MDD computed at
the raw unit count, which is the over-optimistic sentence B-1 was about.
*Suggested fix:* no default, or `None` with the report refusing to render a resolving-power line
when it is absent.

**m-ML-4 — `basis` degradation mislabels a measured design effect as assumed.** `report.py` uses
`"by-construction" if a.basis == b.basis == "by-construction" else "assumed"`, so two arms with a
genuinely **measured** DEFF print `assumed` in the one sentence whose entire job is auditability
(§7.1: *"the design effect and its basis are what make the line auditable"*). Fail-safe for the
decision, false as provenance.
*Suggested fix:* keep the decision rule unchanged; print the weaker of the two **actual** bases.

**m-ML-5 — the conditionality sentence says "scripts" for every unit kind.** `resolving_power_line`
parameterises the sample noun in the first half (`_SAMPLE_NOUN`: item → "items", query → "queries")
and hardcodes *"generalization to unwritten **scripts**"* in the second. On a guard-judge or
embedder pack the sentence reads *"conditional on the 85 items … generalization to unwritten
scripts"*. §4.5.1(ii)'s clause was written for the conversation pack; the claim it makes is right
for all of them, the noun is not.

---

### Nits

**n-ML-1 — the floor and the MDD on the same printed line use different denominators.**
`observable_floor` divides by the unfloored `n_effective`; `min_detectable_difference` floors it
first (`_require_effective`, conservatively and deliberately). At non-integer n_eff the two figures
in one sentence therefore rest on different n — at n_eff = 5.714, the floor is `6/5.714 = 105.0 pp`
while the MDD is computed at n = 5. Immaterial in magnitude; worth one line of consistency.

**n-ML-2 — §3.2a's "at most 3.0 × 10⁻⁴ pp".** The measured maximum divergence between `z = 1.96` and
the pinned constant across the ten fixture bounds is **3.017 × 10⁻⁴ pp**, on the *upper* bound of the
`34,6,0,0` row (the note names the right row). Round it up in the same fold-in that carries m-ML-1.

**n-ML-3 — the pinned `_Z_95` is one ULP away from `NormalDist().inv_cdf(0.975)`, and that is fine
but should be written down.** Measured this run: `_Z_95 = 1.959963984540054` (the same literal
`falkor-chat/server/tests/eval/nlq_scoring.py:59` pins) against
`statistics.NormalDist().inv_cdf(0.975) = 1.9599639845400536` — **not equal**, differing by
4.44 × 10⁻¹⁶. `test_z_95_matches_the_inverse_normal_cdf` asserts `< 1e-12` and is therefore correct
as written; the trap is a future tightening pass turning it into `==`. Worth one clause in the
test's docstring saying the pinned value is the 16-significant-digit decimal of that double, not
the double itself. The fixture bounds are untouched by it (total float error against the 60-digit
truth is ≤ 1.44 × 10⁻¹⁶ either way).

---

## What I re-derived and found clean

Three-way agreement (independent 60-digit / exact-rational re-derivation ⟷ note ⟷ module), so these
are settled rather than merely unchallenged:

- **The five §3.2c fixtures.** All ten MOVER-D bounds reproduce the published 4-dp values exactly;
  the module's float output sits within 1.44 × 10⁻¹⁶ of the 60-digit truth. All five McNemar
  p-values are **bit-exact** against `Fraction` arithmetic (1/32, 1/32, 1/8, 7/64, 79/2048).
- **The `b_min` floor tables.** α = 0.05: `{0:6, 1:8, 2:10, 3:12, 4:13}`; α = 0.025:
  `{0:7, 1:9, 2:11}` — both exactly as §3.2b/§9.1 publish (I additionally derive c=3→13, c=4→15 at
  α = 0.025; no conflict). `_b_min` returns 6 and 7 respectively.
- **§7.1's exact MDD₈₀ column, all ten rows at α = 0.05**: 57.793886, 47.559088, 36.645935,
  25.074758, 20.008921, 19.046379, 15.971939, 12.857448, 9.142149, 6.508549 pp — matching the note's
  57.794 / 47.559 / 36.646 / 25.075 / 20.009 / 19.046 / 15.972 / 12.857 / 9.142 / 6.509. The
  α = 0.025 MDDs (65.6 / 54.2 / 28.7 / 21.9) match too.
- **Rule 3's ceiling, and why it is not cosmetic.** Measured power at n = 40, α = 0.05 is
  **0.797962** at δ = 19.0 pp and **0.802338** at 19.1 pp — the note's 0.798 / 0.8023, to the digit.
  The ceiling genuinely bites at n = 38 (20.009 → 20.1, nearest would say 20.0) and n = 85 (9.142 →
  9.2, nearest 9.1), and the suite asserts both.
- **Rule 5's ρ = 1 identity.** DEFF = 1 + (m−1)ρ = 7 at m = 7, width ratio √7 = 2.6458 (v1.1's
  "≈2.6"), `effective_n(280, 7) == 40.0` — exactly the cluster count; and the mutation
  (`effective_n(280, √7) = 105.8 ≠ 40`) is asserted, so the squaring error is caught in both
  directions.
- **Every Wilson figure the note publishes**: §4.4's five per-position widths (0/12, 12/12, 0/8,
  0/4, 4/4), §3.1's worked marginals (40/40 → [0.912, 1.000], 34/40 → [0.709, 0.929]) and §7.4's
  saturation case (38/38 → [0.908, 1.000] vs 37/38 → [0.865, 0.995]) all reproduce at 50 digits.

Contract rules, checked as *binding in code* rather than honoured by the current call sites:

- **Rule 1** — `from_units` is the only public constructor and the duplicate guard additionally runs
  in `__post_init__`, so no construction route bypasses it; the 48-rows-from-12-script-ids fixture
  raises. Both the module docstring and `report.py`'s header state plainly that this is a
  **backstop, not the mechanism** (N-1), and `report.py` resolves the unit id from
  `PackRef.analysisUnit` with no parameter through which a caller could choose it — which is what
  actually closes the clustered case, and what DC-5(c) tests.
- **Rule 2** — `unit_kind`, `design_effect`, `basis` and `alpha` are all `KEYWORD_ONLY` with
  `Parameter.empty` defaults, asserted per-parameter. No `1.0` anywhere in that signature. (The
  leak is one layer out — m-ML-3.)
- **Rule 3** — `_require_effective` rejects `int` **and** `bool` (the `isinstance(x, bool)` guard is
  right: `True` is an `int`), and both `min_detectable_difference(48, …)` and the `_exact` variant
  raise `TypeError` on a bare count.
- **Rule 4** — all four preconditions raise, not warn, and `alpha == 0.05/len(family)` is asserted
  to 1e-12 so a k-member family cannot be reported at α = 0.05 by oversight. The
  `basis == "assumed"` case moves off McNemar even at DEFF exactly 1.0, which is the correct
  fail-safe. Only the *substitute instrument* is wrong (B-ML-1).
- **Rule 6** — `cluster_bootstrap` is one level, seeded, `seed` keyword-only with no default,
  reproducible, and demonstrably wider than the naive Wilson on ρ = 1 data. It has **no caller** at
  S1, which is acceptable (S2's `validate_pack` owns the `replicatesPerScript > 1` refusal) but is
  also why M-ML-3 and B-ML-1 both had to improvise.

**The orientation fix (asked for explicitly) is correct in both directions, and no sibling path
carries the defect.** Verified by construction, on both decision paths:

- `(34, 0, 6, 0)` renders `"incumbent is better than cand on m: +15.0 pp (95% CI [3.2, 29.1] pp)…"`
  — the difference re-oriented to the winner **and** the interval flipped with it, `(-hi, -lo)`,
  which is the correct transform for a difference interval.
- `(34, 6, 0, 0)` renders the mirror image with A named. Same behaviour at DEFF = 2 on the bootstrap
  path, so the flip is not specific to the MOVER-D branch.
- The two non-significant strings are the sibling paths, and both are safe: the "covers zero" string
  keeps A−B orientation and prints the **signed** difference (`-12.5 pp` with `[-26.9, 1.0]`, so the
  point estimate always sits inside the printed interval); the "instruments disagree" string prints
  **no signed difference at all**, only the interval, so it cannot desynchronise.

**`H ≤ min(script length)` — nothing in the delivered code assumes `H` is pinned to the minimum, and
no denominator became selection-conditioned.** `H` is never derived, clamped or read at S1:
`cleanThroughTurn` is an opaque `BinaryMetric` whose name (`cleanThroughTurn4`) arrives from the
pack manifest, and the paired denominator comes from `_paired_rows`' intersection over actual
`ItemResult`s, not from any script-length arithmetic. There is no hardcoded 12 or 40 in any
denominator. If a future pack did violate the invariant, the resolving-power line would honestly
report the shrunken `n` — the residual risk is **visibility, not correctness**, and it is M-ML-4.

**The report's prose** (checked as words, not just numbers):

- `"Not distinguishable at this sample size."` verbatim, on both non-significant paths.
- The descriptive-vs-instrument line under the Arms table, and the marginal-overlap **diagnostic**
  with its §3.1 inertness footnote — FR-15 visibly honoured, never the verdict.
- `headlineMetric: null` produces **no** headline and no arithmetic above the co-equal metrics, with
  the "printed side by side, in the manifest's declared order" sentence — structurally, not by
  guard, since `ClassificationAggregates` has no field to hold a pooled figure.
- §7.2's four-sentence resolving-power line renders **verbatim**, including the n_eff < 20 power-
  ceiling clause and its correct suppression above 20 — and its two numbers (57.8 pp, 50.0 pp) are
  ones I re-derived independently.
- Verdict 3's requirement that both instruments' individual outcomes appear in the prose is met.

---

## Routing

- **B-ML-1, M-ML-1, M-ML-2, M-ML-3, M-ML-4, m-ML-3, m-ML-4, m-ML-5, n-ML-1, n-ML-3** and the docstring half
  of **m-ML-1** → implementation (`tdd-engineer`, with the invariant test in B-ML-1 written first).
- **m-ML-1** (publish the fixture table at 10 dp, keep the 1e-9 tolerance), **m-ML-2** (§7.1/§7.3
  α = 0.025 floor column: 58.3 → 58.4, 23.3 → 23.4) and **n-ML-2** → the method note
  `docs/plans/small-model-benchmarking-ml.md`, in the fold-in pass after both reviews land. I did
  not edit it in this run, by the brief's constraint.
- **m-ML-2** additionally touches `tests/test_stats.py::test_mdd_and_floor_at_the_family_adjusted_alpha`,
  which must move with the note.

---

## Pass 2 — 2026-09-03, commit `3ad27d3`

**Verdict: needs changes.** 1 new blocker, 1 new major, 1 new minor. Every Pass 1 finding is
**fixed** or correctly **declined** — the fix round is good work, and both of the judgements the
coordinator put to me come back mostly in the implementer's favour. What is not fixed is a corner
the fix opened up: the same class of defect as B-ML-1, surviving in the one configuration
`sqrt(DEFF)` cannot reach.

Re-ran the suite from `model-bench/`: **296 passed in 2.11s**. All figures below re-derived in this
run.

### The three judgements asked for

**1. `sqrt(DEFF)` inflation of a percentile interval — defensible as an interim. Accepted.** The
conversion is exact in the sense claimed, not a fudge: Kish's DEFF is a variance ratio and a CI
half-width scales as `1/sqrt(n)`, so `sqrt(DEFF)` is precisely the factor carrying an interval
computed at `n_units` to one computed at `n_units/DEFF`. I checked the arithmetic against a
hand-computed scaling of the raw percentile interval about the point estimate — exact to 1e-12 at
DEFF ∈ {1, 2, 4, 7} — and confirmed the clamps hold. Scaling the two half-widths **separately**
preserves the resample's skew rather than symmetrising it, which is the right choice of the two
available. Its real limitation is the one the docstring already states — it rescales a variance it
cannot *discover* — plus one it does not: with few clusters a genuine cluster bootstrap degenerates
(the note's own §4.5.1(ii) rejects bootstrapping 3 shapes for exactly this), and `sqrt(DEFF)`
widening does not reproduce that failure mode, it only widens. That gap is covered by a different
guard — the floor is computed at `n_eff = n/DEFF`, so a too-few-clusters configuration is caught by
Rule 7 or by `UnattainablePower` rather than by the interval. The two mechanisms compose; worth
saying so in the note rather than leaving it to be re-derived.

**The deferral argument is sound, and I would not build the structural primitive before S5/S6.**
The seam genuinely does not exist: a paired resample *of clusters of rows* needs a grouping that can
only arrive from a pack declaring `replicatesPerScript > 1`, and Rule 6 makes that a validation
error while only the one-level bootstrap exists. Building the primitive now means building it
against no data, no caller and no test fixture that isn't synthetic — and the thing that would
falsify the interim (a pack whose clusters are not its rows) is the same event that unlocks the
real one. Build it when the pack that needs it is specified, not on a calendar.

**2. Rule 7 as demote-and-name — right on the substitute path, wrong on the McNemar path.** Their
reasoning for demoting is correct and I verified the case they cite: at DEFF = 2 on `(34, 6, 0, 0)`
the widened interval `[0.9, 32.7]` still excludes zero while 15.0 pp sits below the 30.0 pp floor,
so raising there would abort on ordinary clustered data. Demote-and-name is the right reading —
**there**. On the McNemar branch the invariant is a *theorem*, not a guard: I re-confirmed
exhaustively that no `(b, c)` with `p <= 0.05` has `|b−c|/n` below `b_min(0.05)/n`, over
n ∈ {12, 20, 30, 40, 48, 85}, zero violations. A fire on that branch is therefore a bug in the
module, and silently demoting it discards exactly the detector property Rule 7's own docstring
claims for it ("the only cheap check tying together two quantities computed by completely
independent routes"). Split it by path: demote on `cluster-bootstrap`, raise on `mcnemar-exact`.
This is **m-ML-6** below, and it depends on M-ML-6 — as built, the McNemar branch *is* reachable,
but only through the α gap that is itself the defect.

**3. The Holm/floor α gap — I contest the resolution, not the concern.** The concern is right and is
my own rule: a verdict must never contradict the honesty line beside it. But there are two ways to
remove a contradiction, and the build picked the one that bends the decision instead of the one that
corrects the claim. Ruling: **the printed floor is wrong, and the Holm rejection is right.**

Under Holm a member's threshold is data-dependent, so no single α makes the floor's sentence true by
construction — you have to pick one, and the choice is settled by the same principle that settled
the rounding direction in Pass 1: **each bound takes the parameter value that keeps its own claim
true.**

- The floor claims *"differences below Y cannot reach significance at any observed outcome."* That
  is true only at the **loosest step the member can face**, α = 0.05, giving `6/n`. Printed at α/k
  it is `7/n` — and the sentence is **false**, because a 15.0 pp difference at n = 40 (b=6, c=0,
  p = 0.031) *does* reach significance when the member is tested at 0.05. This is the identical
  falsity class as the `15.8` I withdrew in Pass 1: an attainable, significant outcome sitting below
  a printed floor that says it cannot exist.
- The MDD claims *"resolves ≥ X with 80% power."* True only at the **tightest step**, α/k, which is
  where it already is. Unchanged.

Measured counterfactual at n = 40, k = 2:

| case | p | Holm step | floor @ α/k = 17.5 | floor @ 0.05 = 15.0 |
|---|---|---|---|---|
| b=6, c=0, rank 2 | 0.031 | 0.05 | **demoted** (Holm gain lost) | not demoted — **correct** |
| b=6, c=0, rank 1 | 0.031 | 0.025 | not rejected anyway | not rejected anyway |
| b=7, c=0, rank 1 | 0.016 | 0.025 | consistent | consistent |

Floor-at-0.05 is also *conservative for every member*: it understates Y, and understating a bound of
the form "below Y nothing can happen" is the safe direction. And it restores the theorem — with the
floor at the loosest step, Rule 7 becomes unreachable on the McNemar path at **every** Holm step
(verified, 0 violations), which is what makes judgement 2's split coherent.

Cost of the build's resolution, stated so the trade is visible: it reduces Holm to Bonferroni for
any member whose difference lands in `[6/n, 7/n)` — the 15.0–17.5 pp band at n = 40 — which is
*precisely* the band §7.3 already priced as the cost of a second verdict metric. The current build
charges that price twice. The alternative the implementer rejected (printing two floors) is
correctly rejected; the fix is one floor, at the right α.

***What `-ml` should state.*** §7.1: the floor is computed at **α = 0.05, the loosest Holm step any
family member can face**, and the printed sentence names that (`"...cannot reach significance at any
observed outcome, at any Holm step (alpha <= 0.05)"`); the MDD stays at the family-adjusted α/k, and
the line names both. §3.3: add one sentence saying the two figures take different αs on purpose, and
why. §3.4: Rule 4's precondition 3 continues to fix `resolving.alpha == 0.05/len(family)` — that is
the *pre-registration* α and the MDD's — while `ResolvingPower` gains the floor computed at the
unadjusted α. And the generalisation is now worth stating **once** rather than as three special
cases: *every printed bound takes the rounding direction, the α, and the denominator that keep its
own claim true* — Pass 1's rounding adjudication, this α ruling, and the declined n-ML-1 are three
instances of one rule.

### New findings

**B-ML-2 (blocker) — the fail-safe basis degradation is anti-conservative, in the one corner
`sqrt(DEFF)` cannot reach.** At `design_effect == 1.0` with `basis == "assumed"` — the fail-safe
that plan §5 test 12b creates whenever a determinism probe has not run and agreed, i.e. **every
comparison until S2 lands the probe** — the decision moves off McNemar onto the bootstrap, and
`sqrt(1.0) = 1.0`, so nothing is widened. The substitute is then a bare percentile interval that
fires where the exact test does not. Measured at n = 40, `basis="assumed"`, DEFF = 1.0:

| b | c | McNemar p | exact test | bootstrap CI | bootstrap decision |
|---|---|---|---|---|---|
| 7 | 1 | 0.0703 | not distinguishable | [2.5, 27.5] | **distinguishable** |
| 9 | 2 | 0.0654 | not distinguishable | [2.5, 32.5] | **distinguishable** |
| 11 | 3 | 0.0574 | not distinguishable | [2.5, 37.5] | **distinguishable** |

Three inversions in nine spot-checked tables, all at p ≈ 0.057–0.070. Rule 7 does not catch them —
15.0, 17.5 and 20.0 pp are all at or above the 15.0 pp floor — and
`test_the_clustered_interval_is_not_narrower_than_the_mover_d_it_replaces` does not cover them,
because it parameterises DEFF over {2, 4, 7} only. So a degradation introduced to be *safe* makes
the tool declare differences the exact test refuses, on the path that is currently the default.

*Suggested fix, minimal and correct:* on any non-`by-construction` path make the decision a
**conjunction** — distinguishable iff the widened bootstrap CI excludes zero **and**
`mcnemar_exact(b, c) <= alpha_step`. Using McNemar as a *veto* rather than as the decision does not
violate Rule 4: the note's objection is that McNemar **rejects** too readily under clustering, and a
necessary condition only ever removes rejections, so the result is uniformly at least as
conservative as either instrument alone. At DEFF = 1 it restores the exact test's calibration
exactly; at DEFF > 1 it keeps the widened interval as the binding constraint it already is. The
structural version, later: make `"assumed"` require a declared inflation factor > 1, so the basis
that says "we did not measure this" cannot also assert DEFF = 1.

**M-ML-6 (major) — the printed observable floor is computed at α/k, which makes its own sentence
false under Holm.** Full argument and the recommended note wording in judgement 3 above. Touches
`stats.observable_floor`'s call site in `resolving_power`, `_floor_clause`, `resolving_power_line`,
Rule 7's comparison, and `test_rule_7_uses_the_family_adjusted_floor_the_report_prints` (whose
docstring is an honest statement of a trade that should not be taken).

**m-ML-6 (minor) — Rule 7 should raise on the McNemar path and demote only on the substitute
path.** See judgement 2. Sequence it after M-ML-6, since the McNemar branch is only unreachable once
the floor is at the loosest step.

### Dispositions — Pass 1

- **B-ML-1** — **fixed.** `paired_cluster_bootstrap` widens as specified; the CI that was identical
  at DEFF 2/4/7 is now [0.9, 32.7] / [−5.0, 40.0] / [−11.5, 48.1], and the DEFF = 7 headline case
  now renders *not distinguishable* with the interval covering zero. Rule 7 is enforced in
  `verdict()` on every path, against the exact float. Residual is B-ML-2, a different corner.
- **M-ML-1** — **fixed.** `UnattainablePower`, `mdd80: float | None`, and `unattainable_clause`
  replace both sentences; rendered at DEFF = 7 it now says "No difference is resolvable at
  n=5.71429 effective conversations … fewer than the b_min=6 net wins any outcome must reach", and
  no "100.0 pp with 80% power" survives anywhere.
- **M-ML-2** — **fixed.** `holm_steps` carries the step-down stop, `report.py` runs two passes
  because Holm is a family property, `verdict()` receives `alpha_step`/`holm_tested`, and the table
  gained a `decision` column so a threshold is never readable in isolation. The α gap this exposed
  is M-ML-6, not a regression.
- **M-ML-3** — **fixed.** `BinaryMetric.unit` is required with no default; a count whose denominator
  is not the analysis unit prints `— (n is turns; not the analysis unit)` with the §4.4 footnote,
  and the count itself is never suppressed. `_metric_from_dict` refuses to guess a unit.
- **M-ML-4** — **fixed.** `PairedRows` + `_pairing_tally` print the `asymmetry` split per arm,
  unscoreable-in-both, and present-in-one-only, always — including all zeros, which is the case that
  makes a full `n` legible as full.
- **m-ML-1** — **fixed** (implementation half). The docstring's "four orders tighter" is gone and
  replaced with the correct relation between the 4-dp publication and the 1e-9 mandate. The note
  half (publish the ten bounds at 10 dp, keep 1e-9) stays open for the note revision.
- **m-ML-2** — **superseded, confirmed.** The review text asks for a ceiling I withdrew under the
  tie-breaker; truncation is correct and is what shipped. Nobody should re-apply the Pass 1 text.
  The note-side change is the opposite of what m-ML-2 asked: 15.8 → 15.7 (n=38), 7.1 → 7.0 (n=85),
  46.7 → 46.6 (n=15, α=0.025); 58.3 and 23.3 stand.
- **m-ML-3** — **fixed.** `designEffect` and `basis` have no dataclass defaults; the legacy fallback
  lives only in `from_dict`, where it means "written before these fields existed", which is the
  right home for it.
- **m-ML-4** — **fixed.** `_BASIS_STRENGTH` takes the weaker of the two actual bases instead of
  collapsing to `"assumed"`; the decision rule is unchanged and the provenance is now true.
- **m-ML-5** — **fixed.** `generalization to unwritten {sample_noun}`.
- **n-ML-1** — **declined, and the decline is right — I accept it.** Their argument is better than
  my nit: the floor's conservative error is *smaller* and only the unfloored denominator produces
  it, while the MDD's is *larger* and only flooring produces it, so unifying them would make one of
  the two anti-conservative. It is the same principle as the rounding direction and the α ruling
  above, which is why it should be stated once in the note rather than three times.
- **n-ML-2** — open, note-side, unchanged.
- **n-ML-3** — **fixed.** The docstring now records that `_Z_95` is the 16-significant-digit decimal
  of the double rather than the double itself, and asserts the inequality explicitly.

### A correction to my own Pass 1 adjudication

My sweep claimed no `n <= 500` case makes naive truncation misfire. **That was wrong, and the reason
is narrower than "percentage points versus proportions".** I swept `math.floor(x * 1000)`; the code
computes `math.floor(x / precision)`. For `7/40` the two disagree: `x * 1000 == 175.0` exactly,
while `x / 0.001 == 174.99999999999997`, because the double nearest `0.001` is slightly above it and
the division rounds down across the bin edge. Multiplying by an exact power-of-ten-ish double
happened to round the other way, so my sweep tested an expression the code does not use.

Re-run against the code's own expression: over `n <= 1000` naive truncation misfires at exactly
**three** points, all at α = 0.025 — **n = 10, 20 and 40**. `n = 40` is both a published row of §7.1
(17.5 pp) and the `clear_suspend` slice size in §7.3, so the guard is load-bearing on a cell the note
prints, not defensive. `test_the_floor_truncation_is_guarded_against_the_bin_edge` pins it, and it
should stay pinned by that expression rather than by an equivalent-looking one.

---

## Pass 3 — 2026-09-03, commit `95b4c88`, against note **v1.7**

**Verdict: needs changes.** 1 major, 2 minor, 4 nits. **No blocker, and all three Pass 2 findings
are properly closed** — B-ML-2's conjunction, M-ML-6's α split and m-ML-6's path split are each
correct, each pinned by a test, and each survived a mutation I chose rather than the implementer.
The verdict turns on one thing: a **printed sentence that is false on 17% of the tables that print
it**, in the line whose only job is honesty. The decision is right everywhere it fires; the prose
beside it is not.

Fresh reviewer, nothing taken on faith. Suite re-run from `model-bench/`: **314 passed**, `ruff
check .` clean. Every number below was re-derived in this run — exact `Fraction` arithmetic for
McNemar and `b_min`, exhaustive verdict sweeps against the delivered module, and **26 mutations of
my own** (24 killed, 1 equivalent, 1 a real gap — m-ML-8). Every mutation was applied in place and
restored by rewriting the saved original; the working tree is as I found it. Note **v1.7** is the
specification I gated against and I revised it in this run (Part 1 below).

### Part 1 — the three note-side defects the implementer routed here

All three **upheld**, all three fixed in `docs/plans/small-model-benchmarking-ml.md` **v1.7**, plus
two I found while checking them.

1. **§7.2's rendered line was stale against §7.1's v1.6 template — agreed.** The implementer
   implemented §7.1 and said so in the test docstring; that is the right call and §7.2 is now on
   the template, with one added rule: **§7.1's template is authoritative wherever the two diverge
   again.** A worked example presenting itself as *"the string an implementer should test against"*
   is a second home for a mandated string, and this is the second time it drifted.
2. **Rule 4's three-α self-contradiction — agreed, and the two-field shape is what the note now
   requires.** Their reasoning is the note's own: `ResolvingPower` is built *and rendered* before
   the family is ranked, so a third field is `None` until it is not and the number acquires two
   homes. v1.7 says two fields plus `verdict()`'s parameter, explicitly. **I did not ask for the
   literal three-field shape.** While there I fixed the same defect one rule up: Rule 2's dataclass
   sketch still showed a single `alpha` field and `mdd80: float`, both superseded by shipped code —
   and I folded the shipped fifth precondition (`alpha_step ∈ [alpha_mdd, alpha_family]`) into Rule
   4 as precondition 5, since it is the theorem's premise and the note should own it, not inherit it.
3. **The α-split corner — agreed, and the arithmetic checks out.** Re-derived: at `k=2`,
   `n_units=13`, `DEFF=2`, `n_eff=6.5`, `b_min(0.025)=7` exceeds the 6 floored units so `mdd80` is
   `None`, while the floor at `alpha_family` is `6/6.5 = 92.3 pp` and **is attained** — `b=12, c=0`
   on 13 units is a 92.3 pp difference at `p = 2·2⁻¹² = 0.00049`, which clears a rank-2 member's
   0.05 Holm step. So the combined sentence would have been false, the split is correct, and §7.1
   now carries the worked corner rather than leaving it to be rediscovered.

Two more note-side items closed in v1.7: §3.2e verdict 2's closing clause (M-ML-7 below, the note
half) and **n-ML-2**, Pass 1's last open item — the measured divergence is `3.0167 × 10⁻⁴ pp`, so
the published *"at most 3.0 × 10⁻⁴ pp"* was below the value it bounds; it is now `3.1`.

### New findings

**M-ML-7 (major) — verdict 2's closing clause is printed unconditionally and is false whenever the
observed difference exceeds the strict-dominance MDD.** `stats._mdd_clause` always appends
`"; the observed X pp is below that."`, on **both** decision paths. That holds only when
`|diff| < mdd80`, and the case where it fails is not a corner — it is §7.1's own *normal case for a
model swap*, a candidate that wins more than it loses without strictly dominating. Rendered by the
delivered module at `n=20, b=13, c=5`:

> `Not distinguishable at this sample size. Observed difference +40.0 pp, 95% CI [-1.2, 69.0] pp
> covers zero (b=13, c=5, McNemar exact p=0.096). This pack resolves differences of >=36.7 pp with
> 80% power at n=20 effective items (20 units, design effect 1.00, by-construction, alpha=0.05);
> **the observed 40.0 pp is below that.** Neither model is ranked above the other.`

Exhaustive over the by-construction path at n ∈ {12, 15, 20, 30, 38, 40, 48} and every `(b, c)`:
**1,580 tables print the clause and 268 of them (17.0%) print it falsely**; at k=2 the same sweep
gives 112. Not a rounding artefact — the sentence contradicts two numbers inside itself, which is
the same falsity class as the `15.8` withdrawn in Pass 1 and the α/k floor withdrawn in Pass 2, and
the third time this review has ruled on it. The *decision* is correct and conservative in every one
of the 268 (they are all `not distinguishable`, because the discordance mix is not strictly
dominant — which is precisely the reason the two numbers point opposite ways).
*Fix:* note **v1.7 §3.2e** now mandates a conditional clause and gives the alternate wording
verbatim, including the discordance counts, which are what make the two numbers legible together.
Route to implementation with a test at `(n=20, b=13, c=5)` and `(n=30, b=5, c=13)`.

**m-ML-7 (minor) — `floor_clause`'s `> 1.0` boundary is load-bearing and untested.** Mutation
`> 1.0` → `>= 1.0` **survived all 314 tests**. It is not equivalent: at `n_eff` exactly `b_min`
(`n_eff = 6`, `alpha_family = 0.05`, floor `= 1.0`) the mutant prints *"no observed difference can
reach significance … the floor of 6 net wins exceeds the 6 effective units available"* — both
clauses false, since `b=6, c=0` on 6 units is a 100 pp difference at p = 0.031. The shipped `> 1.0`
is right; nothing pins it. *Fix:* one test at `n_eff == b_min` asserting the *"differences below
100.0 pp"* form, not the exceeds-the-units form.

**m-ML-8 (minor) — the MDD sentence stem has two homes, against this module's own stated rule.**
`stats.py:613` (`_mdd_clause`) and `report.py:184` (`resolving_power_line`) each spell
`"This pack resolves differences of >={…} pp with 80% power at "` in full. `provenance`,
`floor_clause` and `unattainable_clause` were all made public specifically so the report could not
carry a second copy — the docstring on `provenance` says *"two copies of this string is one copy and
one drift, and the copy is how `report.py` came to print `alpha=` from a field that no longer
exists"* — and this stem was left behind. Nothing asserts the two agree, and **M-ML-7's fix edits
exactly this string**, so the drift is scheduled rather than hypothetical. *Fix:* export an
`mdd_clause`-style helper and have `report.py` call it, as it already does for the other three.

### Nits

- **n-ML-4** — `holm_steps(p_values, *, alpha: float = 0.05)` restates the literal that
  `ALPHA_FAMILY` exists to be the single home of, in the same module whose constant docstring says
  *"a second literal `0.05` is how they drift apart"*. `report.py` passes
  `pack.metrics.alpha_family`, so the default is unreachable today — which is why it will rot
  unnoticed. Use `ALPHA_FAMILY`, or drop the default.
- **n-ML-5** — `resolving_power` accepts `design_effect < 1.0` (only `<= 0` is refused) although
  Rule 2's sketch says `>= 1.0` and both `verdict()` and `paired_cluster_bootstrap` enforce it.
  `DEFF = 0.5` *doubles* `n_eff` and shrinks **both** printed bounds; the refusal arrives a layer
  later. In a module whose stated shape is *"the anti-conservative version does not typecheck"*, the
  check belongs at construction.
- **n-ML-6** — `"80% power"` is hard-coded in three rendered strings (and in the field name `mdd80`)
  while `power` is a `resolving_power` parameter defaulting to `0.80`. Nobody passes another value;
  a caller who did would get a sentence that lies. Render `{power:.0%}`, or drop the parameter at S1.
- **n-ML-7** *(routes to `architect`, not to implementation)* — `docs/plans/small-model-benchmarking.md`
  §4's surface sketch still shows `holm_steps(..., alpha: float = 0.05)` and describes
  `resolving_power`'s `alpha` in the singular. The plan carries a statistics constant and a signature
  the `-ml` note owns; fold it in when the plan next moves.

### Dispositions — Pass 2

- **B-ML-2** — **fixed, and the veto is uniformly conservative.** A conjunction can only remove
  rejections, so no anti-conservative regime exists relative to either instrument alone; that is
  logic, not measurement. Measured at DEFF = 1 over n ∈ {12, 20, 40}, `b ≤ 18`, `c ≤ 12`: **40**
  tables where the bare percentile interval rejects and the exact test does not are now all vetoed
  (including Pass 2's `(7,1)`, `(9,2)`, `(11,3)`), and **zero** where McNemar rejects while the CI
  covers zero — so at DEFF = 1 the conjunction coincides with the exact test, as claimed. Consistent
  with Rule 4 as v1.7 states it: Rule 4 forbids McNemar deciding *for* distinguishability under
  clustering, and withholding is the opposite operation. Mutation `and` → `or` killed;
  `alpha = alpha_family` instead of the Holm step killed.
- **M-ML-6** — **fixed.** Floor at `alpha_family`, MDD at `alpha_mdd`, both keyword-only with no
  default; `PackMetrics.alpha` and `report.py`'s inline `0.05 / len(family)` are gone and
  `ALPHA_FAMILY` is the only literal. Mutations swapping either bound onto the other α, and
  `PackMetrics.alpha_mdd` returning `alpha_family` undivided, all killed.
- **m-ML-6** — **fixed, and the `mcnemar-exact` branch is genuinely a theorem.** Re-derived, not
  re-read: `p(b, c)` is non-decreasing in `c` at fixed `d = b − c`, so the minimum p at any given
  `|b − c|` is `p(d, 0) = 2·2⁻ᵈ`; hence `p ≤ α ⟹ d ≥ b_min(α)`. Confirmed by exact `Fraction`
  arithmetic over **every** `(b, c)` with `b + c ≤ 1200` (the implementer checked 400): zero
  violations at both αs. **The boundary, which is what matters:** at `|b−c| = b_min − 1` the
  *smallest attainable* p is **0.0625** against α = 0.05 and **0.03125** against α = 0.025 — a 25%
  margin, so the `1e-12` slack in precondition 5 cannot open the theorem, and on that path
  `n_effective == n_units == n` exactly (precondition 1 plus `design_effect == 1.0`), so the floor
  and the difference share a denominator. Equality (`|b−c| = b_min`) does not fire because the
  comparison is strict; mutating `<` to `<=` is killed.
- **The bin-edge guard, reclassified as defensive** — **correct, and I redid the sweep in the units
  the code uses.** With `math.floor(x / 0.001 + 1e-12)` against the unguarded form over `n ≤ 2000`:
  at `b_min = 6` the guard changes nothing, at `b_min = 7` it changes `n = 5, 10, 20, 40` — exactly
  what note §3.4 Rule 3a says. The same sweep in percentage points (`x*100/0.1`) reports `b_min = 6`
  misfiring at n = 125, 250, 500, 1000, 2000 and `b_min = 7` never — the opposite answer, and wrong.
  Keeping the guard on the "any future α reopens it" argument is right; its test is no longer a
  regression pin on a published cell and now says so.
- **n-ML-2** — **closed in v1.7** (note-side): `3.0 × 10⁻⁴` → `3.1 × 10⁻⁴`, measured `3.0167 × 10⁻⁴`
  on the `(34,6,0,0)` upper bound.

### Mutation testing — 26 run by me, 24 killed

The implementer reported 22 run / 21 killed; I did not use their list. Killed: the floor sharing the
MDD's floored denominator (the declined `n-ML-1` harmonisation); the MDD denominator ceiling instead
of flooring; the floor printer rounding to nearest; the veto as a disjunction; Rule 7 firing at
equality; Holm collapsing to Bonferroni; the `sqrt` dropped from the cluster widening; precondition
5 removed; the power region off by one; the floor sentence dropped when `mdd80` is `None`; each
bound computed at the other α; `alpha_mdd` undivided by k; DEFF as the width ratio; the Holm
step-down stop removed; the veto tested at the loosest α; the winner-orientation interval flip
dropped; the `by-construction` check dropped from `mcnemar_may_decide`; the one-sided p; the Wilson
continuity term; `report.py` taking the *smaller* design effect or the *stronger* basis;
`holm_tested` ignored; the duplicate-unit guard removed. Survived: `_percentile` truncating instead
of rounding (one index in 10 000 — equivalent in effect), and **m-ML-7**.

### Routing

- **M-ML-7**, **m-ML-7**, **m-ML-8**, **n-ML-4**, **n-ML-5**, **n-ML-6** → implementation
  (`tdd-engineer`), against note **v1.7**; M-ML-7's wording is in v1.7 §3.2e and needs no invention.
- **n-ML-7** → `architect`, in the plan's own next revision.
- Note-side work is **done in this pass** — `docs/plans/small-model-benchmarking-ml.md` is at v1.7
  and nothing in this review is waiting on it.

---

## Pass 4 — 2026-09-03, commit `d55f4d8`, against note **v1.7** (revised here to **v1.8**)

**Verdict: approve with suggestions.** 1 major, 4 minors, 2 nits. **No blocker, and every Pass 3
finding is properly closed** — each rechecked against the delivered module, not against the fix
round's report. The statistical core is sound: the three αs are separately load-bearing, Rule 3's
two rounding directions and two denominators are each pinned, Rule 7's split by path is enforced
and the theorem side genuinely raises, and B-ML-2's veto genuinely vetoes. **The residue is one
corner that has been narrowed at every pass and has now reached its last layer** — the
`cluster-bootstrap` substitute on the non-`by-construction` path. Pass 1 fixed what it *decided*
with (B-ML-1), Pass 2 fixed its *calibration* (B-ML-2's veto), and what is left is what it
*quantifies with*: the printed 95% CI. That is a follow-up, not a gate on S2 — see the judgement
at the end.

Fresh reviewer, nothing taken on faith. Suite re-run from a sandbox copy of `d55f4d8` outside the
repository: **353 passed**; the repository working tree was never modified. **39 mutations of my
own** (37 killed, 1 equivalent, 1 a real gap — m-ML-12); an exhaustive by-construction render sweep
over n ∈ {12, 15, 20, 30, 38, 40, 48, 60, 85, 90, 100, 120} × k ∈ {1, 2} × every `(b, c)`; and an
**exact** (not simulated) coverage computation for both paired-difference intervals — every paired
table enumerated against its multinomial probability, and every bootstrap resample outcome computed
analytically rather than sampled.

### The three items handed to me — all three decided, all three note-side

1. **The cluster-path decision sentence — adopted into §3.2e, with a correction (m-ML-10).** The
   implementer was right to ship a sentence rather than a false one, and right to route it. It is
   now published verbatim in v1.8 §3.2e(f), in **two variants keyed on the design effect**. The
   shipped text is not adopted unchanged: at `design_effect == 1.00` it closes with *"under
   clustering McNemar rejects too readily"* on a comparison that **declares no clustering**, which
   is P3-3's own defect surviving in the half of the sentence P3-3 did not touch — the *rationale*
   rather than the *widening*. The published variant 2 gives the reason that is true there (a
   design effect that was never established cannot license the exact test to carry a verdict) and
   re-attaches the `because` clause to the instrument choice, which the nearest-attachment parse of
   the shipped text puts on the widening instead.
2. **Pass 3's routed observation 1 — upheld, and it is worse than cosmetic (m-ML-11).** Resolved in
   v1.8 §7.1 with a published qualifier, printed only where `design_effect > 1.0`.
3. **The equality boundary — a defect, and the fix is one word, not a third wording (m-ML-9).** The
   *branch* choice is right, for the reason given. But **equality is reachable at this component's
   own sample sizes**, not only at the round's n = 90/100/120: measured this session, `n_units=85`,
   `k=2`, `DEFF=1.9` gives `mdd80 = 20.0 pp` and `|b−c| = 17` is exactly 20.0 pp — the guard-judge
   pack's own n. Sweeping `6 ≤ n_eff ≤ 200`, equality occurs at n = 90/100/120/150/180 (k=2) and
   n = 200 (k=1), and at n=90 it is **2.4%** (82 of 3 444) of every table that reaches the clause.
   v1.8 changes the published wording to **"is at or above that"** — one comparative true across
   the whole branch, rather than a third string to keep in step with the other two.

### New findings

**M-ML-8 (major) — the fail-safe path fixed the decision and left the *quantification* on the
narrower of the two available intervals.** `stats.verdict:792` prints
`paired_cluster_bootstrap(...)` as the 95% CI on every non-`by-construction` path; at
`design_effect == 1.0` — **every comparison until a determinism probe runs** — `sqrt(1.0)` widens
nothing, so §3.2c's mandated MOVER-D is replaced by a bare percentile interval. Three measured
consequences, in ascending order of seriousness:

1. **It is narrower, systematically.** Exact coverage, every table enumerated: at n=40 under strict
   dominance MOVER-D covers 0.976 at a mean width of 25.63 pp against the bootstrap's **0.939** at
   21.23 pp, and the bootstrap is narrower on **100%** of the probability mass (n=30: 0.983/30.60
   vs **0.942**/24.57; n=85: 0.969/15.32 vs **0.940**/13.56). The report rendered for the
   guard-judge shape prints `95% CI [3.5, 15.3] pp` where MOVER-D on the same table `(77,8,0,0)`
   gives `[3.1, 17.5]`. A fail-safe that fires because *less* is known must not tighten the
   interval.
2. **It degenerates at the sparse discordant counts §3.2b says this lab will see.** At n=30,
   `b=4, c=0` it returns `[3.3, 26.7] pp`, **excluding zero**, against McNemar's exact p = **0.125**
   and MOVER-D's `[−0.6, 29.7]`. The mechanism: with four non-zero rows
   `P(no +1 drawn) = (26/30)³⁰ = 1.4% < 2.5%`, so the 2.5th percentile *cannot* be zero. Under
   strict dominance at n=30 this fires on **20.3%** of the probability mass — each one rendering
   §3.2e verdict 3's *"the interval excludes zero but the exact paired test does not"* on the
   strength of a degenerate interval rather than a real disagreement. The veto keeps the *verdict*
   right in all of them; §3.4 Rule 6's claim that the floor catches what the widening cannot does
   **not** cover this, because the floor is about clusters and this is about non-zero rows.
3. **It is a Monte-Carlo estimate of an *atomic* quantile, so where the target percentile lands
   near an atom boundary the printed bound flips by a whole atom — with the seed *and* with the row
   order.** At the tool-caller pack's own n it is a coin flip, not a tail event:
   `paired_cluster_bootstrap([1.0]*5 + [-1.0]*3 + [0.0]*4, design_effect=1.0, B=10_000, seed=s)`
   prints a lower bound of `-25.0 pp` at `s=0` and `-33.3 pp` at `s=5` — **8.3 pp apart**, split
   **107/93** over 200 seeds and **102/97** over 200 row permutations at one fixed seed. At
   `(b=8, c=0, n=85)` the split is **184/16** over seeds (`[3.5, 16.5]` vs `[3.5, 15.3]`), and the
   exact CDF says why: `0.97281` at `13/85` against `0.98738` at `14/85`, so the 97.5% quantile
   clears the boundary by 0.0022 of probability against a Monte-Carlo standard error of 0.0016.
   Row order matters for the same reason the seed does, and it is not obvious: `Random.choice`
   draws an **index**, so a permutation of the same multiset re-maps a fixed index sequence onto
   different values. **Where the quantile is not near a boundary the bound is perfectly stable** —
   `(b=4, c=0, n=30)` is identical across 60 seeds — which is why this needs the mechanism stated
   rather than a flat "it moves with row order". P3-5 made the seed reproducible; the interval is
   still not a function of the table.

*The note authorised this and the note was wrong* — v1.6–v1.7 Rule 4 said "the strings rendered
against the bootstrap" without ever checking the width. **v1.8 replaces it with the conservative
envelope:** the wider of the `√DEFF`-widened bootstrap and the `√DEFF`-widened MOVER-D. It is
uniformly at least as conservative as either alone, reduces to MOVER-D exactly at DEFF 1.00,
stays responsive to a declared design effect, removes the degeneracy, and can only *remove*
rejections — so it disturbs neither the veto nor Rule 7. Second half of the fix: for a paired
**binary** table the resample distribution is exactly multinomial, so the percentile is a ~30-line
closed form (I verified mine against the shipped `B=10 000` output); resampling buys nothing and
costs the reproducibility in (3). `test_at_deff_one_the_substitute_interval_is_narrower_than_the_mover_d_it_replaces`
(`tests/test_stats.py:1066`) inverts under this fix, and its stated rationale — the conjunction,
not the interval, is what makes the path conservative — survives intact.

**m-ML-9 (minor) — verdict 2's alternate clause prints a strict comparative on a reachable
equality.** Item 3 above. `stats.py:658` is correct as a *branch*; `stats.py:661`'s wording is not.
Note v1.8 §3.2e publishes `is at or above that`; a test at `(n_units=85, k=2, DEFF=1.9, |b−c|=17)`
pins the case that already exists.

**m-ML-10 (minor) — the cluster-path label, and its rationale clause on the default path.** Item 1
above. `stats.py:926-941`. Note v1.8 §3.2e(f) publishes both variants verbatim.

**m-ML-11 (minor) — the floor sentence and the McNemar p beside it live on different denominators
above DEFF 1.** `stats.floor_clause` renders `b_min/n_eff` while the p three lines away is computed
over `n_units` raw rows, so the line reads as a flat self-contradiction. Measured: at
`n_units=12, DEFF=2` the floor prints **100.0 pp** while `b=11, c=0` is 91.7 pp at **p = 0.00098**;
at `n_units=40, DEFF=2`, 30.0 pp beside `b=6, c=0` at p = 0.031. Both are *decided* correctly (Rule
7 demotes), and the resolution — McNemar over raw rows is anti-conservative under clustering, so
the floor over effective units governs — appears nowhere in the rendered text. Note v1.8 §7.1
publishes the qualifier, conditional on `design_effect > 1.0`; §7.2's rendered line is DEFF 1.00
and does not move.

**m-ML-12 (minor) — the veto's α is untested on the path that always takes it.** Mutation: on the
cluster branch, test the veto at `resolving.alpha_family` instead of the Holm step `alpha`
(`stats.py:813`). **Survives all 353 tests.** It is not equivalent — at `n=40, k=2, DEFF=1.0,
basis="assumed"`, `(b=6, c=0)`, `p=0.031`: at `alpha_step=0.025` the shipped code returns *not
distinguishable*, at `alpha_step=0.05` it returns *distinguishable*. So the multiplicity correction
can be removed from the substitute path without a test noticing, on the path every comparison
currently takes. Pass 3's list reports "the veto tested at the loosest α" as killed; whatever that
mutation was, it was not this one. *Fix:* one test at that table asserting both steps.

### Nits

- **n-ML-8** — Rule 7's `mcnemar-exact` **raise** is suppressed by `holm_tested=False`
  (`stats.py:835`, `fired = raw_significant and holm_tested and below_floor`). The theorem is
  `p ≤ alpha_step ≤ alpha_family ⟹ |b−c| ≥ b_min`; it says nothing about whether Holm tested this
  member, so a genuine module bug goes undetected for every member past the stop. Verified: with an
  inconsistent floor, `holm_tested=True` raises `Rule7Violation` and `holm_tested=False` returns
  silently. Split the two conditions — raise on `raw_significant and below_floor`, keep
  `holm_tested` in the demotion condition, where it belongs (nothing visible changes: the
  `not holm_tested` text branch already precedes the `floor_demoted` one).
- **n-ML-9** *(routes to `architect`)* — `docs/plans/small-model-benchmarking.md` §3.8.5 says
  *"§2.1's inventory row above is the only place this plan repeats one"* of the note's κ figures,
  and §6 R-9 then repeats `κ = 0.21` a second time. Both are attributed, so the drift risk is small
  — but v1.7's withdrawal pass is one line short of its own claim. Otherwise the withdrawal holds:
  the plan carries **no** note-owned literal for `z`, the verdict strings, the floors, the MDDs or
  `b_min` (grepped).

### Dispositions — Pass 3

- **M-ML-7** — **fixed, and the sweep is clean.** Re-ran the render sweep that found it: over the
  by-construction path at n ∈ {12,15,20,30,38,40,48,60,85,90,100,120} × k ∈ {1,2} × every `(b,c)`,
  the *"is below that"* clause is now printed **0 times falsely**. The only residual falsity in that
  sentence is the equality wording in the *other* branch — m-ML-9, a different string.
- **m-ML-7** — **fixed and pinned.** At `n_eff == b_min` the *"differences below 100.0 pp"* form
  prints, not the exceeds-the-units form; mutating `> 1.0` → `>= 1.0` is now **killed**.
- **m-ML-8** — **fixed and pinned.** `report.py:191` calls `stats.mdd_clause`; editing the stem in
  `stats.py` alone now fails a report test, which is the scheduled drift the finding predicted.
- **n-ML-4** — **fixed.** `holm_steps(p_values, *, alpha: float)`, no default; `ALPHA_FAMILY` is the
  only `0.05` literal in the module.
- **n-ML-5** — **fixed, and the boundary and the error type are both right.** Verified:
  `resolving_power` accepts 1.0 and refuses 0.0 / 0.5 / 0.999 / −1.0 / **NaN** (the guard is written
  `not design_effect >= 1.0`, which is the NaN-safe form) with `ValueError` — the module's
  convention for a caller-supplied value out of domain, correctly distinct from `Rule7Violation`'s
  `AssertionError`, which is reserved for module bugs. `DEFF = inf` is still refused, one layer
  down and with a message naming `n_effective`; not worth a change.
- **n-ML-6** — **fixed.** `power` is a `ResolvingPower` field; `power=0.90` renders *"resolves
  differences of >=22.0 pp with 90% power"* in all three strings.
- **n-ML-7** — **fixed** (plan v1.7 §4 S1 carries the signature with no α default).

### Mutation testing — 39 run by me, 37 killed

I did not use the fix round's list. **Killed:** each printed bound computed at the other α (floor at
`alpha_mdd`, MDD at `alpha_family`); `provenance` and `floor_clause` each naming the other's α;
Holm run at `alpha_mdd`; `PackMetrics.alpha_mdd` undivided; `verdict`'s α default moved to
`alpha_family`; precondition 3 disabled; `unattainable_clause`'s `b_min` at the wrong α; the MDD
rounding to nearest and flooring; the floor printer ceiling-rounding, dropping its bin-edge guard,
and switching to the `x*1000` form; the floor sharing the MDD's floored denominator;
`_require_effective` ceiling instead of flooring; the Rule 7 raise removed, its comparison made
non-strict, disabled, and pointed at the *printed* floor; the veto dropped, disjoined, and applied
before the floor; `mcnemar_may_decide` losing its basis check; the MDD clause's conditional made
non-strict and unconditional; the widening clause firing at DEFF 1.00; `sqrt(DEFF)` → `DEFF`; the
power region off by one; the one-sided p; Holm's step-down stop; `report.py` taking the smaller
design effect or the stronger basis; `holm_tested` dropped from the decision; `floor_clause`'s
`> 1.0` boundary; the MDD stem re-duplicated in `report.py`. **Survived:** `b_min`'s loop condition
`>` → `>=` (equivalent — `mcnemar_exact(b,0) = 2^(1−b)` is never exactly `0.05/k` for k ≤ 10; the
engineering gate reached the same conclusion as its E8), and **m-ML-12**.

I also re-ran the units trap before touching any numeric claim: `format_floor_pp` computes in
proportions at `precision=0.001`, and both the `x*1000` substitution and the dropped `+1e-12` guard
are killed by the suite, so the expression is pinned by the expression the code uses.

**One claim of mine sharpened on challenge, before this pass was accepted.** M-ML-8(3) first read
*"its last digit moves by one atom with row order at a fixed seed"*, which is true on the table I
measured and **not reproducible on an arbitrary one** — the coordinator could not reproduce it at
`(b=4, c=0, n=30)`, and was right not to be able to: that table's quantiles sit far from an atom
boundary and are identical across 60 seeds. The finding survives with its mechanism stated (above)
rather than as a flat behavioural claim, and its strongest case moved from n=85 (a ~10% flip) to
n=12 (a ~50% flip, 8.3 pp wide). Two details worth keeping out of the next re-derivation: the
effect needs **`B = 10 000`-scale sampling error compared against the exact CDF's distance to the
nearest atom**, not a shuffle count — six shuffles on a 13%-rate table is under one expected hit —
and a *reduced* `B` makes it **more** common, not less (`(b=4, c=2, n=30)` is stable across 60
seeds at `B=10 000` and moves on 8 of 60 at `B=2 000`).

### The judgement asked for: does this residue block S2?

**No. It rides as follow-ups, and I would build S2 on this core.** The reasoning, so the stakeholder
can weigh it rather than take it:

- **Nothing in Pass 4 touches a seam S2 builds against.** Every finding lives inside two functions
  of `stats.py` (`verdict`'s interval selection and its Rule 7 condition) plus four published
  strings. No type, no signature, no stored-record shape, no pack-contract field, and no
  `report.py` structure moves. S2's adapter, runner and scorers consume `ItemResult.scored_outcome`,
  `BinaryMetric.unit`, `PackRef` and `RunResult` — none of which is implicated.
- **Three fix rounds with Pass 3 finding more than Pass 2 is not a decaying core.** It is a
  changing *technique*: passes 1–2 read formulas, pass 3 started rendering output and reading the
  English, and that is where this component's defects have always been. By severity the trend is
  monotone — Pass 1: 1 blocker + 4 majors; Pass 2: 1 blocker + 1 major; Pass 3: 1 major; Pass 4:
  1 major, and that one is the last layer of a corner narrowed at each of the three previous passes
  (decision → calibration → quantification).
- **The one thing I would not defer past S2 is M-ML-8's landing *before the first stakeholder-facing
  comparison is published*.** It changes printed intervals, and changing them after reports exist
  means two reports over the same data disagree — which is the one failure a tool whose value claim
  is "it refuses to report a number it cannot stand behind" cannot absorb. S2 produces no published
  comparison, so the natural slot is the S2 round or immediately after it, on note v1.8.

### Routing

- **M-ML-8**, **m-ML-9**, **m-ML-10**, **m-ML-11**, **m-ML-12**, **n-ML-8** → implementation
  (`tdd-engineer`), against note **v1.8**; every string M-ML-8, m-ML-9, m-ML-10 and m-ML-11 need is
  published verbatim in v1.8 §3.2e, §3.4 Rule 4 and §7.1, and none needs inventing.
- **n-ML-9** → `architect`, in the plan's own next revision.
- Note-side work is **done in this pass** — `docs/plans/small-model-benchmarking-ml.md` is at
  **v1.8** and nothing in this review is waiting on it.
