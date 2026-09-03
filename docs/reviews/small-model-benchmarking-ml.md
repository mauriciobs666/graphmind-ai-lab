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
