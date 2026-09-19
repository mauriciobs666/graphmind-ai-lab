# Small-Model Catalog Sweep — Methodology Review

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — (M9)

## 1. Scope & verdict

Reviewed `model-bench/docs/plans/small-model-catalog-sweep.md` (Track A, FR-6–FR-12) against
`model-bench/docs/requirements/small-model-catalog-sweep.md`, this component's statistics contract
(`docs/plans/small-model-benchmarking-ml.md`, `-ml`), `model-bench/AGENTS.md`'s load-bearing
invariants, and the actual source: `modelbench/stats.py`, `modelbench/report.py`,
`modelbench/packs.py`, `modelbench/results.py`, `modelbench/runner.py`, `modelbench/cli.py`, and all
five `packs/*/pack.json` manifests plus their scorers (`modelbench/scoring/*.py`). This closes the
plan's own §3.4 Q1–Q4 and the two "also independently check" items. It does not repeat
`analyst`'s parallel general-design review (`docs/reviews/small-model-catalog-sweep.md`) except
where its finding 2.1 is squarely mine to resolve — see §2.

**Verdict: needs changes.** One blocker (§2, corroborating `analyst`'s independent 2.1) means FR-8's
family cannot render at the sweep's own scale without a small, scoped `stats.py` signature change.
Q1 (§3) needs a new, correctly-scoped function rather than the plan's literal "reuse
`paired_bootstrap` unchanged." Q3 (§5) needs one combined family, not two independent ladders — the
plan's stated default under-corrects. Q4 (§6) needs the plan's k corrected for guard-judge and its
"top-ranked model" framing dropped. Q2 (§4) is correct as scoped, with one latent-risk note. FR-11's
caveat table (§7) is faithful to `-ml` on spot-check. None of this blocks Units B/C, which are sound
as designed once Unit A's contract is fixed.

## 2. Cross-cutting blocker — `stats.verdict()`'s own precondition rejects the candidate-axis Holm ladder FR-8 needs

`analyst`'s review independently found this (its 2.1) from the code-correctness angle and correctly
routed the resolution to me rather than picking one of its two sketched options. I verified the same
three preconditions independently (`modelbench/stats.py:1185-1219`) before reading `analyst`'s
finding, so this is corroborated twice over, not asserted once.

**The mechanism, restated precisely.** `verdict()`'s Rule 4 preconditions 3 and 5 are:

```python
if abs(resolving.alpha_mdd - resolving.alpha_family / len(family)) > 1e-12:
    raise ValueError(...)
if alpha_step is not None and not (
    resolving.alpha_mdd - 1e-12 <= alpha_step <= resolving.alpha_family + 1e-12
):
    raise ValueError(...)
```

`family` is `Sequence[str]` of **metric names** — the one existing call site
(`report.py:1236-1294`) passes `family=pack.metrics.verdictMetrics` (length 1 or 2 for every shipped
pack) and builds `resolving` with `alpha_mdd=pack.metrics.alpha_mdd == alpha_family/len(family)` by
construction (`packs.py:88-95`). The Holm ladder at that site runs **over `len(family)` p-values**
(one per metric, for one fixed A-vs-B pair), so `step.threshold` is guaranteed to land in
`[alpha_mdd, alpha_family]` — the ladder's own multiplicity and `family`'s length are the same
number by design. That is not a coincidence the plan can lean on for a different axis: it is the
entire reason precondition 3 exists (Rule 4, `-ml` §3.4 — "a k-member family cannot report its MDD
at α=0.05 by oversight").

FR-8 introduces a **second, independent multiplicity axis** — N−1 candidates against one reference —
that has nothing to do with `len(pack.metrics.verdictMetrics)`. Concretely, at this sweep's own scale
(17 chat/vlm models, N−1=16): Holm's tightest candidate-ladder step is `0.05/16 = 0.003125`.

- If `resolving.alpha_mdd` is left at the pack's own metric-level value (0.05 for a k=1 pack, 0.025
  for guard-judge) and `alpha_step=0.003125` is passed, precondition 5 fails immediately —
  `0.05 - 1e-12 <= 0.003125` is false.
- If instead `resolving.alpha_mdd` is rescaled to the candidate axis (`alpha_family/16`, matching the
  plan's own Q4 recommendation) while `family` stays `pack.metrics.verdictMetrics` (length 1 or 2),
  precondition 3 fails instead — `0.003125 != 0.05/len(family)` for every shipped pack.

**Either construction raises on the first candidate tested under Holm, for every one of the four
FR-8-eligible packs, at this sweep's real N.** This is not a corner case reachable only on
pathological input; it is the ordinary path through the sweep's own stated scale, so it is a blocker
for Unit A as literally scoped in plan §3.2, not a risk to flag and defer.

**Resolution, decided (not left to `architect`'s pick between `analyst`'s two sketched options).**
`analyst`'s option (b) — reimplement Rule 7's floor check and the winner label directly from
`PairedOutcomes`/`mcnemar_exact`/`resolving_power.observable_floor`, bypassing `verdict()` — creates
a **second copy of Rule 7's floor enforcement**, which is exactly the failure mode this module's own
convention refuses by name ("two copies of a formula is one copy and one bug," `AGENTS.md`, and Rule
7's own docstring: "no path returns `distinguishable` below `resolving.observable_floor`" — a claim
that stops being checkable-by-construction the moment a second enforcement site exists). Reject it.

**Take option (a): extend `verdict()`'s signature with one new, keyword-only, optional parameter
that decouples the correction divisor from `family`'s membership list, without touching what
`family` is *for*.**

```python
def verdict(
    outcomes: PairedOutcomes,
    *,
    resolving: ResolvingPower,
    metric_name: str,
    family: Sequence[str],
    correction_k: int | None = None,   # NEW — defaults to len(family), unchanged behavior today
    a_label: str = "A",
    b_label: str = "B",
    alpha_step: float | None = None,
    holm_tested: bool = True,
) -> Verdict: ...
```

Inside, precondition 3 becomes `k = correction_k if correction_k is not None else len(family)`,
then the existing check against `resolving.alpha_mdd == resolving.alpha_family / k`. Everything else
is untouched: `metric_name in family` stays exactly as it is and keeps doing exactly one job — a
candidate-axis call still must name a metric that is genuinely one of `pack.metrics.verdictMetrics`,
so a caller cannot smuggle a non-pre-registered metric into a family verdict by way of this change.
Rule 7's floor check and every other precondition stay in the one function that owns them.

- **Why a default rather than a required keyword** (this module's own "nothing that shapes a
  decision carries a default" convention, `AGENTS.md`, does not forbid this one): the existing
  homogeneous-binary call site's behavior is already audited and correct at
  `correction_k = len(family)`; a default that reproduces exactly today's call, unchanged, is not
  the anti-conservative-by-omission shape that convention exists to block (`design_effect`'s default
  would silently *reintroduce* a retired bug; `correction_k`'s default reproduces the one behavior
  already shipped and tested). FR-8's own call site will always pass it explicitly (N−1 or, per §5's
  resolution below, `2·(N−1)`), so the default is never silently relied on where it would be wrong.
- **This is the minimal, single-owner fix.** It adds four lines to one function, touches no other
  precondition, and does not require `-ml`'s Rule 4 text itself to change — precondition 3's own
  *statement* ("a k-member family cannot report its MDD at α=0.05 by oversight") is unaffected; only
  the *source* of k becomes an explicit parameter instead of an inferred `len(family)`, for the one
  caller that has a second, legitimate k to declare.
- **Scope check — does `continuous_verdict()` need the same fix?** No, not for this plan. FR-8
  restricts the reference-anchored family to "chat/vlm models" (four binary-metric packs); the
  embedder pack (the only continuous-verdict pack) has 2 in-scope models total, so N−1=1 and no
  family is meaningful there regardless. Flagging this so the premise doesn't go stale the way `-ml`
  §3.4 itself warns about: **if a future pack ever needs FR-8's family on a continuous headline
  metric with >2 models, `continuous_verdict()`'s identical `family`/`alpha_family`-coupling
  (`stats.py`'s Rule 8 signature) has the same defect and needs the same `correction_k` parameter.**
  Out of scope today, in scope the day that census changes.
- **Test obligation.** Per this module's guard-constant convention, `correction_k` needs the same
  shrink/widen discipline as `_LOWER_IS_BETTER`: a test that a candidate-axis `alpha_step` tighter
  than `pack.metrics.alpha_mdd` is accepted when `correction_k` is passed and still raises when it
  is omitted (mutating the call, not the constant, since this is a parameter rather than a module
  constant — the equivalent obligation is "assert both the passing and the omitted path", not a
  literal shrink/widen of a frozenset).

This is the one item in this review that must land before Unit A's FR-8 code is written — everything
else below is a correctness/framing fix within a buildable design.

## 3. Q1 — standalone CI for a continuous metric (`mrr`)

**Not valid as literally proposed.** "Reuse `stats.paired_bootstrap` unchanged, called on one arm's
own per-item `mrr` values (not a difference)" is mechanically callable (the function only
resamples-and-means whatever sequence it is given) but silently inherits machinery built for a
**different estimand**, and that mismatch is not cosmetic:

- **The support/clamp conversion is wrong for a one-sample mean.** `-ml` §3.4 Rule 4/Rule 8 derives
  the clamp as *the support of the difference*, `(lo − hi, hi − lo)` — e.g. `(−1.0, 1.0)` for MRR's
  `(0.0, 1.0)` support — because every existing caller of this machinery bootstraps a **difference**
  of two arms' values. A one-sample mean of values in `[0, 1]` needs its interval clamped directly to
  `(0.0, 1.0)`, never to `(−1.0, 1.0)`. Passing `support=(0.0, 1.0)` into machinery that internally
  converts it to a difference-support is the wrong number reaching the clamp, silently, for exactly
  the reason `-ml` §3.4 Rule 8 spent a whole paragraph on: *"a metric's support is a fact about the
  metric, known to the party that defined it, so it is declared and forwarded, never inferred"* — an
  inference the other direction (declared-for-a-difference, forwarded-to-a-raw-value) is the same
  category of error with the sign reversed.
- **It is a different decision object, not a relabeled one.** The paired path's whole apparatus
  (percentile levels derived from `family`/`k`, the "excludes zero" decision, `ContinuousVerdict`'s
  `distinguishable` field) exists because the interval answers *"does this difference exclude zero"*.
  A single arm's own mean has no zero to exclude and is never a verdict — it is **descriptive**,
  parallel to Wilson's treatment of a binary rate (`-ml` §3.2(a): "explicitly not the comparison
  instrument"). Reusing a verdict-shaped function for a descriptive one is the wrong function family,
  independent of the clamp bug.
- **Monte Carlo, not closed form, is still correct here** — MRR's per-item values take far more than
  the `v = 3` values `-ml` §3.4 Rule 4's exact enumeration requires (that rule is scoped to the paired
  **binary** table specifically, "the boundary of applicability is three values... and nothing else").
  A seeded percentile bootstrap at `B = 10 000` is the right instrument for this too, unchanged from
  what `-ml` §3.2(d) already prescribes for the paired continuous case.

**What to build instead — a new, small, correctly-scoped function**, sharing the resample *engine*
(draw-with-replacement, seeded, percentile-of-resample-means) but not the difference-support
semantics:

```python
def mean_bootstrap_interval(
    values: Sequence[float], *, B: int, seed: int,
    levels: tuple[Fraction, Fraction], support: tuple[float, float] | None,
) -> tuple[float, float]:
    """Percentile bootstrap CI on a single arm's own mean — never a comparison instrument.
    `support`, when given, clamps the RESULT directly (the metric's own bounds), never converted
    to a difference-support. Refuses non-finite values and len(values) < 2, mirroring Rule 8."""
```

Factor the shared resample loop out of `paired_bootstrap` into a private helper both functions call,
rather than literally calling `paired_bootstrap` with a relabeled argument — this is the "two copies
of a formula is one copy and one bug" rule cutting the other way: the *engine* is one copy either
way, but the *clamp semantics* are legitimately two different things and must not be forced to share
a call site.

**Practical note: this is cheap to get right, not merely cheap to build.**
`ContinuousMetric.support` (`results.py:120-123`) already carries the metric's own bounds — `(0.0,
1.0)` for MRR, stored on the aggregate today — so the ranked table's per-model interval can pass
that value straight through as `mean_bootstrap_interval`'s `support`, with no new judgment call about
what the bound should be. `seed=pack.seed` (reused, per the plan) is fine — a fresh resample over a
different input needs no different seed discipline than the paired path already has.

**Labelling — yes, it needs its own caveat, and it must say more than "descriptive."** Parallel to
`_DESCRIPTIVE_NOTE`, but this one carries an extra warning Wilson's descriptive note does not need:
*"This interval describes this model's own mean; it is not a comparison, and two such intervals
overlapping or not overlapping is not itself a valid basis for a verdict (`-ml` §3.1's marginal-
overlap inertness applies here too) — see FR-8's optional reference-anchored family for an actual
test, when one was run."* Without the second sentence, a reader trained on this report's other
descriptive Wilson cells will do exactly the eyeball comparison FR-7's resolving-power sentence (§6
below) exists to head off.

## 4. Q2 — is `_LOWER_IS_BETTER` complete and correctly scoped?

**Complete for what is actually rankable today.** Checked every pack's `verdictMetrics`/
`headlineMetric` (`packs/*/pack.json`) against its scorer's semantics
(`modelbench/scoring/{retrieval,extraction,toolcalls,grounding}.py`):

| Pack | verdict/headline metric(s) | Polarity | Scorer confirms |
|---|---|---|---|
| embedder | `mrr` | higher-is-better | `retrieval.py:487`, standard reciprocal rank |
| guard-judge | `falseAdvanceRate`, `falseSuspendRate` | **lower-is-better** | `classification.py:209-210,259` — count=1 on the undesired-event flip |
| nlq-generator | `layer1ExactMatchRate` | higher-is-better | `extraction.py:339-340`, standard exact-match count |
| tool-caller | `cleanThroughTurnH` | higher-is-better | `toolcalls.py:961-966`, `outcome="pass"` on clean-through |
| chat-responder | `groundingRate` | higher-is-better | `grounding.py:287-288`, standard checklist-pass count |

No other pack's `verdictMetrics`/`headlineMetric` shares guard-judge's undesired-event polarity, so
`_LOWER_IS_BETTER = frozenset({"falseAdvanceRate", "falseSuspendRate"})` is correct and complete
against every pack.json shipped today.

**One latent-risk note, not a defect to fix now.** `classification.py`'s `_METRIC_BY_TIER` also
defines `falseAdvanceRateBoundary` (the `boundary` tier, n=15) with **the identical undesired-event
polarity** (`advanced=True` is a false-advance event there too) — it is exploratory today
(`-ml` §7.3: "not a verdict metric," never in `guard-judge`'s `verdictMetrics`), so it is out of
`rank_report`'s ranking scope and `_LOWER_IS_BETTER`'s omission of it is correct *as scoped*. But it
is exactly the case Q2's own wording anticipates ("any exploratory one that might later become a
verdict metric") — if `falseAdvanceRateBoundary` is ever promoted into `verdictMetrics`,
`_LOWER_IS_BETTER` must be extended in the same commit, and nothing today makes that dependency
discoverable from the constant alone. **Recommend one comment at `_METRIC_BY_TIER`'s declaration
site** (not a code change, a docstring cross-reference): *"if this tier's metric is ever promoted to
a pack's `verdictMetrics`, `report._LOWER_IS_BETTER` must gain it in the same change."* Cheap,
and it is precisely the kind of cross-file coupling this component's own `AGENTS.md` asks to be
named rather than left implicit.

**On scoping the fix to the new code rather than `PackMetrics` itself (the plan's own open
question):** agree with the plan's default. `PackMetrics` is a fact about a pack's pre-registration
(what gets a verdict and at what k); polarity is a fact about how a metric's raw rate reads, which is
orthogonal and, per §2.4's own finding, already latent in `stats.verdict()`'s shared text — fixing it
there is a separate, larger change this plan correctly declines to fold in (its own recommended
`docs/BACKLOG.md` entry is the right size for that).

## 5. Q3 — one combined family, not two independent per-metric ladders

**The plan's default (two independent per-metric Holm ladders, one over the N−1 candidates for
`falseAdvanceRate`, one over the N−1 candidates for `falseSuspendRate`) under-corrects, and this is
the same principle `-ml` §3.3 already ruled mandatory one axis over — extended, not reinterpreted.**

`-ml` §3.3 is unambiguous about *why* guard-judge's two metrics must share one Holm ladder for a
single A-vs-B comparison: *"Two co-equal verdicts at α=0.05 each carry a ~9.75% chance of at least
one false 'better' under the null — which would hand the stakeholder exactly the fishing artefact
pre-registration exists to prevent."* That argument is about the **compound risk of reporting
multiple co-equal tests in one exercise**, not about metrics specifically — it applies with exactly
the same force, and more so given the larger test count, to FR-8's candidate axis. Two independent
16-test ladders (one per metric), each individually controlled at family-wise α=0.05, bound the
**combined** false-positive risk across all 32 (candidate × metric) tests only by the union bound —
approaching `1 − (1 − 0.05)² ≈ 9.75%`, the exact number `-ml` cites as the reason correction is
"mandatory, not optional" for guard-judge's two metrics in the first place. Running the plan's
default reintroduces, at a larger scale, precisely the risk the pack's own pre-registration was
built to close.

**FR-8's own wording supports the combined reading, not the plan's split one.** FR-8 asks for "**one**
pre-registered, reference-anchored family per pack" (singular). For a pack with a single headline
metric that is naturally one ladder of N−1. For guard-judge — the only no-headline, two-metric pack —
"one family per pack" is satisfied only by **one combined ladder over both metrics' candidate
comparisons together**, size `2·(N−1)`; "two families, each called *the* family for its own metric"
does not satisfy the plan's own chosen wording, let alone the statistical argument above.

**Recommendation: one combined candidate×metric family for guard-judge, size `k = 2·(N−1)` (32 at
this sweep's N=17), Holm-corrected jointly.** Concretely:

```python
p_values: list[float] = []          # length 2*(N-1) for guard-judge, N-1 for every other pack
cells: list[tuple[str, str]] = []   # (candidate_key, metric_name), same order as p_values
for metric in family_members:       # 1 member for four packs, 2 for guard-judge
    for candidate in candidates:    # N-1 models, excluding the reference
        ... build PairedOutcomes, append mcnemar_exact(b, c) ...
steps = stats.holm_steps(p_values, alpha=pack.metrics.alpha_family)  # ONE call, unchanged primitive
```

then render one table per metric by filtering `cells`/`steps` back apart — **no new arithmetic**:
`holm_steps` is already fully generic over `p_values` (`k = len(p_values)` internally, no notion of
which axis contributed which entry — verified by reading its body, `stats.py:1447-1486`), so this is
strictly **simpler** than the plan's two-call design, not more code. This is also the resolution that
determines `correction_k` for §2's fix: `correction_k = len(p_values)` — 16 for a single-metric pack's
family, 32 for guard-judge's combined one — passed once per `verdict()` call, matching whichever
`p_values` list that candidate's `alpha_step` was drawn from.

**Validity check for combining across metrics that score disjoint item subsets.** Holm–Bonferroni
controls the family-wise error rate under **arbitrary dependence** among the p-values in the ladder —
unlike some multiple-comparison procedures, it needs no independence or positive-dependence
assumption to hold (this is the same property `-ml` implicitly relies on wherever it Holm-corrects
guard-judge's own two metrics, which are scored over disjoint item slices too — 40 `clear_suspend`
items for `falseAdvanceRate`, 30 `clear_advance` for `falseSuspendRate`, and Holm was still ruled
mandatory there). Combining the metric axis and the candidate axis into one ladder inherits that same
guarantee; there is no dependence-structure objection to raise.

**Cost, honestly.** This is a real power cost, not a free correctness fix: guard-judge's tightest
step moves from `α/16 = 0.003125` (plan's default, per-metric) to `α/32 = 0.0015625` (this
recommendation). That is the price FR-8's own "one family per pack" design already commits to for
the pack that has two co-equal metrics — it is not a new cost this review introduces, it is the cost
the plan's default was silently avoiding by treating the two metrics' ladders as separable when
FR-8's own multiplicity argument says they are not.

## 6. Q4 — the resolving-power sentence when the ranked table has no `--reference` at all

**Yes, a coherent statement exists, but the plan's mechanics need two corrections: the implied k must
match §5's resolution, and the narrative must not name a specific anchor model.**

**k must be pack-specific, following §5.** The plan's Q4 recommendation computes a single generic
`alpha_mdd = alpha_family/(N-1)` regardless of pack. That is right for the four single-headline-metric
packs but **wrong for guard-judge**: since §5 rules that FR-8's actual family (if run) would be one
combined ladder of size `2·(N-1)`, the honest "if the maximal family this design permits were run"
statement for guard-judge must use `k = 2·(N-1)` (α_mdd = `alpha_family/32` at this sweep's N),
not `N-1`. Printing the single-metric-pack number for guard-judge would understate exactly the cost
§5 just established — the two answers are not independent and the plan's Q4 text, written before Q3
was resolved, does not yet make this connection.

**Drop the "top-ranked model" anchor from the narrative; it names a choice that was not made.**
`resolving_power`'s number is anchor-agnostic — `k` only counts *how many* comparisons a maximal
family would contain, not *which* model is excluded as the reference, so naming "the top-ranked
model" adds nothing the number needs and costs something the sentence shouldn't pay: the top-ranked
model is a **fact about this run's results**, so naming it as the hypothetical anchor reads as if a
comparison against that specific, empirically-best-performing model was pre-registered — the same
choose-after-you-see-the-data risk `-ml`'s whole `verdictMetrics` pre-registration discipline exists
to foreclose, imported into prose rather than arithmetic. Prefer a model-agnostic hypothetical:
*"if this pack's optional reference-anchored family (FR-8) were run — any one of the {N} models here
designated as the reference, the other {N-1} compared against it{, jointly across both verdict
metrics, for guard-judge} — that family of {k} tests would resolve differences of >= X pp with 80%
power (α_mdd = {alpha_family}/{k})."* No model is named; the number is unaffected either way.

**Print it alongside, not instead of, the pack's own already-published single-comparison resolving
power.** `-ml` §7.1–§7.4 already state each pack's resolving power for **one** reference-anchored
comparison, at the pack's own pre-registered `k` (`PackMetrics.k`, 1 or 2) — a number that does not
depend on how many models happen to be in this particular report. That sentence stays true and
useful regardless of N and should still be printed (it is not superseded by the family-size
sentence — it answers "what could a single comparison show," the family sentence answers "what would
scanning every model against one reference cost"). Printing only the N-scaled number without this
baseline loses the fact that the degradation is a property of *scale*, not of the instrument itself.

**Do reuse `resolving_power_line`'s clause structure, but not its "the candidate" wording verbatim.**
`stats.mdd_clause`/`floor_clause` are metric- and model-agnostic and reuse cleanly. The "Best case"
sentence (`report.py:432-434`, *"assumes the candidate wins every {unit} the models differ on"*)
is written for one specific, already-run comparison and reads as false or misleading applied to a
hypothetical family — needs its own variant here ("assumes each of the other models would win every
{unit} it differs from the reference on"), not the existing string reused unedited. Flag this for
Unit A's implementer: `resolving_power_line` itself should probably **not** be called for this
sentence without a `hypothetical: bool` branch or a parallel small function, since silently reusing
it produces a sentence about "the candidate" when no candidate was named.

**Print unconditionally whenever the ranked table has >=2 models**, per the plan's own default — this
part is right and cheap, and it is exactly the honesty statement AC-6 (the requirements doc's own
acceptance criterion) asks the consolidated document's preamble to make at the sweep level; doing it
per-pack too is the correct place for it to first appear, since it is the pack's own sample size and
pack's own metric count that set it.

## 7. FR-11 caveat fidelity — spot-checked against `-ml`, faithful

Checked the plan's §2.3 table (the source for `rank_report`'s restated-caveat dict) against `-ml`
directly, not against the plan's paraphrase of it:

- **Embedder, recall@10 = 37/38.** `-ml` §7.4: "recall@10 = 37/38. Only 1 item is available to win.
  McNemar needs 6. It can never fire in the 'candidate is better' direction... this set can detect a
  materially *worse* embedder but cannot certify a *better* one." Plan's restatement matches exactly.
- **nlq-generator, n=34 not 40.** `-ml` §7.2 row + the v1.25 changelog: "n_eff = 34, not 40" — 6 of 40
  items structurally unanswerable and excluded from the McNemar denominator, `Y=40` still governing
  latency coverage only. Plan's restatement matches.
- **Guard-judge, two co-equal metrics, no headline.** `-ml` §7.3 confirms this exactly
  (`headlineMetric = null`, both metrics "reported with equal weight... no arithmetic combining
  them"). The plan's §2.3 table captures the qualitative shape correctly, **but the FR-11 restated
  caveat for this pack should also carry the concrete numbers** — `-ml` §7.3's own table
  (floor 15.0/20.0 pp, MDD₈₀ 21.9/28.7 pp at the two-member α=0.025, vs. 19.1/25.1 pp at a
  hypothetical k=1) — not just the qualitative "two co-equal metrics" sentence. FR-11's own examples
  (the embedder ceiling, the nlq denominator) are both *quantitative* ceilings; a guard-judge caveat
  that stops at "no single headline" restates less than what FR-11 asks for the other two packs. Fold
  the α=0.025 MDD figures into the pack's caveat string.
- **Tool-caller, n=12, unit=scripts.** `-ml` §7.2/§4.5: 3 shapes × 4 distinct scripts, one observation
  per cluster, DEFF 1.00 by construction, floor 50.0 pp, MDD₈₀ 57.8 pp. Plan's §2.3 table names the
  unit and n correctly; same recommendation as guard-judge applies — the caveat string should carry
  the 50.0/57.8 pp figures, since "very wide floor/MDD at this n" (the plan's own phrasing) is a
  claim FR-11 asks to be restated, not merely characterized.
- **Chat-responder, quality not measured.** Matches `_render_role_caveat`'s existing, shipped string
  verbatim — correctly identified as reusable rather than re-authored.

## 8. Other observations

- **`run --reference`/`RunConfig.referenceKey` claim, verified independently.** Grepped
  `referenceKey` across `modelbench/`: exactly two hits, the field declaration
  (`runner.py:127`) and the CLI assignment (`cli.py:399`); `run_pack` (`runner.py:778`) never reads
  it, and no `RunResult`/fingerprint field carries it. The plan's claim that FR-8's reference
  selection is necessarily report-time, not run-time, is correct.
- **`ContinuousMetric.support` is already the right carrier for Q1's clamp** (§3 above) — worth
  noting to Unit A's implementer as a "no new field needed" fact, since the plan doesn't mention this
  field exists.
- **No other statistically-unsound shortcut found** beyond the four flagged items and the blocker in
  §2. The plan's explicit refusal to reuse `Verdict.text`/`ContinuousVerdict.text` (§2.4/§3.2, on
  account of the polarity defect) is the correct call and is honoured precisely by `_better`/
  `_LOWER_IS_BETTER` doing the winner-label work instead.

## 9. What's solid

- The plan's own identification of the two-arm ceiling (`_comparison_pair` unconditionally returning
  `runs[0], runs[1]`) and of which existing renderers are already N-arm-generic (§2.2) is accurate —
  confirmed by reading `report.py` directly rather than trusting the summary.
- The decision to make `rank_report` a new function rather than a `compare_report` branch, and `rank`
  a new subcommand rather than a `compare --rank` flag, are both right calls for the reasons given
  (keeping an already-dense, honesty-rule-critical function from acquiring a second, orthogonal
  control-flow axis).
- `_LOWER_IS_BETTER`/`_better`'s guard-shaped-constant design, and the explicit mutation-test plan for
  it (§5 test 2), match this component's own convention exactly.
- The consolidated document's design (extraction over already-rendered markdown, never a second
  cross-pack computation) correctly preserves the `load_history()`-takes-one-`packId` structural
  invariant.

## 10. Required changes before implementation, summarized

1. **Blocker.** Add `stats.verdict()`'s `correction_k: int | None = None` keyword-only parameter
   (§2) before Unit A's FR-8 code is written; this is a `stats.py` contract change and needs its own
   review given this module's mutation-testing bar, same standing this plan already grants the §2.4
   polarity-defect backlog entry.
2. Build a new `mean_bootstrap_interval` (or equivalently named) function for Q1 (§3) — do not call
   `paired_bootstrap` with raw values.
3. Resolve FR-8's guard-judge family as one combined `2·(N-1)`-size Holm ladder (§5), not two
   independent `(N-1)`-size ones.
4. Fix the no-`--reference` resolving-power sentence's k for guard-judge to `2·(N-1)` (§6), and
   remove the "top-ranked model" anchor from its prose.
5. Fold guard-judge's and tool-caller's concrete floor/MDD figures into their FR-11 caveat strings
   (§7), not just the qualitative shape.
6. One documentation comment linking `_METRIC_BY_TIER`'s `falseAdvanceRateBoundary` to
   `_LOWER_IS_BETTER` (§4) — cheap, not blocking.

## Pass 2 (2026-09-19)

Re-reviewed `model-bench/docs/plans/small-model-catalog-sweep.md` **Version 2** (`architect`'s
revision addressing this review and `analyst`'s `docs/reviews/small-model-catalog-sweep.md` in one
pass) directly against the plan text itself, not against the revision's own changelog summary.

**Verdict: approve, with two minor suggestions (§Pass 2.2 below).** Nothing here blocks Units A/B/C.

### Disposition of Pass 1 findings

- **§2 blocker (`stats.verdict()`'s `correction_k`).** **Fixed.** §3.2.1's signature and precondition
  rewrite (`k = correction_k if correction_k is not None else len(family)`, checked against
  `resolving.alpha_mdd == resolving.alpha_family / k`) is exactly the mechanism recommended:
  `metric_name in family` untouched, Rule 7's floor enforcement left as the single copy, default
  reproduces today's one call site. The rejected-alternative reasoning (no second floor-enforcement
  site) is reproduced correctly, not just asserted.
- **§3 Q1 (`mean_bootstrap_interval`).** **Fixed.** §3.1's CI-column bullet and §3.4's Q1 recap
  specify a new function clamped directly to `ContinuousMetric.support` (not a difference-support
  conversion), sharing the resample engine with `paired_bootstrap` via a private helper, never
  calling `paired_bootstrap` with raw values — and reproduces this review's own descriptive-only
  caveat text verbatim. §5 test 0 pins the clamp, the refusals, and that the engine is shared rather
  than duplicated.
- **§4 Q2 (`_LOWER_IS_BETTER` completeness).** **Unchanged, correctly.** No fix was needed; the
  latent-risk note is now cross-referenced at both `_LOWER_IS_BETTER`'s own declaration (§3.1.1) and
  as an explicit implementation-step item at `_METRIC_BY_TIER`'s site (§4), closing the
  discoverability gap this review flagged.
- **§5 Q3 (combined candidate×metric family).** **Fixed.** §3.2.2 replaces the two-ladder default
  with one combined `2·(N-1)`-size Holm ladder, one `holm_steps` call over the flattened p-value
  list, `correction_k = len(p_values)` passed per `verdict()` call. Checked the composition question
  I raised in the handback explicitly (whether `correction_k` as specified can actually carry
  `2·(N-1)`, not only `N-1`): yes — `correction_k` is a plain `int` with no coupling to which axis
  produced it, so `resolving.alpha_mdd == alpha_family / 32` for guard-judge and `== alpha_family /
  16` for every other pack both satisfy the same precondition, and §5 test 0's `correction_k=16`
  case plus test 10's dedicated "one call, 32-entry ladder, not two 16-entry ladders" assertion pin
  both magnitudes directly.
- **§6 Q4 (resolving-power sentence without `--reference`).** **Fixed.** §3.4's Q4 recap ties `k` to
  Q3's resolution per pack (`N-1` vs. `2·(N-1)` for guard-judge — the plan's own text calls out that
  its original generic `N-1` was wrong for guard-judge), drops the "top-ranked model" anchor for the
  model-agnostic wording this review specified near-verbatim, keeps it printed alongside (not instead
  of) the unmodified `resolving_power_line`, and flags the "Best case — assumes the candidate..."
  sentence's need for its own hypothetical variant rather than reuse, exactly as raised.
- **§7 FR-11 concrete figures.** **Fixed.** §2.3's table rows for guard-judge (floor 15.0/20.0 pp,
  MDD₈₀ 21.9/28.7 pp at `alpha_mdd=0.025`) and tool-caller (floor 50.0 pp, MDD₈₀ 57.8 pp at n=12
  scripts) now carry the concrete figures this review cited from `-ml` §7.3/§7.2, not only the
  qualitative shape, and §3.1's caveat-column bullet says the same for the rendered string.
- **§8/§9/§10 (other observations, what's solid, summary list).** Superseded by the above dispositions
  and by `analyst`'s own §2.2 (marker-count) and §2.4 (footprint coercion) fixes, which are outside
  this review's remit and not re-checked for anything beyond §Pass 2's interaction check below.

### New findings (Pass 2)

**Pass2-1 [MINOR] — the combined ladder's pre-registration discipline for a candidate with no paired
data isn't explicitly stated or tested for the guard-judge case.** `-ml`'s established rule — "k is
fixed by pre-registration, not by how much data arrived" (review P3-1) — is already enforced in
`compare_report`'s existing homogeneous-binary code (`report.py:1244-1266`): even when
`outcomes.n_units == 0` for one metric, `mcnemar_exact(0, 0) = 1.0` is still appended to `p_values`
and that metric still consumes a Holm rank, so `k` never shrinks just because data is thin. §3.2.2's
pseudocode ("for candidate in candidates: ...build PairedOutcomes, append `mcnemar_exact(b, c)`...")
doesn't restate this for the new combined ladder, and §5's test list doesn't have a dedicated test
pinning it there (test 9 covers the four decision states, including "no verdict — no paired data,"
for a single-metric family; test 10 doesn't repeat a missing-candidate case for guard-judge
specifically, which is exactly where a silent `k`-shrink would first matter at this sweep's scale).
**Not a blocker** — §3.2.2 explicitly commits to reusing "the existing pairwise primitives exactly as
`compare_report`'s homogeneous-binary path already does," and that existing path already gets this
right, so the behavior is very likely inherited correctly by construction. Suggested: one sentence in
§3.2.2 stating this explicitly (mirroring `report.py:1244-1247`'s own comment), and a test 10a — one
guard-judge fixture where a candidate has zero paired items for `falseSuspendRate` — asserting the
combined ladder still has 32 entries, not 31, and every other candidate's Holm threshold is
unaffected.

**Pass2-2 [MINOR] — `mean_bootstrap_interval`'s `levels` argument isn't pinned at its `rank_report`
call site.** §3.1 and §5 test 4 specify `support=(0.0, 1.0)` explicitly for the ranked table's MRR CI
but never state what `levels` is passed. Per this review's Pass 1 answer (§3 above), it must be the
plain, unadjusted 95% levels (`stats.LEVEL_CI95_LO`/`_HI`) — the same status as a descriptive Wilson
cell — never the family-corrected `alpha/(2k)` levels `continuous_verdict` uses for an actual k-member
verdict family, since this interval carries no multiplicity correction. **Not a blocker** — the plan
gives no wrong guidance on this parameter, it simply doesn't pin one yet. Suggested: one clause in
§3.1's CI-column bullet and an explicit `levels=(stats.LEVEL_CI95_LO, stats.LEVEL_CI95_HI)` in test 4.

### Interaction check — `analyst`'s fixes against the statistics

Checked `analyst`'s three landed fixes (marker-count/consolidation variable-count parsing, footprint
non-string-value coercion, latency-`None` display) for any interaction with the Holm/CI machinery
above: **none found.** All three are display/plumbing paths (markdown comment parsing, a `str()`
coercion applied before the footprint ever reaches `rank_report`, a `—` display fallback) that share
no code path with `correction_k`, `mean_bootstrap_interval`, the combined Holm ladder, or the
resolving-power sentence. Guard-judge's two ranked tables contributing two rows to the consolidated
index (rather than one) is consistent with FR-9's non-arithmetic-index requirement — each row is
independently labelled by metric name, nothing combines them.
