# Change History — model-bench

> Dated log of actual changes to the `model-bench` component. Most recent first.

## 2026-09-03 — S1 third gate round: absence is not an outcome, and five sentences that were false

**What:** Closed the third round of gate findings on S1 —
`docs/reviews/small-model-benchmarking-impl.md` `## Pass 3` (**P3-1**…**P3-15**) and
`docs/reviews/small-model-benchmarking-ml.md` `## Pass 3` (**M-ML-7**, **m-ML-7**, **m-ML-8**,
**n-ML-4**…**n-ML-6**) — against method note **v1.7** and plan **v1.7**. Test-first throughout;
314 → 353 tests, and 28 mutations run against the fixes with **27 killed and one equivalent by
construction**. This round is dominated by a single failure mode: **the blocker and four of the six
majors are about what the report *says* rather than what it computes** — two false clauses, one
disclosure the report omitted entirely, and one number it never printed — and every one was found
by rendering a report and reading it, not by an assertion. The remaining two majors are tests that
could not fail.

**P3-1 (blocker) — an arm holding no data for a metric was scored as failing every item.**
`report.py` carried two defaults of its own: a missing `scoreable` entry read as *scoreable*, and a
missing `counts` entry then scored the row a **loss**. Together they rendered
*"cand is better than incumbent: +100.0 pp (95% CI [60.8, 100.0] pp) … p=0.002"* against an arm
whose ten items carried `counts={}` and `scoreable={}` — while the §4.3 tally, whose entire job is
to make dropped rows visible, printed `0 unscoreable in both`. Absence was laundered into the
denominator's complement, the mirror image of the laundering `-ml` §4.3 forbids. Which state a row
is in is now **`ItemResult.scored_outcome`'s call and nothing else's**, with three declared answers:
absent or `False` in `scoreable` is *no outcome* and routes through the tally; `True` **must** carry
a count, and one that does not is refused (`IncompleteItemRecord`) rather than read as a zero — a
scorer that declares an item scored must supply its score. A metric whose paired intersection is
empty now renders an explicit refusal with the tally beneath it and gets **no** `ResolvingPower`
(`n_effective` of zero is not a small sample); its Holm row prints `—` and `no verdict — no paired
data` rather than `mcnemar_exact(0, 0)`'s misleading `1.000`. **This is a contract on S2's scorers**
and is recorded in `AGENTS.md`.

**M-ML-7 / P3-2 (found independently by both gates) — the "not distinguishable" verdict asserted
the observed difference was below the MDD without checking.** `"; the observed X pp is below that."`
was fixed prose, false whenever `|diff| >= mdd80` — which is §7.1's *normal case for a model swap*,
a candidate that wins more than it loses without strictly dominating. Measured by the statistics
gate: 268 of the 1 580 by-construction tables that print the clause printed it falsely. Note v1.7
§3.2e mandates a conditional clause and publishes the alternate wording verbatim, discordance
counts included; it is implemented as published. The comparison is **strict**, and that boundary is
reachable rather than theoretical — swept over `6 <= n <= 120`, `|diff| == mdd80` occurs at k=2 for
n = 90, 100 and 120, where "below that" is false a second way — so it has its own test.

**P3-3 — the fail-safe path claimed a widening that never happened.** The cluster-path label read
*"widened by sqrt(DEFF)=1.00 **for the declared clustering**"* at `design_effect == 1.0`: nothing
was widened and no clustering was declared, on the path **every** comparison carries until S2's
determinism probe lands. What actually displaces McNemar there is Rule 4's other half, the
`basis` — and the sentence never named it. The clause is now conditional: it names the design
effect where one was applied, and the basis where none was. **The sentence itself is not note-owned
prose** — `-ml` §3.4 Rule 4 requires "the design effect and its basis printed" on this path but
publishes no string for it, which is recorded as an open item for `data-scientist`.

**P3-4 — `--negative-control` wrote a durable report indistinguishable from a real comparison.**
The mode puts two copies of one record in both arms, so `b = c = 0` is arithmetic; the report said
nothing about that (`grep -ic negative` returned 0) and was filed beside real comparisons under a
filename differing only in its sequence number. A reader got a plausible validated null. The report
now opens with a banner naming it a wiring smoke check that **cannot fail**, and pointing at the
real negative control (two independent runs, an acceptance step). The code comment that claimed the
report already said this is corrected.

**P3-5 — the bootstrap seed was a literal in the renderer and was never printed.** `report.py`
passed `bootstrap_seed=20260902`, duplicating the manifest's `sampling.seed` in a type that had no
field for it, so the pack's own declaration could not reach the decision — and on the fail-safe path
the seeded bootstrap is what decides. `PackRef` now carries `seed` with **no default**, read from
the manifest (a manifest omitting it is refused by name), and the `decided by:` line prints it where
a resample actually ran. The test asserting the printed line was not enough on its own: it left the
literal alive, printing one seed over an interval resampled at another. It now asserts that two
packs differing only in `seed` render **different intervals** — possible only at a fixture coarse
enough for the percentile to move (n=12, b=5, c=3; at n=40 the rendered bounds are identical at
every seed tried). The fingerprint half stays S2's.

**P3-6, P3-7 — two tests that could not fail.** The suite's only k=2 α assertion was
`"alpha=0.025" in md`, satisfied by the family-wise paragraph rather than by the MDD sentence it was
placed to guard, so `provenance` naming the wrong α survived; it now asserts the whole provenance
parenthetical, and the floor's own α beside it. The exploratory-label test asserted two strings'
presence and not their pairing, so **inverting** the filter — labelling the pre-registered verdict
metrics "exploratory" and hiding the genuinely exploratory ones — was green; it now asserts the
rendered line whole plus the negative. Both inversions now fail.

**The minors and nits.** `compare --session` had no test at all (P3-8) and now has two, from both
directions. `index.csv`'s `valid` column (P3-9), the `armKind` absent-vs-null discriminator
(P3-10), `verdict()`'s and `resolving_power`'s design-effect guards (P3-11), `compare_report`'s
headline-membership guard (P3-12, whose failure mode was a bare `StopIteration`), `wilson_interval`'s
probability clamps and `pack_ref_from_manifest`'s `analysisUnit` check (P3-15) each gained the one
assertion that kills their surviving mutation. `holm_steps`' `alpha` default — the second literal
`0.05` in the module that declares there is only one (P3-13, n-ML-4) — is **removed** rather than
re-pointed at `ALPHA_FAMILY`, matching plan v1.7's signature block. `resolving_power` now refuses
`design_effect < 1.0` at construction instead of `<= 0` (n-ML-5): below 1 a design effect *inflates*
effective *n* and shrinks both printed bounds, and the refusal used to arrive a layer later. The
report filename is now the manifest's `packId`, not the pack directory name (P3-14) — the half
Pass 1's m-6 left behind. m-ML-7's `floor_clause` boundary, m-ML-8's duplicated MDD stem and
n-ML-6's hard-coded `"80% power"` were closed by the same round.

**One correction to a test, not to the code.** `verdict()`'s design-effect precondition raised the
*same sentence* as `paired_cluster_bootstrap`'s identical bound one layer down, so the obvious test
passed with the precondition deleted. What it asserts now is the **ordering** Rule 4 states — every
precondition checked before any instrument is selected — which is visible only when the two orders
produce different errors.

**Verification.** `.venv/bin/python -m pytest -q -m "" -rsx` → **353 passed**, nothing skipped,
deselected or xfailed; `--collect-only` collects 353, so the run count equals the collected count.
`.venv/bin/ruff check .` → clean. 28 distinct mutations run, **27 killed**. The survivor is
**equivalent by construction**: restoring `report.py`'s duplicate of the MDD stem with *identical*
text renders identically, so no test can distinguish it. The mutation that matters — the same stem
edited in `stats.py` while the duplicate stays stale, which is exactly the drift m-ML-8 predicted
M-ML-7's fix would cause — **is** killed, by the test asserting the report renders
`stats.mdd_clause`'s own string.

## 2026-09-03 — S1 second gate round: the floor's α, McNemar as a veto, Rule 7 by path

**What:** Closed the second round of gate findings on S1 —
`docs/reviews/small-model-benchmarking-ml.md` `## Pass 2` (**B-ML-2**, **M-ML-6**, **m-ML-6**) and
`docs/reviews/small-model-benchmarking-impl.md` `## Pass 2` (**P2-1**…**P2-5**) — against method
note **v1.6**, which landed this morning and changed Rules 3, 4, 6 and 7, §3.3, §7.1 and §7.3.
Test-first throughout; 296 → 314 tests, and 22 mutations run with **one survivor, equivalent by
construction** (see below).

**M-ML-6 — the observable floor moves to the unadjusted α, and `ResolvingPower` carries two αs.**
The floor claims *"below Y nothing can reach significance at any observed outcome"*, which is true
only at the **loosest** Holm step a member can face. Printed at α/k it is `7/n` and **false**: at
n=40 a rank-2 member with b=6, c=0 reaches p=0.031, clears its own 0.05 step, and its 15.0 pp sits
below the 17.5 pp the old floor printed. `resolving_power` now takes `alpha_family` **and**
`alpha_mdd`, both keyword-only with no default, and each bound is computed at its own — the floor
at `alpha_family`, the MDD unchanged at `α/k`. §7.1's mandatory sentence names both. The third α,
Holm's data-dependent `alpha_step`, stays `verdict()`'s parameter rather than becoming a field:
it is known only after the family is ranked, so a field would be `None` until it was not, and the
number would have two homes. The sweep went past `stats.py`: `PackMetrics.alpha` — a code-side
restatement of exactly this α — is now `alpha_family` / `alpha_mdd`, and `report.py` reads them
instead of recomputing `0.05 / len(family)` inline. `stats.ALPHA_FAMILY` is the single home of the
unadjusted 0.05.

*Consequence, taken deliberately:* the α/k floor was reducing Holm to Bonferroni for every
difference in `[6/n, 7/n)` — precisely the band §7.3 already prices as the cost of a second verdict
metric — so the build was charging that price twice. It no longer is, and the rendered family table
for the review's own case now reads `distinguishable` where it read `not distinguishable — below
the observable floor`.

**B-ML-2 — the substitute path is a conjunction, not an interval.** At `design_effect == 1.0` with
`basis == "assumed"` — the fail-safe **every** comparison carries until S2 lands the determinism
probe — the decision moves off McNemar and `sqrt(1.0)` widens nothing, so a bare percentile interval
was deciding. Reproduced at n=40: `(b=7, c=1)`, `(9, 2)` and `(11, 3)` all rendered
*distinguishable* at p = 0.057–0.070, where the exact test refuses, and Rule 7 does not catch them
because 15.0, 17.5 and 20.0 pp are all **at or above** the floor. The non-`by-construction` decision
is now *"the widened CI excludes zero **and** `mcnemar_exact <= alpha_step`"*. Note v1.6's Rule 4
permits this explicitly: the objection to McNemar under clustering is that it *rejects* too readily,
and a necessary condition only ever removes rejections, so the pair is uniformly at least as
conservative as either instrument alone. The verdict strings say which instrument played which
role — one sentence for both paths would have contradicted one of them.

**m-ML-6 — Rule 7 splits by path.** On `mcnemar-exact` the invariant is a **theorem** (re-verified
here by binary search over every `b + c <= 400` at both αs, zero violations), so a fire is a module
bug and now raises `Rule7Violation`; silently demoting discarded exactly the detector property the
rule exists for. On `cluster-bootstrap` it stays demote-and-name, because a widened interval and a
shrunken effective *n* legitimately disagree there. Sequenced **after** M-ML-6, as the review
required: at α/k the McNemar branch is reachable and the raise would have fired on correct data. A
fifth precondition makes the theorem's premise checkable rather than assumed — `alpha_step` must
lie in `[alpha_mdd, alpha_family]`, which Holm's own steps do by construction.

**The bin-edge truncation guard stays; its justification was corrected.** It was load-bearing on the
17.5 pp cell at n=40 — **the cell M-ML-6 deleted**. Swept to n ≤ 2000: with `b_min = 7` naive
truncation misfires at n = 5, 10, 20, 40; with `b_min = 6`, which the floor now always uses, never.
So the guard is **defensive**, kept because `b_min` is a function of α and any future α reopens the
hazard, and its test is no longer a regression pin on a published figure. Both the code comment and
the test say so. The `floor(x/precision)` expression is still pinned by the code's own form — a
test now also pins the `precision` parameter, since `floor(x*1000)` agrees at the default and
nowhere else, which is how the original sweep missed the hazard.

**Engineering findings.** **P2-1** — `"measured"` at DEFF 1.0 had no test at Rule 4's branch, and
widening `mcnemar_may_decide` to admit it survived all 296 tests; two mirror tests (unit and report)
now close it. **P2-2** — `validate()` accepted `benchSchemaVersion: true` (`True == 1`) while
`load_history` quarantined it, so `store()` wrote a record the reader refused; the bool guard now
lives at both enforcement points, and a quarantined bool no longer lands in a field typed
`int | None`. **P2-3** — `holm_steps` builds its list without a `None`-filter and `report.py` zips
`strict=True`, so a short ladder raises instead of silently dropping a pre-registered verdict
metric. **P2-4** — accepted in part, with a correction: `Path(".").name` is `""` so `"."` is
redundant, but **`Path("..").name` is `".."`**, so `".."` is *not* already caught and stays; one
third of the guard was unreachable, not two thirds. **P2-5** — `packs.py`'s `contentHash` docstring
now says `None`, matching the code.

**Verification.** `.venv/bin/python -m pytest -q` → `314 passed`; `.venv/bin/ruff check .` → clean.
22 mutations run against the fixes, **21 killed**. The survivor — restoring `holm_steps`'
`None`-filter — is **equivalent by construction**: every index is assigned, so the filter changes
no output on its own. Compounding it with a ladder that actually returns short is killed twice over
(the strict zip and the length invariant), which is the honest statement of what P2-3's fix buys.

## 2026-09-03 — S1 gate remediation: both blockers, all ten majors, and Rule 7

**What:** Fixed the findings of the two independent S1 gates —
`docs/reviews/small-model-benchmarking-impl.md` (`analyst`: 1 blocker, 6 majors, 7 minors, 4 nits)
and `docs/reviews/small-model-benchmarking-ml.md` (`data-scientist`: 1 blocker, 4 majors, 5 minors,
3 nits) — against plan v1.5 and method note v1.5. Test-first throughout; every fix was
mutation-tested and the reviewer's **ten surviving mutations are now all killed**.

**The two blockers.**

- **B-ML-1 — the clustered decision path did not cluster.** `verdict()`'s substitute for McNemar
  was `paired_bootstrap` over the *rows* of the paired table: an i.i.d. resample of observations
  the declared design effect says are correlated, so the interval was identical at DEFF 2, 4 and 7
  and *narrower* than the MOVER-D it replaced. It changed the instrument's name, not its interval.
  New primitive `paired_cluster_bootstrap` inflates the percentile half-widths about the point
  estimate by `sqrt(design_effect)` — the Kish variance ratio is exactly the quantity that converts
  (`-ml` §3.4 Rule 5). **This is the note's "smallest honest version", taken deliberately:** the
  structurally right fix resamples clusters of paired differences, and `PairedOutcomes` carries one
  row per analysis unit with no grouping, which could only come from a pack declaring
  `replicatesPerScript > 1` — something Rule 6 makes a validation error while only the one-level
  `cluster_bootstrap` exists. Building it now would have had no data to consume and no seam to
  reach it.
- **B-1 / M-ML-2 — Holm–Bonferroni was printed and never applied.** `report.py` called `verdict()`
  without `alpha_step`, so every metric was decided at plain Bonferroni α/k, and `holm_thresholds`
  had no step-down stop. `compare_report` now runs **two passes** — Holm is a property of the
  family, so no verdict can be decided until every p-value exists — and `holm_thresholds` is
  replaced by `holm_steps`, returning a `HolmStep` per member with its rank, threshold, `tested`
  and `rejected`. `verdict()` gained `holm_tested`, which is the stop.

**Rule 7 (`-ml` v1.5 §3.4), enforced in `verdict()` rather than left to a test.** No verdict path
returns `distinguishable` when `|diff|` is below `resolving.observable_floor`. Three decisions in
it, each with a reason:

- **It demotes and says so; it does not raise.** The note's contrast is code-versus-test, not
  raise-versus-demote, and a raise would be unreachable in practice: the √DEFF-widened bootstrap
  and McNemar's exact rejection region are different instruments that do not align by construction
  (measured — at DEFF 2 on the `(34, 6, 0, 0)` table the widened interval still excludes zero while
  15.0 pp sits below the 30.0 pp floor). The demotion renders the contradiction it resolved, which
  surfaces the defect more loudly than a traceback the report never prints.
- **It compares against the exact float, never `format_floor_pp`'s truncation** — otherwise the
  invariant inherits the presentation layer's rounding and can fire, or fail to fire, by 0.05 pp.
- **The converse is not asserted.** `|diff| >= floor` does not imply distinguishable; §3.2c's row 4
  `(20, 8, 2, 10)` is the counterexample already in the suite — 15.0 pp exactly on the α=0.05
  floor, p = 7/64, not distinguishable.

It never fires on the McNemar branch: `test_the_mcnemar_path_satisfies_rule_7_by_construction`
checks every `(b, c)` split at n ∈ {12, 20, 30, 38, 40, 48, 85} and α ∈ {0.05, 0.025}. That
asymmetry is what makes it a detector rather than a formality.

**The floor's rounding direction, per the adjudication: the floor truncates, the MDD ceilings.**
`stats.format_floor_pp` is the one place the direction lives, and it is where the report and the
verdict strings both print from — the tests assert **through the formatter**, because re-rounding
inside a test (`round(observable_floor(...) * 100, 1)`) asserts the presentation layer's arithmetic
against itself. `ResolvingPower.observable_floor` stays exact, so Rule 7's guard is not weakened.
Truncation is guarded (`math.floor(x / precision + 1e-12)`), mirroring the MDD's `- 1e-12`, and
**the guard is load-bearing rather than defensive**: `7/40 = 0.175` is `174.99999999999997` bins in
IEEE doubles, so naive truncation prints `17.4` for the α=0.025, n=38–40 row the note publishes as
**17.5**. Corrected cells: 15.8→15.7 (n=38), 7.1→7.0 (n=85), 46.7→46.6 (n=15, α=0.025); 58.3 and
23.3 were already truncations.

**The other majors.**

- `load_history` validated *after* the pack filter, so a record whose `packId` was blanked or
  deleted on disk landed in **neither** returned list — the comparison quietly lost an arm (M-1).
  The filter now drops only a record that *says* it belongs to another pack; it also applies to an
  unknown schema, whose `packId` is readable, and stays **off** `unparseable`, which cannot declare
  one (m-1).
- `RunResult.designEffect`/`basis` lost their dataclass defaults (M-2, m-ML-3, plan v1.5 §3.5). The
  legacy fallback stays in `from_dict`, where it is a reader's §3.4.3 compatibility rule.
- `BinaryMetric` gained a required `unit`, and the Arms table prints a Wilson interval only over
  the analysis unit (M-ML-3). §4.4's first mandatory consequence is verbatim *"Never print a Wilson
  interval over a turn-pooled count"*, and a turn-pooled 142/320 was printing ±5 pp where the
  honest bound is ~48.7 pp. The count is never suppressed; only the precision claim is.
- `_paired_rows` returns a `PairedRows` tally and every verdict prints it — the `asymmetry` count
  §4.3's paired corollary requires, plus rows present in one arm only and unscoreable in both
  (M-5, M-ML-4). It is printed even when nothing was dropped, because otherwise a reader cannot
  tell a shrunken `n` from a full one.
- `min_detectable_difference` raises `UnattainablePower` below `b_min(alpha)` units instead of
  converging on its bisection bracket and returning `1.0`; `ResolvingPower.mdd80` is then `None`
  and the line reads *"No difference is resolvable…"* (M-ML-1). The delivered build printed
  *"resolves differences of >=100.0 pp with 80% power"* where power is identically **zero**.
- A comparison with fewer than two arms has its own reason, and `--models` naming a key with no
  stored run exits **2** rather than silently rendering a one-arm report (M-6).
- The basis/design-effect propagation is now tested at report level (M-3), and prints the **weaker
  of the two actual bases** rather than collapsing to `assumed` — false provenance in the one
  sentence whose job is auditability (m-ML-4). The decision rule is unchanged.
- `REQUIRED_BY_SCHEMA` and `FORBIDDEN_BY_ARM_KIND` are pinned against **independently transcribed
  literals**, by name and by tier (M-4). Parametrizing over them meant deleting an entry deleted
  its test case rather than failing one.

**Minors and nits:** the unpaired label distinguishes a content-hash divergence from a version one
(m-2); `_unit_ids` is called by `_paired_rows` instead of being dead code the docstring names
(m-3); `--role` and the index's `latencyMsP95` gained tests (m-4); `PackRef.contentHash` is
`str | None` so "not yet computed" is expressible (m-5); `compare` filters by the manifest's
`packId`, not the directory name (m-6); `store()` refuses a `runId` that is not a bare filename
(m-7); the tautological assertion is gone (n-1); `Fingerprint` copies its mapping behind a
`MappingProxyType` and hashes its **values** (n-2); an absent `aggregates` block is reported as
`unparseable` rather than repaired into an empty one (n-3); the conditionality clause names the
pack's own sample noun (n-4, m-ML-5); the `-ml` §3.2c fixtures are republished at 10 dp and
asserted at the mandated **1e-9 on the proportion**, with the docstring's margin claim corrected
from four orders to three (m-ML-1); `test_z_95_matches_the_inverse_normal_cdf` records that the
pinned literal is one ULP from `NormalDist().inv_cdf(0.975)` and must not be tightened to `==`
(n-ML-3).

**One finding declined, with its reason.** n-ML-1 asked for the floor and the MDD to share a
denominator (`observable_floor` divides by the unfloored `n_effective`; `min_detectable_difference`
floors first). Unifying them would make one of the two anti-conservative: Rule 3's principle is to
round each printed bound in the direction that keeps its own claim true, and the two claims point
opposite ways — a **larger** MDD is the safe error, a **smaller** floor is. The asymmetry is now
documented at `observable_floor`, which is the one line the finding asked for.

**Two defects found by reading rendered output, not assertions** — the same discipline that caught
the CI-orientation bug at S1. The "Best case — assumes the candidate wins every…" caveat was still
printing where no MDD exists, qualifying a figure that is not on the page; and the clustered label
was appended to two of the five verdict strings rather than all of them, so a reader seeing only a
demoted verdict was never told which instrument produced it.

**Verification, from `model-bench/`:** `.venv/bin/python -m pytest -q` → **296 passed** in 2.12s,
exit 0 (0 failed, 0 skipped, 0 deselected — the `live` marker still deselects nothing because no
live test exists until S2). `.venv/bin/ruff check .` → `All checks passed!`. **34 source mutations
against a scratch copy — 34 killed, 0 survivors**, including all ten the `analyst` gate reported as
surviving and 24 new ones aimed at this change's own fixes. Two of the new ones initially survived,
both because a test asserted a passthrough field instead of the behaviour it gates; both tests were
rewritten onto cases where the mutation changes a verdict.

## 2026-09-03 — S1: fingerprint, results, stats, report, CLI (no model calls)

**What:** Built the harness core per stage S1 of `docs/plans/small-model-benchmarking.md` §4 —
everything that decides whether a number may be printed, and nothing that produces one. No model
calls, no network, no LM Studio, no pack loader: the whole S1 suite runs offline.

- `modelbench/fingerprint.py` — `Fingerprint` (frozen, `armKind`-discriminated), `FieldSpec`,
  `FieldProblem`, `REQUIRED_BY_SCHEMA` (`{schemaVersion: {armKind: {field: spec}}}`) and
  `FORBIDDEN_BY_ARM_KIND`. Fields are held in a **mapping, not dataclass attributes**, because a
  dataclass with `None` defaults collapses *absent* into *null* — the two states plan §3.4.2 exists
  to separate. `validate()` returns problems and never raises; the `deterministic` arm kind
  forbids every model field, so `{"modelKey": "bm25"}` fails loudly on write (plan §3.4.1, gate B-3).
- `modelbench/results.py` — `ItemResult`, `RunResult`, `InvalidRecord`, `BENCH_SCHEMA_VERSION = 1`,
  a **closed union** of five typed aggregate dataclasses, `store()` (raises, no bypass parameter),
  `load_history()` (returns `(valid, invalid)`, re-validating each record against **its own**
  `benchSchemaVersion`), `rebuild_index()` and `models_with_stored_results()`.
- `modelbench/stats.py` — implements `docs/plans/small-model-benchmarking-ml.md` §3.4's six binding
  rules and nothing else: `wilson_interval` (`z` keyword-only, defaulting to the pinned
  `_Z_95 = 1.959963984540054`), `mcnemar_exact`, `mover_d_interval`, `paired_bootstrap`,
  `cluster_bootstrap`, `PairedOutcomes` (duplicate-unit guard in `__post_init__`, so it holds on
  every construction route), `resolving_power`/`ResolvingPower`, `min_detectable_difference`
  (exact bisection over the McNemar rejection region, ceilinged to the printed precision, and
  taking `n_effective: float` so a raw `int` count raises `TypeError`), `observable_floor`,
  `design_effect`/`effective_n`/`width_inflation`, `verdict()` and `holm_thresholds`.
- `modelbench/report.py` — `compare_report()`: the excluded-invalid block (AC-2), the pack
  version/content-hash banners (AC-3), the `SCHEMA VERSIONS IN THIS COMPARISON` line (§3.4.3), the
  comparison-kind line (§3.7), per-arm Wilson intervals labelled *descriptive, not the comparison
  instrument*, the resolving-power line, the three verdict strings (AC-4), Holm–Bonferroni for a
  k>1 family, and the marginal-overlap diagnostic with its footnote.
- `modelbench/packs.py` — `PackRef`, `PackMetrics`, `metrics_from_manifest`,
  `check_sampling_contract`, `pack_ref_from_manifest`. **Not** S2's pack loader: no content hash,
  no AST import walk, no data-file row-count identity. `PackRef` extends Appendix A's five fields
  with `pairingKey` and `analysisUnit`, without which §3.3's analysis-unit resolution has no source.
- `modelbench/roles.py` — FR-21's five roles and `-ml` §3.3's unit-kind column.
- `modelbench/cli.py` + `modelbench/__main__.py` — `compare` (with `--negative-control`),
  `index rebuild`, `models --tested`; §3.6a's closed exit-code set. `attest`, `validate` and `run`
  are S2's and their absence is asserted by a test.
- `run.sh` — the S0 guard block deleted, as S0's own entry said S1 would.

**Two decisions taken here that the plan does not state, both additive and both flagged to
`architect`:**

- **`RunResult` gains `designEffect: float` and `basis`.** §5 test 12b requires `runner` to *set*
  `basis`, and `-ml` §3.4 Rule 4 decides which instrument may decide from it — but the plan's
  `RunResult` shape carries neither, and a report cannot recompute either after the fact. Without
  them S1 done-condition 5b is unsatisfiable. The degradation is fail-safe: any arm not
  `by-construction` drops the comparison to `assumed`, which moves the decision off McNemar.
- **`FieldProblem.reason` gains `"unknown"`** beside Appendix A's four, for a discriminator this
  build cannot interpret — an unrecognized `armKind`, or a `benchSchemaVersion` from the future.
  Forcing either into `absent`/`empty` would mislabel it.

**One defect found and fixed by reading the rendered output rather than the assertions:** when arm
B won, `verdict()` re-oriented the difference to the winner (`+66.7 pp`) but left the confidence
interval in A-minus-B orientation (`[-86.2, -29.9]`) — a positive effect printed beside a wholly
negative interval. Nothing raised; it is a plausible-looking, internally contradictory line, which
is the exact failure mode a measuring instrument must not have. The non-significant strings now
keep the signed A-minus-B difference for the same reason.

**Verification:** `.venv/bin/python -m pytest -q` from `model-bench/` → **233 passed**, exit 0
(0 failed / 0 skipped; the `live` marker deselects nothing at S1 because no live test exists yet).
`.venv/bin/ruff check .` → `All checks passed!`. `./run.sh --help` and `./run.sh models --tested`
both exit 0. Every done-condition test was mutation-tested; the load-bearing one is S1
done-condition 5(c), where pairing on the conversation id instead of the pack-declared
`analysisUnit` is caught by the captured-argument assertion independently of the raise.

## 2026-09-02 — S0: component skeleton

**What:** Created the `model-bench/` component per stage S0 of
`docs/plans/small-model-benchmarking.md` §4 — packaging, scripts, docs skeleton and an empty
package/suite. No harness code: S0's done-condition is deliberately an empty test suite, so that
S1–S8 land against a tree that already builds and lints.

- `pyproject.toml` — `requires-python = ">=3.12"`, **no runtime dependencies** (plan §3.2, a hard
  design constraint), dev extras `pytest>=9.1,<10` + `ruff>=0.14,<0.15`, ruff `select = ["E","F","W","I"]`
  / `line-length = 100` (mcp-monitor's shape), pytest `testpaths = ["tests"]` plus falkor-chat's
  live-test convention verbatim: `addopts = '-ra -m "not live"'` and a `live` marker.
- `setup.sh` — adapted from `mcp-monitor/setup.sh`: idempotent, `--recreate`, resolves paths from the
  script's own location, ends with an import smoke test.
- `run.sh` — the mcp-monitor shape (venv check, then `exec .venv/bin/python -m modelbench "$@"`) with
  an **S0 guard**: `modelbench/__main__.py` does not exist until S1, so the script reports that in
  words and exits 1 rather than `exec`-ing into a `No module named` traceback. S1 deletes the guard.
- `.gitignore` — `.venv/`, `host.json` (the operator-attested fingerprint fields, plan §3.4),
  `results/transcripts/` (raw model output: large, and not needed for any comparison, plan §3.5).
- `README.md` — what the tool is, and the three non-features stated up front: no CI/scheduler, no
  pass/fail gate, no leaderboard or cross-role aggregate.
- `AGENTS.md` — working context: current state, the hard rules (zero runtime deps, FR-23 standalone,
  no cross-role aggregate), the `live` marker, the attested fingerprint fields, and the note that
  an empty suite exits 5.
- `docs/{BACKLOG.md,HISTORY.md}` plus empty `requirements/ plans/ reviews/ test-plans/ test-reports/`
  held by `.gitkeep` files. `BACKLOG.md` is seeded with the two items plan §7 carries forward.
- `modelbench/__init__.py` (`__version__`) and `tests/test_package.py` — one install smoke test,
  asserting `modelbench.__version__` equals the installed distribution's metadata version. The plan
  called for an empty suite at S0, but pytest exits 5 (`EXIT_NOTESTSCOLLECTED`) when nothing is
  collected, so "runs and passes with zero tests collected" cannot return 0 (plan gate finding m1).
  Resolved with this one real test rather than by configuring the exit code away: a permanent
  "no tests ran is fine" setting would still be in place at S5 and would hide a collection
  breakage. The assertion is not filler — that version string is what stamps `benchVersion` into
  every run record (plan §3.4), so a skew between `pyproject.toml` and `__init__.py` fails here.
- Root `AGENTS.md` — a `model-bench/` bullet in **Structure** and a row in **Component docs**. The
  feature's requirements and plan stay at the repo root, where they were written (plan §4 S0).

**One defect found and fixed by reading the rendered output rather than the assertions:** when arm
B won, `verdict()` re-oriented the difference to the winner (`+66.7 pp`) but left the confidence
interval in A-minus-B orientation (`[-86.2, -29.9]`) — a positive effect printed beside a wholly
negative interval. Nothing raised; it is a plausible-looking, internally contradictory line, which
is the exact failure mode a measuring instrument must not have. The non-significant strings now
keep the signed A-minus-B difference for the same reason.

**Verification:** `model-bench/setup.sh` → venv created with Python 3.12.3, `model-bench[dev]`
installed (pytest 9.1.1, ruff 0.14.14), smoke import printed `model-bench 0.1.0`; re-run to confirm
idempotence. `.venv/bin/python -m pytest -q` from `model-bench/` → `1 passed in 0.01s`, exit 0
(0 failed / 0 skipped / 0 deselected). `.venv/bin/ruff check .` → `All checks passed!`.
`./run.sh --help` → the S0 guard's message, exit 1. Note that the test command must be run with
`model-bench/` as the working directory: the repo has no root pytest configuration, so from the repo
root pytest ignores this component's `testpaths` and walks the whole monorepo (measured: 9 collected,
8 collection errors, exit 2).
