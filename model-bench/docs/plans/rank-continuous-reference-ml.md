# `rank --reference` on a continuous verdict metric — correction design

> **Status:** archived · **Owner:** `data-scientist` · **Tracks:** —

## 1. The question and the decision it serves

D-1 (`docs/test-reports/small-model-benchmarking-manual-additions-report.md`) is a real crash:
`./run.sh rank --pack embedder-graphrag-retrieval --reference bm25` raises `MetricKindError`
because `rank_report`'s reference-anchored-family builder unconditionally calls the boolean-only
`_paired_rows`/`item.scored_outcome` accessor for every pack, regardless of whether the pack's
verdict metric is binary or continuous. Wiring in the continuous accessor
(`_paired_diffs`/`item.scored_value`) is not sufficient by itself: `stats.continuous_verdict`, the
function that would decide each candidate-vs-reference interval, has its own multiplicity
correction hard-wired to `k = len(family)` (the pack's own metric-count axis) with no way for a
caller to substitute the larger, independent candidate-count axis FR-8's reference family needs —
exactly the gap `stats.verdict()`'s `correction_k` parameter was added to close on the binary side
(`docs/plans/small-model-catalog-sweep.md` §3.2.1, `stats.py:1230-1238`).

This note answers, for the `architect` plan that builds directly on it: what `k` a continuous
reference-anchored family must use and why; whether `continuous_verdict` needs a new parameter
(and its exact contract) or the fix belongs elsewhere; how a mixed binary/continuous family in
`rank_report`'s reference path should be handled; and what rendering shape a continuous
reference-family table takes. It also surfaces two adjacent, already-shipped defects this same
extension will make either newly-reachable or already-live-but-silently-wrong, both of which the
implementation unit should fix alongside the crash, not separately.

## 2. Findings from the real system

### 2.1 The precedent this must mirror, in exact detail

`rank_report`'s existing binary reference family (`modelbench/report.py:1445-1483`) builds **one**
combined Holm ladder over every `(candidate, metric)` cell in the pack's whole pre-registered
`family = list(pack.metrics.verdictMetrics)` — not `members` (which may be just the headline
metric) — then renders one table per member, filtered back out of that one combined ladder
(`report.py:1453-1472`, comment: "ONE combined call over (metric x candidate), never one call per
metric — the plan's originally-rejected per-metric default under-corrects the family-wise error
rate"). `correction_k = len(combined_p_values) = len(family) * len(candidates)`
(`report.py:1472`), passed into every `stats.verdict()` call for that pack
(`_render_reference_family`, `report.py:1284-1288`). This was resolved, not assumed, by two
independent reviews (`docs/reviews/small-model-catalog-sweep.md` §2.1,
`docs/reviews/small-model-catalog-sweep-ml.md` §2 and §5): a per-metric-independent ladder only
bounds the combined false-positive rate by the union bound, silently under-correcting relative to
one combined family, and FR-8's own requirements wording ("**one** pre-registered,
reference-anchored family per pack," singular — `docs/requirements/small-model-catalog-sweep.md`
FR-8) supports the combined reading.

`stats.verdict()`'s `correction_k: int | None = None` (`stats.py:1216-1301`) is what makes that
possible: it decouples precondition 3's divisor (`k = correction_k if correction_k is not None
else len(family)`, `stats.py:1270`) from `family`'s membership role (`metric_name in family` is
still checked unconditionally, `stats.py:1266-1269`) — "FR-8's reference-anchored family (a
candidate-count axis independent of the pack's own metric-count family) always passes this
explicitly" (`stats.py:1237-1238`).

### 2.2 Why `continuous_verdict` cannot reuse that parameter today, and what is structurally different

`stats.continuous_verdict` (`stats.py:1651-1769`) has the identical `family`/`alpha_family`
coupling `verdict()` had before the fix — `k = len(family)` (`stats.py:1719`) — but **no**
`correction_k` escape hatch. This was flagged and explicitly deferred, not missed, when the binary
fix landed: "`continuous_verdict()` has the identical `family`/`alpha_family` coupling and would
need the same parameter if a future pack ever needs FR-8's family on a continuous headline metric
with more than two in-scope models. Not needed today — the only continuous-headline pack
(`embedder`) has 2 in-scope models total, so FR-8's family is never meaningful there regardless of
this change" (`docs/plans/small-model-catalog-sweep.md` §3.2.1, "Scope note for later, not now").
`embedder-graphrag-retrieval` now has 5 stored model keys (`reports/embedder-graphrag-retrieval-
rank-20260920-03.md`), so that scope note's premise ("never meaningful there") no longer holds —
this is exactly the "later" the note anticipated.

The mechanism is genuinely different, not just a naming gap, and the difference is load-bearing
for where the correction goes:

- **Binary path:** the correction lives in the **Holm step compared against the p-value**
  (`alpha_step`), never in the printed interval. `mover_d_interval`'s `z` defaults to `_Z_95`
  (`stats.py:126`) and every call site in `verdict()` uses that default — the printed CI is a
  fixed, genuine 95% interval regardless of `correction_k`. `_render_reference_family`'s table
  header literally reads `95% CI` (`report.py:1264`) and this is accurate for every row, because
  the multiplicity correction is entirely absorbed by the `Holm-adjusted threshold` column beside
  it, never by the CI's own coverage.
- **Continuous path:** "for a continuous metric the bootstrap interval **is** the test" (§3.2d,
  cited in `continuous_verdict`'s own docstring, `stats.py:1670-1671`) — there is no p-value to
  step against, so `_family_ci_levels` (`stats.py:1574-1594`) puts the correction **directly into
  the interval's own quantile levels**: `alpha/(2k)` and `1 - alpha/(2k)`, computed once and fed to
  `paired_cluster_bootstrap` (`stats.py:1720-1729`). Widening `k` here **widens the printed
  interval itself** — it is not a separate, decoupled decision threshold the way `alpha_step` is
  on the binary side.

This means the two axes cannot be combined by literally reusing `verdict()`'s parameter shape and
calling it done — the *parameter* generalizes cleanly (§3.1 below), but its effect on what gets
printed is qualitatively different, and two consequences fall out of that difference that the
rendering design (§3.4) and two adjacent defects (§2.3, §2.4) all trace back to.

### 2.3 An already-live, silently-wrong sentence this same code path prints today for a continuous pack

`_rank_resolving_power_lines` (`report.py:1298-1352`) runs unconditionally, once per member of
`family`, for **every** pack with `len(runs) >= 2` — it is not gated on `reference` being given,
and it is not gated on metric kind. It calls `stats.resolving_power` (Wilson/McNemar-shaped: a
two-proportion detection-power calculation) and prints both the pack's own already-published
single-comparison figure and the FR-8 "if this family were run" hypothetical, in the vocabulary of
"differences of >=X pp with 80% power" and "alpha_mdd = alpha_family/k".

I pulled a real, already-generated report to confirm this is not hypothetical:

```
$ grep -n "resolves differences\|If this pack's optional" \
    reports/embedder-graphrag-retrieval-rank-20260920-03.md
This pack resolves differences of >=20.1 pp with 80% power at n=38 effective querys ...
If this pack's optional reference-anchored family (FR-8) were run ... that family of 4 tests
would resolve differences of >=25.8 pp with 80% power (alpha_mdd = 0.05/4).
```

`embedder-graphrag-retrieval`'s verdict metric is `mrr`, a continuous ratio statistic with no
McNemar discordant-pair structure at all. `continuous_verdict`'s own docstring states plainly that
none of this vocabulary exists on the continuous path: "(i) `resolving: ResolvingPower` — it
exists to make the observable floor's and the MDD's sentences true, and a continuous metric has
neither" (`stats.py:1675-1677`), and `compare_report`'s continuous branch (§2.4 below) never calls
`resolving_power`, never prints an MDD, and never prints an observable floor for exactly this
reason (`report.py:1707-1710`: "No `holm_steps`, no `mcnemar_exact`, no `resolving_power`: none of
the three exists on this path"). `_rank_resolving_power_lines` disagrees with its own sibling
function's design and is printing a confidently-stated, methodologically meaningless
"pp"/"80% power" figure for MRR, silently, on every rank report this pack has ever produced —
worse than a crash, because nothing signals the number is wrong. This is not introduced by this
work; it predates `--reference` entirely (it fires with no flag at all) and was not caught by the
prior sweep's review passes because those focused on the binary guard-judge pooling defect
(`docs/reviews/small-model-catalog-sweep-impl.md` Unit A.5), never on this function's blanket
kind-agnosticism.

### 2.4 A latent mislabeling this extension makes reachable for the first time

`ContinuousVerdict.text` (`stats.py:1740-1753`) hardcodes the literal string `"95% CI"` in both its
distinguishable and not-distinguishable branches, even though the dataclass carries its own
`alpha_used: float` field (`stats.py:1647`, `1721`, `1767`) — computed but never read by the string
it should govern. Today this is harmless: every shipped continuous pack has `len(family) == 1`
(`embedder-graphrag-retrieval`'s `verdictMetrics: ['mrr']`), so `alpha_used = alpha_family / 1 =
0.05`, and the interval genuinely is a 95% one. The moment `k > 1` becomes reachable on this path —
which is exactly what this extension does, deliberately, for the first time — the label goes
wrong: at `k = 4` (one reference vs. 4 candidates on a single-metric continuous pack, `alpha_family
= 0.05`), `alpha_used = 0.0125`, a **98.75%** two-sided interval, printed under a banner that still
says "95% CI". This is the same class of defect this component's own honesty rules exist to catch
("Every printed bound takes the rounding direction, α, and denominator that keep its own claim
true" — `AGENTS.md` Load-bearing invariants) — it is dormant only because nothing has yet driven
`continuous_verdict` at `k > 1`, and this change is precisely what will.

## 3. Recommended method

### 3.1 `continuous_verdict` gains `correction_k`, mirroring `verdict()`'s parameter exactly — not a wrapper

Add one new keyword-only parameter to `stats.continuous_verdict`:

```python
def continuous_verdict(
    diffs: Sequence[float],
    *,
    metric_name: str,
    family: Sequence[str],
    alpha_family: float,
    unit_kind: str,
    design_effect: float,
    basis: Basis,
    B: int,
    seed: int,
    support: tuple[float, float] | None,
    correction_k: int | None = None,   # NEW — defaults to len(family); today's call site unchanged
    a_label: str = "A",
    b_label: str = "B",
) -> ContinuousVerdict: ...
```

**Semantics, mirroring `verdict()`'s docstring rigor (`stats.py:1230-1238`) exactly:** `family`
still does exactly one job — `metric_name in family` still requires a genuinely pre-registered
metric (`stats.py:1707-1710`, unchanged). The divisor `_family_ci_levels` and `alpha_used` are
computed from is `correction_k` when given, `len(family)` otherwise:

```python
k = correction_k if correction_k is not None else len(family)
levels = _family_ci_levels(alpha_family, k)
alpha_used = alpha_family / k
```

`None` (the default) reproduces the one existing call site (`compare_report`'s continuous branch,
`report.py:1740-1753`) unchanged — not the anti-conservative-by-omission shape this module's
"nothing that shapes a decision carries a default" rule refuses, for the same reason `verdict()`'s
own default was accepted: it reproduces exactly the one already-shipped, already-audited call.
FR-8's reference-anchored family, on the continuous path exactly as on the binary one, always
passes this explicitly.

**Rejected alternative: a new wrapper function around `continuous_verdict` that computes
`_family_ci_levels` externally and passes pre-corrected `levels` in.** Rejected for the same reason
the binary side rejected reimplementing Rule 7's floor check outside `verdict()`
(`docs/plans/small-model-catalog-sweep.md` §3.2.1's "Rejected alternative"): `continuous_verdict`
deliberately exposes **no percentile parameter** so that "a caller cannot render a `k = 3` family
at `1/40`/`39/40` by omission" (`stats.py:1685-1687`) — a wrapper that computed levels externally
and handed them in would defeat exactly the footgun `continuous_verdict`'s own design already
closed. The parameter belongs on the function that owns `_family_ci_levels`'s call, for the same
"two copies of a formula is one copy and one bug" reason the binary fix cites.

**Validation — one asymmetry with `verdict()` worth naming, not silently matching.**
`verdict()`'s precondition 3 has an explicit equality check to raise against
(`resolving.alpha_mdd == resolving.alpha_family / k`, `stats.py:1271`) because `resolving_power` is
built by the caller ahead of time and the check catches a caller who built it against the wrong
`k`. `continuous_verdict` computes `alpha_used` itself from `correction_k` directly — there is no
externally-precomputed value to check for consistency, so there is no equivalent precondition to
add; `k <= 0` is the only new failure mode `correction_k` introduces, and it is unreachable at
every real call site (`rank_report`'s reference path only calls this when `candidates` is
non-empty, so `k = len(family) * len(candidates) >= 1` always). I did not find an explicit `k >= 1`
guard on the binary `correction_k` either (it would surface as a `ZeroDivisionError`/`Fraction`
error, not a named `ValueError`, if `k == 0`) — adding one to `continuous_verdict` alone, while
`verdict()` stays unguarded, would be an inconsistent asymmetry. This is a **minor, optional**
hardening item, not a blocker: either add a shared `k >= 1` guard to both functions in the same
change, or leave both as they are; do not add it to one and not the other.

### 3.2 The combined divisor: `k = len(family) * len(candidates)`, the same value and the same reasoning as the binary path

Directly answering the brief's first fork: **the correct `k` is one combined `N × M` divisor
(candidates × the pack's own verdictMetrics count), identical in formula and in justification to
the binary path's `correction_k = len(combined_p_values) = len(family) * len(candidates)`
(`report.py:1472`) — not a per-metric divisor, and not a different combined formula.**

The reasoning is instrument-agnostic on purpose. The family-wise error rate is a property of the
**set of decisions being rendered under one pre-registration**, not of which statistical
instrument decides each one. The `-ml` review's own argument for combining guard-judge's two
binary metrics into one ladder — "two co-equal verdicts at α=0.05 each carry a ~9.75% chance of at
least one false 'better' under the null... two independent ladders bound the combined
false-positive risk only by the union bound" (`docs/plans/small-model-catalog-sweep.md` §3.2.2) —
applies with identical force regardless of whether each individual test is a Holm-stepped McNemar
comparison or a Bonferroni-widened bootstrap CI: what inflates the combined false-positive risk is
the number of decisions rendered inside the one exploratory family, full stop. A pack with one
continuous headline metric (`mrr`) and 4 candidates against 1 reference has exactly 4 decisions in
its family, the same count a binary pack with 4 candidates and 1 metric would have, and both must
be corrected at the same `k = 4`. If a future pack ever has an all-continuous `verdictMetrics` with
more than one member (none does today — see §3.3 for what "all-continuous" gates on), `k = len(
family) * len(candidates)`, exactly mirroring guard-judge's `2 * (N-1)`.

**Rejected alternative — a per-metric divisor (`k = len(candidates)` alone, correcting each
continuous metric's family independently of any sibling metric).** Rejected for the same
under-correction reason the binary per-metric default was reversed (§2.1): it would only be correct
today by coincidence, since every shipped continuous pack happens to have exactly one verdict
metric (`len(family) == 1` collapses the two formulas to the same number) — but the design must not
bake in an assumption "always true today, silently wrong the day it isn't," which is the exact
failure class this whole extension exists to close. Compute the general formula
(`len(family) * len(candidates)`) even though it is numerically indistinguishable from the
per-metric one at every pack this codebase ships right now; do not special-case `len(family) == 1`
away.

### 3.3 Mixed binary/continuous family in `rank_report`'s reference path: refuse whole, mirroring `compare_report`

**Yes, `rank_report`'s reference-anchored path needs the identical "refuse whole" guard
`compare_report` already has (`report.py:1664-1676`), for the identical reason stated there:** "a
family member's kind is a type fact... A mixed family is refused *whole* — no member is verdicted,
nothing is excluded — because `k` is `len(verdictMetrics)` and pre-registered, so dropping the
minority kind would shrink `k` after the results exist and under-correct the survivors."

This is not reachable by any pack shipped today — every current pack's `verdictMetrics` is
homogeneous (`embedder-graphrag-retrieval`: `['mrr']`, continuous; `guard-judge-understanding`:
`['falseAdvanceRate', 'falseSuspendRate']`, both binary; the remaining three packs each have one
binary headline metric) — but `rank_report`'s reference-family builder loops over the whole
`family` unconditionally today (`report.py:1453`: `for metric in family: for cand in candidates:`),
exactly the same unconditional-loop shape that produced D-1 in the first place. Leaving it
unguarded reproduces D-1's own failure mode the day a pack ever does mix kinds, and this codebase's
own convention is to refuse defensively rather than wait for the crash (`compare_report`'s guard
exists for a case that, per its own comment, was equally unreached at the time it was written).

**Where the guard belongs and what it changes about `rank_report`'s current structure.**
`rank_report` currently builds its combined family once, unconditionally assuming binary
(`report.py:1451-1472`), before the `for metric in members` rendering loop. The fix restructures
this into a three-way dispatch, resolved **once**, before either the binary or continuous combined
family is built:

1. Resolve `kinds = {metric: _metric_kind(reference_run, cand, metric) for metric in family for
   cand in candidates}` and check every metric resolves to the same kind across every candidate it
   is checked against — not just once via `reference_run` alone. `_metric_kind`'s own docstring
   states the aggregate type is "a type fact rather than a guess" *for any arm that has already
   passed DC-10's cross-check* (`report.py:233-234`) — but DC-10 (`_aggregate_item_mismatches`)
   only cross-checks one arm's aggregate against its own items; it says nothing about whether two
   *different* arms agree with each other on the same metric's kind. `rank_report` is the first
   context in this codebase where more than two arms are resolved against one shared `family` at
   once, so this is a genuinely new integrity surface, not a restatement of an existing check —
   flag it to the architect as a real open question (§5) rather than assuming `_metric_kind(a, b,
   name)`'s two-argument shape trivially extends to N arms.
2. `resolved_kinds = set(kinds.values())`. `len(resolved_kinds) > 1` → refuse the whole reference
   family: render one explanatory block (mirroring `_MIXED_FAMILY_MEMBER`'s existing wording,
   `report.py:769`, adapted to name that this is the reference-anchored family specifically, not
   the two-arm comparison) and skip both the binary-ladder and continuous-family code below
   entirely — no partial table for either kind.
3. `resolved_kinds == {"binary"}` → today's existing code, unchanged (`report.py:1453-1472`).
4. `resolved_kinds == {"continuous"}` → the new code this note specifies (§3.2, §3.4).

### 3.4 Rendering shape for a continuous reference-family table

The continuous table cannot reuse the binary table's exact column set, because the two mechanisms
genuinely differ (§2.2): there is no Holm step, so there is no per-row varying threshold, and there
is no "not tested (Holm stops here)" state, because nothing is sequentially stepped — every
candidate's interval is decided independently, at the same fixed `alpha_used` for the whole table.

Recommended columns: `| candidate | diff | (1 − α_family/2k) CI | decision |` — one row per
candidate, computed via `stats.continuous_verdict(diffs_row.diffs, ..., correction_k=k, ...)` per
candidate (`k` fixed once per table, from §3.2). The caption states the actual correction applied
once, numerically, rather than a bare "95% CI" header — e.g. `` `alpha_family=0.05, k=4,
alpha_used=0.0125 (98.75% CI)` `` — both because §2.4's mislabeling defect must not be reproduced
in the *new* rendering path even while it is being fixed in the *existing* one, and because a
per-row threshold column has nothing to show (every row shares the same level).

Decision vocabulary — four states, not the binary table's five (no "not tested," since nothing
steps; no "below the observable floor," since continuous has no observable-floor concept,
`stats.py:1675-1677`):

- `distinguishable` — `ci[0] > 0 or ci[1] < 0` (`ContinuousVerdict.distinguishable`, unchanged).
- `not distinguishable` — the corrected interval covers zero.
- `no verdict — no paired data` — mirroring `_NO_PAIRED_DATA`'s existing wording
  (`report.py:583-587`) exactly, for a candidate with zero paired analysis units against the
  reference.
- `no verdict — one paired unit` — mirroring `_ONE_PAIRED_UNIT`'s existing refusal
  (`report.py:798`, `continuous_verdict`'s own refusal 4, `stats.py:1712-1717`) for exactly one
  paired unit, where an interval would report a CI of zero width as though it were a measurement.

### 3.5 Fix `_rank_resolving_power_lines`'s kind gap in the same unit

Not optional scope creep: this function renders the **unconditioned preview** of the exact family
this note designs ("if this pack's optional reference-anchored family (FR-8) were run...",
`report.py:1346-1351`) for every pack, continuous or not, and it is already live and wrong for
`embedder-graphrag-retrieval` today (§2.3), independent of whether `--reference` is ever passed.
Fixing the reference-anchored family's own correctness while leaving its own no-flag preview
sentence printing meaningless "pp"/"80% power" language for the same metric, in the same report,
a few lines above, is not a coherent deliverable.

**Recommendation: gate the call, not redesign the sentence.** `compare_report`'s continuous branch
establishes the precedent directly — it never calls `resolving_power`, never prints an MDD, never
prints an observable floor for a continuous metric, because none of those concepts is defined on
this path (`stats.py:1675-1677`, `report.py:1707-1710`). `_rank_resolving_power_lines` should
resolve each metric's kind (reusing the same `_metric_kind`-based resolution §3.3 introduces) and
return `[]` immediately for a continuous metric, exactly as it already returns `[]` for `k <= 0`
(`report.py:1326`) or no aggregate data (`report.py:1330-1331`). **Rejected alternative:** invent a
continuous-appropriate power-preview sentence (e.g. bootstrap-half-width-at-k-tests). Rejected for
this unit's scope — no requirement (FR-6/FR-7/FR-8/FR-11/FR-12) asks for a continuous power preview,
`-ml` defines no such concept anywhere, and inventing one now would be new statistical design
bundled into a defect fix rather than answering the question actually asked. If a future need for
this exists, it is its own method note.

### 3.6 Fix `ContinuousVerdict.text`'s coverage label in the same change

`continuous_verdict`'s two `text` templates (`stats.py:1740-1753`) must read the interval's actual
coverage from the `alpha_used` value already computed and already carried on the dataclass, rather
than the literal string `"95% CI"`. Minimal fix: replace the hardcoded string with a formatted
coverage percentage derived from `alpha_used` (`f"{100 * (1 - alpha_used):g}% CI"`), consistent at
`k = 1` (prints `95% CI`, unchanged for every pack shipping today) and correct at `k > 1`. This is
required the moment §3.1's `correction_k` makes `k > 1` reachable at all — leaving it unfixed means
the very first continuous reference-family table this feature ever renders prints a wrong coverage
label, which is precisely the "wrong-but-plausible-looking number" failure mode this component's
own convention names as never to ship silently (`docs/reviews/small-model-catalog-sweep-impl.md`
Unit A.5's own framing, cited in §2.3 above, for a different bug of the same shape).

## 4. Evaluation design — how to know the fix is correct, not just non-crashing

**Unit tests on `stats.py` (mirroring the existing `correction_k` test pair on `verdict()`,
`tests/test_stats.py:587-610`, per this module's stated test obligation for a divisor-decoupling
parameter):**

1. `continuous_verdict(..., correction_k=None)` behaves identically to omitting the parameter, and
   both match today's one existing call site's output bit-for-bit on a fixture with `len(family) ==
   1` — a regression pin, asserted both ways (explicit `correction_k=len(family)` and omitted).
2. `continuous_verdict(..., correction_k=16)` (or any value `!= len(family)`) on a `len(family) ==
   1` fixture produces `alpha_used = alpha_family / 16`, a materially wider interval than the
   `correction_k=None` case on the same `diffs`, and `distinguishable` flips to `False` on a
   fixture engineered to be significant only at the uncorrected level — the same "assert the
   decoupling actually decouples" bar `docs/reviews/small-model-catalog-sweep-impl.md` §Unit A.1
   already set for `verdict()`'s parameter.
3. `ContinuousVerdict.text` at `correction_k > 1` prints the correct coverage percentage
   (`100 * (1 - alpha_used)`), not `95%` — a direct regression test for §3.6, since this is
   precisely the kind of static-string bug a shape-only assertion (mocked `text`) would miss.
4. `_family_ci_levels`/`continuous_verdict` raise or behave sanely at `correction_k` values that
   make `alpha_used` extremely small (e.g. `k` large enough that `alpha/(2k)` underflows toward
   0) — not expected at this codebase's real scale (`k` tops out at `2*(N-1)` for a 19-model sweep,
   i.e. 36), but worth one boundary-value test given `Fraction` arithmetic is exact and won't
   silently round the way a float would.

**Integration tests on `report.py`:**

5. `rank --reference` against a real continuous-metric pack fixture (mirroring
   `embedder-graphrag-retrieval`'s shape) with `>= 2` candidates renders a reference-family table
   with the correct `k = len(family) * len(candidates)`, the four-state decision vocabulary (§3.4),
   and no crash — the direct regression test for D-1 itself.
6. A synthetic mixed-kind pack fixture (one binary, one continuous `verdictMetrics` member) with
   `--reference` given renders the refusal block (§3.3) and computes **no** verdict for either
   metric — mirroring `compare_report`'s existing `test_mixed_family_refused_whole`-shaped
   coverage, extended to the reference-anchored path.
7. `_rank_resolving_power_lines` returns `[]` for a continuous metric regardless of whether
   `--reference` is given — a direct regression test for §3.5, run against the real
   `embedder-graphrag-retrieval` pack shape (the exact fixture that reproduced §2.3's live bug).
8. The existing binary reference-family behavior (`guard-judge-understanding`) is asserted
   byte-for-byte unchanged before and after this change — a regression pin, not just new-path
   coverage, per the coordination ledger's own stated note
   (`docs/plans/rank-continuous-reference-coordination.md`: "must keep working identically for
   `guard-judge-understanding` — regression coverage, not just new-path coverage").

**Acceptance threshold:** all of 1-8 green, plus a live re-run of
`./run.sh rank --pack embedder-graphrag-retrieval --reference bm25` against the real stored corpus
exits 0 and renders a reference-family table whose `k`, decision states, and CI coverage label are
all manually verifiable against this note's formulas — the same live-verification bar
`docs/test-reports/small-model-benchmarking-manual-additions-report.md` originally applied to find
D-1 in the first place.

## 5. Risks & open questions

- **Open question for the architect, not resolved here:** does `_metric_kind`'s two-argument
  `(a, b, name)` shape extend cleanly to resolving kind across N arms (reference + every
  candidate), or does `rank_report`'s reference path need a small N-ary wrapper around it? §3.3
  step 1 specifies the requirement (check every candidate against the reference, not just one) but
  not the exact function signature — an implementation-level API-shape decision, not a statistical
  one, and I'm deliberately not prescribing it.
- **Open question, same location:** if two arms genuinely disagree on a metric's resolved kind
  (a data-integrity problem DC-10 does not catch, since it only cross-checks one arm's aggregate
  against its own items, never cross-arm agreement) — is that the same "refuse whole" outcome as a
  *pre-registered* mixed-kind family (§3.3), or a sharper failure (closer to the existing
  version/hash/schema mismatch banners, `report.py:1396-1420`, which are "visible, never silent,
  and never a reason to drop a record")? I lean toward the latter — a genuine cross-arm kind
  disagreement means the pack's own schema is self-contradictory for that arm, a more severe
  integrity problem than a version drift — but this is worth an explicit call from the architect
  rather than a silent default either way.
- **Not a risk to this design, but worth stating plainly:** §3.2's `k = len(family) *
  len(candidates)` formula is currently numerically indistinguishable from a per-metric `k =
  len(candidates)` on every pack this codebase ships (every continuous pack has `len(family) ==
  1`). The general formula must still be the one implemented (§3.2's rejected-alternative
  reasoning), precisely so it does not need re-deriving the day a second continuous verdict metric
  ships.
- **Scope boundary, restated:** this note does not design a continuous-metric power-preview
  sentence (§3.5's rejected alternative) or a `k >= 1` defensive guard on `correction_k` (§3.1's
  minor item) — both are legitimate follow-ups but neither blocks D-1's fix, and bundling either
  into this unit risks exactly the kind of scope creep this component's own unit-sizing convention
  warns against.
