"""Confidence intervals, the paired decision instruments, and resolving power.

**This module implements `docs/plans/small-model-benchmarking-ml.md`; there is no other source.**
Every formula, constant, threshold, tolerance, bootstrap parameter and verdict string below is that
note's, cited by section. The plan deliberately does not restate them: two copies of a formula is
one copy and one bug (plan §3.9).

The shape of this module is `-ml` §3.4's **seven** binding rules, written so that "the
anti-conservative version does not typecheck, and the honest one is the only one that runs":

1. `PairedOutcomes.from_units` is the constructor, and a repeated analysis-unit id raises.
2. `resolving_power`'s `unit_kind`/`design_effect`/`basis`/`alpha` are keyword-only with **no
   defaults** — a default of 1.0 would rebuild gate B-1 by omission.
3. Every printed bound takes the rounding direction, the α **and** the denominator that keep *its
   own* claim true, and two bounds side by side routinely take opposite values of the same
   parameter: the MDD ceilings, at `alpha_mdd`, over the floored `n_effective`; the floor
   truncates, at `alpha_family`, over the unfloored one. `n_effective` is a `float`, never an
   observation count, and the `8/n` mnemonic is not code.
4. `verdict()` asserts Rule 4's four preconditions — plus a fifth, that `alpha_step` lies between
   the two αs, which is what makes Rule 7's theorem a checked premise — and refuses rather than
   warns. It lets McNemar decide **only** at
   `design_effect == 1.0 and basis == "by-construction"`. The substitute is
   `paired_cluster_bootstrap`, whose interval responds to the declared design effect, **in
   conjunction with McNemar as a necessary condition** — a veto only ever removes rejections, so
   the pair is at least as conservative as either instrument alone (review B-ML-2).
5. The design effect is a **variance** ratio — the width ratio squared.
6. `cluster_bootstrap` is one level only; a pack needing two must fail validation (S2).
7. No verdict path returns `distinguishable` below `resolving.observable_floor`, enforced in
   `verdict()` against the **exact** floor. It is the only cheap check tying together two
   quantities computed by completely independent routes, which is what makes a substituted
   instrument surface as a contradiction rather than as a plausible number. **The response splits
   by path**: on `mcnemar-exact` the invariant is a theorem, so a fire raises; on
   `conservative-envelope` it is a guard, so a fire demotes and names the floor (m-ML-6).

Two of the three αs the note distinguishes are fields of `ResolvingPower` — `alpha_family` (the
floor's) and `alpha_mdd` (the MDD's). The third, `alpha_step`, is Holm's data-dependent threshold
and is known only after the family is ranked, so it arrives as `verdict()`'s parameter rather than
as a field that would be `None` until it is not (review M-ML-6).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb
from typing import Iterable, Literal, Sequence

#: `-ml` §3.2a, gate M-1. Φ⁻¹(0.975), and what
#: `falkor-chat/server/tests/eval/nlq_scoring.py:59` already pins. `1.96` is a typographic
#: rounding of this number, not a competing convention — but the fixtures in §3.2c are asserted at
#: a tolerance four orders tighter than the divergence, so the exact value is load-bearing.
_Z_95: float = 1.959963984540054

#: `-ml` §3.3/§7.1 — the **unadjusted** α. It is the floor's α whatever *k* is, and the numerator
#: of the family-adjusted `α/k` the MDD takes. One home, because three αs now coexist deliberately
#: and a second literal `0.05` is how they drift apart (review M-ML-6).
ALPHA_FAMILY: float = 0.05

Basis = Literal["by-construction", "measured", "assumed"]
DecidedBy = Literal["mcnemar-exact", "conservative-envelope"]


class DuplicateAnalysisUnit(ValueError):
    """A unit id appeared twice, so the rows are not independent (`-ml` §3.4 Rule 1)."""


class UnattainablePower(ValueError):
    """No effect size reaches the requested power at this `n_effective` (`-ml` §7.1)."""


class Rule7Violation(AssertionError):
    """Rule 7 fired on the `mcnemar-exact` path, where it is a **theorem** (`-ml` v1.6 §3.4).

    Not a `ValueError`: nothing the caller passed is out of contract. With the floor at the
    unadjusted α, `p <= alpha_step <= alpha_family` implies `|b − c| >= b_min(alpha_family)`, so a
    significant McNemar result below the floor cannot come from the data — it can only come from a
    defect in one of the two independent routes that computed the two numbers. That is the
    contradiction Rule 7 exists to surface, and demoting it silently would discard it (m-ML-6).
    """


# --- intervals ----------------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, *, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval — this lab's convention, not Clopper-Pearson (`-ml` §3.2a).

    Descriptive only: it says what one arm scored, and is explicitly **not** the comparison
    instrument. Every rate prints as `k/n = p̂ [lo, hi]`, never a bare percentage.
    """
    if n <= 0:
        raise ValueError("wilson_interval needs a non-zero denominator")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact conditional (binomial-sign) test on the discordant counts (`-ml` §3.2b).

    Exact, not the chi-square approximation, which is invalid at the discordant counts this lab
    will actually see (b+c often < 10).
    """
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(m, i) for i in range(k + 1)) * (0.5**m)
    return min(1.0, 2.0 * tail)


def mover_d_interval(a: int, b: int, c: int, d: int, *, z: float = _Z_95) -> tuple[float, float]:
    """Newcombe's square-and-add (MOVER-D) 95% CI for p1 - p2 on paired binary data (`-ml` §3.2c).

    Reuses the Wilson function, so it is consistent with the per-arm reporting by construction.
    The `max(0.0, …)` clamps are required, not cosmetic: the radicand goes slightly negative under
    extreme `phi`.
    """
    n = a + b + c + d
    if n <= 0:
        raise ValueError("mover_d_interval needs a non-empty table")
    p1, p2 = (a + b) / n, (a + c) / n
    diff = p1 - p2
    l1, u1 = wilson_interval(a + b, n, z=z)
    l2, u2 = wilson_interval(a + c, n, z=z)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = ((a * d - b * c) / den) if den > 0 else 0.0
    lo_rad = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    lo = diff - math.sqrt(max(0.0, lo_rad))
    hi_rad = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    hi = diff + math.sqrt(max(0.0, hi_rad))
    return max(-1.0, lo), min(1.0, hi)


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lo: float
    hi: float
    B: int
    seed: int


def paired_bootstrap(
    diffs: Sequence[float], *, B: int, seed: int, levels: tuple[Fraction, Fraction]
) -> tuple[float, float]:
    """Seeded paired percentile bootstrap on per-item differences (`-ml` §3.2d).

    For continuous metrics the CI **is** the test: the decision is that it excludes zero, and no
    separate significance test is run.

    **This is the engine, not an entry point** (`-ml` §3.4 Rule 4). Wiring a continuous verdict
    straight here yields a correct interval that silently ignores the pack's declared design
    effect; `paired_cluster_bootstrap` is §3.2d's entry point.

    **`levels` is required with no default** (§4 S1e Table G). The two quantiles were fixed at
    2.5/97.5 here, and `-ml` §3.3 rules that an all-continuous family with `k > 1` takes its
    Bonferroni correction **in the interval** — it has nowhere else to put it — so a `k = 3` family
    rendered at the fixed pair is a correction that silently did not happen. At `k = 1` the rule
    returns exactly the pair the code used to hard-code, which is what made this invisible rather
    than urgent. A default that is right at `k = 1` and silently wrong above it is the shape this
    plan already refuses for `designEffect`, `BinaryMetric.unit` and `sampling.seed`, and here the
    wrong value is also the *conventional* one, printing a plausible interval beside a family that
    was never corrected. The levels are exact rationals because `alpha/(2k)` is not a decimal
    (`-ml` §11.2.2); who computes them is `-ml` §3.4 Rule 8's `continuous_verdict()`, from
    `alpha_family` and the family size, and nothing else derives a level.

    A **transposed pair raises**: it is the one error that otherwise returns a plausible
    *inverted* interval that no other check sees (`-ml` §11.2.2), and it is a refusal this
    signature could not carry while the pair was two literals.
    """
    if levels[0] >= levels[1]:
        raise ValueError(
            f"levels must be ordered lower then upper, not {levels!r}: a transposed pair returns "
            "an inverted interval that no other check sees (-ml §11.2.2)"
        )
    if not diffs:
        raise ValueError("paired_bootstrap needs at least one difference")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(B))
    return percentile(means, level=levels[0]), percentile(means, level=levels[1])


def paired_cluster_bootstrap(
    diffs: Sequence[float],
    *,
    design_effect: float,
    B: int,
    seed: int,
    clamp: tuple[float, float] | None,
    levels: tuple[Fraction, Fraction],
) -> tuple[float, float]:
    """The paired-difference interval for a clustered design (`-ml` §3.4 Rule 4's second branch).

    **This is the smallest honest version, not the structurally right one, and the difference is
    worth stating.** `PairedOutcomes` carries one row per analysis unit and no grouping, so there is
    no structure in which a paired resample *of clusters of rows* could be expressed; the grouping
    would have to come from a pack declaring `replicatesPerScript > 1`, which Rule 6 makes a
    validation error while only the one-level `cluster_bootstrap` exists. So the interval is
    computed by resampling the rows and then inflating the percentile half-widths about the point
    estimate by `sqrt(design_effect)`.

    That conversion is exact rather than a fudge: `-ml` §3.4 Rule 5 defines the Kish design effect
    as a **variance** ratio and CI half-width scales as `1/sqrt(n)`, so `sqrt(DEFF)` is precisely
    the factor that carries an interval computed at `n_units` to one computed at
    `n_units / DEFF`. What it cannot do is *discover* clustering the declared `design_effect` did
    not already state — which is why the declaration is a required, defaulted-nowhere input
    (Rule 2) rather than something the resample infers.

    The structurally right version, when a pack that needs it exists: give `PairedOutcomes` an
    optional cluster grouping and resample clusters of paired differences.

    **`clamp` and `levels` are both required with no default** (§4 S1e Tables E and G) and are
    forwarded. This is `-ml` §3.2d's **entry point** for every continuous verdict — MRR, `sep_z` —
    called with the pack's declared `design_effect`, the identity widening at 1.00. Two callers
    state the clamp, on two surfaces that must not be collapsed (`-ml` v1.17 §3.4 Rule 8): on the
    **verdict** path `continuous_verdict()` derives it from the metric's own `support` inside
    itself, so no verdict caller states one; the caller that does is §3.8.1's **exploratory**
    `sep_z` comparison, which reaches this entry point directly, has no metric aggregate to ask,
    and states `clamp=None`.
    """
    if design_effect < 1.0:
        raise ValueError("design_effect must be >= 1.0 (-ml §3.4 Rule 4, precondition 4)")
    lo, hi = paired_bootstrap(diffs, B=B, seed=seed, levels=levels)
    return _widen((lo, hi), sum(diffs) / len(diffs), math.sqrt(design_effect), clamp=clamp)


def _widen(
    interval: tuple[float, float],
    point: float,
    scale: float,
    *,
    clamp: tuple[float, float] | None,
) -> tuple[float, float]:
    """Scale an interval's two half-widths about `point`, then clamp to `clamp` or not at all.

    Shared by both arms of `conservative_envelope`, because Rule 4 requires them widened "the same
    way": an envelope of two intervals scaled about two different centres, or by two different
    factors, is not an envelope of the same quantity — and a clamp is part of "the same way".

    **`clamp` is required with no default, and `None` means do not clamp** (§4 S1e Table E). The
    two bounds were the literals `-1.0` and `1.0`: correct for the difference of proportions the
    envelope was written for, and **wrong for `sep_z`** (`-ml` §5.2), whose per-query differences
    are differences of z-scores and are not bounded by 1. Left as literals, a `sep_z` interval
    whose upper bound exceeds 1 is silently clamped to it and the point estimate can land
    **outside its own interval** — a true decision beside a false number, since the verdict is
    decided by the lower bound. A default of `(-1.0, 1.0)` would fail exactly that way, silently
    and in the direction that prints, which is why this plan refuses one here as it does for
    `designEffect`, `BinaryMetric.unit` and `sampling.seed`.
    """
    lo, hi = interval
    widened = (point - (point - lo) * scale, point + (hi - point) * scale)
    if clamp is None:
        return widened
    return (max(clamp[0], widened[0]), min(clamp[1], widened[1]))


def exact_paired_quantiles(
    table: tuple[int, int, int, int], *, levels: tuple[Fraction, Fraction]
) -> tuple[float, float]:
    """The paired **binary** bootstrap's quantiles in closed form (`-ml` v1.11 §3.4 Rule 4).

    For a paired binary table the per-unit differences take exactly three values, so a resample of
    `n` rows with replacement gives `(N+, N0, N-) ~ Multinomial(n, (b/n, (a+d)/n, c/n))` and the
    resample mean is `S/n` with `S = N+ - N-` on the integer support `[-n, n]`. Every atom's
    probability is a rational with denominator `n**n`, so the whole computation is exact integer
    arithmetic and **the atom is never chosen by a float tie-break** — §11.2.1's bin-edge hazard
    *removed* rather than guarded.

    The estimator is `percentile`'s, applied to the exact distribution where that one applies it
    to an empirical sample: `Q(p) = inf{ s/n : F(s) >= p }`, Hyndman-Fan type 1. That is what
    `-ml` §11.2 reason 2 already claims, so the two agree by construction rather than by
    coincidence, and the selector is one exact integer comparison over the level's numerator and
    denominator (`-ml` v1.18 §11.2.2's closing paragraph).

    **Why this replaces the resample rather than merely speeding it up.** The resampled version is
    a Monte-Carlo estimate of an *atomic* quantile, so wherever the target level lands within
    Monte-Carlo error of an atom boundary the printed bound flips by a whole atom, `1/n_units` —
    and the note measures that reaching the **verdict**, not only the digits: at `(1, 25, 12, 2)`,
    n=40, DEFF 1.2 the shipped code returned *distinguishable* on 80 of 150 seeds and *not* on the
    other 70, and moved on 85/65 of 150 row permutations **at one fixed seed**, so the pack's own
    `sampling.seed` did not make it reproducible. Row order is result-file iteration order and no
    author's choice; `random.Random.choice` draws an *index*, so a permutation of the same multiset
    re-maps a fixed index sequence onto different values.

    **The levels are refused on the same two grounds `percentile` refuses one** — a `float` and a
    level outside `(0, 1]` — through the shared `_check_level`, plus the transposed-pair check
    `paired_bootstrap` also carries. Being the same estimator over a known distribution, it owes a
    caller the same errors, and it was giving an `AttributeError` and an `IndexError` instead
    (review P8-6).

    The levels are the caller's, and on this path they are always `LEVEL_CI95_LO`/`LEVEL_CI95_HI`:
    the paired binary path takes its `k` correction in the Holm ladder and never in the interval
    (§3.3), so no family level reaches here. They are **never** the level a resample happened to
    estimate — the shipped `ordered[int(round(pct/100 * (B - 1)))]` picked the 251st and 9750th of
    10000, i.e. levels `251/10001` and `9750/10001`, and the atom selected differs from the exact
    pair's on 2 `(b, c)` pairs at n=30, 12 at n=40 and 113 at n=85.

    Degenerate input needs no special case: `b = c = 0` gives the single atom `0` and both
    quantiles are `0`. The `O(n**2)` enumeration is 3741 terms at n=85 — measured at 3.2 ms
    against 216 ms for the shipped `B = 10 000` resample.

    **The boundary of applicability is three distinct per-unit values, not "binary versus
    continuous".** The exact enumeration costs `C(n + v - 1, v - 1)` terms for `v` values: `v = 3`
    at n=85 is trivial and `v = 10` at n=38 is 1.6e9 and infeasible. MRR is the trap worth naming
    — its per-item values are discrete, so "it is not continuous" invites the wrong generalisation
    — and its `v` is far above 3. §3.2d's continuous metrics keep the seeded resample.
    """
    for level in levels:
        _check_level(level, caller="exact_paired_quantiles()")
    if levels[0] >= levels[1]:
        raise ValueError(
            f"levels must be ordered lower then upper, not {levels!r}: a transposed pair returns "
            "an inverted interval that no other check sees (-ml §11.2.2)"
        )
    a, b, c, d = table
    n = a + b + c + d
    if n <= 0:
        raise ValueError("the paired table describes no rows")
    zero = a + d
    n_factorial = math.factorial(n)
    atoms: dict[int, int] = {}
    for n_plus in range(n + 1):
        for n_minus in range(n - n_plus + 1):
            n_zero = n - n_plus - n_minus
            weight = n_factorial // (
                math.factorial(n_plus) * math.factorial(n_zero) * math.factorial(n_minus)
            )
            weight *= b**n_plus * zero**n_zero * c**n_minus
            if weight:
                atoms[n_plus - n_minus] = atoms.get(n_plus - n_minus, 0) + weight

    total = n**n
    bounds: list[float] = []
    for level in levels:
        cumulative = 0
        for s in sorted(atoms):
            cumulative += atoms[s]
            if level.denominator * cumulative >= level.numerator * total:
                bounds.append(s / n)
                break
    return bounds[0], bounds[1]


def envelope_arms(
    table: tuple[int, int, int, int], *, design_effect: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The envelope's two arms, `sqrt(DEFF)`-widened about the same point estimate.

    Returns `(mover_d, exact_paired_bootstrap)`. It exists because the report is required to name
    **which arm bound each bound** (`-ml` §3.4 Rule 4) — an audit that is cheap and deterministic
    once the resample is gone, and that makes Rule 4's mixture legible to a reader instead of
    inferable only from the code. Both `conservative_envelope` and `verdict()` take their arms
    from here and compose them through `_compose`, and `verdict()` reads the attribution off the
    same pair, so no caller recomputes another's arithmetic (review P8-5).

    **Rule 4's precondition 4 is checked here** (`-ml` §3.4 Rule 4). It used to be, one layer
    down, in the `paired_cluster_bootstrap` call that v1.11's closed form removed — so it retired
    by accident rather than by Table D, which retires only the `n != len(diffs)` guard. Below 1 a
    design effect *inflates* effective *n* and returns a **narrower** interval on both bounds:
    at `(34, 6, 0, 0)`, DEFF 0.5 gave `[6.6, 25.0] pp` against `[3.2, 29.1]` at 1.00. `verdict()`
    guards it at its own entry point too and names itself when it does, so the two refusals stay
    distinguishable by message; this one is the defence on a public function a caller can reach
    without going through `verdict()` (review P8-3).
    """
    a, b, c, d = table
    n = a + b + c + d
    if n <= 0:
        raise ValueError("the paired table describes no rows")
    if not design_effect >= 1.0:  # NaN-safe: `< 1.0` admits a NaN, which clamps to full support
        raise ValueError("design_effect must be >= 1.0 (-ml §3.4 Rule 4, precondition 4)")
    point = (b - c) / n
    scale = math.sqrt(design_effect)
    # Both arms keep the identical clamp: Rule 4 requires them widened the same way, and a clamp
    # is part of "the same way" (§4 S1e Table E).
    mover = _widen(mover_d_interval(a, b, c, d), point, scale, clamp=(-1.0, 1.0))
    exact = _widen(
        exact_paired_quantiles(table, levels=(LEVEL_CI95_LO, LEVEL_CI95_HI)),
        point,
        scale,
        clamp=(-1.0, 1.0),
    )
    return mover, exact


def _compose(
    mover: tuple[float, float], exact: tuple[float, float]
) -> tuple[float, float]:
    """The envelope of the two arms, taken **bound by bound** (`-ml` §3.4 Rule 4).

    One home for one arithmetic. `conservative_envelope` and `verdict()` both need this pair —
    `verdict()` because it also needs the arms themselves for the attribution and so cannot go
    through `conservative_envelope` — and each spelled the `min`/`max` out for itself. Two copies
    of a formula is one copy and one bug (plan §3.9); it is the same rule that retired the two
    private percentiles, and the same rule applies at four lines as at forty (review P8-5).

    Why bound by bound rather than the wider interval whole: see `conservative_envelope`, which
    carries the `(4, 5, 3, 0)` measurement that decides it.

    **This is where `-ml` v1.19 §3.4 Rule 4a's clamp goes**, and it is not applied here yet. Rule 4a
    rules the `(-1, 1)` support bound a property of the *estimand*, so it belongs on the printed
    interval and not on a composition's inputs: `envelope_arms` widens with `clamp=None` and this
    function clamps its own result, returning `(interval, bound_by)` with Rule 4a's three-token
    set. That edit is P8-1's unit, not P8-5's; the two placements commute exactly, so the printed
    numbers do not move when it lands.
    """
    return min(exact[0], mover[0]), max(exact[1], mover[1])


def conservative_envelope(
    table: tuple[int, int, int, int], *, design_effect: float
) -> tuple[float, float]:
    """The interval printed on any non-`by-construction` path (`-ml` v1.11 §3.4 Rule 4).

    **This replaces the bare percentile interval, and the correction is about *width*, not about
    the decision** (review M-ML-8). B-ML-2's veto fixed what this path *decides* with; what it
    *quantifies* with was left on the resample, and the resample is the **narrower** of the two
    available instruments: measured exactly at n=40 under strict dominance, MOVER-D covers 0.976
    at a mean width of 25.63 pp against the bootstrap's 0.939 at 21.23 pp, and the bootstrap is
    narrower on **100%** of the probability mass (n=30: 0.983/30.60 vs 0.942/24.57; n=85:
    0.969/15.32 vs 0.940/13.56). At `design_effect == 1.0` — *every* comparison until S2's
    determinism probe runs — `sqrt(1.0)` widens nothing, so a path taken **because less is known**
    was printing a tighter effect size than the instrument §3.2c mandates.

    The rule: the **wider** of the `sqrt(DEFF)`-widened exact paired bootstrap and the
    `sqrt(DEFF)`-widened MOVER-D, half-widths scaled about the same point estimate the same way.

    **"Wider" is taken bound by bound, and Rule 4's own stated property is what fixes that
    reading:** the envelope is *"uniformly at least as conservative as either alone"*, which
    picking whichever interval has the larger total width does not deliver — at `(4, 5, 3, 0)`,
    the tool-caller pack's own n, the bootstrap is the wider interval while MOVER-D's lower bound
    is the more conservative one, so choosing one interval whole would print a bound tighter than
    an instrument it is supposed to dominate. Bound by bound it also can only ever *remove*
    rejections, which is what leaves the veto and Rule 7 undisturbed.

    **It takes no `diffs`, no `B` and no seed** *(v1.11, §4 S1e Table D)*. The bootstrap arm is
    now `exact_paired_quantiles`, a function of `(b, c, n)` alone, and MOVER-D of `(a, b, c, d)`
    — so there is no second argument left to disagree with the first, and the `n != len(diffs)`
    guard retires with it: the error it caught is **unrepresentable** rather than checked, which
    is the outcome the note prefers to a guard every time.

    What is claimed and what is not: **neither printed bound is ever tighter than MOVER-D's**, so
    §3.2c's instrument is a *floor on conservatism* rather than the interval itself. The stronger
    v1.8 claim that the envelope "reduces to MOVER-D exactly at DEFF 1.00" is **withdrawn as
    false** — measured, the exact arm binds at least one bound on 83.5% of the 455 tables at n=12,
    87.3% at n=30 and 87.4% at n=40. It is a genuine mixture at every design effect. It also
    removes the sparse-count degeneracy, because MOVER-D covers zero exactly when the counts are
    too thin to support excluding it — at n=30 with `b=4, c=0` the *resample* returned
    `[3.3, 26.7] pp`, excluding zero against an exact p of 0.125, since with four non-zero rows
    `P(no +1 drawn) = (26/30)**30 = 1.4% < 2.5%` makes a 2.5th percentile of zero unreachable.
    """
    return _compose(*envelope_arms(table, design_effect=design_effect))


def cluster_bootstrap(
    units: Sequence[Sequence[bool]], *, B: int = 10_000, seed: int
) -> BootstrapResult:
    """One-level cluster bootstrap: resample the **unit**, never the observation.

    `-ml` §3.4 Rule 6.

    Each inner sequence is the observations belonging to one independent unit. The two-level
    (script → replicate) resample returns the moment a pack declares `replicatesPerScript > 1`, and
    its absence must then be a validation error rather than a silently one-level approximation —
    that check is S2's `validate_pack`.
    """
    flat = [obs for unit in units for obs in unit]
    if not flat:
        raise ValueError("cluster_bootstrap needs at least one observation")
    rng = random.Random(seed)
    n_units = len(units)
    rates: list[float] = []
    for _ in range(B):
        drawn = [obs for _ in range(n_units) for obs in rng.choice(units)]
        rates.append(sum(1 for o in drawn if o) / len(drawn) if drawn else 0.0)
    rates.sort()
    point = sum(1 for o in flat if o) / len(flat)
    return BootstrapResult(
        point=point,
        lo=percentile(rates, level=LEVEL_CI95_LO),
        hi=percentile(rates, level=LEVEL_CI95_HI),
        B=B,
        seed=seed,
    )


# --- `-ml` §11.2: the one percentile in the package -----------------------------------------------

#: `-ml` §11.2.2 — the **whole literal level space**. Four names, so a level is never spelled at a
#: call site and a constant can never be built from a float by accident. The only level that is
#: *computed* rather than named is `-ml` §3.4 Rule 8's family pair `alpha/(2k)`, derived inside
#: `continuous_verdict()` from declared quantities and reaching `paired_bootstrap` as its `levels`.
#: The names are the plan's (§4 S1e Table C), adopted from the note's recommendation; the values,
#: the type and the estimator below are the note's.
LEVEL_P50: Fraction = Fraction(1, 2)
LEVEL_P95: Fraction = Fraction(19, 20)
LEVEL_CI95_LO: Fraction = Fraction(1, 40)
LEVEL_CI95_HI: Fraction = Fraction(39, 40)


def _check_level(level: Fraction, *, caller: str) -> None:
    """`-ml` §11.2.2's two refusals on a **level**, in one home for every estimator that takes one.

    `percentile` and `exact_paired_quantiles` are the same estimator — `inf{ v : F(v) >= p }` —
    over an empirical sample and over the exact multinomial distribution respectively (`-ml` §11.2
    reason 2), so they owe a caller the same two named errors. They did not: in
    `exact_paired_quantiles` a `float` level reached `.numerator` and died with an
    `AttributeError`, and a level above 1 fell off the end of the atom loop and raised
    `IndexError` on `bounds[1]` — two accidents where the note publishes a `TypeError` and a
    `ValueError` (review P8-6). Both are reachable from the signature, which advertises `levels`
    as the caller's; neither is reachable from `envelope_arms`, which hard-codes the pair.

    The `(0, 1]` bound is also what makes `exact_paired_quantiles`' atom loop **total**: at
    `level <= 1` the final cumulative `total` satisfies `den * total >= num * total`, so the loop
    always appends and `bounds[1]` always exists.
    """
    if isinstance(level, float):
        raise TypeError(
            f"{caller} needs an exact rational level, not the float {level!r}: a float level "
            "reopens the bin-edge hazard the integer rank closes, and Fraction(0.05) is the "
            "double's exact value rather than 1/20 (-ml §11.2.1, §11.2.2)"
        )
    if not 0 < level <= 1:
        raise ValueError(f"level must lie in (0, 1], not {level!r} (-ml §11.2)")


def percentile(values: Iterable[float], *, level: Fraction) -> float:
    """The empirical quantile function — Hyndman-Fan type 1 (`-ml` §11.2).

    `P = x(r)`, `r = ceil(level * X)` clamped to `[1, X]` over the ascending sample: the inverse
    of the empirical CDF, `inf{ v : F(v) >= level }`. It is a named, externally documented
    definition — R's `quantile(type = 1)`, NumPy's `method = "inverted_cdf"` — so the
    implementation is checkable against a published rule rather than against a docstring.

    **This is the only percentile or quantile in `modelbench`** (`-ml` §11.10(3), §4 S1e Table C).
    `results.py` imports *this object*; both bootstrap call sites in this module call it. The two
    private copies it replaced computed `int(round(pct/100 * (X - 1)))`, whose `round` is
    half-to-even — so the tie-break direction alternated with the sample size — and having two of
    them is what let `index.csv` report `latencyMsP95` at the 50th percentile and stay green
    (review M27). Two copies of a formula is one copy and one bug (plan §3.9).

    **The level is an exact rational, not a decimal** (`-ml` §11.2.2). A `k`-member continuous
    family's level is `alpha/(2k)` — 12.5 permille at `k = 2`, 8.33... at `k = 3` — and no fixed
    decimal unit expresses that, because the denominator is `40k` and 3 divides one of them. The
    four `LEVEL_*` constants above are the whole literal space.

    **The rank is one integer expression** (`-ml` §11.2.1), over the level's numerator and
    denominator rather than over its float value. Measured in the note over levels `n/1000`,
    `n = 1..999`, at `X <= 3000`: `math.ceil(pct / 100 * X)` — the retired copies' spelling —
    diverges from the integer form on **1626** ranks, and `math.ceil(float(level) * X)` on **755**,
    first at `X = 25`, `level = 7/25`, where `float(Fraction(7, 25)) * 25 == 7.000000000000001`.
    Pin the expression below, never an equivalent-looking one.

    Three refusals, and each is the note's — the first two through `_check_level`, which
    `exact_paired_quantiles` shares because it is the same estimator over a known distribution:

    * a `float` level raises `TypeError` — it reopens the bin-edge hazard the integer rank exists
      to close, and `Fraction(0.05)` is *not* `Fraction(1, 20)` (that one is caught where the level
      is built, `-ml` §11.10(2b), because it is a `Fraction` and arrives here legitimately typed);
    * a level outside `(0, 1]` raises `ValueError`;
    * an empty sample raises `ValueError` — whether a figure exists at all is `latency_summary`'s
      decision (`-ml` §11.6), never a `None` returned from here.

    The input is sorted **here**, on a copy. Requiring a sorted argument is a precondition a caller
    can silently violate, and of the two copies this replaced one sorted and the other did not.
    """
    _check_level(level, caller="percentile()")
    ordered = sorted(values)
    if not ordered:
        raise ValueError(
            "percentile() has no values; whether a figure exists at all is latency_summary's "
            "decision, never a None returned from here (-ml §11.2, §11.6)"
        )
    x = len(ordered)
    rank = max(1, min(x, -(-level.numerator * x // level.denominator)))
    return ordered[rank - 1]


# --- Rule 5: the design effect is a variance ratio ------------------------------------------------


def width_inflation(bootstrap_width: float, naive_width: float) -> float:
    """`bootstrap width ÷ naive Wilson width` — the ratio, which is **not** the design effect."""
    if naive_width <= 0:
        raise ValueError("naive_width must be positive")
    return bootstrap_width / naive_width


def design_effect(bootstrap_width: float, naive_width: float) -> float:
    """Kish design effect: the width ratio **squared** (`-ml` §3.4 Rule 5).

    CI half-width scales as 1/sqrt(n), so a variance ratio is what converts to effective n. v1.1
    called the width ratio itself the design effect; following that literally divides by 2.6 where
    the truth is 7, over-stating effective n by ~2.7x.
    """
    return width_inflation(bootstrap_width, naive_width) ** 2


def effective_n(n_observations: int, design_effect: float) -> float:
    if design_effect <= 0:
        raise ValueError("design_effect must be positive")
    return n_observations / design_effect


# --- Rule 1: the paired table ---------------------------------------------------------------------


@dataclass(frozen=True)
class PairedOutcomes:
    """One row per **independent analysis unit** (`-ml` §3.4 Rule 1).

    The duplicate guard runs in `__post_init__` rather than only in `from_units`, so it holds on
    every construction route. It is a **backstop, not the mechanism**: it fires only when the id it
    is handed is the *cluster* key, and 48 conversation ids drawn from 12 scripts are all unique.
    What actually closes the clustered-design case is the pack's `sampling.analysisUnit`
    declaration (plan §3.3), resolved by `report.py` — see `tests/test_report.py`'s DC-5(c) fixture.
    """

    unit_kind: str
    unit_ids: tuple[str, ...]
    a_correct: tuple[bool, ...]
    b_correct: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not (len(self.unit_ids) == len(self.a_correct) == len(self.b_correct)):
            raise ValueError("unit_ids, a_correct and b_correct must be the same length")
        seen: set[str] = set()
        for unit_id in self.unit_ids:
            if unit_id in seen:
                raise DuplicateAnalysisUnit(
                    f"analysis unit {unit_id!r} appears more than once; the rows are not "
                    "independent, so no paired table may be built from them (-ml §3.4 Rule 1)"
                )
            seen.add(unit_id)

    @classmethod
    def from_units(
        cls, unit_kind: str, rows: Iterable[tuple[str, bool, bool]]
    ) -> "PairedOutcomes":
        """The only constructor. Raises `DuplicateAnalysisUnit` if any unit id repeats."""
        materialized = list(rows)
        return cls(
            unit_kind=unit_kind,
            unit_ids=tuple(r[0] for r in materialized),
            a_correct=tuple(bool(r[1]) for r in materialized),
            b_correct=tuple(bool(r[2]) for r in materialized),
        )

    @property
    def table(self) -> tuple[int, int, int, int]:
        """(a, b, c, d) — b = units A wins, c = units B wins. Only discordants carry information."""
        a = b = c = d = 0
        for a_ok, b_ok in zip(self.a_correct, self.b_correct):
            if a_ok and b_ok:
                a += 1
            elif a_ok:
                b += 1
            elif b_ok:
                c += 1
            else:
                d += 1
        return a, b, c, d

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    def unit_diffs(self) -> list[float]:
        """Per-unit paired difference, +1 / 0 / -1, for the clustered decision path."""
        return [float(a) - float(b) for a, b in zip(self.a_correct, self.b_correct)]


# --- Rules 2 and 3: resolving power ---------------------------------------------------------------


@lru_cache(maxsize=None)
def b_min(alpha: float) -> int:
    """Smallest net win count that can reach significance at `alpha` (`-ml` §3.2b's floor table)."""
    b = 1
    while mcnemar_exact(b, 0) > alpha:
        b += 1
        if b > 10_000:  # pragma: no cover - alpha would have to be absurd
            raise ValueError(f"no attainable b_min at alpha={alpha}")
    return b


def _mcnemar_power(n: int, delta: float, *, alpha: float) -> float:
    """P(reject) under strict dominance (`π_c = 0`), the note's power model (`-ml` §7.1).

    That model is the most favourable case, so every MDD it yields is a lower bound on the
    difference actually needed — which is why the report labels the figure `best case`.
    """
    smallest = b_min(alpha)
    if smallest > n:
        return 0.0
    return sum(comb(n, i) * delta**i * (1 - delta) ** (n - i) for i in range(smallest, n + 1))


def _require_effective(n_effective: float) -> int:
    """`n_effective` is a float, never an observation count (`-ml` §3.4 Rule 2).

    Passing a raw `int` count is a visible mislabel at the call site rather than an invisible one
    inside the function — this is gate B-1's detector, and `min_detectable_difference(48)` printing
    16.7 pp for a pack whose honest figure is 57.8 pp is what it exists to stop.
    """
    if isinstance(n_effective, bool) or not isinstance(n_effective, float):
        raise TypeError(
            "min_detectable_difference takes n_effective: float (units / design effect), not a "
            "raw observation count; a bare n cannot describe a clustered design (-ml §3.4 Rule 2)"
        )
    if n_effective <= 0:
        raise ValueError("n_effective must be positive")
    # The binomial rejection region is defined over whole units. Flooring is the conservative
    # direction (fewer units -> a larger, never a smaller, detectable difference).
    return max(1, math.floor(n_effective))


def min_detectable_difference_exact(
    n_effective: float, *, alpha: float, power: float = 0.80
) -> float:
    """The exact δ at which P(reject) first reaches `power`, unrounded (`-ml` §3.4 Rule 3).

    Raises `UnattainablePower` when no effect size reaches `power` — below `b_min(alpha)` units the
    McNemar rejection region is empty and `_mcnemar_power` is identically zero, at which point the
    bisection would converge on its upper bracket and hand back `1.0`. Returning that figure lets
    the report print *"resolves differences of >=100.0 pp with 80% power"* for a configuration whose
    power is exactly **zero**; `-ml` §7.1's discordance-mix table already calls this state
    **unattainable** (review M-ML-1).
    """
    n = _require_effective(n_effective)
    if _mcnemar_power(n, 1.0, alpha=alpha) < power:
        raise UnattainablePower(
            f"no effect size reaches {power:.0%} power at n_effective={n_effective:g} "
            f"(floored to {n} units, fewer than b_min={b_min(alpha)} at alpha={alpha:g}): the "
            "McNemar rejection region is empty, so power is zero at every difference (-ml §7.1)"
        )
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _mcnemar_power(n, mid, alpha=alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def min_detectable_difference(
    n_effective: float, *, alpha: float, power: float = 0.80, precision: float = 0.001
) -> float:
    """MDD₈₀, exact and **rounded up** to the printed precision (`-ml` §3.4 Rule 3).

    Both halves matter. `8/n` is a rule of thumb, conservative but wrong (20.0 pp where the exact
    answer is 19.046), and printing two numbers for one quantity in one report is the defect.
    Rounding *to nearest* would print 19.0 pp, whose measured power is 0.798 — below the 0.80 the
    sentence claims. Ceiling to 19.1 pp gives 0.8023.
    """
    exact = min_detectable_difference_exact(n_effective, alpha=alpha, power=power)
    steps = math.ceil(exact / precision - 1e-12)
    return min(1.0, steps * precision)


def observable_floor(n_effective: float, *, alpha: float) -> float:
    """`b_min(alpha, c=0) / n_effective` (`-ml` §7.1).

    Below it, no paired binary result reaches significance whatever happens. It takes `alpha`
    because the floor is `6/n` only at α=0.05 — at α=0.025 it is `7/n`.

    **It divides by the unfloored `n_effective`, where `min_detectable_difference` floors first,
    and that asymmetry is deliberate** (review n-ML-1, which asked for one line of consistency;
    this is that line, and it declines to unify them). Rule 3's principle is "round each printed
    bound in the direction that keeps its own claim true", and the two claims point opposite ways:
    the MDD says *"resolves >= X with 80% power"*, so a **larger** X is the conservative error and
    flooring `n` produces it; the floor says *"below Y nothing can reach significance"*, so a
    **smaller** Y is the conservative error and only the unfloored denominator produces it.
    Sharing a denominator would make one of the two anti-conservative. `_require_effective` is
    still called for its type check, whose refusal of a raw `int` is Rule 2's.
    """
    _require_effective(n_effective)
    return b_min(alpha) / n_effective


@dataclass(frozen=True)
class ResolvingPower:
    """The two printed bounds and the parameters that make each one's own sentence true.

    **Three αs coexist deliberately** (`-ml` v1.6 §3.4 Rule 4), and two of them live here because
    two of them are known before the family is ranked:

    * `alpha_family` — the unadjusted α. The **floor's**, because the floor asserts an
      impossibility *at any Holm step the member could face*, so it needs the loosest one.
    * `alpha_mdd` — `α/k`. The **MDD's**, because the MDD promises power *whatever rank the member
      draws*, so it needs the tightest one.

    The third, `alpha_step` (`α/(k−i)`, Holm's actual data-dependent threshold), is **not** a field:
    it is known only after the family's p-values are ranked, which is after this object is built
    and rendered. It reaches the one place that uses it as `verdict()`'s parameter, so the number
    has exactly one home rather than a field that is `None` until it is not.
    """

    n_units: int
    unit_kind: str
    #: the floor's α — unadjusted, whatever *k* is (Rule 3's α row, review M-ML-6)
    alpha_family: float
    #: the MDD's α — `alpha_family / k` for a k-member `verdictMetrics` family (§3.3)
    alpha_mdd: float
    design_effect: float
    basis: Basis
    n_effective: float
    observable_floor: float
    #: `None` when no effect size attains the power at this `n_effective` — fewer than
    #: `b_min(alpha_mdd)` effective units, so the rejection region is empty (review M-ML-1).
    mdd80: float | None
    #: The power the MDD was computed at. A **field**, not a literal in three rendered strings:
    #: `resolving_power` takes `power` as a parameter, so a caller passing 0.90 used to get three
    #: sentences claiming 80% (review n-ML-6). Every one of them renders this instead.
    power: float


def resolving_power(
    n_units: int,
    *,
    unit_kind: str,
    design_effect: float,
    basis: str,
    alpha_family: float,
    alpha_mdd: float,
    power: float = 0.80,
) -> ResolvingPower:
    """Every input but `power` is keyword-only with **no default** (`-ml` §3.4 Rule 2).

    A default of `1.0` for `design_effect` would rebuild gate B-1 by omission: the caller who
    forgets clustering is exactly the caller the gate found. **Both αs are inputs for the same
    reason** — defaulting either one is how the floor came to be computed at `α/k` (M-ML-6).

    The two bounds take **opposite** αs and that is Rule 3's principle, not an inconsistency:
    `observable_floor` is called with `alpha_family`, `min_detectable_difference` with `alpha_mdd`.
    `observable_floor` keeps its own `alpha` parameter — `b_min` is a function of α in general, not
    the constant 6 — so the formula stays general while the *choice* lives here, in one auditable
    place.
    """
    # Rule 2's bound, at construction (reviews P3-11, n-ML-5). `<= 0` was the old check: it caught
    # the `ZeroDivisionError` and let `DEFF = 0.5` through, which *doubles* `n_effective` and
    # shrinks **both** printed bounds. `verdict()` and `paired_cluster_bootstrap` both refuse below
    # 1.0, so the value was refused one layer later — after the anti-conservative figure had been
    # computed and could be rendered. In a module whose stated shape is "the anti-conservative
    # version does not typecheck", the bound belongs where the object is made. (`effective_n` keeps
    # the looser `> 0`: it is Rule 5's arithmetic, not a printed bound, and the rho=1 identity test
    # exercises it directly.)
    if not design_effect >= 1.0:  # NaN-safe: `< 1.0` would admit a NaN design effect
        raise ValueError(
            f"design_effect must be >= 1.0, not {design_effect!r}; a design effect below 1 "
            "inflates the effective sample and shrinks both printed bounds (-ml §3.4 Rule 2)"
        )
    n_eff = n_units / design_effect
    try:
        mdd80: float | None = min_detectable_difference(n_eff, alpha=alpha_mdd, power=power)
    except UnattainablePower:
        mdd80 = None
    return ResolvingPower(
        n_units=n_units,
        unit_kind=unit_kind,
        alpha_family=alpha_family,
        alpha_mdd=alpha_mdd,
        design_effect=design_effect,
        basis=basis,  # type: ignore[arg-type]
        n_effective=n_eff,
        observable_floor=observable_floor(n_eff, alpha=alpha_family),
        mdd80=mdd80,
        power=power,
    )


# --- Rule 4: the verdict --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    metric_name: str
    distinguishable: bool
    text: str
    diff: float
    ci: tuple[float, float]
    mcnemar_p: float
    b: int
    c: int
    decided_by: DecidedBy
    #: Which arm of the conservative envelope bound the (lower, upper) printed bound — the audit
    #: `-ml` §3.4 Rule 4 puts in the `- decided by:` bullet's place once the seed parenthetical is
    #: gone. `None` on `mcnemar-exact`, where one instrument produced the whole interval.
    bound_by: tuple[str, str] | None
    marginal_overlap: bool
    alpha_used: float
    floor_demoted: bool = False
    holm_tested: bool = True


def _pp(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}"


def format_floor_pp(value: float, *, precision: float = 0.001) -> str:
    """Render the observable floor in percentage points, **truncated** (`-ml` §3.4 Rule 3).

    One principle, two directions: round each printed bound in the direction that keeps its own
    claim true. The floor carries *"differences below Y cannot reach significance at any observed
    outcome"*, so it must truncate — at n=38 the exact floor is `6/38 = 15.789` pp and printing the
    ceiling `15.8` makes the sentence false, because the attainable observed difference `6/38` is
    below 15.8 and *does* reach significance (b=6, c=0, p=1/32). The MDD rounds the other way, and
    `min_detectable_difference` does that ceiling on the value itself.

    Truncation is **guarded**, mirroring the `- 1e-12` on the ceiling side, because a floor that
    lands exactly on a 0.1 pp bin edge is the common case rather than the exotic one: `7/40 = 0.175`
    is `174.99999999999997` bins in IEEE doubles, so naive truncation prints `17.4` where the value
    is 17.5.

    **That cell was the α=0.025 floor column, which v1.6 deleted** (review M-ML-6), so the guard is
    now **defensive rather than load-bearing on any figure the note prints**: swept over `n <= 2000`
    at the floor's `b_min = 6`, naive truncation never misfires. It is kept because `b_min` is a
    function of α, not the constant 6 — any future α reopens the hazard, and the cost is one term.
    The justification outliving the thing that justified it is itself the point: both changes
    arrived in the same review pass.

    The expression is load-bearing even where the guard is not. `math.floor(x / precision)` and
    `math.floor(x * 1000)` are **not** interchangeable — the double nearest `0.001` is slightly
    above it, so the division rounds down across the bin edge where the multiplication does not —
    and a sweep run against the expression the code does not use is how this was missed the first
    time.

    This is the presentation layer only. `ResolvingPower.observable_floor` stays exact, because
    Rule 7 compares against it and an invariant must not inherit the printer's rounding.
    """
    steps = math.floor(value / precision + 1e-12)
    return f"{steps * precision * 100:.1f}"


def _plural(unit_kind: str) -> str:
    return f"{unit_kind}s"


def provenance(resolving: ResolvingPower, unit_plural: str) -> str:
    """§7.1's mandatory parenthetical — the unit, the raw unit count, the design effect and its
    basis are what make a resolving-power figure auditable, and a bare `n` is the shape gate B-1
    rejected.

    The α it names is **`alpha_mdd`**, because it sits inside the MDD's sentence; the floor's α is
    named by `floor_clause`, in the floor's own sentence. Public because `report.py` renders the
    same parenthetical: two copies of this string is one copy and one drift, and the copy is how
    `report.py` came to print `alpha=` from a field that no longer exists.
    """
    return (
        f"n={resolving.n_effective:g} effective {unit_plural} ({resolving.n_units} units, "
        f"design effect {resolving.design_effect:.2f}, {resolving.basis}, "
        f"alpha={resolving.alpha_mdd:g})"
    )


def unattainable_clause(resolving: ResolvingPower, unit_plural: str) -> str:
    """What replaces the MDD sentence when `mdd80` is `None` (review M-ML-1).

    Below `b_min(alpha_mdd)` effective units the rejection region is empty: power is zero at every
    difference, so quoting the figure prints a number the instrument cannot deliver — the delivered
    build printed *"resolves >=100.0 pp with 80% power"* for a configuration whose power is zero.

    It says nothing about the **floor**, which takes the other α and can still be attainable when
    this is not: at k=2 and `n_eff` in `[6, 7)` the rejection region is empty at α/k = 0.025 while a
    92.3 pp difference *does* reach significance at a 0.05 Holm step. `floor_clause` is the one
    that speaks for the floor, and it is printed either way.
    """
    return (
        f"No difference is resolvable at {provenance(resolving, unit_plural)}: that is fewer than "
        f"the b_min={b_min(resolving.alpha_mdd)} net wins any outcome must reach at that alpha, so "
        f"no effect size attains {resolving.power:.0%} power."
    )


def mdd_clause(resolving: ResolvingPower, unit_plural: str) -> str:
    """§7.1's MDD sentence, without its closing clause or its full stop.

    Public, and public for the same reason `provenance`, `floor_clause` and `unattainable_clause`
    are: `report.py` renders this stem too, in the standalone resolving-power line, and it spelled
    it out in full — the one string of the four left with two homes, in the module whose own
    docstring says *"two copies of this string is one copy and one drift"* (review m-ML-8). The
    drift was scheduled rather than hypothetical: M-ML-7's fix edits exactly this string.

    Callers add the punctuation their sentence needs: a full stop where the line stands alone, or
    `-ml` §3.2e verdict 2's closing clause where an observed difference is being compared to it.
    """
    if resolving.mdd80 is None:  # pragma: no cover - callers branch on `mdd80` before arriving
        raise ValueError("no MDD exists at this n_effective; render `unattainable_clause` instead")
    return (
        f"This pack resolves differences of >={_pp(resolving.mdd80)} pp with "
        f"{resolving.power:.0%} power at {provenance(resolving, unit_plural)}"
    )


def _mdd_clause(resolving: ResolvingPower, unit_plural: str, diff: float, b: int, c: int) -> str:
    """`-ml` v1.7 §3.2e verdict 2's second half, or its unattainable replacement.

    **The closing clause is conditional, and that is the note's rule, not a preference** (review
    M-ML-7 / P3-2). `the observed X pp is below that` was fixed prose and is false whenever
    `|diff| >= mdd80` — §7.1's *normal case for a model swap*, a candidate that wins more than it
    loses without strictly dominating: measured, 268 of the 1 580 by-construction tables that
    print this clause printed it falsely. The alternate wording is v1.7's, verbatim, **discordance
    counts included** — they are the reason the two numbers point opposite ways, and without them
    the sentence looks like the instrument contradicting itself.

    **The comparative is *"at or above"*, not *"above"* (v1.8 §3.2e, review m-ML-9).** The branch
    is `|diff| < mdd80`, so **equality takes this wording** — the right branch, since the stem
    claims the pack resolves differences of `>=` `mdd80`. But `mdd80` is ceilinged onto a 0.1 pp
    grid while `|diff|` is `(b-c)/n`, so the two coincide wherever grid and lattice meet, and that
    is reachable at this component's own sample sizes: `n_units=85, k=2, DEFF=1.9` gives
    `mdd80 = 20.0 pp` and `|b-c| = 17` is exactly 20.0 pp — the guard-judge pack's own n. A strict
    *"is above that"* is then false in the same way *"is below that"* was, one branch over. One
    comparative true across the whole branch, never a third string to keep in step with two others.
    """
    if resolving.mdd80 is None:
        return unattainable_clause(resolving, unit_plural)
    stem = mdd_clause(resolving, unit_plural)
    if abs(diff) < resolving.mdd80:
        return f"{stem}; the observed {_pp(abs(diff))} pp is below that."
    return (
        f"{stem}; the observed {_pp(abs(diff))} pp is at or above that, but the MDD assumes strict "
        f"dominance and this comparison is not strictly dominant (b={b}, c={c}), so the difference "
        f"required for {resolving.power:.0%} power at this discordance mix is larger (§7.1)."
    )


def floor_clause(resolving: ResolvingPower) -> str:
    """The floor half of §7.1's mandatory line, capitalised by the caller where it opens a sentence.

    It names **`alpha_family`**, not the α beside the MDD in the same line, because that is the
    only α at which its own sentence is true: the claim is *"nothing can reach significance at any
    observed outcome"*, and a member can face any Holm step up to the unadjusted α (`-ml` v1.6
    §7.1, review M-ML-6). A reader shown one α cannot tell which bound it governs, so both are
    printed and each is attached to the bound it belongs to.

    Above 100 pp the figure itself is not printable — no observed difference can exceed it — and
    the sentence says that instead of quoting a number like `105.0 pp` that reads as a threshold.
    """
    holm = f"at any Holm step (alpha <= {resolving.alpha_family:g})"
    if resolving.observable_floor > 1.0:
        return (
            f"no observed difference can reach significance {holm}: the floor of "
            f"{b_min(resolving.alpha_family)} net wins exceeds the {resolving.n_effective:g} "
            "effective units available"
        )
    if resolving.design_effect > 1.0:
        # `-ml` v1.8 §7.1's qualifier, published verbatim (review m-ML-11). Above DEFF 1 this
        # sentence and the McNemar p three lines from it live on **different denominators** — the
        # floor on `n_eff`, the p on the raw rows — so the template alone reads as a flat
        # contradiction of the number beside it: at n_units=12, DEFF=2 the floor prints 100.0 pp
        # while `b=11, c=0` is 91.7 pp at p=0.00098. Both are *decided* right (Rule 7 demotes);
        # the resolution — McNemar over raw rows is anti-conservative under clustering, so the
        # floor over effective units governs — appeared nowhere in the rendered text. At DEFF 1.00
        # the denominators coincide, the qualifier is noise, and §7.2's rendered line does not move.
        return (
            f"differences below {format_floor_pp(resolving.observable_floor)} pp cannot reach "
            f"significance at any observed outcome over the {resolving.n_effective:g} effective "
            f"{_plural(resolving.unit_kind)} this design supports, {holm} — the McNemar p printed "
            f"beside it is computed over the {resolving.n_units} raw rows, where clustering makes "
            "it anti-conservative, so it can fall below that alpha without lifting the difference "
            "above this floor"
        )
    return (
        f"differences below {format_floor_pp(resolving.observable_floor)} pp cannot reach "
        f"significance at any observed outcome, {holm}"
    )


def verdict(
    outcomes: PairedOutcomes,
    *,
    resolving: ResolvingPower,
    metric_name: str,
    family: Sequence[str],
    a_label: str = "A",
    b_label: str = "B",
    alpha_step: float | None = None,
    holm_tested: bool = True,
) -> Verdict:
    """Decide one pre-registered metric, or refuse (`-ml` §3.4 Rule 4).

    Raises — never warns, never silently proceeds — unless all of Rule 4's four preconditions
    hold, and unless `alpha_step` lies in `[alpha_mdd, alpha_family]` (Rule 7's premise). McNemar
    exact decides and MOVER-D quantifies **only** at `design_effect == 1.0` and
    `basis == "by-construction"`; otherwise McNemar is anti-conservative, must not decide, and the
    conservative envelope on the paired difference takes its place.

    `alpha_step` carries §3.3's Holm–Bonferroni step for this metric when the caller is testing a
    family; it defaults to `resolving.alpha_mdd`, the tightest step. `holm_tested` is
    the other half of Holm: `False` means a metric ranked ahead of this one failed its own step, so
    Holm stops and this member is **not tested at all**, whatever its own p-value. The
    `resolving.alpha_mdd == resolving.alpha_family / len(family)` precondition is unaffected
    either way.

    **Rule 7 is enforced here, on every path**, not left to a test the implementer may or may not
    write: no verdict returns `distinguishable` when `|diff|` is below `resolving.observable_floor`.
    """
    if resolving.n_units != outcomes.n_units:
        raise ValueError(
            f"resolving power describes {resolving.n_units} units but the paired table has "
            f"{outcomes.n_units}; the printed figure must belong to this table"
        )
    if resolving.unit_kind != outcomes.unit_kind:
        raise ValueError(
            f"unit kind mismatch: resolving power is per {resolving.unit_kind!r}, the table is per "
            f"{outcomes.unit_kind!r}"
        )
    if metric_name not in family:
        raise ValueError(
            f"{metric_name!r} is not in the pre-registered family {list(family)}"
        )
    if abs(resolving.alpha_mdd - resolving.alpha_family / len(family)) > 1e-12:
        raise ValueError(
            f"a {len(family)}-member verdict family must report its MDD at alpha="
            f"{resolving.alpha_family / len(family)}, not {resolving.alpha_mdd} (-ml §3.3). This "
            "is the *pre-registration* alpha and it is unchanged by v1.6: the floor moved to "
            "alpha_family, the MDD did not (-ml §3.4 Rule 4, precondition 3)"
        )
    if resolving.design_effect < 1.0:
        # Rule 4's precondition 4, checked **here and before any instrument is selected**. It is
        # not redundant with `paired_cluster_bootstrap`'s identical bound: that one fires only on
        # the path that reaches the resample, and it fires after the branch has already been taken
        # — so a `verdict()` missing this check raised the *same sentence* from one layer down and
        # was indistinguishable in a test that only read the message (review P3-11, and why this
        # one names its own function).
        raise ValueError(
            "verdict() precondition 4: resolving.design_effect must be >= 1.0, not "
            f"{resolving.design_effect!r} (-ml §3.4 Rule 4)"
        )
    if alpha_step is not None and not (
        resolving.alpha_mdd - 1e-12 <= alpha_step <= resolving.alpha_family + 1e-12
    ):
        # Holm's steps are `alpha_family / (k - i)`, which lie between the two αs by construction.
        # Checking it is what makes Rule 7's McNemar branch a theorem rather than an assumption:
        # the premise `alpha_step <= alpha_family` is exactly what implies `|b - c| >= b_min`, so
        # an out-of-range step would surface as a `Rule7Violation` — a module bug raised at the
        # wrong module (`-ml` v1.6 §3.4 Rules 4 and 7).
        raise ValueError(
            f"alpha_step={alpha_step} is outside [{resolving.alpha_mdd}, "
            f"{resolving.alpha_family}]; a Holm step is alpha_family/(k-i) and cannot leave that "
            "range (-ml §3.3)"
        )

    a, b, c, d = outcomes.table
    n = a + b + c + d
    diff = (b - c) / n
    p = mcnemar_exact(b, c)
    alpha = resolving.alpha_mdd if alpha_step is None else alpha_step

    # FR-15's literal marginal-overlap check, computed and printed as a diagnostic. It costs
    # nothing and the requirement asks for it, but it is inert in this lab's regime (`-ml` §3.1).
    m1 = wilson_interval(a + b, n)
    m2 = wilson_interval(a + c, n)
    overlap = not (m1[1] < m2[0] or m2[1] < m1[0])

    mcnemar_may_decide = resolving.design_effect == 1.0 and resolving.basis == "by-construction"

    if mcnemar_may_decide:
        ci = mover_d_interval(a, b, c, d)
        decided_by: DecidedBy = "mcnemar-exact"
        bound_by: tuple[str, str] | None = None
        raw_significant = p <= alpha
    else:
        # **The envelope, not the resample alone** (`-ml` v1.8 §3.4 Rule 4, review M-ML-8). The
        # veto below fixed what this path *decides* with; the interval it *quantifies* with was
        # the narrower of the two available instruments, on a path taken because less is known.
        #
        # **And no seed reaches here any more** (`-ml` v1.11 Rule 4, §4 S1e Table D). The
        # bootstrap arm is exact, so the interval is a function of the table alone: the arms are
        # taken once here rather than through `conservative_envelope`, because Rule 4 also
        # requires the report to name **which arm bound each bound** and recomputing them in the
        # renderer would be a second home for one arithmetic. The *composition* is not a second
        # home either: `_compose` is the one spelling, shared with `conservative_envelope`
        # (review P8-5).
        mover_arm, exact_arm = envelope_arms((a, b, c, d), design_effect=resolving.design_effect)
        ci = _compose(mover_arm, exact_arm)
        bound_by = (
            "MOVER-D" if mover_arm[0] <= exact_arm[0] else "exact paired bootstrap",
            "MOVER-D" if mover_arm[1] >= exact_arm[1] else "exact paired bootstrap",
        )
        decided_by = "conservative-envelope"
        # **A conjunction, not the interval alone** (review B-ML-2). At `design_effect == 1.0` with
        # a non-`by-construction` basis — the fail-safe every comparison carries until the
        # determinism probe runs, i.e. the *default* path — `sqrt(1.0)` widens nothing and a bare
        # percentile interval was deciding: measured at n=40, `(b=7, c=1)`, `(9, 2)` and `(11, 3)`
        # all rendered distinguishable at p = 0.057-0.070, where the exact test refuses, and all
        # three sit at or above the floor so Rule 7 does not catch them.
        #
        # Using McNemar as a **veto** does not violate Rule 4, and the note says so explicitly: the
        # objection to McNemar under clustering is that it *rejects* too readily, and a necessary
        # condition can only ever *remove* rejections. So the conjunction is uniformly at least as
        # conservative as either instrument alone — at DEFF = 1 it restores the exact test's
        # calibration exactly, and at DEFF > 1 the widened interval remains the binding constraint
        # it already was. What Rule 4 forbids is McNemar deciding *for* distinguishability under
        # clustering; withholding that verdict is the opposite operation.
        raw_significant = (ci[0] > 0 or ci[1] < 0) and p <= alpha

    # `-ml` v1.6 §3.4 Rule 7 — the observable floor is a property of the *decision*, on every
    # path, and it is compared against the exact float, never `format_floor_pp`'s truncation: an
    # invariant that inherits the presentation layer's rounding can fire, or fail to fire, by
    # 0.05 pp.
    #
    # **The response splits by path, because the invariant has two different statuses** (m-ML-6):
    #
    # * on `mcnemar-exact` it is a **theorem** — with the floor at `alpha_family`, significance at
    #   any Holm step implies `|b - c| >= b_min(alpha_family)`, exhaustively over every `(b, c)`
    #   with `b + c <= 400` — so a fire is a defect in this module and must raise. Demoting it
    #   silently would discard exactly the detector property this rule exists for.
    # * on `conservative-envelope` it is a **guard** — a widened interval and a shrunken
    #   can legitimately disagree (at DEFF=2 on `(34, 6, 0, 0)` the interval still excludes zero
    #   at 15.0 pp while the floor has moved to 30.0 pp) — so a fire demotes and names the floor as
    #   the reason. Raising there would abort on ordinary clustered data.
    #
    # The converse is deliberately not enforced either way: `|diff| >= floor` does not imply
    # distinguishable, because significance depends on the discordance split rather than on
    # `b - c` (§3.2c row 4 is the counterexample).
    below_floor = abs(diff) < resolving.observable_floor
    # **The raise and the demotion take different conditions** (review n-ML-8). One `fired` gated
    # both on `holm_tested`, which is not one of the theorem's premises: the theorem is
    # `p <= alpha_step <= alpha_family` implies `|b - c| >= b_min`, and it holds whether or not
    # Holm reached this member — so a genuine module bug went undetected for every member past the
    # stop. `holm_tested` belongs to the *demotion*, where "this member was not tested" is a real
    # reason not to rank it; it has no business suppressing a contradiction between two numbers
    # computed by independent routes.
    if raw_significant and below_floor and decided_by == "mcnemar-exact":
        raise Rule7Violation(
            f"Rule 7 fired on the mcnemar-exact path: p={p:g} <= alpha_step={alpha:g} with "
            f"|diff|={abs(diff):g} below the observable floor {resolving.observable_floor:g} "
            f"(b={b}, c={c}, n={n}). That is impossible for consistent inputs — significance at "
            f"any step up to alpha_family={resolving.alpha_family:g} implies "
            f"|b - c| >= b_min={b_min(resolving.alpha_family)} — so one of the two routes that "
            "produced these numbers is wrong (-ml v1.6 §3.4 Rule 7)"
        )
    floor_demoted = raw_significant and holm_tested and below_floor
    significant = raw_significant and holm_tested and not below_floor

    ci_excludes_zero = ci[0] > 0 or ci[1] < 0
    unit_plural = _plural(resolving.unit_kind)

    if significant:
        # The interval is oriented A-minus-B. When B is the winner the sentence is written
        # winner-first, so the interval must be flipped with it: a positive effect printed beside a
        # wholly negative interval is an internally contradictory line, and this is a measuring
        # instrument.
        winner, loser = (a_label, b_label) if diff >= 0 else (b_label, a_label)
        shown_ci = ci if diff >= 0 else (-ci[1], -ci[0])
        text = (
            f"{winner} is better than {loser} on {metric_name}: +{_pp(abs(diff))} pp "
            f"(95% CI [{_pp(shown_ci[0])}, {_pp(shown_ci[1])}] pp), "
            f"n={n} paired {unit_plural} (unit: {resolving.unit_kind}, design effect "
            f"{resolving.design_effect:.2f}), McNemar exact p={p:.3f} (b={b}, c={c})."
        )
    elif not holm_tested:
        text = (
            f"Not distinguishable at this sample size. Not tested: Holm–Bonferroni stops at the "
            f"first non-rejection in the pre-registered family, and a metric ranked ahead of "
            f"{metric_name} did not clear its own step. McNemar exact p={p:.3f} (b={b}, c={c}) is "
            "printed without a significance claim (§3.3). Neither model is ranked above the other."
        )
    elif floor_demoted:
        # Only the substitute path reaches here: on `mcnemar-exact` a fire raises (m-ML-6), so
        # there is no branch for an instrument name that cannot occur.
        text = (
            f"Not distinguishable at this sample size. The conservative envelope interval "
            f"[{_pp(ci[0])}, {_pp(ci[1])}] pp excludes zero, but the observed "
            f"{_pp(abs(diff))} pp is below this pack's observable floor: "
            f"{floor_clause(resolving)}. An interval alone cannot support a claim the "
            "sample size cannot reach, so the floor decides (-ml §3.4 Rule 7). Neither model is "
            "ranked above the other."
        )
    elif ci_excludes_zero:
        # `-ml` §3.2e verdict 3, "real and not rare" — and on the substitute path it is also where
        # the veto lands, so the closing clause has to say which of the two roles the exact test is
        # playing. On `mcnemar-exact` it is the decision rule; on `conservative-envelope` it is a
        # necessary condition and the interval is the decision rule, which the trailing label then
        # states in full. One sentence for both would contradict one of them.
        role = (
            "the exact test is the decision rule"
            if decided_by == "mcnemar-exact"
            else "on this path the exact test is a necessary condition, and it is not met"
        )
        # `-ml` §3.2e's wording names the MOVER-D interval by its job; the substitute path names
        # the instrument, as the floor-demotion string beside it already does.
        named = (
            "effect-size" if decided_by == "mcnemar-exact" else "conservative envelope"
        )
        text = (
            f"Not distinguishable at this sample size. The {named} interval "
            f"[{_pp(ci[0])}, {_pp(ci[1])}] pp excludes zero but the exact paired test does not "
            f"reach alpha={alpha:g} (b={b}, c={c}, p={p:.3f}). Reported as not distinguishable: "
            f"{role}."
        )
    else:
        text = (
            f"Not distinguishable at this sample size. Observed difference {diff * 100:+.1f} pp, "
            f"95% CI [{_pp(ci[0])}, {_pp(ci[1])}] pp covers zero (b={b}, c={c}, McNemar exact "
            f"p={p:.3f}). {_mdd_clause(resolving, unit_plural, diff, b, c)} "
            "Neither model is ranked above the other."
        )

    if decided_by == "conservative-envelope":
        # Every string on this path carries the label, not two of the five: a reader who sees only
        # one verdict must still be told which instrument produced it (`-ml` §3.4 Rule 4).
        #
        # **What the widening clause says is conditional on whether a widening happened** (review
        # P3-3). `sqrt(DEFF)` is 1.00 at `design_effect == 1.0` — nothing was widened and no
        # clustering was declared — and that is the path *every* comparison carries until S2's
        # determinism probe lands, so the fixed prose *"for the declared clustering"* was false on
        # the default path and true only on the rare one. What actually displaced McNemar there is
        # Rule 4's other half, the `basis`: the design effect is asserted rather than established
        # by construction. `-ml` §3.4 Rule 4 requires "the design effect and its basis printed" on
        # this path; naming the basis here is what discharges the second half of that (the
        # provenance parenthetical prints it three lines away, in the MDD's sentence, where a
        # reader has no reason to read it as the reason the instrument changed).
        if resolving.design_effect > 1.0:
            text += (
                " Decided by the conservative envelope on the paired difference — MOVER-D and "
                "the exact paired bootstrap, the wider of the two at each bound — widened by "
                f"sqrt(DEFF)={math.sqrt(resolving.design_effect):.2f} for the declared "
                f"clustering, in conjunction with McNemar's exact test (p={p:.3f}) as a necessary "
                "condition: under clustering McNemar rejects too readily, so it may withhold a "
                "verdict but never carries one on its own."
            )
        else:
            # **The rationale differs, not just the widening clause** (v1.8 §3.2e(f) variant 2,
            # review m-ML-10). The shipped sentence kept *"under clustering McNemar rejects too
            # readily"* on a comparison that declares **no** clustering — P3-3's own defect
            # surviving in the half P3-3 did not touch, the rationale rather than the widening —
            # and its `because` clause attached, on the nearest-attachment parse, to the widening
            # instead of to the instrument choice. What displaced McNemar is the **basis**; what
            # left sqrt(DEFF) at 1.00 is the **design effect**. Two different causes, so the
            # published variant states each where it belongs.
            text += (
                " Decided by the conservative envelope on the paired difference — MOVER-D and "
                "the exact paired bootstrap, the wider of the two at each bound — the instrument "
                f"here because this comparison's design effect is {resolving.basis} rather than "
                "established by construction — with no widening applied (sqrt(DEFF)=1.00), in "
                f"conjunction with McNemar's exact test (p={p:.3f}) as a necessary condition: a "
                "design effect that was never established cannot license the exact test to carry "
                "a verdict, so it may withhold one but never carries one on its own."
            )

    return Verdict(
        metric_name=metric_name,
        distinguishable=significant,
        text=text,
        diff=diff,
        ci=ci,
        mcnemar_p=p,
        b=b,
        c=c,
        decided_by=decided_by,
        bound_by=bound_by,
        marginal_overlap=overlap,
        alpha_used=alpha,
        floor_demoted=floor_demoted,
        holm_tested=holm_tested,
    )


@dataclass(frozen=True)
class HolmStep:
    """One family member's Holm–Bonferroni outcome (§3.3)."""

    p: float
    rank: int
    threshold: float
    tested: bool
    rejected: bool


def holm_steps(p_values: Sequence[float], *, alpha: float) -> list["HolmStep"]:
    """Holm–Bonferroni across the pre-registered family, in the caller's order (§3.3).

    **`alpha` is keyword-only with no default** (reviews P3-13, n-ML-4, plan §4 S1 at v1.7). It
    carried `= 0.05`, a second literal of the number `ALPHA_FAMILY` exists to be the single home
    of — in the module whose constant docstring says *"a second literal `0.05` is how they drift
    apart"*. `report.py` always passed `pack.metrics.alpha_family`, so the default was unreachable,
    which is what would have let it rot: the family α has one home, and no caller inherits it by
    omission.

    Order the p-values; test the smallest at α/k, the next at α/(k−1), … **stopping at the first
    non-rejection** — every later member is then `tested=False` and can never be rejected, whatever
    its own p-value. That stop is the half the delivered build omitted: without it
    `holm_steps([0.30, 0.04])` prints a 0.05 threshold beside p=0.30 and a reader applying the
    printed rule concludes the opposite of the verdict (B-1, M-ML-2).

    The `threshold` is still reported for a member past the stop, because §3.3 requires the adjusted
    threshold printed beside each p-value and that is what makes the correction auditable; `tested`
    is what says whether the comparison against it means anything.
    """
    k = len(p_values)
    order = sorted(range(k), key=lambda i: p_values[i])
    # Keyed by index and re-read in the caller's order, rather than filtered out of a placeholder
    # list: the old `[s for s in steps if s is not None]` was a type narrowing, and it made a
    # ladder that ever returned short do so **silently** — into a consumer that zipped it
    # un-`strict` against the metric tables, dropping a pre-registered verdict metric from the
    # report with no error (review P2-3). This comprehension cannot shorten; a missing rank is a
    # `KeyError` here rather than an absent row three functions later.
    by_index: dict[int, HolmStep] = {}
    stopped = False
    for rank, idx in enumerate(order):
        threshold = alpha / (k - rank)
        rejected = (not stopped) and p_values[idx] <= threshold
        by_index[idx] = HolmStep(
            p=p_values[idx], rank=rank, threshold=threshold,
            tested=not stopped, rejected=rejected,
        )
        if not stopped and not rejected:
            stopped = True
    return [by_index[i] for i in range(k)]
