"""Confidence intervals, the paired decision instruments, and resolving power.

**This module implements `docs/plans/small-model-benchmarking-ml.md`; there is no other source.**
Every formula, constant, threshold, tolerance, bootstrap parameter and verdict string below is that
note's, cited by section. The plan deliberately does not restate them: two copies of a formula is
one copy and one bug (plan §3.9).

The shape of this module is `-ml` §3.4's six binding rules, written so that "the anti-conservative
version does not typecheck, and the honest one is the only one that runs":

1. `PairedOutcomes.from_units` is the constructor, and a repeated analysis-unit id raises.
2. `resolving_power`'s `unit_kind`/`design_effect`/`basis`/`alpha` are keyword-only with **no
   defaults** — a default of 1.0 would rebuild gate B-1 by omission.
3. `min_detectable_difference` is computed exactly over the McNemar rejection region and rounded
   **up**; the `8/n` mnemonic is not code, and `n_effective` is a `float`, never an observation
   count.
4. `verdict()` asserts four preconditions and refuses, and lets McNemar decide **only** at
   `design_effect == 1.0 and basis == "by-construction"`.
5. The design effect is a **variance** ratio — the width ratio squared.
6. `cluster_bootstrap` is one level only; a pack needing two must fail validation (S2).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Iterable, Literal, Sequence

#: `-ml` §3.2a, gate M-1. Φ⁻¹(0.975), and what
#: `falkor-chat/server/tests/eval/nlq_scoring.py:59` already pins. `1.96` is a typographic
#: rounding of this number, not a competing convention — but the fixtures in §3.2c are asserted at
#: a tolerance four orders tighter than the divergence, so the exact value is load-bearing.
_Z_95: float = 1.959963984540054

Basis = Literal["by-construction", "measured", "assumed"]
DecidedBy = Literal["mcnemar-exact", "cluster-bootstrap"]


class DuplicateAnalysisUnit(ValueError):
    """A unit id appeared twice, so the rows are not independent (`-ml` §3.4 Rule 1)."""


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


def paired_bootstrap(diffs: Sequence[float], *, B: int, seed: int) -> tuple[float, float]:
    """Seeded paired percentile bootstrap on per-item differences (`-ml` §3.2d).

    For continuous metrics the CI **is** the test: the decision is that it excludes zero, and no
    separate significance test is run.
    """
    if not diffs:
        raise ValueError("paired_bootstrap needs at least one difference")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(B))
    return _percentile(means, 2.5), _percentile(means, 97.5)


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
        point=point, lo=_percentile(rates, 2.5), hi=_percentile(rates, 97.5), B=B, seed=seed
    )


def _percentile(ordered: Sequence[float], pct: float) -> float:
    if not ordered:
        raise ValueError("no values")
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


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
def _b_min(alpha: float) -> int:
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
    b_min = _b_min(alpha)
    if b_min > n:
        return 0.0
    return sum(comb(n, i) * delta**i * (1 - delta) ** (n - i) for i in range(b_min, n + 1))


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
    """The exact δ at which P(reject) first reaches `power`, unrounded (`-ml` §3.4 Rule 3)."""
    n = _require_effective(n_effective)
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
    """
    _require_effective(n_effective)
    return _b_min(alpha) / n_effective


@dataclass(frozen=True)
class ResolvingPower:
    n_units: int
    unit_kind: str
    alpha: float
    design_effect: float
    basis: Basis
    n_effective: float
    observable_floor: float
    mdd80: float


def resolving_power(
    n_units: int,
    *,
    unit_kind: str,
    design_effect: float,
    basis: str,
    alpha: float,
    power: float = 0.80,
) -> ResolvingPower:
    """Every input but `power` is keyword-only with **no default** (`-ml` §3.4 Rule 2).

    A default of `1.0` for `design_effect` would rebuild gate B-1 by omission: the caller who
    forgets clustering is exactly the caller the gate found.
    """
    if design_effect <= 0:
        raise ValueError("design_effect must be positive")
    n_eff = n_units / design_effect
    return ResolvingPower(
        n_units=n_units,
        unit_kind=unit_kind,
        alpha=alpha,
        design_effect=design_effect,
        basis=basis,  # type: ignore[arg-type]
        n_effective=n_eff,
        observable_floor=observable_floor(n_eff, alpha=alpha),
        mdd80=min_detectable_difference(n_eff, alpha=alpha, power=power),
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
    marginal_overlap: bool
    alpha_used: float


def _pp(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}"


def _plural(unit_kind: str) -> str:
    return f"{unit_kind}s"


def verdict(
    outcomes: PairedOutcomes,
    *,
    resolving: ResolvingPower,
    metric_name: str,
    family: Sequence[str],
    a_label: str = "A",
    b_label: str = "B",
    alpha_step: float | None = None,
    bootstrap_seed: int | None = None,
) -> Verdict:
    """Decide one pre-registered metric, or refuse (`-ml` §3.4 Rule 4).

    Raises — never warns, never silently proceeds — unless all four preconditions hold. McNemar
    exact decides and MOVER-D quantifies **only** at `design_effect == 1.0` and
    `basis == "by-construction"`; otherwise McNemar is anti-conservative, must not decide, and the
    cluster-bootstrap CI on the paired difference takes its place.

    `alpha_step` carries §3.3's Holm–Bonferroni step for this metric when the caller is testing a
    family; it defaults to `resolving.alpha`, which is the most conservative step. The
    `resolving.alpha == 0.05 / len(family)` precondition is unaffected either way.
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
    if abs(resolving.alpha - 0.05 / len(family)) > 1e-12:
        raise ValueError(
            f"a {len(family)}-member verdict family must be reported at alpha="
            f"{0.05 / len(family)}, not {resolving.alpha} (-ml §3.3)"
        )
    if resolving.design_effect < 1.0:
        raise ValueError("design_effect must be >= 1.0")

    a, b, c, d = outcomes.table
    n = a + b + c + d
    diff = (b - c) / n
    p = mcnemar_exact(b, c)
    alpha = resolving.alpha if alpha_step is None else alpha_step

    # FR-15's literal marginal-overlap check, computed and printed as a diagnostic. It costs
    # nothing and the requirement asks for it, but it is inert in this lab's regime (`-ml` §3.1).
    m1 = wilson_interval(a + b, n)
    m2 = wilson_interval(a + c, n)
    overlap = not (m1[1] < m2[0] or m2[1] < m1[0])

    mcnemar_may_decide = resolving.design_effect == 1.0 and resolving.basis == "by-construction"

    if mcnemar_may_decide:
        ci = mover_d_interval(a, b, c, d)
        decided_by: DecidedBy = "mcnemar-exact"
        significant = p <= alpha
    else:
        if bootstrap_seed is None:
            raise ValueError(
                "the clustered decision path needs a bootstrap seed; it goes into the environment "
                "fingerprint so the report is reproducible (-ml §3.2d)"
            )
        ci = paired_bootstrap(outcomes.unit_diffs(), B=10_000, seed=bootstrap_seed)
        decided_by = "cluster-bootstrap"
        significant = ci[0] > 0 or ci[1] < 0

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
        if decided_by == "cluster-bootstrap":
            text += (
                " Decided by the cluster-bootstrap CI on the paired difference; McNemar's p is "
                "anti-conservative under clustering — not the decision."
            )
    elif ci_excludes_zero:
        text = (
            f"Not distinguishable at this sample size. The effect-size interval "
            f"[{_pp(ci[0])}, {_pp(ci[1])}] pp excludes zero but the exact paired test does not "
            f"reach alpha={alpha:g} (b={b}, c={c}, p={p:.3f}). Reported as not distinguishable: "
            "the exact test is the decision rule."
        )
    else:
        text = (
            f"Not distinguishable at this sample size. Observed difference {diff * 100:+.1f} pp, "
            f"95% CI [{_pp(ci[0])}, {_pp(ci[1])}] pp covers zero (b={b}, c={c}, McNemar exact "
            f"p={p:.3f}). This pack resolves differences of >={_pp(resolving.mdd80)} pp with 80% "
            f"power at n={resolving.n_effective:g} effective {unit_plural} ({resolving.n_units} "
            f"units, design effect {resolving.design_effect:.2f}, {resolving.basis}, "
            f"alpha={resolving.alpha:g}); the observed {_pp(abs(diff))} pp is below that. "
            "Neither model is ranked above the other."
        )
        if decided_by == "cluster-bootstrap":
            text += (
                " Decided by the cluster-bootstrap CI on the paired difference; McNemar's p is "
                "anti-conservative under clustering — not the decision."
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
        marginal_overlap=overlap,
        alpha_used=alpha,
    )


def holm_thresholds(p_values: Sequence[float], *, alpha: float = 0.05) -> list[float]:
    """Holm–Bonferroni step for each p-value, in the caller's order (§3.3).

    Order the p-values; test the smallest at α/k, the next at α/(k−1), … The adjusted threshold is
    printed beside each p-value, which is what makes the correction auditable.
    """
    k = len(p_values)
    order = sorted(range(k), key=lambda i: p_values[i])
    steps = [0.0] * k
    for rank, idx in enumerate(order):
        steps[idx] = alpha / (k - rank)
    return steps
