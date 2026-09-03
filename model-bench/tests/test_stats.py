"""§5 test 6 — `stats`. Every number here comes from `docs/plans/small-model-benchmarking-ml.md`.

Not one figure in this file is a literal invented by the implementation or restated from the plan:
the note is the single source of truth for every formula, constant, threshold and tolerance
(plan §3.9, §7). Section citations are given per test so a reader can check the source.

The centre of gravity is `-ml` §3.4's six binding rules, which exist so that "the anti-conservative
version does not typecheck, and the honest one is the only one that runs".
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from fractions import Fraction

import pytest

from modelbench.stats import (
    _Z_95,
    BootstrapResult,
    DuplicateAnalysisUnit,
    PairedOutcomes,
    Rule7Violation,
    UnattainablePower,
    b_min,
    cluster_bootstrap,
    design_effect,
    effective_n,
    floor_clause,
    format_floor_pp,
    holm_steps,
    mcnemar_exact,
    min_detectable_difference,
    min_detectable_difference_exact,
    mover_d_interval,
    observable_floor,
    paired_bootstrap,
    paired_cluster_bootstrap,
    resolving_power,
    unattainable_clause,
    verdict,
    width_inflation,
    wilson_interval,
)

# `-ml` §3.2c — the five worked (a, b, c, d) rows, **as republished at 10 significant decimal
# places in the note's v1.5** (review m-ML-1). Bounds are in percentage points; the p-values are
# exact rationals and are written as rationals so no decimal expansion is load-bearing.
#
# The precision and the tolerance below are one decision, not two: v1.2-v1.4 published these bounds
# at 4 dp *of a percentage point* (1e-6 as a proportion) while mandating a 1e-9 *proportion*
# tolerance in the same subsection, so the published `3.1763` sat 1.31e-7 from the true
# `3.1762869443...` — 131x the tolerance it was meant to be asserted against, and no implementation
# could satisfy it. 10 dp of a pp is 1e-12 as a proportion, three orders inside the mandate.
REGRESSION_FIXTURES = [
    # a,  b, c,  d,   diff_pp,  lo_pp,          hi_pp,           mcnemar_p
    (34, 6, 0, 0, +15.0, 3.1762869443, 29.0723243665, Fraction(1, 32)),
    (30, 6, 0, 4, +15.0, 3.8506738324, 27.7026867131, Fraction(1, 32)),
    (33, 6, 1, 0, +12.5, -0.9864868353, 26.8581964973, Fraction(1, 8)),
    (20, 8, 2, 10, +15.0, 0.1708978316, 28.7785182732, Fraction(7, 64)),
    (72, 10, 2, 1, +9.4, 1.4800198994, 18.2130920778, Fraction(79, 2048)),
]


# --- the pinned constant (`-ml` §3.2a, gate M-1) ------------------------------------------------


def test_z_95_is_the_notes_constant_and_not_1_96() -> None:
    assert _Z_95 == 1.959963984540054
    assert _Z_95 != 1.96


def test_z_is_keyword_only_and_defaults_to_the_pinned_constant() -> None:
    """`nlq_scoring.py`'s own shape: one module constant, `z` keyword-only (`-ml` §3.2a)."""
    params = inspect.signature(wilson_interval).parameters
    assert params["z"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["z"].default == _Z_95


def test_z_95_matches_the_inverse_normal_cdf() -> None:
    """It is Φ⁻¹(0.975), which is what makes 1.96 a typographic rounding rather than a rival.

    **The tolerance is deliberate and must not be tightened to `==`** (review n-ML-3): the pinned
    literal is the 16-significant-digit *decimal* of that double, not the double itself, so
    `_Z_95 = 1.959963984540054` and `NormalDist().inv_cdf(0.975) = 1.9599639845400536` differ by
    4.44e-16 — one ULP. The fixtures are untouched by it either way (total float error against the
    60-digit truth is <= 1.44e-16), but a future "tighten the tolerances" pass turning this into an
    equality would fail in a module nobody changed.
    """
    from statistics import NormalDist

    assert _Z_95 != NormalDist().inv_cdf(0.975)
    assert abs(_Z_95 - NormalDist().inv_cdf(0.975)) < 1e-12


# --- Wilson (`-ml` §3.2a, §4.4's computed widths) -----------------------------------------------


@pytest.mark.parametrize(
    "successes,n,lo,hi",
    [
        # `-ml` §4.4's per-position widths, computed there at z = 1.959963984540054
        (0, 12, 0.000, 0.242),
        (12, 12, 0.758, 1.000),
        (0, 8, 0.000, 0.324),
        (0, 4, 0.000, 0.490),
        (4, 4, 0.510, 1.000),
        # `-ml` §3.1's worked case
        (40, 40, 0.912, 1.000),
        (34, 40, 0.709, 0.929),
    ],
)
def test_wilson_reproduces_the_notes_worked_intervals(successes, n, lo, hi) -> None:
    got_lo, got_hi = wilson_interval(successes, n)
    assert round(got_lo, 3) == lo
    assert round(got_hi, 3) == hi


def test_wilson_rejects_a_zero_denominator() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


# --- McNemar exact (`-ml` §3.2b) ----------------------------------------------------------------


@pytest.mark.parametrize("a,b,c,d,diff_pp,lo_pp,hi_pp,p", REGRESSION_FIXTURES)
def test_mcnemar_reproduces_the_regression_fixtures(a, b, c, d, diff_pp, lo_pp, hi_pp, p) -> None:
    """`-ml` §3.2c — 'assert to 1e-12 absolute. It is exact; a looser tolerance hides an
    implementation that is not.'"""
    assert abs(mcnemar_exact(b, c) - p) < 1e-12


def test_mcnemar_with_no_discordant_pairs_is_one() -> None:
    assert mcnemar_exact(0, 0) == 1.0


@pytest.mark.parametrize(
    "alpha,floor_table",
    [
        # `-ml` §3.2b / §9.1 — the small-sample floor, "the single most useful number for the
        # whole tool": minimum `b` to reach p <= alpha, per discordant count `c`.
        (0.05, {0: 6, 1: 8, 2: 10, 3: 12, 4: 13}),
        (0.025, {0: 7, 1: 9, 2: 11}),
    ],
)
def test_the_small_sample_floor_table(alpha, floor_table) -> None:
    for c, expected_b in floor_table.items():
        assert mcnemar_exact(expected_b, c) <= alpha
        assert mcnemar_exact(expected_b - 1, c) > alpha


# --- MOVER-D (`-ml` §3.2c) ----------------------------------------------------------------------


@pytest.mark.parametrize("a,b,c,d,diff_pp,lo_pp,hi_pp,p", REGRESSION_FIXTURES)
def test_mover_d_reproduces_the_regression_fixtures(a, b, c, d, diff_pp, lo_pp, hi_pp, p) -> None:
    """`-ml` §3.2c's mandate, restored: **1e-9 absolute on the proportion** (review m-ML-1).

    The margin, stated correctly (the shipped docstring was three orders out): `z = 1.96` moves
    these bounds by ~3e-4 pp = ~3e-6 as a proportion, which is ~3 000x — about three orders — above
    this tolerance. That is what makes the pinned constant load-bearing rather than decorative, and
    `test_the_pinned_z_constant_is_load_bearing_at_this_tolerance` asserts it directly.
    """
    lo, hi = mover_d_interval(a, b, c, d)
    assert abs(lo - lo_pp / 100) < 1e-9
    assert abs(hi - hi_pp / 100) < 1e-9


def test_the_pinned_z_constant_is_load_bearing_at_this_tolerance() -> None:
    """`-ml` §3.2a — the largest divergence is on the `34,6,0,0` row; it must be visible here.

    At the mandated 1e-9 proportion tolerance the rounded constant fails on **every** bound, not
    just this one, so the assertion is not a lucky single row.
    """
    for a, b, c, d, _diff, lo_pp, hi_pp, _p in REGRESSION_FIXTURES:
        rounded_z = mover_d_interval(a, b, c, d, z=1.96)
        assert abs(rounded_z[0] - lo_pp / 100) > 1e-9
        assert abs(rounded_z[1] - hi_pp / 100) > 1e-9


@pytest.mark.parametrize("a,b,c,d,diff_pp,lo_pp,hi_pp,p", REGRESSION_FIXTURES)
def test_the_paired_difference_is_b_minus_c_over_n(a, b, c, d, diff_pp, lo_pp, hi_pp, p) -> None:
    n = a + b + c + d
    assert round((b - c) / n * 100, 1) == diff_pp


def test_mover_d_clamps_a_negative_radicand() -> None:
    """`-ml` §3.2c — 'the max(0.0, …) clamps are required, not cosmetic'."""
    lo, hi = mover_d_interval(40, 0, 0, 0)
    assert -1.0 <= lo <= hi <= 1.0


# --- Rule 1: the paired table is constructible only from independent analysis units --------------


def test_from_units_builds_the_table() -> None:
    rows = [("u1", True, True), ("u2", True, False), ("u3", False, True), ("u4", False, False)]
    outcomes = PairedOutcomes.from_units("item", rows)
    assert outcomes.table == (1, 1, 1, 1)
    assert outcomes.n_units == 4
    assert outcomes.unit_kind == "item"


def test_from_units_raises_on_a_repeated_analysis_unit_id() -> None:
    """`-ml` §3.4 Rule 1 / §9.2(a) — 48 rows from 12 script ids, the shape the gate rejected."""
    rows = [(f"S-{i % 12:02d}", True, False) for i in range(48)]
    with pytest.raises(DuplicateAnalysisUnit):
        PairedOutcomes.from_units("conversation", rows)


def test_direct_construction_cannot_bypass_the_duplicate_guard() -> None:
    """Rule 1 says `from_units` is the only constructor; the guard holds on any route regardless."""
    with pytest.raises(DuplicateAnalysisUnit):
        PairedOutcomes(
            unit_kind="conversation",
            unit_ids=("S-01", "S-01"),
            a_correct=(True, False),
            b_correct=(False, True),
        )


# --- Rule 2: resolving power's inputs have no defaults -------------------------------------------


@pytest.mark.parametrize(
    "omitted", ["unit_kind", "design_effect", "basis", "alpha_family", "alpha_mdd"]
)
def test_resolving_power_inputs_are_keyword_only_with_no_default(omitted: str) -> None:
    """`-ml` §3.4 Rule 2 / §9.2(b) — 'the absence of a default is itself the test'.

    A default of 1.0 for `design_effect` would rebuild gate B-1 by omission: the caller who forgets
    clustering is exactly the caller the gate found.
    """
    kwargs = {
        "unit_kind": "conversation",
        "design_effect": 1.0,
        "basis": "by-construction",
        "alpha_family": 0.05,
        "alpha_mdd": 0.05,
    }
    kwargs.pop(omitted)
    with pytest.raises(TypeError):
        resolving_power(12, **kwargs)

    params = inspect.signature(resolving_power).parameters
    assert params[omitted].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[omitted].default is inspect.Parameter.empty


def test_resolving_power_reports_effective_units() -> None:
    rp = resolving_power(
        48, unit_kind="conversation", design_effect=4.0, basis="assumed",
        alpha_family=0.05, alpha_mdd=0.05,
    )
    assert rp.n_effective == 12.0
    assert rp.n_units == 48


# --- Rule 3: MDD is exact and rounded up ---------------------------------------------------------


@pytest.mark.parametrize(
    "n_eff,exact_pp,printed_pp,floor_pp",
    [
        # `-ml` §7.1's table, α = 0.05. The exact column is shown there so nobody re-derives it.
        # The floor column is the note's **v1.5** one: 15.7 at n=38 and 7.0 at n=85, truncated.
        (12.0, 57.794, "57.8", "50.0"),
        (15.0, 47.559, "47.6", "40.0"),
        (20.0, 36.646, "36.7", "30.0"),
        (30.0, 25.075, "25.1", "20.0"),
        (38.0, 20.009, "20.1", "15.7"),
        (40.0, 19.046, "19.1", "15.0"),
        (48.0, 15.972, "16.0", "12.5"),
        (60.0, 12.857, "12.9", "10.0"),
        (85.0, 9.142, "9.2", "7.0"),
        (120.0, 6.509, "6.6", "5.0"),
    ],
)
def test_mdd_and_floor_reproduce_the_notes_table(n_eff, exact_pp, printed_pp, floor_pp) -> None:
    """The floor is asserted **through the formatter the report prints with**, never re-rounded.

    Re-rounding inside the test (`round(observable_floor(...) * 100, 1)`) reproduces the very
    defect being fixed: it asserts the presentation layer's arithmetic against itself, so a test
    can stay green while the printed line says something the value does not support. Rule 3 gives
    the two directions — MDD **up**, floor **down** — and `format_floor_pp` is the one place the
    floor's direction lives.
    """
    assert abs(min_detectable_difference_exact(n_eff, alpha=0.05) * 100 - exact_pp) < 5e-4
    assert f"{min_detectable_difference(n_eff, alpha=0.05) * 100:.1f}" == printed_pp
    assert format_floor_pp(observable_floor(n_eff, alpha=0.05)) == floor_pp


@pytest.mark.parametrize(
    "n_eff,mdd_pp,floor_pp",
    # `-ml` **v1.6** §7.1/§7.3 — the MDD column at the family-adjusted α = 0.025 (unchanged), and
    # the floor beside it at the **unadjusted** α = 0.05 (review M-ML-6). v1.2–v1.5's second floor
    # column (58.3 / 46.6 / 23.3 / 17.5) is deleted: it asserted impossibilities that are
    # attainable, since a member tested at a 0.05 Holm step reaches significance at `6/n`.
    [
        (12.0, "65.6", "50.0"),
        (15.0, "54.2", "40.0"),
        (30.0, "28.7", "20.0"),
        (40.0, "21.9", "15.0"),
    ],
)
def test_the_mdd_takes_the_family_alpha_and_the_floor_the_unadjusted_one(
    n_eff, mdd_pp, floor_pp
) -> None:
    """The two bounds on one printed line take **opposite** αs, on purpose (`-ml` v1.6 Rule 3).

    The MDD claims *"resolves >= X with 80% power"*, which is true only at the tightest step a
    member can be required to clear, α/k. The floor claims *"below Y nothing can reach significance
    at any observed outcome"*, which is true only at the **loosest** step it can face, the
    unadjusted α — printed at α/k it is `7/n`, and at n=40 a rank-2 member with b=6, c=0 clears its
    0.05 Holm step at p=0.031 while its 15.0 pp sits below the printed 17.5. Same falsity class as
    the `15.8` withdrawn in v1.5.
    """
    assert f"{min_detectable_difference(n_eff, alpha=0.025) * 100:.1f}" == mdd_pp
    assert format_floor_pp(observable_floor(n_eff, alpha=0.05)) == floor_pp


def test_resolving_power_computes_each_bound_at_its_own_alpha() -> None:
    """M-ML-6 in one object: `ResolvingPower` carries both αs and uses each where it belongs.

    The mutation this exists to kill is the delivered one — `observable_floor(n_eff,
    alpha=alpha_mdd)` — which printed 17.5 pp at n=40, k=2 and made the floor's own sentence false.
    """
    rp = resolving_power(
        40, unit_kind="item", design_effect=1.0, basis="by-construction",
        alpha_family=0.05, alpha_mdd=0.025,
    )
    assert rp.alpha_family == 0.05
    assert rp.alpha_mdd == 0.025
    assert rp.observable_floor == 6 / 40            # b_min(0.05), never b_min(0.025) = 7
    assert format_floor_pp(rp.observable_floor) == "15.0"
    assert f"{rp.mdd80 * 100:.1f}" == "21.9"        # and the MDD is still at α/k


def test_the_floor_truncation_is_guarded_against_the_bin_edge() -> None:
    """The guard stays; its justification changed with v1.6, and the wording follows it.

    `7/40 = 0.175` is `174.99999999999997` bins in IEEE doubles, so naive truncation prints
    **17.4** where the value is 17.5. That cell was the α=0.025, n=40 floor — **the column v1.6
    deleted** (M-ML-6) — so this guard is now **defensive, not a regression pin on a published
    figure**: swept over `n <= 2000` with the floor's `b_min = 6`, naive truncation never misfires.
    It is kept because `b_min` is a function of α, not the constant 6 (`-ml` §3.4 Rule 3a): any
    future α reopens the hazard, and the cost is one term.

    It is pinned **by the code's own expression**. `math.floor(x / precision)` and
    `math.floor(x * 1000)` are not interchangeable — the double nearest `0.001` is slightly above
    it, so the division rounds down across the bin edge where the multiplication does not — and
    testing the expression the code does not use is exactly how the original sweep missed this.
    """
    import math

    assert math.floor((7 / 40) / 0.001) == 174  # the code's expression: the wrong bin
    assert math.floor((7 / 40) * 1000) == 175  # an equivalent-looking one: no hazard to find
    assert format_floor_pp(7 / 40) == "17.5"  # guarded, whatever alpha put a 7 there
    # and the truncation is against the caller's `precision`, not a hardcoded 1000: the two agree
    # at the printed precision and nowhere else, which is what makes an "equivalent-looking"
    # rewrite of the expression invisible to a test that only ever passes the default.
    assert format_floor_pp(0.1234, precision=0.01) == "12.0"

    # Every floor the note now prints is `6/n`, and at that numerator the hazard is absent over
    # the whole range any pack could reach. This is what makes the guard defensive.
    for n in range(1, 2001):
        assert math.floor((6 / n) / 0.001 + 1e-12) == math.floor((6 / n) / 0.001)
    for n, expected in [
        (12, "50.0"), (15, "40.0"), (20, "30.0"), (30, "20.0"), (40, "15.0"),
        (48, "12.5"), (60, "10.0"), (120, "5.0"),
    ]:
        assert format_floor_pp(6 / n) == expected


def test_the_floor_truncates_and_never_ceilings_an_inexact_value() -> None:
    """Rule 3's reason, as an assertion: at n=38 the attainable `b=6, c=0` outcome IS significant.

    `6/38 = 15.789` pp reaches `p = 1/32 <= 0.05`, so a printed floor of `15.8` would be a sentence
    falsified by a table the same report can render. Truncation to 15.7 keeps it true.
    """
    exact = observable_floor(38.0, alpha=0.05)
    printed = float(format_floor_pp(exact))
    assert printed <= exact * 100
    assert mcnemar_exact(6, 0) <= 0.05
    assert (6 / 38) * 100 >= printed


def test_the_floor_at_exactly_b_min_effective_units_is_still_attainable() -> None:
    """Review m-ML-7 — `floor_clause`'s `> 1.0` boundary was load-bearing and untested: mutating
    it to `>= 1.0` survived all 314 tests while printing two false clauses at `n_eff == b_min`.

    At `n_eff = 6` with `alpha_family = 0.05` the floor is exactly `6/6 = 1.0`. The `exceeds the
    units available` form would claim no observed difference can reach significance — but `b=6,
    c=0` on 6 units is a 100 pp difference at `p = 0.031`, which does. The strict `>` is what keeps
    the attainable form, and this pins the boundary from both sides.
    """
    rp = resolving_power(
        6, unit_kind="item", design_effect=1.0, basis="by-construction",
        alpha_family=0.05, alpha_mdd=0.05,
    )
    assert rp.observable_floor == 1.0
    assert mcnemar_exact(b_min(0.05), 0) <= 0.05  # the outcome the mutant declares impossible
    clause = floor_clause(rp)
    assert clause.startswith("differences below 100.0 pp cannot reach significance")
    assert "exceeds" not in clause

    # ...and one effective unit fewer is genuinely unattainable, so the other form is not dead.
    thin = resolving_power(
        5, unit_kind="item", design_effect=1.0, basis="by-construction",
        alpha_family=0.05, alpha_mdd=0.05,
    )
    assert thin.observable_floor > 1.0
    assert "exceeds the 5 effective units available" in floor_clause(thin)


def test_mdd_is_ceilinged_so_the_printed_number_actually_delivers_its_power() -> None:
    """`-ml` §3.4 Rule 3 / §9.3 — rounding to nearest would print 19.0 pp, whose measured power is
    0.798, below the 0.80 the sentence claims. 19.1 pp gives 0.8023."""
    from modelbench.stats import _mcnemar_power

    printed = min_detectable_difference(40.0, alpha=0.05)
    assert round(printed * 100, 1) == 19.1
    assert _mcnemar_power(40, printed, alpha=0.05) >= 0.80
    assert _mcnemar_power(40, 0.190, alpha=0.05) < 0.80


def test_mdd_is_not_a_constant() -> None:
    """S1 done-condition 5, first half — it returns different values for different n."""
    values = {min_detectable_difference(float(n), alpha=0.05) for n in (12, 20, 40, 85)}
    assert len(values) == 4


def test_mdd_is_not_the_eight_over_n_rule_of_thumb() -> None:
    """`-ml` §3.4 Rule 3 — 8/n gives 20.0 pp where the note says 19.1 pp; it is not code."""
    assert round(min_detectable_difference(40.0, alpha=0.05) * 100, 1) != round(8 / 40 * 100, 1)


def test_mdd_refuses_a_bare_observation_count() -> None:
    """S1 done-condition 5, the B-1 detector; `-ml` §3.4 Rule 2 and §9.3.

    `min_detectable_difference` takes `n_effective: float`, never `n: int`, so passing a raw
    observation count is a visible mislabel at the call site rather than an invisible one inside
    the function. This is the assertion that would have caught v1.1's
    `min_detectable_difference(48)` printing 16.7 pp for a clustered pack whose honest figure is
    57.8 pp.
    """
    with pytest.raises(TypeError):
        min_detectable_difference(48, alpha=0.05)
    with pytest.raises(TypeError):
        min_detectable_difference_exact(48, alpha=0.05)
    assert min_detectable_difference(48.0, alpha=0.05) > 0


# --- Rule 4: verdict() asserts its preconditions and refuses --------------------------------------


def _outcomes(a: int, b: int, c: int, d: int, unit_kind: str = "item") -> PairedOutcomes:
    rows: list[tuple[str, bool, bool]] = []
    for i in range(a):
        rows.append((f"a{i}", True, True))
    for i in range(b):
        rows.append((f"b{i}", True, False))
    for i in range(c):
        rows.append((f"c{i}", False, True))
    for i in range(d):
        rows.append((f"d{i}", False, False))
    return PairedOutcomes.from_units(unit_kind, rows)


def _rp(
    n: int,
    unit_kind: str = "item",
    deff: float = 1.0,
    basis: str = "by-construction",
    alpha_mdd: float = 0.05,
    alpha_family: float = 0.05,
):
    return resolving_power(
        n, unit_kind=unit_kind, design_effect=deff, basis=basis,
        alpha_family=alpha_family, alpha_mdd=alpha_mdd,
    )


def test_verdict_requires_the_resolving_power_to_belong_to_this_table() -> None:
    """`-ml` §3.4 Rule 4 precondition 1 / §9.2(c)."""
    with pytest.raises(ValueError):
        verdict(_outcomes(34, 6, 0, 0), resolving=_rp(39), metric_name="m", family=["m"])


def test_verdict_requires_matching_unit_kinds() -> None:
    """Rule 4 precondition 2."""
    with pytest.raises(ValueError):
        verdict(
            _outcomes(34, 6, 0, 0),
            resolving=_rp(40, unit_kind="conversation"),
            metric_name="m",
            family=["m"],
        )


def test_verdict_requires_the_family_adjusted_alpha() -> None:
    """Rule 4 precondition 3 — a k-member family cannot be reported at alpha=0.05 by oversight."""
    with pytest.raises(ValueError):
        verdict(
            _outcomes(34, 6, 0, 0),
            resolving=_rp(40, alpha_mdd=0.05),
            metric_name="m",
            family=["m", "other"],
        )
    # the same call at alpha = 0.05/2 is accepted
    verdict(
        _outcomes(34, 6, 0, 0),
        resolving=_rp(40, alpha_mdd=0.025),
        metric_name="m",
        family=["m", "other"],
    )


def test_verdict_requires_the_metric_to_be_in_the_family() -> None:
    with pytest.raises(ValueError):
        verdict(_outcomes(34, 6, 0, 0), resolving=_rp(40), metric_name="unlisted", family=["m"])


def test_verdict_requires_a_design_effect_of_at_least_one() -> None:
    """Rule 4 precondition 4."""
    with pytest.raises(ValueError):
        verdict(
            _outcomes(34, 6, 0, 0), resolving=_rp(40, deff=0.5), metric_name="m", family=["m"]
        )


# --- the three verdict strings (`-ml` §3.2e), which AC-4 is checked against -----------------------


def test_distinguishable_renders_the_notes_first_string() -> None:
    """AC-4's counter-case: 40/40 vs 34/40 IS distinguishable, and the old marginal-overlap rule
    got it backwards (`-ml` §3.1, plan S1 done-condition 3)."""
    v = verdict(_outcomes(34, 6, 0, 0), resolving=_rp(40), metric_name="cleanThroughTurn4",
                family=["cleanThroughTurn4"])
    assert v.distinguishable is True
    assert v.text == (
        "A is better than B on cleanThroughTurn4: +15.0 pp (95% CI [3.2, 29.1] pp), "
        "n=40 paired items (unit: item, design effect 1.00), McNemar exact p=0.031 (b=6, c=0)."
    )
    assert "not distinguishable at this sample size" not in v.text.lower()


def test_not_distinguishable_renders_the_notes_second_string() -> None:
    v = verdict(_outcomes(33, 6, 1, 0), resolving=_rp(40), metric_name="cleanThroughTurn4",
                family=["cleanThroughTurn4"])
    assert v.distinguishable is False
    assert v.text == (
        "Not distinguishable at this sample size. Observed difference +12.5 pp, "
        "95% CI [-1.0, 26.9] pp covers zero (b=6, c=1, McNemar exact p=0.125). "
        "This pack resolves differences of >=19.1 pp with 80% power at n=40 effective items "
        "(40 units, design effect 1.00, by-construction, alpha=0.05); the observed 12.5 pp is "
        "below that. Neither model is ranked above the other."
    )


def test_instruments_disagree_renders_both_components_in_prose() -> None:
    """`-ml` §3.2e verdict 3 — 'real and not rare'. McNemar decides; MOVER-D quantifies."""
    v = verdict(_outcomes(20, 8, 2, 10), resolving=_rp(40), metric_name="m", family=["m"])
    assert v.distinguishable is False
    assert v.text == (
        "Not distinguishable at this sample size. The effect-size interval [0.2, 28.8] pp "
        "excludes zero but the exact paired test does not reach alpha=0.05 (b=8, c=2, p=0.109). "
        "Reported as not distinguishable: the exact test is the decision rule."
    )


def test_the_better_arm_is_named_when_b_wins() -> None:
    """The stated difference and its interval must describe the SAME direction.

    Re-orienting the difference to the winner while leaving the interval in A-minus-B orientation
    prints `+15.0 pp (95% CI [-29.1, -3.2] pp)` — a positive effect with a wholly negative
    interval, which is not a hard failure anywhere but is exactly the kind of internally
    contradictory line a measuring instrument must not emit.
    """
    v = verdict(_outcomes(34, 0, 6, 0), resolving=_rp(40), metric_name="m", family=["m"],
                a_label="cand", b_label="incumbent")
    assert v.text == (
        "incumbent is better than cand on m: +15.0 pp (95% CI [3.2, 29.1] pp), "
        "n=40 paired items (unit: item, design effect 1.00), McNemar exact p=0.031 (b=0, c=6)."
    )


def test_a_negative_observed_difference_keeps_its_sign_when_not_distinguishable() -> None:
    """The non-significant strings stay in A-minus-B orientation, sign included, so the printed
    difference always sits inside the printed interval."""
    v = verdict(_outcomes(33, 1, 6, 0), resolving=_rp(40), metric_name="m", family=["m"])
    assert "Observed difference -12.5 pp, 95% CI [-26.9, 1.0] pp covers zero" in v.text


def test_an_observed_difference_above_the_mdd_renders_the_notes_alternate_clause() -> None:
    """M-ML-7 / P3-2 — the closing clause was fixed prose and false on 17% of the tables that
    printed it. `-ml` v1.7 §3.2e mandates the alternate wording verbatim, discordance counts
    included: they are the reason the two numbers point opposite ways, and without them the
    sentence reads as the instrument contradicting itself.

    This is §7.1's *normal case for a model swap* — a candidate that wins more than it loses
    without strictly dominating — not a corner.
    """
    v = verdict(_outcomes(1, 13, 5, 1), resolving=_rp(20), metric_name="m", family=["m"])
    assert v.distinguishable is False
    assert "is below that" not in v.text
    assert (
        "This pack resolves differences of >=36.7 pp with 80% power at n=20 effective items "
        "(20 units, design effect 1.00, by-construction, alpha=0.05); the observed 40.0 pp is "
        "above that, but the MDD assumes strict dominance and this comparison is not strictly "
        "dominant (b=13, c=5), so the difference required for 80% power at this discordance mix "
        "is larger (§7.1)."
    ) in v.text


def test_the_alternate_clause_names_the_discordance_split_of_the_losing_direction() -> None:
    """The note's own second case, `(n=30, b=5, c=13)` — the candidate *loses* more than it wins,
    and the counts printed must be this table's, not the winning direction's."""
    v = verdict(_outcomes(6, 5, 13, 6), resolving=_rp(30), metric_name="m", family=["m"])
    assert "the observed 26.7 pp is above that, but the MDD assumes strict dominance" in v.text
    assert "not strictly dominant (b=5, c=13)" in v.text


def test_the_rendered_power_is_the_power_the_mdd_was_computed_at() -> None:
    """Review n-ML-6 — `"80% power"` was hard-coded in three rendered strings while `power` is a
    `resolving_power` parameter, so a caller who passed 0.90 got three sentences claiming a power
    the figure beside them does not have."""
    rp = resolving_power(
        40, unit_kind="item", design_effect=1.0, basis="by-construction",
        alpha_family=0.05, alpha_mdd=0.05, power=0.90,
    )
    v = verdict(_outcomes(33, 6, 1, 0), resolving=rp, metric_name="m", family=["m"])
    assert "90% power" in v.text
    assert "80% power" not in v.text
    # ...and the unattainable replacement, the other sentence that names it.
    thin = resolving_power(
        5, unit_kind="item", design_effect=1.0, basis="by-construction",
        alpha_family=0.05, alpha_mdd=0.05, power=0.90,
    )
    assert thin.mdd80 is None
    assert "no effect size attains 90% power." in unattainable_clause(thin, "items")


def test_an_observed_difference_below_the_mdd_keeps_the_notes_original_clause() -> None:
    """The conditional's other branch stays exactly as `-ml` §3.2e verdict 2 publishes it."""
    v = verdict(_outcomes(33, 6, 1, 0), resolving=_rp(40), metric_name="m", family=["m"])
    assert "the observed 12.5 pp is below that." in v.text
    assert "strictly dominant" not in v.text


def test_verdict_carries_the_marginal_overlap_diagnostic() -> None:
    """FR-15's literal rule survives as a printed diagnostic, never as the verdict (`-ml` §3.2)."""
    v = verdict(_outcomes(34, 6, 0, 0), resolving=_rp(40), metric_name="m", family=["m"])
    assert v.marginal_overlap is True
    assert v.distinguishable is True


# --- Rule 4's branch: clustering moves the decision off McNemar -----------------------------------


def test_a_design_effect_above_one_moves_the_decision_to_the_bootstrap() -> None:
    """`-ml` §3.4 Rule 4 / §9.2(d) — McNemar is anti-conservative and must not decide."""
    v = verdict(
        _outcomes(34, 6, 0, 0),
        resolving=_rp(40, deff=2.0, basis="measured"),
        metric_name="m",
        family=["m"],
        bootstrap_seed=20260902,
    )
    assert v.decided_by == "cluster-bootstrap"
    assert "Decided by the cluster-bootstrap CI on the paired difference, widened by " \
        "sqrt(DEFF)=1.41 for the declared clustering, in conjunction with McNemar's exact test" \
        in v.text


def test_an_assumed_basis_also_moves_the_decision_off_mcnemar() -> None:
    """Plan §5 test 12b's fail-safe: a probe that did not run yields `assumed`, and `assumed` must
    move the decision onto the cluster-bootstrap CI even at a design effect of exactly 1.0."""
    v = verdict(
        _outcomes(34, 6, 0, 0),
        resolving=_rp(40, deff=1.0, basis="assumed"),
        metric_name="m",
        family=["m"],
        bootstrap_seed=20260902,
    )
    assert v.decided_by == "cluster-bootstrap"


@pytest.mark.parametrize("basis", ["assumed", "measured"])
def test_the_fail_safe_path_names_the_basis_and_not_a_widening_that_did_not_happen(basis) -> None:
    """Review P3-3 — the trailing label said *"widened by sqrt(DEFF)=1.00 for the declared
    clustering"* on the path **every** comparison takes until S2's determinism probe lands. At
    `design_effect == 1.0` nothing was widened and no clustering was declared: what displaced
    McNemar is the unverified basis, and the sentence never named it.
    """
    v = verdict(
        _outcomes(34, 6, 0, 0),
        resolving=_rp(40, deff=1.0, basis=basis),
        metric_name="m",
        family=["m"],
        bootstrap_seed=20260902,
    )
    assert "for the declared clustering" not in v.text
    assert (
        f"not widened (sqrt(DEFF)=1.00), because this comparison's design effect is {basis} "
        "rather than established by construction"
    ) in v.text


def test_a_real_widening_still_names_the_clustering_it_corrected_for() -> None:
    """The other side of P3-3's conditional: where the design effect *is* above 1.0 the widening
    happened and the original wording is true, so it must survive."""
    v = verdict(
        _outcomes(34, 6, 0, 0),
        resolving=_rp(40, deff=2.0, basis="measured"),
        metric_name="m",
        family=["m"],
        bootstrap_seed=20260902,
    )
    assert "widened by sqrt(DEFF)=1.41 for the declared clustering" in v.text
    assert "not widened" not in v.text


def test_a_measured_basis_at_deff_one_also_moves_the_decision_off_mcnemar() -> None:
    """P2-1 — Rule 4's branch condition is `by-construction`, and the third enum value was untested.

    `mcnemar_may_decide` widened to `basis in ("by-construction", "measured")` — letting a measured
    basis into the McNemar seat at `design_effect == 1.0` — survived all 296 tests: `"assumed"` was
    covered at this boundary and `"measured"` was not. It became reachable in the fix round
    (`report.py` now preserves the weaker of the two *actual* bases instead of collapsing to
    `"assumed"`), and S2's runner is what will start producing it.

    A measured DEFF of exactly 1.0 is an estimate that came back at 1.0, not a design that makes it
    1.0 — the distinction Rule 4 is drawing is *provenance*, not magnitude.
    """
    v = verdict(
        _outcomes(34, 6, 0, 0),
        resolving=_rp(40, deff=1.0, basis="measured"),
        metric_name="m",
        family=["m"],
        bootstrap_seed=20260902,
    )
    assert v.decided_by == "cluster-bootstrap"


def test_the_bootstrap_path_refuses_without_a_seed() -> None:
    """The seed goes into the fingerprint so a report is reproducible (`-ml` §3.2d)."""
    with pytest.raises(ValueError):
        verdict(
            _outcomes(34, 6, 0, 0),
            resolving=_rp(40, deff=2.0, basis="measured"),
            metric_name="m",
            family=["m"],
        )


def test_mcnemar_decides_only_at_deff_one_and_by_construction() -> None:
    v = verdict(_outcomes(34, 6, 0, 0), resolving=_rp(40), metric_name="m", family=["m"])
    assert v.decided_by == "mcnemar-exact"


# --- Rule 5: the design effect is a variance ratio, not a width ratio -----------------------------


def test_design_effect_is_the_squared_width_ratio() -> None:
    """`-ml` §3.4 Rule 5 — v1.1 called the ratio itself the design effect; an implementer following
    it literally would have divided by 2.6 where the truth was 7, over-stating effective n ~2.7x."""
    assert width_inflation(2.6457513110645907, 1.0) == pytest.approx(math.sqrt(7))
    assert design_effect(2.6457513110645907, 1.0) == pytest.approx(7.0)


def test_the_rho_equals_one_identity() -> None:
    """`-ml` §3.4 Rule 5 / §9.4 — 'when rho = 1, effective n must equal the cluster count'.

    §4.4's real case: 280 turns in 40 conversations, within-conversation correlation ~1, m = 7
    turns each. DEFF = 1 + (m-1)*rho = 7, width ratio sqrt(7) = 2.646 (v1.1's '~2.6'), and
    n_eff = 280/7 = 40 — exactly the conversation count. This is the one assertion that catches a
    squaring error in either direction.
    """
    m, rho, clusters = 7, 1.0, 40
    deff = 1 + (m - 1) * rho
    assert deff == 7.0
    assert effective_n(clusters * m, deff) == float(clusters)
    assert design_effect(math.sqrt(deff), 1.0) == pytest.approx(deff)


def test_a_width_ratio_used_as_a_design_effect_fails_the_identity() -> None:
    """The mutation Rule 5 exists to catch: dividing by 2.6 instead of 7."""
    assert effective_n(280, math.sqrt(7.0)) != 40.0


# --- Rule 6: cluster_bootstrap is one level -------------------------------------------------------


def test_cluster_bootstrap_resamples_units_not_observations() -> None:
    """`-ml` §3.4 Rule 6 / §4.4 — identical observations inside a cluster carry no extra info,
    so a clustered resample must be far wider than a naive Wilson interval over the raw count."""
    units = [[True] * 7 for _ in range(20)] + [[False] * 7 for _ in range(20)]
    result = cluster_bootstrap(units, B=2000, seed=20260902)
    assert isinstance(result, BootstrapResult)
    naive_lo, naive_hi = wilson_interval(140, 280)
    assert (result.hi - result.lo) > (naive_hi - naive_lo)
    assert result.point == pytest.approx(0.5)


def test_cluster_bootstrap_is_seeded_and_reproducible() -> None:
    units = [[True, False] for _ in range(12)]
    first = cluster_bootstrap(units, B=500, seed=7)
    assert cluster_bootstrap(units, B=500, seed=7) == first
    assert cluster_bootstrap(units, B=500, seed=8) != first


def test_cluster_bootstrap_seed_is_keyword_only_with_no_default() -> None:
    params = inspect.signature(cluster_bootstrap).parameters
    assert params["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["seed"].default is inspect.Parameter.empty


def test_paired_bootstrap_is_seeded_and_reproducible() -> None:
    diffs = [1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    first = paired_bootstrap(diffs, B=500, seed=3)
    assert paired_bootstrap(diffs, B=500, seed=3) == first


# --- Rule 7: no verdict path returns `distinguishable` below the observable floor -----------------


@pytest.mark.parametrize("deff", [2.0, 4.0, 7.0])
def test_no_clustered_verdict_is_distinguishable_below_the_observable_floor(deff: float) -> None:
    """`-ml` §3.4 Rule 7, on the exact table review B-ML-1 measured.

    The delivered build declared this 15.0 pp difference *distinguishable* at every one of these
    design effects while its own mandatory resolving-power line said nothing below 30.0 / 60.0 /
    105.0 pp could reach significance. A measuring instrument must not emit two statements that
    contradict each other, and here the wrong one was the verdict.
    """
    rp = _rp(40, deff=deff, basis="measured")
    v = verdict(
        _outcomes(34, 6, 0, 0), resolving=rp, metric_name="m", family=["m"],
        bootstrap_seed=20260902,
    )
    assert abs(v.diff) < rp.observable_floor
    assert v.distinguishable is False


def test_rule_7_is_what_catches_the_case_the_widened_interval_still_misses() -> None:
    """The demotion is a real path, not a formality: at DEFF = 2 the interval alone is not enough.

    `sqrt(2) = 1.41` widening leaves the CI excluding zero, so the *interval* still says
    "distinguishable" while the report's own honesty line says nothing below 30.0 pp can reach
    significance at n_eff = 20. Rule 7 is the only thing standing between those two sentences, and
    the rendered text has to say which one won. At DEFF 4 and 7 the widened interval already covers
    zero and the demotion is not needed — which is why the invariant, not the demotion, is the
    property asserted above.
    """
    rp = _rp(40, deff=2.0, basis="measured")
    v = verdict(
        _outcomes(34, 6, 0, 0), resolving=rp, metric_name="m", family=["m"],
        bootstrap_seed=20260902,
    )
    assert v.ci[0] > 0  # the interval on its own excludes zero
    assert v.floor_demoted is True
    assert v.distinguishable is False
    assert "below this pack's observable floor" in v.text
    assert "differences below 30.0 pp cannot reach significance" in v.text
    assert "Rule 7" in v.text


def test_rule_7_compares_against_the_exact_floor_not_the_printed_one() -> None:
    """Rule 7's first detail — otherwise the invariant inherits the presentation layer's rounding.

    At n_eff = 38 the exact floor is `6/38 = 15.789` pp and the printed one is `15.7`. A difference
    of 15.75 pp sits between them: it is above what the report prints and below what can actually
    reach significance, so an invariant built on the printed value would let it through.
    """
    rp = _rp(38, deff=1.0, basis="by-construction")
    assert float(format_floor_pp(rp.observable_floor)) < 15.75 < rp.observable_floor * 100
    assert rp.observable_floor == 6 / 38

    # ...and the distinction has to change a *verdict*, not merely two numbers. At DEFF = 1.335 the
    # exact floor is 20.025 pp and the printed one is 20.0 pp, so an observed 20.0 pp sits in the
    # gap: below what can actually reach significance, not below what the report prints. Only the
    # exact comparison demotes it. Asserting the two floats alone left the mutation
    # `below_floor = abs(diff) < float(format_floor_pp(...)) / 100` alive.
    clustered = _rp(40, deff=1.335, basis="measured")
    assert clustered.observable_floor * 100 == pytest.approx(20.025, abs=1e-3)
    assert format_floor_pp(clustered.observable_floor) == "20.0"
    v = verdict(
        _outcomes(32, 8, 0, 0), resolving=clustered, metric_name="m", family=["m"],
        bootstrap_seed=20260902,
    )
    assert v.diff * 100 == pytest.approx(20.0)
    assert v.ci[0] > 0  # the interval alone would rank it
    assert v.floor_demoted is True
    assert v.distinguishable is False


def test_the_mcnemar_path_satisfies_rule_7_by_construction() -> None:
    """The asymmetry that makes Rule 7 a real detector rather than a formality (`-ml` §3.4 Rule 7).

    Exhaustively over the note's sample sizes and every discordant split, McNemar significance
    already implies `|b - c| / n >= b_min(alpha) / n`. So the invariant never fires on the valid
    branch, and every time it *does* fire it is reporting a defect in the substitute instrument.
    """
    smallest = b_min(0.05)
    assert smallest == 6
    for m in range(401):
        for alpha in (0.05, 0.025):
            # `mcnemar_exact(b, c)` depends on `m = b + c` and `k = min(b, c)` only, and is
            # increasing in `k` while `|b - c| = m - 2k` decreases: so the largest rejecting `k`
            # is the only case that can violate the bound. Binary search it rather than sweeping
            # 80k tables, which is what makes `b + c <= 400` affordable here.
            lo, hi = -1, m // 2 + 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if mcnemar_exact(m - mid, mid) <= alpha:
                    lo = mid
                else:
                    hi = mid
            if lo >= 0:
                assert (m - 2 * lo) >= smallest


def test_rule_7_raises_on_the_mcnemar_path_where_it_is_a_theorem() -> None:
    """m-ML-6 — the response splits by path, because the invariant has two different statuses.

    On `mcnemar-exact` it is a **theorem**: with the floor at the unadjusted α, `p <= alpha_step <=
    alpha_family` implies `|b−c| >= b_min(0.05) = 6`, hence `|diff| >= 6/n`, at every Holm step —
    verified exhaustively over every `(b, c)` with `b + c <= 400` by the test above. So a fire
    there cannot come from the data; it can only come from a defect in one of the two independent
    routes that produce the two numbers, and **silently demoting it discards exactly the detector
    property Rule 7 exists for** ("a substituted instrument surfaces as a contradiction rather than
    as a plausible number"). The only way to reach the branch is to build the corrupted state, so
    that is what this does: a `ResolvingPower` whose floor did not come from `b_min/n_effective`.
    """
    honest = _rp(40)
    corrupted = dataclasses.replace(honest, observable_floor=0.30)
    with pytest.raises(Rule7Violation) as excinfo:
        verdict(_outcomes(34, 6, 0, 0), resolving=corrupted, metric_name="m", family=["m"])
    assert "mcnemar-exact" in str(excinfo.value)

    # ...and the same corruption on the substitute path demotes and names, rather than raising:
    # there the invariant is a guard, and raising would abort on ordinary clustered data.
    substitute = dataclasses.replace(corrupted, design_effect=1.0, basis="assumed")
    v = verdict(
        _outcomes(34, 6, 0, 0), resolving=substitute, metric_name="m", family=["m"],
        bootstrap_seed=20260902,
    )
    assert v.decided_by == "cluster-bootstrap"
    assert v.floor_demoted is True
    assert v.distinguishable is False


def test_verdict_refuses_a_holm_step_outside_the_two_alphas() -> None:
    """The theorem's premise, made checkable rather than assumed (`-ml` v1.6 §3.4 Rule 7).

    Holm's own steps are `α/(k−i)`, which lie in `[alpha_mdd, alpha_family]` by construction. A
    step outside that range is a caller defect, and left unchecked it would surface as the
    `Rule7Violation` above — a "module bug" raised at the wrong module.
    """
    rp = _rp(40, alpha_mdd=0.025)
    for bad in (0.0125, 0.10):
        with pytest.raises(ValueError):
            verdict(
                _outcomes(34, 6, 0, 0), resolving=rp, metric_name="m",
                family=["m", "other"], alpha_step=bad,
            )
    for good in (0.025, 0.05):
        verdict(
            _outcomes(34, 6, 0, 0), resolving=rp, metric_name="m",
            family=["m", "other"], alpha_step=good,
        )


def test_the_converse_of_rule_7_is_not_an_invariant() -> None:
    """Rule 7's second detail — `|diff| >= floor` does **not** imply distinguishable.

    `-ml` §3.2c row 4 is the counterexample already in this suite: n=40, `|diff| = 15.0` pp exactly
    at the α=0.05 floor, `p = 7/64 = 0.109375`, not distinguishable. Significance depends on the
    discordance *split*, not on `b - c`, and asserting the converse would fail on this fixture.
    """
    rp = _rp(40)
    v = verdict(_outcomes(20, 8, 2, 10), resolving=rp, metric_name="m", family=["m"])
    assert abs(v.diff) == pytest.approx(rp.observable_floor)
    assert v.distinguishable is False
    assert v.floor_demoted is False


# --- B-ML-1: the clustered interval must respond to the declared design effect --------------------


def test_the_clustered_interval_widens_with_the_declared_design_effect() -> None:
    """B-ML-1 — the delivered interval was identical to four decimals across DEFF 2, 4 and 7.

    An i.i.d. resample of the rows of the paired table cannot see the clustering those rows are
    declared to carry, so `design_effect` never entered the interval at all: the branch changed the
    instrument's *name* and not its interval.
    """
    widths = []
    for deff in (1.0, 2.0, 4.0, 7.0):
        v = verdict(
            _outcomes(34, 6, 0, 0), resolving=_rp(40, deff=deff, basis="assumed"),
            metric_name="m", family=["m"], bootstrap_seed=20260902,
        )
        widths.append(v.ci[1] - v.ci[0])
    assert widths == sorted(widths)
    assert len(set(round(w, 6) for w in widths)) == 4


# --- B-ML-2: the substitute may never declare what the exact test refuses ------------------------


@pytest.mark.parametrize("b,c", [(7, 1), (9, 2), (11, 3)])
def test_the_substitute_path_never_declares_what_the_exact_test_refuses(b: int, c: int) -> None:
    """B-ML-2 — the untested corner is the **default** one, and it was anti-conservative.

    At `design_effect == 1.0` with `basis == "assumed"` — the fail-safe every comparison carries
    until S2 lands the determinism probe — the decision moves off McNemar and `sqrt(1.0)` widens
    nothing, so a bare percentile interval decided. These three tables were measured firing at
    p = 0.057–0.070, where the exact test refuses. Rule 7 does not catch them: 15.0, 17.5 and
    20.0 pp are all **at or above** the 15.0 pp floor.

    The fix is Rule 4's permitted form: on any non-`by-construction` path the decision is a
    **conjunction** — the widened CI excludes zero **and** `mcnemar_exact <= alpha_step`. The
    objection to McNemar under clustering is that it *rejects* too readily, so a necessary
    condition only ever removes rejections and the result is uniformly at least as conservative as
    either instrument alone.
    """
    rp = _rp(40, deff=1.0, basis="assumed")
    v = verdict(
        _outcomes(40 - b - c, b, c, 0), resolving=rp, metric_name="m", family=["m"],
        bootstrap_seed=20260902,
    )
    assert 0.05 < mcnemar_exact(b, c) < 0.08  # the exact test refuses, and not by a wide margin
    assert v.ci[0] > 0  # the interval on its own would have ranked it
    assert abs(v.diff) >= rp.observable_floor  # so Rule 7 is not what saves it
    assert v.floor_demoted is False
    assert v.distinguishable is False
    assert "The cluster-bootstrap interval" in v.text
    assert "the exact paired test does not reach alpha=0.05" in v.text
    assert "on this path the exact test is a necessary condition, and it is not met" in v.text


@pytest.mark.parametrize("deff,basis", [(1.0, "assumed"), (1.0, "measured"), (2.0, "measured")])
def test_the_conjunction_still_ranks_a_table_both_instruments_accept(deff, basis) -> None:
    """The veto only *removes* rejections — it must not remove the ones both instruments allow.

    `(34, 6, 0, 0)` is p=0.031 and a widened interval that excludes zero at DEFF 1 and 2, and its
    15.0 pp is not below the 15.0 pp floor at DEFF 1. A conjunction that never ranks anything would
    pass B-ML-2's test and be useless.
    """
    rp = _rp(40, deff=deff, basis=basis)
    v = verdict(
        _outcomes(34, 6, 0, 0), resolving=rp, metric_name="m", family=["m"],
        bootstrap_seed=20260902,
    )
    assert v.decided_by == "cluster-bootstrap"
    assert v.distinguishable is (deff == 1.0)  # at DEFF 2 the floor has moved to 30.0 pp
    if deff == 1.0:
        assert "is better than" in v.text


def test_at_deff_one_the_substitute_interval_is_narrower_than_the_mover_d_it_replaces() -> None:
    """Why the veto is *needed* rather than merely permitted — B-ML-2's mechanism, in one number.

    `sqrt(1.0) = 1.0`, so at the default basis the "widened" interval is the bare percentile one,
    and it is **narrower** than the MOVER-D that McNemar's own path would have quantified with.
    The interval cannot be what makes this path conservative, so something else has to be, and
    that is the conjunction. Above DEFF 1 the widening takes over — the test below.
    """
    lo, hi = mover_d_interval(34, 6, 0, 0)
    v = verdict(
        _outcomes(34, 6, 0, 0), resolving=_rp(40, deff=1.0, basis="assumed"),
        metric_name="m", family=["m"], bootstrap_seed=20260902,
    )
    assert (v.ci[1] - v.ci[0]) < (hi - lo)


def test_the_clustered_interval_is_not_narrower_than_the_mover_d_it_replaces() -> None:
    """B-ML-1's second consequence: 22.50 pp wide against MOVER-D's 25.90 pp, as delivered.

    The branch exists *because* McNemar is anti-conservative under clustering, so substituting
    something less conservative than McNemar's own effect-size interval inverts its purpose.
    """
    lo, hi = mover_d_interval(34, 6, 0, 0)
    mover_width = hi - lo
    for deff in (2.0, 4.0, 7.0):
        v = verdict(
            _outcomes(34, 6, 0, 0), resolving=_rp(40, deff=deff, basis="measured"),
            metric_name="m", family=["m"], bootstrap_seed=20260902,
        )
        assert (v.ci[1] - v.ci[0]) > mover_width


def test_paired_cluster_bootstrap_scales_the_half_widths_by_sqrt_deff() -> None:
    """`-ml` §3.4 Rule 5: the Kish design effect is the variance ratio, so `sqrt(DEFF)` is exactly
    the quantity that converts it to a half-width."""
    diffs = [1.0] * 6 + [0.0] * 34
    base_lo, base_hi = paired_bootstrap(diffs, B=2000, seed=7)
    point = sum(diffs) / len(diffs)
    lo, hi = paired_cluster_bootstrap(diffs, design_effect=4.0, B=2000, seed=7)
    assert lo == pytest.approx(point - (point - base_lo) * 2.0)
    assert hi == pytest.approx(point + (base_hi - point) * 2.0)


def test_paired_cluster_bootstrap_refuses_a_design_effect_below_one() -> None:
    with pytest.raises(ValueError):
        paired_cluster_bootstrap([1.0, 0.0], design_effect=0.5, B=10, seed=1)


# --- M-ML-1: no MDD exists when the rejection region is empty -------------------------------------


def test_mdd_refuses_to_invent_a_figure_when_no_effect_size_attains_the_power() -> None:
    """`-ml` §7.1 already has the vocabulary — **unattainable** — it was just not in the code.

    Below `b_min(alpha)` units the McNemar rejection region is empty, power is identically zero at
    every δ, and the bisection converges on its upper bracket. The delivered build returned `1.0`
    and the report rendered *"resolves differences of >=100.0 pp with 80% power"* for a
    configuration whose power is exactly zero. Reachable today: n_units=40, DEFF=7 -> n_eff=5.71.
    """
    from modelbench.stats import _mcnemar_power

    assert _mcnemar_power(5, 1.0, alpha=0.05) == 0.0
    with pytest.raises(UnattainablePower):
        min_detectable_difference(5.71, alpha=0.05)
    with pytest.raises(UnattainablePower):
        min_detectable_difference_exact(5.71, alpha=0.05)
    # and it is genuinely a boundary, not a blanket refusal
    assert min_detectable_difference(6.0, alpha=0.05) > 0


def test_resolving_power_reports_an_absent_mdd_rather_than_a_false_one() -> None:
    rp = resolving_power(
        40, unit_kind="item", design_effect=7.0, basis="measured",
        alpha_family=0.05, alpha_mdd=0.05,
    )
    assert rp.n_effective == pytest.approx(40 / 7)
    assert rp.mdd80 is None
    assert rp.observable_floor > 1.0


# --- B-1 / M-ML-2: Holm-Bonferroni is applied, not merely printed ---------------------------------


def test_holm_steps_are_the_notes_thresholds_by_value() -> None:
    """§3.3 — 'test the smallest at α/k, the next at α/(k−1), …', printed beside each p-value."""
    steps = holm_steps([0.008, 0.031], alpha=0.05)
    assert [s.threshold for s in steps] == [0.025, 0.05]
    assert [s.rank for s in steps] == [0, 1]
    assert [s.rejected for s in steps] == [True, True]
    assert all(s.tested for s in steps)


def test_holm_stops_at_the_first_non_rejection() -> None:
    """M-ML-2's second defect: without the step-down stop the larger threshold is unusable.

    `holm_thresholds([0.30, 0.04])` returned `[0.05, 0.025]`, and a reader applying the printed
    0.05 to p = 0.30 concludes it was rejected. Holm rejects it only if every smaller p already
    cleared its own step, and here p = 0.04 fails at α/2 = 0.025, so nothing is rejected.
    """
    steps = holm_steps([0.30, 0.04], alpha=0.05)
    assert [s.threshold for s in steps] == [0.05, 0.025]
    assert [s.rejected for s in steps] == [False, False]
    assert [s.tested for s in steps] == [False, True]

    # the case the stop actually changes: 0.04 <= its own 0.05 step, but 0.03 failed at 0.025 first
    steps = holm_steps([0.03, 0.04], alpha=0.05)
    assert [s.rejected for s in steps] == [False, False]
    assert [s.tested for s in steps] == [True, False]


def test_holm_steps_returns_one_step_per_family_member_in_the_callers_order() -> None:
    """P2-3 — the ladder is consumed positionally, so its length is part of its contract.

    `holm_steps` ended `return [s for s in steps if s is not None]`, a type narrowing over a
    placeholder list: every index is assigned today, but the filter means a ladder that ever
    returned short would do so **silently**, and `report.py` zipped it un-`strict` against the
    metric tables — dropping a pre-registered verdict metric from the report with no error. The
    list is now built without a filter, so it cannot shorten, and the consumer zips strictly.
    """
    for p_values in ([0.5], [0.008, 0.031], [0.9, 0.01, 0.5, 0.02], [0.04] * 7):
        steps = holm_steps(p_values, alpha=0.05)
        assert len(steps) == len(p_values)
        assert [s.p for s in steps] == list(p_values)  # caller's order, not sorted order
        assert sorted(s.rank for s in steps) == list(range(len(p_values)))


def test_verdict_decides_at_the_holm_step_it_is_given() -> None:
    """B-1 — `alpha_step` was built for exactly this and was passed by nothing.

    b=8, c=1 gives p = 0.0390625: not distinguishable at the plain Bonferroni α/k = 0.025 that
    every metric was decided at, distinguishable at its own Holm step of 0.05. Its 17.5 pp
    difference is above the 15.0 pp floor, so Rule 7 does not demote it — see
    `test_the_unadjusted_floor_does_not_take_back_a_holm_step_down_gain` for the case where the
    two came into conflict, and for which of them v1.6 moved.
    """
    outcomes, rp = _outcomes(31, 8, 1, 0), _rp(40, alpha_mdd=0.025)
    bonferroni = verdict(outcomes, resolving=rp, metric_name="m", family=["m", "other"])
    holm = verdict(
        outcomes, resolving=rp, metric_name="m", family=["m", "other"], alpha_step=0.05
    )
    assert bonferroni.distinguishable is False
    assert holm.distinguishable is True
    assert holm.alpha_used == 0.05


def test_the_unadjusted_floor_does_not_take_back_a_holm_step_down_gain() -> None:
    """M-ML-6's counterfactual, as a verdict rather than as two numbers.

    The delivered build printed the floor at α/k, so a rank-2 member with b=6, c=0 at n=40 cleared
    its 0.05 Holm step (p=0.031) and was then demoted by a 17.5 pp floor — reducing Holm to
    Bonferroni for every difference in `[6/n, 7/n)`, which is *precisely* the band §7.3 already
    prices as the cost of a second verdict metric. The build charged that price twice.

    With the floor at the unadjusted α it is 15.0 pp, the 15.0 pp difference is not below it, and
    the Holm gain survives. The α/k floor is not merely expensive, it is **false**: it says this
    outcome cannot reach significance, and this outcome did.
    """
    rp = _rp(40, alpha_mdd=0.025)
    v = verdict(
        _outcomes(34, 6, 0, 0), resolving=rp, metric_name="m", family=["m", "other"],
        alpha_step=0.05,
    )
    assert mcnemar_exact(6, 0) <= 0.05  # it clears its own Holm step
    assert rp.observable_floor == 6 / 40  # and the floor no longer forbids what just happened
    assert abs(v.diff) == pytest.approx(rp.observable_floor)
    assert v.floor_demoted is False
    assert v.distinguishable is True


def test_a_metric_past_the_holm_stop_is_not_tested_and_says_so() -> None:
    """§3.3 — 'stop at the first non-rejection and mark the remainder not tested'."""
    # b=8, c=1 is p = 0.039 and 17.5 pp, which clears both its Holm step of 0.05 and the 15.0 pp
    # floor — so it *would* be distinguishable, and the stop is the only thing preventing it.
    # Asserting the passthrough `v.holm_tested` on a case that was non-significant anyway left the
    # mutation `significant = raw_significant and not holm_tested` alive.
    outcomes, rp = _outcomes(31, 8, 1, 0), _rp(40, alpha_mdd=0.025)
    tested = verdict(
        outcomes, resolving=rp, metric_name="m", family=["m", "other"], alpha_step=0.05
    )
    assert tested.distinguishable is True

    v = verdict(
        outcomes, resolving=rp, metric_name="m", family=["m", "other"], alpha_step=0.05,
        holm_tested=False,
    )
    assert v.distinguishable is False
    assert v.floor_demoted is False
    assert v.holm_tested is False
    assert "Not tested: Holm–Bonferroni stops at the first non-rejection" in v.text


def test_the_unattainable_clause_quotes_the_mdds_alpha_not_the_floors() -> None:
    """Review P3-6 — `unattainable_clause` quoting `b_min(alpha_family)` instead of
    `b_min(alpha_mdd)` survived all 314 tests: the two differ only at k>1 and nothing asserted the
    figure at k=2.

    This is the note's own §7.1 corner (statistics review Pass 3, Part 1 item 3): at k=2,
    `n_units=13`, `DEFF=2` the rejection region is empty at `alpha_mdd = 0.025`, whose `b_min` is
    **7**, above the 6 floored effective units — while `b_min(alpha_family) = 6` *is* attainable,
    which is precisely why the floor sentence is printed separately and survives. A clause quoting
    6 here contradicts its own "no effect size attains 80% power" in the same sentence.
    """
    rp = resolving_power(
        13, unit_kind="item", design_effect=2.0, basis="measured",
        alpha_family=0.05, alpha_mdd=0.025,
    )
    assert rp.mdd80 is None
    assert b_min(0.05) == 6 and b_min(0.025) == 7  # the divergence the mutant hid
    clause = unattainable_clause(rp, "items")
    assert "b_min=7 net wins any outcome must reach at that alpha" in clause
    assert "alpha=0.025" in clause
    assert "b_min=6" not in clause


# --- P3-11 / n-ML-5: the design-effect guards ---------------------------------------------------


@pytest.mark.parametrize("deff", [0.0, 0.5, 0.999])
def test_resolving_power_refuses_a_design_effect_below_one(deff: float) -> None:
    """Review P3-11 and n-ML-5. Two findings, one guard.

    P3-11: removing `resolving_power`'s check survived the suite, and with it gone a
    `design_effect` of 0 raised a bare `ZeroDivisionError` from `n_units / design_effect` — at the
    one seam S2's runner supplies — instead of the named error.

    n-ML-5: the check was `<= 0`, so `DEFF = 0.5` was accepted although `-ml` §3.4 Rule 2's sketch
    says `>= 1.0` and both `verdict()` and `paired_cluster_bootstrap` enforce it. A design effect
    below 1 *doubles* `n_effective` and shrinks **both** printed bounds — anti-conservative in the
    module whose stated shape is *"the anti-conservative version does not typecheck"* — and the
    refusal arrived a layer later, after the figure had already been computed and could be printed.
    Rule 2's bound belongs at construction.
    """
    with pytest.raises(ValueError, match="design_effect"):
        resolving_power(
            40, unit_kind="item", design_effect=deff, basis="assumed",
            alpha_family=0.05, alpha_mdd=0.05,
        )


def test_resolving_power_accepts_a_design_effect_of_exactly_one() -> None:
    """The boundary from the other side: 1.0 is the no-clustering case and must not be refused."""
    rp = resolving_power(
        40, unit_kind="item", design_effect=1.0, basis="by-construction",
        alpha_family=0.05, alpha_mdd=0.05,
    )
    assert rp.n_effective == 40.0


def test_verdict_refuses_a_design_effect_below_one_before_choosing_an_instrument() -> None:
    """Review P3-11 — Rule 4's **precondition 4**. `-ml` §9 check 2(c) names the other three, all
    of which are tested; removing this one survived the suite.

    **Removing it also survives the obvious test**, and that is the trap worth pinning: at
    `DEFF = 0.5` the McNemar branch is not taken, so the decision falls through to
    `paired_cluster_bootstrap`, which raises the *same sentence* one layer down. A test that only
    reads the message cannot tell the precondition from its echo.

    So the property asserted is the **ordering** Rule 4 states — every precondition is checked
    before any instrument is selected. `bootstrap_seed=None` makes the two orders visibly
    different: with the check present the design effect is refused; without it, the run gets as far
    as choosing the bootstrap and complains about the missing seed instead, having already accepted
    an anti-conservative design effect.
    """
    below = dataclasses.replace(_rp(40), design_effect=0.5, n_effective=80.0)
    with pytest.raises(ValueError, match="precondition 4") as excinfo:
        verdict(_outcomes(34, 6, 0, 0), resolving=below, metric_name="m", family=["m"],
                bootstrap_seed=None)
    assert "bootstrap seed" not in str(excinfo.value)


def test_holm_steps_has_no_alpha_default_to_drift_from_alpha_family() -> None:
    """Reviews P3-13 and n-ML-4 — `holm_steps(p_values, *, alpha: float = 0.05)` restated the
    literal that `ALPHA_FAMILY` exists to be the single home of, 830 lines below that constant's
    own docstring: *"One home, because … a second literal `0.05` is how they drift apart"*.

    `report.py` always passes `pack.metrics.alpha_family`, so the default was **unreachable** —
    which is precisely why it would rot unnoticed. Plan §4 S1's surface sketch is at v1.7 the
    no-default form, and this pins the code to it: the parameter is keyword-only and required, so
    the family α has one home and no caller can inherit a stale one by omission.
    """
    params = inspect.signature(holm_steps).parameters
    assert params["alpha"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["alpha"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        holm_steps([0.01, 0.2])  # type: ignore[call-arg]


def test_the_wilson_bounds_are_clamped_to_a_probability() -> None:
    """Review P3-15 — removing `wilson_interval`'s `max(0.0, …)` / `min(1.0, …)` survived the
    suite. The effect is tiny but it is a *rate*: the unclamped bounds leave `[0, 1]` by one or two
    ulps, and a printed `-0.000` or a `1.000` that is not 1 is an instrument reporting an
    impossible probability.

    Both cells were re-derived here rather than quoted: sweeping every `n <= 400`, the unclamped
    upper bound at `s = n` exceeds 1.0 for **73** values of *n*, worst at `n = 16`
    (`1.0000000000000002`), and the unclamped lower bound at `s = 0` goes below 0.0 for **60**,
    worst at `n = 27` (`-6.94e-18`). So neither case is a one-off.
    """
    assert wilson_interval(16, 16)[1] == 1.0
    assert wilson_interval(0, 27)[0] == 0.0
    # ...and the clamp is not flattening an ordinary interval on the way past.
    lo, hi = wilson_interval(34, 40)
    assert 0.0 < lo < hi < 1.0


def test_an_observed_difference_exactly_at_the_mdd_takes_the_above_that_branch() -> None:
    """M-ML-7's comparison is strict (`<`), and the boundary is **reachable** rather than
    theoretical: mutating it to `<=` survived the round's first mutation pass.

    Swept for the equality `|diff| == mdd80` over every `6 <= n <= 120` at k = 1 and k = 2 — the
    MDD is ceilinged to a multiple of 0.001 and `|diff|` is `(b-c)/n`, so most `n` cannot produce
    it. Three cases can, all at k = 2: `n = 90` (mdd 10.0 pp, `|b-c| = 9`), `n = 100` (9.0 pp, 9)
    and `n = 120` (7.5 pp, 9). At exactly the MDD the pack **does** resolve the difference — the
    sentence beside it says *"resolves differences of >= 10.0 pp"* — so *"the observed 10.0 pp is
    below that"* is false, and it is false for the second time in the same sentence.
    """
    rp = _rp(90, alpha_mdd=0.025)
    assert rp.mdd80 == 0.1
    v = verdict(
        _outcomes(28, 21, 12, 29),
        resolving=rp,
        metric_name="m",
        family=["m", "other"],
        alpha_step=0.05,
    )
    assert v.distinguishable is False
    assert abs(v.diff) == rp.mdd80  # the equality the sweep found
    assert "the observed 10.0 pp is above that" in v.text
    assert "is below that" not in v.text
