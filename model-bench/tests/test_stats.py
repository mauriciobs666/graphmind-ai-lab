"""§5 test 6 — `stats`. Every number here comes from `docs/plans/small-model-benchmarking-ml.md`.

Not one figure in this file is a literal invented by the implementation or restated from the plan:
the note is the single source of truth for every formula, constant, threshold and tolerance
(plan §3.9, §7). Section citations are given per test so a reader can check the source.

The centre of gravity is `-ml` §3.4's six binding rules, which exist so that "the anti-conservative
version does not typecheck, and the honest one is the only one that runs".
"""

from __future__ import annotations

import inspect
import math
from fractions import Fraction

import pytest

from modelbench.stats import (
    _Z_95,
    BootstrapResult,
    DuplicateAnalysisUnit,
    PairedOutcomes,
    UnattainablePower,
    b_min,
    cluster_bootstrap,
    design_effect,
    effective_n,
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


@pytest.mark.parametrize("omitted", ["unit_kind", "design_effect", "basis", "alpha"])
def test_resolving_power_inputs_are_keyword_only_with_no_default(omitted: str) -> None:
    """`-ml` §3.4 Rule 2 / §9.2(b) — 'the absence of a default is itself the test'.

    A default of 1.0 for `design_effect` would rebuild gate B-1 by omission: the caller who forgets
    clustering is exactly the caller the gate found.
    """
    kwargs = {
        "unit_kind": "conversation",
        "design_effect": 1.0,
        "basis": "by-construction",
        "alpha": 0.05,
    }
    kwargs.pop(omitted)
    with pytest.raises(TypeError):
        resolving_power(12, **kwargs)

    params = inspect.signature(resolving_power).parameters
    assert params[omitted].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[omitted].default is inspect.Parameter.empty


def test_resolving_power_reports_effective_units() -> None:
    rp = resolving_power(
        48, unit_kind="conversation", design_effect=4.0, basis="assumed", alpha=0.05
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
    # `-ml` §7.1's α = 0.025 column — the step a two-member `verdictMetrics` family must clear.
    # 46.6 at n=15 is the note's v1.5 correction; 58.3 and 23.3 were already truncations.
    [
        (12.0, "65.6", "58.3"),
        (15.0, "54.2", "46.6"),
        (30.0, "28.7", "23.3"),
        (40.0, "21.9", "17.5"),
    ],
)
def test_mdd_and_floor_at_the_family_adjusted_alpha(n_eff, mdd_pp, floor_pp) -> None:
    assert f"{min_detectable_difference(n_eff, alpha=0.025) * 100:.1f}" == mdd_pp
    assert format_floor_pp(observable_floor(n_eff, alpha=0.025)) == floor_pp


def test_the_floor_truncation_is_guarded_against_the_bin_edge() -> None:
    """`7/40 = 0.175` is `174.99999999999997` bins in IEEE doubles (`-ml` §7.1's α=0.025, n=40 row).

    Naive truncation prints **17.4** for a floor the note publishes as **17.5**, so the `+ 1e-12`
    guard is load-bearing rather than defensive — it mirrors the `- 1e-12` on the MDD's ceiling.
    Every exact-bin row in the note's two floor columns is checked here in one place.
    """
    import math

    assert math.floor((7 / 40) / 0.001) == 174  # unguarded: the wrong bin
    for smallest, n, expected in [
        (6, 12, "50.0"), (6, 15, "40.0"), (6, 20, "30.0"), (6, 30, "20.0"), (6, 40, "15.0"),
        (6, 48, "12.5"), (6, 60, "10.0"), (6, 120, "5.0"), (7, 40, "17.5"),
    ]:
        assert format_floor_pp(smallest / n) == expected


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
    alpha: float = 0.05,
):
    return resolving_power(n, unit_kind=unit_kind, design_effect=deff, basis=basis, alpha=alpha)


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
            resolving=_rp(40, alpha=0.05),
            metric_name="m",
            family=["m", "other"],
        )
    # the same call at alpha = 0.05/2 is accepted
    verdict(
        _outcomes(34, 6, 0, 0),
        resolving=_rp(40, alpha=0.025),
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
    assert "anti-conservative under clustering — not the decision" in v.text


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
    for n in (12, 20, 30, 38, 40, 48, 85):
        for alpha in (0.05, 0.025):
            floor = b_min(alpha) / n
            for b in range(n + 1):
                for c in range(n + 1 - b):
                    if mcnemar_exact(b, c) <= alpha:
                        assert abs(b - c) / n >= floor


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
        40, unit_kind="item", design_effect=7.0, basis="measured", alpha=0.05
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


def test_verdict_decides_at_the_holm_step_it_is_given() -> None:
    """B-1 — `alpha_step` was built for exactly this and was passed by nothing.

    b=8, c=1 gives p = 0.0390625: not distinguishable at the plain Bonferroni α/k = 0.025 that
    every metric was decided at, distinguishable at its own Holm step of 0.05. Its 17.5 pp
    difference sits exactly **on** the α=0.025 floor, so Rule 7 does not demote it — see
    `test_rule_7_uses_the_family_adjusted_floor_the_report_prints` for why that matters.
    """
    outcomes, rp = _outcomes(31, 8, 1, 0), _rp(40, alpha=0.025)
    bonferroni = verdict(outcomes, resolving=rp, metric_name="m", family=["m", "other"])
    holm = verdict(
        outcomes, resolving=rp, metric_name="m", family=["m", "other"], alpha_step=0.05
    )
    assert bonferroni.distinguishable is False
    assert holm.distinguishable is True
    assert holm.alpha_used == 0.05


def test_rule_7_uses_the_family_adjusted_floor_the_report_prints() -> None:
    """Rule 7 is measured against `resolving.observable_floor`, i.e. the floor at **α/k**.

    `-ml` §7.1 fixes the printed floor at α/k because "a metric can be required to clear that
    step", and that is the sentence the report renders beside every verdict. Holm's own step for a
    rank-1 member is the looser α/(k−1), so the two can disagree at the margin: b=6, c=0 at n=40 is
    p=0.031, which clears a Holm step of 0.05, while its 15.0 pp sits below the 17.5 pp floor the
    report prints at α=0.025.

    The conservative resolution is deliberate: the decision follows the floor the report **prints**,
    so a reader can never find the verdict contradicting the honesty line two paragraphs below it.
    The cost is a Holm gain forgone at the boundary; the alternative is printing two floors, which
    multiplies the surface on which the instrument can contradict itself.
    """
    rp = _rp(40, alpha=0.025)
    v = verdict(
        _outcomes(34, 6, 0, 0), resolving=rp, metric_name="m", family=["m", "other"],
        alpha_step=0.05,
    )
    assert mcnemar_exact(6, 0) <= 0.05  # it clears its own Holm step
    assert abs(v.diff) < rp.observable_floor  # and is still below the printed floor
    assert v.distinguishable is False
    assert v.floor_demoted is True


def test_a_metric_past_the_holm_stop_is_not_tested_and_says_so() -> None:
    """§3.3 — 'stop at the first non-rejection and mark the remainder not tested'."""
    # b=8, c=1 is p = 0.039 and 17.5 pp, which clears both its Holm step of 0.05 and the α=0.025
    # floor — so it *would* be distinguishable, and the stop is the only thing preventing it.
    # Asserting the passthrough `v.holm_tested` on a case that was non-significant anyway left the
    # mutation `significant = raw_significant and not holm_tested` alive.
    outcomes, rp = _outcomes(31, 8, 1, 0), _rp(40, alpha=0.025)
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
