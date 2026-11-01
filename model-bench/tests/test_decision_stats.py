"""`modelbench/decision_stats.py` — typed-decision-models U3c.1 (plan v1.3 §3.6; plan2 §3.6, §5
U3c; plan3 §3.6, §5 U3c).

Every expected figure below is the method note's (`docs/plans/typed-decision-models-ml.md`),
re-derived through this module's own formulas — never by running the note's scripts
(`docs/plans/typed-decision-models-ml/`). Exact `p_worse` values are hand-computed binomial tails,
written as fractions so the arithmetic is checkable by eye.
"""

from __future__ import annotations

import ast
import inspect
import math
import random
from fractions import Fraction
from pathlib import Path

import pytest

from modelbench import decision_stats, stats
from modelbench.decision_stats import p_worse
from modelbench.decisions import BAR_KIND_BY_ROLE
from modelbench.packs import load_pack
from modelbench.results import ItemTiming
from modelbench.scoring import classification

# --- p_worse (note §Q3.1) -------------------------------------------------------------------------


def test_p_worse_five_to_nothing_is_one_in_thirty_two():
    # Plan2 §5 U3c: `p_worse(5,0) = 0.03125` — C(5,5)/2^5.
    assert p_worse(5, 0) == 0.03125


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        # Note §Q3.6's worked example, one row per cell: Σ_{i=b}^{m} C(m,i) / 2^m by hand.
        (4, 1, (5 + 1) / 2**5),  # 0.1875
        (6, 2, (28 + 8 + 1) / 2**8),  # 0.1445
        (5, 0, 1 / 2**5),  # 0.0312
        (7, 1, (8 + 1) / 2**8),  # 0.0352
        # Note §Q3.6's closing sentence: the same discordance at 100 per cell, 8 vs 2.
        (8, 2, (45 + 10 + 1) / 2**10),  # 0.055
    ],
)
def test_p_worse_reproduces_the_worked_example(b, c, expected):
    assert p_worse(b, c) == pytest.approx(expected, rel=1e-12)


def test_p_worse_note_rounding_of_the_worked_example():
    assert round(p_worse(4, 1), 4) == 0.1875
    assert round(p_worse(6, 2), 4) == 0.1445
    assert round(p_worse(5, 0), 4) == 0.0312
    assert round(p_worse(7, 1), 4) == 0.0352
    assert round(p_worse(8, 2), 3) == 0.055


def test_p_worse_is_one_when_nothing_is_discordant():
    # Note §Q3.1: `p_worse = 1` when `m = 0`.
    assert p_worse(0, 0) == 1.0


def test_p_worse_sums_from_b_inclusive():
    # One discordant pair, candidate-worse: Σ_{i=1}^{1} = 1/2. A lower bound of b+1 gives 0, b-1
    # gives 1. Two pairs split 1/1: Σ_{i=1}^{2} = (2+1)/4.
    assert p_worse(1, 0) == 0.5
    assert p_worse(1, 1) == 0.75
    assert p_worse(0, 3) == 1.0


def test_p_worse_is_one_sided_toward_the_candidate_being_worse():
    # Candidate better (b < c) is never "worse": the mirror of 5/0 is 1.0, not 0.03125.
    assert p_worse(0, 5) == 1.0
    assert p_worse(1, 4) == pytest.approx(31 / 32)


@pytest.mark.parametrize("bad", [(-1, 0), (0, -1), (True, 0), (0, False), (1.0, 0), (0, "2")])
def test_p_worse_refuses_a_non_count(bad):
    with pytest.raises(ValueError):
        p_worse(*bad)


def test_rejection_needs_at_least_five_discordant_items():
    # Note §Q3.2: "on the exact test the floor is implied anyway: rejection needs b_w ≥ 5".
    assert all(p_worse(b, 0) > decision_stats.FR1_ALPHA for b in range(5))
    assert p_worse(5, 0) <= decision_stats.FR1_ALPHA


# --- pinned constants (plan2 §3.6) ----------------------------------------------------------------


def test_pinned_constants_carry_the_plan_values():
    assert decision_stats.FR1_ALPHA == 0.05
    assert decision_stats.MDD_POWER == 0.80
    assert decision_stats.MDD_GRID_PP == 0.1
    assert decision_stats.BOOTSTRAP_B == 10_000
    assert decision_stats.GUARD_CELL_METHOD_MIN == 50
    assert decision_stats.FR10_CLASS_METHOD_MIN == 52


def _p_class_passes(n, a):
    """Note §Q5.3: P(a class with true agreement `a` measures ≥ 0.90 under `10·k ≥ 9·n`)."""
    k_min = -(-9 * n // 10)
    return sum(math.comb(n, k) * a**k * (1 - a) ** (n - k) for k in range(k_min, n + 1))


def test_the_fr10_class_floor_is_the_notes_criterion_re_derived():
    # Note §Q1.3: the sawtooth's worsts 0.263 at 40, 0.219 at 50, 0.185 at 60; "every n ≥ 52
    # meets the criterion" (a class truly at 0.85 passes with probability ≤ 0.20).
    assert [round(_p_class_passes(n, 0.85), 3) for n in (40, 50, 60)] == [0.263, 0.219, 0.185]
    floor = decision_stats.FR10_CLASS_METHOD_MIN
    assert all(_p_class_passes(n, 0.85) <= 0.20 for n in range(floor, 401))
    assert _p_class_passes(floor - 1, 0.85) > 0.20


# --- mdd (note §Q3.3, §Q3.5) ----------------------------------------------------------------------


def _mdd(n, psi_hat, *, power_target=decision_stats.MDD_POWER):
    return decision_stats.mdd(
        n, psi_hat, alpha=decision_stats.FR1_ALPHA, power_target=power_target
    )


@pytest.mark.parametrize(
    ("n", "psi_hat", "pp"),
    [
        # Note §Q3.3's MDD₈₀ table, rows n = 50, 75, 100; the "ψ̂ ≤ 0.05" column at 0 and 0.05.
        (50, 0.0, 13.1),
        (50, 0.05, 13.1),
        (50, 0.10, 13.1),
        (50, 0.20, 16.4),
        (50, 0.30, 19.9),
        (75, 0.0, 8.8),
        (75, 0.05, 8.8),
        (75, 0.10, 9.4),
        (75, 0.20, 13.3),
        (75, 0.30, 16.3),
        (100, 0.0, 6.7),
        (100, 0.05, 6.7),
        (100, 0.10, 8.2),
        (100, 0.20, 11.6),
        (100, 0.30, 14.1),
    ],
)
def test_mdd_reproduces_the_note_table(n, psi_hat, pp):
    assert _mdd(n, psi_hat) == pytest.approx(pp / 100, abs=1e-12)


def test_mdd_reproduces_the_worked_example_second_row():
    # Note §Q3.6 row 2: 50 per cell, b=6, c=2 -> ψ̂ = 8/50 = 0.16, "detect ≥ 14.5 pp".
    assert _mdd(50, 8 / 50) == pytest.approx(0.145, abs=1e-12)


@pytest.mark.parametrize(
    ("n", "k", "pp"),
    [
        # Note §Q3.5: power target 0.8^{1/(K+1)} at ψ̂ = 0.10, K = 1…3.
        (50, 1, 15.2),
        (50, 2, 16.4),
        (50, 3, 17.2),
        (100, 1, 9.1),
        (100, 2, 9.4),
        (100, 3, 9.6),
    ],
)
def test_mdd_reuse_adjusted_reproduces_the_note(n, k, pp):
    target = decision_stats.MDD_POWER ** (1 / (k + 1))
    assert _mdd(n, 0.10, power_target=target) == pytest.approx(pp / 100, abs=1e-12)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_mdd_is_none_below_five_items(n):
    # Note §Q3.3 "Unattainable": below n = 5 no worsening can reach significance.
    assert _mdd(n, 0.0) is None
    assert _mdd(n, 0.5) is None


def test_mdd_is_a_number_from_five_items():
    # At n = 5 only m = 5 can reject (b = 5), so the search does reach 0.80, near the top.
    value = _mdd(5, 0.0)
    assert value is not None and 0.9 < value <= 1.0


def test_mdd_lands_on_the_grid():
    value = _mdd(75, 0.20)
    assert value == pytest.approx(0.133, abs=1e-12)  # note §Q3.3, n = 75 row
    assert round(value * 1000) == pytest.approx(value * 1000, abs=1e-9)


def test_mdd_never_reports_below_the_best_case():
    # Note §Q3.3 "Never below best case": a larger observed discordance only raises the limit.
    values = [_mdd(100, psi) for psi in (0.0, 0.05, 0.10, 0.20, 0.30)]
    assert values == sorted(values)


def test_mdd_takes_its_alpha_and_power_target_by_keyword_with_no_default():
    params = inspect.signature(decision_stats.mdd).parameters
    for name in ("alpha", "power_target"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": -1, "psi_hat": 0.1},
        {"n": 50.0, "psi_hat": 0.1},
        {"n": True, "psi_hat": 0.1},
        {"n": 50, "psi_hat": -0.01},
        {"n": 50, "psi_hat": 1.01},
        {"n": 50, "psi_hat": float("nan")},
        {"n": 50, "psi_hat": 0.1, "alpha": 0.0},
        {"n": 50, "psi_hat": 0.1, "alpha": 1.0},
        {"n": 50, "psi_hat": 0.1, "power_target": 0.0},
        {"n": 50, "psi_hat": 0.1, "power_target": 1.0},
        {"n": 50, "psi_hat": 0.1, "power_target": float("nan")},
    ],
)
def test_mdd_refuses_an_out_of_range_argument(kwargs):
    full = {"alpha": decision_stats.FR1_ALPHA, "power_target": decision_stats.MDD_POWER, **kwargs}
    with pytest.raises(ValueError):
        decision_stats.mdd(full.pop("n"), full.pop("psi_hat"), **full)


# --- parity_cell / ParityLine (note §Q3.1, §Q3.3–§Q3.5) -------------------------------------------

_Z_ONE_SIDED_95 = 1.6448536269514722  # note §Q3.4's z for U


def _pairs(a, b, c, d):
    """(candidate errs, incumbent errs) pairs: `a` both err, `b` candidate only, `c` incumbent
    only, `d` neither (note §Q3.1's b_w/c_w and §Q3.4's a/d)."""
    return [(True, True)] * a + [(True, False)] * b + [(False, True)] * c + [(False, False)] * d


def _cell(pairs, *, design_effect=1.0, k=0):
    return decision_stats.parity_cell(
        pairs, design_effect=design_effect, k=k, alpha=decision_stats.FR1_ALPHA
    )


@pytest.mark.parametrize(
    ("b", "c", "p", "psi", "mdd_pp", "state"),
    [
        # Note §Q3.6, guard, 50 per cell, K = 0: one row per cell.
        (4, 1, 0.1875, 0.10, 13.1, "holds"),
        (6, 2, 0.1445, 0.16, 14.5, "holds"),
        (5, 0, 0.0312, 0.10, 13.1, "worse"),
        (7, 1, 0.0352, 0.16, 14.5, "worse"),
    ],
)
def test_parity_cell_reproduces_the_worked_example(b, c, p, psi, mdd_pp, state):
    a = 3
    line = _cell(_pairs(a, b, c, 50 - a - b - c))
    assert (line.n, line.n_effective, line.b, line.c) == (50, 50, b, c)
    assert (line.e1, line.e2) == (a + b, a + c)
    assert (line.e1 - line.e2) / line.n == pytest.approx((b - c) / 50)  # the "diff" column
    assert round(line.p_worse, 4) == p
    assert line.psi_hat == pytest.approx(psi)
    assert line.mdd == pytest.approx(mdd_pp / 100, abs=1e-12)
    assert line.mdd_reuse is None  # K = 0: no reuse-adjusted limit is printed (note §Q3.5)
    assert line.state == state


def test_parity_cell_at_one_hundred_per_cell_holds_and_rules_out_more():
    # Note §Q3.6's closing sentence: 8 vs 2 at 100 per cell gives p_worse = 0.055, holds, ≥ 8.2 pp.
    line = _cell(_pairs(0, 8, 2, 90))
    assert round(line.p_worse, 3) == 0.055
    assert line.state == "holds"
    assert line.mdd == pytest.approx(0.082, abs=1e-12)


@pytest.mark.parametrize(("k", "pp"), [(1, 15.2), (2, 16.4), (3, 17.2)])
def test_parity_cell_prints_the_reuse_adjusted_limit_from_k_one(k, pp):
    # Note §Q3.5 at ψ̂ = 0.10, 50 per cell.
    line = _cell(_pairs(2, 3, 2, 43), k=k)
    assert line.psi_hat == pytest.approx(0.10)
    assert line.mdd == pytest.approx(0.131, abs=1e-12)
    assert line.mdd_reuse == pytest.approx(pp / 100, abs=1e-12)


def test_parity_cell_u_is_the_one_sided_mover_d_upper_end():
    a, b, c, d = 2, 6, 2, 40
    line = _cell(_pairs(a, b, c, d))
    assert line.U == stats.mover_d_interval(a, b, c, d, z=_Z_ONE_SIDED_95)[1]
    # One-sided 95%: above the point difference, below the two-sided 95% upper end.
    assert (b - c) / 50 < line.U < stats.mover_d_interval(a, b, c, d)[1]


def test_parity_cell_takes_n_for_the_limit_after_the_design_effect():
    # Note §Q3.3: n = floor(n_c / designEffect); ψ̂ stays m / n_c.
    line = _cell(_pairs(0, 6, 4, 91), design_effect=2.0)
    assert (line.n, line.n_effective) == (101, 50)
    assert line.psi_hat == pytest.approx(10 / 101)
    expected = decision_stats.mdd(
        50, 10 / 101, alpha=decision_stats.FR1_ALPHA, power_target=decision_stats.MDD_POWER
    )
    assert line.mdd == expected


def test_only_the_detection_limits_take_the_design_effect():
    # Note §Q3.2: "only the detection limit takes designEffect"; §Q3.3/§Q3.5: both limits are
    # stated at n = floor(n_c / designEffect). Review P16-m2.
    pairs = _pairs(3, 6, 4, 88)  # n_c = 101 -> n_effective = 50 at DE 2
    plain, inflated = _cell(pairs, design_effect=1.0, k=1), _cell(pairs, design_effect=2.0, k=1)
    psi = 10 / 101
    assert inflated.n_effective == 50
    assert inflated.mdd_reuse == decision_stats.mdd(
        50, psi, alpha=decision_stats.FR1_ALPHA, power_target=decision_stats.MDD_POWER ** 0.5
    )
    assert inflated.mdd_reuse != plain.mdd_reuse  # the inflation moved the reuse limit
    assert inflated.U == plain.U
    assert (inflated.p_worse, inflated.state) == (plain.p_worse, plain.state)


def test_a_p_worse_exactly_at_alpha_is_worse():
    # Note §Q3.1: "worse iff p_worse ≤ 0.05" — the boundary is inclusive. p_worse(5, 0) = 1/32.
    # Review P16-n1.
    line = decision_stats.parity_cell(_pairs(0, 5, 0, 45), design_effect=1.0, k=0, alpha=1 / 32)
    assert line.p_worse == 1 / 32
    assert line.state == "worse"


def test_parity_cell_decision_ignores_the_design_effect():
    # Note §Q3.2: "only the detection limit takes designEffect".
    pairs = _pairs(0, 5, 0, 45)
    assert _cell(pairs, design_effect=3.0).state == "worse"


def test_parity_cell_below_five_items_holds_with_no_limit():
    line = _cell(_pairs(0, 4, 0, 0))
    assert line.p_worse == 1 / 16
    assert line.state == "holds"
    assert line.mdd is None and line.mdd_reuse is None


def test_parity_cell_with_no_pairs_holds_with_nothing_to_estimate():
    line = _cell([], k=2)
    assert (line.n, line.e1, line.e2, line.b, line.c, line.p_worse) == (0, 0, 0, 0, 0, 1.0)
    assert line.state == "holds"
    assert line.U is None and line.psi_hat is None
    assert line.mdd is None and line.mdd_reuse is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"design_effect": 0.99},
        {"design_effect": float("nan")},
        {"k": -1},
        {"k": True},
        {"k": 1.0},
        {"alpha": 0.0},
    ],
)
def test_parity_cell_refuses_an_out_of_contract_argument(kwargs):
    full = {"design_effect": 1.0, "k": 0, "alpha": decision_stats.FR1_ALPHA, **kwargs}
    with pytest.raises(ValueError):
        decision_stats.parity_cell(_pairs(0, 1, 1, 10), **full)


@pytest.mark.parametrize("bad", [[(True,)], [(1, 0)], [(True, None)], ["xx"]])
def test_parity_cell_refuses_a_pair_that_is_not_two_bools(bad):
    with pytest.raises(ValueError):
        _cell(bad)


def test_parity_cell_shaping_arguments_are_keyword_only_with_no_default():
    params = inspect.signature(decision_stats.parity_cell).parameters
    for name in ("design_effect", "k", "alpha"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is inspect.Parameter.empty


# --- combine_replicates (plan2 §3.6 bar states; note §Q6) -----------------------------------------

MET = decision_stats.MET
NOT_MET = decision_stats.NOT_MET
ASSUMED = decision_stats.LOAD_CHANGED
LOW_N = decision_stats.SCORED_N_BELOW_MIN
UNSTABLE = "unstable across replicates — not met"
ONE_LOAD = "not assessable (only 1 load replicate(s) (FR-11a needs 2))"


def test_the_state_strings_are_the_plan_text():
    assert MET == "met"
    assert NOT_MET == "not met"
    assert ASSUMED == "not assessable (load changed mid-run)"
    assert LOW_N == "not assessable (scored n < 200)"


@pytest.mark.parametrize(
    ("states", "load_count", "expected"),
    [
        # Plan2 §5 U3c's five cases.
        ([MET, MET], 2, MET),
        ([MET, NOT_MET], 2, UNSTABLE),
        ([NOT_MET], 1, NOT_MET),  # not met, replicate 2 skipped (note §Q2.0)
        ([MET], 1, ONE_LOAD),
        ([MET, MET], 1, ONE_LOAD),  # the second was resident at start: same load
        # Rule 2 before rule 3/4; rule 3 before rule 4.
        ([NOT_MET, MET], 2, UNSTABLE),
        ([MET, NOT_MET, LOW_N], 2, UNSTABLE),
        ([NOT_MET, LOW_N], 2, NOT_MET),
        ([NOT_MET, NOT_MET], 2, NOT_MET),
        # Rule 4's two reasons.
        ([MET, LOW_N], 2, LOW_N),
        ([LOW_N, LOW_N], 2, LOW_N),
        ([], 0, "not assessable (only 0 load replicate(s) (FR-11a needs 2))"),
    ],
)
def test_combine_replicates_applies_the_four_rules_in_order(states, load_count, expected):
    assert decision_stats.combine_replicates(states, load_count) == expected


@pytest.mark.parametrize(
    ("states", "load_count", "expected"),
    [
        # Plan2 §5 U3c: met + assumed + met (fresh load) gives met.
        ([MET, ASSUMED, MET], 2, MET),
        # An assumed replicate is left out of the conjunction: it is neither met nor not met.
        ([NOT_MET, ASSUMED], 1, NOT_MET),
        ([ASSUMED, NOT_MET], 1, NOT_MET),
        ([MET, ASSUMED], 1, ONE_LOAD),
        ([ASSUMED, ASSUMED], 0, "not assessable (only 0 load replicate(s) (FR-11a needs 2))"),
    ],
)
def test_combine_replicates_leaves_an_assumed_replicate_out(states, load_count, expected):
    assert decision_stats.combine_replicates(states, load_count) == expected


def test_combine_replicates_carries_any_per_replicate_not_assessable_reason():
    timeout = "not assessable (chat timeout ≠ 180 s)"  # plan2 §5 U3c, a caller's reason
    assert decision_stats.combine_replicates([timeout, timeout], 2) == timeout
    assert (
        decision_stats.combine_replicates([LOW_N, timeout], 2)
        == "not assessable (scored n < 200; chat timeout ≠ 180 s)"
    )


@pytest.mark.parametrize(
    ("states", "load_count"),
    [
        ([MET, MET], 3),  # more loads than counted replicates
        ([MET, ASSUMED], 2),  # an assumed replicate is never a load
        ([MET], -1),
        ([MET], True),
        (["holds"], 1),  # a cell state, not a bar state
        (["unstable across replicates — not met"], 1),  # an overall state, not a replicate's
        (["not assessable"], 1),  # no reason
    ],
)
def test_combine_replicates_refuses_an_inconsistent_input(states, load_count):
    with pytest.raises(ValueError):
        decision_stats.combine_replicates(states, load_count)


# --- pair_items (note §Q4.3; plan2 §3.3, §3.6; plan3 §3.6) ----------------------------------------
#
# Every record below is built by the real scorer (`classification.score_answer`/`score_item`) on
# the shipped decision-point fixture pack, so the shapes are the scorer's, never this file's guess.

_DP_PACK = load_pack(Path(__file__).parent / "fixtures" / "packs" / "decision_point_guard")
_FA_U = "falseAdvanceRateByUnderstanding"
_FS_T = "falseSuspendRateByTurns"


def _cells(**present):
    """`PairedItems.outcomes` as it must read: every verdict cell (`()` when empty, P16-m1)
    plus the given metrics' pairs."""
    return {**dict.fromkeys(classification.PER_KIND_VERDICT_METRICS, ()), **present}


def _limits(**present):
    """`PairedItems.over_limit`: every verdict cell at 0 plus the given counts."""
    return {**dict.fromkeys(classification.PER_KIND_VERDICT_METRICS, 0), **present}


def _input(item_id, *, stratum="eval", expected=False, path="understanding", tier="clear_suspend"):
    return {"itemId": item_id, "stratum": stratum, "expected": expected, "path": path, "tier": tier}


def _timing(ms, withheld=None):
    kept = withheld in (None, "load")
    return ItemTiming(wallClockMs=ms if kept else None, calls=(), withheldFor=withheld)


def _score(item_input, answer, timing, **kw):
    keywords = {
        "decided_by": None,
        "probability": None,
        "failed_call": None,
        "reason": None,
        "input_tokens": None,
        **kw,
    }
    outcome = "fail" if answer is None else "pass"
    return classification.score_answer(
        item_input, answer, outcome, timing, pack=_DP_PACK, **keywords
    )


def _inc(item_input, advanced, ms=800.0, withheld=None):
    """The incumbent (chat) arm: a scored answer, or a withheld chat call (`score_item`)."""
    if withheld in ("timeout", "no_response"):
        return classification.score_item(item_input, None, _timing(ms, withheld), pack=_DP_PACK)
    return _score(item_input, advanced, _timing(ms, withheld))


def _decided(item_input, advanced, ms=20.0, withheld=None, by="decision-model"):
    return _score(
        item_input,
        advanced,
        _timing(ms, withheld),
        decided_by=by,
        probability=0.9 if by == "decision-model" else 0.5,
        input_tokens=40,
    )


def _failed(item_input, ms=5000.0, *, over_limit=False):
    """A kept decision-call failure (plan2 §3.3); over-limit on the `decisions` surface (plan3)."""
    reason = "over-limit" if over_limit else None
    return _score(item_input, None, _timing(ms), failed_call="decision", reason=reason)


def _over_limit_handoff(item_input, advanced, ms=900.0):
    """Plan3 §3.3's `handoff`-surface over-limit item whose hand-off succeeded."""
    return _score(item_input, advanced, _timing(ms), decided_by="handoff", reason="over-limit")


def _withheld_handoff(item_input, withheld="timeout", *, over_limit=False):
    """The guard's withheld hand-off: its chat call failed (`withhold` role)."""
    if over_limit:
        return _score(item_input, None, _timing(None, withheld), reason="over-limit")
    return _score(item_input, None, _timing(None, withheld), input_tokens=40)


def _pair(incumbent, candidate):
    return decision_stats.pair_items(incumbent, candidate, role="guard-judge")


def test_pair_items_pairs_two_answered_eval_items_on_every_declared_metric():
    item = _input("e1")  # label False: an advance is a false advance
    paired = _pair([_inc(item, False, 800.0)], [_decided(item, True, 20.0)])
    assert paired.outcomes == _cells(
        **{_FA_U: ((True, False),), "falseAdvanceRate": ((True, False),)}
    )
    assert paired.latency == (("e1", 800.0, 20.0),)
    assert paired.scored_n == 1
    assert paired.drops == {}


def test_pair_items_orders_everything_by_item_id():
    # Note §Q4.1: P is sorted by itemId, so the result depends only on the data and the seed.
    ids = ["e3", "e1", "e2"]
    incumbent = [_inc(_input(i), False, 100.0 * n) for n, i in enumerate(ids, 1)]
    candidate = [_decided(_input(i), i == "e2", 10.0 * n) for n, i in enumerate(ids, 1)]
    paired = _pair(incumbent, list(reversed(candidate)))
    assert [row[0] for row in paired.latency] == ["e1", "e2", "e3"]
    assert paired.latency == (("e1", 200.0, 20.0), ("e2", 300.0, 30.0), ("e3", 100.0, 10.0))
    assert paired.outcomes[_FA_U] == ((False, False), (True, False), (False, False))


def test_a_guard_decision_failure_pairs_as_concordant_and_adds_the_incumbents_time():
    # Note §Q4.3 row "Candidate decision call fails": a concordant pair, y_i = elapsed + x_i.
    item = _input("e1")
    paired = _pair([_inc(item, True, 800.0)], [_failed(item, 5000.0)])
    assert paired.outcomes[_FA_U] == ((True, True),)
    assert paired.latency == (("e1", 800.0, 5800.0),)
    assert paired.scored_n == 1
    assert paired.drops == {}


def test_a_guard_decision_failure_leaves_b_and_c_unchanged_and_adds_one_to_n():
    # Plan2 §5 U3c: `b`, `c` unchanged; `n_c` + 1. Note §Q4.3 "Why a concordant pair".
    base_inc, base_cand = [], []
    for n, (cand_adv, inc_adv) in enumerate([(True, False)] * 3 + [(False, True)] + [(False,) * 2]):
        item = _input(f"e{n}")
        base_inc.append(_inc(item, inc_adv))
        base_cand.append(_decided(item, cand_adv))
    before = _cell(_pair(base_inc, base_cand).outcomes[_FA_U])
    failing = _input("z-failed")
    after = _cell(
        _pair(base_inc + [_inc(failing, True)], base_cand + [_failed(failing)]).outcomes[_FA_U]
    )
    assert (after.b, after.c, after.p_worse) == (before.b, before.c, before.p_worse)
    assert after.n == before.n + 1


def test_a_guard_decision_failure_lands_in_the_incumbents_cell():
    # Plan2 §3.3 "Cell": the failed item has no metric of its own (`scoreable={}`).
    item = _input("t1", expected=True, path="turns", tier="clear_advance")
    failed = _failed(item)
    assert dict(failed.scoreable) == {}
    paired = _pair([_inc(item, False)], [failed])  # incumbent suspends a True label: an error
    assert paired.outcomes == _cells(
        **{_FS_T: ((True, True),), "falseSuspendRate": ((True, True),)}
    )


def test_a_side_check_decision_failure_is_dropped_from_the_side_check_and_counted():
    side = _input("s1", stratum="side-check")
    paired = _pair([_inc(side, False)], [_failed(side)])
    assert paired.outcomes == _cells()
    assert paired.latency == () and paired.scored_n == 0
    assert paired.drops == {("candidate", "side-check decision failure"): 1}


def test_an_answered_side_check_item_pairs_on_its_side_check_metric_only():
    side = _input("s1", stratum="side-check")
    paired = _pair([_inc(side, True)], [_decided(side, False)])
    assert paired.outcomes == _cells(sideCheckFalseAdvanceRate=((False, True),))
    # The side check does not affect pass/fail (v1.3 §3.5): no scored n, no FR-2 latency.
    assert paired.latency == () and paired.scored_n == 0


@pytest.mark.parametrize(
    ("inc_withheld", "cand_withheld", "drops"),
    [
        ("load", None, {("incumbent", "load"): 1}),
        (None, "load", {("candidate", "load"): 1}),
        ("load", "load", {("incumbent", "load"): 1, ("candidate", "load"): 1}),
    ],
)
def test_load_withholding_in_either_arm_drops_the_item_pairwise(inc_withheld, cand_withheld, drops):
    item = _input("e1")
    paired = _pair(
        [_inc(item, False, withheld=inc_withheld)], [_decided(item, True, withheld=cand_withheld)]
    )
    assert paired.outcomes == _cells() and paired.latency == () and paired.scored_n == 0
    assert paired.drops == drops


def test_a_load_withheld_decision_failure_is_dropped_not_paired():
    item = _input("e1")
    failed = _score(item, None, _timing(5000.0, "load"), failed_call="decision")
    paired = _pair([_inc(item, False)], [failed])
    assert paired.outcomes == _cells() and paired.drops == {("candidate", "load"): 1}


@pytest.mark.parametrize("why", ["timeout", "no_response"])
def test_guard_chat_call_withholding_in_either_arm_drops_the_item_pairwise(why):
    a, b = _input("e1"), _input("e2")
    paired = _pair(
        [_inc(a, False, withheld=why), _inc(b, False)],
        [_decided(a, True), _withheld_handoff(b, why)],
    )
    assert paired.outcomes == _cells() and paired.latency == () and paired.scored_n == 0
    assert paired.drops == {("incumbent", why): 1, ("candidate", why): 1}


def test_a_handed_off_over_limit_item_pairs_as_any_hand_off_with_its_own_wall_clock():
    # Plan3 §3.6 rule 1: y_i = its wallClockMs, and its answer is the hand-off LLM's own.
    item = _input("e1")
    paired = _pair([_inc(item, False, 800.0)], [_over_limit_handoff(item, True, 900.0)])
    assert paired.outcomes[_FA_U] == ((True, False),)
    assert paired.latency == (("e1", 800.0, 900.0),)
    assert paired.over_limit == _limits(**{_FA_U: 1, "falseAdvanceRate": 1})


def test_a_decisions_surface_over_limit_item_pairs_as_a_guard_decision_failure():
    # Plan3 §3.6 rule 2: concordant, y_i = elapsed + x_i.
    item = _input("e1")
    paired = _pair([_inc(item, False, 800.0)], [_failed(item, 30.0, over_limit=True)])
    assert paired.outcomes[_FA_U] == ((False, False),)
    assert paired.latency == (("e1", 800.0, 830.0),)
    assert paired.over_limit == _limits(**{_FA_U: 1, "falseAdvanceRate": 1})


def test_a_withheld_over_limit_hand_off_is_dropped_as_chat_withholding():
    item = _input("e1")
    paired = _pair([_inc(item, False)], [_withheld_handoff(item, "timeout", over_limit=True)])
    assert paired.outcomes == _cells() and paired.drops == {("candidate", "timeout"): 1}


def test_over_limit_counts_are_zero_on_every_paired_metric_when_there_are_none():
    # Plan3 §3.6: `overLimit 0/n` still prints, so the count exists at 0 for every cell.
    item = _input("e1")
    paired = _pair([_inc(item, False)], [_decided(item, False, by="handoff")])
    assert paired.over_limit == _limits(falseAdvanceRate=0)


def test_every_verdict_cell_is_present_even_with_no_paired_item():
    # Review P16-m1: three understanding/False items still yield all four FR-1 cells, empty ones
    # as `()` with over-limit 0, so no cell can silently vanish from the bar.
    items = [_input(f"e{i}") for i in range(3)]
    paired = _pair([_inc(i, False) for i in items], [_decided(i, False) for i in items])
    for cell in classification.PER_KIND_VERDICT_METRICS:
        assert cell in paired.outcomes and cell in paired.over_limit
    assert len(paired.outcomes[_FA_U]) == 3
    for cell in set(classification.PER_KIND_VERDICT_METRICS) - {_FA_U}:
        assert paired.outcomes[cell] == ()
        assert paired.over_limit[cell] == 0
    assert _cell(paired.outcomes[_FS_T]).n == 0


def test_pair_items_refuses_arms_over_different_items():
    with pytest.raises(ValueError, match="e2"):
        _pair([_inc(_input("e1"), False)], [_decided(_input("e2"), False)])


def test_pair_items_refuses_a_repeated_item_id():
    item = _input("e1")
    with pytest.raises(ValueError, match="e1"):
        _pair([_inc(item, False), _inc(item, True)], [_decided(item, False)])


def test_pair_items_refuses_a_decision_failure_on_the_incumbent_arm():
    item = _input("e1")
    with pytest.raises(ValueError):
        _pair([_failed(item)], [_decided(item, False)])


@pytest.mark.parametrize("role", ["chat-responder", "join-in-judge", "nonsense"])
def test_pair_items_pairs_speed_up_points_only(role):
    item = _input("e1")
    with pytest.raises(ValueError, match=role):
        decision_stats.pair_items([_inc(item, False)], [_decided(item, False)], role=role)


def test_the_verdict_cell_table_is_keyed_on_the_speed_up_roles_and_holds_the_scorers_cells():
    # AGENTS "A guard's reach lives in an asserted constant": keys bound to BAR_KIND_BY_ROLE's
    # speed-up rows, the value to the scorer's one declaration of the four per-kind rates.
    speed_up = {role for role, kind in BAR_KIND_BY_ROLE.items() if kind == "speed-up"}
    assert set(decision_stats._VERDICT_CELLS_BY_ROLE) == speed_up
    assert decision_stats._VERDICT_CELLS_BY_ROLE["guard-judge"] == (
        classification.PER_KIND_VERDICT_METRICS
    )


def test_a_pooled_only_eval_item_is_still_an_eval_item_by_its_verdict_cell():
    # An eval item is one whose incumbent declares a verdict cell; a pair outside every cell
    # (side check) adds nothing to the scored n. Every turns item lands on its own cell.
    item = _input("t1", expected=True, path="turns", tier="clear_advance")
    paired = _pair([_inc(item, True)], [_decided(item, False)])
    assert paired.outcomes[_FS_T] == ((True, False),)
    assert paired.scored_n == 1


# --- speedup_ratio (note §Q4.1) and mean_latency (note §Q4.2) -------------------------------------

_CI95 = (stats.LEVEL_CI95_LO, stats.LEVEL_CI95_HI)


def _ratio(x, y, *, seed=7, B=200, levels=_CI95):
    return decision_stats.speedup_ratio(x, y, seed=seed, B=B, levels=levels)


def test_speedup_ratio_is_a_ratio_of_means_not_a_mean_of_ratios():
    # Note §Q4.1: R = Σx_i / Σy_i. Here the mean of per-item ratios is (40 + 600/380) / 2 ≈ 20.8.
    result = _ratio([800.0, 600.0], [20.0, 380.0])
    assert result.point == 1400.0 / 400.0


def test_speedup_ratio_interval_collapses_when_every_pair_has_the_same_ratio():
    result = _ratio([40.0, 60.0, 100.0, 7.0], [20.0, 30.0, 50.0, 3.5])
    assert (result.point, result.lo, result.hi) == (2.0, 2.0, 2.0)


def test_speedup_ratio_interval_brackets_the_estimate_and_is_seeded():
    x = [800.0, 650.0, 900.0, 700.0, 760.0, 820.0, 610.0, 990.0]
    y = [20.0, 900.0, 25.0, 30.0, 700.0, 22.0, 18.0, 24.0]
    first = _ratio(x, y, seed=11, B=500)
    assert first.lo < first.point < first.hi
    assert _ratio(x, y, seed=11, B=500) == first
    # Compare the interval itself: the result echoes `seed`, so whole-object inequality would hold
    # even if the seed never reached the resampler.
    assert (_ratio(x, y, seed=12, B=500).lo, _ratio(x, y, seed=12, B=500).hi) != (
        first.lo,
        first.hi,
    )
    assert (first.B, first.seed) == (500, 11)


def test_speedup_ratio_reads_its_interval_through_stats_percentile(monkeypatch):
    # Note §Q4.1: "take stats.percentile" — never a second percentile estimator.
    seen = []
    real = stats.percentile

    def spy(values, *, level):
        values = list(values)
        seen.append((len(values), level))
        return real(values, level=level)

    monkeypatch.setattr(stats, "percentile", spy)
    _ratio([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], B=123)
    assert seen == [(123, stats.LEVEL_CI95_LO), (123, stats.LEVEL_CI95_HI)]


@pytest.mark.parametrize(
    ("x", "y", "kwargs"),
    [
        ([], [], {}),
        ([1.0], [1.0, 2.0], {}),
        ([1.0, float("nan")], [1.0, 2.0], {}),
        ([1.0, 2.0], [0.0, 0.0], {}),
        ([1.0, 2.0], [1.0, -1.0], {}),
        ([1.0, 2.0], [1.0, 2.0], {"B": 0}),
        ([1.0, 2.0], [1.0, 2.0], {"levels": (stats.LEVEL_CI95_HI, stats.LEVEL_CI95_LO)}),
    ],
)
def test_speedup_ratio_refuses_an_unusable_input(x, y, kwargs):
    with pytest.raises(ValueError):
        _ratio(x, y, **kwargs)


def test_mean_latency_is_the_mean_with_the_notes_single_arm_bootstrap():
    # Note §Q4.2: ȳ, and `stats.mean_bootstrap_interval(..., support=None)` at the given levels.
    y = [300.0, 410.0, 280.0, 350.0, 900.0, 20.0]
    result = decision_stats.mean_latency(y, seed=5, B=300, levels=_CI95)
    assert result.point == sum(y) / len(y)
    assert (result.lo, result.hi) == stats.mean_bootstrap_interval(
        y, B=300, seed=5, levels=_CI95, support=None
    )
    assert (result.B, result.seed) == (300, 5)


@pytest.mark.parametrize("name", ["speedup_ratio", "mean_latency"])
def test_bootstrap_arguments_are_keyword_only_with_no_default(name):
    # V1.3 §3.6: `levels` (and every decision-shaping argument) is required.
    params = inspect.signature(getattr(decision_stats, name)).parameters
    for arg in ("seed", "B", "levels"):
        assert params[arg].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[arg].default is inspect.Parameter.empty


# --- never stats.verdict / stats.holm_steps (note §Q3.1–§Q3.2; plan2 §3.6) ------------------------

_FORBIDDEN = frozenset({"verdict", "holm_steps"})


def _forbidden_references(source: str) -> set[str]:
    """Every forbidden name `source` mentions as a name, an attribute, an imported name or alias
    (each dotted part), or a whole string constant (the `getattr(stats, "verdict")` form)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.alias):
            names = node.name.split(".") + ([node.asname] if node.asname else [])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        else:
            continue
        found |= _FORBIDDEN.intersection(names)
    return found


@pytest.mark.parametrize(
    "snippet",
    [
        "stats.verdict(x)",
        "modelbench.stats.holm_steps(p, alpha=a)",
        "from modelbench.stats import verdict",
        "from modelbench.stats import holm_steps as h",
        "from modelbench import stats as s\ns.verdict()",
        "f = getattr(stats, 'holm_steps')",
        "import modelbench.stats.verdict",
        "verdict",
        "lambda: stats.verdict",
    ],
)
def test_the_reference_reader_sees_every_form_of_a_forbidden_reference(snippet):
    # Coverage probe: each form a call can be spelled in is seen. Bound: a name assembled at run
    # time ("hol" + "m_steps") is invisible to any text reader; the patched-out run below is the
    # backstop for that form on every exercised path.
    assert _forbidden_references(snippet)


def test_the_reference_reader_ignores_a_docstring_that_names_the_function():
    assert not _forbidden_references('"""never calls `stats.verdict` or `stats.holm_steps`."""')


def test_decision_stats_never_references_verdict_or_holm_steps():
    source = Path(decision_stats.__file__).read_text(encoding="utf-8")
    assert _forbidden_references(source) == set()


def test_every_public_function_runs_with_verdict_and_holm_steps_patched_out(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("decision_stats called a forbidden stats function")

    for name in _FORBIDDEN:
        monkeypatch.setattr(stats, name, refuse)
    item = _input("e1")
    paired = _pair([_inc(item, False)], [_decided(item, True)])
    _cell(_pairs(1, 6, 2, 41), k=2)
    _cell(paired.outcomes[_FA_U], k=1)
    decision_stats.combine_replicates([MET, NOT_MET], 2)
    _ratio([800.0, 700.0], [20.0, 25.0])
    decision_stats.mean_latency([300.0, 200.0], seed=1, B=50, levels=_CI95)


# --- seed and levels plumbing (coordinator integration mutants) -----------------------------------

_SPREAD_X = [800.0, 650.0, 900.0, 700.0, 760.0, 820.0, 610.0, 990.0, 300.0, 1200.0]
_SPREAD_Y = [20.0, 900.0, 25.0, 30.0, 700.0, 22.0, 18.0, 24.0, 400.0, 15.0]
_NARROW = (Fraction(1, 10), Fraction(9, 10))  # deliberately not the CI95 pair


def _reference_ratio_interval(x, y, *, seed, B, levels):
    """Note §Q4.1's resample, written out independently: B index draws of len(x), each a ratio
    of sums, from `random.Random(seed)`; the interval read by `stats.percentile` at `levels`."""
    rng = random.Random(seed)
    n = len(x)
    ratios = []
    for _ in range(B):
        draw = [rng.randrange(n) for _ in range(n)]
        ratios.append(sum(x[i] for i in draw) / sum(y[i] for i in draw))
    return stats.percentile(ratios, level=levels[0]), stats.percentile(ratios, level=levels[1])


@pytest.mark.parametrize("seed", [11, 4242])
@pytest.mark.parametrize("levels", [_CI95, _NARROW])
def test_speedup_ratio_resamples_with_the_declared_seed_at_the_declared_levels(seed, levels):
    result = decision_stats.speedup_ratio(_SPREAD_X, _SPREAD_Y, seed=seed, B=400, levels=levels)
    expected = _reference_ratio_interval(_SPREAD_X, _SPREAD_Y, seed=seed, B=400, levels=levels)
    assert (result.lo, result.hi) == expected
    assert result.lo < result.hi


@pytest.mark.parametrize("seed", [11, 4242])
@pytest.mark.parametrize("levels", [_CI95, _NARROW])
def test_mean_latency_resamples_with_the_declared_seed_at_the_declared_levels(seed, levels):
    result = decision_stats.mean_latency(_SPREAD_Y, seed=seed, B=400, levels=levels)
    expected = stats.mean_bootstrap_interval(
        _SPREAD_Y, B=400, seed=seed, levels=levels, support=None
    )
    assert (result.lo, result.hi) == expected
    assert result.lo < result.hi


@pytest.mark.parametrize(
    "estimate",
    [
        lambda seed: decision_stats.speedup_ratio(
            _SPREAD_X, _SPREAD_Y, seed=seed, B=400, levels=_CI95
        ),
        lambda seed: decision_stats.mean_latency(_SPREAD_Y, seed=seed, B=400, levels=_CI95),
    ],
    ids=["speedup_ratio", "mean_latency"],
)
def test_the_same_seed_reproduces_and_another_non_zero_seed_moves_the_interval(estimate):
    first, again, other = estimate(11), estimate(11), estimate(4242)
    assert (first.lo, first.hi) == (again.lo, again.hi)
    assert (first.lo, first.hi) != (other.lo, other.hi)
    # Neither non-zero seed's interval is seed 0's: a hard-coded seed cannot pass for both.
    zero = estimate(0)
    assert (zero.lo, zero.hi) not in {(first.lo, first.hi), (other.lo, other.hi)}


@pytest.mark.parametrize(
    "estimate",
    [
        lambda levels: decision_stats.speedup_ratio(
            _SPREAD_X, _SPREAD_Y, seed=11, B=400, levels=levels
        ),
        lambda levels: decision_stats.mean_latency(_SPREAD_Y, seed=11, B=400, levels=levels),
    ],
    ids=["speedup_ratio", "mean_latency"],
)
def test_narrower_levels_give_a_strictly_narrower_interval(estimate):
    wide, narrow = estimate(_CI95), estimate(_NARROW)
    assert wide.lo < narrow.lo < narrow.hi < wide.hi
