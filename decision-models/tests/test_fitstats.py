"""`decisionmodels.fitstats` (plan4 §4.4 U6f.5; §3.11 typed records; fit note §1; note §Q2-§Q3).

Every expected value is worked out by hand in a comment beside it; none is produced by the
implementation or by a second copy of its formula."""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import pytest

from decisionmodels.fitstats import (
    CALIBRATION_FLOOR_PER_CELL,
    LadderResult,
    ScoredCall,
    ceiling_s2,
    ceiling_s350,
    mcnemar_worse_p,
    mcnemar_worse_p_exact,
    mean_handoff_ms,
    roc_auc,
    split_half,
)

# --- one-sided exact McNemar: p_worse = sum_{i=b}^{m} C(m,i) / 2^m, 1 when m = 0 (note §Q3.1) ---


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (0, 0, 1.0),  # m = 0: defined as 1
        (5, 0, 1 / 32),  # m=5, i=5: C(5,5)=1 -> 1/32
        (4, 0, 1 / 16),  # m=4, i=4: 1 -> 1/16 (just above 0.05: does not reject)
        (0, 5, 1.0),  # m=5, i=0..5: 32/32 (candidate never worse)
        (3, 1, 5 / 16),  # m=4, i=3,4: 4+1 = 5 -> 5/16
        (2, 2, 11 / 16),  # m=4, i=2..4: 6+4+1 = 11 -> 11/16
        (8, 2, 56 / 1024),  # m=10, i=8..10: 45+10+1 = 56 -> 0.0546875 (> 0.05)
        (9, 2, 67 / 2048),  # m=11, i=9..11: 55+11+1 = 67 -> 0.0327148... (<= 0.05)
    ],
)
def test_mcnemar_worse_p_hand_computed(b: int, c: int, expected: float) -> None:
    assert mcnemar_worse_p(b, c) == pytest.approx(expected, abs=1e-15)


def test_mcnemar_is_one_sided_not_two_sided() -> None:
    # two-sided (5, 0) would be 2/32 = 0.0625; one-sided is 1/32 = 0.03125
    assert mcnemar_worse_p(5, 0) < 0.05 < 2 * (1 / 32)


def test_mcnemar_argument_order_is_candidate_worse_first() -> None:
    # b = candidate errs and incumbent does not. Swapped (b=0, c=5) must not reject.
    assert mcnemar_worse_p(5, 0) != mcnemar_worse_p(0, 5)


def test_mcnemar_exact_tail_is_a_fraction_equal_to_the_float_over_a_grid() -> None:
    for b in range(13):
        for c in range(13):
            exact = mcnemar_worse_p_exact(b, c)
            assert isinstance(exact, Fraction)
            assert mcnemar_worse_p(b, c) == float(exact)
            m = b + c
            # independent hand form: the sum of C(m, i) for i >= b, over 2^m
            expected = Fraction(sum(math.comb(m, i) for i in range(b, m + 1)), 2**m)
            assert exact == expected


def test_mcnemar_exact_tail_is_not_rounded_for_a_large_m() -> None:
    # m = 100: the numerator exceeds 2^53 so a float round trip would change the fraction
    m, b = 100, 60
    assert mcnemar_worse_p_exact(b, m - b) == Fraction(
        sum(math.comb(m, i) for i in range(b, m + 1)), 2**m
    )
    assert Fraction(float(mcnemar_worse_p_exact(b, m - b))) != mcnemar_worse_p_exact(b, m - b)


def test_mcnemar_exact_tail_literals() -> None:
    assert mcnemar_worse_p_exact(0, 0) == 1
    assert mcnemar_worse_p_exact(5, 0) == Fraction(1, 32)
    assert mcnemar_worse_p_exact(3, 1) == Fraction(5, 16)
    assert mcnemar_worse_p_exact(8, 2) == Fraction(56, 1024)
    # (13, 5): m = 18, i = 13..18: 8568 + 3060 + 816 + 153 + 18 + 1 = 12616 over 2^18
    assert mcnemar_worse_p_exact(13, 5) == Fraction(12616, 262144)
    assert Fraction(1, 21) < mcnemar_worse_p_exact(13, 5) <= Fraction(1, 20)


@pytest.mark.parametrize(("b", "c"), [(-1, 0), (0, -1)])
def test_mcnemar_refuses_negative_counts(b: int, c: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_worse_p(b, c)


# --- ROC-AUC, ties at 0.5: AUC = (wins + 0.5 * ties) / (n_pos * n_neg) ---


def test_auc_perfect_ranking_is_one() -> None:
    # every positive (0.9, 0.8) above every negative (0.3, 0.1): 4/4
    assert roc_auc([0.9, 0.8, 0.3, 0.1], [True, True, False, False]) == 1.0


def test_auc_reversed_ranking_is_zero() -> None:
    assert roc_auc([0.1, 0.3, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_auc_constant_score_is_exactly_half() -> None:
    # all 2*3 = 6 pairs tie: 6 * 0.5 / 6
    assert roc_auc([0.7] * 5, [True, False, True, False, False]) == 0.5


def test_auc_mixed_ranking() -> None:
    # pos {0.8, 0.4}, neg {0.6, 0.2}: (0.8>0.6) 1, (0.8>0.2) 1, (0.4<0.6) 0, (0.4>0.2) 1 -> 3/4
    assert roc_auc([0.8, 0.4, 0.6, 0.2], [True, True, False, False]) == 0.75


def test_auc_counts_a_tie_as_half() -> None:
    # pos {0.5, 0.9}, neg {0.5, 0.1}: tie 0.5 + 1 + 1 + 1 = 3.5 -> 3.5/4
    assert roc_auc([0.5, 0.9, 0.5, 0.1], [True, True, False, False]) == 0.875


def test_auc_unequal_class_sizes() -> None:
    # pos {0.7}, neg {0.7, 0.7, 0.1}: 0.5 + 0.5 + 1 = 2 -> 2/3
    assert roc_auc([0.7, 0.7, 0.7, 0.1], [True, False, False, False]) == pytest.approx(2 / 3)


def test_auc_does_not_depend_on_input_order() -> None:
    scores, labels = [0.8, 0.4, 0.6, 0.2], [True, True, False, False]
    assert roc_auc(scores[::-1], labels[::-1]) == 0.75


@pytest.mark.parametrize("labels", [[True, True], [False, False]])
def test_auc_refuses_a_single_class(labels: list[bool]) -> None:
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([0.2, 0.9], labels)


def test_auc_refuses_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        roc_auc([0.2, 0.9, 0.5], [True, False])


# --- ceilings (note §Q2.0): s_2 = (tL/2 - td)/tLH, s_350 = (350 - td)/tLH, all in ms ---


def test_s2_hand_fixture() -> None:
    # (1000/2 - 20) / 800 = 480/800 = 0.6
    assert ceiling_s2(t_llm_ms=1000.0, t_decision_ms=20.0, t_llm_handoff_ms=800.0) == Fraction(3, 5)


def test_s350_hand_fixture() -> None:
    # (350 - 20) / 600 = 330/600 = 0.55
    assert ceiling_s350(t_decision_ms=20.0, t_llm_handoff_ms=600.0) == Fraction(11, 20)


def test_s2_is_negative_when_decision_alone_misses_the_ceiling() -> None:
    # (60/2 - 20)/100 would be 0.1; with td = 35: (30 - 35)/100 = -0.05
    assert ceiling_s2(t_llm_ms=60.0, t_decision_ms=35.0, t_llm_handoff_ms=100.0) == Fraction(-1, 20)


def test_s2_and_s350_use_different_numerators() -> None:
    # s_2 = (500-20)/1000 = 0.48 ; s_350 = (350-20)/1000 = 0.33
    assert ceiling_s2(1000.0, 20.0, 1000.0) == Fraction(12, 25)
    assert ceiling_s350(20.0, 1000.0) == Fraction(33, 100)


def test_ceilings_are_exact_fractions_from_float_input() -> None:
    # a float converts exactly: 0.1 is not 1/10, so (350 - 0.1)/1 is not 3499/10
    assert ceiling_s350(0.1, 1.0) == 350 - Fraction(0.1)
    assert ceiling_s350(0.1, 1.0) != Fraction(3499, 10)
    # 0.25 and 0.5 are exact binary: (1000/2 - 0.25) / 0.5 = 499.75 / 0.5 = 999.5
    assert ceiling_s2(1000.0, 0.25, 0.5) == Fraction(1999, 2)


def test_ceilings_accept_mixed_int_float_fraction_input() -> None:
    # (200/2 - 0.5) / (1/4) = 99.5 * 4 = 398 ; (350 - 50) / (3/2) = 200
    assert ceiling_s2(200, 0.5, Fraction(1, 4)) == Fraction(398)
    assert ceiling_s350(Fraction(50), 1.5) == Fraction(200)
    assert isinstance(ceiling_s350(20, 600), Fraction)


def test_mean_handoff_is_exact_for_a_float_elapsed_time() -> None:
    # one call of 0.1 s: exactly 1000 * Fraction(0.1) ms, not Fraction(100)
    got = mean_handoff_ms([_call("a", "handoff", "ok", 0.1)])
    assert got == 1000 * Fraction(0.1)
    assert got != Fraction(100)


@pytest.mark.parametrize("t_h", [0.0, -1.0])
def test_ceilings_refuse_non_positive_handoff_latency(t_h: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        ceiling_s2(1000.0, 20.0, t_h)
    with pytest.raises(ValueError, match="positive"):
        ceiling_s350(20.0, t_h)


# --- t-bar_{L|H}: over band items, or over every hand-off call (fit note §1) ---


def _call(item: str, kind: str, outcome: str, seconds: float) -> ScoredCall:
    return ScoredCall(item, kind, None if kind == "handoff" else "t1", outcome, None, seconds)


_CALLS = [
    _call("a", "handoff", "ok", 1.0),
    _call("b", "handoff", "ok", 2.0),
    _call("c", "handoff", "failed", 6.0),
    _call("e", "handoff", "over-limit", 4.0),
    _call("a", "decision", "ok", 0.01),
    _call("d", "decision", "over-limit", 0.5),
]


def test_mean_handoff_over_every_handoff_call_includes_failed_and_ignores_decisions() -> None:
    # (1.0 + 2.0 + 6.0 + 4.0) / 4 = 3.25 s = 3250 ms; the decision calls (0.01, 0.5) are excluded
    assert mean_handoff_ms(_CALLS) == Fraction(3250)


def test_mean_handoff_over_band_items_only() -> None:
    # band {a, b}: (1.0 + 2.0) / 2 = 1.5 s = 1500 ms; c (6.0) is outside the band
    assert mean_handoff_ms(_CALLS, band_item_ids={"a", "b"}) == Fraction(1500)


def test_mean_handoff_counts_an_over_limit_handoff_with_its_time() -> None:
    # note §Q2.1: an over-limit item is a hand-off in every band and enters t̄_{L|H} with its time.
    # band {a, e}: (1.0 + 4.0) / 2 = 2.5 s = 2500 ms (dropping e would give 1000 ms)
    assert mean_handoff_ms(_CALLS, band_item_ids={"a", "e"}) == Fraction(2500)


def test_mean_handoff_band_keeps_a_failed_call_inside_it() -> None:
    # band {b, c}: (2.0 + 6.0) / 2 = 4.0 s
    assert mean_handoff_ms(_CALLS, band_item_ids={"b", "c"}) == Fraction(4000)


def test_mean_handoff_refuses_an_empty_selection() -> None:
    with pytest.raises(ValueError, match="no hand-off"):
        mean_handoff_ms([_call("a", "decision", "ok", 0.1)])
    with pytest.raises(ValueError, match="no hand-off"):
        mean_handoff_ms(_CALLS, band_item_ids=set())


def test_mean_handoff_refuses_a_band_item_without_a_handoff_call() -> None:
    with pytest.raises(ValueError, match="'zz'"):
        mean_handoff_ms(_CALLS, band_item_ids={"a", "zz"})


# --- split-half (note §Q2.4): deterministic, disjoint, order-independent ---


def test_split_half_alternates_over_sorted_ids() -> None:
    # sorted: a b c d e -> positions 0,2,4 = a,c,e ; positions 1,3 = b,d
    assert split_half(["d", "a", "c", "b", "e"]) == (("a", "c", "e"), ("b", "d"))


def test_split_half_even_count() -> None:
    # sorted: 1 2 3 4 -> (1, 3) and (2, 4)
    assert split_half(["4", "2", "3", "1"]) == (("1", "3"), ("2", "4"))


def test_split_half_refuses_fewer_than_two_and_duplicates() -> None:
    with pytest.raises(ValueError, match="at least two"):
        split_half(["a"])
    with pytest.raises(ValueError, match="duplicate"):
        split_half(["a", "b", "a"])


# --- typed records (plan4 §3.11) and the floor ---


def test_calibration_floor_is_fifty() -> None:
    assert CALIBRATION_FLOOR_PER_CELL == 50
    assert type(CALIBRATION_FLOOR_PER_CELL) is int


def test_ladder_result_fields_are_exactly_the_table() -> None:
    assert [f.name for f in dataclasses.fields(LadderResult)] == [
        "family", "rung", "qualityTarget", "policy", "key",
        "agreementUnmet", "latencyUnmet", "barValues",
    ]  # fmt: skip


def _ladder(**over: object) -> LadderResult:
    base: dict[str, object] = {
        "family": "handoff", "rung": 1, "qualityTarget": 0.94, "policy": {"s": 0.2},
        "key": (2, 0), "agreementUnmet": False, "latencyUnmet": True, "barValues": {"fr2": 1.5},
    }  # fmt: skip
    return LadderResult(**{**base, **over})  # type: ignore[arg-type]


def test_ladder_result_carries_values_and_both_flags_independently() -> None:
    r = _ladder()
    assert (r.family, r.rung, r.qualityTarget, r.key) == ("handoff", 1, 0.94, (2, 0))
    assert (r.agreementUnmet, r.latencyUnmet) == (False, True)
    assert r.barValues == {"fr2": 1.5}


def test_scored_call_hand_off_has_no_text() -> None:
    assert ScoredCall("i", "handoff", None, "ok", True, 0.1).textId is None


def test_ladder_result_allows_no_rung_on_the_fallback() -> None:
    assert _ladder(family="decision-only", rung=None).rung is None


@pytest.mark.parametrize(
    "bad",
    [
        {"family": "other"},
        {"rung": 0},
        {"rung": 5},
        {"rung": True},  # a bool is not a rung number
        {"rung": 1.0},
        {"family": "handoff", "rung": None},  # qualityTarget 0.94 is not mcnemar-nonreject
        {"family": "handoff", "rung": None, "qualityTarget": "D_c<=0"},
    ],
)
def test_ladder_result_refuses_closed_set_violations(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _ladder(**bad)


def test_ladder_result_allows_no_rung_on_the_hand_off_family_only_for_mcnemar_nonreject() -> None:
    r = _ladder(family="handoff", rung=None, qualityTarget="mcnemar-nonreject")
    assert r.rung is None
    with pytest.raises(ValueError, match="mcnemar-nonreject"):
        _ladder(family="handoff", rung=None, qualityTarget="D_c<=0")


def test_ladder_result_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _ladder().rung = 2  # type: ignore[misc]


def test_scored_call_fields_are_exactly_the_table() -> None:
    assert [f.name for f in dataclasses.fields(ScoredCall)] == [
        "itemId", "callKind", "textId", "outcome", "answer", "elapsedSeconds",
    ]  # fmt: skip


@pytest.mark.parametrize("outcome", ["ok", "over-limit", "failed"])
def test_scored_call_accepts_each_outcome(outcome: str) -> None:
    assert ScoredCall("i", "decision", "t", outcome, True, 0.1).outcome == outcome


@pytest.mark.parametrize(
    "bad",
    [
        {"outcome": "timeout"},
        {"callKind": "other"},
        {"elapsedSeconds": -0.1},
        {"callKind": "decision", "textId": None},  # a decision call belongs to a text
        {"callKind": "handoff", "textId": "t"},  # a hand-off call has no text
    ],
)
def test_scored_call_refuses_closed_set_violations(bad: dict[str, object]) -> None:
    base: dict[str, object] = {
        "itemId": "i", "callKind": "decision", "textId": "t", "outcome": "ok",
        "answer": None, "elapsedSeconds": 0.1,
    }  # fmt: skip
    with pytest.raises(ValueError):
        ScoredCall(**{**base, **bad})  # type: ignore[arg-type]
