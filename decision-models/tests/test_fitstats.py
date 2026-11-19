"""`decisionmodels.fitstats` (plan4 §4.4 U6f.5; §3.11 typed records; fit note §1; note §Q2-§Q3).

Every expected value is worked out by hand in a comment beside it; none is produced by the
implementation or by a second copy of its formula."""

from __future__ import annotations

import dataclasses

import pytest

from decisionmodels.fitstats import (
    CALIBRATION_FLOOR_PER_CELL,
    LadderResult,
    ScoredCall,
    ceiling_s2,
    ceiling_s350,
    mcnemar_worse_p,
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
    assert ceiling_s2(t_llm_ms=1000.0, t_decision_ms=20.0, t_llm_handoff_ms=800.0) == pytest.approx(
        0.6
    )


def test_s350_hand_fixture() -> None:
    # (350 - 20) / 600 = 330/600 = 0.55
    assert ceiling_s350(t_decision_ms=20.0, t_llm_handoff_ms=600.0) == pytest.approx(0.55)


def test_s2_is_negative_when_decision_alone_misses_the_ceiling() -> None:
    # (60/2 - 20)/100 would be 0.1; with td = 35: (30 - 35)/100 = -0.05
    assert ceiling_s2(t_llm_ms=60.0, t_decision_ms=35.0, t_llm_handoff_ms=100.0) == pytest.approx(
        -0.05
    )


def test_s2_and_s350_use_different_numerators() -> None:
    # s_2 = (500-20)/1000 = 0.48 ; s_350 = (350-20)/1000 = 0.33
    assert ceiling_s2(1000.0, 20.0, 1000.0) == pytest.approx(0.48)
    assert ceiling_s350(20.0, 1000.0) == pytest.approx(0.33)


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
    assert mean_handoff_ms(_CALLS) == pytest.approx(3250.0)


def test_mean_handoff_over_band_items_only() -> None:
    # band {a, b}: (1.0 + 2.0) / 2 = 1.5 s = 1500 ms; c (6.0) is outside the band
    assert mean_handoff_ms(_CALLS, band_item_ids={"a", "b"}) == pytest.approx(1500.0)


def test_mean_handoff_counts_an_over_limit_handoff_with_its_time() -> None:
    # note §Q2.1: an over-limit item is a hand-off in every band and enters t̄_{L|H} with its time.
    # band {a, e}: (1.0 + 4.0) / 2 = 2.5 s = 2500 ms (dropping e would give 1000 ms)
    assert mean_handoff_ms(_CALLS, band_item_ids={"a", "e"}) == pytest.approx(2500.0)


def test_mean_handoff_band_keeps_a_failed_call_inside_it() -> None:
    # band {b, c}: (2.0 + 6.0) / 2 = 4.0 s
    assert mean_handoff_ms(_CALLS, band_item_ids={"b", "c"}) == pytest.approx(4000.0)


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
        {"family": "handoff", "rung": None},  # rung 4 always exists on the hand-off family
    ],
)
def test_ladder_result_refuses_closed_set_violations(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _ladder(**bad)


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
