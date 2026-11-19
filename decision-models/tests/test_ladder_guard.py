"""`decisionmodels.ladder.guard_ladder` (plan4 §4.4 U6f.6; note §Q2.1; fit note §1-§2).

Every fixture is worked out by hand in the comment beside it. Unless a fixture says otherwise: all
hand-off calls are correct and take 1 s, decision calls take 0 s, so `t̄_L = t̄_{L|H} = 1000 ms` and
`s_2 = (1000/2 - 0) / 1000 = 1/2` (rung margin: `s <= 9/20`). A decision call of `d` seconds gives
`s_2 = 1/2 - d` (for d = 0.125: 3/8; 0.1875: 5/16; 0.75: -1/4).

A model says yes iff `p >= h` and no iff `p < l`; items in `[l, h)` are handed off. "Wrong" is
graded against `labels` (True = yes)."""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from decisionmodels.fitstats import LadderResult, ScoredCall, mcnemar_worse_p
from decisionmodels.ladder import guard_ladder


class Fx:
    """A calibration fixture: one hand-off call and one decision call per item."""

    def __init__(self) -> None:
        self.calls: list[ScoredCall] = []
        self.labels: dict[str, bool] = {}
        self.cells: dict[str, str] = {}
        self.n = 0

    def add(
        self,
        cell: str,
        label: bool,
        p: float | str,
        *,
        llm_right: bool = True,
        hand_s: float = 1.0,
        dec_s: float = 0.0,
        hand: str = "ok",
        hand_answer: object = "default",
    ) -> str:
        """`p` is a float (an `ok` decision), `"over"` (over-limit) or `"failed"`."""
        iid = f"i{self.n}"
        self.n += 1
        self.labels[iid] = label
        self.cells[iid] = cell
        answer = None if hand == "failed" else (label if llm_right else not label)
        if hand_answer != "default":
            answer = hand_answer
        self.calls.append(ScoredCall(iid, "handoff", None, hand, answer, hand_s))
        if p == "over":
            dec = ("over-limit", None)
        elif p == "failed":
            dec = ("failed", None)
        else:
            dec = ("ok", p)
        self.calls.append(ScoredCall(iid, "decision", "t1", dec[0], dec[1], dec_s))
        return iid

    def group(self, count: int, *args, **kw) -> None:  # type: ignore[no-untyped-def]
        for _ in range(count):
            self.add(*args, **kw)

    def run(self, family: str, text_id: str = "t1") -> LadderResult:
        return guard_ladder(
            self.calls, labels=self.labels, cells=self.cells, text_id=text_id, family=family
        )


def perfect4(**kw) -> Fx:  # type: ignore[no-untyped-def]
    """.1 F, .2 F, .8 T, .9 T in cell a; model and LLM both right on every item."""
    fx = Fx()
    for label, p in ((False, 0.1), (False, 0.2), (True, 0.8), (True, 0.9)):
        fx.add("a", label, p, **kw)
    return fx


def pairs_identical_p(**kw) -> Fx:  # type: ignore[no-untyped-def]
    """5 x (.4 T) and 5 x (.6 F) (inverted: only handing off all ten fixes them), plus 5 x (.1 F)
    and 5 x (.9 T) that the model gets right. Cell a, 20 items."""
    fx = Fx()
    fx.group(5, "a", False, 0.1, **kw)
    fx.group(5, "a", True, 0.4, **kw)
    fx.group(5, "a", False, 0.6, **kw)
    fx.group(5, "a", True, 0.9, **kw)
    return fx


def pairs_spread_p(**kw) -> Fx:  # type: ignore[no-untyped-def]
    """`pairs_identical_p` with the inverted items at distinct p: T at .40-.44, F at .60-.64."""
    fx = Fx()
    fx.group(5, "a", False, 0.1, **kw)
    for p in (0.40, 0.41, 0.42, 0.43, 0.44):
        fx.add("a", True, p, **kw)
    for p in (0.60, 0.61, 0.62, 0.63, 0.64):
        fx.add("a", False, p, **kw)
    fx.group(5, "a", True, 0.9, **kw)
    return fx


# --- rung 1 ---


def test_rung_1_hand_off_family_picks_the_empty_band_when_the_model_is_perfect() -> None:
    # perfect4: only l = h = .8 has D = 0 at s = 0 (t=.1/.2 call .2 or .1 yes: wrong; t=.9: .8 no
    # wrong; t=0: all yes, .1/.2 wrong; t=1: all no, .8/.9 wrong). s = 0 <= 9/20.
    r = perfect4().run("handoff")
    assert (r.family, r.rung, r.qualityTarget) == ("handoff", 1, "D_c<=0")
    assert r.policy == {"l": 0.8, "h": 0.8, "s": F(0)}
    assert r.key == (F(0), 0, F(0))
    assert (r.agreementUnmet, r.latencyUnmet) == (False, False)
    assert r.barValues["D_c"] == {"a": 0}
    assert r.barValues["s2"] == F(1, 2)
    assert (r.barValues["tL_ms"], r.barValues["tLH_ms"], r.barValues["td_ms"]) == (1000, 1000, 0)


def test_rung_1_decision_only_family_reports_t_and_s_zero() -> None:
    r = perfect4().run("decision-only")
    assert (r.family, r.rung, r.qualityTarget) == ("decision-only", 1, "D_c<=0")
    assert r.policy == {"t": 0.8, "s": F(0)}
    assert r.key == (F(0), 0, F(0))


# --- rung 2 and 3 on the spread fixture ---
# 20 items: 5 x (.1 F), G1 = 5 T at .40-.44, G2 = 5 F at .60-.64, 5 x (.9 T); all LLM calls right.
# Wrong = (G1 items below l) + (G2 items at/above h). A band hands off a contiguous run; one that
# takes the top j of G1 and the bottom i of G2 leaves 10-j-i wrong. D = b (the LLM is never wrong),
# so D <= 0 needs j = i = 5 (s = 10/20 = 1/2). McNemar (b, 0) rejects iff b >= 5 (1/32 <= 1/20;
# b = 4 gives 1/16), so non-rejection needs j+i >= 6 (s >= 3/10). The tied key (4, 4, 3/10) over the
# j+i = 6 bands goes to the smallest l: j = 5, i = 1, i.e. band [.40, .61).


def test_rung_2_mcnemar_band_at_the_margin_when_no_zero_excess_band_fits() -> None:
    # dec 0.125 s: s_2 = 3/8, margin 13/40 >= 3/10: rung 1 needs s = 1/2; rung 2 takes s = 3/10.
    r = pairs_spread_p(dec_s=0.125).run("handoff")
    assert (r.rung, r.qualityTarget) == (2, "mcnemar-nonreject")
    assert r.policy == {"l": 0.4, "h": 0.61, "s": F(3, 10)}
    assert r.key == (4, 4, F(3, 10))
    assert r.barValues["D_c"] == {"a": 4}
    assert r.barValues["m_c"] == {"a": 4}
    assert r.barValues["p_worse"] == {"a": F(1, 16)}
    assert r.barValues["s2"] == F(3, 8)
    assert (r.agreementUnmet, r.latencyUnmet) == (False, False)


def test_rung_3_mcnemar_band_when_only_the_unmargined_ceiling_fits() -> None:
    # dec 0.1875 s: s_2 = (500 - 187.5)/1000 = 5/16; margin 5/16 - 1/20 = 21/80 < 3/10 <= 5/16.
    r = pairs_spread_p(dec_s=0.1875).run("handoff")
    assert (r.rung, r.qualityTarget) == (3, "mcnemar-nonreject")
    assert r.policy == {"l": 0.4, "h": 0.61, "s": F(3, 10)}
    assert r.barValues["s2"] == F(5, 16)
    assert (r.agreementUnmet, r.latencyUnmet) == (False, False)


def test_rung_2_margin_is_an_exact_fraction_not_a_float() -> None:
    # Hand-off times: the ten inverted items 5 s, the ten others 2 s. Every six-item band [.4+, ..)
    # of inverted items has t̄_{L|H} = 5000 ms; t̄_L = (50 + 20)/20 = 3.5 s;
    # s_2 = (3500/2)/5000 = 7/20, so s_2 - 1/20 = 3/10 = s exactly: rung 2 must accept it.
    # (A float 0.05 margin gives 7/20 - 0.05000000000000000277 < 3/10: those bands fail, rung 3
    # takes them instead and `rung == 2` fails.) The seven-item bands (s = 7/20) fail the margin.
    fx = Fx()
    fx.group(5, "a", False, 0.1, hand_s=2.0)
    for p in (0.40, 0.41, 0.42, 0.43, 0.44):
        fx.add("a", True, p, hand_s=5.0)
    for p in (0.60, 0.61, 0.62, 0.63, 0.64):
        fx.add("a", False, p, hand_s=5.0)
    fx.group(5, "a", True, 0.9, hand_s=2.0)
    r = fx.run("handoff")
    assert r.rung == 2
    assert r.policy == {"l": 0.4, "h": 0.61, "s": F(3, 10)}
    assert r.barValues["s2"] == F(7, 20)
    assert r.barValues["tLH_ms"] == F(5000)


def test_a_hand_off_answer_that_only_equals_the_label_is_not_a_right_answer() -> None:
    # An `ok` hand-off answering the int 1 for a True label is wrong (1 == True, but not `is`):
    # the model is right on that item and the LLM is not, so at l = h = .8 the cell has c = 1.
    fx = perfect4()
    fx.add("a", True, 0.9, hand_answer=1)
    r = fx.run("handoff")
    assert r.policy == {"l": 0.8, "h": 0.8, "s": F(0)}
    assert r.barValues["D_c"] == {"a": -1}
    assert r.barValues["m_c"] == {"a": 1}


def test_rung_3_equality_at_s_2_holds_with_non_dyadic_seconds() -> None:
    # Every hand-off call takes 0.1 s (not representable in binary): t̄_L = t̄_{L|H} exactly, so
    # s_2 = 1/2 and the ten inverted items handed off give s = 1/2 = s_2. Summing the times as
    # floats makes t̄_{L|H} differ from t̄_L in the last bit and the equality is lost (rung 4).
    r = pairs_identical_p(hand_s=0.1).run("handoff")
    assert (r.rung, r.qualityTarget) == (3, "D_c<=0")
    assert r.policy == {"l": 0.4, "h": 0.9, "s": F(1, 2)}
    assert r.barValues["s2"] == F(1, 2)
    assert r.latencyUnmet is False


def test_rung_3_not_rung_2_when_s_sits_between_the_margin_and_a_smaller_margin() -> None:
    # Hand-off times: inverted items 200 s, others 79 s. Six-item inverted bands: t̄_{L|H} =
    # 200000 ms, t̄_L = (2000 + 790)/20 s -> s_2 = (139500/2)/200000 = 279/800 = .34875.
    # s = 3/10 lies in (s_2 - 1/20, s_2 - 1/21] = (.29875, .30113]: rung 2 (margin 1/20) fails and
    # rung 3 takes it; a margin of 1/21 would accept it at rung 2.
    fx = Fx()
    fx.group(5, "a", False, 0.1, hand_s=79.0)
    for p in (0.40, 0.41, 0.42, 0.43, 0.44):
        fx.add("a", True, p, hand_s=200.0)
    for p in (0.60, 0.61, 0.62, 0.63, 0.64):
        fx.add("a", False, p, hand_s=200.0)
    fx.group(5, "a", True, 0.9, hand_s=79.0)
    r = fx.run("handoff")
    assert r.rung == 3
    assert r.barValues["s2"] == F(279, 800)
    assert r.barValues["s"] == F(3, 10)


def test_rung_3_takes_the_zero_excess_band_when_no_mcnemar_band_fits_the_margin() -> None:
    # identical p: any partial hand-off leaves all five of one group wrong (b = 5, 1/32: rejects);
    # only the ten inverted items handed off, band [.4, .9), s = 10/20 = 1/2 = s_2 (the equality
    # is accepted: s <= s_2), above the margin 9/20. Rung 3's D <= 0 stage, key (s, max D, width).
    r = pairs_identical_p().run("handoff")
    assert (r.rung, r.qualityTarget) == (3, "D_c<=0")
    assert r.policy == {"l": 0.4, "h": 0.9, "s": F(1, 2)}
    assert r.key == (F(1, 2), 0, F(0.9) - F(0.4))
    assert (r.agreementUnmet, r.latencyUnmet) == (False, False)


def test_rung_4_is_reached_when_latency_cannot_be_met_and_keeps_the_zero_excess_key() -> None:
    # same data, dec 0.75 s: s_2 = 1/2 - 3/4 = -1/4 < 0 so no band fits any ceiling; the smallest-s
    # D <= 0 band is [.4, .9) again.
    r = pairs_identical_p(dec_s=0.75).run("handoff")
    assert (r.rung, r.qualityTarget) == (4, "D_c<=0")
    assert r.policy == {"l": 0.4, "h": 0.9, "s": F(1, 2)}
    assert (r.agreementUnmet, r.latencyUnmet) == (False, True)
    assert r.barValues["s2"] == F(-1, 4)


def test_rung_4_tie_goes_to_the_narrower_band_then_the_smaller_l() -> None:
    # .25 T and .75 F (inverted), dec 0.75 s (s_2 = -1/4). The only D = 0 sets hand off both items:
    # [0, 1) and [.25, 1) (same set, widths 1 and 3/4): the narrower [.25, 1).
    fx = Fx()
    fx.add("a", True, 0.25, dec_s=0.75)
    fx.add("a", False, 0.75, dec_s=0.75)
    r = fx.run("handoff")
    assert r.rung == 4
    assert r.policy == {"l": 0.25, "h": 1.0, "s": F(1)}
    assert r.key == (F(1), 0, F(3, 4))


def test_rung_1_prefers_the_larger_cell_margin_over_the_sweep_order_at_equal_s() -> None:
    # (.2 F), (.4 F with the LLM wrong), (.6 T), one cell. At s = 0:
    #   t = .4: .2 no ok, .4 yes (model wrong, LLM wrong: concordant), .6 yes ok      -> D = 0
    #   t = .6: .2 no ok, .4 no (model right, LLM wrong: c = 1), .6 yes ok            -> D = -1
    #   t = 1:  .6 no wrong (b = 1), .4 no right (c = 1)                              -> D = 0
    #   t = 0 and .2: .2 yes wrong (b = 1), .4 yes neutral, .6 ok                     -> D = 1
    # The larger minimum margin (D = -1) wins over the earlier t = .4.
    fx = Fx()
    fx.add("a", False, 0.2)
    fx.add("a", False, 0.4, llm_right=False)
    fx.add("a", True, 0.6)
    r = fx.run("handoff")
    assert r.rung == 1
    assert r.policy == {"l": 0.6, "h": 0.6, "s": F(0)}
    assert r.barValues["D_c"] == {"a": -1}
    assert r.key == (F(0), -1, F(0))


# --- the note's worked example: (2,0) over (3,1) over (19,16) (decision-only, one worst cell) ---


def test_note_example_selects_the_least_worst_cell_band() -> None:
    # One cell W. The three classes of t (0 and .3 are equivalent: no item below .3) give the
    # worst cell (b, c) = (3, 1) for t in {0, .3}, (2, 0) for t = .7 and (19, 16) for t = 1.
    # Items (p, label, llm right?) = counts n1..n4 (LLM right: model wrong => b) and k1..k4 (LLM
    # wrong: model right => c): n1 = 2 x (.3 T), n2 = 3 x (.3 F), n3 = 17 x (.7 T), n4 = 0;
    # k1 = 1 x (.3 T), k2 = 0, k3 = 0, k4 = 16 x (.7 F).
    #   t in {0,.3}: wrong n2 + n4 = 3, right k1 + k3 = 1          -> (3, 1)
    #   t = .7:      wrong n1 + n4 = 2, right k2 + k3 = 0          -> (2, 0)
    #   t = 1:       wrong n1 + n3 = 19, right k2 + k4 = 16        -> (19, 16)
    # p_worse: (2,0) 1/4, (3,1) 5/16, (19,16) well above 1/20: all non-rejecting, no D <= 0.
    fx = Fx()
    fx.group(2, "W", True, 0.3)
    fx.group(3, "W", False, 0.3)
    fx.group(17, "W", True, 0.7)
    fx.group(1, "W", True, 0.3, llm_right=False)
    fx.group(16, "W", False, 0.7, llm_right=False)
    r = fx.run("decision-only")
    assert (r.rung, r.qualityTarget) == (2, "mcnemar-nonreject")
    assert r.policy == {"t": 0.7, "s": F(0)}
    assert r.barValues["D_c"] == {"W": 2}
    assert r.barValues["m_c"] == {"W": 2}
    assert r.barValues["p_worse"] == {"W": F(1, 4)}
    assert r.key == (2, 2, F(0))
    assert (r.agreementUnmet, r.latencyUnmet) == (False, False)


def test_rung_2_sorts_by_worst_cell_excess_before_discordance_before_total_excess() -> None:
    # R (LLM right): n1 = 2 x (.3 T), n2 = 3 x (.3 F), n3 = 2 x (.7 T), n4 = 0 -> b: {0,.3}: 3,
    #   .7: 2, 1: 4 (b <= 4, c = 0: p_worse >= 1/16, all non-rejecting).
    # Y (LLM wrong, model right -> c): k1 = 2 x (.3 T), k2 = 3 x (.3 F), k3 = 2 x (.7 T),
    #   k4 = 5 x (.7 F) -> m_Y: {0,.3}: k1+k3 = 4, .7: k2+k3 = 5, 1: k2+k4 = 8.
    # Keys (max D, max m): {0,.3}: (3, 4), .7: (2, 5), 1: (4, 8). Sum of D over cells: -1, -3, -4.
    # Least (D, m) is .7; (m, D) would pick t = 0, the total excess t = 1.
    fx = Fx()
    fx.group(2, "R", True, 0.3)
    fx.group(3, "R", False, 0.3)
    fx.group(2, "R", True, 0.7)
    fx.group(2, "Y", True, 0.3, llm_right=False)
    fx.group(3, "Y", False, 0.3, llm_right=False)
    fx.group(2, "Y", True, 0.7, llm_right=False)
    fx.group(5, "Y", False, 0.7, llm_right=False)
    r = fx.run("decision-only")
    assert r.rung == 2
    assert r.policy["t"] == 0.7
    assert r.key == (2, 5, F(0))


def test_rung_2_decision_only_tie_goes_to_the_smaller_t() -> None:
    # (.3 F), (.7 T), (.3 T), llm right: t=0 and t=.3 wrong 1 (.3 F says yes), t=.7 wrong 1 (.3 T
    # says no), t=1 wrong 2 (.7 T and .3 T). Keys (1,1,0) x3: the smallest t = 0. p_worse(1,0)=1/2.
    fx = Fx()
    fx.add("a", False, 0.3)
    fx.add("a", True, 0.7)
    fx.add("a", True, 0.3)
    r = fx.run("decision-only")
    assert r.rung == 2
    assert r.policy["t"] == 0.0
    assert r.key == (1, 1, F(0))


# --- over-limit and failed calls ---
# base: perfect4 plus one extra item.


def _perfect4_plus(p: str, **kw) -> Fx:  # type: ignore[no-untyped-def]
    fx = perfect4()
    fx.add("a", True, p, **kw)
    return fx


@pytest.mark.parametrize("llm_right", [True, False])
def test_over_limit_item_is_a_hand_off_in_every_band_with_its_time_and_adds_nothing_to_d(
    llm_right: bool,
) -> None:
    # perfect4 + one over-limit item whose hand-off call takes 4 s. s floor = 1/5 in every band:
    # t̄_L = (4*1 + 4)/5 = 1.6 s; the set handed off is {the over-limit item}: t̄_{L|H} = 4000 ms;
    # s_2 = 800/4000 = 1/5. Margin 3/20 < 1/5: not rung 1; rung 3 accepts s = s_2 exactly.
    # D_c is 0 whether the LLM was right or wrong on that item (no p: nothing to compare).
    r = _perfect4_plus("over", hand_s=4.0, llm_right=llm_right).run("handoff")
    assert (r.rung, r.qualityTarget) == (3, "D_c<=0")
    assert r.policy == {"l": 0.8, "h": 0.8, "s": F(1, 5)}
    assert r.barValues["D_c"] == {"a": 0}
    assert r.barValues["m_c"] == {"a": 0}
    assert (r.barValues["tL_ms"], r.barValues["tLH_ms"], r.barValues["s2"]) == (
        F(1600),
        F(4000),
        F(1, 5),
    )


@pytest.mark.parametrize("llm_right", [True, False])
def test_over_limit_item_is_concordant_and_no_hand_off_on_the_decision_only_family(
    llm_right: bool,
) -> None:
    # Same data. s = 0; t̄_{L|H} = every hand-off call = t̄_L = 1600 ms; s_2 = 800/1600 = 1/2.
    r = _perfect4_plus("over", hand_s=4.0, llm_right=llm_right).run("decision-only")
    assert r.rung == 1
    assert r.policy == {"t": 0.8, "s": F(0)}
    assert r.barValues["D_c"] == {"a": 0}
    assert r.barValues["m_c"] == {"a": 0}
    assert (r.barValues["tLH_ms"], r.barValues["s2"]) == (F(1600), F(1, 2))


@pytest.mark.parametrize("llm_right", [True, False])
@pytest.mark.parametrize("family", ["handoff", "decision-only"])
def test_a_failed_decision_call_is_concordant_and_never_a_hand_off(
    family: str, llm_right: bool
) -> None:
    # perfect4 + an item whose decision call failed: no p, adds 0 to D_c and m_c, s stays 0.
    r = _perfect4_plus("failed", llm_right=llm_right).run(family)
    assert r.rung == 1
    assert r.barValues["s"] == F(0)
    assert r.barValues["D_c"] == {"a": 0}
    assert r.barValues["m_c"] == {"a": 0}


def test_failed_hand_off_on_a_decided_item_is_fail_closed() -> None:
    # perfect4 with i2 (.8 T) hand-off failed (3 s): the LLM is wrong, the model right (says yes
    # at t = .8) so c = 1, D = -1, m = 1 at l = h = .8. s = 0 settings with D <= 0: t=.8 only
    # (D = -1). A failed call scored LLM-right would give D = 0, m = 0.
    fx = Fx()
    fx.add("a", False, 0.1)
    fx.add("a", False, 0.2)
    fx.add("a", True, 0.8, hand="failed", hand_s=3.0)
    fx.add("a", True, 0.9)
    r = fx.run("handoff")
    assert r.rung == 1
    assert r.policy == {"l": 0.8, "h": 0.8, "s": F(0)}
    assert r.barValues["D_c"] == {"a": -1}
    assert r.barValues["m_c"] == {"a": 1}
    assert r.key == (F(0), -1, F(0))
    assert r.barValues["tL_ms"] == F(1500)  # (1 + 1 + 3 + 1) / 4 s, the failed call's time in


def test_a_failed_hand_off_is_fail_closed_even_if_its_answer_field_equals_the_label() -> None:
    # same data as above, but the failed call carries answer True (= the label): still wrong.
    fx = Fx()
    fx.add("a", False, 0.1)
    fx.add("a", False, 0.2)
    fx.add("a", True, 0.8, hand="failed", hand_answer=True)
    fx.add("a", True, 0.9)
    r = fx.run("handoff")
    assert r.barValues["D_c"] == {"a": -1}


def test_failed_hand_off_time_enters_t_bar_l_given_h_when_the_item_is_handed_off() -> None:
    # i1 (.2 F) and i3 (.8 T) are right and take 1 s; i2 is over-limit (always handed off) and its
    # hand-off call FAILED after 4 s. t̄_L = 6/3 = 2 s; the band's set is {i2}: t̄_{L|H} = 4000 ms;
    # s = 1/3, s_2 = 1000/4000 = 1/4 < s: rung 4 (D <= 0, latency ignored) at l = h = .8.
    fx = Fx()
    fx.add("a", False, 0.2)
    fx.add("a", True, "over", hand="failed", hand_s=4.0)
    fx.add("a", True, 0.8)
    r = fx.run("handoff")
    assert r.rung == 4
    assert r.policy == {"l": 0.8, "h": 0.8, "s": F(1, 3)}
    assert (r.barValues["tL_ms"], r.barValues["tLH_ms"], r.barValues["s2"]) == (
        F(2000),
        F(4000),
        F(1, 4),
    )
    assert (r.agreementUnmet, r.latencyUnmet) == (False, True)


# --- fit note §2: the decision-only fallback `t` ---
# Keys are (max_c D_c, max_c m_c). Cell R gives wrong counts b (LLM right), cell Y gives c-only
# items (LLM wrong, model right). Item groups, with the t at which the model is WRONG:
#   n1 = (.3 T): t in {.7, 1}   n2 = (.3 F): t in {0, .3}
#   n3 = (.7 T): t = 1          n4 = (.7 F): t in {0, .3, .7}
# and RIGHT (k, LLM wrong):
#   k1 = (.3 T): t in {0, .3}   k2 = (.3 F): t in {.7, 1}
#   k3 = (.7 T): t in {0, .3, .7}   k4 = (.7 F): t = 1
# (t = 0 and t = .3 are equivalent: no item sits below .3.)


def _fallback_fixture(dec_s: float = 0.0) -> Fx:
    # R: n1 = 5, n2 = 6, n3 = 2, n4 = 0 -> b(t): {0,.3}: 6, .7: 5, 1: 7; c = 0 so p_worse = 1/64,
    #    1/32, 1/128: every t rejects.
    # Y: k1 = 6, k2 = 7, k3 = 2, k4 = 0 -> m_Y(t): {0,.3}: 8, .7: 9, 1: 7 (D_Y = -m_Y < 0).
    # Keys: {0,.3}: (6, 8); .7: (5, 9); 1: (7, 7). Lexicographic minimum: t = .7 (not the first,
    # the last, nor the minimum of m alone, which is t = 1, nor of D alone with the first-tie rule
    # t = 0 only if it came first: .7 has the strictly smallest D = 5).
    fx = Fx()
    fx.group(5, "R", True, 0.3, dec_s=dec_s)
    fx.group(6, "R", False, 0.3, dec_s=dec_s)
    fx.group(2, "R", True, 0.7, dec_s=dec_s)
    fx.group(6, "Y", True, 0.3, llm_right=False, dec_s=dec_s)
    fx.group(7, "Y", False, 0.3, llm_right=False, dec_s=dec_s)
    fx.group(2, "Y", True, 0.7, llm_right=False, dec_s=dec_s)
    return fx


def test_fixture_a_no_nonrejecting_t_carries_the_least_worst_key() -> None:
    r = _fallback_fixture().run("decision-only")
    assert r.rung is None
    assert r.qualityTarget == "mcnemar-nonreject"
    assert (r.agreementUnmet, r.latencyUnmet) == (True, False)
    assert r.policy == {"t": 0.7, "s": F(0)}
    assert r.key == (5, 9)
    assert r.barValues["D_c"] == {"R": 5, "Y": -9}
    assert r.barValues["m_c"] == {"R": 5, "Y": 9}
    assert r.barValues["p_worse"]["R"] == F(1, 32)


def test_fixture_a2_equal_worst_excess_is_broken_by_the_smaller_discordance() -> None:
    # R: n1 = 6, n2 = 6, n3 = 1, n4 = 0 -> b: {0,.3}: 6, .7: 6, 1: 7 (all reject).
    # Y: k1 = 9, k2 = 8, k3 = 0, k4 = 0 -> m_Y: {0,.3}: 9, .7: 8, 1: 8.
    # Keys: {0,.3}: (6, 9), .7: (6, 8), 1: (7, 8). The least (D, m) is .7 (D alone ties with t = 0).
    fx = Fx()
    fx.group(6, "R", True, 0.3)
    fx.group(6, "R", False, 0.3)
    fx.group(1, "R", True, 0.7)
    fx.group(9, "Y", True, 0.3, llm_right=False)
    fx.group(8, "Y", False, 0.3, llm_right=False)
    r = fx.run("decision-only")
    assert r.rung is None
    assert r.policy["t"] == 0.7
    assert r.key == (6, 8)


def test_fixture_b_identical_keys_carry_the_smaller_t() -> None:
    # 6 x (.3 F) and 7 x (.3 T), LLM right, one cell: wrong 6 at t = 0, 6 at .3, 7 at t = 1.
    # All reject (b >= 5, c = 0). Keys (6,6), (6,6), (7,7): the smaller of the tie, t = 0.
    fx = Fx()
    fx.group(6, "a", False, 0.3)
    fx.group(7, "a", True, 0.3)
    r = fx.run("decision-only")
    assert r.rung is None
    assert r.policy["t"] == 0.0
    assert r.key == (6, 6)
    assert r.agreementUnmet is True


def test_fixture_c_a_nonrejecting_t_wins_the_ordinary_mcnemar_rung() -> None:
    # (.3 F), (.7 T), (.3 T), LLM right: every t has b in {1, 1, 1, 2}, c = 0: p_worse 1/2 or 1/4,
    # none rejects, none has D <= 0: rung 2, the fallback is not reached.
    fx = Fx()
    fx.add("a", False, 0.3)
    fx.add("a", True, 0.7)
    fx.add("a", True, 0.3)
    r = fx.run("decision-only")
    assert (r.rung, r.qualityTarget, r.agreementUnmet) == (2, "mcnemar-nonreject", False)


def test_latency_only_fallback_takes_the_first_attainable_target_d_le_0() -> None:
    # perfect4 with dec 0.75 s: s_2 = (500 - 750)/1000 = -1/4 < s = 0: rungs 1-3 miss latency; the
    # decision-only family reports no rung 4. D <= 0 is attainable (t = .8): target D_c<=0, rung
    # None, outcome reads fit-infeasible: latency (latencyUnmet only), t from rung 1's key.
    r = perfect4(dec_s=0.75).run("decision-only")
    assert r.rung is None
    assert r.qualityTarget == "D_c<=0"
    assert r.policy == {"t": 0.8, "s": F(0)}
    assert r.key == (F(0), 0, F(0))
    assert (r.agreementUnmet, r.latencyUnmet) == (False, True)
    assert r.barValues["s2"] == F(-1, 4)


def test_latency_only_fallback_falls_to_mcnemar_when_no_zero_excess_t_exists() -> None:
    # fixture (c) with dec 0.75 s: no D <= 0, every t non-rejecting: target mcnemar-nonreject,
    # t = 0 (key (1, 1, 0) ties on 0, .3, .7).
    fx = Fx()
    fx.add("a", False, 0.3, dec_s=0.75)
    fx.add("a", True, 0.7, dec_s=0.75)
    fx.add("a", True, 0.3, dec_s=0.75)
    r = fx.run("decision-only")
    assert (r.rung, r.qualityTarget) == (None, "mcnemar-nonreject")
    assert r.policy["t"] == 0.0
    assert r.key == (1, 1, F(0))
    assert (r.agreementUnmet, r.latencyUnmet) == (False, True)


def test_an_agreement_miss_and_a_latency_miss_set_both_flags() -> None:
    # fixture (a) with dec 0.75 s: no non-rejecting t and s = 0 > s_2 = -1/4.
    r = _fallback_fixture(dec_s=0.75).run("decision-only")
    assert r.rung is None
    assert (r.agreementUnmet, r.latencyUnmet) == (True, True)
    assert r.policy["t"] == 0.7
    assert r.barValues["s2"] == F(-1, 4)


# --- inputs ---


def test_decision_calls_of_other_texts_are_ignored() -> None:
    fx = perfect4()
    for i, p in enumerate((0.9, 0.8, 0.2, 0.1)):  # a reversed score for a different text
        fx.calls.append(ScoredCall(f"i{i}", "decision", "t2", "ok", p, 0.0))
    assert fx.run("handoff", "t1").policy["l"] == 0.8
    # t2 reverses the order (.1 T, .2 T, .8 F, .9 F): no s = 0 setting has D <= 0, so not rung 1;
    # rung 2 (b <= 2 < 5 everywhere) takes it.
    assert fx.run("handoff", "t2").rung == 2


def test_t_bar_d_is_the_mean_of_the_texts_decision_calls_in_ms() -> None:
    # decision times 0.25, 0.5, 0.25, 0.5 s: mean 375 ms; s_2 = (500 - 375)/1000 = 1/8
    fx = Fx()
    for label, p, d in ((False, 0.1, 0.25), (False, 0.2, 0.5), (True, 0.8, 0.25), (True, 0.9, 0.5)):
        fx.add("a", label, p, dec_s=d)
    r = fx.run("handoff")
    assert r.barValues["td_ms"] == F(375)
    assert r.barValues["s2"] == F(1, 8)


# --- stuck items (fit note §2): p = 1.0, label no, LLM right: decided yes in every band ---


def _stuck_pair(count: int = 1, **kw) -> Fx:  # type: ignore[no-untyped-def]
    """`count` stuck items (1.0 F, LLM right) plus the perfect pair .2 F, .8 T, cell a."""
    fx = Fx()
    fx.group(count, "a", False, 1.0, **kw)
    fx.add("a", False, 0.2, **kw)
    fx.add("a", True, 0.8, **kw)
    return fx


def test_stuck_item_with_a_band_fixable_pair_reaches_rung_2() -> None:
    # S (1.0 F) is wrong in every band (b = 1). Inverted .4 T / .6 F with .1 F and .9 T: leaving
    # the inverted pair decided costs one more wrong (2, 2, s = 0); handing both off, band [.4, .9)
    # (s = 2/5 <= 9/20), leaves only S: (1, 1), p_worse 1/2. N = 5 non-stuck... N = 5 items:
    # S, .1 F, .4 T, .6 F, .9 T.
    fx = Fx()
    fx.add("a", False, 1.0)
    fx.add("a", False, 0.1)
    fx.add("a", True, 0.4)
    fx.add("a", False, 0.6)
    fx.add("a", True, 0.9)
    r = fx.run("handoff")
    assert (r.rung, r.qualityTarget) == (2, "mcnemar-nonreject")
    assert r.policy == {"l": 0.4, "h": 0.9, "s": F(2, 5)}
    assert r.key == (1, 1, F(2, 5))
    assert r.barValues["stuckCount"] == 1
    assert (r.agreementUnmet, r.latencyUnmet) == (False, False)


def test_stuck_item_with_a_slow_decision_gives_the_latency_fallback_on_the_hand_off_family() -> (
    None
):
    # same data, dec 0.75 s: s_2 = -1/4, no D <= 0 band exists (S), so rungs 1-4 all miss; the
    # fallback with latency ignored: mcnemar-nonreject, rung None, the (max D, max m, s) minimiser.
    fx = Fx()
    for label, p in ((False, 1.0), (False, 0.1), (True, 0.4), (False, 0.6), (True, 0.9)):
        fx.add("a", label, p, dec_s=0.75)
    r = fx.run("handoff")
    assert (r.family, r.rung, r.qualityTarget) == ("handoff", None, "mcnemar-nonreject")
    assert r.policy == {"l": 0.4, "h": 0.9, "s": F(2, 5)}
    assert r.key == (1, 1, F(2, 5))
    assert (r.agreementUnmet, r.latencyUnmet) == (False, True)
    assert r.barValues["stuckCount"] == 1


def test_stuck_item_with_the_perfect_pair_is_rung_2_at_s_zero_or_the_latency_fallback() -> None:
    # S + (.2 F) + (.8 T): b = 1 at l = h = .8 (s = 0), D = 1 everywhere: no rung 1.
    r = _stuck_pair().run("handoff")
    assert (r.rung, r.policy) == (2, {"l": 0.8, "h": 0.8, "s": F(0)})
    r = _stuck_pair(dec_s=0.75).run("handoff")
    assert (r.rung, r.qualityTarget) == (None, "mcnemar-nonreject")
    assert (r.agreementUnmet, r.latencyUnmet) == (False, True)


def test_stuck_items_that_reject_everywhere_give_the_agreement_outcome() -> None:
    # five stuck items: b >= 5, c = 0 in every band (1/32): nothing non-rejects. Minimiser of
    # (max D, max m, s): l = h = .8 (s = 0, the pair right): (5, 5, 0). Latency met (s = 0 <= 1/2).
    r = _stuck_pair(count=5).run("handoff")
    assert (r.rung, r.qualityTarget) == (None, "mcnemar-nonreject")
    assert r.policy == {"l": 0.8, "h": 0.8, "s": F(0)}
    assert r.key == (5, 5, F(0))
    assert (r.agreementUnmet, r.latencyUnmet) == (True, False)
    assert r.barValues["stuckCount"] == 5


def test_hand_off_agreement_fallback_breaks_an_excess_tie_by_the_smaller_s_not_the_smaller_l() -> (
    None
):
    # five stuck items (D = m = 5 whatever else) plus Q (.1 F) and X (.3 T), all right otherwise.
    # (D, m) = (5, 5) at l = h = .3 (s = 0) and at [.1, .3) and [0, .3) (Q handed off, s = 1/7):
    # the sweep's smaller l is the latter, the smaller s is (.3, .3), which is carried.
    fx = Fx()
    fx.group(5, "a", False, 1.0)
    fx.add("a", False, 0.1)
    fx.add("a", True, 0.3)
    r = fx.run("handoff")
    assert r.rung is None
    assert r.policy == {"l": 0.3, "h": 0.3, "s": F(0)}
    assert r.key == (5, 5, F(0))


def test_stuck_count_is_the_p_one_label_no_llm_right_ok_decision_items() -> None:
    fx = Fx()
    fx.add("a", False, 1.0)  # stuck
    fx.add("b", False, 1.0)  # stuck (other cell)
    fx.add("a", False, 1.0, llm_right=False)  # LLM wrong: fixable by nothing but not counted
    fx.add("a", True, 1.0)  # label yes: right at p = 1
    fx.add("a", False, 0.99)  # not 1.0
    fx.add("a", False, 0.0)  # p = 0 is never stuck
    fx.add("a", False, "over")  # no p
    fx.add("a", False, "failed")  # no p
    fx.add("a", False, 1.0, hand="failed")  # hand-off failed: the LLM is not right
    for family in ("handoff", "decision-only"):
        assert fx.run(family).barValues["stuckCount"] == 2


def test_stuck_count_is_zero_without_stuck_items() -> None:
    assert perfect4().run("handoff").barValues["stuckCount"] == 0


@pytest.mark.parametrize("pairs", [pairs_identical_p, pairs_spread_p, perfect4])
@pytest.mark.parametrize("d", [0.0, 0.75])
def test_hand_off_family_always_reaches_a_rung_below_five(pairs, d: float) -> None:  # type: ignore[no-untyped-def]
    assert pairs(dec_s=d).run("handoff").rung in (1, 2, 3, 4)


def test_cell_13_5_rejects_even_though_it_is_above_one_twenty_first() -> None:
    # One cell, every t has b >= 13: at t in {0, .3}: 13 x (.3 F) wrong, 5 x (.3 T, LLM wrong)
    # right: (b, c) = (13, 5), p_worse = 12616/262144 = .04812 <= 1/20: rejects. At t = 1:
    # 13 x (.3 T, LLM right) wrong: (13, 0). No non-rejecting t: fallback, agreement unmet.
    # (An alpha of 1/21 would accept (13, 5) and report rung 2.)
    fx = Fx()
    fx.group(13, "a", False, 0.3)
    fx.group(5, "a", True, 0.3, llm_right=False)
    fx.group(13, "a", True, 0.3)
    r = fx.run("decision-only")
    assert r.rung is None
    assert r.agreementUnmet is True
    assert r.policy["t"] == 0.0
    assert r.barValues["p_worse"] == {"a": F(12616, 262144)}
    assert r.key == (8, 18)


def test_cell_29_17_does_not_reject_though_it_is_below_one_nineteenth() -> None:
    # As above with (b, c) = (29, 17) at t in {0, .3}: m = 46, p_worse = .051903 > 1/20 (and
    # <= 1/19): non-rejecting, so rung 2 takes it; t = 1 gives (29, 0) and rejects.
    # (An alpha of 1/19 would reject it and fall to the fallback.)
    fx = Fx()
    fx.group(29, "a", False, 0.3)
    fx.group(17, "a", True, 0.3, llm_right=False)
    fx.group(29, "a", True, 0.3)
    r = fx.run("decision-only")
    assert r.rung == 2
    assert r.policy["t"] == 0.0
    assert F(1, 20) < r.barValues["p_worse"]["a"] <= F(1, 19)
    assert r.key == (12, 46, F(0))


def test_rung_2_hand_off_tie_between_h_values_of_an_empty_set_takes_the_smaller_h() -> None:
    # (.3 F), (.7 T), (.3 T), LLM right: every s = 0 setting has (b, c) >= (1, 0) and keys tie at
    # (1, 1, 0). (l, h) = (0, 0) and (0, .3) hand off nothing and decide the same way: the smaller
    # h, (0, 0), wins. (This is the only reachable l/h tie: equal-size windows over the p order
    # move l and h together, and equal sets differ only in the l = 0 / h = 0 edge values.)
    fx = Fx()
    fx.add("a", False, 0.3)
    fx.add("a", True, 0.7)
    fx.add("a", True, 0.3)
    r = fx.run("handoff")
    assert r.rung == 2
    assert r.policy == {"l": 0.0, "h": 0.0, "s": F(0)}


def test_p_worse_exact_agrees_with_the_float_helper() -> None:
    r = _fallback_fixture().run("decision-only")
    got = float(r.barValues["p_worse"]["R"])
    assert got == pytest.approx(mcnemar_worse_p(5, 0))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda fx: fx.labels.pop("i0"), "same items"),
        (lambda fx: fx.calls.append(ScoredCall("zz", "handoff", None, "ok", True, 1.0)), "unknown"),
        (lambda fx: fx.calls.append(ScoredCall("i0", "handoff", None, "ok", 1, 1.0)), "duplicate"),
        (lambda fx: fx.calls.pop(0), "without a hand-off"),
        (lambda fx: fx.calls.pop(1), "without a decision"),
    ],
)
def test_malformed_inputs_are_refused(mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    fx = perfect4()
    mutate(fx)
    with pytest.raises(ValueError, match=message):
        fx.run("handoff")


@pytest.mark.parametrize("bad", [1.5, -0.1, True, "x"])
def test_decision_p_outside_the_unit_interval_is_refused(bad) -> None:  # type: ignore[no-untyped-def]
    fx = perfect4()
    fx.calls[1] = ScoredCall("i0", "decision", "t1", "ok", bad, 0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fx.run("handoff")


def test_unknown_family_is_refused() -> None:
    with pytest.raises(ValueError, match="family"):
        perfect4().run("both")
