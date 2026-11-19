"""`predicate_ladder` / `choice_ladder` (plan4 §4.4 U6f.7; note §Q2.2; fit note §1, §4, §5).

Every fixture is worked out by hand beside it. Unless a fixture says otherwise: every hand-off call
is right and takes 1 s, every decision call takes 0 s, so `t̄_L = t̄_{L|H} = 1000 ms` and
`s_350 = 350/1000 = 7/20` (rung margin cap: `s <= 3/10`). A decision call of `d` seconds gives
`s_350 = 7/20 - d`; a band whose hand-off calls take `h` seconds gives `s_350 = 350/(1000 h)`.

Agreement is per true-answer class (predicate: `True`/`False`; choice: the label). A predicate model
says yes iff `p >= h`, no iff `p < l`, and hands off in `[l, h)`; a choice model keeps its choice
iff `confidence >= c` and hands off below. Fail-closed (fit note §4) is the answer "no" for a
predicate (right iff the label is `False`) and "no label" for a choice (always wrong).
`ScoredCall.answer` of an `ok` choice decision is the pair `(choice, confidence)`.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from decisionmodels.fitstats import LadderResult, ScoredCall
from decisionmodels.ladder import choice_ladder, predicate_ladder


class Fx:
    """A calibration fixture: one hand-off call and one decision call per item."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.calls: list[ScoredCall] = []
        self.labels: dict[str, object] = {}
        self.n = 0

    def add(
        self,
        label: object,
        x: object,
        *,
        llm_right: bool = True,
        hand_s: float = 1.0,
        dec_s: float = 0.0,
        hand: str = "ok",
        text: str = "t1",
        hand_answer: object = "default",
    ) -> str:
        """`x`: a predicate `p`, a choice `(choice, confidence)`, `"over"` or `"failed"`."""
        iid = f"i{self.n}"
        self.n += 1
        self.labels[iid] = label
        wrong = (not label) if self.kind == "predicate" else "wrong-label"
        answer = None if hand == "failed" else (label if llm_right else wrong)
        if hand_answer != "default":
            answer = hand_answer
        self.calls.append(ScoredCall(iid, "handoff", None, hand, answer, hand_s))
        if x == "over":
            dec = ("over-limit", None)
        elif x == "failed":
            dec = ("failed", None)
        else:
            dec = ("ok", x)
        self.calls.append(ScoredCall(iid, "decision", text, dec[0], dec[1], dec_s))
        return iid

    def group(self, count: int, *args, **kw) -> None:  # type: ignore[no-untyped-def]
        for _ in range(count):
            self.add(*args, **kw)

    def run(self, family: str, text_id: str = "t1") -> LadderResult:
        ladder = predicate_ladder if self.kind == "predicate" else choice_ladder
        return ladder(self.calls, labels=self.labels, text_id=text_id, family=family)


# --- predicate fixtures ---


def zone(a: int, b: int, *, e: int = 0, zone_s: float = 1.0, d: float = 0.0) -> Fx:
    """200 items, 100 per class. True: `100-a` at p .95 and `a` at p .40 (the model says no: wrong
    unless handed off; the LLM is wrong on the first `e` of them). False: `100-b` at .05 and `b` at
    .60 (the model says yes: wrong unless handed off). The zone's hand-off calls take `zone_s`.

    Settings (values {0, .05, .4, .6, .95, 1}), model wrong counts (True, False):
    - no hand-off, t in (.05, .4]: (0, b); t in (.6, .95]: (a, 0); others worse;
    - band [.4, .95): the whole zone, s = (a+b)/200, wrong (e, 0);
    - [.4, .6) alone leaves b wrong, [.6, .95) alone leaves a wrong (never better than no hand-off).
    """
    fx = Fx("predicate")
    fx.group(100 - a, True, 0.95, dec_s=d)
    fx.group(100 - b, False, 0.05, dec_s=d)
    for i in range(a):
        fx.add(True, 0.4, llm_right=i >= e, hand_s=zone_s, dec_s=d)
    fx.group(b, False, 0.6, hand_s=zone_s, dec_s=d)
    return fx


def interleaved(**kw) -> Fx:  # type: ignore[no-untyped-def]
    """100 per class. 70 True at .95, 70 False at .05, and 60 zone items at p = .4 + i/1000
    (i = 0..59; True when i is even, False when odd) on which the LLM is wrong as well."""
    fx = Fx("predicate")
    fx.group(70, True, 0.95, **kw)
    fx.group(70, False, 0.05, **kw)
    for i in range(60):
        fx.add(i % 2 == 0, 0.4 + i / 1000, llm_right=False, **kw)
    return fx


ZONE_BAND = {"l": 0.4, "h": 0.95}


def test_returns_a_ladder_result_with_the_family_it_was_asked_for() -> None:
    r = zone(10, 10, e=6).run("handoff")
    assert isinstance(r, LadderResult)
    assert r.family == "handoff"
    assert zone(10, 10, e=6).run("decision-only").family == "decision-only"


# --- rung 1: the highest T first ---
# zone(10, 10, e): no hand-off best is 90/100 in a class at s = 0; the zone band is s = 20/200 =
# 1/10 with True agreement (100 - e)/100 (False 100/100). So T = (100 - e)/100 at the band.


@pytest.mark.parametrize(("e", "t"), [(6, "T=0.94"), (7, "T=0.93"), (8, "T=0.92"), (9, "T=0.91")])
def test_rung_1_takes_the_highest_attainable_t_at_the_smallest_s(e: int, t: str) -> None:
    r = zone(10, 10, e=e).run("handoff")
    assert (r.rung, r.qualityTarget) == (1, t)
    assert r.policy == {**ZONE_BAND, "s": F(1, 10)}
    assert r.key == (-F(100 - e, 100), F(1, 10))
    assert (r.agreementUnmet, r.latencyUnmet) == (False, False)


def test_decision_only_family_never_uses_a_band() -> None:
    # The zone band would reach T = .94 at s = 1/10; the empty band reaches .90 at s = 0 only.
    r = zone(10, 10, e=6).run("decision-only")
    assert (r.rung, r.qualityTarget) == (1, "T=0.90")
    assert r.policy == {"t": 0.4, "s": F(0)}


def test_rung_1_with_t_090_takes_the_empty_band_at_s_zero() -> None:
    # e = 10: the band reaches only .90; no hand-off reaches .90 at s = 0 with the first sweep
    # setting (l = h = .4: the .6 items say yes, False 90/100).
    r = zone(10, 10, e=10).run("handoff")
    assert (r.rung, r.qualityTarget) == (1, "T=0.90")
    assert r.policy == {"l": 0.4, "h": 0.4, "s": F(0)}
    assert r.key == (F(-9, 10), F(0))


def test_rung_1_reports_every_bar_value_of_its_setting() -> None:
    r = zone(10, 10, e=6).run("handoff")
    assert r.barValues == {
        "agreement": {True: F(94, 100), False: F(1)},
        "nClass": {True: 100, False: 100},
        "minAgreement": F(94, 100),
        "s": F(1, 10),
        "s350": F(7, 20),
        "tLH_ms": F(1000),
        "tL_ms": F(1000),
        "td_ms": F(0),
        "unfixableByClass": {True: 0, False: 0},
    }


def test_rung_1_beats_a_higher_t_that_needs_the_ceiling() -> None:
    # d = 0.25: s_350 = 1/10, margin cap 1/20. The band (s = 1/10, T = .94) is outside the margin;
    # the empty band gives T = .90 inside it. Rung 1 over every T comes before rung 2.
    r = zone(10, 10, d=0.25).run("handoff")
    assert (r.rung, r.qualityTarget) == (1, "T=0.90")
    assert r.policy == {"l": 0.4, "h": 0.4, "s": F(0)}
    assert r.barValues["s350"] == F(1, 10)


def test_rung_1_accepts_s_exactly_at_the_margin() -> None:
    # 11 per zone: no hand-off gives .89 (no T), the band gives 1.00 at s = 22/200 = 11/100.
    # Band hand-off calls 2.1875 s (exact): t̄_{L|H} = 2187.5, s_350 = 4/25, cap 4/25-1/20 = 11/100.
    r = zone(11, 11, zone_s=2.1875).run("handoff")
    assert (r.rung, r.qualityTarget) == (1, "T=0.94")
    assert r.policy == {**ZONE_BAND, "s": F(11, 100)}
    assert r.barValues["tLH_ms"] == F(4375, 2)
    assert r.barValues["s350"] == F(4, 25)
    assert r.latencyUnmet is False


# --- rung 2 and rung 3 ---


def test_rung_2_equality_at_s_350_holds_with_non_dyadic_seconds() -> None:
    # Zone hand-off calls 0.7 s (not representable in binary; its exact value is a hair under 0.7),
    # 50 + 50 zone items of 200: s = 1/2 and s_350 = 350/(1000 * Fraction(0.7)), a hair above 1/2,
    # so exact arithmetic accepts it at rung 2. Float sums of the times land on the other side of
    # 1/2 (rung 3).
    r = zone(50, 50, zone_s=0.7).run("handoff")
    assert (r.rung, r.qualityTarget, r.latencyUnmet) == (2, "T=0.94", False)
    assert r.policy == {**ZONE_BAND, "s": F(1, 2)}
    assert r.barValues["s350"] == 350 / (1000 * F(0.7))


def test_choice_rung_2_equality_at_s_350_holds_with_non_dyadic_seconds() -> None:
    # 50 hard items per class of 100, 0.7 s each: s = 150/300 = 1/2, s_350 a hair above (as above).
    r = choice_zone(50, hard_s=0.7).run("handoff")
    assert (r.rung, r.qualityTarget, r.latencyUnmet) == (2, "T=0.94", False)
    assert r.policy == {"c": 0.9, "s": F(1, 2)}


def test_rung_2_accepts_s_exactly_at_the_ceiling() -> None:
    # 14 per zone: s = 28/200 = 7/50; hand-off calls 2.5 s: s_350 = 350/2500 = 7/50, cap 9/100.
    r = zone(14, 14, zone_s=2.5).run("handoff")
    assert (r.rung, r.qualityTarget) == (2, "T=0.94")
    assert r.policy == {**ZONE_BAND, "s": F(7, 50)}
    assert r.barValues["s350"] == F(7, 50)
    assert (r.agreementUnmet, r.latencyUnmet) == (False, False)


def test_rung_3_ignores_latency_and_reports_it_unmet() -> None:
    # 4 s hand-off calls: s_350 = 7/80 < s = 7/50.
    r = zone(14, 14, zone_s=4.0).run("handoff")
    assert (r.rung, r.qualityTarget) == (3, "T=0.94")
    assert r.policy == {**ZONE_BAND, "s": F(7, 50)}
    assert r.key == (F(-94, 100), F(7, 50))
    assert (r.agreementUnmet, r.latencyUnmet) == (False, True)
    assert r.barValues["s350"] == F(7, 80)


# --- rung 4 ---


def test_rung_4_maximises_the_minimum_per_class_agreement_not_the_mean() -> None:
    # Zone items 0..59, True at even i, False at odd, the LLM wrong on all: a hand-off helps no one.
    # Single threshold t after k zone items: True wrong ceil(k/2), False wrong 30 - floor(k/2).
    # k = 30: (15, 15) -> (.85, .85) mean .85; k = 0: (0, 30) -> (1, .70) mean .85 (equal mean, the
    # first in the sweep); k = 60: (30, 0) mean .85. Max-min .85 is unique at k = 30 (k = 29: 16).
    r = interleaved().run("handoff")
    t = 0.4 + 30 / 1000
    assert (r.rung, r.qualityTarget) == (4, "max-min-agreement")
    assert r.policy == {"l": t, "h": t, "s": F(0)}
    assert r.key == (F(-17, 20), F(0))
    assert (r.agreementUnmet, r.latencyUnmet) == (True, False)
    assert r.barValues["agreement"] == {True: F(17, 20), False: F(17, 20)}
    assert r.barValues["minAgreement"] == F(17, 20)


def test_rung_4_ties_go_to_the_smaller_s() -> None:
    # Bands that hand off the always-right easy items keep (.85, .85) at s > 0 (e.g. [.05, t)
    # hands off 70 easy False too); the empty band at k = 30 has s = 0 and wins.
    r = interleaved().run("handoff")
    assert r.barValues["s"] == F(0)


def test_rung_4_on_the_decision_only_family() -> None:
    r = interleaved().run("decision-only")
    t = 0.4 + 30 / 1000
    assert (r.rung, r.qualityTarget) == (4, "max-min-agreement")
    assert r.policy == {"t": t, "s": F(0)}
    assert r.agreementUnmet is True


# --- stuck items: never raises ---


def stuck_fixture() -> Fx:
    """True: 9 at p .9. False: 8 at p .1 and one at p = 1.0 (decided yes at every setting; the LLM
    is wrong on it too). False agreement is at most 8/9 < .90, so no T is attainable."""
    fx = Fx("predicate")
    fx.group(9, True, 0.9)
    fx.group(8, False, 0.1)
    fx.add(False, 1.0, llm_right=False)
    return fx


def test_a_stuck_item_reaches_rung_4_with_fit_infeasible_agreement_and_never_raises() -> None:
    # (l = h = .9): .1 no (right), .9 yes (right), 1.0 yes (wrong): (1, 8/9) at s = 0; the first
    # such setting in the sweep. Other settings reaching 8/9 hand off more (s > 0).
    r = stuck_fixture().run("handoff")
    assert (r.rung, r.qualityTarget) == (4, "max-min-agreement")
    assert r.agreementUnmet is True
    assert r.policy == {"l": 0.9, "h": 0.9, "s": F(0)}
    assert r.barValues["agreement"] == {True: F(1), False: F(8, 9)}
    assert r.barValues["unfixableByClass"] == {True: 0, False: 1}


def test_a_stuck_item_on_the_decision_only_family_also_never_raises() -> None:
    r = stuck_fixture().run("decision-only")
    assert (r.rung, r.qualityTarget, r.agreementUnmet) == (4, "max-min-agreement", True)
    assert r.policy == {"t": 0.9, "s": F(0)}
    assert r.barValues["unfixableByClass"] == {True: 0, False: 1}


# --- unfixableByClass: the literal mapping ---


def test_predicate_unfixable_counts_only_ok_decisions_at_p_one_against_no() -> None:
    fx = Fx("predicate")
    fx.add(False, 1.0)  # stuck (LLM right)
    fx.add(False, 1, llm_right=False)  # stuck (int 1, LLM wrong: the LLM is irrelevant)
    fx.add(False, 0.999)  # not stuck: p < 1 can be handed off
    fx.add(False, "over")  # over-limit: no ok decision call, excluded
    fx.add(False, "failed")  # failed decision: excluded
    fx.add(False, 0.1)
    fx.add(True, 1.0)  # a yes item at p = 1 is decided right
    fx.add(True, 1.0)
    fx.add(True, 0.0)  # a yes item at p = 0 is never stuck (l = 0)
    fx.add(True, 0.9)
    r = fx.run("handoff")
    assert r.barValues["unfixableByClass"] == {True: 0, False: 2}
    assert fx.run("decision-only").barValues["unfixableByClass"] == {True: 0, False: 2}


def test_unfixable_is_all_zero_when_no_item_is_stuck() -> None:
    assert zone(10, 10, e=6).run("handoff").barValues["unfixableByClass"] == {
        True: 0,
        False: 0,
    }


def test_choice_unfixable_counts_ok_decisions_at_confidence_one_with_another_choice() -> None:
    fx = Fx("choice")
    fx.add("order", ("question", 1.0))  # stuck, LLM right
    fx.add("order", ("order", 1.0))  # right choice: fine
    fx.add("order", ("question", 0.99))  # not stuck: c = 1 hands it off
    fx.add("order", "over")
    fx.add("order", "failed")
    fx.add("other", ("order", 1.0))  # stuck
    fx.add("other", ("complaint", 1.0), llm_right=False)  # stuck, LLM wrong
    fx.add("other", ("other", 0.7))
    fx.add("question", ("question", 1.0))
    fx.add("question", ("question", 0.2))
    expected = {"order": 1, "other": 2, "question": 0}
    assert fx.run("handoff").barValues["unfixableByClass"] == expected
    assert fx.run("decision-only").barValues["unfixableByClass"] == expected


# --- fail-closed scoring ---


def over_limit_fixture() -> Fx:
    """True: 9 at p .9 and one over-limit. False: 9 at p .1 and one over-limit. All LLM right."""
    fx = Fx("predicate")
    fx.group(9, True, 0.9)
    fx.add(True, "over")
    fx.group(9, False, 0.1)
    fx.add(False, "over")
    return fx


def test_over_limit_is_fail_closed_on_the_decision_only_family() -> None:
    # No hand-off model: the over-limit True item answers "no" (wrong: True 9/10), the False one
    # "no" (right: False 10/10). T = .90 at s = 0, with t = .9 first in the sweep.
    r = over_limit_fixture().run("decision-only")
    assert (r.rung, r.qualityTarget) == (1, "T=0.90")
    assert r.policy == {"t": 0.9, "s": F(0)}
    assert r.barValues["agreement"] == {True: F(9, 10), False: F(1)}


def test_over_limit_is_a_hand_off_in_every_band_on_the_hand_off_family() -> None:
    # Both over-limit items are handed off (right): s = 2/20 = 1/10 at every setting, agreement 1.
    r = over_limit_fixture().run("handoff")
    assert (r.rung, r.qualityTarget) == (1, "T=0.94")
    assert r.policy == {"l": 0.9, "h": 0.9, "s": F(1, 10)}
    assert r.barValues["agreement"] == {True: F(1), False: F(1)}


def test_a_failed_hand_off_is_fail_closed_and_its_time_enters_t_lh() -> None:
    # True: 9 at p .9 and an over-limit item whose hand-off failed after 3 s (closed "no": wrong).
    # False: 10 at p .1. The over-limit item is the only hand-off at l = h = .9: s = 1/20,
    # t̄_{L|H} = 3000, s_350 = 350/3000 = 7/60, cap 1/15 >= 1/20. True agreement 9/10 -> T = .90.
    fx = Fx("predicate")
    fx.group(9, True, 0.9)
    fx.add(True, "over", hand="failed", hand_s=3.0)
    fx.group(10, False, 0.1)
    r = fx.run("handoff")
    assert (r.rung, r.qualityTarget) == (1, "T=0.90")
    assert r.policy == {"l": 0.9, "h": 0.9, "s": F(1, 20)}
    assert r.barValues["tLH_ms"] == F(3000)
    assert r.barValues["s350"] == F(7, 60)
    assert r.barValues["agreement"] == {True: F(9, 10), False: F(1)}


def test_a_failed_hand_off_on_a_no_item_is_right_because_the_closed_answer_is_no() -> None:
    fx = Fx("predicate")
    fx.group(10, True, 0.9)
    fx.group(9, False, 0.1)
    fx.add(False, "over", hand="failed", hand_s=3.0)
    r = fx.run("handoff")
    assert (r.rung, r.qualityTarget) == (1, "T=0.94")
    assert r.barValues["agreement"] == {True: F(1), False: F(1)}
    assert r.barValues["tLH_ms"] == F(3000)


def test_a_failed_decision_is_never_handed_off_and_answers_no() -> None:
    # True: 9 at p .9 and one failed decision (answers "no": wrong, whatever the band). False: 10.
    fx = Fx("predicate")
    fx.group(9, True, 0.9)
    fx.add(True, "failed")
    fx.group(10, False, 0.1)
    for family, policy in (
        ("handoff", {"l": 0.9, "h": 0.9, "s": F(0)}),
        ("decision-only", {"t": 0.9, "s": F(0)}),
    ):
        r = fx.run(family)
        assert (r.rung, r.qualityTarget) == (1, "T=0.90")
        assert r.policy == policy
        assert r.barValues["agreement"] == {True: F(9, 10), False: F(1)}


def test_a_failed_decision_on_a_no_item_is_right() -> None:
    fx = Fx("predicate")
    fx.group(10, True, 0.9)
    fx.group(9, False, 0.1)
    fx.add(False, "failed")
    r = fx.run("decision-only")
    assert (r.rung, r.qualityTarget) == (1, "T=0.94")
    assert r.barValues["agreement"] == {True: F(1), False: F(1)}


def test_a_failed_decision_time_sits_in_t_d_not_in_t_lh() -> None:
    # 20 decision calls, the failed one 2 s, the rest 0: t̄_d = 2000/20 = 100 ms. No hand-off at
    # the chosen setting: t̄_{L|H} falls back to every hand-off call (1000 ms).
    fx = Fx("predicate")
    fx.group(10, True, 0.9)
    fx.group(9, False, 0.1)
    fx.add(False, "failed", dec_s=2.0)
    r = fx.run("handoff")
    assert r.barValues["td_ms"] == F(100)
    assert r.barValues["tLH_ms"] == F(1000)
    assert r.barValues["s350"] == F(1, 4)  # (350 - 100) / 1000


# --- the decision-only family's latency terms ---


def separable(**kw) -> Fx:  # type: ignore[no-untyped-def]
    fx = Fx("predicate")
    fx.group(10, True, 0.9, **kw)
    fx.group(10, False, 0.1, **kw)
    return fx


def test_decision_only_t_lh_is_the_mean_over_every_hand_off_call() -> None:
    fx = Fx("predicate")
    fx.group(10, True, 0.9, hand_s=1.0)
    fx.group(10, False, 0.1, hand_s=2.0)
    assert fx.run("decision-only").barValues["tLH_ms"] == F(1500)


def test_decision_only_rung_1_accepts_s_zero_exactly_at_the_margin() -> None:
    # hand-offs 2 s, decisions 0.25 s: s_350 = (350 - 250)/2000 = 1/20; s = 0 = 1/20 - 1/20.
    r = separable(hand_s=2.0, dec_s=0.25).run("decision-only")
    assert (r.rung, r.qualityTarget) == (1, "T=0.94")
    assert r.policy == {"t": 0.9, "s": F(0)}
    assert r.barValues["s350"] == F(1, 20)


def test_decision_only_rung_2_when_s_zero_is_inside_the_ceiling_only() -> None:
    # hand-offs 2.5 s: s_350 = 100/2500 = 1/25 < 1/20, still >= 0.
    r = separable(hand_s=2.5, dec_s=0.25).run("decision-only")
    assert (r.rung, r.latencyUnmet) == (2, False)


def test_decision_only_rung_3_when_even_s_zero_misses_the_ceiling() -> None:
    # decisions 0.5 s: s_350 = (350 - 500)/1000 < 0.
    r = separable(dec_s=0.5).run("decision-only")
    assert (r.rung, r.qualityTarget, r.latencyUnmet) == (3, "T=0.94", True)
    assert r.barValues["s350"] == F(-3, 20)


# --- inputs ---


def test_calls_of_another_text_are_ignored() -> None:
    fx = zone(10, 10, e=6)
    for i in range(200):
        fx.calls.append(ScoredCall(f"i{i}", "decision", "t2", "ok", 0.5, 0.0))
    assert fx.run("handoff").policy == {**ZONE_BAND, "s": F(1, 10)}
    assert fx.run("handoff", text_id="t2").rung is not None


def test_an_unknown_family_is_refused() -> None:
    with pytest.raises(ValueError, match="family"):
        zone(10, 10).run("both")


def test_a_missing_decision_call_is_refused() -> None:
    fx = separable()
    fx.calls = [c for c in fx.calls if not (c.itemId == "i0" and c.callKind == "decision")]
    with pytest.raises(ValueError, match="decision"):
        fx.run("handoff")


def test_a_missing_hand_off_call_is_refused() -> None:
    fx = separable()
    fx.calls = [c for c in fx.calls if not (c.itemId == "i0" and c.callKind == "handoff")]
    with pytest.raises(ValueError, match="hand-off"):
        fx.run("handoff")


def test_a_call_for_an_unknown_item_is_refused() -> None:
    fx = separable()
    fx.calls.append(ScoredCall("zz", "decision", "t1", "ok", 0.5, 0.0))
    with pytest.raises(ValueError, match="unknown item"):
        fx.run("handoff")


def test_a_duplicate_call_is_refused() -> None:
    fx = separable()
    fx.calls.append(fx.calls[1])
    with pytest.raises(ValueError, match="duplicate"):
        fx.run("handoff")


def test_no_items_is_refused() -> None:
    with pytest.raises(ValueError, match="no calibration items"):
        Fx("predicate").run("handoff")


@pytest.mark.parametrize("p", [-0.1, 1.5, True, "0.5", None])
def test_a_predicate_p_outside_zero_one_is_refused(p: object) -> None:
    fx = separable()
    fx.add(True, p)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fx.run("handoff")


@pytest.mark.parametrize(
    "answer", [("a", 1.5), ("a", -0.1), ("a", True), (1, 0.5), "ab", ("a",), None]
)
def test_a_malformed_choice_answer_is_refused(answer: object) -> None:
    fx = Fx("choice")
    fx.add("a", ("a", 0.9))
    fx.add("a", answer)
    with pytest.raises(ValueError, match="choice answer"):
        fx.run("handoff")


def test_a_predicate_label_that_is_not_a_bool_is_refused() -> None:
    fx = separable()
    fx.labels["i0"] = "yes"
    with pytest.raises(ValueError, match="label"):
        fx.run("handoff")


# --- choice fixtures ---
# Three classes of 100. Per class: `100-w` items answered right at confidence .9 and `w` hard items
# answered wrong at confidence .3 (the LLM right unless `llm_wrong`), whose hand-off calls take
# `hard_s`. c in {0, .3, .9, 1}: c = 0 decides everything (wrong w per class, s = 0); c = .9 hands
# off exactly the hard items (s = 3w/300 = w/100; wrong 0 if the LLM is right); c = .3 decides the
# hard items too (same as c = 0, s = 0, later in the sweep); c = 1 hands off everything.


def choice_zone(w: int, *, hard_s: float = 1.0, llm_wrong: bool = False, d: float = 0.0) -> Fx:
    fx = Fx("choice")
    classes = ("a", "b", "c")
    for k, label in enumerate(classes):
        other = classes[(k + 1) % 3]
        fx.group(100 - w, label, (label, 0.9), dec_s=d)
        fx.group(w, label, (other, 0.3), llm_right=not llm_wrong, hand_s=hard_s, dec_s=d)
    return fx


def test_choice_rung_1_takes_the_highest_t_at_the_smallest_s() -> None:
    # w = 7: c = 0 gives .93 at s = 0; c = .9 gives 1.00 at s = 21/300 = 7/100, so T = .94.
    r = choice_zone(7).run("handoff")
    assert (r.family, r.rung, r.qualityTarget) == ("handoff", 1, "T=0.94")
    assert r.policy == {"c": 0.9, "s": F(7, 100)}
    assert r.key == (F(-94, 100), F(7, 100))
    assert r.barValues["agreement"] == {"a": F(1), "b": F(1), "c": F(1)}
    assert r.barValues["unfixableByClass"] == {"a": 0, "b": 0, "c": 0}


def test_choice_rung_1_with_no_hand_off_needed_takes_c_zero() -> None:
    # w = 5: c = 0 already gives .95 >= .94 at s = 0.
    r = choice_zone(5).run("handoff")
    assert (r.rung, r.qualityTarget) == (1, "T=0.94")
    assert r.policy == {"c": 0, "s": F(0)}


def test_choice_rung_1_accepts_s_exactly_at_the_margin() -> None:
    # w = 11: c = 0 gives .89; c = .9 gives 1.00 at s = 33/300 = 11/100; hard hand-off 2.1875 s.
    r = choice_zone(11, hard_s=2.1875).run("handoff")
    assert (r.rung, r.qualityTarget) == (1, "T=0.94")
    assert r.policy == {"c": 0.9, "s": F(11, 100)}
    assert r.barValues["s350"] == F(4, 25)


def test_choice_rung_2_accepts_s_exactly_at_the_ceiling() -> None:
    r = choice_zone(14, hard_s=2.5).run("handoff")  # s = 42/300 = 7/50 = 350/2500
    assert (r.rung, r.qualityTarget) == (2, "T=0.94")
    assert r.policy == {"c": 0.9, "s": F(7, 50)}


def test_choice_rung_3_ignores_latency() -> None:
    r = choice_zone(14, hard_s=4.0).run("handoff")
    assert (r.rung, r.qualityTarget, r.latencyUnmet) == (3, "T=0.94", True)
    assert r.policy == {"c": 0.9, "s": F(7, 50)}


def test_choice_rung_4_ties_go_to_the_smaller_s() -> None:
    # w = 15 with the LLM wrong on the hard items: every c gives .85 per class; c = 0 has s = 0.
    r = choice_zone(15, llm_wrong=True).run("handoff")
    assert (r.rung, r.qualityTarget) == (4, "max-min-agreement")
    assert r.policy == {"c": 0, "s": F(0)}
    assert r.key == (F(-17, 20), F(0))
    assert (r.agreementUnmet, r.latencyUnmet) == (True, False)


def test_choice_decision_only_family_is_the_single_point_c_zero() -> None:
    r = choice_zone(7).run("decision-only")
    assert (r.family, r.rung, r.qualityTarget) == ("decision-only", 1, "T=0.93")
    assert r.policy == {"c": 0, "s": F(0)}
    assert r.barValues["agreement"] == {"a": F(93, 100), "b": F(93, 100), "c": F(93, 100)}


def test_choice_decision_only_rung_4_when_the_single_point_misses_090() -> None:
    r = choice_zone(15, llm_wrong=True).run("decision-only")
    assert (r.rung, r.qualityTarget, r.agreementUnmet) == (4, "max-min-agreement", True)
    assert r.policy == {"c": 0, "s": F(0)}


def test_choice_over_limit_is_fail_closed_on_decision_only_and_handed_off_otherwise() -> None:
    fx = Fx("choice")
    fx.group(9, "a", ("a", 0.9))
    fx.add("a", "over")
    fx.group(10, "b", ("b", 0.9))
    r = fx.run("decision-only")  # "no label" is never right: class a 9/10
    assert (r.rung, r.qualityTarget) == (1, "T=0.90")
    assert r.barValues["agreement"] == {"a": F(9, 10), "b": F(1)}
    r = fx.run("handoff")  # handed off to a right LLM at every c: 1.00 at s = 1/20
    assert (r.rung, r.qualityTarget) == (1, "T=0.94")
    assert r.policy == {"c": 0, "s": F(1, 20)}


def test_choice_failed_hand_off_is_always_wrong_and_its_time_enters_t_lh() -> None:
    fx = Fx("choice")
    fx.group(9, "a", ("a", 0.9))
    fx.add("a", "over", hand="failed", hand_s=3.0)
    fx.group(10, "b", ("b", 0.9))
    r = fx.run("handoff")
    assert (r.rung, r.qualityTarget) == (1, "T=0.90")
    assert r.policy == {"c": 0, "s": F(1, 20)}
    assert r.barValues["tLH_ms"] == F(3000)
    assert r.barValues["agreement"] == {"a": F(9, 10), "b": F(1)}


def test_choice_failed_decision_is_always_wrong_and_its_time_sits_in_t_d() -> None:
    fx = Fx("choice")
    fx.group(9, "a", ("a", 0.9))
    fx.add("a", "failed", dec_s=2.0)
    fx.group(10, "b", ("b", 0.9))
    for family in ("handoff", "decision-only"):
        r = fx.run(family)
        assert (r.rung, r.qualityTarget) == (1, "T=0.90")
        assert r.barValues["agreement"] == {"a": F(9, 10), "b": F(1)}
        assert r.barValues["td_ms"] == F(100)


# --- input contracts ---


def test_a_failed_hand_off_is_fail_closed_even_if_its_answer_field_equals_the_label() -> None:
    # predicate: a True item whose failed hand-off still carries answer True stays wrong.
    fx = Fx("predicate")
    fx.group(9, True, 0.9)
    fx.add(True, "over", hand="failed", hand_s=3.0, hand_answer=True)
    fx.group(10, False, 0.1)
    assert fx.run("handoff").barValues["agreement"] == {True: F(9, 10), False: F(1)}
    # choice: the same with the label string
    fx = Fx("choice")
    fx.group(9, "a", ("a", 0.9))
    fx.add("a", "over", hand="failed", hand_s=3.0, hand_answer="a")
    fx.group(10, "b", ("b", 0.9))
    assert fx.run("handoff").barValues["agreement"] == {"a": F(9, 10), "b": F(1)}


def test_a_hand_off_answer_that_only_equals_the_label_is_not_a_right_predicate_answer() -> None:
    # 1 == True but is not it: an `ok` hand-off answering the int 1 for a True label is wrong.
    fx = Fx("predicate")
    fx.group(9, True, 0.9)
    fx.add(True, "over", hand_answer=1)
    fx.group(10, False, 0.1)
    assert fx.run("handoff").barValues["agreement"] == {True: F(9, 10), False: F(1)}


def test_a_choice_label_that_is_not_a_string_is_refused() -> None:
    fx = Fx("choice")
    fx.add("a", ("a", 0.9))
    fx.labels["i0"] = 3
    with pytest.raises(ValueError, match="label"):
        fx.run("handoff")


def test_a_list_shaped_choice_answer_is_refused_not_coerced() -> None:
    # The ladders take the pair as a tuple (the collector must build one); a list raises.
    fx = Fx("choice")
    fx.add("a", ("a", 0.9))
    fx.add("a", ["a", 0.9])
    with pytest.raises(ValueError, match="choice answer"):
        fx.run("handoff")


def test_quality_target_strings_are_the_bench_closed_set() -> None:
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "model-bench/modelbench/decisions.py"
    if not src.is_file():
        pytest.skip("model-bench/ is absent")
    tree = ast.parse(src.read_text())
    closed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == (
            "QUALITY_TARGETS_BY_BAR_KIND"
        ):
            closed = {
                c.value
                for c in ast.walk(node.value)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            }
    assert closed, "QUALITY_TARGETS_BY_BAR_KIND not found as an annotated module assignment"
    emitted = {zone(10, 10, e=e).run("handoff").qualityTarget for e in (6, 7, 8, 9)}
    emitted |= {zone(10, 10, e=10).run("handoff").qualityTarget}
    emitted |= {interleaved().run("handoff").qualityTarget}
    assert emitted == {"T=0.94", "T=0.93", "T=0.92", "T=0.91", "T=0.90", "max-min-agreement"}
    assert emitted <= closed
