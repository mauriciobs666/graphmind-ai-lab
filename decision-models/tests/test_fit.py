"""`decisionmodels.fit`: text scores, selection and split-half (plan4 §3.11, §5 row U6f.8b; fit note
§3). The pure half: `LadderResult`s and `ScoredCall`s are hand-built, nothing calls a client.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

import pytest

from decisionmodels import fit
from decisionmodels.fit import (
    analyze_fit,
    best_text,
    eligibility,
    policy_values,
    run_ladder,
    split_half_shifts,
)
from decisionmodels.fitcalls import CollectedCalls
from decisionmodels.fitstats import FAMILIES, LadderResult, ScoredCall
from decisionmodels.questions import Question

F = Fraction


def q(text_id: str, control: bool = False) -> Question:
    return Question(text_id, control, {"type": "predicate", "name": "q", "instructions": text_id})


def lr(
    rung: int | None,
    *,
    family: str = "handoff",
    target: str = "D_c<=0",
    key: tuple[Any, ...] = (F(0),),
    agreement_unmet: bool = False,
    latency_unmet: bool = False,
) -> LadderResult:
    if rung is None and family == "handoff":
        target = "mcnemar-nonreject"
    return LadderResult(
        family=family,
        rung=rung,
        qualityTarget=target,
        policy={},
        key=key,
        agreementUnmet=agreement_unmet,
        latencyUnmet=latency_unmet,
        barValues={},
    )


def pick(texts: Sequence[Question], ladders: dict[str, LadderResult], aucs: dict[str, float]):
    eligible, none_eligible, left = eligibility(texts, aucs, ())
    return best_text(texts, eligible, ladders).id, none_eligible, left


# --- eligibility: the AUC rule only (fit note §3 fixtures a-f) ---


def test_a_lower_rung_text_above_half_beats_a_higher_rung_text_at_or_below_half() -> None:
    texts = [q("A"), q("B")]
    chosen, none_eligible, left = pick(texts, {"A": lr(1), "B": lr(3)}, {"A": 0.48, "B": 0.70})
    assert (chosen, none_eligible, left) == ("B", False, {"A": "auc"})


def test_when_no_text_is_above_half_the_best_rung_is_picked_and_none_is_eligible() -> None:
    texts = [q("A"), q("B")]
    chosen, none_eligible, left = pick(texts, {"A": lr(3), "B": lr(1)}, {"A": 0.5, "B": 0.4})
    assert (chosen, none_eligible, left) == ("B", True, {})


def test_a_text_at_exactly_half_leaves_while_another_is_above_half() -> None:
    texts = [q("A", True), q("B")]
    chosen, _, left = pick(texts, {"A": lr(1), "B": lr(4)}, {"A": 0.5, "B": 0.5000001})
    assert chosen == "B"
    assert left == {"A": "auc"}


def test_eligibility_is_strictly_above_half_and_the_boundary_is_not_a_margin() -> None:
    eligible, none_eligible, _ = eligibility([q("A")], {"A": 0.5}, ())
    assert none_eligible and [t.id for t in eligible] == ["A"]


def test_the_control_at_half_loses_to_a_longer_text_at_a_lower_rung() -> None:
    texts = [q("ctl", True), q("long")]
    chosen, _, left = pick(texts, {"ctl": lr(1), "long": lr(3)}, {"ctl": 0.5, "long": 0.6})
    assert chosen == "long"
    assert left == {"ctl": "auc"}


def test_a_decision_failure_text_does_not_count_so_no_eligible_text_is_true_for_a_lone_one() -> (
    None
):
    texts = [q("ctl", True), q("B")]
    eligible, none_eligible, left = eligibility(texts, {"ctl": 0.4, "B": 0.9}, {"B"})
    assert none_eligible is True
    assert [t.id for t in eligible] == ["ctl"]
    assert left == {"B": "decision-failure"}


def test_the_decision_failure_text_is_not_in_selection_even_when_it_is_the_best() -> None:
    texts = [q("ctl", True), q("B"), q("C")]
    eligible, none_eligible, left = eligibility(texts, {"ctl": 0.4, "B": 0.9, "C": 0.8}, {"B"})
    assert [t.id for t in eligible] == ["C"] and not none_eligible
    assert left == {"B": "decision-failure", "ctl": "auc"}


def test_a_text_that_never_left_has_no_leftselection() -> None:
    _, _, left = eligibility([q("A"), q("B")], {"A": 0.7, "B": 0.8}, ())
    assert left == {}


# --- ranking, one fixture per key (fit note §3) ---


def test_the_lower_rung_wins_before_any_key() -> None:
    texts = [q("A"), q("B")]
    ladders = {"A": lr(2, key=(F(9),)), "B": lr(3, key=(F(0),))}
    assert pick(texts, ladders, {"A": 0.7, "B": 0.7})[0] == "A"


def test_a_rung_none_text_ranks_below_rung_four_whatever_its_key() -> None:
    texts = [q("none", True), q("four")]
    ladders = {
        "none": lr(None, key=(F(0), F(0), F(0))),
        "four": lr(4, key=(F(5), F(5), F(5))),
    }
    assert pick(texts, ladders, {"none": 0.7, "four": 0.7})[0] == "four"


def test_rung_none_ranks_below_rung_four_by_rung_alone_not_by_a_later_key() -> None:
    # same family, same target, key worse for rung 4: only the rung can separate them
    texts = [q("none", True), q("four")]
    ladders = {
        "none": lr(None, family="decision-only", key=(F(0), F(0))),
        "four": lr(4, family="decision-only", key=(F(9), F(9))),
    }
    assert pick(texts, ladders, {"none": 0.7, "four": 0.7})[0] == "four"


def test_rung_four_with_agreement_unmet_still_beats_a_rung_none_text_that_met_it() -> None:
    # a new-ability rung 4 reports agreementUnmet; the rung is compared before that flag
    texts = [q("none", True), q("four")]
    ladders = {
        "none": lr(None, family="decision-only", key=(F(0),)),
        "four": lr(4, family="decision-only", agreement_unmet=True, key=(F(9),)),
    }
    assert pick(texts, ladders, {"none": 0.7, "four": 0.7})[0] == "four"


def test_among_none_texts_a_latency_only_miss_beats_an_agreement_miss_with_a_better_key() -> None:
    texts = [q("agree", True), q("lat")]
    ladders = {
        "agree": lr(None, agreement_unmet=True, key=(F(1), F(1), F(0))),
        "lat": lr(None, latency_unmet=True, key=(F(4), F(9), F(0))),
    }
    assert pick(texts, ladders, {"agree": 0.7, "lat": 0.7})[0] == "lat"


def test_among_none_texts_the_stricter_quality_target_ranks_first() -> None:
    texts = [q("nonreject", True), q("nowworse")]
    ladders = {
        "nonreject": lr(None, family="decision-only", target="mcnemar-nonreject", key=(F(0),)),
        "nowworse": lr(None, family="decision-only", target="D_c<=0", key=(F(3),)),
    }
    assert pick(texts, ladders, {"nonreject": 0.7, "nowworse": 0.7})[0] == "nowworse"


def test_two_agreement_none_texts_rank_by_the_key_exactly() -> None:
    texts = [q("A", True), q("B"), q("C")]
    ladders = {
        "A": lr(None, agreement_unmet=True, key=(F(2), F(5), F(0))),
        "B": lr(None, agreement_unmet=True, key=(F(2), F(3), F(1, 5))),
        "C": lr(None, agreement_unmet=True, key=(F(3), F(1), F(0))),
    }
    assert pick(texts, ladders, dict.fromkeys("ABC", 0.7))[0] == "B"


def test_all_none_with_auc_above_half_still_selects_one_and_is_eligible() -> None:
    texts = [q("A", True), q("B")]
    ladders = {"A": lr(None, key=(F(2),)), "B": lr(None, key=(F(1),))}
    chosen, none_eligible, _ = pick(texts, ladders, {"A": 0.6, "B": 0.6})
    assert (chosen, none_eligible) == ("B", False)


def test_a_key_better_by_one_count_beats_the_control() -> None:
    texts = [q("ctl", True), q("long")]
    ladders = {"ctl": lr(2, key=(F(2), F(4), F(1, 10))), "long": lr(2, key=(F(1), F(9), F(9)))}
    assert pick(texts, ladders, {"ctl": 0.7, "long": 0.7})[0] == "long"


def test_a_float_form_tie_is_an_exact_tie_and_the_control_wins_over_file_order() -> None:
    texts = [q("long"), q("ctl", True)]
    ladders = {
        "long": lr(2, key=(F(1), F(2), F(3, 10))),
        "ctl": lr(2, key=(F(1), F(2), F(0.3).limit_denominator(10))),
    }
    assert pick(texts, ladders, {"long": 0.7, "ctl": 0.7})[0] == "ctl"


def test_a_near_tie_that_a_float_would_round_is_not_a_tie() -> None:
    texts = [q("ctl", True), q("long")]
    ladders = {
        "ctl": lr(2, key=(F(1), F(2), F(3, 10))),
        "long": lr(2, key=(F(1), F(2), F(3, 10) - F(1, 10**18))),
    }
    assert pick(texts, ladders, {"ctl": 0.7, "long": 0.7})[0] == "long"


def test_tied_non_control_texts_go_to_file_order() -> None:
    texts = [q("ctl", True), q("late"), q("early")]
    ladders = {t: lr(1, key=(F(1),)) for t in ("late", "early")}
    ladders["ctl"] = lr(3)
    assert pick(texts, ladders, dict.fromkeys(("ctl", "late", "early"), 0.7))[0] == "late"


def test_the_nonreject_target_ranks_after_no_worse_at_one_rung_before_the_key() -> None:
    texts = [q("ctl", True), q("B")]
    ladders = {
        "ctl": lr(3, target="mcnemar-nonreject", key=(F(0), F(0), F(0))),
        "B": lr(3, target="D_c<=0", key=(F(1, 3), F(0), F(0))),
    }
    assert pick(texts, ladders, {"ctl": 0.7, "B": 0.7})[0] == "B"


def test_new_ability_targets_rank_by_the_key_not_by_the_target_name() -> None:
    texts = [q("ctl", True), q("B")]
    ladders = {
        "ctl": lr(1, family="decision-only", target="T=0.90", key=(F(-9, 10), F(0))),
        "B": lr(1, family="decision-only", target="T=0.94", key=(F(-94, 100), F(0))),
    }
    assert pick(texts, ladders, {"ctl": 0.7, "B": 0.7})[0] == "B"


# --- scenario helpers for analyze_fit ---


def hand(i: str, answer: Any = True, secs: float = 1.0, outcome: str = "ok") -> ScoredCall:
    return ScoredCall(i, "handoff", None, outcome, answer, secs)


def dec(i: str, t: str, answer: Any, secs: float = 0.1, outcome: str = "ok") -> ScoredCall:
    return ScoredCall(i, "decision", t, outcome, answer, secs)


GUARD_P = {  # item -> (label, p): hand-computed in the split-half test below
    "i1": (True, 0.9),
    "i2": (True, 0.8),
    "i3": (False, 0.2),
    "i4": (False, 0.3),
    "i5": (True, 0.7),
    "i6": (True, 0.55),
    "i7": (False, 0.1),
    "i8": (False, 0.6),
}
LABELS = {i: lab for i, (lab, _) in GUARD_P.items()}
CELLS = {i: ("falseSuspend" if lab else "falseAdvance") for i, lab in LABELS.items()}


def guard_calls(texts: dict[str, dict[str, float]]) -> list[ScoredCall]:
    """The LLM is right on every item (1 s each); a decision call takes 0.1 s."""
    calls = [hand(i, lab) for i, lab in LABELS.items()]
    for t, ps in texts.items():
        calls += [dec(i, t, p) for i, p in ps.items()]
    return calls


def collected(calls: Sequence[ScoredCall], max_probabilities: dict | None = None) -> CollectedCalls:
    return CollectedCalls(tuple(calls), 0, max_probabilities or {})


def analyze(role: str, calls, texts, labels=LABELS, cells=CELLS, max_probabilities=None):
    return analyze_fit(
        role,
        collected(calls, max_probabilities),
        labels=labels,
        cells=cells if role == "guard-judge" else None,
        texts=texts,
    )


# --- split-half: both directions, all ids, hand-computed (note §Q2.4) ---


def test_split_half_shifts_both_directions_on_the_hand_computed_fixture() -> None:
    p = {i: v[1] for i, v in GUARD_P.items()}
    forward, backward = split_half_shifts(
        "guard-judge",
        "decision-only",
        guard_calls({"t": p}),
        labels=LABELS,
        cells=CELLS,
        text_id="t",
    )
    # half A = i1 i3 i5 i7 (perfectly separated by t = 0.7), half B = i2 i4 i6 i8
    assert forward.direction == "first->second"
    assert forward.fitIds == ("i1", "i3", "i5", "i7")
    assert forward.scoredIds == ("i2", "i4", "i6", "i8")
    assert forward.fitted.policy == {"t": 0.7, "s": F(0)}
    assert forward.onFit == {
        "s": F(0),
        "n": {"falseAdvance": 2, "falseSuspend": 2},
        "D_c": {"falseAdvance": 0, "falseSuspend": 0},
    }
    # on B, t = 0.7 sends i6 (label yes, p .55) to no: one wrong suspend, the LLM being right
    assert forward.onScored == {
        "s": F(0),
        "n": {"falseAdvance": 2, "falseSuspend": 2},
        "D_c": {"falseAdvance": 0, "falseSuspend": 1},
    }
    assert forward.shift == {"s": F(0), "D_c": {"falseAdvance": 0, "falseSuspend": 1}}
    # half B has no D_c <= 0 setting (i6 .55 yes, i8 .6 no); rung 2's first minimiser is t = 0.55
    assert backward.direction == "second->first"
    assert backward.fitIds == ("i2", "i4", "i6", "i8")
    assert backward.fitted.rung == 2
    assert backward.fitted.policy == {"t": 0.55, "s": F(0)}
    assert backward.onFit == {
        "s": F(0),
        "n": {"falseAdvance": 2, "falseSuspend": 2},
        "D_c": {"falseAdvance": 1, "falseSuspend": 0},
    }
    assert backward.onScored == {
        "s": F(0),
        "n": {"falseAdvance": 2, "falseSuspend": 2},
        "D_c": {"falseAdvance": 0, "falseSuspend": 0},
    }
    assert backward.shift == {"s": F(0), "D_c": {"falseAdvance": -1, "falseSuspend": 0}}


def test_split_half_spans_every_calibration_id_and_the_halves_are_disjoint() -> None:
    p = {i: v[1] for i, v in GUARD_P.items()}
    forward, backward = split_half_shifts(
        "guard-judge", "handoff", guard_calls({"t": p}), labels=LABELS, cells=CELLS, text_id="t"
    )
    assert set(forward.fitIds) | set(forward.scoredIds) == set(LABELS)
    assert not set(forward.fitIds) & set(forward.scoredIds)
    assert (backward.fitIds, backward.scoredIds) == (forward.scoredIds, forward.fitIds)


def test_the_shift_in_s_is_the_difference_of_the_fitted_policys_hand_off_share() -> None:
    # a hand-off family policy whose band catches one item in one half and two in the other
    p = {"i1": 0.9, "i2": 0.5, "i3": 0.5, "i4": 0.3, "i5": 0.7, "i6": 0.55, "i7": 0.1, "i8": 0.6}
    forward, _ = split_half_shifts(
        "guard-judge", "handoff", guard_calls({"t": p}), labels=LABELS, cells=CELLS, text_id="t"
    )
    s_fit, s_scored = forward.onFit["s"], forward.onScored["s"]
    assert forward.shift["s"] == s_scored - s_fit


# --- the policy scorer is the ladder's scoring (property over seeded fixtures) ---


def random_scenario(role: str, rng: random.Random):
    ids = [f"i{n:02d}" for n in range(1, 25)]
    choices = ["order", "question", "other"]
    if role == "message-type-classifier":
        labels = {i: rng.choice(choices) for i in ids}
    else:
        labels = {i: rng.random() < 0.5 for i in ids}
    cells = {i: f"{labels[i]}|{rng.choice('ab')}" for i in ids} if role == "guard-judge" else None
    calls = []
    for i in ids:
        ok = rng.random() < 0.8
        if role == "message-type-classifier":
            answer = labels[i] if ok else rng.choice(choices)
        else:
            answer = labels[i] if ok else not labels[i]
        outcome = rng.choices(["ok", "failed"], [9, 1])[0]
        calls.append(
            hand(i, answer, rng.choice([0.5, 1.0, 2.0]), outcome if outcome == "failed" else "ok")
        )
        kind = rng.choices(["ok", "over-limit", "failed"], [8, 1, 1])[0]
        if kind != "ok":
            calls.append(dec(i, "t", None, 0.1, kind))
        elif role == "message-type-classifier":
            calls.append(dec(i, "t", (rng.choice(choices), rng.choice([0.2, 0.5, 0.9, 1.0]))))
        else:
            calls.append(dec(i, "t", rng.choice([0.0, 0.1, 0.4, 0.4, 0.8, 1.0])))
    calls = [
        ScoredCall(c.itemId, c.callKind, c.textId, c.outcome, None, c.elapsedSeconds)
        if c.callKind == "handoff" and c.outcome == "failed"
        else c
        for c in calls
    ]
    return labels, cells, calls


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("role", ["guard-judge", "join-in-judge", "message-type-classifier"])
def test_policy_values_on_the_fit_set_equal_the_ladders_own_bar_values(
    role: str, family: str
) -> None:
    for seed in range(30):
        labels, cells, calls = random_scenario(role, random.Random(seed))
        ladder = run_ladder(role, calls, labels=labels, cells=cells, text_id="t", family=family)
        values = policy_values(
            role, family, ladder.policy, calls, labels=labels, cells=cells, text_id="t"
        )
        assert values["s"] == ladder.barValues["s"], (role, family, seed)
        if role == "guard-judge":
            bar = ladder.barValues["D_c"]
            assert {k: v for k, v in values["D_c"].items() if k in bar} == bar, (family, seed)
            assert all(v == 0 for k, v in values["D_c"].items() if k not in bar)
        else:
            assert values["agreement"] == ladder.barValues["agreement"], (role, family, seed)


# --- text figures ---


def test_auc_is_the_roc_auc_of_p_against_the_label_with_hand_computed_literals() -> None:
    perfect = {i: (0.9 if lab else 0.1) for i, lab in LABELS.items()}
    reversed_ = {i: (0.1 if lab else 0.9) for i, lab in LABELS.items()}
    mixed = {"i1": 0.9, "i2": 0.5, "i3": 0.5, "i4": 0.1, "i5": 0.3, "i6": 0.7, "i7": 0.8, "i8": 0.2}
    result = analyze(
        "guard-judge",
        guard_calls({"perfect": perfect, "rev": reversed_, "mix": mixed}),
        [q("perfect", True), q("rev"), q("mix")],
    )
    auc = {s.textId: s.auc for s in result.scores if s.candidate == "handoff"}
    # mixed: positives i1 .9, i2 .5, i5 .3, i6 .7 against negatives i3 .5, i4 .1, i7 .8, i8 .2:
    # by hand: .9 beats 4, .5 beats .1 .2 and ties .5 (2.5), .3 beats .1 .2 (2),
    # .7 beats .5 .1 .2 (3) -> (4 + 2.5 + 2 + 3) / 16
    assert auc == {"perfect": 1.0, "rev": 0.0, "mix": 11.5 / 16}


def test_auc_is_half_when_a_class_is_absent_or_the_score_is_constant() -> None:
    constant = dict.fromkeys(LABELS, 0.5)
    result = analyze("guard-judge", guard_calls({"c": constant}), [q("c", True)])
    assert {s.auc for s in result.scores} == {0.5}
    assert result.selected[0].selection.noEligibleText is True


def test_over_limit_items_are_out_of_the_auc_and_counted_as_k_of_n() -> None:
    p = {i: (0.9 if lab else 0.1) for i, lab in LABELS.items()}
    calls = [
        c
        for c in guard_calls({"t": p})
        if not (c.callKind == "decision" and c.itemId in ("i3", "i6"))
    ]
    calls += [dec("i3", "t", None, 0.2, "over-limit"), dec("i6", "t", None, 0.2, "over-limit")]
    calls = [c for c in calls if not (c.callKind == "decision" and c.itemId == "i8")]
    calls.append(dec("i8", "t", None, 5.0, "failed"))  # a failed call is in n and not in k
    score = analyze("guard-judge", calls, [q("t", True)]).scores[0]
    assert score.overLimit == (2, 8)
    assert score.auc == 1.0


def test_the_leftselection_decision_failure_rule_is_for_the_guard_and_spares_the_control() -> None:
    good = {i: (0.9 if lab else 0.1) for i, lab in LABELS.items()}
    calls = guard_calls({"ctl": good, "b": good, "c": good})
    calls = [
        c
        for c in calls
        if not (c.callKind == "decision" and c.itemId == "i2" and c.textId in ("ctl", "b"))
    ]
    calls += [dec("i2", "ctl", None, 5.0, "failed"), dec("i2", "b", None, 5.0, "failed")]
    result = analyze("guard-judge", calls, [q("ctl", True), q("b"), q("c")])
    left = {s.textId: s.leftSelection for s in result.scores if s.candidate == "handoff"}
    assert left == {"ctl": None, "b": "decision-failure", "c": None}
    assert {s.selection.textId for s in result.selected} <= {"ctl", "c"}


def test_an_item_every_text_failed_drops_no_text() -> None:
    good = {i: (0.9 if lab else 0.1) for i, lab in LABELS.items()}
    calls = [
        c
        for c in guard_calls({"ctl": good, "b": good})
        if not (c.callKind == "decision" and c.itemId == "i2")
    ]
    calls += [dec("i2", "ctl", None, 5.0, "failed"), dec("i2", "b", None, 5.0, "failed")]
    result = analyze("guard-judge", calls, [q("ctl", True), q("b")])
    assert {s.leftSelection for s in result.scores} == {None}


def test_a_failed_decision_call_on_a_new_ability_point_is_fail_closed_not_a_leaver() -> None:
    labels = {f"i{n}": n % 2 == 0 for n in range(1, 9)}
    calls = [hand(i, lab) for i, lab in labels.items()]
    for t in ("ctl", "b"):
        calls += [dec(i, t, 0.9 if lab else 0.1) for i, lab in labels.items() if i != "i1"]
    calls += [dec("i1", "b", None, 5.0, "failed"), dec("i1", "ctl", 0.1)]
    result = analyze("join-in-judge", calls, [q("ctl", True), q("b")], labels=labels, cells=None)
    assert {s.leftSelection for s in result.scores} == {None}


def test_a_decision_failure_text_above_half_alone_gives_no_eligible_text_for_both_candidates() -> (
    None
):
    bad = {i: (0.1 if lab else 0.9) for i, lab in LABELS.items()}
    good = {i: (0.9 if lab else 0.1) for i, lab in LABELS.items()}
    calls = guard_calls({"ctl": bad, "b": good})
    calls = [
        c for c in calls if not (c.callKind == "decision" and c.itemId == "i2" and c.textId == "b")
    ]
    calls.append(dec("i2", "b", None, 5.0, "failed"))
    result = analyze("guard-judge", calls, [q("ctl", True), q("b")])
    assert [(s.selection.textId, s.selection.noEligibleText) for s in result.selected] == [
        ("ctl", True),
        ("ctl", True),
    ]


# --- choice: max(probabilities) against confidence ---

CHOICES = {"c1": "order", "c2": "order", "c3": "other", "c4": "other"}


def choice_fixture(extra: bool = False):
    labels = dict(CHOICES)
    # (choice, confidence): c1, c2 right; c3, c4 wrong. confidence AUC: 3 of 4 right pairs = 0.75
    answers = {
        "c1": ("order", 0.9),
        "c2": ("order", 0.6),
        "c3": ("order", 0.8),
        "c4": ("order", 0.3),
    }
    calls = [hand(i, labels[i]) for i in labels]
    calls += [dec(i, "t", a) for i, a in answers.items()]
    # max(probabilities) ranks the right answers below the wrong ones: AUC 0
    maxima = {("c1", "t"): 0.5, ("c2", "t"): 0.45, ("c3", "t"): 0.9, ("c4", "t"): 0.8}
    if extra:
        labels["c5"] = "other"
        calls += [hand("c5", "other"), dec("c5", "t", ("other", 0.5))]
    return labels, calls, maxima


def test_max_prob_auc_is_the_auc_of_the_maximum_probability_not_of_the_confidence() -> None:
    labels, calls, maxima = choice_fixture()
    score = analyze(
        "message-type-classifier",
        calls,
        [q("t", True)],
        labels=labels,
        cells=None,
        max_probabilities=maxima,
    ).scores[0]
    assert score.auc == 0.75
    assert score.maxProbAuc == 0.0


def test_an_ok_choice_call_without_a_maximum_is_dropped_from_max_prob_auc_and_counted() -> None:
    labels, calls, maxima = choice_fixture(extra=True)
    result = analyze(
        "message-type-classifier",
        calls,
        [q("t", True)],
        labels=labels,
        cells=None,
        max_probabilities=maxima,
    )
    assert result.scores[0].maxProbAuc == 0.0
    assert result.maxProbDropped == {"t": 1}
    assert result.scores[0].auc != 0.75  # the confidence AUC still counts c5


def test_a_predicate_point_has_no_max_prob_auc() -> None:
    p = {i: (0.9 if lab else 0.1) for i, lab in LABELS.items()}
    result = analyze("guard-judge", guard_calls({"t": p}), [q("t", True)])
    assert result.scores[0].maxProbAuc is None
    assert result.maxProbDropped == {}


# --- two candidates, one calls tuple ---


def test_two_candidates_read_the_same_calls_and_may_select_different_texts() -> None:
    result = analyze(
        "guard-judge",
        guard_calls({"ctl": dict.fromkeys(LABELS, 0.5)}),
        [q("ctl", True)],
    )
    assert [s.selection.candidate for s in result.selected] == list(FAMILIES)
    assert [s.ladder.family for s in result.selected] == list(FAMILIES)
    assert len(result.scores) == 2


def test_each_candidate_keeps_its_own_text_when_the_families_rank_differently(monkeypatch) -> None:
    ladders = {
        ("handoff", "a"): lr(1, key=(F(1),)),
        ("handoff", "b"): lr(2, key=(F(0),)),
        ("decision-only", "a"): lr(3, family="decision-only", key=(F(0),)),
        ("decision-only", "b"): lr(1, family="decision-only", key=(F(0),)),
    }
    monkeypatch.setattr(
        fit,
        "run_ladder",
        lambda role, calls, *, labels, cells, text_id, family: ladders[(family, text_id)],
    )
    monkeypatch.setattr(fit, "split_half_shifts", lambda *a, **k: (None, None))
    p = {i: (0.9 if lab else 0.1) for i, lab in LABELS.items()}
    result = analyze("guard-judge", guard_calls({"a": p, "b": p}), [q("a", True), q("b")])
    assert [(s.selection.candidate, s.selection.textId) for s in result.selected] == [
        ("handoff", "a"),
        ("decision-only", "b"),
    ]
    assert result.selected[0].ladder is ladders[("handoff", "a")]
    assert {s.selection.noEligibleText for s in result.selected} == {False}


def test_no_eligible_text_is_carried_by_both_selections() -> None:
    rev = {i: (0.1 if lab else 0.9) for i, lab in LABELS.items()}
    result = analyze("guard-judge", guard_calls({"ctl": rev, "b": rev}), [q("ctl", True), q("b")])
    assert [s.selection.noEligibleText for s in result.selected] == [True, True]
    assert {s.leftSelection for s in result.scores} == {None}


# --- the shift: signs, the family, and a cell absent from a half (P54-M2, fit note §6) ---


def fixed_policy(monkeypatch, policy: dict[str, Any], family: str) -> None:
    result = LadderResult(
        family=family,
        rung=1,
        qualityTarget="D_c<=0",
        policy=policy,
        key=(F(0),),
        agreementUnmet=False,
        latencyUnmet=False,
        barValues={},
    )
    monkeypatch.setattr(fit, "run_ladder", lambda *a, **k: result)


P_ALL = {i: v[1] for i, v in GUARD_P.items()}


def test_the_s_shift_is_held_out_minus_fit_half_in_both_directions(monkeypatch) -> None:
    # band [0.4, 0.8): half A hands off i5 (.7): s = 1/4; half B hands off i6 .55 and i8 .6: 1/2
    fixed_policy(monkeypatch, {"l": 0.4, "h": 0.8, "s": F(0)}, "handoff")
    forward, backward = split_half_shifts(
        "guard-judge", "handoff", guard_calls({"t": P_ALL}), labels=LABELS, cells=CELLS, text_id="t"
    )
    assert (forward.onFit["s"], forward.onScored["s"]) == (F(1, 4), F(1, 2))
    # the per-cell n counts every item of the half, handed off or not
    both = {"falseAdvance": 2, "falseSuspend": 2}
    assert forward.onFit["n"] == forward.onScored["n"] == both
    assert backward.onFit["n"] == backward.onScored["n"] == both
    assert forward.shift["s"] == F(1, 4)
    assert (backward.onFit["s"], backward.onScored["s"]) == (F(1, 2), F(1, 4))
    assert backward.shift["s"] == F(-1, 4)


def test_a_decision_only_half_scores_an_over_limit_item_as_concordant_not_handed(
    monkeypatch,
) -> None:
    fixed_policy(monkeypatch, {"t": 0.5, "s": F(0)}, "decision-only")
    calls = [
        c for c in guard_calls({"t": P_ALL}) if not (c.callKind == "decision" and c.itemId == "i3")
    ]
    calls.append(dec("i3", "t", None, 0.2, "over-limit"))
    forward, backward = split_half_shifts(
        "guard-judge", "decision-only", calls, labels=LABELS, cells=CELLS, text_id="t"
    )
    assert forward.onFit["s"] == F(0)  # i3 is in half A; handed off it would be 1/4
    assert forward.shift["s"] == F(0)
    assert backward.onScored["s"] == F(0)
    assert backward.shift["s"] == F(0)
    assert forward.onFit["D_c"] == {"falseAdvance": 0, "falseSuspend": 0}


def test_the_agreement_shift_is_held_out_minus_fit_half_per_class(monkeypatch) -> None:
    # t = 0.55: half A is right on every item; half B misses i8 (label no, p .6): False 1/2
    fixed_policy(monkeypatch, {"t": 0.55, "s": F(0)}, "decision-only")
    forward, backward = split_half_shifts(
        "join-in-judge",
        "decision-only",
        guard_calls({"t": P_ALL}),
        labels=LABELS,
        cells=None,
        text_id="t",
    )
    assert forward.onFit["agreement"] == {True: F(1), False: F(1)}
    assert forward.onScored["agreement"] == {True: F(1), False: F(1, 2)}
    assert forward.shift["agreement"] == {True: F(0), False: F(-1, 2)}
    assert backward.shift["agreement"] == {True: F(0), False: F(1, 2)}


def test_a_cell_absent_from_a_half_is_reported_absent_with_each_halfs_per_cell_n(
    monkeypatch,
) -> None:
    fixed_policy(monkeypatch, {"t": 0.5, "s": F(0)}, "decision-only")
    cells = {i: "y" for i in LABELS} | {"i1": "x", "i3": "x"}  # both in half A
    forward, backward = split_half_shifts(
        "guard-judge", "decision-only", guard_calls({"t": P_ALL}), labels=LABELS, cells=cells,
        text_id="t",
    )  # fmt: skip
    assert forward.onFit["n"] == {"x": 2, "y": 2}
    assert forward.onScored["n"] == {"y": 4}
    assert "x" not in forward.onScored["D_c"]
    # y: half A has no error at t = .5; half B has i8 (no, p .6) decided yes: +1
    assert forward.shift["D_c"] == {"x": None, "y": 1}
    assert backward.onFit["n"] == {"y": 4}
    assert backward.shift["D_c"] == {"x": None, "y": -1}


def test_a_class_absent_from_a_half_has_a_none_agreement_shift(monkeypatch) -> None:
    fixed_policy(monkeypatch, {"t": 0.5, "s": F(0)}, "decision-only")
    # every item of half A (i1 i3 i5 i7) is False; half B keeps i2 and i6 True
    forward, _ = split_half_shifts(
        "join-in-judge",
        "decision-only",
        guard_calls({"t": P_ALL}),
        labels=LABELS | dict.fromkeys(("i1", "i3", "i5", "i7"), False),
        cells=None,
        text_id="t",
    )
    assert forward.onFit["n"] == {False: 4}
    assert forward.shift["agreement"][True] is None
    assert forward.shift["agreement"][False] is not None


# --- an undefined AUC (P54-M3, fit note §6) ---


def test_a_text_with_one_class_among_its_ok_calls_has_auc_half_flagged_undefined() -> None:
    pos_only = {i: 0.9 for i, lab in LABELS.items() if lab}
    calls = guard_calls({"t": pos_only})
    calls = [c for c in calls if not (c.callKind == "decision" and not LABELS[c.itemId])]
    calls += [dec(i, "t", None, 0.2, "over-limit") for i, lab in LABELS.items() if not lab]
    result = analyze("guard-judge", calls, [q("t", True)])
    score = result.scores[0]
    assert (score.auc, score.aucUndefined) == (0.5, True)
    assert result.selected[0].selection.noEligibleText is True


def test_a_text_with_no_ok_call_has_auc_half_flagged_undefined_and_does_not_crash() -> None:
    calls = [hand(i, lab) for i, lab in LABELS.items()]
    calls += [dec(i, "t", None, 0.2, "over-limit") for i in LABELS]
    score = analyze("guard-judge", calls, [q("t", True)]).scores[0]
    assert (score.auc, score.aucUndefined, score.overLimit) == (0.5, True, (8, 8))


def test_a_constant_score_has_a_defined_auc_of_half() -> None:
    constant = dict.fromkeys(LABELS, 0.5)
    score = analyze("guard-judge", guard_calls({"t": constant}), [q("t", True)]).scores[0]
    assert (score.auc, score.aucUndefined) == (0.5, False)


def test_an_undefined_auc_text_leaves_selection_while_another_is_above_half() -> None:
    good = {i: (0.9 if lab else 0.1) for i, lab in LABELS.items()}
    calls = guard_calls({"ctl": good, "u": good})
    calls = [c for c in calls if not (c.callKind == "decision" and c.textId == "u")]
    calls += [dec(i, "u", None, 0.2, "over-limit") for i in LABELS]
    result = analyze("guard-judge", calls, [q("ctl", True), q("u")])
    u = next(s for s in result.scores if s.textId == "u")
    assert (u.aucUndefined, u.leftSelection) == (True, "auc")
    assert {s.selection.textId for s in result.selected} == {"ctl"}


def test_a_choice_text_right_on_every_item_has_undefined_auc_and_no_max_prob_auc() -> None:
    labels = {"c1": "order", "c2": "other", "c3": "order"}
    calls = [hand(i, lab) for i, lab in labels.items()]
    calls += [dec(i, "t", (lab, 0.9)) for i, lab in labels.items()]
    maxima = {(i, "t"): 0.5 for i in labels}
    score = analyze(
        "message-type-classifier", calls, [q("t", True)], labels=labels, cells=None,
        max_probabilities=maxima,
    ).scores[0]  # fmt: skip
    assert (score.auc, score.aucUndefined, score.maxProbAuc) == (0.5, True, None)


def test_a_choice_text_whose_maxima_are_one_class_has_a_none_max_prob_auc_but_a_defined_auc() -> (
    None
):
    labels, calls, maxima = choice_fixture()
    only_right = {k: v for k, v in maxima.items() if k[0] in ("c1", "c2")}
    score = analyze(
        "message-type-classifier", calls, [q("t", True)], labels=labels, cells=None,
        max_probabilities=only_right,
    ).scores[0]  # fmt: skip
    assert (score.auc, score.aucUndefined, score.maxProbAuc) == (0.75, False, None)


# --- the four guard cells, typed by hand (P54-M1; model-bench scoring/classification.py) ---


@pytest.mark.parametrize(
    ("expected", "path", "cell"),
    [
        (False, "understanding", "falseAdvanceRateByUnderstanding"),
        (False, "turns", "falseAdvanceRateByTurns"),
        (True, "understanding", "falseSuspendRateByUnderstanding"),
        (True, "turns", "falseSuspendRateByTurns"),
    ],
)
def test_guard_cell_names_are_the_four_fr13_cells(expected: bool, path: str, cell: str) -> None:
    assert fit.guard_cell({"expected": expected, "path": path}) == cell


def test_split_half_scores_are_attributed_to_the_selected_text_and_candidate() -> None:
    good = {i: (0.9 if lab else 0.1) for i, lab in LABELS.items()}
    bad = {i: (0.1 if lab else 0.9) for i, lab in LABELS.items()}
    result = analyze("guard-judge", guard_calls({"ctl": bad, "b": good}), [q("ctl", True), q("b")])
    for chosen in result.selected:
        assert chosen.score.textId == chosen.selection.textId == "b"
        assert chosen.score.candidate == chosen.selection.candidate
        assert chosen.score.ladder is chosen.ladder
