"""`decisionmodels.policy` (plan4 §4.4 U6f.2; plan3 §3.2; v1.3 §3.3): `apply_policy` against a
byte-copy of model-bench's `apply-policy.jsonl`, the bounds refusal and the one-way hand-off rule.
No fit exists yet; these rows are what a fit's policy search may rely on."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from decisionmodels.lmstudio import DecisionAnswer
from decisionmodels.policy import PolicyDecision, apply_policy, can_hand_off, check_handoff_rule

_COPY = Path(__file__).resolve().parent / "fixtures" / "decision-contract" / "apply-policy.jsonl"
_ROWS = [json.loads(line) for line in _COPY.read_text(encoding="utf-8").splitlines() if line]


def _answer(raw: dict[str, Any]) -> DecisionAnswer:
    if raw["type"] == "predicate":
        return DecisionAnswer("q", "predicate", raw["probability"], None, None, None)
    return DecisionAnswer("q", "choice", None, raw["choice"], None, raw["confidence"])


def _band(low: Any, high: Any) -> dict[str, Any]:
    return {"kind": "predicate-band", "yesAtOrAbove": high, "noBelow": low}


def _conf(c: Any) -> dict[str, Any]:
    return {"kind": "choice-confidence", "minConfidence": c}


_PRED = _answer({"type": "predicate", "probability": 0.5})
_CHOICE = _answer({"type": "choice", "choice": "a", "confidence": 0.5})


def test_the_copy_has_every_golden_row() -> None:
    assert len(_ROWS) == 22


@pytest.mark.parametrize("row", _ROWS, ids=[r["case"] for r in _ROWS])
def test_every_golden_row_passes(row: dict[str, Any]) -> None:
    got = apply_policy(row["policy"], _answer(row["answer"]))
    assert got == PolicyDecision(row["expected"]["handoff"], row["expected"]["answer"])
    # `answer` is the model's own value, never coerced: True/False stay bool.
    assert type(got.answer) is type(row["expected"]["answer"])


@pytest.mark.parametrize(
    "policy",
    [
        _band(0.6, 0.4),  # l > h
        _band(-0.1, 0.5),
        _band(0.1, 1.1),
        _band(0.1, math.nan),
        _band(math.inf, 1.0),
        _band(None, 0.5),
        _band(True, 1.0),
        _band("0.1", 0.5),
        {"kind": "predicate-band", "yesAtOrAbove": 0.5},
        _conf(-0.01),
        _conf(1.01),
        _conf(math.nan),
        _conf(None),
        _conf(True),
        {"kind": "choice-confidence"},
    ],
)
def test_out_of_bounds_policy_refuses(policy: dict[str, Any]) -> None:
    answer = _PRED if policy["kind"] == "predicate-band" else _CHOICE
    with pytest.raises(ValueError, match="apply_policy: policy"):
        apply_policy(policy, answer)


def test_swapped_band_refuses_even_where_the_branch_order_would_hide_it() -> None:
    # l=0.6 > h=0.4, p=0.9: yes-first says yes; no-first says hand-off. The refusal is the point.
    with pytest.raises(ValueError, match="noBelow <= yesAtOrAbove"):
        apply_policy(_band(0.6, 0.4), _answer({"type": "predicate", "probability": 0.9}))


@pytest.mark.parametrize(
    ("policy", "answer"),
    [
        (_band(0.1, 0.9), _CHOICE),
        (_conf(0.5), _PRED),
        ({"kind": "other"}, _PRED),
        ({}, _PRED),
    ],
)
def test_a_policy_cannot_decide_the_other_answer_type(
    policy: dict[str, Any], answer: DecisionAnswer
) -> None:
    with pytest.raises(ValueError, match="cannot decide"):
        apply_policy(policy, answer)


@pytest.mark.parametrize(
    "answer",
    [
        DecisionAnswer("q", "predicate", None, None, None, None),
        DecisionAnswer("q", "choice", None, None, None, 0.9),
        DecisionAnswer("q", "choice", None, "a", None, None),
    ],
)
def test_an_answer_missing_its_value_refuses(answer: DecisionAnswer) -> None:
    policy = _band(0.1, 0.9) if answer.type == "predicate" else _conf(0.5)
    with pytest.raises(ValueError, match="no "):
        apply_policy(policy, answer)


def test_policy_decision_is_answer_none_iff_handoff() -> None:
    with pytest.raises(ValueError):
        PolicyDecision(handoff=True, answer="x")
    with pytest.raises(ValueError):
        PolicyDecision(handoff=False, answer=None)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (_band(0.2, 0.8), True),
        (_band(0.0, 1.0), True),
        (_band(0.5, 0.5), False),  # empty band
        (_band(0.0, 0.0), False),
        (_conf(0.7), True),
        (_conf(1.0), True),
        (_conf(0.0), False),
    ],
)
def test_can_hand_off(policy: dict[str, Any], expected: bool) -> None:
    assert can_hand_off(policy) is expected


@pytest.mark.parametrize("policy", [_band(0.2, 0.8), _conf(0.7)])
def test_a_handing_off_policy_with_no_handoff_model_refuses(policy: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="handoffModel"):
        check_handoff_rule(policy, None)
    check_handoff_rule(policy, "some/model")


@pytest.mark.parametrize("policy", [_band(0.5, 0.5), _conf(0.0)])
def test_an_empty_policy_loads_with_or_without_a_handoff_model(policy: dict[str, Any]) -> None:
    check_handoff_rule(policy, "some/model")  # the overflow-only composite (plan3 §3.2)
    check_handoff_rule(policy, None)


def test_the_handoff_rule_checks_bounds_first() -> None:
    with pytest.raises(ValueError, match="check_handoff_rule: policy"):
        check_handoff_rule(_band(0.9, 0.1), "m")
    with pytest.raises(ValueError, match="check_handoff_rule: policy"):
        check_handoff_rule(_conf(2.0), "m")
