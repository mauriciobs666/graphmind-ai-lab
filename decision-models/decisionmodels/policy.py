"""The decision policies and their one-way hand-off rule (plan4 §4.4 U6f.2; plan3 §3.2; v1.3 §3.3).

A standalone copy of model-bench's `apply_policy` (FR-8: no import), pinned by a byte-copy of its
`apply-policy.jsonl` (`tests/fixtures/decision-contract/`, asserted in `tests/test_policy.py`).
Pure: no I/O, no clock.

- `predicate-band {yesAtOrAbove: h, noBelow: l}`: *yes* iff `p >= h`, *no* iff `p < l`, hand-off
  iff `l <= p < h`. `l == h` is an empty band: never a hand-off.
- `choice-confidence {minConfidence: c}`: the choice iff `confidence >= c`, else hand-off.
  `c == 0` never hands off.

Bounds are `0 <= l <= h <= 1` and `0 <= c <= 1`, refused with `ValueError` before any answer is
read: on `l > h` the yes-first and no-first branch orders disagree, so a copy could silently differ.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from decisionmodels.lmstudio import DecisionAnswer

POLICY_KINDS = ("predicate-band", "choice-confidence")


@dataclass(frozen=True)
class PolicyDecision:
    """A policy's verdict on one answer: `handoff=True` with `answer=None`, or the model's own
    answer (`True`/`False` for a predicate, the chosen value for a choice)."""

    handoff: bool
    answer: bool | str | None

    def __post_init__(self) -> None:
        if (self.answer is None) != self.handoff:
            raise ValueError(
                f"PolicyDecision: answer is None iff handoff (handoff={self.handoff!r}, "
                f"answer={self.answer!r})"
            )


def _probability(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _bounds_problem(policy: Mapping[str, Any]) -> str | None:
    if policy.get("kind") == "predicate-band":
        low, high = policy.get("noBelow"), policy.get("yesAtOrAbove")
        if not (_probability(low) and _probability(high) and low <= high):
            return f"policy band must have 0 <= noBelow <= yesAtOrAbove <= 1, got {low!r}, {high!r}"
        return None
    floor = policy.get("minConfidence")
    if not _probability(floor):
        return f"policy.minConfidence must be in [0, 1], got {floor!r}"
    return None


def can_hand_off(policy: Mapping[str, Any]) -> bool:
    """Whether the policy itself can hand an item off: a non-empty band (`l < h`), or
    `minConfidence > 0`. Over-limit items are a separate, runtime route (plan3 §3.3). Assumes
    in-bounds values; `check_handoff_rule` checks bounds first."""
    if policy.get("kind") == "predicate-band":
        return bool(policy["noBelow"] < policy["yesAtOrAbove"])
    return bool(policy["minConfidence"] > 0)


def check_handoff_rule(policy: Mapping[str, Any], handoff_model: str | None) -> None:
    """Plan3 §3.2's one-way rule. Refuses out-of-bounds values, then a policy that can hand off
    with `handoff_model` of `None`. An empty band or `c == 0` with a hand-off model loads (the
    overflow-only composite), and so does any policy that cannot hand off without one."""
    if policy.get("kind") in POLICY_KINDS:
        problem = _bounds_problem(policy)
        if problem is not None:
            raise ValueError(f"check_handoff_rule: {problem}")
        if handoff_model is None and can_hand_off(policy):
            raise ValueError(
                "check_handoff_rule: a policy that can hand off needs a handoffModel (got null)"
            )
        return
    raise ValueError(f"check_handoff_rule: unknown policy kind {policy.get('kind')!r}")


def apply_policy(policy: Mapping[str, Any], answer: DecisionAnswer) -> PolicyDecision:
    kind = policy.get("kind")
    if kind in POLICY_KINDS:
        problem = _bounds_problem(policy)
        if problem is not None:
            raise ValueError(f"apply_policy: {problem}")
    if kind == "predicate-band" and answer.type == "predicate":
        p = answer.probability
        if p is None:
            raise ValueError("apply_policy: a predicate answer with no probability")
        if p >= policy["yesAtOrAbove"]:
            return PolicyDecision(handoff=False, answer=True)
        if p < policy["noBelow"]:
            return PolicyDecision(handoff=False, answer=False)
        return PolicyDecision(handoff=True, answer=None)
    if kind == "choice-confidence" and answer.type == "choice":
        if answer.confidence is None or answer.choice is None:
            raise ValueError("apply_policy: a choice answer with no choice or confidence")
        if answer.confidence >= policy["minConfidence"]:
            return PolicyDecision(handoff=False, answer=answer.choice)
        return PolicyDecision(handoff=True, answer=None)
    raise ValueError(f"apply_policy: a {kind!r} policy cannot decide a {answer.type!r} answer")
