"""Fit statistics and the typed records the U6f ladders share (plan4 §3.11, §4.4 U6f.5).

Pure functions over plain values, stdlib only (FR-8). The definitions are note §Q3.1 (one-sided
exact McNemar), note §Q2.3 (ROC-AUC, ties at 0.5), note §Q2.0 and fit note §1 (the `s_2`/`s_350`
ceilings and `t̄_{L|H}`) and note §Q2.4 (split-half). Latencies are in milliseconds, as the 350 ms
bar is; a `ScoredCall` carries seconds.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb
from typing import Any

CALIBRATION_FLOOR_PER_CELL = 50

FAMILIES = ("handoff", "decision-only")
_NONREJECT = "mcnemar-nonreject"
CALL_KINDS = ("handoff", "decision")
OUTCOMES = ("ok", "over-limit", "failed")


@lru_cache(maxsize=None)
def mcnemar_worse_p_exact(b: int, c: int) -> Fraction:
    """`p_worse = Σ_{i=b}^{m} C(m,i) / 2^m`, `m = b + c`, exactly; 1 when `m == 0` (note §Q3.1).

    `b` counts items where the candidate errs and the incumbent does not, `c` the reverse; swapping
    them would test the wrong direction. The ladders compare this against `1/20` without rounding.
    """
    if b < 0 or c < 0:
        raise ValueError(f"mcnemar_worse_p: counts must be non-negative, got b={b}, c={c}")
    m = b + c
    if m == 0:
        return Fraction(1)
    return Fraction(sum(comb(m, i) for i in range(b, m + 1)), 2**m)


def mcnemar_worse_p(b: int, c: int) -> float:
    """`mcnemar_worse_p_exact` as a float (the exact value, correctly rounded)."""
    return float(mcnemar_worse_p_exact(b, c))


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """P(a random positive outscores a random negative); a tie counts 0.5 (note §Q2.3)."""
    if len(scores) != len(labels):
        raise ValueError("roc_auc: scores and labels must have the same length")
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        raise ValueError("roc_auc needs both classes present")
    twice = sum(2 if p > n else 1 if p == n else 0 for p in pos for n in neg)
    return twice / (2 * len(pos) * len(neg))


Number = int | float | Fraction


def _check_handoff_latency(t_llm_handoff_ms: Number) -> None:
    if not t_llm_handoff_ms > 0:
        raise ValueError(f"t̄_(L|H) must be positive, got {t_llm_handoff_ms!r}")


def ceiling_s2(t_llm_ms: Number, t_decision_ms: Number, t_llm_handoff_ms: Number) -> Fraction:
    """FR-2's hand-off-share ceiling `(t̄_L/2 − t̄_d) / t̄_{L|H}`, exact (note §Q2.0).

    Inputs may be int, float or `Fraction`; a float converts exactly (`Fraction(x)`), so the result
    is the exact value of the inputs as given and the ladders compare it without rounding.
    """
    _check_handoff_latency(t_llm_handoff_ms)
    return (Fraction(t_llm_ms) / 2 - Fraction(t_decision_ms)) / Fraction(t_llm_handoff_ms)


def ceiling_s350(t_decision_ms: Number, t_llm_handoff_ms: Number) -> Fraction:
    """FR-10's hand-off-share ceiling `(350 − t̄_d) / t̄_{L|H}`, exact (note §Q2.0)."""
    _check_handoff_latency(t_llm_handoff_ms)
    return (350 - Fraction(t_decision_ms)) / Fraction(t_llm_handoff_ms)


@dataclass(frozen=True)
class ScoredCall:
    """One call of one item (§3.11). A `failed` call keeps its elapsed time (enters `t̄_{L|H}`)."""

    itemId: str  # noqa: N815
    callKind: str  # noqa: N815
    textId: str | None  # noqa: N815
    outcome: str
    answer: Any
    elapsedSeconds: float  # noqa: N815

    def __post_init__(self) -> None:
        if self.callKind not in CALL_KINDS:
            raise ValueError(
                f"ScoredCall.callKind must be one of {CALL_KINDS}, got {self.callKind!r}"
            )
        if self.outcome not in OUTCOMES:
            raise ValueError(f"ScoredCall.outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        if not self.elapsedSeconds >= 0:
            raise ValueError(f"ScoredCall.elapsedSeconds must be >= 0, got {self.elapsedSeconds!r}")
        if (self.textId is None) != (self.callKind == "handoff"):
            raise ValueError(
                f"ScoredCall.textId is None iff callKind is handoff "
                f"(callKind={self.callKind!r}, textId={self.textId!r})"
            )


def mean_handoff_ms(
    calls: Iterable[ScoredCall], band_item_ids: Collection[str] | None = None
) -> Fraction:
    """`t̄_{L|H}` in ms, an exact `Fraction`, over hand-off calls: the band's items or (`None`) all.

    Every outcome counts, a `failed` call's time included (§3.11). A band item with no hand-off call
    refuses rather than silently shrinking the mean.
    """
    handoffs = [c for c in calls if c.callKind == "handoff"]
    if band_item_ids is not None:
        missing = sorted(set(band_item_ids) - {c.itemId for c in handoffs})
        if missing:
            raise ValueError(f"band items without a hand-off call: {missing!r}")
        handoffs = [c for c in handoffs if c.itemId in band_item_ids]
    if not handoffs:
        raise ValueError("mean_handoff_ms: no hand-off calls selected")
    return 1000 * sum(Fraction(c.elapsedSeconds) for c in handoffs) / len(handoffs)


def split_half(item_ids: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministic, disjoint halves of calibration (note §Q2.4): sorted ids, alternating.

    Positions 0, 2, 4, … form the first half, so it takes the extra id of an odd count.
    """
    ids = list(item_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("split_half: duplicate item ids")
    if len(ids) < 2:
        raise ValueError("split_half needs at least two items")
    ordered = sorted(ids)
    return tuple(ordered[0::2]), tuple(ordered[1::2])


@dataclass(frozen=True)
class LadderResult:
    """One per (candidate, text), filled by the ladders (§3.11)."""

    family: str
    rung: int | None
    qualityTarget: Any  # noqa: N815
    policy: Any
    key: Any
    agreementUnmet: bool  # noqa: N815
    latencyUnmet: bool  # noqa: N815
    barValues: Any  # noqa: N815

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"LadderResult.family must be one of {FAMILIES}, got {self.family!r}")
        if self.rung is not None and (type(self.rung) is not int or self.rung not in (1, 2, 3, 4)):
            raise ValueError(f"LadderResult.rung must be 1-4 or None, got {self.rung!r}")
        if self.rung is None and self.family == "handoff" and self.qualityTarget != _NONREJECT:
            raise ValueError(
                f"LadderResult.rung None on the hand-off family is valid only with qualityTarget "
                f"{_NONREJECT!r}, got {self.qualityTarget!r}"
            )
