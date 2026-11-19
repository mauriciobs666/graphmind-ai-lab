"""Outcome assembly (plan4 §3.11, §4.4 U6f.9a; note §Q2.0; plan2 §3.2). Pure, stdlib + package only.

`assemble_outcome` turns the chosen text's `LadderResult` and its candidate's `Selection` into the
`(fit.outcome, fit.qualityTarget)` pair a candidate records. A requirement value unmet on
calibration is `fit-infeasible: <bar>`, naming the first of `ranking`, `agreement`, `latency`:
`ranking` iff `Selection.noEligibleText`, else `agreement` iff `agreementUnmet`, else `latency` iff
`latencyUnmet`. Otherwise every requirement value is met and the name is `within-margin` at the
ladder's best rung and target (rung 1 at `D_c<=0` or `T=0.94`), else `below-margin`.

The closed sets are a copy of plan2 §3.2's (the bench's `FIT_OUTCOMES` and
`QUALITY_TARGETS_BY_BAR_KIND`, which FR-8 bars importing); a test holds the copy to the bench's
source and to every target `ladder.py` can emit. `LadderResult.qualityTarget` is `Any`, and
`LadderResult` cannot know its bar kind, so the validation lives here, at the point that writes the
pair into a candidate: an independent check of the ladders' output, like the bench's own
`_fit_problem`.

The bar kind is required: the target is checked against that kind's set, never the union, so a
cross-kind target (`T=0.94` on a guard ladder, `D_c<=0` on a new-ability one) is refused. U6f.9b
derives the kind from the role (guard: `speed-up`, else `new-ability`) and passes the
`SelectedText`'s `selection` and `ladder` together, which belong to one candidate and text; the
pair is not cross-checked beyond the family (a `Selection` beside another text's ladder of the same
family is not detectable here).

Combinations no ladder emits are refused as ladder bugs: rung 4 with no unmet bar; a rung-4
speed-up ladder with `latencyUnmet` false (rung 4 is reached only when no setting meets `s <= s_2`);
`max-min-agreement` with `agreementUnmet` false (emitted only at rung 4, which sets it); rung 1 at
`mcnemar-nonreject` (rung 1 is `D_c<=0` or a `T`).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decisionmodels.fit import Selection
    from decisionmodels.fitstats import LadderResult

FIT_OUTCOMES = frozenset(
    {
        "within-margin",
        "below-margin",
        "fit-infeasible: ranking",
        "fit-infeasible: agreement",
        "fit-infeasible: latency",
    }
)

QUALITY_TARGETS_BY_BAR_KIND: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "speed-up": frozenset({"D_c<=0", "mcnemar-nonreject"}),
        "new-ability": frozenset(
            {"T=0.94", "T=0.93", "T=0.92", "T=0.91", "T=0.90", "max-min-agreement"}
        ),
    }
)

QUALITY_TARGETS = frozenset().union(*QUALITY_TARGETS_BY_BAR_KIND.values())

_BEST_TARGETS = ("D_c<=0", "T=0.94")


def validate_quality_target(target: object, bar_kind: str) -> None:
    """Refuse (`ValueError` naming the value) a `qualityTarget` outside `bar_kind`'s closed set.

    `bar_kind` is `speed-up` or `new-ability`; any other value is refused.
    """
    if bar_kind not in QUALITY_TARGETS_BY_BAR_KIND:
        raise ValueError(
            f"unknown bar kind {bar_kind!r} (one of {sorted(QUALITY_TARGETS_BY_BAR_KIND)})"
        )
    allowed = QUALITY_TARGETS_BY_BAR_KIND[bar_kind]
    if type(target) is not str or target not in allowed:
        raise ValueError(f"qualityTarget {target!r} is not a {bar_kind} target ({sorted(allowed)})")


def _refuse_ladder_bug(ladder: LadderResult, bar_kind: str) -> None:
    target, rung = ladder.qualityTarget, ladder.rung
    unmet = ladder.agreementUnmet or ladder.latencyUnmet
    problem = None
    if rung == 4 and not unmet:
        problem = "rung 4 with no unmet bar"
    elif rung == 4 and bar_kind == "speed-up" and not ladder.latencyUnmet:
        problem = "a speed-up rung 4 with latencyUnmet false"
    elif target == "max-min-agreement" and not ladder.agreementUnmet:
        problem = "max-min-agreement with agreementUnmet false"
    elif rung == 1 and target == "mcnemar-nonreject":
        problem = "rung 1 at mcnemar-nonreject"
    elif rung is None and not unmet:
        problem = "rung None with no unmet bar"
    if problem is not None:
        raise ValueError(
            f"inconsistent LadderResult ({bar_kind}, rung {rung}, {target!r}): {problem}"
        )


def assemble_outcome(ladder: LadderResult, selection: Selection, bar_kind: str) -> tuple[str, str]:
    """The `(fit.outcome, fit.qualityTarget)` pair for one candidate's chosen text."""
    if selection.candidate != ladder.family:
        raise ValueError(
            f"selection is for candidate {selection.candidate!r}, the ladder is {ladder.family!r}"
        )
    validate_quality_target(ladder.qualityTarget, bar_kind)
    _refuse_ladder_bug(ladder, bar_kind)
    target: str = ladder.qualityTarget
    if selection.noEligibleText:
        return "fit-infeasible: ranking", target
    if ladder.agreementUnmet:
        return "fit-infeasible: agreement", target
    if ladder.latencyUnmet:
        return "fit-infeasible: latency", target
    if ladder.rung == 1 and target in _BEST_TARGETS:
        return "within-margin", target
    return "below-margin", target
