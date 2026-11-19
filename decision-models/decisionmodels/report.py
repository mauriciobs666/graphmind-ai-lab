"""The fit report (`docs/plans/typed-decision-models4.md` §3.11 "Fit report", §4.4 U6f.9b).

`fits/<role>-<datasetVersion>-f<N>.fit-report.json`, one per fit, beside the two candidates. It is
not the manifest and never enters `fitManifest.sha256`; it holds no item text (item ids, cell and
class names, counts and figures only). Every call field is descriptive, never gating.

**Serialization.** An exact `fractions.Fraction` is `{"fraction": "a/b", "decimal": <float>}`
everywhere (`1` is `"1/1"`; the decimal is `float(fraction)`), so the exact value and a readable one
travel together. A dataclass is an object of its fields; a tuple a list; a mapping an object whose
keys are strings (a `bool` key is `"true"`/`"false"`). A cell or class with no value on a half is
`null` (reported absent, never 0). The text is `json.dumps(indent=2, sort_keys=True)` plus LF.

Top level: `schema`, `role`, `datasetVersion`, `fitOrdinal`, `callsFile`, `decisionModel`,
`handoffModel`, `decisionTimeoutSeconds`, `questionsSha256`, `handoffSha256`, `residency`
(`start`, `end`: `[{id, state}]`), `calls` (per call kind: `callKind`, `n`, `k`,
`dispositionItemIds`, `medianSeconds`, `maxSeconds`, `wilsonUpper95`, `halfBudgetCount`,
`parseFailures`), `dispositions` (`items`, `byClass`, `cellCounts`) and `candidates` (one per
family: `candidateId`, `chosen` {`textId`, `rung`, `qualityTarget`, `outcome`, `noEligibleText`} and
`texts`, each with the `TextScore` fields, `control`, `maxProbDropped`, `splitHalf` and the `ladder`
(`barValues` included)).

Stdlib + this package only (FR-8); `fit` is imported for typing only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from decisionmodels.outcome import assemble_outcome

if TYPE_CHECKING:
    from decisionmodels.fit import FitRun

REPORT_SCHEMA = "decision-models.fit-report/1"


def jsonable(value: Any) -> Any:
    """`value` as JSON-ready data under the module's serialization rules."""
    if isinstance(value, Fraction):
        return {"fraction": f"{value.numerator}/{value.denominator}", "decimal": float(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {_key(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def _key(key: Any) -> str:
    if isinstance(key, bool):
        return "true" if key else "false"
    return str(key)


def _half(half: Any) -> dict[str, Any]:
    return {
        "direction": half.direction,
        "fittedPolicy": jsonable(half.fitted.policy),
        "fittedRung": half.fitted.rung,
        "onFit": jsonable(half.onFit),
        "onScored": jsonable(half.onScored),
        "shift": jsonable(half.shift),
    }


def build_report(
    run: FitRun, ordinal: int, candidate_ids: Mapping[str, str], bar_kind: str
) -> dict[str, Any]:
    """The report document for `run`; `candidate_ids` maps family -> candidate id."""
    control = {t.id: t.control for t in run.texts}
    candidates: dict[str, Any] = {}
    for chosen in run.analysis.selected:
        family = chosen.selection.candidate
        outcome, target = assemble_outcome(chosen.ladder, chosen.selection, bar_kind)
        texts = []
        for score in (s for s in run.analysis.scores if s.candidate == family):
            texts.append(
                {
                    "textId": score.textId,
                    "control": control[score.textId],
                    "auc": score.auc,
                    "aucUndefined": score.aucUndefined,
                    "maxProbAuc": score.maxProbAuc,
                    "maxProbDropped": run.analysis.maxProbDropped.get(score.textId),
                    "overLimit": {"k": score.overLimit[0], "n": score.overLimit[1]},
                    "leftSelection": score.leftSelection,
                    "splitHalf": [_half(h) for h in score.splitHalf],
                    "ladder": jsonable(score.ladder),
                }
            )
        candidates[family] = {
            "candidateId": candidate_ids[family],
            "chosen": {
                "textId": chosen.selection.textId,
                "rung": chosen.ladder.rung,
                "qualityTarget": target,
                "outcome": outcome,
                "noEligibleText": chosen.selection.noEligibleText,
            },
            "texts": texts,
        }
    return {
        "schema": REPORT_SCHEMA,
        "role": run.inputs.role,
        "datasetVersion": run.inputs.datasetVersion,
        "fitOrdinal": ordinal,
        "callsFile": run.callsFile.name if run.callsFile else None,
        "decisionModel": run.inputs.decisionModel,
        "handoffModel": run.inputs.handoffModel,
        "decisionTimeoutSeconds": run.inputs.decisionTimeoutSeconds,
        "questionsSha256": run.inputs.questionsSha256,
        "handoffSha256": run.inputs.handoffSha256,
        "residency": {"start": jsonable(run.residencyStart), "end": jsonable(run.residencyEnd)},
        "calls": jsonable(dict(run.summaries)),
        "dispositions": {
            "items": [
                {**jsonable(e.disposition), "effect": e.effect} for e in run.dispositions.effects
            ],
            "byClass": jsonable(run.dispositions.byClass),
            "cellCounts": jsonable(run.dispositions.cellCounts),
        },
        "candidates": candidates,
    }
