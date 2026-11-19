"""Recorded item dispositions: the record check, the flag grammar and the per-role effects
(`docs/plans/typed-decision-models4.md` §3.11 "Recorded dispositions", §4.4 U6f.8c, §5 row U6f.8c;
`docs/plans/typed-decision-models-fit-ml.md` §4 "Recorded item disposition").

`build_dispositions` accepts an `--exclude-item` flag only when two `AbortRecord` files in the fits
directory (a) are distinct files with distinct `runId`; (b) agree with the flag on item, call kind
and text, and with each other on `failureKind` and `status`; (c) carry `inputs` equal to the
current run's; (d) have an evidence `failureKind` (`http`, `malformed`; never `timeout` or
`connection`); (e) each show both models resident in `residencyNow`. Anything else raises
`ExclusionRefused` and nothing is written.

`apply_dispositions` applies the effects to the collector's calls, for a disposition whose call
failed in this run only (else `ExclusionDidNotFire`), per role and call kind:

- guard, hand-off failure: the item leaves the calibration set, for every text and both
  candidates (labels, cells and every call of the item are dropped); each guard cell is then
  re-checked against `CALIBRATION_FLOOR_PER_CELL` and one under it refuses;
- guard, decision failure: the item stays. A non-control text leaves selection unless every text
  failed that item (`fit.analyze_fit` reads the `failed` calls); the control, or every text failing,
  leaves the item concordant for that call. Nothing is pruned;
- join-in and message type, hand-off or decision failure: the item stays and the failed call is
  fail-closed (the ladders already score a `failed` call so). Per true-answer class the report
  carries `k` (distinct items under a disposition), `kWrong` (those the fail-closed answer gets
  wrong: every such item for message type, only a label-yes item for join-in, whose fail-closed
  "no" is right on a label-no item), `n_class` and the agreement ceiling
  `1 - kWrong/n_class`.

`fitcalls.summarize_calls` is computed over the full calls before any pruning, so this module
never feeds it.

Stdlib + this package only (FR-8).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

from decisionmodels.fitcalls import (
    ABORT_SCHEMA,
    AbortRecord,
    CollectedCalls,
    Disposition,
    ExclusionDidNotFire,
    FitInputs,
)
from decisionmodels.fitstats import CALIBRATION_FLOOR_PER_CELL, CALL_KINDS, ScoredCall
from decisionmodels.lmstudio import ResidentModel
from decisionmodels.questions import Question

GUARD = "guard-judge"
EVIDENCE_KINDS = ("http", "malformed")  # fit note §4: `timeout` and `connection` never are

EFFECT_ITEM_REMOVED = "item-removed"
EFFECT_TEXT_LEFT = "text-left-selection"
EFFECT_CONCORDANT = "concordant"
EFFECT_FAIL_CLOSED = "fail-closed"


class ExclusionRefused(RuntimeError):
    """The record check or the guard floor re-check refused: nothing is written."""


@dataclass(frozen=True)
class ExclusionFlag:
    """One parsed `--exclude-item <id>:<callKind>[:<textId>]`."""

    itemId: str  # noqa: N815
    callKind: str  # noqa: N815
    textId: str | None  # noqa: N815


@dataclass(frozen=True)
class DispositionEffect:
    """One listed item for the report: the `Disposition` (reason, record names) and its effect."""

    disposition: Disposition
    effect: str


@dataclass(frozen=True)
class ClassFigures:
    """Fit note §4, per true-answer class: `k` items under a disposition, `kWrong` of them
    scored wrong by the fail-closed answer, `n_class` items, and the agreement ceiling
    `1 - kWrong/n_class` for a band that hands them off."""

    label: Any
    k: int
    kWrong: int  # noqa: N815
    nClass: int  # noqa: N815
    ceiling: Fraction


@dataclass(frozen=True)
class DispositionReport:
    """The disposition fields U6f.9b prints."""

    effects: tuple[DispositionEffect, ...]
    byClass: tuple[ClassFigures, ...]  # noqa: N815  # new abilities; empty on the guard
    cellCounts: Mapping[str, int] | None  # noqa: N815  # guard: items per cell after exclusion


@dataclass(frozen=True)
class AppliedDispositions:
    """What `fit.analyze_fit` receives: pruned calls, labels and cells, and the report."""

    collected: CollectedCalls
    labels: Mapping[str, Any]
    cells: Mapping[str, str] | None
    report: DispositionReport


# --- the flag grammar ---


def parse_exclude_item(value: str) -> ExclusionFlag:
    """`<id>:<callKind>[:<textId>]`; `textId` is required exactly when the call kind is `decision`.
    Raises `ValueError` on anything else."""
    parts = value.split(":", 2)
    if len(parts) < 2 or not parts[0] or parts[1] not in CALL_KINDS:
        raise ValueError(
            f"--exclude-item {value!r}: expected <id>:<callKind>[:<textId>], "
            f"callKind one of {CALL_KINDS}"
        )
    item_id, call_kind = parts[0], parts[1]
    text_id = parts[2] if len(parts) == 3 else None
    if call_kind == "decision" and not text_id:
        raise ValueError(f"--exclude-item {value!r}: a decision call needs a textId")
    if call_kind == "handoff" and text_id is not None:
        raise ValueError(f"--exclude-item {value!r}: a handoff call takes no textId")
    return ExclusionFlag(item_id, call_kind, text_id)


# --- the record check ---


def _resident(rows: Any) -> tuple[ResidentModel, ...] | None:
    if rows is None:
        return None
    return tuple(ResidentModel(row["id"], row["state"]) for row in rows)


def record_from_json(data: Mapping[str, Any]) -> AbortRecord:
    """An `AbortRecord` from its file's JSON; `ValueError`/`KeyError`/`TypeError` on a malformed
    one (the schema string, a missing or extra key)."""
    if data["schema"] != ABORT_SCHEMA:
        raise ValueError(f"schema {data['schema']!r}")
    status = data["status"]
    if not (status is None or (isinstance(status, int) and not isinstance(status, bool))):
        raise ValueError(f"status {status!r}")
    return AbortRecord(
        schema=data["schema"],
        runId=str(data["runId"]),
        itemId=data["itemId"],
        callKind=data["callKind"],
        textId=data["textId"],
        failureKind=data["failureKind"],
        status=status,
        elapsedSeconds=data["elapsedSeconds"],
        inputs=FitInputs(**data["inputs"]),
        residencyStart=_resident(data["residencyStart"]) or (),
        residencyNow=_resident(data["residencyNow"]),
    )


def load_abort_records(fits_dir: Path) -> tuple[list[tuple[Path, AbortRecord]], list[str]]:
    """Every readable `abort-*.json` of `fits_dir` in file-name order, and a note per file that is
    not one (unreadable, wrong schema, malformed)."""
    records: list[tuple[Path, AbortRecord]] = []
    skipped: list[str] = []
    for path in sorted(fits_dir.glob("abort-*.json")):
        try:
            records.append((path, record_from_json(json.loads(path.read_text(encoding="utf-8")))))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append(f"{path.name}: not an abort record ({type(exc).__name__}: {exc})")
    return records, skipped


def _record_defect(record: AbortRecord, inputs: FitInputs) -> str | None:
    """Why one record cannot back a disposition: clauses (c), (d) and (e), or `None`."""
    if record.inputs != inputs:
        keys = [k for k in vars(inputs) if getattr(record.inputs, k) != getattr(inputs, k)]
        return f"inputs differ from this run's in {keys}"
    if record.failureKind not in EVIDENCE_KINDS:
        return f"failureKind {record.failureKind!r} is not evidence (only {EVIDENCE_KINDS})"
    if record.residencyNow is None:
        return "residencyNow is None: the models' residency at abort time is unknown"
    resident = {row.id for row in record.residencyNow}
    absent = sorted({inputs.decisionModel, inputs.handoffModel} - resident)
    if absent:
        return f"model(s) not resident in residencyNow: {absent}"
    return None


def _describe(flag: ExclusionFlag) -> str:
    text = "" if flag.textId is None else f", text {flag.textId!r}"
    return f"{flag.callKind} call for item {flag.itemId!r}{text}"


def _build_one(
    flag: ExclusionFlag,
    reason: str,
    inputs: FitInputs,
    records: Sequence[tuple[Path, AbortRecord]],
    skipped: Sequence[str],
) -> Disposition:
    matching = [
        (path, rec)
        for path, rec in records
        if (rec.itemId, rec.callKind, rec.textId) == (flag.itemId, flag.callKind, flag.textId)
    ]
    problems = list(skipped)
    if len(matching) < 2:
        raise ExclusionRefused(
            f"cannot exclude {_describe(flag)}: need two abort records for it, found "
            f"{len(matching)}" + "".join(f"; {p}" for p in problems)
        )
    usable: list[tuple[Path, AbortRecord]] = []
    for path, rec in matching:
        defect = _record_defect(rec, inputs)
        if defect is None:
            usable.append((path, rec))
        else:
            problems.append(f"{path.name}: {defect}")
    for (path_a, rec_a), (path_b, rec_b) in combinations(usable, 2):
        if rec_a.runId == rec_b.runId:
            problems.append(f"{path_a.name} and {path_b.name}: the same runId {rec_a.runId!r}")
        elif (rec_a.failureKind, rec_a.status) != (rec_b.failureKind, rec_b.status):
            problems.append(
                f"{path_a.name} and {path_b.name}: they disagree "
                f"({rec_a.failureKind}/{rec_a.status} against {rec_b.failureKind}/{rec_b.status})"
            )
        else:
            return Disposition(
                itemId=flag.itemId,
                callKind=flag.callKind,
                textId=flag.textId,
                failureKind=rec_a.failureKind,
                status=rec_a.status,
                reason=f"{reason} [records: {path_a.name}, {path_b.name}]",
                recordFiles=(path_a.name, path_b.name),
            )
    raise ExclusionRefused(
        f"cannot exclude {_describe(flag)}: no two abort records qualify: " + "; ".join(problems)
    )


def build_dispositions(
    flags: Sequence[ExclusionFlag], reason: str | None, *, inputs: FitInputs, fits_dir: Path
) -> tuple[Disposition, ...]:
    """One `Disposition` per flag, or `ExclusionRefused` (the record check, module docstring).
    Reads `fits_dir`, writes nothing."""
    if not flags:
        return ()
    if reason is None or not reason.strip():
        raise ExclusionRefused("--exclude-item needs --reason")
    if len(set(flags)) != len(flags):
        raise ExclusionRefused("--exclude-item is given twice for one call")
    records, skipped = load_abort_records(fits_dir) if fits_dir.is_dir() else ([], [])
    return tuple(_build_one(flag, reason.strip(), inputs, records, skipped) for flag in flags)


# --- the effects ---


def _fired_call(calls: Iterable[ScoredCall], disposition: Disposition) -> ScoredCall:
    for call in calls:
        if (call.itemId, call.callKind, call.textId) == disposition.key[:3]:
            if call.outcome != "failed":
                raise ExclusionDidNotFire(disposition, call.outcome)
            return call
    raise ExclusionDidNotFire(disposition, "never made")


def _guard_effects(
    dispositions: Sequence[Disposition],
    calls: Sequence[ScoredCall],
    texts: Sequence[Question],
    removed: set[str],
) -> tuple[DispositionEffect, ...]:
    everyone = {t.id for t in texts}
    control = {t.id for t in texts if t.control}
    failing: dict[str, set[str]] = {}
    for call in calls:
        if call.callKind == "decision" and call.outcome == "failed":
            failing.setdefault(call.itemId, set()).add(call.textId)  # type: ignore[arg-type]
    out = []
    for d in dispositions:
        if d.itemId in removed:
            effect = EFFECT_ITEM_REMOVED
        elif d.textId in control or failing[d.itemId] == everyone:
            effect = EFFECT_CONCORDANT
        else:
            effect = EFFECT_TEXT_LEFT
        out.append(DispositionEffect(d, effect))
    return tuple(out)


def _cell_counts(cells: Mapping[str, str], names: Iterable[str]) -> dict[str, int]:
    counts = dict.fromkeys(sorted(names), 0)
    for cell in cells.values():
        counts[cell] += 1
    return counts


def check_guard_floor(cells: Mapping[str, str], removed: Iterable[str]) -> dict[str, int]:
    """Items per cell after `removed` leave the calibration set; a cell under
    `CALIBRATION_FLOOR_PER_CELL` raises `ExclusionRefused` when anything was removed. Depends only
    on the flags and the calibration cells, so `fit.run_fit` runs it before any call."""
    gone = {i for i in removed if i in cells}
    kept = {i: c for i, c in cells.items() if i not in gone}
    counts = _cell_counts(kept, set(cells.values()))
    if gone:
        for cell, count in counts.items():
            if count < CALIBRATION_FLOOR_PER_CELL:
                raise ExclusionRefused(
                    f"excluding hand-off item(s) {sorted(gone)} leaves cell {cell!r} with "
                    f"{count} items, under the floor of {CALIBRATION_FLOOR_PER_CELL}"
                )
    return counts


def apply_dispositions(
    role: str,
    collected: CollectedCalls,
    dispositions: Sequence[Disposition],
    *,
    labels: Mapping[str, Any],
    cells: Mapping[str, str] | None,
    texts: Sequence[Question],
) -> AppliedDispositions:
    """The calls, labels and cells `analyze_fit` receives (module docstring). Each disposition's
    call must have failed in `collected`, else `ExclusionDidNotFire`; nothing is applied then. A
    guard hand-off exclusion that leaves a cell under the floor raises `ExclusionRefused`."""
    for d in dispositions:
        _fired_call(collected.calls, d)
    if role == GUARD:
        if cells is None:
            raise ValueError("apply_dispositions: the guard needs `cells`")
        removed = {d.itemId for d in dispositions if d.callKind == "handoff"}
        kept_cells = {i: c for i, c in cells.items() if i not in removed}
        counts = check_guard_floor(cells, removed)
        pruned = CollectedCalls(
            tuple(c for c in collected.calls if c.itemId not in removed),
            collected.parseFailures,
            {k: v for k, v in collected.maxProbabilities.items() if k[0] not in removed},
        )
        report = DispositionReport(
            _guard_effects(dispositions, collected.calls, texts, removed), (), counts
        )
        return AppliedDispositions(
            pruned, {i: v for i, v in labels.items() if i not in removed}, kept_cells, report
        )
    by_item = {d.itemId for d in dispositions}
    classes: dict[Any, list[str]] = {}
    for item_id, label in labels.items():
        classes.setdefault(label, []).append(item_id)
    figures = []
    for label, ids in classes.items():
        k = sum(1 for i in ids if i in by_item)
        # join-in's fail-closed answer is "no": right on a label-no item
        k_wrong = k if role != "join-in-judge" or label is True else 0
        figures.append(
            ClassFigures(label, k, k_wrong, len(ids), Fraction(len(ids) - k_wrong, len(ids)))
        )
    effects = tuple(DispositionEffect(d, EFFECT_FAIL_CLOSED) for d in dispositions)
    return AppliedDispositions(
        collected, labels, cells, DispositionReport(effects, tuple(figures), None)
    )
