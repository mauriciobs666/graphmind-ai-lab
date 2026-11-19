"""The fit: text scores, per-candidate selection and the CLI (`python -m decisionmodels.fit`)
(`docs/plans/typed-decision-models4.md` §3.11, §4.4 U6f.8b, §5 row U6f.8b;
`docs/plans/typed-decision-models-fit-ml.md` §3; method note §Q2.3-§Q2.5).

`analyze_fit` is the pure half; `run_fit` makes the calls and writes the calls file, and `main`
emits (`emit.py`, `report.py`). `analyze_fit` reads the collector's shared calls
(`fitcalls.collect_calls`) and, for each family (`handoff`, `decision-only`: the two candidates)
and each question text, runs that role's ladder (`guard_ladder`, `predicate_ladder`,
`choice_ladder`), then selects one text per candidate. It makes no call; both candidates read the
same `CollectedCalls`. Its result holds one `TextScore` per (candidate, text) and one
`SelectedText` per candidate, which `outcome.py` and `emit.py` consume.

`candidate` is the family name (`handoff` or `decision-only`), equal to `LadderResult.family`;
`emit.py` maps it to the candidate ids of §3.11.

Per text, the ranking diagnostic of note §Q2.3: `auc` is the ROC-AUC, over the items with an `ok`
decision call, of the score against a binary target: a predicate's `p` against the label, a choice's
`confidence` against "the choice equals the label" (a choice threshold ranks items by reliability,
note §Q2.5). When a class is absent the figure is undefined and `auc` is 0.5 (no ranking
information; the text leaves selection, as a constant score does). `maxProbAuc` is the same for
`max(probabilities)` (choice only, exploratory, gates nothing): `None` on a predicate point or when
a class is absent. An `ok` choice call without a `maxProbabilities` entry (an empty `probabilities`
list) has no maximum: its item is dropped from that text's `maxProbAuc` pairs and counted in
`FitAnalysis.maxProbDropped`.

Selection (fit note §3), per candidate:
1. on the guard, a non-control text with a `failed` decision call leaves ("decision-failure"),
   except on an item where every text failed; the control never leaves that way;
2. among the rest, a text whose AUC is at or below 0.5 leaves ("auc") while any text has AUC above
   0.5; `noEligibleText` is true exactly when none has (both candidates alike);
3. the best of what remains, by `(rung, agreementUnmet, qualityTarget, key, not control, file
   order)`, smallest first: a `rung` `None` text ranks below rungs 1-4, `agreementUnmet` false
   before true, `mcnemar-nonreject` after every other target of one rung (only guard rung 3 can
   mix targets), the key compared exactly (integers and `Fraction`s), the control, then the
   order of `questions.json`.

Split-half (note §Q2.4): the calibration ids are split by `fitstats.split_half` into halves A and
B; in each direction the ladder is fitted on one half and the fitted policy scored on the other.
`HalfShift` holds the values on both halves and their difference: `s` and each `D_c` (guard) or each
class's agreement (new abilities). The halves span every calibration id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from decisionmodels import render
from decisionmodels.emit import EmitRefused, emit, write_calls_file
from decisionmodels.exclusion import (
    DispositionReport,
    ExclusionFlag,
    ExclusionRefused,
    apply_dispositions,
    build_dispositions,
    check_guard_floor,
    parse_exclude_item,
)
from decisionmodels.fitcalls import (
    CalibrationItem,
    CallSummary,
    CollectedCalls,
    ExclusionDidNotFire,
    FitAborted,
    FitInputs,
    ResidencyRefused,
    check_end,
    check_start,
    collect_calls,
    summarize_calls,
)
from decisionmodels.fitstats import FAMILIES, LadderResult, ScoredCall, roc_auc, split_half
from decisionmodels.hashing import input_hash
from decisionmodels.ladder import (
    TARGET_NONREJECT,
    choice_ladder,
    guard_ladder,
    index_calls,
    predicate_ladder,
)
from decisionmodels.lmstudio import LMStudio, LMStudioError, ResidentModel
from decisionmodels.points import ROLES
from decisionmodels.questions import FIT_DIR, Question, load_questions

GUARD = "guard-judge"
AUC_ELIGIBLE_ABOVE = 0.5
LEFT_AUC = "auc"
LEFT_DECISION_FAILURE = "decision-failure"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

_RENDERERS: Mapping[str, Callable[[Mapping[str, Any]], str]] = {
    "guard-judge": render.render_guard_item,
    "join-in-judge": render.render_join_in_item,
    "message-type-classifier": render.render_message_type_item,
}


# --- typed records (plan4 §3.11) ---


@dataclass(frozen=True)
class HalfShift:
    """One split-half direction: the ladder fitted on `fitIds`, the fitted policy scored on
    `scoredIds`. `onFit`/`onScored` are `{"s": Fraction, "n": {cell: items}, "D_c": {cell: int}}`
    (guard) or `{"s", "n", "agreement": {class: Fraction}}`; `shift` is `onScored - onFit` in `s`
    and per cell or class, `None` for one absent from either half (fit note §6)."""

    direction: str
    fitIds: tuple[str, ...]  # noqa: N815
    scoredIds: tuple[str, ...]  # noqa: N815
    fitted: LadderResult
    onFit: Mapping[str, Any]  # noqa: N815
    onScored: Mapping[str, Any]  # noqa: N815
    shift: Mapping[str, Any]


@dataclass(frozen=True)
class TextScore:
    """One (candidate, text) (§3.11): this unit produces every field."""

    candidate: str
    textId: str  # noqa: N815
    auc: float
    aucUndefined: bool  # noqa: N815  # no `ok` call or one class absent: `auc` is 0.5, unmeasured
    maxProbAuc: float | None  # noqa: N815
    overLimit: tuple[int, int]  # noqa: N815
    splitHalf: tuple[HalfShift, HalfShift]  # noqa: N815
    ladder: LadderResult
    leftSelection: str | None  # noqa: N815


@dataclass(frozen=True)
class Selection:
    """One per candidate (§3.11); `noEligibleText` holds for both candidates or neither."""

    candidate: str
    textId: str  # noqa: N815
    noEligibleText: bool  # noqa: N815


@dataclass(frozen=True)
class SelectedText:
    """What U6f.9a/9b consume: the selection, the chosen question, its ladder and score."""

    selection: Selection
    text: Question
    ladder: LadderResult
    score: TextScore


@dataclass(frozen=True)
class FitAnalysis:
    scores: tuple[TextScore, ...]  # candidate-major, then `questions.json` order
    selected: tuple[SelectedText, ...]  # one per candidate, in `FAMILIES` order
    maxProbDropped: Mapping[str, int]  # noqa: N815  # textId -> `ok` choice items with no maximum


# --- AUC ---


def _auc_or_none(scores: Sequence[float], truths: Sequence[bool]) -> float | None:
    if not scores or all(truths) or not any(truths):
        return None
    return roc_auc(scores, truths)


def _text_figures(
    role: str,
    calls: Iterable[ScoredCall],
    labels: Mapping[str, Any],
    text_id: str,
    max_probabilities: Mapping[tuple[str, str], float],
) -> tuple[float, bool, float | None, tuple[int, int], int]:
    """`(auc, aucUndefined, maxProbAuc, (k, n) over-limit, maxProbAuc-dropped)` for one text."""
    decisions = [c for c in calls if c.callKind == "decision" and c.textId == text_id]
    choice_point = role == "message-type-classifier"
    scores: list[float] = []
    truths: list[bool] = []
    max_scores: list[float] = []
    max_truths: list[bool] = []
    dropped = 0
    for call in decisions:
        if call.outcome != "ok":
            continue
        label = labels[call.itemId]
        if choice_point:
            choice, score = call.answer
            truth = choice == label
        else:
            score, truth = call.answer, label is True
        scores.append(score)
        truths.append(truth)
        if choice_point:
            maximum = max_probabilities.get((call.itemId, text_id))
            if maximum is None:
                dropped += 1
            else:
                max_scores.append(maximum)
                max_truths.append(truth)
    auc = _auc_or_none(scores, truths)
    max_auc = _auc_or_none(max_scores, max_truths) if choice_point else None
    over = sum(1 for c in decisions if c.outcome == "over-limit")
    return (0.5 if auc is None else auc), auc is None, max_auc, (over, len(decisions)), dropped


# --- ladders and the policy scorer ---


def run_ladder(
    role: str,
    calls: Iterable[ScoredCall],
    *,
    labels: Mapping[str, Any],
    cells: Mapping[str, str] | None,
    text_id: str,
    family: str,
) -> LadderResult:
    """The role's ladder for one text over one family (`cells` is the guard's only)."""
    if role == GUARD:
        if cells is None:
            raise ValueError("run_ladder: the guard needs `cells`")
        return guard_ladder(calls, labels=labels, cells=cells, text_id=text_id, family=family)
    if role == "join-in-judge":
        return predicate_ladder(calls, labels=labels, text_id=text_id, family=family)
    if role == "message-type-classifier":
        return choice_ladder(calls, labels=labels, text_id=text_id, family=family)
    raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")


def _band(policy: Mapping[str, Any]) -> tuple[float, float]:
    return (policy["l"], policy["h"]) if "l" in policy else (policy["t"], policy["t"])


def policy_values(
    role: str,
    family: str,
    policy: Mapping[str, Any],
    calls: Iterable[ScoredCall],
    *,
    labels: Mapping[str, Any],
    cells: Mapping[str, str] | None,
    text_id: str,
) -> dict[str, Any]:
    """A fitted policy scored over `labels`' items: `{"s", "n", "D_c"}` (guard) or `{"s", "n",
    "agreement"}`; `n` is the item count per cell (guard) or true-answer class.

    Scoring is the ladder's (`decisionmodels/ladder.py` docstring), applied to a given policy:
    the same routing (below `l` no, `l <= p < h` hand-off, from `h` up yes; a choice below `c`
    hands off), the same fail-closed / concordant rules; `tests/test_fit.py` holds it equal to the
    ladder's own `barValues` on the half the policy was fitted on.
    """
    if family not in FAMILIES:
        raise ValueError(f"policy_values: family must be one of {FAMILIES}, got {family!r}")
    if not labels:
        raise ValueError("policy_values: no items")
    hand, dec = index_calls(list(calls), labels, text_id, "policy_values")
    handoff_family = family == "handoff"
    handed = 0
    d_c: dict[str, int] = {}
    right: dict[Any, int] = {}
    total: dict[Any, int] = {}  # items per cell (guard) or true-answer class
    for item_id, label in labels.items():
        h, d = hand[item_id], dec[item_id]
        is_handed = False
        model: Any = None  # the model's answer, or None when there is none (failed/over-limit)
        if d.outcome == "over-limit":
            is_handed = handoff_family
        elif d.outcome == "ok":
            if role == "message-type-classifier":
                choice, confidence = d.answer
                if confidence < policy["c"]:
                    is_handed = True
                else:
                    model = choice
            else:
                low, high = _band(policy)
                if d.answer < low:
                    model = False
                elif d.answer < high:
                    is_handed = True
                else:
                    model = True
        if is_handed:
            handed += 1
        if role == GUARD:
            d_c.setdefault(cells[item_id], 0)  # type: ignore[index]
            total[cells[item_id]] = total.get(cells[item_id], 0) + 1  # type: ignore[index]
            if is_handed or model is None:
                continue
            llm_wrong = not (h.outcome == "ok" and h.answer is label)
            model_wrong = model is not label
            d_c[cells[item_id]] += model_wrong and not llm_wrong  # type: ignore[index]
            d_c[cells[item_id]] -= llm_wrong and not model_wrong  # type: ignore[index]
            continue
        if role == "join-in-judge":
            llm_right = (h.answer is label) if h.outcome == "ok" else not label
            closed_right = not label  # fail-closed answers "no"
            answered = (model is label) if model is not None else closed_right
            correct = llm_right if is_handed else answered
        else:
            llm_right = h.outcome == "ok" and h.answer == label
            correct = llm_right if is_handed else (model is not None and model == label)
        total[label] = total.get(label, 0) + 1
        right[label] = right.get(label, 0) + bool(correct)
    values: dict[str, Any] = {"s": Fraction(handed, len(labels)), "n": dict(total)}
    if role == GUARD:
        values["D_c"] = d_c
    else:
        values["agreement"] = {k: Fraction(right.get(k, 0), n) for k, n in total.items()}
    return values


def _shift(on_fit: Mapping[str, Any], on_scored: Mapping[str, Any]) -> dict[str, Any]:
    """`onScored - onFit` in `s` and per cell (`D_c`, guard) or class (`agreement`); a cell or class
    absent from either half is `None`, never 0 (fit note §6)."""
    out: dict[str, Any] = {"s": on_scored["s"] - on_fit["s"]}
    key = "D_c" if "D_c" in on_fit else "agreement"
    names = sorted(set(on_fit["n"]) | set(on_scored["n"]))
    out[key] = {
        name: (on_scored[key][name] - on_fit[key][name])
        if name in on_fit["n"] and name in on_scored["n"]
        else None
        for name in names
    }
    return out


def split_half_shifts(
    role: str,
    family: str,
    calls: Sequence[ScoredCall],
    *,
    labels: Mapping[str, Any],
    cells: Mapping[str, str] | None,
    text_id: str,
) -> tuple[HalfShift, HalfShift]:
    """Both directions over every calibration id: fit on half A and score half B, then fit on B
    and score A (note §Q2.4; the partition is `fitstats.split_half`'s)."""
    half_a, half_b = split_half(labels)
    shifts = []
    for direction, fit_ids, scored_ids in (
        ("first->second", half_a, half_b),
        ("second->first", half_b, half_a),
    ):

        def part(ids: tuple[str, ...]) -> dict[str, Any]:
            return {
                "calls": [c for c in calls if c.itemId in ids],
                "labels": {i: labels[i] for i in ids},
                "cells": None if cells is None else {i: cells[i] for i in ids},
            }

        fit_part, scored_part = part(fit_ids), part(scored_ids)
        fitted = run_ladder(
            role,
            fit_part["calls"],
            labels=fit_part["labels"],
            cells=fit_part["cells"],
            text_id=text_id,
            family=family,
        )
        values = [
            policy_values(
                role,
                family,
                fitted.policy,
                p["calls"],
                labels=p["labels"],
                cells=p["cells"],
                text_id=text_id,
            )
            for p in (fit_part, scored_part)
        ]
        shifts.append(
            HalfShift(direction, fit_ids, scored_ids, fitted, values[0], values[1], _shift(*values))
        )
    return shifts[0], shifts[1]


# --- selection (fit note §3) ---


def _rank(index: int, text: Question, ladder: LadderResult) -> tuple[Any, ...]:
    return (
        5 if ladder.rung is None else ladder.rung,
        ladder.agreementUnmet,
        ladder.qualityTarget == TARGET_NONREJECT,
        ladder.key,
        not text.control,
        index,
    )


def eligibility(
    texts: Sequence[Question], aucs: Mapping[str, float], decision_failure: Collection[str]
) -> tuple[list[Question], bool, dict[str, str]]:
    """`(eligible texts, noEligibleText, leftSelection by text id)` (fit note §3). A text that left
    on a decision failure does not count toward "any text has AUC above 0.5"; when none does, every
    text still in selection stays eligible and `noEligibleText` is true."""
    in_selection = [t for t in texts if t.id not in decision_failure]
    eligible = [t for t in in_selection if aucs[t.id] > AUC_ELIGIBLE_ABOVE]
    no_eligible = not eligible
    if no_eligible:
        eligible = in_selection
    left = {t.id: LEFT_DECISION_FAILURE for t in texts if t.id in decision_failure}
    left |= {t.id: LEFT_AUC for t in in_selection if t not in eligible}
    return eligible, no_eligible, left


def best_text(
    texts: Sequence[Question], eligible: Sequence[Question], ladders: Mapping[str, LadderResult]
) -> Question:
    """The best eligible text by fit note §3's order (module docstring); ties are exact."""
    order = {t.id: i for i, t in enumerate(texts)}
    return min(eligible, key=lambda t: _rank(order[t.id], t, ladders[t.id]))


def _decision_failure_texts(
    role: str, calls: Iterable[ScoredCall], texts: Sequence[Question]
) -> set[str]:
    """The guard's texts that leave on a failed decision call (fit note §4): a non-control text
    failing an item that some text answered. Elsewhere a failed decision call is fail-closed."""
    if role != GUARD:
        return set()
    failed_by_item: dict[str, set[str]] = {}
    for call in calls:
        if call.callKind == "decision" and call.outcome == "failed":
            failed_by_item.setdefault(call.itemId, set()).add(call.textId)  # type: ignore[arg-type]
    everyone = {t.id for t in texts}
    left: set[str] = set()
    for failing in failed_by_item.values():
        if failing != everyone:
            left |= failing
    return left & {t.id for t in texts if not t.control}


def analyze_fit(
    role: str,
    collected: CollectedCalls,
    *,
    labels: Mapping[str, Any],
    cells: Mapping[str, str] | None,
    texts: Sequence[Question],
) -> FitAnalysis:
    """Both candidates' ladders, scores and selections over the shared calls. Pure."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    if not texts:
        raise ValueError("analyze_fit: no question texts")
    calls = tuple(collected.calls)
    figures = {}
    dropped: dict[str, int] = {}
    for text in texts:
        *rest, text_dropped = _text_figures(
            role, calls, labels, text.id, collected.maxProbabilities
        )
        figures[text.id] = tuple(rest)
        if role == "message-type-classifier":
            dropped[text.id] = text_dropped
    eligible, no_eligible, left = eligibility(
        texts,
        {t.id: figures[t.id][0] for t in texts},
        _decision_failure_texts(role, calls, texts),
    )
    scores: list[TextScore] = []
    selected: list[SelectedText] = []
    for family in FAMILIES:
        ladders = {
            t.id: run_ladder(role, calls, labels=labels, cells=cells, text_id=t.id, family=family)
            for t in texts
        }
        family_scores = {
            t.id: TextScore(
                candidate=family,
                textId=t.id,
                auc=figures[t.id][0],
                aucUndefined=figures[t.id][1],
                maxProbAuc=figures[t.id][2],
                overLimit=figures[t.id][3],
                splitHalf=split_half_shifts(
                    role, family, calls, labels=labels, cells=cells, text_id=t.id
                ),
                ladder=ladders[t.id],
                leftSelection=left.get(t.id),
            )
            for t in texts
        }
        scores.extend(family_scores.values())
        best = best_text(texts, eligible, ladders)
        selected.append(
            SelectedText(
                Selection(family, best.id, no_eligible),
                best,
                ladders[best.id],
                family_scores[best.id],
            )
        )
    return FitAnalysis(tuple(scores), tuple(selected), dropped)


# --- calibration data ---


@dataclass(frozen=True)
class Calibration:
    items: tuple[CalibrationItem, ...]
    labels: Mapping[str, Any]
    cells: Mapping[str, str] | None
    #: `input_hash` of every calibration item loaded (the manifest's entries, U6f.9b)
    inputHashes: tuple[str, ...] = ()  # noqa: N815
    #: sha256 of the dataset's `items.jsonl` (the candidates' `provenance.sourceSha`)
    itemsSha256: str = ""  # noqa: N815


def guard_cell(item: Mapping[str, Any]) -> str:
    """FR-13's four guard cells: the error kind (`expected` yes: a wrong suspend; no: a wrong
    advance) by the evidence the item carries (`path`)."""
    kind = "falseSuspend" if item["expected"] is True else "falseAdvance"
    return f"{kind}RateBy{'Understanding' if item['path'] == 'understanding' else 'Turns'}"


def load_calibration(role: str, version: str, data_dir: Path = DATA_DIR) -> Calibration:
    """The calibration items of `data/<role>/<version>/` in `splits.json` order, each rendered as
    the one input both the decision call and the hand-off call send."""
    root = data_dir / role / version
    ids = json.loads((root / "splits.json").read_text(encoding="utf-8"))["calibration"]
    rows = {}
    items_bytes = (root / "items.jsonl").read_bytes()
    for line in items_bytes.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["itemId"]] = row
    missing = [i for i in ids if i not in rows]
    if missing:
        raise ValueError(f"calibration ids without an item: {missing[:5]}")
    render_item = _RENDERERS[role]
    items = tuple(CalibrationItem(i, render_item(rows[i]), render_item(rows[i])) for i in ids)
    labels = {i: rows[i]["expected"] for i in ids}
    cells = {i: guard_cell(rows[i]) for i in ids} if role == GUARD else None
    hashes = tuple(input_hash(rows[i], role) for i in ids)
    return Calibration(items, labels, cells, hashes, hashlib.sha256(items_bytes).hexdigest())


# --- the run and its CLI ---


class FitRefused(RuntimeError):
    """The run cannot start or cannot emit; nothing was written."""


@dataclass(frozen=True)
class FitRun:
    """Everything U6f.9b emits from: inputs, snapshots, analysis and the call summaries."""

    inputs: FitInputs
    texts: tuple[Question, ...]
    residencyStart: tuple[ResidentModel, ...]  # noqa: N815
    residencyEnd: tuple[ResidentModel, ...]  # noqa: N815
    collected: CollectedCalls
    analysis: FitAnalysis
    summaries: Mapping[str, CallSummary] = field(default_factory=dict)
    #: the listed dispositions with their effects, per-class figures (U6f.9b prints them)
    dispositions: DispositionReport = DispositionReport((), (), None)
    #: the calls file written before the analysis (U6f.9b)
    callsFile: Path | None = None  # noqa: N815
    inputHashes: tuple[str, ...] = ()  # noqa: N815
    sourceSha: str = ""  # noqa: N815


def validate_url(url: str) -> str:
    """`url` if it is an `http(s)` URL with a host, else `FitRefused` (Pass 44 n1: a bad base URL
    would otherwise surface as a bare `ValueError` from `urllib` at the first call)."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018  (raises ValueError on a bad port)
    except ValueError as exc:
        raise FitRefused(f"bad LM Studio URL {url!r}: {exc}") from exc
    if parts.scheme not in ("http", "https") or not host:
        raise FitRefused(f"bad LM Studio URL {url!r}: need http(s)://host[:port]")
    return url


def run_fit(
    client: Any,
    *,
    role: str,
    dataset_version: str,
    decision_model: str,
    handoff_model: str,
    decision_timeout_seconds: float,
    fits_dir: Path,
    data_dir: Path = DATA_DIR,
    fit_dir: Path = FIT_DIR,
    exclusions: Sequence[ExclusionFlag] = (),
    reason: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> FitRun:
    """Record check, residency start check, the shared calls once, residency end check, the
    dispositions' effects (`exclusion.apply_dispositions`), the analysis. The call summaries are
    computed over the full calls, before any exclusion pruning.

    Raises `FitRefused`/`ResidencyRefused`/`ExclusionRefused`/`ExclusionDidNotFire` (nothing
    written; a bad input is a `FitRefused`) or `FitAborted` (the one abort record).
    """
    if not (
        isinstance(decision_timeout_seconds, int | float)
        and decision_timeout_seconds > 0
        and math.isfinite(decision_timeout_seconds)
    ):
        raise FitRefused(
            "--decision-timeout-seconds must be a positive number, "
            f"got {decision_timeout_seconds!r}"
        )
    if role not in ROLES:
        raise FitRefused(f"unknown role {role!r}; expected one of {ROLES}")
    questions_path = fit_dir / role / "questions.json"
    try:
        texts = load_questions(role, questions_path)
        calibration = load_calibration(role, dataset_version, data_dir)
        questions_sha = hashlib.sha256(questions_path.read_bytes()).hexdigest()
        handoff_sha = hashlib.sha256((fit_dir / role / "handoff.md").read_bytes()).hexdigest()
    except (OSError, ValueError, KeyError) as exc:  # a missing or malformed input, before any call
        raise FitRefused(f"fit inputs unusable: {type(exc).__name__}: {exc}") from exc
    inputs = FitInputs(
        role=role,
        datasetVersion=dataset_version,
        questionsSha256=questions_sha,
        handoffSha256=handoff_sha,
        decisionModel=decision_model,
        handoffModel=handoff_model,
        decisionTimeoutSeconds=float(decision_timeout_seconds),
    )
    dispositions = build_dispositions(exclusions, reason, inputs=inputs, fits_dir=fits_dir)
    if role == "guard-judge":  # fail fast: the floor depends on the flags and the cells only
        check_guard_floor(
            calibration.cells,  # type: ignore[arg-type]
            (d.itemId for d in dispositions if d.callKind == "handoff"),
        )
    start = check_start(client, inputs)
    collected = collect_calls(
        client,
        inputs=inputs,
        items=calibration.items,
        texts=texts,
        residency_start=start,
        fits_dir=fits_dir,
        dispositions=dispositions,
        now=now,
    )
    try:
        end = check_end(client, inputs)
    except ResidencyRefused as exc:  # the calls are recoverable: label why, then refuse
        unreadable = isinstance(exc.__cause__, (LMStudioError, TimeoutError))
        try:
            write_calls_file(
                fits_dir,
                inputs=inputs,
                collected=collected,
                end_residency="unreadable" if unreadable else "refused",
                now=now,
            )
        except OSError as write_error:  # never mask the refusal
            exc.add_note(f"the calls file could not be written: {write_error}")
        raise
    calls_file = write_calls_file(
        fits_dir, inputs=inputs, collected=collected, end_residency="ok", now=now
    )
    applied = apply_dispositions(
        role,
        collected,
        dispositions,
        labels=calibration.labels,
        cells=calibration.cells,
        texts=texts,
    )
    analysis = analyze_fit(
        role, applied.collected, labels=applied.labels, cells=applied.cells, texts=texts
    )
    summaries = summarize_calls(
        collected.calls,
        decision_timeout_seconds=inputs.decisionTimeoutSeconds,
        parse_failures=collected.parseFailures,
    )
    return FitRun(
        inputs,
        tuple(texts),
        start,
        end,
        collected,
        analysis,
        summaries,
        applied.report,
        calls_file,
        calibration.inputHashes,
        calibration.itemsSha256,
    )


def _with_notes(exc: BaseException) -> str:
    """`str(exc)` and its notes (`ResidencyRefused` carries the calls-file write error as one)."""
    return "\n".join([str(exc), *getattr(exc, "__notes__", ())])


def _exclude_item(value: str) -> ExclusionFlag:
    try:
        return parse_exclude_item(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decisionmodels.fit")
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--decision-model", required=True)
    parser.add_argument("--handoff-model", required=True)
    parser.add_argument("--lmstudio-url", required=True)
    parser.add_argument(
        "--decision-timeout-seconds",
        required=True,
        type=float,
        help="the decision call's budget; also what the candidate ships. No default.",
    )
    parser.add_argument(
        "--exclude-item",
        action="append",
        default=[],
        type=_exclude_item,
        metavar="ID:CALLKIND[:TEXTID]",
        help="a recorded item disposition (needs two matching abort records and --reason); "
        "repeatable, never a default; TEXTID exactly when CALLKIND is decision.",
    )
    parser.add_argument(
        "--reason", action="append", help="why the item is excluded; required with --exclude-item"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--fits-dir", type=Path, help="default: <data-dir>/<role>/<dataset-version>/fits"
    )
    return parser


def _summary_lines(run: FitRun) -> list[str]:
    lines = [f"fit {run.inputs.role} {run.inputs.datasetVersion}: calls done, residency held"]
    for chosen in run.analysis.selected:
        sel = chosen.selection
        lines.append(
            f"{sel.candidate}: text {sel.textId!r}, rung {chosen.ladder.rung}, "
            f"noEligibleText={sel.noEligibleText}"
        )
    for effect in run.dispositions.effects:
        d = effect.disposition
        lines.append(f"excluded {d.callKind} call of {d.itemId} ({effect.effect}): {d.reason}")
    for fig in run.dispositions.byClass:
        lines.append(
            f"class {fig.label!r}: k={fig.k} kWrong={fig.kWrong} n_class={fig.nClass} "
            f"ceiling={fig.ceiling}"
        )
    return lines


def main(argv: Sequence[str] | None = None, *, client: Any = None) -> int:
    """Run a fit, write the calls file, the two candidates, the manifest and the report
    (`emit.emit`, all or nothing) and print the selections, dispositions and written paths. Exit 0,
    or 1 on a refusal (the calls file may remain) or an abort (one abort record). `client` is
    injected by tests; the default is `LMStudio(--lmstudio-url)`."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.reason is not None and len(args.reason) > 1:
        parser.error("--reason is given more than once")
    reason = args.reason[0] if args.reason else None
    if bool(args.exclude_item) != (reason is not None):
        parser.error("--exclude-item and --reason go together")
    try:
        url = validate_url(args.lmstudio_url)
        fits_dir = args.fits_dir or args.data_dir / args.role / args.dataset_version / "fits"
        run = run_fit(
            client if client is not None else LMStudio(url),
            role=args.role,
            dataset_version=args.dataset_version,
            decision_model=args.decision_model,
            handoff_model=args.handoff_model,
            decision_timeout_seconds=args.decision_timeout_seconds,
            fits_dir=fits_dir,
            data_dir=args.data_dir,
            exclusions=args.exclude_item,
            reason=reason,
        )
    except (FitRefused, ResidencyRefused, ExclusionRefused, ExclusionDidNotFire, OSError) as exc:
        print(f"refused: {_with_notes(exc)}", file=sys.stderr)
        return 1
    except FitAborted as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 1
    print("\n".join(_summary_lines(run)))
    try:
        _, written = emit(
            run, fits_dir, input_hashes=run.inputHashes, source_sha=run.sourceSha, report=True
        )
    except (EmitRefused, ValueError, OSError) as exc:  # an unemittable ladder, or a write failure
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print("\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
