"""`decisionmodels.exclusion` and the `--exclude-item` flag (plan4 §3.11 "Recorded dispositions",
§5 row U6f.8c; fit note §4 "Recorded item disposition"). Fakes only: abort records are written by
`fitcalls.write_abort_record`; no socket opens.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import fields, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import test_fit_cli as base_cli

from decisionmodels import exclusion, fit
from decisionmodels.exclusion import (
    ExclusionFlag,
    ExclusionRefused,
    apply_dispositions,
    build_dispositions,
    parse_exclude_item,
)
from decisionmodels.fitcalls import (
    AbortRecord,
    CollectedCalls,
    Disposition,
    ExclusionDidNotFire,
    FitInputs,
    summarize_calls,
    write_abort_record,
)
from decisionmodels.fitstats import CALIBRATION_FLOOR_PER_CELL, ScoredCall
from decisionmodels.lmstudio import (
    ChatResult,
    DecisionAnswer,
    DecisionResult,
    LMStudioCallFailed,
    ResidentModel,
)
from decisionmodels.questions import Question, load_questions

DECISION_MODEL = "dm/decider"
HANDOFF_MODEL = "dm/handoff"
BOTH = (ResidentModel(DECISION_MODEL, "loaded"), ResidentModel(HANDOFF_MODEL, "loaded"))
INPUTS = FitInputs("guard-judge", "v1", "a" * 64, "b" * 64, DECISION_MODEL, HANDOFF_MODEL, 30.0)
FLAG = ExclusionFlag("it-1", "handoff", None)
DECISION_FLAG = ExclusionFlag("it-1", "decision", "t1")


# --- the flag grammar ---


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("it-1:handoff", ExclusionFlag("it-1", "handoff", None)),
        ("it-1:decision:t1", ExclusionFlag("it-1", "decision", "t1")),
        ("it-1:decision:a:b", ExclusionFlag("it-1", "decision", "a:b")),
    ],
)
def test_the_flag_grammar_accepts_the_two_shapes(value: str, expected: ExclusionFlag) -> None:
    assert parse_exclude_item(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "it-1", ":handoff", "it-1:other", "it-1:decision", "it-1:decision:", "it-1:handoff:t1"],
)
def test_a_malformed_flag_refuses(value: str) -> None:
    with pytest.raises(ValueError, match="exclude-item"):
        parse_exclude_item(value)


# --- the record check (a)-(e), each clause refused by its own fixture ---


def record(**over: Any) -> AbortRecord:
    base = AbortRecord(
        schema="decision-models.fit-abort/1",
        runId="r1",
        itemId="it-1",
        callKind="handoff",
        textId=None,
        failureKind="http",
        status=500,
        elapsedSeconds=1.5,
        inputs=INPUTS,
        residencyStart=BOTH,
        residencyNow=BOTH,
    )
    return replace(base, **over)


def put(fits: Path, *records: AbortRecord, second: int = 0) -> list[Path]:
    out = []
    for n, rec in enumerate(records):
        stamp = datetime(2026, 1, 1, 0, 0, second + n, tzinfo=UTC)
        out.append(write_abort_record(rec, fits, now=lambda s=stamp: s))
    return out


def build(fits: Path, flag: ExclusionFlag = FLAG, reason: str | None = "oversized"):
    return build_dispositions([flag], reason, inputs=INPUTS, fits_dir=fits)


def test_two_matching_records_build_the_disposition_citing_both_file_names(tmp_path: Path) -> None:
    a, b = put(tmp_path, record(runId="r1"), record(runId="r2"))
    (d,) = build(tmp_path)
    assert d == Disposition("it-1", "handoff", None, "http", 500, d.reason, (a.name, b.name))
    assert "oversized" in d.reason
    assert a.name in d.reason
    assert b.name in d.reason


def test_a_decision_disposition_carries_the_text_and_malformed_has_no_status(
    tmp_path: Path,
) -> None:
    kw = dict(callKind="decision", textId="t1", failureKind="malformed", status=None)
    put(tmp_path, record(runId="r1", **kw), record(runId="r2", **kw))
    (d,) = build(tmp_path, DECISION_FLAG)
    assert d.key == ("it-1", "decision", "t1", "malformed", None)


def test_a_missing_fits_dir_or_no_reason_or_a_duplicate_flag_refuses(tmp_path: Path) -> None:
    with pytest.raises(ExclusionRefused, match="found 0"):
        build(tmp_path / "absent")
    put(tmp_path, record(runId="r1"), record(runId="r2"))
    with pytest.raises(ExclusionRefused, match="--reason"):
        build(tmp_path, reason=None)
    with pytest.raises(ExclusionRefused, match="--reason"):
        build(tmp_path, reason="  ")
    with pytest.raises(ExclusionRefused, match="twice"):
        build_dispositions([FLAG, FLAG], "x", inputs=INPUTS, fits_dir=tmp_path)


def test_clause_a_one_file_is_not_two(tmp_path: Path) -> None:
    put(tmp_path, record())
    with pytest.raises(ExclusionRefused, match="found 1"):
        build(tmp_path)


def test_clause_a_two_files_with_one_runid_refuse(tmp_path: Path) -> None:
    put(tmp_path, record(runId="same"), record(runId="same"))
    with pytest.raises(ExclusionRefused, match="same runId"):
        build(tmp_path)


@pytest.mark.parametrize(
    "other",
    [
        {"itemId": "it-2"},
        {"callKind": "decision", "textId": "t1"},
    ],
)
def test_clause_b_a_record_for_another_item_or_call_does_not_count(
    tmp_path: Path, other: dict[str, Any]
) -> None:
    put(tmp_path, record(runId="r1"), record(runId="r2", **other))
    with pytest.raises(ExclusionRefused, match="found 1"):
        build(tmp_path)


def test_clause_b_a_record_for_another_text_does_not_count(tmp_path: Path) -> None:
    kw = dict(callKind="decision")
    put(tmp_path, record(runId="r1", textId="t1", **kw), record(runId="r2", textId="t2", **kw))
    with pytest.raises(ExclusionRefused, match="found 1"):
        build(tmp_path, DECISION_FLAG)


def test_clause_b_two_http_records_with_statuses_500_and_503_are_not_pooled(
    tmp_path: Path,
) -> None:
    put(tmp_path, record(runId="r1", status=500), record(runId="r2", status=503))
    with pytest.raises(ExclusionRefused, match="disagree"):
        build(tmp_path)


def test_clause_b_records_with_different_failure_kinds_are_not_pooled(tmp_path: Path) -> None:
    put(
        tmp_path,
        record(runId="r1", failureKind="http", status=None),
        record(runId="r2", failureKind="malformed", status=None),
    )
    with pytest.raises(ExclusionRefused, match="disagree"):
        build(tmp_path)


@pytest.mark.parametrize("key", [f.name for f in fields(FitInputs)])
def test_clause_c_inputs_differing_in_any_one_of_the_seven_keys_refuse(
    tmp_path: Path, key: str
) -> None:
    current = getattr(INPUTS, key)
    changed = current + 1 if isinstance(current, float) else f"{current}-other"
    stale = replace(INPUTS, **{key: changed})
    put(tmp_path, record(runId="r1"), record(runId="r2", inputs=stale))
    with pytest.raises(ExclusionRefused, match=key):
        build(tmp_path)


def test_there_are_exactly_seven_input_keys() -> None:
    assert len(fields(FitInputs)) == 7


@pytest.mark.parametrize("kind", ["timeout", "connection"])
def test_clause_d_timeout_and_connection_are_never_evidence(tmp_path: Path, kind: str) -> None:
    put(
        tmp_path,
        record(runId="r1", failureKind=kind, status=None),
        record(runId="r2", failureKind=kind, status=None),
    )
    with pytest.raises(ExclusionRefused, match="not evidence"):
        build(tmp_path)


def test_the_evidence_kinds_are_exactly_http_and_malformed() -> None:
    assert exclusion.EVIDENCE_KINDS == ("http", "malformed")


@pytest.mark.parametrize(
    "resident",
    [None, (ResidentModel(DECISION_MODEL, "loaded"),), (ResidentModel(HANDOFF_MODEL, "loaded"),)],
)
def test_clause_e_a_record_without_both_models_resident_refuses(
    tmp_path: Path, resident: Any
) -> None:
    put(tmp_path, record(runId="r1"), record(runId="r2", residencyNow=resident))
    with pytest.raises(ExclusionRefused, match="resident|residencyNow"):
        build(tmp_path)


def test_a_third_good_record_can_stand_in_for_a_bad_one(tmp_path: Path) -> None:
    stale = replace(INPUTS, questionsSha256="c" * 64)
    a, _, c = put(
        tmp_path, record(runId="r1"), record(runId="r2", inputs=stale), record(runId="r3")
    )
    (d,) = build(tmp_path)
    assert d.recordFiles == (a.name, c.name)


def test_a_garbage_file_is_skipped_and_named_in_the_refusal(tmp_path: Path) -> None:
    (tmp_path / "abort-20260101T000000Z-zz.json").write_text("{not json")
    put(tmp_path, record(), second=5)
    with pytest.raises(ExclusionRefused, match="abort-20260101T000000Z-zz.json"):
        build(tmp_path)


def test_a_stray_json_in_the_fits_dir_is_not_an_abort_record(tmp_path: Path) -> None:
    (a,) = put(tmp_path, record(runId="r1"))
    (tmp_path / "copy.json").write_text(a.read_text())  # a valid record under a non-abort name
    with pytest.raises(ExclusionRefused, match="found 1"):
        build(tmp_path)


def test_an_abort_file_of_another_schema_is_skipped_and_named_in_the_refusal(
    tmp_path: Path,
) -> None:
    paths = put(tmp_path, record(runId="r1"), record(runId="r2"))
    data = json.loads(paths[1].read_text())
    data["schema"] = "decision-models.fit-abort/2"
    paths[1].write_text(json.dumps(data))
    with pytest.raises(ExclusionRefused, match=r"found 1.*not an abort record.*abort/2"):
        build(tmp_path)


@pytest.mark.parametrize("status", ["500", True, 500.0])
def test_a_status_that_is_not_an_int_or_null_makes_the_file_not_a_record(
    tmp_path: Path, status: Any
) -> None:
    paths = put(tmp_path, record(runId="r1"), record(runId="r2"))
    data = json.loads(paths[1].read_text())
    data["status"] = status
    paths[1].write_text(json.dumps(data))
    with pytest.raises(ExclusionRefused, match="found 1"):
        build(tmp_path)


# --- the effects, per role and call kind ---


def ok(item: str, kind: str, text: str | None, answer: Any, secs: float = 1.0) -> ScoredCall:
    return ScoredCall(item, kind, text, "ok", answer, secs)


def bad(item: str, kind: str, text: str | None, secs: float = 2.0) -> ScoredCall:
    return ScoredCall(item, kind, text, "failed", None, secs)


def disposition(item: str, kind: str, text: str | None = None) -> Disposition:
    return Disposition(item, kind, text, "http", 500, "why", ("a.json", "b.json"))


def texts(*ids: str) -> list[Question]:
    return [Question(i, n == 0, {"type": "predicate", "name": "q"}) for n, i in enumerate(ids)]


def guard_data(per_cell: int, texts_: Sequence[str] = ("ctl", "t1", "t2")):
    names = ["cA", "cB", "cC", "cD"]
    labels, cells, calls = {}, {}, []
    for c, name in enumerate(names):
        for n in range(per_cell):
            item = f"{name}-{n}"
            labels[item] = c % 2 == 0
            cells[item] = name
            calls.append(ok(item, "handoff", None, c % 2 == 0))
            calls += [ok(item, "decision", t, 0.5) for t in texts_]
    return labels, cells, calls


def swap(calls: list[ScoredCall], item: str, kind: str, text: str | None) -> list[ScoredCall]:
    return [bad(item, kind, text) if (c.itemId, c.callKind, c.textId) == (item, kind, text) else c
            for c in calls]  # fmt: skip


def collected(calls: list[ScoredCall], mp: dict | None = None) -> CollectedCalls:
    return CollectedCalls(tuple(calls), 0, mp or {})


def test_guard_handoff_exclusion_removes_the_item_from_labels_cells_and_every_call() -> None:
    labels, cells, calls = guard_data(52)
    calls = swap(calls, "cA-0", "handoff", None)
    mp = {("cA-0", "t1"): 0.5, ("cA-1", "t1"): 0.6}
    applied = apply_dispositions(
        "guard-judge",
        collected(calls, mp),
        [disposition("cA-0", "handoff")],
        labels=labels,
        cells=cells,
        texts=texts("ctl", "t1", "t2"),
    )
    assert "cA-0" not in applied.labels
    assert "cA-0" not in applied.cells
    assert all(c.itemId != "cA-0" for c in applied.collected.calls)
    assert len(applied.collected.calls) == len(calls) - 4  # its hand-off and three decisions
    assert dict(applied.collected.maxProbabilities) == {("cA-1", "t1"): 0.6}
    assert applied.report.cellCounts == {"cA": 51, "cB": 52, "cC": 52, "cD": 52}
    assert [e.effect for e in applied.report.effects] == ["item-removed"]
    assert applied.report.byClass == ()


def test_the_floor_boundary_fifty_passes_and_forty_nine_refuses() -> None:
    for per_cell, passes in ((51, True), (50, False)):
        labels, cells, calls = guard_data(per_cell)
        calls = swap(calls, "cA-0", "handoff", None)
        args = ("guard-judge", collected(calls), [disposition("cA-0", "handoff")])
        kw = dict(labels=labels, cells=cells, texts=texts("ctl", "t1", "t2"))
        if passes:
            applied = apply_dispositions(*args, **kw)
            assert applied.report.cellCounts["cA"] == CALIBRATION_FLOOR_PER_CELL
        else:
            with pytest.raises(ExclusionRefused, match="cell 'cA' with 49 items"):
                apply_dispositions(*args, **kw)


def test_the_floor_is_not_rechecked_without_a_handoff_exclusion() -> None:
    labels, cells, calls = guard_data(10)  # far under the floor, and nothing excluded
    calls = swap(calls, "cA-0", "decision", "t1")
    applied = apply_dispositions(
        "guard-judge",
        collected(calls),
        [disposition("cA-0", "decision", "t1")],
        labels=labels,
        cells=cells,
        texts=texts("ctl", "t1", "t2"),
    )
    assert len(applied.labels) == 40
    assert applied.collected.calls == tuple(calls)


@pytest.mark.parametrize(
    ("failing", "effect"),
    [
        (("t1",), "text-left-selection"),
        (("ctl",), "concordant"),
        (("ctl", "t1", "t2"), "concordant"),
        (("t1", "t2", "ctl"), "concordant"),
        (("t1", "ctl"), "text-left-selection"),
    ],
)
def test_guard_decision_exclusion_keeps_the_item_and_labels_its_effect(
    failing: tuple[str, ...], effect: str
) -> None:
    labels, cells, calls = guard_data(3)
    disps = []
    for text in failing:
        calls = swap(calls, "cA-0", "decision", text)
        disps.append(disposition("cA-0", "decision", text))
    applied = apply_dispositions(
        "guard-judge",
        collected(calls),
        disps,
        labels=labels,
        cells=cells,
        texts=texts("ctl", "t1", "t2"),
    )
    assert applied.labels == labels
    assert applied.cells == cells
    assert applied.collected.calls == tuple(calls)
    # the effect of the disposition on `failing[0]`
    assert applied.report.effects[0].effect == effect


def test_a_decision_failure_on_an_item_whose_handoff_is_excluded_is_removed_with_it() -> None:
    labels, cells, calls = guard_data(52)
    calls = swap(swap(calls, "cA-0", "handoff", None), "cA-0", "decision", "t1")
    applied = apply_dispositions(
        "guard-judge",
        collected(calls),
        [disposition("cA-0", "handoff"), disposition("cA-0", "decision", "t1")],
        labels=labels,
        cells=cells,
        texts=texts("ctl", "t1", "t2"),
    )
    assert [e.effect for e in applied.report.effects] == ["item-removed", "item-removed"]


def new_ability_data(role_labels: list[Any]):
    labels = {f"i{n}": lab for n, lab in enumerate(role_labels)}
    calls = []
    for item, lab in labels.items():
        calls.append(ok(item, "handoff", None, lab))
        calls.append(ok(item, "decision", "t1", 0.5))
    return labels, calls


@pytest.mark.parametrize("role", ["join-in-judge", "message-type-classifier"])
@pytest.mark.parametrize("kind", ["handoff", "decision"])
def test_new_ability_failures_keep_the_item_fail_closed_with_per_class_figures(
    role: str, kind: str
) -> None:
    classes = [True, True, True, True, False, False]
    if role == "message-type-classifier":
        classes = ["a", "a", "a", "a", "b", "b"]
    labels, calls = new_ability_data(classes)
    text = None if kind == "handoff" else "t1"
    calls = swap(calls, "i0", kind, text)
    applied = apply_dispositions(
        role,
        collected(calls),
        [disposition("i0", kind, text)],
        labels=labels,
        cells=None,
        texts=texts("t1"),
    )
    assert applied.labels == labels
    assert applied.cells is None
    assert applied.collected.calls == tuple(calls)  # nothing pruned
    assert [e.effect for e in applied.report.effects] == ["fail-closed"]
    first, second = classes[0], classes[-1]
    assert [(f.label, f.k, f.kWrong, f.nClass, f.ceiling) for f in applied.report.byClass] == [
        (first, 1, 1, 4, Fraction(3, 4)),
        (second, 0, 0, 2, Fraction(1)),
    ]
    assert applied.report.cellCounts is None


def test_per_class_k_counts_an_item_once_across_its_dispositions() -> None:
    labels, calls = new_ability_data([True, True, True, True, False])
    calls = swap(swap(calls, "i0", "handoff", None), "i0", "decision", "t1")
    calls = swap(calls, "i1", "decision", "t1")
    applied = apply_dispositions(
        "join-in-judge",
        collected(calls),
        [disposition("i0", "handoff"), disposition("i0", "decision", "t1"),
         disposition("i1", "decision", "t1")],
        labels=labels,
        cells=None,
        texts=texts("t1"),
    )  # fmt: skip
    true_class = applied.report.byClass[0]
    assert (true_class.k, true_class.kWrong, true_class.nClass, true_class.ceiling) == (
        2,
        2,
        4,
        Fraction(1, 2),
    )


def test_a_join_in_label_no_item_is_fail_closed_right_and_a_message_type_item_always_wrong() -> (
    None
):
    labels, calls = new_ability_data([True, True, False, False])
    calls = swap(calls, "i2", "decision", "t1")  # label False: the fail-closed "no" is right
    applied = apply_dispositions(
        "join-in-judge",
        collected(calls),
        [disposition("i2", "decision", "t1")],
        labels=labels,
        cells=None,
        texts=texts("t1"),
    )
    no_class = applied.report.byClass[1]
    assert (no_class.label, no_class.k, no_class.kWrong, no_class.ceiling) == (
        False,
        1,
        0,
        Fraction(1),
    )
    labels, calls = new_ability_data(["a", "a", "b", "b"])
    calls = swap(calls, "i2", "decision", "t1")
    applied = apply_dispositions(
        "message-type-classifier",
        collected(calls),
        [disposition("i2", "decision", "t1")],
        labels=labels,
        cells=None,
        texts=texts("t1"),
    )
    b_class = applied.report.byClass[1]
    assert (b_class.k, b_class.kWrong, b_class.ceiling) == (1, 1, Fraction(1, 2))


@pytest.mark.parametrize(
    "failing",
    [
        [("a-0", "t1")],
        [("a-0", "ctl")],
        [("a-0", "ctl"), ("a-0", "t1"), ("a-0", "t2")],
        [("a-0", "t1"), ("a-1", "t2"), ("a-1", "ctl")],
        [("a-0", "t1"), ("a-0", "ctl")],
    ],
)
def test_guard_effects_agree_with_the_selection_rule_in_fit(failing: list[tuple[str, str]]) -> None:
    """`exclusion` cannot import `fit` (cycle), so the rule is restated there; this pins that the
    texts labelled `text-left-selection` are exactly `fit._decision_failure_texts`."""
    labels = {f"a-{n}": True for n in range(3)}
    cells = dict.fromkeys(labels, "cA")
    calls = [ok(i, "handoff", None, True) for i in labels]
    calls += [ok(i, "decision", t, 0.5) for i in labels for t in ("ctl", "t1", "t2")]
    for item, text in failing:
        calls = swap(calls, item, "decision", text)
    the_texts = texts("ctl", "t1", "t2")
    applied = apply_dispositions(
        "guard-judge",
        collected(calls),
        [disposition(i, "decision", t) for i, t in failing],
        labels=labels,
        cells=cells,
        texts=the_texts,
    )
    left = {
        e.disposition.textId for e in applied.report.effects if e.effect == "text-left-selection"
    }
    assert left == fit._decision_failure_texts("guard-judge", applied.collected.calls, the_texts)


def test_a_disposition_whose_call_succeeded_is_not_applied_and_refuses() -> None:
    labels, cells, calls = guard_data(52)  # the hand-off of cA-0 is `ok`
    for call_outcome in ("ok", "over-limit"):
        edited = [
            replace(c, outcome=call_outcome) if (c.itemId, c.callKind) == ("cA-0", "handoff") else c
            for c in calls
        ]
        with pytest.raises(ExclusionDidNotFire):
            apply_dispositions(
                "guard-judge",
                collected(edited),
                [disposition("cA-0", "handoff")],
                labels=labels,
                cells=cells,
                texts=texts("ctl", "t1", "t2"),
            )


def test_a_disposition_whose_call_was_never_made_refuses() -> None:
    labels, cells, calls = guard_data(52)
    with pytest.raises(ExclusionDidNotFire, match="never made"):
        apply_dispositions(
            "guard-judge",
            collected(calls),
            [disposition("ghost", "handoff")],
            labels=labels,
            cells=cells,
            texts=texts("ctl", "t1", "t2"),
        )


# --- the run: flag, records, summaries over the full calls ---

ROLE = "guard-judge"
VERSION = "v9"
URL = "http://localhost:1234"
CTL = next(q.id for q in load_questions(ROLE) if q.control)


def write_guard_data(root: Path, n_items: int) -> Path:
    data = root / "data"
    target = data / ROLE / VERSION
    target.mkdir(parents=True)
    rows, ids = [], []
    for n in range(1, n_items + 1):
        item_id = f"x-{n:04d}"
        ids.append(item_id)
        rows.append(
            {
                "itemId": item_id,
                "condition": f"condition {n}",
                "understanding": {},
                "turns": [],
                "path": "turns" if n % 4 < 2 else "understanding",
                "expected": n % 2 == 0,
            }
        )
    (target / "items.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    splits = {"rngSeed": 1, "eval": [], "train": [], "calibration": ids}
    (target / "splits.json").write_text(json.dumps(splits))
    return data


class Client:
    """Every call right; `fail_chat` and `fail_decide` raise an http 500 for that item (text)."""

    def __init__(self, data: Path) -> None:
        cal = fit.load_calibration(ROLE, VERSION, data)
        self.by_input = {c.decisionInput: c.itemId for c in cal.items}
        self.labels = cal.labels
        self.text_of = {q.spec["instructions"]: q.id for q in load_questions(ROLE)}
        self.fail_chat: set[str] = set()
        self.fail_decide: set[tuple[str, str]] = set()
        self.touched = 0  # every residency read, chat and decide call

    def residency(self) -> list[ResidentModel]:
        self.touched += 1
        return list(BOTH)

    def chat(self, messages, *, model, temperature, max_tokens, timeout_s):  # type: ignore[no-untyped-def]
        self.touched += 1
        item = self.by_input[messages[-1]["content"]]
        if item in self.fail_chat:
            raise LMStudioCallFailed("boom", status=500, body=None, kind="http", wallClockMs=None)
        return ChatResult(json.dumps({"decision": self.labels[item]}), None, None, 300.0)

    def decide(self, input, questions, *, model, timeout_s):  # type: ignore[no-untyped-def]
        self.touched += 1
        item = self.by_input[input]
        text = self.text_of[questions[0]["instructions"]]
        if (item, text) in self.fail_decide:
            raise LMStudioCallFailed("boom", status=500, body=None, kind="http", wallClockMs=None)
        p = 0.9 if self.labels[item] else 0.1
        answer = DecisionAnswer("q", "predicate", p, None, None, None)
        return DecisionResult((answer,), None, None, 120.0)


def argv(data: Path, *extra: str) -> list[str]:
    base = [
        "--role", ROLE,
        "--dataset-version", VERSION,
        "--decision-model", DECISION_MODEL,
        "--handoff-model", HANDOFF_MODEL,
        "--lmstudio-url", URL,
        "--decision-timeout-seconds", "45.5",
        "--data-dir", str(data),
    ]  # fmt: skip
    return [*base, *extra]


def fits(data: Path) -> Path:
    return data / ROLE / VERSION / "fits"


def abort_files(data: Path) -> list[Path]:
    return sorted(fits(data).glob("abort-*.json")) if fits(data).exists() else []


def two_aborts(data: Path, client: Client) -> None:
    """Two failed runs of one item's hand-off: two abort records with distinct runIds."""
    for _ in range(2):
        assert fit.main(argv(data), client=client) == 1
    assert len(abort_files(data)) == 2


def run(data: Path, client: Client, *extra: str) -> int:
    return fit.main(argv(data, *extra), client=client)


def test_a_rerun_with_the_recorded_exclusion_completes_and_summaries_count_the_full_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_guard_data(tmp_path, 204)  # 51 per cell: one removal leaves 50
    client = Client(data)
    client.fail_chat = {"x-0001"}
    two_aborts(data, client)
    ids = {json.loads(p.read_text())["runId"] for p in abort_files(data)}
    assert len(ids) == 2
    capsys.readouterr()
    code = run(data, client, "--exclude-item", "x-0001:handoff", "--reason", "oversized input")
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "excluded handoff call of x-0001 (item-removed): oversized input [records: " in out.out
    assert len(abort_files(data)) == 2  # no third record
    run_ = fit.run_fit(
        client,
        role=ROLE,
        dataset_version=VERSION,
        decision_model=DECISION_MODEL,
        handoff_model=HANDOFF_MODEL,
        decision_timeout_seconds=45.5,
        fits_dir=fits(data),
        data_dir=data,
        exclusions=[FLAG_X1],
        reason="oversized input",
    )
    assert run_.summaries["handoff"].n == 204  # the full calls, not the 203 kept
    assert run_.summaries["handoff"].k == 1
    assert run_.summaries["handoff"].dispositionItemIds == ("x-0001",)
    assert run_.dispositions.cellCounts is not None
    for score in run_.analysis.scores:  # both halves of every text's split-half skip the item
        halves = [i for shift in score.splitHalf for i in shift.fitIds + shift.scoredIds]
        assert "x-0001" not in halves
        assert len(set(halves)) == 203
    assert min(run_.dispositions.cellCounts.values()) == CALIBRATION_FLOOR_PER_CELL
    (effect,) = run_.dispositions.effects
    assert effect.effect == "item-removed"
    assert len(effect.disposition.recordFiles) == 2
    assert summarize_calls(run_.collected.calls, decision_timeout_seconds=45.5) == {
        **run_.summaries,
        "handoff": replace(run_.summaries["handoff"], parseFailures=0),
    }


FLAG_X1 = ExclusionFlag("x-0001", "handoff", None)


def test_a_rerun_at_the_floor_refuses_writing_nothing(tmp_path: Path) -> None:
    data = write_guard_data(tmp_path, 200)  # 50 per cell: one removal leaves 49
    client = Client(data)
    client.fail_chat = {"x-0001"}
    two_aborts(data, client)
    before = {p.name for p in fits(data).iterdir()}
    client.touched = 0
    assert run(data, client, "--exclude-item", "x-0001:handoff", "--reason", "r") == 1
    assert {p.name for p in fits(data).iterdir()} == before
    assert client.touched == 0  # the floor refuses before the residency read and any call


def test_an_excluded_call_that_succeeds_in_the_rerun_refuses_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_guard_data(tmp_path, 204)
    client = Client(data)
    client.fail_chat = {"x-0001"}
    two_aborts(data, client)
    client.fail_chat = set()  # the failure was transient after all
    before = {p.name for p in fits(data).iterdir()}
    capsys.readouterr()
    assert run(data, client, "--exclude-item", "x-0001:handoff", "--reason", "r") == 1
    assert "did not fire" in capsys.readouterr().err
    assert {p.name for p in fits(data).iterdir()} == before


def test_a_different_call_failing_in_the_rerun_still_aborts_with_a_fresh_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_guard_data(tmp_path, 204)
    client = Client(data)
    client.fail_chat = {"x-0001"}
    two_aborts(data, client)
    client.fail_decide = {("x-0002", CTL)}
    capsys.readouterr()
    assert run(data, client, "--exclude-item", "x-0001:handoff", "--reason", "r") == 1
    assert "aborted" in capsys.readouterr().err
    files = abort_files(data)
    assert len(files) == 3
    records = [json.loads(p.read_text()) for p in files]
    assert len({r["runId"] for r in records}) == 3
    (fresh,) = [r for r in records if r["itemId"] == "x-0002"]
    assert (fresh["callKind"], fresh["textId"]) == ("decision", CTL)


def test_a_flag_without_two_records_refuses_before_any_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_guard_data(tmp_path, 204)
    client = Client(data)
    assert run(data, client, "--exclude-item", "x-0001:handoff", "--reason", "r") == 1
    assert "found 0" in capsys.readouterr().err
    assert client.touched == 0  # the record check runs before the residency read and any call
    assert not fits(data).exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--exclude-item", "x-0001:handoff"],  # no --reason
        ["--reason", "r"],  # a reason with nothing to exclude
        ["--exclude-item", "x-0001:decision", "--reason", "r"],  # decision needs a textId
        ["--exclude-item", "x-0001:handoff:t", "--reason", "r"],  # handoff takes no textId
        ["--exclude-item", "x-0001", "--reason", "r"],
    ],
)
def test_the_flag_grammar_refuses_at_parse_time_and_calls_nothing(
    tmp_path: Path, extra: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_guard_data(tmp_path, 8)
    client = Client(data)
    with pytest.raises(SystemExit) as stop:
        run(data, client, *extra)
    assert stop.value.code == 2
    assert "exclude-item" in capsys.readouterr().err
    assert client.touched == 0
    assert not fits(data).exists()


def test_a_repeated_reason_refuses_at_parse_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_guard_data(tmp_path, 8)
    client = Client(data)
    with pytest.raises(SystemExit) as stop:
        run(data, client, "--exclude-item", "x-0001:handoff", "--reason", "a", "--reason", "b")
    assert stop.value.code == 2
    assert "more than once" in capsys.readouterr().err
    assert client.touched == 0


# --- new abilities at the run level ---


class NewAbilityClient(base_cli.FakeClient):
    """The fit CLI's scripted client, plus an http 500 on the hand-off of `fail_chat` items."""

    def __init__(self, role: str, data: Path) -> None:
        super().__init__(role, data)
        self.fail_chat: set[str] = set()

    def chat(self, messages, *, model, temperature, max_tokens, timeout_s):  # type: ignore[no-untyped-def]
        if self.by_input[messages[-1]["content"]] in self.fail_chat:
            raise LMStudioCallFailed("boom", status=500, body=None, kind="http", wallClockMs=None)
        return super().chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )


@pytest.mark.parametrize(
    ("role", "failing", "expected"),
    [
        # join-in labels: x-0001 False, x-0002 True, ... (n even is True); 4 items per class
        (
            "join-in-judge",
            ["x-0001", "x-0002"],
            {False: (1, 0, 4, Fraction(1)), True: (1, 1, 4, Fraction(3, 4))},
        ),
        # message-type class is expected[n % 5]: x-0001 shares its class with x-0006 (2 items)
        ("message-type-classifier", ["x-0001"], {"item-1-class": (1, 1, 2, Fraction(1, 2))}),
    ],
)
def test_a_new_ability_run_keeps_the_item_fail_closed_and_reports_per_class_figures(
    tmp_path: Path, role: str, failing: list[str], expected: dict[Any, Any]
) -> None:
    data = base_cli.write_data(tmp_path, role)
    client = NewAbilityClient(role, data)
    client.fail_chat = set(failing)
    fits_dir = data / role / base_cli.VERSION / "fits"
    kw = dict(
        role=role,
        dataset_version=base_cli.VERSION,
        decision_model=DECISION_MODEL,
        handoff_model=HANDOFF_MODEL,
        decision_timeout_seconds=base_cli.TIMEOUT,
        fits_dir=fits_dir,
        data_dir=data,
    )
    for _ in range(2):  # each run aborts on the first failing hand-off, leaving one record
        with pytest.raises(fit.FitAborted):
            fit.run_fit(client, **kw)
    # two records exist only for the first failing item; add the rest by aborting on each
    for item in failing[1:]:
        client.fail_chat = {item}
        for _ in range(2):
            with pytest.raises(fit.FitAborted):
                fit.run_fit(client, **kw)
    client.fail_chat = set(failing)
    flags = [ExclusionFlag(i, "handoff", None) for i in failing]
    run_ = fit.run_fit(client, **kw, exclusions=flags, reason="oversized")
    assert [e.effect for e in run_.dispositions.effects] == ["fail-closed"] * len(failing)
    assert run_.summaries["handoff"].n == 8
    assert run_.summaries["handoff"].k == len(failing)
    got = {f.label: (f.k, f.kWrong, f.nClass, f.ceiling) for f in run_.dispositions.byClass}
    # message type keys its one disposed class by the label of item n=1
    want = {
        base_cli.label_of(role, 1) if label == "item-1-class" else label: v
        for label, v in expected.items()
    }
    assert {k: got[k] for k in want} == want
    for score in run_.analysis.scores:  # the items stay in the analysis
        shift = score.splitHalf[0]
        assert len(shift.fitIds + shift.scoredIds) == 8


def test_a_decision_only_exclusion_at_exactly_the_floor_completes(tmp_path: Path) -> None:
    """No hand-off is excluded, so the pre-call floor check must not count the item as removed."""
    data = write_guard_data(tmp_path, 200)  # exactly 50 per cell
    client = Client(data)
    text = next(q.id for q in load_questions(ROLE) if not q.control)
    client.fail_decide = {("x-0001", text)}
    two_aborts(data, client)
    run_ = fit.run_fit(
        client,
        role=ROLE,
        dataset_version=VERSION,
        decision_model=DECISION_MODEL,
        handoff_model=HANDOFF_MODEL,
        decision_timeout_seconds=45.5,
        fits_dir=fits(data),
        data_dir=data,
        exclusions=[ExclusionFlag("x-0001", "decision", text)],
        reason="r",
    )
    assert run_.dispositions.cellCounts is not None
    assert set(run_.dispositions.cellCounts.values()) == {50}
    assert [e.effect for e in run_.dispositions.effects] == ["text-left-selection"]
