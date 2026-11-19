"""U6f.9b, second half: the fit report (`docs/plans/typed-decision-models4.md` §3.11 "Fit report",
§5 row U6f.9b). Hand-built runs for the literal figures; fake-client runs from `test_fit_cli.py` for
the shape on all three roles."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import test_emit as base_emit
import test_fit as base_fit
import test_fit_cli as base_cli

from decisionmodels import emit, fit
from decisionmodels.exclusion import ClassFigures, DispositionEffect, DispositionReport
from decisionmodels.fitcalls import Disposition, FitInputs, summarize_calls
from decisionmodels.lmstudio import ResidentModel
from decisionmodels.questions import Question
from decisionmodels.report import REPORT_SCHEMA, build_report, jsonable

ROLES = ["guard-judge", "join-in-judge", "message-type-classifier"]
TOP_KEYS = {
    "schema", "role", "datasetVersion", "fitOrdinal", "callsFile", "decisionModel",
    "handoffModel", "decisionTimeoutSeconds", "questionsSha256", "handoffSha256", "residency",
    "calls", "dispositions", "candidates",
}  # fmt: skip
TEXT_KEYS = {
    "textId", "control", "auc", "aucUndefined", "maxProbAuc", "maxProbDropped", "overLimit",
    "leftSelection", "splitHalf", "ladder",
}  # fmt: skip
LADDER_KEYS = {
    "family", "rung", "qualityTarget", "policy", "key", "agreementUnmet", "latencyUnmet",
    "barValues",
}  # fmt: skip
HALF_KEYS = {"direction", "fittedPolicy", "fittedRung", "onFit", "onScored", "shift"}
SUMMARY_KEYS = {
    "callKind", "n", "k", "dispositionItemIds", "medianSeconds", "maxSeconds", "wilsonUpper95",
    "halfBudgetCount", "parseFailures",
}  # fmt: skip
F = Fraction


def frac(n: int, d: int) -> dict[str, Any]:
    return {"fraction": f"{n}/{d}", "decimal": n / d}


# --- serialization ---


def test_a_fraction_is_an_object_with_the_exact_string_and_the_decimal_everywhere() -> None:
    assert jsonable(F(3, 4)) == {"fraction": "3/4", "decimal": 0.75}
    assert jsonable(F(1)) == {"fraction": "1/1", "decimal": 1.0}
    assert jsonable(F(-1, 3)) == {"fraction": "-1/3", "decimal": float(F(-1, 3))}
    nested = jsonable({"k": (F(1, 2), [F(2, 5)]), True: None, False: 0})
    assert nested == {"k": [frac(1, 2), [frac(2, 5)]], "true": None, "false": 0}


def test_a_dataclass_is_its_fields_and_a_none_stays_none() -> None:
    inputs = FitInputs("r", "v", "q", "h", "d", "m", 2.5)
    assert jsonable(inputs)["decisionTimeoutSeconds"] == 2.5
    assert jsonable({"a": None})["a"] is None


# --- hand-built runs ---


def texts() -> tuple[Question, ...]:
    return (base_fit.q("ctl", True), base_fit.q("alt"))


def hand_run(
    role: str, calls, labels, cells, dispositions=DispositionReport((), (), None), end=None
):
    collected = base_fit.collected(calls)
    analysis = fit.analyze_fit(role, collected, labels=labels, cells=cells, texts=texts())
    inputs = FitInputs(role, "v1", "a" * 64, "b" * 64, "dm/d", "dm/h", 3.0)
    snapshot = (ResidentModel("dm/d", "loaded"), ResidentModel("dm/h", "loaded"))
    return fit.FitRun(
        inputs,
        texts(),
        snapshot,
        snapshot if end is None else end,
        collected,
        analysis,
        summarize_calls(collected.calls, decision_timeout_seconds=3.0),
        dispositions,
        None,
        tuple(hashlib.sha256(i.encode()).hexdigest() for i in labels),
        "c" * 64,
    )


def stuck_guard_run():
    """Guard: item `s1` (label no) scores p = 1.0 on every text, the LLM right on it: stuck."""
    labels = {**base_fit.LABELS, "s1": False}
    cells = {**base_fit.CELLS, "s1": "falseAdvance"}
    calls = base_fit.guard_calls(
        {t: {**{i: p for i, (_, p) in base_fit.GUARD_P.items()}, "s1": 1.0} for t in ("ctl", "alt")}
    )
    calls.append(base_fit.hand("s1", False))
    return hand_run("guard-judge", calls, labels, cells)


def stuck_join_in_run():
    labels = {"a": True, "b": False, "c": False}
    calls = [base_fit.hand(i, lab) for i, lab in labels.items()]
    for t in ("ctl", "alt"):
        calls += [base_fit.dec("a", t, 0.9), base_fit.dec("b", t, 0.1), base_fit.dec("c", t, 1.0)]
    return hand_run("join-in-judge", calls, labels, None)


def report_of(run) -> dict[str, Any]:
    ids = {"handoff": "H-id", "decision-only": "O-id"}
    kind = emit.BAR_KIND_BY_ROLE[run.inputs.role]
    return json.loads(json.dumps(build_report(run, 1, ids, kind)))


def ladders(report: dict[str, Any]):
    for family in ("handoff", "decision-only"):
        for text in report["candidates"][family]["texts"]:
            yield family, text["textId"], text["ladder"]


def test_the_guard_report_carries_the_stuck_count_literal_never_zero() -> None:
    found = list(ladders(report_of(stuck_guard_run())))
    assert len(found) == 4
    assert [lad["barValues"]["stuckCount"] for _, _, lad in found] == [1, 1, 1, 1]


def test_the_join_in_report_carries_the_unfixable_by_class_literal_never_all_zero() -> None:
    found = list(ladders(report_of(stuck_join_in_run())))
    assert len(found) == 4
    for _, _, lad in found:
        assert lad["barValues"]["unfixableByClass"] == {"true": 0, "false": 1}
        assert lad["barValues"]["nClass"] == {"true": 1, "false": 2}


def test_bar_values_fractions_are_serialized_exactly_and_agree_with_the_ladder() -> None:
    run = stuck_join_in_run()
    chosen = run.analysis.selected[0]
    report = report_of(run)
    by_id = {t["textId"]: t for t in report["candidates"]["handoff"]["texts"]}
    text = by_id[chosen.text.id]
    values = text["ladder"]["barValues"]
    assert values["s"] == jsonable(chosen.ladder.barValues["s"])
    assert values["agreement"] == jsonable(chosen.ladder.barValues["agreement"])
    assert values["tLH_ms"]["fraction"].count("/") == 1
    assert set(values["s"]) == {"fraction", "decimal"}


def test_a_cell_absent_from_a_half_is_null_with_the_halfs_n_missing_it_not_zero() -> None:
    labels = dict(base_fit.LABELS)
    cells = {**base_fit.CELLS, "i1": "lonely"}  # i1 is alone in its cell; sorted first: half one
    scores = {i: p for i, (_, p) in base_fit.GUARD_P.items()}
    calls = base_fit.guard_calls({t: scores for t in ("ctl", "alt")})
    report = report_of(hand_run("guard-judge", calls, labels, cells))
    halves = report["candidates"]["handoff"]["texts"][0]["splitHalf"]
    assert {h["direction"] for h in halves} == {"first->second", "second->first"}
    for half in halves:
        assert set(half) == HALF_KEYS
        assert half["shift"]["D_c"]["lonely"] is None
    first = next(h for h in halves if h["direction"] == "first->second")
    assert "lonely" in first["onFit"]["n"] and "lonely" not in first["onScored"]["n"]
    assert first["shift"]["D_c"]["falseSuspend"] is not None  # a present cell keeps its value


def test_the_report_has_every_text_field_and_the_chosen_text_and_rung() -> None:
    run = stuck_join_in_run()
    report = report_of(run)
    assert set(report) == TOP_KEYS and report["schema"] == REPORT_SCHEMA
    for family, chosen in zip(("handoff", "decision-only"), run.analysis.selected, strict=True):
        block = report["candidates"][family]
        assert block["chosen"]["textId"] == chosen.selection.textId
        assert block["chosen"]["rung"] == chosen.ladder.rung
        assert set(block["chosen"]) == {
            "textId", "rung", "qualityTarget", "outcome", "noEligibleText",
        }  # fmt: skip
        assert [t["textId"] for t in block["texts"]] == ["ctl", "alt"]
        for text in block["texts"]:
            assert set(text) == TEXT_KEYS and set(text["ladder"]) == LADDER_KEYS
            assert set(text["overLimit"]) == {"k", "n"} and len(text["splitHalf"]) == 2
        assert block["texts"][0]["control"] is True and block["texts"][1]["control"] is False


def test_the_undefined_auc_and_the_dropped_max_prob_are_reported() -> None:
    run = stuck_join_in_run()
    texts_ = report_of(run)["candidates"]["handoff"]["texts"]
    assert all(t["aucUndefined"] is False for t in texts_)
    assert all(t["maxProbDropped"] is None for t in texts_)  # a predicate point has none
    assert all(t["maxProbAuc"] is None for t in texts_)
    only_one_class = hand_run(
        "join-in-judge",
        [base_fit.hand("a", True), base_fit.hand("b", True)]
        + [base_fit.dec(i, t, 0.5) for i in "ab" for t in ("ctl", "alt")],
        {"a": True, "b": True},
        None,
    )
    undefined = report_of(only_one_class)["candidates"]["handoff"]["texts"]
    assert all(t["aucUndefined"] for t in undefined)


def test_the_call_summaries_residency_hashes_and_timeout_are_reported() -> None:
    run = stuck_join_in_run()
    report = report_of(run)
    assert set(report["calls"]) == {"handoff", "decision"}
    for kind, summary in report["calls"].items():
        assert set(summary) == SUMMARY_KEYS and summary["callKind"] == kind
    assert report["calls"]["decision"]["n"] == 6 and report["calls"]["handoff"]["n"] == 3
    assert report["calls"]["decision"]["wilsonUpper95"] == run.summaries["decision"].wilsonUpper95
    assert report["residency"] == {
        "start": [{"id": "dm/d", "state": "loaded"}, {"id": "dm/h", "state": "loaded"}],
        "end": [{"id": "dm/d", "state": "loaded"}, {"id": "dm/h", "state": "loaded"}],
    }
    assert (report["questionsSha256"], report["handoffSha256"]) == ("a" * 64, "b" * 64)
    assert report["decisionTimeoutSeconds"] == 3.0
    assert (report["decisionModel"], report["handoffModel"]) == ("dm/d", "dm/h")


def test_the_dispositions_list_each_item_with_reason_records_effect_and_class_figures() -> None:
    disposition = Disposition("c", "handoff", None, "http", 500, "oversized", ("a.json", "b.json"))
    report = DispositionReport(
        (DispositionEffect(disposition, "fail-closed"),),
        (ClassFigures(False, 1, 1, 2, F(1, 2)), ClassFigures(True, 0, 0, 1, F(1))),
        None,
    )
    labels = {"a": True, "b": False, "c": False}
    calls = stuck_join_in_run().collected.calls
    run = hand_run("join-in-judge", calls, labels, None, report)
    out = report_of(run)["dispositions"]
    (item,) = out["items"]
    assert item == {
        "itemId": "c", "callKind": "handoff", "textId": None, "failureKind": "http", "status": 500,
        "reason": "oversized", "recordFiles": ["a.json", "b.json"], "effect": "fail-closed",
    }  # fmt: skip
    assert out["byClass"][0] == {
        "label": False, "k": 1, "kWrong": 1, "nClass": 2, "ceiling": frac(1, 2),
    }  # fmt: skip
    assert out["byClass"][1]["ceiling"] == frac(1, 1)
    assert out["cellCounts"] is None


# --- on disk ---


@pytest.mark.parametrize("role", ROLES)
def test_a_successful_run_writes_the_report_beside_the_candidates_outside_the_manifest(
    tmp_path: Path, role: str
) -> None:
    run, _, data = base_cli.one_run(tmp_path, role)
    fits = base_cli.fits_dir(data, role)
    emission, written = emit.emit(
        run, fits, input_hashes=run.inputHashes, source_sha=run.sourceSha, report=True
    )
    name = f"{role}-{base_cli.VERSION}-f1.fit-report.json"
    assert {p.name for p in written} == {
        name,
        emission.manifest.name,
        f"{role}-{base_cli.VERSION}-handoff-f1.json",
        f"{role}-{base_cli.VERSION}-only-f1.json",
    }
    raw = (fits / name).read_bytes()
    report = json.loads(raw)
    assert set(report) == TOP_KEYS and report["fitOrdinal"] == 1
    assert report["callsFile"] == run.callsFile.name
    assert {f: b["candidateId"] for f, b in report["candidates"].items()} == {
        "handoff": f"{role}-{base_cli.VERSION}-handoff-f1",
        "decision-only": f"{role}-{base_cli.VERSION}-only-f1",
    }
    manifest = (fits / emission.manifest.name).read_bytes()
    assert b"schema" not in manifest and name.encode() not in manifest
    for family in ("handoff", "only"):
        declared = json.loads((fits / f"{role}-{base_cli.VERSION}-{family}-f1.json").read_text())
        assert declared["fitManifest"]["sha256"] == hashlib.sha256(manifest).hexdigest()
        assert declared["fitManifest"]["file"] != name
    base_emit.assert_no_canary(raw.decode(), data, role)


def test_a_report_that_cannot_be_built_writes_no_candidate_and_no_manifest(tmp_path: Path) -> None:
    run, _, data = base_cli.one_run(tmp_path, "guard-judge")
    fits = base_cli.fits_dir(data, "guard-judge")
    before = sorted(p.name for p in fits.iterdir())
    broken = fit.FitRun(**{**run.__dict__, "texts": ()})  # no `control` for the selected text
    with pytest.raises(KeyError):
        emit.emit(broken, fits, input_hashes=run.inputHashes, source_sha=run.sourceSha, report=True)
    assert sorted(p.name for p in fits.iterdir()) == before


def test_the_summary_prints_each_class_figure_and_each_disposition() -> None:
    disposition = Disposition("c", "handoff", None, "http", 500, "oversized", ("a.json", "b.json"))
    report = DispositionReport(
        (DispositionEffect(disposition, "fail-closed"),),
        (ClassFigures(False, 1, 1, 2, F(1, 2)),),
        None,
    )
    run = hand_run(
        "join-in-judge",
        stuck_join_in_run().collected.calls,
        {"a": True, "b": False, "c": False},
        None,
        report,
    )
    lines = fit._summary_lines(run)
    assert "excluded handoff call of c (fail-closed): oversized" in lines
    assert "class False: k=1 kWrong=1 n_class=2 ceiling=1/2" in lines


def test_main_prints_the_written_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = base_cli.write_data(tmp_path, "join-in-judge")
    client = base_cli.FakeClient("join-in-judge", data)
    assert fit.main(base_cli.argv(tmp_path, data, "join-in-judge"), client=client) == 0
    out = capsys.readouterr().out
    assert out.count("wrote ") == 4 and "fit-report.json" in out


def test_the_report_names_the_fit_ordinal_and_matching_candidate_ids_of_a_second_fit(
    tmp_path: Path,
) -> None:
    run, _, data = base_cli.one_run(tmp_path, "guard-judge")
    fits = base_cli.fits_dir(data, "guard-judge")
    kwargs = {"input_hashes": run.inputHashes, "source_sha": run.sourceSha, "report": True}
    emit.emit(run, fits, **kwargs)
    emit.emit(run, fits, **kwargs)
    report = json.loads((fits / "guard-judge-v9-f2.fit-report.json").read_text())
    assert report["fitOrdinal"] == 2
    assert report["candidates"]["handoff"]["candidateId"] == "guard-judge-v9-handoff-f2"
    assert report["candidates"]["decision-only"]["candidateId"] == "guard-judge-v9-only-f2"
    first = json.loads((fits / "guard-judge-v9-f1.fit-report.json").read_text())
    assert first["fitOrdinal"] == 1


def test_the_start_and_end_residency_snapshots_are_reported_separately() -> None:
    end = (
        ResidentModel("dm/d", "idle"),
        ResidentModel("dm/h", "loaded"),
        ResidentModel("dm/x", "loaded"),
    )
    labels = {"a": True, "b": False, "c": False}
    calls = stuck_join_in_run().collected.calls
    report = report_of(hand_run("join-in-judge", calls, labels, None, end=end))
    assert report["residency"]["start"] == [
        {"id": "dm/d", "state": "loaded"},
        {"id": "dm/h", "state": "loaded"},
    ]
    assert report["residency"]["end"] == [
        {"id": "dm/d", "state": "idle"},
        {"id": "dm/h", "state": "loaded"},
        {"id": "dm/x", "state": "loaded"},
    ]


def test_the_over_limit_count_is_the_texts_k_of_n(tmp_path: Path) -> None:
    run, _, data = base_cli.one_run(tmp_path, "guard-judge")
    fits = base_cli.fits_dir(data, "guard-judge")
    emit.emit(run, fits, input_hashes=run.inputHashes, source_sha=run.sourceSha, report=True)
    report = json.loads((fits / "guard-judge-v9-f1.fit-report.json").read_text())
    for family in ("handoff", "decision-only"):
        over = {t["textId"]: t["overLimit"] for t in report["candidates"][family]["texts"]}
        assert over["g-rubric"] == {"k": 1, "n": 8}
        assert over["g-control"] == {"k": 0, "n": 8}
