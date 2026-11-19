"""U6f.9b, first half: the calls file, the two candidates and the manifest
(`docs/plans/typed-decision-models4.md` §3.11, §5 row U6f.9b). Fit runs are the fake-client runs of
`test_fit_cli.py`; no socket opens. The last test loads the emitted bytes through model-bench's real
`Candidate.load` (skips only when `model-bench/` is absent)."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import test_fit_cli as base_cli

from decisionmodels import emit, fit
from decisionmodels.emit import EmissionFile, EmitRefused
from decisionmodels.fitstats import ScoredCall
from decisionmodels.lmstudio import LMStudioError, ResidentModel
from decisionmodels.manifest import parse_manifest

ROLES = ["guard-judge", "join-in-judge", "message-type-classifier"]
REQUIRED_KEYS = {
    "candidateId",
    "role",
    "decisionModel",
    "question",
    "policy",
    "handoffModel",
    "decisionTimeoutSeconds",
    "fit",
    "fitManifest",
    "provenance",
}
FIXED_NOW = datetime(2026, 11, 19, 12, 30, 5, tzinfo=UTC)
_MODEL_BENCH = Path(__file__).resolve().parents[2] / "model-bench"


def a_run(tmp_path: Path, role: str):
    run, client, data = base_cli.one_run(tmp_path, role)
    return run, client, data, base_cli.fits_dir(data, role)


def canaries(data: Path, role: str) -> list[str]:
    """Every item's unique canary tokens (`test_fit_cli.item_row` plants one per text field), found
    by walking each row's strings; at least one per row, so the leak check is never vacuous."""

    def walk(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.startswith("canary-") else []
        if isinstance(value, dict):
            return [t for v in value.values() for t in walk(v)]
        if isinstance(value, list):
            return [t for v in value for t in walk(v)]
        return []

    rows = (data / role / base_cli.VERSION / "items.jsonl").read_text().splitlines()
    found = [walk(json.loads(line)) for line in rows]
    assert len(found) == len(base_cli.ITEM_IDS) and all(len(tokens) == 1 for tokens in found)
    assert len({t for tokens in found for t in tokens}) == len(found)
    return [t for tokens in found for t in tokens]


def assert_no_canary(text: str, data: Path, role: str) -> None:
    assert "canary" not in text
    assert not [t for t in canaries(data, role) if t in text]


def calls_files(fits: Path) -> list[Path]:
    return sorted(fits.glob("calls-*.json"))


def emit_run(run, fits: Path):
    return emit.emit(run, fits, input_hashes=run.inputHashes, source_sha=run.sourceSha)


def with_ladder(run, family: str, **changes: Any):
    """`run` with the `family` candidate's chosen ladder edited."""
    selected = tuple(
        replace(c, ladder=replace(c.ladder, **changes)) if c.selection.candidate == family else c
        for c in run.analysis.selected
    )
    return replace(run, analysis=replace(run.analysis, selected=selected))


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# --- the candidates ---


@pytest.mark.parametrize("role", ROLES)
def test_each_candidate_carries_exactly_the_bench_keys_and_the_right_hand_off_model(
    tmp_path: Path, role: str
) -> None:
    run, _, _, fits = a_run(tmp_path, role)
    emission, _ = emit_run(run, fits)
    handoff = read(fits / f"{role}-{base_cli.VERSION}-handoff-f1.json")
    only = read(fits / f"{role}-{base_cli.VERSION}-only-f1.json")
    assert set(handoff) == set(only) == REQUIRED_KEYS
    assert handoff["handoffModel"] == base_cli.HANDOFF_MODEL
    assert "handoffModel" in only and only["handoffModel"] is None
    assert handoff["candidateId"] == f"{role}-{base_cli.VERSION}-handoff-f1"
    assert only["candidateId"] == f"{role}-{base_cli.VERSION}-only-f1"
    for doc in (handoff, only):
        assert doc["role"] == role
        assert doc["decisionModel"] == base_cli.DECISION_MODEL
        assert doc["decisionTimeoutSeconds"] == base_cli.TIMEOUT
        assert set(doc["fit"]) == {"outcome", "qualityTarget"}
        assert set(doc["fitManifest"]) == {"file", "sha256", "size", "purposes"}
        assert doc["fitManifest"]["purposes"] == ["question-tuning", "cutoff"]
        assert doc["provenance"] == {
            "builtBy": "decision-models",
            "datasetVersion": base_cli.VERSION,
            "sourceSha": run.sourceSha,
        }
    chosen = {c.selection.candidate: c for c in run.analysis.selected}
    assert handoff["question"] == dict(chosen["handoff"].text.spec)
    assert only["question"] == dict(chosen["decision-only"].text.spec)
    assert emission.ordinal == 1


@pytest.mark.parametrize(
    ("role", "targets"),
    [
        ("guard-judge", {"D_c<=0", "mcnemar-nonreject"}),
        ("join-in-judge", {f"T=0.{n}" for n in (94, 93, 92, 91, 90)} | {"max-min-agreement"}),
        (
            "message-type-classifier",
            {f"T=0.{n}" for n in (94, 93, 92, 91, 90)} | {"max-min-agreement"},
        ),
    ],
)
def test_the_fit_pair_comes_from_the_roles_bar_kind_closed_sets(
    tmp_path: Path, role: str, targets: set[str]
) -> None:
    run, _, _, fits = a_run(tmp_path, role)
    emit_run(run, fits)
    for family in ("handoff", "only"):
        fit_block = read(fits / f"{role}-{base_cli.VERSION}-{family}-f1.json")["fit"]
        assert fit_block["qualityTarget"] in targets
        assert fit_block["outcome"] in {
            "within-margin",
            "below-margin",
            "fit-infeasible: ranking",
            "fit-infeasible: agreement",
            "fit-infeasible: latency",
        }


def test_the_policy_maps_each_ladder_policy_to_the_bench_shape() -> None:
    assert emit.candidate_policy("g", {"l": 0.2, "h": 0.7, "s": 0}) == {
        "kind": "predicate-band",
        "yesAtOrAbove": 0.7,
        "noBelow": 0.2,
    }
    assert emit.candidate_policy("g", {"t": 0.4, "s": 0}) == {
        "kind": "predicate-band",
        "yesAtOrAbove": 0.4,
        "noBelow": 0.4,
    }
    assert emit.candidate_policy("m", {"c": 0.6, "s": 0}) == {
        "kind": "choice-confidence",
        "minConfidence": 0.6,
    }
    with pytest.raises(EmitRefused):
        emit.candidate_policy("g", {"s": 0})


@pytest.mark.parametrize(
    ("role", "bad"),
    [
        ("guard-judge", {"l": 0.2, "h": 0.8, "s": 0}),
        ("join-in-judge", {"l": 0.1, "h": 0.9, "s": 0}),
        ("message-type-classifier", {"c": 0.5, "s": 0}),
    ],
)
def test_a_policy_that_can_hand_off_beside_a_null_hand_off_model_refuses_and_writes_nothing(
    tmp_path: Path, role: str, bad: dict[str, Any]
) -> None:
    run, _, _, fits = a_run(tmp_path, role)
    for path in fits.glob("calls-*"):
        path.unlink()
    # the decision-only candidate (handoffModel null) given a band / a c > 0
    broken = with_ladder(run, "decision-only", policy=bad)
    with pytest.raises(ValueError, match="needs a handoffModel"):
        emit_run(broken, fits)
    assert list(fits.iterdir()) == []


def test_the_hand_off_candidate_is_checked_before_any_write_too(tmp_path: Path) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    broken = with_ladder(run, "handoff", policy={"l": 0.9, "h": 0.2, "s": 0})  # l > h
    with pytest.raises(ValueError, match="noBelow"):
        emit_run(broken, fits)
    assert not list(fits.glob("guard-judge-*"))


def test_a_target_of_another_bar_kind_refuses_before_the_write(tmp_path: Path) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    cross = with_ladder(run, "handoff", qualityTarget="T=0.94")
    with pytest.raises(ValueError, match="not a speed-up target"):
        emit_run(cross, fits)
    assert not list(fits.glob("guard-judge-*"))


def test_the_ordinal_is_one_more_than_the_largest_existing_and_never_reused(
    tmp_path: Path,
) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    other = "guard-judge-v8-only-f9.json"  # another dataset version: not this point's
    unrelated = "join-in-judge-v9-handoff-f7.json"
    for name in (other, unrelated, "guard-judge-v9-handoff-f3.json"):
        (fits / name).write_text("{}")
    first, _ = emit_run(run, fits)
    assert first.ordinal == 4
    second, _ = emit_run(run, fits)
    assert second.ordinal == 5
    assert (fits / "guard-judge-v9-handoff-f4.json").read_bytes() != b""
    assert (fits / "guard-judge-v9-only-f5.json").exists()
    assert emit.next_ordinal(fits, "guard-judge", "v9") == 6
    assert emit.next_ordinal(fits / "absent", "guard-judge", "v9") == 1


def test_a_name_that_exists_refuses_the_whole_emission_and_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    monkeypatch.setattr(emit, "next_ordinal", lambda *_a: 1)  # the ordinal the names collide at
    taken = fits / "guard-judge-v9-only-f1.json"  # the last file linked: all others roll back
    taken.write_bytes(b"other")
    before = sorted(p.name for p in fits.iterdir())
    with pytest.raises(EmitRefused, match="already exists"):
        emit.emit(run, fits, input_hashes=run.inputHashes, source_sha=run.sourceSha, report=True)
    assert sorted(p.name for p in fits.iterdir()) == before
    assert taken.read_bytes() == b"other"


def test_an_empty_source_sha_refuses_before_the_write(tmp_path: Path) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    with pytest.raises(EmitRefused, match="sourceSha"):
        emit.emit(run, fits, input_hashes=run.inputHashes, source_sha="")
    assert not list(fits.glob("guard-judge-*"))


# --- the manifest ---


@pytest.mark.parametrize("role", ROLES)
def test_the_manifest_holds_each_calibration_items_input_hash_and_nothing_else(
    tmp_path: Path, role: str
) -> None:
    run, _, data, fits = a_run(tmp_path, role)
    emission, _ = emit_run(run, fits)
    rows = [
        json.loads(line)
        for line in (data / role / base_cli.VERSION / "items.jsonl").read_text().splitlines()
    ]
    fields = {
        "guard-judge": ("condition", "understanding", "turns", "path"),
        "join-in-judge": ("agentName", "priorTurns", "newMessage"),
        "message-type-classifier": ("text", "priorTurn"),
    }[role]
    expected = sorted(
        hashlib.sha256(
            json.dumps(
                {k: r[k] for k in fields if k in r},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        for r in rows
    )
    raw = (fits / emission.manifest.name).read_bytes()
    assert list(parse_manifest(raw)) == expected
    assert len(expected) == len(base_cli.ITEM_IDS)
    assert raw == "".join(f"{h}\n" for h in expected).encode()
    for family in ("handoff", "only"):
        declared = read(fits / f"{role}-{base_cli.VERSION}-{family}-f1.json")["fitManifest"]
        assert declared["file"] == emission.manifest.name
        assert declared["sha256"] == hashlib.sha256(raw).hexdigest()
        assert declared["size"] == len(expected)
    assert not any(row["itemId"].encode() in raw for row in rows)


def test_only_the_manifest_and_the_two_candidates_are_written_by_the_first_half(
    tmp_path: Path,
) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    before = set(fits.iterdir())
    _, written = emit_run(run, fits)
    assert {p.name for p in set(fits.iterdir()) - before} == {p.name for p in written}
    assert sorted(p.name for p in written) == [
        "guard-judge-v9-f1.fit.txt",
        "guard-judge-v9-handoff-f1.json",
        "guard-judge-v9-only-f1.json",
    ]
    assert not list(fits.glob(".*"))  # no temporary left


def test_extra_files_are_written_with_the_rest_and_never_enter_the_manifest(
    tmp_path: Path,
) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    emission, _ = emit.emit(
        run,
        fits,
        input_hashes=run.inputHashes,
        source_sha=run.sourceSha,
        extra=[EmissionFile("guard-judge-v9-f1.fit-report.json", b'{"k": 1}\n')],
    )
    assert (fits / "guard-judge-v9-f1.fit-report.json").read_bytes() == b'{"k": 1}\n'
    assert b"k" not in (fits / emission.manifest.name).read_bytes()


# --- the calls file ---


@pytest.mark.parametrize("role", ROLES)
def test_a_successful_run_leaves_the_calls_file_labelled_ok_with_every_call_and_no_item_text(
    tmp_path: Path, role: str
) -> None:
    run, _, data, fits = a_run(tmp_path, role)
    (path,) = calls_files(fits)
    assert run.callsFile == path
    document = read(path)
    assert document["schema"] == "decision-models.fit-calls/1"
    assert document["endResidency"] == "ok"
    assert len(document["calls"]) == len(run.collected.calls) == len(base_cli.ITEM_IDS) * 6
    first = document["calls"][0]
    assert set(first) == {"itemId", "callKind", "textId", "outcome", "answer", "elapsedSeconds"}
    assert [(c["itemId"], c["callKind"], c["textId"], c["outcome"]) for c in document["calls"]] == [
        (c.itemId, c.callKind, c.textId, c.outcome) for c in run.collected.calls
    ]
    assert [c["elapsedSeconds"] for c in document["calls"]] == [
        c.elapsedSeconds for c in run.collected.calls
    ]
    recorded = {
        (m["itemId"], m["textId"]): m["maxProbability"] for m in document["maxProbabilities"]
    }
    assert recorded == dict(run.collected.maxProbabilities)
    if role == "message-type-classifier":
        assert document["maxProbabilities"]
        assert any(isinstance(c["answer"], list) for c in document["calls"])
    else:
        assert document["maxProbabilities"] == []
    assert_no_canary(path.read_text(), data, role)
    assert path.name.startswith("calls-") and path.name.endswith(f"-{document['runId']}.json")


def test_the_calls_file_name_carries_the_utc_timestamp_and_the_run_id(tmp_path: Path) -> None:
    data = base_cli.write_data(tmp_path, "guard-judge")
    client = base_cli.FakeClient("guard-judge", data)
    fits = base_cli.fits_dir(data, "guard-judge")
    run = fit.run_fit(
        client,
        role="guard-judge",
        dataset_version=base_cli.VERSION,
        decision_model=base_cli.DECISION_MODEL,
        handoff_model=base_cli.HANDOFF_MODEL,
        decision_timeout_seconds=base_cli.TIMEOUT,
        fits_dir=fits,
        data_dir=data,
        now=lambda: FIXED_NOW,
    )
    assert run.callsFile is not None
    stamp, run_id = run.callsFile.name[len("calls-") : -len(".json")].split("-")
    assert stamp == "20261119T123005Z"
    assert len(run_id) == 32 and read(run.callsFile)["runId"] == run_id


def test_an_analysis_exception_leaves_the_calls_file_and_writes_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = base_cli.write_data(tmp_path, "guard-judge")
    client = base_cli.FakeClient("guard-judge", data)
    fits = base_cli.fits_dir(data, "guard-judge")

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("analysis bug")

    monkeypatch.setattr(fit, "analyze_fit", boom)
    with pytest.raises(RuntimeError, match="analysis bug"):
        fit.run_fit(
            client,
            role="guard-judge",
            dataset_version=base_cli.VERSION,
            decision_model=base_cli.DECISION_MODEL,
            handoff_model=base_cli.HANDOFF_MODEL,
            decision_timeout_seconds=base_cli.TIMEOUT,
            fits_dir=fits,
            data_dir=data,
        )
    (path,) = fits.iterdir()
    document = read(path)
    assert path.name.startswith("calls-") and document["endResidency"] == "ok"
    assert len(document["calls"]) == len(base_cli.ITEM_IDS) * 6
    assert_no_canary(path.read_text(), data, "guard-judge")


def test_the_calls_file_exists_before_the_analysis_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = base_cli.write_data(tmp_path, "guard-judge")
    client = base_cli.FakeClient("guard-judge", data)
    fits = base_cli.fits_dir(data, "guard-judge")
    seen: list[int] = []
    real = fit.analyze_fit

    def spy(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        seen.append(len(calls_files(fits)))
        return real(*args, **kwargs)

    monkeypatch.setattr(fit, "analyze_fit", spy)
    fit.run_fit(
        client,
        role="guard-judge",
        dataset_version=base_cli.VERSION,
        decision_model=base_cli.DECISION_MODEL,
        handoff_model=base_cli.HANDOFF_MODEL,
        decision_timeout_seconds=base_cli.TIMEOUT,
        fits_dir=fits,
        data_dir=data,
    )
    assert seen == [1]


def end_failure_run(tmp_path: Path, script: list[Any]):
    data = base_cli.write_data(tmp_path, "guard-judge")
    client = base_cli.FakeClient("guard-judge", data)
    client.residency_script = script
    fits = base_cli.fits_dir(data, "guard-judge")
    return client, fits, data


def run_kwargs(data: Path, fits: Path) -> dict[str, Any]:
    return {
        "role": "guard-judge",
        "dataset_version": base_cli.VERSION,
        "decision_model": base_cli.DECISION_MODEL,
        "handoff_model": base_cli.HANDOFF_MODEL,
        "decision_timeout_seconds": base_cli.TIMEOUT,
        "fits_dir": fits,
        "data_dir": data,
    }


def test_an_end_refusal_writes_the_calls_file_labelled_refused_and_nothing_else(
    tmp_path: Path,
) -> None:
    client, fits, data = end_failure_run(
        tmp_path, [base_cli.BOTH, [ResidentModel(base_cli.HANDOFF_MODEL, "loaded")]]
    )
    with pytest.raises(fit.ResidencyRefused, match="not resident at end"):
        fit.run_fit(client, **run_kwargs(data, fits))
    (path,) = fits.iterdir()
    assert path.name.startswith("calls-")
    document = read(path)
    assert document["endResidency"] == "refused"
    assert len(document["calls"]) == len(base_cli.ITEM_IDS) * 6


def test_an_end_read_that_raises_labels_the_calls_file_unreadable(tmp_path: Path) -> None:
    client, fits, data = end_failure_run(tmp_path, [base_cli.BOTH, LMStudioError("down")])

    def residency() -> list[ResidentModel]:
        client.reads += 1
        if client.reads > 1:
            raise LMStudioError("down")
        return list(base_cli.BOTH)

    client.residency = residency  # type: ignore[method-assign]
    with pytest.raises(fit.ResidencyRefused, match="residency read at end"):
        fit.run_fit(client, **run_kwargs(data, fits))
    (path,) = fits.iterdir()
    assert read(path)["endResidency"] == "unreadable"


def test_a_start_refusal_and_an_abort_write_no_calls_file(tmp_path: Path) -> None:
    client, fits, data = end_failure_run(
        tmp_path, [[ResidentModel(base_cli.DECISION_MODEL, "loaded")]]
    )
    with pytest.raises(fit.ResidencyRefused, match="not resident at start"):
        fit.run_fit(client, **run_kwargs(data, fits))
    assert not fits.exists()
    client, fits, data = end_failure_run(tmp_path / "abort", [base_cli.BOTH])
    client.fail_decide[(base_cli.ITEM_IDS[2], "g-source")] = base_cli.LMStudioCallFailed(
        "boom", status=503, body="", kind="http", wallClockMs=5.0
    )
    with pytest.raises(fit.FitAborted):
        fit.run_fit(client, **run_kwargs(data, fits))
    assert [p.name.split("-")[0] for p in fits.iterdir()] == ["abort"]


def test_the_calls_file_is_not_a_candidate_and_not_in_the_manifest(tmp_path: Path) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    emission, _ = emit_run(run, fits)
    assert run.callsFile is not None
    manifest_names = {emission.manifest.name, *(f.name for f in emission.candidates.values())}
    assert run.callsFile.name not in manifest_names
    assert b"calls" not in (fits / emission.manifest.name).read_bytes()
    assert emit.next_ordinal(fits, "guard-judge", "v9") == 2  # the calls file is no candidate id


def test_the_calls_label_is_closed() -> None:
    with pytest.raises(ValueError, match="endResidency"):
        emit.calls_document("r", None, None, "fine")  # type: ignore[arg-type]


# --- the real check: model-bench's loader ---


@pytest.mark.skipif(not _MODEL_BENCH.is_dir(), reason="model-bench/ absent")
@pytest.mark.parametrize("role", ROLES)
def test_both_candidates_and_the_manifest_load_through_the_benchs_candidate_load(
    tmp_path: Path, role: str
) -> None:
    sys.path.insert(0, str(_MODEL_BENCH))
    try:
        from modelbench.decisions import Candidate, load_fit_manifest
    finally:
        sys.path.remove(str(_MODEL_BENCH))
    run, _, _, fits = a_run(tmp_path, role)
    emission, _ = emit_run(run, fits)
    stage = tmp_path / "bench"
    (stage / "candidates").mkdir(parents=True)
    for path in fits.iterdir():
        if path.name.startswith(role):
            (stage / "candidates" / path.name).write_bytes(path.read_bytes())
    for family in ("handoff", "only"):
        candidate = Candidate.load(stage, f"{role}-{base_cli.VERSION}-{family}-f1")
        loaded = load_fit_manifest(stage, candidate.fitManifest)
        assert loaded.size == len(base_cli.ITEM_IDS)
        assert candidate.handoffModel == (base_cli.HANDOFF_MODEL if family == "handoff" else None)
        assert (
            candidate.contentHash
            == hashlib.sha256(
                emission.candidates["handoff" if family == "handoff" else "decision-only"].data
            ).hexdigest()
        )


def test_each_candidate_carries_its_own_selected_text_and_its_own_outcome(tmp_path: Path) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    other = next(t for t in run.texts if not t.control)
    selected = tuple(
        replace(
            c,
            text=other,
            ladder=replace(
                c.ladder, rung=2, qualityTarget="mcnemar-nonreject", agreementUnmet=True
            ),
        )
        if c.selection.candidate == "decision-only"
        else c
        for c in run.analysis.selected
    )
    mixed = replace(run, analysis=replace(run.analysis, selected=selected))
    emit_run(mixed, fits)
    handoff = read(fits / "guard-judge-v9-handoff-f1.json")
    only = read(fits / "guard-judge-v9-only-f1.json")
    assert only["question"] == dict(other.spec)
    assert handoff["question"] != only["question"]
    assert only["fit"] == {
        "outcome": "fit-infeasible: agreement",
        "qualityTarget": "mcnemar-nonreject",
    }
    assert handoff["fit"]["outcome"] != "fit-infeasible: agreement"


def test_no_eligible_text_is_a_ranking_outcome_on_both_candidates(tmp_path: Path) -> None:
    run, _, _, fits = a_run(tmp_path, "join-in-judge")
    selected = tuple(
        replace(c, selection=replace(c.selection, noEligibleText=True))
        for c in run.analysis.selected
    )
    emit_run(replace(run, analysis=replace(run.analysis, selected=selected)), fits)
    for family in ("handoff", "only"):
        doc = read(fits / f"join-in-judge-v9-{family}-f1.json")
        assert doc["fit"]["outcome"] == "fit-infeasible: ranking"


# --- review round: pins ---


def test_the_ordinal_counts_manifest_and_report_names_too(tmp_path: Path) -> None:
    fits = tmp_path / "fits"
    fits.mkdir()
    assert emit.next_ordinal(fits, "guard-judge", "v9") == 1
    (fits / "guard-judge-v9-f6.fit.txt").write_text("")
    assert emit.next_ordinal(fits, "guard-judge", "v9") == 7
    (fits / "guard-judge-v9-f8.fit-report.json").write_text("{}")
    assert emit.next_ordinal(fits, "guard-judge", "v9") == 9
    (fits / "guard-judge-v8-f20.fit.txt").write_text("")  # another version
    (fits / "calls-20261119T123005Z-f30.json").write_text("{}")  # no point's file
    assert emit.next_ordinal(fits, "guard-judge", "v9") == 9


def test_files_are_linked_manifest_then_report_then_the_candidates_so_none_is_orphaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    linked: list[str] = []
    real = emit.os.link

    def spy(source: Path, target: Path) -> None:
        name = Path(target).name
        if "-handoff-" in name or "-only-" in name:
            assert (fits / "guard-judge-v9-f1.fit.txt").exists()
            assert (fits / "guard-judge-v9-f1.fit-report.json").exists()
        linked.append(name)
        real(source, target)

    monkeypatch.setattr(emit.os, "link", spy)
    emit.emit(run, fits, input_hashes=run.inputHashes, source_sha=run.sourceSha, report=True)
    assert linked == [
        "guard-judge-v9-f1.fit.txt",
        "guard-judge-v9-f1.fit-report.json",
        "guard-judge-v9-handoff-f1.json",
        "guard-judge-v9-only-f1.json",
    ]


def refuse_links(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_links(*_a: Any, **_k: Any) -> None:
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(emit.os, "link", no_links)


def test_without_hard_links_the_calls_file_is_still_written_and_candidates_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, data, fits = a_run(tmp_path, "guard-judge")
    refuse_links(monkeypatch)
    path = emit.write_calls_file(
        fits, inputs=run.inputs, collected=run.collected, end_residency="ok", run_id="f" * 32
    )
    assert read(path)["runId"] == "f" * 32 and len(read(path)["calls"]) == len(run.collected.calls)
    with pytest.raises(OSError):
        emit_run(run, fits)
    assert not list(fits.glob("guard-judge-*")) and not list(fits.glob(".*"))


def test_main_without_hard_links_keeps_the_calls_file_and_refuses_with_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = base_cli.write_data(tmp_path, "guard-judge")
    client = base_cli.FakeClient("guard-judge", data)
    refuse_links(monkeypatch)
    assert fit.main(base_cli.argv(tmp_path, data, "guard-judge"), client=client) == 1
    assert "refused:" in capsys.readouterr().err
    (written,) = base_cli.fits_dir(data, "guard-judge").iterdir()
    assert written.name.startswith("calls-") and read(written)["endResidency"] == "ok"


def test_a_calls_file_write_error_at_an_end_refusal_never_masks_the_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, fits, data = end_failure_run(
        tmp_path, [base_cli.BOTH, [ResidentModel(base_cli.HANDOFF_MODEL, "loaded")]]
    )

    def broken(*_a: Any, **_k: Any) -> Path:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fit, "write_calls_file", broken)
    with pytest.raises(fit.ResidencyRefused, match="not resident at end") as caught:
        fit.run_fit(client, **run_kwargs(data, fits))
    assert any("No space left" in note for note in caught.value.__notes__)


def test_a_timeout_error_end_read_labels_the_calls_file_unreadable(tmp_path: Path) -> None:
    client, fits, data = end_failure_run(tmp_path, [base_cli.BOTH])

    def residency() -> list[ResidentModel]:
        client.reads += 1
        if client.reads > 1:
            raise TimeoutError("slow")
        return list(base_cli.BOTH)

    client.residency = residency  # type: ignore[method-assign]
    with pytest.raises(fit.ResidencyRefused, match="residency read at end"):
        fit.run_fit(client, **run_kwargs(data, fits))
    (path,) = fits.iterdir()
    assert read(path)["endResidency"] == "unreadable"


@pytest.mark.parametrize("role", ROLES)
def test_source_sha_is_the_independent_sha256_of_the_items_file(tmp_path: Path, role: str) -> None:
    run, _, data, fits = a_run(tmp_path, role)
    emit_run(run, fits)
    expected = hashlib.sha256((data / role / base_cli.VERSION / "items.jsonl").read_bytes())
    for family in ("handoff", "only"):
        doc = read(fits / f"{role}-{base_cli.VERSION}-{family}-f1.json")
        assert doc["provenance"]["sourceSha"] == expected.hexdigest()


def test_the_calls_file_inputs_block_equals_the_runs_inputs(tmp_path: Path) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    inputs = read(run.callsFile)["inputs"]
    assert inputs == {
        "role": "guard-judge",
        "datasetVersion": base_cli.VERSION,
        "questionsSha256": run.inputs.questionsSha256,
        "handoffSha256": run.inputs.handoffSha256,
        "decisionModel": base_cli.DECISION_MODEL,
        "handoffModel": base_cli.HANDOFF_MODEL,
        "decisionTimeoutSeconds": base_cli.TIMEOUT,
    }
    assert inputs["handoffSha256"] != inputs["questionsSha256"]
    assert (
        inputs["handoffSha256"]
        == hashlib.sha256((fit.FIT_DIR / "guard-judge" / "handoff.md").read_bytes()).hexdigest()
    )


def test_a_non_finite_number_refuses_a_candidate_but_never_loses_the_calls() -> None:
    with pytest.raises(ValueError):
        emit._json_bytes({"x": float("nan")})
    assert b"NaN" in emit._json_bytes({"x": float("nan")}, strict=False)


def test_a_run_level_exclusion_keeps_the_excluded_item_in_the_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import test_exclusion as ex

    data = ex.write_guard_data(tmp_path, 204)
    client = ex.Client(data)
    client.fail_chat = {"x-0001"}
    ex.two_aborts(data, client)
    capsys.readouterr()
    code = ex.run(data, client, "--exclude-item", "x-0001:handoff", "--reason", "oversized input")
    assert code == 0, capsys.readouterr().err
    fits = ex.fits(data)
    (manifest,) = fits.glob("*.fit.txt")
    lines = manifest.read_text().splitlines()
    assert len(lines) == 204
    row = next(
        json.loads(line)
        for line in (data / ex.ROLE / ex.VERSION / "items.jsonl").read_text().splitlines()
        if json.loads(line)["itemId"] == "x-0001"
    )
    excluded = hashlib.sha256(
        json.dumps(
            {k: row[k] for k in ("condition", "understanding", "turns", "path") if k in row},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    assert excluded in lines
    (candidate, *_) = sorted(fits.glob(f"{ex.ROLE}-{ex.VERSION}-handoff-f*.json"))
    assert json.loads(candidate.read_text())["fitManifest"]["size"] == 204
    (report,) = fits.glob("*.fit-report.json")
    (item,) = json.loads(report.read_text())["dispositions"]["items"]
    assert (item["itemId"], item["effect"]) == ("x-0001", "item-removed")


# --- review round 2 ---


def test_a_non_finite_value_never_loses_the_calls_file(tmp_path: Path) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    calls = (
        ScoredCall("x-0001", "handoff", None, "ok", True, float("inf")),
        ScoredCall("x-0001", "decision", "t", "ok", float("nan"), 0.1),
    )
    collected = replace(run.collected, calls=calls, maxProbabilities={})
    path = emit.write_calls_file(
        fits, inputs=run.inputs, collected=collected, end_residency="ok", run_id="e" * 32
    )
    assert path.exists() and "Infinity" in path.read_text() and "NaN" in path.read_text()


def test_the_fallback_never_deletes_a_foreign_file_at_the_calls_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _, _, fits = a_run(tmp_path, "guard-judge")
    name = f"calls-20261119T123005Z-{'d' * 32}.json"
    (fits / name).write_bytes(b"foreign")
    refuse_links(monkeypatch)
    with pytest.raises(FileExistsError):
        emit.write_calls_file(
            fits,
            inputs=run.inputs,
            collected=run.collected,
            end_residency="ok",
            run_id="d" * 32,
            now=lambda: FIXED_NOW,
        )
    assert (fits / name).read_bytes() == b"foreign"


def broken_calls_write(*_a: Any, **_k: Any) -> Path:
    raise OSError(28, "No space left on device")


def test_main_prints_the_notes_of_a_refusal_whose_calls_file_could_not_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = base_cli.write_data(tmp_path, "guard-judge")
    client = base_cli.FakeClient("guard-judge", data)
    client.residency_script = [base_cli.BOTH, [ResidentModel(base_cli.HANDOFF_MODEL, "loaded")]]
    monkeypatch.setattr(fit, "write_calls_file", broken_calls_write)
    assert fit.main(base_cli.argv(tmp_path, data, "guard-judge"), client=client) == 1
    err = capsys.readouterr().err
    assert "not resident at end" in err and "No space left on device" in err


def test_main_refuses_when_the_calls_file_cannot_be_written_on_a_successful_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = base_cli.write_data(tmp_path, "guard-judge")
    client = base_cli.FakeClient("guard-judge", data)
    monkeypatch.setattr(fit, "write_calls_file", broken_calls_write)
    assert fit.main(base_cli.argv(tmp_path, data, "guard-judge"), client=client) == 1
    err = capsys.readouterr().err
    assert "refused:" in err and "No space left" in err
    assert not list(base_cli.fits_dir(data, "guard-judge").glob("guard-judge-*"))
