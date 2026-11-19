"""`python -m decisionmodels.fit`: the flag contract, the refusals that write nothing and an
end-to-end run per role against a fake client (plan4 §3.11, §5 row U6f.8b). No socket opens.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from decisionmodels import fit, render
from decisionmodels.fit import FitRefused, load_calibration, main, run_fit, validate_url
from decisionmodels.lmstudio import (
    ChatResult,
    DecisionAnswer,
    DecisionResult,
    LMStudioCallFailed,
    OverLimit,
    ResidentModel,
)
from decisionmodels.points import load_point
from decisionmodels.questions import load_questions

DECISION_MODEL = "dm/decider"
HANDOFF_MODEL = "dm/handoff"
BOTH = [ResidentModel(DECISION_MODEL, "loaded"), ResidentModel(HANDOFF_MODEL, "loaded")]
COMPONENT = Path(__file__).resolve().parents[1]
ITEM_IDS = [f"x-{n:04d}" for n in range(1, 9)]
URL = "http://localhost:1234"
VERSION = "v9"
TIMEOUT = 45.5

REPLY = {
    "guard-judge": lambda label: json.dumps({"decision": label}),
    "join-in-judge": lambda label: json.dumps({"reply": label}),
    "message-type-classifier": lambda label: json.dumps({"type": label}),
}


def label_of(role: str, n: int) -> Any:
    if role == "message-type-classifier":
        return load_point(role).expected[n % 5]
    return n % 2 == 0


def item_row(role: str, n: int, item_id: str) -> dict[str, Any]:
    label = label_of(role, n)
    if role == "guard-judge":
        return {
            "itemId": item_id,
            "condition": f"canary-condition-{n}",
            "understanding": {},
            "turns": [],
            "path": "turns" if n % 4 < 2 else "understanding",
            "expected": label,
        }
    if role == "join-in-judge":
        message = {"role": "user", "displayName": "Ana", "text": f"canary-message-{n}"}
        return {
            "itemId": item_id,
            "agentName": "Bot",
            "priorTurns": [],
            "newMessage": message,
            "expected": label,
        }
    return {
        "itemId": item_id,
        "text": f"canary-message-{n}",
        "priorTurn": None,
        "expected": label,
    }


def write_data(root: Path, role: str, version: str = VERSION) -> Path:
    data = root / "data"
    target = data / role / version
    target.mkdir(parents=True)
    rows = [item_row(role, n, i) for n, i in enumerate(ITEM_IDS, start=1)]
    (target / "items.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    splits = {"rngSeed": 1, "eval": [], "train": [], "calibration": ITEM_IDS}
    (target / "splits.json").write_text(json.dumps(splits))
    return data


class FakeClient:
    """A scripted client: the hand-off is always right; the control text ranks the label."""

    def __init__(self, role: str, data: Path, version: str = VERSION) -> None:
        self.role = role
        cal = load_calibration(role, version, data)
        self.by_input = {c.decisionInput: c.itemId for c in cal.items}
        self.labels = cal.labels
        self.text_of = {q.spec["instructions"]: q.id for q in load_questions(role)}
        self.residency_script: list[Any] = [BOTH]
        self.reads = 0
        self.chat_calls = 0
        self.decide_calls = 0
        self.timeouts: list[float] = []
        self.slow_ms: dict[tuple[str, str], float] = {}
        self.fail_decide: dict[tuple[str, str], Exception] = {}

    def residency(self) -> list[ResidentModel]:
        outcome = self.residency_script[min(self.reads, len(self.residency_script) - 1)]
        self.reads += 1
        return list(outcome)

    def chat(self, messages, *, model, temperature, max_tokens, timeout_s):  # type: ignore[no-untyped-def]
        self.chat_calls += 1
        item = self.by_input[messages[-1]["content"]]
        return ChatResult(REPLY[self.role](self.labels[item]), None, None, 300.0)

    def decide(self, input, questions, *, model, timeout_s):  # type: ignore[no-untyped-def]
        self.decide_calls += 1
        self.timeouts.append(timeout_s)
        item = self.by_input[input]
        text = self.text_of[questions[0]["instructions"]]
        if (item, text) in self.fail_decide:
            raise self.fail_decide[(item, text)]
        if text == "g-rubric" and item == ITEM_IDS[0]:
            return OverLimit(900.0)
        result = self.answer(item, text)
        if (item, text) in self.slow_ms:
            return DecisionResult(result.answers, None, None, self.slow_ms[(item, text)])
        return result

    def answer(self, item: str, text: str) -> DecisionResult:
        label = self.labels[item]
        good = text.endswith("control")  # the control ranks the label; every other text reverses it
        if self.role == "message-type-classifier":
            vocab = load_point(self.role).expected
            flipped = ITEM_IDS.index(item) % 4 == 3
            right = flipped != good  # the control is right off the flipped items, the rest on them
            choice = label if right else vocab[(vocab.index(label) + 1) % len(vocab)]
            confidence = 0.9 if right == good else 0.4
            high = 0.8 if confidence > 0.5 else 0.3
            probs = tuple((v, high if v == choice else 0.05) for v in vocab)
            answer = DecisionAnswer("type", "choice", None, choice, probs, confidence)
        else:
            p = (0.9 if label else 0.1) if good else (0.1 if label else 0.9)
            answer = DecisionAnswer("q", "predicate", p, None, None, None)
        return DecisionResult((answer,), None, None, 120.0)


def argv(tmp_path: Path, data: Path, role: str, *extra: str) -> list[str]:
    base = [
        "--role", role,
        "--dataset-version", VERSION,
        "--decision-model", DECISION_MODEL,
        "--handoff-model", HANDOFF_MODEL,
        "--lmstudio-url", URL,
        "--decision-timeout-seconds", str(TIMEOUT),
        "--data-dir", str(data),
    ]  # fmt: skip
    return [*base, *extra]


def fits_dir(data: Path, role: str) -> Path:
    return data / role / VERSION / "fits"


# --- the flag contract ---


def drop_flag(args: list[str], flag: str) -> list[str]:
    i = args.index(flag)
    return args[:i] + args[i + 2 :]


@pytest.mark.parametrize(
    "flag",
    [
        "--role",
        "--dataset-version",
        "--decision-model",
        "--handoff-model",
        "--lmstudio-url",
        "--decision-timeout-seconds",
    ],
)
def test_every_flag_but_the_paths_is_required_and_a_missing_one_calls_nothing(
    tmp_path: Path, flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_data(tmp_path, "guard-judge")
    client = FakeClient("guard-judge", data)
    with pytest.raises(SystemExit) as info:
        main(drop_flag(argv(tmp_path, data, "guard-judge"), flag), client=client)
    assert info.value.code == 2
    assert flag in capsys.readouterr().err
    assert (client.reads, client.chat_calls, client.decide_calls) == (0, 0, 0)
    assert not fits_dir(data, "guard-judge").exists()


def test_the_timeout_flag_has_no_default() -> None:
    action = next(
        a for a in fit._parser()._actions if "--decision-timeout-seconds" in a.option_strings
    )
    assert action.required is True
    assert action.default is None


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "abc"])
def test_a_bad_timeout_refuses_and_writes_nothing(
    tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_data(tmp_path, "guard-judge")
    client = FakeClient("guard-judge", data)
    args = argv(tmp_path, data, "guard-judge")
    args[args.index("--decision-timeout-seconds") + 1] = value
    if value == "abc":
        with pytest.raises(SystemExit):
            main(args, client=client)
    else:
        assert main(args, client=client) == 1
        assert "refused" in capsys.readouterr().err
    assert client.reads == 0
    assert not fits_dir(data, "guard-judge").exists()


@pytest.mark.parametrize(
    "url", ["", "nourl", "localhost:1234", "ftp://host", "http://", "http://host:notaport"]
)
def test_a_bad_lm_studio_url_refuses_before_any_call_and_writes_nothing(
    tmp_path: Path, url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_data(tmp_path, "guard-judge")
    client = FakeClient("guard-judge", data)
    args = argv(tmp_path, data, "guard-judge")
    args[args.index("--lmstudio-url") + 1] = url
    assert main(args, client=client) == 1
    assert "refused" in capsys.readouterr().err
    assert (client.reads, client.chat_calls, client.decide_calls) == (0, 0, 0)
    assert not fits_dir(data, "guard-judge").exists()


def test_validate_url_accepts_http_and_https_with_a_host() -> None:
    assert validate_url("http://localhost:1234") == "http://localhost:1234"
    assert validate_url("https://lm.example/x") == "https://lm.example/x"
    with pytest.raises(FitRefused):
        validate_url("lm.example")


def test_the_exclude_item_flag_needs_a_reason(tmp_path: Path) -> None:
    data = write_data(tmp_path, "guard-judge")
    with pytest.raises(SystemExit):
        main([*argv(tmp_path, data, "guard-judge"), "--exclude-item", "x:handoff"])


def test_the_module_runs_as_a_script_and_requires_the_timeout() -> None:
    done = subprocess.run(
        [sys.executable, "-B", "-m", "decisionmodels.fit", "--role", "guard-judge"],
        cwd=COMPONENT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    assert "--decision-timeout-seconds" in done.stderr


# --- residency: refusals write nothing ---


def test_a_model_not_resident_at_start_refuses_with_no_calls_and_no_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_data(tmp_path, "guard-judge")
    client = FakeClient("guard-judge", data)
    client.residency_script = [[ResidentModel(DECISION_MODEL, "loaded")]]
    assert main(argv(tmp_path, data, "guard-judge"), client=client) == 1
    assert "not resident at start" in capsys.readouterr().err
    assert (client.chat_calls, client.decide_calls) == (0, 0)
    assert not fits_dir(data, "guard-judge").exists()


def test_a_model_gone_at_the_end_refuses_after_the_calls_leaving_only_the_calls_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_data(tmp_path, "guard-judge")
    client = FakeClient("guard-judge", data)
    client.residency_script = [BOTH, [ResidentModel(HANDOFF_MODEL, "loaded")]]
    assert main(argv(tmp_path, data, "guard-judge"), client=client) == 1
    assert "not resident at end" in capsys.readouterr().err
    assert client.chat_calls == len(ITEM_IDS)
    (written,) = fits_dir(data, "guard-judge").iterdir()
    assert written.name.startswith("calls-")
    assert json.loads(written.read_text())["endResidency"] == "refused"


def test_a_failed_call_aborts_with_the_one_abort_record_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_data(tmp_path, "guard-judge")
    client = FakeClient("guard-judge", data)
    client.fail_decide[(ITEM_IDS[2], "g-source")] = LMStudioCallFailed(
        "boom", status=503, body="", kind="http", wallClockMs=5.0
    )
    assert main(argv(tmp_path, data, "guard-judge"), client=client) == 1
    assert "aborted" in capsys.readouterr().err
    written = sorted(p.name for p in fits_dir(data, "guard-judge").iterdir())
    assert len(written) == 1 and written[0].startswith("abort-")


def test_a_questions_or_calibration_input_that_is_missing_refuses_writing_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "empty"
    data.mkdir()
    assert main(argv(tmp_path, data, "guard-judge"), client=object()) == 1
    assert "refused" in capsys.readouterr().err
    assert not (data / "guard-judge").exists()


# --- end to end, one fixture per role ---


@pytest.mark.parametrize("role", ["guard-judge", "join-in-judge", "message-type-classifier"])
def test_a_fit_makes_each_call_once_and_selects_per_candidate(
    tmp_path: Path, role: str, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_data(tmp_path, role)
    client = FakeClient(role, data)
    assert main(argv(tmp_path, data, role), client=client) == 0
    out = capsys.readouterr().out
    assert client.chat_calls == len(ITEM_IDS)
    assert (
        client.decide_calls == len(ITEM_IDS) * 5
    )  # one per (item, text), shared by both candidates
    assert client.reads == 2  # start and end residency
    assert "handoff: text" in out and "decision-only: text" in out
    names = sorted(p.name for p in fits_dir(data, role).iterdir())
    stem = f"{role}-{VERSION}"
    assert [n for n in names if not n.startswith("calls-")] == [
        f"{stem}-f1.fit-report.json",
        f"{stem}-f1.fit.txt",
        f"{stem}-handoff-f1.json",
        f"{stem}-only-f1.json",
    ]
    assert len([n for n in names if n.startswith("calls-")]) == 1
    assert out.count("wrote ") == 4


@pytest.mark.parametrize("role", ["guard-judge", "join-in-judge", "message-type-classifier"])
def test_the_run_scores_every_text_for_both_candidates_and_the_control_ranks_the_label(
    tmp_path: Path, role: str
) -> None:
    data = write_data(tmp_path, role)
    client = FakeClient(role, data)
    run = run_fit(
        client,
        role=role,
        dataset_version=VERSION,
        decision_model=DECISION_MODEL,
        handoff_model=HANDOFF_MODEL,
        decision_timeout_seconds=TIMEOUT,
        fits_dir=fits_dir(data, role),
        data_dir=data,
    )
    scores = run.analysis.scores
    assert len(scores) == 2 * 5
    control = next(s for s in scores if s.textId.endswith("control") and s.candidate == "handoff")
    reversed_ = [s for s in scores if not s.textId.endswith("control")]
    assert control.auc == 1.0
    assert {s.auc for s in reversed_} <= {0.0, 0.5}
    assert [c.selection.textId for c in run.analysis.selected] == [
        next(t.id for t in run.texts if t.control)
    ] * 2
    assert {c.selection.noEligibleText for c in run.analysis.selected} == {False}
    assert all(s.leftSelection in (None, "auc") for s in scores)
    assert run.inputs.decisionTimeoutSeconds == TIMEOUT
    assert set(client.timeouts) == {TIMEOUT}
    assert set(run.summaries) == {"handoff", "decision"}
    for score in scores:
        assert len(score.splitHalf) == 2


def test_the_guard_run_counts_an_over_limit_decision_for_its_text(tmp_path: Path) -> None:
    role = "guard-judge"
    data = write_data(tmp_path, role)
    run = run_fit(
        FakeClient(role, data),
        role=role,
        dataset_version=VERSION,
        decision_model=DECISION_MODEL,
        handoff_model=HANDOFF_MODEL,
        decision_timeout_seconds=TIMEOUT,
        fits_dir=fits_dir(data, role),
        data_dir=data,
    )
    over = {s.textId: s.overLimit for s in run.analysis.scores if s.candidate == "handoff"}
    assert over["g-rubric"] == (1, len(ITEM_IDS))
    assert over["g-control"] == (0, len(ITEM_IDS))


def test_the_choice_run_reports_a_max_probability_auc(tmp_path: Path) -> None:
    role = "message-type-classifier"
    data = write_data(tmp_path, role)
    run = run_fit(
        FakeClient(role, data),
        role=role,
        dataset_version=VERSION,
        decision_model=DECISION_MODEL,
        handoff_model=HANDOFF_MODEL,
        decision_timeout_seconds=TIMEOUT,
        fits_dir=fits_dir(data, role),
        data_dir=data,
    )
    control = next(s for s in run.analysis.scores if s.textId == "m-control")
    assert control.maxProbAuc is not None
    assert run.analysis.maxProbDropped == dict.fromkeys((q.id for q in run.texts), 0)


def test_load_calibration_reads_the_calibration_ids_in_splits_order_with_guard_cells(
    tmp_path: Path,
) -> None:
    data = write_data(tmp_path, "guard-judge")
    cal = load_calibration("guard-judge", VERSION, data)
    assert [c.itemId for c in cal.items] == ITEM_IDS
    first = item_row("guard-judge", 1, ITEM_IDS[0])
    assert cal.items[0].decisionInput == render.render_guard_item(first)
    assert cal.items[0].handoffInput == cal.items[0].decisionInput
    assert cal.cells is not None and set(cal.cells.values()) <= {
        "falseAdvanceRateByTurns",
        "falseAdvanceRateByUnderstanding",
        "falseSuspendRateByTurns",
        "falseSuspendRateByUnderstanding",
    }
    assert (
        load_calibration(
            "join-in-judge", VERSION, write_data(tmp_path / "j", "join-in-judge")
        ).cells
        is None
    )


# --- plumbing pins (P54-m1, P54-m2) ---


def one_run(tmp_path: Path, role: str = "guard-judge", client: FakeClient | None = None):
    data = tmp_path / "data" if client is not None else write_data(tmp_path, role)
    client = client or FakeClient(role, data)
    run = run_fit(
        client,
        role=role,
        dataset_version=VERSION,
        decision_model=DECISION_MODEL,
        handoff_model=HANDOFF_MODEL,
        decision_timeout_seconds=TIMEOUT,
        fits_dir=fits_dir(data, role),
        data_dir=data,
    )
    return run, client, data


def test_the_run_records_each_files_sha256_in_its_own_field(tmp_path: Path) -> None:
    run, _, _ = one_run(tmp_path)
    questions = (fit.FIT_DIR / "guard-judge" / "questions.json").read_bytes()
    handoff = (fit.FIT_DIR / "guard-judge" / "handoff.md").read_bytes()
    assert run.inputs.questionsSha256 == hashlib.sha256(questions).hexdigest()
    assert run.inputs.handoffSha256 == hashlib.sha256(handoff).hexdigest()
    assert run.inputs.questionsSha256 != run.inputs.handoffSha256
    assert (run.inputs.role, run.inputs.datasetVersion) == ("guard-judge", VERSION)
    assert (run.inputs.decisionModel, run.inputs.handoffModel) == (DECISION_MODEL, HANDOFF_MODEL)


@pytest.mark.parametrize("role", ["guard-judge", "join-in-judge", "message-type-classifier"])
def test_load_calibration_renders_every_item_into_both_inputs(tmp_path: Path, role: str) -> None:
    data = write_data(tmp_path, role)
    cal = load_calibration(role, VERSION, data)
    render_item = fit._RENDERERS[role]
    assert len(cal.items) == len(ITEM_IDS)
    for n, (item, item_id) in enumerate(zip(cal.items, ITEM_IDS, strict=True), start=1):
        row = item_row(role, n, item_id)
        assert item.itemId == item_id
        assert item.decisionInput == item.handoffInput == render_item(row)
        assert cal.labels[item_id] == row["expected"]


def test_the_half_budget_count_uses_the_decision_timeout_given(tmp_path: Path) -> None:
    role = "guard-judge"
    data = write_data(tmp_path, role)
    client = FakeClient(role, data)
    client.slow_ms[(ITEM_IDS[1], "g-control")] = 20_000.0  # under TIMEOUT / 2 = 22.75 s
    client.slow_ms[(ITEM_IDS[2], "g-control")] = 23_000.0  # over it
    run, _, _ = one_run(tmp_path, role, client)
    assert run.summaries["decision"].halfBudgetCount == 1
    assert run.summaries["decision"].maxSeconds == 23.0


def test_the_selected_text_carries_the_score_of_its_own_text_and_candidate(tmp_path: Path) -> None:
    run, _, _ = one_run(tmp_path)
    for chosen in run.analysis.selected:
        assert chosen.score.textId == chosen.selection.textId == chosen.text.id
        assert chosen.score.candidate == chosen.selection.candidate


def test_an_analysis_error_after_the_calls_is_not_reported_as_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_data(tmp_path, "guard-judge")
    client = FakeClient("guard-judge", data)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise ValueError("ladder broke")

    monkeypatch.setattr(fit, "analyze_fit", boom)
    with pytest.raises(ValueError, match="ladder broke"):
        main(argv(tmp_path, data, "guard-judge"), client=client)
    assert client.decide_calls == len(ITEM_IDS) * 5
    assert "refused" not in capsys.readouterr().err


def test_a_malformed_calibration_file_is_a_refusal_before_any_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_data(tmp_path, "guard-judge")
    client = FakeClient("guard-judge", data)
    (data / "guard-judge" / VERSION / "items.jsonl").write_text("{not json\n")
    assert main(argv(tmp_path, data, "guard-judge"), client=client) == 1
    assert "refused" in capsys.readouterr().err
    assert client.reads == 0


def test_a_calibration_row_missing_a_field_is_a_refusal_before_any_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = write_data(tmp_path, "guard-judge")
    client = FakeClient("guard-judge", data)
    path = data / "guard-judge" / VERSION / "items.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    del rows[0]["expected"]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert main(argv(tmp_path, data, "guard-judge"), client=client) == 1
    assert "refused" in capsys.readouterr().err
    assert (client.reads, client.chat_calls, client.decide_calls) == (0, 0, 0)
    assert not fits_dir(data, "guard-judge").exists()
