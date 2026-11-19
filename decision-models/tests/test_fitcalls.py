"""`decisionmodels.fitcalls` (plan4 §3.11, §4.4 U6f.8a, §5 row U6f.8a; fit note §4).

The client is a fake and the clock is a fake (`Clock`): a failed call's elapsed time is the clock's
difference, a success's is the client's `wallClockMs / 1000`. Nothing sleeps or opens a socket.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from decisionmodels.fitcalls import (
    AbortRecord,
    CalibrationItem,
    Disposition,
    ExclusionDidNotFire,
    FitAborted,
    FitInputs,
    ResidencyRefused,
    check_end,
    check_start,
    collect_calls,
    summarize_calls,
    wilson_upper,
)
from decisionmodels.fitstats import ScoredCall
from decisionmodels.handoff import (
    HANDOFF_CHAT_TIMEOUT_SECONDS,
    HANDOFF_MAX_TOKENS,
    HANDOFF_TEMPERATURE,
)
from decisionmodels.ladder import choice_ladder, guard_ladder, predicate_ladder
from decisionmodels.lmstudio import (
    ChatResult,
    DecisionAnswer,
    DecisionResult,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    OverLimit,
    ResidentModel,
)
from decisionmodels.questions import Question

DECISION_MODEL = "dm/decider"
HANDOFF_MODEL = "dm/handoff"
BOTH = [ResidentModel(DECISION_MODEL, "loaded"), ResidentModel(HANDOFF_MODEL, "loaded")]

# plan4 §3.11 "AbortRecord" row, keys exactly
ABORT_KEYS = {
    "schema",
    "runId",
    "itemId",
    "callKind",
    "textId",
    "failureKind",
    "status",
    "elapsedSeconds",
    "inputs",
    "residencyStart",
    "residencyNow",
}
INPUT_KEYS = {
    "role",
    "datasetVersion",
    "questionsSha256",
    "handoffSha256",
    "decisionModel",
    "handoffModel",
    "decisionTimeoutSeconds",
}
FILE_RE = re.compile(r"abort-\d{8}T\d{6}Z-[0-9a-f]+\.json")


def inputs_for(role: str = "guard-judge", timeout: float = 30.0) -> FitInputs:
    return FitInputs(
        role=role,
        datasetVersion="v1",
        questionsSha256="a" * 64,
        handoffSha256="b" * 64,
        decisionModel=DECISION_MODEL,
        handoffModel=HANDOFF_MODEL,
        decisionTimeoutSeconds=timeout,
    )


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def timeout_exc() -> Exception:
    return LMStudioCallTimeout("timed out")


def failed_exc(kind: str, status: int | None = None) -> Exception:
    return LMStudioCallFailed(
        f"{kind} failure", status=status, body=None, kind=kind, wallClockMs=None
    )


FAILURES: dict[str, tuple[Callable[[], Exception], int | None]] = {
    "timeout": (timeout_exc, None),
    "connection": (lambda: failed_exc("connection"), None),
    "http": (lambda: failed_exc("http", 503), 503),
    "malformed": (lambda: failed_exc("malformed"), None),
}


def predicate_result(p: float, ms: float = 250.0) -> DecisionResult:
    return DecisionResult((DecisionAnswer("q", "predicate", p, None, None, None),), None, None, ms)


def choice_result(choice: str, confidence: float, ms: float = 250.0) -> DecisionResult:
    """`confidence` (0.75) differs from `max(probabilities)` (0.5), which differs from the first
    probability (0.25): a mutant reading either of those for the other reddens."""
    probabilities = (("question", 0.25), (choice, 0.5), ("other", 0.125))
    answer = DecisionAnswer("q", "choice", None, choice, probabilities, confidence)
    return DecisionResult((answer,), None, None, ms)


class FakeClient:
    """Scripted per item. `chat_script[item]` and `decide_script[(item, text)]` each hold a
    result (or `str` reply for chat) or an exception to raise; the clock advances `step` seconds
    on a raise."""

    def __init__(self, clock: Clock, *, step: float = 7.5) -> None:
        self.clock = clock
        self.step = step
        self.chat_script: dict[str, Any] = {}
        self.decide_script: dict[tuple[str, str], Any] = {}
        self.default_chat: Any = '{"decision": true, "rationale": "ok"}'
        self.default_decide: Any = predicate_result(0.5)
        self.residency_script: list[Any] = [BOTH]
        self.chat_calls: list[dict[str, Any]] = []
        self.decide_calls: list[dict[str, Any]] = []
        self.residency_reads = 0

    def residency(self) -> list[ResidentModel]:
        index = min(self.residency_reads, len(self.residency_script) - 1)
        self.residency_reads += 1
        outcome = self.residency_script[index]
        if isinstance(outcome, Exception):
            raise outcome
        return list(outcome)

    def chat(self, messages, *, model, temperature, max_tokens, timeout_s):  # type: ignore[no-untyped-def]
        user = messages[-1]["content"]
        self.chat_calls.append(
            dict(
                user=user,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
        )
        item = user.removeprefix("h-")
        scripted = self.chat_script.get(item, self.default_chat)
        if isinstance(scripted, Exception):
            self.clock.t += getattr(scripted, "advance", self.step)
            raise scripted
        return ChatResult(scripted, None, None, 400.0)

    def decide(self, input, questions, *, model, timeout_s):  # type: ignore[no-untyped-def]
        text = questions[0]["instructions"]
        self.decide_calls.append(
            dict(input=input, text=text, model=model, timeout_s=timeout_s, n=len(questions))
        )
        item = input.removeprefix("d-")
        scripted = self.decide_script.get((item, text), self.default_decide)
        if isinstance(scripted, Exception):
            self.clock.t += getattr(scripted, "advance", self.step)
            raise scripted
        return scripted


def predicate_texts(*ids: str) -> list[Question]:
    return [
        Question(i, i == ids[0], {"type": "predicate", "name": "q", "instructions": i}) for i in ids
    ]


def choice_texts(labels: Sequence[str], *ids: str) -> list[Question]:
    choices = [{"value": v, "instructions": ""} for v in labels]
    return [
        Question(
            i,
            i == ids[0],
            {"type": "choice", "name": "q", "instructions": i, "choices": choices},
        )
        for i in ids
    ]


def items_for(*ids: str) -> list[CalibrationItem]:
    return [CalibrationItem(i, f"d-{i}", f"h-{i}") for i in ids]


def run(
    client: FakeClient,
    tmp_path: Path,
    *,
    role: str = "guard-judge",
    ids: Sequence[str] = ("i1", "i2", "i3"),
    texts: Sequence[Question] | None = None,
    dispositions: Sequence[Disposition] = (),
):
    return collect_calls(
        client,
        inputs=inputs_for(role),
        items=items_for(*ids),
        texts=texts if texts is not None else predicate_texts("t1", "t2"),
        residency_start=tuple(BOTH),
        fits_dir=tmp_path / "fits",
        dispositions=dispositions,
        clock=client.clock,
        now=lambda: datetime(2026, 11, 19, 12, 30, 45, tzinfo=UTC),
    )


def files(tmp_path: Path) -> list[Path]:
    directory = tmp_path / "fits"
    return sorted(directory.iterdir()) if directory.exists() else []


# --- abort ---


@pytest.mark.parametrize("call_kind", ["handoff", "decision"])
@pytest.mark.parametrize("failure", sorted(FAILURES))
def test_any_failure_aborts_with_exactly_one_record(tmp_path, call_kind, failure):  # type: ignore[no-untyped-def]
    make, status = FAILURES[failure]
    clock = Clock()
    client = FakeClient(clock)
    if call_kind == "handoff":
        client.chat_script["i2"] = make()
    else:
        client.decide_script[("i2", "t2")] = make()
    client.residency_script = [OSError("down")]  # the abort-time read raises
    with pytest.raises(FitAborted) as raised:
        run(client, tmp_path)
    written = files(tmp_path)
    assert [p.name for p in written] == [raised.value.path.name]
    assert FILE_RE.fullmatch(written[0].name)
    assert written[0].name.startswith("abort-20261119T123045Z-")
    body = json.loads(written[0].read_text(encoding="utf-8"))
    assert set(body) == ABORT_KEYS
    assert set(body["inputs"]) == INPUT_KEYS
    assert body["failureKind"] == failure
    assert body["status"] == status
    assert body["callKind"] == call_kind
    assert body["itemId"] == "i2"
    assert body["textId"] == (None if call_kind == "handoff" else "t2")
    assert body["elapsedSeconds"] == 7.5
    assert body["residencyNow"] is None
    assert body["residencyStart"] == [
        {"id": DECISION_MODEL, "state": "loaded"},
        {"id": HANDOFF_MODEL, "state": "loaded"},
    ]
    assert body["runId"] in written[0].name
    assert body["inputs"]["decisionTimeoutSeconds"] == 30.0
    text = written[0].read_text(encoding="utf-8")
    assert "h-i2" not in text and "d-i2" not in text  # no item text
    # the abort stopped the run: nothing after the failing call was made
    if call_kind == "handoff":
        assert [c["user"] for c in client.chat_calls] == ["h-i1", "h-i2"]
    else:
        assert client.decide_calls[-1]["input"] == "d-i2"
        assert not any(c["input"] == "d-i3" for c in client.decide_calls)


def test_abort_record_keys_equal_the_dataclass_fields():
    assert {f.name for f in fields(AbortRecord)} == ABORT_KEYS
    assert {f.name for f in fields(FitInputs)} == INPUT_KEYS


def test_residency_now_is_a_fresh_read_when_it_works(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.chat_script["i1"] = failed_exc("connection")
    client.residency_script = [[ResidentModel(DECISION_MODEL, "loaded")]]
    with pytest.raises(FitAborted) as raised:
        run(client, tmp_path)
    assert raised.value.record.residencyNow == (ResidentModel(DECISION_MODEL, "loaded"),)
    assert raised.value.record.residencyStart == tuple(BOTH)


def test_two_aborted_runs_write_distinct_random_run_ids(tmp_path):  # type: ignore[no-untyped-def]
    names = []
    for _ in range(2):
        client = FakeClient(Clock())
        client.chat_script["i1"] = failed_exc("connection")
        with pytest.raises(FitAborted) as raised:
            run(client, tmp_path)
        names.append(raised.value.record.runId)
    assert names[0] != names[1]
    assert len(files(tmp_path)) == 2


def test_a_success_run_leaves_no_abort_file(tmp_path):  # type: ignore[no-untyped-def]
    run(FakeClient(Clock()), tmp_path)
    assert files(tmp_path) == []


def test_over_limit_is_scored_and_does_not_abort(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.decide_script[("i2", "t1")] = OverLimit(wallClockMs=1500.0)
    collected = run(client, tmp_path)
    over = [c for c in collected.calls if c.outcome == "over-limit"]
    assert over == [ScoredCall("i2", "decision", "t1", "over-limit", None, 1.5)]
    assert len(collected.calls) == 3 * 3  # 3 items x (1 hand-off + 2 texts)
    assert files(tmp_path) == []


# --- residency ---


def test_start_refuses_when_a_model_is_absent():
    for resident in ([BOTH[0]], [BOTH[1]], []):
        client = FakeClient(Clock())
        client.residency_script = [resident]
        with pytest.raises(ResidencyRefused) as raised:
            check_start(client, inputs_for())
        assert raised.value.phase == "start"


def test_start_returns_the_snapshot_when_both_are_resident():
    assert check_start(FakeClient(Clock()), inputs_for()) == tuple(BOTH)


@pytest.mark.parametrize("phase", ["start", "end"])
def test_a_residency_read_that_raises_refuses(phase):  # type: ignore[no-untyped-def]
    check = check_start if phase == "start" else check_end
    for error in (
        LMStudioCallFailed("x", status=None, body=None, kind="connection"),
        LMStudioCallTimeout("x"),
        TimeoutError("x"),
    ):
        client = FakeClient(Clock())
        client.residency_script = [error]
        with pytest.raises(ResidencyRefused) as raised:
            check(client, inputs_for())
        assert raised.value.phase == phase


def test_end_reads_afresh_and_refuses_when_a_model_is_gone(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.residency_script = [BOTH, [BOTH[0]]]
    check_start(client, inputs_for())
    run(client, tmp_path)
    with pytest.raises(ResidencyRefused) as raised:
        check_end(client, inputs_for())
    assert raised.value.phase == "end"
    assert client.residency_reads == 2  # a fresh read, not the start snapshot
    assert files(tmp_path) == []  # an end refusal writes nothing


def test_end_passes_when_both_remain():
    assert check_end(FakeClient(Clock()), inputs_for()) == tuple(BOTH)


# --- exemption scope ---


def disposition(**over: Any) -> Disposition:
    base: dict[str, Any] = dict(
        itemId="i2",
        callKind="decision",
        textId="t2",
        failureKind="http",
        status=503,
        reason="twice",
        recordFiles=("abort-a.json", "abort-b.json"),
    )
    base.update(over)
    return Disposition(**base)


def test_a_disposition_call_is_recorded_failed_and_the_run_continues(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.decide_script[("i2", "t2")] = failed_exc("http", 503)
    collected = run(client, tmp_path, dispositions=[disposition()])
    failed = [c for c in collected.calls if c.outcome == "failed"]
    assert failed == [ScoredCall("i2", "decision", "t2", "failed", None, 7.5)]
    assert len(collected.calls) == 9  # the run went on through i3
    assert files(tmp_path) == []


def test_a_handoff_disposition_is_recorded_failed(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.chat_script["i2"] = failed_exc("malformed")
    d = disposition(callKind="handoff", textId=None, failureKind="malformed", status=None)
    collected = run(client, tmp_path, dispositions=[d])
    assert ScoredCall("i2", "handoff", None, "failed", None, 7.5) in collected.calls
    assert len(collected.calls) == 9


MUTANTS = {
    "same item, other call kind": ("handoff", failed_exc("http", 503)),
    "other text": (("i2", "t1"), failed_exc("http", 503)),
    "other failureKind": (("i2", "t2"), failed_exc("malformed")),
    "other status": (("i2", "t2"), failed_exc("http", 500)),
    "other item": (("i3", "t2"), failed_exc("http", 503)),
}


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_anything_but_exactly_the_disposition_aborts(tmp_path, mutant):  # type: ignore[no-untyped-def]
    where, exc = MUTANTS[mutant]
    client = FakeClient(Clock())
    client.decide_script[("i2", "t2")] = failed_exc("http", 503)  # the disposition fires
    if where == "handoff":
        client.chat_script["i2"] = exc
    else:
        client.decide_script[where] = exc
    with pytest.raises(FitAborted):
        run(client, tmp_path, dispositions=[disposition()])
    assert len(files(tmp_path)) == 1


# --- shapes and plumbing ---


def test_the_calls_the_client_receives_and_the_elapsed_plumbing(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.default_decide = predicate_result(0.25, ms=1234.0)
    collected = run(client, tmp_path, ids=("i1",))
    assert collected.calls == (
        ScoredCall("i1", "handoff", None, "ok", True, 0.4),
        ScoredCall("i1", "decision", "t1", "ok", 0.25, 1.234),
        ScoredCall("i1", "decision", "t2", "ok", 0.25, 1.234),
    )
    assert client.chat_calls == [
        dict(
            user="h-i1",
            model=HANDOFF_MODEL,
            temperature=HANDOFF_TEMPERATURE,
            max_tokens=HANDOFF_MAX_TOKENS,
            timeout_s=HANDOFF_CHAT_TIMEOUT_SECONDS,
        )
    ]
    assert [(c["text"], c["model"], c["timeout_s"], c["n"]) for c in client.decide_calls] == [
        ("t1", DECISION_MODEL, 30.0, 1),
        ("t2", DECISION_MODEL, 30.0, 1),
    ]


def test_a_choice_decision_is_a_tuple_and_message_type_handoff_a_label_or_none(tmp_path):  # type: ignore[no-untyped-def]
    labels = ["order", "question", "complaint", "escalation", "other"]
    client = FakeClient(Clock())
    client.default_decide = choice_result("order", 0.75)
    client.chat_script = {"i1": '{"type": "question"}', "i2": '{"type": "nope"}'}
    collected = run(
        client,
        tmp_path,
        role="message-type-classifier",
        ids=("i1", "i2"),
        texts=choice_texts(labels, "t1"),
    )
    by_key = {(c.itemId, c.callKind): c for c in collected.calls}
    assert by_key[("i1", "decision")].answer == ("order", 0.75)
    assert type(by_key[("i1", "decision")].answer) is tuple
    assert by_key[("i1", "handoff")].answer == "question"
    assert by_key[("i2", "handoff")].answer is None
    assert by_key[("i2", "handoff")].outcome == "ok"
    assert collected.parseFailures == 0  # an unknown label is not a parse failure
    # max(probabilities) is 0.5: neither the confidence (0.75) nor the first probability (0.25)
    assert collected.maxProbabilities == {("i1", "t1"): 0.5, ("i2", "t1"): 0.5}


@pytest.mark.parametrize(
    ("role", "content", "verdict"),
    [
        ("guard-judge", '{"decision": true, "rationale": "yes"}', True),
        ("guard-judge", "no json at all", False),
        ("join-in-judge", '{"reply": true}', True),
        ("join-in-judge", "no json at all", False),
    ],
)
def test_handoff_verdicts_and_parse_failure_counts(tmp_path, role, content, verdict):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.chat_script["i1"] = content
    collected = run(client, tmp_path, role=role, ids=("i1",), texts=predicate_texts("t1"))
    handoff = collected.calls[0]
    assert (handoff.outcome, handoff.answer) == ("ok", verdict)
    assert collected.parseFailures == (0 if "{" in content else 1)


def test_the_decision_timeout_reaches_decide(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    collect_calls(
        client,
        inputs=inputs_for(timeout=12.5),
        items=items_for("i1"),
        texts=predicate_texts("t1"),
        residency_start=tuple(BOTH),
        fits_dir=tmp_path / "fits",
        clock=client.clock,
    )
    assert client.decide_calls[0]["timeout_s"] == 12.5


# --- the round trip through the ladders ---


def round_trip_fixture(role: str):  # type: ignore[no-untyped-def]
    """Six items covering every outcome and a parse failure; returns (client, texts, dispositions,
    labels, ids)."""
    ids = ("i1", "i2", "i3", "i4", "i5", "i6")
    client = FakeClient(Clock())
    if role == "message-type-classifier":
        labels_set = ["order", "question", "complaint", "escalation", "other"]
        texts = choice_texts(labels_set, "t1")
        labels: dict[str, Any] = dict(
            zip(ids, ["order", "question", "order", "other", "order", "question"], strict=True)
        )
        client.default_decide = choice_result("order", 0.6)
        client.default_chat = '{"type": "order"}'
        client.chat_script["i2"] = "unparseable"  # parse failure -> None
        client.decide_script[("i3", "t1")] = OverLimit(900.0)
        client.decide_script[("i5", "t1")] = failed_exc("malformed")
    else:
        texts = predicate_texts("t1")
        key = "decision" if role == "guard-judge" else "reply"
        client.default_decide = predicate_result(0.6)
        client.default_chat = json.dumps({key: True, "rationale": "r"})
        labels = dict(zip(ids, [True, False, True, False, True, False], strict=True))
        client.chat_script["i2"] = "unparseable"  # parse failure -> False
        client.decide_script[("i3", "t1")] = OverLimit(900.0)
        client.decide_script[("i5", "t1")] = failed_exc("malformed")
        client.decide_script[("i6", "t1")] = predicate_result(0.2)
    client.chat_script["i4"] = failed_exc("http", 500)
    dispositions = [
        disposition(itemId="i4", callKind="handoff", textId=None, failureKind="http", status=500),
        disposition(
            itemId="i5", callKind="decision", textId="t1", failureKind="malformed", status=None
        ),
    ]
    return client, texts, dispositions, labels, ids


@pytest.mark.parametrize("family", ["handoff", "decision-only"])
@pytest.mark.parametrize("role", ["guard-judge", "join-in-judge", "message-type-classifier"])
def test_every_call_kind_and_outcome_goes_through_its_ladder(tmp_path, role, family):  # type: ignore[no-untyped-def]
    client, texts, dispositions, labels, ids = round_trip_fixture(role)
    collected = run(client, tmp_path, role=role, ids=ids, texts=texts, dispositions=dispositions)
    outcomes = {(c.callKind, c.outcome) for c in collected.calls}
    assert outcomes == {
        ("handoff", "ok"),
        ("handoff", "failed"),
        ("decision", "ok"),
        ("decision", "over-limit"),
        ("decision", "failed"),
    }
    assert collected.parseFailures == 1
    if role == "guard-judge":
        cells = {i: ("yes" if labels[i] else "no") for i in ids}
        result = guard_ladder(
            collected.calls, labels=labels, cells=cells, text_id="t1", family=family
        )
    elif role == "join-in-judge":
        result = predicate_ladder(collected.calls, labels=labels, text_id="t1", family=family)
    else:
        result = choice_ladder(collected.calls, labels=labels, text_id="t1", family=family)
    assert result.family == family


# --- summaries ---


def test_wilson_literals_from_the_fit_note():
    # k = 0: z^2 / (n + z^2); the note's "about 1.26% at n = 300"
    assert wilson_upper(0, 300) == pytest.approx(0.012643, abs=5e-7)
    assert wilson_upper(0, 300) == pytest.approx(1.96**2 / (300 + 1.96**2), abs=1e-12)
    assert wilson_upper(0, 500) == pytest.approx(0.0076, abs=5e-5)
    # k = 1, n = 300, hand-computed from the formula
    assert wilson_upper(1, 300) == pytest.approx(0.018637, abs=5e-7)


def test_wilson_refuses_a_bad_count():
    for k, n in ((0, 0), (-1, 5), (6, 5)):
        with pytest.raises(ValueError):
            wilson_upper(k, n)


def call(kind: str, outcome: str, seconds: float, item: str = "i") -> ScoredCall:
    return ScoredCall(item, kind, None if kind == "handoff" else "t", outcome, None, seconds)


def test_summaries_count_median_max_k_and_half_budget_on_fake_times():
    decision = [
        call("decision", "ok", 5.0, "a"),  # exactly half of 10: not above
        call("decision", "ok", 5.5, "b"),
        call("decision", "failed", 9.0, "c"),
        call("decision", "over-limit", 1.0, "d"),
    ]
    handoff = [
        call("handoff", "ok", 90.0, "a"),  # exactly half of 180: not above
        call("handoff", "ok", 90.5, "b"),
        call("handoff", "ok", 2.0, "c"),
    ]
    got = summarize_calls(decision + handoff, decision_timeout_seconds=10.0, parse_failures=2)
    d, h = got["decision"], got["handoff"]
    assert (d.n, d.k, d.dispositionItemIds) == (4, 1, ("c",))
    assert (d.medianSeconds, d.maxSeconds, d.halfBudgetCount) == (5.25, 9.0, 2)
    assert d.wilsonUpper95 == pytest.approx(wilson_upper(1, 4))
    assert d.parseFailures == 0
    assert (h.n, h.k, h.dispositionItemIds) == (3, 0, ())
    assert (h.medianSeconds, h.maxSeconds, h.halfBudgetCount) == (90.0, 90.5, 1)
    assert h.wilsonUpper95 == pytest.approx(wilson_upper(0, 3))
    assert h.parseFailures == 2


def test_summaries_use_each_kinds_own_budget():
    # 50 s is above half of a 60 s decision budget but nowhere near half of the 180 s hand-off one
    calls = [call("decision", "ok", 50.0), call("handoff", "ok", 50.0)]
    got = summarize_calls(calls, decision_timeout_seconds=60.0)
    assert got["decision"].halfBudgetCount == 1
    assert got["handoff"].halfBudgetCount == 0


def test_summaries_from_a_collected_run(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.chat_script["i1"] = "unparseable"
    collected = run(client, tmp_path, ids=("i1", "i2"))
    got = summarize_calls(
        collected.calls,
        decision_timeout_seconds=30.0,
        parse_failures=collected.parseFailures,
    )
    assert got["handoff"].n == 2 and got["decision"].n == 4
    assert got["handoff"].parseFailures == 1
    assert got["handoff"].medianSeconds == 0.4


@pytest.mark.parametrize("call_kind", ["handoff", "decision"])
def test_the_same_status_with_another_failure_kind_aborts(tmp_path, call_kind):  # type: ignore[no-untyped-def]
    """`malformed` and `connection` both have status `None`: only `failureKind` differs."""
    client = FakeClient(Clock())
    text = None if call_kind == "handoff" else "t2"
    if call_kind == "handoff":
        client.chat_script["i2"] = failed_exc("connection")
    else:
        client.decide_script[("i2", "t2")] = failed_exc("connection")
    d = disposition(callKind=call_kind, textId=text, failureKind="malformed", status=None)
    with pytest.raises(FitAborted) as raised:
        run(client, tmp_path, dispositions=[d])
    assert raised.value.record.failureKind == "connection"


def test_max_probabilities_only_for_ok_choice_decisions(tmp_path):  # type: ignore[no-untyped-def]
    labels = ["order", "question", "complaint", "escalation", "other"]
    client = FakeClient(Clock())
    client.default_decide = choice_result("order", 0.75)
    client.decide_script[("i2", "t1")] = OverLimit(10.0)
    client.decide_script[("i3", "t1")] = failed_exc("http", 500)
    d = disposition(itemId="i3", textId="t1", status=500)
    collected = run(
        client,
        tmp_path,
        role="message-type-classifier",
        texts=choice_texts(labels, "t1"),
        dispositions=[d],
    )
    assert collected.maxProbabilities == {("i1", "t1"): 0.5}
    predicate = run(FakeClient(Clock()), tmp_path)
    assert predicate.maxProbabilities == {}


# --- the exclusion must fire (fit note section 4) ---


@pytest.mark.parametrize("outcome", ["ok", "over-limit"])
def test_a_disposition_call_that_succeeds_refuses_at_that_call(tmp_path, outcome):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    if outcome == "over-limit":
        client.decide_script[("i2", "t2")] = OverLimit(100.0)
    with pytest.raises(ExclusionDidNotFire) as raised:
        run(client, tmp_path, dispositions=[disposition()])
    assert raised.value.outcome == outcome
    assert raised.value.disposition == disposition()
    assert files(tmp_path) == []
    assert not any(c["input"] == "d-i3" for c in client.decide_calls)  # raised at that call


def test_a_handoff_disposition_whose_call_succeeds_refuses(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    d = disposition(callKind="handoff", textId=None, failureKind="http", status=500)
    with pytest.raises(ExclusionDidNotFire) as raised:
        run(client, tmp_path, dispositions=[d])
    assert raised.value.outcome == "ok"
    assert files(tmp_path) == []
    assert [c["user"] for c in client.chat_calls] == ["h-i1", "h-i2"]


# --- gate P52 minors ---


def test_failure_kind_is_read_from_the_exception_not_its_message(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.decide_script[("i2", "t2")] = LMStudioCallFailed(
        "connection reset by timeout", status=None, body=None, kind="malformed"
    )
    with pytest.raises(FitAborted) as raised:
        run(client, tmp_path)
    assert raised.value.record.failureKind == "malformed"


@pytest.mark.parametrize(
    "gone",
    [[DECISION_MODEL], [HANDOFF_MODEL], [DECISION_MODEL, HANDOFF_MODEL]],
    ids=["decision", "handoff", "both"],
)
def test_the_end_check_refuses_each_model_gone_and_both(gone):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.residency_script = [[m for m in BOTH if m.id not in gone]]
    with pytest.raises(ResidencyRefused) as raised:
        check_end(client, inputs_for())
    assert raised.value.phase == "end"


def advancing(exc: Exception, seconds: float) -> Exception:
    exc.advance = seconds  # type: ignore[attr-defined]
    return exc


def test_every_call_has_its_own_timer_and_summary_ids_are_unique_and_sorted(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.chat_script["i3"] = advancing(failed_exc("connection"), 7.5)
    client.decide_script[("i3", "t1")] = advancing(failed_exc("connection"), 3.0)
    client.decide_script[("i3", "t2")] = advancing(failed_exc("connection"), 2.0)
    client.decide_script[("i1", "t2")] = advancing(failed_exc("connection"), 1.0)
    conn = dict(failureKind="connection", status=None)
    ds = [
        disposition(itemId="i3", callKind="handoff", textId=None, **conn),
        disposition(itemId="i3", textId="t1", **conn),
        disposition(itemId="i3", textId="t2", **conn),
        disposition(itemId="i1", textId="t2", **conn),
    ]
    collected = run(client, tmp_path, dispositions=ds)
    failed = {
        (c.itemId, c.callKind, c.textId): c.elapsedSeconds
        for c in collected.calls
        if c.outcome == "failed"
    }
    assert failed == {
        ("i3", "handoff", None): 7.5,
        ("i3", "decision", "t1"): 3.0,
        ("i3", "decision", "t2"): 2.0,
        ("i1", "decision", "t2"): 1.0,
    }
    got = summarize_calls(collected.calls, decision_timeout_seconds=30.0)
    assert got["decision"].k == 3
    assert got["decision"].dispositionItemIds == ("i1", "i3")  # i3 once, sorted
    assert got["handoff"].dispositionItemIds == ("i3",)


def test_there_is_no_retry(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.decide_script[("i2", "t1")] = failed_exc("http", 503)
    with pytest.raises(FitAborted):
        run(client, tmp_path)
    assert [c["input"] for c in client.decide_calls].count("d-i2") == 1
    other = FakeClient(Clock())
    other.chat_script["i2"] = failed_exc("connection")
    with pytest.raises(FitAborted):
        run(other, tmp_path)
    assert [c["user"] for c in other.chat_calls].count("h-i2") == 1
    exempt = FakeClient(Clock())
    exempt.decide_script[("i2", "t2")] = failed_exc("http", 503)
    run(exempt, tmp_path, dispositions=[disposition()])
    assert [(c["input"], c["text"]) for c in exempt.decide_calls].count(("d-i2", "t2")) == 1


def test_wilson_is_never_above_one():
    assert wilson_upper(5, 5) <= 1.0
    assert wilson_upper(1, 1) <= 1.0


# --- a disposition whose call is never made (P53-m1) ---


@pytest.mark.parametrize(
    "ghost",
    [
        disposition(itemId="nope", textId="t2"),
        disposition(textId="no-such-text"),
        disposition(itemId="nope", callKind="handoff", textId=None, failureKind="http", status=500),
    ],
    ids=["ghost-item", "ghost-text", "ghost-handoff-item"],
)
def test_a_disposition_whose_call_is_never_made_refuses(tmp_path, ghost):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.decide_script[("i2", "t2")] = failed_exc("http", 503)  # the real one still fires
    with pytest.raises(ExclusionDidNotFire) as raised:
        run(client, tmp_path, dispositions=[disposition(), ghost])
    assert raised.value.disposition == ghost
    assert raised.value.outcome == "never made"
    assert files(tmp_path) == []


def test_a_decision_disposition_with_no_texts_refuses(tmp_path):  # type: ignore[no-untyped-def]
    with pytest.raises(ExclusionDidNotFire):
        run(FakeClient(Clock()), tmp_path, texts=[], dispositions=[disposition()])
    assert files(tmp_path) == []


def test_every_fired_disposition_passes_the_after_loops_check(tmp_path):  # type: ignore[no-untyped-def]
    client = FakeClient(Clock())
    client.decide_script[("i2", "t2")] = failed_exc("http", 503)
    client.chat_script["i3"] = failed_exc("http", 500)
    d2 = disposition(itemId="i3", callKind="handoff", textId=None, status=500)
    collected = run(client, tmp_path, dispositions=[disposition(), d2])
    assert sum(c.outcome == "failed" for c in collected.calls) == 2


# --- an empty probabilities list (P53-m2) ---


def test_an_ok_choice_call_with_empty_probabilities_has_no_max_entry(tmp_path):  # type: ignore[no-untyped-def]
    labels = ["order", "question", "complaint", "escalation", "other"]
    client = FakeClient(Clock())
    empty = DecisionAnswer("q", "choice", None, "order", (), 0.75)
    client.decide_script[("i1", "t1")] = DecisionResult((empty,), None, None, 100.0)
    client.default_decide = choice_result("order", 0.75)
    collected = run(
        client,
        tmp_path,
        role="message-type-classifier",
        ids=("i1", "i2"),
        texts=choice_texts(labels, "t1"),
    )
    assert collected.maxProbabilities == {("i2", "t1"): 0.5}
    assert ScoredCall("i1", "decision", "t1", "ok", ("order", 0.75), 0.1) in collected.calls
