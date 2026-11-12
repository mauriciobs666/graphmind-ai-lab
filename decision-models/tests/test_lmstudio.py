"""`decisionmodels.lmstudio`: the fit-only LM Studio client (typed-decision-models plan1 §3.4,
plan2 §3.5, plan3 §3.3/§3.5, U6.1).

Everything here runs against a fake opener: no socket is opened. The over-limit rule is asserted
against the byte-copy of model-bench's `over-limit.jsonl` (`tests/fixtures/decision-contract/`,
guarded by `test_decision_contract.py`), once as the pure predicate and once through the real
`decide` with an `HTTPError` raised by the opener.
"""

from __future__ import annotations

import http.client
import io
import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from decisionmodels import lmstudio
from decisionmodels.lmstudio import (
    LMStudio,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    OverLimit,
)

_OVER_LIMIT_ROWS = [
    json.loads(line)
    for line in (
        Path(__file__).parent / "fixtures" / "decision-contract" / "over-limit.jsonl"
    )
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]

PREDICATE_Q = {"type": "predicate", "name": "advance", "instructions": "Should it advance?"}
CHOICE_Q = {
    "type": "choice",
    "name": "kind",
    "instructions": "Which kind?",
    "choices": [
        {"value": "question", "instructions": "asks"},
        {"value": "statement", "instructions": "states"},
        {"value": "other", "instructions": "else"},
    ],
}
USAGE = {"input_tokens": 41, "output_tokens": 0, "total_tokens": 41}


class FakeResponse:
    def __init__(self, body: bytes, *, read_error: BaseException | None = None) -> None:
        self._body = body
        self._read_error = read_error
        self.closed = False

    def read(self) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._body

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    """Records every request; answers with `reply` (a body, a response, or an exception)."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float] = []
        self.response: FakeResponse | None = None

    def __call__(self, req: urllib.request.Request, *, timeout: float) -> FakeResponse:
        self.requests.append(req)
        self.timeouts.append(timeout)
        if isinstance(self.reply, BaseException):
            raise self.reply
        if isinstance(self.reply, FakeResponse):
            self.response = self.reply
        else:
            body = self.reply if isinstance(self.reply, bytes) else json.dumps(self.reply).encode()
            self.response = FakeResponse(body)
        return self.response

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].data)


def _client(reply: Any, base: str = "http://lm.test:1234") -> tuple[LMStudio, FakeOpener]:
    opener = FakeOpener(reply)
    return LMStudio(base, opener=opener), opener


def _http_error(status: int, body: bytes | None) -> urllib.error.HTTPError:
    """`body=None` is an error body whose read fails."""
    fp: Any = io.BytesIO(body or b"")
    err = urllib.error.HTTPError("http://lm.test/x", status, "boom", {}, fp)  # type: ignore[arg-type]
    if body is None:

        def failing_read(*_a: Any, **_k: Any) -> bytes:
            raise OSError("body read failed")

        err.read = failing_read  # type: ignore[method-assign]
    return err


def _predicate_body(p: float, name: str = "advance") -> dict[str, Any]:
    return {
        "model": "laya",
        "answers": [{"type": "predicate", "name": name, "probability": p}],
        "usage": USAGE,
    }


def _choice_body(choice: str = "other") -> dict[str, Any]:
    return {
        "model": "laya",
        "answers": [
            {
                "type": "choice",
                "name": "kind",
                "choice": choice,
                "probabilities": [
                    {"value": "question", "probability": 0.1},
                    {"value": "statement", "probability": 0.25},
                    {"value": "other", "probability": 0.65},
                ],
                "confidence": 0.65,
            }
        ],
        "usage": USAGE,
    }


# --- decide: request and success ---------------------------------------------------------------


def test_decide_posts_the_pinned_request_to_the_decisions_route() -> None:
    client, opener = _client(_predicate_body(0.8))
    client.decide("some input", [PREDICATE_Q], model="laya", timeout_s=7.5)
    (req,) = opener.requests
    assert req.full_url == "http://lm.test:1234/v1/decisions"
    assert req.get_method() == "POST"
    assert req.get_header("Content-type") == "application/json"
    assert opener.payload == {
        "model": "laya",
        "input": "some input",
        "questions": [PREDICATE_Q],
    }
    assert opener.timeouts == [7.5]


def test_the_timeout_and_model_are_the_callers_each_time() -> None:
    client, opener = _client(_predicate_body(0.5))
    client.decide("a", [PREDICATE_Q], model="m-one", timeout_s=1.25)
    client.decide("b", [PREDICATE_Q], model="m-two", timeout_s=9.0)
    assert opener.timeouts == [1.25, 9.0]
    assert opener.payload["model"] == "m-two"
    assert opener.payload["input"] == "b"


def test_a_trailing_slash_on_the_base_url_is_dropped() -> None:
    client, opener = _client(_predicate_body(0.5), base="http://lm.test:1234/")
    client.decide("a", [PREDICATE_Q], model="laya", timeout_s=1.0)
    assert opener.requests[0].full_url == "http://lm.test:1234/v1/decisions"


def test_a_predicate_answer_is_parsed() -> None:
    client, opener = _client(_predicate_body(0.3125))
    result = client.decide("x", [PREDICATE_Q], model="laya", timeout_s=1.0)
    assert not isinstance(result, OverLimit)
    (answer,) = result.answers
    assert (answer.name, answer.type, answer.probability) == ("advance", "predicate", 0.3125)
    assert answer.choice is None and answer.probabilities is None and answer.confidence is None
    assert result.model == "laya"
    assert result.usage == USAGE
    assert result.wallClockMs is not None and result.wallClockMs >= 0.0
    assert opener.response is not None and opener.response.closed


def test_a_choice_answer_is_parsed_with_its_probabilities_in_order() -> None:
    client, _ = _client(_choice_body("statement"))
    result = client.decide("x", [CHOICE_Q], model="laya", timeout_s=1.0)
    assert not isinstance(result, OverLimit)
    (answer,) = result.answers
    assert answer.choice == "statement"
    assert answer.probabilities == (("question", 0.1), ("statement", 0.25), ("other", 0.65))
    assert answer.confidence == 0.65
    assert answer.probability is None


def test_two_questions_in_one_call_are_answered_in_the_responses_order() -> None:
    body = _choice_body()
    body["answers"].append({"type": "predicate", "name": "advance", "probability": 0.0})
    client, _ = _client(body)
    result = client.decide("x", [PREDICATE_Q, CHOICE_Q], model="laya", timeout_s=1.0)
    assert not isinstance(result, OverLimit)
    assert [a.name for a in result.answers] == ["kind", "advance"]
    assert result.answers[1].probability == 0.0


def test_the_bounds_of_a_probability_are_accepted() -> None:
    for p in (0.0, 1.0, 1):
        client, _ = _client(_predicate_body(p))
        result = client.decide("x", [PREDICATE_Q], model="laya", timeout_s=1.0)
        assert not isinstance(result, OverLimit)
        assert result.answers[0].probability == float(p)


def test_model_and_usage_are_none_when_the_body_omits_or_mistypes_them() -> None:
    body = _predicate_body(0.5)
    body["model"] = 7
    body["usage"] = "lots"
    client, _ = _client(body)
    result = client.decide("x", [PREDICATE_Q], model="laya", timeout_s=1.0)
    assert not isinstance(result, OverLimit)
    assert result.model is None and result.usage is None


# --- decide: a 2xx body that breaks the contract ----------------------------------------------


def _with(body: dict[str, Any], **answer_fields: Any) -> dict[str, Any]:
    body = json.loads(json.dumps(body))
    body["answers"][0].update(answer_fields)
    return body


_BAD_BODIES: list[tuple[str, Any, list[dict[str, Any]]]] = [
    ("not-an-object", [1, 2], [PREDICATE_Q]),
    ("no-answers", {"model": "laya"}, [PREDICATE_Q]),
    ("answers-not-a-list", {"answers": {}}, [PREDICATE_Q]),
    ("answer-not-an-object", {"answers": ["x"]}, [PREDICATE_Q]),
    ("no-name", _with(_predicate_body(0.5), name=""), [PREDICATE_Q]),
    ("score-answer", _with(_predicate_body(0.5), type="score"), [PREDICATE_Q]),
    (
        "score-question-and-answer",
        {"answers": [{"type": "score", "name": "s", "score": 0.5}]},
        [{"type": "score", "name": "s", "instructions": "rate"}],
    ),
    ("unasked-name", _predicate_body(0.5, name="other"), [PREDICATE_Q]),
    ("type-mismatch", _predicate_body(0.5), [{**PREDICATE_Q, "type": "choice", "choices": []}]),
    ("probability-missing", _with(_predicate_body(0.5), probability=None), [PREDICATE_Q]),
    ("probability-string", _with(_predicate_body(0.5), probability="0.5"), [PREDICATE_Q]),
    ("probability-bool", _with(_predicate_body(0.5), probability=True), [PREDICATE_Q]),
    ("probability-above-one", _with(_predicate_body(0.5), probability=1.0001), [PREDICATE_Q]),
    ("probability-negative", _with(_predicate_body(0.5), probability=-0.01), [PREDICATE_Q]),
    ("choice-undeclared", _choice_body("nope"), [CHOICE_Q]),
    ("probabilities-not-a-list", _with(_choice_body(), probabilities="x"), [CHOICE_Q]),
    (
        "probabilities-undeclared-value",
        _with(_choice_body(), probabilities=[{"value": "zzz", "probability": 0.5}]),
        [CHOICE_Q],
    ),
    ("confidence-out-of-range", _with(_choice_body(), confidence=2), [CHOICE_Q]),
    ("question-unanswered", _predicate_body(0.5), [PREDICATE_Q, CHOICE_Q]),
    (
        "answered-twice",
        {
            "answers": [
                {"type": "predicate", "name": "advance", "probability": 0.5},
                {"type": "predicate", "name": "advance", "probability": 0.4},
            ]
        },
        [PREDICATE_Q],
    ),
]


@pytest.mark.parametrize(
    ("reply", "questions"), [(b, q) for _, b, q in _BAD_BODIES], ids=[i for i, _, _ in _BAD_BODIES]
)
def test_a_body_that_breaks_the_contract_is_a_failure_without_a_status(
    reply: Any, questions: list[dict[str, Any]]
) -> None:
    client, _ = _client(reply)
    with pytest.raises(LMStudioCallFailed, match="/v1/decisions") as info:
        client.decide("x", questions, model="laya", timeout_s=1.0)
    assert info.value.status is None
    assert info.value.body is None


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_probability_token_is_refused(token: str) -> None:
    raw = f'{{"answers":[{{"type":"predicate","name":"advance","probability":{token}}}]}}'
    client, _ = _client(raw.encode())
    with pytest.raises(LMStudioCallFailed):
        client.decide("x", [PREDICATE_Q], model="laya", timeout_s=1.0)


def test_an_unparseable_body_is_a_failure_without_a_status() -> None:
    client, _ = _client(b"<html>not json</html>")
    with pytest.raises(LMStudioCallFailed) as info:
        client.decide("x", [PREDICATE_Q], model="laya", timeout_s=1.0)
    assert info.value.status is None and info.value.body is None


# --- over-limit (plan3 §3.3, §3.5) -------------------------------------------------------------


def test_the_marker_is_the_pinned_wording() -> None:
    assert lmstudio.OVER_LIMIT_MARKER == "too large to process"


@pytest.mark.parametrize("row", _OVER_LIMIT_ROWS, ids=[r["case"] for r in _OVER_LIMIT_ROWS])
def test_is_over_limit_matches_over_limit_jsonl(row: dict[str, Any]) -> None:
    assert lmstudio.is_over_limit(row["status"], row["body"]) is row["overLimit"]


_HTTP_ROWS = [r for r in _OVER_LIMIT_ROWS if isinstance(r["status"], int) and r["status"] >= 400]


@pytest.mark.parametrize("row", _HTTP_ROWS, ids=[r["case"] for r in _HTTP_ROWS])
def test_decide_through_an_http_error_follows_over_limit_jsonl(row: dict[str, Any]) -> None:
    body = None if row["body"] is None else row["body"].encode("utf-8")
    client, opener = _client(_http_error(row["status"], body))
    if row["overLimit"]:
        result = client.decide("x " * 800, [PREDICATE_Q], model="laya", timeout_s=3.0)
        assert isinstance(result, OverLimit)
        assert result.wallClockMs >= 0.0
        assert math.isfinite(result.wallClockMs)
        return
    with pytest.raises(LMStudioCallFailed) as info:
        client.decide("x", [PREDICATE_Q], model="laya", timeout_s=3.0)
    assert info.value.status == row["status"]
    assert info.value.body == (row["body"] or "")  # a failed error-body read is ""


def test_an_http_200_error_body_is_a_failure_not_an_over_limit() -> None:
    (row,) = [r for r in _OVER_LIMIT_ROWS if r["case"] == "api-v0-http-200-error-body"]
    client, _ = _client(row["body"].encode())
    with pytest.raises(LMStudioCallFailed) as info:
        client.decide("x", [PREDICATE_Q], model="laya", timeout_s=1.0)
    assert info.value.status is None


def test_an_over_limit_clock_measures_the_failed_call() -> None:
    clock = iter([100.0, 100.25, 999.0])
    original = lmstudio.time.monotonic
    lmstudio.time.monotonic = lambda: next(clock)  # type: ignore[assignment]
    try:
        (row,) = [r for r in _OVER_LIMIT_ROWS if r["case"] == "body-document-at-500"]
        client, _ = _client(_http_error(500, row["body"].encode()))
        result = client.decide("x", [PREDICATE_Q], model="laya", timeout_s=1.0)
    finally:
        lmstudio.time.monotonic = original  # type: ignore[assignment]
    assert result == OverLimit(wallClockMs=250.0)


def _timed(call: Callable[[LMStudio], Any], reply: Any) -> Any:
    clock = iter([10.0, 10.125, 999.0])
    original = lmstudio.time.monotonic
    lmstudio.time.monotonic = lambda: next(clock)  # type: ignore[assignment]
    try:
        client, _ = _client(reply)
        return call(client)
    finally:
        lmstudio.time.monotonic = original  # type: ignore[assignment]


def test_a_successful_decide_reports_the_exact_clock() -> None:
    result = _timed(
        lambda c: c.decide("x", [PREDICATE_Q], model="laya", timeout_s=1.0), _predicate_body(0.5)
    )
    assert result.wallClockMs == 125.0


def test_a_successful_chat_reports_the_exact_clock() -> None:
    result = _timed(
        lambda c: c.chat([], model="m", temperature=0.0, max_tokens=1, timeout_s=1.0),
        _chat_body("ok"),
    )
    assert result.wallClockMs == 125.0


def test_every_question_is_sent() -> None:
    body = _choice_body()
    body["answers"].append({"type": "predicate", "name": "advance", "probability": 0.2})
    client, opener = _client(body)
    client.decide("x", [PREDICATE_Q, CHOICE_Q], model="laya", timeout_s=1.0)
    assert opener.payload["questions"] == [PREDICATE_Q, CHOICE_Q]


def test_chat_reads_the_first_choice() -> None:
    body = _chat_body("first")
    body["choices"].append({"index": 1, "message": {"role": "assistant", "content": "second"}})
    client, _ = _client(body)
    result = client.chat([], model="m", temperature=0.0, max_tokens=1, timeout_s=1.0)
    assert result.content == "first"


def test_the_error_body_is_kept_whole_and_the_message_truncated() -> None:
    long_body = ("x" * 900).encode()
    client, _ = _client(_http_error(502, long_body))
    with pytest.raises(LMStudioCallFailed) as info:
        client.decide("x", [PREDICATE_Q], model="laya", timeout_s=1.0)
    assert info.value.body == "x" * 900
    assert len(str(info.value)) < 700
    assert "HTTP 502" in str(info.value)


def test_a_non_utf8_error_body_is_decoded_with_replacement() -> None:
    client, _ = _client(_http_error(500, b"bad \xff byte"))
    with pytest.raises(LMStudioCallFailed) as info:
        client.decide("x", [PREDICATE_Q], model="laya", timeout_s=1.0)
    assert info.value.body == "bad � byte"


# --- the failure taxonomy, shared by every POST ------------------------------------------------

_TRANSPORT_FAILURES: list[tuple[str, Callable[[], Any], type[Exception]]] = [
    ("timeout", lambda: TimeoutError("slow"), LMStudioCallTimeout),
    ("url-timeout", lambda: urllib.error.URLError(TimeoutError("slow")), LMStudioCallTimeout),
    ("refused", lambda: urllib.error.URLError(ConnectionRefusedError("no")), LMStudioCallFailed),
    ("incomplete-read", lambda: http.client.IncompleteRead(b"x"), LMStudioCallFailed),
    ("socket-error", lambda: ConnectionResetError("reset"), LMStudioCallFailed),
]


@pytest.mark.parametrize(
    ("make", "expected"), [(m, e) for _, m, e in _TRANSPORT_FAILURES],
    ids=[i for i, _, _ in _TRANSPORT_FAILURES],
)
def test_a_transport_failure_is_classified_on_connect(
    make: Callable[[], Any], expected: type[Exception]
) -> None:
    client, _ = _client(make())
    with pytest.raises(expected) as info:
        client.decide("x", [PREDICATE_Q], model="laya", timeout_s=2.5)
    assert type(info.value) is expected
    if expected is LMStudioCallFailed:
        assert info.value.status is None and info.value.body is None  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("make", "expected"), [(m, e) for _, m, e in _TRANSPORT_FAILURES],
    ids=[i for i, _, _ in _TRANSPORT_FAILURES],
)
def test_a_transport_failure_is_classified_on_the_body_read(
    make: Callable[[], Any], expected: type[Exception]
) -> None:
    response = FakeResponse(b"", read_error=make())
    client, _ = _client(response)
    with pytest.raises(expected) as info:
        client.decide("x", [PREDICATE_Q], model="laya", timeout_s=2.5)
    assert type(info.value) is expected
    assert response.closed


def test_the_timeout_message_names_the_budget() -> None:
    client, _ = _client(TimeoutError("slow"))
    with pytest.raises(LMStudioCallTimeout, match="2.5"):
        client.decide("x", [PREDICATE_Q], model="laya", timeout_s=2.5)


def test_every_client_error_is_an_lmstudio_error() -> None:
    assert issubclass(LMStudioCallTimeout, lmstudio.LMStudioError)
    assert issubclass(LMStudioCallFailed, lmstudio.LMStudioError)


# --- chat (the hand-off model's call) ---------------------------------------------------------


def _chat_body(content: Any = "yes") -> dict[str, Any]:
    return {
        "model": "qwen/qwen3-4b-2507",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
    }


def test_chat_posts_the_pinned_request_and_returns_the_content() -> None:
    client, opener = _client(_chat_body("no\n"))
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    result = client.chat(
        messages, model="qwen/qwen3-4b-2507", temperature=0.0, max_tokens=12, timeout_s=4.0
    )
    assert opener.requests[0].full_url == "http://lm.test:1234/api/v0/chat/completions"
    assert opener.payload == {
        "model": "qwen/qwen3-4b-2507",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 12,
    }
    assert opener.timeouts == [4.0]
    assert result.content == "no\n"
    assert result.usage == {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}
    assert result.model == "qwen/qwen3-4b-2507"
    assert result.wallClockMs >= 0.0


def test_chat_parameters_are_the_callers_each_time() -> None:
    client, opener = _client(_chat_body("x"))
    client.chat([{"role": "user", "content": "a"}], model="m1", temperature=0.7, max_tokens=5,
                timeout_s=11.0)
    assert (opener.payload["temperature"], opener.payload["max_tokens"]) == (0.7, 5)
    assert opener.payload["model"] == "m1" and opener.timeouts == [11.0]


@pytest.mark.parametrize(
    "reply",
    [
        {"choices": []},
        {"choices": ["x"]},
        {"choices": [{"message": "x"}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": 5}}]},
        {"nothing": 1},
        [1],
    ],
    ids=["empty", "choice-not-object", "message-not-object", "null-content", "int-content",
         "no-choices", "not-an-object"],
)
def test_a_chat_body_without_usable_content_is_a_failure(reply: Any) -> None:
    client, _ = _client(reply)
    with pytest.raises(LMStudioCallFailed, match="chat/completions") as info:
        client.chat([], model="m", temperature=0.0, max_tokens=1, timeout_s=1.0)
    assert info.value.status is None and info.value.body is None


def test_a_chat_http_error_carries_status_and_body_and_never_an_over_limit() -> None:
    (row,) = [r for r in _OVER_LIMIT_ROWS if r["case"] == "body-document-at-500"]
    client, _ = _client(_http_error(500, row["body"].encode()))
    with pytest.raises(LMStudioCallFailed) as info:
        client.chat([], model="m", temperature=0.0, max_tokens=1, timeout_s=1.0)
    assert info.value.status == 500 and info.value.body == row["body"]


def test_a_chat_timeout_is_a_timeout() -> None:
    client, _ = _client(TimeoutError("slow"))
    with pytest.raises(LMStudioCallTimeout):
        client.chat([], model="m", temperature=0.0, max_tokens=1, timeout_s=1.0)
