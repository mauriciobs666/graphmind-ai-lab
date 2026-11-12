"""The fit-only LM Studio client (typed-decision-models plan1 §3.4, plan2 §3.5, plan3 §3.3/§3.5).

`fit.py` calls three things: the decision model's `POST /v1/decisions` (`LMStudio.decide`), the
hand-off model's chat call (`LMStudio.chat`), each with its own timeout and wall clock, and the
residency read (`LMStudio.residency`, `GET /api/v0/models`, plan4 §3.11). Nothing else: no full
catalog, no load or unload (a human loads both models before a fit; model-bench's runner owns
loading, plan3 §3.7).

This is a standalone copy of the slice of `model-bench/modelbench/lmstudio.py` a fit needs: FR-8
and plan3 §2 forbid importing model-bench. The failure taxonomy is the same, cell for cell:

* `LMStudioCallTimeout` — the budget was exhausted, at the connect or the read phase;
* `LMStudioCallFailed(status, body, kind)` — everything else. `kind` is the closed failure kind:
  `"http"` (the server answered and refused; `status` is set, and only then), `"connection"` (the
  call never completed) or `"malformed"` (a 2xx body that was unusable). `body` is the whole HTTP
  error body (UTF-8, `errors="replace"`, `""` when its read failed) on the `http` rung only,
  `None` elsewhere. All three are required keywords, so a raise site decides.

`failure_kind(exc)` reads the kind from either exception (`"timeout"` for `LMStudioCallTimeout`),
from the exception's type and fields, never from message text (plan4 §3.11 `failureKind`).

**Over-limit** (plan3 §3.3, §3.5). LM Studio refuses an input over the model's physical batch with
HTTP 500 and a body naming it; that is a hard failure, never a truncation. `is_over_limit` is the
one predicate for it, asserted against the byte-copy of model-bench's `over-limit.jsonl`.
`decide` returns `OverLimit(wallClockMs)` for such a call, which the fit's ladders score as plan3
§3.5 says (a hand-off in every band; fail-closed or concordant when there is no hand-off model).
Every other failure raises. `chat` never returns an over-limit: nothing in the plans gives a chat
call one.

Route: `/v1/decisions`, from the live probe of 2026-10-30 (there is no `/api/v0/decisions`).
`output_tokens` in `usage` is always 0 there. A `score` answer was never observed and is refused.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DECISIONS_ROUTE = "/v1/decisions"
_CHAT_ROUTE = "/api/v0/chat/completions"
_MODELS_ROUTE = "/api/v0/models"
_NOT_LOADED = "not-loaded"

#: plan3 §3.2: a call is over-limit iff its status is 500 and its error body contains this,
#: case-sensitively. Any other 500 stays a failure.
OVER_LIMIT_MARKER = "too large to process"

_DECISION_ANSWER_TYPES = frozenset({"predicate", "choice"})
_MESSAGE_LIMIT = 500


KIND_TIMEOUT = "timeout"
KIND_CONNECTION = "connection"
KIND_HTTP = "http"
KIND_MALFORMED = "malformed"
_FAILED_KINDS = frozenset({KIND_CONNECTION, KIND_HTTP, KIND_MALFORMED})


class LMStudioError(RuntimeError):
    """Base for every client failure. Never raised directly."""


class LMStudioCallTimeout(LMStudioError):
    """A call exceeded its `timeout_s`: a censored observation, not a refusal."""

    kind = KIND_TIMEOUT


class LMStudioCallFailed(LMStudioError):
    """A call returned no usable response for a reason other than a timeout. `kind` says which:
    `http` (`status` is set, and only then), `connection` or `malformed`; see the module
    docstring for `status` and `body`. `wallClockMs` is the elapsed time to the error, set on the
    HTTP-error rung only (it is what `OverLimit` reports)."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None,
        body: str | None,
        kind: str,
        wallClockMs: float | None = None,
    ) -> None:
        super().__init__(message)
        if kind not in _FAILED_KINDS:
            raise ValueError(f"kind must be one of {sorted(_FAILED_KINDS)}, not {kind!r}")
        if (kind == KIND_HTTP) != (status is not None):
            raise ValueError(f"kind 'http' iff a status is set (kind={kind!r}, status={status!r})")
        self.kind = kind
        self.status = status
        self.body = body
        self.wallClockMs = wallClockMs


def failure_kind(exc: LMStudioError) -> str:
    """The closed failure kind of a client exception: `timeout`, `connection`, `http` or
    `malformed`. One accessor for both exception types; anything else is a `TypeError`."""
    if isinstance(exc, (LMStudioCallTimeout, LMStudioCallFailed)):
        return exc.kind
    raise TypeError(f"not a client failure: {type(exc).__name__}")


@dataclass(frozen=True)
class DecisionAnswer:
    """One answer. A predicate carries `probability` only; a choice carries `choice`,
    `probabilities` (`(value, probability)` in the response's order) and `confidence`. The other
    type's fields are `None`."""

    name: str
    type: str
    probability: float | None
    choice: str | None
    probabilities: tuple[tuple[str, float], ...] | None
    confidence: float | None


@dataclass(frozen=True)
class DecisionResult:
    answers: tuple[DecisionAnswer, ...]
    model: str | None
    usage: Mapping[str, Any] | None
    wallClockMs: float


@dataclass(frozen=True)
class OverLimit:
    """A decide call LM Studio refused as over the physical batch; `wallClockMs` is the failed
    call's time."""

    wallClockMs: float


@dataclass(frozen=True)
class ResidentModel:
    """One `/api/v0/models` row whose `state` is not `"not-loaded"`: `{id, state}`, the literal
    `state` string kept (the bench's shape; any value but `"not-loaded"` means resident)."""

    id: str
    state: str


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str | None
    usage: Mapping[str, Any] | None
    wallClockMs: float


def is_over_limit(status: int | None, body: str | None) -> bool:
    """plan3 §3.2's rule, pure: HTTP 500 and the marker, case-sensitive, in the body."""
    return status == 500 and body is not None and OVER_LIMIT_MARKER in body


class _Malformed(ValueError):
    """A 2xx body that breaks the contract; `decide`/`chat` turn it into `LMStudioCallFailed`."""


def _probability(raw: Any, *, where: str) -> float:
    # A bool, a string, NaN or inf is refused, never coerced; the range test refuses the
    # non-finite ones (every comparison with NaN is false, inf is out of range).
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _Malformed(f"{where} is not a number: {raw!r}")
    try:
        value = float(raw)
    except OverflowError:  # json accepts integers of up to 4,300 digits
        raise _Malformed(f"{where} is an integer too large to be a probability") from None
    if not 0.0 <= value <= 1.0:
        raise _Malformed(f"{where} is not a probability in [0, 1]: {raw!r}")
    return value


def _unhashable(raw: Any) -> bool:
    """A JSON array or object: a set-membership test on it raises a bare `TypeError`, so each
    membership test below asks this first and refuses it as malformed."""
    return isinstance(raw, (list, dict))


def _parse_answers(
    body: Any, questions: Sequence[Mapping[str, Any]]
) -> tuple[DecisionAnswer, ...]:
    """Every question answered exactly once, by an answer of its own type, a choice naming one of
    its declared values. Anything else raises: a partial answer is a decision nobody made."""
    if not isinstance(body, Mapping):
        raise _Malformed("response body is not a JSON object")
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, list):
        raise _Malformed("response has no 'answers' list")
    asked = {q.get("name"): q for q in questions}
    answers: list[DecisionAnswer] = []
    for index, raw in enumerate(raw_answers):
        where = f"answers[{index}]"
        if not isinstance(raw, Mapping):
            raise _Malformed(f"{where} is not a JSON object")
        name, kind = raw.get("name"), raw.get("type")
        if not isinstance(name, str) or not name:
            raise _Malformed(f"{where} has no name")
        if _unhashable(kind) or kind not in _DECISION_ANSWER_TYPES:
            raise _Malformed(f"{where} ({name!r}) has an unrecognized type {kind!r}")
        question = asked.get(name)
        if question is None:
            raise _Malformed(f"{where} answers {name!r}, which was not asked")
        if question.get("type") != kind:
            raise _Malformed(
                f"{where} ({name!r}) is a {kind!r} answer to a {question.get('type')!r} question"
            )
        if kind == "predicate":
            answers.append(
                DecisionAnswer(
                    name=name,
                    type=kind,
                    probability=_probability(raw.get("probability"), where=f"{where}.probability"),
                    choice=None,
                    probabilities=None,
                    confidence=None,
                )
            )
            continue
        declared = {c.get("value") for c in question.get("choices") or () if isinstance(c, Mapping)}
        choice = raw.get("choice")
        if _unhashable(choice) or choice not in declared:
            raise _Malformed(f"{where}.choice {choice!r} is not a declared value")
        raw_probabilities = raw.get("probabilities")
        if not isinstance(raw_probabilities, list):
            raise _Malformed(f"{where}.probabilities is not a list")
        probabilities: list[tuple[str, float]] = []
        for j, entry in enumerate(raw_probabilities):
            at = f"{where}.probabilities[{j}]"
            value = entry.get("value") if isinstance(entry, Mapping) else None
            if _unhashable(value) or value not in declared:
                raise _Malformed(f"{at}.value {value!r} is not a declared value")
            probabilities.append((value, _probability(entry.get("probability"), where=at)))
        answers.append(
            DecisionAnswer(
                name=name,
                type=kind,
                probability=None,
                choice=choice,
                probabilities=tuple(probabilities),
                confidence=_probability(raw.get("confidence"), where=f"{where}.confidence"),
            )
        )
    answered = [a.name for a in answers]
    if len(answered) != len(set(answered)):
        raise _Malformed(f"a question was answered more than once: {answered}")
    if set(answered) != set(asked):
        raise _Malformed(f"unanswered question(s): {sorted(set(asked) - set(answered), key=str)}")
    return tuple(answers)


def _truncate(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    return text if len(text) <= _MESSAGE_LIMIT else text[:_MESSAGE_LIMIT] + "…"


class LMStudio:
    """`base_url` is the caller's (a plain parameter; this module reads no config). `opener`
    defaults to `urllib.request.urlopen`; every offline test passes a fake, so no socket opens
    outside the one `-m live` test."""

    def __init__(
        self,
        base_url: str,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._opener = opener

    def _send(
        self, req: urllib.request.Request, label: str, timeout_s: float
    ) -> tuple[float, bytes]:
        """`(wallClockMs, raw body)` on a 2xx, the clock stopped at the last byte of the body.
        `label` (`"POST /path"`) prefixes every message."""
        start = time.monotonic()
        try:
            resp = self._opener(req, timeout=timeout_s)
            try:
                raw = resp.read()
            finally:
                resp.close()
        except urllib.error.HTTPError as exc:  # a URLError subclass: must come first
            try:
                try:
                    body = exc.read()
                except (TimeoutError, http.client.HTTPException, OSError):
                    body = b""
            finally:
                exc.close()
            raise LMStudioCallFailed(
                f"{label}: HTTP {exc.code}: {_truncate(body)}",
                status=exc.code,
                body=body.decode("utf-8", errors="replace"),
                kind=KIND_HTTP,
                wallClockMs=(time.monotonic() - start) * 1000.0,
            ) from exc
        except TimeoutError as exc:  # connect- and read-phase alike; not a URLError
            raise LMStudioCallTimeout(f"{label}: timed out after {timeout_s}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise LMStudioCallTimeout(f"{label}: timed out after {timeout_s}s") from exc
            raise LMStudioCallFailed(
                f"{label}: connection failed: {exc.reason}",
                status=None,
                body=None,
                kind=KIND_CONNECTION,
            ) from exc
        except http.client.HTTPException as exc:  # e.g. IncompleteRead; not an OSError
            raise LMStudioCallFailed(
                f"{label}: {type(exc).__name__}: {exc}",
                status=None,
                body=None,
                kind=KIND_CONNECTION,
            ) from exc
        except OSError as exc:
            raise LMStudioCallFailed(
                f"{label}: {type(exc).__name__}: {exc}",
                status=None,
                body=None,
                kind=KIND_CONNECTION,
            ) from exc
        return (time.monotonic() - start) * 1000.0, raw

    @staticmethod
    def _parse(raw: bytes, label: str) -> Any:
        try:
            return json.loads(raw)
        except (ValueError, RecursionError) as exc:  # RecursionError: a deeply nested body
            raise LMStudioCallFailed(
                f"{label}: response is not JSON: {_truncate(raw)}",
                status=None,
                body=None,
                kind=KIND_MALFORMED,
            ) from exc

    def _post(
        self, path: str, payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[float, Any]:
        """`(wallClockMs, parsed JSON)` on a 2xx whose body parses. A body that is not JSON is
        `LMStudioCallFailed(kind="malformed")`."""
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        label = f"POST {path}"
        wall_clock_ms, raw = self._send(req, label, timeout_s)
        return wall_clock_ms, self._parse(raw, label)

    def residency(self, *, timeout_s: float = 10.0) -> list[ResidentModel]:
        """The models LM Studio reports resident: `GET /api/v0/models` rows whose `state` is not
        `"not-loaded"`, in the response's order (a port of the bench's `residency`, plan4 §3.11).
        `[]` is a real answer (nothing loaded). A body with no `data` list, or a row that is not
        an object with a string `id` and `state`, is `LMStudioCallFailed(kind="malformed")`; a
        timeout is `LMStudioCallTimeout`; an HTTP error or no connection is `LMStudioCallFailed`
        of kind `http` or `connection`. Ids are
        compared by the caller, exactly (no case folding)."""
        label = f"GET {_MODELS_ROUTE}"
        req = urllib.request.Request(f"{self._base_url}{_MODELS_ROUTE}", method="GET")
        _, raw = self._send(req, label, timeout_s)
        body = self._parse(raw, label)
        rows = body.get("data") if isinstance(body, Mapping) else None
        if not isinstance(rows, list):
            raise LMStudioCallFailed(
                f"{label}: response has no 'data' list",
                status=None,
                body=None,
                kind=KIND_MALFORMED,
            )
        resident: list[ResidentModel] = []
        for index, row in enumerate(rows):
            model_id = row.get("id") if isinstance(row, Mapping) else None
            state = row.get("state") if isinstance(row, Mapping) else None
            if not isinstance(model_id, str) or not isinstance(state, str):
                raise LMStudioCallFailed(
                    f"{label}: data[{index}] has no string 'id' and 'state'",
                    status=None,
                    body=None,
                    kind=KIND_MALFORMED,
                )
            if state != _NOT_LOADED:
                resident.append(ResidentModel(id=model_id, state=state))
        return resident

    def decide(
        self,
        input: str,
        questions: Sequence[Mapping[str, Any]],
        *,
        model: str,
        timeout_s: float,
    ) -> DecisionResult | OverLimit:
        """One typed-decision call. Returns `OverLimit` for the over-limit refusal (plan3 §3.5);
        raises `LMStudioCallTimeout` on an exhausted budget and `LMStudioCallFailed` for every
        other failure, including a 2xx body that breaks the contract (`kind="malformed"`).
        `input` shadows the builtin deliberately: it is the request's own field name."""
        payload = {"model": model, "input": input, "questions": list(questions)}
        try:
            wall_clock_ms, body = self._post(DECISIONS_ROUTE, payload, timeout_s)
        except LMStudioCallFailed as exc:
            if is_over_limit(exc.status, exc.body):
                assert exc.wallClockMs is not None  # set on every HTTP-error raise
                return OverLimit(wallClockMs=exc.wallClockMs)
            raise
        try:
            answers = _parse_answers(body, questions)
        except _Malformed as exc:
            raise LMStudioCallFailed(
                f"POST {DECISIONS_ROUTE}: {exc}", status=None, body=None, kind=KIND_MALFORMED
            ) from exc
        usage, response_model = body.get("usage"), body.get("model")
        return DecisionResult(
            answers=answers,
            model=response_model if isinstance(response_model, str) else None,
            usage=usage if isinstance(usage, Mapping) else None,
            wallClockMs=wall_clock_ms,
        )

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> ChatResult:
        """The hand-off model's call: `POST /api/v0/chat/completions`, the first choice's text."""
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        wall_clock_ms, body = self._post(_CHAT_ROUTE, payload, timeout_s)
        choices = body.get("choices") if isinstance(body, Mapping) else None
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], Mapping)
            else None
        )
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise LMStudioCallFailed(
                f"POST {_CHAT_ROUTE}: response has no usable message content",
                status=None,
                body=None,
                kind=KIND_MALFORMED,
            )
        usage, response_model = body.get("usage"), body.get("model")
        return ChatResult(
            content=content,
            model=response_model if isinstance(response_model, str) else None,
            usage=usage if isinstance(usage, Mapping) else None,
            wallClockMs=wall_clock_ms,
        )
