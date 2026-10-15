"""Unit tests for `falkorchat.retry` — the K-067 transient-error classifier and the
one retry helper (plan `falkor-chat/docs/plans/extract-transient-retry.md` §3.1,
§3.2, §3.6; test rows U-5…U-8c in §5 step 1). No FalkorDB, no real sleep.
"""

from __future__ import annotations

import logging

import pytest

from falkorchat.modelconfig import ModelResolutionError
from falkorchat.repository import EmbeddingDimensionError
from falkorchat.retry import (
    DEFAULT_PROVIDER_RETRY,
    EMBED_CHUNK_RETRY,
    RetryPolicy,
    call_with_retry,
    is_transient,
)
from falkorchat.transport import ProviderCallError


def _pce(kind=None, status=None, detail=None, causes=()):
    return ProviderCallError("p/m @ url: boom", kind=kind, status=status, detail=detail,
                             causes=causes)


# Verbatim LM Studio shapes (plan §2.2).
_TERMINATED = '{"error":"terminated"}'
_LOAD_FAILED = ('Failed to load model "qwen/qwen3-4b-2507". Error: Engine protocol '
                'startup was aborted.')
_CONTEXT_OVERFLOW = (
    '{"error":{"code":400,"message":"the request exceeds the available context size, '
    'try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9000,'
    '"n_ctx":4096}}'
)
_JINJA = ('{"error":"Error rendering prompt with jinja template: \\"Only user, assistant '
          'and tool roles are supported!\\"."}')


# ── U-5: is_transient truth table ──

_TRANSIENT = {
    "http 500": _pce("http", 500),
    "http 502": _pce("http", 502),
    "http 503": _pce("http", 503),
    "http 599 (5xx upper bound)": _pce("http", 599),
    "http 408": _pce("http", 408),
    "http 429": _pce("http", 429),
    "http 400 terminated": _pce("http", 400, _TERMINATED),
    "http 400 failed to load model": _pce("http", 400, _LOAD_FAILED),
    # Each marker also gets a row carrying it alone, so no marker is masked by another
    # (the verbatim load-failure body above also contains the "engine protocol" one).
    "http 400 failed to load model only": _pce(
        "http", 400, '{"error":"Failed to load model \\"qwen/qwen3-4b-2507\\"."}'),
    "http 400 error loading model": _pce("http", 400, "Error loading model"),
    "http 400 engine startup aborted only": _pce(
        "http", 400, "Engine protocol startup was aborted"),
    "http 400 channel error": _pce("http", 400, "Channel Error"),
    "http 400 TERMINATED uppercase": _pce("http", 400, '{"error":"TERMINATED"}'),
    "connection": _pce("connection"),
    "network": _pce("network"),
    "read": _pce("read"),
    "body_error terminated": _pce("body_error", detail="terminated"),
    "fallback_exhausted all transient": _pce(
        "fallback_exhausted", causes=(_pce("http", 503), _pce("connection"))),
}

_PERMANENT = {
    "http 400 context overflow": _pce("http", 400, _CONTEXT_OVERFLOW),
    "http 400 jinja roles": _pce("http", 400, _JINJA),
    "http 401": _pce("http", 401, '{"error":"unauthorized"}'),
    "http 404": _pce("http", 404),
    "http 422": _pce("http", 422),
    "http 400 detail None": _pce("http", 400, None),
    "timeout": _pce("timeout"),
    "malformed_request": _pce("malformed_request"),
    "invalid_json": _pce("invalid_json"),
    "body_shape": _pce("body_shape"),
    "response_shape": _pce("response_shape"),
    "kind None": ProviderCallError("x"),
    "body_error no marker": _pce("body_error", detail="No models loaded"),
    "fallback_exhausted one permanent cause": _pce(
        "fallback_exhausted", causes=(_pce("http", 503), _pce("http", 401))),
    "fallback_exhausted empty causes": _pce("fallback_exhausted", causes=()),
    "RuntimeError": RuntimeError("terminated"),
    "raw ConnectionError": ConnectionError("reset"),
    "ValueError": ValueError("bad"),
    "EmbeddingDimensionError": EmbeddingDimensionError("dim"),
    "ModelResolutionError": ModelResolutionError("unknown ref"),
}


@pytest.mark.parametrize("exc", list(_TRANSIENT.values()), ids=list(_TRANSIENT))
def test_u5_transient(exc):
    assert is_transient(exc) is True


@pytest.mark.parametrize("exc", list(_PERMANENT.values()), ids=list(_PERMANENT))
def test_u5_permanent(exc):
    assert is_transient(exc) is False


# ── U-5b: the timeout switch ──


def test_u5b_timeout_is_permanent_by_default_and_transient_when_switched_on():
    exc = _pce("timeout")
    assert is_transient(exc) is False
    assert is_transient(exc, retry_timeouts=True) is True


def test_u5b_fallback_exhausted_of_timeouts_follows_the_switch():
    exc = _pce("fallback_exhausted", causes=(_pce("timeout"), _pce("timeout")))
    assert is_transient(exc) is False
    assert is_transient(exc, retry_timeouts=True) is True


def test_u5b_both_shipped_policies_leave_timeout_retry_off():
    # A plan §3.2.1 flip edits this row deliberately (EMBED_CHUNK_RETRY → True),
    # hard-coded; it also turns test_embedding.py's U-17 shipped-policy row red by
    # design (see the comment there).
    assert DEFAULT_PROVIDER_RETRY.retry_timeouts is False
    assert EMBED_CHUNK_RETRY.retry_timeouts is False


# ── call_with_retry helpers ──

_LOGGER = "falkorchat.retry"


class _Script:
    """`fn` for call_with_retry: each call pops the next outcome; an exception
    instance is raised, anything else is returned."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeSleep:
    def __init__(self, events=None):
        self.delays = []
        self._events = events

    def __call__(self, delay):
        self.delays.append(delay)
        if self._events is not None:
            self._events.append("sleep")


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == _LOGGER and r.levelno == logging.WARNING]


def _infos(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == _LOGGER and r.levelno == logging.INFO]


# ── U-6: happy-path retry ──


def test_u6_transient_twice_then_success_returns_the_value_after_three_calls(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    fn = _Script(_pce("http", 503), _pce("connection"), "ok")
    sleep = _FakeSleep()

    result = call_with_retry(fn, sleep=sleep, rand=lambda: 0.0, label="extract chunkId=c1")

    assert result == "ok"
    assert fn.calls == 3
    assert sleep.delays == [2.0, 8.0]
    warnings = _warnings(caplog)
    assert len(warnings) == 2
    assert "attempt 1/4" in warnings[0]
    assert "kind=http" in warnings[0] and "status=503" in warnings[0]
    assert "extract chunkId=c1" in warnings[0]
    assert "attempt 2/4" in warnings[1]
    assert "kind=connection" in warnings[1] and "status=None" in warnings[1]


def test_u6_jitter_is_upward_only_floor_at_the_nominal_delay():
    fn = _Script(_pce("http", 503), _pce("http", 503), "ok")
    sleep = _FakeSleep()

    call_with_retry(fn, sleep=sleep, rand=lambda: 0.9999999)

    assert sleep.delays[0] >= 2.0 and sleep.delays[0] < 3.0
    assert sleep.delays[1] >= 8.0 and sleep.delays[1] < 12.0
    assert sleep.delays[0] > 2.9  # the jitter is actually applied, not ignored


def test_u6_success_on_first_call_never_sleeps():
    sleep = _FakeSleep()
    assert call_with_retry(_Script("ok"), sleep=sleep) == "ok"
    assert sleep.delays == []


# ── U-7: budget exhausted ──


def test_u7_always_transient_is_called_max_attempts_and_reraises_the_last_error(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    errors = [_pce("http", 500 + i) for i in range(4)]
    fn = _Script(*errors)
    sleep = _FakeSleep()

    with pytest.raises(ProviderCallError) as excinfo:
        call_with_retry(fn, sleep=sleep, rand=lambda: 0.0, label="extract chunkId=c7")

    assert fn.calls == 4
    assert sleep.delays == [2.0, 8.0, 30.0]  # every derived point, incl. the 30 s one
    assert excinfo.value is errors[3]
    giving_up = [w for w in _warnings(caplog) if "giving up" in w]
    assert len(giving_up) == 1
    assert "giving up after 4 attempts" in giving_up[0]
    assert "extract chunkId=c7" in giving_up[0]
    assert "kind=http" in giving_up[0] and "status=503" in giving_up[0]


# ── U-8: permanent — no retry ──


def test_u8_permanent_provider_error_is_reraised_after_one_call(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    err = _pce("http", 400, '{"error":{"type":"exceed_context_size_error"}}')
    fn = _Script(err)
    sleep = _FakeSleep()

    with pytest.raises(ProviderCallError) as excinfo:
        call_with_retry(fn, sleep=sleep)

    assert excinfo.value is err
    assert fn.calls == 1
    assert sleep.delays == []
    assert _warnings(caplog) == []


def test_u8_non_provider_error_is_reraised_after_one_call():
    err = RuntimeError("bug")
    fn = _Script(err)
    sleep = _FakeSleep()

    with pytest.raises(RuntimeError) as excinfo:
        call_with_retry(fn, sleep=sleep)

    assert excinfo.value is err
    assert fn.calls == 1
    assert sleep.delays == []


def test_u8_base_exception_propagates_immediately():
    fn = _Script(KeyboardInterrupt())
    sleep = _FakeSleep()
    with pytest.raises(KeyboardInterrupt):
        call_with_retry(fn, sleep=sleep)
    assert fn.calls == 1
    assert sleep.delays == []


# ── U-8b: document-liveness gate (pre-sleep read, then post-sleep read; M-1) ──


class _Gate:
    """`keep_going` scripted per call; records a "gate" event per read."""

    def __init__(self, *answers, events=None):
        self._answers = list(answers)
        self.calls = 0
        self._events = events

    def __call__(self):
        self.calls += 1
        if self._events is not None:
            self._events.append("gate")
        answer = self._answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def test_u8b_pre_sleep_gate_false_stops_before_any_sleep(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    err = _pce("http", 503)
    fn = _Script(err, "never reached")
    sleep = _FakeSleep()
    gate = _Gate(False)

    with pytest.raises(ProviderCallError) as excinfo:
        call_with_retry(fn, sleep=sleep, keep_going=gate, label="extract chunkId=c1")

    assert excinfo.value is err
    assert fn.calls == 1
    assert sleep.delays == []
    infos = _infos(caplog)
    assert len(infos) == 1
    assert "extract chunkId=c1" in infos[0]
    assert "owning document no longer processing; abandoning retry" in infos[0]
    assert "kind=http" in infos[0] and "status=503" in infos[0]
    assert _warnings(caplog) == []  # the gate is read before the retry WARNING


def test_u8b_post_sleep_gate_false_stops_without_another_attempt(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    err = _pce("http", 503)
    fn = _Script(err, "never reached")
    sleep = _FakeSleep()
    gate = _Gate(True, False)

    with pytest.raises(ProviderCallError) as excinfo:
        call_with_retry(fn, sleep=sleep, keep_going=gate, label="embed chunkId=c1")

    assert excinfo.value is err
    assert fn.calls == 1
    assert len(sleep.delays) == 1
    infos = _infos(caplog)
    assert len(infos) == 1 and "embed chunkId=c1" in infos[0]


def test_u8b_gate_is_read_before_and_after_each_sleep():
    events: list[str] = []
    fn = _Script(_pce("http", 503), _pce("http", 503), "ok")
    sleep = _FakeSleep(events)
    gate = _Gate(True, True, True, True, events=events)

    assert call_with_retry(fn, sleep=sleep, keep_going=gate) == "ok"
    assert events == ["gate", "sleep", "gate", "gate", "sleep", "gate"]


def test_u8b_gate_that_raises_is_treated_as_keep_going_and_logged(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    fn = _Script(_pce("http", 503), "ok")
    sleep = _FakeSleep()
    gate = _Gate(RuntimeError("graph down"), RuntimeError("graph down"))

    assert call_with_retry(fn, sleep=sleep, keep_going=gate) == "ok"
    assert fn.calls == 2
    assert gate.calls == 2
    errors = [r for r in caplog.records
              if r.name == _LOGGER and r.levelno == logging.ERROR]
    assert len(errors) == 2
    assert all(r.exc_info is not None for r in errors)


def test_u8b_gate_is_never_read_on_success():
    gate = _Gate()
    assert call_with_retry(_Script("ok"), sleep=_FakeSleep(), keep_going=gate) == "ok"
    assert gate.calls == 0


def test_u8b_gate_is_never_read_on_a_permanent_error():
    gate = _Gate()
    with pytest.raises(ProviderCallError):
        call_with_retry(_Script(_pce("http", 401)), sleep=_FakeSleep(), keep_going=gate)
    assert gate.calls == 0


def test_u8b_gate_is_never_read_after_the_final_attempt():
    gate = _Gate(*[True] * 6)
    fn = _Script(*[_pce("http", 503) for _ in range(4)])
    with pytest.raises(ProviderCallError):
        call_with_retry(fn, sleep=_FakeSleep(), keep_going=gate)
    assert fn.calls == 4
    assert gate.calls == 6  # pre+post for each of the 3 sleeps, none after attempt 4


# ── U-8c: RetryPolicy validation (plan §3.2 invariant) ──


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_attempts": 4, "delays_s": (2.0, 8.0)}, "delays_s"),
        ({"max_attempts": 2, "delays_s": (2.0, 8.0)}, "delays_s"),
        ({"max_attempts": 0, "delays_s": ()}, "max_attempts"),
        ({"max_attempts": 1, "delays_s": (), "jitter": -0.1}, "jitter"),
    ],
    ids=["too-few-delays", "too-many-delays", "zero-attempts", "negative-jitter"],
)
def test_u8c_invalid_policy_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RetryPolicy(**kwargs)


def test_u8c_single_attempt_policy_is_valid_and_never_retries():
    no_retry = RetryPolicy(max_attempts=1, delays_s=())
    err = _pce("http", 503)
    fn = _Script(err, "never reached")
    sleep = _FakeSleep()

    with pytest.raises(ProviderCallError) as excinfo:
        call_with_retry(fn, policy=no_retry, sleep=sleep)

    assert excinfo.value is err
    assert fn.calls == 1
    assert sleep.delays == []


def test_u8c_shipped_policies_match_the_plan():
    for policy in (DEFAULT_PROVIDER_RETRY, EMBED_CHUNK_RETRY):
        assert policy.max_attempts == 4
        assert policy.delays_s == (2.0, 8.0, 30.0)
        assert policy.jitter == 0.5


def test_u5b_call_with_retry_passes_the_policy_timeout_switch_through():
    # The switch is plumbed now so a plan §3.2.1 flip is one constant (U-17, step 3).
    on = RetryPolicy(max_attempts=2, delays_s=(2.0,), retry_timeouts=True)
    fn = _Script(_pce("timeout"), "ok")
    assert call_with_retry(fn, policy=on, sleep=_FakeSleep()) == "ok"
    assert fn.calls == 2

    off = RetryPolicy(max_attempts=2, delays_s=(2.0,))
    fn = _Script(_pce("timeout"), "ok")
    with pytest.raises(ProviderCallError):
        call_with_retry(fn, policy=off, sleep=_FakeSleep())
    assert fn.calls == 1
