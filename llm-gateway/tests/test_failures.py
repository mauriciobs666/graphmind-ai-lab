"""Failure classification: every row of the plan §3.3 tables."""

import json
import socket

import pytest

from llmgateway.failures import (
    HOLDER_REFUSALS,
    REFUSAL_CONTENT_TYPE,
    Condition,
    classify_condition,
    classify_exception,
    classify_response,
)
from llmgateway.types import FailureClass as F

# --- condition table (non-HTTP rows) -------------------------------------------------------

CONDITIONS = {
    Condition.CONNECTION_REFUSED_OR_RESET: F.CONNECTION,
    Condition.HOLDER_SOCKET_UNREACHABLE: F.CONNECTION,
    Condition.BROKER_UPSTREAM_ERROR: F.CONNECTION,
    Condition.BROKER_RESPONSE_TOO_LARGE: F.CONNECTION,
    Condition.SOCKET_TIMEOUT: F.TIMEOUT,
    Condition.BROKER_TIMEOUT: F.TIMEOUT,
    Condition.STRICT_MODE_MISS: F.MODEL_NOT_LOADED,
    Condition.BROKER_DENIAL: F.POLICY_REFUSED,
    Condition.SERVICE_REFUSAL: F.POLICY_REFUSED,
    Condition.BROKER_UNAVAILABLE: F.CONFIG,
}


def test_condition_table_is_complete():
    assert set(CONDITIONS) == set(Condition)


@pytest.mark.parametrize("cond,expected", CONDITIONS.items(), ids=lambda v: str(v))
def test_condition_rows(cond, expected):
    assert classify_condition(cond) is expected


@pytest.mark.parametrize(
    "exc,expected",
    [
        (ConnectionRefusedError(), F.CONNECTION),
        (ConnectionResetError(), F.CONNECTION),
        (FileNotFoundError("holder.sock"), F.CONNECTION),
        (TimeoutError(), F.TIMEOUT),
        (socket.timeout(), F.TIMEOUT),
        (ValueError("caller bug"), F.CONFIG),
    ],
)
def test_exceptions(exc, expected):
    assert classify_exception(exc) is expected


# --- HTTP rows -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [
        (504, F.TIMEOUT),
        (500, F.OUTAGE),
        (502, F.OUTAGE),
        (503, F.OUTAGE),  # includes Ollama queue-full
        (529, F.OUTAGE),
        (599, F.OUTAGE),
        (429, F.RATE_LIMIT),
        (402, F.OUT_OF_CREDIT),
        (401, F.AUTH),
        (403, F.PERMISSION),
        (400, F.REJECTED),
        (404, F.REJECTED),  # Ollama 404
        (408, F.REJECTED),  # "other 4xx"
        (413, F.REJECTED),
        (422, F.REJECTED),
        (499, F.REJECTED),  # no refusal content type: an ordinary 4xx
    ],
)
def test_status_rows(status, expected):
    assert classify_response(status) is expected


def test_context_overflow_and_hosted_ollama_response_format_are_rejected():
    body = b'{"error": {"type": "invalid_request_error", "message": "prompt is too long"}}'
    assert classify_response(400, "application/json", body) is F.REJECTED
    assert classify_response(400, None, b'{"error":"format not supported"}') is F.REJECTED


def test_billing_error_type_is_out_of_credit():
    body = json.dumps({"error": {"type": "billing_error", "message": "x"}}).encode()
    assert classify_response(400, "application/json", body) is F.OUT_OF_CREDIT
    assert classify_response(402, None, b"") is F.OUT_OF_CREDIT


def test_400_matching_owner_pattern_is_out_of_credit_only_for_400():
    body = b'{"error": {"message": "Your credit balance is too low"}}'
    pats = ["credit balance is too low"]
    assert classify_response(400, None, body, out_of_credit_patterns=pats) is F.OUT_OF_CREDIT
    assert classify_response(400, None, body, out_of_credit_patterns=["CREDIT BALANCE"]) is (
        F.OUT_OF_CREDIT
    )
    assert classify_response(400, None, body) is F.REJECTED
    assert classify_response(400, None, body, out_of_credit_patterns=["unrelated"]) is F.REJECTED
    assert classify_response(400, None, body, out_of_credit_patterns=[""]) is F.REJECTED
    assert classify_response(403, None, body, out_of_credit_patterns=pats) is F.PERMISSION
    assert classify_response(422, None, body, out_of_credit_patterns=pats) is F.REJECTED


@pytest.mark.parametrize("status", [400, 404, 500])
def test_lmstudio_no_model_loaded_beats_status(status):
    body = b'{"error": "No models loaded. Please load a model in the developer page."}'
    assert classify_response(status, "application/json", body) is F.MODEL_NOT_LOADED


def test_success_status_is_not_a_failure():
    for status in (200, 204, 301):
        with pytest.raises(ValueError, match="not a failure"):
            classify_response(status)


def test_class_only_is_returned_never_the_body():
    assert isinstance(classify_response(401, None, b"sk-ant-SECRET"), F)


# --- holder refusals -------------------------------------------------------------------------

REFUSAL_TABLE = {
    **dict.fromkeys(
        ["bad_credential", "stale_timestamp", "replayed_nonce", "unknown_caller"], F.CONFIG
    ),
    **dict.fromkeys(
        ["endpoint_unknown", "path_not_allowed", "https_required", "address_forbidden"],
        F.POLICY_REFUSED,
    ),
    **dict.fromkeys(
        ["model_not_pinned", "limit_missing", "limit_over_ceiling", "body_invalid"], F.REJECTED
    ),
    **dict.fromkeys(
        [
            "run_id_missing",
            "run_unmatched",
            "run_bound_reached",
            "window_ceiling_reached",
            "release_cap_reached",
        ],
        F.RUN_BOUND,
    ),
    "key_missing": F.CONFIG,
}


def refusal(code, status=499, content_type=REFUSAL_CONTENT_TYPE):
    return classify_response(status, content_type, json.dumps({"holderRefusal": code}).encode())


def test_refusal_table_is_exactly_the_plan_table():
    assert HOLDER_REFUSALS == REFUSAL_TABLE


@pytest.mark.parametrize("code,expected", REFUSAL_TABLE.items())
def test_each_holder_refusal_code(code, expected):
    assert refusal(code) is expected
    assert not expected.temporary  # none is retried or falls back


def test_refusal_content_type_is_checked_first_and_tolerates_parameters():
    # 499 + refusal type with a body that would otherwise read as no-model-loaded/billing.
    body = b'{"holderRefusal": "run_bound_reached", "x": "no models loaded"}'
    assert classify_response(499, REFUSAL_CONTENT_TYPE + "; charset=utf-8", body) is F.RUN_BOUND
    assert classify_response(499, REFUSAL_CONTENT_TYPE.upper(), body) is F.RUN_BOUND


def test_same_body_without_the_content_type_is_an_ordinary_4xx():
    body = json.dumps({"holderRefusal": "run_bound_reached"}).encode()
    assert classify_response(499, "application/json", body) is F.REJECTED
    assert classify_response(499, None, body) is F.REJECTED


@pytest.mark.parametrize(
    "status,body",
    [
        (499, b'{"holderRefusal": "no_such_code"}'),
        (499, b"not json"),
        (499, b"[]"),
        (499, b'{"holderRefusal": 7}'),
        (499, b"{}"),
        (200, b'{"holderRefusal": "key_missing"}'),
        (429, b'{"holderRefusal": "run_bound_reached"}'),
    ],
)
def test_malformed_or_unknown_refusal_fails_loudly_as_config(status, body):
    assert classify_response(status, REFUSAL_CONTENT_TYPE, body) is F.CONFIG
