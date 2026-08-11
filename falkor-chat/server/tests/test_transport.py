"""Unit tests for the unified HTTP transport (K-042 Landing 1, FR-14/FR-10, plan §4.9).

One case per ordered exception-ladder branch (B-2), each driven through an injected
`opener` so no real HTTP call is ever made. Every case asserts the raised
`ProviderCallError` names provider, model, and the resolved URL — never just the cause.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from falkorchat.transport import ProviderCallError, make_http_transport

URL = "http://host:1234/v1/chat/completions"
PROVIDER = "acme-provider"
MODEL = "acme-model"


def _transport(opener, *, timeout=5.0, headers=None):
    return make_http_transport(
        timeout=timeout, headers=headers, opener=opener, provider=PROVIDER, model=MODEL
    )


def _assert_names_provider_model_url(message: str) -> None:
    assert PROVIDER in message, message
    assert MODEL in message, message
    assert URL in message, message


class _FakeResponse:
    """A minimal opener return value: `.read()` -> bytes, `.close()` a no-op."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        pass


# ── rung 1: HTTPError — MUST be caught before URLError (B-2 dead-code regression) ──


def test_http_error_branch_is_reached_first_and_preserves_the_body():
    def opener(req, timeout=None):
        raise urllib.error.HTTPError(
            URL, 401, "Unauthorized", {},
            _FakeResponse(json.dumps({"error": {"message": "bad key"}}).encode()),
        )

    transport = _transport(opener)
    with pytest.raises(ProviderCallError) as excinfo:
        transport(URL, {"model": MODEL})

    message = str(excinfo.value)
    _assert_names_provider_model_url(message)
    assert "bad key" in message  # the body reaches the message — the dead-code regression guard
    assert "401" in message


# ── rung 2: URLError (connection refused, DNS failure, unknown URL type) ──


def test_url_error_branch():
    def opener(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    transport = _transport(opener)
    with pytest.raises(ProviderCallError) as excinfo:
        transport(URL, {"model": MODEL})

    message = str(excinfo.value)
    _assert_names_provider_model_url(message)
    assert "connection refused" in message


# ── rung 3: a BARE TimeoutError — not a URLError, must be named explicitly (B-2) ──


def test_bare_timeout_error_branch_is_not_swallowed_by_url_error():
    def opener(req, timeout=None):
        raise TimeoutError("timed out")

    transport = _transport(opener)
    with pytest.raises(ProviderCallError) as excinfo:
        transport(URL, {"model": MODEL})

    message = str(excinfo.value)
    _assert_names_provider_model_url(message)
    assert "TimeoutError" in message
    assert "timed out" in message


def test_other_os_error_is_also_caught_by_rung_three():
    def opener(req, timeout=None):
        raise ConnectionResetError("connection reset by peer")

    transport = _transport(opener)
    with pytest.raises(ProviderCallError) as excinfo:
        transport(URL, {"model": MODEL})

    _assert_names_provider_model_url(str(excinfo.value))


# ── rung 4: ValueError — malformed-URL shapes (belt and braces) ──


def test_value_error_branch():
    def opener(req, timeout=None):
        raise ValueError("unknown url type: acme")

    transport = _transport(opener)
    with pytest.raises(ProviderCallError) as excinfo:
        transport(URL, {"model": MODEL})

    message = str(excinfo.value)
    _assert_names_provider_model_url(message)
    assert "unknown url type" in message


# ── rung 5: body-level error — string OR object, on the same server (m-1) ──


def test_body_level_error_as_a_bare_string():
    def opener(req, timeout=None):
        return _FakeResponse(
            json.dumps({"error": "Unexpected endpoint or method."}).encode()
        )

    transport = _transport(opener)
    with pytest.raises(ProviderCallError) as excinfo:
        transport(URL, {"model": MODEL})

    message = str(excinfo.value)
    _assert_names_provider_model_url(message)
    assert "Unexpected endpoint" in message


def test_body_level_error_as_an_object():
    def opener(req, timeout=None):
        return _FakeResponse(
            json.dumps({"error": {"message": "No models loaded"}}).encode()
        )

    transport = _transport(opener)
    with pytest.raises(ProviderCallError) as excinfo:
        transport(URL, {"model": MODEL})

    message = str(excinfo.value)
    _assert_names_provider_model_url(message)
    assert "No models loaded" in message


# ── rung 6: body-shape — a 200 that does not decode to a JSON object ──


def test_body_shape_not_an_object():
    def opener(req, timeout=None):
        return _FakeResponse(json.dumps([1, 2, 3]).encode())

    transport = _transport(opener)
    with pytest.raises(ProviderCallError) as excinfo:
        transport(URL, {"model": MODEL})

    _assert_names_provider_model_url(str(excinfo.value))


def test_body_not_valid_json():
    def opener(req, timeout=None):
        return _FakeResponse(b"not json at all")

    transport = _transport(opener)
    with pytest.raises(ProviderCallError) as excinfo:
        transport(URL, {"model": MODEL})

    _assert_names_provider_model_url(str(excinfo.value))


# ── the happy path + construction-time binding ──


def test_success_returns_the_decoded_body():
    def opener(req, timeout=None):
        return _FakeResponse(json.dumps({"choices": []}).encode())

    transport = _transport(opener)
    assert transport(URL, {"model": MODEL}) == {"choices": []}


def test_opener_receives_the_configured_timeout():
    seen: dict = {}

    def opener(req, timeout=None):
        seen["timeout"] = timeout
        return _FakeResponse(b"{}")

    transport = _transport(opener, timeout=42.0)
    transport(URL, {"model": MODEL})
    assert seen["timeout"] == 42.0


def test_extra_headers_reach_the_request():
    seen: dict = {}

    def opener(req, timeout=None):
        seen["headers"] = dict(req.headers)
        return _FakeResponse(b"{}")

    transport = _transport(opener, headers={"Authorization": "Bearer secret-token"})
    transport(URL, {"model": MODEL})

    assert seen["headers"].get("Authorization") == "Bearer secret-token"
    assert seen["headers"].get("Content-type") == "application/json"


def test_payload_is_posted_as_json():
    seen: dict = {}

    def opener(req, timeout=None):
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(b"{}")

    transport = _transport(opener)
    transport(URL, {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})

    assert seen["method"] == "POST"
    assert seen["body"] == {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
