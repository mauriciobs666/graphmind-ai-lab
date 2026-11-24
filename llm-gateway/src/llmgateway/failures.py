"""Failure classification (docs/plans/llm-gateway.md §3.3, FR-7).

`classify_*` return only the `FailureClass`; the error body is read here and dropped, never
returned or logged. Retry and fallback decisions live in the engine, not here.
"""

from __future__ import annotations

import enum
import json
import re
import socket
from collections.abc import Iterable

from llmgateway.types import FailureClass as F

REFUSAL_CONTENT_TYPE = "application/vnd.llmgateway.refusal+json"
REFUSAL_STATUS = 499

# Holder refusal code -> class (§3.3 table). Completed exchanges: none is retried or falls back.
HOLDER_REFUSALS: dict[str, F] = {
    **dict.fromkeys(
        ("bad_credential", "stale_timestamp", "replayed_nonce", "unknown_caller"), F.CONFIG
    ),
    **dict.fromkeys(
        ("endpoint_unknown", "path_not_allowed", "https_required", "address_forbidden"),
        F.POLICY_REFUSED,
    ),
    **dict.fromkeys(
        ("model_not_pinned", "limit_missing", "limit_over_ceiling", "body_invalid"), F.REJECTED
    ),
    **dict.fromkeys(
        (
            "run_id_missing",
            "run_unmatched",
            "run_bound_reached",
            "window_ceiling_reached",
            "release_cap_reached",
        ),
        F.RUN_BOUND,
    ),
    "key_missing": F.CONFIG,
}


class Condition(enum.Enum):
    """Non-HTTP rows of the §3.3 table: conditions the transport layer observes directly."""

    CONNECTION_REFUSED_OR_RESET = "connection refused/reset"
    HOLDER_SOCKET_UNREACHABLE = "holder socket unreachable"
    BROKER_UPSTREAM_ERROR = "broker upstream-error"
    BROKER_RESPONSE_TOO_LARGE = "broker response-too-large"
    SOCKET_TIMEOUT = "socket timeout"
    BROKER_TIMEOUT = "broker timeout"
    STRICT_MODE_MISS = "strict-mode miss"
    BROKER_DENIAL = "broker denial"
    SERVICE_REFUSAL = "service refusal"
    BROKER_UNAVAILABLE = "BrokerUnavailable"


_CONDITION_CLASS: dict[Condition, F] = {
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

_NO_MODEL_LOADED = re.compile(r"no models? (?:are |is )?(?:currently )?loaded", re.IGNORECASE)


def classify_condition(condition: Condition) -> F:
    return _CONDITION_CLASS[condition]


def classify_exception(exc: BaseException) -> F:
    """Map a transport exception. Anything not recognised is `config`: permanent, never retried
    (a caller bug must not be retried or fall back); only timeouts and OS-level connection
    failures are temporary."""
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return classify_condition(Condition.SOCKET_TIMEOUT)
    if isinstance(exc, OSError):  # refused, reset, no such socket, unreachable
        return classify_condition(Condition.CONNECTION_REFUSED_OR_RESET)
    return F.CONFIG


def _media_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _classify_refusal(status: int, body: bytes) -> F:
    """A refusal-typed response: unknown code or malformed exchange fails loudly as `config`."""
    if status != REFUSAL_STATUS:
        return F.CONFIG
    try:
        code = json.loads(body).get("holderRefusal")
    except (ValueError, AttributeError):
        return F.CONFIG
    return HOLDER_REFUSALS.get(code, F.CONFIG) if isinstance(code, str) else F.CONFIG


def _error_type(body: bytes) -> str | None:
    try:
        err = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return None
    return err.get("type") if isinstance(err, dict) and isinstance(err.get("type"), str) else None


def classify_response(
    status: int,
    content_type: str | None = None,
    body: bytes = b"",
    *,
    out_of_credit_patterns: Iterable[str] = (),
) -> F:
    """Classify a completed HTTP exchange that is not a success (status >= 400).

    The holder-refusal content type is checked first (§3.3). `out_of_credit_patterns` are the
    provider's owner-configured case-insensitive substrings, applied to a 400 only.
    """
    if _media_type(content_type) == REFUSAL_CONTENT_TYPE:
        return _classify_refusal(status, body)
    if status < 400:
        raise ValueError(f"status {status} is not a failure")
    text = body.decode("utf-8", errors="replace").lower()
    if _NO_MODEL_LOADED.search(text):
        return F.MODEL_NOT_LOADED
    if status == 402 or _error_type(body) == "billing_error":
        return F.OUT_OF_CREDIT
    if status == 400 and any(p.lower() in text for p in out_of_credit_patterns if p):
        return F.OUT_OF_CREDIT
    if status == 429:
        return F.RATE_LIMIT
    if status == 504:
        return F.TIMEOUT
    if status >= 500:
        return F.OUTAGE
    if status == 401:
        return F.AUTH
    if status == 403:
        return F.PERMISSION
    return F.REJECTED
