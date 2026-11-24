"""Shared value types and errors (docs/plans/llm-gateway.md §3.2, §3.3).

Stdlib only. `Request` has no model or provider field: the call site's owner configuration
chooses both (FR-2). Unknown request fields are refused, so none can carry a scope marker or
express a loosening (AC-2).
"""

from __future__ import annotations

import enum
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

# Neutral generation settings (FR-4, §3.2); `response_format` and `tools` pass through as
# their own Request fields, not as settings.
NEUTRAL_SETTINGS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
        "seed",
        "stop",
    }
)


# The plain-identifier alphabet shared by `Request.run_id` and the call-record columns that hold
# an identifier (`records.py`): a value outside it is refused before a send, never at write time.
IDENT = re.compile(r"[A-Za-z0-9_:.\-/@+]{1,200}")


class FailureClass(enum.StrEnum):
    """Provider-independent failure classes (FR-7, §3.3)."""

    # Temporary: retry/fallback may apply.
    RATE_LIMIT = "rate_limit"
    OUT_OF_CREDIT = "out_of_credit"
    OUTAGE = "outage"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    MODEL_NOT_LOADED = "model_not_loaded"
    # Permanent: the call fails naming the cause.
    AUTH = "auth"
    PERMISSION = "permission"
    REJECTED = "rejected"
    POLICY_REFUSED = "policy_refused"
    RUN_BOUND = "run_bound"
    CONFIG = "config"

    @property
    def temporary(self) -> bool:
        return self in TEMPORARY


TEMPORARY: frozenset[FailureClass] = frozenset(
    {
        FailureClass.RATE_LIMIT,
        FailureClass.OUT_OF_CREDIT,
        FailureClass.OUTAGE,
        FailureClass.TIMEOUT,
        FailureClass.CONNECTION,
        FailureClass.MODEL_NOT_LOADED,
    }
)
PERMANENT: frozenset[FailureClass] = frozenset(FailureClass) - TEMPORARY


class LoadMode(enum.StrEnum):
    """Local-model loading modes (FR-16, §3.8)."""

    ON_DEMAND = "on_demand"
    EXPLICIT = "explicit"
    STRICT = "strict"


class GatewayError(Exception):
    """Base of every gateway error."""


class EmbeddingsNotAvailable(GatewayError):
    """Embeddings are refused at first delivery (FR-1, AC-23)."""


class HostedFromCallerProcess(GatewayError):
    """A hosted step was attempted from a caller's own process (AC-5, AC-7)."""


class CallFailed(GatewayError):
    """A call failed with a classified cause; `code` names a holder refusal when one applies."""

    def __init__(self, failure_class: FailureClass, message: str, code: str | None = None):
        super().__init__(message)
        self.failure_class = failure_class
        self.code = code


@dataclass(frozen=True, slots=True)
class CallOptions:
    """Per-call switches; `False` returns the first failure unmasked (FR-19, AC-12)."""

    retry: bool = True
    fallback: bool = True


@dataclass(frozen=True, slots=True)
class Request:
    """One gateway call. No model or provider field by design (FR-2)."""

    call_site: str
    messages: tuple[Mapping[str, Any], ...]
    scopes: tuple[str, ...] = ()
    run_id: str | None = None
    tools: tuple[Mapping[str, Any], ...] | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)
    response_format: Mapping[str, Any] | None = None
    options: CallOptions = field(default_factory=CallOptions)
    load_mode: LoadMode | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_site, str) or not self.call_site:
            raise ValueError("call_site must be a non-empty string")
        if self.run_id is not None and not (
            isinstance(self.run_id, str) and IDENT.fullmatch(self.run_id)
        ):
            raise ValueError("run_id must match [A-Za-z0-9_:.\\-/@+]{1,200}")
        unknown = set(self.settings) - NEUTRAL_SETTINGS
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Request:
        """Build from a wire dict, refusing unknown fields (AC-2)."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown request fields: {', '.join(sorted(unknown))}")
        values = dict(data)
        for key in ("messages", "scopes", "tools"):
            if values.get(key) is not None:
                values[key] = tuple(values[key])
        if "options" in values:
            raw = values["options"]
            bad = set(raw) - {"retry", "fallback"}
            if bad:
                raise ValueError(f"unknown option fields: {', '.join(sorted(bad))}")
            values["options"] = CallOptions(**raw)
        if values.get("load_mode") is not None:
            values["load_mode"] = LoadMode(values["load_mode"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts as the provider reported them (FR-5)."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Response:
    """A completed call. `model` is the provider-reported identity (FR-5, FR-9)."""

    call_id: str
    provider: str
    model: str
    hosted: bool
    message: Mapping[str, Any]
    usage: Usage = field(default_factory=Usage)
    provider_detail: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
