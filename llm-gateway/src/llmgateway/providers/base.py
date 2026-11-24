"""The provider contract (docs/plans/llm-gateway.md §3.2, FR-4, FR-15, FR-17, FR-18).

A provider kind is a `Provider` subclass: `build` turns neutral inputs into a wire request,
`parse` reads a successful reply, `classify` maps a failed exchange to a `FailureClass` (the
error body is read there and dropped). A provider adds nothing the caller did not ask for
(FR-18): values pass through unchanged, and a required setting it cannot honour fails the call
naming it (`SettingUnsupported`) before anything is sent.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from llmgateway.failures import classify_response
from llmgateway.types import CallFailed, FailureClass, Usage


class SettingUnsupported(CallFailed):
    """A required generation setting the selected provider cannot honour (FR-4, AC-1)."""

    def __init__(self, provider_id: str, setting: str) -> None:
        super().__init__(
            FailureClass.CONFIG,
            f"provider {provider_id!r} cannot honour required setting {setting!r}",
        )
        self.setting = setting


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Everything `build` may use. `config` is the provider's `[providers.<id>]` table;
    `settings` the effective neutral settings (profile merged with the request's);
    `optional` the names the call site marks optional."""

    provider_id: str
    config: Mapping[str, Any]
    model: str
    messages: tuple[Mapping[str, Any], ...]
    settings: Mapping[str, Any] = field(default_factory=dict)
    optional: frozenset[str] = frozenset()
    tools: tuple[Mapping[str, Any], ...] | None = None
    response_format: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BuiltRequest:
    """A wire request: path under the provider's `base_url`, serialized body bytes, and any
    non-secret extra headers (never a credential: hosted keys belong to the holder)."""

    path: str
    body: bytes
    method: str = "POST"
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedReply:
    """A successful reply. `model` is the provider-reported identity (FR-5, FR-9)."""

    model: str
    message: Mapping[str, Any]
    usage: Usage = field(default_factory=Usage)
    detail: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


def encode_body(payload: Mapping[str, Any]) -> bytes:
    """The one body serializer: values are dumped as given, keys in insertion order, so a seed
    or a JSON schema reaches the wire exactly as the caller supplied it (FR-18)."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def map_settings(
    provider_id: str,
    settings: Mapping[str, Any],
    table: Mapping[str, str | None],
    optional: Collection[str] = (),
) -> dict[str, Any]:
    """Translate neutral settings to wire fields. A name the table maps to `None` (or omits) is
    unsupported: skipped when the call site marks it optional, otherwise it raises
    `SettingUnsupported` naming it. Values are never altered and nothing is added (FR-4, FR-18)."""
    out: dict[str, Any] = {}
    for name, value in settings.items():
        wire = table.get(name)
        if wire is None:
            if name in optional:
                continue
            raise SettingUnsupported(provider_id, name)
        out[wire] = value
    return out


class Provider(ABC):
    """One provider kind. Subclasses set `kind` and may set `ops` (FR-17: named provider-specific
    operations) and `residency` (§3.8, wired by the engine in D6)."""

    kind: str
    ops: Mapping[str, Any] = {}
    residency: Any | None = None

    @abstractmethod
    def build(self, ctx: BuildContext) -> BuiltRequest: ...

    @abstractmethod
    def parse(self, body: bytes, ctx: BuildContext) -> ParsedReply:
        """Read a 2xx body; raise `ValueError`/`KeyError`/`TypeError` when it is malformed."""

    def classify(
        self, status: int, content_type: str | None, body: bytes, config: Mapping[str, Any]
    ) -> FailureClass:
        """Class of a non-2xx exchange. The default is the shared §3.3 table with the provider's
        owner-configured `out_of_credit_patterns`; a kind overrides it only for its own shapes."""
        patterns = config.get("out_of_credit_patterns", ())
        return classify_response(
            status,
            content_type,
            body,
            out_of_credit_patterns=patterns if isinstance(patterns, (list, tuple)) else (),
        )
