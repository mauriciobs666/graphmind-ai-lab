"""Anthropic Messages provider (docs/plans/llm-gateway.md §3.2, §3.4; FR-4, FR-9, FR-15, FR-18).

Always hosted: the key holder sends the request and adds `x-api-key` and `anthropic-version`
(so this adapter sets no header). Wire facts verified 2026-11-24:
  * `POST /v1/messages`: `model, max_tokens` and `messages` required; `system` (string or text
    blocks), `stop_sequences`, `output_config.format = {"type": "json_schema", "schema": ...}`
    for structured outputs, stable, no beta header
    (https://platform.claude.com/docs/en/api/messages,
    https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
  * `system` is top-level: `role: "system"` messages move there; the rest pass unchanged.
  * Usage: `input_tokens` EXCLUDES cache tokens; the total input is `cache_read_input_tokens +
    cache_creation_input_tokens + input_tokens`
    (https://platform.claude.com/docs/en/build-with-claude/prompt-caching). `Usage.input_tokens`
    is that total and `output_tokens` is `usage.output_tokens`, the sum the holder settles on
    (plan §3.4 table). A null cache field counts as absent.
  * Errors: 400 invalid_request (also a spend limit), 401, 402 billing_error, 403, 404, 413,
    429, 500, 504 timeout_error, 529 overloaded_error
    (https://platform.claude.com/docs/en/api/errors): the shared §3.3 table covers each by
    status, so `classify` is not overridden.
NOT verified (the fetched pages were truncated or silent): `temperature`, `top_p` and `top_k` as
Messages parameters (taken from the holder table, plan §3.4), whether newer models reject them,
that `output_tokens` includes thinking tokens, and whether an unpinned alias form exists for a
given model (pinning is exact membership in `pinned_models`, never inferred from the id).
Not translated: `tools` (a tool-call message needs a content-block conversion this release does
not build); a call with tools fails naming them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from llmgateway.providers.base import (
    BuildContext,
    BuiltRequest,
    ParsedReply,
    Provider,
    encode_body,
    map_settings,
)
from llmgateway.providers.openai_compat import reported_count, require_pinned
from llmgateway.types import CallFailed, FailureClass, Usage

PATH = "/v1/messages"
SETTINGS: Mapping[str, str | None] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": None,
    "repeat_penalty": None,
    "frequency_penalty": None,
    "presence_penalty": None,
    "max_tokens": "max_tokens",
    "seed": None,
    "stop": "stop_sequences",
}
CACHE_KEYS = ("cache_creation_input_tokens", "cache_read_input_tokens")


def _system_and_messages(
    messages: tuple[Mapping[str, Any], ...],
) -> tuple[Any, list[Mapping[str, Any]]]:
    blocks: list[Any] = []
    rest: list[Mapping[str, Any]] = []
    for m in messages:
        if m.get("role") != "system":
            rest.append(m)
            continue
        content = m["content"]
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        else:
            blocks.extend(content)
    if not blocks:
        return None, rest
    if len(blocks) == 1 and set(blocks[0]) == {"type", "text"} and blocks[0]["type"] == "text":
        return blocks[0]["text"], rest
    return blocks, rest


class AnthropicProvider(Provider):
    kind = "anthropic"

    def build(self, ctx: BuildContext) -> BuiltRequest:
        require_pinned(ctx)
        if ctx.tools is not None:
            raise CallFailed(
                FailureClass.CONFIG,
                f"provider {ctx.provider_id!r}: tools are not supported by this adapter",
            )
        if "max_tokens" not in ctx.settings:
            raise CallFailed(
                FailureClass.CONFIG,
                f"provider {ctx.provider_id!r} requires setting 'max_tokens'",
            )
        system, messages = _system_and_messages(ctx.messages)
        mapped = map_settings(ctx.provider_id, ctx.settings, SETTINGS, ctx.optional)
        if isinstance(mapped.get("stop_sequences"), str):
            mapped["stop_sequences"] = [mapped["stop_sequences"]]
        payload: dict[str, Any] = {
            "model": ctx.model,
            "max_tokens": mapped.pop("max_tokens"),
        }
        if system is not None:
            payload["system"] = system
        payload["messages"] = messages
        payload.update(mapped)
        rf = ctx.response_format
        if rf is not None:
            if rf.get("type") != "json_schema":
                raise CallFailed(
                    FailureClass.REJECTED,
                    f"provider {ctx.provider_id!r}: response_format type {rf.get('type')!r} "
                    "has no Anthropic equivalent",
                )
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": rf["json_schema"]["schema"]}
            }
        return BuiltRequest(PATH, encode_body(payload))

    def parse(self, body: bytes, ctx: BuildContext) -> ParsedReply:
        data = json.loads(body)
        model, content, usage = data["model"], data["content"], data["usage"]
        if not isinstance(model, str) or not model or not isinstance(content, list):
            raise ValueError("reply model or content is malformed")
        text = ""
        for block in content:
            if block.get("type") == "text":
                if not isinstance(block["text"], str):
                    raise ValueError("text block is not a string")
                text += block["text"]
        parts = [reported_count(usage, k) for k in ("input_tokens", *CACHE_KEYS)]
        known = [p for p in parts if p is not None]
        counts = Usage(sum(known) if known else None, reported_count(usage, "output_tokens"))
        detail = {k: data[k] for k in ("stop_reason", "stop_sequence") if data.get(k) is not None}
        return ParsedReply(model, {"role": "assistant", "content": text}, counts, detail, data)


PROVIDER = AnthropicProvider()
