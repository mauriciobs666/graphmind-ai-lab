"""Generic OpenAI-compatible chat completions provider (docs/plans/llm-gateway.md §3.2; FR-4,
FR-9, FR-15, FR-18). Also the base of `lmstudio` and home of the small helpers every adapter in
this package shares (`reported_count`, `require_pinned`, the `ProviderOp` builders).

Wire facts, verified 2026-11-24 against the vendors' current documentation:
  * LM Studio's `/v1/chat/completions` lists `model, messages, temperature, top_p, top_k,
    max_tokens, stream, stop, presence_penalty, frequency_penalty, logit_bias, repeat_penalty,
    seed` (https://lmstudio.ai/docs/developer/openai-compat/chat-completions). `min_p` is not
    listed there, and the holder's closed body table (plan §3.4) has no `min_p` either, so
    `min_p` is unsupported here: a call site that declares it fails naming it, or marks it
    optional and it is skipped.
  * Structured output is `response_format = {"type": "json_schema", "json_schema": {...}}`
    (https://lmstudio.ai/docs/developer/openai-compat/structured-output); it is passed through
    exactly as given.
  * Usage is `usage.prompt_tokens` / `usage.completion_tokens`, the fields the holder settles
    on (plan §3.4); reasoning tokens are counted by the provider inside `completion_tokens`
    and are not split out here.
Nothing is added to the body (no `stream`, which defaults to false): FR-18.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from llmgateway.providers.base import (
    BuildContext,
    BuiltRequest,
    ParsedReply,
    Provider,
    encode_body,
    map_settings,
)
from llmgateway.types import CallFailed, FailureClass, Usage

ProviderOp = Callable[[Mapping[str, Any]], BuiltRequest]
"""A named provider-specific operation (FR-17): `params` in, the wire request out. The caller
(service, D8) sends it through its transport and returns the provider's JSON verbatim."""


def get_op(path: str) -> ProviderOp:
    return lambda params: BuiltRequest(path, b"", "GET")


def post_op(path: str) -> ProviderOp:
    return lambda params: BuiltRequest(path, encode_body(params))


def reported_count(container: Mapping[str, Any], key: str) -> int | None:
    """A token count as the provider reported it: absent or null is `None`, a non-negative int
    is itself, anything else is a malformed reply (`ValueError`)."""
    value = container.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} is not a non-negative integer")
    return value


def is_hosted_config(config: Mapping[str, Any]) -> bool:
    """A provider with a `key_ref` is reached through the key holder with a key it injects."""
    return bool(config.get("key_ref"))


def require_pinned(ctx: BuildContext) -> None:
    """FR-9, AC-18: a hosted call names an exact member of the owner's `pinned_models`; an alias
    (`-latest`, an undated id) is not a member, so it fails here before anything is built. The
    holder repeats the check with the same rule (`model_not_pinned`)."""
    if not is_hosted_config(ctx.config):
        return
    pinned = ctx.config.get("pinned_models")
    if not isinstance(pinned, list) or ctx.model not in pinned:
        raise CallFailed(
            FailureClass.REJECTED,
            f"provider {ctx.provider_id!r}: model {ctx.model!r} is not an exact pinned model",
        )


class OpenAICompatProvider(Provider):
    kind = "openai_compat"
    path = "/v1/chat/completions"
    SETTINGS: Mapping[str, str | None] = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": None,
        "repeat_penalty": "repeat_penalty",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "max_tokens": "max_tokens",
        "seed": "seed",
        "stop": "stop",
    }

    def build(self, ctx: BuildContext) -> BuiltRequest:
        require_pinned(ctx)
        payload: dict[str, Any] = {"model": ctx.model, "messages": list(ctx.messages)}
        payload.update(map_settings(ctx.provider_id, ctx.settings, self.SETTINGS, ctx.optional))
        if ctx.tools is not None:
            payload["tools"] = list(ctx.tools)
        if ctx.response_format is not None:
            payload["response_format"] = ctx.response_format
        return BuiltRequest(self.path, encode_body(payload))

    def parse(self, body: bytes, ctx: BuildContext) -> ParsedReply:
        data = json.loads(body)
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("reply has no choices")
        message = choices[0]["message"]
        model = data["model"]
        if not isinstance(model, str) or not model or not isinstance(message, Mapping):
            raise ValueError("reply model or message is malformed")
        usage = data.get("usage")
        counts = Usage()
        if usage is not None:
            counts = Usage(
                reported_count(usage, "prompt_tokens"), reported_count(usage, "completion_tokens")
            )
        return ParsedReply(model, message, counts, self.detail(data, choices[0]), data)

    def detail(self, data: Mapping[str, Any], choice: Mapping[str, Any]) -> dict[str, Any]:
        finish = choice.get("finish_reason")
        return {} if finish is None else {"finish_reason": finish}


PROVIDER = OpenAICompatProvider()
