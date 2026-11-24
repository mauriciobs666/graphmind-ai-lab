"""Ollama provider, native API (docs/plans/llm-gateway.md §3.2, §3.8; FR-4, FR-9, FR-15, FR-16,
FR-17, FR-18). Local Ollama and hosted Ollama cloud (a `key_ref` provider; the holder adds
`Authorization: Bearer <key>`, https://docs.ollama.com/api/authentication per Pass 5).

Wire facts verified 2026-11-24 (https://docs.ollama.com/api/chat, `/api/ps`, `/modelfile`,
`/cloud`, `/api/errors`):
  * `POST /api/chat`: `model, messages, tools, format` (`"json"` or a JSON Schema object),
    `options` (`seed, temperature, top_k, top_p, min_p, stop, num_ctx, num_predict`, extra keys
    allowed), `stream` (default TRUE, so `stream: false` is the one field this adapter must add),
    `think`, `keep_alive`. Reply: `model, message{role, content, thinking, tool_calls},
    done_reason, prompt_eval_count, eval_count`, durations in nanoseconds.
  * The Modelfile reference lists `repeat_penalty` but neither `frequency_penalty` nor
    `presence_penalty`, so those two are unsupported (fail naming, or skipped when optional).
  * Cloud models do not support structured outputs: a hosted call with `response_format` fails
    `rejected` before sending. Load settings are refused on hosted, and no parallelism setting
    exists in the Modelfile reference: `parallel` fails naming it (plan D-2a), local or hosted.
  * Error bodies are `{"error": "<string>"}`; 404/429/500/502 are documented, the queue-full 503
    is observed behaviour only (undocumented); all classify by the shared §3.3 table.
  * NOT documented, hence unverified: whether `eval_count` includes thinking tokens (counted as
    reported); what a differing `num_ctx` does to a loaded model (the engine, D6, never sends
    one); that an empty `messages` list loads a model (plan §3.8 cites Ollama's `api.md`);
    `/api/tags` and `/api/show` shapes (ops only name the paths).
The chat body never carries `num_ctx` (FR-18: nothing added); a `keep_alive` or `think` appears
only when the provider's own configuration sets it, never on hosted (the holder's closed body
table has no `keep_alive`).
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
    SettingUnsupported,
    encode_body,
    map_settings,
)
from llmgateway.providers.openai_compat import (
    get_op,
    is_hosted_config,
    post_op,
    reported_count,
    require_pinned,
)
from llmgateway.types import CallFailed, FailureClass, Usage

CHAT_PATH = "/api/chat"
SETTINGS: Mapping[str, str | None] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repeat_penalty": "repeat_penalty",
    "frequency_penalty": None,
    "presence_penalty": None,
    "max_tokens": "num_predict",
    "seed": "seed",
    "stop": "stop",
}
LOAD_SETTINGS: Mapping[str, str] = {"context_length": "num_ctx"}
DETAIL_KEYS = (
    "done_reason",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
    "prompt_eval_cached_count",
)


def _format(ctx: BuildContext) -> Any:
    rf = ctx.response_format
    if rf is None:
        return None
    if rf.get("type") == "json_schema":
        return rf["json_schema"]["schema"]
    if rf.get("type") == "json_object":
        return "json"
    raise CallFailed(
        FailureClass.REJECTED,
        f"provider {ctx.provider_id!r}: response_format type {rf.get('type')!r} has no "
        "Ollama equivalent",
    )


class OllamaProvider(Provider):
    kind = "ollama"
    ops = {
        "ollama.ps": get_op("/api/ps"),
        "ollama.tags": get_op("/api/tags"),
        "ollama.show": post_op("/api/show"),
    }

    def build(self, ctx: BuildContext) -> BuiltRequest:
        hosted = is_hosted_config(ctx.config)
        require_pinned(ctx)
        if hosted and ctx.response_format is not None:
            raise CallFailed(
                FailureClass.REJECTED,
                f"provider {ctx.provider_id!r}: hosted Ollama does not support structured "
                "outputs (response_format)",
            )
        if "parallel" in ctx.config:
            raise SettingUnsupported(ctx.provider_id, "parallel")
        keep_alive = ctx.config.get("keep_alive")
        if hosted and keep_alive is not None:
            raise CallFailed(
                FailureClass.CONFIG,
                f"provider {ctx.provider_id!r}: keep_alive is a load setting, refused on hosted",
            )
        payload: dict[str, Any] = {"model": ctx.model, "messages": list(ctx.messages)}
        if ctx.tools is not None:
            payload["tools"] = list(ctx.tools)
        options = map_settings(ctx.provider_id, ctx.settings, SETTINGS, ctx.optional)
        if options:
            payload["options"] = options
        fmt = _format(ctx)
        if fmt is not None:
            payload["format"] = fmt
        if "think" in ctx.config:
            payload["think"] = ctx.config["think"]
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        payload["stream"] = False
        return BuiltRequest(CHAT_PATH, encode_body(payload))

    def parse(self, body: bytes, ctx: BuildContext) -> ParsedReply:
        data = json.loads(body)
        model, message = data["model"], data["message"]
        if not isinstance(model, str) or not model or not isinstance(message, Mapping):
            raise ValueError("reply model or message is malformed")
        usage = Usage(reported_count(data, "prompt_eval_count"), reported_count(data, "eval_count"))
        detail = {k: data[k] for k in DETAIL_KEYS if k in data}
        return ParsedReply(model, message, usage, detail, data)

    # --- residency support for D6 (§3.8): request builders and a parser, no I/O ---

    def loaded_request(self) -> BuiltRequest:
        return BuiltRequest("/api/ps", b"", "GET")

    def parse_loaded(self, body: bytes) -> list[dict[str, Any]]:
        """Loaded models as `{id, context_length, raw}`; `raw` is the provider's own record."""
        return [
            {"id": rec["model"], "context_length": rec.get("context_length"), "raw": rec}
            for rec in json.loads(body)["models"]
        ]

    def load_request(
        self,
        provider_id: str,
        model: str,
        settings: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> BuiltRequest:
        """An empty-`messages` chat with the load settings: `context_length` -> `options.num_ctx`
        and an optional `keep_alive`. `parallel` fails naming it (D-2a); so does any other
        name, and every load setting on a hosted provider."""
        options: dict[str, Any] = {}
        payload: dict[str, Any] = {"model": model, "messages": []}
        for name, value in settings.items():
            if name in LOAD_SETTINGS:
                options[LOAD_SETTINGS[name]] = value
            elif name == "keep_alive":
                payload["keep_alive"] = value
            else:
                raise SettingUnsupported(provider_id, name)
        if is_hosted_config(config):
            name = next(iter(settings), "load settings")
            raise CallFailed(
                FailureClass.CONFIG,
                f"provider {provider_id!r}: load setting {name!r} is refused on hosted Ollama",
            )
        if options:
            payload["options"] = options
        payload["stream"] = False
        return BuiltRequest(CHAT_PATH, encode_body(payload))


PROVIDER = OllamaProvider()
