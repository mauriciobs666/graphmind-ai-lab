"""LM Studio provider (docs/plans/llm-gateway.md §3.2, §3.8; FR-4, FR-15, FR-16, FR-17, FR-18).

OpenAI-compatible chat (see `openai_compat`) plus what is LM Studio's own:
  * `stats`, `model_info` and `runtime` of a chat reply go verbatim into
    `Response.provider_detail` (AC-20). The v0 REST page documents them on
    `/api/v0/chat/completions` (https://lmstudio.ai/docs/developer/rest/endpoints); that they
    are also returned by `/v1/chat/completions` is NOT verified here (live smoke).
  * Residency support for D6 (§3.8): `loaded_request` is `GET /api/v0/models`, the path ccg's
    manifest allows; `parse_loaded` keeps each loaded entry verbatim in `raw` and reads the
    context from `loaded_context_length` (used by ccg's fingerprint; not on the vendor's v0 page,
    so unverified against the docs). `load_request` is `POST /api/v1/models/load` with the
    documented fields `model, context_length, eval_batch_size, flash_attention, num_experts,
    offload_kv_cache_to_gpu, echo_load_config` (https://lmstudio.ai/docs/developer/rest/load).
    That page documents no parallelism setting, so `parallel` fails naming it.
  * `ops` (FR-17): `lmstudio.models` (GET `/api/v0/models`) and `lmstudio.decisions`
    (`POST /v1/decisions`; the path is inferred from the vendor's Python example at
    https://lmstudio.ai/docs/developer/openai-compat/decisions, not stated there).
A no-model-loaded error body is classified `model_not_loaded` by the shared §3.3 classifier.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from llmgateway.providers.base import BuiltRequest, SettingUnsupported, encode_body
from llmgateway.providers.openai_compat import OpenAICompatProvider, get_op, post_op

LOADED_PATH = "/api/v0/models"
LOAD_PATH = "/api/v1/models/load"
LOAD_SETTINGS = frozenset(
    {
        "context_length",
        "eval_batch_size",
        "flash_attention",
        "num_experts",
        "offload_kv_cache_to_gpu",
        "echo_load_config",
    }
)
DETAIL_KEYS = ("stats", "model_info", "runtime")


class LMStudioProvider(OpenAICompatProvider):
    kind = "lmstudio"
    ops = {
        "lmstudio.models": get_op(LOADED_PATH),
        "lmstudio.decisions": post_op("/v1/decisions"),
    }

    def detail(self, data: Mapping[str, Any], choice: Mapping[str, Any]) -> dict[str, Any]:
        out = super().detail(data, choice)
        out.update({k: data[k] for k in DETAIL_KEYS if k in data})
        return out

    def loaded_request(self) -> BuiltRequest:
        return BuiltRequest(LOADED_PATH, b"", "GET")

    def parse_loaded(self, body: bytes) -> list[dict[str, Any]]:
        """Loaded models as `{id, context_length, raw}`; `raw` is the provider's own record."""
        out = []
        for rec in json.loads(body)["data"]:
            if rec.get("state") == "loaded":
                ctx_len = rec.get("loaded_context_length")
                out.append({"id": rec["id"], "context_length": ctx_len, "raw": rec})
        return out

    def load_request(
        self, provider_id: str, model: str, settings: Mapping[str, Any]
    ) -> BuiltRequest:
        for name in settings:
            if name not in LOAD_SETTINGS:
                raise SettingUnsupported(provider_id, name)
        return BuiltRequest(LOAD_PATH, encode_body({"model": model, **settings}))


PROVIDER = LMStudioProvider()
