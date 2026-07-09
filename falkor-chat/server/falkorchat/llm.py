"""LM Studio chat client for the K-013 AI responder.

Mirrors `embedding.py`: an injectable `Transport` seam makes the OpenAI-compatible
`/v1/chat/completions` request payload + response parsing unit-testable offline;
production injects the stdlib urllib transport. The `LLM` protocol is what the
`AgentResponder` depends on — unit tests inject a deterministic stub, so no test
ever hits a live model.

The responder calls `.complete(...)` **before** the guarded §4 write, so LLM
latency or failure short-circuits before anything is posted (failure isolation).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable, Protocol

from . import config

# A transport is `(url, json_payload) -> parsed_json_dict`. Injected so the
# client's response parsing is unit-testable without a live LM Studio server.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]

# Chat messages are OpenAI-shaped `{"role": ..., "content": ...}` dicts.
ChatMessage = dict[str, str]


class LLM(Protocol):
    """Anything that turns a chat message list into a completion string."""

    def complete(self, messages: list[ChatMessage]) -> str: ...


def _urllib_transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Default HTTP transport (stdlib only — no runtime HTTP dependency).

    POSTs `payload` as JSON and returns the decoded JSON response. Network/HTTP
    errors propagate to the responder, which runs out-of-band from the write path.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (fixed, config-controlled URL)
        return json.loads(resp.read().decode("utf-8"))


class LMStudioLLM:
    """OpenAI-compatible `/v1/chat/completions` client (LM Studio backend).

    `transport` is injectable for tests; it defaults to a stdlib urllib POST.
    """

    def __init__(
        self,
        base_url: str = config.LLM_BASE_URL,
        model: str = config.LLM_MODEL,
        *,
        transport: Transport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._transport = transport or _urllib_transport

    def complete(self, messages: list[ChatMessage]) -> str:
        resp = self._transport(
            f"{self._base_url}/chat/completions",
            {"model": self._model, "messages": list(messages)},
        )
        return resp["choices"][0]["message"]["content"]
