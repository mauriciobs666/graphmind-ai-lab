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
from dataclasses import dataclass, field
import urllib.request
from typing import Any, Callable, Protocol

from . import config

# A transport is `(url, json_payload) -> parsed_json_dict`. Injected so the
# client's response parsing is unit-testable without a live LM Studio server.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]

# Chat messages are OpenAI-shaped `{"role": ..., "content": ...}` dicts.
ChatMessage = dict[str, str]


@dataclass(frozen=True)
class ToolCall:
    """One resolved tool call — OpenAI-shaped, with `arguments` parsed to a dict.

    `id` is the call id the model (or our fallback) assigned; U8 echoes it back
    on the `tool` result message. `arguments` is always a dict here (the native
    OpenAI shape delivers it as a JSON *string*, which `chat` decodes) — the
    caller never re-parses.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatResult:
    """A single assistant turn: free text and/or a list of tool calls.

    `text` is the message content (may be None/empty when the model only calls
    tools); `tool_calls` is empty on a plain-text turn. `is_tool_call` is the
    convenience the agent loop (U8) branches on.
    """

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)


class LLM(Protocol):
    """The LLM seam: single-shot completion plus tool-calling chat.

    `complete` is what `AgentResponder` (M2) depends on; `chat` is the
    tool-calling extension the M3 executor's agent nodes (U8) drive.
    """

    def complete(self, messages: list[ChatMessage]) -> str: ...

    def chat(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]]
    ) -> ChatResult: ...


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

    def chat(
        self, messages: list[ChatMessage], tools: list[dict[str, Any]]
    ) -> ChatResult:
        """One tool-calling turn: POST `messages` + granted `tools`, parse a `ChatResult`.

        Dual-shape parsing (Q3): prefer the structured `message.tool_calls` field;
        if it is absent, fall back to detecting a tool call emitted as JSON inside
        `message.content` (the known LM Studio / Qwen3 failure mode). When neither
        yields a tool call, the turn is plain text. Name-against-granted-set and
        arg-schema validation live in the agent loop (U8), not here — `chat` only
        parses the wire shape into `ChatResult`.
        """
        resp = self._transport(
            f"{self._base_url}/chat/completions",
            {"model": self._model, "messages": list(messages), "tools": list(tools)},
        )
        return _parse_chat_message(resp["choices"][0]["message"])


def _parse_chat_message(message: dict[str, Any]) -> ChatResult:
    """Turn one OpenAI `choices[0].message` dict into a `ChatResult`.

    Order matters (Q3): the structured `tool_calls` field is authoritative; only
    when it is absent/empty do we probe `content` for a tool call emitted as JSON
    (the Qwen3 fallback). Content that is not a tool call is returned as text.
    """
    content = message.get("content")

    native = _parse_native_tool_calls(message.get("tool_calls"))
    if native:
        return ChatResult(text=content, tool_calls=native)

    embedded = _parse_content_tool_calls(content)
    if embedded:
        return ChatResult(text=None, tool_calls=embedded)

    return ChatResult(text=content, tool_calls=[])


def _parse_native_tool_calls(raw: Any) -> list[ToolCall]:
    """Parse the structured OpenAI `message.tool_calls` list into `ToolCall`s.

    Each entry nests the call under `function` with `arguments` as a JSON *string*
    (OpenAI shape); we decode it. LM Studio occasionally hands back an already-parsed
    dict — tolerated. Unparseable/nameless entries are skipped rather than raised.
    """
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function", entry)
        name = fn.get("name") if isinstance(fn, dict) else None
        if not name:
            continue
        args = _coerce_arguments(fn.get("arguments") if isinstance(fn, dict) else None)
        call_id = entry.get("id") or f"call_{i}"
        calls.append(ToolCall(id=str(call_id), name=str(name), arguments=args))
    return calls


def _parse_content_tool_calls(content: Any) -> list[ToolCall]:
    """Fallback: detect a tool call encoded as JSON inside `content` (Q3, Qwen3).

    Accepts the OpenAI-ish `{"name": ..., "arguments": ...}` and the
    structured-output `{"action": ..., "args": ...}` shapes, a `{"function": {...}}`
    wrapper, and a `{"tool_calls": [...]}` envelope. Plain prose (no recognizable
    tool-call keys) yields no calls, so it stays text.
    """
    obj = _extract_json_object(content)
    if obj is None:
        return []
    if isinstance(obj.get("tool_calls"), list):
        return _parse_native_tool_calls(obj["tool_calls"])
    call = _normalize_tool_call(obj)
    return [call] if call else []


def _normalize_tool_call(obj: dict[str, Any], index: int = 0) -> ToolCall | None:
    """Map one loosely-shaped dict to a `ToolCall`, or None if it isn't a call."""
    fn = obj.get("function")
    if isinstance(fn, dict):
        obj = fn
    name = obj.get("name") or obj.get("action") or obj.get("tool")
    if not name:
        return None
    args = _coerce_arguments(
        obj.get("arguments") if "arguments" in obj else obj.get("args")
    )
    call_id = obj.get("id") or f"call_{index}"
    return ToolCall(id=str(call_id), name=str(name), arguments=args)


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    """Normalize a tool-call `arguments` value to a dict.

    OpenAI delivers arguments as a JSON string; some backends pre-parse to a dict.
    Anything else (or unparseable JSON) becomes an empty dict — U8's schema check
    is the real gate, so `chat` never raises on a malformed arg blob.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_json_object(content: Any) -> dict[str, Any] | None:
    """Best-effort: pull a single JSON object out of a `content` string.

    Handles a bare JSON object, one fenced in a ```json code block, and a JSON
    object embedded in surrounding prose (first `{` … last `}`). Returns None when
    no object parses — the caller then treats the content as plain text.
    """
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None

    # Strip a leading/trailing markdown code fence if present.
    if text.startswith("```"):
        inner = text[3:]
        if inner[:4].lower() == "json":
            inner = inner[4:]
        text = inner.rsplit("```", 1)[0].strip()

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
