"""Unit tests for the LM Studio chat client (K-013), offline via injected transport.

Mirrors `test_embedding.py`: the real `LMStudioLLM` is exercised with an injected
transport so its request payload + `choices[0].message.content` parsing are pinned
without a live LM Studio server. Unit tests never touch the network.
"""

from __future__ import annotations

import json

from falkorchat.llm import LMStudioLLM


def test_lmstudio_llm_parses_content_and_posts_expected_payload():
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "hi there"}}]}

    llm = LMStudioLLM(
        base_url="http://localhost:1234/v1", model="qwen/qwen3-4b-2507",
        transport=fake_transport,
    )

    out = llm.complete([{"role": "user", "content": "hello"}])

    assert out == "hi there"
    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    assert captured["payload"] == {
        "model": "qwen/qwen3-4b-2507",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_lmstudio_llm_strips_trailing_slash_on_base_url():
    def fake_transport(url: str, payload: dict) -> dict:
        assert url == "http://localhost:1234/v1/chat/completions"
        return {"choices": [{"message": {"content": "ok"}}]}

    llm = LMStudioLLM(
        base_url="http://localhost:1234/v1/", model="m", transport=fake_transport
    )
    assert llm.complete([{"role": "user", "content": "x"}]) == "ok"


def test_lmstudio_llm_defaults_model_and_base_url_from_config():
    from falkorchat import config

    def fake_transport(url: str, payload: dict) -> dict:
        assert url == f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
        assert payload["model"] == config.LLM_MODEL
        return {"choices": [{"message": {"content": "ok"}}]}

    llm = LMStudioLLM(transport=fake_transport)
    assert llm.complete([{"role": "user", "content": "x"}]) == "ok"


# --- chat(): tool-calling seam (U6) ------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "graphrag_retrieve",
            "description": "Retrieve context.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
]


def _chat_transport(message: dict):
    """Build a fake transport returning `message` as the sole choice, capturing the payload."""
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": [{"message": message}]}

    return fake_transport, captured


def test_chat_sends_tools_field_and_hits_chat_completions():
    fake_transport, captured = _chat_transport(
        {"role": "assistant", "content": "hello"}
    )
    llm = LMStudioLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    llm.chat([{"role": "user", "content": "hi"}], _TOOLS)

    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["payload"]["tools"] == _TOOLS
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["payload"]["model"] == "m"


def test_chat_parses_native_tool_calls_shape():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "graphrag_retrieve",
                    "arguments": json.dumps({"query": "reset password"}),
                },
            }
        ],
    }
    fake_transport, _ = _chat_transport(message)
    llm = LMStudioLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "help"}], _TOOLS)

    assert result.is_tool_call
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_abc"
    assert call.name == "graphrag_retrieve"
    assert call.arguments == {"query": "reset password"}


def test_chat_parses_content_embedded_json_fallback():
    # Qwen3 failure mode: the tool call arrives as JSON text in `content`,
    # with an empty/absent structured `tool_calls` field.
    message = {
        "role": "assistant",
        "content": json.dumps(
            {"name": "graphrag_retrieve", "arguments": {"query": "billing"}}
        ),
    }
    fake_transport, _ = _chat_transport(message)
    llm = LMStudioLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "help"}], _TOOLS)

    assert result.is_tool_call
    call = result.tool_calls[0]
    assert call.name == "graphrag_retrieve"
    assert call.arguments == {"query": "billing"}


def test_chat_parses_structured_output_action_shape_in_fenced_content():
    # Structured-output prompting shape `{"action", "args"}`, wrapped in a
    # markdown code fence and surrounding prose.
    blob = json.dumps({"action": "graphrag_retrieve", "args": {"query": "vpn"}})
    message = {
        "role": "assistant",
        "content": f"Sure, let me look that up.\n```json\n{blob}\n```",
    }
    fake_transport, _ = _chat_transport(message)
    llm = LMStudioLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "help"}], _TOOLS)

    assert result.is_tool_call
    call = result.tool_calls[0]
    assert call.name == "graphrag_retrieve"
    assert call.arguments == {"query": "vpn"}


def test_chat_returns_plain_text_when_no_tool_call():
    message = {"role": "assistant", "content": "The office opens at 9am."}
    fake_transport, _ = _chat_transport(message)
    llm = LMStudioLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "hours?"}], _TOOLS)

    assert not result.is_tool_call
    assert result.tool_calls == []
    assert result.text == "The office opens at 9am."


def test_chat_treats_non_tool_json_content_as_text():
    # A JSON object with no recognizable tool-call keys is a plain answer,
    # not a tool call — it must stay text.
    payload = json.dumps({"answer": 42, "confidence": "high"})
    message = {"role": "assistant", "content": payload}
    fake_transport, _ = _chat_transport(message)
    llm = LMStudioLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "q"}], _TOOLS)

    assert not result.is_tool_call
    assert result.text == payload


def test_chat_tolerates_string_arguments_that_fail_to_parse():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "function": {"name": "graphrag_retrieve", "arguments": "not json"}}
        ],
    }
    fake_transport, _ = _chat_transport(message)
    llm = LMStudioLLM(base_url="http://x/v1", model="m", transport=fake_transport)

    result = llm.chat([{"role": "user", "content": "x"}], _TOOLS)

    assert result.is_tool_call
    assert result.tool_calls[0].arguments == {}


def test_complete_still_omits_tools_field():
    # Regression: the completion path the responder uses must not gain a `tools`
    # field or change shape.
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "unchanged"}}]}

    llm = LMStudioLLM(base_url="http://x/v1", model="m", transport=fake_transport)
    out = llm.complete([{"role": "user", "content": "hello"}])

    assert out == "unchanged"
    assert "tools" not in captured["payload"]
    assert captured["payload"] == {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
    }
