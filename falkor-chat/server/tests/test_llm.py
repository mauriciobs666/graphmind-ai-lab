"""Unit tests for the LM Studio chat client (K-013), offline via injected transport.

Mirrors `test_embedding.py`: the real `LMStudioLLM` is exercised with an injected
transport so its request payload + `choices[0].message.content` parsing are pinned
without a live LM Studio server. Unit tests never touch the network.
"""

from __future__ import annotations

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
