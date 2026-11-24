"""The adapters agree with the holder's closed per-kind table (plan §3.4): a body each adapter
builds for a hosted step passes `keyholder.validate_body`, and the usage an adapter reports
equals what the holder settles (`keyholder.parse_usage`). Read-only use of D7's module."""

import json

import pytest

from llmgateway import keyholder
from llmgateway.providers import REGISTRY
from llmgateway.providers.base import BuildContext

MSGS = ({"role": "user", "content": "hello"},)
RF = {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "object"}}}
MODEL = "pinned-1"
CASES = {
    "anthropic": (
        {"temperature": 0, "top_p": 1, "top_k": 0, "max_tokens": 384, "stop": ["x"]},
        RF,
        {
            "model": MODEL,
            "content": [{"type": "text", "text": "x"}],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "cache_creation_input_tokens": 7,
                "cache_read_input_tokens": 11,
            },
        },
    ),
    "openai_compat": (
        {
            "temperature": 0,
            "top_p": 1,
            "top_k": 0,
            "repeat_penalty": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 384,
            "seed": 0,
            "stop": ["x"],
        },
        RF,
        {
            "model": MODEL,
            "choices": [{"message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        },
    ),
    "ollama": (
        {"temperature": 0, "top_p": 1, "top_k": 0, "max_tokens": 384, "seed": 0, "stop": ["x"]},
        None,
        {
            "model": MODEL,
            "message": {"role": "assistant", "content": "x"},
            "prompt_eval_count": 3,
            "eval_count": 5,
        },
    ),
}


def holder_provider(kind):
    return keyholder.Provider(
        id="p",
        kind=kind,
        base_url="https://api.example.com",
        key_ref="k",
        pinned_models=frozenset({MODEL}),
        max_tokens_ceiling=1000,
        anthropic_version="2023-06-01",
        allow_private=False,
        extra_body_keys=frozenset(),
    )


def ctx(settings, rf):
    cfg = {"key_ref": "k", "pinned_models": [MODEL]}
    return BuildContext("p", cfg, MODEL, MSGS, settings, frozenset(), None, rf)


@pytest.mark.parametrize("kind", sorted(CASES))
def test_a_full_hosted_body_passes_the_holder_validation_with_its_limit(kind):
    settings, rf, _ = CASES[kind]
    req = REGISTRY.get(kind).build(ctx(settings, rf))
    limit, model = keyholder.validate_body(holder_provider(kind), req.body)
    assert (limit, model) == (384, MODEL)
    assert req.path in keyholder.KINDS[kind].paths


@pytest.mark.parametrize("kind", ["openai_compat", "ollama"])
def test_a_missing_limit_is_the_holders_to_refuse_not_the_adapters_to_invent(kind):
    settings, rf, _ = CASES[kind]
    settings = {k: v for k, v in settings.items() if k != "max_tokens"}
    req = REGISTRY.get(kind).build(ctx(settings, rf))
    with pytest.raises(keyholder.BodyRefused) as exc:
        keyholder.validate_body(holder_provider(kind), req.body)
    assert exc.value.code == "limit_missing"


@pytest.mark.parametrize("kind", sorted(CASES))
def test_reported_usage_equals_what_the_holder_settles(kind):
    _, _, reply = CASES[kind]
    body = json.dumps(reply).encode()
    parsed = REGISTRY.get(kind).parse(body, ctx({}, None))
    settled = keyholder.parse_usage(holder_provider(kind), body)
    assert (
        settled
        == parsed.usage.input_tokens + parsed.usage.output_tokens
        == 3 + 5 + (18 if kind == "anthropic" else 0)
    )
