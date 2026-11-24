"""`openai_compat` and `lmstudio` adapters (plan §3.2, §3.8; FR-4, FR-9, FR-15, FR-16, FR-17,
FR-18; AC-1, AC-8, AC-11, AC-13, AC-20), executed against the stand-in server."""

import json

import pytest
from gwstandin import error
from provider_env import MSGS, SCHEMA_JSON, SCHEMA_RF, Env, json_reply

from llmgateway.providers import REGISTRY
from llmgateway.providers.base import BuildContext, SettingUnsupported
from llmgateway.types import CallFailed
from llmgateway.types import FailureClass as F

KINDS = ["openai_compat", "lmstudio"]
REPLY = {
    "model": "reported-1",
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
}


@pytest.fixture
def env(tmp_path):
    made = []

    def _make(kind="openai_compat", replies=(json_reply(REPLY),), **kw):
        e = Env(tmp_path / f"e{len(made)}", kind, replies, **kw)
        made.append(e)
        return e

    yield _make
    for e in made:
        e.close()


def ctx(**kw):
    base = {"provider_id": "p", "config": {}, "model": "m1", "messages": MSGS}
    return BuildContext(**{**base, **kw})


def test_registered_providers_are_of_their_kind():
    for kind in KINDS:
        assert REGISTRY.get(kind).kind == kind


@pytest.mark.parametrize("kind", KINDS)
def test_seed_and_schema_reach_the_wire_byte_exact(env, kind):
    """AC-11 / FR-18: asserted on the raw bytes the server received, nothing else in the body."""
    e = env(kind)
    e.call(settings={"temperature": 0.0, "seed": 9007199254740993}, response_format=SCHEMA_RF)
    want = (
        '{"model":"m1","messages":[{"role":"user","content":"hello"}],'
        '"temperature":0.0,"seed":9007199254740993,'
        f'"response_format":{{"type":"json_schema","json_schema":{{"name":"ans","schema":{SCHEMA_JSON}}}}}}}'
    )
    assert e.last == want.encode()
    assert e.sent[0]["path"] == "/v1/chat/completions"


@pytest.mark.parametrize("kind", KINDS)
def test_no_settings_means_only_model_and_messages(env, kind):
    e = env(kind)
    e.call()
    assert json.loads(e.last) == {"model": "m1", "messages": [dict(MSGS[0])]}


@pytest.mark.parametrize("kind", KINDS)
def test_supported_settings_map_one_to_one(env, kind):
    e = env(kind)
    s = {
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "repeat_penalty": 1.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 384,
        "seed": 0,
        "stop": ["\n\n"],
    }
    e.call(settings=s)
    assert json.loads(e.last) == {"model": "m1", "messages": [dict(MSGS[0])], **s}


@pytest.mark.parametrize("kind", KINDS)
def test_required_unsupported_setting_fails_naming_it_and_sends_nothing(env, kind):
    e = env(kind)
    exc = e.fail(settings={"temperature": 0, "min_p": 0.05})
    assert exc.failure_class is F.CONFIG and "min_p" in str(exc)
    assert e.sent == []


@pytest.mark.parametrize("kind", KINDS)
def test_optional_unsupported_setting_is_skipped(env, kind):
    e = env(kind, optional=["min_p"])
    e.call(settings={"temperature": 0, "min_p": 0.05})
    assert json.loads(e.last)["temperature"] == 0 and "min_p" not in json.loads(e.last)


@pytest.mark.parametrize("kind", KINDS)
def test_tools_pass_through_unchanged(env, kind):
    e = env(kind)
    tools = ({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}},)
    e.call(tools=tools)
    assert json.loads(e.last)["tools"] == list(tools)


@pytest.mark.parametrize("kind", KINDS)
def test_reply_identity_message_and_usage_as_reported(env, kind):
    r = env(kind).call()
    assert (r.model, r.message["content"]) == ("reported-1", "hi")
    assert (r.usage.input_tokens, r.usage.output_tokens) == (11, 7)
    assert r.provider_detail["finish_reason"] == "stop" and r.raw["model"] == "reported-1"


def test_reasoning_tokens_count_as_reported_in_completion_tokens(env):
    reply = {**REPLY, "usage": {"prompt_tokens": 5, "completion_tokens": 900}}
    assert env(replies=(json_reply(reply),)).call().usage.output_tokens == 900


def test_absent_usage_is_none_not_zero(env):
    reply = {k: v for k, v in REPLY.items() if k != "usage"}
    r = env(replies=(json_reply(reply),)).call()
    assert (r.usage.input_tokens, r.usage.output_tokens) == (None, None)


@pytest.mark.parametrize(
    "reply",
    [
        {**REPLY, "choices": []},
        {**REPLY, "usage": {"prompt_tokens": "7", "completion_tokens": 1}},
        {**REPLY, "usage": {"prompt_tokens": True, "completion_tokens": 1}},
        {**REPLY, "model": ""},
        {"error": "shape"},
    ],
)
def test_malformed_replies_fail_as_config_not_a_crash(env, reply):
    exc = env(replies=(json_reply(reply),)).fail()
    assert exc.failure_class is F.CONFIG


def test_lmstudio_stats_runtime_and_model_info_are_returned_verbatim(env):
    extra = {
        "stats": {"tokens_per_second": 41.5, "stop_reason": "eosFound"},
        "model_info": {"arch": "llama", "quant": "Q4_K_M", "context_length": 4096},
        "runtime": {"name": "llama.cpp", "version": "1.2.3"},
    }
    r = env("lmstudio", replies=(json_reply({**REPLY, **extra}),)).call()
    for key, value in extra.items():
        assert r.provider_detail[key] == value


def test_openai_compat_returns_no_lmstudio_detail_and_has_no_residency_or_ops(env):
    r = env(replies=(json_reply({**REPLY, "stats": {"x": 1}}),)).call()
    assert "stats" not in r.provider_detail
    p = REGISTRY.get("openai_compat")
    assert p.ops == {} and p.residency is None
    assert not hasattr(p, "loaded_request") and not hasattr(p, "load_request")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "reply, want",
    [
        (error(503), F.OUTAGE),
        (error(429), F.RATE_LIMIT),
        (error(401), F.AUTH),
        (error(400, "context overflow"), F.REJECTED),
        (error(404, "Model not found"), F.REJECTED),
        (error(400, "No models loaded. Please load a model."), F.MODEL_NOT_LOADED),
    ],
)
def test_failure_classes_follow_the_shared_table(env, kind, reply, want):
    assert env(kind, replies=(reply,)).fail().failure_class is want


# --- LM Studio residency support and ops (D6 / FR-17) ---


def lm():
    return REGISTRY.get("lmstudio")


MODELS_V0 = {
    "object": "list",
    "data": [
        {"id": "a", "type": "llm", "state": "not-loaded", "max_context_length": 8192},
        {
            "id": "b",
            "type": "llm",
            "publisher": "p",
            "arch": "llama",
            "quantization": "Q4_K_M",
            "state": "loaded",
            "loaded_context_length": 4096,
        },
        {"id": "emb", "type": "embeddings", "state": "loaded", "loaded_context_length": 512},
    ],
}


def test_lmstudio_loaded_reads_the_v0_models_path_ccg_allows():
    req = lm().loaded_request()
    assert (req.method, req.path, req.body) == ("GET", "/api/v0/models", b"")


def test_lmstudio_parse_loaded_keeps_loaded_entries_verbatim():
    got = lm().parse_loaded(json.dumps(MODELS_V0).encode())
    assert [m["id"] for m in got] == ["b", "emb"]
    assert got[0] == {"id": "b", "context_length": 4096, "raw": MODELS_V0["data"][1]}


def test_lmstudio_load_request_body_is_exact_and_unknown_settings_fail_naming_them():
    req = lm().load_request("p", "m1", {"context_length": 4096})
    assert (req.method, req.path) == ("POST", "/api/v1/models/load")
    assert req.body == b'{"model":"m1","context_length":4096}'
    with pytest.raises(SettingUnsupported, match="parallel"):
        lm().load_request("p", "m1", {"context_length": 4096, "parallel": 4})


def test_lmstudio_ops_are_named_and_the_decisions_body_passes_through():
    ops = lm().ops
    assert set(ops) == {"lmstudio.models", "lmstudio.decisions"}
    models = ops["lmstudio.models"]({})
    assert (models.method, models.path) == ("GET", "/api/v0/models")
    params = {"model": "dm", "input": "I was charged twice.", "questions": [{"name": "angry"}]}
    dec = ops["lmstudio.decisions"](params)
    assert (dec.method, dec.path) == ("POST", "/v1/decisions")
    assert dec.body == (
        b'{"model":"dm","input":"I was charged twice.","questions":[{"name":"angry"}]}'
    )


# --- hosted generic endpoint: FR-9 pin refusal owned by the adapter ---


@pytest.mark.parametrize("model", ["gpt-latest", "unpinned-1", "GPT-1", "gpt-1 "])
def test_hosted_alias_or_unpinned_model_is_refused_before_anything_is_built(model):
    p = REGISTRY.get("openai_compat")
    with pytest.raises(CallFailed, match="not an exact pinned model") as exc:
        p.build(ctx(config={"key_ref": "k1", "pinned_models": ["gpt-1"]}, model=model))
    assert exc.value.failure_class is F.REJECTED


def test_hosted_exact_pinned_model_is_built_and_local_is_not_checked():
    p = REGISTRY.get("openai_compat")
    body = p.build(ctx(config={"key_ref": "k1", "pinned_models": ["gpt-1"]}, model="gpt-1")).body
    assert json.loads(body)["model"] == "gpt-1"
    assert json.loads(p.build(ctx(model="whatever")).body)["model"] == "whatever"


def test_hosted_call_through_a_forwarding_transport_names_the_pinned_model(env):
    e = env(hosted=True, model="m1")
    r = e.call(settings={"max_tokens": 50})
    assert r.hosted is True and json.loads(e.last)["max_tokens"] == 50
