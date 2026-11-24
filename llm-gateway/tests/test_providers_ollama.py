"""Ollama adapter, native API (plan §3.2, §3.8; FR-4, FR-9, FR-15, FR-16, FR-17, FR-18; AC-1,
AC-11, AC-13; decision D-2a), executed against the stand-in server."""

import json

import pytest
from gwstandin import error
from provider_env import MSGS, SCHEMA_JSON, SCHEMA_RF, Env, json_reply

from llmgateway.providers import REGISTRY
from llmgateway.providers.base import BuildContext, SettingUnsupported
from llmgateway.types import CallFailed
from llmgateway.types import FailureClass as F

REPLY = {
    "model": "llama3:8b",
    "message": {"role": "assistant", "content": "hi", "thinking": "hmm"},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 13,
    "eval_count": 21,
    "total_duration": 5,
    "eval_duration": 4,
}


@pytest.fixture
def env(tmp_path):
    made = []

    def _make(replies=(json_reply(REPLY),), **kw):
        e = Env(tmp_path / f"e{len(made)}", "ollama", replies, **kw)
        made.append(e)
        return e

    yield _make
    for e in made:
        e.close()


def ctx(**kw):
    base = {"provider_id": "p", "config": {}, "model": "m1", "messages": MSGS}
    return BuildContext(**{**base, **kw})


P = REGISTRY.get("ollama")
HOSTED = {"key_ref": "k1", "pinned_models": ["m1"]}


def test_registered_provider_is_ollama():
    assert P.kind == "ollama"


def test_seed_and_schema_reach_the_wire_byte_exact_with_stream_false(env):
    """AC-11: the schema becomes `format` unchanged; `stream: false` is the only addition."""
    e = env()
    e.call(settings={"temperature": 0.0, "seed": 9007199254740993}, response_format=SCHEMA_RF)
    want = (
        '{"model":"m1","messages":[{"role":"user","content":"hello"}],'
        '"options":{"temperature":0.0,"seed":9007199254740993},'
        f'"format":{SCHEMA_JSON},"stream":false}}'
    )
    assert e.last == want.encode()
    assert e.sent[0]["path"] == "/api/chat"


def test_no_settings_means_no_options_and_never_a_num_ctx(env):
    e = env()
    e.call()
    assert json.loads(e.last) == {"model": "m1", "messages": [dict(MSGS[0])], "stream": False}


def test_settings_translate_under_options_with_max_tokens_as_num_predict(env):
    e = env()
    s = {
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "min_p": 0.05,
        "repeat_penalty": 1.0,
        "max_tokens": 384,
        "seed": 0,
        "stop": ["\n\n"],
    }
    e.call(settings=s)
    want = {k: v for k, v in s.items() if k != "max_tokens"} | {"num_predict": 384}
    assert json.loads(e.last)["options"] == want


@pytest.mark.parametrize("name", ["frequency_penalty", "presence_penalty"])
def test_penalties_ollama_does_not_document_fail_naming_them(env, name):
    e = env()
    exc = e.fail(settings={"temperature": 0, name: 0.5})
    assert exc.failure_class is F.CONFIG and name in str(exc) and e.sent == []


def test_optional_unsupported_is_skipped(env):
    e = env(optional=["presence_penalty"])
    e.call(settings={"presence_penalty": 0.5, "seed": 3})
    assert json.loads(e.last)["options"] == {"seed": 3}


def test_json_object_becomes_format_json_and_other_types_are_rejected(env):
    e = env()
    e.call(response_format={"type": "json_object"})
    assert json.loads(e.last)["format"] == "json"
    with pytest.raises(CallFailed, match="no Ollama equivalent") as exc:
        P.build(ctx(response_format={"type": "regex"}))
    assert exc.value.failure_class is F.REJECTED


def test_think_and_keep_alive_appear_only_when_configured(env):
    plain = env()
    plain.call()
    assert not {"think", "keep_alive"} & set(json.loads(plain.last))
    e = env(extra='think = false\nkeep_alive = "30m"\n')
    e.call()
    body = json.loads(e.last)
    assert body["think"] is False and body["keep_alive"] == "30m"


def test_tools_pass_through(env):
    e = env()
    tools = ({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}},)
    e.call(tools=tools)
    assert json.loads(e.last)["tools"] == list(tools)


def test_usage_is_prompt_eval_and_eval_count_thinking_as_reported(env):
    r = env().call()
    assert (r.usage.input_tokens, r.usage.output_tokens) == (13, 21)
    assert r.message["thinking"] == "hmm" and r.model == "llama3:8b"
    assert r.provider_detail["done_reason"] == "stop" and r.provider_detail["eval_duration"] == 4


def test_a_reply_without_prompt_eval_count_has_none_input_tokens(env):
    reply = {k: v for k, v in REPLY.items() if k != "prompt_eval_count"}
    r = env(replies=(json_reply(reply),)).call()
    assert (r.usage.input_tokens, r.usage.output_tokens) == (None, 21)


@pytest.mark.parametrize(
    "reply",
    [
        {**REPLY, "message": "text"},
        {**REPLY, "eval_count": -1},
        {**REPLY, "eval_count": "21"},
        {k: v for k, v in REPLY.items() if k != "model"},
    ],
)
def test_malformed_replies_fail_as_config(env, reply):
    assert env(replies=(json_reply(reply),)).fail().failure_class is F.CONFIG


@pytest.mark.parametrize(
    "reply, want",
    [
        (error(404, "model 'x' not found"), F.REJECTED),
        (error(503, "server busy"), F.OUTAGE),
        (error(429), F.RATE_LIMIT),
        (error(502), F.OUTAGE),
        (error(500), F.OUTAGE),
        (error(400, "invalid"), F.REJECTED),
        (error(401), F.AUTH),
    ],
)
def test_failure_classes_follow_the_shared_table(env, reply, want):
    assert env(replies=(reply,)).fail().failure_class is want


def test_error_body_with_a_plain_string_is_classified_by_status(env):
    body = (400, "application/json", b'{"error":"context length exceeded"}')
    assert env(replies=(body,)).fail().failure_class is F.REJECTED


# --- D-2a: parallelism ---


def test_parallel_in_provider_configuration_fails_naming_it_and_sends_nothing(env):
    e = env(extra="parallel = 4\n")
    exc = e.fail()
    assert exc.failure_class is F.CONFIG and "parallel" in str(exc) and e.sent == []


def test_parallel_as_a_load_setting_fails_naming_it_local_or_hosted():
    with pytest.raises(SettingUnsupported, match="parallel"):
        P.load_request("p", "m1", {"context_length": 4096, "parallel": 2}, {})
    with pytest.raises(SettingUnsupported, match="parallel"):
        P.load_request("p", "m1", {"parallel": 2}, HOSTED)


# --- load / residency support (D6) ---


def test_load_request_is_an_empty_chat_with_num_ctx_and_keep_alive():
    req = P.load_request("p", "m1", {"context_length": 8192, "keep_alive": "10m"}, {})
    assert (req.method, req.path) == ("POST", "/api/chat")
    assert req.body == (
        b'{"model":"m1","messages":[],"keep_alive":"10m","options":{"num_ctx":8192},"stream":false}'
    )
    assert json.loads(P.load_request("p", "m1", {}, {}).body) == {
        "model": "m1",
        "messages": [],
        "stream": False,
    }


def test_unknown_load_setting_fails_naming_it():
    with pytest.raises(SettingUnsupported, match="gpu_layers"):
        P.load_request("p", "m1", {"gpu_layers": 3}, {})


def test_hosted_load_settings_are_refused():
    with pytest.raises(CallFailed, match="context_length") as exc:
        P.load_request("p", "m1", {"context_length": 4096}, HOSTED)
    assert exc.value.failure_class is F.CONFIG


def test_loaded_request_and_parse_use_api_ps_with_context_length_and_raw():
    req = P.loaded_request()
    assert (req.method, req.path, req.body) == ("GET", "/api/ps", b"")
    rec = {"name": "llama3:8b", "model": "llama3:8b", "context_length": 4096, "size_vram": 1}
    got = P.parse_loaded(json.dumps({"models": [rec]}).encode())
    assert got == [{"id": "llama3:8b", "context_length": 4096, "raw": rec}]
    assert P.parse_loaded(b'{"models":[]}') == []


def test_ops_name_ps_tags_and_show():
    assert set(P.ops) == {"ollama.ps", "ollama.tags", "ollama.show"}
    assert P.ops["ollama.ps"]({}).path == "/api/ps" and P.ops["ollama.tags"]({}).method == "GET"
    show = P.ops["ollama.show"]({"model": "m1"})
    assert (show.method, show.path, show.body) == ("POST", "/api/show", b'{"model":"m1"}')


# --- hosted Ollama: chat only, pinned (FR-9) ---


def test_hosted_ollama_response_format_is_rejected_and_nothing_is_sent(env):
    e = env(hosted=True)
    exc = e.fail(settings={"max_tokens": 50}, response_format=SCHEMA_RF)
    assert exc.failure_class is F.REJECTED and "structured" in str(exc) and e.sent == []


def test_hosted_ollama_chat_sends_num_predict_and_stream_false(env):
    e = env(hosted=True)
    r = e.call(settings={"max_tokens": 50, "temperature": 0})
    assert r.hosted is True
    assert json.loads(e.last)["options"] == {"temperature": 0, "num_predict": 50}


@pytest.mark.parametrize("model", ["llama3-latest", "llama3", "m1 "])
def test_hosted_alias_or_unpinned_model_is_refused(model):
    with pytest.raises(CallFailed, match="not an exact pinned model") as exc:
        P.build(ctx(config=HOSTED, model=model))
    assert exc.value.failure_class is F.REJECTED


def test_hosted_keep_alive_is_refused_not_sent():
    with pytest.raises(CallFailed, match="keep_alive") as exc:
        P.build(ctx(config={**HOSTED, "keep_alive": "5m"}))
    assert exc.value.failure_class is F.CONFIG


@pytest.mark.parametrize("hosted", [False, True])
def test_float_settings_reach_the_wire_with_their_exact_values(env, hosted):
    """FR-18: no rounding, local and hosted, asserted on the raw bytes the stand-in received."""
    e = env(hosted=hosted)
    e.call(settings={"temperature": 0.35, "top_p": 0.955, "seed": 9007199254740993})
    assert b'"options":{"temperature":0.35,"top_p":0.955,"seed":9007199254740993}' in e.last


def test_hosted_ollama_body_is_byte_exact_on_the_wire(env):
    e = env(hosted=True)
    e.call(settings={"temperature": 0.0, "seed": 9007199254740993})
    want = (
        '{"model":"m1","messages":[{"role":"user","content":"hello"}],'
        '"options":{"temperature":0.0,"seed":9007199254740993},"stream":false}'
    )
    assert e.last == want.encode()
    assert e.sent[0]["path"] == "/api/chat"
