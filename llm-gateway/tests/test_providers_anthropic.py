"""Anthropic Messages adapter (plan §3.2, §3.4; FR-4, FR-9, FR-15, FR-18; AC-1, AC-11, AC-18),
executed against the stand-in server through a transport that carries the hosted step (the real
holder is D7/D8's)."""

import json

import pytest
from gwstandin import error
from provider_env import MSGS, SCHEMA_JSON, SCHEMA_RF, Env, json_reply

from llmgateway.providers import REGISTRY
from llmgateway.providers.base import BuildContext
from llmgateway.types import CallFailed
from llmgateway.types import FailureClass as F

MODEL = "claude-pinned-1"
REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": MODEL,
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 1000,
    },
}
S = {"max_tokens": 64}


@pytest.fixture
def env(tmp_path):
    made = []

    def _make(replies=(json_reply(REPLY),), **kw):
        e = Env(tmp_path / f"e{len(made)}", "anthropic", replies, hosted=True, model=MODEL, **kw)
        made.append(e)
        return e

    yield _make
    for e in made:
        e.close()


def ctx(**kw):
    base = {
        "provider_id": "p",
        "config": {"key_ref": "k1", "pinned_models": [MODEL]},
        "model": MODEL,
        "messages": MSGS,
        "settings": S,
    }
    return BuildContext(**{**base, **kw})


P = REGISTRY.get("anthropic")


def test_registered_provider_is_anthropic():
    assert P.kind == "anthropic"


def test_schema_and_settings_reach_the_wire_byte_exact_and_the_path_is_v1_messages(env):
    """AC-11: the schema is `output_config.format.schema` unchanged; no header is built."""
    e = env()
    e.call(settings={"max_tokens": 64, "temperature": 0.0, "top_k": 0}, response_format=SCHEMA_RF)
    want = (
        f'{{"model":"{MODEL}","max_tokens":64,"messages":[{{"role":"user","content":"hello"}}],'
        f'"temperature":0.0,"top_k":0,'
        f'"output_config":{{"format":{{"type":"json_schema","schema":{SCHEMA_JSON}}}}}}}'
    )
    assert e.last == want.encode()
    assert e.sent[0]["path"] == "/v1/messages"
    assert P.build(ctx()).headers == {}


def test_minimal_body_is_model_max_tokens_messages_only(env):
    e = env()
    e.call(settings=S)
    assert json.loads(e.last) == {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [dict(MSGS[0])],
    }


def test_system_messages_move_to_the_top_level_system_field():
    msgs = (
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": "again"},
    )
    body = json.loads(P.build(ctx(messages=msgs)).body)
    assert body["system"] == "be terse"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    two = json.loads(
        P.build(
            ctx(messages=({"role": "system", "content": "a"}, {"role": "system", "content": "b"}))
        ).body
    )
    assert two["system"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


def test_stop_becomes_stop_sequences_a_string_is_wrapped_a_list_is_unchanged():
    assert json.loads(P.build(ctx(settings={**S, "stop": ["a", "b"]})).body)["stop_sequences"] == [
        "a",
        "b",
    ]
    assert json.loads(P.build(ctx(settings={**S, "stop": "a"})).body)["stop_sequences"] == ["a"]


@pytest.mark.parametrize(
    "name", ["seed", "repeat_penalty", "frequency_penalty", "presence_penalty"]
)
def test_unsupported_required_setting_fails_naming_it_and_sends_nothing(env, name):
    """AC-1: e.g. `seed` on Anthropic."""
    e = env()
    exc = e.fail(settings={**S, "temperature": 0, name: 1})
    assert exc.failure_class is F.CONFIG and name in str(exc) and e.sent == []


def test_min_p_is_unsupported_and_optional_unsupported_is_skipped(env):
    assert "min_p" in str(env().fail(settings={**S, "min_p": 0.1}))
    e = env(optional=["seed"])
    e.call(settings={**S, "seed": 5})
    assert "seed" not in json.loads(e.last)


def test_max_tokens_is_required_and_the_failure_names_it(env):
    e = env()
    exc = e.fail(settings={"temperature": 0})
    assert exc.failure_class is F.CONFIG and "max_tokens" in str(exc) and e.sent == []


def test_usage_totals_input_with_cache_tokens_and_keeps_output_as_reported(env):
    """input_tokens excludes cache tokens (prompt-caching doc): total = 10 + 100 + 1000."""
    r = env().call(settings=S)
    assert (r.usage.input_tokens, r.usage.output_tokens) == (1110, 20)


def test_null_or_absent_cache_fields_count_as_absent(env):
    usage = {"input_tokens": 7, "output_tokens": 9, "cache_creation_input_tokens": None}
    r = env(replies=(json_reply({**REPLY, "usage": usage}),)).call(settings=S)
    assert (r.usage.input_tokens, r.usage.output_tokens) == (7, 9)


def test_thinking_output_tokens_are_taken_as_reported(env):
    usage = {"input_tokens": 1, "output_tokens": 4096}
    r = env(replies=(json_reply({**REPLY, "usage": usage}),)).call(settings=S)
    assert r.usage.output_tokens == 4096


def test_reply_model_is_the_recorded_identity_and_text_blocks_join(env):
    reply = {
        **REPLY,
        "model": "claude-pinned-1-20260101",
        "content": [
            {"type": "thinking", "thinking": "hm", "signature": "s"},
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ],
    }
    e = env(replies=(json_reply(reply),))
    r = e.call(settings=S)
    assert r.model == "claude-pinned-1-20260101" and r.hosted is True
    assert r.message == {"role": "assistant", "content": "ab"}
    assert r.raw["content"][0]["type"] == "thinking" and r.provider_detail == {
        "stop_reason": "end_turn"
    }
    (row,) = e.records.attempts()
    assert row["model"] == "claude-pinned-1-20260101"


@pytest.mark.parametrize(
    "reply",
    [
        {**REPLY, "content": "text"},
        {k: v for k, v in REPLY.items() if k != "usage"},
        {**REPLY, "usage": {"input_tokens": "1", "output_tokens": 1}},
        {**REPLY, "content": [{"type": "text", "text": 5}]},
    ],
)
def test_malformed_replies_fail_as_config(env, reply):
    assert env(replies=(json_reply(reply),)).fail(settings=S).failure_class is F.CONFIG


@pytest.mark.parametrize("model", ["claude-sonnet-latest", "claude-sonnet", MODEL + " ", "other"])
def test_alias_or_unpinned_model_is_refused_before_anything_is_built(model):
    with pytest.raises(CallFailed, match="not an exact pinned model") as exc:
        P.build(ctx(model=model))
    assert exc.value.failure_class is F.REJECTED


def test_a_pinned_model_listed_by_the_owner_is_accepted():
    cfg = {"key_ref": "k1", "pinned_models": [MODEL, "claude-other-9"]}
    assert json.loads(P.build(ctx(config=cfg, model="claude-other-9")).body)["model"] == (
        "claude-other-9"
    )


def test_tools_and_non_schema_response_formats_are_refused_not_dropped():
    with pytest.raises(CallFailed, match="tools") as exc:
        P.build(ctx(tools=({"type": "function"},)))
    assert exc.value.failure_class is F.CONFIG
    with pytest.raises(CallFailed, match="no Anthropic equivalent") as exc:
        P.build(ctx(response_format={"type": "json_object"}))
    assert exc.value.failure_class is F.REJECTED


def typed(status, error_type):
    body = {"type": "error", "error": {"type": error_type, "message": "m"}}
    return json_reply(body, status)


@pytest.mark.parametrize(
    "reply, want",
    [
        (
            (529, "application/json", b'{"type":"error","error":{"type":"overloaded_error"}}'),
            F.OUTAGE,
        ),
        (
            (402, "application/json", b'{"type":"error","error":{"type":"billing_error"}}'),
            F.OUT_OF_CREDIT,
        ),
        (error(429), F.RATE_LIMIT),
        (error(401), F.AUTH),
        (error(403), F.PERMISSION),
        (error(413), F.REJECTED),
        (error(404), F.REJECTED),
        (error(500), F.OUTAGE),
        (error(504), F.TIMEOUT),
        (error(400, "bad schema"), F.REJECTED),
    ],
)
def test_failure_classes_follow_the_shared_table(env, reply, want):
    assert env(replies=(reply,)).fail(settings=S).failure_class is want


def test_a_spend_limit_400_is_out_of_credit_only_through_the_owner_pattern(env):
    body = (
        400,
        "application/json",
        b'{"type":"error","error":{"type":"invalid_request_error",'
        b'"message":"You have reached your specified API usage limits."}}',
    )
    assert env(replies=(body,)).fail(settings=S).failure_class is F.REJECTED
    e = env(replies=(body,), extra='out_of_credit_patterns = ["api usage limits"]\n')
    assert e.fail(settings=S).failure_class is F.OUT_OF_CREDIT
