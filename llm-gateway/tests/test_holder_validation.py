"""Holder request validation, usage tables, address rules, scrubbing, start policy
(plan §3.4 "Request validation"; §5.2 FR-8 (7), FR-10 (6))."""

import base64
import json
import urllib.parse

import pytest

from llmgateway.config import GatewayConfig
from llmgateway.keyholder import (
    BodyRefused,
    HolderStartError,
    Provider,
    address_allowed,
    build_policy,
    is_json_error_shape,
    key_forms,
    parse_usage,
    releasable,
    scrub,
    validate_body,
)


def prov(kind="anthropic", **over):
    base = dict(
        id="p",
        kind=kind,
        base_url="https://api.example",
        key_ref="/k",
        anthropic_version="v",
        pinned_models=frozenset({"m-20250101"}),
        max_tokens_ceiling=1000,
        allow_private=False,
        extra_body_keys=frozenset(),
    )
    base.update(over)
    return Provider(**base)


def refused(provider, doc):
    body = doc if isinstance(doc, bytes) else json.dumps(doc).encode()
    with pytest.raises(BodyRefused) as err:
        validate_body(provider, body)
    return err.value.code


MSGS = [{"role": "user", "content": "x"}]
ANTH = {"model": "m-20250101", "max_tokens": 100, "messages": MSGS}
OAI = {"model": "m-20250101", "max_tokens": 100, "messages": MSGS}
OLL = {"model": "m-20250101", "options": {"num_predict": 100}, "messages": MSGS}


@pytest.mark.parametrize(
    "kind,doc,limit",
    [
        ("anthropic", ANTH, 100),
        ("openai_compat", OAI, 100),
        (
            "lmstudio",
            {**OAI, "max_completion_tokens": 7} | {"max_tokens": None}
            and {"model": "m-20250101", "messages": MSGS, "max_completion_tokens": 7},
            7,
        ),
        ("ollama", OLL, 100),
    ],
)
def test_valid_body_per_kind_yields_the_limit(kind, doc, limit):
    assert validate_body(prov(kind), json.dumps(doc).encode()) == (limit, "m-20250101")


@pytest.mark.parametrize(
    "kind,doc",
    [
        ("anthropic", {"model": "m-20250101", "messages": MSGS}),
        ("openai_compat", {"model": "m-20250101", "messages": MSGS}),
        ("ollama", {"model": "m-20250101", "messages": MSGS}),  # hosted Ollama with no num_predict
        ("ollama", {"model": "m-20250101", "messages": MSGS, "options": {"temperature": 0}}),
    ],
)
def test_absent_limit_is_refused_never_defaulted(kind, doc):
    assert refused(prov(kind), doc) == "limit_missing"


def test_over_ceiling_is_refused_not_clamped():
    assert refused(prov(), {**ANTH, "max_tokens": 1001}) == "limit_over_ceiling"
    assert (
        refused(prov("ollama"), {**OLL, "options": {"num_predict": 5000}}) == "limit_over_ceiling"
    )
    assert validate_body(prov(), json.dumps({**ANTH, "max_tokens": 1000}).encode())[0] == 1000


@pytest.mark.parametrize("bad", [0, -5, 1.5, "100", True, None, [100]])
def test_ill_typed_limit_is_body_invalid(bad):
    assert refused(prov(), {**ANTH, "max_tokens": bad}) == "body_invalid"


def test_both_openai_aliases_refused():
    doc = {"model": "m-20250101", "messages": MSGS, "max_tokens": 5, "max_completion_tokens": 5}
    assert refused(prov("openai_compat"), doc) == "body_invalid"
    assert refused(prov("lmstudio"), doc) == "body_invalid"


@pytest.mark.parametrize(
    "kind,doc",
    [
        ("anthropic", {**ANTH, "max_completion_tokens": 5}),  # another kind's alias
        ("anthropic", {**ANTH, "options": {"num_predict": 5}}),
        ("anthropic", {**ANTH, "output_config": {"max_tokens": 9999999}}),  # nested alias
        ("openai_compat", {**OAI, "response_format": {"max_tokens": 1}}),
        ("ollama", {**OLL, "max_tokens": 5}),
        ("ollama", {**OLL, "options": {"num_predict": 5, "max_tokens": 9}}),
        ("ollama", {**OLL, "format": {"num_predict": 9}}),
    ],
)
def test_aliases_and_nested_limit_keys_refused(kind, doc):
    assert refused(prov(kind), doc) == "body_invalid"


def test_duplicate_keys_at_any_depth_and_non_json_refused():
    assert (
        refused(prov(), b'{"model":"m-20250101","max_tokens":1,"max_tokens":2,"messages":[]}')
        == "body_invalid"
    )
    assert (
        refused(
            prov(), b'{"model":"m-20250101","max_tokens":1,"messages":[{"role":"u","role":"v"}]}'
        )
        == "body_invalid"
    )
    assert (
        refused(
            prov(),
            b'{"model":"m-20250101","max_tokens":1,"messages":[],"output_config":{"a":1,"a":1}}',
        )
        == "body_invalid"
    )
    for junk in (b"not json", b"[]", b'"x"', b"", b"\xff\xfe", b'{"model":NaN}'):
        assert refused(prov(), junk) == "body_invalid"


@pytest.mark.parametrize(
    "key", ["tools", "cache_control", "thinking", "stream_options", "metadata", "user"]
)
def test_keys_outside_the_allow_list_refused(key):
    assert refused(prov(), {**ANTH, key: {}}) == "body_invalid"


def test_cache_control_and_tools_refused_anywhere():
    msg = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}],
        }
    ]
    assert refused(prov(), {**ANTH, "messages": msg}) == "body_invalid"
    assert (
        refused(prov(), {**ANTH, "system": [{"type": "text", "text": "s", "cache_control": {}}]})
        == "body_invalid"
    )
    assert (
        refused(prov("openai_compat"), {**OAI, "response_format": {"tools": 1}}) == "body_invalid"
    )


def test_extra_body_keys_opt_in_is_owner_only():
    p = prov(extra_body_keys=frozenset({"tools", "thinking"}))
    doc = {**ANTH, "tools": [{"name": "t"}], "thinking": {"type": "enabled", "budget_tokens": 5}}
    assert validate_body(p, json.dumps(doc).encode())[0] == 100
    assert refused(p, {**ANTH, "cache_control": {}}) == "body_invalid"  # still not listed


def test_stream_must_be_false():
    assert refused(prov(), {**ANTH, "stream": True}) == "body_invalid"
    assert validate_body(prov(), json.dumps({**ANTH, "stream": False}).encode())[0] == 100


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-5-latest",
        "claude-sonnet-4-5",
        "m-2025010",
        "m-20250101 ",
        "M-20250101",
        "m-2025*",
        "",
        None,
        5,
        ["m-20250101"],
    ],
)
def test_model_must_be_an_exact_pinned_member(model):
    assert refused(prov(), {**ANTH, "model": model}) == "model_not_pinned"
    assert refused(prov(), {k: v for k, v in ANTH.items() if k != "model"}) == "model_not_pinned"


# --- usage ---


def usage(kind, doc):
    return parse_usage(prov(kind), json.dumps(doc).encode())


def test_anthropic_usage_settles_every_field_including_cache_tokens():
    doc = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 4000,
        }
    }
    assert usage("anthropic", doc) == 4330
    assert usage("anthropic", {"usage": {"input_tokens": 1, "output_tokens": 2}}) == 3


def test_openai_and_ollama_usage():
    assert usage("openai_compat", {"usage": {"prompt_tokens": 5, "completion_tokens": 6}}) == 11
    assert (
        usage(
            "lmstudio", {"usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 99}}
        )
        == 11
    )
    assert usage("ollama", {"prompt_eval_count": 8, "eval_count": 9}) == 17


@pytest.mark.parametrize(
    "kind,doc",
    [
        ("anthropic", {}),
        ("anthropic", {"usage": {"input_tokens": 1}}),
        ("anthropic", {"usage": {"input_tokens": 1, "output_tokens": "2"}}),
        ("anthropic", {"usage": {"input_tokens": 1, "output_tokens": -2}}),
        ("anthropic", {"usage": {"input_tokens": True, "output_tokens": 2}}),
        (
            "anthropic",
            {"usage": {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": "x"}},
        ),
        ("openai_compat", {"usage": {"prompt_tokens": 1}}),
        ("ollama", {"eval_count": 5}),
        ("ollama", [1]),
    ],
)
def test_unparseable_usage_is_none(kind, doc):
    assert usage(kind, doc) is None


def test_non_json_usage_is_none():
    assert parse_usage(prov(), b"<html>") is None


# --- release classification ---


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 422, 429, 500, 502, 503, 529])
def test_json_error_responses_are_releasable(status):
    assert releasable(status, b'{"type":"error","error":{"type":"x"}}')


@pytest.mark.parametrize("status", [408, 504, 524])
def test_edge_timeouts_are_never_released(status):
    assert not releasable(status, b'{"error":{}}')


@pytest.mark.parametrize("status", [500, 502, 503, 529])
def test_5xx_without_the_provider_error_shape_is_kept(status):
    for body in (b"<html>gateway</html>", b"", b"[]", b'{"message":"x"}'):
        assert not releasable(status, body)


def test_4xx_releasable_regardless_of_body_and_success_never():
    assert releasable(429, b"<html>")
    assert not releasable(200, b'{"error":{}}')
    assert not releasable(302, b"")
    assert is_json_error_shape(b'{"error":"x"}') and not is_json_error_shape(b"nope")


# --- addresses ---


@pytest.mark.parametrize(
    "ip,allow,ok",
    [
        ("8.8.8.8", False, True),
        ("2606:4700::1111", False, True),
        ("127.0.0.1", False, False),
        ("127.0.0.1", True, True),
        ("::1", False, False),
        ("::1", True, True),
        ("10.0.0.5", False, False),
        ("192.168.1.1", False, False),
        ("172.16.0.1", False, False),
        ("10.0.0.5", True, True),
        ("100.64.0.1", False, False),
        ("169.254.169.254", False, False),
        ("169.254.169.254", True, False),  # metadata: never
        ("169.254.1.1", True, False),
        ("fd00:ec2::254", True, False),
        ("fe80::1", True, False),
        ("::ffff:169.254.169.254", True, False),
        ("::ffff:127.0.0.1", False, False),
        ("0.0.0.0", True, False),
        ("224.0.0.1", True, False),
    ],
)
def test_address_rules(ip, allow, ok):
    assert address_allowed(ip, allow) is ok


# --- scrubbing ---

KEY = "sk-ant-FAKE-Zq9/+Ab=c-1234567890"


def test_scrub_removes_raw_url_and_base64_forms():
    raw = KEY.encode()
    variants = [
        raw,
        urllib.parse.quote(KEY, safe="").encode(),
        urllib.parse.quote_plus(KEY).encode(),
        base64.b64encode(raw),
        base64.urlsafe_b64encode(raw),
        raw.hex().encode(),
    ]
    for v in variants:
        out = scrub(b"before " + v + b" after", KEY)
        assert v not in out, v
    for shift in (1, 2):  # the key encoded mid-stream, at each base64 alignment
        full = base64.b64encode(b"ab"[:shift] + raw)
        out = scrub(b"x" + full + b"y", KEY)
        assert full[4:-4] not in out, shift
    assert scrub(b"nothing here", KEY) == b"nothing here"
    assert scrub(b"x", None) == b"x"
    assert all(len(f) >= 8 for f in key_forms(KEY))


# --- start policy ---


def cfg(providers=None, holder=None, run_bounds=(), protected=True):
    return GatewayConfig(
        protected=protected,
        providers=providers
        if providers is not None
        else {
            "p": {
                "kind": "anthropic",
                "base_url": "https://api.example",
                "key_ref": "/k",
                "pinned_models": ["m-1"],
                "max_tokens_ceiling": 100,
            }
        },
        holder=holder
        if holder is not None
        else {"window_seconds": 60, "window_max_tokens": 10, "window_max_released": 2},
        run_bounds=tuple(run_bounds),
    )


def test_policy_builds_for_a_valid_config():
    pol = build_policy(cfg())
    assert set(pol.providers) == {"https://api.example"}
    assert pol.ceilings.window_seconds == 60 and pol.unmatched == "refuse"


@pytest.mark.parametrize(
    "holder",
    [
        {},
        {"window_seconds": 60, "window_max_tokens": 10},
        {"window_seconds": 0, "window_max_tokens": 10, "window_max_released": 2},
        {"window_seconds": 60, "window_max_tokens": -1, "window_max_released": 2},
        {"window_seconds": 60.5, "window_max_tokens": 10, "window_max_released": 2},
        {"window_seconds": "60", "window_max_tokens": 10, "window_max_released": 2},
        {"window_seconds": 31 * 86400, "window_max_tokens": 10, "window_max_released": 2},
        {"window_seconds": True, "window_max_tokens": 10, "window_max_released": 2},
    ],
)
def test_hosted_providers_without_a_valid_ceiling_refuse_to_start(holder):
    with pytest.raises(HolderStartError):
        build_policy(cfg(holder=holder))


def test_no_hosted_provider_needs_no_ceiling():
    assert build_policy(cfg(providers={}, holder={})).providers == {}


def test_unprotected_config_refuses_to_start():
    with pytest.raises(HolderStartError, match="owner-protected"):
        build_policy(cfg(protected=False))


@pytest.mark.parametrize(
    "holder_extra,bounds",
    [
        ({"unmatched": "allow"}, ()),
        ({"unmatched": "cap"}, ()),  # cap without default_bound
        ({"unmatched": "cap", "default_bound": 0}, ()),
        ({}, [{"match": "", "max_tokens": 5}]),
        ({}, [{"match": "eval-*", "max_tokens": 0}]),
        ({}, [{"match": "eval-*"}]),
        ({"ca_file": 5}, ()),
        ({"upstream_timeout_seconds": 0}, ()),
    ],
)
def test_invalid_bounds_refuse_to_start(holder_extra, bounds):
    base = {"window_seconds": 60, "window_max_tokens": 10, "window_max_released": 2}
    with pytest.raises(HolderStartError):
        build_policy(cfg(holder={**base, **holder_extra}, run_bounds=bounds))


@pytest.mark.parametrize(
    "over",
    [
        {"kind": "mystery"},
        {"pinned_models": []},
        {"pinned_models": ["a", 3]},
        {"max_tokens_ceiling": 0},
        {"max_tokens_ceiling": None},
        {"key_ref": ""},
        {"extra_body_keys": "tools"},
        {"allow_private": "yes"},
    ],
)
def test_invalid_provider_refuses_to_start(over):
    p = {
        "kind": "anthropic",
        "base_url": "https://api.example",
        "key_ref": "/k",
        "pinned_models": ["m-1"],
        "max_tokens_ceiling": 100,
    }
    with pytest.raises(HolderStartError):
        build_policy(cfg(providers={"p": {**p, **over}}))
