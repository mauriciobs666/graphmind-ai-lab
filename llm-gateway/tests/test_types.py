"""D1 types: failure classes, Request shape and refusals, errors (plan §3.2, §3.3)."""

import dataclasses

import pytest

from llmgateway.types import (
    NEUTRAL_SETTINGS,
    PERMANENT,
    TEMPORARY,
    CallFailed,
    CallOptions,
    EmbeddingsNotAvailable,
    FailureClass,
    GatewayError,
    HostedFromCallerProcess,
    LoadMode,
    Request,
)

MSG = ({"role": "user", "content": "hi"},)


def test_failure_classes_match_the_plan_table():
    assert {c.value for c in TEMPORARY} == {
        "rate_limit", "out_of_credit", "outage", "timeout", "connection", "model_not_loaded",
    }
    assert {c.value for c in PERMANENT} == {
        "auth", "permission", "rejected", "policy_refused", "run_bound", "config",
    }


def test_temporary_and_permanent_partition_every_class():
    assert TEMPORARY.isdisjoint(PERMANENT)
    assert TEMPORARY | PERMANENT == set(FailureClass)


def test_temporary_property_follows_the_sets():
    assert FailureClass.RATE_LIMIT.temporary is True
    assert FailureClass.AUTH.temporary is False
    assert FailureClass.RUN_BOUND.temporary is False


def test_load_modes():
    assert {m.value for m in LoadMode} == {"on_demand", "explicit", "strict"}


def test_neutral_settings_are_exactly_the_plan_list():
    assert NEUTRAL_SETTINGS == {
        "temperature", "top_p", "top_k", "min_p", "repeat_penalty", "frequency_penalty",
        "presence_penalty", "max_tokens", "seed", "stop",
    }


def test_request_has_no_model_or_provider_field():
    names = {f.name for f in dataclasses.fields(Request)}
    assert names == {
        "call_site", "messages", "scopes", "run_id", "tools", "settings",
        "response_format", "options", "load_mode",
    }
    assert "model" not in names and "provider" not in names


def test_request_defaults_retry_and_fallback_on():
    r = Request(call_site="ccg.interpret", messages=MSG)
    assert r.options == CallOptions(retry=True, fallback=True)
    assert r.scopes == () and r.run_id is None and r.load_mode is None


def test_request_is_frozen():
    r = Request(call_site="x", messages=MSG)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.call_site = "y"  # type: ignore[misc]


def test_request_rejects_empty_call_site():
    with pytest.raises(ValueError, match="call_site"):
        Request(call_site="", messages=MSG)


def test_request_rejects_unknown_setting_naming_it():
    with pytest.raises(ValueError, match="mirostat"):
        Request(call_site="x", messages=MSG, settings={"seed": 1, "mirostat": 2})


def test_request_accepts_neutral_settings_unchanged():
    s = {"seed": 7, "temperature": 0.0, "max_tokens": 10}
    assert Request(call_site="x", messages=MSG, settings=s).settings == s


def test_from_dict_refuses_unknown_field():
    with pytest.raises(ValueError, match="marker"):
        Request.from_dict({"call_site": "x", "messages": [], "marker": "unrestricted"})


def test_from_dict_refuses_model_and_provider():
    for bad in ("model", "provider"):
        with pytest.raises(ValueError, match=bad):
            Request.from_dict({"call_site": "x", "messages": [], bad: "claude"})


def test_from_dict_refuses_unknown_option_field():
    with pytest.raises(ValueError, match="hosted"):
        Request.from_dict({"call_site": "x", "messages": [], "options": {"hosted": True}})


def test_from_dict_builds_a_full_request():
    r = Request.from_dict(
        {
            "call_site": "x",
            "messages": [{"role": "user", "content": "a"}],
            "scopes": ["ccg:repo:a"],
            "run_id": "eval-1",
            "tools": [{"name": "t"}],
            "settings": {"seed": 1},
            "response_format": {"type": "json_schema"},
            "options": {"retry": False, "fallback": False},
            "load_mode": "strict",
        }
    )
    assert r.messages == ({"role": "user", "content": "a"},)
    assert r.scopes == ("ccg:repo:a",)
    assert r.tools == ({"name": "t"},)
    assert r.options == CallOptions(retry=False, fallback=False)
    assert r.load_mode is LoadMode.STRICT
    assert r.run_id == "eval-1"


def test_from_dict_rejects_bad_load_mode():
    with pytest.raises(ValueError):
        Request.from_dict({"call_site": "x", "messages": [], "load_mode": "always"})


def test_errors_share_a_base():
    for exc in (EmbeddingsNotAvailable, HostedFromCallerProcess, CallFailed):
        assert issubclass(exc, GatewayError)


def test_call_failed_carries_class_and_code():
    e = CallFailed(FailureClass.RUN_BOUND, "refused", code="window_ceiling_reached")
    assert e.failure_class is FailureClass.RUN_BOUND
    assert e.code == "window_ceiling_reached"
    assert "refused" in str(e)
    assert CallFailed(FailureClass.AUTH, "x").code is None


@pytest.mark.parametrize("bad", ["run 1 (a)", "", "x" * 201, "é", 7, b"r"])
def test_run_id_outside_the_record_alphabet_is_refused_at_construction(bad):
    with pytest.raises(ValueError, match="run_id"):
        Request("cs", (), run_id=bad)
    with pytest.raises(ValueError, match="run_id"):
        Request.from_dict({"call_site": "cs", "messages": [], "run_id": bad})


@pytest.mark.parametrize("good", ["eval-1", "run:7/a.b@c+d_e", "x" * 200])
def test_run_id_inside_the_record_alphabet_is_accepted(good):
    assert Request("cs", (), run_id=good).run_id == good
