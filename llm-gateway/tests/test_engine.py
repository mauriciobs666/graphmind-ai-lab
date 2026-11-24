"""Engine: chain walk, retry/fallback, records, policy and transport seams
(plan §3.2, §3.3, §3.7, §3.9; FR-1, FR-4, FR-5, FR-7, FR-11, FR-18, FR-19; AC-3, AC-4, AC-8, AC-9,
AC-11, AC-12, AC-22, AC-23, AC-24)."""

import json
import sqlite3

import pytest
from gwstandin import OK_BODY as GOOD_BODY
from gwstandin import FakeProvider, StandIn, error, refusal, write_config

from llmgateway.engine import AttemptsFailed, Gateway, PreparedStep
from llmgateway.failures import Condition
from llmgateway.policy import PolicyRefused
from llmgateway.providers import ProviderRegistry
from llmgateway.providers.base import encode_body
from llmgateway.records import RecordStore
from llmgateway.transport import DirectTransport, TransportFailure, WireResponse
from llmgateway.types import (
    CallFailed,
    CallOptions,
    EmbeddingsNotAvailable,
    HostedFromCallerProcess,
    Request,
)
from llmgateway.types import (
    FailureClass as F,
)

MSGS = ({"role": "user", "content": "hello"},)
SCHEMA = {"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object"}}}


def config_text(a, b):
    return f"""
[deployment]
local_endpoints = ["{a}", "{b}"]

[providers.lm1]
kind = "openai_compat"
base_url = "{a}"
model = "m1"

[providers.lm2]
kind = "openai_compat"
base_url = "{b}"
model = "m2"

[providers.claude]
kind = "anthropic"
base_url = "https://api.anthropic.com"
model = "claude-pinned-1"

[scopes.plain]
[scopes.open]
record_text = true
[scopes.secret]
marker = "restricted"

[call_sites.plain]
component = "answer.plain"
chain = ["lm1"]

[call_sites.retry]
component = "answer.retry"
chain = ["lm1"]
retry = {{ max = 1, on = ["connection", "timeout", "outage"] }}

[call_sites.retryperm]
component = "answer.retryperm"
chain = ["lm1"]
retry = {{ max = 2, on = ["outage", "auth", "rejected", "policy_refused", "config"] }}

[call_sites.fb]
component = "answer.fb"
chain = ["lm1", "lm2"]

[call_sites.retryfb]
component = "answer.retryfb"
chain = ["lm1", "lm2"]
retry = {{ max = 1, on = ["outage"] }}

[call_sites.optional]
component = "answer.optional"
chain = ["lm1"]
optional_settings = ["top_k"]

[call_sites.profile]
component = "answer.profile"
chain = ["lm1"]
settings = {{ seed = 7, temperature = 0.0 }}

[call_sites.texty]
component = "answer.texty"
scopes = ["open"]
chain = ["lm1"]

[call_sites.hosted]
component = "answer.hosted"
scopes = ["plain"]
chain = ["claude", "lm1"]

[call_sites.hostedsecret]
component = "answer.hostedsecret"
scopes = ["secret"]
chain = ["claude", "lm1"]

[call_sites.bounded]
component = "answer.bounded"
chain = ["lm1"]

[[run_bounds]]
match = "eval-*"
max_tokens = 5
"""


class Env:
    def __init__(self, tmp_path, replies_a=(), replies_b=()):
        self.a, self.b = StandIn(*(replies_a or ())), StandIn(*(replies_b or ()))
        self.src = write_config(tmp_path / "root", config_text(self.a.url, self.b.url))
        self.records = RecordStore(tmp_path / "state")
        self.registry = ProviderRegistry({})
        for kind in ("openai_compat", "anthropic"):
            self.registry.register(kind, FakeProvider(kind))

    def gateway(self, transport=None, **kw):
        t = transport or DirectTransport(self.src.get)
        return Gateway(self.src, t, self.records, registry=self.registry, **kw)

    def close(self):
        self.a.close()
        self.b.close()


@pytest.fixture
def make(tmp_path):
    envs = []

    def _make(replies_a=(), replies_b=()):
        e = Env(tmp_path, replies_a, replies_b)
        envs.append(e)
        return e

    yield _make
    for e in envs:
        e.close()


def ok():
    return (200, "application/json", GOOD_BODY)


def req(site="plain", **kw):
    return Request(site, MSGS, **kw)


def rows(env, call_id=None):
    return env.records.attempts(call_id)


# --- success, records ---


def test_success_returns_reported_identity_and_one_record(make):
    env = make()
    r = env.gateway().call(req(run_id="r1"))
    assert (r.provider, r.model, r.hosted) == ("lm1", "reported-model", False)
    assert r.message["content"] == "hi" and (r.usage.input_tokens, r.usage.output_tokens) == (3, 2)
    (row,) = rows(env)
    assert row["call_id"] == r.call_id and row["attempt"] == 1 and row["kind"] == "first"
    assert (row["outcome"], row["model"], row["component"], row["run_id"]) == (
        "ok",
        "reported-model",
        "answer.plain",
        "r1",
    )
    assert (row["input_tokens"], row["output_tokens"], row["hosted"]) == (3, 2, False)
    assert row["duration_ms"] is not None


def test_no_model_or_provider_field_selects_the_model_from_configuration(make):
    env = make()
    env.gateway().call(req("fb"))
    assert json.loads(env.a.received[0]["body"])["model"] == "m1"


def test_embed_is_refused_with_a_clear_error(make):
    env = make()
    with pytest.raises(EmbeddingsNotAvailable, match="embeddings are not available"):
        env.gateway().embed(["x"])
    assert env.a.received == [] and rows(env) == []


# --- FR-18 / AC-11: nothing added, nothing changed ---


def test_seed_and_schema_reach_the_provider_byte_identical(make):
    env = make()
    env.gateway().call(req(settings={"seed": 12345, "temperature": 0.0}, response_format=SCHEMA))
    got = env.a.received[0]["body"]
    expected = encode_body(
        {
            "model": "m1",
            "messages": list(MSGS),
            "seed": 12345,
            "temperature": 0.0,
            "response_format": SCHEMA,
        }
    )
    assert got == expected
    data = json.loads(got)
    assert set(data) == {"model", "messages", "seed", "temperature", "response_format"}
    assert data["seed"] == 12345 and data["response_format"] == SCHEMA


def test_profile_settings_are_sent_and_request_overrides_per_name(make):
    env = make()
    env.gateway().call(req("profile", settings={"seed": 9}))
    data = json.loads(env.a.received[0]["body"])
    assert (
        data["seed"] == 9
        and data["temperature"] == 0.0
        and set(data) == {"model", "messages", "seed", "temperature"}
    )


def test_required_setting_the_provider_cannot_honour_fails_naming_it_before_any_send(make):
    env = make()
    with pytest.raises(AttemptsFailed, match="top_k") as e:
        env.gateway().call(req(settings={"top_k": 5}))
    assert e.value.failure_class is F.CONFIG and env.a.received == []
    assert rows(env)[0]["failure_class"] == "config"


def test_optional_unsupported_setting_is_dropped_not_substituted(make):
    env = make()
    env.gateway().call(req("optional", settings={"top_k": 5, "seed": 1}))
    assert set(json.loads(env.a.received[0]["body"])) == {"model", "messages", "seed"}


# --- retry (FR-19, AC-22) ---


def test_retry_once_on_a_listed_transient_failure_recorded_as_an_attempt(make):
    env = make([error(503), ok()])
    r = env.gateway().call(req("retry"))
    assert len(env.a.received) == 2
    first, second = rows(env, r.call_id)
    assert (first["attempt"], first["outcome"], first["failure_class"]) == (1, "failed", "outage")
    assert (second["attempt"], second["kind"], second["reason"], second["outcome"]) == (
        2,
        "retry",
        "outage",
        "ok",
    )


def test_retry_stops_at_max(make):
    env = make([error(503)])
    with pytest.raises(AttemptsFailed) as e:
        env.gateway().call(req("retry"))
    assert len(env.a.received) == 2 and e.value.classes == (F.OUTAGE, F.OUTAGE)


@pytest.mark.parametrize(
    "reply,cls",
    [
        (error(401), "auth"),
        (error(403), "permission"),
        (error(400), "rejected"),
        (error(404), "rejected"),
        (refusal("endpoint_unknown"), "policy_refused"),
        (refusal("run_bound_reached"), "run_bound"),
        (refusal("bad_credential"), "config"),
    ],
)
def test_permanent_failure_is_never_retried_even_if_listed(make, reply, cls):
    env = make([reply, ok()])
    with pytest.raises(AttemptsFailed) as e:
        env.gateway().call(req("retry"))
    assert len(env.a.received) == 1 and e.value.failure_class.value == cls


@pytest.mark.parametrize("reply", [error(401), error(400), refusal("endpoint_unknown")])
def test_a_permanent_class_listed_in_on_is_still_never_retried(make, reply):
    env = make([reply, ok()])
    with pytest.raises(AttemptsFailed):
        env.gateway().call(req("retryperm"))
    assert len(env.a.received) == 1


def test_listed_class_only_a_transient_class_not_in_on_is_not_retried(make):
    env = make([error(429), ok()])  # rate_limit is temporary but not in `on`
    with pytest.raises(AttemptsFailed):
        env.gateway().call(req("retry"))
    assert len(env.a.received) == 1


def test_no_retry_configured_means_no_retry(make):
    env = make([error(503), ok()])
    with pytest.raises(AttemptsFailed):
        env.gateway().call(req("plain"))
    assert len(env.a.received) == 1


def test_retry_switched_off_by_the_caller_gives_the_first_failure_unmasked(make):
    env = make([error(503), ok()])
    with pytest.raises(AttemptsFailed) as e:
        env.gateway().call(req("retry", options=CallOptions(retry=False)))
    assert len(env.a.received) == 1 and e.value.classes == (F.OUTAGE,)


def test_retry_does_not_leak_to_the_next_step_for_free(make):
    env = make([error(503)], [ok()])
    r = env.gateway().call(req("retryfb"))  # retry lm1 once (still 503), then fall back
    assert len(env.a.received) == 2 and len(env.b.received) == 1
    kinds = [x["kind"] for x in rows(env, r.call_id)]
    assert kinds == ["first", "retry", "fallback"]


# --- fallback (FR-19, AC-3, AC-12, AC-24) ---


@pytest.mark.parametrize(
    "reply,cls",
    [
        (error(429), "rate_limit"),
        (error(402), "out_of_credit"),
        (error(504), "timeout"),
        (error(503), "outage"),
        (error(500, "No models loaded. Please load a model."), "model_not_loaded"),
    ],
)
def test_temporary_failure_falls_back_and_both_attempts_are_recorded(make, reply, cls):
    env = make([reply], [ok()])
    r = env.gateway().call(req("fb"))
    assert r.provider == "lm2" and r.model == "reported-model"
    one, two = rows(env, r.call_id)
    assert (one["provider"], one["outcome"], one["failure_class"], one["kind"]) == (
        "lm1",
        "failed",
        cls,
        "first",
    )
    assert (two["provider"], two["kind"], two["reason"], two["attempt"]) == (
        "lm2",
        "fallback",
        cls,
        2,
    )
    assert one["call_id"] == two["call_id"]


def test_connection_failure_falls_back(make):
    env = make(None, [ok()])
    env.a.close()  # lm1 refuses connections
    r = env.gateway().call(req("fb"))
    assert r.provider == "lm2"
    assert rows(env, r.call_id)[0]["failure_class"] == "connection"


@pytest.mark.parametrize(
    "reply,cls",
    [
        (error(401), "auth"),
        (error(403), "permission"),
        (error(400), "rejected"),
        (error(413), "rejected"),
        (refusal("endpoint_unknown"), "policy_refused"),
        (refusal("model_not_pinned"), "rejected"),
        (refusal("window_ceiling_reached"), "run_bound"),
        (refusal("key_missing"), "config"),
    ],
)
def test_permanent_failure_never_falls_back_and_names_the_code(make, reply, cls):
    env = make([reply], [ok()])
    with pytest.raises(AttemptsFailed) as e:
        env.gateway().call(req("fb"))
    assert env.b.received == [] and e.value.failure_class.value == cls
    assert len(rows(env)) == 1
    if reply[1].startswith("application/vnd"):
        assert e.value.code and e.value.code in str(e.value)


def test_fallback_switched_off_gives_the_first_failure_and_no_other_model_answers(make):
    env = make([error(429)], [ok()])
    with pytest.raises(AttemptsFailed) as e:
        env.gateway().call(req("fb", options=CallOptions(fallback=False)))
    assert e.value.failure_class is F.RATE_LIMIT and env.b.received == []
    assert len(rows(env)) == 1


def test_both_switched_off_with_retry_and_fallback_configured_is_unmasked(make):
    env = make([error(503), ok()], [ok()])
    with pytest.raises(AttemptsFailed) as e:
        env.gateway().call(req("retryfb", options=CallOptions(retry=False, fallback=False)))
    assert len(env.a.received) == 1 and env.b.received == [] and e.value.classes == (F.OUTAGE,)


def test_single_step_chain_without_retry_has_neither(make):
    env = make([error(429), ok()], [ok()])
    with pytest.raises(AttemptsFailed):
        env.gateway().call(req("plain"))
    assert len(env.a.received) == 1 and env.b.received == []


def test_every_step_failing_raises_with_all_classes(make):
    env = make([error(429)], [error(503)])
    with pytest.raises(AttemptsFailed) as e:
        env.gateway().call(req("fb"))
    assert e.value.classes == (F.RATE_LIMIT, F.OUTAGE) and e.value.failure_class is F.OUTAGE
    assert [r["attempt"] for r in rows(env, e.value.call_id)] == [1, 2]


# --- hosted steps, policy, FR-11 ---


class HostedTransport:
    """reaches_hosted = True: hosted steps answer from a script, local steps go direct."""

    reaches_hosted = True

    def __init__(self, direct, hosted_replies):
        self.direct, self.script, self.hosted_calls = direct, list(hosted_replies), []

    def exchange(self, request):
        if not request.hosted:
            return self.direct.exchange(request)
        self.hosted_calls.append(request)
        reply = self.script[min(len(self.hosted_calls), len(self.script)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


def hosted_ok():
    return WireResponse(200, GOOD_BODY, "application/json")


def test_hosted_attempt_falls_back_to_local_with_two_records_one_call_id(make):
    env = make(None, None)
    t = HostedTransport(
        DirectTransport(env.src.get), [WireResponse(429, b"{}", "application/json")]
    )
    r = env.gateway(t).call(req("hosted"))
    one, two = rows(env, r.call_id)
    assert (one["provider"], one["hosted"], one["failure_class"]) == ("claude", True, "rate_limit")
    assert (two["provider"], two["hosted"], two["kind"]) == ("lm1", False, "fallback")
    assert r.hosted is False and len(t.hosted_calls) == 1
    assert t.hosted_calls[0].run_id is None and t.hosted_calls[0].component == "answer.hosted"


def test_hosted_chain_through_a_transport_that_cannot_reach_hosted_sends_nothing(make):
    env = make()
    with pytest.raises(HostedFromCallerProcess):
        env.gateway().call(req("hosted"))  # DirectTransport
    assert env.a.received == [] and rows(env) == []


def test_engine_itself_refuses_a_hosted_chain_on_a_transport_that_cannot_reach_hosted(make):
    env = make()
    t = ScriptedTransport(WireResponse(200, GOOD_BODY, "application/json"))  # would answer
    with pytest.raises(HostedFromCallerProcess):
        env.gateway(t).call(req("hosted"))
    assert t.seen == [] and rows(env) == []


def test_policy_refusal_before_any_send_and_no_fallback_to_the_local_step(make):
    env = make()
    t = HostedTransport(DirectTransport(env.src.get), [hosted_ok()])
    with pytest.raises(PolicyRefused) as e:
        env.gateway(t).call(req("hostedsecret"))  # restricted scope, hosted default, no local_chain
    assert e.value.reason == "restricted_hosted"
    assert t.hosted_calls == [] and env.a.received == [] and rows(env) == []


def test_holder_policy_refusal_is_permanent_so_the_local_step_is_not_tried(make):
    env = make()
    t = HostedTransport(
        DirectTransport(env.src.get),
        [
            WireResponse(
                499,
                b'{"holderRefusal":"endpoint_unknown"}',
                "application/vnd.llmgateway.refusal+json",
            )
        ],
    )
    with pytest.raises(AttemptsFailed) as e:
        env.gateway(t).call(req("hosted"))
    assert e.value.failure_class is F.POLICY_REFUSED and env.a.received == []


# --- the Transport seam (D8/D10 plug in here) ---


class ScriptedTransport:
    reaches_hosted = False

    def __init__(self, *outcomes):
        self.outcomes, self.seen = list(outcomes), []

    def exchange(self, request):
        self.seen.append(request)
        out = self.outcomes[min(len(self.seen), len(self.outcomes)) - 1]
        if isinstance(out, Exception):
            raise out
        return out


@pytest.mark.parametrize(
    "exc,cls",
    [
        (TransportFailure(Condition.BROKER_UPSTREAM_ERROR), F.CONNECTION),
        (TransportFailure(Condition.BROKER_TIMEOUT), F.TIMEOUT),
        (TransportFailure(Condition.BROKER_DENIAL), F.POLICY_REFUSED),
        (TransportFailure(Condition.BROKER_UNAVAILABLE), F.CONFIG),
        (ConnectionResetError(), F.CONNECTION),
        (TimeoutError(), F.TIMEOUT),
        (CallFailed(F.POLICY_REFUSED, "denied", "x"), F.POLICY_REFUSED),
        (RuntimeError("bug"), F.CONFIG),
    ],
)
def test_any_transport_failure_is_classified_by_the_shared_tables(make, exc, cls):
    env = make()
    with pytest.raises(AttemptsFailed) as e:
        env.gateway(ScriptedTransport(exc)).call(req())
    assert e.value.failure_class is cls


def test_other_transports_receive_the_full_wire_request(make):
    env = make()
    t = ScriptedTransport(WireResponse(200, GOOD_BODY, "application/json"))
    env.gateway(t).call(req("texty", scopes=("plain",), run_id="r9"))
    (w,) = t.seen
    assert (w.provider_id, w.kind, w.base_url, w.path, w.hosted) == (
        "lm1",
        "openai_compat",
        env.a.url,
        "/v1/chat/completions",
        False,
    )
    assert (w.run_id, w.component, w.scopes) == ("r9", "answer.texty", ("open", "plain"))


def test_unparseable_success_reply_is_a_config_failure_not_retried(make):
    env = make([(200, "application/json", b"<html>not it</html>"), ok()])
    with pytest.raises(AttemptsFailed) as e:
        env.gateway().call(req("retry"))
    assert e.value.failure_class is F.CONFIG and len(env.a.received) == 1


# --- records: content rules (FR-5, AC-17) ---


def test_error_body_text_never_reaches_the_record_store(make):
    env = make([error(401, "SECRET-PROVIDER-TEXT sk-ant-FAKE")])
    with pytest.raises(AttemptsFailed) as e:
        env.gateway().call(req())
    assert "SECRET" not in str(e.value) and "sk-ant" not in str(e.value)
    with sqlite3.connect(env.records.path) as db:
        dump = "\n".join(db.iterdump())
    assert "SECRET" not in dump and "sk-ant" not in dump


def test_text_recorded_only_when_every_scope_opts_in(make):
    env = make()
    env.gateway().call(req("texty"))
    env.gateway().call(req("plain"))
    texty, plain = rows(env)
    assert "hello" in texty["request_text"] and "hi" in texty["response_text"]
    assert plain["request_text"] is None and plain["response_text"] is None


# --- run bound (§3.9) ---


def test_library_run_bound_refuses_before_sending_once_reached(make):
    env = make()
    g = env.gateway()
    g.call(req("bounded", run_id="eval-1"))  # 3 + 2 = 5 tokens: the bound
    with pytest.raises(PolicyRefused) as e:
        g.call(req("bounded", run_id="eval-1"))
    assert e.value.reason == "run_bound_reached" and len(env.a.received) == 1
    g.call(req("bounded", run_id="other"))  # unmatched run ids are unbounded here


# --- residency hook (D6 wires it) ---


class Hook:
    def __init__(self, model="hooked", state="loaded:x", fail=None):
        self.model, self.state, self.fail, self.ctxs = model, state, fail, []

    def prepare(self, ctx):
        self.ctxs.append(ctx)
        if self.fail:
            raise self.fail
        return PreparedStep(self.model, self.state)


def test_residency_hook_chooses_the_model_and_the_state_is_recorded(make):
    env = make()
    hook = Hook()
    env.gateway(residency=hook).call(req())
    assert json.loads(env.a.received[0]["body"])["model"] == "hooked"
    assert rows(env)[0]["loaded_state"] == "loaded:x" and hook.ctxs[0].step.provider == "lm1"


def test_residency_hook_failure_is_the_attempts_failure_and_can_fall_back(make):
    env = make(None, [ok()])
    hook = Hook(fail=CallFailed(F.MODEL_NOT_LOADED, "not loaded"))
    calls = []
    orig = hook.prepare

    def prepare(ctx):
        calls.append(ctx.step.provider)
        return PreparedStep("m2") if ctx.step.provider == "lm2" else orig(ctx)

    hook.prepare = prepare
    r = env.gateway(residency=hook).call(req("fb"))
    assert calls == ["lm1", "lm2"] and env.a.received == [] and r.provider == "lm2"


def test_missing_model_is_a_config_failure(make):
    env = make()
    with pytest.raises(AttemptsFailed, match="no model"):
        env.gateway(residency=Hook(model=None)).call(req())
    assert env.a.received == []
