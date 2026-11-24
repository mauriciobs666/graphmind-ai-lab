"""Load and residency (plan §3.8, S10; FR-16, FR-17; AC-13), executed against an in-process
stand-in that models residency (a loaded list that a load request extends) and logs every request
it receives. Assertions are on what the stand-in received: strict sends no load, on-demand sends
exactly one, a second copy is never requested, unload endpoints are never called."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from gwstandin import write_config
from provider_env import ForwardTransport, provider_table

from llmgateway.engine import AttemptsFailed, Gateway
from llmgateway.load import (
    LoadedModel,
    LoadResidency,
    Residency,
    bind_loaded,
    state_label,
)
from llmgateway.providers import REGISTRY
from llmgateway.records import RecordStore
from llmgateway.transport import DirectTransport
from llmgateway.types import CallFailed, LoadMode, Request
from llmgateway.types import FailureClass as F

MSGS = ({"role": "user", "content": "hello"},)
LM_REC = {
    "type": "llm",
    "publisher": "acme",
    "arch": "qwen3",
    "quantization": "Q4_K_M",
    "state": "loaded",
}
EMBED_REC = {"type": "embeddings", "publisher": "nomic", "arch": "bert", "state": "loaded"}


class Lab:
    """One stand-in for LM Studio (`kind="lmstudio"`) or Ollama. `models` maps id -> (record,
    context); a load request adds an entry. Every request is logged as (method, path, body)."""

    def __init__(self, kind, models=()):
        self.kind = kind
        self.models = dict(models)
        self.log: list[tuple[str, str, dict | None]] = []
        self.load_status = 200
        self.loaded_body: bytes | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self):
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if n else b""
                body = json.loads(raw) if raw else None
                outer.log.append((self.command, self.path, body))
                status, data = outer.route(self.command, self.path, body)
                out = data if isinstance(data, bytes) else json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            do_GET = do_POST = do_DELETE = do_PUT = _serve

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()

    def route(self, method, path, body):
        if self.kind == "lmstudio":
            if (method, path) == ("GET", "/api/v0/models"):
                if self.loaded_body is not None:
                    return 200, self.loaded_body
                data = [
                    {"id": mid, **rec, "loaded_context_length": ctx}
                    for mid, (rec, ctx) in self.models.items()
                ]
                return 200, {"data": data}
            if (method, path) == ("POST", "/api/v1/models/load"):
                if self.load_status != 200:
                    return self.load_status, {"error": "boom"}
                self.models[body["model"]] = (LM_REC, body.get("context_length", 4096))
                return 200, {"status": "loaded"}
            if (method, path) == ("POST", "/v1/chat/completions"):
                return 200, {
                    "model": body["model"],
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
        else:
            if (method, path) == ("GET", "/api/ps"):
                data = [
                    {"model": mid, "context_length": ctx, **rec}
                    for mid, (rec, ctx) in self.models.items()
                ]
                return 200, {"models": data}
            if (method, path) == ("POST", "/api/chat"):
                if body["messages"] == []:
                    if self.load_status != 200:
                        return self.load_status, {"error": "boom"}
                    ctx = (body.get("options") or {}).get("num_ctx", 4096)
                    self.models[body["model"]] = ({}, ctx)
                    return 200, {"model": body["model"], "done": True}
                return 200, {
                    "model": body["model"],
                    "message": {"role": "assistant", "content": "hi"},
                    "prompt_eval_count": 1,
                    "eval_count": 1,
                }
        return 404, {"error": "unknown"}

    @property
    def requests(self):
        return [f"{m} {p}" for m, p, _ in self.log]

    def loads(self):
        return [b for m, p, b in self.log if (m, p) in LOAD_ENDPOINTS or _ollama_load(m, p, b)]

    def chats(self):
        return [
            b
            for m, p, b in self.log
            if p in ("/v1/chat/completions", "/api/chat") and b and b.get("messages")
        ]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


LOAD_ENDPOINTS = {("POST", "/api/v1/models/load")}


def _ollama_load(method, path, body):
    return (method, path) == ("POST", "/api/chat") and body is not None and body["messages"] == []


class Rig:
    def __init__(self, tmp_path, lab, *, site="", provider="", model="m1", residency=True):
        self.lab = lab
        cfg = f"""
[deployment]
local_endpoints = ["{lab.url}"]

[providers.p]
kind = "{lab.kind}"
base_url = "{lab.url}"
model = "{model}"
{provider}
[scopes.plain]

[call_sites.cs]
component = "answer.cs"
scopes = ["plain"]
chain = ["p"]
{site}
"""
        self.src = write_config(tmp_path / "root", cfg)
        self.records = RecordStore(tmp_path / "state")
        self.transport = DirectTransport(self.src.get)
        hook = LoadResidency(self.transport) if residency else None
        self.gateway = Gateway(self.src, self.transport, self.records, residency=hook)

    def call(self, **kw):
        return self.gateway.call(Request("cs", MSGS, **kw))

    def fail(self, **kw) -> AttemptsFailed:
        with pytest.raises(AttemptsFailed) as exc:
            self.call(**kw)
        return exc.value

    def rows(self):
        return self.records.attempts()


@pytest.fixture
def rig(tmp_path):
    made = []

    def _make(kind="lmstudio", models=(), **kw):
        lab = Lab(kind, models)
        r = Rig(tmp_path / f"r{len(made)}", lab, **kw)
        made.append(lab)
        return r

    yield _make
    for lab in made:
        lab.close()
    for lab in made:  # no test ever reaches an unload endpoint or a mutating verb but POST/GET
        assert not any("unload" in p or m in ("DELETE", "PUT") for m, p, _ in lab.log), lab.requests


def lm(model="m1", ctx=4096, rec=LM_REC):
    return {model: (rec, ctx)}


# --- strict ---


def test_strict_absent_fails_model_not_loaded_and_sends_no_load(rig):
    r = rig(site='load_mode = "strict"')
    exc = r.fail()
    assert exc.failure_class is F.MODEL_NOT_LOADED and exc.classes == (F.MODEL_NOT_LOADED,)
    assert r.lab.requests == ["GET /api/v0/models"]  # the read only: no load, no chat
    (row,) = r.rows()
    assert row["failure_class"] == "model_not_loaded" and row["loaded_state"] == "not_loaded:m1"


def test_strict_present_calls_and_records_the_loaded_model_without_loading(rig):
    r = rig(models=lm(), site='load_mode = "strict"')
    resp = r.call()
    assert resp.model == "m1"
    assert r.lab.requests == ["GET /api/v0/models", "POST /v1/chat/completions"]
    assert r.lab.loads() == []
    assert r.rows()[0]["loaded_state"] == "loaded:m1"


def test_request_load_mode_overrides_the_call_site(rig):
    r = rig(site='load_mode = "on_demand"')
    assert r.fail(load_mode=LoadMode.STRICT).failure_class is F.MODEL_NOT_LOADED
    assert r.lab.loads() == []


# --- on demand ---


def test_on_demand_absent_loads_exactly_once_reads_first_and_records_it(rig):
    r = rig()  # on_demand is the default
    r.call()
    assert r.lab.requests == [
        "GET /api/v0/models",
        "POST /api/v1/models/load",
        "POST /v1/chat/completions",
    ]
    assert len(r.lab.loads()) == 1 and r.lab.loads()[0] == {"model": "m1"}
    assert r.rows()[0]["loaded_state"] == "loaded_on_demand:m1"


def test_on_demand_never_creates_a_second_copy(rig):
    r = rig()
    r.call()
    r.call()
    r.call()
    assert len(r.lab.loads()) == 1 and len(r.lab.models) == 1
    assert [row["loaded_state"] for row in r.rows()] == [
        "loaded_on_demand:m1",
        "loaded:m1",
        "loaded:m1",
    ]


def test_on_demand_present_sends_no_load(rig):
    r = rig(models=lm())
    r.call()
    assert r.lab.loads() == []


def test_load_settings_reach_the_load_request(rig):
    r = rig(site="load = {context_length = 8192, flash_attention = true}")
    r.call()
    assert r.lab.loads() == [{"model": "m1", "context_length": 8192, "flash_attention": True}]


def test_load_failure_is_classified_and_nothing_is_chatted(rig):
    r = rig()
    r.lab.load_status = 500
    exc = r.fail()
    assert exc.failure_class is F.OUTAGE and r.lab.chats() == []


def test_unreadable_loaded_reply_fails_config_and_loads_nothing(rig):
    r = rig()
    r.lab.loaded_body = b'{"nope": 1}'
    assert r.fail().failure_class is F.CONFIG
    assert r.lab.requests == ["GET /api/v0/models"]


# --- mismatch and acceptance ---


@pytest.mark.parametrize(
    "site",
    [
        "load = {context_length = 8192}",
        'load_mode = "strict"\nload = {context_length = 8192}',
        'load_mode = "explicit"\nload = {context_length = 8192}',
    ],
)
def test_configured_context_differing_from_loaded_fails_naming_it_in_every_mode(rig, site):
    r = rig(models=lm(ctx=4096), site=site)
    exc = r.fail()
    assert exc.failure_class is F.CONFIG
    assert "context_length 4096" in str(exc) and "configured 8192" in str(exc)
    assert r.lab.requests == ["GET /api/v0/models"]  # never reloaded, never chatted
    assert r.rows()[0]["loaded_state"] == "mismatch:m1"


def test_mismatch_when_context_is_unreported_fails_closed(rig):
    r = rig(models=lm(ctx=None), site="load = {context_length = 8192}")
    assert "unreported" in str(r.fail())
    assert r.lab.loads() == []


def test_configured_context_equal_to_loaded_reuses_it(rig):
    r = rig(models=lm(ctx=8192), site="load = {context_length = 8192}")
    r.call()
    assert r.lab.loads() == [] and r.rows()[0]["loaded_state"] == "loaded:m1"


def test_no_configured_settings_accepts_whatever_is_loaded(rig):
    r = rig(models=lm(ctx=777), site='load_mode = "strict"')
    r.call()
    assert r.rows()[0]["loaded_state"] == "loaded:m1"


def test_provider_table_load_settings_apply_when_the_site_has_none(rig):
    r = rig(models=lm(ctx=4096), provider="load = {context_length = 8192}")
    assert r.fail().failure_class is F.CONFIG


# --- explicit ---


def test_explicit_call_without_a_load_never_loads(rig):
    r = rig(site='load_mode = "explicit"')
    assert r.fail().failure_class is F.MODEL_NOT_LOADED
    assert r.lab.loads() == []


def test_explicit_gateway_load_loads_once_is_recorded_and_then_calls_run(rig):
    r = rig(site='load_mode = "explicit"\nload = {context_length = 8192}')
    got = r.gateway.load("cs", run_id="r1")
    assert got.model == "m1" and got.loaded_state == "loaded_explicit:m1"
    assert r.lab.loads() == [{"model": "m1", "context_length": 8192}]
    assert r.lab.chats() == []
    (row,) = r.rows()
    assert (row["outcome"], row["loaded_state"], row["run_id"]) == (
        "ok",
        "loaded_explicit:m1",
        "r1",
    )
    r.gateway.load("cs")  # resident now: reused, not loaded again
    r.call()
    assert len(r.lab.loads()) == 1
    assert [x["loaded_state"] for x in r.rows()] == [
        "loaded_explicit:m1",
        "loaded:m1",
        "loaded:m1",
    ]


def test_explicit_gateway_load_of_a_mismatched_model_fails_and_is_recorded(rig):
    r = rig(models=lm(ctx=4096), site="load = {context_length = 8192}")
    with pytest.raises(CallFailed, match="configured 8192") as exc:
        r.gateway.load("cs")
    assert exc.value.failure_class is F.CONFIG and r.lab.loads() == []
    (row,) = r.rows()
    assert (row["outcome"], row["loaded_state"]) == ("failed", "mismatch:m1")


def test_gateway_load_without_residency_fails_config_before_any_request(rig):
    r = rig(residency=False)
    with pytest.raises(CallFailed, match="no residency") as exc:
        r.gateway.load("cs")
    assert exc.value.failure_class is F.CONFIG and r.lab.log == []


# --- bind to the loaded chat model ---


def test_bind_loaded_skips_embedding_models_and_never_loads(rig):
    models = {**lm("embed-1", 512, EMBED_REC), **lm("chat-1", 4096)}
    r = rig(models=models, model="absent", site="bind_loaded = true")
    resp = r.call()
    assert resp.model == "chat-1"
    assert r.lab.chats()[0]["model"] == "chat-1"
    assert r.lab.loads() == []
    assert r.rows()[0]["loaded_state"] == "loaded:chat-1"


def test_bind_loaded_prefers_the_configured_model(rig):
    models = {**lm("chat-1"), **lm("chat-2")}
    r = rig(models=models, model="chat-2", site="bind_loaded = true")
    assert r.call().model == "chat-2"


def test_bind_loaded_with_only_an_embedding_model_fails_and_loads_nothing(rig):
    r = rig(models=lm("embed-1", 512, EMBED_REC), site="bind_loaded = true")
    exc = r.fail()
    assert exc.failure_class is F.MODEL_NOT_LOADED and r.lab.loads() == []


def test_bind_loaded_on_an_empty_endpoint_fails_and_loads_nothing(rig):
    r = rig(site="bind_loaded = true")
    assert r.fail().failure_class is F.MODEL_NOT_LOADED
    assert r.lab.requests == ["GET /api/v0/models"]


def test_bind_loaded_applies_the_mismatch_rule(rig):
    r = rig(models=lm("chat-1", 4096), site="bind_loaded = true\nload = {context_length = 9}")
    assert r.fail().failure_class is F.CONFIG


def test_bind_loaded_unit_picks_first_chat_when_no_preference_matches():
    a, b = LoadedModel("e", 1, EMBED_REC), LoadedModel("c", 2, LM_REC)
    assert bind_loaded([a, b], "zzz") is b
    with pytest.raises(CallFailed):
        bind_loaded([a])


# --- LoadedModel carries the raw record (S17: fingerprint, quantization) ---


def test_residency_loaded_returns_the_verbatim_provider_record(rig):
    r = rig(models=lm("chat-1", 4096))
    prov = REGISTRY.get("lmstudio")
    got = Residency(r.transport, prov, "p", r.src.get().providers["p"]).loaded()
    assert [(m.id, m.context_length) for m in got] == [("chat-1", 4096)]
    raw = got[0].raw
    assert (raw["publisher"], raw["arch"], raw["quantization"]) == ("acme", "qwen3", "Q4_K_M")
    assert raw["loaded_context_length"] == 4096


def test_loaded_wire_request_carries_component_scopes_and_run_id(rig):
    r = rig()
    seen = []
    inner = r.transport

    class Spy:
        reaches_hosted = False

        def exchange(self, wire):
            seen.append(wire)
            return inner.exchange(wire)

    gw = Gateway(r.src, Spy(), r.records, residency=LoadResidency(Spy()))
    gw.call(Request("cs", MSGS, run_id="run-1"))
    loaded_wire = seen[0]
    assert (loaded_wire.method, loaded_wire.path) == ("GET", "/api/v0/models")
    assert loaded_wire.component == "answer.cs" and loaded_wire.scopes == ("plain",)
    assert loaded_wire.run_id == "run-1" and loaded_wire.hosted is False


def test_state_label_fits_the_record_alphabet():
    s = state_label("loaded", "Org/Model-4B@Q4 " + "x" * 80)
    assert len(s) == 64 and s == s.lower() and "/" not in s and " " not in s


def test_model_ids_with_slashes_and_capitals_are_recorded(rig):
    r = rig(models=lm("Org/Model-4B"), model="Org/Model-4B")
    r.call()
    assert r.rows()[0]["loaded_state"] == "loaded:org_model-4b"


# --- the no-op default refuses what it cannot honour ---


@pytest.mark.parametrize(
    "site", ['load_mode = "strict"', "load = {context_length = 1}", "bind_loaded = true"]
)
def test_default_gateway_refuses_residency_config_it_cannot_honour(rig, site):
    r = rig(site=site, residency=False)
    assert r.fail().failure_class is F.CONFIG and r.lab.log == []


def test_default_gateway_refuses_a_request_load_mode(rig):
    r = rig(residency=False)
    assert r.fail(load_mode=LoadMode.STRICT).failure_class is F.CONFIG and r.lab.log == []


def test_default_gateway_without_residency_config_is_untouched(rig):
    r = rig(residency=False)
    r.call()
    assert r.lab.requests == ["POST /v1/chat/completions"]


def test_invalid_load_mode_and_load_table_fail_config(rig):
    assert rig(site='load_mode = "bogus"').fail().failure_class is F.CONFIG
    assert rig(site='load = "big"').fail().failure_class is F.CONFIG


# --- providers without residency pass through ---


def test_a_provider_kind_without_residency_is_untouched(tmp_path):
    from gwstandin import StandIn

    s = StandIn()
    cfg = f"""
[deployment]
local_endpoints = ["{s.url}"]
[providers.p]
kind = "openai_compat"
base_url = "{s.url}"
model = "m1"
[scopes.plain]
[call_sites.cs]
component = "answer.cs"
scopes = ["plain"]
chain = ["p"]
load_mode = "strict"
"""
    src = write_config(tmp_path / "root", cfg)
    t = DirectTransport(src.get)
    gw = Gateway(src, t, RecordStore(tmp_path / "st"), residency=LoadResidency(t))
    gw.call(Request("cs", MSGS))
    assert [x["path"] for x in s.received] == ["/v1/chat/completions"]
    s.close()


# --- Ollama ---


def ollama_ps(model="m1", ctx=4096):
    return {model: ({}, ctx)}


def test_ollama_on_demand_loads_with_empty_messages_and_num_ctx(rig):
    r = rig("ollama", site="load = {context_length = 8192}")
    r.call()
    assert r.lab.requests == ["GET /api/ps", "POST /api/chat", "POST /api/chat"]
    load = r.lab.log[1][2]
    assert load["messages"] == [] and load["options"] == {"num_ctx": 8192}
    assert r.rows()[0]["loaded_state"] == "loaded_on_demand:m1"


def test_ollama_never_sends_a_num_ctx_to_a_loaded_model(rig):
    """A loaded model with another context fails; the chat of a matching one carries no num_ctx."""
    r = rig("ollama", models=ollama_ps(ctx=4096), site="load = {context_length = 8192}")
    assert "configured 8192" in str(r.fail())
    assert r.lab.requests == ["GET /api/ps"]
    ok = rig("ollama", models=ollama_ps(ctx=8192), site="load = {context_length = 8192}")
    ok.call()
    assert ok.lab.loads() == []
    assert "options" not in ok.lab.chats()[0]


def test_ollama_strict_absent_sends_no_load(rig):
    r = rig("ollama", site='load_mode = "strict"')
    assert r.fail().failure_class is F.MODEL_NOT_LOADED
    assert r.lab.requests == ["GET /api/ps"]


def test_ollama_second_call_does_not_load_again(rig):
    r = rig("ollama")
    r.call()
    r.call()
    assert len(r.lab.loads()) == 1


@pytest.mark.parametrize("present", [False, True])
def test_ollama_parallel_in_load_settings_fails_naming_it_before_anything_is_sent(rig, present):
    r = rig(
        "ollama",
        models=ollama_ps() if present else (),
        site="load = {context_length = 4096, parallel = 2}",
    )
    exc = r.fail()
    assert exc.failure_class is F.CONFIG and "parallel" in str(exc)
    assert r.lab.log == []


def test_ollama_parallel_on_the_provider_fails_naming_it_and_loads_nothing(rig):
    r = rig("ollama", models=ollama_ps(), provider="parallel = 2")
    exc = r.fail()
    assert "parallel" in str(exc) and r.lab.chats() == [] and r.lab.loads() == []


def test_ollama_context_present_unreported_with_configured_fails_closed(rig):
    r = rig("ollama", models={"m1": ({}, None)}, site="load = {context_length = 4096}")
    assert "unreported" in str(r.fail())


def test_ollama_untagged_model_matches_its_latest_tag(rig):
    r = rig("ollama", models=ollama_ps("m1:latest"), site='load_mode = "strict"')
    r.call()
    assert r.lab.loads() == []


def test_hosted_ollama_load_settings_are_refused_before_any_request(tmp_path):
    from gwstandin import StandIn

    s = StandIn()
    cfg = f"""
[deployment]
local_endpoints = ["{s.url}"]
[providers.p]
{provider_table("ollama", s.url, hosted=True)}
[scopes.plain]
[call_sites.cs]
component = "answer.cs"
scopes = ["plain"]
chain = ["p"]
load = {{context_length = 4096}}
"""
    src = write_config(tmp_path / "root", cfg)
    t = ForwardTransport(s.url)
    gw = Gateway(src, t, RecordStore(tmp_path / "st"), residency=LoadResidency(t))
    with pytest.raises(AttemptsFailed, match="hosted") as exc:
        gw.call(Request("cs", MSGS))
    assert exc.value.failure_class is F.CONFIG and s.received == []
    s.close()


def test_ollama_strict_with_matching_context_calls_without_loading(rig):
    r = rig(
        "ollama",
        models=ollama_ps(ctx=4096),
        site='load_mode = "strict"\nload = {context_length = 4096}',
    )
    r.call()
    assert r.lab.requests == ["GET /api/ps", "POST /api/chat"]
