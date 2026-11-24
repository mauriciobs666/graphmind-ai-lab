"""The service end to end (docs/plans/llm-gateway.md §3.1, §3.3, §3.4, §5.2; FR-1, FR-8, FR-11,
FR-17, AC-5, AC-7): a real service process on a Unix socket, a real holder process, an HTTPS
stand-in provider and a local stand-in, driven through `client.py`."""

import json
import pathlib

import pytest
from holder_harness import CALLER_SECRET, MODEL, OTHER_SECRET, Env, StandIn, anthropic_ok, make_cert
from service_harness import MSGS, LocalStandIn, Stack

from llmgateway.client import CallFailed, EmbeddingsNotAvailable
from llmgateway.engine import Gateway
from llmgateway.records import RecordStore
from llmgateway.transport import DirectTransport
from llmgateway.types import HostedFromCallerProcess, Request


@pytest.fixture
def make_stack(tmp_path):
    made = []

    def _make(*, responder=None):
        cert, key = make_cert(tmp_path)
        env = Env(tmp_path / f"h{len(made)}", StandIn(cert, key, responder), cert)
        local = LocalStandIn()
        stack = Stack(env, local)
        made.append(stack)
        return stack

    yield _make
    for stack in made:
        stack.close()


@pytest.fixture
def stack(make_stack):
    s = make_stack()
    s.start_holder()
    s.start_service()
    return s


def attempts(stack, call_id):
    return RecordStore(stack.state_dir).attempts(call_id)


# --- the routes: local, hosted ---


def test_a_local_call_goes_out_from_the_service_and_is_recorded(stack):
    reply = stack.client().call("local", MSGS)
    assert (reply.provider, reply.model, reply.hosted) == ("lm", "lm-reported", False)
    assert reply.message["content"] == "local" and reply.usage.output_tokens == 2
    assert len(stack.local.chats) == 1 and stack.env.standin.seen == []
    (row,) = attempts(stack, reply.call_id)
    assert row["provider"] == "lm" and row["outcome"] == "ok" and not row["hosted"]
    assert stack.calls_log() == []  # a local call never reaches the holder


def test_a_hosted_call_round_trips_service_holder_upstream(stack):
    reply = stack.client().call("hosted", MSGS, run_id="eval-1")
    assert (reply.provider, reply.model, reply.hosted) == ("anth", MODEL, True)
    assert reply.message and reply.usage.input_tokens == 19 and reply.usage.output_tokens == 7
    (seen,) = stack.env.standin.seen
    assert seen.path == "/v1/messages" and json.loads(seen.body)["model"] == MODEL
    assert seen.headers["x-api-key"] == stack.env.marker  # the holder, not the service, added it
    (line,) = stack.calls_log()
    assert json.loads(line)["caller"] == "service" and json.loads(line)["run_id"] == "eval-1"
    (row,) = attempts(stack, reply.call_id)
    assert row["hosted"] and row["run_id"] == "eval-1" and row["provider"] == "anth"


def test_a_retry_signs_a_fresh_nonce_and_succeeds(make_stack):
    answers = iter(
        [(503, {"Content-Type": "application/json"}, b'{"error":{"type":"overloaded"}}')]
    )
    s = make_stack(responder=lambda seen: next(answers, None) or anthropic_ok())
    s.start_holder()
    s.start_service()
    reply = s.client().call("hosted", MSGS, run_id="eval-1")
    assert reply.hosted and len(s.env.standin.seen) == 2
    assert len(s.calls_log()) == 2
    assert [r["attempt"] for r in attempts(s, reply.call_id)] == [1, 2]


def test_the_key_and_the_credential_appear_in_no_service_output(stack):
    failing = stack.client()
    with pytest.raises(CallFailed):
        failing.call("hosted", MSGS)  # no run id: a refusal
    stack.client().call("hosted", MSGS, run_id="eval-1")
    out, err = stack.stop_service()
    for needle in (stack.env.marker, CALLER_SECRET.hex()):
        assert needle not in out + err


# --- holder refusals through the service ---


def refusal_case(stack, name, **kw):
    """One call that the holder refuses with `name`; returns (error, standin hits)."""
    with pytest.raises(CallFailed) as info:
        stack.client().call(kw.pop("site", "hosted_fb"), MSGS, **{"run_id": "eval-1", **kw})
    return info.value


@pytest.mark.parametrize(
    "code, cls, kw",
    [
        ("limit_over_ceiling", "rejected", {"site": "too_big"}),
        ("run_id_missing", "run_bound", {"run_id": None}),
        ("run_unmatched", "run_bound", {"run_id": "other-1"}),
    ],
)
def test_holder_refusals_map_to_their_class_and_are_never_retried(stack, code, cls, kw):
    err = refusal_case(stack, code, **kw)
    assert (err.failure_class, err.code) == (cls, code)
    assert err.classes == (cls,)  # one attempt: no retry, no fallback to the local step
    assert stack.env.standin.seen == [] and stack.local.chats == []
    (row,) = attempts(stack, err.call_id)
    assert row["failure_class"] == cls and row["outcome"] == "failed"


def test_an_unpinned_model_is_rejected_before_anything_is_sent(stack):
    """The adapter pins too (FR-9): this one never reaches the holder (`model_not_pinned` there
    is the same rule, unreachable from a configured call site)."""
    err = refusal_case(stack, "none", site="unpinned")
    assert err.failure_class == "rejected" and err.classes == ("rejected",)
    assert stack.calls_log() == [] and stack.env.standin.seen == []


def test_window_ceiling_key_missing_and_credential_refusals(make_stack):
    s = make_stack()
    s.configure(holder=s.env.holder_table(window_max_tokens=50))
    s.start_holder()
    s.start_service()
    err = refusal_case(s, "window_ceiling_reached")
    assert (err.failure_class, err.code, err.classes) == (
        "run_bound",
        "window_ceiling_reached",
        ("run_bound",),
    )
    assert s.env.standin.seen == []


def test_key_missing_is_config(stack):
    stack.env.key_path.unlink()
    err = refusal_case(stack, "key_missing")
    assert (err.failure_class, err.code, err.classes) == ("config", "key_missing", ("config",))


def test_a_wrong_credential_and_an_unknown_caller_are_config_never_retried(make_stack):
    s = make_stack()
    s.start_holder()
    s.write_credential(OTHER_SECRET)
    s.start_service()
    err = refusal_case(s, "bad_credential")
    assert (err.failure_class, err.code, err.classes) == ("config", "bad_credential", ("config",))
    (s.env.root / "state" / "registry" / "service.secret").unlink()
    err = refusal_case(s, "unknown_caller")
    assert (err.failure_class, err.code, err.classes) == ("config", "unknown_caller", ("config",))
    assert s.env.standin.seen == [] and s.local.chats == []


def test_only_a_stopped_holder_is_connection_and_it_is_retried(stack):
    stack.stop_holder()
    with pytest.raises(CallFailed) as info:
        stack.client().call("hosted", MSGS, run_id="eval-1")
    assert info.value.failure_class == "connection"
    assert info.value.classes == ("connection", "connection")  # 1 + max 1 retry
    # the fallback chain then reaches the local step: temporary classes may fall back
    assert stack.client().call("hosted_fb", MSGS, run_id="eval-1").hosted is False


def test_a_restricted_scope_is_refused_by_the_service_before_the_holder(stack):
    before = stack.calls_log()
    with pytest.raises(CallFailed) as info:
        stack.client().call("restricted", MSGS, run_id="eval-1")
    assert info.value.failure_class == "policy_refused"
    assert stack.calls_log() == before and stack.env.standin.seen == []
    # a request can only add scopes: adding a restricted one to a hosted site refuses it too
    with pytest.raises(CallFailed) as info:
        stack.client().call("hosted", MSGS, run_id="eval-1", scopes=["secret"])
    assert info.value.failure_class == "policy_refused"
    assert stack.calls_log() == before and stack.env.standin.seen == []


def test_hosted_without_a_credential_fails_config_and_local_still_works(make_stack):
    s = make_stack()
    s.start_holder()
    s.start_service(credential=str(s.env.root / "absent.cred"))
    with pytest.raises(CallFailed) as info:
        s.client().call("hosted", MSGS, run_id="eval-1")
    assert info.value.failure_class == "config" and "credential" in str(info.value)
    assert s.env.standin.seen == [] and s.calls_log() == []
    assert s.client().call("local", MSGS).hosted is False
    _, err = s.stop_service()
    assert CALLER_SECRET.hex() not in err


def test_hosted_from_the_callers_own_process_still_fails(stack, tmp_path):
    src = stack.env.source()
    gateway = Gateway(src, DirectTransport(src.get), RecordStore(tmp_path / "lib-state"))
    with pytest.raises(HostedFromCallerProcess):
        gateway.call(Request("hosted", tuple(MSGS), run_id="eval-1"))
    assert stack.env.standin.seen == []


# --- FR-17: provider-specific detail ---


def test_provider_detail_reaches_the_client(stack):
    reply = stack.client().call("local", MSGS)
    assert reply.provider_detail["stats"] == {"tokens_per_second": 42.5}
    assert reply.provider_detail["runtime"] == {"name": "llama.cpp"}


def test_a_provider_op_runs_on_the_first_local_step_that_offers_it(make_stack):
    s = make_stack()
    s.start_holder()
    s.start_service()
    result = s.client().provider_op("local", "lmstudio.models")
    assert result["status"] == 200 and result["body"]["data"][0]["id"] == "m1"
    assert s.local.received[0]["path"] == "/api/v0/models"
    with pytest.raises(CallFailed) as info:
        s.client().provider_op("local", "ollama.ps")
    assert info.value.failure_class == "policy_refused"


def test_load_op_reuses_a_resident_model_and_reports_the_state(make_stack):
    s = make_stack()
    s.start_holder()
    s.start_service()
    got = s.client().load("local_load")
    assert got["model"] == "m1" and got["loaded_state"].startswith("loaded:")
    assert [r["path"] for r in s.local.received] == ["/api/v0/models"]


# --- protocol edges ---


def test_embeddings_are_refused_with_a_clear_error(stack):
    with pytest.raises(EmbeddingsNotAvailable, match="embeddings"):
        stack.client().embed(["x"])


def test_unknown_request_fields_unknown_ops_and_versions_are_refused(stack):
    c = stack.client()
    for bad in (
        {"call_site": "local", "messages": MSGS, "model": "gpt-x"},
        {"call_site": "local", "messages": MSGS, "marker": "unrestricted"},
        {"call_site": "local", "messages": MSGS, "options": {"strict": True}},
    ):
        with pytest.raises(CallFailed) as info:
            c.send(bad)
        assert info.value.failure_class == "policy_refused"
    for payload in ({"op": "reboot"}, {"op": "call", "request": "x"}):
        with pytest.raises(CallFailed) as info:
            c._roundtrip(payload)
        assert info.value.failure_class == "policy_refused"
    with pytest.raises(CallFailed) as info:
        c.send({"call_site": "nope", "messages": MSGS})
    assert info.value.failure_class == "config"
    assert stack.local.chats == []


def test_the_socket_is_private_and_a_second_service_is_refused(stack):
    st = pathlib.Path(stack.socket).stat()
    assert st.st_mode & 0o777 == 0o600
    assert pathlib.Path(stack.socket).parent.stat().st_mode & 0o777 == 0o700
    import subprocess
    import sys

    from service_harness import LAUNCHER

    spec = {
        "config": str(stack.env.config_path),
        "uid": __import__("os").getuid(),
        "anchor": str(stack.env.root),
        "socket": stack.socket,
        "state_dir": stack.state_dir,
        "credential": str(stack.cred),
    }
    res = subprocess.run(
        [sys.executable, str(LAUNCHER), json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert res.returncode == 2 and "already listening" in res.stdout
    assert stack.client().call("local", MSGS)  # the first one is unharmed


def test_a_write_config_unprotected_by_ownership_disables_hosted(stack):
    stack.env.config_path.chmod(0o666)  # group/other-writable: no longer owner-protected
    with pytest.raises(CallFailed) as info:
        stack.client().call("hosted", MSGS, run_id="eval-1")
    assert info.value.failure_class in ("config", "policy_refused")
    assert stack.env.standin.seen == [] and stack.calls_log() == []


def test_unknown_envelope_keys_are_refused_for_every_op_and_bad_scopes_are_typed(stack):
    c = stack.client()
    cases = [
        {"op": "call", "request": {"call_site": "local", "messages": MSGS}, "extra": 1},
        {"op": "load", "call_site": "local", "extra": 1},
        {"op": "provider_op", "call_site": "local", "name": "lmstudio.models", "extra": 1},
        {"op": "embed", "extra": 1},
    ]
    for payload in cases:
        with pytest.raises(CallFailed) as info:
            c._roundtrip(payload)
        assert info.value.failure_class == "policy_refused", payload
    # the shared validation names the field, not a TypeError from deep in the engine
    for payload, text in (
        ({"op": "load", "call_site": "local", "scopes": [1]}, "scopes must be strings"),
        (
            {"op": "provider_op", "call_site": "local", "name": "x", "scopes": [[]]},
            "scopes must be",
        ),
        ({"op": "load", "call_site": "local", "run_id": 5}, "run_id must be a string"),
    ):
        with pytest.raises(CallFailed, match=text):
            c._roundtrip(payload)
    assert stack.local.chats == []
    with pytest.raises(CallFailed, match="unknown request fields: extra"):
        c._roundtrip({"op": "embed", "extra": 1})
    with pytest.raises(CallFailed, match="unsupported protocol version"):
        c._roundtrip({"op": "embed", "v": 2})
