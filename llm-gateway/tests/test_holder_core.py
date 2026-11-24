"""The holder core, in process: authentication, the refusal table, the FR-10 ledger as the holder
drives it, release by outcome, and key hygiene (plan §3.3, §3.4; §5.2 FR-8, FR-10)."""

import json
import os
import threading
import time

import pytest
from holder_harness import (
    CALLER_SECRET,
    MODEL,
    OTHER_SECRET,
    Clock,
    Env,
    StandIn,
    anthropic_ok,
    body_for,
    call,
    make_cert,
    tree_search,
)

from llmgateway import framing
from llmgateway.failures import HOLDER_REFUSALS, classify_response
from llmgateway.keyholder import HolderStartError, UpstreamResult, build_policy


@pytest.fixture(scope="session")
def cert(tmp_path_factory):
    return make_cert(tmp_path_factory.mktemp("tls"))


@pytest.fixture
def env(tmp_path, cert):
    standin = StandIn(*cert)
    e = Env(tmp_path / "root", standin, cert[0])
    yield e
    standin.close()


class FakeUpstream:
    """Scripted upstream: logs every send; `script` is a list of results or a callable."""

    def __init__(self, script=None, delay=0.0):
        self.script = script
        self.delay = delay
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, **kw):
        with self.lock:
            self.calls.append(kw)
            n = len(self.calls) - 1
        time.sleep(self.delay)
        if callable(self.script):
            return self.script(n, kw)
        if isinstance(self.script, list):
            return self.script[min(n, len(self.script) - 1)]
        return ok_result()


def ok_result(inp=100, out=50):
    return UpstreamResult(
        True, "completed", 200, "application/json", anthropic_ok(inp, out, 0, 0)[2]
    )


def err_result(status, body=b'{"type":"error","error":{"type":"x","message":"m"}}'):
    return UpstreamResult(True, "completed", status, "application/json", body)


def window_total(h):
    return h.ledger.window_total(h.ledger and 3600)


def refused_code(reply):
    return reply.refusal_code


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def holder(env, clock):
    up = FakeUpstream()
    h = env.holder(clock=clock, upstream=up)
    h.fake = up
    yield h
    h.close()


# --- a real exchange against the HTTPS stand-in ---


def test_real_https_exchange_sends_the_exact_body_with_the_key_and_settles_usage(env, clock):
    h = env.holder(clock=clock)
    body = body_for(100)
    reply = call(h, body=body)
    assert reply.authenticated and reply.status == 200 and reply.close_reason == "completed"
    seen = env.standin.seen
    assert len(seen) == 1
    assert seen[0].body == body and seen[0].path == "/v1/messages"
    assert seen[0].headers["x-api-key"] == env.marker
    assert seen[0].headers["anthropic-version"] == "2023-06-01"
    assert h.ledger.window_total(3600) == 11 + 7 + 3 + 5  # every usage field settled
    log = [json.loads(line) for line in open(f"{env.state_root}/calls.log")]
    assert log == [
        {
            "ts": int(clock.now),
            "caller": "broker",
            "run_id": "eval-1",
            "endpoint": env.endpoint,
            "status": 200,
            "tokens": 26,
        }
    ]
    h.close()


@pytest.mark.parametrize(
    "kind,header", [("openai_compat", "authorization"), ("ollama", "authorization")]
)
def test_bearer_kinds_send_authorization(env, clock, kind, header):
    path = {"openai_compat": "/v1/chat/completions", "ollama": "/api/chat"}[kind]
    env.write_config(providers={"anth": env.provider(kind=kind)})
    up = FakeUpstream(
        [
            UpstreamResult(
                True,
                "completed",
                200,
                "application/json",
                json.dumps(
                    {
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        "prompt_eval_count": 1,
                        "eval_count": 1,
                    }
                ).encode(),
            )
        ]
    )
    h = env.holder(clock=clock, upstream=up)
    doc = (
        {"model": MODEL, "messages": [], "max_tokens": 10}
        if kind == "openai_compat"
        else {"model": MODEL, "messages": [], "options": {"num_predict": 10}}
    )
    reply = call(h, body=json.dumps(doc).encode(), path=path)
    assert reply.status == 200
    assert up.calls[0]["headers"][header.title()] == "Bearer " + env.marker
    assert "x-api-key" not in up.calls[0]["headers"]
    h.close()


def test_address_is_validated_then_connected_to_with_the_hostname_for_sni(env, clock):
    env.write_config(
        providers={
            "anth": env.provider(base_url="https://api.example.test:8443", allow_private=False)
        }
    )
    up = FakeUpstream()
    h = env.holder(clock=clock, upstream=up, resolver=lambda host, port: ["8.8.8.8"])
    reply = call(h, endpoint="https://api.example.test:8443")
    assert reply.status == 200
    kw = up.calls[0]
    assert (kw["host"], kw["port"], kw["ip"]) == ("api.example.test", 8443, "8.8.8.8")
    h.close()


def test_a_name_resolving_to_a_private_address_is_refused_before_any_connection(env, clock):
    env.write_config(
        providers={"anth": env.provider(base_url="https://api.example.test", allow_private=False)}
    )
    up = FakeUpstream()
    for ips in (["10.0.0.1"], ["8.8.8.8", "169.254.169.254"], ["127.0.0.1"], []):
        h = env.holder(clock=clock, upstream=up, resolver=lambda host, port, ips=ips: ips)
        assert call(h, endpoint="https://api.example.test").refusal_code == "address_forbidden"
        h.close()
    assert up.calls == []


# --- authentication (§5.2 FR-8 (5)) ---


def test_valid_credential_is_served(holder):
    assert call(holder).status == 200


def test_unknown_caller_and_wrong_secret_are_refused_unsigned(holder):
    r = call(holder, caller="mallory")
    assert r.refusal_code == "unknown_caller" and not r.authenticated
    r = call(holder, secret=OTHER_SECRET)
    assert r.refusal_code == "bad_credential" and not r.authenticated
    for name in ("../etc/x", "BROKER", "", "a" * 40, "br/oker"):
        assert call(holder, caller=name).refusal_code == "unknown_caller"
    assert holder.fake.calls == []


def test_a_caller_name_cannot_walk_out_of_the_registry(holder, env):
    outside = env.root / "state" / "evil.secret"
    outside.write_text(OTHER_SECRET.hex() + "\n")
    outside.chmod(0o600)
    assert call(holder, caller="../evil", secret=OTHER_SECRET).refusal_code == "unknown_caller"
    assert (
        call(holder, caller="evil/../broker", secret=CALLER_SECRET).refusal_code == "unknown_caller"
    )
    assert holder.fake.calls == []


def test_each_tampered_field_fails_the_mac(holder):
    endpoint = holder.s.config_source.get().providers["anth"]["base_url"]
    frame, nonce = framing.build_request(
        CALLER_SECRET,
        caller="broker",
        method="POST",
        endpoint=endpoint,
        path="/v1/messages",
        run_id="eval-1",
        body=body_for(),
        ts=int(holder.s.clock()),
    )
    header, body = framing.decode_payload(frame[4:])
    tampers = [
        ("run_id", "eval-2"),
        ("run_id", "eval-1|"),
        ("path", "/v1/messages/"),
        ("method", "PUT"),
        ("endpoint", endpoint + "/"),
        ("ts", header["ts"] + 1),
        ("nonce", nonce + "x"),
    ]
    for key, value in tampers:
        forged = framing.encode_frame({**header, key: value}, body)
        reply = framing.parse_reply(
            holder.process(forged[4:])[4:], CALLER_SECRET, nonce, require_mac=False
        )
        assert reply.refusal_code == "bad_credential", key
    forged = framing.encode_frame(header, body + b" ")
    assert (
        framing.parse_reply(
            holder.process(forged[4:])[4:], CALLER_SECRET, nonce, require_mac=False
        ).refusal_code
        == "bad_credential"
    )
    assert holder.fake.calls == []


def test_delimiter_shift_in_run_id_cannot_move_a_field(holder):
    """A run_id carrying `|` (or a length-looking prefix) cannot be re-split into other fields."""
    endpoint = holder.s.config_source.get().providers["anth"]["base_url"]
    signed = framing.build_request(
        CALLER_SECRET,
        caller="broker",
        method="POST",
        endpoint=endpoint,
        path="/v1/messages",
        run_id="eval-1|/v1/messages",
        body=body_for(),
        ts=int(holder.s.clock()),
    )[0]
    header, body = framing.decode_payload(signed[4:])
    # move the tail of run_id into the path field
    forged = framing.encode_frame(
        {**header, "run_id": "eval-1", "path": "|/v1/messages/v1/messages"}, body
    )
    r = framing.parse_reply(
        holder.process(forged[4:])[4:], CALLER_SECRET, header["nonce"], require_mac=False
    )
    assert r.refusal_code == "bad_credential"
    # the crafted run_id is itself signed and served as one opaque id (here: unmatched, refused)
    assert call(holder, run_id="eval-1|/v1/messages").status == 200  # glob eval-* matches
    assert call(holder, run_id="other|x").refusal_code == "run_unmatched"


def test_stale_timestamp_after_a_clock_jump_fails_loudly_then_recovers(holder, clock):
    now = int(clock.now)
    assert call(holder, ts=now - 31).refusal_code == "stale_timestamp"
    assert call(holder, ts=now + 31).refusal_code == "stale_timestamp"
    assert call(holder, ts=now - 30).status == 200
    clock.now += 3600  # the holder's own clock jumps: every caller is stale until it resyncs
    assert call(holder, ts=now).refusal_code == "stale_timestamp"
    assert call(holder, ts=int(clock.now)).status == 200
    assert len(holder.fake.calls) == 2


def test_replayed_nonce_is_refused_and_a_retry_signs_a_fresh_one(holder):
    assert call(holder, nonce="n-1").status == 200
    assert call(holder, nonce="n-1").refusal_code == "replayed_nonce"
    assert call(holder, nonce="n-2").status == 200  # the retry's fresh nonce
    assert call(holder, nonce="n-1", caller="broker").refusal_code == "replayed_nonce"
    holder.s.clock.now += (
        100  # past the window the nonce may be forgotten (then ts is stale anyway)
    )
    assert call(holder, nonce="n-1").status == 200


def test_nonces_are_cached_only_after_the_mac_verifies(holder):
    for i in range(500):
        call(holder, secret=OTHER_SECRET, nonce=f"flood-{i}")  # unauthenticated flood
        call(holder, caller="nobody", nonce=f"flood2-{i}")
    assert len(holder._nonces) == 0
    call(holder, nonce="real", ts=int(holder.s.clock()) - 3600)  # valid MAC, stale: not cached
    assert len(holder._nonces) == 0
    call(holder, nonce="real2")
    assert len(holder._nonces) == 1


def test_nonce_cache_is_bounded_and_pruned(env, clock):
    h = env.holder(clock=clock, upstream=FakeUpstream(), max_nonces=20)
    for i in range(20):
        assert call(h, nonce=f"n{i}").status == 200
    assert call(h, nonce="one-too-many").refusal_code == "replayed_nonce"  # fails closed
    clock.now += 100
    assert call(h, nonce="after-prune").status == 200
    assert len(h._nonces) == 1
    h.close()


def test_malformed_frames_are_dropped_without_a_reply(holder):
    for payload in (b"", b"\x00\x00\x00\x03abc", framing.encode_frame({"caller": "x"}, b"")[4:]):
        assert holder.process(payload) is None


def test_response_mac_is_over_the_request_nonce(holder):
    endpoint = holder.s.config_source.get().providers["anth"]["base_url"]
    frame, nonce = framing.build_request(
        CALLER_SECRET,
        caller="broker",
        method="POST",
        endpoint=endpoint,
        path="/v1/messages",
        run_id="eval-1",
        body=body_for(),
        ts=int(holder.s.clock()),
    )
    raw = holder.process(frame[4:])
    framing.parse_reply(raw[4:], CALLER_SECRET, nonce)
    with pytest.raises(framing.ReplyRejected):
        framing.parse_reply(raw[4:], CALLER_SECRET, "someone-elses-nonce")


# --- the refusal table (§3.3), per code ---

CASES = {
    "bad_credential": dict(secret=OTHER_SECRET),
    "unknown_caller": dict(caller="ghost"),
    "stale_timestamp": dict(ts=1),
    "endpoint_unknown": dict(endpoint="https://elsewhere.example"),
    "path_not_allowed": dict(path="/v1/complete"),
    "model_not_pinned": dict(body=body_for(model="claude-sonnet-4-5-latest")),
    "limit_missing": dict(body=json.dumps({"model": MODEL, "messages": []}).encode()),
    "limit_over_ceiling": dict(body=body_for(5000)),
    "body_invalid": dict(body=body_for(100, tools=[{"name": "x"}])),
    "run_id_missing": dict(run_id=""),
    "run_unmatched": dict(run_id="adhoc-1"),
}


@pytest.mark.parametrize("code", sorted(CASES))
def test_each_refusal_is_a_completed_499_exchange_with_the_table_class(holder, env, code):
    reply = call(holder, **CASES[code])
    assert (reply.status, reply.content_type, reply.close_reason) == (
        499,
        framing.REFUSAL_CONTENT_TYPE,
        "completed",
    )
    assert reply.refusal_code == code
    assert json.loads(reply.body) == {"holderRefusal": code}
    assert classify_response(reply.status, reply.content_type, reply.body) is HOLDER_REFUSALS[code]
    assert holder.fake.calls == [] and env.standin.seen == []
    assert holder.ledger.window_total(3600) == 0


def test_https_required(env, clock):
    env.write_config(providers={"anth": env.provider(base_url="http://api.example")})
    h = env.holder(clock=clock, upstream=FakeUpstream())
    assert call(h, endpoint="http://api.example").refusal_code == "https_required"
    h.close()


def test_address_forbidden_loopback_without_allow_private(env, clock):
    env.write_config(providers={"anth": env.provider(allow_private=False)})
    up = FakeUpstream()
    h = env.holder(clock=clock, upstream=up)
    r = call(h)
    assert r.refusal_code == "address_forbidden" and up.calls == [] and env.standin.seen == []
    h.close()


def test_key_missing_fails_closed_and_moves_nothing(holder, env):
    os.unlink(env.key_path)
    r = call(holder)
    assert r.refusal_code == "key_missing" and holder.ledger.window_total(3600) == 0
    for bad in (b"", b"  \n", b"has space inside\n", b"\xff\xfe"):
        env.key_path.write_bytes(bad)
        assert call(holder).refusal_code == "key_missing"
    env.key_path.write_text("k\n")
    os.unlink(env.key_path)
    os.symlink("/etc/hostname", env.key_path)
    assert call(holder).refusal_code == "key_missing"  # O_NOFOLLOW
    assert holder.fake.calls == []


def test_run_bound_window_and_release_cap_codes(env, clock):
    env.write_config(
        holder=env.holder_table(window_max_tokens=3000, window_max_released=1),
        run_bounds=[
            {"match": "eval-*", "max_tokens": 10_000},
            {"match": "small-*", "max_tokens": 1500},
        ],
    )
    up = FakeUpstream([ok_result(10, 10), err_result(429), err_result(429)])
    h = env.holder(clock=clock, upstream=up)
    assert call(h, body=body_for(1000)).status == 200  # settles 20
    assert call(h, run_id="small-1", body=body_for(1450)).refusal_code == "run_bound_reached"
    assert call(h, run_id="eval-2", body=body_for(2900)).refusal_code == "window_ceiling_reached"
    assert call(h, body=body_for(100)).status == 429  # released (cap of 1 spent)
    assert call(h, body=body_for(100)).refusal_code == "release_cap_reached"
    assert len(up.calls) == 2  # the refused ones never sent
    h.close()


def test_every_table_code_has_a_holder_test():
    elsewhere = {
        "replayed_nonce",
        "https_required",
        "address_forbidden",
        "key_missing",
        "run_bound_reached",
        "window_ceiling_reached",
        "release_cap_reached",
    }
    assert set(CASES) | elsewhere == set(HOLDER_REFUSALS)


# --- release by outcome (§3.4; §5.2 FR-10 (6)) ---


def total(h):
    return h.ledger.window_total(3600)


def test_error_storm_of_429_moves_nothing_until_the_release_cap_then_refuses_before_send(
    env, clock
):
    up = FakeUpstream([err_result(429)])
    h = env.holder(clock=clock, upstream=up)  # window_max_released = 5
    for _ in range(5):
        assert call(h).status == 429
        assert total(h) == 0 and h.ledger.run_total("eval-1") == 0
    assert call(h).refusal_code == "release_cap_reached"
    assert len(up.calls) == 5
    clock.now += 3601  # the release count rolls off
    assert call(h).status == 429 and len(up.calls) == 6
    h.close()


@pytest.mark.parametrize(
    "result",
    [
        err_result(504),
        err_result(408),
        err_result(524),
        err_result(502, b"<html>Bad gateway</html>"),
        err_result(500, b""),
        UpstreamResult(True, "completed", 200, "application/json", b'{"no":"usage"}'),
        UpstreamResult(True, "completed", 200, "application/json", b"<html>"),
        UpstreamResult(True, "timeout"),
        UpstreamResult(True, "upstream-error"),
        UpstreamResult(True, "response-too-large"),
        UpstreamResult(True, "completed", 302, "text/html", b""),
    ],
    ids=lambda r: f"{r.close_reason}-{r.status}-{r.body[:12]}",
)
def test_sent_but_unconfirmed_calls_keep_their_reservation_until_the_window_rolls(
    env, clock, result
):
    h = env.holder(clock=clock, upstream=FakeUpstream([result]))
    body = body_for(1000)
    call(h, body=body)
    held = 1000 + len(body)
    assert total(h) == held and h.ledger.run_total("eval-1") == held
    clock.now += 3601
    assert total(h) == 0  # rolled off the window
    h.close()


def test_a_cap_bucket_holds_kept_reservations_and_recovers(env, clock):
    env.write_config(holder=env.holder_table(unmatched="cap", default_bound=3000), run_bounds=[])
    h = env.holder(clock=clock, upstream=FakeUpstream([UpstreamResult(True, "timeout")]))
    body = body_for(1000)
    assert call(h, body=body, run_id="x1").close_reason == "timeout"
    assert call(h, body=body, run_id="").close_reason == "timeout"  # missing id: the same bucket
    assert call(h, body=body, run_id="x3").refusal_code == "run_bound_reached"
    clock.now += 3601
    assert call(h, body=body, run_id="x4").close_reason == "timeout"
    h.close()


def test_never_written_requests_release_the_whole_reservation(env, clock):
    for reason in ("upstream-error", "timeout"):
        h = env.holder(clock=clock, upstream=FakeUpstream([UpstreamResult(False, reason)]))
        r = call(h)
        assert r.close_reason == reason and r.status == 0 and r.authenticated
        assert total(h) == 0
        h.close()


def test_a_call_refused_before_any_write_moves_nothing(env, clock):
    h = env.holder(clock=clock, upstream=FakeUpstream())
    for kw in (CASES["model_not_pinned"], CASES["limit_over_ceiling"], CASES["run_unmatched"]):
        call(h, **kw)
    assert total(h) == 0 and h.ledger.window_total(10**9) == 0
    h.close()


def test_usage_above_the_reservation_is_recorded_and_the_next_call_refused(env, clock):
    env.write_config(run_bounds=[{"match": "eval-*", "max_tokens": 5000}])
    up = FakeUpstream([ok_result(4900, 400)])  # provider-injected tokens beyond the 1100 reserved
    h = env.holder(clock=clock, upstream=up)
    assert call(h, body=body_for(1000)).status == 200
    assert h.ledger.run_total("eval-1") == 5300
    assert call(h, body=body_for(10)).refusal_code == "run_bound_reached"
    h.close()


def test_settle_replaces_the_reservation_with_the_reported_usage(env, clock):
    h = env.holder(clock=clock, upstream=FakeUpstream([ok_result(30, 20)]))
    call(h, body=body_for(2000))
    assert h.ledger.run_total("eval-1") == 50  # not the 2000+ reserved
    h.close()


# --- FR-10 suites (§5.2) ---


def test_run_bound_accumulates_and_the_call_that_exceeds_is_refused_with_no_send(env, clock):
    env.write_config(run_bounds=[{"match": "eval-*", "max_tokens": 1000}])
    up = FakeUpstream([ok_result(100, 100)])
    h = env.holder(clock=clock, upstream=up)
    body = body_for(300)
    outcomes = [call(h, body=body).refusal_code or "ok" for _ in range(8)]
    fits = (1000 - (300 + len(body))) // 200 + 1  # settled 200 each; the call must still reserve
    assert outcomes == ["ok"] * fits + ["run_bound_reached"] * (8 - fits) and fits == 3
    assert h.ledger.run_total("eval-1") <= 1000
    sent = len(up.calls)
    assert call(h, body=body_for(900)).refusal_code == "run_bound_reached"
    assert len(up.calls) == sent
    h.close()


def test_eight_concurrent_calls_near_the_bound_cannot_overshoot(env, clock):
    env.write_config(run_bounds=[{"match": "eval-*", "max_tokens": 3500}])
    up = FakeUpstream([ok_result(10, 10)], delay=0.3)
    h = env.holder(clock=clock, upstream=up)
    body = body_for(1000)  # reservation 1000 + len(body) ~ 1090: three fit under 3500
    replies, barrier = [], threading.Barrier(8)

    def go(i):
        barrier.wait()
        replies.append(call(h, body=body, nonce=f"c{i}"))

    threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(up.calls) == 3
    assert sorted(r.refusal_code or "ok" for r in replies) == ["ok"] * 3 + ["run_bound_reached"] * 5
    assert h.ledger.run_total("eval-1") <= 3500
    h.close()


def test_100_fresh_run_ids_stop_at_the_window_ceiling_and_it_rolls_off(env, clock):
    env.write_config(
        holder=env.holder_table(window_max_tokens=100_000, window_max_released=1000),
        run_bounds=[{"match": "eval-*", "max_tokens": 20_000}],
    )
    h = env.holder(clock=clock, upstream=FakeUpstream([ok_result(4000, 1000)]))
    codes = [
        call(h, run_id=f"eval-{n}", body=body_for(100)).refusal_code or "ok" for n in range(100)
    ]
    assert codes.count("ok") == 20 and set(codes[20:]) == {"window_ceiling_reached"}
    assert total(h) == 100_000
    clock.now += 3601
    assert call(h, run_id="eval-after").status == 200
    h.close()


def test_missing_and_fresh_ids_are_refused_or_share_one_cap_bucket(env, clock):
    h = env.holder(clock=clock, upstream=FakeUpstream())
    assert call(h, run_id="").refusal_code == "run_id_missing"
    assert {call(h, run_id=f"fresh-{i}").refusal_code for i in range(10)} == {"run_unmatched"}
    h.close()
    env.write_config(holder=env.holder_table(unmatched="cap", default_bound=2500))
    h = env.holder(clock=clock, upstream=FakeUpstream([ok_result(500, 500)]))
    codes = [
        call(h, run_id=f"fresh-{i}", body=body_for(100)).refusal_code or "ok" for i in range(10)
    ]
    assert codes.count("ok") == 3 and set(codes[3:]) == {"run_bound_reached"}  # 3 x 1000 settled
    h.close()


def test_ledger_survives_a_holder_restart_and_two_holders_cannot_share_one(env, clock):
    env.write_config(run_bounds=[{"match": "eval-*", "max_tokens": 1500}])
    h = env.holder(clock=clock, upstream=FakeUpstream([ok_result(700, 700)]))
    assert call(h, body=body_for(100)).status == 200
    with pytest.raises(HolderStartError):
        env.holder(clock=clock)  # the lock
    h.close()
    again = env.holder(clock=clock, upstream=FakeUpstream())
    assert again.ledger.run_total("eval-1") == 1400
    assert call(again, body=body_for(100)).refusal_code == "run_bound_reached"
    again.close()


# --- configuration ---


def test_holder_refuses_to_start_on_an_unprotected_configuration(env, clock):
    env.config_path.chmod(0o666)
    with pytest.raises(HolderStartError, match="owner-protected"):
        env.holder(clock=clock)


def test_a_config_that_becomes_unprotected_or_invalid_fails_closed_per_call(env, clock):
    h = env.holder(clock=clock, upstream=FakeUpstream())
    assert call(h).status == 200
    env.config_path.chmod(0o666)
    assert call(h, nonce="x1").refusal_code == "endpoint_unknown"
    env.config_path.chmod(0o644)
    assert call(h, nonce="x2").status == 200
    env.config_path.write_text("this is [not toml")
    assert call(h, nonce="x3").refusal_code == "endpoint_unknown"
    h.close()


def test_pinned_models_change_applies_on_the_next_call(env, clock):
    h = env.holder(clock=clock, upstream=FakeUpstream())
    assert call(h).status == 200
    env.write_config(providers={"anth": env.provider(pinned_models=["other-20250101"])})
    assert call(h).refusal_code == "model_not_pinned"
    h.close()


def test_hosted_provider_without_a_ceiling_refuses_to_start(env, clock):
    env.write_config(holder={"unmatched": "refuse"})
    with pytest.raises(HolderStartError, match="window"):
        env.holder(clock=clock)


def test_writable_code_root_stops_the_holder(env, clock):
    with pytest.raises(HolderStartError, match="code root"):
        env.holder(clock=clock, code_root_check=lambda: ["/opt/llmgateway: group/other-writable"])
    env.holder(clock=clock, code_root_check=lambda: []).close()


def test_local_providers_are_not_served_by_the_holder(env, clock):
    env.write_config(
        providers={"loc": env.provider(kind="lmstudio", base_url="http://127.0.0.1:1234")},
        deployment={"local_endpoints": ["http://127.0.0.1:1234"]},
    )
    h = env.holder(clock=clock)
    assert build_policy(h.s.config_source.get()).providers == {}
    h.close()


# --- key hygiene (§5.2 FR-8 (1)-(4)) ---


def test_key_is_read_per_call_so_rotation_needs_no_restart(env, clock):
    h = env.holder(clock=clock)
    call(h)
    env.key_path.write_text("sk-ant-FAKE-rotated-0001\n")
    call(h)
    assert [s.headers["x-api-key"] for s in env.standin.seen] == [
        env.marker,
        "sk-ant-FAKE-rotated-0001",
    ]
    h.close()


def test_a_failing_call_echoing_the_key_leaks_none(env, clock, capfd):
    def responder(seen):
        key = seen.headers["x-api-key"]
        return (
            401,
            {"Content-Type": "application/json"},
            json.dumps(
                {
                    "type": "error",
                    "error": {"type": "authentication_error", "message": f"bad key {key}"},
                }
            ).encode(),
        )

    env.standin.responder = responder
    h = env.holder(clock=clock)
    reply = call(h)
    assert reply.status == 401 and env.marker.encode() not in reply.body
    assert b"[redacted]" in reply.body
    h.close()
    out, err = capfd.readouterr()
    assert env.marker not in out + err
    # the whole tree: only the key file itself (and nothing the holder wrote) holds the marker
    hits = tree_search([env.root], env.marker.encode())
    assert hits == [str(env.key_path)]
    # positive control: a planted copy is found
    planted = env.root / "state" / "planted.txt"
    planted.write_text("x " + env.marker)
    assert str(planted) in tree_search([env.root], env.marker.encode())


def test_missing_key_and_unknown_provider_error_text_carry_no_key(env, clock, capfd):
    h = env.holder(clock=clock, upstream=FakeUpstream())
    os.unlink(env.key_path)
    call(h)
    call(h, endpoint="https://nope.example")
    h.close()
    out, err = capfd.readouterr()
    assert env.marker not in out + err
    assert tree_search([env.root], env.marker.encode()) == []


def test_internal_errors_log_only_the_exception_type(env, clock, capfd):
    def boom(**kw):
        raise RuntimeError("secret detail " + kw["headers"]["x-api-key"])

    h = env.holder(clock=clock, upstream=boom)
    reply = call(h)
    assert (reply.close_reason, reply.status, reply.authenticated) == ("upstream-error", 0, True)
    assert total(h) > 0  # an upstream that blew up mid-call keeps its reservation
    h.close()
    out, err = capfd.readouterr()
    assert "RuntimeError" in err and env.marker not in out + err and "secret detail" not in err


def test_calls_log_has_no_bodies_and_is_holder_only(env, clock):
    h = env.holder(clock=clock, upstream=FakeUpstream())
    call(h, body=body_for(100, system="SENSITIVE-PROMPT-TEXT"))
    call(h, run_id="adhoc")
    h.close()
    path = f"{env.state_root}/calls.log"
    text = open(path).read()
    assert "SENSITIVE-PROMPT-TEXT" not in text and env.marker not in text
    assert [json.loads(line)["status"] for line in text.splitlines()] == [200, 499]
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"


# --- the upstream, really ---


def test_a_failed_tls_handshake_never_wrote_and_releases(env, clock, tmp_path_factory):
    other_cert, _ = make_cert(tmp_path_factory.mktemp("tls2"))
    env.write_config(holder=env.holder_table(ca_file=other_cert))  # wrong CA: the handshake fails
    h = env.holder(clock=clock)
    r = call(h)
    assert r.close_reason == "upstream-error" and env.standin.seen == []
    assert total(h) == 0
    h.close()


def test_connection_refused_never_wrote_and_releases(env, clock):
    env.write_config(providers={"anth": env.provider(base_url="https://127.0.0.1:1")})
    h = env.holder(clock=clock)
    r = call(h, endpoint="https://127.0.0.1:1")
    assert r.close_reason == "upstream-error" and total(h) == 0
    h.close()


def test_a_hanging_provider_times_out_and_keeps_the_reservation(env, clock):
    env.standin.responder = lambda s: (time.sleep(2), anthropic_ok())[1]
    env.write_config(holder=env.holder_table(upstream_timeout_seconds=0.5))
    h = env.holder(clock=clock)
    start = time.monotonic()
    r = call(h, body=body_for(1000))
    assert r.close_reason == "timeout" and time.monotonic() - start < 1.8
    assert total(h) > 1000
    h.close()


def test_oversized_provider_response_is_cut_off_and_kept(env, clock):
    env.standin.responder = lambda s: (200, {"Content-Type": "application/json"}, b"x" * 5000)
    env.write_config(holder=env.holder_table(max_response_bytes=1000))
    h = env.holder(clock=clock)
    r = call(h, body=body_for(100))
    assert r.close_reason == "response-too-large" and total(h) > 100
    h.close()


def test_redirects_are_not_followed(env, clock):
    env.standin.responder = lambda s: (302, {"Location": "https://127.0.0.1:1/x"}, b"")
    h = env.holder(clock=clock)
    r = call(h)
    assert r.status == 302 and len(env.standin.seen) == 1
    h.close()


def test_provider_error_passes_through_as_a_completed_exchange(env, clock):
    env.standin.responder = lambda s: (
        429,
        {"Content-Type": "application/json"},
        b'{"type":"error","error":{"type":"rate_limit_error"}}',
    )
    h = env.holder(clock=clock)
    r = call(h)
    assert (r.status, r.close_reason) == (429, "completed")
    assert classify_response(r.status, r.content_type, r.body).value == "rate_limit"
    h.close()
