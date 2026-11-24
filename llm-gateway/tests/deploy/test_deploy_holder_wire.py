"""Tier 2: the real holder on the real socket, spoken to as the agent speaks (plan §5.2 FR-8 (5);
AC-5 direct-call half; security Pass 10 (b), (g)).

Every request goes to `https://deploy-probe.invalid:443`, which is never a configured provider, so
a request that passes authentication is answered `endpoint_unknown` (signed) and nothing is ever
sent upstream and no key is touched. Credentials are read only to sign; nothing here prints one.
"""

import os
import time

import deploylib as dl
import pytest

from llmgateway import framing

SOCK = dl.SOCKET


def _code(reply):
    return reply.refusal_code


def test_a_valid_broker_credential_is_authenticated_and_reaches_provider_lookup(
    holder_proc, broker_secret
):
    r = dl.wire(SOCK, broker_secret, caller="broker")
    assert (r.status, _code(r), r.authenticated) == (499, "endpoint_unknown", True)


def test_a_valid_service_credential_is_authenticated_too(holder_proc, service_secret):
    r = dl.wire(SOCK, service_secret, caller="service")
    assert (r.status, _code(r), r.authenticated) == (499, "endpoint_unknown", True)


def test_no_wrong_or_swapped_credential_is_served(holder_proc, broker_secret, service_secret):
    wrong = bytes(32)
    r = dl.wire(SOCK, wrong, caller="broker")
    assert (_code(r), r.authenticated) == ("bad_credential", False)
    r = dl.wire(
        SOCK, broker_secret, caller="service"
    )  # the broker's secret under the service's name
    assert (_code(r), r.authenticated) == ("bad_credential", False)
    r = dl.wire(SOCK, service_secret, caller="broker")
    assert (_code(r), r.authenticated) == ("bad_credential", False)
    r = dl.wire(SOCK, broker_secret, caller="nobody")
    assert (_code(r), r.authenticated) == ("unknown_caller", False)


def test_a_replayed_request_is_refused(holder_proc, broker_secret):
    frame, nonce = framing.build_request(
        broker_secret,
        caller="broker",
        method="POST",
        endpoint=dl.PROBE_ENDPOINT,
        path="/v1/messages",
        run_id="deploy-probe-replay",
        body=b"{}",
    )
    first = dl.roundtrip(SOCK, frame, broker_secret, nonce)
    second = dl.roundtrip(SOCK, frame, broker_secret, nonce)
    assert (_code(first), _code(second)) == ("endpoint_unknown", "replayed_nonce")


def test_a_stale_or_future_timestamp_is_refused_and_a_small_skew_is_not(holder_proc, broker_secret):
    now = int(time.time())
    assert _code(dl.wire(SOCK, broker_secret, caller="broker", ts=now - 60)) == "stale_timestamp"
    assert _code(dl.wire(SOCK, broker_secret, caller="broker", ts=now + 60)) == "stale_timestamp"
    assert _code(dl.wire(SOCK, broker_secret, caller="broker", ts=now - 15)) == "endpoint_unknown"


def test_wsl_clock_skew_is_observed(holder_proc, broker_secret, capsys):
    """Plan §3.4 / security Pass 10 (g): the 30 s window against this host's clock. The holder
    and the broker share one kernel clock, so the windows only matter if the clock steps between
    signing and checking (a Windows sleep/resume steps a WSL2 clock). The stale refusals above are
    the seen-once evidence; this prints how far the Windows clock is, when interop allows."""
    skew = dl.windows_clock_skew()
    with capsys.disabled():
        print(
            "\n[deploy] WSL clock vs Windows host: "
            + ("interop unavailable" if skew is None else f"{skew:+.2f} s")
            + " (holder skew window is 30 s)"
        )
    now = int(time.time())
    inside = dl.wire(SOCK, broker_secret, caller="broker", ts=now - 29)
    outside = dl.wire(SOCK, broker_secret, caller="broker", ts=now - 32)
    assert (_code(inside), _code(outside)) == ("endpoint_unknown", "stale_timestamp")


def test_junk_and_oversized_frames_are_dropped_without_a_hang(holder_proc):
    assert dl.silent_close(SOCK, b"\x00\x01garbage-not-a-frame")
    assert dl.silent_close(SOCK, (0xFFFFFFF0).to_bytes(4, "big"))  # a 4 GiB declared frame
    assert dl.silent_close(SOCK, b"")  # connect-and-close


def test_the_holder_socket_directory_is_the_holders(holder_account, holder_proc):
    framing.check_socket_dir(SOCK, holder_account.pw_uid)  # a caller's own pre-connect check
    with pytest.raises(framing.HolderUnreachable):
        framing.check_socket_dir(SOCK, os.getuid())  # control: wrong owner is refused


# --- rotation (the runbook rotates, then sets the env vars below) -------------------------------

ROTATED = {
    "broker": "LLMGATEWAY_DEPLOY_OLD_CRED_BROKER",
    "service": "LLMGATEWAY_DEPLOY_OLD_CRED_SERVICE",
}


@pytest.mark.parametrize("name", sorted(ROTATED))
def test_rotation_replaces_the_credential_and_the_old_one_stops_working(holder_proc, name):
    env = ROTATED[name]
    old_path = os.environ.get(env)
    if not old_path:
        pytest.skip(
            f"rotation step not run: set {env} to a copy of the pre-rotation credential (RUNBOOK)"
        )
    old, new = dl.load_credential(old_path), dl.load_credential(dl.cred_path(name))
    dl.assert_secrets_differ(
        old, new, "the credential did not change: `install.sh caller ... --rotate` was not run"
    )
    r_old = dl.wire(SOCK, old, caller=name)
    assert (_code(r_old), r_old.authenticated) == ("bad_credential", False)
    r_new = dl.wire(SOCK, new, caller=name)  # no holder restart: the registry is read per request
    assert (_code(r_new), r_new.authenticated) == ("endpoint_unknown", True)


def test_the_holder_log_line_shape_is_checked_by_the_human_not_here(holder_proc):
    """`calls.log` is holder-only; the runbook has the human read it with `sudo ... log` and paste
    the lines carrying run ids `deploy-probe-*` (no body, no key). The agent must not read it."""
    assert dl.read_attempt(f"{dl.STATE_ROOT}/calls.log").status == "denied"
