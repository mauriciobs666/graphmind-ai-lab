"""Tier 2: the gateway service as the systemd --user unit runs it (plan §3.1, §3.4; FR-8, FR-11).

The service reads `/etc/llmgateway/gateway.toml` (root-trusted) and its credential once at start;
these tests drive it through the stdlib client, which needs no hosted provider."""

import os
import stat
import subprocess

import deploylib as dl
import pytest

from llmgateway import client as gw


@pytest.fixture(scope="module")
def service_pid(install):
    out = subprocess.run(
        ["systemctl", "--user", "show", "llmgateway", "-p", "ActiveState", "-p", "MainPID"],
        capture_output=True,
        text=True,
    )
    props = dict(ln.split("=", 1) for ln in out.stdout.splitlines() if "=" in ln)
    if props.get("ActiveState") != "active":
        pytest.fail(
            f"llmgateway.service (user) is {props.get('ActiveState')!r}: "
            "`systemctl --user enable --now llmgateway`; `journalctl --user -u llmgateway`"
        )
    return int(props["MainPID"])


def test_service_socket_is_private_to_the_account(service_pid):
    path = gw.default_socket_path()
    d = os.lstat(os.path.dirname(path))
    assert stat.S_ISDIR(d.st_mode) and d.st_uid == os.getuid() and stat.S_IMODE(d.st_mode) == 0o700
    s = os.lstat(path)
    assert stat.S_ISSOCK(s.st_mode) and s.st_uid == os.getuid() and stat.S_IMODE(s.st_mode) == 0o600


def test_service_answers_and_refuses_embeddings(service_pid):
    with pytest.raises(gw.EmbeddingsNotAvailable):
        gw.Client(timeout=30).embed()


def test_service_refuses_an_unknown_call_site_without_touching_a_provider(service_pid):
    with pytest.raises(gw.CallFailed) as exc:
        gw.Client(timeout=30).call(
            "deploy-probe-no-such-call-site", [{"role": "user", "content": "x"}], retry=False
        )
    assert exc.value.failure_class in ("config", "policy_refused", "rejected")


def test_service_process_carries_no_credential_content(service_pid, service_secret):
    needle = dl.secret_needle(service_secret)
    r = dl.probe_process(service_pid, needle)
    assert r["cmdline"] == dl.Access("read", b"0")
    assert r["environ"] == dl.Access("read", b"0")
    assert r["mem"].status == "read"  # own uid: the credential IS in its memory (stated limitation)


def test_service_credential_is_a_real_separate_secret(service_secret, broker_secret):
    dl.assert_secrets_differ(
        service_secret, broker_secret, "the service and the broker share one credential"
    )
