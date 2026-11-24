"""Tier 2: the hosted path on the real host (FR-8, AC-5; plan §5.2 FR-8 (1), (3)).

Two opt-in tests, both through the real service, holder account and TLS to the real provider:

* a FAKE key (`sk-ant-FAKE-...`, installed by the human with `--key-from-stdin`): the provider
  answers 401, the call fails as class `auth`, no error text carries the key, and a host-wide
  search of everything the agent account can read finds the fake key nowhere, while the same
  search finds it where it is planted (positive control). The test refuses any value that is not
  FAKE-prefixed, so a real key can never be handed to it by mistake.
* `needs_provider_key`: one small billable call with a REAL key the human installed; the test
  never reads or prints it.

No https stand-in: the plan (§5.1, §5.2 tier 2) names only the human's real-provider smoke, and a
stand-in needs a test CA in the root-owned `[holder] ca_file`; tier 1 already runs that against
the real holder code.
"""

import os
import pathlib

import deploylib as dl
import pytest

from llmgateway import client as gw

REPO = pathlib.Path(__file__).resolve().parents[3]
FAKE_FILE, SITE, RUN = (
    "LLMGATEWAY_DEPLOY_FAKE_KEY_FILE",
    "LLMGATEWAY_DEPLOY_SMOKE_CALL_SITE",
    "LLMGATEWAY_DEPLOY_SMOKE_RUN_ID",
)
SEARCH_ROOTS = ["/tmp", "/var/tmp", "/etc", "/opt", f"/run/user/{os.getuid()}", dl.home()]


def _call(site, run):
    return gw.Client(timeout=120).call(
        site,
        [{"role": "user", "content": "Reply with the single word: ok"}],
        run_id=run,
        settings={"max_tokens": 16},
        retry=False,
        fallback=False,
    )


def _fake_key():
    """The FAKE key from the file the human installed it from (never from the environment: an
    exported key would sit in this shell's and pytest's own environ, which the search covers)."""
    path = os.environ.get(FAKE_FILE)
    if not path:
        pytest.skip(
            "fake-key hosted test not opted in: install a FAKE key with "
            "`install.sh holder-account --key-from-stdin anthropic < FILE` (RUNBOOK), then "
            f"export {FAKE_FILE}=FILE, {SITE} and {RUN}"
        )
    key = pathlib.Path(path).read_text().strip()
    assert key.startswith("sk-ant-FAKE-"), (
        f"{FAKE_FILE} must hold a FAKE-prefixed value, never a real key"
    )
    return path, key


def test_a_fake_key_hosted_call_fails_auth_leaks_nothing_and_is_found_only_where_planted(
    install, holder_proc, service_secret, tmp_path
):
    key_file, key = _fake_key()
    site, run = os.environ.get(SITE), os.environ.get(RUN)
    assert site and run, f"export {SITE} and {RUN} (RUNBOOK)"
    needle = key.encode()
    assert os.path.exists(gw.default_socket_path()), (
        "the gateway service is not running: `systemctl --user enable --now llmgateway`"
    )

    try:
        _call(site, run)
    except gw.CallFailed as exc:
        failure = exc
    else:
        pytest.fail("a FAKE key was accepted by the provider")
    if failure.failure_class in ("connection", "timeout", "outage"):
        pytest.skip(f"no usable egress to the provider ({failure.failure_class}); rerun online")
    assert failure.failure_class == "auth", (failure.failure_class, failure.code)
    leaked = any(
        key in t for t in (str(failure), repr(failure), str(failure.code), str(failure.classes))
    )
    assert not leaked, "the key appears in the error text"

    # Positive control first: the same needle, planted where an agent can reach, is found.
    planted = tmp_path / "planted.txt"
    planted.write_bytes(b"x" + needle + b"y")
    child = dl.MarkerChild(marker=key)
    try:
        assert str(planted) in dl.surface_hits(needle, ["/tmp"], budget_seconds=120)
        assert child.pid in dl.process_hits(needle)
    finally:
        child.close()
        planted.unlink()

    # The file the human installed the key from is the one place excluded (the runbook deletes it).
    found = dl.surface_hits(needle, SEARCH_ROOTS, exclude=(key_file,))
    assert found == [], "the key is in an agent-readable file (paths listed above)"
    assert dl.process_hits(needle) == [], "the key is in a process's argv or environment"
    in_history = dl.git_history_hits(str(REPO), key)
    assert not in_history, "the key is in git history"


@pytest.mark.needs_provider_key
def test_one_real_hosted_call_through_the_service_reaches_the_provider_and_returns():
    site, run = os.environ.get(SITE), os.environ.get(RUN)
    if os.environ.get(FAKE_FILE) or not (site and run):
        pytest.skip(
            f"needs_provider_key: skipped unless a REAL key is installed (not a fake) and {SITE} "
            f"and {RUN} are set to send ONE small billable call; this test never reads the key"
        )
    r = _call(site, run)
    assert r.hosted is True
