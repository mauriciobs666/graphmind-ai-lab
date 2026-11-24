"""Tier-2 (`-m deploy`) fixtures: tests that probe the REAL root-owned install, run as the agent
account (the human's own, not root) after the steps in `llm-gateway/deploy/RUNBOOK.md`.

Every test in this directory is marked `deploy`, so the default run deselects it. On a host with
no install `test_00_install_is_present` fails naming the missing path and every test that needs
the install skips pointing at it; the control tests that need nothing installed still run.
"""

import os
import pathlib
import pwd
import sys

import pytest

TESTS = pathlib.Path(__file__).resolve().parents[1]
for p in (str(TESTS), str(pathlib.Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import deploylib as dl  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_provider_key: needs a real hosted provider key in the holder; skipped with a "
        "message unless the human opts in (never reads or prints a key)",
    )
    # A traceback must never print locals: a test frame holds a caller credential.
    config.option.showlocals = False


def pytest_collection_modifyitems(config, items):
    here = pathlib.Path(__file__).resolve().parent
    for item in items:
        if here in pathlib.Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.deploy)


@pytest.fixture(autouse=True, scope="session")
def agent_account():
    """The probes mean nothing as root: they must run as the account agents run as."""
    if os.geteuid() == 0:
        pytest.fail(
            "run the deploy suite as the agent account (your own user), not root or via sudo: "
            "a root probe can read everything and proves nothing"
        )


@pytest.fixture(scope="session")
def install():
    """Skip (pointing at the failing presence test) when the code root is absent."""
    if not os.path.isdir(dl.CODE_ROOT):
        pytest.skip(
            f"no install at {dl.CODE_ROOT}: see test_00_install_is_present and {dl.RUNBOOK}"
        )
    return dl.CODE_ROOT


@pytest.fixture(scope="session")
def holder_account(install):
    try:
        return pwd.getpwnam(dl.HOLDER)
    except KeyError:
        pytest.fail(
            f"account {dl.HOLDER} does not exist: run `install.sh holder-account` ({dl.RUNBOOK})"
        )


@pytest.fixture(scope="session")
def holder_proc(holder_account):
    """MainPID of the running llmholder.service (a hard failure when it is not active)."""
    props = dl.systemctl_show("llmholder", "ActiveState", "MainPID")
    if props.get("ActiveState") != "active":
        pytest.fail(
            f"llmholder.service is {props.get('ActiveState')!r}, not active: "
            "`sudo systemctl enable --now llmholder`; `systemctl status llmholder`"
        )
    pid = int(props["MainPID"])
    assert pid > 1
    return pid


@pytest.fixture
def marker_child():
    child = dl.MarkerChild()
    yield child
    child.close()


@pytest.fixture(scope="session")
def broker_secret(install):
    path = dl.cred_path("broker")
    if not os.path.exists(path):
        pytest.fail(f"{path} is missing: `install.sh caller broker ...` ({dl.RUNBOOK})")
    return dl.load_credential(path)


@pytest.fixture(scope="session")
def service_secret(install):
    path = dl.cred_path("service")
    if not os.path.exists(path):
        pytest.fail(f"{path} is missing: `install.sh caller service ...` ({dl.RUNBOOK})")
    return dl.load_credential(path)
