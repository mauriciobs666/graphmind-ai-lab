"""Tier 2: `keyholder check-deployment` against the real layout, and the table rows an agent can
stat (plan §3.4 artifact table). The full root run of `install.sh check` is a runbook step: the
agent cannot stat inside the holder's 0700 state directory, which is itself asserted here."""

import os

import deploylib as dl
import pytest


def _extra():
    uid = os.getuid()
    args = []
    for name in ("broker", "service"):
        if os.path.exists(dl.cred_path(name)):
            args += ["--caller-cred", f"{uid}:{dl.cred_path(name)}"]
    if os.path.exists(dl.user_unit_path()):
        args += ["--user-unit", f"{uid}:{dl.user_unit_path()}"]
    return args


def test_code_only_check_is_green(install):
    out = dl.run_check_deployment("--code-only", "--root", install)
    assert (out.returncode, out.stdout, out.stderr) == (0, "", "")


def test_full_check_as_the_agent_fails_only_inside_the_holders_state_directory(holder_account):
    out = dl.run_check_deployment(*_extra())
    lines = out.stdout.splitlines()
    assert dl.check_output_problems(lines) == [], out.stderr
    assert out.returncode == 1 and lines, (
        "the agent can stat inside /var/lib/llmholder: it is not 0700 holder-owned"
    )


def test_holder_state_directory_is_0700_and_unreadable_by_the_agent(holder_account):
    assert dl.lstat_problems(dl.STATE_ROOT, uid=holder_account.pw_uid, mode=0o700, kind="dir") == []
    with pytest.raises(PermissionError):
        os.listdir(dl.STATE_ROOT)
    for sub in ("keys", "registry"):
        assert dl.read_attempt(f"{dl.STATE_ROOT}/{sub}").status == "denied"
        with pytest.raises(PermissionError):
            os.lstat(f"{dl.STATE_ROOT}/{sub}")


def test_run_directory_is_0711_traverse_only_and_socket_is_0666(holder_account, holder_proc):
    uid = holder_account.pw_uid
    assert dl.lstat_problems(dl.RUN_DIR, uid=uid, mode=0o711, kind="dir") == []
    assert dl.lstat_problems(dl.SOCKET, uid=uid, mode=0o666, kind="sock") == []
    with pytest.raises(PermissionError):
        os.listdir(dl.RUN_DIR)  # traverse only: the agent cannot list the directory


def test_caller_credentials_are_caller_owned_0600_one_link(install):
    for name in ("broker", "service"):
        path = dl.cred_path(name)
        assert os.path.exists(path), f"{path} missing: `install.sh caller {name} ...`"
        assert dl.credential_shape_ok(path), f"{path}: not a regular, owner-only, 64-hex file"
        assert dl.lstat_problems(path, uid=os.getuid(), mode=0o600, kind="file") == []


def test_holder_cannot_read_the_callers_credential_and_agent_cannot_read_the_registry(
    holder_account,
):
    """The reverse isolation: the registry secrets are holder-owned 0600 under a 0700 directory."""
    for name in ("broker", "service"):
        assert dl.read_attempt(f"{dl.STATE_ROOT}/registry/{name}.secret").status == "denied"
    # The credential is mine, 0600: a different uid (the holder) has no group/other bits to use.
    for name in ("broker", "service"):
        assert os.stat(dl.cred_path(name)).st_mode & 0o077 == 0
