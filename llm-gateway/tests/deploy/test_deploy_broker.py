"""Tier 2: the ccg broker against the real root-owned chain and the real holder (security
Pass 10 tier-2 list (a), (b), (c), (e), (f); plan §3.6 installed manifest; AC-21).

Each test starts its own throwaway broker on a private temp socket and audit directory (never the
broker the runbook leaves running), so it cannot disturb it. A hosted forward through the holder is
`needs_provider_key` and lives in test_deploy_hosted.py."""

import os
import pathlib
import select
import shutil
import subprocess
import tempfile
import time

import deploylib as dl
import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
CCG_PY = pathlib.Path(os.environ.get("LLMGATEWAY_DEPLOY_CCG_PYTHON", REPO / "cpg/.venv/bin/python"))
INSTALLED_MANIFEST = f"{dl.ETC_DIR}/ccg/system.yaml"
TRACKED_MANIFEST = REPO / "cpg" / "ccg" / "system.yaml"


@pytest.fixture
def work():
    """A private directory under the home: the broker refuses a socket directory below the
    sticky, world-writable /tmp (its strict ancestor rule), so pytest's tmp_path cannot be used."""
    path = pathlib.Path(tempfile.mkdtemp(prefix="llmgw-probe.", dir=dl.home()))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def ccg_python():
    if not CCG_PY.exists():
        pytest.skip(
            f"no ccg interpreter at {CCG_PY}: set LLMGATEWAY_DEPLOY_CCG_PYTHON (see RUNBOOK)"
        )
    return str(CCG_PY)


def _env(tmp):
    return {"PATH": "/usr/bin:/bin", "HOME": dl.home(), "CCG_AUDIT_DIR": str(tmp / "audit")}


def _run(py, tmp, *args, timeout=60):
    (tmp / "audit").mkdir(mode=0o700, exist_ok=True)
    return subprocess.run(
        [py, "-m", "ccg.broker.server", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/",
        env=_env(tmp),
        stdin=subprocess.DEVNULL,
    )


def _flags(manifest, holder_uid, cred=None, *, owner_protected=True):
    flags = [
        "--system",
        str(manifest),
        "--key-holder-socket",
        dl.SOCKET,
        "--holder-uid",
        str(holder_uid),
    ]
    if owner_protected:
        flags.append("--require-owner-protected")
    if cred:
        flags += ["--holder-cred", cred]
    return flags


def test_broker_refuses_the_tracked_agent_writable_manifest_with_exit_69(
    ccg_python, holder_account, work
):
    r = _run(
        ccg_python,
        work,
        *_flags(TRACKED_MANIFEST, holder_account.pw_uid, dl.cred_path("broker")),
        "--socket",
        str(work / "b" / "b.sock"),
    )
    assert r.returncode == 69, (r.returncode, r.stderr)
    assert "precondition" in r.stderr and "system.yaml" in r.stderr


def test_broker_refuses_a_user_owned_copy_of_the_manifest_with_exit_69(
    ccg_python, holder_account, work
):
    mine = work / "system.yaml"
    shutil.copy(TRACKED_MANIFEST, mine)
    mine.chmod(0o644)
    r = _run(
        ccg_python,
        work,
        *_flags(mine, holder_account.pw_uid, dl.cred_path("broker")),
        "--socket",
        str(work / "b" / "b.sock"),
    )
    assert r.returncode == 69, (r.returncode, r.stderr)


def test_broker_refuses_the_agents_own_uid_as_holder_uid_with_exit_64(ccg_python, work):
    r = _run(ccg_python, work, *_flags(INSTALLED_MANIFEST, os.getuid(), dl.cred_path("broker")))
    assert r.returncode == 64, (r.returncode, r.stderr)


def test_broker_refuses_a_holder_socket_without_owner_protection_with_exit_64(
    ccg_python, holder_account, work
):
    r = _run(
        ccg_python,
        work,
        *_flags(
            INSTALLED_MANIFEST, holder_account.pw_uid, dl.cred_path("broker"), owner_protected=False
        ),
    )
    assert r.returncode == 64, (r.returncode, r.stderr)


def test_broker_refuses_a_credential_that_is_not_caller_owned_0600(
    ccg_python, holder_account, work
):
    loose = work / "loose.cred"
    loose.write_text("00" * 32 + "\n")
    loose.chmod(0o644)
    r = _run(
        ccg_python,
        work,
        *_flags(INSTALLED_MANIFEST, holder_account.pw_uid, str(loose)),
        "--socket",
        str(work / "b" / "b.sock"),
    )
    assert r.returncode == 69, (r.returncode, r.stderr)


def _start(py, tmp, *args):
    (tmp / "audit").mkdir(mode=0o700, exist_ok=True)
    (tmp / "b").mkdir(mode=0o700, exist_ok=True)
    proc = subprocess.Popen(
        [py, "-m", "ccg.broker.server", *args, "--socket", str(tmp / "b" / "b.sock")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/",
        env=_env(tmp),
        stdin=subprocess.DEVNULL,
    )
    line, end = "", time.monotonic() + 30
    while time.monotonic() < end:
        if select.select([proc.stderr], [], [], 0.5)[0]:
            line = proc.stderr.readline()
            break
        if proc.poll() is not None:
            break
    return proc, line


def _stop(proc):
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    for f in (proc.stdout, proc.stderr):
        if f:
            f.close()


def test_broker_starts_on_the_real_root_owned_chain_with_the_holder_forwarder(
    ccg_python, holder_account, holder_proc, work
):
    proc, line = _start(
        ccg_python,
        work,
        *_flags(INSTALLED_MANIFEST, holder_account.pw_uid, dl.cred_path("broker")),
    )
    try:
        assert proc.poll() is None and "listening" in line, (proc.poll(), line)
    finally:
        _stop(proc)


def test_broker_process_shows_the_credential_path_but_never_its_content(
    ccg_python, holder_account, holder_proc, broker_secret, work
):
    cred = dl.cred_path("broker")
    proc, line = _start(ccg_python, work, *_flags(INSTALLED_MANIFEST, holder_account.pw_uid, cred))
    try:
        assert "listening" in line, line
        needle = broker_secret.hex().encode()
        r = dl.probe_process(proc.pid, needle)
        assert dl.proc_file(proc.pid, "cmdline").data.count(cred.encode()) == 1
        assert r["cmdline"] == dl.Access("read", b"0")  # the path, not the content
        assert r["environ"] == dl.Access("read", b"0")
    finally:
        _stop(proc)


def test_the_installed_credential_survives_a_rotation_and_a_broker_restart(
    ccg_python, holder_account, holder_proc, work
):
    """Only after the runbook's rotation step (it exports the old credential's path)."""
    env = "LLMGATEWAY_DEPLOY_OLD_CRED_BROKER"
    if not os.environ.get(env):
        pytest.skip(f"rotation step not run: set {env} (RUNBOOK)")
    proc, line = _start(
        ccg_python,
        work,
        *_flags(INSTALLED_MANIFEST, holder_account.pw_uid, dl.cred_path("broker")),
    )
    try:
        assert "listening" in line, line
    finally:
        _stop(proc)
    r = dl.wire(dl.SOCKET, dl.load_credential(dl.cred_path("broker")), caller="broker")
    assert (r.refusal_code, r.authenticated) == ("endpoint_unknown", True)
