"""Tier 2: the AC-5 agent probe (plan §5.2 FR-8 tier 2; FR-8, AC-5).

The probe runs as the agent runs: the human's own account, a shell, `/proc`, `ptrace`, `docker`,
`sudo`. Each test pairs a POSITIVE CONTROL (the same probe finds a planted FAKE marker in an
own-uid file/process, a place agents can reach) with the probe aimed at the holder, which must be
refused. The holder's key is never read: the holder-side assertions are "denied", never "found".
"""

import errno
import grp
import os
import pwd
import subprocess

import deploylib as dl
import pytest

EXPECTED_CMDLINE = b"/usr/bin/python3\0-I\0/opt/llmgateway/bin/keyholder.py\0"


# --- the holder's account ------------------------------------------------------------------------


def test_holder_account_is_a_locked_down_system_account(holder_account):
    assert holder_account.pw_uid != os.getuid() and holder_account.pw_uid != 0
    assert holder_account.pw_uid < 1000, "not a system account"
    assert holder_account.pw_dir == dl.STATE_ROOT
    assert holder_account.pw_shell in ("/usr/sbin/nologin", "/sbin/nologin", "/bin/false")
    assert grp.getgrgid(holder_account.pw_gid).gr_name == dl.HOLDER
    for g in grp.getgrall():
        if g.gr_name in ("docker", "sudo", "wheel", "admin", "adm", "root", "shadow"):
            assert dl.HOLDER not in g.gr_mem, f"{dl.HOLDER} is in group {g.gr_name}"


def test_agent_shares_no_group_with_the_holder(holder_account):
    assert holder_account.pw_gid not in os.getgroups() + [os.getgid()]
    for g in grp.getgrall():
        if g.gr_gid == holder_account.pw_gid:
            assert pwd.getpwuid(os.getuid()).pw_name not in g.gr_mem


# --- files ---------------------------------------------------------------------------------------


def test_probe_finds_a_planted_marker_in_an_agent_reachable_key_directory(tmp_path):
    """Positive control: the same kind of place as the holder's (0700 dir, 0600 key file)."""
    keys = tmp_path / "state" / "keys"
    keys.mkdir(parents=True, mode=0o700)
    (tmp_path / "state").chmod(0o700)
    marker = dl.fake_marker()
    (keys / "anthropic.key").write_text(marker + "\n")
    (keys / "anthropic.key").chmod(0o600)
    hits, _denied = dl.tree_hits(str(tmp_path / "state"), marker.encode())
    assert hits == [str(keys / "anthropic.key")]


def test_agent_cannot_reach_the_holders_files(holder_account):
    hits, denied = dl.tree_hits(dl.STATE_ROOT, b"sk-")
    assert hits == [] and dl.STATE_ROOT in denied, (
        "the agent could list or read inside the holder's state directory"
    )
    for rel in (
        "keys/anthropic.key",
        "keys/openai.key",
        "registry/broker.secret",
        "registry/service.secret",
        "ledger.sqlite3",
        "ledger.sqlite3-wal",
        "calls.log",
    ):
        a = dl.read_attempt(f"{dl.STATE_ROOT}/{rel}")
        # `denied`, never `gone`: a missing-file answer would mean the probe never got that far.
        assert a.status == "denied", f"{rel}: {a.status} {a.err}"
    assert dl.write_attempt(f"{dl.STATE_ROOT}/calls.log").status == "denied"
    assert dl.create_attempt(dl.STATE_ROOT).status == "denied"
    assert dl.create_attempt(f"{dl.STATE_ROOT}/keys").status == "denied"


def test_agent_cannot_list_the_run_directory(holder_account, holder_proc):
    """Traverse only (0711): listing is refused. Connecting is test_deploy_holder_wire."""
    assert dl.read_attempt(dl.RUN_DIR).status in ("denied", "error")
    with pytest.raises(PermissionError):
        os.listdir(dl.RUN_DIR)


# --- processes: environ, cmdline, memory, fds, ptrace -------------------------------------------


def test_probe_finds_a_planted_marker_in_an_own_uid_process(marker_child):
    """Positive control for the process probe: argv, environment and memory of own-uid."""
    r = dl.probe_process(marker_child.pid, marker_child.marker.encode())
    assert r["cmdline"] == dl.Access("read", b"1")
    assert r["environ"] == dl.Access("read", b"1")
    assert r["mem"] == dl.Access("read", b"1")
    assert (r["maps"].status, r["fd"].status, r["cwd"].status) == ("read", "read", "read")


def test_ptrace_seize_succeeds_on_an_own_uid_process(marker_child):
    """Positive control for the ptrace attempt (no Yama on this host: same-uid is allowed)."""
    assert dl.ptrace_seize(marker_child.pid) == (0, 0)


def test_agent_probe_of_the_holder_process_is_refused(holder_proc, marker_child):
    needle = marker_child.marker.encode()  # a fake: the holder's key is never known or read
    r = dl.probe_process(holder_proc, needle)
    for name in ("environ", "mem", "maps", "fd", "cwd"):
        assert r[name].status == "denied", f"/proc/{holder_proc}/{name}: {r[name]}"
    assert r["cmdline"] == dl.Access("read", b"0")
    assert dl.proc_file(holder_proc, "cmdline").data == EXPECTED_CMDLINE
    assert dl.proc_file(holder_proc, "mem").status == "denied"


def test_agent_cannot_ptrace_the_holder(holder_proc):
    assert dl.ptrace_seize(holder_proc) == (-1, errno.EPERM)


def _ps_env_visible(pid: int) -> bool:
    assert os.path.exists(f"/proc/{pid}"), f"pid {pid} is gone: both `ps` outputs would be empty"
    plain = subprocess.run(
        ["ps", "ww", "-o", "args=", "-p", str(pid)], capture_output=True, text=True
    ).stdout
    with_env = subprocess.run(
        ["ps", "eww", "-o", "args=", "-p", str(pid)], capture_output=True, text=True
    ).stdout
    return plain.strip() != with_env.strip()


def test_ps_shows_the_environment_of_an_own_uid_process_but_not_the_holders(
    marker_child, holder_proc
):
    assert _ps_env_visible(marker_child.pid), "positive control: `ps e` should show own-uid env"
    assert not _ps_env_visible(holder_proc)


def test_the_holder_process_is_non_dumpable_and_has_no_core_files(holder_proc, holder_account):
    """`_harden`: PR_SET_DUMPABLE=0 makes /proc/<pid> root-owned; RLIMIT_CORE=0; LimitCORE=0."""
    assert os.stat(f"/proc/{holder_proc}").st_uid == 0, (
        "holder is dumpable (PR_SET_DUMPABLE not applied)"
    )
    limits = dl.proc_file(holder_proc, "limits").data.decode()
    core = [ln for ln in limits.splitlines() if ln.startswith("Max core file size")][0].split()
    assert core[-3:-1] == ["0", "0"], core
    uid = str(holder_account.pw_uid)
    assert dl.proc_status(holder_proc)["Uid"].split() == [uid] * 4


# --- docker, sudo, other uids ---------------------------------------------------------------------


def test_container_socket_probe_is_blind_without_a_reachable_socket(tmp_path):
    """Positive control: the probe reports a socket the agent can connect to, via the list, via
    `DOCKER_HOST=unix://` and via `DOCKER_HOST=tcp://`, and nothing for a closed or absent one."""
    import socket

    p = str(tmp_path / "d.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(p)
    srv.listen(5)
    tcp = socket.socket()
    tcp.bind(("127.0.0.1", 0))
    tcp.listen(5)
    try:
        assert dl.docker_reachable(p) is True
        assert dl.reachable_container_sockets({}, extra=(p,), system=()) == [p]
        assert dl.reachable_container_sockets({"DOCKER_HOST": "unix://" + p}, system=()) == [
            "unix://" + p
        ]
        host = f"tcp://127.0.0.1:{tcp.getsockname()[1]}"
        assert dl.reachable_container_sockets({"DOCKER_HOST": host}, system=()) == [host]
    finally:
        srv.close()
        tcp.close()
    assert dl.docker_reachable(str(tmp_path / "absent.sock")) is False
    assert dl.reachable_container_sockets({"DOCKER_HOST": "tcp://127.0.0.1:1"}, system=()) == []


def test_agent_account_has_no_docker_access(holder_account):
    """Docker-group access is root-equivalent (`docker run -v /var/lib/llmholder:/h` reads the
    holder's files), so a dedicated account protects nothing while the agent can use a container
    runtime (docs/reviews/llm-gateway-security.md B2). A red result here leaves AC-5 and D14
    OPEN. Remove the account from the group and log in again (RUNBOOK step 0)."""
    docker_gid = grp.getgrnam("docker").gr_gid if _has_group("docker") else None
    assert docker_gid is None or docker_gid not in os.getgroups(), (
        "this session's account is in group docker (root-equivalent): see RUNBOOK step 0"
    )
    reachable = dl.reachable_container_sockets()
    assert reachable == [], f"container sockets reachable by the agent account: {reachable}"
    if docker_gid is not None:
        members = grp.getgrgid(docker_gid).gr_mem
        assert pwd.getpwuid(os.getuid()).pw_name not in members, (
            "listed in /etc/group's docker line"
        )


def _has_group(name: str) -> bool:
    try:
        grp.getgrnam(name)
        return True
    except KeyError:
        return False


def test_sudo_into_the_holder_account_is_refused(holder_account):
    """No sudoers rule: `-n` never prompts, `-k` ignores a cached ticket from the human's sudo.
    No own-uid positive control exists (a successful sudo needs privilege); the message check is
    what keeps a non-refusal failure (sudo missing, wrong flags) from passing."""
    refused, err = dl.sudo_refused("/usr/bin/true")
    assert refused, f"sudo -n -u {dl.HOLDER} /usr/bin/true was not refused: {err!r}"
    assert "password is required" in err or "not allowed" in err or "may not run" in err, err
    refused, err = dl.sudo_refused(dl.PY, "-I", f"{dl.CODE_ROOT}/bin/keyholder.py", "log")
    assert refused, f"the holder's log is readable without a password: {err!r}"


def test_runuser_into_the_holder_is_refused_for_a_non_root_caller(holder_account):
    """`runuser` is root-only. There is no own-uid positive control (it would need privilege); the
    assertion is the refusal message itself, not merely a non-zero exit."""
    out = subprocess.run(
        ["runuser", "-u", dl.HOLDER, "--", "/usr/bin/id"],
        capture_output=True,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )
    assert out.returncode != 0 and out.stdout == ""
    assert "may not be used by non-root" in out.stderr, out.stderr
