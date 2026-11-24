"""The tier-2 helpers proved against controls that need no install: pid 1 as a foreign-uid process,
own-uid planted markers, a temp tree, and a REAL tier-1 holder process on a temp socket. These run
on any host under `-m deploy`; they fail if a probe went blind or a check went vacuous."""

import os
import pathlib
import re
import socket
import stat
import time

import deploylib as dl
import pytest
from holder_harness import CALLER_SECRET, UID, Env, HolderProc, StandIn, make_cert

from llmgateway import framing

REPO = pathlib.Path(__file__).resolve().parents[2]


# --- the unit expectation is the installer's template, and every deviation is caught -------------


def _template() -> str:
    text = (REPO / "deploy" / "install.sh").read_text()
    m = re.search(r"cat >\"\$tmp\" <<'UNIT'\n(.*?)\nUNIT\n", text, re.S)
    assert m, "the unit heredoc moved in install.sh"
    return m.group(1)


def test_the_expected_unit_equals_the_installers_template():
    assert dl.unit_problems(_template()) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("NoNewPrivileges=yes\n", ""),
        ("ProtectSystem=strict", "ProtectSystem=full"),
        ("CapabilityBoundingSet=", "CapabilityBoundingSet=CAP_NET_ADMIN"),
        ("User=llmholder", "User=root"),
        ("LimitCORE=0\n", ""),
        ("PrivateTmp=yes", "PrivateTmp=no"),
        ("ProtectProc=invisible", "ProtectProc=default"),
        ("RuntimeDirectoryMode=0711", "RuntimeDirectoryMode=0755"),
        (
            "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
            "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK",
        ),
        ("ExecStart=/usr/bin/python3 -I", "ExecStart=/usr/bin/python3"),
        ("Restart=on-failure", "Restart=on-failure\nEnvironment=PYTHONPATH=/tmp"),
        ("Restart=on-failure", "Restart=on-failure\nExecStartPre=/bin/true"),
        ("Restart=on-failure", "Restart=on-failure\nUser=llmholder"),
        ("ReadWritePaths=/var/lib/llmholder", "ReadWritePaths=/var/lib/llmholder /home"),
    ],
)
def test_every_deviation_from_the_hardening_list_is_reported(old, new):
    text = _template()
    assert old in text
    assert dl.unit_problems(text.replace(old, new)), (old, new)


def test_agent_run_check_output_filter_is_exactly_the_holder_state_denials():
    ok = f"{dl.STATE_ROOT}/keys: Permission denied"
    assert dl.check_output_problems([ok]) == []
    for bad in (
        f"{dl.STATE_ROOT}/keys: mode 0755, want 0700",
        "/etc/llmgateway/gateway.toml: Permission denied",
        f"{dl.STATE_ROOT}/keys: missing",
        "/opt/llmgateway: owned by uid 1000",
    ):
        assert dl.check_output_problems([ok, bad]) == [bad]


# --- file probes ----------------------------------------------------------------------------------


def test_file_attempts_distinguish_denied_from_allowed(tmp_path):
    closed = tmp_path / "closed"
    closed.mkdir(mode=0o500)
    assert dl.create_attempt(str(closed)).status == "denied"
    f = closed / "f"
    open_dir = tmp_path / "open"
    open_dir.mkdir(mode=0o700)
    assert dl.create_attempt(str(open_dir)).status == "read"
    assert list(open_dir.iterdir()) == []  # the probe file was removed
    g = open_dir / "g"
    g.write_text("x")
    g.chmod(0o400)
    assert dl.write_attempt(str(g)).status == "denied"
    g.chmod(0o600)
    assert dl.write_attempt(str(g)).status == "read"
    assert f.exists() is False
    assert dl.read_attempt(str(tmp_path / "absent")).status == "gone"
    closed.chmod(0o700)


def test_lstat_problems_report_owner_mode_and_kind(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    f.chmod(0o600)
    assert dl.lstat_problems(str(f), uid=os.getuid(), mode=0o600, kind="file") == []
    assert len(dl.lstat_problems(str(f), uid=os.getuid() + 1, mode=0o600, kind="file")) == 1
    assert len(dl.lstat_problems(str(f), uid=os.getuid(), mode=0o644, kind="file")) == 1
    assert len(dl.lstat_problems(str(f), uid=os.getuid(), mode=0o600, kind="dir")) == 1
    link = tmp_path / "l"
    link.symlink_to(f)
    assert dl.lstat_problems(
        str(link), uid=os.getuid(), mode=0o600, kind="file"
    )  # lstat: not a file


def test_code_root_check_reports_writable_symlinked_and_foreign_entries(tmp_path):
    root = tmp_path / "opt" / "llmgateway"
    (root / "bin").mkdir(parents=True)
    for d in (tmp_path / "opt", root, root / "bin"):
        d.chmod(0o755)
    (root / "bin" / "k.py").write_text("x")
    (root / "bin" / "k.py").chmod(0o644)
    me, anchor = os.getuid(), str(tmp_path)
    assert dl.code_root_problems(str(root), trusted=me, anchor=anchor) == []
    (root / "bin" / "k.py").chmod(0o664)
    assert any(
        "group/other-writable" in p
        for p in dl.code_root_problems(str(root), trusted=me, anchor=anchor)
    )
    (root / "bin" / "k.py").chmod(0o644)
    (root / "link").symlink_to("/etc")
    assert any(
        "not a regular file" in p
        for p in dl.code_root_problems(str(root), trusted=me, anchor=anchor)
    )
    (root / "link").unlink()
    assert any(
        "owner uid" in p for p in dl.code_root_problems(str(root), trusted=me + 1, anchor=anchor)
    )
    (tmp_path / "opt").chmod(0o777)
    assert any(
        "opt: group/other-writable" in p
        for p in dl.code_root_problems(str(root), trusted=me, anchor=anchor)
    )


def test_credential_shape(tmp_path):
    c = tmp_path / "c.cred"
    c.write_text("ab" * 32 + "\n")
    c.chmod(0o600)
    assert dl.credential_shape_ok(str(c))
    c.chmod(0o640)
    assert not dl.credential_shape_ok(str(c))
    c.chmod(0o600)
    c.write_text("zz" * 32 + "\n")
    assert not dl.credential_shape_ok(str(c))
    c.write_text("ab" * 31 + "\n")
    assert not dl.credential_shape_ok(str(c))
    d = tmp_path / "d.cred"
    os.link(c, d)
    c.write_text("ab" * 32 + "\n")
    assert not dl.credential_shape_ok(str(c))  # two links


# --- process probes: foreign uid is refused, own uid is not -----------------------------------


def test_pid_1_is_a_foreign_uid_process_that_every_probe_is_refused():
    """A denial control that needs no install: init runs as root."""
    needle = b"sk-ant-FAKE-never"
    r = dl.probe_process(1, needle)
    for name in ("environ", "mem", "maps", "fd", "cwd"):
        assert r[name].status == "denied", (name, r[name])
    assert dl.ptrace_seize(1) == (-1, 1)  # EPERM
    assert dl.proc_file(1, "mem").status == "denied"


def test_probe_process_distinguishes_marker_present_from_absent(marker_child):
    assert dl.probe_process(marker_child.pid, b"sk-ant-FAKE-" + b"0" * 32)["mem"] == dl.Access(
        "read", b"0"
    )
    assert dl.probe_process(marker_child.pid, b"sk-ant-FAKE-" + b"0" * 32)["environ"] == dl.Access(
        "read", b"0"
    )
    assert dl.probe_process(marker_child.pid, marker_child.marker.encode())["mem"] == dl.Access(
        "read", b"1"
    )


def test_sudo_refusal_helper_reports_a_refusal_for_an_unknown_target():
    refused, _err = dl.sudo_refused("/usr/bin/true")  # llmholder is absent or refused: both refuse
    assert refused


def test_docker_helper_is_false_for_a_closed_socket(tmp_path):
    p = str(tmp_path / "closed.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(p)
    srv.listen(1)
    os.chmod(p, 0o000)
    try:
        assert dl.docker_reachable(p) is False
    finally:
        srv.close()


# --- the wire helpers against a real (tier-1) holder process ----------------------------------


@pytest.fixture
def live(tmp_path):
    cert, key = make_cert(tmp_path)
    standin = StandIn(cert, key)
    env = Env(tmp_path / "env", standin, cert)
    proc = HolderProc(env)
    yield env
    proc.stop()
    standin.close()


def test_wire_probes_against_a_real_holder_process(live):
    sock = live.socket_path
    r = dl.wire(sock, CALLER_SECRET, caller="broker")
    assert (r.status, r.refusal_code, r.authenticated) == (499, "endpoint_unknown", True)
    r = dl.wire(sock, bytes(32), caller="broker")
    assert (r.refusal_code, r.authenticated) == ("bad_credential", False)
    r = dl.wire(sock, CALLER_SECRET, caller="nobody")
    assert (r.refusal_code, r.authenticated) == ("unknown_caller", False)
    now = int(time.time())
    assert (
        dl.wire(sock, CALLER_SECRET, caller="broker", ts=now - 60).refusal_code == "stale_timestamp"
    )
    assert (
        dl.wire(sock, CALLER_SECRET, caller="broker", ts=now - 29).refusal_code
        == "endpoint_unknown"
    )
    assert (
        dl.wire(sock, CALLER_SECRET, caller="broker", ts=now - 32).refusal_code == "stale_timestamp"
    )
    frame, nonce = framing.build_request(
        CALLER_SECRET,
        caller="broker",
        method="POST",
        endpoint=dl.PROBE_ENDPOINT,
        path="/v1/messages",
        run_id="r",
        body=b"{}",
    )
    a = dl.roundtrip(sock, frame, CALLER_SECRET, nonce)
    b = dl.roundtrip(sock, frame, CALLER_SECRET, nonce)
    assert (a.refusal_code, b.refusal_code) == ("endpoint_unknown", "replayed_nonce")
    assert dl.silent_close(sock, b"\x00\x01garbage-not-a-frame")
    assert dl.silent_close(sock, (0xFFFFFFF0).to_bytes(4, "big"))
    framing.check_socket_dir(sock, UID)
    assert stat.S_ISSOCK(os.lstat(sock).st_mode)


def test_silent_close_is_false_for_a_peer_that_answers_or_hangs(tmp_path):
    import threading

    def serve(path, behaviour):
        srv = socket.socket(socket.AF_UNIX)
        srv.bind(path)
        srv.listen(1)

        def run():
            conn, _ = srv.accept()
            if behaviour == "answer":
                conn.sendall(b"x")
            else:
                time.sleep(1.5)
            conn.close()
            srv.close()

        threading.Thread(target=run, daemon=True).start()

    answer, hang = str(tmp_path / "a.sock"), str(tmp_path / "h.sock")
    serve(answer, "answer")
    serve(hang, "hang")
    assert dl.silent_close(answer, b"junk", timeout=3) is False
    assert dl.silent_close(hang, b"junk", timeout=0.4) is False


def test_host_search_finds_a_planted_needle_in_files_and_processes_and_only_there(tmp_path):
    needle = "sk-ant-FAKE-" + "ab" * 16
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f.txt").write_bytes(b"..." + needle.encode() + b"...")
    (tmp_path / "sub" / "other.txt").write_bytes(b"nothing here")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "skipped.txt").write_bytes(needle.encode())
    assert dl.surface_hits(needle.encode(), [str(tmp_path)]) == [str(tmp_path / "sub" / "f.txt")]
    assert dl.surface_hits(b"sk-ant-FAKE-absent", [str(tmp_path)]) == []
    excluded = str(tmp_path / "sub" / "f.txt")
    assert dl.surface_hits(needle.encode(), [str(tmp_path)], exclude=(excluded,)) == []
    child = dl.MarkerChild(marker=needle)
    try:
        assert child.pid in dl.process_hits(needle.encode())
        assert child.pid not in dl.process_hits(b"sk-ant-FAKE-absent")
    finally:
        child.close()


def test_git_history_pickaxe_finds_a_needle_in_a_history_and_not_otherwise(tmp_path):
    import subprocess

    env = {
        "PATH": "/usr/bin:/bin",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    r = tmp_path / "r"
    r.mkdir()
    for argv in (["init", "-q"], ["config", "commit.gpgsign", "false"]):
        subprocess.run(["git", "-C", str(r), *argv], check=True, env=env)
    (r / "f").write_text("sk-ant-FAKE-histneedle\n")
    subprocess.run(["git", "-C", str(r), "add", "f"], check=True, env=env)
    subprocess.run(["git", "-C", str(r), "commit", "-q", "-m", "x"], check=True, env=env)
    assert dl.git_history_hits(str(r), "sk-ant-FAKE-histneedle") is True
    assert dl.git_history_hits(str(r), "sk-ant-FAKE-absent") is False
