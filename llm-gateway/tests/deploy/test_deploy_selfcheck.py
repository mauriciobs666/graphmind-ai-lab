"""The tier-2 helpers proved against controls that need no install: pid 1 as a foreign-uid process,
own-uid planted markers, a temp tree, and a REAL tier-1 holder process on a temp socket. These run
on any host under `-m deploy`; they fail if a probe went blind or a check went vacuous."""

import os
import pathlib
import pwd
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
    r = dl.surface_hits(needle.encode(), [str(tmp_path)])
    assert (r.hits, r.complete) == ([str(tmp_path / "sub" / "f.txt")], True)
    r = dl.surface_hits(b"sk-ant-FAKE-absent", [str(tmp_path)])
    assert (r.hits, r.complete) == ([], True) and r.files == 2
    excluded = str(tmp_path / "sub" / "f.txt")
    r = dl.surface_hits(needle.encode(), [str(tmp_path)], exclude=(excluded,))
    assert (r.hits, r.complete) == ([], True)
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


def test_a_failing_test_prints_no_credential_under_any_traceback_style(tmp_path):
    """The paste-back holds pytest output. Control: plain bytes DO leak (operands of an assert,
    and the arguments of the last frame under `--tb=long`), so the grep is not vacuous; a
    `Secret` (what `load_credential` returns) leaks in neither place, in any style."""
    import subprocess
    import sys

    raw = os.urandom(32)
    forms = (raw.hex(), repr(bytes(raw)))
    case = tmp_path / "test_leak_probe.py"
    case.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(pathlib.Path(dl.__file__).parent)!r})\n"
        "import deploylib as dl\n"
        f"RAW = bytes.fromhex({raw.hex()!r})\n"
        "import pytest\n"
        "@pytest.fixture\n"
        "def cred():\n    return dl.Secret(RAW)\n"
        "def helper(secret):\n    raise RuntimeError('boom')\n"
        "def test_plain_assert():\n    assert RAW != RAW, 'same'\n"
        "def test_plain_arg_in_last_frame():\n    helper(RAW)\n"
        "def test_secret_assert(cred):\n    assert cred != cred, 'same'\n"
        "def test_secret_arg_in_last_frame(cred):\n    helper(cred)\n"
        "def test_secret_fixture_arg_last_frame(cred):\n    raise RuntimeError('boom')\n"
        "def test_helper(cred):\n    dl.assert_secrets_differ(cred, cred, 'same')\n"
    )

    def run(style):
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                f"--tb={style}",
                str(case),
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        ).stdout
        sections = {}
        for part in out.split("\n____"):
            for name in (
                "plain_assert",
                "plain_arg",
                "secret_assert",
                "secret_arg",
                "secret_fixture",
                "helper",
            ):
                if part.lstrip("_ ").startswith("test_" + name):
                    sections[name] = part
        return sections

    for style in ("long", "auto", "short", "native"):
        s = run(style)
        assert set(s) == {
            "plain_assert",
            "plain_arg",
            "secret_assert",
            "secret_arg",
            "secret_fixture",
            "helper",
        }
        for name in ("secret_assert", "secret_arg", "secret_fixture", "helper"):
            assert not any(f in s[name] for f in forms), (style, name)
        if style == "long":  # control: the leak is real for plain bytes
            assert any(f in s["plain_assert"] for f in forms)
            assert any(f in s["plain_arg"] for f in forms)
    assert not any(f in str(dl.Secret(raw)) + repr(dl.Secret(raw)) for f in forms)
    dl.assert_secrets_differ(dl.Secret(raw), os.urandom(32), "distinct secrets are distinct")
    with pytest.raises(AssertionError) as exc:
        dl.assert_secrets_differ(dl.Secret(raw), dl.Secret(raw), "same")
    assert not any(f in str(exc.value) for f in forms)


def test_a_search_that_did_not_finish_is_never_complete(tmp_path):
    needle = b"sk-ant-FAKE-" + b"cd" * 16
    (tmp_path / "a").write_bytes(b"nothing")
    # a missing root and an unreadable root
    r = dl.surface_hits(needle, [str(tmp_path / "absent")])
    assert (r.hits, r.complete) == ([], False) and "absent" in r.problems[0]
    closed = tmp_path / "closed"
    closed.mkdir(mode=0o000)
    try:
        r = dl.surface_hits(needle, [str(tmp_path), str(closed)])
        assert r.complete is False and any("closed" in p for p in r.problems)
    finally:
        closed.chmod(0o700)
    # an exhausted budget
    r = dl.surface_hits(needle, [str(tmp_path)], budget_seconds=-1)
    assert r.complete is False and "budget" in r.problems[0]
    # a hit across a chunk boundary is found
    pad = b"q" * (dl.CHUNK - 7)
    (tmp_path / "split").write_bytes(pad + needle + b"tail")
    r = dl.surface_hits(needle, [str(tmp_path)])
    assert str(tmp_path / "split") in r.hits and r.complete


def test_a_very_large_sparse_file_is_scanned_not_skipped_or_reported(tmp_path):
    """No size cap: a 700 MiB sparse file (past the old 512 MiB limit) with the needle at its end
    is found, and the search stays complete."""
    needle = b"sk-ant-FAKE-" + b"ef" * 16
    big = tmp_path / "film.bin"
    with open(big, "wb") as fh:
        fh.truncate(700 << 20)
        fh.seek((700 << 20) - 64)
        fh.write(needle)
    clean = tmp_path / "clean.bin"
    with open(clean, "wb") as fh:
        fh.truncate(700 << 20)
    r = dl.surface_hits(needle, [str(tmp_path)])
    assert (r.hits, r.complete) == ([str(big)], True) and r.files == 2


def test_container_exposure_reports_group_and_sockets_and_nothing_otherwise(tmp_path):
    import grp
    import socket

    gid = os.getgid() + 12345
    docker = grp.struct_group(("docker", "x", gid, []))
    mine = grp.struct_group(("docker", "x", gid, [pwd.getpwuid(os.getuid()).pw_name]))
    none = lambda name: (_ for _ in ()).throw(KeyError(name))  # noqa: E731
    assert dl.container_exposure({}, (), [], lambda n: docker) == []
    assert dl.container_exposure({}, (), [], none) == []
    assert "group docker" in dl.container_exposure({}, (), [gid], lambda n: docker)[0]
    assert "/etc/group" in dl.container_exposure({}, (), [], lambda n: mine)[0]
    p = str(tmp_path / "d.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(p)
    srv.listen(1)
    try:
        assert dl.container_exposure({}, (p,), [], none) == [f"container socket reachable: {p}"]
    finally:
        srv.close()


def test_secret_hex_and_str_are_redacted_but_still_usable():
    s = dl.Secret(bytes(range(32)))
    assert "000102" not in f"{s.hex()} {s.hex()!r} {s!s} {s!r}"
    assert s.hex().encode() == bytes(range(32)).hex().encode()
    assert dl.secret_needle(s) in bytes(range(32)).hex().encode()


def test_journal_search_reports_an_unreadable_journal_as_incomplete(monkeypatch):
    real = dl.subprocess.Popen

    def fake(argv, **kw):
        return real(["/bin/sh", "-c", "echo boom >&2; exit 1"], **kw)

    monkeypatch.setattr(dl.subprocess, "Popen", fake)
    r = dl.journal_search(b"sk-ant-FAKE-x")
    assert r.complete is False and r.hits == []


def test_tree_differences_names_presence_bytes_and_mode_changes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        (root / "bin").mkdir(parents=True)
        (root / "bin" / "k.py").write_bytes(b"x")
        (root / "f").write_bytes(b"y")
    assert dl.tree_differences(str(a), str(b)) == []
    (b / "f").write_bytes(b"z")
    (b / "extra").write_bytes(b"")
    (a / "gone").write_bytes(b"")
    (b / "bin" / "k.py").chmod(0o755)
    assert dl.tree_differences(str(a), str(b)) == [
        "only in the sha: gone",
        "only installed: extra",
        "differs: bin/k.py",
        "differs: f",
    ]
