"""`check-deployment`, `add-caller`, `install-user-file`, `log` (plan §3.4; §5.2 FR-8 (9)).

Ownership by root or by `llmholder` cannot be created without root, so the artifact table is
checked through an injected stat provider (the plan's seam) and the real filesystem is used for
the checks that need no foreign uid: lstat rules, writable paths, planted symlinks. What needs a
real second uid is tier 2 (see the report)."""

import os
import pathlib
import stat
import subprocess

import pytest
from holder_harness import UID, Env, StandIn, make_cert

from llmgateway.keyholder import (
    HolderStartError,
    Layout,
    check_code_root,
    check_deployment,
    main,
    read_log,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT_UID, HOLDER_UID, CALLER_UID = 0, 990, 1000
D, F, S, L = stat.S_IFDIR, stat.S_IFREG, stat.S_IFSOCK, stat.S_IFLNK


class FakeFS:
    def __init__(self, entries):
        self.e = dict(entries)

    def lstat(self, path):
        if path not in self.e:
            raise FileNotFoundError(path)
        mode, uid = self.e[path]
        return os.stat_result((mode, 0, 0, 1, uid, 0, 0, 0, 0, 0))

    def listdir(self, path):
        prefix = path.rstrip("/") + "/"
        return sorted(
            {p[len(prefix) :].split("/")[0] for p in self.e if p.startswith(prefix) and p != path}
        )


def good_layout():
    return Layout(
        code_root="/opt/llmgateway",
        holder_uid=HOLDER_UID,
        caller_creds=[(CALLER_UID, "/home/u/.config/llmgateway/callers/broker.cred")],
        user_units=[(CALLER_UID, "/home/u/.config/systemd/user/llmgateway.service")],
        state_dirs=[(CALLER_UID, "/home/u/.local/state/llmgateway")],
    )


def good_entries():
    e = {}
    for d in (
        "/",
        "/opt",
        "/opt/llmgateway",
        "/opt/llmgateway/src",
        "/opt/llmgateway/src/llmgateway",
        "/opt/llmgateway/bin",
        "/opt/llmgateway/deploy",
    ):
        e[d] = (D | 0o755, ROOT_UID)
    for f in (
        "/opt/llmgateway/src/llmgateway/keyholder.py",
        "/opt/llmgateway/bin/keyholder.py",
        "/opt/llmgateway/deploy/install.sh",
    ):
        e[f] = (F | 0o644, ROOT_UID)
    e["/etc/systemd/system/llmholder.service"] = (F | 0o644, ROOT_UID)
    for d in ("/etc/llmgateway", "/etc/llmgateway/ccg"):
        e[d] = (D | 0o755, ROOT_UID)
    for f in ("/etc/llmgateway/gateway.toml", "/etc/llmgateway/ccg/system.yaml"):
        e[f] = (F | 0o644, ROOT_UID)
    for d in ("/var/lib/llmholder", "/var/lib/llmholder/keys", "/var/lib/llmholder/registry"):
        e[d] = (D | 0o700, HOLDER_UID)
    for f in (
        "/var/lib/llmholder/keys/anthropic.key",
        "/var/lib/llmholder/registry/broker.secret",
        "/var/lib/llmholder/registry/service.secret",
        "/var/lib/llmholder/ledger.sqlite3",
        "/var/lib/llmholder/ledger.sqlite3-wal",
        "/var/lib/llmholder/ledger.sqlite3-shm",
        "/var/lib/llmholder/ledger.sqlite3.lock",
        "/var/lib/llmholder/calls.log",
    ):
        e[f] = (F | 0o600, HOLDER_UID)
    e["/run/llmholder"] = (D | 0o711, HOLDER_UID)
    e["/run/llmholder/holder.sock"] = (S | 0o666, HOLDER_UID)
    e["/home/u/.config/llmgateway/callers/broker.cred"] = (F | 0o600, CALLER_UID)
    e["/home/u/.config/systemd/user/llmgateway.service"] = (F | 0o644, CALLER_UID)
    e["/home/u/.local/state/llmgateway"] = (D | 0o700, CALLER_UID)
    return e


def run_check(entries):
    return check_deployment(good_layout(), FakeFS(entries))


def test_a_correct_layout_passes():
    assert run_check(good_entries()) == []


MUTATIONS = {
    "socket dir 0700 (callers could not connect)": ("/run/llmholder", (D | 0o700, HOLDER_UID)),
    "socket dir owned by the caller (fake holder)": ("/run/llmholder", (D | 0o711, CALLER_UID)),
    "socket 0660": ("/run/llmholder/holder.sock", (S | 0o660, HOLDER_UID)),
    "socket is a regular file": ("/run/llmholder/holder.sock", (F | 0o666, HOLDER_UID)),
    "credential owned by llmholder": (
        "/home/u/.config/llmgateway/callers/broker.cred",
        (F | 0o600, HOLDER_UID),
    ),
    "credential 0644": ("/home/u/.config/llmgateway/callers/broker.cred", (F | 0o644, CALLER_UID)),
    "credential is a symlink": (
        "/home/u/.config/llmgateway/callers/broker.cred",
        (L | 0o777, CALLER_UID),
    ),
    "key 0640": ("/var/lib/llmholder/keys/anthropic.key", (F | 0o640, HOLDER_UID)),
    "key owned by the caller": ("/var/lib/llmholder/keys/anthropic.key", (F | 0o600, CALLER_UID)),
    "registry secret group-readable": (
        "/var/lib/llmholder/registry/broker.secret",
        (F | 0o640, HOLDER_UID),
    ),
    "state dir 0755": ("/var/lib/llmholder", (D | 0o755, HOLDER_UID)),
    "ledger world-readable": ("/var/lib/llmholder/ledger.sqlite3", (F | 0o644, HOLDER_UID)),
    "calls.log owned by root": ("/var/lib/llmholder/calls.log", (F | 0o600, ROOT_UID)),
    "unit file writable by the caller": (
        "/etc/systemd/system/llmholder.service",
        (F | 0o644, CALLER_UID),
    ),
    "gateway.toml group-writable": ("/etc/llmgateway/gateway.toml", (F | 0o664, ROOT_UID)),
    "manifest copy owned by the caller": (
        "/etc/llmgateway/ccg/system.yaml",
        (F | 0o644, CALLER_UID),
    ),
    "manifest copy is a symlink": ("/etc/llmgateway/ccg/system.yaml", (L | 0o777, ROOT_UID)),
    "etc dir 0775": ("/etc/llmgateway", (D | 0o775, ROOT_UID)),
    "caller state dir 0755": ("/home/u/.local/state/llmgateway", (D | 0o755, CALLER_UID)),
    "user unit is a fifo": (
        "/home/u/.config/systemd/user/llmgateway.service",
        (stat.S_IFIFO | 0o644, CALLER_UID),
    ),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_planted_wrong_owner_or_mode_fails(name):
    path, value = MUTATIONS[name]
    entries = good_entries()
    entries[path] = value
    problems = run_check(entries)
    assert any(p.startswith(path) for p in problems), (name, problems)


@pytest.mark.parametrize(
    "path",
    [
        "/run/llmholder",
        "/run/llmholder/holder.sock",
        "/var/lib/llmholder/keys",
        "/etc/systemd/system/llmholder.service",
        "/home/u/.config/llmgateway/callers/broker.cred",
        "/var/lib/llmholder/registry",
        "/etc/llmgateway/gateway.toml",
        "/etc/llmgateway/ccg/system.yaml",
    ],
)
def test_an_absent_required_artifact_is_flagged(path):
    entries = good_entries()
    del entries[path]
    assert any(p == f"{path}: missing" for p in run_check(entries))


def test_optional_ledger_files_may_be_absent_before_first_start():
    entries = good_entries()
    for n in ("ledger.sqlite3", "ledger.sqlite3-wal", "ledger.sqlite3-shm", "ledger.sqlite3.lock"):
        del entries["/var/lib/llmholder/" + n]
    assert run_check(entries) == []


CODE_MUTATIONS = {
    "code file writable by a non-root uid": (
        "/opt/llmgateway/src/llmgateway/keyholder.py",
        (F | 0o644, CALLER_UID),
    ),
    "code file group-writable": ("/opt/llmgateway/bin/keyholder.py", (F | 0o664, ROOT_UID)),
    "code dir world-writable": ("/opt/llmgateway/src", (D | 0o777, ROOT_UID)),
    "code root owned by the caller": ("/opt/llmgateway", (D | 0o755, CALLER_UID)),
    "ancestor writable by the caller": ("/opt", (D | 0o775, ROOT_UID)),
    "ancestor owned by the caller": ("/opt", (D | 0o755, CALLER_UID)),
    "root dir sticky world-writable": ("/", (D | 0o777, ROOT_UID)),
    "symlink inside the code root": (
        "/opt/llmgateway/src/llmgateway/keyholder.py",
        (L | 0o777, ROOT_UID),
    ),
    "symlinked directory in the code root": ("/opt/llmgateway/deploy", (L | 0o777, ROOT_UID)),
    "fifo in the code root": (
        "/opt/llmgateway/deploy/install.sh",
        (stat.S_IFIFO | 0o644, ROOT_UID),
    ),
    "socket in the code root": ("/opt/llmgateway/deploy/install.sh", (S | 0o644, ROOT_UID)),
    "block device in the code root": (
        "/opt/llmgateway/deploy/install.sh",
        (stat.S_IFBLK | 0o644, ROOT_UID),
    ),
    "symlinked ancestor": ("/opt", (L | 0o777, ROOT_UID)),
}


@pytest.mark.parametrize("name", sorted(CODE_MUTATIONS))
def test_code_root_rules_each_fail_even_in_code_only_mode(name):
    path, value = CODE_MUTATIONS[name]
    entries = good_entries()
    entries[path] = value
    problems = check_deployment(good_layout(), FakeFS(entries), code_only=True)
    assert problems, name


def test_a_new_root_owned_file_passes_but_one_owned_by_the_caller_fails():
    entries = good_entries()
    entries["/opt/llmgateway/src/llmgateway/extra.py"] = (F | 0o644, ROOT_UID)
    assert check_deployment(good_layout(), FakeFS(entries), code_only=True) == []
    entries["/opt/llmgateway/src/llmgateway/extra.py"] = (F | 0o644, CALLER_UID)
    assert check_deployment(good_layout(), FakeFS(entries), code_only=True)


# --- real filesystem ---


def real_tree(tmp_path):
    root = tmp_path / "opt" / "llmgateway"
    (root / "src").mkdir(parents=True)
    (root / "src" / "m.py").write_text("x = 1\n")
    for d in (tmp_path, tmp_path / "opt", root, root / "src"):
        d.chmod(0o755)
    (root / "src" / "m.py").chmod(0o644)
    return root


def test_real_code_root_checks_lstat_and_writability(tmp_path):
    root = real_tree(tmp_path)
    kw = dict(trusted_uids={UID}, anchor=str(tmp_path))
    assert check_code_root(str(root), **kw) == []
    (root / "src" / "m.py").chmod(0o666)
    assert any("writable" in p for p in check_code_root(str(root), **kw))
    (root / "src" / "m.py").chmod(0o644)
    (root / "src" / "link.py").symlink_to("/etc/hostname")
    assert any("not a regular file" in p for p in check_code_root(str(root), **kw))
    (root / "src" / "link.py").unlink()
    os.mkfifo(root / "src" / "pipe")
    assert any("not a regular file" in p for p in check_code_root(str(root), **kw))
    os.unlink(root / "src" / "pipe")
    assert check_code_root(
        str(root), trusted_uids={UID + 1}, anchor=str(tmp_path)
    )  # not owned by a trusted uid


def test_real_symlinked_ancestor_is_refused(tmp_path):
    real = real_tree(tmp_path)
    link = tmp_path / "via"
    link.symlink_to(tmp_path / "opt")
    problems = check_code_root(str(link / "llmgateway"), trusted_uids={UID}, anchor=str(tmp_path))
    assert any("ancestor is not a plain directory" in p for p in problems)
    assert real.exists()


def test_a_writable_code_root_also_stops_the_holder(tmp_path_factory, tmp_path):
    cert = make_cert(tmp_path_factory.mktemp("tls"))
    standin = StandIn(*cert)
    env = Env(tmp_path / "root", standin, cert[0])
    code = real_tree(tmp_path)
    try:
        env.holder(
            code_root_check=lambda: check_code_root(
                str(code), trusted_uids={UID}, anchor=str(tmp_path)
            )
        ).close()
        (code / "src" / "m.py").chmod(0o666)
        with pytest.raises(HolderStartError, match="code root"):
            env.holder(
                code_root_check=lambda: check_code_root(
                    str(code), trusted_uids={UID}, anchor=str(tmp_path)
                )
            )
    finally:
        standin.close()


def test_real_planted_symlink_credential_is_not_a_regular_file(tmp_path):
    cred = tmp_path / "cred"
    cred.write_text("x")
    cred.chmod(0o600)
    link = tmp_path / "broker.cred"
    link.symlink_to(cred)
    layout = Layout(
        code_root=str(real_tree(tmp_path)),
        holder_uid=UID,
        trusted_uids=frozenset({UID}),
        anchor=str(tmp_path),
        caller_creds=[(UID, str(link))],
        etc_dir=str(tmp_path / "e"),
        unit_path=str(tmp_path / "u"),
        state_root=str(tmp_path / "s"),
        run_dir=str(tmp_path / "r"),
        socket_path=str(tmp_path / "r" / "s"),
    )
    problems = check_deployment(layout)
    assert f"{link}: not a file" in problems
    assert f"{cred}: not a file" not in problems


# --- add-caller (real subprocess; the caller-uid drop itself is tier 2) ---


SHIM = str(REPO / "bin" / "keyholder.py")


def add_caller(tmp_path, *extra, name="broker", out=None, home=None, state=None):
    home = home or tmp_path / "home"
    state = state or tmp_path / "state"
    out = out or home / ".config" / "llmgateway" / "callers" / f"{name}.cred"
    return subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            SHIM,
            "add-caller",
            name,
            "--caller-uid",
            str(UID),
            "--holder-uid",
            str(UID),
            "--home",
            str(home),
            "--out",
            str(out),
            "--state-root",
            str(state),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def acct(tmp_path):
    (tmp_path / "home").mkdir(mode=0o750)
    (tmp_path / "state" / "registry").mkdir(parents=True, mode=0o700)
    return tmp_path


def cred_path(t, name="broker"):
    return t / "home" / ".config" / "llmgateway" / "callers" / f"{name}.cred"


def test_add_caller_writes_both_copies_with_the_same_secret_and_never_prints_it(acct):
    res = add_caller(acct)
    assert res.returncode == 0, res.stderr
    cred, reg = cred_path(acct), acct / "state" / "registry" / "broker.secret"
    secret = cred.read_text().strip()
    assert len(bytes.fromhex(secret)) == 32 and reg.read_text().strip() == secret
    assert secret not in res.stdout + res.stderr
    for p in (cred, reg):
        st = p.stat()
        assert stat.S_IMODE(st.st_mode) == 0o600 and st.st_uid == UID
    for d in (cred.parent, cred.parent.parent):  # created by the caller, 0700
        assert stat.S_IMODE(d.stat().st_mode) == 0o700 and d.stat().st_uid == UID
    assert not [p for p in cred.parent.iterdir() if p.name.endswith(".tmp")]


def test_add_caller_twice_refuses_and_rotate_replaces_both_copies(acct):
    assert add_caller(acct).returncode == 0
    old = cred_path(acct).read_text()
    again = add_caller(acct)
    assert again.returncode == 1 and "--rotate" in again.stderr
    assert cred_path(acct).read_text() == old
    assert add_caller(acct, "--rotate").returncode == 0
    new = cred_path(acct).read_text()
    assert new != old and (acct / "state" / "registry" / "broker.secret").read_text() == new


@pytest.mark.parametrize(
    "planted", ["home/.config", "home/.config/llmgateway", "home/.config/llmgateway/callers"]
)
def test_a_planted_symlink_at_each_component_is_not_followed(acct, planted):
    decoy = acct / "decoy"
    decoy.mkdir()
    parts = planted.split("/")
    base = acct
    for part in parts[:-1]:
        base = base / part
        base.mkdir(exist_ok=True, mode=0o700)
    (base / parts[-1]).symlink_to(decoy)
    res = add_caller(acct)
    assert res.returncode == 1 and "symlink" in res.stderr
    assert list(decoy.rglob("*")) == []  # nothing was created through the link
    assert not (acct / "state" / "registry" / "broker.secret").exists()  # and no half-done add


def test_a_planted_symlink_as_home_is_not_followed(acct):
    real = acct / "realhome"
    real.mkdir()
    (acct / "linkhome").symlink_to(real)
    res = add_caller(acct, home=acct / "linkhome", out=acct / "linkhome" / "c.cred")
    assert res.returncode == 1 and list(real.iterdir()) == []


def test_rotate_over_a_planted_symlink_replaces_the_link_not_its_target(acct):
    assert add_caller(acct).returncode == 0
    victim = acct / "victim"
    victim.write_text("precious")
    cred = cred_path(acct)
    cred.unlink()
    cred.symlink_to(victim)
    assert add_caller(acct, "--rotate").returncode == 0
    assert victim.read_text() == "precious"
    assert not cred.is_symlink() and len(bytes.fromhex(cred.read_text().strip())) == 32


def test_a_directory_not_owned_by_the_caller_is_refused(acct):
    res = add_caller(acct, home=pathlib.Path("/usr"), out=pathlib.Path("/usr/share/llmgw/c.cred"))
    assert res.returncode == 1 and "not owned" in res.stderr
    assert not pathlib.Path("/usr/share/llmgw").exists()


@pytest.mark.parametrize("out", ["relative/c.cred", "/etc/c.cred", "{h}/../x.cred", "{h}"])
def test_out_must_be_inside_the_callers_home(acct, out):
    res = add_caller(acct, out=out.format(h=acct / "home"))
    assert res.returncode == 1 and "refused" in res.stderr


def test_bad_caller_names_and_a_missing_registry_are_refused(acct):
    for bad in ("../x", "A", "a b", "x" * 40):
        assert add_caller(acct, name=bad, out=acct / "home" / "c.cred").returncode == 1
    res = add_caller(acct, state=acct / "nostate")
    assert res.returncode == 1 and not cred_path(acct).exists()  # no orphan credential


def test_install_user_file_writes_stdin_as_the_caller(acct):
    out = acct / "home" / ".config" / "systemd" / "user" / "llmgateway.service"
    res = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            SHIM,
            "install-user-file",
            "--caller-uid",
            str(UID),
            "--home",
            str(acct / "home"),
            "--out",
            str(out),
        ],
        input=b"[Unit]\nDescription=x\n",
        capture_output=True,
        timeout=30,
    )
    assert res.returncode == 0, res.stderr
    assert (
        out.read_bytes() == b"[Unit]\nDescription=x\n" and stat.S_IMODE(out.stat().st_mode) == 0o644
    )


# --- log ---


def test_log_prints_calls_log_read_only_and_refuses_a_symlink(tmp_path, capsys):
    (tmp_path / "calls.log").write_text('{"ts":1}\n')
    assert read_log(str(tmp_path)) == '{"ts":1}\n'
    assert main(["log"], state_root=str(tmp_path)) == 0
    assert capsys.readouterr().out == '{"ts":1}\n'
    (tmp_path / "calls.log").unlink()
    (tmp_path / "calls.log").symlink_to("/etc/hostname")
    assert main(["log"], state_root=str(tmp_path)) == 1


def test_log_takes_no_options():
    with pytest.raises(SystemExit):
        main(["log", "--state-root", "/tmp"])


def test_the_cli_check_deployment_reports_and_exits_nonzero(tmp_path):
    root = real_tree(tmp_path)
    (root / "src" / "m.py").chmod(0o666)
    res = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            SHIM,
            "check-deployment",
            "--code-only",
            "--root",
            str(root),
            "--anchor",
            str(tmp_path),
            "--trusted-uid",
            str(UID),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 1 and "m.py: group/other-writable" in res.stdout
