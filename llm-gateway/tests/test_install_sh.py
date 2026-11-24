"""`deploy/install.sh` where it can run unprivileged (plan §3.4, §5.2 FR-8 (9),(9b)).

Everything here runs the real script in its `--test-root` mode (refused as root; no useradd,
chown, systemctl). What needs real root, a second uid or systemd is tier 2 and listed in the report.
"""

import hashlib
import os
import pathlib
import shutil
import socket
import stat
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
UID = os.getuid()
BASH = "/bin/bash"


def copy_tree(dest: pathlib.Path):
    for sub in ("src", "bin", "deploy"):
        shutil.copytree(
            REPO / sub, dest / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for dirpath, _dirs, files in os.walk(dest):
        os.chmod(dirpath, 0o755)
        for f in files:
            os.chmod(os.path.join(dirpath, f), 0o755 if f.endswith((".sh",)) else 0o644)


@pytest.fixture
def tr(tmp_path):
    """A test root with the code root installed under it (a copy of this tree)."""
    root = tmp_path / "tr"
    root.mkdir(mode=0o755)
    code = root / "opt" / "llmgateway"
    code.mkdir(parents=True)
    copy_tree(code)
    for d in (root, root / "opt"):
        d.chmod(0o755)
    (tmp_path / "home").mkdir(mode=0o750)
    return root


def sh(tr, *args, script=None, stdin="", flags=None, cwd="/"):
    script = script or tr / "opt" / "llmgateway" / "deploy" / "install.sh"
    flags = (
        ["--test-root", str(tr), "--test-home", str(tr.parent / "home")] if flags is None else flags
    )
    return subprocess.run(
        [BASH, str(script), *flags, *args],
        input=stdin,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        cwd=cwd,
        env={"PATH": "/usr/bin:/bin"},
    )


def test_script_is_valid_bash():
    assert subprocess.run([BASH, "-n", str(REPO / "deploy" / "install.sh")]).returncode == 0


def test_usage_and_argument_handling(tr):
    assert sh(tr).returncode == 2
    assert sh(tr, "bogus").returncode == 2
    assert sh(tr, "holder-account", "--nope").returncode == 2
    assert sh(tr, "caller", "only-a-name").returncode == 2
    assert "provider name" in sh(tr, "holder-account", "--key-from-stdin", "Bad Name").stderr
    assert "caller name" in sh(tr, "caller", "../x", str(UID), "/p").stderr
    assert "uid" in sh(tr, "caller", "broker", "12x", "/p").stderr
    assert "absolute" in sh(tr, "caller", "broker", str(UID), "rel").stderr
    assert sh(tr, "owner-files", "--gateway-toml", "/x").returncode == 2
    assert (
        "must be run as root"
        in subprocess.run(
            [BASH, str(tr / "opt/llmgateway/deploy/install.sh"), "holder-account"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        ).stderr
    )  # production mode, unprivileged
    assert (
        "needs --test-root"
        in subprocess.run(
            [BASH, str(tr / "opt/llmgateway/deploy/install.sh"), "--test-home", "/tmp", "check"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        ).stderr
    )


def test_dry_run_changes_nothing_and_prints_the_plan(tr):
    before = sorted(str(p) for p in tr.rglob("*"))
    res = sh(tr, "holder-account", flags=["--dry-run", "--test-root", str(tr)])
    assert res.returncode == 0 and "DRY-RUN:" in res.stdout
    assert "/var/lib/llmholder" in res.stdout
    assert sorted(str(p) for p in tr.rglob("*")) == before
    res = sh(
        tr,
        "caller",
        "broker",
        str(UID),
        str(tr.parent / "home" / "c.cred"),
        flags=["--dry-run", "--test-root", str(tr)],
    )
    assert (
        res.returncode == 0
        and "add-caller" in res.stdout
        and not (tr.parent / "home" / "c.cred").exists()
    )


# --- the modified-script cases ---


def test_a_copy_outside_the_code_root_refuses_to_run(tr, tmp_path):
    elsewhere = tmp_path / "checkout" / "llm-gateway" / "deploy"
    elsewhere.mkdir(parents=True)
    shutil.copy(REPO / "deploy" / "install.sh", elsewhere / "install.sh")
    res = sh(tr, "holder-account", script=elsewhere / "install.sh")
    assert res.returncode == 1 and "not running from the installed code root" in res.stderr
    assert not (tr / "var").exists()


def test_a_symlinked_invocation_refuses(tr, tmp_path):
    link = tmp_path / "innocent.sh"
    link.symlink_to(tr / "opt/llmgateway/deploy/install.sh")
    res = sh(tr, "holder-account", script=link)
    assert res.returncode == 1 and "symlink" in res.stderr and not (tr / "var").exists()


def test_a_tree_the_agent_could_write_refuses(tr):
    code = tr / "opt" / "llmgateway"
    script = code / "deploy" / "install.sh"
    script.chmod(
        0o775
    )  # group-writable (the modified-script case: anyone in the group can edit it)
    res = sh(tr, "holder-account")
    assert res.returncode == 1 and "not protected" in res.stderr and "install.sh" in res.stderr
    script.chmod(0o755)
    (code / "src" / "llmgateway" / "keyholder.py").chmod(0o666)
    assert "not protected" in sh(tr, "holder-account").stderr
    (code / "src" / "llmgateway" / "keyholder.py").chmod(0o644)
    code.chmod(0o777)
    assert "not protected" in sh(tr, "holder-account").stderr
    assert not (tr / "var").exists()


def test_a_symlink_planted_in_the_code_root_refuses(tr):
    code = tr / "opt" / "llmgateway"
    (code / "src" / "llmgateway" / "extra.py").symlink_to("/etc/hostname")
    res = sh(tr, "holder-account")
    assert res.returncode == 1 and "not a regular file" in res.stderr and not (tr / "var").exists()


def test_a_replaced_entry_script_that_is_a_symlink_refuses(tr, tmp_path):
    code = tr / "opt" / "llmgateway"
    (tmp_path / "evil.py").write_text("print('x')")
    (code / "bin" / "keyholder.py").unlink()
    (code / "bin" / "keyholder.py").symlink_to(tmp_path / "evil.py")
    res = sh(tr, "holder-account")
    assert res.returncode == 1 and "symlink" in res.stderr


# --- the real steps in test mode ---


def modes(path):
    st = os.lstat(path)
    return stat.S_IMODE(st.st_mode), st.st_uid


def test_holder_account_builds_the_state_tree_unit_and_key_without_echoing_it(tr):
    key = "sk-ant-FAKE-installer-marker-0001"
    res = sh(tr, "holder-account", "--key-from-stdin", "anthropic", stdin=key + "\n")
    assert res.returncode == 0, res.stderr
    assert key not in res.stdout + res.stderr
    state = tr / "var" / "lib" / "llmholder"
    assert modes(state)[0] == 0o700
    assert modes(state / "keys")[0] == 0o700 and modes(state / "registry")[0] == 0o700
    kf = state / "keys" / "anthropic.key"
    assert kf.read_text() == key + "\n" and modes(kf)[0] == 0o600
    unit = (tr / "etc/systemd/system/llmholder.service").read_text()
    assert modes(tr / "etc/systemd/system/llmholder.service")[0] == 0o644
    for line in (
        "User=llmholder",
        "RuntimeDirectory=llmholder",
        "RuntimeDirectoryMode=0711",
        "ExecStart=/usr/bin/python3 -I /opt/llmgateway/bin/keyholder.py",
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
        "ReadWritePaths=/var/lib/llmholder",
        "PrivateTmp=yes",
        "LimitCORE=0",
        "ProtectProc=invisible",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "CapabilityBoundingSet=",
    ):
        assert line in unit.splitlines(), line
    assert "sudoers" not in unit


@pytest.mark.parametrize("bad", ["", "\n", "has space\n", "two\nlines\n", "x" * 5000 + "\n"])
def test_a_bad_key_on_stdin_is_refused_and_not_stored(tr, bad):
    res = sh(tr, "holder-account", "--key-from-stdin", "anthropic", stdin=bad)
    assert res.returncode == 1 and "one non-empty printable token" in res.stderr
    assert not (tr / "var/lib/llmholder/keys/anthropic.key").exists()
    assert "x" * 50 not in res.stdout + res.stderr


def test_owner_files_shows_a_diff_and_installs_exactly_the_bytes_it_read(tr, tmp_path):
    toml, manifest = tmp_path / "g.toml", tmp_path / "system.yaml"
    toml.write_bytes(b"[holder]\nwindow_seconds = 60\n")
    manifest.write_bytes(b"systems: []\n\x00\xff binary tail\n")
    refused = sh(
        tr, "owner-files", "--gateway-toml", str(toml), "--manifest", str(manifest), stdin="n\n"
    )
    assert refused.returncode == 1 and "not confirmed" in refused.stderr
    assert not (tr / "etc/llmgateway/gateway.toml").exists()
    ok = sh(
        tr, "owner-files", "--gateway-toml", str(toml), "--manifest", str(manifest), stdin="y\ny\n"
    )
    assert ok.returncode == 0, ok.stderr
    for src, dest in (
        (toml, "etc/llmgateway/gateway.toml"),
        (manifest, "etc/llmgateway/ccg/system.yaml"),
    ):
        assert (tr / dest).read_bytes() == src.read_bytes()
        assert modes(tr / dest)[0] == 0o644
    assert modes(tr / "etc/llmgateway")[0] == 0o755 and modes(tr / "etc/llmgateway/ccg")[0] == 0o755
    toml.write_bytes(b"[holder]\nwindow_seconds = 99\n")
    again = sh(
        tr, "owner-files", "--gateway-toml", str(toml), "--manifest", str(manifest), stdin="y\ny\n"
    )
    assert "-window_seconds = 60" in again.stdout and "+window_seconds = 99" in again.stdout


def test_owner_files_refuses_a_symlinked_source(tr, tmp_path):
    real = tmp_path / "real.toml"
    real.write_text("x")
    link = tmp_path / "link.toml"
    link.symlink_to(real)
    res = sh(
        tr, "owner-files", "--gateway-toml", str(link), "--manifest", str(real), stdin="y\ny\n"
    )
    assert res.returncode == 1 and "symlink" in res.stderr


def test_caller_and_service_unit_end_to_end_then_check_passes_and_each_break_is_caught(
    tr, tmp_path
):
    home = tmp_path / "home"
    assert sh(tr, "holder-account").returncode == 0
    out = home / ".config/llmgateway/callers/broker.cred"
    res = sh(tr, "caller", "broker", str(UID), str(out))
    assert res.returncode == 0, res.stderr
    secret = out.read_text().strip()
    assert secret not in res.stdout + res.stderr
    assert (tr / "var/lib/llmholder/registry/broker.secret").read_text().strip() == secret
    unit = home / ".config/systemd/user/llmgateway.service"
    (tr / "opt/llmgateway/bin/service.py").write_text("# the D8 shim\n")
    (tr / "opt/llmgateway/bin/service.py").chmod(0o644)
    assert sh(tr, "service-unit", str(UID), str(home)).returncode == 0
    assert "ExecStart=/usr/bin/python3 -I /opt/llmgateway/" in unit.read_text()

    # the installed layout, as the human would verify it
    for name, mode in (("etc", 0o755),):
        (tr / name).chmod(mode)
    (tr / "etc/llmgateway").mkdir(exist_ok=True, mode=0o755)
    (tr / "etc/llmgateway/ccg").mkdir(exist_ok=True, mode=0o755)
    for f in ("gateway.toml", "ccg/system.yaml"):
        (tr / "etc/llmgateway" / f).write_text("x")
        (tr / "etc/llmgateway" / f).chmod(0o644)
    run = tr / "run" / "llmholder"
    run.mkdir(parents=True)
    run.chmod(0o711)
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(str(run / "holder.sock"))
    (run / "holder.sock").chmod(0o666)
    (tr / "var/lib/llmholder/calls.log").write_text("")  # a started holder creates it
    (tr / "var/lib/llmholder/calls.log").chmod(0o600)
    cred_args = ["--caller-cred", f"{UID}:{out}", "--user-unit", f"{UID}:{unit}"]
    ok = sh(tr, "check", *cred_args)
    assert ok.returncode == 0 and ok.stdout == "", ok.stdout + ok.stderr

    breaks = {
        "socket dir 0700": lambda: run.chmod(0o700),
        "credential 0644": lambda: out.chmod(0o644),
        "key-tree mode": lambda: (tr / "var/lib/llmholder").chmod(0o755),
        "unit writable": lambda: (tr / "etc/systemd/system/llmholder.service").chmod(0o666),
    }
    restore = {
        "socket dir 0700": lambda: run.chmod(0o711),
        "credential 0644": lambda: out.chmod(0o600),
        "key-tree mode": lambda: (tr / "var/lib/llmholder").chmod(0o700),
        "unit writable": lambda: (tr / "etc/systemd/system/llmholder.service").chmod(0o644),
    }
    for name, do in breaks.items():
        do()
        bad = sh(tr, "check", *cred_args)
        assert bad.returncode == 1 and bad.stdout, name
        restore[name]()
    assert sh(tr, "check", *cred_args).returncode == 0
    out.unlink()
    out.symlink_to(home / "elsewhere")
    assert "not a file" in sh(tr, "check", *cred_args).stdout  # a symlinked credential
    srv.close()


def test_caller_refuses_planted_symlinks_at_each_component_through_the_installer(tr, tmp_path):
    home = tmp_path / "home"
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    assert sh(tr, "holder-account").returncode == 0
    (home / ".config").mkdir(mode=0o700)
    (home / ".config" / "llmgateway").symlink_to(decoy)
    res = sh(tr, "caller", "broker", str(UID), str(home / ".config/llmgateway/callers/broker.cred"))
    assert res.returncode == 1 and "symlink" in res.stderr
    assert (
        list(decoy.rglob("*")) == []
        and not (tr / "var/lib/llmholder/registry/broker.secret").exists()
    )


# --- the bootstrap pipeline (§3.4, §5.2 FR-8 (9b)) ---


def git(cwd, *args, env=None):
    base = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(cwd),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={**base, **(env or {})},
    ).stdout


def test_bootstrap_pipeline_extracts_exactly_the_committed_tree_despite_a_hostile_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "llm-gateway").mkdir(parents=True)
    git(repo, "init", "-q")
    for sub in ("src", "bin", "deploy"):
        shutil.copytree(
            REPO / sub,
            repo / "llm-gateway" / sub,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    (repo / "llm-gateway" / "link-out").symlink_to(
        "/etc/hostname"
    )  # a tracked symlink survives tar
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "reviewed")
    sha = git(repo, "rev-parse", "HEAD").strip()
    committed = {
        line.split("\t")[1]: line.split()[2]
        for line in git(repo, "ls-tree", "-r", sha, "llm-gateway").splitlines()
    }

    marker = tmp_path / "PWNED"
    hook = tmp_path / "hook.sh"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    smudge = tmp_path / "smudge.sh"  # (a `;` would start a comment in .git/config)
    smudge.write_text(f"#!/bin/sh\ntouch {marker}\ncat\n")
    smudge.chmod(0o755)
    cfg = repo / ".git" / "config"
    cfg.write_text(
        cfg.read_text() + f"[core]\n\tfsmonitor = {hook}\n\thooksPath = {tmp_path}\n"
        f'[filter "x"]\n\tsmudge = {smudge}\n\tclean = cat\n'
    )
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text("* filter=x\n")
    (repo / "llm-gateway" / "src" / "llmgateway" / "evil.pyc").write_bytes(b"\0\0")
    (repo / "llm-gateway" / "src" / "llmgateway" / "x.pth").write_text("import os\n")
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "llm-gateway" / "ignored.txt").write_text("ignored")
    script = repo / "llm-gateway" / "deploy" / "install.sh"
    script.write_text(script.read_text() + "\necho MODIFIED-WORKING-TREE\n")  # not committed

    scratch = tmp_path / "scratch.git"
    dest = tmp_path / "opt-llmgateway.new"
    dest.mkdir()
    clean_env = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    git(tmp_path, "clone", "-q", "--no-local", "--bare", str(repo), str(scratch), env=clean_env)
    archive = subprocess.run(
        ["git", "-C", str(scratch), "archive", sha, "llm-gateway"],
        capture_output=True,
        check=True,
        env={"PATH": "/usr/bin:/bin", **clean_env},
    ).stdout
    subprocess.run(
        ["tar", "-x", "--no-same-owner", "--strip-components=1", "-C", str(dest)],
        input=archive,
        check=True,
    )

    assert not marker.exists()  # no fsmonitor, hook or smudge filter ran
    extracted = {}
    for p in sorted(dest.rglob("*")):
        rel = "llm-gateway/" + str(p.relative_to(dest))
        if p.is_symlink():
            extracted[rel] = "link"
        elif p.is_file():
            extracted[rel] = hashlib.sha1(
                b"blob %d\0" % p.stat().st_size + p.read_bytes()
            ).hexdigest()
    for path, blob in committed.items():
        if path.endswith("link-out"):
            assert extracted[path] == "link"
        else:
            assert extracted[path] == blob, path  # byte for byte the committed content
    assert not {k for k in extracted if k.endswith((".pyc", ".pth", "ignored.txt"))}
    assert "MODIFIED-WORKING-TREE" not in (dest / "deploy" / "install.sh").read_text()
    # tracked symlinks survive extraction; the documented `find` check catches them before the move
    found = subprocess.run(
        ["find", str(dest), "!", "-type", "f", "!", "-type", "d"], capture_output=True, text=True
    ).stdout.split()
    assert found == [str(dest / "link-out")]
    (dest / "link-out").unlink()
    assert (
        subprocess.run(
            ["find", str(dest), "!", "-type", "f", "!", "-type", "d"],
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )

    # the unit's command runs from the extracted tree, with -I, from an unrelated cwd
    res = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            str(dest / "bin" / "keyholder.py"),
            "check-deployment",
            "--code-only",
            "--root",
            str(dest),
            "--anchor",
            str(tmp_path),
            "--trusted-uid",
            str(UID),
        ],
        capture_output=True,
        text=True,
        cwd="/",
        timeout=30,
    )
    assert "Traceback" not in res.stderr and res.returncode in (0, 1)

    # positive control: the unscrubbed in-place `git archive` of the same repo DOES run the filter
    marker.unlink(missing_ok=True)
    subprocess.run(
        ["git", "-C", str(repo), "archive", sha, "llm-gateway"],
        capture_output=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert marker.exists(), "the planted filter must be live, or this test proves nothing"


# --- review P5-2 .. P5-5 ---


def run_owner_files(tr, toml, manifest, *, answers="y\ny\n", extra_env=None, flags=None):
    return sh(
        tr, "owner-files", "--gateway-toml", str(toml), "--manifest", str(manifest), stdin=answers
    )


def test_owner_files_shows_the_whole_file_visibly_with_its_hash_and_installs_those_bytes(
    tr, tmp_path
):
    toml, manifest = tmp_path / "g.toml", tmp_path / "system.yaml"
    tail = "local_endpoints = [\"http://127.0.0.1:1\"]  # HIDDEN-TAIL"
    body = (b"# pad\n" * 3000) + tail.encode() + b"\n\x1b[2J\x1b]0;evil\x07\r\n"
    assert len(body) > 4000
    toml.write_bytes(body)
    manifest.write_bytes(b"systems: []\n")
    res = run_owner_files(tr, toml, manifest)
    assert res.returncode == 0, res.stderr
    assert tail in res.stdout  # beyond the old 4000-byte preview
    assert "\x1b" not in res.stdout and "\x07" not in res.stdout and "\r" not in res.stdout
    assert "^[[2J" in res.stdout and "^M" in res.stdout
    assert hashlib.sha256(body).hexdigest() in res.stdout
    assert (tr / "etc/llmgateway/gateway.toml").read_bytes() == body


def test_owner_files_diff_against_an_installed_file_is_control_byte_safe(tr, tmp_path):
    toml, manifest = tmp_path / "g.toml", tmp_path / "system.yaml"
    manifest.write_bytes(b"systems: []\n")
    toml.write_bytes(b"a = 1\n")
    assert run_owner_files(tr, toml, manifest).returncode == 0
    toml.write_bytes(b"a = 2\x1b[31m\n")
    res = run_owner_files(tr, toml, manifest)
    assert "\x1b" not in res.stdout and "+a = 2^[[31m" in res.stdout


def test_owner_files_refuses_an_oversized_source_and_installs_nothing(tr, tmp_path):
    toml, manifest = tmp_path / "g.toml", tmp_path / "system.yaml"
    toml.write_bytes(b"x" * 65537)
    manifest.write_bytes(b"systems: []\n")
    res = run_owner_files(tr, toml, manifest)
    assert res.returncode == 1 and "larger than" in res.stderr
    assert not (tr / "etc/llmgateway/gateway.toml").exists()
    toml.write_bytes(b"x" * 65536)  # exactly the cap is accepted
    assert run_owner_files(tr, toml, manifest).returncode == 0


def test_owner_files_installs_nothing_if_the_scratch_bytes_change_after_they_were_shown(
    tr, tmp_path
):
    toml, manifest = tmp_path / "g.toml", tmp_path / "system.yaml"
    toml.write_bytes(b"a = 1\n")
    manifest.write_bytes(b"systems: []\n")
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir(mode=0o700)
    script = tr / "opt" / "llmgateway" / "deploy" / "install.sh"
    proc = subprocess.Popen(
        [BASH, str(script), "--test-root", str(tr), "owner-files", "--gateway-toml", str(toml),
         "--manifest", str(manifest)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch_dir)},
    )
    seen = ""
    while "--- end (sha256" not in seen:
        ch = proc.stdout.read(1)
        assert ch, proc.stderr.read()
        seen += ch
    (scratched,) = scratch_dir.glob("*/gateway.toml")
    scratched.write_bytes(b"a = 1\nlocal_endpoints = ['http://evil']\n")  # a swap after display
    proc.stdin.write("y\n")
    proc.stdin.close()  # EOF: a second confirmation (the manifest) would be refused
    err = proc.stderr.read()
    assert proc.wait(30) == 1 and "changed after they were shown" in err
    assert not (tr / "etc/llmgateway/gateway.toml").exists()


def test_the_source_is_read_as_the_invoking_user_in_production_mode():
    text = (REPO / "deploy" / "install.sh").read_text()
    assert 'runuser -u "$SUDO_USER" -- head -c' in text
    assert "cat -- \"$src\"" not in text and "head -c 4000" not in text


def test_a_code_root_owned_or_writable_by_another_is_refused_before_any_python_runs(tr, tmp_path):
    code = tr / "opt" / "llmgateway"
    marker = tmp_path / "PYTHON-RAN"
    # the planted code would create the marker if it ran (as root, in production)
    (code / "src" / "llmgateway" / "keyholder.py").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('x')\n"
    )
    (code / "src" / "llmgateway" / "keyholder.py").chmod(0o666)
    res = sh(tr, "holder-account")
    assert res.returncode == 1 and "not protected" in res.stderr and not marker.exists()
    (code / "src" / "llmgateway" / "keyholder.py").chmod(0o644)
    (code / "src" / "llmgateway" / "extra.py").symlink_to("/etc/hostname")
    res = sh(tr, "holder-account")
    assert res.returncode == 1 and "not a regular file" in res.stderr and not marker.exists()
    (code / "src" / "llmgateway" / "extra.py").unlink()
    (code / "src" / "llmgateway" / "keyholder.py").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('x')\n"
    )
    (tr / "opt").chmod(0o777)  # a writable ancestor
    res = sh(tr, "holder-account")
    assert res.returncode == 1 and "ancestor" in res.stderr and not marker.exists()


def test_the_shim_refuses_without_isolated_mode(tr, tmp_path):
    shim = tr / "opt" / "llmgateway" / "bin" / "keyholder.py"
    res = subprocess.run(
        ["/usr/bin/python3", str(shim), "log"], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path)}, cwd="/",
    )
    assert res.returncode != 0 and "-I" in res.stderr and "Traceback" not in res.stderr


def test_service_unit_dry_run_succeeds_with_the_shipped_shim(tr, tmp_path):
    home = tmp_path / "home"
    assert sh(tr, "holder-account").returncode == 0
    dry = sh(tr, "service-unit", str(UID), str(home), flags=["--dry-run", "--test-root", str(tr)])
    assert dry.returncode == 0 and "DRY-RUN" in dry.stdout


def test_service_unit_refuses_while_the_service_shim_is_absent(tr, tmp_path):
    home = tmp_path / "home"
    assert sh(tr, "holder-account").returncode == 0
    (tr / "opt" / "llmgateway" / "bin" / "service.py").unlink()
    res = sh(tr, "service-unit", str(UID), str(home))
    assert res.returncode == 1 and "service.py does not exist" in res.stderr
    assert not (home / ".config/systemd/user/llmgateway.service").exists()
    dry = sh(tr, "service-unit", str(UID), str(home), flags=["--dry-run", "--test-root", str(tr)])
    assert dry.returncode == 1
