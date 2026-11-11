"""Hardened git for a hostile repository (security review of bundle spec6, m3).

A repository under analysis is untrusted: its own ``.git/config`` and ``.gitattributes`` can name commands that git runs
(``core.fsmonitor`` on an index read, a ``diff.<driver>.textconv`` on ``git blame``). Every git call that touches such a
repository goes through ``ccg.gitsafe``. The planted repository is built in ``tmp_path``; a marker file proves a command ran.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ccg import gitsafe, units as un
from extractors.joern.normalize import blame
from test_units import manifest_of

ODD = ["-x", "--output=EVIL", "sp ace.py", "ünï.py"]


def _plain_env(home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(HOME=str(home), GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return env


def _g(repo: Path, env, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def _marker_cmd(marker: Path) -> str:
    return f"touch {marker}"


@pytest.fixture
def hostile(tmp_path):
    """A repository whose own configuration names a command for every route a read-only call could take. Returns
    ``(repo, markers_dir, commit)``; configuration is planted after the commit, so building it runs nothing."""
    repo, marks, home = tmp_path / "repo", tmp_path / "marks", tmp_path / "home"
    for d in (repo, marks, home):
        d.mkdir()
    env = _plain_env(home)
    _g(repo, env, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("a = 1\n")
    for name in ODD:
        (repo / name).write_text("x = 1\n")
    (repo / ".gitattributes").write_text("* diff=evil filter=evil\n")
    _g(repo, env, "add", "-A", "--", ".")
    _g(repo, env, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "c")
    head = _g(repo, env, "cat-file", "commit", "HEAD")                              # forge a `gpgsig` header: never verified before git tries to
    forged = head.replace("\n\n", "\ngpgsig -----BEGIN PGP SIGNATURE-----\n \n AAAA\n -----END PGP SIGNATURE-----\n\n", 1) + "\n"
    new = subprocess.run(["git", "-C", str(repo), "hash-object", "-t", "commit", "-w", "--literal", "--stdin"], input=forged, check=True,
                         capture_output=True, text=True, env=env).stdout.strip()
    _g(repo, env, "update-ref", "refs/heads/main", new)
    gpg = tmp_path / "gpg.sh"
    gpg.write_text(f"#!/bin/sh\n{_marker_cmd(marks / 'gpg')}\nexit 1\n")
    gpg.chmod(0o755)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for h in ("post-checkout", "post-merge", "pre-commit", "reference-transaction", "fsmonitor-watchman"):
        p = hooks / h
        p.write_text(f"#!/bin/sh\n{_marker_cmd(marks / ('hook-' + h))}\n")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    cfg = {
        "core.fsmonitor": f"{_marker_cmd(marks / 'fsmonitor')}; echo",
        "log.showSignature": "true",
        "gpg.program": str(gpg), "gpg.openpgp.program": str(gpg), "gpg.x509.program": str(gpg), "gpg.ssh.program": str(gpg),
        "gpg.format": "openpgp",
        "core.hooksPath": str(hooks),
        "core.pager": f"{_marker_cmd(marks / 'pager')}; cat",
        "core.sshCommand": _marker_cmd(marks / "ssh"),
        "diff.external": _marker_cmd(marks / "extdiff"),
        "diff.evil.textconv": f"sh -c '{_marker_cmd(marks / 'textconv')}; cat \"$0\"'",
        "filter.evil.clean": f"{_marker_cmd(marks / 'clean')}; cat",
        "filter.evil.smudge": f"{_marker_cmd(marks / 'smudge')}; cat",
    }
    for k, v in cfg.items():
        _g(repo, env, "config", k, v)
    return repo, marks, _g(repo, env, "rev-parse", "HEAD")


def _ran(marks: Path) -> list[str]:
    return sorted(p.name for p in marks.iterdir())


def _premise(repo: Path, marks: Path, *args: str) -> None:
    """A bare git over the planted repository runs the planted command: the test would be vacuous otherwise."""
    subprocess.run(["git", "-C", str(repo), *args], capture_output=True)


def test_premise_a_bare_git_runs_the_planted_commands(hostile):
    repo, marks, _ = hostile
    _premise(repo, marks, "ls-files")
    _premise(repo, marks, "blame", "--porcelain", "HEAD", "--", "a.py")
    _premise(repo, marks, "log", "-1", "--format=%ct %H", "HEAD", "--", "a.py")
    assert {"fsmonitor", "textconv", "gpg"} <= set(_ran(marks))


# --- units: the file listing ----------------------------------------------------------------------------------------


def test_git_files_over_a_hostile_repository_runs_no_command_and_still_lists_the_files(hostile):
    repo, marks, _ = hostile
    got = un._git_files(repo)
    assert sorted(got) == sorted([".gitattributes", "a.py", *ODD])
    assert _ran(marks) == []


def test_discover_units_over_a_hostile_repository_runs_no_command(hostile):
    repo, marks, _ = hostile
    got = un.discover_units(repo, manifest_of(("r", ".")))
    assert [(u.rootPath, u.language) for u in got if u.language == "python"] == [(".", "python")]
    assert _ran(marks) == []


def test_git_files_ignores_host_configuration_and_injected_config_variables(tmp_path, monkeypatch):
    repo, marks, home = tmp_path / "r", tmp_path / "m", tmp_path / "h"
    for d in (repo, marks, home, home / ".config" / "git"):
        d.mkdir(parents=True, exist_ok=True)
    env = _plain_env(home)
    _g(repo, env, "init", "-q")
    (repo / "a.py").write_text("")
    _g(repo, env, "add", "-A")
    cmd = f"{_marker_cmd(marks / 'fsm')}; echo"
    (home / ".gitconfig").write_text(f"[core]\n\tfsmonitor = {cmd}\n")
    (home / ".config" / "git" / "config").write_text(f"[core]\n\tfsmonitor = {cmd}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    assert un._git_files(repo) == ["a.py"]
    assert _ran(marks) == []                                                      # HOME / XDG configuration
    monkeypatch.setenv("HOME", str(tmp_path / "none"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "none"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", cmd)
    assert un._git_files(repo) == ["a.py"]
    assert _ran(marks) == []                                                      # GIT_CONFIG_COUNT injection


def test_git_files_is_not_redirected_by_git_dir(tmp_path, monkeypatch):
    mine, other, home = tmp_path / "mine", tmp_path / "other", tmp_path / "h"
    home.mkdir()
    env = _plain_env(home)
    for d, f in ((mine, "mine.py"), (other, "other.py")):
        d.mkdir()
        _g(d, env, "init", "-q")
        (d / f).write_text("")
        _g(d, env, "add", "-A")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert un._git_files(mine) == ["mine.py"]


# --- blame and the commit's tree ----------------------------------------------------------------------------------------


def test_blame_over_a_hostile_repository_runs_no_textconv(hostile):
    repo, marks, c = hostile
    per_line, _ = blame.blame(repo, c, "a.py")
    assert per_line == {1: c}
    assert _ran(marks) == []


@pytest.mark.parametrize("args", [
    ("ls-files", "-z"), ("ls-tree", "-r", "-z", "HEAD"), ("cat-file", "blob", "HEAD:a.py"), ("rev-parse", "HEAD"),
    ("rev-parse", "--is-shallow-repository"), ("log", "-1", "--format=%ct %H", "HEAD", "--", "a.py"),
    ("log", "-1", "--show-signature", "HEAD"), ("show", "HEAD"), ("log", "-p", "-1"), ("log", "-1", "--stat"), ("diff", "4b825dc642cb6eb9a060e54bf8d69288fbee4904", "HEAD"),
    ("blame", "--porcelain", "HEAD", "--", "a.py"), ("format-patch", "-1", "--stdout"), ("whatchanged", "-1", "-p"),
    ("log", "-1", "-p", "--no-ext-diff", "--no-textconv", "HEAD"), ("show", "--no-ext-diff", "--no-textconv", "HEAD"),
    ("blame", "--porcelain", "--no-textconv", "--ignore-revs-file=", "HEAD", "--", "a.py"),
])
def test_no_git_call_of_the_helper_runs_a_planted_command(hostile, args):
    repo, marks, _ = hostile
    gitsafe.git(repo, *args)
    assert _ran(marks) == []


def test_option_like_file_names_are_files_not_options(hostile):
    repo, marks, c = hostile
    for name in ("-x", "--output=EVIL"):
        assert blame.blame(repo, c, name)[0] == {1: c}
        assert blame.path_last_commit(repo, c, name)[1] == c
    assert not (repo / "EVIL").exists()
    assert gitsafe.git(repo, "ls-files", "-z", paths=["-x"]) == b"-x\0"                  # after `--`, not an option
    assert gitsafe.git(repo, "ls-files", "-z", paths=["--output=EVIL"]) == b"--output=EVIL\0"


def test_a_branch_name_with_odd_characters_does_not_matter(hostile):
    repo, marks, c = hostile
    name = "feat/ünï\"q'$x`y`;-z"
    env = _plain_env(repo.parent)
    _g(repo, env, "branch", name, "HEAD")
    _g(repo, env, "symbolic-ref", "HEAD", "refs/heads/" + name)      # (a checkout would run the planted filter)
    for m in (repo.parent / "marks").iterdir():                                    # the setup's own bare git ran a hook
        m.unlink()
    assert un._git_files(repo)
    assert gitsafe.git(repo, "rev-parse", "HEAD").decode().strip() == c
    assert blame.blame(repo, c, "a.py")[0] == {1: c}
    assert _ran(repo.parent / "marks") == []


def test_a_replace_ref_does_not_change_the_content_read(hostile, tmp_path):
    repo, marks, _ = hostile
    env = _plain_env(tmp_path / "home")
    oid = _g(repo, env, "rev-parse", "HEAD:a.py")
    fake = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input="forged\n", check=True,
                          capture_output=True, text=True, env=env).stdout.strip()
    _g(repo, env, "replace", oid, fake)
    assert _g(repo, env, "cat-file", "blob", oid) == "forged"                      # premise
    assert gitsafe.git(repo, "cat-file", "blob", oid) == b"a = 1\n"


# --- the helper's own contract ----------------------------------------------------------------------------------------------


def _spy(monkeypatch):
    seen = []
    real = subprocess.run

    def spy(argv, **kw):
        seen.append((argv, kw))
        return real(argv, **kw)
    monkeypatch.setattr(gitsafe.subprocess, "run", spy)
    return seen


def test_every_call_is_a_list_without_a_shell_with_a_timeout_and_a_quiet_stdin(hostile, monkeypatch):
    repo, _, _ = hostile
    seen = _spy(monkeypatch)
    gitsafe.git(repo, "rev-parse", "HEAD")
    (argv, kw), = seen
    assert isinstance(argv, list) and not kw.get("shell")
    assert isinstance(kw["timeout"], (int, float)) and kw["timeout"] > 0
    assert kw["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("pair", ["log.showSignature=false", "gpg.program=false", "gpg.openpgp.program=false", "gpg.x509.program=false",
                                  "gpg.ssh.program=false", "core.fsmonitor=false", "core.hooksPath=/dev/null", "core.pager=cat", "protocol.allow=never",
                                  "core.sshCommand=false", "core.gitProxy=false", "core.askPass=false",
                                  "core.attributesFile=/dev/null"])
def test_every_call_disables_the_command_running_settings(hostile, monkeypatch, pair):
    repo, _, _ = hostile
    seen = _spy(monkeypatch)
    gitsafe.git(repo, "rev-parse", "HEAD")
    argv = seen[0][0]
    assert pair in argv
    assert ["-c", pair] == argv[argv.index(pair) - 1:argv.index(pair) + 1]


def test_every_call_uses_literal_pathspecs_and_no_replace_objects_and_never_trusts_a_directory(hostile, monkeypatch):
    repo, _, _ = hostile
    seen = _spy(monkeypatch)
    gitsafe.git(repo, "rev-parse", "HEAD")
    argv = seen[0][0]
    assert "--literal-pathspecs" in argv and "--no-replace-objects" in argv
    assert not [a for a in argv if "safe.directory" in a]       # `safe.directory=*` would switch the ownership check off


def test_the_environment_has_no_git_variable_but_the_config_nulls_and_an_empty_home(hostile, monkeypatch):
    repo, _, _ = hostile
    monkeypatch.setenv("GIT_SSH_COMMAND", "x")
    monkeypatch.setenv("GIT_EXEC_PATH", "/nope")
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "x")
    seen = _spy(monkeypatch)
    gitsafe.git(repo, "rev-parse", "HEAD")
    env = seen[0][1]["env"]
    assert {k for k in env if k.startswith("GIT_")} == {
        "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM", "GIT_ALLOW_PROTOCOL", "GIT_NO_LAZY_FETCH"}
    assert env["GIT_ALLOW_PROTOCOL"] == "none" and env["GIT_NO_LAZY_FETCH"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == env["GIT_CONFIG_SYSTEM"] == os.devnull and env["GIT_CONFIG_NOSYSTEM"] == "1"
    for var in ("HOME", "XDG_CONFIG_HOME"):
        assert Path(env[var]).is_dir() and os.listdir(env[var]) == [] and env[var] != os.environ.get(var)


def _fake_git(tmp_path: Path, monkeypatch, body: str) -> None:
    d = tmp_path / "bin"
    d.mkdir()
    p = d / "git"
    p.write_text(f"#!{sys.executable}\n{body}\n")
    p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ['PATH']}")


def test_a_call_past_its_timeout_raises_git_timeout_and_a_file_listing_falls_back(tmp_path, monkeypatch):
    _fake_git(tmp_path, monkeypatch, "import time\ntime.sleep(30)")
    with pytest.raises(gitsafe.GitTimeout):
        gitsafe.git(tmp_path, "ls-files", timeout=0.5)
    assert issubclass(gitsafe.GitTimeout, subprocess.SubprocessError)
    monkeypatch.setattr(gitsafe, "DEFAULT_TIMEOUT", 0.5)
    assert un._git_files(tmp_path) is None


def test_output_past_the_cap_raises_and_a_file_listing_falls_back(tmp_path, monkeypatch):
    _fake_git(tmp_path, monkeypatch, "import sys\nsys.stdout.write('x' * 100000)")
    with pytest.raises(gitsafe.GitOutputTooLarge):
        gitsafe.git(tmp_path, "ls-files", max_output=1000)
    assert gitsafe.git(tmp_path, "ls-files", max_output=200000) == b"x" * 100000
    monkeypatch.setattr(gitsafe, "DEFAULT_MAX_OUTPUT", 1000)
    assert un._git_files(tmp_path) is None


def test_a_failing_call_raises_called_process_error_with_stderr(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as e:
        gitsafe.git(tmp_path, "cat-file", "blob", "0" * 40)
    assert e.value.stderr


def test_a_writer_that_never_stops_is_cut_at_the_output_cap_not_at_the_timeout(tmp_path, monkeypatch):
    _fake_git(tmp_path, monkeypatch, "import sys\nwhile True:\n    sys.stdout.write('x' * 65536)\n    sys.stdout.flush()")
    with pytest.raises(gitsafe.GitOutputTooLarge):
        gitsafe.git(tmp_path, "ls-files", max_output=10000, timeout=20)


def test_blame_gives_its_timeout_to_every_call(hostile, monkeypatch):
    repo = hostile
    seen = _spy(monkeypatch)
    monkeypatch.setattr(blame, "GIT_TIMEOUT", 7)
    blame.git(repo[0], "rev-parse", "HEAD")
    assert seen[0][1]["timeout"] == 7


def test_no_module_runs_a_bare_git():
    """Every ``git`` argv in ``ccg`` and ``extractors`` (outside ``indexers`` and ``markdown``, which never touch a repository
    under analysis) is built by ``ccg.gitsafe``: no source line holds a ``"git"`` literal but gitsafe's own."""
    root = Path(gitsafe.__file__).resolve().parents[1]
    bad = []
    for sub in ("ccg", "extractors/joern"):
        for f in (root / sub).rglob("*.py"):
            if f.name != "gitsafe.py" and '"git"' in f.read_text():
                bad.append(str(f.relative_to(root)))
    assert bad == []
    assert '"git"' in Path(gitsafe.__file__).read_text()                          # not vacuous


# --- the configuration list is a bound, not proof: one planted repository per executable-config route ------------------------


def _raw(repo, *args):
    """``git`` through the helper's ``command()`` and ``environment()`` only, without ``git()``'s own flags and checks: what
    the configuration and the environment alone stop. (Hooks and ``core.pager`` have no read-only route; their ``-c`` is pinned by
    the wiring test above, not by a behaviour.)"""
    return subprocess.run(gitsafe.command(repo, *args), capture_output=True, env=gitsafe.environment())


@pytest.mark.parametrize("args", [("log", "-1", "--format=%ct %H", "HEAD", "--", "a.py"), ("log", "-1", "--show-signature")])
def test_the_configuration_alone_stops_a_forged_signature_from_running_gpg(hostile, args):
    repo, marks, _ = hostile
    _raw(repo, *args)
    assert _ran(marks) == []


def test_the_configuration_alone_stops_a_ssh_command(tmp_path):
    marks = tmp_path / "m"
    marks.mkdir()
    repo, oid = _promisor_repo(tmp_path, "ssh://host/x", marks, ssh=True)
    _raw(repo, "cat-file", "blob", oid)
    assert _ran(marks) == []


def _promisor_repo(tmp_path, url, marks, ssh=False):
    """A repository whose blob is missing and whose promisor remote is ``url``: any read of the blob lazy-fetches."""
    repo = tmp_path / "prom"
    repo.mkdir()
    env = _plain_env(tmp_path / "home")
    _g(repo, env, "init", "-q")
    (repo / "a.py").write_text("a = 1\n")
    _g(repo, env, "add", "-A")
    _g(repo, env, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "c")
    oid = _g(repo, env, "rev-parse", "HEAD:a.py")
    script = tmp_path / "ext.sh"
    script.write_text(f"#!/bin/sh\n{_marker_cmd(marks / 'ext')}\nexit 1\n")
    script.chmod(0o755)
    for k, v in {"core.repositoryformatversion": "1", "extensions.partialClone": "origin", "remote.origin.promisor": "true",
                 "remote.origin.url": url if ssh else f"ext::{script}", "protocol.ext.allow": "always", "protocol.ssh.allow": "always",
                 "core.sshCommand": str(script)}.items():
        _g(repo, env, "config", k, v)
    obj = repo / ".git" / "objects" / oid[:2] / oid[2:]
    obj.chmod(0o644)
    obj.unlink()
    return repo, oid


def test_premise_a_bare_git_runs_the_promisor_ext_command(tmp_path):
    marks = tmp_path / "m"
    marks.mkdir()
    repo, oid = _promisor_repo(tmp_path, "", marks)
    subprocess.run(["git", "-C", str(repo), "cat-file", "blob", oid], capture_output=True)
    assert _ran(marks) == ["ext"]


@pytest.mark.parametrize("via", ["git", "batch"])
def test_a_missing_object_never_lazy_fetches_from_a_repository_chosen_remote(tmp_path, via):
    marks = tmp_path / "m"
    marks.mkdir()
    repo, oid = _promisor_repo(tmp_path, "", marks)
    if via == "git":
        with pytest.raises(subprocess.CalledProcessError):
            gitsafe.git(repo, "cat-file", "blob", oid)
    else:                                                  # the `cat-file --batch` Popen of extractors/joern/run.py
        p = subprocess.run(gitsafe.command(repo, "cat-file", "--batch"), input=oid.encode() + b"\n", capture_output=True,
                           env=gitsafe.environment())
        assert b"missing" in p.stdout
    assert _ran(marks) == []


@pytest.mark.parametrize("var", ["GIT_ALLOW_PROTOCOL", "GIT_NO_LAZY_FETCH"])
def test_each_of_the_two_network_kill_switches_alone_stops_the_promisor_fetch(tmp_path, var):
    marks = tmp_path / "m"
    marks.mkdir()
    repo, oid = _promisor_repo(tmp_path, "", marks)
    env = gitsafe.environment()
    for other in {"GIT_ALLOW_PROTOCOL", "GIT_NO_LAZY_FETCH"} - {var}:
        del env[other]
    subprocess.run(gitsafe.command(repo, "cat-file", "blob", oid), capture_output=True, env=env)
    assert _ran(marks) == []


# --- M1: the helper protects diff readers itself -----------------------------------------------------------------------------


def test_premise_a_bare_show_and_diff_run_the_planted_textconv(hostile):
    repo, marks, _ = hostile
    _premise(repo, marks, "show", "HEAD")
    assert "textconv" in _ran(marks)


@pytest.mark.parametrize("args", [("cat-file", "--textconv", "HEAD:a.py"), ("cat-file", "--filters", "HEAD:a.py"), ("checkout", "HEAD"),
                                  ("status",), ("fetch",), ("config", "--list"), ("--exec-path=/x", "ls-files"), ("-c", "core.fsmonitor=x", "ls-files"),
                                  (), ("",)])
def test_a_subcommand_that_reads_diffs_unsafely_or_is_off_the_list_is_refused(hostile, args):
    repo, marks, _ = hostile
    with pytest.raises(gitsafe.GitError):
        gitsafe.git(repo, *args)
    assert _ran(marks) == []


@pytest.mark.parametrize("sub", ["diff", "show", "log", "whatchanged", "format-patch", "blame"])
def test_a_diff_reader_gets_no_ext_diff_and_no_textconv_from_the_helper(hostile, monkeypatch, sub):
    repo, _, _ = hostile
    seen = _spy(monkeypatch)
    try:
        gitsafe.git(repo, sub, "HEAD")
    except subprocess.CalledProcessError:
        pass
    argv = seen[0][0]
    tail = argv[argv.index(sub):]
    assert "--no-ext-diff" in tail and "--no-textconv" in tail


# --- M2/M3: a repository that redirects or rewrites its own storage or history is refused ----------------------------------------


def test_a_dot_git_file_that_redirects_to_another_git_dir_is_refused(tmp_path, hostile):
    repo, _, _ = hostile
    g = tmp_path / "g"
    g.mkdir()
    (g / ".git").write_text(f"gitdir: {repo / '.git'}\n")
    assert subprocess.run(["git", "-C", str(g), "ls-files"], capture_output=True).stdout          # premise: bare git reads the other repository
    with pytest.raises(gitsafe.UnsafeRepository):
        gitsafe.git(g, "ls-files")
    assert un._git_files(g) is None                                                        # discovery falls back to walking `g`


@pytest.mark.parametrize("rel", ["objects/info/alternates", "info/grafts", "commondir"])
def test_a_repository_with_alternates_grafts_or_a_commondir_is_refused(hostile, rel):
    repo, _, _ = hostile
    f = repo / ".git" / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("../../../elsewhere\n")
    with pytest.raises(gitsafe.UnsafeRepository) as e:
        gitsafe.git(repo, "rev-parse", "HEAD")
    assert str(f) not in str(e.value) and rel in str(e.value)
    assert un._git_files(repo) is None


def test_a_subdirectory_of_a_work_tree_and_a_bare_repository_are_not_refused(hostile, tmp_path):
    repo, _, c = hostile
    (repo / "sub").mkdir()
    assert gitsafe.git(repo / "sub", "rev-parse", "HEAD").decode().strip() == c
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(bare)], check=True, capture_output=True, env=_plain_env(tmp_path / "h"))
    assert gitsafe.git(bare, "rev-parse", "HEAD").decode().strip() == c


def test_a_shallow_clone_reports_itself_shallow_through_the_helper(hostile, tmp_path):
    repo, _, _ = hostile
    shallow = tmp_path / "sh"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow)], check=True, capture_output=True,
                   env=_plain_env(tmp_path / "h"))
    assert gitsafe.git(shallow, "rev-parse", "--is-shallow-repository").decode().strip() == "true"


# --- m2: a NUL byte is the helper's typed error, not a ValueError ------------------------------------------------------------------


@pytest.mark.parametrize("kw", [{"paths": ["a\0b"]}, {}])
def test_a_nul_in_an_argument_or_path_is_a_git_error(hostile, kw):
    repo, _, _ = hostile
    args = ("ls-files", "a\0b") if not kw else ("ls-files",)
    with pytest.raises(gitsafe.GitError):
        gitsafe.git(repo, *args, **kw)


# --- the staging stream of extractors/joern/run.py ---------------------------------------------------------------------------------


def _stage(tmp_path, files):
    from ccg.units import Unit
    from extractors.joern import run
    repo = tmp_path / "srepo"
    repo.mkdir()
    env = _plain_env(tmp_path / "home")
    _g(repo, env, "init", "-q")
    for rel, text in files.items():
        (repo / rel).write_text(text)
    _g(repo, env, "add", "-A")
    _g(repo, env, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "c")
    sha = _g(repo, env, "rev-parse", "HEAD")
    return run, repo, sha, Unit("1", "r", ".", "python"), tmp_path / "dest"


def test_staging_refuses_a_blob_past_the_per_blob_cap(tmp_path, monkeypatch):
    run, repo, sha, u, dest = _stage(tmp_path, {"a.py": "a = 1\n", "big.py": "x" * 5000})
    assert run.stage_unit(repo, sha, u, [], dest) == 2                                   # premise: it stages without a cap
    monkeypatch.setattr(run, "MAX_BLOB_BYTES", 1000)
    with pytest.raises(run.StageError, match="big.py.*cap|cap.*big.py"):
        run.stage_unit(repo, sha, u, [], tmp_path / "dest2")


def test_staging_refuses_a_stage_past_the_total_cap(tmp_path, monkeypatch):
    run, repo, sha, u, dest = _stage(tmp_path, {"a.py": "a" * 600, "b.py": "b" * 600})
    monkeypatch.setattr(run, "MAX_STAGE_BYTES", 1000)
    with pytest.raises(run.StageError, match="total"):
        run.stage_unit(repo, sha, u, [], dest)


def test_staging_is_bounded_by_a_deadline(tmp_path, monkeypatch):
    run, repo, sha, u, dest = _stage(tmp_path, {"a.py": "a = 1\n"})
    _fake_git(tmp_path, monkeypatch, "import sys, time\nif 'rev-parse' in sys.argv:\n    print('0' * 40)\nelif 'ls-tree' in sys.argv:\n"
              "    sys.stdout.write('100644 blob ' + '1' * 40 + '\\ta.py\\0')\nelse:\n    time.sleep(30)")
    monkeypatch.setattr(run, "STAGE_TIMEOUT", 0.5)
    t0 = __import__("time").monotonic()
    with pytest.raises(run.StageError, match="timed out"):
        run.stage_unit(repo, sha, u, [], dest)
    assert __import__("time").monotonic() - t0 < 10


@pytest.mark.parametrize("kind", ["PGP SIGNATURE", "SIGNED MESSAGE"])      # (ssh needs a parseable signature and ssh-keygen: wiring-pinned only)
def test_the_configuration_alone_stops_each_signature_format_from_running_its_program(hostile, kind):
    """``--show-signature`` is passed explicitly, so ``log.showSignature`` is not what stops it: each ``gpg.<fmt>.program`` is."""
    repo, marks, _ = hostile
    env = _plain_env(repo.parent)
    body = _g(repo, env, "cat-file", "commit", "HEAD").split("gpgsig ")[0]
    forged = body + f"gpgsig -----BEGIN {kind}-----\n \n AAAA\n -----END {kind}-----\n\nm\n"
    oid = subprocess.run(["git", "-C", str(repo), "hash-object", "-t", "commit", "-w", "--literal", "--stdin"], input=forged, check=True,
                         capture_output=True, text=True, env=env).stdout.strip()
    subprocess.run(["git", "-C", str(repo), "log", "-1", "--show-signature", oid], capture_output=True)          # premise
    assert "gpg" in _ran(marks)
    for m in marks.iterdir():
        m.unlink()
    _raw(repo, "log", "-1", "--show-signature", oid)
    assert _ran(marks) == []


def test_a_dot_git_symlink_to_another_git_dir_is_refused(tmp_path, hostile):
    repo, _, _ = hostile
    g = tmp_path / "g"
    g.mkdir()
    (g / ".git").symlink_to(repo / ".git")
    assert subprocess.run(["git", "-C", str(g), "ls-files"], capture_output=True).stdout          # premise
    with pytest.raises(gitsafe.UnsafeRepository):
        gitsafe.git(g, "ls-files")


def test_the_redirect_is_found_from_a_subdirectory_and_in_a_bare_repository(tmp_path, hostile):
    repo, _, c = hostile
    (repo / "sub").mkdir()
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "grafts").write_text("")
    with pytest.raises(gitsafe.UnsafeRepository):
        gitsafe.git(repo / "sub", "rev-parse", "HEAD")
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(bare)], check=True, capture_output=True, env=_plain_env(tmp_path / "h"))
    (bare / "objects" / "info").mkdir(exist_ok=True)
    (bare / "objects" / "info" / "alternates").write_text("/elsewhere\n")
    with pytest.raises(gitsafe.UnsafeRepository):
        gitsafe.git(bare, "rev-parse", "HEAD")


def test_a_dangling_symlink_named_like_a_redirect_file_is_refused_too(hostile):
    repo, _, _ = hostile
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "grafts").symlink_to("/nonexistent-target")
    with pytest.raises(gitsafe.UnsafeRepository):
        gitsafe.git(repo, "rev-parse", "HEAD")


# --- N1: the caller cannot re-enable a driver or name a path the helper writes or reads -------------------------------------------------


@pytest.mark.parametrize("args", [
    ("show", "--ext-diff", "HEAD"), ("show", "--textconv", "HEAD"), ("show", "--ext-d", "HEAD"), ("show", "--textc", "HEAD"), ("diff", "--output=/nonexistent-gs/x", "HEAD"),
    ("diff", "--output", "/nonexistent-gs/x"), ("diff", "-O/nonexistent-gs/x", "HEAD"), ("diff", "-O", "/nonexistent-gs/x"), ("format-patch", "-o", "/nonexistent-gs/d", "-1"),
    ("format-patch", "--output-directory=/nonexistent-gs/d", "-1"), ("log", "--git-dir=/x"), ("log", "--work-tree=/x"), ("log", "--exec-path=/x"),
    ("log", "-c"), ("log", "--config-env=a=B"), ("log", "--attr-source=HEAD"), ("log", "--open-files-in-pager"), ("diff", "--no-index", "/a", "/b"),
])
def test_an_option_that_re_enables_a_driver_or_names_a_path_is_refused(hostile, args):
    repo, marks, _ = hostile
    with pytest.raises(gitsafe.UnsafeRepository):
        gitsafe.git(repo, *args)
    assert _ran(marks) == [] and not Path("/nonexistent-gs/x").exists() and not Path("/nonexistent-gs/d").exists()


def test_the_hardening_flags_come_after_the_callers_options_and_before_any_double_dash(hostile, monkeypatch):
    repo, _, _ = hostile
    seen = _spy(monkeypatch)
    gitsafe.git(repo, "log", "-1", "-p", "HEAD", "--", "a.py")
    argv = seen[0][0]
    tail = argv[argv.index("log"):]
    assert tail.index("--no-textconv") > tail.index("-p") and tail.index("--no-ext-diff") > tail.index("HEAD")
    assert tail.index("--no-textconv") < tail.index("--") and tail[-2:] == ["--", "a.py"]


def test_a_later_flag_beats_an_earlier_one_so_the_order_is_load_bearing(hostile):
    """Premise of the ordering rule: git takes the last of ``--ext-diff`` / ``--no-ext-diff``."""
    repo, marks, _ = hostile
    subprocess.run(gitsafe.command(repo, "show", "--no-ext-diff", "--no-textconv", "--textconv", "HEAD"), capture_output=True,
                   env=gitsafe.environment())
    assert "textconv" in _ran(marks)


# --- N2/N3: the cat-file stream of stage_unit -----------------------------------------------------------------------------------------


def test_the_cat_file_stream_is_checked_by_construction_not_by_the_calls_before_it(tmp_path, monkeypatch):
    run, repo, sha, u, dest = _stage(tmp_path, {"a.py": "a = 1\n"})
    entries = run._tree_entries(repo, sha, ".")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "grafts").write_text("")
    monkeypatch.setattr(run, "_resolve_commit", lambda r, c: sha)          # the calls before the stream would have refused already
    monkeypatch.setattr(run, "_tree_entries", lambda r, s, root: entries)
    popen = []
    monkeypatch.setattr(run.subprocess, "Popen", lambda *a, **k: popen.append(a) or pytest.fail("Popen reached"))
    with pytest.raises(run.StageError):
        run.stage_unit(repo, sha, u, [], dest)
    assert popen == []


def test_a_timeout_during_the_last_blob_fails_the_stage_instead_of_staging_a_truncated_blob(tmp_path, monkeypatch):
    run, repo, sha, u, dest = _stage(tmp_path, {"a.py": "a = 1\n"})
    _fake_git(tmp_path, monkeypatch, "import sys, time\nif 'rev-parse' in sys.argv:\n    print('0' * 40)\nelif 'ls-tree' in sys.argv:\n"
              "    sys.stdout.write('100644 blob ' + '1' * 40 + '\\ta.py\\0')\nelse:\n    sys.stdin.readline()\n"
              "    sys.stdout.write('1' * 40 + ' blob 10\\nabc')\n    sys.stdout.flush()\n    time.sleep(30)")
    monkeypatch.setattr(run, "STAGE_TIMEOUT", 1.0)
    with pytest.raises(run.StageError, match="timed out"):
        run.stage_unit(repo, sha, u, [], dest)
    assert not list(dest.rglob("a.py"))
