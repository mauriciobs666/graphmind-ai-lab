"""How the normalizer talks to git and to the staged tree (P1-05 5b gate F2-F5): the built commit is not HEAD, an argument
named like an option or a ref is a path, host git configuration cannot change a result, and a file name from the facts
cannot leave the stage."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from extractors.joern import run as joern_run
from extractors.joern.normalize import assemble_bundle, blame, core
from normalize_assembly_helpers import git_history
from normalize_helpers import py_facts

UUID = "11111111-1111-4111-8111-111111111111"
BUILT_AT = "2026-01-01T00:00:00Z"
D = ["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"]
S = ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"]
H1 = "a = 1\nb = 2\nc = 3\n"
H2 = "a = 1\nb = 22\nc = 3\n"


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """c1 is the built commit; c2 and c3 change lines and files that c1 holds, so HEAD (c3) differs from it everywhere."""
    path = tmp_path_factory.mktemp("hist")
    c = git_history(path, [
        (D[0], {"e.py": "", "h.py": H1, "-n.py": "x = 1\n", "HEAD": "y\n", "*.ts": "", "z.ts": "1\n", "m.py": "1\n2\n3\n4\n5\n"}),
        (D[1], {"e.py": "x = 1\n", "h.py": H2, "z.ts": "2\n", "m.py": "1\n2\n3\nfour\n5\n"}),
        (D[2], {"e.py": "", "-n.py": "x = 2\n"})])
    return path, c


# --- F2: abbreviated porcelain blocks ----------------------------------------------------------------------------------
def test_every_line_of_one_commit_is_blamed_to_it_though_porcelain_abbreviates_the_later_blocks(repo):
    path, (c1, c2, _) = repo
    per, times = blame.blame(path, c1, "m.py")
    assert per == {1: c1, 2: c1, 3: c1, 4: c1, 5: c1}
    per, times = blame.blame(path, c2, "m.py")
    assert per == {1: c1, 2: c1, 3: c1, 4: c2, 5: c1} and set(times) == {c1, c2}


@pytest.mark.parametrize("width", [40, 64])
def test_the_porcelain_parser_takes_sha1_and_sha256_commit_ids(width):
    h = "a" * width
    text = f"{h} 1 1 2\nauthor x\ncommitter-time 100\n\tl1\n{h} 2 2\n\tl2\n"
    assert blame.parse_porcelain(text) == ({1: h, 2: h}, {h: 100})


# --- F3: the built commit is not HEAD; an argument is a path, never a ref or an option ------------------------------------
def test_blame_reads_the_built_commit_not_head(repo):
    path, (c1, _, _) = repo
    assert blame.blame(path, c1, "h.py")[0] == {1: c1, 2: c1, 3: c1}               # HEAD would give line 2 to c2


def test_the_fallback_is_the_last_commit_up_to_the_built_commit_not_up_to_head(repo):
    path, (c1, c2, c3) = repo
    assert blame.path_last_commit(path, c1, "e.py") == (S[0], c1)
    assert blame.path_last_commit(path, c2, "e.py") == (S[1], c2)
    assert blame.path_last_commit(path, c3, "e.py") == (S[2], c3)


@pytest.mark.parametrize("name", ["HEAD", "-n.py"])
def test_a_path_named_like_a_ref_or_an_option_is_a_path(repo, name):
    path, (c1, _, _) = repo
    assert blame.blame(path, c1, name)[0] == {1: c1}
    assert blame.path_last_commit(path, c1, name) == (S[0], c1)


def asm(tmp_path, repo, commit, files, *, root=".", stage=None):
    path, _ = repo
    return assemble_bundle.assemble(lang="python", unit_root=root, repo_id="r", facts=py_facts(files), stage=stage or path,
                                    astgen_dir=None, repo=path, commit=commit, audit_ids=[UUID],
                                    extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)


def test_every_symbol_of_a_unit_built_at_an_old_commit_carries_that_commits_date_and_id(tmp_path, repo):
    path, (c1, _, _) = repo
    records, _ = asm(tmp_path, repo, c1, {"h.py": H2, "e.py": ""})        # the stage holds c3's files; the blame is c1's
    syms = [r for r in records if r["type"] == "symbol"]
    assert syms and {(r["lastChangedAt"], r["lastChangedCommit"]) for r in syms} == {(S[0], c1)}


def test_a_unit_below_the_repository_root_has_that_directorys_tree_and_blames_its_files_by_repository_path(tmp_path):
    src = "def f():\n    pass\n"
    path = tmp_path / "repo"
    c1 = git_history(path, [(D[0], {"sub/x.py": src, "sub/e.py": "", "o.txt": "o"}), (D[1], {"sub/e.py": "q\n"}),
                            (D[2], {"sub/e.py": ""})])[0]
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "x.py").write_text(src)
    (stage / "e.py").write_text("")
    records, _ = assemble_bundle.assemble(lang="python", unit_root="sub", repo_id="r", facts=py_facts({"x.py": src, "e.py": ""}),
                                          stage=stage, astgen_dir=None, repo=path, commit=c1, audit_ids=[UUID],
                                          extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)
    git = lambda rev: subprocess.run(["git", "-C", str(path), "rev-parse", rev], check=True, capture_output=True, text=True).stdout.strip()
    assert records[0]["builtTree"] == git(f"{c1}:sub") != git(f"{c1}^{{tree}}")
    syms = [r for r in records if r["type"] == "symbol"]
    assert {r["path"] for r in syms} == {"sub/x.py", "sub/e.py"}
    assert {(r["lastChangedAt"], r["lastChangedCommit"]) for r in syms} == {(S[0], c1)}      # e.py by the fallback, at c1 not HEAD


# --- F4: literal pathspecs, hermetic git, timeouts, commit shape, a clear error ----------------------------------------------
def test_a_path_with_glob_characters_is_that_file_not_a_pattern(repo):
    path, (_, _, c3) = repo
    assert blame.path_last_commit(path, c3, "*.ts") == (S[0], repo[1][0])         # the empty file literally named `*.ts`
    with pytest.raises(blame.HistoryError, match=r"\?\.ts"):
        blame.path_last_commit(path, c3, "?.ts")                                  # no such file: not "z.ts" by globbing


def test_a_path_with_no_history_at_the_commit_is_a_clear_error_not_an_index_error(repo):
    path, (c1, _, _) = repo
    with pytest.raises(blame.HistoryError, match=f"nope.py.* at {c1[:12]}: no commit"):
        blame.path_last_commit(path, c1, "nope.py")


@pytest.mark.parametrize("how", ["env", "repo-config"])
def test_blame_ignore_revs_configuration_cannot_change_the_commit_a_line_is_blamed_to(tmp_path, monkeypatch, how):
    path = tmp_path / "r"
    c1, c2 = git_history(path, [(D[0], {"i.py": "a\nb\n"}), (D[1], {"i.py": "a\nB\n"})])
    ignore = tmp_path / "ignore"
    ignore.write_text(c2 + "\n")
    if how == "env":
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "blame.ignoreRevsFile")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(ignore))
    else:
        subprocess.run(["git", "-C", str(path), "config", "blame.ignoreRevsFile", str(ignore)], check=True)
    bare = subprocess.run(["git", "-C", str(path), "blame", "--porcelain", c2, "--", "i.py"], capture_output=True, text=True).stdout
    assert blame.parse_porcelain(bare)[0][2] == c1                                  # premise: for a bare git the config flips it
    assert blame.blame(path, c2, "i.py")[0] == {1: c1, 2: c2}


def test_git_runs_with_literal_pathspecs_a_timeout_and_no_host_configuration(repo, monkeypatch):
    path, (c1, _, _) = repo
    seen = []
    real = subprocess.run

    def spy(argv, **kw):
        seen.append((argv, kw))
        return real(argv, **kw)
    monkeypatch.setattr(blame.subprocess, "run", spy)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "x.y")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "z")
    blame.blame(path, c1, "h.py")
    blame.path_last_commit(path, c1, "e.py")
    blame.require_full_history(path)
    assert len(seen) == 3
    for argv, kw in seen:
        assert "--literal-pathspecs" in argv and isinstance(kw.get("timeout"), (int, float)) and kw["timeout"] > 0
        env = kw["env"]
        assert env["GIT_CONFIG_GLOBAL"] == env["GIT_CONFIG_SYSTEM"] == "/dev/null" and not any(
            k.startswith("GIT_CONFIG_COUNT") or k.startswith("GIT_CONFIG_KEY") for k in env)


@pytest.mark.parametrize("commit, ok", [("a" * 40, True), ("0" * 64, True), ("HEAD", False), ("--output=x", False), ("abc123", False),
                                        ("A" * 40, False), ("g" * 40, False), ("a" * 41, False), ("", False)])
def test_a_commit_id_is_40_or_64_lowercase_hex(commit, ok):
    assert blame.is_commit_id(commit) is ok


def test_assemble_refuses_a_commit_that_is_not_an_id(repo):
    with pytest.raises(assemble_bundle.BuildError, match="commit"):
        asm(None, repo, "HEAD", {"h.py": H2})


# --- F5: file names from the facts stay inside the stage; non-UTF-8 is a typed refusal ---------------------------------------
@pytest.fixture
def staged(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "ok.py").write_text("x = 1\n")
    (tmp_path / "outside.py").write_text("secret = 1\n")
    (stage / "link.py").symlink_to(tmp_path / "outside.py")
    return stage, tmp_path / "outside.py"


@pytest.mark.parametrize("name", ["../outside.py", "OUTSIDE_ABS", "link.py", "a/../../outside.py"])
def test_a_file_name_that_leaves_the_stage_is_refused_and_never_read(staged, monkeypatch, name):
    stage, outside = staged
    name = str(outside) if name == "OUTSIDE_ABS" else name
    reads = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append(Path(self).resolve()), real(self))[1])
    src = core.Source(stage)
    for call in (src.exists, src.text, src.raw):
        with pytest.raises(core.SourceRefused, match="outside.py|link.py"):
            call(name)
    assert outside.resolve() not in reads


def test_a_confined_name_reads_normally(staged):
    stage, _ = staged
    src = core.Source(stage)
    assert src.exists("ok.py") and src.text("ok.py") == "x = 1\n" and src.raw("ok.py") == b"x = 1\n" and not src.exists("missing.py")


def test_assemble_never_reads_a_file_outside_the_stage_even_when_the_facts_name_it(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    c1 = git_history(stage, [(D[0], {"h.py": H1})])[0]
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n")
    facts = py_facts({"h.py": H1})
    facts["f"].append({"k": "f", "name": "../outside.py"})
    reads = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append(Path(self).resolve()), real(self))[1])
    with pytest.raises(core.SourceRefused, match="outside.py"):
        assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts, stage=stage, astgen_dir=None, repo=stage,
                                 commit=c1, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC)
    assert outside.resolve() not in reads


def test_a_source_file_that_is_not_utf8_fails_the_unit_with_a_typed_error_naming_it(tmp_path):
    stage = tmp_path / "repo"
    c1 = git_history(stage, [(D[0], {"a.py": "x = 1\n"})])[0]
    (stage / "a.py").write_bytes(b"x = '\xe9'\n")                       # Latin-1
    with pytest.raises(core.SourceRefused, match="a.py") as e:
        assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=py_facts({"a.py": "x = 1\n"}), stage=stage,
                                 astgen_dir=None, repo=stage, commit=c1, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC)
    assert not isinstance(e.value, UnicodeDecodeError)


def test_a_git_call_that_times_out_is_a_history_error(repo, monkeypatch):
    def hang(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])
    monkeypatch.setattr(blame.subprocess, "run", hang)
    with pytest.raises(blame.HistoryError, match="git blame in .* timed out after"):
        blame.blame(repo[0], repo[1][0], "h.py")


def test_git_does_not_inherit_variables_that_redirect_it_to_another_repository(tmp_path, monkeypatch):
    mine, other = tmp_path / "mine", tmp_path / "other"
    (c1,) = git_history(mine, [(D[0], {"a.py": "a\n"})])
    git_history(other, [(D[1], {"b.py": "b\n"})])
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other / ".git" / "index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(other / ".git" / "objects"))
    monkeypatch.setenv("GIT_NAMESPACE", "elsewhere")
    bare = subprocess.run(["git", "-C", str(mine), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert bare != c1                                                              # premise: a bare git is redirected
    assert blame.git(mine, "rev-parse", "HEAD").decode().strip() == c1
    assert not [k for k in blame._env() if k.startswith("GIT_") and k not in
                ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM")]


def test_last_changed_over_a_region_far_beyond_the_blamed_lines_is_bounded_and_ignores_the_unblamed_ones():
    from normalize_helpers import run_bounded
    code = """
import pytest
from extractors.joern.normalize import blame
per, times = {1: "a" * 40, 2: "b" * 40}, {"a" * 40: 100, "b" * 40: 200}
assert blame.last_changed(per, times, [(2, 10**12), (-10**12, 1)]) == ("1970-01-01T00:03:20Z", "b" * 40)
try:
    blame.last_changed(per, times, [(5, 10**12)])
except ValueError:
    print("ok")
"""
    r = run_bounded(code)                                              # an unbounded walk dies of the address-space cap: returncode != 0
    assert (r.returncode, r.stdout.strip()) == (0, "ok"), r.stderr[-300:]
