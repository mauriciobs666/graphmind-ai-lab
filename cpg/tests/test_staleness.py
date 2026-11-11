"""P1-20 — ``ccg.staleness.unit_staleness`` and ``local_head``: a unit's built commit against the LOCAL head.

Temporary git repositories only. No remote exists in any of them, and nothing here (or in ``staleness.py``) fetches.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccg import staleness as S

CPG = Path(__file__).resolve().parent.parent
ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "PATH": os.environ["PATH"], "GIT_AUTHOR_NAME": "t",
       "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=ENV).stdout.strip()


def commit(repo: Path, files: dict[str, str], msg: str = "c") -> str:
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "clone"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    return r


def unit(root: str, built: str):
    return SimpleNamespace(rootPath=root, builtCommit=built)


def test_the_same_commit_is_fresh_in_both_modes(repo):
    c1 = commit(repo, {"a/x": "1"})
    for mode in S.MODES:
        s = S.unit_staleness(unit("a", c1), c1, repo=repo, mode=mode)
        assert (s.stale, s.commits_behind, s.files_changed) == (False, 0, 0) and s.head == c1


def test_files_mode_looks_only_under_the_unit_root(repo):
    c1 = commit(repo, {"a/x": "1", "b/y": "1"})
    c2 = commit(repo, {"b/y": "2"})                                      # touches another directory only
    assert S.unit_staleness(unit("a", c1), c2, repo=repo).stale is False
    s = S.unit_staleness(unit("b", c1), c2, repo=repo)
    assert (s.stale, s.files_changed, s.commits_behind, s.basis) == (True, 1, None, "files")


def test_files_mode_root_dot_is_the_whole_repository_and_a_file_root_is_that_file(repo):
    c1 = commit(repo, {"a/x": "1", "b/y": "1"})
    c2 = commit(repo, {"b/y": "2"})
    assert S.unit_staleness(unit(".", c1), c2, repo=repo).files_changed == 1
    assert S.unit_staleness(unit("b/y", c1), c2, repo=repo).stale is True
    assert S.unit_staleness(unit("a/x", c1), c2, repo=repo).stale is False


def test_a_root_that_is_a_prefix_of_a_sibling_name_does_not_match_it(repo):
    c1 = commit(repo, {"app/x": "1", "app2/x": "1"})
    c2 = commit(repo, {"app2/x": "2"})
    assert S.unit_staleness(unit("app", c1), c2, repo=repo).stale is False


def test_commits_mode_counts_commits_even_when_no_file_under_the_root_changed(repo):
    c1 = commit(repo, {"a/x": "1", "b/y": "1"})
    commit(repo, {"b/y": "2"})
    c3 = commit(repo, {"b/y": "3"})
    s = S.unit_staleness(unit("a", c1), c3, repo=repo, mode="commits")
    assert (s.stale, s.commits_behind, s.files_changed, s.basis) == (True, 2, None, "commits")


def test_the_head_is_the_checked_out_head_of_the_local_clone(repo):
    commit(repo, {"a": "1"})
    c2 = commit(repo, {"a": "2"})
    assert S.local_head(repo) == c2 and len(c2) == 40


def test_a_built_commit_the_clone_lacks_is_stale_and_says_so(repo):
    c1 = commit(repo, {"a/x": "1"})
    s = S.unit_staleness(unit("a", "f" * 40), c1, repo=repo)
    assert (s.stale, s.basis, s.files_changed, s.commits_behind) == (True, "unknown-commit", None, None)


@pytest.mark.parametrize("bad", ["HEAD", "main", "--all", "abc", "A" * 40, "", None, 7])
def test_commits_are_full_lowercase_ids_for_both_arguments(repo, bad):
    c1 = commit(repo, {"a": "1"})
    with pytest.raises(ValueError):
        S.unit_staleness(unit("a", bad), c1, repo=repo)
    with pytest.raises(ValueError):
        S.unit_staleness(unit("a", c1), bad, repo=repo)


def test_an_unknown_mode_is_refused(repo):
    c1 = commit(repo, {"a": "1"})
    with pytest.raises(ValueError):
        S.unit_staleness(unit("a", c1), c1, repo=repo, mode="remote")


def test_a_pathspec_is_literal_so_a_glob_root_matches_only_that_name(repo):
    c1 = commit(repo, {"a/x": "1", "*/y": "1"})
    c2 = commit(repo, {"a/x": "2"})
    assert S.unit_staleness(unit("*", c1), c2, repo=repo).stale is False          # `*` is a directory name here, not a wildcard


def test_only_local_read_only_git_subcommands_run_and_no_remote_is_configured(repo, monkeypatch):
    c1 = commit(repo, {"a/x": "1"})
    c2 = commit(repo, {"a/x": "2"})
    seen = []
    real = S.gitsafe.git
    monkeypatch.setattr(S.gitsafe, "git", lambda r, *a, **k: (seen.append(a[0]), real(r, *a, **k))[1])
    S.local_head(repo)
    S.unit_staleness(unit("a", c1), c2, repo=repo)
    S.unit_staleness(unit("a", c1), c2, repo=repo, mode="commits")
    assert set(seen) == {"rev-parse", "diff", "rev-list"}
    assert git(repo, "remote") == "" and S.HEAD_NOTE == "head is the local clone's"
