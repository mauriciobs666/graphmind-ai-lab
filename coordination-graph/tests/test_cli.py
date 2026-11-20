"""The offline CLI: `lint [--count]`, `facts`, `history-coverage`, `migrate` argument handling."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from coordgraph.cli import main

COMPONENT = Path(__file__).resolve().parents[1]
PROJECT = {"key": "GAL", "legacy_paths": {"*/docs/BACKLOG.md": "$1", "docs/BACKLOG.md": "cpg"}}


def put(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    put(tmp_path, ".work-items/project.json", json.dumps(PROJECT))
    put(tmp_path, "falkor-chat/docs/BACKLOG.md",
        "# B\n\n### K-001 — One\n\n- **Status:** 🔵 proposed\n"
        "- **Touches:** fact:Falkor-Chat/x.py#Foo (relies); file:a.py\n\n"
        "### K-002 — Two\n\n- **Status:** 🔵 proposed\n- **Touches:** fact:other.py#bar (changes)\n")
    put(tmp_path, "falkor-chat/docs/HISTORY.md", "# H\n\n## 2026-01-01 — K-010 closed: a\n\nmentions K-011 and K-001\n")
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(tmp_path))
    return tmp_path


def test_lint_count_prints_items_per_file_and_a_total(repo, capsys):
    assert main(["lint", "--count"]) == 0
    assert capsys.readouterr().out.splitlines() == ["falkor-chat/docs/BACKLOG.md\t2", "total\t2"]


def test_lint_exit_code_follows_errors_not_warnings(repo, capsys):
    assert main(["lint", "falkor-chat/docs/BACKLOG.md"]) == 0
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n### K-001 — One\n\n- **Status:** 🔵 proposed\n"
        "- **Depends on:** K-404\n")
    assert main(["lint"]) == 1
    assert "unresolvable-dependency" in capsys.readouterr().out


def test_facts_lists_lowercased_tokens_with_mode_and_filters(repo, capsys):
    assert main(["facts"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "falkor-chat/x.py#foo\trelies\tfalkor-chat:K-001\tfalkor-chat/docs/BACKLOG.md",
        "other.py#bar\tchanges\tfalkor-chat:K-002\tfalkor-chat/docs/BACKLOG.md"]
    assert main(["facts", "--like", "OTHER"]) == 0
    assert capsys.readouterr().out.count("\n") == 1


def test_history_coverage_prints_matched_and_unmatched_per_file(repo, capsys):
    assert main(["history-coverage", "-v"]) == 0
    out = capsys.readouterr().out
    # K-010 delivered (matched); K-011 mentioned only (unmatched); K-001 is open in the backlog (excluded)
    assert "falkor-chat/docs/HISTORY.md\tmentioned 3\topen 1\tmatched 1\tunmatched 1\t50%" in out
    assert "unmatched K-011" in out and out.splitlines()[-1] == "total\tmatched 1\tunmatched 1\t50%"


def test_apply_takes_exactly_one_covered_path(repo, capsys):
    assert main(["migrate", "--apply"]) == 2
    assert main(["migrate", "--apply", "falkor-chat/docs/BACKLOG.md", "docs/BACKLOG.md"]) == 2
    assert main(["migrate", "--apply", "falkor-chat/docs/plans/x.md"]) == 2
    assert "one file per run" in capsys.readouterr().err


def test_migrate_needs_a_mode(repo):
    with pytest.raises(SystemExit) as exc:
        main(["migrate"])
    assert exc.value.code == 2


def test_python_dash_m_and_the_shell_wrapper_dispatch_to_the_same_cli(repo, tmp_path):
    env = {**os.environ, "COORDGRAPH_REPO_ROOT": str(repo)}
    run = subprocess.run([sys.executable, "-m", "coordgraph", "lint", "--count"], cwd=COMPONENT, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0 and run.stdout.splitlines()[-1] == "total\t2"
    home = tmp_path / "comp"
    (home / ".venv" / "bin").mkdir(parents=True)
    (home / "coordgraph.sh").write_bytes((COMPONENT / "coordgraph.sh").read_bytes())
    shim = home / ".venv" / "bin" / "python"
    shim.write_text(f'#!/usr/bin/env bash\nPYTHONPATH={COMPONENT} exec {sys.executable} "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    (home / "coordgraph.sh").chmod(0o755)
    wrapped = subprocess.run([str(home / "coordgraph.sh"), "lint", "--count"], env=env, capture_output=True, text=True)
    assert wrapped.returncode == 0 and wrapped.stdout == run.stdout
