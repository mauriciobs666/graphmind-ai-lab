"""`wi init` minus the key claim (plan 3.9): project.json, I01/I02, baseline, .gitattributes."""
import json

import pytest

from conftest import git
from wilib import events, project, rules
from wilib.store import Store

GITATTR = ".work-items/registry.jsonl merge=union"


def test_init_writes_project_json(repo):
    project.init(repo, "GAL", ["a", "b"])
    pj = json.loads((repo / ".work-items" / "project.json").read_text())
    assert pj["key"] == "GAL" and pj["components"] == ["a", "b"]
    assert pj["legacy_namespaces"] == {}


def test_init_seeds_defects_and_kaizen_initiatives(repo):
    project.init(repo, "GAL", ["a"])
    st = Store(repo, "GAL").read_state()
    assert set(st.items) == {"GAL-I01", "GAL-I02"}
    assert st.items["GAL-I01"].title == "Defects" and st.items["GAL-I02"].title == "Kaizen"
    assert all(i.type == "I" and i.parent is None for i in st.items.values())
    assert st.next_number("I") == 3


def test_init_flushes_seeds_into_tracked_registry(repo):
    project.init(repo, "GAL", ["a"])
    evs = events.read_events((repo / ".work-items" / "registry.jsonl").read_text())
    assert sorted(e["id"] for e in evs) == ["GAL-I01", "GAL-I02"]


def test_init_records_existing_requirements_as_baseline(repo):
    for rel in ("docs/requirements/b.md", "docs/requirements/a.md", "comp/docs/requirements/c.md"):
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# x\n")
    (repo / "docs" / "plans").mkdir()
    (repo / "docs" / "plans" / "p.md").write_text("# plan\n")
    project.init(repo, "GAL", ["a"])
    assert (repo / ".work-items" / "baseline.txt").read_text().splitlines() == [
        "comp/docs/requirements/c.md", "docs/requirements/a.md", "docs/requirements/b.md"]


def test_init_empty_baseline_when_no_requirements(repo):
    project.init(repo, "GAL", ["a"])
    assert (repo / ".work-items" / "baseline.txt").read_text() == ""


def test_init_adds_gitattributes_union_line(repo):
    project.init(repo, "GAL", ["a"])
    assert (repo / ".gitattributes").read_text().splitlines() == [GITATTR]


def test_init_preserves_existing_gitattributes(repo):
    (repo / ".gitattributes").write_text("*.png binary")      # no trailing newline
    project.init(repo, "GAL", ["a"])
    assert (repo / ".gitattributes").read_text().splitlines() == ["*.png binary", GITATTR]


def test_init_is_idempotent(repo):
    project.init(repo, "GAL", ["a"])
    (repo / "docs" / "requirements").mkdir(parents=True)
    (repo / "docs" / "requirements" / "late.md").write_text("x")
    project.init(repo, "GAL", ["a"])
    assert (repo / ".gitattributes").read_text().count(GITATTR) == 1
    assert len(Store(repo, "GAL").read_state().items) == 2
    assert (repo / ".work-items" / "baseline.txt").read_text() == ""     # not regrown
    assert len((repo / ".work-items" / "registry.jsonl").read_text().splitlines()) == 2


def test_init_refuses_a_different_key(repo):
    project.init(repo, "GAL", ["a"])
    with pytest.raises(project.InitError, match="GAL"):
        project.init(repo, "ZZZ", ["a"])


@pytest.mark.parametrize("key", ["gal", "G", "TOOLONGX", "GA1", "GAL-", ""])
def test_init_refuses_malformed_key(repo, key):
    with pytest.raises(rules.MalformedId):
        project.init(repo, key, ["a"])
    assert not (repo / ".work-items").exists()


def test_init_works_without_git(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    project.init(root, "GAL", ["a"])
    assert set(Store(root, "GAL").read_state().items) == {"GAL-I01", "GAL-I02"}
    assert (root / ".gitattributes").exists()


def test_init_leaves_working_tree_commitable_and_allocation_after_init_is_i03(repo):
    project.init(repo, "GAL", ["a"])
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    s = Store(repo, "GAL")
    with s.transaction() as tx:
        assert tx.state.next_number("I") == 3
    assert git(repo, "status", "--porcelain") == ""


def test_init_ignores_the_lock_file_so_a_later_git_add_never_commits_it(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    project.init(root, "GAL", ["a"])
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    assert (root / ".work-items" / ".lock").exists()
    out = subprocess.run(["git", "status", "--porcelain", "-uall"], cwd=root,
                         capture_output=True, text=True).stdout
    assert ".lock" not in out and "project.json" in out
