"""Legacy minting and bare/qualified resolution (plan 3.6; FR-12, FR-20; AC-6)."""
import hashlib
import io
import json
import os
import subprocess
import sys
import builtins
import time

import pytest

from conftest import SCRIPTS
from wilib import events, project, resolve
from wilib.store import Store

WI = SCRIPTS / "wi.py"
NAMESPACES = {"K-": ["architect", "devops", "falkor-chat"], "C-": ["cpg"]}


def wi(cwd, *args):
    return subprocess.run([sys.executable, str(WI), *args], cwd=cwd, capture_output=True, text=True)


@pytest.fixture
def proj(repo):
    project.init(repo, "GAL", ["a"])
    pj = repo / ".work-items" / "project.json"
    data = json.loads(pj.read_text())
    data["legacy_namespaces"] = NAMESPACES
    pj.write_text(json.dumps(data))
    return repo


def test_namespaces_load_from_project_json(proj):
    assert project.load_namespaces(proj) == NAMESPACES


def test_ac6_two_homes_same_old_name_get_two_ids_and_remint_is_idempotent(proj):
    a = wi(proj, "mint-legacy", "architect:K-004", "--type", "K")
    d = wi(proj, "mint-legacy", "devops:K-004", "--type", "K")
    assert (a.returncode, a.stdout) == (0, "GAL-I02-K0001\n")
    assert (d.returncode, d.stdout) == (0, "GAL-I02-K0002\n")
    again = wi(proj, "mint-legacy", "architect:K-004", "--type", "K")
    assert (again.returncode, again.stdout) == (0, "GAL-I02-K0001\n")
    news = [e for e in events.read_events(Store(proj, "GAL").journal_path.read_text())
            if e["ev"] == "new" and e.get("legacy")]
    assert [e["legacy"] for e in news] == [{"home": "architect", "name": "K-004"},
                                           {"home": "devops", "name": "K-004"}]


def test_qualified_old_name_resolves_to_minted_item(proj):
    wi(proj, "mint-legacy", "architect:K-004", "--type", "K", "--title", "old thing")
    wi(proj, "mint-legacy", "devops:K-004", "--type", "K")
    r = wi(proj, "resolve", "devops:K-004", "--json")
    d = json.loads(r.stdout)
    assert (r.returncode, d["id"], d["matched_as"], d["input"]) == (
        0, "GAL-I02-K0002", "legacy", "devops:K-004")
    assert json.loads(wi(proj, "resolve", "architect:K-004", "--json").stdout)["id"] == "GAL-I02-K0001"


def test_bare_name_with_one_declared_home_resolves_after_mint_and_exits_5_before(proj):
    r = wi(proj, "resolve", "C-310")
    assert r.returncode == 5 and "cpg:C-310" in r.stderr
    assert wi(proj, "mint-legacy", "cpg:C-310", "--type", "F", "--title", "t").stdout == "GAL-F0001\n"
    r = wi(proj, "resolve", "C-310", "--json")
    d = json.loads(r.stdout)
    assert (r.returncode, d["id"], d["matched_as"]) == (0, "GAL-F0001", "bare")


def test_bare_name_with_several_declared_homes_is_ambiguous_exit_3_listing_candidates(proj):
    wi(proj, "mint-legacy", "architect:K-004", "--type", "K")
    r = wi(proj, "resolve", "K-004")
    assert r.returncode == 3
    for cand in ("architect:K-004", "devops:K-004", "falkor-chat:K-004", "GAL-I02-K0001"):
        assert cand in r.stderr


def test_ambiguity_is_driven_by_declared_homes_even_when_only_one_is_minted(proj):
    # exactly one K-004 minted; still ambiguous (3), not resolved
    wi(proj, "mint-legacy", "devops:K-004", "--type", "K")
    assert wi(proj, "resolve", "K-004").returncode == 3


def test_unknown_exits_4(proj):
    assert wi(proj, "resolve", "Z-9").returncode == 4                      # undeclared prefix
    assert wi(proj, "resolve", "cpg:K-004").returncode == 4                # home does not own prefix
    assert wi(proj, "resolve", "ghost:K-004").returncode == 4              # undeclared home


def test_qualified_known_but_unminted_exits_5(proj):
    assert wi(proj, "resolve", "architect:K-004").returncode == 5


def test_mint_refuses_undeclared_home_or_citing_only_home(proj):
    r = wi(proj, "mint-legacy", "model-bench:K-004", "--type", "K")
    assert r.returncode == 1 and "model-bench" in r.stderr
    assert wi(proj, "mint-legacy", "no-colon", "--type", "K").returncode == 1


def tree_hashes(root):
    out = {}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", ".work-items")]
        for f in files:
            p = os.path.join(dirpath, f)
            out[os.path.relpath(p, root)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def test_ac6_no_existing_document_is_edited(proj):
    for rel, text in (("docs/BACKLOG.md", "## C-310 thing\n"),
                      ("claude/architect/kaizen/plan.md", "## K-004 a\n"),
                      ("claude/devops/kaizen/plan.md", "## K-004 b\n")):
        (proj / rel).parent.mkdir(parents=True, exist_ok=True)
        (proj / rel).write_text(text)
    before = tree_hashes(proj)
    assert len(before) >= 4                                   # docs + .gitattributes
    for a in (("architect:K-004", "K"), ("devops:K-004", "K"), ("cpg:C-310", "F"),
              ("architect:K-004", "K")):
        assert wi(proj, "mint-legacy", a[0], "--type", a[1]).returncode == 0
    wi(proj, "resolve", "K-004"), wi(proj, "flush")
    assert tree_hashes(proj) == before


def test_mint_is_allocation_only_opens_nothing_outside_registry_and_journal(proj, monkeypatch):
    from wilib import legacy
    opened = []
    real_open, real_os_open = io.open, os.open

    def rec_open(f, *a, **k):
        opened.append(os.fspath(f)) if not isinstance(f, int) else None
        return real_open(f, *a, **k)

    def rec_os_open(p, *a, **k):
        opened.append(os.fspath(p))
        return real_os_open(p, *a, **k)

    monkeypatch.setattr(builtins, "open", rec_open)
    monkeypatch.setattr(io, "open", rec_open)
    monkeypatch.setattr(os, "open", rec_os_open)
    assert legacy.mint(proj, "architect:K-004", "K", title="x") == "GAL-I02-K0001"
    assert opened
    allowed = (str(proj / ".work-items"), str(proj / ".git" / "work-items"))
    stray = [p for p in opened if not os.path.realpath(p).startswith(
        tuple(os.path.realpath(a) for a in allowed))]
    assert stray == []


CHILD = (
    "import sys, time; sys.path.insert(0, sys.argv[1])\n"
    "from wilib import legacy\n"
    "while time.time() < float(sys.argv[3]): pass\n"
    "print(legacy.mint(sys.argv[2], 'architect:K-004', 'K'))\n"
)


def test_ac6_eight_concurrent_mints_of_one_legacy_item_yield_one_new_event(proj):
    start = time.time() + 1.5
    procs = [subprocess.Popen([sys.executable, "-c", CHILD, str(SCRIPTS), str(proj), str(start)],
                              stdout=subprocess.PIPE, text=True) for _ in range(8)]
    outs = []
    for p in procs:
        out, _ = p.communicate()
        assert p.returncode == 0
        outs.append(out.strip())
    assert outs == ["GAL-I02-K0001"] * 8
    evs = events.read_events(Store(proj, "GAL").journal_path.read_text())
    assert len([e for e in evs if e["ev"] == "new" and e.get("legacy")]) == 1
    assert len([e for e in evs if e["ev"] == "new" and e["type"] == "K"]) == 1


# ---- args are honoured: assert on the resulting item ----------------------
def test_cli_mint_honours_parent_and_title(proj):
    wi(proj, "new", "I")                                           # GAL-I03
    r = wi(proj, "mint-legacy", "cpg:C-310", "--type", "F", "--parent", "GAL-I03",
           "--title", "the C-310 thing")
    assert (r.returncode, r.stdout) == (0, "GAL-I03-F0001\n")
    item = Store(proj, "GAL").read_state().items["GAL-I03-F0001"]
    assert (item.title, item.parent) == ("the C-310 thing", "GAL-I03")


def test_mint_api_honours_parent_and_title(proj):
    from wilib import legacy
    wi(proj, "new", "I")
    assert legacy.mint(proj, "cpg:C-310", "F", parent="GAL-I03", title="t") == "GAL-I03-F0001"
    item = Store(proj, "GAL").read_state().items["GAL-I03-F0001"]
    assert (item.title, item.parent) == ("t", "GAL-I03")


def test_mint_home_defaults_to_declared_component_else_repo_and_can_be_overridden(proj):
    pj = proj / ".work-items" / "project.json"
    data = json.loads(pj.read_text())
    data["components"] = ["a", "cpg"]
    pj.write_text(json.dumps(data))
    wi(proj, "mint-legacy", "cpg:C-310", "--type", "F")
    wi(proj, "mint-legacy", "architect:K-004", "--type", "K")
    wi(proj, "mint-legacy", "devops:K-004", "--type", "K", "--home", "a")
    items = Store(proj, "GAL").read_state().items
    assert (items["GAL-F0001"].home, items["GAL-I02-K0001"].home, items["GAL-I02-K0002"].home) == (
        "cpg", "repo", "a")


def test_move_and_retire_of_a_legacy_name_say_to_resolve_first(proj):
    wi(proj, "mint-legacy", "architect:K-004", "--type", "K")
    for cmd in (("move", "architect:K-004", "none"), ("retire", "architect:K-004")):
        r = wi(proj, *cmd)
        assert r.returncode == 1 and "wi resolve architect:K-004" in r.stderr


# ---- plan test 11: allocation / minting leave the working tree clean -------
def test_mint_and_allocation_leave_git_status_clean(proj):
    from conftest import git
    git(proj, "add", "-A")
    git(proj, "commit", "-qm", "init")
    wi(proj, "mint-legacy", "architect:K-004", "--type", "K")
    wi(proj, "new", "F", "--slug", "x")
    wi(proj, "resolve", "K-004")
    assert git(proj, "status", "--porcelain") == ""


# ---- plan test 12: mint on A and on B, flush B, merge only B ---------------
def test_minted_id_reaches_main_with_its_new_event_when_only_b_is_merged(proj, tmp_path):
    from conftest import git
    git(proj, "add", "-A")
    git(proj, "commit", "-qm", "init")
    wt = tmp_path / "wt"
    git(proj, "worktree", "add", "-q", "-b", "B", str(wt))
    a = wi(proj, "mint-legacy", "architect:K-004", "--type", "K")      # on A (main)
    b = wi(wt, "mint-legacy", "devops:K-004", "--type", "K")           # on B
    assert (a.stdout, b.stdout) == ("GAL-I02-K0001\n", "GAL-I02-K0002\n")
    assert wi(wt, "flush").returncode == 0
    git(wt, "add", "-A")
    git(wt, "commit", "-qm", "B")
    git(proj, "merge", "-q", "--no-edit", "B")
    tracked = events.read_events((proj / ".work-items" / "registry.jsonl").read_text())
    ids = [e["id"] for e in tracked if e["ev"] == "new"]
    assert "GAL-I02-K0002" in ids                      # the merged branch's citation has its `new`
    again = wi(proj, "mint-legacy", "devops:K-004", "--type", "K")
    assert again.stdout == "GAL-I02-K0002\n"
