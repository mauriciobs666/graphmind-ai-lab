import subprocess
import sys
from pathlib import Path

from conftest import SCRIPTS, git
from wilib import events, project
from wilib.store import Store

CHILD = (
    "import sys; sys.path.insert(0, sys.argv[1])\n"
    "from wilib import alloc\n"
    "for _ in range(int(sys.argv[3])):\n"
    "    print(alloc.new(sys.argv[2], 'F'))\n"
)


def test_ac2_eight_processes_allocate_200_distinct_ids(repo):
    project.init(repo, "GAL", ["repo"])
    procs = [subprocess.Popen([sys.executable, "-c", CHILD, str(SCRIPTS), str(repo), "25"],
                              stdout=subprocess.PIPE, text=True) for _ in range(8)]
    ids = []
    for p in procs:
        out, _ = p.communicate()
        assert p.returncode == 0
        ids += out.split()
    assert len(ids) == 200 and len(set(ids)) == 200
    assert set(ids) == {f"GAL-F{n:04d}" for n in range(1, 201)}
    assert len(Store(repo, "GAL").read_state().items) == 202   # + I01, I02


def wi(cwd, *args):
    r = subprocess.run([sys.executable, str(SCRIPTS / "wi.py"), *args], cwd=cwd,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_ac2_two_tasks_under_one_story_on_two_worktrees_merge(repo, tmp_path):
    project.init(repo, "GAL", ["repo"])
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "init")
    wt = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "-b", "b", str(wt))
    feature = wi(repo, "new", "F")
    story = wi(repo, "new", "S", "--parent", feature)
    assert (feature, story) == ("GAL-F0001", "GAL-F0001-S01")
    ta = wi(repo, "new", "T", "--parent", story)
    tb = wi(wt, "new", "T", "--parent", story)       # sees A's allocations via the journal
    assert (ta, tb) == ("GAL-F0001-S01-T01", "GAL-F0001-S01-T02")
    for root, name in ((repo, "a"), (wt, "b")):
        Store(root, "GAL").flush()
        git(root, "add", ".work-items/registry.jsonl")
        git(root, "commit", "-qm", name)
    git(repo, "merge", "-q", "--no-edit", "b")
    Store(repo, "GAL").flush()
    text = (repo / ".work-items" / "registry.jsonl").read_text()
    evs = events.read_events(text)
    assert len({e["evid"] for e in evs}) == len(text.splitlines())
    ids = {e["id"] for e in evs if e["ev"] == "new"}
    assert {ta, tb, story, feature} <= ids
