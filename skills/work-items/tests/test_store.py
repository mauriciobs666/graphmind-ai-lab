"""Journal, tracked registry, lock, ts, reseed, flush (plan 3.1)."""
import builtins
import io
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from conftest import SCRIPTS, git
from wilib import events, store as store_mod
from wilib.store import Store

_n = iter(range(1, 10**6))


def mk(evid, id_="GAL-F0001", ts=None):
    return {"evid": evid, "ts": ts if ts is not None else next(_n), "ev": "new", "id": id_,
            "type": "F", "parent": None, "home": "repo"}


def line(ev):
    return events.dump_event(ev)


def journal_ids(s):
    return [e["evid"] for e in events.read_events(s.journal_path.read_text())]


def tracked_ids(root):
    p = root / ".work-items" / "registry.jsonl"
    return [e["evid"] for e in events.read_events(p.read_text())] if p.exists() else []


def alloc_feature(s):
    with s.transaction() as tx:
        n = tx.state.next_number("F")
        id_ = f"GAL-F{n:04d}"
        tx.append({"ev": "new", "id": id_, "type": "F", "parent": None, "home": "repo"})
    return id_


# ------------------------------------------------------------- journal
def test_journal_lives_in_git_common_dir(repo):
    s = Store(repo, "GAL")
    alloc_feature(s)
    assert s.journal_path == repo / ".git" / "work-items" / "GAL" / "journal.jsonl"
    assert s.journal_path.exists()


def test_append_never_touches_tracked_file(repo):
    s = Store(repo, "GAL")
    alloc_feature(s)
    assert not (repo / ".work-items" / "registry.jsonl").exists()
    assert git(repo, "status", "--porcelain") == ""


def test_journal_shared_across_worktrees(repo, tmp_path):
    (repo / "f").write_text("x")
    git(repo, "add", "f")
    git(repo, "commit", "-qm", "c")
    wt = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "-b", "b", str(wt))
    a, b = Store(repo, "GAL"), Store(wt, "GAL")
    assert a.journal_path == b.journal_path
    assert b.journal_path.is_absolute()
    ida = alloc_feature(a)
    idb = alloc_feature(b)
    assert ida != idb
    with b.transaction() as tx:
        assert {ida, idb} <= set(tx.state.items)


def test_evid_and_ts_stamped_by_store(repo):
    s = Store(repo, "GAL", clock=lambda: 1000)
    with s.transaction() as tx:
        ev = tx.append({"ev": "new", "id": "GAL-F0001", "type": "F", "parent": None, "home": "repo"})
    assert ev["ts"] == 1000 and ev["evid"]
    events.validate_event(ev)


def test_ts_survives_backward_clock(repo):
    s = Store(repo, "GAL", clock=lambda: 5000)
    alloc_feature(s)
    s2 = Store(repo, "GAL", clock=lambda: 10)       # clock stepped back
    with s2.transaction() as tx:
        ev = tx.append({"ev": "retire", "uid": "GAL-F0001"})
    assert ev["ts"] == 5001


def test_transaction_state_sees_own_appends(repo):
    s = Store(repo, "GAL")
    with s.transaction() as tx:
        tx.append({"ev": "new", "id": "GAL-F0001", "type": "F", "parent": None, "home": "repo"})
        assert tx.state.next_number("F") == 2


def test_read_state_without_events_is_empty(repo):
    assert Store(repo, "GAL").read_state().items == {}


# ------------------------------------------------------------- concurrency
WORKER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {scripts!r})
    from wilib.store import Store
    root, go, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    import os
    while not os.path.exists(go):
        pass
    s = Store(root, "GAL")
    out = []
    for _ in range(n):
        with s.transaction() as tx:
            k = tx.state.next_number("F")
            i = "GAL-F%04d" % k
            tx.append({{"ev": "new", "id": i, "type": "F", "parent": None, "home": "repo"}})
        out.append(i)
    print(" ".join(out))
""")


def test_eight_processes_allocate_distinct_ids(repo, tmp_path):
    go = tmp_path / "go"
    procs = [subprocess.Popen([sys.executable, "-c", WORKER.format(scripts=str(SCRIPTS)),
                               str(repo), str(go), "25"], stdout=subprocess.PIPE, text=True)
             for _ in range(8)]
    go.write_text("go")
    ids = []
    for p in procs:
        out, _ = p.communicate(timeout=120)
        assert p.returncode == 0
        ids += out.split()
    assert len(ids) == 200 and len(set(ids)) == 200
    assert len(journal_ids(Store(repo, "GAL"))) == 200


# ------------------------------------------------------------- flush
def test_flush_copies_journal_to_tracked(repo):
    s = Store(repo, "GAL")
    alloc_feature(s)
    alloc_feature(s)
    s.flush()
    assert tracked_ids(repo) == journal_ids(s)
    assert len(tracked_ids(repo)) == 2


def test_flush_is_idempotent(repo):
    s = Store(repo, "GAL")
    alloc_feature(s)
    s.flush()
    first = (repo / ".work-items" / "registry.jsonl").read_bytes()
    s.flush()
    assert (repo / ".work-items" / "registry.jsonl").read_bytes() == first


def test_flush_appends_only_missing_and_keeps_tracked_order(repo):
    s = Store(repo, "GAL")
    alloc_feature(s)
    s.flush()
    before = tracked_ids(repo)
    alloc_feature(s)
    s.flush()
    after = tracked_ids(repo)
    assert after[:1] == before and len(after) == 2


def test_flush_copies_other_branches_events_too(repo):
    s = Store(repo, "GAL")
    alloc_feature(s)
    s.flush()
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "c")
    git(repo, "checkout", "-qb", "other")
    alloc_feature(s)              # journal only
    git(repo, "checkout", "-q", "main")
    s.flush()                     # from main: still copies other's event
    assert len(tracked_ids(repo)) == 2


def test_flush_drops_duplicate_evid_lines_keeping_first(repo):
    reg = repo / ".work-items" / "registry.jsonl"
    reg.parent.mkdir()
    e0, a1, b1, a2, b2 = (mk(x, f"GAL-F{i:04d}") for i, x in enumerate("e0 a1 b1 a2 b2".split(), 1))
    reg.write_text("".join(line(e) for e in [e0, a1, b1, a2, b1, a1, a2, b2]))
    Store(repo, "GAL").flush()
    assert tracked_ids(repo) == ["e0", "a1", "b1", "a2", "b2"]


def test_flush_splits_glued_lines(repo):
    reg = repo / ".work-items" / "registry.jsonl"
    reg.parent.mkdir()
    reg.write_text(line(mk("x1", "GAL-F0001")).rstrip() + line(mk("x2", "GAL-F0002")))
    Store(repo, "GAL").flush()
    assert len(reg.read_text().splitlines()) == 2


def test_differing_order_union_merge_dedup(repo):
    """Plan test 6: A flushes a1 b1 a2, B flushes a1 a2 b2; union merge duplicates."""
    (repo / ".gitattributes").write_text(".work-items/registry.jsonl merge=union\n")
    reg = repo / ".work-items" / "registry.jsonl"
    reg.parent.mkdir()
    ev = {k: mk(k, f"GAL-F{i:04d}") for i, k in enumerate(["e0", "a1", "b1", "a2", "b2"], 1)}
    reg.write_text(line(ev["e0"]))
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    git(repo, "checkout", "-qb", "A")
    reg.write_text("".join(line(ev[k]) for k in ["e0", "a1", "b1", "a2"]))
    git(repo, "commit", "-qam", "A")
    git(repo, "checkout", "-q", "main")
    git(repo, "checkout", "-qb", "B")
    reg.write_text("".join(line(ev[k]) for k in ["e0", "a1", "a2", "b2"]))
    git(repo, "commit", "-qam", "B")
    git(repo, "merge", "-q", "--no-edit", "A")
    merged = reg.read_text().splitlines()
    assert len(merged) > len(set(merged))            # the duplication is real
    Store(repo, "GAL").flush()
    dups = subprocess.run("sort .work-items/registry.jsonl | uniq -d", shell=True, cwd=repo,
                          capture_output=True, text=True).stdout
    assert dups == ""
    lines = reg.read_text().splitlines()
    assert len(set(lines)) == len(lines) == 5
    assert set(tracked_ids(repo)) == {"e0", "a1", "b1", "a2", "b2"}


def test_branches_allocate_distinct_after_flush_and_merge(repo, tmp_path):
    (repo / ".gitattributes").write_text(".work-items/registry.jsonl merge=union\n")
    (repo / "f").write_text("x")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "c")
    wt = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "-b", "B", str(wt))
    a, b = Store(repo, "GAL"), Store(wt, "GAL")
    ids = [alloc_feature(a), alloc_feature(b), alloc_feature(a), alloc_feature(b)]
    assert len(set(ids)) == 4
    a.flush()
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "A")
    b.flush()
    git(wt, "add", "-A")
    git(wt, "commit", "-qm", "B")
    git(repo, "merge", "-q", "--no-edit", "B")
    a.flush()
    assert set(e["id"] for e in events.read_events(
        (repo / ".work-items" / "registry.jsonl").read_text())) == set(ids)


# ------------------------------------------------------------- reseed
def test_reseed_from_tracked_file_when_journal_missing(repo):
    s = Store(repo, "GAL")
    alloc_feature(s)
    s.flush()
    import shutil
    shutil.rmtree(repo / ".git" / "work-items")
    assert alloc_feature(Store(repo, "GAL")) == "GAL-F0002"


def test_reseed_appends_tracked_events_missing_from_journal(repo):
    s = Store(repo, "GAL")
    alloc_feature(s)
    reg = repo / ".work-items" / "registry.jsonl"
    reg.parent.mkdir()
    reg.write_text(line(mk("t1", "GAL-F0007", ts=1)))     # arrived via a merge
    assert alloc_feature(s) == "GAL-F0008"
    assert "t1" in journal_ids(s)


def test_reseed_from_ref_after_journal_deletion(repo):
    s = Store(repo, "GAL")
    alloc_feature(s)
    alloc_feature(s)
    s.flush()
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "reg")
    git(repo, "checkout", "-qb", "feature")
    (repo / ".work-items" / "registry.jsonl").unlink()     # working copy lacks it
    git(repo, "commit", "-qam", "drop")
    import shutil
    shutil.rmtree(repo / ".git" / "work-items")
    assert not (repo / ".work-items" / "registry.jsonl").exists()
    # main's committed registry is only reachable through the ref
    assert alloc_feature(Store(repo, "GAL")) == "GAL-F0003"


def test_reseed_from_remote_ref(repo, tmp_path):
    s = Store(repo, "GAL")
    alloc_feature(s)
    s.flush()
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "reg")
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(repo), str(clone))
    git(clone, "checkout", "-qb", "nothing", "HEAD")
    git(clone, "rm", "-q", "-r", ".work-items")
    git(clone, "commit", "-qm", "rm")
    git(clone, "branch", "-D", "main")            # only refs/remotes/origin/main keeps it
    assert not (clone / ".work-items").exists()
    assert alloc_feature(Store(clone, "GAL")) == "GAL-F0002"


def test_ref_scan_only_on_missing_journal(repo):
    """With a live journal, refs are not consulted (reseed is 'once, on recovery')."""
    s = Store(repo, "GAL")
    alloc_feature(s)
    s.flush()
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "reg")
    # a ref event never seen by the journal or the tracked file
    reg = repo / ".work-items" / "registry.jsonl"
    reg.write_text(reg.read_text() + line(mk("ghost", "GAL-F0009", ts=1)))
    git(repo, "commit", "-qam", "ghost")
    git(repo, "checkout", "-q", "HEAD~1")        # working copy lacks ghost
    assert alloc_feature(Store(repo, "GAL")) == "GAL-F0002"


# ------------------------------------------------------------- non-git
def test_non_git_mode_writes_tracked_file_directly(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    s = Store(root, "GAL")
    assert s.journal_path is None
    assert alloc_feature(s) == "GAL-F0001"
    assert alloc_feature(s) == "GAL-F0002"
    assert tracked_ids(root) and len(tracked_ids(root)) == 2
    s.flush()                                     # no-op, no crash
    assert len(tracked_ids(root)) == 2
    assert alloc_feature(Store(root, "GAL")) == "GAL-F0003"


def test_non_git_then_git_init_reseeds_from_tracked(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    alloc_feature(Store(root, "GAL"))
    git(root, "init", "-q")
    assert alloc_feature(Store(root, "GAL")) == "GAL-F0002"


# ------------------------------------------------------------- AC-5
def test_store_reads_only_registry_files(repo, tmp_path):
    docs = repo / "docs" / "plans"
    docs.mkdir(parents=True)
    (docs / "x.md").write_text("**Item:** `GAL-F0500`\n")
    (repo / "docs" / "BACKLOG.md").write_text("GAL-F9999\n")
    locked = repo / "docs" / "secret.md"
    locked.write_text("nope")
    locked.chmod(0)
    s = Store(repo, "GAL")
    seen = []
    real_open, real_io_open, real_os_open, real_scandir = builtins.open, io.open, os.open, os.scandir

    def rec(fn):
        def inner(path, *a, **k):
            if isinstance(path, (str, bytes, os.PathLike)):
                seen.append(os.fsdecode(path))
            return fn(path, *a, **k)
        return inner

    mp = pytest.MonkeyPatch()
    mp.setattr(builtins, "open", rec(real_open))
    mp.setattr(io, "open", rec(real_io_open))
    mp.setattr(os, "open", rec(real_os_open))
    mp.setattr(os, "scandir", rec(real_scandir))
    try:
        assert alloc_feature(s) == "GAL-F0001"
        s.flush()
        s.read_state()
    finally:
        mp.undo()
        locked.chmod(0o644)
    assert seen
    allowed = (str(repo / ".work-items"), str(repo / ".git" / "work-items"))
    stray = [p for p in seen if not os.path.abspath(p).startswith(allowed)]
    assert stray == []


def test_append_after_missing_trailing_newline_keeps_one_event_per_line(tmp_path):
    root = tmp_path / "plain"
    (root / ".work-items").mkdir(parents=True)
    reg = root / ".work-items" / "registry.jsonl"
    reg.write_text(line(mk("n1", "GAL-F0001")).rstrip("\n"))
    alloc_feature(Store(root, "GAL"))
    assert len(reg.read_text().splitlines()) == 2


def test_noop_flush_does_not_rewrite_the_file(repo):
    s = Store(repo, "GAL")
    alloc_feature(s)
    s.flush()
    reg = repo / ".work-items" / "registry.jsonl"
    os.utime(reg, ns=(1, 1))
    s.flush()
    assert reg.stat().st_mtime_ns == 1
