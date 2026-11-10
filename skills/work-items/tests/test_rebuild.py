"""Step 7: ``wi rebuild --from-headers`` (AC-12).  Everything lives under tmp_path git repos."""
import json
import subprocess
import sys

import pytest

from conftest import SCRIPTS, git
from wilib import alloc, events, headers, legacy, links, move, project
from wilib.store import Store

WI = SCRIPTS / "wi.py"
HEADER = "> **Status:** active · **Owner:** `tico` · **Tracks:** —"


def wi(cwd, *args):
    return subprocess.run([sys.executable, str(WI), *args], cwd=cwd, capture_output=True, text=True)


def doc(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# Title\n\n{HEADER}\n\nbody\n", encoding="utf-8")
    return p


@pytest.fixture
def proj(repo):
    project.init(repo, "GAL", ["a"])
    cfg = repo / ".work-items" / "project.json"
    data = json.loads(cfg.read_text())
    data["legacy_namespaces"] = {"K-": ["architect"]}
    cfg.write_text(json.dumps(data))
    return repo


def lose_both_copies(root):
    store = Store(root, "GAL")
    store.journal_path.unlink()
    store.registry_path.unlink()
    assert not store.journal_path.exists() and not store.registry_path.exists()


@pytest.fixture
def lost(proj):
    """A feature moved twice, a legacy-minted feature, a plain one; headers synced; both copies deleted."""
    alloc.new(proj, "I")                                   # GAL-I03
    alloc.new(proj, "I")                                   # GAL-I04
    alloc.new(proj, "F", slug="feat")                      # GAL-F0001
    for rel in ("docs/requirements/feat.md", "docs/plans/feat.md", "docs/plans/feat-coordination.md"):
        doc(proj, rel)
    move.move(proj, "GAL-F0001", "GAL-I03")                # GAL-I03-F0001
    move.move(proj, "GAL-I03-F0001", "GAL-I04")            # GAL-I04-F0001
    from wilib import legacy
    legacy.mint(proj, "architect:K-004", "K", title="old")  # GAL-I02-K0001
    alloc.new(proj, "F", slug="plain")                     # GAL-F0002
    doc(proj, "docs/plans/plain.md")
    doc(proj, "a/docs/plans/kz.md")
    # legacy item gets a slug so it has a document
    from wilib import links
    links.set_attr(proj, "GAL-I02-K0001", "slug", "kz")
    headers.sync(proj)
    return proj


def test_ac12_both_copies_deleted_then_every_header_recorded_alias_resolves(lost):
    before = wi(lost, "resolve", "GAL-F0001")
    assert before.stdout == "GAL-I04-F0001  [GAL-F0001: alias of GAL-I04-F0001]\n"
    lose_both_copies(lost)
    gone = wi(lost, "resolve", "GAL-F0001")                  # the loss is real, not vacuous
    assert (gone.returncode, gone.stdout) == (4, "")
    dry = wi(lost, "rebuild", "--from-headers", ".")
    assert dry.returncode == 0 and dry.stdout.startswith("dry run: nothing written")
    assert wi(lost, "resolve", "GAL-F0001").returncode == 4          # dry run wrote nothing
    done = wi(lost, "rebuild", "--from-headers", ".", "--write", "--accept-reissue")
    assert done.returncode == 0, done.stderr
    assert "not recoverable from headers" in done.stdout
    assert wi(lost, "resolve", "GAL-F0001").stdout == before.stdout
    assert wi(lost, "resolve", "GAL-I03-F0001").stdout == "GAL-I04-F0001  [GAL-I03-F0001: alias of GAL-I04-F0001]\n"
    assert wi(lost, "resolve", "GAL-I04-F0001").stdout == "GAL-I04-F0001\n"
    assert wi(lost, "resolve", "feat").stdout == "GAL-I04-F0001  [slug feat]\n"
    assert wi(lost, "resolve", "plain").stdout == "GAL-F0002  [slug plain]\n"
    assert wi(lost, "resolve", "architect:K-004").stdout == \
        "GAL-I02-K0001  [architect:K-004: alias of GAL-I02-K0001]\n"


def test_rebuild_counters_follow_header_ids_and_aliases(lost):
    lose_both_copies(lost)
    assert wi(lost, "rebuild", "--from-headers", ".", "--write", "--accept-reissue").returncode == 0
    assert alloc.new(lost, "F") == "GAL-F0003"             # F0001 (alias) and F0002 are taken
    assert alloc.new(lost, "I") == "GAL-I05"               # I03, I04 seen only through aliases/paths


def test_rebuild_restores_the_tracked_copy_too_and_is_stable(lost):
    lose_both_copies(lost)
    wi(lost, "rebuild", "--from-headers", ".", "--write", "--accept-reissue")
    assert Store(lost, "GAL").registry_path.read_text().count("\n") > 0
    again = wi(lost, "rebuild", "--from-headers", ".", "--write", "--accept-reissue")
    assert again.returncode == 1 and "not lost" in again.stderr


def test_rebuild_refuses_when_only_one_copy_is_deleted(lost):
    Store(lost, "GAL").registry_path.unlink()               # the journal still holds everything
    r = wi(lost, "rebuild", "--from-headers", ".", "--write", "--accept-reissue")
    assert r.returncode == 1
    assert "not lost" in r.stderr


def test_rebuild_dry_run_names_what_it_would_recover_and_what_it_cannot(lost):
    lose_both_copies(lost)
    out = wi(lost, "rebuild", "--from-headers", ".").stdout
    assert "item GAL-I04-F0001 slug=feat aliases=GAL-F0001,GAL-I03-F0001" in out
    assert "item GAL-F0002 slug=plain aliases=-" in out
    assert "stories, tasks" in out and "no document" in out


def test_rebuild_with_no_documents_recovers_nothing_and_says_so(proj):
    lose_both_copies(proj)
    out = wi(proj, "rebuild", "--from-headers", ".")
    assert "not recoverable" in out.stdout and "item " not in out.stdout


def _rebuilt_state(root):
    lose_both_copies(root)
    assert wi(root, "rebuild", "--from-headers", ".", "--write", "--accept-reissue").returncode == 0
    return Store(root, "GAL").read_state()


def test_rebuild_keeps_the_promoted_item_a_feature_not_a_kaizen(proj):
    alloc.new(proj, "K", slug="kz")                           # GAL-I02-K0001
    doc(proj, "docs/plans/kz.md")
    move.promote(proj, "GAL-I02-K0001", "F")                  # -> GAL-F0001, retype F
    headers.sync(proj)
    s = _rebuilt_state(proj)
    item = s.items["GAL-I02-K0001"]
    assert (item.id, item.type, item.aliases) == ("GAL-F0001", "F", ["GAL-I02-K0001"])
    assert alloc.new(proj, "F") == "GAL-F0002"


def test_rebuild_parent_uid_of_a_child_born_after_its_parents_move_is_the_parents_birth_id(proj):
    alloc.new(proj, "I")                                      # GAL-I03
    alloc.new(proj, "F", slug="feat")                         # GAL-F0001
    move.move(proj, "GAL-F0001", "GAL-I03")                   # GAL-I03-F0001
    alloc.new(proj, "T", parent="GAL-I03-F0001")              # born under the moved parent
    from wilib import links
    links.set_attr(proj, "GAL-I03-F0001-T01", "slug", "chore")
    doc(proj, "docs/plans/feat.md")
    doc(proj, "docs/plans/chore.md")
    headers.sync(proj)
    s = _rebuilt_state(proj)
    assert s.items["GAL-I03-F0001-T01"].parent == "GAL-F0001"   # the uid, not the current ID


# ---------------------------------------------------------------- step 7b: history, floors, guard
def commit_registry(root, msg):
    Store(root, "GAL").flush()
    git(root, "add", ".work-items")
    git(root, "commit", "-q", "-m", msg)


def lose_everything(root):
    """Delete the tracked file IN A COMMIT (no ref tip carries it) and the journal."""
    git(root, "rm", "-q", ".work-items/registry.jsonl")
    git(root, "commit", "-q", "-m", "drop registry")
    Store(root, "GAL").journal_path.unlink()
    assert not (root / ".work-items" / "registry.jsonl").exists()
    assert "registry.jsonl" not in git(root, "ls-tree", "-r", "--name-only", "HEAD")


@pytest.fixture
def committed(proj):
    """Two committed registry versions; every kind of event; then everything lost."""
    alloc.new(proj, "F", slug="feat")                          # GAL-F0001 (documented)
    alloc.new(proj, "F")                                       # GAL-F0002 (undocumented)
    alloc.new(proj, "S", parent="GAL-F0001")                   # GAL-F0001-S01
    alloc.new(proj, "S", parent="GAL-F0001")                   # GAL-F0001-S02
    legacy.mint(proj, "architect:K-004", "K", title="old")     # GAL-I02-K0001
    alloc.new(proj, "K")                                       # GAL-I02-K0002
    move.promote(proj, "GAL-I02-K0002", "F")                   # GAL-F0003
    alloc.retire(proj, "GAL-F0002")
    links.link(proj, "GAL-F0001", "relates", "GAL-F0002")
    doc(proj, "docs/plans/feat.md")
    headers.sync(proj)
    commit_registry(proj, "v1")
    alloc.new(proj, "D")                                       # GAL-I01-D0001, second version
    commit_registry(proj, "v2")
    lose_everything(proj)
    return proj


def test_history_rebuild_restores_full_state_and_counters(committed):
    gone = wi(committed, "resolve", "architect:K-004")
    assert (gone.returncode, gone.stdout) == (5, "")           # the loss is real: known, not minted
    r = wi(committed, "rebuild", "--write")
    assert r.returncode == 0, r.stderr
    assert alloc.new(committed, "F") == "GAL-F0004"
    assert alloc.new(committed, "S", parent="GAL-F0001") == "GAL-F0001-S03"
    assert alloc.new(committed, "K") == "GAL-I02-K0003"
    assert alloc.new(committed, "D") == "GAL-I01-D0002"


def test_history_rebuild_mint_legacy_returns_the_original_id(committed):
    wi(committed, "rebuild", "--write")
    assert legacy.mint(committed, "architect:K-004", "K") == "GAL-I02-K0001"


def test_history_rebuild_keeps_promotion_retirement_and_links(committed):
    wi(committed, "rebuild", "--write")
    s = Store(committed, "GAL").read_state()
    promoted = s.items["GAL-I02-K0002"]
    assert (promoted.id, promoted.type, promoted.aliases) == ("GAL-F0003", "F", ["GAL-I02-K0002"])
    assert s.items["GAL-F0002"].retired is True
    assert s.items["GAL-F0001"].retired is False
    assert s.links[("relates", "GAL-F0001")] == ["GAL-F0002"]
    assert s.items["GAL-F0001"].slug == "feat"


def test_history_union_has_no_duplicate_events(committed):
    wi(committed, "rebuild", "--write")
    store = Store(committed, "GAL")
    evids = [e["evid"] for e in events.read_events(store.journal_path.read_text())]
    assert len(evids) == len(set(evids)) > 10
    tracked = [e["evid"] for e in events.read_events(store.registry_path.read_text())]
    assert sorted(tracked) == sorted(evids)


def test_history_rebuild_reads_a_version_no_other_version_contains(proj):
    alloc.new(proj, "F")                                       # GAL-F0001
    commit_registry(proj, "v1")
    git(proj, "checkout", "-q", "-b", "side")
    alloc.new(proj, "F")                                       # GAL-F0002, committed on a side branch only
    commit_registry(proj, "side")
    git(proj, "checkout", "-q", "main")
    git(proj, "rm", "-q", ".work-items/registry.jsonl")
    git(proj, "commit", "-q", "-m", "drop")
    git(proj, "checkout", "-q", "side")
    git(proj, "rm", "-q", ".work-items/registry.jsonl")
    git(proj, "commit", "-q", "-m", "drop")
    git(proj, "checkout", "-q", "main")
    Store(proj, "GAL").journal_path.unlink()
    assert wi(proj, "rebuild", "--write").returncode == 0
    assert alloc.new(proj, "F") == "GAL-F0003"                 # --all: the side branch counts


def test_history_rebuild_adds_header_only_items_without_duplicating_history_ones(committed):
    assert wi(committed, "rebuild", "--write").returncode == 0
    alloc.new(committed, "F", slug="late")                     # GAL-F0004, never committed
    doc(committed, "docs/plans/late.md")
    headers.sync(committed)
    lose_both_copies(committed)                                # history still lacks F0004
    r = wi(committed, "rebuild", "--from-headers", ".", "--write")
    assert r.returncode == 0, r.stderr
    s = Store(committed, "GAL").read_state()
    assert s.items["GAL-F0004"].slug == "late"
    news = [e["id"] for e in events.read_events(Store(committed, "GAL").journal_path.read_text())
            if e["ev"] == "new"]
    assert news.count("GAL-F0001") == 1 and news.count("GAL-I01") == 1
    assert alloc.new(committed, "F") == "GAL-F0005"


def test_no_history_write_is_refused_with_the_reissue_risk_and_writes_nothing(proj):
    store = Store(proj, "GAL")
    store.journal_path.unlink()
    store.registry_path.unlink()
    r = wi(proj, "rebuild", "--write")
    assert r.returncode == 1
    assert "reissue" in r.stderr and "--accept-reissue" in r.stderr
    assert not store.journal_path.exists() and not store.registry_path.exists()
    assert "rebuilt" not in json.loads((proj / ".work-items" / "project.json").read_text())


def test_no_history_dry_run_warns_about_reissue(proj):
    store = Store(proj, "GAL")
    store.journal_path.unlink()
    store.registry_path.unlink()
    r = wi(proj, "rebuild")
    assert r.returncode == 0 and "reissue" in r.stdout


def test_accept_reissue_writes_and_floor_f_starts_the_counter_above_it(proj):
    store = Store(proj, "GAL")
    store.journal_path.unlink()
    store.registry_path.unlink()
    r = wi(proj, "rebuild", "--write", "--accept-reissue", "--floor", "F=6")
    assert r.returncode == 0, r.stderr
    s = store.read_state()
    assert s.items["GAL-F0006"].retired is True
    assert alloc.new(proj, "F") == "GAL-F0007"


def test_floor_for_stories_and_kaizen(lost):
    lose_both_copies(lost)
    r = wi(lost, "rebuild", "--from-headers", ".", "--write", "--accept-reissue",
           "--floor", "S:GAL-F0002=4", "--floor", "K=9")
    assert r.returncode == 0, r.stderr
    assert alloc.new(lost, "S", parent="GAL-F0002") == "GAL-F0002-S05"
    assert alloc.new(lost, "K") == "GAL-I02-K0010"


def test_floor_below_an_existing_number_is_harmless(committed):
    r = wi(committed, "rebuild", "--write", "--floor", "F=1")
    assert r.returncode == 0, r.stderr
    assert alloc.new(committed, "F") == "GAL-F0004"


@pytest.mark.parametrize("spec,msg", [("S:GAL-F9999=3", "unknown parent"), ("S=3", "parent"),
                                      ("F:GAL-F0001=3", "parent"), ("F=x", "floor"),
                                      ("F=0", "floor")])
def test_bad_floor_is_refused_and_writes_nothing(committed, spec, msg):
    r = wi(committed, "rebuild", "--write", "--floor", spec)
    assert r.returncode == 1 and msg in r.stderr
    assert not Store(committed, "GAL").journal_path.exists()


def test_non_empty_registry_is_refused_even_with_history_and_accept_reissue(proj):
    alloc.new(proj, "F")
    commit_registry(proj, "v1")
    r = wi(proj, "rebuild", "--write", "--accept-reissue")
    assert r.returncode == 1 and "not lost" in r.stderr
    assert Store(proj, "GAL").read_state().items["GAL-F0001"].type == "F"


def test_successful_write_records_the_rebuilt_marker(committed):
    wi(committed, "rebuild", "--write")
    marker = json.loads((committed / ".work-items" / "project.json").read_text())["rebuilt"]
    assert marker["history_events"] > 10 and marker["reissue_accepted"] is False


# ---------------------------------------------------------------- P9-2: stale ref tip
@pytest.fixture
def stale_tip(proj):
    """Registry v1 (F0001) committed and kept on branch ``old``; main then commits v2 (F0001-F0005),
    deletes the registry in a commit, and loses the journal: the only tip copy is the stale one."""
    alloc.new(proj, "F")
    commit_registry(proj, "v1")
    git(proj, "branch", "old")
    for _ in range(4):
        alloc.new(proj, "F")                                   # F0002..F0005
    commit_registry(proj, "v2")
    lose_everything(proj)
    return proj


def test_rebuild_applies_history_when_an_older_copy_survives_on_another_ref(stale_tip):
    r = wi(stale_tip, "rebuild", "--write")
    assert r.returncode == 0, r.stderr
    assert alloc.new(stale_tip, "F") == "GAL-F0006"            # history's maximum, not the stale tip's


def test_rebuild_over_stale_tip_leaves_no_duplicate_events(stale_tip):
    wi(stale_tip, "rebuild", "--write")
    ids = [e["evid"] for e in Store(stale_tip, "GAL").read_journal()]      # raw: no dedup applied
    assert len(ids) == len(set(ids))


def test_rebuild_still_refuses_when_the_journal_is_present(stale_tip):
    alloc.new(stale_tip, "F")                                  # a live journal again
    r = wi(stale_tip, "rebuild", "--write")
    assert r.returncode == 1 and "not lost" in r.stderr


def test_rebuild_still_refuses_when_only_the_tracked_copy_is_present(proj):
    alloc.new(proj, "F")
    commit_registry(proj, "v1")
    Store(proj, "GAL").journal_path.unlink()
    r = wi(proj, "rebuild", "--write")
    assert r.returncode == 1 and "not lost" in r.stderr
