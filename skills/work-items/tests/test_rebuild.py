"""Step 7: ``wi rebuild --from-headers`` (AC-12).  Everything lives under tmp_path git repos."""
import json
import subprocess
import sys

import pytest

from conftest import SCRIPTS
from wilib import alloc, headers, move, project
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
    done = wi(lost, "rebuild", "--from-headers", ".", "--write")
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
    assert wi(lost, "rebuild", "--from-headers", ".", "--write").returncode == 0
    assert alloc.new(lost, "F") == "GAL-F0003"             # F0001 (alias) and F0002 are taken
    assert alloc.new(lost, "I") == "GAL-I05"               # I03, I04 seen only through aliases/paths


def test_rebuild_restores_the_tracked_copy_too_and_is_stable(lost):
    lose_both_copies(lost)
    wi(lost, "rebuild", "--from-headers", ".", "--write")
    assert Store(lost, "GAL").registry_path.read_text().count("\n") > 0
    again = wi(lost, "rebuild", "--from-headers", ".", "--write")
    assert again.returncode == 1 and "not lost" in again.stderr


def test_rebuild_refuses_when_only_one_copy_is_deleted(lost):
    Store(lost, "GAL").registry_path.unlink()               # the journal still holds everything
    r = wi(lost, "rebuild", "--from-headers", ".", "--write")
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
    assert wi(root, "rebuild", "--from-headers", ".", "--write").returncode == 0
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
