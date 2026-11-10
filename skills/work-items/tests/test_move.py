"""move / promote: subtree rename, aliases, local numbers, legality (FR-4, FR-22, FR-24; AC-3, AC-9, AC-17)."""
import pytest

from wilib import alloc, move, project, resolve, rules
from wilib.store import Store


@pytest.fixture
def proj(repo):
    project.init(repo, "GAL", ["a"])
    alloc.new(repo, "I")                                   # GAL-I03
    alloc.new(repo, "I")                                   # GAL-I04
    f = alloc.new(repo, "F", parent="GAL-I03", slug="feat")    # GAL-I03-F0001
    s1 = alloc.new(repo, "S", parent=f)                    # -S01
    s2 = alloc.new(repo, "S", parent=f)                    # -S02
    alloc.new(repo, "T", parent=s1)                        # -S01-T01
    alloc.new(repo, "T", parent=s2)                        # -S02-T01
    alloc.new(repo, "T", parent=s2)                        # -S02-T02
    alloc.new(repo, "T", parent=f)                         # -T01 directly under feature
    return repo


def st(root):
    return Store(root, "GAL").read_state()


def r(root, name):
    return resolve.resolve(st(root), name)


def events_count(root):
    return len(Store(root, "GAL").journal_path.read_text().splitlines())


# ---- AC-3: feature to another initiative -----------------------------------
def test_feature_move_renames_item_and_whole_subtree(proj):
    assert move.move(proj, "GAL-I03-F0001", "GAL-I04") == "GAL-I04-F0001"
    s = st(proj)
    assert sorted(i.id for i in s.items.values() if i.type in "FST") == [
        "GAL-I04-F0001", "GAL-I04-F0001-S01", "GAL-I04-F0001-S01-T01",
        "GAL-I04-F0001-S02", "GAL-I04-F0001-S02-T01", "GAL-I04-F0001-S02-T02",
        "GAL-I04-F0001-T01"]


def test_feature_move_keeps_local_numbers_not_renumbered(proj):
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    assert r(proj, "GAL-I03-F0001-S02-T02")["id"] == "GAL-I04-F0001-S02-T02"
    assert r(proj, "GAL-I03-F0001-S01")["id"] == "GAL-I04-F0001-S01"


def test_every_old_id_resolves_and_reports_alias(proj):
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    for old, new in [("GAL-I03-F0001", "GAL-I04-F0001"),
                     ("GAL-I03-F0001-S01", "GAL-I04-F0001-S01"),
                     ("GAL-I03-F0001-S02-T01", "GAL-I04-F0001-S02-T01"),
                     ("GAL-I03-F0001-T01", "GAL-I04-F0001-T01")]:
        rec = r(proj, old)
        assert (rec["id"], rec["matched_as"], rec["input"]) == (new, "alias", old)


def test_second_move_resolves_from_the_first_id(proj):
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    assert move.move(proj, "GAL-I04-F0001", "none") == "GAL-F0001"
    for old in ("GAL-I03-F0001-S02-T02", "GAL-I04-F0001-S02-T02"):
        rec = r(proj, old)
        assert (rec["id"], rec["matched_as"]) == ("GAL-F0001-S02-T02", "alias")
    assert sorted(r(proj, "GAL-F0001")["aliases"]) == ["GAL-I03-F0001", "GAL-I04-F0001"]
    assert r(proj, "GAL-I03-F0001")["id"] == "GAL-F0001"


def test_move_sets_parent_uid_and_event_shape(proj):
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    assert r(proj, "GAL-I04-F0001")["parent"] == "GAL-I04"
    ev = [e for e in Store(proj, "GAL").journal_path.read_text().splitlines() if '"move"' in e]
    assert len(ev) == 1


def test_move_accepts_alias_as_item_and_parent(proj):
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    alloc.new(proj, "I")                                   # GAL-I05
    assert move.move(proj, "GAL-I03-F0001", "GAL-I05") == "GAL-I05-F0001"


# ---- AC-17: FR/AC citations -------------------------------------------------
def test_ac17_requirement_citations_follow_the_feature(proj):
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    for old, new in [("GAL-I03-F0001-FR2a", "GAL-I04-F0001-FR2a"),
                     ("GAL-I03-F0001-AC21a", "GAL-I04-F0001-AC21a"),
                     ("GAL-I03-F0001-AC3", "GAL-I04-F0001-AC3")]:
        rec = r(proj, old)
        assert (rec["id"], rec["matched_as"], rec["input"]) == (new, "alias", old)
    assert r(proj, "GAL-I04-F0001-AC21a")["matched_as"] == "current"


def test_ac17_citation_survives_two_moves(proj):
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    move.move(proj, "GAL-I04-F0001", "none")
    assert r(proj, "GAL-I03-F0001-AC21a")["id"] == "GAL-F0001-AC21a"
    assert r(proj, "GAL-I04-F0001-FR2a")["id"] == "GAL-F0001-FR2a"


# ---- FR-22: stories and tasks take the next free local number --------------
@pytest.fixture
def two_features(proj):
    alloc.new(proj, "F", slug="other")                     # GAL-F0002
    alloc.new(proj, "S", parent="GAL-F0002")               # GAL-F0002-S01
    return proj


def test_story_move_takes_next_free_local_number_and_subtree_follows(two_features):
    p = two_features
    assert move.move(p, "GAL-I03-F0001-S02", "GAL-F0002") == "GAL-F0002-S02"
    assert r(p, "GAL-F0002-S02-T02")["id"] == "GAL-F0002-S02-T02"
    rec = r(p, "GAL-I03-F0001-S02-T01")
    assert (rec["id"], rec["matched_as"]) == ("GAL-F0002-S02-T01", "alias")
    assert r(p, "GAL-F0002-S02")["parent"] == "GAL-F0002"


def test_story_moved_back_takes_a_fresh_number_never_its_old_one(two_features):
    p = two_features
    move.move(p, "GAL-I03-F0001-S02", "GAL-F0002")
    assert move.move(p, "GAL-F0002-S02", "GAL-I03-F0001") == "GAL-I03-F0001-S03"
    for old in ("GAL-I03-F0001-S02", "GAL-F0002-S02"):
        assert r(p, old)["id"] == "GAL-I03-F0001-S03"
    assert r(p, "GAL-I03-F0001-S02-T02")["id"] == "GAL-I03-F0001-S03-T02"


def test_task_move_takes_next_local_number(proj):
    # T under S01 (T01) moves under S02 (T01, T02 issued) -> T03
    assert move.move(proj, "GAL-I03-F0001-S01-T01", "GAL-I03-F0001-S02") == "GAL-I03-F0001-S02-T03"


def test_move_story_to_feature_whose_numbers_are_taken_by_moved_out_items(two_features):
    p = two_features
    move.move(p, "GAL-F0002-S01", "GAL-I03-F0001")         # -> F0001-S03
    alloc_id = alloc.new(p, "S", parent="GAL-F0002")
    assert alloc_id == "GAL-F0002-S02"                     # S01 was issued, even though it moved out


# ---- AC-9: promotion ---------------------------------------------------------
@pytest.fixture
def kaizen(proj):
    k = alloc.new(proj, "K", title="x")                    # GAL-I02-K0001
    alloc.new(proj, "T", parent=k)
    alloc.new(proj, "T", parent=k)
    return proj


def test_ac9_promote_kaizen_to_feature_tasks_follow(kaizen):
    p = kaizen
    assert move.promote(p, "GAL-I02-K0001", "F", parent="GAL-I04") == "GAL-I04-F0002"
    s = st(p)
    assert s.items["GAL-I02-K0001"].type == "F"
    assert sorted(i.id for i in s.items.values() if i.id.startswith("GAL-I04-F0002")) == [
        "GAL-I04-F0002", "GAL-I04-F0002-T01", "GAL-I04-F0002-T02"]
    rec = r(p, "GAL-I02-K0001-T02")
    assert (rec["id"], rec["matched_as"]) == ("GAL-I04-F0002-T02", "alias")
    assert r(p, "GAL-I02-K0001")["id"] == "GAL-I04-F0002"


def test_promote_without_initiative_and_next_feature_number(kaizen):
    p = kaizen
    assert move.promote(p, "GAL-I02-K0001", "F") == "GAL-F0002"
    assert alloc.new(p, "F") == "GAL-F0003"
    assert alloc.new(p, "T", parent="GAL-F0002") == "GAL-F0002-T03"


def test_promote_only_kaizen_to_feature(proj):
    with pytest.raises(move.MoveError):
        move.promote(proj, "GAL-I03-F0001", "F")
    k = alloc.new(proj, "K")
    with pytest.raises(move.MoveError):
        move.promote(proj, k, "S")


def test_promote_into_reserved_initiative_refused(kaizen):
    with pytest.raises(rules.IllegalParent):
        move.promote(kaizen, "GAL-I02-K0001", "F", parent="GAL-I01")


# ---- FR-25: illegal parents --------------------------------------------------
@pytest.mark.parametrize("item, parent", [
    ("GAL-I03-F0001", "GAL-I01"),                 # feature under Defects
    ("GAL-I03-F0001", "GAL-I02"),                 # feature under Kaizen
    ("GAL-I03-F0001", "GAL-I03-F0001-S01"),       # feature under a story
    ("GAL-I03-F0001-S01", "GAL-I04"),             # story under an initiative
    ("GAL-I03-F0001-S01", "none"),                # story under the project
    ("GAL-I03", "GAL-I04"),                       # initiative under an initiative
    ("GAL-I03-F0001-S01-T01", "GAL-I04"),         # task under an initiative
])
def test_illegal_parent_refused_and_nothing_written(proj, item, parent):
    before = events_count(proj)
    with pytest.raises(rules.IllegalParent):
        move.move(proj, item, parent)
    assert events_count(proj) == before


def test_defect_cannot_leave_defects_initiative(proj):
    d = alloc.new(proj, "D")
    with pytest.raises(rules.IllegalParent):
        move.move(proj, d, "GAL-I04")


def test_unknown_item_or_parent_and_noop_refused(proj):
    for args in (("GAL-F9999", "GAL-I04"), ("GAL-I03-F0001", "GAL-I99")):
        with pytest.raises(move.MoveError):
            move.move(proj, *args)
    with pytest.raises(move.MoveError):
        move.move(proj, "GAL-I03-F0001", "GAL-I03")        # already there


def test_move_into_own_subtree_refused(proj):
    # a cycle is always an FR-25 violation, so it is caught by legality
    with pytest.raises(rules.IllegalParent):
        move.move(proj, "GAL-I03-F0001", "GAL-I03-F0001-S01")


# ---- children created after a move are children of the uid, not of the old id
def test_story_created_under_moved_feature_continues_numbering_and_follows_next_move(proj):
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    assert alloc.new(proj, "S", parent="GAL-I04-F0001") == "GAL-I04-F0001-S03"
    assert alloc.new(proj, "S", parent="GAL-I03-F0001") == "GAL-I04-F0001-S04"   # via alias
    assert r(proj, "GAL-I04-F0001-S03")["parent"] == "GAL-I04-F0001"
    assert move.move(proj, "GAL-I04-F0001", "none") == "GAL-F0001"
    for old in ("GAL-I04-F0001-S03", "GAL-I04-F0001-S04"):
        assert r(proj, old)["id"] == old.replace("GAL-I04-F0001", "GAL-F0001")
        assert r(proj, old)["matched_as"] == "alias"
    assert alloc.new(proj, "S", parent="GAL-F0001") == "GAL-F0001-S05"
