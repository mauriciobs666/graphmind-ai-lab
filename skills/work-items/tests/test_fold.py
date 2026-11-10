"""Event schema, tolerant reader, deterministic fold and counters."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from wilib import events  # noqa: E402
from wilib.events import EventError  # noqa: E402

_seq = iter(range(1, 10**6))


def new(id_, type_, parent=None, ts=None, **kw):
    n = next(_seq)
    return {"evid": f"e{n:04d}", "ts": ts if ts is not None else n, "ev": "new",
            "id": id_, "type": type_, "parent": parent, "home": kw.pop("home", "repo"), **kw}


def move(uid, parent, moves, ts=None, **kw):
    n = next(_seq)
    return {"evid": f"e{n:04d}", "ts": ts if ts is not None else n, "ev": "move",
            "uid": uid, "parent": parent,
            "moves": [{"uid": u, "old": o, "new": nw} for u, o, nw in moves], **kw}


def link(rel, frm, to, ts=None, op=None):
    n = next(_seq)
    return {"evid": f"e{n:04d}", "ts": ts if ts is not None else n, "ev": "link",
            "rel": rel, "from": frm, "to": to,
            "op": op or ("set" if rel == "concerns" else "add")}


def setattr_(uid, attr, value, ts=None):
    n = next(_seq)
    return {"evid": f"e{n:04d}", "ts": ts if ts is not None else n, "ev": "set",
            "uid": uid, "attr": attr, "value": value}


def retire(uid, ts=None):
    n = next(_seq)
    return {"evid": f"e{n:04d}", "ts": ts if ts is not None else n, "ev": "retire", "uid": uid}


# ---------------------------------------------------------------- schema
def test_valid_events_pass_validation():
    f = new("GAL-F0001", "F")
    for ev in [f, new("GAL-F0002", "F", slug="x", title="T", legacy={"home": "architect", "name": "K-004"}),
               move("GAL-F0001", None, [("GAL-F0001", "GAL-F0001", "GAL-I03-F0001")], retype="F"),
               link("concerns", "GAL-I01-D0001", None), link("fixed-by", "a", "b"),
               setattr_("GAL-F0001", "milestone", "M3"), retire("GAL-F0001")]:
        events.validate_event(ev)


@pytest.mark.parametrize("mutate", [
    lambda e: e.pop("evid"), lambda e: e.pop("ts"), lambda e: e.update(ts="1"),
    lambda e: e.update(ev="delete"), lambda e: e.pop("id"), lambda e: e.pop("type"),
    lambda e: e.pop("parent"), lambda e: e.pop("home"),
    lambda e: e.update(legacy={"home": "x"}),
])
def test_invalid_new_events_refused(mutate):
    ev = new("GAL-F0001", "F")
    mutate(ev)
    with pytest.raises(EventError):
        events.validate_event(ev)


def test_invalid_other_events_refused():
    bad = [
        link("owns", "a", "b"),                                  # closed relation set
        setattr_("u", "status", "done"),                         # attr not in the set
        {**move("u", None, [("u", "a", "b")]), "moves": [{"uid": "u", "old": "a"}]},
        {**move("u", None, [("u", "a", "b")]), "moves": "x"},
        {"evid": "e", "ts": 1, "ev": "retire"},                  # no uid
        [1, 2],                                                  # not an object
    ]
    for ev in bad:
        with pytest.raises(EventError):
            events.validate_event(ev)


def test_event_ids_must_satisfy_the_grammar():
    with pytest.raises(EventError):
        events.validate_event(new("GAL-F42", "F"))
    with pytest.raises(EventError):
        events.validate_event(move("u", None, [("u", "GAL-F0001", "GAL-X1")]))


# ---------------------------------------------------------------- reader
def test_reader_reads_one_event_per_line():
    a, b = new("GAL-F0001", "F"), new("GAL-F0002", "F")
    text = events.dump_event(a) + events.dump_event(b)
    assert events.read_events(text) == [a, b]


def test_dump_event_is_one_newline_terminated_line():
    line = events.dump_event(new("GAL-F0001", "F"))
    assert line.endswith("\n") and line.count("\n") == 1


def test_reader_splits_a_glued_line_into_two_events():
    a, b, c = (new(f"GAL-F000{i}", "F") for i in (1, 2, 3))
    glued = json.dumps(a) + json.dumps(b) + "\n" + json.dumps(c) + "\n"
    assert events.read_events(glued) == [a, b, c]


def test_reader_tolerates_missing_final_newline_and_blank_lines():
    a, b = new("GAL-F0001", "F"), new("GAL-F0002", "F")
    text = json.dumps(a) + "\n\n  \n" + json.dumps(b)
    assert events.read_events(text) == [a, b]


def test_reader_glued_with_whitespace_between_objects():
    a, b = new("GAL-F0001", "F"), new("GAL-F0002", "F")
    assert events.read_events(json.dumps(a) + "  " + json.dumps(b) + "\n") == [a, b]


def test_reader_refuses_garbage_naming_the_line():
    text = events.dump_event(new("GAL-F0001", "F")) + "not json\n"
    with pytest.raises(EventError, match="line 2"):
        events.read_events(text)


def test_reader_refuses_garbage_trailing_a_valid_object():
    text = json.dumps(new("GAL-F0001", "F")) + "}}\n"
    with pytest.raises(EventError, match="line 1"):
        events.read_events(text)


def test_reader_validates_schema():
    with pytest.raises(EventError, match="line 1"):
        events.read_events('{"evid":"e","ts":1,"ev":"nope"}\n')


# ---------------------------------------------------------------- fold: state
def test_fold_builds_items_by_uid_with_current_id():
    f = new("GAL-F0001", "F", slug="x", title="X", home="falkor-chat")
    st = events.fold([f])
    it = st.items["GAL-F0001"]
    assert (it.id, it.type, it.parent, it.home, it.slug, it.title) == (
        "GAL-F0001", "F", None, "falkor-chat", "x", "X")
    assert not it.retired and it.aliases == []
    assert st.resolve_uid("GAL-F0001") == "GAL-F0001"


def test_fold_deduplicates_by_evid():
    f = new("GAL-F0001", "F")
    st = events.fold([f, dict(f), dict(f)])
    assert len(st.items) == 1 and st.next_number("F") == 2


def test_fold_applies_set_link_retire_and_legacy():
    f = new("GAL-F0001", "F", legacy={"home": "architect", "name": "K-004"})
    d = new("GAL-I01-D0001", "D", parent="I01uid")
    evs = [f, d, setattr_("GAL-F0001", "milestone", "M3"),
           setattr_("GAL-F0001", "touched", ["cpg", "claude"]),
           setattr_("GAL-F0001", "where", "docs/x.md"),
           link("concerns", d["id"], "GAL-F0001"), retire("GAL-F0001")]
    st = events.fold(evs)
    it = st.items["GAL-F0001"]
    assert (it.milestone, it.touched, it.where, it.retired) == ("M3", ["cpg", "claude"], "docs/x.md", True)
    assert st.links[("concerns", d["id"])] == ["GAL-F0001"]
    assert st.legacy[("architect", "K-004")] == "GAL-F0001"


def test_link_changed_then_cleared():
    d = new("GAL-I01-D0001", "D")
    evs = [d, link("concerns", d["id"], "a"), link("concerns", d["id"], "b")]
    assert events.fold(evs).links[("concerns", d["id"])] == ["b"]
    assert ("concerns", d["id"]) not in events.fold(evs + [link("concerns", d["id"], None)]).links


def test_move_renames_subtree_and_keeps_every_old_id_as_alias():
    i3, i4 = new("GAL-I03", "I"), new("GAL-I04", "I")
    f = new("GAL-I03-F0001", "F", parent=i3["id"])
    s = new("GAL-I03-F0001-S01", "S", parent=f["id"])
    t = new("GAL-I03-F0001-S01-T01", "T", parent=s["id"])
    m = move(f["id"], i4["id"], [
        (f["id"], "GAL-I03-F0001", "GAL-I04-F0001"),
        (s["id"], "GAL-I03-F0001-S01", "GAL-I04-F0001-S01"),
        (t["id"], "GAL-I03-F0001-S01-T01", "GAL-I04-F0001-S01-T01")])
    st = events.fold([i3, i4, f, s, t, m])
    assert [st.items[u].id for u in (f["id"], s["id"], t["id"])] == [
        "GAL-I04-F0001", "GAL-I04-F0001-S01", "GAL-I04-F0001-S01-T01"]
    assert st.items[f["id"]].parent == i4["id"]
    assert st.items[s["id"]].parent == f["id"]  # untouched descendant parent uid
    assert st.items[t["id"]].aliases == ["GAL-I03-F0001-S01-T01"]
    for old, uid in [("GAL-I03-F0001", f["id"]), ("GAL-I03-F0001-S01", s["id"]),
                     ("GAL-I03-F0001-S01-T01", t["id"]), ("GAL-I04-F0001-S01", s["id"])]:
        assert st.resolve_uid(old) == uid


def test_second_move_still_resolves_from_the_first_id():
    f = new("GAL-I03-F0001", "F")
    m1 = move(f["id"], None, [(f["id"], "GAL-I03-F0001", "GAL-I04-F0001")])
    m2 = move(f["id"], None, [(f["id"], "GAL-I04-F0001", "GAL-F0001")])
    st = events.fold([f, m1, m2])
    assert st.items[f["id"]].id == "GAL-F0001"
    assert st.items[f["id"]].aliases == ["GAL-I03-F0001", "GAL-I04-F0001"]
    assert st.resolve_uid("GAL-I03-F0001") == f["id"]


def test_move_with_retype_changes_type_and_promotion_gets_feature_number():
    k = new("GAL-I02-K0007", "K", parent="I02")
    m = move(k["id"], None, [(k["id"], "GAL-I02-K0007", "GAL-F0003")], retype="F")
    st = events.fold([k, m])
    assert st.items[k["id"]].type == "F" and st.items[k["id"]].id == "GAL-F0003"
    assert st.resolve_uid("GAL-I02-K0007") == k["id"]
    assert st.next_number("F") == 4 and st.next_number("K") == 8


def test_events_for_unknown_uid_are_reported_not_fatal():
    st = events.fold([new("GAL-F0001", "F"), retire("GAL-F0999"),
                      setattr_("GAL-F0998", "title", "x")])
    assert len(st.items) == 1
    assert [e["uid"] for e in st.orphans] == ["GAL-F0999", "GAL-F0998"]


# ---------------------------------------------------------------- fold: ordering
def test_move_stamped_before_its_new_still_applies():
    f = new("GAL-I03-F0001", "F", ts=500)
    m = move(f["id"], None, [(f["id"], "GAL-I03-F0001", "GAL-F0001")], ts=100)  # clock stepped back
    for order in ([f, m], [m, f]):
        st = events.fold(order)
        assert st.items[f["id"]].id == "GAL-F0001"
        assert not st.orphans


def test_non_new_events_apply_by_ts_then_evid():
    f = new("GAL-F0001", "F", ts=1)
    a = setattr_(f["id"], "title", "A", ts=20)
    b = setattr_(f["id"], "title", "B", ts=10)
    assert events.fold([f, a, b]).items[f["id"]].title == "A"
    # equal ts: higher evid wins, whatever the input order
    c = {**setattr_(f["id"], "title", "C", ts=30), "evid": "zzz"}
    d = {**setattr_(f["id"], "title", "D", ts=30), "evid": "aaa"}
    assert events.fold([f, d, c]).items[f["id"]].title == "C"
    assert events.fold([f, c, d]).items[f["id"]].title == "C"


def test_fold_is_independent_of_input_order():
    f = new("GAL-I03-F0001", "F")
    s = new("GAL-I03-F0001-S01", "S", parent=f["id"])
    m = move(f["id"], None, [(f["id"], "GAL-I03-F0001", "GAL-F0001"),
                             (s["id"], "GAL-I03-F0001-S01", "GAL-F0001-S01")])
    evs = [f, s, m, setattr_(s["id"], "title", "t")]
    a = events.fold(evs)
    b = events.fold(list(reversed(evs)))
    assert {u: (i.id, i.title, i.aliases) for u, i in a.items.items()} == \
           {u: (i.id, i.title, i.aliases) for u, i in b.items.items()}


def test_last_ts_is_the_maximum_over_the_union():
    st = events.fold([new("GAL-F0001", "F", ts=7), retire("GAL-F0001", ts=900), new("GAL-F0002", "F", ts=50)])
    assert st.last_ts == 900
    assert events.fold([]).last_ts == 0


def test_stamp_ts_is_monotonic_even_when_the_clock_steps_back():
    assert events.stamp_ts(100, 5000) == 5000
    assert events.stamp_ts(5000, 4000) == 5001     # backward step
    assert events.stamp_ts(5000, 5000) == 5001     # same tick
    assert events.stamp_ts(0, 1) == 1


# ---------------------------------------------------------------- counters
def test_project_counters_per_type_and_floors():
    st = events.fold([])
    assert (st.next_number("I"), st.next_number("F"), st.next_number("D"), st.next_number("K")) == (3, 1, 1, 1)
    st = events.fold([new("GAL-I03", "I"), new("GAL-I07", "I"), new("GAL-F0042", "F"),
                      new("GAL-I01-D0015", "D"), new("GAL-I02-K0009", "K")])
    assert (st.next_number("I"), st.next_number("F"), st.next_number("D"), st.next_number("K")) == (8, 43, 16, 10)


def test_feature_counter_is_project_wide_whatever_the_parent():
    st = events.fold([new("GAL-I03-F0005", "F", parent="i3"), new("GAL-I04-F0006", "F", parent="i4"),
                      new("GAL-F0007", "F")])
    assert st.next_number("F") == 8


def test_local_counters_two_features_stories_both_start_at_s01():
    f1, f2 = new("GAL-F0001", "F"), new("GAL-F0002", "F")
    st = events.fold([f1, f2])
    assert st.next_number("S", f1["id"]) == 1 and st.next_number("S", f2["id"]) == 1
    st = events.fold([f1, f2, new("GAL-F0001-S01", "S", parent=f1["id"]),
                      new("GAL-F0001-S02", "S", parent=f1["id"])])
    assert st.next_number("S", f1["id"]) == 3 and st.next_number("S", f2["id"]) == 1


def test_local_counters_are_per_type_under_one_parent():
    f = new("GAL-F0001", "F")
    st = events.fold([f, new("GAL-F0001-S01", "S", parent=f["id"]),
                      new("GAL-F0001-T05", "T", parent=f["id"])])
    assert st.next_number("S", f["id"]) == 2 and st.next_number("T", f["id"]) == 6


def test_hundredth_story_is_s100_and_width_follows_the_number():
    f = new("GAL-F0001", "F")
    evs = [f] + [new(f"GAL-F0001-S{n:02d}", "S", parent=f["id"]) for n in range(1, 100)]
    st = events.fold(evs)
    assert st.next_number("S", f["id"]) == 100


def test_move_out_never_reduces_the_old_parents_counter():
    f1, f2 = new("GAL-F0001", "F"), new("GAL-F0002", "F")
    s1, s2 = (new(f"GAL-F0001-S0{n}", "S", parent=f1["id"]) for n in (1, 2))
    m = move(s2["id"], f2["id"], [(s2["id"], "GAL-F0001-S02", "GAL-F0002-S01")])
    st = events.fold([f1, f2, s1, s2, m])
    assert st.next_number("S", f1["id"]) == 3     # S02 was issued there, still counted
    assert st.next_number("S", f2["id"]) == 2     # move-in took S01 under f2
    assert st.items[s2["id"]].parent == f2["id"]


def test_move_back_takes_the_next_free_number_not_the_old_one():
    f1, f2 = new("GAL-F0001", "F"), new("GAL-F0002", "F")
    s = new("GAL-F0001-S01", "S", parent=f1["id"])
    m1 = move(s["id"], f2["id"], [(s["id"], "GAL-F0001-S01", "GAL-F0002-S01")])
    m2 = move(s["id"], f1["id"], [(s["id"], "GAL-F0002-S01", "GAL-F0001-S02")])
    st = events.fold([f1, f2, s, m1, m2])
    assert st.next_number("S", f1["id"]) == 3 and st.next_number("S", f2["id"]) == 2


def test_alias_ids_count_toward_global_counters():
    # a feature promoted away from I03 still burns F0042; a K moved to F burns both
    f = new("GAL-I03-F0042", "F")
    m = move(f["id"], None, [(f["id"], "GAL-I03-F0042", "GAL-F0042")])
    assert events.fold([f, m]).next_number("F") == 43
    k = new("GAL-I02-K0020", "K")
    pm = move(k["id"], None, [(k["id"], "GAL-I02-K0020", "GAL-F0050")], retype="F")
    st = events.fold([k, pm])
    assert st.next_number("K") == 21 and st.next_number("F") == 51


def test_retired_numbers_are_never_reissued():
    st = events.fold([new("GAL-F0001", "F"), retire("GAL-F0001")])
    assert st.next_number("F") == 2
    assert st.resolve_uid("GAL-F0001") == "GAL-F0001"


def test_duplicate_orphan_event_is_reported_once():
    o = retire("GAL-F0999")
    assert len(events.fold([o, dict(o)]).orphans) == 1


def test_local_counter_is_the_maximum_not_the_last_applied():
    f = new("GAL-F0001", "F")
    s2 = new("GAL-F0001-S02", "S", parent=f["id"], ts=1)   # applied before S01
    s1 = new("GAL-F0001-S01", "S", parent=f["id"], ts=2)
    assert events.fold([f, s1, s2]).next_number("S", f["id"]) == 3


def test_numbers_in_move_entries_count_even_when_the_item_is_unknown():
    f = new("GAL-I03-F0001", "F")
    m = move(f["id"], None, [(f["id"], "GAL-I03-F0001", "GAL-F0001"),
                             ("ghost", "GAL-I02-K0030-T01", "GAL-F0044-T01")])
    st = events.fold([f, m])
    assert st.next_number("K") == 31 and st.next_number("F") == 45
