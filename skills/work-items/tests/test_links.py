"""link / set: closed relations, ops, attributes, home validation (FR-11, FR-15, FR-16, FR-19; AC-8, AC-11)."""
import json

import pytest

from conftest import SCRIPTS
from wilib import alloc, cli, events, links, move, project, resolve
from wilib.store import Store


@pytest.fixture
def proj(repo):
    project.init(repo, "GAL", ["cpg", "claude"])
    alloc.new(repo, "I")                                        # GAL-I03
    alloc.new(repo, "F", parent="GAL-I03", slug="feat")         # GAL-I03-F0001
    alloc.new(repo, "S", parent="GAL-I03-F0001")                # GAL-I03-F0001-S01
    alloc.new(repo, "D")                                        # GAL-I01-D0001
    alloc.new(repo, "D")                                        # GAL-I01-D0002
    return repo


def rec(root, name):
    return resolve.resolve(Store(root, "GAL").read_state(), name)


# ---- AC-8: concerns is functional; the defect's ID never changes ------------
def test_concerns_set_replace_clear_keeps_defect_id(proj):
    d = "GAL-I01-D0001"
    links.link(proj, d, "concerns", "GAL-I03-F0001")
    assert rec(proj, d)["links"] == {"concerns": ["GAL-I03-F0001"]}
    links.link(proj, d, "concerns", "GAL-I03-F0001-S01")
    assert rec(proj, d)["links"] == {"concerns": ["GAL-I03-F0001-S01"]}
    links.link(proj, d, "concerns", clear=True)
    r = rec(proj, d)
    assert (r["id"], r["parent"], r["links"]) == (d, "GAL-I01", {})


def test_defect_is_never_a_child_of_what_it_concerns(proj):
    links.link(proj, "GAL-I01-D0001", "concerns", "GAL-I03-F0001")
    s = Store(proj, "GAL").read_state()
    assert s.items["GAL-I01-D0001"].parent == "GAL-I01"
    assert [i.id for i in s.items.values() if i.parent == "GAL-I03-F0001"] == ["GAL-I03-F0001-S01"]


def test_concerns_event_carries_op_set_and_null_on_clear(proj):
    links.link(proj, "GAL-I01-D0001", "concerns", "GAL-I03-F0001")
    links.link(proj, "GAL-I01-D0001", "concerns", clear=True)
    evs = [e for e in events.read_events(Store(proj, "GAL").journal_path.read_text()) if e["ev"] == "link"]
    assert [(e["op"], e["to"]) for e in evs] == [("set", "GAL-I03-F0001"), ("set", None)]


# ---- multi-valued relations ---------------------------------------------------
def test_fixed_by_adds_two_targets_in_order_and_removes_one(proj):
    d = "GAL-I01-D0001"
    links.link(proj, d, "fixed-by", "GAL-I03-F0001-S01")
    links.link(proj, d, "fixed-by", "GAL-I03-F0001")
    assert rec(proj, d)["links"] == {"fixed-by": ["GAL-I03-F0001-S01", "GAL-I03-F0001"]}
    links.link(proj, d, "fixed-by", "GAL-I03-F0001-S01", op="remove")
    assert rec(proj, d)["links"] == {"fixed-by": ["GAL-I03-F0001"]}


def test_relates_add_is_idempotent_and_remove_of_absent_is_noop(proj):
    d = "GAL-I01-D0001"
    links.link(proj, d, "relates", "GAL-I01-D0002")
    links.link(proj, d, "relates", "GAL-I01-D0002")
    links.link(proj, d, "relates", "GAL-I03-F0001", op="remove")
    assert rec(proj, d)["links"] == {"relates": ["GAL-I01-D0002"]}


def test_removing_last_target_drops_the_relation_key(proj):
    d = "GAL-I01-D0001"
    links.link(proj, d, "relates", "GAL-I01-D0002")
    links.link(proj, d, "relates", "GAL-I01-D0002", op="remove")
    assert rec(proj, d)["links"] == {}


def test_wrong_op_for_relation_is_refused_and_writes_nothing(proj):
    n = len(Store(proj, "GAL").journal_path.read_text().splitlines())
    with pytest.raises(links.LinkError, match="concerns"):
        links.link(proj, "GAL-I01-D0001", "concerns", "GAL-I03-F0001", op="add")
    with pytest.raises(links.LinkError, match="fixed-by"):
        links.link(proj, "GAL-I01-D0001", "fixed-by", "GAL-I03-F0001", op="set")
    with pytest.raises(links.LinkError):
        links.link(proj, "GAL-I01-D0001", "fixed-by", clear=True)
    assert len(Store(proj, "GAL").journal_path.read_text().splitlines()) == n


def test_unknown_relation_refused(proj):
    with pytest.raises(links.LinkError, match="blocks"):
        links.link(proj, "GAL-I01-D0001", "blocks", "GAL-I03-F0001")


def test_unknown_source_or_target_refused(proj):
    with pytest.raises(resolve.ResolveError):
        links.link(proj, "GAL-I09-F0001", "relates", "GAL-I03-F0001")
    with pytest.raises(resolve.ResolveError):
        links.link(proj, "GAL-I01-D0001", "relates", "GAL-I09")


def test_target_required_unless_clear_or_remove_form(proj):
    with pytest.raises(links.LinkError):
        links.link(proj, "GAL-I01-D0001", "concerns")


# ---- links follow renames ------------------------------------------------------
def test_link_to_moved_item_shows_current_id(proj):
    d = "GAL-I01-D0001"
    links.link(proj, d, "concerns", "GAL-I03-F0001-S01")
    links.link(proj, d, "fixed-by", "GAL-I03-F0001")
    alloc.new(proj, "I")                                        # GAL-I04
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    assert rec(proj, d)["links"] == {"concerns": ["GAL-I04-F0001-S01"],
                                     "fixed-by": ["GAL-I04-F0001"]}


def test_link_accepts_alias_as_target(proj):
    alloc.new(proj, "I")
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    links.link(proj, "GAL-I01-D0001", "relates", "GAL-I03-F0001")
    assert rec(proj, "GAL-I01-D0001")["links"] == {"relates": ["GAL-I04-F0001"]}


# ---- AC-11: attributes -----------------------------------------------------------
def test_set_milestone_home_slug_where_and_touched(proj):
    f = "GAL-I03-F0001"
    links.set_attr(proj, f, "milestone", "M3")
    links.set_attr(proj, f, "home", "cpg")
    links.set_attr(proj, f, "slug", "renamed")
    links.set_attr(proj, f, "where", "docs/x.md")
    links.set_attr(proj, f, "touched", "cpg,claude")
    r = rec(proj, f)
    assert (r["milestone"], r["home"], r["slug"], r["where"], r["touched"]) == (
        "M3", "cpg", "renamed", "docs/x.md", ["cpg", "claude"])
    assert r["id"] == f


def test_set_never_changes_any_id(proj):
    before = sorted(i.id for i in Store(proj, "GAL").read_state().items.values())
    for attr, val in (("milestone", "M4"), ("home", "claude"), ("touched", "cpg"),
                      ("slug", "zz"), ("where", "w")):
        links.set_attr(proj, "GAL-I03-F0001", attr, val)
    assert sorted(i.id for i in Store(proj, "GAL").read_state().items.values()) == before
    assert rec(proj, "GAL-I03-F0001")["aliases"] == []


def test_unknown_home_refused_and_unchanged(proj):
    with pytest.raises(links.LinkError, match="nope"):
        links.set_attr(proj, "GAL-I03-F0001", "home", "nope")
    assert rec(proj, "GAL-I03-F0001")["home"] == "repo"


def test_touched_components_validated(proj):
    with pytest.raises(links.LinkError, match="nope"):
        links.set_attr(proj, "GAL-I03-F0001", "touched", "cpg,nope")
    assert rec(proj, "GAL-I03-F0001")["touched"] == []


def test_unknown_attribute_refused(proj):
    with pytest.raises(links.LinkError, match="colour"):
        links.set_attr(proj, "GAL-I03-F0001", "colour", "red")


def test_set_on_unknown_item_refused(proj):
    with pytest.raises(resolve.ResolveError):
        links.set_attr(proj, "GAL-I09", "milestone", "M1")


# ---- fold / schema ------------------------------------------------------------
def _ev(n, **kw):
    return {"evid": f"e{n}", "ts": n, "ev": "link", "from": "d", "rel": "relates", **kw}


def test_validate_refuses_missing_op_on_multivalued_and_add_on_concerns():
    with pytest.raises(events.EventError):
        events.validate_event(_ev(1, to="x"))
    with pytest.raises(events.EventError):
        events.validate_event(_ev(1, rel="concerns", op="add", to="x"))
    with pytest.raises(events.EventError):
        events.validate_event(_ev(1, op="add", to=None))
    with pytest.raises(events.EventError):
        events.validate_event(_ev(1, op="bogus", to="x"))
    events.validate_event(_ev(1, rel="concerns", to="x"))       # op defaults to set


def test_fold_orders_by_ts_not_arrival():
    evs = [_ev(2, op="remove", to="a"), _ev(1, op="add", to="a"), _ev(3, op="add", to="b")]
    assert events.fold(evs).links[("relates", "d")] == ["b"]


# ---- CLI ---------------------------------------------------------------------------
def test_cli_link_and_set_roundtrip(proj, capsys):
    run = lambda *a: cli.main(list(a), cwd=proj)
    assert run("link", "GAL-I01-D0001", "concerns", "GAL-I03-F0001") == 0
    assert run("link", "GAL-I01-D0001", "fixed-by", "GAL-I03-F0001-S01") == 0
    assert run("link", "GAL-I01-D0001", "fixed-by", "GAL-I03-F0001") == 0
    assert run("link", "GAL-I01-D0001", "fixed-by", "GAL-I03-F0001-S01", "--remove") == 0
    assert run("set", "GAL-I03-F0001", "milestone=M3") == 0
    assert run("set", "GAL-I03-F0001", "touched=cpg,claude") == 0
    capsys.readouterr()
    assert run("resolve", "GAL-I01-D0001", "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["links"] == {"concerns": ["GAL-I03-F0001"], "fixed-by": ["GAL-I03-F0001"]}
    assert run("link", "GAL-I01-D0001", "concerns", "--clear") == 0
    assert run("resolve", "GAL-I01-D0001", "--json") == 0
    assert json.loads(capsys.readouterr().out)["links"] == {"fixed-by": ["GAL-I03-F0001"]}
    assert run("resolve", "GAL-I03-F0001", "--json") == 0
    assert json.loads(capsys.readouterr().out)["milestone"] == "M3"


def test_cli_errors_exit_1_and_unknown_item_exit_4(proj, capsys):
    run = lambda *a: cli.main(list(a), cwd=proj)
    assert run("link", "GAL-I01-D0001", "concerns", "GAL-I03-F0001", "--remove") == 1
    assert run("set", "GAL-I03-F0001", "home=nope") == 1
    assert run("set", "GAL-I03-F0001", "milestone") == 1
    assert run("set", "GAL-I09", "milestone=M1") == 4


def test_fold_defaults_missing_op_to_set_for_concerns():
    base = {"ev": "link", "rel": "concerns", "from": "d"}
    evs = [{**base, "evid": "e1", "ts": 1, "to": "a"}, {**base, "evid": "e2", "ts": 2, "to": "b"}]
    assert events.fold(evs).links[("concerns", "d")] == ["b"]
