"""P1-23c (behaviours (2)-(4)): distance, grouping and folding of the walk's items, and the reverse question (``docs/plans/connected-code-graph-p1-answers.md`` P1-23).

Contract: ``-answer-spec2.md`` §3.5 (the ``unit`` item's sort slots), ``-answer-spec.md`` §5.1 (``directness``), §5.4 (a folded group enumerates every member, display only),
§6 (the ``items`` order: repository, component's ``rootPath``, unit, distance, confirmed before possible, path, line, subject id; ``groups`` by ``(kind, key)``);
``docs/plans/connected-code-graph.md`` §3.7 / ``-p0-ml.md`` §8.3 (a resolved call's other candidates stay ``possible``/``resolution-unverified`` and are folded away in
rendering when the language's resolved-call precision is at or above 0.95 over at least 60 sites; below, or unmeasured, they are listed; a fold never drops a member).

Group keys are graph data: the order is a total order over stable keys (code points, never a locale), independent of dict or input order, and a unicode or empty
component neither crashes nor merges.
"""
from __future__ import annotations

import dataclasses
import random

import pytest
from engine_walk_helpers import G, MemoryReader

from ccg.engine import bfs, group
from ccg.engine.group import UnitRef
from ccg.engine.reader import GraphDataError
from test_engine_bfs import run


# ---------------------------------------------------------------------------------------------- synthetic items

def hop(label, ident, state="end", edge=None):
    return {"node": (label, ident), "state": (state, ""), "edge": edge}


def edge(typ="CALLS", dir_="<-", status="confirmed", reason=None):
    return {"type": typ, "dir": dir_, "status": status, "reason": reason}


def synth(ident, *, distance=1, status="confirmed", repo="r", unit="u", path="a.py", line=1, root="p", kind="symbol", chain=None, site=None):
    """``(item, facts, units)`` for one item. A ``possible`` item's chain ends in a ``name-match`` hop unless ``chain`` says otherwise; ``site`` makes it a
    ``resolution-unverified`` candidate of the call site ``site`` (a depends-on chain: anchor, ``HAS_CALL ->`` site, ``CALLS_NAME ->`` the item)."""
    if chain is None:
        if site is not None:
            chain = [hop("Symbol", "anchor", "value"), hop("CallSite", site, "value", edge("HAS_CALL", "->")),
                     hop("Symbol", ident, "value", edge("CALLS_NAME", "->", "possible", "resolution-unverified"))]
        else:
            last = edge("CALLS", "<-", status, None if status == "confirmed" else "name-match")
            chain = [hop("Symbol", "anchor", "value"), hop("Symbol", ident, "end", last)]
    possible = sum(1 for s in chain if s["edge"] and s["edge"]["status"] == "possible")
    item = bfs.Item(kind, ident, distance, possible, len(chain) - 1, chain)
    label = {"symbol": "Symbol", "config": "ConfigEntry", "unit": "Unit"}[kind]
    facts = {(label, ident): {"repo": repo, "unit": unit, "path": path, "line": line}}
    return item, facts, {unit: UnitRef(root, "python")}


def many(specs):
    items, facts, units = [], {}, {}
    for ident, kw in specs:
        i, f, u = synth(ident, **kw)
        items.append(i), facts.update(f), units.update(u)
    return items, facts, units


def order(listing):
    return [p.item.id for p in listing.items]


# ---------------------------------------------------------------------------------------------- the sort

LEVELS = ("repo", "root", "unit", "distance", "status", "path", "line", "id")
LOW = {"repo": "a", "root": "a", "unit": "a", "distance": 1, "status": "confirmed", "path": "a.py", "line": 1, "id": "a"}
HIGH = {"repo": "b", "root": "b", "unit": "b", "distance": 2, "status": "possible", "path": "b.py", "line": 2, "id": "b"}
MID = {"repo": "m", "root": "m", "unit": "m", "distance": 1, "status": "confirmed", "path": "m.py", "line": 5, "id": "m"}


def precedence_pair(level: int):
    """Two items that differ at ``LEVELS[level]`` in A's favour and at every later level in B's: the order reads the earlier level first."""
    out = []
    for who in "AB":
        attrs = {}
        for i, name in enumerate(LEVELS):
            if i < level:
                attrs[name] = MID[name]
            elif i == level:
                attrs[name] = LOW[name] if who == "A" else HIGH[name]
            else:
                attrs[name] = HIGH[name] if who == "A" else LOW[name]
        out.append(attrs)
    return out


@pytest.mark.parametrize("level", range(len(LEVELS)), ids=LEVELS)
def test_the_order_is_repository_component_unit_distance_confirmed_before_possible_path_line_subject_id_read_in_that_order(level):
    a, b = precedence_pair(level)
    items, facts, units = [], {}, {}
    for at in (a, b):
        i, f, u = synth(at["id"], distance=at["distance"], status=at["status"], repo=at["repo"], unit=f"u-{at['unit']}", path=at["path"], line=at["line"], root=at["root"])
        items.append(i), facts.update(f), units.update(u)
    got = group.group_items(items, facts, units)
    assert order(got) == order(group.group_items(items[::-1], facts, units))
    assert order(got)[0] == items[0].id, f"{LEVELS[level]} must decide before every later level"


def test_a_confirmed_item_precedes_a_possible_one_at_the_same_distance_and_distance_precedes_status():
    items, facts, units = many([("far-confirmed", dict(distance=2)), ("near-possible", dict(distance=1, status="possible")), ("near-confirmed", dict(distance=1)),
                                ("far-possible", dict(distance=2, status="possible"))])
    assert order(group.group_items(items, facts, units)) == ["near-confirmed", "near-possible", "far-confirmed", "far-possible"]


@pytest.mark.parametrize("seed", range(25))
def test_the_order_is_independent_of_input_order_and_of_the_insertion_order_of_the_facts_and_units(seed):
    rng = random.Random(seed)
    specs = [(f"s{i}", dict(repo=rng.choice("ab"), unit=f"u{rng.randrange(3)}", root=rng.choice(["", "x", "é", "Z"]), distance=rng.randrange(3),
                            status=rng.choice(["confirmed", "possible"]), path=rng.choice(["a.py", "b.py"]), line=rng.randrange(3))) for i in range(30)]
    # a unit has one root: make it a function of the unit id
    roots = {}
    specs = [(i, {**kw, "root": roots.setdefault((kw["repo"], kw["unit"]), kw["root"]), "unit": f"{kw['repo']}-{kw['unit']}"}) for i, kw in specs]
    items, facts, units = many(specs)
    base = order(group.group_items(items, facts, units))
    items2, facts2, units2 = list(items), dict(facts), dict(units)
    rng.shuffle(items2)
    f_keys, u_keys = list(facts2), list(units2)
    rng.shuffle(f_keys), rng.shuffle(u_keys)
    facts2, units2 = {k: facts2[k] for k in f_keys}, {k: units2[k] for k in u_keys}
    assert order(group.group_items(items2, facts2, units2)) == base
    assert order(group.group_items(items[::-1], facts, units)) == base


def test_the_order_is_by_code_point_not_by_locale_collation():
    # a locale-aware collation puts "é" between "e" and "f" and "Z" after "a"; code points put "Z" < "a" < "e" < "z" < "é"
    items, facts, units = many([(f"i{n}", dict(unit=f"u{n}", root=r)) for n, r in enumerate(["é", "z", "a", "Z", "e"])])
    got = group.group_items(items, facts, units)
    assert [p.root for p in got.items] == ["Z", "a", "e", "z", "é"]


def test_a_composed_and_a_decomposed_component_stay_two_sections_and_an_empty_component_is_its_own_first_section():
    items, facts, units = many([("c", dict(unit="u1", root="é")), ("d", dict(unit="u2", root="é")), ("e", dict(unit="u3", root="")), ("f", dict(unit="u4", root=" "))])
    got = group.group_items(items, facts, units)
    secs = got.sections()
    assert [s.root for s in secs] == ["", " ", "é", "é"]
    assert len(secs) == 4 and [len(s.items) for s in secs] == [1, 1, 1, 1]


def test_two_units_with_one_root_in_one_repo_are_two_components_and_one_root_in_two_repos_are_two_too():
    items, facts, units = many([("a", dict(unit="u1", root="svc")), ("b", dict(unit="u2", root="svc")), ("c", dict(unit="u3", root="svc", repo="other"))])
    secs = group.group_items(items, facts, units).sections()
    assert [(s.repo, s.root, s.unit, s.distance) for s in secs] == [("other", "svc", "u3", 1), ("r", "svc", "u1", 1), ("r", "svc", "u2", 1)]


def test_sections_run_repository_then_component_then_distance_with_confirmed_first_inside_a_section():
    items, facts, units = many([
        ("a1", dict(distance=2)), ("a2", dict(distance=1, status="possible")), ("a3", dict(distance=1)), ("b1", dict(repo="s", unit="v", root="q", distance=1)),
        ("a4", dict(unit="u9", root="p0", distance=3))])
    secs = group.group_items(items, facts, units).sections()
    assert [(s.repo, s.root, s.distance, [p.item.id for p in s.items]) for s in secs] == [
        ("r", "p", 1, ["a3", "a2"]), ("r", "p", 2, ["a1"]), ("r", "p0", 3, ["a4"]), ("s", "q", 1, ["b1"])]
    assert all(all(p.repo == s.repo and p.root == s.root and p.unit == s.unit and p.distance == s.distance for p in s.items) for s in secs)


def test_a_unit_item_sorts_by_its_root_in_the_path_slot_and_zero_in_the_line_slot_and_a_config_item_by_its_own_path_and_line():
    items, facts, units = many([("U", dict(kind="unit", unit="u", path="zzz", line=99, root="m", status="possible")), ("S", dict(unit="u", path="m", line=1, root="m", status="possible")),
                                ("C", dict(kind="config", unit="u", path="a/cfg.yaml", line=7, root="m", status="possible"))])
    got = group.group_items(items, facts, units)
    assert {p.item.id: (p.path, p.pos) for p in got.items} == {"U": ("m", 0), "C": ("a/cfg.yaml", 7), "S": ("m", 1)}
    assert order(got) == ["C", "U", "S"]      # path "a/cfg.yaml" < "m"; then the unit item's line 0 < the symbol's line 1


# ---------------------------------------------------------------------------------------------- the walk's graphs

def set_root(g, unit, root):
    next(u for u in g.rows.nodes["Unit"] if u["unitId"] == unit)["rootPath"] = root


def candidates_graph(*, lang="python", cand_lang="python", n_sites=1, resolved=True):
    """``F`` (unit ``ua``, ``lang``) holds ``n_sites`` call sites, each resolved to ``T<i>`` and name-matching two other candidates ``X<i>a``/``X<i>b`` that live in unit
    ``ub`` (``cand_lang``) of another repo; with ``resolved=False`` the sites resolve to nothing (the candidates are then ``name-match``)."""
    g = G(repo="repo-a", unit="ua", lang=lang)
    set_root(g, "ua", "svc/a")
    g.unit_node("ub", "repo-b", root="lib/b", lang=cand_lang)
    g.sym("F", path="f.py", start=3)
    for i in range(n_sites):
        c = g.site(f"c{i}", "F", line=10 + i)
        g.sym(f"T{i}", unit="ua", path="t.py", start=1 + i)
        if resolved:
            g.calls(c, f"T{i}")
        g.calls_name(c, f"T{i}", name=f"fn{i}")
        for tag in "ab":
            g.sym(f"X{i}{tag}", unit="ub", repo="repo-b", path="x.py", start=1 + i)
            g.calls_name(c, f"X{i}{tag}", name=f"fn{i}")
    return g


def walk_listing(g, anchor=("Symbol", "F"), **kw):
    return group.listing(MemoryReader(g), anchor, **kw)


# ---------------------------------------------------------------------------------------------- (2) distance

def mixed_graph():
    """A call chain through a ``Ret`` summary (steps inside a function, distance 0), a link (``VIA`` 1, ``TO`` 0) and a plain callee."""
    g = G(unit="ua")
    for s in ("A", "B", "C", "P", "K"):
        g.sym(s)
    g.site("cB", "B"), g.site("cC", "C"), g.calls("cB", "A"), g.calls("cC", "B")
    g.ret("rB", "B"), g.data("Ret", "rB", "CallSite", "cB", "r", "ret")
    g.site("cA", "A"), g.calls("cA", "K")
    g.cp("cp1"), g.link("L", provider="P", cp="cp1"), g.via("cA", "L")
    return g


INTER_FUNCTION = {"CALLS", "CALLS_NAME", "VIA", "CONSUMES", "BOUND_BY"}


def chain_distance(item):
    return sum(s["edge"]["type"] in INTER_FUNCTION for s in item.chain[1:])


def test_the_walks_distance_is_the_inter_function_hop_count_and_the_placed_item_carries_it_unchanged_direct_is_one():
    g = mixed_graph()
    for anchor, column in ((("Symbol", "A"), bfs.IMPACT), (("Symbol", "A"), bfs.DEPENDS_ON)):
        w = bfs.walk(MemoryReader(g), anchor, column=column)
        placed = {p.item.id: p for p in group.group_items(w.items(), w.facts, {"ua": UnitRef("", "python")}).items}
        assert placed, column
        for p in placed.values():
            assert p.distance == p.item.distance == chain_distance(p.item)       # group.py recomputes nothing: the walk's figure is the inter-function count
            assert p.directness == ("direct" if p.distance == 1 else "indirect")
    imp = {p.item.id: p for p in group.group_items(*walk_parts(g, bfs.IMPACT)).items}
    assert (imp["B"].distance, imp["B"].directness) == (1, "direct")
    assert (imp["C"].distance, imp["C"].directness) == (2, "indirect")     # cC -> B's Ret summary: steps inside B cost 0, the call into B costs 1
    dep = {p.item.id: p for p in group.group_items(*walk_parts(g, bfs.DEPENDS_ON)).items}
    assert (dep["K"].distance, dep["K"].directness) == (1, "direct")
    assert (dep["P"].distance, dep["P"].directness) == (1, "direct")       # VIA 1, TO 0


def walk_parts(g, column):
    w = bfs.walk(MemoryReader(g), ("Symbol", "A"), column=column)
    return w.items(), w.facts, {"ua": UnitRef("", "python")}


def test_directness_is_direct_exactly_at_distance_one():
    assert [group.directness(d) for d in (0, 1, 2, 3)] == ["indirect", "direct", "indirect", "indirect"]


# ---------------------------------------------------------------------------------------------- (3) the fold decision

def test_the_threshold_is_exactly_ninety_five_hundredths_over_at_least_sixty_resolved_sites():
    assert (group.FOLD_THRESHOLD.numerator, group.FOLD_THRESHOLD.denominator) == (19, 20) and group.MIN_RESOLVED_SITES == 60


@pytest.mark.parametrize("correct,n,expect", [
    (57, 60, True),        # exactly 0.95
    (114, 120, True),      # exactly 0.95
    (95, 100, True),       # exactly 0.95
    (56, 60, False),       # 0.9333
    (94, 99, False),       # 0.9495: just below
    (110, 116, False),     # 0.948, Python as P0 measured it
    (77, 81, True),        # 0.9506: just above
    (60, 60, True),
    (59, 59, False),       # 1.0 but under the 60-site floor: unmeasured, listed
    (5, 5, False),
    (0, 0, False),         # nothing measured
])
def test_a_language_folds_at_or_above_point_nine_five_with_at_least_sixty_resolved_sites(correct, n, expect):
    assert group.folds(correct, n) is expect


@pytest.mark.parametrize("bad", [(-1, 60), (61, 60), (1.0, 60), (57, 60.0), (True, 60), (57, True), ("57", 60), (None, 60)])
def test_a_malformed_measurement_is_refused_not_coerced(bad):
    with pytest.raises(ValueError):
        group.folds(*bad)


def test_fold_languages_needs_every_configuration_to_pass_and_an_unmeasured_language_is_listed():
    measured = {"go": [(108, 108)], "javascript": [(102, 102), (57, 60)], "python": [(110, 116)], "java": [(47, 47)], "cpp": [(5, 5)],
                "both-fail-one": [(100, 100), (56, 60)], "empty": []}
    assert group.fold_languages(measured) == frozenset({"go", "javascript"})      # the P0 table's Fold rows, Python/Java/C++ listed
    assert group.fold_languages({}) == frozenset()


def test_fold_languages_accepts_any_mapping_order():
    a = {"x": [(100, 100)], "y": [(100, 100)]}
    assert group.fold_languages(a) == group.fold_languages(dict(reversed(list(a.items()))))


# ---------------------------------------------------------------------------------------------- (3) folding over real walks

def test_a_resolved_calls_other_candidates_stay_possible_resolution_unverified_in_the_walk_and_are_listed_when_the_language_is_not_folded():
    got = walk_listing(candidates_graph(), question="depends-on")
    assert got.groups == ()
    by = {p.item.id: p.item for p in got.items}
    assert set(by) == {"T0", "X0a", "X0b"}
    assert (by["T0"].status, by["T0"].reason) == ("confirmed", None)
    assert all((by[i].status, by[i].reason) == ("possible", "resolution-unverified") for i in ("X0a", "X0b"))


def test_a_folded_language_moves_the_other_candidates_into_one_group_per_resolved_site_and_drops_none():
    g = candidates_graph(n_sites=2)
    listed = walk_listing(g, question="depends-on")
    folded = walk_listing(g, question="depends-on", fold_languages={"python"})
    assert [p.item.id for p in folded.items] == ["T0", "T1"]
    assert [(gr.kind, gr.key, gr.reason, gr.count) for gr in folded.groups] == [("other-possible-callees", "c0", "resolution-unverified", 2),
                                                                              ("other-possible-callees", "c1", "resolution-unverified", 2)]
    assert [[m.item.id for m in gr.members] for gr in folded.groups] == [["X0a", "X0b"], ["X1a", "X1b"]]
    expanded = {p.item.id: p for p in (*folded.items, *(m for gr in folded.groups for m in gr.members))}
    assert set(expanded) == {p.item.id for p in listed.items}                      # nothing dropped, nothing twice
    assert len(expanded) == len(listed.items) == 6
    by_listed = {p.item.id: p for p in listed.items}
    for ident, p in expanded.items():
        assert p == by_listed[ident]                                                 # every member keeps its own chain, subject and keys
    assert all(m.item.chain and m.item.status == "possible" for gr in folded.groups for m in gr.members)


def test_the_fold_decision_is_the_resolved_call_sites_language_not_the_candidates():
    g = candidates_graph(lang="go", cand_lang="python")        # the site is Go, the candidates Python
    assert walk_listing(g, question="depends-on", fold_languages={"python"}).groups == ()
    assert len(walk_listing(g, question="depends-on", fold_languages={"go"}).groups) == 1


def test_name_matches_of_an_unresolved_site_are_never_folded_whatever_the_language():
    g = candidates_graph(resolved=False)
    got = walk_listing(g, question="depends-on", fold_languages={"python"})
    assert got.groups == ()
    assert {(p.item.id, p.item.reason) for p in got.items} == {("T0", "name-match"), ("X0a", "name-match"), ("X0b", "name-match")}


def test_the_impact_column_folds_the_caller_that_reaches_the_anchor_only_by_a_resolved_sites_other_candidate():
    g = G(unit="ua")
    set_root(g, "ua", "r")
    for s in ("S", "T", "B"):
        g.sym(s)
    g.site("c", "B"), g.calls("c", "T"), g.calls_name("c", "S")
    folded = walk_listing(g, ("Symbol", "S"), fold_languages={"python"})
    assert folded.items == () and [(x.key, [m.item.id for m in x.members]) for x in folded.groups] == [("c", ["B"])]
    member = folded.groups[0].members[0]
    assert (member.item.status, member.item.reason, member.distance) == ("possible", "resolution-unverified", 1)
    assert [p.item.id for p in walk_listing(g, ("Symbol", "S")).items] == ["B"]


def test_only_an_item_whose_first_possible_hop_is_the_resolved_sites_other_candidate_is_grouped():
    later = [hop("Symbol", "anchor", "value"), hop("Symbol", "m", "value", edge("CALLS", "->", "possible", "name-match")),
             hop("CallSite", "c", "value", edge("HAS_CALL", "->")), hop("Symbol", "x", "value", edge("CALLS_NAME", "->", "possible", "resolution-unverified"))]
    other = [hop("Symbol", "anchor", "value"), hop("CallSite", "c", "value", edge("HAS_CALL", "->")), hop("Symbol", "y", "value", edge("CALLS_NAME", "->", "possible", "name-match"))]
    wrong_edge = [hop("Symbol", "anchor", "value"), hop("CallSite", "c", "value", edge("HAS_CALL", "->")), hop("Symbol", "z", "value", edge("CALLS", "->", "possible", "resolution-unverified"))]
    items, facts, units = [], {}, {}
    for ident, chain in (("x", later), ("y", other), ("z", wrong_edge)):
        i, f, u = synth(ident, status="possible", chain=chain)
        items.append(i), facts.update(f), units.update(u)
    facts[("CallSite", "c")] = {"unit": "u"}
    got = group.group_items(items, facts, units, {"python"})
    assert got.groups == () and order(got) == ["x", "y", "z"]


def test_groups_are_ordered_by_kind_and_key_and_members_by_the_items_rule_whatever_the_input_order():
    specs = [(f"m{n}", dict(site=s, status="possible", distance=d, path=p, line=ln, unit="u"))
             for n, (s, d, p, ln) in enumerate([("kb", 1, "b.py", 1), ("ka", 2, "a.py", 9), ("ka", 1, "z.py", 3), ("kb", 1, "a.py", 7), ("", 1, "a.py", 1), ("é", 1, "a.py", 1)])]
    items, facts, units = many(specs)
    for s in ("ka", "kb", "", "é"):
        facts[("CallSite", s)] = {"unit": "u"}
    base = group.group_items(items, facts, units, {"python"})
    assert [g.key for g in base.groups] == ["", "ka", "kb", "é"]                      # code point order; "" and "é" are keys, not errors
    assert [[m.item.id for m in g.members] for g in base.groups] == [["m4"], ["m2", "m1"], ["m3", "m0"], ["m5"]]
    for seed in range(15):
        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        again = group.group_items(shuffled, facts, units, {"python"})
        assert [(g.key, [m.item.id for m in g.members]) for g in again.groups] == [(g.key, [m.item.id for m in g.members]) for g in base.groups]


def test_two_sites_with_look_alike_keys_are_two_groups():
    specs = [("m0", dict(site="cé", status="possible")), ("m1", dict(site="cé", status="possible")), ("m2", dict(site="c", status="possible"))]
    items, facts, units = many(specs)
    for s in ("cé", "cé", "c"):
        facts[("CallSite", s)] = {"unit": "u"}
    got = group.group_items(items, facts, units, {"python"})
    assert len(got.groups) == 3 and all(g.count == 1 for g in got.groups)


def test_a_group_and_a_listing_carry_no_precision_figure_and_the_fold_input_is_a_set_of_languages():
    for cls in (group.FoldGroup, group.Listing, group.Placed, group.Section):
        assert not [f.name for f in dataclasses.fields(cls) if "precision" in f.name or "threshold" in f.name or "fold" in f.name]
    assert [f.name for f in dataclasses.fields(group.FoldGroup)] == ["kind", "key", "reason", "members"]
    got = walk_listing(candidates_graph(), question="depends-on", fold_languages=frozenset({"python"}))
    assert got.groups[0].count == len(got.groups[0].members) == 2


def test_a_candidate_whose_site_unit_has_no_language_is_listed_not_folded():
    g = candidates_graph()
    next(u for u in g.rows.nodes["Unit"] if u["unitId"] == "ua")["language"] = None
    assert walk_listing(g, question="depends-on", fold_languages={"python"}).groups == ()


# ---------------------------------------------------------------------------------------------- (4) the reverse question

def test_the_reverse_question_is_the_depends_on_column_and_impact_the_impact_column_with_the_same_marking_and_grouping():
    g = candidates_graph()
    rev = walk_listing(g, question="depends-on", fold_languages={"python"})
    w = bfs.walk(MemoryReader(g), ("Symbol", "F"), column=bfs.DEPENDS_ON)
    assert {i.id for i in w.items()} == {p.item.id for p in rev.items} | {m.item.id for gr in rev.groups for m in gr.members}
    assert group.QUESTIONS == {"impact": bfs.IMPACT, "depends-on": bfs.DEPENDS_ON}
    # F has no caller: the impact question over the same anchor lists nothing, the reverse question lists its callees
    assert walk_listing(g, question="impact").items == ()
    assert walk_listing(g).items == ()                                           # impact is the default question
    # the marking: the resolved callee first (confirmed), repository -> component order across two repositories
    plain = walk_listing(g, question="depends-on")
    assert [(p.repo, p.root, p.item.id, p.item.status) for p in plain.items] == [("repo-a", "svc/a", "T0", "confirmed"), ("repo-b", "lib/b", "X0a", "possible"),
                                                                                 ("repo-b", "lib/b", "X0b", "possible")]


def test_an_unknown_question_is_refused():
    with pytest.raises(ValueError):
        walk_listing(candidates_graph(), question="callers")


def test_the_reverse_question_marks_a_never_built_units_crossing_possible_and_groups_it_by_its_own_root():
    g = G(unit="ua")
    set_root(g, "ua", "r")
    g.unit_node("ud", "repo-z", root="gone/svc")
    g.sym("F"), g.site("c", "F"), g.link("L", to_unit="ud"), g.via("c", "L")
    got = walk_listing(g, question="depends-on")
    assert [(p.item.kind, p.item.id, p.item.status, p.item.reason, p.repo, p.unit, p.root, p.path, p.pos) for p in got.items] == [
        ("unit", "ud", "possible", "target-not-mapped", "repo-z", "ud", "gone/svc", "gone/svc", 0)]


# ---------------------------------------------------------------------------------------------- hostile and malformed data

def test_listing_reads_every_unit_it_needs_in_one_read_and_never_a_unit_it_does_not():
    g = candidates_graph(n_sites=2)
    r = MemoryReader(g)
    group.listing(r, ("Symbol", "F"), question="depends-on", fold_languages={"python"})
    reads = [c for c in r.calls if c[0] == "units"]
    assert len(reads) >= 1
    assert set().union(*(set(c[1]["ids"]) for c in reads)) >= {"ua", "ub"}


def test_an_item_whose_unit_has_no_unit_row_is_a_graph_data_error_that_names_no_value():
    items, facts, units = many([("SECRET-item", dict(unit="SECRET-unit"))])
    with pytest.raises(GraphDataError) as e:
        group.group_items(items, facts, {})
    assert "SECRET" not in str(e.value)


@pytest.mark.parametrize("drop", ["repo", "unit", "path", "line"])
def test_an_item_missing_a_grouping_fact_is_a_graph_data_error_not_a_guess(drop):
    items, facts, units = many([("a", dict())])
    del facts[("Symbol", "a")][drop]
    with pytest.raises(GraphDataError):
        group.group_items(items, facts, units)


def test_hostile_strings_in_every_key_neither_crash_nor_merge():
    nasty = ["", " ", "\u0000", "\ud800", "a\nb", "‮evil", "💥", "../../etc", "'; DROP", "é", "é"]
    specs = [(f"i{n}", dict(unit=f"u{n}", root=s, repo=f"r{n % 2}", path=s, status="possible", site=s)) for n, s in enumerate(nasty)]
    items, facts, units = many(specs)
    for s in nasty:
        facts[("CallSite", s)] = {"unit": "u0"}
    plain = group.group_items(items, facts, units)
    folded = group.group_items(items, facts, units, {"python"})
    assert len(plain.items) == len(nasty) and len({p.root for p in plain.items}) == len(nasty)
    assert len(folded.groups) == len(nasty) and {g.key for g in folded.groups} == set(nasty)
    assert order(folded) == []


# ---------------------------------------------------------------------------------------------- gaps the mutation run found

def test_a_malformed_pair_raises_even_after_a_pair_that_does_not_fold():
    with pytest.raises(ValueError):
        group.fold_languages({"x": [(1, 60), (True, 60)]})
    with pytest.raises(ValueError):
        group.fold_languages({"x": [(100, 100)], "y": [(0, 60), (61, 60)]})


def test_a_candidate_whose_site_names_no_unit_is_listed_not_folded():
    items, facts, units = many([("x", dict(site="c", status="possible"))])
    facts[("CallSite", "c")] = {}
    got = group.group_items(items, facts, units, {"python"})
    assert got.groups == () and order(got) == ["x"]


def test_a_candidate_site_with_two_call_site_ends_is_ambiguous_and_listed():
    chain = [hop("CallSite", "c1", "value"), hop("CallSite", "c2", "value", edge("CALLS_NAME", "->", "possible", "resolution-unverified"))]
    item, facts, units = synth("x", status="possible", chain=chain)
    assert group.candidate_site(item) is None
    assert group.group_items([item], facts, units, {"python"}).groups == ()


def test_the_site_of_a_candidate_is_the_call_site_at_either_end_of_the_hop():
    imp = [hop("Symbol", "anchor", "value"), hop("CallSite", "c", "path", edge("CALLS_NAME", "<-", "possible", "resolution-unverified")), hop("Symbol", "b", "end", edge("HAS_CALL", "<-"))]
    dep = [hop("Symbol", "anchor", "value"), hop("CallSite", "d", "value", edge("HAS_CALL", "->")), hop("Symbol", "x", "value", edge("CALLS_NAME", "->", "possible", "resolution-unverified"))]
    assert group.candidate_site(synth("b", status="possible", chain=imp)[0]) == "c"
    assert group.candidate_site(synth("x", status="possible", chain=dep)[0]) == "d"
    assert group.candidate_site(synth("a")[0]) is None


def test_the_resolved_call_sites_unit_is_read_even_when_no_item_lives_in_it_and_needed_units_names_it():
    g = candidates_graph()
    g.sym("T9", unit="ub", repo="repo-b", path="t9.py")
    t0 = next(n for n in g.rows.nodes["Symbol"] if n["symbolId"] == "T0")
    t0["unitId"], t0["repoId"] = "ub", "repo-b"                  # so that no item lives in ``ua``, the site's unit
    g.rows.edges[("CallSite", "CALLS", "Symbol")] = [e for e in g.rows.edges[("CallSite", "CALLS", "Symbol")] if e["t"] != "T0"] + [{"s": "c0", "t": "T9"}]
    w = bfs.walk(MemoryReader(g), ("Symbol", "F"), column=bfs.DEPENDS_ON)
    assert {i.id for i in w.items()} >= {"T9", "T0"}     # T0 is now a name-match candidate too: the site's resolved callee is T9, in ``ub``
    assert group.needed_units(w.items(), w.facts, folding=True) == {"ua", "ub"}
    # no item lives in ``ua`` once T0 is excluded: ask for the items of ``ub`` alone
    only_ub = [i for i in w.items() if w.facts[("Symbol", i.id)]["unit"] == "ub"]
    assert group.needed_units(only_ub, w.facts, folding=True) == {"ua", "ub"} and group.needed_units(only_ub, w.facts, folding=False) == {"ub"}
    listing_ = walk_listing(g, question="depends-on", fold_languages={"python"})
    assert [(x.key, x.count) for x in listing_.groups] == [("c0", 3)]


def test_two_items_of_one_unit_that_name_different_repositories_are_two_sections():
    items, facts, units = many([("a", dict(unit="u", repo="r1")), ("b", dict(unit="u", repo="r2"))])
    assert [(s.repo, s.unit) for s in group.group_items(items, facts, units).sections()] == [("r1", "u"), ("r2", "u")]


@pytest.mark.parametrize("row", [("u", "r", "root"), ("u", "r", 7, "python", None, None, None, None, None, True), ("u", "r", "root", 5, None, None, None, None, None, True),
                                 (3, "r", "root", "python", None, None, None, None, None, True)])
def test_a_malformed_units_row_is_a_graph_data_error(row):
    class R:
        def read(self, name, params=None):
            assert name == "units"
            return [row]
    with pytest.raises(GraphDataError):
        group.unit_refs(R(), ["u"])


def test_unit_refs_reads_root_and_language_once_and_nothing_for_no_ids():
    class R:
        calls = []

        def read(self, name, params=None):
            self.calls.append((name, params))
            return [("u2", "r", "root/2", None, None, None, None, None, None, True), ("u1", "r", "", "go", None, None, None, None, None, False)]
    r = R()
    assert group.unit_refs(r, []) == {} and r.calls == []
    assert group.unit_refs(r, ["u2", "u1", "u1"]) == {"u1": UnitRef("", "go"), "u2": UnitRef("root/2", None)}
    assert r.calls == [("units", {"ids": ["u1", "u2"]})]


# ---------------------------------------------------------------------------------------------- review F1/F4/F5: own arrival, short-circuit, sections

CAND = edge("CALLS_NAME", "<-", "possible", "resolution-unverified")
CAND_OUT = edge("CALLS_NAME", "->", "possible", "resolution-unverified")


def impact_chain(*tail):
    return [hop("Symbol", "anchor", "value"), hop("CallSite", "cB", "path", CAND), hop("Symbol", "B", "end", edge("HAS_CALL", "<-")), *tail]


def depends_chain(*tail):
    return [hop("Symbol", "anchor", "value"), hop("CallSite", "cF", "value", edge("HAS_CALL", "->")), hop("Symbol", "B", "value", CAND_OUT), *tail]


@pytest.mark.parametrize("chain,expect", [
    (impact_chain(), "cB"),
    (depends_chain(), "cF"),
    (impact_chain(hop("CallSite", "cC", "path", edge("CALLS", "<-")), hop("Symbol", "C", "end", edge("HAS_CALL", "<-"))), None),   # a caller of the candidate
    (impact_chain(hop("Symbol", "C", "end", edge("HAS_CALL", "<-"))), None),                                                         # a second HAS_CALL <- (two-hop tail)
    (depends_chain(hop("CallSite", "cT", "value", edge("HAS_CALL", "->")), hop("Symbol", "T", "value", edge("CALLS", "->"))), None),  # the candidate's own callee
    (depends_chain(hop("CallSite", "cT", "value", edge("HAS_CALL", "->"))), None),                                                    # HAS_CALL -> tail
    (depends_chain(hop("Symbol", "T", "value", edge("HAS_CALL", "<-"))), None),                                                      # a <- owner tail on a -> hop
    ([*impact_chain()[:2], hop("Symbol", "B", "end", edge("CALLS", "<-"))], None),                                                  # one tail step of another edge type
    ([*impact_chain()[:2], hop("Symbol", "B", "end", edge("HAS_CALL", "->"))], None),                                                # one HAS_CALL tail step in the wrong direction
    ([*impact_chain()[:2], hop("Symbol", "B", "end", None)], None),                                                                  # a malformed tail step with no edge
    (impact_chain()[:2], None),                                                                                                       # impact hop with no owner step
    ([hop("Symbol", "anchor", "value"), hop("Symbol", "m", "value", edge("CALLS", "->", "possible", "name-match")), hop("CallSite", "cB", "path", CAND),
      hop("Symbol", "B", "end", edge("HAS_CALL", "<-"))], None),                                                                    # HAS_CALL <- after a non-CALLS_NAME first possible hop
])
def test_only_an_item_whose_own_arrival_is_the_candidate_hop_has_a_site(chain, expect):
    assert group.candidate_site(synth("x", status="possible", chain=chain)[0]) == expect


@pytest.mark.parametrize("column", ["impact", "depends-on"])
def test_an_item_downstream_of_a_candidate_stays_listed_ungrouped_and_the_group_holds_only_the_candidate(column):
    if column == "impact":
        chains = {"B": impact_chain(), "C": impact_chain(hop("CallSite", "cC", "path", edge("CALLS", "<-")), hop("Symbol", "C", "end", edge("HAS_CALL", "<-")))}
        sites = ("cB", "cC")
    else:
        chains = {"B": depends_chain(), "T": depends_chain(hop("CallSite", "cT", "value", edge("HAS_CALL", "->")), hop("Symbol", "T", "value", edge("CALLS", "->")))}
        sites = ("cF", "cT")
    items, facts, units = [], {}, {}
    for ident, chain in chains.items():
        i, f, u = synth(ident, status="possible", chain=chain, distance=len(chain) // 2)
        items.append(i), facts.update(f), units.update(u)
    for s in sites:
        facts[("CallSite", s)] = {"unit": "u"}
    got = group.group_items(items, facts, units, {"python"})
    assert [(g.key, [m.item.id for m in g.members]) for g in got.groups] == [(sites[0], ["B"])]
    (other,) = [i for i in chains if i != "B"]
    assert order(got) == [other]
    assert (got.items[0].item.status, got.items[0].item.reason) == ("possible", "resolution-unverified")


def downstream_graph():
    g = G(unit="ua")
    set_root(g, "ua", "r")
    for s in ("S", "T", "B", "C", "F", "K"):
        g.sym(s)
    g.site("cB", "B"), g.calls("cB", "T"), g.calls_name("cB", "S", name="n1"), g.site("cC", "C"), g.calls("cC", "B")
    g.ret("rB", "B"), g.data("Ret", "rB", "CallSite", "cB", "r", "ret")
    return g


def test_a_real_impact_walk_lists_the_caller_of_a_candidate_ungrouped():
    got = walk_listing(downstream_graph(), ("Symbol", "S"), fold_languages={"python"})
    assert [(g.key, [m.item.id for m in g.members]) for g in got.groups] == [("cB", ["B"])]
    assert [(p.item.id, p.item.distance, p.item.reason) for p in got.items] == [("C", 2, "resolution-unverified")]


def test_a_real_depends_on_walk_lists_the_candidates_own_callee_ungrouped():
    g = G(unit="ua")
    set_root(g, "ua", "r")
    for s in ("F", "A", "B", "T"):
        g.sym(s)
    g.site("cF", "F"), g.calls("cF", "A"), g.calls_name("cF", "B", name="n"), g.site("cB", "B"), g.calls("cB", "T")
    got = walk_listing(g, ("Symbol", "F"), question="depends-on", fold_languages={"python"})
    assert [(x.key, [m.item.id for m in x.members]) for x in got.groups] == [("cF", ["B"])]
    assert [(p.item.id, p.item.distance, p.item.reason) for p in got.items] == [("A", 1, None), ("T", 2, "resolution-unverified")]


def test_with_no_fold_languages_a_site_whose_unit_has_no_unit_row_is_never_looked_up():
    items, facts, units = many([("x", dict(site="c", status="possible"))])
    facts[("CallSite", "c")] = {"unit": "ghost"}
    assert order(group.group_items(items, facts, units)) == ["x"]
    with pytest.raises(GraphDataError):
        group.group_items(items, facts, units, {"python"})
    g = candidates_graph()
    r = MemoryReader(g)
    assert [p.item.id for p in group.listing(r, ("Symbol", "F"), question="depends-on").items] == ["T0", "X0a", "X0b"]


def test_a_line_that_is_a_bool_is_refused_like_a_missing_line():
    items, facts, units = many([("a", dict())])
    facts[("Symbol", "a")]["line"] = True
    with pytest.raises(GraphDataError):
        group.group_items(items, facts, units)


def test_the_sort_key_is_the_spec_6_row_and_has_no_trailing_kind_slot():
    p = group.group_items(*many([("a", dict(distance=2, status="possible", repo="r", unit="u", path="p", line=3, root="q"))])).items[0]
    assert p.sort_key() == ("r", "q", "u", 2, 1, "p", 3, "a")


def test_sections_build_one_section_object_per_section_not_one_per_item(monkeypatch):
    built = []
    real = group.Section
    monkeypatch.setattr(group, "Section", lambda *a: built.append(a) or real(*a))
    n = 2000
    items, facts, units = many([(f"i{k:06d}", dict()) for k in range(n)])
    got = group.group_items(items, facts, units)
    secs = got.sections()
    assert len(secs) == 1 and [p.item.id for p in secs[0].items] == [f"i{k:06d}" for k in range(n)]
    assert isinstance(secs[0].items, tuple) and len(built) == 1


def test_a_section_after_the_first_collects_all_its_items():
    items, facts, units = many([("a", dict(unit="u1", root="a")), ("b1", dict(unit="u2", root="b", line=1)), ("b2", dict(unit="u2", root="b", line=2)), ("b3", dict(unit="u2", root="b", line=3))])
    assert [(s.unit, [p.item.id for p in s.items]) for s in group.group_items(items, facts, units).sections()] == [("u1", ["a"]), ("u2", ["b1", "b2", "b3"])]
