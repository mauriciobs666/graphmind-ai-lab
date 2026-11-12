"""P1-21 anchors, unit half: the resolvers over a scripted ``Reader`` (the live half is ``tests/test_engine_reader_live.py``).

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-21 behaviours (2)-(6); ``docs/plans/connected-code-graph-p0-graph.md`` §1.3 (the three
repo-scoped location reads), §1.4 (the File+line rule: narrowest containing interval, span or owned region; a span match beats a region match, then the
larger ``depth``); ``-p1-graph.md`` §5 (the anchor texts)."""
from __future__ import annotations

import unicodedata

import pytest
from engine_helpers import ScriptedReader

from ccg.engine import anchors as A
from ccg.engine import config
from ccg.engine.reader import GraphDataError

SYM = "0123456789abcdef0123456789abcdef"


def _sym(n, qname, start, end, regions=None, depth=0):
    """A ``symbols_of_file`` row: ``(symbolId, qname, lineStart, lineEnd, ownedRegions, depth)``."""
    return (f"s{n:02d}", qname, start, end, None if regions is None else tuple(regions), depth)


def _reader(symbols=(), *, repo="repo-a", file_rows=None, skipped=None, repos=("repo-a",)):
    script = {
        "repo_exists": lambda p: [(r,) for r in repos if r == p["repo"]],
        "file_by_path": lambda p: (file_rows if file_rows is not None else [("f1", "u1")]) if p["repo"] == repo else [],
        "skipped_by_path": lambda p: skipped or [],
        "symbols_of_file": lambda p: list(symbols),
    }
    return ScriptedReader(script)


# ------------------------------------------------------------------------------------- inputs and the repo

@pytest.mark.parametrize("kw", [
    {"repo": "Repo-A"}, {"repo": ""}, {"repo": "a b"}, {"repo": "a'}) DELETE"}, {"repo": None}, {"repo": 5}, {"repo": "a\n"},
    {"path": ""}, {"path": None}, {"path": "a\x00b"}, {"path": "x" * 5000}, {"path": 7},
    {"line": 0}, {"line": -3}, {"line": "4"}, {"line": 2.5}, {"line": True},
    {"symbol": ""}, {"symbol": 5},
])
def test_a_bad_input_is_a_clean_error_before_any_read(kw):
    r = _reader()
    args = {"repo": "repo-a", "path": "app/a.py", "line": 1} | kw
    with pytest.raises(A.AnchorError):
        A.resolve_location(r, **args)
    assert r.calls == []


def test_a_line_and_a_symbol_together_are_refused():
    r = _reader()
    with pytest.raises(A.AnchorError, match="line.*symbol|symbol.*line"):
        A.resolve_location(r, "repo-a", "app/a.py", line=3, symbol="m.f")
    assert r.calls == []


def test_an_unknown_repo_is_a_clean_error_after_the_repo_read_and_before_the_path_reads():
    r = _reader()
    with pytest.raises(A.UnknownRepo):
        A.resolve_location(r, "repo-z", "app/a.py", line=3)
    assert r.names() == ["repo_exists"]


# ------------------------------------------------------------------------------------- the three reads, in order

def test_a_mapped_file_stops_after_the_file_read_and_reads_no_symbol_without_a_line():
    r = _reader()
    got = A.resolve_location(r, "repo-a", "app/a.py")
    assert r.names() == ["repo_exists", "file_by_path"]
    assert (got.state, got.unit_id, got.reason, got.status, got.anchors) == ("mapped", "u1", None, "unresolved", ())


def test_a_skipped_file_is_not_mapped_with_its_reason_and_unit():
    r = _reader(file_rows=[], skipped=[("u7", "test")])
    got = A.resolve_location(r, "repo-a", "tests/test_a.py", line=3)
    assert r.names() == ["repo_exists", "file_by_path", "skipped_by_path"]
    assert (got.state, got.unit_id, got.reason, got.status, got.anchors) == ("not mapped", "u7", "test", "unresolved", ())


def test_a_path_in_neither_set_is_not_found_a_declaration_file_included():
    r = _reader(file_rows=[])
    for path in ("app/types.d.ts", "stubs/x.pyi", "vendor/lib/x.py"):
        got = A.resolve_location(r, "repo-a", path, line=1)
        assert (got.state, got.unit_id, got.reason, got.status) == ("not found", None, None, "unresolved"), path
    assert "symbols_of_file" not in r.names()


def test_the_path_reads_are_repo_scoped_the_repo_and_path_travel_as_parameters_untouched():
    r = _reader(repos=("repo-a", "repo-b"), repo="repo-b", file_rows=[("fb", "ub")])
    got = A.resolve_location(r, "repo-b", "app/a.py")
    assert got.unit_id == "ub"
    assert r.calls[1] == ("file_by_path", {"repo": "repo-b", "path": "app/a.py"})
    got = A.resolve_location(_reader(repos=("repo-a", "repo-b"), repo="repo-b"), "repo-a", "app/a.py")
    assert got.state == "not found"                       # the same path in the other repo is not this repo's file


def test_two_files_for_one_path_are_refused_not_guessed():
    r = _reader(file_rows=[("f1", "u1"), ("f2", "u2")])
    with pytest.raises(GraphDataError):
        A.resolve_location(r, "repo-a", "app/a.py", line=1)


# ------------------------------------------------------------------------------------- the file+line rule

def test_the_narrowest_containing_span_wins():
    r = _reader([_sym(1, "m.C", 1, 100), _sym(2, "m.C.f", 10, 30, depth=1), _sym(3, "m.C.f.g", 15, 20, depth=2), _sym(4, "m.h", 40, 50)])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=17)
    assert got.status == "resolved" and [a.symbol_id for a in got.anchors] == ["s03"] and got.via == "span"
    assert A.resolve_location(r, "repo-a", "app/a.py", line=25).anchors[0].symbol_id == "s02"
    assert A.resolve_location(r, "repo-a", "app/a.py", line=60).anchors[0].symbol_id == "s01"


def test_the_interval_ends_are_inclusive_and_the_line_one_outside_is_not_in():
    r = _reader([_sym(1, "m.f", 10, 20)])
    for line, ok in ((9, False), (10, True), (20, True), (21, False)):
        got = A.resolve_location(r, "repo-a", "app/a.py", line=line)
        assert (got.status == "resolved") is ok, line
    assert A.resolve_location(r, "repo-a", "app/a.py", line=9).detail == "no-symbol-at-line"


def test_an_owned_region_outside_the_span_still_anchors_and_beats_a_wider_enclosing_span():
    r = _reader([_sym(1, "m.Outer", 1, 200), _sym(2, "m.Decorated", 80, 100, regions=[30, 35])])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=32)
    assert [a.symbol_id for a in got.anchors] == ["s02"] and got.via == "owned-region"
    assert A.resolve_location(r, "repo-a", "app/a.py", line=90).anchors[0].symbol_id == "s02"
    assert A.resolve_location(r, "repo-a", "app/a.py", line=36).anchors[0].symbol_id == "s01"


def test_the_ends_of_an_owned_region_are_inclusive_too():
    r = _reader([_sym(1, "m.Outer", 1, 200), _sym(2, "m.f", 80, 100, regions=[30, 35])])
    for line, want in ((29, "s01"), (30, "s02"), (35, "s02"), (36, "s01")):
        assert A.resolve_location(r, "repo-a", "app/a.py", line=line).anchors[0].symbol_id == want, line


def test_every_owned_region_of_a_symbol_counts_not_only_the_first():
    r = _reader([_sym(1, "m.Outer", 1, 200), _sym(2, "m.f", 80, 100, regions=[30, 35, 50, 55])])
    assert A.resolve_location(r, "repo-a", "app/a.py", line=52).anchors[0].symbol_id == "s02"


def test_at_equal_width_a_span_match_beats_an_owned_region_match():
    r = _reader([_sym(1, "m.viaRegion", 80, 100, regions=[30, 35]), _sym(2, "m.viaSpan", 30, 35)])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=32)
    assert [a.symbol_id for a in got.anchors] == ["s02"] and got.via == "span"
    r = _reader([_sym(2, "m.viaSpan", 30, 35), _sym(1, "m.viaRegion", 80, 100, regions=[30, 35])])     # whatever the read order
    assert A.resolve_location(r, "repo-a", "app/a.py", line=32).anchors[0].symbol_id == "s02"


def test_a_span_match_beats_a_deeper_symbol_that_matches_only_through_a_region_of_the_same_width():
    r = _reader([_sym(1, "m.viaRegion", 80, 100, regions=[30, 35], depth=3), _sym(2, "m.viaSpan", 30, 35, depth=0)])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=32)
    assert [a.symbol_id for a in got.anchors] == ["s02"] and got.via == "span"


@pytest.mark.parametrize("line", [30, 32, 34])
def test_a_region_one_line_narrower_than_a_rival_span_beats_it_over_the_whole_region_including_its_ends(line):
    """Width is compared before kind: region [30,34] (width 4) against span [30,35] (width 5)."""
    r = _reader([_sym(1, "m.rivalSpan", 30, 35), _sym(2, "m.regionOwner", 80, 100, regions=[30, 34])])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=line)
    assert [a.symbol_id for a in got.anchors] == ["s02"] and got.via == "owned-region"
    assert A.resolve_location(r, "repo-a", "app/a.py", line=35).anchors[0].symbol_id == "s01"      # one past the region: only the span holds it


@pytest.mark.parametrize(("region", "rival", "winner", "via"), [
    ((30, 34), (30, 35), "s02", "owned-region"),            # region narrower by one
    ((30, 35), (30, 35), "s01", "span"),                    # equal widths: the span wins
    ((30, 36), (30, 35), "s01", "span"),                    # region wider by one
    ((31, 35), (30, 35), "s02", "owned-region"),            # narrower by one, boundary at the shared end
])
def test_region_against_rival_span_width_delta(region, rival, winner, via):
    r = _reader([_sym(1, "m.rivalSpan", *rival), _sym(2, "m.regionOwner", 80, 100, regions=list(region))])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=35 if region[1] >= 35 else 34)
    assert [a.symbol_id for a in got.anchors] == [winner] and got.via == via


def test_a_symbol_whose_span_and_own_region_both_hold_the_line_is_anchored_by_the_narrower_region():
    r = _reader([_sym(1, "m.f", 30, 60, regions=[40, 44])])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=42)
    assert got.anchors[0].symbol_id == "s01" and got.via == "owned-region"
    assert A.resolve_location(r, "repo-a", "app/a.py", line=50).via == "span"


def test_a_narrower_interval_beats_a_deeper_symbol_and_a_span_match():
    r = _reader([_sym(1, "m.wideDeep", 30, 40, depth=5), _sym(2, "m.narrowShallow", 80, 100, regions=[31, 33], depth=0)])
    assert A.resolve_location(r, "repo-a", "app/a.py", line=32).anchors[0].symbol_id == "s02"


def test_at_equal_width_and_kind_the_larger_depth_wins():
    r = _reader([_sym(1, "m.shallow", 40, 45, depth=0), _sym(2, "m.deep", 40, 45, depth=2), _sym(3, "m.mid", 40, 45, depth=1)])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=42)
    assert [a.symbol_id for a in got.anchors] == ["s02"]


def test_a_tie_the_rule_cannot_break_is_ambiguous_with_every_tied_symbol_as_an_option_never_a_pick():
    r = _reader([_sym(2, "m.b", 90, 95, depth=1), _sym(1, "m.a", 90, 95, depth=1), _sym(3, "m.c", 80, 120)])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=92)
    assert got.status == "ambiguous" and got.anchors == () and [o.symbol_id for o in got.options] == ["s01", "s02"] and got.options_total == 2


def test_the_options_of_an_ambiguity_are_capped_and_the_total_says_so():
    n = config.OPTIONS_MAX + 5
    r = _reader([_sym(i, f"m.f{i}", 5, 6) for i in range(n)])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=5)
    assert got.status == "ambiguous" and len(got.options) == config.OPTIONS_MAX and got.options_total == n
    assert [o.symbol_id for o in got.options] == sorted(o.symbol_id for o in got.options)


def test_the_anchor_carries_qname_repo_path_unit_and_symbol_id_and_the_text_is_data_untouched():
    nasty = "m.f' }) DETACH DELETE (n) //  ‮"
    r = _reader([_sym(1, nasty, 1, 9)])
    got = A.resolve_location(r, "repo-a", "app/a.py", line=3)
    a, = got.anchors
    assert (a.symbol_id, a.qname, a.repo_id, a.path, a.unit_id, a.line_start, a.line_end) == ("s01", nasty, "repo-a", "app/a.py", "u1", 1, 9)


def test_a_symbol_name_in_the_file_anchors_by_exact_qname_only():
    r = _reader([_sym(1, "m.C", 1, 50), _sym(2, "m.C.f", 10, 20, depth=1), _sym(3, "m.f", 60, 70)])
    got = A.resolve_location(r, "repo-a", "app/a.py", symbol="m.C.f")
    assert got.status == "resolved" and [a.symbol_id for a in got.anchors] == ["s02"] and got.via == "qname"
    for miss in ("f", "m.C.F", "m.", "M.C.f"):
        got = A.resolve_location(r, "repo-a", "app/a.py", symbol=miss)
        assert (got.status, got.detail) == ("unresolved", "no-such-symbol"), miss


def test_one_qname_twice_in_a_file_is_ambiguous():
    r = _reader([_sym(1, "m.f", 1, 5), _sym(2, "m.f", 10, 15)])
    got = A.resolve_location(r, "repo-a", "app/a.py", symbol="m.f")
    assert got.status == "ambiguous" and [o.symbol_id for o in got.options] == ["s01", "s02"]


@pytest.mark.parametrize("row", [
    ("s01", "m.f", "1", 9, None, 0), ("s01", "m.f", 1, None, None, 0), ("s01", None, 1, 9, None, 0), (None, "m.f", 1, 9, None, 0),
    ("s01", "m.f", 9, 1, None, 0), ("s01", "m.f", 1, 9, (1, 2, 3), 0), ("s01", "m.f", 1, 9, (5, 3), 0), ("s01", "m.f", 1, 9, ("a", "b"), 0),
    ("s01", "m.f", True, 9, None, 0), ("s01", "m.f", 1, 9, None, "x"), ("s01", "m.f", 1, 9), ("s01", "m.f", 1, 9, None, 0, "extra"),
])
def test_a_malformed_symbol_row_is_refused_not_repaired(row):
    with pytest.raises(GraphDataError):
        A.resolve_location(_reader([row]), "repo-a", "app/a.py", line=3)


def test_a_malformed_row_error_carries_no_row_text():
    sentinel = "SENTINEL-GRAPH-DATA-7f3a"
    for bad in ([(sentinel, sentinel, "1", 9, None, 0)], [(sentinel, "m.f", 9, 1, None, 0)], [(sentinel, "m.f", 1, 9, (5, 3), 0)], [(sentinel, "m.f", 1, 9, (sentinel,), 0)]):
        with pytest.raises(GraphDataError) as ei:
            A.resolve_location(_reader(bad), "repo-a", "app/a.py", line=3)
        assert sentinel not in str(ei.value) and sentinel not in repr(ei.value)
    with pytest.raises(GraphDataError) as ei:
        A.resolve_location(_reader(file_rows=[(sentinel, "u1"), (sentinel, "u2")]), "repo-a", "app/a.py", line=3)
    assert sentinel not in str(ei.value)
    with pytest.raises(GraphDataError) as ei:
        A.resolve_contract(ScriptedReader({"cp_by_key": [(sentinel, sentinel)]}), "k")
    assert sentinel not in str(ei.value) and sentinel not in repr(ei.value)


def test_a_missing_depth_reads_as_zero_and_missing_regions_as_none():
    r = _reader([("s01", "m.f", 1, 9, None, None)])
    assert A.resolve_location(r, "repo-a", "app/a.py", line=3).anchors[0].symbol_id == "s01"


# ------------------------------------------------------------------------------------- symbol anchors by qname and by name

def test_a_qname_in_a_repo_resolves_to_one_symbol_with_its_qname():
    r = ScriptedReader({"repo_exists": [("repo-a",)], "symbol_by_qname": [("s1", "u1", "app/a.py", 3, 9, None)]})
    got = A.resolve_symbol(r, "repo-a", "m.f")
    assert got.status == "resolved" and got.anchors[0] == A.SymbolAnchor("s1", "m.f", "repo-a", "app/a.py", "u1", 3, 9)
    assert r.calls[1] == ("symbol_by_qname", {"repo": "repo-a", "qname": "m.f"})
    r2 = ScriptedReader({"repo_exists": [("repo-b",)]})
    A.resolve_symbol(r2, "repo-b", "m.g")
    assert r2.calls == [("repo_exists", {"repo": "repo-b"}), ("symbol_by_qname", {"repo": "repo-b", "qname": "m.g"})]


def test_a_qname_with_no_symbol_is_unresolved_and_with_several_is_ambiguous_and_an_unknown_repo_is_an_error():
    none = ScriptedReader({"repo_exists": [("repo-a",)]})
    assert A.resolve_symbol(none, "repo-a", "m.f").status == "unresolved"
    many = ScriptedReader({"repo_exists": [("repo-a",)], "symbol_by_qname": [("s2", "u1", "p", 1, 2, None), ("s1", "u1", "q", 3, 4, None)]})
    got = A.resolve_symbol(many, "repo-a", "m.f")
    assert got.status == "ambiguous" and [o.symbol_id for o in got.options] == ["s1", "s2"]
    with pytest.raises(A.UnknownRepo):
        A.resolve_symbol(ScriptedReader(), "repo-z", "m.f")


def test_a_name_resolves_across_repos_by_the_normalized_name_and_several_hits_are_options():
    rows = [("s2", "repo-b", "u2", "pkg.Straße", "b.py", 3), ("s1", "repo-a", "u1", "pkg.Straße", "a.py", 4)]
    r = ScriptedReader({"symbol_by_name": rows})
    got = A.resolve_symbol_by_name(r, "Straße")
    assert r.calls == [("symbol_by_name", {"name": unicodedata.normalize("NFC", "straße".casefold())})]
    assert got.status == "ambiguous" and [o.symbol_id for o in got.options] == ["s1", "s2"]
    got = A.resolve_symbol_by_name(ScriptedReader({"repo_exists": [("repo-a",)], "symbol_by_name": rows}), "Straße", repo="repo-a")
    assert got.status == "resolved" and got.anchors[0].symbol_id == "s1"
    assert A.resolve_symbol_by_name(ScriptedReader({}), "nothing").status == "unresolved"


def test_a_name_in_an_unknown_repo_is_an_error_like_the_other_repo_scoped_resolvers():
    r = ScriptedReader({"repo_exists": lambda p: [(p["repo"],)] if p["repo"] == "repo-a" else [], "symbol_by_name": [("s1", "repo-a", "u1", "q", "p", 1)]})
    with pytest.raises(A.UnknownRepo):
        A.resolve_symbol_by_name(r, "x", repo="repo-z")
    assert "symbol_by_name" not in r.names()
    assert A.resolve_symbol_by_name(r, "x", repo="repo-a").status == "resolved"
    assert ("repo_exists", {"repo": "repo-a"}) in r.calls
    r = ScriptedReader({"symbol_by_name": [("s1", "repo-a", "u1", "q", "p", 1)]})
    A.resolve_symbol_by_name(r, "x")                                   # no repo given: no repo read
    assert r.names() == ["symbol_by_name"]


def test_a_symbol_row_without_a_path_keeps_it_none_not_an_invented_empty_path():
    r = ScriptedReader({"symbol_by_name": [("s1", "repo-a", None, "q", None, 1)]})
    assert A.resolve_symbol_by_name(r, "x").anchors[0].path is None


def test_a_name_that_is_blank_after_normalization_is_refused_before_any_read():
    r = ScriptedReader({"symbol_by_name": [("s1", "repo-a", "u1", "q", "p", 1)]})
    for blank in ("   ", "\t\n ", "\u2003"):
        with pytest.raises(A.AnchorError):
            A.resolve_symbol_by_name(r, blank)
    assert r.calls == []


def test_the_name_is_normalized_like_the_bundle_does_nfc_casefold_nfc_strip():
    assert A.name_norm("Ae\u0301") == "a\u00e9" and A.name_norm("\u00c9") == "\u00e9" and A.name_norm(" X ") == "x"
    assert A.name_norm("  Ab́C ") == unicodedata.normalize("NFC", unicodedata.normalize("NFC", "  Ab́C ").casefold()).strip()
    assert A.name_norm("T\u0308") == "\u1e97"                      # the outer NFC: casefold alone leaves t + U+0308
    assert A.name_norm("\u0345\u0300") == "\u0300\u03b9"          # the inner NFC: composing before the fold would give U+1F76
    assert A.name_norm("ǅ") == unicodedata.normalize("NFC", "ǅ".casefold())


@pytest.mark.parametrize("name", ["\u01f0x", "\u0390", "Stra\u00dfe", "\u0130stanbul", "T\u00e9", "\u01c5", "\u1e97", "plain"])
def test_name_norm_is_what_the_bundle_validator_accepts_and_the_plain_casefold_is_what_it_rejects(name):
    """Drift guard (the rule lives in ``ccg.bundle`` too): a symbol stamped with our ``name_norm`` passes the bundle's own ``nameNorm`` check."""
    import test_bundle as TB
    from ccg import bundle, ids
    norm = A.name_norm(name)
    qname = TB.P_CLIENT + "::" + name
    sym = TB.symbol_rec(ids.symbol_id(TB.LANG, TB.REPO, qname), qname, "function", name, 20, 30, TB.S_MOD, nameNorm=norm)
    bundle.validate(TB.ordered(TB.counted(TB.rich_records() + [sym])))
    if norm != name.casefold():
        with pytest.raises(bundle.BundleError, match="nameNorm"):
            bundle.validate(TB.ordered(TB.counted(TB.rich_records() + [{**sym, "nameNorm": name.casefold()}])))


# ------------------------------------------------------------------------------------- contract-point anchors

CPS = [("cp1", "http", "GET /shop/api/health"), ("cp2", "http", "GET /shop/api/items/{}"), ("cp3", "http", "POST /shop/api/items"),
       ("cp4", "http", "GET /shop/api/{**}"), ("cp5", "mcp_tool", "billing.invoice"), ("cp6", "mcp_tool", "billing")]


def _cp_reader(kind_rows=None):
    return ScriptedReader({
        "cp_by_key": lambda p: [r for r in CPS if r[2] == p["key"]],
        "cp_keys": lambda p: [(c, k) for c, kd, k in (kind_rows or CPS) if kd == p["kind"]],
    })


def test_a_key_that_is_stored_resolves_by_exact_keynorm_and_reads_nothing_else():
    r = _cp_reader()
    got = A.resolve_contract(r, "GET /shop/api/health")
    assert got.status == "resolved" and got.anchors == (A.ContractAnchor("cp1", "http", "GET /shop/api/health"),) and got.via == "exact"
    assert r.names() == ["cp_by_key"]


def test_a_kind_filters_the_exact_hits_and_a_key_in_two_kinds_without_a_kind_is_ambiguous():
    two = ScriptedReader({"cp_by_key": [("c1", "http", "x"), ("c2", "topic", "x")]})
    got = A.resolve_contract(two, "x")
    assert got.status == "ambiguous" and [o.cp_id for o in got.options] == ["c1", "c2"]
    assert A.resolve_contract(ScriptedReader({"cp_by_key": [("c1", "http", "x"), ("c2", "topic", "x")]}), "x", kind="topic").anchors[0].cp_id == "c2"


def test_a_concrete_key_resolves_to_the_template_that_serves_it_through_the_linker_matcher():
    r = _cp_reader()
    got = A.resolve_contract(r, "GET /shop/api/items/42", kind="http")
    assert got.status == "ambiguous" and [o.cp_id for o in got.options] == ["cp2", "cp4"]          # the specific template and the catch-all both serve it
    got = A.resolve_contract(_cp_reader(), "POST /shop/api/items/42", kind="http")
    assert got.status == "unresolved" and got.detail == "no-contract-point"
    only = _cp_reader([c for c in CPS if c[0] != "cp4"])
    got = A.resolve_contract(only, "GET /shop/api/items/42", kind="http")
    assert got.status == "resolved" and got.anchors[0].cp_id == "cp2" and got.via == "path-template"
    assert only.names() == ["cp_by_key", "cp_keys"]


def test_a_url_shaped_http_key_is_normalized_before_the_exact_read():
    r = _cp_reader()
    got = A.resolve_contract(r, "get http://h:80/shop/api/health?x=1", kind="http")
    assert got.anchors[0].cp_id == "cp1" and r.calls[0] == ("cp_by_key", {"key": "GET /shop/api/health"})


def test_contract_options_are_in_cp_id_order_whatever_the_read_order():
    r = _cp_reader([c for c in reversed(CPS)])
    got = A.resolve_contract(r, "GET /shop/api/items/42", kind="http")
    assert [o.cp_id for o in got.options] == ["cp2", "cp4"]


def test_only_the_kinds_matcher_applies_a_key_the_linker_would_not_link_is_unresolved():
    """The linker picks the matcher per kind (http path-template, otherwise the pack's, default exact): a prefix reading of ``GET /files.json`` or ``tool.sub`` anchors a point no link exists for."""
    r = ScriptedReader({"cp_by_key": [], "cp_keys": [("c1", "GET /files")]})
    got = A.resolve_contract(r, "GET /files.json", kind="http")
    assert got.status == "unresolved" and got.detail == "no-contract-point"
    got = A.resolve_contract(_cp_reader(), "billing.invoice.create", kind="mcp_tool")             # cp6 "billing" is a prefix of it, cp5 "billing.invoice" too
    assert got.status == "unresolved" and got.detail == "no-contract-point"
    got = A.resolve_contract(ScriptedReader({"cp_by_key": [], "cp_keys": [("c1", "tool")]}), "tool.sub", kind="mcp_tool")
    assert got.status == "unresolved"
    got = A.resolve_contract(ScriptedReader({"cp_by_key": [], "cp_keys": [("c1", "GET /a/{}")]}), "GET /a/b", kind="http")
    assert got.status == "resolved" and got.anchors[0].cp_id == "c1" and got.via == "path-template"


def test_an_exact_only_kind_reads_no_key_scan():
    r = _cp_reader()
    A.resolve_contract(r, "billing.invoice.create", kind="mcp_tool")
    assert r.names() == ["cp_by_key"]


def test_the_matcher_table_equals_what_the_shipped_packs_declare_and_the_linker_defaults_to():
    """The engine cannot read packs at run time; this pins its table to every shipped ``link`` row so the two cannot drift."""
    import yaml
    from pathlib import Path
    declared = {}
    for f in sorted((Path(__file__).resolve().parent.parent / "packs").glob("*/pack.yaml")):
        for d in (yaml.safe_load(f.read_text()) or {}).get("link") or []:
            assert declared.setdefault(d["kind"], d["matcher"]) == d["matcher"], d
    assert declared, "no shipped pack declares a link row"
    for kind, matcher in declared.items():
        assert A.contract_matcher(kind) == matcher, kind
    assert A.contract_matcher("http") == "path-template"                      # the linker's built-in default, whether or not a pack declares it
    assert A.contract_matcher("no_such_kind") == "exact"
    assert set(A._KIND_MATCHER) <= set(declared) | {"http"}


def test_without_a_kind_a_missing_exact_key_scans_nothing():
    r = _cp_reader()
    assert A.resolve_contract(r, "GET /shop/api/items/42").status == "unresolved"
    assert r.names() == ["cp_by_key"]


def test_a_bad_kind_or_key_is_refused_before_any_read():
    r = _cp_reader()
    for kw in ({"key": ""}, {"key": None}, {"key": "a\x00"}, {"key": "x" * 5000}, {"key": "k", "kind": "Http"}, {"key": "k", "kind": "a b"}, {"key": "k", "kind": 3}):
        with pytest.raises(A.AnchorError):
            A.resolve_contract(r, **kw)
    assert r.calls == []


def test_contract_options_are_capped():
    wide = [(f"c{i:03d}", "http", "GET /a/{**}") for i in range(config.OPTIONS_MAX + 3)]
    r = ScriptedReader({"cp_by_key": [], "cp_keys": lambda p: [(c, k) for c, _, k in wide]})
    got = A.resolve_contract(r, "GET /a/b/c", kind="http")
    assert got.status == "ambiguous" and len(got.options) == config.OPTIONS_MAX and got.options_total == config.OPTIONS_MAX + 3


# ------------------------------------------------------------------------------------- config anchors

ENTRIES = [("e9", "u1", 9, "server.port", None), ("e2", "u1", 3, "db.host", None), ("e3", "u1", 5, None, "command"), ("e4", "u1", 7, "x", "compose"),
           ("e5", "u1", 11, None, None), ("e0", "u1", 13, "db.host.name", None)]


def _cfg_reader(rows=ENTRIES):
    return ScriptedReader({"repo_exists": [("repo-a",)], "config_by_path": rows,
                           "file_by_path": [("f", "u1")], "skipped_by_path": []})


def test_a_key_path_in_a_config_file_resolves_and_a_launch_entry_is_never_an_anchor():
    r = _cfg_reader()
    got = A.resolve_config(r, "repo-a", "cfg/app.yaml", key_path="db.host")
    assert got.status == "resolved" and got.anchors == (A.ConfigAnchor("e2", "repo-a", "cfg/app.yaml", "db.host", "u1", 3),)
    assert r.calls[1] == ("config_by_path", {"repo": "repo-a", "path": "cfg/app.yaml"})
    assert A.resolve_config(_cfg_reader(), "repo-a", "cfg/app.yaml", key_path="x").status == "unresolved"       # e4 carries a launchKind
    assert A.resolve_config(_cfg_reader(), "repo-a", "cfg/app.yaml", key_path="command").detail == "no-such-key"


def test_without_a_key_path_every_non_launch_entry_is_an_option_in_key_order():
    got = A.resolve_config(_cfg_reader(), "repo-a", "cfg/app.yaml")
    assert got.status == "ambiguous" and [o.key_path for o in got.options] == ["db.host", "db.host.name", "server.port"] and got.detail == "key-path-not-given"


def test_a_config_path_with_no_entries_says_why_through_the_location_lookup():
    r = ScriptedReader({"repo_exists": [("repo-a",)], "config_by_path": [], "file_by_path": [], "skipped_by_path": [("u9", "frontend-ignore")]})
    got = A.resolve_config(r, "repo-a", "cfg/skip.yaml", key_path="a")
    assert (got.status, got.state, got.unit_id, got.reason) == ("unresolved", "not mapped", "u9", "frontend-ignore")
    r = ScriptedReader({"repo_exists": [("repo-a",)], "config_by_path": [], "file_by_path": [("f", "u1")], "skipped_by_path": []})
    got = A.resolve_config(r, "repo-a", "cfg/empty.yaml", key_path="a")
    assert (got.status, got.state, got.detail) == ("unresolved", "mapped", "no-entries")
    with pytest.raises(A.UnknownRepo):
        A.resolve_config(ScriptedReader(), "repo-z", "p", key_path="a")


@pytest.mark.parametrize("row", [("e", "u", "3", "a", None), ("e", "u", 3, 5, None), (None, "u", 3, "a", None), ("e", "u", 3, "a")])
def test_a_malformed_config_row_is_refused(row):
    with pytest.raises(GraphDataError):
        A.resolve_config(_cfg_reader([row]), "repo-a", "cfg/app.yaml", key_path="a")
