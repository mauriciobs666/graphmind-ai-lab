"""P1-22 ``ccg.engine.state``: the state kinds, access-path overlap (``-bundle-spec3.md`` §1.1, §5.1) and the argument-to-parameter match of the ``arg(i)`` rows
(``docs/plans/connected-code-graph-p1-answers.md`` P1-22 (4): receiver as index -1, ``kw.<n>`` by ``Param.name``)."""
from __future__ import annotations

import pytest

from ccg.engine import state as S


def test_the_state_kinds_are_the_closed_set_of_the_plan():
    assert S.KINDS == ("value", "path", "contract", "key", "end", "site", "function")
    assert S.VALUE == ("value", "") and S.CONTRACT == ("contract", "") and S.KEY == ("key", "") and S.END == ("end", "") and S.FUNCTION == ("function", "")


@pytest.mark.parametrize("a,b,want", [
    ("p0", "p0", True), ("p0", "p0.path", True), ("p0.path", "p0", True), ("p0.path", "p0.method", False),
    ("p1", "p10", False), ("p10", "p1", False), ("p0", 'p0["a.b"]', True), ('p0["a"]', 'p0["a.b"]', False), ("r", "r.x", True), ("r", "ret", False),
])
def test_path_overlap_is_equal_or_a_segment_prefix(a, b, want):
    assert S.path_overlaps(a, b) is want and S.path_overlaps(b, a) is want


def test_a_value_state_matches_every_data_edge_and_a_path_state_only_an_overlapping_one():
    assert S.data_edge_matches(S.VALUE, "p9.q")
    assert S.data_edge_matches(("path", "p0.path"), "p0")
    assert not S.data_edge_matches(("path", "p0.path"), "p0.method")


@pytest.mark.parametrize("path,index,name,want", [
    ("a0", 0, "x", "p0"), ("a0.q.r", 0, "x", "p0.q.r"), ("a2", 2, "z", "p2"), ("a1", 0, "x", None), ("a10", 1, "x", None), ("a1", 10, "x", None),
    ("recv", -1, "self", "recv"), ("recv.f", -1, "self", "recv.f"), ("recv", 0, "x", None), ("a0", -1, "self", None),
    ("kw.prefix", 3, "prefix", "p3"), ("kw.prefix.k", 3, "prefix", "p3.k"), ("kw.prefix", 3, "other", None), ('kw["café"]', 1, "café", "p1"),
    ('kw["café"].z', 1, "café", "p1.z"), ("kw.a", -1, "a", None),
    ("p0", 0, "x", None), ("r", 0, "x", None), ("ret", 0, "x", None), ("", 0, "x", None), ("a", 0, "x", None), ("a01", 1, "x", None), ("a0x", 0, "x", None),
])
def test_an_argument_path_matches_exactly_the_parameter_it_names(path, index, name, want):
    assert S.param_path(path, index, name) == want


def test_a_receiver_is_never_matched_by_a_positional_argument_and_kw_never_matches_the_receiver():
    assert S.param_path("a0", -1, "self") is None
    assert S.param_path("kw.self", -1, "self") is None


def test_a_malformed_bracket_key_matches_nothing_and_never_raises():
    for bad in ('kw["', 'kw["a', 'kw[1]', 'kw["a"', 'kw["\\ud800"]'):
        assert S.param_path(bad, 0, "a") is None


@pytest.mark.parametrize("path,want", [
    ("a0", True), ("a12.q", True), ("recv", True), ("recv.f", True), ("kw.prefix", True), ('kw["café"]', True), ('a1["x-y"]', True),
    ("r", False), ("r.x", False), ("ret", False), ("p0", False), ("p0.a", False), ("", False), ("a", False), ("a01", False), ("kw", False), ("kw.", False), ("recvx", False),
])
def test_only_an_argument_path_enters_a_callee(path, want):
    assert S.is_arg_path(path) is want


def test_a_site_state_carries_the_site_id_as_its_path():
    assert S.site_state("cs1") == ("site", "cs1") and "site" in S.KINDS
