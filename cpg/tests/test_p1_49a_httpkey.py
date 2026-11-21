"""P1-49a (red first): the unknown-origin key rule of `ccg.packrt.httpkey` (rules 1-3) and the two marker predicates.

Plan: `docs/plans/connected-code-graph-p1-linking.md` P1-49 ("Two predicates", rules 1-3, "Coverage tests (49a)" items 1, 2,
5-direct); the analyst's experiments are in `docs/reviews/connected-code-graph-p1-5d-49.md` Pass 3/4.

Marker vocabulary (plan, `Corpus.alternatives`): `\\x00` (`UNKNOWN`) is a name that should resolve (importRef, ref cycle, depth
cap); `\\x01` (`UNKNOWN_VALUE`, new) is a value not statically known (parameter, environment read, call result, a `ref` with no
`constant` record, the alternatives cap). The literal characters are written out in every test on purpose: the tests never
import the markers from the code under test, so a wrong constant in the code cannot hide.

Coverage grid: `reference_split`/`reference_key` below are an independent model written from rules 1-3 only; they generate the
expected `(key, origin_dropped, unresolved_in_path)` and the status for the *live* cells of
position x scheme x remainder x count x cause, and the code is checked against them, both at `normalize_key_parts` and through
`run_packs` (the `_one` status test). Cells that cannot occur are not generated (an authority position needs a scheme; a
`//` URL with a known host is outside the rules' text, see the report). The cause axis is meaningful only for a leading whole
segment without a scheme; the test asserts that by running every other cell under both causes and requiring equal output.

Every test that needs an API that does not exist yet fails through `need()` with a message naming it (not a typo, not an
ImportError on collection), so each red is attributable.
"""
from __future__ import annotations

import itertools
import re

import pytest

from packrt_helpers import call, file_, sym, vs
from ccg.keynorm import CONSUMER, PROVIDER
from ccg.packrt import httpkey, run_packs

X = "\x01"   # UNKNOWN_VALUE: a value not statically known
Z = "\x00"   # UNKNOWN: a name that should resolve


def need(name: str):
    obj = getattr(httpkey, name, None)
    if obj is None:
        pytest.fail(f"ccg.packrt.httpkey.{name} does not exist yet (P1-49b adds it)", pytrace=False)
    return obj


# ---------------------------------------------------------------------------------------------------------------------
# the two predicates and the marker constants
# ---------------------------------------------------------------------------------------------------------------------

def test_the_unknown_value_marker_is_x01_and_the_old_marker_stays_x00():
    assert httpkey.UNKNOWN == Z
    assert need("UNKNOWN_VALUE") == X


@pytest.mark.parametrize("piece,expected", [
    (X, True), (Z, True),                       # exactly one marker: an alternative that is unknown as a whole
    (X + X, False), (Z + Z, False), (Z + X, False), (X + Z, False),     # two adjacent pieces: a string under construction
    ("", False), ("/a", False), ("a" + X, False), (X + "/shop/api/session", False), ("/a?q=" + X, False), ("{?}", False),
])
def test_is_wholly_unresolved_is_true_only_for_exactly_one_marker(piece, expected):
    assert need("is_wholly_unresolved")(piece) is expected


@pytest.mark.parametrize("s,expected", [
    (X, True), (Z, True), (X + X, True), (Z + Z, True), ("a" + X + "b", True), (X + "/shop/api/session", True),
    ("/a?q=" + X, True), ("/a/" + Z, True),
    ("", False), ("/a", False), ("{?}", False), ("GET /x", False),
])
def test_has_unresolved_is_true_when_either_marker_occurs_anywhere(s, expected):
    assert need("has_unresolved")(s) is expected


def test_the_two_predicates_differ_exactly_on_a_partly_unknown_string():
    whole, has = need("is_wholly_unresolved"), need("has_unresolved")
    for s in (X + X, Z + Z, X + "/shop/api/session", "a" + X):
        assert has(s) and not whole(s), s
    for s in (X, Z):
        assert has(s) and whole(s), s


# ---------------------------------------------------------------------------------------------------------------------
# normalize_path: a segment holding either marker becomes {?}
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected", [
    ("/a/" + X + "/b", "/a/{?}/b"), ("/a" + X + "b/c", "/{?}/c"), ("/a/" + Z, "/a/{?}"),
    ("/" + X, "/{?}"), ("/" + Z + "/x", "/{?}/x"), ("/a/" + X + X + "/b", "/a/{?}/b"),
])
def test_normalize_path_turns_a_segment_holding_either_marker_into_a_placeholder(path, expected):
    assert httpkey.normalize_path(path, CONSUMER) == expected


# ---------------------------------------------------------------------------------------------------------------------
# the reference model: rules 1-3, written from the plan text and nothing else
# ---------------------------------------------------------------------------------------------------------------------

_SCHEME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.\-]*:)?//")


def reference_split(url: str) -> tuple[str, bool, bool]:
    """``(path, origin_dropped, unresolved_in_path)`` of a consumer URL (rules 1-3)."""
    m = _SCHEME.match(url)
    if m:                                                   # rule 1: the authority is dropped, a marker in it is origin_dropped
        rest = url[m.end():]
        cut = min((i for i in (rest.find("/"), rest.find("?"), rest.find("#")) if i >= 0), default=len(rest))
        authority, remainder = rest[:cut], rest[cut:]
        dropped = X in authority or Z in authority
    else:                                                   # rule 2: a leading run of x01 followed by "/" is the origin
        k = 0
        while k < len(url) and url[k] == X:
            k += 1
        if k and url[k:k + 1] == "/":
            dropped, remainder = True, url[k:]
        else:
            dropped, remainder = False, url
    for ch in "?#":                                         # rule 3: the path is judged after the origin drop and the query cut
        cut = remainder.find(ch)
        if cut >= 0:
            remainder = remainder[:cut]
    return remainder, dropped, (X in remainder or Z in remainder)


def reference_key(verb: str, url: str) -> tuple[str, bool, bool]:
    path, dropped, unresolved = reference_split(url)
    p = path if path.startswith("/") else "/" + path
    segs = ["{?}" if (X in s or Z in s) else s for s in p.split("/")[1:]]
    return f"{verb} /" + "/".join(segs), dropped, unresolved


def reference_outcome(key: str, status: str, reason: str) -> tuple:
    """Rule 4 (P1-55), written from the plan: a consumer path whose segments are all `{?}` (at least one) is keyless."""
    segs = [x for x in key.split(" ", 1)[1].split("/") if x]
    return (None, "possible", "runtime-built") if segs and all(x == "{?}" for x in segs) else (key, status, reason)


def reference_status(url: str) -> tuple[str, str]:
    _, dropped, unresolved = reference_key("GET", url)
    return ("possible", "partial-key") if (dropped or unresolved) else ("confirmed", "key-match")


# ---------------------------------------------------------------------------------------------------------------------
# the grid: position x scheme x remainder x count x cause (live cells only)
# ---------------------------------------------------------------------------------------------------------------------

FILLS = (lambda m: m, lambda m: "a" + m, lambda m: m + ":8000", lambda m: "a" + m + "b")     # whole, prefix text, suffix text, both
REMAINDERS = ("/x", "/x/y", "", "?q=1")


def cells(marker: str, count: int):
    """Every live URL for ``marker`` repeated ``count`` times: ``(position, URL)``."""
    m = marker * count
    out = []
    for fill in FILLS:
        f = fill(m)
        for rem in REMAINDERS:
            for scheme in ("http://", "//"):                                # authority position: needs a scheme
                out.append(("authority", scheme + f + rem))
            out.append(("lead0", f + rem))                                  # first character of a scheme-less URL
            for host in ("", "http://h"):
                out.append(("seg1", host + "/" + f + rem))                  # first path segment after a "/"
        for host in ("", "http://h"):
            out.append(("mid", host + "/a/" + f + "/b"))
            out.append(("trail", host + "/a/" + f))
            out.append(("query", host + "/a?q=" + f))
            out.append(("query", host + "/a?q=" + f + "&z=1"))
            out.append(("fragment", host + "/a#" + f))
    return sorted(set(out))


def is_leading_whole(position: str, url: str, marker: str) -> bool:
    """The only place the cause axis can matter: a leading whole run, scheme-less."""
    return position == "lead0" and url.startswith(marker) and re.match(re.escape(marker) + "+(?:/|$|\\?)", url) is not None


GRID = [(pos, url, marker, count) for marker in (X, Z) for count in (1, 2) for pos, url in cells(marker, count)]


def test_the_grid_is_big_enough_to_mean_something_and_never_generates_a_dead_cell():
    assert len(GRID) > 400
    positions = {p for p, *_ in GRID}
    assert positions == {"authority", "lead0", "seg1", "mid", "trail", "query", "fragment"}
    for pos, url, *_ in GRID:
        assert "///" not in url and "\x00\x01" not in url and "\x01\x00" not in url       # no empty segment, no mixed run


def test_the_cause_axis_changes_the_model_only_for_a_leading_whole_run_without_a_scheme():
    """Self-check of the reference model: swapping x00 for x01 changes (key, dropped) only in the cell class the plan names."""
    by_shape = {}
    for marker in (X, Z):
        for count in (1, 2):
            for pos, url in cells(marker, count):
                by_shape.setdefault((pos, count, url.replace(marker, "M")), {})[marker] = reference_key("GET", url)
    differing = 0
    for (pos, count, shape), got in by_shape.items():
        if got[X] != got[Z]:
            differing += 1
            assert pos == "lead0" and shape.startswith("M" * count) and re.match("M+(?:/|$)", shape), (pos, shape, got)
    assert differing > 0


@pytest.mark.parametrize("position,url,marker,count", GRID, ids=lambda v: repr(v) if isinstance(v, str) else None)
def test_grid_normalize_key_parts_equals_the_reference_model(position, url, marker, count):
    nk = need("normalize_key_parts")("GET " + url, "consumer")
    key, dropped, unresolved = reference_key("GET", url)
    assert (nk.key, nk.origin_dropped, nk.unresolved_in_path) == (key, dropped, unresolved), (position, url)
    assert httpkey.normalize_key("GET " + url, "consumer") == key                        # the thin wrapper returns .key
    assert httpkey.url_path_of(url) == need("split_url")(url)[0]                         # and the path wrapper returns split_url's path


def one_call(url_element: dict):
    return [file_("f1", "c.py"), sym("f", "f", fid="f1", path="c.py"), call("c1", "f", "g"), vs("c1", "a0", url_element)]


URL_PACK = {"packFormat": 1, "id": "hand", "rules": [{
    "id": "get", "match": {"callsite": {"callee": "g"}}, "extract": {"url": {"arg": "a0"}},
    "emit": [{"consumes": {"kind": "http", "key": "GET {url}"}}]}]}

# the element a cause stands for, and the marker the plan says it emits
CAUSES = {
    "parameter": ({"unknown": "point-derived"}, X),
    "environment-read": ({"unknown": "runtime-built"}, X),
    "importRef": ({"importRef": {"module": "cfg", "name": "BASE"}}, Z),
    "ref-miss": ({"ref": "m::MISSING"}, X),
}


def template_of(url: str, element_for) -> dict:
    """A bundle ``template`` element for a URL string whose marker characters stand for ``element_for(marker)``."""
    parts: list[dict] = []
    buf = ""
    for ch in url:
        if ch in (X, Z):
            if buf:
                parts.append({"const": buf})
                buf = ""
            parts.append(element_for(ch))
        else:
            buf += ch
    if buf:
        parts.append({"const": buf})
    return {"template": parts}


def e2e_cells():
    """Every grid URL as (position, url-with-markers, cause-name): all four causes at a leading whole run, parameter and importRef elsewhere."""
    out = []
    for pos, url, marker, count in GRID:
        causes = CAUSES if is_leading_whole(pos, url, marker) else {k: v for k, v in CAUSES.items() if k in ("parameter", "importRef")}
        for name, (_, m) in causes.items():
            if m == marker:
                out.append((pos, url, name))
    return out


E2E = e2e_cells()


def test_every_cause_is_exercised_at_the_leading_whole_run():
    assert {n for p, u, n in E2E if p == "lead0" and u[0] in (X, Z)} == set(CAUSES)


@pytest.mark.parametrize("position,url,cause", E2E, ids=lambda v: repr(v) if isinstance(v, str) else None)
def test_grid_status_through_run_packs_is_the_models_status_and_the_models_key(position, url, cause):
    """``_one`` end to end: a consumer slot built from the cause's element kind gets the model's key and its status."""
    element, marker = CAUSES[cause]
    value = template_of(url, lambda m: element)
    (c,) = run_packs([one_call(value)], [URL_PACK]).consumers
    key, _, _ = reference_key("GET", url)
    status, reason = reference_status(url)
    assert (c["key"], c["status"], c["reason"]) == reference_outcome(key, status, reason), (position, url, cause)


# ---------------------------------------------------------------------------------------------------------------------
# the hand-written spot set (expected values written by hand, not derived)
# ---------------------------------------------------------------------------------------------------------------------

#            url                          path of the key  origin_dropped  unresolved_in_path
SPOTS = [
    (X,                                 "/{?}",            False, True),      # the sole unknown is a segment, not an origin
    (X + "/x",                          "/x",              True,  False),     # a leading run of x01 before "/" is the origin
    (X + X + "/x",                      "/x",              True,  False),     # several adjacent pieces are one origin
    (Z + "/x",                          "/{?}/x",          False, True),      # x00 (a path-prefix constant that failed to resolve) keeps its segment
    (Z + Z + "/x",                      "/{?}/x",          False, True),
    (X + "api/x",                       "/{?}/x",          False, True),      # a partial segment
    ("a" + X + "/b",                    "/{?}/b",          False, True),
    (X + ":8000/x",                     "/{?}/x",          False, True),
    (X + "?q=1",                        "/{?}",            False, True),      # not followed by "/": not an origin
    ("http://" + X + "/a",              "/a",              True,  False),     # authority position
    ("//" + Z + "/a",                   "/a",              True,  False),
    ("http://" + X + ":8000/a",         "/a",              True,  False),
    ("http://" + X,                     "/",               True,  False),     # authority runs to the end
    ("http://h/a",                      "/a",              False, False),     # a known host leaves origin_dropped false
    ("http://h#" + X,                   "/",               False, False),     # a "#" ends the authority: a marker after it is the fragment's
    ("//h#" + X,                        "/",               False, False),
    ("http://h?" + X,                   "/",               False, False),     # so does a "?" (the marker is the query's)
    ("http://h#" + X + "/a",            "/",               False, False),     # the path is empty: "/a" after the "#" is fragment text
    ("/x/" + X + "/y",                  "/x/{?}/y",        False, True),      # mid-path
    ("/x/" + X,                         "/x/{?}",          False, True),      # trailing
    ("/" + X + "/x",                    "/{?}/x",          False, True),      # first segment after a slash is not "leading no-scheme"
    ("http://h/" + X + "/x",            "/{?}/x",          False, True),
    ("/items?q=" + X,                   "/items",          False, False),     # query-only unknown: not in the key
    ("/a#" + X,                         "/a",              False, False),     # fragment-only unknown
    (X + "/items?q=" + X,               "/items",          True,  False),     # origin drop and a query unknown: only the origin counts
]


@pytest.mark.parametrize("url,path,dropped,unresolved", SPOTS, ids=[repr(s[0]) for s in SPOTS])
def test_spot_rows_normalize_key_parts(url, path, dropped, unresolved):
    nk = need("normalize_key_parts")("GET " + url, "consumer")
    assert (nk.key, nk.origin_dropped, nk.unresolved_in_path) == ("GET " + path, dropped, unresolved)
    assert reference_key("GET", url) == ("GET " + path, dropped, unresolved)            # the hand table and the model agree


@pytest.mark.parametrize("url,path,dropped,unresolved", SPOTS, ids=[repr(s[0]) for s in SPOTS])
def test_spot_rows_status_through_run_packs(url, path, dropped, unresolved):
    value = template_of(url, lambda m: {"unknown": "point-derived"} if m == X else {"importRef": {"module": "m", "name": "N"}})
    (c,) = run_packs([one_call(value)], [URL_PACK]).consumers
    want = ("possible", "partial-key") if (dropped or unresolved) else ("confirmed", "key-match")
    assert (c["key"], c["status"], c["reason"]) == reference_outcome("GET " + path, *want)


@pytest.mark.parametrize("cause,url,key,status", [
    ("parameter", X + "/x", "GET /x", "possible"),                   # a parameter origin: dropped, never confirmed
    ("environment-read", X + "/x", "GET /x", "possible"),
    ("importRef", Z + "/x", "GET /{?}/x", "possible"),               # an unresolved name keeps its segment
    ("ref-miss", X + "/x", "GET /x", "possible"),
])
def test_the_cause_axis_rows_the_four_element_kinds_at_a_leading_whole_segment(cause, url, key, status):
    element, _ = CAUSES[cause]
    (c,) = run_packs([one_call(template_of(url, lambda m: element))], [URL_PACK]).consumers
    assert (c["key"], c["status"], c["reason"]) == (key, status, "partial-key")


def test_a_wholly_known_url_stays_confirmed_so_the_downgrade_is_not_blanket():
    (c,) = run_packs([one_call({"const": "http://svc:8000/items"})], [URL_PACK]).consumers
    assert (c["key"], c["status"], c["reason"]) == ("GET /items", "confirmed", "key-match")


# ---------------------------------------------------------------------------------------------------------------------
# NormalizedKey / split_url shape and the unchanged wrappers
# ---------------------------------------------------------------------------------------------------------------------

def test_normalized_key_is_a_named_tuple_with_the_three_planned_fields():
    nk = need("normalize_key_parts")("GET http://h/a", "consumer")
    assert isinstance(nk, tuple) and nk._fields == ("key", "origin_dropped", "unresolved_in_path")
    assert tuple(nk) == ("GET /a", False, False)
    assert need("NormalizedKey")._fields == ("key", "origin_dropped", "unresolved_in_path")


def test_split_url_returns_the_path_and_whether_the_origin_was_dropped():
    split = need("split_url")
    assert split("http://h/a?q=1#f") == ("/a", False)
    assert split("http://" + X + "/a") == ("/a", True)
    assert split(X + "/a") == ("/a", True)
    assert split("/a?q=" + X) == ("/a", False)


def test_the_provider_form_is_not_touched_by_the_origin_rules():
    nk = need("normalize_key_parts")("GET /api/items/{id}", "provider")
    assert (nk.key, nk.origin_dropped, nk.unresolved_in_path) == ("GET /api/items/{}", False, False)
    assert httpkey.normalize_key("GET /api/items/{id}", "provider") == nk.key


def test_a_bad_form_is_still_refused():
    with pytest.raises(ValueError, match="form must be"):
        need("normalize_key_parts")("GET /a", "elsewhere")


def test_the_thin_wrappers_keep_todays_behaviour_for_marker_free_input():
    assert httpkey.normalize_key("get http://h:8000/a/b?x=1", "consumer") == "GET /a/b"
    assert httpkey.url_path_of("https://h/a/b?x=1#f") == "/a/b"
    assert httpkey.url_path_of("/a?x=1") == "/a"
    assert httpkey.normalize_key("/a/b", "provider") == "* /a/b"
