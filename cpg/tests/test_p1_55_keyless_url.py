"""P1-55 (red first): a consumer URL whose path names no literal segment is keyless, and the `fetch` pack's own site.

Plan: `docs/plans/connected-code-graph-p1-linking.md` P1-55. Cause: `docs/test-reports/connected-code-graph-p1-report.md` D3, D4.
The marker characters are written out on purpose (`\\x01` is a value not statically known), never imported from the code under test.
Every key in the reference table is *normalized by the code* from a raw URL, so the table covers the whole chain `split_url` ->
`normalize_path` -> predicate, not the predicate on a hand-written key.
"""
from __future__ import annotations

import copy

import pytest

from packrt_helpers import call, data, file_, pack, param, sym, vs
from ccg.packrt import httpkey, run_packs
from test_packrt_run import fetch_chain

FETCH = pack("fetch")
U = "\x01"

#: raw consumer string (verb optional) -> is the normalized key keyless?  The expectation is written from the rule alone:
#: at least one `{?}` segment and no literal segment; the root `/` stays a key.
TABLE = [
    (f"{U}/{U}", True),                    # f"{get_base()}/{ROUTES[name]}" (AC6-URL-1)
    (f"{U}{U}", True),                     # fetch(base + path) (AC6-URL-2)
    (f"/{U}", True),
    (f"/{U}/{U}", True),                   # two unknown segments: still nothing literal
    (f"/{U}a", True),                      # a segment holding a marker is `{?}` whole
    (f"{U}/{U}?x=1", True),
    (f"GET {U}/{U}", True),
    (f"* {U}/{U}", True),                  # any-method verb
    (f"POST /{U}", True),
    (f"http://{U}/{U}", True),             # unknown origin, unknown path
    ("/", False),                          # the root alone is a key
    (f"{U}/", False),                      # unknown origin, root path
    ("?q=1", False),                       # query only: the path is the root
    (f"{U}?q={U}", True),                  # an unknown prefix before the query is an unknown path (`base + "?q="`)
    (f"/a/{U}", False),                    # a literal segment: partial-key stays a key
    (f"/{U}/a", False),
    (f"/api/{U}/{U}", False),
    ("/api/x", False),
    (f"https://{U}/a", False),
]


@pytest.mark.parametrize("raw,keyless", TABLE)
def test_the_key_shape_table(raw, keyless):
    key = httpkey.normalize_key(raw, "consumer")
    assert httpkey.names_no_literal_segment(key) is keyless, key


def test_a_known_param_segment_is_not_unknown():
    """`/{}` (a route template's own parameter, from `{id}`) holds no `{?}`: the predicate needs an unknown segment, not just no literal."""
    key = httpkey.normalize_key("/{id}", "consumer")
    assert httpkey.names_no_literal_segment(key) is False, key


# ---------------------------------------------------------------------------------------------------------------- `_one` (run.py)

UNK = {"unknown": "point-derived"}


def leaf_call(*values, verb=None):
    """One function holding a `fetch(<values>)` seed call whose URL is built from `values` and is itself the leaf (no caller)."""
    recs = [file_("f1", "web/dyn.ts", lang="javascript"), sym("go", "go", fid="f1", path="web/dyn.ts"),
            call("fe", "go", "fetch", line=5, resolution="external", external="lib/fetch.js:<module>.fetch"), vs("fe", "a0", *values)]
    return recs


def view(c):
    return (c["key"], c["status"], c["reason"])


def consumers(recs):
    return run_packs([recs], [FETCH]).consumers


@pytest.mark.parametrize("url", [
    [{"template": [UNK, {"const": "/"}, UNK]}],                      # f"{get_base()}/{ROUTES[name]}"
    [{"concat": [UNK, UNK]}],                                        # base + path
])
def test_a_url_with_no_literal_segment_is_one_keyless_possible_runtime_built_fact(url):
    assert [view(c) for c in consumers(leaf_call(*url))] == [(None, "possible", "runtime-built")]


@pytest.mark.parametrize("url,expected", [
    ([{"concat": [{"const": "/a/"}, UNK]}], ("GET /a/{?}", "possible", "partial-key")),     # a literal segment keeps the key
    ([{"concat": [UNK, {"const": "/health"}]}], ("GET /health", "possible", "partial-key")),
    ([{"const": "/"}], ("GET /", "confirmed", "key-match")),                               # the root alone is a key (a confirmed literal)
])
def test_a_url_with_a_literal_segment_or_the_root_stays_a_key(url, expected):
    (c,) = consumers(leaf_call(*url))
    assert (c["key"], c["status"]) == expected[:2] and c["key"] == expected[0]


def test_an_unknown_origin_with_the_root_path_stays_a_key():
    (c,) = consumers(leaf_call({"concat": [UNK, {"const": "/"}]}))
    assert view(c)[0] == "GET /" and c["status"] == "possible"


def test_an_unknown_verb_with_a_literal_path_is_not_keyless():
    recs = leaf_call({"const": "/api/x"})
    recs.append(vs("fe", "a1.method", {"unknown": "point-derived"}))
    (c,) = consumers(recs)
    assert c["key"] == "* /api/x"


# ---------------------------------------------------------------------------------------------------------------- the `ownSite` pair

def wrapper_never_called(*values):
    """`go(path)` does `fetch(<values>)` (`base + path`: one marker per piece); no call edge reaches `go`."""
    return [
        file_("f1", "web/dyn.ts", lang="javascript"),
        sym("go", "go", fid="f1", path="web/dyn.ts", start=3), param("p0", "go", 0, "path", line=3),
        call("fe", "go", "fetch", line=5, resolution="external", external="lib/fetch.js:<module>.fetch"),
        data("fe", "p0", "a0", "p0"), vs("fe", "a0", {"concat": [UNK, UNK]}),
    ]


def without_own_site():
    off = copy.deepcopy(FETCH)
    off["liftings"][0].pop("ownSite")
    return off


def test_the_fetch_pack_declares_own_site():
    assert FETCH["liftings"][0]["ownSite"] is True


def test_with_own_site_a_wrapper_no_call_edge_reaches_is_a_keyless_consumer_at_its_own_site():
    out = consumers(wrapper_never_called())
    assert [(c["anchorId"], c["siteId"], view(c)) for c in out] == [("go", "fe", (None, "possible", "runtime-built"))]


def test_without_own_site_the_same_wrapper_is_no_consumer():
    """The pair that makes `ownSite` load-bearing: the identical records under the pack minus the one setting."""
    assert run_packs([wrapper_never_called()], [without_own_site()]).consumers == []


def test_a_wrapper_with_a_caller_is_still_the_callers_consumer_not_its_own():
    out = consumers(fetch_chain())
    assert [(c["anchorId"], c["key"]) for c in out] == [("caller", "GET /api/x")]
