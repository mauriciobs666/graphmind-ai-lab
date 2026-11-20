"""P1-49b: the three optional pack-format words of `liftings` (`seeds[].scope`, `ownSite`, `urlLike`) and `httpkey.is_url_like`.

Plan: `docs/plans/connected-code-graph-p1-linking.md` P1-49 "Format additions" (all optional, default off, so the `fetch` pack
is unchanged). The tests run the real `fetch` pack with one option switched on over `test_packrt_run.fetch_chain()`-shaped
records: `get(path)` forwards its parameter to the seed `fe`, `caller` passes a literal to `get`.
"""
from __future__ import annotations

import copy

import pytest

from packrt_helpers import call, data, file_, imp, param, sym, vs
from ccg.packrt import httpkey, run_packs
from test_packrt_run import FETCH, fetch_chain

X = "\x01"
Z = "\x00"
UNK = {"unknown": "point-derived"}


def fetch_with(*, own=None, url_like=None, scope=None):
    p = copy.deepcopy(FETCH)
    lifting = p["liftings"][0]
    if own is not None:
        lifting["ownSite"] = own
    if url_like is not None:
        lifting["urlLike"] = url_like
    if scope is not None:
        lifting["seeds"][0]["scope"] = scope
    return p


def run(recs, p):
    return run_packs([recs], [p])


def view(c):
    return (c["siteId"], c["key"], c["status"], c["reason"])


def without_caller(*seed_values):
    """The wrapper ``get`` and its seed ``fe`` with no call edge reaching ``get`` (``pool.submit(get, ...)``, an untyped receiver)."""
    recs = [r for r in fetch_chain() if r.get("id") not in ("caller", "c1", "vs-c1-a0")]
    return recs + ([vs("fe", "a0", *seed_values)] if seed_values else [])


def with_caller(value):
    recs = fetch_chain()
    recs[-1] = vs("c1", "a0", value)
    return recs


# ---------------------------------------------------------------------------------------------------------------------
# ownSite
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("own,expected", [(None, []), (False, []), (True, [("fe", "GET /own/x", "confirmed", "lifted-role")])],
                         ids=["absent", "off", "on"])
def test_own_site_makes_a_wrapper_that_no_call_edge_reaches_a_consumer_at_its_own_site(own, expected):
    out = run(without_caller("/own/x"), fetch_with(own=own))
    assert [view(c) for c in out.consumers] == expected


def test_own_site_leaves_a_wrapper_with_a_lifted_caller_a_non_leaf_so_only_the_caller_is_the_consumer():
    recs = with_caller("/api/x") + [vs("fe", "a0", "/own/x")]
    for own in (False, True):
        out = run(recs, fetch_with(own=own))
        assert [c["siteId"] for c in out.consumers] == ["c1"], own
        assert {i["siteId"]: i["leaf"] for i in out.lifted} == {"fe": False, "c1": True}, own


def test_own_site_makes_an_uncalled_intermediate_wrapper_the_leaf_and_the_seed_stays_a_non_leaf():
    """`outer(q)` calls `get(q)` and nothing calls `outer`: its call site `oc` has no slot beyond it, so with ownSite it is the leaf."""
    recs = fetch_chain()[:-2] + [
        sym("outer", "outer", fid="f1", path="web/client.ts", start=50), param("q0", "outer", 0, "q", line=51),
        call("oc", "outer", "get", line=52, resolution="resolved", targets=("get",)), data("oc", "q0", "a0", "p0"),
        vs("oc", "a0", "/mid/x")]
    off = run(recs, fetch_with(own=False))
    on = run(recs, fetch_with(own=True))
    assert off.consumers == []                                           # `oc` has a parameter below it: not a leaf today
    assert [view(c) for c in on.consumers] == [("oc", "GET /mid/x", "confirmed", "lifted-role")]
    assert {i["siteId"]: i["leaf"] for i in on.lifted} == {"fe": False, "oc": True}


# ---------------------------------------------------------------------------------------------------------------------
# urlLike
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("alt,url_like", [
    ("/a", True), ("//h/a", True), ("http://h/a", True), ("https://h", True), ("ws://h/x", True),
    (X + "/a", True), (X + X + "/a", True), (Z + "/a", True), (Z + X + "/a", True),
    ("", False), ("a", False), ("timeout", False), ("items/x", False), (X, False), (Z, False), (X + X, False),
    (X + "a/b", False), ("a/b", False), ("h://x", True), ("1http://x", False), ("a:b", False), ("mailto:x", False),
])
def test_is_url_like(alt, url_like):
    assert httpkey.is_url_like(alt) is url_like


def test_url_like_drops_a_leaf_with_no_url_like_alternative_and_keeps_the_intermediate_slot_untouched():
    out = run(with_caller("timeout"), fetch_with(url_like=True))
    assert out.consumers == []
    assert [i["siteId"] for i in out.lifted] == ["fe"]                  # the leaf `c1` is gone from `lifted`: no keyless fact either
    assert out.lifted[0]["leaf"] is False


def test_url_like_off_keeps_the_same_leaf_as_a_consumer():
    out = run(with_caller("timeout"), fetch_with(url_like=False))
    assert [view(c) for c in out.consumers] == [("c1", "GET /timeout", "confirmed", "lifted-role")]


def test_url_like_keeps_only_the_url_like_alternatives_of_a_leaf():
    recs = fetch_chain()
    recs[-1] = vs("c1", "a0", "timeout", "/x")
    out = run(recs, fetch_with(url_like=True))
    (leaf,) = [i for i in out.lifted if i["siteId"] == "c1"]
    assert leaf["alternatives"] == ["/x"] and leaf["partialAlternatives"] is False
    assert [view(c) for c in out.consumers] == [("c1", "GET /x", "confirmed", "lifted-role")]


def test_url_like_decides_partial_alternatives_before_the_filter_so_a_path_beside_an_unknown_stays_partial():
    recs = fetch_chain()
    recs[-1] = vs("c1", "a0", "/a", X)
    out = run(recs, fetch_with(url_like=True))
    (leaf,) = [i for i in out.lifted if i["siteId"] == "c1"]
    assert leaf["partialAlternatives"] is True and sorted(leaf["alternatives"]) == sorted(["/a", X])
    assert [view(c) for c in out.consumers] == [("c1", "GET /a", "possible", "partial-key")]


def test_url_like_keeps_a_wholly_unresolved_alternative_beside_a_url_like_one_and_drops_a_leaf_of_only_unknowns():
    recs = fetch_chain()
    recs[-1] = vs("c1", "a0", X)
    out = run(recs, fetch_with(url_like=True))
    assert out.consumers == [] and [i["siteId"] for i in out.lifted] == ["fe"]
    out_off = run(recs, fetch_with(url_like=False))
    assert [i["siteId"] for i in out_off.lifted] == ["c1", "fe"] and out_off.consumers == []


def test_url_like_applies_to_a_leaf_made_by_own_site_too():
    own = fetch_with(own=True, url_like=True)
    assert run(without_caller("timeout"), own).consumers == []
    assert [c["siteId"] for c in run(without_caller("/own/x"), own).consumers] == ["fe"]


# ---------------------------------------------------------------------------------------------------------------------
# seeds[].scope
# ---------------------------------------------------------------------------------------------------------------------

IMPORTS_FETCH = {"imports": [{"module": {"prefix": "node-fetch"}}]}


def test_a_seed_scope_gates_on_the_seed_calls_own_file():
    plain = fetch_chain()
    with_import = plain + [imp("f1", "node-fetch")]
    p = fetch_with(scope=IMPORTS_FETCH)
    assert run(plain, p).consumers == []
    assert [c["siteId"] for c in run(with_import, p).consumers] == ["c1"]


def test_a_seed_scope_is_the_seeds_file_not_the_callers_file():
    """The wrapper `get` and its seed live in `f1`; the caller `c1` is in `f2`, which imports the library. The seed's file decides."""
    recs = fetch_chain() + [file_("f2", "web/page.ts"), imp("f2", "node-fetch")]
    recs = [dict(r, symbolId="caller2") if r.get("id") == "c1" else r for r in recs] + [
        sym("caller2", "caller2", fid="f2", path="web/page.ts", start=5)]
    assert run(recs, fetch_with(scope=IMPORTS_FETCH)).consumers == []
    recs_wrapper_imports = recs + [imp("f1", "node-fetch")]
    assert [c["siteId"] for c in run(recs_wrapper_imports, fetch_with(scope=IMPORTS_FETCH)).consumers] == ["c1"]


def test_a_rule_scope_still_means_the_leaf_sites_file_and_is_independent_of_the_seed_scope():
    p = fetch_with()
    p["rules"][0]["scope"] = IMPORTS_FETCH
    recs = fetch_chain()
    assert run(recs, p).consumers == []
    assert [c["siteId"] for c in run(recs + [imp("f1", "node-fetch")], p).consumers] == ["c1"]


# ---------------------------------------------------------------------------------------------------------------------
# off is the old behaviour
# ---------------------------------------------------------------------------------------------------------------------

def test_with_all_three_words_off_or_absent_the_fetch_pack_gives_the_same_facts():
    for recs in (fetch_chain(), fetch_chain(init=True), without_caller("/own/x"), with_caller("timeout")):
        a = run(recs, FETCH)
        b = run(recs, fetch_with(own=False, url_like=False))
        assert a == b


# ---------------------------------------------------------------------------------------------------------------------
# the schema vocabulary (optional words, packFormat stays 1)
# ---------------------------------------------------------------------------------------------------------------------

def _errors(p):
    from ccg.rulepack import pack_errors
    return pack_errors(p, dirname="fetch")


def test_the_schema_accepts_the_three_optional_words_and_packformat_stays_1():
    p = fetch_with(own=True, url_like=True, scope=IMPORTS_FETCH)
    assert _errors(p) == [] and p["packFormat"] == 1


@pytest.mark.parametrize("edit", [
    lambda p: p["liftings"][0].update(ownSite="yes"),
    lambda p: p["liftings"][0].update(urlLike=1),
    lambda p: p["liftings"][0]["seeds"][0].update(scope={"imports": [{"module": {"prefix": "x"}}], "bogus": 1}),
    lambda p: p["liftings"][0]["seeds"][0].update(scope={"role": ["nowhere"]}),
    lambda p: p["liftings"][0].update(ownsite=True),
], ids=["ownSite-type", "urlLike-type", "scope-unknown-key", "scope-role-enum", "misspelt-word"])
def test_the_schema_rejects_a_malformed_optional_word(edit):
    p = fetch_with()
    edit(p)
    assert _errors(p) != []


# ---------------------------------------------------------------------------------------------------------------------
# rule 4: transparency is `every` alternative wholly unresolved, not the first
# ---------------------------------------------------------------------------------------------------------------------

def test_a_seed_slot_holding_a_parameter_beside_a_known_alternative_is_opaque():
    """`[\\x01, "/known"]` sorts the marker first: a test on the first alternative alone would call the slot transparent."""
    recs = fetch_chain() + [vs("fe", "a0", UNK, "/known")]
    (c,) = run(recs, FETCH).consumers
    assert (c["status"], c["reason"]) == ("possible", "partial-key")
    assert c["evidence"]["opaqueSlot"] == {"siteId": "fe", "arg": "a0"}
