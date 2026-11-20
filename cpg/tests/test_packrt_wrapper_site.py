"""P1-51b: ``liftings[].wrapperSite`` -- a wrapper whose template is informative is the consumer, not its callers.

Plan: ``docs/plans/connected-code-graph-p1-linking.md`` P1-51 part B ("51b"); gate ``docs/reviews/connected-code-graph-p1-51-53.md``
(B1, M5).

The test holds a REFERENCE MODEL that shares no code with ``ccg/packrt/lift.py`` and does not call ``httpkey``: the shipped
fixtures hold no informative wrapper, so a check over them is green before the engine edit; the synthetic chains below are the
positive control. Every alternative shape of ``SHAPES`` carries, written by hand and not computed, whether it is URL-like,
whether it is informative (a path segment other than ``{?}``), its consumer key path and whether the key is ``possible`` by
itself (an unresolved piece in the path, or an unknown origin; a query-only unknown stays ``confirmed``). The model composes
them over chains of 1-3 slots (leaf first, the seed last) and is compared, cell by cell, with ``run_packs``.

Axes (plan, tests): slot kind (leaf, intermediate, seed), alternative shape (no origin, ``\\x01`` origin, known origin,
``\\x00`` prefix, one partial segment, query-only, scheme), alternative count (one; mixed with a bare ``\\x01``), informative slots
on the chain (0; 1; 2 with a non-informative one nearer), leaves per ``X`` (1, 2), verb source at ``X`` (literal, parameter),
``urlLike`` on and off, ``wrapperSite`` on and off.
"""
from __future__ import annotations

import copy
import functools
import itertools

import pytest

from packrt_helpers import call, data, file_, imp, pack, param, sym, vs
from ccg.linker import link
from ccg.packrt import run_packs
from ccg.rulepack import pack_errors

PID = "python-http-client"
X = "\x01"
Z = "\x00"
REQUEST = "urllib.py:<module>.<member>(request).Request.__init__"

# name -> (alternative text, url_like, informative, consumer key path, key is possible by itself)
SHAPES = {
    "lit": ("/items", True, True, "/items", False),
    "tmpl": ("/documents/" + X, True, True, "/documents/{?}", True),
    "origin": (X + "/shop/api/messages?a=" + X, True, True, "/shop/api/messages", True),
    "known": ("http://h/api/x", True, True, "/api/x", False),
    "scheme_unk": ("http://" + X + "/api/x", True, True, "/api/x", True),
    "zprefix": (Z + "/api/x", True, True, "/{?}/api/x", True),
    "partseg": ("/api" + X, True, False, "/{?}", True),
    "partseg2": ("/a/b" + X, True, True, "/a/{?}", True),
    "query": ("/x?q=" + X, True, True, "/x", False),
    "queryonly": (X + "?q=1", False, False, "/{?}", True),
    "twomark": (X + X, False, False, "/{?}", True),
    "plain": ("plain", False, False, "/plain", False),
    "bare": (X, False, False, None, None),
    # P1-51b ruling (plan, "informative"): after the authority and with the query and fragment cut, the remainder is the root `/`
    # or has a segment other than `{?}`; a bare origin names no resource. Expectations are written from that text.
    "root": ("/", True, True, "/", False),
    "root_unk": (X + "/", True, True, "/", True),
    "rootorigin": ("http://h/", True, True, "/", False),
    "origin_only": ("http://localhost:1234", True, False, "/", False),
    "origin_unk": ("https://" + X, True, False, "/", True),
    "slash_mark": (X + "/" + X, True, False, "/{?}", True),
    "origin_query": ("http://h?q=1", True, False, "/", False),                  # the query is cut before the remainder is read
    "root_query": ("/?q=1", True, True, "/", False),
}
# a slot is a tuple of shape names (its alternatives)
SLOTS = [("lit",), ("tmpl",), ("origin",), ("known",), ("scheme_unk",), ("zprefix",), ("partseg",), ("partseg2",), ("query",),
         ("queryonly",), ("twomark",), ("plain",), ("bare",),
         ("lit", "bare"), ("tmpl", "bare"), ("partseg", "bare"), ("lit", "plain"),
         ("root",), ("root_unk",), ("rootorigin",), ("origin_only",), ("origin_unk",), ("slash_mark",), ("origin_query",), ("root_query",),
         ("origin_only", "bare"), ("root", "bare"), ("origin_only", "lit")]


# ---------------------------------------------------------------------------------------------------------------------
# the reference model
# ---------------------------------------------------------------------------------------------------------------------

def informative(slot) -> bool:
    return any(SHAPES[s][2] for s in slot)


def model(chain, *, wrapper_site, url_like, verbs=None, fallback="GET"):
    """``(consumers, lifted)`` the rule says, for ``chain`` (leaf first, the seed last; each slot a tuple of shape names).

    ``consumers``: ``{(chain index, key, status)}``; ``lifted``: ``{(chain index, leaf?, resolved?)}`` -- the items a leaf
    that is dropped (URL filter, or beyond an informative slot) does not have. ``verbs`` maps a chain index to the literal verb
    that slot passes (``None``: the verb is the companion's ``bareDefault``, ``fallback``)."""
    n = len(chain)
    x = None
    if wrapper_site:
        x = next((i for i in range(1, n) if informative(chain[i])), None)    # the scan starts after the leaf

    def verb_from(start):
        if verbs is None:
            return fallback
        return next((verbs[i] for i in range(start, n) if i in verbs), "*")

    lifted, consumers = set(), set()
    for i, slot in enumerate(chain):
        is_leaf = i == 0
        if is_leaf and x is not None:
            continue                                                      # dropped: X is the consumer
        if i == x:
            is_leaf = True
        shown = [s for s in slot if not url_like or not is_leaf or SHAPES[s][1] or s == "bare"]
        if is_leaf and url_like and not any(SHAPES[s][1] for s in slot):
            continue                                                      # no URL-like alternative: dropped
        partial = "bare" in slot
        resolved = any(s != "bare" for s in shown)
        lifted.add((i, is_leaf, resolved))
        if not (is_leaf and resolved):
            continue
        opaque = any(any(s != "bare" for s in chain[j]) for j in range(i + 1, n))
        for s in shown:
            if s == "bare":
                continue
            possible = partial or opaque or SHAPES[s][4]
            consumers.add((i, f"{verb_from(i)} {SHAPES[s][3]}", "possible" if possible else "confirmed"))
    return consumers, lifted


# ---------------------------------------------------------------------------------------------------------------------
# records for a chain: seed in F{n-1}, F{k+1} is called by slot k, the leaf is slot 0
# ---------------------------------------------------------------------------------------------------------------------

def records(chain, *, lifting="py-get", verbs=None, leaves=1):
    n = len(chain)
    recs = [file_("f1", "svc/client.py"), imp("f1", "requests"), imp("f1", "urllib.request")]
    for k in range(n):
        recs.append(sym(f"F{k}", f"f{k}", fid="f1", path="svc/client.py", start=10 * (k + 1)))
    sites = [(f"s{k}", f"F{k}", k) for k in range(n)]
    if leaves == 2:
        recs.append(sym("FB", "fb", fid="f1", path="svc/client.py", start=900))
        sites.append(("sB", "FB", 0))
    for sid, owner, k in sites:
        if k == n - 1:
            recs.append(call(sid, owner, "get", line=10 * (k + 1) + 1) if lifting == "py-get"
                        else call(sid, owner, "Request", line=10 * (k + 1) + 1, resolution="external", external=REQUEST))
        else:
            recs.append(call(sid, owner, f"f{k + 1}", line=10 * (k + 1) + 1 + (500 if sid == "sB" else 0),
                             resolution="resolved", targets=(f"F{k + 1}",)))
        recs.append(vs(sid, "a0", *[SHAPES[s][0] for s in chain[k]]))
        if k >= 1 and sid != "sB":
            recs += [param(f"P{k}", f"F{k}", 0, "u"), data(sid, f"P{k}", "a0", "p0")]
        if verbs is not None:          # the seed's `method=` keyword; a wrapper passes its second parameter, a caller's `a1`
            vpath = "kw.method" if k == n - 1 else "a1"
            if k in verbs:
                recs.append(vs(sid, vpath, verbs[k]))
            if k >= 1 and sid != "sB":
                recs += [param(f"Q{k}", f"F{k}", 1, "m"), data(sid, f"Q{k}", vpath, "p1")]
    return recs


_SHIPPED = pack(PID)


@functools.cache
def the_pack(*, wrapper_site, url_like=None):
    """The shipped pack with ``wrapperSite`` on every lifting (or off) and ``py-get``'s ``urlLike`` as asked; read-only (cached)."""
    p = copy.deepcopy(_SHIPPED)
    for lf in p["liftings"]:
        if wrapper_site:
            lf["wrapperSite"] = True
        else:
            lf.pop("wrapperSite", None)
        if url_like is not None and lf["id"] == "py-get":
            lf["urlLike"] = url_like
    return p


def observed(recs, p):
    out = run_packs([recs], [p])
    idx = lambda site: 0 if site in ("s0", "sB") else int(site[1:])          # noqa: E731
    return ({(idx(c["siteId"]), c["key"], c["status"]) for c in out.consumers},
            {(idx(i["siteId"]), i["leaf"], i["resolved"]) for i in out.lifted}, out)


def key_for(model_consumers):
    return {(i, k, s) for i, k, s in model_consumers}


# ---------------------------------------------------------------------------------------------------------------------
# the format word is in the schema and the shipped pack carries it on the seven liftings
# ---------------------------------------------------------------------------------------------------------------------

def test_the_shipped_pack_sets_wrapper_site_on_the_seven_liftings_and_validates():
    p = pack(PID)
    assert pack_errors(p, dirname=PID) == []
    assert {lf["id"]: lf.get("wrapperSite") for lf in p["liftings"]} == {
        "py-get": True, "py-post": True, "py-put": True, "py-patch": True, "py-delete": True, "py-request": True, "py-urlopen": True}


def test_the_schema_accepts_wrapper_site_only_as_a_boolean():
    p = copy.deepcopy(pack(PID))
    p["liftings"][0]["wrapperSite"] = "yes"
    assert pack_errors(p, dirname=PID) != []


def test_py_request_does_not_get_url_like_flipped_on():
    """REJECTED (gate B1): `urlLike` on `py-request` drops the bare-`\\x01` leaf before the scan, so AC-5 is 6 of 8, not 8."""
    assert "urlLike" not in next(lf for lf in pack(PID)["liftings"] if lf["id"] == "py-request")


# ---------------------------------------------------------------------------------------------------------------------
# cell by cell against the reference model
# ---------------------------------------------------------------------------------------------------------------------

def cells():
    for leaf, seed in itertools.product(SLOTS, SLOTS):
        yield (leaf, seed)
    for leaf, mid, seed in itertools.product(SLOTS, [("bare",), ("twomark",), ("tmpl",), ("lit",), ("partseg",), ("lit", "bare")],
                                             [("bare",), ("lit",), ("partseg",)]):
        yield (leaf, mid, seed)
    for slot in SLOTS:
        yield (slot,)


@pytest.mark.parametrize("wrapper_site", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("url_like", [False, True], ids=["urlLike-off", "urlLike-on"])
def test_every_cell_of_the_reference_model(wrapper_site, url_like):
    wrong = []
    for chain in cells():
        want_c, want_l = model(chain, wrapper_site=wrapper_site, url_like=url_like)
        got_c, got_l, _ = observed(records(chain), the_pack(wrapper_site=wrapper_site, url_like=url_like))
        if (got_c, got_l) != (want_c, want_l):
            wrong.append((chain, sorted(want_c ^ got_c), sorted(want_l ^ got_l)))
    assert wrong == []


def test_the_positive_control_the_model_differs_from_the_rule_off_on_informative_wrappers():
    """Guards the guard: some cells must change with `wrapperSite`, else the grid above proves nothing about the rule."""
    moved = [c for c in cells() if model(c, wrapper_site=True, url_like=True) != model(c, wrapper_site=False, url_like=True)]
    assert len(moved) > 100


@pytest.mark.parametrize("wrapper_site", [False, True], ids=["off", "on"])
def test_two_leaves_of_one_wrapper_yield_one_consumer_at_the_wrapper_when_it_is_informative(wrapper_site):
    wrong = []
    for x_slot, leaf_slot in itertools.product(SLOTS, (("bare",), ("lit",), ("twomark",))):
        chain = (leaf_slot, x_slot, ("bare",))
        want = model(chain, wrapper_site=wrapper_site, url_like=True)
        got_c, got_l, out = observed(records(chain, leaves=2), the_pack(wrapper_site=wrapper_site, url_like=True))
        if (got_c, got_l) != want:
            wrong.append((chain, sorted(want[0] ^ got_c), sorted(want[1] ^ got_l)))
        if wrapper_site and informative(x_slot):
            assert [i["siteId"] for i in out.lifted if i["leaf"]] == ["s1"]               # X once; s0 and sB are dropped
            assert {c["siteId"] for c in out.consumers} == {"s1"}
    assert wrong == []


@pytest.mark.parametrize("verbs,start,want", [
    ({1: "DELETE"}, 1, "DELETE"),          # the verb is a literal at X
    ({2: "POST"}, 1, "POST"),              # a literal at the seed, below X
    ({0: "DELETE"}, 1, "*"),               # the literal is beyond X (the leaf's): X's own chain has none: a parameter passes it
    ({}, 1, "*"),                          # a method parameter into X that nothing fills
])
def test_the_verb_of_the_wrapper_site_comes_from_its_own_chain(verbs, start, want):
    chain = (("bare",), ("tmpl",), ("bare",))
    want_c, _ = model(chain, wrapper_site=True, url_like=False, verbs=verbs)
    assert {v.split(" ")[0] for _, v, _ in want_c} == {want}
    got_c, _, _ = observed(records(chain, lifting="py-request", verbs=verbs), the_pack(wrapper_site=True))
    assert got_c == want_c


def test_a_request_without_a_method_is_a_get_and_the_verb_grid_agrees_with_the_model():
    wrong = []
    for chain in itertools.product(SLOTS, SLOTS, [("bare",), ("lit",)]):
        for verbs in (None, {0: "PUT"}, {1: "DELETE"}, {2: "POST"}):
            for ws in (False, True):
                want_c, want_l = model(chain, wrapper_site=ws, url_like=False, verbs=verbs)
                got_c, got_l, _ = observed(records(chain, lifting="py-request", verbs=verbs), the_pack(wrapper_site=ws))
                if (got_c, got_l) != (want_c, want_l):
                    wrong.append((chain, verbs, ws))
    assert wrong == []


# ---------------------------------------------------------------------------------------------------------------------
# the linker emits no keyless fact for a dropped leaf; pinned limits
# ---------------------------------------------------------------------------------------------------------------------

def linked_keys(recs, p):
    facts = run_packs([recs], [p])
    bundle = [{"type": "header", "unitId": "U", "unit": {"repoId": "r", "rootPath": ".", "lang": "python"}}] + recs
    got = link(facts, [bundle], packs=[p])
    return facts, got


def test_a_dropped_bare_leaf_leaves_no_keyless_possible_fact_through_link():
    chain = (("bare",), ("tmpl",), ("bare",))
    off_facts, off = linked_keys(records(chain), the_pack(wrapper_site=False, url_like=False))
    on_facts, on = linked_keys(records(chain), the_pack(wrapper_site=True, url_like=False))
    # without the rule the unresolved leaf is a keyless fact for the linker; with it the leaf is gone and X carries the key
    assert [i["siteId"] for i in off_facts.lifted if i["leaf"] and not i["resolved"]] == ["s0"]
    assert [i["siteId"] for i in on_facts.lifted if i["leaf"] and not i["resolved"]] == []
    from ccg.linker.link import _unknown_lifted
    assert [c["siteId"] for c in _unknown_lifted(off_facts, [the_pack(wrapper_site=False, url_like=False)])] == ["s0"]
    assert _unknown_lifted(on_facts, [the_pack(wrapper_site=True, url_like=False)]) == []
    assert [c["key"] for c in on_facts.consumers] == ["GET /documents/{?}"]


def test_the_scan_starts_after_the_leaf_so_an_informative_leaf_stays_the_consumer_when_nothing_below_is_informative():
    chain = (("lit",), ("bare",), ("bare",))
    got_c, _, _ = observed(records(chain), the_pack(wrapper_site=True))
    assert got_c == {(0, "GET /items", "confirmed")}


def test_a_non_informative_slot_nearer_than_an_informative_one_is_skipped():
    chain = (("bare",), ("twomark",), ("lit",))
    got_c, _, _ = observed(records(chain), the_pack(wrapper_site=True))
    assert got_c == {(2, "GET /items", "confirmed")}


def test_the_first_informative_slot_wins_when_two_are_informative():
    chain = (("bare",), ("tmpl",), ("lit",))
    got_c, _, _ = observed(records(chain), the_pack(wrapper_site=True))
    assert got_c == {(1, "GET /documents/{?}", "possible")}


def test_a_query_only_unknown_stays_confirmed_at_the_wrapper_and_an_unknown_path_piece_is_possible():
    got_q, _, _ = observed(records((("bare",), ("query",), ("bare",))), the_pack(wrapper_site=True))
    got_p, _, _ = observed(records((("bare",), ("partseg2",), ("bare",))), the_pack(wrapper_site=True))
    assert got_q == {(1, "GET /x", "confirmed")} and got_p == {(1, "GET /a/{?}", "possible")}


def test_the_scan_starts_at_the_slot_right_after_the_leaf_also_when_a_call_result_links_them_without_a_parameter_node():
    """`url = norm(path)` then `get(url)`: the chain is leaf (`norm`'s argument) -> seed, with no parameter node between them, so
    `chain_of(leaf)[1]` is a slot. A scan that starts one element later would miss the seed's own template."""
    recs = [file_("f1", "svc/client.py"), imp("f1", "requests"),
            sym("F", "fetch_all", fid="f1", path="svc/client.py", start=10),
            sym("N", "norm", fid="f1", path="svc/client.py", start=1), param("PN", "N", 0, "p"), {"type": "ret", "id": "RN", "symbolId": "N", "line": 2, "ordinal": 0},
            data("RN", "PN", "ret", "p0"),
            call("cn", "F", "norm", line=11, resolution="resolved", targets=("N",)), vs("cn", "a0", X),
            call("sg", "F", "get", line=12), vs("sg", "a0", "/items"), data("sg", "cn", "a0", "r")]
    def view(wrapper_site):
        out = run_packs([recs], [the_pack(wrapper_site=wrapper_site, url_like=False)])
        return [(c["siteId"], c["key"], c["status"]) for c in out.consumers], sorted((i["siteId"], i["leaf"]) for i in out.lifted)

    assert view(False) == ([], [("cn", True), ("sg", False)])           # the leaf `cn` is unresolved; the seed has a slot beyond it
    assert view(True) == ([("sg", "GET /items", "confirmed")], [("sg", True)])


def test_a_parameter_that_feeds_two_seed_slots_belongs_to_the_first_one_reached_pinned_limit():
    """`def f(u): get("/a" + u); get(u + u)` called as `f(x)`: the first-discovered predecessor chain (seeds sorted by id) goes
    through `g1`, which is informative, so the caller `c` is dropped and `g1` is its wrapper site; `g2` stays its own site
    (a pinned limit, plan 51b rule). Had the chain gone through `g2`, `c` would stay in `lifted` as a keyless leaf."""
    recs = [file_("f1", "svc/client.py"), imp("f1", "requests"),
            sym("F", "f", fid="f1", path="svc/client.py", start=10), param("P", "F", 0, "u"),
            call("g1", "F", "get", line=11), vs("g1", "a0", "/a/" + X), data("g1", "P", "a0", "p0"),
            call("g2", "F", "get", line=12), vs("g2", "a0", X + X), data("g2", "P", "a0", "p0"),
            sym("G", "g", fid="f1", path="svc/client.py", start=30),
            call("c", "G", "f", line=31, resolution="resolved", targets=("F",)), vs("c", "a0", X)]
    out = run_packs([recs], [the_pack(wrapper_site=True, url_like=False)])
    assert [(c["siteId"], c["key"]) for c in out.consumers] == [("g1", "GET /a/{?}"), ("g2", "GET /{?}")]
    assert sorted(i["siteId"] for i in out.lifted) == ["g1", "g2"]


@pytest.mark.parametrize("shape,counts", [("root", True), ("root_unk", True), ("rootorigin", True), ("origin_only", False),
                                           ("origin_unk", False), ("slash_mark", False), ("origin_query", False), ("root_query", True)])
def test_the_plan_s_counts_and_does_not_count_examples_decide_the_wrapper_site(shape, counts):
    """Plan 51b: counts `/`, `\\x01/`, `http://h/`; does not count `http://localhost:1234`, `https://\\x01`, `\\x01/\\x01`. A bare
    origin is a base the caller completes, so the caller stays the consumer."""
    chain = (("lit",), (shape,), ("bare",))
    got_c, _, _ = observed(records(chain), the_pack(wrapper_site=True, url_like=True))
    assert {i for i, _, _ in got_c} == ({1} if counts else {0})


def test_a_parameter_that_feeds_two_arguments_of_one_call_belongs_to_the_first_one_reached_by_data_id_pinned_limit():
    """`def f(x, y): get(x + y)`, `def g(u): f("/documents/" + u, u)`, caller `h(): g("/other")`: `g`'s parameter is reached from
    `c.a0` or `c.a1` depending on the order the flows into the seed are visited (sorted data-record ids), so the wrapper site
    is `c`'s informative `a0` and `h1` is dropped. With the reverse order the chain would go through the bare `a1` and `h1`
    would stay the consumer."""
    recs = [file_("f1", "svc/client.py"), imp("f1", "requests"),
            sym("F", "f", fid="f1", path="svc/client.py", start=10), param("PX", "F", 0, "x"), param("PY", "F", 1, "y"),
            call("sg", "F", "get", line=11), vs("sg", "a0", X + X), data("sg", "PX", "a0", "p0"), data("sg", "PY", "a0", "p1"),
            sym("G", "g", fid="f1", path="svc/client.py", start=30), param("PU", "G", 0, "u"),
            call("c", "G", "f", line=31, resolution="resolved", targets=("F",)), vs("c", "a0", "/documents/" + X), vs("c", "a1", X),
            data("c", "PU", "a0", "p0"), data("c", "PU", "a1", "p0"),
            sym("H", "h", fid="f1", path="svc/client.py", start=50),
            call("h1", "H", "g", line=51, resolution="resolved", targets=("G",)), vs("h1", "a0", "/other")]
    out = run_packs([recs], [the_pack(wrapper_site=True, url_like=True)])
    assert [(c["siteId"], c["key"]) for c in out.consumers] == [("c", "GET /documents/{?}")]
    assert "h1" not in {i["siteId"] for i in out.lifted}
