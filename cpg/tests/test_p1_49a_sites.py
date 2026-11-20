"""P1-49a (red first): the per-site marker predicates, rule 4 (the opaque seed slot), the query-only rule and the two-provider link.

Plan: `docs/plans/connected-code-graph-p1-linking.md` P1-49 ("Marker sweep" table, rule 4, "Coverage tests (49a)" items 3-6);
analyst experiments: `docs/reviews/connected-code-graph-p1-5d-49.md` Pass 3 (N5) and Pass 4.

Two ways to put a marker in front of a site:

* **by element kind** (`{"unknown": ...}`, `{"importRef": ...}`, a `ref` with no record): what the extractor emits, so the
  emission matrix and the end-to-end rows read as the plan does;
* **as a literal `const` holding the marker character** (`"\\x01/shop/api/session"`): the site sees the exact string the plan
  names, whatever the emitter does, so each site is tested alone with a `\\x01`-only and a mixed alternative.

Each site's row of the plan table is one section below. `\\x01` is `UNKNOWN_VALUE` (a value not statically known), `\\x00` is
`UNKNOWN` (a name that should resolve); both are spelled out, never imported from the code under test.
"""
from __future__ import annotations

import inspect

import pytest

from packrt_helpers import call, constant, data, file_, param, ret, sym, vs
from ccg.linker import bind as bind_mod, link
from ccg.packrt import run_packs
from ccg.packrt.extract import MAX_ALTERNATIVES, MAX_REF_DEPTH, Corpus
from test_linker_bind import consumer as bind_consumer, server_world
from test_packrt_launch import mount_records
from test_packrt_launch import run as launch_run
from test_packrt_run import FASTAPI, FETCH, app_records, facts, fetch_chain

X = "\x01"
Z = "\x00"
UNK = {"unknown": "point-derived"}


def consumers(recs, *packs):
    return facts(recs, *(packs or (FETCH,))).consumers


def view(c):
    return (c["key"], c["status"], c["reason"])


def with_url(*values):
    """``fetch_chain()`` whose caller passes ``values`` (const strings or elements) as the URL."""
    recs = fetch_chain()
    recs[-1] = vs("c1", "a0", *values)
    return recs


# ---------------------------------------------------------------------------------------------------------------------
# extract.py:Corpus.alternatives (the emitting site): the element-kind matrix
# ---------------------------------------------------------------------------------------------------------------------

def _corpus(*records):
    return Corpus([list(records)])


def _chain_of_constants(n):
    """K0 -> ref K1 -> ... -> K<n> = "x": a ref chain longer than the depth cap when ``n`` > ``MAX_REF_DEPTH``."""
    return [constant(f"m::K{i}", {"ref": f"m::K{i + 1}"}) for i in range(n)] + [constant(f"m::K{n}", "x")]


@pytest.mark.parametrize("element,records,expected", [
    (UNK, (), [X]),                                                              # an unknown element: not statically known
    ({"unknown": "runtime-built"}, (), [X]),
    ({"unknown": "value-set-cap"}, (), [X]),
    ({"ref": "m::MISSING"}, (), [X]),                                            # a ref with no constant record
    ({"ref": "m::C"}, (constant("m::C", UNK),), [X]),                            # a constant whose own value is unknown
    ({"importRef": {"module": "cfg", "name": "BASE"}}, (), [Z]),                 # an imported name the resolver could not map
    ({"ref": "m::A"}, (constant("m::A", {"ref": "m::B"}), constant("m::B", {"ref": "m::A"})), [Z]),     # a ref cycle
    ({"ref": "m::K0"}, tuple(_chain_of_constants(MAX_REF_DEPTH + 4)), [Z]),      # the depth cap
    ({"concat": [{"const": "/a"}, UNK]}, (), ["/a" + X]),                        # a mixed string under construction
    ({"concat": [{"const": "/a"}, {"importRef": {"module": "cfg", "name": "N"}}]}, (), ["/a" + Z]),
    ({"template": [UNK, UNK]}, (), [X + X]),
])
def test_alternatives_marks_each_element_kind_with_the_marker_the_plan_names(element, records, expected):
    assert _corpus(*records).alternatives(element) == expected


def test_the_alternatives_cap_appends_x01_not_x00():
    c = _corpus(constant("m::W", *[str(i) for i in range(9)]))
    got = c.alternatives({"concat": [{"ref": "m::W"}, {"ref": "m::W"}]})
    assert len(got) == MAX_ALTERNATIVES + 1 and got[-1] == X


def test_a_value_set_holding_a_known_and_an_unknown_element_keeps_both_alternatives():
    c = _corpus()
    assert sorted(c.value_alternatives([{"const": "/a"}, UNK])) == sorted(["/a", X])


# ---------------------------------------------------------------------------------------------------------------------
# lift.py:_verbs_at (whole)
# ---------------------------------------------------------------------------------------------------------------------

def _with_verb(*values):
    recs = fetch_chain(init=True)
    recs[-1] = vs("fe", "a1.method", *values)
    return recs


def test_a_verb_slot_holding_only_x01_is_partial_and_falls_to_the_unknown_verb():
    (c,) = consumers(_with_verb(X))
    assert view(c) == ("* /api/x", "possible", "partial-key")


def test_a_verb_slot_holding_get_beside_x01_gives_get_and_is_partial():
    (c,) = consumers(_with_verb("GET", X))
    assert view(c) == ("GET /api/x", "possible", "partial-key")


def test_a_verb_slot_of_the_unknown_element_does_the_same_as_the_literal_marker():
    assert [view(c) for c in consumers(_with_verb("GET", UNK))] == [("GET /api/x", "possible", "partial-key")]


# ---------------------------------------------------------------------------------------------------------------------
# lift.py:run_lifting `resolved` (whole): the analyst's `\x01/shop/api/session` case
# ---------------------------------------------------------------------------------------------------------------------

def _lifted_at(out, site):
    return next(i for i in out.lifted if i["siteId"] == site)


def test_a_caller_slot_x01_slash_shop_api_session_is_resolved_and_keeps_its_path_possible():
    out = facts(with_url(X + "/shop/api/session"), FETCH)
    assert _lifted_at(out, "c1")["resolved"] is True
    (c,) = out.consumers
    assert view(c) == ("GET /shop/api/session", "possible", "partial-key")


def test_a_caller_slot_that_is_only_x01_is_not_resolved_and_makes_no_consumer():
    out = facts(with_url(X), FETCH)
    assert _lifted_at(out, "c1")["resolved"] is False
    assert out.consumers == []


def test_a_caller_slot_of_two_adjacent_x01_is_resolved_because_it_is_not_wholly_unresolved():
    out = facts(with_url(X + X), FETCH)
    assert _lifted_at(out, "c1")["resolved"] is True
    (c,) = out.consumers
    assert (c["status"], c["reason"]) == ("possible", "partial-key")


def test_the_same_three_rows_hold_for_x00():
    assert _lifted_at(facts(with_url(Z), FETCH), "c1")["resolved"] is False
    assert _lifted_at(facts(with_url(Z + Z), FETCH), "c1")["resolved"] is True


# ---------------------------------------------------------------------------------------------------------------------
# lift.py:run_lifting both `partialAlternatives` (whole, by list membership), with and without a verb companion
# ---------------------------------------------------------------------------------------------------------------------

NO_COMPANION = {"packFormat": 1, "id": "nc", "liftings": [{
    "id": "fetch-url", "role": {"kind": "http", "slot": "url"},
    "seeds": [{"call": {"callee": "fetch", "resolution": ["external", "unresolved"]}, "arg": "a0"}], "companions": []}],
    "rules": [{"id": "client", "match": {"lifted": {"lifting": "fetch-url"}}, "extract": {"url": {"lifted": "url"}},
               "emit": [{"consumes": {"kind": "http", "key": "GET {url}"}}]}]}


@pytest.mark.parametrize("the_pack", [FETCH, NO_COMPANION], ids=["with-verb-companion", "without-companion"])
def test_a_caller_slot_holding_a_path_beside_x01_is_a_partial_alternative_set(the_pack):
    out = facts(with_url("/a", X), the_pack)
    assert _lifted_at(out, "c1")["partialAlternatives"] is True
    assert [view(c) for c in out.consumers] == [("GET /a", "possible", "partial-key")]


@pytest.mark.parametrize("the_pack", [FETCH, NO_COMPANION], ids=["with-verb-companion", "without-companion"])
def test_an_unknown_in_the_query_only_is_not_a_partial_alternative_set(the_pack):
    out = facts(with_url("/a?q=" + X), the_pack)
    assert _lifted_at(out, "c1")["partialAlternatives"] is False                 # decided in `_one`, not here
    assert [view(c) for c in out.consumers] == [("GET /a", "confirmed", "lifted-role")]


# ---------------------------------------------------------------------------------------------------------------------
# run.py:_apply the `url` list (whole)
# ---------------------------------------------------------------------------------------------------------------------

def test_the_url_list_drops_a_wholly_unknown_alternative_and_keeps_the_partly_unknown_one():
    out = facts(with_url(X + "/shop/api/session", X), FETCH)
    assert [view(c) for c in out.consumers] == [("GET /shop/api/session", "possible", "partial-key")]


# ---------------------------------------------------------------------------------------------------------------------
# run.py:_one `unresolved` (has): a mount prefix; and the query-only unknown stays confirmed
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", [X + "/api", Z + "/api", "/api" + X, "/ap" + X + "i"], ids=repr)
def test_a_mount_prefix_holding_either_marker_drops_the_provider_with_the_unresolved_part_diagnostic(prefix):
    out = facts(app_records(prefix=prefix), FASTAPI)
    assert out.providers == []
    assert [d["reason"] for d in out.diagnostics] == ["unresolved-part"]


def test_a_mount_prefix_with_a_clean_and_a_marked_alternative_keeps_only_the_clean_provider():
    recs = [r for r in app_records() if r["id"] != "vs-inc-kw.prefix"] + [vs("inc", "kw.prefix", "/api", X + "/v2")]
    out = facts(recs, FASTAPI)
    assert [p["key"] for p in out.providers] == ["GET /api/health"]
    assert [d["reason"] for d in out.diagnostics] == ["unresolved-part"]


URL_PACK = {"packFormat": 1, "id": "hand", "rules": [{
    "id": "get", "match": {"callsite": {"callee": "g"}}, "extract": {"url": {"arg": "a0"}},
    "emit": [{"consumes": {"kind": "http", "key": "GET {url}"}}]}]}


def _direct(*values):
    return [file_("f1", "c.py"), sym("f", "f", fid="f1", path="c.py"), call("c1", "f", "g"), vs("c1", "a0", *values)]


@pytest.mark.parametrize("value", [{"template": [{"const": "/items?q="}, UNK]}, "/items?q=" + X, "/items?q=" + Z,
                                   {"template": [{"const": "/items#"}, UNK]}], ids=["element", "x01", "x00", "fragment"])
def test_a_direct_unknown_in_the_query_or_fragment_only_stays_confirmed(value):
    (c,) = run_packs([_direct(value)], [URL_PACK]).consumers
    assert view(c) == ("GET /items", "confirmed", "key-match")


def test_a_lifted_unknown_in_the_query_only_stays_confirmed_too():
    value = {"template": [{"const": "/items?q="}, UNK]}
    (c,) = consumers(with_url(value))
    assert view(c) == ("GET /items", "confirmed", "lifted-role")


def test_an_origin_drop_is_possible_direct_and_lifted():
    value = {"template": [UNK, {"const": "/items"}]}
    (d,) = run_packs([_direct(value)], [URL_PACK]).consumers
    (l,) = consumers(with_url(value))
    assert view(d) == ("GET /items", "possible", "partial-key")
    assert view(l) == ("GET /items", "possible", "partial-key")


def test_an_unknown_origin_beside_a_query_unknown_is_possible_because_of_the_origin_only():
    value = {"template": [UNK, {"const": "/items?q="}, UNK]}
    (d,) = run_packs([_direct(value)], [URL_PACK]).consumers
    assert view(d) == ("GET /items", "possible", "partial-key")


# ---------------------------------------------------------------------------------------------------------------------
# launch.py:launch_facts `mountPaths` (has)
# ---------------------------------------------------------------------------------------------------------------------

def _mount_paths(recs):
    return [f["path"] for f in launch_run(recs).launch_facts if f["fact"] == "mountPath"]


@pytest.mark.parametrize("path", [X + "/mcp", Z + "/mcp", "/mcp" + X, "/m" + X + "p"], ids=repr)
def test_a_mount_path_holding_either_marker_emits_no_mount_path_fact(path):
    assert _mount_paths(mount_records(path=path)) == []


def test_a_clean_mount_path_still_emits_its_fact_and_a_marked_sibling_alternative_does_not():
    assert _mount_paths(mount_records()) == ["/mcp"]
    recs = [r for r in mount_records() if r["id"] != "vs-mt-a0"] + [vs("mt", "a0", "/mcp", X + "/y")]
    assert _mount_paths(recs) == ["/mcp"]


# ---------------------------------------------------------------------------------------------------------------------
# mount.py:MountIndex.facts (has; the stored prefix)
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", [X + "/api", Z + "/api"], ids=repr)
def test_a_marked_mount_prefix_is_possible_partial_key_and_is_stored_as_a_placeholder_never_a_raw_marker(prefix):
    (m,) = facts(app_records(prefix=prefix), FASTAPI).mounts
    assert (m["status"], m["reason"]) == ("possible", "partial-key")
    assert m["prefixes"] == ["{?}/api"]
    assert not any(X in p or Z in p for p in m["prefixes"])


def test_a_clean_mount_prefix_is_confirmed_and_stored_as_written():
    (m,) = facts(app_records(prefix="/api"), FASTAPI).mounts
    assert (m["status"], m["reason"], m["prefixes"]) == ("confirmed", "key-match", ["/api"])


def test_a_prefix_set_with_a_marked_member_is_possible_and_stores_no_raw_marker():
    recs = [r for r in app_records() if r["id"] != "vs-inc-kw.prefix"] + [vs("inc", "kw.prefix", "/api", X + "/v2")]
    (m,) = facts(recs, FASTAPI).mounts
    assert (m["status"], m["reason"]) == ("possible", "partial-key")
    assert sorted(m["prefixes"]) == ["/api", "{?}/v2"]


# ---------------------------------------------------------------------------------------------------------------------
# linker/bind.py: no change; a bind value with x01 is not treated specially (binding text is a different channel)
# ---------------------------------------------------------------------------------------------------------------------

def test_bind_py_treats_x01_in_binding_text_as_any_other_character_and_x00_as_before():
    assert bind_mod.bundle_url("http://\x01:8200/api/x") is True
    assert bind_mod.bundle_url("http://\x02:8200/api/x") is True
    assert bind_mod.bundle_url("http://\x00:8200/api/x") is False

    def outcome(url):
        f, b = server_world(url)
        return bind_mod.Binder.bind(__import__("test_linker_bind").make_binder(f, b), bind_consumer()) is None
    assert outcome("http://\x01:8200/api/x") == outcome("http://\x02:8200/api/x")      # x01 is no more special than x02
    assert outcome("http://\x00:8200/api/x") is True                                     # the NUL screen is unchanged


def test_bind_py_source_still_speaks_only_of_the_nul_screen():
    src = inspect.getsource(bind_mod)
    assert '"\\x00"' in src and "\\x01" not in src


# ---------------------------------------------------------------------------------------------------------------------
# rule 4: a lifted fact is the URL only if every slot below the leaf is transparent (the analyst's seed-sibling experiment)
# ---------------------------------------------------------------------------------------------------------------------

SEED_SIBLINGS = {
    "concat-unknown-unknown (self.base + path)": {"concat": [UNK, UNK]},
    "template-unknown-unknown (f'{BASE_URL}{path}')": {"template": [UNK, UNK]},
    "concat-const-unknown ('http://h' + path)": {"concat": [{"const": "http://h"}, UNK]},
}


@pytest.mark.parametrize("shape", sorted(SEED_SIBLINGS))
def test_a_seed_slot_with_an_unresolved_sibling_makes_the_lifted_fact_possible_and_records_the_opaque_slot(shape):
    recs = fetch_chain() + [vs("fe", "a0", SEED_SIBLINGS[shape])]
    out = facts(recs, FETCH)
    (c,) = out.consumers
    assert view(c) == ("GET /api/x", "possible", "partial-key")
    assert c["evidence"]["opaqueSlot"] == {"siteId": "fe", "arg": "a0"}
    assert _lifted_at(out, "c1")["opaqueSlot"] == {"siteId": "fe", "arg": "a0"}


def test_control_the_unchanged_fetch_chain_stays_confirmed_lifted_role():
    out = facts(fetch_chain(), FETCH)
    (c,) = out.consumers
    assert view(c) == ("GET /api/x", "confirmed", "lifted-role")
    assert "opaqueSlot" not in c["evidence"]
    assert not _lifted_at(out, "c1").get("opaqueSlot")


def test_control_a_seed_slot_of_the_bare_unknown_parameter_stays_confirmed_lifted_role():
    out = facts(fetch_chain() + [vs("fe", "a0", UNK)], FETCH)
    (c,) = out.consumers
    assert view(c) == ("GET /api/x", "confirmed", "lifted-role")
    assert "opaqueSlot" not in c["evidence"]


def test_a_seed_slot_of_two_adjacent_marker_literals_is_opaque_and_one_marker_literal_is_transparent():
    (opaque,) = consumers(fetch_chain() + [vs("fe", "a0", X + X)])
    (bare,) = consumers(fetch_chain() + [vs("fe", "a0", X)])
    assert opaque["status"] == "possible" and bare["status"] == "confirmed"


def test_a_two_hop_chain_whose_middle_slot_is_concat_const_unknown_is_possible_and_names_the_middle_slot():
    recs = fetch_chain() + [
        sym("outer", "outer", fid="f1", path="web/client.ts", start=50), param("q0", "outer", 0, "p", line=51),
        call("oc", "outer", "get", line=52, resolution="resolved", targets=("get",)), data("oc", "q0", "a0", "p0"),
        vs("oc", "a0", {"concat": [{"const": "/p"}, UNK]}),
        sym("top", "top", fid="f1", path="web/client.ts", start=70),
        call("tc", "top", "outer", line=71, resolution="resolved", targets=("outer",)), vs("tc", "a0", "/api/deep")]
    got = {c["siteId"]: c for c in consumers(recs)}
    assert view(got["tc"]) == ("GET /api/deep", "possible", "partial-key")
    assert got["tc"]["evidence"]["opaqueSlot"] == {"siteId": "oc", "arg": "a0"}
    assert view(got["c1"]) == ("GET /api/x", "confirmed", "lifted-role")              # the other chain is still all-transparent


def test_an_opaque_seed_slot_directly_behind_a_call_result_hop_is_found_by_the_rule_4_check():
    """P1-49b gate m1: `fetch(ident("/api/x"))` where the seed slot is `"/p" + <unknown>`: the leaf is `ident`'s argument, and the
    seed slot sits at index 1 of the chain, reached through a `call-ret` hop (a parameter hop would make that index a param)."""
    recs = [
        file_("f1", "web/client.ts", lang="javascript"),
        sym("ident", "ident", fid="f1", path="web/client.ts", start=5), param("ip0", "ident", 0, "x", line=6), ret("rt", "ident"),
        data("rt", "ip0", "ret", "p0"),
        sym("outer", "outer", fid="f1", path="web/client.ts", start=10),
        call("ic", "outer", "ident", line=11, resolution="resolved", targets=("ident",)), vs("ic", "a0", "/api/x"),
        call("fe", "outer", "fetch", line=12, resolution="external", external="lib/fetch.js:<module>.fetch"),
        data("fe", "ic", "a0", "r"), vs("fe", "a0", {"concat": [{"const": "/p"}, UNK]}),
    ]
    out = facts(recs, FETCH)
    (c,) = out.consumers
    assert c["siteId"] == "ic" and view(c) == ("GET /api/x", "possible", "partial-key")
    assert c["evidence"]["opaqueSlot"] == {"siteId": "fe", "arg": "a0"}


def _second_argument_chain(second):
    """`top -> outer(p) -> get(b, p) -> fetch(path)`: the middle slot is the SECOND argument of `get(prefix, path)` (`oc`, `a1`),
    holding `second`; its first argument (`oc`, `a0`) is the constant `/b`."""
    return [
        file_("f1", "web/client.ts", lang="javascript"),
        sym("get", "get", fid="f1", path="web/client.ts", start=10), param("g0", "get", 0, "prefix", line=11), param("g1", "get", 1, "path", line=11),
        call("fe", "get", "fetch", line=12, resolution="external", external="lib/fetch.js:<module>.fetch"), data("fe", "g1", "a0", "p1"),
        sym("outer", "outer", fid="f1", path="web/client.ts", start=50), param("q0", "outer", 0, "p", line=51),
        call("oc", "outer", "get", line=52, resolution="resolved", targets=("get",)), data("oc", "q0", "a1", "p0"),
        vs("oc", "a0", "/b"), vs("oc", "a1", second),
        sym("top", "top", fid="f1", path="web/client.ts", start=70),
        call("tc", "top", "outer", line=71, resolution="resolved", targets=("outer",)), vs("tc", "a0", "/api/deep")]


def test_the_opaque_slot_names_the_argument_path_it_is_at_not_a_fixed_first_argument():
    """P1-49b gate m3: an opaque second argument is reported as `a1`."""
    (c,) = consumers(_second_argument_chain({"concat": [{"const": "/p"}, UNK]}))
    assert c["siteId"] == "tc" and view(c) == ("GET /api/deep", "possible", "partial-key")
    assert c["evidence"]["opaqueSlot"] == {"siteId": "oc", "arg": "a1"}


def test_a_transparent_second_argument_is_judged_on_its_own_value_not_on_the_constant_first_argument():
    (c,) = consumers(_second_argument_chain(UNK))
    assert view(c) == ("GET /api/deep", "confirmed", "lifted-role") and "opaqueSlot" not in c["evidence"]


def test_the_caller_side_template_with_an_unknown_origin_is_possible_by_rule_2_and_has_no_opaque_slot():
    (c,) = consumers(with_url({"concat": [UNK, {"const": "/api/x"}]}))
    assert view(c) == ("GET /api/x", "possible", "partial-key")
    assert "opaqueSlot" not in c["evidence"]


def test_known_limit_a_wrapper_composing_the_url_in_a_local_stays_confirmed_lifted_role():
    """PINNED KNOWN LIMIT (plan P1-49 rule 4; analyst Pass 4). `url = f"{BASE_URL}{path}"; Request(url)` leaves the seed slot a
    bare `point-derived` unknown, which the bundle cannot tell from the parameter itself (a `data` record and the `unknown`
    reason carry no whole-versus-part information), so the lifted fact stays `confirmed`/`lifted-role` and no `opaqueSlot` is
    recorded. Telling them apart needs a bundle value-set element for a derived local plus the extractor's read of local
    initializers: a P2 item (owner `architect`). A later extractor improvement flips this test on purpose."""
    recs = fetch_chain() + [vs("fe", "a0", {"unknown": "point-derived"})]
    (c,) = consumers(recs)
    assert view(c) == ("GET /api/x", "confirmed", "lifted-role")
    assert "opaqueSlot" not in c["evidence"]


# ---------------------------------------------------------------------------------------------------------------------
# end to end: one path in two units, a consumer with an unknown base, linked possible/partial-key to both
# ---------------------------------------------------------------------------------------------------------------------

E2E_PACK = {"packFormat": 1, "id": "hand", "rules": [
    {"id": "serve", "match": {"callsite": {"callee": "serve"}}, "extract": {}, "emit": [{"provides": {"kind": "http", "key": "GET /health"}}]},
    {"id": "get", "match": {"callsite": {"callee": "g"}}, "extract": {"url": {"arg": "a0"}},
     "emit": [{"consumes": {"kind": "http", "key": "GET {url}"}}]}]}


def _unit(unit, owner, *, serves=True, url=None):
    fid = f"f-{unit}"
    recs = [file_(fid, f"{unit}/app.py", unit=unit), sym(owner, owner, fid=fid, path=f"{unit}/app.py")]
    if serves:
        recs.append(call(f"serve-{unit}", owner, "serve"))
    if url is not None:
        recs += [call(f"c-{unit}", owner, "g"), vs(f"c-{unit}", "a0", url)]
    return recs


def _world(url):
    bundles = [_unit("u1", "P1"), _unit("u2", "P2"), _unit("web", "C", serves=False, url=url)]
    out = run_packs(bundles, [E2E_PACK])
    return out, link(out, bundles)


def test_an_unknown_base_links_to_both_units_possible_partial_key():
    out, links = _world({"template": [UNK, {"const": "/health"}]})
    (c,) = out.consumers
    assert view(c) == ("GET /health", "possible", "partial-key")
    assert sorted((l.provider, l.status, l.reason, l.key_norm) for l in links.links) == [
        ("P1", "possible", "partial-key", "GET /health"), ("P2", "possible", "partial-key", "GET /health")]


def test_control_a_known_base_links_to_both_units_confirmed():
    out, links = _world({"const": "http://svc:8000/health"})
    assert sorted((l.provider, l.status, l.reason) for l in links.links) == [
        ("P1", "confirmed", "key-match"), ("P2", "confirmed", "key-match")]
