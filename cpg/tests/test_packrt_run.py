"""P1-16 `ccg.packrt.run_packs` over planted bundles: rule emission, resolution and status, role lifting, mount composition.

Plan: `docs/plans/connected-code-graph-p1-linking.md` § P1-16 behaviours (1)-(5), (7) and (9). The shipped packs
(`cpg/packs/{fastapi,fetch,fastmcp}`) are loaded through `ccg.rulepack.load_pack`; one case renames every framework token of
the fastapi pack and expects the same facts, so a branch on a pack's name cannot pass. The stored-slice equivalence is
`tests/test_packrt_conformance.py`.
"""
import copy
import json

import pytest

from packrt_helpers import call, cond, constant, ctrl, data, dec, file_, imp, pack, param, sym, vs
from ccg.packrt import PackFacts, run_packs
from ccg.packrt.match import UnsupportedSubject
from ccg.rulepack import reason_row

FASTAPI = pack("fastapi")
FETCH = pack("fetch")
FASTMCP = pack("fastmcp")


def facts(records, *packs) -> PackFacts:
    return run_packs([records], list(packs))


def keys(fs):
    return sorted((f["key"], f["status"], f["reason"]) for f in fs)


def app_records(*, prefix="/api", route="/health", verb="get", extra=()):
    """A factory ``build`` holding ``@router.<verb>(route)`` on handler ``h``, mounted by ``create_app`` under ``prefix``."""
    recs = [
        file_("f1", "svc/app.py"), imp("f1", "fastapi", "FastAPI"),
        sym("build", "build_router", fid="f1", path="svc/app.py"),
        sym("h", "health", fid="f1", path="svc/app.py", parent="build"),
        dec("d1", "h", f"router.{verb}", ("a0", route), line=7),
        sym("mk", "create_app", fid="f1", path="svc/app.py", start=20),
        call("fa", "mk", "FastAPI"), call("bc", "mk", "build_router", resolution="resolved", targets=("build",)),
        call("inc", "mk", "include_router", line=25), data("inc", "bc", "a0", "r"),
    ]
    if prefix is not None:
        recs.append(vs("inc", "kw.prefix", prefix))
    return recs + list(extra)


# ---- (1) a literal key, (2) a key built from a ref

def test_a_literal_route_under_a_mount_is_confirmed_key_match_and_names_the_mount_hop():
    out = facts(app_records(), FASTAPI)
    (p,) = out.providers
    assert (p["edge"], p["kind"], p["key"], p["status"], p["reason"]) == ("PROVIDES", "http", "GET /api/health", "confirmed", "key-match")
    assert (p["anchorId"], p["siteId"], p["line"], p["packId"], p["ruleId"]) == ("h", "d1", 7, "fastapi", "route")
    (m,) = p["evidence"]["mounts"]
    assert m["how"] == "mounted" and [s["siteId"] for s in m["sites"]] == ["inc"] and m["conditions"] == []
    assert reason_row(p["reason"])["status"] == "confirmed"


def test_the_verb_comes_from_the_decorator_name_and_the_any_method_rule_gives_star():
    assert keys(facts(app_records(verb="post"), FASTAPI).providers) == [("POST /api/health", "confirmed", "key-match")]
    assert keys(facts(app_records(verb="api_route"), FASTAPI).providers) == [("* /api/health", "confirmed", "key-match")]


def test_a_route_built_from_a_ref_resolves_through_the_value_set():
    recs = app_records(route={"concat": [{"ref": "svc/app.py::BASE"}, {"const": "/h"}]}, extra=[constant("svc/app.py::BASE", "/v2")])
    assert keys(facts(recs, FASTAPI).providers) == [("GET /api/v2/h", "confirmed", "key-match")]


def test_a_provider_key_with_an_unresolved_part_is_dropped_and_listed_not_emitted():
    recs = app_records(route={"concat": [{"unknown": "runtime-built"}, {"const": "/h"}]})
    out = facts(recs, FASTAPI)
    assert out.providers == []
    assert out.diagnostics == [{"packId": "fastapi", "ruleId": "route", "anchorId": "h", "siteId": "d1", "line": 7,
                                "reason": "unresolved-part"}]


def test_an_unresolved_mount_prefix_drops_the_provider_too():
    recs = app_records(prefix={"unknown": "point-derived"})
    out = facts(recs, FASTAPI)
    assert out.providers == [] and len(out.diagnostics) == 1


def test_a_decorator_without_a_path_emits_nothing_and_a_file_without_the_import_gate_is_not_read():
    assert facts(app_records(route=None) and [r for r in app_records() if r["type"] != "decorator"]
                 + [dec("d1", "h", "router.get")], FASTAPI).providers == []
    ungated = [r for r in app_records() if r["type"] != "import"]
    assert facts(ungated, FASTAPI).providers == []


def test_a_consumer_key_with_an_unresolved_segment_is_possible_partial_key_with_a_placeholder():
    pack = {"packFormat": 1, "id": "c", "rules": [{
        "id": "get", "match": {"callsite": {"callee": "get_json"}},
        "extract": {"url": {"arg": "a0"}}, "emit": [{"consumes": {"kind": "http", "key": "GET {url}"}}]}]}
    recs = [file_("f1", "c.py"), sym("f", "f", fid="f1", path="c.py"), call("c1", "f", "get_json"),
            vs("c1", "a0", {"concat": [{"const": "/u/"}, {"unknown": "point-derived"}]})]
    (c,) = facts(recs, pack).consumers
    assert (c["key"], c["status"], c["reason"]) == ("GET /u/{?}", "possible", "partial-key")


def test_a_non_http_consumer_with_an_unresolved_key_is_possible_runtime_built_without_a_key():
    recs = [file_("f1", "c.py"), imp("f1", "mcp"), sym("f", "f", fid="f1", path="c.py"), call("c1", "f", "call_tool"),
            vs("c1", "a0", {"unknown": "point-derived"})]
    (c,) = facts(recs, FASTMCP).consumers
    assert (c["kind"], c["key"], c["status"], c["reason"]) == ("mcp_tool", None, "possible", "runtime-built")
    assert reason_row("runtime-built") and reason_row("runtime-built")["status"] == "possible"


# ---- (7) mcp tools

def tool_records(**dec_args):
    return [file_("f1", "srv/mcp.py"), imp("f1", "mcp.server.fastmcp", "FastMCP"),
            sym("t", "send_message", fid="f1", path="srv/mcp.py"), dec("d", "t", "mcp.tool", *dec_args.items())]


def test_a_tool_name_is_kw_name_then_a0_then_the_function_name():
    assert keys(facts(tool_records(), FASTMCP).providers) == [("send_message", "confirmed", "key-match")]
    assert keys(facts(tool_records(a0="first"), FASTMCP).providers) == [("first", "confirmed", "key-match")]
    assert keys(facts(tool_records(**{"kw.name": "named", "a0": "first"}), FASTMCP).providers) == [("named", "confirmed", "key-match")]


def test_the_tool_rule_needs_a_fastmcp_import_in_the_file_and_a_tool_shaped_decorator():
    assert facts([r for r in tool_records() if r["type"] != "import"], FASTMCP).providers == []
    other = [r if r["type"] != "decorator" else dec("d", "t", "mcp.resource") for r in tool_records()]
    assert facts(other, FASTMCP).providers == []
    alt = [r if r["type"] != "import" else imp("f1", "fastmcp") for r in tool_records()]
    assert len(facts(alt, FASTMCP).providers) == 1                      # the second import gate


def test_a_call_tool_site_is_a_consumes_mcp_tool_anchored_at_its_enclosing_function():
    recs = [file_("f1", "cli/x.py"), imp("f1", "mcp", "ClientSession"), sym("f", "run", fid="f1", path="cli/x.py"),
            call("c1", "f", "call_tool", line=9), vs("c1", "a0", "send_message")]
    (c,) = facts(recs, FASTMCP).consumers
    assert (c["edge"], c["kind"], c["key"], c["status"], c["reason"], c["anchorId"], c["siteId"], c["line"]) == (
        "CONSUMES", "mcp_tool", "send_message", "confirmed", "key-match", "f", "c1", 9)
    assert facts([r for r in recs if r["type"] != "import"], FASTMCP).consumers == []


# ---- (3) role lifting

def fetch_chain(*, data_status="confirmed", data_reason="intra-flow", init=False):
    """``get(path)`` forwards its parameter to ``fetch``; ``caller`` passes a literal to ``get``."""
    recs = [
        file_("f1", "web/client.ts", lang="javascript"),
        sym("get", "get", fid="f1", path="web/client.ts", start=10), param("p0", "get", 0, "path", line=11),
        call("fe", "get", "fetch", line=12, resolution="external", external="lib/fetch.js:<module>.fetch"),
        data("fe", "p0", "a0", "p0", status=data_status, reason=data_reason),
        sym("caller", "caller", fid="f1", path="web/client.ts", start=30),
        call("c1", "caller", "get", line=31, resolution="resolved", targets=("get",)), vs("c1", "a0", "/api/x"),
    ]
    if init:
        recs += [vs("fe", "a1.method", "POST")]
    return recs


def test_a_parameter_that_flows_to_a_recognized_client_call_makes_the_function_a_client_and_its_caller_the_consumer():
    out = facts(fetch_chain(), FETCH)
    (c,) = out.consumers
    assert (c["key"], c["status"], c["reason"], c["anchorId"], c["siteId"]) == ("GET /api/x", "confirmed", "lifted-role", "caller", "c1")
    hops = c["evidence"]["hops"]
    assert [(h["ord"], h["kind"], h["id"], h["urlPath"]) for h in hops] == [
        (0, "callsite", "c1", "a0"), (1, "symbol", "get", "p0"), (2, "callsite", "fe", "a0")]
    assert hops[2]["callee"] == "fetch"


def test_the_verb_role_lifts_beside_the_url_and_a_bare_call_defaults_to_get():
    (c,) = facts(fetch_chain(), FETCH).consumers
    assert c["key"].startswith("GET ")                                  # fetch(url): bareDefault
    recs = fetch_chain(init=True)
    recs.append(data("fe", "p0", "a1.method", "p1.method"))             # method now arrives from a second parameter
    recs += [param("p1", "get", 1, "init")]
    recs.append(vs("c1", "a1.method", "DELETE"))
    (c2,) = facts(recs, FETCH).consumers
    assert c2["key"] == "DELETE /api/x"
    assert [h["verbPath"] for h in c2["evidence"]["hops"]][0] == "a1.method"


def test_a_verb_that_is_not_an_http_method_is_the_any_method_star_and_an_unresolved_one_is_partial():
    recs = fetch_chain(init=True)
    recs[-1] = vs("fe", "a1.method", "FETCHALL")
    (c,) = facts(recs, FETCH).consumers
    assert c["key"] == "* /api/x"
    recs[-1] = vs("fe", "a1.method", "GET", {"unknown": "point-derived"})
    (c,) = facts(recs, FETCH).consumers
    assert (c["key"], c["status"], c["reason"]) == ("GET /api/x", "possible", "partial-key")


def test_a_possible_data_hop_makes_the_lifted_consumer_possible_and_names_the_record_(data_reason="shell-dynamic"):
    (c,) = facts(fetch_chain(data_status="possible", data_reason=data_reason), FETCH).consumers
    assert (c["status"], c["reason"]) == ("possible", "partial-key")
    assert c["evidence"]["weakenedBy"] == "d-fe-a0-p0-p0"
    (ok,) = facts(fetch_chain(), FETCH).consumers
    assert ok["status"] == "confirmed" and "weakenedBy" not in ok["evidence"]


def test_a_shell_function_lifts_the_same_way_through_a_hand_written_pack_with_no_fetch_in_it():
    pack = {"packFormat": 1, "id": "sh", "liftings": [{
        "id": "curl-url", "role": {"kind": "http", "slot": "url"},
        "seeds": [{"call": {"callee": "curl"}, "arg": "a0"}], "companions": []}],
        "rules": [{"id": "client", "match": {"lifted": {"lifting": "curl-url"}}, "extract": {"url": {"lifted": "url"}},
                   "emit": [{"consumes": {"kind": "http", "key": "* {url}"}}]}]}
    recs = [file_("f1", "ops/get.sh", lang="shell"),
            sym("fn", "fetch_json", fid="f1", path="ops/get.sh", kind="function"), param("p0", "fn", 0, "$1"),
            call("cu", "fn", "curl", kind="command"), data("cu", "p0", "a0", "p0"),
            sym("scr", "run", fid="f1", path="ops/get.sh", kind="script"),
            call("c1", "scr", "fetch_json", kind="command", resolution="resolved", targets=("fn",)),
            vs("c1", "a0", "http://localhost:8000/health")]
    (c,) = facts(recs, pack).consumers
    assert (c["key"], c["anchorId"], c["status"]) == ("* /health", "scr", "confirmed")


def test_lifting_runs_to_a_fixpoint_through_two_wrappers_and_a_returned_parameter():
    recs = fetch_chain()
    recs += [sym("outer", "outer", fid="f1", path="web/client.ts", start=50), param("q0", "outer", 0, "p", line=51),
             call("oc", "outer", "get", line=52, resolution="resolved", targets=("get",)), data("oc", "q0", "a0", "p0"),
             sym("top", "top", fid="f1", path="web/client.ts", start=70),
             call("tc", "top", "outer", line=71, resolution="resolved", targets=("outer",)), vs("tc", "a0", "/api/deep")]
    out = facts(recs, FETCH)
    got = {c["siteId"]: c["key"] for c in out.consumers}
    assert got == {"c1": "GET /api/x", "tc": "GET /api/deep"}           # `oc` forwards a parameter: it is a hop, not a leaf
    deep = next(c for c in out.consumers if c["siteId"] == "tc")
    hops = deep["evidence"]["hops"]
    assert [(h["ord"], h["id"], h["urlPath"]) for h in hops] == [(0, "tc", "a0"), (1, "outer", "p0"), (2, "get", "p0"), (3, "fe", "a0")]
    assert hops[1]["viaSiteId"] == "oc"                                   # `outer` reaches `get` through its call `oc`
    assert {i["siteId"]: (i["leaf"], i["resolved"]) for i in out.lifted} == {
        "fe": (False, False), "oc": (False, False), "c1": (True, True), "tc": (True, True)}


def test_a_resolved_slot_that_has_a_lifted_child_is_not_a_consumer_but_stays_in_the_lifted_list():
    """O-13: the slice emitted every resolved slot; the pack emits only resolved leaves."""
    recs = fetch_chain()
    recs += [sym("outer", "outer", fid="f1", path="web/client.ts", start=50), param("q0", "outer", 0, "p"),
             call("oc", "outer", "get", line=52, resolution="resolved", targets=("get",)), data("oc", "q0", "a0", "p0"),
             vs("oc", "a0", "/api/default")]                                # `oc` resolves itself AND feeds a caller
    recs += [sym("top", "top", fid="f1", path="web/client.ts", start=70),
             call("tc", "top", "outer", line=71, resolution="resolved", targets=("outer",)), vs("tc", "a0", "/api/deep")]
    out = facts(recs, FETCH)
    assert "oc" not in {c["siteId"] for c in out.consumers}
    assert {i["siteId"]: (i["resolved"], i["leaf"]) for i in out.lifted}["oc"] == (True, False)


def test_a_fetch_seed_must_be_external_or_unresolved():
    recs = [r if r["id"] != "fe" else {**r, "resolution": "resolved", "targets": [{"symbolId": "get", "qname": "q"}]}
            for r in fetch_chain()]
    assert facts(recs, FETCH).consumers == []


# ---- (5) mount composition

def test_mounts_compose_outward_outermost_prefix_first_to_any_depth():
    recs = app_records(prefix="/api") + [
        sym("main", "main", fid="f1", path="svc/app.py", start=40),
        call("mc", "main", "make_app", resolution="resolved", targets=("mk",)),
        call("inc2", "main", "include_router", line=45), data("inc2", "mc", "a0", "r"), vs("inc2", "kw.prefix", "/v1")]
    (p,) = facts(recs, FASTAPI).providers
    assert p["key"] == "GET /v1/api/health" and p["status"] == "confirmed"
    assert [s["siteId"] for s in p["evidence"]["mounts"][0]["sites"]] == ["inc2", "inc"]


def test_a_mount_cycle_stops():
    recs = app_records() + [call("inc3", "build", "include_router"), call("bc2", "build", "create_app", resolution="resolved", targets=("mk",)),
                            data("inc3", "bc2", "a0", "r")]
    out = facts(recs, FASTAPI)
    assert out.providers and all(p["status"] == "confirmed" for p in out.providers)


def test_a_conditional_mount_keeps_both_variants_with_the_condition():
    recs = app_records(prefix="/api") + [
        call("inc_dev", "mk", "include_router", line=30), data("inc_dev", "bc", "a0", "r"), vs("inc_dev", "kw.prefix", "/api"),
        cond("k1", "mk", "dev_surface", line=29), ctrl("inc_dev", "k1")]
    (p,) = facts(recs, FASTAPI).providers
    variants = p["evidence"]["mounts"]
    assert len(variants) == 2 and p["key"] == "GET /api/health"
    assert sorted(len(v["conditions"]) for v in variants) == [0, 1]
    (cv,) = [v for v in variants if v["conditions"]]
    assert cv["conditions"] == [{"siteId": "inc_dev", "line": 29, "code": "dev_surface"}]
    mounts = {m["siteId"]: m for m in facts(recs, FASTAPI).mounts}
    assert mounts["inc_dev"]["conditions"] == [{"siteId": "inc_dev", "line": 29, "code": "dev_surface"}] and mounts["inc"]["conditions"] == []
    assert mounts["inc"]["prefixes"] == ["/api"] and mounts["inc"]["factoryIds"] == ["build"]


def test_two_mounts_with_different_prefixes_give_two_keys():
    recs = app_records(prefix="/a") + [call("incb", "mk", "include_router"), data("incb", "bc", "a0", "r"), vs("incb", "kw.prefix", "/b")]
    assert keys(facts(recs, FASTAPI).providers) == [("GET /a/health", "confirmed", "key-match"), ("GET /b/health", "confirmed", "key-match")]


def test_an_include_without_a_prefix_argument_mounts_at_the_default_empty_prefix():
    assert keys(facts(app_records(prefix=None), FASTAPI).providers) == [("GET /health", "confirmed", "key-match")]


def test_a_mount_call_in_a_file_without_the_import_gate_mounts_nothing():
    """The declaration's own `scope.imports` gates the include call's file, not only the decorator's."""
    recs = [r for r in app_records() if r["id"] not in ("mk", "fa", "bc", "inc") and not r["id"].startswith(("d-inc", "vs-inc"))]
    recs += [file_("f2", "svc/main.py"), sym("mk", "create_app", fid="f2", path="svc/main.py"),
             call("fa", "mk", "FastAPI"), call("bc", "mk", "build_router", resolution="resolved", targets=("build",)),
             call("inc", "mk", "include_router"), data("inc", "bc", "a0", "r"), vs("inc", "kw.prefix", "/api")]
    (p,) = facts(recs, FASTAPI).providers
    assert (p["key"], p["status"], p["evidence"]["mounts"][0]["how"]) == ("GET /health", "possible", "unmounted-router")
    assert facts(recs, FASTAPI).mounts == []


def test_a_handler_in_a_function_that_builds_the_app_is_a_root_and_needs_no_mount():
    recs = [file_("f1", "svc/a.py"), imp("f1", "fastapi"), sym("m", "main", fid="f1", path="svc/a.py"),
            call("fa", "m", "FastAPI"), sym("h", "ping", fid="f1", path="svc/a.py", parent="m"), dec("d", "h", "app.get", ("a0", "/ping"))]
    (p,) = facts(recs, FASTAPI).providers
    assert (p["key"], p["status"], p["evidence"]["mounts"][0]["how"]) == ("GET /ping", "confirmed", "app")


def test_a_function_holding_fastapi_plus_a_local_apirouter_is_possible_not_a_root():
    """Spec §12.1 fixture: the decorator record names no receiver, so which object the handler is on is undecidable."""
    recs = [file_("f1", "svc/a.py"), imp("f1", "fastapi"), sym("m", "main", fid="f1", path="svc/a.py"),
            call("fa", "m", "FastAPI"), call("ar", "m", "APIRouter"),
            sym("h", "join", fid="f1", path="svc/a.py", parent="m"), dec("d", "h", "inner.get", ("a0", "/join"))]
    (p,) = facts(recs, FASTAPI).providers
    assert (p["key"], p["status"], p["reason"], p["evidence"]["mounts"][0]["how"]) == ("GET /join", "possible", "partial-key", "unmounted-router")


def test_a_factory_whose_router_carries_its_own_prefix_is_possible_with_the_lower_bound_key():
    """Spec §12.1 fixture: `APIRouter(prefix="/v1")` mounted under `/api` is served at `/api/v1/x`; the pack cannot compose both."""
    recs = app_records(prefix="/api", route="/x") + [call("ar", "build", "APIRouter"), vs("ar", "kw.prefix", "/v1")]
    (p,) = facts(recs, FASTAPI).providers
    assert (p["key"], p["status"], p["reason"]) == ("GET /api/x", "possible", "partial-key")
    plain = app_records(prefix="/api", route="/x") + [call("ar", "build", "APIRouter")]      # APIRouter() with no prefix argument
    assert facts(plain, FASTAPI).providers[0]["status"] == "confirmed"
    other = app_records(prefix="/api", route="/x") + [call("ar", "build", "Other"), vs("ar", "kw.prefix", "/v1")]
    assert facts(other, FASTAPI).providers[0]["status"] == "confirmed"      # a prefix argument on some other call


def test_a_router_held_in_a_module_variable_stays_unmounted_possible_o3():
    """`router = APIRouter()` then `app.include_router(router)`: no record links the variable to its decorators (O-3)."""
    recs = [file_("f1", "svc/r.py"), imp("f1", "fastapi"), sym("mod", "<module>", fid="f1", path="svc/r.py", kind="module"),
            call("ar", "mod", "APIRouter"), sym("h", "list_items", fid="f1", path="svc/r.py", parent="mod"),
            dec("d", "h", "router.get", ("a0", "/items")),
            call("inc", "mod", "include_router"), data("inc", "ar", "a0", "r"), vs("inc", "kw.prefix", "/shop")]
    (p,) = facts(recs, FASTAPI).providers
    assert (p["key"], p["status"], p["reason"]) == ("GET /items", "possible", "partial-key")


def test_a_variable_router_beside_a_mounted_factory_does_not_borrow_its_mount():
    recs = app_records() + [sym("h2", "other", fid="f1", path="svc/app.py"), dec("d2", "h2", "r.get", ("a0", "/loose"))]
    out = {p["key"]: p["status"] for p in facts(recs, FASTAPI).providers}
    assert out == {"GET /api/health": "confirmed", "GET /loose": "possible"}


# ---- no pack-specific Python

def test_renaming_every_framework_token_of_the_pack_gives_the_same_facts():
    rename = {"fastapi": "flasky", "include_router": "register_blueprint", "APIRouter": "Blueprint", "FastAPI": "Flask"}

    def sub(x):
        if isinstance(x, str):
            for a, b in rename.items():
                x = x.replace(a, b)
            return x
        if isinstance(x, list):
            return [sub(i) for i in x]
        if isinstance(x, dict):
            return {sub(k): sub(v) for k, v in x.items()}
        return x

    flasky = sub(copy.deepcopy(FASTAPI))
    recs = sub(app_records())
    recs += sub([call("ar", "mk", "APIRouter")])
    base = facts(app_records() + [call("ar", "mk", "APIRouter")], FASTAPI)
    got = facts(recs, flasky)
    strip = lambda fs: [{k: v for k, v in f.items() if k != "packId"} for f in fs]
    assert strip(got.providers) == strip(base.providers) and len(got.providers) == 1
    assert got.providers[0]["packId"] == "flasky"


# ---- determinism

def test_output_is_independent_of_record_and_pack_order():
    recs = app_records() + tool_records()
    a = facts(recs, FASTAPI, FASTMCP)
    b = run_packs([list(reversed(recs))], [FASTMCP, FASTAPI])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_an_unsupported_subject_or_emission_is_refused_loudly():
    from ccg.packrt.match import UnsupportedSubject
    pack = {"packFormat": 1, "id": "z", "rules": [{"id": "r", "match": {"query": {"query": "q"}}, "extract": {}, "emit": [{"provides": {"kind": "http", "key": "x"}}]}]}
    with pytest.raises(UnsupportedSubject):
        facts(app_records(), pack)
    pack2 = {"packFormat": 1, "id": "z", "rules": [{"id": "r", "match": {"callsite": {"callee": "FastAPI"}}, "extract": {}, "emit": [{"launches": {"target": "x"}}]}]}
    with pytest.raises(UnsupportedSubject):
        facts(app_records(), pack2)


def _fastapi_with(**over):
    pack = copy.deepcopy(FASTAPI)
    pack.update(over)
    return pack


def test_a_pack_declaring_two_mounts_is_refused_not_composed_from_the_first_only():
    second = copy.deepcopy(FASTAPI["mounts"][0]) | {"id": "other"}
    with pytest.raises(UnsupportedSubject, match="mounts"):
        facts(app_records(), _fastapi_with(mounts=[FASTAPI["mounts"][0], second]))
    facts(app_records(), FASTAPI)                                      # control: the one-declaration shipped pack still runs


def test_a_class_decorator_mount_is_refused_not_a_key_error():
    cls = {"id": "cls", "kind": "class-decorator", "decorator": {"form": "decorator", "name": "x"}, "prefix": {"arg": "a0"}}
    with pytest.raises(UnsupportedSubject, match="class-decorator"):
        facts(app_records(), _fastapi_with(mounts=[cls]))


def test_a_non_empty_embedded_declaration_is_refused_not_ignored_and_an_empty_one_is_not():
    emb = {"call": {"callee": "execute"}, "arg": "a0", "language": "sql"}
    with pytest.raises(UnsupportedSubject, match="embedded"):
        facts(app_records(), _fastapi_with(embedded=[emb]))
    assert facts(app_records(), _fastapi_with(embedded=[])).providers == facts(app_records(), FASTAPI).providers


def test_the_key_form_of_an_emission_is_carried_on_the_fact_and_defaults_to_literal():
    (lit,) = facts(one_call(), callsite_pack(http_rule("a"))).providers
    (glob,) = facts(one_call(), callsite_pack(http_rule("a", keyForm="glob"))).providers
    assert (lit["keyForm"], glob["keyForm"]) == ("literal", "glob")


def test_a_mark_emission_is_a_mark_item():
    pack = {"packFormat": 1, "id": "m", "rules": [{"id": "r", "match": {"callsite": {"callee": "FastAPI"}}, "extract": {},
                                                     "emit": [{"mark": {"as": "source", "class": "net"}}]}]}
    (mk,) = facts(app_records(), pack).marks
    assert (mk["as"], mk["class"], mk["anchorId"], mk["siteId"]) == ("source", "net", "mk", "fa")


# ---- cases the mutation pass found unpinned

def callsite_pack(*rules):
    return {"packFormat": 1, "id": "hand", "rules": list(rules)}


def http_rule(rid, **emit):
    return {"id": rid, "match": {"callsite": {"callee": "g"}}, "extract": {}, "emit": [{"provides": {"kind": "http", "key": "GET /x", **emit}}]}


def one_call():
    return [file_("f1", "a.py"), sym("f", "f", fid="f1", path="a.py"), call("c1", "f", "g")]


def test_an_emission_may_state_its_own_status_and_reason():
    (p,) = facts(one_call(), callsite_pack(http_rule("a", status="possible", reason="partial-key"))).providers
    assert (p["status"], p["reason"]) == ("possible", "partial-key")


def test_duplicates_collapse_and_a_confirmed_copy_wins_whatever_the_rule_order():
    weak, strong = http_rule("a", status="possible", reason="partial-key"), http_rule("b")
    for rules in ([weak, strong], [strong, weak]):
        (p,) = facts(one_call(), callsite_pack(*rules)).providers
        assert (p["status"], p["reason"]) == ("confirmed", "key-match")


def test_an_unresolved_alternative_beside_a_resolved_one_adds_no_key_but_makes_the_consumer_possible():
    recs = fetch_chain()
    recs[-1] = vs("c1", "a0", "/api/x", {"unknown": "point-derived"})
    (c,) = facts(recs, FETCH).consumers
    assert (c["key"], c["status"], c["reason"]) == ("GET /api/x", "possible", "partial-key")


def test_a_second_argument_whose_method_does_not_resolve_is_not_a_bare_call():
    """`fetch(url, init)` with an unresolved `init.method` is `*`, not the bare-call `GET` (value-set route)."""
    recs = fetch_chain(init=True)
    recs[-1] = vs("fe", "a1.method", {"unknown": "point-derived"})
    assert [c["key"] for c in facts(recs, FETCH).consumers] == ["* /api/x"]


def test_a_second_argument_fed_by_data_is_not_a_bare_call_either():
    recs = fetch_chain() + [param("p1", "get", 1, "init"), data("fe", "p1", "a1", "p1")]
    assert [c["key"] for c in facts(recs, FETCH).consumers] == ["* /api/x"]       # data route: no value set, still has a second argument


def test_only_the_call_result_feeding_the_child_argument_mounts():
    other_arg = [r for r in app_records() if r["id"] != "d-inc-a0-bc-r"] + [data("inc", "bc", "a1", "r")]
    (p,) = facts(other_arg, FASTAPI).providers
    assert p["status"] == "possible" and p["evidence"]["mounts"][0]["how"] == "unmounted-router"
    field_of_result = [r for r in app_records() if r["id"] != "d-inc-a0-bc-r"] + [data("inc", "bc", "a0", "r.router")]
    (q,) = facts(field_of_result, FASTAPI).providers
    assert q["status"] == "possible"                                               # a field of the result is not the factory's router


def test_a_sort_removed_from_both_candidates_and_output_would_be_seen_by_the_order_test():
    """The two sorts back each other up, so this pins the pair: records in reverse order give byte-identical output."""
    recs = app_records(prefix="/a") + [
        sym("h0", "alpha", fid="f1", path="svc/app.py", parent="build"), dec("d0", "h0", "router.get", ("a0", "/alpha")),
        sym("h9", "omega", fid="f1", path="svc/app.py", parent="build"), dec("d9", "h9", "router.get", ("a0", "/omega"))]
    recs += fetch_chain()[1:] + [call("c0", "caller", "get", line=29, resolution="resolved", targets=("get",)), vs("c0", "a0", "/api/y")]
    got = facts(recs, FASTAPI, FETCH)
    assert len(got.providers) == 3 and len(got.consumers) == 2
    assert json.dumps(got, sort_keys=True) == json.dumps(run_packs([list(reversed(recs))], [FETCH, FASTAPI]), sort_keys=True)
    assert [p["key"] for p in got.providers] == sorted(p["key"] for p in got.providers)    # the order itself is by key, not by arrival
