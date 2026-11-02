"""S12 (P0-11b) tests for ``assemble.py``: keyNorm, role lifting, providers, matching, links.

Unit tests run on hand-made mini-bundles (``Mini``); the last section runs SL-1 and SL-2 on the real S11
bundles (``cpg/.ccg-work/vertical-slice/run1``, gitignored) and is skipped when they are absent.
Run: ``cd cpg && .venv/bin/python -m pytest spikes/vertical-slice/tests``.
"""
import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # cpg/

import assemble as A  # noqa: E402
from ccg.bundle import read_bundle  # noqa: E402
from ccg.ids import link_id  # noqa: E402

# ---------------------------------------------------------------------------------------------
# keyNorm: EP2 §2 K1-K22, K30-K33, K42-K48 in provider form, through the whole writer pipeline.
# ---------------------------------------------------------------------------------------------

ORACLE = [
    ("K1", "GET /shop/api/health", "GET /shop/api/health"),
    ("K2", "POST /hooks/x7k2q9w4m1p8z3v6", "POST /hooks/{#}"),
    ("K3", "POST /hooks/a1b2c3d4e5f6g7h8", "POST /hooks/{#}"),
    ("K4", "POST /hooks/{}", "POST /hooks/{}"),
    ("K5", "POST /notify/x7k2-q9w4-m1p8-z3v6", "POST /notify/{#}"),
    ("K6", "POST /notify/x7k2+q9w4+m1p8+z3v6", "POST /notify/{#}"),
    ("K7", "GET /builds/3f2a9c1e5b7d", "GET /builds/{#}"),
    ("K8", "DELETE /items/550e8400-e29b-41d4-a716-446655440000", "DELETE /items/{#}"),
    ("K9", "POST /cb/sk-abcdefghijklmnopqrs", "POST /cb/{#}"),
    ("K10", "POST /services/T024BE7LD/B01ABCDEFGH/abcdefghijklmnopqrstuvwx",
     "POST /services/T024BE7LD/B01ABCDEFGH/{#}"),
    ("K11", "GET /t/3f2a9c1e5b7d/x7k2q9w4m1p8z3v6/info", "GET /t/{#}/{#}/info"),
    ("K12", "post hooks//x7k2q9w4m1p8z3v6/", "POST /hooks/{#}/"),
    ("K13", "GET /hooks/x7k2q9w4%6d1p8z3v6", "GET /hooks/{#}"),
    ("K14", "GET /a/x7k2q9w4%2fm1p8z3v6", "GET /a/x7k2q9w4%2Fm1p8z3v6"),
    ("K15", "GET /x/{x7k2q9w4m1p8z3v6}", "GET /x/{}"),
    ("K16", "GET /static/{p:path}", "GET /static/{**}"),
    ("K17", "GET /ntfy/kxqzmwvptryhbnjdlc", "GET /ntfy/kxqzmwvptryhbnjdlc"),
    ("K18", "GET /wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "GET /wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    ("K19", "GET /release-2024-notes/s3-bucket-v2-us-east-1", "GET /release-2024-notes/s3-bucket-v2-us-east-1"),
    ("K20", "POST /hooks/{#}", "POST /hooks/{#}"),
    ("K21", "GET /x/{?}", "GET /x/{?}"),
    ("K22", "GET /x7k2q9w4m1p8z3v6", "GET /{#}"),
    ("K30", "GET /f/{x7k2q9w4m1p8z3v6}.json", "GET /f/{#}"),
    ("K31", "GET /a/x7k2q9w4m1p8z3v6/../b", "GET /a/b"),
    ("K32", "GET /a/./x7k2q9w4m1p8z3v6", "GET /a/{#}"),
    ("K33", "POST /hooks/x7k2q9w4m1p8", "POST /hooks/x7k2q9w4m1p8"),
    ("K42", "GET /a/%2E%2E/b", "GET /b"),
    ("K43", "GET /a/b/..", "GET /a/"),
    ("K44", "GET /a/b/.", "GET /a/b/"),
    ("K45", "GET /a/b/%2E%2E", "GET /a/"),
    ("K46", "GET /a/..", "GET /"),
    ("K47", "GET /..", "GET /"),
    ("K48", "GET /a/%2%45%2%45/b", "GET /a/%2E%2E/b"),
]


@pytest.mark.parametrize("rid,given,expected", ORACLE, ids=[r[0] for r in ORACLE])
def test_oracle_provider_form(rid, given, expected):
    assert A.normalize_key(given, "provider") == expected


@pytest.mark.parametrize("rid,given,expected", [r for r in ORACLE if r[0] != "K48"], ids=lambda x: x if len(x) < 5 else "")
def test_oracle_outputs_are_fixed_points(rid, given, expected):
    assert A.normalize_key(expected, "provider") == expected


def test_k48_is_not_a_fixed_point_a_second_pass_gives_b():
    assert A.normalize_key("GET /a/%2E%2E/b", "provider") == "GET /b"


def test_percent_decoding_comes_before_dot_resolution_rfc3986_order():
    # decode first: the encoded dot segment is resolved; the reverse order would leave %2E%2E in place
    assert A.normalize_key("GET /a/b/%2e%2e/c") == "GET /a/c"
    assert A.normalize_key("GET /a/%7Euser/%2f") == "GET /a/~user/%2F"


def test_consumer_form_uses_question_placeholder_k39():
    assert A.normalize_key("POST /services/T024BE7LD/B01ABCDEFGH/abcdefghijklmnopqrstuvwx", "consumer") == \
        "POST /services/T024BE7LD/B01ABCDEFGH/{?}"
    assert A.normalize_key("GET /shop/api/health", "consumer") == "GET /shop/api/health"


def test_bare_path_has_any_method_verb():
    assert A.normalize_key("/x") == "* /x"
    assert A.normalize_key("") == "* /"


@pytest.mark.parametrize("raw,expected", [
    ("/items/:id", "/items/{}"), ("/items/<id>", "/items/{}"), ("/items/<int:id>", "/items/{}"),
    ("/items/[id]", "/items/{}"), ("/items/{id:int}", "/items/{}"), ("/f/[...p]", "/f/{**}"),
    ("/f/{*p}", "/f/{**}"), ("/f/*", "/f/{**}"), ("/f/<path:p>", "/f/{**}"),
])
def test_step7_parameter_syntaxes(raw, expected):
    assert A.normalize_key("GET " + raw) == "GET " + expected


@pytest.mark.parametrize("url,path", [
    ("https://user:pw@host:8000/a/b?x=1#f", "/a/b"), ("http://host", ""), ("http://host?x=1", ""),
    ("/a/b?x=1", "/a/b"), ("/a/b#frag", "/a/b"), ("a/b", "a/b"),
])
def test_step2_path_only(url, path):
    assert A.url_path_of(url) == path


# ---------------------------------------------------------------------------------------------
# Mini-bundle builder.
# ---------------------------------------------------------------------------------------------


class Mini:
    """Hand-made records. Ids are readable strings; the assembler does not validate them as hashes."""

    _serial = itertools.count(1)  # ids are unique across every Mini, as hashes are across bundles

    def __init__(self):
        self.r: list[dict] = []

    def _id(self, kind):
        return f"{kind}{next(Mini._serial):04d}"

    def sym(self, qname, parent=None, line=1):
        sid = f"sym:{qname}"
        rec = {"type": "symbol", "id": sid, "qname": qname, "path": qname.split("::")[0], "lineStart": line,
               "lineEnd": line + 5, "kind": "function", "name": qname.split("::")[1].split(".")[-1]}
        if parent:
            rec["parentId"] = parent
        self.r.append(rec)
        return sid

    def param(self, sym, idx, line=1):
        pid = self._id("param")
        self.r.append({"type": "param", "id": pid, "symbolId": sym, "index": idx, "name": f"p{idx}", "line": line})
        return pid

    def ret(self, sym, line=1):
        rid = self._id("ret")
        self.r.append({"type": "ret", "id": rid, "symbolId": sym, "ordinal": 0, "line": line})
        return rid

    def cond(self, sym, code, line=1):
        cid = self._id("cond")
        self.r.append({"type": "cond", "id": cid, "symbolId": sym, "ordinal": 0, "line": line, "code": code})
        return cid

    def ctrl(self, point, cond):
        self.r.append({"type": "controlledBy", "id": self._id("ctl"), "point": point, "cond": cond})

    def call(self, sym, callee, line, targets=(), **vs):
        cid = self._id("cs")
        self.r.append({"type": "callsite", "id": cid, "symbolId": sym, "calleeName": callee, "line": line,
                       "ordinal": 0, "callKind": "call", "resolution": "resolved" if targets else "external",
                       "targets": [{"symbolId": t, "qname": t} for t in targets]})
        for arg, values in vs.items():
            self.vs(cid, arg.replace("__", "."), values)
        return cid

    def vs(self, cs, arg, values):
        self.r.append({"type": "valueSet", "id": self._id("vs"), "callSiteId": cs, "arg": arg, "values": values})

    def data(self, use, def_, to, frm):
        self.r.append({"type": "data", "id": self._id("data"), "use": use, "def": def_, "toPath": to,
                       "fromPath": frm, "status": "confirmed", "reason": "intra-flow"})

    def const(self, qname, values):
        self.r.append({"type": "constant", "id": self._id("const"), "qname": qname, "values": values, "line": 1})

    def dec(self, target, name, path, line=1):
        self.r.append({"type": "decorator", "id": self._id("dec"), "targetId": target, "ordinal": 0,
                       "form": "decorator", "name": name, "line": line,
                       "args": [{"arg": "a0", "values": [{"const": path}]}]})


def const(v):
    return [{"const": v}]


def tmpl(*parts):
    return [{"template": [{"const": p} if isinstance(p, str) and not p.startswith("ref:") else {"ref": p[4:]}
                          for p in parts]}]


EP = "web/src/api/endpoints.ts"
CL = "web/src/api/client.ts"
SF = "srv/storefront_api.py"
APP = "srv/app.py"
DEV = "srv/api.py"


def web_bundle(prefix="/shop/api", route="/health", method="GET", extra_site=False):
    """endpoints.ts getHealth -> client.ts apiFetch(opts) -> buildUrl(path) -> fetch (the AC-1 shapes)."""
    m = Mini()
    get_health = m.sym(f"{EP}::getHealth", line=16)
    api_fetch = m.sym(f"{CL}::apiFetch", line=105)  # a multi-line signature: the parameter is on 107
    build_url = m.sym(f"{CL}::buildUrl", line=71)
    m.const(f"{EP}::API_PREFIX", const(prefix))
    opts = m.param(api_fetch, 0, 107)
    path = m.param(build_url, 0, 73)
    ret = m.ret(build_url, 74)
    m.data(ret, path, "ret", "p0")
    site = m.call(get_health, "apiFetch", 17, [api_fetch],
                  **{"a0__path": tmpl(f"ref:{EP}::API_PREFIX", route), "a0__method": const(method)})
    bu = m.call(api_fetch, "buildUrl", 108, [build_url], a0=[{"unknown": "point-derived"}])
    m.data(bu, opts, "a0", "p0.path")
    fe = m.call(api_fetch, "fetch", 118, [], a0=[{"unknown": "point-derived"}],
                a1__method=[{"unknown": "point-derived"}])
    m.data(fe, bu, "a0", "r")
    m.data(fe, opts, "a1.method", "p0.method")
    if extra_site:
        m.call(get_health, "apiFetch", 20, [api_fetch],
               **{"a0__path": tmpl(f"ref:{EP}::API_PREFIX", route), "a0__method": const(method)})
    return m.r, dict(site=site, get_health=get_health, api_fetch=api_fetch, build_url=build_url, fetch=fe, via=bu)


def server_bundle(prefix="/shop/api", route="/health", cond="shop is not None"):
    """app.py create_app: FastAPI(), the bare app.get, a conditional include of the storefront router."""
    m = Mini()
    create_app = m.sym(f"{APP}::create_app", line=480)
    m.call(create_app, "FastAPI", 481)
    bare = m.sym(f"{APP}::create_app.health", parent=create_app, line=497)
    m.dec(bare, "app.get", "/health", 497)
    store = m.sym(f"{SF}::build_storefront_router", line=900)
    health = m.sym(f"{SF}::build_storefront_router.health", parent=store, line=955)
    m.dec(health, "router.get", route, 954)
    m.const(f"{SF}::API_PREFIX", const(prefix))
    factory_call = m.call(create_app, "build_storefront_router", 522, [store])
    inc = m.call(create_app, "include_router", 521, [], kw__prefix=[{"ref": f"{SF}::API_PREFIX"}])
    m.data(inc, factory_call, "a0", "r")
    if cond:
        c = m.cond(create_app, cond, 516)
        m.ctrl(inc, c)
    dev = m.sym(f"{DEV}::build_router", line=40)
    dev_health = m.sym(f"{DEV}::build_router.health", parent=dev, line=58)
    m.dec(dev_health, "router.get", "/health", 58)
    dev_call = m.call(create_app, "build_router", 486, [dev])
    dev_inc = m.call(create_app, "include_router", 485, [])
    m.data(dev_inc, dev_call, "a0", "r")
    dc = m.cond(create_app, "dev_surface", 484)
    m.ctrl(dev_inc, dc)
    return m.r, dict(health=health, bare=bare, dev_health=dev_health, store=store, create_app=create_app, inc=inc)


def run(*bundles):
    return A.assemble(list(bundles))


# ---------------------------------------------------------------------------------------------
# Consumers: fetch seed, role lifting, values.
# ---------------------------------------------------------------------------------------------


def test_lifted_chain_is_call_then_apifetch_then_buildurl_then_fetch_with_paths():
    web, ids = web_bundle()
    out = run(web)
    [cons] = out["consumers"]
    assert cons["siteId"] == ids["site"] and cons["keys"] == ["GET /shop/api/health"]
    hops = cons["liftedVia"]
    got = [(h["kind"], h["id"], h["urlPath"], h["verbPath"], h["line"]) for h in hops]
    assert got == [
        ("callsite", ids["site"], "a0.path", "a0.method", 17),
        ("symbol", ids["api_fetch"], "p0.path", "p0.method", 107),  # the parameter's line, not the symbol's
        ("symbol", ids["build_url"], "p0", None, 73),
        ("callsite", ids["fetch"], "a0", "a1.method", 118),
    ]
    assert [h["ord"] for h in hops] == [0, 1, 2, 3]
    assert (hops[1]["lineStart"], hops[2]["lineStart"]) == (105, 71)
    assert (hops[1]["viaSiteId"], hops[1]["viaLine"]) == (ids["via"], 108)  # apiFetch hands it on at its buildUrl call
    assert "viaSiteId" not in hops[2] and "viaSiteId" not in hops[0]


def test_value_flows_through_the_return_value_only_for_the_data_the_callee_returns():
    web, ids = web_bundle()
    # break buildUrl's ret <- p0 flow: the url no longer reaches apiFetch's parameter
    web = [r for r in web if not (r["type"] == "data" and r["toPath"] == "ret")]
    out = run(web)
    assert out["consumers"] == [] and out["links"] == []


def test_ref_constant_resolved_across_bundles():
    web, _ = web_bundle(prefix="/shop/api")
    web = [r for r in web if r["type"] != "constant"]
    other = Mini()
    other.const(f"{EP}::API_PREFIX", const("/other"))
    [cons] = run(web, other.r)["consumers"]
    assert cons["keys"] == ["GET /other/health"]


def test_unresolved_ref_makes_a_question_segment_not_a_dropped_site():
    web, _ = web_bundle()
    web = [r for r in web if r["type"] != "constant"]
    web = [dict(r, values=[{"template": [{"ref": f"{EP}::API_PREFIX"}, {"const": "/health"}]}])
           if r["type"] == "valueSet" and r["arg"] == "a0.path" else r for r in web]
    [cons] = run(web)["consumers"]
    assert cons["keys"] == ["GET /{?}/health"]  # an unresolved prefix is one unresolved segment (documented limit)


def test_wholly_unresolved_url_is_reported_and_makes_no_link():
    web, ids = web_bundle()
    web = [dict(r, values=[{"unknown": "point-derived"}]) if r["type"] == "valueSet" and r["arg"] == "a0.path" else r
           for r in web]
    out = run(web)
    assert out["links"] == [] and out["consumers"] == []
    assert [u["siteId"] for u in out["unresolvedConsumers"]] == [ids["site"]]


def test_verb_alternatives_give_one_key_each_and_unknown_verb_is_any():
    web, _ = web_bundle()
    for r in web:
        if r["type"] == "valueSet" and r["arg"] == "a0.method":
            r["values"] = [{"const": "get"}, {"const": "POST"}]
    [cons] = run(web)["consumers"]
    assert cons["keys"] == ["GET /shop/api/health", "POST /shop/api/health"]
    for r in web:
        if r["type"] == "valueSet" and r["arg"] == "a0.method":
            r["values"] = [{"unknown": "point-derived"}]
    [cons] = run(web)["consumers"]
    assert cons["keys"] == ["* /shop/api/health"]


def test_bare_fetch_with_literal_url_and_no_init_is_get():
    m = Mini()
    s = m.sym("web/a.ts::load")
    m.call(s, "fetch", 3, [], a0=const("/shop/api/health"))
    [cons] = run(m.r)["consumers"]
    assert cons["keys"] == ["GET /shop/api/health"]
    assert [h["kind"] for h in cons["liftedVia"]] == ["callsite"]


def test_bare_fetch_with_an_init_whose_method_is_unknown_is_any_method():
    m = Mini()
    s = m.sym("web/a.ts::load")
    m.call(s, "fetch", 3, [], a0=const("/x"), a1__method=[{"unknown": "point-derived"}])
    [cons] = run(m.r)["consumers"]
    assert cons["keys"] == ["* /x"]


def test_absolute_url_keeps_only_the_path():
    m = Mini()
    s = m.sym("web/a.ts::load")
    m.call(s, "fetch", 3, [], a0=const("http://localhost:8000/shop/api/health?x=1#f"), a1__method=const("GET"))
    [cons] = run(m.r)["consumers"]
    assert cons["keys"] == ["GET /shop/api/health"]


def test_consumer_literal_failing_the_token_screen_is_masked_with_question():
    m = Mini()
    s = m.sym("web/a.ts::load")
    m.call(s, "fetch", 3, [], a0=const("/hooks/x7k2q9w4m1p8z3v6"), a1__method=const("POST"))
    [cons] = run(m.r)["consumers"]
    assert cons["keys"] == ["POST /hooks/{?}"]


def test_a_callee_with_a_cycle_through_constants_terminates_unresolved():
    m = Mini()
    s = m.sym("web/a.ts::load")
    m.const("web/a.ts::A", [{"ref": "web/a.ts::B"}])
    m.const("web/a.ts::B", [{"ref": "web/a.ts::A"}])
    m.call(s, "fetch", 3, [], a0=[{"ref": "web/a.ts::A"}], a1__method=const("GET"))
    out = run(m.r)
    assert out["consumers"] == [] and len(out["unresolvedConsumers"]) == 1


def test_consumer_site_reached_only_through_a_non_fetch_call_is_not_seeded():
    m = Mini()
    s = m.sym("web/a.ts::load")
    m.call(s, "axios", 3, [], a0=const("/shop/api/health"))
    assert run(m.r)["consumers"] == []


# ---------------------------------------------------------------------------------------------
# Providers: composition and conditions.
# ---------------------------------------------------------------------------------------------


def prov(out, key):
    return [p for p in out["providers"] if p["key"] == key]


def test_storefront_provider_is_prefix_plus_route_with_its_mount_condition_recorded():
    srv, ids = server_bundle()
    out = run(srv)
    [p] = prov(out, "GET /shop/api/health")
    assert p["handlerId"] == ids["health"] and p["status"] == "confirmed"
    [mount] = p["mounts"]
    assert mount["how"] == "mounted"
    assert [c["code"] for c in mount["conditions"]] == ["shop is not None"]
    assert [s["siteId"] for s in mount["sites"]] == [ids["inc"]]


def test_dev_router_provider_has_dev_surface_and_bare_registration_is_the_app():
    srv, ids = server_bundle()
    out = run(srv)
    bare = [p for p in prov(out, "GET /health")]
    assert {p["handlerId"] for p in bare} == {ids["bare"], ids["dev_health"]}
    dev = next(p for p in bare if p["handlerId"] == ids["dev_health"])
    assert [c["code"] for m in dev["mounts"] for c in m["conditions"]] == ["dev_surface"]
    app = next(p for p in bare if p["handlerId"] == ids["bare"])
    assert app["mounts"] == [{"how": "app", "sites": [], "conditions": []}]


def test_counterfactual_prefix_changes_the_composed_key():
    srv, ids = server_bundle(prefix="/x/api")
    out = run(srv)
    assert prov(out, "GET /x/api/health") and not prov(out, "GET /shop/api/health")


def test_nested_include_composes_outermost_first():
    m = Mini()
    root = m.sym("srv/app.py::create_app")
    m.call(root, "FastAPI", 1)
    outer = m.sym("srv/o.py::outer")
    inner = m.sym("srv/i.py::inner", line=50)
    h = m.sym("srv/i.py::inner.h", parent=inner)
    m.dec(h, "router.post", "/leaf")
    f_outer = m.call(root, "outer", 5, [outer])
    inc_outer = m.call(root, "include_router", 6, [], kw__prefix=const("/a"))
    m.data(inc_outer, f_outer, "a0", "r")
    f_inner = m.call(outer, "inner", 7, [inner])
    inc_inner = m.call(outer, "include_router", 8, [], kw__prefix=const("/b"))
    m.data(inc_inner, f_inner, "a0", "r")
    out = run(m.r)
    [p] = prov(out, "POST /a/b/leaf")
    assert [s["siteId"] for s in p["mounts"][0]["sites"]] == [inc_outer, inc_inner]


def test_router_never_included_nor_in_an_app_is_possible_not_confirmed():
    m = Mini()
    f = m.sym("srv/t.py::make")
    h = m.sym("srv/t.py::make.h", parent=f)
    m.dec(h, "router.get", "/orphan")
    [p] = run(m.r)["providers"]
    assert p["key"] == "GET /orphan" and p["status"] == "possible" and p["mounts"][0]["how"] == "unmounted-router"


def test_same_handler_mounted_twice_is_one_provider_with_two_mounts_and_distinct_prefixes_two():
    m = Mini()
    root = m.sym("srv/app.py::create_app")
    m.call(root, "FastAPI", 1)
    fac = m.sym("srv/r.py::fac")
    h = m.sym("srv/r.py::fac.h", parent=fac)
    m.dec(h, "router.get", "/h")
    for line, pre in ((5, "/v1"), (7, "/v2")):
        fc = m.call(root, "fac", line, [fac])
        inc = m.call(root, "include_router", line + 1, [], kw__prefix=const(pre))
        m.data(inc, fc, "a0", "r")
    out = run(m.r)
    assert sorted(p["key"] for p in out["providers"]) == ["GET /v1/h", "GET /v2/h"]


def test_unresolved_prefix_drops_that_alternative_and_says_so():
    m = Mini()
    root = m.sym("srv/app.py::create_app")
    m.call(root, "FastAPI", 1)
    fac = m.sym("srv/r.py::fac")
    h = m.sym("srv/r.py::fac.h", parent=fac)
    m.dec(h, "router.get", "/h")
    fc = m.call(root, "fac", 5, [fac])
    inc = m.call(root, "include_router", 6, [], kw__prefix=[{"unknown": "runtime-built"}])
    m.data(inc, fc, "a0", "r")
    out = run(m.r)
    assert out["providers"] == [] and len(out["droppedProviderAlternatives"]) == 1


def test_non_http_decorators_are_ignored():
    m = Mini()
    f = m.sym("srv/t.py::make")
    h = m.sym("srv/t.py::make.h", parent=f)
    m.dec(h, "server.tool", "/x")
    m.dec(h, "router.fetch_all", "/x")
    assert run(m.r)["providers"] == []


def test_route_decorator_without_a_path_argument_is_ignored():
    m = Mini()
    f = m.sym("srv/t.py::make")
    h = m.sym("srv/t.py::make.h", parent=f)
    m.r.append({"type": "decorator", "id": "d", "targetId": h, "ordinal": 0, "form": "decorator",
                "name": "router.get", "line": 1, "args": []})
    assert run(m.r)["providers"] == []


def test_provider_literal_failing_the_screen_is_masked_with_hash():
    m = Mini()
    root = m.sym("srv/a.py::create_app")
    m.call(root, "FastAPI", 1)
    h = m.sym("srv/a.py::create_app.hook", parent=root)
    m.dec(h, "app.post", "/hooks/x7k2q9w4m1p8z3v6")
    [p] = run(m.r)["providers"]
    assert p["key"] == "POST /hooks/{#}"


# ---------------------------------------------------------------------------------------------
# Matching.
# ---------------------------------------------------------------------------------------------

MATCH = [
    ("GET /a/b", "GET /a/b", ("confirmed", "key-match", "exact")),
    ("GET /a/{}", "GET /a/42", ("confirmed", "key-match", "path-template")),
    ("GET /a/{}", "GET /a/{?}", ("possible", "partial-key", "path-template")),
    ("GET /a/b", "GET /a/{?}", ("possible", "partial-key", "path-template")),
    ("GET /a/{#}", "GET /a/{?}", ("possible", "partial-key", "path-template")),
    ("GET /a/{#}", "GET /a/b", None),
    ("GET /a/{#}", "GET /a/{}", None),
    ("GET /a/{**}", "GET /a/b/c", ("confirmed", "key-match", "path-template")),
    ("GET /a/{**}", "GET /a/b/{?}", ("possible", "partial-key", "path-template")),
    ("GET /a/{**}", "GET /a", None),
    ("GET /a/{}", "GET /a/", None),
    ("GET /a/{}", "GET /a/b/c", None),
    ("GET /a/b/c", "GET /a/b", None),
    ("GET /a", "GET /a/", ("possible", "partial-key", "path-template")),
    ("GET /a/", "GET /a", ("possible", "partial-key", "path-template")),
    ("GET /a", "POST /a", None),
    ("* /a", "POST /a", ("possible", "partial-key", "path-template")),
    ("GET /a", "* /a", ("possible", "partial-key", "path-template")),
    ("* /a", "* /a", ("confirmed", "key-match", "exact")),
    ("GET /a/b", "GET /a/c", None),
    ("GET /", "GET /", ("confirmed", "key-match", "exact")),
]


@pytest.mark.parametrize("prov_key,cons_key,expected", MATCH, ids=[f"{i}" for i in range(len(MATCH))])
def test_match_table(prov_key, cons_key, expected):
    assert A.match_keys(prov_key, cons_key) == expected


# ---------------------------------------------------------------------------------------------
# Links.
# ---------------------------------------------------------------------------------------------


def only_link(out):
    [link] = out["links"]
    return link


def test_sl1_shape_one_confirmed_http_link_keyed_and_chained():
    web, w = web_bundle()
    srv, s = server_bundle()
    out = run(web, srv)
    link = only_link(out)
    assert (link["kind"], link["status"], link["reason"], link["keyNorm"]) == \
        ("http", "confirmed", "key-match", "GET /shop/api/health")
    assert link["provider"]["anchorId"] == s["health"] and link["consumer"]["anchorId"] == w["get_health"]
    assert link["id"] == link_id("http", "GET /shop/api/health", s["health"], w["get_health"])
    assert [h["id"] for h in link["sites"][0]["liftedVia"]] == \
        [w["site"], w["api_fetch"], w["build_url"], w["fetch"]]
    assert out["unmatchedConsumers"] == []


def test_sl2_counterfactual_prefix_unmatches_the_consumer():
    web, _ = web_bundle()
    srv, _ = server_bundle(prefix="/x/api")
    out = run(web, srv)
    assert out["links"] == []
    assert [u["keys"] for u in out["unmatchedConsumers"]] == [["GET /shop/api/health"]]
    assert prov(out, "GET /x/api/health")


def test_only_the_storefront_provider_links_not_the_bare_health_ones():
    web, _ = web_bundle(prefix="")
    srv, s = server_bundle()
    out = run(web, srv)
    assert {l["provider"]["anchorId"] for l in out["links"]} == {s["bare"], s["dev_health"]}
    web, _ = web_bundle()
    assert {l["provider"]["anchorId"] for l in run(web, srv)["links"]} == {s["health"]}


def test_two_sites_in_one_consumer_symbol_make_one_link_with_per_site_chains():
    web, w = web_bundle(extra_site=True)
    srv, _ = server_bundle()
    link = only_link(run(web, srv))
    assert len(link["sites"]) == 2
    assert {s["siteId"] for s in link["sites"]} == {h["siteId"] for s in link["sites"] for h in s["liftedVia"]}
    assert all(len(s["liftedVia"]) == 4 for s in link["sites"])


def test_link_status_is_the_best_over_its_sites():
    web, _ = web_bundle(extra_site=True)
    # the second site's url gets an unresolved tail: partial, the first stays exact
    for r in web:
        if r["type"] == "valueSet" and r["arg"] == "a0.path" and r["callSiteId"] != next(
                x["id"] for x in web if x["type"] == "callsite" and x["line"] == 17):
            r["values"] = [{"template": [{"ref": f"{EP}::API_PREFIX"}, {"const": "/"}, {"unknown": "x"}]}]
    srv, _ = server_bundle()
    link = only_link(run(web, srv))
    assert sorted(s["status"] for s in link["sites"]) == ["confirmed", "possible"]
    assert link["status"] == "confirmed"


def test_masked_consumer_matches_masked_provider_only_as_possible():
    web = Mini()
    s = web.sym("web/a.ts::hook")
    web.call(s, "fetch", 3, [], a0=const("/hooks/x7k2q9w4m1p8z3v6"), a1__method=const("POST"))
    srv = Mini()
    root = srv.sym("srv/a.py::create_app")
    srv.call(root, "FastAPI", 1)
    h = srv.sym("srv/a.py::create_app.hook", parent=root)
    srv.dec(h, "app.post", "/hooks/a1b2c3d4e5f6g7h8")
    link = only_link(run(web.r, srv.r))
    assert (link["keyNorm"], link["status"], link["reason"]) == ("POST /hooks/{#}", "possible", "partial-key")


def test_unmounted_provider_caps_the_link_at_possible():
    web = Mini()
    s = web.sym("web/a.ts::go")
    web.call(s, "fetch", 3, [], a0=const("/orphan"), a1__method=const("GET"))
    srv = Mini()
    f = srv.sym("srv/t.py::make")
    h = srv.sym("srv/t.py::make.h", parent=f)
    srv.dec(h, "router.get", "/orphan")
    link = only_link(run(web.r, srv.r))
    assert (link["status"], link["reason"]) == ("possible", "partial-key")


def test_link_carries_no_candidate_count_each_site_still_records_its_consumer_key():
    web = Mini()
    s = web.sym("web/a.ts::go")
    web.call(s, "fetch", 3, [], a0=const("/x"), a1__method=const("GET"))
    srv = Mini()
    root = srv.sym("srv/a.py::create_app")
    srv.call(root, "FastAPI", 1)
    for n, route in (("a", "/x"), ("b", "/{id}"), ("c", "/other")):
        h = srv.sym(f"srv/a.py::create_app.{n}", parent=root)
        srv.dec(h, "app.get", route)
    out = run(web.r, srv.r)
    assert sorted(l["keyNorm"] for l in out["links"]) == ["GET /x", "GET /{}"]
    assert all("candidateCount" not in l for l in out["links"])  # D6: derived per VIA edge, one home (render helper)
    assert {s["consumerKey"] for l in out["links"] for s in l["sites"]} == {"GET /x"}


def test_whole_object_flow_carries_the_method_path_through_a_wrapper():
    m = Mini()
    wrapper = m.sym("web/w.ts::send")
    p_url, p_init = m.param(wrapper, 0), m.param(wrapper, 1)
    fe = m.call(wrapper, "fetch", 5, [], a0=[{"unknown": "point-derived"}], a1=[{"unknown": "point-derived"}])
    m.data(fe, p_url, "a0", "p0")
    m.data(fe, p_init, "a1", "p1")  # fetch(url, init): the whole init object, not `.method`
    caller = m.sym("web/w.ts::save")
    m.call(caller, "send", 9, [wrapper], a0=const("/x"), a1__method=const("POST"))
    [cons] = run(m.r)["consumers"]
    assert cons["keys"] == ["POST /x"]
    assert [h["verbPath"] for h in cons["liftedVia"]] == ["a1.method", "p1.method", "a1.method"]


def test_assembly_is_deterministic_and_input_order_independent():
    web, _ = web_bundle(extra_site=True)
    srv, _ = server_bundle()
    a = json.dumps(run(web, srv), sort_keys=True)
    b = json.dumps(run(list(reversed(srv)), list(reversed(web))), sort_keys=True)
    assert a == b


def test_an_uncalled_wrapper_around_fetch_is_no_consumer_and_no_unresolved_site():
    m = Mini()
    s = m.sym("web/a.ts::f")
    p = m.param(s, 0)
    c = m.call(s, "fetch", 3, [], a0=[{"unknown": "point-derived"}], a1__method=const("GET"))
    m.data(c, p, "a0", "p0")
    out = run(m.r)
    assert out["consumers"] == [] and out["unresolvedConsumers"] == [] and out["links"] == []


# ---------------------------------------------------------------------------------------------
# Unresolved alternatives cap a site at possible (proposal 3.4, spec3 7 `partial-key`).
# ---------------------------------------------------------------------------------------------


def _alt_case(path_values=None, method_values=None):
    web, w = web_bundle()
    for r in web:
        if r["type"] == "valueSet" and r["callSiteId"] == w["site"]:
            if r["arg"] == "a0.path" and path_values is not None:
                r["values"] = path_values
            if r["arg"] == "a0.method" and method_values is not None:
                r["values"] = method_values
    srv, _ = server_bundle()
    return run(web, srv)


def test_unresolved_path_alternative_caps_the_link_at_possible_and_flags_the_consumer():
    good = {"template": [{"ref": f"{EP}::API_PREFIX"}, {"const": "/health"}]}
    out = _alt_case(path_values=[good, {"unknown": "point-derived"}])
    [cons] = out["consumers"]
    assert cons["keys"] == [HEALTH_KEY] and cons["partialAlternatives"] is True
    link = only_link(out)
    assert (link["status"], link["reason"]) == ("possible", "partial-key")
    assert [(s["status"], s["reason"]) for s in link["sites"]] == [("possible", "partial-key")]


def test_unresolved_verb_alternative_caps_the_link_at_possible():
    out = _alt_case(method_values=[{"const": "GET"}, {"unknown": "point-derived"}])
    [cons] = out["consumers"]
    assert cons["keys"] == [HEALTH_KEY] and cons["partialAlternatives"] is True
    assert (only_link(out)["status"], only_link(out)["reason"]) == ("possible", "partial-key")


def test_fully_resolved_alternatives_leave_the_flag_false_and_the_link_confirmed():
    out = _alt_case()
    assert out["consumers"][0]["partialAlternatives"] is False
    assert only_link(out)["status"] == "confirmed"


HEALTH_KEY = "GET /shop/api/health"


# ---------------------------------------------------------------------------------------------
# Small pins: verb text, NFC, URL schemes, path prefixes, caps, api_route.
# ---------------------------------------------------------------------------------------------


def test_verb_text_outside_the_known_set_is_any_method_not_get():
    m = Mini()
    s = m.sym("web/a.ts::load")
    m.call(s, "fetch", 3, [], a0=const("/x"), a1__method=const("FETCH"))
    assert run(m.r)["consumers"][0]["keys"] == ["* /x"]


def test_keynorm_applies_nfc_before_everything_else():
    assert A.normalize_key("GET /cafe\u0301") == "GET /caf\u00e9"


def test_bad_form_raises_a_clear_value_error():
    with pytest.raises(ValueError, match="form must be"):
        A.normalize_key("GET /x", "both")


@pytest.mark.parametrize("url,path", [("git+ssh://h/x/y", "/x/y"), ("s3.v2://b/k", "/k"), ("x1://h", "")])
def test_step2_accepts_the_full_rfc3986_scheme_grammar(url, path):
    assert A.url_path_of(url) == path


def test_path_prefix_includes_bracketed_keys_and_not_longer_names():
    assert A._is_path_prefix("a1", 'a1["x-y"]') and A._is_path_prefix("a1", "a1.method")
    assert A._is_path_prefix("a1", "a1") and not A._is_path_prefix("a1", "a10")
    c = A.Corpus([[{"type": "data", "id": "d", "use": "u", "def": "p", "toPath": "a1", "fromPath": "p1",
                    "status": "confirmed", "reason": "intra-flow"}]])
    assert [(d["id"], dp) for d, dp in c.flows_into("u", 'a1["x-y"]')] == [("d", 'p1["x-y"]')]


def test_alternative_cap_replaces_the_excess_by_unknown():
    nine = [{"const": str(i)} for i in range(9)]
    c = A.Corpus([[{"type": "constant", "id": "c1", "qname": "q::N1", "values": nine, "line": 1},
                   {"type": "constant", "id": "c2", "qname": "q::N2", "values": nine, "line": 1}]])
    assert len(c.alternatives({"concat": [{"ref": "q::N1"}, {"const": "x"}]})) == 9  # under the cap
    many = c.alternatives({"concat": [{"ref": "q::N1"}, {"ref": "q::N2"}]})  # 81 combinations
    assert len(many) == A.MAX_ALTERNATIVES + 1 and many[-1] == A.UNKNOWN


def _const_chain(n):
    recs = [{"type": "constant", "id": f"c{i}", "qname": f"q::C{i}", "line": 1,
             "values": [{"ref": f"q::C{i + 1}"}] if i < n else [{"const": "end"}]} for i in range(n + 1)]
    return A.Corpus([recs])


def test_ref_chain_of_eight_resolves_and_nine_does_not():
    assert _const_chain(7).alternatives({"ref": "q::C0"}) == ["end"]  # 8 refs
    assert _const_chain(8).alternatives({"ref": "q::C0"}) == [A.UNKNOWN]  # 9 refs


def test_api_route_decorator_is_any_method():
    m = Mini()
    root = m.sym("srv/a.py::create_app")
    m.call(root, "FastAPI", 1)
    h = m.sym("srv/a.py::create_app.any", parent=root)
    m.dec(h, "app.api_route", "/any")
    [p] = run(m.r)["providers"]
    assert p["key"] == "* /any" and p["status"] == "confirmed"


# ---------------------------------------------------------------------------------------------
# The coverage probe: every key of every output record kind, on distinct ids and line numbers.
# ---------------------------------------------------------------------------------------------

W1, W2 = "w/ep.ts", "w/cl.ts"
S1, S2 = "s/app.py", "s/r.py"


def probe_bundles():
    """Two bundles whose every id and line number is distinct, so a field read from the wrong source
    cannot coincide with the right one. Returns ``(web, server, I)`` with the ids the tests need."""
    w = Mini()
    send = w.sym(f"{W2}::send", line=401)
    p_opts = w.param(send, 0, 403)
    fetch = w.call(send, "fetch", 411, [], a0=[{"unknown": "point-derived"}], a1__method=const("GET"))
    w.data(fetch, p_opts, "a0", "p0.path")
    w.data(fetch, p_opts, "a1.method", "p0.method")  # opts.method ?? "GET"

    def caller(name, line, calls):
        sid = w.sym(f"{W1}::{name}", line=line)
        return sid, [w.call(sid, "send", ln, [send], a0__path=path, a0__method=meth) for ln, path, meth in calls]

    save, (s311, s312) = caller("save", 301, [(311, const("/v1/items"), const("POST")),
                                              (312, const("/v1/other"), const("POST"))])
    list2, (s361,) = caller("list2", 351, [(361, [{"const": "/v1/b"}, {"const": "/v1/a"}], const("GET"))])
    loose, (s376,) = caller("loose", 371, [(376, [{"unknown": "point-derived"}], const("GET"))])
    nomatch, (s381,) = caller("nomatch", 378, [(381, const("/zzz/none"), const("GET"))])
    half, (s391,) = caller("half", 386, [(391, [{"const": "/v1/items"}, {"unknown": "point-derived"}], const("POST"))])

    v = Mini()
    app = v.sym(f"{S1}::create_app", line=501)
    v.call(app, "FastAPI", 502)
    h_any = v.sym(f"{S1}::create_app.any", parent=app, line=521)
    v.dec(h_any, "app.api_route", "/any", 520)
    make = v.sym(f"{S2}::make", line=601)
    hs = {}
    for name, line, deco, route in (("items", 621, ("router.post", 620), "/items"),
                                    ("item_id", 631, ("router.post", 630), "/{id}"),
                                    ("a", 641, ("router.get", 640), "/a"), ("b", 651, ("router.get", 650), "/b"),
                                    ("dup", 661, ("router.get", 660), "/dup"),
                                    ("other", 671, ("router.post", 670), "/other")):
        hs[name] = v.sym(f"{S2}::make.{name}", parent=make, line=line)
        v.dec(hs[name], deco[0], route, deco[1])
    fcall = v.call(app, "make", 610, [make])
    inc = v.call(app, "include_router", 611, [], kw__prefix=[{"const": "/v1"}, {"const": "/v1/"}])
    v.data(inc, fcall, "a0", "r")
    cond = v.cond(app, "ready", 609)
    v.ctrl(inc, cond)
    make2 = v.sym(f"{S2}::make2", line=701)
    h_d = v.sym(f"{S2}::make2.d", parent=make2, line=721)
    v.dec(h_d, "router.get", "/d", 720)
    fcall2 = v.call(app, "make2", 710, [make2])
    inc2 = v.call(app, "include_router", 711, [], kw__prefix=[{"unknown": "runtime-built"}])
    v.data(inc2, fcall2, "a0", "r")
    orphan = v.sym(f"{S2}::orphan", line=801)
    h_o = v.sym(f"{S2}::orphan.o", parent=orphan, line=811)
    v.dec(h_o, "router.get", "/orph", 810)
    I = dict(send=send, fetch=fetch, save=save, list2=list2, loose=loose, nomatch=nomatch, half=half,
             s311=s311, s312=s312, s361=s361, s376=s376, s381=s381, s391=s391, app=app, h_any=h_any, hs=hs,
             inc=inc, inc2=inc2, h_d=h_d, h_o=h_o)
    return w.r, v.r, I


def has_exactly(rec, keys, what):
    assert set(rec) == set(keys), (what, "unexpected", set(rec) - set(keys), "missing", set(keys) - set(rec))


@pytest.fixture(scope="module")
def probe():
    web, srv, I = probe_bundles()
    return run(web, srv), I


def test_probe_providers_every_field(probe):
    out, I = probe
    hs = I["hs"]

    def mount(how, sites=(), conds=()):
        return {"how": how, "sites": [{"siteId": i, "path": S1, "line": ln} for i, ln in sites],
                "conditions": [{"siteId": i, "line": ln, "code": c} for i, ln, c in conds]}

    mounted = mount("mounted", [(I["inc"], 611)], [(I["inc"], 609, "ready")])
    exp = [
        ("* /any", I["h_any"], f"{S1}::create_app.any", 520, "app.api_route", "confirmed", [mount("app")]),
        ("GET /orph", I["h_o"], f"{S2}::orphan.o", 810, "router.get", "possible", [mount("unmounted-router")]),
        ("GET /v1/a", hs["a"], f"{S2}::make.a", 640, "router.get", "confirmed", [mounted]),
        ("GET /v1/b", hs["b"], f"{S2}::make.b", 650, "router.get", "confirmed", [mounted]),
        ("GET /v1/dup", hs["dup"], f"{S2}::make.dup", 660, "router.get", "confirmed", [mounted]),
        ("POST /v1/items", hs["items"], f"{S2}::make.items", 620, "router.post", "confirmed", [mounted]),
        ("POST /v1/other", hs["other"], f"{S2}::make.other", 670, "router.post", "confirmed", [mounted]),
        ("POST /v1/{}", hs["item_id"], f"{S2}::make.item_id", 630, "router.post", "confirmed", [mounted]),
    ]
    assert len(out["providers"]) == len(exp)
    for got, (key, hid, qn, line, deco, status, mounts) in zip(out["providers"], exp):
        has_exactly(got, ["handlerId", "qname", "path", "line", "decorator", "key", "mounts", "status", "reason"],
                    "provider")
        assert (got["key"], got["handlerId"], got["qname"], got["path"], got["line"], got["decorator"]) == \
            (key, hid, qn, qn.split("::")[0], line, deco)
        assert (got["status"], got["reason"]) == (status, "key-match" if status == "confirmed" else "partial-key")
        assert got["mounts"] == mounts  # the two `/v1` prefix spellings collapse to one mount
        for m in got["mounts"]:
            has_exactly(m, ["how", "sites", "conditions"], "mount")
            for x in m["sites"]:
                has_exactly(x, ["siteId", "path", "line"], "mount site")
            for x in m["conditions"]:
                has_exactly(x, ["siteId", "line", "code"], "condition")


def test_probe_consumers_unmatched_unresolved_dropped_every_field(probe):
    out, I = probe
    exp = [  # siteId, symbolId, path, line, keys, partialAlternatives
        (I["s311"], I["save"], W1, 311, ["POST /v1/items"], False),
        (I["s312"], I["save"], W1, 312, ["POST /v1/other"], False),
        (I["s361"], I["list2"], W1, 361, ["GET /v1/a", "GET /v1/b"], False),
        (I["s381"], I["nomatch"], W1, 381, ["GET /zzz/none"], False),
        (I["s391"], I["half"], W1, 391, ["POST /v1/items"], True),
    ]
    got = sorted(out["consumers"], key=lambda c: c["line"])
    assert len(got) == len(exp)
    for g, (sid, sym, path, line, keys, partial) in zip(got, exp):
        has_exactly(g, ["siteId", "symbolId", "path", "line", "urlPath", "keys", "partialAlternatives", "liftedVia"],
                    "consumer")
        assert (g["siteId"], g["symbolId"], g["path"], g["line"], g["urlPath"], g["keys"],
                g["partialAlternatives"]) == (sid, sym, path, line, "a0.path", keys, partial)
        assert [h["ord"] for h in g["liftedVia"]] == [0, 1, 2]
    [um] = out["unmatchedConsumers"]
    has_exactly(um, ["siteId", "path", "line", "keys"], "unmatched")
    assert um == {"siteId": I["s381"], "path": W1, "line": 381, "keys": ["GET /zzz/none"]}
    [un] = out["unresolvedConsumers"]
    has_exactly(un, ["siteId", "path", "line"], "unresolved")
    assert un == {"siteId": I["s376"], "path": W1, "line": 376}
    [dr] = out["droppedProviderAlternatives"]
    has_exactly(dr, ["handlerId", "line", "reason"], "dropped")
    assert dr == {"handlerId": I["h_d"], "line": 720, "reason": "unresolved-part"}


def test_probe_links_every_field(probe):
    out, I = probe
    hs = I["hs"]
    by = {(l["consumer"]["anchorId"], l["provider"]["anchorId"]): l for l in out["links"]}
    assert len(by) == len(out["links"]) == 7

    def site(sid, line, key, status, reason, matcher):
        return dict(siteId=sid, line=line, consumerKey=key, status=status, reason=reason, matcher=matcher)

    exp = {  # (consumer, handler): keyNorm, status, reason, matcher, provider line, sites
        (I["save"], hs["items"]): ("POST /v1/items", "confirmed", "key-match", "exact", 620,
                                   [site(I["s311"], 311, "POST /v1/items", "confirmed", "key-match", "exact")]),
        (I["save"], hs["item_id"]): ("POST /v1/{}", "confirmed", "key-match", "path-template", 630,
                                     [site(I["s311"], 311, "POST /v1/items", "confirmed", "key-match", "path-template"),
                                      site(I["s312"], 312, "POST /v1/other", "confirmed", "key-match", "path-template")]),
        (I["save"], hs["other"]): ("POST /v1/other", "confirmed", "key-match", "exact", 670,
                                   [site(I["s312"], 312, "POST /v1/other", "confirmed", "key-match", "exact")]),
        (I["half"], hs["items"]): ("POST /v1/items", "possible", "partial-key", "exact", 620,
                                   [site(I["s391"], 391, "POST /v1/items", "possible", "partial-key", "exact")]),
        (I["half"], hs["item_id"]): ("POST /v1/{}", "possible", "partial-key", "path-template", 630,
                                     [site(I["s391"], 391, "POST /v1/items", "possible", "partial-key", "path-template")]),
        (I["list2"], hs["a"]): ("GET /v1/a", "confirmed", "key-match", "exact", 640,
                                [site(I["s361"], 361, "GET /v1/a", "confirmed", "key-match", "exact")]),
        (I["list2"], hs["b"]): ("GET /v1/b", "confirmed", "key-match", "exact", 650,
                                [site(I["s361"], 361, "GET /v1/b", "confirmed", "key-match", "exact")]),
    }
    assert set(by) == set(exp)
    for k, (key, status, reason, matcher, pline, sites) in exp.items():
        l = by[k]
        has_exactly(l, ["type", "id", "kind", "keyNorm", "provider", "consumer", "sites", "status",
                        "reason", "matcher"], "link")
        has_exactly(l["provider"], ["anchorId", "qname", "path", "line", "mounts"], "link.provider")
        has_exactly(l["consumer"], ["anchorId", "qname", "path"], "link.consumer")
        assert (l["type"], l["kind"], l["id"], l["keyNorm"]) == ("link", "http", link_id("http", key, k[1], k[0]), key)
        assert (l["status"], l["reason"], l["matcher"]) == (status, reason, matcher)
        assert l["provider"]["line"] == pline and l["provider"]["path"] == S2
        assert l["provider"]["qname"].startswith(f"{S2}::make.") and l["provider"]["mounts"]
        assert l["consumer"]["path"] == W1 and l["consumer"]["qname"].startswith(f"{W1}::")
        assert [{kk: vv for kk, vv in s.items() if kk != "liftedVia"} for s in l["sites"]] == sites
        for s in l["sites"]:
            has_exactly(s, ["siteId", "line", "consumerKey", "status", "reason", "matcher", "liftedVia"], "link.site")
            assert [h["ord"] for h in s["liftedVia"]] == [0, 1, 2]
            assert {h["siteId"] for h in s["liftedVia"]} == {s["siteId"]}


def test_probe_lifted_hops_every_field_and_the_first_verb_on_the_chain_wins(probe):
    out, I = probe
    [c] = [c for c in out["consumers"] if c["siteId"] == I["s311"]]
    exp = [
        dict(kind="callsite", id=I["s311"], path=W1, line=311, callee="send", urlPath="a0.path",
             verbPath="a0.method", ord=0),
        dict(kind="symbol", id=I["send"], qname=f"{W2}::send", path=W2, line=403, lineStart=401,
             urlPath="p0.path", verbPath="p0.method", ord=1),
        dict(kind="callsite", id=I["fetch"], path=W2, line=411, callee="fetch", urlPath="a0",
             verbPath="a1.method", ord=2),
    ]
    assert c["liftedVia"] == exp
    # the outer call says POST, the inner `opts.method ?? "GET"` says GET: the consumer-side call wins
    assert c["keys"] == ["POST /v1/items"]


# ---------------------------------------------------------------------------------------------
# SL-1 / SL-2 on the real S11 bundles (skipped where the gitignored bundles are absent).
# ---------------------------------------------------------------------------------------------

RUN1 = HERE.parents[1] / ".ccg-work" / "vertical-slice" / "run1"
FCS = RUN1 / "0fb899742d4d3687495c30357ca16253.bundle.jsonl"
SP = RUN1 / "c2341bca0404a6a3eadab1d72ba5f4db.bundle.jsonl"
PINNED = "defca60ff16534aa12a74e35856de9be98bfc710"
real = pytest.mark.skipif(not (FCS.exists() and SP.exists()), reason="S11 bundles not present (gitignored)")

HEALTH = "GET /shop/api/health"
ENDPOINTS = "salesperson/src/api/endpoints.ts"
CLIENT = "salesperson/src/api/client.ts"
STOREFRONT = "falkor-chat/server/falkorchat/storefront_api.py"


@pytest.fixture(scope="module")
def real_records():
    return read_bundle(FCS), read_bundle(SP)


@pytest.fixture(scope="module")
def real_out(real_records):
    return A.assemble([copy.deepcopy(r) for r in real_records])


def source_line(path, n):
    return subprocess.run(["git", "show", f"{PINNED}:{path}"], capture_output=True, text=True, check=True,
                          cwd=HERE.parents[2]).stdout.splitlines()[n - 1]


@real
def test_sl1_exactly_one_confirmed_link_endpoints_health_to_storefront_handler(real_out):
    health = [l for l in real_out["links"] if l["keyNorm"] == HEALTH]
    assert len(health) == 1
    [link] = health
    assert (link["kind"], link["status"], link["reason"]) == ("http", "confirmed", "key-match")
    assert link["provider"]["path"] == STOREFRONT
    assert link["provider"]["qname"] == f"{STOREFRONT}::build_storefront_router.health"
    assert link["consumer"]["qname"] == f"{ENDPOINTS}::getHealth"
    # no other confirmed link reaches the endpoints.ts health consumer
    assert [l for l in real_out["links"] if l["consumer"]["qname"] == f"{ENDPOINTS}::getHealth"] == [link]


@real
def test_sl1_chain_order_paths_verbs_and_source_lines(real_out):
    [link] = [l for l in real_out["links"] if l["keyNorm"] == HEALTH]
    [site] = link["sites"]
    hops = site["liftedVia"]
    assert [(h["kind"], h["path"], h["urlPath"], h["verbPath"]) for h in hops] == [
        ("callsite", ENDPOINTS, "a0.path", "a0.method"),
        ("symbol", CLIENT, "p0.path", "p0.method"),
        ("symbol", CLIENT, "p0", None),
        ("callsite", CLIENT, "a0", "a1.method"),
    ]
    assert [h.get("qname", h.get("callee")) for h in hops] == [
        "apiFetch", f"{CLIENT}::apiFetch", f"{CLIENT}::buildUrl", "fetch"]
    assert [h["ord"] for h in hops] == [0, 1, 2, 3]
    expect = {0: "apiFetch<HealthResponse>(", 1: "export async function apiFetch", 2: "function buildUrl(",
              3: "await fetch(url"}
    for h in hops:
        assert expect[h["ord"]] in source_line(h["path"], h["line"]), (h["ord"], h["path"], h["line"])
    assert "buildUrl(opts.path" in source_line(CLIENT, hops[1]["viaLine"])


@real
def test_sl2_three_health_providers_with_mounts_and_conditions(real_out):
    h = [p for p in real_out["providers"] if p["key"].endswith("/health")]
    by = {p["qname"].split("falkorchat/")[1]: p for p in h}
    assert set(by) == {"storefront_api.py::build_storefront_router.health", "api.py::build_router.health",
                       "app.py::create_app.health"}
    store = by["storefront_api.py::build_storefront_router.health"]
    assert store["key"] == HEALTH
    assert [c["code"] for m in store["mounts"] for c in m["conditions"]] == ["shop is not None"]
    assert [s["path"] for m in store["mounts"] for s in m["sites"]] == ["falkor-chat/server/falkorchat/app.py"]
    dev = by["api.py::build_router.health"]
    assert dev["key"] == "GET /health"
    assert "dev_surface" in [c["code"] for m in dev["mounts"] for c in m["conditions"]]
    bare = by["app.py::create_app.health"]
    assert bare["key"] == "GET /health" and bare["mounts"][0]["how"] == "app"
    assert {l["provider"]["qname"] for l in real_out["links"] if l["keyNorm"] == HEALTH} == {store["qname"]}


@real
def test_sl2_counterfactual_staged_api_prefix(real_records):
    fcs, sp = copy.deepcopy(real_records[0]), copy.deepcopy(real_records[1])
    [const_rec] = [r for r in fcs if r["type"] == "constant" and r["qname"] == f"{STOREFRONT}::API_PREFIX"]
    assert const_rec["values"] == [{"const": "/shop/api"}]
    const_rec["values"] = [{"const": "/x/api"}]
    out = A.assemble([fcs, sp])
    assert any(p["key"] == "GET /x/api/health" and p["qname"].endswith("build_storefront_router.health")
               for p in out["providers"])
    assert not any(p["key"] == HEALTH for p in out["providers"])
    assert not any(l["keyNorm"] == HEALTH for l in out["links"])
    assert any(HEALTH in u["keys"] for u in out["unmatchedConsumers"])


@real
def test_step_8a_changes_no_key_on_the_real_bundles(real_out):
    keys = [k for c in real_out["consumers"] for k in c["keys"]] + [p["key"] for p in real_out["providers"]
                                                                       if "/shop/api" in p["key"]]
    assert keys and not any("{#}" in k or "{?}" in k for k in keys)


@real
def test_every_endpoints_consumer_resolves_and_links_confirmed(real_out):
    assert len(real_out["consumers"]) == 11 and real_out["unmatchedConsumers"] == []
    assert all(l["status"] == "confirmed" for l in real_out["links"])


@real
def test_assembly_of_real_bundles_is_deterministic(real_records, real_out):
    again = A.assemble([copy.deepcopy(r) for r in reversed(real_records)])
    assert json.dumps(again, sort_keys=True) == json.dumps(real_out, sort_keys=True)
