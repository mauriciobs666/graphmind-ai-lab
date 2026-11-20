"""P1-52: the TypeScript MCP-client consumer pack (`cpg/packs/ts-mcp-client/`), as a whole.

`behaviour_failures(pack)` is the pack's behavioural contract as one function that returns what a pack gets wrong: the validator, the
positive cases, the URL binding (an offline `Binder` over a synthetic port-8200 fact) and the negatives (`negatives.yaml`). The
leaf-mutation registration in `test_rule_pack_leaf_mutation.py` mutates every leaf of the pack file and requires it to report something.

Plan: `docs/plans/connected-code-graph-p1-linking.md` P1-52. The `Link`-level `binding-confirmed` rows (LNK-02/06/12) need the 8200
fact that only P1-53b's `scripts/Dockerfile` creates; they are not tested here.
"""
from __future__ import annotations

import copy

import pytest
import yaml

from packrt_helpers import PACKS, call, data, file_, imp, pack, param, sym, vs
from ccg.packrt import run_packs
from ccg.rulepack import load_pack, pack_errors, validate_pack
from linker_bind_helpers import make_binder, pack_facts, port_fact, unit_bundle
from ccg.linker import link
from linker_helpers import cons, prov, sym_in
from test_linker_bind import PORTS_8200, consumer as bind_consumer

PID = "ts-mcp-client"
NEGATIVES = PACKS / PID / "negatives.yaml"
SDK = "@modelcontextprotocol/sdk/client/index.js"
UNK = {"unknown": "point-derived"}
PATH = "src/mcp/client.ts"


def module(*imports, path=PATH):
    return [file_("f1", path, lang="typescript")] + [imp("f1", m) for m in imports] + [sym("fn", "loadItem", fid="f1", path=path)]


def site(cid="c1", callee="callTool", resolution="unresolved", owner="fn", name="get_item"):
    """``<c>.callTool({ name: <name> })``: the call and the value set of its object-literal ``name`` field."""
    recs = [call(cid, owner, callee, line=10, resolution=resolution)]
    if name is not None:
        recs.append(vs(cid, "a0.name", name))
    return recs


def wrapper(*imports, caller_name="reindex"):
    """``export function callTool(name) { return c.callTool({ name }) }`` in client.ts, called as ``callTool("reindex")`` from Admin.tsx."""
    return module(*imports) + [
        sym("w", "callTool", fid="f1", path=PATH, start=30), param("p0", "w", 0, "name"),
        call("rq", "w", "callTool", line=31, resolution="unresolved"), data("rq", "p0", "a0.name", "p0"), vs("rq", "a0.name", UNK),
        file_("f2", "src/ui/Admin.tsx", lang="typescript"), sym("m", "onReindex", fid="f2", path="src/ui/Admin.tsx", start=4),
        call("cs", "m", "callTool", line=4, resolution="resolved", targets=("w",)), vs("cs", "a0", caller_name)]


def wrapper_alone(*imports):
    """The wrapper of `wrapper` with no call edge reaching it: its own site is the only candidate consumer."""
    return [r for r in wrapper(*imports) if r.get("id") not in ("cs", "vs-cs-a0") and r.get("callSiteId") != "cs" and r.get("symbolId") != "m"
            and r.get("id") != "f2"]


# ---------------------------------------------------------------------------------------------------------------------
# negatives.yaml
# ---------------------------------------------------------------------------------------------------------------------

def negative_records(case: dict) -> list[dict]:
    recs = module(*case["imports"])
    if case["callee"] is not None:
        recs.append(call("c1", "fn", case["callee"], resolution=case["resolution"]))
        if case["name"] is not None:
            recs.append(vs("c1", "a0.name", case["name"]))
    return recs


def negatives_failures(the_pack: dict) -> list[str]:
    bad = []
    for lifting_id, cases in yaml.safe_load(NEGATIVES.read_text())["negatives"].items():
        for name, case in cases.items():
            got = run_packs([negative_records(case)], [the_pack]).consumers
            if got:
                bad.append(f"{lifting_id}/{name}: {[(c['key'], c['ruleId']) for c in got]}")
    return bad


def test_negatives_file_names_only_liftings_of_the_pack():
    assert set(yaml.safe_load(NEGATIVES.read_text())["negatives"]) <= {lf["id"] for lf in pack(PID)["liftings"]}


# ---------------------------------------------------------------------------------------------------------------------
# the URL binding: `scope: unit` (gate review B2)
# ---------------------------------------------------------------------------------------------------------------------

def bound_units(the_pack: dict, *, same_file: bool = False, callee: str = "URL", arg: str = "a0", kind: str = "mcp_tool",
                other_unit_url: bool = False, extra_url: str | None = None):
    """The units a `kind` consumer in `src/ui/Admin.tsx` binds to when `new URL("http://localhost:8200/mcp")` is in another file
    of the same unit `web` (or in unit `other`, `other_unit_url`) and unit `svc` owns loopback 8200."""
    svc = unit_bundle("svc", records=[*sym_in("P", "svc"), *PORTS_8200])
    cpath = PATH if same_file else "src/ui/Admin.tsx"
    url_unit = "other" if other_unit_url else "web"
    extra = [call("u2", "U", "URL"), vs("u2", "a0", extra_url)] if extra_url else []
    web = unit_bundle("web", repo="fx-web", lang="javascript", records=[
        *sym_in("C", "web", path=cpath), *([] if other_unit_url else sym_in("U", "web", path=PATH)),
        *([] if other_unit_url else [call("uc", "U", callee), vs("uc", arg, "http://localhost:8200/mcp"), *extra])])
    bundles = [svc, web]
    launch = [port_fact("svc", 8200, "env-default", ["L3", "L4"])]
    if extra_url:                                   # a second unit that owns the extra URL's port 9999
        bundles.append(unit_bundle("svc2", records=[*sym_in("P2", "svc2"), *PORTS_8200]))
        launch.append(port_fact("svc2", 9999, "env-default", ["L3", "L4"]))
    if other_unit_url:
        bundles.append(unit_bundle("other", repo="fx-web", lang="javascript", records=[
            *sym_in("U", "other", path=PATH), call("uc", "U", callee), vs("uc", arg, "http://localhost:8200/mcp")]))
    facts = pack_facts(providers=[prov("P", kind, "reindex")], consumers=[cons("C", kind, "reindex", "s1")],
                       launch=launch)
    b = make_binder(facts, bundles, packs=[the_pack]).bind(bind_consumer(kind=kind, key="reindex", anchor="C"))
    return None if b is None else b.units


def bind_failures(the_pack: dict) -> list[str]:
    bad = []
    for label, got, want in (
            ("URL in another file of the unit", bound_units(the_pack), ("svc",)),
            ("URL in the consumer's own file", bound_units(the_pack, same_file=True), ("svc",)),
            ("URL in another unit", bound_units(the_pack, other_unit_url=True), None),
            ("not a URL constructor", bound_units(the_pack, callee="Request"), None),
            ("not argument a0", bound_units(the_pack, arg="a1"), None),
            ("not an mcp_tool consumer", bound_units(the_pack, kind="http"), None)):
        if got != want:
            bad.append(f"bind {label}: {got} != {want}")
    return bad


def test_a_url_in_another_file_of_the_unit_binds_an_mcp_tool_consumer_to_the_unit_that_owns_the_port():
    assert bound_units(pack(PID)) == ("svc",)


def test_the_binding_does_not_cross_units():
    assert bound_units(pack(PID), other_unit_url=True) is None


def test_the_file_scope_mutant_leaves_the_consumer_in_another_file_unbound():
    """REJECTED (gate review B2): `scope: file` leaves LNK-12's consumer (`Admin.tsx`) unbound, the URL being in `mcp/client.ts`."""
    p = copy.deepcopy(pack(PID))
    p["urlBindings"][0]["urlSources"][0]["scope"] = "file"
    assert bound_units(p) is None and bound_units(p, same_file=True) == ("svc",)


def test_known_limit_another_literal_url_in_the_unit_adds_an_alternative_to_every_consumer():
    """Pinned (plan P1-52 Limit): any other literal `new URL(...)` of the unit joins the consumer's URL set, so a second URL on a port
    another unit owns adds that unit as an alternative."""
    assert bound_units(pack(PID), extra_url="http://localhost:9999/x") == ("svc", "svc2")


# ---------------------------------------------------------------------------------------------------------------------
# the whole contract as one function
# ---------------------------------------------------------------------------------------------------------------------

def link_failures(the_pack: dict) -> list[str]:
    """The pack's `link` declaration read end to end: tool names match exactly, so a template-shaped provider key is not matched by
    a literal consumer key (read as `path-template`, `a/{}` would link `a/7`)."""
    b = [r for part in (sym_in("P", "svc"), sym_in("C", "web")) for r in part]
    facts = pack_facts(providers=[prov("P", "mcp_tool", "a/{}")], consumers=[cons("C", "mcp_tool", "a/7", "s1")])
    got = [(x.status, x.matcher) for x in link(facts, [b], packs=[the_pack]).links]
    return [] if not any(m == "path-template" for _, m in got) else [f"link: {got}"]


def behaviour_failures(the_pack: dict) -> list[str]:
    """What ``the_pack`` gets wrong, as short strings (empty for the shipped pack)."""
    errs = pack_errors(the_pack, dirname=PID)
    if errs:
        return [f"invalid: {errs[0]}"]
    bad: list[str] = []

    def got(recs):
        return [(c["key"], c["status"], c["ruleId"]) for c in run_packs([recs], [the_pack]).consumers]

    cases = [
        ("literal name", module(SDK) + site(), [("get_item", "confirmed", "ts-call-tool")]),
        ("another sdk import path", module("@modelcontextprotocol/sdk/client/streamableHttp.js") + site(), [("get_item", "confirmed", "ts-call-tool")]),
        ("second import is the sdk", module("react", SDK) + site(), [("get_item", "confirmed", "ts-call-tool")]),
        ("no sdk import", module("react") + site(), []),
        ("a similar package", module("@modelcontextprotocol-x/sdk/client") + site(), []),
        ("another package of the same scope", module("@modelcontextprotocol/inspector") + site(), []),
        ("known limit: a bare `@modelcontextprotocol/sdk` import (no subpath) is not the gate", module("@modelcontextprotocol/sdk") + site(), []),
        ("own-site wrapper nobody calls: its unknown name makes no consumer", wrapper_alone(SDK), []),
        ("another callee", module(SDK) + site(callee="callTools"), []),
        ("a typed `external` callTool is no seed", module(SDK) + [call("c1", "fn", "callTool", resolution="external", external="x.ts:<module>.callTool"),
                                                                  vs("c1", "a0.name", "get_item")], []),
        ("a resolved callTool is no seed", module(SDK) + [sym("m", "m", fid="f1", path=PATH, start=60)]
         + [call("c1", "fn", "callTool", resolution="resolved", targets=("m",)), vs("c1", "a0.name", "x")], []),
        ("name is not the a0 argument", module(SDK) + [call("c1", "fn", "callTool"), vs("c1", "a0", "get_item")], []),
        ("name in the second argument", module(SDK) + [call("c1", "fn", "callTool"), vs("c1", "a1.name", "get_item")], []),
        ("no argument", module(SDK) + site(name=None), []),
        ("wrapper: the caller's literal is the key (through the wrapper's parameter), the wrapper's own site is no second consumer",
         wrapper(SDK), [("reindex", "confirmed", "ts-call-tool")]),
    ]
    for label, recs, want in cases:
        if got(recs) != want:
            bad.append(f"{label}: {got(recs)} != {want}")
    # `ownSite`: a wrapper no call edge reaches is a leaf at its own site (without it the wrapper's parameter makes it a non-leaf)
    lifted = [(x["siteId"], x["leaf"], x["resolved"]) for x in run_packs([wrapper_alone(SDK)], [the_pack]).lifted]
    if lifted != [("rq", True, False)]:
        bad += [f"own-site wrapper lifted fact: {lifted}"]
    bad += link_failures(the_pack)
    bad += bind_failures(the_pack)
    bad += [f"negative {n}" for n in negatives_failures(the_pack)]
    return bad


def test_the_pack_loads_and_passes_the_validator_and_is_a_paper_status_v1_pack():
    p = load_pack(PACKS / PID / "pack.yaml")
    assert pack_errors(p, dirname=PID) == []
    assert (p["id"], p["status"], p["packFormat"]) == (PID, "paper", 1)
    validate_pack(p, dirname=PID)


def test_the_shipped_pack_has_no_behaviour_failures():
    assert behaviour_failures(pack(PID)) == []


# ---------------------------------------------------------------------------------------------------------------------
# valid-alternative mutants: another VALID value of a leaf, which the harness's `Z`/+1/flip never produces
# ---------------------------------------------------------------------------------------------------------------------

def _lifting(p: dict) -> dict:
    return p["liftings"][0]


@pytest.mark.parametrize("label, change", [
    ("binding scope file", lambda p: p["urlBindings"][0]["urlSources"][0].update(scope="file")),
    ("binding scope dropped (the default is file)", lambda p: p["urlBindings"][0]["urlSources"][0].pop("scope")),
    ("binding also for http", lambda p: p["urlBindings"][0].update(kinds=["mcp_tool", "http"])),
    ("link matcher path-template", lambda p: p["link"][0].update(matcher="path-template")),
    ("seed resolution widened to resolved", lambda p: _lifting(p)["seeds"][0]["call"].update(resolution=["unresolved", "resolved"])),
    ("seed resolution widened to external", lambda p: _lifting(p)["seeds"][0]["call"].update(resolution=["unresolved", "external"])),
    ("seed argument a0", lambda p: _lifting(p)["seeds"][0].update(arg="a0")),
    ("import gate removed", lambda p: _lifting(p)["seeds"][0].pop("scope")),
    ("import gate widened to the whole scope", lambda p: _lifting(p)["seeds"][0]["scope"]["imports"][0]["module"].update(glob="@modelcontextprotocol/**")),
    ("import gate as a prefix", lambda p: _lifting(p)["seeds"][0]["scope"]["imports"][0].update(module={"prefix": "@modelcontextprotocol/sdk"})),
    ("ownSite dropped", lambda p: _lifting(p).pop("ownSite")),
    ("urlLike on", lambda p: _lifting(p).update(urlLike=True)),
])
def test_a_valid_alternative_of_a_pack_value_is_caught(label, change):
    p = copy.deepcopy(pack(PID))
    change(p)
    assert behaviour_failures(p) != [], label
