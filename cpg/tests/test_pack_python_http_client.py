"""P1-49b: the `python-http-client` consumer pack (`cpg/packs/python-http-client/`), as a whole.

`test_p1_49a_pack_python_http_client.py` (qa-engineer, red first) pins each shape; this file is the pack's behavioural contract:

* `behaviour_failures(pack)` runs the validator, the positive cases, the base-URL binding and the negatives (`negatives.yaml`) in
  one function that returns what a pack gets wrong. The leaf-mutation registration in `test_rule_pack_leaf_mutation.py` mutates
  every scalar leaf of the pack file and requires this function to fail for each (the pack has no block in the rule-pack spec);
* the rejected alternatives of the plan (a value test at the seed, the import gate on the rules instead of the seeds) are run
  here as mutants of the shipped pack and must be caught: deleting the chosen mechanism only proves the code runs.

Plan: `docs/plans/connected-code-graph-p1-linking.md` P1-49 "Pack", "Untyped verbs (decided)".
"""
from __future__ import annotations

import copy

import pytest
import yaml

from packrt_helpers import PACKS, call, data, file_, imp, pack, param, sym, vs
from ccg.packrt import run_packs
from ccg.rulepack import PackError, load_pack, pack_errors, validate_pack
from linker_bind_helpers import make_binder, pack_facts, port_fact, unit_bundle, url_lifted
from linker_helpers import cons, prov, sym_in
from test_linker_bind import PORTS_8200, consumer as bind_consumer

PID = "python-http-client"
NEGATIVES = PACKS / PID / "negatives.yaml"
UNK = {"unknown": "point-derived"}
VERBS = ("get", "post", "put", "patch", "delete")
LIBS = ("requests", "httpx")
REQUEST = "urllib.py:<module>.<member>(request).Request.__init__"
URLOPEN = "urllib.py:<module>.<member>(request).urlopen"


def module(*imports):
    return [file_("f1", "svc/client.py")] + [imp("f1", m) for m in imports] + [sym("fn", "fetch_all", fid="f1", path="svc/client.py")]


def untyped(cid, verb, *, owner="fn", line=10):
    return call(cid, owner, verb, line=line, resolution="unresolved")


def external(cid, name, target, *, owner="fn", line=10):
    return call(cid, owner, name, line=line, resolution="external", external=target)


def own_site_wrapper(site, *imports, seed_slot):
    """``def w(path): <site>(path)`` with no call edge reaching ``w``: the wrapper's own site is the only candidate consumer."""
    return module(*imports) + [sym("w", "w", fid="f1", path="svc/client.py", start=30), param("p0", "w", 0, "path"),
                              site, data(site["id"], "p0", "a0", "p0"), vs(site["id"], "a0", seed_slot)]


def two_files(wrapper_imports, caller_imports):
    """``w`` (a wrapper over an untyped ``get(path)``) in ``svc/w.py`` and its type-backed caller ``m`` in ``svc/m.py``."""
    return ([file_("f1", "svc/w.py")] + [imp("f1", m) for m in wrapper_imports]
            + [file_("f2", "svc/m.py")] + [imp("f2", m) for m in caller_imports]
            + [sym("w", "w", fid="f1", path="svc/w.py", start=30), param("p0", "w", 0, "path"),
               untyped("rq", "get", owner="w"), data("rq", "p0", "a0", "p0"), vs("rq", "a0", UNK),
               sym("m", "m", fid="f2", path="svc/m.py", start=50),
               call("cs", "m", "w", line=51, resolution="resolved", targets=("w",)), vs("cs", "a0", "/items")])


# ---------------------------------------------------------------------------------------------------------------------
# negatives.yaml
# ---------------------------------------------------------------------------------------------------------------------

def negative_records(case: dict) -> list[dict]:
    recs = module(*case["imports"])
    if case["callee"] is not None:
        recs.append(call("c1", "fn", case["callee"], resolution=case["resolution"], external=case.get("external")))
        if case["a0"] is not None:
            recs.append(vs("c1", "a0", case["a0"]))
    return recs


def negatives_failures(the_pack: dict) -> list[str]:
    bad = []
    for lifting_id, cases in yaml.safe_load(NEGATIVES.read_text())["negatives"].items():
        for name, case in cases.items():
            got = run_packs([negative_records(case)], [the_pack]).consumers
            if got:
                bad.append(f"{lifting_id}/{name}: {[(c['key'], c['ruleId']) for c in got]}")
    return bad


def test_negatives_file_names_only_liftings_of_the_pack_and_every_lifting_with_a_negative_exists():
    named = set(yaml.safe_load(NEGATIVES.read_text())["negatives"])
    assert named <= {lf["id"] for lf in pack(PID)["liftings"]}


# ---------------------------------------------------------------------------------------------------------------------
# the whole contract as one function
# ---------------------------------------------------------------------------------------------------------------------

def bind_failures(the_pack: dict) -> list[str]:
    """`httpx.Client(base_url=...)` binds an http consumer of the SAME file to the unit serving that port, and nothing else does."""
    def bound(*, same_file=True, kind="http", callee="Client", target="httpx.py:<module>.Client.__init__", arg="kw.base_url"):
        svc = unit_bundle("svc", records=[*sym_in("P", "svc"), *PORTS_8200])
        web = unit_bundle("web", repo="fx-web", records=[
            *sym_in("C", "web", path="web/c.py"), *sym_in("C2", "web", path="web/o.py"),
            external("cl", callee, target, owner="C" if same_file else "C2"), vs("cl", arg, "http://localhost:8200")])
        facts = pack_facts(providers=[prov("P", kind, "GET /api/x")], consumers=[cons("C", kind, "GET /api/x", "s1")],
                           lifted=[url_lifted("s1", "C", "/api/x", pack=PID)], launch=[port_fact("svc", 8200, "env-default", ["L3", "L4"])])
        b = make_binder(facts, [svc, web], packs=[the_pack]).bind(bind_consumer(kind=kind))
        return None if b is None else b.units

    bad = []
    for label, got, want in (
            ("same file", bound(), ("svc",)), ("other file", bound(same_file=False), None),
            ("not a Client", bound(callee="Session", target="httpx.py:<module>.Session.__init__"), None),
            ("other keyword", bound(arg="kw.timeout"), None), ("other library", bound(target="other.py:<module>.Client.__init__"), None),
            ("not http", bound(kind="mcp_tool"), None)):
        if got != want:
            bad.append(f"bind {label}: {got} != {want}")
    return bad


def behaviour_failures(the_pack: dict) -> list[str]:
    """What ``the_pack`` gets wrong, as short strings (empty for the shipped pack)."""
    errs = pack_errors(the_pack, dirname=PID)
    if errs:
        return [f"invalid: {errs[0]}"]
    bad: list[str] = []

    def got(recs):
        return [(c["key"], c["status"], c["ruleId"]) for c in run_packs([recs], [the_pack]).consumers]

    cases: list[tuple[str, list[dict], list[tuple]]] = []
    for verb in VERBS:
        for lib in LIBS:
            cases.append((f"{lib}.{verb} literal", module(lib) + [untyped("c1", verb), vs("c1", "a0", "http://svc:8000/items")],
                          [(f"{verb.upper()} /items", "confirmed", verb)]))
            cases.append((f"{lib}.{verb} own-site wrapper", own_site_wrapper(untyped("rq", verb, owner="w"), lib,
                                                                           seed_slot={"template": [{"const": "/items/"}, UNK]}),
                          [(f"{verb.upper()} /items/{{?}}", "possible", verb)]))
            for other in [x for x in LIBS if x != lib]:      # the import gate: the other library alone is enough, the first prefix alone too
                cases.append((f"{other} alone gates {verb}", module(other) + [untyped("c1", verb), vs("c1", "a0", "/x")],
                              [(f"{verb.upper()} /x", "confirmed", verb)]))
        cases.append((f"{verb} no import", module("os") + [untyped("c1", verb), vs("c1", "a0", "/x")], []))
        cases.append((f"{verb} non-url", module("requests") + [untyped("c1", verb), vs("c1", "a0", "timeout")], []))
        cases.append((f"{verb} typed non-http", module("requests") + [external("c1", verb, "__builtin.dict." + verb), vs("c1", "a0", "/x")], []))
    cases += [("seed file has no import, caller has", two_files((), ("requests",)), []),
              ("seed file imports, caller has none", two_files(("requests",), ()), [("GET /items", "confirmed", "get")])]
    ul = lambda *body: module("urllib.request") + list(body)                                   # noqa: E731
    cases += [
        ("Request bare", ul(external("r1", "Request", REQUEST), vs("r1", "a0", "http://h/d")), [("GET /d", "confirmed", "request")]),
        ("Request method", ul(external("r1", "Request", REQUEST), vs("r1", "a0", "http://h/d"), vs("r1", "kw.method", "POST")),
         [("POST /d", "confirmed", "request")]),
        ("Request data", ul(external("r1", "Request", REQUEST), vs("r1", "a0", "http://h/d"), vs("r1", "kw.data", UNK)),
         [("* /d", "confirmed", "request")]),
        ("Request method unknown", ul(external("r1", "Request", REQUEST), vs("r1", "a0", "http://h/d"), vs("r1", "kw.method", "TELEPORT")),
         [("* /d", "confirmed", "request")]),
        ("Request own-site wrapper", own_site_wrapper(external("rq", "Request", REQUEST, owner="w"), "urllib.request",
                                                       seed_slot={"template": [{"const": "/items/"}, UNK]}),
         [("GET /items/{?}", "possible", "request")]),
        ("Request no urllib", module("requests") + [external("r1", "Request", REQUEST), vs("r1", "a0", "/d")], []),
        ("urlopen literal", ul(external("u1", "urlopen", URLOPEN), vs("u1", "a0", "http://h/health")), [("GET /health", "confirmed", "urlopen")]),
        ("urlopen own-site wrapper", own_site_wrapper(external("rq", "urlopen", URLOPEN, owner="w"), "urllib.request",
                                                       seed_slot={"template": [{"const": "/items/"}, UNK]}),
         [("GET /items/{?}", "possible", "urlopen")]),
        ("urlopen of a Request object", ul(external("u1", "urlopen", URLOPEN), vs("u1", "a0", UNK)), []),
        ("urlopen of a non-url string", ul(external("u1", "urlopen", URLOPEN), vs("u1", "a0", "payload")), []),
        ("urlopen no urllib", module("requests") + [external("u1", "urlopen", URLOPEN), vs("u1", "a0", "/x")], []),
    ]
    for label, recs, want in cases:
        if got(recs) != want:
            bad.append(f"{label}: {got(recs)} != {want}")
    bad += bind_failures(the_pack)
    bad += [f"negative {n}" for n in negatives_failures(the_pack)]
    return bad


def test_the_shipped_pack_has_no_behaviour_failures():
    assert behaviour_failures(pack(PID)) == []


def test_the_pack_validates_through_validate_pack():
    validate_pack(load_pack(PACKS / PID / "pack.yaml"), dirname=PID)


# ---------------------------------------------------------------------------------------------------------------------
# the plan's rejected alternatives, as mutants of the shipped pack
# ---------------------------------------------------------------------------------------------------------------------

def mutant(change) -> dict:
    p = copy.deepcopy(pack(PID))
    change(p)
    return p


def test_mutant_a_value_test_at_the_seed_is_caught():
    """REJECTED (plan "Untyped verbs (decided)"): seeding a verb only when its argument is URL-like text. A bare parameter
    `path` and `self.base + path` hold no constant at the seed and must be seeded (every wrapper, `LNK-11`)."""
    def seed_value_test(p):
        for lf in p["liftings"]:
            if lf["id"] in {f"py-{v}" for v in VERBS}:
                lf["seeds"][0]["call"]["args"] = [{"arg": "a0", "value": {"prefix": "/"}}]
    assert behaviour_failures(mutant(seed_value_test)) != []


def test_mutant_the_import_gate_on_the_rules_instead_of_the_seeds_is_caught():
    """REJECTED (plan Format additions): a rule's `scope` names the LEAF's file, which for a lifted fact is the caller's."""
    def gate_on_rules(p):
        for lf in p["liftings"]:
            for s in lf["seeds"]:
                scope = s.pop("scope")
                for r in p["rules"]:
                    if r["match"]["lifted"]["lifting"] == lf["id"]:
                        r["scope"] = scope
    assert behaviour_failures(mutant(gate_on_rules)) != []
