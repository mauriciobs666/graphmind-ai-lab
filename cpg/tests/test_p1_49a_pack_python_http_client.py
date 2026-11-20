"""P1-49a (red first): conformance of the `python-http-client` consumer pack (`cpg/packs/python-http-client/`) and its wrappers.

Plan: `docs/plans/connected-code-graph-p1-linking.md` P1-49 "Pack" and "Wrapper reached through a name-linked caller (decided)".
The pack does not exist yet: every test here fails first at `the_pack()` with "does not exist yet", then, once 49b ships a pack,
on the behaviour. 49b's own `test_pack_python_http_client.py` (negatives file, leaf-mutation harness, `SERVED`) is not written here.

Record shapes are the ones the Python extractor emitted in the stored bundles (`cpg/.ccg-work/bundles/graphmind-ai-lab/`, probed
for this step, 2026-11-19):

* `requests.get(u)`, `httpx.get(u)` and `client.post(p)` on an untyped client: `calleeName` is the bare verb, `resolution`
  `unresolved`, **no `externalTarget`**, and the record carries no receiver, so only the file's imports can gate them;
* `httpx.Client(base_url=...)`: `calleeName` `Client`, external `httpx.py:<module>.Client.__init__`;
* `urllib.request.Request(...)` / `urlopen(...)`: external `urllib.py:<module>.<member>(request).Request.__init__` and
  `...urlopen`.

If 49b's probe finds a different `externalTarget`, change `SHAPES` below, nothing else.
"""
from __future__ import annotations

import pytest

from packrt_helpers import PACKS, call, data, file_, imp, pack, param, sym, vs
from ccg.packrt import run_packs
from ccg.rulepack import pack_errors

PID = "python-http-client"
UNK = {"unknown": "point-derived"}
X = "\x01"

SHAPES = {                                   # (calleeName, resolution, externalTarget)
    "verb": lambda verb: (verb, "unresolved", None),
    "Client": ("Client", "external", "httpx.py:<module>.Client.__init__"),
    "Request": ("Request", "external", "urllib.py:<module>.<member>(request).Request.__init__"),
    "urlopen": ("urlopen", "external", "urllib.py:<module>.<member>(request).urlopen"),
    "dict.get": ("get", "external", "__builtin.dict.get"),
    "environ.get": ("get", "external", "os.py:<module>.<member>(environ).get"),
}


def the_pack() -> dict:
    if not (PACKS / PID / "pack.yaml").is_file():
        pytest.fail(f"cpg/packs/{PID}/pack.yaml does not exist yet (P1-49b ships the pack)", pytrace=False)
    return pack(PID)


def site(cid, shape, *, owner="fn", line=10):
    name, resolution, external = shape
    return call(cid, owner, name, line=line, resolution=resolution, external=external)


def module(*imports):
    """One Python file with ``imports`` (``(module, *names)``) and the function ``fn`` that holds the calls."""
    return [file_("f1", "svc/client.py")] + [imp("f1", m, *names) for m, *names in imports] + [sym("fn", "fetch_all", fid="f1", path="svc/client.py")]


def view(c):
    return (c["kind"], c["key"], c["status"], c["reason"])


def dview(c):
    """A direct site's (kind, key, status): the reason (`key-match` or `lifted-role`) depends on whether 49b seeds by rule or by lifting."""
    return (c["kind"], c["key"], c["status"])


def consumers(recs):
    return run_packs([recs], [the_pack()]).consumers


# ---------------------------------------------------------------------------------------------------------------------
# the pack file
# ---------------------------------------------------------------------------------------------------------------------

def test_the_pack_loads_validates_and_only_consumes_http():
    p = the_pack()
    assert p["id"] == PID and pack_errors(p, dirname=PID) == []
    emitted = [(name, body["kind"]) for r in p["rules"] for e in r["emit"] for name, body in e.items()]
    assert emitted and all(x == ("consumes", "http") for x in emitted)


def test_the_pack_declares_the_httpx_client_base_url_as_a_file_scoped_callsite_url_source_for_http_only():
    p = the_pack()
    sources = [(b["kinds"], s) for b in p.get("urlBindings", ()) for s in b.get("urlSources", ())]
    assert any("http" in kinds and "callsite" in s and s.get("arg") == "kw.base_url" and s.get("scope") == "file"
               for kinds, s in sources), sources


def test_every_lifting_seed_carries_scope_imports_and_no_rule_needs_one():
    """The gate lives on the seed (`seeds[].scope`, plan P1-49 Format additions): a lifted fact's rule `scope` would name the
    leaf's file, which for a lifted caller is not the wrapper's file."""
    p = the_pack()
    assert p["liftings"] and all(s.get("scope", {}).get("imports") for lf in p["liftings"] for s in lf["seeds"])
    assert p["rules"] and not any(r.get("scope") for r in p["rules"])


def two_files(wrapper_imports, caller_imports):
    """``w`` (a wrapper over an untyped ``<verb>(path)``) in ``svc/w.py`` and its type-backed caller ``m`` in ``svc/m.py``."""
    return ([file_("f1", "svc/w.py")] + [imp("f1", m) for m in wrapper_imports]
            + [file_("f2", "svc/m.py")] + [imp("f2", m) for m in caller_imports]
            + [sym("w", "_get", fid="f1", path="svc/w.py", start=30), param("p0", "w", 0, "path", line=31),
               site("rq", SHAPES["verb"]("get"), owner="w", line=32), data("rq", "p0", "a0", "p0"), vs("rq", "a0", UNK),
               sym("m", "run", fid="f2", path="svc/m.py", start=50),
               call("cs", "m", "_get", line=51, resolution="resolved", targets=("w",)), vs("cs", "a0", "/items")])


def test_a_wrapper_in_a_file_without_an_http_import_called_from_a_file_with_one_is_dropped_the_seeds_file_decides():
    assert consumers(two_files(wrapper_imports=(), caller_imports=("requests",))) == []


def test_a_wrapper_in_an_importing_file_called_from_a_non_importing_file_is_kept_at_the_caller():
    (c,) = consumers(two_files(wrapper_imports=("requests",), caller_imports=()))
    assert (c["siteId"], c["anchorId"], c["key"], c["status"]) == ("cs", "m", "GET /items", "confirmed")


# ---------------------------------------------------------------------------------------------------------------------
# requests.<verb>(url) and httpx.<verb>(url)
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lib", ["requests", "httpx"])
@pytest.mark.parametrize("verb", ["get", "post", "put", "patch", "delete"])
def test_a_module_verb_call_with_a_literal_url_is_a_confirmed_http_consumer_with_that_verb(lib, verb):
    recs = module((lib,)) + [site("c1", SHAPES["verb"](verb)), vs("c1", "a0", "http://svc:8000/items")]
    (c,) = consumers(recs)
    assert dview(c) == ("http", f"{verb.upper()} /items", "confirmed")
    assert (c["edge"], c["anchorId"], c["siteId"]) == ("CONSUMES", "fn", "c1")


@pytest.mark.parametrize("imports", [(), (("os",),), (("json",), ("pathlib", "Path"))], ids=["none", "os", "json-pathlib"])
def test_the_import_gate_a_file_importing_no_http_client_makes_no_consumer(imports):
    recs = module(*imports) + [site("c1", SHAPES["verb"]("get")), vs("c1", "a0", "/items")]
    assert consumers(recs) == []


def test_an_unknown_base_in_an_fstring_drops_the_origin_and_is_possible_never_confirmed():
    recs = module(("requests",)) + [site("c1", SHAPES["verb"]("get")), vs("c1", "a0", {"template": [UNK, {"const": "/items"}]})]
    (c,) = consumers(recs)
    assert dview(c) == ("http", "GET /items", "possible") and c["reason"] == "partial-key"


def test_a_constant_base_with_a_known_path_stays_confirmed_and_a_known_base_keeps_the_path_only():
    recs = module(("requests",)) + [site("c1", SHAPES["verb"]("post")), vs("c1", "a0", "http://localhost:8000/job/trigger")]
    (c,) = consumers(recs)
    assert (c["key"], c["status"]) == ("POST /job/trigger", "confirmed")


# ---------------------------------------------------------------------------------------------------------------------
# httpx.Client(base_url=...) then <client>.<verb>(path)
# ---------------------------------------------------------------------------------------------------------------------

def client_module(base, *calls):
    recs = module(("httpx",)) + [site("cl", SHAPES["Client"], line=5), vs("cl", "kw.base_url", base)]
    for i, (verb, path) in enumerate(calls):
        recs += [site(f"c{i}", SHAPES["verb"](verb), line=20 + i), vs(f"c{i}", "a0", path)]
    return recs


def test_a_client_verb_call_keys_on_the_path_alone_and_a_parameter_segment_is_a_placeholder():
    recs = client_module(UNK, ("post", {"template": [{"const": "/threads/"}, UNK, {"const": "/messages"}]}), ("get", "/health"))
    got = {c["siteId"]: dview(c) for c in consumers(recs)}
    assert got == {"c0": ("http", "POST /threads/{?}/messages", "possible"),                    # a path segment is unknown
                   "c1": ("http", "GET /health", "confirmed")}


def test_the_client_base_path_prefix_is_not_joined_to_the_key():
    (c,) = consumers(client_module("http://svc:8000/api/v1", ("get", "/x")))
    assert c["key"] == "GET /x"


def test_the_client_constructor_call_is_not_itself_a_consumer():
    recs = client_module("http://svc:8000")
    assert consumers(recs) == []


# ---------------------------------------------------------------------------------------------------------------------
# urllib.request.Request(url, method=...) and a literal urlopen(str)
# ---------------------------------------------------------------------------------------------------------------------

def urllib_module(*body):
    return module(("urllib.request", "Request", "urlopen")) + list(body)


def test_a_request_takes_its_verb_from_the_method_keyword_and_defaults_to_get():
    post = urllib_module(site("r1", SHAPES["Request"]), vs("r1", "a0", "http://svc:8200/documents"), vs("r1", "kw.method", "POST"))
    bare = urllib_module(site("r1", SHAPES["Request"]), vs("r1", "a0", "http://svc:8200/documents"))
    assert [c["key"] for c in consumers(post)] == ["POST /documents"]
    assert [c["key"] for c in consumers(bare)] == ["GET /documents"]


@pytest.mark.parametrize("kw", ["kw.data", "kw.headers", "kw.timeout"])
def test_another_keyword_on_request_withholds_the_bare_get_default_and_the_verb_is_any(kw):
    """RULING (plan P1-49, "Tests that 49b changes or adds" (b)): `lift._verbs_at` withholds `bareDefault` when the seed's
    container argument (`kw`) holds any value set or flow, deliberately: `data=` makes a urllib `Request` a POST, and `headers=`
    cannot be told apart from it by a `kw.*` head. The earlier "stays GET" expectation was wrong, not a limit. The bare
    `Request(url)` stays GET and `Request(url, method="POST")` stays POST (tests above)."""
    recs = urllib_module(site("r1", SHAPES["Request"]), vs("r1", "a0", "http://svc:8200/documents"), vs("r1", kw, UNK))
    assert [c["key"] for c in consumers(recs)] == ["* /documents"]


def test_a_literal_urlopen_is_a_get_consumer():
    recs = urllib_module(site("u1", SHAPES["urlopen"]), vs("u1", "a0", "http://svc:8200/health"))
    assert [dview(c) for c in consumers(recs)] == [("http", "GET /health", "confirmed")]


def test_urlopen_of_a_non_url_object_is_not_a_consumer():
    """`urlopen(req)`: the argument is a `Request` object (an unknown element), not a URL string."""
    recs = urllib_module(site("u1", SHAPES["urlopen"]), vs("u1", "a0", UNK))
    assert consumers(recs) == []


# ---------------------------------------------------------------------------------------------------------------------
# negatives that the stored bundles show are real: a `.get` that is not HTTP, inside a file that imports an HTTP client
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape,arg", [("dict.get", "key"), ("environ.get", "FX_PEER_BASE"), ("dict.get", "/looks/like/a/path")],
                         ids=["dict.get", "os.environ.get", "dict.get-with-a-path-string"])
def test_a_typed_non_http_get_is_not_a_consumer_even_in_a_file_importing_requests(shape, arg):
    recs = module(("requests",)) + [site("c1", SHAPES[shape]), vs("c1", "a0", arg)]
    assert consumers(recs) == []


def test_an_untyped_dot_get_with_a_non_url_string_key_is_not_a_consumer_in_a_file_importing_requests():
    """`cfg.get("timeout")` on an untyped receiver has the same record as `requests.get(u)` (no receiver, no `externalTarget`);
    only the literal can tell them apart. If the pack format cannot express that, the plan says: return it to the architect."""
    recs = module(("requests",)) + [site("c1", SHAPES["verb"]("get")), vs("c1", "a0", "timeout")]
    assert consumers(recs) == []


def dict_get_wrapper(arg):
    """``def lookup(name): return d.get(name)`` (untyped ``d``) called with ``arg``: the leaf is the caller's slot."""
    return module(("requests",)) + [
        sym("w", "lookup", fid="f1", path="svc/client.py", start=30), param("p0", "w", 0, "name", line=31),
        site("rq", SHAPES["verb"]("get"), owner="w", line=32), data("rq", "p0", "a0", "p0"), vs("rq", "a0", UNK),
        sym("m", "run", fid="f1", path="svc/client.py", start=50),
        call("cs", "m", "lookup", line=51, resolution="resolved", targets=("w",)), vs("cs", "a0", arg)]


def test_a_parameter_fed_dot_get_called_with_a_non_url_string_is_dropped_at_the_leaf():
    assert consumers(dict_get_wrapper("timeout")) == []


def test_RESIDUAL_LIMIT_a_parameter_fed_dot_get_called_with_a_url_like_string_is_a_consumer():
    """Pinned by name (plan P1-49 "Untyped verbs"): `routes.get("/health")` in a file importing `requests` cannot be told from
    `requests.get("/health")` in the stored bundles, so a URL-like argument is kept; it enters P1-36's precision measure."""
    (c,) = consumers(dict_get_wrapper("/x"))
    assert (c["siteId"], c["key"]) == ("cs", "GET /x")


# ---------------------------------------------------------------------------------------------------------------------
# wrappers: carried by the lifting fixpoint when 5d keeps the call that reaches them
# ---------------------------------------------------------------------------------------------------------------------

def wrapper(*, seed_slot=None, verb="post"):
    """``def _get(self, path): return requests.<verb>(<seed_slot or path>)`` plus the function ``m`` that will call it."""
    recs = module(("requests",)) + [
        sym("w", "_get", fid="f1", path="svc/client.py", start=30), param("p0", "w", 0, "path", line=31),
        site("rq", SHAPES["verb"](verb), owner="w", line=32), data("rq", "p0", "a0", "p0"),
        sym("m", "run", fid="f1", path="svc/client.py", start=50)]
    if seed_slot is not None:
        recs.append(vs("rq", "a0", seed_slot))
    return recs


SELF_BASE_PLUS_PATH = {"concat": [UNK, UNK]}           # `self.base + path`


UNTYPED_CALL = lambda: wrapper(seed_slot=SELF_BASE_PLUS_PATH) + [          # noqa: E731
    call("cs", "m", "_get", line=51, resolution="unresolved"), vs("cs", "a0", "/items")]


@pytest.mark.parametrize("receiver", ["self", "untyped-parameter"])
def test_a_wrapper_called_through_an_untyped_receiver_is_not_lifted_to_the_caller(receiver):
    """After 5d the call is name-linked: `resolution` unresolved, `targets` empty, so `callers_of` has no edge (pinned limit)."""
    assert "cs" not in {c["siteId"] for c in consumers(UNTYPED_CALL())}, receiver


def test_a_wrapper_with_no_lifted_caller_and_a_partly_known_template_is_itself_the_possible_partial_key_consumer():
    """`ownSite` (plan P1-49 Format additions): the seed slot `f"/items/{x}"` is the wrapper's own template, so its own site is
    the consumer, `possible`/`partial-key` because a path piece is unresolved (`/items/{?}`); it holds for a call through an
    untyped receiver (not lifted) exactly as for a wrapper nobody calls."""
    recs = wrapper(seed_slot={"template": [{"const": "/items/"}, UNK]}) + [call("cs", "m", "_get", line=51, resolution="unresolved"),
                                                                           vs("cs", "a0", "/items")]
    got = consumers(recs)
    assert [(c["siteId"], c["key"], c["status"], c["reason"]) for c in got] == [("rq", "POST /items/{?}", "possible", "partial-key")]


def test_a_wrapper_with_no_lifted_caller_and_a_template_with_no_url_like_text_is_no_consumer():
    """`self.base + path` is `\\x01\\x01` at the seed slot: a `POST /{?}` key would only match any one-segment provider."""
    assert consumers(UNTYPED_CALL()) == []


def test_the_same_wrapper_through_a_constructor_typed_receiver_is_lifted_to_the_caller():
    """The call is type-backed (`resolved`, a target): the caller is the consumer, anchored at the calling function."""
    recs = wrapper(seed_slot=UNK) + [
        call("cs", "m", "_get", line=51, resolution="resolved", targets=("w",)), vs("cs", "a0", "/items")]
    (c,) = consumers(recs)
    assert (c["siteId"], c["anchorId"], c["key"], c["status"], c["reason"]) == ("cs", "m", "POST /items", "confirmed", "lifted-role")


def test_lnk_11_shape_a_constructor_typed_wrapper_over_self_base_plus_path_is_possible_with_the_opaque_slot():
    recs = wrapper(seed_slot=SELF_BASE_PLUS_PATH) + [
        call("cs", "m", "_get", line=51, resolution="resolved", targets=("w",)), vs("cs", "a0", "/items")]
    (c,) = consumers(recs)
    assert (c["siteId"], c["key"], c["status"], c["reason"]) == ("cs", "POST /items", "possible", "partial-key")
    assert c["evidence"]["opaqueSlot"] == {"siteId": "rq", "arg": "a0"}


def http_json(*, seed_slot=None):
    """``def http_json(method, url): urllib.request.Request(url, method=method)`` and its bare same-module caller ``run``."""
    recs = module(("urllib.request", "Request", "urlopen")) + [
        sym("w", "http_json", fid="f1", path="svc/client.py", start=30),
        param("p0", "w", 0, "method", line=31), param("p1", "w", 1, "url", line=31),
        site("rq", SHAPES["Request"], owner="w", line=33), data("rq", "p1", "a0", "p1"), data("rq", "p0", "kw.method", "p0"),
        sym("m", "run_participant", fid="f1", path="svc/client.py", start=50)]
    if seed_slot is not None:
        recs.append(vs("rq", "a0", seed_slot))
    return recs


def test_a_bare_call_of_a_wrapper_is_lifted_with_the_lifted_verb_and_a_parameter_base_drops_its_origin():
    """`load_demo.run_participant`: `http_json(method, url)` called as `http_json("GET", f"{base_url}/shop/api/session")`."""
    recs = http_json() + [
        call("cs", "m", "http_json", line=51, resolution="resolved", targets=("w",)),
        vs("cs", "a0", "GET"), vs("cs", "a1", {"template": [UNK, {"const": "/shop/api/session"}]})]
    (c,) = consumers(recs)
    assert (c["siteId"], c["anchorId"], c["key"], c["status"], c["reason"]) == (
        "cs", "m", "GET /shop/api/session", "possible", "partial-key")


def test_known_limit_a_wrapper_that_composes_the_url_in_a_local_before_request_stays_confirmed_lifted_role():
    """`seed_nlq_eval_corpus._http_json`: `url = f"{BASE_URL}{path}"; Request(url, ...)`. The seed slot is a bare `point-derived`
    unknown, indistinguishable from the parameter, so the lifted fact is `confirmed`/`lifted-role` (plan P1-49 rule 4 limit)."""
    recs = http_json(seed_slot=UNK) + [
        call("cs", "m", "http_json", line=51, resolution="resolved", targets=("w",)), vs("cs", "a0", "GET"), vs("cs", "a1", "/health")]
    (c,) = consumers(recs)
    assert (c["key"], c["status"], c["reason"]) == ("GET /health", "confirmed", "lifted-role")
