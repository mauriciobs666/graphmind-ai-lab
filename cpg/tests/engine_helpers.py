"""Shared plumbing for ``test_engine_*.py`` (P1-21): a falkordb-py-shaped fake database (records every call, never writes), a scripted ``Reader`` for the
anchor logic, and the small assembled graph the live tests load. Nothing here touches ``~/.config/ccg``, ``falkordb-dev`` or ``ccg-falkordb``."""
from __future__ import annotations

from types import SimpleNamespace


class FakeGraph:
    """``ro_query`` answers from ``db.script`` (``text -> rows | callable(params) -> rows | Exception``); ``query`` is a write and fails the test."""

    def __init__(self, db, key):
        self.db, self.key = db, key

    def ro_query(self, cypher, params=None, timeout=None):
        self.db.calls.append(("ro_query", self.key, cypher, params, timeout))
        if len(self.db.calls) > 200:
            raise AssertionError("runaway: the reader keeps re-sending a query (an unbounded retry loop)")
        hit = self.db.script.get(cypher, [])
        if isinstance(hit, BaseException):
            raise hit
        rows = hit(params) if callable(hit) else hit
        return SimpleNamespace(result_set=[list(r) for r in rows])

    def query(self, *a, **k):
        raise AssertionError("the reader issued a write command (GRAPH.QUERY)")

    def delete(self, *a, **k):
        raise AssertionError("the reader issued GRAPH.DELETE")


class FakeDb:
    def __init__(self, script=None):
        self.script = dict(script or {})
        self.calls: list[tuple] = []
        self.keys: list[str] = []

    def select_graph(self, key):
        self.keys.append(key)
        return FakeGraph(self, key)

    @property
    def connection(self):
        raise AssertionError("the reader used the raw connection")

    def ro_calls(self):
        return [c for c in self.calls if c[0] == "ro_query"]


class ScriptedReader:
    """The ``Reader`` surface the anchors use: ``read(name, params)`` answers from ``script`` (``name -> rows | callable(params)``), records the calls."""

    def __init__(self, script=None, *, system="shop"):
        self.script, self.system, self.calls = dict(script or {}), system, []

    def read(self, name, params=None):
        self.calls.append((name, dict(params or {})))
        hit = self.script.get(name, [])
        rows = hit(params) if callable(hit) else hit
        return [tuple(r) for r in rows]

    def names(self):
        return [c[0] for c in self.calls]


# ------------------------------------------------------------------------------------------------ the small assembled graph (live tests)

#: ids and props of the planted facts, named once so the live tests and the expectations read the same
EVIL_QNAME = "x' }) DETACH DELETE (n) //"


def mini_rows(system: str = "shop"):
    """A small ``GraphRows`` in the assembler's shapes (``-p1-graph.md`` §1-§2), planted for the reader's cases: two repos with one path in both, a skipped
    file, symbols for the File+line rule (nesting, an owned region outside the span, a span/region tie, a depth tie, a full tie), two same-name candidates
    with a resolved and an unresolved site, one link with two sites that each carry their own binding chain and one entry bound twice, a launch entry, a
    ``TO_UNIT`` link, three data/control points, and a symbol whose qname is a Cypher injection."""
    from loader_helpers import snapshot_row
    from ccg.assemble.rows import GraphRows
    r = GraphRows()

    def node(label, **props):
        r.nodes.setdefault(label, []).append(props)

    def edge(src, etype, dst, s, t, **p):
        r.edges.setdefault((src, etype, dst), []).append({"s": s, "t": t, **({"p": p} if p else {})})

    for repo in ("repo-a", "repo-b"):
        node("Repo", repoId=repo, headCommit="c" * 40)
    node("Unit", unitId="ua", repoId="repo-a", rootPath="", language="python", buildStatus="fresh", flowAvailable=True, builtCommit="c" * 40,
         builtAt="2026-11-10T00:00:00Z", bundleSchemaVersion="1.1")
    node("Unit", unitId="ub", repoId="repo-b", rootPath="", language="python", buildStatus="stale-kept", flowAvailable=False, builtCommit="d" * 40,
         failedAt="2026-11-11T00:00:00Z", failReason="extract:failed")
    node("Unit", unitId="un", repoId="repo-a", rootPath="svc/new", language="python", buildStatus="not-mapped", flowAvailable=False, failReason="manifest:never-built")
    edge("Unit", "IN_REPO", "Repo", "ua", "repo-a")
    edge("Unit", "IN_REPO", "Repo", "ub", "repo-b")
    edge("Unit", "IN_REPO", "Repo", "un", "repo-a")
    for fid, repo, unit, path in (("fa1", "repo-a", "ua", "app/a.py"), ("fa2", "repo-a", "ua", "app/b.py"), ("fb1", "repo-b", "ub", "app/a.py")):
        node("File", fileId=fid, repoId=repo, unitId=unit, path=path, kind="code", lang="python")
        edge("Unit", "HAS_FILE", "File", unit, fid)
    node("SkippedFile", repoId="repo-a", unitId="ua", path="tests/test_a.py", reason="test")
    edge("Unit", "SKIPS", "SkippedFile", "ua", "tests/test_a.py")

    def sym(sid, file_id, repo, unit, path, qname, start, end, depth=0, owned=None, name=None):
        props = dict(symbolId=sid, repoId=repo, unitId=unit, kind="function", name=name or qname.rsplit(".", 1)[-1], nameNorm=(name or qname.rsplit(".", 1)[-1]).casefold(),
                     qname=qname, path=path, lineStart=start, lineEnd=end, depth=depth)
        if owned:
            props["ownedRegions"] = owned
        node("Symbol", **props)
        edge("File", "DEFINES", "Symbol", file_id, sid)

    A = ("fa1", "repo-a", "ua", "app/a.py")
    sym("s_outer", *A, "app.Outer", 1, 100)
    sym("s_f", *A, "app.Outer.f", 10, 30, depth=1)
    sym("s_g", *A, "app.Outer.f.g", 15, 20, depth=2)
    sym("s_span", *A, "app.span", 120, 125)
    sym("s_dec", *A, "app.decorated", 150, 170, owned=[120, 125, 130, 135])
    sym("s_t1", *A, "app.t1", 200, 205, depth=1)
    sym("s_t2", *A, "app.t2", 200, 205, depth=1)
    sym("s_run1", *A, "app.run", 300, 310, name="run")
    sym("s_run2", *A, "app.other.run", 320, 330, name="run")
    sym("s_caller", *A, "app.caller", 400, 420)
    sym("s_handler", *A, "app.handler", 500, 520)
    sym("s_caller2", *A, "app.caller2", 600, 620)
    sym("s_caller3", *A, "app.caller3", 700, 720)
    sym("s_evil", "fa2", "repo-a", "ua", "app/b.py", EVIL_QNAME, 1, 9, name="evil")
    sym("s_b_outer", "fb1", "repo-b", "ub", "app/a.py", "app.Outer", 1, 10)

    # call sites and names: a resolved site (CALLS s_run1) and an unresolved one, both with the two same-name candidates
    node("Name", nameKey="run")
    edge("Name", "RESOLVES_TO", "Symbol", "run", "s_run1")
    edge("Name", "RESOLVES_TO", "Symbol", "run", "s_run2")
    for cid, owner, line in (("c_res", "s_caller", 401), ("c_unres", "s_caller", 402), ("c_fetch1", "s_caller2", 601), ("c_fetch1b", "s_caller2", 602),
                             ("c_fetch2", "s_caller3", 701)):
        node("CallSite", callSiteId=cid, repoId="repo-a", unitId="ua", path="app/a.py", line=line, calleeName="run" if cid.startswith("c_res") or cid == "c_unres" else "fetch")
        edge("Symbol", "HAS_CALL", "CallSite", owner, cid)
    edge("CallSite", "CALLS", "Symbol", "c_res", "s_run1")
    edge("CallSite", "CALLS_NAME", "Name", "c_res", "run")
    edge("CallSite", "CALLS_NAME", "Name", "c_unres", "run")

    # points, data and control flow
    node("Param", pointId="p1", line=11, index=0, name="x")
    node("Ret", pointId="r1", line=29)
    node("Cond", pointId="k1", line=12, code="if x")
    edge("Symbol", "HAS_PARAM", "Param", "s_f", "p1")
    edge("Symbol", "HAS_RETURN", "Ret", "s_f", "r1")
    edge("Symbol", "HAS_COND", "Cond", "s_f", "k1")
    edge("CallSite", "DATA", "Param", "c_res", "p1", fromPath="p0", toPath="p0", status="possible", reason="field-truncated")
    edge("CallSite", "DATA", "Param", "c_unres", "p1", fromPath="p0.a", toPath="p0")
    edge("CallSite", "CONTROLLED_BY", "Cond", "c_res", "k1")
    edge("Param", "CONTROLLED_BY", "Cond", "p1", "k1")

    # contract points, links, bindings
    for cp, kind, key in (("cp_health", "http", "GET /shop/api/health"), ("cp_items", "http", "GET /shop/api/items/{}"), ("cp_post", "http", "POST /shop/api/items"),
                          ("cp_tool", "mcp_tool", "billing.invoice")):
        node("ContractPoint", cpId=cp, kind=kind, keyNorm=key)
    edge("Symbol", "PROVIDES", "ContractPoint", "s_handler", "cp_health")
    edge("Symbol", "PROVIDES", "ContractPoint", "s_handler", "cp_items", status="possible", reason="runtime-built")
    edge("CallSite", "CONSUMES", "ContractPoint", "c_fetch1", "cp_health")
    node("Link", linkId="l1", kind="http", status="confirmed", reason="binding-confirmed", matcher="exact")
    node("Link", linkId="l2", kind="http", status="possible", reason="port-shared", matcher="path-template")
    node("Link", linkId="l3", kind="http", status="possible", reason="target-not-mapped", matcher="exact")
    edge("Link", "TO", "Symbol", "l1", "s_handler")
    edge("Link", "ON", "ContractPoint", "l1", "cp_health")
    edge("Link", "ON", "ContractPoint", "l2", "cp_items")
    edge("Link", "TO_UNIT", "Unit", "l3", "un")
    edge("CallSite", "VIA", "Link", "c_fetch1", "l1", status="confirmed", reason="binding-confirmed", consumerKey="GET /shop/api/health")
    edge("CallSite", "VIA", "Link", "c_fetch2", "l1", status="possible", reason="example-config", consumerKey="GET /shop/api/health")
    edge("CallSite", "VIA", "Link", "c_fetch1", "l2", status="possible", reason="port-shared", consumerKey="GET /shop/api/health")
    edge("CallSite", "VIA", "Link", "c_fetch1b", "l2", status="possible", reason="port-shared", consumerKey="GET /shop/api/health")   # a second site of the symbol reaching l2: count(DISTINCT) matters
    edge("Link", "LIFTED_VIA", "CallSite", "l1", "c_fetch1", siteId="c_fetch1", ord=0, line=601, urlPath="/api/health", verbPath="GET", viaSiteId="c_fetch1", viaLine=601)
    edge("Link", "LIFTED_VIA", "Symbol", "l1", "s_caller2", siteId="c_fetch1", ord=1, line=600)
    edge("Link", "LIFTED_VIA", "CallSite", "l1", "c_fetch2", siteId="c_fetch2", ord=0, line=701)
    node("ConfigEntry", entryId="eX", repoId="repo-a", unitId="ua", path="cfg/app.yaml", line=3, keyPath="services.api.url", valueHash="v1")
    node("ConfigEntry", entryId="eZ", repoId="repo-a", unitId="ua", path="cfg/app.yaml", line=9, keyPath="services.api.port", valueHash="v2")
    node("ConfigEntry", entryId="eL", repoId="repo-a", unitId="ua", path="cfg/app.yaml", line=20, launchKind="command", argv0="uvicorn")
    edge("Link", "BOUND_BY", "ConfigEntry", "l1", "eX", siteId="c_fetch1", ord=0, port=8000, portOrigin="explicit")
    edge("Link", "BOUND_BY", "ConfigEntry", "l1", "eL", siteId="c_fetch1", ord=1, port=8001, portOrigin="pack-default", packId="fastapi")
    edge("Link", "BOUND_BY", "ConfigEntry", "l1", "eX", siteId="c_fetch2", ord=1, port=9000, portOrigin="compose")
    edge("Link", "BOUND_BY", "ConfigEntry", "l1", "eZ", siteId="c_fetch2", ord=0)

    r.snapshot = snapshot_row(system, "p121")
    return r
