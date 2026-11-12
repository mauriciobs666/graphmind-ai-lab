"""Shared plumbing for the P1-22 walk tests: a small graph builder over the assembler's ``GraphRows`` shape and ``MemoryReader``, an in-memory ``Reader`` that
answers every expansion the walk uses with the exact columns of ``ccg.engine.reader.QUERIES`` (``-p1-graph.md`` §5), so a unit walk and a live walk over the same
rows can be compared (``tests/test_engine_bfs_live.py`` runs ``CypherReader`` and ``MemoryReader`` over one graph and asserts they agree row for row).
Nothing here touches ``~/.config/ccg``, ``falkordb-dev`` or ``ccg-falkordb``."""
from __future__ import annotations

from collections import Counter, defaultdict

from ccg.assemble.rows import GraphRows
from ccg.engine.reader import BoundTarget, ReaderError, QUERIES, _row_key

ID_PROP = {"Symbol": "symbolId", "CallSite": "callSiteId", "Param": "pointId", "Ret": "pointId", "Cond": "pointId", "Link": "linkId",
           "ContractPoint": "cpId", "ConfigEntry": "entryId", "Unit": "unitId", "Name": "nameKey"}
OWNER_REL = {"CallSite": "HAS_CALL", "Param": "HAS_PARAM", "Ret": "HAS_RETURN", "Cond": "HAS_COND"}


class G:
    """A graph in ``GraphRows`` shape with one-line adders. Every adder returns the id it made; ``u`` is the unit the symbol lives in (flag-true by default, set
    explicitly: P1-23's flag gate must change no pin here)."""

    def __init__(self, repo: str = "repo-a", unit: str = "ua", *, flow: bool = True):
        self.rows = GraphRows()
        self.repo, self.unit = repo, unit
        self.unit_node(unit, repo, flow=flow)

    def node(self, label, **props):
        self.rows.nodes.setdefault(label, []).append(props)

    def edge(self, src, etype, dst, s, t, **p):
        self.rows.edges.setdefault((src, etype, dst), []).append({"s": s, "t": t, **({"p": p} if p else {})})

    def unit_node(self, unit, repo=None, *, flow=True, root=""):
        self.node("Unit", unitId=unit, repoId=repo or self.repo, rootPath=root, language="python", buildStatus="fresh", flowAvailable=flow)
        return unit

    def sym(self, sid, qname=None, *, path="a.py", start=1, end=9, unit=None, repo=None, name=None):
        qname = qname or sid
        self.node("Symbol", symbolId=sid, repoId=repo or self.repo, unitId=unit or self.unit, kind="function", name=name or qname, nameNorm=(name or qname).casefold(),
                  qname=qname, path=path, lineStart=start, lineEnd=end, depth=0)
        return sid

    def site(self, cid, owner, *, line=5, callee="f", path="a.py", unit=None, repo=None):
        self.node("CallSite", callSiteId=cid, repoId=repo or self.repo, unitId=unit or self.unit, path=path, line=line, calleeName=callee)
        self.edge("Symbol", "HAS_CALL", "CallSite", owner, cid)
        return cid

    def param(self, pid, owner, *, index=0, name="x", line=2):
        self.node("Param", pointId=pid, line=line, index=index, name=name)
        self.edge("Symbol", "HAS_PARAM", "Param", owner, pid)
        return pid

    def ret(self, pid, owner, *, line=8):
        self.node("Ret", pointId=pid, line=line)
        self.edge("Symbol", "HAS_RETURN", "Ret", owner, pid)
        return pid

    def cond(self, pid, owner, *, line=4, code="if x"):
        self.node("Cond", pointId=pid, line=line, code=code)
        self.edge("Symbol", "HAS_COND", "Cond", owner, pid)
        return pid

    def calls(self, cid, target, **p):
        self.edge("CallSite", "CALLS", "Symbol", cid, target, **p)

    def calls_name(self, cid, target, name="f"):
        """The compound hop: the site calls the name, the name resolves to the symbol (no ``CALLS`` edge: pass ``resolved=`` via :meth:`calls`)."""
        if not any(n["nameKey"] == name for n in self.rows.nodes.get("Name", [])):
            self.node("Name", nameKey=name)
        self.edge("CallSite", "CALLS_NAME", "Name", cid, name)
        if not any(e["s"] == name and e["t"] == target for e in self.rows.edges.get(("Name", "RESOLVES_TO", "Symbol"), [])):
            self.edge("Name", "RESOLVES_TO", "Symbol", name, target)

    def data(self, use_label, use, def_label, d, from_path, to_path, **p):
        self.edge(use_label, "DATA", def_label, use, d, fromPath=from_path, toPath=to_path, **p)

    def controlled(self, label, point, cond):
        self.edge(label, "CONTROLLED_BY", "Cond", point, cond)

    def cp(self, cpid, key="GET /x", kind="http"):
        self.node("ContractPoint", cpId=cpid, kind=kind, keyNorm=key)
        return cpid

    def provides(self, sid, cpid, status="confirmed", reason="key-match"):
        self.edge("Symbol", "PROVIDES", "ContractPoint", sid, cpid, status=status, reason=reason)

    def consumes(self, cid, cpid, status="confirmed", reason="key-match"):
        self.edge("CallSite", "CONSUMES", "ContractPoint", cid, cpid, status=status, reason=reason)

    def link(self, lid, *, provider=None, cp=None, to_unit=None, status="confirmed", reason="key-match", kind="http", matcher="exact"):
        self.node("Link", linkId=lid, kind=kind, status=status, reason=reason, matcher=matcher)
        if provider:
            self.edge("Link", "TO", "Symbol", lid, provider)
        if cp:
            self.edge("Link", "ON", "ContractPoint", lid, cp)
        if to_unit:
            self.edge("Link", "TO_UNIT", "Unit", lid, to_unit)
        return lid

    def via(self, cid, lid, status="confirmed", reason="key-match", key="GET /x"):
        self.edge("CallSite", "VIA", "Link", cid, lid, status=status, reason=reason, consumerKey=key)

    def entry(self, eid, *, key_path="svc.url", path="cfg/a.yaml", line=3, launch=None, unit=None, repo=None):
        props = dict(entryId=eid, repoId=repo or self.repo, unitId=unit or self.unit, path=path, line=line)
        props.update(launchKind=launch) if launch else props.update(keyPath=key_path)
        self.node("ConfigEntry", **props)
        return eid

    def bound(self, lid, eid, site, ord=0, **p):
        self.edge("Link", "BOUND_BY", "ConfigEntry", lid, eid, siteId=site, ord=ord, **p)


class MemoryReader:
    """A ``Reader`` over ``G.rows``: the same names, parameters and columns as ``CypherReader``, rows in the same total order. Records every call. Edges are
    indexed by endpoint, so a slice-sized graph (tens of thousands of edges) walks in seconds."""

    def __init__(self, g: G, *, system: str = "shop"):
        self.system, self.calls = system, []
        self._nodes = {lab: {n[ID_PROP[lab]]: n for n in ns} for lab, ns in g.rows.nodes.items() if lab in ID_PROP}
        self._by_t = defaultdict(lambda: defaultdict(list))   # (src, type, dst) -> target id -> [(source id, props)]
        self._by_s = defaultdict(lambda: defaultdict(list))   # (src, type, dst) -> source id -> [(target id, props)]
        for key, lst in g.rows.edges.items():
            for e in lst:
                self._by_t[key][e["t"]].append((e["s"], e.get("p", {})))
                self._by_s[key][e["s"]].append((e["t"], e.get("p", {})))
        self._types = defaultdict(list)                       # type -> [(src label, dst label)]
        for src, et, dst in g.rows.edges:
            self._types[et].append((src, dst))

    # -- helpers
    def n(self, label, ident):
        return self._nodes.get(label, {}).get(ident, {})

    def into(self, src, etype, dst, i):
        """The edges ``(x:src)-[etype]->(i:dst)``: ``[(x id, props)]``."""
        return self._by_t[(src, etype, dst)].get(i, []) if (src, etype, dst) in self._by_t else []

    def outof(self, src, etype, dst, i):
        """The edges ``(i:src)-[etype]->(y:dst)``: ``[(y id, props)]``."""
        return self._by_s[(src, etype, dst)].get(i, []) if (src, etype, dst) in self._by_s else []

    def _sym_row(self, sid):
        s = self.n("Symbol", sid)
        return (sid, s.get("repoId"), s.get("unitId"), s.get("qname"), s.get("path"), s.get("lineStart"))

    def _site_row(self, cid):
        c = self.n("CallSite", cid)
        return (cid, c.get("repoId"), c.get("unitId"), c.get("path"), c.get("line"), c.get("calleeName"))

    def _link_row(self, lid):
        lk = self.n("Link", lid)
        return (lid, lk.get("kind"), lk.get("status"), lk.get("reason"), lk.get("matcher"))

    @staticmethod
    def _st(p, default_status="confirmed", default_reason=None):
        return (p.get("status", default_status), p.get("reason", default_reason))

    def read(self, name, params=None):
        self.calls.append((name, dict(params or {})))
        if name not in QUERIES:
            raise ReaderError("unknown query")
        fn = getattr(self, "_q_" + name.replace(":", "_"), None)
        if fn is None:
            raise NotImplementedError(f"MemoryReader has no {name}")
        out = []
        for i in list((params or {}).get("ids", [])):
            out.extend(fn(i))
        return sorted((tuple(r) for r in out), key=_row_key)

    # -- impact
    def _q_calls_in(self, i):
        return [(i, *self._site_row(c), *self._st(p, "confirmed", "resolved-call")) for c, p in self.into("CallSite", "CALLS", "Symbol", i)]

    def _resolved(self, c):
        return len(self._by_s[("CallSite", "CALLS", "Symbol")].get(c, [])) if ("CallSite", "CALLS", "Symbol") in self._by_s else 0

    def _unresolved_pair(self, c, s):
        if any(t == s for t, _ in self.outof("CallSite", "CALLS", "Symbol", c)):
            return False
        return not any(t == s for t, _ in self.outof("CallSite", "DISMISSED", "Symbol", c))

    def _name_targets(self, c):
        for nm, _ in self.outof("CallSite", "CALLS_NAME", "Name", c):
            for s, _ in self.outof("Name", "RESOLVES_TO", "Symbol", nm):
                yield s

    def _q_name_in(self, i):
        out = []
        for n_, _ in self.into("Name", "RESOLVES_TO", "Symbol", i):
            for c, _ in self.into("CallSite", "CALLS_NAME", "Name", n_):
                if self._unresolved_pair(c, i):
                    out.append((i, *self._site_row(c), self._resolved(c)))
        return out

    def _q_name_out(self, i):
        out = []
        for s in self._name_targets(i):
            if self._unresolved_pair(i, s):
                out.append((i, *self._sym_row(s), self._resolved(i)))
        return out

    def _data_in(self, label, i):
        out = []
        for src, dst in self._types["DATA"]:
            if dst == label:
                for u, p in self.into(src, "DATA", dst, i):
                    out.append((i, src, u, self.n(src, u).get("line"), p.get("fromPath"), p.get("toPath"), *self._st(p, "confirmed", "intra-flow")))
        return out

    def _q_data_in_Param(self, i):
        return self._data_in("Param", i)

    def _q_data_in_CallSite(self, i):
        return self._data_in("CallSite", i)

    def _data_out(self, label, i):
        out = []
        for src, dst in self._types["DATA"]:
            if src == label:
                for d, p in self.outof(src, "DATA", dst, i):
                    out.append((i, dst, d, self.n(dst, d).get("line"), p.get("fromPath"), p.get("toPath"), *self._st(p, "confirmed", "intra-flow")))
        return out

    def _q_data_out_CallSite(self, i):
        return self._data_out("CallSite", i)

    def _q_data_out_Ret(self, i):
        return self._data_out("Ret", i)

    def _q_data_out_Cond(self, i):
        return self._data_out("Cond", i)

    def _q_controlled_in(self, i):
        out = []
        for src, dst in self._types["CONTROLLED_BY"]:
            for p, _ in self.into(src, "CONTROLLED_BY", dst, i):
                out.append((i, src, p, self.n(src, p).get("line")))
        return out

    def _controlled_out(self, label, i):
        return [(i, k, self.n("Cond", k).get("line"), self.n("Cond", k).get("code")) for k, _ in self.outof(label, "CONTROLLED_BY", "Cond", i)]

    def _q_controlled_out_CallSite(self, i):
        return self._controlled_out("CallSite", i)

    def _q_controlled_out_Ret(self, i):
        return self._controlled_out("Ret", i)

    def _q_controlled_out_Cond(self, i):
        return self._controlled_out("Cond", i)

    def _q_controlled_out_Param(self, i):
        return self._controlled_out("Param", i)

    def _q_to_in(self, i):
        return [(i, *self._link_row(lid)) for lid, _ in self.into("Link", "TO", "Symbol", i)]

    def _q_via_in(self, i):
        return [(i, *self._site_row(c), p.get("status"), p.get("reason"), p.get("consumerKey")) for c, p in self.into("CallSite", "VIA", "Link", i)]

    def _q_consumes_in(self, i):
        return [(i, *self._site_row(c), p.get("status"), p.get("reason")) for c, p in self.into("CallSite", "CONSUMES", "ContractPoint", i)]

    def _q_on_in(self, i):
        return [(i, *self._link_row(lid)) for lid, _ in self.into("Link", "ON", "ContractPoint", i)]

    def _q_provides_in(self, i):
        return [(i, *self._sym_row(s), p.get("status"), p.get("reason")) for s, p in self.into("Symbol", "PROVIDES", "ContractPoint", i)]

    def _q_bound_in(self, i):
        return [(i, lid, self.n("Link", lid).get("kind"), self.n("Link", lid).get("status"), self.n("Link", lid).get("reason"), p.get("siteId"), p.get("ord"))
                for lid, p in self.into("Link", "BOUND_BY", "ConfigEntry", i)]

    def _owner(self, label, i):
        out = []
        for s, _ in self.into("Symbol", OWNER_REL[label], label, i):
            sy = self.n("Symbol", s)
            out.append((i, s, sy.get("repoId"), sy.get("unitId"), sy.get("qname"), sy.get("path"), sy.get("lineStart"), sy.get("lineEnd"), sy.get("ownedRegions")))
        return out

    def _q_owner_of_CallSite(self, i):
        return self._owner("CallSite", i)

    def _q_owner_of_Param(self, i):
        return self._owner("Param", i)

    def _q_owner_of_Ret(self, i):
        return self._owner("Ret", i)

    def _q_owner_of_Cond(self, i):
        return self._owner("Cond", i)

    # -- depends-on
    def _q_calls_out(self, i):
        return [(i, *self._sym_row(s), *self._st(p, "confirmed", "resolved-call")) for s, p in self.outof("CallSite", "CALLS", "Symbol", i)]

    def _has_out(self, label, i, cols):
        return [(i, p, *cols(self.n(label, p))) for p, _ in self.outof("Symbol", OWNER_REL[label], label, i)]

    def _q_has_out_CallSite(self, i):
        return self._has_out("CallSite", i, lambda n: (n.get("line"), n.get("calleeName")))

    def _q_has_out_Param(self, i):
        return self._has_out("Param", i, lambda n: (n.get("line"), n.get("index"), n.get("name")))

    def _q_has_out_Ret(self, i):
        return self._has_out("Ret", i, lambda n: (n.get("line"),))

    def _q_has_out_Cond(self, i):
        return self._has_out("Cond", i, lambda n: (n.get("line"), n.get("code")))

    def _q_consumes_out(self, i):
        return [(i, cp, self.n("ContractPoint", cp).get("kind"), self.n("ContractPoint", cp).get("keyNorm"), p.get("status"), p.get("reason"))
                for cp, p in self.outof("CallSite", "CONSUMES", "ContractPoint", i)]

    def _q_provides_out(self, i):
        return [(i, cp, self.n("ContractPoint", cp).get("kind"), self.n("ContractPoint", cp).get("keyNorm"), p.get("status"), p.get("reason"))
                for cp, p in self.outof("Symbol", "PROVIDES", "ContractPoint", i)]

    def _q_via_out(self, i):
        return [(i, *self._link_row(lid), p.get("status"), p.get("reason"), p.get("consumerKey")) for lid, p in self.outof("CallSite", "VIA", "Link", i)]

    def _q_to_out(self, i):
        return [(i, *self._sym_row(s)) for s, _ in self.outof("Link", "TO", "Symbol", i)]

    def _q_to_unit_out(self, i):
        return [(i, u, self.n("Unit", u).get("repoId"), self.n("Unit", u).get("rootPath"), self.n("Unit", u).get("buildStatus")) for u, _ in self.outof("Link", "TO_UNIT", "Unit", i)]

    def bound_of_link(self, link_ids):
        self.calls.append(("bound_of_link", {"ids": list(link_ids)}))
        rows = []
        for i in link_ids:
            for e, p in self.outof("Link", "BOUND_BY", "ConfigEntry", i):
                en = self.n("ConfigEntry", e)
                rows.append((i, e, p.get("siteId"), p.get("ord"), en.get("repoId"), en.get("unitId"), en.get("path"), en.get("line"), en.get("keyPath"),
                             en.get("launchKind"), p.get("port"), p.get("portOrigin"), p.get("packId")))
        picked = {}
        for row in sorted(rows, key=_row_key):
            link, entry, site, ord_, repo, unit, path, line, key_path, launch, port, origin, pack = row
            if not isinstance(key_path, str) or launch is not None:
                continue
            picked.setdefault((link, entry), BoundTarget(link, entry, site, ord_, repo, unit, path, line, key_path, port, origin, pack))
        return list(picked.values())

    def recipe_queries(self):
        return [{"name": n, "cypher": QUERIES[n]} for n in sorted({c[0] for c in self.calls if c[0] in QUERIES})]

    def names(self):
        return [c[0] for c in self.calls]


def union(parts: dict[str, G]) -> G:
    """One graph out of several fixtures, every id of fixture ``k`` prefixed ``k:`` (node ids, ``unitId``, ``siteId``, ``nameKey``), so one live graph holds them all
    and a walk over fixture ``k`` anchors at ``k:<id>``. Id-bearing properties of the rows are the only ones rewritten."""
    out = G.__new__(G)
    out.rows, out.repo, out.unit = GraphRows(), "repo-a", None
    for k, g in parts.items():
        pre = f"{k}:"
        for label, nodes in g.rows.nodes.items():
            for n in nodes:
                n = dict(n)
                for prop in {ID_PROP.get(label), "unitId"}:
                    if prop and isinstance(n.get(prop), str):
                        n[prop] = pre + n[prop]
                out.rows.nodes.setdefault(label, []).append(n)
        for key, edges in g.rows.edges.items():
            for e in edges:
                e2 = {"s": pre + e["s"], "t": pre + e["t"]}
                if "p" in e:
                    e2["p"] = {**e["p"], **({"siteId": pre + e["p"]["siteId"]} if "siteId" in e["p"] else {})}
                out.rows.edges.setdefault(key, []).append(e2)
    return out
