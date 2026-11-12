"""Shared plumbing for the P1-39b answer-builder tests (``tests/test_engine_answer*.py``): worlds written in readable names (``tests/engine_walk_helpers.G``), rewritten
into the 32-hex ids the answer schema needs (:func:`hexify`), and an in-memory ``Reader`` that also answers the reads the builder adds to the walk's (``snapshot``,
``lifted_of``, ``bound_out``, and the three anchor re-reads). Nothing here touches ``~/.config/ccg``, ``falkordb-dev`` or ``ccg-falkordb``."""
from __future__ import annotations

import hashlib
import random
from pathlib import PurePosixPath

from engine_walk_helpers import ID_PROP, G, MemoryReader

from ccg import answer, ids
from ccg.assemble.rows import GraphRows
from ccg.engine.reader import _row_key

C1 = "a" * 40
T1 = "2026-11-01T10:00:00Z"
T2 = "2026-11-02T11:30:05Z"
SNAP = "0123456789abcdef0123456789abcdef"


def hid(name: str) -> str:
    """A stable 32-hex id for a readable name."""
    return hashlib.md5(name.encode()).hexdigest()


def hexify(g: G) -> tuple[G, dict[str, str]]:
    """``g`` with every id (node ids, ``unitId``, edge ``siteId``) rewritten to 32-hex; a symbol's id is the real ``ids.symbol_id`` of its (language by path, repo, qname)
    so V15 recomputes it. Returns the new graph and the old-name -> new-id map. Names must be unique across labels. Every fresh unit gets a build record."""
    idmap: dict[str, str] = {}
    for n in g.rows.nodes.get("Symbol", []):
        lang = answer.EXT_LANG.get(PurePosixPath(n["path"]).suffix, "python")
        idmap[n["symbolId"]] = ids.symbol_id(lang, n["repoId"], n["qname"])

    def m(x):
        return idmap.setdefault(x, hid(x)) if isinstance(x, str) else x

    out = G.__new__(G)
    out.rows, out.repo, out.unit = GraphRows(), g.repo, m(g.unit) if g.unit else None
    for label, nodes in g.rows.nodes.items():
        for n in nodes:
            n = dict(n)
            for prop in {ID_PROP.get(label), "unitId"}:
                if prop and prop != "nameKey" and isinstance(n.get(prop), str):
                    n[prop] = m(n[prop])
            if label == "Unit" and n.get("buildStatus") == "fresh":
                n.setdefault("builtCommit", C1)
                n.setdefault("builtAt", T1)
            out.rows.nodes.setdefault(label, []).append(n)
    for (src, et, dst), edges in g.rows.edges.items():
        for e in edges:
            e2 = {"s": e["s"] if src == "Name" else m(e["s"]), "t": e["t"] if dst == "Name" else m(e["t"])}
            if "p" in e:
                e2["p"] = {k: (m(v) if k in ("siteId", "viaSiteId") else v) for k, v in e["p"].items()}
            out.rows.edges.setdefault((src, et, dst), []).append(e2)
    return out, idmap


class AnswerReader(MemoryReader):
    """``MemoryReader`` plus the reads the builder adds: ``snapshot`` (``snapshot_rows``), ``lifted_of`` and ``bound_out`` (by ``pairs``, over ``LIFTED_VIA`` /
    ``BOUND_BY`` edges), ``symbol_by_qname``, ``cp_by_key`` and ``config_by_path``. Every call is recorded."""

    def __init__(self, g, snapshot_rows=None):
        super().__init__(g)
        self.snapshot_rows = [(SNAP, 0, answer.EMPTY_VERDICT_LOG_HASH, "1.1", "1.0")] if snapshot_rows is None else snapshot_rows

    def read(self, name, params=None):
        fn = getattr(self, "_x_" + name, None)
        if fn is None:
            return super().read(name, params)
        self.calls.append((name, dict(params or {})))
        return sorted((tuple(r) for r in fn(**(params or {}))), key=_row_key)

    def _x_snapshot(self):
        return self.snapshot_rows

    def _x_lifted_of(self, pairs):
        out = []
        for link, site in pairs:
            for label in ("CallSite", "Symbol"):
                for t, p in self.outof("Link", "LIFTED_VIA", label, link):
                    if p.get("siteId") != site:
                        continue
                    n = self.n(label, t)
                    out.append((link, site, p.get("ord"), label, t, n.get("repoId"), n.get("unitId"), n.get("path"), p.get("line"), p.get("urlPath"), p.get("verbPath"),
                                p.get("viaSiteId"), p.get("viaLine"), n.get("calleeName") if label == "CallSite" else n.get("qname")))
        return out

    def _x_bound_out(self, pairs):
        out = []
        for link, site in pairs:
            for e, p in self.outof("Link", "BOUND_BY", "ConfigEntry", link):
                if p.get("siteId") != site:
                    continue
                n = self.n("ConfigEntry", e)
                out.append((link, site, p.get("ord"), e, n.get("repoId"), n.get("unitId"), n.get("path"), n.get("line"), n.get("keyPath"), n.get("launchKind"), n.get("protocol"),
                            n.get("port"), n.get("envVar"), n.get("mountPath"), n.get("service"), n.get("argv0"), p.get("port"), p.get("portOrigin"), p.get("packId")))
        return out

    def _x_symbol_by_qname(self, repo, qname):
        return [(s["symbolId"], s.get("unitId"), s.get("path"), s.get("lineStart"), s.get("lineEnd"), s.get("ownedRegions"))
                for s in self._nodes.get("Symbol", {}).values() if s.get("repoId") == repo and s.get("qname") == qname]

    def _x_cp_by_key(self, key):
        return [(c["cpId"], c.get("kind"), c.get("keyNorm")) for c in self._nodes.get("ContractPoint", {}).values() if c.get("keyNorm") == key]

    def _x_config_by_path(self, repo, path):
        return [(e["entryId"], e.get("unitId"), e.get("line"), e.get("keyPath"), e.get("launchKind"))
                for e in self._nodes.get("ConfigEntry", {}).values() if e.get("repoId") == repo and e.get("path") == path]


class Shuffled:
    """Every reply of ``inner`` reshuffled by ``seed`` (``reverse``: and reversed): nothing may depend on the order a reply arrives in (FR-19). ``recipe_queries`` and
    ``system`` pass through."""

    def __init__(self, inner, seed, reverse=False):
        self.inner, self.system, self.seed, self.reverse = inner, inner.system, seed, reverse

    def read(self, name, params=None):
        rows = list(self.inner.read(name, params))
        random.Random(f"{self.seed}:{name}").shuffle(rows)
        return rows[::-1] if self.reverse else rows

    def bound_of_link(self, ids_):
        out = list(self.inner.bound_of_link(ids_))
        random.Random(f"{self.seed}:bol").shuffle(out)
        return out[::-1] if self.reverse else out

    def recipe_queries(self):
        return self.inner.recipe_queries()
