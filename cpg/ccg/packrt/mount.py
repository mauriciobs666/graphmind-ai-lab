"""Mount composition and conditional mounts (proposal §3.3 "B3"; rule-pack spec §1.6 ``mounts``).

The semantics are the S12 recognizer's (``assemble.py`` ``_include_sites``, ``_mount_chains``, ``providers``), read from the
pack's declaration instead of hard-coded: the mount call is any call matching ``call`` under ``scope``, its child is the call
whose result feeds ``child.arg``, and a handler is mounted by every chain of mount calls that reaches its parent function.
``roots`` and ``constructorPrefix`` are the two cases the frozen records cannot decide (spec §2.3): a handler on a router
whose variable they cannot name, and a router that carries a prefix of its own. Both give the declaration's ``unmounted``
status, never ``confirmed``.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

from ccg.packrt.extract import Corpus, Ctx, eval_expr
from ccg.packrt.httpkey import UNKNOWN
from ccg.packrt.match import clause_hit, scope_ok


class MountIndex:
    """The mount calls one pack declares, indexed by the factory whose result each mounts."""

    def __init__(self, corpus: Corpus, pack_id: str, decl: dict):
        self.corpus, self.pack_id, self.decl = corpus, pack_id, decl
        self.sites: dict[str, list[dict]] = defaultdict(list)       # factory symbolId -> mount calls
        child = decl["child"]
        for cid in sorted(corpus.callsites):
            site = corpus.callsites[cid]
            if not clause_hit("callsite", decl["call"], site, corpus):
                continue
            if not scope_ok(decl.get("scope"), corpus.file_id_of(site), corpus):
                continue
            for d in corpus.data_by_use.get(cid, ()):
                made_by = corpus.callsites.get(d["def"])
                if d["toPath"] == child["arg"] and d["fromPath"] == "r" and made_by is not None:
                    for t in made_by["targets"]:
                        self.sites[t["symbolId"]].append(site)

    # -- facts ---------------------------------------------------------------------------------
    def prefix_alternatives(self, site: dict) -> list[str]:
        return eval_expr(self.decl["prefix"], Ctx(self.corpus, site)) or [""]

    def conditions(self, site: dict) -> list[dict]:
        c = self.corpus
        return [{"siteId": site["id"], "line": c.conds[k]["line"], "code": c.conds[k]["code"]}
                for k in sorted(c.controlled_by.get(site["id"], ()))]

    def facts(self) -> list[dict]:
        """One ``MOUNTED_BY`` fact per mount call, with its prefix alternatives and the conditions that control it."""
        seen: dict[str, dict] = {}
        for factory, sites in sorted(self.sites.items()):
            for s in sites:
                f = seen.setdefault(s["id"], {"factoryIds": []})
                f["factoryIds"].append(factory)
                f["site"] = s
        out = []
        for sid, f in sorted(seen.items()):
            s = f["site"]
            prefixes = self.prefix_alternatives(s)
            resolved = not any(UNKNOWN in p for p in prefixes)
            out.append({
                "packId": self.pack_id, "mountId": self.decl["id"], "siteId": sid, "symbolId": s["symbolId"],
                "path": self.corpus.symbols[s["symbolId"]]["path"], "line": s["line"],
                "factoryIds": sorted(set(f["factoryIds"])), "prefixes": [p.replace(UNKNOWN, "{?}") for p in prefixes],
                "conditions": self.conditions(s) if self.decl.get("recordConditions") else [],
                "status": "confirmed" if resolved else "possible", "reason": "key-match" if resolved else "partial-key",
            })
        return out

    # -- chains --------------------------------------------------------------------------------
    def chains(self, factory_id: str, _stack: tuple = ()) -> list[list[dict]]:
        """Mount-call chains that reach ``factory_id``, outermost first, to any depth (a cycle stops)."""
        out = []
        for s in self.sites.get(factory_id, ()):
            if s["symbolId"] in _stack:
                continue
            for o in self.chains(s["symbolId"], _stack + (factory_id,)) or [[]]:
                out.append(o + [s])
        return out

    def _calls_in(self, symbol_id: str, clause: dict) -> list[dict]:
        return [c for c in self.corpus.callsites_of.get(symbol_id, ()) if clause_hit("callsite", clause, c, self.corpus)]

    def _has_own_prefix(self, parent_id: str | None) -> bool:
        if not parent_id:
            return False
        for cp in self.decl.get("constructorPrefix", ()):
            if any(self.corpus.arg_values(c, cp["arg"]) is not None for c in self._calls_in(parent_id, cp["call"])):
                return True
        return False

    def _is_root(self, parent_id: str | None) -> bool:
        if not parent_id:
            return False
        for root in self.decl.get("roots", ()):
            if self._calls_in(parent_id, root["call"]) and not any(self._calls_in(parent_id, nw) for nw in root.get("notWith", ())):
                return True
        return False

    def variants(self, handler: dict) -> list[dict]:
        """The ways ``handler`` is mounted: ``{how, chain, ownPrefix}`` with ``how`` ``mounted``, ``app`` or ``unmounted-router``."""
        parent = handler.get("parentId")
        own = self._has_own_prefix(parent)
        chains = self.chains(parent) if parent else []
        if chains:
            return [{"how": "mounted", "chain": c, "ownPrefix": own} for c in chains]
        if self._is_root(parent):
            return [{"how": "app", "chain": [], "ownPrefix": own}]
        return [{"how": "unmounted-router", "chain": [], "ownPrefix": own}]

    def mount_alternatives(self, variant: dict) -> list[str]:
        """The joined prefix strings of a chain (outermost first); a piece that is unresolved carries ``UNKNOWN``."""
        sets = [self.prefix_alternatives(s) for s in variant["chain"]]
        return ["".join(combo) for combo in itertools.product(*sets)] if sets else [""]

    def evidence(self, variant: dict) -> dict:
        c = self.corpus
        return {
            "how": variant["how"],
            "sites": [{"siteId": s["id"], "path": c.symbols[s["symbolId"]]["path"], "line": s["line"]} for s in variant["chain"]],
            "conditions": [cd for s in variant["chain"] for cd in self.conditions(s)] if self.decl.get("recordConditions") else [],
        }

    def status(self, variant: dict) -> tuple[str, str]:
        """``confirmed/key-match`` unless the declaration's ``unmounted`` status applies (an unmounted router or its own prefix)."""
        if variant["how"] == "unmounted-router" or variant["ownPrefix"]:
            u = self.decl.get("unmounted", {"status": "possible", "reason": "partial-key"})
            return u["status"], u["reason"]
        return "confirmed", "key-match"

