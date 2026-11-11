"""Builders for the linker tests (P1-17): hand-made ``PackFacts`` over readable-id bundles.

A fact has the shape ``ccg.packrt.run_packs`` emits (``_Run._item`` plus ``edge``/``kind``/``key``/``keyForm``/``status``/
``reason``/``evidence``); only the fields the linker reads are guaranteed. Units: ``fileId`` ``f-<unit>`` lives in unit
``<unit>``; a symbol is placed in a file by ``sym_in``.
"""
from __future__ import annotations

from ccg.linker import Binding, BoundEntry, Unmapped
from ccg.packrt import PackFacts
from packrt_helpers import call, file_, sym


def sym_in(sid: str, unit: str, name: str | None = None, role: str = "primary", path: str | None = None):
    """``[file, symbol]`` records: symbol ``sid`` in a file of ``unit`` (one file per (unit, role))."""
    p = path or f"{unit}/{role}.py"
    fid = f"f-{unit}-{role}-{p}"
    return [file_(fid, p, unit=unit, role=role), sym(sid, name or sid, fid=fid, path=p)]


def fact(edge: str, anchor: str, kind: str, key: str | None, *, site: str | None = None, status: str = "confirmed",
         reason: str = "key-match", form: str = "literal", evidence: dict | None = None, line: int = 1) -> dict:
    return {"packId": "p", "ruleId": "r", "anchorId": anchor, "qname": f"q::{anchor}", "path": "x.py",
            "siteId": site or f"site-{anchor}", "line": line, "edge": edge, "kind": kind, "key": key,
            "keyForm": form, "status": status, "reason": reason, "evidence": evidence or {}}


def prov(anchor: str, kind: str, key: str | None, **kw) -> dict:
    return fact("PROVIDES", anchor, kind, key, **kw)


def cons(anchor: str, kind: str, key: str | None, site: str, **kw) -> dict:
    if key is None:
        kw.setdefault("status", "possible")
        kw.setdefault("reason", "runtime-built")
    return fact("CONSUMES", anchor, kind, key, site=site, **kw)


def facts(providers=(), consumers=(), lifted=()) -> PackFacts:
    return PackFacts(list(providers), list(consumers), [], list(lifted), [], [], [])


def lifted_leaf(pack: str, lifting: str, site: str, symbol: str, *, hops=(), resolved=False) -> dict:
    return {"packId": pack, "lifting": lifting, "siteId": site, "symbolId": symbol, "path": "x.py", "line": 1,
            "urlPath": "a0", "alternatives": [] if not resolved else ["/x"], "resolved": resolved, "leaf": True,
            "hops": list(hops), "verbs": None, "partialAlternatives": False, "weakenedBy": None,
            "status": "possible", "reason": "partial-key"}


def shell_site(cid: str, owner: str, callee: str = "mcp-call", reason: str = "shell-dynamic"):
    r = call(cid, owner, callee, kind="command")
    r.update(status="possible", reason=reason)
    return r


def mention(cid: str, owner: str, name: str):
    r = call(cid, owner, name, kind="mention")
    r.update(status="possible", reason="prompt-mention")
    return r


class FakeBindings:
    """A ``Bindings`` provider keyed by consumer site id; records what it was asked."""

    def __init__(self, by_site: dict[str, Binding], mismatches=()):
        self.by_site, self.asked, self._mism = by_site, [], list(mismatches)

    def bind(self, consumer: dict):
        self.asked.append(consumer["siteId"])
        return self.by_site.get(consumer["siteId"])

    def key_mismatches(self) -> list[str]:
        return list(self._mism)


def binding(*units: str, entries=("e1",), unmapped: Unmapped | None = None, weaken=None) -> Binding:
    return Binding(units=tuple(units), entries=tuple(BoundEntry(e) for e in entries), unmapped=unmapped, weaken=weaken)


def header(unit: str, repo: str = "r") -> dict:
    return {"type": "header", "unitId": unit, "unit": {"repoId": repo, "rootPath": ".", "lang": "config"}}


def cfg_bundle(unit: str, entry_id: str, *, path: str = "deploy/compose.yaml", example: bool = False, launch_id: str | None = None,
               file_role: str = "primary", repo: str = "r") -> list[dict]:
    """A bundle of one config file with a ``configEntry`` (and optionally a ``launchFact``): no ``unitId`` on either record."""
    out = [header(unit, repo), {"type": "file", "id": f"cf-{unit}", "path": path, "unitId": unit, "role": file_role, "repoId": repo, "lang": "config"},
           {"type": "configEntry", "id": entry_id, "repoId": repo, "path": path, "line": 1, "keyPath": "k", "example": example}]
    if launch_id:
        out.append({"type": "launchFact", "id": launch_id, "path": path, "line": 2, "ordinal": 0, "kind": "command", "origin": "explicit"})
    return out
