"""``run_packs``: evaluate declarative rule packs over unit bundles (plan P1-16; rule-pack spec §1).

No pack-specific Python lives here: every behaviour is read from ``cpg/packs/<name>/pack.yaml`` (loaded by
:func:`ccg.rulepack.load_pack`); a branch on a pack's id or framework is a defect. The runtime's output is a pure function of
the bundles and the packs (spec §1.8): iteration is in record-id order, duplicates collapse on
``(edge, anchor, kind, key, site)`` with a ``confirmed`` copy winning, and every item is sorted by its identifying fields.
"""
from __future__ import annotations

import itertools
import json
import re
from typing import NamedTuple

from ccg.packrt.extract import Corpus, Ctx, eval_expr, slot_mode
from ccg.packrt.httpkey import UNKNOWN, normalize_key
from ccg.packrt.launch import launch_facts
from ccg.packrt.lift import run_lifting
from ccg.packrt.match import Hit, UnsupportedSubject, clause_hit, scope_ok
from ccg.packrt.mount import MountIndex

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_EDGES = {"provides": "PROVIDES", "consumes": "CONSUMES"}
_STRENGTH = {"confirmed": 0, "possible": 1}


class PackFacts(NamedTuple):
    """Everything the packs say about the bundles. ``diagnostics`` lists what a rule dropped (spec §1.4: a diagnostic is not a
    registry code and reaches no graph element)."""
    providers: list[dict]
    consumers: list[dict]
    mounts: list[dict]
    lifted: list[dict]
    launch_facts: list[dict]
    marks: list[dict]
    diagnostics: list[dict]


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True)


class _Run:
    def __init__(self, bundles: list[list[dict]], packs: list[dict]):
        self.corpus = Corpus(bundles)
        self.packs = sorted(packs, key=lambda p: p["id"])
        self.edges: dict[tuple, dict] = {}
        self.mounts: list[dict] = []
        self.lifted: list[dict] = []
        self.launch: list[dict] = []
        self.marks: list[dict] = []
        self.diagnostics: list[dict] = []

    # -- candidates ----------------------------------------------------------------------------
    def _candidates(self, rule: dict, liftings: dict[str, list[dict]]):
        """``(record, hit, lifted item)`` for each clause of the rule's ``match``, in record-id order."""
        c = self.corpus
        match = rule["match"]
        for clause in match["anyOf"] if "anyOf" in match else [match]:
            ((subject, body),) = clause.items()
            if subject == "decorator":
                for d in sorted(c.decorators, key=lambda d: d["id"]):
                    if hit := clause_hit("decorator", body, d, c):
                        yield d, hit, None
            elif subject == "callsite":
                for cid in sorted(c.callsites):
                    if hit := clause_hit("callsite", body, c.callsites[cid], c):
                        yield c.callsites[cid], hit, None
            elif subject == "lifted":
                for item in liftings[body["lifting"]]:
                    if item["resolved"] and item["leaf"]:
                        site = c.callsites[item["siteId"]]
                        yield site, Hit(site["calleeName"]), item
            else:
                raise UnsupportedSubject(f"match subject {subject!r} is not evaluated by the P1-16 runtime")

    # -- one pack ------------------------------------------------------------------------------
    def run_pack(self, pack: dict) -> None:
        c = self.corpus
        self._refuse_what_is_not_evaluated(pack)
        indexes = [MountIndex(c, pack["id"], d) for d in pack.get("mounts", ())]
        for ix in indexes:
            self.mounts += ix.facts()
        liftings = {}
        for lf in pack.get("liftings", ()):
            liftings[lf["id"]] = run_lifting(c, pack["id"], lf)
            self.lifted += liftings[lf["id"]]
        for rule in pack.get("rules", ()):
            for rec, hit, item in self._candidates(rule, liftings):
                if scope_ok(rule.get("scope"), c.file_id_of(rec), c):
                    self._apply(pack, rule, rec, hit, item, indexes[0] if indexes else None)
        if "launch" in pack:
            self.launch += launch_facts(c, pack)

    @staticmethod
    def _refuse_what_is_not_evaluated(pack: dict) -> None:
        """Refuse, never read as empty or as the first of several: no shipped pack needs these (spec §1.6 ``mounts``, ``embedded``)."""
        mounts = pack.get("mounts", ())
        if len(mounts) > 1:
            raise UnsupportedSubject(f"pack {pack['id']!r} declares {len(mounts)} mounts; the P1-16 runtime composes one")
        for d in mounts:
            if d.get("kind", "call") != "call":
                raise UnsupportedSubject(f"mounts kind {d['kind']!r} is not evaluated by the P1-16 runtime")
        if pack.get("embedded"):
            raise UnsupportedSubject("embedded declarations are not evaluated by the P1-16 runtime")

    def _apply(self, pack, rule, rec, hit, item, mounts: MountIndex | None) -> None:
        c = self.corpus
        anchor = c.symbols[rec.get("targetId") or rec["symbolId"]]
        ctx = Ctx(c, rec, hit, anchor, None if item is None else {
            "url": [a for a in item["alternatives"] if a != UNKNOWN], "verb": item["verbs"] or []})
        for emit in rule["emit"]:
            ((name, body),) = emit.items()
            if name == "mark":
                self.marks.append(self._item(pack, rule, rec, anchor) | {
                    "as": body["as"], "class": body["class"], "status": "confirmed", "reason": None})
            elif name in _EDGES:
                self._emit_edge(pack, rule, rec, anchor, ctx, item, mounts, _EDGES[name], body)
            else:
                raise UnsupportedSubject(f"emission {name!r} is not produced by the P1-16 runtime")

    def _item(self, pack, rule, rec, anchor) -> dict:
        return {"packId": pack["id"], "ruleId": rule["id"], "anchorId": anchor["id"], "qname": anchor["qname"],
                "path": anchor["path"], "siteId": rec["id"], "line": rec["line"]}

    # -- one emission --------------------------------------------------------------------------
    def _emit_edge(self, pack, rule, rec, anchor, ctx, item, mounts, edge, body) -> None:
        kind, template = body["kind"], body["key"]
        names = _PLACEHOLDER.findall(template)
        slots = {n: eval_expr(rule["extract"][n], ctx) for n in names if n != "mount"}
        modes = {n: slot_mode(rule["extract"][n], self._default_mode(edge, kind)) for n in slots}
        variants = mounts.variants(anchor) if "mount" in names and mounts else [None]   # an empty slot gives an empty product: no fact
        for variant in variants:
            mount_alts = mounts.mount_alternatives(variant) if variant else [""]
            for combo in itertools.product(mount_alts, *slots.values()):
                self._one(pack, rule, rec, anchor, item, mounts, edge, body, names, modes, variant, combo, slots)

    @staticmethod
    def _default_mode(edge: str, kind: str) -> str:
        if edge == "CONSUMES":
            return "partial"
        return "config-derived" if kind == "topic" else "drop"

    def _one(self, pack, rule, rec, anchor, item, mounts, edge, body, names, modes, variant, combo, slots) -> None:
        kind = body["kind"]
        values = {"mount": combo[0], **dict(zip(slots, combo[1:]))}
        unresolved = [n for n in names if UNKNOWN in values[n]]
        used_modes = {"drop" if n == "mount" else modes[n] for n in unresolved}
        base = self._item(pack, rule, rec, anchor)
        if "drop" in used_modes:
            self.diagnostics.append({k: base[k] for k in ("packId", "ruleId", "anchorId", "siteId", "line")}
                                    | {"reason": "unresolved-part"})
            return
        if "config-derived" in used_modes:
            raise UnsupportedSubject("onUnknown config-derived is not produced by the P1-16 runtime")
        raw = _PLACEHOLDER.sub(lambda m: values[m.group(1)], body["key"])
        status, reason = "confirmed", ("lifted-role" if item else "key-match")
        key = raw
        if unresolved and (kind != "http" or "runtime-built" in used_modes):
            key, status, reason = None, "possible", "runtime-built"
        elif kind == "http":
            key = normalize_key(raw, "provider" if edge == "PROVIDES" else "consumer")
            if "{?}" in key and unresolved:
                status, reason = "possible", "partial-key"
        if item and item["status"] != "confirmed" and status == "confirmed":
            status, reason = item["status"], item["reason"]
        evidence: dict = {}
        if variant is not None:
            if mounts is not None:
                status, reason = self._weaker((status, reason), mounts.status(variant))
            evidence["mounts"] = [mounts.evidence(variant)]
        if item:
            evidence["hops"] = item["hops"]
            if item["weakenedBy"]:
                evidence["weakenedBy"] = item["weakenedBy"]
        if "status" in body or "reason" in body:
            status, reason = body.get("status", status), body.get("reason", reason)
        fact = base | {"edge": edge, "kind": kind, "key": key, "keyForm": body.get("keyForm", "literal"),
                       "status": status, "reason": reason, "evidence": evidence}
        if edge == "PROVIDES" and rec["type"] == "decorator":
            fact["decorator"] = rec["name"]
        self._collapse(fact)

    @staticmethod
    def _weaker(a: tuple[str, str], b: tuple[str, str]) -> tuple[str, str]:
        return b if _STRENGTH[b[0]] > _STRENGTH[a[0]] else a

    def _collapse(self, fact: dict) -> None:
        """Duplicates collapse on ``(edge, anchor, kind, key, site)``; a ``confirmed`` copy wins; mount variants merge."""
        k = (fact["edge"], fact["anchorId"], fact["kind"], fact["key"], fact["siteId"])
        old = self.edges.get(k)
        if old is None:
            self.edges[k] = fact
            return
        merged = old if _STRENGTH[old["status"]] <= _STRENGTH[fact["status"]] else fact
        other = fact if merged is old else old
        mounts = merged["evidence"].get("mounts", []) + [m for m in other["evidence"].get("mounts", []) if m not in merged["evidence"].get("mounts", [])]
        merged["evidence"] = merged["evidence"] | ({"mounts": mounts} if mounts else {})
        self.edges[k] = merged

    # -- result --------------------------------------------------------------------------------
    def result(self) -> PackFacts:
        for f in self.edges.values():
            if "mounts" in f["evidence"]:
                f["evidence"]["mounts"].sort(key=_canon)
        order = lambda f: (f["kind"], f["key"] or "", f["anchorId"], f["siteId"])
        edges = sorted(self.edges.values(), key=order)
        return PackFacts(
            providers=[f for f in edges if f["edge"] == "PROVIDES"],
            consumers=[f for f in edges if f["edge"] == "CONSUMES"],
            mounts=sorted(self.mounts, key=lambda m: (m["packId"], m["siteId"])),
            lifted=sorted(self.lifted, key=lambda l: (l["packId"], l["lifting"], l["siteId"], l["urlPath"])),
            launch_facts=sorted(self.launch, key=_canon),
            marks=sorted(self.marks, key=_canon),
            diagnostics=sorted(self.diagnostics, key=_canon),
        )


def run_packs(bundles: list[list[dict]], packs: list[dict]) -> PackFacts:
    """Run ``packs`` (each as loaded by :func:`ccg.rulepack.load_pack`) over the unit ``bundles`` (lists of records)."""
    run = _Run(bundles, packs)
    for pack in run.packs:
        run.run_pack(pack)
    return run.result()
