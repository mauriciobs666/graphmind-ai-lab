"""``link(facts, bundles) -> Links`` (P1-17; plan ``docs/plans/connected-code-graph-p1-linking.md``, graph contract
``docs/plans/connected-code-graph-p1-graph.md`` §1-§2, bundle spec3 BF6).

Pack facts (``PROVIDES`` and ``CONSUMES``, plus the fully unknown consumers that live only in ``PackFacts.lifted``) and the unit
bundles in; reified :class:`Link` objects out. One ``Link`` per (kind, provider key, provider anchor, consumer *symbol*): the consumer's
call sites that reach it, whatever their consumer keys, are its :class:`Via` entries, each with its own consumer key, status and
reason.

**Status.** A site's status is the weakest hop on its chain (consumer fact, provider fact, the key match, an example file, a
shared port, a never-built target); ``Link.status`` is the best over its sites (graph note §2: "stays best over its sites").
**Reason (O-9).** A site's ``VIA`` reason is ``binding-confirmed`` iff the site is ``confirmed`` and the link has a ``BOUND_BY``
entry carrying that site's ``siteId``; a confirmed unbound site keeps ``key-match``; a possible site keeps its own reason.
``Link.reason`` is ``binding-confirmed`` iff some confirmed site is, else the reason of the site that sets the best status
(ties: lowest ``siteId``). The reason of a possible site is the first cause in this order: ``example-config`` (the consumer's or
the provider's file has ``role = example``), ``shell-dynamic`` (the consumer *call site's* own reason; the ``CONSUMES`` edge
cannot carry it), ``target-not-mapped``, ``port-shared``, a binding's own ``weaken`` reason, the consumer fact's own reason (``prompt-mention`` or, when it is
not a ``Link`` reason such as ``runtime-built`` on a known key, ``partial-key``), the provider fact's, then the match's
(``partial-key``, ``name-only-provider``, ``unknown-key``, ``binding-narrowed``).

**Keys.** ``Link.key_norm`` (and the ``linkId`` input) is the *provider's* ``keyNorm``, the ``ContractPoint`` the link is ``ON``;
``""`` for a keyless or name-only provider. The consumer's key lives on each :class:`Via`. A link into a never-built unit has no
provider, so its ``key_norm`` is the consumer's key (``""`` if keyless) and its anchor is ``"unit:"+unitId`` (or
``"unmapped:"+ref`` for a path outside every manifest repo). A key reaching here is already ``keyNorm`` and is never normalized
again (:mod:`ccg.linker.matchers`). ``Link.matcher`` is that of the best site (lowest ``siteId`` on a tie): ``exact``,
``path-template``, ``glob``, ``prefix``, ``regex``, ``name`` (name-only provider) or ``none`` (no key comparison ran).

**Kinds and matchers.** ``http`` links by ``path-template`` (its ``keyNorm`` carries the template syntax) and every other kind
by ``exact``, unless a pack's ``link`` declaration says otherwise; two packs declaring different matchers for one kind is an
error, never a pick. A pattern key (``keyForm``) is matched by its own form.

**Unknown keys.** A consumer with no key links to *every* provider of its kind, each its own ``Link`` (``unknown-key``), none
collapsed (a ``binding-narrowed`` link when a binding selects units). A fully unknown consumer whose key never reaches
``consumers`` is read from ``lifted`` (``resolved: False``, ``leaf: True``); its kind comes from the pack rule whose ``match``
names that lifting, so ``packs`` must be passed whenever such a leaf exists.

**Prompt mentions.** Only a *tool* mention is a consumer: a ``mcp__<server>__<tool>`` name, or a bare name (a leading ``./`` is
stripped before matching, the extractor keeps names as written) that equals a provider's tool. Script and command mentions name
no provider and stay ``CallSite`` nodes; a ``doc``-role file's sites are never callers.

The bindings seam (``Bindings``) is how dispatch 2's ``bind.py`` plugs in: see :class:`Bindings`.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from ccg import ids
from ccg.linker.matchers import Match, match_keys
from ccg.packrt.extract import Corpus
from ccg.packrt.run import PackFacts
from ccg.rulepack import load_registry_rows

__all__ = ["Binding", "BoundEntry", "Bindings", "Link", "Links", "Unmapped", "Via", "count_links", "link"]

_STRENGTH = {"confirmed": 0, "possible": 1}
_EVIDENCE_ORDER = {"LIFTED_VIA": 0, "BOUND_BY": 1}
# Reasons a Link or VIA may carry as a consumer fact's own (registry rows on Link, plus the two the plan names for it).
_OWN_REASONS = frozenset({"prompt-mention", "shell-dynamic", "partial-key", "example-config"})
_MCP_TOOL = re.compile(r"mcp__[A-Za-z0-9][A-Za-z0-9-]{0,63}__([A-Za-z0-9_][A-Za-z0-9_-]{0,63})")


@dataclass(frozen=True)
class BoundEntry:
    """One ``BOUND_BY`` target: a ``ConfigEntry`` id plus the port evidence it supplied (``-p1-graph.md`` §2)."""
    entry_id: str
    port: int | None = None
    port_origin: str | None = None      # explicit | env-default | compose | pack-default
    pack_id: str | None = None          # iff port_origin == 'pack-default'


def _link_reason_codes() -> frozenset[str]:
    """The registry codes whose ``on`` list holds ``Link`` (``Link (TO_UNIT)`` counts: it is a ``Link`` row)."""
    return frozenset(r["code"] for r in load_registry_rows() if any(t == "Link" or t.startswith("Link ") for t in r["on"]))


@dataclass(frozen=True)
class Unmapped:
    """The chain ends outside the built code: ``unit_id`` is a never-built unit, or ``None`` for a path outside every manifest
    repo (trust rule b). ``ref`` is a stable, value-free id: the unit id in the first case, ``tuple_id(["unmapped", normalised
    path])`` in the second; it is the ``linkId`` anchor (``"unit:"+ref`` or ``"unmapped:"+ref``)."""
    unit_id: str | None
    ref: str


@dataclass(frozen=True)
class Binding:
    """What a binding chain says about one consumer site.

    ``units``: the units the chain selects. One unit selects it (a confirmed key match through it is
    ``binding-confirmed``); several (two units on one port) make the link ``possible``/``port-shared``.
    ``entries``: the chain, ``ord`` 0 the entry nearest the consumer; never empty (a binding with no entry has no ``BOUND_BY``
    edge, so by O-9 it could not change a reason).
    ``unmapped``: the chain ends outside the built code, so there is no provider: the link has no ``TO``, is
    ``possible``/``target-not-mapped``, and ``Link.to_unit`` is ``unmapped.unit_id`` (``TO_UNIT`` iff not ``None``).
    ``weaken``: ``("possible", <reason>)`` for a possible hop inside the chain (the inferred Compose rule, trust rule e's loopback
    residual); the reason must be a registry code valid on ``Link``. It ranks after ``port-shared``.
    """
    units: tuple[str, ...] = ()
    entries: tuple[BoundEntry, ...] = ()
    unmapped: Unmapped | None = None
    weaken: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("a Binding needs at least one chain entry")
        if self.weaken is not None:
            status, reason = self.weaken
            if status != "possible" or reason not in _link_reason_codes():
                raise ValueError(f"weaken must be ('possible', <a registry reason valid on Link>), got {self.weaken!r}")


class Bindings(Protocol):
    """The seam dispatch 2's ``bind.py`` implements (URL, env/Compose and launch binding).

    ``bind(consumer)`` is called once per consumer *site* (``consumer["siteId"]``; the dict also has ``kind``, ``key`` (or
    ``None``), ``anchorId``, ``line``) and returns a :class:`Binding`, or ``None`` when no chain selects a unit: the site then
    degrades to the unbound rules (``key-match`` for a confirmed literal match, ``unknown-key`` for no key). A binding
    restricts the candidate providers to its ``units``; a key that matches a provider outside them makes no link.
    ``key_mismatches()`` (optional) returns the ``bind-key-mismatch <unitId>`` lines for the refresh report (BR-1 to BR-3);
    they are returned beside the links. The provider must resolve lexically and never dial or open a path (trust rule a).
    """

    def bind(self, consumer: dict) -> Binding | None: ...


class Via(NamedTuple):
    """One consumer call site's ``VIA`` entry on a ``Link``."""
    site_id: str
    line: int
    status: str
    reason: str
    consumer_key: str | None


class Link(NamedTuple):
    link_id: str
    kind: str
    key_norm: str
    provider: str | None          # provider anchor (symbol id or entry id); None for a link into a never-built unit
    consumer: str                 # consumer anchor: the enclosing symbol (or a config entry id)
    status: str
    reason: str
    matcher: str
    param_map: dict               # no P1 pack declares a parameter map: always empty
    to_unit: str | None           # the never-built unit of a target-not-mapped link (None for a path outside every manifest repo)
    sites: tuple[Via, ...]
    evidence: tuple[dict, ...]    # ordered LIFTED_VIA then BOUND_BY entries per site, each with its ``siteId`` and ``ord``


class Links(NamedTuple):
    links: list[Link]
    bind_key_mismatch: list[str]
    counts: dict


# -- consumers and providers ----------------------------------------------------------------------
def _matchers_by_kind(packs) -> dict[str, str]:
    out: dict[str, tuple[str, str]] = {}
    for pack in sorted(packs, key=lambda p: p["id"]):
        for decl in pack.get("link", ()):
            prev = out.get(decl["kind"])
            if prev and prev[0] != decl["matcher"]:
                raise ValueError(f"packs {prev[1]!r} and {pack['id']!r} declare different link matchers for {decl['kind']!r}")
            out[decl["kind"]] = (decl["matcher"], pack["id"])
    return {k: v[0] for k, v in out.items()}


def _lifted_kinds(packs) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pack in packs:
        for rule in pack.get("rules", ()):
            match = rule["match"]
            for clause in match["anyOf"] if "anyOf" in match else [match]:
                body = clause.get("lifted")
                if body is None:
                    continue
                for emit in rule["emit"]:
                    kind = emit.get("consumes", {}).get("kind")
                    if kind and kind not in out[(pack["id"], body["lifting"])]:
                        out[(pack["id"], body["lifting"])].append(kind)
    return out


def _consumer(fact: dict, hops=None) -> dict:
    return {"siteId": fact["siteId"], "anchorId": fact["anchorId"], "kind": fact["kind"], "key": fact["key"],
            "keyForm": fact.get("keyForm", "literal"), "line": fact["line"], "status": fact["status"],
            "reason": fact["reason"], "hops": (fact.get("evidence") or {}).get("hops", []) if hops is None else hops}


def _unknown_lifted(facts: PackFacts, packs) -> list[dict]:
    out, kinds = [], None
    for item in facts.lifted:
        if item["resolved"] or not item["leaf"]:
            continue
        kinds = _lifted_kinds(packs) if kinds is None else kinds
        names = kinds.get((item["packId"], item["lifting"]))
        if not names:
            what = (f"no rule of the given packs consumes lifting {item['lifting']!r} (pack {item['packId']!r})" if packs
                    else f"the unresolved lifted leaf of {item['packId']}/{item['lifting']} needs packs= naming the rule "
                         "that consumes that lifting; none were passed")
            raise ValueError(f"link(): {what}")
        out += [{"siteId": item["siteId"], "anchorId": item["symbolId"], "kind": k, "key": None, "keyForm": "literal",
                 "line": item["line"], "status": "possible", "reason": "runtime-built", "hops": item["hops"]} for k in names]
    return out


def _tool_name(callee: str) -> str:
    m = _MCP_TOOL.fullmatch(callee)
    if m:
        return m.group(1)
    while callee.startswith("./"):
        callee = callee[2:]
    return callee


def _mentions(corpus: Corpus) -> list[dict]:
    return [{"siteId": cid, "anchorId": c["symbolId"], "kind": "mcp_tool", "key": _tool_name(c["calleeName"]),
             "keyForm": "literal", "line": c["line"], "status": "possible", "reason": "prompt-mention", "hops": []}
            for cid, c in sorted(corpus.callsites.items()) if c["callKind"] == "mention"]


class _Index:
    """Anchor to file role / unit / name, from the bundles.

    A symbol anchor resolves through its file. A ``configEntry``/``launchFact`` anchor (a record id) takes its bundle header's
    ``unitId`` (neither record carries one); its ``repoId`` is its own (``configEntry``) or ``header.unit.repoId``
    (``launchFact`` has none), and its role is ``example`` iff a ``configEntry``'s ``example`` is true or its file record, looked up
    by ``(repoId, path)``, has ``role = example``; otherwise that file's role.
    """

    def __init__(self, corpus: Corpus, bundles: list[list[dict]]):
        self.c = corpus
        self._entries: dict[str, tuple[str | None, str | None]] = {}
        for records in bundles:
            header = next((r for r in records if r["type"] == "header"), None)
            unit = header["unitId"] if header else None
            repo = header["unit"]["repoId"] if header else None
            roles = {(f.get("repoId"), f["path"]): f["role"] for f in records if f["type"] == "file"}
            for r in records:
                if r["type"] in ("configEntry", "launchFact"):
                    file_role = roles.get((r.get("repoId", repo), r["path"]))
                    self._entries[r["id"]] = (unit, "example" if r.get("example") else file_role)

    def role(self, anchor: str) -> str | None:
        if anchor in self._entries:
            return self._entries[anchor][1]
        s = self.c.symbols.get(anchor)
        f = self.c.files.get(s["fileId"]) if s else None
        return f["role"] if f else None

    def unit(self, anchor: str) -> str | None:
        if anchor in self._entries:
            return self._entries[anchor][0]
        s = self.c.symbols.get(anchor)
        return self.c.unit_of_file(s["fileId"]) if s else None

    def name(self, anchor: str) -> str | None:
        s = self.c.symbols.get(anchor)
        return s["name"] if s else None


# -- one consumer against the providers -----------------------------------------------------------
def _candidates(c: dict, providers: list[dict], matcher: str, ix: _Index, binding: Binding | None, refused: set):
    """``(provider fact, Match-or-reason)`` for each provider of ``c``'s kind that the key (and any binding) admits."""
    for p in providers:
        if binding is not None and ix.unit(p["anchorId"]) not in binding.units:
            continue
        if c["key"] is None:
            yield p, Match("none", True), ("binding-narrowed" if binding is not None else "unknown-key")
        elif p["key"] is None:
            if ix.name(p["anchorId"]) == c["key"]:
                yield p, Match("name", True), "name-only-provider"
        else:
            got = match_keys(c["kind"], matcher, p["key"], c["key"], provider_form=p.get("keyForm", "literal"),
                             consumer_form=c["keyForm"], refused=refused)
            if got is not None:
                yield p, got, "partial-key"


def _own(reason: str) -> str:
    return reason if reason in _OWN_REASONS else "partial-key"


def _causes(c: dict, p: dict | None, match: Match, match_reason: str, ix: _Index, corpus: Corpus,
            binding: Binding | None) -> list[tuple[str, str]]:
    """The weakening hops of one site, in the order the module docstring gives (reasons of a ``possible`` site)."""
    causes: list[tuple[str, str]] = []
    site = corpus.callsites.get(c["siteId"])
    if ix.role(c["anchorId"]) == "example" or (p is not None and ix.role(p["anchorId"]) == "example"):
        causes.append(("possible", "example-config"))
    if site is not None and site.get("reason") == "shell-dynamic":
        causes.append(("possible", "shell-dynamic"))
    if binding is not None and binding.unmapped is not None:
        causes.append(("possible", "target-not-mapped"))
    elif binding is not None and len(set(binding.units)) > 1:
        causes.append(("possible", "port-shared"))
    if binding is not None and binding.weaken is not None:
        causes.append(binding.weaken)
    if c["status"] != "confirmed" and c["key"] is not None:
        causes.append(("possible", _own(c["reason"])))
    if p is not None and p["key"] is not None and p["status"] != "confirmed":
        causes.append(("possible", _own(p["reason"])))
    if match.partial:
        causes.append(("possible", match_reason))
    return causes


def link(facts: PackFacts, bundles: list[list[dict]], *, packs=(), bindings: Bindings | None = None) -> Links:
    """Link ``facts`` over ``bundles``; ``packs`` (as loaded by ``load_pack``) supply ``link`` matchers and the kind of a lifted
    leaf; ``bindings`` is the seam of :class:`Bindings`."""
    corpus, packs = Corpus(bundles), list(packs)
    ix = _Index(corpus, bundles)
    refused: set = set()
    matchers = {"http": "path-template"} | _matchers_by_kind(packs)
    providers: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(facts.providers, key=lambda f: (f["kind"], f["key"] or "", f["anchorId"], f["siteId"])):
        providers[p["kind"]].append(p)
    consumers = [_consumer(f) for f in facts.consumers] + _unknown_lifted(facts, packs) + _mentions(corpus)
    consumers = [c for c in consumers if ix.role(c["anchorId"]) != "doc"]
    consumers.sort(key=lambda c: (c["siteId"], c["kind"], c["key"] or "", c["anchorId"]))

    bound: dict[str, Binding | None] = {}
    groups: dict[str, dict] = {}
    for c in consumers:
        if bindings is not None and c["siteId"] not in bound:
            bound[c["siteId"]] = bindings.bind(c)
        binding = bound.get(c["siteId"])
        if binding is not None and binding.unmapped is not None:
            _add(groups, c, None, Match("none", False), "partial-key", ix, corpus, binding)
            continue
        for p, match, why in _candidates(c, providers.get(c["kind"], ()), matchers.get(c["kind"], "exact"), ix, binding, refused):
            _add(groups, c, p, match, why, ix, corpus, binding)
    links = sorted((_build(lid, g) for lid, g in groups.items()),
                   key=lambda l: (l.kind, l.key_norm, l.provider or l.to_unit or "", l.consumer, l.link_id))
    mismatches = list(bindings.key_mismatches()) if bindings is not None and hasattr(bindings, "key_mismatches") else []
    return Links(links, mismatches, count_links(links, len(refused)))


def _add(groups, c, p, match, why, ix, corpus, binding) -> None:
    unmapped = binding.unmapped if binding is not None else None
    if p is None:
        anchor = ("unit:" if unmapped.unit_id is not None else "unmapped:") + unmapped.ref
        key = c["key"] if c["key"] is not None else ""            # no provider: the consumer's key is the only one
    else:
        anchor, key = p["anchorId"], p["key"] or ""                # the ContractPoint the link is ON
    lid = ids.link_id(c["kind"], key, anchor, c["anchorId"])
    g = groups.setdefault(lid, {"kind": c["kind"], "key": key, "provider": None if p is None else p["anchorId"],
                                "consumer": c["anchorId"], "to_unit": None if unmapped is None else unmapped.unit_id, "sites": {}})
    causes = _causes(c, p, match, why, ix, corpus, binding)
    status, reason = causes[0] if causes else ("confirmed", "key-match")
    k = (c["siteId"], c["key"] or "")
    prev = g["sites"].get(k)
    if prev is not None and _STRENGTH[prev["status"]] <= _STRENGTH[status]:
        return
    g["sites"][k] = {"consumer": c, "status": status, "reason": reason, "binding": binding, "matcher": match.matcher}


def _evidence(sid: str, site: dict) -> list[dict]:
    out = [{"edge": "LIFTED_VIA", "siteId": sid} | dict(h) for h in site["consumer"]["hops"]]
    seen: set[str] = set()
    b = site["binding"]
    for e in (b.entries if b is not None else ()):
        if e.entry_id in seen:                       # one edge per (link, site, entry), at the entry's lowest ord
            continue
        seen.add(e.entry_id)
        rec = {"edge": "BOUND_BY", "siteId": sid, "ord": len(seen) - 1, "entryId": e.entry_id}
        for k, v in (("port", e.port), ("portOrigin", e.port_origin), ("packId", e.pack_id)):
            if v is not None:
                rec[k] = v
        out.append(rec)
    return out


def _build(lid: str, g: dict) -> Link:
    order = sorted(g["sites"])                        # (siteId, consumer key)
    first_of_site: dict[str, dict] = {}
    for sid, _ in order:
        first_of_site.setdefault(sid, g["sites"][(sid, _)])
    evidence = [e for sid, s in sorted(first_of_site.items()) for e in _evidence(sid, s)]     # one chain per site, not per key
    evidence.sort(key=lambda e: (e["siteId"], _EVIDENCE_ORDER[e["edge"]], e["ord"]))
    bound_sites = {e["siteId"] for e in evidence if e["edge"] == "BOUND_BY"}
    vias = []
    for k in order:
        s = g["sites"][k]
        reason = "binding-confirmed" if s["status"] == "confirmed" and k[0] in bound_sites else s["reason"]
        vias.append((Via(k[0], s["consumer"]["line"], s["status"], reason, s["consumer"]["key"]), s["matcher"]))
    best, matcher = min(vias, key=lambda v: (_STRENGTH[v[0].status], v[0].site_id, v[0].consumer_key or ""))
    reason = "binding-confirmed" if any(v.status == "confirmed" and v.reason == "binding-confirmed" for v, _ in vias) else best.reason
    return Link(lid, g["kind"], g["key"], g["provider"], g["consumer"], best.status, reason,
                matcher, {}, g["to_unit"], tuple(v for v, _ in vias), tuple(evidence))


# -- (11) counts ----------------------------------------------------------------------------------
def count_links(links: list[Link], refused_patterns: int = 0) -> dict:
    """Totals by kind, status and reason, and the unknown-key fan-out: how many consumer sites had no key and the most
    providers one of them links to (the "Link counts under runtime-built URLs at scale" figure)."""
    fan: Counter = Counter()
    for l in links:
        for v in l.sites:
            if v.consumer_key is None:
                fan[v.site_id] += 1
    return {"links": len(links),
            "by_kind": dict(sorted(Counter(l.kind for l in links).items())),
            "by_status": dict(sorted(Counter(l.status for l in links).items())),
            "by_reason": dict(sorted(Counter(l.reason for l in links).items())),
            "unknown_key_consumers": len(fan), "max_fan_out": max(fan.values(), default=0),
            "refused_patterns": refused_patterns}
