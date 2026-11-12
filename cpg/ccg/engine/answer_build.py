"""The answer builder (C-1001 P1-39b): the walk's output as the canonical answer document, validated by ``ccg.answer`` before it leaves the engine.

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-39 behaviours (0)-(3), (5), (6); ``-answer-spec2.md`` §3 (hops; wiring items §3.3; the ``site`` state §3.6; the ``Unit``
hop §3.5), §4.3 (``builtAt``/``failedAt`` are in the answer, outside a 1.1 answer's ``facts_bytes``), §5 (labelled rows), §6, §7 (V1-V19), §8; ``-answer-spec.md`` §4-§6.
:func:`build_answer` is the one entry point: it resolves nothing (the anchor is what ``ccg.engine.anchors`` returned), runs the walk, groups and folds the items, reads the evidence the
hops carry, and returns the **canonical** document (every array sorted per spec §6 by ``ccg.answer.canonicalize``, never by the engine's raw code-point order).

**The mapping (0).** A hop is the walk's step, field for field: its ``state`` is the walk state's kind (``value``, ``path``, ``function``, ``contract``, ``key``, ``end``, ``site``;
:data:`HOP_STATE`), with ``accessPath`` for ``path`` only (a candidate walk's ``end`` state with the path ``callee`` is ``end``: the path is part of the walk's key, not of the answer);
``edgeType``/``dir``/``status`` are the edge's; ``reason`` is carried by a ``possible`` hop only (a confirmed hop's code is the edge's, not the answer's: the golden's shape, V7);
``candidateCount`` rides every ``VIA`` hop; a ``Link`` hop carries the ``lifted``/``bound`` evidence of the site it was entered for (or, before a wiring ``ConfigEntry`` hop, exactly that
entry); the item's subject, ``repoId``/``unitId`` (the last hop's, V7), ``directness`` and ``viaData`` follow from the chain.

**Build errors, never defaults.** A fact the schema requires and the graph does not give is a :class:`BuildError` (a :class:`~ccg.engine.reader.GraphDataError`), with a static message
that names no value: an unknown ``candidateCount`` (``bfs.candidate_counts`` returns ``None`` for "N unknown", and the schema wants ``>= 1``: never ``1``), an anchor the snapshot does not
hold (the walk itself raises only for an anchor whose flow gate reads its unit; a symbol, contract or config anchor in ``value``/``contract`` state would walk to an empty answer, so the
anchor is re-read first: a stale-snapshot race is an error, never a "no impact" answer), a ``Unit`` hop whose unit is not ``not-mapped``, a snapshot that changes during the build, a
document the validator rejects (the message names the rule id, never a value).

**Determinism (3).** The document holds no clock; the walk, the grouping and the evidence reads are total orders, and the canonical order is ``ccg.answer.canonicalize``'s. The bytes of
``ccg.answer.facts_bytes(doc)`` are therefore identical for one request run twice and across rebuilds from the same commits (a rebuild's ``builtAt`` is outside the facts).

**Canonicalization (6).** ``group`` orders by raw code point, which is **not** the canonical order. Every string is NFC-normalized and every identity string stripped
(``ccg.answer.IDENTITY_KEYS``) before the document is validated, and the arrays are then sorted by ``canonicalize``; the engine order is never the bytes' order.

**The fold's inputs (5).** :data:`FOLD_MEASURED` is the P0 fitness report's headline "P off / on" column (``docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md``
§a1.2), transcribed with its citation; ``group.fold_languages`` derives the set the grouping folds (today ``javascript`` and ``go``). P1-15 has not delivered an indexer: its M2 results,
when one is adopted, replace that language's pairs here (plan P1-39 (5)); ``measured=`` replaces the whole mapping for a caller that has them.

**Documented limit against AC-5: the ``Ret`` summary is context-insensitive** (plan P1-22 (4)). A function reached through a ``Ret`` lists every caller of it as a resolved ``CALLS <-``
hop, not only the call site the change entered by. No P1 acceptance criterion measures it: every P1 language's ``flowAvailable`` is false, so the ``arg(i)``/``Ret`` rows are followed in no
measured unit and flag-true units exist only in fixtures; a flag that flips true by a decision record needs a call-string row, a table change and a version bump first. Pinned by
``tests/test_engine_bfs_rows.py::test_a_changed_argument_enters_the_callee_and_its_ret_summary_reaches_the_other_callers``.

**Facts the walk's reads do not carry, read here through existing texts** (``-p1-graph.md`` §5; no new query): a depends-on symbol item has no ``lineEnd``/``ownedRegions`` (``calls_out``,
``to_out`` return neither), so the builder reads ``symbol_by_qname`` for those symbols only and matches the row by ``symbolId``; a call site reached by a ``DATA`` or ``CONTROLLED_BY`` row has
no ``calleeName`` (those texts return only its line), so ``owner_of:CallSite`` then ``has_out:CallSite`` name it. A site the graph cannot name fails the hop (``BuildError``).

**Candidate tables.** ``table=`` is an engine-internal candidate row table (``ccg.engine.flow.candidate_table``), never a request field; its document carries :data:`CANDIDATE_VERSION`. V18
(``-answer-spec2.md`` §5) encodes the shipped table only, so a candidate walk's added hops (V-flow's labelled ``DATA``/``CONTROLLED_BY``/``arg(i)`` hops, V-both's ``HAS_CALL ->`` exits
from ``function``) are refused with rule V18, never returned: a candidate walk is measured through ``Walk.items()``, and a shipped table other than §5's needs §5 and V18 revised first
(``tests/test_engine_answer_v18.py`` pins exactly which documents are refused and why). P1-39c measures candidate walks through ``Walk.items()`` and needs no document, so ``table=`` and
:data:`CANDIDATE_VERSION` have no caller in P1; they are kept (not removed) to pin V18's refusal of a flag-false candidate-table document, and are rewritten when a shipped table other than §5's is chosen.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ccg import answer
from ccg.engine import anchors as A
from ccg.engine import bfs, config, group
from ccg.engine import staleness as ST
from ccg.engine.reader import GraphDataError, Reader
from ccg.engine.state import Node

#: walk state kind -> the answer's ``Hop.state`` (``-answer-spec.md`` §5.2; ``site``: ``-answer-spec2.md`` §3.6). ``accessPath`` follows ``path`` only (V3).
HOP_STATE: dict[str, str] = {"value": "value", "path": "path", "function": "function", "contract": "contract", "key": "key", "end": "end", "site": "site"}

#: ``recipe.algorithm.version`` of a document built from a candidate row table (``bfs.walk(table=...)``): no published definition, so no published version
CANDIDATE_VERSION = "0.0"

#: the P0 fitness report §a1.2 headline "P off / on" cells, per ``Unit.language`` (TS/JS is ``"javascript"``), as ``(correct, resolved sites)`` per configuration. A cell that gives one
#: figure ("22/25") is one pair; "110/116 / same" is the figure twice. Kotlin is unmeasured in that table, so it is absent (listed, never folded). The override runs of §a1.3 are not read.
FOLD_MEASURED: dict[str, tuple[tuple[int, int], ...]] = {
    "python": ((110, 116), (110, 116)),
    "javascript": ((102, 102), (102, 102)),
    "go": ((108, 108), (108, 108)),
    "java": ((47, 47), (50, 50)),
    "csharp": ((22, 25),),
    "cpp": ((5, 5),),
}

class BuildError(GraphDataError):
    """The answer cannot be built from what the graph holds. The message is static: it names no value read from the graph."""


def _fail(what: str):
    raise BuildError(what)


# ---------------------------------------------------------------------------------------------
# Strings: NFC and strip, before validation (6).
# ---------------------------------------------------------------------------------------------

def clean(value: Any, key: str | None = None) -> Any:
    """``value`` with every string NFC-normalized and every identity string (``ccg.answer.IDENTITY_KEYS``) stripped, recursively; ``key`` is the nearest object key (an array's elements
    inherit it, as in ``ccg.answer``'s V1 check)."""
    if isinstance(value, str):
        value = unicodedata.normalize("NFC", value)
        return value.strip() if key in answer.IDENTITY_KEYS else value
    if isinstance(value, dict):
        return {k: clean(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v, key) for v in value]
    return value


# ---------------------------------------------------------------------------------------------
# The anchor: re-read from the snapshot, then the request's anchor and hop 0.
# ---------------------------------------------------------------------------------------------

def _anchor_parts(anchor: Any) -> tuple[Node, dict[str, Any], dict[str, Any]]:
    """``(walk node, the request's anchor entry, hop 0's facts)`` of a resolved anchor."""
    if isinstance(anchor, A.SymbolAnchor):
        if not (anchor.path and anchor.unit_id):
            _fail("the symbol anchor carries no path or unit")
        entry = {"type": "symbol", "symbolId": anchor.symbol_id, "qname": anchor.qname, "repoId": anchor.repo_id, "path": anchor.path}
        facts = {"repo": anchor.repo_id, "unit": anchor.unit_id, "path": anchor.path, "line": anchor.line_start, "name": anchor.qname}
        return ("Symbol", anchor.symbol_id), entry, facts
    if isinstance(anchor, A.ContractAnchor):
        entry = {"type": "contract", "cpId": anchor.cp_id, "kind": anchor.kind, "keyNorm": anchor.key_norm}
        return ("ContractPoint", anchor.cp_id), entry, {"kind": anchor.kind, "name": anchor.key_norm}
    if isinstance(anchor, A.ConfigAnchor):
        if not anchor.unit_id:
            _fail("the config anchor carries no unit")
        entry = {"type": "config", "entryId": anchor.entry_id, "repoId": anchor.repo_id, "path": anchor.path, "keyPath": anchor.key_path}
        facts = {"repo": anchor.repo_id, "unit": anchor.unit_id, "path": anchor.path, "line": anchor.line, "keyPath": anchor.key_path}
        return ("ConfigEntry", anchor.entry_id), entry, facts
    raise BuildError("the anchor is not a resolved symbol, contract or config anchor")


def _verify_anchor(reader: Reader, anchor: Any) -> None:
    """The anchor must still be in the snapshot: one of the resolver's own reads, matched by id (and by the facts the anchor carries). Absent is an error, never an empty answer."""
    if isinstance(anchor, A.SymbolAnchor):
        rows = [r for r in reader.read("symbol_by_qname", {"repo": anchor.repo_id, "qname": anchor.qname}) if len(r) == 6 and r[0] == anchor.symbol_id]
        ok = len(rows) == 1 and rows[0][1:4] == (anchor.unit_id, anchor.path, anchor.line_start)
    elif isinstance(anchor, A.ContractAnchor):
        rows = [r for r in reader.read("cp_by_key", {"key": anchor.key_norm}) if len(r) == 3 and r[0] == anchor.cp_id]
        ok = len(rows) == 1 and rows[0][1:3] == (anchor.kind, anchor.key_norm)
    else:
        rows = [r for r in reader.read("config_by_path", {"repo": anchor.repo_id, "path": anchor.path}) if len(r) == 5 and r[0] == anchor.entry_id]
        ok = len(rows) == 1 and rows[0][1:5] == (anchor.unit_id, anchor.line, anchor.key_path, None)
    if not ok:
        _fail("the anchor is not in the snapshot")


# ---------------------------------------------------------------------------------------------
# Hops and items.
# ---------------------------------------------------------------------------------------------

def hop_state(state: tuple[str, str]) -> dict[str, str]:
    """``state`` (the walk's ``(kind, path)``) as the hop's ``state`` and, for ``path`` only, ``accessPath``."""
    kind, path = state
    out = {"state": HOP_STATE[kind]}
    if kind == "path":
        out["accessPath"] = path
    return out


def _need(facts: Mapping[str, Any], *keys: str) -> None:
    for k in keys:
        if facts.get(k) is None:
            raise BuildError(f"a hop lacks its {k}")


def _hop(i: int, chain: Sequence[Mapping[str, Any]], facts: Mapping[Node, Mapping[str, Any]], ev: _Evidence) -> dict[str, Any]:
    step = chain[i]
    label, ident = step["node"]
    f = facts.get(step["node"], {})
    hop: dict[str, Any] = {"kind": label, "id": ident, **hop_state(step["state"])}
    if label == "Unit":
        _need(f, "repo", "unit")
        hop["repo"], hop["unit"] = f["repo"], f["unit"]
    elif label not in ("ContractPoint", "Link"):
        _need(f, "repo", "unit", "path", "line")
        hop.update(repo=f["repo"], unit=f["unit"], path=f["path"], line=f["line"])
    if label == "Symbol":
        _need(f, "name")
        hop["qname"] = f["name"]
    elif label == "CallSite":
        _need(f, "name")
        hop["calleeName"] = f["name"]
    elif label == "Cond":
        if f.get("code") is not None:
            hop["code"] = f["code"]
    elif label == "ContractPoint":
        _need(f, "kind", "name")
        hop["contractKind"], hop["keyNorm"] = f["kind"], f["name"]
    elif label == "ConfigEntry":
        _need(f, "keyPath")
        hop["keyPath"] = f["keyPath"]
    e = step["edge"]
    if i:
        hop.update(edgeType=e["type"], dir=e["dir"], status=e["status"])
        if e["status"] == "possible":
            hop["reason"] = e["reason"]
        if e["type"] == "VIA":
            prev = chain[i - 1]["node"][1]
            hop["candidateCount"] = ev.counts[(prev, ident) if e["dir"] == "->" else (ident, prev)]
    if label == "Link":
        _link_evidence(hop, i, chain, facts, ev)
    return hop


def _wiring_bound(entry: Node, facts: Mapping[Node, Mapping[str, Any]]) -> dict[str, Any]:
    """The one ``Bound`` of a wiring item's ``Link`` hop: the entry's own facts and the ``ord``/port evidence of its lowest ``(siteId, ord)`` edge (``-answer-spec2.md`` §3.3)."""
    f = facts[entry]
    _need(f, "repo", "unit", "path", "line", "keyPath")
    b = f.get("bound", {})
    _need(b, "ord")
    return _put({"ord": b["ord"], "entryId": entry[1], "repo": f["repo"], "unit": f["unit"], "path": f["path"], "line": f["line"], "keyPath": f["keyPath"]},
                port=b.get("port"), portOrigin=b.get("portOrigin"), packId=b.get("packId"))


def _link_evidence(hop: dict[str, Any], i: int, chain: Sequence[Mapping[str, Any]], facts, ev: _Evidence) -> None:
    site = _site_of_link(chain, i)
    if site is not None:
        key = (chain[i]["node"][1], site)
        if ev.lifted.get(key):
            hop["lifted"] = ev.lifted[key]
        if ev.bound.get(key):
            hop["bound"] = ev.bound[key]
        return
    nxt = chain[i + 1] if i + 1 < len(chain) else None
    if nxt is not None and nxt["edge"] is not None and (nxt["edge"]["type"], nxt["edge"]["dir"]) == ("BOUND_BY", "->"):
        hop["bound"] = [_wiring_bound(nxt["node"], facts)]


def _site_of_link(chain: Sequence[Mapping[str, Any]], i: int) -> str | None:
    """The call site a ``Link`` hop was entered for (V5): the preceding ``CallSite`` when the hop arrived by ``VIA ->`` (depends-on), the following one when it is left by ``VIA <-``
    (impact). ``None`` before a wiring ``ConfigEntry`` hop: no site was entered."""
    e = chain[i]["edge"]
    if e is not None and (e["type"], e["dir"]) == ("VIA", "->"):
        return chain[i - 1]["node"][1]
    if i + 1 < len(chain):
        nx = chain[i + 1]["edge"]
        if nx is not None and (nx["type"], nx["dir"]) == ("VIA", "<-"):
            return chain[i + 1]["node"][1]
    return None


def _malformed(name: str) -> GraphDataError:
    return GraphDataError(f"query {name!r} returned a malformed row")


def _str(name: str, v: Any, *, none: bool = False) -> Any:
    if v is None and none:
        return None
    if not isinstance(v, str) or not v:
        raise _malformed(name)
    return v


def _int(name: str, v: Any, *, none: bool = False) -> Any:
    if v is None and none:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise _malformed(name)
    return v


def _put(d: dict[str, Any], **kw: Any) -> dict[str, Any]:
    d.update({k: v for k, v in kw.items() if v is not None})
    return d


def _lifted(r: tuple) -> dict[str, Any]:
    n = "lifted_of"
    if len(r) != 14:
        raise _malformed(n)
    out = {"ord": _int(n, r[2]), "siteId": _str(n, r[1]), "kind": _str(n, r[3]), "id": _str(n, r[4]), "repo": _str(n, r[5], none=True), "unit": _str(n, r[6], none=True),
           "path": _str(n, r[7], none=True), "line": _int(n, r[8], none=True), "urlPath": _str(n, r[9], none=True), "name": _str(n, r[13], none=True)}
    out = {k: v for k, v in out.items() if v is not None}
    return _put(out, verbPath=_str(n, r[10], none=True), viaSiteId=_str(n, r[11], none=True), viaLine=_int(n, r[12], none=True))


def _bound(r: tuple) -> dict[str, Any]:
    n = "bound_out"
    if len(r) != 19:
        raise _malformed(n)
    out = _put({}, ord=_int(n, r[2]), entryId=_str(n, r[3]), repo=_str(n, r[4], none=True), unit=_str(n, r[5], none=True), path=_str(n, r[6], none=True),
               line=_int(n, r[7], none=True), keyPath=_str(n, r[8], none=True))
    kind = _str(n, r[9], none=True)
    if kind is not None:
        out["launch"] = _put({"launchKind": kind}, protocol=_str(n, r[10], none=True), port=_int(n, r[11], none=True), envVar=_str(n, r[12], none=True),
                             mountPath=_str(n, r[13], none=True), service=_str(n, r[14], none=True), argv0=_str(n, r[15], none=True))
    return _put(out, port=_int(n, r[16], none=True), portOrigin=_str(n, r[17], none=True), packId=_str(n, r[18], none=True))


class _Evidence:
    """What the hops of a set of chains need beyond the walk: ``candidateCount`` of every ``VIA`` hop and the ``lifted``/``bound`` entries of every ``Link`` hop's site, each read once."""

    def __init__(self, reader: Reader, chains: Iterable[Sequence[Mapping[str, Any]]]):
        count_pairs: set[tuple[str, str]] = set()      # (site, link)
        link_pairs: set[tuple[str, str]] = set()       # (link, site)
        for chain in chains:
            for i, step in enumerate(chain):
                e = step["edge"]
                if e is not None and e["type"] == "VIA":
                    prev, here = chain[i - 1]["node"][1], step["node"][1]
                    count_pairs.add((prev, here) if e["dir"] == "->" else (here, prev))
                if step["node"][0] == "Link":
                    site = _site_of_link(chain, i)
                    if site is not None:
                        link_pairs.add((step["node"][1], site))
        counts = bfs.candidate_counts(reader, sorted(count_pairs))
        if any(n is None for n in counts.values()):
            _fail("the candidate count of a link is unknown")
        self.counts: dict[tuple[str, str], int] = counts
        pairs = [list(p) for p in sorted(link_pairs)]
        self.lifted: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.bound: dict[tuple[str, str], list[dict[str, Any]]] = {}
        if pairs:
            for r in reader.read("lifted_of", {"pairs": pairs}):
                self.lifted.setdefault((_str("lifted_of", r[0]), _str("lifted_of", r[1])), []).append(_lifted(r))
            for r in reader.read("bound_out", {"pairs": pairs}):
                self.bound.setdefault((_str("bound_out", r[0]), _str("bound_out", r[1])), []).append(_bound(r))


def _subject(item: bfs.Item, facts: Mapping[Node, Mapping[str, Any]], spans: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    label = {"symbol": "Symbol", "config": "ConfigEntry", "unit": "Unit"}[item.kind]
    f = facts[(label, item.id)]
    if item.kind == "unit":
        return {"type": "unit", "unitId": item.id}
    if item.kind == "config":
        _need(f, "path", "line", "keyPath")
        return {"type": "config", "entryId": item.id, "path": f["path"], "line": f["line"], "keyPath": f["keyPath"]}
    _need(f, "name", "path", "line")
    span = {**{k: f[k] for k in ("lineEnd", "ownedRegions") if k in f}, **spans.get(item.id, {})}
    _need(span, "lineEnd")
    subject = {"type": "symbol", "symbolId": item.id, "qname": f["name"], "path": f["path"], "lineStart": f["line"], "lineEnd": span["lineEnd"]}
    if span.get("ownedRegions") is not None:
        subject["ownedRegions"] = list(span["ownedRegions"])
    return subject


def _item(p: group.Placed, facts, spans, ev: _Evidence) -> dict[str, Any]:
    item = p.item
    chain = [_hop(i, item.chain, facts, ev) for i in range(len(item.chain))]
    out: dict[str, Any] = {"subject": _subject(item, facts, spans), "repoId": chain[-1]["repo"], "unitId": chain[-1]["unit"], "status": item.status}
    if item.status == "possible":
        out["reason"] = item.reason
    out.update(directness=p.directness, distance=item.distance, viaData=item.via_data, chain=chain)
    return out


def _site_names(reader: Reader, chains: Iterable[Sequence[Mapping[str, Any]]], facts: dict[Node, dict[str, Any]]) -> None:
    """``calleeName`` of the call sites on the chains that the walk's reads did not give one: a site a ``DATA`` or ``CONTROLLED_BY`` row reaches carries only its ``line`` (those texts
    return no callee name), and the schema requires it on every ``CallSite`` hop. Filled through two existing reads, never a new text: ``owner_of:CallSite`` for the site's owner and
    ``has_out:CallSite`` for that owner's sites (matched by id). A site the graph cannot name stays without one, and the hop then fails as a build error."""
    wanted = sorted({step["node"][1] for chain in chains for step in chain if step["node"][0] == "CallSite" and facts.get(step["node"], {}).get("name") is None})
    if not wanted:
        return
    owners = set()
    for r in reader.read("owner_of:CallSite", {"ids": wanted}):
        if len(r) != 9:
            raise _malformed("owner_of:CallSite")
        owners.add(_str("owner_of:CallSite", r[1]))
    for r in reader.read("has_out:CallSite", {"ids": sorted(owners)}):
        if len(r) != 4:
            raise _malformed("has_out:CallSite")
        site, name = _str("has_out:CallSite", r[1]), _str("has_out:CallSite", r[3], none=True)
        if name is not None and site in wanted:
            facts.setdefault(("CallSite", site), {})["name"] = name


def _spans(reader: Reader, items: Iterable[bfs.Item], facts: Mapping[Node, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """``lineEnd``/``ownedRegions`` of the symbol items whose facts lack them (a callee or provider: ``calls_out``/``to_out`` read no span), by the ``symbol_by_qname`` read, matched by id."""
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        f = facts.get(("Symbol", it.id), {})
        if it.kind != "symbol" or f.get("lineEnd") is not None or it.id in out:
            continue
        _need(f, "repo", "name")
        for r in reader.read("symbol_by_qname", {"repo": f["repo"], "qname": f["name"]}):
            if len(r) != 6:
                raise GraphDataError("query 'symbol_by_qname' returned a malformed row")
            if r[0] == it.id and r[4] is not None:
                out[it.id] = {"lineEnd": r[4], **({"ownedRegions": r[5]} if r[5] is not None else {})}
    return out


# ---------------------------------------------------------------------------------------------
# The build.
# ---------------------------------------------------------------------------------------------

def build_answer(reader: Reader, anchor: Any, *, question: str = "impact", measured: Mapping[str, Iterable[tuple[int, int]]] | None = None,
                 table: Sequence[bfs.Row] | None = None) -> dict[str, Any]:
    """The validated, canonical answer document for ``question`` (``"impact"`` or ``"depends-on"``) asked of ``anchor`` (a ``ccg.engine.anchors`` symbol, contract or config anchor).

    ``measured`` replaces :data:`FOLD_MEASURED`; ``table`` is an engine-internal candidate row table (never a request field): such a document carries :data:`CANDIDATE_VERSION`.
    Raises :class:`BuildError` (or the reader's ``GraphDataError``/``ReaderError``) and never returns a document ``ccg.answer.validate`` rejects."""
    if question not in group.QUESTIONS:
        raise ValueError(f"unknown question {question!r}"[:64])
    node, request_anchor, anchor_facts = _anchor_parts(anchor)
    stamp0 = ST.stamp(reader)
    _verify_anchor(reader, anchor)
    w = bfs.walk(reader, node, None, column=group.QUESTIONS[question], table=table)
    walked = w.items()
    folds = group.fold_languages(FOLD_MEASURED if measured is None else measured)
    refs = group.unit_refs(reader, group.needed_units(walked, w.facts, folding=bool(folds)))
    listing = group.group_items(walked, w.facts, refs, folds)
    placed = list(listing.items) + [m for g in listing.groups for m in g.members]
    facts = {k: dict(v) for k, v in w.facts.items()}
    facts[node] = {**facts.get(node, {}), **anchor_facts}
    spans = _spans(reader, (p.item for p in placed), facts)
    _site_names(reader, (p.item.chain for p in placed), facts)
    ev = _Evidence(reader, (p.item.chain for p in placed))
    items = [_item(p, facts, spans, ev) for p in listing.items]
    groups = [{"kind": g.kind, "key": g.key, "reason": g.reason, "count": g.count, "members": [_item(m, facts, spans, ev) for m in g.members]} for g in listing.groups]
    evidence_units = {x["unit"] for it in items + [m for g in groups for m in g["members"]] for h in it["chain"] for x in (*h.get("lifted", ()), *h.get("bound", ())) if "unit" in x}
    staleness = ST.staleness_block(reader, (p.item for p in placed), facts, evidence_units)
    stamp = ST.stamp(reader)
    if stamp != stamp0:
        _fail("the snapshot changed during the build")
    doc = {
        "answerSchemaVersion": "1.1",
        "interpretation": {"status": "resolved", "request": {"useCase": "impact", "direction": question, "anchors": [request_anchor]}},
        "recipe": {"algorithm": {"id": "bfs-v1", "version": CANDIDATE_VERSION if table is not None else w.version()},
                   "queries": reader.recipe_queries(), "guards": {"visitedSeedsAnchor": True, "levelGuard": config.LEVEL_GUARD}},
        "stamp": stamp,
        "staleness": staleness,
        "items": items,
        "groups": groups,
        "depthBound": {"bound": w.bound, "unexplored": w.unexplored},
    }
    doc = clean(doc)
    try:
        answer.validate(doc)
        return answer.canonicalize(doc)
    except answer.AnswerError as exc:
        raise BuildError(f"the built answer does not validate (rule {str(exc).split(':', 1)[0][:8]})") from None
