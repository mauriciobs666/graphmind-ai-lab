"""Grouping, folding and the two questions over the walk's items (C-1001 P1-23c, behaviours (2)-(4); ``docs/plans/connected-code-graph-p1-answers.md`` P1-23).

Contract: ``-answer-spec.md`` §5.1 (``directness``), §5.4 (a folded group is display only and enumerates every member as a full item), §6 (the ``items`` order, the
``groups`` order); ``-answer-spec2.md`` §3.5 (a ``unit`` item sorts by its ``rootPath`` in the path slot and 0 in the line slot); ``docs/plans/connected-code-graph.md`` §3.7
and ``-p0-ml.md`` §8.3 (the fold rule). This module builds neither the answer document nor the builder (P1-39): it orders what the walk found and decides what folds.

**Distance (2)** is the walk's: :attr:`ccg.engine.bfs.Item.distance` is the sum of the increments of the chain's rows, which are 1 for exactly the inter-function hops
(call, link, shared state, config) and 0 for every step inside a function (``-answer-spec2.md`` §1.2; pinned in ``tests/test_engine_bfs_pins.py``). Nothing here
recomputes it; :class:`Placed` carries it and :func:`directness` reads it (``direct`` exactly at 1).

**Order (3).** :func:`group_items` places every item under ``(repoId, rootPath of its unit, unitId, distance, confirmed before possible, subject path, line, subject id)``,
the §6 rule, so the listing is grouped repository, component, distance (:meth:`Listing.sections`). Every key is compared as a Python ``str`` (code points: never a locale's
collation, never normalized, so a composed and a decomposed spelling are two keys), and no step reads an insertion order: the same items in any order, with their facts
and units in any order, give the same listing. A key is graph data and only compared, never interpreted. **This is the order of raw code points, not the canonical order**
(NFC strings, ``ccg.answer.canonicalize``): P1-39 normalizes before emission and must not treat this order as the answer's bytes.

**The fold.** An item is another candidate of a *resolved* call site only when its **own arrival** is the candidate hop (:func:`candidate_site`): its first ``possible`` hop is
``CALLS_NAME`` with reason ``resolution-unverified`` (the walk derives that reason from ``count(x)`` of the ``name_in``/``name_out`` read, ``ccg.engine.bfs``) and nothing
follows it but the arrival itself (depends-on: the last hop; impact: the one ``HAS_CALL <-`` owner step). An item past a candidate (a caller of it, its own callee) stays
in ``items``, ``possible``/``resolution-unverified`` by its first possible hop, ungrouped (``-answer-spec2.md`` §6, the §5.4 row). A site that resolves to nothing gives
``name-match``, never folded. When the **language of the site's unit** is in ``fold_languages`` the candidates of one site become one group (``other-possible-callees``,
reason ``resolution-unverified``, key the site's id), each member a full item with its own chain; otherwise they stay in ``items``. **A fold moves an item, it never drops
one**: the items and the members together are exactly the input, each once.

**The precision figure stops here.** ``fold_languages`` is the only input the fold reads, a set of ``Unit.language`` values (TS/JS is ``"javascript"``): :func:`fold_languages`
derives it from per-language measured ``(correct, resolved sites)`` pairs by the rule of ``-p0-ml.md`` §8.3 (at or above :data:`FOLD_THRESHOLD`, over at least
:data:`MIN_RESOLVED_SITES` sites, in every configuration measured; a language not measured is listed). The pairs are supplied by P1-39 behaviour (5) (the P0 fitness
report §a1.2; P1-15's M2 for an adopted indexer), not by this module; :class:`FoldGroup` and :class:`Listing` carry no figure, so the renderers receive none.

**The reverse question (4)** is the ``depends-on`` column of the same walk, marked and grouped by the same code (:data:`QUESTIONS`, :func:`listing`).
"""
from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, NamedTuple

from ccg.engine import bfs
from ccg.engine.reader import GraphDataError, Reader
from ccg.engine.state import Node

#: the questions the engine answers and the walk column each one is (the reverse question, "what does this depend on?", is the ``depends-on`` column)
QUESTIONS: dict[str, str] = {"impact": bfs.IMPACT, "depends-on": bfs.DEPENDS_ON}

#: the one fold/list threshold (``docs/plans/connected-code-graph.md`` §3.7): exact, so a measured 57/60 is at it and 94/99 is below it
FOLD_THRESHOLD = Fraction(19, 20)
#: fewer resolved sites than this is *unmeasured* and listed, never folded (``-p0-ml.md`` §8.3)
MIN_RESOLVED_SITES = 60

GROUP_KIND = "other-possible-callees"
GROUP_REASON = "resolution-unverified"

_LABEL = {"symbol": "Symbol", "config": "ConfigEntry", "unit": "Unit"}


class UnitRef(NamedTuple):
    root: str            # ``Unit.rootPath``: the component label the grouping uses
    lang: str | None     # ``Unit.language``


# ---------------------------------------------------------------------------------------------
# The fold decision.
# ---------------------------------------------------------------------------------------------

def _count(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError("a measured count is an int")
    return v


def folds(correct: int, resolved: int) -> bool:
    """Whether one measurement folds: at least :data:`MIN_RESOLVED_SITES` resolved sites and ``correct / resolved`` at or above :data:`FOLD_THRESHOLD` (exact arithmetic)."""
    correct, resolved = _count(correct), _count(resolved)
    if not 0 <= correct <= resolved:
        raise ValueError("a measurement has 0 <= correct <= resolved")
    return resolved >= MIN_RESOLVED_SITES and Fraction(correct, resolved) >= FOLD_THRESHOLD


def fold_languages(measured: Mapping[str, Iterable[tuple[int, int]]]) -> frozenset[str]:
    """The languages whose other candidates fold: every measurement of the language (one per configuration) folds, and there is at least one. Every pair is checked,
    so a malformed one raises whatever the others say."""
    out = set()
    for lang, pairs in measured.items():
        verdicts = [folds(c, n) for c, n in pairs]
        if verdicts and all(verdicts):
            out.add(lang)
    return frozenset(out)


# ---------------------------------------------------------------------------------------------
# Placed items, groups, sections.
# ---------------------------------------------------------------------------------------------

def directness(distance: int) -> str:
    """``direct`` exactly when the distance is 1 (``-answer-spec.md`` §5.1)."""
    return "direct" if distance == 1 else "indirect"


@dataclass(frozen=True)
class Placed:
    """A walk item with its grouping keys: ``repo``, the unit and its ``root`` (the component), and the ``path``/``pos`` slots of the §6 order."""
    item: bfs.Item
    repo: str
    unit: str
    root: str
    path: str
    pos: int

    @property
    def distance(self) -> int:
        return self.item.distance

    @property
    def status(self) -> str:
        return self.item.status

    @property
    def directness(self) -> str:
        return directness(self.item.distance)

    @property
    def chain(self) -> list[dict]:
        return self.item.chain

    def sort_key(self) -> tuple:
        return (self.repo, self.root, self.unit, self.item.distance, 0 if self.item.status == "confirmed" else 1, self.path, self.pos, self.item.id)


@dataclass(frozen=True)
class FoldGroup:
    """One folded group: the other candidates of the resolved call site ``key``; every member is a full :class:`Placed` with its own chain."""
    kind: str
    key: str
    reason: str
    members: tuple[Placed, ...]

    @property
    def count(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class Section:
    """A run of items sharing repository, component (unit) and distance, confirmed first."""
    repo: str
    root: str
    unit: str
    distance: int
    items: tuple[Placed, ...]


@dataclass(frozen=True)
class Listing:
    """``items`` in the §6 order and ``groups`` by ``(kind, key)``, members in the items order."""
    items: tuple[Placed, ...]
    groups: tuple[FoldGroup, ...]

    def sections(self) -> tuple[Section, ...]:
        runs: list[tuple[tuple, list[Placed]]] = []
        for p in self.items:
            key = (p.repo, p.root, p.unit, p.distance)
            if runs and runs[-1][0] == key:
                runs[-1][1].append(p)
            else:
                runs.append((key, [p]))
        return tuple(Section(*key, tuple(members)) for key, members in runs)


def _bad(what: str) -> GraphDataError:
    return GraphDataError(f"grouping: {what}")


def candidate_site(item: bfs.Item) -> str | None:
    """The id of the resolved call site whose other candidate ``item`` is, or ``None``. ``item``'s **own arrival** must be the candidate hop: its first ``possible`` hop is
    ``CALLS_NAME`` with reason ``resolution-unverified`` and nothing follows it but the arrival itself. Depends-on (``->``): the hop is the chain's last hop and the site is
    its source. Impact (``<-``): the hop arrives at the site, followed by exactly one step, the ``HAS_CALL <-`` owner hop. An item past a candidate (a caller of it, its own
    callee) is not a candidate of that site (``-answer-spec2.md`` §6, the §5.4 row) and stays listed. A hop with no single ``CallSite`` end is ambiguous: ``None``."""
    chain = item.chain
    for i, step in enumerate(chain):
        e = step["edge"]
        if e is None or e["status"] != "possible":
            continue
        if e["type"] != "CALLS_NAME" or e.get("reason") != GROUP_REASON:
            return None
        tail = chain[i + 1:]
        if e["dir"] == "<-":
            if len(tail) != 1 or tail[0]["edge"] is None or (tail[0]["edge"]["type"], tail[0]["edge"]["dir"]) != ("HAS_CALL", "<-"):
                return None
        elif tail:
            return None
        ends = [s["node"] for s in chain[max(i - 1, 0):i + 1] if s["node"][0] == "CallSite"]
        return ends[0][1] if len(ends) == 1 else None
    return None


def needed_units(items: Iterable[bfs.Item], facts: Mapping[Node, Mapping[str, Any]], *, folding: bool) -> set[str]:
    """The unit ids :func:`group_items` looks up: every item's, and with ``folding`` every candidate site's."""
    need: set[str] = set()
    for item in items:
        u = facts.get((_LABEL[item.kind], item.id), {}).get("unit")
        if isinstance(u, str):
            need.add(u)
        site = candidate_site(item) if folding else None
        if site is not None:
            su = facts.get(("CallSite", site), {}).get("unit")
            if isinstance(su, str):
                need.add(su)
    return need


def _place(item: bfs.Item, facts: Mapping[Node, Mapping[str, Any]], units: Mapping[str, UnitRef]) -> Placed:
    f = facts.get((_LABEL[item.kind], item.id))
    if f is None:
        raise _bad("an item has no facts")
    repo, unit = f.get("repo"), f.get("unit")
    if not isinstance(repo, str) or not isinstance(unit, str):
        raise _bad("an item lacks its repository or unit")
    ref = units.get(unit)
    if ref is None:
        raise _bad("an item names a unit that has no Unit row")
    if item.kind == "unit":
        return Placed(item, repo, unit, ref.root, ref.root, 0)
    path, line = f.get("path"), f.get("line")
    if not isinstance(path, str) or isinstance(line, bool) or not isinstance(line, int):
        raise _bad("an item lacks its path or line")
    return Placed(item, repo, unit, ref.root, path, line)


def group_items(items: Iterable[bfs.Item], facts: Mapping[Node, Mapping[str, Any]], units: Mapping[str, UnitRef], fold_languages: Collection[str] = frozenset()) -> Listing:
    """Order ``items`` (a walk's :meth:`~ccg.engine.bfs.Walk.items`, with its ``facts``) by the §6 rule and fold the other candidates of the resolved call sites whose
    language is in ``fold_languages``. ``units`` maps every unit id of :func:`needed_units` to its :class:`UnitRef`. A fold moves items into groups and drops none."""
    placed = [_place(i, facts, units) for i in items]
    plain: list[Placed] = []
    folded: dict[str, list[Placed]] = {}
    for p in placed:
        site = candidate_site(p.item) if fold_languages else None
        if site is not None and _folds_site(site, facts, units, fold_languages):
            folded.setdefault(site, []).append(p)
        else:
            plain.append(p)
    plain.sort(key=Placed.sort_key)
    groups = tuple(FoldGroup(GROUP_KIND, key, GROUP_REASON, tuple(sorted(members, key=Placed.sort_key))) for key, members in sorted(folded.items()))
    return Listing(tuple(plain), groups)


def _folds_site(site: str, facts: Mapping[Node, Mapping[str, Any]], units: Mapping[str, UnitRef], languages: Collection[str]) -> bool:
    unit = facts.get(("CallSite", site), {}).get("unit")
    if not isinstance(unit, str):
        return False   # a site that cannot say its unit cannot say its language: listed
    ref = units.get(unit)
    if ref is None:
        raise _bad("a call site names a unit that has no Unit row")
    return ref.lang is not None and ref.lang in languages


def unit_refs(reader: Reader, ids: Iterable[str]) -> dict[str, UnitRef]:
    need = sorted(set(ids))
    out: dict[str, UnitRef] = {}
    if not need:
        return out
    for row in reader.read("units", {"ids": need}):
        if len(row) != 10 or not isinstance(row[0], str) or not isinstance(row[2], str) or not (row[3] is None or isinstance(row[3], str)):
            raise GraphDataError("query 'units' returned a malformed row")
        out[row[0]] = UnitRef(row[2], row[3])
    return out


def listing(reader: Reader, anchor: Node, state: tuple[str, str] | None = None, *, question: str = "impact", fold_languages: Collection[str] = frozenset(),
            bound: int | None = None, table=None) -> Listing:
    """Walk ``question``'s column from ``anchor`` and return the grouped listing. ``question`` is a key of :data:`QUESTIONS`; the reverse question is ``"depends-on"``."""
    if question not in QUESTIONS:
        raise ValueError(f"unknown question {question!r}")
    w = bfs.walk(reader, anchor, state, column=QUESTIONS[question], bound=bound, table=table)
    items = w.items()
    refs = unit_refs(reader, needed_units(items, w.facts, folding=bool(fold_languages)))
    return group_items(items, w.facts, refs, fold_languages)
