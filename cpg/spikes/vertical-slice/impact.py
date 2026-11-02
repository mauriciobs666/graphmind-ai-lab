"""S13 (P0-12) -- the slice's impact walk: a minimal level-synchronous BFS with a state per entry.

Contract: graph note ``docs/plans/connected-code-graph-graph.md`` §6.1 (the BFS carries a state per
frontier entry), §6.2 (the normative edge x direction table), §6.3 (``DATA {fromPath, toPath}``, access
paths); bundle spec v1.3 §5.1 (paths), §5.10 (``DATA``); P0 plan ``docs/plans/connected-code-graph-p0-2.md``
S13 row, SL-4, SL-5 (M4).

Only the rows of §6.2 the slice needs are implemented, and nothing else is followed:

==============================  =====================================================================
``CALLS <-``                    callee changed -> call site (distance +1)
``HAS_CALL/PARAM/RETURN/COND <-``  point -> enclosing function (reported item, distance +0)
``DATA <-`` (path-matched)      def changed at path ``P`` -> every use whose ``fromPath`` overlaps ``P``
``TO <-``, ``VIA <-``           provider changed -> link -> consumer call sites (distance +1)
``PROVIDES ->``                 a provider whose key changed switches to ``contract`` mode
``PROVIDES <-`` (contract)      the contract point -> its providers (the start of a contract walk)
==============================  =====================================================================

Function-level continuation (``docs/plans/connected-code-graph-p0-graph.md`` 3.3, finding M2): ``bfs-v1`` has
no ``arg(i)``/``Ret``-summary rows, so a function entered from a call site, parameter or condition
(``HAS_CALL|PARAM|COND <-``, state ``function``) is only *assumed* to pass the change on. Its ``CALLS <-`` and
``TO <-`` hops are ``possible`` with reason ``function-level-continuation``. A function entered from a ``Ret``
(``HAS_RETURN <-``, state ``value``) and a ``CALLS <-``/``TO <-`` from the anchor itself stay ``confirmed``.

Not followed: ``CALLS_NAME``/``RESOLVES_TO`` (the name hop), ``CONTROLLED_BY``, ``READS``/``WRITES``,
``BOUND_BY``, ``CALLS ->`` into a callee, ``HAS_PARAM ->``, the request-direction ``VIA ->``.

State per entry: ``("value", "")`` the node itself changed; ``("path", P)`` the value at access path
``P`` of this point changed; ``("contract", "")`` a contract mode walk; ``("key", "")`` a provider whose
key changed; ``("end", "")`` a reported item that is not walked further.

Cost of a path is ``(possible hops, distance, length)``, compared in that order; ties break on the
predecessor's stable key. A node is re-expanded only when a later arrival has a strictly smaller
``(possible, distance)`` (the walk is level-synchronous, so length never improves later), and the
``visited`` set plus ``max_levels`` keep a cycle (G-3) from looping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

Node = tuple[str, str]  # (label, id)
State = tuple[str, str]  # (kind, path)
Key = tuple[str, str, str, str]  # (label, id, state kind, state path)
Cost = tuple[int, int, int]  # (possible hops, distance, length)

POINTS = ("CallSite", "Param", "Ret", "Cond")
DEF_LABELS = ("CallSite", "Param")  # a DATA edge's target is only ever one of these (bundle spec §5.10)
OWNER_REL = {"CallSite": "HAS_CALL", "Param": "HAS_PARAM", "Ret": "HAS_RETURN", "Cond": "HAS_COND"}
ID_PROP = {"Symbol": "symbolId", "CallSite": "callSiteId", "Param": "pointId", "Ret": "pointId", "Cond": "pointId",
           "Link": "linkId", "ContractPoint": "cpId"}
RESULT_SET_LIMIT = 10_000  # GRAPH.CONFIG RESULTSET_SIZE on ccg-falkordb: a longer reply is silently truncated
CHUNK = 1_000

VALUE = ("value", "")
CONTRACT = ("contract", "")
KEY = ("key", "")
END = ("end", "")
FUNCTION = ("function", "")  # a function entered from a non-Ret point: it continues only at function level
FUNCTION_LEVEL = "function-level-continuation"


class ImpactError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# Access-path matching (BF4).
# ---------------------------------------------------------------------------------------------


def path_overlaps(state_path: str, edge_path: str) -> bool:
    """Whether a change at ``state_path`` and a use of ``edge_path`` touch the same value.

    Equal, or one is a segment prefix of the other (``p0`` covers ``p0.path``; ``p0.path`` covers
    ``p0.path.x``). ``p1`` is not a prefix of ``p10``, and ``p0.path`` does not touch ``p0.method``.
    """
    if state_path == edge_path:
        return True
    shorter, longer = sorted((state_path, edge_path), key=len)
    return longer.startswith(shorter) and longer[len(shorter)] in ".["


def data_edge_matches(state: State, from_path: str) -> bool:
    """``value`` (the whole point changed) matches every edge; ``path`` only an overlapping one."""
    return state[0] == "value" or path_overlaps(state[1], from_path)


# ---------------------------------------------------------------------------------------------
# The reader seam.
# ---------------------------------------------------------------------------------------------


class Reader(Protocol):
    """One method per (edge type, direction) the slice follows; each takes a batch of ids."""

    def calls_in(self, symbol_ids: list[str]) -> list[tuple[str, str, str]]: ...  # (symbol, callsite, status)
    def owner_of(self, label: str, ids: list[str]) -> list[tuple[str, str]]: ...  # (point, symbol)
    def data_in(self, label: str, ids: list[str]) -> list[tuple[str, str, str, str, str, str]]: ...
    # (def id, use label, use id, fromPath, toPath, status)
    def to_in(self, symbol_ids: list[str]) -> list[tuple[str, str, str]]: ...  # (symbol, link, link status)
    def via_in(self, link_ids: list[str]) -> list[tuple[str, str, str]]: ...  # (link, callsite, status)
    def provides_in(self, cp_ids: list[str]) -> list[tuple[str, str]]: ...  # (cp, symbol)
    def provides_out(self, symbol_ids: list[str]) -> list[tuple[str, str]]: ...  # (symbol, cp)


class CypherReader:
    """The live reader: one parameterized, read-only query per call, chunked, never silently truncated."""

    def __init__(self, run: Callable[[str, dict], list[list]]):
        self.run = run

    def _rows(self, cypher: str, ids: list[str]) -> list[list]:
        out: list[list] = []
        for i in range(0, len(ids), CHUNK):
            part = self.run(cypher, {"ids": ids[i:i + CHUNK]})
            if len(part) >= RESULT_SET_LIMIT:
                raise ImpactError(f"reply reached RESULTSET_SIZE ({RESULT_SET_LIMIT}); it may be truncated")
            out.extend(part)
        return sorted(out)

    def calls_in(self, symbol_ids):
        return [tuple(r) for r in self._rows(
            "UNWIND $ids AS i MATCH (c:CallSite)-[e:CALLS]->(s:Symbol {symbolId: i}) "
            "RETURN i, c.callSiteId, coalesce(e.status, 'confirmed')", symbol_ids)]

    def owner_of(self, label, ids):
        rel = OWNER_REL[label]
        return [tuple(r) for r in self._rows(
            f"UNWIND $ids AS i MATCH (s:Symbol)-[:{rel}]->(p:{label} {{{ID_PROP[label]}: i}}) "
            "RETURN i, s.symbolId", ids)]

    def data_in(self, label, ids):
        if label not in DEF_LABELS:
            return []
        return [tuple(r) for r in self._rows(
            f"UNWIND $ids AS i MATCH (u)-[d:DATA]->(p:{label} {{{ID_PROP[label]}: i}}) "
            "RETURN i, labels(u)[0], coalesce(u.callSiteId, u.pointId), d.fromPath, d.toPath, coalesce(d.status, 'confirmed')", ids)]

    def to_in(self, symbol_ids):
        return [tuple(r) for r in self._rows(
            "UNWIND $ids AS i MATCH (l:Link)-[:TO]->(s:Symbol {symbolId: i}) RETURN i, l.linkId, l.status",
            symbol_ids)]

    def via_in(self, link_ids):
        return [tuple(r) for r in self._rows(
            "UNWIND $ids AS i MATCH (c:CallSite)-[v:VIA]->(l:Link {linkId: i}) RETURN i, c.callSiteId, v.status",
            link_ids)]

    def provides_in(self, cp_ids):
        return [tuple(r) for r in self._rows(
            "UNWIND $ids AS i MATCH (s:Symbol)-[:PROVIDES]->(cp:ContractPoint {cpId: i}) RETURN i, s.symbolId",
            cp_ids)]

    def provides_out(self, symbol_ids):
        return [tuple(r) for r in self._rows(
            "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:PROVIDES]->(cp:ContractPoint) RETURN i, cp.cpId",
            symbol_ids)]


# ---------------------------------------------------------------------------------------------
# The walk.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    type: str
    direction: str  # "<-" against the stored direction, "->" along it
    status: str
    reason: str | None = None

    def as_dict(self) -> dict:
        out = {"type": self.type, "dir": self.direction, "status": self.status}
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass
class Entry:
    cost: Cost
    pred: Key | None
    edge: Edge | None


@dataclass
class Item:
    symbol_id: str
    distance: int
    possible_hops: int
    status: str
    length: int
    chain: list[dict]  # [{"node": (label, id), "state": (kind, path), "edge": {...} | None}], anchor first

    @property
    def hops(self) -> int:
        return len(self.chain) - 1

    @property
    def via_data(self) -> bool:
        """Whether the best path holds a ``DATA`` hop (never read as "path-matched" for a function-level item)."""
        return any(s["edge"] and s["edge"]["type"] == "DATA" for s in self.chain)

    @property
    def reason(self) -> str | None:
        """``function-level-continuation`` when a hop of the best path rests on it, else ``None``."""
        return next((s["edge"]["reason"] for s in self.chain if s["edge"] and s["edge"].get("reason")), None)


@dataclass
class Result:
    anchor: Key
    levels: int
    frontier_sizes: list[int]
    visited: set[Key]
    best: dict[Key, Entry]

    @property
    def peak_frontier(self) -> int:
        return max(self.frontier_sizes, default=0)

    @property
    def state_count(self) -> int:
        return len(self.visited)

    @property
    def field_path_states(self) -> int:
        """Visited states whose path has a field segment (``p0.path``, ``a1.method``): M4's count."""
        return sum(1 for k in self.visited if k[2] == "path" and ("." in k[3] or "[" in k[3]))

    def states_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for k in self.visited:
            out[k[2]] = out.get(k[2], 0) + 1
        return dict(sorted(out.items()))

    def reached(self, label: str) -> dict[str, tuple[Cost, list[State]]]:
        """``id -> (best cost, states)`` of every visited node with this label."""
        out: dict[str, tuple[Cost, list[State]]] = {}
        for k in sorted(self.visited):
            if k[0] != label:
                continue
            cost = self.best[k].cost
            prev = out.get(k[1])
            states = (prev[1] if prev else []) + [(k[2], k[3])]
            out[k[1]] = (min(cost, prev[0]) if prev else cost, states)
        return out

    def chain(self, key: Key) -> list[dict]:
        steps: list[dict] = []
        cur: Key | None = key
        while cur is not None:
            e = self.best[cur]
            steps.append({"node": (cur[0], cur[1]), "state": (cur[2], cur[3]), "edge": e.edge.as_dict() if e.edge else None})
            cur = e.pred
        steps.reverse()
        return steps

    def items(self) -> list[Item]:
        """One item per impacted ``Symbol``.

        An impacted symbol is one entered from one of its points (``HAS_* <-``): a function whose body
        holds a reached point. The anchor and a provider reached by ``PROVIDES <-`` (the contract's own
        side) are not items; nor is the anchor, however a cycle re-enters it. A symbol entered in two states
        (`value` and `function`) is one item, on its cheaper arrival.
        """
        picked: dict[str, tuple[Cost, Key]] = {}
        for k in sorted(self.best):
            e = self.best[k]
            if k[0] != "Symbol" or e.edge is None or e.edge.type not in OWNER_REL.values() or k[:2] == self.anchor[:2]:
                continue
            if k[1] not in picked or e.cost < picked[k[1]][0]:
                picked[k[1]] = (e.cost, k)
        return [Item(k[1], c[1], c[0], "possible" if c[0] else "confirmed", c[2], self.chain(k))
                for c, k in (picked[i] for i in sorted(picked))]


def _stable(key: Key | None) -> str:
    return "" if key is None else json.dumps(key)


def _status_hops(status: str) -> int:
    return 0 if status == "confirmed" else 1


def _edge(type_: str, direction: str, status: str, from_kind: str) -> Edge:
    """An edge whose possible status comes from leaving a function-level state carries the reason."""
    return Edge(type_, direction, status, FUNCTION_LEVEL if from_kind == "function" else None)


def _expand(reader: Reader, frontier: list[Key]) -> list[tuple[Key, Node, State, Edge, int, int]]:
    """All candidate steps from the frontier: ``(from key, to node, to state, edge, +possible, +distance)``."""
    out: list[tuple[Key, Node, State, Edge, int, int]] = []
    groups: dict[tuple[str, str], list[Key]] = {}
    for k in frontier:
        groups.setdefault((k[0], k[2]), []).append(k)

    def by_id(keys: list[Key]) -> dict[str, list[Key]]:
        m: dict[str, list[Key]] = {}
        for k in keys:
            m.setdefault(k[1], []).append(k)
        return m

    for (label, kind), keys in sorted(groups.items()):
        ids = sorted({k[1] for k in keys})
        src = by_id(keys)

        def emit(i: str, node: Node, state: State, edge: Edge, dposs: int, ddist: int) -> None:
            for k in src[i]:
                out.append((k, node, state, edge, dposs, ddist))

        if label == "Symbol":
            if kind in ("value", "function"):
                for sid, cs, status in reader.calls_in(ids):  # CALLS <-
                    if kind == "function":
                        status = "possible"
                    emit(sid, ("CallSite", cs), ("path", "r"), _edge("CALLS", "<-", status, kind), _status_hops(status), 1)
            if kind in ("value", "function", "contract"):
                mode = CONTRACT if kind == "contract" else VALUE
                for sid, link, status in reader.to_in(ids):  # TO <-
                    status = "possible" if kind == "function" else "confirmed"
                    emit(sid, ("Link", link), mode, _edge("TO", "<-", status, kind), _status_hops(status), 0)
            if kind == "key":
                for sid, cp in reader.provides_out(ids):  # PROVIDES ->
                    emit(sid, ("ContractPoint", cp), CONTRACT, Edge("PROVIDES", "->", "confirmed"), 0, 0)
        elif label in POINTS:
            for pid, sid in reader.owner_of(label, ids):  # HAS_* <-
                to_state = END if kind == "contract" else VALUE if label == "Ret" else FUNCTION
                emit(pid, ("Symbol", sid), to_state, Edge(OWNER_REL[label], "<-", "confirmed"), 0, 0)
            if kind in ("value", "path") and label in DEF_LABELS:
                for did, ulabel, uid, from_path, to_path, status in reader.data_in(label, ids):  # DATA <-
                    for k in src[did]:
                        if data_edge_matches((k[2], k[3]), from_path):
                            out.append((k, (ulabel, uid), ("path", to_path), Edge("DATA", "<-", status),
                                        _status_hops(status), 0))
        elif label == "Link":
            for lid, cs, status in reader.via_in(ids):  # VIA <-
                to_state = ("path", "r") if kind == "value" else CONTRACT
                emit(lid, ("CallSite", cs), to_state, Edge("VIA", "<-", status), _status_hops(status), 1)
        elif label == "ContractPoint":
            for cp, sid in reader.provides_in(ids):  # PROVIDES <- (contract)
                emit(cp, ("Symbol", sid), CONTRACT, Edge("PROVIDES", "<-", "confirmed"), 0, 0)
    return out


def walk(reader: Reader, anchor: Node, state: State = VALUE, *, max_levels: int = 64) -> Result:
    """Level-synchronous BFS from ``anchor`` in ``state``; raises :class:`ImpactError` past ``max_levels``."""
    start: Key = (anchor[0], anchor[1], state[0], state[1])
    best: dict[Key, Entry] = {start: Entry((0, 0, 0), None, None)}
    visited: set[Key] = {start}
    frontier: list[Key] = [start]
    sizes: list[int] = []
    level = 0
    while frontier:
        level += 1
        if level > max_levels:
            raise ImpactError(f"walk exceeded {max_levels} levels (a cycle the visited set did not stop?)")
        sizes.append(len(frontier))
        arrivals: dict[Key, tuple[Cost, str, Key, Edge]] = {}
        for frm, node, st, edge, dposs, ddist in _expand(reader, sorted(frontier)):
            key: Key = (node[0], node[1], st[0], st[1])
            base = best[frm].cost
            cost: Cost = (base[0] + dposs, base[1] + ddist, level)
            cand = (cost, _stable(frm), frm, edge)
            if key not in arrivals or cand[:2] < arrivals[key][:2]:
                arrivals[key] = cand
        nxt: list[Key] = []
        for key in sorted(arrivals):
            cost, _, frm, edge = arrivals[key]
            if key in visited and not cost[:2] < best[key].cost[:2]:
                continue
            visited.add(key)
            best[key] = Entry(cost, frm, edge)
            nxt.append(key)
        frontier = nxt
    return Result(start, level, sizes, visited, best)


# ---------------------------------------------------------------------------------------------
# Rendering: a Link hop expands to its LIFTED_VIA chain (graph note §3.3, §6.3).
# ---------------------------------------------------------------------------------------------

LIFTED_QUERY = (
    "MATCH (l:Link {linkId: $link})-[r:LIFTED_VIA]->(t) WHERE r.siteId = $site "
    "RETURN r.ord, labels(t)[0], coalesce(t.callSiteId, t.symbolId), t.path, r.line, r.urlPath, r.verbPath, "
    "r.viaSiteId, r.viaLine, coalesce(t.calleeName, t.qname) ORDER BY r.ord"
)


def lifted_chain(run: Callable[[str, dict], list[list]], link_id: str, site_id: str) -> list[dict]:
    """The ordered ``LIFTED_VIA`` hops of one consumer site on a link: consumer call first, seed last."""
    rows = run(LIFTED_QUERY, {"link": link_id, "site": site_id})
    return [{"ord": r[0], "label": r[1], "id": r[2], "path": r[3], "line": r[4], "urlPath": r[5], "verbPath": r[6],
             "viaSiteId": r[7], "viaLine": r[8], "name": r[9]} for r in rows]


def render_item(run: Callable[[str, dict], list[list]], item: Item) -> list[dict]:
    """The item's chain with every ``Link`` hop followed by its expanded ``LIFTED_VIA`` hops.

    The consumer site is the call site entered by the ``VIA`` step that follows the link.
    """
    out: list[dict] = []
    steps = item.chain
    for i, step in enumerate(steps):
        out.append(step)
        label, ident = step["node"]
        if label == "Link" and i + 1 < len(steps) and steps[i + 1]["node"][0] == "CallSite":
            site = steps[i + 1]["node"][1]
            out.extend({"lifted": h} for h in lifted_chain(run, ident, site))
    return out


def state_summary(result: Result) -> dict[str, Any]:
    return {
        "levels": result.levels, "peakFrontier": result.peak_frontier, "frontierSizes": result.frontier_sizes,
        "states": result.state_count, "fieldPathStates": result.field_path_states,
        "statesByKind": result.states_by_kind(),
    }

