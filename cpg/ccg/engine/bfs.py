"""The answer engine's walk (C-1001 P1-22): a level-synchronous BFS with a state per frontier entry, driven by a row table that is data.

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-22; the row table of ``docs/plans/connected-code-graph-graph.md`` §6.2 and ``-p0-graph.md`` §3
(states, visited set, cost); ``-answer-spec2.md`` §1.2 (the version is the fingerprint of the walk definition), §3.3 (wiring items), §3.5 (``TO_UNIT``), §4
(depth bound, ``unexplored``), §5; ``-p1-graph.md`` §5 (the reads). The walk reads only through ``ccg.engine.reader.Reader.read(<name>, {"ids": [...]})`` (and
the typed ``bound_of_link``); the reader sorts and bounds, the walk interprets: ``count(x)`` of ``name_in``/``name_out`` decides ``name-match`` versus
``resolution-unverified`` here and nowhere else.

**The table is the walk.** :data:`ROWS` is a tuple of :class:`Row` records; :func:`walk` interprets it and has no edge type, direction, state or increment of its own.
A row added to the code without a table entry cannot exist; a row added to the table changes :func:`definition`, whose fingerprint no longer equals a pinned
:data:`ALGORITHM_VERSIONS` literal, so :func:`algorithm_version` raises (``-answer-spec2.md`` §1.2). The fingerprint cannot see a code change to the cost order,
the ``unexplored`` rule or an adapter that leaves the identifier strings alone; the pinned walk tests hold those.

**Cost** of a chain is ``(possible hops, distance, length, own-possible hops)`` compared in that order, then the predecessor's stable key. **The bounded result is
everything reachable within the bound** (``-answer-spec2.md`` §4.1): at each state the walk keeps every *arrival* that no other arrival dominates (``a`` dominates
``b`` when it has no more possible hops and no greater distance; an equal pair is dominated, so a cycle ends), expands the state once per kept arrival, and shows the
lexicographic minimum of the kept arrivals as ``best``. An :class:`Entry` holds its predecessor's entry object, so every item's chain recomputes to the item's own cost.
The visited set is seeded with the anchor and keyed ``(label, id, state kind, state path)``; with the level guard it stops every cycle (``DATA`` self-edge and
cycles, cond-to-cond ``CONTROLLED_BY``, recursion; F-12, C1).

**Distance** counts inter-function hops (call, link, shared state, config) and steps inside a function count 0 (graph note §6.2). The bound
(``config.DEFAULT_DEPTH_BOUND``) never cuts an increment-0 row; an increment-1 row from an arrival already at the bound is not followed, its stored edges are kept on
that arrival, and :attr:`Walk.unexplored` counts the distinct stored edges of the final kept arrivals (``-answer-spec2.md`` §4.2).

**A config entry reaches only the sites it binds** (``-p1-graph.md`` G7, §5): ``BOUND_BY <-`` enters the ``Link`` in a ``site(siteId)`` state and ``VIA <-`` from it lists
that site's ``CallSite`` only. A ``Link`` is never an anchor (:class:`WalkError`).

Not built here (named, never half-built):
  * SEAM(P1-23): the flow-fallback gate on ``Unit.flowAvailable`` (``DATA``/``CONTROLLED_BY``/``arg(i)`` rows not followed in a flag-false unit, the ``function`` state
    and its labelled ``flow-unavailable:<lang>`` rows, the V-flow/V-both/V-union row-table variants), item grouping and the builder. Every unit is walked with the
    precise rows. ``Unit`` facts are not read here; ``walk.facts`` carries the unit ids. P1-23 splits the hop-status and own-edge-status variable (marked SEAM below).
  * SEAM(P1-23/P1-39): ``Link`` evidence (``lifted_of``, ``bound_out``, ``candidate_count``), the ``Hop``/``Item`` documents, ``staleness`` and the stamp.
  * **Rows of graph note §6.2 not walked in P1** (the plan's "as written" is narrowed to these four, each on purpose): depends-on ``BOUND_BY ->`` (a ``Link`` is left only by
    ``TO ->``/``TO_UNIT ->``, spec2 §3.3 V5), depends-on ``CONSUMES ->`` (no item kind carries it), ``VIA ->`` in ``arg(i)`` state (needs ``paramMap``, not stored,
    ``-p1-graph.md`` G12), and from a ``Param``, ``<-`` then ``CALLS <-`` (P5's RCA). Also not walked: ``READS``/``WRITES``, ``DEFINED_BY``, ``LAUNCHES``, ``RAN``,
    ``DEPENDS_ON``/``FROM_DEP`` (no P1 hop; ``-answer-spec2.md`` §3), ``MOUNTED_BY``.
  * **The ``arg(i)`` summary is context-insensitive** (accepted for P1, plan P1-22 (4)): a function reached through a ``Ret`` lists every caller of it as a resolved
    ``CALLS <-`` hop, not only the call site the change entered by. Pinned by ``test_a_changed_argument_enters_the_callee_and_its_ret_summary_reaches_the_other_callers``
    (``tests/test_engine_bfs_rows.py``). A flag that flips true needs a call-string row first (a table change and a version bump).
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ccg import ids as _ids
from ccg.engine import config
from ccg.engine import state as S
from ccg.engine.reader import GraphDataError, Reader
from ccg.engine.state import Key, Node, State

IMPACT, DEPENDS_ON = "impact", "depends-on"
COLUMNS = (IMPACT, DEPENDS_ON)

#: the identifier strings of the parts of the definition that are code (``-answer-spec2.md`` §1.2); changing the behaviour behind one means changing the string
COST_ORDER = "possible,distance,length,own-possible,stable-key"
FLOW_FALLBACK = "none"  # SEAM(P1-23): the flag gate and its labelled rows
UNEXPLORED_RULE = "distinct-stored-edges,state-at-bound,increment-1,once"

_OWNER = {"CallSite": "HAS_CALL", "Param": "HAS_PARAM", "Ret": "HAS_RETURN", "Cond": "HAS_COND"}
_OWNER_EDGES = frozenset(_OWNER.values())
_LABELS = ("Symbol", "CallSite", "Param", "Ret", "Cond", "Link", "ContractPoint", "ConfigEntry", "Unit")


class WalkError(RuntimeError):
    """The level guard fired, a request is malformed, or no version maps to the walk definition. Names no value read from the graph."""


# ---------------------------------------------------------------------------------------------
# The row table.
# ---------------------------------------------------------------------------------------------

#: the ``where`` tokens the walk branches on (a table field no code reads is not in the table): ``match-path`` (``DATA``: the state's path must overlap the edge's
#: ``fromPath``), ``arg-path`` (only a state whose path names an argument enters), ``site-match`` (a ``Link`` in ``site`` state lists only the ``CallSite`` whose id is the
#: state's path)
WHERE_TOKENS = ("match-path", "arg-path", "site-match")
#: the to-state tokens: a literal kind, ``path:r`` (the call's result), ``path:to`` (the edge's ``toPath``), ``path:arg`` (the entering path, unchanged),
#: ``path:param`` (the index-matched parameter path, none when the parameter is not the one the path names), ``site`` (the ``siteId`` the step carries)
TO_TOKENS = ("value", "contract", "end", "key", "function", "path:r", "path:to", "path:arg", "path:param", "site")
#: reads an adapter issues beyond its row's own: part of the definition (``-answer-spec2.md`` §1.2)
EXTRA_READS = {"consumes_in": ("on_in", "via_in")}


@dataclass(frozen=True)
class Row:
    """One row of the walk definition: from a node of ``src`` in one of ``kinds`` along ``edge`` in ``dir``.

    ``read`` is the reader name (or ``bound_of_link``), ``adapter`` names the code that maps its columns to steps, ``to`` one of :data:`TO_TOKENS`, ``inc`` the
    distance increment, ``where`` one of :data:`WHERE_TOKENS` or empty.
    """
    id: str
    column: str
    src: str
    kinds: tuple[str, ...]
    read: str
    adapter: str
    edge: str
    dir: str
    to: str
    inc: int
    where: str = ""

    def record(self) -> dict[str, Any]:
        return {"id": self.id, "column": self.column, "src": self.src, "kinds": list(self.kinds), "read": self.read, "adapter": self.adapter, "edge": self.edge,
                "dir": self.dir, "to": self.to, "inc": self.inc, "where": self.where}


def _build_rows() -> tuple[Row, ...]:
    def R(id, column, src, kinds, read, adapter, edge, dir, dst, to, inc, where=""):   # ``dst`` documents the arriving label; the walk never reads it
        return Row(id, column, src, kinds, read, adapter, edge, dir, to, inc, where)

    VP = ("value", "path")
    impact = [
        R("calls_in", IMPACT, "Symbol", ("value",), "calls_in", "calls_in", "CALLS", "<-", "CallSite", "path:r", 1),
        R("name_in", IMPACT, "Symbol", ("value",), "name_in", "name_in", "CALLS_NAME", "<-", "CallSite", "path:r", 1),
        R("to_in:value", IMPACT, "Symbol", ("value",), "to_in", "link_in", "TO", "<-", "Link", "value", 0),
        R("to_in:contract", IMPACT, "Symbol", ("contract",), "to_in", "link_in", "TO", "<-", "Link", "contract", 0),
        R("provides_out", IMPACT, "Symbol", ("key",), "provides_out", "provides_out", "PROVIDES", "->", "ContractPoint", "contract", 0),
        # the arg(i)/Ret rows: a changed argument enters the callee, the index-matched parameter carries it on; the Ret summary is calls_in from a Ret-reached function
        R("arg_calls_out", IMPACT, "CallSite", ("path",), "calls_out", "calls_out", "CALLS", "->", "Symbol", "path:arg", 1, "arg-path"),
        R("arg_name_out", IMPACT, "CallSite", ("path",), "name_out", "name_out", "CALLS_NAME", "->", "Symbol", "path:arg", 1, "arg-path"),
        R("arg_has_param", IMPACT, "Symbol", ("path",), "has_out:Param", "has_out", "HAS_PARAM", "->", "Param", "path:param", 0),
    ]
    for label in ("CallSite", "Param", "Ret", "Cond"):
        impact.append(R(f"owner:{label}", IMPACT, label, VP, f"owner_of:{label}", "owner_of", _OWNER[label], "<-", "Symbol", "value" if label == "Ret" else "end", 0))
    impact.append(R("owner:CallSite:contract", IMPACT, "CallSite", ("contract",), "owner_of:CallSite", "owner_of", "HAS_CALL", "<-", "Symbol", "end", 0))
    impact += [
        R("data_in:Param", IMPACT, "Param", VP, "data_in:Param", "data_in", "DATA", "<-", "*", "path:to", 0, "match-path"),
        R("data_in:CallSite", IMPACT, "CallSite", VP, "data_in:CallSite", "data_in", "DATA", "<-", "*", "path:to", 0, "match-path"),
        R("controlled_in", IMPACT, "Cond", VP, "controlled_in", "controlled_in", "CONTROLLED_BY", "<-", "*", "value", 0),
        R("via_in:value", IMPACT, "Link", ("value",), "via_in", "via_in", "VIA", "<-", "CallSite", "path:r", 1),
        R("via_in:contract", IMPACT, "Link", ("contract",), "via_in", "via_in", "VIA", "<-", "CallSite", "contract", 1),
        R("provides_in", IMPACT, "ContractPoint", ("contract",), "provides_in", "provides_in", "PROVIDES", "<-", "Symbol", "contract", 0),
        R("on_in", IMPACT, "ContractPoint", ("contract",), "on_in", "link_in", "ON", "<-", "Link", "contract", 0),
        R("consumes_in", IMPACT, "ContractPoint", ("contract",), "consumes_in", "consumes_in", "CONSUMES", "<-", "CallSite", "contract", 1),
        # a config entry reaches only the sites it binds (-p1-graph.md G7, §5): the Link is entered in a ``site`` state carrying the bound siteId
        R("bound_in", IMPACT, "ConfigEntry", ("value",), "bound_in", "bound_in", "BOUND_BY", "<-", "Link", "site", 1),
        R("via_in:site", IMPACT, "Link", ("site",), "via_in", "via_in", "VIA", "<-", "CallSite", "path:r", 1, "site-match"),
        R("wiring", IMPACT, "Link", ("value", "contract"), "bound_of_link", "bound_of_link", "BOUND_BY", "->", "ConfigEntry", "end", 1),   # never from a Link a config entry entered: that Link is in ``site`` state
    ]
    depends = [
        R("has_out:CallSite", DEPENDS_ON, "Symbol", ("value",), "has_out:CallSite", "has_out", "HAS_CALL", "->", "CallSite", "value", 0),
        R("has_out:Param", DEPENDS_ON, "Symbol", ("value",), "has_out:Param", "has_out", "HAS_PARAM", "->", "Param", "value", 0),
        R("has_out:Ret", DEPENDS_ON, "Symbol", ("value",), "has_out:Ret", "has_out", "HAS_RETURN", "->", "Ret", "value", 0),
        R("has_out:Cond", DEPENDS_ON, "Symbol", ("value",), "has_out:Cond", "has_out", "HAS_COND", "->", "Cond", "value", 0),
        R("calls_out", DEPENDS_ON, "CallSite", ("value",), "calls_out", "calls_out", "CALLS", "->", "Symbol", "value", 1),
        R("name_out", DEPENDS_ON, "CallSite", ("value",), "name_out", "name_out", "CALLS_NAME", "->", "Symbol", "value", 1),
        R("via_out", DEPENDS_ON, "CallSite", ("value",), "via_out", "via_out", "VIA", "->", "Link", "value", 1),
        R("to_out", DEPENDS_ON, "Link", ("value",), "to_out", "to_out", "TO", "->", "Symbol", "value", 0),
        R("to_unit_out", DEPENDS_ON, "Link", ("value",), "to_unit_out", "to_unit_out", "TO_UNIT", "->", "Unit", "end", 0),
    ]
    for label in ("CallSite", "Ret", "Cond"):
        depends.append(R(f"data_out:{label}", DEPENDS_ON, label, ("value",), f"data_out:{label}", "data_out", "DATA", "->", "*", "value", 0))
    for label in ("CallSite", "Ret", "Cond", "Param"):
        depends.append(R(f"controlled_out:{label}", DEPENDS_ON, label, ("value",), f"controlled_out:{label}", "controlled_out", "CONTROLLED_BY", "->", "Cond", "value", 0))
    return tuple(impact + depends)


#: the shipped row table (``algorithm.version`` 1.1)
ROWS: tuple[Row, ...] = _build_rows()


def definition(table: Sequence[Row] | None = None, *, bound: int | None = None) -> dict[str, Any]:
    """The walk definition record (``-answer-spec2.md`` §1.2): the table plus one identifier per part that is code, and the depth bound (index D3)."""
    rows = ROWS if table is None else tuple(table)
    return {"id": "bfs-v1", "rows": [r.record() for r in rows], "items": {c: sorted([*k, v] for k, v in _ITEM_RULES[c].items()) for c in COLUMNS},
            "extraReads": {k: list(v) for k, v in sorted(EXTRA_READS.items())}, "costOrder": COST_ORDER, "flowFallback": FLOW_FALLBACK, "unexplored": UNEXPLORED_RULE,
            "depthBound": config.DEFAULT_DEPTH_BOUND if bound is None else bound}


def fingerprint(record: Mapping[str, Any]) -> str:
    """``ccg.ids.tuple_id`` over the record's canonical JSON."""
    return _ids.tuple_id(["walkDefinition", json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)])


#: ``recipe.algorithm.version`` -> the pinned fingerprint of the definition that version names. P1 ships one entry; P1-23 re-pins it before P1-39 publishes it
#: (``-answer-spec2.md`` §1.2: a published entry is never edited).
ALGORITHM_VERSIONS: dict[str, str] = {"1.1": "c314fe7c084ba7b951ad6aea505b089d"}


def algorithm_version(record: Mapping[str, Any] | None = None) -> str:
    """The version whose pinned fingerprint is ``record``'s (the shipped definition when omitted); raises :class:`WalkError` when none maps to it."""
    fp = fingerprint(definition() if record is None else record)
    for version in sorted(ALGORITHM_VERSIONS):
        if ALGORITHM_VERSIONS[version] == fp:
            return version
    raise WalkError("no algorithm version maps to this walk definition")


# ---------------------------------------------------------------------------------------------
# Steps: what an adapter makes of the rows of one read.
# ---------------------------------------------------------------------------------------------

@dataclass
class Step:
    src: str                       # the id asked for (column 0 of every expansion row)
    label: str                     # the arriving node's label
    id: str
    status: str = "confirmed"
    reason: str | None = None
    to_path: str | None = None     # DATA: the edge's toPath
    from_path: str | None = None   # DATA: the edge's fromPath
    facts: dict[str, Any] = field(default_factory=dict)
    src_facts: dict[str, Any] = field(default_factory=dict)
    param: tuple[int, str | None] | None = None   # has_out:Param: (index, name)
    site: str | None = None        # bound_in: the siteId the entry binds


def _bad(name: str) -> GraphDataError:
    return GraphDataError(f"query {name!r} returned a malformed row")


def _s(name: str, v: Any, *, none: bool = False) -> Any:
    if v is None and none:
        return None
    if not isinstance(v, str) or not v:
        raise _bad(name)
    return v


def _path(name: str, v: Any) -> str:
    """An access path: a string, empty only as the ``toPath`` of a ``Cond`` use (``-bundle-spec3.md`` §5.1); an absent one reads as empty (a ``DATA`` edge always has both)."""
    if v is None:
        return ""
    if not isinstance(v, str):
        raise _bad(name)
    return v


def _i(name: str, v: Any, *, none: bool = False) -> Any:
    if v is None and none:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise _bad(name)
    return v


def _status(name: str, status: Any, reason: Any) -> tuple[str, str | None]:
    if status not in ("confirmed", "possible"):
        raise _bad(name)
    if reason is not None and not isinstance(reason, str):
        raise _bad(name)
    if status == "possible" and not reason:
        raise _bad(name)  # a possible hop always says why (FR-8)
    return status, reason


def _rows(reader: Reader, name: str, ids: list[str], arity: int) -> list[tuple]:
    out = []
    for row in reader.read(name, {"ids": ids}):
        if len(row) != arity:
            raise _bad(name)
        out.append(row)
    return out


def _facts(**kw: Any) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


def _site_facts(n: str, r: tuple) -> dict[str, Any]:
    return _facts(repo=_s(n, r[2], none=True), unit=_s(n, r[3], none=True), path=_s(n, r[4], none=True), line=_i(n, r[5], none=True), name=_s(n, r[6], none=True))


def _sym_facts(n: str, r: tuple) -> dict[str, Any]:
    """``symbolId, repoId, unitId, qname, path, lineStart`` from index 1."""
    return _facts(repo=_s(n, r[2], none=True), unit=_s(n, r[3], none=True), name=_s(n, r[4], none=True), path=_s(n, r[5], none=True), line=_i(n, r[6], none=True))


def _link_facts(n: str, kind: Any, status: Any, reason: Any, matcher: Any) -> dict[str, Any]:
    st, rs = _status(n, status, reason)
    return _facts(kind=_s(n, kind, none=True), status=st, reason=rs, matcher=_s(n, matcher, none=True))


Adapter = Callable[[Reader, Row, list[str], Mapping[Node, dict]], list[Step]]


def _calls_in(reader, row, ids, facts):
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 9):
        st, rs = _status(n, r[7], r[8])
        out.append(Step(_s(n, r[0]), "CallSite", _s(n, r[1]), st, rs, facts=_site_facts(n, r)))
    return out


def _candidate_reason(n: str, count: Any) -> str:
    """The one place the ``name_in``/``name_out`` ``count(x)`` column is read (``-p1-graph.md`` §5): ``count(x) > 0`` means the site is resolved elsewhere."""
    c = _i(n, count)
    if c < 0:
        raise _bad(n)
    return "resolution-unverified" if c > 0 else "name-match"


def _name_in(reader, row, ids, facts):
    n = row.read
    return [Step(_s(n, r[0]), "CallSite", _s(n, r[1]), "possible", _candidate_reason(n, r[7]), facts=_site_facts(n, r)) for r in _rows(reader, n, ids, 8)]


def _calls_out(reader, row, ids, facts):
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 9):
        st, rs = _status(n, r[7], r[8])
        out.append(Step(_s(n, r[0]), "Symbol", _s(n, r[1]), st, rs, facts=_sym_facts(n, r)))
    return out


def _name_out(reader, row, ids, facts):
    n = row.read
    return [Step(_s(n, r[0]), "Symbol", _s(n, r[1]), "possible", _candidate_reason(n, r[7]), facts=_sym_facts(n, r)) for r in _rows(reader, n, ids, 8)]


def _data(reader, row, ids, facts):
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 8):
        st, rs = _status(n, r[6], r[7])
        label = _s(n, r[1])
        if label not in _LABELS:
            raise _bad(n)
        out.append(Step(_s(n, r[0]), label, _s(n, r[2]), st, rs, to_path=_path(n, r[5]), from_path=_path(n, r[4]), facts=_facts(line=_i(n, r[3], none=True))))
    return out


def _controlled_in(reader, row, ids, facts):
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 4):
        label = _s(n, r[1])
        if label not in _LABELS:
            raise _bad(n)
        out.append(Step(_s(n, r[0]), label, _s(n, r[2]), facts=_facts(line=_i(n, r[3], none=True))))
    return out


def _controlled_out(reader, row, ids, facts):
    n = row.read
    return [Step(_s(n, r[0]), "Cond", _s(n, r[1]), facts=_facts(line=_i(n, r[2], none=True), code=r[3] if isinstance(r[3], str) else None))
            for r in _rows(reader, n, ids, 4)]


def _link_in(reader, row, ids, facts):
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 6):
        lf = _link_facts(n, r[2], r[3], r[4], r[5])
        out.append(Step(_s(n, r[0]), "Link", _s(n, r[1]), lf["status"], lf.get("reason"), facts=lf))
    return out


def _via_in(reader, row, ids, facts):
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 10):
        st, rs = _status(n, r[7], r[8])
        out.append(Step(_s(n, r[0]), "CallSite", _s(n, r[1]), st, rs, facts={**_site_facts(n, r), **_facts(consumerKey=_s(n, r[9], none=True))}))
    return out


def _consumes_in(reader, row, ids, facts):
    """``CONSUMES <-`` from a contract point, minus the consumers some ``Link`` on that point already reaches by ``VIA`` (those arrive through the ``Link``, which
    carries the evidence); what is left is a consumer no provider matched: found, and reported ``unmatched`` (``-p0-graph.md`` §3.2)."""
    n = row.read
    steps = []
    for r in _rows(reader, n, ids, 9):
        st, rs = _status(n, r[7], r[8])
        steps.append(Step(_s(n, r[0]), "CallSite", _s(n, r[1]), st, rs, facts=_site_facts(n, r)))
    if not steps:
        return steps
    links: dict[str, set[str]] = {}
    for r in _rows(reader, "on_in", ids, 6):
        links.setdefault(_s("on_in", r[0]), set()).add(_s("on_in", r[1]))
    lids = sorted({lid for ls in links.values() for lid in ls})
    matched: set[tuple[str, str]] = set()
    if lids:
        by_link = {}
        for r in _rows(reader, "via_in", lids, 10):
            by_link.setdefault(_s("via_in", r[0]), set()).add(_s("via_in", r[1]))
        matched = {(cp, site) for cp, ls in links.items() for lid in ls for site in by_link.get(lid, ())}
    return [s for s in steps if (s.src, s.id) not in matched]


def _provides_in(reader, row, ids, facts):
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 9):
        st, rs = _status(n, r[7], r[8])
        out.append(Step(_s(n, r[0]), "Symbol", _s(n, r[1]), st, rs, facts=_sym_facts(n, r)))
    return out


def _provides_out(reader, row, ids, facts):
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 6):
        st, rs = _status(n, r[4], r[5])
        out.append(Step(_s(n, r[0]), "ContractPoint", _s(n, r[1]), st, rs, facts=_facts(kind=_s(n, r[2], none=True), name=_s(n, r[3], none=True))))
    return out


def _bound_in(reader, row, ids, facts):
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 7):
        lf = _link_facts(n, r[2], r[3], r[4], None)
        out.append(Step(_s(n, r[0]), "Link", _s(n, r[1]), lf["status"], lf.get("reason"), facts=lf, site=_s(n, r[5])))   # a row with no siteId is malformed
    return out


def _owner_of(reader, row, ids, facts):
    """The owner join (``-p1-graph.md`` §5): the enclosing ``Symbol`` of a point; a point carries no ``repo``/``unit``/``path`` of its own, so the row also
    returns those as the point's facts."""
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 9):
        lend, owned = _i(n, r[7], none=True), r[8]
        if owned is not None and not (isinstance(owned, tuple) and all(isinstance(x, int) and not isinstance(x, bool) for x in owned)):
            raise _bad(n)
        sym = _facts(repo=_s(n, r[2], none=True), unit=_s(n, r[3], none=True), name=_s(n, r[4], none=True), path=_s(n, r[5], none=True),
                     line=_i(n, r[6], none=True), lineEnd=lend, ownedRegions=owned)
        out.append(Step(_s(n, r[0]), "Symbol", _s(n, r[1]), facts=sym, src_facts=_facts(repo=sym.get("repo"), unit=sym.get("unit"), path=sym.get("path"))))
    return out


def _has_out(reader, row, ids, facts):
    n = row.read
    label = n.split(":", 1)[1]
    arity = {"CallSite": 4, "Param": 5, "Ret": 3, "Cond": 4}[label]
    out = []
    for r in _rows(reader, n, ids, arity):
        f = _facts(line=_i(n, r[2], none=True))
        param = None
        if label == "CallSite":
            f.update(_facts(name=_s(n, r[3], none=True)))
        elif label == "Cond":
            f.update(_facts(code=r[3] if isinstance(r[3], str) else None))
        elif label == "Param":
            param = (_i(n, r[3]), _s(n, r[4], none=True))
            f.update(_facts(index=param[0], name=param[1]))
        out.append(Step(_s(n, r[0]), label, _s(n, r[1]), facts=f, param=param))
    return out


def _via_out(reader, row, ids, facts):
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 9):
        lf = _link_facts(n, r[2], r[3], r[4], r[5])
        st, rs = _status(n, r[6], r[7])
        out.append(Step(_s(n, r[0]), "Link", _s(n, r[1]), st, rs, facts={**lf, **_facts(consumerKey=_s(n, r[8], none=True))}))
    return out


def _to_out(reader, row, ids, facts):
    """``TO ->``: the hop's status and reason are the ``Link``'s (``-answer-spec2.md`` §3.3), read from the ``Link`` the walk is standing on."""
    n = row.read
    out = []
    for r in _rows(reader, n, ids, 7):
        lf = facts.get(("Link", _s(n, r[0])), {})
        st, rs = _status(n, lf.get("status"), lf.get("reason"))
        out.append(Step(_s(n, r[0]), "Symbol", _s(n, r[1]), st, rs, facts=_sym_facts(n, r)))
    return out


def _to_unit_out(reader, row, ids, facts):
    """``TO_UNIT ->`` reaches the never-built unit: ``possible``/``target-not-mapped`` whatever the link says (``-answer-spec2.md`` §3.5, the registry row)."""
    n = row.read
    return [Step(_s(n, r[0]), "Unit", _s(n, r[1]), "possible", "target-not-mapped",
                 facts=_facts(repo=_s(n, r[2], none=True), unit=_s(n, r[1]), rootPath=r[3] if isinstance(r[3], str) else None,
                              buildStatus=_s(n, r[4], none=True))) for r in _rows(reader, n, ids, 5)]


def _bound_of_link(reader, row, ids, facts):
    """Wiring items (``-answer-spec2.md`` §3.3): each ``configEntry``-origin ``BOUND_BY`` target of the link, once, with the ``Bound`` of the entry's lowest
    ``(siteId, ord)`` edge (the typed read's first-row pick); the hop's status and reason are the ``Link``'s."""
    out = []
    for t in reader.bound_of_link(ids):
        lf = facts.get(("Link", t.link_id), {})
        st, rs = _status("bound_of_link", lf.get("status"), lf.get("reason"))
        bound = _facts(ord=t.ord, port=t.port, portOrigin=t.port_origin, packId=t.pack_id)
        out.append(Step(t.link_id, "ConfigEntry", t.entry_id, st, rs, facts=_facts(repo=t.repo_id, unit=t.unit_id, path=t.path, line=t.line, keyPath=t.key_path,
                                                                                    bound=bound)))
    return out


ADAPTERS: dict[str, Adapter] = {
    "calls_in": _calls_in, "name_in": _name_in, "calls_out": _calls_out, "name_out": _name_out, "data_in": _data, "data_out": _data,
    "controlled_in": _controlled_in, "controlled_out": _controlled_out, "link_in": _link_in, "via_in": _via_in, "consumes_in": _consumes_in,
    "provides_in": _provides_in, "provides_out": _provides_out, "bound_in": _bound_in, "owner_of": _owner_of, "has_out": _has_out, "via_out": _via_out,
    "to_out": _to_out, "to_unit_out": _to_unit_out, "bound_of_link": _bound_of_link,
}


# ---------------------------------------------------------------------------------------------
# The walk.
# ---------------------------------------------------------------------------------------------

Cost = tuple[int, int, int, int]  # (possible hops, distance, length, own-possible hops)


@dataclass(frozen=True)
class Edge:
    type: str
    dir: str  # "<-" against the stored direction, "->" along it
    status: str
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "dir": self.dir, "status": self.status}
        if self.reason:
            out["reason"] = self.reason
        return out

    def sort_key(self) -> tuple:
        return (self.type, self.dir, self.status, self.reason or "")


@dataclass(eq=False)
class Entry:
    """One arrival at a state: the chain's cost, the predecessor's *entry* (captured when the arrival is made, so a chain recomputes to its own cost whatever arrives
    at the predecessor's state later), the edge it arrived by, and the stored edges the bound cut when this arrival was expanded."""
    cost: Cost
    pred: "Entry | None"
    edge: Edge | None
    key: Key
    cuts: frozenset = frozenset()

    @property
    def pred_key(self) -> Key | None:
        return None if self.pred is None else self.pred.key


def _dominates(a: Entry, b: Entry) -> bool:
    """``a`` dominates ``b`` when it has no more possible hops and no greater distance; an equal pair is dominated, so a cycle ends (``-answer-spec2.md`` §4.1)."""
    return a.cost[0] <= b.cost[0] and a.cost[1] <= b.cost[1]


def _chain(entry: Entry) -> list[dict]:
    steps: list[dict] = []
    cur: Entry | None = entry
    while cur is not None:
        steps.append({"node": (cur.key[0], cur.key[1]), "state": (cur.key[2], cur.key[3]), "edge": cur.edge.as_dict() if cur.edge else None})
        cur = cur.pred
    steps.reverse()
    return steps


@dataclass
class Item:
    """One reached thing the column lists: ``symbol`` (impact: a function that holds a reached point; depends-on: a callee or provider), ``config`` (a wiring
    entry) or ``unit`` (a never-built unit). ``chain`` is the best path, anchor first."""
    kind: str
    id: str
    distance: int
    possible_hops: int
    length: int
    chain: list[dict]

    @property
    def status(self) -> str:
        return "possible" if self.possible_hops else "confirmed"

    @property
    def hops(self) -> int:
        return len(self.chain) - 1

    @property
    def via_data(self) -> bool:
        return any(s["edge"] and s["edge"]["type"] == "DATA" for s in self.chain)

    @property
    def reason(self) -> str | None:
        """The reason of the first (nearest the anchor) possible hop; ``None`` for a confirmed item."""
        return next((s["edge"].get("reason") for s in self.chain if s["edge"] and s["edge"]["status"] == "possible"), None)


_ITEM_RULES: dict[str, dict[tuple[str, str, str], str]] = {
    IMPACT: {("Symbol", e, "<-"): "symbol" for e in _OWNER_EDGES} | {("ConfigEntry", "BOUND_BY", "->"): "config"},
    DEPENDS_ON: {("Symbol", "CALLS", "->"): "symbol", ("Symbol", "CALLS_NAME", "->"): "symbol", ("Symbol", "TO", "->"): "symbol", ("Unit", "TO_UNIT", "->"): "unit"},
}


@dataclass
class Walk:
    """The result. ``arrivals`` holds, per state, every arrival no other arrival dominates (``-answer-spec2.md`` §4.1); ``best`` the lexicographic minimum of them,
    for display. ``cut`` is the union of the cut edges of the final kept arrivals: a dominated arrival's cuts are gone with it."""
    anchor: Key
    column: str
    bound: int
    table: tuple[Row, ...]
    levels: int
    frontier_sizes: list[int]
    visited: set[Key]
    arrivals: dict[Key, list[Entry]]
    best: dict[Key, Entry]
    cut: frozenset[tuple[str, str, str]]
    facts: dict[Node, dict[str, Any]]

    @property
    def unexplored(self) -> int:
        """``|E_cut|``: the distinct stored edges the bound cut (``-answer-spec2.md`` §4.2)."""
        return len(self.cut)

    def version(self) -> str:
        """The ``algorithm.version`` of the definition this walk ran, or :class:`WalkError` when no version maps to it."""
        return algorithm_version(definition(self.table, bound=self.bound))

    def chain(self, key: Key) -> list[dict]:
        """The chain of the state's best arrival (its own predecessors, not whatever the predecessors' states hold now)."""
        return _chain(self.best[key])

    def items(self) -> list[Item]:
        """One item per reached thing of the column: listed when some chain within the bound reaches it, on the best such chain; the anchor is never an item, however a
        cycle re-enters it."""
        rules = _ITEM_RULES[self.column]
        picked: dict[tuple[str, str], tuple[tuple, Entry, str]] = {}
        for k in sorted(self.arrivals):
            if k[:2] == self.anchor[:2]:
                continue
            for e in self.arrivals[k]:
                kind = rules.get((k[0], e.edge.type, e.edge.dir)) if e.edge is not None else None
                if kind is None:
                    continue
                ident = (kind, k[1])
                rank = (e.cost, k, _stable(e.pred_key), e.edge.sort_key())
                if ident not in picked or rank < picked[ident][0]:
                    picked[ident] = (rank, e, kind)
        return [Item(kind, e.key[1], e.cost[1], e.cost[0], e.cost[2], _chain(e)) for _, e, kind in (picked[i] for i in sorted(picked))]


def _stable(key: Key | None) -> str:
    return "" if key is None else json.dumps(key)


def _default_state(label: str) -> State:
    return S.CONTRACT if label == "ContractPoint" else S.VALUE


def _check_request(anchor: Any, state: State, column: Any, bound: Any, level_guard: Any) -> None:
    if not (isinstance(anchor, tuple) and len(anchor) == 2 and anchor[0] in _LABELS and isinstance(anchor[1], str) and anchor[1]):
        raise WalkError("the anchor is a (label, id) pair of a known label")
    if anchor[0] == "Link":
        raise WalkError("a Link is never an anchor (no P1 anchor resolves to one); anchor at its provider, its contract point or a config entry")
    if not (isinstance(state, tuple) and len(state) == 2 and state[0] in S.KINDS and isinstance(state[1], str)):
        raise WalkError("the start state is a (kind, path) pair of a known kind")
    if column not in COLUMNS:
        raise WalkError("the column is impact or depends-on")
    for name, v in (("bound", bound), ("level_guard", level_guard)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise WalkError(f"{name} is a positive integer")


def _edge_id(row: Row, src_id: str, to_id: str) -> tuple[str, str, str]:
    """A stored edge by its type and its two endpoint ids, in the stored direction (dependent first): the compound ``CALLS_NAME`` by its ``CallSite`` and
    ``Symbol`` ends (``-answer-spec2.md`` §4.2)."""
    return (row.edge, to_id, src_id) if row.dir == "<-" else (row.edge, src_id, to_id)


def _to_state(row: Row, step: Step, k: Key) -> State | None:
    t = row.to
    if t == "path:to":
        return S.path_state(step.to_path or "")
    if t == "path:r":
        return S.path_state("r")
    if t == "path:arg":
        return S.path_state(k[3])
    if t == "path:param":
        if step.param is None:
            raise WalkError("a param-match row reads a Param")
        p = S.param_path(k[3], step.param[0], step.param[1])
        return None if p is None else S.path_state(p)
    if t == "site":
        return S.site_state(step.site or "")
    if t in ("value", "contract", "end", "key", "function"):
        return (t, "")
    raise WalkError("unknown to-state token")


def _merge(facts: dict[Node, dict[str, Any]], node: Node, new: Mapping[str, Any]) -> None:
    cur = facts.setdefault(node, {})
    for k, v in new.items():
        cur.setdefault(k, v)


def _owner_join(reader: Reader, facts: dict[Node, dict[str, Any]], visited: set[Key]) -> None:
    """A point carries no ``repo``/``unit``/``path`` of its own: those are its owner's, read by the owner join for every reached point that lacks them."""
    for label in ("CallSite", "Param", "Ret", "Cond"):
        need = sorted({k[1] for k in visited if k[0] == label and "unit" not in facts.get((label, k[1]), {})})
        if not need:
            continue
        for r in _owner_of(reader, Row("owner", IMPACT, label, (), f"owner_of:{label}", "owner_of", _OWNER[label], "<-", "Symbol", "end", 0), need, facts):
            _merge(facts, (label, r.src), r.src_facts)


@dataclass
class _Move:
    frm: Entry
    row: Row
    step: Step
    to_state: State
    edge: Edge


def _group_rows(table: Sequence[Row], column: str) -> dict[tuple[str, str], list[Row]]:
    by_group: dict[tuple[str, str], list[Row]] = {}
    for row in table:
        if row.where not in ("", *WHERE_TOKENS):
            raise WalkError("a row names a where-token the walk does not branch on")
        if row.to not in TO_TOKENS:
            raise WalkError("a row names a to-state token the walk does not know")
        if row.column == column:
            for kind in row.kinds:
                by_group.setdefault((row.src, kind), []).append(row)
    return by_group


def _expand(reader: Reader, by_group: Mapping[tuple[str, str], Sequence[Row]], entries: Sequence[Entry], facts: dict[Node, dict[str, Any]]) -> list[_Move]:
    """Every (arrival, row, step) the rows allow from ``entries``, before the bound: the row semantics, shared with the oracle of ``tests/slice``."""
    out: list[_Move] = []
    groups: dict[tuple[str, str], list[Entry]] = {}
    for e in sorted(entries, key=lambda e: (e.key, e.cost, _stable(e.pred_key))):
        groups.setdefault((e.key[0], e.key[2]), []).append(e)
    for (label, kind), group in sorted(groups.items()):
        for row in by_group.get((label, kind), ()):
            use = list(group)
            if row.where == "arg-path":
                use = [e for e in use if S.is_arg_path(e.key[3])]
            if not use:
                continue
            src: dict[str, list[Entry]] = {}
            for e in use:
                src.setdefault(e.key[1], []).append(e)
            for step in ADAPTERS[row.adapter](reader, row, sorted(src), facts):
                if step.src not in src:
                    raise GraphDataError(f"query {row.read!r} returned a row for an id it was not asked")
                _merge(facts, (step.label, step.id), step.facts)
                if step.src_facts:
                    _merge(facts, (label, step.src), step.src_facts)
                edge = Edge(row.edge, row.dir, step.status, step.reason)
                for e in src[step.src]:
                    if row.where == "match-path" and not S.data_edge_matches((e.key[2], e.key[3]), step.from_path or ""):
                        continue
                    if row.where == "site-match" and step.id != e.key[3]:
                        continue
                    to_state = _to_state(row, step, e.key)
                    if to_state is not None:
                        out.append(_Move(e, row, step, to_state, edge))
    return out


def walk(reader: Reader, anchor: Node, state: State | None = None, *, column: str = IMPACT, bound: int | None = None, table: Sequence[Row] | None = None,
         level_guard: int | None = None) -> Walk:
    """Level-synchronous BFS from ``anchor`` in ``state`` over ``table`` (the shipped :data:`ROWS` by default).

    ``bound`` defaults to ``config.DEFAULT_DEPTH_BOUND`` and ``level_guard`` to ``config.LEVEL_GUARD``; a walk that needs more levels raises :class:`WalkError`
    rather than looping or truncating. A variant ``table`` is walked as given; only the shipped one has a version (:meth:`Walk.version`).

    The bounded result is everything reachable within the bound (``-answer-spec2.md`` §4.1): at each state the walk keeps every arrival no other arrival dominates
    on ``(possible hops, distance)`` and expands the state once per kept arrival.
    """
    bound = config.DEFAULT_DEPTH_BOUND if bound is None else bound
    level_guard = config.LEVEL_GUARD if level_guard is None else level_guard
    state = _default_state(anchor[0]) if state is None and isinstance(anchor, tuple) and anchor else state
    _check_request(anchor, state, column, bound, level_guard)
    rows_table = ROWS if table is None else tuple(table)
    by_group = _group_rows(rows_table, column)

    start: Key = (anchor[0], anchor[1], state[0], state[1])
    root = Entry((0, 0, 0, 0), None, None, start)
    kept: dict[Key, list[Entry]] = {start: [root]}
    facts: dict[Node, dict[str, Any]] = {}
    frontier: list[Entry] = [root]
    sizes: list[int] = []
    level = 0
    while frontier:
        level += 1
        if level > level_guard:
            raise WalkError(f"the walk exceeded {level_guard} levels (a cycle the visited set did not stop?)")
        sizes.append(len(frontier))
        cuts: dict[int, set[tuple[str, str, str]]] = {}
        cands: dict[Key, list[tuple]] = {}
        for mv in _expand(reader, by_group, frontier, facts):
            base = mv.frm.cost
            if mv.row.inc and base[1] + mv.row.inc > bound:
                cuts.setdefault(id(mv.frm), set()).add(_edge_id(mv.row, mv.frm.key[1], mv.step.id))
                continue
            # SEAM(P1-23): ``own`` is both the hop's status (cost[0], the possible-hops count) and its own edge's status (cost[3], the own-possible tie-break). P1-23's
            # labelled rows make a hop ``possible`` while its own edge is ``confirmed``: split this variable there, or the tie-break can never decide anything.
            own = 1 if mv.step.status == "possible" else 0
            cost: Cost = (base[0] + own, base[1] + mv.row.inc, level, base[3] + own)
            key: Key = (mv.step.label, mv.step.id, mv.to_state[0], mv.to_state[1])
            cands.setdefault(key, []).append((cost, _stable(mv.frm.key), mv.edge.sort_key(), mv.frm, mv.edge))
        for e in frontier:
            e.cuts = frozenset(cuts.get(id(e), ()))
        nxt: list[Entry] = []
        for key in sorted(cands):
            for cost, _, _, frm, edge in sorted(cands[key], key=lambda c: c[:3]):
                new = Entry(cost, frm, edge, key)
                mine = kept.setdefault(key, [])
                if any(_dominates(a, new) for a in mine):
                    continue
                mine[:] = [a for a in mine if not _dominates(new, a)]
                mine.append(new)
                nxt.append(new)
        frontier = [e for e in nxt if any(x is e for x in kept[e.key])]
    visited = set(kept)
    _owner_join(reader, facts, visited)
    best = {k: min(lst, key=lambda e: (e.cost, _stable(e.pred_key), e.edge.sort_key() if e.edge else ())) for k, lst in kept.items()}
    cut = frozenset().union(*(e.cuts for lst in kept.values() for e in lst))
    return Walk(start, column, bound, rows_table, level, sizes, visited, kept, best, cut, facts)
