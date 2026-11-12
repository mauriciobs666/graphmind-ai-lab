"""The flow fallback of the answer engine's walk (C-1001 P1-23, behaviour (1)): what a unit's ``Unit.flowAvailable`` decides, and the candidate row tables.

Contract: ``docs/plans/connected-code-graph-answer-spec2.md`` §5 (which rows a flag-false unit does not follow, the closed list of labelled rows, the labelling), §1.2
(the walk definition and its version), ``-p1-ml.md`` §3.5 (the four walks P1-39 measures) and §6.6; ``-p1-answers.md`` P1-23 (1).

**What is decided here, and what is not.** The *table* is the walk (``ccg.engine.bfs``): a row carries a ``gate`` (whose unit's flag it needs) and a ``labelled`` bit
(a hop on it is ``possible``); :func:`gate_open` and :func:`labelled_edge` are the two functions that give those fields their meaning, and :class:`Units` reads the
flags and languages the walk gates on, through the one ``units`` read of the reader (``-p1-graph.md`` §5), once per unit per walk. Nothing here knows an edge type.

**Gates** name the nodes whose flags a row looks at: the row's source node (``src``), the node it arrives at (``dst``), or both. ``src:true`` is the precise
``DATA``/``CONTROLLED_BY``/``arg(i)`` rows, followed only in a flag-true unit; ``both:true`` is ``CALLS ->`` into a callee under an argument path (a flag-false callee
has no ``path`` state to enter); ``dst:true``/``dst:false`` split a row that arrives at a function by the flag of the unit the function lives in (``end``/``value`` in a
flag-true unit, ``function`` in a flag-false one); ``src:false`` and ``any:false`` are the twins of a candidate walk (the same row, followed where the shipped one is not,
labelled).

**An absent flag is false** (``-answer-spec2.md`` §5's fail-closed backstop; the assembler always writes it). A node that names a unit with no ``Unit`` row, and a
labelled hop whose unit has no language, raise :class:`ccg.engine.reader.GraphDataError`: a label that cannot say ``<lang>`` is not guessed.

**Candidate walks** (:func:`candidate_table`) are row-table variants chosen by an engine-internal argument of ``bfs.walk(table=...)``, never by a request field; only the
shipped table has an ``algorithm.version``. ``V-flow`` follows the precise rows in a flag-false unit as well, every hop on them labelled; ``V-both`` lists, for a function
the walk reached only at call level, the callees as well as the callers, every added hop labelled; ``V-union`` holds both sets of added rows. P1-39's rule picks the
shipped table (``-p1-ml.md`` §3.5); until it does the shipped one is ``-answer-spec2.md`` §5's.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import NamedTuple

from ccg.engine.reader import GraphDataError, Reader

#: the gate tokens a row may carry (a gate no code reads is not here)
GATES = ("", "src:true", "src:false", "dst:true", "dst:false", "both:true", "any:false")
#: the gates that need the source node's unit / the arriving node's unit
SRC_GATES = frozenset({"src:true", "src:false", "both:true", "any:false"})
DST_GATES = frozenset({"dst:true", "dst:false", "both:true", "any:false"})


class UnitInfo(NamedTuple):
    flag: bool           # ``Unit.flowAvailable``; absent reads false
    lang: str | None     # ``Unit.language``: the ``<lang>`` of ``flow-unavailable:<lang>``


class Units:
    """The flag and language of each unit a walk touched, read once per unit (``units`` read, ids chunked by the reader)."""

    def __init__(self, reader: Reader):
        self._reader = reader
        self.info: dict[str, UnitInfo] = {}

    def ensure(self, ids: Iterable[str | None]) -> None:
        need = sorted({i for i in ids if i is not None and i not in self.info})
        if not need:
            return
        for row in self._reader.read("units", {"ids": need}):
            if len(row) != 10 or not isinstance(row[0], str) or row[0] not in need:
                raise GraphDataError("query 'units' returned a malformed row")
            lang = row[3]
            if lang is not None and not (isinstance(lang, str) and lang):
                raise GraphDataError("query 'units' returned a malformed row")
            self.info[row[0]] = UnitInfo(row[9] is True, lang)
        missing = [i for i in need if i not in self.info]
        if missing:
            raise GraphDataError("a reached node names a unit that has no Unit row")

    def get(self, unit_id: str | None) -> UnitInfo:
        if unit_id is None:
            raise GraphDataError("a reached node carries no unit, so its flow flag cannot be read")
        return self.info[unit_id]


def gate_open(gate: str, src: UnitInfo | None, dst: UnitInfo | None) -> bool:
    """Whether a row with ``gate`` is followed from a source node in ``src``'s unit to a node in ``dst``'s (``None``: that node is not asked for)."""
    if gate == "":
        return True
    if gate == "src:true":
        return src.flag
    if gate == "src:false":
        return not src.flag
    if gate == "dst:true":
        return dst.flag
    if gate == "dst:false":
        return not dst.flag
    if gate == "both:true":
        return src.flag and dst.flag
    if gate == "any:false":
        return not (src.flag and dst.flag)
    raise ValueError(f"unknown gate {gate!r}")


def labelled_lang(src: UnitInfo | None, dst: UnitInfo | None) -> str:
    """The ``<lang>`` of a labelled hop: the language of the flag-false unit it leaves (the source's, else the arriving node's; a labelled row from ``function`` always
    leaves a flag-false source)."""
    for info in (src, dst):
        if info is not None and not info.flag:
            return _lang(info)
    return _lang(src if src is not None else dst)


def _lang(info: UnitInfo | None) -> str:
    if info is None or info.lang is None:
        raise GraphDataError("a labelled hop's unit has no language")
    return info.lang


def labelled_edge(own_status: str, own_reason: str | None, lang: str) -> tuple[str, str | None]:
    """The status and reason of a hop on a labelled row (``-answer-spec2.md`` §5, Labelling): always ``possible``; ``flow-unavailable:<lang>`` when the hop's own edge is
    confirmed, its own reason when it is already ``possible`` (``name-match``, ``resolution-unverified``, a ``possible`` link)."""
    if own_status == "possible":
        return "possible", own_reason
    return "possible", f"flow-unavailable:{lang}"


# ---------------------------------------------------------------------------------------------
# The candidate walks (-p1-ml.md §3.5).
# ---------------------------------------------------------------------------------------------

#: the four walks P1-39 measures; ``as-built`` is the shipped table
CANDIDATES = ("as-built", "V-flow", "V-both", "V-union")


def candidate_table(name: str):
    """The row table of one candidate walk, for ``bfs.walk(table=...)`` (an engine argument, never a request field; only the shipped table has an ``algorithm.version``).

    * ``as-built``: the shipped table (``bfs.ROWS``);
    * ``V-flow``: every precise row a flag-false unit does not follow (``DATA``, ``CONTROLLED_BY``, the ``arg(i)`` rows) gains a twin followed *only* where the shipped one is
      not (``src:false``, ``any:false`` for ``CALLS ->`` into a callee, whose twin is followed when either unit is flag-false), every hop on it labelled;
    * ``V-both``: from a function the walk reached only at call level (``function``, distance ``d``), its callees as well as its callers, to the depth bound (``-p1-ml.md``
      §3.5): ``HAS_CALL ->`` (increment 0) to its call sites (state ``end``), then ``CALLS ->``/``CALLS_NAME ->`` (increment 1) to the callee, an impact item; from a callee so
      reached the same pair again, so a callee at call depth ``k`` has distance ``d + k`` and is listed iff within the bound. A callee is in state kind ``end`` with the path
      ``callee`` (the answer's hop state stays ``end``; ``function`` is not reused), so a callee's own callers are never walked and an ordinary flag-true ``end`` function never
      expands callees. The callee chain is not flag-gated. Every added hop is labelled; the callee arrival's item kind is the row's own (``Row.item``), so the item rules enter a
      fingerprint only with a table that ships them;
    * ``V-union``: both sets of added rows.
    """
    from ccg.engine import bfs   # the table is bfs's record type; bfs imports this module for the gate, so the import waits for the call

    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate walk {name!r}")
    rows = bfs.ROWS
    twin_gate = {"src:true": "src:false", "both:true": "any:false"}
    v_flow = tuple(dataclasses.replace(r, id=f"{r.id}:flow", gate=twin_gate[r.gate], labelled=True) for r in rows if r.gate in twin_gate)
    v_both = (
        bfs.Row("has_out:CallSite:fn:both", bfs.IMPACT, "Symbol", ("function",), "has_out:CallSite", "has_out", "HAS_CALL", "->", "end", 0, labelled=True),
        bfs.Row("has_out:CallSite:callee:both", bfs.IMPACT, "Symbol", ("end",), "has_out:CallSite", "has_out", "HAS_CALL", "->", "end", 0, "callee-state", labelled=True),
        bfs.Row("calls_out:both", bfs.IMPACT, "CallSite", ("end",), "calls_out", "calls_out", "CALLS", "->", "end:callee", 1, labelled=True, item="symbol"),
        bfs.Row("name_out:both", bfs.IMPACT, "CallSite", ("end",), "name_out", "name_out", "CALLS_NAME", "->", "end:callee", 1, labelled=True, item="symbol"),
    )
    return {"as-built": rows, "V-flow": rows + v_flow, "V-both": rows + v_both, "V-union": rows + v_flow + v_both}[name]
