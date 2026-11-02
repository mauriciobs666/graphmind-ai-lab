"""S20 (P0-15) -- the BFS side of the benchmark: ``bfs-v1`` as implemented in ``impact.py``, instrumented.

Which BFS is benchmarked
------------------------
The walk is **``cpg/spikes/vertical-slice/impact.py`` ``walk()``** (committed ``cb1b2214``) carrying S17-L2's
``function`` state kind (a function entered from a ``Param``/``Cond``/``CallSite`` continues only at function level:
its ``CALLS <-`` / ``TO <-`` hops are ``possible`` with reason ``function-level-continuation``). Nothing of the walk is
re-implemented here; :func:`impact_provenance` records the file's SHA-256 so a result names the exact source.

This module adds only instrumentation and two things ``impact.py`` does not have, both done by wrapping
``impact._expand`` for the duration of one call (never by editing ``impact.py``):

* **a level cap** (``levels``): the wrapper returns no steps once the level counter exceeds the cap, so the walk
  ends by its own termination test; ``levels`` is the depth bound the answer would state;
* **the ``CONTROLLED_BY <-`` row** (graph note §6.2: changed condition -> the points it guards), absent from
  ``impact.py`` (its docstring: "Not followed: ... ``CONTROLLED_BY``") but present in §3.2's "§6.2 as written" and
  the only way a ``cond -> cond`` cycle is ever walked. Used by the ``full`` mode only; the guarded point continues
  in ``value`` state at distance +0, confirmed.

Two modes
---------
``segment``  call-only reverse reach: ``CALLS <-`` and ``HAS_CALL <-`` only (the one segment ``algo.SSpaths`` can
             stand in for: single direction, edge-typed); ``levels = 2 * D`` is ``D`` call hops.
``full``     every row ``impact.py`` follows, plus ``CONTROLLED_BY <-``; ``SSpaths`` cannot express it (several
             directions and states), so it is benchmarked alone, for latency and the repeat-diff.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
VS = HERE.parent / "vertical-slice"
sys.path.insert(0, str(VS))

import impact as I  # noqa: E402

IMPACT_FILE = Path(I.__file__).resolve()
ALGORITHM = "bfs-v1"


def impact_provenance() -> dict:
    """Which ``impact.py`` was benchmarked: path, SHA-256, git HEAD and whether the working tree differs from it."""
    root = IMPACT_FILE.parents[3]
    rel = str(IMPACT_FILE.relative_to(root))

    def git(*a):
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()

    return {"file": rel, "sha256": hashlib.sha256(IMPACT_FILE.read_bytes()).hexdigest(),
            "lastCommit": git("log", "-1", "--format=%h", "--", rel),
            "workingTreeModified": bool(git("status", "--porcelain", "--", rel)),
            "hasFunctionState": hasattr(I, "FUNCTION"), "algorithm": ALGORITHM}


# ---------------------------------------------------------------------------------------------
# Instrumentation.
# ---------------------------------------------------------------------------------------------


@dataclass
class Tracker:
    """Per-answer measurements: every read's latency (tagged by edge type), every level's duration."""
    cap: int | None = None
    level: int = 0
    reads: list[tuple[int, str, float, int, int]] = field(default_factory=list)  # (level, tag, seconds, ids, rows)
    levels: list[tuple[int, int, int, float]] = field(default_factory=list)  # (level, frontier, steps, seconds)


def _tag(cypher: str) -> str:
    for rel in ("CONTROLLED_BY", "HAS_CALL", "HAS_PARAM", "HAS_RETURN", "HAS_COND", "CALLS", "DATA", "TO", "VIA", "PROVIDES"):
        if f"[:{rel}]" in cypher or f"[d:{rel}]" in cypher or f"[v:{rel}]" in cypher:
            return rel
    return "other"


class TimedRun:
    """A ``run(cypher, params) -> rows`` for ``CypherReader`` that times each read and logs it to a tracker."""

    def __init__(self, graph, tracker: Tracker, timeout: int | None = None):
        self.graph, self.tracker, self.timeout = graph, tracker, timeout

    def __call__(self, cypher: str, params: dict) -> list[list]:
        t0 = time.perf_counter()
        rows = self.graph.ro_query(cypher, params, self.timeout).result_set if self.timeout else self.graph.ro_query(cypher, params).result_set
        self.tracker.reads.append((self.tracker.level, _tag(cypher), time.perf_counter() - t0, len(params.get("ids", ())), len(rows)))
        return rows


class SplitRows:
    """Graph note Q7: "a reply of exactly 10,000 rows is treated as truncated and re-split". ``impact.py`` raises at
    ``RESULT_SET_LIMIT`` instead (its 1,000-id chunk is unsafe on a hub-rich frontier, measured here); this mixin halves
    the id chunk and retries, and raises only when a single id alone fills the reply. ``splits`` counts the halvings."""

    splits = 0

    def _rows(self, cypher: str, ids: list[str]) -> list[list]:
        out: list[list] = []
        stack = [ids[i:i + I.CHUNK] for i in range(0, len(ids), I.CHUNK)][::-1]
        while stack:
            part = stack.pop()
            rows = self.run(cypher, {"ids": part})
            if len(rows) >= I.RESULT_SET_LIMIT:
                if len(part) == 1:
                    raise I.ImpactError(f"one id fills the reply ({I.RESULT_SET_LIMIT} rows): {part[0]}")
                self.splits += 1
                mid = len(part) // 2
                stack.extend([part[mid:], part[:mid]])
                continue
            out.extend(rows)
        return sorted(out)


class SegmentReader(SplitRows, I.CypherReader):
    """Call-only reverse reach: ``CALLS <-`` and ``HAS_CALL <-``; every other row of ``bfs-v1`` is switched off."""

    def owner_of(self, label, ids):
        return super().owner_of(label, ids) if label == "CallSite" else []

    def data_in(self, label, ids):
        return []

    def to_in(self, symbol_ids):
        return []

    def via_in(self, link_ids):
        return []

    def provides_in(self, cp_ids):
        return []

    def provides_out(self, symbol_ids):
        return []


class FullReader(SplitRows, I.CypherReader):
    """Every row ``impact.py`` follows, plus ``CONTROLLED_BY <-``."""

    def controlled_in(self, cond_ids):
        return [tuple(r) for r in self._rows(
            "UNWIND $ids AS i MATCH (p)-[:CONTROLLED_BY]->(c:Cond {pointId: i}) "
            "RETURN i, labels(p)[0], coalesce(p.callSiteId, p.pointId)", cond_ids)]


@contextmanager
def instrumented(tracker: Tracker, *, controlled_by: bool, chunk: int | None = None):
    """Wrap ``impact._expand``: count levels, time them, cap them, optionally add the ``CONTROLLED_BY <-`` row."""
    orig, orig_chunk = I._expand, I.CHUNK

    def expand(reader, frontier):
        tracker.level += 1
        if tracker.cap is not None and tracker.level > tracker.cap:
            return []
        t0 = time.perf_counter()
        out = orig(reader, frontier)
        if controlled_by:
            ids = sorted({k[1] for k in frontier if k[0] == "Cond" and k[2] in ("value", "path")})
            if ids:
                src: dict[str, list] = {}
                for k in frontier:
                    if k[0] == "Cond" and k[2] in ("value", "path"):
                        src.setdefault(k[1], []).append(k)
                for cid, label, pid in reader.controlled_in(ids):
                    for k in src[cid]:
                        out.append((k, (label, pid), I.VALUE, I.Edge("CONTROLLED_BY", "<-", "confirmed"), 0, 0))
        tracker.levels.append((tracker.level, len(frontier), len(out), time.perf_counter() - t0))
        return out

    I._expand = expand
    if chunk:
        I.CHUNK = chunk
    try:
        yield
    finally:
        I._expand, I.CHUNK = orig, orig_chunk


# ---------------------------------------------------------------------------------------------
# The answer.
# ---------------------------------------------------------------------------------------------


def canonical_items(result: I.Result) -> list[list]:
    """The ``(item, status, distance, chain)`` set the FR-19 repeat-diff compares, in a stable order."""
    out = []
    for it in result.items():
        chain = [[list(s["node"]), list(s["state"]), s["edge"]] for s in it.chain]
        out.append([it.symbol_id, it.status, it.distance, it.possible_hops, it.length, chain])
    return sorted(out, key=lambda r: r[0])


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def answer(graph, anchor_symbol: str, *, mode: str, levels: int, timeout: int | None = None, chunk: int | None = None) -> dict:
    """One full ``bfs-v1`` answer from a ``Symbol`` anchor in ``value`` state; returns items, digest and timings."""
    if mode not in ("segment", "full"):
        raise ValueError(mode)
    tracker = Tracker(cap=levels)
    run = TimedRun(graph, tracker, timeout)
    reader = SegmentReader(run) if mode == "segment" else FullReader(run)
    t0 = time.perf_counter()
    with instrumented(tracker, controlled_by=(mode == "full"), chunk=chunk):
        result = I.walk(reader, ("Symbol", anchor_symbol), I.VALUE, max_levels=levels + 2)
    items = canonical_items(result)
    seconds = time.perf_counter() - t0
    return {"items": items, "digest": digest(items), "seconds": seconds, "nItems": len(items), "tracker": tracker,
            "splits": reader.splits,
            "states": result.state_count, "levels": result.levels, "frontierSizes": result.frontier_sizes}
