"""S20 (P0-15) -- the 1,000-file synthetic v2 graph, generated deterministically and loaded per D16.

Contract: ``docs/plans/connected-code-graph-p0.md`` §4.5 (S20), ``docs/plans/connected-code-graph-p0-2.md`` S20 row,
graph note ``docs/plans/connected-code-graph-graph.md`` §6.4 (v2 densities: about 455 nodes and 2,400 edges per
file), ``docs/plans/connected-code-graph-p0-graph.md`` §4 (load protocol: batches at most ``BATCH_MAX`` rows,
indexes before the load, constraints after and polled, a zero-retry connection, an explicit query timeout whenever
``TIMEOUT_MAX`` is nonzero).

What is synthetic and what is copied from the slice (``ccg_p0slice``, read once, counts only):

* per-file node and edge densities follow §6.4's v2 estimate, not the slice's 257 nodes / 614 edges per file;
* the ``DATA`` (use label, def label, fromPath, toPath) mix is the slice's top-14 table (:data:`DATA_PATHS`);
* the call graph is a layered DAG with a hub tail (a callee's layer is drawn below the caller's, so in-degree is
  power-law), plus a few percent back-edges so recursion and cycles exist; 8 % of ``CALLS`` are ``possible``;
* planted cycles (:meth:`Synth.planted`): a ``cond -> cond -> cond -> cond`` ``CONTROLLED_BY`` cycle, a two-point
  ``DATA`` cycle and a ``DATA`` self-edge, inside functions that directly call one of the first hub anchors, so
  every walk from those anchors reaches them;
* ``CALLS`` and ``HAS_CALL`` carry a numeric ``w`` (confirmed 1, possible 1000), which ``algo.SSpaths`` needs as its
  ``weightProp``; the production loader (D8) does not store it, so this is a benchmark-only property.

Keys: only ``ccg_p0spike`` and ``ccg_p0spike__*`` are ever touched (:func:`check_spike_key` guards every call).

Usage::

    python synth.py --stats                      # generate and count, no database
    python synth.py --build [--files 1000]       # build ccg_p0spike__build_synth, constraints, RENAME to ccg_p0spike
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
import time
from bisect import bisect_left
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # cpg/
sys.path.insert(0, str(HERE.parent / "vertical-slice"))

import load as L  # noqa: E402  (the loader's helpers: zero_retry, password, batches, statements, poll_constraints)
from load import BATCH_MAX  # noqa: E402

SPIKE_KEY = "ccg_p0spike"
BUILD_KEY = f"{SPIKE_KEY}__build_synth"
_KEY = re.compile(rf"{SPIKE_KEY}(__[A-Za-z0-9_]+)?")
SEED = 20261101


class SynthError(RuntimeError):
    pass


def check_spike_key(key: str) -> str:
    """Only ``ccg_p0spike`` and ``ccg_p0spike__*`` may be touched by S20/S21 (the live rules)."""
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        raise SynthError(f"refusing key {key!r}: only {SPIKE_KEY!r} and {SPIKE_KEY + '__'!r}* may be touched")
    return key


# ---------------------------------------------------------------------------------------------
# The store: the loader's LiveStore (zero-retry, socket timeout 300 s, port 6379 refused) re-keyed to the spike keys.
# ---------------------------------------------------------------------------------------------


class SpikeStore(L.LiveStore):
    """``LiveStore`` with the spike key guard in place of the loader's R-B guard (one connection, retries zero)."""

    def exists(self, key):
        key = check_spike_key(key)
        return bool(self.conn.exists(key))

    def query(self, key, cypher, params=None, timeout=None):
        key = check_spike_key(key)
        graph = self.db.select_graph(key)
        return graph.query(cypher, params, timeout) if timeout else graph.query(cypher, params)

    def ro_query(self, key, cypher, params=None, timeout=None):
        key = check_spike_key(key)
        graph = self.db.select_graph(key)
        return graph.ro_query(cypher, params, timeout) if timeout else graph.ro_query(cypher, params)

    def create_unique_constraint(self, key, label, prop):
        key = check_spike_key(key)
        self.conn.execute_command("GRAPH.CONSTRAINT", "CREATE", key, "UNIQUE", "NODE",
                                  L._ident(label), "PROPERTIES", 1, L._ident(prop))

    def constraints(self, key):
        key = check_spike_key(key)
        res = self.db.select_graph(key).query("CALL db.constraints()")
        return [{"type": r[0], "label": r[1], "properties": r[2], "entity": r[3], "status": r[4]} for r in res.result_set]

    def rename(self, src, dst):
        src, dst = check_spike_key(src), check_spike_key(dst)
        self.conn.rename(src, dst)

    def delete(self, key):
        key = check_spike_key(key)
        self.db.select_graph(key).delete()

    def memory(self, key):
        key = check_spike_key(key)
        raw = self.conn.execute_command("GRAPH.MEMORY", "USAGE", key)
        return {raw[i]: raw[i + 1] for i in range(0, len(raw), 2)}


# ---------------------------------------------------------------------------------------------
# Densities (per file) -- graph note §6.4 v2: ~455 nodes, ~2,400 edges.
# ---------------------------------------------------------------------------------------------

SYMS_PER_FILE = (29, 41)          # -> ~35
CALLS_PER_SYM = (0, 15)           # mean ~7.5
PARAMS_PER_SYM = (0, 4)           # mean ~1.8
RETS_PER_SYM = (0, 1)             # unused: Ret is Bernoulli(RET_P)
CONDS_PER_SYM = (0, 3)            # mean ~1.5
RET_P = 0.6
DATA_PER_POINT = 3.6
CONTROLLED_PER_POINT = 0.45
RESOLVED_P = 0.6                  # a call site with a CALLS edge
POSSIBLE_P = 0.08                 # of CALLS edges
BACK_EDGE_P = 0.04                # CALLS to a layer at or above the caller's
LAYER_SKEW = 2.5                  # callee layer = caller layer * u**LAYER_SKEW: top hub = 2.4 % of CALLS edges (slice: 2.3 %)
NAME_VOCAB_PER_FILE = 14
N_UNITS = 5
LANG = "python"

#: (use label, def label, def-side fromPath, use-side toPath, weight): the slice's top-14 DATA mix (counts of 16,424)
DATA_PATHS = (
    ("CallSite", "CallSite", "r", "a0", 2900), ("CallSite", "CallSite", "r", "recv", 1612),
    ("CallSite", "Param", "p0", "a0", 1229), ("Cond", "CallSite", "r", "", 1130),
    ("Ret", "CallSite", "r", "ret", 1034), ("CallSite", "Param", "p0", "recv", 681),
    ("CallSite", "Param", "p1", "a0", 536), ("CallSite", "CallSite", "r", "a1", 471),
    ("CallSite", "Param", "recv", "recv", 219), ("CallSite", "Param", "p2", "a0", 129),
    ("CallSite", "CallSite", "r", "kw.llm", 127), ("CallSite", "CallSite", "r", "a3", 127),
    ("CallSite", "CallSite", "r", "a2", 125), ("Cond", "Param", "p0", "", 89),
)
_DP_CUM = list(itertools.accumulate(w for *_, w in DATA_PATHS))

NODE_ORDER = ("Repo", "Unit", "File", "Symbol", "Param", "Ret", "Cond", "CallSite", "Name")
EDGE_ORDER = (
    ("Unit", "IN_REPO", "Repo"), ("Unit", "HAS_FILE", "File"), ("File", "DEFINES", "Symbol"),
    ("Symbol", "HAS_CALL", "CallSite"), ("Symbol", "HAS_PARAM", "Param"), ("Symbol", "HAS_RETURN", "Ret"),
    ("Symbol", "HAS_COND", "Cond"), ("CallSite", "CALLS", "Symbol"), ("CallSite", "CALLS_NAME", "Name"),
    ("Name", "RESOLVES_TO", "Symbol"),
    ("CallSite", "DATA", "Param"), ("CallSite", "DATA", "CallSite"), ("Ret", "DATA", "CallSite"),
    ("Cond", "DATA", "Param"), ("Cond", "DATA", "CallSite"),
    ("CallSite", "CONTROLLED_BY", "Cond"), ("Ret", "CONTROLLED_BY", "Cond"), ("Cond", "CONTROLLED_BY", "Cond"),
    ("Param", "CONTROLLED_BY", "Cond"),
)


def _h(*parts) -> str:
    return ":".join(str(p) for p in parts)


class Synth:
    """The whole synthetic graph, lazily: shapes and the call-target table are precomputed (small), rows are streamed."""

    def __init__(self, files: int = 1000, seed: int = SEED):
        if files < 12:
            raise SynthError("at least 12 files (the planted cycles need hub callers)")
        self.files, self.seed = files, seed
        rng = random.Random(f"{seed}:shape")
        self.units = [f"U{i}" for i in range(N_UNITS)]
        self.shapes: list[list[tuple[str, int, int, int, int]]] = []  # per file: (symbolId, nCalls, nParams, nRets, nConds)
        self.sym_file: list[int] = []
        self.sym_ids: list[str] = []
        self.sym_name: list[int] = []
        self.sym_shape: list[tuple[int, int, int, int]] = []
        self.vocab = max(50, NAME_VOCAB_PER_FILE * files)
        for f in range(files):
            row = []
            for s in range(rng.randint(*SYMS_PER_FILE)):
                sid = f"S{f:04d}.{s:02d}"
                shape = (rng.randint(*CALLS_PER_SYM), rng.randint(*PARAMS_PER_SYM),
                         1 if rng.random() < RET_P else 0, rng.randint(*CONDS_PER_SYM))
                row.append((sid, *shape))
                self.sym_file.append(f)
                self.sym_ids.append(sid)
                self.sym_shape.append(shape)
                # symbol names: Zipf-ish over the vocabulary (popular names are defined many times, V13's shape)
                self.sym_name.append(min(self.vocab - 1, int(self.vocab * rng.random() ** 2.2)))
            self.shapes.append(row)
        n = len(self.sym_ids)
        self.n_symbols = n
        order = list(range(n))
        rng.shuffle(order)
        self.layer = [0] * n                     # layer[i] = position of symbol i in a global order; callees go lower
        for pos, i in enumerate(order):
            self.layer[i] = pos
        self.at_layer = order                    # at_layer[pos] = symbol index
        self.sym_index = {s: i for i, s in enumerate(self.sym_ids)}
        # call-target table: callee[(symbol index, j)] = symbol index | None, status possible?
        self.callee: dict[tuple[int, int], tuple[int | None, bool, int]] = {}
        self.indegree = [0] * n
        for i in range(n):
            for j in range(self.sym_shape[i][0]):
                r = random.Random(f"{seed}:call:{i}:{j}")
                tgt, possible = None, False
                if r.random() < RESOLVED_P and n > 1:
                    if r.random() < BACK_EDGE_P:
                        tgt = r.randrange(n)
                    else:
                        lim = self.layer[i]
                        pos = int(lim * r.random() ** LAYER_SKEW)
                        tgt = self.at_layer[pos] if lim > 0 else None
                    possible = r.random() < POSSIBLE_P
                    if tgt is not None:
                        self.indegree[tgt] += 1
                name = self.sym_name[tgt] if tgt is not None else min(self.vocab - 1, int(self.vocab * r.random() ** 2.2))
                self.callee[(i, j)] = (tgt, possible, name)
        self._planted = self._plant()

    # ---- ids --------------------------------------------------------------------------------

    def file_id(self, f: int) -> str:
        return f"F{f:04d}"

    def unit_of(self, f: int) -> str:
        return self.units[f * N_UNITS // self.files]

    @staticmethod
    def call_id(sid: str, j: int) -> str:
        return f"{sid}.c{j}"

    @staticmethod
    def param_id(sid: str, j: int) -> str:
        return f"{sid}.p{j}"

    @staticmethod
    def ret_id(sid: str, j: int) -> str:
        return f"{sid}.r{j}"

    @staticmethod
    def cond_id(sid: str, j: int) -> str:
        return f"{sid}.k{j}"

    @staticmethod
    def name_key(n: int) -> str:
        return f"N{n:06d}"

    # ---- planted cycles ---------------------------------------------------------------------

    def hubs(self) -> list[int]:
        """Symbol indexes by descending ``CALLS`` in-degree (ties by index)."""
        return sorted(range(self.n_symbols), key=lambda i: (-self.indegree[i], i))

    def _plant(self) -> list[dict]:
        """For each of the first 6 hubs: the first function (not yet used) that calls it, has two or more calls and
        three or more conditions gets a cond cycle, a DATA 2-cycle and a self-edge."""
        hubs = self.hubs()[:6]
        want = {h: None for h in hubs}
        used: set[int] = set()
        for i in range(self.n_symbols):
            nc, _np, _nr, nk = self.sym_shape[i]
            if nc < 2 or nk < 3:
                continue
            for j in range(nc):
                t = self.callee[(i, j)][0]
                if t in want and want[t] is None and i not in used:
                    want[t] = (i, j)
                    used.add(i)
                    break
        return [{"hub": self.sym_ids[h], "function": self.sym_ids[want[h][0]], "hubCall": want[h][1]}
                for h in hubs if want[h] is not None]

    def planted(self) -> list[dict]:
        return list(self._planted)

    # ---- anchors ----------------------------------------------------------------------------

    def anchors(self, n: int = 50) -> list[dict]:
        """``n`` seeded anchors: 15 hubs (top in-degree), 20 mid (in-degree 4..20) and 15 leaves (in-degree 1-2)."""
        rng = random.Random(f"{self.seed}:anchors")
        by = self.hubs()
        hubs = by[:15]
        mid = [i for i in by if 4 <= self.indegree[i] <= 20]
        low = [i for i in by if 1 <= self.indegree[i] <= 2]
        k_mid, k_low = round(n * 0.4), n - 15 - round(n * 0.4)
        picked = [(i, "hub") for i in hubs] + [(i, "mid") for i in sorted(rng.sample(mid, k_mid))] + \
                 [(i, "low") for i in sorted(rng.sample(low, k_low))]
        return [{"symbolId": self.sym_ids[i], "class": c, "indegree": self.indegree[i]} for i, c in picked]

    # ---- row streams ------------------------------------------------------------------------

    def _points(self, i: int) -> dict[str, list[str]]:
        sid = self.sym_ids[i]
        nc, np_, nr, nk = self.sym_shape[i]
        return {"CallSite": [self.call_id(sid, j) for j in range(nc)], "Param": [self.param_id(sid, j) for j in range(np_)],
                "Ret": [self.ret_id(sid, j) for j in range(nr)], "Cond": [self.cond_id(sid, j) for j in range(nk)]}

    def nodes(self, label: str) -> Iterator[dict]:
        if label == "Repo":
            yield {"repoId": "R0", "headCommit": "0" * 40}
        elif label == "Unit":
            for u in self.units:
                yield {"unitId": u, "repoId": "R0", "rootPath": f"{u.lower()}/", "language": LANG, "builtCommit": "0" * 40,
                       "builtTree": "1" * 40, "builtAt": "2026-11-01T00:00:00Z", "buildStatus": "fresh"}
        elif label == "File":
            for f in range(self.files):
                yield {"fileId": self.file_id(f), "repoId": "R0", "unitId": self.unit_of(f), "path": f"{self.unit_of(f).lower()}/m{f:04d}.py",
                       "kind": "code", "role": "code", "lang": LANG, "contentHash": f"{f:040x}"}
        elif label == "Symbol":
            for i, sid in enumerate(self.sym_ids):
                f = self.sym_file[i]
                path = f"{self.unit_of(f).lower()}/m{f:04d}.py"
                line = 10 + 17 * int(sid.rsplit(".", 1)[1])
                yield {"symbolId": sid, "repoId": "R0", "unitId": self.unit_of(f), "kind": "function", "name": f"n{self.sym_name[i]}",
                       "nameNorm": f"n{self.sym_name[i]}", "qname": f"{path}::{sid}", "path": path, "lineStart": line,
                       "lineEnd": line + 15, "fingerprint": f"{i:016x}", "lang": LANG, "complexity": 1 + i % 9, "depth": 0}
        elif label == "Param":
            for i, sid in enumerate(self.sym_ids):
                for j in range(self.sym_shape[i][1]):
                    yield {"pointId": self.param_id(sid, j), "index": j, "name": f"p{j}", "line": 10 + j}
        elif label == "Ret":
            for i, sid in enumerate(self.sym_ids):
                for j in range(self.sym_shape[i][2]):
                    yield {"pointId": self.ret_id(sid, j), "line": 20 + j}
        elif label == "Cond":
            for i, sid in enumerate(self.sym_ids):
                for j in range(self.sym_shape[i][3]):
                    yield {"pointId": self.cond_id(sid, j), "line": 12 + j, "code": f"x{j} > {j}"}
        elif label == "CallSite":
            for i, sid in enumerate(self.sym_ids):
                f = self.sym_file[i]
                for j in range(self.sym_shape[i][0]):
                    yield {"callSiteId": self.call_id(sid, j), "repoId": "R0", "unitId": self.unit_of(f),
                           "path": f"{self.unit_of(f).lower()}/m{f:04d}.py", "line": 11 + j, "calleeName": f"n{self.callee[(i, j)][2]}"}
        elif label == "Name":
            used = sorted({v[2] for v in self.callee.values()} | set(self.sym_name))
            for n in used:
                yield {"nameKey": self.name_key(n), "lang": LANG, "name": f"n{n}"}
        else:
            raise SynthError(f"no node stream for {label}")

    def edges(self, group: tuple[str, str, str]) -> Iterator[dict]:
        src, et, dst = group
        if group == ("Unit", "IN_REPO", "Repo"):
            for u in self.units:
                yield {"s": u, "t": "R0"}
        elif group == ("Unit", "HAS_FILE", "File"):
            for f in range(self.files):
                yield {"s": self.unit_of(f), "t": self.file_id(f)}
        elif group == ("File", "DEFINES", "Symbol"):
            for i, sid in enumerate(self.sym_ids):
                yield {"s": self.file_id(self.sym_file[i]), "t": sid}
        elif et in ("HAS_CALL", "HAS_PARAM", "HAS_RETURN", "HAS_COND"):
            label = dst
            for i, sid in enumerate(self.sym_ids):
                for pid in self._points(i)[label]:
                    row = {"s": sid, "t": pid}
                    if et == "HAS_CALL":
                        row["p"] = {"w": 1}
                    yield row
        elif group == ("CallSite", "CALLS", "Symbol"):
            for i, sid in enumerate(self.sym_ids):
                for j in range(self.sym_shape[i][0]):
                    tgt, possible, _ = self.callee[(i, j)]
                    if tgt is not None:
                        p = {"w": 1000, "status": "possible", "reason": "name-matched"} if possible else {"w": 1}
                        yield {"s": self.call_id(sid, j), "t": self.sym_ids[tgt], "p": p}
        elif group == ("CallSite", "CALLS_NAME", "Name"):
            for i, sid in enumerate(self.sym_ids):
                for j in range(self.sym_shape[i][0]):
                    yield {"s": self.call_id(sid, j), "t": self.name_key(self.callee[(i, j)][2])}
        elif group == ("Name", "RESOLVES_TO", "Symbol"):
            called = {v[2] for v in self.callee.values()}
            for i, sid in enumerate(self.sym_ids):
                if self.sym_name[i] in called:
                    yield {"s": self.name_key(self.sym_name[i]), "t": sid}
        elif et == "DATA":
            yield from self._data_edges(src, dst)
        elif et == "CONTROLLED_BY":
            yield from self._controlled_edges(src)
        else:
            raise SynthError(f"no edge stream for {group}")

    def _data_rows(self, i: int) -> list[tuple[str, str, str, str, str, str]]:
        """``(use label, use id, def label, def id, fromPath, toPath)`` of one function, deterministic."""
        pts = self._points(i)
        npts = sum(len(v) for v in pts.values())
        rng = random.Random(f"{self.seed}:data:{i}")
        want = round(DATA_PER_POINT * npts)
        seen: set[tuple] = set()
        out = []
        for _ in range(want * 3):
            if len(out) >= want:
                break
            ul, dl, fp, tp, _w = DATA_PATHS[min(bisect_left(_DP_CUM, rng.random() * _DP_CUM[-1]), len(DATA_PATHS) - 1)]
            if not pts[ul] or not pts[dl]:
                continue
            u, d = rng.choice(pts[ul]), rng.choice(pts[dl])
            if u == d and not (ul == dl == "CallSite"):
                continue
            row = (ul, u, dl, d, fp, tp)
            if (u, d, fp, tp) in seen:
                continue
            seen.add((u, d, fp, tp))
            out.append(row)
        for pl in self._planted:
            if self.sym_index[pl["function"]] != i:
                continue
            sid, j = pl["function"], pl["hubCall"]
            hub_call = self.call_id(sid, j)
            conds = [self.cond_id(sid, k) for k in range(self.sym_shape[i][3])]
            # the hub call's result reaches the first condition (so the cond cycle is reachable), a 2-point DATA cycle
            # between the hub call and a second call site, and a self-edge on the hub call (loop-carried)
            planted = []
            if conds:
                planted.append(("Cond", conds[0], "CallSite", hub_call, "r", ""))
            other = next((self.call_id(sid, k) for k in range(self.sym_shape[i][0]) if k != j), None)
            if other:
                planted.append(("CallSite", other, "CallSite", hub_call, "r", "a0"))
                planted.append(("CallSite", hub_call, "CallSite", other, "r", "a0"))
            planted.append(("CallSite", hub_call, "CallSite", hub_call, "r", "recv"))
            for row in planted:
                if (row[1], row[3], row[4], row[5]) not in seen:  # a planted row may repeat a random one (2 of 2.4 M did)
                    seen.add((row[1], row[3], row[4], row[5]))
                    out.append(row)
        return out

    def _data_edges(self, src: str, dst: str) -> Iterator[dict]:
        for i in range(self.n_symbols):
            for ul, u, dl, d, fp, tp in self._data_rows(i):
                if (ul, dl) == (src, dst):
                    yield {"s": u, "t": d, "p": {"fromPath": fp, "toPath": tp}}

    def _controlled_edges(self, src: str) -> Iterator[dict]:
        for i, sid in enumerate(self.sym_ids):
            pts = self._points(i)
            conds = pts["Cond"]
            if not conds:
                continue
            rng = random.Random(f"{self.seed}:ctl:{i}")
            seen: set[tuple[str, str]] = set()
            npts = sum(len(v) for v in pts.values())
            for _ in range(round(CONTROLLED_PER_POINT * npts)):
                label = rng.choice(["CallSite", "CallSite", "Ret", "Param", "Cond"])
                if not pts[label]:
                    continue
                p, c = rng.choice(pts[label]), rng.choice(conds)
                if p == c or (p, c) in seen:
                    continue
                seen.add((p, c))
                if label == src:
                    yield {"s": p, "t": c}
            for pl in self._planted:
                if pl["function"] == sid and len(conds) >= 1:
                    # cond -> cond -> cond -> cond cycle (a 1-cycle k0 -> k0 is not planted: G-4 drops self-edges)
                    ring = conds[:3] if len(conds) >= 3 else conds
                    edges = [(ring[a], ring[(a + 1) % len(ring)]) for a in range(len(ring))] if len(ring) >= 2 else []
                    edges += [(self.call_id(sid, pl["hubCall"]), ring[0])]
                    for p, c in edges:
                        if (p, c) in seen:
                            continue
                        seen.add((p, c))
                        label = "Cond" if p.split(".")[-1].startswith("k") else "CallSite"
                        if label == src:
                            yield {"s": p, "t": c}

    # ---- facts about the graph, for tests and the report ------------------------------------

    def cond_cycles(self) -> list[list[str]]:
        """The planted ``CONTROLLED_BY`` cond-only cycles (ids, ring order)."""
        out = []
        for pl in self._planted:
            i = self.sym_index[pl["function"]]
            k = self.sym_shape[i][3]
            if k >= 2:
                out.append([self.cond_id(pl["function"], x) for x in range(min(3, k))])
        return out


# ---------------------------------------------------------------------------------------------
# The load (D16): the loader's helpers, the spike key guard.
# ---------------------------------------------------------------------------------------------


def stream_batches(it: Iterator[dict], size: int = BATCH_MAX) -> Iterator[list[dict]]:
    if size < 1 or size > BATCH_MAX:
        raise SynthError(f"batch size must be in 1..BATCH_MAX ({BATCH_MAX}), got {size}")
    while chunk := list(itertools.islice(it, size)):
        yield chunk


def load_synth(store: SpikeStore, synth: Synth, build_key: str = BUILD_KEY, final_key: str = SPIKE_KEY, *,
               drop_stale: bool = False, batch_size: int = BATCH_MAX, log=print) -> dict:
    check_spike_key(build_key)
    check_spike_key(final_key)
    if store.exists(build_key):
        if not drop_stale:
            raise SynthError(f"{build_key} exists; pass drop_stale")
        store.delete(build_key)
    rep: dict = {"batches": 0, "maxBatch": 0, "nodes": {}, "edges": {}, "seconds": {}}
    tmax = store.timeout_max()
    timeout = tmax or None
    rep["timeoutMax"] = tmax
    before = store.used_memory()
    t0 = time.monotonic()
    for stmt in L.index_statements():
        store.query(build_key, stmt)
    rep["seconds"]["indexes"] = time.monotonic() - t0
    t0 = time.monotonic()
    for label in NODE_ORDER:
        n = 0
        for chunk in stream_batches(synth.nodes(label), batch_size):
            store.query(build_key, L.node_statement(label), {"rows": chunk}, timeout)
            rep["batches"] += 1
            rep["maxBatch"] = max(rep["maxBatch"], len(chunk))
            n += len(chunk)
        rep["nodes"][label] = n
    rep["seconds"]["nodes"] = time.monotonic() - t0
    t0 = time.monotonic()
    for group in EDGE_ORDER:
        n = 0
        first = None
        it = synth.edges(group)
        for chunk in stream_batches(it, batch_size):
            with_props = any("p" in r for r in chunk)
            if with_props:
                chunk = [r if "p" in r else {**r, "p": {}} for r in chunk]
            res = store.query(build_key, L.edge_statement(*group, with_props), {"rows": chunk}, timeout)
            if res.relationships_created != len(chunk):
                raise SynthError(f"{group}: {len(chunk)} rows but {res.relationships_created} edges created")
            rep["batches"] += 1
            rep["maxBatch"] = max(rep["maxBatch"], len(chunk))
            n += len(chunk)
        rep["edges"]["-".join(group)] = n
        log(f"  {group}: {n}")
    rep["seconds"]["edges"] = time.monotonic() - t0
    t0 = time.monotonic()
    cons = L.constraint_statements()
    for label, prop in cons:
        store.create_unique_constraint(build_key, label, prop)
    rep["constraints"] = [(c["label"], c["status"]) for c in
                          L.poll_constraints(store, build_key, expected=len(cons))]
    rep["seconds"]["constraints"] = time.monotonic() - t0
    rep["memory"] = store.memory(build_key)
    rep["usedMemoryDeltaBytes"] = store.used_memory() - before
    t0 = time.monotonic()
    store.rename(build_key, final_key)
    rep["seconds"]["rename"] = time.monotonic() - t0
    rep["nodeTotal"] = sum(rep["nodes"].values())
    rep["edgeTotal"] = sum(rep["edges"].values())
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", type=int, default=1000)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--drop-stale", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6390)
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    synth = Synth(args.files)
    print(f"shapes in {time.monotonic() - t0:.1f}s: {synth.n_symbols} symbols; max in-degree {max(synth.indegree)}", file=sys.stderr)
    if args.stats:
        nodes = {l: sum(1 for _ in synth.nodes(l)) for l in NODE_ORDER}
        edges = {"-".join(g): sum(1 for _ in synth.edges(g)) for g in EDGE_ORDER}
        print(json.dumps({"nodes": nodes, "nodeTotal": sum(nodes.values()), "edges": edges, "edgeTotal": sum(edges.values()),
                          "perFile": [sum(nodes.values()) / args.files, sum(edges.values()) / args.files],
                          "planted": synth.planted(), "condCycles": synth.cond_cycles()}, indent=1, sort_keys=True))
    if args.build:
        rep = load_synth(SpikeStore(args.host, args.port), synth, drop_stale=args.drop_stale)
        print(json.dumps(rep, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
