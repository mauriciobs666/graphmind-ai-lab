#!/usr/bin/env python3
"""superset_dev.py -- the measuring tool of pre-registration ``ccg-p1-flow-fallback-v1`` (C-1001 P1-39c).

Spec: line 2 of ``cpg/eval/preregistration.jsonl`` (``tool.contract``, ``tool.refuseList``, ``pins.corpusAndBuilder``, ``statistic``, ``rule``),
read from the **committed blob** ``git show <sha>:cpg/eval/preregistration.jsonl`` (``<sha>`` = ``HEAD`` resolved once, at start; a working-tree edit
of the record moves nothing). Entry points: ``python cpg/eval/superset_dev.py A|B`` and the live test ``cpg/tests/test_engine_superset_dev.py``
(``CCG_SUPERSET_PHASE=A|B``). Offline proof: ``cpg/tests/test_superset_dev_tool.py`` (one planted input per refusal).

Phase A (build): refuse; build every unit of ``cpg/ccg/system.yaml`` at ``EVAL_COMMIT`` on the caller's own throwaway FalkorDB, load it, print the
snapshot commit, unit commits, per-unit build status, graph digest, image ids; **no walk, no result file**; stop. The instance is left up (phase B reuses it).
Phase B (measure): refuse unless the live graph equals the committed ``-snapshot`` line; four candidate tables x four dev anchors through
``ccg.engine.bfs.walk(...).items()`` (never documents), each walk twice, byte-equal else void; write ``cpg/eval/results/`` only.

The ``-tool`` line (appended by the implementer once, before phase A; one line per id, so a post-pin fix needs a new tool id) has this shape::

    {"id": "ccg-p1-flow-fallback-v1-tool",
     "sha256":    {<the 3 TOOL_FILES: this module, the live test, the offline test>: <sha256 hex>},
     "extraPins": {<the 23 EXTRA_PIN_FILES: the build path's imported files the record's builderFiles omits, and the DEFAULT_PACKS YAML>: <sha256 hex>}}

Each map must hold exactly its constant's paths, every file must exist and match; a missing or extra key, or a differing byte, refuses (exit 2).

Decisions the record leaves open (each stated in the output):

* **G1.** Joern units are staged by ``extractors.joern.run.stage_unit`` with ``git ls-tree -r`` + ``git cat-file --batch`` of the commit's tree, never ``git archive``
  and never the work tree. The tool asserts that structurally (:func:`assert_stage_reads_the_commit`) and prints the narrower claim; it does **not** claim a
  ``git archive``. History (``lastChanged*``) is read from the repository's git objects.
* **G2.** The content digest is :func:`digest_of` (v1, below). Phase B **reuses the live instance** phase A left up and re-computes the digest on it; it does not
  rebuild. A rebuild-and-compare would test build determinism, which this record does not claim.
* **G3.** Phase A refuses when a unit hosting a dev anchor or impacted symbol is not ``fresh`` at ``EVAL_COMMIT``; both phases refuse a dev id absent from the graph.
* **G4.** Pins come from the committed blob; append-only means every earlier committed version of the record file is a line prefix of the next.
* "Snapshot commit" (the record's phrase): the ``Snapshot`` node carries no commit, so it is ``EVAL_COMMIT`` (the commit every unit was built at) and the ``snapshotId``.

Custody: reads only the dev file; never opens ``~/.local/share/ccg-eval`` or ``~/.config/ccg``; refuses :6379 and :6390 and any port not published by a container
carrying the ``ccg-falkordb.throwaway=1`` label.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

CPG = Path(__file__).resolve().parent.parent
REPO = CPG.parent

RECORD_ID = "ccg-p1-flow-fallback-v1"
TOOL_ID = f"{RECORD_ID}-tool"
SNAPSHOT_ID = f"{RECORD_ID}-snapshot"
RECORD_PATH = "cpg/eval/preregistration.jsonl"
EVAL_COMMIT_PATH = "cpg/eval/EVAL_COMMIT"
#: the files the ``-tool`` line's ``sha256`` pins: the module, the live test that drives it and the offline refusal proof
TOOL_FILES = ("cpg/eval/superset_dev.py", "cpg/tests/test_engine_superset_dev.py", "cpg/tests/test_superset_dev_tool.py")
#: what the record's ``builderFiles`` does not pin but the build imports or reads (the ``-tool`` line's ``extraPins``): the 14 files an unfiltered ``sys.modules`` sweep
#: of the build path adds (``ccg.refresh``/``build``/``assemble``/``linker``/``packrt``/``store.*``/``engine.*``, ``extractors.joern.run``), and the ``DEFAULT_PACKS`` YAML
EXTRA_PIN_FILES = (
    "cpg/ccg/engine/__init__.py",
    "cpg/extractors/__init__.py",
    "cpg/extractors/common/__init__.py",
    "cpg/extractors/common/errors.py",
    "cpg/extractors/common/writer_rules.py",
    "cpg/extractors/config/__init__.py",
    "cpg/extractors/config/extract.py",
    "cpg/extractors/config/formats.py",
    "cpg/extractors/config/launch.py",
    "cpg/extractors/markdown/__init__.py",
    "cpg/extractors/markdown/extract.py",
    "cpg/extractors/markdown/globs.py",
    "cpg/extractors/shell/__init__.py",
    "cpg/extractors/shell/extract.py",
    "cpg/packs/fastapi/pack.yaml",
    "cpg/packs/fastmcp/pack.yaml",
    "cpg/packs/fetch/pack.yaml",
    "cpg/packs/infra-signatures/negatives.yaml",
    "cpg/packs/infra-signatures/pack.yaml",
    "cpg/packs/kafka/pack.yaml",
    "cpg/packs/severity-rules/pack.yaml",
    "cpg/packs/shell-mcp-call/negatives.yaml",
    "cpg/packs/shell-mcp-call/pack.yaml",
)
RESULT_PATH = "cpg/eval/results/ccg-p1-flow-fallback-v1.json"
SYSTEM = "flowfb"
HOST = "127.0.0.1"
FORBIDDEN_PORTS = (6379, 6390)
DEPTH_BOUND = 2
CANDIDATES = ("as-built", "V-flow", "V-both", "V-union")
#: the record's fixed tie-break order (ascending candidate_table row count: 49, 53, 62, 66)
TIE_ORDER = ("as-built", "V-both", "V-flow", "V-union")
LANGUAGE_OF_EXT = {".py": "python", ".ts": "javascript", ".tsx": "javascript", ".js": "javascript"}
#: the record's as-built prediction, by the impacted file's language (pooled 10/28; python 6/9; TS/JS 4/19)
AS_BUILT_PREDICTION = {"pooled": (10, 28), "python": (6, 9), "javascript": (4, 19)}
DIGEST_ALGORITHM = "ccg-flowfb-graph-digest-v1"
DIGEST_DESCRIPTION = ("sha256 over the header line, then every node line 'N|<label>|<props>' and edge line 'E|<srcLabel>|<type>|<dstLabel>|<srcId>|<dstKey>|<props>', "
                      "sorted bytewise and joined by '\\n'; <props> = properties() as JSON (sort_keys, compact separators, ensure_ascii); ids are each label's id "
                      "property (ccg.store.loader.ID_PROP; SkippedFile: unitId|path); the edge groups are ccg.assemble.rows.EDGE_GROUPS")
PAGE = 4000                                  # rows per page, below the server's RESULTSET_SIZE (10000)
RESULTSET_SIZE = 10000
STAGE_CLAIM = ("Joern units are staged from the commit's tree by git ls-tree -r + git cat-file --batch (extractors.joern.run.stage_unit), not git archive and "
               "not the work tree; history is read from the repository's git objects at the commit")


class Refusal(Exception):
    """The run refuses to start (or to continue): exit 2, before any walk."""


class VoidRun(Refusal):
    """The run is void (a tool defect or a non-deterministic walk): exit 3."""


# --------------------------------------------------------------------------------------------------------------- git and files

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def git(root: Path, *args: str) -> bytes:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, env=env, stdin=subprocess.DEVNULL, timeout=120)
    if done.returncode != 0:
        raise Refusal(f"git {args[0]} failed (exit {done.returncode})")
    return done.stdout


def capture_head(root: Path) -> str:
    """HEAD resolved to a full id, once; every later blob read names this id, never the symbol."""
    return git(root, "rev-parse", "--verify", "HEAD").decode().strip()


def _lines(blob: bytes) -> list[bytes]:
    parts = blob.split(b"\n")
    return parts[:-1] if parts and parts[-1] == b"" else parts


def committed_record_lines(root: Path, head: str) -> list[bytes]:
    return _lines(git(root, "show", f"{head}:{RECORD_PATH}"))


def check_append_only(root: Path, head: str) -> None:
    """Every earlier committed version of the record file is a line prefix of the next (and of the blob at ``head``)."""
    commits = git(root, "log", "--format=%H", "--reverse", head, "--", RECORD_PATH).decode().split()
    if not commits:
        raise Refusal(f"{RECORD_PATH} has no committed version at {head[:12]}")
    prev: list[bytes] | None = None
    for c in commits:
        cur = _lines(git(root, "show", f"{c}:{RECORD_PATH}"))
        if prev is not None and cur[:len(prev)] != prev:
            raise Refusal(f"{RECORD_PATH} is not append-only: a line committed earlier is not byte-identical in {c[:12]}")
        prev = cur


def _line_of(lines: Sequence[bytes], ident: str) -> dict | None:
    found = []
    for raw in lines:
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("id") == ident:
            found.append(d)
    if len(found) > 1:
        raise Refusal(f"the committed record file holds {len(found)} lines with id {ident!r}")
    return found[0] if found else None


# --------------------------------------------------------------------------------------------------------------- the pins

HEX64 = frozenset("0123456789abcdef")


def _is_sha(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and set(v) <= HEX64


def pinned_files(rec: Mapping) -> dict[str, str]:
    """path (repo-relative) -> sha256 for every file the record pins: the dev file, the engine files, the flag registry, the plans, the builder files,
    ``cpg/tests/test_engine_bfs_rows.py``. A group the record does not carry is a refusal: a pin set cannot silently shrink."""
    p = rec["pins"]
    out: dict[str, str] = {}

    def take(group: str, d: Mapping, *, under: str = "") -> None:
        n = 0
        for k, v in d.items():
            if _is_sha(v):
                out[k if "/" in k or not under else f"{under}{k}"] = v
                n += 1
        if n == 0:
            raise Refusal(f"the record pins no file under {group}")

    take("pins.dev", p["dev"])
    take("pins.engine", p["engine"], under="cpg/ccg/engine/")
    take("pins.flagRegistry", p["flagRegistry"])
    take("pins.plans", p["plans"])
    take("pins.corpusAndBuilder.builderFiles", p["corpusAndBuilder"]["builderFiles"])
    take("retLimit.pins", rec["retLimit"]["pins"])
    if "cpg/tests/test_engine_bfs_rows.py" not in out:
        raise Refusal("the record does not pin cpg/tests/test_engine_bfs_rows.py")
    return out


def check_pins(root: Path, rec: Mapping) -> int:
    pins = pinned_files(rec)
    bad = []
    for path, want in sorted(pins.items()):
        f = Path(root) / path
        if not f.is_file() or sha256_file(f) != want:
            bad.append(path)
    if bad:
        raise Refusal(f"{len(bad)} pinned file(s) differ from the record: {', '.join(bad[:5])}")
    return len(pins)


def check_eval_commit(root: Path, rec: Mapping) -> str:
    """``cpg/eval/EVAL_COMMIT`` equals the pin (the pin is the commit id itself, not a sha256), and that commit is an object of the repository."""
    want = rec["pins"]["evalCommit"]
    f = Path(root) / EVAL_COMMIT_PATH
    got = f.read_text().strip() if f.is_file() else None
    if got != want:
        raise Refusal(f"{EVAL_COMMIT_PATH} does not equal the pin")
    git(root, "cat-file", "-e", f"{want}^{{commit}}")
    return want


def check_tool_line(root: Path, lines: Sequence[bytes]) -> dict:
    """The committed ``-tool`` line carries two maps of path -> sha256, each with exactly its constant's keys (:data:`TOOL_FILES` under ``sha256``,
    :data:`EXTRA_PIN_FILES` under ``extraPins``); every file must exist and match."""
    line = _line_of(lines, TOOL_ID)
    if line is None:
        raise Refusal(f"the committed record file has no {TOOL_ID} line (an uncommitted line does not count)")
    for key, files in (("sha256", TOOL_FILES), ("extraPins", EXTRA_PIN_FILES)):
        pinned = line.get(key)
        if not isinstance(pinned, dict) or set(pinned) != set(files):
            raise Refusal(f"the {TOOL_ID} line must pin exactly the {len(files)} files of {key}: {', '.join(files)}")
        if not all(_is_sha(v) for v in pinned.values()):
            raise Refusal(f"the {TOOL_ID} line's {key} holds a value that is not a sha256")
        for path in files:
            f = Path(root) / path
            if not f.is_file() or sha256_file(f) != pinned[path]:
                raise Refusal(f"{path} differs from the {TOOL_ID} line")
    return line


SNAPSHOT_KEYS = ("snapshotCommit", "snapshotId", "unitCommits", "unitStatus", "graphDigest", "digestAlgorithm", "toolchainImageId", "falkordbImageDigest")


def check_snapshot_line(lines: Sequence[bytes]) -> dict:
    line = _line_of(lines, SNAPSHOT_ID)
    if line is None:
        raise Refusal(f"the committed record file has no {SNAPSHOT_ID} line (an uncommitted line does not count)")
    missing = [k for k in SNAPSHOT_KEYS if k not in line]
    if missing:
        raise Refusal(f"the {SNAPSHOT_ID} line lacks {', '.join(missing)}")
    return line


def check_port(env: Mapping[str, str], docker_ps: Callable[[], str]) -> int:
    """A throwaway only: ``CCG_FALKORDB_PORT`` is not 6379/6390 and is published by a container labelled ``ccg-falkordb.throwaway=1``; ``CCG_CONFIG_DIR``
    is a scratch directory, never ``~/.config/ccg``."""
    raw = env.get("CCG_FALKORDB_PORT", "")
    if not raw.isdigit() or int(raw) in FORBIDDEN_PORTS:
        raise Refusal("CCG_FALKORDB_PORT must name a throwaway instance's port (not 6379 or 6390)")
    port = int(raw)
    if f"{HOST}:{port}->" not in docker_ps():
        raise Refusal("the port is not published by a container labelled ccg-falkordb.throwaway=1")
    cfg = env.get("CCG_CONFIG_DIR")
    if not cfg:
        raise Refusal("CCG_CONFIG_DIR must name the throwaway's scratch directory")
    real = (Path.home() / ".config" / "ccg").resolve()
    here = Path(cfg).expanduser().resolve()
    if here == real or real in here.parents:
        raise Refusal("CCG_CONFIG_DIR must not be ~/.config/ccg")
    return port


def assert_stage_reads_the_commit(source: str | None = None) -> str:
    """G1: the Joern stage reads the commit's tree through ``git cat-file --batch`` and never ``git archive`` (an argv element equal to ``archive``)."""
    if source is None:
        from extractors.joern import run as joern_run
        source = inspect.getsource(joern_run.stage_unit)
    fn = ast.parse(textwrap.dedent(source)).body[0]
    strings = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)}      # whole constants: the docstring's prose is none of them
    if "archive" in strings or not {"cat-file", "--batch"} <= strings:
        raise Refusal("stage_unit no longer stages from the commit's tree through git cat-file --batch")
    return STAGE_CLAIM


# --------------------------------------------------------------------------------------------------------------- the dev file

@dataclass(frozen=True)
class DevItem:
    item_id: str
    anchor_id: str
    impacted_id: str
    anchor_repo: str
    anchor_path: str
    impacted_repo: str
    impacted_path: str
    anchor_qname: str
    signature: str
    language: str


def load_dev(root: Path, rec: Mapping) -> list[DevItem]:
    items = []
    for raw in (Path(root) / "cpg/eval/dev/ac5-flow.jsonl").read_text().splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        path = d["impacted"]["path"]
        lang = LANGUAGE_OF_EXT.get(os.path.splitext(path)[1])
        if lang is None:
            raise Refusal("a dev item's impacted file has an extension the record's language rule does not name")
        items.append(DevItem(d["itemId"], d["anchorId"], d["impactedId"], d["anchor"]["repoId"], d["anchor"]["path"], d["impacted"]["repoId"], path,
                             d["anchor"]["qname"], ",".join(h["kind"] for h in d["chain"]), lang))
    dev = rec["pins"]["dev"]
    if len(items) != dev["items"] or len({i.anchor_id for i in items}) != dev["anchorClusters"]:
        raise Refusal("the dev file's item or anchor-cluster count differs from the record")
    return items


# --------------------------------------------------------------------------------------------------------------- the graph digest (G2)

def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def digest_of(nodes: Mapping[str, Sequence[Mapping]], edges: Mapping[tuple[str, str, str], Sequence[tuple]]) -> str:
    """``nodes``: label -> property maps; ``edges``: (src label, type, dst label) -> (srcId, dstKey, props) rows. The algorithm is :data:`DIGEST_DESCRIPTION`."""
    lines = [f"N|{label}|{canon(props)}" for label, rows in nodes.items() for props in rows]
    lines += [f"E|{s}|{t}|{d}|{sid}|{did}|{canon(props)}" for (s, t, d), rows in edges.items() for sid, did, props in rows]
    lines.sort(key=lambda x: x.encode("utf-8"))
    return sha256_bytes((DIGEST_ALGORITHM + "\n" + "\n".join(lines)).encode("utf-8"))


def dump_graph(graph, id_prop: Mapping[str, str], edge_groups: Sequence[tuple[str, str, str]], *, page: int = PAGE) -> tuple[dict, dict]:
    """Read the whole live graph as ``GRAPH.RO_QUERY`` pages below the server's result cap (a page that fills it is split, then refused: never truncated).
    ``graph.ro(cypher, params)`` returns row tuples. No ``UNWIND`` range, no write."""
    nodes: dict[str, list[dict]] = {}
    for label, idp in id_prop.items():
        rows: list[dict] = []
        cursor = ""
        while True:
            got = graph.ro(f"MATCH (n:{label}) WHERE n.{idp} > $c RETURN properties(n) ORDER BY n.{idp} LIMIT {page}", {"c": cursor})
            rows += [r[0] for r in got]
            if len(got) < page:
                break
            cursor = got[-1][0][idp]
        nodes[label] = rows
    skip, got = 0, []
    nodes["SkippedFile"] = []
    while True:
        got = graph.ro(f"MATCH (n:SkippedFile) RETURN properties(n) ORDER BY n.unitId, n.path SKIP {skip} LIMIT {page}", {})
        nodes["SkippedFile"] += [r[0] for r in got]
        if len(got) < page:
            break
        skip += page
    ids_of = {label: [p[id_prop[label]] for p in rows] for label, rows in nodes.items() if label in id_prop}
    edges: dict[tuple[str, str, str], list[tuple]] = {}
    for (s, t, d) in edge_groups:
        sid_p = id_prop[s]
        cols = "b.unitId, b.path" if d == "SkippedFile" else f"b.{id_prop[d]}"
        out: list[tuple] = []
        pending = [ids_of[s][i:i + 500] for i in range(0, len(ids_of[s]), 500)]
        while pending:
            chunk = pending.pop()
            got = graph.ro(f"MATCH (a:{s})-[r:{t}]->(b:{d}) WHERE a.{sid_p} IN $ids RETURN a.{sid_p}, {cols}, properties(r)", {"ids": chunk})
            if len(got) >= RESULTSET_SIZE:
                if len(chunk) == 1:
                    raise Refusal("one source node's edges fill the server's result cap; the digest would be truncated")
                half = len(chunk) // 2
                pending += [chunk[:half], chunk[half:]]
                continue
            for row in got:
                dst = "|".join(row[1:-1])
                out.append((row[0], dst, row[-1]))
        edges[(s, t, d)] = out
    return nodes, edges


def graph_digest(graph) -> tuple[str, dict]:
    from ccg.store.loader import ID_PROP
    from ccg.assemble.rows import EDGE_GROUPS
    nodes, edges = dump_graph(graph, {**ID_PROP, "Snapshot": "snapshotId"}, EDGE_GROUPS)       # Snapshot is not in the loader's ID_PROP but is a node of the graph
    counts = {"nodes": sum(len(v) for v in nodes.values()), "edges": sum(len(v) for v in edges.values())}
    # the dump enumerates labels and edge groups, so a label or group outside the enumeration would be silently absent: cross-check against the graph's own totals
    total_n, total_e = graph.ro("MATCH (n) RETURN count(n)", {})[0][0], graph.ro("MATCH ()-[r]->() RETURN count(r)", {})[0][0]
    if total_n != counts["nodes"]:
        raise Refusal(f"the digest dump holds {counts['nodes']} nodes but the graph holds {total_n}: a label is outside the enumeration")
    if total_e != counts["edges"]:
        raise Refusal(f"the digest dump holds {counts['edges']} edges but the graph holds {total_e}: an edge group is outside the enumeration")
    return digest_of(nodes, edges), counts


# --------------------------------------------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96) -> dict:
    """Wilson score interval, 95%; 28/28 gives a lower bound of 0.879, 4/4 gives 0.510."""
    if n == 0:
        return {"k": k, "n": n, "s": None, "lower": None, "upper": None}
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"k": k, "n": n, "s": p, "lower": (centre - half) / den, "upper": (centre + half) / den}


def listed(walk_items, item: DevItem) -> bool:
    """The record's statistic: some Item has kind == 'symbol' and id == impactedId (a different (kind, node) pair does not count; the anchor is never an item)."""
    return any(i.kind == "symbol" and i.id == item.impacted_id for i in walk_items)


CALLEE_STATE = ("end", "callee")


def callee_hops(chain: Sequence[Mapping]) -> int:
    """Hops that arrive at a callee at call depth: ``CALLS`` / ``CALLS_NAME`` ``->`` into the walk state ``("end", "callee")``. The argument rows (``arg_calls_out``,
    ``arg_name_out``) also walk ``CALLS ->`` but arrive in a ``path`` state: an argument flow, not a callee."""
    return sum(1 for s in chain if s.get("edge") and s["edge"]["type"] in ("CALLS", "CALLS_NAME") and s["edge"]["dir"] == "->"
               and tuple(s["state"]) == CALLEE_STATE)


def second_level_callee_only(walk_items, item: DevItem) -> dict | None:
    """The dev item is listed and its best chain holds >= 2 callee hops (:func:`callee_hops`): listed only through a callee at call depth 2."""
    for i in walk_items:
        if i.kind == "symbol" and i.id == item.impacted_id and callee_hops(i.chain) >= 2:
            return {"itemId": item.item_id, "chain": json.loads(canon(i.chain))}
    return None


def _cell(listed_flags: Sequence[bool]) -> dict:
    return {"listed": sum(listed_flags), "n": len(listed_flags)}


def _walk_json(items) -> str:
    return canon([{"kind": i.kind, "id": i.id, "distance": i.distance, "possibleHops": i.possible_hops, "length": i.length, "chain": i.chain} for i in items])


def measure(reader, dev: Sequence[DevItem], *, walk: Callable, tables: Callable[[str], object], impact) -> dict:
    """Run every candidate over every dev anchor (twice each) and return the candidates block. ``walk`` is ``ccg.engine.bfs.walk``, ``tables`` is
    ``ccg.engine.flow.candidate_table``, ``impact`` the impact column token; the depth bound is the engine's default, never passed (the record varies nothing else)."""
    anchors = sorted({i.anchor_id for i in dev})
    cands: dict[str, dict] = {}
    for name in CANDIDATES:
        listed_ids: dict[str, bool] = {}
        burden: dict[str, int] = {}
        second: list[dict] = []
        for a in anchors:
            runs = []
            for _ in range(2):
                w = walk(reader, ("Symbol", a), column=impact, table=tables(name))
                if w.bound != DEPTH_BOUND:
                    raise VoidRun("a walk ran with a depth bound other than the engine default of 2")
                runs.append((w, w.items()))
            if _walk_json(runs[0][1]) != _walk_json(runs[1][1]):
                raise VoidRun(f"walk {name} from anchor {a[:8]} is not byte-equal across two runs")
            its = runs[0][1]
            burden[a] = len(its)
            for d in (x for x in dev if x.anchor_id == a):
                listed_ids[d.item_id] = listed(its, d)
                hit = second_level_callee_only(its, d)                 # a hit is a listed item by construction
                if hit:
                    second.append(hit)
        cands[name] = {"listedIds": sorted(k for k, v in listed_ids.items() if v), "unlistedIds": sorted(k for k, v in listed_ids.items() if not v),
                       "_flags": listed_ids, "burdenPerAnchor": burden, "burdenSum": sum(burden.values()), "burdenMax": max(burden.values()),
                       "secondLevelCalleeOnly": sorted(second, key=lambda x: x["itemId"])}
    return _report(cands, dev, anchors)


def _report(cands: dict[str, dict], dev: Sequence[DevItem], anchors: Sequence[str]) -> dict:
    by_id = {d.item_id: d for d in dev}
    out: dict[str, dict] = {}
    for name, c in cands.items():
        flags = c.pop("_flags")
        pooled = _cell(list(flags.values()))
        lang, sig = defaultdict(list), defaultdict(list)
        clusters = {}
        for iid, ok in flags.items():
            d = by_id[iid]
            lang[d.language].append(ok)
            sig[d.signature].append(ok)
            clusters.setdefault(d.anchor_id, []).append(ok)
        full = sum(all(v) for v in clusters.values())
        out[name] = {**c, "pooled": {**pooled, **wilson(pooled["listed"], pooled["n"])},
                     "byLanguage": {k: _cell(v) for k, v in sorted(lang.items())}, "bySignature": {k: _cell(v) for k, v in sorted(sig.items())},
                     "byAnchorCluster": {a: {**_cell(v), "allListed": all(v), "qname": next(d.anchor_qname for d in dev if d.anchor_id == a)} for a, v in sorted(clusters.items())},
                     "clusterLevel": wilson(full, len(clusters))}
        if name not in ("V-both", "V-union"):
            out[name].pop("secondLevelCalleeOnly")
    union = sorted(set(out["V-flow"]["listedIds"]) | set(out["V-both"]["listedIds"]))
    if not set(union) <= set(out["V-union"]["listedIds"]):
        raise VoidRun("V-union's listed set is not a superset of the V-flow / V-both union: a tool defect")
    s_asb = out["as-built"]
    pred = {"pooled": AS_BUILT_PREDICTION["pooled"], **{k: AS_BUILT_PREDICTION[k] for k in ("python", "javascript")}}
    measured = {"pooled": (s_asb["pooled"]["listed"], s_asb["pooled"]["n"]),
                **{k: (s_asb["byLanguage"].get(k, {"listed": 0})["listed"], s_asb["byLanguage"].get(k, {"n": 0})["n"]) for k in ("python", "javascript")}}
    return {"candidates": out,
            "setUnionLowerBound": {"listedIds": union, **wilson(len(union), len(dev))},
            "asBuiltComparison": {k: {"predicted": list(pred[k]), "measured": list(measured[k]), "equal": pred[k] == measured[k]} for k in pred},
            "rule": rule_pick(out)}


def rule_pick(cands: Mapping[str, Mapping]) -> dict:
    """The record's rule: the walk with the lowest dev-burden sum among those with the maximum s; tie: lower max per-anchor burden; then TIE_ORDER."""
    top = max(c["pooled"]["listed"] for c in cands.values())
    pool = [n for n in TIE_ORDER if cands[n]["pooled"]["listed"] == top]
    pick = min(pool, key=lambda n: (cands[n]["burdenSum"], cands[n]["burdenMax"], TIE_ORDER.index(n)))
    return {"pick": pick, "maxListed": top, "safeClaim": top == cands[pick]["pooled"]["n"], "among": pool}


# --------------------------------------------------------------------------------------------------------------- run context and dependencies

@dataclass
class Ctx:
    root: Path
    head: str
    phase: str
    rec: dict
    lines: list[bytes]
    env: Mapping[str, str]
    port: int
    eval_commit: str
    dev: list[DevItem]
    tool_sha: dict
    pinned: int
    snapshot: dict | None = None
    stage_claim: str = ""


@dataclass
class Deps:
    """The seams: real by default, replaced by the offline tests."""
    docker_ps: Callable[[], str] = field(default=lambda: _docker("ps", "--filter", "label=ccg-falkordb.throwaway=1", "--format", "{{.Ports}}"))
    build_and_load: Callable[[Ctx], dict] = field(default=lambda ctx: build_and_load(ctx))
    open_graph: Callable[[Ctx], object] = field(default=lambda ctx: LiveGraph(ctx.port))
    images: Callable[[Ctx], dict] = field(default=lambda ctx: image_ids(ctx.port))
    walk: Callable | None = None
    tables: Callable | None = None
    impact: object = None
    loadavg: Callable[[], tuple] = field(default=os.getloadavg)


def _docker(*args: str) -> str:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def preflight(root: Path, head: str, phase: str, env: Mapping[str, str], deps: Deps) -> Ctx:
    """Every refusal of ``tool.refuseList``, in order, before anything is built or walked."""
    if phase not in ("A", "B"):
        raise Refusal("the phase is A or B")
    port = check_port(env, deps.docker_ps)
    lines = committed_record_lines(root, head)
    check_append_only(root, head)
    rec = _line_of(lines, RECORD_ID)
    if rec is None:
        raise Refusal(f"the committed record file has no {RECORD_ID} line")
    pinned = check_pins(root, rec)
    commit = check_eval_commit(root, rec)
    dev = load_dev(root, rec)
    tool = check_tool_line(root, lines)
    snap = check_snapshot_line(lines) if phase == "B" else None
    claim = assert_stage_reads_the_commit()
    return Ctx(Path(root), head, phase, rec, list(lines), env, port, commit, dev, tool, pinned, snap, claim)


def check_dev_ids(graph, ctx: Ctx) -> dict:
    """Both phases (G3): every dev anchor and impacted symbol id is a loaded Symbol of its file, and its unit is ``fresh``, built at EVAL_COMMIT, ``flowAvailable`` false.
    Returns the hosting units' graph rows. A missing id or an unbuilt host is an error, never an unlisted cluster or item."""
    reader = graph.reader
    hosts: dict[str, tuple] = {}
    for it in ctx.dev:
        for sid, repo, path in ((it.anchor_id, it.anchor_repo, it.anchor_path), (it.impacted_id, it.impacted_repo, it.impacted_path)):
            syms = {r[0] for r in reader.read("symbols_of_file", {"repo": repo, "path": path})}
            if sid not in syms:
                raise Refusal(f"dev item {it.item_id[:8]}: a dev id is absent from the loaded graph")
            files = reader.read("file_by_path", {"repo": repo, "path": path})
            if len(files) != 1:
                raise Refusal(f"dev item {it.item_id[:8]}: the file of a dev id is not exactly one File node")
            hosts[files[0][1]] = ()
    rows = {r[0]: r for r in reader.read("units", {"ids": sorted(hosts)})}
    for uid in hosts:
        r = rows.get(uid)
        if r is None or r[4] != "fresh" or r[5] != ctx.eval_commit:
            raise Refusal(f"unit {uid[:8]} hosts a dev anchor or impacted node and did not build fresh at EVAL_COMMIT")
        if r[9] is not False:
            raise Refusal(f"unit {uid[:8]} has flowAvailable {r[9]!r}: the record's premise is that every unit's flowAvailable is false")
        hosts[uid] = r
    return hosts


def unit_rows(graph, unit_ids: Sequence[str]) -> dict[str, dict]:
    rows = graph.reader.read("units", {"ids": sorted(unit_ids)})
    return {r[0]: {"buildStatus": r[4], "builtCommit": r[5], "failReason": r[8]} for r in rows}


def _environment(ctx: Ctx, deps: Deps, images: dict, load_before: tuple) -> dict:
    return {"head": ctx.head, "evalCommit": ctx.eval_commit, "toolSha256": ctx.tool_sha["sha256"], "python": platform.python_version(), "images": images,
            "loadAverageBefore": list(load_before), "loadAverageAfter": list(deps.loadavg()), "pinnedFilesVerified": ctx.pinned, "extraPinsVerified": len(ctx.tool_sha["extraPins"]),
            "digestAlgorithm": DIGEST_ALGORITHM, "digestDescription": DIGEST_DESCRIPTION, "stageClaim": ctx.stage_claim, "system": SYSTEM}


def phase_a(ctx: Ctx, deps: Deps) -> dict:
    load_before = deps.loadavg()
    built = deps.build_and_load(ctx)
    graph = deps.open_graph(ctx)
    check_dev_ids(graph, ctx)
    rows = unit_rows(graph, built["builds"])
    digest, counts = graph_digest(graph)
    images = deps.images(ctx)
    status = {u: {**rows.get(u, {"buildStatus": None, "builtCommit": None, "failReason": None}), "ok": b["ok"], "buildFailReason": b["failReason"]}
              for u, b in sorted(built["builds"].items())}
    return {"phase": "A", "snapshotCommit": ctx.eval_commit, "snapshotId": built["snapshotId"], "unitCommits": {u: v["builtCommit"] for u, v in status.items()},
            "unitStatus": status, "graphDigest": digest, "digestAlgorithm": DIGEST_ALGORITHM, "graphCounts": counts,
            "toolchainImageId": images["toolchainImageId"], "falkordbImageDigest": images["falkordbImageDigest"],
            "environment": _environment(ctx, deps, images, load_before),
            "next": f"append the line {SNAPSHOT_ID} (keys: {', '.join(SNAPSHOT_KEYS)}) from this output to {RECORD_PATH} and commit it; then run phase B on this same instance"}


def check_snapshot_matches(graph, ctx: Ctx, images: dict, digest: str) -> None:
    snap = ctx.snapshot
    assert snap is not None
    if snap["snapshotCommit"] != ctx.eval_commit:
        raise Refusal("the snapshot line's commit is not EVAL_COMMIT")
    if snap["digestAlgorithm"] != DIGEST_ALGORITHM:
        raise Refusal("the snapshot line's digest algorithm is not this tool's")
    if digest != snap["graphDigest"]:
        raise Refusal("the live graph's digest differs from the committed snapshot line")
    rows = unit_rows(graph, snap["unitCommits"])
    if {u: r["builtCommit"] for u, r in rows.items()} != snap["unitCommits"]:
        raise Refusal("the live graph's unit commits differ from the committed snapshot line")
    if {u: r["buildStatus"] for u, r in rows.items()} != {u: s["buildStatus"] for u, s in snap["unitStatus"].items()}:
        raise Refusal("the live graph's unit build status differs from the committed snapshot line")
    snap_rows = graph.reader.read("snapshot", {})
    if [r[0] for r in snap_rows] != [snap["snapshotId"]]:
        raise Refusal("the live graph's Snapshot differs from the committed snapshot line")
    if images["falkordbImageDigest"] != snap["falkordbImageDigest"]:
        raise Refusal("the falkordb image differs from the committed snapshot line")


def phase_b(ctx: Ctx, deps: Deps) -> tuple[dict, Path]:
    load_before = deps.loadavg()
    graph = deps.open_graph(ctx)
    images = deps.images(ctx)
    digest, _ = graph_digest(graph)
    check_snapshot_matches(graph, ctx, images, digest)           # reuse of the live instance, digest re-checked: nothing below runs on a different graph
    check_dev_ids(graph, ctx)
    walk, tables, impact = deps.walk, deps.tables, deps.impact
    if walk is None:
        from ccg.engine import bfs, flow
        walk, tables, impact = bfs.walk, flow.candidate_table, bfs.IMPACT
    res = measure(graph.reader, ctx.dev, walk=walk, tables=tables, impact=impact)
    result = {"recordId": RECORD_ID, "phase": "B", "depthBound": 2, "items": len(ctx.dev), "anchorClusters": len({d.anchor_id for d in ctx.dev}),
              "unlistedPerWalk": {n: c["unlistedIds"] for n, c in res["candidates"].items()},
              "secondLevelCalleeOnlyDefinition": "a dev item listed whose best chain holds at least two callee hops (CALLS or CALLS_NAME, direction ->, arriving in walk state (end, callee); an argument-flow hop arrives in a path state and does not count)",
              "inSample": "V-flow and V-both were designed after the dev hop kinds were known; 28/28 is a screen, validated on held-out only at the AC-5 gate",
              **res, "environment": _environment(ctx, deps, images, load_before)}
    out = ctx.root / RESULT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "xb") as fh:                               # never overwrites a result
            fh.write((json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    except FileExistsError:
        raise Refusal("a result file already exists; the tool never overwrites one") from None
    return result, out


def use_scratch_dirs(env: Mapping[str, str]) -> None:
    """The credentials and audit directories are the throwaway's own, under its ``CCG_CONFIG_DIR``, in both phases (so phase B reads what phase A wrote), and
    ``CCG_FORBID_REAL_CREDENTIALS=1`` makes the custody claim fail-closed (``ccg.atrest.assert_not_real_credentials``). Writes ``os.environ``: a caller restores it."""
    cfg = Path(env["CCG_CONFIG_DIR"])
    os.environ["CCG_CREDENTIALS_DIR"] = str(cfg / "credentials")
    os.environ["CCG_AUDIT_DIR"] = str(cfg / "audit")
    os.environ["CCG_FORBID_REAL_CREDENTIALS"] = "1"             # ccg.atrest's test mode: anything that touches ~/.config/ccg raises instead of reading it


def main(argv: Sequence[str] | None = None, *, deps: Deps | None = None, env: Mapping[str, str] | None = None, root: Path | None = None,
         out=None, err=None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0] not in ("A", "B"):
        print("usage: superset_dev.py A|B", file=err)
        return 2
    root = Path(root) if root is not None else REPO
    deps = deps or Deps()
    env = dict(os.environ) if env is None else env
    try:
        head = capture_head(root)
        ctx = preflight(root, head, argv[0], env, deps)
        use_scratch_dirs(env)
        if argv[0] == "A":
            print(json.dumps(phase_a(ctx, deps), indent=1, sort_keys=True), file=out)
        else:
            result, path = phase_b(ctx, deps)
            print(json.dumps({"result": str(path), "rule": result["rule"]}, sort_keys=True), file=out)
    except VoidRun as exc:
        print(f"VOID: {exc}", file=err)
        return 3
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=err)
        return 2
    return 0


# --------------------------------------------------------------------------------------------------------------- live plumbing (throwaway only)

class LiveGraph:
    """The engine-role connection to the throwaway's live key: ``reader`` (the engine's CypherReader) and ``ro`` (the digest's read-only pages)."""

    def __init__(self, port: int, system: str = SYSTEM):
        from ccg import sysname
        from ccg.engine import config
        from ccg.engine.reader import CypherReader
        from ccg.store import acl, loader as L
        pw = acl.read_credentials(system)["engine"]
        db = L.connect(HOST, port, username=sysname.acl_user("engine", system), password=pw)
        self.reader = CypherReader(db, system)
        self._graph = db.select_graph(sysname.live_key(system))
        self._timeout = config.QUERY_TIMEOUT_MS

    def ro(self, text: str, params: Mapping) -> list[tuple]:
        return [tuple(r) for r in self._graph.ro_query(text, dict(params), self._timeout).result_set]


def image_ids(port: int) -> dict:
    """The toolchain image id (``CCG_TOOLCHAIN_IMAGE``, else ``infra/toolchain/build.sh --print-tag``, as the sandbox runner resolves it) and the falkordb image
    (the throwaway container's image id and its repo digest)."""
    tag = os.environ.get("CCG_TOOLCHAIN_IMAGE") or subprocess.run(
        [str(CPG / "infra/toolchain/build.sh"), "--print-tag"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, stdin=subprocess.DEVNULL, timeout=60).stdout.strip()
    tool_id = _docker("image", "inspect", "--format", "{{.Id}}", tag).strip() if tag else ""
    fid = _docker("inspect", "--format", "{{.Image}}", f"ccg-falkordb-tw-{port}").strip()
    digests = _docker("image", "inspect", "--format", "{{json .RepoDigests}}", fid).strip() if fid else ""
    if not tool_id or not fid:
        raise Refusal("the toolchain image id or the falkordb image digest cannot be read")
    return {"toolchainImageId": tool_id, "falkordbImageDigest": f"{fid} {digests}"}


def unknown_kind_as_failed(real: Callable) -> Callable:
    """``refresh._run``'s ``one``: an ``UnknownUnitKind`` (a caller bug) fails that unit (``extract:failed``) and never aborts the build."""
    from ccg.build import BundleResult, UnknownUnitKind

    def one(unit, repo, commit, sys_name, ctx):
        try:
            return real(unit, repo, commit, sys_name, ctx)
        except UnknownUnitKind:
            return BundleResult(unit, commit, False, fail_reason="extract:failed")
    return one


def check_bind_keys(links) -> None:
    """``refresh._run`` demotes a unit whose bind key mismatches to ``stale-kept``; a snapshot with one is not the all-fresh build the record pins: refuse."""
    if links.bind_key_mismatch:
        raise Refusal(f"the linker reports {len(links.bind_key_mismatch)} bind-key mismatch(es): refresh would have demoted those units")


def build_and_load(ctx: Ctx) -> dict:
    """Phase A's build: every unit of ``cpg/ccg/system.yaml`` at EVAL_COMMIT (``commits[repoId] = EVAL_COMMIT``) with the pinned builder, system ``flowfb``,
    loaded as ``ccg_build_flowfb`` and renamed to the live key as admin. The steps are ``ccg.refresh._run``'s (build, resolve, link, assemble, load) with the
    heads replaced by EVAL_COMMIT and the swap replaced by the admin rename (as ``tests/test_engine_flow_live.py``; ``:Snapshot`` is the loader's last write, so ``load`` returning
    means a complete build; ``ccg.store.swap.swap`` adds a verdict lock, a stamp check and a replay of an empty verdict log, which protect nothing on a one-writer throwaway;
    phase B's digest comparison is the check on the swapped graph). Mirrored from ``_run``: ``UnknownUnitKind`` fails the unit (:func:`unknown_kind_as_failed`).
    The one other divergence: ``bind_key_mismatch`` is a refusal (:func:`check_bind_keys`), not a demotion. Credentials live in the throwaway's
    ``CCG_CONFIG_DIR`` only."""
    from ccg import atrest, refresh, rulepack, sysname
    from ccg.assemble import NotMappedUnit, build_rows
    from ccg.build import BuildContext, build_unit, build_units
    from ccg.linker import Binder, link
    from ccg.manifest import load_manifest
    from ccg.packrt import run_packs
    from ccg.sandbox_audit import default_audit_dir
    from ccg.store import acl, loader as L
    from ccg.units import discover_units
    cfg = Path(ctx.env["CCG_CONFIG_DIR"])
    work = Path(tempfile.mkdtemp(prefix="flowfb-work-", dir=cfg))
    work.chmod(0o700)
    manifest_path = CPG / "ccg" / "system.yaml"
    mf = load_manifest(manifest_path)
    repos = {r.repo_id: ctx.root / r.path for r in mf.repos}
    commits = {rid: ctx.eval_commit for rid in repos}
    units = discover_units(ctx.root, mf, commit=commits)
    keys = Path(os.environ["CCG_CREDENTIALS_DIR"])
    built = build_units(units, repos, commits, SYSTEM, BuildContext(root=work, keys_root=keys, manifest=manifest_path), build=unknown_kind_as_failed(build_unit))
    results = {r.unit.unitId: r for r in built.results}
    audit, now = default_audit_dir(), refresh._rfc3339(None)
    records_by_unit, stale, not_mapped, builds = {}, {}, [], {}
    for u in units:
        status, records, commit, when, reason = refresh._resolve(SYSTEM, u, results.get(u.unitId), work, audit, now)
        res = results.get(u.unitId)
        builds[u.unitId] = {"ok": bool(res and res.ok), "failReason": res.fail_reason if res else None, "status": status, "builtCommit": commit}
        if records is None:
            not_mapped.append(NotMappedUnit(u.unitId, u.repoId, u.rootPath, u.language, reason))
            continue
        records_by_unit[u.unitId] = records
        if status == refresh.STALE_KEPT:
            stale[u.unitId] = (when, reason)
    if not records_by_unit:
        raise Refusal("no unit built: nothing to load")
    bundles = [records_by_unit[u.unitId] for u in units if u.unitId in records_by_unit]
    loaded = [rulepack.load_pack(CPG / "packs" / n / "pack.yaml") for n in refresh.DEFAULT_PACKS]
    facts = run_packs(bundles, loaded)
    binder = Binder(facts, bundles, key=atrest.hmac_key(SYSTEM, keys), repos={r.repo_id: r.path for r in mf.repos}, packs=loaded, units=units)
    links = link(facts, bundles, packs=loaded, bindings=binder)
    check_bind_keys(links)
    rows = build_rows(bundles, links, facts, system=SYSTEM, packs=loaded, audit_dir=audit, stale=stale, not_mapped=not_mapped)
    admin = L.connect(HOST, ctx.port, username=None, password=(cfg / "falkordb-admin.pass").read_text().strip())
    acl.create_users(admin, SYSTEM)
    creds = acl.read_credentials(SYSTEM)
    build_db = L.connect(HOST, ctx.port, username=sysname.acl_user("build", SYSTEM), password=creds["build"])
    report = L.load(L.LiveStore(build_db, SYSTEM), rows, SYSTEM, drop_stale=True)
    admin.connection.rename(report.build_key, sysname.live_key(SYSTEM))
    return {"snapshotId": rows.snapshot["snapshotId"], "builds": builds}


if __name__ == "__main__":
    sys.path.insert(0, str(CPG))
    raise SystemExit(main())
