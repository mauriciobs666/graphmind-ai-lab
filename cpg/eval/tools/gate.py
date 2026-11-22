#!/usr/bin/env python3
"""gate.py - the AC-5 gate harness (P1-32b; ``docs/plans/connected-code-graph-p1-gates.md`` P1-32, 32b bullet).

Spec: eval protocol ``docs/plans/connected-code-graph-eval-protocol-ml.md`` §5.1-§5.4 (hit rule H1-H4, HA-1..HA-5), §12.1 (fail
procedure, floor E12), §12.3 (pre-registration); ``docs/plans/connected-code-graph-p1-ml.md`` §5, §6.1-§6.5.
Imported, never reimplemented: ``match_keys`` (``E``, ``h1_symbol_credit``, ``source_check_needed``, ``renormalize_check``,
``contract_id``), ``split_manifest`` (``verify_manifest``, ``FRESH``/exposure shapes), ``noise.path_excluded``, ``oracle.wilson``,
``ac24_smoke`` (tree hashes, custodian root, the committed-copy read).

Layers (each pure; every I/O seam is injected so the offline tier needs no store, no model and no held-out data):

* verification: :func:`verify_heldout`, :func:`observe_tree`, :func:`prereg_comparison`, :func:`first_contact`,
  :func:`renormalize_stored_gold`, :func:`pin_report` (R-6), :func:`harness_pin` (self-pin), :func:`scope_report`;
* the gate snapshot: :class:`StoreSnapshot` over a read-only runner (``GRAPH.RO_QUERY``), or any object with the same methods;
* C5 / hit rule: :func:`check_stamp`, :func:`check_request_echo`, :func:`replay_chain`, :func:`score_answer`, :func:`score_gate`;
* fail procedure and report: :func:`decide_gate`, :func:`build_report`.

Every output that is derived from held-out items is written under the custodian root (``-p1-ml.md`` §6.4) by :func:`write_custodian`;
stdout and committed reports carry counts, hashes and classes only. Exit: 0 ok, 1 gate result not a pass / findings, 2 refused, 4 bad input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_CPG = _HERE.parents[1]
for _p in (str(_HERE), str(_CPG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ac24_smoke as _ac                    # noqa: E402  (tree hashes, custodian root, committed copy)
import match_keys as _mk                    # noqa: E402  (E, h1_symbol_credit, renormalize_check, contract_id)
import noise as _noise                      # noqa: E402  (path_excluded, p90, median)
import split_manifest as _sm                # noqa: E402  (verify_manifest)
from oracle import wilson as _wilson        # noqa: E402  (the one Wilson implementation)

PREREG_SCHEMA = "ccg-prereg-v1"
PREREG_FIELDS = ("packSetHash", "coreTreeHash", "algorithm", "defaultDepthBound")
FLOOR = 35                                   # EP §12.1 step 5 (E12)
MIN_COUNTING = 75                            # AC-5 pooled
MIN_STRATUM = 8                              # AC-5 per stratum
EMPTY_VERDICT_LOG_HASH = "c482b1cf54631b37eb4572dc86a68fb1"     # HA-4: the literal, never ``answer.EMPTY_VERDICT_LOG_HASH``
STRATA = _mk.STRATA
HARNESS_FILES = tuple(f"cpg/eval/tools/{n}" for n in (
    "gate.py", "match_keys.py", "split_manifest.py", "noise.py", "oracle.py", "ac24_smoke.py", "export_kit.py"))
STATUSES = ("confirmed", "possible")        # H2
TRIAGE = ("key-error", "extraction", "recognizer", "linker-engine", "depth-bound", "dynamic")    # EP §12.1 step 2


class Refusal(RuntimeError):
    """The gate declines to run or to label a run (counts only in the message)."""


class HarnessError(RuntimeError):
    """A C5 harness error: the gate verdict is blocked."""


# =============================================================================================
# verification layer
# =============================================================================================

def verify_heldout(manifest_path: Path, heldout_dir: Path, eval_commit_file: Path | None = None) -> dict:
    """``split_manifest.verify_manifest`` plus the manifest's ``evalCommit`` against ``EVAL_COMMIT``. -> ``{"problems": [...]}`` (names only)."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    problems = list(_sm.verify_manifest(manifest, Path(heldout_dir)))
    ecf = eval_commit_file or _HERE.parent / "EVAL_COMMIT"
    if manifest.get("evalCommit") != Path(ecf).read_text().strip():
        problems.append("evalCommit: differs from EVAL_COMMIT")
    return {"ok": not problems, "problems": problems}


def _porcelain(repo: Path, *paths: str) -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=repo, capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def observe_tree(repo: Path | None = None) -> dict:
    """What the harness recomputes from the working tree: both hashes and the dirty paths under ALL of ``cpg/`` (EP §12.3: no uncommitted
    change under ``cpg/``), plus ``ac24_smoke``'s ignored-importable-file scan. Read-only git."""
    repo = Path(repo) if repo else _CPG.parent
    base = _ac.observe_tree(repo=repo)
    dirty = sorted(set(base["dirty"]) | set(_porcelain(repo, "cpg")))
    return {"coreTreeHash": base["coreTreeHash"], "packSetHash": base["packSetHash"], "head": base["head"],
            "dirty": dirty, "clean": not dirty}


def engine_observed() -> dict:
    """``algorithm`` and ``defaultDepthBound`` as the engine at this tree emits them (the build's own values)."""
    from ccg.engine import bfs, config                      # lazy: the offline tier injects ``observed``
    return {"algorithm": {"id": "bfs-v1", "version": bfs.algorithm_version()}, "defaultDepthBound": config.DEFAULT_DEPTH_BOUND}


def read_prereg(path: Path, gate_id: str, committed_text: Callable[[Path], str] | None = None) -> dict:
    return find_prereg_line(path, gate_id, committed_text)[1]


def find_prereg_line(path: Path, gate_id: str, committed_text: Callable[[Path], str] | None = None) -> tuple[str, dict]:
    """``(raw line, record)``: the ONE ``ccg-prereg-v1`` record for ``gate_id`` (its ``stratum``), present in the file AND in HEAD's committed copy; else :class:`Refusal`."""
    path = Path(path)
    if not path.exists():
        raise Refusal("no preregistration file")
    committed = (committed_text or _ac._git_head_text)(path).splitlines()
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("schema") == PREREG_SCHEMA and d.get("stratum") == gate_id:
            found.append((line, d))
    if len(found) != 1:
        raise Refusal(f"the preregistration file holds {len(found)} {PREREG_SCHEMA} records for {gate_id} (exactly one is required)")
    if found[0][0].strip() not in committed:
        raise Refusal("the record is present but not committed (HEAD's copy does not hold it)")
    for k in PREREG_FIELDS:
        if k not in found[0][1]:
            raise Refusal(f"the record has no {k}")
    return found[0][0], found[0][1]


def prereg_comparison(record: Mapping, observed: Mapping) -> dict:
    """EP §12.3: the four fields, each by equality (``algorithm`` as ``{id, version}``, strings compared exactly)."""
    fields = {}
    for k in PREREG_FIELDS:
        a, b = record.get(k), observed.get(k)
        fields[k] = bool(k in record and k in observed and type(a) is type(b) and a == b)
    return {"fields": fields, "allEqual": all(fields.values()), "differs": [k for k, v in fields.items() if not v]}


def first_contact(record: Mapping, observed: Mapping) -> dict:
    """The run is "first contact" only if all four fields equal the committed record AND ``cpg/`` has no uncommitted change;
    otherwise it may run only as "after adaptation" (ML §4.5)."""
    cmp = prereg_comparison(record, observed)
    dirty = list(observed.get("dirty") or ())
    ok = cmp["allEqual"] and not dirty and observed.get("clean", not dirty) is not False
    return {"firstContact": bool(ok), "label": "first-contact" if ok else "after-adaptation", "fieldsDiffering": cmp["differs"],
            "dirtyPaths": len(dirty), "fields": cmp["fields"]}


def renormalize_stored_gold(records: Iterable[Mapping]) -> dict:
    """``match_keys.renormalize_check`` over the STORED contract locators and ids of the key items (anchor, and impacted when a contract),
    before any scoring. Any difference stops the gate (:func:`require_clean_renormalize`). -> counts and differing stored ids."""
    entries = []
    for r in records:
        for loc_key, id_key in (("anchor", "anchorId"), ("impacted", "impactedId")):
            loc = r.get(loc_key)
            if isinstance(loc, dict) and loc.get("type") == "contract":
                entries.append({"id": r.get(id_key), "locator": loc, "form": "provider"})
    return _mk.renormalize_check(entries)


def require_clean_renormalize(rep: Mapping) -> None:
    if rep.get("differences"):
        raise Refusal(f"stored gold renormalize-check: {rep['differences']} of {rep['checked']} differ; the gate stops before scoring")


_VERSION = re.compile(r"\*\*Version:\*\*\s*([0-9][0-9A-Za-z.]*)")


def pin_report(eval_manifest: Mapping, repo: Path, heldout_manifest: Mapping | None = None) -> dict:
    """R-6: the protocol pin strings the manifest records (never edited) against the committed documents' ``Version:`` header."""
    docs = (eval_manifest.get("protocol") or {}).get("documents") or []
    recorded = {d["path"]: str(d["version"]) for d in docs}
    committed, diffs = {}, []
    for p, v in recorded.items():
        f = Path(repo) / p
        m = _VERSION.search(f.read_text(encoding="utf-8")[:4000]) if f.is_file() else None
        committed[p] = m.group(1) if m else None
        if committed[p] != v:
            diffs.append({"path": p, "recorded": v, "committed": committed[p]})
    string = (eval_manifest.get("protocol") or {}).get("protocolString")
    rebuilt = "+".join(f"{p}@{v}" for p, v in recorded.items())
    return {"recorded": recorded, "committed": committed, "differences": diffs, "protocolStringConsistent": string == rebuilt,
            "heldoutProtocolEqual": None if heldout_manifest is None else heldout_manifest.get("protocol") == string}


def _git_show(repo: Path, rev: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=repo, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def record_commit(repo: Path, prereg_rel: str, record_line: str) -> str | None:
    """The commit that first added the record's line to ``preregistration.jsonl`` (``git log -S``, oldest)."""
    out = subprocess.run(["git", "log", "--reverse", "--format=%H", "-S", record_line.strip(), "--", prereg_rel], cwd=repo,
                         capture_output=True, text=True, check=True).stdout.split()
    return out[0] if out else None


def harness_pin(repo: Path, record_commit_sha: str, files: Sequence[str] = HARNESS_FILES, rescored_unchanged: bool | None = None) -> dict:
    """Self-pin (``-p1-ml.md`` §6.1): sha256 of each harness file now vs ``git show <recordCommit>:<path>``. An edit after the record (or a file the
    record lacks) makes the run "after adaptation", unless the caller re-scored with both versions and no item's score changed
    (``rescored_unchanged=True``), which is reported."""
    repo, rows, differ = Path(repo), {}, []
    for rel in files:
        f = repo / rel
        now = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
        at = _git_show(repo, record_commit_sha, rel)
        then = hashlib.sha256(at).hexdigest() if at is not None else None
        rows[rel] = {"same": now is not None and now == then}
        if not rows[rel]["same"]:
            differ.append(rel)
    excepted = bool(differ) and rescored_unchanged is True
    return {"files": rows, "differ": differ, "afterAdaptation": bool(differ) and not excepted, "rescoreException": excepted}


def rescore_changes(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Item ids whose outcome differs between two scorings (the defect-fix exception needs this to be empty)."""
    return sorted(i for i in set(a) | set(b) if a.get(i) != b.get(i))


def scope_patterns(eval_manifest: Mapping) -> list[str]:
    """The gate's frame is the census scope: ``exclusions.paths`` plus the content-check files, read from ``eval-manifest.yaml``
    (not the system manifest; ``-p1-ml.md`` §6.3)."""
    ex = eval_manifest.get("exclusions") or {}
    files = [f["path"] for f in (ex.get("contentCheck") or {}).get("files") or []]
    return list(ex.get("paths") or []) + files


def scope_disagreement(eval_manifest: Mapping, system_manifest: Mapping) -> dict:
    """The system manifest's first repository's ``exclusions`` against the eval manifest's; a difference is a reported defect."""
    ex = ((system_manifest.get("repos") or [{}])[0].get("exclusions")) or {}
    sys_patterns = list(ex.get("paths") or []) + list((ex.get("contentCheck") or {}).get("files") or [])
    ev = scope_patterns(eval_manifest)
    return {"agree": sorted(sys_patterns) == sorted(ev), "onlyInEval": len(set(ev) - set(sys_patterns)),
            "onlyInSystem": len(set(sys_patterns) - set(ev))}


def scope_report(records: Sequence[Mapping], patterns: Sequence[str]) -> dict:
    """Held-out items whose impacted (or anchor) file lies in the excluded scope: ``out-of-scope``, triaged ``key-error``, never scored."""
    flagged = [r["itemId"] for r in records if item_out_of_scope(r, patterns)]
    return {"outOfScope": len(flagged), "triage": {"key-error": len(flagged)}, "itemIds": flagged}


def item_out_of_scope(rec: Mapping, patterns: Sequence[str]) -> bool:
    paths = [rec.get(k, {}).get("path") for k in ("impacted", "anchor")]
    return any(p and _noise.path_excluded(p, list(patterns)) for p in paths)


def build_at_eval_commit(eval_commit: str, dest: Path, repo: Path | None = None, refresh_fn: Callable[..., Any] | None = None,
                         system: str = "ccg", **refresh_kw) -> Path:
    """The gate build: a LOCAL CLONE of the repository at ``EVAL_COMMIT`` under ``dest`` (the custodian root; never the user's tree),
    then the P1-20 refresh (``ccg.refresh.refresh``) with that clone as ``repo_root``. ``refresh_fn`` is injected in tests."""
    if not re.fullmatch(r"[0-9a-f]{40}", eval_commit):
        raise Refusal("EVAL_COMMIT is not a full 40-hex id")
    repo, dest = Path(repo) if repo else _CPG.parent, Path(dest)
    if dest.exists():
        raise Refusal("the build directory already exists")
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(repo), str(dest)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", "--detach", eval_commit], check=True, capture_output=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if head != eval_commit:
        raise Refusal("the build worktree is not at EVAL_COMMIT")
    if refresh_fn is None:
        from ccg import refresh as _r
        refresh_fn = _r.refresh
    refresh_fn(system, repo_root=dest, **refresh_kw)
    return dest


# =============================================================================================
# the gate snapshot (EP §5.2.1 HA-1: replay against the stored graph)
# =============================================================================================

#: hop ``kind`` -> (graph label, id property) (EP §5.2.1 HA-1)
LABELS = {"Symbol": "symbolId", "CallSite": "callSiteId", "Param": "pointId", "Ret": "pointId", "Cond": "pointId",
          "ContractPoint": "cpId", "Link": "linkId", "ConfigEntry": "entryId"}
_ANY_SRC = ("CallSite", "Ret", "Cond")
#: ``edgeType`` -> (row class, stored (source label, target label) pairs). R replayed now, D deferred (loaded only if the snapshot has it),
#: X cannot arrive (a harness error). The 21-string allow-list of EP §5.2.1; a relationship type is never a query parameter.
EDGES: dict[str, tuple[str, tuple]] = {
    "CALLS": ("R", (("CallSite", "Symbol"),)), "CALLS_NAME": ("R", (("CallSite", "Symbol"),)),
    "HAS_CALL": ("R", (("Symbol", "CallSite"),)), "HAS_PARAM": ("R", (("Symbol", "Param"),)),
    "HAS_RETURN": ("R", (("Symbol", "Ret"),)), "HAS_COND": ("R", (("Symbol", "Cond"),)),
    "DATA": ("R", tuple((s, d) for s in _ANY_SRC for d in ("Param", "CallSite"))),
    "CONTROLLED_BY": ("R", tuple((s, "Cond") for s in (*_ANY_SRC, "Param"))),
    "VIA": ("R", (("CallSite", "Link"),)), "TO": ("R", (("Link", "Symbol"),)), "ON": ("R", (("Link", "ContractPoint"),)),
    "CONSUMES": ("R", (("CallSite", "ContractPoint"),)), "PROVIDES": ("R", (("Symbol", "ContractPoint"),)),
    "READS": ("D", (("CallSite", "ContractPoint"),)), "WRITES": ("D", (("CallSite", "ContractPoint"), ("ConfigEntry", "ContractPoint"))),
    "BOUND_BY": ("D", (("Link", "ConfigEntry"),)), "RAN": ("D", (("Symbol", "Symbol"),)),
    "DEFINED_BY": ("X", ()), "LAUNCHES": ("X", ()), "DEPENDS_ON": ("X", ()), "FROM_DEP": ("X", ()),
}


def _edge_cypher(etype: str, sl: str, dl: str) -> str:
    """Only closed-set strings reach the query text (``etype`` from :data:`EDGES`, labels and id properties from :data:`LABELS`)."""
    assert etype in EDGES and sl in LABELS and dl in LABELS
    sp, dp = LABELS[sl], LABELS[dl]
    if etype == "CALLS_NAME":
        return f"MATCH (a:{sl} {{{sp}: $a}})-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(b:{dl} {{{dp}: $b}}) RETURN count(*)"
    return f"MATCH (a:{sl} {{{sp}: $a}})-[:{etype}]->(b:{dl} {{{dp}: $b}}) RETURN count(*)"


class StoreSnapshot:
    """The gate snapshot over a read-only runner ``run(cypher, params) -> rows`` (``GRAPH.RO_QUERY``; ``snapshot_export.RoRunner``).
    Reads are memoized: one gate answers each question once."""

    def __init__(self, run: Callable[[str, dict | None], list]):
        self._run, self._memo = run, {}

    def _q(self, cypher: str, params: dict | None = None) -> list:
        key = (cypher, json.dumps(params, sort_keys=True))
        if key not in self._memo:
            self._memo[key] = self._run(cypher, params)
        return self._memo[key]

    def snapshot_id(self) -> str:
        rows = self._run("MATCH (s:Snapshot) RETURN s.snapshotId", None)
        if len(rows) != 1:
            raise HarnessError(f"snapshot: {len(rows)} Snapshot nodes, expected 1")
        return rows[0][0]

    def node_exists(self, label: str, nid: str) -> bool:
        return bool(self._q(f"MATCH (n:{label} {{{LABELS[label]}: $id}}) RETURN count(n)", {"id": nid})[0][0])

    def edge_joins(self, etype: str, src: tuple, dst: tuple) -> bool:
        return bool(self._q(_edge_cypher(etype, src[0], dst[0]), {"a": src[1], "b": dst[1]})[0][0])

    def edge_type_loaded(self, etype: str) -> bool:
        assert etype in EDGES
        return bool(self._q(f"MATCH ()-[r:{etype}]->() RETURN r LIMIT 1"))

    @staticmethod
    def _regions(v):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                return None
        return v

    def _symbol_row(self, r) -> dict:
        d = {"symbolId": r[0], "qname": r[1], "path": r[2], "lineStart": r[3], "lineEnd": r[4]}
        owned = self._regions(r[5])
        return {**d, **({"ownedRegions": owned} if owned else {})}

    def symbol(self, sid: str) -> dict | None:
        rows = self._q("MATCH (s:Symbol {symbolId: $id}) RETURN s.symbolId, s.qname, s.path, s.lineStart, s.lineEnd, s.ownedRegions", {"id": sid})
        return self._symbol_row(rows[0]) if rows else None

    def config_entry(self, eid: str) -> dict | None:
        rows = self._q("MATCH (c:ConfigEntry {entryId: $id}) RETURN c.entryId, c.path, c.keyPath", {"id": eid})
        return {"entryId": rows[0][0], "path": rows[0][1], "keyPath": rows[0][2]} if rows else None

    def symbols_of_path(self, path: str) -> list[dict]:
        rows = self._q("MATCH (s:Symbol {path: $p}) RETURN s.symbolId, s.qname, s.path, s.lineStart, s.lineEnd, s.ownedRegions ORDER BY s.symbolId",
                       {"p": path})
        return [self._symbol_row(r) for r in rows]


# =============================================================================================
# C5: stamp, request echo, chain replay (HA-4, HA-3, HA-1)
# =============================================================================================

def check_stamp(ans: Mapping, pinned_snapshot_id: str) -> list[str]:
    """HA-4: ``snapshotId`` equals the id read from the ``:Snapshot`` node, ``verdictLogOffset == 0``, ``verdictLogHash`` is the literal. -> problems."""
    st = ans.get("stamp") or {}
    out = []
    if st.get("snapshotId") != pinned_snapshot_id:
        out.append("snapshotId")
    if st.get("verdictLogOffset") != 0 or isinstance(st.get("verdictLogOffset"), bool):
        out.append("verdictLogOffset")
    if st.get("verdictLogHash") != EMPTY_VERDICT_LOG_HASH:
        out.append("verdictLogHash")
    return out


def _anchor_ids(req: Mapping) -> set:
    return {a.get("symbolId") or a.get("cpId") or a.get("entryId") for a in req.get("anchors") or []}


def check_request_echo(ans: Mapping, submitted: Mapping) -> dict:
    """HA-3: ``status == resolved`` and ``request`` equals the submitted one (useCase, direction, anchor id set, no ``scope``)."""
    interp = ans.get("interpretation") or {}
    if interp.get("status") != "resolved":
        return {"state": "not-resolved", "lookups": len(interp.get("lookups") or [])}
    req = interp.get("request") or {}
    same = (req.get("useCase") == submitted.get("useCase") and req.get("direction") == submitted.get("direction")
            and _anchor_ids(req) == _anchor_ids(submitted) and "scope" not in req and "scope" not in submitted)
    return {"state": "ok" if same else "unequal", "lookups": 0}


def replay_chain(chain: Sequence[Mapping], snap) -> tuple[str, str]:
    """HA-1 replay of one chain against the gate snapshot -> ``("ok"|"fail"|"error", code)``. ``fail`` fails H3 (a miss); ``error`` is a
    harness error (an ``edgeType`` outside the allow-list, a class-D row against a snapshot that does not load it, a class-X row, a bad ``dir`` or kind):
    neither a hit nor a miss, and the gate verdict is blocked."""
    if not chain:
        return "fail", "empty-chain"
    for h in chain:
        if h.get("kind") not in LABELS:
            return "error", "hop-kind"
    for i in range(1, len(chain)):
        h = chain[i]
        et, d = h.get("edgeType"), h.get("dir")
        if et not in EDGES:
            return "error", "edgeType-not-allowed"
        if d not in ("->", "<-"):
            return "error", "dir"
        cls = EDGES[et][0]
        if cls == "X":
            return "error", "class-X"
        if cls == "D" and not snap.edge_type_loaded(et):
            return "error", "class-D-not-loaded"
    for h in chain:
        if not snap.node_exists(h["kind"], h["id"]):
            return "fail", "node-absent"
    for i in range(1, len(chain)):
        prev, h = chain[i - 1], chain[i]
        a, b = ((prev, h) if h["dir"] == "->" else (h, prev))
        if (a["kind"], b["kind"]) not in EDGES[h["edgeType"]][1]:
            return "fail", "kind-pair"
        if not snap.edge_joins(h["edgeType"], (a["kind"], a["id"]), (b["kind"], b["id"])):
            return "fail", "no-edge"
    return "ok", ""


# =============================================================================================
# the hit rule (EP §5.2 H1-H4) with the HA-2 / HA-5 listings
# =============================================================================================

class Acc:
    """What one gate run lists besides hits and misses (counts by kind; never content)."""

    def __init__(self):
        self.defects, self.harness_errors, self.masked = Counter(), Counter(), 0


def mask_keypath(s: str) -> str:
    from ccg import answer as _answer
    return _answer.mask_identifier("keyPath", s)


def _subject_id(subj: Mapping) -> str | None:
    return subj.get("symbolId") or subj.get("entryId") or subj.get("unitId")


def _h1(x: Mapping, rec: Mapping, snap, acc: Acc) -> bool:
    """H1 on the SNAPSHOT's values (HA-2): the answer's copies are compared and a mismatch is listed as an engine defect."""
    subj, e = x.get("subject") or {}, rec["impacted"]
    if e.get("type") == "symbol":
        if subj.get("type") != "symbol":
            return False
        real = snap.symbol(subj.get("symbolId"))
        if real is None:
            acc.defects["subject-not-in-snapshot"] += 1
            return False
        for k in ("qname", "lineStart", "lineEnd", "ownedRegions"):
            if subj.get(k) != real.get(k):
                acc.defects[f"subject-{k}"] += 1
        return _mk.h1_symbol_credit(real, {"qname": e["qname"]}, rec["lines"])
    if e.get("type") == "config":
        if subj.get("type") != "config":
            return False
        real = snap.config_entry(subj.get("entryId"))
        if real is None:
            acc.defects["subject-not-in-snapshot"] += 1
            return False
        for k in ("path", "keyPath"):
            if subj.get(k) != real.get(k):
                acc.defects[f"subject-{k}"] += 1
        if real["path"] != e["path"]:
            return False
        got, want = mask_keypath(real["keyPath"]), mask_keypath(e["keyPath"])
        if got != real["keyPath"] or want != e["keyPath"]:
            acc.masked += 1                                  # HA-5: every comparison where masking changed a segment is listed
        return got == want or got.startswith(want + ".") or got.startswith(want + "[")
    return False


def candidates(ans: Mapping) -> Iterable[Mapping]:
    """H4: *x* is an item or a group MEMBER; a group's header or count is never *x*."""
    yield from ans.get("items") or []
    for g in ans.get("groups") or []:
        yield from g.get("members") or []


def evaluate_item(rec: Mapping, ask_id: str, ans: Mapping, snap, acc: Acc) -> dict:
    """The hit decision for one key item against its anchor's answer. -> ``{"hit", "stage", ["chain"]}``; ``stage`` is the deepest of H1-H3
    any candidate reached (``none`` when no candidate is *e*): the triage hint, with ``extractionAbsent`` and ``depthBoundCut`` flags."""
    stage_rank = {"none": 0, "H1": 1, "H2": 2, "H3": 3}
    best, hit_chain = "none", None
    for x in candidates(ans):
        if not _h1(x, rec, snap, acc):
            continue
        if x.get("status") not in STATUSES:                                  # H2
            best = max(best, "H1", key=stage_rank.get)
            continue
        chain = x.get("chain") or []
        subj_id = _subject_id(x.get("subject") or {})
        if not chain or chain[0].get("id") != ask_id or chain[-1].get("id") != subj_id:      # H3, shape
            best = max(best, "H2", key=stage_rank.get)
            continue
        verdict, code = replay_chain(chain, snap)                           # H3, replay (HA-1)
        if verdict == "error":
            acc.harness_errors[code] += 1
            best = max(best, "H2", key=stage_rank.get)
            continue
        if verdict == "fail":
            best = max(best, "H2", key=stage_rank.get)
            continue
        if hit_chain is None or len(chain) < len(hit_chain):
            hit_chain = chain
    if hit_chain is not None:
        return {"hit": True, "stage": "H3", "chain": hit_chain}
    e = rec["impacted"]
    absent = e.get("type") == "symbol" and _mk.source_check_needed({"qname": e["qname"]}, rec["lines"], snap.symbols_of_path(e["path"]))
    return {"hit": False, "stage": best, "extractionAbsent": bool(absent),
            "depthBoundCut": bool((ans.get("depthBound") or {}).get("unexplored"))}


# =============================================================================================
# asking and scoring a gate run
# =============================================================================================

def unit_probe(rec: Mapping, records: Sequence[Mapping]) -> str:
    """EP §5.1: a ``unit`` anchor is asked from its probe: of its ``serves[]`` points, the one with the fewest items in the list, ties by
    smallest ``cpId``."""
    served = sorted({_mk.contract_id(loc) for loc in rec.get("serves") or []})
    if not served:
        raise HarnessError("unit anchor without a served contract point")
    n = Counter(r["anchorId"] for r in records)
    return min(served, key=lambda c: (n.get(c, 0), c))


def ask_id_of(rec: Mapping, records: Sequence[Mapping]) -> str:
    return unit_probe(rec, records) if rec["anchor"].get("type") == "unit" else rec["anchorId"]


def build_request(rec: Mapping, ask_id: str) -> dict:
    """The structured impact request of EP §5.1 (never plain language, no ``scope``)."""
    a = rec["anchor"]
    if a["type"] == "symbol":
        anchor = {"type": "symbol", "symbolId": ask_id, "qname": a["qname"], "repoId": a["repoId"], "path": a["path"]}
    elif a["type"] == "config":
        anchor = {"type": "config", "entryId": ask_id, "repoId": a["repoId"], "path": a["path"], "keyPath": a["keyPath"]}
    else:
        loc = a if a["type"] == "contract" else next(x for x in rec["serves"] if _mk.contract_id(x) == ask_id)
        anchor = {"type": "contract", "cpId": ask_id, "kind": loc["kind"], "keyNorm": loc["keyNorm"]}
    return {"useCase": "impact", "direction": "impact", "anchors": [anchor]}


def _asks(records: Sequence[Mapping]) -> dict:
    out = {}
    for r in sorted(records, key=lambda r: r["itemId"]):
        out.setdefault(ask_id_of(r, records), r)
    return out


def check_cache_empty(cache_len: Callable[[], int]) -> None:
    """C5: the interpretation/answer cache is asserted empty for the gate snapshot before the first question."""
    n = cache_len()
    if n != 0:
        raise Refusal(f"C5: the cache is not empty for the gate snapshot ({n} entries); the run stops")


def check_repeat_cache_hits(repeats: Sequence[Mapping]) -> dict:
    """C5 for AC-24: every repeat carries ``provenance.cacheHit == false``. ``repeats`` is ``[{"repeat": 1..5, "answer": doc}]`` for one item.
    A ``true`` on repeat 1 STOPS the run (the cache was not empty); a ``true`` on repeats 2-5 VOIDS that item's repeats."""
    def hit(r):
        prov = ((r.get("answer") or {}).get("interpretation") or {}).get("provenance") or {}
        return prov.get("cacheHit") is not False
    bad = sorted(r["repeat"] for r in repeats if hit(r))
    return {"stop": 1 in bad, "voided": bool(bad) and 1 not in bad, "badRepeats": bad}


def run_asks(records: Sequence[Mapping], snap, ask: Callable[[dict], Mapping], cache_len: Callable[[], int]) -> dict:
    """One question per distinct asked anchor, after the cache assertion; ``snapshotId`` is read from the ``:Snapshot`` node BEFORE the first
    question. A contract anchor whose ``cpId`` is absent from the gate graph is not asked (an ``anchor-not-in-graph`` miss)."""
    check_cache_empty(cache_len)
    pinned = snap.snapshot_id()
    answers = {}
    for aid, rec in _asks(records).items():
        if rec["anchor"].get("type") in ("contract", "unit") and not snap.node_exists("ContractPoint", aid):
            continue
        req = build_request(rec, aid)
        answers[aid] = {"request": req, "answer": ask(req)}
    return {"snapshotId": pinned, "answers": answers}


def score_gate(records: Sequence[Mapping], run: Mapping, snap, patterns: Sequence[str]) -> dict:
    """The hit rule over every key item (``run`` is :func:`run_asks`' result). -> per-item outcomes plus what the run lists.

    Outcomes: ``hit``, ``miss`` (with ``reason``), ``out-of-scope`` (triaged ``key-error``; never a hit, a miss or a wrong-possible item),
    ``void`` (a C5 stamp failure: the answer is void and the run is reported). Held-out ids never leave this structure but the custodian file."""
    acc, items, stops = Acc(), {}, Counter()
    pinned, answers = run["snapshotId"], run["answers"]
    anchor_state = {}
    for aid, entry in answers.items():
        bad = check_stamp(entry["answer"], pinned)
        echo = check_request_echo(entry["answer"], entry["request"])
        if echo["state"] == "unequal":
            acc.defects["request-echo"] += 1
        anchor_state[aid] = {"stamp": bad, "echo": echo}
        if bad:
            stops["C5:stamp:" + ",".join(bad)] += 1
    for rec in sorted(records, key=lambda r: r["itemId"]):
        iid, stratum = rec["itemId"], rec["derivedStratum"]
        base = {"stratum": stratum, "anchorId": rec["anchorId"], "counting": bool(rec.get("counting"))}
        if item_out_of_scope(rec, patterns):
            items[iid] = {**base, "outcome": "out-of-scope", "triage": "key-error"}
            continue
        aid = ask_id_of(rec, records)
        if aid not in answers:
            items[iid] = {**base, "outcome": "miss", "reason": "anchor-not-in-graph"}
            continue
        st = anchor_state[aid]
        if st["stamp"]:
            items[iid] = {**base, "outcome": "void", "reason": "stamp"}
        elif st["echo"]["state"] != "ok":
            items[iid] = {**base, "outcome": "miss", "reason": "not-resolved" if st["echo"]["state"] == "not-resolved" else "request-unequal"}
        else:
            r = evaluate_item(rec, aid, answers[aid]["answer"], snap, acc)
            items[iid] = {**base, "outcome": "hit" if r["hit"] else "miss",
                          **({"hops": len(r["chain"]), "interFunctionHops": sum(1 for h in r["chain"][1:] if h["kind"] == "Symbol")} if r["hit"]
                             else {"reason": "no-qualifying-item", "stage": r["stage"], "extractionAbsent": r["extractionAbsent"],
                                   "depthBoundCut": r["depthBoundCut"]})}
    return {"items": items, "defects": dict(acc.defects), "harnessErrors": dict(acc.harness_errors), "maskedComparisons": acc.masked,
            "stops": dict(stops)}


# =============================================================================================
# the fail procedure (EP §12.1) and the report (EP §5.3, §12.1 step 6; ``-p1-ml.md`` §6.1)
# =============================================================================================

FLAGS = ("kept", "kept-cluster-fresh", "fresh-cluster")          # ``regate-flags.jsonl`` (split_manifest.regate-list)


def _scored(result: Mapping) -> dict:
    """Items that are a hit or a miss. Out-of-scope items (triaged key-error) and void items are neither."""
    return {i: o for i, o in result["items"].items() if o["outcome"] in ("hit", "miss")}


def _counts(items: Mapping) -> dict:
    sc = {i: o for i, o in items.items() if o["outcome"] in ("hit", "miss")}
    by = defaultdict(int)
    for o in sc.values():
        by[o["stratum"]] += 1
    return {"scored": len(sc), "counting": sum(1 for o in sc.values() if o["counting"]), "byStratum": dict(by)}


def decide_gate(result: Mapping, flags: Mapping[str, str] | None = None, moved_out: Mapping[str, int] | None = None) -> dict:
    """EP §12.1. With no ``flags`` this is gate 1: pass iff every scored item is a hit over >= 75 counting items and >= 8 items per stratum.
    With ``flags`` (itemId -> kept / kept-cluster-fresh / fresh-cluster) it is a re-gate: a pass needs (1) every item a hit, (2) the composition
    rule (H >= 75, every H_s >= 8, and per stratum that had a miss at least as many fresh-cluster items as were ``moved_out``) and (3) at least
    F = 35 COUNTING items on fresh clusters; (1) and (2) without (3) is "all hits, floor not met", never a pass.

    Precedence: ``stopped`` (a C5 stamp failure) > ``fail`` (a miss, reported as run) > ``blocked`` (an engine defect, a harness error or a
    listed masked comparison, which block a pass) > the re-gate outcomes > ``pass``. Out-of-scope items are neither hits nor misses."""
    items = _scored(result)
    misses = sorted(i for i, o in items.items() if o["outcome"] == "miss")
    voided = sorted(i for i, o in result["items"].items() if o["outcome"] == "void")
    blockers = []
    if result.get("defects"):
        blockers.append("engine-defects")
    if result.get("harnessErrors"):
        blockers.append("harness-errors")
    if result.get("maskedComparisons"):
        blockers.append("masked-comparisons")
    cnt = _counts(result["items"])
    strata_ok = all(cnt["byStratum"].get(s, 0) >= MIN_STRATUM for s in STRATA)
    base = {"misses": len(misses), "voided": len(voided), "blockers": blockers, "H": cnt["counting"], "scored": cnt["scored"],
            "minimumsMet": cnt["counting"] >= MIN_COUNTING and strata_ok, "byStratum": cnt["byStratum"],
            "missClusters": sorted({result["items"][i]["anchorId"] for i in misses}), "movedOutByStratum": dict(Counter(items[i]["stratum"] for i in misses))}
    if result.get("stops") or voided:
        return {**base, "verdict": "stopped", "kind": "gate" if flags is None else "re-gate"}
    if misses:
        return {**base, "verdict": "fail", "kind": "gate" if flags is None else "re-gate", "rule": "the first run is the result (EP §12.1 step 1)"}
    if flags is None:
        v = "blocked" if blockers else ("pass" if base["minimumsMet"] else "minimums-not-met")
        return {**base, "verdict": v, "kind": "gate"}
    fresh = [i for i in items if flags.get(i) == "fresh-cluster"]
    fresh_counting = sum(1 for i in fresh if items[i]["counting"])
    fresh_by = Counter(items[i]["stratum"] for i in fresh)
    need = dict(moved_out or {})
    shortfall = {s: n - fresh_by.get(s, 0) for s, n in need.items() if fresh_by.get(s, 0) < n}
    composition = base["minimumsMet"] and not shortfall
    floor = fresh_counting >= FLOOR
    out = {**base, "kind": "re-gate", "freshCounting": fresh_counting, "floor": FLOOR, "floorMet": floor, "compositionMet": composition,
           "compositionShortfall": shortfall}
    if blockers:
        return {**out, "verdict": "blocked"}
    if not composition:
        return {**out, "verdict": "all hits, composition not met"}
    if not floor:
        return {**out, "verdict": "all hits, floor not met"}
    return {**out, "verdict": "pass"}


def first_run_marker(custodian: Path, list_sha256: str, payload: Mapping) -> Path:
    """EP §12.1 step 7: the gate is never re-run on the list that was run. The marker is created atomically; an existing one refuses the run."""
    marker = Path(custodian) / f"gate-run-{list_sha256[:16]}.marker"
    try:
        _ac.take_marker(marker, payload)
    except _ac.Refusal:
        raise Refusal("this list was already run: a re-run on the same list is refused (EP §12.1 step 7); a new list is required") from None
    return marker


# --- intervals -------------------------------------------------------------------------------

_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
         2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def t975(df: int) -> float:
    if df < 1:
        raise ValueError("df < 1")
    if df <= len(_T975):
        return _T975[df - 1]
    z = 1.959964                                                  # Cornish-Fisher expansion of the t quantile
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def ratio_interval(units: Sequence[tuple[int, int]]) -> dict:
    """P0-ML §7.3's linearized ratio interval over cluster units ``(m_i, y_i)`` (items, hits): ``R = sum y / sum m``,
    ``V = sum (y_i - R m_i)^2 / (n (n - 1) m_bar^2)``, ``R +- t(0.975, n-1) sqrt(V)`` clipped to [0, 1]. A unit with ``m_i = 0`` is dropped."""
    u = [(m, y) for m, y in units if m > 0]
    n = len(u)
    if n < 2:
        return {"n": n, "R": None, "lower": None, "upper": None}
    sm, sy = sum(m for m, _ in u), sum(y for _, y in u)
    r = sy / sm
    mbar = sm / n
    var = sum((y - r * m) ** 2 for m, y in u) / (n * (n - 1) * mbar ** 2)
    half = t975(n - 1) * math.sqrt(var)
    return {"n": n, "R": round(r, 3), "lower": round(max(0.0, r - half), 3), "upper": round(min(1.0, r + half), 3)}


def _recall(outcomes: Sequence[Mapping], key: str, cluster_of: Mapping | None = None) -> dict:
    """Recall with the unit being the item, the anchor or the cluster. A unit counts only if ALL its items are hits."""
    if key == "item":
        k = sum(1 for o in outcomes if o["outcome"] == "hit")
        return _wilson(k, len(outcomes))
    unit = (lambda o: o["anchorId"]) if key == "anchor" else (lambda o: (cluster_of or {}).get(o["anchorId"], o["anchorId"]))
    g = defaultdict(list)
    for o in outcomes:
        g[unit(o)].append(o["outcome"] == "hit")
    return _wilson(sum(1 for v in g.values() if all(v)), len(g))


def _rec_block(outcomes, cluster_of) -> dict:
    units = defaultdict(lambda: [0, 0])
    for o in outcomes:
        c = (cluster_of or {}).get(o["anchorId"], o["anchorId"])
        units[c][0] += 1
        units[c][1] += o["outcome"] == "hit"
    return {"items": _recall(outcomes, "item"), "anchors": _recall(outcomes, "anchor"), "clusters": _recall(outcomes, "cluster", cluster_of),
            "clusterUnit": "cluster" if cluster_of else "anchor", "clusterRatioInterval": ratio_interval([tuple(v) for v in units.values()])}


def _dist(vals: Sequence[int]) -> dict:
    return {"n": len(vals), "median": float(_noise.median(list(vals))) if vals else None, "p90": _noise.p90(list(vals)) if vals else None}


def build_report(result: Mapping, decision: Mapping, *, flags: Mapping[str, str] | None = None, cluster_of: Mapping | None = None,
                 exposure: Sequence[Mapping] = (), exposure_strata: Mapping | None = None, context: Mapping | None = None) -> dict:
    """Counts, hashes and classes only (never item content, never an id). Recall is given over items, distinct anchors and clusters, per
    stratum and pooled, each with its Wilson interval, and the cluster interval of P0-ML §7.3 (anchors as clusters when no map is given).
    ``context`` carries the preflight blocks (first contact, pins, harness pin, scope) verbatim."""
    items = _scored(result)
    rows = list(items.items())
    per = {s: _rec_block([o for _, o in rows if o["stratum"] == s], cluster_of) for s in STRATA if any(o["stratum"] == s for _, o in rows)}
    miss = [o for _, o in rows if o["outcome"] == "miss"]
    oos = [o for o in result["items"].values() if o["outcome"] == "out-of-scope"]
    rep = {"verdict": decision["verdict"], "kind": decision["kind"], "H": decision["H"], "H_all": len(rows), "scored": len(rows),
           "minimumsMet": decision["minimumsMet"], "blockers": decision["blockers"],
           "recall": {"pooled": _rec_block([o for _, o in rows], cluster_of), "byStratum": per},
           "misses": {"count": len(miss), "byReason": dict(Counter(o["reason"] for o in miss)), "byStage": dict(Counter(o.get("stage", "-") for o in miss)),
                      "anchorNotInGraph": sum(1 for o in miss if o["reason"] == "anchor-not-in-graph"),
                      "extractionAbsent": sum(1 for o in miss if o.get("extractionAbsent")), "depthBoundCut": sum(1 for o in miss if o.get("depthBoundCut")),
                      "triage": {c: 0 for c in TRIAGE}, "triageNote": "to be filled by the gate runner and the method owner (EP §12.1 step 2)"},
           "outOfScope": {"count": len(oos), "triage": {"key-error": len(oos)}},
           "voided": decision["voided"], "stops": dict(result.get("stops", {})), "engineDefects": dict(result.get("defects", {})),
           "harnessErrors": dict(result.get("harnessErrors", {})), "maskedComparisons": result.get("maskedComparisons", 0),
           "chainLength": {s: _dist([o["interFunctionHops"] for _, o in rows if o["stratum"] == s and o["outcome"] == "hit"]) for s in per},
           "chainLengthUnit": "inter-function hops: Symbol hops after the first, over hit items (the shortest qualifying chain)",
           "units": "items, distinct anchors, clusters (`clusterUnit` says which map was given)"}
    if flags is not None:
        rep["regate"] = _regate_block(items, flags, decision, exposure, exposure_strata)
    rep.update(context or {})
    return rep


def _regate_block(items: Mapping, flags: Mapping, decision: Mapping, exposure: Sequence[Mapping], exposure_strata: Mapping | None = None) -> dict:
    """EP §12.1 step 6: fresh-cluster recall, kept items as a regression check (never recall evidence), the cumulative first-exposure record."""
    def fig(sel):
        rows = [o for i, o in items.items() if flags.get(i) == sel]
        return {"k": sum(1 for o in rows if o["outcome"] == "hit"), "n": len(rows), "wilson": _wilson(sum(1 for o in rows if o["outcome"] == "hit"), len(rows))}
    fresh = fig("fresh-cluster")
    prior = {r["itemId"]: r for r in exposure}
    # each item once, at the first gate that exposed it: every row already on record (listed now or not), then this gate's items not on record
    cum_rows = [((exposure_strata or {}).get(i), bool(r["hit"])) for i, r in prior.items()]
    cum_rows += [(o["stratum"], o["outcome"] == "hit") for i, o in items.items() if i not in prior]
    known = sorted({st for st, _ in cum_rows if st})
    cum = {"pooled": _wilson(sum(h for _, h in cum_rows), len(cum_rows)),
           "byStratum": {s: _wilson(sum(h for st, h in cum_rows if st == s), sum(1 for st, _ in cum_rows if st == s)) for s in known},
           "unassignedStratum": sum(1 for st, _ in cum_rows if not st)}
    claim = (f"fresh-cluster {fresh['k']}/{fresh['n']}, Wilson lower bound {fresh['wilson']['lower']}; the cumulative record is given beside it"
             if decision["verdict"] == "pass" else "no AC-5 pass claimed")
    return {"freshCluster": fresh, "keptRegressionCheck": {**fig("kept"), "note": "not recall evidence: selected on having passed"},
            "keptClusterFresh": {**fig("kept-cluster-fresh"), "note": "scored, not fresh-anchor evidence"}, "cumulativeFirstExposure": cum,
            "freshCounting": decision.get("freshCounting"), "floor": FLOOR, "floorMet": decision.get("floorMet"), "claim": claim,
            "note": "0.951 is claimed only for a first gate and is never claimed for a re-gate (EP §5.3)"}


def exposure_rows(result: Mapping, flags: Mapping | None = None) -> list[dict]:
    """``{itemId, hit}`` for every scored item, for the ``exposure`` verb (custodian file only)."""
    return [{"itemId": i, "hit": o["outcome"] == "hit"} for i, o in sorted(_scored(result).items())]


def write_custodian(root: Path, name: str, rows: Sequence[Mapping] | Mapping, home: Path | None = None) -> Path:
    """Held-out-derived output: refused unless ``root`` is the custodian area ``~/.local/share/ccg-eval/<...>`` (``-p1-ml.md`` §6.4)."""
    root = Path(root).resolve()
    area = _ac.custodian_area(home).resolve()
    if root != area and area not in root.parents:
        raise Refusal("REFUSED: held-out-derived files go under the custodian root, never elsewhere")
    root.mkdir(parents=True, exist_ok=True)
    f = root / name
    f.write_text(("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)) if not isinstance(rows, Mapping) else json.dumps(rows, sort_keys=True) + "\n",
                 encoding="utf-8")
    return f


# =============================================================================================
# CLI: ``verify`` (preflight, no held-out content read) and ``score`` (a recorded run against the gate snapshot)
# =============================================================================================

def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_yaml(p: Path) -> dict:
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def preflight(repo: Path, gate_id: str, *, observed: Mapping | None = None, tree: Mapping | None = None, committed_text=None,
              record_commit_sha: str | None = None) -> dict:
    """Everything that needs no held-out content: the four-field comparison and first-contact label, the pin strings (R-6), the harness
    self-pin and the scope agreement. Counts, hashes and classes only."""
    repo = Path(repo)
    prereg = repo / "cpg" / "eval" / "preregistration.jsonl"
    line, rec = find_prereg_line(prereg, gate_id, committed_text)
    tree = dict(tree) if tree is not None else observe_tree(repo)
    obs = {**tree, **(observed if observed is not None else engine_observed())}
    fc = first_contact(rec, obs)
    em = _load_yaml(repo / "cpg" / "eval" / "eval-manifest.yaml")
    hm = repo / "cpg" / "eval" / "heldout-manifest.json"
    pins = pin_report(em, repo, json.loads(hm.read_text()) if hm.is_file() else None)
    rc = record_commit_sha or record_commit(repo, "cpg/eval/preregistration.jsonl", line)
    pin = harness_pin(repo, rc) if rc else {"afterAdaptation": True, "differ": ["record-commit-not-found"], "files": {}, "rescoreException": False}
    sysm = repo / "cpg" / "ccg" / "system.yaml"
    scope = scope_disagreement(em, _load_yaml(sysm)) if sysm.is_file() else {"agree": None}
    return {"firstContact": fc, "pins": pins, "harnessPin": pin, "scope": scope, "recordCommit": rc,
            "label": "after-adaptation" if (not fc["firstContact"] or pin["afterAdaptation"]) else "first-contact"}


def main(argv=None, snapshot=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--gate-id", required=True)
    v.add_argument("--repo", default=str(_CPG.parent))
    v.add_argument("--record-commit")
    sc = sub.add_parser("score")
    for f in ("--gate-id", "--lists", "--run", "--custodian"):
        sc.add_argument(f, required=True)
    sc.add_argument("--flags")
    sc.add_argument("--exposure")
    sc.add_argument("--moved-out", help="JSON {stratum: n}")
    sc.add_argument("--host", default="127.0.0.1")
    sc.add_argument("--port", type=int)
    sc.add_argument("--graph")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "verify":
            rep = preflight(Path(a.repo), a.gate_id, record_commit_sha=a.record_commit)
            print(json.dumps(rep, sort_keys=True))
            return 0 if rep["label"] == "first-contact" else 1
        cust = Path(a.custodian).expanduser().resolve()
        area = _ac.custodian_area().resolve()
        if cust != area and area not in cust.parents:
            print("REFUSED: --custodian is outside ~/.local/share/ccg-eval", file=sys.stderr)
            return 2
        repo = _CPG.parent
        records = [r for f in a.lists.split(",") for r in _jsonl(Path(f))]
        require_clean_renormalize(renormalize_stored_gold(records))
        run = json.loads(Path(a.run).read_text(encoding="utf-8"))
        em = _load_yaml(repo / "cpg" / "eval" / "eval-manifest.yaml")
        if snapshot is None:
            from falkordb import FalkorDB
            sys.path.insert(0, str(_HERE))
            from snapshot_export import RoRunner
            snapshot = StoreSnapshot(RoRunner(FalkorDB(host=a.host, port=a.port).select_graph(a.graph)))
        flags = {r["itemId"]: r["flag"] for r in _jsonl(Path(a.flags))} if a.flags else None
        expo = _jsonl(Path(a.exposure)) if a.exposure else []
        res = score_gate(records, run, snapshot, scope_patterns(em))
        dec = decide_gate(res, flags, json.loads(a.moved_out) if a.moved_out else None)
        rep = build_report(res, dec, flags=flags, exposure=expo)
        write_custodian(cust, "gate-outcomes.jsonl", [{"itemId": i, **o} for i, o in sorted(res["items"].items())])
        write_custodian(cust, "gate-exposure.jsonl", exposure_rows(res))
        write_custodian(cust, "gate-report.json", rep)
        print(json.dumps(rep, sort_keys=True))
        return 0 if dec["verdict"] == "pass" else 1
    except Refusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, HarnessError) as e:
        print(f"bad input: {type(e).__name__}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
