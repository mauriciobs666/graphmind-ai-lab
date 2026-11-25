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
import ast
import hashlib
import json
import math
import contextlib
import datetime
import os
import re
import subprocess
import sys
import tempfile
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
import withheld_shapes as _ws               # noqa: E402  (the Q8 withheld-shape predicate: pure, stdlib only; in the harness pin)
import split_manifest as _sm                # noqa: E402  (verify_manifest)
from oracle import wilson as _wilson        # noqa: E402  (the one Wilson implementation)

PREREG_SCHEMA = "ccg-prereg-v1"
PREREG_FIELDS = ("packSetHash", "coreTreeHash", "algorithm", "defaultDepthBound")
FLOOR = 35                                   # EP §12.1 step 5 (E12)
MIN_COUNTING = 75                            # AC-5 pooled, over the gated strata
MIN_STRATUM = 8                              # AC-5 per GATED stratum (a withheld stratum has no floor)
EMPTY_VERDICT_LOG_HASH = "c482b1cf54631b37eb4572dc86a68fb1"     # HA-4: the literal, never ``answer.EMPTY_VERDICT_LOG_HASH``
STRATA = _mk.STRATA
STRATA_GATED = ("http", "mcp", "scripts")      # the gated strata of the first held-out spend (stakeholder, 2026-11-24, ``docs/plans/connected-code-graph-p1-closeout.md`` §1); the rest are withheld
#: the reason each non-gated stratum is withheld, reported beside its COUNT (never ids). ``flow`` is gated only when the flow flag is true (:func:`gated_strata`).
WITHHELD_REASONS = {
    "flow": "flow-unavailable, not closed: the language's flow flag is false, so flow items are neither asked nor scored (``-p1-gate-marker-ml.md`` §4)",
    "wiring": "withheld by stakeholder decision: wiring linking is not built (the dev wiring items need an answer-spec change), so wiring items are neither asked nor scored",
}
DEV_PROBE_SCHEMA = "ccg-dev-probe-v1"            # the dev-gold probe result the rehearsal writes (:func:`dev_probe_check`)
FLOW_DECISION = Path("cpg") / "eval" / "decisions" / "ccg-p1-flow-python-v2-decision.json"
ENTRY_POINTS = ("gate.py", "gate_run.py")                # the modules that run the gate; the pin is their import closure under ``cpg/eval/tools/``
STATUSES = ("confirmed", "possible")        # H2
TRIAGE = ("key-error", "extraction", "recognizer", "linker-engine", "depth-bound", "dynamic")    # EP §12.1 step 2


def _imported_names(src: str) -> set[str]:
    """Every module name the source can import, anywhere in it (top level or lazily inside a function): ``import a``, ``from a import b``,
    ``from . import a``, ``importlib.import_module("a")``, ``__import__("a")``. The first dotted component only."""
    out: set[str] = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            out.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            if n.level and not n.module:
                out.update(a.name for a in n.names)
            elif n.module:
                out.add(n.module.split(".")[0])
        elif isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            f = n.func
            if (isinstance(f, ast.Attribute) and f.attr == "import_module") or (isinstance(f, ast.Name) and f.id in ("__import__", "import_module")):
                out.add(n.args[0].value.split(".")[0])
    return out


def harness_files(tools_dir: Path | None = None, entries: Sequence[str] = ENTRY_POINTS) -> tuple[str, ...]:
    """The self-pin set (``-p1-ml.md`` §6.1), built from the import graph: every ``.py`` under ``tools_dir`` that the entry points import, directly,
    lazily or transitively, plus ``cpg/eval/EVAL_COMMIT``. A hand-listed set would silently miss a new import. A name that matches a file in the
    directory counts as that file (fail-closed against a shadowing module)."""
    tools = Path(tools_dir) if tools_dir else _HERE
    seen: list[str] = []
    todo = [e for e in entries if (tools / e).is_file()]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.append(name)
        for mod in _imported_names((tools / name).read_text(encoding="utf-8")):
            if (tools / f"{mod}.py").is_file() and f"{mod}.py" not in seen:
                todo.append(f"{mod}.py")
    return tuple(f"cpg/eval/tools/{n}" for n in sorted(seen)) + ("cpg/eval/EVAL_COMMIT",)


HARNESS_FILES = harness_files()


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


#: what an operator must do about the store once a build has loaded the live graph into it and the run was refused afterwards (D6): the list is untouched
#: (no marker), the store is not reusable. Counts and instructions only; shared by every refusal that can follow the build.
STORE_REMEDY = ("the list is not spent (no marker was taken), but the built graph stays in this store and the store refuses every later fresh run: "
                "for the next run provision a new empty store (create its ACL users again) and rebuild (a fresh build takes 16-21 minutes)")
SYSTEM_MANIFEST = Path("cpg") / "ccg" / "system.yaml"


def manifest_system(repo: Path) -> str | None:
    """The ``system`` the working tree's manifest (``cpg/ccg/system.yaml``) declares; ``None`` when it declares none. The manifest is read from the
    working tree: at ``EVAL_COMMIT`` it does not exist (it was added later), and the gate's preflight has already required ``cpg/`` clean."""
    doc = _load_yaml(Path(repo) / SYSTEM_MANIFEST)
    name = doc.get("system") if isinstance(doc, dict) else None
    return name if isinstance(name, str) else None


@contextlib.contextmanager
def _audit_dir_env(audit_dir):
    """The sandbox runner (``cpg/sandbox/offline-run.sh``, reached through ``extractors.indexers.run`` and ``extractors.joern.run``) appends to
    ``$CCG_AUDIT_DIR`` and reads no argument for it: re-point the variable at the private audit directory for the build, then restore it (D3)."""
    if audit_dir is None:
        yield
        return
    had = os.environ.get("CCG_AUDIT_DIR")
    os.environ["CCG_AUDIT_DIR"] = str(audit_dir)
    try:
        yield
    finally:
        pending = None
        for _ in range(2):                                          # a signal handler can raise between the two statements of the restore: retry once, then re-raise it
            try:
                if had is None:
                    os.environ.pop("CCG_AUDIT_DIR", None)
                else:
                    os.environ["CCG_AUDIT_DIR"] = had
                break
            except BaseException as exc:                            # noqa: BLE001 - re-raised below, after the variable is restored
                pending = pending or exc
        if pending is not None:
            raise pending


def require_complete_build(report) -> None:
    """A gate graph with a unit that is not ``fresh`` (``not-mapped`` or ``stale-kept``) is not the graph the held-out keys were written against: refuse, counts only."""
    units = getattr(report, "units", None)
    if not isinstance(units, list) or not units:
        raise Refusal("the build returned no unit report, so its completeness is unknown: refused. " + STORE_REMEDY)
    bad = Counter(getattr(u, "status", "?") for u in units if getattr(u, "status", None) != "fresh")
    if bad:
        raise Refusal(f"the build is incomplete: {sum(bad.values())} of {len(units)} units are not fresh ("
                      + ", ".join(f"{n} {k}" for k, n in sorted(bad.items())) + "): refused. " + STORE_REMEDY)


def build_at_eval_commit(eval_commit: str, dest: Path, repo: Path | None = None, refresh_fn: Callable[..., Any] | None = None,
                         system: str = "ccg", **refresh_kw) -> Path:
    """The gate build: a LOCAL CLONE of the repository at ``EVAL_COMMIT`` under ``dest`` (the custodian root; never the user's tree),
    then the P1-20 refresh (``ccg.refresh.refresh``) with that clone as ``repo_root``. ``refresh_fn`` is injected in tests. A private ``audit_dir`` in
    ``refresh_kw`` is also exported to the sandbox runner for the duration (D3), and the build is refused unless every unit came out ``fresh``."""
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
    with _audit_dir_env(refresh_kw.get("audit_dir")):
        report = refresh_fn(system, repo_root=dest, **refresh_kw)
    require_complete_build(report)
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


def region_pairs(v):
    """Flat ``[s0, e0, ...]`` (the store's and the answer's shape) -> ``[[s0, e0], ...]``; nested pairs pass through; ``None``/empty -> ``None``."""
    if not v:
        return None
    if all(isinstance(r, (list, tuple)) for r in v):
        return [list(r) for r in v]
    if len(v) % 2 or any(isinstance(n, (list, tuple, bool)) or not isinstance(n, int) for n in v):
        raise HarnessError("ownedRegions: not a flat even-length integer list")
    return [[v[i], v[i + 1]] for i in range(0, len(v), 2)]


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

    def snapshot_inputs(self) -> dict:
        """The ``:Snapshot`` node's identity inputs, for :func:`store_content_id`."""
        rows = self._run("MATCH (s:Snapshot) RETURN s.snapshotId, s.manifestHash, s.packSetHash, s.codeHash", None)
        if len(rows) != 1:
            raise HarnessError(f"snapshot: {len(rows)} Snapshot nodes, expected 1")
        return dict(zip(("snapshotId", "manifestHash", "packSetHash", "codeHash"), rows[0]))

    def unit_rows(self) -> list:
        """``[unitId, buildStatus, builtCommit, flowAvailable, failReason]`` of every ``:Unit`` node, sorted by id."""
        rows = self._run("MATCH (u:Unit) RETURN u.unitId, u.buildStatus, u.builtCommit, u.flowAvailable, u.failReason", None)
        return sorted(([r[0], r[1], r[2], bool(r[3]), r[4] or ""] for r in rows), key=lambda r: r[0])

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
        """``ownedRegions`` is stored flat (``[s0, e0, s1, e1, ...]``, ``ccg.assemble.rows._flatten_regions``); ``match_keys`` reads pairs, so the
        snapshot hands it pairs. An odd-length stored list is a harness error."""
        d = {"symbolId": r[0], "qname": r[1], "path": r[2], "lineStart": r[3], "lineEnd": r[4]}
        owned = region_pairs(self._regions(r[5]))
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
    if not all(isinstance(h, dict) and isinstance(h.get("id"), str) for h in chain):
        return "error", "hop-shape"
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


def stored_keypath(key_path: str) -> str:
    """A ``keyPath`` in the form the build STORES (``ccg.assemble.rows._mask_key_path``: each screen-failing segment ``[#…]``). A value already in
    that form (it holds the loader's mask) is returned as it is: the loader's function is not idempotent (it masks a stored path whole). Not
    :func:`mask_keypath` (the ``[#]`` form an ANSWER prints): the store, and so ``real["keyPath"]``, never holds that one."""
    from ccg.assemble import rows
    return key_path if rows.MASK in key_path else rows._mask_key_path(key_path)     # noqa: SLF001 - the one keyPath masking the loader writes


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
        for k in ("qname", "lineStart", "lineEnd"):
            if subj.get(k) != real.get(k):
                acc.defects[f"subject-{k}"] += 1
        try:
            same = region_pairs(subj.get("ownedRegions")) == region_pairs(real.get("ownedRegions"))     # the answer's flat copy against the stored one
        except HarnessError:                                                                          # a malformed copy is an engine defect
            same = False
        if not same:
            acc.defects["subject-ownedRegions"] += 1
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
        got, want = stored_keypath(real["keyPath"]), stored_keypath(e["keyPath"])      # N2: both in the stored form, the producer's own
        if got != real["keyPath"] or want != e["keyPath"] or "[#" in got:
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
    best, hit_chain, hit_x, err = "none", None, None, None
    for x in candidates(ans):
        if not _h1(x, rec, snap, acc):
            continue
        if x.get("status") not in STATUSES:                                  # H2
            best = max(best, "H1", key=stage_rank.get)
            continue
        chain = x.get("chain") or []
        subj_id = _subject_id(x.get("subject") or {})
        if chain and not (isinstance(chain[0], dict) and isinstance(chain[-1], dict)):
            acc.harness_errors["hop-shape"] += 1
            err = err or "hop-shape"
            continue
        if not chain or chain[0].get("id") != ask_id or chain[-1].get("id") != subj_id:      # H3, shape
            best = max(best, "H2", key=stage_rank.get)
            continue
        verdict, code = replay_chain(chain, snap)                           # H3, replay (HA-1)
        if verdict == "error":                                              # EP §5.2.1 (g)/(h): neither a hit nor a miss, the verdict is blocked
            acc.harness_errors[code] += 1
            err = err or code
            continue
        if verdict == "fail":
            best = max(best, "H2", key=stage_rank.get)
            continue
        if hit_chain is None or len(chain) < len(hit_chain):
            hit_chain, hit_x = chain, x
    if hit_chain is not None:
        return {"hit": True, "stage": "H3", "chain": hit_chain, "distance": hit_x.get("distance")}
    if err is not None:
        return {"hit": False, "error": err}
    e = rec["impacted"]
    absent = e.get("type") == "symbol" and _mk.source_check_needed({"qname": e["qname"]}, rec["lines"], snap.symbols_of_path(e["path"]))
    return {"hit": False, "stage": best, "extractionAbsent": bool(absent),
            "depthBoundCut": bool((ans.get("depthBound") or {}).get("unexplored"))}


# =============================================================================================
# asking and scoring a gate run
# =============================================================================================

def unit_probe(rec: Mapping, records: Sequence[Mapping]) -> str:
    """The record-local probe rule (of the record's ``serves[]`` the point with the fewest items in the list, ties by smallest ``cpId``). It is NOT
    what a unit anchor is asked from: that is the S47 probe map (EP §5.1; :func:`ask_id_of`); this is only its cross-check."""
    served = sorted({_mk.contract_id(loc) for loc in rec.get("serves") or []})
    if not served:
        raise HarnessError("unit anchor without a served contract point")
    n = Counter(r["anchorId"] for r in records)
    return min(served, key=lambda c: (n.get(c, 0), c))


def read_probe_map(path: Path, home: Path | None = None) -> dict:
    """The S47 probe map ``{unit anchorId: probe cpId}`` (``split_manifest.split(...)["probe"]``, written custodian-side), read only from the
    custodian area. Never printed."""
    area = _ac.custodian_area(home).resolve()
    p = Path(path).expanduser().resolve()
    if p != area and area not in p.parents:
        raise Refusal("the probe map must be under the custodian area, never elsewhere")
    m = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(m, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in m.items()):
        raise HarnessError("the probe map is not {unit anchorId: cpId}")
    return m


def ask_id_of(rec: Mapping, probes: Mapping | None = None) -> str:
    """The id a record's anchor is asked by: its own ``anchorId``, or for a ``unit`` anchor its S47 probe. A unit item with no entry in ``probes`` is a
    harness error (fail closed: the probe is never re-derived from the list passed in, which is not the population S47 used)."""
    if rec["anchor"].get("type") != "unit":
        return rec["anchorId"]
    pid = (probes or {}).get(rec["anchorId"])
    if not pid:
        raise HarnessError("unit anchor without an S47 probe entry")
    return pid


def locator_index(records: Sequence[Mapping]) -> dict:
    """``cpId -> contract locator`` over every contract anchor and every ``serves[]`` entry of the records (what a probe request is built from)."""
    out = {}
    for r in records:
        for loc in [r["anchor"]] + list(r.get("serves") or []):
            if loc.get("type") == "contract":
                out[_mk.contract_id(loc)] = loc
    return out


def build_request(rec: Mapping, ask_id: str, locators: Mapping | None = None) -> dict:
    """The structured impact request of EP §5.1 (never plain language, no ``scope``). A unit anchor is asked from its probe contract point, whose
    locator comes from ``locators`` (:func:`locator_index`) or the record's ``serves[]``."""
    a = rec["anchor"]
    if a["type"] == "symbol":
        anchor = {"type": "symbol", "symbolId": ask_id, "qname": a["qname"], "repoId": a["repoId"], "path": a["path"]}
    elif a["type"] == "config":
        anchor = {"type": "config", "entryId": ask_id, "repoId": a["repoId"], "path": a["path"], "keyPath": a["keyPath"]}
    else:
        loc = a if a["type"] == "contract" else (locators or {}).get(ask_id) or next((x for x in rec.get("serves") or [] if _mk.contract_id(x) == ask_id), None)
        if loc is None:
            raise HarnessError("the probe contract point has no locator in any record")
        anchor = {"type": "contract", "cpId": ask_id, "kind": loc["kind"], "keyNorm": loc["keyNorm"]}
    return {"useCase": "impact", "direction": "impact", "anchors": [anchor]}


def _asks(records: Sequence[Mapping], probes: Mapping | None = None) -> dict:
    out = {}
    for r in sorted(records, key=lambda r: r["itemId"]):
        out.setdefault(ask_id_of(r, probes), r)
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


def run_asks(records: Sequence[Mapping], snap, ask: Callable[[dict], Mapping], cache_len: Callable[[], int], probes: Mapping | None = None) -> dict:
    """One question per distinct asked anchor, after the cache assertion; ``snapshotId`` is read from the ``:Snapshot`` node BEFORE the first
    question. A contract anchor whose ``cpId`` is absent from the gate graph is not asked (an ``anchor-not-in-graph`` miss). ``probes`` is the S47
    probe map: a unit item without an entry is a :class:`HarnessError` before anything is asked. The result records the probes used and how many
    differ from the record-local cross-check :func:`unit_probe`."""
    asks = _asks(records, probes)                                            # fail closed on a missing probe before the cache check and any question
    locators = locator_index(records)
    for aid, rec in asks.items():
        if rec["anchor"].get("type") == "unit":
            build_request(rec, aid, locators)                                 # a probe point no record can locate fails here, before the first question
    check_cache_empty(cache_len)
    pinned = snap.snapshot_id()
    answers = {}
    for aid, rec in asks.items():
        if rec["anchor"].get("type") in ("contract", "unit") and not snap.node_exists("ContractPoint", aid):
            continue
        req = build_request(rec, aid, locators)
        got = ask(req)
        answers[aid] = dict(got) if isinstance(got, AnchorFault) else {"request": req, "answer": got}
    units = {r["anchorId"]: r for r in records if r["anchor"].get("type") == "unit"}
    used = {u: ask_id_of(r, probes) for u, r in units.items()}
    differs = sum(1 for u, r in units.items() if r.get("serves") and unit_probe(r, records) != used.get(u))
    return {"snapshotId": pinned, "answers": answers, **({"probes": {u: used[u] for u in units}} if units else {}), "probeCrossCheckDiffers": differs}


_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


ENGINE_ERROR = "engine-error"
REPEAT_STALL = "repeat-stall"           # ``via`` of an engine-error entry converted from k consecutive timeouts at one position (-p1-gate-marker-ml.md §1.1.3)


class AnchorFault(dict):
    """What an ``ask`` returns (instead of an answer document) for an anchor whose ask failed and was RECORDED as such (``gate_run``'s policy):
    ``run_asks`` stores it as the anchor's entry in place of ``{"request", "answer"}``, and ``score_gate`` reads it."""


def _error_class(value: Any) -> str:
    """The class name of a recorded harness error, or ``unclassified``: only an identifier ever reaches the outcome or the report."""
    return value if isinstance(value, str) and _CLASS_NAME.fullmatch(value) else "unclassified"


def score_gate(records: Sequence[Mapping], run: Mapping, snap, patterns: Sequence[str]) -> dict:
    """The hit rule over every key item (``run`` is :func:`run_asks`' result). -> per-item outcomes plus what the run lists.

    Outcomes: ``hit``, ``miss`` (with ``reason``), ``out-of-scope`` (triaged ``key-error``; never a hit, a miss or a wrong-possible item),
    ``error`` (a harness error: neither a hit nor a miss, EP §5.2.1 (g)/(h)), ``void`` (a C5 stamp failure: the answer is void and the run is
    reported). Held-out ids never leave this structure but the custodian file.

    **The per-anchor harness-error entry** (the shape ``gate_run`` follows once it records a post-marker ``HarnessError`` for ONE anchor instead of
    ending the whole run): ``run["answers"][<asked anchor id>] = {"harnessError": "<exception class name>"}`` in place of
    ``{"request": ..., "answer": ...}``. Every item asked by that anchor is ``error`` (reason = the class name), counted per item in
    ``harnessErrors[<class name>]``, and the verdict is ``blocked`` (:func:`decide_gate`). Only the class name is carried: a value that is not an
    identifier of at most 64 characters is reported as ``unclassified`` (a message, an id or a path never reaches the report). An entry that also
    carries an ``answer`` is still an error (fail closed). A deterministic failure of the whole run still ends without a ``run.json``;
    ``gate_run`` records it.

    **The per-anchor engine-error entry** (``-p1-gate-marker-ml.md`` §3): ``{"outcome": "engine-error", "exceptionClass": "<class name>"}`` (no request,
    no answer). Every item of that anchor is a ``miss`` with reason ``engine-error`` (the ``fail`` tier, in the recall denominator), counted per class in
    ``engineErrors``; an entry that also carries a request or an answer is a harness error (fail closed)."""
    acc, items, stops = Acc(), {}, Counter()
    pinned, answers = run["snapshotId"], run["answers"]
    anchor_state = {}
    engine_errors: Counter = Counter()
    for aid, entry in answers.items():
        if "harnessError" in entry:
            anchor_state[aid] = {"harnessError": _error_class(entry["harnessError"])}
            continue
        if entry.get("outcome") == ENGINE_ERROR:
            if "answer" in entry or "request" in entry:                    # a fault entry that also carries an answer is not trusted (fail closed)
                anchor_state[aid] = {"harnessError": "unclassified"}
            else:
                anchor_state[aid] = {"engineError": _error_class(entry.get("exceptionClass")),
                                     **({"via": REPEAT_STALL, "stalls": entry["stalls"]} if entry.get("via") == REPEAT_STALL and type(entry.get("stalls")) is int else {})}
            continue
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
        aid = ask_id_of(rec, run.get("probes"))
        if aid not in answers:
            items[iid] = {**base, "outcome": "miss", "reason": "anchor-not-in-graph"}
            continue
        st = anchor_state[aid]
        if "engineError" in st:                                            # a miss in the `fail` tier, in the recall denominator (-p1-gate-marker-ml.md §3)
            engine_errors[st["engineError"]] += 1
            items[iid] = {**base, "outcome": "miss", "reason": ENGINE_ERROR, "exceptionClass": st["engineError"], "stage": "none",
                          **({"via": REPEAT_STALL} if "via" in st else {}),
                          "extractionAbsent": False, "depthBoundCut": False}
        elif "harnessError" in st:
            acc.harness_errors[st["harnessError"]] += 1
            items[iid] = {**base, "outcome": "error", "reason": st["harnessError"]}
        elif st["stamp"]:
            items[iid] = {**base, "outcome": "void", "reason": "stamp"}
        elif st["echo"]["state"] != "ok":
            items[iid] = {**base, "outcome": "miss", "reason": "not-resolved" if st["echo"]["state"] == "not-resolved" else "request-unequal"}
        else:
            r = evaluate_item(rec, aid, answers[aid]["answer"], snap, acc)
            if r.get("error"):
                items[iid] = {**base, "outcome": "error", "reason": r["error"]}
            else:
                items[iid] = {**base, "outcome": "hit" if r["hit"] else "miss",
                              **({"hops": len(r["chain"]), "interFunctionHops": sum(1 for h in r["chain"][1:] if h["kind"] == "Symbol"),
                                  "distance": r["distance"]} if r["hit"]
                                 else {"reason": "no-qualifying-item", "stage": r["stage"], "extractionAbsent": r["extractionAbsent"],
                                       "depthBoundCut": r["depthBoundCut"]})}
    return {"items": items, "defects": dict(acc.defects), "harnessErrors": dict(acc.harness_errors), "maskedComparisons": acc.masked,
            "stops": dict(stops), "engineErrors": dict(engine_errors)}


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


def gated_strata(flow_flag: bool) -> tuple[str, ...]:
    """The strata a run asks and a score counts: :data:`STRATA_GATED`, plus ``flow`` only when the flow flag is true. ``wiring`` is withheld whatever the flag."""
    return (*STRATA_GATED, "flow") if flow_flag else STRATA_GATED


def withhold_strata(records: Sequence[Mapping], gated: Sequence[str] = STRATA_GATED) -> tuple[list, dict]:
    """The records of the gated strata, and the COUNT withheld per stratum (never ids): a withheld item is not asked, scored or resolved
    (``-p1-gate-marker-ml.md`` §4.2). Used by ``score`` here and by the run verb."""
    kept = [r for r in records if r["derivedStratum"] in gated]
    return kept, dict(Counter(r["derivedStratum"] for r in records if r["derivedStratum"] not in gated))


def flow_flag(record_path: Path, committed_text: Callable[[Path], str] | None = None) -> bool:
    """The flow flag from the committed ``ccg-p1-flow-python-v2-decision`` record (never from the command line; ``-p1-gate-marker-ml.md`` §4.1).
    True only if the file is present, parses, its ``decision.verdict`` starts with ``TRUE`` AND HEAD's committed copy holds the same text;
    everything else (no record, unparsable, edited after the commit, any other verdict) is false."""
    path = Path(record_path)
    try:
        text = path.read_text(encoding="utf-8")
        if (committed_text or _ac._git_head_text)(path) != text:
            return False
        verdict = (json.loads(text).get("decision") or {}).get("verdict")
    except (OSError, ValueError, AttributeError, subprocess.CalledProcessError, Refusal, _ac.Refusal):      # fail closed: never a traceback
        return False
    return isinstance(verdict, str) and verdict.startswith("TRUE")


def decide_gate(result: Mapping, flags: Mapping[str, str] | None = None, moved_out: Mapping[str, int] | None = None,
                gated_strata: Sequence[str] = STRATA_GATED) -> dict:
    """EP §12.1. With no ``flags`` this is gate 1: pass iff every scored item is a hit over >= 75 counting items and >= 8 items per GATED stratum
    (``gated_strata``; default :data:`STRATA_GATED`: a withheld stratum is neither scored nor counted toward ``minimumsMet``). With ``flags`` (itemId -> kept /
    kept-cluster-fresh / fresh-cluster) it is a re-gate: a pass needs (1) every item a hit, (2) the composition rule (H >= 75, every H_s >= 8, and per
    stratum that had a miss at least as many fresh-cluster items as were ``moved_out``) and (3) at least F = 35 COUNTING items on fresh clusters; (1) and
    (2) without (3) is "all hits, floor not met", never a pass.

    Precedence (``-p1-gate-marker-ml.md`` §3): ``stopped`` (a C5 stamp failure or void) > ``blocked`` by a HARNESS ERROR (an item neither hit nor miss) >
    ``fail`` (a miss, reported as run) > ``blocked`` by an engine defect or a listed masked comparison (these block a pass only) > the re-gate
    outcomes > ``pass``. Out-of-scope and errored items are neither hits nor misses. A scored item in a stratum outside ``gated_strata`` is a
    :class:`Refusal` (a withheld stratum must not reach the verdict)."""
    items = _scored(result)
    strays = {o["stratum"] for o in result["items"].values() if o["outcome"] in ("hit", "miss", "error") and o["stratum"] not in gated_strata}
    if strays:
        raise Refusal(f"{sum(1 for o in result['items'].values() if o['stratum'] in strays)} scored items lie in a stratum that is not gated")
    misses = sorted(i for i, o in items.items() if o["outcome"] == "miss")
    voided = sorted(i for i, o in result["items"].items() if o["outcome"] == "void")
    errored = [i for i, o in result["items"].items() if o["outcome"] == "error"]
    blockers = []
    if result.get("harnessErrors") or errored:
        blockers.append("harness-errors")
    if result.get("defects"):
        blockers.append("engine-defects")
    if result.get("maskedComparisons"):
        blockers.append("masked-comparisons")
    cnt = _counts(result["items"])
    strata_ok = all(cnt["byStratum"].get(s, 0) >= MIN_STRATUM for s in gated_strata)
    kind = "gate" if flags is None else "re-gate"
    base = {"misses": len(misses), "voided": len(voided), "errored": len(errored), "blockers": blockers, "H": cnt["counting"], "scored": cnt["scored"],
            "minimumsMet": cnt["counting"] >= MIN_COUNTING and strata_ok, "byStratum": cnt["byStratum"], "gatedStrata": list(gated_strata),
            "missClusters": sorted({result["items"][i]["anchorId"] for i in misses}), "movedOutByStratum": dict(Counter(items[i]["stratum"] for i in misses))}
    if result.get("stops") or voided:
        return {**base, "verdict": "stopped", "kind": kind}
    if "harness-errors" in blockers:
        return {**base, "verdict": "blocked", "kind": kind}
    if misses:
        return {**base, "verdict": "fail", "kind": kind, "rule": "the first run is the result (EP §12.1 step 1)"}
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


def list_key(records: Iterable[Mapping]) -> str:
    """The canonical identity of a held-out list (``-p1-gate-marker-ml.md`` §1.1): the full 64-hex sha256 over the canonical JSON of each distinct
    record, sorted. Byte order, blank lines, key order and duplicate records do not change it."""
    canon = sorted({_mk.canon(r) for r in records})
    return hashlib.sha256("\n".join(canon).encode("utf-8")).hexdigest()


def list_key_from_text(text: str) -> str:
    return list_key(json.loads(l) for l in text.splitlines() if l.strip())


def first_run_marker(custodian: Path, list_sha256: str, payload: Mapping) -> Path:
    """EP §12.1 step 7: the gate is never re-run on the list that was run. ``list_sha256`` is :func:`list_key` (full 64 hex; a prefix is a truncated
    key and is refused). The marker is created atomically; an existing one refuses the run."""
    if not re.fullmatch(r"[0-9a-f]{64}", list_sha256):
        raise Refusal("the list key is not a full 64-hex sha256")
    marker = Path(custodian) / f"gate-run-{list_sha256}.marker"
    marker.parent.mkdir(parents=True, exist_ok=True)
    # the payload is written and fsynced to a temp file FIRST, then ``os.link``ed onto the name (atomic; refuses an existing name): a process killed at
    # any point leaves no marker or a complete one, never an empty or torn one (a marker no human may remove would be a permanent deadlock)
    fd, tmp = tempfile.mkstemp(dir=marker.parent, prefix=".gate-run-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_ac.canonical_json(payload) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, marker)
        except FileExistsError:
            raise Refusal("this list was already run: a re-run on the same list is refused (EP §12.1 step 7); a new list is required") from None
    finally:
        os.unlink(tmp)                                           # the temp name only; a marker is never removed
    dfd = os.open(marker.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return marker


def read_marker(custodian: Path, list_sha256: str) -> dict | None:
    """The once-only marker of a list as a dict, or ``None`` when there is none, it cannot be read, or it is not a JSON object with a string ``state``
    (``gate_run`` writes it; this only reads it, so ``score`` never trusts a run file without one)."""
    try:
        doc = json.loads((Path(custodian) / f"gate-run-{list_sha256}.marker").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and isinstance(doc.get("state"), str) else None


def marker_state(custodian: Path, list_sha256: str) -> str | None:
    """The state of the once-only marker of a list (``started`` | ``asked`` | ``failed:<class>``), or ``None`` (see :func:`read_marker`)."""
    doc = read_marker(custodian, list_sha256)
    return doc["state"] if doc else None


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


def _chain_length(rows: Sequence[Mapping]) -> dict:
    """The engine's ``distance`` over hit items; the Symbol-hop count (``interFunctionHops``) is reported beside it (a ConfigEntry hit has 0 of those)."""
    return {**_dist([o["distance"] for o in rows if o.get("distance") is not None]), "symbolHops": _dist([o["interFunctionHops"] for o in rows])}


def build_report(result: Mapping, decision: Mapping, *, flags: Mapping[str, str] | None = None, cluster_of: Mapping | None = None,
                 exposure: Sequence[Mapping] = (), exposure_strata: Mapping | None = None, context: Mapping | None = None,
                 gated_strata: Sequence[str] = STRATA_GATED, withheld: Mapping[str, int] | None = None, flow_flag: bool | None = None,
                 withheld_shapes: Mapping | None = None) -> dict:
    """Counts, hashes and classes only (never item content, never an id). Recall is given over items, distinct anchors and clusters, per
    stratum and pooled, each with its Wilson interval, and the cluster interval of P0-ML §7.3 (anchors as clusters when no map is given).
    ``context`` carries the preflight blocks (first contact, pins, harness pin, scope) verbatim. Every stratum
    outside ``gated_strata`` is reported by COUNT only, as withheld, with its reason (``withheld``); a withheld stratum has no recall, interval or floor, and when
    ``flow`` is withheld the report also says "flow-unavailable, not closed" with the flag and the count."""
    items = _scored(result)
    rows = list(items.items())
    per = {s: _rec_block([o for _, o in rows if o["stratum"] == s], cluster_of) for s in gated_strata if any(o["stratum"] == s for _, o in rows)}
    miss = [o for _, o in rows if o["outcome"] == "miss"]
    oos = [o for o in result["items"].values() if o["outcome"] == "out-of-scope"]
    errored = sum(1 for o in result["items"].values() if o["outcome"] == "error")
    eng_all = [o for o in miss if o.get("reason") == ENGINE_ERROR]               # its own line, never folded into the harness errors
    eng = [o for o in eng_all if o.get("via") != REPEAT_STALL]                   # raised exceptions; the repeat-stall conversions are a sub-line of their own
    stalled = [o for o in eng_all if o.get("via") == REPEAT_STALL]
    rep = {"verdict": decision["verdict"], "kind": decision["kind"], "H": decision["H"], "H_all": len(rows), "scored": len(rows),
           "minimumsMet": decision["minimumsMet"], "blockers": decision["blockers"],
           "recall": {"pooled": _rec_block([o for _, o in rows], cluster_of), "byStratum": per},
           "misses": {"count": len(miss), "byStratum": dict(Counter(o["stratum"] for o in miss)), "byReason": dict(Counter(o["reason"] for o in miss)), "byStage": dict(Counter(o.get("stage", "-") for o in miss)),
                      "anchorNotInGraph": sum(1 for o in miss if o["reason"] == "anchor-not-in-graph"),
                      "extractionAbsent": sum(1 for o in miss if o.get("extractionAbsent")), "depthBoundCut": sum(1 for o in miss if o.get("depthBoundCut")),
                      "triage": {c: 0 for c in TRIAGE}, "triageNote": "to be filled by the gate runner and the method owner (EP §12.1 step 2)"},
           "outOfScope": {"count": len(oos), "triage": {"key-error": len(oos)}, "outcome": "proposed, unconfirmed"},   # EP §12.1 step 2: the stakeholder confirms
           "erroredItems": errored,
           "engineErrors": {"count": len(eng), "byStratum": dict(Counter(o["stratum"] for o in eng)), "byClass": dict(Counter(o["exceptionClass"] for o in eng)),
                              "repeatStall": {"count": len(stalled), "byStratum": dict(Counter(o["stratum"] for o in stalled))}},
           "voided": decision["voided"], "stops": dict(result.get("stops", {})), "engineDefects": dict(result.get("defects", {})),
           "harnessErrors": dict(result.get("harnessErrors", {})), "maskedComparisons": result.get("maskedComparisons", 0),
           "chainLength": {s: _chain_length([o for _, o in rows if o["stratum"] == s and o["outcome"] == "hit"]) for s in per},
           "chainLengthUnit": "the engine's `distance` over hit items (the shortest qualifying chain); `symbolHops` = Symbol hops after the first, beside it",
           "units": "items, distinct anchors, clusters (`clusterUnit` says which map was given)"}
    rep["withheld"] = {s: {"count": int((withheld or {}).get(s, 0)), "reason": WITHHELD_REASONS[s]} for s in STRATA if s not in gated_strata}      # counts and reasons only
    if withheld_shapes is not None:                                           # the Q8 shapes, counts only (``withheld_counts``): per family and total, beside the hits
        rep["withheldShapes"] = dict(withheld_shapes)
    if "flow" not in gated_strata:
        rep["flow"] = {"status": "flow-unavailable, not closed", "flag": bool(flow_flag), "withheld": int((withheld or {}).get("flow", 0))}
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
    cum_rows = [((exposure_strata or {}).get(i) or r.get("stratum"), bool(r["hit"])) for i, r in prior.items()]
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


def miss_breakdown(records: Sequence[Mapping], result: Mapping) -> list[dict]:
    """The fine per-kind breakdown ``{stratum, anchorType, impactedType, items, hits, misses}`` over the scored items (CO-5). Held-out-derived: the caller writes
    it to the custodian ``work/`` directory only (:func:`write_custodian`) and never prints it or puts it in the report; types and counts, no id, path or name."""
    scored, cells = _scored(result), {}
    for rec in records:
        o = scored.get(rec["itemId"])
        if o is None:
            continue
        c = cells.setdefault((o["stratum"], rec["anchor"]["type"], rec["impacted"]["type"]), {"items": 0, "hits": 0, "misses": 0})
        c["items"] += 1
        c["hits" if o["outcome"] == "hit" else "misses"] += 1
    return [{"stratum": k[0], "anchorType": k[1], "impactedType": k[2], **v} for k, v in sorted(cells.items())]


def exposure_rows(result: Mapping, flags: Mapping | None = None) -> list[dict]:
    """``{itemId, hit, stratum}`` for every scored item, for the ``exposure`` verb (custodian file only); the stratum is what P1-43's per-stratum
    cumulative record reads."""
    return [{"itemId": i, "hit": o["outcome"] == "hit", "stratum": o["stratum"]} for i, o in sorted(_scored(result).items())]


def write_custodian(root: Path, name: str, rows: Sequence[Mapping] | Mapping, home: Path | None = None) -> Path:
    """Held-out-derived output: refused unless ``root`` is the custodian area ``~/.local/share/ccg-eval/<...>`` (``-p1-ml.md`` §6.4)."""
    root = Path(root).resolve()
    area = _ac.custodian_area(home).resolve()
    if root != area and area not in root.parents:
        raise Refusal("held-out-derived files go under the custodian root, never elsewhere")
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


def _git_out(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


def run_context(repo: Path, pf: Mapping, *, allow_after_adaptation: bool = False) -> dict:
    """M5 (EP §12.3: the record precedes the first build). Refuses unless the record's commit was found, is an ancestor of HEAD, and no file under ``cpg/``
    outside ``cpg/eval/`` and ``*.md`` changed since it; refuses an ``after-adaptation`` label unless ``allow_after_adaptation`` (the label is then
    recorded, since a first run's label cannot be repaired afterwards). -> the preflight block ``gate_run`` stores in ``run.json`` and ``score`` prints:
    label, record commit, HEAD, first-contact counts, the harness pin with each pinned file's sha256 (counts, hashes and names only)."""
    repo = Path(repo)
    rc = pf.get("recordCommit")
    if not rc:
        raise Refusal("the record's commit was not found in git log: the record must be committed before the run")
    if subprocess.run(["git", "merge-base", "--is-ancestor", rc, "HEAD"], cwd=repo, capture_output=True).returncode != 0:
        raise Refusal("the record's commit is not an ancestor of HEAD")
    touched = [p for p in _git_out(repo, "diff", "--name-only", rc, "HEAD", "--", "cpg").split()
               if not (p.startswith("cpg/eval/") or p.endswith(".md"))]
    if touched:
        raise Refusal(f"{len(touched)} file{'s' if len(touched) != 1 else ''} under cpg/ outside cpg/eval/ changed after the record's commit")
    if pf["label"] != "first-contact" and not allow_after_adaptation:
        raise Refusal("the preflight label is after-adaptation: the run is refused unless it is explicitly allowed and labelled")
    fc, hp = pf["firstContact"], pf["harnessPin"]
    sha = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in HARNESS_FILES if (repo / rel).is_file()}
    return {"label": pf["label"], "afterAdaptationAllowed": bool(allow_after_adaptation and pf["label"] != "first-contact"),
            "recordCommit": rc, "head": _git_out(repo, "rev-parse", "HEAD").strip(),
            "firstContact": {"firstContact": fc["firstContact"], "fieldsDiffering": list(fc["fieldsDiffering"]), "dirtyPaths": fc["dirtyPaths"]},
            "harnessPin": {"afterAdaptation": hp["afterAdaptation"], "differ": list(hp["differ"]), "sha256": sha},
            "pins": {"differences": len(pf["pins"]["differences"]), "protocolStringConsistent": pf["pins"]["protocolStringConsistent"]},
            "scope": dict(pf["scope"])}


def require_run_context(run: Mapping) -> dict:
    """``score`` needs the preflight block the run recorded; a run file without one is refused."""
    ctx = run.get("preflight")
    if not isinstance(ctx, dict) or ctx.get("label") not in ("first-contact", "after-adaptation") or not ctx.get("recordCommit"):
        raise Refusal("the run file carries no preflight block: the run did not record its label")
    return ctx


def gate_kind(record: Mapping) -> tuple[str, dict | None]:
    """The kind of gate and, for a re-gate, the moved-out counts, from the COMMITTED prereg record (``kind``: ``gate`` | ``re-gate``; a re-gate carries
    ``movedOut`` ``{stratum: n}``), never from the command line. A record that does not say is refused, not read as gate 1."""
    kind = record.get("kind")
    if kind == "gate" and "movedOut" not in record:
        return "gate", None
    if kind == "re-gate":
        mo = record.get("movedOut")
        if isinstance(mo, dict) and all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in mo.items()):
            return "re-gate", dict(mo)
        raise Refusal("the record is a re-gate with no usable movedOut counts")
    raise Refusal("the committed record does not state a valid gate kind (gate | re-gate)")


class PrecheckRefusal(Refusal):
    """A refusal of :func:`score_inputs_check`: raised by ``run`` before the marker (nothing spent) and by ``score``."""


# --- the dev-gold probe: an attestation bound to the run it vouches for (P6-1) ---------------------------------------------------------------------
#
# Design choice (CO-5 round 2): the probe is the PRODUCT of a dev rehearsal run (:func:`write_dev_probe`, the ``dev-probe`` verb), not the gate asking the dev gold
# itself before its marker. Asking dev at ``run`` would need a built store and an engine ask before the marker, so a dev miss would strand a ~30 minute build
# (the list stays unspent but the store is lost) where a bad probe now refuses at exit 5 before the lock, the build and the marker, and it would turn the CO-9
# rehearsal into a gate-internal step instead of the separately reported run it is meant to be. The attestation is made forgery-resistant instead: the gate
# computes the dev items from the committed gold, compares the record commit, the harness sha map, the gold hashes and (after its own build) the CONTENT snapshot id the
# file carries (:func:`content_snapshot_id`: the stamped snapshot id minus the per-build HMAC-keyed bundle fields, which no two builds share), checks every key-error id against the committed stakeholder record, and pins the file's sha256 in the marker. A list made only of dev items is a
# dev rehearsal and needs no probe (it cannot spend a held-out list); every other list does.

DEV_KEY_ERRORS = Path("cpg") / "eval" / "decisions" / "ccg-dev-key-errors-v1.json"
DEV_KEY_ERRORS_SCHEMA = "ccg-dev-key-errors-v1"
DEV_GOLD = "cpg/eval/dev/ac5-{}.jsonl"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
KEY_ERROR_CONFIRMER = "user"               # only the stakeholder confirms a dev key-error (``ccg-dev-key-errors-v1``): never an agent


WITHHELD_CAP = 0.25                      # the scoped AC-5 wording: over a quarter of the counting scripts items withheld is over the cap (20 or more of 79)


def eval_commit_reader(repo: Path) -> Callable[[str], str | None]:
    """``path -> text`` of ``path`` in the tree at ``cpg/eval/EVAL_COMMIT`` (the tree the gate builds; ``git show``, read-only), ``None`` for a path that is not
    there or when the repository has no usable ``EVAL_COMMIT``: the predicate then withholds nothing, the conservative side."""
    try:
        rev = (Path(repo) / "cpg" / "eval" / "EVAL_COMMIT").read_text(encoding="utf-8").strip()
    except OSError:
        return lambda _rel: None
    return lambda rel: _git_show_text(Path(repo), rev, rel)


def item_withheld_shape(rec: Mapping, file_text: Callable[[str], str | None]) -> str | None:
    """The Q8 shape letter of a SCRIPTS item (the predicate on the impacted file's text at the build tree and the gold lines), else ``None``."""
    imp = rec.get("impacted") if isinstance(rec.get("impacted"), Mapping) else {}
    path = imp.get("path")
    lang = _ws.language_tag(path) if isinstance(path, str) else None
    text = file_text(path) if lang else None
    if rec.get("derivedStratum") != "scripts" or text is None or "\r" in text:           # a bare CR: the line numbering is ambiguous, so the item is asked (fail closed)
        return None
    return _ws.withheld_shape({"path": path, "lines": rec.get("lines") or []}, text, lang)


def withheld_counts(records: Iterable[Mapping], file_text: Callable[[str], str | None]) -> dict:
    """``{"scriptsCounting": n, "withheld": n, "byShape": {letter: n}}``: counts only, over the counting scripts items."""
    rows = [r for r in records if r.get("derivedStratum") == "scripts" and r.get("counting")]
    shapes = Counter(x for x in (item_withheld_shape(r, file_text) for r in rows) if x)
    return {"scriptsCounting": len(rows), "withheld": sum(shapes.values()), "byShape": dict(sorted(shapes.items()))}


def withhold_shapes(records: Sequence[Mapping], file_text: Callable[[str], str | None]) -> tuple[list, dict]:
    """``(kept, counts)``: the scripts items of a Q8 withheld shape, counting or not, are NOT asked or scored (the scoped AC-5 wording: "every counting item outside the
    named withheld shapes"; ruling on the prereg reviews' B1/M2). The decision is the predicate's alone, on the gold locator and the build tree's file text
    (:func:`eval_commit_reader`): nothing in the record but its stratum, ``counting`` flag, impacted path and gold lines is read. ``counts`` is
    :func:`withheld_counts` over ``records`` (the COUNTING items only: a record with no ``counting`` key is non-counting here and in the scorer; the same set the cap is
    checked on); floors, bounds and the verdict are then over ``kept``."""
    kept = [r for r in records if not item_withheld_shape(r, file_text)]
    return kept, withheld_counts(records, file_text)


def check_floors_after_exclusion(kept: Sequence[Mapping], counts: Mapping, gated: Sequence[str]) -> None:
    """N4: when the exclusion removed counting items, the KEPT list must still be able to meet the AC-5 floors (:data:`MIN_COUNTING` pooled counting items,
    :data:`MIN_STRATUM` per gated stratum), else the run could only end in "floor not met": :class:`PrecheckRefusal`, so ``run`` exits 5 before the lock, the
    build and the marker and the list is not spent. Counts only. A list the exclusion did not touch is not judged here (its floors are ``score``'s)."""
    if not counts["withheld"]:
        return
    pooled = sum(1 for r in kept if r.get("derivedStratum") in gated and r.get("counting"))
    low = sorted(s for s in gated if sum(1 for r in kept if r.get("derivedStratum") == s) < MIN_STRATUM)
    if pooled < MIN_COUNTING or low:
        raise PrecheckRefusal(f"after the Q8 exclusion ({counts['withheld']} withheld) the list holds {pooled} counting items (floor {MIN_COUNTING}) and "
                              f"{len(low)} gated strata under {MIN_STRATUM} items ({', '.join(low) or 'none'}): the floors cannot be met, the list is not spent")


def check_withheld_cap(counts: Mapping) -> None:
    """Over the cap (``withheld`` more than :data:`WITHHELD_CAP` of the counting scripts items, i.e. 20 or more of 79) the scoped wording does not hold:
    :class:`PrecheckRefusal`, so ``run`` exits 5 before the lock, the build and the marker and the list is not spent. Counts only."""
    if counts["withheld"] > WITHHELD_CAP * counts["scriptsCounting"]:
        raise PrecheckRefusal(f"the Q8 predicate withholds {counts['withheld']} of {counts['scriptsCounting']} counting scripts items: over the {int(WITHHELD_CAP * 100)}% cap, "
                              f"the scoped AC-5 wording does not hold and the list is not spent")


def read_dev_probe(path: Path) -> tuple[Any, str]:
    """``(document, sha256 of the exact bytes)`` of the rehearsal's probe file (dev data, builder-visible: any path). Unreadable or not JSON -> :class:`PrecheckRefusal`."""
    try:
        raw = Path(path).read_bytes()
        return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError):
        raise PrecheckRefusal("the dev-gold probe result is missing or unreadable: the rehearsal writes it, and the gate does not start without it") from None


def _dev_gold(repo: Path, stratum: str) -> tuple[list[dict], str]:
    f = Path(repo) / DEV_GOLD.format(stratum)
    try:
        raw = f.read_bytes()
        return [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()], hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError):
        raise PrecheckRefusal(f"the committed dev gold of {stratum} is unreadable") from None


def is_dev_rehearsal(repo: Path, records: Sequence[Mapping]) -> bool:
    """True iff the list is non-empty and EVERY record, canonicalised (``match_keys.canon``), equals a committed dev gold record: such a run cannot spend a
    held-out list, so it needs no probe (it is what produces one). By content, never by ``itemId``: a record that reuses a dev id with other content is not
    exempt, and neither is a list holding one honest copy beside such a twin."""
    dev = set()
    for s in STRATA:
        try:
            dev |= {_mk.canon(r) for r in _dev_gold(repo, s)[0]}
        except (PrecheckRefusal, KeyError, TypeError, ValueError):
            return False
    try:
        return bool(records) and all(_mk.canon(r) in dev for r in records)
    except (TypeError, ValueError):
        return False


def _git_show_text(repo: Path, rev: str, rel: str) -> str | None:
    """``git show <rev>:<rel>`` of ``repo`` (read-only), or ``None`` when it fails (no such commit or path). Decoded from the bytes with ``\\r\\n`` read as one
    newline and a BARE ``\\r`` kept as it is (universal-newline decoding would turn it into a line break and shift every later line, N3)."""
    try:
        raw = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{rel}"], capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return raw.decode("utf-8", errors="replace").replace("\r\n", "\n")


def confirmed_key_errors(repo: Path, committed_text: Callable[[Path], str] | None = None, record_commit: str | None = None) -> set[str]:
    """The item ids the stakeholder confirmed as dev key-errors, from the COMMITTED record ``cpg/eval/decisions/ccg-dev-key-errors-v1.json``
    (``{"schema", "confirmed": [{"itemId", "confirmedBy": "user", "date": "YYYY-MM-DD"}]}``, ids distinct; the working copy must equal HEAD's). Missing,
    uncommitted, edited or malformed (an entry with no stakeholder, no real date or a repeated id included) -> :class:`PrecheckRefusal`. ``confirmedBy`` is a
    self-asserted string, not a signature: the control is the commit. With ``record_commit`` (the freeze commit the marker pins) the file at THAT commit must
    equal the working copy too; ``None`` is a refusal (the default never skips the check), so a commit made after the freeze that widens the record is refused (F4)."""
    path = Path(repo) / DEV_KEY_ERRORS
    try:
        text = path.read_text(encoding="utf-8")
        if (committed_text or _ac._git_head_text)(path) != text:
            raise ValueError
        if record_commit is None or _git_show_text(Path(repo), record_commit, DEV_KEY_ERRORS.as_posix()) != text:       # no record commit: fail closed
            raise ValueError
        doc = json.loads(text)
        ids = [c["itemId"] for c in doc["confirmed"]]
        if doc["schema"] != DEV_KEY_ERRORS_SCHEMA or not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError
        for c in doc["confirmed"]:
            if c.get("confirmedBy") != KEY_ERROR_CONFIRMER or not (isinstance(c.get("date"), str) and _ISO_DATE.fullmatch(c["date"])):
                raise ValueError
            datetime.date.fromisoformat(c["date"])
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError, Refusal, _ac.Refusal):
        raise PrecheckRefusal("the committed key-error confirmation record is missing, edited after its commit or malformed") from None
    return set(ids)


def _harness_sha(repo: Path) -> dict:
    return {rel: hashlib.sha256((Path(repo) / rel).read_bytes()).hexdigest() for rel in HARNESS_FILES if (Path(repo) / rel).is_file()}


def dev_probe_check(probe: Any, repo: Path, record_commit: str, gated: Sequence[str] = STRATA_GATED, committed_text: Callable[[Path], str] | None = None,
                    file_text: Callable[[str], str | None] | None = None) -> None:
    """The pre-gate probe of ``docs/plans/connected-code-graph-p1-closeout.md`` §3.5, as an attestation. ``probe`` is ``{"schema": DEV_PROBE_SCHEMA,
    "recordCommit", "snapshotId", "contentSnapshotId", "harnessSha256": {file: sha}, "devGoldSha256": {stratum: sha}, "byStratum": {stratum: {"hitIds": [...],
    "keyErrorIds": [...]}}}``. The gate takes NOTHING countable from it: per gated stratum the dev items are those of the committed gold file, and
    ``hitIds`` plus ``keyErrorIds`` must be exactly that set, disjoint and duplicate-free, with every key-error id in the committed confirmation record. The
    file must carry the freeze (record) commit, the current harness sha map (the import-closure files, equal as a set), the hash of each gated gold file and
    a snapshot id (informational) and a ``contentSnapshotId`` (:func:`content_snapshot_id`; compared with the gate's own build by :func:`check_probe_snapshot`). A stratum outside ``gated`` is not read.
    A dev item the Q8 predicate withholds (:func:`item_withheld_shape`, computed here from the committed gold and the build tree's file text, never taken from
    the probe) is EXCUSED from the hit requirement: every dev item must be a hit, a withheld shape or a committed key-error. Counts, strata and field names
    only in a refusal (dev data)."""
    if not isinstance(probe, Mapping) or probe.get("schema") != DEV_PROBE_SCHEMA or not isinstance(probe.get("byStratum"), Mapping):
        raise PrecheckRefusal(f"the dev-gold probe result is not a `{DEV_PROBE_SCHEMA}` document: the gate does not start without one")
    if probe.get("recordCommit") != record_commit:
        raise PrecheckRefusal("the dev-gold probe is for another record commit than this run's: the rehearsal is not at the freeze commit")
    if not isinstance(probe.get("harnessSha256"), Mapping) or dict(probe["harnessSha256"]) != _harness_sha(repo):
        raise PrecheckRefusal("the dev-gold probe's harness file hashes differ from the current harness files: the rehearsal ran other harness code")
    if not (isinstance(probe.get("snapshotId"), str) and probe["snapshotId"]):
        raise PrecheckRefusal("the dev-gold probe carries no snapshot id of the build it vouches for")
    if not (isinstance(probe.get("contentSnapshotId"), str) and probe["contentSnapshotId"]):
        raise PrecheckRefusal("the dev-gold probe carries no content snapshot id: it was written by an older harness; re-run the dev rehearsal on the current "
                              "harness and write a new probe with `gate.py dev-probe`")
    gold_sha = probe.get("devGoldSha256") if isinstance(probe.get("devGoldSha256"), Mapping) else {}
    confirmed = confirmed_key_errors(repo, committed_text, record_commit)       # the record must be usable whenever a probe is accepted, key-errors or not
    file_text = file_text or eval_commit_reader(repo)
    for s in gated:
        records, sha = _dev_gold(repo, s)
        if gold_sha.get(s) != sha:
            raise PrecheckRefusal(f"the dev-gold probe was not made against the committed dev gold of {s}: its hash differs")
        row = probe["byStratum"].get(s)
        if not isinstance(row, Mapping) or not all(isinstance(row.get(k), list) and all(isinstance(i, str) for i in row[k]) for k in ("hitIds", "keyErrorIds")):
            raise PrecheckRefusal(f"the dev-gold probe has no usable item lists for the gated stratum {s}")
        items = [r["itemId"] for r in records]
        hits, key_errors = row["hitIds"], row["keyErrorIds"]
        listed = hits + key_errors
        if len(set(listed)) != len(listed) or not set(listed) <= set(items):
            raise PrecheckRefusal(f"the dev-gold probe is inconsistent for {s}: duplicated ids, or ids that are not dev items")
        if key_errors:
            if not set(key_errors) <= confirmed:
                raise PrecheckRefusal(f"dev-gold probe: {s} lists {len(set(key_errors) - confirmed)} key-error id(s) the committed stakeholder record does not confirm")
        excused = {r["itemId"] for r in records if item_withheld_shape(r, file_text)}
        if len(set(listed) | excused) < len(set(items)):
            raise PrecheckRefusal(f"dev-gold probe: {s} has {len(hits)} hits of {len(set(items))} dev items ({len(key_errors)} stakeholder-confirmed key-errors, "
                                  f"{len(excused - set(listed))} withheld shapes): every dev item of a gated stratum must be a hit, a withheld shape or a key-error before the gate starts")


#: The HMAC-keyed fields of a bundle (``ccg/valuehash.py``, ``ccg/atrest.py`` ``bind_key_check``): the system key is random per build, so they
#: differ between two builds of one commit. The record fields ``valueHash`` (``configEntry``), ``defaultHash`` (``launchFact``) and the header's ``bindKeyCheck`` are left
#: out of the content id's bundle hashes, and so are the keyed members of ``bindParts`` (:data:`KEYED_BIND_PARTS`); nothing else is. No answer path reads them
#: (``tests/test_gate_content_snapshot_id.py`` guards ``ccg/engine``, ``ccg/front``, ``ccg/store``, ``clients/``), and the linker only compares keyed hashes with each other inside one build.
KEYED_BUNDLE_FIELDS = ("valueHash", "defaultHash", "bindKeyCheck")
#: The keyed members of a ``configEntry``'s ``bindParts`` (``ccg/valuehash.py`` ``BIND_HOST_TAG`` / ``BIND_PATH_TAG``): only these two are removed; the object, its
#: presence and the clear ``scheme``, ``hostKind``, ``host`` and ``port`` stay in the content id, and ``pathHmacs`` is replaced by its length (``pathSegments``,
#: which ``ccg/linker/bind.py`` reads) so the segment count is kept.
KEYED_BIND_PARTS = ("hostHmac", "pathHmacs")


def content_bundle_hash(records: Sequence[Mapping]) -> str:
    """``ccg.assemble.snapshot.bundle_hash`` of ``records`` with :data:`KEYED_BUNDLE_FIELDS` removed from every record (the header's §8.5 exempt fields are
    removed by that function as before)."""
    from ccg.assemble import snapshot as SN
    def strip(r: Mapping) -> dict:
        out = {k: v for k, v in r.items() if k not in KEYED_BUNDLE_FIELDS}
        if isinstance(out.get("bindParts"), Mapping):
            parts = out["bindParts"]
            out["bindParts"] = {**{k: v for k, v in parts.items() if k not in KEYED_BIND_PARTS},
                                "pathSegments": len(parts["pathHmacs"]) if isinstance(parts.get("pathHmacs"), (list, tuple)) else None}   # the count is clear; the binder reads it
        return out

    return SN.bundle_hash([strip(r) for r in records])


def content_snapshot_id(system: str, pack_set: str, code: str, units: Iterable[Sequence]) -> str:
    """The identity the dev-gold probe vouches for (``docs/reviews/connected-code-graph-p1-gate-snapshot-rca.md`` §5 (b)): the stamped snapshot id's own
    inputs (``ccg.assemble.snapshot.manifest_hash``: graph model version, ``packSetHash``, ``codeHash`` and every unit's id, status, flow flag and fail
    reason) with each unit's bundle hashed by :func:`content_bundle_hash`, so a different key does not change it and a different commit, pack set, code
    state or bundle content does. ``units`` are ``[unitId, buildStatus, records | None, flowAvailable, failReason | ""]``; ``None`` records (a unit with no
    bundle) hash to ``""`` as in the stamped id. The tag differs from the stamped id's, so the two are never confused. The stamped ``snapshotId`` is not touched."""
    from ccg import ids
    from ccg.assemble import snapshot as SN
    rows = [[uid, status, "" if records is None else content_bundle_hash(records), bool(flow), fail or ""] for uid, status, records, flow, fail in units]
    return ids.tuple_id(["contentSnapshot", system, SN.manifest_hash(pack_set, code, rows)])


def _stored_bundle(system: str, unit: str, build: str, root: Path) -> list[dict]:
    """One stored bundle as the refresh read it (``ccg.refresh._gate``: the at-rest custody read, then the validator)."""
    from ccg import atrest, bundle
    return bundle.loads(atrest.read_bundle(system, unit, build, root))


def store_content_id(snap, system: str, work_root: Path, read_bundle: Callable[..., list[dict]] | None = None) -> str:
    """The content id of the build ``snap`` reads: its ``:Snapshot`` and ``:Unit`` rows plus the stored bundles under ``work_root`` (the build's private
    work area). The stamped id is recomputed from the same bundles first and must equal the stored one, else :class:`HarnessError`: the bundles read are
    exactly the ones the store was built from. ``read_bundle(system, unit, build, root)`` is injected in tests."""
    from ccg.assemble import snapshot as SN
    read_bundle = read_bundle or _stored_bundle
    inputs = snap.snapshot_inputs()
    units, plain = [], []
    for uid, status, built, flow, fail in snap.unit_rows():
        try:
            records = None if status == "not-mapped" else read_bundle(system, uid, built, Path(work_root))
        except (OSError, ValueError):                                  # AtRestError is a PermissionError; BundleError a ValueError: neither carries a path or value out
            raise HarnessError("the build's stored bundle of a unit is missing or unreadable under its work root: the content id cannot be computed. " + STORE_REMEDY) from None
        units.append([uid, status, records, flow, fail])
        plain.append([uid, status, "" if records is None else SN.bundle_hash(records), bool(flow), fail or ""])
    stamped = SN.snapshot_id(system, SN.manifest_hash(inputs["packSetHash"], inputs["codeHash"], plain))
    if stamped != inputs["snapshotId"]:
        raise HarnessError("the stored bundles do not reproduce the stored snapshot id: they are not the bundles this graph was built from")
    return content_snapshot_id(system, inputs["packSetHash"], inputs["codeHash"], units)


def check_probe_snapshot(probe: Mapping, built_content_id: str) -> None:
    """The probe must vouch for what the gate built: its ``contentSnapshotId`` equals :func:`store_content_id` of the built graph. The stamped
    ``snapshotId`` is never compared (it embeds a per-build random key, so two builds of one commit differ). Raised after the build, so the caller's
    stranded-store remedy applies; the list is not spent."""
    vouched = probe.get("contentSnapshotId")
    if not (isinstance(vouched, str) and vouched):
        raise Refusal("the dev-gold probe carries no content snapshot id: it was written by an older harness. Re-run the dev rehearsal on the current "
                      "harness and write a new probe with `gate.py dev-probe`")
    if vouched != built_content_id:
        raise Refusal("the dev-gold probe vouches for another build content than this run built (its commit, pack set, code or unit bundles differ): "
                      "the rehearsal must be run at this EVAL_COMMIT with this harness, and its probe written with `gate.py dev-probe`")


def write_dev_probe(repo: Path, run: Mapping, snapshot, committed_text: Callable[[Path], str] | None = None) -> dict:
    """The attestation, as the product of a dev rehearsal: ``run`` is the rehearsal's ``run.json`` (its ``answers`` are the ``answers.jsonl`` entries), scored by the
    SAME :func:`score_gate` path against the committed dev gold of ``gated_strata(run's flow flag)`` on ``snapshot`` (the live store; its snapshot id must be the
    run's). ``hitIds`` are the items scored ``hit``; ``keyErrorIds`` are the other items the committed record confirms. Everything else it carries is read from the run
    (record commit and harness sha map from its preflight block, the content snapshot id the run recorded off its own build) or from the repository (gold hashes). It is a file-level attestation: ``run`` is trusted for the rehearsal's having happened (a run file composed by hand
    on a store holding that snapshot yields a probe), which is why the probe carries no run id: one would bind nothing."""
    repo = Path(repo)
    ctx, key = run.get("preflight"), run.get("listKey")
    if not isinstance(ctx, Mapping) or not ctx.get("recordCommit") or not isinstance((ctx.get("harnessPin") or {}).get("sha256"), Mapping) \
            or not (isinstance(key, str) and _HEX64.fullmatch(key)) or not isinstance(run.get("snapshotId"), str) \
            or not (isinstance(run.get("contentSnapshotId"), str) and run["contentSnapshotId"]):
        raise Refusal("the run file is not a rehearsal run: it carries no preflight block, list key, snapshot id or content snapshot id (a rehearsal run by an older harness)")
    if snapshot.snapshot_id() != run["snapshotId"]:
        raise Refusal("the live :Snapshot node differs from the run file's snapshot id: the probe is written on the store the rehearsal ran against")
    strata = gated_strata(bool(run.get("flowFlag")))
    patterns = scope_patterns(_load_yaml(repo / "cpg" / "eval" / "eval-manifest.yaml"))
    records, shas = [], {}
    for s in strata:
        recs, shas[s] = _dev_gold(repo, s)
        records += recs
    result = score_gate(records, run, snapshot, patterns)
    confirmed = confirmed_key_errors(repo, committed_text, ctx["recordCommit"])
    by = {}
    for s in strata:
        ids = sorted(r["itemId"] for r in records if r["derivedStratum"] == s)
        hit = [i for i in ids if result["items"][i]["outcome"] == "hit"]
        by[s] = {"hitIds": hit, "keyErrorIds": [i for i in ids if i not in hit and i in confirmed]}
    return {"schema": DEV_PROBE_SCHEMA, "recordCommit": ctx["recordCommit"], "snapshotId": run["snapshotId"],
            "contentSnapshotId": run["contentSnapshotId"], "harnessSha256": dict(ctx["harnessPin"]["sha256"]), "devGoldSha256": shas, "byStratum": by}


def score_inputs_check(repo: Path, gate_id: str, pf: Mapping, records: Sequence[Mapping], *, flags_given: bool | None = None, committed_text=None,
                       dev_probe: Any = None, gated: Sequence[str] = STRATA_GATED, probe_check: bool = True,
                       file_text: Callable[[str], str | None] | None = None) -> dict:
    """EVERY refusal of ``score`` that depends only on inputs known BEFORE the marker, in one place that both verbs call (``run`` before it takes the
    marker, exit 5, nothing spent; ``score`` first thing after it has read the lists): the stored gold re-normalises cleanly, the working tree is clean,
    the committed record states a valid gate kind (and, for a re-gate, usable ``movedOut``), ``--flags`` matches that kind (``flags_given`` is ``None``
    where the flags file is not known yet, as at ``run``), the eval manifest parses, and the dev-gold probe (``dev_probe``, the parsed rehearsal result;
    ``None`` is a refusal unless every record is a dev gold item, :func:`is_dev_rehearsal`) attests every dev item of each ``gated`` stratum a hit and is
    bound to this record commit, harness and dev gold (:func:`dev_probe_check`). ``score`` passes ``probe_check=False``: it does not
    re-validate the content (a harness-only cure after the run, ``-p1-gate-marker-ml.md`` §3, changes the harness hashes the probe carries) but requires the
    marker's pinned sha256 to equal the file's, i.e. the very file ``run`` validated before its marker. -> ``{"kind", "movedOut", "patterns"}``. A refusal that a later
    command can raise from such an input and the first irreversible step does not is a list spent for nothing."""
    require_clean_renormalize(renormalize_stored_gold(records))
    if pf["firstContact"]["dirtyPaths"]:
        raise PrecheckRefusal(f"the working tree is dirty ({pf['firstContact']['dirtyPaths']} paths under cpg/)")
    try:
        _, record = find_prereg_line(Path(repo) / "cpg" / "eval" / "preregistration.jsonl", gate_id, committed_text)
        kind, moved_out = gate_kind(record)
    except PrecheckRefusal:
        raise
    except Refusal as e:
        raise PrecheckRefusal(str(e)) from None
    if flags_given is not None:
        if kind == "re-gate" and not flags_given:
            raise PrecheckRefusal("the committed record is a re-gate: --flags is required")
        if kind == "gate" and flags_given:
            raise PrecheckRefusal("the committed record is gate 1: --flags is refused")
    try:
        patterns = scope_patterns(_load_yaml(Path(repo) / "cpg" / "eval" / "eval-manifest.yaml"))
    except Exception as e:                                       # noqa: BLE001 - an unreadable manifest is a refusal, whatever the parser raised
        raise PrecheckRefusal(f"the eval manifest is unusable ({type(e).__name__})") from None
    if probe_check and not is_dev_rehearsal(repo, records):      # a dev-only list cannot spend a held-out list; every other list needs the attestation
        dev_probe_check(dev_probe, repo, pf.get("recordCommit"), gated, committed_text, file_text)
    return {"kind": kind, "movedOut": moved_out, "patterns": patterns}


FORBIDDEN_PORTS = (6379, 6390, 16390, 16400)          # falkordb-dev, the ccg dev store and the other services of this lab: the gate never reads them


def reader_db_factory(system: str, credential: Callable[[str, str], str] | None = None) -> Callable[[str, int], Any]:
    """``(host, port) -> FalkorDB`` opened as the system's READ-ONLY ``raw`` role (``ccg_raw_<system>``: ``GRAPH.RO_QUERY`` on the live key, nothing
    else; ``ccg.store.acl``), with the credential of the credentials directory. Never the default user (a password-protected store refuses it, D1) and
    never a role that can write."""
    from ccg import sysname

    def factory(host: str, port: int):
        from falkordb import FalkorDB
        cred = credential
        if cred is None:
            from clients.common import _credential as cred
        return FalkorDB(host=host, port=port, username=sysname.acl_user("raw", system), password=cred(system, "raw"))
    return factory


def snapshot_graph(host: str, port: int | None, graph: str | None, system: str, db_factory: Callable[[str, int], Any] | None = None,
                   credential: Callable[[str, str], str] | None = None):
    """The read-only graph handle for the gate snapshot. Refuses (before any connection) a missing or forbidden port, a non-loopback host, and an
    explicit ``graph`` that is not the system's live key; the graph is derived from ``system`` when not given. Without a ``db_factory`` the connection is
    :func:`reader_db_factory`'s."""
    from ccg import sysname
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise Refusal("the gate reads a store on the loopback host only")
    if port is None or port in FORBIDDEN_PORTS or not 1 <= port <= 65535:
        raise Refusal("a store port is required and must not be one of the lab's other services")
    key = sysname.live_key(system)
    if graph is not None and graph != key:
        raise Refusal("--graph and --system disagree")
    if db_factory is None:
        db_factory = reader_db_factory(system, credential)
    return db_factory(host, port).select_graph(key)


def within_custodian(path: Path, home: Path | None, what: str) -> Path:
    """The resolved path, or :class:`Refusal` unless it lies under the custodian area (a symlink or ``..`` out of it is resolved first)."""
    area = _ac.custodian_area(home).resolve()
    p = Path(path).expanduser().resolve()
    if p != area and area not in p.parents:
        raise Refusal(f"{what} must be under the custodian area, never elsewhere")
    return p


def main(argv=None, snapshot=None, repo: Path | None = None, preflight_fn: Callable[..., dict] | None = None, committed_text=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--gate-id", required=True)
    v.add_argument("--repo", default=str(_CPG.parent))
    v.add_argument("--record-commit")
    sc = sub.add_parser("score")
    for f in ("--gate-id", "--lists", "--run", "--custodian"):
        sc.add_argument(f, required=True)
    sc.add_argument("--flags", help="re-gate only: regate-flags.jsonl (the kind itself comes from the committed record)")
    sc.add_argument("--exposure")
    sc.add_argument("--dev-probe", help="the dev-gold probe the rehearsal wrote (`dev-probe` verb): the very file `run` pinned in its marker; omitted only for a dev-only list")
    dp = sub.add_parser("dev-probe", help="write the dev-gold probe from a dev rehearsal's run.json, scored by the gate's own path against the committed dev gold")
    for f in ("--run", "--out"):
        dp.add_argument(f, required=True)
    for f, kw in (("--host", {"default": "127.0.0.1"}), ("--port", {"type": int}), ("--graph", {}), ("--repo", {}), ("--system", {})):
        dp.add_argument(f, **kw)
    sc.add_argument("--flow-decision", help="the committed ccg-p1-flow-python-v2-decision record (default: the repository's)")
    sc.add_argument("--host", default="127.0.0.1")
    sc.add_argument("--port", type=int)
    sc.add_argument("--graph")
    sc.add_argument("--repo", help="the repository whose manifest names the default system and whose tree is preflighted (default: this checkout's)")
    sc.add_argument("--system", help="default: the system the repository's manifest (cpg/ccg/system.yaml) declares")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "verify":
            rep = preflight(Path(a.repo), a.gate_id, record_commit_sha=a.record_commit)
            print(json.dumps(rep, sort_keys=True))
            return 0 if rep["label"] == "first-contact" else 1
        if a.cmd == "dev-probe":
            repo = Path(repo) if repo else Path(a.repo) if a.repo else _CPG.parent
            run = json.loads(Path(a.run).read_text(encoding="utf-8"))
            if snapshot is None:
                sys.path.insert(0, str(_HERE))
                from snapshot_export import RoRunner
                snapshot = StoreSnapshot(RoRunner(snapshot_graph(a.host, a.port, a.graph, a.system or manifest_system(repo) or "ccg")))
            doc = write_dev_probe(repo, run, snapshot, committed_text)
            Path(a.out).write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")
            reader = eval_commit_reader(repo)
            rows, complete = {}, True
            for s, r in doc["byStratum"].items():
                recs = _dev_gold(repo, s)[0]
                listed = set(r["hitIds"]) | set(r["keyErrorIds"])
                shapes = Counter(x for x in (item_withheld_shape(rec, reader) for rec in recs if rec["itemId"] not in listed) if x)
                rows[s] = {"items": len(recs), "hits": len(r["hitIds"]), "keyErrors": len(r["keyErrorIds"]), "withheld": sum(shapes.values()), "withheldByShape": dict(sorted(shapes.items()))}
                complete = complete and rows[s]["hits"] + rows[s]["keyErrors"] + rows[s]["withheld"] == rows[s]["items"]
            print(json.dumps({"strata": rows, "complete": complete}, sort_keys=True))
            return 0 if complete else 1
        cust = within_custodian(Path(a.custodian), None, "--custodian")
        repo = Path(repo) if repo else Path(a.repo) if a.repo else _CPG.parent
        records = []
        for f in a.lists.split(","):
            records += _jsonl(within_custodian(Path(f), None, "--lists"))
        pf = (preflight_fn or preflight)(repo, a.gate_id)
        if is_dev_rehearsal(repo, records):                  # a dev-only list: no probe was required or pinned, and a file named here is not read
            probe_sha = None
        else:
            _, probe_sha = read_dev_probe(Path(a.dev_probe)) if a.dev_probe else (None, None)
        chk = score_inputs_check(repo, a.gate_id, pf, records, flags_given=bool(a.flags), committed_text=committed_text, probe_check=False)      # what run checked before its marker
        kind, moved_out = chk["kind"], chk["movedOut"]
        key = list_key(records)                           # over every record of --lists, withheld strata included (the key gate_run took the marker under)
        if marker_state(cust, key) != "asked":            # -p1-gate-marker-ml.md §1.1.7: a started or failed run, or none, is never scored; first, so a run that
            raise Refusal("the run has no marker in state `asked` for this list: score refuses")    # never completed (no run file) is this refusal (D7)
        run = json.loads(within_custodian(Path(a.run), None, "--run").read_text(encoding="utf-8"))
        ctx = require_run_context(run)
        if pf.get("recordCommit") != ctx["recordCommit"]:          # CO-5 / P5-3: the named checkout is the run's checkout, whatever --repo says
            raise Refusal("the checkout's record commit differs from the one the run recorded: score refuses (score from the run's own checkout)")
        if run.get("listKey") != key:
            raise Refusal("the run file is not the run of these lists (its list key differs or is missing)")
        if snapshot is None:
            sys.path.insert(0, str(_HERE))
            from snapshot_export import RoRunner
            snapshot = StoreSnapshot(RoRunner(snapshot_graph(a.host, a.port, a.graph, a.system or manifest_system(repo) or "ccg")))
        live = snapshot.snapshot_id()
        if live != run.get("snapshotId"):
            raise Refusal("the live :Snapshot node differs from the run file's pin: score stops")
        # N3: the RUN's flow flag and withheld set decide what was asked, so they decide what is scored. The current decision record is never read here:
        # a flip between run and score would score never-asked items as `anchor-not-in-graph` and turn a complete run into a `fail`.
        flag = run.get("flowFlag")
        if not isinstance(flag, bool) or not isinstance(run.get("withheld"), dict):
            raise Refusal("the run file carries no recorded flow flag and withheld set: the run did not record them")
        if (read_marker(cust, key) or {}).get("pins", {}).get("flowFlag") != flag:
            raise Refusal("the run file's flow flag differs from the marker's pin: score refuses")
        gated = gated_strata(flag)
        if (read_marker(cust, key) or {}).get("pins", {}).get("devProbeSha256") != probe_sha:        # the very file `run` pinned (None for a dev-only list)
            raise Refusal("the dev-gold probe file is not the one the run pinned in its marker: score refuses")
        records, withheld = withhold_strata(records, gated)
        if withheld != run["withheld"]:
            raise Refusal("the lists do not reproduce the withheld set the run recorded: score refuses")
        # The split is recomputed from the tree at EVAL_COMMIT and compared by COUNTS with the run's. A committed edit of cpg/eval/EVAL_COMMIT after the run is
        # out of the threat model (it is in HARNESS_FILES, pinned by the record, and the owner's edit is a visible commit): no digest of the withheld ids.
        records, shapes = withhold_shapes(records, eval_commit_reader(repo))
        if shapes != run.get("withheldShapes"):
            raise Refusal("the lists do not reproduce the withheld-shape counts the run recorded: score refuses")
        flags = {r["itemId"]: r["flag"] for r in _jsonl(within_custodian(Path(a.flags), None, "--flags"))} if a.flags else None
        expo = _jsonl(within_custodian(Path(a.exposure), None, "--exposure")) if a.exposure else []
        res = score_gate(records, run, snapshot, chk["patterns"])
        dec = decide_gate(res, flags, moved_out, gated_strata=gated)
        label = "first-contact" if ctx["label"] == "first-contact" and pf["label"] == "first-contact" else "after-adaptation"
        rep = build_report(res, dec, flags=flags, exposure=expo, gated_strata=gated, withheld=withheld, flow_flag=flag,
                           withheld_shapes=shapes,
                           context={"label": label, "preflight": ctx, "scoreLabel": pf["label"]})
        write_custodian(cust, "gate-outcomes.jsonl", [{"itemId": i, **o} for i, o in sorted(res["items"].items())])
        write_custodian(cust, "gate-exposure.jsonl", exposure_rows(res))
        write_custodian(cust / "work", "gate-miss-breakdown.jsonl", miss_breakdown(records, res))     # CO-5: the fine breakdown is custodian-only, never printed
        write_custodian(cust, "gate-report.json", rep)
        print(json.dumps(rep, sort_keys=True))
        return 0 if dec["verdict"] == "pass" else 1
    except Refusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except Exception as e:                       # exit 1 is "not a pass"; every other failure class is its own code
        print(f"bad input: {type(e).__name__}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
