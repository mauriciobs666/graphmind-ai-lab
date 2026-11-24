"""CO-2 plumbing: a throwaway git repository of host-language files -> the real build path (``ccg.build``) -> the real
assembler (``ccg.assemble.rows.build_rows``) -> the real answer builder over an in-memory reader. Nothing here is derived
from the code under test; every expected value is written in the test that asserts it. No store, no Docker, no network."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from engine_answer_helpers import AnswerReader

from ccg import atrest, build, bundle, ids
from ccg.assemble.rows import build_rows
from ccg.engine import anchors
from ccg.engine import answer_build as AB
from ccg.linker import Links
from ccg.packrt import PackFacts
from ccg.units import Unit
from extractors.markdown.globs import load_globs

CPG = Path(__file__).resolve().parents[1]
SYSTEM = "co2"
REPO = "r"


def git(path: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def assemble(tmp_path: Path, files: dict, *, langs=("shell", "markdown"), tweak=None):
    """Commit ``files`` to a fresh repository, build one whole-repository unit per language of ``langs`` and assemble the rows.
    ``files`` maps a path to its text, or a repository id to such a map (several repositories, each its own git repository).
    ``tweak(records)`` may rewrite a bundle's records (a hand-made input the extractors do not emit) before the assembler reads them."""
    repos = files if all(isinstance(v, dict) for v in files.values()) else {REPO: files}
    work, keys, audit = tmp_path / "work", tmp_path / "keys", tmp_path / "audit"
    for d in (work, keys, audit):
        d.mkdir(mode=0o700, parents=True)
    where, heads = {}, {}
    for repo_id, tree in repos.items():
        repo = where[repo_id] = tmp_path / "repo" / repo_id
        for rel, text in tree.items():
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        git(repo, "init", "-q", "-b", "main")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "c1")
        heads[repo_id] = git(repo, "rev-parse", "HEAD")
    units = [Unit(ids.unit_id(rid, ".", lang), rid, ".", lang) for rid in repos for lang in langs]
    ctx = build.BuildContext(units=tuple(units), root=work, keys_root=keys, built_at="2026-11-11T00:00:00Z",
                             globs=load_globs(CPG / "eval" / "eval-manifest.yaml"))
    bundles = []
    for u in units:
        res = build.build_unit(u, where[u.repoId], heads[u.repoId], SYSTEM, ctx)
        assert res.ok, (u.language, res.fail_reason)
        bundles.append(bundle.loads(atrest.read_bundle(SYSTEM, u.unitId, heads[u.repoId], work)))
    if tweak is not None:
        bundles = [tweak(b) for b in bundles]
    rows = build_rows(bundles, Links([], [], {}), PackFacts([], [], [], [], [], [], []), system=SYSTEM, packs=[], audit_dir=audit)
    return SimpleNamespace(rows=rows, bundles=bundles)


def script_symbol(world, path: str, repo: str = REPO) -> dict:
    (sym,) = [s for s in world.rows.nodes["Symbol"] if s["path"] == path and s["name"] == "<script>" and s["repoId"] == repo]
    return sym


def calls_edges(world) -> list[dict]:
    """Every ``CallSite -[CALLS]-> Symbol`` row: site id, target symbol id, status, reason (defaults read as the reader does)."""
    out = []
    for e in world.rows.edges.get(("CallSite", "CALLS", "Symbol"), []):
        p = e.get("p", {})
        out.append({"site": e["s"], "target": e["t"], "status": p.get("status", "confirmed"), "reason": p.get("reason", "resolved-call")})
    return out


def edges_into(world, path: str, repo: str = REPO) -> list[dict]:
    t = script_symbol(world, path, repo)["symbolId"]
    return [e for e in calls_edges(world) if e["target"] == t]


def ask_impact(world, path: str) -> dict:
    """The real answer builder's ``impact`` document for the ``<script>`` symbol at ``path``."""
    n = script_symbol(world, path)
    anchor = anchors.SymbolAnchor(n["symbolId"], n["qname"], n["repoId"], n["path"], n["unitId"], n["lineStart"], n["lineEnd"])
    return AB.build_answer(AnswerReader(world), anchor)


def item_names(doc: dict) -> list[str]:
    return sorted(i["subject"]["qname"] for i in doc["items"])


def sites_of(world, path: str) -> list[dict]:
    """The call sites the file at ``path`` holds, by line: ``(line, calleeName, id)`` as dicts."""
    return sorted(({"line": c["line"], "calleeName": c["calleeName"], "id": c["callSiteId"]} for c in world.rows.nodes.get("CallSite", [])
                   if c["path"] == path), key=lambda c: (c["line"], c["calleeName"]))


def edge_from(world, path: str, callee: str) -> list[dict]:
    """The ``CALLS`` edges whose call site is in ``path`` and named ``callee``, with the target's path."""
    by_id = {s["symbolId"]: s["path"] for s in world.rows.nodes["Symbol"]}
    ids_ = {c["id"] for c in sites_of(world, path) if c["calleeName"] == callee}
    return [{**e, "targetPath": by_id[e["target"]]} for e in calls_edges(world) if e["site"] in ids_]
