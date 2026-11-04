#!/usr/bin/env python3
"""Materialize the four fixture repositories as git repositories (S39, TP-001).

    cd cpg && .venv/bin/python fixtures/materialize.py [--out .ccg-work/fixtures] [--force] [--json]

Plan: ``docs/test-plans/connected-code-graph-fixtures.md`` section 8 and section 9 (TP-001). The checked-in
tree under ``cpg/fixtures/system1/<repo>/`` is each repository's *base tree*; the manifest's ``history``
block (schema ``cpg/ccg/schemas/fixture-manifest-v1.schema.json``) says how to reach every commit: H0 is the
base tree plus ``H0.ops``, each later commit its predecessor plus its ``ops``, forward ops only
(``add_file``, ``replace_text``, ``delete_file``). Branches are a base commit plus ops.

Determinism: the result depends only on the base tree bytes, the manifests, and the constants below. The
author, the committer and both dates are fixed; git is run with no system or global config, no hooks, no
signing, no CRLF conversion; file modes are normalized to 0644/0755; tags are lightweight; the process
umask, TZ and HOME do not matter. Two runs produce identical SHAs for every commit, branch and tag.

Safety: ``--force`` removes ``<out>/<repo>`` only when it carries this script's marker (``.git/ccg-fixture``).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SYSTEM = HERE / "system1"
EXPECTED = SYSTEM / "expected"
CPG_ROOT = HERE.parent
DEFAULT_OUT = CPG_ROOT / ".ccg-work" / "fixtures"
REPOS = ("fx-svc", "fx-web", "fx-lib", "fx-job")
MARKER = "ccg-fixture"

AUTHOR_NAME = "Fixture Author"
AUTHOR_EMAIL = "fixture@example.invalid"
COMMIT_TIME = "12:00:00 +0000"  # on the manifest's date
BRANCH_TIME = "13:00:00 +0000"  # a branch commit: the base commit's date, one hour later
SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".pytest_cache"}
SKIP_SUFFIXES = (".pyc", ".pyo")


class MaterializeError(RuntimeError):
    pass


def _fix_keys(o):
    """YAML 1.1 reads an unquoted ``on``/``off`` key as a boolean; the SK rows use them as block names."""
    if isinstance(o, dict):
        return {({True: "on", False: "off"}.get(k, k) if isinstance(k, bool) else k): _fix_keys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_fix_keys(v) for v in o]
    return o


def load_manifest(repo: str, expected_dir: Path = EXPECTED) -> dict:
    with open(expected_dir / f"{repo}.yaml", encoding="utf-8") as fh:
        return _fix_keys(yaml.safe_load(fh))


def _git_env(date: str | None) -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": "/nonexistent",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "TZ": "UTC",
        "LC_ALL": "C",
    }
    if date is not None:
        env.update(
            GIT_AUTHOR_NAME=AUTHOR_NAME,
            GIT_AUTHOR_EMAIL=AUTHOR_EMAIL,
            GIT_AUTHOR_DATE=date,
            GIT_COMMITTER_NAME=AUTHOR_NAME,
            GIT_COMMITTER_EMAIL=AUTHOR_EMAIL,
            GIT_COMMITTER_DATE=date,
        )
    return env


def _git(cwd: Path, *args: str, date: str | None = None) -> str:
    cfg = ["-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-c", "core.hooksPath=/dev/null",
           "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", "-c", "core.fileMode=true",
           "-c", "core.quotepath=false"]
    r = subprocess.run(["git", *cfg, *args], cwd=cwd, env=_git_env(date), capture_output=True, text=True)
    if r.returncode != 0:
        raise MaterializeError(f"git {' '.join(args)} failed in {cwd}: {r.stderr.strip()}")
    return r.stdout.strip()


def _copy_base(src: Path, dst: Path) -> None:
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(SKIP_SUFFIXES):
                continue
            s = Path(root) / name
            d = dst / s.relative_to(src)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, d)
            d.chmod(0o755 if os.access(s, os.X_OK) and s.stat().st_mode & 0o111 else 0o644)


def apply_ops(tree: Path, ops: list[dict], where: str) -> None:
    """Apply forward ops to a working tree. Every precondition is strict (a missed anchor is an error)."""
    for i, op in enumerate(ops):
        kind, rel = op["op"], op["path"]
        target = tree / rel
        tag = f"{where} op {i} ({kind} {rel})"
        if kind == "add_file":
            if target.exists():
                raise MaterializeError(f"{tag}: file already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(op["content"].encode("utf-8"))
            target.chmod(0o644)
        elif kind == "delete_file":
            if not target.is_file():
                raise MaterializeError(f"{tag}: no such file")
            target.unlink()
        elif kind == "replace_text":
            if not target.is_file():
                raise MaterializeError(f"{tag}: no such file")
            text = target.read_bytes().decode("utf-8")
            n = text.count(op["from"])
            if n != 1:
                raise MaterializeError(f"{tag}: anchor occurs {n} times, expected exactly once")
            target.write_bytes(text.replace(op["from"], op["to"], 1).encode("utf-8"))
        else:
            raise MaterializeError(f"{tag}: unknown op")


def _stamp(date: str, clock: str) -> str:
    return f"{date} {clock}"


def materialize_repo(repo: str, out: Path, expected_dir: Path = EXPECTED, system_dir: Path = SYSTEM) -> dict:
    manifest = load_manifest(repo, expected_dir)
    hist = manifest["history"]
    dst = out / repo
    dst.mkdir(parents=True)
    _git(dst, "init", "-q", "-b", "main")
    (dst / ".git" / MARKER).write_text("ccg fixture repository (cpg/fixtures/materialize.py)\n")
    _copy_base(system_dir / repo, dst)
    commits: dict[str, str] = {}
    dates: dict[str, str] = {}
    for c in hist["commits"]:
        apply_ops(dst, c["ops"], f"{repo} {c['id']}")
        _git(dst, "add", "-A", "--force")
        stamp = _stamp(c["date"], COMMIT_TIME)
        _git(dst, "commit", "-q", "--allow-empty", "-m", f"{c['id']}: {c['message']}", date=stamp)
        commits[c["id"]] = _git(dst, "rev-parse", "HEAD")
        dates[c["id"]] = c["date"]
        _git(dst, "tag", c["id"])
    head_id = hist["commits"][-1]["id"]
    branches: dict[str, str] = {"main": commits[head_id]}
    for b in hist.get("branches", []):
        _git(dst, "checkout", "-q", "-b", b["name"], commits[b["base"]])
        apply_ops(dst, b["ops"], f"{repo} branch {b['name']}")
        _git(dst, "add", "-A", "--force")
        stamp = _stamp(dates[b["base"]], BRANCH_TIME)
        _git(dst, "commit", "-q", "--allow-empty", "-m", f"{b['name']}: {b.get('changeSet', 'change set')}", date=stamp)
        branches[b["name"]] = _git(dst, "rev-parse", "HEAD")
        _git(dst, "checkout", "-q", "main")
    status = _git(dst, "status", "--porcelain")
    if status:
        raise MaterializeError(f"{repo}: working tree not clean after materialization: {status!r}")
    return {"commits": commits, "branches": branches, "tags": dict(commits), "head": commits[head_id]}


def materialize(out: Path = DEFAULT_OUT, repos=REPOS, force: bool = False,
                expected_dir: Path = EXPECTED, system_dir: Path = SYSTEM) -> dict:
    """Create the sibling repositories under *out*; return {repo: {commits, branches, tags, head}}."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        d = out / repo
        if d.exists():
            if not force:
                raise MaterializeError(f"{d} exists; pass force=True / --force to rebuild it")
            if not (d / ".git" / MARKER).is_file():
                raise MaterializeError(f"{d} exists and is not a materialized fixture repository; refusing to remove it")
            shutil.rmtree(d)
    return {repo: materialize_repo(repo, out, expected_dir, system_dir) for repo in repos}


# ---------------------------------------------------------------------------------------------------------
# Manifest helpers (shared by tests/test_fixtures.py and the cross.yaml emitter)
# ---------------------------------------------------------------------------------------------------------
import re  # noqa: E402

SHORT = {"svc": "fx-svc", "web": "fx-web", "lib": "fx-lib", "job": "fx-job"}
LONG = set(REPOS)
CROSS_PATH = EXPECTED / "cross.yaml"
# rows kept in cross.yaml although one manifest holds them: the S39 builder gate checks them (AB-1, LNK-42/43)
CROSS_UNIT_ROWS = ("AB-1", "LNK-42", "LNK-43")
# row fields that are free text, never a locator
PROSE = {"note", "item", "whyUnprovable", "expected", "shape_syntax", "effect", "planted", "why", "call", "query",
         "versionSkew_note", "consumer_form", "publish_call_site", "same_version", "change", "pin", "stub",
         "expectedOverrideOn", "expectedOverrideOff", "on", "off", "message", "stateBefore", "stateAfter"}
_LOC = re.compile(r"(?<![\w/.-])(svc|web|lib|job|fx-svc|fx-web|fx-lib|fx-job):([^\s:@\"']+)(?:::(\S+)| @(\S+))?")
_DICT_REPO_KEYS = ("repoId", "path")


def _strip_tail(token: str) -> str:
    token = token.rstrip(".,;")
    while token.endswith(")") and token.count(")") > token.count("("):
        token = token[:-1].rstrip(".,;")
    return token


def parse_locator(text: str):
    """Parse one locator string ``repo:path::qname`` / ``repo:path @keyPath`` / ``repo:path``; None if not one."""
    m = _LOC.match(text.strip())
    if not m:
        return None
    repo = SHORT.get(m.group(1), m.group(1))
    qname = _strip_tail(m.group(3)) if m.group(3) else None
    key = _strip_tail(m.group(4)) if m.group(4) else None
    return {"repo": repo, "path": m.group(2), "qname": qname, "keyPath": key}


def locator_string(loc: dict) -> str:
    if loc.get("qname"):
        return f"{loc['repo']}:{loc['path']}::{loc['qname'].split('::', 1)[-1] if loc['qname'].startswith(loc['path'] + '::') else loc['qname']}"
    if loc.get("keyPath"):
        return f"{loc['repo']}:{loc['path']} @{loc['keyPath']}"
    return f"{loc['repo']}:{loc['path']}"


def _walk(value, field):
    """Yield (field, locator dict) for every locator inside *value* (structured dicts and strings)."""
    if isinstance(value, dict):
        if "repoId" in value and "path" in value and "unitId" not in value and "lang" not in value:
            qn = value.get("qname")
            if qn and qn.startswith(value["path"] + "::"):
                qn = qn[len(value["path"]) + 2:]
            if qn and value.get("signature"):          # fx-svc keeps a Java section-4.3 signature in its own field
                qn += value["signature"]
            yield field, {"repo": value["repoId"], "path": value["path"], "qname": qn,
                          "keyPath": value.get("keyPath"), "objectKey": value.get("objectKey")}
            return
        for k, v in value.items():
            k = {True: "on", False: "off"}.get(k, k)
            if k in PROSE:
                continue
            yield from _walk(v, k if field is None or field == "" else field)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v, field)
    elif isinstance(value, str):
        for m in _LOC.finditer(value):
            loc = parse_locator(value[m.start():])
            if loc:
                yield field, loc


def row_locators(row: dict):
    """All (field, locator) pairs of a row, in field order. A ``*_line`` field is attached as ``loc['line']``."""
    out = []
    for k, v in row.items():
        k = {True: "on", False: "off"}.get(k, k)
        if k in PROSE or k.endswith("_line"):
            continue
        for field, loc in _walk(v, k):
            loc = dict(loc)
            line = row.get(f"{k}_line")
            if line is not None:
                loc["line"] = line
            out.append((k, loc))
    return out


ROLE = {"P": "provider", "provider": "provider", "C": "consumer", "consumer": "consumer"}


def row_view(row: dict) -> dict:
    """The comparable core of a row, whatever the builder's dialect: id, status, reason, gate, key, kind."""
    return {"id": row["id"], "kind": row.get("kind"), "status": row.get("status", row.get("st")),
            "reason": row.get("reason"), "gate": row["gate"], "key": row.get("keyNorm", row.get("key"))}


def load_all(expected_dir: Path = EXPECTED) -> dict:
    return {r: load_manifest(r, expected_dir) for r in REPOS}


def holders(manifests: dict) -> dict:
    """{row id: {repo: row}} across the four manifests, ids in first-seen order."""
    out: dict = {}
    for repo in REPOS:
        for row in manifests[repo]["rows"]:
            out.setdefault(row["id"], {})[repo] = row
    return out


class ReconcileError(ValueError):
    pass


def reconcile(manifests: dict) -> list[dict]:
    """The cross.yaml rows: every id held by several manifests (deduped by id), plus CROSS_UNIT_ROWS."""
    rows = []
    for rid, held in holders(manifests).items():
        if len(held) < 2 and rid not in CROSS_UNIT_ROWS:
            continue
        views = {repo: row_view(r) for repo, r in held.items()}
        merged = {}
        for attr in ("kind", "status", "reason", "gate", "key"):
            vals = {repr(v[attr]) for v in views.values() if v[attr] is not None}
            if len(vals) > 1:
                raise ReconcileError(f"{rid}: holders disagree on {attr}: {sorted(vals)}")
            merged[attr] = next((v[attr] for v in views.values() if v[attr] is not None), None)
        buckets = {"provider": [], "consumer": [], "refs": []}
        for repo, r in held.items():
            for field, loc in row_locators(r):
                s = locator_string(loc)
                b = buckets[ROLE.get(field, "refs")]
                if s not in b:
                    b.append(s)
        refs = [x for x in buckets["refs"] if x not in buckets["provider"] and x not in buckets["consumer"]]
        first = held.get("fx-svc") or next(iter(held.values()))
        row = {"id": rid}
        if merged["kind"]:
            row["kind"] = merged["kind"]
        if merged["key"] is not None:
            row["keyNorm"] = merged["key"]
        if merged["status"]:
            row["status"] = merged["status"]
        if merged["reason"]:
            row["reason"] = merged["reason"]
        row.update(gate=merged["gate"], stratum="fixture-dev", serves=list(first.get("serves") or []),
                   holders=list(held), why="multi-manifest" if len(held) > 1 else "cross-unit")
        for extra in ("scenario", "unit"):
            val = next((r[extra] for r in held.values() if extra in r), None)
            if val is not None:
                row[extra] = val
        for name in ("provider", "consumer"):
            if buckets[name]:
                row[name] = buckets[name]
        if refs:
            row["refs"] = refs
        rows.append(row)
    return rows


def emit_cross(manifests: dict | None = None) -> str:
    manifests = manifests or load_all()
    doc = {"manifest": {"repoId": "cross", "formatVersion": 1, "stratum": "fixture-dev",
                        "plan": "docs/test-plans/connected-code-graph-fixtures.md",
                        "note": "generated by cpg/fixtures/materialize.py --emit-cross; do not hand-edit"},
           "rows": reconcile(manifests)}
    head = ("# Cross-repository rows (S39): every row id held by several per-repo manifests, deduplicated, plus the\n"
            "# cross-unit build-state rows AB-1, LNK-42, LNK-43. Locators: repoId:path::qname | repoId:path @keyPath.\n"
            "# Regenerate: cd cpg && .venv/bin/python fixtures/materialize.py --emit-cross\n")
    return head + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10000)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="parent directory of the four repositories")
    ap.add_argument("--force", action="store_true", help="rebuild repositories this script created earlier")
    ap.add_argument("--json", action="store_true", help="print the SHA map as JSON")
    ap.add_argument("--emit-cross", action="store_true", help="rewrite expected/cross.yaml from the four manifests and exit")
    ns = ap.parse_args(argv)
    if ns.emit_cross:
        CROSS_PATH.write_text(emit_cross())
        print(f"wrote {CROSS_PATH}")
        return 0
    try:
        shas = materialize(ns.out, force=ns.force)
    except MaterializeError as exc:
        print(f"materialize: {exc}", file=sys.stderr)
        return 1
    if ns.json:
        print(json.dumps(shas, indent=2, sort_keys=True))
    else:
        for repo, info in shas.items():
            print(f"{repo}: head {info['head'][:12]}  " + " ".join(f"{k}={v[:7]}" for k, v in info["tags"].items())
                  + ("  branches: " + " ".join(f"{k}={v[:7]}" for k, v in info["branches"].items() if k != "main") if len(info["branches"]) > 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
