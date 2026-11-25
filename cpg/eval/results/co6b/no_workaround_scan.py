#!/usr/bin/env python3
"""No-workaround scan of the ``ccg-prereg-v1`` / ``ac5-gate-1`` entry criterion, item 2 (``docs/plans/connected-code-graph-p1-closeout-prereg.md`` §3).

Run from ``cpg/`` on a clean tree:  ``python eval/results/co6b/no_workaround_scan.py [--probe ccg-dev-probe-v1.json]``  (read-only; dev gold only).

Definition (exact):
* literals: from every record of ``eval/dev/ac5-{http,mcp,scripts,wiring,flow}.jsonl``: ``itemId``, ``anchorId``, ``impactedId`` (class id); the ``path`` of
  ``anchor``, ``impacted`` and each ``serves`` locator (path); their ``qname`` (qname); a contract locator's ``keyNorm`` / ``key`` / ``cpId`` (key).
  A literal shorter than 4 characters is dropped. Basenames and path tails are NOT literals (informational count only).
* files: every git-tracked file under ``ccg/``, ``extractors/``, ``packs/`` (any extension, ``*.md`` included so prose hits are disclosed), plus
  ``eval/tools/withheld_shapes.py`` (the Q8 predicate: outside those directories, dev-derived, in the harness pin).
* a hit: the literal occurs as a substring of the file text. Hits in ``*.md`` are reported as ``prose``, the others as ``code``.
* second block: the Q8 predicate applied to the dev scripts items (a pure call on the gold locator and the file text at ``EVAL_COMMIT``): the flagged
  set by shape letter; with ``--probe`` the engine status (hit / key-error / neither) of each flagged item from that rehearsal probe.
Prints canonical JSON.
"""
from __future__ import annotations

import collections
import glob
import json
import subprocess
import sys
from pathlib import Path

CPG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CPG / "eval" / "tools"))
sys.path.insert(0, str(CPG))
import gate as G  # noqa: E402


def literals() -> dict[str, set[str]]:
    out: dict[str, set[str]] = collections.defaultdict(set)

    def add(v, cls):
        if isinstance(v, str) and len(v) >= 4:
            out[v].add(cls)

    def loc(l):
        if isinstance(l, dict):
            add(l.get("path"), "path")
            add(l.get("qname"), "qname")
            for k in ("keyNorm", "key", "cpId"):
                add(l.get(k), "key")

    for f in sorted(glob.glob(str(CPG / "eval" / "dev" / "ac5-*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for k in ("itemId", "anchorId", "impactedId"):
                add(r.get(k), "id")
            loc(r.get("anchor"))
            loc(r.get("impacted"))
            for s in r.get("serves") or []:
                loc(s)
    return out


def scan() -> dict:
    lits = literals()
    names = subprocess.run(["git", "ls-files", "-z", "--", "ccg", "extractors", "packs"], cwd=CPG, capture_output=True).stdout.decode().split("\0")
    names = [n for n in names if n] + ["eval/tools/withheld_shapes.py"]
    texts = {}
    for n in names:
        try:
            texts[n] = (CPG / n).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass
    hits = []
    for lit, cls in sorted(lits.items()):
        files = sorted(n for n, t in texts.items() if lit in t)
        if files:
            hits.append({"literal": lit, "classes": sorted(cls), "kinds": sorted({"prose" if f.endswith(".md") else "code" for f in files}), "files": files})
    return {"literals": len(lits), "byClass": dict(collections.Counter(c for cl in lits.values() for c in cl)), "files": len(texts),
            "withheldShapesPyScanned": "eval/tools/withheld_shapes.py" in texts, "hitLiterals": len(hits), "hits": hits}


def flagged(probe_path: str | None) -> dict:
    repo = CPG.parent
    reader = G.eval_commit_reader(repo)
    recs = [json.loads(l) for l in open(CPG / "eval" / "dev" / "ac5-scripts.jsonl", encoding="utf-8")]
    probe = json.load(open(probe_path)) if probe_path else None
    row = (probe or {}).get("byStratum", {}).get("scripts", {})
    rows = []
    for r in recs:
        s = G.item_withheld_shape(r, reader)
        if s:
            status = None
            if probe:
                status = "hit" if r["itemId"] in row.get("hitIds", []) else "key-error" if r["itemId"] in row.get("keyErrorIds", []) else "neither"
            rows.append({"itemId": r["itemId"], "shape": s, "engineStatus": status})
    return {"devScriptsItems": len(recs), "flagged": len(rows), "byShape": dict(sorted(collections.Counter(x["shape"] for x in rows).items())),
            "engineStatusSource": probe_path or "not given: read from the rehearsal's probe", "items": rows}


if __name__ == "__main__":
    probe = sys.argv[sys.argv.index("--probe") + 1] if "--probe" in sys.argv else None
    print(json.dumps({"scan": scan(), "predicateFlaggedDev": flagged(probe)}, sort_keys=True, separators=(",", ":")))
