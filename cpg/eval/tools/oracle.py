#!/usr/bin/env python3
"""oracle.py - runtime-oracle scorer (P1-36; ``docs/plans/connected-code-graph-p1-ml.md`` §6.5, EP §11).

    oracle.py --capture <run>/all.jsonl --links LINKS.jsonl --misses-out CUSTODIAN/misses.jsonl

``LINKS.jsonl`` is the gate snapshot's Link set dumped by the custodian, one ``{"element", "keyNorm"}`` per line
(element ``path::func`` or ``path``); this tool never queries a store. stdout is a counts-only JSON report; every
oracle miss goes to ``--misses-out`` (refused inside the repo), never to stdout.

Rules (P1-47 gate rulings, fixed here before any Link data is seen):
* Scope. In scope = a non-test row whose path (query stripped) is ``/shop/api`` or under it: the storefront API
  (the SPA, and any other client of it). Everything else (load_append.py, load_test.sh, LM Studio on localhost:1234,
  ...) is ``out_of_scope``: counted, never scored. A row whose host is ``testserver`` (Starlette TestClient), whose
  class is ``testclient`` or whose frame is under a ``tests/`` directory is test code, outside the denominator.
* Grain. A row whose frame names no function (every SPA edge: wrappers hide it) matches at FILE grain, any Link whose
  element is in that file, and is reported as its own ``file`` class; function-grain rows need the exact element.
  The ``pooledAcrossGrains`` line is for reference only.
* Keys. Compared per path segment on the raw runtime path: a static ``{?}``/``{}``/``{#}`` matches any one non-empty
  runtime segment, a literal must be equal; the verb must be equal (``*`` matches any). Method + path-without-query.
  Distinct-key and distinct-edge identity is the capture's own ``match_keys`` keyNorm (UUIDs masked, short ids not).
* An in-scope row without a frame file is ``unattributed``: counted, not scored, never guessed.
* Statistics: observed in-scope distinct edges (element, keyNorm) whose some raw row matches a Link, over observed
  distinct edges; the same over distinct keys; Wilson 95% on each. Reported beside seeded recall, never pooled.
Exit: 0 ok, 2 usage/refused, 4 bad input.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STOREFRONT_PREFIX = "/shop/api"
PLACEHOLDERS = frozenset({"{?}", "{}", "{#}"})
UNRUN_ENTRY_POINTS = (
    "falkor-chat/web/app.js", "salesperson/scripts/load_demo.py", "salesperson/scripts/swap_bench.py",
    "salesperson/scripts/language_salience_probe.py", "salesperson/scripts/qwen_watch.sh", "salesperson/scripts/install_node.sh")


def wilson(k: int, n: int, z: float = 1.959964) -> dict:
    """95% Wilson score interval, rounded to 3 places; ``s``/``lower``/``upper`` are None for n == 0."""
    if n == 0:
        return {"k": k, "n": n, "s": None, "lower": None, "upper": None}
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "s": round(p, 3), "lower": round(max(0.0, centre - half), 3),
            "upper": round(min(1.0, centre + half), 3)}


def _path(row: dict) -> str:
    return urlsplit(row["path"]).path or "/"


def classify(row: dict) -> str:
    """``testclient`` | ``in_scope`` | ``out_of_scope`` (module docstring, Scope)."""
    file = (row.get("frame") or {}).get("file") or ""
    if row.get("cls") == "testclient" or row.get("host") == "testserver" or "/tests/" in "/" + file:
        return "testclient"
    p = _path(row)
    if p == STOREFRONT_PREFIX or p.startswith(STOREFRONT_PREFIX + "/"):
        return "in_scope"
    return "out_of_scope"


def key_matches(static_key: str, method: str, raw_path: str) -> bool:
    """Does a static ``"<VERB> <path>"`` key cover this runtime request? Per path segment."""
    verb, _, spath = static_key.partition(" ")
    if verb != "*" and verb.upper() != method.upper():
        return False
    ss, rs = spath.split("/"), (urlsplit(raw_path).path or "/").split("/")
    if len(ss) != len(rs):
        return False
    return all(r != "" and s in PLACEHOLDERS or s == r for s, r in zip(ss, rs))


def _keynorm(row: dict) -> str:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parents[1]))
    import match_keys as M  # noqa: E402
    return M.normalize_contract({"kind": "http", "keyNorm": f"{row['method']} {_path(row)}"}, "consumer")["keyNorm"]


def _stat(hit: int, n: int) -> dict:
    return wilson(hit, n)


def score(rows: list[dict], links: list[tuple[str, str]]) -> tuple[dict, list[dict]]:
    """``(report, misses)``. ``links`` are ``(element, keyNorm)`` pairs of the gate snapshot's Links."""
    by_file: dict[str, list[str]] = {}
    for el, key in links:
        by_file.setdefault(el.split("::")[0], []).append((el, key))
    counts = {"in_scope": 0, "out_of_scope": 0, "testclient": 0}
    unattributed = 0
    oos_edges: set = set()
    edges: dict = {}                       # (grain, element, keyNorm) -> hit?
    for r in rows:
        c = classify(r)
        counts[c] += 1
        if c == "out_of_scope":
            fr = r.get("frame") or {}
            oos_edges.add((fr.get("file"), fr.get("func"), _keynorm(r)))
        if c != "in_scope":
            continue
        fr = r.get("frame") or {}
        if not fr.get("file"):
            unattributed += 1
            continue
        file, func = fr["file"], fr.get("func")
        grain = "function" if func else "file"
        element = f"{file}::{func}" if func else file
        cand = by_file.get(file, [])
        hit = any((grain == "file" or el == element) and key_matches(k, r["method"], r["path"]) for el, k in cand)
        ident = (grain, element, _keynorm(r))
        edges[ident] = edges.get(ident, False) or hit

    def block(sel) -> dict:
        es = {i: h for i, h in edges.items() if sel(i)}
        keys: dict = {}
        for (_, _, key), h in es.items():
            keys[key] = keys.get(key, False) or h
        return {"edges": _stat(sum(es.values()), len(es)), "keys": _stat(sum(keys.values()), len(keys))}

    misses = [{"element": el, "keyNorm": key, "grain": g} for (g, el, key), h in sorted(edges.items()) if not h]
    report = {
        "rows": len(rows), "inScopeRows": counts["in_scope"], "outOfScope": counts["out_of_scope"],
        "testclient": counts["testclient"], "outOfScopeEdges": len(oos_edges), "unattributed": unattributed,
        "classes": {g: block(lambda i, g=g: i[0] == g) for g in ("function", "file")},
        "pooledAcrossGrains": block(lambda i: True),
        "misses": len(misses),
        "unrunEntryPoints": list(UNRUN_ENTRY_POINTS),
        "notes": ["no mcp_tool oracle edge exists at P1", "/order/advance is one key: transitions are not distinguished"],
    }
    return report, misses


def _jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--links", required=True)
    ap.add_argument("--misses-out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.misses_out).expanduser().resolve()
    if REPO in out.parents:
        print(f"REFUSED: --misses-out {out} is inside the repo; it belongs under the custodian root", file=sys.stderr)
        return 2
    try:
        rows = _jsonl(a.capture)
        links = [(l["element"], l["keyNorm"]) for l in _jsonl(a.links)]
    except (OSError, ValueError, KeyError) as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 4
    report, misses = score(rows, links)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(m, sort_keys=True) + "\n" for m in misses), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
