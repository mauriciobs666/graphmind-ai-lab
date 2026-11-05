#!/usr/bin/env python3
"""strip_gold.py — QW.jsonl -> QW-nogold.jsonl for the annotators (S48, P0-41; eval-protocol §13.2).

    strip_gold.py --eval-commit SHA --in QW.jsonl --out QW-nogold.jsonl --aid-map work/aid-map.jsonl
    strip_gold.py --verify --eval-commit SHA --in QW.jsonl --out QW-nogold.jsonl --aid-map work/aid-map.jsonl

Output by WHITELIST: each line is exactly ``{aid, lang, text}`` with
``aid = SHA-256("ccg-p0f-an-v1|" + EVAL_COMMIT + "|" + qid)[:12]``; lines sorted by ``aid`` so
neither QW's order nor its numbering reaches the annotators. The ``aid -> qid`` map goes to
``--aid-map`` only (``work/``). Prints counts and file paths, never item content.

``--verify``: every ``qid`` of the input maps to exactly one output line; every output object has
exactly the keys {aid, lang, text}; no output line contains a ``gold``, ``goldOptions``,
``intent``, ``group`` or ``ambiguityKind`` substring (plain case-sensitive substring, as §13.2
words it: a question whose text happens to contain one of those words fails verification).
Also checks the output is sorted and agrees with the map.

Exit: 0 ok · 1 invalid input / verification failed · 2 usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

KEYS = {"aid", "lang", "text"}
FORBIDDEN = ("gold", "goldOptions", "intent", "group", "ambiguityKind")


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_aid(eval_commit: str, qid: str) -> str:
    return hashlib.sha256(f"ccg-p0f-an-v1|{eval_commit}|{qid}".encode("utf-8")).hexdigest()[:12]


def read_qw(path: Path) -> list[dict]:
    rows, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            raise ValueError(f"{path.name}:{n}: not JSON")
        if not isinstance(r, dict):
            raise ValueError(f"{path.name}:{n}: not an object")
        for k in ("qid", "lang", "text"):
            if not isinstance(r.get(k), str) or not r[k]:
                raise ValueError(f"{path.name}:{n}: missing or non-string {k!r}")
        if r["qid"] in seen:
            raise ValueError(f"{path.name}:{n}: duplicate qid")
        seen.add(r["qid"])
        rows.append(r)
    return rows


def strip(rows: list[dict], eval_commit: str):
    out, amap = [], []
    for r in rows:
        aid = make_aid(eval_commit, r["qid"])
        out.append({"aid": aid, "lang": r["lang"], "text": r["text"]})
        amap.append({"aid": aid, "qid": r["qid"]})
    out.sort(key=lambda o: o["aid"])
    amap.sort(key=lambda o: o["aid"])
    if len({o["aid"] for o in out}) != len(out):
        raise ValueError("aid collision")
    return out, amap


def verify(rows: list[dict], eval_commit: str, out_path: Path, map_path: Path) -> list[str]:
    errs = []
    raw = out_path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in raw if l.strip()]
    objs = []
    for i, l in enumerate(lines, 1):
        try:
            o = json.loads(l)
        except ValueError:
            errs.append(f"line {i}: not JSON")
            continue
        objs.append(o)
        if not isinstance(o, dict) or set(o) != KEYS:
            errs.append(f"line {i}: keys are not exactly {sorted(KEYS)}")
        for bad in FORBIDDEN:
            if bad in l:
                errs.append(f"line {i}: contains {bad!r}")
    aids = [o.get("aid") for o in objs if isinstance(o, dict)]
    if aids != sorted(aids):
        errs.append("output is not sorted by aid")
    for r in rows:
        want = make_aid(eval_commit, r["qid"])
        if aids.count(want) != 1:
            errs.append(f"a qid maps to {aids.count(want)} output lines (want exactly 1)")
    if len(objs) != len(rows):
        errs.append(f"output has {len(objs)} lines for {len(rows)} qids")
    try:
        mp = [json.loads(l) for l in map_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, ValueError):
        mp = None
        errs.append("aid map unreadable")
    if mp is not None:
        got = {m.get("aid"): m.get("qid") for m in mp}
        want = {make_aid(eval_commit, r["qid"]): r["qid"] for r in rows}
        if got != want or len(mp) != len(rows):
            errs.append("aid map disagrees with the input")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--eval-commit", required=True)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--aid-map", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        rows = read_qw(a.inp)
        if a.verify:
            errs = verify(rows, a.eval_commit, a.out, a.aid_map)
            for e in errs:
                print(f"strip_gold: verify: {e}", file=sys.stderr)
            print(f"verify: {'FAIL' if errs else 'ok'}\nerrors: {len(errs)}\nitems: {len(rows)}")
            return 1 if errs else 0
        out, amap = strip(rows, a.eval_commit)
    except (ValueError, OSError) as e:
        print(f"strip_gold: {e}", file=sys.stderr)
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.aid_map.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(canon(o) + "\n" for o in out), encoding="utf-8")
    a.aid_map.write_text("".join(canon(o) + "\n" for o in amap), encoding="utf-8")
    print(f"items: {len(out)}\nout: {a.out}\naid-map: {a.aid_map}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
