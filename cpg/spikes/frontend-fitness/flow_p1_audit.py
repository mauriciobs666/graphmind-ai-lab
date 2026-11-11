#!/usr/bin/env python3
"""P1-34b: audit tooling of the pre-registration `ccg-p1-flow-python-v1` (`docs/plans/connected-code-graph-p1-flow-prereg.md` §0.2, §2.4, §2.5).
Standard library only; independent of the P0 `adjudication/tools/python/flowaud_draw.py` (hard-wired to P0, never imported or reused).

    flow_p1_audit.py flowaud-p1 --labels labels/python-flow-p1.jsonl [--extend] [--out F]
        F1 = 60 pairs by ascending SHA-256("ccg-p1a-v1-flowaud|python|pair|<ordinal>|<pairIndex>") over ALL pairs (uniform), plus a SEPARATE set
        of up to 20 `unsure` pairs not already drawn (same hash); F2 = first 6 methods by "...|method|<ordinal>", --extend adds the next 3.
        Emits ONLY ordinals, pair indexes and the unsure flag -- never label content.
    flow_p1_audit.py adjlist-p1 --answers A.jsonl --truth T.jsonl --out LIST.jsonl      (run by `teco`, never by an adjudicator)
        n = pairs missed in either configuration (decision cell).  List = the n misses + connected pairs (connected in both configurations)
        by ascending SHA-256("ccg-p1a-v1-adj|python|<commit>|conn|<ordinal>|<pairIndex>"): 120 - n of them when n <= 60, 60 when n > 60
        (all of them if fewer).  Order = ascending hash of "...|shuf|<ordinal>|<pairIndex>".  Rows carry a WHITELISTED key set: no status.
        Prints the list's sha256.  `--verify LIST` re-derives the list and aborts on any deviation (a missing miss, duplicates, length,
        composition, order)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

COMMIT = "0caf85e3ccb757801d15d6babe55a63477a486dd"
FLOWAUD_SEED = "ccg-p1a-v1-flowaud|python"
ADJ_SEED = f"ccg-p1a-v1-adj|python|{COMMIT}"
F1_PAIRS = 60
F1_UNSURE = 20
F2_METHODS = 6
F2_EXTEND = 3
LIST_LEN = 120
CONN_CAP = 60
ROW_KEYS = ("ordinal", "pairIndex", "kind", "source", "sink", "evidence")   # the whitelist: nothing else reaches an adjudicator
STATUS_WORDS = {"connected", "on", "off", "miss", "missed", "status", "confidence", "adjudication", "y", "G", "M"}


class AuditError(SystemExit):
    pass


def H(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _idx(v):
    return type(v) is int and v >= 0   # not bool, float, str or negative


def _need_ints(rows, label, *fields):
    """Abort (value-free: the file label, the row number and the field name only) unless every named field is a non-negative int."""
    for n, r in enumerate(rows, 1):
        for f in fields:
            if not isinstance(r, dict) or not _idx(r.get(f)):
                raise AuditError(f"{label}: row {n}: `{f}` must be a non-negative integer")


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------------------------------------------------- flowaud-p1
def flowaud(label_rows, extend=False):
    _need_ints(label_rows, "labels", "ordinal")
    pairs, methods = [], []
    for m in label_rows:
        methods.append(m["ordinal"])
        for i, p in enumerate(m["pairs"]):
            pairs.append((m["ordinal"], i, p.get("confidence") == "unsure"))
    if len(set(methods)) != len(methods):
        raise AuditError("labels: a duplicate ordinal")
    order = sorted(pairs, key=lambda t: H(f"{FLOWAUD_SEED}|pair|{t[0]}|{t[1]}"))
    f1 = order[:F1_PAIRS]
    drawn = {(o, i) for o, i, _ in f1}
    unsure = [t for t in order if t[2] and (t[0], t[1]) not in drawn][:F1_UNSURE]
    meth = sorted(methods, key=lambda o: H(f"{FLOWAUD_SEED}|method|{o}"))
    row = lambda t: {"ordinal": t[0], "pairIndex": t[1], "unsure": t[2]}  # noqa: E731
    out = {"hash": "sha256", "pairKey": f"{FLOWAUD_SEED}|pair|<ordinal>|<pairIndex>", "methodKey": f"{FLOWAUD_SEED}|method|<ordinal>",
           "nMethods": len(methods), "nPairs": len(pairs),
           "F1": [row(t) for t in f1], "F1unsure": [row(t) for t in unsure],
           "F2": {"methods": meth[:F2_METHODS]}}
    if extend:
        out["F2"]["extension"] = meth[F2_METHODS:F2_METHODS + F2_EXTEND]
    return out


# ------------------------------------------------------------------------------------------------------------------------ adjlist-p1
def _classify(answers):
    _need_ints(answers, "answers", "ordinal", "idx")
    keys = [(a["ordinal"], a["idx"]) for a in answers]
    if len(set(keys)) != len(keys):
        raise AuditError("answers: a duplicate (ordinal, idx)")
    if any(a.get("on") is None for a in answers):
        raise AuditError("answers: the `on` configuration is missing")
    misses = [k for a, k in zip(answers, keys) if not a["connected"] or not a["on"]["connected"]]
    conn = [k for a, k in zip(answers, keys) if a["connected"] and a["on"]["connected"]]
    return misses, conn


def _truth_index(truth_rows, answers):
    _need_ints(truth_rows, "truth", "ordinal")
    idx = {}
    for m in truth_rows:
        for i, p in enumerate(m["pairs"]):
            idx[(m["ordinal"], i)] = p
    if set(idx) != {(a["ordinal"], a["idx"]) for a in answers}:
        raise AuditError("answers and truth disagree on the golden pairs")
    return idx


def _row(k, pair):
    row = {"ordinal": k[0], "pairIndex": k[1], **{key: pair[key] for key in ROW_KEYS[2:] if key in pair}}
    leaked = _status_keys(row)
    if leaked:
        raise AuditError(f"list row carries a status-like key: {sorted(leaked)}")
    return row


def _status_keys(o):
    found = set()
    if isinstance(o, dict):
        for k, v in o.items():
            if k in STATUS_WORDS:
                found.add(k)
            found |= _status_keys(v)
    elif isinstance(o, list):
        for v in o:
            found |= _status_keys(v)
    return found


def build_list(answers, truth_rows):
    misses, conn = _classify(answers)
    truth = _truth_index(truth_rows, answers)
    n = len(misses)
    want = LIST_LEN - n if n <= CONN_CAP else CONN_CAP
    chosen = sorted(conn, key=lambda k: H(f"{ADJ_SEED}|conn|{k[0]}|{k[1]}"))[:max(want, 0)]
    keys = sorted(misses + chosen, key=lambda k: H(f"{ADJ_SEED}|shuf|{k[0]}|{k[1]}"))
    return [_row(k, truth[k]) for k in keys]


def verify_list(rows, answers, truth_rows):
    """Abort unless `rows` is exactly the list `build_list` derives (every miss listed, no duplicates, length, composition, order, whitelist)."""
    _need_ints(rows, "list", "ordinal", "pairIndex")
    keys = [(r.get("ordinal"), r.get("pairIndex")) for r in rows]
    if len(set(keys)) != len(keys):
        raise AuditError("list: duplicates")
    if any(set(r) - set(ROW_KEYS) for r in rows):
        raise AuditError("list: a row carries a key outside the whitelist")
    misses, _ = _classify(answers)
    absent = set(misses) - set(keys)
    if absent:
        raise AuditError(f"list: {len(absent)} missed pair(s) are not listed")
    if rows != build_list(answers, truth_rows):
        raise AuditError("list: length, composition, seed or order differs from the rule")
    return True


def list_bytes(rows):
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


# ------------------------------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flowaud-p1")
    f.add_argument("--labels", required=True)
    f.add_argument("--extend", action="store_true")
    f.add_argument("--out")
    a = sub.add_parser("adjlist-p1")
    a.add_argument("--answers", required=True)
    a.add_argument("--truth", required=True)
    a.add_argument("--out")
    a.add_argument("--verify")
    ns = ap.parse_args(argv)
    if ns.cmd == "flowaud-p1":
        text = json.dumps(flowaud(_jsonl(ns.labels), ns.extend), indent=1)
        if ns.out:
            open(ns.out, "w").write(text + "\n")
        print(text)
        return 0
    answers, truth = _jsonl(ns.answers), _jsonl(ns.truth)
    if ns.verify:
        verify_list(_jsonl(ns.verify), answers, truth)
        print(hashlib.sha256(open(ns.verify, "rb").read()).hexdigest())
        return 0
    if not ns.out:
        raise AuditError("adjlist-p1: --out or --verify is required")
    rows = build_list(answers, truth)
    data = list_bytes(rows)
    open(ns.out, "wb").write(data)
    print(hashlib.sha256(data).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
