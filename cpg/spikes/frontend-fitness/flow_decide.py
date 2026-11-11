#!/usr/bin/env python3
"""P1-34 flow decision (pre-registration `ccg-p1-flow-python-v1`, §0.1/§5): the pinned script that turns the DECISION cell's scoring output,
the frozen truth file, the adjudication list and the two adjudicators' rulings into y, G, the margin M and the verdict, per configuration.
Read-only: it writes nothing and opens no held-out path.

    cd cpg && .venv/bin/python spikes/frontend-fitness/flow_decide.py --record cpg/eval/preregistration.jsonl \\
        --results results/p1/fixed-new/projection-python-p1.json --answers results/p1/fixed-new/projection-python-p1.answers.jsonl \\
        --truth truth/python.flow-p1-final.jsonl --list <adjudication list> \\
        --rulings-ds adjudication/python-flow-p1-rulings-ds.jsonl --rulings-analyst adjudication/python-flow-p1-rulings-analyst.jsonl

D = the listed pairs ruled `overturned` (not a real pair) by BOTH adjudicators; a pair ruled so by one only stays.  Exactly D is removed: a
connected pair in D leaves both y and G (y-1, G-1, M-1); a missed pair in D leaves G only (G-1, M+19).  A configuration passes iff G >= 120,
20*y >= 19*G and 20*k >= 19*n (projection success k/n); M = 20*y - 19*G (the number of invalid connected pairs the verdict tolerates).
Outcome = the first matching row of the §5 table: 2 (G < 120 in either configuration), 3 (20k < 19n in either), 4 (20y < 19G in either), 5 (TRUE).
The verdict with D empty is printed as a diagnostic, never decisive.  Row 1 (a pinned sha/digest differs) is `s28_metrics.check_record`: refused,
nothing printed; the transcript check, `flowcheck` and the Part F completion are not observable here.  The list must contain every pair missed
in either configuration, and each rulings file must rule exactly the listed pairs, once each.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MIN_PAIRS = 120
RULINGS = {"miss-stands", "overturned", "endpoint-naming-error", "valid"}   # P0 vocabulary plus `valid` for a listed connected pair; only `overturned` is not-real


class DecisionError(SystemExit):
    pass


def _cfg(g, y, k, n):
    m = 20 * y - 19 * g
    return {"G": g, "y": y, "20y": 20 * y, "19G": 19 * g, "M": m, "pairsOK": g >= MIN_PAIRS, "recallOK": 20 * y >= 19 * g,
            "projectionSuccess": {"k": k, "n": n, "20k": 20 * k, "19n": 19 * n}, "preconditionOK": n > 0 and 20 * k >= 19 * n}


def _verdict(answers, removed, ps):
    keep = [a for a in answers if (a["ordinal"], a["idx"]) not in removed]
    g = len(keep)
    cfg = {"off": _cfg(g, sum(1 for a in keep if a["connected"]), *ps["off"]),
           "on": _cfg(g, sum(1 for a in keep if a["on"]["connected"]), *ps["on"])}
    v = list(cfg.values())
    if any(not c["pairsOK"] for c in v):
        row, verdict = 2, "FALSE:insufficient-sample"
    elif any(not c["preconditionOK"] for c in v):
        row, verdict = 3, "FALSE:precondition"
    elif any(not c["recallOK"] for c in v):
        row, verdict = 4, "FALSE:measured"
    else:
        row, verdict = 5, "TRUE"
    return {"perConfig": cfg, "row": row, "verdict": verdict,
            "discordantPairs": sorted(f"{a['ordinal']}|{a['idx']}" for a in keep if a["connected"] != a["on"]["connected"])}


def _rulings(rows, listed, label):
    ruled = {}
    for r in rows:
        k = (r["ordinal"], r["pairIndex"])
        if r.get("adjudication") not in RULINGS:
            raise DecisionError(f"{label}: unknown adjudication on {k[0]}|{k[1]}")
        if k in ruled:
            raise DecisionError(f"{label}: two rulings for {k[0]}|{k[1]}")
        ruled[k] = r["adjudication"]
    if set(ruled) != listed:
        raise DecisionError(f"{label}: must rule exactly the listed pairs ({len(set(ruled) ^ listed)} differ)")
    return {k for k, v in ruled.items() if v == "overturned"}


def decide(answers, listed, rulings_ds, rulings_analyst, ps):
    """answers: the decision cell's per-pair rows ({ordinal, idx, connected, on: {connected}}); listed: [(ordinal, pairIndex)] of the adjudication
    list; rulings_*: one adjudicator's rows ({ordinal, pairIndex, adjudication}); ps: {"off": (k, n), "on": (k, n)} -> the verdict dict."""
    keys = [(a["ordinal"], a["idx"]) for a in answers]
    if len(set(keys)) != len(keys):
        raise DecisionError("answers: a duplicate (ordinal, idx)")
    if any(a.get("on") is None for a in answers):
        raise DecisionError("answers: the `on` configuration is missing")
    lst = set(listed)
    if len(lst) != len(listed) or not lst <= set(keys):
        raise DecisionError("list: duplicates, or a pair that is not in the answers")
    misses = {(a["ordinal"], a["idx"]) for a in answers if not a["connected"] or not a["on"]["connected"]}
    if misses - lst:
        raise DecisionError(f"list: {len(misses - lst)} missed pair(s) are not listed")
    d = _rulings(rulings_ds, lst, "rulings-ds") & _rulings(rulings_analyst, lst, "rulings-analyst")
    main_ = _verdict(answers, d, ps)
    return {"outcome": main_["verdict"], "D": len(d), "listed": len(lst), "misses": len(misses), "decision": main_,
            "diagnosticDEmpty": _verdict(answers, set(), ps),
            "rule": "per config: G >= 120, 20y >= 19G, 20k >= 19n; M = 20y - 19G; TRUE iff both configs pass; D removed"}


def _jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--list", required=True, dest="listing")
    ap.add_argument("--rulings-ds", required=True)
    ap.add_argument("--rulings-analyst", required=True)
    ap.add_argument("--run-record", help="the s28-run.json the decision cell was scored from (default: the standard one); its CPG digests are pinned by --record")
    ap.add_argument("--lang", default="python")
    a = ap.parse_args(argv)
    import yaml
    import s28_metrics as sm   # the same pins the scorer is gated by (this script's own sha included)
    commit = yaml.safe_load(open(sm.HERE / "corpora.yaml"))["corpora"][a.lang]["commit"]
    runs = json.loads((sm.resolve_path(a.run_record) if a.run_record else sm.WORK / a.lang / "s28-run.json").read_text())["runs"]
    sm.check_record(sm.load_record(a.record), sm.load_record(a.record, sm.RECORD_ID + "-draw"), a.lang, commit, sm.flow_paths(a.lang, "p1", sm.DECISION_CELL)["sample"], runs)
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    cell = res.get("cell") or {}
    if res.get("flowSet") != "p1" or cell.get("name") != sm.DECISION_CELL or cell.get("role") != "decision":
        raise DecisionError("--results is not the decision cell (flowSet p1, cell fixed-new)")
    answers = _jsonl(a.answers)
    truth_pairs = sum(len(r["pairs"]) for r in _jsonl(a.truth))
    ps = {c: (res["cfg"][c]["projectionSuccess"]["k"], res["cfg"][c]["projectionSuccess"]["n"]) for c in ("off", "on")}
    if truth_pairs != len(answers) or any(res["cfg"][c]["pooled"]["n"] != len(answers) for c in ("off", "on")):
        raise DecisionError("answers, truth and results disagree on the number of golden pairs")
    if res["cfg"]["off"]["pooled"]["k"] != sum(1 for x in answers if x["connected"]) or res["cfg"]["on"]["pooled"]["k"] != sum(1 for x in answers if x["on"]["connected"]):
        raise DecisionError("answers and results disagree on the connected pairs")
    listed = [(r["ordinal"], r["pairIndex"]) for r in _jsonl(a.listing)]
    out = decide(answers, listed, _jsonl(a.rulings_ds), _jsonl(a.rulings_analyst), ps)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
