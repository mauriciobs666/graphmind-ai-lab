#!/usr/bin/env python3
"""Compare two `run_nlq_golden_set_eval.py` result files pair by pair (K-066 U3, Q0/Q2).

Prints per-dataset Layer 1 counts for each file, and every pair whose `layer1Correct`
differs, with the candidate's raw `toolCompletion` and `toolResult` for any regression
(baseline right, candidate wrong). Stdlib only; reads JSON, writes nothing.

    python3 nlq_compare.py BASELINE.json CANDIDATE.json [--dataset catalog] [--json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter


def _load(path: str) -> tuple[str, dict[str, dict]]:
    d = json.load(open(path, encoding="utf-8"))
    return d.get("model", "?"), {r["id"]: r for r in d["records"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--dataset", default=None, help="restrict the diff to one dataset (e.g. catalog)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    bm, b = _load(a.baseline)
    cm, c = _load(a.candidate)

    def tally(recs: dict[str, dict]) -> dict:
        t: Counter = Counter()
        for r in recs.values():
            t[(r["dataset"], "n")] += 1
            t[(r["dataset"], "ok")] += bool(r["layer1Correct"])
        return {f"{k[0]}.{k[1]}": v for k, v in sorted(t.items())}

    diffs = []
    for pid in sorted(set(b) | set(c)):
        rb, rc = b.get(pid), c.get(pid)
        ds = (rb or rc)["dataset"]
        if a.dataset and ds != a.dataset:
            continue
        ob = None if rb is None else bool(rb["layer1Correct"])
        oc = None if rc is None else bool(rc["layer1Correct"])
        if ob != oc:
            kind = "regression" if ob and not oc else ("improvement" if oc and not ob else "missing")
            e = {"id": pid, "dataset": ds, "shape": (rb or rc)["shape"], "kind": kind,
                 "baseline": ob, "candidate": oc, "question": (rb or rc)["question"]}
            if rc is not None:
                e.update(expected=rc.get("expected"), candidateToolCompletion=rc.get("toolCompletion"),
                         candidateToolResult=rc.get("toolResult"), candidateReason=rc.get("layer1Reason"))
            diffs.append(e)
    out = {"baseline": {"path": a.baseline, "model": bm, "layer1": tally(b)},
           "candidate": {"path": a.candidate, "model": cm, "layer1": tally(c)}, "diffs": diffs}
    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        for side in ("baseline", "candidate"):
            print(f"{side}: {out[side]['model']}  {out[side]['layer1']}")
        print(f"diffs ({len(diffs)}):")
        for e in diffs:
            print(f"  {e['id']} [{e['dataset']}/{e['shape']}] {e['kind']}: baseline={e['baseline']} candidate={e['candidate']}")
            if e["kind"] == "regression":
                print(f"    question: {e['question']}\n    expected: {e.get('expected')}\n"
                      f"    reason:   {e.get('candidateReason')}\n"
                      f"    toolCompletion: {e.get('candidateToolCompletion')}\n    toolResult: {e.get('candidateToolResult')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
