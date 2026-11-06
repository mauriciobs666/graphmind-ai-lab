#!/usr/bin/env python3
"""s47_scripted_steps.py — replay record of the S47 scripted steps.

UNTESTED. This is the code that was run for S47 (eval-protocol §7.4 correction, §10.3 ambiguity join,
§10.2 AC-24 split), saved so the run can be replayed. It is to be replaced by `split_manifest.py` verbs
(`apply-corrections`, `ambiguity`, the AC-24 half of `split`) in P1. Paths and the eval commit are
arguments; the script holds no keys, item ids or question text.

    s47_scripted_steps.py --work WORK --eval-commit SHA --heldout-dir D --dev-dir D \\
        [--match-scoped WORK/s45/match-scoped.json] [--out-match WORK/s45/match-corrected.json] \\
        [--corrections WORK/corrections.jsonl] [--log WORK/s45/ac24-log.json]

Inputs under WORK: s45/match-scoped.json, s46-marks-template.jsonl (n -> itemId), s46-verify.jsonl
(n -> verdict), questions/{QW,AN1,AN2}.jsonl, aid-map.jsonl. Outputs: the corrected match file, the
corrections log, `ac24.jsonl` in the held-out dir and `ac24.jsonl` in the dev dir. Prints counts only.
"""
import argparse
import collections
import copy
import hashlib
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent.parent))
import match_keys as M  # noqa: E402


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def dump_jsonl(p, rows):
    Path(p).write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                               for x in rows), encoding="utf-8")


def corrections(a, work):
    """EP §7.4: an incorrect union item whose note gives no locator is removed. Logged."""
    tmpl = {x["n"]: x for x in jl(work / "s46-marks-template.jsonl")}
    bad = [v["n"] for v in jl(work / "s46-verify.jsonl") if v["verdict"] == "incorrect" and tmpl[v["n"]]["kind"] == "union"]
    m = json.load(open(a.match_scoped, encoding="utf-8"))
    before = len(m["records"])
    log = []
    for n in bad:
        iid = tmpl[n]["itemId"]
        m["records"] = [r for r in m["records"] if r["itemId"] != iid]
        log.append({"n": n, "itemId": iid, "action": "removed", "reason": "EP7.4: incorrect, note gives no locator",
                    "verifier": "analyst (AI)"})
    Path(a.out_match).write_text(json.dumps(m, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
                                 encoding="utf-8")
    Path(a.corrections).write_text("".join(json.dumps(x) + "\n" for x in log), encoding="utf-8")
    print(f"corrections: {len(log)}\nrecords: {before} -> {len(m['records'])}")


def ac24(a, work):
    """EP §10.3 join of AN1/AN2 with QW's intent, then §10.2 split by rank."""
    qw = jl(work / "questions" / "QW.jsonl")
    amap = {r["aid"]: r["qid"] for r in jl(work / "aid-map.jsonl")}
    lab = {}
    for w in ("AN1", "AN2"):
        for r in jl(work / "questions" / f"{w}.jsonl"):
            lab.setdefault(r["aid"], {})[w] = r["label"]
    byq = {amap[k]: v for k, v in lab.items()}
    out, lang_out, agree, final = collections.Counter(), collections.defaultdict(collections.Counter), 0, []
    for q in qw:
        a1, a2 = byq[q["qid"]]["AN1"], byq[q["qid"]]["AN2"]
        agree += a1 == a2
        if a1 != a2:
            res = "drop-disagree"
        elif a1 == "ambiguous" and q["intent"] == "ambiguous":
            res = "ambiguity"
        elif a1 == "unambiguous" and q["intent"] == "unambiguous":
            res = "unambiguous"
        elif a1 == "ambiguous":
            res = "drop-writer-unamb-ann-amb"
        else:
            res = "drop-writer-amb-ann-unamb"
        out[res] += 1
        lang_out[q["lang"]][res] += 1
        if res in ("ambiguity", "unambiguous"):
            final.append((q, res))
    n = len(qw)
    p1 = sum(byq[q["qid"]]["AN1"] == "ambiguous" for q in qw) / n
    p2 = sum(byq[q["qid"]]["AN2"] == "ambiguous" for q in qw) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    print(f"agreement: {agree}/{n}\noutcomes: {dict(out)}\nkappa: {round((agree / n - pe) / (1 - pe), 4) if pe < 1 else None}")

    rw = collections.Counter()

    def normgold(g, q):
        g = copy.deepcopy(g)
        for an in g["anchors"]:
            if an["type"] == "contract":
                nn = M.normalize_contract(dict(an), "provider")
                if nn != an:
                    rw[(q["lang"], q["group"])] += 1
                an.clear()
                an.update(nn)
        return g

    recs = []
    for q, res in final:
        r = {"qid": q["qid"], "lang": q["lang"], "group": q["group"], "text": q["text"], "outcome": res,
             "original": {"gold": q["gold"], "goldOptions": q["goldOptions"], "ambiguityKind": q["ambiguityKind"]}}
        if res == "unambiguous":
            r["gold"] = normgold(q["gold"], q)
            r["goldForm"] = "provider"
        else:
            r["ambiguityKind"] = q["ambiguityKind"]
            r["goldOptions"] = [normgold(o, q) for o in q["goldOptions"]]
        recs.append(r)
    seed = "ccg-p0f-ac24-v1|" + a.eval_commit
    for r in recs:
        r["rank"] = sha(seed + "|" + r["qid"])
    held, dev = [], []
    cells = collections.defaultdict(list)
    for r in recs:
        if r["outcome"] == "unambiguous":
            cells[(r["group"], r["lang"])].append(r)
    for v in cells.values():
        v.sort(key=lambda r: r["rank"])
        held += v[:6]
        dev += v[6:]
    amb = collections.defaultdict(list)
    for r in recs:
        if r["outcome"] == "ambiguity":
            amb[r["lang"]].append(r)
    for v in amb.values():
        v.sort(key=lambda r: r["rank"])
        dev += v[:1]
        held += v[1:]
    dump_jsonl(Path(a.heldout_dir) / "ac24.jsonl", sorted(held, key=lambda x: x["qid"]))
    dump_jsonl(Path(a.dev_dir) / "ac24.jsonl", sorted(dev, key=lambda x: x["qid"]))
    Path(a.log).write_text(json.dumps({"outcomes": dict(out), "rewrites": {str(k): v for k, v in rw.items()}}),
                           encoding="utf-8")
    print(f"gold-rewrites: {sum(rw.values())}\nheld-out: {len(held)}\ndev: {len(dev)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--eval-commit", required=True)
    ap.add_argument("--heldout-dir", type=Path, required=True)
    ap.add_argument("--dev-dir", type=Path, required=True)
    ap.add_argument("--match-scoped", type=Path)
    ap.add_argument("--out-match", type=Path)
    ap.add_argument("--corrections", type=Path)
    ap.add_argument("--log", type=Path)
    a = ap.parse_args()
    work = a.work
    a.match_scoped = a.match_scoped or work / "s45" / "match-scoped.json"
    a.out_match = a.out_match or work / "s45" / "match-corrected.json"
    a.corrections = a.corrections or work / "corrections.jsonl"
    a.log = a.log or work / "s45" / "ac24-log.json"
    for d in (a.heldout_dir, a.dev_dir):
        d.mkdir(parents=True, exist_ok=True)
    corrections(a, work)
    ac24(a, work)


if __name__ == "__main__":
    main()
