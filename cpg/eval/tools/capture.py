#!/usr/bin/env python3
"""capture.py — capture-recapture estimates per derived stratum (S49 second half, P0-42b).

Spec: eval-protocol ``docs/plans/connected-code-graph-eval-protocol-ml.md`` §6.4-§6.6, §13.5; the
oracle is its Appendix A (formulas below are the Appendix's, same operation order, no rounding).

    capture.py --match work/match.json --out work/capture.json
               [--nearmiss F] [--adjudication F] [--invalid F]

Input: ``match.json`` from ``match_keys.py match`` (``records[].history`` is a bit string over
(KA, KA2, KB); ``derivedStratum``; ``inScope`` (boolean, required: the SCOPED file): an out-of-scope item is not
part of the census and is left out). Per stratum: list P = items found by KA or KA2, list C = items
found by KB; t is the per-stratum ``t`` of ``match.json`` (3 iff KA2 ran there; EP §6.4, §13.5).

Output is counts and estimates only, no locators. Exit: 0 ok, 2 usage, 4 missing/invalid input.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import exp, log, sqrt
from pathlib import Path

STRATA = ("http", "mcp", "scripts", "wiring", "flow")
SESSIONS = ("KA", "KA2", "KB")          # history bit order, as match_keys.SESSION_ORDER
Z = 1.959963984540054                   # two-sided 95%, Appendix A
TRIGGER_NUM, TRIGGER_DEN = 95, 100      # EP §6.5: completeness below 0.95; integer test as Appendix A
INFORMATIVE_MIN_R = 8                   # EP §6.5 "tiny strata": r < 8 is reported, not informative


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def chapman(n1, n2, m):
    N = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    return N, var


def chao_mh(r, f1, f2, t):
    return r + ((t - 1) / t) * f1 * (f1 - 1) / (2 * (f2 + 1))


def lognormal_ci(r, N, var):
    """-> (lo, hi, degenerate). ``f0 <= 0`` or ``Var = 0``: ``[r, r]``, flagged."""
    f0 = N - r
    if f0 <= 0 or var <= 0:
        return r, r, True
    C = exp(Z * sqrt(log(1 + var / f0 ** 2)))
    return r + f0 / C, r + f0 * C, False


def history_counts(hist: dict[str, int]) -> dict:
    """``hist``: bit-string (KA, KA2, KB) -> number of records. EP §6.4."""
    r = sum(hist.values())
    n1 = sum(v for k, v in hist.items() if k[0] == "1" or k[1] == "1")
    n2 = sum(v for k, v in hist.items() if k[2] == "1")
    m = sum(v for k, v in hist.items() if (k[0] == "1" or k[1] == "1") and k[2] == "1")
    f = {1: 0, 2: 0, 3: 0}
    for k, v in hist.items():
        f[k.count("1")] = f.get(k.count("1"), 0) + v
    per = {s: sum(v for k, v in hist.items() if k[i] == "1") for i, s in enumerate(SESSIONS)}
    return {"r": r, "n1": n1, "n2": n2, "m": m, "f1": f[1], "f2": f[2], "f3": f[3], "perSession": per}


def estimate(hist: dict[str, int], t: int) -> dict:
    c = history_counts(hist)
    r, n1, n2, m = c["r"], c["n1"], c["n2"], c["m"]
    out = dict(c, t=t, informative=r >= INFORMATIVE_MIN_R)
    if r == 0:
        out.update(nChapman=0.0, var=0.0, chapmanInterval=None, nChao=0.0, nStar=0.0, binds=None,
                   chaoBinds=False, completeness=None, trigger=True)
        return out
    Nc, var = chapman(n1, n2, m)
    Nh = chao_mh(r, c["f1"], c["f2"], t)
    lo, hi, degenerate = lognormal_ci(r, Nc, var)
    Ns = max(Nc, Nh)
    chao_binds = Nh > Nc
    out.update(
        nChapman=Nc, var=var,
        chapmanInterval={"lo": lo, "hi": hi, "degenerate": degenerate},
        nChao=Nh, nStar=Ns, binds="chao" if chao_binds else "chapman", chaoBinds=chao_binds,
        completeness=r / Ns,
        trigger=r * TRIGGER_DEN < TRIGGER_NUM * Ns,
    )
    return out


def build(match: dict, extras: dict | None = None) -> dict:
    """``match``: the SCOPED match.json. ``t`` per stratum comes from its ``t`` map (EP §6.4, §13.5), never from
    a flag. A record without a boolean ``inScope`` is refused; ``inScope: false`` records are outside the census."""
    tmap = match.get("t")
    if not isinstance(tmap, dict) or set(tmap) != set(STRATA) or any(v not in (2, 3) for v in tmap.values()):
        raise ValueError("match.json has no valid per-stratum 't' map (EP §13.5)")
    out_of_scope = 0
    by: dict[str, dict[str, int]] = {s: {} for s in STRATA}
    hops: dict[str, dict] = {s: {} for s in STRATA}
    for g in match["records"]:
        if not isinstance(g.get("inScope"), bool):
            raise ValueError("record without a boolean 'inScope': run match_keys scope first (EP §6.4)")
        if g["inScope"] is False:
            out_of_scope += 1
            continue
        s = g["derivedStratum"]
        if s not in by:
            raise ValueError(f"unknown derived stratum {s!r}")
        h = g["history"]
        if len(h) != len(SESSIONS) or set(h) - {"0", "1"}:
            raise ValueError(f"bad history {h!r}")
        by[s][h] = by[s].get(h, 0) + 1
        for mem in g.get("members") or []:
            k = str(mem.get("hops"))
            hops[s][k] = hops[s].get(k, 0) + 1
    strata = {}
    for s in STRATA:
        if tmap[s] == 2 and any(k[1] == "1" for k in by[s]):
            raise ValueError(f"t = 2 for {s} but a record carries a KA2 bit: the 't' map understates the sessions")
        strata[s] = dict(estimate(by[s], tmap[s]), hops=hops[s])
    sr = sum(v["r"] for v in strata.values())
    sn = sum(v["nStar"] for v in strata.values())
    out = {
        "schema": "ccg-capture-v1",
        "strata": strata,
        "pooled": {"r": sr, "nStar": sn, "completeness": (sr / sn) if sn else None,
                   "note": "sum r / sum N-star, stratified; no pooled interval (EP §6.4)"},
        "outOfScope": out_of_scope,
        "unclassified": match.get("unclassified"),
        "stratumDisagreements": match.get("stratumDisagreements"),
    }
    out.update(extras or {})
    return out


def _count_lines(p) -> int:
    return sum(1 for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nearmiss", type=Path)
    ap.add_argument("--adjudication", type=Path)
    ap.add_argument("--invalid", type=Path)
    a = ap.parse_args(argv)
    try:
        match = json.loads(a.match.read_text(encoding="utf-8"))
        extras = {}
        if a.nearmiss:
            extras["nearmissPairs"] = _count_lines(a.nearmiss)
        if a.adjudication:
            rows = [json.loads(l) for l in a.adjudication.read_text(encoding="utf-8").splitlines() if l.strip()]
            extras["adjudication"] = {v: sum(1 for r in rows if r.get("verdict") == v) for v in ("same", "different")}
        if a.invalid:
            extras["invalidItems"] = _count_lines(a.invalid)
        res = build(match, extras)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"capture: {e}", file=sys.stderr)
        return 4
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(canon(res) + "\n", encoding="utf-8")
    for s in STRATA:
        v = res["strata"][s]
        c = "n/a" if v["completeness"] is None else f"{v['completeness']:.3f}"
        print(f"{s}: r={v['r']} n1={v['n1']} n2={v['n2']} m={v['m']} t={v['t']} completeness={c} trigger={str(v['trigger']).lower()}")
    p = res["pooled"]
    print(f"pooled: r={p['r']} completeness={'n/a' if p['completeness'] is None else format(p['completeness'], '.3f')}")
    print(f"out: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
