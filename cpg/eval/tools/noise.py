#!/usr/bin/env python3
"""noise.py - noise frame, sample, labels and estimates (P1-36; EP §11, ``docs/plans/connected-code-graph-p1-ml.md`` §5, §6.3-6.4).

    noise.py draw  --answers A.jsonl --exclusions X.json --gate-id G --snapshot-id S --custodian ROOT

``A.jsonl`` is one answer per line, ``{"anchorId", "items": [answer-v1 items], "groups": [answer-v1 groups]}``;
``X.json`` is the scope source's ``exclusions.paths`` (§6.3). Nothing here queries a store or a model: the answers
(and, for scoring, the snapshot) are injected. stdout is a counts-only JSON report; the frame, the sample and every
other held-out-derived file go under ``--custodian`` (§6.4), which is refused inside the repo.

Frame (EP §11.1 + §5.1): every listed item and expanded group member; out-of-scope items leave the frame before
allocation (counted per answer and reason, never labelled); confirmed items are a census; possible items are
stratified by (reason, distance band 1 / 2 / >=3). Exit: 0 ok, 2 usage/refused, 4 bad input.

``score`` requires complete labels (EP §11.1-11.2): every census item and every drawn item must be labelled, else exit 4
with counts only. A second-labeller file (``noise-labels2.jsonl``) must carry exactly the keys of ``noise-spot.jsonl``
(EP §11.4: 20 labels drawn from the labelled set); with no overlap the agreement is "not measured", never "stable".
The bootstrap drops a resample whose ``RC + Rhat`` is 0; the percentile indices then scale to the resamples kept
(``int(n*0.025)``, ``int(n*0.975)-1``) and ``used`` reports how many. ``pooledAcrossGrains`` (oracle) is reference only.

With a baseline, ``score`` also reads ``noise-baseline.jsonl`` + ``noise-recall.json`` (and, together, the key-scoped
``noise-baseline-key.jsonl`` + ``noise-recall-key.json``), ``system-hits.jsonl`` and ``--gate-record`` (``defaultDepthBound``);
the in-scope reduction pairs ``countInScope`` with the in-scope burden and the all-listed one ``count`` with the all-listed
burden, never crossed. It exits 4 (counts only) on an anchor-set, count, bound or ``itemSetSha256`` mismatch
(``docs/plans/connected-code-graph-p1-noise-export-graph.md`` §6-§7). Each ``answers`` line carries ``depthBound.bound``.

Choices not fixed by EP §11 (the code's, for the data-scientist to pin):
* Draw hash input: ``sha256(f"{seed}|{anchorId}|{itemId}")`` hex, where EP gives only the seed and ``anchorId|itemId``.
* ``itemId`` is ``"<unitId>:<subjectId>"`` (symbolId / entryId / unitId by subject type), not the bare subject id.
* The spot-check (§11.4) seed is ``"ccg-noise-v1|<gateId>|<snapshotId>|spot"`` (EP gives none), same hash input form.
* Allocation remainder ties break by ``(reason, band)`` order; a capped stratum's unspent share goes to the leftover rule
  (largest unsampled stratum), not to the next remainder.
* Strata with ``N_h <= 5`` are taken whole; a frame at or under the target is sampled whole.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from oracle import wilson  # noqa: E402,F401  (one Wilson implementation for both tools)

SAMPLE_TARGET = 100
MIN_PER_STRATUM = 5


def band(distance: int) -> int:
    """EP §11.1: distance bands 1, 2 and >= 3."""
    return 1 if distance <= 1 else 2 if distance == 2 else 3


def _glob_re(pattern: str) -> re.Pattern:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out + r"\Z")


def path_excluded(path: str, patterns: list[str]) -> bool:
    """A directory pattern (trailing ``/``) is a prefix; any other pattern is a whole-path glob (``**`` crosses ``/``)."""
    for p in patterns:
        if p.endswith("/"):
            if path.startswith(p):
                return True
        elif _glob_re(p).match(path):
            return True
    return False


_SUBJECT_ID = {"symbol": "symbolId", "config": "entryId", "unit": "unitId"}


def item_path(it: dict) -> str | None:
    return it["subject"].get("path")


def item_id(it: dict) -> str:
    s = it["subject"]
    return f"{it['unitId']}:{s[_SUBJECT_ID[s['type']]]}"


def split_class(it: dict) -> str:
    """Own edge status of the item's possible hops (P1-36 addition 4; spec2 §5): a chain whose every possible hop is
    ``flow-unavailable:*`` has the status its edge would have in a flag-true unit, i.e. ``confirmed``."""
    if it["status"] == "confirmed":
        return "confirmed"
    reasons = [h["reason"] for h in it["chain"] if h.get("status") == "possible"] or [it["reason"]]
    return "confirmed-in-flag-true" if all(r.startswith("flow-unavailable:") for r in reasons) else "possible"


def _listed(ans: dict):
    yield from ans["items"]
    for g in ans["groups"]:
        yield from g["members"]


def build_frame(answers: list[dict], exclusions: list[str]) -> dict:
    """The frame (module docstring). ``burden`` lists are per answer in anchorId order, ids never leave the custodian."""
    frame, census, strata, oos, burden_in, burden_all, bad_reason = [], [], {}, {}, [], [], 0
    for ans in sorted(answers, key=lambda a: a["anchorId"]):
        n_in = n_all = 0
        for it in _listed(ans):
            n_all += 1
            path = item_path(it)
            reason = it.get("reason", "confirmed")
            if path is not None and path_excluded(path, exclusions):
                d = oos.setdefault(ans["anchorId"], {})
                d[reason] = d.get(reason, 0) + 1
                continue
            if it["status"] == "possible" and not isinstance(it.get("reason"), str):
                bad_reason += 1
                continue
            n_in += 1
            row = {"anchorId": ans["anchorId"], "itemId": item_id(it), "status": it["status"],
                   "reason": it.get("reason"), "band": band(it["distance"]), "distance": it["distance"],
                   "split": split_class(it)}
            frame.append(row)
            if it["status"] == "confirmed":
                census.append(row)
            else:
                key = (row["reason"], row["band"])
                strata[key] = strata.get(key, 0) + 1
        burden_in.append(n_in)
        burden_all.append(n_all)
    if bad_reason:
        raise ValueError(f"{bad_reason} possible item(s) in the frame without a string reason")
    dup = len(frame) - len({(r["anchorId"], r["itemId"]) for r in frame})
    if dup:
        raise ValueError(f"{dup} duplicate (anchorId, itemId) row(s) in the frame")
    return {"frame": frame, "census": census, "strata": strata, "outOfScope": oos,
            "burden": {"inScope": burden_in, "allListed": burden_all}}


def allocate(sizes: dict, target: int = SAMPLE_TARGET) -> dict:
    """EP §11.2: ``min(N_h, 5)`` per stratum, then the rest of ``target`` by largest remainder in exact integer
    arithmetic capped at ``N_h``; capped leftovers go to the largest unsampled stratum. Minima past the target are the sample."""
    keys = sorted(sizes)
    alloc = {k: min(sizes[k], MIN_PER_STRATUM) for k in keys}
    rest = target - sum(alloc.values())
    total = sum(sizes.values())
    if rest <= 0:
        return alloc
    if total <= target:
        return dict(sizes)
    quota = {k: rest * sizes[k] // total for k in keys}
    rem = {k: rest * sizes[k] % total for k in keys}
    for k in keys:
        alloc[k] = min(sizes[k], alloc[k] + quota[k])
    left = target - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-rem[k], k))[: max(0, rest - sum(quota.values()))]:
        if alloc[k] < sizes[k] and left > 0:
            alloc[k] += 1
            left -= 1
    while left > 0:
        k = min(keys, key=lambda k: (alloc[k] - sizes[k], k))
        if sizes[k] == alloc[k]:
            break
        alloc[k] += 1
        left -= 1
    return alloc


def draw(frame: list[dict], alloc: dict, gate_id: str, snapshot_id: str) -> list[dict]:
    """Within each stratum, hash sort on ``anchorId|itemId`` under ``seed = "ccg-noise-v1|" + gateId + "|" + snapshotId``."""
    seed = "ccg-noise-v1|" + gate_id + "|" + snapshot_id
    out = []
    for k in sorted(alloc):
        cands = [r for r in frame if r["status"] == "possible" and (r["reason"], r["band"]) == k]
        cands.sort(key=lambda r: hashlib.sha256(f"{seed}|{r['anchorId']}|{r['itemId']}".encode()).hexdigest())
        out.extend(cands[: alloc[k]])
    return out


def _sums(rows: list[dict], labels: dict, where=None) -> tuple[Fraction, Fraction]:
    """``(What, Rhat)`` = sums over strata of ``N_h f_h / n_h`` and ``N_h t_h / n_h`` (EP §11.3). ``where`` restricts the
    frame rows counted into ``N_h``; the rate comes from the labelled items that pass it, and a stratum with none of
    them borrows its full-stratum rate. A stratum with no label at all is an error."""
    by_h: dict = {}
    for r in rows:
        if r["status"] == "possible":
            by_h.setdefault((r["reason"], r["band"]), []).append(r)
    w, rr = Fraction(0), Fraction(0)
    for h, rs in sorted(by_h.items()):
        lab = [labels[(r["anchorId"], r["itemId"])] for r in rs if (r["anchorId"], r["itemId"]) in labels]
        if not lab:
            raise ValueError(f"stratum {h}: no labelled item")
        sel = [r for r in rs if where is None or where(r)]
        sub = [labels[(r["anchorId"], r["itemId"])] for r in sel if (r["anchorId"], r["itemId"]) in labels] or lab
        w += Fraction(len(sel) * sub.count("false"), len(sub))
        rr += Fraction(len(sel) * sub.count("real"), len(sub))
    return w, rr


def _q(w: Fraction, r: Fraction, rc: int) -> Fraction | None:
    return w / (rc + r) if rc + r else None


def estimate(rows: list[dict], labels: dict) -> dict:
    """EP §11.3 point estimates over the in-scope frame ``rows`` and ``labels`` ``{(anchorId, itemId): "real"|"false"}``."""
    conf = [labels[(r["anchorId"], r["itemId"])] for r in rows if r["status"] == "confirmed" and (r["anchorId"], r["itemId"]) in labels]
    rc = conf.count("real")
    rc_b = {b: sum(1 for r in rows if r["status"] == "confirmed" and r["band"] == b and labels.get((r["anchorId"], r["itemId"])) == "real")
            for b in (1, 2, 3)}
    w, rhat = _sums(rows, labels)
    out = {"What": w, "Rhat": rhat, "RC": rc, "Q": _q(w, rhat, rc),
           "possiblePrecision": rhat / (rhat + w) if rhat + w else None,
           "confirmedPrecision": wilson(rc, len(conf)), "falseConfirmed": conf.count("false"), "byBand": {}, "byReason": {},
           "bySplit": {}}
    for b in (1, 2, 3):
        if any(r["status"] == "possible" and r["band"] == b for r in rows):
            wb, rb = _sums(rows, labels, lambda r, b=b: r["band"] == b)
            out["byBand"][b] = {"What": wb, "Rhat": rb, "RC": rc_b[b], "Q": _q(wb, rb, rc_b[b])}
    lab_poss = [(r, labels[(r["anchorId"], r["itemId"])]) for r in rows
                if r["status"] == "possible" and (r["anchorId"], r["itemId"]) in labels]
    for reason in sorted({r["reason"] for r, _ in lab_poss}):
        ls = [l for r, l in lab_poss if r["reason"] == reason]
        out["byReason"][reason] = {**wilson(ls.count("real"), len(ls)), "descriptive": len(ls) < 20}
    out["reasonsUnder005"] = [k for k, v in out["byReason"].items() if v["s"] is not None and v["s"] < 0.05]
    for sp in sorted({r["split"] for r in rows if r["status"] == "possible"}):
        ws, rs_ = _sums(rows, labels, lambda r, sp=sp: r["split"] == sp)
        out["bySplit"][sp] = {"What": ws, "Rhat": rs_,
                              "possiblePrecision": rs_ / (rs_ + ws) if rs_ + ws else None}
    return out


def p90(values: list[int]) -> int:
    """EP §11.3: the exact ceiling rank, ``-(-9 * n // 10)`` (1-based)."""
    v = sorted(values)
    return v[-(-9 * len(v) // 10) - 1]


def median(values: list[int]) -> Fraction:
    v = sorted(values)
    n = len(v)
    return Fraction(v[n // 2]) if n % 2 else Fraction(v[n // 2 - 1] + v[n // 2], 2)


def reduction(baseline: list[int], system: list[int], system_recall: tuple[int, int], baseline_recall: tuple[int, int]) -> dict:
    """``sum(baseline burden) / sum(system burden)``; flagged when the baseline's recall (k, n) is lower than the system's."""
    ss = sum(system)
    flagged = baseline_recall[0] * system_recall[1] < system_recall[0] * baseline_recall[1]
    higher = baseline_recall[0] * system_recall[1] > system_recall[0] * baseline_recall[1]
    return {"reduction": Fraction(sum(baseline), ss) if ss else None, "flagged": flagged, "baselineRecallHigher": higher}


def thresholds(q, w, r, rc, p90_in, reduction) -> dict:
    """The ratified bars (``-p1-ml.md`` §5.3), judged exactly; only Q <= 4 is gated at gate 1, the others gate from the next."""
    return {
        "Q": {"limit": 4, "met": q is not None and q <= 4, "gated": True},
        "possiblePrecision": {"limit": "0.20", "met": r + w > 0 and 4 * r >= w, "gated": False},
        "p90InScope": {"limit": 100, "met": p90_in <= 100, "gated": False},
        "reduction": {"limit": 5, "met": reduction is not None and reduction >= 5, "gated": False},
    }


def agreement(first: dict, second: dict) -> dict:
    """Raw agreement and Cohen's kappa over the items both labellers labelled; below 0.8 raw the labels are unstable."""
    keys = sorted(set(first) & set(second))
    n = len(keys)
    agree = sum(first[k] == second[k] for k in keys)
    out = {"n": n, "agree": agree, "rawAgreement": round(agree / n, 3) if n else None,
           "labelsUnstable": agree * 5 < 4 * n if n else None, "kappa": None, "wilson": wilson(agree, n)}
    if not n:
        out["note"] = "not measured"
    else:
        pe = sum(Fraction(sum(first[k] == c for k in keys) * sum(second[k] == c for k in keys), n * n) for c in ("real", "false"))
        if pe != 1:
            out["kappa"] = round(float((Fraction(agree, n) - pe) / (1 - pe)), 3)
    return out


MIN_POSSIBLE_SPOT = 10


def split_agreement(first: dict, second: dict, status: dict) -> dict:
    """Raw agreement (with its Wilson interval) over the shared keys, split by the item's frame ``status``
    (``{(anchorId, itemId): "confirmed"|"possible"}``): the spot draw is mostly confirmed items, the flagged figure is not."""
    out = {}
    for cls in ("confirmed", "possible"):
        keys = [k for k in sorted(set(first) & set(second)) if status.get(k) == cls]
        agree = sum(first[k] == second[k] for k in keys)
        out[cls] = {"n": len(keys), "agree": agree, "rawAgreement": round(agree / len(keys), 3) if keys else None,
                    "wilson": wilson(agree, len(keys))}
    return out


def possible_precision_note(labels_unstable, n_possible_spot: int) -> str | None:
    """With fewer than 10 possible-item spot keys the possible-precision figure has no agreement behind it (m2)."""
    if n_possible_spot < MIN_POSSIBLE_SPOT:
        return "agreement not measured on possible items"
    return "labels unstable" if labels_unstable else None


def percentile_indices(b: int) -> tuple[int, int]:
    """0-based percentile indices: 250 and 9749 at B = 10000."""
    return int(b * 0.025), int(b * 0.975) - 1


def _agg(rows: list[dict], labels: dict, anchors: list[str], baseline: dict, all_burden: dict) -> dict:
    """Per-anchor vectors. The two reductions pair like with like (F3): ``baseline_in`` (``countInScope``) with the
    in-scope ``burden``, ``baseline_all`` (``count``) with ``burden_all``."""
    out = {a: {"strata": {}, "rc": 0, "burden": 0, "burden_all": all_burden.get(a, 0),
               "base_in": baseline.get(a, {}).get("countInScope", 0), "base_all": baseline.get(a, {}).get("count", 0)}
           for a in anchors}
    for r in rows:
        g = out[r["anchorId"]]
        g["burden"] += 1
        lab = labels.get((r["anchorId"], r["itemId"]))
        if r["status"] == "confirmed":
            g["rc"] += lab == "real"
            continue
        c = g["strata"].setdefault((r["reason"], r["band"]), [0, 0, 0, 0])   # N, n, f, t
        c[0] += 1
        if lab is not None:
            c[1] += 1
            c[2] += lab == "false"
            c[3] += lab == "real"
    return out


def bootstrap(rows: list[dict], labels: dict, gate_id: str, B: int = 10000, anchors: list[str] | None = None,
              baseline: dict | None = None, all_burden: dict | None = None) -> dict:
    """Percentile intervals over **answers** (EP §11.3): answers sorted by anchorId, ``random.Random("ccg-noise-boot-v1|" +
    gateId)``, ``B`` resamples with replacement by ``randrange``; a stratum with ``N_h > 0`` and no labelled item in a
    resample uses its full-sample rate. ``baseline`` is ``{anchorId: {count, countInScope, ...}}`` and ``all_burden``
    ``{anchorId: all-listed burden}``. Returns ``{"Q", "reduction", "reductionAll", "p90InScope"}``, each
    ``{lower, upper, used}``: ``reduction`` is ``Σ countInScope / Σ in-scope burden``, ``reductionAll`` ``Σ count / Σ all-listed burden``."""
    anchors = sorted(set(anchors) if anchors is not None else {r["anchorId"] for r in rows})
    if baseline and all_burden is None:
        raise ValueError("a baseline needs the all-listed burden per anchor")
    agg = _agg(rows, labels, anchors, baseline or {}, all_burden or {})
    full: dict = {}
    for g in agg.values():
        for h, c in g["strata"].items():
            t = full.setdefault(h, [0, 0, 0, 0])
            t[:] = [x + y for x, y in zip(t, c)]
    rng = random.Random("ccg-noise-boot-v1|" + gate_id)
    qs, reds, reds_all, p90s = [], [], [], []
    for _ in range(B):
        picks = [agg[anchors[rng.randrange(len(anchors))]] for _ in anchors]
        tot: dict = {}
        for g in picks:
            for h, c in g["strata"].items():
                t = tot.setdefault(h, [0, 0, 0, 0])
                t[:] = [x + y for x, y in zip(t, c)]
        w = rr = Fraction(0)
        for h, (nh, n, f, t) in tot.items():
            if n == 0:
                n, f, t = full[h][1:]
            if n:
                w += Fraction(nh * f, n)
                rr += Fraction(nh * t, n)
        rc = sum(g["rc"] for g in picks)
        if rc + rr:
            qs.append(w / (rc + rr))
        burden = sum(g["burden"] for g in picks)
        if baseline and burden:
            reds.append(Fraction(sum(g["base_in"] for g in picks), burden))
        burden_all = sum(g["burden_all"] for g in picks)
        if baseline and burden_all:
            reds_all.append(Fraction(sum(g["base_all"] for g in picks), burden_all))
        p90s.append(p90([g["burden"] for g in picks]))

    def interval(vals: list) -> dict:
        vals = sorted(vals)
        if not vals:
            return {"lower": None, "upper": None, "used": 0}
        lo, hi = percentile_indices(len(vals))
        return {"lower": vals[lo], "upper": vals[hi], "used": len(vals)}
    return {"Q": interval(qs), "reduction": interval(reds), "reductionAll": interval(reds_all), "p90InScope": interval(p90s)}


def unbound_multi_unit(snapshot: dict) -> dict:
    """Q-NC5 = A: unbound links whose key has providers in more than one unit (a known false-confirmed shape at P1).
    ``snapshot`` is injected: ``{"links": [{keyNorm, bound, status}], "providers": [{keyNorm, unitId}]}``."""
    units: dict = {}
    for p in snapshot["providers"]:
        units.setdefault(p["keyNorm"], set()).add(p["unitId"])
    hit = [l for l in snapshot["links"] if not l["bound"] and len(units.get(l["keyNorm"], ())) > 1]
    return {"unboundMultiUnit": len(hit), "unboundMultiUnitConfirmed": sum(l["status"] == "confirmed" for l in hit)}


def spot_check(keys: list, gate_id: str, snapshot_id: str, k: int = 20) -> list:
    """EP §11.4: 20 labels drawn by hash sort from the labelled set, for the second labeller."""
    seed = "ccg-noise-v1|" + gate_id + "|" + snapshot_id + "|spot"
    return sorted(keys, key=lambda x: hashlib.sha256(f"{seed}|{x[0]}|{x[1]}".encode()).hexdigest())[:k]


def item_set_sha256(rows: list[dict]) -> str:
    """Identity of a scored item set (``docs/plans/connected-code-graph-p1-noise-export-graph.md`` §6): sha256 (UTF-8) of the
    lines ``anchorId|<type>:<repoId>:<path>:<qname or keyPath>|<l1,l2,...>`` (``lines`` ascending), the list sorted in code-point
    order and joined by ``\n`` (no trailing newline). ``rows`` are ``system-hits.jsonl`` rows."""
    lines = []
    for r in rows:
        i = r["impacted"]
        who = i["qname"] if "qname" in i else i["keyPath"]
        lines.append(f"{r['anchorId']}|{i['type']}:{i['repoId']}:{i['path']}:{who}|{','.join(str(n) for n in sorted(r['lines']))}")
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def _f(x, nd: int = 3):
    return None if x is None else round(float(x), nd)


def _ci(i: dict) -> dict:
    return {"lower": _f(i["lower"]), "upper": _f(i["upper"]), "used": i["used"]}


def _dist(vals: list[int]) -> dict:
    return {"median": _f(median(vals)), "p90": p90(vals)}


def _read_labels(path: Path, allowed: set) -> dict:
    out = {}
    for r in _jsonl(str(path)):
        key = (r["anchorId"], r["itemId"])
        if key not in allowed:
            raise ValueError("a label names an item outside the draw")
        if r["label"] not in ("real", "false"):
            raise ValueError("a label is neither real nor false")
        out[key] = r["label"]
    return out


def _require_complete(labels: dict, drawn: list[dict], census: list[dict]) -> None:
    """EP §11.1-11.2: every census item and every drawn item must carry a label. A missing one would drop a real or a
    false item from the estimate and move the gates the favourable way, so it is an error. Counts only in the message."""
    miss_c = sum((r["anchorId"], r["itemId"]) not in labels for r in census)
    miss_d = sum((r["anchorId"], r["itemId"]) not in labels for r in drawn)
    if miss_c or miss_d:
        raise ValueError(f"labels incomplete: {miss_c} of {len(census)} census and {miss_d} of {len(drawn)} drawn items unlabelled")


def _require_spot(labels2: dict, spot_path: Path) -> None:
    """EP §11.4: the second labeller relabels exactly the 20 spot keys ``spot`` wrote, no more and no fewer."""
    if not spot_path.exists():
        raise ValueError("a second-labeller file needs the noise-spot.jsonl list written by `spot`")
    spot = {(r["anchorId"], r["itemId"]) for r in _jsonl(str(spot_path))}
    if set(labels2) != spot:
        raise ValueError(f"second-labeller keys are not the spot list: {len(set(labels2) & spot)} of {len(spot)} spot keys, "
                         f"{len(set(labels2) - spot)} other")


def _check_recall(recall: dict, hits: list[dict], label: str) -> None:
    """Refuse a recall file that cannot be a pair of recalls over the harness's own item list (counts only)."""
    def pairs(d: dict, where: str) -> None:
        for key in ("system", "baseline", "systemSym", "baselineSym"):
            if key in d:
                k, n = d[key]
                if not (isinstance(k, int) and isinstance(n, int) and n > 0 and 0 <= k <= n):
                    raise ValueError(f"{label}: {where}pair {key} violates 0 <= k <= n with n > 0")
    pairs(recall, "")
    if recall.get("itemSetSha256") != item_set_sha256(hits):
        raise ValueError(f"{label}: itemSetSha256 differs from the one of system-hits.jsonl")
    if "exKeyError" in recall:
        ex = recall["exKeyError"]
        pairs(ex, "exKeyError ")
        if ex.get("itemSetSha256") != item_set_sha256([h for h in hits if not h["keyError"]]):
            raise ValueError(f"{label}: exKeyError itemSetSha256 differs from the one of system-hits.jsonl without key errors")


def _check_baseline(baseline: dict, answers: list[dict], depth_bound: int, label: str) -> None:
    ids = {a["anchorId"] for a in answers}
    missing, extra = len(ids - set(baseline)), len(set(baseline) - ids)
    if missing or extra:
        raise ValueError(f"{label}: {missing} answer(s) without a baseline row, {extra} baseline row(s) without an answer")
    rows = list(baseline.values())
    bad = sum(not (0 <= r["countInScope"] <= r["count"]) for r in rows)
    if bad:
        raise ValueError(f"{label}: {bad} baseline row(s) violate 0 <= countInScope <= count")
    bad = sum(bool(r.get("unresolved")) and r["count"] != 0 for r in rows)
    if bad:
        raise ValueError(f"{label}: {bad} unresolved baseline row(s) with count != 0")
    bad = sum(r["bound"] != depth_bound for r in rows)
    if bad:
        raise ValueError(f"{label}: {bad} baseline row(s) with bound != the record's defaultDepthBound {depth_bound}")


def _check_inputs(answers, baseline, recall, system_hits, depth_bound, baseline_key, recall_key) -> None:
    if baseline is None:
        return
    if depth_bound is None:
        raise ValueError("a baseline needs the record's defaultDepthBound (the depth bound)")
    if recall is None or system_hits is None:
        raise ValueError("a baseline needs noise-recall.json and system-hits.jsonl")
    bad = sum(not isinstance(a.get("depthBound"), dict) or a["depthBound"].get("bound") != depth_bound for a in answers)
    if bad:
        raise ValueError(f"{bad} answer(s) with a missing or different depthBound.bound than the record's defaultDepthBound {depth_bound}")
    _check_baseline(baseline, answers, depth_bound, "noise-baseline.jsonl")
    _check_recall(recall, system_hits, "noise-recall.json")
    if (baseline_key is None) != (recall_key is None):
        raise ValueError("key-scoped variant: the baseline and its recall file come together")
    if baseline_key is not None:
        _check_baseline(baseline_key, answers, depth_bound, "key-scoped noise-baseline-key.jsonl")
        _check_recall(recall_key, system_hits, "key-scoped noise-recall-key.json")


HIGHER_NOTE = ("baseline recall higher than the system's: the reduction is a burden ratio, not at equal recall, "
               "and an upper bound")


def _wilson_pairs(d: dict) -> dict:
    return {k: wilson(*d[k]) for k in ("system", "baseline", "systemSym", "baselineSym") if k in d}


def _variant(baseline: dict, recall: dict, answers: list[dict], anchors: list[str], f: dict, boot: dict) -> dict:
    """One baseline variant's counts-only report: reduction (both pairings, each with its interval, and by anchor type),
    the baseline's composition, and the recall figures beside it (``-p1-noise-export-graph.md`` §6-§7)."""
    bin_, ball = f["burden"]["inScope"], f["burden"]["allListed"]
    rows = [baseline[a] for a in anchors]
    r_in = reduction([r["countInScope"] for r in rows], bin_, tuple(recall["system"]), tuple(recall["baseline"]))
    r_all = reduction([r["count"] for r in rows], ball, tuple(recall["system"]), tuple(recall["baseline"]))
    by_type: dict = {}
    for r, n_in, n_all in zip(rows, bin_, ball):
        t = by_type.setdefault(r["type"], {"anchors": 0, "baselineCount": 0, "baselineInScope": 0, "baselineConfirmedOwn": 0,
                                           "systemInScope": 0, "systemAllListed": 0})
        t["anchors"] += 1
        t["baselineCount"] += r["count"]
        t["baselineInScope"] += r["countInScope"]
        t["baselineConfirmedOwn"] += r["countConfirmedOwn"]
        t["systemInScope"] += n_in
        t["systemAllListed"] += n_all
    for t in by_type.values():
        t["reductionInScope"] = _f(Fraction(t["baselineInScope"], t["systemInScope"])) if t["systemInScope"] else None
        t["reductionAllListed"] = _f(Fraction(t["baselineCount"], t["systemAllListed"])) if t["systemAllListed"] else None
    red = {"inScope": _f(r_in["reduction"]), "allListed": _f(r_all["reduction"]), "flagged": r_in["flagged"],
           "baselineRecallHigher": r_in["baselineRecallHigher"], "kind": "burden ratio",
           "interval": _ci(boot["reduction"]), "intervalAllListed": _ci(boot["reductionAll"]), "byType": by_type}
    if r_in["baselineRecallHigher"]:
        red["note"] = HIGHER_NOTE
    sym = {}
    if "systemSym" in recall and "baselineSym" in recall:
        rs = reduction([], [], tuple(recall["systemSym"]), tuple(recall["baselineSym"]))
        sym = {"symbolOnly": {"flagged": rs["flagged"], "baselineRecallHigher": rs["baselineRecallHigher"]}}
    rc = {**_wilson_pairs(recall), "paired": recall.get("paired"), "configGold": recall.get("configGold"),
          "flagged": r_in["flagged"], "baselineRecallHigher": r_in["baselineRecallHigher"], **sym}
    if "exKeyError" in recall:
        rc["exKeyError"] = _wilson_pairs(recall["exKeyError"])
    non_symbol = sum(it["subject"]["type"] != "symbol" for a in answers for it in _listed(a))
    comp = {"anchors": len(rows), "unresolved": sum(bool(r.get("unresolved")) for r in rows),
            "byName": {k: sum(r["byName"][k] for r in rows) for k in ("resolvedSite", "unresolvedSite")},
            "nonSymbolSystemItems": non_symbol}
    return {"reduction": red, "baseline": comp, "recall": rc}


def score(answers: list[dict], exclusions: list[str], labels: dict, gate_id: str, snapshot_id: str, B: int = 10000,
          labels2: dict | None = None, baseline: dict | None = None, recall: dict | None = None,
          snapshot: dict | None = None, *, system_hits: list[dict] | None = None, depth_bound: int | None = None,
          baseline_key: dict | None = None, recall_key: dict | None = None) -> dict:
    """The counts-only report: no anchor or item id appears in it. ``baseline`` is ``{anchorId: row}``, a row being
    ``{anchorId, type, count, countInScope, countConfirmedOwn, byName, unresolved?, bound}`` (``-p1-noise-export-graph.md`` §7)."""
    _check_inputs(answers, baseline, recall, system_hits, depth_bound, baseline_key, recall_key)
    f = build_frame(answers, exclusions)
    anchors = sorted(a["anchorId"] for a in answers)
    est = estimate(f["frame"], labels)
    bin_, ball = f["burden"]["inScope"], f["burden"]["allListed"]
    boot = bootstrap(f["frame"], labels, gate_id, B, anchors, baseline, dict(zip(anchors, ball)) if baseline is not None else None)
    by_reason: dict = {}
    for d in f["outOfScope"].values():
        for k, v in d.items():
            by_reason[k] = by_reason.get(k, 0) + v
    e = {"Q": _f(est["Q"]), "QInterval": _ci(boot["Q"]), "possiblePrecision": _f(est["possiblePrecision"]),
         "What": _f(est["What"]), "Rhat": _f(est["Rhat"]), "RC": est["RC"],
         "confirmedPrecision": est["confirmedPrecision"], "falseConfirmed": est["falseConfirmed"],
         "byBand": {str(b): {"Q": _f(v["Q"]), "RC": v["RC"]} for b, v in est["byBand"].items()},
         "byReason": est["byReason"], "reasonsUnder005": est["reasonsUnder005"],
         "bySplit": {k: {"What": _f(v["What"]), "Rhat": _f(v["Rhat"]), "possiblePrecision": _f(v["possiblePrecision"])}
                     for k, v in est["bySplit"].items()}}
    rep = {"answers": len(answers), "frame": len(f["frame"]), "census": len(f["census"]),
           "outOfScope": {"perAnswer": [sum(f["outOfScope"].get(a, {}).values()) for a in anchors], "byReason": by_reason},
           "burden": {"inScope": {**_dist(bin_), "p90Interval": _ci(boot["p90InScope"])}, "allListed": _dist(ball)},
           "estimates": e, "agreement": None, "labelsUnstable": None}
    if labels2 is not None:
        rep["agreement"] = agreement(labels, labels2)
        rep["labelsUnstable"] = rep["agreement"]["labelsUnstable"]
        status = {(r["anchorId"], r["itemId"]): r["status"] for r in f["frame"]}
        rep["agreement"].update(split_agreement(labels, labels2, status))
        note = possible_precision_note(rep["labelsUnstable"], rep["agreement"]["possible"]["n"])
        if note:
            e["possiblePrecisionNote"] = note
    red = None
    if baseline is not None:
        v = _variant(baseline, recall, answers, anchors, f, boot)
        red = Fraction(sum(baseline[a]["countInScope"] for a in anchors), sum(bin_)) if sum(bin_) else None
        rep.update(v)
        if baseline_key is not None:
            kb = bootstrap(f["frame"], labels, gate_id, B, anchors, baseline_key, dict(zip(anchors, ball)))
            rep["keyScoped"] = _variant(baseline_key, recall_key, answers, anchors, f, kb)
    rep["thresholds"] = thresholds(est["Q"], est["What"], est["Rhat"], est["RC"], p90(bin_), red)
    if snapshot is not None:
        rep["unbound"] = unbound_multi_unit(snapshot)
    return rep


def _jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _custodian(path: str) -> Path | None:
    out = Path(path).expanduser().resolve()
    return None if out == REPO or REPO in out.parents else out


def _dump(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def _read_baseline(path: Path) -> dict:
    rows = _jsonl(str(path))
    out = {r["anchorId"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"{path.name}: {len(rows) - len(out)} duplicate anchorId row(s)")
    return out


def _read_variant(cust: Path, bname: str, rname: str):
    """``(baseline, recall)`` or ``(None, None)`` when neither file exists; one without the other is refused."""
    bp, rp = cust / bname, cust / rname
    if bp.exists() != rp.exists():
        raise ValueError(f"{bname} and {rname} come together: one of them is missing")
    if not bp.exists():
        return None, None
    return _read_baseline(bp), json.loads(rp.read_text(encoding="utf-8"))


def _score_cmd(a, cust: Path, answers, exclusions, f) -> int:
    try:
        alloc = allocate(f["strata"])
        drawn = draw(f["frame"], alloc, a.gate_id, a.snapshot_id)
        allowed = {(r["anchorId"], r["itemId"]) for r in drawn + f["census"]}
        labels = _read_labels(cust / "noise-labels.jsonl", allowed)
        _require_complete(labels, drawn, f["census"])
        l2p = cust / "noise-labels2.jsonl"
        labels2 = _read_labels(l2p, allowed) if l2p.exists() else None
        if labels2 is not None:
            _require_spot(labels2, cust / "noise-spot.jsonl")
        baseline, recall = _read_variant(cust, "noise-baseline.jsonl", "noise-recall.json")
        base_key, recall_key = _read_variant(cust, "noise-baseline-key.jsonl", "noise-recall-key.json")
        hits = depth = None
        if baseline is not None:
            if not a.gate_record:
                raise ValueError("a baseline needs --gate-record (the ac5-gate-1 record with defaultDepthBound)")
            depth = json.loads(Path(a.gate_record).read_text(encoding="utf-8"))["defaultDepthBound"]
            hits = _jsonl(str(cust / "system-hits.jsonl"))
        snap = json.loads(Path(a.snapshot).read_text(encoding="utf-8")) if a.snapshot else None
        rep = score(answers, exclusions, labels, a.gate_id, a.snapshot_id, a.boot, labels2, baseline, recall, snap,
                    system_hits=hits, depth_bound=depth, baseline_key=base_key, recall_key=recall_key)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 4
    print(json.dumps(rep, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    for f in ("--answers", "--exclusions", "--gate-id", "--snapshot-id", "--custodian"):
        d.add_argument(f, required=True)
    sc = sub.add_parser("score")
    for f in ("--answers", "--exclusions", "--gate-id", "--snapshot-id", "--custodian"):
        sc.add_argument(f, required=True)
    sc.add_argument("--boot", type=int, default=10000)
    sp = sub.add_parser("spot")
    for f in ("--gate-id", "--snapshot-id", "--custodian"):
        sp.add_argument(f, required=True)
    sc.add_argument("--snapshot")
    sc.add_argument("--gate-record", help="the ac5-gate-1 record (JSON with defaultDepthBound); required with a baseline")
    a = ap.parse_args(argv)
    cust = _custodian(a.custodian)
    if cust is None:
        print(f"REFUSED: --custodian {a.custodian} is inside the repo; held-out-derived files belong under the custodian root", file=sys.stderr)
        return 2
    if a.cmd == "spot":
        try:
            keys = sorted((r["anchorId"], r["itemId"]) for r in _jsonl(str(cust / "noise-labels.jsonl")))
        except (OSError, ValueError, KeyError) as e:
            print(f"bad input: {e}", file=sys.stderr)
            return 4
        picked = spot_check(keys, a.gate_id, a.snapshot_id)
        _dump(cust / "noise-spot.jsonl", [{"anchorId": x, "itemId": y} for x, y in picked])
        print(json.dumps({"labelled": len(keys), "spot": len(picked)}))
        return 0
    try:
        answers = _jsonl(a.answers)
        exclusions = json.loads(Path(a.exclusions).read_text(encoding="utf-8"))
        f = build_frame(answers, exclusions)
    except (OSError, ValueError, KeyError) as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 4
    if a.cmd == "score":
        return _score_cmd(a, cust, answers, exclusions, f)
    alloc = allocate(f["strata"])
    sample = draw(f["frame"], alloc, a.gate_id, a.snapshot_id)
    cust.mkdir(parents=True, exist_ok=True)
    _dump(cust / "noise-frame.jsonl", f["frame"])
    _dump(cust / "noise-sample.jsonl", sample)
    _dump(cust / "noise-census.jsonl", f["census"])
    (cust / "noise-out-of-scope.json").write_text(json.dumps(f["outOfScope"], sort_keys=True), encoding="utf-8")
    print(json.dumps({"answers": len(answers), "frame": len(f["frame"]), "census": len(f["census"]),
                      "strata": len(f["strata"]), "sample": len(sample),
                      "outOfScope": sum(sum(v.values()) for v in f["outOfScope"].values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
