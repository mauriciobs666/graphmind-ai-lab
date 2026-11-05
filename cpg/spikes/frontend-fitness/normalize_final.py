"""Deterministic, shape-only converter: the seven S57 `truth/<lang>.final.jsonl` + `adjudication/<lang>.noise.json`
shapes -> the one canonical shape defined in `validate_final.py` (read its docstring first).

    python normalize_final.py --out DIR [lang ...]

Writes NEW files `<out>/<lang>.final.jsonl` and `<out>/<lang>.noise.json`; never writes into
`truth/` or `adjudication/` (refused). Labels are not changed: before writing, every language is checked
for label-set equality (site -> T / external / N / notACall / dynamic / scored / reason / basis), and any
shape it cannot convert without guessing raises instead of defaulting. Two conversions touch more than
spelling and are recorded here: `strata` compound tags `A:B` become two tags `A`, `B` (so javascript #199's
`audit-U:AS` now carries `AS`), and noise `errorKind` `schema` becomes `shape` (§5.4 steps 6-7 vocabulary).
The original noise.json is kept verbatim under `detail`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import hoist_open_points
import validate_final as vf

HERE = Path(__file__).resolve().parent


class AlreadyCanonical(ValueError):
    """The input is already in the canonical shape; there is nothing to convert."""


# ---- final.jsonl ----------------------------------------------------------------------------------

def _external(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        out = []
        for x in raw:
            if isinstance(x, dict):
                if x.get("kind") != "external" or not isinstance(x.get("package"), str):
                    raise ValueError(f"external item not convertible without guessing: {x!r}")
                out.append(x["package"])
            elif isinstance(x, str):
                out.append(x)
            else:
                raise ValueError(f"external item {x!r}")
        return out
    raise ValueError(f"external {raw!r}")


def _site(row, site_lookup):
    raw = row.get("site")
    if isinstance(raw, dict):
        raw = f"{raw['path']}:{raw['line']}"
    looked = site_lookup.get(row["siteId"])
    if raw is None:
        if looked is None:
            raise ValueError(f"no site for siteId {row['siteId']}")
        return looked
    if looked is not None and looked != raw:
        raise ValueError(f"site {raw!r} conflicts with sample/input {looked!r} (siteId {row['siteId']})")
    return raw


def _evidence(row):
    present = [k for k in ("evidence", "citation", "citations") if k in row]
    if len(present) != 1:
        raise ValueError(f"exactly one of evidence/citation/citations expected, got {present}")
    v = row[present[0]]
    if isinstance(v, list):
        if not all(isinstance(x, str) for x in v):
            raise ValueError("evidence list with non-str item")
        return "; ".join(dict.fromkeys(v))
    return v


def _strata(tags):
    return list(dict.fromkeys(p for t in tags for p in t.split(":")))


def normalize_row(row, site_lookup) -> dict:
    out = {
        "ordinal": row["ordinal"], "siteId": row["siteId"], "site": _site(row, site_lookup),
        "T": [dict(t) for t in row["T"]], "N": list(row["N"]), "external": _external(row.get("external")),
        "notACall": row["notACall"], "dynamic": row["dynamic"], "scored": row["scored"],
        "notScoredReason": row["notScoredReason"], "basis": row["basis"], "strata": _strata(row["strata"]),
        "evidence": _evidence(row),
    }
    return out


def label_set(row, site) -> frozenset:
    """Every decision-bearing field of a row, read from any of the raw spellings (independent of normalize_row)."""
    ext = row.get("external")
    pk = [] if ext is None else [ext] if isinstance(ext, str) else ext
    items = {("site", site), ("siteId", row["siteId"]), ("ordinal", row["ordinal"])}
    items |= {("ext", x["package"] if isinstance(x, dict) else x) for x in pk}
    items |= {("T", tuple(sorted(t.items(), key=lambda kv: kv[0]))) for t in row["T"]}
    items |= {("N", x) for x in row["N"]}
    items |= {(k, row[k]) for k in ("notACall", "dynamic", "scored", "notScoredReason", "basis")}
    return frozenset(items)


def assert_labels_equal(raw_rows, new_rows, site_lookup):
    assert len(raw_rows) == len(new_rows), "line count changed"
    for r, n in zip(raw_rows, new_rows):
        site = n["site"]
        before, after = label_set(r, site), label_set(n, site)
        assert before == after, f"label set changed at ordinal {r['ordinal']}: {sorted(before ^ after, key=str)}"
        assert len(n["external"]) == len(_external(r.get("external"))) and len(n["T"]) == len(r["T"])


# ---- noise.json -----------------------------------------------------------------------------------

def errorkinds(d):
    out = {}
    for k, v in d.items():
        k = "shape" if k == "schema" else k
        out[k] = out.get(k, 0) + v
    return out


def adjudicated_from_combos(n, W, combos):
    bw, bk = Counter(), Counter()
    for key, v in combos.items():
        w, k = key.split("|")
        bw[w] += v
        bk[k] += v
    return {"n": n, "W": W, "byWrong": dict(bw), "byErrorKind": errorkinds(bk)}


def _ag(agreed, comparable, rate, wilson):
    return {"agreed": agreed, "comparable": comparable, "rate": rate, "wilson95": wilson}


def _refe(wrong, comparable, rate, wilson):
    return {"wrong": wrong, "comparable": comparable, "rate": rate, "wilson95": wilson}


def _he(all_p, all_c, res_p, res_c, flag):
    return {"e_all": {"point": all_p, "conservative": all_c}, "e_res": {"point": res_p, "conservative": res_c},
            "noiseLimited": flag}


def _aud(s, N_s, n, k, w):
    return {"N_s": N_s, "n": n, "k": k, "wilson95": w}


def _byKind(role=(), coordinate=(), shape=()):
    return {"role": sorted(role), "coordinate": sorted(coordinate), "shape": sorted(shape)}


def _closed(byKind):
    return {"count": len({x for v in byKind.values() for x in v}), "byKind": byKind}


def _zero(v, what):
    if v != 0:
        raise ValueError(f"{what} = {v!r}: a closed-class claim must be mapped explicitly, not defaulted to none")


def _python(d, N):
    a, cc = d["agreement"], d["closedClass"]
    return dict(
        comparable=d["comparable"]["comparable"], referenceSilent=d["comparable"]["referenceSilent"],
        agreement=_ag(a["agreed"], a["comparable"], a["rate"], a["wilson95"]),
        adjudicated=adjudicated_from_combos(d["adjudicated"]["n"], d["adjudicated"]["W_labelWrong"], d["adjudicated"]["byWrongAndErrorKind"]),
        referenceError=_refe(d["referenceError"]["adjudicatedWrongOnComparable"], d["referenceError"]["comparable"], d["referenceError"]["rate"], d["referenceError"]["wilson95"]),
        audit={s: _aud(s, x["N_s"], x["n"], x["errors_k"], x["wilson95"]) for s, x in d["auditTable"].items()},
        handError=_he(d["handError"]["e_all"], d["handError"]["e_all_conservative"], d["handError"]["e_res"], d["handError"]["e_res_conservative"], d["handError"]["noiseLimited"]),
        census={"targetsRead": d["census"]["targetsRead"]},
        closedClasses=_closed(_byKind(**cc["instancesByOrdinal"])),
        unscored={"byReason": dict(d["unscored"]), "underivable": d["underivable"]},
        _basisCounts=d["basisCounts"])


def _javascript(d, N):
    he = d["handErrorRate"]
    _zero(d["census"]["roleErrors"], "javascript census.roleErrors")
    _zero(he["roleErrors"], "javascript handErrorRate.roleErrors")
    r = d["referenceErrorRate"]
    return dict(
        comparable=d["comparable"], referenceSilent=N - d["comparable"],
        agreement=_ag(d["agreedComparable"], d["comparable"], d["agreementRate"], d["agreementWilson95"]),
        adjudicated={"n": d["adjudicated"]["total"], "W": d["adjudicated"]["W_labelWrong"], "byWrong": d["adjudicated"]["byWrong"], "byErrorKind": errorkinds(d["adjudicated"]["byErrorKind"])},
        referenceError=_refe(r["wrongSites"], r["comparable"], r["rate"], r["wilson95"]),
        audit={s: _aud(s, x["population"], x["n"], x["errors"], x["wilson95"]) for s, x in d["auditTable"].items()},
        handError=_he(he["withRoleErrors"], he["conservativeWilsonBoundWithRole"], he["withoutRoleErrors"], he["conservativeWilsonBoundWithoutRole"], he["noiseLimited"]),
        census={"targetsRead": d["census"]["entries"]},
        closedClasses=_closed(_byKind()),
        unscored={"byReason": {k: v for k, v in d["unscored"].items() if k in ("notACall",)}, "underivable": d["unscored"]["underivable"]})


def _go(d, N):
    a, he, r = d["agreement"]["overall"], d["handErrorRate"], d["referenceErrorRate"]
    _zero(d["role"]["labelErrors"], "go role.labelErrors")
    _zero(he["roleErrors"], "go handErrorRate.roleErrors")
    return dict(
        comparable=d["comparable"], referenceSilent=d["referenceSilent"]["total"],
        agreement=_ag(a["agreed"], a["comparable"], a["rate"], a["wilson95"]),
        adjudicated={"n": d["adjudicated"]["n"], "W": d["adjudicated"]["W_labelWrong"], "byWrong": d["adjudicated"]["wrong"], "byErrorKind": errorkinds(d["adjudicated"]["errorKind"])},
        referenceError=_refe(r["adjudicatedReferenceWrong"], r["outOf"], r["rate"], r["wilson95"]),
        audit={s: _aud(s, x["N_s"], x["n"], x["k"], x["wilson95"]) for s, x in d["auditTable"].items()},
        handError=_he(he["point"], he["conservativeWilsonUpper"], he["pointWithoutRoleErrors"], he["conservativeWithoutRoleErrors"], he["noiseLimitedByPoint"]),
        census={"targetsRead": d["role"]["censusItems"]},
        closedClasses=_closed(_byKind()),
        unscored={"byReason": dict(d["scored"]["unscored"]), "underivable": round(d["scored"]["underivableShare"] * N)})


def _java(d, N):
    he, r, a = d["handErrorRate"], d["referenceErrorRate"], d["agreementRate"]
    _zero(he["withRoleErrors"]["labelRoleErrors"], "java labelRoleErrors")
    return dict(
        comparable=d["comparable"]["comparable"], referenceSilent=d["comparable"]["silentReference"],
        agreement=_ag(a["agreed"], a["comparable"], a["rate"], a["wilson95"]),
        adjudicated={"n": d["adjudicated"]["n"], "W": d["adjudicated"]["W"], "byWrong": d["adjudicated"]["byWrong"], "byErrorKind": errorkinds(d["adjudicated"]["byErrorKind"])},
        referenceError=_refe(r["referenceErrorsFound"], r["comparable"], r["rate"], r["wilson"]),
        audit={s: _aud(s, x["N_s"], x["n"], x["errors"], x.get("wilson")) for s, x in d["auditTable"].items()},
        handError=_he(he["withRoleErrors"]["point"], he["withRoleErrors"]["conservativeAIwilsonUpper"], he["withoutRoleErrors"]["point"], he["withoutRoleErrors"]["conservativeAIwilsonUpper"], he["noiseLimited"]),
        census={"targetsRead": None},  # java's noise.json records no census count; null is allowed with no closed class
        closedClasses=_closed(_byKind()),
        unscored={"byReason": {}, "underivable": d["scoring"]["underivable"]},
        _basisCounts=d["basisCounts"])


def _csharp(d, N):
    he, a, ad = d["handErrorRate"], d["agreement"], d["adjudicated"]
    _zero(he["roleErrors"], "csharp handErrorRate.roleErrors")
    _zero(d["census"]["labelErrors"], "csharp census.labelErrors")
    r = ad["referenceWrongOverComparable"]
    return dict(
        comparable=d["comparable"]["n"], referenceSilent=d["comparable"]["referenceSilent"],
        agreement=_ag(a["agreed"], a["comparable"], a["rate"], a["wilson95"]),
        adjudicated={"n": ad["n"], "W": ad["W_labelWrong"], "byWrong": ad["byWrong"], "byErrorKind": errorkinds(ad["byErrorKind"])},
        referenceError=_refe(r["k"], r["n"], r["rate"], r["wilson95"]),
        audit={s: _aud(s, x["N_s"], x["U_n"], x["U_errors"], x["U_wilson"]) for s, x in d["audit"].items()},
        handError=_he(he["withRoleErrors"]["point"], he["withRoleErrors"]["conservative"], he["withoutRoleErrors"]["point"], he["withoutRoleErrors"]["conservative"], he["noiseLimited_point"]),
        census={"targetsRead": d["census"]["entries"]},
        closedClasses=_closed(_byKind()),
        unscored={"byReason": dict(d["unscored"]), "underivable": d["underivable"]})


def _cpp(d, N):
    he, r, cc = d["handErrorRate"], d["referenceError"], d["closedClasses"]
    ad, a = d["adjudicated"], d["agreement"]
    return dict(
        comparable=d["comparable"]["comparable"], referenceSilent=d["comparable"]["silent"],
        agreement=_ag(a["agreed_comparable"], d["comparable"]["comparable"], a["rate"], a["wilson95"]),
        adjudicated={"n": ad["n"], "W": ad["W"], "byWrong": ad["byWrong"], "byErrorKind": errorkinds(ad["byErrorKind"])},
        referenceError=_refe(r["adjudicatedRefErrors_comparable"], r["comparable"], r["point"], r["wilson95"]),
        audit={s: _aud(s, x["N_s"], x["n"], x["k"], x["wilson"]) for s, x in d["audit"].items() if s in vf.AUDIT_STRATA},
        handError=_he(he["e_all"]["point"], he["e_all"]["conservative"], he["e_res"]["point"], he["e_res"]["conservative"], d["noiseLimited"]),
        census={"targetsRead": d["census"]["targetsRead"]},
        closedClasses=_closed(_byKind(role=cc["role"]["ordinals"], shape=cc["shape"]["ordinals"])),
        unscored={"byReason": dict(d["unscored"]), "underivable": d["underivable"]["count"]})


def _kotlin(d, N):
    he, r, cc = d["handError"], d["referenceError"], d["closedClasses"]
    a, ad = d["agreement"]["overall"], d["adjudicated"]
    aud = {}
    for s, x in d["auditTable"].items():
        aud[s] = _aud(s, x["populationNs"], x["n"], x.get("errorsAll", 0 if x["n"] == 0 else None), x.get("wilsonAll"))
    return dict(
        comparable=a["comparable"], referenceSilent=N - a["comparable"],
        agreement=_ag(a["agreed"], a["comparable"], a["rate"], a["wilson95"]),
        adjudicated={**adjudicated_from_combos(ad["n"], ad["W_all"], ad["byWrongErrorKind"])},
        referenceError=_refe(r["adjudicatedSitesReferenceWrong"], r["comparableSites"], r["rate"], None),
        audit=aud,
        handError=_he(he["e_all"], he["e_all_conservative"], he["e_res"], he["e_res_conservative"], he["noiseLimited"]),
        census={"targetsRead": d["census"]["targetsRead"]},
        closedClasses=_closed(_byKind(role=cc["role_primaryConstructor"]["instancesByOrdinal"], coordinate=cc["coordinate"]["instances"], shape=cc["shape"]["instances"])),
        unscored={"byReason": {}, "underivable": d["underivable"]["n"]},
        _basisCounts=d["basisCounts"])


_NOISE = {"python": _python, "javascript": _javascript, "go": _go, "java": _java, "csharp": _csharp, "cpp": _cpp, "kotlin": _kotlin}


def normalize_noise(lang, noise, rows) -> dict:
    """`rows`: the canonical final rows (basisCounts and the unscored reasons are recomputed from them)."""
    env = _NOISE[lang](noise, len(rows))
    claimed = env.pop("_basisCounts", None)
    basis = dict(Counter(r["basis"] for r in rows))
    if claimed is not None and {k: v for k, v in claimed.items() if v} != basis:
        raise ValueError(f"{lang}: noise basisCounts {claimed} differs from the final file {basis}")
    reasons = dict(Counter(r["notScoredReason"] for r in rows if r["notScoredReason"]))
    src = env["unscored"]
    if {k: v for k, v in src["byReason"].items() if v} != {k: v for k, v in reasons.items() if k != "underivable"} \
            or src["underivable"] != reasons.get("underivable", 0):
        raise ValueError(f"{lang}: noise unscored {src} differs from the final file {reasons}")
    env["unscored"] = {"byReason": reasons, "underivable": reasons.get("underivable", 0)}
    out = {"lang": lang, "N": len(rows), **{k: env[k] for k in (
        "comparable", "referenceSilent", "agreement", "adjudicated", "referenceError", "audit", "handError",
        "census", "closedClasses", "unscored")}, "basisCounts": basis}
    # v1.12 openPointsS27: carried over from wherever the source nested it; a v1.11 source has none and the
    # output then (correctly) fails the validator's required-key rule rather than getting an invented entry
    found: list = []
    hoist_open_points._find(noise, (), found)
    if len(found) > 1:
        raise ValueError(f"{lang}: {len(found)} openPointsS27 entries in the source")
    if found:
        out[hoist_open_points.KEY] = found[0][1][hoist_open_points.KEY]
    out["detail"] = noise
    return out


# ---- driver ---------------------------------------------------------------------------------------

def _jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _site_lookup(lang, adj_dir, sample_dir):
    look = {r["siteId"]: f"{r['path']}:{r['line']}" for r in _jsonl(Path(sample_dir) / f"{lang}.jsonl")}
    for r in _jsonl(Path(adj_dir) / f"{lang}.input.jsonl"):
        s = r.get("site")
        if isinstance(s, dict):
            v = f"{s['path']}:{s['line']}"
            if look.setdefault(r["siteId"], v) != v:
                raise ValueError(f"{lang}: sample and input.jsonl disagree on the site of {r['siteId']}")
    return look


def normalize_lang(lang, truth_dir, adj_dir, sample_dir, out_dir):
    out_dir = Path(out_dir).resolve()
    if out_dir in {Path(truth_dir).resolve(), Path(adj_dir).resolve()}:
        raise ValueError("refusing to write into truth/ or adjudication/")
    raw_noise = json.loads((Path(adj_dir) / f"{lang}.noise.json").read_text())
    if "detail" in raw_noise and "referenceSilent" in raw_noise:
        raise AlreadyCanonical(f"{lang}: {Path(adj_dir) / (lang + '.noise.json')} is already canonical; nothing to normalize")
    raw = _jsonl(Path(truth_dir) / f"{lang}.final.jsonl")
    look = _site_lookup(lang, adj_dir, sample_dir)
    rows = [normalize_row(r, look) for r in raw]
    assert_labels_equal(raw, rows, look)
    noise = normalize_noise(lang, raw_noise, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{lang}.final.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out_dir / f"{lang}.noise.json").write_text(json.dumps(noise, ensure_ascii=False, indent=1) + "\n")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("langs", nargs="*")
    ap.add_argument("--out", required=True, help="directory for the new files (never truth/ or adjudication/)")
    ap.add_argument("--truth", default=str(HERE / "truth"))
    ap.add_argument("--adjudication", default=str(HERE / "adjudication"))
    ap.add_argument("--sample", default=str(HERE / "sample"))
    a = ap.parse_args(argv)
    for lang in a.langs or vf.LANGS:
        try:
            rows = normalize_lang(lang, a.truth, a.adjudication, a.sample, a.out)
        except AlreadyCanonical as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        print(f"{lang}: {len(rows)} lines -> {a.out}/{lang}.final.jsonl (+ noise.json); labels equal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
