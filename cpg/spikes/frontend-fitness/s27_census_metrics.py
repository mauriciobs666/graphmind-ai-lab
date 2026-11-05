#!/usr/bin/env python3
"""S27c step 4: ML §7.2 census metrics (parse success, method coverage, line fidelity, value-set recovery, decorator shape)
for ONE language, from `truth/<lang>.defs.jsonl` (make_defs_census.py) and the two `measure.sc` outputs (off / on).

    cd cpg && .venv/bin/python spikes/frontend-fitness/s27_census_metrics.py <lang> [--off P] [--on P] [--defs P] [--out P]

Writes `results/<lang>.census.json`: `{lang, frontendOnly, label, definitions:{...}, off:{...}, on:{...}}`, every metric a
`{k, n, p, wilson95}` (s27_metrics.prop). Rule sources: report §a1.6 item 3 and ML §7.2; names and offsets come from
`s27_metrics` (`spec_names`, `METHOD_LINE_OFFSET`, `NONSYMBOL_NAME`, `prop`) and matcher-table.md, extended here, never redefined.

Input schema expected from graph-dba's `measure.sc` change (the existing `meta`/`file`/`method`/`site` records are unchanged):
  meta   : + "censusSchema": 1 ; + "frontendOnly": true for the overlay-free Kotlin import. Without censusSchema the
           value-set and decorator metrics are reported {"status":"unmeasured"}; coverage, parse success and line fidelity
           need only the existing records.
  method : + "ann": [{"name": str, "node": "ANNOTATION"|"CALL"|..., "lits": [raw literal source text, ...]}] -- the method's
           annotation/decorator names as Joern exposes them (qualified or simple) and the raw `code` of every literal argument.
           (`l`/`le` are `lineNumber`/`lineNumberEnd` as Joern reports them, i.e. 0-based for C#.)
  carg   : one per argument of each frame-site CALL that matched (`site.matched`), joined by the frame's siteKey:
           {"k":"carg","key":<siteKey>,"order":<1-based position of the argument among ALL Joern arguments, receiver included>,
            "recv":<true when that CALL's first argument is its receiver>,"kind":<node label, e.g. "LITERAL","IDENTIFIER","CALL">,
            "lit":<raw source text of the Literal, else null>,"nreach":<number of reaching definitions of the argument, 0 off>,
            "reach":[raw source text of each reaching definition that is a Literal]}
  The census argIndex (0-based over explicit arguments, receiver excluded) is `order - 1 - recv`.

Judgments (consistent with ML §7.2 and §a1.6):
  * Candidate METHODs per definition: unclaimed, non-vendored, in the file, name accepted (see `accepted_names`), start line equal to
    nameLine or declStartLine at METHOD_LINE_OFFSET; else the innermost unclaimed METHOD spanning nameLine within 8 lines of its
    start. An anonymous definition matches by position only (exact start line, `ANON_NAME`). Definitions are visited in (nameLine,
    source order); ties by source order of the METHODs. Non-symbol definitions never claim a METHOD and are not in any denominator.
  * Line fidelity: over matched pairs, METHOD.l == nameLine + offset and METHOD.le == endLine + offset (variant: l == declStartLine
    + offset). The offset applies to both ends.
  * Parse success: census file has a FILE record and (its `methods` count > 0 or a non-vendored method record).
  * Value-set recovery: matched call (site.matched) whose argument is a LITERAL with equal normalised value, or has exactly one
    reaching definition (nreach == 1) which is a Literal of equal value. Values are compared after
    `make_defs_census.normalise_literal`, the same function that produced the census value.
  * Decorator shape: denominator = symbol definitions with >= 1 decorator. Name share (primary, per definition): the matched
    METHOD exposes every decorator name of the definition (last dotted/`::` segment, case-sensitive); an unmatched definition
    exposes nothing. Per-decorator name share and, per tree-sitter node kind, the same pair are reported beside it. Literal share:
    per decorator with >= 1 literal argument, every census literal value appears among the normalised literals of the same-named
    annotations of the matched METHOD.
  * Kotlin: when `meta.frontendOnly` is true, value-set recovery is `{"status":"unmeasured"}` and the result is labelled
    "frontend-only"; a configuration whose measure file is absent is `null`.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_defs_census as mdc  # noqa: E402
import s27_metrics as sm  # noqa: E402

ANON_NAME = re.compile(r"^(<lambda>\d*|<anon>.*|anonymous\d*|lambda\d*)$")
SEARCH_WINDOW = 8


def accepted_names(d: dict):
    """Joern METHOD names that satisfy the name rule for definition `d`, or None for an anonymous one (position only)."""
    if d.get("anon") or d.get("name") is None:
        return None
    names = set(sm.spec_names(d["qname"]))
    names.add(d["nameKey"])
    if d.get("kind") == "ctor":
        names.add("<init>")
    return names


def last_segment(name: str) -> str:
    return re.split(r"\.|::", name.strip())[-1] if name else ""


def read_measure(path) -> dict:
    meta, files = {}, {}
    methods = collections.defaultdict(list)
    sites, cargs = {}, collections.defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        k = r["k"]
        if k == "meta":
            meta = r
        elif k == "file":
            files[r["path"]] = r["methods"]
        elif k == "method":
            if not r.get("vendored"):
                methods[r["file"]].append(r)
        elif k == "site":
            sites[r["key"]] = r
        elif k == "carg":
            cargs[r["key"]][r["order"] - 1 - (1 if r.get("recv") else 0)] = r
    return {"meta": meta, "files": files, "methods": methods, "sites": sites, "cargs": cargs}


def read_defs(path) -> dict:
    out = {"meta": {}, "files": [], "defs": collections.defaultdict(list), "args": []}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        k = r["k"]
        if k == "meta":
            out["meta"] = r
        elif k == "file":
            out["files"].append(r["path"])
        elif k == "def":
            out["defs"][r["path"]].append(r)
        elif k == "arg":
            out["args"].append(r)
    return out


def match_file(defs, methods, off):
    """One-to-one matching of one file's symbol definitions to its METHODs -> {def index: method record}."""
    claimed, out = set(), {}
    order = sorted((i for i, d in enumerate(defs) if d["symbol"]), key=lambda i: (defs[i]["nameLine"], i))
    for i in order:
        d = defs[i]
        names = accepted_names(d)
        nl, dl = d["nameLine"] + off, d["declStartLine"] + off
        cands = [(j, m) for j, m in enumerate(methods) if j not in claimed
                 and ((m["name"] in names) if names is not None else bool(ANON_NAME.search(m["name"]) or ":program" in m.get("fullName", "")))]
        exact = [(j, m) for j, m in cands if m["l"] == nl or m["l"] == dl]
        pick = None
        if exact:
            pick = min(exact, key=lambda jm: (min(abs(jm[1]["l"] - nl), abs(jm[1]["l"] - dl)), jm[0]))
        elif names is not None:
            near = [(j, m) for j, m in cands if m["l"] <= nl <= m["le"] and nl - m["l"] <= SEARCH_WINDOW]
            if near:
                pick = min(near, key=lambda jm: (nl - jm[1]["l"], jm[0]))
        if pick is not None:
            claimed.add(pick[0])
            out[i] = pick[1]
    return out


def _unmeasured(reason):
    return {"status": "unmeasured", "reason": reason}


def census_block(lang: str, D: dict, M: dict) -> dict:
    off = sm.METHOD_LINE_OFFSET[lang]
    meta = M["meta"]
    schema = meta.get("censusSchema") == 1
    frontend_only = bool(meta.get("frontendOnly"))
    sym = ncov = nstart = nstart_decl = nend_ok = 0
    nline = nline_decl = 0
    matched_by = {}
    for path, defs in D["defs"].items():
        symbols = [d for d in defs if d["symbol"]]
        sym += len(symbols)
        if path not in M["files"]:
            continue
        mm = match_file(defs, M["methods"].get(path, []), off)
        matched_by[path] = (defs, mm)
        for i, m in mm.items():
            d = defs[i]
            ncov += 1
            end_ok = m["le"] == d["endLine"] + off
            nline += 1 if (m["l"] == d["nameLine"] + off and end_ok) else 0
            nline_decl += 1 if (m["l"] == d["declStartLine"] + off and end_ok) else 0
    parsed = sum(1 for p in D["files"] if p in M["files"] and (M["files"][p] > 0 or M["methods"].get(p)))
    block = {
        "parseSuccess": sm.prop(parsed, len(D["files"])),
        "methodCoverage": sm.prop(ncov, sym),
        "lineFidelity": sm.prop(nline, ncov),
        "lineFidelityDeclStart": sm.prop(nline_decl, ncov),
    }
    if frontend_only:
        block["valueSetRecovery"] = _unmeasured("frontend-only import: no dataflow overlay")
    elif not schema:
        block["valueSetRecovery"] = _unmeasured("measure output lacks censusSchema")
    else:
        rec, byk = 0, collections.defaultdict(lambda: [0, 0])
        for a in D["args"]:
            s = M["sites"].get(a["siteKey"])
            ok = False
            if s and s.get("matched"):
                c = M["cargs"].get(a["siteKey"], {}).get(a["argIndex"])
                if c:
                    if c.get("kind") == "LITERAL" and c.get("lit") is not None and mdc.normalise_literal(lang, c["lit"]) == a["value"]:
                        ok = True
                    elif c.get("nreach") == 1 and len(c.get("reach") or []) == 1 and mdc.normalise_literal(lang, c["reach"][0]) == a["value"]:
                        ok = True
            rec += ok
            byk[a["valueKind"]][0] += ok
            byk[a["valueKind"]][1] += 1
        block["valueSetRecovery"] = sm.prop(rec, len(D["args"]))
        block["valueSetRecoveryByKind"] = {k: sm.prop(v[0], v[1]) for k, v in sorted(byk.items())}
    if not schema:
        block["decorators"] = _unmeasured("measure output lacks censusSchema")
    else:
        block["decorators"] = decorator_block(lang, D, matched_by)
    return block


def decorator_block(lang, D, matched_by) -> dict:
    defs_k = defs_n = dec_k = dec_n = lit_k = lit_n = 0
    by_node = collections.defaultdict(lambda: [0, 0, 0, 0])  # name k, name n, lit k, lit n
    for path, defs in D["defs"].items():
        mm = matched_by.get(path, (defs, {}))[1]
        for i, d in enumerate(defs):
            if not d["symbol"] or not d["decorators"]:
                continue
            m = mm.get(i)
            exposed = collections.defaultdict(set)
            for a in (m.get("ann") or []) if m else []:
                exposed[last_segment(a["name"])].update(mdc.normalise_literal(lang, x) for x in a.get("lits") or [])
            all_ok = True
            for dec in d["decorators"]:
                seg = last_segment(dec["name"])
                ok = seg in exposed
                all_ok &= ok
                dec_n += 1
                dec_k += ok
                by_node[dec["node"]][1] += 1
                by_node[dec["node"]][0] += ok
                if dec["literalArgs"]:
                    lok = ok and all(x["value"] in exposed[seg] for x in dec["literalArgs"])
                    lit_n += 1
                    lit_k += lok
                    by_node[dec["node"]][3] += 1
                    by_node[dec["node"]][2] += lok
            defs_n += 1
            defs_k += all_ok
    return {"nameShare": sm.prop(defs_k, defs_n), "nameShareDecorators": sm.prop(dec_k, dec_n), "literalShare": sm.prop(lit_k, lit_n),
            "byNode": {k: {"nameShareDecorators": sm.prop(v[0], v[1]), "literalShare": sm.prop(v[2], v[3])} for k, v in sorted(by_node.items())}}


def run(lang: str, off_path=None, on_path=None, defs_path=None) -> dict:
    D = read_defs(defs_path or mdc.out_path(lang))
    result = {"lang": lang, "definitions": {"files": len(D["files"]), "defs": sum(len(v) for v in D["defs"].values()),
                                              "symbolDefs": sum(1 for v in D["defs"].values() for d in v if d["symbol"]),
                                              "args": len(D["args"])}}
    for cfg, p in (("off", off_path), ("on", on_path)):
        if p is None:
            fs = sorted(glob.glob(str(sm.WORK / lang / cfg / "*.measure.jsonl")), key=lambda x: Path(x).stat().st_mtime)
            p = fs[-1] if fs else None
        if p is None:
            result[cfg] = None
            continue
        M = read_measure(p)
        result[cfg] = census_block(lang, M=M, D=D)
        result[cfg]["measure"] = str(p)
        result.setdefault("frontendOnly", False)
        result["frontendOnly"] = result["frontendOnly"] or bool(M["meta"].get("frontendOnly"))
    result.setdefault("frontendOnly", False)
    result["label"] = "frontend-only" if result["frontendOnly"] else "frontend+overlays"
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lang")
    ap.add_argument("--off")
    ap.add_argument("--on")
    ap.add_argument("--defs")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = run(a.lang, a.off, a.on, a.defs)
    out = Path(a.out) if a.out else HERE / "results" / f"{a.lang}.census.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({b: {k: v for k, v in res[b].items() if k in ("parseSuccess", "methodCoverage", "lineFidelity")} for b in ("off", "on") if res[b]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
