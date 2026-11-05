#!/usr/bin/env python3
"""S27c step 4: ML §7.2 census metrics (parse success, method coverage, line fidelity, value-set recovery, decorator shape)
for ONE language, from `truth/<lang>.defs.jsonl` (make_defs_census.py) and the two `measure.sc` outputs (off / on).

    cd cpg && .venv/bin/python spikes/frontend-fitness/s27_census_metrics.py <lang> [--off P] [--on P] [--defs P] [--out P]

Writes `results/<lang>.census.json`: `{lang, frontendOnly, label, definitions:{...}, off:{...}, on:{...}}`, every metric a
`{k, n, p, wilson95}` (s27_metrics.prop). Rule sources: report §a1.6 item 3 and ML §7.2; names and offsets come from
`s27_metrics` (`spec_names`, `METHOD_LINE_OFFSET`, `NONSYMBOL_NAME`, `prop`) and matcher-table.md, extended here, never redefined.

Input schema expected from graph-dba's `measure.sc` change. The existing `meta`/`file`/`method`/`site` records are unchanged except for the
additions below. NOTHING below exists in measure.sc today; the committed fixtures `tests/data/census/<lang>.measure.jsonl` are SYNTHETIC
(built from the test fixtures by `synth_measure`) and fix the format, not Joern's behaviour: graph-dba verifies each field against real Joern
per frontend and may ask for changes here.
  meta   : + "censusSchema": 1 ; + "frontendOnly": true for the overlay-free import (Kotlin). Without censusSchema the value-set and
           decorator metrics are `{"status":"unmeasured"}`; coverage, parse success and line fidelity need only the existing records.
  method : + "ann": [{"name": str, "node": "ANNOTATION"|"CALL"|..., "lits": [raw source text of each literal argument]}] -- the method's
           annotation/decorator names as Joern exposes them (qualified or simple; compared on the last `.`/`::` segment) and the raw
           `code` of each literal argument. `l`/`le` are Joern's lineNumber/lineNumberEnd as reported (0-based for C#).
  carg   : one record per argument of each frame-site CALL that matched (`site.matched`), joined by the frame's siteKey:
           {"k":"carg","key":<siteKey>,"order":<int>,"recv":<bool>,"kind":<label>,"lit":<str|null>,"nreach":<int>,"reach":[<str>]}
           * `order` = the 1-based RANK of the argument among the call's arguments sorted by Joern `argumentIndex` (ascending, ties by
             node order), the receiver INCLUDED when the call has one. It is not the raw argumentIndex: frontends put the receiver at
             argumentIndex 0 or 1 and number the rest differently, so measure.sc must rank, not copy.
           * `recv` = true when the call's receiver is one of the arguments counted in `order` (so rank 1 is the receiver); computed per
             call, false for a plain call. The census argIndex (0-based over explicit arguments) is `order - 1 - recv`.
           * `kind` = the argument node label ("LITERAL", "IDENTIFIER", "CALL", ...); `lit` = raw source text of the Literal when
             kind is LITERAL, else null; `nreach` = number of reaching definitions of an IDENTIFIER argument (0 when the dataflow
             overlay is absent); `reach` = raw source text of each reaching definition that is a Literal.

Judgments (consistent with ML §7.2 and §a1.6):
  * Candidate METHODs per definition: unclaimed, non-vendored, in the file, name accepted (see `accepted_names`), start line equal to
    nameLine or declStartLine at METHOD_LINE_OFFSET; else the innermost unclaimed METHOD spanning nameLine within 8 lines of its
    start. An anonymous definition matches by position only (exact start line, `ANON_NAME`). Definitions are visited in (nameLine,
    source order); ties by source order of the METHODs. Non-symbol definitions never claim a METHOD and are not in any denominator.
  * Line fidelity: over matched pairs, METHOD.l == nameLine + offset and METHOD.le == endLine + offset (variant: l == declStartLine
    + offset). The offset applies to both ends.
  * Parse success: census file has a FILE record and (its `methods` count > 0 or a non-vendored method record). For python/javascript
    this is vacuous (a `<module>`/`:program` METHOD always exists); `parseSuccessStrict` is beside it: over files whose census has a
    symbol definition, those with at least one non-synthetic METHOD (not `NONSYMBOL_NAME`).
  * Strata (ruling): every figure is the headline over the whole frame (as shipped); `strata.inCpgFiles` / `strata.absentFiles` split the
    same figures by whether the file has a FILE node. Files absent from the CPG (cpp tests, Go `_test.go`, C# tests) stay in the headline
    denominator: the frame-versus-import gap is reported, not hidden.
  * Operator names: c2cpg names operator methods by the bare token (`()`, `=`, `==`, `<<`); `accepted_names` adds that alias.
  * Macro stratum (cpp): definitions of kind "macro" are outside the headline denominator; `macroStratum` gives coverage with them.
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
    for k in list(names):
        if k.startswith("operator"):
            # c2cpg names operator methods by the bare operator token: `=`, `()`, `==`, `<<` (observed: 149 such METHODs in the
            # cpp off measure); `operator new` -> `new`. csharpsrc2cpg emitted no operator METHOD at all in the existing measure
            # (10 C# operator defs, no METHOD within 2 lines), so no C# alias is added until one is observed. Kotlin `operator fun`
            # is an ordinary identifier name and needs none.
            rest = k[len("operator"):].strip()
            if rest:
                names.add(rest)
    return names


def last_segment(name: str) -> str:
    return re.split(r"\.|::", name.strip())[-1] if name else ""


# Observed per frontend (results/<lang>.decorator-probe.json, matcher-table.md "Census fields"): which decorator facts Joern exposes.
NO_DECORATOR_NODE = {"cpp", "go"}          # no ANNOTATION/CALL node: name share and literal share are unmeasurable, not 0
LITERALS_ONLY_IN_CODE = {"javascript"}     # ANNOTATION has no literal children: the arguments are only in `code`
_CPP_STR = re.compile(r'(?:u8|u|U|L)?R"[^()\\ ]{0,16}\(.*?\)[^()\\ ]{0,16}"|(?:u8|u|U|L)?"(?:\\.|[^"\\])*"', re.S)
_ANY_STR = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`', re.S)


def joern_literal_value(lang: str, raw, node=None):
    """Normalised value of a Literal's source `code` as Joern reports it. Java ANNOTATION_LITERAL.code is already unquoted;
    C++ keeps adjacent literals (`"a" "b"`) as one LITERAL."""
    if raw is None:
        return None
    if lang == "java" and node == "ANNOTATION_LITERAL":
        return raw
    if lang == "cpp":
        parts = _CPP_STR.findall(raw)
        if len(parts) > 1:
            vals = [mdc.normalise_literal("cpp", x) for x in parts]
            return None if any(v is None for v in vals) else "".join(vals)
    return mdc.normalise_literal(lang, raw)


def ann_values(lang: str, a: dict) -> set:
    nodes = a.get("litNodes") or [None] * len(a.get("lits") or [])
    return {joern_literal_value(lang, x, n) for x, n in zip(a.get("lits") or [], nodes)}


def code_literals(lang: str, code: str) -> set:
    """JS: string literals parsed out of the ANNOTATION `code` text (the only place Joern keeps them)."""
    return {mdc.normalise_literal(lang, x) for x in _ANY_STR.findall(code or "")}


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


def match_file(defs, methods, off, with_macros=False):
    """One-to-one matching of one file's symbol definitions to its METHODs -> {def index: method record}.
    `with_macros` also offers the macro-stratum definitions (kind "macro", symbol:false) for matching."""
    claimed, out = set(), {}
    order = sorted((i for i, d in enumerate(defs) if d["symbol"] or (with_macros and d.get("kind") == "macro")),
                   key=lambda i: (defs[i]["nameLine"], i))
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


def split_defs(D: dict, in_set) -> tuple:
    """(D restricted to files in `in_set`, D restricted to the others): the frame-versus-import strata."""
    def part(keep):
        return {"meta": D["meta"], "files": [f for f in D["files"] if (f in in_set) == keep],
                "defs": {p: v for p, v in D["defs"].items() if (p in in_set) == keep},
                "args": [a for a in D["args"] if (a["path"] in in_set) == keep]}
    return part(True), part(False)


def census_block(lang: str, D: dict, M: dict, strata: bool = True) -> dict:
    """Headline (as-shipped, whole frame) block; `strata` adds the in-CPG-file-set / absent-file split (ruling: absent files
    such as cpp TEST files stay in the denominator, the gap is reported, not hidden)."""
    block = _census_core(lang, D, M)
    if strata:
        din, dabs = split_defs(D, set(M["files"]))
        block["strata"] = {"inCpgFiles": _census_core(lang, din, M), "absentFiles": _census_core(lang, dabs, M)}
    return block


def _census_core(lang: str, D: dict, M: dict) -> dict:
    off = sm.METHOD_LINE_OFFSET[lang]
    meta = M["meta"]
    schema = meta.get("censusSchema") == 1
    frontend_only = bool(meta.get("frontendOnly"))
    sym = ncov = nstart = nstart_decl = nend_ok = 0
    nline = nline_decl = 0
    end_delta = collections.Counter()
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
            nstart += 1 if m["l"] == d["nameLine"] + off else 0
            end_delta[m["le"] - (d["endLine"] + off)] += 1
            nline += 1 if (m["l"] == d["nameLine"] + off and end_ok) else 0
            nline_decl += 1 if (m["l"] == d["declStartLine"] + off and end_ok) else 0
    parsed = sum(1 for p in D["files"] if p in M["files"] and (M["files"][p] > 0 or M["methods"].get(p)))
    # python/javascript: `<module>`/`:program` always exists, so `parseSuccess` is vacuous there (ML §7.2 as written); the strict
    # figure asks for a non-synthetic METHOD in the files whose census has a symbol definition.
    with_defs = [p for p in D["files"] if any(d["symbol"] for d in D["defs"].get(p, []))]
    strict = sum(1 for p in with_defs if any(not sm.NONSYMBOL_NAME.search(m["name"]) for m in M["methods"].get(p, [])))
    macro_defs = sum(1 for v in D["defs"].values() for d in v if d.get("kind") == "macro")
    block = {
        "parseSuccess": sm.prop(parsed, len(D["files"])),
        "parseSuccessStrict": sm.prop(strict, len(with_defs)),
        "methodCoverage": sm.prop(ncov, sym),
        "lineFidelity": sm.prop(nline, ncov),
        "lineFidelityDeclStart": sm.prop(nline_decl, ncov),
        "lineFidelityStartOnly": sm.prop(nstart, ncov),
        # METHOD.lineNumberEnd minus the census end line (offset applied), most common first: a constant offset reads here as one bucket
        "lineEndDelta": {str(k): v for k, v in end_delta.most_common(8)},
    }
    if macro_defs:
        mk_ = tot = 0
        for path, defs in D["defs"].items():
            if path in M["files"]:
                mm = match_file(defs, M["methods"].get(path, []), off, with_macros=True)
                mk_ += sum(1 for i in mm if defs[i].get("kind") == "macro")
                tot += len(mm)
        block["macroStratum"] = {"macroDefs": macro_defs, "macroDefsMatched": sm.prop(mk_, macro_defs),
                                 "methodCoverageWithMacros": sm.prop(tot, sym + macro_defs)}
    if frontend_only:
        block["valueSetRecovery"] = _unmeasured("frontend-only import: no dataflow overlay")
    elif not schema:
        block["valueSetRecovery"] = _unmeasured("measure output lacks censusSchema")
    else:
        rec, byk = 0, collections.defaultdict(lambda: [0, 0])
        why = collections.Counter()
        for a in D["args"]:
            s = M["sites"].get(a["siteKey"])
            ok, cause = False, None
            if not (s and s.get("matched")):
                cause = "callUnmatched"
            elif a["siteKey"] not in M["cargs"]:
                cause = "matchedCallHasNoArgumentNodes"      # e.g. csharp unqualified calls: zero `argument`s in csharpsrc2cpg
            else:
                c = M["cargs"][a["siteKey"]].get(a["argIndex"])
                if c is None:
                    cause = "noArgumentAtIndex"
                elif c.get("kind") == "LITERAL" and c.get("lit") is not None and joern_literal_value(lang, c["lit"]) == a["value"]:
                    ok = True
                elif c.get("kind") == "LITERAL":
                    cause = "literalValueDiffers"
                elif c.get("nreach") == 1 and len(c.get("reach") or []) == 1 and joern_literal_value(lang, c["reach"][0]) == a["value"]:
                    ok = True
                else:
                    cause = "notALiteralNoSingleLiteralDefinition"
            rec += ok
            if cause:
                why[cause] += 1
            byk[a["valueKind"]][0] += ok
            byk[a["valueKind"]][1] += 1
        block["valueSetRecovery"] = sm.prop(rec, len(D["args"]))
        block["valueSetUnrecoveredBy"] = dict(sorted(why.items()))
        block["valueSetRecoveryByKind"] = {k: sm.prop(v[0], v[1]) for k, v in sorted(byk.items())}
    if not schema:
        block["decorators"] = _unmeasured("measure output lacks censusSchema")
    else:
        block["decorators"] = decorator_block(lang, D, matched_by)
    return block


def decorator_block(lang, D, matched_by) -> dict:
    defs_k = defs_n = dec_k = dec_n = lit_k = lit_n = code_k = dm_k = dm_n = 0
    by_node = collections.defaultdict(lambda: [0, 0, 0, 0])  # name k, name n, lit k, lit n
    for path, defs in D["defs"].items():
        mm = matched_by.get(path, (defs, {}))[1]
        for i, d in enumerate(defs):
            if not d["symbol"] or not d["decorators"]:
                continue
            m = mm.get(i)
            exposed = collections.defaultdict(set)
            from_code = collections.defaultdict(set)
            for a in (m.get("ann") or []) if m else []:
                exposed[last_segment(a["name"])].update(ann_values(lang, a))
                from_code[last_segment(a["name"])].update(code_literals(lang, a.get("code")))
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
                    code_k += ok and all(x["value"] in from_code[seg] for x in dec["literalArgs"])
                    lit_n += 1
                    lit_k += lok
                    by_node[dec["node"]][3] += 1
                    by_node[dec["node"]][2] += lok
            defs_n += 1
            defs_k += all_ok
            if m is not None:
                dm_n += 1
                dm_k += all_ok
    out = {"nameShare": sm.prop(defs_k, defs_n), "nameShareDecorators": sm.prop(dec_k, dec_n), "literalShare": sm.prop(lit_k, lit_n),
           "byNode": {k: {"nameShareDecorators": sm.prop(v[0], v[1]), "literalShare": sm.prop(v[2], v[3])} for k, v in sorted(by_node.items())},
           "nameShareMatchedDefs": sm.prop(dm_k, dm_n), "decoratedDefs": defs_n, "decorators": dec_n, "decoratorsWithLiteralArgs": lit_n}
    if lang in NO_DECORATOR_NODE:
        why = _unmeasured("frontend exposes no decorator node (decorator-probe)")
        out.update(nameShareMatchedDefs=why, nameShare=why, nameShareDecorators=why, literalShare=why, byNode={})
    elif lang in LITERALS_ONLY_IN_CODE:
        out["literalShare"] = _unmeasured("ANNOTATION has no literal nodes; arguments only in its code text (see literalShareFromCode)")
        out["literalShareFromCode"] = sm.prop(code_k, lit_n)
        for v in out["byNode"].values():
            v["literalShare"] = out["literalShare"]
    return out


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
