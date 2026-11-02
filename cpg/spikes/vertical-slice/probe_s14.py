#!/usr/bin/env python3
"""S14 (P0-12) report probe: measurements the slice report quotes that no earlier script emits.

Read-only. Bundle-side checks need only the two bundles of a run directory; the BFS item-set checks
read the loaded ``ccg_p0slice`` on the dedicated ``ccg-falkordb`` (reservation R-B, reads only).

    cd cpg && .venv/bin/python spikes/vertical-slice/probe_s14.py [RUN_DIR] [--no-db] > probe.json

Sections of the JSON written to stdout:

* ``files``     -- File-record counts per unit against the non-empty source files in the staged tree.
* ``dataSelf``  -- ``data`` records with ``use == def`` per unit (the loader keeps them).
* ``tests``     -- providers and mounts that live under ``tests/`` (S12 finding 3).
* ``symbolHops``-- whether any symbol hop of a link chain took ``_symbol_hop``'s ``lineStart`` fallback (N16a).
* ``i18n``      -- the ``src/i18n`` files the JS bundle lacks and the call sites that name their exports.
* ``pslice``    -- bundle-spec-§10 [P-slice] evidence the bundles can show (counts per feature, an independent
                   fingerprint recomputation from ``git show``, Protocol/ABC stubs, ``{#}`` keys).
* ``links``     -- per link: status, hop counts, chain, candidateCount.
* ``bfs``       -- (live) item sets and state counts for the apiFetch ``p0`` anchors, as built (V0) and under
                   a *what-if* gate (V2: ``HAS_*`` <- from a non-``Ret`` point reports the function but does
                   not walk it further). V2 is an experiment for graph-dba, not a proposal of this script.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPG = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CPG))

import impact as I  # noqa: E402
import load as L  # noqa: E402
from assemble import assemble  # noqa: E402
from ccg.bundle import read_bundle  # noqa: E402

COMMIT = "defca60ff16534aa12a74e35856de9be98bfc710"
SRC_EXT = {"falkor-chat/server": (".py",), "salesperson": (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")}
STAGE = {"falkor-chat/server": "fcs", "salesperson": "sp"}


def _nonempty_sources(stage: Path, exts: tuple[str, ...]) -> list[str]:
    return sorted(str(p.relative_to(stage)) for p in stage.rglob("*")
                  if p.is_file() and p.suffix in exts and p.stat().st_size > 0 and "node_modules" not in p.parts)


def files_section(run: Path, bundles: dict) -> dict:
    out = {}
    for root, recs in bundles.items():
        stage = run / STAGE[root] / "stage"
        src = _nonempty_sources(stage, SRC_EXT[root])
        file_recs = [r for r in recs if r["type"] == "file"]
        out[root] = {"fileRecords": len(file_recs), "nonEmptySourceFilesInStage": len(src),
                     "fileRecordsByKind": dict(Counter(r["kind"] for r in file_recs))}
    return out


def data_self_section(bundles: dict) -> dict:
    out = {}
    for root, recs in bundles.items():
        labels = {r["id"]: r["type"] for r in recs if r["type"] in ("param", "ret", "cond", "callsite")}
        selfs = [r for r in recs if r["type"] == "data" and r["use"] == r["def"]]
        out[root] = {"dataRecords": sum(1 for r in recs if r["type"] == "data"), "useEqualsDef": len(selfs),
                     "examples": [{"kind": labels.get(r["use"]), "fromPath": r["fromPath"], "toPath": r["toPath"],
                                   "status": r["status"], "reason": r["reason"], "id": r["id"]} for r in selfs[:5]]}
    return out


def tests_section(facts: dict) -> dict:
    provs = facts["providers"]
    under_tests = [p for p in provs if "/tests/" in p["path"]]
    mount_paths = [(p["qname"], s["path"]) for p in provs for m in p["mounts"] for s in m["sites"]]
    return {"providers": len(provs), "providersUnderTests": len(under_tests),
            "providersUnderTestsByStatus": dict(Counter(p["status"] for p in under_tests)),
            "providersByStatus": dict(Counter(p["status"] for p in provs)),
            "providersMountedFromATestFile": sorted({q for q, path in mount_paths if "/tests/" in path}),
            "providersMountedOnlyFromTestFiles": sorted(
                p["qname"] for p in provs
                if p["mounts"] and all(s["path"].split("/")[2:3] == ["tests"] for m in p["mounts"] for s in m["sites"])
                and any(m["sites"] for m in p["mounts"])),
            "droppedAlternatives": facts["droppedProviderAlternatives"]}


def symbol_hops_section(bundles: dict, facts: dict) -> dict:
    params = {}
    syms = {}
    for recs in bundles.values():
        for r in recs:
            if r["type"] == "param":
                params[(r["symbolId"], r["index"])] = r["line"]
            elif r["type"] == "symbol":
                syms[r["id"]] = r
    rows = []
    for link in facts["links"]:
        for site in link["sites"]:
            for h in site["liftedVia"]:
                if h["kind"] != "symbol":
                    continue
                idx = int(re.match(r"p(\d+)", h["urlPath"]).group(1))
                has_param = (h["id"], idx) in params
                rows.append({"hasParamRecord": has_param, "lineIsParamLine": h["line"] == params.get((h["id"], idx)),
                             "line": h["line"], "lineStart": syms[h["id"]]["lineStart"]})
    return {"symbolHops": len(rows), "withoutParamRecord(fallback taken)": sum(1 for r in rows if not r["hasParamRecord"]),
            "lineDiffersFromLineStart": sum(1 for r in rows if r["line"] != r["lineStart"])}


def i18n_section(bundles: dict) -> dict:
    sp = bundles["salesperson"]
    files = [r["path"] for r in sp if r["type"] == "file"]
    git = subprocess.run(["git", "ls-tree", "-r", "--name-only", COMMIT, "salesperson/src/i18n"],
                         capture_output=True, text=True, cwd=CPG.parent, check=True).stdout.split()
    prod = [p for p in git if not re.search(r"\.(test|spec)\.", p)]
    exported = set()
    for p in prod:
        text = subprocess.run(["git", "show", f"{COMMIT}:{p}"], capture_output=True, text=True, cwd=CPG.parent,
                              check=True).stdout
        exported |= set(re.findall(r"export\s+(?:default\s+)?(?:async\s+)?(?:function|const|class)\s+(\w+)", text))
    sites = Counter(r["calleeName"] for r in sp if r["type"] == "callsite" and r["calleeName"] in exported)
    untargeted = sum(1 for r in sp if r["type"] == "callsite" and r["calleeName"] in exported and not r["targets"])
    return {"i18nSourceFilesAtCommit": len(git), "productionI18nFiles": prod,
            "fileRecordsUnderI18n": [f for f in files if "/i18n/" in f],
            "exportedNamesOfProductionI18n": sorted(exported),
            "callSitesNamingThoseExports": dict(sites), "...of which with no targets": untargeted}


def _git_show(path: str) -> str:
    return subprocess.run(["git", "show", f"{COMMIT}:{path}"], capture_output=True, cwd=CPG.parent,
                          check=True).stdout.decode("utf-8")


def fingerprint_check(bundles: dict) -> dict:
    """Recompute every symbol's ``fingerprint`` (spec §5.3) from the pinned commit, sharing no code with normalize.py."""
    import hashlib
    out = {}
    for root, recs in bundles.items():
        files = {r["id"]: r["path"] for r in recs if r["type"] == "file"}
        cache: dict[str, list[str]] = {}
        bad = total = 0
        for r in recs:
            if r["type"] != "symbol":
                continue
            path = files[r["fileId"]]
            lines = cache.setdefault(path, _git_show(path).split("\n"))
            picked = list(lines[r["lineStart"] - 1:r["lineEnd"]])
            for a, b in r.get("ownedRegions") or []:
                picked += lines[a - 1:b]
            text = "\n".join(x for x in (l.strip() for l in picked) if x)
            total += 1
            bad += hashlib.sha256(text.encode("utf-8")).hexdigest() != r["fingerprint"]
        out[root] = {"symbols": total, "fingerprintMismatches": bad}
    return out


def protocol_stubs(bundles: dict) -> dict:
    """Methods with an ``...``/docstring-only body inside a ``Protocol``/``ABC`` class of ``falkor-chat/server/falkorchat``."""
    import ast
    recs = bundles["falkor-chat/server"]
    qnames = {r["qname"] for r in recs if r["type"] == "symbol"}
    paths = sorted({r["path"] for r in recs if r["type"] == "file" and r["path"].startswith("falkor-chat/server/falkorchat/")})
    found = []
    for path in paths:
        tree = ast.parse(_git_show(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {ast.unparse(b).split(".")[-1].split("[")[0] for b in node.bases}
            if not bases & {"Protocol", "ABC"}:
                continue
            for fn in node.body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = [b for b in fn.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                                       and isinstance(b.value.value, str))]
                    if not body or all(isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and b.value.value is Ellipsis
                                       for b in body):
                        found.append(f"{path}::{node.name}.{fn.name}")
    return {"stubMethodsInProtocolOrABCClasses": len(found), "allHaveASymbol (a property's symbol is `X.name.<get>`)": all(q in qnames or q + ".<get>" in qnames for q in found),
            "examples": found[:6]}


def pslice_section(bundles: dict, facts: dict) -> dict:
    out = {"fingerprints": fingerprint_check(bundles), "protocolStubs": protocol_stubs(bundles)}
    for root, recs in bundles.items():
        hdr = next(r for r in recs if r["type"] == "header")
        forms = Counter(next(iter(e)) for r in recs if r["type"] == "valueSet" for e in r["values"])
        params = [r for r in recs if r["type"] == "param"]
        out[root] = {
            "headerKeys": sorted(hdr), "fieldDepthCap": hdr["fieldDepthCap"], "vendoredDirs": hdr["vendoredDirs"],
            "counts": hdr["counts"], "valueSetElementForms": dict(forms),
            "paramIndexMinusOne": sum(1 for r in params if r["index"] == -1),
            "dataReasons": dict(Counter(f"{r['status']}/{r['reason']}" for r in recs if r["type"] == "data")),
            "dataFromPathHeads": dict(Counter(re.split(r"[.\[]", r["fromPath"])[0] for r in recs if r["type"] == "data")),
            "dataToPathHeads": dict(Counter(re.split(r"[.\[]", r["toPath"])[0] for r in recs if r["type"] == "data")),
            "kwPathValueSets": sum(1 for r in recs if r["type"] == "valueSet" and r["arg"].startswith("kw.")),
            "symbolKinds": dict(Counter(r["kind"] for r in recs if r["type"] == "symbol")),
            "anonSymbols": sum(1 for r in recs if r["type"] == "symbol" and "<anon>" in r["qname"]),
            "symbolsWithOwnedRegions": sum(1 for r in recs if r["type"] == "symbol" and r.get("ownedRegions")),
            "fileRoles": dict(Counter(r["role"] for r in recs if r["type"] == "file")),
            "fileKinds": dict(Counter(r["kind"] for r in recs if r["type"] == "file")),
            "importRecords": sum(1 for r in recs if r["type"] == "import"),
        }
    keys = [p["key"] for p in facts["providers"]] + [k for c in facts["consumers"] for k in c["keys"]]
    out["httpKeys"] = {"providerKeys": len(facts["providers"]), "keysWithMaskPlaceholder{#}": sum("{#}" in k for k in keys),
                       "keysWithConsumerPlaceholder{?}": sum("{?}" in k for k in keys),
                       "keysWithPathParam{}": sum("{}" in k for k in keys)}
    return out


def links_section(facts: dict) -> dict:
    import render  # D6: candidateCount left the Link; N is derived per VIA edge
    via = [(l["consumer"]["anchorId"], s["siteId"], l["id"], s["consumerKey"]) for l in facts["links"] for s in l["sites"]]
    rows = []
    for l in facts["links"]:
        for s in l["sites"]:
            rows.append({"keyNorm": l["keyNorm"], "consumer": l["consumer"]["qname"].split("::")[1], "status": l["status"],
                         "liftedViaHops": len(s["liftedVia"]), "candidateCount": render.candidate_count(via, s["siteId"], l["id"], s["consumerKey"]),
                         "sitesOnLink": len(l["sites"]),
                         "chain": [(h.get("callee") or h["qname"].split("::")[1]) + f"@{h['path'].split('/')[-1]}:{h['line']}"
                                   for h in s["liftedVia"]]})
    return {"byStatus": dict(Counter(l["status"] for l in facts["links"])),
            "byReason": dict(Counter(l["reason"] for l in facts["links"])),
            "byMatcher": dict(Counter(l["matcher"] for l in facts["links"])),
            "liftedViaHopHistogram": dict(Counter(r["liftedViaHops"] for r in rows)), "rows": rows}


# ---------------------------------------------------------------------------------------------
# BFS (live, read-only).
# ---------------------------------------------------------------------------------------------


class Gate:
    """V2 what-if: a ``HAS_*`` <- step from a non-``Ret`` point reports the function but ends there."""

    def __enter__(self):
        self.orig = I._expand

        def gated(reader, frontier):
            out = []
            for frm, node, st, edge, dp, dd in self.orig(reader, frontier):
                if edge.type in I.OWNER_REL.values() and edge.direction == "<-" and frm[0] != "Ret" and st == I.VALUE:
                    st = I.END
                out.append((frm, node, st, edge, dp, dd))
            return out

        I._expand = gated
        return self

    def __exit__(self, *exc):
        I._expand = self.orig


def bfs_section(store, bundles: dict) -> dict:
    run = lambda q, p=None: store.query(L.FINAL_KEY, q, p).result_set  # noqa: E731
    reader = I.CypherReader(run)

    def sym(qname):
        return run("MATCH (s:Symbol {qname: $q}) RETURN s.symbolId", {"q": qname})[0][0]

    api, bu = sym("salesperson/src/api/client.ts::apiFetch"), sym("salesperson/src/api/client.ts::buildUrl")
    p0 = run("MATCH (:Symbol {symbolId: $s})-[:HAS_PARAM]->(p:Param {index: 0}) RETURN p.pointId", {"s": api})[0][0]
    paths = {sid: p for sid, p in run("MATCH (s:Symbol) RETURN s.symbolId, s.path")}

    def summarize(res):
        items = res.items()
        by_file = Counter(paths[i.symbol_id] for i in items)
        return {"items": len(items), "itemSetSha": __import__("hashlib").sha256(
                    "\n".join(sorted(i.symbol_id for i in items)).encode()).hexdigest()[:12],
                "itemsInEndpointsTs": by_file.get("salesperson/src/api/endpoints.ts", 0),
                "itemsByFileTop": by_file.most_common(4), **I.state_summary(res)}

    def sets(label):
        out, ids = {}, {}
        for path in ("p0", "p0.path", "p0.method", "p0.query", "p0.nothing"):
            res = I.walk(reader, ("Param", p0), ("path", path))
            out[f"param apiFetch.p0 @ {path}"] = summarize(res)
            ids[path] = {i.symbol_id for i in res.items()}
        res = I.walk(reader, ("Symbol", bu))
        out["value buildUrl"] = summarize(res)
        out["value buildUrl"]["listsApiFetch"] = api in {i.symbol_id for i in res.items()}
        out["itemSetsEqualAcrossPaths(p0,p0.path,p0.method)"] = ids["p0"] == ids["p0.path"] == ids["p0.method"]
        return out

    v0 = sets("V0")
    with Gate():
        v2 = sets("V2")
    return {"V0_as_built": v0, "V2_what_if_gate": v2}


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    run = Path(argv[0]) if argv else CPG / ".ccg-work" / "vertical-slice" / "s14a"
    bundles = {}
    for p in sorted(run.glob("*.bundle.jsonl")):
        recs = read_bundle(p)
        bundles[next(r for r in recs if r["type"] == "header")["unit"]["rootPath"]] = recs
    facts = assemble([bundles["falkor-chat/server"], bundles["salesperson"]])
    out = {"run": str(run), "files": files_section(run, bundles), "dataSelf": data_self_section(bundles),
           "tests": tests_section(facts), "symbolHops": symbol_hops_section(bundles, facts),
           "i18n": i18n_section(bundles), "pslice": pslice_section(bundles, facts), "links": links_section(facts)}
    if "--no-db" not in sys.argv:
        out["bfs"] = bfs_section(L.LiveStore("127.0.0.1", 6390), bundles)
    json.dump(out, sys.stdout, indent=1, sort_keys=True, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
