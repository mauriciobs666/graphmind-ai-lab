"""Regenerate ``p1_57a_fetch_sites.json`` from real extractor output (P1-57a golden: the real JS/TS ``fetch(...)`` sites).

Usage, from ``cpg/``::

    .venv/bin/python tests/data/p1_57a_fetch_sites_regen.py <bundles-dir> tests/data/p1_57a_fetch_sites.json

``<bundles-dir>`` holds ``*.bundle.jsonl`` files from a real ``ccg.build.build_units`` run over this repository (the stored proving-repository
bundles, ``.ccg-work/bundles/graphmind-ai-lab``). Kept, per bundle: every ``callsite`` named ``fetch`` whose file path ends ``.ts`` or ``.js``
(`salesperson/src/api/client.ts:118`, `falkor-chat/web/app.js:37`), with its file, owning symbol, ``valueSet`` and ``data`` records, so the file
``path`` shape the ``fetch`` pack's ``scope.path`` reads is the extractor's own. Nothing is hand-edited. Each bundle's ``builtCommit`` (its
``<unitId>.state.json``) is written into the JSON's ``source``; the golden was taken at ``24cdff79dc87377c06e8330d4159c9eba4c85ad2``.
"""
import glob
import json
import sys


def main(bundles_dir: str, out_path: str) -> None:
    keep, commits = [], set()
    for f in sorted(glob.glob(f"{bundles_dir}/*.bundle.jsonl")):
        recs = [json.loads(line) for line in open(f)]
        syms = {r["id"]: r for r in recs if r["type"] == "symbol"}
        files = {r["id"]: r for r in recs if r["type"] == "file"}
        sites = [r for r in recs if r["type"] == "callsite" and r["calleeName"] == "fetch"
                 and files[syms[r["symbolId"]]["fileId"]]["path"].endswith((".ts", ".js"))]
        if not sites:
            continue
        ids = {s["id"] for s in sites}
        keep += [files[syms[s["symbolId"]]["fileId"]] for s in sites] + [syms[s["symbolId"]] for s in sites] + sites
        keep += [r for r in recs if r["type"] == "valueSet" and r["callSiteId"] in ids]
        keep += [r for r in recs if r["type"] == "data" and (r["use"] in ids or r["def"] in ids)]
        commits.add(json.load(open(f.replace(".bundle.jsonl", "").rsplit(".", 1)[0] + ".state.json"))["builtCommit"])
    uniq = list({r["id"]: r for r in keep}.values())
    json.dump({"source": f"real extractor output (ccg.build over this repository, builtCommit {', '.join(sorted(commits))}); "
                         "regenerate with tests/data/p1_57a_fetch_sites_regen.py", "records": uniq},
              open(out_path, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
