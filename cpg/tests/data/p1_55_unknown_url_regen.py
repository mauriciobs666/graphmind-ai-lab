"""Regenerate ``p1_55_unknown_url.json`` from real extractor output (P1-55 golden: the one real wholly-unknown-URL consumer site).

Usage, from ``cpg/``::

    .venv/bin/python tests/data/p1_55_unknown_url_regen.py <bundles-dir> tests/data/p1_55_unknown_url.json

``<bundles-dir>`` holds ``*.bundle.jsonl`` files from a real ``ccg.build.build_units`` run over this repository (the stored proving-repository
bundles, ``.ccg-work/bundles/graphmind-ai-lab``). The bundle read is the one that carries ``decision-models/decisionmodels/lmstudio.py``; the
site is the ``urllib.request.Request(f"{self._base_url}{_MODELS_ROUTE}", method="GET")`` callsite in ``LMStudioClient.residency`` (line 366), whose
URL valueSet is ``template [unknown, unknown]``. Kept: the file, the owning symbol, the ``urllib`` imports of the file, that callsite and its
``valueSet`` and ``data`` records. Nothing is hand-edited.
"""
import glob
import json
import sys

PATH = "decision-models/decisionmodels/lmstudio.py"
LINE = 366
TARGET = "urllib.py:<module>.<member>(request).Request.__init__"


def main(bundles_dir: str, out_path: str) -> None:
    for f in sorted(glob.glob(f"{bundles_dir}/*.bundle.jsonl")):
        recs = [json.loads(line) for line in open(f)]
        files = [r for r in recs if r["type"] == "file" and r["path"] == PATH]
        if not files:
            continue
        fid = files[0]["id"]
        syms = {r["id"]: r for r in recs if r["type"] == "symbol"}
        (site,) = [r for r in recs if r["type"] == "callsite" and r["line"] == LINE and r.get("externalTarget") == TARGET
                   and syms[r["symbolId"]]["path"] == PATH]
        keep = [files[0], syms[site["symbolId"]], site]
        keep += [r for r in recs if r["type"] == "import" and r["fileId"] == fid]
        keep += [r for r in recs if r["type"] == "valueSet" and r["callSiteId"] == site["id"]]
        keep += [r for r in recs if r["type"] == "data" and site["id"] in (r["use"], r["def"])]
        json.dump({"source": "real extractor output (ccg.build over this repository); regenerate with tests/data/p1_55_unknown_url_regen.py",
                   "records": keep}, open(out_path, "w"), indent=1, sort_keys=True)
        return
    raise SystemExit("no bundle carries " + PATH)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
