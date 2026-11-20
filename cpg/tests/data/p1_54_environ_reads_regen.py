"""Regenerate ``p1_54_environ_reads.json`` from real extractor output (P1-54 golden; the one reproducible source of those records).

Usage, from ``cpg/``::

    .venv/bin/python tests/data/p1_54_environ_reads_regen.py <bundles-dir> tests/data/p1_54_environ_reads.json

``<bundles-dir>`` holds ``*/*.bundle.jsonl`` files read from a real ``ccg.build.build_units`` run (the sandbox runner, ``area/work/bundles``)
over the materialized fixture system (``fixtures/materialize.py``) in which ``fx-svc/clients/peer.py`` was rewritten to the ten probe
shapes ``shape_a`` .. ``shape_j``: one function per shape, named so (``os.environ.get``, f-string, local, ``os.getenv``, ``dict.get``,
non-empty default, ``.rstrip`` chain, ``.setdefault``, ``from os import environ`` and ``getenv``). ``jobs`` is
``fx-svc/clients/jobs.py`` as committed. The bundle that carries ``clients/peer.py`` is the one read.
"""
import glob
import json
import sys

SHAPES = ("shape_a", "shape_b", "shape_c", "shape_d", "shape_e", "shape_f", "shape_g", "shape_h", "shape_i", "shape_j")


def main(bundles_dir: str, out_path: str) -> None:
    recs = []
    for f in glob.glob(f"{bundles_dir}/*/*.bundle.jsonl"):
        rr = [json.loads(line) for line in open(f)]
        if any(r.get("type") == "file" and r.get("path") == "clients/peer.py" for r in rr):
            recs = rr
    syms = {r["id"]: r for r in recs if r["type"] == "symbol"}
    cs = {r["id"]: r for r in recs if r["type"] == "callsite"}
    vss = [r for r in recs if r["type"] == "valueSet"]
    dat = [r for r in recs if r["type"] == "data"]
    out = {"source": "real extractor output (Joern, ccg.build over the fixture system) with fx-svc clients/peer.py rewritten to shapes a-j; "
                     "jobs = fx-svc/clients/jobs.py as committed; regenerate with tests/data/p1_54_environ_reads_regen.py", "shapes": {}}

    def take(ids):
        return {"callsites": [cs[i] for i in ids], "valueSets": [v for v in vss if v["callSiteId"] in ids],
                "data": [d for d in dat if d["use"] in ids and d["def"] in ids]}

    for sname in SHAPES:
        ids = [i for i, c in cs.items() if syms.get(c["symbolId"], {}).get("name") == sname]
        t = take(ids)
        out["shapes"][sname] = {"symbolId": next(c["symbolId"] for c in t["callsites"]), **t}
    mod = [s["id"] for s in syms.values() if s["qname"] == "clients/jobs.py::<module>"][0]
    ids = [i for i, c in cs.items() if c["symbolId"] == mod and "environ" in (c.get("externalTarget") or "")]
    out["jobs"] = {"symbolId": mod, "callsites": [cs[i] for i in ids], "valueSets": [v for v in vss if v["callSiteId"] in ids],
                   "data": [d for d in dat if d["def"] in ids]}
    json.dump(out, open(out_path, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
