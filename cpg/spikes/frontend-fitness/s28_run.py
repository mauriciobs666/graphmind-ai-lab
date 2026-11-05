#!/usr/bin/env python3
"""S28 (P0-21) projection runs for ONE language: S11's extract.sc through the sandbox runner, both configurations.

    cd cpg && .venv/bin/python spikes/frontend-fitness/s28_run.py <lang> [--only off|on]

Reuses the CPGs S27 built (`s27-run.json`, `<work>/joern/<lang>/{off,on}`): per configuration it runs the D15 content
check (cpg-content-check.sc) and then extract.sc, both through offline-run.sh in `--mode offline`, exactly as
`s27_run.py` runs measure.sc. extract.sc is not edited. The record goes to `<work>/joern/<lang>/s28-run.json`.
Opens no label, truth or adjudication file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s27_run as s27  # noqa: E402

CPG = s27.CPG
GUARD = HERE / "s28-extract-bodiless-guard.sc"
EXTRACT = CPG / "spikes" / "vertical-slice" / "extract.sc"
WORK = CPG / ".ccg-work" / "frontend-fitness" / "joern"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    ap.add_argument("--only", choices=["off", "on"])
    ap.add_argument("--guard", action="store_true", help="run s28-extract-bodiless-guard.sc (extract.sc + a one-line bodiless-METHOD guard); records under the *.extract-guard keys")
    ap.add_argument("--stats", action="store_true", help="run s28-cpg-stats.sc (whole-CPG node/edge counts) on each CPG; record key <cfg>.cpg-stats")
    a = ap.parse_args()
    lang = a.lang
    base = WORK / lang
    stage = base / "stage"
    s27rec = json.loads((base / "s27-run.json").read_text())
    rec_path = base / "s28-run.json"
    record = json.loads(rec_path.read_text()) if rec_path.exists() else {"lang": lang, "runs": {}}
    for cfg in ("off", "on"):
        if a.only not in (None, cfg):
            continue
        p = s27rec["runs"][f"{cfg}.parse"]
        if not p.get("cpgFile"):
            print(f"{cfg}: no CPG", flush=True)
            continue
        cpgname = Path(p["cpgFile"]).name
        out = base / cfg
        if a.stats:
            sc = HERE / "s28-cpg-stats.sc"
            r = s27.run([s27.RUNNER, "--backend", "container", "--mode", "offline", "--frontend", "joern", "--script-file", str(sc), "--source", str(stage),
                         "--output", str(out), "--", "--script", "/scripts/" + sc.name, "--param", f"cpgFile=/out/{cpgname}"], out / "s28-stats.log", f"{cfg}.cpg-stats")
            r["lines"] = [l for l in (out / "s28-stats.log").read_text().splitlines() if l.startswith("CPGSTATS")]
            record["runs"][f"{cfg}.cpg-stats"] = r
            rec_path.write_text(json.dumps(record, indent=1))
            continue
        r = s27.run([s27.RUNNER, "--backend", "container", "--mode", "offline", "--frontend", "joern",
                     "--script-file", str(CPG / "sandbox" / "scripts" / "cpg-content-check.sc"), "--source", str(stage),
                     "--output", str(out), "--", "--script", "/scripts/cpg-content-check.sc", "--param", f"cpgFiles=/out/{cpgname}"],
                    out / "s28-content.log", f"{cfg}.content-check")
        txt = (out / "s28-content.log").read_text()
        r["contentLines"] = [l for l in txt.splitlines() if l.startswith(("METHOD_COUNT", "CONTENT"))]
        record["runs"][f"{cfg}.content-check"] = r
        ok = r["rc"] == 0 and any(l.endswith("=ok") for l in r["contentLines"])
        rec_path.write_text(json.dumps(record, indent=1))
        if not ok:
            print(f"{cfg}: content check failed -> no projection", flush=True)
            continue
        import time
        t = time.time()
        r = s27.run([s27.RUNNER, "--backend", "container", "--mode", "offline", "--frontend", "joern",
                     "--script-file", str(GUARD if a.guard else EXTRACT), "--source", str(stage), "--output", str(out), "--",
                     "--script", "/scripts/" + (GUARD if a.guard else EXTRACT).name, "--param", f"cpgFile=/out/{cpgname}",
                     "--param", "out=/out/@AUDIT_ID@.facts.jsonl", "--param", f"lang={lang}"],
                    out / ("s28-extract-guard.log" if a.guard else "s28-extract.log"), f"{cfg}.extract" + ("-guard" if a.guard else ""))
        f = s27.newest(out, "*.facts.jsonl", t)
        r["factsFile"] = str(f) if f else None
        r["factsMB"] = round(f.stat().st_size / 1e6, 2) if f else None
        r["cpgFile"] = p["cpgFile"]
        r["cpgMB"] = p.get("cpgMB")
        record["runs"][f"{cfg}.extract" + ("-guard" if a.guard else "")] = r
        rec_path.write_text(json.dumps(record, indent=1))
    print("record:", rec_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
