#!/usr/bin/env python3
"""S27: inspect the UNMATCHED labelled sites of one language (ML §7.1), per configuration, through the sandbox runner.

    cd cpg && .venv/bin/python spikes/frontend-fitness/s27_probe.py <lang>

Reads results/<lang>.json (`unmatchedLabelled`, written by s27_metrics.py) and <work>/joern/<lang>/s27-run.json; runs
`unmatched-probe.sc` on each configuration's CPG; writes results/<lang>.unmatched.json. The classification is mechanical:
`absent-no-call-on-line` (no CALL at all on that file/line) or `calls-present` (the calls found are listed; a call whose
name equals calleeName or starts with it flags `possibleMatcherError`, for a human look). Opens no label or truth file.
"""
import json, subprocess, sys, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPG = HERE.parent.parent
sys.path.insert(0, str(HERE))
lang = sys.argv[1]
WORK = CPG / ".ccg-work" / "frontend-fitness" / "joern" / lang
res = json.loads((HERE / "results" / f"{lang}.json").read_text())
rec = json.loads((WORK / "s27-run.json").read_text())
priv = {json.loads(l)["siteKey"]: json.loads(l) for l in open(HERE / "sample" / f"{lang}.private.jsonl")}
lineoff = -1 if lang == "csharp" else 0
out = {"lang": lang, "configs": {}}
for cfg in ("off", "on"):
    sites = res["unmatchedLabelled"][cfg]
    if not sites:
        out["configs"][cfg] = []
        continue
    stage = WORK / f"probe-stage-{cfg}"
    stage.mkdir(exist_ok=True)
    lines = []
    for s in sites:
        p = s["siteKey"]
        path = p.split(":")[0]
        lo = int(p[len(path) + 1:].split(":")[0])
        lines.append(f"{path}\t{lo}\t{priv[p]['nameLine']}")
    (stage / "unmatched.tsv").write_text("\n".join(sorted(set(lines))) + "\n")
    cpgname = Path(rec["runs"]["off.parse" if cfg == "off" else "on.parse"]["cpgFile"]).name
    odir = WORK / cfg
    before = set(glob.glob(str(odir / "*.probe.jsonl")))
    cmd = [str(CPG / "sandbox" / "offline-run.sh"), "--backend", "container", "--mode", "offline", "--frontend", "joern",
           "--script-file", str(HERE / "unmatched-probe.sc"), "--source", str(stage), "--output", str(odir), "--",
           "--script", "/scripts/unmatched-probe.sc", "--param", f"cpgFile=/out/{cpgname}", "--param", "sites=/src/unmatched.tsv",
           "--param", f"lineOff={lineoff}", "--param", "out=/out/@AUDIT_ID@.probe.jsonl"]
    with open(odir / "probe.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    new = sorted(set(glob.glob(str(odir / "*.probe.jsonl"))) - before)
    probes = {}
    if rc == 0 and new:
        for l in open(new[-1]):
            r = json.loads(l)
            probes[(r["path"], r["lo"], r["hi"])] = r
    rows = []
    for s in sites:
        p = priv[s["siteKey"]]
        path0 = s["siteKey"].split(":")[0]
        pr = probes.get((path0, int(s["siteKey"][len(path0)+1:].split(":")[0]), p["nameLine"]))
        callee = p["calleeName"]
        if pr is None:
            cls = "probe-failed"
        elif not pr["calls"]:
            cls = "absent-no-call-on-line" if pr["fileNode"] else "absent-no-file-node"
        else:
            cls = "calls-present"
        flag = bool(pr and any(c["name"] == callee or c["name"].startswith(callee) or c["name"].endswith("." + callee) or c["name"].endswith("?." + callee) for c in pr["calls"]))
        rows.append({**s, "calleeName": callee, "nameLine": p["nameLine"], "classification": cls,
                     "possibleMatcherError": flag and cls == "calls-present", "fileNode": pr and pr["fileNode"],
                     "callsOnLine": pr and pr["calls"], "probeRc": rc})
    out["configs"][cfg] = rows
(HERE / "results" / f"{lang}.unmatched.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({c: [(r["siteKey"], r["classification"], r["possibleMatcherError"]) for r in v] for c, v in out["configs"].items()}, indent=1))
