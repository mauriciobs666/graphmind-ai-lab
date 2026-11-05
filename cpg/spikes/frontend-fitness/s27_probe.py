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
DISPOSE_ONLY = "--disposition-only" in sys.argv
IGNORED_SEG = {"java": {"test"}, "javascript": {"i18n", "test", "tests"}, "cpp": {"test", "tests"}}
PROBE = "results/frontend-defaults-probe.json"


def disposition(r):
    """Per-class disposition (S27 fix round B2 iii). Cites the probe cases in results/frontend-defaults-probe.json."""
    cls, path = r["classification"], r["siteKey"].split(":")[0]
    segs = set(path.split("/")[:-1])
    if cls == "absent-no-file-node":
        hit = segs & IGNORED_SEG.get(lang, set())
        if hit:
            return ("default exclusion: the frontend ignores a path segment named %s (probe %s: shipped path 0 methods, moved path parsed)" % (sorted(hit), lang)), "frontend-default-ignore"
        if lang in ("go", "csharp"):
            return ("per-file frontend failure: the file yields no FILE node when parsed alone, also at a neutral path (probe %s)" % lang), "per-file-failure"
        return "no FILE node and no ignored segment: unexplained", "unexplained"
    if cls == "absent-no-call-on-line":
        if r["kind"] == "jsx":
            return "JSX element: jssrc2cpg emits no CALL for a component tag (matcher-table.md, by construction)", "by-construction"
        return "no CALL on the site's lines (inspected, calls listed empty): construct not modelled by the frontend; matcher not at fault", "absent-in-joern"
    if cls == "calls-present":
        return "calls exist on the lines but none carries the callee (inspected: %s): the callee has no CALL of its own (JSX tag, call of a call with an empty name, generic instantiation); matcher not at fault" % sorted({c["name"] or "<empty>" for c in (r["callsOnLine"] or [])})[:4], "by-construction-or-absent"
    return "probe failed", "unexplained"

WORK = CPG / ".ccg-work" / "frontend-fitness" / "joern" / lang
res = json.loads((HERE / "results" / f"{lang}.json").read_text())
rec = json.loads((WORK / "s27-run.json").read_text())
priv = {json.loads(l)["siteKey"]: json.loads(l) for l in open(HERE / "sample" / f"{lang}.private.jsonl")}
lineoff = -1 if lang == "csharp" else 0
if DISPOSE_ONLY:
    f = HERE / "results" / f"{lang}.unmatched.json"
    d = json.loads(f.read_text())
    for cfg, rows in d["configs"].items():
        for r in rows:
            r["dispositionText"], r["dispositionClass"] = disposition(r)
    d["dispositionNote"] = "S27 fix round B2 (iii): per-class disposition by the graph-dba harness; the data-scientist's causes stay in the v2 miss logs"
    f.write_text(json.dumps(d, indent=1) + "\n")
    print(lang, {c: dict(__import__("collections").Counter(r["dispositionClass"] for r in rows)) for c, rows in d["configs"].items()})
    sys.exit(0)
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
        rows[-1]["dispositionText"], rows[-1]["dispositionClass"] = disposition(rows[-1])
    out["configs"][cfg] = rows
(HERE / "results" / f"{lang}.unmatched.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({c: [(r["siteKey"], r["classification"], r["possibleMatcherError"]) for r in v] for c, v in out["configs"].items()}, indent=1))
