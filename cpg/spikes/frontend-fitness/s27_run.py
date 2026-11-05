#!/usr/bin/env python3
"""S27 (P0-20) Joern runs for ONE language, both ML §7 configurations (off / on).

    cd cpg && .venv/bin/python spikes/frontend-fitness/s27_run.py <lang> [--only off|on] [--on-network]

Sequence (plan p0-2 §4.6 S25 item 6, S27): stage (`run_fitness.py joern <lang>` does the staging), then
  off.parse         offline-run.sh --mode offline, from the clean `git archive` corpus tree
  off.content-check cpg-content-check.sc through the runner (plan D15)
  off.measure       measure.sc through the runner
  on.parse          toolchain image, network ON, the language's fetch flag (FETCH_ON_FLAG; none for python/javascript/cpp)
  on.content-check, on.measure   through the offline runner (the CPG is a local file)
Everything is timed; the record goes to <work>/joern/<lang>/s27-run.json. Opens no label, truth or adjudication
file. Reads nothing but the commands `run_fitness.py` prints. Peak container memory is sampled with `docker stats`
on containers carrying a `ccg.sandbox.auditId` label (the runner's) or the label `s27=<lang>` (the on-parse).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPG = HERE.parent.parent
sys.path.insert(0, str(HERE))
import run_fitness as rf  # noqa: E402

RUNNER = str(CPG / "sandbox" / "offline-run.sh")


class MemPoller(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mib = 0.0
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                ids = subprocess.run(["docker", "ps", "-q", "--filter", "label=s27"], capture_output=True, text=True).stdout.split()
                ids += subprocess.run(["docker", "ps", "-q", "--filter", "label=ccg.sandbox.auditId"], capture_output=True, text=True).stdout.split()
                if ids:
                    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}"] + ids, capture_output=True, text=True).stdout
                    for line in out.splitlines():
                        m = re.match(r"([\d.]+)(KiB|MiB|GiB)", line.strip())
                        if m:
                            v = float(m.group(1)) * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[m.group(2)]
                            self.peak_mib = max(self.peak_mib, v)
            except Exception:
                pass
            time.sleep(2)


def run(argv, log: Path, label: str) -> dict:
    poller = MemPoller()
    poller.start()
    t0 = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))
    with log.open("w") as fh:
        p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    poller.stop = True
    poller.join(timeout=10)
    rec = {"stage": label, "argv": argv[:6] + ["..."], "rc": p.returncode, "startedUtc": started_utc,
           "wallSeconds": round(wall, 1), "peakMemMiB": round(poller.peak_mib, 1), "log": str(log)}
    print(json.dumps(rec), flush=True)
    return rec


def newest(d: Path, pattern: str, since: float) -> Path | None:
    c = [p for p in d.glob(pattern) if p.stat().st_mtime >= since - 1]
    return max(c, key=lambda p: p.stat().st_mtime) if c else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    ap.add_argument("--only", choices=["off", "on"])
    ap.add_argument("--measure-only", action="store_true", help="re-run the content check and measure.sc on the CPGs already built (a matcher fix)")
    ap.add_argument("--nooverlays", action="store_true", help="joern-parse --nooverlays (Kotlin: ReachingDefPass aborts the default-overlay stage with exit 1)")
    ap.add_argument("--heap", default="", help="extra -J-Xmx for joern-parse frontends, e.g. 6g")
    a = ap.parse_args()
    lang = a.lang
    info = rf.cmd_joern(lang)  # stages frame/sample.private, prints nothing
    base = Path(info["stage"]).parent
    stage = Path(info["stage"])
    src = rf.src_dir(lang)
    cfg = rf.corpus_cfg(lang)
    vend = ",".join(rf.vendored(cfg))
    fe = rf.JOERN_LANG[lang]
    flag = rf.FETCH_ON_FLAG.get(lang)
    rec_path = base / "s27-run.json"
    record = json.loads(rec_path.read_text()) if rec_path.exists() else {"lang": lang, "runs": {}}
    tag = subprocess.run([str(CPG / "infra" / "toolchain" / "build.sh"), "--print-tag"], capture_output=True, text=True).stdout.strip()
    record["toolchainImage"] = tag

    def measure(cfgname: str, cpgfile: str):
        out = base / cfgname
        t = time.time()
        r = run([RUNNER, "--backend", "container", "--mode", "offline", "--frontend", "joern",
                 "--script-file", str(HERE / "measure.sc"), "--source", str(stage), "--output", str(out), "--",
                 "--script", "/scripts/measure.sc", "--param", f"cpgFile=/out/{cpgfile}", "--param", "frame=/src/frame.jsonl",
                 "--param", "sample=/src/sample.private.jsonl", "--param", f"lang={lang}", "--param", f"vendored={vend}",
                 "--param", "out=/out/@AUDIT_ID@.measure.jsonl"], out / "measure.log", f"{cfgname}.measure")
        m = newest(out, "*.measure.jsonl", t)
        r["measureFile"] = str(m) if m else None
        record["runs"][f"{cfgname}.measure"] = r

    def content(cfgname: str, cpgfile: str):
        out = base / cfgname
        r = run([RUNNER, "--backend", "container", "--mode", "offline", "--frontend", "joern",
                 "--script-file", str(CPG / "sandbox" / "scripts" / "cpg-content-check.sc"), "--source", str(stage), "--output", str(out), "--",
                 "--script", "/scripts/cpg-content-check.sc", "--param", f"cpgFiles=/out/{cpgfile}"], out / "content.log", f"{cfgname}.content-check")
        txt = (out / "content.log").read_text()
        r["contentLines"] = [l for l in txt.splitlines() if l.startswith(("METHOD_COUNT", "CONTENT"))]
        record["runs"][f"{cfgname}.content-check"] = r
        return r["rc"] == 0 and any(l.endswith("=ok") for l in r["contentLines"])

    if a.measure_only:
        for cfgname in ("off", "on"):
            if a.only in (None, cfgname):
                p = record["runs"][f"{cfgname}.parse"]
                if not p.get("cpgFile"):
                    continue
                if content(cfgname, Path(p["cpgFile"]).name):
                    measure(cfgname, Path(p["cpgFile"]).name)
        rec_path.write_text(json.dumps(record, indent=1))
        print("record:", rec_path)
        return 0

    if a.only in (None, "off"):
        out = base / "off"
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        extra = (["--nooverlays"] if a.nooverlays else []) + (["--frontend-args", f"-J-Xmx{a.heap}"] if a.heap else [])
        r = run([RUNNER, "--backend", "container", "--mode", "offline", "--frontend", "joern-parse", "--source", str(src),
                 "--output", str(out), "--codebase", lang, "--", "/src", "--language", fe] + extra, out / "parse.log", "off.parse")
        cpg = newest(out, "*.cpg.bin", t)
        r["cpgFile"] = str(cpg) if cpg else None
        r["cpgMB"] = round(cpg.stat().st_size / 1e6, 2) if cpg else None
        record["runs"]["off.parse"] = r
        if cpg and r["rc"] == 0:
            if content("off", cpg.name):
                measure("off", cpg.name)
            else:
                print("off: content check failed -> parse failure of every file", flush=True)
        rec_path.write_text(json.dumps(record, indent=1))

    if a.only in (None, "on"):
        out = base / "on"
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("on.cpg.bin"):
            old.unlink()
        argv = ["docker", "run", "--rm", "--label", f"s27={lang}", "--name", f"s27-{lang}-on", "--hostname", "localhost", "--memory", "4g",
                "--user", "1000:1000", "-v", f"{src}:/src:ro", "-v", f"{out}:/out", tag,
                "joern-parse", "/src", "--language", fe, "-o", "/out/on.cpg.bin"]
        if a.nooverlays:
            argv += ["--nooverlays"]
        if flag:
            argv += ["--frontend-args", flag]
        if a.heap:
            argv += ["-J-Xmx" + a.heap]
        r = run(argv, out / "parse.log", "on.parse")
        r["fetchFlag"] = flag
        cpg = out / "on.cpg.bin"
        r["cpgFile"] = str(cpg) if cpg.exists() else None
        r["cpgMB"] = round(cpg.stat().st_size / 1e6, 2) if cpg.exists() else None
        record["runs"]["on.parse"] = r
        if cpg.exists() and r["rc"] == 0:
            if content("on", cpg.name):
                measure("on", cpg.name)
            else:
                print("on: content check failed", flush=True)
        rec_path.write_text(json.dumps(record, indent=1))
    print("record:", rec_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
