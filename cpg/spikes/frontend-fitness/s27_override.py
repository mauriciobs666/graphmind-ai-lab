#!/usr/bin/env python3
"""S27 fix round A2: the segment-rename override. Secondary measurement; the as-shipped run is the headline.

    cd cpg && .venv/bin/python spikes/frontend-fitness/s27_override.py <lang> [--measure-only]

The frontends skip files under a path segment the probe shows to be ignored by default (java: `test`; javascript: `i18n`
(+ `test`, `tests`); cpp: `test`, `tests`). There is no CLI override that removes a default, so the corpus is copied with those
segments renamed to `ccgseg<name>` and parsed (off: runner; on: toolchain image + the language's fetch flag). The frame and
private sample are staged with the renamed paths, `measure.sc` runs unchanged, and every `ccgseg<name>` in its output is mapped
back to `<name>` so `s27_metrics.py --variant override` joins the original frame/truth paths. Opens no label/truth.
"""
import json, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPG = HERE.parent.parent
sys.path.insert(0, str(HERE))
import run_fitness as rf  # noqa: E402
import s27_run as sr  # noqa: E402

SEGS = {"java": ["test"], "javascript": ["i18n", "test", "tests"], "cpp": ["test", "tests"]}
PFX = "ccgseg"


def ren(path, segs):
    parts = path.split("/")
    return "/".join([(PFX + x if i < len(parts) - 1 and x in segs else x) for i, x in enumerate(parts)])


def main():
    lang = sys.argv[1]
    mo = "--measure-only" in sys.argv
    segs = set(SEGS[lang])
    base = CPG / ".ccg-work" / "frontend-fitness" / "joern" / f"{lang}-override"
    src = base / "src"
    stage = base / "stage"
    if not mo:
        if base.exists():
            shutil.rmtree(base)
        orig = rf.src_dir(lang)
        for p in sorted(Path(orig).rglob("*")):
            rel = p.relative_to(orig).as_posix()
            if p.is_file():
                # rename directory segments only (the file name itself is kept)
                parts = rel.split("/")
                new = "/".join([(PFX + x if x in segs else x) for x in parts[:-1]] + [parts[-1]])
                (src / new).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, src / new)
    stage.mkdir(parents=True, exist_ok=True)
    for name, srcp in (("frame.jsonl", rf.frame_path(lang)), ("sample.private.jsonl", rf.private_path(lang))):
        with open(stage / name, "w") as fh:
            for r in rf.read_jsonl(srcp):
                if "path" in r:
                    r["path"] = ren(r["path"], segs)
                r["siteKey"] = ren(r["siteKey"].split(":")[0], segs) + ":" + ":".join(r["siteKey"].split(":")[1:]) if "siteKey" in r else None
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cfg = rf.corpus_cfg(lang)
    vend = ",".join(rf.vendored(cfg))
    fe, flag = rf.JOERN_LANG[lang], rf.FETCH_ON_FLAG.get(lang)
    tag = subprocess.run([str(CPG / "infra" / "toolchain" / "build.sh"), "--print-tag"], capture_output=True, text=True).stdout.strip()
    rec_path = base / "s27-run.json"
    record = json.loads(rec_path.read_text()) if rec_path.exists() else {"lang": lang, "variant": "override", "segmentsRenamed": sorted(segs), "runs": {}}
    record["toolchainImage"] = tag
    runner = str(CPG / "sandbox" / "offline-run.sh")

    def back(path):
        txt = path.read_text()
        txt = re.sub(PFX + r"(" + "|".join(map(re.escape, segs)) + r")(?=/)", r"\1", txt)
        path.write_text(txt)

    def content_measure(cfgname, cpgfile):
        out = base / cfgname
        r = sr.run([runner, "--backend", "container", "--mode", "offline", "--frontend", "joern", "--script-file",
                    str(CPG / "sandbox" / "scripts" / "cpg-content-check.sc"), "--source", str(stage), "--output", str(out), "--",
                    "--script", "/scripts/cpg-content-check.sc", "--param", f"cpgFiles=/out/{cpgfile}"], out / "content.log", f"{cfgname}.content-check")
        r["contentLines"] = [l for l in (out / "content.log").read_text().splitlines() if l.startswith(("METHOD_COUNT", "CONTENT"))]
        record["runs"][f"{cfgname}.content-check"] = r
        if not (r["rc"] == 0 and any(l.endswith("=ok") for l in r["contentLines"])):
            return
        t = time.time()
        r = sr.run([runner, "--backend", "container", "--mode", "offline", "--frontend", "joern", "--script-file", str(HERE / "measure.sc"),
                    "--source", str(stage), "--output", str(out), "--", "--script", "/scripts/measure.sc", "--param", f"cpgFile=/out/{cpgfile}",
                    "--param", "frame=/src/frame.jsonl", "--param", "sample=/src/sample.private.jsonl", "--param", f"lang={lang}",
                    "--param", f"vendored={vend}", "--param", "out=/out/@AUDIT_ID@.measure.jsonl"], out / "measure.log", f"{cfgname}.measure")
        m = sr.newest(out, "*.measure.jsonl", t)
        if m:
            back(m)  # map the renamed segments back to the frame's paths
            r["measureFile"] = str(m)
        record["runs"][f"{cfgname}.measure"] = r

    for cfgname in ("off", "on"):
        out = base / cfgname
        out.mkdir(parents=True, exist_ok=True)
        if mo:
            cpg = Path(record["runs"][f"{cfgname}.parse"]["cpgFile"])
        else:
            t = time.time()
            if cfgname == "off":
                r = sr.run([runner, "--backend", "container", "--mode", "offline", "--frontend", "joern-parse", "--source", str(src),
                            "--output", str(out), "--codebase", lang, "--", "/src", "--language", fe], out / "parse.log", "off.parse")
                cpg = sr.newest(out, "*.cpg.bin", t)
            else:
                argv = ["docker", "run", "--rm", "--label", f"s27={lang}-override", "--name", f"s27-{lang}-override-on", "--hostname", "localhost",
                        "--memory", "4g", "--user", "1000:1000", "-v", f"{src}:/src:ro", "-v", f"{out}:/out", tag, "joern-parse", "/src",
                        "--language", fe, "-o", "/out/on.cpg.bin"] + (["--frontend-args", flag] if flag else [])
                r = sr.run(argv, out / "parse.log", "on.parse")
                cpg = out / "on.cpg.bin" if (out / "on.cpg.bin").exists() else None
            r["cpgFile"] = str(cpg) if cpg else None
            r["cpgMB"] = round(cpg.stat().st_size / 1e6, 2) if cpg else None
            record["runs"][f"{cfgname}.parse"] = r
        if cpg and cpg.exists():
            content_measure(cfgname, cpg.name)
        rec_path.write_text(json.dumps(record, indent=1))
    print("record:", rec_path)


main()
