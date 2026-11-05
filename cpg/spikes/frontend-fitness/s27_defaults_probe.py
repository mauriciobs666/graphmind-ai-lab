#!/usr/bin/env python3
"""S27 fix round A2: do the Joern frontends ignore files by DEFAULT? Minimal probes through the sandbox runner (offline).

    cd cpg && .venv/bin/python spikes/frontend-fitness/s27_defaults_probe.py [lang ...]

Each case copies chosen corpus files into a fresh tree (as shipped path, or moved to a neutral path), parses it with
`joern-parse` (optionally with extra --frontend-args flags) and reads the FILE nodes back through cpg-content-check.sc.
Writes results/frontend-defaults-probe.json. Opens no label/truth.
"""
import json, shutil, subprocess, sys, glob, re, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPG = HERE.parent.parent
WORK = CPG / ".ccg-work" / "frontend-fitness" / "s27-probes"
CORP = CPG / ".ccg-work" / "corpora"
RUNNER = str(CPG / "sandbox" / "offline-run.sh")
FE = {"java": "javasrc", "javascript": "jssrc", "go": "golang", "csharp": "csharpsrc", "cpp": "c"}

J = "src/test/java/org/springframework/validation/PetAgeValidatorTest.java"
JM = "src/main/java/org/springframework/samples/petclinic/model/Owner.java"
CASES = {
 "java": [("shipped-test-path", [(J, J)], []), ("moved-neutral-path", [(J, "moved/PetAgeValidatorTest.java")], []),
          ("moved-into-test-dir-name-only", [(J, "src/test/java/x/Foo.java")], []),
          ("shipped+exclude-regex-nonmatching", [(J, J)], ["--exclude-regex", "zzzz"]),
          ("control-main-file", [(JM, JM)], []),
          ("direct-javasrc2cpg-shipped", [(J, J)], [], "javasrc2cpg"),
          ("direct-javasrc2cpg+exclude-regex-nonmatching", [(J, J)], ["--exclude-regex", "zzzz"], "javasrc2cpg"),
          ("direct-javasrc2cpg-moved-neutral", [(J, "moved/PetAgeValidatorTest.java")], [], "javasrc2cpg")],
 "javascript": [("shipped-i18n-path", [("src/i18n/format.ts", "src/i18n/format.ts")], []),
                ("moved-neutral-path", [("src/i18n/format.ts", "src/lib/format.ts")], []),
                ("shipped+exclude-regex-nonmatching", [("src/i18n/format.ts", "src/i18n/format.ts")], ["--exclude-regex", "zzzz"]),
                ("control-api", [("src/api/polling.ts", "src/api/polling.ts")], []),
                ("segment-test", [("src/api/polling.ts", "src/test/polling.ts")], []),
                ("segment-tests", [("src/api/polling.ts", "src/tests/polling.ts")], []),
                ("direct-jssrc2cpg-shipped-i18n", [("src/i18n/format.ts", "src/i18n/format.ts")], [], "jssrc2cpg"),
                ("direct-jssrc2cpg-i18n+exclude-regex-nonmatching", [("src/i18n/format.ts", "src/i18n/format.ts")], ["--exclude-regex", "zzzz"], "jssrc2cpg")],
 "go": [("rfc9457-alone", [("rfc9457.go", "rfc9457.go")], []), ("csrf-alone", [("middleware/csrf.go", "middleware/csrf.go")], []),
        ("binder_test-alone", [("binder_test.go", "binder_test.go")], []), ("control-echo", [("echo.go", "echo.go")], [])],
 "csharp": [("scopes-alone", [("src/Authorization/ScopesAuthorizer.cs", "src/Authorization/ScopesAuthorizer.cs")], []),
            ("acceptancesteps-alone", [("testing/AcceptanceSteps.cs", "testing/AcceptanceSteps.cs")], []),
            ("scopes-moved", [("src/Authorization/ScopesAuthorizer.cs", "moved/ScopesAuthorizer.cs")], [])],
 "cpp": [("gtest_unittest-alone", [("googletest/test/gtest_unittest.cc", "googletest/test/gtest_unittest.cc")], []),
         ("gtest_unittest-moved", [("googletest/test/gtest_unittest.cc", "moved/gtest_unittest.cc")], []),
         ("control-gtest.cc", [("googletest/src/gtest.cc", "googletest/src/gtest.cc")], []),
         ("segment-test", [("googletest/src/gtest.cc", "googletest/test/gtest.cc")], []),
         ("segment-tests", [("googletest/src/gtest.cc", "googletest/tests/gtest.cc")], []),
         ("direct-c2cpg-segment-test", [("googletest/src/gtest.cc", "googletest/test/gtest.cc")], [], "c2cpg"),
         ("direct-c2cpg-segment-test+exclude-regex-nonmatching", [("googletest/src/gtest.cc", "googletest/test/gtest.cc")], ["--exclude-regex", "zzzz"], "c2cpg")],
}


def run(argv, log):
    with open(log, "w") as fh:
        return subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT).returncode


def main():
    langs = sys.argv[1:] or list(CASES)
    out = {}
    for lang in langs:
        root = CORP / lang / "src"
        for case in CASES[lang]:
            name, files, extra = case[:3]
            direct = case[3] if len(case) > 3 else None
            d = WORK / lang / name
            if d.exists():
                shutil.rmtree(d)
            (d / "src").mkdir(parents=True)
            (d / "out").mkdir()
            for a, b in files:
                (d / "src" / b).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(root / a, d / "src" / b)
            t = time.time()
            if direct:
                # the frontend binary itself: the runner appends only `--output /out/<id>.cpg.bin`, so extra flags are not followed by anything
                argv = [RUNNER, "--backend", "container", "--mode", "offline", "--frontend", direct, "--source", str(d / "src"),
                        "--output", str(d / "out"), "--", "/src"] + extra
            else:
                argv = [RUNNER, "--backend", "container", "--mode", "offline", "--frontend", "joern-parse", "--source", str(d / "src"),
                        "--output", str(d / "out"), "--codebase", lang, "--", "/src", "--language", FE[lang]]
                if extra:
                    argv += ["--frontend-args"] + extra
            rc = run(argv, d / "parse.log")
            cpgs = sorted(glob.glob(str(d / "out" / "*.cpg.bin")))
            rec = {"lang": lang, "case": name, "files": [b for a, b in files], "extraArgs": extra, "command": " ".join(a.replace(str(CPG) + "/", "") for a in argv), "parseRc": rc, "wall": round(time.time() - t, 1)}
            if cpgs:
                c = Path(cpgs[-1]).name
                rc2 = run([RUNNER, "--backend", "container", "--mode", "offline", "--frontend", "joern", "--script-file",
                           str(CPG / "sandbox" / "scripts" / "cpg-content-check.sc"), "--source", str(d / "src"), "--output", str(d / "out"), "--",
                           "--script", "/scripts/cpg-content-check.sc", "--param", f"cpgFiles=/out/{c}"], d / "content.log")
                txt = (d / "content.log").read_text()
                rec["fileMethods"] = [l.split("\t", 1)[1] for l in txt.splitlines() if l.startswith("FILE_METHODS")]
                rec["methodCount"] = [l for l in txt.splitlines() if l.startswith("METHOD_COUNT")]
                rec["contentRc"] = rc2
            else:
                rec["noCpg"] = True
            log = (d / "parse.log").read_text()
            rec["parseLogNonWarn"] = [l[:200] for l in log.splitlines() if re.search(r"error|exception|ignor|skip|excluded|cannot|fail", l, re.I)][:5]
            out.setdefault(lang, []).append(rec)
            print(json.dumps({k: rec[k] for k in ("lang", "case", "parseRc", "fileMethods", "methodCount") if k in rec}), flush=True)
    p = HERE / "results" / "frontend-defaults-probe.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev.update(out)
    prev["_helpExcerpts"] = {
        "note": "--help of the frontends in the toolchain image (Joern v4.0.579), lines on exclusion; none offers a way to remove a default",
        "jssrc2cpg": ["--exclude <file1>  files or folders to exclude ...", "--exclude-regex <value>  a regex specifying files to exclude ..."],
        "javasrc2cpg": ["--exclude <file1> ...", "--exclude-regex <value> ..."], "c2cpg": ["--exclude <file1> ...", "--exclude-regex <value> ...", "--include <value> header include paths"],
        "gosrc2cpg": ["--exclude <file1> ...", "--exclude-regex <value> ..."], "csharpsrc2cpg": ["--exclude <file1> ...", "--exclude-regex <value> ..."],
        "joern-parse": "own options + `--frontend-args` (variadic, verbatim); the runner appends `--output` after it, so any flag given after --frontend-args yields `Unknown option --output` (the earlier cases)"}
    p.write_text(json.dumps(prev, indent=1) + "\n")


main()
