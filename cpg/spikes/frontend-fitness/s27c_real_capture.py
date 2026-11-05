#!/usr/bin/env python3
"""S27c-F4: stage the planted census sources (the FIX sources of tests/test_defs_census.py, plus decorator-probe extras) for a real
Joern capture of measure.sc's census fields.

    cd cpg && .venv/bin/python spikes/frontend-fitness/s27c_real_capture.py stage <lang>   # -> tests/data/census/real-src/<lang>/

Writes `src/<rel>` (the planted source), `frame.jsonl` (one frame row per call site that has a census `arg` record, from the
census extractor's own siteKey/line/column) and an empty `sample.private.jsonl`. The Joern runs are in the shell lines printed by `cmds`.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPG = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CPG / "tests"))
import make_defs_census as mdc  # noqa: E402
import test_defs_census as T  # noqa: E402

# Extra planted files: a receiver call with string arguments (and, for Python, a keyword argument), and richer decorators for the probes.
EXTRA = {
    "python": ("r.py", """class Q:
    def run(self):
        obj.call("x", "y", k="z")
        self.call("w")
@dec
@pkg.mod.route("/a", methods=["GET", "POST"], name="n")
def deco2(): pass
@dec2()
class Cls: pass
"""),
    "javascript": ("r.ts", """class Q {
  run() { obj.call("x", "y"); this.call("w"); }
}
@Component({selector: "s"})
class Cmp { @Input() x: string; @HostListener("click", ["$event"]) onClick(e) {} }
function plain() {}
"""),
    "java": ("R.java", """public class R {
  void run() { obj.call("x", "y"); this.call("w"); }
  @RequestMapping(value = "/a", method = RequestMethod.GET) @Deprecated void m2() {}
  @SuppressWarnings({"a", "b"}) void m3() {}
}
"""),
    "kotlin": ("r.kt", """class R {
  fun run() { obj.call("x", "y"); this.call("w") }
  @GetMapping("/a") @Deprecated("old") fun m2() {}
}
"""),
    "csharp": ("R.cs", """class R {
  void Run() { obj.Call("x", "y"); this.Call("w"); }
  [Route("/a"), Obsolete("old")] void M2() {}
  [HttpGet] void M3() {}
}
"""),
    "go": ("r.go", """package p
const K2 = "k2"
func call(a ...string) {}
func run() { obj.call("x", "y"); s.call("w"); call("p", "q", K2) }
"""),
    "cpp": ("r.cpp", """struct R { void m(); };
void run(R* p, R o) { p->call("x", "y"); o.call("w"); }
[[deprecated("old")]] void d1() {}
[[nodiscard]] int d2() { return 0; }
"""),
}

OUT = CPG / "tests" / "data" / "census" / "real-src"


def stage(lang):
    d = OUT / lang
    (d / "src").mkdir(parents=True, exist_ok=True)
    args = []
    for rel, src in (T.FIX[lang][:2], EXTRA[lang]):
        p = d / "src" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
        args += mdc.census_file(lang, rel, src.encode())[1]
    rel = None
    seen, rows = set(), []
    for a in args:
        if a["siteKey"] in seen:
            continue
        seen.add(a["siteKey"])
        callee = re.split(r"[.:>\-]+", a["callText"].strip())[-1]
        rows.append({"siteKey": a["siteKey"], "path": a["path"], "line": a["line"], "nameLine": a["line"], "column": a["column"],
                     "nameColumn": a["column"], "calleeName": callee, "callText": a["callText"],
                     "kind": "method" if "." in a["callText"] else "plain", "nodeType": "call", "grammar": lang})
    (d / "frame.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (d / "sample.private.jsonl").write_text("")
    print(lang, len(rows), "frame rows")


if __name__ == "__main__":
    if sys.argv[1] == "stage":
        for l in (sys.argv[2:] or sorted(T.FIX)):
            stage(l)


# ------------------------------------------------------------------ decorator probes (results/<lang>.decorator-probe.json)
# `probe <lang> <dprobe.jsonl>` writes the observed shapes. The prose below states what decorator-probe.sc printed for the planted sources of
# tests/data/census/real-src/<lang>/src and nothing else; constructs the sources do not contain are listed under `unobserved`.
PROBE = {
    "python": dict(node="CALL (no ANNOTATION node)", observed=True, shapes=[
        dict(construct="@app.route(\"/x\", methods=[\"GET\"]) def h", where="m.py:11-12",
             joern="pysrc2cpg rewrites the definition to `h = app.route(\"/x\", methods = [\"GET\"])(def h(...))`. The METHOD `h` has annotation count 0 and no decorator child. "
                   "The decorator is a CALL (name `route`, code `app.route(\"/x\", methods = [\"GET\"])`, line 11) whose LITERAL arguments are `\"/x\"` and, inside a "
                   "`<operator>.listLiteral` CALL argument, `\"GET\"`. It is the receiver of a dynamic-dispatch CALL with an empty name whose argument 1 is the METHOD_REF of `h`.",
             example="CALL route <- receiver of CALL '' (args: METHOD_REF h) <- argument 2 of CALL <operator>.assignment h = ..."),
        dict(construct="@overload def ov", where="m.py:9-10", joern="Bare decorator: CALL named `overload` (code `overload(def ov(...))`) with the METHOD_REF as argument 1, no literals."),
        dict(construct="@dec @pkg.mod.route(\"/a\", methods=[\"GET\",\"POST\"], name=\"n\") def deco2", where="r.py:5-7",
             joern="Nested: CALL `dec` whose argument 1 is a BLOCK (`tmp0 = pkg.mod` then the CALL `route` with LITERALs \"/a\", \"GET\", \"POST\" and keyword literal \"n\" (argumentName `name`)) applied to the METHOD_REF. "
                   "The outer CALL's line is the first decorator's line (5); `route` is on line 6."),
    ], notExposed=["a keyword argument carries argumentIndex -1 (see measure.sc `order` note)", "the decorator's receiver path (`pkg.mod`) is split over a fieldAccess CALL, only the final name is the CALL's `name`"],
        unobserved=["decorators on a class: the planted `@dec2() class Cls` left no node containing `dec2` in the CPG (grep of the probe output), so a class decorator is not exposed in this sample"]),
    "javascript": dict(node="ANNOTATION", observed=True, shapes=[
        dict(construct="@Get(\"/p\") route() {}  (TypeScript, .ts)", where="m.ts:8", joern="ANNOTATION node, name `Get`, fullName `Get`, code `@Get(\"/p\")`, AST child of the METHOD `route` (annotation count 1). The ANNOTATION has no AST children: the argument `\"/p\"` is only in its `code`.",
             example="METHOD route -> ANNOTATION '@Get(\"/p\")' (children: none)"),
        dict(construct="@HostListener(\"click\", [\"$event\"]) onClick(e) {}", where="r.ts:5", joern="ANNOTATION `HostListener` under METHOD `onClick`, no children, arguments only in `code`."),
        dict(construct="@Component({selector: \"s\"}) class Cmp", where="r.ts:4", joern="ANNOTATION `Component` whose AST parent is the TYPE_DECL `Cmp`, no children."),
        dict(construct="@Input() x: string;  (class field)", where="r.ts:5", joern="ANNOTATION `Input` whose AST parent is a MEMBER."),
    ], notExposed=["literal arguments are not nodes: they are only in the ANNOTATION `code` text"],
        unobserved=["plain .js files with decorators (the sources are .ts)"]),
    "java": dict(node="ANNOTATION", observed=True, shapes=[
        dict(construct="@GetMapping(\"/x\") void m()", where="J.java:7", joern="ANNOTATION name `GetMapping`, fullName `ANY`, parent METHOD; child ANNOTATION_PARAMETER_ASSIGN (code `\"/x\"`) with children ANNOTATION_PARAMETER `value` and ANNOTATION_LITERAL whose code is `/x` (no quotes).",
             example="METHOD m -> ANNOTATION GetMapping -> ANNOTATION_PARAMETER_ASSIGN -> [ANNOTATION_PARAMETER value, ANNOTATION_LITERAL '/x']"),
        dict(construct="@Override void ov()  /  @Deprecated", where="J.java:6, R.java:3", joern="Marker annotation: ANNOTATION node, no children; fullName is the resolved `java.lang.Override` / `java.lang.Deprecated`."),
        dict(construct="@RequestMapping(value = \"/a\", method = RequestMethod.GET)", where="R.java:3", joern="Two ANNOTATION_PARAMETER_ASSIGN children; `value = \"/a\"` has an ANNOTATION_LITERAL `/a`, `method = RequestMethod.GET` has only the ANNOTATION_PARAMETER `method` (the non-literal value is not a literal node)."),
        dict(construct="@SuppressWarnings({\"a\", \"b\"})", where="R.java:4", joern="ANNOTATION_PARAMETER_ASSIGN whose value child is an ARRAY_INITIALIZER (code `{ \"a\", \"b\" }`); its children are ANNOTATION_LITERALs with code `a` and `b` (no quotes)."),
    ], notExposed=["ANNOTATION_LITERAL code is the decoded text without quotes (`/x`), unlike C# LITERAL (`\"/a\"`) and Kotlin ANNOTATION_LITERAL (`\"fa\"`)"],
        unobserved=["annotations on classes, fields or parameters"]),
    "kotlin": dict(node="ANNOTATION", observed=True, frontendOnly=True, shapes=[
        dict(construct="@Foo(\"fa\") fun a()", where="m.kt:2-3", joern="ANNOTATION name `Foo`, fullName `<unresolvedNamespace>.Foo`, code `@Foo(\"fa\")` (line 2), AST parent METHOD `a` (line 3); one direct child ANNOTATION_LITERAL with code `\"fa\"` (quotes kept).",
             example="METHOD a -> ANNOTATION Foo -> ANNOTATION_LITERAL '\"fa\"'"),
        dict(construct="@GetMapping(\"/a\") @Deprecated(\"old\") fun m2()", where="r.kt:3", joern="Two ANNOTATION nodes on METHOD `m2`, each with one ANNOTATION_LITERAL child; `Deprecated` resolves to `kotlin.Deprecated`."),
    ], notExposed=[], unobserved=["annotations on classes or properties", "an overlay-complete CPG: the default overlays abort on Kotlin (ReachingDefPass), every figure here is from the overlay-free import"]),
    "csharp": dict(node="ANNOTATION", observed=True, shapes=[
        dict(construct="[HttpGet(\"/a\")] public void M()", where="C.cs:2", joern="ANNOTATION name `HttpGet`, fullName `HttpGetAttribute`, code `HttpGet(\"/a\")` (no brackets), parent METHOD; one child of label LITERAL with code `\"/a\"`.",
             example="METHOD M -> ANNOTATION HttpGet -> LITERAL '\"/a\"'"),
        dict(construct="[Route(\"/a\"), Obsolete(\"old\")] void M2()", where="R.cs:2", joern="Two ANNOTATION nodes on METHOD `M2`, each with a LITERAL child."),
        dict(construct="[HttpGet] void M3()", where="R.cs:3", joern="ANNOTATION with no children."),
    ], notExposed=["a plain unqualified call `F(...)` has zero `argument`s in csharpsrc2cpg although its code holds the arguments (relevant to the argument census, not to attributes)"],
        unobserved=["attributes on classes, properties or parameters", "named attribute arguments"]),
    "cpp": dict(node=None, observed=True, shapes=[
        dict(construct="[[nodiscard(\"why\")]] int f()  /  [[deprecated(\"old\")]] void d1()  /  [[nodiscard]] int d2()", where="m.cpp:10, r.cpp:3-4",
             joern="No ANNOTATION node and no CALL/LITERAL/IDENTIFIER for the attribute exists in the CPG: the planted attribute text occurs only inside the `code` property of the METHOD (`[[nodiscard(\"why\")]] int f() {...`) and of the file's `<global>` AST child.")],
        notExposed=["attribute name and argument as nodes"], unobserved=["attributes on classes or variables", "GNU `__attribute__` / MSVC `__declspec`"]),
    "go": dict(node=None, observed=True, shapes=[], notExposed=[], unobserved=[],
               note="Go has no decorator or annotation construct (report §a1.6: tree-sitter node none); the Go CPG of the planted sources has no ANNOTATION node."),
}


def probe(lang, path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    ann = [r for r in rows if r["k"] == "annotation"]
    meth_ann = {(r["file"], r["name"], r["l"]): r["annCount"] for r in rows if r["k"] == "mchild" and r["annCount"]}
    P = PROBE[lang]
    out = {"lang": lang, "joern": "4.0.579", "source": f"cpg/tests/data/census/real-src/{lang}/src", "script": "cpg/spikes/frontend-fitness/decorator-probe.sc",
           "node": P["node"], "annotationNodes": len(ann), "methodsWithAnnotation": [{"file": k[0], "name": k[1], "l": k[2], "annotations": v} for k, v in sorted(meth_ann.items())],
           "annotationParentLabels": sorted({a["parent"]["label"] for a in ann}), "shapes": P["shapes"], "notExposed": P["notExposed"], "unobserved": P["unobserved"]}
    if P.get("note"):
        out["note"] = P["note"]
    if P.get("frontendOnly"):
        out["frontendOnly"] = True
    if ann:
        a = ann[0]
        out["example"] = {"lang": lang, "node": a["children"] and a["children"][0]["label"] or "ANNOTATION", "annotation": {k: a[k] for k in ("file", "line", "name", "fullName", "code", "parent", "children")}}
    return out


def freeze(lang, measure_path):
    """tests/data/census/<lang>.real.measure.jsonl: the raw measure.sc output, only meta.cpg (an audit id) replaced by a stable name."""
    rows = [json.loads(l) for l in Path(measure_path).read_text().splitlines() if l.strip()]
    rows[0]["cpg"] = f"/out/{lang}.real.cpg.bin"
    rows[0]["real"] = True
    dest = CPG / "tests" / "data" / "census" / f"{lang}.real.measure.jsonl"
    dest.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    print("wrote", dest, len(rows), "records")


def check(lang):
    """Run the existing census join (s27_census_metrics.census_block) over the planted sources and the frozen real measure file."""
    import s27_census_metrics as cm
    M = cm.read_measure(CPG / "tests" / "data" / "census" / f"{lang}.real.measure.jsonl")
    d_by, args, files = {}, [], []
    for rel, src in (T.FIX[lang][:2], EXTRA[lang]):
        d, a, _ = mdc.census_file(lang, rel, src.encode())
        d_by[rel] = d
        args += a
        files.append(rel)
    D = {"meta": {}, "files": files, "defs": d_by, "args": args}
    b = cm.census_block(lang, D, M)
    print(json.dumps({k: b[k] for k in b if k in ("parseSuccess", "methodCoverage", "lineFidelity", "valueSetRecovery", "decoratorShape")}, indent=1)[:3000])


if __name__ == "__main__" and sys.argv[1] == "freeze":
    freeze(sys.argv[2], sys.argv[3])
if __name__ == "__main__" and sys.argv[1] == "check":
    check(sys.argv[2])
if __name__ == "__main__" and sys.argv[1] == "probe":
    lang, path = sys.argv[2], sys.argv[3]
    dest = HERE / "results" / f"{lang}.decorator-probe.json"
    dest.write_text(json.dumps(probe(lang, path), indent=1, sort_keys=True) + "\n")
    print("wrote", dest)


# ------------------------------------------------------------------ the 12 census measure runs
def census_measure(lang):
    """measure.sc (census schema) over the CPGs S27 already built (s27-run.json: off.parse / on.parse cpgFile), through the offline runner,
    into the same joern/<lang>/{off,on} dirs (a new audit-id file; nothing is overwritten) and copied to results/<lang>.census-{off,on}.measure.jsonl.
    Kotlin: the CPGs are frontend-only (no default overlays), so --param frontendOnly=true."""
    import subprocess, time
    import run_fitness as rf
    info = rf.cmd_joern(lang)
    stage = Path(info["stage"])
    base = stage.parent
    rec = json.loads((base / "s27-run.json").read_text())
    vend = ",".join(rf.vendored(rf.corpus_cfg(lang)))
    out_rec = {}
    for cfgname in ("off", "on"):
        cpgfile = Path(rec["runs"][f"{cfgname}.parse"]["cpgFile"])
        out = base / cfgname
        t0 = time.time()
        argv = [str(CPG / "sandbox" / "offline-run.sh"), "--backend", "container", "--mode", "offline", "--frontend", "joern",
                "--script-file", str(HERE / "measure.sc"), "--source", str(stage), "--output", str(out), "--",
                "--script", "/scripts/measure.sc", "--param", f"cpgFile=/out/{cpgfile.name}", "--param", "frame=/src/frame.jsonl",
                "--param", "sample=/src/sample.private.jsonl", "--param", f"lang={lang}", "--param", f"vendored={vend}",
                "--param", "out=/out/@AUDIT_ID@.measure.jsonl"]
        if lang == "kotlin":
            argv += ["--param", "frontendOnly=true"]
        with (out / "census-measure.log").open("w") as fh:
            p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT)
        new = [x for x in out.glob("*.measure.jsonl") if x.stat().st_mtime >= t0 - 1]
        new = max(new, key=lambda x: x.stat().st_mtime) if new else None
        dest = HERE / "results" / f"{lang}.census-{cfgname}.measure.jsonl"
        if new and p.returncode == 0:
            dest.write_bytes(new.read_bytes())
        out_rec[cfgname] = {"rc": p.returncode, "cpg": str(cpgfile), "measure": str(new) if new else None, "copied": str(dest) if new and p.returncode == 0 else None,
                            "wallSeconds": round(time.time() - t0, 1)}
        print(lang, cfgname, json.dumps(out_rec[cfgname]), flush=True)
    (base / "s27c-census-run.json").write_text(json.dumps(out_rec, indent=1))


if __name__ == "__main__" and sys.argv[1] == "measure12":
    sys.path.insert(0, str(HERE))
    for l in sys.argv[2:]:
        census_measure(l)
