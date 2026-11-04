"""S25 frontend-fitness harness tests (cpg/spikes/frontend-fitness/run_fitness.py).

Pure-python: tree-sitter frame extraction on snippets, the hash-sort draw, excerpt/column arithmetic,
the C5 pilot, the SCIP/callgraph reference mapping, and the firewall (no Joern/reference text in
sample/ or frame/ files). Run: cd cpg && .venv/bin/python -m pytest -q tests/test_frontend_fitness.py
"""
import json
import os
import sys
from pathlib import Path

import pytest

# CCG_FF_DIR points the suite at a COPY of the spike directory (mutation testing); default is the real one
FF = Path(os.environ.get("CCG_FF_DIR") or (Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"))
sys.path.insert(0, str(FF))
import run_fitness as rf  # noqa: E402


def sites(lang, src, name="x"):
    ext = {"python": ".py", "javascript": ".ts", "java": ".java", "kotlin": ".kt", "csharp": ".cs", "go": ".go", "cpp": ".cc"}[lang]
    data = src.encode()
    key, tree = rf.parse_file(lang, name + ext, data)
    L = rf.Lines(data)
    out = []
    for s in sorted(rf.frame_sites(lang, key, tree.root_node, data), key=lambda s: (s.start, s.end)):
        line, col = L.lc(s.start)
        nn = s.name_node
        out.append((line, col, rf.ws(data[s.start:s.end].decode()), nn.text.decode() if nn is not None else None, s.kind))
    return out


def texts(lang, src):
    return [(t, n) for (_, _, t, n, _) in sites(lang, src)]


def test_python_chain_two_sites_share_position_and_decorator_call_excluded():
    got = sites("python", "a.b().c()\n")
    assert [(l, c, t, n) for l, c, t, n, _ in got] == [(1, 1, "a.b", "b"), (1, 1, "a.b().c", "c")]
    src = "@app.get(make_path())\ndef f():\n    g()\n"
    assert texts("python", src) == [("g", "g")]          # decorator call and the call inside its args are out


def test_javascript_new_tagged_template_jsx_and_decorator():
    ts = "class A { f(){ a.b(1); new Foo(2); tag`x${1}`; } }\n"
    assert texts("javascript", ts) == [("a.b", "b"), ("new Foo", "Foo"), ("tag", "tag")]
    assert sites("javascript", ts)[2][4] == "tagged-template"
    deco = "@Component({a: f()})\nclass A { g(){ h() } }\n"
    assert texts("javascript", deco) == [("h", "h")]


def test_javascript_jsx_uppercase_only(tmp_path):
    src = "const x = <Foo a={1}><B.C/><div/></Foo>;\n".encode()
    key, tree = rf.parse_file("javascript", "x.tsx", src)
    got = [src[s.start:s.end].decode() for s in sorted(rf.frame_sites("javascript", key, tree.root_node, src), key=lambda s: s.start)]
    assert got == ["Foo", "B.C"]


def test_java_sites():
    src = "class A { A(){ super(1); } void f(){ a.b(1); new Foo(2); this.x(); Util.s(); } }\n"
    assert texts("java", src) == [("super", "super"), ("a.b", "b"), ("new Foo", "Foo"), ("this.x", "x"), ("Util.s", "s")]
    assert [k for *_, k in sites("java", src)][-1] == "qualified"


def test_kotlin_trailing_lambda_is_one_site_and_delegation_is_a_site():
    src = "class A(val x:Int): B(x) { fun f() { foo(2){ it }; bar { 1 }; a.b(1) } }\n"
    assert texts("kotlin", src) == [("B", "B"), ("foo", "foo"), ("bar", "bar"), ("a.b", "b")]


def test_csharp_sites():
    src = "class A { void f(){ a.B<int>(1); new Foo(2); this.x(); } }\n"
    assert texts("csharp", src) == [("a.B<int>", "B"), ("new Foo", "Foo"), ("this.x", "x")]


def test_go_conversion_stays_in_frame_and_generic_call_name():
    src = "package p\nfunc f(){ x := T(y); g(1); a.B(2); h[int](3) }\n"
    got = texts("go", src)
    assert ("T", "T") in got and ("g", "g") in got and ("a.B", "B") in got
    assert any(t == "h[int]" and n == "h" for t, n in got)


def test_cpp_sites_and_macro_shape():
    src = "void f(){ a::b<int>(1); new Foo(2); x->y(); MY_M(1); }\n"
    got = sites("cpp", src)
    assert [(t, n) for _, _, t, n, _ in got] == [("a::b<int>", "b"), ("new Foo", "Foo"), ("x->y", "y"), ("MY_M", "MY_M")]
    assert got[-1][4] == "macro-shaped"


def test_columns_are_code_points_not_bytes():
    got = sites("python", "x = 'é'; f()\n")
    assert got == [(1, 10, "f", "f", "plain")]   # bytes would say 11


def test_hash_sort_is_pure_and_matches_spec_formula():
    rows = [{"siteKey": k} for k in ("b:1:1:x", "a:1:1:x", "c:1:1:x")]
    seed = rf.seed_for("sample", "python", "deadbeef")
    assert seed == "ccg-p0a-v1|python|deadbeef"
    order = rf.hash_sort(rows, seed)
    import hashlib
    exp = sorted(((hashlib.sha256((seed + "|" + r["siteKey"]).encode()).hexdigest(), r["siteKey"]) for r in rows))
    assert [(h, r["siteKey"]) for h, r in order] == exp
    assert rf.seed_for("flow", "go", "s") == "ccg-p0a-v1-flow|go|s"
    assert rf.seed_for("c5", "go", "s") == "ccg-p0a-v1-c5|go|s"


def test_excerpt_is_three_before_three_after():
    lines = [f"l{i}" for i in range(1, 21)]
    ex = rf.excerpt(lines, 10).split("\n")
    assert len(ex) == 7 and ex[0].endswith("l7") and ex[-1].endswith("l13")
    assert len(rf.excerpt(lines, 1).split("\n")) == 4


def test_flow_population_python_qname_and_filters(tmp_path):
    src = (
        "class C:\n"
        "    def m(self, x):\n"
        "        if x:\n"
        "            g(x)\n"
        "        return x\n"
        "        # pad\n"
        "def top(a):\n"
        "    if a:\n"
        "        h(a)\n"
        "    return a\n"
        "    #\n"
        "def tiny(a): g(a)\n"
    )
    (tmp_path / "m.py").write_text(src)
    pop = rf.flow_population("python", tmp_path, {})
    assert sorted(p["qname"] for p in pop) == ["m.py::C.m", "m.py::top"]   # 'tiny' is < 5 lines


def test_generated_header_excluded(tmp_path):
    (tmp_path / "a.go").write_text("// Code generated by x. DO NOT EDIT.\npackage a\nfunc f(){ g() }\n")
    (tmp_path / "b.go").write_text("package a\nfunc f(){ g() }\n")
    assert [r for r, _ in rf.iter_source_files("go", tmp_path, {})] == ["b.go"]
    vend = tmp_path / "vendor" / "x"
    vend.mkdir(parents=True)
    (vend / "c.go").write_text("package x\nfunc f(){ g() }\n")
    assert [r for r, _ in rf.iter_source_files("go", tmp_path, {})] == ["b.go"]


# ---------------------------------------------------------------- make_adjudication_input.py

import ast  # noqa: E402
import importlib.util  # noqa: E402


def _load_adj(path=None):
    spec = importlib.util.spec_from_file_location("adj_under_test", path or (FF / "make_adjudication_input.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _label(o, callees, **kw):
    d = {"ordinal": o, "siteId": f"{o:016x}", "callees": callees, "notACall": False, "dynamic": False,
         "confidence": "sure", "evidence": "x"}
    d.update(kw)
    return d


def _ref(o, static, impls=(), **kw):
    d = {"ordinal": o, "siteId": f"{o:016x}", "site": {"path": "a.py", "line": o, "column": 1, "callText": "f"},
         "static": static, "impls": list(impls), "tool": "t", "toolVersion": "1"}
    d.update(kw)
    return d


def _write(root, lang, labels, refs):
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "truth").mkdir(parents=True, exist_ok=True)
    (root / "labels" / f"{lang}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    (root / "truth" / f"{lang}.ref.jsonl").write_text("".join(json.dumps(x) + "\n" for x in refs))


INT = lambda p, l: {"kind": "internal", "path": p, "qname": f"{p}::x", "defLine": l}  # noqa: E731
LINT = lambda p, l: {"kind": "internal", "path": p, "qname": f"{p}::x", "defLine": l}  # noqa: E731


def test_adjudication_input_statuses(tmp_path):
    adj = _load_adj()
    labels = [
        _label(1, [INT("a.py", 3)]),                                   # agreed
        _label(2, [{"kind": "external", "package": "os"}]),            # agreed (both external)
        _label(3, [INT("a.py", 9)]),                                   # reference-silent
        _label(4, [{"kind": "external", "package": "os"}]),            # label external, reference internal
        _label(5, [INT("a.py", 3), INT("a.py", 7)]),                   # disagree: reference has one
        _label(6, [INT("a.py", 3)], confidence="unsure"),              # unsure
        _label(7, [], notACall=True),                                  # agreed with silent
        _label(8, [INT("a.py", 3)]),                                   # agreed with dispatch expansion
    ]
    refs = [
        _ref(1, {"kind": "internal", "path": "a.py", "qname": "a.py::x", "defLine": 3}),
        _ref(2, {"kind": "external", "package": "os"}),
        _ref(3, {"kind": "silent"}),
        _ref(4, {"kind": "internal", "path": "a.py", "qname": "a.py::x", "defLine": 3}),
        _ref(5, {"kind": "internal", "path": "a.py", "qname": "a.py::x", "defLine": 3}),
        _ref(6, {"kind": "internal", "path": "a.py", "qname": "a.py::x", "defLine": 3}),
        _ref(7, {"kind": "silent"}),
        _ref(8, {"kind": "declared", "path": "a.py", "qname": "a.py::x", "defLine": 3}, impls=[]),
    ]
    _write(tmp_path, "python", labels, refs)
    rows, summary = adj.build("python", tmp_path)
    got = {r["ordinal"]: (r["status"], r["reasons"]) for r in rows}
    assert got[1] == ("agreed", []) and got[2] == ("agreed", []) and got[7] == ("agreed", []) and got[8] == ("agreed", [])
    assert got[3] == ("needs-adjudication", ["reference-silent"])
    assert got[4] == ("needs-adjudication", ["label-external-reference-internal"])
    assert got[5] == ("needs-adjudication", ["disagree-targets"])
    assert got[6] == ("needs-adjudication", ["unsure"])
    assert summary["labelled"] == 8 and summary["agreed"] == 4 and summary["agreementRate"] == 0.5
    # every output line carries no Joern column
    for r in rows:
        assert not ({"res", "nm", "joern", "results"} & set(r))


def test_adjudication_input_rejects_bad_label_and_skips_and_site_mismatch(tmp_path):
    adj = _load_adj()
    good = _label(1, [INT("a.py", 3)])
    ref1 = _ref(1, {"kind": "silent"})
    for bad in (
        {**good, "extra": 1},                                  # extra key
        {**good, "confidence": "maybe"},                       # enum
        {**good, "evidence": "x" * 301},                       # length
        {**good, "callees": [{"kind": "internal", "path": "a.py"}]},   # incomplete callee
        {**good, "callees": [{"kind": "external"}]},
        {**good, "notACall": "no"},
    ):
        _write(tmp_path, "python", [bad], [ref1])
        with pytest.raises(adj.LabelSchemaError):
            adj.build("python", tmp_path)
    _write(tmp_path, "python", [_label(2, [])], [_ref(2, {"kind": "silent"})])      # skipped ordinal 1
    with pytest.raises(adj.LabelSchemaError):
        adj.build("python", tmp_path)
    _write(tmp_path, "python", [good], [{**ref1, "siteId": "f" * 16}])               # siteId differs
    with pytest.raises(adj.LabelSchemaError):
        adj.build("python", tmp_path)
    _write(tmp_path, "python", [good, good], [ref1])                                  # repeated / missing reference
    with pytest.raises(adj.LabelSchemaError):
        adj.build("python", tmp_path)


def test_adjudication_input_cli_writes_only_adjudication_dir(tmp_path):
    adj = _load_adj()
    _write(tmp_path, "go", [_label(1, [])], [_ref(1, {"kind": "silent"})])
    assert adj.main(["go", "--root", str(tmp_path)]) == 0
    assert (tmp_path / "adjudication" / "go.input.jsonl").exists()
    with pytest.raises(SystemExit):
        adj.main(["go", "--root", str(tmp_path), "--out", str(tmp_path / "elsewhere" / "x.jsonl")])


def test_adjudication_script_static_firewall():
    """The script may name only labels/, truth/ and adjudication/ -- no sample/, results/, frame/, corpus path, private file."""
    src = (FF / "make_adjudication_input.py").read_text()
    tree = ast.parse(src)
    consts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            consts.append(node.value)
    doc = ast.get_docstring(tree)
    code_strs = [c for c in consts if c != doc and not (len(c) > 200)]
    for forbidden in ("results", "sample", "frame", ".ccg-work", "private", "corpora", "joern", "brief", "final"):
        assert not any(forbidden in c.lower() for c in code_strs), forbidden
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
               {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {"__future__", "argparse", "json", "sys", "pathlib"}
    opens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open"]
    assert len(opens) == 2            # the reader and the writer, nothing else


# ---------------------------------------------------------------- reference mapping, brief, firewall

def test_scip_name_and_package():
    assert rf.scip_name("scip-python python falkorchat 1 `falkorchat.executor`/_str_list().") == "_str_list"
    assert rf.scip_name("scip-java maven . . Shape#area().") == "area"
    assert rf.scip_name("scip-clang cxx . . Shape#area(abcdef0123456789).") == "area"
    assert rf.scip_name("scip-java maven . . Circle#<init>().") == "<init>"
    assert rf.scip_package("scip-python python fastapi 0.1 fastapi/FastAPI#") == "fastapi"


def test_go_callee_parts():
    assert rf.go_callee_parts("(*github.com/x/echo/v5.Echo).Add") == ("github.com/x/echo/v5", "Echo", "Add")
    assert rf.go_callee_parts("github.com/x/echo/v5.New") == ("github.com/x/echo/v5", None, "New")
    assert rf.go_callee_parts("fmt.Println") == ("fmt", None, "Println")


def test_occ_range_both_scip_json_shapes():
    assert rf._occ_range({"range": [3, 4, 9]}) == (3, 4, 3, 9)
    assert rf._occ_range({"range": [3, 4, 5, 9]}) == (3, 4, 5, 9)
    assert rf._occ_range({"TypedRange": {"SingleLineRange": {"line": 7, "start_character": 1, "end_character": 4}}}) == (7, 1, 7, 4)


def test_brief_template_is_verbatim_and_validates():
    t = rf.brief_template()
    assert t.startswith("**You are labelling call sites in a {LANG} codebase, blind.**")
    assert "**Resume:**" in t or "Resume:" in t
    ver, ex = rf.spec_excerpt("go")
    assert ver and "### 4.1 Common grammar" in ex and "#### Go" in ex and "#### C and C++" not in ex
    good = t.replace("{LANG}", "Go").replace("{SAMPLE_FILE}", "s").replace("{CORPUS_ROOT}", "r") \
            .replace("{LABEL_FILE}", "l").replace("{GENERATED_DIRS}", "g").replace("{VENDORED_DIRS}", "v") \
            .replace("{CAP}", "500").replace("{SPEC_VERSION}", "1") + "\n" + ex
    assert rf.validate_brief(good) == []
    assert rf.validate_brief(good.replace("500", "{CAP}", 1))           # an unfilled slot is reported


def test_firewall_scan_flags_forbidden_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "HERE", tmp_path)
    (tmp_path / "frame").mkdir(); (tmp_path / "sample").mkdir()
    (tmp_path / "frame" / "x.jsonl").write_text(json.dumps({"path": "a"}) + "\n")
    (tmp_path / "sample" / "x.jsonl").write_text(json.dumps({"path": "a", "static": 1}) + "\n")
    bad = rf.firewall_scan()
    assert len(bad) == 1 and bad[0].startswith("x.jsonl:1: forbidden key")


def test_committed_artifacts_reproduce_for_fixed_corpora():
    """frame/sample/private/brief of python and javascript are a pure function of corpora.yaml + the pinned tree."""
    langs = [l for l in ("python", "javascript") if rf.src_dir(l).exists() and rf.frame_path(l).exists()]
    if not langs:
        pytest.skip("corpora not fetched (run_fitness.py fetch)")
    r = rf.cmd_verify(langs)
    assert r["problems"] == []


def test_flow_population_length_and_parameter_bounds(tmp_path):
    def fn(name, body_lines, params="a"):
        body = "".join(f"    g(a)\n" if i == 0 else f"    x{i} = {i}\n" for i in range(body_lines - 2))
        return f"def {name}({params}):\n    if a:\n{body}"
    # a function of N lines = 1 (def) + 1 (if) + (N-2) body lines; `if a:` needs a body line, supplied by g(a)
    parts = {"l4": 4, "l5": 5, "l25": 25, "l26": 26}
    src = "".join(fn(n, k) + "\n\n" for n, k in parts.items())
    src += "def noparam():\n    if 1:\n        g(1)\n        h = 2\n        k = 3\n\n"
    src += "class C:\n    def selfonly(self):\n        if 1:\n            g(1)\n            h = 2\n            k = 3\n"
    (tmp_path / "m.py").write_text(src)
    got = sorted(p["qname"].split("::")[1] for p in rf.flow_population("python", tmp_path, {}))
    assert got == ["l25", "l5"]        # 4 and 26 lines out; no parameter / `self` only out
