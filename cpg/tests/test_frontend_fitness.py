"""S25 frontend-fitness harness tests (cpg/spikes/frontend-fitness/run_fitness.py).

Pure-python: tree-sitter frame extraction on snippets, the hash-sort draw, excerpt/column arithmetic,
the C5 pilot, the SCIP/callgraph reference mapping, and the firewall (no Joern/reference text in
sample/ or frame/ files). Run: cd cpg && .venv/bin/python -m pytest -q tests/test_frontend_fitness.py
"""
import json
import os
import re
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


# ---------------------------------------------------------------- ML §5.2 macro-argument rule, §3.2 C5 classes

def _occ(sym, c0=0, c1=1):
    return {"sym": sym, "c0": c0, "c1": c1, "roles": 0}


PRINT = "cxx . . $ testing/Print()."
PRINT2 = "cxx . . $ testing/Print(0123456789abcdef)."


def test_macro_window_symbols_distinct_names_and_bounds():
    occ = {("t.cc", 4): [_occ(PRINT)],                     # before the window start (invocation on line 5)
           ("t.cc", 5): [_occ("cxx . . $ testing/EXPECT_EQ#"), _occ("local 7"), _occ("cxx . . $ testing/Other().")],
           ("t.cc", 6): [_occ(PRINT)], ("t.cc", 7): [_occ(PRINT)],
           ("t.cc", 8): [_occ(PRINT2)],                    # after the site's name line (7)
           ("u.cc", 6): [_occ(PRINT2)]}                    # another file
    assert rf.macro_window_symbols(occ, "t.cc", 5, 7, "Print") == [PRINT]       # one symbol, three occurrences
    assert rf.macro_window_symbols(occ, "t.cc", 5, 8, "Print") == sorted([PRINT, PRINT2])   # end line is inclusive
    assert rf.macro_window_symbols(occ, "t.cc", 4, 7, "Print") == [PRINT]       # start line is inclusive (same symbol)
    assert rf.macro_window_symbols(occ, "t.cc", 5, 5, "Print") == []            # window of one line: nothing named Print
    assert rf.macro_window_symbols(occ, "t.cc", 5, 7, "Nope") == []             # name must equal the callee name
    assert rf.macro_window_symbols(occ, "t.cc", 5, 7, "7") == []                # a local is never a candidate
    near = {("t.cc", 5): [_occ("cxx . . $ testing/PrintAll()."), _occ("cxx . . $ testing/Pri()."), _occ("cxx . . $ testing/Print().")]}
    assert rf.macro_window_symbols(near, "t.cc", 5, 5, "Print") == ["cxx . . $ testing/Print()."]   # equality, not prefix


CPP_SRC = (
    "#ifndef T_H\n"
    "#define T_H\n"
    "int Print(int x) { return x; }\n"
    "void f() {\n"
    "  g(Print(1));\n"
    "  EXPECT_EQ(\"é\", Print(2));\n"
    "  OUTER(1,\n"
    "        INNER(Print(3)));\n"
    "  EXPECT_EQ(1, MY_M(4));\n"
    "  Print(5).run();\n"
    "}\n"
    "#endif\n"
)


def _cpp_rows(tmp_path, src=CPP_SRC, name="t.cc"):
    (tmp_path / name).write_text(src)
    return {(r["line"], r["callText"]): r for r in rf.iter_frame("cpp", tmp_path, {})}


def test_site_node_offset_is_code_points_not_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "src_dir", lambda lang: tmp_path)
    rows = _cpp_rows(tmp_path)
    for key in [(5, "Print"), (6, "Print"), (8, "Print"), (9, "MY_M"), (10, "Print")]:
        r = rows[key]
        node, off, _, data = rf._site_node("cpp", r, {})
        assert data[off:off + len(key[1])].decode() == key[1]
        assert node.type == "identifier" and node.text.decode() == key[1]


def test_macro_invocation_start_outermost_and_args_only(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "src_dir", lambda lang: tmp_path)
    rows = _cpp_rows(tmp_path)
    st = lambda key: rf.macro_invocation_start("cpp", rows[key], {})  # noqa: E731
    assert st((5, "Print")) is None            # g is not macro-shaped
    assert st((6, "Print")) == 6               # EXPECT_EQ on the same line
    assert st((8, "Print")) == 7               # OUTER (line 7) encloses INNER (line 8): the outermost wins
    assert st((8, "INNER")) == 7               # INNER itself is macro-shaped, still inside OUTER's arguments
    assert st((7, "OUTER")) is None            # a macro invocation is not inside its own arguments
    assert st((10, "Print")) is None
    assert rf.macro_invocation_start("c", rows[(6, "Print")], {}) is None   # C++ only


def test_site_tags_closed_classes_cpp(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "src_dir", lambda lang: tmp_path)
    src = ("#ifndef G\n#define G\nvoid f() {\n  a();\n#if WIN\n  b();\n#else\n  c();\n#endif\n  MY_M(1);\n}\n#endif\n")
    rows = _cpp_rows(tmp_path, src)
    tg = lambda key: rf.site_tags("cpp", rows[key], {})  # noqa: E731
    assert tg((4, "a")) == []                                  # the include guard is not a non-guard group
    assert tg((6, "b")) == ["preproc-branch"] and tg((8, "c")) == ["preproc-branch"]
    assert tg((10, "MY_M")) == ["macro-callee"]
    assert set(rf.CLOSED_CLASSES["cpp"]) == {"preproc-branch", "macro-callee"}


def test_site_tags_csharp_and_go_classes():
    row = lambda text, name: {"callText": text, "calleeName": name}  # noqa: E731
    assert rf.site_tags("csharp", row("a.B<int>", "B"), {}) == ["generic-callee"]
    assert rf.site_tags("csharp", row("new C<T>", "C"), {}) == ["generic-callee"]
    assert rf.site_tags("csharp", row("nameof", "nameof"), {}) == ["nameof"]
    assert rf.site_tags("csharp", row("a.B", "B"), {}) == []
    assert rf.site_tags("go", row("len", "len"), {}) == ["builtin"]
    assert rf.site_tags("go", row("a.len", "len"), {}) == ["builtin"]      # name-based class, as the pilot's
    assert rf.site_tags("go", row("foo", "foo"), {}) == []
    assert rf.site_tags("java", row("List<String>", "x"), {}) == []        # no class outside the closed list


def _mk_scip(extra_occ, sym_defs=None):
    """A one-header, one-test SCIP index: `Print` is defined at lib.h:3 (0-based line 2)."""
    doc_h = {"relative_path": "t.h", "occurrences": [{"symbol": PRINT, "symbol_roles": 1, "range": [2, 4, 9]}]}
    doc_t = {"relative_path": "t.cc", "occurrences": extra_occ}
    return {"metadata": {"tool_info": {"name": "scip-clang", "version": "0"}}, "documents": [doc_h, doc_t]}


def _refmap(tmp_path, monkeypatch, src, occ, lang="cpp"):
    (tmp_path / "t.h").write_text("int Print(int x) { return x; }\n\n")
    (tmp_path / "t.cc").write_text(src)
    monkeypatch.setattr(rf, "src_dir", lambda l: tmp_path)
    monkeypatch.setattr(rf, "corpus_cfg", lambda l: {})
    frame = [r for r in rf.iter_frame(lang, tmp_path, {}) if r["path"] == "t.cc"]
    by_key = {r["siteKey"]: r for r in frame}
    sample = [{"ordinal": i, "siteId": f"{i:016x}", "path": r["path"], "line": r["line"], "column": r["column"], "callText": r["callText"]}
              for i, r in enumerate(frame, 1)]
    priv = [{"siteKey": r["siteKey"]} for r in frame]
    rows = rf.refmap_scip(lang, _mk_scip(occ), sample, priv, "0", by_key)
    return {(r["site"]["line"], r["site"]["callText"]): r for r in rows}


def _ref_occ(line1, c0, c1, sym=PRINT):
    return {"symbol": sym, "symbol_roles": 0, "range": [line1 - 1, c0, c1]}


SRC_MACRO = "void f() {\n  EXPECT_EQ(1,\n            Print(2));\n}\n"


def test_refmap_macro_argument_one_symbol_is_relaxed_reference(tmp_path, monkeypatch):
    # scip-clang puts the callee's symbol at the macro invocation's columns (line 2, `EXPECT_EQ`)
    r = _refmap(tmp_path, monkeypatch, SRC_MACRO, [_ref_occ(2, 2, 11)])[(3, "Print")]
    assert r["static"]["kind"] == "internal" and r["static"]["path"] == "t.h" and r["static"]["defLine"] == 3
    assert r.get("relaxed") is True and "ambiguous" not in r and "byConstruction" not in r
    # the macro invocation line itself is in the window (its first line), and so is the site's own line
    r = _refmap(tmp_path, monkeypatch, SRC_MACRO, [_ref_occ(3, 2, 7)])[(3, "Print")]
    assert r.get("relaxed") is True


def test_refmap_macro_argument_window_bounds_and_name(tmp_path, monkeypatch):
    src = "void f() {\n  EXPECT_EQ(1,\n            Print(2));\n  other();\n}\n"
    before = _refmap(tmp_path, monkeypatch, src, [_ref_occ(1, 2, 11)])[(3, "Print")]    # line before the invocation
    after = _refmap(tmp_path, monkeypatch, src, [_ref_occ(4, 2, 7)])[(3, "Print")]      # line after the site's name line
    for r in (before, after):
        assert r["static"] == {"kind": "silent"} and "relaxed" not in r and "ambiguous" not in r
    other = _refmap(tmp_path, monkeypatch, src, [_ref_occ(2, 2, 11, "cxx . . $ testing/Other().")])[(3, "Print")]
    assert other["static"] == {"kind": "silent"} and "relaxed" not in r


def test_refmap_macro_argument_two_symbols_is_silent_ambiguous(tmp_path, monkeypatch):
    r = _refmap(tmp_path, monkeypatch, SRC_MACRO, [_ref_occ(2, 2, 11), _ref_occ(3, 2, 7, PRINT2)])[(3, "Print")]
    assert r["static"] == {"kind": "silent"} and r.get("ambiguous") is True and "relaxed" not in r
    # the same symbol twice is one symbol
    r = _refmap(tmp_path, monkeypatch, SRC_MACRO, [_ref_occ(2, 2, 11), _ref_occ(2, 20, 25)])[(3, "Print")]
    assert r.get("relaxed") is True and "ambiguous" not in r


def test_refmap_macro_rule_applies_everywhere_classes_tag_only_what_stays_silent(tmp_path, monkeypatch):
    # ML §3.2: a site leaves the C5 denominator only if still silent after §5.2's mapping, so the rule applies to a macro-callee
    # site and to a site under a non-guard #if (a compiled branch has occurrences)
    src = "void f() {\n  EXPECT_EQ(1, MY_M(2));\n}\n"
    r = _refmap(tmp_path, monkeypatch, src, [_ref_occ(2, 2, 11, "cxx . . $ testing/MY_M().")])[(2, "MY_M")]
    assert r["static"]["kind"] == "external" and r.get("relaxed") is True and "byConstruction" not in r
    r = _refmap(tmp_path, monkeypatch, src, [])[(2, "MY_M")]
    assert r["static"] == {"kind": "silent"} and r["byConstruction"] == ["macro-callee"] and "relaxed" not in r
    pad = "int pad1; int pad2; int pad3; int pad4; int pad5; int pad6; int pad7; int pad8;\n"
    src = pad + "#if WIN\nvoid f() {\n  EXPECT_EQ(1, Print(2));\n}\n#endif\n"
    r = _refmap(tmp_path, monkeypatch, src, [_ref_occ(4, 2, 11)])[(4, "Print")]
    assert r["static"]["kind"] == "internal" and r.get("relaxed") is True and "byConstruction" not in r
    r = _refmap(tmp_path, monkeypatch, src, [])[(4, "Print")]                    # an uncompiled branch: no occurrence at all
    assert r["static"] == {"kind": "silent"} and r["byConstruction"] == ["preproc-branch"]
    # a site outside any macro argument is never relaxed
    src = "void f() {\n  g(Print(2));\n}\n"
    r = _refmap(tmp_path, monkeypatch, src, [_ref_occ(2, 2, 3)])[(2, "Print")]
    assert r["static"] == {"kind": "silent"} and "relaxed" not in r and "byConstruction" not in r
    # a strict match is kept as is: no relaxed flag
    r = _refmap(tmp_path, monkeypatch, SRC_MACRO, [_ref_occ(3, 12, 17)])[(3, "Print")]
    assert r["static"]["kind"] == "internal" and "relaxed" not in r


CTOR = "cxx . . $ testing/Double#Double(ba8836b59fb42830)."
CTOR2 = "cxx . . $ testing/Double#Double(0123456789abcdef)."
TYPE = "cxx . . $ testing/Double#"


def test_macro_window_merges_a_type_symbol_into_its_constructor_only():
    occ = {("t.cc", 5): [_occ(TYPE), _occ(CTOR)]}
    assert rf.macro_window_symbols(occ, "t.cc", 5, 5, "Double") == [CTOR]
    assert rf.macro_window_symbols(occ, "t.cc", 5, 5, "Double", merge_type_ctor=False) == sorted([TYPE, CTOR])
    assert rf.macro_window_symbols({("t.cc", 5): [_occ(TYPE)]}, "t.cc", 5, 5, "Double") == [TYPE]      # a lone type stays
    assert rf.macro_window_symbols({("t.cc", 5): [_occ(CTOR), _occ(CTOR2)]}, "t.cc", 5, 5, "Double") == sorted([CTOR, CTOR2])
    nested = "cxx . . $ testing/Double#Inner#Double(0123456789abcdef)."      # a nested type's constructor is another target
    assert rf.macro_window_symbols({("t.cc", 5): [_occ(TYPE), _occ(nested)]}, "t.cc", 5, 5, "Double") == sorted([TYPE, nested])
    scoped = "cxx . . $ testing/Double#sub/Double(0123456789abcdef)."       # a scoped (not member) descriptor is not its constructor
    assert rf.macro_window_symbols({("t.cc", 5): [_occ(TYPE), _occ(scoped)]}, "t.cc", 5, 5, "Double") == sorted([TYPE, scoped])
    # a constructor of ANOTHER type with the same name does not absorb the type
    other = "cxx . . $ testing/Other#Double(0123456789abcdef)."
    assert rf.macro_window_symbols({("t.cc", 5): [_occ(TYPE), _occ(other)]}, "t.cc", 5, 5, "Double") == sorted([TYPE, other])


def test_refmap_macro_argument_constructor_pair_is_one_symbol_flagged(tmp_path, monkeypatch):
    src = "void f() {\n  EXPECT_EQ(1,\n            Double(2));\n}\n"
    r = _refmap(tmp_path, monkeypatch, src, [_ref_occ(2, 2, 11, TYPE), _ref_occ(2, 2, 11, CTOR)])[(3, "Double")]
    assert r.get("relaxed") is True and r.get("typeCtorMerged") is True and "ambiguous" not in r
    r = _refmap(tmp_path, monkeypatch, src, [_ref_occ(2, 2, 11, CTOR)])[(3, "Double")]
    assert r.get("relaxed") is True and "typeCtorMerged" not in r


def test_refmap_error_region_is_informational_not_a_class(tmp_path, monkeypatch):
    src = "void f( {\n  Print(2);\n}\n"                     # tree-sitter ERROR region, no occurrence
    r = _refmap(tmp_path, monkeypatch, src, [])[(2, "Print")]
    assert rf.site_tags("cpp", {"path": "t.cc", "line": 2, "column": 3, "callText": "Print", "calleeName": "Print", "kind": "plain"}, {}) == ["error-region"]
    assert r["static"] == {"kind": "silent"} and "byConstruction" not in r


def test_refmap_byconstruction_only_on_silent_rows(tmp_path, monkeypatch):
    src = "void f() {\n  a(Print(2));\n}\n"
    r = _refmap(tmp_path, monkeypatch, src, [_ref_occ(2, 4, 9)])[(2, "Print")]
    assert r["static"]["kind"] == "internal" and "byConstruction" not in r


def _prow(sid, kind, **kw):
    d = {"static": {"kind": kind}}
    d.update(kw)
    return sid, d


def test_pilot_coverage_raw_adjusted_cap_and_strict():
    pilot = [{"siteId": f"s{i}", "path": "x", "line": 1, "column": 1, "callText": "f", "calleeName": "f"} for i in range(50)]
    rows = {}
    for i in range(50):
        rows[f"s{i}"] = {"static": {"kind": "internal"}}
    for i in range(0, 3):
        rows[f"s{i}"] = {"static": {"kind": "internal"}, "relaxed": True}           # covered by §5.2's rule
    for i in range(3, 10):
        rows[f"s{i}"] = {"static": {"kind": "silent"}, "byConstruction": ["generic-callee"]}
    rows["s10"] = {"static": {"kind": "silent"}, "byConstruction": ["nameof", "generic-callee"]}   # two classes, one site
    rows["s11"] = {"static": {"kind": "silent"}, "ambiguous": True}
    rows["s12"] = {"static": {"kind": "silent"}}
    c = rf.pilot_coverage("csharp", pilot, rows)
    assert (c["silent"], c["withReference"], c["coverage"]) == (10, 40, 0.8)
    assert c["relaxed"] == 3 and c["coverageStrict"] == round(37 / 50, 4)
    assert c["ambiguous"] == 1
    assert c["excluded"] == 8 and c["excludedByClass"] == {"generic-callee": 8, "nameof": 1}
    assert c["adjustedDenominator"] == 42 and c["adjustedCoverage"] == round(40 / 42, 4)
    assert c["exclusionCapExceeded"] is False and c["exclusionCap"] == 10
    rows["s13"] = {"static": {"kind": "silent"}, "byConstruction": ["generic-callee"]}
    rows["s14"] = {"static": {"kind": "silent"}, "byConstruction": ["generic-callee"]}
    c = rf.pilot_coverage("csharp", pilot, rows)
    assert c["excluded"] == 10 and c["exclusionCapExceeded"] is False       # exactly 10 of 50 is allowed
    for i in range(15, 16):
        rows[f"s{i}"] = {"static": {"kind": "silent"}, "byConstruction": ["generic-callee"]}
    c = rf.pilot_coverage("csharp", pilot, rows)
    assert c["excluded"] == 11 and c["exclusionCapExceeded"] is True       # the cap is a flag, never silently clipped
    assert c["coverage"] == round(37 / 50, 4)                               # raw is untouched


# ---------------------------------------------------------------- C# generic audit draw (ML §3.2, prefix ccg-p0a-v1-gen|)

def _gen_inputs():
    pub = [{"ordinal": i, "siteId": f"{i:016x}", "callText": t} for i, t in
           enumerate(["a.B<int>", "x.y", "nameof", "new C<T>", "p.q<int>", "z"], 1)]
    priv = [{"ordinal": i, "siteKey": f"f.cs:{i}:1:{p['callText']}", "calleeName": p["callText"].split(".")[-1].split("<")[0].replace("new ", "")}
            for i, p in enumerate(pub, 1)]
    return pub, priv


def test_gen_audit_order_is_hash_sort_over_class_sites_only(monkeypatch):
    monkeypatch.setattr(rf, "corpus_cfg", lambda l: {"commit": "abc"})
    pub, priv = _gen_inputs()
    ref = [{"ordinal": i, "static": {"kind": "silent"}, "byConstruction": ["generic-callee"]} for i in (1, 3, 5)]
    ref += [{"ordinal": 4, "static": {"kind": "internal"}}, {"ordinal": 2, "static": {"kind": "silent"}},
            {"ordinal": 6, "static": {"kind": "silent"}, "byConstruction": ["generic-callee"]}]
    got = rf.build_gen_audit("csharp", pub, priv, ref)
    # ordinal 4 is a class site the reference speaks on, 6 is silent but not in a class text-wise, 2 is silent in no class
    assert sorted(g["ordinal"] for g in got) == [1, 3, 5]
    seed = "ccg-p0a-v1-gen|csharp|abc"
    want = sorted((rf.sha256hex(seed + "|" + pr["siteKey"]), pr["siteKey"], pr["ordinal"]) for pr in priv if pr["ordinal"] in (1, 3, 5))
    assert [g["ordinal"] for g in got] == [w[2] for w in want]
    assert [g["hash"] for g in got] == [w[0] for w in want] and [g["rank"] for g in got] == [1, 2, 3]
    assert {g["ordinal"]: g["classes"] for g in got}[3] == ["nameof"]
    assert not ({"static", "res", "label"} & set(got[0]))
    with pytest.raises(SystemExit):
        rf.build_gen_audit("go", pub, priv, ref)


def test_gen_audit_select_all_internal_and_first_k_external():
    order = [{"rank": i, "ordinal": i, "siteId": f"{i:016x}", "hash": "h", "classes": ["generic-callee"]} for i in range(1, 11)]
    ext = {"kind": "external", "package": "p"}
    intl = {"kind": "internal", "path": "a", "qname": None, "defLine": 1}
    decl = {"kind": "declared", "path": "a", "qname": None, "defLine": 2}
    labels = {}
    for i in range(1, 11):
        labels[i] = {"ordinal": i, "callees": [intl] if i in (2, 9) else [decl] if i == 5 else [ext], "notACall": i == 10}
    got = rf.gen_audit_select(order, [labels[i] for i in labels], k=3)
    assert [(g["ordinal"], g["stratum"]) for g in got] == [(1, "external"), (2, "internal"), (3, "external"), (4, "external"), (5, "internal"), (9, "internal")]
    assert rf.gen_audit_select(order, [labels[i] for i in labels], k=40)[-1]["ordinal"] == 9 and len(rf.gen_audit_select(order, list(labels.values()), k=40)) == 9


def test_adjudication_input_carries_stratum_tags_of_the_reference(tmp_path):
    adj = _load_adj()
    labels = [_label(1, [INT("a.py", 3)]), _label(2, [INT("a.py", 3)]), _label(3, [], notACall=True)]
    refs = [_ref(1, INT("a.py", 3), relaxed=True), _ref(2, {"kind": "silent"}, ambiguous=True),
            _ref(3, {"kind": "silent"}, byConstruction=["builtin"])]
    _write(tmp_path, "python", labels, refs)
    rows, _ = adj.build("python", tmp_path)
    ref = {r["ordinal"]: r["reference"] for r in rows}
    assert ref[1]["relaxed"] is True and ref[2]["ambiguous"] is True and ref[3]["byConstruction"] == ["builtin"]
    assert "relaxed" not in ref[2] and "byConstruction" not in ref[1]


# ---------------------------------------------------------------- S56 flow brief + flowcheck (flow-brief.template.md)

FLOW_LANGS = ["python", "javascript", "java", "kotlin", "csharp", "go", "cpp"]
FSRC = "def handle(req, flag):\n    if flag and check(req):\n        return run(req)\n    return run(req)"


def _flow_sample(n=3):
    return [{"ordinal": i, "path": f"m{i}.py", "qname": f"m{i}.py::handle", "lineStart": 10, "lineEnd": 13, "source": FSRC} for i in range(1, n + 1)]


@pytest.fixture
def flowenv(tmp_path, monkeypatch):
    (tmp_path / "sample").mkdir()
    (tmp_path / "labels").mkdir()
    monkeypatch.setattr(rf, "flow_path", lambda l: tmp_path / "sample" / f"{l}-flow.jsonl")
    monkeypatch.setattr(rf, "flow_label_path", lambda l: tmp_path / "labels" / f"{l}-flow.jsonl")
    monkeypatch.setattr(rf, "flow_brief_path", lambda l: tmp_path / "sample" / f"{l}-flow.brief.md")
    (tmp_path / "sample" / "x").write_text("")
    for l in FLOW_LANGS:
        (tmp_path / "sample" / f"{l}-flow.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _flow_sample()))
    return tmp_path


@pytest.mark.parametrize("lang", FLOW_LANGS)
def test_flow_brief_fills_validates_and_is_deterministic(flowenv, lang):
    a, b = rf.fill_flow_brief(lang), rf.fill_flow_brief(lang)
    assert a == b and rf.validate_flow_brief(lang, a) == []
    assert "{N_METHODS}" not in a and f"inside 3 {rf.LANG_LABEL[lang]} methods" in a and "ordinal **3**" in a                           # the count came from the sample file
    assert f"The short-circuit operators of {rf.LANG_LABEL[lang]} are {rf.FLOW_OPS[lang]}." in a
    assert rf.FLOW_OP_EXAMPLES[lang] in a and rf.FLOW_SCOPE_NOTES[lang] in a and rf.FLOW_RECEIVER_NOTE[lang] in a
    assert len(re.findall("joern", a, flags=re.I)) == 1
    assert str(flowenv / "labels" / f"{lang}-flow.jsonl") in a


def test_flow_brief_operator_lists_differ_per_language(flowenv):
    ops = {l: rf.FLOW_OPS[l] for l in FLOW_LANGS}
    assert "`and`, `or`" == ops["python"] and "`??`" in ops["javascript"] and "`??`" in ops["csharp"]
    assert "elvis" in ops["kotlin"] and "`??`" not in ops["java"] and "`??`" not in ops["go"]
    for l in FLOW_LANGS:
        t = rf.fill_flow_brief(l)
        for o in FLOW_LANGS:
            if ops[o] != ops[l]:
                assert f"are {ops[o]}." not in t


def test_flow_spec_excerpt_is_redacted_and_names_sections():
    x = rf.flow_spec_excerpt()
    for h in ("### 5.4 `param`", "### 5.5 `ret` and `cond`", "### 5.6 `callsite`", "### 5.7 The call-site ordinal"):
        assert h in x
    assert "Joern" not in x and "CDG sources decide" not in x and "<operator>" not in x
    assert "**Which calls are call sites:**" in x and "(language operators)" in x


def test_flow_spec_redaction_fails_loudly_when_spec_drifts(tmp_path, monkeypatch):
    t = rf.SPEC3.read_text(encoding="utf-8")
    import re as _re
    drift55 = _re.sub(r"Joern's CDG sources decide", "The comparison tool decides", t)
    drift56 = t.replace("(Joern's `<operator>.*` calls)", "(Joern's operator calls)")
    drift_bullet = t.replace("- **Decorator and annotation", "- **Decorators and annotations")
    assert drift55 != t and drift56 != t and drift_bullet != t
    for name, txt in (("a.md", drift55), ("b.md", drift56), ("c.md", drift_bullet)):
        (tmp_path / name).write_text(txt, encoding="utf-8")
        monkeypatch.setattr(rf, "SPEC3", tmp_path / name)
        with pytest.raises(SystemExit):
            rf.flow_spec_excerpt()


@pytest.mark.parametrize("leak", ["see results/ here", "the truth/ dir", "ref.jsonl", "adjudication log", "run extract.sc", "the projection output",
                                  "CONTROLLED_BY edges", "CDG", "DATA edge", "CALLS_NAME", "FULL_NAME", "cpg.bin", "0.95", "recall",
                                  "threshold", "flowAvailable", "labels/python.jsonl", "x.brief.md", "S26", "S28", "S56", "S57"])
def test_flow_brief_leak_patterns_each_reject(flowenv, leak):
    good = rf.fill_flow_brief("python")
    assert rf.validate_flow_brief("python", good) == []
    errs = rf.validate_flow_brief("python", good + "\n" + leak + "\n")
    assert any("leak pattern" in e for e in errs)


def test_flow_brief_validator_rejects_slots_sections_ops_joern_count(flowenv):
    good = rf.fill_flow_brief("python")
    assert any("unfilled" in e for e in rf.validate_flow_brief("python", good + " {SCOPE_NOTES}"))
    for must in ("**Output.**", "**What counts as a condition (`cond`).**", "short-circuit operands that guard a point", "**What to label: pairs.**",
                 "**Appendix: bundle spec points**", "### 5.4 `param`", "### 5.5 `ret` and `cond`", "### 5.7 The call-site ordinal",
                 rf.FLOW_OPS["python"], rf.FLOW_OP_EXAMPLES["python"]):
        assert any("missing" in e for e in rf.validate_flow_brief("python", good.replace(must, "REMOVED", 1))), must
    # the per-language operator list of ANOTHER language does not satisfy the check
    assert any("missing" in e for e in rf.validate_flow_brief("python", good.replace(rf.FLOW_OPS["python"], rf.FLOW_OPS["go"])))
    assert any("Joern" in e for e in rf.validate_flow_brief("python", good + " and Joern again"))
    assert any("Joern" in e for e in rf.validate_flow_brief("python", good + " and joern again"))      # any case
    assert any("Joern" in e for e in rf.validate_flow_brief("python", good.replace("Joern", "the tool")))
    js = rf.fill_flow_brief("javascript")
    assert rf.validate_flow_brief("javascript", js) == []
    for must in ("`&&`, `||` and `??`", "Every operand of such an operator", "`props.onClick && props.onClick(e)`"):
        assert any("javascript: missing" in e for e in rf.validate_flow_brief("javascript", js.replace(must, "REMOVED"))), must


def test_flowbrief_command_writes_file_and_refuses_a_leak(flowenv, monkeypatch):
    r = rf.cmd_flowbrief("go")
    assert (flowenv / "sample" / "go-flow.brief.md").read_text(encoding="utf-8") == rf.fill_flow_brief("go") and r["lang"] == "go"
    monkeypatch.setitem(rf.FLOW_SCOPE_NOTES, "go", "see truth/ for details")
    with pytest.raises(SystemExit):
        rf.cmd_flowbrief("go")
    (flowenv / "sample" / "go-flow.jsonl").unlink()
    with pytest.raises(SystemExit):
        rf.fill_flow_brief("go")


def test_verify_rederives_the_flow_brief(flowenv, monkeypatch):
    langs = [l for l in ("python",) if rf.src_dir(l).exists() and rf.frame_path(l).exists()]
    if not langs:
        pytest.skip("corpora not fetched")
    (flowenv / "sample" / "python-flow.jsonl").write_text((FF / "sample" / "python-flow.jsonl").read_text())
    rf.cmd_flowbrief("python")
    assert [p for p in rf.cmd_verify(langs)["problems"] if "flow brief" in p] == []
    p = flowenv / "sample" / "python-flow.brief.md"
    p.write_text(p.read_text() + "\nedited\n")
    assert "python: flow brief differs" in rf.cmd_verify(langs)["problems"]
    p.write_text(rf.fill_flow_brief("python").replace("**Output.**", "x"))
    assert "python: flow brief invalid" in rf.cmd_verify(langs)["problems"]


# ---- flowcheck

def _lab(ordinal=1, pairs=None, **kw):
    d = {"ordinal": ordinal, "path": f"m{ordinal}.py", "qname": f"m{ordinal}.py::handle", "pairs": pairs if pairs is not None else [GOOD_PAIR]}
    d.update(kw)
    return d


GOOD_PAIR = {"kind": "data", "source": {"point": "param", "index": 0, "name": "req"},
             "sink": {"point": "call", "calleeName": "check", "ordinal": 0, "line": 11}, "evidence": "req passed to check", "confidence": "sure"}
CTRL = {"kind": "control", "source": {"point": "cond", "line": 11, "code": "flag and check(req)"},
        "sink": {"point": "ret", "ordinal": 0, "line": 12}, "evidence": "if guards return", "confidence": "unsure"}


def _check(flowenv, rows, lang="python"):
    (flowenv / "labels" / f"{lang}-flow.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rf.cmd_flowcheck(lang)


def test_flowcheck_accepts_valid_and_reports_counts(flowenv):
    r = _check(flowenv, [_lab(1, [GOOD_PAIR, CTRL]), _lab(2, [], noPairReason="no branch reaches a call"), _lab(3, [GOOD_PAIR], unlisted=[{"line": 11, "construct": "walrus"}])])
    assert r["nErrors"] == 0 and r["complete"] is True and r["pairs"] == 3 and r["data"] == 2 and r["control"] == 1
    assert r["unsure"] == 1 and r["withPair"] == 2 and r["withoutPair"] == 1 and r["belowFloor120"] is True
    assert _check(flowenv, [_lab(1)])["complete"] is False and _check(flowenv, [_lab(1)])["nErrors"] == 0     # resumable
    assert rf.cmd_flowcheck("go")["labelled"] == 0


def _pair(**over):
    p = json.loads(json.dumps(GOOD_PAIR))
    for k, v in over.items():
        p[k] = v
    return p


@pytest.mark.parametrize("name,rows,frag", [
    ("call-ordinal-too-high", [_lab(1, [_pair(sink={"point": "call", "calleeName": "check", "ordinal": 1, "line": 11})])], "fewer occurrences"),
    ("callee-not-in-method", [_lab(1, [_pair(sink={"point": "call", "calleeName": "nope", "ordinal": 0, "line": 11})])], "fewer occurrences"),
    ("cond-code-not-in-method", [_lab(1, [dict(CTRL, source={"point": "cond", "line": 11, "code": "flag or other"})])], "cond code"),
    ("cond-as-data-source", [_lab(1, [dict(CTRL, kind="data")])], "cannot be a data source"),
    ("param-as-control-source", [_lab(1, [_pair(kind="control")])], "cannot be a control source"),
    ("source-equals-sink", [_lab(1, [_pair(kind="data", source={"point": "ret", "ordinal": 0, "line": 12}, sink={"point": "ret", "ordinal": 0, "line": 12})])], "cannot be a data source"),
    ("source-equals-sink-cond", [_lab(1, [dict(CTRL, sink=CTRL["source"])])], "source equals sink"),
    ("skipped-ordinal", [_lab(2)], "expected ordinal 1"),
    ("line-below-span", [_lab(1, [_pair(sink={"point": "call", "calleeName": "check", "ordinal": 0, "line": 9})])], "outside 10-13"),
    ("line-above-span", [_lab(1, [_pair(sink={"point": "call", "calleeName": "check", "ordinal": 0, "line": 14})])], "outside 10-13"),
    ("evidence-too-long", [_lab(1, [_pair(evidence="x" * 301)])], "evidence"),
    ("evidence-empty", [_lab(1, [_pair(evidence=" ")])], "evidence"),
    ("empty-pairs-no-reason", [_lab(1, [])], "noPairReason"),
    ("duplicate-pair", [_lab(1, [GOOD_PAIR, GOOD_PAIR])], "duplicate"),
    ("wrong-qname", [_lab(1, qname="other")], "expected ordinal 1"),
    ("extra-key-in-endpoint", [_lab(1, [_pair(source={"point": "param", "index": 0, "name": "req", "x": 1})])], "keys"),
    ("bad-point", [_lab(1, [_pair(source={"point": "zzz"})])], "bad point"),
    ("bad-confidence", [_lab(1, [_pair(confidence="maybe")])], "kind/confidence"),
    ("param-index-below-minus-1", [_lab(1, [_pair(source={"point": "param", "index": -2, "name": "req"})])], "bad param"),
    ("non-int-line", [_lab(1, [_pair(sink={"point": "call", "calleeName": "check", "ordinal": 0, "line": "11"})])], "non-integer"),
    ("call-name-is-whole-word", [_lab(1, [_pair(sink={"point": "call", "calleeName": "chec", "ordinal": 0, "line": 11})])], "fewer occurrences"),
    ("data-sink-param", [_lab(1, [_pair(sink={"point": "param", "index": 0, "name": "req"})])], "cannot be a data sink"),
    ("ordinal-not-sample-ordinal", [dict(_lab(1), ordinal=2)], "expected ordinal 1"),
    ("path-not-sample-path", [dict(_lab(1), path="zzz.py")], "expected ordinal 1"),
    ("bad-unlisted", [_lab(1, unlisted=[{"line": 11}])], "unlisted"),
    ("more-lines-than-methods", [_lab(1), _lab(2), _lab(3), _lab(4)], "more lines than methods"),
    ("unknown-top-key", [_lab(1, extra=1)], "keys"),
])
def test_flowcheck_rejects(flowenv, name, rows, frag):
    r = _check(flowenv, rows)
    assert r["nErrors"] >= 1 and any(frag in e for e in r["errors"]), (name, r["errors"])
    assert r["complete"] is False


def test_flowcheck_boundaries_whitespace_length_and_complete(flowenv):
    edge = [_pair(sink={"point": "call", "calleeName": "check", "ordinal": 0, "line": 10}),
            _pair(sink={"point": "call", "calleeName": "check", "ordinal": 0, "line": 13})]          # both ends of lineStart..lineEnd
    ws_cond = dict(CTRL, source={"point": "cond", "line": 11, "code": "flag   and\n check(req)"})      # whitespace collapses
    assert _check(flowenv, [_lab(1, edge + [ws_cond])])["nErrors"] == 0
    cond = "x == " + "a" * 250
    rows = [dict(r, source="if " + cond + ":\n    f(x)") for r in _flow_sample()]
    (flowenv / "sample" / "python-flow.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = _check(flowenv, [_lab(1, [dict(CTRL, source={"point": "cond", "line": 11, "code": cond})])])
    assert any("cond code" in e for e in r["errors"])
    (flowenv / "sample" / "python-flow.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _flow_sample()))
    r = _check(flowenv, [_lab(1), _lab(2, [_pair(evidence="")]), _lab(3)])
    assert r["labelled"] == 3 and r["nErrors"] == 1 and r["complete"] is False
    assert _check(flowenv, [_lab(1), _lab(2), _lab(3)])["complete"] is True


def test_flowcheck_cli_exit_codes_and_not_json(flowenv, capsys):
    (flowenv / "labels" / "python-flow.jsonl").write_text("{not json\n")
    assert rf.cmd_flowcheck("python")["nErrors"] == 1
    (flowenv / "labels" / "python-flow.jsonl").write_text(json.dumps(_lab(1)) + "\n")
    assert rf.main(["flowcheck", "python"]) == 0
    (flowenv / "labels" / "python-flow.jsonl").write_text(json.dumps(_lab(2)) + "\n")
    assert rf.main(["flowcheck", "python"]) == 1


# ---------------------------------------------------------------- TS/JS prod-only census (ML v1.6 §3.1, bundle spec4 §1 `test` pattern)

@pytest.mark.parametrize("rel,expected", [
    ("src/a.ts", False), ("src/a.test.ts", True), ("src/a.spec.tsx", True), ("src/a.test.js", True), ("src/a.spec.mjs", True),
    ("e2e/a.ts", True), ("src/e2e/a.ts", True), ("src/__tests__/a.ts", True), ("src/__tests__/deep/er/a.ts", True),
    ("tests/a.ts", False), ("src/test/a.ts", False), ("src/attest.ts", False), ("src/e2e.ts", False), ("src/__tests__.ts", False),
    ("src/a.test.json", False), ("src/__tests__", False), ("src/a.test.ts.bak", False), ("src/a.spec", False), ("src/test.ts", False), ("src/a.tests.ts", False), ("src/a.test.d.ts", False),   # declaration files leave the frame earlier
])
def test_is_test_file_javascript_pattern(rel, expected):
    assert rf.is_test_file("javascript", rel) is expected


@pytest.mark.parametrize("lang", ["python", "java", "kotlin", "csharp", "go", "cpp"])
def test_is_test_file_is_javascript_only(lang):
    for rel in ("a.test.ts", "e2e/a.py", "src/__tests__/a.go", "tests/test_a.py", "a.spec.js"):
        assert rf.is_test_file(lang, rel) is False


PLANTED = {"src/a.ts": "export function a(x: number) { return f(x); }\n",
           "src/a.test.ts": "it('x', () => { g(1); h(2); });\n",
           "src/b.spec.tsx": "it('y', () => { g(3); });\n",
           "e2e/flow.ts": "run(1);\n",
           "src/__tests__/c.ts": "check(2);\n",
           "tests/kept.ts": "keep(3);\n"}


def _plant(tmp_path, files=PLANTED):
    for rel, txt in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
    return tmp_path


def test_frame_drops_planted_test_files_for_javascript_only(tmp_path):
    root = _plant(tmp_path)
    got = sorted({r["path"] for r in rf.iter_frame("javascript", root, {})})
    assert got == ["src/a.ts", "tests/kept.ts"]                 # `tests/` is not in the closed pattern
    py = tmp_path / "py"
    (py / "tests").mkdir(parents=True)
    (py / "tests" / "test_a.py").write_text("f(1)\n")
    (py / "a.test.py").write_text("g(1)\n")
    assert sorted({r["path"] for r in rf.iter_frame("python", py, {})}) == ["a.test.py", "tests/test_a.py"]    # Python keeps tests


def test_flow_population_drops_planted_test_files(tmp_path):
    body = "export function m(a: number) {\n  if (a) {\n    f(a);\n  }\n  return a;\n}\n"
    _plant(tmp_path, {"src/m.ts": body, "src/m.test.ts": body, "e2e/m.ts": body, "src/__tests__/m.ts": body})
    pop = rf.flow_population("javascript", tmp_path, {})
    assert [p["path"] for p in pop] == ["src/m.ts"]
    pyp = tmp_path / "py"
    pyp.mkdir()
    (pyp / "test_m.py").write_text("def m(a):\n    x = 1\n    if a:\n        f(a)\n    return a\n")
    assert [p["path"] for p in rf.flow_population("python", pyp, {})] == ["test_m.py"]


def test_flow_population_line_bounds_parameters(tmp_path):
    def fn(n):
        return "export function m%d(a: number) {\n  if (a) {\n    f(a);\n  }\n%s  return a;\n}\n" % (n, "  x();\n" * (n - 6))
    (tmp_path / "m.ts").write_text("".join(fn(n) for n in (6, 25, 26, 60, 61)))
    names = lambda lo, hi: sorted(p["qname"].split("::")[1] for p in rf.flow_population("javascript", tmp_path, {}, lo, hi))  # noqa: E731
    assert names(5, 25) == ["m25", "m6"]
    assert names(26, 60) == ["m26", "m60"]


def test_build_sample_is_census_for_javascript_and_capped_elsewhere(tmp_path, monkeypatch):
    (tmp_path / "f.ts").write_text("\n".join(f"a{i}();" for i in range(7)) + "\n")
    monkeypatch.setattr(rf, "src_dir", lambda l: tmp_path)
    monkeypatch.setattr(rf, "corpus_cfg", lambda l: {"commit": "abc"})
    monkeypatch.setattr(rf, "CAP", 3)
    rows = list(rf.iter_frame("javascript", tmp_path, {}))
    assert len(rows) == 7
    pub, priv = rf.build_sample("javascript", rows)
    assert [p["ordinal"] for p in pub] == list(range(1, 8)) and len(priv) == 7
    pub2, _ = rf.build_sample("python", rows)
    assert len(pub2) == 3
    want = [h[:16] for h, _ in rf.hash_sort(rows, rf.seed_for("sample", "python", "abc"))[:3]]
    assert [p["siteId"] for p in pub2] == want                                      # the cap is a prefix of the §4.4 order
    assert [p["siteId"] for p in pub] == [h[:16] for h, _ in rf.hash_sort(rows, rf.seed_for("sample", "javascript", "abc"))]


def test_census_brief_has_n_and_no_stop_rule():
    if not rf.sample_path("javascript").exists():
        pytest.skip("javascript sample not drawn")
    n = len(rf.read_jsonl(rf.sample_path("javascript")))
    t = rf.fill_brief("javascript")
    assert rf.validate_brief(t, "javascript") == []
    assert f"a census of {n} sites" in t and f"ordinal **{n}**" in t
    assert "**150**" not in t and "whichever comes first" not in t and "{CAP}" not in t
    assert any("stop rule" in e for e in rf.validate_brief(t + "\n**150**, or after ordinal **500**, whichever comes first.", "javascript"))
    if rf.sample_path("python").exists():
        py = rf.fill_brief("python")
        assert "**150**" in py and "whichever comes first" in py and rf.validate_brief(py, "python") == []


def test_census_brief_fails_loudly_if_the_template_stop_rule_drifts(monkeypatch):
    if not rf.sample_path("javascript").exists():
        pytest.skip("javascript sample not drawn")
    monkeypatch.setattr(rf, "brief_template", lambda: "x {CAP} y\n")
    with pytest.raises(SystemExit):
        rf.fill_brief("javascript")


def test_committed_javascript_census_invariants():
    names = [rf.frame_path("javascript"), rf.sample_path("javascript"), rf.private_path("javascript"), rf.flow_path("javascript"),
             rf.flow_tier2_path("javascript"), FF / "truth" / "javascript.ref.jsonl", FF / "truth" / "javascript.c5-pilot.jsonl"]
    if not all(p.exists() for p in names):
        pytest.skip("javascript artifacts absent")
    frame, sample, priv, flow, tier2, ref, pilot = (rf.read_jsonl(p) for p in names)
    assert len(frame) == 523 and len({r["path"] for r in frame}) == 40
    assert not [r for r in frame if rf.is_test_file("javascript", r["path"])]
    assert [s["ordinal"] for s in sample] == list(range(1, 524)) and len({s["siteId"] for s in sample}) == 523
    assert {p["siteKey"] for p in priv} == {r["siteKey"] for r in frame}            # the census is the whole frame
    sha = rf.corpus_cfg("javascript")["commit"]
    order = rf.hash_sort(frame, rf.seed_for("sample", "javascript", sha))
    assert [p["siteKey"] for p in priv] == [r["siteKey"] for _, r in order]       # §4.4 hash order
    assert [r["ordinal"] for r in ref] == list(range(1, 524)) and [r["siteId"] for r in ref] == [s["siteId"] for s in sample]
    assert len(flow) == 12 and len(tier2) == 3 and not [m for m in flow + tier2 if rf.is_test_file("javascript", m["path"])]
    assert all(5 <= m["lineEnd"] - m["lineStart"] + 1 <= 25 for m in flow) and all(26 <= m["lineEnd"] - m["lineStart"] + 1 <= 60 for m in tier2)
    assert len(pilot) == 50 and {p["siteKey"] for p in pilot} <= {r["siteKey"] for r in frame}
    pk = [p["siteKey"] for p in pilot]
    assert pk == [r["siteKey"] for _, r in rf.hash_sort(frame, rf.seed_for("c5", "javascript", sha))[:50]]


def test_cmd_brief_validates_with_the_language(monkeypatch, tmp_path):
    monkeypatch.setattr(rf, "brief_path", lambda l: tmp_path / f"{l}.brief.md")      # a mutant must not write the real brief
    stop = "text **150**, or after ordinal **9**, whichever comes first.\n**Output.** **Writing a target.** **Appendix: `qname` grammar** ### 4.1 Common grammar"
    monkeypatch.setattr(rf, "fill_brief", lambda lang: stop)
    with pytest.raises(SystemExit):
        rf.cmd_brief("javascript")


def test_tier2_flow_draw_is_26_to_60_lines_at_most_three_in_hash_order(tmp_path, monkeypatch):
    def fn(n):
        return "export function m%d(a: number) {\n  if (a) {\n    f(a);\n  }\n%s  return a;\n}\n" % (n, "  x();\n" * (n - 6))
    (tmp_path / "m.ts").write_text("".join(fn(n) for n in (25, 26, 30, 40, 60, 61)))
    monkeypatch.setattr(rf, "src_dir", lambda l: tmp_path)
    monkeypatch.setattr(rf, "corpus_cfg", lambda l: {"commit": "abc"})
    got = rf.build_flow_tier2("javascript")
    pop = rf.flow_population("javascript", tmp_path, {}, 26, 60)
    assert len(pop) == 4 and len(got) == 3 and [g["ordinal"] for g in got] == [1, 2, 3]
    assert [g["qname"] for g in got] == [m["qname"] for _, m in rf.hash_sort(pop, rf.seed_for("flow", "javascript", "abc"))[:3]]
    names = {g["qname"].split("::")[1] for g in got}
    assert not (names & {"m25", "m61"})
    assert all(26 <= g["lineEnd"] - g["lineStart"] + 1 <= 60 for g in got)


def test_verify_checks_the_tier2_flow_sample(tmp_path, monkeypatch):
    if not (rf.src_dir("javascript").exists() and rf.flow_tier2_path("javascript").exists()):
        pytest.skip("javascript artifacts absent")
    assert [p for p in rf.cmd_verify(["javascript"])["problems"] if "tier-2" in p] == []
    rows = rf.read_jsonl(rf.flow_tier2_path("javascript"))[:2]
    alt = tmp_path / "t2.jsonl"
    alt.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(rf, "flow_tier2_path", lambda l: alt)
    assert "javascript: flow tier-2 sample differs" in rf.cmd_verify(["javascript"])["problems"]
    alt.unlink()
    assert "javascript: flow tier-2 sample differs" in rf.cmd_verify(["javascript"])["problems"]
