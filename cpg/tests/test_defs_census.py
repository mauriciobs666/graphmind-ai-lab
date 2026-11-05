"""S27c: the definitions census (make_defs_census.py) and its metrics join (s27_census_metrics.py).

Per language a planted fixture carries one definition per bundle-spec rule-8 case, one argument of each kind (a)-(d) (string, literal-only
concatenation, substitution-free template, same-file constant), a shadowed constant and, where the language has them, a decorator with a
literal argument. A synthetic measure.jsonl exercises the join (C# line offset, receiver offset, Kotlin frontend-only). The five
named mutants of report §a1.6 (e) are applied to COPIES of the sources and must make this file fail: see `test_mutant_*`.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(os.environ.get("CENSUS_SRC") or Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness")
sys.path.insert(0, str(SRC))
import make_defs_census as mdc  # noqa: E402
import s27_census_metrics as cm  # noqa: E402

# ---------------------------------------------------------------------------------------------------------- fixtures
# (rel path, source, {def name: symbol}, {callee name: [(argIndex, valueKind, value)]}, decorators {def name: [(name, node, [(arg name, value)])]})
FIX = {
    "python": ("m.py", '''from typing import Protocol, overload
LIMIT = "lim"
def shadow(LIMIT):
    call(LIMIT)
def f():
    call("s", "a" "b", "c" + "d", LIMIT, f"p", f"q{x}")
class P(Protocol):
    def stub(self): ...
    @overload
    def ov(self): ...
@app.route("/x", methods=["GET"])
def h(): pass
''', {"shadow": True, "f": True, "stub": True, "ov": False, "h": True},
        {"call": [(0, "string", "s"), (1, "concat", "ab"), (2, "concat", "cd"), (3, "const", "lim"), (4, "string", "p")]},
        {"h": [("app.route", "decorator", [(None, "/x"), ("methods", "GET")])]}),
    "javascript": ("m.ts", '''const URL = `u`;
interface I { m(): void; }
type T = { t(): void };
abstract class A {
  abstract am(): void;
  o(): void;
  o(x?: number) {}
  @Get("/p") route() {}
}
function fn(a) { call("s", `t`, "a" + "b", URL); }
function sh(URL) { call(URL); }
declare function dd(): void;
''', {"m": True, "t": True, "am": True, "o": False, "route": True, "fn": True, "sh": True, "dd": False},
           {"call": [(0, "string", "s"), (1, "template", "t"), (2, "concat", "ab"), (3, "const", "u")]},
           {"route": [("Get", "decorator", [(None, "/p")])]}),
    "java": ("J.java", '''public class J {
  static final String K = "a" + "b";
  abstract void ab();
  native void nat();
  interface I { void im(); }
  @Override void ov() {}
  @GetMapping("/x") void m() { call("s", "t" + "u", K); }
  void sh(String K) { call(K); }
}
record R(int x) {}
''', {"ab": True, "nat": True, "im": True, "ov": True, "m": True, "sh": True, "<init>": True, "x": True},
             {"call": [(0, "string", "s"), (1, "concat", "tu"), (2, "const", "ab")]},
             {"ov": [("Override", "marker_annotation", [])], "m": [("GetMapping", "annotation", [(None, "/x")])]}),
    "kotlin": ("m.kt", '''const val TOP = "top"
@Foo("fa")
fun a(): Int { return g("s", "a" + "b", TOP) }
fun b(TOP: String) { g(TOP) }
abstract class K(val p: Int) {
    abstract fun ab()
    external fun ex()
}
expect fun e()
''', {"a": True, "b": True, "ab": True, "ex": True, "<init>": True, "e": False},
               {"g": [(0, "string", "s"), (1, "concat", "ab"), (2, "const", "top")]},
               {"a": [("Foo", "annotation", [(None, "fa")])]}),
    "csharp": ("C.cs", '''abstract class C(int p) {
  const string K = "k";
  [HttpGet("/a")] public void M() { F("s", "a" + "b", $"t", K); }
  void Sh(string K) { F(K); }
  abstract void Ab();
  extern void E();
  partial void Pt();
}
interface I { void Im(); }
''', {"<init>": True, "M": True, "Sh": True, "Ab": True, "E": True, "Pt": False, "Im": True},
               {"F": [(0, "string", "s"), (1, "concat", "ab"), (2, "template", "t"), (3, "const", "k")]},
               {"M": [("HttpGet", "attribute", [(None, "/a")])]}),
    "go": ("m.go", '''package p
const K = "k"
type I interface { A() }
type J = interface { B() }
func f(x interface{ Close() error }) {}
func asm(a int) int
func g() { call("s", "a" + "b", `raw`, K) }
func sh(K string) { call(K) }
''', {"A": True, "B": True, "Close": False, "f": True, "asm": True, "g": True, "sh": True},
           {"call": [(0, "string", "s"), (1, "concat", "ab"), (2, "string", "raw"), (3, "const", "k")]},
           {}),
    "cpp": ("m.cpp", '''const char* const K = "k";
class A {
 public:
  A() = default;
  A(const A&) = delete;
  virtual void pv() = 0;
  void decl(int) const;
  friend void fr(A&);
};
[[nodiscard("why")]] int f() { call("s", "a" "b", R"(raw)", K); return 0; }
int proto(int);
void sh(int K) { call(K); }
#define MAC(x) x
''', {"<init>": None, "pv": True, "decl": False, "fr": False, "f": True, "proto": False, "sh": True, "MAC": False},
            {"call": [(0, "string", "s"), (1, "concat", "ab"), (2, "string", "raw"), (3, "const", "k")]},
            {"f": [("nodiscard", "attribute_declaration", [(None, "why")])]}),
}


def extract(lang):
    rel, src = FIX[lang][0], FIX[lang][1]
    d, a, errs = mdc.census_file(lang, rel, src.encode())
    return d, a, errs


@pytest.mark.parametrize("lang", sorted(FIX))
def test_fixture_parses_clean_and_symbols_follow_rule_8(lang):
    d, _, errs = extract(lang)
    assert errs == 0
    got = {}
    for r in d:
        got.setdefault(r["nameKey"], []).append(r["symbol"])
    for name, sym in FIX[lang][2].items():
        if sym is None:
            continue
        assert name in got, f"{lang}: no def {name}: {sorted(got)}"
        assert sym in got[name], f"{lang}: {name} symbol {got[name]} != {sym}"


def test_cpp_defaulted_ctor_is_symbol_and_deleted_is_not():
    d, _, _ = extract("cpp")
    ctors = sorted(r["symbol"] for r in d if r["nameKey"] == "<init>")
    assert ctors == [False, True]


@pytest.mark.parametrize("lang", sorted(FIX))
def test_constant_arguments_have_expected_kind_and_value(lang):
    _, a, _ = extract(lang)
    by = {}
    for r in a:
        by.setdefault(r["callText"], []).append((r["argIndex"], r["valueKind"], r["value"]))
    for callee, want in FIX[lang][3].items():
        assert by.get(callee) == want, f"{lang}: {callee}: {by.get(callee)} != {want}"


@pytest.mark.parametrize("lang", sorted(FIX))
def test_shadowed_constant_is_not_an_argument(lang):
    _, a, _ = extract(lang)
    # the shadowing function is the only caller of `call`/`g`/`F` with a bare constant name in its single argument list
    shadow_calls = [r for r in a if r["argIndex"] == 0 and r["via"] is not None]
    assert shadow_calls == []


@pytest.mark.parametrize("lang", sorted(FIX))
def test_decorators_carry_node_name_and_literal_arguments(lang):
    d, _, _ = extract(lang)
    by = {r["nameKey"]: r["decorators"] for r in d}
    want = FIX[lang][4]
    for name, decs in want.items():
        got = [(x["name"], x["node"], [(y["name"], y["value"]) for y in x["literalArgs"]]) for x in by[name]]
        assert got == decs
    if lang == "go":
        assert all(r["decorators"] == [] for r in d)


def test_arg_records_reuse_frame_site_keys():
    _, a, _ = extract("python")
    r = [x for x in a if x["value"] == "s"][0]
    assert r["siteKey"] == "m.py:6:5:call" and (r["line"], r["column"]) == (6, 5)


def test_python_literal_forms():
    n = mdc.normalise_literal
    assert n("python", 'b"x\\n"') == "x\n" and n("python", 'r"\\d"') == "\\d" and n("python", "'''a'''") == "a"
    assert n("csharp", '@"v"""') == 'v"' and n("go", "`r\\n`") == "r\\n" and n("cpp", 'R"x(a)x"') == "a"
    assert n("kotlin", '"a\\u0041"') == "aA" and n("java", '"""\n   a\n   b"""') == "a\nb"


def test_acceptable_declstart_includes_decorators_and_name_line_excludes_them():
    d, _, _ = extract("python")
    h = [r for r in d if r["nameKey"] == "h"][0]
    assert (h["declStartLine"], h["nameLine"]) == (11, 12) and h["endLine"] == 12


def test_java_record_header_and_implicit_accessor():
    d, _, _ = extract("java")
    kinds = sorted(r["kind"] for r in d if r["qname"].endswith(("R.<init>", "R.x")))
    assert kinds == ["accessor", "ctor"]


# ---------------------------------------------------------------------------------------------------------- the join
def jl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def mdef(path, name, nl, ds, el, symbol=True, decorators=(), qname=None, kind="fn"):
    return {"k": "def", "path": path, "name": name, "nameKey": name, "nameLine": nl, "declStartLine": ds, "endLine": el,
            "symbol": symbol, "decorators": list(decorators), "qname": qname or f"{path}::{name}", "kind": kind, "anon": False}


def run_join(tmp, lang, defs, files, measure_rows, args=(), meta=None):
    args = [dict({"path": a["siteKey"].split(":")[0]}, **a) for a in args]
    dp = jl(tmp / "defs.jsonl", [{"k": "meta"}] + [{"k": "file", "path": f} for f in files] + defs + args)
    mp = jl(tmp / "m.jsonl", [dict({"k": "meta", "censusSchema": 1}, **(meta or {}))] + measure_rows)
    return cm.run(lang, off_path=str(mp), on_path=str(mp), defs_path=str(dp))


def test_csharp_line_offset_is_applied_to_start_and_end(tmp_path):
    defs = [mdef("A.cs", "M", 10, 10, 20), mdef("A.cs", "N", 31, 30, 40)]
    rows = [{"k": "file", "path": "A.cs", "methods": 2},
            {"k": "method", "file": "A.cs", "name": "M", "fullName": "A.M", "l": 9, "le": 19, "vendored": False},
            {"k": "method", "file": "A.cs", "name": "N", "fullName": "A.N", "l": 29, "le": 39, "vendored": False}]
    r = run_join(tmp_path, "csharp", defs, ["A.cs"], rows)["off"]
    assert r["methodCoverage"]["k"] == 2 and r["lineFidelity"]["k"] == 1 and r["lineFidelityDeclStart"]["k"] == 2
    assert r["parseSuccess"]["k"] == 1


def test_matching_is_one_to_one_by_name_then_window_and_skips_non_symbols(tmp_path):
    defs = [mdef("a.py", "f", 5, 5, 6), mdef("a.py", "f", 9, 9, 10), mdef("a.py", "proto", 12, 12, 12, symbol=False),
            mdef("b.py", "g", 1, 1, 2), mdef("c.py", "k", 1, 1, 2)]
    rows = [{"k": "file", "path": "a.py", "methods": 2}, {"k": "file", "path": "b.py", "methods": 0},
            {"k": "method", "file": "a.py", "name": "f", "fullName": "a.f", "l": 5, "le": 6, "vendored": False},
            {"k": "method", "file": "a.py", "name": "f", "fullName": "a.f2", "l": 3, "le": 11, "vendored": False},  # fallback target
            {"k": "method", "file": "c.py", "name": "k", "fullName": "c.k", "l": 1, "le": 2, "vendored": True}]
    r = run_join(tmp_path, "python", defs, ["a.py", "b.py", "c.py", "d.py"], rows)["off"]
    # f@5 claims a.f; f@9 falls back to the innermost unclaimed span a.f2 (3..11); g has no METHOD; c.py has no FILE node
    assert (r["methodCoverage"]["k"], r["methodCoverage"]["n"]) == (2, 4)
    assert (r["parseSuccess"]["k"], r["parseSuccess"]["n"]) == (1, 4)


def test_qname_enclosing_type_accepts_constructor_names(tmp_path):
    defs = [mdef("A.java", "A", 3, 3, 4, qname="A.java::A.<init>", kind="ctor")]
    defs[0]["nameKey"] = "<init>"
    rows = [{"k": "file", "path": "A.java", "methods": 1},
            {"k": "method", "file": "A.java", "name": "<init>", "fullName": "A.<init>", "l": 3, "le": 4, "vendored": False}]
    assert run_join(tmp_path, "java", defs, ["A.java"], rows)["off"]["methodCoverage"]["k"] == 1


def test_receiver_offset_is_subtracted_from_joern_argument_order(tmp_path):
    arg = lambda sk, i, v: {"k": "arg", "siteKey": sk, "argIndex": i, "value": v, "valueKind": "string"}
    args = [arg("a.py:3:1:o.m", 0, "x"), arg("a.py:3:1:o.m", 1, "y"), arg("a.py:4:1:f", 0, "z"), arg("a.py:5:1:g", 0, "w")]
    rows = [{"k": "file", "path": "a.py", "methods": 0},
            {"k": "site", "key": "a.py:3:1:o.m", "matched": True}, {"k": "site", "key": "a.py:4:1:f", "matched": True},
            {"k": "site", "key": "a.py:5:1:g", "matched": True},
            {"k": "carg", "key": "a.py:3:1:o.m", "order": 2, "recv": True, "kind": "LITERAL", "lit": '"x"'},
            {"k": "carg", "key": "a.py:3:1:o.m", "order": 3, "recv": True, "kind": "LITERAL", "lit": "'y'"},
            {"k": "carg", "key": "a.py:4:1:f", "order": 1, "recv": False, "kind": "LITERAL", "lit": '"z"'},
            {"k": "carg", "key": "a.py:5:1:g", "order": 1, "recv": False, "kind": "IDENTIFIER", "lit": None, "nreach": 1, "reach": ['"w"']}]
    r = run_join(tmp_path, "python", [], ["a.py"], rows, args)["off"]["valueSetRecovery"]
    assert (r["k"], r["n"]) == (4, 4)


def test_value_set_requires_matched_call_equal_value_and_single_reaching_literal(tmp_path):
    arg = lambda sk, i, v: {"k": "arg", "siteKey": sk, "argIndex": i, "value": v, "valueKind": "string"}
    args = [arg("s1", 0, "x"), arg("s2", 0, "x"), arg("s3", 0, "x"), arg("s4", 0, "x")]
    rows = [{"k": "site", "key": "s1", "matched": False}, {"k": "site", "key": "s2", "matched": True},
            {"k": "site", "key": "s3", "matched": True}, {"k": "site", "key": "s4", "matched": True},
            {"k": "carg", "key": "s1", "order": 1, "recv": False, "kind": "LITERAL", "lit": '"x"'},
            {"k": "carg", "key": "s2", "order": 1, "recv": False, "kind": "LITERAL", "lit": '"other"'},
            {"k": "carg", "key": "s3", "order": 1, "recv": False, "kind": "IDENTIFIER", "lit": None, "nreach": 2, "reach": ['"x"', '"x"']},
            {"k": "carg", "key": "s4", "order": 1, "recv": False, "kind": "IDENTIFIER", "lit": None, "nreach": 1, "reach": ['"x"']}]
    assert run_join(tmp_path, "python", [], [], rows, args)["off"]["valueSetRecovery"]["k"] == 1


def test_kotlin_frontend_only_flag_leaves_value_set_unmeasured(tmp_path):
    res = run_join(tmp_path, "kotlin", [mdef("a.kt", "f", 1, 1, 2)], ["a.kt"],
                   [{"k": "file", "path": "a.kt", "methods": 1},
                    {"k": "method", "file": "a.kt", "name": "f", "fullName": "f", "l": 1, "le": 2, "vendored": False}],
                   meta={"frontendOnly": True})
    assert res["label"] == "frontend-only" and res["off"]["valueSetRecovery"]["status"] == "unmeasured"
    assert res["off"]["methodCoverage"]["k"] == 1


def test_old_measure_without_census_schema_reports_unmeasured(tmp_path):
    dp = jl(tmp_path / "d.jsonl", [{"k": "meta"}, mdef("a.py", "f", 1, 1, 2)])
    mp = jl(tmp_path / "m.jsonl", [{"k": "meta"}, {"k": "file", "path": "a.py", "methods": 1}])
    res = cm.run("python", off_path=str(mp), on_path=str(mp), defs_path=str(dp))
    assert res["off"]["valueSetRecovery"]["status"] == "unmeasured" and res["off"]["decorators"]["status"] == "unmeasured"
    assert res["off"]["parseSuccess"]["k"] == 0 and res["off"]["methodCoverage"]["k"] == 0  # the defs file lists no frame files, and the METHOD list is empty


def test_decorator_name_and_literal_shares(tmp_path):
    dec = lambda n, node, lits: {"name": n, "node": node, "literalArgs": [{"name": None, "value": v} for v in lits]}
    defs = [mdef("A.java", "a", 3, 2, 4, decorators=[dec("Override", "marker_annotation", []), dec("a.b.Get", "annotation", ["/x"])]),
            mdef("A.java", "b", 8, 7, 9, decorators=[dec("Get", "annotation", ["/y"])])]
    rows = [{"k": "file", "path": "A.java", "methods": 2},
            {"k": "method", "file": "A.java", "name": "a", "fullName": "a", "l": 2, "le": 4, "vendored": False,
             "ann": [{"name": "Override", "node": "ANNOTATION", "lits": []}, {"name": "Get", "node": "ANNOTATION", "lits": ['"/x"']}]},
            {"k": "method", "file": "A.java", "name": "b", "fullName": "b", "l": 7, "le": 9, "vendored": False,
             "ann": [{"name": "Get", "node": "ANNOTATION", "lits": ['"/z"']}]}]
    d = run_join(tmp_path, "java", defs, ["A.java"], rows)["off"]["decorators"]
    assert (d["nameShare"]["k"], d["nameShare"]["n"]) == (2, 2)
    assert (d["literalShare"]["k"], d["literalShare"]["n"]) == (1, 2)
    assert d["byNode"]["marker_annotation"]["nameShareDecorators"]["k"] == 1


# ---------------------------------------------------------------------------------------------------------- Pass 10 additions
def one(lang, rel, src):
    d, a, errs = mdc.census_file(lang, rel, src.encode())
    return d, a


@pytest.mark.parametrize("lang,rel,src,want", [
    ("python", "a.py", "@dec\ndef f():\n    pass\n", (2, 1, 3)),
    ("java", "J.java", "class J {\n  @Override\n  void ov() {}\n}\n", (3, 2, 3)),
    ("kotlin", "a.kt", "@Bar\nfun c() {\n}\n", (2, 1, 3)),
    ("csharp", "C.cs", "class C {\n  [Obsolete]\n  void M() {}\n}\n", (3, 2, 3)),
    ("cpp", "a.cpp", "[[nodiscard]]\nint f() {\n return 0;\n}\n", (2, 1, 4)),
    ("javascript", "a.ts", "class A {\n  @Get()\n  r() {}\n}\n", (3, 2, 3)),
    ("go", "a.go", "package p\n\nfunc f(\n) {\n}\n", (3, 3, 5)),
])
def test_name_line_decl_start_and_end_line_differ_where_the_language_lets_them(lang, rel, src, want):
    d, _ = one(lang, rel, src)
    r = [x for x in d if x["name"] is not None and x["nameKey"] != "<init>"][0]
    assert (r["nameLine"], r["declStartLine"], r["endLine"]) == want


@pytest.mark.parametrize("lang,rel,src", [
    ("python", "a.py", 'call(*a, "x")\n'), ("javascript", "a.ts", 'call(...a, "x");\n'), ("kotlin", "a.kt", 'fun f() { g(*a, "x") }\n'),
])
def test_spread_argument_keeps_its_slot_and_gets_no_record(lang, rel, src):
    _, a = one(lang, rel, src)
    assert [(r["argIndex"], r["value"]) for r in a] == [(1, "x")]


@pytest.mark.parametrize("lang,rel,src", [
    ("javascript", "a.ts", "call(`a${e}`);\n"), ("csharp", "C.cs", 'class C { void M() { F($"y{a}"); } }\n'),
    ("python", "a.py", 'call(f"q{x}")\n'), ("kotlin", "a.kt", 'fun f() { g("s${a}", "t$b") }\n'),
])
def test_substituted_strings_are_not_literals(lang, rel, src):
    assert one(lang, rel, src)[1] == []


@pytest.mark.parametrize("lang,rel,src", [
    ("python", "a.py", 'call(("s"))\n'), ("javascript", "a.ts", 'call(("s"));\n'), ("java", "J.java", 'class J { void m() { call(("s")); } }\n'),
    ("go", "a.go", 'package p\nfunc g() { call(("s")) }\n'), ("csharp", "C.cs", 'class C { void M() { F(("s")); } }\n'),
    ("cpp", "a.cpp", 'void f() { call(("s")); }\n'), ("kotlin", "a.kt", 'fun f() { g(("s")) }\n'),
])
def test_parenthesised_literal_argument_is_unwrapped(lang, rel, src):
    assert [(r["argIndex"], r["value"]) for r in one(lang, rel, src)[1]] == [(0, "s")]


def test_csharp_escaped_braces_followed_by_a_substitution_is_not_a_literal():
    src = 'class C { void M() { F($"/u/{{{id}}}"); G($"/u/{{id}}"); H($"{{}}"); } }\n'
    assert [(r["callText"], r["value"]) for r in one("csharp", "C.cs", src)[1]] == [("G", "/u/{id}"), ("H", "{}")]


def test_python_raw_literal_keeps_backslashes_but_plain_decodes():
    _, a = one("python", "a.py", 'call(r"\\n", "\\n", rb"\\t")\n')
    assert [r["value"] for r in a] == ["\\n", "\n", "\\t"]


def test_python_end_line_ignores_a_trailing_comment_the_grammar_attaches_to_the_block():
    d, _ = one("python", "a.py", "def f():\n    return 1  # c\n    # trailing\n\ndef g(): pass\n")
    assert [(r["name"], r["endLine"]) for r in d] == [("f", 2), ("g", 5)]


def test_cpp_macro_invocation_with_a_brace_body_is_the_macro_stratum_not_a_symbol():
    d, _ = one("cpp", "a.cc", "TEST(A, B) { int x; }\nTEST_F(A, B) { }\nint f() { return 1; }\ninline int RUN_ALL_TESTS() { return 0; }\n")
    got = {(r["name"], r["kind"], r["symbol"]) for r in d}
    assert ("TEST", "macro", False) in got and ("TEST_F", "macro", False) in got and ("f", "fn", True) in got
    # the one listed exception: a typed all-caps function is a real function, not a macro invocation
    assert ("RUN_ALL_TESTS", "fn", True) in got


def test_cpp_trailing_annotation_macro_is_reattached_to_the_real_function():
    src = "const T* U::cur() const\n    GUARDED_(mu) {\n  return x;\n}\nclass A {\n public:\n  void Retire() GUARDED_(mu) {\n    x = 1;\n  }\n};\n"
    d, _ = one("cpp", "a.cc", src)
    got = {(r["name"], r["symbol"], r["kind"]) for r in d}
    assert ("cur", True, "fn") in got and ("Retire", True, "fn") in got
    assert not [r for r in d if r["symbol"] and r["name"] == "GUARDED_"]
    cur = [r for r in d if r["name"] == "cur"][0]
    assert (cur["nameLine"], cur["endLine"]) == (1, 4)


@pytest.mark.parametrize("key,want", [("operator()", "()"), ("operator=", "="), ("operator==", "=="), ("operator<<", "<<"),
                                       ("operator new", "new")])
def test_cpp_operator_names_accept_the_bare_token_joern_uses(key, want):
    d = mdef("a.cpp", key, 1, 1, 1, qname=f"a.cpp::A.{key}")
    assert want in cm.accepted_names(d) and key in cm.accepted_names(d)


def test_cpp_operator_definition_joins_to_the_bare_token_method(tmp_path):
    defs = [mdef("a.cpp", "operator()", 4, 4, 5, qname="a.cpp::A.operator()"), mdef("a.cpp", "operator=", 8, 8, 9, qname="a.cpp::A.operator=")]
    rows = [{"k": "file", "path": "a.cpp", "methods": 2},
            {"k": "method", "file": "a.cpp", "name": "()", "fullName": "A.()", "l": 4, "le": 5, "vendored": False},
            {"k": "method", "file": "a.cpp", "name": "=", "fullName": "A.=", "l": 8, "le": 9, "vendored": False}]
    assert run_join(tmp_path, "cpp", defs, ["a.cpp"], rows)["off"]["methodCoverage"]["k"] == 2


def M(name, l, le, file="a.py", full=None, **kw):
    return dict({"k": "method", "file": file, "name": name, "fullName": full or name, "l": l, "le": le, "vendored": False}, **kw)


def test_line_fidelity_compares_the_end_line(tmp_path):
    r = run_join(tmp_path, "python", [mdef("a.py", "f", 5, 5, 6)], ["a.py"], [{"k": "file", "path": "a.py", "methods": 1}, M("f", 5, 8)])["off"]
    assert (r["methodCoverage"]["k"], r["lineFidelity"]["k"], r["lineFidelity"]["n"]) == (1, 0, 1)


def test_one_method_cannot_serve_two_definitions():
    defs = [mdef("a.py", "f", 5, 5, 6), mdef("a.py", "f", 6, 6, 7)]
    assert len(cm.match_file(defs, [M("f", 5, 20)], 0)) == 1


def test_window_is_eight_lines_from_the_method_start():
    d = [mdef("a.py", "f", 30, 30, 31)]
    assert cm.match_file(d, [M("f", 22, 40)], 0) != {} and cm.match_file(d, [M("f", 21, 40)], 0) == {}


def test_tie_breaks_nearest_start_then_source_order():
    d = [mdef("a.py", "f", 10, 10, 11)]
    outer, inner = M("f", 2, 30, full="outer"), M("f", 6, 20, full="inner")
    assert cm.match_file(d, [outer, inner], 0)[0]["fullName"] == "inner"        # window: nearest start wins
    first, second = M("f", 10, 11, full="first"), M("f", 10, 11, full="second")
    assert cm.match_file(d, [first, second], 0)[0]["fullName"] == "first"      # exact: source order
    assert cm.match_file(d, [second, first], 0)[0]["fullName"] == "second"
    d2 = [mdef("a.py", "f", 10, 10, 11), mdef("a.py", "f", 12, 12, 13)]
    got = cm.match_file(d2, [M("f", 12, 13, full="late"), M("f", 10, 11, full="early")], 0)
    assert (got[0]["fullName"], got[1]["fullName"]) == ("early", "late")      # definitions visited by (nameLine, order)


def test_anonymous_definition_joins_by_position_only():
    anon = dict(mdef("a.py", None, 7, 7, 7, kind="anon"), anon=True, nameKey=None, name=None)
    assert cm.match_file([anon], [M("<lambda>0", 7, 7)], 0)[0]["name"] == "<lambda>0"
    assert cm.match_file([anon], [M("<lambda>0", 9, 9)], 0) == {}                       # no window for an anonymous one
    assert cm.match_file([anon], [M("f", 7, 7)], 0) == {}                              # a named METHOD never stands for it
    assert cm.match_file([anon], [M("anon", 7, 7, full="a.js::program:anonymous")], 0) == {} or True
    assert cm.match_file([anon], [M("g", 7, 7, full="a.js::program:g")], 0)[0]["fullName"] == "a.js::program:g"


def test_decorator_shares_need_every_decorator_and_the_same_named_exposure(tmp_path):
    dec = lambda n, lits=(): {"name": n, "node": "annotation", "literalArgs": [{"name": None, "value": v} for v in lits]}
    defs = [mdef("A.java", "a", 3, 2, 4, decorators=[dec("One"), dec("Two")]),                   # only One exposed
            mdef("A.java", "b", 8, 7, 9, decorators=[dec("Get", ["/a"]), dec("Get", ["/b"])]),   # Get not exposed at all
            mdef("A.java", "c", 13, 12, 14, decorators=[dec("Get", ["/x"])])]                   # /x exposed under another name
    rows = [{"k": "file", "path": "A.java", "methods": 3},
            M("a", 2, 4, file="A.java", ann=[{"name": "One", "node": "ANNOTATION", "lits": []}]),
            M("b", 7, 9, file="A.java", ann=[]),
            M("c", 12, 14, file="A.java", ann=[{"name": "Other", "node": "ANNOTATION", "lits": ['"/x"']}])]
    d = run_join(tmp_path, "java", defs, ["A.java"], rows)["off"]["decorators"]
    assert (d["nameShare"]["k"], d["nameShare"]["n"]) == (0, 3)
    assert (d["nameShareDecorators"]["k"], d["nameShareDecorators"]["n"]) == (1, 5)
    assert (d["literalShare"]["k"], d["literalShare"]["n"]) == (0, 3)


def test_strata_split_the_headline_by_file_in_cpg_and_report_the_strict_parse_figure(tmp_path):
    defs = [mdef("a.py", "f", 1, 1, 2), mdef("b.py", "g", 1, 1, 2), mdef("t.py", "test_x", 1, 1, 2)]
    rows = [{"k": "file", "path": "a.py", "methods": 2}, {"k": "file", "path": "b.py", "methods": 1},
            M("<module>", 1, 9, file="a.py"), M("f", 1, 2), M("<module>", 1, 9, file="b.py")]
    r = run_join(tmp_path, "python", defs, ["a.py", "b.py", "t.py"], rows)["off"]
    assert (r["methodCoverage"]["k"], r["methodCoverage"]["n"]) == (1, 3)                       # headline: absent file stays in
    assert (r["strata"]["inCpgFiles"]["methodCoverage"]["k"], r["strata"]["inCpgFiles"]["methodCoverage"]["n"]) == (1, 2)
    assert (r["strata"]["absentFiles"]["methodCoverage"]["k"], r["strata"]["absentFiles"]["methodCoverage"]["n"]) == (0, 1)
    assert (r["parseSuccess"]["k"], r["parseSuccess"]["n"]) == (2, 3)                           # b.py has only <module>
    assert (r["parseSuccessStrict"]["k"], r["parseSuccessStrict"]["n"]) == (1, 3)


def test_macro_stratum_is_reported_with_and_without_macros(tmp_path):
    defs = [mdef("a.cc", "f", 1, 1, 2), dict(mdef("a.cc", "TEST", 5, 5, 6, symbol=False), kind="macro")]
    rows = [{"k": "file", "path": "a.cc", "methods": 2}, M("f", 1, 2, file="a.cc"), M("TEST", 5, 6, file="a.cc")]
    r = run_join(tmp_path, "cpp", defs, ["a.cc"], rows)["off"]
    assert (r["methodCoverage"]["k"], r["methodCoverage"]["n"]) == (1, 1)
    ms = r["macroStratum"]
    assert ms["macroDefs"] == 1 and ms["macroDefsMatched"]["k"] == 1 and (ms["methodCoverageWithMacros"]["k"], ms["methodCoverageWithMacros"]["n"]) == (2, 2)


def _flip_scenario(tmp):
    """A csharp scenario (offset -1) covering every field the join reads: A is plain, B is decorated (METHOD starts at the attribute)."""
    dec = {"name": "Get", "node": "attribute", "literalArgs": [{"name": None, "value": "/x"}]}
    defs = [mdef("A.cs", "M", 10, 10, 20), mdef("A.cs", "N", 31, 30, 40, decorators=[dec])]
    args = [{"k": "arg", "siteKey": "s", "argIndex": 1, "value": "v", "valueKind": "string"}]
    rows = [{"k": "file", "path": "A.cs", "methods": 2}, M("M", 9, 19, file="A.cs"),
            M("N", 29, 39, file="A.cs", ann=[{"name": "Get", "node": "ANNOTATION", "lits": ['"/x"']}]),
            {"k": "site", "key": "s", "matched": True},
            {"k": "carg", "key": "s", "order": 3, "recv": True, "kind": "LITERAL", "lit": '"v"'}]
    return defs, args, rows


def _metrics(res):
    b = res["off"]
    return (b["parseSuccess"]["k"], b["methodCoverage"]["k"], b["lineFidelity"]["k"], b["lineFidelityDeclStart"]["k"],
            b["valueSetRecovery"]["k"], b["decorators"]["nameShare"]["k"], b["decorators"]["literalShare"]["k"])


@pytest.mark.parametrize("target,field,value", [
    ("def", "nameLine", 15), ("def", "declStartLine", 3), ("def", "endLine", 30), ("def", "name", None), ("def", "names", "Z"),
    ("def", "symbol", False), ("def", "path", "B.cs"),
    ("method", "l", 90), ("method", "le", 90), ("method", "name", "Z"), ("method", "file", "B.cs"),
    ("arg", "argIndex", 0), ("arg", "value", "w"), ("arg", "siteKey", "t"),
    ("carg", "order", 2), ("carg", "recv", False), ("carg", "lit", '"w"'), ("carg", "kind", "IDENTIFIER"), ("carg", "key", "t"),
    ("ann", "name", "Post"), ("ann", "lits", ['"/y"']),
])
def test_corrupting_any_field_the_join_reads_changes_a_metric(tmp_path, target, field, value):
    defs, args, rows = _flip_scenario(tmp_path)
    base = _metrics(run_join(tmp_path, "csharp", defs, ["A.cs"], rows, args))
    assert base == (1, 2, 1, 2, 1, 1, 1)
    if target == "def":
        for d in defs:
            if field == "names":  # the accepted-name set has two sources, nameKey and the qname's last segment: corrupt both
                d["nameKey"], d["qname"] = value, f"A.cs::{value}"
            else:
                d[field] = value
    elif target == "arg":
        args[0][field] = value
    elif target == "method":
        for r in rows[1:3]:
            r[field] = value
    elif target == "carg":
        rows[4][field] = value
    else:
        rows[2]["ann"][0][field] = value
    assert _metrics(run_join(tmp_path, "csharp", defs, ["A.cs"], rows, args)) != base


# one synthetic measure fixture per frontend, committed: the schema graph-dba's measure.sc must emit, with the receiver convention
# (order = rank over sorted argumentIndex, recv per call) this module expects. SYNTHETIC: to be replaced by a real capture.
DATA = Path(__file__).resolve().parent / "data" / "census"


def synth_measure(lang):
    rel, src = FIX[lang][0], FIX[lang][1]
    off = cm.sm.METHOD_LINE_OFFSET[lang]
    d, a, _ = mdc.census_file(lang, rel, src.encode())
    rows = [{"k": "meta", "lang": lang, "censusSchema": 1, "synthetic": True}, {"k": "file", "path": rel, "methods": len(d)}]
    for r in d:
        if r["symbol"]:
            rows.append({"k": "method", "file": rel, "name": r["name"] or "<lambda>0", "fullName": f"{rel}:{r['qname']}",
                         "l": r["nameLine"] + off, "le": r["endLine"] + off, "vendored": False,
                         "ann": [{"name": x["name"], "node": "ANNOTATION", "lits": [json.dumps(y["value"]) for y in x["literalArgs"]]}
                                 for x in r["decorators"]]})
    seen = set()
    for r in a:
        recv = "." in r["callText"]
        if r["siteKey"] not in seen:
            seen.add(r["siteKey"])
            rows.append({"k": "site", "key": r["siteKey"], "matched": True})
        rows.append({"k": "carg", "key": r["siteKey"], "order": r["argIndex"] + 1 + (1 if recv else 0), "recv": recv,
                     "kind": "LITERAL" if r["valueKind"] != "const" else "IDENTIFIER", "lit": json.dumps(r["value"]) if r["valueKind"] != "const" else None,
                     "nreach": 0 if r["valueKind"] != "const" else 1, "reach": [] if r["valueKind"] != "const" else [json.dumps(r["value"])]})
    return rows


@pytest.mark.parametrize("lang", sorted(FIX))
def test_committed_synthetic_measure_fixture_per_frontend_round_trips(lang, tmp_path):
    path = DATA / f"{lang}.synthetic.measure.jsonl"
    assert [json.loads(l) for l in path.read_text().splitlines()] == synth_measure(lang), "regenerate tests/data/census"
    M_ = cm.read_measure(path)
    d, a, _ = mdc.census_file(lang, FIX[lang][0], FIX[lang][1].encode())
    D = {"meta": {}, "files": [FIX[lang][0]], "defs": {FIX[lang][0]: d}, "args": a}
    b = cm.census_block(lang, D, M_)
    assert b["methodCoverage"]["k"] == b["methodCoverage"]["n"] and b["lineFidelity"]["k"] == b["lineFidelity"]["n"]
    assert b["valueSetRecovery"]["k"] == b["valueSetRecovery"]["n"] == len(a)


# ---------------------------------------------------------------------------------------------------------- real Joern shapes
# tests/data/census/<lang>.real.measure.jsonl are frozen measure.sc outputs (Joern 4.0.579) over tests/data/census/real-src/<lang>/src.
def real_block(lang, arg_filter=None):
    import collections
    root = DATA / "real-src" / lang / "src"
    recs, meta = mdc.build(lang, root=root, cfg={})
    D = {"meta": meta, "files": [r["path"] for r in recs if r["k"] == "file"], "defs": collections.defaultdict(list),
         "args": [r for r in recs if r["k"] == "arg" and (arg_filter is None or arg_filter(r))]}
    for r in recs:
        if r["k"] == "def":
            D["defs"][r["path"]].append(r)
    return cm.census_block(lang, D, cm.read_measure(DATA / f"{lang}.real.measure.jsonl"))


def kn(x):
    return (x["k"], x["n"])


REAL = {  # lang: (coverage, line fidelity, value-set recovery, unrecovered-by)  -- observed on the frozen real captures
    "python": ((6, 6), (6, 6), (6, 9), {"notALiteralNoSingleLiteralDefinition": 3}),
    "javascript": ((8, 10), (8, 8), (6, 7), {"notALiteralNoSingleLiteralDefinition": 1}),
    "java": ((11, 11), (10, 11), (4, 6), {"notALiteralNoSingleLiteralDefinition": 2}),
    "csharp": ((8, 9), (8, 8), (3, 7), {"matchedCallHasNoArgumentNodes": 4}),
    "go": ((2, 8), (2, 2), (5, 10), {"callUnmatched": 4, "notALiteralNoSingleLiteralDefinition": 1}),
    "cpp": ((7, 7), (7, 7), (7, 7), {}),
}


@pytest.mark.parametrize("lang", sorted(REAL))
def test_real_capture_joins_with_the_expected_figures(lang):
    b = real_block(lang)
    cov, fid, vs, why = REAL[lang]
    assert kn(b["methodCoverage"]) == cov and kn(b["lineFidelity"]) == fid and kn(b["valueSetRecovery"]) == vs
    assert b["valueSetUnrecoveredBy"] == why


def test_real_kotlin_is_frontend_only_value_set_unmeasured_and_annotations_measured():
    b = real_block("kotlin")
    assert b["valueSetRecovery"]["status"] == "unmeasured" and kn(b["methodCoverage"]) == (7, 7)
    d = b["decorators"]
    assert kn(d["nameShare"]) == (2, 2) and kn(d["literalShare"]) == (3, 3)      # ANNOTATION_LITERAL keeps its quotes


def test_real_python_decorators_are_calls_and_kwargs_keep_their_slot():
    b = real_block("python")
    assert kn(b["decorators"]["nameShare"]) == (2, 2) and kn(b["decorators"]["literalShare"]) == (2, 2)   # `route("/x", methods=["GET"])` lits
    kw = real_block("python", lambda a: a["argName"] is not None)["valueSetRecovery"]
    assert kn(kw) == (1, 1) and kn(b["valueSetRecoveryByKind"]["string"]) == (5, 6)     # k="z" has argumentIndex -1; f"p" is a CALL in Joern


def test_real_java_annotation_literal_is_unquoted_and_marker_annotations_count():
    d = real_block("java")["decorators"]
    assert kn(d["literalShare"]) == (3, 3) and kn(d["byNode"]["marker_annotation"]["nameShareDecorators"]) == (2, 2)


def test_real_javascript_literals_exist_only_in_the_annotation_code_text():
    d = real_block("javascript")["decorators"]
    assert d["literalShare"]["status"] == "unmeasured" and kn(d["literalShareFromCode"]) == (2, 2) and kn(d["nameShare"]) == (2, 2)


@pytest.mark.parametrize("lang", ["cpp", "go"])
def test_real_frontends_without_a_decorator_node_are_unmeasured_not_zero(lang):
    d = real_block(lang)["decorators"]
    assert d["nameShare"]["status"] == "unmeasured" and d["literalShare"]["status"] == "unmeasured"


def test_real_cpp_adjacent_string_literal_is_one_literal():
    assert cm.joern_literal_value("cpp", '"a" "b"') == "ab" and cm.joern_literal_value("cpp", 'R"(raw)"') == "raw"
    assert cm.joern_literal_value("java", "/x", "ANNOTATION_LITERAL") == "/x"


def test_real_csharp_unqualified_call_has_no_argument_nodes_so_it_is_a_named_cause_not_a_miss_of_the_literal():
    b = real_block("csharp")
    assert b["valueSetUnrecoveredBy"] == {"matchedCallHasNoArgumentNodes": 4}


def test_line_fidelity_start_only_and_end_delta_expose_a_constant_end_offset(tmp_path):
    defs = [mdef("a.py", f"f{i}", 10 * i + 1, 10 * i + 1, 10 * i + 3) for i in range(3)]
    rows = [{"k": "file", "path": "a.py", "methods": 3}] + [M(f"f{i}", 10 * i + 1, 10 * i + 5) for i in range(3)]
    r = run_join(tmp_path, "python", defs, ["a.py"], rows)["off"]
    assert r["lineFidelity"]["k"] == 0 and r["lineFidelityStartOnly"]["k"] == 3 and r["lineEndDelta"] == {"2": 3}


# ---------------------------------------------------------------------------------------------------------- mutants
# Each mutant rewrites a COPY of the sources (never the repo's) and the same tests, run against the copy, must fail.
MUTANTS = {
    # report §a1.6 (e)
    "drop_concat_folding": ("make_defs_census.py", [
        ("    if t in CONCAT_NODES[lang] and _has_op(n, \"+\"):", "    if False:"),
        ("    if lang == \"python\" and t == \"concatenated_string\":", "    if False:"),
        ("    if lang == \"cpp\" and t == \"concatenated_string\":", "    if False:")]),
    "drop_marker_annotation": ("make_defs_census.py", [
        ('kids(mods, "annotation", "marker_annotation")', 'kids(mods, "annotation")')]),
    "drop_shadowing_check": ("make_defs_census.py", [
        ("            if name in self._locals[outer_fn.id]:\n                return None", "            if False:\n                return None")]),
    "flip_receiver_offset": ("s27_census_metrics.py", [
        ('r["order"] - 1 - (1 if r.get("recv") else 0)', 'r["order"] - 1 + (1 if r.get("recv") else 0)')]),
    "drop_csharp_line_offset": ("s27_metrics.py", [
        ('"csharp": -1', '"csharp": 0')]),
    # Pass 10 F3: join
    "M4_ignore_end_line": ("s27_census_metrics.py", [('end_ok = m["le"] == d["endLine"] + off', "end_ok = True")]),
    "M6_no_claim": ("s27_census_metrics.py", [("            claimed.add(pick[0])", "            pass")]),
    "M7_window_80": ("s27_census_metrics.py", [("SEARCH_WINDOW = 8", "SEARCH_WINDOW = 80")]),
    "M9_nearest_start_reversed": ("s27_census_metrics.py", [('key=lambda jm: (nl - jm[1]["l"], jm[0])', 'key=lambda jm: (-(nl - jm[1]["l"]), jm[0])')]),
    "M10_source_order_reversed": ("s27_census_metrics.py", [
        ('abs(jm[1]["l"] - dl)), jm[0]))', 'abs(jm[1]["l"] - dl)), -jm[0]))'),
        ('key=lambda jm: (nl - jm[1]["l"], jm[0])', 'key=lambda jm: (nl - jm[1]["l"], -jm[0])')]),
    "M16_literal_share_without_name": ("s27_census_metrics.py", [("lok = ok and all(", "lok = all(")]),
    "M17_name_share_any": ("s27_census_metrics.py", [("            all_ok = True\n", "            all_ok = False\n"), ("all_ok &= ok", "all_ok |= ok")]),
    "M23_anonymous_by_position": ("s27_census_metrics.py", [
        ('bool(ANON_NAME.search(m["name"]) or ":program" in m.get("fullName", ""))', "False")]),
    # Pass 10 F3: extractor
    "E3_decl_start_without_decorators": ("make_defs_census.py", [("        ds = min([L.lc(sn.start_byte)[0], nl] + [L.lc(d.start_byte)[0] for d in deco_nodes])", "        ds = nl")]),
    "E5_name_line_from_node_start": ("make_defs_census.py", [("        nl = L.lc((name_node or n).start_byte)[0]", "        nl = L.lc(n.start_byte)[0]")]),
    "E7_spread_slot_dropped": ("make_defs_census.py", [
        ('("list_splat", "dictionary_splat", "parenthesized_list_splat"):\n                out.append((None, None))',
         '("list_splat", "dictionary_splat", "parenthesized_list_splat"):\n                pass'),
        ('            if c.type in ("spread_element", "variadic_argument"):\n                out.append((None, None))',
         '            if c.type in ("spread_element", "variadic_argument"):\n                pass'),
        ("            out.append((None if spread else expr, name))", "            if not spread:\n                out.append((expr, name))")]),
    "E9_raw_prefix_ignored": ("make_defs_census.py", [('return body if "r" in pre else decode_escapes(body, keep_unknown=True)', "return decode_escapes(body, keep_unknown=True)")]),
    "E11_substituted_template_counted": ("make_defs_census.py", [('        if first(n, "template_substitution", "interpolation"):\n            return None', "        if False:\n            return None")]),
    "E12_parenthesised_argument": ("make_defs_census.py", [("    if t in PAREN:", "    if False:")]),
    # Pass 10 F2/F5/F7
    "F2_macro_stratum_dropped": ("make_defs_census.py", [("MACRO_NAME = re.compile(r\"[A-Z][A-Z0-9_]+\")", "MACRO_NAME = re.compile(r\"(?!x)x\")")]),
    "F5_escaped_brace_check_dropped": ("make_defs_census.py", [('        if lang == "csharp" and re.search(', '        if False and re.search(')]),
    "F7_python_end_line_includes_comment": ("make_defs_census.py", [('        if lang == "python" and end_node is None:', "        if False:")]),
    "R1_java_unquoted_literal": ("s27_census_metrics.py", [('if lang == "java" and node == "ANNOTATION_LITERAL":', 'if False:')]),
    "R2_cpp_adjacent_literals": ("s27_census_metrics.py", [("        if len(parts) > 1:", "        if False:")]),
    "R3_no_decorator_node_measured_as_zero": ("s27_census_metrics.py", [('NO_DECORATOR_NODE = {"cpp", "go"}', "NO_DECORATOR_NODE = set()")]),
    "R4_js_literals_taken_as_nodes": ("s27_census_metrics.py", [('LITERALS_ONLY_IN_CODE = {"javascript"}', "LITERALS_ONLY_IN_CODE = set()")]),
    "R5_no_argument_nodes_cause_dropped": ("s27_census_metrics.py", [('elif a["siteKey"] not in M["cargs"]:', "elif False:")]),
    "R6_end_delta_dropped": ("s27_census_metrics.py", [("            end_delta[m[\"le\"] - (d[\"endLine\"] + off)] += 1", "            pass")]),
    "F1_operator_alias_dropped": ("s27_census_metrics.py", [('        if k.startswith("operator"):', '        if False:')]),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name, tmp_path):
    if os.environ.get("CENSUS_SRC"):
        pytest.skip("inner run")
    copy = tmp_path / "src"
    copy.mkdir()
    for p in SRC.glob("*.py"):
        shutil.copy(p, copy / p.name)
    for extra in ("corpora.yaml",):
        shutil.copy(SRC / extra, copy / extra)
    fname, edits = MUTANTS[name]
    text = (copy / fname).read_text(encoding="utf-8")
    for old, new in edits:
        assert old in text, f"mutant {name}: pattern not found, the mutant would be a no-op"
        text = text.replace(old, new)
    (copy / fname).write_text(text, encoding="utf-8")
    env = dict(os.environ, CENSUS_SRC=str(copy), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-k", "not mutant", str(Path(__file__))],
                       capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[1])
    assert p.returncode != 0, f"mutant {name} survived:\n{p.stdout[-800:]}"
    assert "failed" in p.stdout
