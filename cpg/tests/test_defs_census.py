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
    dp = jl(tmp / "defs.jsonl", [{"k": "meta"}] + [{"k": "file", "path": f} for f in files] + defs + list(args))
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


# ---------------------------------------------------------------------------------------------------------- mutants
# Each mutant rewrites a COPY of the sources (never the repo's) and the same tests, run against the copy, must fail.
MUTANTS = {
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
