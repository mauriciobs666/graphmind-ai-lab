"""P1-05 (5a) — the normalizer core (`extractors/joern/normalize/core.py`): name and path rules, staged-source access,
scope numbering, value sets, the runtime check.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-05 behaviours (1) and (4); rules in
`docs/plans/connected-code-graph-bundle-spec2.md` §1.1-§1.2 and `-bundle-spec3.md` §4.1. Every expected value is written
out by hand, never derived from the function under test.
"""
from __future__ import annotations

import json
import sys

import pytest

from extractors.joern.normalize import core
from normalize_helpers import builder


# --- name_norm (spec2 §1.2) ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw, want", [
    ("Straße", "strasse"),            # casefold, not lower()
    ("ﬁle", "file"),                  # a ligature folds
    ("A\u030a", "\u00e5"),            # decomposed A + ring: the result is NFC (and folded)
    ("  Name \t", "name"),            # stripped
    ("MixedCase", "mixedcase"),
    ("\u0390", "\u0390"),            # casefold decomposes it (iota, dialytika, tonos); the result is NFC again
])
def test_name_norm(raw, want):
    assert core.name_norm(raw) == want


# --- quote_field / path_from / kw_head (spec2 §1.1) -----------------------------------------------------------------


@pytest.mark.parametrize("key, want", [
    ("abc", ".abc"),
    ("a_b$9", ".a_b$9"),
    ("_", "._"),
    ("1a", '["1a"]'),                 # an identifier cannot start with a digit
    ("a-b", '["a-b"]'),
    ("", '[""]'),
    ("a\n", '["a\\n"]'),              # a trailing newline is not an identifier (the `$` anchor trap)
    ("é", '["é"]'),                   # non-ASCII is quoted, and written unescaped
    ('q"t', '["q\\"t"]'),
])
def test_quote_field(key, want):
    assert core.quote_field(key) == want


@pytest.mark.parametrize("head, fields, want", [
    ("p", [], ("p", False)),
    ("p", ["a", "b", "c"], ("p.a.b.c", False)),          # exactly the cap
    ("p", ["a", "b", "c", "d"], ("p.a.b.c", True)),      # one over: cut, flagged
    ("p", ["a b", "c"], ('p["a b"].c', False)),
    ("kw.x", ["a", "b", "c", "d", "e"], ("kw.x.a.b.c", True)),
])
def test_path_from_caps_the_fields_after_the_head(head, fields, want):
    assert core.path_from(head, fields) == want


@pytest.mark.parametrize("name, want", [("name", "kw.name"), ("a b", 'kw["a b"]')])
def test_kw_head(name, want):
    assert core.kw_head(name) == want


# --- Source ---------------------------------------------------------------------------------------------------------


def test_source_lines_split_on_newline_only(tmp_path):
    (tmp_path / "a.py").write_bytes("x\r\ny\u2028z\x0cw\n".encode())
    src = core.Source(tmp_path)
    assert src.lines("a.py") == ["x\r", "y\u2028z\x0cw", ""]


def test_source_reads_utf8_and_caches(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("é\n", encoding="utf-8")
    src = core.Source(tmp_path)
    assert src.text("a.py") == "é\n"
    p.write_text("changed")
    assert src.text("a.py") == "é\n"


def test_source_refuses_bytes_that_are_not_utf8(tmp_path):
    (tmp_path / "a.py").write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        core.Source(tmp_path).text("a.py")


def test_source_exists_means_a_regular_file(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "f.py").write_text("")
    src = core.Source(tmp_path)
    assert src.exists("f.py") and not src.exists("d") and not src.exists("nope.py")


# --- facts ----------------------------------------------------------------------------------------------------------


def test_load_facts_groups_records_by_kind(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in ({"k": "f", "name": "a"}, {"k": "m", "id": "1"}, {"k": "f", "name": "b"})))
    got = core.load_facts(p)
    assert [o["name"] for o in got["f"]] == ["a", "b"] and len(got["m"]) == 1 and got["absent"] == []


def test_lang_files_python_is_py_only_sorted_and_unique():
    facts = {"f": [{"name": n} for n in ("b.py", "a.py", "a.py", "c.pyi", "d.ts", "N/A", "e.py.bak")]}
    assert core.lang_files(facts, "python") == ["a.py", "b.py"]


def test_lang_files_javascript_takes_every_js_ts_extension():
    names = ["a.ts", "b.tsx", "c.js", "d.jsx", "e.mjs", "f.cjs", "g.mts", "h.cts", "i.py", "j.json"]
    facts = {"f": [{"name": n} for n in names]}
    assert core.lang_files(facts, "javascript") == ["a.ts", "b.tsx", "c.js", "d.jsx", "e.mjs", "f.cjs", "g.mts", "h.cts"]


# --- runtime (behaviour 4) ------------------------------------------------------------------------------------------


def test_the_runtime_check_accepts_cpython_312():
    core.require_runtime((3, 12, 3), "cpython")
    core.require_runtime()               # the interpreter running this suite


@pytest.mark.parametrize("vi, impl", [((3, 13, 0), "cpython"), ((3, 11, 9), "cpython"), ((3, 12, 0), "pypy"), ((2, 12), "cpython")])
def test_the_runtime_check_refuses_anything_else(vi, impl):
    with pytest.raises(core.RuntimeRefused, match="CPython 3.12"):
        core.require_runtime(vi, impl)


def test_a_builder_asserts_the_runtime_at_start(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 13, 0, "final", 0))
    with pytest.raises(core.RuntimeRefused):
        core.Builder("python", ".", "r", {}, core.Source(tmp_path))


# --- Builder ---------------------------------------------------------------------------------------------------------


def test_builder_path_rewriting_follows_the_unit_root(tmp_path):
    at_root = core.Builder("python", ".", "r", {}, core.Source(tmp_path))
    nested = core.Builder("python", "falkor-chat/server", "r", {}, core.Source(tmp_path))
    assert at_root.P("a/b.py") == "a/b.py"
    assert nested.P("a/b.py") == "falkor-chat/server/a/b.py"


def test_builder_column_base_is_one_for_python_and_zero_for_javascript(tmp_path):
    assert core.Builder("python", ".", "r", {}, core.Source(tmp_path)).col_base == 1
    assert core.Builder("javascript", ".", "r", {}, core.Source(tmp_path)).col_base == 0


# --- Scope, seg, number_and_qualify (spec3 §4.1) --------------------------------------------------------------------


def scope(kind, base, parent=None, line=1, col=1, nid="0", accessor=None):
    return core.Scope(key=(kind, nid), kind=kind, name=base, base=base, line=line, col=col, lineEnd=line, parent=parent,
                      file="a.py", jname=base, node={"id": nid}, accessor=accessor)


def test_seg_numbers_only_duplicates_but_always_numbers_an_anon():
    assert core.seg(scope("function", "f")) == "f"
    s = scope("function", "f")
    s.dup = 3
    assert core.seg(s) == "f#3"
    a = scope("lambda", "<anon>")
    assert core.seg(a) == "<anon>#1"
    p = scope("accessor", "x", accessor="<get>")
    assert core.seg(p) == "x.<get>"
    p.dup = 2
    assert core.seg(p) == "x.<get>#2"


def test_numbering_is_per_parent_base_and_accessor_and_orders_by_position_then_cpg_id(tmp_path):
    B = core.Builder("python", ".", "r", {}, core.Source(tmp_path))
    mod = scope("module", "<module>", nid="m")
    c1 = scope("type", "C", mod, line=1, nid="c1")
    c2 = scope("type", "C", mod, line=9, nid="c2")
    f_in_c1 = scope("function", "f", c1, line=2, nid="f1")
    f_in_c2 = scope("function", "f", c2, line=10, nid="f2")
    # same line and column: the CPG id decides, as a string
    t_b = scope("function", "t", mod, line=20, col=5, nid="9")
    t_a = scope("function", "t", mod, line=20, col=5, nid="10")
    t_late = scope("function", "t", mod, line=21, col=1, nid="1")
    scopes = {s.key: s for s in (mod, c2, c1, f_in_c2, f_in_c1, t_late, t_b, t_a)}   # insertion order opposes the id order
    core.number_and_qualify(B, scopes)
    assert (mod.qname, c1.qname, c2.qname) == ("a.py::<module>", "a.py::C", "a.py::C#2")
    assert (f_in_c1.qname, f_in_c2.qname) == ("a.py::C.f", "a.py::C#2.f")     # numbering restarts under each parent
    assert (t_a.qname, t_b.qname, t_late.qname) == ("a.py::t", "a.py::t#2", "a.py::t#3")


def test_numbering_orders_by_line_before_column():
    B = core.Builder.__new__(core.Builder)
    B.root = "."
    mod = scope("module", "<module>", nid="m")
    right_early = scope("lambda", "<anon>", mod, line=5, col=30, nid="1")
    left_late = scope("lambda", "<anon>", mod, line=6, col=2, nid="2")
    core.number_and_qualify(B, {s.key: s for s in (left_late, right_early, mod)})
    assert (right_early.qname, left_late.qname) == ("a.py::<anon>#1", "a.py::<anon>#2")


# --- sym_name, span_start, fingerprint ------------------------------------------------------------------------------


def test_sym_name_of_a_constructor_is_its_types_name():
    t = scope("type", "Widget")
    ctor = scope("constructor", "<init>", t)
    assert core.sym_name(ctor) == "Widget"
    assert core.sym_name(scope("function", "run", t)) == "run"
    assert core.sym_name(scope("function", "<init>", None)) == "<init>"


def test_span_start_reaches_up_to_the_earliest_decorator(tmp_path):
    B = core.Builder("python", ".", "r", {}, core.Source(tmp_path))
    s = scope("function", "f", line=10)
    assert core.span_start(B, s) == 10
    s.decos = [("a", {"l": 8}), ("b", {"l": 9})]
    assert core.span_start(B, s) == 8
    s.decos = [("late", {"l": 12})]
    assert core.span_start(B, s) == 10


def test_fingerprint_hashes_the_stripped_nonblank_lines_of_the_decorator_inclusive_span(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n@d\ndef f():\n\n    return 1\n\ny = 2\n")
    B = core.Builder("python", ".", "r", {}, core.Source(tmp_path))
    s = scope("function", "f", line=3)
    s.file, s.lineEnd = "a.py", 5
    s.decos = [("d", {"l": 2})]
    # lines 2-5 -> "@d", "def f():", "", "    return 1"; blanks dropped, each stripped, joined with \n
    assert core.fingerprint(B, s) == "19e899b158cd65e83d161febf42d506a5995158da2a508cb046bab2e7e365fd7"
    s.lineEnd = 1                         # an end line above the start still covers the start line: the decorator line alone
    assert core.fingerprint(B, s) == "465d2682181501e3a78d379f6b7f1e601a29fe52df157a17b40742665122669c"
    s.lineEnd = 5
    s.decos = []                          # without the decorator line the hash differs
    assert core.fingerprint(B, s) == "f4c1dc7477122951937dc3c02ba03b2b4d603d045286fdd8accfa74b5b50ff83"


# --- Resolver (value sets) ------------------------------------------------------------------------------------------


@pytest.fixture
def R(tmp_path):
    B = core.Builder("python", "svc", "r", {}, core.Source(tmp_path))
    imports = {"a.py": [
        {"module": ".cfg", "names": [{"name": "TIMEOUT"}, {"name": "x", "alias": "y"}], "resolvedPath": "svc/cfg.py", "_sub": None},
        {"module": "pkg", "names": [{"name": "sub"}], "resolvedPath": "svc/pkg/__init__.py", "_sub": "svc/pkg/sub.py"},
        {"module": "ext", "names": [{"name": "Thing"}, {"name": "*"}], "resolvedPath": None, "_sub": None},
    ]}
    return core.Resolver(B, imports, {"svc/a.py": {"svc/a.py::LIMIT"}})


@pytest.mark.parametrize("code, want", [
    ('"é"', {"const": "é"}), ("'x'", {"const": "x"}), ("True", {"const": True}), ("42", {"const": 42}),
    ("`plain`", {"const": "plain"}),
])
def test_resolver_literals(R, code, want):
    got = R.element({"o": "lit", "c": code}, "a.py")
    assert json.dumps(got) == json.dumps(want)      # `True` is not `1`: compare as written


@pytest.mark.parametrize("code", ["1.5", "[1]", "f(x)", "`has ${x}`", "None"])
def test_resolver_unparsable_or_non_scalar_literal_is_runtime_built(R, code):
    assert R.element({"o": "lit", "c": code}, "a.py") == {"unknown": "runtime-built"}


def test_resolver_normalizes_a_literal_to_nfc(R):
    assert R.element({"o": "lit", "c": "'A\\u030a'"}, "a.py") == {"const": "\u00c5"}


@pytest.mark.parametrize("name, want", [
    ("LIMIT", {"ref": "svc/a.py::LIMIT"}),                              # a constant of the file
    ("TIMEOUT", {"ref": "svc/cfg.py::TIMEOUT"}),                       # a resolved import
    ("y", {"ref": "svc/cfg.py::x"}),                                   # through its alias, to the original name
    ("Thing", {"importRef": {"module": "ext", "name": "Thing"}}),      # unresolved import
    ("sub", {"importRef": {"module": "pkg", "name": "sub"}}),          # a submodule import is a reference by module
    ("None", {"unknown": "runtime-built"}), ("null", {"unknown": "runtime-built"}),
    ("other", {"unknown": "point-derived"}),
])
def test_resolver_names(R, name, want):
    assert R.element({"o": "name", "n": name}, "a.py") == want


def test_resolver_a_star_import_binds_no_name(R):
    assert R.element({"o": "name", "n": "*"}, "a.py") == {"unknown": "point-derived"}


@pytest.mark.parametrize("base, field, want", [
    ("TIMEOUT", "v", {"ref": "svc/cfg.py::TIMEOUT.v"}),
    ("sub", "K", {"ref": "svc/pkg/sub.py::K"}),
    ("Thing", "K", {"importRef": {"module": "ext", "name": "Thing.K"}}),
    ("nobody", "K", {"unknown": "point-derived"}),
])
def test_resolver_attributes(R, base, field, want):
    assert R.element({"o": "attr", "b": {"o": "name", "n": base}, "f": field}, "a.py") == want


def test_resolver_attribute_of_a_non_name_is_point_derived(R):
    assert R.element({"o": "attr", "b": {"o": "call"}, "f": "x"}, "a.py") == {"unknown": "point-derived"}


def test_resolver_templates_and_concatenation_drop_empty_parts(R):
    lit = lambda c: {"o": "lit", "c": c}
    assert R.element({"o": "fmt", "p": [lit("'a'"), lit("''"), {"o": "name", "n": "other"}]}, "a.py") == {
        "template": [{"const": "a"}, {"unknown": "point-derived"}]}
    assert R.element({"o": "add", "p": [lit("'a'"), lit("'b'")]}, "a.py") == {"concat": [{"const": "a"}, {"const": "b"}]}
    assert R.element({"o": "fmt", "p": [lit("''")]}, "a.py") == {"const": ""}


def test_resolver_depth_cap_and_nested_template_cap(R):
    lit = {"o": "lit", "c": "'a'"}
    deep = lit
    for _ in range(4):
        deep = {"o": "add", "p": [deep, {"o": "name", "n": "other"}]}
    # depth 1 add -> depth 2 add -> depth 3 add (with a nested concat) -> capped
    assert R.element(deep, "a.py") == {"concat": [{"concat": [{"unknown": "value-set-cap"}, {"unknown": "point-derived"}]},
                                                  {"unknown": "point-derived"}]}
    assert R.element(lit, "a.py", depth=5) == {"unknown": "value-set-cap"}


def test_resolver_calls_and_other_operators(R):
    assert R.element({"o": "call"}, "a.py") == {"unknown": "point-derived"}
    assert R.element({"o": "weird"}, "a.py") == {"unknown": "runtime-built"}
    assert R.values({"o": "lit", "c": "1"}, "a.py") == [{"const": 1}]
    assert core.only_unknown({"unknown": "x"}) and not core.only_unknown({"const": 1})


# --- collapse -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text, limit, want", [
    ("  a \n b\t\tc  ", 200, "a b c"),             # whitespace runs become one space, the ends are stripped
    ("x" * 300, 200, "x" * 200),                   # the default limit
    ("abc def", 4, "abc"),                         # a cut that leaves a trailing space is stripped again
    ("yyyyy  z", 6, "yyyyy"),
    ("abcdef", 3, "abc"),
    ("", 5, ""),
])
def test_collapse(text, limit, want):
    assert core.collapse(text, limit) == want


def test_collapse_defaults_to_two_hundred_characters():
    assert core.collapse("x" * 300) == "x" * 200 and core.collapse("y " * 150) == ("y " * 100).rstrip()


# --- the enclosing symbol of an import (the oracle's rule) -----------------------------------------------------------


def test_an_import_belongs_to_the_widest_non_module_non_type_span_that_holds_its_line(tmp_path):
    B = core.Builder("python", ".", "r", {}, core.Source(tmp_path))
    mod = scope("module", "<module>", nid="m")
    typ = scope("type", "T", mod, nid="t")
    outer = scope("function", "outer", mod, line=10, nid="o")
    inner = scope("function", "inner", outer, line=12, nid="i")
    other = scope("function", "other", mod, line=30, nid="x")
    for s, end in ((mod, 40), (typ, 9), (outer, 20), (inner, 14), (other, 33)):
        s.lineEnd, s.file, s.sid = end, "a.py", "sid-" + s.base
    other.decos = [("d", {"l": 29})]
    scopes = {s.key: s for s in (mod, typ, inner, outer, other)}           # the inner span is listed first on purpose
    assert core.import_symbol_id(B, scopes, "a.py", 10) == "sid-outer"       # a span holds its first line
    assert core.import_symbol_id(B, scopes, "a.py", 29) == "sid-other"       # and its decorators' lines
    assert core.import_symbol_id(B, scopes, "a.py", 28) is None
    assert core.import_symbol_id(B, scopes, "a.py", 13) == "sid-outer"      # the slice's rule: the widest span wins
    assert core.import_symbol_id(B, scopes, "a.py", 11) == "sid-outer"
    assert core.import_symbol_id(B, scopes, "a.py", 33) == "sid-other"
    assert core.import_symbol_id(B, scopes, "a.py", 20) == "sid-outer" and core.import_symbol_id(B, scopes, "a.py", 21) is None
    assert core.import_symbol_id(B, scopes, "a.py", 5) is None             # a module- or type-level line: no symbol
    assert core.import_symbol_id(B, scopes, "b.py", 13) is None
