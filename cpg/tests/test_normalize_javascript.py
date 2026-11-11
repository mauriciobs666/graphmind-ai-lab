"""P1-05 (5b) — the TS/JS frontend of the normalizer (`extractors/joern/normalize/javascript.py`).

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-05 behaviours (1) (TS/JS half) and (2). Rules:
`docs/plans/connected-code-graph-bundle-spec3.md` §4 (qname grammar, `<anon>#k`, constructors), §5.5 (`ret` synthesis, with
`-bundle-spec4.md` §2) and §5.7. Two kinds of test: hand-written facts and Babel nodes for each rule (expected values written by
hand), and one **real-shape golden**: the raw facts, source and astgen AST of one salesperson file
(`tests/fixtures/normalize_js/`, copied out of the stored slice run) whose expected records are the stored slice bundle's
own, minus `lastChanged*` (those need history; the whole-bundle comparison is in `tests/slice/`).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ccg import bundle
from extractors.joern import run as joern_run
from extractors.joern.normalize import assemble_bundle, core, javascript
from normalize_assembly_helpers import git_repo

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "normalize_js"
KINDS = ("f", "m", "t", "p", "r", "c", "cs", "const", "d", "cb", "deco")


def facts_of(**kinds):
    out = {k: [] for k in KINDS}
    out.update(kinds)
    return out


def m(id_, name, full, line, pk, pid, file="a.ts", le=None, c=0):
    return {"k": "m", "id": id_, "name": name, "full": full, "file": file, "l": line, "le": le or line, "c": c, "pk": pk,
            "pid": pid, "pfull": "", "code": ""}


def t(id_, name, full, line, pk="BLOCK", pid="x", file="a.ts", c=0):
    return {"k": "t", "id": id_, "name": name, "full": full, "file": file, "l": line, "c": c, "pk": pk, "pid": pid, "pfull": ""}


def builder(tmp_path, files, facts, root="."):
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    return core.Builder("javascript", root, "r", facts, core.Source(tmp_path))


# --- scope tree ------------------------------------------------------------------------------------------------------
CLASS_SRC = "// c\nclass C {\n  constructor(a) {}\n  m() { return () => 1; }\n}\nfunction f() { class In {} }\n"


def class_facts():
    return facts_of(
        f=[{"k": "f", "name": "a.ts"}],
        m=[m("1", ":program", "a.ts::program", 1, "TYPE_DECL", "t0", le=6),
           m("2", "<init>", "a.ts::program:C:<init>", 2, "TYPE_DECL", "t1"),          # the synthetic one: line 2 has no `constructor`
           m("3", "<init>", "a.ts::program:C:<init>2", 3, "TYPE_DECL", "t1", le=3),    # the written one
           m("4", "m", "a.ts::program:C:m", 4, "TYPE_DECL", "t1"),
           m("5", "<lambda>0", "a.ts::program:C:m:<lambda>0", 4, "METHOD", "4", c=16),
           m("6", "f", "a.ts::program:f", 6, "METHOD", "1"),
           m("7", "inner", "a.ts::program:f:inner", 6, "TYPE_DECL", "t9")],       # a member of the TYPE_DECL Joern makes for f
        t=[t("t0", ":program", "a.ts::program", 1, "NAMESPACE_BLOCK"),
           t("t1", "C", "a.ts::program:C", 2, "BLOCK"),
           t("t2", "<init>", "a.ts::program:C:<init>", 2),                              # no type of its own
           t("t3", "<lambda>0", "a.ts::program:C:m:<lambda>0", 4),
           t("t4", "m", "a.ts::program:C:m", 4),                                      # a method's own TYPE_DECL: shares its fullName
           t("t5", "<anon-class>0", "a.ts::program:<anon-class>0", 5),                 # Joern's object-type-literal artifact
           t("t8", "In", "a.ts::program:f:In", 6, "METHOD", "6"),
           t("t9", "f", "a.ts::program:f", 6)])                                        # f's own TYPE_DECL


def test_a_class_a_written_constructor_a_method_a_lambda_and_a_nested_class_get_their_qnames_and_kinds(tmp_path):
    B = builder(tmp_path, {"a.ts": CLASS_SRC}, class_facts())
    scopes, m2s, deco = javascript.build_scopes(B)
    got = {s.qname: (s.kind, s.parent.qname if s.parent else None) for s in scopes.values()}
    assert got == {
        "a.ts::<module>": ("module", None),
        "a.ts::C": ("type", "a.ts::<module>"),
        "a.ts::C.<init>": ("constructor", "a.ts::C"),
        "a.ts::C.m": ("method", "a.ts::C"),
        "a.ts::C.m.<anon>#1": ("lambda", "a.ts::C.m"),
        "a.ts::f": ("function", "a.ts::<module>"),
        "a.ts::f.In": ("type", "a.ts::f"),
        "a.ts::f.inner": ("function", "a.ts::f"),
    }
    assert deco == {}                                                    # TS/JS decorators are not extracted in P1


def test_the_synthetic_constructors_code_belongs_to_its_type_and_it_is_not_a_symbol(tmp_path):
    B = builder(tmp_path, {"a.ts": CLASS_SRC}, class_facts())
    scopes, m2s, _ = javascript.build_scopes(B)
    assert m2s["2"].qname == "a.ts::C" and "2" in m2s["2"].cpg_ids
    assert m2s["3"].qname == "a.ts::C.<init>"
    assert core.sym_name(m2s["3"]) == "C"                                 # a constructor is named by its type (spec3 §4.1 rule 10)


def test_a_method_in_a_functions_type_decl_hangs_under_the_function_of_the_same_full_name(tmp_path):
    B = builder(tmp_path, {"a.ts": CLASS_SRC}, class_facts())
    scopes, _, _ = javascript.build_scopes(B)
    by = {s.qname: s for s in scopes.values()}
    assert by["a.ts::f.inner"].parent is by["a.ts::f"]


def test_a_scope_whose_parent_cannot_be_resolved_hangs_under_its_files_module(tmp_path):
    facts = class_facts()
    facts["m"].append(m("9", "orphan", "a.ts::program:orphan", 6, "TYPE_DECL", "nowhere"))
    B = builder(tmp_path, {"a.ts": CLASS_SRC}, facts)
    scopes, _, _ = javascript.build_scopes(B)
    assert {s.qname: s.parent.qname for s in scopes.values() if s.base == "orphan"} == {"a.ts::orphan": "a.ts::<module>"}


def test_an_init_on_line_1_is_a_constructor_and_one_on_a_line_past_the_file_or_before_it_is_not(tmp_path):
    src = "constructor() {}\nx\nconstructor() {}"
    base = [m("0", ":program", "a.ts::program", 1, "TYPE_DECL", "t0", le=3)]
    for line, want in ((1, True), (0, False), (4, False)):
        facts = facts_of(f=[{"k": "f", "name": "a.ts"}],
                         m=base + [m("5", "<init>", "a.ts::program:C:<init>", line, "TYPE_DECL", "t1")],
                         t=[t("t1", "C", "a.ts::program:C", 1)])
        B = builder(tmp_path, {"a.ts": src}, facts)
        scopes, _, _ = javascript.build_scopes(B)
        assert (("m", "5") in scopes) is want, line


def test_a_method_inside_a_synthetic_init_hangs_under_the_class_the_init_belongs_to(tmp_path):
    facts = class_facts()
    facts["m"].append(m("20", "<lambda>1", "a.ts::program:C:<init>:<lambda>1", 3, "METHOD", "2", c=5))
    B = builder(tmp_path, {"a.ts": CLASS_SRC}, facts)
    scopes, _, _ = javascript.build_scopes(B)
    assert scopes[("m", "20")].parent.qname == "a.ts::C"


def test_a_method_whose_method_parent_is_unknown_or_not_hosted_by_a_type_decl_hangs_under_its_module(tmp_path):
    facts = class_facts()
    facts["m"].append(m("21", "lost", "a.ts::program:lost", 6, "METHOD", "nowhere"))
    # a skipped synthetic-looking init whose own parent is a METHOD, with a pid that happens to be a type scope's id
    facts["m"].append(m("22", "<init>", "a.ts::program:odd:<init>", 2, "METHOD", "t1"))
    facts["m"].append(m("23", "child", "a.ts::program:odd:child", 6, "METHOD", "22"))
    B = builder(tmp_path, {"a.ts": CLASS_SRC}, facts)
    scopes, _, _ = javascript.build_scopes(B)
    assert scopes[("m", "21")].parent.qname == "a.ts::<module>"
    assert scopes[("m", "23")].parent.qname == "a.ts::<module>"


def test_an_unresolved_parent_falls_back_to_the_module_of_its_own_file(tmp_path):
    facts = facts_of(f=[{"k": "f", "name": "a.ts"}, {"k": "f", "name": "b.ts"}],
                     m=[m("1", ":program", "a.ts::program", 1, "TYPE_DECL", "t0"),
                        m("2", ":program", "b.ts::program", 1, "TYPE_DECL", "t1", file="b.ts"),
                        m("3", "orphan", "b.ts::program:orphan", 1, "TYPE_DECL", "nowhere", file="b.ts")])
    B = builder(tmp_path, {"a.ts": "x\n", "b.ts": "y\n"}, facts)
    scopes, _, _ = javascript.build_scopes(B)
    assert scopes[("m", "3")].parent is scopes[("m", "2")]


def test_only_ts_and_js_files_make_symbols(tmp_path):
    facts = class_facts()
    facts["f"].append({"k": "f", "name": "b.py"})
    facts["m"].append(m("8", "<module>", "b.py:<module>", 1, "NAMESPACE_BLOCK", "n", file="b.py"))
    B = builder(tmp_path, {"a.ts": CLASS_SRC, "b.py": "x = 1\n"}, facts)
    scopes, _, _ = javascript.build_scopes(B)
    assert {s.file for s in scopes.values()} == {"a.ts"}


def test_two_arrow_functions_in_one_function_are_numbered_by_position_whatever_the_fact_order(tmp_path):
    src = "function f() {\n  const a = () => 1;\n  const b = () => 2;\n}\n"
    for order in ((0, 1), (1, 0)):
        lam = [m("2", "<lambda>0", "a.ts::program:f:<lambda>0", 2, "METHOD", "1", c=12),
               m("3", "<lambda>1", "a.ts::program:f:<lambda>1", 3, "METHOD", "1", c=12)]
        facts = facts_of(f=[{"k": "f", "name": "a.ts"}],
                         m=[m("0", ":program", "a.ts::program", 1, "TYPE_DECL", "t0", le=4),
                            m("1", "f", "a.ts::program:f", 1, "METHOD", "0", le=4)] + [lam[i] for i in order])
        B = builder(tmp_path, {"a.ts": src}, facts)
        scopes, _, _ = javascript.build_scopes(B)
        assert {s.qname: s.line for s in scopes.values() if s.kind == "lambda"} == {"a.ts::f.<anon>#1": 2, "a.ts::f.<anon>#2": 3}


def redeclared_function_facts(order, *, one_line=False, swap_ids=False):
    """Two declarations of one function (one fullName), each with an inner function: six method facts, in ``order``.
    ``one_line``: both on line 1, told apart by column; ``swap_ids``: the later declaration has the lower CPG id."""
    l2, c2 = (1, 40) if one_line else (2, 0)
    id1, id2 = ("2", "1") if swap_ids else ("1", "2")
    ms = [m("0", ":program", "a.ts::program", 1, "TYPE_DECL", "t0", le=2),
          m(id1, "f", "a.ts::program:f", 1, "METHOD", "0"),
          m(id2, "f", "a.ts::program:f", l2, "METHOD", "0", c=c2),
          m("3", "inner", "a.ts::program:f:inner", 1, "TYPE_DECL", "t1"),        # a member of the first f's TYPE_DECL
          m("4", "inner", "a.ts::program:f:inner", l2, "TYPE_DECL", "t2", c=c2 + 15),   # a member of the second f's
          m("5", "other", "a.ts::program:o", 2, "TYPE_DECL", "t0")]
    ts = [t("t0", ":program", "a.ts::program", 1, "NAMESPACE_BLOCK"),
          t("t1", "f", "a.ts::program:f", 1), t("t2", "f", "a.ts::program:f", l2, c=c2)]
    return facts_of(f=[{"k": "f", "name": "a.ts"}], m=[ms[i] for i in order], t=ts)


WANT_REDECL = {(("a.ts::<module>", None), ("a.ts::f", "a.ts::<module>"), ("a.ts::f#2", "a.ts::<module>"),
                ("a.ts::f#2.inner", "a.ts::f#2"), ("a.ts::f.inner", "a.ts::f"), ("a.ts::other", "a.ts::<module>"))}


@pytest.mark.parametrize("one_line, swap_ids", [(False, False), (False, True), (True, False)])
def test_two_declarations_of_one_function_each_keep_their_own_inner_functions_in_every_fact_order(tmp_path, one_line, swap_ids):
    import itertools
    src = ("function f() { function inner() {} }" + (" " * 3 if one_line else "\n") + "function f() { function inner() {} }\n")
    trees = set()
    for order in itertools.permutations(range(6)):
        B = builder(tmp_path, {"a.ts": src}, redeclared_function_facts(order, one_line=one_line, swap_ids=swap_ids))
        scopes, _, _ = javascript.build_scopes(B)
        trees.add(tuple(sorted((s.qname, s.parent.qname if s.parent else None) for s in scopes.values())))
    assert trees == WANT_REDECL


def test_two_classes_of_one_name_each_keep_their_own_members_in_every_type_and_method_order(tmp_path):
    import itertools
    src = "class C { m() {} }\nclass C { m() {} }\n"
    trees = set()
    for tord in itertools.permutations(range(3)):
        for mord in itertools.permutations(range(3)):
            ms = [m("0", ":program", "a.ts::program", 1, "TYPE_DECL", "t0", le=2),
                  m("1", "m", "a.ts::program:C:m", 1, "TYPE_DECL", "t1"), m("2", "m", "a.ts::program:C:m", 2, "TYPE_DECL", "t2")]
            ts = [t("t0", ":program", "a.ts::program", 1, "NAMESPACE_BLOCK"), t("t1", "C", "a.ts::program:C", 1),
                  t("t2", "C", "a.ts::program:C", 2)]
            facts = facts_of(f=[{"k": "f", "name": "a.ts"}], m=[ms[i] for i in mord], t=[ts[i] for i in tord])
            scopes, _, _ = javascript.build_scopes(builder(tmp_path, {"a.ts": src}, facts))
            trees.add(tuple(sorted((s.qname, s.parent.qname if s.parent else None) for s in scopes.values())))
    assert trees == {(("a.ts::<module>", None), ("a.ts::C", "a.ts::<module>"), ("a.ts::C#2", "a.ts::<module>"),
                      ("a.ts::C#2.m", "a.ts::C#2"), ("a.ts::C.m", "a.ts::C"))}


def test_the_module_scope_starts_at_line_1_col_1_and_owns_its_program_method(tmp_path):
    B = builder(tmp_path, {"a.ts": CLASS_SRC}, class_facts())
    scopes, m2s, _ = javascript.build_scopes(B)
    mod = scopes[("m", "1")]
    assert (mod.kind, mod.line, mod.col, mod.lineEnd) == ("module", 1, 1, 6)
    assert mod.cpg_ids == ["1"] and m2s["1"] is mod


def test_an_empty_files_module_still_spans_line_1(tmp_path):
    facts = facts_of(f=[{"k": "f", "name": "a.ts"}], m=[m("1", ":program", "a.ts::program", 1, "TYPE_DECL", "t0")])
    B = builder(tmp_path, {"a.ts": ""}, facts)
    scopes, _, _ = javascript.build_scopes(B)
    assert scopes[("m", "1")].lineEnd == 1


@pytest.mark.parametrize("name", ["<init>", "<lambda>0", ":program", "<anon-class>0"])
def test_a_type_decl_named_as_a_joern_artifact_is_not_a_symbol_even_when_no_method_shares_its_full_name(tmp_path, name):
    facts = class_facts()
    facts["t"].append(t("t77", name, "a.ts::program:unmatched:" + name, 6))
    B = builder(tmp_path, {"a.ts": CLASS_SRC}, facts)
    scopes, _, _ = javascript.build_scopes(B)
    assert not [s for s in scopes.values() if s.node.get("id") == "t77"]


# --- imports ---------------------------------------------------------------------------------------------------------
def tree(tmp_path, names, root="."):
    B = builder(tmp_path, {n: "x\n" for n in names}, facts_of(), root=root)
    return B


@pytest.mark.parametrize("importer, spec, want", [
    ("src/x.ts", "./b", "src/b.tsx"),                      # an extension is appended, in the order .ts .tsx .js ...
    ("src/x.ts", "./a", "src/a.ts"),                       # .ts before .tsx when both exist
    ("src/x.ts", "./lib", "src/lib/index.js"),             # a directory resolves to its index
    ("src/x.ts", "./b.tsx", "src/b.tsx"),                  # the written name itself first
    ("src/x.ts", "./both", "src/both.ts"),                 # a file before a directory index
    ("src/x.ts", "../src/a", "src/a.ts"),                  # `..` and `.` segments normalised
    ("src/x.ts", "../../outside", None),                   # climbing past the unit root
    ("src/x.ts", "pkg", None),                             # a bare specifier is not a unit file
    ("src/x.ts", "./c.json", None),                        # an existing file that is not TS/JS
    ("src/x.ts", "./missing", None),
])
def test_js_resolve(tmp_path, importer, spec, want):
    B = tree(tmp_path, ["src/a.ts", "src/a.tsx", "src/b.tsx", "src/lib/index.js", "src/both.ts", "src/both/index.ts", "src/c.json"])
    assert javascript.js_resolve(B, importer, spec) == want


def test_js_resolve_prefixes_the_unit_root(tmp_path):
    B = tree(tmp_path, ["src/a.ts"], root="web")
    assert javascript.js_resolve(B, "src/x.ts", "./a") == "web/src/a.ts"


def spec(kind, local, imported=None):
    s = {"type": kind, "local": {"name": local}}
    if imported is not None:
        s["imported"] = imported
    return s


def import_decl(source, specifiers, line=1, col=0):
    return {"type": "ImportDeclaration", "source": {"value": source}, "specifiers": specifiers,
            "loc": {"start": {"line": line, "column": col}}}


def test_import_specifiers_are_named_default_and_namespace_with_aliases_and_sorted_and_deduplicated(tmp_path):
    B = tree(tmp_path, ["src/x.ts", "src/dep.ts"])
    node = import_decl("./dep", [
        spec("ImportSpecifier", "z", {"name": "b"}), spec("ImportSpecifier", "a", {"name": "a"}),
        spec("ImportSpecifier", "b2", {"name": "b"}), spec("ImportSpecifier", "z", {"name": "b"}),     # a repeated alias
        spec("ImportDefaultSpecifier", "D"), spec("ImportNamespaceSpecifier", "NS"),
        spec("ImportSpecifier", "q", {"value": "str-name"})])                                          # a string-named export
    [r] = javascript.js_imports(B, "src/x.ts", {"body": [node]})
    assert r["names"] == [{"name": "*", "alias": "NS"}, {"name": "a"}, {"name": "b", "alias": "b2"}, {"name": "b", "alias": "z"},
                          {"name": "default", "alias": "D"}, {"name": "str-name", "alias": "q"}]
    assert (r["module"], r["dynamic"], r["resolvedPath"], r["line"], r["col"]) == ("./dep", False, "src/dep.ts", 1, 0)


def test_a_plain_name_sorts_before_its_own_aliases(tmp_path):
    B = tree(tmp_path, ["src/x.ts", "src/dep.ts"])
    node = import_decl("./dep", [spec("ImportSpecifier", "c", {"name": "b"}), spec("ImportSpecifier", "b", {"name": "b"}),
                                 spec("ImportSpecifier", "a", {"name": "b"})])
    [r] = javascript.js_imports(B, "src/x.ts", {"body": [node]})
    assert r["names"] == [{"name": "b"}, {"name": "b", "alias": "a"}, {"name": "b", "alias": "c"}]


def test_a_dynamic_import_with_a_literal_resolves_and_one_without_is_dynamic_and_unresolved(tmp_path):
    B = tree(tmp_path, ["src/x.ts", "src/dep.ts"])
    lit = {"type": "CallExpression", "callee": {"type": "Import"}, "arguments": [{"type": "StringLiteral", "value": "./dep"}],
           "loc": {"start": {"line": 3, "column": 4}}}
    var = {"type": "CallExpression", "callee": {"type": "Import"}, "arguments": [{"type": "Identifier", "name": "p"}],
           "loc": {"start": {"line": 5, "column": 0}}}
    other = {"type": "CallExpression", "callee": {"type": "Identifier"}, "arguments": [], "loc": {"start": {"line": 6, "column": 0}}}
    out = javascript.js_imports(B, "src/x.ts", {"a": [var, other], "b": lit})
    assert [(r["line"], r["module"], r["dynamic"], r["resolvedPath"], r["names"]) for r in out] == [
        (3, "./dep", True, "src/dep.ts", []), (5, "<dynamic>", True, None, [])]


def test_imports_are_ordered_by_position_not_by_ast_walk_order(tmp_path):
    B = tree(tmp_path, ["src/x.ts"])
    a, b = import_decl("p", [], line=9, col=0), import_decl("q", [], line=2, col=5)
    assert [r["module"] for r in javascript.js_imports(B, "src/x.ts", [a, b])] == ["q", "p"]
    assert [r["module"] for r in javascript.js_imports(B, "src/x.ts", [b, a])] == ["q", "p"]


# --- type spans (UTF-16) ---------------------------------------------------------------------------------------------
def utf16_len(s):
    return len(s.encode("utf-16-le")) // 2


def decl(kind, name, line, end, with_id=True):
    n = {"type": kind, "loc": {"start": {"line": line, "column": 0}}, "end": end}
    if with_id:
        n["id"] = {"name": name}
    return n


def test_the_utf16_index_counts_an_astral_character_as_two_units():
    assert javascript.utf16_line_index("\uffff\n") == [1]                       # U+FFFF is one unit, the last BMP code point
    assert javascript.utf16_line_index("\U00010000\n") == [2]                   # U+10000 is the first astral one: two units
    assert javascript.utf16_line_index("a\n\U0001F600b\n") == [1, 5]
    assert javascript.utf16_line_index("\n\n") == [0, 1]


def test_end_line_is_the_line_of_the_last_character_of_the_node():
    nl = [4, 9]                                        # three lines: [0,4] [5,9] [10..]
    assert [javascript.end_line(nl, e) for e in (1, 4, 5, 6, 9, 10, 11)] == [1, 1, 1, 2, 2, 2, 3]


def test_a_type_declaration_after_an_astral_character_gets_its_end_line_from_utf16_units():
    text = "const s = '\U0001F600\U0001F600\U0001F600\U0001F600';\ninterface I {\n  a: 1;\n}\nconst t = 1;\n"
    start = utf16_len("const s = '\U0001F600\U0001F600\U0001F600\U0001F600';\n")
    end = start + utf16_len("interface I {\n  a: 1;\n}")
    out = javascript.js_type_spans({"body": [decl("TSInterfaceDeclaration", "I", 2, end)]}, text)
    assert dict(out) == {"I": [(2, 4, "TSInterfaceDeclaration")]}       # code points would put the end past line 4


def test_only_named_type_nodes_make_spans_and_a_repeated_name_keeps_every_declaration():
    text = "interface I {}\ntype T = 1;\nenum E {}\nclass K {}\nconst c = class {};\nnamespace N {}\ninterface I {}\n"
    nodes = [decl("TSInterfaceDeclaration", "I", 1, 14), decl("TSTypeAliasDeclaration", "T", 2, 26),
             decl("TSEnumDeclaration", "E", 3, 35), decl("ClassDeclaration", "K", 4, 45),
             decl("ClassExpression", "", 5, 64, with_id=False), decl("TSModuleDeclaration", "N", 6, 85),
             decl("TSInterfaceDeclaration", "I", 7, 100), decl("FunctionDeclaration", "fn", 1, 5)]
    out = javascript.js_type_spans({"body": nodes}, text)
    assert sorted(out) == ["E", "I", "K", "N", "T"] and len(out["I"]) == 2
    assert out["I"][0][2] == "TSInterfaceDeclaration"


def test_a_type_takes_the_span_of_the_declaration_that_starts_nearest_its_cpg_line(tmp_path):
    src = "interface I {\n}\n\n\n\n\n\n\n\ninterface I {\n}\n"
    second = utf16_len("interface I {\n}\n" + "\n" * 7)                 # where line 10 starts
    ast_ = {"body": [decl("TSInterfaceDeclaration", "I", 1, 15),
                     decl("TSInterfaceDeclaration", "I", 10, second + utf16_len("interface I {\n}"))]}
    facts = facts_of(f=[{"k": "f", "name": "a.ts"}],
                     m=[m("0", ":program", "a.ts::program", 1, "TYPE_DECL", "t0", le=11)],
                     t=[t("t0", ":program", "a.ts::program", 1, "NAMESPACE_BLOCK"), t("t1", "I", "a.ts::program:I", 9)])
    B = builder(tmp_path, {"a.ts": src}, facts)
    scopes, _, _ = javascript.build_scopes(B)
    javascript.apply_js_type_spans(B, scopes, {"a.ts": ast_})
    typ = next(s for s in scopes.values() if s.is_type)
    assert (typ.line, typ.lineEnd) == (10, 11)


def test_the_nearest_declaration_is_judged_by_its_start_line_not_its_end_line(tmp_path):
    ast_ = {"body": [decl("TSInterfaceDeclaration", "I", 5, 6),          # start 3 away from line 8, end 2 away
                     decl("TSInterfaceDeclaration", "I", 10, 30)]}        # start 2 away from line 8, end 22 away
    facts = facts_of(f=[{"k": "f", "name": "a.ts"}],
                     m=[m("0", ":program", "a.ts::program", 1, "TYPE_DECL", "t0", le=35)],
                     t=[t("t0", ":program", "a.ts::program", 1, "NAMESPACE_BLOCK"), t("t1", "I", "a.ts::program:I", 8)])
    B = builder(tmp_path, {"a.ts": "\n" * 35}, facts)
    scopes, _, _ = javascript.build_scopes(B)
    javascript.apply_js_type_spans(B, scopes, {"a.ts": ast_})
    typ = next(s for s in scopes.values() if s.is_type)
    assert (typ.line, typ.lineEnd) == (10, 30)


def test_a_type_without_an_ast_keeps_the_one_line_span_the_cpg_gave_it(tmp_path):
    facts = facts_of(f=[{"k": "f", "name": "a.ts"}],
                     m=[m("0", ":program", "a.ts::program", 1, "TYPE_DECL", "t0", le=3)],
                     t=[t("t0", ":program", "a.ts::program", 1, "NAMESPACE_BLOCK"), t("t1", "I", "a.ts::program:I", 2)])
    B = builder(tmp_path, {"a.ts": "x\ninterface I {}\n\n"}, facts)
    scopes, _, _ = javascript.build_scopes(B)
    javascript.apply_js_type_spans(B, scopes, {})
    typ = next(s for s in scopes.values() if s.is_type)
    assert (typ.line, typ.lineEnd) == (2, 2)


# --- parameters ------------------------------------------------------------------------------------------------------
def test_the_this_parameter_is_dropped_and_the_others_are_indexed_from_zero():
    ps = [{"i": 0, "name": "this", "l": 1, "c": 0}, {"i": 1, "name": "a", "l": 1, "c": 0}, {"i": 2, "name": "b", "l": 1, "c": 0}]
    assert [(i, n) for i, n, _ in javascript.js_params(ps)] == [(0, "a"), (1, "b")]


def test_a_destructured_or_unnamed_parameter_is_named_by_its_collapsed_source_code_at_most_60_characters():
    long = "{ " + "alpha,   " * 12 + "}"
    ps = [{"i": 1, "name": "_tmp_0", "l": 1, "c": 0, "code": "{ a,\n   b }"}, {"i": 2, "name": "", "l": 1, "c": 0, "code": long},
          {"i": 3, "name": "", "l": 1, "c": 0}]
    got = [n for _, n, _ in javascript.js_params(ps)]
    assert got[0] == "{ a, b }" and got[2] == "param"
    assert len(got[1]) == 60 and "  " not in got[1] and got[1] == core.collapse(long, 60)


# --- whole-file assembly: the real-shape golden ----------------------------------------------------------------------
REL = "src/components/message/composerNotice.ts"
UUID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    work = tmp_path_factory.mktemp("jsgolden")
    repo = work / "repo"
    shutil.copytree(FIXTURE / "stage", repo / "salesperson")
    commit = git_repo(repo, {})
    return repo, commit, core.load_facts(FIXTURE / "facts.jsonl")


def assemble_golden(golden, facts=None, **kw):
    repo, commit, loaded = golden
    return assemble_bundle.assemble(lang="javascript", unit_root="salesperson", repo_id="graphmind-ai-lab",
                                    facts=facts or loaded, stage=repo / "salesperson", astgen_dir=FIXTURE / "astgen", repo=repo,
                                    commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC,
                                    built_at="2026-01-01T00:00:00Z", **kw)


def expected():
    return [json.loads(line) for line in (FIXTURE / "expected.jsonl").read_text(encoding="utf-8").splitlines()]


def strip_history(r):
    return {k: v for k, v in r.items() if not k.startswith("lastChanged")}


def test_a_real_typescript_file_yields_the_records_the_slice_stored_for_it(golden):
    records, _ = assemble_golden(golden)
    want = {r["id"]: r for r in expected()}
    got = {r["id"]: strip_history(r) for r in records[1:]}
    assert got == want
    assert len(got) == len(records) - 1                                 # no id repeated


def test_the_golden_file_holds_the_shapes_it_is_there_for(golden):
    by = {}
    for r in expected():
        by.setdefault(r["type"], []).append(r)
    syms = {r["qname"]: r["kind"] for r in by["symbol"]}
    assert syms == {"salesperson/src/components/message/composerNotice.ts::<module>": "module",
                    "salesperson/src/components/message/composerNotice.ts::ComposerNotice": "type",
                    "salesperson/src/components/message/composerNotice.ts::ComposerNoticeTone": "type",
                    "salesperson/src/components/message/composerNotice.ts::composerNoticeFor": "function"}
    iface = next(r for r in by["symbol"] if r["name"] == "ComposerNotice")
    assert (iface["lineStart"], iface["lineEnd"]) == (16, 19)           # the Babel span of the interface, not the CPG's one line
    assert any(r["resolvedPath"].endswith("api/dispatch.ts") for r in by["import"] if "resolvedPath" in r)
    assert len(by["ret"]) == 13 and len(by["controlledBy"]) == 27


def test_every_record_of_the_golden_passes_the_bundle_validator(golden):
    records, _ = assemble_golden(golden)
    bundle.loads(bundle.dumps(records))


# --- an expression-bodied arrow function has a synthesized ret (spec4 §2, spec3 §5.5) ---------------------------------
ARROW_SRC = "const f = (q) => q;\nfunction g() {\n  return 1;\n}\n"


def arrow_facts(**extra):
    return facts_of(f=[{"k": "f", "name": "a.ts"}],
                    m=[m("0", ":program", "a.ts::program", 1, "TYPE_DECL", "t0", le=4),
                       m("1", "<lambda>0", "a.ts::program:<lambda>0", 1, "METHOD", "0", c=10),
                       m("2", "g", "a.ts::program:g", 2, "METHOD", "0", le=4)],
                    t=[t("t0", ":program", "a.ts::program", 1, "NAMESPACE_BLOCK")],
                    r=[{"k": "r", "m": "2", "id": "9", "l": 3, "c": 2}], **extra)


def test_an_arrow_function_without_a_return_gets_one_synthesized_ret_at_its_line_and_a_function_does_not(tmp_path):
    repo = tmp_path / "repo"
    commit = git_repo(repo, {"a.ts": ARROW_SRC})
    records, report = assemble_bundle.assemble(lang="javascript", unit_root=".", repo_id="r", facts=arrow_facts(), stage=repo,
                                               astgen_dir=None, repo=repo, commit=commit, audit_ids=[UUID],
                                               extract_script=joern_run.EXTRACT_SC, built_at="2026-01-01T00:00:00Z")
    syms = {r["id"]: r["qname"] for r in records if r["type"] == "symbol"}
    rets = sorted((syms[r["symbolId"]], r["ordinal"], r["line"]) for r in records if r["type"] == "ret")
    assert rets == [("a.ts::<anon>#1", 0, 1), ("a.ts::g", 0, 3)]
    assert report["syntheticRetsForExpressionLambdas"] == 1


def test_a_type_decl_before_every_declaration_of_its_function_belongs_to_the_first_of_them():
    ms = {"1": m("1", "f", "a.ts::f", 5, "METHOD", "0"), "2": m("2", "f", "a.ts::f", 9, "METHOD", "0"), "3": m("3", "g", "a.ts::g", 1, "METHOD", "0")}
    m2s = {k: core.Scope(key=("m", k), kind="function", name="f", base="f", line=1, col=0, lineEnd=1, parent=None, file="a.ts",
                         jname="f", node=ms[k]) for k in ms}
    td = t("t1", "f", "a.ts::f", 2)
    assert javascript.function_of_type_decl(td, ms, m2s) is m2s["1"]
    assert javascript.function_of_type_decl(t("t2", "f", "a.ts::f", 9), ms, m2s) is m2s["2"]
    assert javascript.function_of_type_decl(t("t3", "h", "a.ts::h", 9), ms, m2s) is None
