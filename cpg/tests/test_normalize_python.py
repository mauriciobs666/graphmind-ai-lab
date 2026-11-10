"""P1-05 (5a) — the Python frontend of the normalizer (`extractors/joern/normalize/python.py`).

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-05 behaviours (1) (Python half) and the determinism
finding of the P1-04 gate. Rules: `docs/plans/connected-code-graph-bundle-spec3.md` §4 (qname grammar, `<anon>#k`,
accessors, decorators, overloads). The raw facts come from `normalize_helpers.py` (ids deliberately not in source
order, `le` overshooting as pysrc2cpg's does); the expected qnames, kinds and spans are written by hand.
"""
from __future__ import annotations

import textwrap

import pytest

from extractors.joern.normalize import core, python
from normalize_helpers import builder

SRC = textwrap.dedent('''\
    import os


    def top():
        pass


    def top():
        pass


    class C:
        def m(self):
            pass

        @property
        def x(self):
            return 1

        @x.setter
        def x(self, v):
            pass

        @staticmethod
        def s(a):
            pass

        class Inner:
            def deep(self):
                pass


    def outer():
        f = lambda a: a; g = lambda: 1
        h = [lambda: 2][0]
        def inner():
            pass
''')

#: qname -> (kind, parent qname or None, lineStart incl. decorators, lineEnd), counted by hand from SRC.
WANT = {
    "pkg/a.py::<module>": ("module", None, 1, 37),
    "pkg/a.py::top": ("function", "pkg/a.py::<module>", 4, 5),
    "pkg/a.py::top#2": ("function", "pkg/a.py::<module>", 8, 9),
    "pkg/a.py::C": ("type", "pkg/a.py::<module>", 12, 30),
    "pkg/a.py::C.m": ("method", "pkg/a.py::C", 13, 14),
    "pkg/a.py::C.x.<get>": ("accessor", "pkg/a.py::C", 16, 18),
    "pkg/a.py::C.x.<set>": ("accessor", "pkg/a.py::C", 20, 22),
    "pkg/a.py::C.s": ("method", "pkg/a.py::C", 24, 26),
    "pkg/a.py::C.Inner": ("type", "pkg/a.py::C", 28, 30),
    "pkg/a.py::C.Inner.deep": ("method", "pkg/a.py::C.Inner", 29, 30),
    "pkg/a.py::outer": ("function", "pkg/a.py::<module>", 33, 37),
    "pkg/a.py::outer.<anon>#1": ("lambda", "pkg/a.py::outer", 34, 34),
    "pkg/a.py::outer.<anon>#2": ("lambda", "pkg/a.py::outer", 34, 34),
    "pkg/a.py::outer.<anon>#3": ("lambda", "pkg/a.py::outer", 35, 35),
    "pkg/a.py::outer.inner": ("function", "pkg/a.py::outer", 36, 37),
}


def summary(scopes):
    return {s.qname: (s.kind, s.parent.qname if s.parent else None, s.line, s.lineEnd) for s in scopes.values()}


def scopes_of(tmp_path, files=None, **kw):
    B = builder(tmp_path, files or {"pkg/a.py": SRC}, **kw)
    scopes, m2s, deco = python.build_scopes(B)
    return B, scopes, m2s, deco


# --- the qname grammar, kinds and parents -------------------------------------------------------------------------


def test_the_golden_file_has_the_lines_the_expectations_assume():
    lines = SRC.split("\n")
    assert lines[3] == "def top():" and lines[27] == "    class Inner:" and lines[35] == "    def inner():"


def test_qnames_kinds_and_parents_follow_the_scope_tree(tmp_path):
    _, scopes, _, _ = scopes_of(tmp_path)
    got = {q: v[:2] for q, v in summary(scopes).items()}
    assert got == {q: v[:2] for q, v in WANT.items()}


def test_the_module_scope_spans_the_file_and_an_empty_file_one_line(tmp_path):
    _, scopes, _, _ = scopes_of(tmp_path, {"e.py": ""})
    (m,) = scopes.values()
    assert (m.qname, m.line, m.lineEnd) == ("e.py::<module>", 1, 1)


def test_synthetic_class_machinery_and_overload_stubs_are_not_symbols(tmp_path):
    src = textwrap.dedent('''\
        from typing import overload
        import typing


        @overload
        def f(a: int) -> int: ...
        @typing.overload
        def f(a: str) -> str: ...
        def f(a): return a
        class K:
            pass
    ''')
    B, scopes, m2s, _ = scopes_of(tmp_path, {"o.py": src})
    # the two overload stubs vanish *before* numbering, so the implementation is `f`, not `f#3`
    assert sorted(s.qname for s in scopes.values()) == ["o.py::<module>", "o.py::K", "o.py::f"]
    names = {m["name"] for m in B.facts["m"]}
    assert {"<body>"} <= names                                  # the machinery is in the facts ...
    assert all("<body>" not in s.qname for s in scopes.values())  # ... and not in the symbols
    assert len(m2s) == 3 and any(s.cpg_ids for s in scopes.values())


def test_the_stub_overloads_do_not_leave_the_method_to_cpg_id_map(tmp_path):
    src = "import typing\n\n\n@typing.overload\ndef f(a): ...\n\n\ndef f(a): return a\n"
    B, scopes, m2s, _ = scopes_of(tmp_path, {"o.py": src})
    stub = next(m for m in B.facts["m"] if m["name"] == "f" and m["l"] == 5)
    assert stub["id"] not in m2s and len(m2s) == 2


def test_decorator_names_are_read_from_the_source_without_arguments(tmp_path):
    src = "class C:\n    @property\n    def a(self): return 1\n    @a.deleter\n    def a(self): pass\n"
    _, scopes, _, _ = scopes_of(tmp_path, {"d.py": src})
    assert sorted(s.qname for s in scopes.values()) == ["d.py::<module>", "d.py::C", "d.py::C.a.<del>", "d.py::C.a.<get>"]


def test_a_property_setter_must_match_the_function_name(tmp_path):
    src = "class C:\n    @other.setter\n    def a(self, v): pass\n"
    _, scopes, _, _ = scopes_of(tmp_path, {"d.py": src})
    assert sorted(s.qname for s in scopes.values()) == ["d.py::<module>", "d.py::C", "d.py::C.a"]


def test_only_python_files_make_symbols(tmp_path):
    B = builder(tmp_path, {"a.py": "def f(): pass\n"})
    B.facts["f"].append({"k": "f", "name": "b.txt"})
    B.facts["m"].append({"k": "m", "id": "9", "name": "g", "full": "b.txt:<module>.g", "file": "b.txt", "l": 1, "le": 1,
                         "c": 1, "pk": "METHOD", "pid": "1", "pfull": "", "code": ""})
    B.facts["m"].append({"k": "m", "id": "8", "name": "<module>", "full": "b.txt:<module>", "file": "b.txt", "l": 1,
                         "le": 1, "c": 1, "pk": "NAMESPACE_BLOCK", "pid": "ns", "pfull": "", "code": ""})
    scopes, _, _ = python.build_scopes(B)
    assert sorted(s.qname for s in scopes.values()) == ["a.py::<module>", "a.py::f"]


def test_a_unit_below_the_repo_root_prefixes_every_qname(tmp_path):
    _, scopes, _, _ = scopes_of(tmp_path, {"p/a.py": "def f(): pass\n"}, root="falkor-chat/server")
    assert sorted(s.qname for s in scopes.values()) == ["falkor-chat/server/p/a.py::<module>", "falkor-chat/server/p/a.py::f"]


def test_decorators_of_a_function_are_attached_with_their_written_names(tmp_path):
    _, scopes, _, deco = scopes_of(tmp_path)
    by = {s.qname: s for s in scopes.values()}
    assert [n for n, _ in by["pkg/a.py::C.x.<get>"].decos] == ["property"]
    assert [n for n, _ in by["pkg/a.py::C.x.<set>"].decos] == ["x.setter"]
    assert [n for n, _ in by["pkg/a.py::C.s"].decos] == ["staticmethod"]
    assert by["pkg/a.py::top"].decos == []
    assert len(deco) == 3


# --- spans from the source parse ----------------------------------------------------------------------------------


def test_spans_come_from_the_source_parse_and_include_decorators(tmp_path):
    B, scopes, _, _ = scopes_of(tmp_path)
    misses = python.apply_py_spans(B, scopes)
    assert misses == 0 and B.report["pythonSpanLookupMisses"] == 0
    got = {q: (v[2], v[3]) for q, v in summary(scopes).items() if not q.endswith("<module>")}
    assert got == {q: v[2:] for q, v in WANT.items() if not q.endswith("<module>")}
    assert max(s.node["le"] for s in scopes.values() if s.kind == "function") > max(s.lineEnd for s in scopes.values() if s.kind == "function")  # le overshoots, the parse does not


def test_a_scope_the_source_parse_cannot_find_keeps_its_cpg_span_and_is_counted(tmp_path):
    B, scopes, _, _ = scopes_of(tmp_path, {"a.py": "def f():\n    pass\n"})
    f = next(s for s in scopes.values() if s.base == "f")
    f.line = 7
    before = (f.line, f.lineEnd)
    assert python.apply_py_spans(B, scopes) == 1
    assert (f.line, f.lineEnd) == before


# --- imports --------------------------------------------------------------------------------------------------------


IMPORT_SRC = textwrap.dedent('''\
    import os, sys
    import a.b as c
    from . import sibling
    from .mod import thing as t, thing, other, thing as t
    from ..up import u
    from pkg import deep
    from lib.q import *


    def f():
        from .mod import late
''')
FILES = {
    "svc/pkg/__init__.py": "",
    "svc/pkg/a.py": IMPORT_SRC,
    "svc/pkg/sibling.py": "",
    "svc/pkg/mod.py": "",
    "svc/up.py": "",
    "svc/pkg/deep/__init__.py": "",
}


def imports(tmp_path, file="svc/pkg/a.py", files=None, root="R"):
    B = builder(tmp_path, files or FILES, root=root)
    return B, python.py_imports(B, file)


def test_imports_are_listed_by_position_with_one_record_per_alias_free_name(tmp_path):
    _, recs = imports(tmp_path)
    got = [(r["line"], r["col"], r["module"], r["names"], r["resolvedPath"]) for r in recs]
    assert got[:2] == [(1, 0, "os", [], None), (1, 0, "sys", [], None)]      # `import a, b`: two records, in written order
    assert got[2] == (2, 0, "a.b", [], None)                                 # `as c` is dropped for a plain import
    assert [g[2] for g in got] == ["os", "sys", "a.b", ".", ".mod", "..up", "pkg", "lib.q", ".mod"]
    assert [g[0] for g in got] == [1, 1, 2, 3, 4, 5, 6, 7, 11]               # the import inside `f` is listed, by line


def test_from_import_names_are_deduplicated_and_sorted(tmp_path):
    _, recs = imports(tmp_path)
    (r,) = [r for r in recs if r["line"] == 4]
    assert r["names"] == [{"name": "other"}, {"name": "thing"}, {"name": "thing", "alias": "t"}]


def test_relative_and_absolute_imports_resolve_to_files_of_the_unit_under_the_unit_root(tmp_path):
    _, recs = imports(tmp_path)
    by = {(r["line"], r["module"]): r for r in recs}
    assert by[(3, ".")]["resolvedPath"] == "R/svc/pkg/__init__.py"
    assert by[(3, ".")]["_subs"] == {"sibling": "R/svc/pkg/sibling.py"}      # a submodule import
    assert by[(4, ".mod")]["resolvedPath"] == "R/svc/pkg/mod.py"
    assert by[(5, "..up")]["resolvedPath"] == "R/svc/up.py"
    assert by[(6, "pkg")]["resolvedPath"] is None                            # absolute: resolved from the unit's own dir, not found here
    assert by[(7, "lib.q")]["resolvedPath"] is None and by[(7, "lib.q")]["names"] == [{"name": "*"}]


def test_a_file_with_a_syntax_error_has_no_imports(tmp_path):
    B = builder(tmp_path, {"a.py": "x = 1\n"})
    (tmp_path / "a.py").write_text("import os\ndef (:\n")
    assert python.py_imports(B, "a.py") == []


def test_py_resolve_prefers_a_module_file_over_a_package_and_finds_package_submodules(tmp_path):
    B = builder(tmp_path, {"m.py": "", "m/__init__.py": "", "q/__init__.py": "", "q/s/__init__.py": "", "q/t.py": "", "q/u.py": "",
                           "q/u/__init__.py": "", "q/v/__init__.py": "", "a.py": ""})
    assert python.py_resolve(B, "a.py", ".m", []) == ("m.py", {})
    assert python.py_resolve(B, "a.py", ".q", ["t", "s", "none"]) == ("q/__init__.py", {"t": "q/t.py", "s": "q/s/__init__.py"})
    assert python.py_resolve(B, "a.py", ".nothing", []) == (None, {})
    assert python.py_resolve(B, "a.py", ".q", ["u"])[1] == {"u": "q/u.py"}      # a module file beats a same-named package
    assert python.py_resolve(B, "a.py", ".q", ["v"])[1] == {"v": "q/v/__init__.py"}


# --- determinism: the facts' order is not part of the input --------------------------------------------------------


def describe(scopes):
    return sorted((q, v[0], v[1]) for q, v in summary(scopes).items())


@pytest.mark.parametrize("seed", range(8))
def test_scopes_are_the_same_for_any_order_of_the_raw_facts(tmp_path, seed):
    """P1-04 gate: Joern's record order varies between runs, so the normalizer must not depend on it."""
    _, plain, _, _ = scopes_of(tmp_path / "plain")
    _, shuffled, _, _ = scopes_of(tmp_path / "shuf", shuffle=seed)
    assert describe(shuffled) == describe(plain)
    assert {q for q, *_ in describe(plain)} == set(WANT)


def test_imports_are_the_same_for_any_order_of_the_facts(tmp_path):
    # imports are read from the source, so the facts' order is irrelevant by construction; pinned as a contract
    a = imports(tmp_path / "a")[1]
    b = python.py_imports(builder(tmp_path / "b", FILES, root="R", shuffle=3), "svc/pkg/a.py")
    assert a == b


def test_the_shuffle_really_reorders_the_facts():
    from normalize_helpers import py_facts
    assert [m["id"] for m in py_facts({"a.py": SRC})["m"]] != [m["id"] for m in py_facts({"a.py": SRC}, shuffle=1)["m"]]


def test_async_functions_and_classes_nested_in_functions(tmp_path):
    src = "async def a():\n    pass\n\n\ndef f():\n    class L:\n        def g(self):\n            pass\n"
    B, scopes, _, _ = scopes_of(tmp_path, {"n.py": src})
    python.apply_py_spans(B, scopes)
    got = {q: v for q, v in summary(scopes).items()}
    assert got == {
        "n.py::<module>": ("module", None, 1, 8),
        "n.py::a": ("function", "n.py::<module>", 1, 2),
        "n.py::f": ("function", "n.py::<module>", 5, 8),
        "n.py::f.L": ("type", "n.py::f", 6, 8),
        "n.py::f.L.g": ("method", "n.py::f.L", 7, 8),
    }


def test_a_scope_with_a_decorator_that_is_not_a_plain_name_keeps_the_cpg_name(tmp_path):
    src = "@[d][0]\ndef f(): pass\n"
    B, scopes, _, _ = scopes_of(tmp_path, {"n.py": src})
    f = next(s for s in scopes.values() if s.base == "f")
    assert [n for n, _ in f.decos] == ["cpg-name"]


# --- parent attachment: the CPG's parent kinds ---------------------------------------------------------------------
# In the slice's facts every symbol's parent is reached through the first branch (an enclosing METHOD); the other
# branches are the slice's own, ported unchanged, and pinned here with hand-built facts.


PARENT_SRC = "class K:\n    pass\n\n\ndef g():\n    pass\n"


def with_extra_method(tmp_path, kw):
    B = builder(tmp_path, {"a.py": PARENT_SRC})
    by = {m["name"]: m for m in B.facts["m"]}
    t_of = {t["name"]: t for t in B.facts["t"]}
    B.facts["m"].append({"k": "m", "id": "7001", "name": "h", "full": "a.py:<module>.h", "file": "a.py", "l": 8, "le": 9,
                         "c": 1, "pfull": "", "code": "", **kw(by, t_of, B)})
    scopes, _, _ = python.build_scopes(B)
    return {s.qname: (s.parent.qname if s.parent else None) for s in scopes.values()}


def test_a_method_whose_cpg_parent_is_a_type_decl_hangs_under_that_class(tmp_path):
    got = with_extra_method(tmp_path, lambda by, t, B: {"pk": "TYPE_DECL", "pid": t["K"]["id"]})
    assert got["a.py::K.h"] == "a.py::K"


def test_a_method_in_a_functions_type_decl_hangs_under_the_function_of_the_same_full_name(tmp_path):
    def extra(by, t, B):
        B.facts["t"].append({"k": "t", "id": "8001", "name": "g", "full": "a.py:<module>.g", "file": "a.py", "l": 5, "c": 1,
                             "pk": "METHOD", "pid": by["<module>"]["id"], "pfull": ""})
        return {"pk": "TYPE_DECL", "pid": "8001"}
    assert with_extra_method(tmp_path, extra)["a.py::g.h"] == "a.py::g"


def test_a_method_nested_in_an_unlisted_synthetic_method_hangs_under_that_methods_class(tmp_path):
    def extra(by, t, B):
        B.facts["m"].append({"k": "m", "id": "7002", "name": "<fakeNew>", "full": "a.py:<module>.K.<fakeNew>2", "file": "a.py",
                             "l": 1, "le": 1, "c": 1, "pk": "TYPE_DECL", "pid": t["K"]["id"], "pfull": "", "code": ""})
        return {"pk": "METHOD", "pid": "7002"}
    assert with_extra_method(tmp_path, extra)["a.py::K.h"] == "a.py::K"


def test_a_method_with_no_resolvable_parent_hangs_under_its_files_module(tmp_path):
    got = with_extra_method(tmp_path, lambda by, t, B: {"pk": "METHOD", "pid": "no-such-id"})
    assert got["a.py::h"] == "a.py::<module>"


def test_a_body_method_without_a_class_type_is_not_a_symbol(tmp_path):
    B = builder(tmp_path, {"a.py": "def f(): pass\n"})
    B.facts["m"].append({"k": "m", "id": "7003", "name": "<body>", "full": "a.py:<module>.Ghost.<body>", "file": "a.py", "l": 1,
                         "le": 1, "c": 1, "pk": "TYPE_DECL", "pid": "none", "pfull": "", "code": ""})
    scopes, _, _ = python.build_scopes(B)
    assert sorted(s.qname for s in scopes.values()) == ["a.py::<module>", "a.py::f"]


# --- a class defined twice in one scope (P1-05 gate F1) -------------------------------------------------------------
# Live pysrc2cpg: the second `class B` keeps the first's fullName, and so do the members of both (a redefined
# function, by contrast, is suffixed `$redefinitionN`). Scopes must not depend on which record came last.

REDEF_SRC = textwrap.dedent('''\
    class B:
        @staticmethod
        def m(a):
            pass


    class B:
        def m(self):
            pass

        def n(self):
            pass
''')
REDEF_WANT = {
    "r.py::<module>": ("module", None, 1, 12),
    "r.py::B": ("type", "r.py::<module>", 1, 4),
    "r.py::B.m": ("method", "r.py::B", 2, 4),
    "r.py::B#2": ("type", "r.py::<module>", 7, 12),
    "r.py::B#2.m": ("method", "r.py::B#2", 8, 9),
    "r.py::B#2.n": ("method", "r.py::B#2", 11, 12),
}


def redef(tmp_path, **kw):
    B, scopes, _, _ = scopes_of(tmp_path, {"r.py": REDEF_SRC}, **kw)
    python.apply_py_spans(B, scopes)
    return B, scopes


def test_the_redefined_class_fixture_has_the_real_shape(tmp_path):
    B, _ = redef(tmp_path)
    bodies = [m for m in B.facts["m"] if m["name"] == "<body>"]
    ms = [m for m in B.facts["m"] if m["name"] == "m"]
    assert len(bodies) == 2 and bodies[0]["full"] == bodies[1]["full"]       # same fullName, as pysrc2cpg gives
    assert len(ms) == 2 and ms[0]["full"] == ms[1]["full"]


def test_a_class_defined_twice_in_one_scope_is_two_symbols_with_their_own_spans_and_members(tmp_path):
    _, scopes = redef(tmp_path)
    assert summary(scopes) == REDEF_WANT


def test_a_decorator_belongs_to_the_function_it_precedes_even_when_two_share_a_full_name(tmp_path):
    _, scopes = redef(tmp_path)
    by = {s.qname: s for s in scopes.values()}
    assert [n for n, _ in by["r.py::B.m"].decos] == ["staticmethod"]
    assert by["r.py::B#2.m"].decos == []


@pytest.mark.parametrize("seed", range(200))
def test_a_redefined_class_builds_the_same_scopes_for_every_order_of_the_facts(tmp_path, seed):
    _, scopes = redef(tmp_path, shuffle=seed)
    assert summary(scopes) == REDEF_WANT
    by = {s.qname: s for s in scopes.values()}
    assert [n for n, _ in by["r.py::B.m"].decos] == ["staticmethod"] and by["r.py::B#2.m"].decos == []


# --- submodule paths are normalised like the module path (gate F2) --------------------------------------------------


@pytest.mark.parametrize("root, prefix", [(".", ""), ("R", "R/")])
def test_a_bare_dot_import_at_the_unit_root_resolves_every_path_without_a_dot_segment(tmp_path, root, prefix):
    files = {"a.py": "from . import sibling\nfrom .. import up\n", "__init__.py": "", "sibling.py": "", "up.py": ""}
    B = builder(tmp_path, files, root=root)
    one, two = python.py_imports(B, "a.py")
    assert (one["resolvedPath"], one["_subs"]) == (prefix + "__init__.py", {"sibling": prefix + "sibling.py"})
    assert (two["resolvedPath"], two["_subs"]) == (prefix + "__init__.py", {"up": prefix + "up.py"})


def test_the_submodule_reference_a_value_set_builds_from_it_is_a_real_path(tmp_path):
    B = builder(tmp_path, {"a.py": "from . import sibling\n", "sibling.py": ""}, root="R")
    (rec,) = python.py_imports(B, "a.py")
    R = core.Resolver(B, {}, {})
    R.alias["a.py"]["sibling"] = (rec["module"], "sibling", rec["_subs"]["sibling"], rec["resolvedPath"])
    assert R.element({"o": "attr", "b": {"o": "name", "n": "sibling"}, "f": "X"}, "a.py") == {"ref": "R/sibling.py::X"}


# --- the remaining gaps of the gate (F3) ----------------------------------------------------------------------------


def test_two_different_aliases_of_one_name_are_two_entries(tmp_path):
    B = builder(tmp_path, {"a.py": "from m import a as b, a as c, a\n"})
    (rec,) = python.py_imports(B, "a.py")
    assert rec["names"] == [{"name": "a"}, {"name": "a", "alias": "b"}, {"name": "a", "alias": "c"}]


def test_a_decorator_on_the_first_line_of_a_file_is_read_from_the_source(tmp_path):
    _, scopes, _, _ = scopes_of(tmp_path, {"d.py": "@dec.x(1)\ndef f(): pass\n"})
    f = next(s for s in scopes.values() if s.base == "f")
    assert [n for n, _ in f.decos] == ["dec.x"] and f.decos[0][1]["l"] == 1


def test_the_orphan_fallback_picks_the_module_of_the_orphans_own_file(tmp_path):
    B = builder(tmp_path, {"a.py": "def f(): pass\n", "b.py": "def g(): pass\n"})
    B.facts["m"].append({"k": "m", "id": "7100", "name": "h", "full": "b.py:<module>.h", "file": "b.py", "l": 3, "le": 3,
                         "c": 1, "pk": "METHOD", "pid": "no-such-id", "pfull": "", "code": ""})
    scopes, _, _ = python.build_scopes(B)
    assert {s.qname: s.parent.qname for s in scopes.values() if s.parent}["b.py::h"] == "b.py::<module>"


def test_the_orphan_fallback_is_counted_in_the_report(tmp_path):
    B = builder(tmp_path, {"a.py": "def f(): pass\n"})
    python.build_scopes(B)
    assert B.report["pythonParentFallbacks"] == 0
    for i in (1, 2):
        B.facts["m"].append({"k": "m", "id": f"710{i}", "name": f"h{i}", "full": f"a.py:<module>.h{i}", "file": "a.py", "l": 3,
                             "le": 3, "c": 1, "pk": "METHOD", "pid": "none", "pfull": "", "code": ""})
    python.build_scopes(B)
    assert B.report["pythonParentFallbacks"] == 2


def test_a_depth_one_to_three_template_nests_and_a_fourth_level_is_capped(tmp_path):
    B = core.Builder("python", ".", "r", {}, core.Source(tmp_path))
    R = core.Resolver(B, {}, {})
    e = {"o": "lit", "c": "'a'"}
    for _ in range(3):
        e = {"o": "fmt", "p": [e]}
    assert R.element(e, "a.py") == {"template": [{"template": [{"template": [{"const": "a"}]}]}]}
    assert R.element({"o": "fmt", "p": [e]}, "a.py") == {"template": [{"template": [{"unknown": "value-set-cap"}]}]}
    c = {"o": "add", "p": [{"o": "add", "p": [{"o": "add", "p": [{"o": "add", "p": [{"o": "lit", "c": "'a'"}]}]}]}]}
    assert R.element(c, "a.py") == {"concat": [{"concat": [{"unknown": "value-set-cap"}]}]}


def test_a_decorator_is_owned_even_when_the_functions_cpg_line_is_the_decorators_own(tmp_path):
    B, _ = redef(tmp_path)
    for m in B.facts["m"]:
        if m["name"] == "m" and m["l"] == 3:
            m["l"] = 2                       # a CPG that reports the first decorator's line for the function
    scopes, _, _ = python.build_scopes(B)
    by = {s.qname: s for s in scopes.values()}
    assert [n for n, _ in by["r.py::B.m"].decos] == ["staticmethod"] and by["r.py::B#2.m"].decos == []


def test_the_owner_among_equal_fullnames_is_chosen_by_position_not_by_cpg_id(tmp_path):
    B, _ = redef(tmp_path)
    ms = [m for m in B.facts["m"] if m["name"] == "m"]
    first, second = sorted(ms, key=lambda m: m["l"])
    first["id"], second["id"] = "9999", "1"             # the later function has the smaller id
    scopes, _, _ = python.build_scopes(B)
    by = {s.qname: s for s in scopes.values()}
    assert [n for n, _ in by["r.py::B.m"].decos] == ["staticmethod"]


@pytest.mark.parametrize("seed", range(12))
def test_decorators_of_one_function_keep_their_written_order_for_every_fact_order(tmp_path, seed):
    src = "@a\n@b.c(1)\n@d\ndef f(): pass\n"
    _, scopes, _, _ = scopes_of(tmp_path, {"d.py": src}, shuffle=seed)
    f = next(s for s in scopes.values() if s.base == "f")
    assert [n for n, _ in f.decos] == ["a", "b.c", "d"]
