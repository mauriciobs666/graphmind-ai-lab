"""S11 unit tests -- the pure parts of the slice pipeline (no Joern, no sandbox).

Run from the component directory:  cd cpg && python -m pytest spikes/vertical-slice/tests
What the live pipeline proves (qnames of the two real units, determinism, validation) is recorded in
``cpg/spikes/vertical-slice/S11-notes.md``; these tests pin the rules those runs rely on, and each was
mutation-checked (a deliberately broken implementation turns it red).
"""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))   # cpg/

import normalize as N  # noqa: E402
import verify_qnames as V  # noqa: E402


# -- verify_qnames: Python rules (bundle spec §4.1, §4.2) ----------------------------------------------
def pyq(src, path="m.py"):
    return V.py_file(path, src)


def test_python_anonymous_scopes_are_numbered_from_one_in_source_order():
    q = pyq("f = lambda: 0\ng = lambda: 1\n")
    assert "m.py::<anon>#1" in q and "m.py::<anon>#2" in q
    assert "m.py::<anon>" not in q


def test_python_duplicates_get_hash_n_from_two():
    q = pyq("def f(): pass\ndef f(): pass\ndef f(): pass\n")
    assert {"m.py::f", "m.py::f#2", "m.py::f#3"} <= set(q)


def test_python_property_accessors_are_two_segments_and_do_not_collide():
    src = (
        "class C:\n"
        "    @property\n    def x(self): return 1\n"
        "    @x.setter\n    def x(self, v): pass\n"
        "    @x.deleter\n    def x(self): pass\n"
    )
    q = pyq(src)
    assert {"m.py::C.x.<get>", "m.py::C.x.<set>", "m.py::C.x.<del>"} <= set(q)
    assert not any(k.endswith("x#2.<set>") for k in q)


def test_python_decorator_and_default_lambdas_belong_to_the_enclosing_scope():
    src = (
        "def outer():\n"
        "    @deco(ids=lambda c: c)\n"
        "    def test_x(a=lambda: 0):\n"
        "        pass\n"
    )
    q = pyq(src)
    assert "m.py::outer.<anon>#1" in q and "m.py::outer.<anon>#2" in q
    assert "m.py::outer.test_x.<anon>#1" not in q


def test_python_overload_stubs_are_not_symbols_and_span_includes_decorators():
    src = (
        "from typing import overload\n"
        "@overload\ndef f(a: int) -> int: ...\n"
        "@overload\ndef f(a: str) -> str: ...\n"
        "@dec\n"
        "@dec2(\n    1)\n"
        "def f(a): return a\n"
    )
    q = pyq(src)
    assert q["m.py::f"] == (6, 9) and "m.py::f#2" not in q


def test_python_control_blocks_add_no_segment_and_top_level_has_no_module_prefix():
    q = pyq("if True:\n    def g(): pass\n")
    assert "m.py::g" in q and q["m.py::<module>"] == (1, 2)


# -- verify_qnames: TS rules over a hand-made Babel AST ---------------------------------------------------
def _fn(t, line, endline, **kw):
    n = {"type": t, "loc": {"start": {"line": line, "column": 0}}, "start": 0, "end": endline * 10}
    n.update(kw)
    return n


def test_js_binding_names_the_function_value_and_other_callbacks_are_anon():
    text = "\n".join(["x"] * 10) + "\n"
    arrow = _fn("ArrowFunctionExpression", 2, 3, body={})
    cb = _fn("ArrowFunctionExpression", 5, 6, body={})
    prog = {"program": {"type": "Program", "body": [
        {"type": "VariableDeclaration", "declarations": [
            {"type": "VariableDeclarator", "id": {"type": "Identifier", "name": "onAbort"}, "init": arrow}]},
        {"type": "CallExpression", "callee": {"type": "Identifier", "name": "setTimeout"}, "arguments": [cb]},
    ]}}
    q = V.js_file("a.ts", text, prog)
    assert "a.ts::onAbort" in q and "a.ts::<anon>#1" in q


def test_js_named_function_expression_takes_its_own_name_not_the_binding():
    text = "\n".join(["x"] * 10) + "\n"
    fe = _fn("FunctionExpression", 2, 3, id={"type": "Identifier", "name": "g"}, body={})
    prog = {"program": {"type": "Program", "body": [
        {"type": "VariableDeclaration", "declarations": [
            {"type": "VariableDeclarator", "id": {"type": "Identifier", "name": "f"}, "init": fe}]}]}}
    q = V.js_file("a.ts", text, prog)
    assert "a.ts::g" in q and "a.ts::f" not in q


# -- normalize.py helpers -------------------------------------------------------------------------
def test_field_quoting_is_ascii_ident_only():
    assert N.quote_field("method") == ".method"
    assert N.quote_field("café") == '["café"]'
    assert N.quote_field("content-type") == '["content-type"]'
    assert N.quote_field("a.b") == '["a.b"]'
    assert N.quote_field("") == '[""]'
    assert N.quote_field("$x_1") == ".$x_1"


def test_path_cap_is_three_fields_after_the_head_and_marks_truncation():
    assert N.path_from("a1", ["a", "b", "c"]) == ("a1.a.b.c", False)
    assert N.path_from("a1", ["a", "b", "c", "d"]) == ("a1.a.b.c", True)
    # the kw head's own field is not counted (-spec2 §1.1)
    assert N.path_from(N.kw_head("headers"), ["a", "b", "c"]) == ("kw.headers.a.b.c", False)
    assert N.path_from(N.kw_head("headers"), ["a", "b", "c", "d"])[1] is True


def test_name_norm_is_nfc_of_casefold_of_nfc():
    assert N.name_norm("  Café ") == "café"
    # U+01F0: casefold of the NFC string is not NFC; the outer NFC repairs it (-spec2 §1.2)
    out = N.name_norm("\u01f0")
    import unicodedata
    assert unicodedata.is_normalized("NFC", out) and out == "\u01f0"
    assert N.name_norm("ǅ") == "ǆ"
    assert N.name_norm("Straße") == "strasse"   # casefold, not lower


def test_g4_drops_self_edges_and_repeats_in_first_seen_order():
    kept, s, d = N.g4_filter([("p", "c"), ("c", "c"), ("p", "c"), ("q", "c"), ("c", "c")])
    assert kept == [("p", "c"), ("q", "c")] and (s, d) == (2, 1)


def test_g4_mutants_keep_what_they_name():
    edges = [("p", "c"), ("c", "c"), ("p", "c")]
    assert N.g4_filter(edges, "self")[0] == [("p", "c"), ("c", "c")]
    assert N.g4_filter(edges, "dedup")[0] == [("p", "c"), ("p", "c")]


def test_last_changed_takes_latest_committer_time_and_breaks_ties_by_lower_commit():
    per = {1: "b" * 40, 2: "a" * 40, 3: "c" * 40}
    times = {"a" * 40: 100, "b" * 40: 100, "c" * 40: 50}
    at, commit = N.last_changed(per, times, [(1, 3)])
    assert commit == "a" * 40 and at == "1970-01-01T00:01:40Z"
    with pytest.raises(ValueError):
        N.last_changed(per, times, [(10, 12)])


def test_chained_same_name_calls_are_ordered_inner_first():
    src = "x = a.foo().foo()\n"
    B = types.SimpleNamespace(col_base=1, src=types.SimpleNamespace(lines=lambda f: src.split("\n")))
    inner = {"id": "1", "name": "foo", "l": 1, "c": 5, "code": "a.foo()"}
    outer = {"id": "2", "name": "foo", "l": 1, "c": 5, "code": "a.foo().foo()"}
    pos = N.locate_callees(B, "f.py", [outer, inner])
    assert pos["1"][1] < pos["2"][1]


# -- compare_bundles (SL-6) --------------------------------------------------------------------------
def _write(p, header_extra, body=b'{"id":"1","type":"file"}\n'):
    h = {"type": "header", "x": 1}
    h.update(header_extra)
    p.write_bytes(json.dumps(h, sort_keys=True, separators=(",", ":")).encode() + b"\n" + body)


def run_cmp(a, b):
    return subprocess.run([sys.executable, str(HERE / "compare_bundles.py"), str(a), str(b)], capture_output=True)


def test_compare_ignores_exactly_the_two_exempt_header_fields(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, {"builtAt": "2026-01-01T00:00:00Z", "sandboxAuditIds": ["x"]})
    _write(b, {"builtAt": "2027-01-01T00:00:00Z", "sandboxAuditIds": ["y", "z"]})
    assert run_cmp(a, b).returncode == 0
    _write(b, {"builtAt": "2027-01-01T00:00:00Z", "sandboxAuditIds": ["y"], "x": 2})
    assert run_cmp(a, b).returncode == 1
    _write(b, {"builtAt": "2027-01-01T00:00:00Z", "sandboxAuditIds": ["y"]}, body=b'{"id":"2","type":"file"}\n')
    assert run_cmp(a, b).returncode == 1


def test_optional_call_and_generic_call_positions_are_found_and_synthetic_calls_are_not():
    src = "const a = onReset?.();\nconst b = apiFetch<T>(x);\nfor (const k of xs) {}\n"
    B = types.SimpleNamespace(col_base=0, src=types.SimpleNamespace(lines=lambda f: src.split("\n")))
    calls = [{"id": "1", "name": "onReset", "l": 1, "c": 10, "code": "onReset?.()"},
             {"id": "2", "name": "apiFetch", "l": 2, "c": 10, "code": "apiFetch<T>(x)"},
             {"id": "3", "name": "next", "l": 3, "c": 0, "code": "_iterator_0.next()"}]
    pos = N.locate_callees(B, "f.ts", calls)
    assert pos["1"] == (1, 10) and pos["2"] == (2, 10)
    assert "3" not in pos      # `next` never occurs in the source: a synthesized call, not a call site


def test_init_symbol_name_is_the_types_name():
    ty = types.SimpleNamespace(base="ApiError")
    ctor = types.SimpleNamespace(base="<init>", parent=ty)
    assert N.sym_name(ctor) == "ApiError"
    assert N.sym_name(types.SimpleNamespace(base="f", parent=ty)) == "f"


# -- assemble / projection on a synthetic facts fixture ---------------------------------------------------
# These drive the real ``assemble`` (and ``js_imports``) on a tiny repo built by the test, so the rules that
# only show up in the assembled records -- §1.5 import de-dup, cond->cond controlledBy and G-4, complexity,
# fingerprint blank-line rule, the empty-file lastChanged* fallback -- fail when broken (S11 review Pass 1,
# finding 2).  The live corpus holds none of the repeated-name / empty-file cases, so only this can pin them.
import hashlib  # noqa: E402
import os  # noqa: E402

MOD_PY = (
    "from m import b as a, a as b, a, a, a as b\n"   # 1  (source order unsorted, repeats)
    "import os, os\n"                                  # 2
    "\n"                                               # 3
    "\n"                                               # 4
    "def f(x):\n"                                      # 5
    "    if x:\n"                                      # 6   cond C1
    "        if x > 1:\n"                              # 7   cond C2, nested under C1
    "            return 1\n"                           # 8   ret R1
    "    return 2\n"                                   # 9   ret R2
    "\n"                                               # 10
    "\n"                                               # 11
    "def g():\n"                                       # 12
    "    y = 1\n"                                      # 13
    "\n"                                               # 14  blank line inside the span
    "        \n"                                       # 15  whitespace-only line inside the span
    "    return y\n"                                   # 16
)
T1, T2, T3 = "2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"


def _git(repo, *args, when=None):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@e")
    if when:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = when
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def _commit(repo, msg, when):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg, when=when)
    return _git(repo, "rev-parse", "HEAD")


def _m(i, name, full, file, l, le, pk="METHOD", pid="1"):
    return {"k": "m", "id": i, "name": name, "full": full, "file": file, "l": l, "le": le, "c": 1, "pk": pk,
            "pid": pid, "pfull": "", "code": ""}


FACTS = [
    {"k": "f", "name": "mod.py"}, {"k": "f", "name": "empty.py"},
    _m("1", "<module>", "mod.py:<module>", "mod.py", 1, 16, "NAMESPACE_BLOCK", "100"),
    _m("2", "f", "mod.py:<module>.f", "mod.py", 5, 9),
    _m("3", "g", "mod.py:<module>.g", "mod.py", 12, 16),
    _m("4", "<module>", "empty.py:<module>", "empty.py", 1, 1, "NAMESPACE_BLOCK", "101"),
    {"k": "c", "m": "2", "id": "11", "l": 6, "c": 8, "code": "x", "lab": "IDENTIFIER"},
    {"k": "c", "m": "2", "id": "12", "l": 7, "c": 12, "code": "x > 1", "lab": "CALL"},
    {"k": "r", "m": "2", "id": "21", "l": 8, "c": 13},
    {"k": "r", "m": "2", "id": "22", "l": 9, "c": 5},
    # (point, cond): R1->C2 twice (a duplicate pair), R1->C1, C2->C1 (cond-to-cond), C1->C1 (self-edge)
    {"k": "cb", "m": "2", "pk": "r", "pid": "21", "cond": "12"},
    {"k": "cb", "m": "2", "pk": "r", "pid": "21", "cond": "12"},
    {"k": "cb", "m": "2", "pk": "r", "pid": "21", "cond": "11"},
    {"k": "cb", "m": "2", "pk": "c", "pid": "12", "cond": "11"},
    {"k": "cb", "m": "2", "pk": "c", "pid": "11", "cond": "11"},
]


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    repo = tmp_path_factory.mktemp("repo")
    _git(repo, "init", "-q")
    unit = repo / "unit"          # the unit is a sub-directory: `git rev-parse <commit>:<root>` needs a real tree
    unit.mkdir()
    (unit / "mod.py").write_text(MOD_PY)
    (unit / "empty.py").write_text("x = 1\n")
    c1 = _commit(repo, "one", T1)
    (unit / "empty.py").write_text("")            # emptied: no blamed line is left at HEAD
    c2 = _commit(repo, "two", T2)
    (unit / "other.py").write_text("y = 2\n")     # a later commit, so the newest overall is not empty.py's
    c3 = _commit(repo, "three", T3)
    facts_path = tmp_path_factory.mktemp("facts") / "facts.jsonl"
    facts_path.write_text("".join(json.dumps(o) + "\n" for o in FACTS))
    script = facts_path.parent / "extract.sc"
    script.write_text("// stand-in\n")
    a = types.SimpleNamespace(lang="python", repo_id="r", repo=str(repo), commit=c3, stage=str(unit), unit_root="unit",
                              astgen=None, audit_ids="00000000-0000-4000-8000-000000000000",
                              extract_script=str(script), g4_mutant=None)
    records, report = N.assemble(a, N.load_facts(facts_path))
    return types.SimpleNamespace(records=records, report=report, c1=c1, c2=c2, c3=c3)


def recs(asm, typ):
    return [r for r in asm.records if r["type"] == typ]


def sym(asm, qname):
    return next(r for r in recs(asm, "symbol") if r["qname"] == qname)


def test_import_names_repeat_is_written_once_but_alias_forms_are_distinct(assembled):
    # spec3 §1.5: `from m import b as a, a as b, a, a, a as b` -> a, a as b, b as a (sorted, exact repeats gone)
    im = [r for r in recs(assembled, "import") if r["module"] == "m"]
    assert len(im) == 1
    assert im[0]["names"] == [{"name": "a"}, {"name": "a", "alias": "b"}, {"name": "b", "alias": "a"}]


def test_import_ordinals_are_dense_in_source_order(assembled):
    im = sorted(recs(assembled, "import"), key=lambda r: r["ordinal"])
    assert [r["ordinal"] for r in im] == list(range(len(im)))
    assert [r["line"] for r in im] == sorted(r["line"] for r in im)


def test_js_import_repeated_specifiers_are_written_once():
    def spec(kind, imported=None, local=None):
        d = {"type": kind, "local": {"name": local}}
        if imported:
            d["imported"] = {"name": imported}
        return d
    ast_ = {"type": "Program", "body": [{
        "type": "ImportDeclaration", "source": {"value": "pkg"}, "loc": {"start": {"line": 1, "column": 0}},
        "specifiers": [spec("ImportSpecifier", "a", "a"), spec("ImportSpecifier", "a", "a"),
                       spec("ImportSpecifier", "a", "b"), spec("ImportSpecifier", "a", "b"),
                       spec("ImportDefaultSpecifier", local="d"), spec("ImportDefaultSpecifier", local="d"),
                       spec("ImportNamespaceSpecifier", local="ns"), spec("ImportNamespaceSpecifier", local="ns")]}]}
    B = types.SimpleNamespace(src=None, P=lambda f: f, root=".")
    (r,) = N.js_imports(B, "x.ts", ast_)
    assert r["names"] == [{"name": "*", "alias": "ns"}, {"name": "a"}, {"name": "a", "alias": "b"},
                          {"name": "default", "alias": "d"}]


def test_cond_nested_under_cond_gets_a_cond_to_cond_controlled_by(assembled):
    conds = {c["ordinal"]: c["id"] for c in recs(assembled, "cond")}
    assert set(conds) == {0, 1}                        # C1 (line 6) = 0, C2 (line 7) = 1
    edges = {(e["point"], e["cond"]) for e in recs(assembled, "controlledBy")}
    assert (conds[1], conds[0]) in edges               # the cond-to-cond edge: C2 is controlled by C1


def test_g4_self_edge_and_duplicate_pair_are_dropped_the_rest_written(assembled):
    conds = {c["ordinal"]: c["id"] for c in recs(assembled, "cond")}
    ret0 = next(r["id"] for r in recs(assembled, "ret") if r["ordinal"] == 0)
    edges = [(e["point"], e["cond"]) for e in recs(assembled, "controlledBy")]
    assert sorted(edges) == sorted({(conds[1], conds[0]), (ret0, conds[1]), (ret0, conds[0])})
    assert all(p != c for p, c in edges) and len(edges) == len(set(edges))
    assert assembled.report["controlledBy"] == {"rawEdges": 5, "selfEdgesDropped": 1, "duplicatePairsDropped": 1,
                                                "unmapped": 0, "written": 3, "condToCond": 1}


def test_complexity_is_one_plus_the_symbols_own_cond_records(assembled):
    # As implemented: complexity = 1 + count(own `cond` records).  NOTE spec3 §5.3 says McCabe over the CFG's
    # decision points + 1; whether it should be that is an OPEN ruling (SL-7 candidate, S11 review finding 3) --
    # this test pins the code's definition, not a settled spec ruling.
    assert sym(assembled, "unit/mod.py::f")["complexity"] == 3      # two conds
    assert sym(assembled, "unit/mod.py::g")["complexity"] == 1      # none
    assert sym(assembled, "unit/mod.py::<module>")["complexity"] == 1


def test_fingerprint_drops_blank_and_whitespace_only_lines_and_strips_each_line(assembled):
    # spec3 §5.3: sha256 of the span's non-blank lines, each stripped, joined with \n
    want = hashlib.sha256(b"def g():\ny = 1\nreturn y").hexdigest()
    assert sym(assembled, "unit/mod.py::g")["fingerprint"] == want
    assert sym(assembled, "unit/mod.py::g")["lineEnd"] == 16


def test_empty_file_last_changed_is_the_newest_commit_touching_the_path(assembled):
    # an empty file's <module> has no blamed line: fall back to `git log -1` of the path -- the NEWEST commit
    # touching it (c2, which emptied it), not the oldest (c1, which created it) nor the repo HEAD (c3)
    s = sym(assembled, "unit/empty.py::<module>")
    assert (s["lastChangedAt"], s["lastChangedCommit"]) == (T2, assembled.c2)
    assert assembled.report["blameFallbackFiles"] == ["unit/empty.py"]


def test_non_empty_symbol_last_changed_comes_from_blame(assembled):
    s = sym(assembled, "unit/mod.py::g")
    assert (s["lastChangedAt"], s["lastChangedCommit"]) == (T1, assembled.c1)
