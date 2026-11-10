"""P1-16 `ccg.packrt.match`: matchers, scope gates and the decorator/callsite clauses (rule-pack spec §1.2, §1.3).

Plan: `docs/plans/connected-code-graph-p1-linking.md` § P1-16. Each case plants the record that separates the spec's
reading from its nearest alternative (the boundary a `prefix` stops at, the `*` a glob does not cross, an `any` that is
not an `all`).
"""
import pytest

from packrt_helpers import call, dec, file_, imp, sym, vs
from ccg.packrt.extract import Corpus
from ccg.packrt import match as M


# ---- matchers (§1.3)

def test_a_bare_scalar_is_exact_and_a_list_is_any_of():
    assert M.text_hit("a", "a") and not M.text_hit("a", "ab")
    assert M.text_hit(["x", {"prefix": "y"}], "y.z") and not M.text_hit(["x", "y"], "z")


@pytest.mark.parametrize("m,s,ok", [
    ({"exact": "uvicorn"}, "uvicorn", True),
    ({"basename": "uvicorn"}, "/venv/bin/uvicorn", True),
    ({"basename": "uvicorn"}, "uvicorn2", False),
    ({"basename": "uvicorn"}, "<redacted>", False),
    ({"prefix": "fastapi"}, "fastapi", True),
    ({"prefix": "fastapi"}, "fastapi.responses", True),       # a module name: boundary is `.`
    ({"prefix": "fastapi"}, "fastapix", False),
    ({"glob": "**FastMCP.streamable_http_app"}, "mcp/server/fastmcp.py:<module>.FastMCP.streamable_http_app", True),
    ({"glob": "*FastMCP.streamable_http_app"}, "mcp/server/fastmcp.py:<module>.FastMCP.streamable_http_app", False),  # * stops at /
    ({"glob": "a?c"}, "abc", True),
    ({"glob": "a?c"}, "ac", False),
    ({"glob": "a.c"}, "abc", False),                           # `.` is a literal in a glob, not a regex dot
])
def test_matcher_semantics(m, s, ok):
    assert bool(M.text_hit(m, s)) is ok


def test_regex_is_whole_string_and_case_sensitive_unless_asked():
    rx = {"regex": r"[a-z]+\.get"}
    assert M.text_hit(rx, "router.get") and not M.text_hit(rx, "xrouter.get2") and not M.text_hit(rx, "Router.get")
    assert M.text_hit({"regex": "router", "ignoreCase": True}, "ROUTER")


def test_regex_groups_are_reachable_for_a_group_extraction():
    hit = M.text_hit({"regex": r"[A-Za-z_]\w*\.(get|post)"}, "router.post")
    assert hit.group(1) == "post"


def test_path_matchers_normalize_both_operands_once_from_the_original_text():
    assert M.path_hit({"prefix": "/mcp"}, "/mcp/x") and M.path_hit({"prefix": "/mcp"}, "/mcp")
    assert not M.path_hit({"prefix": "/mcp"}, "/mcpx")
    assert not M.path_hit({"prefix": "/mcp"}, "/mcp/%2E%2E/admin")        # decodes to /admin
    assert M.path_hit({"prefix": "/mcp"}, "/mcp/./x")
    assert M.path_hit({"exact": "/a//b"}, "/a/b")
    assert M.path_hit({"template": "/u/{}"}, "/u/7") and not M.path_hit({"template": "/u/{}"}, "/u/")
    assert M.path_hit({"template": "/f/{**}"}, "/f/a/b") and not M.path_hit({"template": "/f/{**}"}, "/f")


# ---- scope (§1.2)

def corpus_with(*records):
    return Corpus([list(records)])


def test_an_imports_gate_is_satisfied_when_any_entry_is_and_names_narrow_it():
    c = corpus_with(file_("f1", "a.py"), imp("f1", "fastapi.routing", "APIRouter"), sym("h", "h"))
    gate = lambda *e, **k: M.scope_ok({"imports": list(e)}, "f1", c)
    assert gate({"module": {"prefix": "fastapi"}})
    assert gate({"module": {"prefix": "flask"}}, {"module": {"prefix": "fastapi"}})        # any-of
    assert not gate({"module": {"prefix": "flask"}})
    assert gate({"module": {"prefix": "fastapi"}, "names": ["APIRouter"]})
    assert not gate({"module": {"prefix": "fastapi"}, "names": ["Request"]})


def test_an_imports_gate_reads_only_the_matched_records_own_file():
    c = corpus_with(file_("f1", "a.py"), file_("f2", "b.py"), imp("f2", "fastapi"))
    assert not M.scope_ok({"imports": [{"module": {"prefix": "fastapi"}}]}, "f1", c)
    assert M.scope_ok({"imports": [{"module": {"prefix": "fastapi"}}]}, "f2", c)


def test_scope_path_notpath_and_role_gates():
    c = corpus_with(file_("f1", "svc/tests/t.py", role="primary"), file_("f2", "svc/x.py", role="example"))
    assert M.scope_ok({"path": {"prefix": "svc"}}, "f1", c)
    assert not M.scope_ok({"notPath": {"glob": "**/tests/**"}}, "f1", c)
    assert M.scope_ok({"notPath": {"glob": "**/tests/**"}}, "f2", c)
    assert M.scope_ok({"role": ["primary"]}, "f1", c) and not M.scope_ok({"role": ["primary"]}, "f2", c)


def test_no_scope_is_open():
    assert M.scope_ok(None, "anything", corpus_with())


# ---- clauses

def test_a_decorator_clause_tests_form_name_and_target_kind():
    c = corpus_with(file_("f1", "a.py"), sym("cls", "C", kind="type"), sym("h", "h", parent="cls", kind="method"),
                    dec("d", "h", "router.get", ("a0", "/x")))
    d = c.decorators[0]
    assert M.clause_hit("decorator", {"form": "decorator", "name": {"regex": r"\w+\.(get)"}}, d, c).group(1) == "get"
    assert not M.clause_hit("decorator", {"form": "annotation"}, d, c)
    assert not M.clause_hit("decorator", {"name": {"exact": "router.post"}}, d, c)
    assert M.clause_hit("decorator", {"target": {"kind": "method", "parentKind": "type"}}, d, c)
    assert not M.clause_hit("decorator", {"target": {"parentKind": "function"}}, d, c)
    assert M.clause_hit("decorator", {"target": {"name": "h"}}, d, c)


def test_args_entries_all_hold_and_a_value_needs_some_const_alternative():
    c = corpus_with(file_("f1", "a.py"), sym("h", "h"),
                    dec("d", "h", "r.get", ("a0", "/x", {"unknown": "point-derived"}), ("kw.tags", "t")))
    d = c.decorators[0]
    ok = lambda args: bool(M.clause_hit("decorator", {"args": args}, d, c))
    assert ok([{"arg": "a0"}])
    assert ok([{"arg": "a0", "value": {"exact": "/x"}}])
    assert not ok([{"arg": "a0", "value": {"exact": "/y"}}])
    assert not ok([{"arg": "a1"}])                                       # absent
    assert ok([{"arg": "a0"}, {"arg": "kw.tags", "value": "t"}])
    assert not ok([{"arg": "a0"}, {"arg": "kw.tags", "value": "u"}])      # every entry must hold


def test_a_callsite_clause_tests_callee_resolution_kind_and_external_target():
    c = corpus_with(file_("f1", "a.py"), sym("f", "f"),
                    call("c1", "f", "fetch", resolution="external", external="lib/x.js:<module>.fetch"),
                    call("c2", "f", "fetch", resolution="resolved", targets=("t",)),
                    vs("c1", "a0", "/x"))
    c1, c2 = c.callsites["c1"], c.callsites["c2"]
    cl = lambda clause, s: bool(M.clause_hit("callsite", clause, s, c))
    assert cl({"callee": "fetch"}, c1)
    assert cl({"callee": "fetch", "resolution": ["external", "unresolved"]}, c1)
    assert not cl({"callee": "fetch", "resolution": ["external", "unresolved"]}, c2)    # resolved is outside the list
    assert cl({"externalTarget": {"glob": "**fetch"}}, c1) and not cl({"externalTarget": {"glob": "**fetch"}}, c2)
    assert cl({"target": {"exact": "q::t"}}, c2) and not cl({"target": {"exact": "q::t"}}, c1)
    assert cl({"callKind": "call"}, c1) and not cl({"callKind": "new"}, c1)
    assert cl({"args": [{"arg": "a0", "value": "/x"}]}, c1) and not cl({"args": [{"arg": "a0"}]}, c2)


def test_an_unknown_subject_is_refused_not_ignored():
    with pytest.raises(M.UnsupportedSubject, match="query"):
        M.clause_hit("query", {"query": "q"}, {}, corpus_with())
