"""P1-05 5d (the JS/TS half of the type rule) — `assemble(lang="javascript")` and `type_backed("javascript", ...)` over hand-written facts.

The values are what `jssrc2cpg` 4.0.579 writes, per the 5d probe (`tests/test_normalize_name_linked.py`, end of file): `this.m()` in a class method has `rt`
`a.ts::program:C` and the target `a.ts::program:C:m` (typed: the separator is `:`); `p: any` gives `rt` `ANY` while Joern still links the class
method; a bare parameter is `METHOD_PARAMETER_IN`; a function, a class or an imported name is `LOCAL`. Expected outcomes are written by hand,
one per cell of the axes {member typed to the base, `ANY` base, module-typed `this`} x {bare parameter, bare `LOCAL`}; nothing is computed by
the code under test.
"""
from __future__ import annotations

import pytest

from extractors.joern import run as joern_run
from extractors.joern.normalize import assemble_bundle
from normalize_assembly_helpers import git_repo
from test_normalize_javascript import facts_of, m, t

UUID = "11111111-1111-4111-8111-111111111111"
BUILT_AT = "2026-01-01T00:00:00Z"
SRC = """\
class C {
  constructor(a) {}
  m() { return 1; }
  run() { this.m(); }
}
function f(p, cb, k) {
  p.m();
  cb(1);
  k.m();
  mod.m();
  local(2);
}
function local(x) { return x; }
function cb(x) { return x; }
"""
CM, LOCAL, CB = "a.ts::program:C:m", "a.ts::program:local", "a.ts::program:cb"
C_PROG = "a.ts::program:C"


def facts():
    return facts_of(
        f=[{"k": "f", "name": "a.ts"}],
        m=[m("1", ":program", "a.ts::program", 1, "TYPE_DECL", "t0", le=14),
           m("2", "<init>", f"{C_PROG}:<init>", 2, "TYPE_DECL", "t1", le=2),
           m("3", "m", CM, 3, "TYPE_DECL", "t1"),
           m("4", "run", f"{C_PROG}:run", 4, "TYPE_DECL", "t1"),
           m("6", "f", "a.ts::program:f", 6, "METHOD", "1", le=12),
           m("7", "local", LOCAL, 13, "METHOD", "1"),
           m("8", "cb", CB, 14, "METHOD", "1")],
        t=[t("t0", ":program", "a.ts::program", 1, "NAMESPACE_BLOCK"), t("t1", "C", C_PROG, 1)])


def cs(i, enc, name, line, col, rt, ins):
    return {"k": "cs", "m": enc, "id": str(i), "name": name, "mfn": "", "l": line, "c": col, "code": f"{name}()", "dt": "DYNAMIC_DISPATCH",
            "rt": rt, "in": [{"id": mid, "full": full} for mid, full in ins], "out": [], "args": []}


ROWS = {
    # marker: (call fact, expected (status, reason, bundle targets, dropped into nameLinked))
    "this-typed-to-the-class": (cs(9001, "4", "m", 4, 10, C_PROG, [("3", CM)]), ("confirmed", "resolved-call", ["a.ts::C.m"], [])),
    "any-base": (cs(9002, "6", "m", 7, 4, "ANY", [("3", CM)]), ("possible", "name-match", [], ["a.ts::C.m"])),
    "bare-parameter": (cs(9003, "6", "cb", 8, 2, "METHOD_PARAMETER_IN", [("8", CB)]), ("possible", "name-match", [], ["a.ts::cb"])),
    "member-typed-to-the-base": (cs(9004, "6", "m", 9, 4, C_PROG, [("3", CM)]), ("confirmed", "resolved-call", ["a.ts::C.m"], [])),
    "module-typed-this": (cs(9005, "6", "m", 10, 6, "a.ts::program", [("3", CM)]), ("possible", "name-match", [], ["a.ts::C.m"])),
    "bare-local": (cs(9006, "6", "local", 11, 2, "LOCAL", [("7", LOCAL)]), ("confirmed", "resolved-call", ["a.ts::local"], [])),
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("js") / "repo"
    commit = git_repo(path, {"a.ts": SRC})
    fx = facts()
    fx["cs"] = [row for row, _ in ROWS.values()]
    records, report = assemble_bundle.assemble(lang="javascript", unit_root=".", repo_id="r", facts=fx, stage=path, astgen_dir=None, repo=path,
                                               commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)
    return {r["line"]: r for r in records if r["type"] == "callsite"}, report


@pytest.mark.parametrize("marker", sorted(ROWS))
def test_javascript_row(built, marker):
    by_line, report = built
    row, want = ROWS[marker]
    rec = by_line[row["l"]]
    dropped = sorted(x["qname"] for x in report["nameLinked"].get(rec["id"], []))
    assert (rec["status"], rec["reason"], sorted(x["qname"] for x in rec["targets"]), dropped) == want


def test_the_javascript_self_rule_is_not_applied(built):
    assert built[1]["selfResolvedSites"] == [] and built[1]["nameLinkCounts"] == {"nameLinkedDemoted": 3, "selfResolved": 0}


@pytest.mark.parametrize("rt, target, want", [
    (C_PROG, CM, True),                                   # `k.m()` / `this.m()`: the base's type + ":" + the member
    (C_PROG, "a.ts::program:C.m", False),                 # the python separator is not javascript's
    ("a.ts::program", CM, False), ("ANY", CM, False),
    ("LOCAL", CM, True), ("METHOD_PARAMETER_IN", CM, False), (None, CM, True),
])
def test_type_backed_over_javascript_facts(rt, target, want):
    c = {"name": "m", **({} if rt is None else {"rt": rt})}
    assert assemble_bundle.type_backed("javascript", c, target) is want
