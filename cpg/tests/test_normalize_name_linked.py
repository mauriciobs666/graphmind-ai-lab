"""P1-05 step 5d — name-linked calls are not `confirmed` (`extractors/joern/normalize/assemble_bundle.py:assemble`).

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § 5d. Joern's `NaiveCallLinker` links a DYNAMIC_DISPATCH call to every
method of that name; a call's targets are `confirmed`/`resolved-call` only if type-backed. `extract.sc` tags each such call with
`rt` (member call: the base's `typeFullName`; bare-identifier call: the label of the declaration the identifier refers to).
The rows below are the probe table (a live `pysrc2cpg`/`jssrc2cpg` probe, Joern 4.0.579, recorded in the 5d probe note at the
end of this file) as hand-written synthetic facts: `rt`, the callee's `in` fullNames and the expected outcome of each row are
written here by hand; nothing is computed with the code under test. A dropped Joern target goes to `report["nameLinked"]`.
"""
from __future__ import annotations

import ast

import pytest

from extractors.joern import run as joern_run
from extractors.joern.normalize import assemble_bundle
from normalize_assembly_helpers import git_repo
from normalize_helpers import py_facts

UUID = "11111111-1111-4111-8111-111111111111"
BUILT_AT = "2026-01-01T00:00:00Z"
DYN = "DYNAMIC_DISPATCH"
MOD = "r.py:<module>"


def of(records, kind):
    return [r for r in records if r["type"] == kind]


def method(facts, dotted):
    """The method fact of ``dotted`` (``A.m2``, ``run.inner``); ``K.<body>`` is the class body (a constructor's callee)."""
    want = f"{MOD}.{dotted}"
    for m in facts["m"]:
        if m["full"] == want or m["full"].replace(".<body>.", ".") == want:
            return m
    raise KeyError(dotted)


def add_params(facts, src):
    """One ``p`` fact per source parameter (the receiver of a method included), in source order."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            m = next(m for m in facts["m"] if m["name"] == node.name and m["l"] == node.lineno)
            for k, a in enumerate(node.args.args):
                facts["p"].append({"k": "p", "m": m["id"], "i": k, "name": a.arg, "l": a.lineno, "c": a.col_offset + 1, "code": a.arg})


def call(marker, name, rt, ins, *, enc, recv=None, dt=DYN):
    """A call row: the line carrying ``# marker``; ``ins``: ``[(dotted method, joern fullName)]``; ``recv``: the receiver name (arg 0)."""
    return {"marker": marker, "name": name, "rt": rt, "ins": ins, "enc": enc, "recv": recv, "dt": dt}


def build(tmp_path, src, rows):
    """(records by marker line -> callsite, report) of ``src`` with one cs fact per row."""
    path = tmp_path / "repo"
    commit = git_repo(path, {"r.py": src})
    facts = py_facts({"r.py": src})
    add_params(facts, src)
    lines = src.splitlines()
    for n, r in enumerate(rows):
        ln = next(i for i, t in enumerate(lines, start=1) if f"# {r['marker']}" in t)
        f = method(facts, r["enc"])
        cs = {"k": "cs", "m": f["id"], "id": str(9000 + n), "name": r["name"], "mfn": "", "l": ln, "c": 1, "code": f"{r['name']}()",
              "dt": r["dt"], "in": [{"id": method(facts, d)["id"], "full": full} for d, full in r["ins"]], "out": [],
              "args": [{"i": 0, "an": "", "f": [], "e": r["recv"] if isinstance(r["recv"], dict) else {"o": "name", "n": r["recv"]}}]
              if r["recv"] else []}
        if r["rt"] is not None:
            cs["rt"] = r["rt"]
        facts["cs"].append(cs)
    records, report = assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None,
                                               repo=path, commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC,
                                               built_at=BUILT_AT)
    return {r["line"]: r for r in of(records, "callsite")}, report, lines


def view(by_line, report, lines, marker):
    ln = next(i for i, t in enumerate(lines, start=1) if f"# {marker}" in t)
    rec = by_line[ln]
    dropped = sorted(t["qname"] for t in report.get("nameLinked", {}).get(rec["id"], []))
    return rec["status"], rec["reason"], sorted(t["qname"] for t in rec["targets"]), dropped


# --- the probe table: member and bare-identifier calls --------------------------------------------------------------
SRC = '''\
def uniq(x):
    return x


def dup():
    return 0


def local(x):
    return x


def cb(x):
    return x


class K:
    def __init__(self):
        pass

    def meth(self):
        return 1


class A:
    def dup(self):
        return 1

    def m3(self):
        return 3


class B(A):
    pass


def run(p, cb2, cb):
    p.uniq(2)  # untyped-unique
    p.dup()  # untyped-ambiguous
    k.meth()  # typed-base
    mod.local(3)  # module-base
    bb.m3()  # inherited-on-subclass
    x.dup()  # typed-with-extra
    cb(1)  # param-cb
    local(1)  # bare-module-function
    uniq(1)  # bare-imported
    sh2(1)  # shadowing-local
    K()  # bare-constructor
    mod.K()  # module-attr-constructor
    old.uniq(2)  # no-rt-old-facts
    cb2(2)  # param-cb2
    static.local(1)  # static-dispatch


def outer():
    def inner():
        return 1
    inner()  # bare-nested-def
'''

K_MEM = f"{MOD}.K.meth"
ROWS = [
    call("untyped-unique", "uniq", "ANY", [("uniq", f"{MOD}.uniq")], enc="run"),
    call("untyped-ambiguous", "dup", "ANY", [("dup", f"{MOD}.dup"), ("A.dup", f"{MOD}.A.dup")], enc="run"),
    call("typed-base", "meth", f"{MOD}.K", [("K.meth", K_MEM)], enc="run"),
    call("module-base", "local", MOD, [("local", f"{MOD}.local")], enc="run"),
    call("inherited-on-subclass", "m3", f"{MOD}.B", [("A.m3", f"{MOD}.A.m3")], enc="run"),
    call("typed-with-extra", "dup", f"{MOD}.A", [("A.dup", f"{MOD}.A.dup"), ("dup", f"{MOD}.dup")], enc="run"),
    call("param-cb", "cb", "METHOD_PARAMETER_IN", [("cb", f"{MOD}.cb")], enc="run"),
    call("bare-module-function", "local", "LOCAL", [("local", f"{MOD}.local")], enc="run"),
    call("bare-imported", "uniq", "LOCAL", [("uniq", f"{MOD}.uniq")], enc="run"),
    call("shadowing-local", "sh2", "LOCAL", [("local", f"{MOD}.local")], enc="run"),
    call("bare-constructor", "K", "LOCAL", [("K.<body>", f"{MOD}.K.<body>")], enc="run"),
    call("module-attr-constructor", "K", MOD, [("K.<body>", f"{MOD}.K.<body>")], enc="run"),
    call("no-rt-old-facts", "uniq", None, [("uniq", f"{MOD}.uniq")], enc="run"),
    call("param-cb2", "cb2", "METHOD_PARAMETER_IN", [("cb", f"{MOD}.cb")], enc="run"),
    call("static-dispatch", "local", None, [("local", f"{MOD}.local")], enc="run", dt="STATIC_DISPATCH"),
    call("bare-nested-def", "inner", "LOCAL", [("outer.inner", f"{MOD}.outer.inner")], enc="outer"),
]
CONF, POSS = ("confirmed", "resolved-call"), ("possible", "name-match")
#: marker -> (status, reason, bundle targets, dropped into nameLinked)
WANT = {
    "untyped-unique": (*POSS, [], ["r.py::uniq"]),
    "untyped-ambiguous": (*POSS, [], ["r.py::A.dup", "r.py::dup"]),
    "typed-base": (*CONF, ["r.py::K.meth"], []),
    "module-base": (*CONF, ["r.py::local"], []),
    "inherited-on-subclass": (*POSS, [], ["r.py::A.m3"]),
    "typed-with-extra": (*CONF, ["r.py::A.dup"], ["r.py::dup"]),
    "param-cb": (*POSS, [], ["r.py::cb"]),
    "bare-module-function": (*CONF, ["r.py::local"], []),
    "bare-imported": (*CONF, ["r.py::uniq"], []),
    "shadowing-local": (*CONF, ["r.py::local"], []),
    "bare-constructor": (*CONF, ["r.py::K.__init__"], []),
    "module-attr-constructor": (*CONF, ["r.py::K.__init__"], []),
    "no-rt-old-facts": (*CONF, ["r.py::uniq"], []),
    "param-cb2": (*POSS, [], ["r.py::cb"]),
    "static-dispatch": (*CONF, ["r.py::local"], []),
    "bare-nested-def": (*CONF, ["r.py::outer.inner"], []),
}


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    return build(tmp_path_factory.mktemp("table"), SRC, ROWS)


@pytest.mark.parametrize("marker", sorted(WANT))
def test_probe_table_row(table, marker):
    assert view(*table, marker) == WANT[marker]


def test_the_table_has_a_row_for_every_expectation():
    assert {r["marker"] for r in ROWS} == set(WANT)


def test_a_fully_demoted_site_is_unresolved_with_no_targets_and_the_schema_legal_form(table):
    by_line, report, lines = table
    ln = next(i for i, t in enumerate(lines, start=1) if "# untyped-unique" in t)
    rec = by_line[ln]
    assert (rec["resolution"], rec["targets"], rec["status"], rec["reason"]) == ("unresolved", [], "possible", "name-match")
    assert "externalTarget" not in rec


def test_name_linked_names_exactly_the_sites_that_lost_a_target_and_counts_them(table):
    by_line, report, lines = table
    want = {"untyped-unique", "untyped-ambiguous", "inherited-on-subclass", "typed-with-extra", "param-cb", "param-cb2"}
    ids_ = {by_line[next(i for i, t in enumerate(lines, start=1) if f"# {m}" in t)]["id"] for m in want}
    assert set(report["nameLinked"]) == ids_
    assert report["nameLinkCounts"] == {"nameLinkedDemoted": 6, "selfResolved": 0}
    for entry in report["nameLinked"].values():
        assert all(set(t) == {"symbolId", "qname"} for t in entry)
        assert entry == sorted(entry, key=lambda t: t["symbolId"])


def test_a_bundle_with_name_links_keeps_the_header_counts_valid(tmp_path):
    from ccg import bundle
    path = tmp_path / "repo"
    commit = git_repo(path, {"r.py": SRC})
    facts = py_facts({"r.py": SRC})
    add_params(facts, SRC)
    records, _ = assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None,
                                          repo=path, commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC,
                                          built_at=BUILT_AT)
    assert set(records[0]["counts"]) == set(bundle.RECORD_TYPES[1:])


def test_the_normalizer_version_is_0_7():
    assert assemble_bundle.NORMALIZER_VERSION == "0.7"


def test_a_short_type_name_receiver_stays_demoted_as_a_counted_limit(tmp_path):
    """Pins a LIMIT (P1-50, `extractors/joern/README.md`), not a goal: Joern gives `rt` as the bare annotation name (`Campaign`) when the annotated
    class is imported and not resolved, so `target == rt + "." + name` fails and a call that does reach `Campaign.run` is `possible`/`name-match`
    (status only: the candidate is listed and `merge_scip` re-confirms it when SCIP agrees). Accepting the short name would also confirm an
    annotation that names a same-named class of another unit."""
    src = "class Campaign:\n    def run(self):\n        return 1\n\n\ndef go(c: Campaign):\n    c.run()  # short-rt\n"
    rows = [call("short-rt", "run", "Campaign", [("Campaign.run", f"{MOD}.Campaign.run")], enc="go", recv="c")]
    by_line, report, lines = build(tmp_path, src, rows)
    assert view(by_line, report, lines, "short-rt") == (*POSS, [], ["r.py::Campaign.run"])


# --- the `self` rule ------------------------------------------------------------------------------------------------
SELF_SRC = '''\
def m():
    return 9


class C:
    def m(self):
        return 1

    def f(self):
        self.m()  # self-unique

    def g(self):
        self.nothere()  # self-undefined

    @classmethod
    def cm(cls):
        cls.m()  # cls-unique

    @staticmethod
    def sm(self):
        self.m()  # static-self

    def h(self):
        def inner():
            self.m()  # self-nested
        inner()
        self.x.m()  # self-attr-chain

    def k(this):
        this.m()  # receiver-not-self

    def o(self):
        other.m()  # other-receiver-in-self-method

    def p(self):
        self.m()  # self-typed-to-the-module


class D(C):
    def n(self):
        self.m()  # self-inherited


class E:
    def e1(self):
        return 2

    def f(self):
        self.e1()  # self-overridden-in-unit


class F(E):
    def e1(self):
        return 3


class P:
    @property
    def pm(self):
        return 4

    def f(self):
        self.pm()  # self-property
'''
FM = f"{MOD}.m"
SELF_ROWS = [
    call("self-unique", "m", "ANY", [("m", FM), ("C.m", f"{MOD}.C.m")], enc="C.f", recv="self"),
    call("self-undefined", "nothere", "ANY", [("m", FM)], enc="C.g", recv="self"),
    call("cls-unique", "m", "ANY", [("m", FM), ("C.m", f"{MOD}.C.m")], enc="C.cm", recv="cls"),
    call("static-self", "m", "ANY", [("m", FM), ("C.m", f"{MOD}.C.m")], enc="C.sm", recv="self"),
    call("self-nested", "m", "ANY", [("m", FM), ("C.m", f"{MOD}.C.m")], enc="C.h.inner", recv="self"),
    call("self-attr-chain", "m", "ANY", [("m", FM), ("C.m", f"{MOD}.C.m")], enc="C.h",
         recv={"o": "attr", "b": {"o": "name", "n": "self"}, "f": "x"}),       # `self.x.m()`: the receiver is the attribute `self.x`
    call("receiver-not-self", "m", "ANY", [("m", FM), ("C.m", f"{MOD}.C.m")], enc="C.k", recv="this"),
    call("other-receiver-in-self-method", "m", "ANY", [("m", FM), ("C.m", f"{MOD}.C.m")], enc="C.o", recv="other"),
    call("self-typed-to-the-module", "m", MOD, [("m", FM), ("C.m", f"{MOD}.C.m")], enc="C.p", recv="self"),
    call("self-inherited", "m", "ANY", [("m", FM), ("C.m", f"{MOD}.C.m")], enc="D.n", recv="self"),
    call("self-property", "pm", "ANY", [("P.pm", f"{MOD}.P.pm")], enc="P.f", recv="self"),
    call("self-overridden-in-unit", "e1", "ANY", [("E.e1", f"{MOD}.E.e1"), ("F.e1", f"{MOD}.F.e1")], enc="E.f", recv="self"),
]
SELF_WANT = {
    # `m` is unique to the enclosing class among the class methods; the same-named module function cannot be reached by `self.m`
    "self-unique": (*CONF, ["r.py::C.m"], ["r.py::m"]),
    "self-undefined": (*POSS, [], ["r.py::m"]),
    "cls-unique": (*POSS, [], ["r.py::C.m", "r.py::m"]),
    "static-self": (*POSS, [], ["r.py::C.m", "r.py::m"]),
    "self-nested": (*POSS, [], ["r.py::C.m", "r.py::m"]),
    "self-attr-chain": (*POSS, [], ["r.py::C.m", "r.py::m"]),
    "receiver-not-self": (*POSS, [], ["r.py::C.m", "r.py::m"]),
    "other-receiver-in-self-method": (*POSS, [], ["r.py::C.m", "r.py::m"]),
    # a type-backed target wins: the module-typed `self` backs the module function `m` (rt + ".m"), the `self` rule does not replace it
    "self-typed-to-the-module": (*CONF, ["r.py::m"], ["r.py::C.m"]),
    "self-inherited": (*POSS, [], ["r.py::C.m", "r.py::m"]),
    # the pinned override test: a subclass of the unit overriding `m` is a second class-method candidate
    "self-overridden-in-unit": (*POSS, [], ["r.py::E.e1", "r.py::F.e1"]),
    # a property accessor is not a method that `self.m()` dispatches to
    "self-property": (*POSS, [], ["r.py::P.pm.<get>"]),
}


@pytest.fixture(scope="module")
def self_table(tmp_path_factory):
    return build(tmp_path_factory.mktemp("self"), SELF_SRC, SELF_ROWS)


@pytest.mark.parametrize("marker", sorted(SELF_WANT))
def test_self_rule_row(self_table, marker):
    assert view(*self_table, marker) == SELF_WANT[marker]


def test_the_self_rule_counts_its_one_site_and_name_links_it_too(self_table):
    by_line, report, lines = self_table
    assert report["nameLinkCounts"]["selfResolved"] == 1
    assert report["nameLinkCounts"]["nameLinkedDemoted"] == len(SELF_ROWS)    # every row dropped a target (the self site its module function)


def test_the_self_resolved_sites_are_listed_by_call_site_id_for_the_acceptance_draw(self_table):
    """The acceptance population (`random.Random(20261119).sample` over the sites the rule confirmed, sorted by id) needs the ids, not only a count."""
    by_line, report, lines = self_table
    ln = next(i for i, t in enumerate(lines, start=1) if "# self-unique" in t)
    assert report["selfResolvedSites"] == [by_line[ln]["id"]]
    assert report["nameLinkCounts"]["selfResolved"] == len(report["selfResolvedSites"])


def test_a_self_resolved_site_that_keeps_its_one_stored_target_is_still_listed(tmp_path):
    """Joern had already narrowed the site to the class method: nothing is dropped, yet the type rule demoted it and the `self` rule restored it."""
    src = "class C:\n    def m(self):\n        return 1\n\n    def f(self):\n        self.m()  # ok\n"
    rows = [call("ok", "m", "ANY", [("C.m", f"{MOD}.C.m")], enc="C.f", recv="self")]
    by_line, report, lines = build(tmp_path, src, rows)
    assert view(by_line, report, lines, "ok") == (*CONF, ["r.py::C.m"], [])
    assert len(report["selfResolvedSites"]) == 1 and report["nameLinked"] == {}


def test_a_self_receiver_is_the_methods_own_receiver_parameter_not_just_the_name_self(tmp_path):
    """`self` that is not parameter 0 of an instance method (a staticmethod's first parameter named `self` is index 0, not -1)
    does not qualify; the row above pins it, this one pins that a method's own `self` does: same facts, a plain method."""
    src = "def m():\n    return 9\n\n\nclass C:\n    def m(self):\n        return 1\n\n    def f(self):\n        self.m()  # ok\n"
    rows = [call("ok", "m", "ANY", [("m", FM), ("C.m", f"{MOD}.C.m")], enc="C.f", recv="self")]
    by_line, report, lines = build(tmp_path, src, rows)
    assert view(by_line, report, lines, "ok") == (*CONF, ["r.py::C.m"], ["r.py::m"])


# --- the `self` rule counts definers in the whole unit, not among Joern's candidates (P1-50 Part A) ---------------------------------
#: Another class of the unit defines the method, so a `self.m()` is not confirmed to the enclosing class's own: it could dispatch to the
#: other one. The axes: the other definer is {a subclass, a superclass, an unrelated class} x {listed in the fact's `in`, absent from
#: `in` (Joern's own type recovery had narrowed the call to the enclosing class's method)}. The `absent` cells are the defect: the old
#: rule counted the candidates and confirmed them. A same-named module function is not a class method and does not count.
OTHER_SRC = '''\
class Base:
    def ov_in(self):
        return 1

    def ov_out(self):
        return 1

    def caller(self):
        self.ov_in()  # sub-in
        self.ov_out()  # sub-out


class Sub(Base):
    def ov_in(self):
        return 2

    def ov_out(self):
        return 2

    def sp_in(self):
        return 3

    def sp_out(self):
        return 3

    def caller2(self):
        self.sp_in()  # super-in
        self.sp_out()  # super-out


class Root:
    def sp_in(self):
        return 4

    def sp_out(self):
        return 4


class U1:
    def un_in(self):
        return 5

    def un_out(self):
        return 5

    def caller(self):
        self.un_in()  # unrelated-in
        self.un_out()  # unrelated-out


class U2:
    def un_in(self):
        return 6

    def un_out(self):
        return 6


def fn_only():
    return 7


class Solo:
    def fn_only(self):
        return 8

    def solo_f(self):
        self.fn_only()  # module-function-of-the-same-name


class PropOwner:
    @property
    def pr(self):
        return 9


class PropUser:
    def pr(self):
        return 10

    def caller(self):
        self.pr()  # property-of-another-class
'''


def other_row(marker, name, enc, *defs):
    return call(marker, name, "ANY", [(f"{c}.{name}", f"{MOD}.{c}.{name}") for c in defs], enc=enc, recv="self")


OTHER_ROWS = [
    other_row("sub-in", "ov_in", "Base.caller", "Base", "Sub"),
    other_row("sub-out", "ov_out", "Base.caller", "Base"),
    other_row("super-in", "sp_in", "Sub.caller2", "Sub", "Root"),
    other_row("super-out", "sp_out", "Sub.caller2", "Sub"),
    other_row("unrelated-in", "un_in", "U1.caller", "U1", "U2"),
    other_row("unrelated-out", "un_out", "U1.caller", "U1"),
    other_row("module-function-of-the-same-name", "fn_only", "Solo.solo_f", "Solo"),
    other_row("property-of-another-class", "pr", "PropUser.caller", "PropUser"),
]
OTHER_WANT = {
    "sub-in": (*POSS, [], ["r.py::Base.ov_in", "r.py::Sub.ov_in"]),
    "sub-out": (*POSS, [], ["r.py::Base.ov_out"]),
    "super-in": (*POSS, [], ["r.py::Root.sp_in", "r.py::Sub.sp_in"]),
    "super-out": (*POSS, [], ["r.py::Sub.sp_out"]),
    "unrelated-in": (*POSS, [], ["r.py::U1.un_in", "r.py::U2.un_in"]),
    "unrelated-out": (*POSS, [], ["r.py::U1.un_out"]),
    "module-function-of-the-same-name": (*CONF, ["r.py::Solo.fn_only"], []),
    "property-of-another-class": (*POSS, [], ["r.py::PropUser.pr"]),             # a property accessor is a definer too: `self.pr` could be it
}


@pytest.fixture(scope="module")
def other_table(tmp_path_factory):
    return build(tmp_path_factory.mktemp("other"), OTHER_SRC, OTHER_ROWS)


@pytest.mark.parametrize("marker", sorted(OTHER_WANT))
def test_self_rule_counts_the_other_definers_of_the_unit_not_the_candidates_joern_listed(other_table, marker):
    assert view(*other_table, marker) == OTHER_WANT[marker]


def test_the_other_definer_table_has_a_row_for_every_expectation_and_confirms_exactly_one_site(other_table):
    assert {r["marker"] for r in OTHER_ROWS} == set(OTHER_WANT)
    assert other_table[1]["nameLinkCounts"]["selfResolved"] == 1


def test_another_file_defining_the_name_in_the_unit_also_blocks_the_rule(tmp_path):
    """The symbol table is the unit's, not the file's: a class of a sibling module defining the name is another definer."""
    path = tmp_path / "repo"
    src = "class C:\n    def m(self):\n        return 1\n\n    def f(self):\n        self.m()  # ok\n"
    commit = git_repo(path, {"r.py": src, "s.py": "class D:\n    def m(self):\n        return 2\n"})
    facts = py_facts({"r.py": src, "s.py": "class D:\n    def m(self):\n        return 2\n"})
    add_params(facts, src)
    f = method(facts, "C.f")
    facts["cs"].append({"k": "cs", "m": f["id"], "id": "9000", "name": "m", "mfn": "", "l": 6, "c": 1, "code": "m()", "dt": DYN, "rt": "ANY",
                        "in": [{"id": method(facts, "C.m")["id"], "full": f"{MOD}.C.m"}], "out": [],
                        "args": [{"i": 0, "an": "", "f": [], "e": {"o": "name", "n": "self"}}]})
    records, report = assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None,
                                               repo=path, commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC,
                                               built_at=BUILT_AT)
    [rec] = [r for r in of(records, "callsite") if r["calleeName"] == "m" and r["line"] == 6]
    assert (rec["status"], rec["reason"], rec["targets"]) == ("possible", "name-match", []) and report["selfResolvedSites"] == []


# 5d probe note (Joern 4.0.579, throwaway snippets; the rows above are these results written by hand):
#   python: untyped parameter `p.uniq(2)` rt ANY and links the module function; `k.meth()` rt `b.py:<module>.K`, target `...K.meth`;
#   `b.helper(3)` rt `b.py:<module>`, target `...<module>.helper`; `self.m()` rt ANY; `cls.m()` rt `a.py:<module>.A` (TYPED: the
#   target `A.m2` is rt + ".m2", so a real `cls.m()` stays confirmed, which is why the measured repo demotes 0); parameter `cb(1)`
#   refsTo METHOD_PARAMETER_IN, module function / nested def / imported name / aliased name / constructor / builtin / shadowing
#   local `sh2 = local; sh2(1)` refsTo LOCAL. javascript: `p: any` rt ANY, links the class method; `this.m()` in a class method rt
#   `a.ts::program:A`, target `a.ts::program:A:m2`; bare `cb` METHOD_PARAMETER_IN, `local`/`helper`/`new K()` LOCAL.
