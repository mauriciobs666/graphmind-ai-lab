"""The 5d overlay helpers (`tests/overlay_5d.py`): the independent `ast` reference of the `self` rule, and the facts overlay.

The reference never imports `assemble`; its expectations here are hand-written over small sources.
"""
from __future__ import annotations

import json
import textwrap

import pytest

import overlay_5d as O


def tree(tmp_path, files):
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(textwrap.dedent(text), encoding="utf-8")
    return tmp_path


def sites(tmp_path, files, **kw):
    return dict(O.ast_self_sites(tree(tmp_path, files), "u", **kw))


UNIQUE = {"a.py": textwrap.dedent("""\
    class C:
        def m(self):
            return 1

        def f(self):
            return self.m()
""")}


def test_a_self_call_of_a_method_the_class_defines_once_is_listed_with_the_repo_path_and_the_callee_line(tmp_path):
    assert sites(tmp_path, UNIQUE) == {("u/a.py", 6, "m"): 1}


def test_a_subclass_override_in_the_unit_removes_the_site_from_the_strict_reference_only(tmp_path):
    files = {"a.py": UNIQUE["a.py"] + "\nclass D(C):\n    def m(self):\n        return 2\n"}
    assert sites(tmp_path, files) == {}
    assert sites(tmp_path / "x", files, strict=False) == {("u/a.py", 6, "m"): 1}          # a different class names `m`, loosely still own


def test_another_file_defining_the_name_counts_as_another_class_of_the_unit(tmp_path):
    files = {**UNIQUE, "b.py": "class E:\n    def m(self):\n        pass\n"}
    assert sites(tmp_path, files) == {}


@pytest.mark.parametrize("body", [
    "def f(self):\n            def inner():\n                return self.m()\n            return inner()",       # a nested function
    "def f(self):\n            return (lambda: self.m())()",                                                           # a lambda
    "def f(this):\n            return this.m()",                                                                        # not `self`
    "def f(self):\n            return self.x.m()",                                                                      # a chain
    "def f(self):\n            return other.m()",                                                                       # another receiver
    "@staticmethod\n    def f(self):\n        return self.m()",                                             # a staticmethod
    "def f(self):\n            class In:\n                def g(me):\n                    return self.m()\n            return In",   # a nested class
], ids=["nested-def", "lambda", "other-name", "chain", "other-receiver", "staticmethod", "nested-class"])
def test_only_a_direct_self_call_in_the_methods_own_body_qualifies(tmp_path, body):
    src = f"class C:\n    def m(self):\n        return 1\n\n    {body}\n"
    assert sites(tmp_path, {"a.py": src}) == {}


def test_a_method_the_class_defines_twice_or_as_a_property_does_not_qualify(tmp_path):
    twice = "class C:\n    def m(self):\n        pass\n\n    def m(self):\n        pass\n\n    def f(self):\n        return self.m()\n"
    prop = "class C:\n    @property\n    def m(self):\n        pass\n\n    def f(self):\n        return self.m()\n"
    assert sites(tmp_path / "t", {"a.py": twice}) == {}
    assert sites(tmp_path / "p", {"a.py": prop}) == {}


def test_a_multiline_call_is_keyed_by_the_line_of_the_callee_name(tmp_path):
    src = "class C:\n    def m(self):\n        pass\n\n    def f(self):\n        return (self\n                .m())\n"
    assert sites(tmp_path, {"a.py": src}) == {("u/a.py", 7, "m"): 1}


def test_two_calls_on_one_line_are_two_sites_and_a_root_unit_has_no_prefix(tmp_path):
    src = "class C:\n    def m(self):\n        pass\n\n    def f(self):\n        return self.m() + self.m()\n"
    assert dict(O.ast_self_sites(tree(tmp_path, {"a.py": src}), ".")) == {("a.py", 6, "m"): 2}


def test_an_async_method_and_a_comprehension_in_the_body_qualify(tmp_path):
    src = ("class C:\n    async def m(self):\n        pass\n\n    async def f(self):\n"
           "        return [self.m() for _ in range(2)]\n")
    assert sites(tmp_path, {"a.py": src}) == {("u/a.py", 6, "m"): 1}


# -- the facts overlay --------------------------------------------------------------------------------------------------------
def test_the_overlay_adds_rt_only_to_listed_cs_facts_and_appends_the_listed_consts(tmp_path, monkeypatch):
    fake = {"units": {"t": {"rt": ["ANY"], "sites": [["10", "S1", 0, []]], "constFacts": [{"k": "const", "m": "1", "name": "X", "l": 1, "e": {}}],
                            "constRecords": []}}}
    monkeypatch.setattr(O, "extras", lambda: fake)
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(o) for o in [{"k": "cs", "id": "10", "name": "f"}, {"k": "cs", "id": "11", "name": "g"},
                                                      {"k": "m", "id": "1"}]) + "\n")
    out = O.overlaid_facts(src, "t", tmp_path / "out.jsonl")
    got = [json.loads(l) for l in out.read_text().splitlines()]
    assert got == [{"k": "cs", "id": "10", "name": "f", "rt": "ANY"}, {"k": "cs", "id": "11", "name": "g"}, {"k": "m", "id": "1"},
                   {"k": "const", "m": "1", "name": "X", "l": 1, "e": {}}]


def test_a_stale_table_naming_a_fact_the_oracle_lacks_is_an_error(tmp_path, monkeypatch):
    fake = {"units": {"t": {"rt": ["ANY"], "sites": [["99", "S1", 0, []]], "constFacts": [], "constRecords": []}}}
    monkeypatch.setattr(O, "extras", lambda: fake)
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"k": "cs", "id": "10", "name": "f"}) + "\n")
    with pytest.raises(AssertionError, match="not in"):
        O.overlaid_facts(src, "t", tmp_path / "out.jsonl")


def test_the_extras_file_is_self_consistent():
    e = O.extras()
    assert e["joern"] == "v4.0.579"
    for tag, u in e["units"].items():
        assert all(0 <= i < len(u["rt"]) for _, _, i, _ in u["sites"])
        assert len({c for c, *_ in u["sites"]}) == len(u["sites"]) == len({s for _, s, *_ in u["sites"]})
        assert set(u["selfByJoernNarrowing"] if "selfByJoernNarrowing" in u else []) <= {s for _, s, *_ in u["sites"]}
    assert len(e["units"]["fcs"]["sites"]) == 1479 and len(e["units"]["fcs"]["selfByJoernNarrowing"]) == 117
    assert len(e["units"]["sp"]["constFacts"]) == 8 and len(e["units"]["sp"]["constRecords"]) == 3
