"""The f-string reference (`tests/fstring_reference.py`) against the probe golden and planted sources: it never imports the normalizer."""
from __future__ import annotations

import ast
import json
import sys
import textwrap
from pathlib import Path

import pytest

import fstring_reference as F

GOLD = Path(__file__).resolve().parent / "golden" / "fstring"
PROBE = json.loads((GOLD / "probe.json").read_text(encoding="utf-8"))


def test_the_reference_does_not_import_the_normalizer():
    tree = ast.parse((Path(F.__file__)).read_text(encoding="utf-8"))
    mods = {n.module if isinstance(n, ast.ImportFrom) else a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import) else [None]) if True}
    assert not any(m and ("extractors" in m or "normalize" in m or "ccg" in m) for m in mods), mods


def test_the_reference_pieces_of_every_non_concatenated_probe_row_equal_the_hand_written_expectation(tmp_path):
    """Merged constant text as the raw source segment; a formatted value is one unknown. The hand-written golden is the oracle for the reference."""
    src = (GOLD / PROBE["probeFile"]).read_text(encoding="utf-8")
    tree = ast.parse(src)
    by_line = {r["line"]: r for r in PROBE["rows"]}
    seen = 0
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "get":
            [arg] = call.args
            row = by_line[call.func.end_lineno]
            assert isinstance(arg, ast.JoinedStr), row["source"]
            if F.is_implicit_concatenation(src, arg):
                assert row["joern"]["l"] == "<operator>.stringExpressionList"
                continue
            exp = row["expected"]
            exp = {"template": [], **exp} if exp == {"const": ""} else exp
            want = exp["template"] if "template" in exp else [exp]
            assert F.pieces(src, arg) == want, row["source"]
            seen += 1
    assert seen == 16


def test_implicit_concatenation_is_detected_only_at_the_top_level_of_the_literal():
    mk = lambda text: (text, ast.parse(text).body[0].value)
    for text, want in [('f"a{x}" "b"', True), ('"p" f"{x}"', True), ('f"a{x}"', False), ("f\"{d['k']}\"", False),
                       ('f"{f\'{y}\'}"', False), ('(f"a"\n f"b{x}")', True)]:
        src, node = mk(text)
        assert F.is_implicit_concatenation(src, node) is want, text


def test_scan_keys_calls_by_file_callee_line_and_argument_path_and_finds_constants_and_decorators(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text(textwrap.dedent('''\
        BASE = f"/api/{V}/x"
        PLAIN = "no"


        @app.get(f"/d/{i}")
        def h():
            c.post(f"/threads/{t}",
                   data={"a": f"x{y}", "b": {"c": f"z{w}", "d": {"e": {"f": f"deep{q}"}}}},
                   label=f"k{z}")
            g(f"a{x}" "b")
    '''), encoding="utf-8")
    f = F.scan(tmp_path, "svc")
    assert f.constants == {"svc/pkg/m.py::BASE": [{"const": "/api/"}, F.UNKNOWN, {"const": "/x"}]}
    assert f.decorators == [(("svc/pkg/m.py", 5, "a0"), [{"const": "/d/"}, F.UNKNOWN])]
    got = {k: v[0][0] for k, v in f.calls.items()}
    assert got[("svc/pkg/m.py", 7, "a0")] == [{"const": "/threads/"}, F.UNKNOWN]
    assert got[("svc/pkg/m.py", 7, "kw.data.a")] == [{"const": "x"}, F.UNKNOWN]
    assert got[("svc/pkg/m.py", 7, "kw.data.b.c")] == [{"const": "z"}, F.UNKNOWN]
    assert ("svc/pkg/m.py", 7, "kw.data.b.d.e.f") not in got                              # three fields deep is the cap (`.b.d.e`), the next is not read
    assert got[("svc/pkg/m.py", 7, "kw.label")] == [{"const": "k"}, F.UNKNOWN]
    assert f.calls[("svc/pkg/m.py", 10, "a0")][0][1] is True                              # implicit concatenation is flagged, not dropped


def test_a_root_unit_has_no_path_prefix(tmp_path):
    (tmp_path / "a.py").write_text('f(f"{x}/y")\n', encoding="utf-8")
    assert list(F.scan(tmp_path, ".").calls) == [("a.py", 1, "a0")]


def test_everywhere_lists_every_fstring_once_and_not_the_ones_nested_in_a_replacement_field(tmp_path):
    (tmp_path / "a.py").write_text('rows = [{"k": f"S{i}"}]\nx = f"a{f\'b{y}\'}c"\n', encoding="utf-8")
    got = F.scan(tmp_path, ".").everywhere
    assert sorted(got, key=len) == [[{"const": "S"}, F.UNKNOWN], [{"const": "a"}, F.UNKNOWN, {"const": "c"}]]       # `ast.walk` order is not source order
