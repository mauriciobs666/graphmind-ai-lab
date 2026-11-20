"""P1-50 Part B: the constant text of a Python f-string (`normalize.core.Resolver.element`).

Joern hands the pieces of a Python f-string (`formatString`) over as raw source text, never quoted
(`tests/golden/fstring/probe.json`, recorded from the shipped `extract.sc` through a live Joern 4.0.579 and re-checked by the opt-in live
row below), so a `lit` piece of a `fmt` in a Python file is `{"const": <c>}`, not a literal to parse. Expected elements are written by hand
in the golden from each f-string's source text. Everything else must not change: JavaScript/TypeScript templates (backtick regex), `concat`,
every literal that is not a piece of a Python `fmt`, the empty-piece filter and the depth cap.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from extractors.joern import run
from extractors.joern.normalize import core

CPG = Path(__file__).resolve().parent.parent
GOLD = CPG / "tests" / "golden" / "fstring"
PROBE = json.loads((GOLD / "probe.json").read_text(encoding="utf-8"))
ROWS = PROBE["rows"]


def resolver(tmp_path, lang="python"):
    return core.Resolver(core.Builder(lang, ".", "r", {}, core.Source(tmp_path)), {}, {})


@pytest.mark.parametrize("row", ROWS, ids=[f"L{r['line']}" for r in ROWS])
def test_a_python_fstring_gets_its_constant_pieces_as_the_probe_recorded_them(tmp_path, row):
    assert resolver(tmp_path).element(row["joern"], "a.py") == row["expected"], row["source"]


def test_the_probe_covers_the_shapes_the_plan_names():
    sources = {r["source"] for r in ROWS}
    for want in ('f"/a/{x}"', 'f"it\'s {x}"', 'f"{a}\'b\'{c}"', 'f"{a}1{b}"', 'f"{{lit}}{x}"', 'f"a\\n{x}"', '"p" f"{x}"', 'f"{x:{w}.{p}}"', 'f""'):
        assert want in sources, want


def test_a_text_piece_that_would_parse_as_a_literal_is_still_text(tmp_path):
    """`'b'` and `1` are source text of the f-string, not a string and an integer."""
    R = resolver(tmp_path)
    fmt = lambda c: {"o": "fmt", "p": [{"o": "lit", "c": c}, {"o": "unk", "l": "<operator>.formattedValue"}]}
    assert R.element(fmt("'b'"), "a.py")["template"][0] == {"const": "'b'"}
    assert R.element(fmt("1"), "a.py")["template"][0] == {"const": "1"}
    assert R.element(fmt("True"), "a.py")["template"][0] == {"const": "True"}


def test_an_empty_text_piece_is_still_filtered(tmp_path):
    fmt = {"o": "fmt", "p": [{"o": "lit", "c": ""}, {"o": "unk", "l": "<operator>.formattedValue"}]}
    assert resolver(tmp_path).element(fmt, "a.py") == {"template": [{"unknown": "runtime-built"}]}


def test_the_text_is_not_normalized_or_cut(tmp_path):
    R = resolver(tmp_path)
    text = "Å " + "x" * 300
    assert R.element({"o": "fmt", "p": [{"o": "lit", "c": text}, {"o": "unk", "l": "u"}]}, "a.py")["template"][0] == {"const": text}


def test_a_piece_past_the_depth_cap_is_still_capped(tmp_path):
    fmt = {"o": "fmt", "p": [{"o": "lit", "c": "a"}]}
    assert resolver(tmp_path).element(fmt, "a.py", depth=4) == {"template": [{"unknown": "value-set-cap"}]}


@pytest.mark.parametrize("code, want", [("'a'", {"const": "a"}), ('"a"', {"const": "a"}), ("42", {"const": 42}), ("a b", {"unknown": "runtime-built"})])
def test_a_literal_that_is_not_a_piece_of_a_python_fmt_is_still_parsed(tmp_path, code, want):
    R = resolver(tmp_path)
    assert R.element({"o": "lit", "c": code}, "a.py") == want
    assert R.element({"o": "add", "p": [{"o": "lit", "c": code}]}, "a.py") == {"concat": [want]}


def test_the_pieces_of_a_javascript_template_are_parsed_as_before(tmp_path):
    R = resolver(tmp_path, "javascript")
    fmt = lambda c: {"o": "fmt", "p": [{"o": "lit", "c": c}, {"o": "name", "n": "other"}]}
    assert R.element(fmt("`/a/`"), "a.ts")["template"][0] == {"const": "/a/"}
    assert R.element(fmt("'/a/'"), "a.ts")["template"][0] == {"const": "/a/"}
    assert R.element(fmt("/a/"), "a.ts")["template"][0] == {"unknown": "runtime-built"}          # unparsable text is unknown, as before


def test_a_python_fmt_nested_in_a_concat_gets_constants_too(tmp_path):
    inner = {"o": "fmt", "p": [{"o": "lit", "c": "/x/"}, {"o": "unk", "l": "u"}]}
    got = resolver(tmp_path).element({"o": "add", "p": [{"o": "lit", "c": "'p'"}, inner]}, "a.py")
    assert got == {"concat": [{"const": "p"}, {"template": [{"const": "/x/"}, {"unknown": "runtime-built"}]}]}


# -- the probe, live ----------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("CCG_FSTRING_LIVE") != "1", reason="UNRUN: set CCG_FSTRING_LIVE=1 (bounded local Joern, one JVM)")
def test_the_recorded_probe_is_what_the_shipped_script_writes_now(tmp_path):
    """Re-runs the probe: parse `probe_fstrings.py.txt` as `src/f.py`, run the shipped `extract.sc`, compare each call's argument-1 element."""
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(GOLD / PROBE["probeFile"], src / "f.py")
    env = {**os.environ, "JAVA_OPTS": "-Xmx1500m"}
    joern_env = CPG.parent / "skills" / "joern-cpg" / "scripts" / "joern-env.sh"
    script = (f". {joern_env} && cd {tmp_path} && timeout -s KILL 200 joern-parse src --language pythonsrc -o p.cpg.bin >parse.log 2>&1 && "
              f"timeout -s KILL 200 joern --script {run.EXTRACT_SC} --param cpgFile=p.cpg.bin --param out=facts.jsonl --param lang=python >ex.log 2>&1")
    subprocess.run(["bash", "-c", script], check=True, env=env)
    got = {}
    for line in (tmp_path / "facts.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o["k"] == "cs" and o["name"] == "get":
            got[o["l"]] = o["args"][1]["e"]
    assert got == {r["line"]: r["joern"] for r in ROWS}
