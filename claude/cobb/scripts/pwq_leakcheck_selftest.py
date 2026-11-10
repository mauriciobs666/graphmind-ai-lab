#!/usr/bin/env python3
"""pwq_leakcheck_selftest.py -- plain assertion script (house style, see check_content_loss_selftest.py)
for `pwq_leakcheck.py`, plus a mutation runner.

    python3 pwq_leakcheck_selftest.py            # run the checks against pwq_leakcheck.py
    python3 pwq_leakcheck_selftest.py --mutate   # apply each mutant to a scratch copy; every one must go red

`PWQ_LEAKCHECK_DIR` points the checks at another copy of the module (the mutation runner uses it);
`pwq_kb.py` is always imported from this directory.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = Path(os.environ.get("PWQ_LEAKCHECK_DIR", HERE))
sys.path[:0] = [str(TARGET), str(HERE)]
sys.dont_write_bytecode = True
import pwq_leakcheck as lc  # noqa: E402

SOURCE = """---
name: x
---
Preamble text.

## Alpha bravo charlie delta echo
First body mentions zulu yankee xray whiskey and victor.

## Second Title Here
Other body talks about kilo lima mike november oscar. See `input()` calls.
"""


def sections():
    return lc.parse_source(SOURCE).sections


def hits_for(query, section=1):
    data = {"positives": [{"id": "P1", "section": section, "query": query}], "negatives": []}
    return lc.check(data, sections())


def test_tokens():
    assert lc.tokens("Hello, World! input() a_b 9x") == ["hello", "world", "input", "a", "b", "9x"]


def test_planted_body_4gram_caught():
    h = hits_for("The thing about zulu yankee xray whiskey somewhere")
    assert len(h) == 1, h
    assert (h[0]["id"], h[0]["where"], h[0]["gram"]) == ("P1", "body", "zulu yankee xray whiskey"), h


def test_three_gram_is_not_a_leak():
    assert hits_for("we see zulu yankee xray only") == []


def test_title_4gram_caught():
    h = hits_for("my alpha bravo charlie delta question")
    assert [(x["where"], x["gram"]) for x in h] == [("title", "alpha bravo charlie delta")], h


def test_case_folded():
    assert hits_for("ZULU Yankee XRAY Whiskey") != []
    assert hits_for("alpha BRAVO Charlie delta") != []


def test_punctuation_ignored():
    assert hits_for("zulu, yankee; (xray) -- whiskey!") != []


def test_only_own_section_is_checked():
    assert hits_for("kilo lima mike november", section=1) == []
    assert hits_for("kilo lima mike november", section=2) != []


def test_unknown_section_raises():
    try:
        hits_for("anything at all goes here", section=3)
    except lc.KBError:
        return
    raise AssertionError("section 3 does not exist and must raise")


def test_short_query_and_clean_query():
    assert hits_for("zulu yankee xray") == []
    assert hits_for("a wholly unrelated situation about something else entirely") == []


def test_negatives_are_not_section_scoped():
    data = {"positives": [], "negatives": [{"id": "N1", "kind": "off", "query": "no overlap whatsoever with the text"}]}
    assert lc.check(data, sections()) == []


def test_cli_exit_codes():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d, "src.md")
        src.write_text(SOURCE, encoding="utf-8")
        for name, q, want in (("clean.json", "nothing shared here at all", 0),
                              ("leak.json", "zulu yankee xray whiskey", 1)):
            p = Path(d, name)
            p.write_text(json.dumps({"positives": [{"id": "P1", "section": 1, "query": q}], "negatives": []}))
            r = subprocess.run([sys.executable, "-B", str(TARGET / "pwq_leakcheck.py"), "--set", str(p), "--source", str(src)],
                               capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(HERE)})
            assert r.returncode == want, (name, r.returncode, r.stdout, r.stderr)
            if want:
                assert "P1" in r.stdout and "zulu yankee xray whiskey" in r.stdout, r.stdout


CHECKS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

# (name, file substring, replacement) -- each must make at least one check fail.
MUTANTS = [
    ("n-gram length 4 -> 3", "N = 4", "N = 3"),
    ("n-gram length 4 -> 5", "N = 4", "N = 5"),
    ("case folding dropped", ".lower()", ""),
    ("tokenizer keeps punctuation", 're.findall(r"[a-z0-9]+"', 're.findall(r"\\S+"'),
    ("tokenizer drops digits", 're.findall(r"[a-z0-9]+"', 're.findall(r"[a-z]+"'),
    ("title not checked", 'for where, text in (("title", sec.heading), ("body", sec.body))', 'for where, text in (("body", sec.body),)'),
    ("body not checked", 'for where, text in (("title", sec.heading), ("body", sec.body))', 'for where, text in (("title", sec.heading),)'),
    ("section off by one", "secs[p[\"section\"] - 1]", "secs[p[\"section\"] % len(secs)]"),
    ("checks every section", "sec = secs[p[\"section\"] - 1]", "sec = secs[0]"),
    ("CLI always exits 0", "return 1 if hits else 0", "return 0"),
    ("unknown section silent", "raise KBError(", "return []; raise KBError("),
]


def run_checks():
    failed = []
    for fn in CHECKS:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed.append((fn.__name__, repr(e)[:200]))
    return failed


def mutate():
    src = (HERE / "pwq_leakcheck.py").read_text(encoding="utf-8")
    survivors = 0
    base = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve())], capture_output=True, text=True)
    assert base.returncode == 0, "baseline must be green before mutating:\n" + base.stdout
    print(f"{'mutant':34} result")
    for name, old, new in MUTANTS:
        assert src.count(old) >= 1, f"mutant {name!r}: pattern not found"
        with tempfile.TemporaryDirectory() as d:
            Path(d, "pwq_leakcheck.py").write_text(src.replace(old, new, 1), encoding="utf-8")
            r = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve())], capture_output=True, text=True,
                               env={**os.environ, "PWQ_LEAKCHECK_DIR": d})
        killed = r.returncode != 0
        survivors += not killed
        print(f"{name:34} {'KILLED' if killed else 'SURVIVED'}")
    print(f"{len(MUTANTS) - survivors}/{len(MUTANTS)} killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    if "--mutate" in sys.argv:
        sys.exit(mutate())
    failed = run_checks()
    for n, e in failed:
        print(f"FAIL {n}: {e}")
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    sys.exit(1 if failed else 0)
