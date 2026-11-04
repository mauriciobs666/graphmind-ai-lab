"""S32: pins for the infrastructure-signature pack's regexes, samples, libraries and runtime-detection patterns (spec 8.1, 8.2).

The spec test checks that a sample matches its own regex and that no negative does; it cannot see an edit that leaves both true
(a sample with a word added, a library renamed, the last alternative of a regex changed). ``data/infra_probes.tsv`` pins what it
cannot: per entry one ``alt`` line for every alternative of the regex at any group depth, the ``sample`` and the ``library``
verbatim. The probes are sampled from the regexes (a generator in the S32 delivery notes), so they are a regression pin that
makes an edit visible, not evidence about what a runtime prints; the spec's ``authored`` caveat covers all of them.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

import test_rule_pack_spec as spec

PROBES = Path(__file__).parent / "data/infra_probes.tsv"
ENTRIES = {e["id"]: e for e in spec.sigs()}


def probe_rows() -> list[tuple[str, str, str]]:
    rows = [tuple(ln.split("\t", 2)) for ln in PROBES.read_text().splitlines() if ln and not ln.startswith("#")]
    assert all(len(r) == 3 for r in rows)
    return rows


def arms(rx: str) -> list[tuple[int, int]]:
    """(start, end) of every arm of every alternation of ``rx``, the top level and each ``( ... | ... )`` group alike."""
    out, stack, cls, esc = [], [0], False, False
    frames: list[list[tuple[int, int]]] = [[]]
    i = 0
    while i < len(rx):
        ch = rx[i]
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif cls:
            cls = ch != "]"
        elif ch == "[":
            cls = True
        elif ch == "(":
            i += 3 if rx.startswith("(?:", i) else 1
            stack.append(i)
            frames.append([])
            continue
        elif ch == "|":
            frames[-1].append((stack[-1], i))
            stack[-1] = i + 1
        elif ch == ")":
            frames[-1].append((stack.pop(), i))
            arm_list = frames.pop()
            if len(arm_list) > 1:
                out += arm_list
        i += 1
    frames[-1].append((stack[0], len(rx)))
    if len(frames[-1]) > 1:
        out += frames[-1]
    return out


def without_arm(rx: str, arm: tuple[int, int]) -> str:
    """``rx`` with one alternative replaced by a pattern that never matches."""
    return rx[: arm[0]] + "(?!)" + rx[arm[1]:]


def test_the_pins_name_exactly_the_spec_entries():
    rows = probe_rows()
    assert {r[0] for r in rows} == set(ENTRIES)
    by = collections.defaultdict(collections.Counter)
    for rid, kind, _ in rows:
        by[rid][kind] += 1
    for rid, e in ENTRIES.items():
        assert by[rid]["alt"] >= 1 and by[rid]["sample"] == 1 and by[rid]["library"] == (1 if "library" in e else 0), rid


def test_every_probe_line_matches_its_entry():
    bad = [(rid, line) for rid, kind, line in probe_rows() if kind == "alt" and not re.search(ENTRIES[rid]["regex"], line)]
    assert not bad, bad


def test_the_pinned_samples_are_the_specs_samples_verbatim():
    pinned = {rid: v for rid, kind, v in probe_rows() if kind == "sample"}
    assert pinned == {rid: e["sample"] for rid, e in ENTRIES.items()}


def test_the_pinned_libraries_are_the_specs_libraries_verbatim():
    pinned = {rid: v for rid, kind, v in probe_rows() if kind == "library"}
    assert pinned == {rid: e["library"] for rid, e in ENTRIES.items() if "library" in e}


def test_every_alternative_of_every_regex_has_a_probe_line_that_only_it_accepts():
    """Deleting any one alternative (at any group depth) must stop at least one pinned line matching: no alternative is unwitnessed."""
    lines = collections.defaultdict(list)
    for rid, kind, line in probe_rows():
        if kind == "alt":
            lines[rid].append(line)
    unwitnessed = []
    total = 0
    for rid, e in ENTRIES.items():
        for arm in arms(e["regex"]):
            total += 1
            mutant = without_arm(e["regex"], arm)
            if all(re.search(mutant, ln) for ln in lines[rid]):
                unwitnessed.append((rid, e["regex"][arm[0]:arm[1]]))
    assert total >= 140, total
    assert not unwitnessed, unwitnessed


def test_the_arm_splitter_finds_nested_and_top_level_alternatives():
    assert [("a|b(?:c|d)e|f")[s:t] for s, t in arms("a|b(?:c|d)e|f")] == ["c", "d", "a", "b(?:c|d)e", "f"]
    assert arms("abc") == [] and [("x[|]y|z")[s:t] for s, t in arms("x[|]y|z")] == ["x[|]y", "z"]
    assert [("a\\|b")[s:t] for s, t in arms("a\\|b")] == []


# ---------------------------------------------------------------- runtime detection (8.1 `runtimes`)

DETECT = {
    "python": ['Traceback (most recent call last):', '  File "/app/x.py", line 3, in main'],
    "node": ["    at Socket.<anonymous> (/app/x.js:10:5)", "    at async run /app/x.js:10:5", "node:internal/process/task_queues:95:5"],
    "jvm": ["    at com.acme.Foo.bar(Foo.java:12)", "    at com.acme.Foo.bar(Foo.kt:12)", "    at com.acme.Foo.bar(Foo.scala:12)", "Caused by: java.net.ConnectException"],
    "dotnet": ["   at Acme.Foo.Bar() in /src/Foo.cs:line 12"],
    "go": ["goroutine 1 [running]:", "\tmain.go:12 +0x1d"],
}


@pytest.mark.parametrize("rt,line", [(rt, ln) for rt, ls in DETECT.items() for ln in ls])
def test_each_runtime_pattern_detects_each_of_its_lines_and_no_other_runtime_does(rt, line):
    hits = [r for r, rx in spec.RUNTIMES.items() if re.search(rx, line)]
    assert hits == [rt], hits


def test_every_alternative_of_every_runtime_pattern_is_witnessed_by_a_detection_line():
    assert set(DETECT) == set(spec.RUNTIMES)
    for rt, rx in spec.RUNTIMES.items():
        for arm in arms(rx):
            assert not all(re.search(without_arm(rx, arm), ln) for ln in DETECT[rt]), (rt, rx[arm[0]:arm[1]])


@pytest.mark.parametrize("line", ["ConnectionRefusedError: [Errno 111] Connection refused", "Error: connect ECONNREFUSED 127.0.0.1:5432", "plain text", "    at ",
                                  "java.net.ConnectException: Connection refused"])
def test_a_line_of_no_runtime_marker_detects_no_runtime(line):
    assert [r for r, rx in spec.RUNTIMES.items() if re.search(rx, line)] == []
