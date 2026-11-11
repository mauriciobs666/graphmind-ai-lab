"""S28 flow projection: s28_core.py (facts -> points/edges, endpoint join) and s28_metrics.py (pair answers) on planted fixtures.

No Joern: a planted `extract.sc`-style facts file, a planted source tree and planted golden pairs. Covers pair keying, the endpoint tiers
(exact; ret by unique line; call by unique line `ordinalShift`; the sk == tk guard), CONTROLLED_BY and DATA edge direction, param index, call
ordinal and the C# 0-based line offset. One regression test reproduces the six pooled recall figures from the committed
results/projection-<lang>.answers.jsonl and truth/<lang>.flow-final.jsonl. The analyst's mutants run on COPIES of the sources
(`test_mutant_*`, via the CENSUS_SRC env var like test_defs_census.py) and each must make the other tests here fail.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"
SRC = Path(os.environ.get("CENSUS_SRC") or REPO)
sys.path.insert(0, str(SRC))
import s28_core as C  # noqa: E402
import s28_metrics as sm  # noqa: E402

# ----------------------------------------------------------------------------------------------------------- planted python method
PY_SRC = """def f(self, x, y):
    if x:
        return g(x)
    v = h(y)
    return g(v) + g(y)
"""


def py_facts():
    rows = [
        {"k": "m", "id": "1", "name": "f", "full": "a.py:f", "file": "a.py", "l": 1, "le": 5, "c": 1, "pk": "TYPE_DECL", "pid": "9", "pfull": "", "code": "f"},
        {"k": "p", "m": "1", "i": 0, "name": "self", "l": 1, "c": 7, "code": "self"},
        {"k": "p", "m": "1", "i": 1, "name": "x", "l": 1, "c": 13, "code": "x"},
        {"k": "p", "m": "1", "i": 2, "name": "y", "l": 1, "c": 16, "code": "y"},
        {"k": "r", "m": "1", "id": "10", "l": 3, "c": 9}, {"k": "r", "m": "1", "id": "11", "l": 5, "c": 5},
        {"k": "c", "m": "1", "id": "20", "l": 2, "c": 8, "code": "x", "lab": "IDENTIFIER"},
        {"k": "cs", "m": "1", "id": "30", "name": "g", "l": 3, "c": 16, "code": "g(x)"},
        {"k": "cs", "m": "1", "id": "31", "name": "h", "l": 4, "c": 9, "code": "h(y)"},
        {"k": "cs", "m": "1", "id": "32", "name": "g", "l": 5, "c": 12, "code": "g(v)"},
        {"k": "cs", "m": "1", "id": "33", "name": "g", "l": 5, "c": 19, "code": "g(y)"},
        # DATA: y -> h(y); g(x) -> return@3; g(y) -> return@5; x -> g(x)
        {"k": "d", "m": "1", "uk": "cs", "uid": "31", "dk": "p", "did": "1", "di": 2},
        {"k": "d", "m": "1", "uk": "r", "uid": "10", "dk": "cs", "did": "30"},
        {"k": "d", "m": "1", "uk": "r", "uid": "11", "dk": "cs", "did": "33"},
        {"k": "d", "m": "1", "uk": "cs", "uid": "30", "dk": "p", "did": "1", "di": 1},
        # CONTROLLED_BY: return@3 is controlled by the `x` condition; a CDG self edge is dropped
        {"k": "cb", "m": "1", "pk": "r", "pid": "10", "cond": "20"},
        {"k": "cb", "m": "1", "pk": "c", "pid": "20", "cond": "20"},
        {"k": "stats", "loadMs": "1", "projectionMs": "1", "totalMs": "2", "methods": 1, "cdgRaw": 2, "cdgSelf": 1, "dataRaw": 4, "histogram": {}},
    ]
    return rows


P = lambda i, n: {"point": "param", "index": i, "name": n}                       # noqa: E731
CALL = lambda n, k, l: {"point": "call", "calleeName": n, "ordinal": k, "line": l}  # noqa: E731
RET = lambda k, l: {"point": "ret", "ordinal": k, "line": l}                      # noqa: E731
COND = lambda l, c: {"point": "cond", "line": l, "code": c}                       # noqa: E731

PY_PAIRS = [  # (name, kind, source, sink, expected status, via-tag on an endpoint or None)
    ("param->call", "data", P(1, "y"), CALL("h", 0, 4), "connected", None),
    ("data reversed", "data", CALL("h", 0, 4), P(1, "y"), "not-connected", None),
    ("call->ret", "data", CALL("g", 0, 3), RET(0, 3), "connected", None),
    ("second g on the line is ordinal 2", "data", CALL("g", 2, 5), RET(1, 5), "connected", None),
    ("first g on the line is ordinal 1 and does not feed the ret", "data", CALL("g", 1, 5), RET(1, 5), "not-connected", None),
    ("control cond->ret", "control", COND(2, "x"), RET(0, 3), "connected", None),
    ("control reversed", "control", RET(0, 3), COND(2, "x"), "not-connected", None),
    ("param index 0 is x", "data", P(0, "x"), CALL("g", 0, 3), "connected", None),
    ("ret by unique line", "data", CALL("g", 0, 3), RET(7, 3), "connected", "unique-ret-on-line"),
    ("call by unique line (ordinalShift)", "data", P(0, "x"), CALL("g", 9, 3), "connected", "ordinalShift"),
    ("call on no such line", "data", P(1, "y"), CALL("h", 0, 6), "endpoint-unresolved", None),
    ("same point both ends", "data", CALL("h", 0, 4), CALL("h", 0, 4), "endpoint-same-point", None),
    ("several same-name calls on the line, wrong ordinal", "data", P(0, "x"), CALL("g", 5, 5), "endpoint-unresolved", None),
    ("two-hop chain x -> g(x) -> ret", "data", P(0, "x"), RET(0, 3), "connected", None),
]

CS_SRC = """class A {
  int M(int x) {
    if (x > 0) {
      return G(x);
    }
    return 0; } }
"""


def cs_facts():
    # csharpsrc2cpg numbers lines from 0: every `l` below is the 1-based source line minus 1
    return [
        {"k": "m", "id": "1", "name": "M", "full": "A.M", "file": "A.cs", "l": 1, "le": 5, "c": 3, "pk": "TYPE_DECL", "pid": "9", "pfull": "", "code": "M"},
        {"k": "p", "m": "1", "i": 0, "name": "this", "l": 1, "c": 3, "code": "this"},
        {"k": "p", "m": "1", "i": 1, "name": "x", "l": 1, "c": 13, "code": "int x"},
        {"k": "r", "m": "1", "id": "10", "l": 3, "c": 6}, {"k": "r", "m": "1", "id": "11", "l": 5, "c": 4},
        {"k": "c", "m": "1", "id": "20", "l": 2, "c": 8, "code": "x > 0", "lab": "CALL"},
        {"k": "cs", "m": "1", "id": "30", "name": "G", "l": 3, "c": 13, "code": "G(x)"},
        {"k": "d", "m": "1", "uk": "cs", "uid": "30", "dk": "p", "did": "1", "di": 1},
        {"k": "d", "m": "1", "uk": "r", "uid": "10", "dk": "cs", "did": "30"},
        {"k": "cb", "m": "1", "pk": "r", "pid": "10", "cond": "20"},
        {"k": "stats", "loadMs": "1", "projectionMs": "1", "totalMs": "2", "methods": 1, "cdgRaw": 1, "cdgSelf": 0, "dataRaw": 2, "histogram": {}},
    ]


CS_PAIRS = [
    ("param->call", "data", P(0, "x"), CALL("G", 0, 4), "connected", None),
    ("call->ret", "data", CALL("G", 0, 4), RET(0, 4), "connected", None),
    ("control", "control", COND(3, "x > 0"), RET(0, 4), "connected", None),
]


def run_cfg(tmp, lang, rel, src_text, facts, qname, span, pairs):
    root = tmp / "src"
    root.mkdir(exist_ok=True)
    (root / rel).write_text(src_text, encoding="utf-8")
    fp = tmp / "facts.jsonl"
    fp.write_text("\n".join(json.dumps(r) for r in facts) + "\n", encoding="utf-8")
    samp = [{"ordinal": 1, "path": rel, "qname": qname, "lineStart": span[0], "lineEnd": span[1]}]
    gold = [{"ordinal": 1, "path": rel, "pairs": [{"kind": k, "source": s, "sink": t, "confidence": "sure"} for _, k, s, t, *_ in pairs]}]
    return sm.analyse_cfg(lang, "off", {"factsFile": str(fp)}, samp, gold, {}, {}, C.Src(root), "x")


@pytest.fixture
def py(tmp_path):
    return run_cfg(tmp_path, "python", "a.py", PY_SRC, py_facts(), "a.py::f", (1, 5), PY_PAIRS)


@pytest.mark.parametrize("i", range(len(PY_PAIRS)), ids=[p[0] for p in PY_PAIRS])
def test_planted_pair_status(py, i):
    _, res, answers, spur, mm, *_ = py
    name, kind, s, t, status, via = PY_PAIRS[i]
    a = answers[(1, i)]
    assert a["status"] == status, (name, a)
    assert a["connected"] == (status == "connected")
    if via:
        d = a["endpointDetail"]
        assert via in {d["source"].get("via"), d["sink"].get("via")}


def test_two_hop_chain_reports_its_hops_and_single_edges_report_one(py):
    answers = py[2]
    assert answers[(1, 13)]["hops"] == 2 and answers[(1, 0)]["hops"] == 1


def test_pooled_recall_and_unresolved_classes(py):
    res = py[1]
    assert (res["pooled"]["k"], res["pooled"]["n"]) == (8, 14)
    assert res["status"] == {"connected": 8, "not-connected": 3, "endpoint-unresolved": 2, "endpoint-same-point": 1}
    assert (res["control"]["k"], res["control"]["n"]) == (1, 2)


def test_every_projected_edge_is_covered_so_no_spurious_pair_and_control_direction_is_sink_then_cond(py):
    spur, per_m = py[3], py[6]
    M = per_m[0][1]
    assert len(M.data) == 4 and len(M.ctl) == 1
    ret10, cond20 = [k for k in M.ctl][0]
    assert ret10[0] == "r" and cond20[0] == "c"          # CONTROLLED_BY edge = (controlled point, condition)
    assert spur == []


def test_spurious_control_pairs_use_the_same_direction_as_the_projection(tmp_path):
    ok = [p for p in PY_PAIRS if p[0] == "control cond->ret"]
    rev = [p for p in PY_PAIRS if p[0] == "control reversed"]
    spur_ok = run_cfg(tmp_path, "python", "a.py", PY_SRC, py_facts(), "a.py::f", (1, 5), ok)[3]
    assert [s for s in spur_ok if s[1] == "control"] == []
    spur_rev = run_cfg(tmp_path, "python", "a.py", PY_SRC, py_facts(), "a.py::f", (1, 5), rev)[3]
    assert len([s for s in spur_rev if s[1] == "control"]) == 1


def test_param_and_call_and_ret_points_are_keyed_as_the_spec_says(py):
    M = py[6][0][1]
    assert sorted(M.params) == [-1, 0, 1]                   # python: `self` (Joern 0) is spec -1, x -> 0, y -> 1
    assert [(c[1], c[2], c[3]) for c in sorted(M.calls, key=lambda c: (c[3], c[2]))] == [("g", 0, 3), ("h", 0, 4), ("g", 1, 5), ("g", 2, 5)]
    assert [(r[1], r[2]) for r in M.rets] == [(0, 3), (1, 5)]


def test_csharp_zero_based_lines_are_shifted_by_one(tmp_path):
    _, res, answers, *_ = run_cfg(tmp_path, "csharp", "A.cs", CS_SRC, cs_facts(), "A.cs::A.M", (2, 6), CS_PAIRS)
    assert [answers[(1, i)]["status"] for i in range(3)] == ["connected"] * 3
    assert C.LINE_OFFSET["csharp"] == 1


def test_pair_keying_is_by_ordinal_kind_and_collapsed_endpoint_identity():
    pr = {"kind": "data", "source": P(1, "y"), "sink": CALL("g", 2, 5)}
    assert sm.pair_id(3, pr) == (3, "data", ("param", 1), ("call", "g", 2))
    assert sm.pair_id(4, pr) != sm.pair_id(3, pr)
    assert sm.pair_id(3, dict(pr, kind="control")) != sm.pair_id(3, pr)
    assert sm.pair_id(3, dict(pr, sink=CALL("g", 3, 5))) != sm.pair_id(3, pr)
    c = {"kind": "control", "source": COND(2, "a   ==  b"), "sink": RET(0, 3)}
    assert sm.pair_id(1, c)[2] == ("cond", "a == b")
    assert sm.pair_id(1, dict(c, sink=RET(1, 3))) != sm.pair_id(1, c)


def test_load_tags_keys_audit_decisions_by_pair_id(tmp_path, monkeypatch):
    (tmp_path / "adjudication").mkdir()
    pr = {"kind": "data", "source": P(1, "y"), "sink": CALL("g", 2, 5)}
    rows = [{"ordinal": 3, "pair": pr, "decision": "valid", "convention": "conv-a"},
            {"ordinal": 4, "pair": pr, "decision": "invalid", "convention": "conv-b"}, {"ordinal": 9, "note": "no pair"}]
    (tmp_path / "adjudication" / "zz-flow-audit.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(sm, "HERE", tmp_path)
    tags, removed = sm.load_tags("zz")
    assert tags == {sm.pair_id(3, pr): {"conv-a"}} and removed == {sm.pair_id(4, pr): ["conv-b"]}


# ----------------------------------------------------------------------------------------------------------- regression on the real results
POOLED = {"python": (307, 480), "javascript": (60, 151), "go": (246, 433), "java": (314, 465), "csharp": (227, 392), "cpp": (116, 349)}


@pytest.mark.parametrize("lang", sorted(POOLED))
def test_committed_answers_reproduce_the_pooled_recall_and_cover_exactly_the_truth_pairs(lang):
    ans = [json.loads(l) for l in open(REPO / "results" / f"projection-{lang}.answers.jsonl", encoding="utf-8")]
    gold = [json.loads(l) for l in open(REPO / "truth" / f"{lang}.flow-final.jsonl", encoding="utf-8")]
    truth = {(g["ordinal"], i): p for g in gold for i, p in enumerate(g["pairs"])}
    assert {(a["ordinal"], a["idx"]) for a in ans} == set(truth)
    assert all(a["pair"]["source"] == truth[(a["ordinal"], a["idx"])]["source"] and a["kind"] == truth[(a["ordinal"], a["idx"])]["kind"] for a in ans)
    r = sm.prop(sum(1 for a in ans if a["connected"]), len(ans))
    assert (r["k"], r["n"]) == POOLED[lang]


# ----------------------------------------------------------------------------------------------------------- P1-09 join fixes
# The two cond-join fixes (`docs/reviews/connected-code-graph-p1-09-ml.md`): (a) a label that is a python `for` header resolves to the
# line's UNIQUE `iteratorNonEmptyOrException` cond, only when no existing tier matched; (b) a tier "equal ignoring whitespace and
# parentheses" before `contains`. Planted facts only (no corpus pair): each test fails on the P0 join for the stated reason.
ITER = "iteratorNonEmptyOrException"


def cond_facts(conds, cbs=(), calls=()):
    """one python method `f` (lines 1-4) with the given conds [(id, line, code)], calls [(id, name, line)], cb [(pk, pid, cond)]."""
    rows = [{"k": "m", "id": "1", "name": "f", "full": "a.py:f", "file": "a.py", "l": 1, "le": 4, "c": 1, "pk": "TYPE_DECL", "pid": "9", "pfull": "", "code": "f"},
            {"k": "p", "m": "1", "i": 1, "name": "xs", "l": 1, "c": 7, "code": "xs"}]
    rows += [{"k": "c", "m": "1", "id": i, "l": ln, "c": 5, "code": code, "lab": "UNKNOWN"} for i, ln, code in conds]
    rows += [{"k": "cs", "m": "1", "id": i, "name": n, "l": ln, "c": 9, "code": n + "(x)"} for i, n, ln in calls]
    rows += [{"k": "cb", "m": "1", "pk": pk, "pid": pid, "cond": c} for pk, pid, c in cbs]
    rows.append({"k": "stats", "loadMs": "1", "projectionMs": "1", "totalMs": "2", "methods": 1, "cdgRaw": len(cbs), "cdgSelf": 0, "dataRaw": 0, "histogram": {}})
    return rows


FOR_SRC = "def f(xs):\n    for x in xs:\n        use(x)\n    return 1\n"


def ctl(tmp, src, facts, source, sink=CALL("use", 0, 3)):
    pairs = [("p", "control", source, sink, None, None)]
    _, res, answers, *_ = run_cfg(tmp, "python", "a.py", src, facts, "a.py::f", (1, 4), pairs)
    return answers[(1, 0)]


def test_for_header_label_resolves_to_the_unique_iterator_cond(tmp_path):
    facts = cond_facts([("20", 2, ITER)], [("cs", "30", "20")], [("30", "use", 3)])
    a = ctl(tmp_path, FOR_SRC, facts, COND(2, "x in xs"))
    assert a["status"] == "connected" and a["endpointDetail"]["source"]["via"] == "for-header-iterator"


def test_for_header_rule_leaves_two_iterator_conds_on_the_line_unresolved(tmp_path):
    facts = cond_facts([("20", 2, ITER), ("21", 2, ITER)], [("cs", "30", "20")], [("30", "use", 3)])
    a = ctl(tmp_path, FOR_SRC, facts, COND(2, "x in xs"))
    assert a["status"] == "endpoint-unresolved" and a["endpointDetail"]["source"]["reason"] == "cond-several-iterator"


def test_for_header_rule_requires_the_label_line_to_be_the_loop_header(tmp_path):
    src = "def f(xs):\n    z = a if x in xs else b\n        use(x)\n    return 1\n"   # an iterator cond on the line, but not a `for` header
    facts = cond_facts([("20", 2, ITER)], [("cs", "30", "20")], [("30", "use", 3)])
    assert ctl(tmp_path, src, facts, COND(2, "x in xs"))["status"] == "endpoint-unresolved"
    src2 = "def f(xs):\n    async for x in xs:\n        use(x)\n    return 1\n"
    (tmp_path / "b").mkdir()
    assert ctl(tmp_path / "b", src2, facts, COND(2, "x in xs"))["status"] == "connected"      # `async for` is a header too


def test_an_existing_tier_wins_over_the_for_header_rule(tmp_path):
    facts = cond_facts([("20", 2, ITER), ("21", 2, "x in xs")], [("cs", "30", "21")], [("30", "use", 3)])
    a = ctl(tmp_path, FOR_SRC, facts, COND(2, "x in xs"))
    assert a["status"] == "connected" and "via" not in a["endpointDetail"]["source"]        # resolved by the exact tier, to cond 21


def test_a_contains_match_is_not_overridden_by_the_for_header_rule(tmp_path):
    facts = cond_facts([("20", 2, ITER), ("21", 2, "x")], [("cs", "30", "21")], [("30", "use", 3)])
    a = ctl(tmp_path, FOR_SRC, facts, COND(2, "x in xs"))
    assert a["endpointDetail"]["source"]["via"] == "contains" and a["status"] == "connected"


PAREN_SRC = "def f(self, i):\n    if not (self._open or i in self._r):\n        use(i)\n    return 1\n"


def test_paren_insensitive_tier_resolves_to_the_whole_condition_not_the_operand(tmp_path):
    whole = "not self._open or i in self._r"
    facts = cond_facts([("20", 2, whole), ("21", 2, "self._open")], [("cs", "30", "20")], [("30", "use", 3)])
    a = ctl(tmp_path, PAREN_SRC, facts, COND(2, "not (self._open or i in self._r)"))
    assert a["status"] == "connected" and a["endpointDetail"]["source"]["via"] == "equal-ignoring-whitespace-and-parentheses"


def test_the_conjunction_split_by_the_frontend_still_resolves_by_contains_exactly_as_before(tmp_path):
    src = "def f(e):\n    if e.s is not None and 500 <= e.s <= 599:\n        use(e)\n    return 1\n"
    facts = cond_facts([("20", 2, "e.s is not None"), ("21", 2, "500 <= tmp5")], [("cs", "30", "20")], [("30", "use", 3)])
    a = ctl(tmp_path, src, facts, COND(2, "e.s is not None and 500 <= e.s <= 599"))
    assert a["endpointDetail"]["source"]["via"] == "contains" and a["status"] == "connected"


def test_two_paren_equal_conds_on_the_line_stay_unresolved(tmp_path):
    facts = cond_facts([("20", 2, "not a or b"), ("21", 2, "not (a) or b")], [("cs", "30", "20")], [("30", "use", 3)])
    src = "def f(a, b):\n    if not (a or b):\n        use(a)\n    return 1\n"
    a = ctl(tmp_path, src, facts, COND(2, "not (a or b)"))
    assert a["status"] == "endpoint-unresolved"


def test_a_resolved_header_endpoint_without_a_chain_stays_not_connected(tmp_path):
    """Only endpoints change: the iterator cond resolves, the projection has no edge, so the pair is not connected."""
    facts = cond_facts([("20", 2, ITER)], [], [("30", "use", 3)])
    a = ctl(tmp_path, FOR_SRC, facts, COND(2, "x in xs"))
    assert a["status"] == "not-connected" and a["endpointDetail"]["source"]["via"] == "for-header-iterator"


def test_the_p0_join_switch_restores_the_old_resolution(tmp_path, monkeypatch):
    facts = cond_facts([("20", 2, ITER)], [("cs", "30", "20")], [("30", "use", 3)])
    monkeypatch.setattr(C, "JOIN_P0_ONLY", True)
    assert ctl(tmp_path, FOR_SRC, facts, COND(2, "x in xs"))["status"] == "endpoint-unresolved"


# --- gate P1-09 (analyst) additions: planted-fact tests for the line restriction and the guards of the two join fixes
def test_iterator_cond_on_another_line_is_not_taken(tmp_path):
    facts = cond_facts([("20", 3, ITER)], [("cs", "30", "20")], [("30", "use", 3)])     # the only iterator cond is on line 3, label on line 2
    assert ctl(tmp_path, FOR_SRC, facts, COND(2, "x in xs"))["status"] == "endpoint-unresolved"


def test_paren_equal_cond_on_another_line_is_not_taken(tmp_path):
    src = "def f(a, b):\n    if not (a or b):\n        use(a)\n    return 1\n"
    facts = cond_facts([("20", 3, "not a or b")], [("cs", "30", "20")], [("30", "use", 3)])
    assert ctl(tmp_path, src, facts, COND(2, "not (a or b)"))["status"] == "endpoint-unresolved"


def test_header_rule_does_not_override_several_contains(tmp_path):
    """Plan wording ("only when the existing tiers fail"): two `contains` candidates are an ambiguity the header rule must not resolve."""
    facts = cond_facts([("20", 2, ITER), ("21", 2, "x"), ("22", 2, "xs")], [("cs", "30", "20")], [("30", "use", 3)])
    a = ctl(tmp_path, FOR_SRC, facts, COND(2, "x in xs"))
    assert a["status"] == "endpoint-unresolved" and a["endpointDetail"]["source"]["reason"] == "cond-several-contains"


def test_for_inside_a_comprehension_line_is_not_a_header(tmp_path):
    src = "def f(xs):\n    ys = [y for x in xs]\n        use(x)\n    return 1\n"
    facts = cond_facts([("20", 2, ITER)], [("cs", "30", "20")], [("30", "use", 3)])
    assert ctl(tmp_path, src, facts, COND(2, "x in xs"))["status"] == "endpoint-unresolved"


# ----------------------------------------------------------------------------------------------------------- recompute from the stored facts (opt-in)
POOLED_P1_JOIN = {"python": (336, 480), "javascript": (60, 151), "go": (246, 433), "java": (314, 465), "csharp": (227, 392), "cpp": (116, 349)}
# P0's pins (`POOLED`) stay the OLD-join record; this is the SAME shipped projection scored by the P1 join (python +29, nothing else moves).
# The status relabels (recall unchanged) the P1 join causes; expected lists are the recompute's output, not typed values.
# python (10): pairs into/out of a `for` header whose chain the SHIPPED projection lacks; their endpoint now resolves, they stay misses.
# csharp (2, RULED: data-scientist note Pass 2, accepted): ordinal 25, the label is wrapped in one outer pair of parentheses; it was
# `cond-several-contains`, the join fix matches the whole condition; recall 227/392 is unchanged and no connected pair moves.
RELABELLED = {"python": sorted(["12|1", "14|4", "17|15", "18|10", "18|3", "21|0", "22|3", "23|9", "24|2", "6|7"]), "csharp": ["25|2", "25|6"]}
MANIFEST = Path(__file__).resolve().parent / "golden" / "projection" / "join-new-hits-python.json"


def _recompute(lang):
    """(old-join answers, new-join answers) over the stored P0 facts of `lang`, `analyse_cfg` as P0 ran it (dev data only)."""
    import yaml
    work = REPO.parents[1] / ".ccg-work" / "frontend-fitness" / "joern" / lang
    if not (work / "s28-run.json").is_file():
        pytest.skip(f"UNRUN, not green: {work}/s28-run.json absent (gitignored P0 facts)")
    runs = json.loads((work / "s28-run.json").read_text())["runs"]
    ek = sm.EXTRACT_KEYS[lang]
    run = runs[f"off.{ek}"]
    if not run.get("factsFile") or not Path(run["factsFile"]).is_file():
        pytest.skip(f"UNRUN, not green: facts file of {lang} absent")
    cfg = yaml.safe_load(open(REPO / "corpora.yaml"))["corpora"][lang]
    samp = [json.loads(l) for l in open(REPO / "sample" / f"{lang}-flow.jsonl", encoding="utf-8")]
    gold = [json.loads(l) for l in open(REPO / "truth" / f"{lang}.flow-final.jsonl", encoding="utf-8")]
    tags, removed = sm.load_tags(lang)
    src = C.Src(C.CORPORA / lang / "src")
    out = []
    for old in (True, False):
        C.JOIN_P0_ONLY = old
        try:
            _, res, answers, *_ = sm.analyse_cfg(lang, "off", run, samp, gold, tags, removed, src, cfg["commit"])
        finally:
            C.JOIN_P0_ONLY = False
        out.append({f"{k[0]}|{k[1]}": (a["status"], json.dumps(a.get("endpointDetail", {}), sort_keys=True, default=str)) for k, a in answers.items()})
    return out


@pytest.mark.skipif(not os.environ.get("CCG_S28_RECOMPUTE"), reason="UNRUN: opt-in (CCG_S28_RECOMPUTE=1; scores the stored P0 facts, ~1 min)")
@pytest.mark.parametrize("lang", sorted(POOLED_P1_JOIN))
def test_p1_join_changes_only_the_listed_python_pairs(lang):
    """Closed list (ml note Q3/C4/C6): python 307 -> 336 on the unchanged shipped projection (+29, the committed manifest), 0 flips from
    connected, and every other pair's status identical; no other language changes at all. Resolution (endpointDetail) of a pair connected
    before may change only for the one named (9,8)."""
    old, new = _recompute(lang)
    assert old.keys() == new.keys()
    k_old = sum(1 for v in old.values() if v[0] == "connected")
    k_new = sum(1 for v in new.values() if v[0] == "connected")
    assert k_old == POOLED[lang][0] and k_new == POOLED_P1_JOIN[lang][0]
    gained = sorted(k for k in new if new[k][0] == "connected" and old[k][0] != "connected")
    lost = [k for k in old if old[k][0] == "connected" and new[k][0] != "connected"]
    assert not lost
    # a pair whose endpoint now resolves but still has no chain moves "endpoint-unresolved" -> "not-connected": a relabel, not recall
    relabel = sorted(k for k in new if new[k][0] != old[k][0] and k not in gained)
    assert all(old[k][0] == "endpoint-unresolved" and new[k][0] == "not-connected" for k in relabel)
    assert relabel == RELABELLED.get(lang, [])                                  # nothing else changes status
    if lang == "python":
        assert gained == sorted(json.loads(MANIFEST.read_text())["pairIds"]) and len(gained) == 29
        re_resolved = sorted(k for k in old if old[k][0] == "connected" and old[k][1] != new[k][1])
        assert re_resolved == ["9|8"]
    else:
        assert not gained
        assert [k for k in old if old[k][0] == "connected" and old[k][1] != new[k][1]] == []


# ----------------------------------------------------------------------------------------------------------- mutants
MUTANTS = {
    "control_direction_in_answer": ("s28_metrics.py", [("ok = (tk, sk) in M.ctl", "ok = (sk, tk) in M.ctl")]),
    "control_direction_in_core": ("s28_core.py", [("M.ctl.add((pt, cd))", "M.ctl.add((cd, pt))")]),
    "control_direction_in_spurious": ("s28_metrics.py", [("(r[0], r[1]) if pr[\"kind\"] == \"data\" else (r[1], r[0])", "(r[0], r[1]) if pr[\"kind\"] == \"data\" else (r[0], r[1])")]),
    "data_edge_reversed": ("s28_core.py", [("M.data.add((df, use))", "M.data.add((use, df))")]),
    "param_index_plus_one": ("s28_core.py", [('return {p["i"]: (-1 if p["i"] == 0 else p["i"] - 1) for p in ps}', 'return {p["i"]: (-1 if p["i"] == 0 else p["i"]) for p in ps}')]),
    "call_ordinal_plus_one": ("s28_core.py", [('M.calls.append((("cs", c["id"]), name, k, pos[c["id"]][0]))', 'M.calls.append((("cs", c["id"]), name, k + 1, pos[c["id"]][0]))')]),
    "csharp_offset_1_to_0": ("s28_core.py", [('LINE_OFFSET = {"csharp": 1}', 'LINE_OFFSET = {"csharp": 0}')]),
    "ret_unique_line_tier_removed": ("s28_core.py", [('        if len(online) == 1:\n            det["via"] = "unique-ret-on-line"', '        if False:\n            det["via"] = "unique-ret-on-line"')]),
    "call_unique_line_tier_removed": ("s28_core.py", [('        if len(online) == 1:\n            det["via"] = "ordinalShift"', '        if False:\n            det["via"] = "ordinalShift"')]),
    "same_point_guard_removed": ("s28_metrics.py", [("if sk is None or tk is None or sk == tk:", "if sk is None or tk is None:")]),
    "call_key_without_ordinal": ("s28_metrics.py", [('return ("call", ep["calleeName"], ep["ordinal"])', 'return ("call", ep["calleeName"])')]),
    "pair_key_without_kind": ("s28_metrics.py", [('return (ordinal, pr["kind"], ckey(pr["source"]), ckey(pr["sink"]))', 'return (ordinal, ckey(pr["source"]), ckey(pr["sink"]))')]),
    "join_for_header_rule_off": ("s28_core.py", [("if not ct and not JOIN_P0_ONLY and _for_header(M, line, code):", "if False:")]),
    "join_for_header_uniqueness_removed": ("s28_core.py", [('            if len(its) == 1:\n                det["via"] = "for-header-iterator"', '            if len(its) >= 1:\n                det["via"] = "for-header-iterator"')]),
    "join_for_header_text_guard_removed": ("s28_core.py", [("and _for_header(M, line, code):", "and True:")]),
    "join_paren_tier_removed": ("s28_core.py", [("        if not JOIN_P0_ONLY:  # P1-09 (b)", "        if False:  # P1-09 (b)")]),
    "join_paren_tier_after_contains": ("s28_core.py", [
        ("        if not JOIN_P0_ONLY:  # P1-09 (b)", "        if False:  # P1-09 (b)"),
        ("        if not ct and not JOIN_P0_ONLY and _for_header(M, line, code):",
         "        _np = lambda x: \"\".join(x.replace(\"(\", \"\").replace(\")\", \"\").split())  # noqa: E731\n"
         "        _pe = [c for c in online if _np(c[2]) == _np(code)]\n"
         "        if len(_pe) == 1:\n            det[\"via\"] = \"equal-ignoring-whitespace-and-parentheses\"\n            return _pe[0][0], det\n"
         "        if not ct and not JOIN_P0_ONLY and _for_header(M, line, code):")]),
    "join_paren_tier_ignores_whitespace_only": ("s28_core.py", [('noparen = lambda x: "".join(x.replace("(", "").replace(")", "").split())', 'noparen = lambda x: "".join(x.split())')]),
    "join_iterator_lookup_not_line_restricted": ("s28_core.py", [("its = [c for c in online if c[2] == ITER_COND]", "its = [c for c in M.conds if c[2] == ITER_COND]")]),
    "join_paren_tier_not_line_restricted": ("s28_core.py", [("ep_ = [c for c in online if noparen(c[2]) == noparen(code)]", "ep_ = [c for c in M.conds if noparen(c[2]) == noparen(code)]")]),
    "join_header_rule_overrides_contains_ambiguity": ("s28_core.py", [("if not ct and not JOIN_P0_ONLY and _for_header(M, line, code):", "if not JOIN_P0_ONLY and _for_header(M, line, code):")]),
    "join_header_text_loosened": ("s28_core.py", [('return text.startswith(f"for {code}:") or text.startswith(f"async for {code}:")', 'return f"for {code}" in text')]),
    "reach_one_hop_only": ("s28_core.py", [("                if y not in seen:\n                    seen.add(y)\n                    nxt.append(y)", "                pass")]),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name, tmp_path):
    if os.environ.get("CENSUS_SRC"):
        pytest.skip("inner run")
    copy = tmp_path / "src"
    copy.mkdir()
    for p in REPO.glob("*.py"):
        shutil.copy(p, copy / p.name)
    shutil.copy(REPO / "corpora.yaml", copy / "corpora.yaml")
    fname, edits = MUTANTS[name]
    text = (copy / fname).read_text(encoding="utf-8")
    for old, new in edits:
        assert old in text, f"mutant {name}: pattern not found, the mutant would be a no-op"
        text = text.replace(old, new)
    (copy / fname).write_text(text, encoding="utf-8")
    env = dict(os.environ, CENSUS_SRC=str(copy), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "not mutant", str(Path(__file__))],
                       capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[1])
    assert p.returncode != 0 and "failed" in p.stdout, f"mutant {name} survived:\n{p.stdout[-600:]}"
