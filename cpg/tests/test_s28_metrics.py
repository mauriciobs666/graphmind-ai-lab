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
