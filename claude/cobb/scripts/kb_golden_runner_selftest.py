#!/usr/bin/env python3
"""kb_golden_runner_selftest.py -- selftest and mutation target for `kb_golden_runner.py` (W7a).

Plain assertion script, same house style as `pwq_kb_selftest.py`; no network (a scripted fake searcher and a recording
fake transport). Expected values are written out by hand from the rules, never computed by the code under test.

Run: `python3 claude/cobb/scripts/kb_golden_runner_selftest.py`   -> "ALL PASS", exit 0; else exit 1.
Mutation runs: point `KB_GOLDEN_RUNNER_PATH` at a mutated copy of `kb_golden_runner.py`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location("kb_golden_runner_under_test", os.environ.get("KB_GOLDEN_RUNNER_PATH", HERE / "kb_golden_runner.py"))
g = importlib.util.module_from_spec(_spec)
sys.modules["kb_golden_runner_under_test"] = g
_spec.loader.exec_module(g)

TEMPLATE = "Instruct: X\nQuery: {situation}"
A, B, C, D, E_ = ("a" * 32, "b" * 32, "c" * 32, "d" * 32, "e" * 32)


def chunks(*pairs):
    return [{"chunkId": f"c{i}", "documentId": d, "seq": 0, "score": s, "text": ""} for i, (d, s) in enumerate(pairs)]


class FakeSearch:
    def __init__(self, script):
        self.script, self.calls = script, []

    def search(self, ws, query, limit):
        self.calls.append((ws, query, limit))
        return self.script[query][:limit]


def pair(pid, situation, expected=(), distractors=(), applies=True, cls=None, reason=None):
    cls = cls or ("negative" if not expected else "single" if len(expected) == 1 else "family")
    return {"id": pid, "kb": "k", "tags": ["a"], "answer_class": cls, "situation": situation, "expected": list(expected),
            "distractors": list(distractors), "applies": applies, "na_reason": reason, "pilot": False}


def fx(*pairs, r6=E_):
    return {"schema": 1, "description": "t", "r6_document": r6, "pairs": list(pairs)}


def q(situation):
    return TEMPLATE.replace("{situation}", situation, 1)


def run(fixture, script, floors=(0.43, 0.42), primary=0.43):
    s = FakeSearch({q(k): v for k, v in script.items()})
    return g.run_all(fixture, s, TEMPLATE, ws="w", primary_floor=primary, floors=floors), s


def t01_query_is_prefixed_with_limit_5():
    tpl = g.kb.extract_prefix((REPO / "skills/agent-kb-retrieval/SKILL.md").read_text(encoding="utf-8"))
    assert tpl == ("Instruct: Given a coding agent's description of its current situation, retrieve the distilled "
                   "technique or rule that applies to it.\nQuery: {situation}"), tpl
    s = FakeSearch({q("sit"): chunks((A, 0.2))})
    g.run_all(fx(pair("P1", "sit", [A])), s, TEMPLATE, ws="w", primary_floor=0.43, floors=(0.43,))
    assert s.calls == [("w", "Instruct: X\nQuery: sit", 5)], s.calls


def t02_hit_only_inside_top5():
    f = fx(pair("P1", "s1", [A]), pair("P2", "s2", [A]))
    six = chunks((B, .1), (B, .1), (B, .1), (B, .1), (B, .1), (A, .2))
    r, _ = run(f, {"s1": chunks((B, .1), (A, .3)), "s2": six})
    p1, p2 = r["pairs"]
    assert p1["hit_top5"] is True and p1["found"] == 1 and p1["docs"][1] == {"documentId": A, "score": 0.3, "rank": 2}, p1
    assert p2["hit_top5"] is False and p2["found"] == 0, p2


def t03_document_level_best_score_and_rank():
    r, _ = run(fx(pair("P1", "s", [A])), {"s": chunks((B, .30), (A, .25), (B, .31), (A, .35))})
    p = r["pairs"][0]
    assert [d["documentId"] for d in p["docs"]] == [B, A], p["docs"]
    assert p["expected_hits"][0] == {"documentId": A, "found": True, "rank": 2, "score": 0.25, "under_floor": {"0.43": True, "0.42": True}}, p


def t04_floor_rejects_strictly_above():
    f = fx(pair("P1", "a", [A]), pair("P2", "b", [A]), pair("P3", "c", [A]))
    r, _ = run(f, {"a": chunks((A, 0.43)), "b": chunks((A, 0.425)), "c": chunks((A, 0.4301))})
    h = [(p["hit_floor"]["0.43"], p["hit_floor"]["0.42"], p["hit_top5"]) for p in r["pairs"]]
    assert h == [(True, False, True), (True, False, True), (False, False, True)], h
    assert r["pairs"][0]["hit_floor_primary"] is True and r["pairs"][2]["hit_floor_primary"] is False
    r2, _ = run(f, {"a": chunks((A, 0.43)), "b": chunks((A, 0.425)), "c": chunks((A, 0.4301))}, floors=(0.42,), primary=0.42)
    assert [p["hit_floor_primary"] for p in r2["pairs"]] == [False, False, False]


def t05_family_is_set_recall_strict_hit():
    f = fx(pair("F1", "s", [A, B]), pair("F2", "t", [A, B]))
    r, _ = run(f, {"s": chunks((A, .2), (C, .3)), "t": chunks((A, .2), (B, .44))})
    p, p2 = r["pairs"]
    assert (p["found"], p["total"], p["hit_top5"]) == (1, 2, False), p
    assert (p2["found"], p2["total"], p2["hit_top5"], p2["hit_floor"]["0.43"], p2["floor_found"]["0.43"]) == (2, 2, True, False, 1), p2
    a = r["aggregates"]["0.43"]
    assert (a["siblings_found_top5"], a["siblings_total"], a["siblings_found_floor"]) == (3, 4, 2), a
    assert (a["family_pairs_hit_top5"], a["family_pairs"]) == (1, 2), a


def t06_pairs_map_to_their_own_documents():
    f = fx(pair("P1", "s1", [A]), pair("P2", "s2", [B]))
    r, _ = run(f, {"s1": chunks((B, .1)), "s2": chunks((A, .1))})  # each returns the other pair's document
    assert [p["hit_top5"] for p in r["pairs"]] == [False, False], r["pairs"]
    r, _ = run(f, {"s1": chunks((A, .1)), "s2": chunks((B, .1))})
    assert [p["hit_top5"] for p in r["pairs"]] == [True, True]


def t07_distractor_outranking_flagged():
    f = fx(pair("P1", "s1", [A], [D]), pair("P2", "s2", [A], [D]), pair("P3", "s3", [A], [D]))
    r, _ = run(f, {"s1": chunks((D, .2), (A, .3)), "s2": chunks((A, .2), (D, .3)), "s3": chunks((A, .2))})
    d = [p["distractors"][0] for p in r["pairs"]]
    assert d == [{"documentId": D, "in_top5": True, "score": 0.2, "outranks": True},
                 {"documentId": D, "in_top5": True, "score": 0.3, "outranks": False},
                 {"documentId": D, "in_top5": False, "score": None, "outranks": False}], d
    assert r["aggregates"]["0.43"]["distractors_outranking"] == 1


def t08_negatives_closest_score_and_false_accepts():
    f = fx(pair("P1", "s", [A]), pair("N1", "n1"), pair("N2", "n2"))
    r, _ = run(f, {"s": chunks((A, .2)), "n1": chunks((B, .44), (C, .46)), "n2": chunks((B, .425), (C, .6))})
    n1, n2 = r["pairs"][1], r["pairs"][2]
    assert (n1["closest_score"], n2["closest_score"]) == (0.44, 0.425), (n1, n2)
    assert n1["false_accept"] == {"0.43": False, "0.42": False} and n2["false_accept"] == {"0.43": True, "0.42": False}, (n1, n2)
    a = r["aggregates"]["0.43"]
    assert (a["negatives"], a["negative_false_accepts"], a["closest_negative_score"]) == (2, 1, 0.425), a
    assert (r["aggregates"]["0.42"]["negative_false_accepts"]) == 0
    assert a["pairs_applicable"] == 1 and a["pairs_hit_top5"] == 1, a  # negatives never count as recall pairs


def t09_applicability_flag_excludes_and_carries_reason():
    f = fx(pair("P1", "s1", [A]), pair("P2", "s2", [B], applies=False, reason="doc absent in ws"))
    r, _ = run(f, {"s1": chunks((A, .2)), "s2": chunks((B, .2))})
    p2 = r["pairs"][1]
    assert p2["applies"] is False and p2["na_reason"] == "doc absent in ws" and p2["hit_top5"] is None and p2["hit_floor_primary"] is None, p2
    a = r["aggregates"]["0.43"]
    assert (a["pairs_total"], a["pairs_not_applicable"], a["pairs_applicable"], a["pairs_hit_top5"]) == (2, 1, 1, 1), a
    assert a["singles"] == 1 and a["singles_hit_top5"] == 1, a


def t10_fixture_validation_rejects_bad_shapes():
    bad = [fx(pair("P1", "s", [A], applies=False)),                    # not applicable needs a reason
           fx(pair("P1", "s", [A]), pair("P1", "t", [B])),              # duplicate id
           fx(pair("P1", "s", ["xyz"])),                                # not a 32-hex id
           fx(pair("P1", " ", [A])),                                    # empty situation
           fx(pair("N1", "s", applies=True, cls="negative")),           # negative must not apply to recall
           fx(pair("P1", "s", [A]), r6="nope")]
    for b in bad:
        try:
            g.validate_fixture(b)
        except g.kb.KBError:
            continue
        raise AssertionError(f"accepted {b}")
    g.validate_fixture(fx(pair("P1", "s", [A]), pair("N1", "n", applies=False, reason="negative")))


def t11_committed_fixture_matches_the_plan_table():
    fixture = g.load_fixture(g.DEFAULT_FIXTURE)
    ps = fixture["pairs"]
    assert len(ps) == 45 and sum(p["answer_class"] == "negative" for p in ps) == 6, len(ps)
    pos = [p for p in ps if p["answer_class"] != "negative"]
    assert len(pos) == 39 and sum(len(p["expected"]) for p in pos) == 46 and sum(len(p["distractors"]) for p in pos) == 2
    assert sorted(p["id"] for p in ps if p["answer_class"] == "family") == ["Q1", "R1", "R2", "R3", "R4", "R5", "R6"]
    assert [p["applies"] for p in ps if p["answer_class"] == "negative"] == [False] * 6
    assert all(p["applies"] for p in pos)
    by = {p["id"]: p for p in ps}
    assert by["R1"]["expected"] == ["d41d743e5c344eccbf46715f54cbab44", "5c2ef4055ff34ea6a1ba5f7eb329ae4c"]
    assert by["R7"]["expected"] == ["d3c4ef7117c343bfaf003d1ad7fc44b8"] and by["R7"]["distractors"] == ["4fe63fb109bd4a04ba479389a29c8e08"]
    assert by["F1"]["distractors"] == ["719fc4c69d1845ad9298b32c0adecdc6"] and by["L2"]["expected"] == ["089f80e62c0f458992f086fae82ef557"]
    assert by["N4"]["situation"].startswith("How should I structure a React Server Components")
    assert fixture["r6_document"] == "5b1b477ff67e4e3b81c57f14899bbe48" and fixture["r6_document"] in by["R6"]["expected"]
    ids = [d for p in pos for d in p["expected"]]
    assert len(set(ids)) == 46 and all(re.fullmatch(r"[0-9a-f]{32}", d) for d in ids)


def t12_r6_document_scores_recorded():
    f = fx(pair("R6", "s", [A, E_]), r6=E_)
    r, _ = run(f, {"s": chunks((A, .2), (C, .3), (E_, .4405))})
    assert r["r6_document"] == {"documentId": E_, "pair": "R6", "in_top5": True, "rank": 3, "score": 0.4405,
                                "under_floor": {"0.43": False, "0.42": False}}, r["r6_document"]
    r, _ = run(f, {"s": chunks((A, .2))})
    assert r["r6_document"]["in_top5"] is False and r["r6_document"]["score"] is None


def t13_only_get_search_requests_are_made():
    seen = []

    def transport(method, url, body=None):
        seen.append((method, url, body))
        return 200, chunks((A, .2))

    target = g.kb.Target("pwq-test", "http://localhost:8210", False)
    g.run_all(fx(pair("P1", "sit", [A])), g.kb.RestSearcher(target, transport), TEMPLATE, ws="pwq-test", primary_floor=0.42, floors=(0.42,))
    assert len(seen) == 1 and seen[0][0] == "GET" and seen[0][2] is None, seen
    assert seen[0][1].startswith("http://localhost:8210/documents/search?") and "limit=5" in seen[0][1], seen


def main_run(argv, script=None, fixture=None):
    tmp = Path(tempfile.mkdtemp())
    fpath = tmp / "fx.json"
    fpath.write_text(json.dumps(fixture or fx(pair("P1", "s", [A]), pair("N1", "n", applies=False, reason="neg"))), encoding="utf-8")
    skill = tmp / "SKILL.md"
    skill.write_text(g.kb.DEFAULT_SKILL.read_text(encoding="utf-8"), encoding="utf-8")
    tpl = g.kb.extract_prefix(skill.read_text(encoding="utf-8"))
    s = FakeSearch({g.kb.build_query(tpl, k): v for k, v in (script or {"s": chunks((A, .2)), "n": chunks((B, .5))}).items()})
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = g.main(["--fixture", str(fpath), "--skill", str(skill)] + argv, searcher=s)
    return code, out.getvalue(), err.getvalue(), s


def t14_main_json_and_text_output():
    code, out, err, s = main_run(["--base-url", "http://localhost:8210", "--ws", "pwq-test", "--floor", "0.42", "--json"])
    assert code == 0, (code, err)
    d = json.loads(out)
    assert d["ws"] == "pwq-test" and d["base_url"] == "http://localhost:8210" and d["primary_floor"] == 0.42, d.keys()
    assert d["floors"] == [0.42, 0.43] and set(d["aggregates"]) == {"0.42", "0.43"} and [p["id"] for p in d["pairs"]] == ["P1", "N1"]
    assert d["pairs"][0]["hit_floor_primary"] is True and len(s.calls) == 2
    code, out, err, _ = main_run(["--base-url", "http://localhost:8200", "--ws", "agent-team", "--floor", "0.43"])
    assert code == 0 and "P1" in out and "HIT" in out and "floor 0.43" in out, out


def t15_target_mismatch_refused_without_searching():
    code, out, err, s = main_run(["--base-url", "http://localhost:8200", "--ws", "pwq-test", "--floor", "0.43"])
    assert code == 2 and "target table" in err and s.calls == [], (code, err)


def main() -> int:
    tests = sorted((n, f) for n, f in globals().items() if re.match(r"t\d\d_", n) and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}\n" + "".join(traceback.format_exc().splitlines(True)[-4:]))
    print(f"{len(tests) - failed}/{len(tests)} passed")
    print("ALL PASS" if not failed else "FAILURES")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
