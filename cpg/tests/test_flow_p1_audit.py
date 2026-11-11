"""P1-34b: flow_p1_audit.py (`flowaud-p1`, `adjlist-p1`).  Synthetic fixtures only (no real labels, no custodian data); the hash rule is
re-implemented independently here; mutants run on COPIES via CCG_FF_DIR (like test_flow_decide.py)."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REAL = Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"
FF = Path(os.environ.get("CCG_FF_DIR") or REAL)
sys.path.insert(0, str(FF))
import flow_p1_audit as fa  # noqa: E402

COMMIT = "0caf85e3ccb757801d15d6babe55a63477a486dd"
SECRET = "ZZ-DISTINCTIVE-LABEL-TEXT-ZZ"


def sha(s):  # independent of the module under test
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ fixtures
def labels(n_methods=12, per=10, unsure_every=3):
    rows = []
    for o in range(1, n_methods + 1):
        pairs = [{"kind": "data", "source": {"point": "param", "name": SECRET}, "sink": {"point": "call", "calleeName": SECRET, "line": 5},
                  "confidence": "unsure" if (o * per + i) % unsure_every == 0 else "sure", "evidence": SECRET} for i in range(per)]
        rows.append({"ordinal": o, "path": SECRET, "qname": SECRET, "pairs": pairs})
    return rows


def truth_and_answers(n_pairs, miss_off=(), miss_on=(), per=10, extra=None):
    truth, answers = [], []
    for g in range(n_pairs):
        o, i = 1 + g // per, g % per
        if i == 0:
            truth.append({"ordinal": o, "pairs": []})
        pair = {"kind": "data", "source": {"point": "param", "name": f"s{g}"}, "sink": {"point": "call", "calleeName": f"c{g}", "line": g},
                "confidence": "sure", "evidence": f"e{g}", **(extra or {})}
        truth[-1]["pairs"].append(pair)
        answers.append({"ordinal": o, "idx": i, "connected": g not in miss_off, "on": {"connected": g not in miss_on}})
    return truth, answers


def key_of(g, per=10):
    return (1 + g // per, g % per)


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --------------------------------------------------------------------------------------------------------------------- flowaud-p1
def test_f1_is_60_pairs_by_ascending_independent_hash_over_all_pairs():
    rows = labels()
    out = fa.flowaud(rows)
    allp = [(m["ordinal"], i) for m in rows for i in range(len(m["pairs"]))]
    want = sorted(allp, key=lambda t: sha(f"ccg-p1a-v1-flowaud|python|pair|{t[0]}|{t[1]}"))[:60]
    assert [(r["ordinal"], r["pairIndex"]) for r in out["F1"]] == want and len(want) == 60


def test_unsure_set_is_separate_up_to_20_not_already_drawn_same_hash():
    rows = labels(n_methods=20, per=10)          # 200 pairs, a third unsure
    out = fa.flowaud(rows)
    drawn = {(r["ordinal"], r["pairIndex"]) for r in out["F1"]}
    uns = [(m["ordinal"], i) for m in rows for i, p in enumerate(m["pairs"]) if p["confidence"] == "unsure" and (m["ordinal"], i) not in drawn]
    want = sorted(uns, key=lambda t: sha(f"ccg-p1a-v1-flowaud|python|pair|{t[0]}|{t[1]}"))[:20]
    got = [(r["ordinal"], r["pairIndex"]) for r in out["F1unsure"]]
    assert got == want and len(got) == 20 and not (set(got) & drawn) and all(r["unsure"] for r in out["F1unsure"])


def test_unsure_set_is_shorter_when_fewer_remain():
    rows = labels(n_methods=7, per=10, unsure_every=1000003)   # a single unsure pair possible (index 0 of ordinal*10 multiples)
    out = fa.flowaud(rows)
    assert len(out["F1unsure"]) <= 20
    drawn = {(r["ordinal"], r["pairIndex"]) for r in out["F1"]}
    assert all((r["ordinal"], r["pairIndex"]) not in drawn for r in out["F1unsure"])


def test_f1_flag_is_the_pairs_own_unsure_flag():
    rows = labels()
    out = fa.flowaud(rows)
    for r in out["F1"]:
        assert r["unsure"] == (rows[r["ordinal"] - 1]["pairs"][r["pairIndex"]]["confidence"] == "unsure")


def test_f2_is_first_6_methods_by_method_hash_and_extend_adds_next_3():
    rows = labels(n_methods=15)
    want = sorted(range(1, 16), key=lambda o: sha(f"ccg-p1a-v1-flowaud|python|method|{o}"))
    out = fa.flowaud(rows)
    assert out["F2"] == {"methods": want[:6]}
    ext = fa.flowaud(rows, extend=True)
    assert ext["F2"] == {"methods": want[:6], "extension": want[6:9]}


def test_output_never_contains_label_content():
    rows = labels()
    text = json.dumps(fa.flowaud(rows, extend=True))
    assert SECRET not in text and "path" not in text and "qname" not in text and "evidence" not in text
    assert all(set(r) == {"ordinal", "pairIndex", "unsure"} for r in fa.flowaud(rows)["F1"])


def test_cli_output_has_no_label_content(tmp_path):
    lab = tmp_path / "labels.jsonl"
    write(lab, labels())
    r = subprocess.run([sys.executable, str(FF / "flow_p1_audit.py"), "flowaud-p1", "--labels", str(lab), "--extend", "--out", str(tmp_path / "o.json")],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and SECRET not in r.stdout + r.stderr + (tmp_path / "o.json").read_text()
    assert len(json.loads(r.stdout)["F1"]) == 60


def test_duplicate_ordinal_in_labels_aborts():
    rows = labels(3)
    rows[1]["ordinal"] = 1
    with pytest.raises(fa.AuditError):
        fa.flowaud(rows)


def test_deterministic_and_input_order_independent():
    rows = labels()
    assert fa.flowaud(rows) == fa.flowaud(rows)
    a, b = fa.flowaud(rows), fa.flowaud(list(reversed(rows)))
    assert a["F1"] == b["F1"] and a["F2"] == b["F2"]


def test_p0_flowaud_draw_is_not_imported_or_referenced():
    src = (FF / "flow_p1_audit.py").read_text()
    assert "import flowaud_draw" not in src and "from flowaud_draw" not in src and "ccg-p0a" not in src
    assert "flowaud_draw" not in sys.modules


# ---------------------------------------------------------------------------------------------------------------------- adjlist-p1
def expected_list(answers, n_conn_want_fn=None):
    miss = [(a["ordinal"], a["idx"]) for a in answers if not a["connected"] or not a["on"]["connected"]]
    conn = [(a["ordinal"], a["idx"]) for a in answers if a["connected"] and a["on"]["connected"]]
    n = len(miss)
    want = 120 - n if n <= 60 else 60
    chosen = sorted(conn, key=lambda t: sha(f"ccg-p1a-v1-adj|python|{COMMIT}|conn|{t[0]}|{t[1]}"))[:want]
    return sorted(miss + chosen, key=lambda t: sha(f"ccg-p1a-v1-adj|python|{COMMIT}|shuf|{t[0]}|{t[1]}"))


def test_list_matches_independent_hash_rule_with_misses_in_either_configuration():
    truth, answers = truth_and_answers(480, miss_off={3, 77}, miss_on={77, 200, 400})
    rows = fa.build_list(answers, truth)
    want = expected_list(answers)
    assert [(r["ordinal"], r["pairIndex"]) for r in rows] == want and len(rows) == 120
    for g in (3, 77, 200, 400):
        assert key_of(g) in want                      # every miss, whichever configuration missed it


def test_zero_misses_lists_120_connected_pairs():
    truth, answers = truth_and_answers(300)
    rows = fa.build_list(answers, truth)
    assert len(rows) == 120 and [(r["ordinal"], r["pairIndex"]) for r in rows] == expected_list(answers)


@pytest.mark.parametrize("n,length", [(59, 120), (60, 120), (61, 121), (70, 130)])
def test_boundary_n_60_61(n, length):
    truth, answers = truth_and_answers(480, miss_off=set(range(n)))
    rows = fa.build_list(answers, truth)
    assert len(rows) == length
    assert [(r["ordinal"], r["pairIndex"]) for r in rows] == expected_list(answers)
    assert sum(1 for r in rows if (r["ordinal"] - 1) * 10 + r["pairIndex"] >= n) == (120 - n if n <= 60 else 60)


def test_fewer_connected_pairs_than_wanted_lists_all_of_them():
    truth, answers = truth_and_answers(100, miss_off={1, 2})
    rows = fa.build_list(answers, truth)
    assert len(rows) == 100


def test_rows_are_whitelisted_and_status_free():
    truth, answers = truth_and_answers(200, miss_off={5}, extra={"connected": False, "status": "miss", "confidence": "sure", "adjudication": "x"})
    rows = fa.build_list(answers, truth)
    assert all(set(r) <= set(fa.ROW_KEYS) for r in rows)
    assert all(set(r) == {"ordinal", "pairIndex", "kind", "source", "sink", "evidence"} for r in rows)
    text = json.dumps(rows)
    for w in ('"connected"', '"status"', '"miss"', '"confidence"', '"on"', '"adjudication"'):
        assert w not in text


def test_status_key_nested_inside_a_pair_is_refused():
    truth, answers = truth_and_answers(200, miss_off={0})   # pair 0 is a miss, so it is always listed
    truth[0]["pairs"][0]["source"]["connected"] = False
    with pytest.raises(fa.AuditError):
        fa.build_list(answers, truth)


def test_miss_and_connected_rows_are_indistinguishable_in_shape():
    truth, answers = truth_and_answers(200, miss_off={5})
    rows = fa.build_list(answers, truth)
    assert len({tuple(sorted(r)) for r in rows}) == 1


def test_planted_missing_miss_aborts_verify():
    truth, answers = truth_and_answers(480, miss_off={3, 77})
    rows = fa.build_list(answers, truth)
    fa.verify_list(rows, answers, truth)
    bad = [r for r in rows if (r["ordinal"], r["pairIndex"]) != key_of(3)]
    with pytest.raises(fa.AuditError, match="missed pair"):
        fa.verify_list(bad, answers, truth)


def test_duplicate_wrong_length_wrong_order_and_extra_key_abort_verify():
    truth, answers = truth_and_answers(480, miss_off={3})
    rows = fa.build_list(answers, truth)
    with pytest.raises(fa.AuditError, match="duplicates"):
        fa.verify_list(rows + [rows[0]], answers, truth)
    with pytest.raises(fa.AuditError):
        fa.verify_list(rows[:-1], answers, truth)                     # a dropped connected pair: composition
    with pytest.raises(fa.AuditError):
        fa.verify_list(list(reversed(rows)), answers, truth)          # order
    with pytest.raises(fa.AuditError, match="whitelist"):
        fa.verify_list([{**rows[0], "connected": True}] + rows[1:], answers, truth)


def test_wrong_connected_selection_aborts_verify():
    truth, answers = truth_and_answers(480, miss_off={3})
    rows = fa.build_list(answers, truth)
    swap = next(g for g in range(480) if key_of(g) not in {(r["ordinal"], r["pairIndex"]) for r in rows} and g != 3)
    o, i = key_of(swap)
    rows[-1] = {**rows[-1], "ordinal": o, "pairIndex": i}
    with pytest.raises(fa.AuditError):
        fa.verify_list(rows, answers, truth)


def test_answers_truth_disagreement_and_missing_on_abort():
    truth, answers = truth_and_answers(200)
    with pytest.raises(fa.AuditError):
        fa.build_list(answers[:-1], truth)
    a2 = [dict(a) for a in answers]
    a2[0]["on"] = None
    with pytest.raises(fa.AuditError):
        fa.build_list(a2, truth)
    with pytest.raises(fa.AuditError):
        fa.build_list(answers + [answers[0]], truth)


def test_deterministic_and_input_order_independent_with_sha(tmp_path):
    truth, answers = truth_and_answers(480, miss_off={3, 9})
    a, t = tmp_path / "a.jsonl", tmp_path / "t.jsonl"
    write(a, answers)
    write(t, truth)
    cmd = [sys.executable, str(FF / "flow_p1_audit.py"), "adjlist-p1", "--answers", str(a), "--truth", str(t)]
    r1 = subprocess.run(cmd + ["--out", str(tmp_path / "l1.jsonl")], capture_output=True, text=True, timeout=30)
    write(a, list(reversed(answers)))
    r2 = subprocess.run(cmd + ["--out", str(tmp_path / "l2.jsonl")], capture_output=True, text=True, timeout=30)
    assert r1.returncode == r2.returncode == 0
    assert r1.stdout.strip() == r2.stdout.strip() == hashlib.sha256((tmp_path / "l1.jsonl").read_bytes()).hexdigest()
    assert (tmp_path / "l1.jsonl").read_bytes() == (tmp_path / "l2.jsonl").read_bytes()
    v = subprocess.run(cmd + ["--verify", str(tmp_path / "l1.jsonl")], capture_output=True, text=True, timeout=30)
    assert v.returncode == 0 and v.stdout.strip() == r1.stdout.strip()
    lines = (tmp_path / "l1.jsonl").read_text().splitlines()
    (tmp_path / "bad.jsonl").write_text("\n".join(lines[1:]) + "\n")
    bad = subprocess.run(cmd + ["--verify", str(tmp_path / "bad.jsonl")], capture_output=True, text=True, timeout=30)
    assert bad.returncode != 0


def test_adjlist_without_out_or_verify_aborts(tmp_path):
    truth, answers = truth_and_answers(200)
    write(tmp_path / "a.jsonl", answers)
    write(tmp_path / "t.jsonl", truth)
    r = subprocess.run([sys.executable, str(FF / "flow_p1_audit.py"), "adjlist-p1", "--answers", str(tmp_path / "a.jsonl"), "--truth", str(tmp_path / "t.jsonl")],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode != 0


# ------------------------------------------------------------------------------------------------- non-int keys are refused, value-free
BAD = [SECRET, True, 1.0, -1, None]


@pytest.mark.parametrize("bad", BAD, ids=repr)
def test_non_int_label_ordinal_aborts_without_echoing_the_value(bad):
    rows = labels(3)
    rows[1]["ordinal"] = bad
    with pytest.raises(fa.AuditError) as e:
        fa.flowaud(rows)
    assert SECRET not in str(e.value) and "row 2" in str(e.value) and "ordinal" in str(e.value)


@pytest.mark.parametrize("bad", BAD, ids=repr)
def test_non_int_answers_truth_and_list_keys_abort_value_free(bad):
    truth, answers = truth_and_answers(200, miss_off={0})
    rows = fa.build_list(answers, truth)
    cases = [lambda: fa.build_list([{**answers[0], "ordinal": bad}] + answers[1:], truth),
             lambda: fa.build_list([{**answers[0], "idx": bad}] + answers[1:], truth),
             lambda: fa.build_list(answers, [{**truth[0], "ordinal": bad}] + truth[1:]),
             lambda: fa.verify_list([{**rows[0], "pairIndex": bad}] + rows[1:], answers, truth)]
    for c in cases:
        with pytest.raises(fa.AuditError) as e:
            c()
        assert SECRET not in str(e.value) and "non-negative integer" in str(e.value) and "row 1" in str(e.value)


def test_cli_non_int_ordinal_aborts_with_no_echo(tmp_path):
    rows = labels(3)
    rows[0]["ordinal"] = SECRET
    write(tmp_path / "l.jsonl", rows)
    r = subprocess.run([sys.executable, str(FF / "flow_p1_audit.py"), "flowaud-p1", "--labels", str(tmp_path / "l.jsonl")], capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and SECRET not in r.stdout + r.stderr
