"""P1-34a: flow_decide.py (the pinned decision script of `ccg-p1-flow-python-v1`).  Synthetic scoring output, list and rulings; the planted
arithmetic of the pre-registration (§0.1) is asserted exactly, and the mutants (CCG_FF_DIR copies) must each fail a test."""
import io
import json
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REAL = Path(__file__).resolve().parents[1] / "spikes" / "frontend-fitness"
FF = Path(os.environ.get("CCG_FF_DIR") or REAL)
sys.path.insert(0, str(FF))
import flow_decide as fd  # noqa: E402
import s28_metrics as sm  # noqa: E402
import test_flow_p1 as P  # noqa: E402  (the p1 world and record helpers)

OKPS = {"off": (20, 20), "on": (20, 20)}


def answers(n, off_missed=(), on_missed=()):
    return [{"ordinal": 1 + i // 20, "idx": i % 20, "connected": i not in off_missed, "on": {"connected": i not in on_missed}} for i in range(n)]


def key(a):
    return (a["ordinal"], a["idx"])


def rule(rows, verdicts):
    """verdicts: {index: ruling}; every listed pair gets one ('valid' by default)."""
    return [{"ordinal": key(a)[0], "pairIndex": key(a)[1], "adjudication": verdicts.get(i, "valid")} for i, a in enumerate(rows)]


def run(rows, ds=None, an=None, ps=OKPS, listed=None):
    listed = [key(a) for a in rows] if listed is None else listed
    return fd.decide(rows, listed, rule(rows, ds or {}) if not isinstance(ds, list) else ds, rule(rows, an or {}) if not isinstance(an, list) else an, ps)


def cfg(out, c, which="decision"):
    return out[which]["perConfig"][c]


# ----------------------------------------------------------------------------------------------------------- the rule, in integers
def test_456_of_480_is_exactly_the_bar_and_M_is_zero():
    out = run(answers(480, off_missed=range(24), on_missed=range(24)))
    c = cfg(out, "off")
    assert (c["y"], c["G"], c["20y"], c["19G"], c["M"]) == (456, 480, 9120, 9120, 0) and out["outcome"] == "TRUE"
    assert out["decision"]["row"] == 5


def test_455_of_480_fails_recall_with_M_minus_20():
    out = run(answers(480, off_missed=range(25), on_missed=range(25)))
    assert cfg(out, "off")["M"] == -20 and out["outcome"] == "FALSE:measured" and out["decision"]["row"] == 4


def test_one_configuration_failing_recall_is_row_4_and_the_discordant_pairs_are_listed():
    out = run(answers(480, off_missed=range(24), on_missed=range(25)))
    assert out["outcome"] == "FALSE:measured" and cfg(out, "off")["recallOK"] and not cfg(out, "on")["recallOK"]
    assert out["decision"]["discordantPairs"] == ["2|4"]


def test_each_configuration_is_scored_by_its_own_column():
    out = run(answers(480, off_missed=range(30), on_missed=range(10)))
    assert (cfg(out, "off")["y"], cfg(out, "on")["y"]) == (450, 470)


def test_G_must_reach_120_and_exactly_120_is_enough():
    assert run(answers(120))["outcome"] == "TRUE"
    out = run(answers(119))
    assert out["outcome"] == "FALSE:insufficient-sample" and out["decision"]["row"] == 2


def test_projection_success_19_of_20_passes_and_below_is_the_precondition_row():
    assert run(answers(120), ps={"off": (19, 20), "on": (19, 20)})["outcome"] == "TRUE"
    out = run(answers(120), ps={"off": (19, 20), "on": (18, 20)})
    assert out["outcome"] == "FALSE:precondition" and cfg(out, "on")["projectionSuccess"] == {"k": 18, "n": 20, "20k": 360, "19n": 380}


def test_row_precedence_insufficient_before_precondition_before_recall():
    rows = answers(100, off_missed=range(50))
    assert run(rows, ps={"off": (1, 20), "on": (1, 20)})["outcome"] == "FALSE:insufficient-sample"
    rows = answers(200, off_missed=range(50))
    assert run(rows, ps={"off": (1, 20), "on": (1, 20)})["outcome"] == "FALSE:precondition"


# ----------------------------------------------------------------------------------------------------------- D, the doubly-ruled set
def test_a_miss_ruled_not_real_by_both_leaves_G_only_so_G_drops_one_and_M_rises_19():
    rows = answers(480, off_missed=[0], on_missed=[0])
    base = run(rows)
    out = run(rows, ds={0: "overturned"}, an={0: "overturned"})
    for c in ("off", "on"):
        assert (cfg(out, c)["G"], cfg(out, c)["y"]) == (cfg(base, c)["G"] - 1, cfg(base, c)["y"])
        assert cfg(out, c)["M"] == cfg(base, c)["M"] + 19
    assert out["D"] == 1 and base["D"] == 0


def test_a_connected_pair_ruled_not_real_by_both_leaves_y_and_G_so_M_falls_one():
    # 20(y-1) - 19(G-1) = M - 1: a set holding an invalid connected pair overstates the true margin by one
    rows = answers(480, off_missed=[0], on_missed=[0])
    base = run(rows)
    out = run(rows, ds={5: "overturned"}, an={5: "overturned"})
    for c in ("off", "on"):
        assert (cfg(out, c)["G"], cfg(out, c)["y"]) == (cfg(base, c)["G"] - 1, cfg(base, c)["y"] - 1)
        assert cfg(out, c)["M"] == cfg(base, c)["M"] - 1


@pytest.mark.parametrize("who", ["ds", "an"])
def test_a_pair_ruled_not_real_by_one_adjudicator_only_leaves_nothing(who):
    rows = answers(480, off_missed=[0, 1], on_missed=[0, 1])
    one = run(rows, **{who: {0: "overturned", 7: "overturned"}})
    none = run(rows)
    assert one["D"] == 0 and one["decision"] == none["decision"]


def test_the_two_adjudicators_must_overturn_the_same_pair():
    rows = answers(480, off_missed=[0, 1], on_missed=[0, 1])
    out = run(rows, ds={0: "overturned"}, an={1: "overturned"})
    assert out["D"] == 0


def test_the_D_empty_verdict_is_printed_as_a_diagnostic_and_is_not_the_outcome():
    rows = answers(480, off_missed=range(25), on_missed=range(25))          # 455: FALSE with D empty
    d = {i: "overturned" for i in range(3)}                                  # three doubly-ruled misses: G 477, M = -20 + 57 = 37
    out = run(rows, ds=d, an=d)
    assert out["outcome"] == "TRUE" and cfg(out, "off")["M"] == 37
    assert out["diagnosticDEmpty"]["verdict"] == "FALSE:measured" and cfg(out, "off", "diagnosticDEmpty")["M"] == -20


def test_other_rulings_do_not_remove_a_pair():
    rows = answers(480, off_missed=[0], on_missed=[0])
    out = run(rows, ds={0: "endpoint-naming-error"}, an={0: "endpoint-naming-error"})
    assert out["D"] == 0 and cfg(out, "off")["G"] == 480


# ----------------------------------------------------------------------------------------------------------- list and rulings validation
def test_every_missed_pair_in_either_configuration_must_be_listed():
    rows = answers(480, off_missed=[0], on_missed=[1])
    with pytest.raises(SystemExit, match="not listed"):
        run(rows, listed=[key(a) for a in rows[2:]], ds=rule(rows[2:], {}), an=rule(rows[2:], {}))


@pytest.mark.parametrize("miss_in", ["off", "on"])
def test_a_miss_in_one_configuration_only_must_be_listed_and_isolates_exactly_that_pair(miss_in):
    rows = answers(480, off_missed=[3] if miss_in == "off" else [], on_missed=[3] if miss_in == "on" else [])
    rest = [a for a in rows if key(a) != key(rows[3])]
    with pytest.raises(SystemExit, match=r"1 missed pair\(s\) are not listed"):
        run(rows, listed=[key(a) for a in rest], ds=rule(rest, {}), an=rule(rest, {}))
    out = run(rows)                                                                   # listed: accepted, counted once
    assert out["misses"] == 1


def test_a_listed_pair_that_is_not_in_the_answers_and_a_duplicate_listing_are_refused():
    rows = answers(480, off_missed=[0], on_missed=[0])
    ghost = [key(a) for a in rows] + [(999, 999)]
    with pytest.raises(SystemExit, match="not in the answers"):
        fd.decide(rows, ghost, rule(rows, {}) + [{"ordinal": 999, "pairIndex": 999, "adjudication": "valid"}], rule(rows, {}) + [{"ordinal": 999, "pairIndex": 999, "adjudication": "valid"}], OKPS)
    with pytest.raises(SystemExit, match="duplicates"):
        fd.decide(rows, [key(a) for a in rows] + [key(rows[0])], rule(rows, {}), rule(rows, {}), OKPS)


def test_the_miss_and_connected_predicates_agree_with_the_adjudication_list_builder():
    import flow_p1_audit as fa
    # every combination: connected/missed in off only, on only, both, neither
    rows = [{"ordinal": 1, "idx": i, "connected": c_off, "on": {"connected": c_on}}
            for i, (c_off, c_on) in enumerate([(True, True), (False, True), (True, False), (False, False), (True, True)])]
    audit_misses, audit_conn = fa._classify(rows)
    assert sorted(audit_misses) == [(1, 1), (1, 2), (1, 3)] and sorted(audit_conn) == [(1, 0), (1, 4)]
    for k in [key(a) for a in rows]:                                                   # leaving out ONE pair from the list: refused iff it is a miss
        rest = [a for a in rows if key(a) != k]
        try:
            fd.decide(rows, [key(a) for a in rest], rule(rest, {}), rule(rest, {}), OKPS)
            refused = False
        except SystemExit:
            refused = True
        assert refused == (k in audit_misses), k
    assert fd.decide(rows, audit_misses, rule([a for a in rows if key(a) in audit_misses], {}), rule([a for a in rows if key(a) in audit_misses], {}), OKPS)["misses"] == len(audit_misses)


def test_the_docstring_states_the_connected_pair_margin_as_minus_one():
    assert "y-1, G-1, M-1" in fd.__doc__ and not __import__("re").search(r"M\+1\b", fd.__doc__) and "G-1, M+19" in fd.__doc__


def test_each_rulings_file_must_rule_exactly_the_listed_pairs_once():
    rows = answers(480, off_missed=[0], on_missed=[0])
    good = rule(rows, {})
    with pytest.raises(SystemExit, match="rulings-ds"):
        run(rows, ds=good[:-1], an=good)
    with pytest.raises(SystemExit, match="rulings-analyst"):
        run(rows, ds=good, an=good + [good[0]])
    with pytest.raises(SystemExit, match="unknown adjudication"):
        run(rows, ds=rule(rows, {0: "banana"}), an=good)


def test_a_list_of_misses_plus_a_sample_of_connected_pairs_is_accepted_and_unlisted_pairs_stay():
    rows = answers(480, off_missed=[0, 1], on_missed=[0, 1])
    lst = [key(rows[i]) for i in (0, 1, 2, 3)]
    sub = [rows[i] for i in (0, 1, 2, 3)]
    d = [{"ordinal": key(a)[0], "pairIndex": key(a)[1], "adjudication": "overturned" if i == 0 else "miss-stands"} for i, a in enumerate(sub)]
    out = fd.decide(rows, lst, d, d, OKPS)
    assert out["D"] == 1 and out["listed"] == 4 and cfg(out, "off")["G"] == 479


def test_duplicate_answer_keys_and_missing_on_column_are_refused():
    rows = answers(10)
    with pytest.raises(SystemExit):
        fd.decide(rows + [rows[0]], [key(a) for a in rows], rule(rows, {}), rule(rows, {}), OKPS)
    bad = [dict(rows[0], on=None)] + rows[1:]
    with pytest.raises(SystemExit, match="on"):
        fd.decide(bad, [key(a) for a in bad], rule(rows, {}), rule(rows, {}), OKPS)


# ----------------------------------------------------------------------------------------------------------- the CLI on a planted p1 world
def cli(tmp_path, monkeypatch, results=None):
    ff = P.p1_world(tmp_path, monkeypatch)
    rec = P.write_record(tmp_path, ff)
    assert sm.main(["python", "--flow-set", "p1", "--record", rec]) == 0
    assert sm.main(["python", "--flow-set", "p1", "--record", rec, "--join", "p0"]) == 0
    d = ff / "results" / "p1" / "fixed-new"
    rows = [json.loads(l) for l in open(d / "projection-python-p1.answers.jsonl")]
    lst = tmp_path / "list.jsonl"
    lst.write_text("".join(json.dumps({"ordinal": r["ordinal"], "pairIndex": r["idx"]}) + "\n" for r in rows))
    ru = tmp_path / "ru.jsonl"
    ru.write_text("".join(json.dumps({"ordinal": r["ordinal"], "pairIndex": r["idx"], "adjudication": "miss-stands"}) + "\n" for r in rows))
    argv = ["--record", rec, "--results", str(results or d / "projection-python-p1.json"), "--answers", str(d / "projection-python-p1.answers.jsonl"),
            "--truth", str(ff / "truth" / "python.flow-p1-final.jsonl"), "--list", str(lst), "--rulings-ds", str(ru), "--rulings-analyst", str(ru)]
    return ff, argv


def test_the_cli_prints_the_verdict_for_the_decision_cell_and_writes_nothing(tmp_path, monkeypatch):
    ff, argv = cli(tmp_path, monkeypatch)
    before = P.tree(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert fd.main(argv) == 0
    out = json.loads(buf.getvalue())
    assert out["outcome"] == "FALSE:insufficient-sample" and out["decision"]["perConfig"]["off"]["G"] == 14      # the planted world has 14 golden pairs
    assert P.tree(tmp_path) == before


def test_the_cli_refuses_a_diagnostic_cell_as_the_decision_input(tmp_path, monkeypatch):
    ff, argv = cli(tmp_path, monkeypatch, results=None)
    argv[argv.index("--results") + 1] = str(ff / "results" / "p1" / "fixed-p0" / "projection-python-p1.json")
    with pytest.raises(SystemExit, match="not the decision cell"):
        fd.main(argv)


def test_the_cli_refuses_when_a_pinned_script_changed_since_the_record(tmp_path, monkeypatch):
    ff, argv = cli(tmp_path, monkeypatch)
    (ff / "flow_decide.py").write_text((ff / "flow_decide.py").read_text() + "\n# edited\n")
    with pytest.raises(SystemExit, match="NOT-MEASURED:void"):
        fd.main(argv)


def _fail(tmp_path, monkeypatch, edit, match):
    ff, argv = cli(tmp_path, monkeypatch)
    edit(ff, argv)
    with pytest.raises(SystemExit, match=match):
        fd.main(argv)


def test_the_cli_refuses_a_truth_file_with_an_extra_pair(tmp_path, monkeypatch):
    def edit(ff, argv):
        t = ff / "truth" / "python.flow-p1-final.jsonl"
        rows = [json.loads(l) for l in t.read_text().splitlines()]
        rows[0]["pairs"].append(dict(rows[0]["pairs"][0]))
        t.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _fail(tmp_path, monkeypatch, edit, "disagree on the number of golden pairs")


def test_the_cli_refuses_results_whose_pooled_n_differs_from_the_answers(tmp_path, monkeypatch):
    def edit(ff, argv):
        r = Path(argv[argv.index("--results") + 1])
        d = json.loads(r.read_text())
        d["cfg"]["on"]["pooled"]["n"] += 1
        r.write_text(json.dumps(d))
    _fail(tmp_path, monkeypatch, edit, "disagree on the number of golden pairs")


def test_the_cli_refuses_answers_whose_connected_count_differs_from_the_results(tmp_path, monkeypatch):
    def edit(ff, argv):
        a = Path(argv[argv.index("--answers") + 1])
        rows = [json.loads(l) for l in a.read_text().splitlines()]
        rows[0]["connected"] = not rows[0]["connected"]
        a.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _fail(tmp_path, monkeypatch, edit, "disagree on the connected pairs")


def test_the_cli_refuses_answers_whose_on_connected_count_differs_from_the_results(tmp_path, monkeypatch):
    def edit(ff, argv):
        a = Path(argv[argv.index("--answers") + 1])
        rows = [json.loads(l) for l in a.read_text().splitlines()]
        rows[0]["on"]["connected"] = not rows[0]["on"]["connected"]
        a.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _fail(tmp_path, monkeypatch, edit, "disagree on the connected pairs")


def test_the_cli_refuses_when_the_cpg_changed_since_the_record_and_accepts_an_explicit_run_record(tmp_path, monkeypatch):
    ff, argv = cli(tmp_path, monkeypatch)
    rr = tmp_path / "work" / "python" / "s28-run.json"
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert fd.main(argv + ["--run-record", str(rr)]) == 0
    (tmp_path / "work" / "python" / "off.cpg.bin").write_bytes(b"REBUILT")
    with pytest.raises(SystemExit, match="NOT-MEASURED:void.*cpgs.off"):
        fd.main(argv)                                                                 # the default run record
    with pytest.raises(SystemExit, match="cpgs.off"):
        fd.main(argv + ["--run-record", str(rr)])


def test_the_cli_resolves_the_record_repo_relative_from_cpg(tmp_path, monkeypatch):
    ff, argv = cli(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    (repo / "cpg" / "eval").mkdir(parents=True)
    (repo / "cpg" / "eval" / "prereg.jsonl").write_text(Path(argv[1]).read_text())
    monkeypatch.setattr(sm, "REPO", repo)
    monkeypatch.chdir(repo / "cpg")
    argv[1] = "cpg/eval/prereg.jsonl"
    with redirect_stdout(io.StringIO()):
        assert fd.main(argv) == 0


# ----------------------------------------------------------------------------------------------------------- mutants
MUTANTS = {
    "recall_gt_for_ge": [('"recallOK": 20 * y >= 19 * g', '"recallOK": 20 * y > 19 * g')],
    "recall_20y_19g_swapped": [('"recallOK": 20 * y >= 19 * g', '"recallOK": 19 * y >= 20 * g')],
    "pairs_threshold_gt": [('"pairsOK": g >= MIN_PAIRS', '"pairsOK": g > MIN_PAIRS')],
    "projection_gt_for_ge": [('n > 0 and 20 * k >= 19 * n', 'n > 0 and 20 * k > 19 * n')],
    "D_is_union": [('d = _rulings(rulings_ds, lst, "rulings-ds") & _rulings(', 'd = _rulings(rulings_ds, lst, "rulings-ds") | _rulings(')],
    "second_rulings_file_dropped": [(' & _rulings(rulings_analyst, lst, "rulings-analyst")', "")],
    "rulings_ignored": [('return {k for k, v in ruled.items() if v == "overturned"}', "return set()")],
    "connected_pair_in_D_keeps_y": [('"off": _cfg(g, sum(1 for a in keep if a["connected"]), *ps["off"])', '"off": _cfg(g, sum(1 for a in answers if a["connected"]), *ps["off"])')],
    "off_scored_by_on_column": [('"off": _cfg(g, sum(1 for a in keep if a["connected"]), *ps["off"])', '"off": _cfg(g, sum(1 for a in keep if a["on"]["connected"]), *ps["off"])')],
    "unlisted_misses_allowed": [("    if misses - lst:", "    if False:")],
    "rulings_need_not_match_list": [("    if set(ruled) != listed:", "    if False:")],
    "misses_defined_by_off_only": [('misses = {(a["ordinal"], a["idx"]) for a in answers if not a["connected"] or not a["on"]["connected"]}', 'misses = {(a["ordinal"], a["idx"]) for a in answers if not a["connected"]}')],
    "misses_defined_by_on_only": [('misses = {(a["ordinal"], a["idx"]) for a in answers if not a["connected"] or not a["on"]["connected"]}', 'misses = {(a["ordinal"], a["idx"]) for a in answers if not a["on"]["connected"]}')],
    "misses_defined_by_both_columns": [('if not a["connected"] or not a["on"]["connected"]}', 'if not a["connected"] and not a["on"]["connected"]}')],
    "list_may_name_pairs_outside_the_answers": [("    if len(lst) != len(listed) or not lst <= set(keys):", "    if len(lst) != len(listed):")],
    "list_duplicates_allowed": [("    if len(lst) != len(listed) or not lst <= set(keys):", "    if not lst <= set(keys):")],
    "truth_pair_count_not_checked": [("    if truth_pairs != len(answers) or any(", "    if any(")],
    "results_pooled_n_not_checked": [(' or any(res["cfg"][c]["pooled"]["n"] != len(answers) for c in ("off", "on")):', ":")],
    "connected_counts_not_checked": [('    if res["cfg"]["off"]["pooled"]["k"] != sum(', "    if False and res[\"cfg\"][\"off\"][\"pooled\"][\"k\"] != sum(")],
    "on_connected_count_not_checked": [(' or res["cfg"]["on"]["pooled"]["k"] != sum(1 for x in answers if x["on"]["connected"]):', ":")],
    "run_record_not_checked_for_cpg": [("a.lang, commit, sm.flow_paths(a.lang, \"p1\", sm.DECISION_CELL)[\"sample\"], runs)", "a.lang, commit, sm.flow_paths(a.lang, \"p1\", sm.DECISION_CELL)[\"sample\"], {})")],
    "diagnostic_uses_D": [('"diagnosticDEmpty": _verdict(answers, set(), ps)', '"diagnosticDEmpty": _verdict(answers, d, ps)')],
    "decision_cell_not_checked": [('or cell.get("name") != sm.DECISION_CELL or cell.get("role") != "decision"', "")],
    "row_order_precondition_first": [("    if any(not c[\"pairsOK\"] for c in v):\n        row, verdict = 2, \"FALSE:insufficient-sample\"\n    elif any(not c[\"preconditionOK\"] for c in v):\n        row, verdict = 3, \"FALSE:precondition\"",
                                      "    if any(not c[\"preconditionOK\"] for c in v):\n        row, verdict = 3, \"FALSE:precondition\"\n    elif any(not c[\"pairsOK\"] for c in v):\n        row, verdict = 2, \"FALSE:insufficient-sample\"")],
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name, tmp_path):
    if os.environ.get("CCG_FF_DIR"):
        pytest.skip("inner run")
    copy = tmp_path / "src"
    copy.mkdir()
    for p in REAL.glob("*.py"):
        shutil.copy(p, copy / p.name)
    for n in ("corpora.yaml", "flow-brief.template.md"):
        shutil.copy(REAL / n, copy / n)
    text = (copy / "flow_decide.py").read_text(encoding="utf-8")
    for old, new in MUTANTS[name]:
        assert old in text, f"mutant {name}: pattern not found, the mutant would be a no-op"
        text = text.replace(old, new)
    (copy / "flow_decide.py").write_text(text, encoding="utf-8")
    env = dict(os.environ, CCG_FF_DIR=str(copy), CENSUS_SRC=str(copy), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "not mutant", str(Path(__file__))],
                       capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[1])
    assert p.returncode != 0 and "failed" in p.stdout, f"mutant {name} survived:\n{p.stdout[-600:]}"
