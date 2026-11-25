"""Q8 / scoped AC-5 wording ("every counting item outside the named withheld shapes"): a held-out scripts item of a withheld shape is NEVER asked or scored
(ruling on ``docs/reviews/connected-code-graph-p1-closeout-prereg-ml.md`` B1 and ``-prereg.md`` M2); the verdict, floors and bounds are over the SCORED set and the
report carries the withheld counts beside the scored counts. The refutation fixture: a held-out-shaped list with an unlisted withheld-shape item (must not
fail the run), a decoy that is not a withheld shape and is not listed either (must still be a miss), and a twin carrying forged shape fields. Offline."""
import json

import pytest

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, OBS as H_OBS
import gate_run as R

FENCED = "```\nplain\n```\n"                              # line 2 is inside a fence: shape B
PLAIN = "plain\nplain\n"


def reader(repo):
    return lambda rel: FENCED if rel == "doc.md" else PLAIN


def scripts(n, path, **extra):
    """A scripts item whose anchor the engine's answers do not list (a miss if it is scored)."""
    rec = H.item(n, H.contract("http", f"GET /nowhere{n}"), {"type": "symbol", "repoId": "r", "qname": f"{path}::f", "path": path}, lines=(2,), stratum="scripts")
    rec.update(extra)
    return rec


@pytest.fixture(autouse=True)
def _reader(monkeypatch):
    monkeypatch.setattr(G, "eval_commit_reader", reader)


# ------------------------------------------------------------------------------------------------ the split
def test_withhold_shapes_drops_only_counting_scripts_items_the_predicate_names_and_counts_them():
    a, d, nc, h = scripts(1, "doc.md"), scripts(2, "other.md"), scripts(3, "doc.md", counting=False), {**scripts(4, "doc.md"), "derivedStratum": "http"}
    kept, counts = G.withhold_shapes([a, d, nc, h], reader(None))
    assert [r["itemId"] for r in kept] == [d["itemId"], nc["itemId"], h["itemId"]]
    assert counts == {"scriptsCounting": 2, "withheld": 1, "byShape": {"B": 1}}


def test_a_forged_shape_field_in_the_record_neither_withholds_nor_saves_an_item():
    forged = scripts(2, "other.md", text=FENCED, shape="B", withheldShape="B", withheld=True)
    kept, counts = G.withhold_shapes([forged], reader(None))
    assert kept == [forged] and counts["withheld"] == 0


# ------------------------------------------------------------------------------------------------ run: not asked
def test_run_does_not_ask_a_withheld_shape_item_and_records_the_counts(tmp_path, monkeypatch):
    e = Env(tmp_path)
    recs = [scripts(1, "doc.md"), scripts(2, "other.md"), scripts(3, "other.md"), scripts(4, "other.md")]
    e.set_records(recs)
    seen = []
    real = G.run_asks
    monkeypatch.setattr(G, "run_asks", lambda records, *a, **k: (seen.append([r["itemId"] for r in records]), real(records, *a, **k))[1])
    monkeypatch.setenv("HOME", str(e.home))
    assert R.main(e.argv(), deps=e.deps()) == R.EXIT_OK
    assert seen == [[r["itemId"] for r in recs[1:]]]                                    # the withheld item is not among the asked records
    assert e.marker()["withheldShapes"] == {"scriptsCounting": 4, "withheld": 1, "byShape": {"B": 1}}
    run = json.loads(e.run_files()[0].read_text())
    assert run["withheldShapes"] == e.marker()["withheldShapes"]


# ------------------------------------------------------------------------------------------------ score: the refutation fixture
def score_report(tmp_path, monkeypatch, capsys, extra):
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    e.set_records([e.w.rec, *extra])
    e.main()
    return json.loads(capsys.readouterr().out)


def test_an_unlisted_withheld_shape_item_is_not_a_miss_and_is_reported_beside_the_scored_counts(tmp_path, monkeypatch, capsys):
    rep = score_report(tmp_path, monkeypatch, capsys, [scripts(1, "doc.md")])
    assert rep["misses"]["count"] == 0 and rep["scored"] == 1
    assert rep["withheldShapes"] == {"scriptsCounting": 1, "withheld": 1, "byShape": {"B": 1}}


def test_a_decoy_that_is_not_a_withheld_shape_and_is_not_listed_is_still_a_miss(tmp_path, monkeypatch, capsys):
    rep = score_report(tmp_path, monkeypatch, capsys, [scripts(1, "doc.md"), scripts(2, "other.md")])
    assert rep["misses"]["count"] == 1 and rep["misses"]["byStratum"] == {"scripts": 1} and rep["scored"] == 2
    assert rep["withheldShapes"]["withheld"] == 1


def test_a_twin_carrying_forged_shape_fields_is_still_a_miss(tmp_path, monkeypatch, capsys):
    rep = score_report(tmp_path, monkeypatch, capsys, [scripts(2, "other.md", text=FENCED, shape="B", withheldShape="B")])
    assert rep["misses"]["count"] == 1 and rep["withheldShapes"]["withheld"] == 0


def test_score_refuses_a_run_whose_recorded_shape_counts_the_lists_do_not_reproduce(tmp_path, monkeypatch, capsys):
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    e.set_records([e.w.rec, scripts(1, "doc.md")])
    e.write_run(withheldShapes={"scriptsCounting": 1, "withheld": 0, "byShape": {}})
    assert e.main() == 2 and "withheld" in capsys.readouterr().err
    e.write_run(withheldShapes=None)
    assert e.main() == 2


def test_a_stratum_pushed_under_its_floor_by_the_exclusion_is_not_a_pass(tmp_path, monkeypatch, capsys):
    """What the harness does: nothing special. The floors are over the scored set, so a run that all-hits but has fewer than 8 scored scripts items reports
    ``minimumsMet`` false and a verdict that is not ``pass``; the run itself was spent."""
    rep = score_report(tmp_path, monkeypatch, capsys, [scripts(1, "doc.md")])
    assert rep["verdict"] != "pass" and rep["minimumsMet"] is not True


# ------------------------------------------------------------------------------------------------ the freeze rule: a cpg/tests edit after the record
def test_a_cpg_tests_edit_committed_after_the_record_is_refused_by_run_context_and_an_eval_edit_is_not(tmp_path):
    """Reported, not changed: only ``cpg/eval/**`` and ``*.md`` may change after the record's commit; a test file under ``cpg/tests/`` is outside both."""
    e = Env(tmp_path)
    pf = G.preflight(e.repo, "ac5-gate-1", observed=H_OBS)
    (e.repo / "cpg/eval/note.json").write_text('{"after": 1}\n')
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "eval edit")
    G.run_context(e.repo, pf, allow_after_adaptation=True)
    (e.repo / "cpg/tests/test_x.py").write_text("X = 2\n")
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "tests edit")
    with pytest.raises(G.Refusal, match="outside cpg/eval/"):
        G.run_context(e.repo, pf, allow_after_adaptation=True)
