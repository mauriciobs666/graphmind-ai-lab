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

REAL_READER = G.eval_commit_reader                           # the autouse fixture below replaces ``G.eval_commit_reader``
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
def test_withhold_shapes_drops_every_scripts_item_the_predicate_names_and_counts_only_counting_ones():
    a, d, h = scripts(1, "doc.md"), scripts(2, "other.md"), {**scripts(4, "doc.md"), "derivedStratum": "http"}
    nc = scripts(3, "doc.md", counting=False)                          # N2: excluded by the predicate regardless of ``counting``, not in the cap's numbers
    kept, counts = G.withhold_shapes([a, d, nc, h], reader(None))
    assert [r["itemId"] for r in kept] == [d["itemId"], h["itemId"]]
    assert counts == {"scriptsCounting": 2, "withheld": 1, "byShape": {"B": 1}}


def test_a_record_with_no_counting_key_is_non_counting_in_both_the_split_and_the_scorer():
    """N2: one default. No key: not in the cap's denominator (split) and not a counting item (scorer: ``bool(rec.get("counting"))``)."""
    bare = {k: v for k, v in scripts(1, "other.md").items() if k != "counting"}
    assert G.withheld_counts([bare], reader(None)) == {"scriptsCounting": 0, "withheld": 0, "byShape": {}}
    bare_shaped = {k: v for k, v in scripts(2, "doc.md").items() if k != "counting"}
    kept, counts = G.withhold_shapes([bare_shaped], reader(None))
    assert kept == [] and counts["withheld"] == 0 and counts["scriptsCounting"] == 0


def test_a_forged_shape_field_in_the_record_neither_withholds_nor_saves_an_item():
    forged = scripts(2, "other.md", text=FENCED, shape="B", withheldShape="B", withheld=True)
    kept, counts = G.withhold_shapes([forged], reader(None))
    assert kept == [forged] and counts["withheld"] == 0


# ------------------------------------------------------------------------------------------------ run: not asked
def test_run_does_not_ask_a_withheld_shape_item_and_records_the_counts(tmp_path, monkeypatch):
    e = Env(tmp_path)
    recs = floor_list(e, 30, 30, 20)
    seen = []
    real = G.run_asks
    monkeypatch.setattr(G, "run_asks", lambda records, *a, **k: (seen.append({r["itemId"] for r in records}), real(records, *a, **k))[1])
    assert run_main_(e, monkeypatch) == R.EXIT_OK
    shaped = {r["itemId"] for r in recs if r.get("impacted", {}).get("path") == "doc.md"}
    assert len(shaped) == 1 and seen == [{r["itemId"] for r in recs} - shaped]           # the withheld item is not among the asked records
    assert e.marker()["withheldShapes"] == {"scriptsCounting": 21, "withheld": 1, "byShape": {"B": 1}}
    assert json.loads(e.run_files()[0].read_text())["withheldShapes"] == e.marker()["withheldShapes"]


# ------------------------------------------------------------------------------------------------ score: the refutation fixture
def score_report(tmp_path, monkeypatch, capsys, extra, shapes):
    """``shapes``: the withheld-shape counts the run would have recorded, written out as a literal by each test."""
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    e.set_records([e.w.rec, *extra])
    e.write_run(withheldShapes=shapes)
    e.main()
    return json.loads(capsys.readouterr().out)


def test_an_unlisted_withheld_shape_item_is_not_a_miss_and_is_reported_beside_the_scored_counts(tmp_path, monkeypatch, capsys):
    rep = score_report(tmp_path, monkeypatch, capsys, [scripts(1, "doc.md")], {"scriptsCounting": 1, "withheld": 1, "byShape": {"B": 1}})
    assert rep["misses"]["count"] == 0 and rep["scored"] == 1
    assert rep["withheldShapes"] == {"scriptsCounting": 1, "withheld": 1, "byShape": {"B": 1}}


def test_a_decoy_that_is_not_a_withheld_shape_and_is_not_listed_is_still_a_miss(tmp_path, monkeypatch, capsys):
    rep = score_report(tmp_path, monkeypatch, capsys, [scripts(1, "doc.md"), scripts(2, "other.md")], {"scriptsCounting": 2, "withheld": 1, "byShape": {"B": 1}})
    assert rep["misses"]["count"] == 1 and rep["misses"]["byStratum"] == {"scripts": 1} and rep["scored"] == 2
    assert rep["withheldShapes"]["withheld"] == 1


def test_a_twin_carrying_forged_shape_fields_is_still_a_miss(tmp_path, monkeypatch, capsys):
    rep = score_report(tmp_path, monkeypatch, capsys, [scripts(2, "other.md", text=FENCED, shape="B", withheldShape="B")],
                      {"scriptsCounting": 1, "withheld": 0, "byShape": {}})
    assert rep["misses"]["count"] == 1 and rep["withheldShapes"]["withheld"] == 0


def test_score_refuses_a_run_whose_recorded_shape_counts_the_lists_do_not_reproduce(tmp_path, monkeypatch, capsys):
    from test_gate_run import ScoreEnv
    e = ScoreEnv(tmp_path, monkeypatch)
    e.set_records([e.w.rec, scripts(1, "doc.md")])
    e.write_run(withheldShapes={"scriptsCounting": 1, "withheld": 0, "byShape": {}})
    assert e.main() == 2 and "withheld" in capsys.readouterr().err
    e.write_run(withheldShapes=None)
    assert e.main() == 2


def floor_list(e, http, mcp, plain_scripts, shaped=1):
    """A list of the given kept sizes per stratum plus ``shaped`` withheld-shape scripts items (http/mcp are copies of the world's hit)."""
    recs = [{**e.w.rec, "itemId": f"h{k:03d}".ljust(32, "0"), "derivedStratum": "http", "counting": True} for k in range(http)]
    recs += [{**e.w.rec, "itemId": f"m{k:03d}".ljust(32, "0"), "derivedStratum": "mcp", "counting": True} for k in range(mcp)]
    recs += [scripts(100 + k, "other.md") for k in range(plain_scripts)] + [scripts(200 + k, "doc.md") for k in range(shaped)]
    e.set_records(recs)
    return recs


@pytest.mark.parametrize("http,mcp,sc,why", [(30, 30, 14, "pooled 74"), (40, 40, 7, "scripts 7")])
def test_run_refuses_before_the_marker_when_the_exclusion_leaves_the_floors_unmet(tmp_path, monkeypatch, capsys, http, mcp, sc, why):
    """N4: a run that would knowingly be unable to pass is not spent: exit 5, no lock, no build, no marker; the message names counts only."""
    monkeypatch.setattr(R, "_Lock", LockDouble)
    LockDouble.instances.clear()
    e = Env(tmp_path)
    floor_list(e, http, mcp, sc)
    assert run_main_(e, monkeypatch) == R.EXIT_RESOLVE
    assert e.markers() == [] and e.refresh_kw is None and not any(l.taken for l in LockDouble.instances)
    assert "floor" in capsys.readouterr().err


@pytest.mark.parametrize("http,mcp,sc", [(30, 30, 15), (40, 40, 8)])
def test_run_with_the_exclusion_exactly_at_the_floors_goes_ahead(tmp_path, monkeypatch, http, mcp, sc):
    e = Env(tmp_path)
    floor_list(e, http, mcp, sc)
    assert run_main_(e, monkeypatch) == R.EXIT_OK


def test_the_floor_refusal_is_only_about_the_exclusion_a_small_list_with_nothing_withheld_still_runs(tmp_path, monkeypatch):
    e = Env(tmp_path)
    floor_list(e, 2, 2, 3, shaped=0)
    assert run_main_(e, monkeypatch) == R.EXIT_OK


def run_main_(e, monkeypatch):
    monkeypatch.setenv("HOME", str(e.home))
    return R.main(e.argv(), deps=e.deps())


class LockDouble:
    instances = []

    def __init__(self):
        self.taken = False
        LockDouble.instances.append(self)

    def take(self, cust, key):
        self.taken = True

    def release(self):
        pass


# ------------------------------------------------------------------------------------------------ N3: a bare CR cannot shift lines
def test_an_impacted_file_with_a_bare_cr_is_not_withheld_it_is_asked(tmp_path):
    repo = H.make_repo(tmp_path)
    (repo / "cpg/eval").mkdir(parents=True, exist_ok=True)
    (repo / "x.md").write_bytes(b"a\rb\n```\nx\n```\nreal\n")            # bare CR: the lines read differently by line-splitting; gold line 5 is plain
    (repo / "crlf.md").write_bytes(b"a\r\n```\nx\r\n```\nreal\r\n")
    H.git(repo, "add", "-A")
    H.git(repo, "commit", "-qm", "cr")
    (repo / "cpg/eval/EVAL_COMMIT").write_text(H.head(repo) + "\n")
    read = REAL_READER(repo)
    plant = scripts(1, "x.md", lines=[5])
    assert G.item_withheld_shape(plant, read) is None                              # fail closed: asked
    assert G.item_withheld_shape(scripts(4, "x.md", lines=[4]), read) is None        # fenced in the raw line view, ambiguous with a CR-aware one: asked, not withheld
    fenced_crlf = scripts(2, "crlf.md", lines=[3])
    assert G.item_withheld_shape(fenced_crlf, read) == "B"                         # CRLF files keep their line numbers
    assert G.item_withheld_shape(scripts(3, "crlf.md", lines=[5]), read) is None


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
